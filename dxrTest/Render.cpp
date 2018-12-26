#include "Render.h"

#include <fstream>

#include "SceneData.h"
#include "D3D12RaytracingHelpers.hpp"
#include "raytracing2.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

const uint32_t vertexBufferSize = boxVerticesSize + planeVerticesSize + icosahedronVerticesSize + coneVerticesSize;
const uint32_t indexBufferSize = boxIndicesSize + planeIndicesSize + icosahedronIndicesSize + coneIndicesSize;

const uint32_t boxVerticesStart = 0;
const uint32_t boxIndicesStart = 0;

const uint32_t planeVerticesStart = boxVerticesStart + boxVerticesCount;
const uint32_t planeIndicesStart = boxIndicesStart + boxIndicesCount;

const uint32_t icosahedronVerticesStart = planeVerticesStart + planeVerticesCount;
const uint32_t icosahedronIndicesStart = planeIndicesStart + planeIndicesCount;

const uint32_t coneVerticesStart = icosahedronVerticesStart + icosahedronVerticesCount;
const uint32_t coneIndicesStart = icosahedronIndicesStart + icosahedronIndicesCount;

const wchar_t* hitGroupName = L"MyHitGroup";
const wchar_t* raygenShaderName = L"MyRaygenShader";
const wchar_t* closestHitShaderName = L"MyClosestHitShader";
const wchar_t* missShaderName = L"MyMissShader";

// Hit groups.
const wchar_t* hitGroupNames2[] = {
    L"triangleHitGroup", L"shadowrayTriangleHitGroup"
};
const wchar_t* raygenShaderName2 = L"rayGen";
const wchar_t* closestHitShaderName2 = L"closestHitTriangle";
const wchar_t* missShaderNames2[] = {
  L"missRadiance",
  L"missShadow"
};

inline uint32_t align(const uint32_t &size, const uint32_t &alignment) {
  return (size + (alignment - 1)) & ~(alignment - 1);
}

template <typename T>
constexpr uint32_t sizeAliquot32bits(const T &obj) {
  return (sizeof(obj) - 1) / sizeof(uint32_t) + 1;
}

bool enableComputeRaytracingFallback(IDXGIAdapter1* adapter) {
  ID3D12Device* testDevice = nullptr;
  UUID experimentalFeatures[] = {D3D12ExperimentalShaderModels};

  return SUCCEEDED(D3D12EnableExperimentalFeatures(1, experimentalFeatures, nullptr, nullptr))
      && SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&testDevice)));
}

void allocateUAVBuffer(ID3D12Device* pDevice, const uint64_t &bufferSize, ID3D12Resource **ppResource, D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_COMMON, const wchar_t* resourceName = nullptr) {
  HRESULT hr;

  const auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  const auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  hr = pDevice->CreateCommittedResource(
    &uploadHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    initialResourceState,
    nullptr,
    IID_PPV_ARGS(ppResource)
  );
  throwIfFailed(hr, "allocateUAVBuffer failed");

  if (resourceName != nullptr) {
    (*ppResource)->SetName(resourceName);
  }
}

void allocateUploadBuffer(ID3D12Device* pDevice, void *pData, const uint64_t &datasize, ID3D12Resource **ppResource, const wchar_t* resourceName = nullptr) {
  HRESULT hr;
  auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
  auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(datasize);

  hr = pDevice->CreateCommittedResource(
    &uploadHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(ppResource));
  throwIfFailed(hr, "allocateUploadBuffer failed");

  if (resourceName != nullptr) {
    (*ppResource)->SetName(resourceName);
  }

  void *pMappedData;
  (*ppResource)->Map(0, nullptr, &pMappedData);
  memcpy(pMappedData, pData, datasize);
  (*ppResource)->Unmap(0, nullptr);
}

ID3D12Resource* allocateAndMap(ID3D12Device* device, const uint32_t &bufferSize, void** mappingPtr = nullptr, const wchar_t* resourceName = nullptr) {
  HRESULT hr;
  auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

  auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
  ID3D12Resource* res = nullptr;
  hr = device->CreateCommittedResource(
    &uploadHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&res)
  );
  throwIfFailed(hr, "Cannot create upload resource");

  if (resourceName != nullptr) res->SetName(resourceName);

  if (mappingPtr != nullptr) {
    // We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
    hr = res->Map(0, nullptr, mappingPtr);
    throwIfFailed(hr, "Cannot map upload resource");
  }

  return res;
}

uint32_t allocateDescriptor(ID3D12DescriptorHeap* heap, uint32_t &descriptorsAllocated, const uint32_t &descriptorSize, D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, const uint32_t &descriptorIndexToUse) {
  uint32_t index = descriptorIndexToUse;

  auto descriptorHeapCpuBase = heap->GetCPUDescriptorHandleForHeapStart();
  if (descriptorIndexToUse >= heap->GetDesc().NumDescriptors) {
    index = descriptorsAllocated++;
  }

  *cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, index, descriptorSize);
  return index;
}

bool getHardwareAdapter(IDXGIFactory4 *factory, IDXGIAdapter1 **adapter) {
  HRESULT hr;
  IDXGIAdapter1 *tmpAdapter = nullptr;  // adapters are the graphics card (this includes the embedded
                                        // graphics on the motherboard)

  uint32_t adapterIndex = 0;  // we'll start looking for directx 12  compatible
                              // graphics devices starting at index 0

  bool adapterFound = false;  // set this to true when a good one was found

  // find first hardware gpu that supports d3d 12
  while (factory->EnumAdapters1(adapterIndex, &tmpAdapter) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc;
    tmpAdapter->GetDesc1(&desc);

    // we want a device that is compatible with direct3d 12 (feature level 11 or
    // higher)
    hr = D3D12CreateDevice(tmpAdapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr);
    if (SUCCEEDED(hr)) {
      adapterFound = true;
      break;
    }

    adapterIndex++;
  }

  *adapter = tmpAdapter;
  return true;
}

DX12Render::DX12Render() {
  fallback.forceComputeFallback = false;
  fallback.raytracingOutputResourceUAVDescriptorHeapIndex = UINT32_MAX;
  rayGenCB.viewport = {-1.0f, -1.0f, 1.0f, 1.0f};
  planeMaterialCB.albedo = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
  planeMaterialCB.reflectanceCoef = 0.25f;
  planeMaterialCB.diffuseCoef = 1.0f;
  planeMaterialCB.specularCoef = 0.4f;
  planeMaterialCB.specularPower = 50.0f;
  planeMaterialCB.stepScale = 1.0f;

  oldViewProj = glm::mat4(1.0f);

  fallback.colorBufferHeapIndex = UINT32_MAX;
  fallback.depthBufferHeapIndex = UINT32_MAX;
  fallback.normalBufferHeapIndex = UINT32_MAX;

  filter.filterOutputUAVDescIndex = UINT32_MAX;

  filter.colorBufferHeapIndex = UINT32_MAX;
  filter.depthBufferHeapIndex = UINT32_MAX;
  filter.lastFrameColorHeapIndex = UINT32_MAX;
  filter.lastFrameDepthHeapIndex = UINT32_MAX;
}

DX12Render::~DX12Render() {
  cleanup();
}

void DX12Render::init(HWND hwnd, const uint32_t &width, const uint32_t &height, bool fullscreen) {
  HRESULT hr;

  UINT dxgiFactoryFlags = 0;
  const bool useWarpDevice = false;

#ifdef _DEBUG
  ID3D12Debug *debugController = nullptr;
  hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
  throwIfFailed(hr, "Failed to get debug interface");
  
  debugController->EnableDebugLayer();

  // Enable additional debug layers.
  dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

  // -- Create the Device -- //

  IDXGIFactory4 *factory = nullptr;
  hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory));
  throwIfFailed(hr, "Failed to create IDXGIFactory");

  if (useWarpDevice) {
    IDXGIAdapter *warpAdapter = nullptr;
    hr = factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
    throwIfFailed(hr, "Failed to enum warp adapter");

    hr = D3D12CreateDevice(warpAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    throwIfFailed(hr, "Failed to create DX12 device");
  } else {
    IDXGIAdapter1 *hardwareAdapter = nullptr;
    const bool res = getHardwareAdapter(factory, &hardwareAdapter);
    throwIfFailed(res ? 0 : -1, "Could not find apropriate hardware");

    const bool fallbackSupported = enableComputeRaytracingFallback(hardwareAdapter);
    throwIf(!fallbackSupported, "Ray tracing fallback layer is not supported");

    hr = D3D12CreateDevice(hardwareAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    throwIfFailed(hr, "Failed to create DX12 device");
  }

  // -- Create the Command Queue -- //

  D3D12_COMMAND_QUEUE_DESC cqDesc = {};  // we will be using all the default values

  hr = device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&commandQueue));  // create the command queue
  throwIfFailed(hr, "Failed to create CommandQueue");

  // -- Create the Swap Chain (double/tripple buffering) -- //

  const DXGI_MODE_DESC backBufferDesc{
    // this is to describe our display mode
    width,                       // buffer width
    height,                      // buffer height
    {0, 0},                      // default value
    DXGI_FORMAT_R8G8B8A8_UNORM,  // format of the buffer (rgba 32 bits, 8 bits for each chanel)
    DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,  // default value
    DXGI_MODE_SCALING_UNSPECIFIED          // default value
  };

  // describe our multi-sampling. We are not multi-sampling, so we set the count
  // to 1 (we need at least one sample of course)
  const DXGI_SAMPLE_DESC sampleDesc{
      1,  // multisample count (no multisampling, so we just put 1, since we still need 1 sample)
      0   // default value
  };

  // Describe and create the swap chain.
  DXGI_SWAP_CHAIN_DESC swapChainDesc{
      backBufferDesc,                   // our back buffer description
      sampleDesc,                       // our multi-sampling description
      DXGI_USAGE_RENDER_TARGET_OUTPUT,  // this says the pipeline will render to this swap chain
      frameBufferCount,                 // number of buffers we have
      hwnd,                             // handle to our window
      !fullscreen,  // set to true, then if in fullscreen must call
                    // SetFullScreenState with true for full screen to get
                    // uncapped fps
      DXGI_SWAP_EFFECT_FLIP_DISCARD,  // dxgi will discard the buffer (data) after we call present
      0                               // default value
  };

  IDXGISwapChain *tempSwapChain;

  factory->CreateSwapChain(
    commandQueue,    // the queue will be flushed once the swap chain is created
    &swapChainDesc,  // give it the swap chain description we created above
    &tempSwapChain   // store the created swap chain in a temp IDXGISwapChain interface
  );

  swapChain = static_cast<IDXGISwapChain3 *>(tempSwapChain);

  frameIndex = swapChain->GetCurrentBackBufferIndex();

  // -- Create the Back Buffers (render target views) Descriptor Heap -- //

  const D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{
      D3D12_DESCRIPTOR_HEAP_TYPE_RTV,   // we are creating a RTV heap
      frameBufferCount,                 // number of descriptors for this heap.
      D3D12_DESCRIPTOR_HEAP_FLAG_NONE,  // this heap is a render target view heap
      0  // the number of descriptors we will store in this descriptor heap
  };

  hr = device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvDescriptorHeap));
  throwIfFailed(hr, "Failed to create descriptor heap for swapchain");

  // get the size of a descriptor in this heap (this is a rtv heap, so only rtv
  // descriptors should be stored in it. descriptor sizes may vary from device
  // to device, which is why there is no set size and we must ask the device to
  // give us the size. we will use this size to increment a descriptor handle
  // offset
  rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

  // get a handle to the first descriptor in the descriptor heap. a handle is
  // basically a pointer, but we cannot literally use it like a c++ pointer.
  CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

  // Create a RTV for each buffer (double buffering is two buffers, tripple
  // buffering is 3).
  for (int i = 0; i < frameBufferCount; i++) {
    // first we get the n'th buffer in the swap chain and store it in the n'th
    // position of our ID3D12Resource array
    hr = swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i]));
    throwIfFailed(hr, "Failed to get swapchain resource");

    // the we "create" a render target view which binds the swap chain buffer
    // (ID3D12Resource[n]) to the rtv handle
    device->CreateRenderTargetView(renderTargets[i], nullptr, rtvHandle);

    // we increment the rtv handle by the rtv descriptor size we got above
    rtvHandle.Offset(1, rtvDescriptorSize);
  }

  // -- Create the Command Allocators -- //

  for (int i = 0; i < frameBufferCount; i++) {
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator[i]));
    throwIfFailed(hr, "Failed to create CommandAllocator");
  }

  // create the command list with the first allocator
  hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator[frameIndex], NULL, IID_PPV_ARGS(&commandList));
  throwIfFailed(hr, "Failed to create GraphicsCommandList");

  // command lists are created in the recording state. our main loop will set it
  // up for recording again so close it now
  commandList->Close();

  // -- Create a Fence & Fence Event -- //

  // create the fences
  for (int i = 0; i < frameBufferCount; i++) {
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence[i]));
    throwIfFailed(hr, "Failed to create Fence");
 
    fenceValue[i] = 0;  // set the initial fence value to 0
  }

  contextFenceValue = 1;

  // create a handle to a fence event
  fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  throwIfFailed(fenceEvent == nullptr ? -1 : 0, "Failed to create FenceEvent");

  // gBuffer creation

  const D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

  const D3D12_CLEAR_VALUE colorOptimizedClearValue = {
    DXGI_FORMAT_R8G8B8A8_UNORM,
    {0.0f, 0.0f, 0.0f, 1.0f}
  };

  hr = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // a default heap
    D3D12_HEAP_FLAG_NONE, // no flags
    &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1, 1, 0, flags, D3D12_TEXTURE_LAYOUT_UNKNOWN), // resource description for a buffer
    D3D12_RESOURCE_STATE_RENDER_TARGET,
    &colorOptimizedClearValue,
    IID_PPV_ARGS(&gBuffer.color));
  throwIfFailed(hr, "Could not create GBuffer color");

  const D3D12_CLEAR_VALUE normalOptimizedClearValue = {
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    {0.0f, 0.0f, 0.0f, 0.0f}
  };

  hr = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // a default heap
    D3D12_HEAP_FLAG_NONE, // no flags
    &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, 1, 1, 1, 0, flags, D3D12_TEXTURE_LAYOUT_UNKNOWN), // resource description for a buffer
    D3D12_RESOURCE_STATE_RENDER_TARGET,
    &normalOptimizedClearValue,
    IID_PPV_ARGS(&gBuffer.normal));
  throwIfFailed(hr, "Could not create GBuffer normal");

  // одного достаточно??
  const D3D12_DESCRIPTOR_HEAP_DESC gBufferRTVHeapDesc{
      D3D12_DESCRIPTOR_HEAP_TYPE_RTV,   // we are creating a RTV heap
      2,                                // number of descriptors for this heap.
      D3D12_DESCRIPTOR_HEAP_FLAG_NONE,  // this heap is a render target view heap
      0  // the number of descriptors we will store in this descriptor heap
  };

  hr = device->CreateDescriptorHeap(&gBufferRTVHeapDesc, IID_PPV_ARGS(&gBuffer.cDescriptorHeap));
  throwIfFailed(hr, "Could not create GBuffer color descriptor heap");

  const uint32_t gBufferRTVSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

  CD3DX12_CPU_DESCRIPTOR_HANDLE gBufHandle(gBuffer.cDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

  // тут у нас должны быть уже созданные ресурсы
  device->CreateRenderTargetView(gBuffer.color, nullptr, gBufHandle);
  gBufHandle.Offset(1, gBufferRTVSize);
  device->CreateRenderTargetView(gBuffer.normal, nullptr, gBufHandle);

  // create a depth stencil descriptor heap so we can get a pointer to the depth stencil buffer
  const D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    1,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    0
  };

  hr = device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&gBuffer.dDescriptorHeap));
  throwIfFailed(hr, "Could not create GBuffer depth descriptor heap");

  D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
  depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
  depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

  const D3D12_CLEAR_VALUE depthOptimizedClearValue = {
    DXGI_FORMAT_D32_FLOAT,
    {1.0f, 0}
  };

  //D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | 
  const D3D12_RESOURCE_FLAGS depthFlags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 1, 1, 0, depthFlags),
    D3D12_RESOURCE_STATE_DEPTH_WRITE,
    &depthOptimizedClearValue,
    IID_PPV_ARGS(&gBuffer.depth)
  );
  throwIfFailed(hr, "Could not create GBuffer depth");

  gBuffer.dDescriptorHeap->SetName(L"GBuffer depth");

  device->CreateDepthStencilView(gBuffer.depth, &depthStencilDesc, gBuffer.dDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

  // buffer creation

  std::vector<uint32_t> indices;
  std::vector<Vertex> vertices;
  loadModels(indices, vertices);

  const uint32_t boxVStart = planeMaterialCB.box.verticesStart;
  const uint32_t boxVSize = planeMaterialCB.box.verticesSize;
  const uint32_t planeVStart = planeMaterialCB.plane.verticesStart;
  const uint32_t planeVSize = planeMaterialCB.plane.verticesSize;
  const uint32_t icosphereVStart = planeMaterialCB.icosphere.verticesStart;
  const uint32_t icosphereVSize = planeMaterialCB.icosphere.verticesSize;
  const uint32_t coneVStart = planeMaterialCB.cone.verticesStart;
  const uint32_t coneVSize = planeMaterialCB.cone.verticesSize;
  const size_t vertexSize = boxVSize + planeVSize + icosphereVSize + coneVSize;

  const uint32_t boxIStart = planeMaterialCB.box.indicesStart;
  const uint32_t boxISize = planeMaterialCB.box.indicesSize;
  const uint32_t planeIStart = planeMaterialCB.plane.indicesStart;
  const uint32_t planeISize = planeMaterialCB.plane.indicesSize;
  const uint32_t icosphereIStart = planeMaterialCB.icosphere.indicesStart;
  const uint32_t icosphereISize = planeMaterialCB.icosphere.indicesSize;
  const uint32_t coneIStart = planeMaterialCB.cone.indicesStart;
  const uint32_t coneISize = planeMaterialCB.cone.indicesSize;
  const size_t indexSize = boxISize + planeISize + icosphereISize + coneISize;

  //const size_t boxDataSize = sizeof(boxVertices);

  // create default heap
  // default heap is memory on the GPU. Only the GPU has access to this memory
  // To get data into this heap, we will have to upload the data using
  // an upload heap
  hr = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // a default heap
    D3D12_HEAP_FLAG_NONE, // no flags
    //&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize), // resource description for a buffer
    &CD3DX12_RESOURCE_DESC::Buffer(vertexSize), // resource description for a buffer
    D3D12_RESOURCE_STATE_COPY_DEST, // we will start this heap in the copy destination state since we will copy data
                                    // from the upload heap to this heap
    nullptr, // optimized clear value must be null for this type of resource. used for render targets and depth/stencil buffers
    IID_PPV_ARGS(&boxVertexBuffer));
  throwIfFailed(hr, "Could not create vertex buffer");

  // we can give resource heaps a name so when we debug with the graphics debugger we know what resource we are looking at
  boxVertexBuffer->SetName(L"Vertex Buffer Resource Heap");

  // create upload heap
  // upload heaps are used to upload data to the GPU. CPU can write to it, GPU can read from it
  // We will upload the vertex buffer using this heap to the default heap
  ID3D12Resource* vBufferUploadHeap;
  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), // upload heap
    D3D12_HEAP_FLAG_NONE, // no flags
    &CD3DX12_RESOURCE_DESC::Buffer(vertexSize), // resource description for a buffer
    D3D12_RESOURCE_STATE_GENERIC_READ, // GPU will read from this buffer and copy its contents to the default heap
    nullptr,
    IID_PPV_ARGS(&vBufferUploadHeap));
  throwIfFailed(hr, "Could not create vertex upload buffer");

  vBufferUploadHeap->SetName(L"Vertex Buffer Upload Resource Heap");

  //uint32_t iBufferSize = sizeof(boxIndices);
  //const size_t iBufferSize = 

  // create default heap to hold index buffer
  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // a default heap
    D3D12_HEAP_FLAG_NONE, // no flags
    //&CD3DX12_RESOURCE_DESC::Buffer(indexBufferSize), // resource description for a buffer
    &CD3DX12_RESOURCE_DESC::Buffer(indexSize), // resource description for a buffer
    D3D12_RESOURCE_STATE_COPY_DEST, // start in the copy destination state
    nullptr, // optimized clear value must be null for this type of resource
    IID_PPV_ARGS(&boxIndexBuffer));
  throwIfFailed(hr, "Could not create index buffer");

  // we can give resource heaps a name so when we debug with the graphics debugger we know what resource we are looking at
  boxIndexBuffer->SetName(L"Index Buffer Resource Heap");

  // create upload heap to upload index buffer
  ID3D12Resource* iBufferUploadHeap;
  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), // upload heap
    D3D12_HEAP_FLAG_NONE, // no flags
    &CD3DX12_RESOURCE_DESC::Buffer(indexSize), // resource description for a buffer
    D3D12_RESOURCE_STATE_GENERIC_READ, // GPU will read from this buffer and copy its contents to the default heap
    nullptr,
    IID_PPV_ARGS(&iBufferUploadHeap));
  throwIfFailed(hr, "Could not create index upload buffer");

  iBufferUploadHeap->SetName(L"Index Buffer Upload Resource Heap");

  /*Vertex* vData = nullptr;
  vBufferUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&vData));
  memcpy(&vData[boxVerticesStart], boxVertices, boxVerticesSize);
  memcpy(&vData[planeVerticesStart], planeVertices, planeVerticesSize);
  memcpy(&vData[icosahedronVerticesStart], icosahedronVertices, icosahedronVerticesSize);
  memcpy(&vData[coneVerticesStart], coneVertices, coneVerticesSize);
  vBufferUploadHeap->Unmap(0, nullptr);*/

  //throwIf(true, "indices size " + std::to_string(indices.size()) + " vertices size " + std::to_string(vertices.size()));

  Vertex* vData = nullptr;
  vBufferUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&vData));
  throwIf(boxVStart >= vertices.size(), "boxVStart >= vertices size" + std::to_string(vertices.size()));
  memcpy(&vData[boxVStart], &vertices[boxVStart], boxVSize);

  throwIf(planeVStart >= vertices.size(), "planeVStart >= vertices size" + std::to_string(vertices.size()));
  memcpy(&vData[planeVStart], &vertices[planeVStart], planeVSize);

  throwIf(icosphereVStart >= vertices.size(), "icosphereVStart >= vertices size" + std::to_string(vertices.size()));
  memcpy(&vData[icosphereVStart], &vertices[icosphereVStart], icosphereVSize);

  throwIf(coneVStart >= vertices.size(), "coneVStart >= vertices size" + std::to_string(vertices.size()));
  memcpy(&vData[coneVStart], &vertices[coneVStart], coneVSize);
  vBufferUploadHeap->Unmap(0, nullptr);

  /*uint32_t* iData = nullptr;
  iBufferUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&iData));
  memcpy(&iData[boxIndicesStart], boxIndices, boxIndicesSize);
  memcpy(&iData[planeIndicesStart], planeIndices, planeIndicesSize);
  memcpy(&iData[icosahedronIndicesStart], icosahedronIndices, icosahedronIndicesSize);
  memcpy(&iData[coneIndicesStart], coneIndices, coneIndicesSize);
  iBufferUploadHeap->Unmap(0, nullptr);*/

  uint32_t* iData = nullptr;
  iBufferUploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&iData));

  throwIf(boxIStart >= indices.size(), "boxIStart >= indices size" + std::to_string(indices.size()));
  memcpy(&iData[boxIStart], &indices[boxIStart], boxISize);

  throwIf(planeIStart >= indices.size(), "planeIStart >= indices size" + std::to_string(indices.size()));
  memcpy(&iData[planeIStart], &indices[planeIStart], planeISize);

  throwIf(icosphereIStart >= indices.size(), "icosphereIStart >= indices size" + std::to_string(indices.size()));
  memcpy(&iData[icosphereIStart], &indices[icosphereIStart], icosphereISize);

  throwIf(coneIStart >= indices.size(), "coneIStart >= indices size" + std::to_string(indices.size()));
  memcpy(&iData[coneIStart], &indices[coneIStart], coneISize);
  iBufferUploadHeap->Unmap(0, nullptr);

  // we are now creating a command with the command list to copy the data from the upload heap to the default heap
  commandList->Reset(commandAllocator[frameIndex], nullptr);
  //UpdateSubresources(commandList, boxVertexBuffer, vBufferUploadHeap, 0, 0, _countof(vertexData), vertexData);
  //UpdateSubresources(commandList, boxVertexBuffer, vBufferUploadHeap, boxVerticesSize, 0, _countof(vertexData2), vertexData2);

  commandList->CopyBufferRegion(boxVertexBuffer, 0, vBufferUploadHeap, 0, vertexSize);

  // transition the vertex buffer data from copy destination state to vertex buffer state
  commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(boxVertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

  // we are now creating a command with the command list to copy the data from the upload heap to the default heap
  /*UpdateSubresources(commandList, boxIndexBuffer, iBufferUploadHeap, 0, 0, _countof(indexData), indexData);
  UpdateSubresources(commandList, boxIndexBuffer, iBufferUploadHeap, boxIndicesSize, 0, _countof(indexData2), indexData2);*/

  commandList->CopyBufferRegion(boxIndexBuffer, 0, iBufferUploadHeap, 0, indexSize);

  // transition the vertex buffer data from copy destination state to vertex buffer state
  commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(boxIndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

  // Now we execute the command list to upload the initial assets (triangle data)
  commandList->Close();
  ID3D12CommandList* ppCommandLists[] = {commandList};
  commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // increment the fence value now, otherwise the buffer might not be uploaded by the time we start drawing
  /*++fenceValue[frameIndex];
  hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
  if (FAILED(hr)) {
    MessageBox(0, L"Could not copy data to gpu buffer", L"Error", MB_OK);
    return false;
  }*/

  waitForRenderContext();

  const size_t constantBufferSize = (sizeof(glm::mat4) + 255) & ~255; // CB size is required to be 256-byte aligned

  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize),
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&constantBuffer));

  throwIfFailed(hr, "Could not create constant buffer");

  // we can give resource heaps a name so when we debug with the graphics debugger we know what resource we are looking at
  constantBuffer->SetName(L"view proj matrix buffer");

  constantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&constantBufferPtr));

  /*if (constantBufferPtr == nullptr) {
    throw std::runtime_error("Constant buffer null ptr");
  }*/

  const D3D12_DESCRIPTOR_HEAP_DESC constantBufferHeapDesc{
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,    // we are creating a CBV heap
    1,                                         // number of descriptors for this heap.
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    0  // the number of descriptors we will store in this descriptor heap
  };

  hr = device->CreateDescriptorHeap(&constantBufferHeapDesc, IID_PPV_ARGS(&constantBufferDescriptor));
  throwIfFailed(hr, "Could not create descriptor heap for constant buffer");

  const D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {
    constantBuffer->GetGPUVirtualAddress(),
    constantBufferSize
  };

  device->CreateConstantBufferView(&cbvDesc, constantBufferDescriptor->GetCPUDescriptorHandleForHeapStart());

  instanceBufferSize = 10 * sizeof(glm::mat4);

  hr = device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer(instanceBufferSize),
    D3D12_RESOURCE_STATE_GENERIC_READ, // ???
    nullptr,
    IID_PPV_ARGS(&instanceBuffer));

  throwIfFailed(hr, "Failed to create instance buffer");

  instanceBuffer->SetName(L"Instance Buffer Resource Heap");

  instanceBufferView.BufferLocation = instanceBuffer->GetGPUVirtualAddress();
  instanceBufferView.StrideInBytes = sizeof(glm::mat4);
  instanceBufferView.SizeInBytes = instanceBufferSize;

  instanceBuffer->Map(0, nullptr, reinterpret_cast<void**>(&instanceBufferPtr));

  //ZeroMemory(&constantBufferPtr, sizeof(constantBufferPtr));

  // create a vertex buffer view for the triangle. We get the GPU memory address to the vertex pointer using the GetGPUVirtualAddress() method
  boxVertexBufferView.BufferLocation = boxVertexBuffer->GetGPUVirtualAddress();
  boxVertexBufferView.StrideInBytes = sizeof(Vertex);
  boxVertexBufferView.SizeInBytes = vertexSize;

  // create a vertex buffer view for the triangle. We get the GPU memory address to the vertex pointer using the GetGPUVirtualAddress() method
  boxIndexBufferView.BufferLocation = boxIndexBuffer->GetGPUVirtualAddress();
  boxIndexBufferView.Format = DXGI_FORMAT_R32_UINT; // 32-bit unsigned integer (this is what a dword is, double word, a word is 2 bytes)
  boxIndexBufferView.SizeInBytes = indexSize;

  // Fill out the Viewport
  viewport.TopLeftX = 0;
  viewport.TopLeftY = 0;
  viewport.Width = float(width);
  viewport.Height = float(height);
  viewport.MinDepth = 0.0f;
  viewport.MaxDepth = 1.0f;

  // Fill out a scissor rect
  scissorRect.left = 0;
  scissorRect.top = 0;
  scissorRect.right = width;
  scissorRect.bottom = height;

  SAFE_RELEASE(vBufferUploadHeap)
  SAFE_RELEASE(iBufferUploadHeap)

  //createRTResources(width, height);
  //createFilterResources(width, height);
}

void DX12Render::initRT(const uint32_t &width, const uint32_t &height, const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer) {
  // Initialize raytracing pipeline.

  // Create raytracing interfaces: raytracing device and commandlist.
  createRayTracingFallbackDevice();

  // Create root signatures for the shaders.
  createRootSignatures();

  // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
  createRaytracingPSO();

  // Create a heap for descriptors.
  createRTDescriptorHeap();

  // здесь мы копируем геометрию на устройство, не особо полезная штука
  //buildGeometry();
  createDescriptors();
  initializeScene();

  // Build raytracing acceleration structures from the generated geometry.
  // здесь мне нужно создать top и bottom level структуры, я пока еще не очень понимаю как это выглядит
  // как эти структуры соответствуют ресурсам шейдеров? 
  //buildAccelerationStructures();
  buildAccelerationStructures2(boxBuffer, icosahedronBuffer, coneBuffer);

  // Build shader tables, which define shaders and their local root arguments.
  buildShaderTables();

  // Create an output 2D texture to store the raytracing result to.
  createRaytracingOutputResource(width, height);
}

void DX12Render::initFilter(const uint32_t &width, const uint32_t &height) {
  createFilterResources(width, height);
}

void DX12Render::recreatePSO() {
  SAFE_RELEASE(pipelineStateObject)

  HRESULT hr;

  // create root signature

  // create a descriptor range (descriptor table) and fill it out
  // this is a range of descriptors inside a descriptor heap
  const D3D12_DESCRIPTOR_RANGE descriptorTableRanges[] = {
    { // only one range right now
      D3D12_DESCRIPTOR_RANGE_TYPE_CBV,     // this is a range of constant buffer views (descriptors)
      1,                                   // we only have one constant buffer, so the range is only 1
      0,                                   // start index of the shader registers in the range
      0,                                   // space 0. can usually be zero
      D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND // this appends the range to the end of the root signature descriptor tables
    }
  };

  // create a descriptor table
  const D3D12_ROOT_DESCRIPTOR_TABLE descriptorTable{
    _countof(descriptorTableRanges), // we only have one range
    descriptorTableRanges            // the pointer to the beginning of our ranges array
  };

  // create a root parameter and fill it out
  const D3D12_ROOT_PARAMETER rootParameters[] = {
    { // only one parameter right now
      D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, // this is a descriptor table
      descriptorTable,                            // this is our descriptor table for this root parameter
      D3D12_SHADER_VISIBILITY_VERTEX              // our pixel shader will be the only shader accessing this parameter for now
    }
  };

  CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
  rootSignatureDesc.Init(
    _countof(rootParameters), 
    rootParameters, 
    0, 
    nullptr, 
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
  );

  ID3DBlob* signature;
  hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
  throwIfFailed(hr, "Failed to serialize root signature");

  if (rootSignature == nullptr) {
    hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
    throwIfFailed(hr, "Failed to create root signature");
  }

  // create vertex and pixel shaders

  // when debugging, we can compile the shader files at runtime.
  // but for release versions, we can compile the hlsl shaders
  // with fxc.exe to create .cso files, which contain the shader
  // bytecode. We can load the .cso files at runtime to get the
  // shader bytecode, which of course is faster than compiling
  // them at runtime

#if defined(_DEBUG)
      // Enable better shader debugging with the graphics debugging tools.
  UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compileFlags = 0;
#endif

  // compile vertex shader
  ID3DBlob* vertexShader; // d3d blob for holding vertex shader bytecode
  ID3DBlob* errorBuff; // a buffer holding the error data if any
  hr = D3DCompileFromFile(L"default.hlsl",
    nullptr,
    nullptr,
    "vertexMain",
    "vs_5_0",
    compileFlags,
    0,
    &vertexShader,
    &errorBuff);
  if (FAILED(hr)) {
    OutputDebugStringA((char*)errorBuff->GetBufferPointer());
    throw std::runtime_error("Vertex shader creation error");
  }

  // fill out a shader bytecode structure, which is basically just a pointer
  // to the shader bytecode and the size of the shader bytecode
  const D3D12_SHADER_BYTECODE vertexShaderBytecode = {
    vertexShader->GetBufferPointer(),
    vertexShader->GetBufferSize()
  };

  // compile pixel shader
  ID3DBlob* pixelShader;
  hr = D3DCompileFromFile(L"default.hlsl",
    nullptr,
    nullptr,
    "pixelMain",
    "ps_5_0",
    compileFlags,
    0,
    &pixelShader,
    &errorBuff);
  if (FAILED(hr)) {
    OutputDebugStringA((char*)errorBuff->GetBufferPointer());
    throw std::runtime_error("Pixel shader creation error");
  }

  // fill out shader bytecode structure for pixel shader
  const D3D12_SHADER_BYTECODE pixelShaderBytecode = {
    pixelShader->GetBufferPointer(),
    pixelShader->GetBufferSize()
  };

  // create input layout

  // The input layout is used by the Input Assembler so that it knows
  // how to read the vertex data bound to it.

  const D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
    {
      "POSITION", // semantic name (name of the parameter, the input assembler will associate this attribute to an input with the same semantic name in the shaders)
      0,          // semantic index (this is only needed if more than one element have the same semantic name)
      DXGI_FORMAT_R32G32B32A32_FLOAT, // this will define the format this attribute is in (4 floats)
      0,          // input slot (each vertex buffer is bound to a slot)
      offsetof(Vertex, pos), // this is the offset in bytes from the beginning of the vertex structure to the start of this attribute
      D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, // this specifies if this element is per vertex or per instance
      0           // this is the number of instances to draw before going to the next element
    },
    { "NORMAL", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    { "TEX_COORDS", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(Vertex, texCoords), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },

    // не следует забывать об этой еденичке на конце!!!
    { "MODEL_MATRIX", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0*sizeof(glm::vec4), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    { "MODEL_MATRIX", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 1*sizeof(glm::vec4), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    { "MODEL_MATRIX", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 2*sizeof(glm::vec4), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    { "MODEL_MATRIX", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 3*sizeof(glm::vec4), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 }
  };

  // fill out an input layout description structure
  const D3D12_INPUT_LAYOUT_DESC inputLayoutDesc = {
    inputLayout,
    // we can get the number of elements in an array by "sizeof(array) / sizeof(arrayElementType)"
    sizeof(inputLayout) / sizeof(D3D12_INPUT_ELEMENT_DESC)
  };

  const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp{ // a stencil operation structure, again does not really matter since stencil testing is turned off
    D3D12_STENCIL_OP_KEEP,       // what the stencil operation should do when a pixel fragment fails the stencil test
    D3D12_STENCIL_OP_KEEP,       // what the stencil operation should do when the stencil test passes but the depth test fails
    D3D12_STENCIL_OP_KEEP,       // what the stencil operation should do when stencil and depth tests both pass
    D3D12_COMPARISON_FUNC_ALWAYS // the function the stencil test should use
  };

  const D3D12_DEPTH_STENCIL_DESC depthDesc{
    true,                             // enable depth testing
    D3D12_DEPTH_WRITE_MASK_ALL,       // can write depth data to all of the depth/stencil buffer
    D3D12_COMPARISON_FUNC_LESS,       // pixel fragment passes depth test if destination pixel's depth is less than pixel fragment's
    false,                            // disable stencil test
    D3D12_DEFAULT_STENCIL_READ_MASK,  // a default stencil read mask (doesn't matter at this point since stencil testing is turned off)
    D3D12_DEFAULT_STENCIL_WRITE_MASK, // a default stencil write mask (also doesn't matter)
    defaultStencilOp,                 // both front and back facing polygons get the same treatment 
    defaultStencilOp
  };

  // create a pipeline state object (PSO)

  // In a real application, you will have many pso's. for each different shader
  // or different combinations of shaders, different blend states or different rasterizer states,
  // different topology types (point, line, triangle, patch), or a different number
  // of render targets you will need a pso

  // VS is the only required shader for a pso. You might be wondering when a case would be where
  // you only set the VS. It's possible that you have a pso that only outputs data with the stream
  // output, and not on a render target, which means you would not need anything after the stream
  // output.

  const DXGI_SAMPLE_DESC sampleDesc{
    1,  // multisample count (no multisampling, so we just put 1, since we still need 1 sample)
    0   // default value
  };

  const D3D12_RASTERIZER_DESC rasterizer{
    D3D12_FILL_MODE_SOLID, // D3D12_FILL_MODE_WIREFRAME
    D3D12_CULL_MODE_BACK,  // CullMode 
    true,                  // FrontCounterClockwise
    0,                     // DepthBias
    0.0f,                  // DepthBiasClamp
    0.0f,                  // SlopeScaledDepthBias
    false,                 // DepthClipEnable
    false,                 // MultisampleEnable
    false,                 // AntialiasedLineEnable
    0,                     // ForcedSampleCount
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF // ConservativeRaster
  };

  D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{
    rootSignature,          // the root signature that describes the input data this pso needs
    vertexShaderBytecode,   // structure describing where to find the vertex shader bytecode and how large it is
    pixelShaderBytecode,    // same as VS but for pixel shader
    {},                     // D3D12_SHADER_BYTECODE (domain shader)
    {},                     // D3D12_SHADER_BYTECODE (hull shader)
    {},                     // D3D12_SHADER_BYTECODE (geometry shader)
    {},                     // Used to send data from the pipeline (after geometry shader, or after vertex shader if geometry shader is not defined) to your app
    CD3DX12_BLEND_DESC(D3D12_DEFAULT), // a default rasterizer state
    0xffffffff,             // sample mask has to do with multi-sampling. 0xffffffff means point sampling is done
    //CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT), // a default rasterizer state
    rasterizer,
    depthDesc,              // This is the state of the depth/stencil buffer
    inputLayoutDesc,        // the structure describing our input layout
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED, // this is used when a triangle strip topology is defined
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, // type of topology we are drawing
    2,                     // render target count
    {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32B32A32_FLOAT}, // format of the render target (8?)
    DXGI_FORMAT_D32_FLOAT, // explaining the format of each depth/stencil buffer
    sampleDesc,            // must be the same sample description as the swapchain and depth/stencil buffer
    0,                     // a bit mask saying which GPU adapter to use
    {},                    // you can cache PSO's, such as into files, so the next time your initialize the PSO, compilation will happen much much faster
    D3D12_PIPELINE_STATE_FLAG_NONE // the debug option will give extra information that is helpful when debugging (now off)
  };

  // create the pso
  hr = device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineStateObject));
  throwIfFailed(hr, "Failed to create graphics pipeline");
}

void DX12Render::prepareRender(const uint32_t &instanceCount, const glm::mat4 &viewMatrix) {
  HRESULT hr;

  //SAFE_RELEASE(instanceBuffer)

  ASSERT((instanceCount + 1) * sizeof(glm::mat4) < UINT32_MAX)

  const uint32_t realInstanceCount = instanceCount + 1;
  const uint32_t size = realInstanceCount * sizeof(glm::mat4);

  //ASSERT(realInstanceCount == 2)

  //hr = device->CreateCommittedResource(
  //  &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
  //  D3D12_HEAP_FLAG_NONE,
  //  &CD3DX12_RESOURCE_DESC::Buffer(size),
  //  D3D12_RESOURCE_STATE_COPY_DEST, // ???
  //  nullptr,
  //  IID_PPV_ARGS(&instanceBuffer));

  //if (FAILED(hr)) {
  //  MessageBox(0, L"Failed to create instance buffer", L"Error", MB_OK);
  //  return false;
  //}

  //instanceBuffer->SetName(L"Instance Buffer Resource Heap");

  //instanceBufferView.BufferLocation = instanceBuffer->GetGPUVirtualAddress();
  //instanceBufferView.StrideInBytes = sizeof(glm::mat4);
  //instanceBufferView.SizeInBytes = size;

  //ID3D12Resource* uploadHeap;
  //device->CreateCommittedResource(
  //  &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
  //  D3D12_HEAP_FLAG_NONE,
  //  &CD3DX12_RESOURCE_DESC::Buffer(sizeof(glm::mat4)),
  //  D3D12_RESOURCE_STATE_GENERIC_READ,
  //  nullptr,
  //  IID_PPV_ARGS(&uploadHeap));
  //uploadHeap->SetName(L"Instance Buffer Upload Resource Heap");

  //const glm::mat4 matrix = glm::mat4(1.0f);

  //// store vertex buffer in upload heap
  //D3D12_SUBRESOURCE_DATA matrixData{
  //  reinterpret_cast<const BYTE*>(&matrix), // pointer to our vertex array
  //  sizeof(glm::mat4),                    // size of all our triangle vertex data
  //  sizeof(glm::mat4)                     // also the size of our triangle vertex data
  //};

  //waitForCurrentFrame();

  //// we are now creating a command with the command list to copy the data from the upload heap to the default heap
  //hr = commandAllocator[frameIndex]->Reset();
  //hr = commandList->Reset(commandAllocator[frameIndex], nullptr);
  //UpdateSubresources(commandList, instanceBuffer, uploadHeap, 0, 0, 1, &matrixData);

  //// transition the vertex buffer data from copy destination state to vertex buffer state
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(instanceBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

  //// Now we execute the command list to upload the initial assets (triangle data)
  //commandList->Close();
  //ID3D12CommandList* ppCommandLists[] = {commandList};
  //commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  //// increment the fence value now, otherwise the buffer might not be uploaded by the time we start drawing
  ///*++fenceValue[frameIndex];
  //hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
  //if (FAILED(hr)) {
  //  MessageBox(0, L"Could not copy data to gpu buffer", L"Error", MB_OK);
  //  return false;
  //}*/

  //waitForRenderContext();

  //SAFE_RELEASE(uploadHeap)

  if (instanceBufferSize < size) {
    instanceBuffer->Unmap(0, nullptr);
    SAFE_RELEASE(instanceBuffer)

    instanceBufferSize = size;

    hr = device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
      D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(instanceBufferSize),
      D3D12_RESOURCE_STATE_GENERIC_READ, // ???
      nullptr,
      IID_PPV_ARGS(&instanceBuffer));

    throwIfFailed(hr, "Failed to create instance buffer");

    instanceBuffer->SetName(L"Instance Buffer Resource Heap");

    instanceBufferView.BufferLocation = instanceBuffer->GetGPUVirtualAddress();
    instanceBufferView.StrideInBytes = sizeof(glm::mat4);
    instanceBufferView.SizeInBytes = instanceBufferSize;

    hr = instanceBuffer->Map(0, nullptr, reinterpret_cast<void**>(&instanceBufferPtr));
    throwIfFailed(hr, "Failed to map instance buffer");
  }

  //const glm::mat4 matrix = glm::mat4(1.0f);
  //const glm::mat4 matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f));
  const glm::mat4 matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f));

  memcpy(&instanceBufferPtr[0], &matrix, sizeof(glm::mat4));
  /*for (uint32_t i = 0; i < realInstanceCount; ++i) {
    memcpy(&instanceBufferPtr[i], &matrix, sizeof(glm::mat4));
  }*/

  memcpy(constantBufferPtr, &viewMatrix, sizeof(glm::mat4));
}

void DX12Render::computePartHost(GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer) {
  uint32_t index = 1; // плоскость - первая

  throwIf(boxBuffer.size() == 0, "box buffer size == 0");
  throwIf(icosahedronBuffer.size() == 0, "icosahedron buffer size == 0");
  throwIf(coneBuffer.size() == 0, "cone buffer size == 0");

  for (uint32_t i = 0; i < boxBuffer.size(); ++i) {
    // может еденичную матрицу?
    boxBuffer[i].currentOrn = glm::translate(glm::mat4(1.0f), glm::vec3(boxBuffer[i].pos));

    // ориентацию пока делать не буду

    boxBuffer[i].currentOrn = glm::scale(boxBuffer[i].currentOrn, glm::vec3(boxBuffer[i].scale));

    instanceBufferPtr[index] = boxBuffer[i].currentOrn;
    ++index;
  }

  for (uint32_t i = 0; i < icosahedronBuffer.size(); ++i) {
    // может еденичную матрицу?
    icosahedronBuffer[i].currentOrn = glm::translate(glm::mat4(1.0f), glm::vec3(icosahedronBuffer[i].pos));

    // ориентацию пока делать не буду
    icosahedronBuffer[i].currentOrn = glm::scale(icosahedronBuffer[i].currentOrn, glm::vec3(icosahedronBuffer[i].scale));

    instanceBufferPtr[index] = icosahedronBuffer[i].currentOrn;
    ++index;
  }

  for (uint32_t i = 0; i < coneBuffer.size(); ++i) {
    // может еденичную матрицу?
    coneBuffer[i].currentOrn = glm::translate(glm::mat4(1.0f), glm::vec3(coneBuffer[i].pos));

    // ориентацию пока делать не буду
    coneBuffer[i].currentOrn = glm::scale(coneBuffer[i].currentOrn, glm::vec3(coneBuffer[i].scale));

    instanceBufferPtr[index] = coneBuffer[i].currentOrn;
    ++index;
  }
}

void DX12Render::updateSceneData(const glm::vec4 &cameraPos, const glm::mat4 &viewProj) {
  const glm::mat4 invViewProj = glm::inverse(viewProj);

  sceneConstantBufferPtr->projectionToWorld = invViewProj;
  sceneConstantBufferPtr->cameraPosition = cameraPos;

  filterConstantDataPtr->projToPrevProj = oldViewProj * invViewProj;
  oldViewProj = viewProj;

  sceneConstantBufferPtr->elapsedTime = float(rand())/float(RAND_MAX);
}

void DX12Render::nextFrame() {
  HRESULT hr;

  // We have to wait for the gpu to finish with the command allocator before we reset it
  //if (!waitForFrame()) {
  //  return false;
  //}

  hr = commandAllocator[frameIndex]->Reset();
  throwIfFailed(hr, "Command allocator resetting failed");

  // тут нужно указать псо для gbuffer
  hr = commandList->Reset(commandAllocator[frameIndex], pipelineStateObject);
  throwIfFailed(hr, "Command list resetting failed");
}

void DX12Render::computePart() {
  // здесь неплохо было бы посчитать матрицы
  throw std::runtime_error("ComputePart not implemented yet");
}

// тут мне скорее всего нужно количество объектов разного типа
void DX12Render::gBufferPart(const uint32_t &boxCount, const uint32_t &icosahedronCount, const uint32_t &coneCount) {
  //HRESULT hr;

  // начинаем записывать команды к исполнению (ничего не требуется дополнительно)
  commandList->RSSetViewports(1, &viewport); // вьюпорт
  commandList->RSSetScissorRects(1, &scissorRect); // скиссор

  // информация текстурках которые мы будем заполнять нормалями и цветом
  CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(gBuffer.cDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), 0, rtvDescriptorSize);
  // буфер глубины
  CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(gBuffer.dDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

  // здесь мы биндим гБуфер (цвет, нормали, глубина)
  // true указывает что мы хотим прочитать дескриптор из одного указателя (то есть они лежат в одном месте друг за другом)
  commandList->OMSetRenderTargets(2, &rtvHandle, true, &dsvHandle);

  // Clear the render target by using the ClearRenderTargetView command
  const float clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
  commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
  commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

  // рут сигнатура (это для шейдеров, я не до конца понимаю зачем это) (PipelineLayout ?)
  commandList->SetGraphicsRootSignature(rootSignature);

  // еще мне нужно указать на константный буфер, куда я положил viewproj матрицу
  ID3D12DescriptorHeap* descriptorHeaps[] = {constantBufferDescriptor};
  commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
  commandList->SetGraphicsRootDescriptorTable(0, constantBufferDescriptor->GetGPUDescriptorHandleForHeapStart());

  commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // топология
  //commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP); // топология
  commandList->IASetVertexBuffers(0, 1, &boxVertexBufferView); // вершинный и индексный буфер будет один?
  commandList->IASetVertexBuffers(1, 1, &instanceBufferView);
  commandList->IASetIndexBuffer(&boxIndexBufferView);
  
  const uint32_t planeStart = 0;
  const uint32_t boxStart = planeStart + 1;
  const uint32_t icosahedronStart = boxStart + boxCount;
  const uint32_t coneStart = icosahedronStart + icosahedronCount;

  const uint32_t boxVStart = planeMaterialCB.box.verticesStart;
  const uint32_t planeVStart = planeMaterialCB.plane.verticesStart;
  const uint32_t icosphereVStart = planeMaterialCB.icosphere.verticesStart;
  const uint32_t coneVStart = planeMaterialCB.cone.verticesStart;

  const uint32_t boxIStart = planeMaterialCB.box.indicesStart;
  const uint32_t planeIStart = planeMaterialCB.plane.indicesStart;
  const uint32_t icosphereIStart = planeMaterialCB.icosphere.indicesStart;
  const uint32_t coneIStart = planeMaterialCB.cone.indicesStart;

  const uint32_t boxICount = planeMaterialCB.box.indicesCount;
  const uint32_t planeICount = planeMaterialCB.plane.indicesCount;
  const uint32_t icosphereICount = planeMaterialCB.icosphere.indicesCount;
  const uint32_t coneICount = planeMaterialCB.cone.indicesCount;

  commandList->DrawIndexedInstanced(planeICount,     1,                planeIStart,     planeVStart,     planeStart);
  commandList->DrawIndexedInstanced(boxICount,       boxCount,         boxIStart,       boxVStart,       boxStart);
  commandList->DrawIndexedInstanced(icosphereICount, icosahedronCount, icosphereIStart, icosphereVStart, icosahedronStart);
  commandList->DrawIndexedInstanced(coneICount,      coneCount,        coneIStart,      coneVStart,      coneStart);

  //// нам обязательно нужно сменить лайоут текстурки в которую мы будем писать
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST));
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE));
  //commandList->CopyResource(renderTargets[frameIndex], gBuffer.color);
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET));

  //// после того как нарисуем нам также необходимо обратно поменять лайоут текстурки
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT));
}

void DX12Render::rayTracingPart() {
  // мне нужно подключить ресурсы с предыдущего шага. (синхронизация?)
  // и в шейдере вызвать трассировку по новым данным
  // задать верные данные для акселератион структур

  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_GENERIC_READ));
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_GENERIC_READ));
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_GENERIC_READ));

  {
    const D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_GENERIC_READ),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_GENERIC_READ),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_GENERIC_READ)
    };

    commandList->ResourceBarrier(_countof(barriers), barriers);
  }

  // не хочет чистить uav ресурс, что делать?
  /*const float clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
  commandList->ClearUnorderedAccessViewFloat(fallback.outputResourceDescriptors.gpuDescriptorHandle, 
                                             fallback.outputResourceDescriptors.cpuDescriptorHandle, 
                                             fallback.raytracingOutput, 
                                             clearColor, 
                                             0, nullptr);*/

  //static const auto dispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc) {
  //  // Since each shader table has only one shader record, the stride is same as the size.
  //  dispatchDesc->HitGroupTable.StartAddress = fallback.hitGroupShaderTable->GetGPUVirtualAddress();
  //  dispatchDesc->HitGroupTable.SizeInBytes = fallback.hitGroupShaderTable->GetDesc().Width;
  //  dispatchDesc->HitGroupTable.StrideInBytes = fallback.hitGroupShaderTableStrideInBytes;
  //  //dispatchDesc->HitGroupTable.StrideInBytes = dispatchDesc->HitGroupTable.SizeInBytes;
  //  dispatchDesc->MissShaderTable.StartAddress = fallback.missShaderTable->GetGPUVirtualAddress();
  //  dispatchDesc->MissShaderTable.SizeInBytes = fallback.missShaderTable->GetDesc().Width;
  //  dispatchDesc->MissShaderTable.StrideInBytes = fallback.missShaderTableStrideInBytes;
  //  //dispatchDesc->MissShaderTable.StrideInBytes = dispatchDesc->MissShaderTable.SizeInBytes;
  //  dispatchDesc->RayGenerationShaderRecord.StartAddress = fallback.rayGenShaderTable->GetGPUVirtualAddress();
  //  dispatchDesc->RayGenerationShaderRecord.SizeInBytes = fallback.rayGenShaderTable->GetDesc().Width;
  //  dispatchDesc->Width = 1280;
  //  dispatchDesc->Height = 720;
  //  dispatchDesc->Depth = 1;
  //  commandList->SetPipelineState1(stateObject);
  //  commandList->DispatchRays(dispatchDesc);
  //};

  commandList->SetComputeRootSignature(fallback.globalRootSignature);

  /*D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
  fallback.commandList->SetDescriptorHeaps(1, &rtHeap);
  commandList->SetComputeRootDescriptorTable(static_cast<UINT>(GlobalRootSignatureParams::OUTPUT_VIEW_SLOT), fallback.raytracingOutputResourceUAVGpuDescriptor);
  fallback.commandList->SetTopLevelAccelerationStructure(static_cast<UINT>(GlobalRootSignatureParams::ACCELERATION_STRUCTURE_SLOT), fallback.topLevelAccelerationStructurePointer);
  dispatchRays(fallback.commandList, fallback.stateObject, &dispatchDesc);*/

  // в туториале здесь копируют константные буферы сцены и прочее
  // пока что не буду этого делать

  const D3D12_DISPATCH_RAYS_DESC dispatchDesc{
    { // RayGenerationShaderRecord
      fallback.rayGenShaderTable->GetGPUVirtualAddress(), // StartAddress
      fallback.rayGenShaderTable->GetDesc().Width         // SizeInBytes
    },
    { // MissShaderTable
      fallback.missShaderTable->GetGPUVirtualAddress(), // StartAddress
      fallback.missShaderTable->GetDesc().Width,        // SizeInBytes
      fallback.missShaderTableStrideInBytes             // StrideInBytes
    },
    { // HitGroupTable
      fallback.hitGroupShaderTable->GetGPUVirtualAddress(),
      fallback.hitGroupShaderTable->GetDesc().Width,
      fallback.hitGroupShaderTableStrideInBytes
    },
    { // CallableShaderTable
      0,
      0,
      0
    },
    1280, // Width
    720,  // Height
    1     // Depth
  };

  fallback.commandList->SetDescriptorHeaps(1, &rtHeap.handle);
  // тут нужно подать вершинный и индексный буфер
  // но и не только их, еще текстуры цвет, нормали, глубина

  commandList->SetComputeRootDescriptorTable(static_cast<uint32_t>(GlobalRootSignatureParams::OUTPUT_VIEW_SLOT), fallback.outputResourceDescriptors.gpuDescriptorHandle);
  fallback.commandList->SetTopLevelAccelerationStructure(static_cast<uint32_t>(GlobalRootSignatureParams::ACCELERATION_STRUCTURE_SLOT), fallback.topLevelAccelerationStructurePointer);

  commandList->SetComputeRootDescriptorTable(static_cast<uint32_t>(GlobalRootSignatureParams::GBUFFER_TEXTURES), fallback.colorBufferDescriptor);
  commandList->SetComputeRootConstantBufferView(static_cast<uint32_t>(GlobalRootSignatureParams::SCENE_CONSTANT), fallback.sceneConstantBuffer->GetGPUVirtualAddress());
  commandList->SetComputeRootDescriptorTable(static_cast<uint32_t>(GlobalRootSignatureParams::VERTEX_BUFFERS), fallback.indexDescs.gpuDescriptorHandle);

  //D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
  /*fallback.commandList->SetDescriptorHeaps(1, &rtHeap);
  commandList->SetComputeRootDescriptorTable(static_cast<UINT>(GlobalRootSignatureParams::OUTPUT_VIEW_SLOT), fallback.raytracingOutputResourceUAVGpuDescriptor);
  fallback.commandList->SetTopLevelAccelerationStructure(static_cast<UINT>(GlobalRootSignatureParams::ACCELERATION_STRUCTURE_SLOT), fallback.topLevelAccelerationStructurePointer);*/
  //dispatchRays(fallback.commandList, fallback.stateObject, &dispatchDesc);

  fallback.commandList->SetPipelineState1(fallback.stateObject);
  fallback.commandList->DispatchRays(&dispatchDesc);

  // нам обязательно нужно сменить лайоут текстурки в которую мы будем писать
  //{
  //  const D3D12_RESOURCE_BARRIER barriers[] = {
  //    CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
  //    CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)
  //  };

  //  commandList->ResourceBarrier(_countof(barriers), barriers);
  //}
  ////commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST));
  ////commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
  //commandList->CopyResource(renderTargets[frameIndex], fallback.raytracingOutput);

  //{
  //  const D3D12_RESOURCE_BARRIER barriers[] = {
  //    CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
  //    CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT),
  //    CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET),
  //    CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.normal, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET),
  //    CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE)
  //  };

  //  commandList->ResourceBarrier(_countof(barriers), barriers);
  //}


  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  // после того как нарисуем нам также необходимо обратно поменять лайоут текстурки
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT));

  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET));
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.normal, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET));
  //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE));
}

void DX12Render::filterPart() {
  // здесь нужно воспользоваться temporal acumulation
  // затем передать данные на вход фильтру

  // то есть два шейдера 100%, какие то вещи дополнительные?
  // какие данные мне понадобятся? изображение после трассировки... что то еще? (количество лучей попавших в точку?)
  // temporal acumulation должен собирать эти данные в кучу + собирать данные с предыдущего кадра + считать количество попавших лучей?
  // projToPrevProj() - необходимо получить координаты пикселя на предыдущем кадре 
  // а это значит что нужно еще хранить предыдущий viewProj (ну и текущий инвертированный видимо)

  {
    const D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(fallback.raytracingOutput),
      CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_GENERIC_READ)
    };

    commandList->ResourceBarrier(_countof(barriers), barriers);
  }

  commandList->SetComputeRootSignature(filter.rootSignature);

  commandList->SetDescriptorHeaps(1, &filter.heap.handle);

  commandList->SetComputeRootDescriptorTable(0, filter.filterOutputUAVDesc);
  commandList->SetComputeRootDescriptorTable(1, filter.colorBufferDescriptor); // достаточно?
  commandList->SetComputeRootConstantBufferView(2, filter.constantBuffer->GetGPUVirtualAddress());

  commandList->SetPipelineState(filter.pso);

  #define LOCAL_WORK_GROUP 16

  const uint32_t dispatchX = glm::ceil(float(1280) / float(LOCAL_WORK_GROUP));
  const uint32_t dispatchY = glm::ceil(float(720)  / float(LOCAL_WORK_GROUP));
  commandList->Dispatch(dispatchX, dispatchY, 1);

  {
    const D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
      CD3DX12_RESOURCE_BARRIER::Transition(filter.filterOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
      CD3DX12_RESOURCE_BARRIER::Transition(filter.colorLast, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST),
      CD3DX12_RESOURCE_BARRIER::Transition(filter.depthLast, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_DEST),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COPY_SOURCE)
    };

    commandList->ResourceBarrier(_countof(barriers), barriers);
  }
  
  // мне еще нужно скопировать в ласт буферы
  commandList->CopyResource(renderTargets[frameIndex], filter.filterOutput);
  commandList->CopyResource(filter.colorLast, filter.filterOutput);
  commandList->CopyResource(filter.depthLast, gBuffer.depth);

  {
    const D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::Transition(filter.filterOutput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
      CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.color, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.normal, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET),
      CD3DX12_RESOURCE_BARRIER::Transition(gBuffer.depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE),
      CD3DX12_RESOURCE_BARRIER::Transition(filter.colorLast, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
      CD3DX12_RESOURCE_BARRIER::Transition(filter.depthLast, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ),
      CD3DX12_RESOURCE_BARRIER::Transition(fallback.raytracingOutput, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    };

    commandList->ResourceBarrier(_countof(barriers), barriers);
  }
}

void DX12Render::endFrame() {
  HRESULT hr;

  hr = commandList->Close();
  throwIfFailed(hr, "Command list closing failed");

  // create an array of command lists (only one command list here)
  ID3D12CommandList* ppCommandLists[] = {commandList};

  // execute the array of command lists
  commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // this command goes in at the end of our command queue. we will know when our command queue 
  // has finished because the fence value will be set to "fenceValue" from the GPU since the command
  // queue is being executed on the GPU
  /*hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
  if (FAILED(hr)) {
    MessageBox(0, L"Command queue signaling failed", L"Error", MB_OK);
    return false;
  }*/

  // present the current backbuffer
  hr = swapChain->Present(1, 0);
  throwIfFailed(hr, "Swapchain presenting failed");

  moveToNextFrame();
}

void DX12Render::cleanup() {
  if (device == nullptr) return;
  
  SAFE_RELEASE(device)
  
  if (swapChain == nullptr) return;

  for (uint32_t i = 0; i < frameBufferCount; ++i) {
    frameIndex = i;
    waitForRenderContext();
  }

  // get swapchain out of full screen before exiting
  BOOL fs = false;
  if (swapChain->GetFullscreenState(&fs, NULL)) swapChain->SetFullscreenState(false, NULL);

  SAFE_RELEASE(commandQueue)
  SAFE_RELEASE(rtvDescriptorHeap)
  SAFE_RELEASE(commandList)
  SAFE_RELEASE(depthStencilBuffer)
  SAFE_RELEASE(dsDescriptorHeap)

  SAFE_RELEASE(pipelineStateObject)
  SAFE_RELEASE(rootSignature)

  SAFE_RELEASE(boxVertexBuffer)
  SAFE_RELEASE(boxIndexBuffer)
  SAFE_RELEASE(instanceBuffer)

  SAFE_RELEASE(constantBuffer)
  SAFE_RELEASE(constantBufferDescriptor)
  constantBufferPtr = nullptr;

  SAFE_RELEASE(gBuffer.color)
  SAFE_RELEASE(gBuffer.normal)
  SAFE_RELEASE(gBuffer.depth)
  SAFE_RELEASE(gBuffer.cDescriptorHeap)
  SAFE_RELEASE(gBuffer.dDescriptorHeap)

  for (int i = 0; i < frameBufferCount; ++i) {
    SAFE_RELEASE(renderTargets[i])
    SAFE_RELEASE(commandAllocator[i])
    SAFE_RELEASE(fence[i])
  };

  SAFE_RELEASE(rtHeap.handle)

  SAFE_RELEASE(fallback.device)
  SAFE_RELEASE(fallback.commandList)
  SAFE_RELEASE(fallback.stateObject)
  SAFE_RELEASE(fallback.globalRootSignature)
  SAFE_RELEASE(fallback.localRootSignature)
  SAFE_RELEASE(fallback.accelerationStructure)
  SAFE_RELEASE(fallback.bottomLevelAccelerationStructure)
  SAFE_RELEASE(fallback.topLevelAccelerationStructure)
  SAFE_RELEASE(fallback.raytracingOutput)
  SAFE_RELEASE(fallback.missShaderTable)
  SAFE_RELEASE(fallback.hitGroupShaderTable)
  /*for (uint32_t i = 0; i < raysTypeCount; ++i) {
    SAFE_RELEASE(fallback.missShaderTable[i])
    SAFE_RELEASE(fallback.hitGroupShaderTable[i])
  }*/
  SAFE_RELEASE(fallback.rayGenShaderTable)

  CloseHandle(fenceEvent);
}

//void DX12Render::waitForFrame() {
//  HRESULT hr;
//
//  // Update the frame index.
//  frameIndex = swapChain->GetCurrentBackBufferIndex();
//
//  // If the next frame is not ready to be rendered yet, wait until it is ready.
//  if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex]) {
//    hr = fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
//    throwIfFailed(hr, "Fence setting event failed");
//
//    WaitForSingleObjectEx(fenceEvent, INFINITE, FALSE);
//  }
//
//  // Set the fence value for the next frame.
//  //fenceValue[frameIndex] = currentFenceValue + 1;
//  ++fenceValue[frameIndex];
//  // достаточно ли этого? или нужно обязательно значение прошлого кадра? (не понимаю)
//}

//void DX12Render::waitForCurrentFrame() {
//  HRESULT hr;
//
//  // Schedule a Signal command in the queue.
//  /*hr = commandQueue->Signal(fence[frameIndex], fenceValue[frameIndex]);
//  if (FAILED(hr)) {
//    MessageBox(0, L"Command queue signal failed", L"Error", MB_OK);
//    return false;
//  }*/
//
//  // Wait until the fence has been processed.
//  hr = fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
//  throwIfFailed(hr, "Fence setting event failed");
//
//  WaitForSingleObjectEx(fenceEvent, INFINITE, FALSE);
//
//  // Increment the fence value for the current frame.
//  fenceValue[frameIndex]++;
//}

void DX12Render::waitForRenderContext() {
  HRESULT hr;

  // Add a signal command to the queue.
  hr = commandQueue->Signal(fence[frameIndex], contextFenceValue);
  throwIfFailed(hr, "Command queue signaling failed");

  // Instruct the fence to set the event object when the signal command completes.
  hr = fence[frameIndex]->SetEventOnCompletion(contextFenceValue, fenceEvent);
  throwIfFailed(hr, "Fence setting event failed");

  contextFenceValue++;

  // Wait until the signal command has been processed.
  WaitForSingleObject(fenceEvent, INFINITE);
}

// Cycle through the frame resources. This method blocks execution if the 
// next frame resource in the queue has not yet had its previous contents 
// processed by the GPU.
void DX12Render::moveToNextFrame() {
  HRESULT hr;

  // Assign the current fence value to the current frame.
  fenceValue[frameIndex] = contextFenceValue;

  // Signal and increment the fence value.
  hr = commandQueue->Signal(fence[frameIndex], contextFenceValue);
  throwIfFailed(hr, "Command queue signaling failed");

  contextFenceValue++;

  // Update the frame index.
  frameIndex = swapChain->GetCurrentBackBufferIndex();

  // If the next frame is not ready to be rendered yet, wait until it is ready.
  if (fence[frameIndex]->GetCompletedValue() < fenceValue[frameIndex]) {
    hr = fence[frameIndex]->SetEventOnCompletion(fenceValue[frameIndex], fenceEvent);
    throwIfFailed(hr, "Fence setting event failed");

    WaitForSingleObject(fenceEvent, INFINITE);
  }
}

ID3D12Device* DX12Render::getDevice() const {
  return device;
}

void load(std::string &path, std::vector<uint32_t> &indices, std::vector<Vertex> &vertices) {
  std::string warn;
  std::string err;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), nullptr, true);

  throwIf(!ret, err);

  //throwIf(true, "shapes size " + std::to_string(shapes.size()) + " indices " + std::to_string(shapes[0].mesh.indices.size()) + " vertices " + std::to_string(attrib.vertices.size()));

  // Loop over shapes
  uint32_t index = 0;
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      uint32_t vertex_offset = vertices.size();
      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
        //tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
        //tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
        
        const glm::vec4 vert = glm::vec4(vx, vy, vz, 1.0f);
        const glm::vec4 normal = glm::vec4(nx, ny, nz, 0.0f);
        const glm::vec2 texCoords = glm::vec2(0.0f, 0.0f);

        const Vertex vertex{
          vert,
          normal,
          texCoords
        };

        vertices.push_back(vertex);
        indices.push_back(index);
        ++index;
      }

      throwIf(fv < 3, "bad obj file");



      index_offset += fv;

      // per-face material
      shapes[s].mesh.material_ids[f];
    }
  }
}

void DX12Render::loadModels(std::vector<uint32_t> &indices, std::vector<Vertex> &vertices) {
  uint32_t lastIndicesCount = 0;
  uint32_t lastVerticesCount = 0;

  {
    std::string obj = "plane.obj";
    load(obj, indices, vertices);

    planeMaterialCB.plane.indicesStart = lastIndicesCount;
    planeMaterialCB.plane.indicesCount = indices.size() - lastIndicesCount;
    planeMaterialCB.plane.indicesSize = planeMaterialCB.plane.indicesCount * sizeof(uint32_t);

    planeMaterialCB.plane.verticesStart = lastVerticesCount;
    planeMaterialCB.plane.verticesCount = vertices.size() - lastVerticesCount;
    planeMaterialCB.plane.verticesSize = planeMaterialCB.plane.verticesCount * sizeof(Vertex);

    lastIndicesCount = indices.size();
    lastVerticesCount = vertices.size();
  }

  //throwIf(indices.empty(), "wrong indices");
  //throwIf(vertices.empty(), "wrong vertices");

  {
    std::string obj = "box.obj";
    load(obj, indices, vertices);

    planeMaterialCB.box.indicesStart = lastIndicesCount;
    planeMaterialCB.box.indicesCount = indices.size() - lastIndicesCount;
    planeMaterialCB.box.indicesSize = planeMaterialCB.box.indicesCount * sizeof(uint32_t);

    planeMaterialCB.box.verticesStart = lastVerticesCount;
    planeMaterialCB.box.verticesCount = vertices.size() - lastVerticesCount;
    planeMaterialCB.box.verticesSize = planeMaterialCB.box.verticesCount * sizeof(Vertex);

    lastIndicesCount = indices.size();
    lastVerticesCount = vertices.size();
  }

  {
    std::string obj = "icosphere.obj";
    load(obj, indices, vertices);

    planeMaterialCB.icosphere.indicesStart = lastIndicesCount;
    planeMaterialCB.icosphere.indicesCount = indices.size() - lastIndicesCount;
    planeMaterialCB.icosphere.indicesSize = planeMaterialCB.icosphere.indicesCount * sizeof(uint32_t);

    planeMaterialCB.icosphere.verticesStart = lastVerticesCount;
    planeMaterialCB.icosphere.verticesCount = vertices.size() - lastVerticesCount;
    planeMaterialCB.icosphere.verticesSize = planeMaterialCB.icosphere.verticesCount * sizeof(Vertex);

    lastIndicesCount = indices.size();
    lastVerticesCount = vertices.size();
  }

  {
    std::string obj = "cone.obj";
    load(obj, indices, vertices);

    planeMaterialCB.cone.indicesStart = lastIndicesCount;
    planeMaterialCB.cone.indicesCount = indices.size() - lastIndicesCount;
    planeMaterialCB.cone.indicesSize = planeMaterialCB.cone.indicesCount * sizeof(uint32_t);

    planeMaterialCB.cone.verticesStart = lastVerticesCount;
    planeMaterialCB.cone.verticesCount = vertices.size() - lastVerticesCount;
    planeMaterialCB.cone.verticesSize = planeMaterialCB.cone.verticesCount * sizeof(Vertex);

    lastIndicesCount = indices.size();
    lastVerticesCount = vertices.size();
  }
}

//void DX12Render::createRTResources(const uint32_t &width, const uint32_t &height) {
//  // Initialize raytracing pipeline.
//
//  // Create raytracing interfaces: raytracing device and commandlist.
//  createRayTracingFallbackDevice();
//
//  // Create root signatures for the shaders.
//  createRootSignatures();
//
//  // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
//  createRaytracingPSO();
//
//  // Create a heap for descriptors.
//  createRTDescriptorHeap();
//
//  // здесь мы копируем геометрию на устройство, не особо полезная штука
//  //buildGeometry();
//  createDescriptors();
//  initializeScene();
//
//  // Build raytracing acceleration structures from the generated geometry.
//  // здесь мне нужно создать top и bottom level структуры, я пока еще не очень понимаю как это выглядит
//  // как эти структуры соответствуют ресурсам шейдеров? 
//  buildAccelerationStructures();
//  //buildAccelerationStructures2();
//
//  // Build shader tables, which define shaders and their local root arguments.
//  buildShaderTables();
//
//  // Create an output 2D texture to store the raytracing result to.
//  createRaytracingOutputResource(width, height);
//}

void DX12Render::createRayTracingFallbackDevice() {
  HRESULT hr;

  CreateRaytracingFallbackDeviceFlags createDeviceFlags = fallback.forceComputeFallback ?
    CreateRaytracingFallbackDeviceFlags::ForceComputeFallback :
    CreateRaytracingFallbackDeviceFlags::None;

  hr = D3D12CreateRaytracingFallbackDevice(device, createDeviceFlags, 0, IID_PPV_ARGS(&fallback.device));
  throwIfFailed(hr, "Could not create raytracing fallback device");

  fallback.device->QueryRaytracingCommandList(commandList, IID_PPV_ARGS(&fallback.commandList));
}

void DX12Render::createRootSignatures() {
  //HRESULT hr;

  // Global Root Signature
  // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
  {
    // в сэмплах майкрософт в хелло ворлде используются 2 слота прибавится ли их количество в будущем? да
    // я только не понимаю правила порядка объявления этих штук в hlsl

    /*CD3DX12_DESCRIPTOR_RANGE UAVDescriptor;
    UAVDescriptor.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

    CD3DX12_ROOT_PARAMETER rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::COUNT)];
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::OUTPUT_VIEW_SLOT)].InitAsDescriptorTable(1, &UAVDescriptor);
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::ACCELERATION_STRUCTURE_SLOT)].InitAsShaderResourceView(0);

    CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);

    serializeRootSignature(globalRootSignatureDesc, &fallback.globalRootSignature);*/

    CD3DX12_DESCRIPTOR_RANGE ranges[3];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1);
    ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 3);

    CD3DX12_ROOT_PARAMETER rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::COUNT)];
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::OUTPUT_VIEW_SLOT)].InitAsDescriptorTable(1, &ranges[0]);
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::ACCELERATION_STRUCTURE_SLOT)].InitAsShaderResourceView(0);
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::GBUFFER_TEXTURES)].InitAsDescriptorTable(1, &ranges[2]);
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::SCENE_CONSTANT)].InitAsConstantBufferView(0);
    rootParameters[static_cast<uint32_t>(GlobalRootSignatureParams::VERTEX_BUFFERS)].InitAsDescriptorTable(1, &ranges[1]);

    CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);

    serializeRootSignature(globalRootSignatureDesc, &fallback.globalRootSignature);
  }

  // Local Root Signature
  // This is a root signature that enables a shader to have unique arguments that come from shader tables.
  {
    CD3DX12_ROOT_PARAMETER rootParameters[static_cast<uint32_t>(LocalRootSignatureParams::COUNT)];
    rootParameters[static_cast<uint32_t>(LocalRootSignatureParams::VIEWPORT_CONSTANT_SLOT)].InitAsConstants(sizeAliquot32bits(planeMaterialCB), 1, 0);

    CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
    localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

    serializeRootSignature(localRootSignatureDesc, &fallback.localRootSignature);
  }
}

void DX12Render::serializeRootSignature(const D3D12_ROOT_SIGNATURE_DESC &desc, ID3D12RootSignature** rootSig) {
  HRESULT hr;
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;

  hr = fallback.device->D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
  const std::string err = error ? static_cast<char*>(error->GetBufferPointer()) : "";
  throwIfFailed(hr, err);
  
  hr = fallback.device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(rootSig));
  throwIfFailed(hr, "Cannot create root signature");
}

void DX12Render::createRaytracingPSO() {
  HRESULT hr;

#if defined(_DEBUG)
  // Enable better shader debugging with the graphics debugging tools.
  UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compileFlags = 0;
#endif

  //ID3DBlob* rayTracingShader = nullptr; // d3d blob for holding vertex shader bytecode
  //ID3DBlob* errorBuff = nullptr; // a buffer holding the error data if any
  //hr = D3DCompileFromFile(L"rayTracing.hlsl",
  //  nullptr,
  //  nullptr,
  //  "",
  //  "lib_6_3",
  //  compileFlags,
  //  0,
  //  &rayTracingShader,
  //  &errorBuff);

  //if (FAILED(hr) || rayTracingShader == nullptr) {
  //  OutputDebugStringA((char*)errorBuff->GetBufferPointer());
  //  throw std::runtime_error("Raygen shader creation error");
  //}

  //unsigned char program[] = "float4 main() : SV_Target { return 1; }";
  //IDxcLibrary *pLibrary;
  //IDxcBlobEncoding *pSource;
  //DxcCreateInstance(CLSID_DxcLibrary, __uuidof(IDxcLibrary), (void **)&pLibrary);
  //pLibrary->CreateBlobWithEncodingFromPinned(program, _countof(program), CP_UTF8, &pSource);

  // fill out a shader bytecode structure, which is basically just a pointer
  // to the shader bytecode and the size of the shader bytecode
  D3D12_SHADER_BYTECODE rayTracingShaderBytecode = {
    /*rayTracingShader->GetBufferPointer(),
    rayTracingShader->GetBufferSize()*/
    (void*)raytracing,
    ARRAYSIZE(raytracing)
  };

  // Create 7 subobjects that combine into a RTPSO:
    // Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
    // Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
    // This simple sample utilizes default shader association except for local root signature subobject
    // which has an explicit association specified purely for demonstration purposes.
    // 1 - DXIL library
    // 2 - Triangle hit group (1 x 2 ray rypes)
    // 1 - Shader config
    // 2 - Local root signature and association (more)
    // 1 - Global root signature
    // 1 - Pipeline config
  CD3D12_STATE_OBJECT_DESC raytracingPipeline{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE};

  // DXIL library
  // This contains the shaders and their entrypoints for the state object.
  // Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
  auto dxil = raytracingPipeline.CreateSubobject<CD3D12_DXIL_LIBRARY_SUBOBJECT>();
  dxil->SetDXILLibrary(&rayTracingShaderBytecode);

  /*dxil->DefineExport(raygenShaderName);
  dxil->DefineExport(closestHitShaderName);
  dxil->DefineExport(missShaderName);*/

  // Triangle hit group
  // A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
  // In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
  // скорее всего мне нужно задать только треугольную геометрию
  for (uint32_t i = 0; i < raysTypeCount; ++i) {
    auto hitGroup = raytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
    if (i == 0) {
      hitGroup->SetClosestHitShaderImport(closestHitShaderName2);
    }

    hitGroup->SetHitGroupExport(hitGroupNames2[i]);
    hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
  }

  /*auto hitGroup = raytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
  hitGroup->SetClosestHitShaderImport(closestHitShaderName);
  hitGroup->SetHitGroupExport(hitGroupName);
  hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);*/

  // Shader config
  // Defines the maximum sizes in bytes for the ray payload and attribute structure.
  auto shaderConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
  const uint32_t payloadSize = glm::max(sizeof(RayPayload), sizeof(ShadowRayPayload)); // float4 color
  // нужны ли они мне?
  const uint32_t attributeSize = 8; // в хит шейдерах всегда должен быть аттрибьют
  // по умолчанию он 8 байт (я не знаю что там)
  shaderConfig->Config(payloadSize, attributeSize);

  // CreateComputePipelineState: Root Signature doesn't match Compute Shader: Shader SRV descriptor range (RegisterSpace=0, NumDescriptors=1, BaseShaderRegister=1) is not fully bound in root signature
  // Local root signature and shader association
  createLocalRootSignatureSubobjects(&raytracingPipeline);
  // This is a root signature that enables a shader to have unique arguments that come from shader tables.

  // Global root signature
  // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
  auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3D12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
  globalRootSignature->SetRootSignature(fallback.globalRootSignature);

  // Pipeline config
  // Defines the maximum TraceRay() recursion depth.
  auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
  // PERFOMANCE TIP: Set max recursion depth as low as needed 
  // as drivers may apply optimization strategies for low recursion depths. 
  const uint32_t maxRecursionDepth = MAX_RAY_RECURSION_DEPTH; // 3 
  pipelineConfig->Config(maxRecursionDepth);

  hr = fallback.device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&fallback.stateObject));
  throwIfFailed(hr, "Couldn't create DirectX Raytracing state object.");
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void DX12Render::createLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* raytracingPipeline) {
  // нам походу локал рут сигнатура нужна только в хит шейдерах
  {
    auto localRootSignature = raytracingPipeline->CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
    localRootSignature->SetRootSignature(fallback.localRootSignature);

    // Shader association
    auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
    rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
    rootSignatureAssociation->AddExport(raygenShaderName2);
    rootSignatureAssociation->AddExports(hitGroupNames2);
  }
}

void DX12Render::createRTDescriptorHeap() {
  HRESULT hr;
  const uint32_t bottomStructCount = bottomLevelCount; // 4x bottom level acceleration struct
  const uint32_t topStructCount = 1;                   // top level acceleration struct
  const uint32_t rtBuffers = 2;                        // const buffer, material
  const uint32_t indexVertexBuffers = 2;               // index and vertex buffers
  const uint32_t rtTextures = 4;                       // output, color, normal, depth
  const uint32_t descriptorsCount = bottomStructCount + topStructCount + rtBuffers + indexVertexBuffers + rtTextures;

  const D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    descriptorsCount, // ?
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    0
  };

  // Allocate a heap for 13 descriptors:
  // 5 (?) - 4x bottom and a top level acceleration structure
  // 2 - rt buffers (const buffer, material)
  // 2 - index and vertex buffers
  // 4 - rt textures (output, color, normal, depth)
  hr = device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&rtHeap.handle));
  throwIfFailed(hr, "Could not create descriptor heap for ray tracing");
  rtHeap.handle->SetName(L"descriptor heap for ray tracing");

  rtHeap.hardwareSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

//void DX12Render::buildAccelerationStructures() {
//  waitForRenderContext();
//
//  HRESULT hr;
//
//  hr = commandAllocator[frameIndex]->Reset();
//  throwIfFailed(hr, "Command allocator resetting failed");
//
//  // тут нужно указать псо для gbuffer
//  hr = commandList->Reset(commandAllocator[frameIndex], nullptr);
//  throwIfFailed(hr, "Command list resetting failed");
//
//  // здесь нужно создать GEOMETRY_TYPE_COUNT боттом структур
//  // +1 топ структуру (правильно ли?)
//  // мне скорее всего ненужны аабб структуры, так как у меня нет процедурной геометрии
//
//  std::vector<AccelerationStructureBuffers> bottomASs;
//  std::array<std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>, bottomLevelCount> geometryDescs;
//  {
//    buildGeometryDesc(geometryDescs);
//
//    for (uint32_t i = 0; i < bottomLevelCount; ++i) {
//      const AccelerationStructureBuffers as = buildBottomLevel(geometryDescs[i]);
//      bottomASs.push_back(as);
//    }
//  }
//
//  // Batch all resource barriers for bottom-level AS builds.
//  D3D12_RESOURCE_BARRIER resourceBarriers[bottomLevelCount];
//  for (UINT i = 0; i < bottomLevelCount; ++i) {
//    resourceBarriers[i] = CD3DX12_RESOURCE_BARRIER::UAV(bottomASs[i].accelerationStructure);
//  }
//  commandList->ResourceBarrier(bottomLevelCount, resourceBarriers);
//
//  // короч у меня проблема жуткая со структурами. Скорее всего мне их нужно с нуля посоздавать
//  // Build top-level AS.
//  AccelerationStructureBuffers topLevelAS = buildTopLevel(bottomASs.size(), bottomASs.data());
//
//  hr = commandList->Close();
//  throwIfFailed(hr, "Failed to close command list");
//  ID3D12CommandList *commandLists[] = {commandList};
//  commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
//
//  // Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
//  waitForRenderContext();
//
//  // Store the AS buffers. The rest of the buffers will be released once we exit the function.
//  for (UINT i = 0; i < bottomLevelCount; i++) {
//    fallback.bottomLevels[i] = bottomASs[i].accelerationStructure;
//  }
//  fallback.topLevel = topLevelAS.accelerationStructure;
//
//  // может эти структуры нужно пересобирать каждый кадр для движущихся объектов?
//  // ну то что нужно пересобирать это более менее понятно (что конкретно должно происходить?)
//  for (UINT i = 0; i < bottomLevelCount; i++) {
//    SAFE_RELEASE(bottomASs[i].instanceDesc)
//    SAFE_RELEASE(bottomASs[i].scratch)
//  }
//  SAFE_RELEASE(topLevelAS.instanceDesc)
//  SAFE_RELEASE(topLevelAS.scratch)
//  
//  
//
//  //const D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc{
//  //  D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
//
//  //  // Mark the geometry as opaque. 
//  //  // PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
//  //  // Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
//  //  D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
//  //  {
//  //    0,
//  //    DXGI_FORMAT_R32_UINT,
//  //    DXGI_FORMAT_R32G32B32_FLOAT,
//  //    boxIndicesCount + planeIndicesCount + icosahedronIndicesCount + coneIndicesCount,
//  //    boxVerticesCount + planeVerticesCount + icosahedronVerticesCount + coneVerticesCount,
//  //    boxIndexBuffer->GetGPUVirtualAddress(),
//  //    {
//  //      boxVertexBuffer->GetGPUVirtualAddress(),
//  //      sizeof(Vertex)
//  //    }
//  //  }
//  //};
//
//  //// Get required sizes for an acceleration structure.
//  //const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
//  //D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs{
//  //  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
//  //  buildFlags,
//  //  1,
//  //  D3D12_ELEMENTS_LAYOUT_ARRAY,
//  //  0
//  //};
//
//  //D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo{};
//  //fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
//
//  //throwIf(topLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0, "topLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0");
//
//  //D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomLevelInputs = topLevelInputs;
//  //bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
//  //bottomLevelInputs.pGeometryDescs = &geometryDesc;
//
//  //D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo{};
//  //fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &bottomLevelPrebuildInfo);
//
//  //throwIf(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0, "bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0");
//
//  //ID3D12Resource* scratchResource = nullptr;
//  //allocateUAVBuffer(device, 
//  //                  glm::max(topLevelPrebuildInfo.ScratchDataSizeInBytes, bottomLevelPrebuildInfo.ScratchDataSizeInBytes), 
//  //                  &scratchResource, 
//  //                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 
//  //                  L"ScratchResource");
//
//  //// Allocate resources for acceleration structures.
//  //  // Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
//  //  // Default heap is OK since the application doesn’t need CPU read/write access to them. 
//  //  // The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
//  //  // and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
//  //  //  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
//  //  //  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
//  //{
//  //  D3D12_RESOURCE_STATES initialResourceState = fallback.device->GetAccelerationStructureResourceState();
//
//  //  allocateUAVBuffer(device, 
//  //                    bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, 
//  //                    &fallback.bottomLevelAccelerationStructure, 
//  //                    initialResourceState, 
//  //                    L"BottomLevelAccelerationStructure");
//  //  allocateUAVBuffer(device, 
//  //                    topLevelPrebuildInfo.ResultDataMaxSizeInBytes, 
//  //                    &fallback.topLevelAccelerationStructure, 
//  //                    initialResourceState, 
//  //                    L"TopLevelAccelerationStructure");
//  //}
//
//  //// Note on Emulated GPU pointers (AKA Wrapped pointers) requirement in Fallback Layer:
//  //// The primary point of divergence between the DXR API and the compute-based Fallback layer is the handling of GPU pointers. 
//  //// DXR fundamentally requires that GPUs be able to dynamically read from arbitrary addresses in GPU memory. 
//  //// The existing Direct Compute API today is more rigid than DXR and requires apps to explicitly inform the GPU what blocks of memory it will access with SRVs/UAVs.
//  //// In order to handle the requirements of DXR, the Fallback Layer uses the concept of Emulated GPU pointers, 
//  //// which requires apps to create views around all memory they will access for raytracing, 
//  //// but retains the DXR-like flexibility of only needing to bind the top level acceleration structure at DispatchRays.
//  ////
//  //// The Fallback Layer interface uses WRAPPED_GPU_POINTER to encapsulate the underlying pointer
//  //// which will either be an emulated GPU pointer for the compute - based path or a GPU_VIRTUAL_ADDRESS for the DXR path.
//
//  //// Create an instance desc for the bottom-level acceleration structure.
//  //ID3D12Resource* instanceDescs = nullptr;
//  //{
//  //  D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC instanceDesc{};
//  //  instanceDesc.Transform[0][0] = instanceDesc.Transform[1][1] = instanceDesc.Transform[2][2] = 1;
//  //  instanceDesc.InstanceMask = 1;
//  //  uint32_t numBufferElements = static_cast<uint32_t>(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes) / sizeof(uint32_t);
//  //  instanceDesc.AccelerationStructure = createFallbackWrappedPointer(fallback.bottomLevelAccelerationStructure, numBufferElements);
//  //  allocateUploadBuffer(device, &instanceDesc, sizeof(instanceDesc), &instanceDescs, L"InstanceDescs");
//  //}
//
//  //{
//  //  // Create a wrapped pointer to the acceleration structure.
//  //  uint32_t numBufferElements = static_cast<uint32_t>(topLevelPrebuildInfo.ResultDataMaxSizeInBytes) / sizeof(uint32_t);
//  //  fallback.topLevelAccelerationStructurePointer = createFallbackWrappedPointer(fallback.topLevelAccelerationStructure, numBufferElements);
//  //}
//
//  //// Bottom Level Acceleration Structure desc
//  //D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc{
//  //  fallback.bottomLevelAccelerationStructure->GetGPUVirtualAddress(),
//  //  bottomLevelInputs,
//  //  0,
//  //  scratchResource->GetGPUVirtualAddress()
//  //};
//
//  //topLevelInputs.InstanceDescs = instanceDescs->GetGPUVirtualAddress();
//  //D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc{
//  //  fallback.topLevelAccelerationStructure->GetGPUVirtualAddress(),
//  //  topLevelInputs,
//  //  0,
//  //  scratchResource->GetGPUVirtualAddress()
//  //};
//
//  //const auto buildAccelerationStructure = [&](auto* raytracingCommandList) {
//  //  raytracingCommandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
//  //  commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(fallback.bottomLevelAccelerationStructure));
//  //  raytracingCommandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
//  //};
//
//  //// Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
//  //ID3D12DescriptorHeap *pDescriptorHeaps[] = {rtHeap};
//  //fallback.commandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);
//  //buildAccelerationStructure(fallback.commandList);
//
//  //// Kick off acceleration structure construction.
//  //hr = commandList->Close();
//  //throwIfFailed(hr, "Failed to close command list");
//  //ID3D12CommandList *commandLists[] = {commandList};
//  //commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
//
//  //// Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
//  //waitForRenderContext();
//}

void DX12Render::buildAccelerationStructures2(const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer) {
  waitForRenderContext();

  HRESULT hr;

  hr = commandAllocator[frameIndex]->Reset();
  throwIfFailed(hr, "Command allocator resetting failed");

  // тут нужно указать псо для gbuffer
  hr = commandList->Reset(commandAllocator[frameIndex], nullptr);
  throwIfFailed(hr, "Command list resetting failed");

  AccelerationStructureBuffers bottomLevelBuffers[bottomLevelCount];

  std::array<std::vector<BottomASCreateInfo>, bottomLevelCount> createInfos;
  uint32_t index = static_cast<uint32_t>(GeometryType::PLANE);
  //uint32_t index = 0;
  createInfos[index].resize(1);
  createInfos[index][0].indexCount = planeMaterialCB.plane.indicesCount; //planeIndicesCount
  createInfos[index][0].indexOffset = planeMaterialCB.plane.indicesStart * sizeof(uint32_t); //planeIndicesStart 
  createInfos[index][0].indexBuffer = boxIndexBuffer;
  createInfos[index][0].vertexCount = planeMaterialCB.plane.verticesCount; //planeVerticesCount
  createInfos[index][0].vertexOffset = planeMaterialCB.plane.verticesStart * sizeof(Vertex);//planeVerticesStart
  createInfos[index][0].vertexStride = sizeof(Vertex);
  createInfos[index][0].vertexBuffer = boxVertexBuffer;
  
  index = static_cast<uint32_t>(GeometryType::AABB);
  //index = 1;
  createInfos[index].resize(1);
  createInfos[index][0].indexCount = planeMaterialCB.box.indicesCount; //boxIndicesCount
  createInfos[index][0].indexOffset = planeMaterialCB.box.indicesStart * sizeof(uint32_t); //boxIndicesStart
  createInfos[index][0].indexBuffer = boxIndexBuffer;
  createInfos[index][0].vertexCount = planeMaterialCB.box.verticesCount; //boxVerticesCount
  createInfos[index][0].vertexOffset = planeMaterialCB.box.verticesStart * sizeof(Vertex); //boxVerticesStart
  createInfos[index][0].vertexStride = sizeof(Vertex);
  createInfos[index][0].vertexBuffer = boxVertexBuffer;

  index = static_cast<uint32_t>(GeometryType::ICOSAHEDRON);
  //index = 2;
  createInfos[index].resize(1);
  createInfos[index][0].indexCount = planeMaterialCB.icosphere.indicesCount; //icosahedronIndicesCount
  createInfos[index][0].indexOffset = planeMaterialCB.icosphere.indicesStart * sizeof(uint32_t); //icosahedronIndicesStart
  createInfos[index][0].indexBuffer = boxIndexBuffer;
  createInfos[index][0].vertexCount = planeMaterialCB.icosphere.verticesCount; //icosahedronVerticesCount
  createInfos[index][0].vertexOffset = planeMaterialCB.icosphere.verticesStart * sizeof(Vertex); //icosahedronVerticesStart
  createInfos[index][0].vertexStride = sizeof(Vertex);
  createInfos[index][0].vertexBuffer = boxVertexBuffer;

  index = static_cast<uint32_t>(GeometryType::CONE);
  //index = 3;
  createInfos[index].resize(1);
  createInfos[index][0].indexCount = planeMaterialCB.cone.indicesCount; //coneIndicesCount
  createInfos[index][0].indexOffset = planeMaterialCB.cone.indicesStart * sizeof(uint32_t); //coneIndicesStart
  createInfos[index][0].indexBuffer = boxIndexBuffer;
  createInfos[index][0].vertexCount = planeMaterialCB.cone.verticesCount; //coneVerticesCount
  createInfos[index][0].vertexOffset = planeMaterialCB.cone.verticesStart * sizeof(Vertex); //coneVerticesStart
  createInfos[index][0].vertexStride = sizeof(Vertex);
  createInfos[index][0].vertexBuffer = boxVertexBuffer;

  for (uint32_t i = 0; i < bottomLevelCount; ++i) {
    bottomLevelBuffers[i] = buildBottomLevel(createInfos[i]);
  }

  WRAPPED_GPU_POINTER bottomLevels[bottomLevelCount];
  for (uint32_t i = 0; i < bottomLevelCount; ++i) {
    bottomLevels[i] = createFallbackWrappedPointer(bottomLevelBuffers[i].accelerationStructure, bottomLevelBuffers[i].resultDataMaxSizeInBytes / sizeof(uint32_t));
  }

  AccelerationStructureBuffers topLevelBuffers = buildTopLevel(bottomLevels, boxBuffer, icosahedronBuffer, coneBuffer);

  hr = commandList->Close();
  throwIfFailed(hr, "Failed to close command list");
  ID3D12CommandList *commandLists[] = {commandList};
  commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

  // Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
  waitForRenderContext();

  // Store the AS buffers. The rest of the buffers will be released once we exit the function.
  for (uint32_t i = 0; i < bottomLevelCount; i++) {
    fallback.bottomLevels[i] = bottomLevelBuffers[i].accelerationStructure;
  }
  fallback.topLevel = topLevelBuffers.accelerationStructure;

  // может эти структуры нужно пересобирать каждый кадр для движущихся объектов?
  // ну то что нужно пересобирать это более менее понятно (что конкретно должно происходить?)
  for (UINT i = 0; i < bottomLevelCount; i++) {
    SAFE_RELEASE(bottomLevelBuffers[i].instanceDesc)
    SAFE_RELEASE(bottomLevelBuffers[i].scratch)
  }
  SAFE_RELEASE(topLevelBuffers.instanceDesc)
  SAFE_RELEASE(topLevelBuffers.scratch)
}

void DX12Render::buildShaderTables() {
  // нам потребуются: 
  // один шейдер генерации
  // 2 шейдера промаха (для обычных лучей и для теневых)
  // и 2 шейдера треугольной геометрии (для обычных лучей и для теневых) 
  // по идее этого достаточно, в примере еще используется процедурная геометрия, но она мне нинужна
  void* rayGenShaderID = nullptr;
  void* missShaderIDs[raysTypeCount] = {nullptr, nullptr};
  void* hitGroupShaderIDs_TriangleGeometry[raysTypeCount] = {nullptr, nullptr};

  static const auto getShaderIDs = [&](auto* stateObjectProperties) {
    rayGenShaderID = stateObjectProperties->GetShaderIdentifier(raygenShaderName2);

    for (uint32_t i = 0; i < raysTypeCount; ++i) {
      missShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(missShaderNames2[i]);
    }

    for (uint32_t i = 0; i < raysTypeCount; ++i) {
      hitGroupShaderIDs_TriangleGeometry[i] = stateObjectProperties->GetShaderIdentifier(hitGroupNames2[i]);
    }

    /*for (uint32_t r = 0; r < IntersectionShaderType::Count; r++) {
      for (uint32_t c = 0; c < RayType::Count; c++) {
        hitGroupShaderIDs_AABBGeometry[r][c] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames_AABBGeometry[r][c]);
      }
    }*/
  };

  // Get shader identifiers.
  uint32_t shaderIdentifierSize2;
  getShaderIDs(fallback.stateObject);
  shaderIdentifierSize2 = fallback.device->GetShaderIdentifierSize();

  // RayGen shader table.
  {
    struct RootArguments {
      PrimitiveConstantBuffer materialCb;
    };

    RootArguments rootArg;
    rootArg.materialCb = planeMaterialCB;

    const uint32_t numShaderRecords = 1;
    const uint32_t shaderRecordSize = shaderIdentifierSize2 + sizeof(RootArguments);
    const uint32_t shaderRecordSizeAligned = align(shaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

    const uint32_t size = numShaderRecords * shaderRecordSizeAligned;
    void* mapped = nullptr;
    fallback.rayGenShaderTable = allocateAndMap(device, size, &mapped, L"RayGenShaderTable");

    uint8_t* copyDest = static_cast<uint8_t*>(mapped);

    // копируем шейдер
    memcpy(copyDest, rayGenShaderID, shaderRecordSize);
    // копируем локальный буфер
    memcpy(copyDest + shaderIdentifierSize2, &rootArg, sizeof(rootArg));
  }

  {
    const uint32_t numShaderRecords = raysTypeCount;
    const uint32_t shaderRecordSize = shaderIdentifierSize2; // No root arguments
    const uint32_t shaderRecordSizeAligned = align(shaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    fallback.missShaderTableStrideInBytes = shaderRecordSizeAligned;

    const uint32_t size = shaderRecordSizeAligned * numShaderRecords;
    void* mapped = nullptr;
    fallback.missShaderTable = allocateAndMap(device, size, &mapped, L"MissShaderTable");

    uint8_t* copyDest = static_cast<uint8_t*>(mapped);
    for (uint32_t i = 0; i < numShaderRecords; ++i) {
      memcpy(copyDest, missShaderIDs[i], shaderRecordSize);
      copyDest += shaderRecordSizeAligned;
    }
  }

  {
    struct RootArguments {
      PrimitiveConstantBuffer materialCb;
    };

    const uint32_t numShaderRecords = raysTypeCount;
    const uint32_t shaderRecordSize = shaderIdentifierSize2 + sizeof(RootArguments);
    const uint32_t shaderRecordSizeAligned = align(shaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    fallback.hitGroupShaderTableStrideInBytes = shaderRecordSizeAligned;
    
    // тут мы можем задать локальные переменные для каждого инстанса
    // не пойму только как сделать все это дело в динамике?
    {
      RootArguments rootArg;
      rootArg.materialCb = planeMaterialCB;

      const uint32_t size = shaderRecordSizeAligned * numShaderRecords;
      void* mapped = nullptr;
      fallback.hitGroupShaderTable = allocateAndMap(device, size, &mapped, L"HitGroupShaderTable");

      uint8_t* copyDest = static_cast<uint8_t*>(mapped);
      for (uint32_t i = 0; i < numShaderRecords; ++i) {
        memcpy(copyDest, hitGroupShaderIDs_TriangleGeometry[i], shaderIdentifierSize2);
        memcpy(copyDest + shaderIdentifierSize2, &rootArg, sizeof(rootArg));
        copyDest += shaderRecordSizeAligned;
      }
    }
  }

  //void* rayGenShaderIdentifier = nullptr;
  //void* missShaderIdentifier = nullptr;
  //void* hitGroupShaderIdentifier = nullptr;

  //static const auto getShaderIdentifiers = [&](auto* stateObjectProperties) {
  //    rayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(raygenShaderName);
  //    missShaderIdentifier = stateObjectProperties->GetShaderIdentifier(missShaderName);
  //    hitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier(hitGroupName);
  //};

  //// Get shader identifiers.
  //uint32_t shaderIdentifierSize;
  //getShaderIdentifiers(fallback.stateObject);
  //shaderIdentifierSize = fallback.device->GetShaderIdentifierSize();

  //// Ray gen shader table
  //{
  //  struct RootArguments {
  //      RayGenConstantBuffer cb;
  //  } rootArguments;
  //  rootArguments.cb = rayGenCB;

  //  /*uint32_t numShaderRecords = 1;
  //  uint32_t shaderRecordSize = shaderIdentifierSize + sizeof(rootArguments);
  //  ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
  //  rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIdentifier, shaderIdentifierSize, &rootArguments, sizeof(rootArguments)));
  //  m_rayGenShaderTable = rayGenShaderTable.GetResource();*/

  //  const uint32_t size = shaderIdentifierSize + sizeof(rootArguments);

  //  void* mapped = nullptr;
  //  fallback.rayGenShaderTable = allocateAndMap(device, size, &mapped, L"RayGenShaderTable");

  //  uint8_t* copyDest = static_cast<uint8_t*>(mapped);
  //  // у нас имеется 2 вещи которые нужно скопировать
  //  memcpy(copyDest, rayGenShaderIdentifier, shaderIdentifierSize);
  //  memcpy(copyDest + shaderIdentifierSize, &rootArguments, sizeof(rootArguments));
  //}

  //// Miss shader table
  //{
  //  const uint32_t size = shaderIdentifierSize;
  //  void* mapped = nullptr;
  //  fallback.missShaderTable = allocateAndMap(device, size, &mapped, L"MissShaderTable");

  //  uint8_t* copyDest = static_cast<uint8_t*>(mapped);
  //  // у нас имеется 1 вещь которую нужно скопировать
  //  memcpy(copyDest, missShaderIdentifier, size);
  //}

  //// Hit group shader table
  //{
  //  const uint32_t size = shaderIdentifierSize;
  //  void* mapped = nullptr;
  //  fallback.hitGroupShaderTable = allocateAndMap(device, size, &mapped, L"MissShaderTable");

  //  uint8_t* copyDest = static_cast<uint8_t*>(mapped);
  //  // у нас имеется 1 вещь которую нужно скопировать
  //  memcpy(copyDest, hitGroupShaderIdentifier, size);
  //}
}

void DX12Render::createRaytracingOutputResource(const uint32_t &width, const uint32_t &height) {
  HRESULT hr;
  //const DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
  const DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;

  // Create the output resource. The dimensions and format should match the swap-chain.
  auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  hr = device->CreateCommittedResource(
    &defaultHeapProperties, 
    D3D12_HEAP_FLAG_NONE, 
    &uavDesc, 
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 
    nullptr, 
    IID_PPV_ARGS(&fallback.raytracingOutput)
  );
  throwIfFailed(hr, "Could not create raytracing output resource");

  fallback.raytracingOutput->SetName(L"Raytracing output resource");

  D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
  fallback.raytracingOutputResourceUAVDescriptorHeapIndex = allocateDescriptor(rtHeap, &uavDescriptorHandle, fallback.raytracingOutputResourceUAVDescriptorHeapIndex);

  D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
  UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(fallback.raytracingOutput, nullptr, &UAVDesc, uavDescriptorHandle);
  fallback.outputResourceDescriptors.gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(rtHeap.handle->GetGPUDescriptorHandleForHeapStart(), fallback.raytracingOutputResourceUAVDescriptorHeapIndex, rtHeap.hardwareSize);
  fallback.outputResourceDescriptors.cpuDescriptorHandle = uavDescriptorHandle;
}

void DX12Render::createDescriptors() {
  HRESULT hr;
  auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

  auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(SceneConstantBuffer));
  hr = device->CreateCommittedResource(
    &uploadHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&fallback.sceneConstantBuffer));
  fallback.sceneConstantBuffer->SetName(L"Scene constant buffer");
  throwIfFailed(hr, "Could not create scene constant buffer");

  // We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay
  hr = fallback.sceneConstantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&sceneConstantBufferPtr));
  throwIfFailed(hr, "Could not map scene constant buffer");

  const uint32_t indexCount = boxIndicesCount + planeIndicesCount + icosahedronIndicesCount + coneIndicesCount;
  const uint32_t vertexCount = boxVerticesCount + planeVerticesCount + icosahedronVerticesCount + coneVerticesCount;

  // Vertex buffer is passed to the shader along with index buffer as a descriptor range.
  uint32_t descriptorIndexIB = createBufferSRV(rtHeap, boxIndexBuffer, &fallback.indexDescs, indexCount, sizeof(uint32_t));
  uint32_t descriptorIndexVB = createBufferSRV(rtHeap, boxVertexBuffer, &fallback.vertexDescs, vertexCount, sizeof(Vertex));
  throwIf(descriptorIndexVB != descriptorIndexIB + 1, "Vertex Buffer descriptor index must follow that of Index Buffer descriptor index");

  /*createTextureUAV(gBuffer.color,  fallback.colorBufferHeapIndex,  fallback.colorBufferDescriptor);
  createTextureUAV(gBuffer.normal, fallback.normalBufferHeapIndex, fallback.normalBufferDescriptor);
  createTextureUAV(gBuffer.depth,  fallback.depthBufferHeapIndex,  fallback.depthBufferDescriptor);*/

  createTextureSRV(rtHeap, gBuffer.color, DXGI_FORMAT_R8G8B8A8_UNORM,  fallback.colorBufferHeapIndex,  fallback.colorBufferDescriptor);
  createTextureSRV(rtHeap, gBuffer.normal, DXGI_FORMAT_R32G32B32A32_FLOAT, fallback.normalBufferHeapIndex, fallback.normalBufferDescriptor);
  createTextureSRV(rtHeap, gBuffer.depth, DXGI_FORMAT_R32_FLOAT,  fallback.depthBufferHeapIndex,  fallback.depthBufferDescriptor);
}

void DX12Render::initializeScene() {
  {
    sceneConstantBufferPtr->projectionToWorld = glm::mat4(1.0f);
    sceneConstantBufferPtr->cameraPosition = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    sceneConstantBufferPtr->lightAmbientColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    float d = 0.6f;
    sceneConstantBufferPtr->lightDiffuseColor = glm::vec4(d, d, d, 1.0f);
    sceneConstantBufferPtr->lightPosition = glm::vec4(0.0f, -18.0f, -20.0f, 0.0f);
    sceneConstantBufferPtr->reflectance = 0.3f;
    sceneConstantBufferPtr->elapsedTime = 0.0f;
  }
}

void DX12Render::createFilterResources(const uint32_t &width, const uint32_t &height) {
  createFilterDescriptorHeap();

  createFilterOutputTexture(width, height);

  createFilterLastFrameData(width, height);

  createFilterConstantBuffer();

  createFilterPSO();
}

void DX12Render::createFilterDescriptorHeap() {
  HRESULT hr;

  // возможно здесь все же придется еще раз создать дескрипторы для основных текстурок
  const uint32_t filterBuffers = 1;                        // const buffer
  const uint32_t filterTextures = 1+2+2;                   // output, color, depth, last color, last depth
  const uint32_t descriptorsCount = filterBuffers + filterTextures;

  const D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    descriptorsCount, // ?
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    0
  };

  hr = device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&filter.heap.handle));
  throwIfFailed(hr, "Could not create descriptor heap for filter");
  filter.heap.handle->SetName(L"descriptor heap for filter");

  filter.heap.hardwareSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  filter.heap.allocatedCount = 0;
}

void DX12Render::createFilterOutputTexture(const uint32_t &width, const uint32_t &height) {
  HRESULT hr;
  
  const DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;

  // Create the output resource. The dimensions and format should match the swap-chain.
  auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  hr = device->CreateCommittedResource(
    &defaultHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &uavDesc,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    nullptr,
    IID_PPV_ARGS(&filter.filterOutput)
  );
  throwIfFailed(hr, "Could not create raytracing output resource");

  filter.filterOutput->SetName(L"Filter output resource");

  D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
  filter.filterOutputUAVDescIndex = allocateDescriptor(filter.heap, &uavDescriptorHandle, filter.filterOutputUAVDescIndex);

  D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
  UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(filter.filterOutput, nullptr, &UAVDesc, uavDescriptorHandle);
  filter.filterOutputUAVDesc = CD3DX12_GPU_DESCRIPTOR_HANDLE(filter.heap.handle->GetGPUDescriptorHandleForHeapStart(), filter.filterOutputUAVDescIndex, filter.heap.hardwareSize);
}

void DX12Render::createFilterLastFrameData(const uint32_t &width, const uint32_t &height) {
  // создадим для начала для текущих текстурок

  //createTextureSRV(filter.heap, gBuffer.color, DXGI_FORMAT_R8G8B8A8_UNORM, filter.colorBufferHeapIndex, filter.colorBufferDescriptor);
  createTextureSRV(filter.heap, fallback.raytracingOutput, DXGI_FORMAT_R8G8B8A8_UNORM, filter.colorBufferHeapIndex, filter.colorBufferDescriptor);
  createTextureSRV(filter.heap, gBuffer.depth, DXGI_FORMAT_R32_FLOAT, filter.depthBufferHeapIndex, filter.depthBufferDescriptor);

  throwIf(filter.colorBufferHeapIndex + 1 != filter.depthBufferHeapIndex, "bad heap index");

  HRESULT hr;

  // color buffer

  DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;

  // Create the output resource. The dimensions and format should match the swap-chain.
  auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

  auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
  hr = device->CreateCommittedResource(
    &defaultHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &uavDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&filter.colorLast)
  );
  throwIfFailed(hr, "Could not create filter color resource");

  filter.colorLast->SetName(L"Filter last color resource");

  /*D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
  filter.filterOutputUAVDescIndex = allocateDescriptor(filter.heap, &uavDescriptorHandle, filter.filterOutputUAVDescIndex);

  D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
  UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(filter.filterOutput, nullptr, &UAVDesc, uavDescriptorHandle);
  filter.filterOutputUAVDesc = CD3DX12_GPU_DESCRIPTOR_HANDLE(filter.heap.handle->GetGPUDescriptorHandleForHeapStart(), filter.filterOutputUAVDescIndex, filter.heap.hardwareSize);*/

  createTextureSRV(filter.heap, filter.colorLast, DXGI_FORMAT_R8G8B8A8_UNORM, filter.lastFrameColorHeapIndex, filter.lastFrameColorDescriptor);
  throwIf(filter.depthBufferHeapIndex + 1 != filter.lastFrameColorHeapIndex, "bad heap index");

  // depth buffer

  format = DXGI_FORMAT_D32_FLOAT;

  // Create the output resource. The dimensions and format should match the swap-chain.
  uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

  hr = device->CreateCommittedResource(
    &defaultHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &uavDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&filter.depthLast)
  );
  throwIfFailed(hr, "Could not create filter depth resource");

  filter.depthLast->SetName(L"Filter last depth resource");

  /*uavDescriptorHandle = {0};
  filter.filterOutputUAVDescIndex = allocateDescriptor(filter.heap, &uavDescriptorHandle, filter.filterOutputUAVDescIndex);

  UAVDesc = {};
  UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(filter.filterOutput, nullptr, &UAVDesc, uavDescriptorHandle);
  filter.filterOutputUAVDesc = CD3DX12_GPU_DESCRIPTOR_HANDLE(filter.heap.handle->GetGPUDescriptorHandleForHeapStart(), filter.filterOutputUAVDescIndex, filter.heap.hardwareSize);*/

  createTextureSRV(filter.heap, filter.depthLast, DXGI_FORMAT_R32_FLOAT, filter.lastFrameDepthHeapIndex, filter.lastFrameDepthDescriptor);
  throwIf(filter.lastFrameColorHeapIndex + 1 != filter.lastFrameDepthHeapIndex, "bad heap index");
}

void DX12Render::createFilterConstantBuffer() {
  HRESULT hr;
  auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

  auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(FilterConstantData));
  hr = device->CreateCommittedResource(
    &uploadHeapProperties,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&filter.constantBuffer));
  throwIfFailed(hr, "Could not create scene constant buffer");

  filter.constantBuffer->SetName(L"Scene constant buffer");

  // We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay
  hr = filter.constantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&filterConstantDataPtr));
  throwIfFailed(hr, "Could not map scene constant buffer");
}

void DX12Render::createFilterPSO() {
  HRESULT hr;

  // create root signature

  CD3DX12_DESCRIPTOR_RANGE ranges[2];
  ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
  ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0);
  //ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);

  // create a descriptor table
  //const D3D12_ROOT_DESCRIPTOR_TABLE descriptorTable{
  //  _countof(ranges), // we only have one range
  //  ranges            // the pointer to the beginning of our ranges array
  //};

  // create a root parameter and fill it out
  //const D3D12_ROOT_PARAMETER rootParameters[] = {
  //  { // only one parameter right now
  //    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, // this is a descriptor table
  //    descriptorTable,                            // this is our descriptor table for this root parameter
  //    D3D12_SHADER_VISIBILITY_ALL                 // our pixel shader will be the only shader accessing this parameter for now
  //  }
  //};

  // тут по идее можно по другому раскидать
  CD3DX12_ROOT_PARAMETER rootParameters[3];
  rootParameters[0].InitAsDescriptorTable(1, &ranges[0]);
  rootParameters[1].InitAsDescriptorTable(1, &ranges[1]);
  //rootParameters[2].InitAsDescriptorTable(1, &ranges[2]);
  rootParameters[2].InitAsConstantBufferView(0, 0);

  CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
  rootSignatureDesc.Init(
    _countof(rootParameters),
    rootParameters,
    0,
    nullptr,
    D3D12_ROOT_SIGNATURE_FLAG_NONE
  );

  ID3DBlob* signature;
  hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
  throwIfFailed(hr, "Failed to serialize root signature");

  if (filter.rootSignature == nullptr) {
    hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&filter.rootSignature));
    throwIfFailed(hr, "Failed to create root signature");
  }

#if defined(_DEBUG)
  // Enable better shader debugging with the graphics debugging tools.
  UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compileFlags = 0;
#endif

  // compile vertex shader
  ID3DBlob* computeShader; // d3d blob for holding vertex shader bytecode
  ID3DBlob* errorBuff; // a buffer holding the error data if any
  hr = D3DCompileFromFile(
    L"reproj.hlsl",
    nullptr,
    nullptr,
    "main",
    "cs_5_0",
    compileFlags,
    0,
    &computeShader,
    &errorBuff
  );

  if (FAILED(hr)) {
    OutputDebugStringA((char*)errorBuff->GetBufferPointer());
    throw std::runtime_error("Compute shader creation error");
  }

  // fill out a shader bytecode structure, which is basically just a pointer
  // to the shader bytecode and the size of the shader bytecode
  const D3D12_SHADER_BYTECODE computeShaderBytecode = {
    computeShader->GetBufferPointer(),
    computeShader->GetBufferSize()
  };

  const D3D12_COMPUTE_PIPELINE_STATE_DESC desc{
    filter.rootSignature,// рут сигнатура
    computeShaderBytecode,// шейдер
    0,
    {
      nullptr,
      0
    },
    D3D12_PIPELINE_STATE_FLAG_NONE
  };

  hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&filter.pso));
  throwIfFailed(hr, "Failed to create compute shader");

  //SAFE_RELEASE(computeShader)
  //SAFE_RELEASE(errorBuff)
}

//void DX12Render::buildGeometryDesc(std::array<std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>, bottomLevelCount> &descs) {
//  // Mark the geometry as opaque. 
//  // PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
//  // Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
//  D3D12_RAYTRACING_GEOMETRY_FLAGS geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
//  descs[0].resize(GEOMETRY_TYPE_COUNT);
//
//  // описание бокса (я не понимаю как сделать одну геометрию на много объектов)
//  //if (static_cast<uint32_t>(GeometryType::AABB) < bottomLevelCount) 
//  {
//    //descs[static_cast<uint32_t>(GeometryType::AABB)].resize(1);
//    const D3D12_RAYTRACING_GEOMETRY_DESC boxDesc{
//      D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
//      geometryFlags,
//      {
//        0,
//        DXGI_FORMAT_R32_UINT,
//        DXGI_FORMAT_R32G32B32_FLOAT,
//        boxIndicesCount,
//        boxVerticesCount,
//        boxIndexBuffer->GetGPUVirtualAddress() + boxIndicesStart * sizeof(uint32_t),
//        {
//          boxVertexBuffer->GetGPUVirtualAddress() + boxVerticesStart * sizeof(Vertex),
//          sizeof(Vertex)
//        }
//      }
//    };
//
//    descs[0][static_cast<uint32_t>(GeometryType::AABB)] = boxDesc;
//  }
//
//  // геометрия плоскости
//  //if (static_cast<uint32_t>(GeometryType::PLANE) < bottomLevelCount) 
//  {
//    //descs[static_cast<uint32_t>(GeometryType::PLANE)].resize(1);
//    const D3D12_RAYTRACING_GEOMETRY_DESC planeDesc{
//      D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
//      geometryFlags,
//      {
//        0,
//        DXGI_FORMAT_R32_UINT,
//        DXGI_FORMAT_R32G32B32_FLOAT,
//        planeIndicesCount,
//        planeVerticesCount,
//        boxIndexBuffer->GetGPUVirtualAddress() + planeIndicesStart * sizeof(uint32_t),
//        {
//          boxVertexBuffer->GetGPUVirtualAddress() + planeVerticesStart * sizeof(Vertex),
//          sizeof(Vertex)
//        }
//      }
//    };
//
//    descs[0][static_cast<uint32_t>(GeometryType::PLANE)] = planeDesc;
//  }
//
//
//  //if (static_cast<uint32_t>(GeometryType::ICOSAHEDRON) < bottomLevelCount)
//  {
//    //descs[static_cast<uint32_t>(GeometryType::ICOSAHEDRON)].resize(1);
//    const D3D12_RAYTRACING_GEOMETRY_DESC icosahedronDesc{
//      D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
//      geometryFlags,
//      {
//        0,
//        DXGI_FORMAT_R32_UINT,
//        DXGI_FORMAT_R32G32B32_FLOAT,
//        icosahedronIndicesCount,
//        icosahedronVerticesCount,
//        boxIndexBuffer->GetGPUVirtualAddress() + icosahedronIndicesStart * sizeof(uint32_t),
//        {
//          boxVertexBuffer->GetGPUVirtualAddress() + icosahedronVerticesStart * sizeof(Vertex),
//          sizeof(Vertex)
//        }
//      }
//    };
//
//    descs[0][static_cast<uint32_t>(GeometryType::ICOSAHEDRON)] = icosahedronDesc;
//  }
//
//  //if (static_cast<uint32_t>(GeometryType::CONE) < bottomLevelCount) 
//  {
//    //descs[static_cast<uint32_t>(GeometryType::CONE)].resize(1);
//    const D3D12_RAYTRACING_GEOMETRY_DESC coneDesc{
//      D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
//      geometryFlags,
//      {
//        0,
//        DXGI_FORMAT_R32_UINT,
//        DXGI_FORMAT_R32G32B32_FLOAT,
//        coneIndicesCount,
//        coneVerticesCount,
//        boxIndexBuffer->GetGPUVirtualAddress() + coneIndicesStart * sizeof(uint32_t),
//        {
//          boxVertexBuffer->GetGPUVirtualAddress() + coneVerticesStart * sizeof(Vertex),
//          sizeof(Vertex)
//        }
//      }
//    };
//
//    descs[0][static_cast<uint32_t>(GeometryType::CONE)] = coneDesc;
//  }
//}

//AccelerationStructureBuffers DX12Render::buildBottomLevel(const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> &geometryDescs, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags) {
//  ID3D12Resource* scratch = nullptr;
//  ID3D12Resource* bottomLevelAS = nullptr;
//
//  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomInputs{
//    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
//    buildFlags,
//    static_cast<uint32_t>(geometryDescs.size()),
//    D3D12_ELEMENTS_LAYOUT_ARRAY,
//    0
//  };
//  bottomInputs.pGeometryDescs = geometryDescs.data();
//
//  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
//  fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&bottomInputs, &bottomLevelPrebuildInfo);
//
//  throwIf(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0, "ResultDataMaxSizeInBytes == 0");
//
//  allocateUAVBuffer(device, bottomLevelPrebuildInfo.ScratchDataSizeInBytes, &scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");
//
//  {
//    D3D12_RESOURCE_STATES initialResourceState;
//    initialResourceState = fallback.device->GetAccelerationStructureResourceState();
//
//    allocateUAVBuffer(device, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, &bottomLevelAS, initialResourceState, L"BottomLevelAccelerationStructure");
//  }
//
//  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc{
//    bottomLevelAS->GetGPUVirtualAddress(),
//    bottomInputs,
//    0,
//    scratch->GetGPUVirtualAddress()
//  };
//
//  // Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
//  ID3D12DescriptorHeap *pDescriptorHeaps[] = {rtHeap};
//  fallback.commandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);
//  fallback.commandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
//
//  AccelerationStructureBuffers buffers{
//    scratch,
//    bottomLevelAS,
//    nullptr,
//    bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes
//  };
//
//  return buffers;
//}

AccelerationStructureBuffers DX12Render::buildBottomLevel(const std::vector<BottomASCreateInfo> &infos) {
  ID3D12Resource* scratch = nullptr;
  ID3D12Resource* bottomLevelAS = nullptr;

  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDesc(infos.size());

  for (uint32_t i = 0; i < geomDesc.size(); ++i) {
    geomDesc[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomDesc[i].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDesc[i].Triangles.IndexCount = infos[i].indexCount;
    geomDesc[i].Triangles.IndexBuffer = infos[i].indexBuffer->GetGPUVirtualAddress() + infos[i].indexOffset;
    geomDesc[i].Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomDesc[i].Triangles.VertexCount = infos[i].vertexCount;
    geomDesc[i].Triangles.VertexBuffer.StartAddress = infos[i].vertexBuffer->GetGPUVirtualAddress() + infos[i].vertexOffset;
    geomDesc[i].Triangles.VertexBuffer.StrideInBytes = infos[i].vertexStride;
    geomDesc[i].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomDesc[i].Triangles.Transform3x4 = 0;
  }

  // Get the size requirements for the scratch and AS buffers
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
  inputs.NumDescs = geomDesc.size();
  inputs.pGeometryDescs = geomDesc.data();
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
  fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

  D3D12_RESOURCE_STATES initialResourceState;
  initialResourceState = fallback.device->GetAccelerationStructureResourceState();

  // Create the buffers. They need to support UAV, and since we are going to immediately use them, we create them with an unordered-access state
  allocateUAVBuffer(device, info.ScratchDataSizeInBytes, &scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");
  allocateUAVBuffer(device, info.ResultDataMaxSizeInBytes, &bottomLevelAS, initialResourceState, L"BottomLevelAccelerationStructure");

  // Create the bottom-level AS
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
  asDesc.Inputs = inputs;
  asDesc.DestAccelerationStructureData = bottomLevelAS->GetGPUVirtualAddress();
  asDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

  fallback.commandList->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

  // We need to insert a UAV barrier before using the acceleration structures in a raytracing operation
  D3D12_RESOURCE_BARRIER uavBarrier = {};
  uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uavBarrier.UAV.pResource = bottomLevelAS;
  commandList->ResourceBarrier(1, &uavBarrier);

  AccelerationStructureBuffers buffers{
    scratch,
    bottomLevelAS,
    nullptr,
    info.ResultDataMaxSizeInBytes
  };

  return buffers;
}

void createResourceForInstance(ID3D12Device* device, const std::vector<WRAPPED_GPU_POINTER> &bottomAddresses, ID3D12Resource** res) {
  const uint32_t objectsTypeCount = 4;
  const uint32_t planeCount = 1;
  const uint32_t aabbCount = 20;
  const uint32_t icosahedronCount = 20;
  const uint32_t coneCount = 20;
  const uint32_t objectsCount = planeCount + aabbCount + icosahedronCount + coneCount;

  /*const float trans[3][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                             {0.0f, 1.0f, 0.0f, 0.0f},
                             {0.0f, 0.0f, 1.0f, 0.0f}};*/

  const glm::mat3x4 matrix = glm::mat3x4(1.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 1.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 1.0f, 0.0f);

  uint32_t inst = 0;
  std::vector<D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC> descs(objectsCount);
  if (static_cast<uint32_t>(GeometryType::PLANE) < bottomLevelCount) {
    for (uint32_t i = 0; i < planeCount; ++i) {
      //memcpy(reinterpret_cast<glm::mat3x4*>(descs[i].Transform), &matrix, sizeof(glm::mat3x4));

      descs[i].Transform[0][0] = 1.0f;
      descs[i].Transform[1][1] = 1.0f;
      descs[i].Transform[2][2] = 1.0f;
      descs[i].InstanceID = inst;
      descs[i].InstanceMask = 1;
      descs[i].InstanceContributionToHitGroupIndex = i;
      descs[i].Flags = 0;
      //descs[i].AccelerationStructure = bottomAddresses[static_cast<uint32_t>(GeometryType::PLANE)];
      descs[i].AccelerationStructure = bottomAddresses[i];

      ++inst;
    }
  }

  if (static_cast<uint32_t>(GeometryType::AABB) < bottomLevelCount) {
    for (uint32_t i = 0; i < aabbCount; ++i) {
      memcpy(reinterpret_cast<glm::mat3x4*>(descs[i].Transform), &matrix, sizeof(glm::mat3x4));

      descs[i].InstanceID = i;
      descs[i].InstanceMask = 1;
      descs[i].InstanceContributionToHitGroupIndex = 0;
      descs[i].Flags = 0;
      descs[i].AccelerationStructure = bottomAddresses[static_cast<uint32_t>(GeometryType::AABB)];
    }
  }

  if (static_cast<uint32_t>(GeometryType::ICOSAHEDRON) < bottomLevelCount) {
    for (uint32_t i = 0; i < icosahedronCount; ++i) {
      memcpy(reinterpret_cast<glm::mat3x4*>(descs[i].Transform), &matrix, sizeof(glm::mat3x4));

      descs[i].InstanceID = i;
      descs[i].InstanceMask = 1;
      descs[i].InstanceContributionToHitGroupIndex = 0;
      descs[i].Flags = 0;
      descs[i].AccelerationStructure = bottomAddresses[static_cast<uint32_t>(GeometryType::ICOSAHEDRON)];
    }
  }

  if (static_cast<uint32_t>(GeometryType::CONE) < bottomLevelCount) {
    for (uint32_t i = 0; i < coneCount; ++i) {
      memcpy(reinterpret_cast<glm::mat3x4*>(descs[i].Transform), &matrix, sizeof(glm::mat3x4));

      descs[i].InstanceID = i;
      descs[i].InstanceMask = 1;
      descs[i].InstanceContributionToHitGroupIndex = 0;
      descs[i].Flags = 0;
      descs[i].AccelerationStructure = bottomAddresses[static_cast<uint32_t>(GeometryType::CONE)];
    }
  }

  const uint64_t bufferSize = descs.size() * sizeof(descs[0]);
  allocateUploadBuffer(device, descs.data(), bufferSize, res, L"InstanceDescs");
}

//AccelerationStructureBuffers DX12Render::buildTopLevel(const uint32_t &count, AccelerationStructureBuffers* bottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags) {
//  ID3D12Resource* scratch = nullptr;
//  ID3D12Resource* topLevelAS = nullptr;
//
//  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topInputs{
//    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
//    buildFlags,
//    count,
//    D3D12_ELEMENTS_LAYOUT_ARRAY,
//    0
//  };
//
//  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
//  fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&topInputs, &topLevelPrebuildInfo);
//
//  throwIf(topLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0, "ResultDataMaxSizeInBytes == 0");
//
//  allocateUAVBuffer(device, topLevelPrebuildInfo.ScratchDataSizeInBytes, &scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");
//
//  {
//    D3D12_RESOURCE_STATES initialResourceState;
//    initialResourceState = fallback.device->GetAccelerationStructureResourceState();
//
//    allocateUAVBuffer(device, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &topLevelAS, initialResourceState, L"TopLevelAccelerationStructure");
//  }
//
//  ID3D12Resource* instanceDescsResource = nullptr;
//  std::vector<D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC> instanceDescs(count);
//  std::vector<WRAPPED_GPU_POINTER> bottomLevelASaddresses(count);
//  for (uint32_t i = 0; i < count; ++i) {
//    bottomLevelASaddresses[i] = createFallbackWrappedPointer(bottomLevelAS[i].accelerationStructure, bottomLevelAS[i].resultDataMaxSizeInBytes / sizeof(uint32_t));
//  }
//
//  // ничего не понимаю на самом деле. Зачем это нужно? Изменять местоположение объекта через это?
//  createResourceForInstance(device, bottomLevelASaddresses, &instanceDescsResource);
//
//  uint32_t numBufferElements = static_cast<uint32_t>(topLevelPrebuildInfo.ResultDataMaxSizeInBytes) / sizeof(uint32_t);
//  fallback.topLevelAccelerationStructurePointer = createFallbackWrappedPointer(topLevelAS, numBufferElements);
//
//  topInputs.InstanceDescs = instanceDescsResource->GetGPUVirtualAddress();
//  const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc{
//    topLevelAS->GetGPUVirtualAddress(),
//    topInputs,
//    0,
//    scratch->GetGPUVirtualAddress()
//  };
//
//  // Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
//  ID3D12DescriptorHeap *pDescriptorHeaps[] = {rtHeap};
//  fallback.commandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);
//  fallback.commandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
//
//  AccelerationStructureBuffers buffers{
//    scratch,
//    topLevelAS,
//    instanceDescsResource,
//    topLevelPrebuildInfo.ResultDataMaxSizeInBytes
//  };
//
//  return buffers;
//}

void buildInstanceDescs(const uint32_t instID, const GPUBuffer<ComputeData> &buffer, const uint32_t &offset, const WRAPPED_GPU_POINTER &ASptr, std::vector<D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC> &descs) {
  for (uint32_t i = 0; i < buffer.size(); ++i) {
    //if (instId >= objCount) break;

    const glm::mat4 m = glm::transpose(buffer[i].currentOrn); // GLM is column major, the INSTANCE_DESC is row major
    /*glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 0.0f));
    m = glm::scale(m, glm::vec3(1.0f, 1.0f, 1.0f));
    m = glm::transpose(m);*/
    
    memcpy(descs[offset + i].Transform, &m, sizeof(descs[offset + i].Transform));

    //descs[i].InstanceID = instId; // This value will be exposed to the shader via InstanceID()
    descs[offset + i].InstanceID = instID;
    descs[offset + i].InstanceContributionToHitGroupIndex = 0;  // This is the offset inside the shader-table. Since we have unique constant-buffer for each instance, we need a different offset
    descs[offset + i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    descs[offset + i].InstanceMask = 1;

    descs[offset + i].AccelerationStructure = ASptr;

    //++instId;
  }
}

AccelerationStructureBuffers DX12Render::buildTopLevel(WRAPPED_GPU_POINTER bottomLevels[bottomLevelCount], const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer) {
  ID3D12Resource* scratch = nullptr;
  ID3D12Resource* topLevelAS = nullptr;
  const uint32_t objCount = 1 + boxBuffer.size() + icosahedronBuffer.size() + coneBuffer.size();

  // First, get the size of the TLAS buffers and create them
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
  inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
  inputs.NumDescs = objCount;
  inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
  fallback.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

  D3D12_RESOURCE_STATES initialResourceState;
  initialResourceState = fallback.device->GetAccelerationStructureResourceState();

  // Create the buffers. They need to support UAV, and since we are going to immediately use them, we create them with an unordered-access state
  allocateUAVBuffer(device, info.ScratchDataSizeInBytes, &scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");
  allocateUAVBuffer(device, info.ResultDataMaxSizeInBytes, &topLevelAS, initialResourceState, L"BottomLevelAccelerationStructure");

  // The instance desc should be inside a buffer, create and map the buffer
  std::vector<D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC> descs(objCount);
  
  const uint32_t planeCount = 1;
  const uint32_t aabbCount = boxBuffer.size();
  const uint32_t icosahedronCount = icosahedronBuffer.size();
  const uint32_t coneCount = coneBuffer.size();

  const glm::mat4 transformation = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f));

  uint32_t offset = 0;
  uint32_t instId = 0;
  for (uint32_t i = 0; i < planeCount; ++i) {
    //if (instId >= objCount) break;

    const glm::mat4 m = glm::transpose(transformation); // GLM is column major, the INSTANCE_DESC is row major
    memcpy(descs[offset + i].Transform, &m, sizeof(descs[offset + i].Transform));

    //descs[i].InstanceID = instId; // This value will be exposed to the shader via InstanceID()
    descs[offset + i].InstanceID = PLANE_ID;
    descs[offset + i].InstanceContributionToHitGroupIndex = 0;  // This is the offset inside the shader-table. Since we have unique constant-buffer for each instance, we need a different offset
    descs[offset + i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    descs[offset + i].InstanceMask = 1;

    descs[offset + i].AccelerationStructure = bottomLevels[0];

    ++instId;
  }

  offset += planeCount;
  buildInstanceDescs(BOX_ID, boxBuffer, offset, bottomLevels[1], descs);

  offset += boxBuffer.size();
  buildInstanceDescs(ICOSAHEDRON_ID, icosahedronBuffer, offset, bottomLevels[2], descs);

  offset += icosahedronBuffer.size();
  buildInstanceDescs(CONE_ID, coneBuffer, offset, bottomLevels[3], descs);

  //for (uint32_t i = 0; i < aabbCount; ++i) {
  //  //if (instId >= objCount) break;

  //  const glm::mat4 m = glm::transpose(boxBuffer[i].currentOrn); // GLM is column major, the INSTANCE_DESC is row major
  //  //const glm::mat4 m = boxBuffer[i].currentOrn;
  //  memcpy(descs[i].Transform, &m, sizeof(descs[i].Transform));

  //  descs[i].InstanceID = BOX_ID; // This value will be exposed to the shader via InstanceID()
  //  descs[i].InstanceContributionToHitGroupIndex = 0;  // This is the offset inside the shader-table. Since we have unique constant-buffer for each instance, we need a different offset
  //  descs[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
  //  descs[i].InstanceMask = 1;

  //  descs[i].AccelerationStructure = bottomLevels[1];

  //  ++instId;
  //}

  //for (uint32_t i = 0; i < icosahedronCount; ++i) {
  //  //if (instId >= objCount) break;

  //  const glm::mat4 m = glm::transpose(icosahedronBuffer[i].currentOrn); // GLM is column major, the INSTANCE_DESC is row major
  //  //const glm::mat4 m = icosahedronBuffer[i].currentOrn;
  //  memcpy(descs[i].Transform, &m, sizeof(descs[i].Transform));

  //  descs[i].InstanceID = ICOSAHEDRON_ID; // This value will be exposed to the shader via InstanceID()
  //  descs[i].InstanceContributionToHitGroupIndex = 0;  // This is the offset inside the shader-table. Since we have unique constant-buffer for each instance, we need a different offset
  //  descs[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
  //  descs[i].InstanceMask = 1;

  //  descs[i].AccelerationStructure = bottomLevels[2];

  //  ++instId;
  //}

  //for (uint32_t i = 0; i < coneCount; ++i) {
  //  //if (instId >= objCount) break;

  //  const glm::mat4 m = glm::transpose(coneBuffer[i].currentOrn); // GLM is column major, the INSTANCE_DESC is row major
  //  //const glm::mat4 m = coneBuffer[i].currentOrn;
  //  memcpy(descs[i].Transform, &m, sizeof(descs[i].Transform));

  //  descs[i].InstanceID = CONE_ID; // This value will be exposed to the shader via InstanceID()
  //  descs[i].InstanceContributionToHitGroupIndex = 0;  // This is the offset inside the shader-table. Since we have unique constant-buffer for each instance, we need a different offset
  //  descs[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
  //  descs[i].InstanceMask = 1;

  //  descs[i].AccelerationStructure = bottomLevels[3];

  //  ++instId;
  //}

  ID3D12Resource* instDesc = nullptr;
  allocateUploadBuffer(device, descs.data(), sizeof(descs[0])*descs.size(), &instDesc, L"InstanceDescs");

  uint32_t numBufferElements = static_cast<uint32_t>(info.ResultDataMaxSizeInBytes) / sizeof(uint32_t);
  fallback.topLevelAccelerationStructurePointer = createFallbackWrappedPointer(topLevelAS, numBufferElements);

  // Create the TLAS
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
  asDesc.Inputs = inputs;
  asDesc.Inputs.InstanceDescs = instDesc->GetGPUVirtualAddress();
  asDesc.DestAccelerationStructureData = topLevelAS->GetGPUVirtualAddress();
  asDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

  // Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
  ID3D12DescriptorHeap *pDescriptorHeaps[] = {rtHeap.handle};
  fallback.commandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);
  fallback.commandList->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

  // We need to insert a UAV barrier before using the acceleration structures in a raytracing operation
  D3D12_RESOURCE_BARRIER uavBarrier = {};
  uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uavBarrier.UAV.pResource = topLevelAS;
  commandList->ResourceBarrier(1, &uavBarrier);

  AccelerationStructureBuffers buffers{
    scratch,
    topLevelAS,
    instDesc,
    info.ResultDataMaxSizeInBytes
  };

  return buffers;
}

WRAPPED_GPU_POINTER DX12Render::createFallbackWrappedPointer(ID3D12Resource* resource, const uint32_t &bufferNumElements) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC rawBufferUavDesc = {
    /*DXGI_FORMAT_R32_TYPELESS,
    D3D12_UAV_DIMENSION_BUFFER,
    {
      0,
      bufferNumElements,
      0,
      0,
      D3D12_BUFFER_UAV_FLAG_RAW
    }*/
  };
  rawBufferUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  rawBufferUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  rawBufferUavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
  rawBufferUavDesc.Buffer.NumElements = bufferNumElements;

  D3D12_CPU_DESCRIPTOR_HANDLE bottomLevelDescriptor;

  // Only compute fallback requires a valid descriptor index when creating a wrapped pointer.
  UINT descriptorHeapIndex = 0;
  if (!fallback.device->UsingRaytracingDriver()) {
    descriptorHeapIndex = allocateDescriptor(rtHeap, &bottomLevelDescriptor);
    device->CreateUnorderedAccessView(resource, nullptr, &rawBufferUavDesc, bottomLevelDescriptor);
  }

  return fallback.device->GetWrappedPointerSimple(descriptorHeapIndex, resource->GetGPUVirtualAddress());
}

uint32_t DX12Render::allocateDescriptor(DescriptorHeap &heap, D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, const uint32_t &descriptorIndexToUse) {
  uint32_t index = descriptorIndexToUse;

  auto descriptorHeapCpuBase = heap.handle->GetCPUDescriptorHandleForHeapStart();
  if (descriptorIndexToUse >= heap.handle->GetDesc().NumDescriptors) {
    index = heap.allocatedCount++;
  }

  *cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, index, heap.hardwareSize);
  return index;
}

// Create a SRV for a buffer.
uint32_t DX12Render::createBufferSRV(DescriptorHeap &heap, ID3D12Resource* res, DXDescriptors* buffer, const uint32_t &numElements, const uint32_t &elementSize) {
  // SRV
  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srvDesc.Buffer.NumElements = numElements;
  if (elementSize == 0) {
    srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srvDesc.Buffer.StructureByteStride = 0;
  } else {
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srvDesc.Buffer.StructureByteStride = elementSize;
  }
  UINT descriptorIndex = allocateDescriptor(heap, &buffer->cpuDescriptorHandle);
  device->CreateShaderResourceView(res, &srvDesc, buffer->cpuDescriptorHandle);
  buffer->gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(heap.handle->GetGPUDescriptorHandleForHeapStart(), descriptorIndex, heap.hardwareSize);
  return descriptorIndex;
}

void DX12Render::createTextureUAV(DescriptorHeap &heap, ID3D12Resource* res, uint32_t &index, D3D12_GPU_DESCRIPTOR_HANDLE &handle) {
  D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
  // указать индекс вторым аргументом может быть полезно для пересоздания дескриптора
  index = allocateDescriptor(heap, &uavDescriptorHandle);

  D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
  UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(res, nullptr, &UAVDesc, uavDescriptorHandle);
  handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(heap.handle->GetGPUDescriptorHandleForHeapStart(), index, heap.hardwareSize);
}

void DX12Render::createTextureSRV(DescriptorHeap &heap, ID3D12Resource* res, const DXGI_FORMAT &format, uint32_t &index, D3D12_GPU_DESCRIPTOR_HANDLE &handle) {
  D3D12_CPU_DESCRIPTOR_HANDLE srvDescriptorHandle;
  // указать индекс вторым аргументом может быть полезно для пересоздания дескриптора
  index = allocateDescriptor(heap, &srvDescriptorHandle);

  D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
  SRVDesc.Format = format;
  SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  SRVDesc.Texture2D = {
    0,
    1,
    0,
    0.0f
  };
  SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  device->CreateShaderResourceView(res, &SRVDesc, srvDescriptorHandle);
  handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(heap.handle->GetGPUDescriptorHandleForHeapStart(), index, heap.hardwareSize);
}