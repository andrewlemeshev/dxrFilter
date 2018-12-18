#ifndef RENDER_H
#define RENDER_H

#define SAFE_RELEASE(p) if (p) p->Release(); p = nullptr;

#include "stdafx.h"

#include <cstdint>
#include <cstddef>
#include <array>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
//#include <DirectXMath.h>
#include "d3dx12.h"
#include <dxcapi.h>

#include "D3D12RaytracingFallback.h"

#include "Shared.h"
#include "Buffer.h"

class CD3D12_STATE_OBJECT_DESC;

const uint32_t raysTypeCount = 2;
const uint32_t bottomLevelCount = 4;

enum GeometryType : uint32_t {
  PLANE = 0,
  AABB,
  ICOSAHEDRON,
  CONE,
  GEOMETRY_TYPE_COUNT
};

struct AccelerationStructureBuffers {
  ID3D12Resource* scratch;
  ID3D12Resource* accelerationStructure;
  ID3D12Resource* instanceDesc;    // Used only for top-level AS
  uint64_t        resultDataMaxSizeInBytes;
};

struct DXDescriptors {
  D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle;
  D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;
};

struct DescriptorHeap {
  ID3D12DescriptorHeap* handle;
  uint32_t allocatedCount;
  uint32_t hardwareSize;
};

// по всей видимости мне нужно 3 (бокс, икосахедрон, кон) боттом левел акселлератион структуры (а хотя может 4?)
// и одна топ левел? 
struct RayTracing {
  bool forceComputeFallback;

  ID3D12RaytracingFallbackDevice* device;
  ID3D12RaytracingFallbackCommandList* commandList;
  ID3D12RaytracingFallbackStateObject* stateObject;
  WRAPPED_GPU_POINTER topLevelAccelerationStructurePointer;

  ID3D12RootSignature* globalRootSignature;
  ID3D12RootSignature* localRootSignature;

  // это теперь видимо не нужно 
  ID3D12Resource* accelerationStructure;
  ID3D12Resource* bottomLevelAccelerationStructure;
  ID3D12Resource* topLevelAccelerationStructure;

  ID3D12Resource* raytracingOutput;
  //D3D12_GPU_DESCRIPTOR_HANDLE raytracingOutputResourceUAVGpuDescriptor;
  DXDescriptors outputResourceDescriptors;
  uint32_t raytracingOutputResourceUAVDescriptorHeapIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE colorBufferDescriptor;
  uint32_t colorBufferHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE normalBufferDescriptor;
  uint32_t normalBufferHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE depthBufferDescriptor;
  uint32_t depthBufferHeapIndex;

  DXDescriptors vertexDescs;
  DXDescriptors indexDescs;

  ID3D12Resource* missShaderTable;
  ID3D12Resource* hitGroupShaderTable;
  ID3D12Resource* rayGenShaderTable;
  uint32_t missShaderTableStrideInBytes;
  uint32_t hitGroupShaderTableStrideInBytes;

  ID3D12Resource* bottomLevels[bottomLevelCount];
  ID3D12Resource* topLevel;

  ID3D12Resource* sceneConstantBuffer;
};

struct Filter {
  ID3D12Resource* filterOutput;
  ID3D12Resource* colorLast;
  ID3D12Resource* depthLast;

  D3D12_GPU_DESCRIPTOR_HANDLE filterOutputUAVDesc;
  uint32_t filterOutputUAVDescIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE colorBufferDescriptor;
  uint32_t colorBufferHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE depthBufferDescriptor;
  uint32_t depthBufferHeapIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE lastFrameColorDescriptor;
  uint32_t lastFrameColorHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE lastFrameDepthDescriptor;
  uint32_t lastFrameDepthHeapIndex;

  ID3D12Resource* constantBuffer;

  DescriptorHeap heap;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;
  // что-то еще?
};

enum class GlobalRootSignatureParams : uint32_t {
  OUTPUT_VIEW_SLOT = 0,
  ACCELERATION_STRUCTURE_SLOT,
  GBUFFER_TEXTURES,
  SCENE_CONSTANT,
  VERTEX_BUFFERS,
  COUNT
};

enum class LocalRootSignatureParams : uint32_t {
  VIEWPORT_CONSTANT_SLOT = 0,
  COUNT
};

struct BottomASCreateInfo {
  uint32_t indexCount;
  uint64_t indexOffset;
  uint32_t vertexCount;
  uint64_t vertexOffset;
  uint64_t vertexStride;

  ID3D12Resource* indexBuffer;
  ID3D12Resource* vertexBuffer;
};

struct GBuffer {
  // скорее всего потребуется один (два, один для глубины) дескриптор
  ID3D12Resource* color = nullptr;
  ID3D12Resource* normal = nullptr;
  ID3D12Resource* depth = nullptr;

  ID3D12DescriptorHeap* cDescriptorHeap = nullptr;
  ID3D12DescriptorHeap* dDescriptorHeap = nullptr;
  //ID3D12DescriptorHeap* nDescriptorHeap = nullptr;
};

const uint32_t frameBufferCount = 3;

class DX12Render {
public:
  DX12Render();
  ~DX12Render();

  void init(HWND hwnd, const uint32_t &width, const uint32_t &height, bool fullscreen);
  void initRT(const uint32_t &width, const uint32_t &height, const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer);
  void initFilter(const uint32_t &width, const uint32_t &height);

  // тут нужно еще создать псо (так чтобы его легко можно было пересоздавать)
  void recreatePSO();
  // еще пересоздать буферы, например инстансный, и обновить view матрицу
  void prepareRender(const uint32_t &instanceCount, const glm::mat4 &viewMatrix);

  void computePartHost(GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer);

  void updateSceneData(const glm::vec4 &cameraPos, const glm::mat4 &viewProj);

  void nextFrame();
  void computePart();
  void gBufferPart(const uint32_t &boxCount, const uint32_t &icosahedronCount, const uint32_t &coneCount);
  void rayTracingPart();
  void filterPart();
  void endFrame();

  void cleanup();

  //void waitForFrame();
  //void waitForCurrentFrame();

  void waitForRenderContext();
  void moveToNextFrame();

  ID3D12Device* getDevice() const;
private:
  ID3D12Device* device = nullptr;
  IDXGISwapChain3* swapChain = nullptr;
  ID3D12CommandQueue* commandQueue = nullptr;
  ID3D12DescriptorHeap* rtvDescriptorHeap = nullptr;
  ID3D12Resource* renderTargets[frameBufferCount];
  ID3D12CommandAllocator* commandAllocator[frameBufferCount];
  ID3D12GraphicsCommandList* commandList = nullptr;
  ID3D12Fence* fence[frameBufferCount];
  HANDLE fenceEvent;
  uint64_t fenceValue[frameBufferCount];
  uint64_t contextFenceValue;
  uint32_t rtvDescriptorSize;
  uint32_t frameIndex;

  // несколько псо и сигнатур
  ID3D12PipelineState* pipelineStateObject = nullptr;
  ID3D12RootSignature* rootSignature = nullptr;

  // меняться это не будет скорее всего
  D3D12_VIEWPORT viewport;
  D3D12_RECT scissorRect;

  // box data would be here (probably not only box)
  ID3D12Resource* boxVertexBuffer = nullptr;
  D3D12_VERTEX_BUFFER_VIEW boxVertexBufferView;
  ID3D12Resource* boxIndexBuffer = nullptr;
  D3D12_INDEX_BUFFER_VIEW boxIndexBufferView;
  ID3D12Resource* instanceBuffer = nullptr;
  D3D12_VERTEX_BUFFER_VIEW instanceBufferView;
  uint32_t instanceBufferSize = 0;

  glm::mat4* instanceBufferPtr = nullptr;

  ID3D12Resource* constantBuffer = nullptr;
  ID3D12DescriptorHeap* constantBufferDescriptor = nullptr;
  glm::mat4* constantBufferPtr = nullptr;

  ID3D12Resource* depthStencilBuffer = nullptr;
  ID3D12DescriptorHeap* dsDescriptorHeap = nullptr;

  GBuffer gBuffer;

  // orientation buffer
  ID3D12Resource* ornBuffer = nullptr;
  D3D12_VERTEX_BUFFER_VIEW ornBufferView;

  RayTracing fallback;
  SceneConstantBuffer* sceneConstantBufferPtr = nullptr;
  RayGenConstantBuffer rayGenCB;
  PrimitiveConstantBuffer planeMaterialCB;
  /*ID3D12DescriptorHeap* rtHeap = nullptr;
  uint32_t rtDescriptorsAllocated;
  uint32_t rtDescriptorSize;*/

  DescriptorHeap rtHeap;

  Filter filter;
  glm::mat4 oldViewProj;
  FilterConstantData* filterConstantDataPtr = nullptr;

  void loadModels(std::vector<uint32_t> &indices, std::vector<Vertex> &vertices);

  void createRTResources(const uint32_t &width, const uint32_t &height);
  void createRayTracingFallbackDevice();
  void createRootSignatures();
  void serializeRootSignature(const D3D12_ROOT_SIGNATURE_DESC &desc, ID3D12RootSignature** rootSig);
  void createRaytracingPSO();
  void createLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* raytracingPipeline);
  void createRTDescriptorHeap();
  void buildAccelerationStructures();
  void buildAccelerationStructures2(const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer);
  void buildShaderTables();
  void createRaytracingOutputResource(const uint32_t &width, const uint32_t &height);
  void createDescriptors();
  void initializeScene();

  void createFilterResources(const uint32_t &width, const uint32_t &height);
  void createFilterDescriptorHeap();
  void createFilterOutputTexture(const uint32_t &width, const uint32_t &height);
  void createFilterLastFrameData(const uint32_t &width, const uint32_t &height);
  void createFilterConstantBuffer();
  void createFilterPSO();

  //void buildGeometryDesc(std::array<std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>, bottomLevelCount> &descs);
  //AccelerationStructureBuffers buildBottomLevel(const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> &geometryDescs, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE);
  AccelerationStructureBuffers buildBottomLevel(const std::vector<BottomASCreateInfo> &infos);
  //AccelerationStructureBuffers buildTopLevel(const uint32_t &count, AccelerationStructureBuffers* bottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE);
  AccelerationStructureBuffers buildTopLevel(WRAPPED_GPU_POINTER bottomLevels[bottomLevelCount], const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer);

  WRAPPED_GPU_POINTER createFallbackWrappedPointer(ID3D12Resource* resource, const uint32_t &bufferNumElements);
  uint32_t allocateDescriptor(DescriptorHeap &heap, D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, const uint32_t &descriptorIndexToUse = UINT32_MAX);
  uint32_t createBufferSRV(DescriptorHeap &heap, ID3D12Resource* res, DXDescriptors* buffer, const uint32_t &numElements, const uint32_t &elementSize);
  void createTextureUAV(DescriptorHeap &heap, ID3D12Resource* res, uint32_t &index, D3D12_GPU_DESCRIPTOR_HANDLE &handle);
  void createTextureSRV(DescriptorHeap &heap, ID3D12Resource* res, const DXGI_FORMAT &format, uint32_t &index, D3D12_GPU_DESCRIPTOR_HANDLE &handle);
};

#endif
