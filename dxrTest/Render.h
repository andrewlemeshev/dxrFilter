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
#include "DebugShared.h"
#include "Buffer.h"

#include "imgui/imgui.h"

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

  ID3D12Resource* accelerationStructure;
  ID3D12Resource* bottomLevelAccelerationStructure;
  ID3D12Resource* topLevelAccelerationStructure;

  ID3D12Resource* raytracingOutput;
  ID3D12Resource* shadowOutput;
  //D3D12_GPU_DESCRIPTOR_HANDLE raytracingOutputResourceUAVGpuDescriptor;
  DXDescriptors outputResourceDescriptors;
  uint32_t raytracingOutputResourceUAVDescriptorHeapIndex;
  DXDescriptors shadowDescriptors;
  uint32_t shadowHeapIndex;

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

  ID3D12Resource* bilateralOutput;

  ID3D12Resource* pixelAdditionOutput;

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

  D3D12_GPU_DESCRIPTOR_HANDLE pixelAdditionOutputDesc;
  uint32_t pixelAdditionOutputIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE pixelAdditionInputDesc;
  uint32_t pixelAdditionInputIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE pixelAdditionNormalDesc;
  uint32_t pixelAdditionNormalIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE bilateralOutputUAVDesc;
  uint32_t bilateralOutputUAVDescIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE bilateralInputUAVDesc;
  uint32_t bilateralInputUAVDescIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE bilateralNormalDesc;
  uint32_t bilateralNormalHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE bilateralDepthDesc;
  uint32_t bilateralDepthHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE bilateralPixelDataDesc;
  uint32_t bilateralPixelDataHeapIndex;

  ID3D12Resource* constantBuffer;
  FilterConstantBuffer bilateralBuffer;
  // возможно мне пригодится еще один constant buffer

  DescriptorHeap heap;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;

  ID3D12RootSignature* bilateralRootSignature;
  ID3D12PipelineState* bilateralPSO;

  ID3D12RootSignature* pixelAdditionRootSignature;
  ID3D12PipelineState* pixelAdditionPSO;
  // что-то еще?
};

struct LightningCalculation {
  ID3D12Resource* output;

  D3D12_GPU_DESCRIPTOR_HANDLE outputDescriptor;
  uint32_t outputHeapIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE colorDescriptor;
  uint32_t colorHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE normalDescriptor;
  uint32_t normalHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE depthDescriptor;
  uint32_t depthHeapIndex;
  D3D12_GPU_DESCRIPTOR_HANDLE shadowDescriptor;
  uint32_t shadowHeapIndex;

  DescriptorHeap heap;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;

  ID3D12Resource* sceneConstantBuffer;
};

struct ToneMapping {
  ID3D12Resource* output;

  D3D12_GPU_DESCRIPTOR_HANDLE outputUAVDesc;
  uint32_t outputUAVDescIndex;

  D3D12_GPU_DESCRIPTOR_HANDLE filterDesc;
  uint32_t filterDescIndex;

  DescriptorHeap heap;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;
};

struct Gui {
  ID3D12Resource* font;
  D3D12_GPU_DESCRIPTOR_HANDLE fontDesc;
  uint32_t fontDescIndex;

  ID3D12Resource* vertices;
  ID3D12Resource* indices;
  size_t vertexBufferSize;
  size_t indexBufferSize;
  D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
  D3D12_INDEX_BUFFER_VIEW indexBufferView;

  ID3D12DescriptorHeap* rtvHeap;

  glm::mat4 matrix;

  DescriptorHeap heap;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;
};

struct Profiler {
  ID3D12QueryHeap* queryHeap;
  size_t queryCount;
  size_t currentQuery;
};

struct TimeProfiler : public Profiler {
  ID3D12Resource* buffer;
  uint64_t frequency;

  void timeStamp(ID3D12GraphicsCommandList* cmdList);
};

enum class Visualize : uint32_t {
  color,
  normals,
  depths,
  shadows,
  temporal,
  pixelDatas,
  bilateral,
  lightning,
  count
};

struct DebugVisualizer {
  DescriptorHeap heap;

  D3D12_CPU_DESCRIPTOR_HANDLE texCPUDesc;
  D3D12_GPU_DESCRIPTOR_HANDLE texDesc;
  uint32_t texDescIndex;

  DebugBuffer buffer;

  ID3D12RootSignature* rootSignature;
  ID3D12PipelineState* pso;
};

enum class GlobalRootSignatureParams : uint32_t {
  OUTPUT_VIEW_SLOT = 0,
  ACCELERATION_STRUCTURE_SLOT,
  GBUFFER_TEXTURES,
  SCENE_CONSTANT,
  VERTEX_BUFFERS,
  RANDOM_TEXTURE,
  SHADOW_TEXTURE,
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

  void initImGui();

  void copyRenderDrawData();

  // тут нужно еще создать псо (так чтобы его легко можно было пересоздавать)
  void recreatePSO();
  // еще пересоздать буферы, например инстансный, и обновить view матрицу
  void prepareRender(const uint32_t &instanceCount, const glm::mat4 &viewMatrix);

  void computePartHost(GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer);

  void updateSceneData(const glm::vec4 &cameraPos, const glm::mat4 &viewProj);

  void renderGui(const glm::uvec2 &winPos, const glm::uvec2 &winSize);

  // передобавляем в дескриптор таргет
  void visualize(const Visualize v);

  void nextFrame();
  void computePart();
  void gBufferPart(const uint32_t &boxCount, const uint32_t &icosahedronCount, const uint32_t &coneCount);
  void rayTracingPart();
  void filterPart();
  void debugPart();
  void guiPart();
  void endFrame();

  void cleanup();

  //void waitForFrame();
  //void waitForCurrentFrame();

  void waitForRenderContext();
  void moveToNextFrame();

  ID3D12Device* getDevice() const;
  ID3D12Resource* getTimeStampResource() const;
  size_t getTimeStampCount() const;
  uint64_t getTimeStampFrequency() const;

  FilterConstantBuffer* constantBufferData();
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
  DXGI_FORMAT colorBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  DXGI_FORMAT normalBufferFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
  DXGI_FORMAT depthBufferFormat = DXGI_FORMAT_D32_FLOAT;

  // orientation buffer
  ID3D12Resource* ornBuffer = nullptr;
  D3D12_VERTEX_BUFFER_VIEW ornBufferView;

  //DXGI_FORMAT rayTracingOutputFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  DXGI_FORMAT rayTracingOutputFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
  //DXGI_FORMAT shadowOutputFormat = DXGI_FORMAT_R32_FLOAT;
  DXGI_FORMAT shadowOutputFormat = DXGI_FORMAT_R32G32_FLOAT;
  RayTracing fallback;
  SceneConstantBuffer* sceneConstantBufferPtr = nullptr;
  RayGenConstantBuffer rayGenCB;
  PrimitiveConstantBuffer planeMaterialCB;
  /*ID3D12DescriptorHeap* rtHeap = nullptr;
  uint32_t rtDescriptorsAllocated;
  uint32_t rtDescriptorSize;*/

  ID3D12Resource* uintRandText = nullptr;
  D3D12_GPU_DESCRIPTOR_HANDLE uintRandTextDescriptor;
  uint32_t uintRandTextHeapIndex;

  DescriptorHeap rtHeap;

  //DXGI_FORMAT filterOutputFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  DXGI_FORMAT filterOutputFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
  //DXGI_FORMAT bilateralOutputFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
  DXGI_FORMAT pixelAdditionOutputFormat = DXGI_FORMAT_R32G32_UINT;
  DXGI_FORMAT bilateralOutputFormat = DXGI_FORMAT_R32_FLOAT;
  Filter filter;
  glm::mat4 oldViewProj;
  FilterConstantData* filterConstantDataPtr = nullptr;

  DXGI_FORMAT toneMappingOutputFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  ToneMapping toneMapping;

  DXGI_FORMAT lightningOutputFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
  LightningCalculation lightning;

  Gui gui;

  TimeProfiler perfomance;

  DebugVisualizer visualizer;
  Visualize debugType;

  void resolveQuery();

  void loadModels(std::vector<uint32_t> &indices, std::vector<Vertex> &vertices);

  void deinitPSO();

  void createPerformanceProfiler();

  void createDefferedPSO();

  //void createRTResources(const uint32_t &width, const uint32_t &height);
  void createRayTracingFallbackDevice();
  void createRootSignatures();
  void serializeRootSignature(const D3D12_ROOT_SIGNATURE_DESC &desc, ID3D12RootSignature** rootSig);
  void createRaytracingPSO();
  void createLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* raytracingPipeline);
  void createRTDescriptorHeap();
  //void buildAccelerationStructures();
  void buildAccelerationStructures2(const GPUBuffer<ComputeData> &boxBuffer, const GPUBuffer<ComputeData> &icosahedronBuffer, const GPUBuffer<ComputeData> &coneBuffer);
  void buildShaderTables();
  void createRaytracingOutputResource(const uint32_t &width, const uint32_t &height);
  void createShadowOutputResource(const uint32_t &width, const uint32_t &height);
  void createDescriptors();
  void initializeScene();

  void createRandomUintTexture(const uint32_t &width, const uint32_t &height, ID3D12Resource** upload, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* footprint);

  void createFilterResources(const uint32_t &width, const uint32_t &height);
  void createFilterDescriptorHeap();
  void createFilterOutputTexture(const uint32_t &width, const uint32_t &height);
  void createFilterLastFrameData(const uint32_t &width, const uint32_t &height);
  void createFilterConstantBuffer();
  void createFilterPSO();

  void createPixelAdditionOutputTexture(const uint32_t &width, const uint32_t &height);
  void createPixelAdditionConstantBuffer();
  void createPixelAdditionPSO();

  void createBilateralOutputTexture(const uint32_t &width, const uint32_t &height);
  void createBilateralConstantBuffer();
  void createBilateralPSO();

  void createLightningResources(const uint32_t &width, const uint32_t &height);
  void createLightningDescriptorHeap();
  void createLightningOutputTexture(const uint32_t &width, const uint32_t &height);
  void createLightningConstantBuffer(); // наверное вообще не потребуется
  void createLightningPSO();

  void createToneMappingResources(const uint32_t &width, const uint32_t &height);
  void createToneMappingDescriptorHeap();
  void createToneMappingOutputTexture(const uint32_t &width, const uint32_t &height);
  void createToneMappingPSO();

  void createDebugVisualizerResources();
  void createDebugVisualizerDescriptorHeap();
  void createDebugVisualizePSO();

  void createGuiDescriptorHeap();
  void createFontTexture();
  void createGuiBuffers();
  void createGuiPSO();

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

  struct TextureUAVCreateInfo {
    DXGI_FORMAT format;
    uint32_t width;
    uint32_t height;
    DescriptorHeap* heap;

    ID3D12Resource** texture;
    D3D12_GPU_DESCRIPTOR_HANDLE* descriptor;
    uint32_t* heapIndex;
  };

  // cp - compute pipeline
  struct CPCreateInfo {
    std::wstring computeShaderPath;
    uint32_t rootParametersCount;
    CD3DX12_ROOT_PARAMETER* rootParameters;

    ID3D12RootSignature** rootSignature;
    ID3D12PipelineState** PSO;
  };

  void createUAVTexture(const TextureUAVCreateInfo &info);
  void createDescriptorHeap(const uint32_t &descriptorCount, DescriptorHeap &heap, const std::string &name);
  void createComputeShader(const CPCreateInfo &info);

  void rebindDebugSRV(ID3D12Resource* res, const DXGI_FORMAT &format);
};

#endif
