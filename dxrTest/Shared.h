#ifndef SHARED_H
#define SHARED_H

#ifdef __CPLUSPLUS
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/transform.hpp>

typedef glm::vec2 float2;
typedef glm::vec3 float3;
typedef glm::vec4 float4;
typedef glm::mat4 float4x4;
typedef uint32_t uint;

#define column_major 
#endif

#define MAX_RAY_RECURSION_DEPTH 3

#define PLANE_ID 0
#define BOX_ID 1
#define ICOSAHEDRON_ID 2
#define CONE_ID 3

// hardcoded
#define PLANE_INDEX_START 36
#define PLANE_VERTEX_START 24
#define BOX_INDEX_START 0
#define BOX_VERTEX_START 0
#define ICOSAHEDRON_INDEX_START 42
#define ICOSAHEDRON_VERTEX_START 28
#define CONE_INDEX_START 102
#define CONE_VERTEX_START 88

#define EPSILON 0.001f

struct Viewport {
  float left;
  float top;
  float right;
  float bottom;
};

struct RayGenConstantBuffer {
  Viewport viewport;
  Viewport stencil;
};

struct RayPayload {
  float4 color;
  uint   recursionDepth;
};

struct ShadowRayPayload {
  bool hit;
};

struct SceneConstantBuffer {
  column_major float4x4 projectionToWorld;
  float4   cameraPosition;
  float4   lightPosition;
  float4   lightAmbientColor;
  float4   lightDiffuseColor;
  float    reflectance;
  float    elapsedTime;                 // Elapsed application time.
};

struct ModelData {
  uint indicesStart;
  uint indicesCount;
  uint indicesSize;

  uint verticesStart;
  uint verticesCount;
  uint verticesSize;
};

// Attributes per primitive type.
struct PrimitiveConstantBuffer {
  float4 albedo;
  float reflectanceCoef;
  float diffuseCoef;
  float specularCoef;
  float specularPower;
  float stepScale;                      // Step scale for ray marching of signed distance primitives. 
                                        // - Some object transformations don't preserve the distances and 
                                        //   thus require shorter steps.
  float3 padding;

  ModelData plane;
  ModelData box;
  ModelData icosphere;
  ModelData cone;
};

// Attributes per primitive instance.
struct PrimitiveInstanceConstantBuffer {
  uint instanceIndex;
  uint primitiveType; // Procedural primitive type
};

// Dynamic attributes per primitive instance.
struct PrimitiveInstancePerFrameBuffer {
  float4x4 localSpaceToBottomLevelAS;   // Matrix from local primitive space to bottom-level object space.
  float4x4 bottomLevelASToLocalSpace;   // Matrix from bottom-level object space to local primitive space.
};

//struct Vertex {
//  float3 position;
//  float3 normal;
//};

struct Vertex {
  float4 pos;
  float4 normal;
  float2 texCoords; // ����� �� ��� �� ��� �����������?
};

struct FilterConstantData {
  float4x4 projToPrevProj;
};

#ifdef __CPLUSPLUS
struct ComputeData {
  glm::vec4 pos;
  glm::vec4 angVel;
  glm::vec4 scale; // w == objType?

  glm::mat4 currentOrn; // quaternions?
};

struct Material {
  // ��� ���?
  glm::vec4 color; // ?
  float reflection;
  float koef;
  float dummy1;
  float dummy2;
};

//struct Vertex {
//  glm::vec4 pos;
//  glm::vec4 normal;
//  glm::vec2 texCoords; // ����� �� ��� �� ��� �����������?
//};

// ��� ����� ������ ������� ��� ����������
// ��������� �����������? �������� viewproj ������� ����������
struct Matrices {
  glm::mat4 proj;
  glm::mat4 view;
  glm::mat4 invProj;
  glm::mat4 invView;

  glm::mat4 lastProj;
  glm::mat4 lastView;
  glm::mat4 lastInvProj;
  glm::mat4 lastInvView;

  // �������� ���������� ������ ���� 4
  //glm::mat4 viewproj;
  //glm::mat4 invViewproj;
  //glm::mat4 lastViewproj;
  //glm::mat4 lastInvViewproj;

  // ���� ����� � ����� ����� (lastViewproj * invViewproj)
  // �����? (������ ����� �������, �������� � �����-�� ������� ����� ����� ������ proj ������������)
  //glm::mat4 currentToPrevious;
};
#endif

// ��� ����� �� ��� ��� ��������� ������� �� ���������� �����

// ����� z ����������
//float zOverW = tex2D(depthTexture, texCoord);
// H ��� ������� ������� ������� � ����� �� -1 �� 1 (�������� � ������� dx12 ���������� �� �����)
//float4 H = float4(texCoord.x * 2 - 1, (1 - texCoord.y) * 2 - 1, zOverW, 1);
// ���� �������� world-space ������� ������� ������� �� ��������������� viewproj �������
//float4 D = mul(H, g_ViewProjectionInverseMatrix);
// ����� �������� �� w (��� ��� �������� ����������? ���������� �� ����� �������?)
//float4 worldPos = D / D.w;
// ������� ������� ������� �������
//float4 currentPos = H;
// ���������� ������� world-space ������� � �������������� ���������� viewproj �������� (���������� �� �����?)
//float4 previousPos = mul(worldPos, g_previousViewProjectionMatrix);
// ����������� � ������������ ���������� [-1,1] �������� �� w
//previousPos /= previousPos.w;
// ���������� ������� ������� � ���������� ����� �������� �������� �������
//float2 velocity = (currentPos - previousPos) / 2.f;

#endif // !SHARED_H
