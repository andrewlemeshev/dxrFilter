#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#include "Shared.h"
#include "noise.hlsl"

static const float4 backgroundColor = float4(0.8f, 0.9f, 1.0f, 1.0f);
static const float inShadowRadiance = 0.35f;
static const float2 resolution = float2(1280, 720);

RaytracingAccelerationStructure globalScene : register(t0, space0);
ConstantBuffer<SceneConstantBuffer> constantBuffer : register(b0);
ConstantBuffer<PrimitiveConstantBuffer> material : register(b1); // local

// Triangle resources
//ByteAddressBuffer g_indices : register(t1, space0);
StructuredBuffer<uint> g_indices : register(t1, space0);
StructuredBuffer<Vertex> g_vertices : register(t2, space0);

RWTexture2D<float4> RenderTarget : register(u0);
//RWTexture2D<float4> colors : register(u1);
//RWTexture2D<float4> normals : register(u2);
//RWTexture2D<float> depths : register(u3);
Texture2D<float4> colors : register(t3);
Texture2D<float4> normals : register(t4);
Texture2D<float> depths : register(t5);

RWTexture2D<uint> randomTexture : register(u1);
RWTexture2D<float2> shadowTarget : register(u2);

typedef BuiltInTriangleIntersectionAttributes MyAttributes;

//struct RayPayload {
//  float hitDist;
//};

struct Ray {
  float3 pos;
  float3 dir;
};

// Diffuse lighting calculation.
float calculateDiffuseCoefficient(const float3 hitPosition, const float3 incidentLightRay, const float3 normal) {
  float fNDotL = saturate(dot(-incidentLightRay, normal));
  return fNDotL;
}

// Phong lighting specular component
float4 calculateSpecularCoefficient(const float3 hitPosition, const float3 rayDir, const float3 incidentLightRay, const float3 normal, const float specularPower) {
  /*const float3 reflected = reflect(incidentLightRay, normal);
  const float3 reflectedLightRay = normalize(reflected);

  const float3 invNormWorldRayDir = normalize(-rayDir);
  const float3 dotLightWorldRay = dot(reflectedLightRay, invNormWorldRayDir);

  return float4(pow(saturate(dotLightWorldRay), specularPower), 1.0f);*/

  float3 reflectedLightRay = normalize(reflect(incidentLightRay, normal));
  return pow(saturate(dot(reflectedLightRay, normalize(-rayDir))), specularPower);
}

// Fresnel reflectance - schlick approximation.
float3 fresnelReflectanceSchlick(const float3 I, const float3 N, const float3 f0) {
  float cosi = saturate(dot(-I, N));
  return f0 + (1 - f0)*pow(1 - cosi, 5);
}

float4 projectToWorldSpace(const float4 projSpaceCoord) {
  float4 newCoord = mul(constantBuffer.projectionToWorld, projSpaceCoord);
  //float4 newCoord = mul(projSpaceCoord, constantBuffer.projectionToWorld);
  newCoord = float4(newCoord.xyz / newCoord.w, 1.0f);

  return newCoord;
}

// Retrieve hit world position.
float3 HitWorldPosition() {
  return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Phong lighting model = ambient + diffuse + specular components.
float4 сalculatePhong(const float4 albedo, const float3 hitPos, const float3 rayDir, const float3 normal, const bool isInShadow, const float diffuseCoef = 1.0, const float specularCoef = 1.0, const float specularPower = 50) {
  float3 hitPosition = hitPos;
  float3 lightPosition = constantBuffer.lightPosition.xyz;
  float shadowFactor = isInShadow ? inShadowRadiance : 1.0f;
  float3 incidentLightRay = normalize(hitPosition - lightPosition);

  // Diffuse component.
  float4 lightDiffuseColor = constantBuffer.lightDiffuseColor;
  float NdotL = saturate(dot(-incidentLightRay, normal));
  NdotL = max(NdotL, 0.0f);
  float4 diffuseColor = shadowFactor * diffuseCoef * NdotL * lightDiffuseColor * albedo;

  // Specular component.
  float4 specularColor = float4(0, 0, 0, 0);
  if (!isInShadow) {
    //const float specularCoef2 = 1.0; 
    //const float specularPower2 = 50;

    float4 lightSpecularColor = float4(1, 1, 1, 1);
    //float4 Ks = calculateSpecularCoefficient(hitPosition, rayDir, incidentLightRay, normal, specularPower);
    float3 reflectedLightRay = normalize(reflect(incidentLightRay, normal));
    float4 Ks = pow(saturate( /*max(*/ dot(reflectedLightRay, normalize(rayDir)) /*, 0.0f)*/), specularPower);
    specularColor = specularCoef * Ks * lightSpecularColor;
  }

  // Ambient component.
  // Fake AO: Darken faces with normal facing downwards/away from the sky a little bit.
  //float4 ambientColor = float4(0, 0, 0, 0);
  float4 ambientColor = constantBuffer.lightAmbientColor;
  float4 ambientColorMin = constantBuffer.lightAmbientColor - 0.1;
  float4 ambientColorMax = constantBuffer.lightAmbientColor;
  float a = 1 - saturate(dot(normal, float3(0, -1, 0)));
  ambientColor = albedo * lerp(ambientColorMin, ambientColorMax, a);

  return ambientColor + diffuseColor + specularColor;
}

float4 traceRadianceRay(const Ray ray, const uint currentRayRecursionDepth) {
  if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH) {
    return float4(0, 0, 0, 0);
  }

  // Set the ray's extents.
  RayDesc rayDesc;
  rayDesc.Origin = ray.pos;
  rayDesc.Direction = ray.dir;
  // Set TMin to a zero value to avoid aliasing artifacts along contact areas.
  // Note: make sure to enable face culling so as to avoid surface face fighting.
  rayDesc.TMin = 0;
  rayDesc.TMax = 10000;

  RayPayload rayPayload = {float4(0, 0, 0, 0), currentRayRecursionDepth + 1};

  TraceRay(globalScene,
    RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
    ~0, // any instances
    0, // hit group offset (for shadow rays it will be different)
    0, // geometry stride (почему 2?? в туториале по количеству лучей)
    0, // miss shader offset (for shadow rays it will be different)
    rayDesc, rayPayload);

  return rayPayload.color;
}

bool traceShadowRay(const Ray ray, const uint currentRayRecursionDepth, inout ShadowRayPayload load) {
  if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH) {
    return false;
  }

  // Set the ray's extents.
  RayDesc rayDesc;
  rayDesc.Origin = ray.pos;
  rayDesc.Direction = ray.dir;
  // Set TMin to a zero value to avoid aliasing artifcats along contact areas.
  // Note: make sure to enable back-face culling so as to avoid surface face fighting.
  rayDesc.TMin = 0;
  rayDesc.TMax = 10000;

  // Initialize shadow ray payload.
  // Set the initial value to true since closest and any hit shaders are skipped. 
  // Shadow miss shader, if called, will set it to false.
  //ShadowRayPayload shadowPayload = {true, 0.0f};

  TraceRay(globalScene,
    RAY_FLAG_CULL_BACK_FACING_TRIANGLES
    //| RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    | RAY_FLAG_FORCE_OPAQUE,             // ~skip any hit shaders
    //| RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // ~skip closest hit shaders,
    ~0, // any instances
    1, // hit group offset
    0, // geometry stride (почему 2??)
    1, // miss shader offset
    rayDesc, load);

  return load.hit;
}

float random(float2 p) {
  // e^pi (Gelfond's constant)
  // 2^sqrt(2) (GelfondЦSchneider constant)
  const float2 K1 = float2(23.14069263277926, 2.665144142690225);

  //return fract( cos( mod( 12345678., 256. * dot(p,K1) ) ) ); // ver1
  //return fract(cos(dot(p,K1)) * 123456.); // ver2
  return frac(cos(dot(p, K1)) * 12345.6789); // ver3
}

static const float4 magic = float4(1111.1111, 3141.5926, 2718.2818, 0);

float3 permute(float3 x) {
  return mod289(((x*34.0) + 1.0)*x);
}

// glibc
static const uint multiplier = 1103515245;
static const uint increment = 12345;
static const uint modulus = 1 << 31;

uint LCG() {
  const uint2 coord = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);
  const uint num = randomTexture[coord];
  const uint nextNum = (multiplier * num + increment) % modulus;
  randomTexture[coord] = nextNum;

  return nextNum;
}

float randomFloat02() {
  return (LCG() & 0xFFFFFF) / 16777216.0f;
}

float3 randomInUnitSphere() {
  const float z = randomFloat02() * 2.0f - 1.0f;
  const float t = randomFloat02() * 2.0f * 3.1415926f;
  const float r = sqrt(max(0.0, 1.0f - z * z));
  const float x = r * cos(t);
  const float y = r * sin(t);
  float3 res = float3(x, y, z);
  res *= pow(randomFloat02(), 1.0 / 3.0);
  return res;
}

uint RNG(inout uint state) {
  uint x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 15;
  state = x;
  return x;
}

float randomFloat01(inout uint state) {
  return (RNG(state) & 0xFFFFFF) / 16777216.0f;
}

float3 randomInUnitSphere(const uint state) {
  uint tmp = state;
  const float z = randomFloat01(tmp) * 2.0f - 1.0f;
  const float t = randomFloat01(tmp) * 2.0f * 3.1415926f;
  const float r = sqrt(max(0.0, 1.0f - z * z));
  const float x = r * cos(t);
  const float y = r * sin(t);
  float3 res = float3(x, y, z);
  res *= pow(randomFloat01(tmp), 1.0 / 3.0);
  return res;
}

float3 randPosOnLightSphere(const float2 k, const float seed) {
  //const float camPosX = constantBuffer.cameraPosition.x == 0.0f ? 1.0f : constantBuffer.cameraPosition.x;
  //const float camPosZ = constantBuffer.cameraPosition.z == 0.0f ? 1.0f : constantBuffer.cameraPosition.z;
  const float3 lightPos = constantBuffer.lightPosition.xyz;
  const float radius = constantBuffer.lightRadius;

  // нужно помен€ть рандом, сделать его на основе uint текстуры
  // то есть сгенерить текстуру с рандомными uint'ами
  // чтобы убрать патерн у случайных чисел
  // + нужно нарисовать источник света
  // + нужно сделать какой нибудь фильтр (билатеральный)
  // фильтр должен работать и без темпоральной аккумул€ции
  // нужно понимать че и как работает

  float3 velocity;
  velocity = randomInUnitSphere();

  velocity = normalize(velocity);
  // normalize

  return lightPos + velocity * radius;
}

[shader("raygeneration")]
void rayGen() {
  const uint2 DTid = uint2(DispatchRaysIndex().x, DispatchRaysIndex().y);
  const float2 xy = DTid.xy+0.5f;

  const float2 k = float2(xy) / float2(DispatchRaysDimensions().xy);

  // Screen position for the ray
  float2 screenPos = (k)*2.0f - 1.0f;

  // Invert Y for DirectX-style coordinates
  screenPos.y = -screenPos.y;

  const float2 readGBufferAt2 = float2(xy.x, -xy.y);
  const float2 readGBufferAt1 = float2(xy.x, xy.y);

  // Read depth and normal
  const float depth = depths.Load(int3(readGBufferAt1, 0));

  if (depth == 1.0f) {
    RenderTarget[DTid] = backgroundColor;
    return;
  }

  const float4 projSpace = float4(screenPos, depth, 1.0f);
  const float4 pos = projectToWorldSpace(projSpace);
  const float4 cameraPos = constantBuffer.cameraPosition;

  const float4 dir = normals.Load(int3(readGBufferAt1, 0));
  const float4 albedoColor = colors.Load(int3(readGBufferAt1, 0));
  //const float4 albedoColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
  uint currentRecursionDepth = 0;
  const float4 cameraRayDir = normalize(pos - cameraPos);

  //const float seed = frac(constantBuffer.cameraPosition.x*constantBuffer.cameraPosition.y);
  const float seed = constantBuffer.elapsedTime;
  float3 rndPos = randPosOnLightSphere(k, seed);

  // Shadow component.
  // Trace a shadow ray.
  //constantBuffer.lightPosition.xyz
  const float3 dir2 = normalize(rndPos - pos.xyz);
  Ray shadowRay = {pos.xyz - EPSILON*dir2, dir2};
  ShadowRayPayload load = {true, 0.0f};
  bool shadowRayHit = traceShadowRay(shadowRay, currentRecursionDepth, load);

  // лучи нужно кидать на рандомную точку 
  // радиус
  // 

  // Reflected component.
  //float4 reflectedColor = float4(0, 0, 0, 0); 
  //// коэффициент отражени€ какой?
  //if (material.reflectanceCoef > 0.001) {
  //  // Trace a reflection ray.
  //  const float3 tmp = float3(cameraRayDir.x, cameraRayDir.y, cameraRayDir.z);
  //  Ray reflectionRay = {pos.xyz, normalize(reflect(tmp, dir.xyz))};
  //  float4 reflectionColor = traceRadianceRay(reflectionRay, currentRecursionDepth);

  //  float3 fresnelR = fresnelReflectanceSchlick(cameraRayDir.xyz, dir.xyz, albedoColor.xyz);
  //  reflectedColor = material.reflectanceCoef * float4(fresnelR, 1) * reflectionColor;
  //}

  //float4 phongColor = сalculatePhong(albedoColor, pos.xyz, cameraRayDir.xyz, dir.xyz, shadowRayHit, material.diffuseCoef, material.specularCoef, material.specularPower);
  //float4 color = phongColor + reflectedColor;

  //// Apply visibility falloff.
  //float t = depth;
  //color = lerp(color, backgroundColor, 1.0 - exp(-0.000002*t*t*t));

  //RenderTarget[DTid] = color;
  shadowTarget[DTid] = float2(float(shadowRayHit), load.hitDist);

  // как верно передать тень в другую текстуру?
  // мне при этом нужно вычислить свет на сцене
  // или потом это сделать?

  // ну собственно нужно вычислить здесь точку где есть тень, 
  // передать туда rayHitDist, и затем далее вычислить свет
  // следовательно здесь нужно просто получить дистанцию
  // затем запустить шейдер со светом и использовать скринспейс тень
}

// closest hit shaders

[shader("closesthit")]
void closestHitTriangle(inout RayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr) {
  // Get the base index of the triangle's first 16 bit index.
  const uint indexSizeInBytes = 4; // uint 32 bits
  const uint indicesPerTriangle = 3;
  const uint triangleIndexStride = indexSizeInBytes * indicesPerTriangle;
  //const uint baseIndex = PrimitiveIndex() * triangleIndexStride;

  uint baseIndex = 0;

  switch (InstanceID()) {
    case PLANE_ID: {
      //baseIndex = PLANE_INDEX_START * indexSizeInBytes + PrimitiveIndex() * triangleIndexStride;
      //baseIndex = PLANE_INDEX_START + PrimitiveIndex() * indicesPerTriangle;
      baseIndex = material.plane.indicesStart + PrimitiveIndex() * indicesPerTriangle;
    }
    break;
    case BOX_ID: {
      //baseIndex = BOX_INDEX_START * indexSizeInBytes + PrimitiveIndex() * triangleIndexStride;
      //baseIndex = BOX_INDEX_START + PrimitiveIndex() * indicesPerTriangle;
      baseIndex = material.box.indicesStart + PrimitiveIndex() * indicesPerTriangle;
    }
    break;
    case ICOSAHEDRON_ID: {
      //baseIndex = ICOSAHEDRON_INDEX_START * indexSizeInBytes + PrimitiveIndex() * triangleIndexStride;
      //baseIndex = ICOSAHEDRON_INDEX_START + PrimitiveIndex() * indicesPerTriangle;
      baseIndex = material.icosphere.indicesStart + PrimitiveIndex() * indicesPerTriangle;
    }
    break;
    case CONE_ID: {
      //baseIndex = CONE_INDEX_START * indexSizeInBytes + PrimitiveIndex() * triangleIndexStride;
      //baseIndex = CONE_INDEX_START + PrimitiveIndex() * indicesPerTriangle;
      baseIndex = material.cone.indicesStart + PrimitiveIndex() * indicesPerTriangle;
    }
    break;
  }

  // здесь мы должны получить индексы из g_indices
  //uint3 indices = g_indices.Load3(baseIndex);
  uint indices = g_indices[baseIndex];
  uint vertexIndex = 0;

  if (InstanceID() == PLANE_ID) vertexIndex = indices + material.plane.verticesStart;
  else if (InstanceID() == BOX_ID) vertexIndex = indices + material.box.verticesStart;
  else if (InstanceID() == ICOSAHEDRON_ID) vertexIndex = indices + material.icosphere.verticesStart;
  else if (InstanceID() == CONE_ID) vertexIndex = indices + material.cone.verticesStart;
  // а из вершин g_vertices получить нормаль
  float4 normal = g_vertices[vertexIndex].normal;

  // PERFORMANCE TIP: it is recommended to avoid values carry over across TraceRay() calls. 
  // Therefore, in cases like retrieving HitWorldPosition(), it is recomputed every time.

  // Shadow component.
  // Trace a shadow ray.
  float3 hitPosition = HitWorldPosition();
  Ray shadowRay = {hitPosition, normalize(constantBuffer.lightPosition.xyz - hitPosition)};
  ShadowRayPayload load = {true, 0.0f};
  bool shadowRayHit = traceShadowRay(shadowRay, rayPayload.recursionDepth, load);

  // дальше мы считаем цвет
  // где то здесь же мы считаем отражение
  // хот€ наверное все это мы должны считать не здесь, а при генерации луча (так как у нас как бы уже есть один уровень трассировки)

  // Reflected component.
  float4 reflectedColor = float4(0, 0, 0, 0);
  // коэффициент отражени€ какой?
  if (material.reflectanceCoef > 0.001) {
    // Trace a reflection ray.
    Ray reflectionRay = {hitPosition, reflect(WorldRayDirection(), normal.xyz)};
    float4 reflectionColor = traceRadianceRay(reflectionRay, rayPayload.recursionDepth);

    float3 fresnelR = fresnelReflectanceSchlick(WorldRayDirection(), normal.xyz, material.albedo.xyz);
    reflectedColor = material.reflectanceCoef * float4(fresnelR, 1) * reflectionColor;
  }

  float4 phongColor = сalculatePhong(material.albedo, hitPosition, WorldRayDirection(), normal.xyz, shadowRayHit, material.diffuseCoef, material.specularCoef, material.specularPower);
  float4 color = phongColor + reflectedColor;

  // Apply visibility falloff.
  float t = RayTCurrent();
  color = lerp(color, backgroundColor, 1.0 - exp(-0.000002*t*t*t));

  rayPayload.color = color;
}

[shader("closesthit")]
void shadowClosestHitTriangle(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  payload.hit = true;
  payload.hitDist = RayTCurrent();
}

// miss shaders

[shader("miss")]
void missRadiance(inout RayPayload payload) {
  const float4 color = float4(backgroundColor);
  payload.color = color;
}

[shader("miss")]
void missShadow(inout ShadowRayPayload payload) {
  payload.hit = false;
  payload.hitDist = 0.0f;
}

// intersection shaders
// нужны ли они мне?

#endif // RAYTRACING_HLSL