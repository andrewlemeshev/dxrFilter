#ifndef COMPUTE_LIGHTNING_HLSL
#define COMPUTE_LIGHTNING_HLSL

#include "Shared.h"

static const float4 backgroundColor = float4(0.8f, 0.9f, 1.0f, 1.0f);
static const float inShadowRadiance = 0.35f;
static const uint2 resolution = uint2(1280, 720);

cbuffer ConstantBuffer : register(b0) {
  SceneConstantBuffer constantBuffer;
};
//ConstantBuffer<SceneConstantBuffer> constantBuffer : register(b0);

Texture2D<float> shadows : register(t0);

Texture2D<float4> colors : register(t1);
Texture2D<float4> normals : register(t2);
Texture2D<float> depths : register(t3);

RWTexture2D<float4> output : register(u0);

float4 projectToWorldSpace(const float4 projSpaceCoord) {
  float4 newCoord = mul(constantBuffer.projectionToWorld, projSpaceCoord);
  newCoord = float4(newCoord.xyz / newCoord.w, 1.0f);

  return newCoord;
}

struct CalculatePhongData {
  float4 albedo; 
  float4 hitPos; 
  float4 rayDir; 
  float4 normal;
  bool isInShadow;
  float shadowCoef;
  float diffuseCoef; 
  float specularCoef;
  float specularPower;
};

//float4 calculatePhong(const float4 albedo, const float3 hitPos, const float3 rayDir, const float3 normal, const bool isInShadow, const float diffuseCoef = 1.0, const float specularCoef = 1.0, const float specularPower = 50) {
float4 calculatePhong(const CalculatePhongData data) {
  const float4 incidentLightRay = normalize(data.hitPos - constantBuffer.lightPosition);

  // Diffuse component.
  const float NdotL = max(saturate(dot(-incidentLightRay, data.normal)), 0.0f);
  //const float coef = (data.isInShadow ? 1.0f - data.shadowCoef : 1.0f);
  const float coef = 1.0f - data.shadowCoef;
  const float4 diffuseColor = coef * data.diffuseCoef * NdotL * constantBuffer.lightDiffuseColor * data.albedo;

  // Specular component.
  float4 specularColor = float4(0, 0, 0, 0);
  //if (!data.isInShadow) {
    //const float specularCoef2 = 1.0; 
    //const float specularPower2 = 50;

    const float4 lightSpecularColor = float4(1, 1, 1, 1);
    //float4 Ks = calculateSpecularCoefficient(hitPosition, rayDir, incidentLightRay, normal, specularPower);
    const float4 reflectedLightRay = normalize(reflect(-incidentLightRay, data.normal));
    const float4 Ks = pow(saturate( /*max(*/ dot(reflectedLightRay, normalize(data.rayDir)) /*, 0.0f)*/), data.specularPower);
    specularColor = data.specularCoef * Ks * lightSpecularColor * (1.0f - data.shadowCoef);
  //}

  // Ambient component.
  // Fake AO: Darken faces with normal facing downwards/away from the sky a little bit.
  //float4 ambientColor = float4(0, 0, 0, 0);
  float4 ambientColor = constantBuffer.lightAmbientColor;
  const float4 ambientColorMin = constantBuffer.lightAmbientColor - 0.1;
  const float4 ambientColorMax = constantBuffer.lightAmbientColor;
  float a = 1 - saturate(dot(data.normal, float4(0, -1, 0, 0)));
  ambientColor = data.albedo * lerp(ambientColorMin, ambientColorMax, a);

  return ambientColor + diffuseColor + specularColor;
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  // у нас в shadow.x будут приходить размытые значения
  // по идее то ли значения ниже определенного порога должны игнорироваться
  // то ли еще дополнительная квадратичная функция должна быть, которая практически полностью убирает большую часть пенумбры
  // в итоге остаться должна четкая тень (область в которой все пиксели пересекают объект) 
  // + небольшая часть перехода от тени к отсутствию, переход должен быть плавным
  // хотя с другой стороны у объектов почти всегда есть четкая полутень от ярких объектов
  // мне возможно нужно будет поискать контур, как это сделать?

  const uint2 coord = uint2(DTid.x, DTid.y);

  if (coord.x > resolution.x, coord.y > resolution.y) return;

  const float2 k = float2(coord) / float2(resolution);

  // Screen position for the ray
  // Invert Y for DirectX-style coordinates
  const float2 screenPos = float2(k.x*2.0f - 1.0f, -(k.y*2.0f - 1.0f));

  // Read depth and normal
  const float depth = depths.Load(int3(coord, 0));

  if (depth == 1.0f) {
    output[coord] = backgroundColor;
    return;
  }

  const float4 projSpace = float4(screenPos, depth, 1.0f);
  const float4 pos = projectToWorldSpace(projSpace);
  const float4 normal = normals.Load(int3(coord, 0));
  const float4 color = colors.Load(int3(coord, 0));
  const float2 shadow = shadows.Load(int3(coord, 0));
  const float4 cameraRayDir = normalize(pos - constantBuffer.cameraPosition);

  const float diffuseCoef = 1.0f;
  const float specularCoef = 0.4f;
  const float specularPower = 50.0f;

  const CalculatePhongData data = {
    color,
    pos,
    cameraRayDir,
    normal,
    shadow.x > 0.0f,
    shadow.x,
    diffuseCoef, 
    specularCoef, 
    specularPower
  };

  // reflectedColor
  const float4 phongColor = calculatePhong(data);
  float4 finalColor = phongColor;

  // Apply visibility falloff.
  float t = depth;
  finalColor = lerp(finalColor, backgroundColor, 1.0 - exp(-0.000002*t*t*t));

  output[coord] = finalColor;
}

#endif