#ifndef DEBUG_OUTPUT_HLSL
#define DEBUG_OUTPUT_HLSL

#include "DebugShared.h"

// короче не работает - черный экран
// где я мог напортачить? чет не очень понимаю
// нужно проверить пиксом

static const float4 trianglePos[] = {
  float4(-1.0f, -1.0f,  0.0f,  1.0f),
  float4( 3.0f, -1.0f,  0.0f,  1.0f),
  float4(-1.0f,  3.0f,  0.0f,  1.0f)
};

static const float2 triangleUV[] = {
  float2( 0.0f,  1.0f),
  float2( 2.0f,  1.0f),
  float2( 0.0f, -1.0f)
};

struct VS_OUTPUT {
  float4 pos : SV_POSITION;
  float2 uv : TEXCOORD0;
};

VS_OUTPUT vertexMain(const uint vertexId : SV_VertexID) {
  VS_OUTPUT output;
  output.pos = trianglePos[vertexId];
  output.uv = triangleUV[vertexId];
  return output;
}

struct FS_OUTPUT {
  float4 color : SV_TARGET0;
};

cbuffer constantBuffer : register(b0) {
  DebugBuffer debugBuffer;
};

SamplerState sampler0 : register(s0);
Texture2D debug : register(t0);
//Texture2D colors : register(t1);
//Texture2D normals : register(t2);
//Texture2D depths : register(t3);
//Texture2D shadows : register(t4);
//Texture2D pixelDatas : register(t5);
//Texture2D bilateral : register(t6);
//Texture2D lightning : register(t7);

// в cpp мы меняем текстурку в зависимости от наших предпочтений
// по идее достаточно одной текстурки debug, мы перезаполняем дескрипторы
// как быть с uint текстурками? понятие не имею
// нужно для них что-то специальное сделать

FS_OUTPUT pixelMain(const VS_OUTPUT vs) {
  FS_OUTPUT output;

  output.color = debug.Sample(sampler0, vs.uv.xy);
  if (bool(debugBuffer.pixelDatas)) {
    output.color = float4(float(output.color.x) / float(output.color.y), 0.0f, 0.0f, 1.0f);
  } else {
    output.color *= debugBuffer.multiplier;
  }

  return output;
}

#endif