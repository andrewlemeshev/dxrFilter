#ifndef DEBUG_OUTPUT_HLSL
#define DEBUG_OUTPUT_HLSL

struct DebugBuffer {
  float multiplier;
  uint colors;
  uint normals;
  uint depths;
  uint pixelDatas;
  uint bilateral;
  uint lightning;
};

struct VS_OUTPUT {
  float4 pos : SV_POSITION;
};

struct FS_OUTPUT {
  float4 color : SV_TARGET0;
};

SamplerState sampler0 : register(s0);
Texture2D debug : register(t0);
Texture2D colors : register(t1);
Texture2D normals : register(t2);
Texture2D depths : register(t3);
Texture2D shadows : register(t4);
Texture2D pixelDatas : register(t5);
Texture2D bilateral : register(t6);
Texture2D lightning : register(t7);

// в cpp мы меняем текстурку в зависимости от наших предпочтений
// по идее достаточно одной текстурки debug, мы перезаполняем дескрипторы
// как быть с uint текстурками? понятие не имею
// нужно для них что-то специальное сделать

FS_OUTPUT pixelMain(const VS_OUTPUT vs) {
  FS_OUTPUT output;

  output.color = multiplier * debug.Sample(sampler0, vs.pos.xy);

  return output;
}

#endif