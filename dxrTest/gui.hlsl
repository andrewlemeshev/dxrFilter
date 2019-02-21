struct VS_INPUT {
  float2 pos : POSITION;
  float2 uv  : TEXCOORD0;
  float4 col : COLOR0;
};

struct VS_OUTPUT {
  float4 pos : SV_POSITION;
  float4 col : COLOR0;
  float2 uv  : TEXCOORD0;
};

struct FS_OUTPUT {
  float4 color : SV_TARGET0;
};

cbuffer constantBuffer : register(b0) {
  column_major float4x4 vpMat;
};

// simple vertex shader
// additional attention to SV_POSITION attribute
VS_OUTPUT vertexMain(VS_INPUT input) {
  VS_OUTPUT output;

  output.pos = mul(vpMat, float4(input.pos, 0.0f, 1.0f));

  output.col = input.col;
  output.uv = input.uv;

  return output;
}

SamplerState sampler0 : register(s0);
Texture2D texture0 : register(t0);

// simple pixel shader
FS_OUTPUT pixelMain(VS_OUTPUT input) {
  FS_OUTPUT output;

  output.color = input.col * texture0.Sample(sampler0, input.uv);

  return output;
}