struct VS_INPUT {
  float4 pos : POSITION;
  float4 normal : NORMAL;
  float2 texCoords : TEX_COORDS;

  column_major float4x4 model : MODEL_MATRIX;
};

struct VS_OUTPUT {
  float4 pos : SV_POSITION;
  float4 color : COLOR;
  float4 normal : NORMAL;
};

struct FS_OUTPUT {
  float4 color : SV_TARGET0;
  float4 normal : SV_TARGET1;
};

cbuffer ConstantBuffer : register(b0) {
  column_major float4x4 vpMat;
};

// simple vertex shader
// additional attention to SV_POSITION attribute
VS_OUTPUT vertexMain(VS_INPUT input) {
  VS_OUTPUT output;

  float4x4 mat = mul(vpMat, input.model);
  //float4x4 mat = input.model;

  //output.pos = mul(mul(input.pos, input.model), vpMat);
  //output.pos = mul(vpMat, mul(input.model, input.pos));
  output.pos = mul(mat, input.pos);
  //output.pos = mul(input.pos, vpMat);
  //output.color = float4(abs(input.normal.x), abs(input.normal.y), abs(input.normal.z), 1.0f);
  /*output.color = float4(input.normal.x < 0.0f ? 0.5f*abs(input.normal.x) : input.normal.x, 
                        input.normal.y < 0.0f ? 0.5f*abs(input.normal.y) : input.normal.y,
                        input.normal.z < 0.0f ? 0.5f*abs(input.normal.z) : input.normal.z,
                        1.0f);*/

  output.color = float4(0.8f, 0.8f, 0.8f, 1.0f);
  //output.normal = mul(input.model, input.normal);
  output.normal = input.normal;

  return output;
}

// simple pixel shader
FS_OUTPUT pixelMain(VS_OUTPUT input) {
  FS_OUTPUT output;

  //output.color = float4(input.normal.x < 0.0f ? 0.5f : input.normal.x, input.normal.y < 0.0f ? 0.5f : input.normal.y, input.normal.z < 0.0f ? 0.5f : input.normal.z, 1.0f);
  output.color = input.color;
  output.normal = input.normal;

  return output;
}