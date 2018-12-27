RWTexture2D<float4> outputColor : register(u0);
Texture2D<float4> inputColor : register(t0);

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  const uint2 coords = DTid.xy; // глобальный * количество * локальный тред

  const float4 currentColor = inputColor.Load(uint3(coords, 0));
  //const float4 currentColor = float4(0.4f, 0.5f, 0.6f, 1.0f);
  outputColor[coords] = currentColor;
}