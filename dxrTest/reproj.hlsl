#ifndef REPROJ_HLSL
#define REPROJ_HLSL

//#include "Shared.h"

static const uint2 resolution = uint2(1280, 720);
static const float2 texelSize = 1.0f / float2(resolution);
static const float threshold = 0.01f;

// Sub-sample positions for 16x TAA
static const float2 SAMPLE_LOCS_16[16] = {
  float2(-8.0f,  0.0f) / 8.0f,
  float2(-6.0f, -4.0f) / 8.0f,
  float2(-3.0f, -2.0f) / 8.0f,
  float2(-2.0f, -6.0f) / 8.0f,
  float2( 1.0f, -1.0f) / 8.0f,
  float2( 2.0f, -5.0f) / 8.0f,
  float2( 6.0f, -7.0f) / 8.0f,
  float2( 5.0f, -3.0f) / 8.0f,
  float2( 4.0f,  1.0f) / 8.0f,
  float2( 7.0f,  4.0f) / 8.0f,
  float2( 3.0f,  5.0f) / 8.0f,
  float2( 0.0f,  7.0f) / 8.0f,
  float2(-1.0f,  3.0f) / 8.0f,
  float2(-4.0f,  6.0f) / 8.0f,
  float2(-7.0f,  8.0f) / 8.0f,
  float2(-5.0f,  2.0f) / 8.0f
};

// Sub-sample positions for 8x TAA
static const float2 SAMPLE_LOCS_8[8] = {
  float2(-7.0f,  1.0f) / 8.0f,
  float2(-5.0f, -5.0f) / 8.0f,
  float2(-1.0f, -3.0f) / 8.0f,
  float2( 3.0f, -7.0f) / 8.0f,
  float2( 5.0f, -1.0f) / 8.0f,
  float2( 7.0f,  7.0f) / 8.0f,
  float2( 1.0f,  3.0f) / 8.0f,
  float2(-3.0f,  5.0f) / 8.0f
};

// Let's assume that we are using 8x
#define SAMPLE_LOCS SAMPLE_LOCS_8
#define SAMPLE_COUNT 8

RWTexture2D<float4> outputColor : register(u0);
Texture2D<float4> colors : register(t0);
//Texture2D<float4> normals : register(t1);
Texture2D<float> depths : register(t1);

Texture2D<float4> lastColors : register(t2);
//Texture2D<float4> lastNormals : register(t1);
Texture2D<float> lastDepths : register(t3);

//ConstantBuffer<FilterConstantData> cBuffer : register(b0);
cbuffer ConstantBuffer : register(b0) {
  float4x4 projToPrevProj;
};

float computeLuminance(const float4 color) {
  return 0.299*color.r + 0.587*color.g + 0.114*color.b;
  //return (color.r + color.r + color.r + color.b + color.g + color.g + color.g + color.g) >> 3;
}

float4 getPrevCoords(const float4 projSpaceCoord) {
  float4 prevCoord = mul(projToPrevProj, projSpaceCoord);
  prevCoord /= prevCoord.w;

  return prevCoord;
}

float4 aabbClamp(const uint2 coords, const float4 currentColor, const float4 prevColor, inout float modulationFactor);
float calcModulationLumDiff(const float4 currentColor, const float4 prevColor, const float4 boxMax, const float modulationFactor);

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  uint2 coords = DTid.xy; // глобальный * количество * локальный тред

  if (coords.x >= resolution.x || coords.y >= resolution.y) return;

  const float depth = depths.Load(uint3(coords, 0)); // получаем глубину

  if (depth == 1.0f) {
    const float4 currentColor = colors.Load(uint3(coords, 0));
    outputColor[coords] = currentColor;
    //outputColor[coords] = float4(0.4f, 0.5f, 0.6f, 1.0f);
    return;
  }

  float2 screenPos = ((float2(coords)+0.5f) / float2(resolution)) * 2.0f - 1.0f;
  screenPos.y = screenPos.y;

  const float4 projSpaceCoords = float4(screenPos, depth, 1.0f);
  const float4 prevCoords = getPrevCoords(projSpaceCoords);
  const float projectedDepth = prevCoords.z;
  const float2 projectedCoords = (prevCoords.xy) * 0.5f + 0.5f;
  // / prevCoords.w

  // теперь что? нам по идее нужно почекать был ли этот пиксель на предыдущем кадре
  // то есть у него должна отличаться глубина? или что?

  //const float prevDepth = lastDepths.Load(uint3(projectedCoords, 0));

  const float4 currentColor = colors.Load(uint3(coords, 0));

  float4 outColor = currentColor;

  // 0.2 коэффициент

  // мне нужно брать 4 точки вокруг точки
  // считать веса 
  // веса это 4 точки вокруг текущего значения projectedCoords

  // (x1, y1) - (x1 + 1, y1 + 1)
  // Вес (x1, y1) = (x1 - x) * (y1 - y)

  // (x - x1) - 0 => 1
  //          - 1 => 0

  // билинейная фильтрация

  // мы берем несколько точек, вычисляем веса (для исторического значения), складываем их
  // потом складываем историческое значени + текущее и получаем новое историческое значение

  // сначало нужно посчитать координаты ближайших пикселей, как это сделать?
  // у нас есть texelSize (размер пикселя), по идее мне нужно разделить одно на другое
  // пиксель индекс ???
  //const uint2 index = uint2(uint(projectedCoords.x / texelSize.x), uint(projectedCoords.y / texelSize.y));
  const uint2 i = uint2(projectedCoords * resolution);
  const uint2 indices[] = {
    i,
    i + uint2(0, 1),
    i + uint2(1, 0),
    i + uint2(1, 1)
  };

  const float2 texCoords[] = {
    indices[0] / float2(resolution),
    indices[1] / float2(resolution),
    indices[2] / float2(resolution),
    indices[3] / float2(resolution)
  };

  float4 lastColorArr[] = {
    float4(0.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 0.0f, 0.0f)
  };

  float4 finalColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint k = 0; k < 4; ++k) {
    const float d = lastDepths.Load(uint3(indices[k], 0));

    if (abs(projectedDepth - d) < threshold) {
      lastColorArr[k] = lastColors.Load(uint3(indices[k], 0));
    }

    //const float2 diff = abs(projectedCoords - texCoords[k]);
    // че делать? мне нужен какой нибудь коэффициент
    // коэффициент должен быть связан с texelSize
    // то есть нужно разделить diff на texelSize
    // а может и нет
    //const float2 koef = diff / texelSize;
    //const float2 koef = diff * float2(resolution);
    const float2 koef = abs(projectedCoords - texCoords[k]) * float2(resolution);
    const float koefFinal = koef.x * koef.y;

    finalColor += lastColorArr[k] * koefFinal;
  }

  //finalColor /= 4.0f;
  const float alpha = 0.2f;
  //finalColor = alpha * currentColor + (1.0f - alpha) * finalColor;
  finalColor = currentColor;
  outputColor[coords] = finalColor;
  //outputColor[uint2(projectedCoords * resolution)] = finalColor;

  // по идее тут нужно сравнить до некоего предела
  //if (abs(projectedDepth - prevDepth) < threshold) 
  //{
  //  const float4 prevColor = lastColors.Load(uint3(projectedCoords, 0));
  //  //const float4 prevColor = lastColors.Load(uint3(coords, 0));

  //  float modulationFactor = 1.0 / 16.0;

  //  const float4 clampedColor = aabbClamp(coords, currentColor, prevColor, modulationFactor);

  //  // а теперь что? взять средний цвет? (эт вообще жизнеспособная идея, но она мне чет не нравится)
  //  // скорее тут нужно отдать приоритет все же предыдущему цвету
  //  //outColor += prevColor;
  //  //outColor /= 2.0f;

  //  // либо можно попробовать миксовать, как выбрать коэффициент?
  //  //outColor = lerp(currentColor, prevColor, modulationFactor);
  //  outColor = lerp(currentColor, clampedColor, modulationFactor);
  //}

  //outputColor[coords] = outColor;

  /*float ModulationFactor = 1.0 / 16.0;

  vec3 CurrentSubpixel = textureLod(CurrentBuffer, UV, 0.0).rgb;
  vec3 History = textureLod(HistoryBuffer, UV, 0.0).rgb;
  OutColor = mix(CurrentSubpixel, History, ModulationFactor);*/
}

//float4x4 jitterMatrix() {
//  const uint subsampleIdx = m_FrameCount % SAMPLE_COUNT;
//
//  const float2 subsampleSize = texelSize * 2.0f; // That is the size of the subsample in NDC
//
//  const float2 S = SAMPLE_LOCS[subsampleIdx]; // In [-1, 1]
//
//  float2 subsample = S * subsampleSize; // In [-SubsampleSize, SubsampleSize] range
//  subsample *= 0.5f; // In [-SubsampleSize / 2, SubsampleSize / 2] range
//
//  float4x4 jM = float4x4(
//    1.0f, 0.0f, 0.0f, 0.0f,
//    0.0f, 1.0f, 0.0f, 0.0f,
//    0.0f, 0.0f, 1.0f, 0.0f,
//    subsample.x, subsample.y, 0.0f, 1.0f
//  );
//
//  return jM * viewProj;
//}

float4 aabbClamp(const uint2 coords, const float4 currentColor, const float4 prevColor, inout float modulationFactor) {
  const float VARIANCE_CLIPPING_GAMMA = 1.0f;

  const float4 nearColor0 = colors.Load(int3(coords, 0), int2( 1,  0));
  const float4 nearColor1 = colors.Load(int3(coords, 0), int2( 0,  1));
  const float4 nearColor2 = colors.Load(int3(coords, 0), int2(-1,  0));
  const float4 nearColor3 = colors.Load(int3(coords, 0), int2( 0, -1));

  // Compute the two moments
  /*const float3 M1 = currentColor.xyz + nearColor0.xyz + nearColor1.xyz + nearColor2.xyz + nearColor3.xyz;
  const float3 M2 = currentColor.xyz * currentColor.xyz +
                    nearColor0.xyz * nearColor0.xyz + nearColor1.xyz * nearColor1.xyz +
                    nearColor2.xyz * nearColor2.xyz + nearColor3.xyz * nearColor3.xyz;

  const float3 MU = M1 / 5.0f;
  const float3 sigma = sqrt(M2 / 5.0f - MU * MU);

  const float4 boxMin = float4(MU - VARIANCE_CLIPPING_GAMMA * sigma, 1.0f);
  const float4 boxMax = float4(MU + VARIANCE_CLIPPING_GAMMA * sigma, 1.0f);*/

  const float4 boxMin = min(currentColor, min(nearColor0, min(nearColor1, min(nearColor2, nearColor3))));
  const float4 boxMax = max(currentColor, max(nearColor0, max(nearColor1, max(nearColor2, nearColor3))));

  modulationFactor = calcModulationLumDiff(currentColor, prevColor, boxMax, modulationFactor);

  return clamp(prevColor, boxMin, boxMax);
}

float calcModulationLumDiff(const float4 currentColor, const float4 prevColor, const float4 boxMax, const float modulationFactor) {
  const float EPSILON = 0.0001f;

  float lum0 = computeLuminance(currentColor);
  float lum1 = computeLuminance(prevColor);

  float diff = abs(lum0 - lum1) / (EPSILON + max(lum0, max(lum1, computeLuminance(boxMax))));
  diff = 1.0 - diff;
  diff *= diff;

  return modulationFactor * diff;
}

//float4 temporal_reprojection(float2 ss_txc, float2 ss_vel, float vs_dist) {
//  float4 texel0 = sample_color(_MainTex, ss_txc);
//  float4 texel1 = sample_color(_PrevTex, ss_txc - ss_vel);
//
//  const float _SubpixelThreshold = 0.5;
//  const float _GatherBase = 0.5;
//  const float _GatherSubpixelMotion = 0.1666;
//
//  // првильно ли я скопировал? где то должна быть и матрица
//  float2 texel_vel = ss_vel / texelSize.xy;
//  float texel_vel_mag = length(texel_vel) * vs_dist;
//  float k_subpixel_motion = saturate(_SubpixelThreshold / (FLT_EPS + texel_vel_mag));
//  float k_min_max_support = _GatherBase + _GatherSubpixelMotion * k_subpixel_motion;
//
//  float2 ss_offset01 = k_min_max_support * float2(-texelSize.x, texelSize.y);
//  float2 ss_offset11 = k_min_max_support * float2(texelSize.x, texelSize.y);
//  float4 c00 = sample_color(_MainTex, uv - ss_offset11);
//  float4 c10 = sample_color(_MainTex, uv - ss_offset01);
//  float4 c01 = sample_color(_MainTex, uv + ss_offset01);
//  float4 c11 = sample_color(_MainTex, uv + ss_offset11);
//
//  float4 cmin = min(c00, min(c10, min(c01, c11)));
//  float4 cmax = max(c00, max(c10, max(c01, c11)));
//
//  texel1 = clamp(texel1, cmin, cmax);
//
//  float lum0 = Luminance(texel0.rgb);
//  float lum1 = Luminance(texel1.rgb);
//
//  float unbiased_diff = abs(lum0 - lum1) / max(lum0, max(lum1, 0.2));
//  float unbiased_weight = 1.0 - unbiased_diff;
//  float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
//  float k_feedback = lerp(_FeedbackMin, _FeedbackMax, unbiased_weight_sqr);
//
//  // output
//  return lerp(texel0, texel1, k_feedback);
//}

// note: clips towards aabb center + p.w
float4 clip_aabb(
  float3 aabb_min, // cn_min
  float3 aabb_max, // cn_max
  float4 p,        // c_in
  float4 q         // c_hist 
) {
  float3 p_clip = 0.5 * (aabb_max + aabb_min);
  float3 e_clip = 0.5 * (aabb_max - aabb_min);

  float4 v_clip = q - float4(p_clip, p.w);
  float3 v_unit = v_clip.xyz / e_clip;
  float3 a_unit = abs(v_unit);
  float  ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

  if (ma_unit > 1.0) return float4(p_clip, p.w) + v_clip / ma_unit;
  else return q;// point inside aabb
}

#endif