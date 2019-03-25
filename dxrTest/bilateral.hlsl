#ifndef BILATERAL_HLSL
#define BILATERAL_HLSL

#include "Shared.h"

#define PI 3.14159265358979323846264338327950288419716939937510
#define EPS 1e-5
#define EQUAL_EPSILON 0.0001f

static const uint diameter = 31;
//static const float sigmaI = 12.0f;
//static const float sigmaS = 16.0f;

static const float sigmaS = 6.0f;
static const float sigmaL = 4.0f;

static const uint2 resolution = uint2(1280, 720);

//RWTexture2D<float4> output : register(u0);
//Texture2D<float4> colors : register(t0);
RWTexture2D<float> output : register(u0);
Texture2D<float2> colors : register(t0);
Texture2D<float4> normals : register(t1);
Texture2D<float> depths : register(t2);
Texture2D<uint2> pixelDatas : register(t3);

cbuffer ConstantBuffer : register(b0) {
  FilterConstantBuffer constantBuffer;
};

float distance(const uint2 xy, const uint2 ij) {
  const int xi = int(xy.x - ij.x);
  const int yj = int(xy.y - ij.y);
  return float(sqrt(xi*xi + yj* yj));
}

float gaussian(const float x, const float sigma) {
  return exp(-(x*x / (2 * sigma*sigma)) / (2 * PI * sigma*sigma));
}

float lum(const float4 color) {
  return length(color.xyz);
}

bool eq(const float4 first, const float4 second) {
  const float4 tmp = abs(first - second);

  return tmp.x < EQUAL_EPSILON && tmp.y < EQUAL_EPSILON && tmp.z < EQUAL_EPSILON && tmp.w < EQUAL_EPSILON;
}

#define BILATERAL_WORK_GROUP 16

[numthreads(BILATERAL_WORK_GROUP, BILATERAL_WORK_GROUP, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  const uint2 coord = uint2(DTid.x, DTid.y);

  const bool withinBounds = coord.x < resolution.x && coord.y < resolution.y;

  const float sigS = max(sigmaS, EPS);
  const float sigL = max(sigmaL, EPS);

  const float facS = -1. / (2.*sigS*sigS);
  const float facL = -1. / (2.*sigL*sigL);

  float sumW = 0.0f;
  //float4 sumC = float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sumC = 0.0f;
  const float halfSize = sigS * 2;
  const uint radius = diameter / 2;

  //const float l = lum(colors[coord]);
  const float2 current = colors[coord];
  const float4 normal = normals[coord];
  const uint2 pixelData = pixelDatas[coord];
  const float coef = float(pixelData.y) / float(pixelData.x);
  const float l = coef < 0.07f ? 0.0f : (coef > 0.93f ? 1.0f : current.x);
  const float dist = current.y;

  const float2 texelSize = 1.0f / float2(resolution);

  //float Z = texPosition.Sample(colorSampler, input.uv).z;
  const float depthK = 1.0f;
  const float depth = depths[coord] * depthK;
  const float Z_X1 = coord.x-1 > 0            ? depths[uint2(coord.x-1, coord.y)] : 1e9;
  const float Z_X2 = coord.x+1 < resolution.x ? depths[uint2(coord.x+1, coord.y)] : 1e9;
  const float Z_Y1 = coord.y-1 > 0            ? depths[uint2(coord.x, coord.y-1)] : 1e9;
  const float Z_Y2 = coord.y+1 < resolution.y ? depths[uint2(coord.x, coord.y+1)] : 1e9;
  const float Z_DDX = abs(Z_X1 - depth) < abs(Z_X2 - depth) ? depth - Z_X1 : Z_X2 - depth;
  const float Z_DDY = abs(Z_Y1 - depth) < abs(Z_Y2 - depth) ? depth - Z_Y1 : Z_Y2 - depth;

  // ��� ����� ���-�� ������� � ��������� 
  // ��� �� ��� ����� ������ � ����������� �� ���� ��� � ���� ����
  // � ���� ���� ��������� �� ������� (��� �� ��������� ������� ������?)
  // � ���� ���� ������� ����� �������� � ������
  // ���� ����� =1 �� �� ���� ������ ������ �� ���� (�� ���� 0)
  // ���� ����� =0 �� ��� ������? ��� �� ������ �������� (�� �� ���� ���� ��������� ������� �� ������� ����, �� ����� ������)
  // ���� ����� �� 0 �� 1 ��� �����������? ����� ��� � ���� ������ ����� ������������ ������� �������
  // rayHitDist ����� ����� ������� �� ��������� �� ��������� ���������, ���� �� ��� ��� ����� ����������?
  // ����������, ��� �� ���� ��� ������� ����� ����������� ���������� � ������� �������
  // ��� ���� �� ������� ��� ��� ������������, ����������� � ��� ���� ����������� ����������
  // ��� ������ ��� ���� ��� ������ ���������, �� ������ ����� � ������ ����� ������ ���
  // + �� ����� ������� �� � ���� ����� ���� ���� ����������

  // ����� �������������� ���� ���������� �� ���� ���� ��� �������� ����������
  // ������ � ���� ���������� ������ ������� �������� � ���������� ������
  // �� 
  // �� ���������� ������ ���������
  // ���� ������� ������� �� ����� (������ ������ ����� ��� ����� ��������� ��������������� ��� ��� ����)
  // �������� ������� ������ (����� ������� �� ���� � �������)
  // �� ����� ������� ��� ��������� ��������������� rayHitDist, ����� ���� ��� �� ����� �������������� ��������
  const float shadowZero = abs(1.0f - coef);
  //const float shadowZero = coef;
  const float rayHitDist = min(dist, 1.0f);
  // ����� ���� ��� �� �� ��� �� ����� ��������

  // ��� ����� ����� ��������, �� �������� ����� �����
  // ������ ����� ����� ������, �� ��� ����� �������
  //const uint newDiameter = uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter * max(abs(l - coef), 0.2f); // 1.0f - coef
  // ���� ��� �� ��������� � max'��, ��� ��-�� ���� �������� ������� ����
  //const uint newDiameter = uint(withinBounds) * uint(coef > 0.07f && coef < 0.93f) * diameter * max(abs(l - coef), 0.2f);
  //const uint newDiameter = uint(withinBounds) * uint(coef > 0.07f && coef < 0.93f) * diameter * (dist == 0.0f ? abs(1.0f - coef) : min(dist, 1.0f));
  //const uint newDiameter = uint(withinBounds) * uint(coef > 0.07f && coef < 0.93f) * diameter * max(abs(1.0f - coef), min(dist, 1.0f));
  //const uint newDiameter = uint(withinBounds) * uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter * max(abs(1.0f - coef), min(dist, 1.0f));
  //const uint newDiameter = uint(withinBounds) * uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter * shadowZero;
  const uint newDiameter = uint(withinBounds) * uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter;

  const uint kernelSize = radius * 2 + 1;
  //const uint kernelSize = diameter;
  const float sigma = (2 * pow(radius / 5.5f, 2)); // make the kernel span 6 sigma
  float result = 0.0f;
  float totalWeight = 0.0f;

  // ���� �������� �������������� ������� � ���? ������ ����� �������� ����������� � ���,
  // ��� ������ ��������� ��������� ��� ������� ���������, �������� �������� �����������
  // ��� ��� ���������� ��� ������ �������� �� ����� �������? ����� ����� � ������� ���� ������� ������
  // �� ���� �� ��� ����� �������� ��������� ����������� ������, �� ��������� ������� ��������� ��� ����� ��������� ��������
  // ������������� �� ���� ������� �� ��� ������ ��������� ���� �� ��������� ��������
  // � ������ ��� ����� ������� � ��� ������ ���� ������� ����

  //float gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
  //float gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

  //float color_weight[256 * jointChannels];
  //float space_weight[diameter*diameter];
  //int space_ofs_jnt[diameter*diameter];
  //int space_ofs_src[diameter*diameter];

  //int jointChannels = 1;
  //int srcChannels = 1;

  //for (uint i = 0; i < 256 * jointChannels; ++i) {
  //  color_weight[i] = exp(i*i*gauss_color_coeff);
  //}

  //int maxk = 0;
  //for (int i = -radius; i <= radius; i++) {
  //  for (int j = -radius; j <= radius; j++) {
  //    float r = sqrt(float(i*i) + float(j*j));
  //    if (r > radius) continue;

  //    space_weight[maxk] = exp(r*r*gauss_space_coeff);
  //    space_ofs_jnt[maxk] = int(i*jim.cols*jointChannels + j * jointChannels);
  //    space_ofs_src[maxk++] = int(i*sim.cols*srcChannels + j * srcChannels);
  //  }
  //}

  //for (uint i = 0; i < height; ++i) {
  //  for (uint j = 0; j < width; ++j) {
  //    float sum = 0.0f, wsum = 0.0f;
  //    const float val0 = ...; // ������ �������� ������ joints?

  //    for (uint k = 0; k < maxk; ++k) {
  //      const float val = ...; // joints?
  //      const float vals = ...; // ��������
  //      const float w = space_weight[k] * color_weight[abs(val-val0)];

  //      sum += vals * w;
  //      wsum += w;
  //    }

  //    output[i, j] = sum / wsum;
  //  }
  //}

  const float normalK = 1.0f;
  const uint width123 = 10;
  for (uint x = 0; x < newDiameter; ++x) {
    for (uint y = 0; y < width123; ++y) {
      const int2 offset = int2(radius - x, y);
      const uint2 neighbor = coord + offset;//int2(offset, y);
      //const uint2 neighbor = coord + int2(offset, 0);
      //const float l = length(float2(offset));

      const bool withinBoundsLocal = neighbor.x < resolution.x && neighbor.y < resolution.y;

      const uint2 neighborPixelData = pixelDatas[neighbor];
      const float neighborCoef = float(neighborPixelData.y) / float(neighborPixelData.x);
      const float invNeighborCoef = abs(1.0f - neighborCoef);
      const float expCoef = exp(invNeighborCoef / sigma);

      const float kernelWeight = exp(-length(float2(offset)) / sigma);
      //float2 offset = float2(float(offset.x), float(offset.y)) * texelSize;

      const float ZTmp = depths[neighbor] * depthK;
      const float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(offset.x, offset.y)));
      //const float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(offset, 0)));
      const float ZDelta = abs(depth - ZTmp);
      float ZWeight = exp(-ZDelta / (ZDeltaApprox + 0.001));

      const float4 NTmp = normals[neighbor];
      //NTmp = normalize(NTmp * 2 - 1);
      float normalWeight = pow(max(0, dot(normal, NTmp)), 128) * normalK;

      //ZWeight = 1;
      //normalWeight = 1;
      const float weight = ZWeight * normalWeight; //  * expCoef

      const float2 offsetData = colors[neighbor];
      const float offsetColor = offsetData.x;
      const float dist = offsetData.y;
      //result += offsetColor * kernelWeight * weight * uint(withinBoundsLocal);// *invNeighborCoef;
      result += neighborCoef * kernelWeight * weight * uint(withinBoundsLocal);
      totalWeight += kernelWeight * weight * uint(withinBoundsLocal);
    }
  }

  //output[coord] = totalWeight == 0.0f ? l : result / totalWeight;
  //result = 0.0f;
  //totalWeight = 0.0f;

  for (uint x = 0; x < newDiameter; ++x) {
    for (uint y = 0; y < width123; ++y) {
      const int2 offset = int2(y, radius - x);
      const uint2 neighbor = coord + offset;// int2(y, offset);
      //const float l = length(float2(offset));

      const bool withinBoundsLocal = neighbor.x < resolution.x && neighbor.y < resolution.y;

      const uint2 neighborPixelData = pixelDatas[neighbor];
      const float neighborCoef = float(neighborPixelData.y) / float(neighborPixelData.x);
      const float invNeighborCoef = abs(1.0f - neighborCoef);
      const float expCoef = exp(invNeighborCoef / sigma);

      const float kernelWeight = exp(-length(float2(offset)) / sigma);
      //float2 offset = float2(float(offset.x), float(offset.y)) * texelSize;

      const float ZTmp = depths[neighbor] * depthK;
      const float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(offset.x, offset.y)));
      //const float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(0, offset)));
      const float ZDelta = abs(depth - ZTmp);
      float ZWeight = exp(-ZDelta / (ZDeltaApprox + 0.001));

      const float4 NTmp = normals[neighbor];
      //NTmp = normalize(NTmp * 2 - 1);
      float normalWeight = pow(max(0, dot(normal, NTmp)), 128) * normalK;

      //ZWeight = 1;
      //normalWeight = 1;
      const float weight = ZWeight * normalWeight; //  * expCoef

      const float2 offsetData = colors[neighbor];
      const float offsetColor = offsetData.x;
      const float dist = offsetData.y;
      //result += offsetColor * kernelWeight * weight * uint(withinBoundsLocal);
      result += neighborCoef * kernelWeight * weight * uint(withinBoundsLocal);
      totalWeight += kernelWeight * weight * uint(withinBoundsLocal);
    }
  }

  //newDiameter = 32;

  // ����� ������ ����������� ����, ��-�� ���� ���������� �������� �� �����
  //const uint newDiameter = uint(withinBounds) * uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter;
  //const uint newDiameter = uint(withinBounds) * uint(pixelData.y != 0 && pixelData.y != pixelData.x) * diameter * abs(1.0f - coef);
  //for (uint i = 0; i < newDiameter; ++i) {
  //  for (uint j = 0; j < newDiameter; ++j) {
  //    const int2 offset = int2(radius - i, radius - j);
  //    const uint2 neighbor = coord + offset;
  //    //const float l = length(float2(offset));

  //    const bool withinBoundsLocal = neighbor.x < resolution.x && neighbor.y < resolution.y;

  //    const uint2 neighborPixelData = pixelDatas[neighbor];
  //    const float neighborCoef = float(neighborPixelData.y) / float(neighborPixelData.x);
  //    const float invNeighborCoef = abs(1.0f - neighborCoef);
  //    const float expCoef = exp(invNeighborCoef / sigma);

  //    const float kernelWeight = exp(-length(float2(offset.x, offset.y)) / sigma);
  //    //float2 offset = float2(float(offset.x), float(offset.y)) * texelSize;
  //    
  //    const float ZTmp = depths[neighbor];
  //    const float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(offset.x, offset.y)));
  //    const float ZDelta = abs(depth - ZTmp);
  //    float ZWeight = exp(-ZDelta / (ZDeltaApprox + 0.001));
  //    
  //    const float4 NTmp = normals[neighbor];
  //    //NTmp = normalize(NTmp * 2 - 1);
  //    float normalWeight = pow(max(0, dot(normal, NTmp)), 128);
  //    
  //    //ZWeight = 1;
  //    //normalWeight = 1;
  //    const float weight = ZWeight * normalWeight; //  * expCoef
  //    
  //    const float2 offsetData = colors[neighbor];
  //    const float offsetColor = offsetData.x;
  //    const float dist = offsetData.y;
  //    //float coef1337 = lerp(0, 1, offsetData.x); // shadowZero
  //    //result += coef1337 * kernelWeight * weight * uint(withinBoundsLocal);
  //    result += offsetColor * kernelWeight * weight * uint(withinBoundsLocal);
  //    totalWeight += kernelWeight * weight * uint(withinBoundsLocal); //  + dist // ������ ����� ������� ����������

  //    //

  //    //const int2 offset = int2(radius - i, radius - j);
  //    //const uint2 neighbor = coord + offset;

  //    //const float4 normalNeighbor = normals[neighbor];
  //    //const uint2 neighborPixelData = pixelDatas[neighbor];
  //    //const float neighborCoef = float(neighborPixelData.y) / float(neighborPixelData.x);
  //    ////if (!eq(normal, normalNeighbor)) continue;
  //    //const bool b = eq(normal, normalNeighbor);

  //    //const float2 offsetData = colors[neighbor];
  //    //const float offsetColor = offsetData.x;
  //    //const float dist = offsetData.y;

  //    //// �����, ��� ����� ��� �� �������� ���� ������ � ���������
  //    //// ������ ��� ����� ��� ��������� ����������� ��� ����� �������
  //    //// ��� ������ ��������� ��� ������ ���? ������ ��� �� ���� �� �����
  //    //// ��� ��� ��� ����� ����� ������� � ������� ���� ���� ����� ���� ������,
  //    //// ��� ������� � ������� ���� ���
  //    //// ��� ��� �������? ��������� ������?
  //    //
  //    //// ��������� ��������� ���� ������ �������� ������� ��������� � ���������� ����
  //    //// ��� ����� ����� �� ��� �����, �� � �� ������� �� ����� �����������

  //    //// ���� ������, ��� ����� ����� ���� "��������" �� ���� �������� �� �������� ����
  //    //// ������� � �������� ����� ���� "��������" �� �������, �� ��� �� ���� �� � ����� ���������
  //    //// ��� ��� �������? ��� ��� ���� � ����� ��� � ���� ������� ����������� ������� ������
  //    //// �� ���� � �������� 9�9 ���� �������� 1 ������� � �����

  //    //// �����, ������ ���������� ��������
  //    //// �� �������� ���������� ����� ����� ���������� ����������� � ���������� �����������
  //    //// ������� ����� ������� � ������� ���������� ������ �����������
  //    //// ���� ���� ��������� ��������� �� ����� �����, 
  //    //// �� � ������ ������� ��� ������ ������ rayHitDist 
  //    //// � ������ ���� ������ ���������� �������� � ������� ��� ������ 10% ���������
  //    //// �� ������ ����� ��� ����� ����� ��� ������ �� ������, 
  //    //// �������� ���� ��������� � ������� ��� - ����������
  //    //// ��� ������ � ���������? 
  //    //// ������ ������������� ������ �� ���� ��������� ����� ������� ��� ���� ����� ��� � ������������

  //    //const float distS = length(float2(offset)) * dist * (0.5f);
  //    ////float distL = lum(offsetColor) - l;
  //    ////const float distL = offsetColor - l;
  //    //const float distL = neighborCoef - coef;

  //    //const float wS = exp(facS*float(distS*distS));
  //    //const float wL = exp(facL*float(distL*distL));
  //    //const float w = wS * wL * float(b);

  //    //sumW += w;
  //    //sumC += offsetColor * w;
  //  }
  //}

  //output[coord] = sumW == 0.0f ? float2(l, dist) : float2(sumC / sumW, dist);
  //output[coord] = totalWeight == 0.0f ? float2(l, dist) : float2(result / totalWeight, dist);
  output[coord] = totalWeight == 0.0f ? l : result / totalWeight;
  //output[coord] = totalWeight == 0.0f ? output[coord]+l : output[coord]+result / totalWeight;
  //output[coord] = colors[coord];
}


//Texture2D texSSAO : register(t0);
//Texture2D texPosition : register(t1);
//Texture2D texNormal : register(t2);
//sampler colorSampler : register(s10);
//
//struct VS_Output {
//  float2 uv: TEXCOORD0;
//};
//
//float main(VS_Output input) : SV_Target {
//  const int blurRange = 8;
//  float TotalWeight = 0;
//
//  int2 textureSize;
//  texSSAO.GetDimensions(textureSize.x, textureSize.y);
//  float2 texelSize = 1.0 / float2(textureSize);
//  float result = 0.0;
//
//  float Z = texPosition.Sample(colorSampler, input.uv).z;
//  float Z_X1 = input.uv.x > 0 ? texPosition.Sample(colorSampler, float2(input.uv.x - texelSize.x, input.uv.y)) : 1e9;
//  float Z_X2 = input.uv.x < 1 ? texPosition.Sample(colorSampler, float2(input.uv.x + texelSize.x, input.uv.y)) : 1e9;
//  float Z_Y1 = input.uv.y > 0 ? texPosition.Sample(colorSampler, float2(input.uv.x, input.uv.y - texelSize.y)) : 1e9;
//  float Z_Y2 = input.uv.y < 1 ? texPosition.Sample(colorSampler, float2(input.uv.x, input.uv.y + texelSize.y)) : 1e9;
//  float Z_DDX = abs(Z_X1 - Z) < abs(Z_X2 - Z) ? Z - Z_X1 : Z_X2 - Z;
//  float Z_DDY = abs(Z_Y1 - Z) < abs(Z_Y2 - Z) ? Z - Z_Y1 : Z_Y2 - Z;
//
//  float3 NVal = texNormal.Sample(colorSampler, input.uv).xyz;
//  NVal = normalize(NVal * 2 - 1);
//
//  int KernelSize = blurRange * 2 + 1;
//  float Sigma = (2 * pow((KernelSize - 1.0) / 6.0, 2)); // make the kernel span 6 sigma
//
//  for (int x = -blurRange; x <= blurRange; x++) {
//    for (int y = -blurRange; y <= blurRange; y++) {
//      float KernelWeight = exp(-length(float2(x, y)) / Sigma);
//      float2 offset = float2(float(x), float(y)) * texelSize;
//
//      float ZTmp = texPosition.Sample(colorSampler, input.uv + offset).z;
//      float ZDeltaApprox = abs(dot(float2(Z_DDX, Z_DDY), float2(x, y)));
//      float ZDelta = abs(Z - ZTmp);
//      float ZWeight = exp(-ZDelta / (ZDeltaApprox + 0.001));
//
//      float3 NTmp = texNormal.Sample(colorSampler, input.uv + offset).xyz;
//      NTmp = normalize(NTmp * 2 - 1);
//      float NormalWeight = pow(max(0, dot(NVal, NTmp)), 128);
//
//      ZWeight = 1;
//      float Weight = ZWeight * NormalWeight;
//
//      result += texSSAO.Sample(colorSampler, input.uv + offset).r  KernelWeight  Weight;
//      TotalWeight += KernelWeight * Weight;
//    }
//  }
//
//  float outFragColor = result / TotalWeight;
//  return outFragColor;
//}

#endif