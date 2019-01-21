#define PI 3.14159265358979323846264338327950288419716939937510
#define EPS 1e-5
#define EQUAL_EPSILON 0.0001f

static const uint diameter = 5;
//static const float sigmaI = 12.0f;
//static const float sigmaS = 16.0f;

static const float sigmaS = 6.0f;
static const float sigmaL = 4.0f;

static const uint2 resolution = uint2(1280, 720);

//RWTexture2D<float4> output : register(u0);
//Texture2D<float4> colors : register(t0);
RWTexture2D<float2> output : register(u0);
Texture2D<float2> colors : register(t0);
Texture2D<float4> normals : register(t1);

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

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
  const uint2 coord = uint2(DTid.x, DTid.y);
  //uint radius = diameter / 2;

  ///*if (coord.x < 2 || coord.y < 2) {
  //  output[coord] = colors[coord];
  //  return;
  //}*/

  //float3 filtered = float3(0.0f, 0.0f, 0.0f);
  //float wP = 0.0f;
  //uint2 neighbor = uint2(0, 0);

  //for (uint i = 0; i < diameter; ++i) {
  //  for (uint j = 0; j < diameter; ++j) {
  //    neighbor.x = coord.x - int(radius - i);
  //    neighbor.y = coord.y - int(radius - j);
  //    const float gi = gaussian(colors[neighbor] - output[coord], sigmaI);
  //    const float gs = gaussian(distance(coord, neighbor), sigmaS);
  //    const float w = gi * gs;
  //    filtered = filtered + colors[neighbor].xyz * w;
  //    wP = wP + w;
  //  }
  //}

  //filtered = filtered / wP;
  //output[coord] = filtered;

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
  const float l = current.x;
  const float dist = current.y;
  const float4 normal = normals[coord];
  for (uint i = 0; i < diameter; i++) {
    for (uint j = 0; j < diameter; j++) {
      const int2 offset = int2(radius - i, radius - j);
      const uint2 neighbor = coord + offset;

      const float4 normalNeighbor = normals[neighbor];
      if (!eq(normal, normalNeighbor)) continue;

      const float offsetColor = colors[neighbor].x;

      const float distS = length(float2(offset));
      //float distL = lum(offsetColor) - l;
      const float distL = offsetColor - l;

      const float wS = exp(facS*float(distS*distS));
      const float wL = exp(facL*float(distL*distL));
      const float w = wS * wL;

      sumW += w;
      sumC += offsetColor * w;
    }
  }

  output[coord] = float2(sumC / sumW, dist);
  //output[coord] = colors[coord];
}