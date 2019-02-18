#ifndef PIXEL_ADDITION_HLSL
#define PIXEL_ADDITION_HLSL

#define EQUAL_EPSILON 0.0001f

#define WORKGROUP_SIZE 16

static const uint diameter = 9;
static const uint2 resolution = uint2(1280, 720);

RWTexture2D<uint2> output : register(u0);
Texture2D<float2> colors : register(t0);
Texture2D<float4> normals : register(t1);

groupshared uint pixelsCount;
groupshared uint shadowsCount;

bool eq(const float4 first, const float4 second) {
  const float4 tmp = abs(first - second);

  return tmp.x < EQUAL_EPSILON && tmp.y < EQUAL_EPSILON && tmp.z < EQUAL_EPSILON && tmp.w < EQUAL_EPSILON;
}

[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(const uint3 groupID : SV_GroupID,
          const uint3 groupThreadID : SV_GroupThreadID,
          const uint  groupIndex : SV_GroupIndex) {
  const uint radius = diameter / 2;
  const uint2 coord = uint2(groupID.x, groupID.y);
  const float4 normal = normals[coord];
  const uint shaderCount = (diameter + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

  pixelsCount = 0;
  shadowsCount = 0;

  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint i = 0; i < shaderCount; ++i) {
    const uint2 threadId = uint2(i*WORKGROUP_SIZE + groupThreadID.x, i*WORKGROUP_SIZE + groupThreadID.y);
    const bool redundant = (threadId.x < diameter) && (threadId.y < diameter);

    //const int2 offset = int2(radius - groupThreadID.x, radius - groupThreadID.y);
    const int2 offset = int2(radius - threadId.x, radius - threadId.y);
    const uint2 neighbor = coord + offset;

    const bool withinBounds = neighbor.x < resolution.x && neighbor.y < resolution.y;

    const uint2 finalCoord = neighbor * uint(withinBounds)+uint2(0, 0);

    // как тут не выходить за пределы ресурса? и чтоб без if

    const float2 shadow = colors[finalCoord];
    const float4 neighborNormal = normals[finalCoord];
    const bool equal = eq(normal, neighborNormal);

    InterlockedAdd(pixelsCount, uint(redundant && withinBounds && equal));
    InterlockedAdd(shadowsCount, uint(redundant && withinBounds && equal && shadow.x > 0.0f));
  }

  GroupMemoryBarrierWithGroupSync();

  if (groupIndex == 0) {
    output[coord] = uint2(pixelsCount, shadowsCount);
  }
}

#endif