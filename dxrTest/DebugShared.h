#ifndef DEBUG_SHARED_HLSL
#define DEBUG_SHARED_HLSL

#ifdef __cplusplus
#include <cstdint>
typedef uint32_t uint;
#endif

struct DebugBuffer {
  float multiplier;
  uint colors;
  uint normals;
  uint depths;
  uint pixelDatas;
  uint bilateral;
  uint lightning;
};

#endif