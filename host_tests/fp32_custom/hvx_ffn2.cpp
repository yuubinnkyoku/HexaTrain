// Diagnostic-only audited NN FFN2 K128 and QKV K64; N64/M32, no fallback.
#include <cstring>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#if !defined(__hexagon__) || !defined(__HVX__) || __HVX_LENGTH__ != 128
#error Ordered FFN2 requires real Hexagon HVX in 128-byte mode
#endif
static_assert(sizeof(HVX_Vector) == 128, "HVX vector width");
static_assert(sizeof(float) == 4, "FP32 words");
#include "hvx_ordered_nn.h"
extern "C" __attribute__((noinline)) void hexatrain_ordered_hvx_ffn2(
    const float* a, const float* b, float* y) {
    ordered_hvx_nn<128>(a,b,y);
}
extern "C" __attribute__((noinline)) void hexatrain_ordered_hvx_qkv_nn64(
    const float* a, const float* b, float* y) {
    ordered_hvx_nn<64>(a,b,y);
}
