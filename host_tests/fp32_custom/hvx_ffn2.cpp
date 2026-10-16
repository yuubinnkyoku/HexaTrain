// Diagnostic-only NN [32,128] x [128,64]. No alternate geometry or fallback.
#include <cstring>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#if !defined(__hexagon__) || !defined(__HVX__) || __HVX_LENGTH__ != 128
#error Ordered FFN2 requires real Hexagon HVX in 128-byte mode
#endif
static_assert(sizeof(HVX_Vector) == 128, "HVX vector width");
static_assert(sizeof(float) == 4, "FP32 words");
extern "C" __attribute__((noinline)) void hexatrain_ordered_hvx_ffn2(
    const float* a, const float* b, float* y) {
    for (unsigned m = 0; m < 32; ++m) {
        for (unsigned tile = 0; tile < 2; ++tile) {
            HVX_Vector acc = Q6_V_vzero();
            for (unsigned q = 0; q < 128; ++q) {
                int bits;
                std::memcpy(&bits, a + m * 128 + q, 4);
                const HVX_Vector av = Q6_V_vsplat_R(bits);
                HVX_Vector bv;
                // Exact 128-byte copy: no assumed QNN pointer alignment.
                std::memcpy(&bv, b + q * 64 + tile * 32, 128);
                const HVX_Vector product = Q6_Vsf_vmpy_VsfVsf(av, bv);
                acc = Q6_Vsf_vadd_VsfVsf(acc, product);
            }
            std::memcpy(y + m * 64 + tile * 32, &acc, 128);
        }
    }
}
