#pragma once
// Fixed N64/M32 NN helper. Each lane owns one column; never reduce q in SIMD.
template<unsigned K> __attribute__((always_inline)) inline void ordered_hvx_nn(
    const float* a, const float* b, float* y) {
    static_assert(K == 64 || K == 128, "Only separately audited diagnostic geometries");
    for (unsigned m = 0; m < 32; ++m) {
        for (unsigned tile = 0; tile < 2; ++tile) {
            HVX_Vector acc = Q6_V_vzero();
            for (unsigned q = 0; q < K; ++q) {
                int bits;
                std::memcpy(&bits, a + m * K + q, 4);
                const HVX_Vector av = Q6_V_vsplat_R(bits);
                HVX_Vector bv;
                std::memcpy(&bv, b + q * 64 + tile * 32, 128);
                const HVX_Vector product = Q6_Vsf_vmpy_VsfVsf(av, bv);
                acc = Q6_Vsf_vadd_VsfVsf(acc, product);
            }
            std::memcpy(y + m * 64 + tile * 32, &acc, 128);
        }
    }
}
