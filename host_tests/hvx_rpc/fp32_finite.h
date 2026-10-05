// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <float.h>
#include <stdint.h>
#include <string.h>

// Integer classification avoids the Hexagon libc _FDclass call for every
// element of every Muon intermediate. All IEEE binary32 NaNs and infinities
// have an all-ones exponent; subnormals and signed zero remain finite. memcpy
// keeps the load valid under strict aliasing. No floating-point arithmetic,
// normalization, reduction order, or validation coverage is changed.
#if defined(__cplusplus)
static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                  FLT_MAX_EXP == 128, "IEEE binary32 is required");
#else
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128, "IEEE binary32 is required");
#endif

static inline int hexatrain_fp32_all_finite(const float* values, int count) {
  uint32_t invalid = 0;
  for (int i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, values + i, sizeof(bits));
    invalid |= (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
  }
  return invalid == 0;
}
