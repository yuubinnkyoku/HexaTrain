// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "hvx_rpc/fp32_finite.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

static bool checkBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return bool(hexatrain_fp32_all_finite(&value, 1)) == std::isfinite(value);
}

int main() {
  // Cover every exponent and sign, including signaling/quiet NaN payloads,
  // boundary mantissas, signed zero, subnormals, and largest finite values.
  for (uint32_t sign : {UINT32_C(0), UINT32_C(0x80000000)})
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
      for (uint32_t mantissa : {UINT32_C(0), UINT32_C(1), UINT32_C(0x3fffff),
                               UINT32_C(0x400000), UINT32_C(0x7fffff)})
        if (!checkBits(sign | (exponent << 23) | mantissa)) return 1;
  uint32_t bits = 1;
  for (int sample = 0; sample < 1000000; ++sample) {
    bits ^= bits << 13; bits ^= bits >> 17; bits ^= bits << 5;
    if (!checkBits(bits)) return 2;
  }
  if (!hexatrain_fp32_all_finite(nullptr, 0)) return 3;
  // Partial vector-width tails and bad values at any position must fail.
  for (int count : {1, 31, 32, 33, 63, 64, 65, 4096, 8192}) {
    std::vector<float> values(count + 1, 0.5f);
    float* unaligned = values.data() + 1;
    if (!hexatrain_fp32_all_finite(unaligned, count)) return 4;
    for (int index : {0, count / 2, count - 1}) {
      for (uint32_t bad : {UINT32_C(0x7f800000), UINT32_C(0xff800000),
                           UINT32_C(0x7f800001), UINT32_C(0x7fc00000)}) {
        std::memcpy(unaligned + index, &bad, sizeof(bad));
        if (hexatrain_fp32_all_finite(unaligned, count)) return 5;
      }
      unaligned[index] = 0.5f;
    }
  }
  std::puts("fp32_finite_classification=PASS (1002560 patterns, tails, bad positions)");
  return 0;
}
