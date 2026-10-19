// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only self test for the FP16 helpers and the numeric oracle used by
// probe_host.c. It verifies the shared IEEE binary16 conversion and the
// expected hi / remainder values, without needing the device.
//
// Build (host, no Hexagon SDK required):
//   cc -std=c11 -O2 -o probe_host_selftest probe_host_selftest.c -lm

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fp16_util.h"

/* Whether the toolchain exposes a native _Float16 type that can serve as an
 * independent reference implementation for the shared converter. Clang
 * supports _Float16 as an extension on most targets, so it must be included
 * here; otherwise a clang build silently skips the cross-check. */
#if defined(__clang__)
#define HMX_HAS_NATIVE_FLOAT16 1
#elif defined(__GNUC__) && (__GNUC__ >= 13)
#define HMX_HAS_NATIVE_FLOAT16 1
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 202311L)
#define HMX_HAS_NATIVE_FLOAT16 1
#else
#define HMX_HAS_NATIVE_FLOAT16 0
#endif

static int failures = 0;

static void expect_bits(const char *name, uint16_t actual, uint16_t want) {
  if (actual != want) {
    printf("FAIL %s: got 0x%04x want 0x%04x\n", name, actual, want);
    failures++;
  } else {
    printf("ok   %s = 0x%04x\n", name, actual);
  }
}

static void expect_double(const char *name, double actual, double want, double tol) {
  if (fabs(actual - want) > tol) {
    printf("FAIL %s: got %.17g want %.17g\n", name, actual, want);
    failures++;
  } else {
    printf("ok   %s = %.17g\n", name, actual);
  }
}

/* Every finite binary16 value must survive a round trip unchanged. */
static void test_roundtrip_all_finite(void) {
  int mismatches = 0;
  uint16_t first_bad = 0;
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    const uint16_t pattern = (uint16_t)bits;
    const int exponent = (pattern >> 10) & 31;
    if (exponent == 31) continue; /* inf / NaN are not round-tripped */
    const double value = fp16_to_double(pattern);
    const uint16_t back = double_to_fp16(value);
    if (back != pattern) {
      if (!mismatches) first_bad = pattern;
      mismatches++;
    }
  }
  if (mismatches) {
    printf("FAIL roundtrip: %d mismatches (first bad pattern 0x%04x)\n",
           mismatches, first_bad);
    failures++;
  } else {
    printf("ok   roundtrip: all 63488 finite binary16 values preserved\n");
  }
}

/* Values that a correct converter must reproduce exactly. */
static void test_known_values(void) {
  expect_bits("fp16(1.0)", double_to_fp16(1.0), 0x3c00);
  expect_bits("fp16(66.0)", double_to_fp16(66.0), 0x5420);
  expect_bits("fp16(64.125)", double_to_fp16(64.125), 0x5402);
  expect_bits("fp16(64.0625)", double_to_fp16(64.0625), 0x5401);
  expect_bits("fp16(64.1875)", double_to_fp16(64.1875), 0x5403);
  expect_bits("fp16(-64.0625)", double_to_fp16(-64.0625), 0xd401);
  expect_bits("fp16(1.0009765625)", double_to_fp16(1.0009765625), 0x3c01);
  expect_bits("fp16(0.015625)", double_to_fp16(0.015625), 0x2400);
  expect_bits("fp16(0.00390625)", double_to_fp16(0.00390625), 0x1c00);
  expect_bits("fp16(-0.015625)", double_to_fp16(-0.015625), 0xa400);
  expect_bits("fp16(-0.00390625)", double_to_fp16(-0.00390625), 0x9c00);
  expect_bits("fp16(0.001953125)", double_to_fp16(0.001953125), 0x1800);
  expect_bits("fp16(0.0009765625)", double_to_fp16(0.0009765625), 0x1400);
  expect_bits("fp16(0.00048828125)", double_to_fp16(0.00048828125), 0x1000);
  expect_bits("fp16(0.000244140625)", double_to_fp16(0.000244140625), 0x0c00);
  expect_bits("fp16(0.0001220703125)", double_to_fp16(0.0001220703125), 0x0800);
  expect_bits("fp16(64.5)", double_to_fp16(64.5), 0x5408);
  expect_bits("fp16(65.0)", double_to_fp16(65.0), 0x5410);
  expect_bits("fp16(32.25)", double_to_fp16(32.25), 0x5008);
  expect_bits("fp16(132.0)", double_to_fp16(132.0), 0x5820);
  expect_bits("fp16(33.0)", double_to_fp16(33.0), 0x5020);
  expect_bits("fp16(129.0)", double_to_fp16(129.0), 0x5808);
  /* smallest subnormal and the boundary to the smallest normal */
  expect_bits("fp16(2^-24)", double_to_fp16(ldexp(1.0, -24)), 0x0001);
  expect_bits("fp16(2^-14)", double_to_fp16(ldexp(1.0, -14)), 0x0400);
  expect_bits("fp16(0)", double_to_fp16(0.0), 0x0000);
  expect_bits("fp16(-0)", double_to_fp16(-0.0), 0x8000);
}

/* Midpoint values must round to the even neighbour. */
static void test_ties_to_even(void) {
  /* 1.0 has ULP 2^-10; the midpoint between 1.0 and 1.0009765625 is
   * 1.00048828125, which must round to 1.0 (even). */
  expect_bits("tie below 1.0 -> 1.0", double_to_fp16(1.00048828125), 0x3c00);
  /* The midpoint between 1.0009765625 and 1.001953125 must round up to
   * 1.001953125 (even). */
  expect_bits("tie above 1.0 -> 1.001953125",
              double_to_fp16(1.00146484375), 0x3c02);
  /* 64 has ULP 2^-4; the midpoint between 64 and 64.0625 must round to
   * 64 (even significand). */
  expect_bits("tie below 64 -> 64", double_to_fp16(64.03125), 0x5400);
  /* The midpoint between 64.0625 and 64.125 must round to 64.125. */
  expect_bits("tie above 64 -> 64.125", double_to_fp16(64.09375), 0x5402);
  /* Negative values must follow the same rule. */
  expect_bits("negative tie -> -1.0", double_to_fp16(-1.00048828125), 0xbc00);
  expect_bits("negative tie -> -1.001953125",
              double_to_fp16(-1.00146484375), 0xbc02);
}

/* The oracle math used by probe_host.c for q=6..10. */
static void test_oracle(void) {
  struct { int q; uint16_t hi; uint16_t remainder; } cases[] = {
      {6, 0x5420, 0x2400},
      {7, 0x5410, 0x1c00},
      {8, 0x5408, 0x1400},
      {9, 0x5404, 0x0c00},
      {10, 0x5402, 0x0400},
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const int q = cases[i].q;
    const double a = 1.0 + ldexp(1.0, -q);
    const double product = a * a;
    const double scaled = product * 64.0;
    const double hi_value = fp16_to_double(double_to_fp16(scaled));
    const double remainder = scaled - hi_value;
    char name[64];
    snprintf(name, sizeof(name), "q=%d hi", q);
    expect_bits(name, double_to_fp16(hi_value), cases[i].hi);
    snprintf(name, sizeof(name), "q=%d remainder", q);
    expect_bits(name, double_to_fp16(remainder), cases[i].remainder);
    /* The remainder must be exactly representable as a negative FP16 too. */
    snprintf(name, sizeof(name), "q=%d negative remainder", q);
    expect_bits(name, double_to_fp16(hi_value - scaled),
                (uint16_t)(cases[i].remainder | 0x8000u));
    /* Reconstructing with (hi + remainder) / 64 must recover the product. */
    const double reconstructed =
        (fp16_to_double(cases[i].hi) + fp16_to_double(cases[i].remainder)) / 64.0;
    snprintf(name, sizeof(name), "q=%d reconstruction", q);
    expect_double(name, reconstructed, product, 0.0);
  }
}

/* Cross-check against the compiler's own binary16 conversion when the
 * toolchain provides _Float16. This is an independent implementation, so
 * a shared bug in our converter would still be caught. */
static void test_against_native_float16(void) {
#if HMX_HAS_NATIVE_FLOAT16
  int compared = 0;
  int mismatches = 0;
  uint16_t first_bad = 0;
  double first_bad_value = 0.0;
  /* Exactly representable inputs: round trip must agree. */
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    const uint16_t pattern = (uint16_t)bits;
    const int exponent = (pattern >> 10) & 31;
    if (exponent == 31) continue;
    const double value = fp16_to_double(pattern);
    const _Float16 native_value = (_Float16)value;
    uint16_t native_bits = 0;
    memcpy(&native_bits, &native_value, sizeof(native_bits));
    const uint16_t ours = double_to_fp16(value);
    compared++;
    if (ours != native_bits) {
      if (!mismatches) {
        first_bad = pattern;
        first_bad_value = value;
      }
      mismatches++;
    }
  }
  /* Non-representable inputs: midpoints between adjacent FP16 values and
   * points just below / above them, plus the overflow boundary. This is
   * what exercises the rounding (ties-to-even) logic rather than the
   * trivial identity path. */
  double prev_value = 0.0;
  for (uint32_t bits = 1; bits <= 0x7bffu; ++bits) {
    const uint16_t pattern = (uint16_t)bits;
    const int exponent = (pattern >> 10) & 31;
    if (exponent == 31) continue;
    const double value = fp16_to_double(pattern);
    if (value <= prev_value) continue;
    const double ulp = value - prev_value;
    /* midpoint (exact tie) */
    const double mid = (prev_value + value) * 0.5;
    /* point just below and just above the midpoint */
    const double below = mid - ulp * 0.25;
    const double above = mid + ulp * 0.25;
    const double probes[3] = {below, mid, above};
    for (int p = 0; p < 3; ++p) {
      const _Float16 native_value = (_Float16)probes[p];
      uint16_t native_bits = 0;
      memcpy(&native_bits, &native_value, sizeof(native_bits));
      const uint16_t ours = double_to_fp16(probes[p]);
      compared++;
      if (ours != native_bits) {
        if (!mismatches) {
          first_bad = pattern;
          first_bad_value = probes[p];
        }
        mismatches++;
      }
    }
    prev_value = value;
  }
  /* Overflow boundary region. */
  {
    const double probes[] = {65504.0, 65519.0, 65520.0, 65521.0, 65536.0};
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
      const _Float16 native_value = (_Float16)probes[i];
      uint16_t native_bits = 0;
      memcpy(&native_bits, &native_value, sizeof(native_bits));
      const uint16_t ours = double_to_fp16(probes[i]);
      compared++;
      if (ours != native_bits) {
        if (!mismatches) {
          first_bad_value = probes[i];
        }
        mismatches++;
      }
    }
  }
  if (mismatches) {
    printf("FAIL native _Float16: %d/%d mismatches (first bad pattern 0x%04x = %.17g)\n",
           mismatches, compared, first_bad, first_bad_value);
    failures++;
  } else {
    printf("ok   native _Float16: %d values agree (incl. midpoints/overflow)\n",
           compared);
  }
#else
  printf("skip native _Float16 cross-check: not supported by this toolchain\n");
#endif
}

/* Subnormal boundaries and the transition to the smallest normal. */
static void test_subnormal_boundaries(void) {
  /* Smallest subnormal and its neighbours. */
  expect_bits("fp16(2^-24)", double_to_fp16(ldexp(1.0, -24)), 0x0001);
  /* Half of the smallest subnormal quantum rounds to zero (tie -> even). */
  expect_bits("fp16(2^-25)", double_to_fp16(ldexp(1.0, -25)), 0x0000);
  /* 1.5 quanta rounds up to 2 quanta. */
  expect_bits("fp16(3*2^-25)", double_to_fp16(3.0 * ldexp(1.0, -25)), 0x0002);
  /* Largest subnormal. */
  expect_bits("fp16(largest subnormal)", double_to_fp16(1023.0 * ldexp(1.0, -24)),
              0x03ff);
  /* Midpoint between largest subnormal and smallest normal rounds to the
   * smallest normal (even neighbour). */
  expect_bits("fp16(mid subnormal->normal)",
              double_to_fp16(1023.5 * ldexp(1.0, -24)), 0x0400);
  /* Smallest normal. */
  expect_bits("fp16(2^-14)", double_to_fp16(ldexp(1.0, -14)), 0x0400);
  /* Just below smallest normal rounds to largest subnormal. */
  expect_bits("fp16(2^-14 - 2^-24)", double_to_fp16(ldexp(1.0, -14) - ldexp(1.0, -24)),
              0x03ff);
}

/* Values around the largest finite FP16 and the overflow threshold. */
static void test_near_max_finite(void) {
  expect_bits("fp16(65504)", double_to_fp16(65504.0), 0x7bff);
  expect_bits("fp16(65472)", double_to_fp16(65472.0), 0x7bfe);
  /* 65519 rounds down to max finite. */
  expect_bits("fp16(65519)", double_to_fp16(65519.0), 0x7bff);
  /* 65520 is the midpoint and rounds up to infinity. */
  expect_bits("fp16(65520)", double_to_fp16(65520.0), 0x7c00);
  /* Anything strictly beyond the threshold is infinity. */
  expect_bits("fp16(65521)", double_to_fp16(65521.0), 0x7c00);
  expect_bits("fp16(1e9)", double_to_fp16(1e9), 0x7c00);
  /* Negative side. */
  expect_bits("fp16(-65520)", double_to_fp16(-65520.0), 0xfc00);
  expect_bits("fp16(-65504)", double_to_fp16(-65504.0), 0xfbff);
}

/* Midpoint (tie) rounding across several binades, including negative. */
static void test_midpoint_rounding_extra(void) {
  /* binade around 2: ULP 2^-9; midpoint between 2.0 and 2.001953125. */
  expect_bits("tie 2.0 binade -> 2.0",
              double_to_fp16(2.0 + ldexp(1.0, -10)), 0x4000);
  expect_bits("tie 2.0 binade -> 2.001953125",
              double_to_fp16(2.0 + 3.0 * ldexp(1.0, -10)), 0x4002);
  /* binade around 16: ULP 2^-6. */
  expect_bits("tie 16 binade low", double_to_fp16(16.0 + ldexp(1.0, -7)),
              0x4c00);
  /* Negative midpoint. */
  expect_bits("negative tie 2.0 binade",
              double_to_fp16(-(2.0 + ldexp(1.0, -10))), 0xc000);
}

int main(void) {
  test_known_values();
  test_ties_to_even();
  test_midpoint_rounding_extra();
  test_subnormal_boundaries();
  test_near_max_finite();
  test_roundtrip_all_finite();
  test_against_native_float16();
  test_oracle();
  if (failures) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
