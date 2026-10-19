// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only self test for the FP16 oracle used by probe_host.c.
// It verifies that the double -> FP16 rounding and the expected hi /
// remainder values are correct, without needing the device.
//
// Build (host, no Hexagon SDK required):
//   cc -std=c11 -O2 -o probe_host_selftest probe_host_selftest.c -lm

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static double fp16_to_double(uint16_t bits) {
  const int sign = (bits & 0x8000u) ? -1 : 1;
  const int exp = (bits >> 10) & 31;
  const int frac = bits & 1023;
  if (exp == 31) return frac ? NAN : sign * INFINITY;
  if (!exp) return sign * ldexp((double)frac, -24);
  return sign * ldexp((double)(1024 + frac), exp - 25);
}

static uint16_t double_to_fp16(double value) {
  if (value == 0.0) return 0;
  const double magnitude = fabs(value);
  int exp = 0;
  if (magnitude >= ldexp(1.0, -14)) {
    exp = (int)floor(log2(magnitude)) + 1;
  } else {
    exp = -14;
  }
  int mantissa_bits = 10;
  if (exp <= -14) mantissa_bits = 10 + (exp + 14);
  if (mantissa_bits < 0) mantissa_bits = 0;
  const double quantum = ldexp(1.0, exp - mantissa_bits);
  double scaled = value / quantum;
  double rounded = (scaled >= 0.0) ? floor(scaled + 0.5) : ceil(scaled - 0.5);
  if (fabs(scaled - floor(scaled)) == 0.5) {
    const double half = floor(scaled * 0.5);
    rounded = 2.0 * half;
  }
  double result = rounded * quantum;
  if (fabs(result) >= ldexp(1.0, exp) && exp > -14) {
    exp += 1;
    mantissa_bits = 10;
    const double q2 = ldexp(1.0, exp - mantissa_bits);
    scaled = value / q2;
    rounded = (scaled >= 0.0) ? floor(scaled + 0.5) : ceil(scaled - 0.5);
    if (fabs(scaled - floor(scaled)) == 0.5) {
      const double half = floor(scaled * 0.5);
      rounded = 2.0 * half;
    }
    result = rounded * q2;
  }
  uint32_t bits = 0;
  if (result < 0) bits |= 0x8000u;
  double mag = fabs(result);
  if (mag == 0.0) return (uint16_t)bits;
  const int e = (int)floor(log2(mag));
  if (e < -14) {
    const double sub = ldexp(1.0, -24);
    bits |= (uint32_t)(mag / sub + 0.5);
    return (uint16_t)bits;
  }
  const double frac = (mag / ldexp(1.0, e)) - 1.0;
  bits |= (uint32_t)((e + 15) << 10);
  bits |= (uint32_t)(frac * 1024.0 + 0.5);
  return (uint16_t)bits;
}

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

int main(void) {
  /* Round trip of known FP16 values. */
  expect_bits("fp16(1.0)", double_to_fp16(1.0), 0x3c00);
  expect_bits("fp16(66.0)", double_to_fp16(66.0), 0x5420);
  expect_bits("fp16(64.125)", double_to_fp16(64.125), 0x5402);
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

  /* The oracle math used by probe_host.c for q=6..10. */
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
    /* The main value is the FP16 rounding of the scaled product. */
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

  if (failures) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
