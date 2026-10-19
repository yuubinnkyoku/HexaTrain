// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Shared IEEE 754 binary16 (half) helpers for the HMX residual probe.
// Used by both probe_host.c and probe_host_selftest.c so that the oracle
// and its test can never drift apart.

#ifndef HEXATRAIN_FP16_UTIL_H
#define HEXATRAIN_FP16_UTIL_H

#include <math.h>
#include <stdint.h>

/* Decode an IEEE binary16 bit pattern into a double. */
static inline double fp16_to_double(uint16_t bits) {
  const uint16_t sign = (bits & 0x8000u) ? 0x8000u : 0u;
  const int exp = (bits >> 10) & 31;
  const int frac = bits & 1023;
  double magnitude;
  if (exp == 31) {
    magnitude = frac ? NAN : INFINITY;
  } else if (exp == 0) {
    magnitude = ldexp((double)frac, -24);
  } else {
    magnitude = ldexp((double)(1024 + frac), exp - 25);
  }
  return sign ? -magnitude : magnitude;
}

/* Round a double to the nearest IEEE binary16 value (round to nearest,
 * ties to even) and return the raw bit pattern.
 *
 * The normal range uses the true half ULP, i.e. 2^(e-10) for a value with
 * unbiased exponent e. Subnormals use the fixed quantum 2^-24, signed zero
 * is preserved, and magnitudes that round past the largest finite value
 * become infinity. */
static inline uint16_t double_to_fp16(double value) {
  if (value != value) return 0x7e00u;             /* NaN */
  const uint16_t sign = (value < 0.0 || (value == 0.0 && signbit(value)))
                            ? 0x8000u : 0u;
  const double magnitude = fabs(value);
  if (magnitude == 0.0) return sign;              /* +/-0 */
  if (magnitude > 65520.0) return (uint16_t)(sign | 0x7c00u); /* overflow */

  int binary_exponent = 0;
  double fraction = frexp(magnitude, &binary_exponent);
  /* magnitude = fraction * 2^binary_exponent with 0.5 <= fraction < 1,
   * so the unbiased exponent is binary_exponent - 1. */
  const int exponent = binary_exponent - 1;

  if (exponent < -14) {
    /* Subnormal: fixed quantum 2^-24. */
    const double quantum = ldexp(1.0, -24);
    const double scaled = magnitude / quantum;
    const double floor_value = floor(scaled);
    const double delta = scaled - floor_value;
    double rounded;
    if (delta > 0.5) {
      rounded = floor_value + 1.0;
    } else if (delta < 0.5) {
      rounded = floor_value;
    } else {
      /* Exact tie: choose the even neighbour. */
      rounded = (fmod(floor_value, 2.0) == 0.0) ? floor_value : floor_value + 1.0;
    }
    if (rounded >= 1024.0) {
      /* Rounded up into the smallest normal value. */
      return (uint16_t)(sign | (1u << 10));
    }
    return (uint16_t)(sign | (uint32_t)rounded);
  }

  /* Normal range: quantum 2^(exponent-10), significand in [1024, 2048). */
  const double quantum = ldexp(1.0, exponent - 10);
  const double scaled = magnitude / quantum;
  const double floor_value = floor(scaled);
  const double delta = scaled - floor_value;
  double rounded;
  if (delta > 0.5) {
    rounded = floor_value + 1.0;
  } else if (delta < 0.5) {
    rounded = floor_value;
  } else {
    rounded = (fmod(floor_value, 2.0) == 0.0) ? floor_value : floor_value + 1.0;
  }
  int result_exponent = exponent;
  uint32_t significand = (uint32_t)rounded;
  if (significand == 2048u) {
    /* Rounding carried into the next binade. */
    result_exponent += 1;
    significand = 1024u;
  }
  if (result_exponent > 15) return (uint16_t)(sign | 0x7c00u); /* overflow */
  const uint32_t exponent_field = (uint32_t)(result_exponent + 15);
  const uint32_t fraction_field = significand - 1024u;
  return (uint16_t)(sign | (exponent_field << 10) | fraction_field);
}

#endif /* HEXATRAIN_FP16_UTIL_H */
