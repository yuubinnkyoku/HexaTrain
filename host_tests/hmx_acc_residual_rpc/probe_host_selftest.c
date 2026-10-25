// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only self test for the FP16 helpers and the numeric oracle used by
// probe_host.c. It verifies the shared IEEE binary16 conversion two ways:
//
//   1. Exact round trip of every one of the 65536 binary16 bit patterns
//      (63488 finite values including +0 / -0, 2 infinities, 2046 NaN
//      payloads), with the partition counts asserted exactly.
//   2. Cross-check against the compiler's own native _Float16 conversion,
//      when the toolchain provides one: exactly-representable inputs, a
//      systematic sweep of non-representable doubles between every adjacent
//      pair of finite values (both signs), the overflow boundary, and named
//      subnormal / tie / signed-zero boundary probes.
//
// Build (host, no Hexagon SDK required):
//   gcc -std=gnu11 -O2 -Wall -Wextra -Werror -o probe_host_selftest probe_host_selftest.c -lm
//   clang -std=gnu11 -O2 -Wall -Wextra -Werror -rtlib=compiler-rt -o probe_host_selftest probe_host_selftest.c
//   (clang needs the MSVC environment: run vcvars64.bat first.
//    scripts/run_hmx_selftest.ps1 wires both compilers up.)
//
// Output contract (scripts/run_hmx_selftest.ps1 relies on this):
//   - every section prints the number of comparisons it executed;
//   - the last line is exactly one of:
//       selftest=PASS compared=N mismatches=0 total_comparisons=M
//       selftest=PASS_NATIVE_SKIPPED compared=0 mismatches=0 total_comparisons=M
//       selftest=FAIL failures=F compared=N mismatches=M total_comparisons=M
//   - exit status 0 only for the PASS lines. PASS_NATIVE_SKIPPED is printed
//     only when the toolchain has no _Float16 type at all. A native
//     cross-check that compiles but executes zero comparisons (or fewer than
//     the documented partition) is a FAIL with a non-zero exit status.

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

/* Documented partition of the 65536 binary16 bit patterns. */
#define HMX_PATTERN_COUNT 65536L
#define HMX_FINITE_COUNT 63488L   /* exponent field != 31; includes +0 / -0 */
#define HMX_INF_COUNT 2L          /* 0x7c00 / 0xfc00 */
#define HMX_NAN_COUNT 2046L       /* 0x7c01..0x7fff and 0xfc01..0xffff */
#define HMX_SIGNED_ZERO_COUNT 2L  /* 0x0000 / 0x8000 (inside the finite set) */
/* The non-representable sweep iterates prev patterns 0x0000..0x7bfe, i.e.
 * every adjacent pair of positive finite values, and mirrors each probe to
 * the negative side. */
#define HMX_SWEEP_PAIR_COUNT 0x7bffL

static int failures = 0;
static long comparisons = 0; /* total comparisons executed, all sections */

/* --- classification helpers (bit level, no locale / libc dependence) --- */

static int bits_is_nan(uint16_t bits) {
  return ((bits & 0x7c00u) == 0x7c00u) && ((bits & 0x03ffu) != 0u);
}

static int double_is_nan(double value) { return value != value; }

static int double_is_inf(double value) {
  return value == INFINITY || value == -INFINITY;
}

static void expect_bits(const char *name, uint16_t actual, uint16_t want) {
  comparisons++;
  if (actual != want) {
    printf("FAIL %s: got 0x%04x want 0x%04x\n", name, actual, want);
    failures++;
  } else {
    printf("ok   %s = 0x%04x\n", name, actual);
  }
}

static void expect_double(const char *name, double actual, double want, double tol) {
  comparisons++;
  if (fabs(actual - want) > tol) {
    printf("FAIL %s: got %.17g want %.17g\n", name, actual, want);
    failures++;
  } else {
    printf("ok   %s = %.17g\n", name, actual);
  }
}

/* Per-section comparison accounting: every test prints how many comparisons
 * it actually executed so a silent early exit cannot masquerade as a pass. */
static void report_section(const char *name, long comparisons_before,
                           int failures_before) {
  const long n = comparisons - comparisons_before;
  const int f = failures - failures_before;
  if (f) {
    printf("FAIL section %s: comparisons=%ld failures=%d\n", name, n, f);
  } else {
    printf("ok   section %s: comparisons=%ld\n", name, n);
  }
}

/* Round trip of every binary16 bit pattern. Finite patterns (63488,
 * including +0 and -0) must survive decode -> encode unchanged; infinities
 * must also survive bit-for-bit. NaN payloads are deliberately checked by
 * classification only: the decoder collapses payloads to a quiet NaN, and
 * native _Float16 conversions canonicalize NaN payloads differently across
 * targets, so a bitwise NaN comparison would be unreliable. */
static void test_roundtrip_all_patterns(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
  long finite = 0, inf = 0, nan = 0, signed_zero = 0, mismatches = 0;
  uint16_t first_bad = 0;
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    const uint16_t pattern = (uint16_t)bits;
    const double value = fp16_to_double(pattern);
    const int exponent = (pattern >> 10) & 31;
    const int fraction = pattern & 1023;
    const int want_sign = (pattern & 0x8000u) ? 1 : 0;
    int ok;
    comparisons++;
    if (exponent == 31 && fraction == 0) {
      /* +infinity / -infinity: classification and bitwise round trip. */
      inf++;
      ok = double_is_inf(value) && (!!signbit(value) == want_sign) &&
           double_to_fp16(value) == pattern;
    } else if (exponent == 31) {
      /* NaN payload: decode must yield NaN, re-encode must yield a NaN
       * pattern (payload identity is not required, see comment above). */
      nan++;
      ok = double_is_nan(value) && bits_is_nan(double_to_fp16(value));
    } else {
      finite++;
      if (pattern == 0x0000u || pattern == 0x8000u) {
        /* Signed zero must keep its sign through decode and re-encode. */
        signed_zero++;
        ok = value == 0.0 && (!!signbit(value) == want_sign) &&
             double_to_fp16(value) == pattern;
      } else {
        ok = double_to_fp16(value) == pattern;
      }
    }
    if (!ok) {
      if (!mismatches) first_bad = pattern;
      mismatches++;
    }
  }
  printf("roundtrip patterns: comparisons=%ld (finite=%ld incl signed_zero=%ld"
         " inf=%ld nan=%ld)\n",
         finite + inf + nan, finite, signed_zero, inf, nan);
  if (mismatches) {
    printf("FAIL roundtrip: %ld mismatches (first bad pattern 0x%04x)\n",
           mismatches, first_bad);
    failures++;
  }
  if (finite != HMX_FINITE_COUNT || inf != HMX_INF_COUNT ||
      nan != HMX_NAN_COUNT || signed_zero != HMX_SIGNED_ZERO_COUNT ||
      finite + inf + nan != HMX_PATTERN_COUNT) {
    printf("FAIL roundtrip: pattern partition wrong (finite=%ld inf=%ld"
           " nan=%ld signed_zero=%ld; expected finite=%ld inf=%ld nan=%ld"
           " signed_zero=%ld total=%ld)\n",
           finite, inf, nan, signed_zero, HMX_FINITE_COUNT, HMX_INF_COUNT,
           HMX_NAN_COUNT, HMX_SIGNED_ZERO_COUNT, HMX_PATTERN_COUNT);
    failures++;
  } else if (!mismatches) {
    printf("ok   roundtrip: all %ld patterns preserved"
           " (%ld finite incl signed zero, %ld inf, %ld nan)\n",
           HMX_PATTERN_COUNT, finite, inf, nan);
  }
  report_section("roundtrip_all_patterns", comparisons_before, failures_before);
}

/* Values that a correct converter must reproduce exactly. */
static void test_known_values(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
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
  report_section("known_values", comparisons_before, failures_before);
}

/* Midpoint values must round to the even neighbour. */
static void test_ties_to_even(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
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
  report_section("ties_to_even", comparisons_before, failures_before);
}

/* The oracle math used by probe_host.c for q=6..10. */
static void test_oracle(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
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
  report_section("oracle", comparisons_before, failures_before);
}

#if HMX_HAS_NATIVE_FLOAT16
_Static_assert(sizeof(_Float16) == 2, "native _Float16 must be 16-bit");

/* Accounting for the native cross-check. first_bad records the bit pattern
 * our converter produced for first_bad_value (the first disagreement). */
static long native_compared = 0;
static int native_mismatches = 0;
static uint16_t native_first_bad = 0;
static double native_first_bad_value = 0.0;

/* Compare double_to_fp16(value) against the compiler's native conversion.
 *
 * NaN note: a bitwise NaN comparison is unreliable across targets because
 * native _Float16 conversions may canonicalize payloads (and the decoder
 * already collapses payloads), so NaN inputs are accepted when BOTH sides
 * classify as NaN. Finite values and infinities are compared bitwise. */
static void native_compare(double value) {
  const _Float16 native_value = (_Float16)value;
  uint16_t native_bits = 0;
  memcpy(&native_bits, &native_value, sizeof(native_bits));
  const uint16_t ours = double_to_fp16(value);
  int same;
  native_compared++;
  comparisons++;
  if (bits_is_nan(ours) || bits_is_nan(native_bits)) {
    same = bits_is_nan(ours) && bits_is_nan(native_bits);
  } else {
    same = (ours == native_bits);
  }
  if (!same) {
    if (!native_mismatches) {
      native_first_bad = ours;
      native_first_bad_value = value;
    }
    native_mismatches++;
  }
}
#endif /* HMX_HAS_NATIVE_FLOAT16 */

/* Cross-check against the compiler's own binary16 conversion when the
 * toolchain provides _Float16. This is an independent implementation, so a
 * shared bug in our converter would still be caught. */
static void test_against_native_float16(void) {
#if HMX_HAS_NATIVE_FLOAT16
  const long comparisons_before = comparisons;
  const int failures_before = failures;
  long representable = 0, inf_count = 0, nan_count = 0;
  long sweep_pairs = 0, sweep_compared = 0;
  long boundary_compared = 0;
  unsigned explicit_compared = 0;
  int partition_failures = 0;

  /* (a) Exactly representable inputs: all 63488 finite patterns (signed
   * zero included), both infinities and all 2046 NaN payloads. */
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    const uint16_t pattern = (uint16_t)bits;
    const int exponent = (pattern >> 10) & 31;
    const int fraction = pattern & 1023;
    if (exponent == 31 && fraction == 0) {
      inf_count++;
    } else if (exponent == 31) {
      nan_count++;
    } else {
      representable++;
    }
    native_compare(fp16_to_double(pattern));
  }

  /* (b) Systematic sweep of non-representable doubles: for every adjacent
   * pair of positive finite values probe fixed fractions of the ULP between
   * them (midpoint and just below / above it) plus a point inside the next
   * interval, each mirrored to the negative side. Every probe is an exact
   * double (ULPs are powers of two, offsets are multiples of 1/8), so the
   * sweep is fully deterministic. */
  {
    static const double offsets[] = {0.125, 0.25, 0.375, 0.5,
                                     0.625, 0.75, 0.875, 1.25};
    const unsigned n_offsets =
        (unsigned)(sizeof(offsets) / sizeof(offsets[0]));
    for (uint32_t prev_bits = 0; prev_bits <= 0x7bfeu; ++prev_bits) {
      const double prev = fp16_to_double((uint16_t)prev_bits);
      const double next = fp16_to_double((uint16_t)(prev_bits + 1u));
      const double ulp = next - prev;
      sweep_pairs++;
      for (unsigned i = 0; i < n_offsets; ++i) {
        const double probe = prev + offsets[i] * ulp;
        native_compare(probe);
        sweep_compared++;
        native_compare(-probe);
        sweep_compared++;
      }
    }
    if (sweep_pairs != HMX_SWEEP_PAIR_COUNT) {
      printf("FAIL native _Float16 sweep: %ld pairs executed, expected %ld\n",
             sweep_pairs, HMX_SWEEP_PAIR_COUNT);
      failures++;
      partition_failures++;
    }
    if (sweep_compared != 2 * HMX_SWEEP_PAIR_COUNT * (long)n_offsets) {
      printf("FAIL native _Float16 sweep: %ld comparisons, expected %ld\n",
             sweep_compared,
             2 * HMX_SWEEP_PAIR_COUNT * (long)n_offsets);
      failures++;
      partition_failures++;
    }
  }

  /* (c) Overflow boundary region, both signs. */
  {
    static const double probes[] = {65504.0, 65519.0, 65520.0,
                                    65521.0, 65536.0};
    for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
      native_compare(probes[i]);
      boundary_compared++;
      native_compare(-probes[i]);
      boundary_compared++;
    }
    if (boundary_compared != 10) {
      printf("FAIL native _Float16 boundary: %ld comparisons, expected 10\n",
             boundary_compared);
      failures++;
      partition_failures++;
    }
  }

  /* (d) Named boundary probes: subnormal edges, subnormal -> normal
   * transition, tie-to-even samples (both signs) and signed zero. */
  {
    double probes[18];
    unsigned n = 0;
    probes[n++] = 0.0;
    probes[n++] = -0.0;
    probes[n++] = ldexp(1.0, -24);                    /* 2^-24 */
    probes[n++] = ldexp(1.0, -25);                    /* 2^-25 tie */
    probes[n++] = 3.0 * ldexp(1.0, -25);              /* 1.5 subnormal quanta */
    probes[n++] = 1023.0 * ldexp(1.0, -24);           /* largest subnormal */
    probes[n++] = 1023.5 * ldexp(1.0, -24);           /* mid subnormal->normal */
    probes[n++] = ldexp(1.0, -14);                    /* smallest normal */
    probes[n++] = ldexp(1.0, -14) - ldexp(1.0, -24);  /* 2^-14 - quantum */
    probes[n++] = ldexp(1.0, -14) + ldexp(1.0, -24);  /* 2^-14 + quantum */
    probes[n++] = 1.00048828125;                      /* tie -> 1.0 */
    probes[n++] = 1.00146484375;                      /* tie -> 1.001953125 */
    probes[n++] = 64.03125;                           /* tie -> 64 */
    probes[n++] = 64.09375;                           /* tie -> 64.125 */
    probes[n++] = -1.00048828125;
    probes[n++] = -1.00146484375;
    probes[n++] = -64.03125;
    probes[n++] = -64.09375;
    for (unsigned i = 0; i < n; ++i) {
      native_compare(probes[i]);
    }
    explicit_compared = n;
    if (n != 18u) {
      printf("FAIL native _Float16 explicit probes: %u built, expected 18\n",
             n);
      failures++;
      partition_failures++;
    }
  }

  /* Per-section comparison accounting: a cross-check that compiled but
   * executed zero (or too few) comparisons is a defect, not a pass. */
  if (representable != HMX_FINITE_COUNT || inf_count != HMX_INF_COUNT ||
      nan_count != HMX_NAN_COUNT) {
    printf("FAIL native _Float16 pattern partition: representable=%ld"
           " (expected %ld), inf=%ld (expected %ld), nan=%ld"
           " (expected %ld)\n",
           representable, HMX_FINITE_COUNT, inf_count, HMX_INF_COUNT,
           nan_count, HMX_NAN_COUNT);
    failures++;
    partition_failures++;
  }
  if (native_compared == 0) {
    printf("FAIL native _Float16: executed zero comparisons\n");
    failures++;
    partition_failures++;
  }

  printf("native representable: comparisons=%ld (incl. +0/-0)\n",
         representable);
  printf("native specials: comparisons=%ld (inf=%ld bitwise,"
         " nan=%ld classify-only)\n",
         inf_count + nan_count, inf_count, nan_count);
  printf("native non-representable sweep: comparisons=%ld"
         " (8 ULP fractions x %ld pairs x 2 signs)\n",
         sweep_compared, sweep_pairs);
  printf("native overflow boundary: comparisons=%ld (both signs)\n",
         boundary_compared);
  printf("native explicit boundaries: comparisons=%u (subnormal/tie"
         "/signed-zero)\n",
         explicit_compared);
  if (native_mismatches) {
    printf("FAIL native _Float16: %d/%ld mismatches (first bad pattern"
           " 0x%04x = %.17g)\n",
           native_mismatches, native_compared, native_first_bad,
           native_first_bad_value);
    failures++;
  }
  if (!native_mismatches && !partition_failures) {
    printf("ok   native _Float16: compared=%ld mismatches=0\n",
           native_compared);
  }
  report_section("against_native_float16", comparisons_before,
                 failures_before);
#else
  printf("skip native _Float16 cross-check: not supported by this toolchain\n");
  printf("skip section against_native_float16: comparisons=0"
         " (toolchain lacks _Float16)\n");
#endif
}

/* Subnormal boundaries and the transition to the smallest normal. */
static void test_subnormal_boundaries(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
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
  /* One subnormal quantum above the smallest normal. */
  expect_bits("fp16(2^-14 + 2^-24)", double_to_fp16(ldexp(1.0, -14) + ldexp(1.0, -24)),
              0x0401);
  /* Negative side mirrors. */
  expect_bits("fp16(-2^-24)", double_to_fp16(-ldexp(1.0, -24)), 0x8001);
  expect_bits("fp16(-2^-25)", double_to_fp16(-ldexp(1.0, -25)), 0x8000);
  expect_bits("fp16(-3*2^-25)", double_to_fp16(-3.0 * ldexp(1.0, -25)), 0x8002);
  expect_bits("fp16(-largest subnormal)",
              double_to_fp16(-1023.0 * ldexp(1.0, -24)), 0x83ff);
  expect_bits("fp16(-mid subnormal->normal)",
              double_to_fp16(-1023.5 * ldexp(1.0, -24)), 0x8400);
  expect_bits("fp16(-2^-14)", double_to_fp16(-ldexp(1.0, -14)), 0x8400);
  expect_bits("fp16(-2^-14 + 2^-24)",
              double_to_fp16(-(ldexp(1.0, -14) + ldexp(1.0, -24))), 0x8401);
  expect_bits("fp16(-2^-14 - 2^-24)",
              double_to_fp16(-(ldexp(1.0, -14) - ldexp(1.0, -24))), 0x83ff);
  report_section("subnormal_boundaries", comparisons_before, failures_before);
}

/* Values around the largest finite FP16 and the overflow threshold. */
static void test_near_max_finite(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
  expect_bits("fp16(65504)", double_to_fp16(65504.0), 0x7bff);
  expect_bits("fp16(65472)", double_to_fp16(65472.0), 0x7bfe);
  /* 65519 rounds down to max finite. */
  expect_bits("fp16(65519)", double_to_fp16(65519.0), 0x7bff);
  /* 65520 is the midpoint and rounds up to infinity. */
  expect_bits("fp16(65520)", double_to_fp16(65520.0), 0x7c00);
  /* Anything strictly beyond the threshold is infinity. */
  expect_bits("fp16(65521)", double_to_fp16(65521.0), 0x7c00);
  expect_bits("fp16(65536)", double_to_fp16(65536.0), 0x7c00);
  expect_bits("fp16(1e9)", double_to_fp16(1e9), 0x7c00);
  /* Negative side. */
  expect_bits("fp16(-65504)", double_to_fp16(-65504.0), 0xfbff);
  expect_bits("fp16(-65519)", double_to_fp16(-65519.0), 0xfbff);
  expect_bits("fp16(-65520)", double_to_fp16(-65520.0), 0xfc00);
  expect_bits("fp16(-65521)", double_to_fp16(-65521.0), 0xfc00);
  expect_bits("fp16(-65536)", double_to_fp16(-65536.0), 0xfc00);
  expect_bits("fp16(-1e9)", double_to_fp16(-1e9), 0xfc00);
  report_section("near_max_finite", comparisons_before, failures_before);
}

/* Midpoint (tie) rounding across several binades, including negative. */
static void test_midpoint_rounding_extra(void) {
  const long comparisons_before = comparisons;
  const int failures_before = failures;
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
  report_section("midpoint_rounding_extra", comparisons_before, failures_before);
}

int main(void) {
  test_known_values();
  test_ties_to_even();
  test_midpoint_rounding_extra();
  test_subnormal_boundaries();
  test_near_max_finite();
  test_roundtrip_all_patterns();
  test_against_native_float16();
  test_oracle();
#if HMX_HAS_NATIVE_FLOAT16
  {
    const long compared = native_compared;
    const long mismatches = (long)native_mismatches;
    if (failures) {
      printf("selftest=FAIL failures=%d compared=%ld mismatches=%ld"
             " total_comparisons=%ld\n",
             failures, compared, mismatches, comparisons);
      return 1;
    }
    printf("selftest=PASS compared=%ld mismatches=0 total_comparisons=%ld\n",
           compared, comparisons);
  }
#else
  if (failures) {
    printf("selftest=FAIL failures=%d compared=0 mismatches=0"
           " total_comparisons=%ld\n",
           failures, comparisons);
    return 1;
  }
  printf("selftest=PASS_NATIVE_SKIPPED compared=0 mismatches=0"
         " total_comparisons=%ld\n",
         comparisons);
#endif
  return 0;
}
