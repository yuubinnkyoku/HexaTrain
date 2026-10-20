// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only simulation of FP16 hi/lo input splitting for Phase E-2 Path B.
// No device, no SDK, no adb. The FP16 oracle is the shared fp16_util.h
// (RN ties-to-even, subnormal quantum 2^-24, overflow -> inf).
//
// The README currently claims the hi/lo split is exact because after
// removing hi the residual satisfies |Alo| <= 2^-11*|A| and is therefore
// "representable in FP16". That is NOT generally exact: the residual can
// need up to ~21 mantissa bits, it can fall into the FP16 subnormal
// range (fixed quantum 2^-24, or round to zero), and a single-precision
// subtraction adds its own rounding. This simulator MEASURES the
// residual of
//
//   hi   = RN16(x)
//   lo   = RN16(x - hi)              (split in double arithmetic)
//   rec  = decode(hi) + decode(lo)   (sum in double)
//
// and the GEMM synthesis error of Path B against a plain-FP16 baseline
// and a double oracle. Output is delimiter-stable one-line stats
// consumed by scripts/run_fp16x2_split_sim.ps1 and cited in the README.
//
// Groups:
//   G1  split recovery, normal range: 100000 log-uniform random doubles
//       across magnitudes 2^-13..2^15 plus 56 hardcoded hard cases.
//       Checks the README's claimed bound |recover-x| <= 2^-23*|x|.
//   G2  hard-case table rows (README citation), including the reviewer
//       example x = 1 + 2^-11 - 2^-23 under BOTH split variants.
//   G3  same recovery with the x-hi subtraction carried out in float
//       (single) precision instead of double. The float-variant error
//       is DECOMPOSED per case: e0 = |x - (double)(float)x| (double->
//       float input conversion, relative bound 2^-24 vs |x|), e1 =
//       float-subtraction rounding (relative bound 2^-24 vs |sub|),
//       and the final |x - recover| of the float split. All G3 numbers
//       are REPORT-ONLY (never FAIL); every comparison is
//       relative-vs-relative or absolute-vs-absolute.
//   G4  subnormal/underflow sweep, magnitudes 2^-20..2^-15.
//   G5  GEMM synthesis: plain FP16 vs Path-B 3/4-GEMM split variants
//       (both operands split; W-only split where activations are
//       already RN-FP16; separate float accumulators combined in
//       double vs one chained float accumulator).
//   G6  combine loss: float-combined vs double-combined partial sums.
//
// Exit status is 0 only if no FAIL line was emitted. Hard FAILs are the
// mathematically proven G1 invariants (when lo is a normal FP16,
// |recover-x| <= 2^-23*|x|; when lo is subnormal/zero, |recover-x| <=
// 2^-25 absolute) and the G5 assertion that both-operands-split beats
// plain FP16. Everything else — the README's unconditional 2^-23 bound
// and ALL G3 float-variant numbers — is REPORT-ONLY (see the
// asserted-vs-reported comment inside g1_g3_recovery). The README's
// unconditional 2^-23 bound is reported as the finding
// `split_normal_ok=0 ... bound_cases_exceeded=k` (every violation is
// classified; violations with a normal lo would be a FAIL, they are
// proven impossible above).
//
// Build (scripts/run_fp16x2_split_sim.ps1 wires gcc + optional clang):
//   gcc -std=gnu11 -O2 -Wall -Wextra -Werror -o hmx_fp16x2_split_sim hmx_fp16x2_split_sim.c -lm
//   clang -std=gnu11 -O2 -Wall -Wextra -Werror -rtlib=compiler-rt -o hmx_fp16x2_split_sim hmx_fp16x2_split_sim.c
//
// The PRNG is a fixed-seed xorshift64; every number below is
// reproducible run to run.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp16_util.h"

/* ---------------- deterministic PRNG ---------------- */

static uint64_t g_rng = 0x243F6A8885A308D3ull;

static uint64_t rng64(void) {
  uint64_t x = g_rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  g_rng = x;
  return x;
}

/* Uniform in [0,1) with 53 random bits. */
static double rng_unit(void) {
  return (double)(rng64() >> 11) * (1.0 / 9007199254740992.0);
}

/* ---------------- bookkeeping ---------------- */

static long g_checks = 0;
static long g_fail = 0;

/* ---------------- split helpers ---------------- */

typedef struct {
  uint16_t hi;
  uint16_t lo;
} fp16_pair;

/* Double-arithmetic split (G1): lo = RN16(x - hi). */
static fp16_pair split_double(double x) {
  fp16_pair p;
  p.hi = double_to_fp16(x);
  p.lo = double_to_fp16(x - fp16_to_double(p.hi));
  return p;
}

/* Single-precision-arithmetic split (G3): the residual subtraction is
 * carried out in float precision, lo = RN16((float)x - (float)hi).
 * Operand order matches the double variant so the two are comparable;
 * only the precision of the subtraction differs. (float)x first rounds
 * x to 24-bit mantissa, which is the dominant extra error.
 *
 * The per-case error budget is decomposed so the two float-rounding
 * sources are not conflated:
 *   e0      = |x - (double)(float)x|          input conversion, <= 2^-24*|x|
 *   sub     = (float)((float)x - (float)hi)   the float subtraction result
 *   exact   = (double)(float)x - f16d(hi)     exact residual of the float
 *                                             operands (exact in double
 *                                             for this test range)
 *   e1      = |(double)sub - exact|           float-subtraction rounding,
 *                                             <= ulp_f(|sub|)/2 (i.e.
 *                                             <= 2^-24*|sub|, sub normal)
 * e0/e1 are report-only; the recovery itself is unchanged:
 * lo = RN16(sub). */
static fp16_pair split_float_parts(double x, double *e0, double *e1,
                                   double *sub_abs) {
  fp16_pair p;
  float xf, hf, sub;
  double exact;
  p.hi = double_to_fp16(x);
  xf = (float)x;
  hf = (float)fp16_to_double(p.hi);
  sub = xf - hf;
  exact = (double)xf - fp16_to_double(p.hi);
  p.lo = double_to_fp16((double)sub);
  *e0 = fabs(x - (double)xf);
  *e1 = fabs((double)sub - exact);
  *sub_abs = fabs((double)sub);
  return p;
}

static fp16_pair split_float(double x) {
  double e0, e1, sub_abs;
  return split_float_parts(x, &e0, &e1, &sub_abs);
}

static double pair_recover(fp16_pair p) {
  return fp16_to_double(p.hi) + fp16_to_double(p.lo);
}

static int lo_is_normal(uint16_t lo) {
  return ((lo >> 10) & 31u) != 0u;
}

/* ---------------- G1 + G3: split recovery ---------------- */

/* 56 hardcoded hard cases: mantissa-dense values, near 65504, near
 * 2^-14, just below/above binade boundaries, small-x bound-break
 * demonstrations. Floating arithmetic on literals is a C constant
 * expression, so these are valid static initializers. */
static const double kHard[] = {
    /* mantissa-dense near 1 (the 1+2^-11+2^-23+2^-24 family) */
    1.0 + 0x1p-11 + 0x1p-23 + 0x1p-24,
    1.0 + 0x1p-11 - 0x1p-23, /* reviewer example */
    1.0 + 0x1p-11 + 0x1p-23,
    1.0 + 0x1p-11 + 0x1p-24,
    1.0 + 0x1p-11 - 0x1p-24,
    1.0 - 0x1p-11 + 0x1p-23 + 0x1p-24,
    1.0 - 0x1p-11 - 0x1p-24,
    1.0 + 0x1p-10 + 0x1p-23 + 0x1p-24,
    1.0 - 0x1p-10 + 0x1p-22 + 0x1p-24,
    1.0 + 0x1p-1 + 0x1p-12 + 0x1p-23,
    1.0 + 0x1p-23 + 0x1p-24,
    1.0 - 0x1p-23 - 0x1p-24,
    1.0 + 0x1.ff8p-12, /* 1 + 2040*2^-23-ish dense */
    /* negatives of the dense family */
    -(1.0 + 0x1p-11 + 0x1p-23 + 0x1p-24),
    -(1.0 + 0x1p-11 - 0x1p-23),
    -(1.0 + 0x1p-10 + 0x1p-23 + 0x1p-24),
    /* near 65504 (largest finite FP16) and the top binade (ULP 32) */
    65504.0,
    65488.0, /* midpoint tie between 65472 and 65504 */
    65488.0 + 0x1p-10,
    65488.0 - 0x1p-10,
    65472.0,
    65472.0 + 0x1p-10,
    65504.0 - 32.0,
    65504.0 - 33.0,
    65504.0 - 1.5,
    65503.5,
    /* 2^15 binade and 2^14 binade edges */
    32768.0,
    32768.0 - 16.0,
    32768.0 + 16.0,
    32768.0 - 16.5,
    16384.0 - 8.0,
    16384.0 + 8.0,
    16384.0 - 8.0 + 0x1p-14,
    /* near 2^-14 (smallest normal FP16) */
    0x1p-14,
    0x1p-14 + 0x1p-24,
    0x1p-14 + 0x1p-25, /* tie: hi -> 2^-14, lo -> 0 */
    0x1p-14 - 0x1p-25, /* tie: hi -> 2^-14, lo -> 0 */
    0x1p-14 + 0x1p-24 + 0x1p-30,
    0x1p-13,
    0x1p-13 + 0x1p-23,
    0x1p-13 + 0x1p-24 + 0x1p-30,
    0x1p-15, /* below smallest normal: hi subnormal */
    0x1p-15 + 0x1p-25,
    /* binade boundaries just below/above */
    1.0 - 0x1p-24,
    1.0 + 0x1p-24,
    2.0 - 0x1p-23,
    2.0 + 0x1p-23,
    0.5 - 0x1p-25,
    0.5 + 0x1p-25,
    0.25 - 0x1p-26,
    0.25 + 0x1p-26,
    16.0 - 0x1p-20,
    16.0 + 0x1p-20,
    1024.0 - 0x1p-14,
    1024.0 + 0x1p-14,
    /* small-x cases that demonstrate the subnormal-lo bound break */
    0x1p-13 + 0x1p-25,
    0x1p-8 + 0x1p-20,
    0x1p-5 + 0x1p-17 + 0x1p-30,
};
#define G1_HARD_COUNT ((long)(sizeof kHard / sizeof kHard[0]))

static double g3_max_ratio = 0.0;
static double g3_max_ratio_lo_normal = 0.0;
static double g3_sum_abs = 0.0;
static long g3_viol = 0;
static long g3_viol_lo_normal = 0;
static long g3_viol_lo_subnormal = 0;
/* e0: double->float input conversion, ratio e0/|x| vs the RN bound 2^-24. */
static double g3_e0_max_ratio = 0.0;
static double g3_e0_sum_ratio = 0.0;
static long g3_e0_viol = 0;
/* e1: float-subtraction rounding, ratio e1/|sub| vs the RN bound 2^-24
 * (relative stats only over sub != 0; exact cancellations counted). */
static double g3_e1_max_ratio = 0.0;
static double g3_e1_sum_ratio = 0.0;
static long g3_e1_viol = 0;
static long g3_e1_nonzero = 0;
static long g3_e1_zero = 0;

static void g1_g3_recovery(void) {
  const long total = 100000 + G1_HARD_COUNT;
  double max_abs = 0.0, sum_abs = 0.0, max_ratio = 0.0;
  double max_ratio_lo_normal = 0.0, sub_abs_max = 0.0;
  double worst_x = 0.0, worst_rec = 0.0;
  long n = 0, viol = 0, viol_lo_normal = 0, viol_lo_sub = 0;
  long i;

  for (i = 0; i < total; ++i) {
    double x;
    if (i < 100000) {
      /* log-uniform magnitude in [2^-13, 2^15); alternate full 53-bit
       * mantissas with half-ULP-adjacent mantissas to stress RN ties. */
      int e = -13 + (int)(rng_unit() * 28.0);
      double m;
      if (e > 14) e = 14;
      if (i & 1) {
        m = 1.0 + rng_unit();
      } else {
        m = 1.0 + ((double)(rng64() % 2048u) + 0.5) / 2048.0;
      }
      x = ldexp(m, e);
      if (rng64() & 1u) x = -x;
    } else {
      x = kHard[i - 100000];
    }
    if (x == 0.0 || !isfinite(x)) continue;

    {
      fp16_pair pd = split_double(x);
      double e0, e1, sub_abs;
      fp16_pair pf = split_float_parts(x, &e0, &e1, &sub_abs);
      double recd = pair_recover(pd);
      double errd = fabs(recd - x);
      double ratiod = errd / fabs(x);
      double errf = fabs(pair_recover(pf) - x);
      double ratiof = errf / fabs(x);
      int lown = lo_is_normal(pd.lo);
      /* G3 classifies by the FLOAT variant's own lo (pf.lo), which is
       * the lo actually used in the float recovery; it can differ from
       * the double variant's pd.lo. */
      int lownf = lo_is_normal(pf.lo);
      double bound;

      ++n;
      sum_abs += errd;
      g3_sum_abs += errf;
      if (errd > max_abs) max_abs = errd;
      if (ratiod > max_ratio) {
        max_ratio = ratiod;
        worst_x = x;
        worst_rec = recd;
      }
      if (ratiof > g3_max_ratio) g3_max_ratio = ratiof;
      if (lownf && ratiof > g3_max_ratio_lo_normal) {
        g3_max_ratio_lo_normal = ratiof;
      }
      if (lown && ratiod > max_ratio_lo_normal) max_ratio_lo_normal = ratiod;
      if (!lown && errd > sub_abs_max) sub_abs_max = errd;

      /* e0: double->float input conversion error, RELATIVE to |x|
       * (like-for-like against the RN bound 2^-24). */
      {
        double r0 = e0 / fabs(x);
        g3_e0_sum_ratio += r0;
        if (r0 > g3_e0_max_ratio) g3_e0_max_ratio = r0;
        if (r0 > ldexp(1.0, -24)) ++g3_e0_viol;
      }
      /* e1: pure float-subtraction rounding, RELATIVE to |sub| (the
       * exact result of the float subtraction). sub == 0 is an exact
       * cancellation (then e1 == 0 too); it has no relative scale and
       * is counted separately. */
      if (sub_abs == 0.0) {
        ++g3_e1_zero;
      } else {
        double r1 = e1 / sub_abs;
        ++g3_e1_nonzero;
        g3_e1_sum_ratio += r1;
        if (r1 > g3_e1_max_ratio) g3_e1_max_ratio = r1;
        if (r1 > ldexp(1.0, -24)) ++g3_e1_viol;
      }

      /* README's claimed bound for the DOUBLE variant (absolute vs
       * absolute): |recover-x| <= 2^-23*|x|. */
      bound = ldexp(fabs(x), -23);
      if (errd > bound) {
        ++viol;
        if (lown) {
          ++viol_lo_normal;
        } else {
          ++viol_lo_sub;
        }
      }
      /* Same claim applied to the FLOAT variant, in RELATIVE form:
       * ratiof vs 2^-23 (relative vs relative -- units match). This was
       * the G3 defect: it used to compare the dimensionless ratiof
       * against the ABSOLUTE bound 2^-23*|x|. REPORT-ONLY. */
      if (ratiof > ldexp(1.0, -23)) {
        ++g3_viol;
        if (lownf) {
          ++g3_viol_lo_normal;
        } else {
          ++g3_viol_lo_subnormal;
        }
      }

      /* ASSERTED (a break FAILs the run): the proven G1 double-arith
       * invariants only, absolute-vs-absolute —
       *   lo normal     => |recover-x| <= 2^-23*|x|
       *   lo subnormal  => |recover-x| <= 2^-25 absolute
       *   recovery finite
       * (plus the G5 synthesis assertion elsewhere).
       * REPORT-ONLY (never FAIL): the README's unconditional bound for
       * the double variant (split_normal_* lines) and ALL G3
       * float-variant numbers — the float path adds e0 and e1, whose
       * per-case total constant is not provable a priori, so G3 is
       * reported as the e0/e1 component bounds plus the final-recovery
       * ratio, each compared like-for-like. */
      ++g_checks;
      if (lown) {
        if (errd > bound) {
          ++g_fail;
          printf("FAIL g1_lo_normal_bound x=%.17g hi=0x%04x lo=0x%04x "
                 "abs_err=%.17g bound=%.17g\n",
                 x, (unsigned)pd.hi, (unsigned)pd.lo, errd, bound);
        }
      } else {
        if (errd > ldexp(1.0, -25)) {
          ++g_fail;
          printf("FAIL g1_subnormal_lo_abs_bound x=%.17g hi=0x%04x "
                 "lo=0x%04x abs_err=%.17g bound=2^-25\n",
                 x, (unsigned)pd.hi, (unsigned)pd.lo, errd);
        }
      }
      if (!isfinite(recd) || !isfinite(errd)) {
        ++g_fail;
        printf("FAIL g1_nonfinite x=%.17g hi=0x%04x lo=0x%04x\n", x,
               (unsigned)pd.hi, (unsigned)pd.lo);
      }
    }
  }

  printf("split_normal_ok=%d count=%ld max_ratio=%.6e "
         "bound_cases_exceeded=%ld\n",
         viol == 0 ? 1 : 0, n, max_ratio, viol);
  printf("split_normal_breakdown=viol_lo_normal=%ld viol_lo_subnormal=%ld "
         "viol_fraction=%.6e\n",
         viol_lo_normal, viol_lo_sub,
         n > 0 ? (double)viol / (double)n : 0.0);
  printf("split_normal_abs=max_abs=%.6e mean_abs=%.6e "
         "lo_normal_max_ratio=%.6e subnormal_lo_max_abs=%.6e "
         "worst_x=%.17g worst_recover=%.17g\n",
         max_abs, n > 0 ? sum_abs / (double)n : 0.0, max_ratio_lo_normal,
         sub_abs_max, worst_x, worst_rec);
  /* G3 float-variant decomposition (all report-only):
   * conv  = e0/|x| vs the double->float RN bound 2^-24,
   * e1    = e1/|sub| vs the float-subtraction RN bound 2^-24,
   * sub   = final |x-recover|/|x| vs the README bound 2^-23. */
  printf("split_fp16arith_conv=count=%ld max_ratio=%.6e mean_ratio=%.6e "
         "bound=%.6e bound_cases_exceeded=%ld\n",
         n, g3_e0_max_ratio,
         n > 0 ? g3_e0_sum_ratio / (double)n : 0.0, ldexp(1.0, -24),
         g3_e0_viol);
  printf("split_fp16arith_e1=count=%ld zero_sub=%ld max_ratio=%.6e "
         "mean_ratio=%.6e bound=%.6e bound_cases_exceeded=%ld\n",
         n, g3_e1_zero, g3_e1_max_ratio,
         g3_e1_nonzero > 0 ? g3_e1_sum_ratio / (double)g3_e1_nonzero : 0.0,
         ldexp(1.0, -24), g3_e1_viol);
  printf("split_fp16arith_sub=count=%ld max_ratio=%.6e mean_abs=%.6e "
         "max_ratio_double=%.6e bound=%.6e bound_cases_exceeded=%ld "
         "viol_lo_normal=%ld viol_lo_subnormal=%ld "
         "lo_normal_max_ratio=%.6e\n",
         n, g3_max_ratio, n > 0 ? g3_sum_abs / (double)n : 0.0, max_ratio,
         ldexp(1.0, -23), g3_viol, g3_viol_lo_normal, g3_viol_lo_subnormal,
         g3_max_ratio_lo_normal);
}

/* ---------------- G2: hard-case table ---------------- */

static void print_row(const char *name, const char *variant, double x,
                      fp16_pair p) {
  double rec = pair_recover(p);
  double err = fabs(rec - x);
  printf("split_g2_row name=%s variant=%s x=%.17g hi=0x%04x lo=0x%04x "
         "lo_normal=%d recover=%.17g abs_err=%.6e rel_err=%.6e\n",
         name, variant, x, (unsigned)p.hi, (unsigned)p.lo,
         lo_is_normal(p.lo), rec, err, err / fabs(x));
}

static void g2_table(void) {
  static const struct {
    const char *name;
    double x;
    int also_float;
  } rows[] = {
      {"reviewer_1p2m11m23", 1.0 + 0x1p-11 - 0x1p-23, 1},
      {"dense_1p11_23_24", 1.0 + 0x1p-11 + 0x1p-23 + 0x1p-24, 1},
      {"dense_neg_1p11_23", -(1.0 + 0x1p-11 - 0x1p-23), 0},
      {"near65504_tie", 65488.0, 0},
      {"near65504_tie_minus_eps", 65488.0 - 0x1p-10, 0},
      {"near_2powneg14_tie", 0x1p-14 + 0x1p-25, 0},
      {"binade_edge_1_minus", 1.0 - 0x1p-24, 0},
      {"smallx_subnormal_lo", 0x1p-13 + 0x1p-25, 1},
  };
  size_t r;
  for (r = 0; r < sizeof rows / sizeof rows[0]; ++r) {
    print_row(rows[r].name, "double", rows[r].x, split_double(rows[r].x));
    if (rows[r].also_float) {
      print_row(rows[r].name, "float", rows[r].x, split_float(rows[r].x));
    }
    g_checks += 2;
  }
}

/* ---------------- G4: subnormal / underflow sweep ---------------- */

static void g4_subnormal_sweep(void) {
  /* Magnitudes 2^-20 .. 2^-15: every value is below the smallest normal
   * FP16 (2^-14), so hi lands on the fixed 2^-24 quantum (or zero) and
   * the residual |x-hi| <= 2^-25 almost always rounds to lo = 0. */
  double max_rel = 0.0, max_abs = 0.0;
  double bound_break_min_mag = 0.0;
  int have_break = 0;
  long count = 0, lo_zero = 0, bound_break = 0;
  int i;

  for (i = 0; i < 200; ++i) {
    double mag = ldexp(1.0, -20) * pow(2.0, 5.0 * (double)i / 199.0);
    double xs[2];
    int s;
    xs[0] = mag;
    xs[1] = -mag;
    for (s = 0; s < 2; ++s) {
      double x = xs[s];
      fp16_pair p = split_double(x);
      double rec = pair_recover(p);
      double err = fabs(rec - x);
      double rel = err / fabs(x);
      ++count;
      if (p.lo == 0u || p.lo == 0x8000u) ++lo_zero;
      if (err > max_abs) max_abs = err;
      if (rel > max_rel) max_rel = rel;
      if (err > ldexp(fabs(x), -23)) {
        ++bound_break;
        if (!have_break || fabs(x) < bound_break_min_mag) {
          bound_break_min_mag = fabs(x);
          have_break = 1;
        }
      }
      ++g_checks;
      /* Proven absolute floor for this range: |recover-x| <= 2^-25. */
      if (err > ldexp(1.0, -25)) {
        ++g_fail;
        printf("FAIL g4_subnormal_abs_bound x=%.17g hi=0x%04x lo=0x%04x "
               "abs_err=%.17g\n",
               x, (unsigned)p.hi, (unsigned)p.lo, err);
      }
    }
  }
  /* Plus a random log-uniform set in the same magnitude range. */
  for (i = 0; i < 2000; ++i) {
    double mag = ldexp(1.0, -20) * pow(2.0, 5.0 * rng_unit());
    double x = (rng64() & 1u) ? mag : -mag;
    fp16_pair p = split_double(x);
    double rec = pair_recover(p);
    double err = fabs(rec - x);
    double rel = err / fabs(x);
    ++count;
    if (p.lo == 0u || p.lo == 0x8000u) ++lo_zero;
    if (err > max_abs) max_abs = err;
    if (rel > max_rel) max_rel = rel;
    if (err > ldexp(fabs(x), -23)) {
      ++bound_break;
      if (!have_break || fabs(x) < bound_break_min_mag) {
        bound_break_min_mag = fabs(x);
        have_break = 1;
      }
    }
    ++g_checks;
    if (err > ldexp(1.0, -25)) {
      ++g_fail;
      printf("FAIL g4_subnormal_abs_bound x=%.17g hi=0x%04x lo=0x%04x "
             "abs_err=%.17g\n",
             x, (unsigned)p.hi, (unsigned)p.lo, err);
    }
  }
  printf("split_subnormal_sweep=count=%ld max_rel=%.6e max_abs=%.6e "
         "lo_zero_count=%ld bound_break_count=%ld bound_break_min_mag=%.6e\n",
         count, max_rel, max_abs, lo_zero, bound_break,
         have_break ? bound_break_min_mag : 0.0);
}

/* ---------------- G5 + G6: GEMM synthesis ---------------- */

#define SYN_MAXN 64

static double gW[SYN_MAXN * SYN_MAXN];
static double gX[SYN_MAXN * SYN_MAXN];
static double gYt[SYN_MAXN * SYN_MAXN];
static double gAhi[SYN_MAXN * SYN_MAXN];
static double gAlo[SYN_MAXN * SYN_MAXN];
static double gWhi[SYN_MAXN * SYN_MAXN];
static double gWlo[SYN_MAXN * SYN_MAXN];
static double gA16[SYN_MAXN * SYN_MAXN];
static double gW16[SYN_MAXN * SYN_MAXN];
static double gYd[SYN_MAXN * SYN_MAXN];  /* double-combined B result */
static double gYf[SYN_MAXN * SYN_MAXN];  /* float-combined B result */

/* Relative error of Y vs the double oracle Yt, floor on |Yt| to avoid
 * division explosions on exact cancellations. */
static void rel_stats(const double *Y, const double *Yt, long elems,
                      double *mean_out, double *max_out) {
  double sum = 0.0, mx = 0.0;
  long i;
  for (i = 0; i < elems; ++i) {
    double den = fabs(Yt[i]);
    double rel;
    if (den < 1e-12) den = 1e-12;
    rel = fabs(Y[i] - Yt[i]) / den;
    sum += rel;
    if (rel > mx) mx = rel;
  }
  *mean_out = elems > 0 ? sum / (double)elems : 0.0;
  *max_out = mx;
}

static void run_synthesis(int n, double *plain_max_out,
                          double *b3sep_max_out) {
  const long nn = (long)n * (long)n;
  double mean, mx;
  int i, j, k;

  g_rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)n;
  for (i = 0; i < nn; ++i) {
    gW[i] = 4.0 * rng_unit() - 2.0;
    gX[i] = 4.0 * rng_unit() - 2.0;
  }
  /* Plant a few mantissa-hard entries inside [-2,2]. */
  gW[0] = 1.0 + 0x1p-11 - 0x1p-23;      /* reviewer example */
  gX[0] = 1.0 + 0x1p-11 + 0x1p-23 + 0x1p-24;
  gW[1] = -(2.0 - 0x1p-13);
  gX[1] = -(1.0 + 0x1p-11 - 0x1p-23);
  gW[2] = 0x1.001ffep+0;
  gX[2] = 0x1.fffcp-1;

  /* Double oracle Yt = X * W (same operand order as the simulated
   * paths below; matmul is not commutative). */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      double acc = 0.0;
      for (k = 0; k < n; ++k) {
        acc += gX[(long)i * n + k] * gW[(long)k * n + j];
      }
      gYt[(long)i * n + j] = acc;
    }
  }

  /* Splits (double arithmetic) and plain RN-FP16 roundings. */
  for (i = 0; i < nn; ++i) {
    fp16_pair pa = split_double(gX[i]);
    fp16_pair pw = split_double(gW[i]);
    gAhi[i] = fp16_to_double(pa.hi);
    gAlo[i] = fp16_to_double(pa.lo);
    gWhi[i] = fp16_to_double(pw.hi);
    gWlo[i] = fp16_to_double(pw.lo);
    gA16[i] = fp16_to_double(double_to_fp16(gX[i]));
    gW16[i] = fp16_to_double(double_to_fp16(gW[i]));
  }

  /* Plain-FP16 path: RN16 both operands, exact fp16*fp16 products,
   * single float accumulator (FAU FP32 accumulate emulation). */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(gA16[(long)i * n + k] * gW16[(long)k * n + j]);
      }
      gYd[(long)i * n + j] = (double)acc;
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_plain=mean=%.6e max=%.6e\n", n, mean, mx);
  *plain_max_out = mx;

  /* B-3GEMM, both operands split, three separate float accumulators,
   * combined in double. (T4=Alo*Wlo is omitted here and measured by
   * the b4_sep variant below.) */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
      for (k = 0; k < n; ++k) {
        t1 += (float)(gAhi[(long)i * n + k] * gWhi[(long)k * n + j]);
        t2 += (float)(gAhi[(long)i * n + k] * gWlo[(long)k * n + j]);
        t3 += (float)(gAlo[(long)i * n + k] * gWhi[(long)k * n + j]);
      }
      gYd[(long)i * n + j] = (double)t1 + (double)t2 + (double)t3;
      gYf[(long)i * n + j] =
          (double)(float)((float)((float)t1 + t2) + t3);
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_b3_sep=mean=%.6e max=%.6e\n", n, mean, mx);
  *b3sep_max_out = mx;
  {
    double cmean, cmx, d;
    long idx;
    double sum = 0.0, dmax = 0.0;
    for (idx = 0; idx < nn; ++idx) {
      double den = fabs(gYt[idx]);
      if (den < 1e-12) den = 1e-12;
      d = fabs(gYf[idx] - gYd[idx]) / den;
      sum += d;
      if (d > dmax) dmax = d;
    }
    cmean = nn > 0 ? sum / (double)nn : 0.0;
    cmx = dmax;
    printf("synth_combine_loss_n%d=mean=%.6e max=%.6e\n", n, cmean, cmx);
  }

  /* B-4GEMM: additionally T4 = Alo*Wlo, separate accumulators. */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f;
      for (k = 0; k < n; ++k) {
        t1 += (float)(gAhi[(long)i * n + k] * gWhi[(long)k * n + j]);
        t2 += (float)(gAhi[(long)i * n + k] * gWlo[(long)k * n + j]);
        t3 += (float)(gAlo[(long)i * n + k] * gWhi[(long)k * n + j]);
        t4 += (float)(gAlo[(long)i * n + k] * gWlo[(long)k * n + j]);
      }
      gYd[(long)i * n + j] =
          (double)t1 + (double)t2 + (double)t3 + (double)t4;
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_b4_sep=mean=%.6e max=%.6e\n", n, mean, mx);

  /* B-3GEMM chained into ONE float accumulator (same-ACC model):
   * all T1 terms, then all T2, then all T3, one final double read. */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(gAhi[(long)i * n + k] * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(gAhi[(long)i * n + k] * gWlo[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(gAlo[(long)i * n + k] * gWhi[(long)k * n + j]);
      }
      gYd[(long)i * n + j] = (double)acc;
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_b3_chain=mean=%.6e max=%.6e\n", n, mean, mx);

  /* W-only split: activations are already RN-FP16 in today's pipeline,
   * so Alo = 0 and Ahi = RN16(X). Only the weights carry a lo part.
   * Separate float accumulators, double combine. */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float t1 = 0.0f, t2 = 0.0f;
      for (k = 0; k < n; ++k) {
        t1 += (float)(gA16[(long)i * n + k] * gWhi[(long)k * n + j]);
        t2 += (float)(gA16[(long)i * n + k] * gWlo[(long)k * n + j]);
      }
      gYd[(long)i * n + j] = (double)t1 + (double)t2;
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_wonly_sep=mean=%.6e max=%.6e\n", n, mean, mx);

  /* W-only split chained into one float accumulator. */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(gA16[(long)i * n + k] * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(gA16[(long)i * n + k] * gWlo[(long)k * n + j]);
      }
      gYd[(long)i * n + j] = (double)acc;
    }
  }
  rel_stats(gYd, gYt, nn, &mean, &mx);
  printf("synth_n%d_wonly_chain=mean=%.6e max=%.6e\n", n, mean, mx);

  g_checks += 3 * nn; /* plain / b3_sep element comparisons asserted-ish */
}

static void g5_g6_synthesis(void) {
  int ok16, ok64;
  double p16 = 0.0, b16 = 0.0, p64 = 0.0, b64 = 0.0;

  run_synthesis(16, &p16, &b16);
  run_synthesis(64, &p64, &b64);
  printf("synth_n16_plain_vs_b3sep=plain_max_over_b3sep_max=%.6e\n",
         b16 > 0.0 ? p16 / b16 : 0.0);
  printf("synth_n64_plain_vs_b3sep=plain_max_over_b3sep_max=%.6e\n",
         b64 > 0.0 ? p64 / b64 : 0.0);

  /* Hard assertion: with BOTH operands split, the max relative error
   * must beat plain FP16 (expected ~2 decades). */
  ok16 = b16 < p16;
  ok64 = b64 < p64;
  printf("synth_both_split_beats_plain=%d\n", (ok16 && ok64) ? 1 : 0);
  ++g_checks;
  if (!(ok16 && ok64)) {
    ++g_fail;
    printf("FAIL synth_both_split_beats_plain n16_plain=%.6e "
           "n16_b3sep=%.6e n64_plain=%.6e n64_b3sep=%.6e\n",
           p16, b16, p64, b64);
  }
}

int main(void) {
  g1_g3_recovery();
  g2_table();
  g4_subnormal_sweep();
  g5_g6_synthesis();
  if (g_fail == 0) {
    printf("fp16x2_split_sim=PASS cases=%ld\n", g_checks);
    return 0;
  }
  printf("fp16x2_split_sim=FAIL failures=%ld cases=%ld\n", g_fail, g_checks);
  return 1;
}
