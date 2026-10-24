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
//   G7  stored-domain comparison (Stage E-3A, reviewer-mandated): the
//       G5 synthesis paths re-measured WITH the HMX store step — every
//       result value is committed to FP16 (RN16, i.e.
//       fp16_to_double(double_to_fp16(v))), exactly as the V81 path's
//       `cvt.hf = acc` does, and the STORED values are compared against
//       the same double oracle (ABSOLUTE errors are primary). Without
//       the store, the G5/G6 numbers are pre-store CPU figures only.
//       Two activation regimes per n in {16,64} on the SAME fixed
//       tensors/seeds: fp32carried (both operands split, as G5) and
//       fp16exact (activations exactly FP16, Alo = 0, the G5 W-only
//       configuration). Plus pinned scalar fixtures (asserted) and the
//       b2_cpu_gate decision line.
//   G8  tile-domain E-3C0 (Phase E-3C0, CPU-only design/oracle/
//       interface gate BEFORE any device work): one HMX tile
//       32 spatial x 32 out-ch x 32 in-ch (PRM v81 §2.2 geometry,
//       out[s][o] = sum_i A[s][i]*W[o][i], reduction over input
//       channel). The ACC chain is modeled twice: chained float
//       (primary device approximation, consistent with every measured
//       law so far) and chained double (`*_dblbr` secondary bracket
//       for the 37-bit FP cell pair). The final convert is ONE
//       bias-0 convert with scale 64 (0x5400): stored = RN16(64*acc),
//       applied IDENTICALLY to plain / b3 / wonly; de-scaling
//       comparisons use stored/64 vs the pre-scale double oracle.
//       Spatial lanes s0..s3 are pre-registered discriminating case
//       classes (positive split-sensitive / negative final store /
//       B-2 anchor + near-zero cancellation / subnormal-edge store);
//       lanes 4..31 and out-ch 4..27 are seeded realistic data.
//       Negative final store is UNMEASURED on device: the CPU model
//       predicts signed RN16 and the clamp-to-zero alternative
//       (law-4 OUTPUT-BIAS behavior) is printed too; only the CPU
//       signed-RN16 helper is asserted against pinned bits (E-3C1
//       device run is the discriminator). Includes the
//       RN16(64*x)/64 grid-preservation invariant test with the
//       breaking bands printed explicitly, pinned cell-bit asserts,
//       the device-output->cell mapping table under two layout
//       hypotheses (L1 dst[32*s+o] vs L2 dst[32*o+s]) with a
//       distinctness assert, per-class/tile metrics, and the
//       e3c0_cpu_gate decision line.
//
// Exit status is 0 only if no FAIL line was emitted. Hard FAILs are the
// mathematically proven G1 invariants (when lo is a normal FP16,
// |recover-x| <= 2^-23*|x|; when lo is subnormal/zero, |recover-x| <=
// 2^-25 absolute), the G5 assertion that both-operands-split beats
// plain FP16, the G7 pinned-fixture bit asserts (exact
// mathematical identities: a mismatch means the simulator or the
// shared oracle is wrong), and the G8 E-3C0 pinned cell-bit asserts +
// layout-mapping distinctness asserts (exact dyadic fixture
// identities: a mismatch means the simulator model or the shared
// oracle is wrong). Everything else — the README's
// unconditional 2^-23 bound, ALL G3 float-variant numbers, and ALL
// other G7/G8 numbers (cells, metrics, grid bands, gates) — is
// REPORT-ONLY (see the asserted-vs-reported comment inside
// g1_g3_recovery). The README's
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

/* Deterministic synthesis inputs shared by G5/G6 and G7: reseeds the
 * PRNG with the same per-n seed, fills the SAME fixed tensors (uniform
 * [-2,2] plus the planted mantissa-hard entries), builds the double
 * oracle Yt = X * W (same operand order as the simulated paths; matmul
 * is not commutative) and the split/plain-FP16 operand arrays.
 * Extracted verbatim from run_synthesis so both groups measure
 * bit-identical inputs. */
static void syn_build_inputs(int n) {
  const long nn = (long)n * (long)n;
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

  /* Double oracle Yt = X * W. */
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
}

static void run_synthesis(int n, double *plain_max_out,
                          double *b3sep_max_out) {
  const long nn = (long)n * (long)n;
  double mean, mx;
  int i, j, k;

  syn_build_inputs(n);

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

/* ---------------- G7: stored-domain comparison (Stage E-3A) ----------------
 *
 * Reviewer (PR #14 comment 6293114616): the G5/G6 figures are
 * PRE-STORE CPU numbers — the real V81 path commits `cvt.hf = acc`,
 * i.e. every accumulator result is rounded to FP16 before it leaves the
 * MAC. G7 re-measures the SAME fixed tensors (syn_build_inputs, same
 * seeds, same double oracle) with the store applied, and compares the
 * STORED values against the oracle. ABSOLUTE errors are primary;
 * relative stats use an explicit near-zero floor (elements with
 * |ref| < G7_REL_FLOOR are excluded from the relative pool and counted
 * in near_zero_count instead — never max(|ref|, eps) denominators).
 *
 * Regimes (activation side only; the weight split is always
 * Whi/Wlo = split_double(W)):
 *   fp32carried  Ahi/Alo = split_double(X)      (both operands split)
 *   fp16exact    Ahi = RN16(X), Alo = 0         (activations exactly FP16;
 *                                                = the G5 W-only config)
 * plain is the plain FP16 GEMM baseline (RN16 operands, exact products,
 * one chained float accumulator) and is therefore identical in both
 * regimes — it is the yardstick the regime's b3 is measured against.
 *
 * REFERENCES (reviewer correction, E-3C0 item 4): every cell is scored
 * against the end-to-end oracle gYt = X_original * W_original — the
 * UN-suffixed fields, semantics byte-stable vs the historical lines.
 * The fp16exact cells are ADDITIONALLY scored against the
 * kernel-given-input reference g7k = RN16(X_original) *
 * RN16(W_original), the exact product of the operands the kernel
 * actually receives — fields suffixed _vskref. The fp32carried cells
 * OMIT the _vskref fields (chosen for line uniformity + byte-stable
 * history): there the kernel's given operands are the hi/lo SPLIT
 * pairs, whose exact reconstruction equals gYt up to ~2^-23 split
 * residuals, so there is no single RN16-collapsed operand set to score
 * against.
 *
 * ASSERTED in G7 (a break FAILs the run): only the pinned scalar
 * fixture bits below — exact mathematical identities. Every cell
 * metric and the b2_cpu_gate line are REPORT-ONLY. */

#define G7_REL_FLOOR 0.05 /* min |oracle| pooled into relative stats */
#define G7_TIE_TOL 1e-9   /* abs-error delta below this counts as a tie */

static double gY7p[SYN_MAXN * SYN_MAXN];   /* plain internal (pre-store) */
static double gY7b3[SYN_MAXN * SYN_MAXN];  /* b3 single-ACC internal */
static double gY7b3s[SYN_MAXN * SYN_MAXN]; /* b3 sep-ACC + float combine */
static double gY7b4[SYN_MAXN * SYN_MAXN];  /* b4 single-ACC internal */
static double gY7k[SYN_MAXN * SYN_MAXN];   /* kernel-given-input ref
                                            * (fp16exact cells only):
                                            * RN16(X) * RN16(W) */
static double g7_err[SYN_MAXN * SYN_MAXN]; /* abs-error scratch (sorted) */
static long g7_improved_cells = 0;

/* The store step the V81 path performs: cvt.hf = acc commits the float
 * accumulator to FP16. Everything "stored" below goes through this. */
static double store16(double v) { return fp16_to_double(double_to_fp16(v)); }

static int cmp_double_asc(const void *pa, const void *pb) {
  double a = *(const double *)pa;
  double b = *(const double *)pb;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/* mean/median/p95/max/rmse of the abs errors in err[], sorted in place.
 * p95 = value at 1-based rank ceil(0.95*nn); median = average of the
 * two central values for even nn. Both deterministic. */
static void abs_error_stats(double *err, long nn, double *mean,
                            double *median, double *p95, double *max,
                            double *rmse) {
  double sum = 0.0, sq = 0.0, mx = 0.0;
  long i, rank;
  for (i = 0; i < nn; ++i) {
    double e = err[i];
    sum += e;
    sq += e * e;
    if (e > mx) mx = e;
  }
  qsort(err, (size_t)nn, sizeof(double), cmp_double_asc);
  *mean = sum / (double)nn;
  *rmse = sqrt(sq / (double)nn);
  *max = mx;
  *median = (nn % 2 != 0) ? err[nn / 2]
                          : 0.5 * (err[nn / 2 - 1] + err[nn / 2]);
  rank = (long)ceil(0.95 * (double)nn);
  if (rank < 1) rank = 1;
  if (rank > nn) rank = nn;
  *p95 = err[rank - 1];
}

static double stored_mean_abs_vs(const double *Y, const double *ref,
                                 long nn) {
  double sum = 0.0;
  long i;
  for (i = 0; i < nn; ++i) sum += fabs(store16(Y[i]) - ref[i]);
  return sum / (double)nn;
}

/* Pinned scalar fixtures (reviewer table). Only the EXPECTED bits are
 * literals; every value is derived through the same helpers the
 * synthesis uses (split_double / double_to_fp16 / fp16_to_double and
 * the G5 accumulation order: all T1, then all T2, then all T3, one
 * retained float ACC, one final store). A mismatch is a simulator bug
 * -> FAIL. */
static void b2_fixtures(void) {
  static const struct {
    const char *name;
    double a;
    uint16_t ahi_exp, alo_exp, plain_exp, b3_exp, full_exp;
  } fx[] = {
      /* case 0 "positive": A = 1+2^-11 is the exact RN16 tie between
       * 1.0 (even mantissa) and 1+2^-10 -> hi = 0x3c00, lo = +2^-11.
       * plain: (1.0*1.0)*64 = 64.0 -> 0x5400.
       * b3: 64*(1 + 2^-11 + 2^-11) = 64.0625 -> 0x5401.
       * full product 64*(1+2^-11)^2 = 64.0625152587890625 -> also
       * 0x5401: this coincidence is fixture-specific (the dropped
       * Alo*Wlo = 64*2^-22 is far below the 2^-4 store ulp) — do NOT
       * generalise "dropping lo*lo never costs a store ulp". */
      {"b2_case0", 1.0 + 0x1p-11, 0x3c00, 0x1000, 0x5400, 0x5401, 0x5401},
      /* case 1 "cancellation": A = 1-2^-12 is the exact RN16 tie
       * between 1-2^-10 and 1.0 -> hi = 0x3c00, lo = -2^-12 (0x8c00).
       * plain: 64.0 -> 0x5400. b3: 64*(1-2^-11) = 63.96875 -> 0x53ff.
       * full: 64*(1-2^-12)^2 = 63.968753814697265625 -> 0x53ff. */
      {"b2_case1", 1.0 - 0x1p-12, 0x3c00, 0x8c00, 0x5400, 0x53ff, 0x53ff},
  };
  size_t t;
  long mismatches = 0;

  for (t = 0; t < sizeof fx / sizeof fx[0]; ++t) {
    double a = fx[t].a;
    fp16_pair p = split_double(a); /* A-hi/lo AND W-hi/lo (W := A) */
    uint16_t a16 = double_to_fp16(a); /* plain-FP16 operand */
    float plain_acc = 0.0f, b3_acc = 0.0f;
    uint16_t plain_stored, b3_stored, full_rn16;
    double scaled_full;
    int it;

    for (it = 0; it < 64; ++it) {
      plain_acc += (float)(fp16_to_double(a16) * fp16_to_double(a16));
    }
    for (it = 0; it < 64; ++it) { /* T1 = Ahi*Whi */
      b3_acc += (float)(fp16_to_double(p.hi) * fp16_to_double(p.hi));
    }
    for (it = 0; it < 64; ++it) { /* T2 = Ahi*Wlo */
      b3_acc += (float)(fp16_to_double(p.hi) * fp16_to_double(p.lo));
    }
    for (it = 0; it < 64; ++it) { /* T3 = Alo*Whi */
      b3_acc += (float)(fp16_to_double(p.lo) * fp16_to_double(p.hi));
    }
    plain_stored = double_to_fp16((double)plain_acc);
    b3_stored = double_to_fp16((double)b3_acc);
    scaled_full = 64.0 * a * a; /* exact in double for both fixtures */
    full_rn16 = double_to_fp16(scaled_full);

    printf("split_b2_fixture name=%s a=%.17g ahi_bits=0x%04x "
           "alo_bits=0x%04x plain_stored=0x%04x b3_stored=0x%04x "
           "full_rn16=0x%04x scaled_full=%.17g\n",
           fx[t].name, a, (unsigned)p.hi, (unsigned)p.lo,
           (unsigned)plain_stored, (unsigned)b3_stored,
           (unsigned)full_rn16, scaled_full);

#define G7_FIXTURE_CHECK(field, got, want)                            \
  do {                                                                \
    ++g_checks;                                                       \
    if ((got) != (want)) {                                            \
      ++g_fail;                                                       \
      ++mismatches;                                                   \
      printf("FAIL b2_fixture name=%s field=" #field " got=0x%04x "   \
             "want=0x%04x\n",                                         \
             fx[t].name, (unsigned)(got), (unsigned)(want));          \
    }                                                                 \
  } while (0)
    G7_FIXTURE_CHECK(ahi_bits, p.hi, fx[t].ahi_exp);
    G7_FIXTURE_CHECK(alo_bits, p.lo, fx[t].alo_exp);
    G7_FIXTURE_CHECK(plain_stored, plain_stored, fx[t].plain_exp);
    G7_FIXTURE_CHECK(b3_stored, b3_stored, fx[t].b3_exp);
    G7_FIXTURE_CHECK(full_rn16, full_rn16, fx[t].full_exp);
#undef G7_FIXTURE_CHECK
  }
  printf("split_b2_fixture_mismatch=%ld\n", mismatches);
}

static void g7_cell(int n, int carry32) {
  const long nn = (long)n * (long)n;
  const char *regime = carry32 ? "fp32carried" : "fp16exact";
  /* stored-domain stats */
  double ps_mean, ps_median, ps_p95, ps_max, ps_rmse;
  double bs_mean, bs_median, bs_p95, bs_max, bs_rmse;
  /* internal (pre-store) means + secondary variants */
  double pi_mean = 0.0, bi_mean = 0.0;
  double b3sep_int_mean, b3sep_st_mean, b4_int_mean, b4_st_mean;
  long improved = 0, equal = 0, worsened = 0;
  long bit_diff = 0, plain_opt = 0, b3_opt = 0;
  long near_zero = 0, pooled = 0;
  double rel_sum = 0.0, rel_mean;
  /* kernel-given-input reference stats (fp16exact cells only) */
  double pk_mean = 0.0, bk_mean = 0.0;
  long pk_opt = 0, bk_opt = 0, k_improved = 0, k_equal = 0, k_worsened = 0;
  int i, j, k;

  syn_build_inputs(n); /* SAME tensors / oracle / splits as G5 */

  /* Plain-FP16 baseline (identical in both regimes: the plain GEMM
   * always consumes RN16 activations). */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(gA16[(long)i * n + k] * gW16[(long)k * n + j]);
      }
      gY7p[(long)i * n + j] = (double)acc;
    }
  }

  /* B3 device model: THREE products chained into ONE retained float
   * ACC (all T1, then all T2, then all T3), one final store. In the
   * fp16exact regime Alo = 0 so the T3 loop adds exact zeros. */
#define G7_AHI(r) (carry32 ? gAhi[(r)] : gA16[(r)])
#define G7_ALO(r) (carry32 ? gAlo[(r)] : 0.0)
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_AHI((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_AHI((long)i * n + k) * gWlo[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_ALO((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      gY7b3[(long)i * n + j] = (double)acc;
    }
  }

  /* Secondary: separate 3 accumulators + float combine (G5's b3_sep). */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
      for (k = 0; k < n; ++k) {
        t1 += (float)(G7_AHI((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        t2 += (float)(G7_AHI((long)i * n + k) * gWlo[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        t3 += (float)(G7_ALO((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      gY7b3s[(long)i * n + j] =
          (double)(float)((float)((float)t1 + t2) + t3);
    }
  }

  /* B4: additionally T4 = Alo*Wlo, single retained ACC (same-ACC
   * model extended by one product). */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_AHI((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_AHI((long)i * n + k) * gWlo[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_ALO((long)i * n + k) * gWhi[(long)k * n + j]);
      }
      for (k = 0; k < n; ++k) {
        acc += (float)(G7_ALO((long)i * n + k) * gWlo[(long)k * n + j]);
      }
      gY7b4[(long)i * n + j] = (double)acc;
    }
  }
#undef G7_AHI
#undef G7_ALO

  /* Internal (pre-store) mean abs vs the double oracle. */
  for (i = 0; i < nn; ++i) {
    pi_mean += fabs(gY7p[i] - gYt[i]);
    bi_mean += fabs(gY7b3[i] - gYt[i]);
  }
  pi_mean /= (double)nn;
  bi_mean /= (double)nn;
  b3sep_int_mean = 0.0;
  b4_int_mean = 0.0;
  for (i = 0; i < nn; ++i) {
    b3sep_int_mean += fabs(gY7b3s[i] - gYt[i]);
    b4_int_mean += fabs(gY7b4[i] - gYt[i]);
  }
  b3sep_int_mean /= (double)nn;
  b4_int_mean /= (double)nn;

  /* STORED-domain absolute-error stats (PRIMARY): plain and b3. */
  for (i = 0; i < nn; ++i) g7_err[i] = fabs(store16(gY7p[i]) - gYt[i]);
  abs_error_stats(g7_err, nn, &ps_mean, &ps_median, &ps_p95, &ps_max,
                  &ps_rmse);
  for (i = 0; i < nn; ++i) g7_err[i] = fabs(store16(gY7b3[i]) - gYt[i]);
  abs_error_stats(g7_err, nn, &bs_mean, &bs_median, &bs_p95, &bs_max,
                  &bs_rmse);
  b3sep_st_mean = stored_mean_abs_vs(gY7b3s, gYt, nn);
  b4_st_mean = stored_mean_abs_vs(gY7b4, gYt, nn);

  /* Per-element STORED pairwise: raw-bit disagreement, oracle-optimal
   * bits (RN16(oracle) — the best any FP16 output can do), and
   * improved/equal/worsened on absolute errors with the G7_TIE_TOL tie
   * policy (|delta| <= tol counts as equal). Relative stats pooled only
   * over |oracle| >= G7_REL_FLOOR. */
  for (i = 0; i < nn; ++i) {
    uint16_t pb = double_to_fp16(gY7p[i]);
    uint16_t bb = double_to_fp16(gY7b3[i]);
    uint16_t ob = double_to_fp16(gYt[i]);
    double ap = fabs(fp16_to_double(pb) - gYt[i]);
    double ab = fabs(fp16_to_double(bb) - gYt[i]);
    if (pb != bb) ++bit_diff;
    if (pb == ob) ++plain_opt;
    if (bb == ob) ++b3_opt;
    if (ap - ab > G7_TIE_TOL) {
      ++improved;
    } else if (ab - ap > G7_TIE_TOL) {
      ++worsened;
    } else {
      ++equal;
    }
    if (fabs(gYt[i]) >= G7_REL_FLOOR) {
      rel_sum += ab / fabs(gYt[i]);
      ++pooled;
    } else {
      ++near_zero;
    }
  }
  rel_mean = pooled > 0 ? rel_sum / (double)pooled : 0.0;

  /* Kernel-given-input reference (reviewer correction): for the
   * fp16exact regime the kernel receives RN16(X) as the activation
   * operand, so score the SAME stored values additionally against the
   * exact double product RN16(X) * RN16(W). Same tie policy as above.
   * fp32carried cells skip this (no _vskref fields, see the G7 section
   * comment). */
  if (!carry32) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < n; ++j) {
        double acc = 0.0;
        for (k = 0; k < n; ++k) {
          acc += gA16[(long)i * n + k] * gW16[(long)k * n + j];
        }
        gY7k[(long)i * n + j] = acc;
      }
    }
    pk_mean = stored_mean_abs_vs(gY7p, gY7k, nn);
    bk_mean = stored_mean_abs_vs(gY7b3, gY7k, nn);
    for (i = 0; i < nn; ++i) {
      uint16_t pb = double_to_fp16(gY7p[i]);
      uint16_t bb = double_to_fp16(gY7b3[i]);
      double ap = fabs(fp16_to_double(pb) - gY7k[i]);
      double ab = fabs(fp16_to_double(bb) - gY7k[i]);
      if (pb == double_to_fp16(gY7k[i])) ++pk_opt;
      if (bb == double_to_fp16(gY7k[i])) ++bk_opt;
      if (ap - ab > G7_TIE_TOL) {
        ++k_improved;
      } else if (ab - ap > G7_TIE_TOL) {
        ++k_worsened;
      } else {
        ++k_equal;
      }
    }
  }

  /* CPU decision basis: does STORED b3 beat STORED plain in mean abs? */
  if (bs_mean < ps_mean) ++g7_improved_cells;

  printf("split_b2_stored=n%d_%s regime=%s n=%d "
         "plain_int_mean_abs=%.6e "
         "plain_stored_mean_abs=%.6e plain_stored_median_abs=%.6e "
         "plain_stored_p95_abs=%.6e plain_stored_max_abs=%.6e "
         "plain_stored_rmse=%.6e b3_int_mean_abs=%.6e "
         "b3_stored_mean_abs=%.6e b3_stored_median_abs=%.6e "
         "b3_stored_p95_abs=%.6e b3_stored_max_abs=%.6e "
         "b3_stored_rmse=%.6e "
         "b3_stored_vs_plain_stored_improved=%.6e equal=%.6e worsened=%.6e "
         "tie_tol=%.6e stored_bit_diff_n=%ld stored_bit_diff_frac=%.6e "
         "plainoptimal_bits_frac=%.6e b3optimal_bits_frac=%.6e "
         "rel_floor=%.6e b3_stored_mean_rel_ge_floor=%.6e "
         "near_zero_count=%ld "
         "b3sep_int_mean_abs=%.6e b3sep_stored_mean_abs=%.6e "
         "b4_int_mean_abs=%.6e b4_stored_mean_abs=%.6e",
         n, regime, regime, n, pi_mean, ps_mean, ps_median, ps_p95, ps_max,
         ps_rmse, bi_mean, bs_mean, bs_median, bs_p95, bs_max, bs_rmse,
         (double)improved / (double)nn, (double)equal / (double)nn,
         (double)worsened / (double)nn, G7_TIE_TOL, bit_diff,
         (double)bit_diff / (double)nn, (double)plain_opt / (double)nn,
         (double)b3_opt / (double)nn, G7_REL_FLOOR, rel_mean, near_zero,
         b3sep_int_mean, b3sep_st_mean, b4_int_mean, b4_st_mean);
  /* Appended _vskref fields: fp16exact cells only (fp32carried omits
   * them — see the G7 section comment; its line stays byte-identical
   * to the historical format). Two printfs, no newline between them,
   * so the cell remains ONE grep-able line. */
  if (!carry32) {
    printf(" plain_stored_mean_abs_vskref=%.6e "
           "b3_stored_mean_abs_vskref=%.6e "
           "plainoptimal_bits_vskref_frac=%.6e "
           "b3optimal_bits_vskref_frac=%.6e "
           "b3_stored_vs_plain_stored_improved_vskref=%.6e "
           "equal_vskref=%.6e worsened_vskref=%.6e",
           pk_mean, bk_mean, (double)pk_opt / (double)nn,
           (double)bk_opt / (double)nn, (double)k_improved / (double)nn,
           (double)k_equal / (double)nn, (double)k_worsened / (double)nn);
  }
  printf("\n");
}

static void g7_stored_domain(void) {
  static const int ns[2] = {16, 64};
  int ni, ri;

  printf("g7_reference_note=fp16exact_cells_report_end_to_end_ref_and_"
         "kernel_given_input_ref_rn16x_mul_rn16w\n");
  b2_fixtures();
  for (ni = 0; ni < 2; ++ni) {
    for (ri = 0; ri < 2; ++ri) {
      g7_cell(ns[ni], ri == 0);
    }
  }
  /* Reviewer decision line (E-3C0 item 4): k of 4 (n, regime) cells
   * where STORED b3 mean abs error < STORED plain mean abs error.
   * Threshold is explicit and a PARTIAL result is NEVER called PASS:
   * PASS requires all 4 cells. Report-only. */
  if (g7_improved_cells == 4) {
    printf("b2_cpu_gate=PASS improves=4/4 threshold=all_4_cells "
           "basis=stored_b3_mean_abs_lt_plain_stored_mean_abs\n");
  } else if (g7_improved_cells == 0) {
    printf("b2_cpu_gate=NO_IMPROVE improves=0/4 threshold=all_4_cells "
           "basis=stored_b3_mean_abs_lt_plain_stored_mean_abs\n");
  } else {
    printf("b2_cpu_gate=PARTIAL improves=%ld/4 threshold=all_4_cells "
           "basis=stored_b3_mean_abs_lt_plain_stored_mean_abs\n",
           g7_improved_cells);
  }
}

/* ==================================================================
 * G8: tile-domain E-3C0 (Phase E-3C0, CPU-only gate BEFORE device)
 * ==================================================================
 *
 * Geometry (HMX PRM v81 §2.2, recon-verified — encoded, nothing new):
 *   activation crouton = 32 spatial x 32 in-ch  (A[s][i])
 *   weight              = 32 out-ch  x 32 in-ch  (W[o][i])
 *   output              = 32 spatial x 32 out-ch
 *   cell: out[s][o] = sum_{i=0..31} A[s][i] * W[o][i]   (reduction over
 *   input channel, orientation A * W^T).
 *
 * Accumulator modeled twice:
 *   primary  = ONE chained FLOAT accumulator (device approximation
 *              consistent with every measured law so far): plain over
 *              32 products; b3 over 96 products in three passes
 *              (all Ahi*Whi, then all Ahi*Wlo, then all Alo*Whi) in
 *              the SAME chain; wonly (Alo == 0) over 64 products in
 *              two passes in one chain.
 *   bracket  = the same b3 pass order carried in DOUBLE
 *              (`*_dblbr` fields/line: 37-bit FP cell pair upper
 *              bracket — a bracket, not a device claim).
 *
 * Final convert (mode-41 measured law): ONE bias-0 convert, scale 64
 * (0x5400): stored_bits = RN16(64*acc) for every variant, identically.
 * De-scaled comparison value = fp16_to_double(bits)/64 vs the
 * pre-scale double oracle gT8Yt = sum_i X*W_truth.
 *
 * Negative final store is UNMEASURED on device. The CPU model here
 * predicts the ordinary SIGNED RN16 (exact -64 -> 0xd400, verified as
 * a helper pin below); the clamp-to-zero candidate (0x0000, law-4
 * OUTPUT-BIAS behavior) is printed alongside on the s1 line. The
 * device run E-3C1 is the discriminator — no device behavior is
 * asserted here.
 *
 * Layout note: element 0 of the stored 32x32 convert state is device
 * -verified as cell (s=0, o=0); the byte order of ALL other cells is
 * PRM-figure-derived only, so the readback index -> cell mapping is
 * printed under BOTH hypotheses (L1 dst[32*s+o] out-ch fastest vs L2
 * dst[32*o+s] out-ch major) and the mapped cells are asserted to have
 * DISTINCT oracle values so E-3C1 can discriminate layout from raw
 * bits. Do not assume CPU row-major maps to HMX.
 */

#define T8 32
#define T8_CELLS (T8 * T8)

static double gT8X[T8_CELLS];  /* activation truth (wider precision) */
static double gT8W[T8_CELLS];  /* weight truth */
static double gT8Yt[T8_CELLS]; /* double oracle sum_i X*W_truth */
static double gT8A16[T8_CELLS], gT8W16[T8_CELLS]; /* plain operands */
static double gT8Ahi[T8_CELLS], gT8Alo[T8_CELLS]; /* split activations */
static double gT8Whi[T8_CELLS], gT8Wlo[T8_CELLS]; /* split weights */
static double gT8P[T8_CELLS];  /* plain acc (float chain) */
static double gT8B[T8_CELLS];  /* b3 acc (float chain, 96 products) */
static double gT8Bd[T8_CELLS]; /* b3 acc (double chain, dblbr) */
static double gT8N[T8_CELLS];  /* wonly acc (float chain, 64 products) */
static uint16_t gT8Ps[T8_CELLS];  /* stored RN16(64*plain) */
static uint16_t gT8Bs[T8_CELLS];  /* stored RN16(64*b3) */
static uint16_t gT8Ds[T8_CELLS];  /* stored RN16(64*b3_dblbr) */
static uint16_t gT8Ns[T8_CELLS];  /* stored RN16(64*wonly) */
static uint16_t gT8Sp[T8_CELLS];  /* shape 0 - shape 4 split, plain acc */
static uint16_t gT8Sb[T8_CELLS];  /* shape 0 - shape 4 split, b3 acc */
static uint16_t gT8Os[T8_CELLS];  /* expected = RN16(64*oracle_true) */
static double gT8E[T8_CELLS];     /* abs-error scratch (sorted in place) */
static long gT8_pin_mismatch = 0;
static int gT8_anchor_ok = 1;
static int gT8_cond_a = 0, gT8_cond_d = 0;

static double t8_descale(uint16_t bits) {
  return fp16_to_double(bits) / 64.0;
}

static int t8_class_of(int s) { return s < 4 ? s : 4; }

static const char *t8_stored_class(uint16_t bits) {
  unsigned exp = ((unsigned)bits >> 10) & 0x1fu;
  unsigned man = (unsigned)bits & 0x3ffu;
  if (exp == 0u && man == 0u) return "zero";
  if (exp == 0u) return "subnormal";
  return "normal";
}

/* Deterministic tile: seeded realistic data (G5-like U(-2,2) over all
 * in-channels), then the pre-registered case-class lanes/rows are
 * overwritten with exact dyadic fixtures (hand-derived; every pinned
 * bit below was computed through the same helpers + this chain
 * order). Realistic lanes use all 32 in-ch; discriminating lanes may
 * activate fewer channels (zero channels contribute exact 0). */
static void t8_build(void) {
  int s, o, i;

  g_rng = 0xE3C0117E5EEDF00Dull;
  for (i = 0; i < T8_CELLS; ++i) {
    gT8W[i] = 4.0 * rng_unit() - 2.0;
    gT8X[i] = 4.0 * rng_unit() - 2.0;
  }

  /* Special W rows: out-ch 0..3 case-class anchors, 28..31 s3
   * subnormal-edge targets; out-ch 4..27 stay seeded realistic. */
  for (o = 0; o < T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < T8; ++i) gT8W[(long)o * T8 + i] = 0.0;
    }
  }
  gT8W[0 * T8 + 0] = 1.0 + 0x1p-11; /* positive split-sensitive row */
  gT8W[0 * T8 + 1] = 0x1p-11;
  gT8W[0 * T8 + 2] = 0x1p-13;
  gT8W[1 * T8 + 0] = 1.0 - 0x1p-12;  /* B-2 case-1 anchor row */
  gT8W[2 * T8 + 0] = 0x1p-9;         /* s3 boundary row (64*acc=2^-14) */
  gT8W[3 * T8 + 0] = 1.0;            /* near-zero cancellation row */
  gT8W[3 * T8 + 1] = 1.0 + 0x1p-12;
  gT8W[28 * T8 + 0] = 0x1p-10;
  gT8W[29 * T8 + 0] = 0x1p-10 - 0x1p-21;
  gT8W[30 * T8 + 0] = 3.0 * 0x1p-12;
  gT8W[31 * T8 + 0] = 0x1p-13;

  /* Special A lanes 0..3; lanes 4..31 stay seeded realistic. */
  for (s = 0; s <= 3; ++s)
    for (i = 0; i < T8; ++i) gT8X[(long)s * T8 + i] = 0.0;
  gT8X[0 * T8 + 0] = 1.0 + 0x1p-11; /* s0: positive multi-channel sum */
  gT8X[0 * T8 + 1] = 0x1p-11;
  gT8X[0 * T8 + 2] = 0x1p-13;
  for (i = 0; i < T8; ++i)          /* s1 = negation of s0 */
    gT8X[1 * T8 + i] = -gT8X[i];
  gT8X[2 * T8 + 0] = 1.0 - 0x1p-12; /* s2: B-2 case-1 + negated twin */
  gT8X[2 * T8 + 1] = -(1.0 - 0x1p-12);
  gT8X[3 * T8 + 0] = 0x1p-10;       /* s3: subnormal-edge lane */

  /* Double oracle out[s][o] = sum_i X[s][i] * W[o][i]. */
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      double acc = 0.0;
      for (i = 0; i < T8; ++i)
        acc += gT8X[(long)s * T8 + i] * gT8W[(long)o * T8 + i];
      gT8Yt[(long)s * T8 + o] = acc;
    }
  }

  /* Plain operands and (double-arithmetic) hi/lo splits. */
  for (i = 0; i < T8_CELLS; ++i) {
    fp16_pair pa = split_double(gT8X[i]);
    fp16_pair pw = split_double(gT8W[i]);
    gT8A16[i] = fp16_to_double(double_to_fp16(gT8X[i]));
    gT8W16[i] = fp16_to_double(double_to_fp16(gT8W[i]));
    gT8Ahi[i] = fp16_to_double(pa.hi);
    gT8Alo[i] = fp16_to_double(pa.lo);
    gT8Whi[i] = fp16_to_double(pw.hi);
    gT8Wlo[i] = fp16_to_double(pw.lo);
  }
}

/* Per-cell variants; store = ONE convert, scale 64, signed RN16. */
static void t8_compute(void) {
  int s, o, i;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      const long ai = (long)s * T8;
      const long wi = (long)o * T8;
      const long ci = ai + o;
      float p = 0.0f, b = 0.0f, w = 0.0f;
      double bd = 0.0;

      for (i = 0; i < T8; ++i) /* plain: RN16(X)*RN16(W), exact
                                * products, one float chain */
        p += (float)(gT8A16[ai + i] * gT8W16[wi + i]);

      for (i = 0; i < T8; ++i) { /* b3 pass 1: Ahi*Whi */
        b += (float)(gT8Ahi[ai + i] * gT8Whi[wi + i]);
        bd += gT8Ahi[ai + i] * gT8Whi[wi + i];
      }
      for (i = 0; i < T8; ++i) { /* pass 2: Ahi*Wlo */
        b += (float)(gT8Ahi[ai + i] * gT8Wlo[wi + i]);
        bd += gT8Ahi[ai + i] * gT8Wlo[wi + i];
      }
      for (i = 0; i < T8; ++i) { /* pass 3: Alo*Whi (one chain) */
        b += (float)(gT8Alo[ai + i] * gT8Whi[wi + i]);
        bd += gT8Alo[ai + i] * gT8Whi[wi + i];
      }

      for (i = 0; i < T8; ++i) /* wonly: activations not split */
        w += (float)(gT8A16[ai + i] * gT8Whi[wi + i]);
      for (i = 0; i < T8; ++i)
        w += (float)(gT8A16[ai + i] * gT8Wlo[wi + i]);

      gT8P[ci] = (double)p;
      gT8B[ci] = (double)b;
      gT8Bd[ci] = bd;
      gT8N[ci] = (double)w;
      gT8Ps[ci] = double_to_fp16(64.0 * (double)p);
      gT8Bs[ci] = double_to_fp16(64.0 * (double)b);
      gT8Ds[ci] = double_to_fp16(64.0 * bd);
      gT8Ns[ci] = double_to_fp16(64.0 * (double)w);
      gT8Os[ci] = double_to_fp16(64.0 * gT8Yt[ci]);
    }
  }
}

#define T8_PIN(name, cell_s, cell_o, field, got, want)                \
  do {                                                                \
    ++g_checks;                                                       \
    if ((got) != (want)) {                                            \
      ++g_fail;                                                       \
      ++gT8_pin_mismatch;                                             \
      printf("FAIL e3c0_pin name=%s cell=(s%d,o%d) field=" #field     \
             " got=0x%04x want=0x%04x\n",                             \
             (name), (cell_s), (cell_o), (unsigned)(got),             \
             (unsigned)(want));                                       \
    }                                                                 \
  } while (0)

/* Pinned cell bits. Expected literals were derived numerically first
 * (exact dyadics through this exact chain order), then asserted:
 *   (s0,o0) plain 1+2^-23 chain -> 64*(1+2^-22) rounds to 64.0 ->
 *     0x5400; b3 (pass1 drops the 2^-26 term) -> 64*(1+2^-10+2^-22)
 *     rounds to 64.0625 -> 0x5401 (plain != b3: discrimination cell).
 *   (s1,o0) = negation of (s0,o0): signed RN16 -> 0xd400 / 0xd401;
 *     the clamp-to-zero candidate (law-4 OUTPUT-BIAS) is 0x0000 for
 *     both and is printed, not asserted (device UNMEASURED).
 *   (s2,o1) EXACT B-2 case-1: plain acc=1 -> 0x5400, b3 acc=1-2^-11
 *     -> 63.96875 -> 0x53ff (device-confirmed anchor, cond_c).
 *   (s2,o3) near-zero cancellation: plain 1-1 = 0 -> 0x0000; b3
 *     -2^-12 -> RN16(-2^-6) -> 0xA400.
 *   (s3,o28..o31) subnormal-edge: 64*acc = 2^-14 (smallest normal),
 *     just below 2^-14 (rounds up), 3*2^-16 (subnormal), 2^-17
 *     (deep subnormal: 2^-17 = 128*2^-24 -> mantissa 128) ->
 *     0x0400 / 0x0400 / 0x0300 / 0x0080. */
static void t8_pins(void) {
  static const struct {
    const char *name;
    int s, o;
    uint16_t plain, b3;
  } pins[] = {
      {"s0_positive", 0, 0, 0x5400, 0x5401},
      {"s1_negative", 1, 0, 0xd400, 0xd401},
      {"s2_b2_anchor", 2, 1, 0x5400, 0x53ff},
      {"s2_nearzero", 2, 3, 0x0000, 0xa400},
      {"s3_edge_o28", 3, 28, 0x0400, 0x0400},
      {"s3_edge_o29", 3, 29, 0x0400, 0x0400},
      {"s3_edge_o30", 3, 30, 0x0300, 0x0300},
      {"s3_edge_o31", 3, 31, 0x0080, 0x0080},
  };
  size_t t;

  for (t = 0; t < sizeof pins / sizeof pins[0]; ++t) {
    const long ci = (long)pins[t].s * T8 + pins[t].o;
    T8_PIN(pins[t].name, pins[t].s, pins[t].o, plain_bits, gT8Ps[ci],
           pins[t].plain);
    T8_PIN(pins[t].name, pins[t].s, pins[t].o, b3_bits, gT8Bs[ci],
           pins[t].b3);
    if (pins[t].s == 2 && pins[t].o == 1) {
      if (gT8Ps[ci] != pins[t].plain || gT8Bs[ci] != pins[t].b3)
        gT8_anchor_ok = 0;
    }
    printf("e3c0_cell name=%s cell=(s%d,o%d) plain_bits=0x%04x "
           "b3_bits=0x%04x oracle=%.17g plain_descaled=%.17g "
           "b3_descaled=%.17g",
           pins[t].name, pins[t].s, pins[t].o, (unsigned)gT8Ps[ci],
           (unsigned)gT8Bs[ci], gT8Yt[ci], t8_descale(gT8Ps[ci]),
           t8_descale(gT8Bs[ci]));
    if (pins[t].s == 1) {
      /* Negative store: CPU prediction = signed RN16 (asserted above
       * via the helper), clamp-to-zero candidate printed alongside.
       * Device UNMEASURED -> E-3C1 discriminates. */
      printf(" store_model=signed_rn16 plain_clamp0=0x0000 "
             "b3_clamp0=0x0000");
    }
    printf("\n");
  }
#undef T8_PIN

  /* s3 edge documentation lines: stored band for each target cell. */
  {
    int o;
    for (o = 28; o <= 31; ++o) {
      const long ci = 3L * T8 + o;
      printf("e3c0_s3_edge o=%d acc=%.17g sixtyfour_acc=%.17g "
             "plain_bits=0x%04x b3_bits=0x%04x plain_class=%s "
             "b3_class=%s coincide=%d\n",
             o, gT8P[ci], 64.0 * gT8P[ci], (unsigned)gT8Ps[ci],
             (unsigned)gT8Bs[ci], t8_stored_class(gT8Ps[ci]),
             t8_stored_class(gT8Bs[ci]),
             gT8Ps[ci] == gT8Bs[ci] ? 1 : 0);
    }
  }
}

/* Device-output -> cell mapping under both layout hypotheses. The
 * mapped cells must have DISTINCT oracle values so a returned raw bit
 * pattern discriminates L1 vs L2 on device (E-3C1). Same-cell indices
 * (0, 33) cannot discriminate and are noted, not asserted. */
static void t8_map(void) {
  static const int idxs[8] = {0, 1, 2, 3, 32, 33, 64, 96};
  int k;
  for (k = 0; k < 8; ++k) {
    const int ix = idxs[k];
    const int s1 = ix / T8, o1 = ix % T8; /* L1: dst[32*s + o] */
    const int s2 = ix % T8, o2 = ix / T8; /* L2: dst[32*o + s] */
    const long c1 = (long)s1 * T8 + o1;
    const long c2 = (long)s2 * T8 + o2;

    printf("e3c0_map idx=%d hyp=L1 cell=(s%d,o%d) plain_bits=0x%04x "
           "b3_bits=0x%04x oracle=%.17g\n",
           ix, s1, o1, (unsigned)gT8Ps[c1], (unsigned)gT8Bs[c1],
           gT8Yt[c1]);
    printf("e3c0_map idx=%d hyp=L2 cell=(s%d,o%d) plain_bits=0x%04x "
           "b3_bits=0x%04x oracle=%.17g\n",
           ix, s2, o2, (unsigned)gT8Ps[c2], (unsigned)gT8Bs[c2],
           gT8Yt[c2]);
    if (c1 == c2) {
      printf("e3c0_map idx=%d same_cell=1 note=hypotheses_agree\n", ix);
      continue;
    }
    ++g_checks;
    if (gT8Yt[c1] == gT8Yt[c2]) {
      ++g_fail;
      ++gT8_pin_mismatch;
      printf("FAIL e3c0_map idx=%d distinct_oracle cells=(s%d,o%d)/"
             "(s%d,o%d) oracle=%.17g\n",
             ix, s1, o1, s2, o2, gT8Yt[c1]);
    }
  }
}

/* Full-tile + per-case-class stored-domain metrics vs the double
 * oracle. ABSOLUTE de-scaled errors are primary; relative errors are
 * pooled over |oracle| >= G7_REL_FLOOR with the near-zero count
 * reported separately (same policy as G7). Also: oracle-optimal bit
 * rate, improved/equal/worsened b3 vs plain (G7_TIE_TOL), and the
 * plain-vs-b3 raw stored-bit disagreement. Sets cond_a. */
static void t8_metrics(void) {
  const long nn = (long)T8 * T8;
  double pm, pmed, pp95, pmax, prmse;
  double bm, bmed, bp95, bmax, brmse;
  double dm, dmed, dp95, dmax, drmse;
  double wm, wmed, wp95, wmax, wrmse;
  long improved = 0, equal = 0, worsened = 0;
  long popt = 0, bopt = 0, dopt = 0, nopt = 0;
  long bitdiff = 0, near_zero = 0;
  double prel = 0.0, brel = 0.0;
  long prel_n = 0, brel_n = 0;
  long w_imp = 0, w_eq = 0, w_wor = 0, w_bitdiff = 0;
  long i;
  static const char *cname[5] = {"s0_positive", "s1_negative",
                                 "s2_mixed", "s3_edge", "realistic"};
  int c;

  for (i = 0; i < nn; ++i) {
    gT8E[i] = fabs(t8_descale(gT8Ps[i]) - gT8Yt[i]);
    if (gT8Ps[i] == gT8Os[i]) ++popt;
    if (gT8Bs[i] == gT8Os[i]) ++bopt;
    if (gT8Ds[i] == gT8Os[i]) ++dopt;
    if (gT8Ns[i] == gT8Os[i]) ++nopt;
    if (gT8Ps[i] != gT8Bs[i]) ++bitdiff;
    if (gT8Ps[i] != gT8Ns[i]) ++w_bitdiff;
  }
  abs_error_stats(gT8E, nn, &pm, &pmed, &pp95, &pmax, &prmse);

  for (i = 0; i < nn; ++i) gT8E[i] = fabs(t8_descale(gT8Bs[i]) - gT8Yt[i]);
  abs_error_stats(gT8E, nn, &bm, &bmed, &bp95, &bmax, &brmse);

  for (i = 0; i < nn; ++i) gT8E[i] = fabs(t8_descale(gT8Ds[i]) - gT8Yt[i]);
  abs_error_stats(gT8E, nn, &dm, &dmed, &dp95, &dmax, &drmse);

  for (i = 0; i < nn; ++i) gT8E[i] = fabs(t8_descale(gT8Ns[i]) - gT8Yt[i]);
  abs_error_stats(gT8E, nn, &wm, &wmed, &wp95, &wmax, &wrmse);

  for (i = 0; i < nn; ++i) {
    const double ap = fabs(t8_descale(gT8Ps[i]) - gT8Yt[i]);
    const double ab = fabs(t8_descale(gT8Bs[i]) - gT8Yt[i]);
    const double aw = fabs(t8_descale(gT8Ns[i]) - gT8Yt[i]);
    if (ap - ab > G7_TIE_TOL) {
      ++improved;
    } else if (ab - ap > G7_TIE_TOL) {
      ++worsened;
    } else {
      ++equal;
    }
    if (ap - aw > G7_TIE_TOL) {
      ++w_imp;
    } else if (aw - ap > G7_TIE_TOL) {
      ++w_wor;
    } else {
      ++w_eq;
    }
    if (fabs(gT8Yt[i]) < G7_REL_FLOOR) {
      ++near_zero;
    } else {
      prel += ap / fabs(gT8Yt[i]);
      brel += ab / fabs(gT8Yt[i]);
      ++prel_n;
      ++brel_n;
    }
  }
  /* E-3C0 condition (a): b3 full-tile stored mean-abs < plain
   * (fp32carried regime — this whole tile is fp32carried: both
   * operands split from the wider-precision truth). */
  gT8_cond_a = (bm < pm);

  printf("e3c0_tile cells=%ld "
         "plain_stored_mean_abs=%.6e plain_stored_median_abs=%.6e "
         "plain_stored_p95_abs=%.6e plain_stored_max_abs=%.6e "
         "plain_stored_rmse=%.6e b3_stored_mean_abs=%.6e "
         "b3_stored_median_abs=%.6e b3_stored_p95_abs=%.6e "
         "b3_stored_max_abs=%.6e b3_stored_rmse=%.6e "
         "b3_dblbr_stored_mean_abs=%.6e wonly_stored_mean_abs=%.6e "
         "plain_optimal_bits_frac=%.6e b3_optimal_bits_frac=%.6e "
         "wonly_optimal_bits_frac=%.6e improved=%ld equal=%ld "
         "worsened=%ld tie_tol=%.6e bit_diff_n=%ld bit_diff_frac=%.6e "
         "rel_floor=%.6e plain_stored_mean_rel_ge_floor=%.6e "
         "b3_stored_mean_rel_ge_floor=%.6e near_zero_count=%ld\n",
         nn, pm, pmed, pp95, pmax, prmse, bm, bmed, bp95, bmax, brmse,
         dm, wm, (double)popt / (double)nn, (double)bopt / (double)nn,
         (double)nopt / (double)nn, improved, equal, worsened,
         G7_TIE_TOL, bitdiff, (double)bitdiff / (double)nn,
         G7_REL_FLOOR, prel_n ? prel / (double)prel_n : 0.0,
         brel_n ? brel / (double)brel_n : 0.0, near_zero);

  printf("e3c0_wonly stored_mean_abs=%.6e optimal_bits_frac=%.6e "
         "improved_vs_plain=%ld equal_vs_plain=%ld worsened_vs_plain=%ld "
         "bit_diff_vs_plain_n=%ld\n",
         wm, (double)nopt / (double)nn, w_imp, w_eq, w_wor, w_bitdiff);

  printf("e3c0_dblbr b3_float_stored_mean_abs=%.6e "
         "b3_dblbr_stored_mean_abs=%.6e delta_dblbr_minus_float=%.6e "
         "dblbr_optimal_bits_frac=%.6e\n",
         bm, dm, dm - bm, (double)dopt / (double)nn);

  /* Per case-class: plain vs b3 stored mean-abs, improved/equal/
   * worsened, oracle-optimal bit rates, raw-bit disagreement. */
  for (c = 0; c < 5; ++c) {
    int s, o;
    long cnt = 0, imp = 0, eq = 0, wor = 0, po = 0, bo = 0, bd = 0;
    double cmed, cp95, cmx, crmse;
    double cpm, cbm;

    for (s = 0; s < T8; ++s) {
      if (t8_class_of(s) != c) continue;
      for (o = 0; o < T8; ++o) {
        const long i = (long)s * T8 + o;
        const double ap = fabs(t8_descale(gT8Ps[i]) - gT8Yt[i]);
        const double ab = fabs(t8_descale(gT8Bs[i]) - gT8Yt[i]);
        if (ap - ab > G7_TIE_TOL) {
          ++imp;
        } else if (ab - ap > G7_TIE_TOL) {
          ++wor;
        } else {
          ++eq;
        }
        if (gT8Ps[i] == gT8Os[i]) ++po;
        if (gT8Bs[i] == gT8Os[i]) ++bo;
        if (gT8Ps[i] != gT8Bs[i]) ++bd;
        ++cnt;
      }
    }
    cnt = 0;
    for (s = 0; s < T8; ++s) {
      if (t8_class_of(s) != c) continue;
      for (o = 0; o < T8; ++o) {
        const long i = (long)s * T8 + o;
        gT8E[cnt++] = fabs(t8_descale(gT8Ps[i]) - gT8Yt[i]);
      }
    }
    abs_error_stats(gT8E, cnt, &cpm, &cmed, &cp95, &cmx, &crmse);
    cnt = 0;
    for (s = 0; s < T8; ++s) {
      if (t8_class_of(s) != c) continue;
      for (o = 0; o < T8; ++o) {
        const long i = (long)s * T8 + o;
        gT8E[cnt++] = fabs(t8_descale(gT8Bs[i]) - gT8Yt[i]);
      }
    }
    abs_error_stats(gT8E, cnt, &cbm, &cmed, &cp95, &cmx, &crmse);
    printf("e3c0_class name=%s cells=%ld plain_stored_mean_abs=%.6e "
           "b3_stored_mean_abs=%.6e improved=%ld equal=%ld worsened=%ld "
           "plain_optimal_bits_frac=%.6e b3_optimal_bits_frac=%.6e "
           "bit_diff_n=%ld\n",
           cname[c], cnt, cpm, cbm, imp, eq, wor,
           (double)po / (double)cnt, (double)bo / (double)cnt, bd);
  }
}

/* Grid-preservation invariant: does RN16(64*x)/64 equal RN16(x)?
 * 64 = 2^6 so scaling commutes with RN-rounding EXACTLY while both
 * ends stay normal and no overflow occurs; it can only break in the
 * subnormal store band (|64x| < 2^-14, i.e. |x| < 2^-20 — and more
 * broadly |x| < 2^-14 where the FP16 input itself is subnormal) and
 * above the overflow threshold (|64x| > 65504). Tested on a directed
 * magnitude sweep (both signs, several fractions per binade) and on
 * every tile cell (plain + b3 accs); the breaking bands are printed
 * with data. Sets cond_d. */
static void t8_grid_preserve(void) {
  static const double fr[7] = {1.0, 1.5, 1.9999999, 1.0000001,
                               0.9999999, 0.4, 1.4};
  const double hold_lo = 0x1p-14, hold_hi = 65504.0 / 64.0; /* 1023.5 */
  long n = 0, hold_b = 0, over_n = 0, over_b = 0;
  long sub_br_lo = 0, sub_br_hi = 0, sub_ok_lo = 0, sub_ok_hi = 0;
  double sub_br_min = 0.0, sub_br_max = 0.0;
  int have_br = 0;
  long tile_p_break = 0, tile_b_break = 0, tile_hold_break = 0;
  int e, f, sign, i;

  for (e = -30; e <= 17; ++e) {
    for (f = 0; f < (int)(sizeof fr / sizeof fr[0]); ++f) {
      const double x = ldexp(fr[f], e);
      for (sign = 0; sign < 2; ++sign) {
        const double v = sign ? -x : x;
        const double a = fabs(v);
        const double des = t8_descale(double_to_fp16(64.0 * v));
        const double rn = fp16_to_double(double_to_fp16(v));
        ++n;
        if (des != rn) {
          if (a >= hold_lo && a <= hold_hi) {
            ++hold_b; /* must be 0: proven scale-invariance band */
          } else if (a > hold_hi) {
            ++over_n;
            ++over_b;
          } else { /* |v| < 2^-14: subnormal store/input band */
            if (a < 0x1p-20) {
              ++sub_br_lo;
            } else {
              ++sub_br_hi;
            }
            if (!have_br) {
              sub_br_min = a;
              sub_br_max = a;
              have_br = 1;
            } else {
              if (a < sub_br_min) sub_br_min = a;
              if (a > sub_br_max) sub_br_max = a;
            }
          }
        } else if (a > hold_hi) {
          ++over_n; /* both inf (x > 65504): agreement, not success */
        } else if (a < hold_lo) {
          if (a < 0x1p-20) {
            ++sub_ok_lo;
          } else {
            ++sub_ok_hi;
          }
        }
      }
    }
  }

  for (i = 0; i < T8_CELLS; ++i) {
    const double ap = fabs(gT8P[i]);
    const double dp = t8_descale(double_to_fp16(64.0 * gT8P[i]));
    const double rp = fp16_to_double(double_to_fp16(gT8P[i]));
    const double ab = fabs(gT8B[i]);
    const double db = t8_descale(double_to_fp16(64.0 * gT8B[i]));
    const double rb = fp16_to_double(double_to_fp16(gT8B[i]));
    if (dp != rp) {
      ++tile_p_break;
      if (ap >= hold_lo && ap <= hold_hi) ++tile_hold_break;
    }
    if (db != rb) {
      ++tile_b_break;
      if (ab >= hold_lo && ab <= hold_hi) ++tile_hold_break;
    }
  }

  /* cond(d): grid-preservation holds everywhere except the documented
   * subnormal and overflow bands (zero breaks inside the proven
   * hold band, in the sweep AND on the tile). */
  gT8_cond_d = (hold_b == 0 && tile_hold_break == 0);

  printf("e3c0_grid sweep_n=%ld hold_band=[2^-14,1023.5] hold_breaks=%ld "
         "sub_breaks=%ld sub_br_lt_2^-20=%ld sub_br_ge_2^-20=%ld "
         "sub_ok_lt_2^-20=%ld sub_ok_ge_2^-20=%ld over_n=%ld "
         "over_breaks=%ld sub_br_min=%.6e sub_br_max=%.6e "
         "tile_plain_breaks=%ld tile_b3_breaks=%ld tile_hold_breaks=%ld "
         "band_sub=|acc|<2^-14(64*acc_subnormal_or_x_subnormal) "
         "band_over=|acc|>1023.5(64*acc_gt_65504)\n",
         n, hold_b, sub_br_lo + sub_br_hi, sub_br_lo, sub_br_hi,
         sub_ok_lo, sub_ok_hi, over_n, over_b,
         have_br ? sub_br_min : 0.0, have_br ? sub_br_max : 0.0,
         tile_p_break, tile_b_break, tile_hold_break);
}

/* G9: E-3D sign-separated store model on the G8 tile.
 *
 * Method: duplicate every logical output channel into two physical channels,
 * convert the same ACC with shape 0 and shape 4, and rebuild
 * y = s0 - s4 on the HVX side.
 *
 * Model preconditions: bo = 0 and one shared positive scale (64). Under the
 * measured clamp law exactly one half is non-zero, so only one RN16 rounding
 * occurs and the FP16 subtraction is exact. The split therefore cannot change
 * the stored value: this check asserts that the published plain and b3 stored
 * words are reproduced BIT-FOR-BIT, and that the stored-domain improvement is
 * carried over unchanged. The tile values are CPU models; the only device
 * inputs are the E-3D four slots (identity/negation shapes + negative clamp).
 * NaN/Inf payloads are out of scope and do not occur on this tile. */
static uint16_t t8_split_store(double acc, uint16_t *s0, uint16_t *s4) {
  *s0 = double_to_fp16(64.0 * acc);
  *s4 = double_to_fp16(-64.0 * acc);
  *s0 = (*s0 & 0x8000u) ? 0x0000u : *s0; /* measured negative clamp */
  *s4 = (*s4 & 0x8000u) ? 0x0000u : *s4;
  return double_to_fp16(fp16_to_double(*s0) - fp16_to_double(*s4));
}

static void t8_sign_split(void) {
  const long fail_before = g_fail;
  long i;
  long p_bits_diff = 0, b_bits_diff = 0;
  long p_both_nonzero = 0, b_both_nonzero = 0, one_sided = 0, both_zero = 0;
  long improved = 0, equal = 0, worsened = 0, popt = 0, bopt = 0;
  double pm, pmed, pp95, pmax, prmse;
  double bm, bmed, bp95, bmax, brmse;
  double ratio;

  for (i = 0; i < T8_CELLS; ++i) {
    uint16_t p0, p4, b0, b4;
    gT8Sp[i] = t8_split_store(gT8P[i], &p0, &p4);
    gT8Sb[i] = t8_split_store(gT8B[i], &b0, &b4);
    if (gT8Sp[i] != gT8Ps[i]) ++p_bits_diff;
    if (gT8Sb[i] != gT8Bs[i]) ++b_bits_diff;
    if (b0 != 0u && b4 != 0u) ++b_both_nonzero;
    if (p0 != 0u && p4 != 0u) {
      ++p_both_nonzero;
    } else if (p0 != 0u || p4 != 0u) {
      ++one_sided;
    } else {
      ++both_zero;
    }
  }

  for (i = 0; i < T8_CELLS; ++i)
    gT8E[i] = fabs(t8_descale(gT8Sp[i]) - gT8Yt[i]);
  abs_error_stats(gT8E, T8_CELLS, &pm, &pmed, &pp95, &pmax, &prmse);
  for (i = 0; i < T8_CELLS; ++i)
    gT8E[i] = fabs(t8_descale(gT8Sb[i]) - gT8Yt[i]);
  abs_error_stats(gT8E, T8_CELLS, &bm, &bmed, &bp95, &bmax, &brmse);

  for (i = 0; i < T8_CELLS; ++i) {
    const double ap = fabs(t8_descale(gT8Sp[i]) - gT8Yt[i]);
    const double ab = fabs(t8_descale(gT8Sb[i]) - gT8Yt[i]);
    if (ap - ab > G7_TIE_TOL) {
      ++improved;
    } else if (ab - ap > G7_TIE_TOL) {
      ++worsened;
    } else {
      ++equal;
    }
    if (gT8Sp[i] == gT8Os[i]) ++popt;
    if (gT8Sb[i] == gT8Os[i]) ++bopt;
  }
  ratio = (bm > 0.0) ? pm / bm : 0.0;

  printf("sign_split_g9 cells=%ld plain_recon_mean_abs=%.6e "
         "b3_recon_mean_abs=%.6e plain_over_b3=%.6f "
         "plain_bits_diff=%ld b3_bits_diff=%ld plain_both_nonzero=%ld "
         "b3_both_nonzero=%ld plain_one_sided=%ld plain_both_zero=%ld "
         "improved=%ld equal=%ld worsened=%ld "
         "plain_optimal_bits_frac=%.6e b3_optimal_bits_frac=%.6e\n",
         (long)T8_CELLS, pm, bm, ratio, p_bits_diff, b_bits_diff,
         p_both_nonzero, b_both_nonzero, one_sided, both_zero, improved,
         equal, worsened, (double)popt / (double)T8_CELLS,
         (double)bopt / (double)T8_CELLS);

  ++g_checks;
  if (p_bits_diff != 0 || b_bits_diff != 0) {
    ++g_fail;
    printf("FAIL sign_split_g9 bits_diff plain=%ld b3=%ld\n", p_bits_diff,
           b_bits_diff);
  }
  ++g_checks;
  if (p_both_nonzero != 0 || b_both_nonzero != 0) {
    ++g_fail;
    printf("FAIL sign_split_g9 both_nonzero plain=%ld b3=%ld\n",
           p_both_nonzero, b_both_nonzero);
  }
  /* The 1.85x figure belongs to b3 vs plain on this tile; the split must
   * carry it over, not improve or degrade it. */
  ++g_checks;
  if (!(ratio > 1.80 && ratio < 1.90)) {
    ++g_fail;
    printf("FAIL sign_split_g9 ratio=%.6f\n", ratio);
  }
  printf("sign_split_g9_verdict=%s accuracy_change=%s\n",
         g_fail == fail_before ? "ENCODING_PRESERVED" : "FAIL",
         g_fail == fail_before ? "none" : "unexpected");
}

/* E-3C0 decision line. PASS requires ALL of:
 *   (a) b3 full-tile stored mean-abs < plain (fp32carried),
 *   (b) every pinned cell assert + layout-distinctness assert passes,
 *   (c) the B-2 anchor (s2,o1) reproduces 0x5400/0x53ff,
 *   (d) grid-preservation holds outside the documented bands.
 * A partial result is NEVER called PASS. */
static void t8_gate(void) {
  const int a = gT8_cond_a;
  const int b = (gT8_pin_mismatch == 0);
  const int c = gT8_anchor_ok;
  const int d = gT8_cond_d;

  printf("e3c0_cond_a=%d\n", a);
  printf("e3c0_cond_b=%d\n", b);
  printf("e3c0_cond_c=%d\n", c);
  printf("e3c0_cond_d=%d\n", d);
  if (a && b && c && d) {
    printf("e3c0_cpu_gate=PASS basis=e3c0_cond_a_and_b_and_c_and_d "
           "threshold=all_4_conds\n");
  } else if (b && c) {
    printf("e3c0_cpu_gate=PARTIAL basis=e3c0_cond_a_and_b_and_c_and_d "
           "threshold=all_4_conds\n");
  } else {
    printf("e3c0_cpu_gate=FAIL basis=e3c0_cond_a_and_b_and_c_and_d "
           "threshold=all_4_conds\n");
  }
}

static void t8_run(void) {
  t8_build();
  t8_compute();
  t8_pins();
  t8_map();
  t8_metrics();
  t8_grid_preserve();
  t8_sign_split();
  t8_gate();
}

int main(void) {
  g1_g3_recovery();
  g2_table();
  g4_subnormal_sweep();
  g5_g6_synthesis();
  g7_stored_domain();
  t8_run();
  if (g_fail == 0) {
    printf("fp16x2_split_sim=PASS cases=%ld\n", g_checks);
    return 0;
  }
  printf("fp16x2_split_sim=FAIL failures=%ld cases=%ld\n", g_fail, g_checks);
  return 1;
}
