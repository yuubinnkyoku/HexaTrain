// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// V81 HMX convert-feedback EXPERIMENT, not an FP32 GEMM implementation.
// Derived from the public V81 HMX instruction specification (80-N2040-62 AA),
// not copied from any third-party kernel.
//
// Test x = (1 + 2^-10)^2 = 1 + 2^-9 + 2^-20.
// FP16 x FP16 -> HMX ACC; convert with scale 64:
//   hi = round_fp16(64*x)
//   residual = round_fp16(-64*x + min(+65504, previous_convert))
// Then compare whether (hi - residual)/64 recovers x.
//
// Two trials: first-convert Rs[8]=0 and Rs[8]=1 (extra feedback bits).
// No assertion that the undocumented intermediate rounding preserves residual.
#include "hexatrain_hmx_acc_residual.h"

#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hmx_identity.h"
#include "hmx_mode43.h"
#include "bias20.h"
#include "fp16_util.h" /* E-3C1: shared RN16 / fp16<->double helpers */

#if !defined(__HEXAGON_ARCH__) || __HEXAGON_ARCH__ != 81
#error This probe must be built for the V81 DSP target.
#endif

enum {
  TILE_BYTES = 2048,
  VTCM_BYTES = 12288,
  BIAS_BYTES = 256,
  HMX_MAGIC = HMX_IDENTITY_MAGIC,
  HMX_ERR_ARG = 14001,
  HMX_ERR_ALLOC = 14002,
  HMX_ERR_VTCM = 14003,
  HMX_ERR_LOCK = 14004,
  HMX_ERR_UNLOCK = 14005,   /* legacy hmx_unlock (v1), unused by this probe */
  HMX_ERR_POWER = 14006,    /* HAP_power_set power-up request failed */
  HMX_ERR_UNLOCK2 = 14007,  /* cleanup stage (a): hmx_unlock2 failed */
  HMX_ERR_RELEASE = 14008,  /* cleanup stage (b): compute_res_release failed */
  HMX_ERR_POWER_DOWN = 14009, /* cleanup stage (c): power-down failed */
  PROBE_REVISION = HMX_IDENTITY_EXPECTED_REVISION,
  MODE_IDENTITY = HMX_IDENTITY_MODE,
  MODE_MAX = 43, /* E-3C1R highest mode; must mirror probe_host.c HMX_PROBE_MODE_MAX */
};

typedef struct probe_context {
  uint32_t tag;
} probe_context_t;

static void init_bias20(uint32_t *dst, uint16_t scale_bits,
                        uint16_t output_bias_bits, unsigned extra4,
                        unsigned shape) {
  // V81 Table 4: 32 low words then 32 high words, not interleaved uint64.
  // Low[15:0] scale (FP16), low[31:16] output bias (FP16);
  // high[10:8] shape, high[7:4] output-bias extra4 nibble,
  // high[3:0] scale extra4 (left 0). The cell packing lives in bias20.h
  // so the host oracle and the CPU spec test use the same encoding.
  memset(dst, 0, BIAS_BYTES);
  for (int ch = 0; ch < 32; ++ch) {
    dst[ch] = bias20_lo_cell(scale_bits, output_bias_bits);
    dst[32 + ch] = bias20_hi_cell(shape, extra4, 0);
  }
}

static void init_bias(uint32_t *dst, uint16_t scale_bits,
                      uint16_t output_bias_bits, unsigned shape) {
  init_bias20(dst, scale_bits, output_bias_bits, 0, shape);
}

/* ---- E-3C1 (mode 42) tile-build helpers ----
 * Deterministic 32x32 tile ported from the CPU oracle G8 in
 * hmx_fp16x2_split_sim.c (rng64/rng_unit verbatim; integer xorshift is
 * bit-exact on every target). The double truth tiles live in file scope
 * so the DSP stack stays small. */
#define E3C1_T8 32
#define E3C1_T8_CELLS (E3C1_T8 * E3C1_T8)

static uint64_t e3c1_rng;
static double e3c1_x[E3C1_T8_CELLS]; /* activation truth (wider precision) */
static double e3c1_w[E3C1_T8_CELLS]; /* weight truth */

static uint64_t e3c1_rng64(void) {
  uint64_t x = e3c1_rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  e3c1_rng = x;
  return x;
}

/* Uniform in [0,1) with 53 random bits. */
static double e3c1_rng_unit(void) {
  return (double)(e3c1_rng64() >> 11) * (1.0 / 9007199254740992.0);
}

/* Float-arithmetic split residual: statement order mirrors the sim's
 * split_float_parts — hi = RN16(x) is computed by the caller FIRST,
 * then (float)x - (float)hi in FLOAT arithmetic as separate statements
 * (no FMA can contract), then lo = RN16(sub). The plain operand is
 * exactly hi = RN16(x), so the plain variant reuses the hi tiles. */
static uint16_t e3c1_split_lo(double x, uint16_t hi_bits) {
  float xf = (float)x;
  float hf = (float)fp16_to_double(hi_bits);
  float sub = xf - hf;
  return double_to_fp16((double)sub);
}

static void run_one(uint8_t *base, unsigned extra_four_bits, int q, int mode,
                    int *hi_bits, int *residual_bits) {
  uint16_t *act = (uint16_t *)(base + 0 * TILE_BYTES);
  uint16_t *wgt = (uint16_t *)(base + 1 * TILE_BYTES);
  uint16_t *hi = (uint16_t *)(base + 2 * TILE_BYTES);
  uint16_t *residual = (uint16_t *)(base + 3 * TILE_BYTES);
  uint32_t *bias0 = (uint32_t *)(base + 4 * TILE_BYTES);
  uint32_t *bias1 = (uint32_t *)((uint8_t *)bias0 + BIAS_BYTES);

  if (mode == 40) {
    /* B-1 experiment (PR #14, user-approved): TWO MMA packets accumulate
     * into the SAME ACC with NO convert and NO FP16 store between them;
     * a single bias-0 convert (plain RN-FP16, scale 64) reads the ACC at
     * the very end. mxclracc runs exactly once per trial, at the start.
     * The weight row is exact 1.0 (0x3c00), so each MMA contributes just
     * its activation scalar (product act[0][0]*wgt[0][0], rest zero).
     *   trial 0 (extra_four_bits==0): (+1)+(+2) accumulate 3*64=192
     *     =0x5a00; overwrite 2*64=128=0x5800.
     *   trial 1 (extra_four_bits==1): (+2)+(-1) accumulate 1*64=64
     *     =0x5400; overwrite -64 -> 0xd400 or a clamp to 0x0000
     *     (negative bias-0 convert is unmeasured; both overwrite-
     *     consistent, contrast only, per the user's B-1 table). */
    uint16_t s1;
    uint16_t s2;
    uint16_t *act2 = (uint16_t *)(base + 5 * TILE_BYTES);
    if (extra_four_bits) {
      s1 = 0x4000u;  /* +2.0 */
      s2 = 0xbc00u;  /* -1.0 */
    } else {
      s1 = 0x3c00u;  /* +1.0 */
      s2 = 0x4000u;  /* +2.0 */
    }
    memset(act, 0, TILE_BYTES);
    memset(wgt, 0, TILE_BYTES);
    memset(act2, 0, TILE_BYTES);
    memset(hi, 0xa5, TILE_BYTES);
    memset(residual, 0xa5, TILE_BYTES);
    act[0] = s1;
    wgt[0] = 0x3c00u;  /* multiplier exactly 1.0 */
    act2[0] = s2;
    init_bias(bias0, 0x5400, 0, 0);
    init_bias(bias1, 0, 0, 0);
    {
      const uintptr_t bias0_reg = (uintptr_t)bias0;
      const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
      const uintptr_t act1_reg = (uintptr_t)act;
      const uintptr_t act2_reg = (uintptr_t)act2;
      const uintptr_t wgt_reg = (uintptr_t)wgt;
      const uint32_t spatial_mask = (14u << 7);
      const uint32_t act_rt = spatial_mask | (31u << 2);
      const uint32_t wgt_rt = (uint32_t)(TILE_BYTES - 1);
      const uint32_t wr_rt = spatial_mask;

      asm volatile("mxclracc.hf\n" ::: "memory");
      asm volatile(
          "{ activation.hf = mxmem(%0, %1)\n"
          "  weight.hf = mxmem(%2, %3) }\n"
          : : "r"(act1_reg), "r"(act_rt), "r"(wgt_reg), "r"(wgt_rt)
          : "memory");
      /* Second MMA packet issued immediately: between the two packets
       * there is NO convert, NO FP16 store and NO clear — the packet
       * pair above is the only thing scheduled between the MMAs. */
      asm volatile(
          "{ activation.hf = mxmem(%0, %1)\n"
          "  weight.hf = mxmem(%2, %3) }\n"
          : : "r"(act2_reg), "r"(act_rt), "r"(wgt_reg), "r"(wgt_rt)
          : "memory");

      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");

      /* Single final convert: bias set 0, retain ACC, no feedback
       * rounding (Phase-D style Rs = 1: Rs[0]=1 only). */
      asm volatile("cvt.hf = acc(%0)\n" : : "r"(1u) : "memory");
      asm volatile("mxmem(%0, %1) = cvt\n"
                   : : "r"((uintptr_t)hi), "r"(wr_rt) : "memory");
    }
    *hi_bits = (int)hi[0];
    *residual_bits = 0;
    return;
  }

  if (mode == 41) {
    /* B-2 experiment (PR #14 comment 6293114616, stage E-3B, user-ap-
     * proved: one build, one device run): three MMA packets accumulate
     * into the SAME ACC — Ahi*Whi, Ahi*Wlo, Alo*Whi (the three-product
     * hs lo synthesis; lo*lo is deliberately omitted) — with NO
     * convert, NO FP16 store and NO clear between packets, then ONE
     * bias-0 convert (plain RN-FP16, scale 64) and ONE store.
     * mxclracc runs exactly once per subcase, at the start.
     * Pre-registered fixtures (expected bits fixed; do not retune):
     *   trial 0: Ahi=Whi=0x3c00 (1.0), Alo=Wlo=0x1000 (+2^-11)
     *     -> ACC = 1+2^-10, scaled = 64.0625 -> expect 0x5401
     *     (plain 1-product context: 0x5400)
     *   trial 1: Ahi=Whi=0x3c00, Alo=Wlo=0x8c00 (-2^-12)
     *     -> ACC = 1-2^-11, scaled = 63.96875 -> expect 0x53ff
     *     (plain context: 0x5400) */
    uint16_t alo_bits;
    uint16_t *act_hi = (uint16_t *)(base + 0 * TILE_BYTES);
    uint16_t *act_lo = (uint16_t *)(base + 1 * TILE_BYTES);
    uint16_t *wgt_hi = (uint16_t *)(base + 2 * TILE_BYTES);
    uint16_t *wgt_lo = (uint16_t *)(base + 3 * TILE_BYTES);
    uint16_t *out = (uint16_t *)(base + 5 * TILE_BYTES); /* store dst */
    if (extra_four_bits) {
      alo_bits = 0x8c00u;  /* -2^-12 */
    } else {
      alo_bits = 0x1000u;  /* +2^-11 */
    }
    memset(act_hi, 0, TILE_BYTES);
    memset(act_lo, 0, TILE_BYTES);
    memset(wgt_hi, 0, TILE_BYTES);
    memset(wgt_lo, 0, TILE_BYTES);
    memset(out, 0xa5, TILE_BYTES);
    act_hi[0] = 0x3c00u;   /* Ahi = Whi = exactly 1.0 */
    wgt_hi[0] = 0x3c00u;
    act_lo[0] = alo_bits;  /* Alo = Wlo */
    wgt_lo[0] = alo_bits;
    init_bias(bias0, 0x5400, 0, 0);
    init_bias(bias1, 0, 0, 0);
    {
      const uintptr_t bias0_reg = (uintptr_t)bias0;
      const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
      const uintptr_t act_hi_reg = (uintptr_t)act_hi;
      const uintptr_t act_lo_reg = (uintptr_t)act_lo;
      const uintptr_t wgt_hi_reg = (uintptr_t)wgt_hi;
      const uintptr_t wgt_lo_reg = (uintptr_t)wgt_lo;
      const uintptr_t out_reg = (uintptr_t)out;
      const uint32_t spatial_mask = (14u << 7);
      const uint32_t act_rt = spatial_mask | (31u << 2);
      const uint32_t wgt_rt = (uint32_t)(TILE_BYTES - 1);
      const uint32_t wr_rt = spatial_mask;

      asm volatile("mxclracc.hf\n" ::: "memory");
      /* packet 1: Ahi*Whi */
      asm volatile(
          "{ activation.hf = mxmem(%0, %1)\n"
          "  weight.hf = mxmem(%2, %3) }\n"
          : : "r"(act_hi_reg), "r"(act_rt), "r"(wgt_hi_reg), "r"(wgt_rt)
          : "memory");
      /* packet 2: Ahi*Wlo — same act tile (intentional), new weight */
      asm volatile(
          "{ activation.hf = mxmem(%0, %1)\n"
          "  weight.hf = mxmem(%2, %3) }\n"
          : : "r"(act_hi_reg), "r"(act_rt), "r"(wgt_lo_reg), "r"(wgt_rt)
          : "memory");
      /* packet 3: Alo*Whi — new activation, Whi tile back (intentional) */
      asm volatile(
          "{ activation.hf = mxmem(%0, %1)\n"
          "  weight.hf = mxmem(%2, %3) }\n"
          : : "r"(act_lo_reg), "r"(act_rt), "r"(wgt_hi_reg), "r"(wgt_rt)
          : "memory");

      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");

      /* One final convert (bias set 0, retain, no feedback rounding,
       * Rs = 1: Rs[0] only) then the single FP16 store. */
      asm volatile("cvt.hf = acc(%0)\n" : : "r"(1u) : "memory");
      asm volatile("mxmem(%0, %1) = cvt\n"
                   : : "r"(out_reg), "r"(wr_rt) : "memory");
    }
    *hi_bits = (int)out[0];
    *residual_bits = 0;
    return;
  }

  if (mode == 42) {
    /* E-3C1 (PR #14 comment 6294110662, user-approved: one build, one
     * device run): ONE fixed realistic 32x32x32 multi-channel tile
     * (bit-match of the CPU oracle G8 t8_build — the statement order
     * below mirrors it verbatim) run through TWO variants, returning 4
     * stored cells through the existing results[4] ABI (metadata[9]
     * untouched, kernel_us still measured in metadata[3]):
     *   trial 0 (extra_four_bits==0): PLAIN — mxclracc, then ONE MMA
     *     packet {act_hi, wgt_hi}; plain operands are RN16(X)/RN16(W)
     *     and act_hi/wgt_hi ARE RN16(X)/RN16(W), so the plain variant
     *     reuses those two tiles (no extra tiles), then bias0 load,
     *     bias1 load, convert, store.
     *   trial 1 (extra_four_bits==1): B3 — mxclracc, then THREE
     *     full-tile MMA packets {act_hi,wgt_hi},{act_hi,wgt_lo},
     *     {act_lo,wgt_hi} accumulating into the SAME ACC with NO
     *     convert, NO store and NO clear between packets (B-1/B-2
     *     confirmed law), then ONE bias-0 convert (plain RN-FP16,
     *     scale 64) and ONE store.
     * Both trials read back out[0] (cell (s0,o0) under BOTH layout
     * candidates) and out[32].
     * Pre-registered expectations (FIXED before measurement; do not
     * retune):
     *   out[0]:  plain 0x5400, B3 0x5401
     *   out[32]: the negative final store is UNMEASURED — signed RN16
     *     (0xd400 plain / 0xd401 B3) vs a clamp to 0x0000 are both
     *     candidates and the device discriminates (E-3C1 negative-store
     *     discriminator).
     *   Layout: out[32] = cell (s1,o0), value = -oracle (L1 spatial-
     *     block layout) or 0x5400 = cell (s0,o1) positive (L2
     *     transposed layout); the device discriminates L1 vs L2.
     * VTCM layout: act_hi +0, act_lo +1, wgt_hi +2, wgt_lo +3 (all
     * TILE_BYTES), bias0 +4*TILE, bias1 +4*TILE+256, out +5*TILE
     * (fills VTCM_BYTES exactly). */
    uint16_t *act_hi = (uint16_t *)(base + 0 * TILE_BYTES);
    uint16_t *act_lo = (uint16_t *)(base + 1 * TILE_BYTES);
    uint16_t *wgt_hi = (uint16_t *)(base + 2 * TILE_BYTES);
    uint16_t *wgt_lo = (uint16_t *)(base + 3 * TILE_BYTES);
    uint16_t *out = (uint16_t *)(base + 5 * TILE_BYTES); /* store dst */
    int idx;
    int s, o, i;

    /* ---- t8_build port: statement order verbatim ---- */
    e3c1_rng = 0xE3C0117E5EEDF00Dull;
    for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
      double u = e3c1_rng_unit();
      double v = u * 4.0;
      double val = v - 2.0;
      e3c1_w[idx] = val;
      u = e3c1_rng_unit();
      v = u * 4.0;
      val = v - 2.0;
      e3c1_x[idx] = val;
    }

    /* Special W rows: out-ch 0..3 case-class anchors, 28..31 s3
     * subnormal-edge targets; out-ch 4..27 stay seeded realistic. */
    for (o = 0; o < E3C1_T8; ++o) {
      if (o <= 3 || o >= 28) {
        for (i = 0; i < E3C1_T8; ++i) e3c1_w[(long)o * E3C1_T8 + i] = 0.0;
      }
    }
    e3c1_w[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11; /* positive split-sensitive row */
    e3c1_w[0 * E3C1_T8 + 1] = 0x1p-11;
    e3c1_w[0 * E3C1_T8 + 2] = 0x1p-13;
    e3c1_w[1 * E3C1_T8 + 0] = 1.0 - 0x1p-12;  /* B-2 case-1 anchor row */
    e3c1_w[2 * E3C1_T8 + 0] = 0x1p-9;         /* s3 boundary row (64*acc=2^-14) */
    e3c1_w[3 * E3C1_T8 + 0] = 1.0;            /* near-zero cancellation row */
    e3c1_w[3 * E3C1_T8 + 1] = 1.0 + 0x1p-12;
    e3c1_w[28 * E3C1_T8 + 0] = 0x1p-10;
    e3c1_w[29 * E3C1_T8 + 0] = 0x1p-10 - 0x1p-21;
    e3c1_w[30 * E3C1_T8 + 0] = 3.0 * 0x1p-12;
    e3c1_w[31 * E3C1_T8 + 0] = 0x1p-13;

    /* Special A lanes 0..3; lanes 4..31 stay seeded realistic. */
    for (s = 0; s <= 3; ++s)
      for (i = 0; i < E3C1_T8; ++i) e3c1_x[(long)s * E3C1_T8 + i] = 0.0;
    e3c1_x[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11; /* s0: positive multi-channel sum */
    e3c1_x[0 * E3C1_T8 + 1] = 0x1p-11;
    e3c1_x[0 * E3C1_T8 + 2] = 0x1p-13;
    for (i = 0; i < E3C1_T8; ++i)          /* s1 = negation of s0 */
      e3c1_x[1 * E3C1_T8 + i] = -e3c1_x[i];
    e3c1_x[2 * E3C1_T8 + 0] = 1.0 - 0x1p-12; /* s2: B-2 case-1 + negated twin */
    e3c1_x[2 * E3C1_T8 + 1] = -(1.0 - 0x1p-12);
    e3c1_x[3 * E3C1_T8 + 0] = 0x1p-10;       /* s3: subnormal-edge lane */

    /* Derive the four fp16 operand tiles per cell: hi = RN16(value) is
     * the plain operand (plain reuses act_hi/wgt_hi), lo = the sim's
     * float-split residual. Pure per-cell function of the FINAL double
     * value, so deriving after all overwrites is equivalent to the
     * sim's build-then-derive order. */
    for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
      uint16_t ah = double_to_fp16(e3c1_x[idx]);
      uint16_t wh = double_to_fp16(e3c1_w[idx]);
      act_hi[idx] = ah;
      act_lo[idx] = e3c1_split_lo(e3c1_x[idx], ah);
      wgt_hi[idx] = wh;
      wgt_lo[idx] = e3c1_split_lo(e3c1_w[idx], wh);
    }

    init_bias(bias0, 0x5400, 0, 0);
    init_bias(bias1, 0, 0, 0);
    memset(out, 0xa5, TILE_BYTES);
    {
      const uintptr_t bias0_reg = (uintptr_t)bias0;
      const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
      const uintptr_t act_hi_reg = (uintptr_t)act_hi;
      const uintptr_t act_lo_reg = (uintptr_t)act_lo;
      const uintptr_t wgt_hi_reg = (uintptr_t)wgt_hi;
      const uintptr_t wgt_lo_reg = (uintptr_t)wgt_lo;
      const uintptr_t out_reg = (uintptr_t)out;
      const uint32_t spatial_mask = (14u << 7);
      const uint32_t act_rt = spatial_mask | (31u << 2);
      const uint32_t wgt_rt = (uint32_t)(TILE_BYTES - 1);
      const uint32_t wr_rt = spatial_mask;

      asm volatile("mxclracc.hf\n" ::: "memory");
      if (extra_four_bits) {
        /* B3: three packets, SAME ACC, nothing between them. */
        asm volatile(
            "{ activation.hf = mxmem(%0, %1)\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            : : "r"(act_hi_reg), "r"(act_rt), "r"(wgt_hi_reg), "r"(wgt_rt)
            : "memory");
        asm volatile(
            "{ activation.hf = mxmem(%0, %1)\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            : : "r"(act_hi_reg), "r"(act_rt), "r"(wgt_lo_reg), "r"(wgt_rt)
            : "memory");
        asm volatile(
            "{ activation.hf = mxmem(%0, %1)\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            : : "r"(act_lo_reg), "r"(act_rt), "r"(wgt_hi_reg), "r"(wgt_rt)
            : "memory");
      } else {
        /* Plain: ONE packet; the plain tiles ARE act_hi/wgt_hi. */
        asm volatile(
            "{ activation.hf = mxmem(%0, %1)\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            : : "r"(act_hi_reg), "r"(act_rt), "r"(wgt_hi_reg), "r"(wgt_rt)
            : "memory");
      }

      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
      asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");

      /* One final convert (bias set 0, retain, no feedback rounding,
       * Rs = 1: Rs[0] only) then the single FP16 store. */
      asm volatile("cvt.hf = acc(%0)\n" : : "r"(1u) : "memory");
      asm volatile("mxmem(%0, %1) = cvt\n"
                   : : "r"(out_reg), "r"(wr_rt) : "memory");
    }
    *hi_bits = (int)out[0];
    *residual_bits = (int)out[32];
    return;
  }

  // a = 1 + 2^-q as exact FP16 (q in [6,10]).
  uint16_t a_bits;
  if (mode == 21 || mode == 22 || mode == 37) {
    a_bits = 0x3c00u;  // 1.0 for zero-residual / exact-zero controls
  } else if ((mode >= 27 && mode <= 29) ||
             (mode >= 31 && mode <= 36) || mode == 38) {
    a_bits = 0x3c04u;  // 1+2^-8 (Phase D: q bound to 8, forced explicitly)
  } else {
    a_bits = (uint16_t)(0x3c00u + (1u << (10 - q)));
  }
  // (spatial=0, channel=0) and (channel=0, out=0) both map to tile[0].
  memset(act, 0, TILE_BYTES);
  memset(wgt, 0, TILE_BYTES);
  memset(hi, 0xa5, TILE_BYTES);
  memset(residual, 0xa5, TILE_BYTES);
  act[0] = a_bits;
  wgt[0] = a_bits;

  // 64.0 FP16 = 0x5400, +65504 FP16 = 0x7bff.
  // Bias set 0: 64*x. Bias set 1: -64*x + min(65504, feedback).
  init_bias(bias0, 0x5400, 0, 0);           // shape(x) = x
  // bias1 configuration depends on mode.
  if (mode == 0) {
    init_bias(bias1, 0x5400, 0x7bff, 4);   // original feedback
  } else if (mode == 1) {
    init_bias(bias1, 0x5400, 0, 4);        // shape-only, no output bias
  } else if (mode == 2) {
    init_bias(bias1, 0, 0x4400, 0);        // output-bias-only: +4
  } else if (mode == 3) {
    init_bias(bias1, 0, 0x7bff, 0);        // feedback-only: 65504, shape 0
  } else if (mode == 4) {
    init_bias(bias1, 0x5400, 0x5420, 4);  // constant bias: output_bias=66
  } else if (mode == 5) {
    init_bias(bias0, 0, 0x4400, 0);        // bias set0 also +4 for first
    init_bias(bias1, 0, 0x4400, 0);        // bias set1 +4
  } else if (mode == 6) {
    init_bias(bias0, 0x5400, 0, 0);        // same set0 twice
    init_bias(bias1, 0x5400, 0, 0);
  } else if (mode == 7) {
    init_bias(bias0, 0, 0x4400, 0);        // set0 alone +4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 8) {
    init_bias(bias0, 0, 0, 0);
    init_bias(bias1, 0, 0x4400, 0);        // set1 alone +4
  } else if (mode == 9) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0xd400, 0, 0);        // set1 negative scale -64
  } else if (mode == 10) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0x5400, 0x7bff, 4);   // set1 feedback (spec order)
  } else if (mode == 11) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 +64
    init_bias(bias1, 0x5400, 0, 0);        // set1 +64 (positive switch)
  } else if (mode == 12) {
    init_bias(bias0, 0xd400, 0, 0);        // set0 -64
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 13) {
    init_bias(bias0, 0, 0xc400, 0);        // set0 output bias -4 (0xc400)
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 14) {
    init_bias(bias0, 0x5400, 0, 0);        // P: +64, bias 0
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 15) {
    init_bias(bias0, 0x5400, 0xc400, 0);   // NB: +64, bias -4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 16) {
    init_bias(bias0, 0xd400, 0x5800, 0);   // NS: -64, bias +128
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 17) {
    init_bias(bias0, 0x5400, 0x5800, 4);   // SH: +64, shape=4, bias +128
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 18) {
    init_bias(bias0, 0x5400, 0xd420, 0);   // R+: +64, bias -66
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 19) {
    init_bias(bias0, 0, 0xc400, 0);        // N: scale 0, bias -4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 20) {
    // R+ generalized: bias1 = -hi (negative hi) as output bias, scale +64.
    uint16_t hi_bits = 0x5402;
    if (q == 6) hi_bits = 0x5420;
    else if (q == 7) hi_bits = 0x5410;
    else if (q == 8) hi_bits = 0x5408;
    else if (q == 9) hi_bits = 0x5404;
    else if (q == 10) hi_bits = 0x5402;
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0x5400, hi_bits | 0x8000u, 0);  // set1 scale +64, bias -hi
  } else if (mode == 21) {
    init_bias(bias0, 0x5400, 0xd400, 0);   // zero-residual control: scale +64, bias -64
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 22) {
    // zero residual 2-CVT: a=1.0, bias0 +64, bias1 +64/-64
    init_bias(bias0, 0x5400, 0, 0);
    init_bias(bias1, 0x5400, 0xd400, 0);
  } else if (mode == 23) {
    // single convert with known -hi bias
    uint16_t hi_bits = 0x5402;
    if (q == 6) hi_bits = 0x5420;
    else if (q == 7) hi_bits = 0x5410;
    else if (q == 8) hi_bits = 0x5408;
    else if (q == 9) hi_bits = 0x5404;
    else if (q == 10) hi_bits = 0x5402;
    init_bias(bias0, 0x5400, hi_bits | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 24) {
    // scale sweep: scale=32 (0x5000), hi=33 (0x5020), bias=-33
    init_bias(bias0, 0x5000, 0x5020 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 25) {
    // scale sweep: scale=64 (0x5400), hi=66 (0x5420), bias=-66
    init_bias(bias0, 0x5400, 0x5420 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 26) {
    // scale sweep: scale=128 (0x5800), hi=132 (0x5820), bias=-132
    init_bias(bias0, 0x5800, 0x5820 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 27) {
    // q=8 scale=32: hi=32.25 (0x5008), bias=-32.25
    init_bias(bias0, 0x5000, 0x5008 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 28) {
    // q=8 scale=64: hi=64.5 (0x5408), bias=-64.5
    init_bias(bias0, 0x5400, 0x5408 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 29) {
    // q=8 scale=128: hi=129 (0x5808), bias=-129
    init_bias(bias0, 0x5800, 0x5808 | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else if (mode >= 31 && mode <= 39) {
    /* Phase D/E (extra4 nibble sweep, README "Phase D"):
     * feedback disabled, single-CVT (bias set 0 only), q bound to 8,
     * scale 64 (0x5400), a=b=0x3c04 except the mode-37 fallback.
     * The 20-bit output-bias value = -(64.5 + n*2^-8) with the FP16 half
     * 0xD408 and the extra4 nibble n = phase_d_extra4_nibble(mode);
     * mode 37 uses the exact-zero fallback encoding (-64.0, nibble 0,
     * a=b=1.0) because 64.5 + 2^-10 is off the 20-bit grid. Mode 39 is
     * the E-1 range discriminator: base 0xD404 = -64.25, nibble 0, so
     * the convert target is 64.25 grid units (README "Phase E-1"); the
     * FP16 half per mode is owned by bias20.h. */
    const int n = phase_d_extra4_nibble(mode);
    init_bias20(bias0, 0x5400, phase_d_out_bias_fp16(mode),
                (unsigned)n, 0);
    init_bias(bias1, 0, 0, 0);
  } else {
    init_bias(bias1, 0x5400, 0x7bff, 4);
  }

  // V81: bias load address must be 256B aligned; low two bits select set.
  const uintptr_t bias0_reg = (uintptr_t)bias0;
  const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
  const uintptr_t act_reg = (uintptr_t)act;
  const uintptr_t wgt_reg = (uintptr_t)wgt;
  // Spatial mask 0b11100 (YYYXX), all 32 input channels.
  const uint32_t spatial_mask = (14u << 7);
  const uint32_t act_rt = spatial_mask | (31u << 2);
  const uint32_t wr_rt = spatial_mask;

  // Intrinsics are not required: use the documented HMX assembly syntax.
  // activation and weight must appear in one instruction packet.
  asm volatile("mxclracc.hf\n" ::: "memory");
  asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3) }\n"
      : : "r"(act_reg), "r"(act_rt),
          "r"(wgt_reg), "r"((uint32_t)(TILE_BYTES - 1))
      : "memory");

  asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
  if (mode != 10) {
    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");
  }

  // First convert retains ACC; optionally keeps 4 extra feedback bits.
  uint32_t first_rs = 1u | ((extra_four_bits & 1u) << 8);
  if (mode == 5 || mode == 8) first_rs |= (1u << 12);  // first convert uses bias set1
  if (mode >= 31 && mode <= 39) {
    /* Phase D: feedback disabled. PRM 80-N2040-62 AA §2.3.3 Table 6
     * (p.15): "For convert transfer instruction without feedback, set
     * Rs[bias_sel] ... Set the rest of the register to 0" — so Rs[8]
     * (Rnd, polynomial-feedback rounding) stays 0 in BOTH trials and
     * only Rs[0]=1 (retain ACC) remains, matching every other
     * single-CVT mode in this probe. */
    first_rs = 1u;
  }
  asm volatile("cvt.hf = acc(%0)\n" : : "r"(first_rs) : "memory");
  asm volatile("mxmem(%0, %1) = cvt\n"
               : : "r"((uintptr_t)hi), "r"(wr_rt) : "memory");

  if (mode == 10) {
    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");
  }

  if (mode == 38) {
    /* Phase D readback probe. PRM §2.3.1 "Bias store instruction"
     * (p.15, hmx_hexagon_protos.h Q6_mxmem2_bias_A):
     *   void Q6_mxmem2_bias_A(void* A);   asm: mxmem2(Rs32)=bias
     * stores the bias register used by the convert transfer above back
     * to VTCM — 64 bits per output channel, address 256B aligned, the
     * register field is the same as the bias load (Fig.14: [31:8]
     * address, [1:0] set index), so Rs = scratch|0 stores set 0 (the
     * set that was loaded into and converted with). The scratch row is
     * 2048-aligned (hence 256B-aligned) at base+5*TILE_BYTES, inside
     * the 12 KiB reserve. Cells follow the Fig.13 layout: lo(0) at
     * bytes 0..3, hi(0) at bytes 128..131 — NOT bytes 4..7, the two
     * 32-bit halves of channel 0 live in the two 128-byte vectors. */
    uint8_t *scratch = base + 5 * TILE_BYTES;
    memset(scratch, 0xa5, BIAS_BYTES);  /* a skipped store stays visible */
    const uintptr_t scratch_reg = (uintptr_t)scratch;  /* set index 0 */
    asm volatile("mxmem2(%0) = bias\n" : : "r"(scratch_reg) : "memory");
    const uint32_t *cells = (const uint32_t *)scratch;
    *hi_bits = (int)(cells[0]);    /* lo(0): scale|obias fp16 pair */
    *residual_bits = (int)(cells[32]); /* hi(0): extra4|shape nibbles */
    return;
  }

  if (mode == 7 || mode == 8 || mode == 12 || mode == 13 || mode == 14 || mode == 15 || mode == 16 || mode == 17 || mode == 18 || mode == 19 || mode == 21 || mode == 23 || mode == 24 || mode == 25 || mode == 26 || mode == 27 || mode == 28 || mode == 29 || (mode >= 31 && mode <= 39)) {
    *hi_bits = hi[0];
    *residual_bits = 0;
    return;
  }

  // Rs[3:2]=1 -> previous convert state used as output bias.
  // Rs[4]=0 -> min; Rs[0]=0 -> clear/swap after convert.
  // This is a *hypothesis test*, not proof that h-x survives rounding.
  uint32_t second_rs;
  if (mode == 6) {
    second_rs = 0;  // bias set 0, no fb
  } else {
    second_rs = (1u << 12);  // bias set 1
    if (mode == 0 || mode == 3 || mode == 10) second_rs |= (1u << 2);  // fb_dest=1
  }
  asm volatile("cvt.hf = acc(%0)\n" : : "r"(second_rs) : "memory");
  asm volatile("mxmem(%0, %1) = cvt\n"
               : : "r"((uintptr_t)residual), "r"(wr_rt) : "memory");

  *hi_bits = hi[0];
  *residual_bits = residual[0];
}

/* E-3C1R (mode 43): corrected-packing, plain-only, ONE-trial confirmation.
 *
 * Same fixture, packet shape, bias0 scale=64 and no-feedback convert as
 * mode 42's plain trial. The ONLY intended data change is that the plain
 * Ahi/Whi cells are written at the PRM pair-interleaved VTCM offsets
 * instead of row-major:
 *   act_u16(s,i)   = (s>>1)*64 + i*2 + (s&1)
 *   wgt_u16(o,i)   = (i>>1)*64 + o*2 + (i&1)
 * The hardware store writes convert cell (s,o) at
 *   store_u16(s,o) = (s>>1)*64 + o*2 + (s&1)
 * so the ONE converted plain tile is read back at the four slots
 * results[0]=out[0], [1]=out[1], [2]=out[2], [3]=out[32].
 *
 * The fixture construction below is a VERBATIM copy of the mode-42 block;
 * it is intentionally not factored out so mode 42's historical code path
 * stays untouched. Any change here must be mirrored in mode 42 and
 * re-validated by hmx_prm_layout_sim.c.
 *
 * Pre-registered expectations (fixed before the run; do not retune):
 *   0x5400, 0xd400 (signed) or 0x0000 (clamp), 0x5400, 0x55e3.
 */
static void run_mode43_one_trial(uint8_t *base, int *results) {
  uint16_t *act = (uint16_t *)(base + 0 * TILE_BYTES);
  uint16_t *wgt = (uint16_t *)(base + 1 * TILE_BYTES);
  uint32_t *bias0 = (uint32_t *)(base + 4 * TILE_BYTES);
  uint32_t *bias1 = (uint32_t *)((uint8_t *)bias0 + BIAS_BYTES);
  uint16_t *out = (uint16_t *)(base + 5 * TILE_BYTES);
  int idx;
  int s, o, i;

  /* ---- t8_build port: statement order verbatim (mode 42) ---- */
  e3c1_rng = 0xE3C0117E5EEDF00Dull;
  for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
    double u = e3c1_rng_unit();
    double v = u * 4.0;
    double val = v - 2.0;
    e3c1_w[idx] = val;
    u = e3c1_rng_unit();
    v = u * 4.0;
    val = v - 2.0;
    e3c1_x[idx] = val;
  }

  /* Special W rows: out-ch 0..3 case-class anchors, 28..31 s3
   * subnormal-edge targets; out-ch 4..27 stay seeded realistic. */
  for (o = 0; o < E3C1_T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < E3C1_T8; ++i) e3c1_w[(long)o * E3C1_T8 + i] = 0.0;
    }
  }
  e3c1_w[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11; /* positive split-sensitive row */
  e3c1_w[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_w[0 * E3C1_T8 + 2] = 0x1p-13;
  e3c1_w[1 * E3C1_T8 + 0] = 1.0 - 0x1p-12;  /* B-2 case-1 anchor row */
  e3c1_w[2 * E3C1_T8 + 0] = 0x1p-9;         /* s3 boundary row (64*acc=2^-14) */
  e3c1_w[3 * E3C1_T8 + 0] = 1.0;            /* near-zero cancellation row */
  e3c1_w[3 * E3C1_T8 + 1] = 1.0 + 0x1p-12;
  e3c1_w[28 * E3C1_T8 + 0] = 0x1p-10;
  e3c1_w[29 * E3C1_T8 + 0] = 0x1p-10 - 0x1p-21;
  e3c1_w[30 * E3C1_T8 + 0] = 3.0 * 0x1p-12;
  e3c1_w[31 * E3C1_T8 + 0] = 0x1p-13;

  /* Special A lanes 0..3; lanes 4..31 stay seeded realistic. */
  for (s = 0; s <= 3; ++s)
    for (i = 0; i < E3C1_T8; ++i) e3c1_x[(long)s * E3C1_T8 + i] = 0.0;
  e3c1_x[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11; /* s0: positive multi-channel sum */
  e3c1_x[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_x[0 * E3C1_T8 + 2] = 0x1p-13;
  for (i = 0; i < E3C1_T8; ++i)          /* s1 = negation of s0 */
    e3c1_x[1 * E3C1_T8 + i] = -e3c1_x[i];
  e3c1_x[2 * E3C1_T8 + 0] = 1.0 - 0x1p-12; /* s2: B-2 case-1 + negated twin */
  e3c1_x[2 * E3C1_T8 + 1] = -(1.0 - 0x1p-12);
  e3c1_x[3 * E3C1_T8 + 0] = 0x1p-10;       /* s3: subnormal-edge lane */

  /* Plain operands: hi = RN16(value), exactly the tiles mode 42's plain
   * trial uses. Pack each logical cell at the PRM offsets; every one of
   * the 1024 u16 lanes is written exactly once (checked on CPU). */
  memset(act, 0, TILE_BYTES);
  memset(wgt, 0, TILE_BYTES);
  for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
    const int row = idx >> 5; /* activation spatial / weight output channel */
    const int col = idx & 31; /* input channel */
    const uint16_t ah = double_to_fp16(e3c1_x[idx]);
    const uint16_t wh = double_to_fp16(e3c1_w[idx]);
    act[((row >> 1) * 64) + col * 2 + (row & 1)] = ah;
    wgt[((col >> 1) * 64) + row * 2 + (col & 1)] = wh;
  }

  init_bias(bias0, 0x5400, 0, 0);
  init_bias(bias1, 0, 0, 0);
  memset(out, 0xa5, TILE_BYTES);
  {
    const uintptr_t bias0_reg = (uintptr_t)bias0;
    const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
    const uintptr_t act_reg = (uintptr_t)act;
    const uintptr_t wgt_reg = (uintptr_t)wgt;
    const uintptr_t out_reg = (uintptr_t)out;
    const uint32_t spatial_mask = (14u << 7);
    const uint32_t act_rt = spatial_mask | (31u << 2);
    const uint32_t wgt_rt = (uint32_t)(TILE_BYTES - 1);
    const uint32_t wr_rt = spatial_mask;

    asm volatile("mxclracc.hf\n" ::: "memory");
    /* Plain: ONE packet, deliberately the only MMA in this mode. */
    asm volatile(
        "{ activation.hf = mxmem(%0, %1)\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        : : "r"(act_reg), "r"(act_rt), "r"(wgt_reg), "r"(wgt_rt)
        : "memory");

    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");

    /* One final convert (bias set 0, retain, no feedback rounding,
     * Rs = 1: Rs[0] only) then the single FP16 store. */
    asm volatile("cvt.hf = acc(%0)\n" : : "r"(1u) : "memory");
    asm volatile("mxmem(%0, %1) = cvt\n"
                 : : "r"(out_reg), "r"(wr_rt) : "memory");
  }

  results[0] = (int)out[0];
  results[1] = (int)out[1];
  results[2] = (int)out[2];
  results[3] = (int)out[32];
}

int hexatrain_hmx_acc_residual_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  if (!handle) return HMX_ERR_ARG;
  probe_context_t *context = (probe_context_t *)calloc(1, sizeof(*context));
  if (!context) return HMX_ERR_ALLOC;
  context->tag = HMX_MAGIC;
  *handle = (remote_handle64)(uintptr_t)context;
  return 0;
}

int hexatrain_hmx_acc_residual_close(remote_handle64 handle) {
  probe_context_t *context = (probe_context_t *)(uintptr_t)handle;
  if (!context || context->tag != HMX_MAGIC) return HMX_ERR_ARG;
  context->tag = 0;
  free(context);
  return 0;
}

int hexatrain_hmx_acc_residual_run(remote_handle64 handle,
                                   int *results, int resultsLen,
                                   int *metadata, int metadataLen,
                                   int q, int mode) {
  probe_context_t *context = (probe_context_t *)(uintptr_t)handle;
  if (!context || context->tag != HMX_MAGIC ||
       !results || resultsLen < 4 || !metadata || metadataLen < 9) {
    return HMX_ERR_ARG;
  }
  if (q < 6 || q > 10 || mode < 0 || mode > MODE_MAX) {
    return HMX_ERR_ARG;
  }
  /* Some modes are bound to a specific q (fixed bias values). */
  if ((mode == 24 || mode == 25 || mode == 26) && q != 6) return HMX_ERR_ARG;
  if ((mode == 27 || mode == 28 || mode == 29) && q != 8) return HMX_ERR_ARG;
  if (mode == 18 && q != 6) return HMX_ERR_ARG;
  if (mode == 22 && q != 6) return HMX_ERR_ARG;
  /* B-1 (mode 40) uses fixed operands independent of q; the host mirrors
   * q=6 like modes 18/22. */
  if (mode == 40 && q != 6) return HMX_ERR_ARG;
  /* B-2 (mode 41) also has fixed fixtures independent of q; host mirrors. */
  if (mode == 41 && q != 6) return HMX_ERR_ARG;
  /* E-3C1 (mode 42) runs the fixed realistic tile independent of q;
   * the host mirrors q=6 like modes 18/22/40/41. E-3C1R (mode 43) reuses
   * that fixture/packet and is likewise bound to q=6. */
  if (mode == 42 && q != 6) return HMX_ERR_ARG;
  if (mode == 43 && q != 6) return HMX_ERR_ARG;
  /* Phase D modes 31..39 are hard-bound to q=8 (fixed -64.5/-64.25/-64.0
   * payload and 1+2^-8 operand); the host mirrors this gate. */
  if (mode >= 31 && mode <= 39 && q != 8) return HMX_ERR_ARG;
  memset(results, 0, (size_t)resultsLen * sizeof(int));
  memset(metadata, 0, (size_t)metadataLen * sizeof(int));

  /* Identity mode: no HMX execution, no resource acquisition.
   * results[0..3] carry the DSP-side revision, q, mode and arch so the
   * host can verify that the deployed skel matches this source.
   * metadata[2] is unused in this mode (normal runs report VTCM bytes
   * there), so it echoes the build id; 0 means "unknown / legacy build". */
  if (mode == MODE_IDENTITY) {
    results[0] = PROBE_REVISION;
    results[1] = q;
    results[2] = mode;
    results[3] = __HEXAGON_ARCH__;
    metadata[0] = HMX_MAGIC;
    metadata[1] = __HEXAGON_ARCH__;
    metadata[HMX_IDENTITY_BUILD_ID_SLOT] = HMX_PROBE_BUILD_ID;
    metadata[7] = 0;  // HMX not executed
    metadata[8] = 0;  // no trials
    return 0;
  }
  metadata[0] = HMX_MAGIC;
  metadata[1] = __HEXAGON_ARCH__;
  metadata[2] = VTCM_BYTES;
  metadata[3] = 0;  // DSP kernel microseconds
  metadata[4] = 0;  // power status
  metadata[5] = 0;  // mode A (HMX-only) lock status
  metadata[6] = 0;  // mode B (combined) lock status
  metadata[7] = 0;  // HMX executed flag
  metadata[8] = 0;  // trial count

  static int power_client = 0;
  HAP_power_request_t power_req;
  memset(&power_req, 0, sizeof(power_req));
  power_req.type = HAP_power_set_HMX_v2;
  power_req.hmx_v2.set_power = 1;
  power_req.hmx_v2.power_up = 1;
  power_req.hmx_v2.set_clock = 0;
  int power_status = HAP_power_set(&power_client, &power_req);
  metadata[4] = power_status;
  if (power_status) {
    /* Early exit BEFORE anything was acquired: no compute_res resource and
     * no VTCM reserve exist yet, so there is nothing to unlock or release
     * here. The defensive power-down request below stays (it can only clear
     * a partially granted vote), and the EARLIEST failing stage wins:
     * HMX_ERR_POWER (14006) regardless of the defensive request's result. */
    HAP_power_request_t power_down;
    memset(&power_down, 0, sizeof(power_down));
    power_down.type = HAP_power_set_HMX_v2;
    power_down.hmx_v2.set_power = 1;
    power_down.hmx_v2.power_up = 0;
    power_down.hmx_v2.set_clock = 0;
    (void)HAP_power_set(&power_client, &power_down);
    return HMX_ERR_POWER;
  }

  int combined_lock_status = 0;
  int hmx_only_lock_status = 0;
  int hmx_only_unlock_status = 0;
  int hmx_only_release_status = 0;
  int combined_unlock_status = 0;
  int combined_release_status = 0;
  int power_down_status = 0;
  uint8_t *vtcm = NULL;
  int combined_resource = 0;
  int hmx_only_resource = 0;
  const uint64_t start = HAP_perf_get_time_us();

  /* Mode A: HMX-only resource (htp-ops-lib style).
   * Cleanup stage (a)+(b) for this resource, executed exactly once:
   * unlock2 only if lock2 returned 0, then release. A failed lock means
   * there is no held lock, so unlock is skipped (never unlock un-locked). */
  {
    compute_res_attr_t attr;
    int status = HAP_compute_res_attr_init(&attr);
    if (!status) status = HAP_compute_res_attr_set_hmx_param(&attr, 1);
    if (!status) {
      hmx_only_resource = HAP_compute_res_acquire(&attr, 10000);
      if (hmx_only_resource) {
        hmx_only_lock_status = HAP_compute_res_hmx_lock2(
            hmx_only_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
        if (!hmx_only_lock_status) {
          hmx_only_unlock_status = HAP_compute_res_hmx_unlock2(
              hmx_only_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
        }
        hmx_only_release_status =
            HAP_compute_res_release(hmx_only_resource);
        hmx_only_resource = 0;
      } else {
        hmx_only_lock_status = -1;
      }
    } else {
      hmx_only_lock_status = status;
    }
  }
  metadata[5] = hmx_only_lock_status;

  /* Mode B: combined VTCM + HMX resource (current style).
   * Same single-pass cleanup: unlock2 only after a lock2 success, then
   * release. A VTCM reserve failure (HMX_ERR_VTCM) means the lock was
   * never taken, so unlock is skipped and only release runs. */
  {
    compute_res_attr_t attr;
    int status = HAP_compute_res_attr_init(&attr);
    if (!status) status = HAP_compute_res_attr_set_vtcm_param(&attr, VTCM_BYTES, 1);
    if (!status) status = HAP_compute_res_attr_set_hmx_param(&attr, 1);
    if (!status) {
      combined_resource = HAP_compute_res_acquire(&attr, 10000);
      if (combined_resource) {
        vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
        if (!vtcm || ((uintptr_t)vtcm & (TILE_BYTES - 1u))) {
          combined_lock_status = HMX_ERR_VTCM;
        } else {
          combined_lock_status = HAP_compute_res_hmx_lock2(
              combined_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
          metadata[6] = combined_lock_status;
          if (!combined_lock_status) {
            /* Execute HMX only if lock succeeded. */
            if (mode == 43) {
              /* E-3C1R is genuinely ONE trial: a dedicated handler reads
               * four cells from ONE converted plain tile. Do not force
               * four cells into two run_one() scalar outputs. */
              run_mode43_one_trial(vtcm, results);
              metadata[7] = 1;
              metadata[8] = HMX_MODE43_TRIALS;
            } else {
              run_one(vtcm, 0, (int)q, mode, &results[0], &results[1]);
              run_one(vtcm, 1, (int)q, mode, &results[2], &results[3]);
              if (mode == 38) {
                /* Mode 38 contract: the bias readback payload lives in
                 * results[0..1] only (lo(0), hi(0)); slots 2/3 are zeroed.
                 * Both trials still execute, so metadata[8]=2 stays true. */
                results[2] = 0;
                results[3] = 0;
              }
              metadata[7] = 1;
              metadata[8] = 2;
            }
            combined_unlock_status = HAP_compute_res_hmx_unlock2(
                combined_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
          }
        }
        combined_release_status =
            HAP_compute_res_release(combined_resource);
        combined_resource = 0;
      } else {
        combined_lock_status = -1;
        metadata[6] = combined_lock_status;
      }
    } else {
      combined_lock_status = status;
      metadata[6] = combined_lock_status;
    }
  }

  const uint64_t elapsed = HAP_perf_get_time_us() - start;
  metadata[3] = elapsed > 0x7fffffffULL ? 0x7fffffff : (int)elapsed;

  /* Cleanup stage (c): power down after every acquired resource is gone. */
  {
    HAP_power_request_t power_down;
    memset(&power_down, 0, sizeof(power_down));
    power_down.type = HAP_power_set_HMX_v2;
    power_down.hmx_v2.set_power = 1;
    power_down.hmx_v2.power_up = 0;
    power_down.hmx_v2.set_clock = 0;
    power_down_status = HAP_power_set(&power_client, &power_down);
  }

  /* Resource cleanup return codes (the metadata layout is ABI-fixed at 9
   * slots, so cleanup failures are reported through the run() status only):
   *
   *   stage order (also the tie-break order when several stages fail):
   *     (a-1) mode A hmx_unlock2  -> 14007 HMX_ERR_UNLOCK2
   *     (a-2) mode A release      -> 14008 HMX_ERR_RELEASE
   *     (b-1) mode B hmx_unlock2  -> 14007 HMX_ERR_UNLOCK2
   *     (b-2) mode B release      -> 14008 HMX_ERR_RELEASE
   *     (c)   HAP_power_set down  -> 14009 HMX_ERR_POWER_DOWN
   *   The FIRST failing stage is returned; unlock2 is only ever called
   *   after a lock2 success and release only after an acquire success, so
   *   no stage is ever called twice or out of order.
   *
   * metadata stays truthful on every path: metadata[4] keeps the power-up
   * status, metadata[5]/[6] keep the lock2 statuses, metadata[7]/[8] keep
   * the execution status. A cleanup failure therefore does not overwrite
   * any metadata slot. */
  int cleanup_error = 0;
  if (cleanup_error == 0 && hmx_only_unlock_status != 0) {
    cleanup_error = HMX_ERR_UNLOCK2;
  }
  if (cleanup_error == 0 && hmx_only_release_status != 0) {
    cleanup_error = HMX_ERR_RELEASE;
  }
  if (cleanup_error == 0 && combined_unlock_status != 0) {
    cleanup_error = HMX_ERR_UNLOCK2;
  }
  if (cleanup_error == 0 && combined_release_status != 0) {
    cleanup_error = HMX_ERR_RELEASE;
  }
  if (cleanup_error == 0 && power_down_status != 0) {
    cleanup_error = HMX_ERR_POWER_DOWN;
  }

  /* A cleanup failure must be visible to the host on any path where
   * resources were acquired, so it takes precedence; the execution state
   * remains readable in metadata[7] (and the lock stages in [5]/[6]). */
  if (cleanup_error != 0) return cleanup_error;
  if (metadata[7] == 1) return 0;
  return HMX_ERR_LOCK;
}
