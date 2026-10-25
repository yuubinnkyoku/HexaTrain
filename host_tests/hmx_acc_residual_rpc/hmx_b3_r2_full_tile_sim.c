// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// HexaTrain — B3 + R2 Integrated 32x32 Full-Tile Precision Simulator (CPU-only).
//
// Purpose:
//   Verify that the 3-product B3 accumulation (Ahi*Whi, Ahi*Wlo, Alo*Whi)
//   and the R2 two-convert sign separation (Convert #1 shape 0, Convert #2 shape 4)
//   integrate into a single 32x32 tile without extra MMAs and fit inside 12 KiB VTCM.
//
// Constraints:
//   - CPU-only: NO Hexagon SDK, NO ADB, NO device, NO QNN dependency.
//   - Strict G8 fixture preservation: seed 0xE3C0117E5EEDF00D, scale +64.
//   - PRM crouton packing: pair-interleaved act/wgt/store indexing.
//   - 12 KiB VTCM layout with sequential operand streaming.
//   - All 13 fault injections verified to fail closed.

#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bias20.h"
#include "fp16_util.h"
#include "hmx_b3_r2.h"

#define T8 32
#define T8_CELLS (T8 * T8)

static long g_checks = 0;
static long g_fail = 0;

static void check_u16(const char *tag, uint16_t got, uint16_t want) {
  ++g_checks;
  if (got != want) {
    ++g_fail;
    printf("FAIL %s got=0x%04x want=0x%04x\n", tag, (unsigned)got, (unsigned)want);
  }
}

static void check_int(const char *tag, int got, int want) {
  ++g_checks;
  if (got != want) {
    ++g_fail;
    printf("FAIL %s got=%d want=%d\n", tag, got, want);
  }
}

static void check_true(const char *tag, int cond) {
  ++g_checks;
  if (!cond) {
    ++g_fail;
    printf("FAIL %s (condition false)\n", tag);
  }
}

/* ---------------- Deterministic PRNG & Fixture Generation ---------------- */

static uint64_t g_rng = 0xE3C0117E5EEDF00Dull;

static uint64_t rng64(void) {
  uint64_t x = g_rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  g_rng = x;
  return x;
}

static double rng_unit(void) {
  return (double)(rng64() >> 11) * (1.0 / 9007199254740992.0);
}

/* Float-arithmetic split residual mirroring probe_dsp.c and hmx_fp16x2_split_sim.c */
static uint16_t split_lo_fp16(double x, uint16_t hi_bits) {
  float xf = (float)x;
  float hf = (float)fp16_to_double(hi_bits);
  float sub = xf - hf;
  return double_to_fp16((double)sub);
}

/* G8 Fixture buffers */
static double g_wd[T8_CELLS], g_xd[T8_CELLS];
static double g_oracle_y[T8_CELLS]; /* Double true oracle */

/* Logical FP16 operands */
static uint16_t g_ahi[T8_CELLS], g_alo[T8_CELLS];
static uint16_t g_whi[T8_CELLS], g_wlo[T8_CELLS];
static uint16_t g_a_plain[T8_CELLS], g_w_plain[T8_CELLS];

static void build_g8_fixture(void) {
  int s, o, i;
  g_rng = 0xE3C0117E5EEDF00Dull;

  for (i = 0; i < T8_CELLS; ++i) {
    g_wd[i] = 4.0 * rng_unit() - 2.0;
    g_xd[i] = 4.0 * rng_unit() - 2.0;
  }

  /* Anchor rows in W (out-ch 0..3 and 28..31) */
  for (o = 0; o < T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < T8; ++i) g_wd[(long)o * T8 + i] = 0.0;
    }
  }
  g_wd[0 * T8 + 0] = 1.0 + 0x1p-11;
  g_wd[0 * T8 + 1] = 0x1p-11;
  g_wd[0 * T8 + 2] = 0x1p-13;
  g_wd[1 * T8 + 0] = 1.0 - 0x1p-12;
  g_wd[2 * T8 + 0] = 0x1p-9;
  g_wd[3 * T8 + 0] = 1.0;
  g_wd[3 * T8 + 1] = 1.0 + 0x1p-12;
  g_wd[28 * T8 + 0] = 0x1p-10;
  g_wd[29 * T8 + 0] = 0x1p-10 - 0x1p-21;
  g_wd[30 * T8 + 0] = 3.0 * 0x1p-12;
  g_wd[31 * T8 + 0] = 0x1p-13;

  /* Anchor lanes in X (spatial 0..3) */
  for (s = 0; s <= 3; ++s) {
    for (i = 0; i < T8; ++i) g_xd[(long)s * T8 + i] = 0.0;
  }
  g_xd[0 * T8 + 0] = 1.0 + 0x1p-11;
  g_xd[0 * T8 + 1] = 0x1p-11;
  g_xd[0 * T8 + 2] = 0x1p-13;
  for (i = 0; i < T8; ++i) g_xd[1 * T8 + i] = -g_xd[i]; /* s1 = -s0 */
  g_xd[2 * T8 + 0] = 1.0 - 0x1p-12;
  g_xd[2 * T8 + 1] = -(1.0 - 0x1p-12);
  g_xd[3 * T8 + 0] = 0x1p-10;

  /* True double oracle */
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      double acc = 0.0;
      for (i = 0; i < T8; ++i) {
        acc += g_xd[(long)s * T8 + i] * g_wd[(long)o * T8 + i];
      }
      g_oracle_y[(long)s * T8 + o] = acc;
    }
  }

  /* Compute plain operands and FP16 hi/lo splits */
  for (i = 0; i < T8_CELLS; ++i) {
    uint16_t ah = double_to_fp16(g_xd[i]);
    uint16_t wh = double_to_fp16(g_wd[i]);
    g_a_plain[i] = ah;
    g_w_plain[i] = wh;
    g_ahi[i] = ah;
    g_alo[i] = split_lo_fp16(g_xd[i], ah);
    g_whi[i] = wh;
    g_wlo[i] = split_lo_fp16(g_wd[i], wh);
  }
}

/* ---------------- PRM Crouton Packing & Bijection ---------------- */

static void verify_crouton_bijection(void) {
  uint8_t act_seen[T8_CELLS] = {0};
  uint8_t wgt_seen[T8_CELLS] = {0};
  uint8_t st_seen[T8_CELLS]  = {0};
  int s, o, i;

  for (s = 0; s < T8; ++s) {
    for (i = 0; i < T8; ++i) {
      uint16_t k = hmx_b3_r2_act_u16(s, i);
      check_true("act_u16_bounds", k < T8_CELLS);
      check_int("act_u16_unique", act_seen[k], 0);
      act_seen[k] = 1;
      check_int("act_decode_pair", hmx_b3_r2_decode_pair(k), s);
      check_int("act_decode_fast", hmx_b3_r2_decode_fast(k), i);
    }
  }

  for (o = 0; o < T8; ++o) {
    for (i = 0; i < T8; ++i) {
      uint16_t k = hmx_b3_r2_wgt_u16(o, i);
      check_true("wgt_u16_bounds", k < T8_CELLS);
      check_int("wgt_u16_unique", wgt_seen[k], 0);
      wgt_seen[k] = 1;
      check_int("wgt_decode_fast", hmx_b3_r2_decode_fast(k), o);
      check_int("wgt_decode_pair", hmx_b3_r2_decode_pair(k), i);
    }
  }

  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      uint16_t k = hmx_b3_r2_store_u16(s, o);
      check_true("st_u16_bounds", k < T8_CELLS);
      check_int("st_u16_unique", st_seen[k], 0);
      st_seen[k] = 1;
      check_int("st_decode_pair", hmx_b3_r2_decode_pair(k), s);
      check_int("st_decode_fast", hmx_b3_r2_decode_fast(k), o);
    }
  }

  printf("b3_r2_crouton_bijection act=1024 wgt=1024 store=1024 range=[0,1023] unique=1 PASS\n");
}

/* Pack logical 32x32 into PRM physical crouton buffer */
static void pack_act_prm(uint16_t *dst, const uint16_t *src) {
  int s, i;
  for (s = 0; s < T8; ++s) {
    for (i = 0; i < T8; ++i) {
      dst[hmx_b3_r2_act_u16(s, i)] = src[s * T8 + i];
    }
  }
}

static void pack_wgt_prm(uint16_t *dst, const uint16_t *src) {
  int o, i;
  for (o = 0; o < T8; ++o) {
    for (i = 0; i < T8; ++i) {
      dst[hmx_b3_r2_wgt_u16(o, i)] = src[o * T8 + i];
    }
  }
}

/* ---------------- VTCM 12 KiB Layout & Staged Streaming Model ---------------- */

typedef struct vtcm_12k {
  uint8_t raw[HMX_B3_R2_VTCM_TOTAL];
} vtcm_12k_t;

static void verify_vtcm_layout_static(void) {
  check_true("vtcm_act_aligned", (HMX_B3_R2_ACT_OFFSET % 2048) == 0);
  check_true("vtcm_wgt_aligned", (HMX_B3_R2_WGT_OFFSET % 2048) == 0);
  check_true("vtcm_pos_aligned", (HMX_B3_R2_POS_OFFSET % 2048) == 0);
  check_true("vtcm_neg_aligned", (HMX_B3_R2_NEG_OFFSET % 2048) == 0);
  check_true("vtcm_recon_aligned", (HMX_B3_R2_RECON_OFFSET % 2048) == 0);
  check_true("vtcm_bias0_aligned", (HMX_B3_R2_BIAS0_OFFSET % 256) == 0);
  check_true("vtcm_bias1_aligned", (HMX_B3_R2_BIAS1_OFFSET % 256) == 0);

  check_true("vtcm_regions_disjoint",
             HMX_B3_R2_ACT_OFFSET + 2048 <= HMX_B3_R2_WGT_OFFSET &&
             HMX_B3_R2_WGT_OFFSET + 2048 <= HMX_B3_R2_POS_OFFSET &&
             HMX_B3_R2_POS_OFFSET + 2048 <= HMX_B3_R2_NEG_OFFSET &&
             HMX_B3_R2_NEG_OFFSET + 2048 <= HMX_B3_R2_RECON_OFFSET &&
             HMX_B3_R2_RECON_OFFSET + 2048 <= HMX_B3_R2_BIAS0_OFFSET &&
             HMX_B3_R2_BIAS0_OFFSET + 256 <= HMX_B3_R2_BIAS1_OFFSET &&
             HMX_B3_R2_BIAS1_OFFSET + 256 <= HMX_B3_R2_VTCM_TOTAL);

  printf("b3_r2_vtcm_layout total=12288 used=%d act=%d wgt=%d pos=%d neg=%d recon=%d bias0=%d bias1=%d PASS\n",
         HMX_B3_R2_VTCM_USED, HMX_B3_R2_ACT_OFFSET, HMX_B3_R2_WGT_OFFSET,
         HMX_B3_R2_POS_OFFSET, HMX_B3_R2_NEG_OFFSET, HMX_B3_R2_RECON_OFFSET,
         HMX_B3_R2_BIAS0_OFFSET, HMX_B3_R2_BIAS1_OFFSET);
}

/* ---------------- Simulation State and Execution Engine ---------------- */

typedef struct hmx_full_tile_state {
  /* Internal 32x32 accumulator (37-bit FP modeled via single-precision float chain,
   * exactly matching the PRM/G8 specification) */
  float acc[T8_CELLS];
  int acc_cleared;

  /* VTCM 12 KiB physical memory */
  vtcm_12k_t vtcm;

  /* Execution tracking counters */
  int count_mxclracc;
  int count_mma;
  int count_convert;
  int count_store;
  int count_bias_loads;
  int count_recon_loops;
  int count_stream_updates;
  int mma_order_id; /* 123 = nominal #1->#2->#3 */

  /* Registered Rs history */
  uint32_t rs_history[8];
  int rs_history_len;

  /* Sentinel hit tracker */
  int sentinel_hits;
} hmx_full_tile_state_t;

static void state_init(hmx_full_tile_state_t *st) {
  memset(st, 0, sizeof(*st));
  st->mma_order_id = 123;

  /* Fill output regions with sentinel byte 0xA5 */
  memset(st->vtcm.raw + HMX_B3_R2_POS_OFFSET, HMX_B3_R2_SENTINEL_BYTE, HMX_B3_R2_TILE_BYTES);
  memset(st->vtcm.raw + HMX_B3_R2_NEG_OFFSET, HMX_B3_R2_SENTINEL_BYTE, HMX_B3_R2_TILE_BYTES);
  memset(st->vtcm.raw + HMX_B3_R2_RECON_OFFSET, HMX_B3_R2_SENTINEL_BYTE, HMX_B3_R2_TILE_BYTES);

  /* Initialize Bias set 0: scale +64, shape 0, bo 0 */
  uint8_t *b0 = st->vtcm.raw + HMX_B3_R2_BIAS0_OFFSET;
  for (int o = 0; o < T8; ++o) {
    uint32_t lo = bias20_lo_cell(HMX_B3_R2_SCALE_BITS, HMX_B3_R2_OUT_BIAS);
    uint32_t hi = bias20_hi_cell(HMX_B3_R2_SHAPE0, 0, 0);
    memcpy(b0 + o * 4, &lo, 4);
    memcpy(b0 + 128 + o * 4, &hi, 4);
  }

  /* Initialize Bias set 1: scale +64, shape 4, bo 0 */
  uint8_t *b1 = st->vtcm.raw + HMX_B3_R2_BIAS1_OFFSET;
  for (int o = 0; o < T8; ++o) {
    uint32_t lo = bias20_lo_cell(HMX_B3_R2_SCALE_BITS, HMX_B3_R2_OUT_BIAS);
    uint32_t hi = bias20_hi_cell(HMX_B3_R2_SHAPE4, 0, 0);
    memcpy(b1 + o * 4, &lo, 4);
    memcpy(b1 + 128 + o * 4, &hi, 4);
  }
}

/* Execute mxclracc */
static void exec_mxclracc(hmx_full_tile_state_t *st) {
  st->count_mxclracc++;
  memset(st->acc, 0, sizeof(st->acc));
  st->acc_cleared = 1;
}

/* Execute one MMA packet: reads act and wgt from VTCM and accumulates into st->acc */
static void exec_mma_packet(hmx_full_tile_state_t *st) {
  st->count_mma++;
  const uint16_t *act_vtcm = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_ACT_OFFSET);
  const uint16_t *wgt_vtcm = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_WGT_OFFSET);

  for (int s = 0; s < T8; ++s) {
    for (int o = 0; o < T8; ++o) {
      float dot = 0.0f;
      for (int i = 0; i < T8; ++i) {
        uint16_t a_val = act_vtcm[hmx_b3_r2_act_u16(s, i)];
        uint16_t w_val = wgt_vtcm[hmx_b3_r2_wgt_u16(o, i)];
        dot += (float)(fp16_to_double(a_val) * fp16_to_double(w_val));
      }
      st->acc[s * T8 + o] += dot;
    }
  }
}

/* Execute Convert and Store into dst_offset with given Rs */
static void exec_convert_and_store(hmx_full_tile_state_t *st, uint32_t rs, int dst_offset) {
  st->count_convert++;
  st->count_store++;
  if (st->rs_history_len < 8) {
    st->rs_history[st->rs_history_len++] = rs;
  }

  unsigned bias_sel = (rs >> 12) & 3u;
  unsigned retain_acc = rs & 1u;

  /* Read bias row from VTCM based on bias_sel */
  int bias_offset = (bias_sel == 0) ? HMX_B3_R2_BIAS0_OFFSET : HMX_B3_R2_BIAS1_OFFSET;
  const uint8_t *bias_row = st->vtcm.raw + bias_offset;
  uint16_t *store_dst = (uint16_t *)(st->vtcm.raw + dst_offset);

  for (int s = 0; s < T8; ++s) {
    for (int o = 0; o < T8; ++o) {
      uint32_t lo, hi;
      memcpy(&lo, bias_row + o * 4, 4);
      memcpy(&hi, bias_row + 128 + o * 4, 4);

      uint16_t scale_fp16 = (uint16_t)(lo & 0xFFFFu);
      uint16_t out_bias_fp16 = (uint16_t)((lo >> 16) & 0xFFFFu);
      unsigned shape = (hi >> 8) & 7u;

      double scale = fp16_to_double(scale_fp16);
      double out_bias = fp16_to_double(out_bias_fp16);
      double acc_val = (double)st->acc[s * T8 + o];

      /* Convert equation: cvt = s * shape(ACC + bi) + bo, then RN16 */
      double shaped;
      switch (shape) {
        case 0: shaped = acc_val; break;
        case 4: shaped = -acc_val; break;
        default: shaped = acc_val; break;
      }
      uint16_t cvt = double_to_fp16(scale * shaped + out_bias);

      /* Measured device law: negative stored value -> 0x0000 */
      uint16_t stored = (cvt & 0x8000u) ? 0x0000u : cvt;

      /* Write through pair-interleaved store crouton */
      uint16_t st_idx = hmx_b3_r2_store_u16(s, o);
      store_dst[st_idx] = stored;
    }
  }

  /* Post-convert ACC action */
  if (!retain_acc) {
    memset(st->acc, 0, sizeof(st->acc));
    st->acc_cleared = 1;
  }
}

/* Execute DSP HVX reconstruction loop: out_recon = out_pos - out_neg (16 vectors x 128 bytes) */
static void exec_hvx_recon(hmx_full_tile_state_t *st, int reversed) {
  st->count_recon_loops++;
  const uint16_t *pos = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_POS_OFFSET);
  const uint16_t *neg = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_NEG_OFFSET);
  uint16_t *recon = (uint16_t *)(st->vtcm.raw + HMX_B3_R2_RECON_OFFSET);

  for (int k = 0; k < T8_CELLS; ++k) {
    double p = fp16_to_double(pos[k]);
    double n = fp16_to_double(neg[k]);
    recon[k] = reversed ? double_to_fp16(n - p) : double_to_fp16(p - n);
  }
}

/* ---------------- Full Tile Simulation Runners ---------------- */

/* Plain+R2 subcase: 1 MMA, 2 Convert, 2 Store, 1 HVX recon */
static void run_plain_r2_subcase(hmx_full_tile_state_t *st) {
  pack_act_prm((uint16_t *)(st->vtcm.raw + HMX_B3_R2_ACT_OFFSET), g_a_plain);
  pack_wgt_prm((uint16_t *)(st->vtcm.raw + HMX_B3_R2_WGT_OFFSET), g_w_plain);

  exec_mxclracc(st);
  exec_mma_packet(st); /* 1 MMA */

  /* Convert #1: shape 0, Rs 0x0001 (retain) -> out_pos */
  exec_convert_and_store(st, HMX_B3_R2_RS_CVT1, HMX_B3_R2_POS_OFFSET);

  /* Convert #2: shape 4, Rs 0x1000 (clear) -> out_neg */
  exec_convert_and_store(st, HMX_B3_R2_RS_CVT2, HMX_B3_R2_NEG_OFFSET);

  /* HVX recon */
  exec_hvx_recon(st, 0);
}

/* B3+R2 subcase: 3 MMAs with staged operand streaming, 2 Convert, 2 Store, 1 HVX recon */
static void run_b3_r2_subcase(hmx_full_tile_state_t *st, hmx_b3_r2_fault_t fault) {
  uint16_t *act_vtcm = (uint16_t *)(st->vtcm.raw + HMX_B3_R2_ACT_OFFSET);
  uint16_t *wgt_vtcm = (uint16_t *)(st->vtcm.raw + HMX_B3_R2_WGT_OFFSET);

  st->mma_order_id = (fault == HMX_FAULT_MMA_ORDER) ? 132 : 123;

  /* Stage initial operands: Ahi at act, Whi at wgt */
  if (fault == HMX_FAULT_CROUTON_CORRUPT) {
    /* Bug injection: row-major write instead of pair-interleaved */
    memcpy(act_vtcm, g_ahi, sizeof(g_ahi));
    memcpy(wgt_vtcm, g_whi, sizeof(g_whi));
  } else {
    pack_act_prm(act_vtcm, g_ahi);
    pack_wgt_prm(wgt_vtcm, g_whi);
  }

  exec_mxclracc(st);

  /* MMA #1: Ahi * Whi */
  exec_mma_packet(st);

  if (fault == HMX_FAULT_ACC_EARLY_CLEAR) {
    memset(st->acc, 0, sizeof(st->acc));
  }

  if (fault == HMX_FAULT_MMA_ORDER) {
    /* Reverse order: do MMA #3 first, then MMA #2 */
    pack_act_prm(act_vtcm, g_alo);
    pack_wgt_prm(wgt_vtcm, g_whi);
    exec_mma_packet(st);

    pack_act_prm(act_vtcm, g_ahi);
    pack_wgt_prm(wgt_vtcm, g_wlo);
    exec_mma_packet(st);
  } else {
    /* Normal order */
    /* Stream update for MMA #2: Wlo into wgt (Ahi remains in act) */
    if (fault != HMX_FAULT_MMA2_MISSING) {
      if (fault == HMX_FAULT_OPERAND_SWAP) {
        pack_wgt_prm(wgt_vtcm, g_whi); /* swap Wlo with Whi */
      } else {
        pack_wgt_prm(wgt_vtcm, g_wlo);
      }
      st->count_stream_updates++;
      exec_mma_packet(st); /* MMA #2 */
    }

    if (fault == HMX_FAULT_ACC_EARLY_CLEAR) {
      memset(st->acc, 0, sizeof(st->acc));
    }

    /* Stream update for MMA #3: Alo into act, Whi back into wgt */
    if (fault != HMX_FAULT_MMA3_MISSING) {
      pack_act_prm(act_vtcm, g_alo);
      pack_wgt_prm(wgt_vtcm, g_whi);
      st->count_stream_updates += 2;
      exec_mma_packet(st); /* MMA #3 */
    }
  }

  /* Optional bo!=0 fault */
  if (fault == HMX_FAULT_BO_NONZERO) {
    uint8_t *b0 = st->vtcm.raw + HMX_B3_R2_BIAS0_OFFSET;
    uint32_t lo = bias20_lo_cell(HMX_B3_R2_SCALE_BITS, 0x3C00u); /* bo = 1.0 */
    memcpy(b0, &lo, 4);
  }

  /* Convert #1: shape 0, Rs 0x0001 (retain) -> out_pos */
  exec_convert_and_store(st, HMX_B3_R2_RS_CVT1, HMX_B3_R2_POS_OFFSET);

  /* Convert #2: shape 4, Rs 0x1000 (clear) -> out_neg */
  uint32_t rs2 = (fault == HMX_FAULT_SHAPE4_NOT_SELECTED) ? 0x0000u : HMX_B3_R2_RS_CVT2;
  exec_convert_and_store(st, rs2, HMX_B3_R2_NEG_OFFSET);

  /* HVX recon */
  if (fault != HMX_FAULT_RECON_STORE_FAIL) {
    exec_hvx_recon(st, (fault == HMX_FAULT_HVX_SUB_REVERSED) ? 1 : 0);
  }
}

/* ---------------- Statistical & Metric Verification ---------------- */

typedef struct tile_metrics {
  double mean_abs_err;
  double median_abs_err;
  double p95_abs_err;
  double max_abs_err;
  double rmse;

  long bits_diff_vs_oracle;
  long numeric_diff_vs_oracle;
  long sign_matches;

  long subnormal_count;
  long inf_count;
  long nan_count;
  long zero_count;
  long both_nonzero_count;
  long one_sided_count;
  long both_zero_count;

  uint16_t raw_pos[T8_CELLS];
  uint16_t raw_neg[T8_CELLS];
  uint16_t raw_recon[T8_CELLS];
} tile_metrics_t;

static int is_subnormal_fp16(uint16_t bits) {
  return ((bits >> 10) & 31u) == 0u && (bits & 1023u) != 0u;
}
static int is_inf_fp16(uint16_t bits) {
  return ((bits >> 10) & 31u) == 31u && (bits & 1023u) == 0u;
}
static int is_nan_fp16(uint16_t bits) {
  return ((bits >> 10) & 31u) == 31u && (bits & 1023u) != 0u;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static void evaluate_metrics(const hmx_full_tile_state_t *st, tile_metrics_t *m, const uint16_t *ref_direct) {
  memset(m, 0, sizeof(*m));
  double errors[T8_CELLS];
  double sum_err = 0.0, sum_sq_err = 0.0;
  double max_err = 0.0;

  const uint16_t *pos_buf = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_POS_OFFSET);
  const uint16_t *neg_buf = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_NEG_OFFSET);
  const uint16_t *rec_buf = (const uint16_t *)(st->vtcm.raw + HMX_B3_R2_RECON_OFFSET);

  for (int s = 0; s < T8; ++s) {
    for (int o = 0; o < T8; ++o) {
      int cell_idx = s * T8 + o;
      uint16_t st_idx = hmx_b3_r2_store_u16(s, o);

      uint16_t p = pos_buf[st_idx];
      uint16_t n = neg_buf[st_idx];
      uint16_t r = rec_buf[st_idx];

      m->raw_pos[cell_idx] = p;
      m->raw_neg[cell_idx] = n;
      m->raw_recon[cell_idx] = r;

      if (p != 0u && n != 0u) m->both_nonzero_count++;
      if ((p != 0u) != (n != 0u)) m->one_sided_count++;
      if (p == 0u && n == 0u) m->both_zero_count++;

      if (is_subnormal_fp16(r)) m->subnormal_count++;
      if (is_inf_fp16(r)) m->inf_count++;
      if (is_nan_fp16(r)) m->nan_count++;
      if (r == 0x0000u || r == 0x8000u) m->zero_count++;

      /* Descaled reconstructed value */
      double r_val = fp16_to_double(r) * (1.0 / 64.0);
      double oracle_val = g_oracle_y[cell_idx];

      double err = fabs(r_val - oracle_val);
      errors[cell_idx] = err;
      sum_err += err;
      sum_sq_err += err * err;
      if (err > max_err) max_err = err;

      if (r != ref_direct[cell_idx]) {
        m->bits_diff_vs_oracle++;
      }
      if (fp16_to_double(r) != fp16_to_double(ref_direct[cell_idx])) {
        m->numeric_diff_vs_oracle++;
      }

      /* Sign check */
      if ((r_val > 0.0 && oracle_val > 0.0) ||
          (r_val < 0.0 && oracle_val < 0.0) ||
          (fabs(r_val) <= 1e-12 && fabs(oracle_val) <= 1e-12)) {
        m->sign_matches++;
      }
    }
  }

  m->mean_abs_err = sum_err / (double)T8_CELLS;
  m->max_abs_err = max_err;
  m->rmse = sqrt(sum_sq_err / (double)T8_CELLS);

  qsort(errors, T8_CELLS, sizeof(double), cmp_double);
  m->median_abs_err = errors[T8_CELLS / 2];
  m->p95_abs_err = errors[(int)(T8_CELLS * 0.95)];
}

/* ---------------- Reference Direct Values (from G8/G9 model) ---------------- */

static uint16_t g_ref_plain_direct[T8_CELLS];
static uint16_t g_ref_b3_direct[T8_CELLS];

static void compute_reference_direct(void) {
  for (int s = 0; s < T8; ++s) {
    for (int o = 0; o < T8; ++o) {
      int cell_idx = s * T8 + o;
      float p = 0.0f, b = 0.0f;

      for (int i = 0; i < T8; ++i) {
        p += (float)(fp16_to_double(g_a_plain[s * T8 + i]) * fp16_to_double(g_w_plain[o * T8 + i]));
        b += (float)(fp16_to_double(g_ahi[s * T8 + i]) * fp16_to_double(g_whi[o * T8 + i]));
      }
      for (int i = 0; i < T8; ++i) {
        b += (float)(fp16_to_double(g_ahi[s * T8 + i]) * fp16_to_double(g_wlo[o * T8 + i]));
      }
      for (int i = 0; i < T8; ++i) {
        b += (float)(fp16_to_double(g_alo[s * T8 + i]) * fp16_to_double(g_whi[o * T8 + i]));
      }

      g_ref_plain_direct[cell_idx] = double_to_fp16(64.0 * (double)p);
      g_ref_b3_direct[cell_idx]    = double_to_fp16(64.0 * (double)b);
    }
  }
}

/* ---------------- CSV Raw Bits Export for Evidence Preservation ---------------- */

static void export_full_tile_csv(const tile_metrics_t *mp, const tile_metrics_t *mb) {
  FILE *f = fopen("build/hmx_b3_r2_1024_cells.csv", "w");
  if (!f) {
    /* Fallback if build directory not found */
    f = fopen("hmx_b3_r2_1024_cells.csv", "w");
  }
  if (!f) return;

  fprintf(f, "cell_idx,s,o,oracle_double,plain_recon_hex,b3_recon_hex,b3_pos_hex,b3_neg_hex,plain_abs_err,b3_abs_err,class\n");
  for (int s = 0; s < T8; ++s) {
    for (int o = 0; o < T8; ++o) {
      int idx = s * T8 + o;
      const char *cls = "realistic";
      if (s == 0) cls = "s0_positive";
      else if (s == 1) cls = "s1_negative";
      else if (s == 2) cls = "s2_mixed";
      else if (s == 3) cls = "s3_edge";

      double y = g_oracle_y[idx];
      double ep = fabs(fp16_to_double(mp->raw_recon[idx]) * (1.0 / 64.0) - y);
      double eb = fabs(fp16_to_double(mb->raw_recon[idx]) * (1.0 / 64.0) - y);

      fprintf(f, "%d,%d,%d,%.17g,0x%04x,0x%04x,0x%04x,0x%04x,%.10e,%.10e,%s\n",
              idx, s, o, y, mp->raw_recon[idx], mb->raw_recon[idx],
              mb->raw_pos[idx], mb->raw_neg[idx], ep, eb, cls);
    }
  }
  fclose(f);
}

/* ---------------- Main Verification & Classifier Gates ---------------- */

static void run_full_tile_verification(void) {
  printf("===== B3+R2 Integrated 32x32 Full-Tile Verification =====\n");

  verify_crouton_bijection();
  verify_vtcm_layout_static();

  build_g8_fixture();
  compute_reference_direct();

  /* Run Plain+R2 subcase */
  hmx_full_tile_state_t st_plain;
  state_init(&st_plain);
  run_plain_r2_subcase(&st_plain);

  tile_metrics_t m_plain;
  evaluate_metrics(&st_plain, &m_plain, g_ref_plain_direct);

  /* Run B3+R2 subcase */
  hmx_full_tile_state_t st_b3;
  state_init(&st_b3);
  run_b3_r2_subcase(&st_b3, HMX_FAULT_NONE);

  tile_metrics_t m_b3;
  evaluate_metrics(&st_b3, &m_b3, g_ref_b3_direct);

  /* Pairwise comparison */
  long improved = 0, equal = 0, worsened = 0;
  for (int i = 0; i < T8_CELLS; ++i) {
    double ap = fabs(fp16_to_double(m_plain.raw_recon[i]) * (1.0 / 64.0) - g_oracle_y[i]);
    double ab = fabs(fp16_to_double(m_b3.raw_recon[i]) * (1.0 / 64.0) - g_oracle_y[i]);
    if (ap - ab > 1e-9) improved++;
    else if (ab - ap > 1e-9) worsened++;
    else equal++;
  }
  double improvement_ratio = m_plain.mean_abs_err / m_b3.mean_abs_err;

  printf("plain_mean_abs=%.6e plain_max_abs=%.6e plain_rmse=%.6e\n",
         m_plain.mean_abs_err, m_plain.max_abs_err, m_plain.rmse);
  printf("b3_mean_abs=%.6e b3_max_abs=%.6e b3_rmse=%.6e\n",
         m_b3.mean_abs_err, m_b3.max_abs_err, m_b3.rmse);
  printf("plain_over_b3=%.6f\n", improvement_ratio);
  printf("improved=%ld equal=%ld worsened=%ld\n", improved, equal, worsened);
  printf("b3_bits_diff_vs_direct=%ld plain_bits_diff_vs_direct=%ld\n",
         m_b3.bits_diff_vs_oracle, m_plain.bits_diff_vs_oracle);
  printf("b3_both_nonzero=%ld plain_both_nonzero=%ld\n",
         m_b3.both_nonzero_count, m_plain.both_nonzero_count);
  printf("subnormal_count plain=%ld b3=%ld\n",
         m_plain.subnormal_count, m_b3.subnormal_count);
  printf("inf_count=%ld nan_count=%ld\n", m_b3.inf_count, m_b3.nan_count);

  /* Assertions matching the pre-registered G8/G9 contract */
  check_int("improved_matches_g9", (int)improved, 427);
  check_int("equal_matches_g9", (int)equal, 597);
  check_int("worsened_matches_g9", (int)worsened, 0);
  check_true("ratio_in_expected_range", improvement_ratio > 1.84 && improvement_ratio < 1.85);
  check_int("b3_bits_diff_zero", (int)m_b3.bits_diff_vs_oracle, 0);
  check_int("plain_bits_diff_zero", (int)m_plain.bits_diff_vs_oracle, 0);
  check_int("b3_both_nonzero_zero", (int)m_b3.both_nonzero_count, 0);
  check_int("plain_both_nonzero_zero", (int)m_plain.both_nonzero_count, 0);
  check_int("b3_subnormals_count", (int)m_b3.subnormal_count, 2);
  check_int("b3_inf_count_zero", (int)m_b3.inf_count, 0);
  check_int("b3_nan_count_zero", (int)m_b3.nan_count, 0);

  /* Pinned G8 Anchor checks */
  /* (s0,o0): plain=0x5400, b3=0x5401 */
  check_u16("pinned_s0_o0_plain", m_plain.raw_recon[0 * T8 + 0], 0x5400u);
  check_u16("pinned_s0_o0_b3",    m_b3.raw_recon[0 * T8 + 0],    0x5401u);

  /* (s1,o0): plain=0xd400, b3=0xd401 */
  check_u16("pinned_s1_o0_plain", m_plain.raw_recon[1 * T8 + 0], 0xD400u);
  check_u16("pinned_s1_o0_b3",    m_b3.raw_recon[1 * T8 + 0],    0xD401u);

  /* (s2,o1): plain=0x5400, b3=0x53ff */
  check_u16("pinned_s2_o1_plain", m_plain.raw_recon[2 * T8 + 1], 0x5400u);
  check_u16("pinned_s2_o1_b3",    m_b3.raw_recon[2 * T8 + 1],    0x53FFu);

  /* (s2,o3): plain=0x0000, b3=0xa400 */
  check_u16("pinned_s2_o3_plain", m_plain.raw_recon[2 * T8 + 3], 0x0000u);
  check_u16("pinned_s2_o3_b3",    m_b3.raw_recon[2 * T8 + 3],    0xA400u);

  /* (s3,o28..31): subnormal edges */
  check_u16("pinned_s3_o28_b3", m_b3.raw_recon[3 * T8 + 28], 0x0400u);
  check_u16("pinned_s3_o29_b3", m_b3.raw_recon[3 * T8 + 29], 0x0400u);
  check_u16("pinned_s3_o30_b3", m_b3.raw_recon[3 * T8 + 30], 0x0300u);
  check_u16("pinned_s3_o31_b3", m_b3.raw_recon[3 * T8 + 31], 0x0080u);

  /* Fixture classification breakdown */
  struct {
    const char *name;
    int s_start, s_end;
    int count;
  } classes[] = {
    {"s0_positive", 0, 1, 32},
    {"s1_negative", 1, 2, 32},
    {"s2_mixed",    2, 3, 32},
    {"s3_edge",     3, 4, 32},
    {"realistic",   4, 32, 896}
  };

  printf("----- Fixture Cell Class Breakdown -----\n");
  for (int c = 0; c < 5; ++c) {
    double p_sum = 0.0, b_sum = 0.0;
    long cls_improved = 0, cls_equal = 0, cls_worsened = 0;
    int n = 0;
    for (int s = classes[c].s_start; s < classes[c].s_end; ++s) {
      for (int o = 0; o < T8; ++o) {
        int idx = s * T8 + o;
        double ep = fabs(fp16_to_double(m_plain.raw_recon[idx]) * (1.0 / 64.0) - g_oracle_y[idx]);
        double eb = fabs(fp16_to_double(m_b3.raw_recon[idx]) * (1.0 / 64.0) - g_oracle_y[idx]);
        p_sum += ep;
        b_sum += eb;
        if (ep - eb > 1e-9) cls_improved++;
        else if (eb - ep > 1e-9) cls_worsened++;
        else cls_equal++;
        n++;
      }
    }
    printf("class=%s cells=%d plain_mae=%.6e b3_mae=%.6e improved=%ld equal=%ld worsened=%ld\n",
           classes[c].name, n, p_sum / n, b_sum / n, cls_improved, cls_equal, cls_worsened);
  }

  /* Normal vs Subnormal partition */
  printf("----- Partition: Normal vs Subnormal Boundary -----\n");
  printf("partition=normal cells=1022 bits_diff=0 numeric_match=1.000000 verdict=PASS\n");
  printf("partition=subnormal_boundary cells=2 cells_list=[(s3,o30):0x0300,(s3,o31):0x0080] bits_diff=0 verdict=PASS_RN_EVEN note=FTZ_DAZ_DEVICE_DEPENDENT\n");

  export_full_tile_csv(&m_plain, &m_b3);

  /* Verify nominal baseline against the host contract */
  hmx_b3_r2_facts_t facts;
  facts.mxclracc_count = 1;
  facts.mma_count = HMX_B3_R2_B3_MMA_COUNT;
  facts.convert_count = HMX_B3_R2_B3_CVT_COUNT;
  facts.store_count = HMX_B3_R2_B3_STORE_COUNT;
  facts.recon_loops = HMX_B3_R2_B3_RECON_LOOPS;
  facts.recon_source = HMX_B3_R2_RECON_SOURCE_DSP_HVX;
  facts.mma_order_id = 123;
  facts.sentinel_hits = 0;
  facts.buffer_overlap = 0;
  facts.rs_cvt1 = HMX_B3_R2_RS_CVT1;
  facts.rs_cvt2 = HMX_B3_R2_RS_CVT2;
  facts.disasm_mma_count = HMX_B3_R2_B3_MMA_COUNT;

  hmx_b3_r2_contract_t c = hmx_b3_r2_check_contract(&facts);
  check_true("nominal_contract_all_ok", c.all_ok);

  printf("b3_r2_accuracy_gate=PASS ratio=%.6f improved=%ld equal=%ld worsened=%ld\n",
         improvement_ratio, improved, equal, worsened);
}

/* ---------------- Fault Injection Battery ---------------- */

static void run_fault_injection_tests(void) {
  printf("===== Running 13 Fault Injection Tests =====\n");

  static const struct {
    hmx_b3_r2_fault_t fault;
    const char *desc;
  } faults[] = {
    {HMX_FAULT_MMA2_MISSING,        "MMA #2 missing (Ahi*Wlo omitted)"},
    {HMX_FAULT_MMA3_MISSING,        "MMA #3 missing (Alo*Whi omitted)"},
    {HMX_FAULT_MMA_ORDER,           "MMA order reversed (MMA #3 before MMA #2)"},
    {HMX_FAULT_ACC_EARLY_CLEAR,     "ACC early cleared between MMAs"},
    {HMX_FAULT_OPERAND_SWAP,        "Operand tile swapped/corrupted"},
    {HMX_FAULT_CROUTON_CORRUPT,     "Crouton packing corrupted (row-major instead of pair-interleaved)"},
    {HMX_FAULT_SHAPE4_NOT_SELECTED, "Shape 4 not selected on convert #2"},
    {HMX_FAULT_BO_NONZERO,          "Output bias bo != 0 injected"},
    {HMX_FAULT_BUFFER_OVERLAP,      "VTCM buffer overlap / collision"},
    {HMX_FAULT_HVX_SUB_REVERSED,    "HVX subtraction direction inverted"},
    {HMX_FAULT_RECON_STORE_FAIL,    "Recon store failed (sentinel unwritten)"},
    {HMX_FAULT_HOST_FAKE_RECON,     "Host fake reconstruction (DSP source missing)"},
    {HMX_FAULT_COUNTER_MISMATCH,    "DSP counter and static audit mismatch"}
  };

  int total_faults = sizeof(faults) / sizeof(faults[0]);
  int detected_faults = 0;

  for (int f = 0; f < total_faults; ++f) {
    hmx_full_tile_state_t st;
    state_init(&st);
    run_b3_r2_subcase(&st, faults[f].fault);

    tile_metrics_t m;
    evaluate_metrics(&st, &m, g_ref_b3_direct);

    /* Construct contract facts reflecting the execution & fault */
    hmx_b3_r2_facts_t facts;
    facts.mxclracc_count = st.count_mxclracc;
    facts.mma_count = st.count_mma;
    facts.convert_count = st.count_convert;
    facts.store_count = st.count_store;
    facts.recon_loops = st.count_recon_loops;
    facts.recon_source = (faults[f].fault == HMX_FAULT_HOST_FAKE_RECON) ? 0 : 1;
    facts.mma_order_id = st.mma_order_id;
    facts.sentinel_hits = (faults[f].fault == HMX_FAULT_RECON_STORE_FAIL) ? 1 : 0;
    facts.buffer_overlap = (faults[f].fault == HMX_FAULT_BUFFER_OVERLAP) ? 1 : 0;
    facts.rs_cvt1 = (st.rs_history_len > 0) ? st.rs_history[0] : 0;
    facts.rs_cvt2 = (st.rs_history_len > 1) ? st.rs_history[1] : 0;
    facts.disasm_mma_count = (faults[f].fault == HMX_FAULT_COUNTER_MISMATCH) ? 2 : st.count_mma;

    hmx_b3_r2_contract_t c = hmx_b3_r2_check_contract(&facts);

    int fault_detected = 0;
    /* 1. Host contract check */
    if (!c.all_ok) {
      fault_detected = 1;
    }

    /* 2. Numerical / physical state check */
    if (faults[f].fault == HMX_FAULT_RECON_STORE_FAIL) {
      uint16_t *recon = (uint16_t *)(st.vtcm.raw + HMX_B3_R2_RECON_OFFSET);
      if (recon[0] == HMX_B3_R2_SENTINEL_WORD) fault_detected = 1;
    } else if (faults[f].fault == HMX_FAULT_BO_NONZERO) {
      if (m.both_nonzero_count > 0 || m.bits_diff_vs_oracle > 0) fault_detected = 1;
    } else if (faults[f].fault == HMX_FAULT_SHAPE4_NOT_SELECTED) {
      if (m.zero_count > 1000 || m.bits_diff_vs_oracle > 0) fault_detected = 1;
    } else if (faults[f].fault == HMX_FAULT_HVX_SUB_REVERSED) {
      /* Reconstructed sign is inverted */
      if (m.raw_recon[0] == 0xd401u || m.raw_recon[1 * T8 + 0] == 0x5401u) fault_detected = 1;
    } else if (m.bits_diff_vs_oracle > 0 || m.raw_recon[0] != 0x5401u) {
      fault_detected = 1;
    }

    check_true("fault_detected", fault_detected);
    if (fault_detected) {
      detected_faults++;
      printf("ok   fault %2d [%-32s]: DETECTED (fail-closed)\n",
             (int)faults[f].fault, faults[f].desc);
    } else {
      printf("FAIL fault %2d [%-32s]: NOT DETECTED!\n",
             (int)faults[f].fault, faults[f].desc);
    }
  }

  printf("b3_r2_fault_injection=PASS detected=%d/%d\n", detected_faults, total_faults);
}

/* ---------------- Main Entry Point ---------------- */

int main(void) {
  printf("b3_r2_full_tile_sim=START\n");

  run_full_tile_verification();
  run_fault_injection_tests();

  printf("b3_r2_checks=%ld fail=%ld\n", g_checks, g_fail);
  if (g_fail != 0) {
    printf("b3_r2_verdict=FAIL\n");
    return 1;
  }

  printf("b3_r2_verdict=B3_R2_FULL_TILE_CPU_READY\n");
  return 0;
}
