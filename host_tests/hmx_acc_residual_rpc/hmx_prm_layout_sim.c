// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only model of the Hexagon V81 HMX VTCM pair-interleaved physical
// layout, replaying the Phase E-3C1 mode-42 fixture. No SDK, no
// device, no adb: every number is reproducible on the host through the
// shared fp16_util.h oracle.
//
// Spec evidence — Qualcomm Hexagon V81 HMX PRM 80-N2040-62 Rev AA:
//   §2.2.1 activation VTCM organization (Figure 4)
//   §2.2.2 weights (Figure 8 + §3.1 write_wgt_vtcm)
//   §2.3 convert state (Figure 16)
//   §2.4 write convert state (Figures 18/19 + §3.1)
//
// Physical u16 indices (byte = 2*index):
//   act(s,i)   = (s>>1)*64 + i*2 + (s&1)
//   wgt(o,i)   = (i>>1)*64 + o*2 + (i&1)
//   store(s,o) = (s>>1)*64 + o*2 + (s&1)
//
// The bug this file models: mode 42 wrote its operand tiles ROW-MAJOR
// (logical A[s][i] at s*32+i, W[o][i] at o*32+i), but the HMX MMA
// reads the pair-interleaved physical layout above — even u16 lanes
// hold the even member of an adjacent pair (even spatial for
// activations, even input-channel for weights), odd lanes the odd
// partner, and the outer 64-u16 block pairs two adjacent rows. The
// host's row-major view of its own buffers therefore does NOT match
// what the DSP computed. Replaying the recorded mode-42 device data
// through the PRM layout reproduces all four measured slots exactly
// (out[0] pre 0xcf88 clamped to 0x0000; out[32] plain 0x59f6 /
// b3 0x59f7 at the unique (s0,o16) cell); the other three packing
// combinations do not.
//
// The negative-store clamp modeled in convert_store() — a convert
// pre with the sign bit set stores as 0x0000 — is the EMPIRICAL
// README law 4 (device output-bias feedback behavior), NOT a PRM
// statement; the PRM says nothing about it. It is modeled here only
// because the mode-42 measurement showed out[0] = 0x0000 for the
// negative pre 0xcf88.
//
// Register encoding used by mode 42: 5-bit spatial mask 0x1C
// (11100 = YYYXX, the 4x8 crouton) spread by the register field as
// ((m>>1)<<7) | ((m&1)<<1) = 0x700, hence act_rt = 0x77C,
// wgt_rt = 2048-1 = 0x7FF, wr_rt = 0x700 (asserted below).
//
// Build (scripts/run_prm_layout_sim.ps1 wires gcc + optional clang;
// both compile lines include -ffp-contract=off):
//   gcc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off
//       -o hmx_prm_layout_sim hmx_prm_layout_sim.c -lm
//   clang -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off
//       -rtlib=compiler-rt -o hmx_prm_layout_sim hmx_prm_layout_sim.c

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fp16_util.h"
#include "hmx_mode43.h"

#define T8 32
#define T8_CELLS (T8 * T8)

/* ---------------- bookkeeping ---------------- */

static long g_checks = 0;
static long g_fail = 0;

static void check_u16(const char *tag, uint16_t got, uint16_t want) {
  ++g_checks;
  if (got != want) {
    ++g_fail;
    printf("FAIL %s got=0x%04x want=0x%04x\n", tag, (unsigned)got,
           (unsigned)want);
  }
}

static void check_int(const char *tag, int got, int want) {
  ++g_checks;
  if (got != want) {
    ++g_fail;
    printf("FAIL %s got=%d want=%d\n", tag, got, want);
  }
}

/* ---------------- deterministic PRNG (mode-42 / t8_build seed) ------ */

static uint64_t g_rng;

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

/* Float-arithmetic split residual: statement order mirrors
 * probe_dsp.c e3c1_split_lo — hi = RN16(x) is computed by the caller
 * FIRST, then (float)x - (float)hi in FLOAT arithmetic as separate
 * statements (no FMA can contract), then lo = RN16(sub). */
static uint16_t split_lo(double x, uint16_t hi_bits) {
  float xf = (float)x;
  float hf = (float)fp16_to_double(hi_bits);
  float sub = xf - hf;
  return double_to_fp16((double)sub);
}

/* ---------------- PRM physical layout (80-N2040-62 Rev AA) ---------- */

/* Activation crouton, §2.2.1 Figure 4: pair-interleaved — even u16
 * lanes hold even spatial, odd lanes the odd partner; the outer
 * 64-u16 block pairs two adjacent 32-lane rows. */
static uint16_t prm_act_u16(int s, int i) {
  return (uint16_t)(((s >> 1) * 64) + i * 2 + (s & 1));
}

/* Weight crouton, §2.2.2 Figure 8 / §3.1 write_wgt_vtcm: pair-
 * interleaved over the INPUT channel — even u16 lanes hold even i,
 * odd lanes the odd partner. */
static uint16_t prm_wgt_u16(int o, int i) {
  return (uint16_t)(((i >> 1) * 64) + o * 2 + (i & 1));
}

/* Convert-state / store crouton, §2.3 Figure 16 + §2.4 Figures
 * 18/19 / §3.1: pair-interleaved over the SPATIAL axis (same shape as
 * the activation crouton). The device writes convert cell (s,o) at
 * this physical u16 index; the host readback out[k] is index k. */
static uint16_t prm_store_u16(int s, int o) {
  return (uint16_t)(((s >> 1) * 64) + o * 2 + (s & 1));
}

/* Decode a physical u16 index back to logical coordinates (inverse of
 * the formulas above: the pair bit is idx&1, the outer-64 block is
 * idx>>6, the fast axis is bits 1..5). */
static int prm_decode_pair(int idx) { return 2 * (idx >> 6) + (idx & 1); }
static int prm_decode_fast(int idx) { return (idx >> 1) & 31; }

/* Register-field spread of the 5-bit spatial mask (mode 42). */
static unsigned sp_mask_field(unsigned m) {
  return ((m >> 1) << 7) | ((m & 1u) << 1);
}

/* One bias-0 convert, scale 64 (mode-41/42 measured law): pre =
 * RN16(64*acc). The store step applies the EMPIRICAL README law-4
 * negative clamp (NOT PRM): a pre with the sign bit set stores as
 * 0x0000 (device output-bias feedback behavior). */
static uint16_t convert_store(uint16_t pre) {
  return (pre & 0x8000u) ? 0x0000u : pre;
}

/* ---------------- fixture (verbatim port of mode 42 / t8_build) ----- */

static double g_wd[T8_CELLS]; /* weight truth (wider precision) */
static double g_xd[T8_CELLS]; /* activation truth (wider precision) */
static uint16_t g_ahi[T8_CELLS], g_alo[T8_CELLS]; /* split activations */
static uint16_t g_whi[T8_CELLS], g_wlo[T8_CELLS]; /* split weights */

static void fixture_build(void) {
  int k, s, o, i;

  /* Seeded realistic data, statement order verbatim from probe_dsp.c
   * mode 42 / hmx_fp16x2_split_sim.c t8_build: W first, then X, per
   * cell. */
  g_rng = 0xE3C0117E5EEDF00Dull;
  for (k = 0; k < T8_CELLS; ++k) {
    g_wd[k] = 4.0 * rng_unit() - 2.0;
    g_xd[k] = 4.0 * rng_unit() - 2.0;
  }

  /* Special W rows: out-ch 0..3 case-class anchors, 28..31 s3
   * subnormal-edge targets; out-ch 4..27 stay seeded realistic. */
  for (o = 0; o < T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < T8; ++i) g_wd[(long)o * T8 + i] = 0.0;
    }
  }
  g_wd[0 * T8 + 0] = 1.0 + 0x1p-11; /* positive split-sensitive row */
  g_wd[0 * T8 + 1] = 0x1p-11;
  g_wd[0 * T8 + 2] = 0x1p-13;
  g_wd[1 * T8 + 0] = 1.0 - 0x1p-12; /* B-2 case-1 anchor row */
  g_wd[2 * T8 + 0] = 0x1p-9;        /* s3 boundary row (64*acc=2^-14) */
  g_wd[3 * T8 + 0] = 1.0;           /* near-zero cancellation row */
  g_wd[3 * T8 + 1] = 1.0 + 0x1p-12;
  g_wd[28 * T8 + 0] = 0x1p-10;
  g_wd[29 * T8 + 0] = 0x1p-10 - 0x1p-21;
  g_wd[30 * T8 + 0] = 3.0 * 0x1p-12;
  g_wd[31 * T8 + 0] = 0x1p-13;

  /* Special A lanes 0..3; lanes 4..31 stay seeded realistic. */
  for (s = 0; s <= 3; ++s)
    for (i = 0; i < T8; ++i) g_xd[(long)s * T8 + i] = 0.0;
  g_xd[0 * T8 + 0] = 1.0 + 0x1p-11; /* s0: positive multi-channel sum */
  g_xd[0 * T8 + 1] = 0x1p-11;
  g_xd[0 * T8 + 2] = 0x1p-13;
  for (i = 0; i < T8; ++i) /* s1 = negation of s0 */
    g_xd[1 * T8 + i] = -g_xd[i];
  g_xd[2 * T8 + 0] = 1.0 - 0x1p-12; /* s2: B-2 case-1 + negated twin */
  g_xd[2 * T8 + 1] = -(1.0 - 0x1p-12);
  g_xd[3 * T8 + 0] = 0x1p-10; /* s3: subnormal-edge lane */

  /* Four fp16 operand tiles per cell: hi = RN16(value) is the plain
   * operand; lo = the float-arithmetic residual (probe_dsp.c
   * e3c1_split_lo statement order). */
  for (k = 0; k < T8_CELLS; ++k) {
    uint16_t ah = double_to_fp16(g_xd[k]);
    uint16_t wh = double_to_fp16(g_wd[k]);
    g_ahi[k] = ah;
    g_alo[k] = split_lo(g_xd[k], ah);
    g_whi[k] = wh;
    g_wlo[k] = split_lo(g_wd[k], wh);
  }
}

/* ---------------- two VTCM write modes per tile ---------------- */

enum { PACK_ROW = 0, PACK_PRM = 1 };

/* Row-major write: memcpy of the logical array (mode 42's bug). */
static void write_row(uint16_t *dst, const uint16_t *logical) {
  memcpy(dst, logical, sizeof(uint16_t) * T8_CELLS);
}

/* PRM-packed activation write: buffer[act(s,i)] = logical A[s][i]. */
static void write_act_prm(uint16_t *dst, const uint16_t *logical) {
  int s, i;
  for (s = 0; s < T8; ++s)
    for (i = 0; i < T8; ++i) dst[prm_act_u16(s, i)] = logical[s * T8 + i];
}

/* PRM-packed weight write: buffer[wgt(o,i)] = logical W[o][i]. */
static void write_wgt_prm(uint16_t *dst, const uint16_t *logical) {
  int o, i;
  for (o = 0; o < T8; ++o)
    for (i = 0; i < T8; ++i) dst[prm_wgt_u16(o, i)] = logical[o * T8 + i];
}

static uint16_t g_ahi_row[T8_CELLS], g_ahi_prm[T8_CELLS];
static uint16_t g_alo_row[T8_CELLS], g_alo_prm[T8_CELLS];
static uint16_t g_whi_row[T8_CELLS], g_whi_prm[T8_CELLS];
static uint16_t g_wlo_row[T8_CELLS], g_wlo_prm[T8_CELLS];

static void write_all_tiles(void) {
  write_row(g_ahi_row, g_ahi);
  write_act_prm(g_ahi_prm, g_ahi);
  write_row(g_alo_row, g_alo);
  write_act_prm(g_alo_prm, g_alo);
  write_row(g_whi_row, g_whi);
  write_wgt_prm(g_whi_prm, g_whi);
  write_row(g_wlo_row, g_wlo);
  write_wgt_prm(g_wlo_prm, g_wlo);
}

/* ---------------- device compute reading through the PRM layout ---- */

static uint16_t g_plain_pre[T8_CELLS], g_plain_st[T8_CELLS];
static uint16_t g_b3_pre[T8_CELLS], g_b3_st[T8_CELLS];
static uint16_t g_pout_pre[T8_CELLS], g_pout_st[T8_CELLS]; /* plain phys */
static uint16_t g_bout_pre[T8_CELLS], g_bout_st[T8_CELLS]; /* b3 phys */

/* Plain: ONE MMA packet {act_hi, wgt_hi} — a single chained FLOAT
 * accumulator over 32 products, mirroring the G8 float chain, then
 * one convert (scale 64). The device reads operand (s,i)/(o,i)
 * through the PRM physical indices, not logical row-major. */
static void compute_plain(const uint16_t *a, const uint16_t *w) {
  int s, o, i;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      float acc = 0.0f;
      for (i = 0; i < T8; ++i) {
        acc += (float)(fp16_to_double(a[prm_act_u16(s, i)]) *
                       fp16_to_double(w[prm_wgt_u16(o, i)]));
      }
      g_plain_pre[s * T8 + o] = double_to_fp16(64.0 * (double)acc);
      g_plain_st[s * T8 + o] = convert_store(g_plain_pre[s * T8 + o]);
    }
  }
}

/* B3: THREE full-tile MMA packets into the SAME retained float ACC
 * with no convert/store/clear between them (mode-42 B-1/B-2 law),
 * pass order Ahi*Whi, then Ahi*Wlo, then Alo*Whi; one final
 * bias-0 convert (scale 64). */
static void compute_b3(const uint16_t *ahi, const uint16_t *alo,
                       const uint16_t *whi, const uint16_t *wlo) {
  int s, o, i;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      float acc = 0.0f;
      for (i = 0; i < T8; ++i) { /* pass 1: Ahi*Whi */
        acc += (float)(fp16_to_double(ahi[prm_act_u16(s, i)]) *
                       fp16_to_double(whi[prm_wgt_u16(o, i)]));
      }
      for (i = 0; i < T8; ++i) { /* pass 2: Ahi*Wlo */
        acc += (float)(fp16_to_double(ahi[prm_act_u16(s, i)]) *
                       fp16_to_double(wlo[prm_wgt_u16(o, i)]));
      }
      for (i = 0; i < T8; ++i) { /* pass 3: Alo*Whi (one chain) */
        acc += (float)(fp16_to_double(alo[prm_act_u16(s, i)]) *
                       fp16_to_double(whi[prm_wgt_u16(o, i)]));
      }
      g_b3_pre[s * T8 + o] = double_to_fp16(64.0 * (double)acc);
      g_b3_st[s * T8 + o] = convert_store(g_b3_pre[s * T8 + o]);
    }
  }
}

/* The device writes convert cell (s,o) through the §2.3/§2.4 store
 * crouton at physical u16 index store(s,o); the host readback out[k]
 * is physical index k. */
static void place_out(const uint16_t *pre_cell, const uint16_t *st_cell,
                      uint16_t *pre_phys, uint16_t *st_phys) {
  int s, o;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      const uint16_t k = prm_store_u16(s, o);
      pre_phys[k] = pre_cell[s * T8 + o];
      st_phys[k] = st_cell[s * T8 + o];
    }
  }
}

/* ---------------- mapping tables + register asserts ---------------- */

static void check_and_print_maps(void) {
  static const int act_idxs[8] = {0, 1, 2, 3, 32, 33, 64, 65};
  static const int wgt_idxs[7] = {0, 1, 2, 3, 32, 33, 64};
  static const int st_idxs[7] = {0, 1, 2, 3, 32, 33, 64};
  static const struct {
    int s, i, idx;
  } act_exp[8] = {{0, 0, 0},  {1, 0, 1},  {0, 1, 2},  {1, 1, 3},
                  {0, 16, 32}, {1, 16, 33}, {2, 0, 64}, {3, 0, 65}};
  static const struct {
    int o, i, idx;
  } wgt_exp[7] = {
      {0, 0, 0}, {0, 1, 1}, {1, 0, 2}, {1, 1, 3}, {16, 0, 32}, {16, 1, 33}, {0, 2, 64}};
  static const struct {
    int s, o, idx;
  } st_exp[7] = {
      {0, 0, 0}, {1, 0, 1}, {0, 1, 2}, {1, 1, 3}, {0, 16, 32}, {1, 16, 33}, {2, 0, 64}};
  int k;

  /* Forward-index asserts: the PRM formulas must land the known
   * logical cells on the pinned physical indices. */
  for (k = 0; k < 8; ++k) {
    char tag[64];
    sprintf(tag, "prm_act_u16 s=%d i=%d", act_exp[k].s, act_exp[k].i);
    check_u16(tag, prm_act_u16(act_exp[k].s, act_exp[k].i),
              (uint16_t)act_exp[k].idx);
  }
  for (k = 0; k < 7; ++k) {
    char tag[64];
    sprintf(tag, "prm_wgt_u16 o=%d i=%d", wgt_exp[k].o, wgt_exp[k].i);
    check_u16(tag, prm_wgt_u16(wgt_exp[k].o, wgt_exp[k].i),
              (uint16_t)wgt_exp[k].idx);
  }
  for (k = 0; k < 7; ++k) {
    char tag[64];
    sprintf(tag, "prm_store_u16 s=%d o=%d", st_exp[k].s, st_exp[k].o);
    check_u16(tag, prm_store_u16(st_exp[k].s, st_exp[k].o),
              (uint16_t)st_exp[k].idx);
  }

  /* Decoded tables (stable output). */
  for (k = 0; k < 8; ++k) {
    const int idx = act_idxs[k];
    printf("prm_layout_act_map idx=%d byte=%d spatial=%d in_ch=%d\n", idx,
           2 * idx, prm_decode_pair(idx), prm_decode_fast(idx));
  }
  for (k = 0; k < 7; ++k) {
    const int idx = wgt_idxs[k];
    printf("prm_layout_wgt_map idx=%d byte=%d out_ch=%d in_ch=%d\n", idx,
           2 * idx, prm_decode_fast(idx), prm_decode_pair(idx));
  }
  for (k = 0; k < 7; ++k) {
    const int idx = st_idxs[k];
    printf("prm_layout_store_map idx=%d byte=%d spatial=%d out_ch=%d\n",
           idx, 2 * idx, prm_decode_pair(idx), prm_decode_fast(idx));
  }
}

static void check_and_print_regs(void) {
  const unsigned sm = 0x1Cu;
  const unsigned smf = sp_mask_field(sm);
  const unsigned act_rt = smf | (31u << 2);
  const unsigned wgt_rt = 2048u - 1u;
  const unsigned wr_rt = smf;

  ++g_checks;
  if (act_rt != 0x77Cu || wgt_rt != 0x7FFu || wr_rt != 0x700u) {
    ++g_fail;
    printf("FAIL prm_regs act_rt=0x%x wgt_rt=0x%x wr_rt=0x%x "
           "want act_rt=0x77c wgt_rt=0x7ff wr_rt=0x700\n",
           act_rt, wgt_rt, wr_rt);
  }
  printf("prm_layout_regs act_rt=0x%03x wgt_rt=0x%03x wr_rt=0x%03x "
         "sm=0x%02x crouton=4x8\n",
         act_rt, wgt_rt, wr_rt, sm);
}

/* ---------------- as-written uniqueness scan ---------------- */

static void print_uniqueness(void) {
  int s, o, n_plain = 0, n_b3 = 0, n_pair = 0, us = -1, uo = -1;
  int store_idx;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      const int hitp = g_plain_st[s * T8 + o] == 0x59f6u;
      const int hitb = g_b3_st[s * T8 + o] == 0x59f7u;
      if (hitp) ++n_plain;
      if (hitb) ++n_b3;
      if (hitp && hitb) {
        ++n_pair;
        us = s;
        uo = o;
      }
    }
  }
  check_int("e3c1_uniqueness_pair_count", n_pair, 1);
  check_int("e3c1_uniqueness_plain_count", n_plain, 1);
  check_int("e3c1_uniqueness_b3_count", n_b3, 1);
  check_int("e3c1_uniqueness_cell_s", us, 0);
  check_int("e3c1_uniqueness_cell_o", uo, 16);
  store_idx = (us >= 0) ? (int)prm_store_u16(us, uo) : -1;
  printf("prm_e3c1_uniqueness plain=%d b3=%d cell_s=%d cell_o=%d "
         "store_idx=%d\n",
         n_plain, n_b3, us, uo, store_idx);
}

/* ---------------- four packing combinations ---------------- */

typedef struct {
  const char *act_label;
  const char *wgt_label;
  int act_prm;
  int wgt_prm;
  int has_12;
  uint16_t p0, p1, p2, p32; /* plain pre at physical 0/1/2/32 */
  uint16_t b0, b1, b2, b3;  /* b3 pre at physical 0/1/2/32 */
  uint16_t s0p, s0b;        /* plain/b3 STORE at physical 0 */
  int expect_match;
} combo_t;

static void combo_check(const combo_t *c, const char *variant,
                        const char *field, uint16_t got, uint16_t want) {
  ++g_checks;
  if (got != want) {
    ++g_fail;
    printf("FAIL e3c1_combo act=%s wgt=%s variant=%s field=%s "
           "got=0x%04x want=0x%04x\n",
           c->act_label, c->wgt_label, variant, field, (unsigned)got,
           (unsigned)want);
  }
}

static void run_combos(void) {
  /* Expected literals were derived independently by replaying the
   * recorded mode-42 device data through the PRM layout; a mismatch
   * is an implementation slip, NOT a reason to retune. */
  static const combo_t combos[] = {
      {"row", "row", 0, 0, 1, 0xcf88, 0xa388, 0xd5ef, 0x59f6,
       0xcf88, 0xa386, 0xd5ef, 0x59f7, 0x0000, 0x0000, 1},
      {"row", "prm", 0, 1, 0, 0x5400, 0, 0, 0x54e8,
       0x5401, 0, 0, 0x54e8, 0x5400, 0x5401, 0},
      {"prm", "row", 1, 0, 0, 0x5400, 0, 0, 0x5400,
       0x5401, 0, 0, 0x5400, 0x5400, 0x5401, 0},
      {"prm", "prm", 1, 1, 1, 0x5400, 0xd400, 0x5400, 0x55e3,
       0x5401, 0xd401, 0x5400, 0x55e3, 0x5400, 0x5401, 0},
  };
  size_t c;
  for (c = 0; c < sizeof combos / sizeof combos[0]; ++c) {
    const combo_t *cb = &combos[c];
    const uint16_t *ahi = cb->act_prm ? g_ahi_prm : g_ahi_row;
    const uint16_t *alo = cb->act_prm ? g_alo_prm : g_alo_row;
    const uint16_t *whi = cb->wgt_prm ? g_whi_prm : g_whi_row;
    const uint16_t *wlo = cb->wgt_prm ? g_wlo_prm : g_wlo_row;
    int match;

    compute_plain(ahi, whi);
    compute_b3(ahi, alo, whi, wlo);
    place_out(g_plain_pre, g_plain_st, g_pout_pre, g_pout_st);
    place_out(g_b3_pre, g_b3_st, g_bout_pre, g_bout_st);

    combo_check(cb, "plain", "out0_pre", g_pout_pre[0], cb->p0);
    combo_check(cb, "b3", "out0_pre", g_bout_pre[0], cb->b0);
    combo_check(cb, "plain", "out0_store", g_pout_st[0], cb->s0p);
    combo_check(cb, "b3", "out0_store", g_bout_st[0], cb->s0b);
    combo_check(cb, "plain", "out32_pre", g_pout_pre[32], cb->p32);
    combo_check(cb, "b3", "out32_pre", g_bout_pre[32], cb->b3);
    combo_check(cb, "plain", "out32_store", g_pout_st[32], cb->p32);
    combo_check(cb, "b3", "out32_store", g_bout_st[32], cb->b3);
    if (cb->has_12) {
      combo_check(cb, "plain", "out1_pre", g_pout_pre[1], cb->p1);
      combo_check(cb, "b3", "out1_pre", g_bout_pre[1], cb->b1);
      combo_check(cb, "plain", "out2_pre", g_pout_pre[2], cb->p2);
      combo_check(cb, "b3", "out2_pre", g_bout_pre[2], cb->b2);
    }

    /* The measured signature: ONLY row/row yields out0 store 0x0000
     * AND out32 plain 0x59f6 / b3 0x59f7. */
    match = (g_pout_st[0] == 0x0000u && g_bout_st[0] == 0x0000u &&
             g_pout_st[32] == 0x59f6u && g_bout_st[32] == 0x59f7u)
                ? 1
                : 0;
    printf("prm_e3c1_combo act=%s wgt=%s out0_store=0x%04x "
           "out32_plain=0x%04x out32_b3=0x%04x match=%d\n",
           cb->act_label, cb->wgt_label, (unsigned)g_pout_st[0],
           (unsigned)g_pout_st[32], (unsigned)g_bout_st[32], match);
    ++g_checks;
    if (match != cb->expect_match) {
      ++g_fail;
      printf("FAIL e3c1_combo act=%s wgt=%s match got=%d want=%d\n",
             cb->act_label, cb->wgt_label, match, cb->expect_match);
    }

    if (cb->act_prm && cb->wgt_prm) {
      printf("prm_e3c1_packed plain out0=0x%04x out1=0x%04x "
             "out2=0x%04x out32=0x%04x\n",
             (unsigned)g_pout_pre[0], (unsigned)g_pout_pre[1],
             (unsigned)g_pout_pre[2], (unsigned)g_pout_pre[32]);
      printf("prm_e3c1_packed b3 out0=0x%04x out1=0x%04x "
             "out2=0x%04x out32=0x%04x\n",
             (unsigned)g_bout_pre[0], (unsigned)g_bout_pre[1],
             (unsigned)g_bout_pre[2], (unsigned)g_bout_pre[32]);
    }
  }
}

/* ---------------- all-index packing contract ---------------- */

/* Every one of the 1024 logical cells must map to a distinct u16 index in
 * [0, 1023], and the documented inverse must recover the logical pair.
 * This covers act, wgt and store with the same formulas the DSP uses. */
static void check_pack_bijection(void) {
  int seen[T8_CELLS];
  int s, o, i, idx, count;
  char tag[128];

  memset(seen, 0, sizeof seen);
  count = 0;
  for (s = 0; s < T8; ++s) {
    for (i = 0; i < T8; ++i) {
      idx = prm_act_u16(s, i);
      ++g_checks;
      if (idx >= T8_CELLS || seen[idx]) {
        ++g_fail;
        printf("FAIL prm_act_bijection s=%d i=%d idx=%d seen=%d\n", s, i,
               idx, seen[idx]);
      } else {
        seen[idx] = 1;
        ++count;
      }
      snprintf(tag, sizeof tag, "prm_act_decode_pair s=%d i=%d", s, i);
      check_int(tag, prm_decode_pair(idx), s);
      snprintf(tag, sizeof tag, "prm_act_decode_fast s=%d i=%d", s, i);
      check_int(tag, prm_decode_fast(idx), i);
    }
  }
  check_int("prm_act_bijection_count", count, T8_CELLS);

  memset(seen, 0, sizeof seen);
  count = 0;
  for (o = 0; o < T8; ++o) {
    for (i = 0; i < T8; ++i) {
      idx = prm_wgt_u16(o, i);
      ++g_checks;
      if (idx >= T8_CELLS || seen[idx]) {
        ++g_fail;
        printf("FAIL prm_wgt_bijection o=%d i=%d idx=%d seen=%d\n", o, i,
               idx, seen[idx]);
      } else {
        seen[idx] = 1;
        ++count;
      }
      snprintf(tag, sizeof tag, "prm_wgt_decode_i o=%d i=%d", o, i);
      check_int(tag, prm_decode_pair(idx), i);
      snprintf(tag, sizeof tag, "prm_wgt_decode_o o=%d i=%d", o, i);
      check_int(tag, prm_decode_fast(idx), o);
    }
  }
  check_int("prm_wgt_bijection_count", count, T8_CELLS);

  memset(seen, 0, sizeof seen);
  count = 0;
  for (s = 0; s < T8; ++s) {
    for (o = 0; o < T8; ++o) {
      idx = prm_store_u16(s, o);
      ++g_checks;
      if (idx >= T8_CELLS || seen[idx]) {
        ++g_fail;
        printf("FAIL prm_store_bijection s=%d o=%d idx=%d seen=%d\n", s, o,
               idx, seen[idx]);
      } else {
        seen[idx] = 1;
        ++count;
      }
      snprintf(tag, sizeof tag, "prm_store_decode_s s=%d o=%d", s, o);
      check_int(tag, prm_decode_pair(idx), s);
      snprintf(tag, sizeof tag, "prm_store_decode_o s=%d o=%d", s, o);
      check_int(tag, prm_decode_fast(idx), o);
    }
  }
  check_int("prm_store_bijection_count", count, T8_CELLS);
  printf("prm_pack_bijection act=1024 wgt=1024 store=1024 "
         "range=[0,1023] unique=1\n");
}

/* VTCM allocation used by the probe: six 2 KiB tiles = 12 KiB. Mode 43
 * uses act(0), wgt(1), bias0(4), bias1(4*TILE+256), out(5); mode 42 also
 * uses hi(2)/lo(3). Every 2 KiB tile is disjoint and inside the reserve;
 * the two 256-byte bias sets sit between the four operand tiles and the
 * out tile. */
static void check_vtcm_layout(void) {
  enum { TILE_BYTES = 2048, BIAS_BYTES = 256, VTCM_BYTES = 12288 };
  const int act_off = 0;
  const int wgt_off = 1 * TILE_BYTES;
  const int hi_off = 2 * TILE_BYTES;
  const int lo_off = 3 * TILE_BYTES;
  const int bias0_off = 4 * TILE_BYTES;
  const int bias1_off = bias0_off + BIAS_BYTES;
  const int out_off = 5 * TILE_BYTES;

  check_int("vtcm_act_within", act_off + TILE_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_wgt_within", wgt_off + TILE_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_hi_within", hi_off + TILE_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_lo_within", lo_off + TILE_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_bias1_within", bias1_off + BIAS_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_out_within", out_off + TILE_BYTES <= VTCM_BYTES, 1);
  check_int("vtcm_operand_tiles_disjoint",
            act_off + TILE_BYTES <= wgt_off &&
                wgt_off + TILE_BYTES <= hi_off &&
                hi_off + TILE_BYTES <= lo_off &&
                lo_off + TILE_BYTES <= bias0_off,
            1);
  check_int("vtcm_bias_sets_disjoint",
            bias0_off + BIAS_BYTES <= bias1_off &&
                bias1_off + BIAS_BYTES <= out_off,
            1);
  check_int("vtcm_mode43_operand_bias_disjoint",
            wgt_off + TILE_BYTES <= bias0_off &&
                bias1_off + BIAS_BYTES <= out_off,
            1);
  check_int("vtcm_out_within_reserve", out_off + TILE_BYTES <= VTCM_BYTES, 1);
  printf("prm_vtcm_layout total=%d act=%d wgt=%d hi=%d lo=%d bias0=%d "
         "bias1=%d out=%d mode43_used=%d\n",
         VTCM_BYTES, act_off, wgt_off, hi_off, lo_off, bias0_off, bias1_off,
         out_off, 3 * TILE_BYTES + 2 * BIAS_BYTES);
}

/* The mode-42 as-written replay must stay deterministic and unchanged.
 * Recompute it twice, compare all 1024 physical pre/store cells, and pin
 * the four recorded device slots. */
static uint16_t g_mode42_plain_pre_snap[T8_CELLS];
static uint16_t g_mode42_plain_st_snap[T8_CELLS];
static uint16_t g_mode42_b3_pre_snap[T8_CELLS];
static uint16_t g_mode42_b3_st_snap[T8_CELLS];

static void check_mode42_replay_deterministic(void) {
  int run;
  for (run = 0; run < 2; ++run) {
    compute_plain(g_ahi_row, g_whi_row);
    compute_b3(g_ahi_row, g_alo_row, g_whi_row, g_wlo_row);
    place_out(g_plain_pre, g_plain_st, g_pout_pre, g_pout_st);
    place_out(g_b3_pre, g_b3_st, g_bout_pre, g_bout_st);
    if (run == 0) {
      memcpy(g_mode42_plain_pre_snap, g_pout_pre, sizeof g_pout_pre);
      memcpy(g_mode42_plain_st_snap, g_pout_st, sizeof g_pout_st);
      memcpy(g_mode42_b3_pre_snap, g_bout_pre, sizeof g_bout_pre);
      memcpy(g_mode42_b3_st_snap, g_bout_st, sizeof g_bout_st);
    }
  }
  check_int("mode42_plain_pre_deterministic",
            memcmp(g_mode42_plain_pre_snap, g_pout_pre, sizeof g_pout_pre) == 0,
            1);
  check_int("mode42_plain_store_deterministic",
            memcmp(g_mode42_plain_st_snap, g_pout_st, sizeof g_pout_st) == 0,
            1);
  check_int("mode42_b3_pre_deterministic",
            memcmp(g_mode42_b3_pre_snap, g_bout_pre, sizeof g_bout_pre) == 0,
            1);
  check_int("mode42_b3_store_deterministic",
            memcmp(g_mode42_b3_st_snap, g_bout_st, sizeof g_bout_st) == 0, 1);
  check_u16("mode42_recorded_out0_store", g_pout_st[0], 0x0000);
  check_u16("mode42_recorded_out32_plain", g_pout_st[32], 0x59f6);
  check_u16("mode42_recorded_out32_b3", g_bout_st[32], 0x59f7);
  printf("prm_e3c1_mode42_replay deterministic=1 out0_store=0x%04x "
         "out32_plain=0x%04x out32_b3=0x%04x\n",
         (unsigned)g_pout_st[0], (unsigned)g_pout_st[32],
         (unsigned)g_bout_st[32]);
}

/* Mode-43 pre-registered predictions. The store column uses the empirical
 * negative clamp (0x0000) only as the model alternative; the signed
 * pre-store value 0xd400 is the other pre-registered candidate. */
static void check_mode43_predictions(void) {
  compute_plain(g_ahi_prm, g_whi_prm);
  place_out(g_plain_pre, g_plain_st, g_pout_pre, g_pout_st);
  check_u16("mode43_out0_pre", g_pout_pre[0], 0x5400);
  check_u16("mode43_out1_pre_signed", g_pout_pre[1], 0xd400);
  check_u16("mode43_out2_pre", g_pout_pre[2], 0x5400);
  check_u16("mode43_out32_pre", g_pout_pre[32], 0x55e3);
  check_u16("mode43_out0_store", g_pout_st[0], 0x5400);
  check_u16("mode43_out1_store_clamp_model", g_pout_st[1], 0x0000);
  check_u16("mode43_out2_store", g_pout_st[2], 0x5400);
  check_u16("mode43_out32_store", g_pout_st[32], 0x55e3);
  printf("prm_e3c1r_mode43_packed pre out0=0x%04x out1=0x%04x out2=0x%04x "
         "out32=0x%04x store out0=0x%04x out1=0x%04x out2=0x%04x "
         "out32=0x%04x signed_candidate=0xd400 clamp_model=0x0000\n",
         (unsigned)g_pout_pre[0], (unsigned)g_pout_pre[1],
         (unsigned)g_pout_pre[2], (unsigned)g_pout_pre[32],
         (unsigned)g_pout_st[0], (unsigned)g_pout_st[1],
         (unsigned)g_pout_st[2], (unsigned)g_pout_st[32]);
}

/* CPU test of the EXACT host-side mode-43 decision procedure in
 * hmx_mode43.h: all three positive checks gate slot-1 interpretation,
 * signed/clamp verdicts are fixed, and the host contract requires exactly
 * one executed trial. */
static void check_mode43_host_contract(void) {
  static const int signed_bits[4] = {0x5400, 0xd400, 0x5400, 0x55e3};
  static const int clamp_bits[4] = {0x5400, 0x0000, 0x5400, 0x55e3};
  static const int b3_slot1[4] = {0x5400, 0xd401, 0x5400, 0x55e3};
  static const int bad_out0[4] = {0x5401, 0xd400, 0x5400, 0x55e3};
  static const int bad_out2[4] = {0x5400, 0xd400, 0x5401, 0x55e3};
  static const int bad_out32[4] = {0x5400, 0xd400, 0x5400, 0x55e2};
  hmx_mode43_report_t r;
  const long fail_before = g_fail;

  r = hmx_mode43_classify(signed_bits, HMX_MODE43_TRIALS, 1);
  check_int("mode43.signed.verdict",
            r.verdict == HMX_MODE43_LAYOUT_CONFIRMED_SIGNED, 1);
  check_int("mode43.signed.positive_checks", r.positive_checks, 3);
  check_int("mode43.signed.positive_ok", r.positive_ok, 1);
  check_int("mode43.signed.slot1_signed", r.slot1_signed, 1);
  check_int("mode43.signed.trials_ok", r.trials_ok, 1);
  check_int("mode43.signed.executed_ok", r.hmx_executed_ok, 1);

  r = hmx_mode43_classify(clamp_bits, HMX_MODE43_TRIALS, 1);
  check_int("mode43.clamp.verdict",
            r.verdict == HMX_MODE43_LAYOUT_CONFIRMED_CLAMP, 1);
  check_int("mode43.clamp.slot1_clamp", r.slot1_clamp, 1);

  r = hmx_mode43_classify(b3_slot1, HMX_MODE43_TRIALS, 1);
  check_int("mode43.b3_slot1.verdict",
            r.verdict == HMX_MODE43_RESIDUAL_MISMATCH, 1);
  check_int("mode43.b3_slot1.positive_ok", r.positive_ok, 1);
  check_int("mode43.b3_slot1.slot1_interpreted",
            r.slot1_signed || r.slot1_clamp, 0);

  /* Positive checks gate slot-1 interpretation. */
  r = hmx_mode43_classify(bad_out0, HMX_MODE43_TRIALS, 1);
  check_int("mode43.bad_out0.positive_checks", r.positive_checks, 2);
  check_int("mode43.bad_out0.verdict",
            r.verdict == HMX_MODE43_RESIDUAL_MISMATCH, 1);
  r = hmx_mode43_classify(bad_out2, HMX_MODE43_TRIALS, 1);
  check_int("mode43.bad_out2.positive_checks", r.positive_checks, 2);
  r = hmx_mode43_classify(bad_out32, HMX_MODE43_TRIALS, 1);
  check_int("mode43.bad_out32.positive_checks", r.positive_checks, 2);

  /* One-trial host contract: 0 or 2 is a fail-closed violation. */
  r = hmx_mode43_classify(signed_bits, 2, 1);
  check_int("mode43.trials2.trials_ok", r.trials_ok, 0);
  check_int("mode43.trials2.observed", r.observed_trials, 2);
  r = hmx_mode43_classify(signed_bits, 0, 1);
  check_int("mode43.trials0.trials_ok", r.trials_ok, 0);
  r = hmx_mode43_classify(signed_bits, HMX_MODE43_TRIALS, 0);
  check_int("mode43.not_executed.executed_ok", r.hmx_executed_ok, 0);

  printf("prm_e3c1r_host_contract=%s signed=1 clamp=1 residual=1 "
         "positive_gate=1 trials_required=%d\n",
         g_fail == fail_before ? "PASS" : "FAIL", HMX_MODE43_TRIALS);
}

/* ---------------- main ---------------- */

int main(void) {
  printf("prm_layout_sim=start\n");
  printf("prm_layout_sim_source=80-N2040-62_Rev_AA\n");

  fixture_build();
  write_all_tiles();
  check_and_print_maps();
  check_and_print_regs();
  check_pack_bijection();
  check_vtcm_layout();

  /* As-written replay: mode 42's actual VTCM contents (row-major
   * tiles) read through the PRM layout, then stored through the
   * convert-state crouton. */
  compute_plain(g_ahi_row, g_whi_row);
  compute_b3(g_ahi_row, g_alo_row, g_whi_row, g_wlo_row);
  place_out(g_plain_pre, g_plain_st, g_pout_pre, g_pout_st);
  place_out(g_b3_pre, g_b3_st, g_bout_pre, g_bout_st);

  printf("prm_e3c1_aswritten plain out0_pre=0x%04x out0_store=0x%04x "
         "out1=0x%04x out2=0x%04x out32=0x%04x\n",
         (unsigned)g_pout_pre[0], (unsigned)g_pout_st[0],
         (unsigned)g_pout_pre[1], (unsigned)g_pout_pre[2],
         (unsigned)g_pout_pre[32]);
  printf("prm_e3c1_aswritten b3 out0_pre=0x%04x out0_store=0x%04x "
         "out1=0x%04x out2=0x%04x out32=0x%04x\n",
         (unsigned)g_bout_pre[0], (unsigned)g_bout_st[0],
         (unsigned)g_bout_pre[1], (unsigned)g_bout_pre[2],
         (unsigned)g_bout_pre[32]);

  check_mode42_replay_deterministic();
  print_uniqueness();
  run_combos();
  check_mode43_predictions();
  check_mode43_host_contract();

  printf("prm_layout_checks=%u fail=%u\n", (unsigned)g_checks,
         (unsigned)g_fail);
  if (g_fail != 0) {
    printf("prm_layout_sim=FAIL\n");
    return 1;
  }
  printf("prm_layout_sim=PASS\n");
  return 0;
}
