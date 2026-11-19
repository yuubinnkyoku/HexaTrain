// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Hexagon-sim harness for the D8 operand fixture.
//
// The device run stops at the operand-FNV gate: the four tiles the DSP builds do
// not hash to the registered constants, while the same generator run on the CPU
// reproduces all four. The difference is therefore in how the DSP evaluates the
// generator, not in the generator's logic.
//
// This program runs the generator on the V81 simulator with the same compiler
// and flags as the stamped skel, and prints the four FNVs plus the raw tile
// bytes. If it reproduces the device values (57a0ec09 / d0e0c444 / 9c9fe034 /
// 29a90a5c) then the difference can be bisected offline, byte by byte, with no
// device contact.
//
// It is a diagnostic. It does not touch the device, changes no registered value,
// and does not retire B3_R2_SYNC_UNPROVEN.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fp16_util.h"
#include "hmx_d8_contract.h"
#include "hmx_mode46.h"

#define E3C1_T8 32
#define E3C1_T8_CELLS (E3C1_T8 * E3C1_T8)
#define TILE_BYTES (HMX_MODE46_TILE_VECTORS * HMX_MODE46_VECTOR_BYTES)

/* ---- copied verbatim from probe_dsp.c ------------------------------------ */
static uint64_t e3c1_rng;
static double e3c1_x[E3C1_T8_CELLS];
static double e3c1_w[E3C1_T8_CELLS];

static uint64_t e3c1_rng64(void) {
  uint64_t x = e3c1_rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  e3c1_rng = x;
  return x;
}

static double e3c1_rng_unit(void) {
  return (double)(e3c1_rng64() >> 11) * (1.0 / 9007199254740992.0);
}

static uint16_t e3c1_split_lo(double x, uint16_t hi_bits) {
  float xf = (float)x;
  float hf = (float)fp16_to_double(hi_bits);
  float sub = xf - hf;
  return double_to_fp16((double)sub);
}

/* ---- copied verbatim from hmx_mode46_d8.h build_d8_operands -------------- */
static void build(uint16_t *ahi_local, uint16_t *alo_local,
                  uint16_t *whi_local, uint16_t *wlo_local) {
  int idx;
  int s, o, i;

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

  for (o = 0; o < E3C1_T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < E3C1_T8; ++i) e3c1_w[(long)o * E3C1_T8 + i] = 0.0;
    }
  }
  e3c1_w[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11;
  e3c1_w[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_w[0 * E3C1_T8 + 2] = 0x1p-13;
  e3c1_w[1 * E3C1_T8 + 0] = 1.0 - 0x1p-12;
  e3c1_w[2 * E3C1_T8 + 0] = 0x1p-9;
  e3c1_w[3 * E3C1_T8 + 0] = 1.0;
  e3c1_w[3 * E3C1_T8 + 1] = 1.0 + 0x1p-12;
  e3c1_w[28 * E3C1_T8 + 0] = 0x1p-10;
  e3c1_w[29 * E3C1_T8 + 0] = 0x1p-10 - 0x1p-21;
  e3c1_w[30 * E3C1_T8 + 0] = 3.0 * 0x1p-12;
  e3c1_w[31 * E3C1_T8 + 0] = 0x1p-13;

  for (s = 0; s <= 3; ++s)
    for (i = 0; i < E3C1_T8; ++i) e3c1_x[(long)s * E3C1_T8 + i] = 0.0;
  e3c1_x[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11;
  e3c1_x[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_x[0 * E3C1_T8 + 2] = 0x1p-13;
  for (i = 0; i < E3C1_T8; ++i)
    e3c1_x[1 * E3C1_T8 + i] = -e3c1_x[i];
  e3c1_x[2 * E3C1_T8 + 0] = 1.0 - 0x1p-12;
  e3c1_x[2 * E3C1_T8 + 1] = -(1.0 - 0x1p-12);
  e3c1_x[3 * E3C1_T8 + 0] = 0x1p-10;

  memset(ahi_local, 0, TILE_BYTES);
  memset(alo_local, 0, TILE_BYTES);
  memset(whi_local, 0, TILE_BYTES);
  memset(wlo_local, 0, TILE_BYTES);
  for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
    const int row = idx >> 5;
    const int col = idx & 31;
    const uint16_t ah = double_to_fp16(e3c1_x[idx]);
    const uint16_t wh = double_to_fp16(e3c1_w[idx]);
    const uint16_t al = e3c1_split_lo(e3c1_x[idx], ah);
    const uint16_t wl = e3c1_split_lo(e3c1_w[idx], wh);
    ahi_local[((row >> 1) * 64) + col * 2 + (row & 1)] = ah;
    alo_local[((row >> 1) * 64) + col * 2 + (row & 1)] = al;
    whi_local[((col >> 1) * 64) + row * 2 + (col & 1)] = wh;
    wlo_local[((col >> 1) * 64) + row * 2 + (col & 1)] = wl;
  }
}

static uint32_t tile_fnv(const uint16_t *tile) {
  const uint8_t *p = (const uint8_t *)tile;
  uint32_t h = 2166136261u;
  size_t i;
  for (i = 0; i < TILE_BYTES; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;
  }
  return h;
}

int main(void) {
  static uint16_t ahi[E3C1_T8_CELLS], alo[E3C1_T8_CELLS];
  static uint16_t whi[E3C1_T8_CELLS], wlo[E3C1_T8_CELLS];
  const char *names[4] = {"ahi", "alo", "whi", "wlo"};
  uint16_t *tiles[4];
  uint32_t want[4];
  int k, i;

  tiles[0] = ahi; tiles[1] = alo; tiles[2] = whi; tiles[3] = wlo;
  want[0] = (uint32_t)HMX_MODE46_FNV_AHI;
  want[1] = (uint32_t)HMX_MODE46_FNV_ALO;
  want[2] = (uint32_t)HMX_MODE46_FNV_WHI;
  want[3] = (uint32_t)HMX_MODE46_FNV_WLO;

  build(ahi, alo, whi, wlo);

  for (k = 0; k < 4; ++k) {
    const uint32_t f = tile_fnv(tiles[k]);
    printf("sim_fixture %s fnv=%08x registered=%08x %s\n", names[k],
           (unsigned)f, (unsigned)want[k], f == want[k] ? "match" : "MISMATCH");
  }

  /* The device reported these; name them so the record can say whether the
   * simulator agrees with the device or with the registration. */
  {
    static const uint32_t dev[4] = {0x57A0EC09u, 0xD0E0C444u, 0x9C9FE034u,
                                    0x29A90A5Cu};
    for (k = 0; k < 4; ++k) {
      const uint32_t f = tile_fnv(tiles[k]);
      printf("sim_vs_device %s sim=%08x device=%08x %s\n", names[k],
             (unsigned)f, (unsigned)dev[k],
             f == dev[k] ? "AGREES_WITH_DEVICE" : "differs_from_device");
    }
  }

  /* The first 16 cells of each tile, so a byte-level bisect has a starting
   * point even when the digests differ. */
  for (k = 0; k < 4; ++k) {
    printf("sim_tile_head %s", names[k]);
    for (i = 0; i < 16; ++i) printf(" %04x", (unsigned)tiles[k][i]);
    printf("\n");
  }

  /* The first eight generator values, which is where a rounding difference
   * would first show up. */
  {
    uint64_t save = e3c1_rng;
    int n;
    e3c1_rng = 0xE3C0117E5EEDF00Dull;
    printf("sim_rng_head");
    for (n = 0; n < 8; ++n) {
      double u = e3c1_rng_unit();
      double v = u * 4.0;
      double val = v - 2.0;
      printf(" %04x", (unsigned)double_to_fp16(val));
    }
    printf("\n");
    e3c1_rng = save;
  }

  return 0;
}
