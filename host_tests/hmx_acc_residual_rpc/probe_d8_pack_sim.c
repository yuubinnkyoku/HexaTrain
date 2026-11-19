// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Hexagon-sim harness for the D8 prologue, pack included.
//
// The first sim harness ran only the generator and reproduced the registered
// constants, which ruled out FMA contraction but left one step untested: the
// generator's output is copied into VTCM by d8_pack_tile's 16 x 128 B vmem
// stores, and the FNV then reads VTCM back with scalar loads. That path has
// never run on the simulator.
//
// This program runs the whole prologue - generate, pack into a VTCM-like
// buffer, read back, hash - and prints the four FNVs. If they match the device
// values (57a0ec09 / d0e0c444 / 9c9fe034 / 29a90a5c) the cause is the pack
// layout and the question is settled offline. If they match the registered
// constants, the pack is innocent and the difference is in floating-point
// execution.
//
// It is a diagnostic. It does not touch the device, changes no registered value,
// and does not retire B3_R2_SYNC_UNPROVEN.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "fp16_util.h"
#include "hmx_d8_contract.h"
#include "hmx_mode46.h"

#define E3C1_T8 32
#define E3C1_T8_CELLS (E3C1_T8 * E3C1_T8)
#define TILE_BYTES (HMX_MODE46_TILE_VECTORS * HMX_MODE46_VECTOR_BYTES)
#define VTCM_BYTES 12288

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

/* ---- copied verbatim from hmx_mode46_d8.h -------------------------------- */
static void d8_pack_tile(uint8_t *dst_ptr, const uint16_t *src) {
  const uintptr_t dst = (uintptr_t)dst_ptr;
  int v;
  for (v = 0; v < HMX_MODE46_TILE_VECTORS; ++v) {
    const HVX_Vector chunk =
        *(const HVX_Vector *)((const uint8_t *)src +
                              (size_t)(v * HMX_MODE46_VECTOR_BYTES));
    asm volatile("vmem(%0) = %1\n"
                 : : "r"(dst + (uintptr_t)(v * HMX_MODE46_VECTOR_BYTES)),
                     "v"(chunk)
                 : "memory");
  }
}

static void build(uint8_t *base, uint16_t *ahi, uint16_t *alo,
                  uint16_t *whi, uint16_t *wlo) {
  static uint16_t ahi_local[E3C1_T8_CELLS];
  static uint16_t alo_local[E3C1_T8_CELLS];
  static uint16_t whi_local[E3C1_T8_CELLS];
  static uint16_t wlo_local[E3C1_T8_CELLS];
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

  /* PACK AHI, PACK ALO, PACK WHI, PACK WLO, in that order. */
  d8_pack_tile(base + 0u, ahi_local);
  d8_pack_tile(base + 2048u, alo_local);
  d8_pack_tile(base + 4096u, whi_local);
  d8_pack_tile(base + 6144u, wlo_local);

  /* The handler then hashes the VTCM pointers, not the locals. */
  (void)ahi; (void)alo; (void)whi; (void)wlo;
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

/* A VTCM-like buffer. The device gets this from HAP_compute_res; here it is
 * static storage, which is enough because the question is the byte layout, not
 * the allocation. */
static uint8_t vtcm[VTCM_BYTES] __attribute__((aligned(128)));

int main(void) {
  const char *names[4] = {"ahi", "alo", "whi", "wlo"};
  uint16_t *tiles[4];
  uint32_t want[4];
  static const uint32_t dev[4] = {0x57A0EC09u, 0xD0E0C444u, 0x9C9FE034u,
                                   0x29A90A5Cu};
  int k, i;

  tiles[0] = (uint16_t *)(vtcm + HMX_MODE46_AHI_OFFSET);
  tiles[1] = (uint16_t *)(vtcm + HMX_MODE46_ALO_OFFSET);
  tiles[2] = (uint16_t *)(vtcm + HMX_MODE46_WHI_OFFSET);
  tiles[3] = (uint16_t *)(vtcm + HMX_MODE46_WLO_OFFSET);
  want[0] = (uint32_t)HMX_MODE46_FNV_AHI;
  want[1] = (uint32_t)HMX_MODE46_FNV_ALO;
  want[2] = (uint32_t)HMX_MODE46_FNV_WHI;
  want[3] = (uint32_t)HMX_MODE46_FNV_WLO;

  memset(vtcm, 0, VTCM_BYTES);
  build(vtcm, tiles[0], tiles[1], tiles[2], tiles[3]);

  for (k = 0; k < 4; ++k) {
    const uint32_t f = tile_fnv(tiles[k]);
    printf("sim_pack %s fnv=%08x registered=%08x device=%08x %s\n", names[k],
           (unsigned)f, (unsigned)want[k], (unsigned)dev[k],
           f == want[k] ? "MATCH_REGISTERED"
                        : (f == dev[k] ? "MATCH_DEVICE" : "matches_neither"));
  }

  /* The first 16 cells after the pack, so a byte-level bisect has a starting
   * point. */
  for (k = 0; k < 4; ++k) {
    printf("sim_pack_head %s", names[k]);
    for (i = 0; i < 16; ++i) printf(" %04x", (unsigned)tiles[k][i]);
    printf("\n");
  }

  return 0;
}
