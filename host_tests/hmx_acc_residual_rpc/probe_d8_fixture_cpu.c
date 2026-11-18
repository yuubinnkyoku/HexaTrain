// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU diagnostic for the D8 operand fixture.
//
// The device run stopped at the handler's own operand-FNV gate: the four tiles
// the DSP generated did not hash to the registered constants. The guard FNV was
// correct, so the FNV function and the VTCM read path work; the difference is in
// the operand bytes.
//
// This program reproduces the fixture generation from hmx_mode46_d8.h on the CPU
// and hashes the four tiles exactly as the handler does. It answers one question
// and nothing else:
//
//   does the fixture generation, run on the CPU, produce the registered FNVs?
//
//   - if yes, the generation is right and the device difference is in how the
//     bytes reach VTCM (the pack, or a missing barrier before the scalar read);
//   - if no, the generation itself is wrong and the registered constants were
//     never matched by any implementation.
//
// The helpers are copied verbatim from probe_dsp.c (e3c1_rng64,
// e3c1_rng_unit, e3c1_split_lo) and fp16_util.h (double_to_fp16,
// fp16_to_double), not reimplemented, because a reimplementation would only test
// itself.
//
// This is a diagnostic, not a device result. It does not touch the DSP, does not
// change any registered value, and does not retire B3_R2_SYNC_UNPROVEN.
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

/* Uniform in [0,1) with 53 random bits. */
static double e3c1_rng_unit(void) {
  return (double)(e3c1_rng64() >> 11) * (1.0 / 9007199254740992.0);
}

/* Float-arithmetic split residual: statement order mirrors the sim's
 * split_float_parts — hi = RN16(x) is computed by the caller FIRST,
 * then (float)x - (float)hi in FLOAT arithmetic as separate statements
 * (no FMA can contract), then lo = RN16(sub). */
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
  uint32_t f[4];
  uint32_t want[4];
  const char *names[4] = {"ahi", "alo", "whi", "wlo"};
  int fail = 0, i;

  build(ahi, alo, whi, wlo);
  f[0] = tile_fnv(ahi);
  f[1] = tile_fnv(alo);
  f[2] = tile_fnv(whi);
  f[3] = tile_fnv(wlo);
  want[0] = (uint32_t)HMX_MODE46_FNV_AHI;
  want[1] = (uint32_t)HMX_MODE46_FNV_ALO;
  want[2] = (uint32_t)HMX_MODE46_FNV_WHI;
  want[3] = (uint32_t)HMX_MODE46_FNV_WLO;

  for (i = 0; i < 4; ++i) {
    printf("d8_fixture %s cpu=%08x registered=%08x %s\n", names[i],
           (unsigned)f[i], (unsigned)want[i], f[i] == want[i] ? "match" : "MISMATCH");
    if (f[i] != want[i]) fail = 1;
  }

  /* The device reported these four values; name them so the record can say
   * whether the CPU agrees with the device or with the registration. */
  {
    static const uint32_t dev[4] = {0x57A0EC09u, 0xD0E0C444u, 0x9C9FE034u,
                                    0x29A90A5Cu};
    for (i = 0; i < 4; ++i) {
      printf("d8_vs_device %s cpu=%08x device=%08x %s\n", names[i],
             (unsigned)f[i], (unsigned)dev[i],
             f[i] == dev[i] ? "AGREES_WITH_DEVICE" : "differs_from_device");
    }
  }

  if (fail) {
    printf("d8_fixture_cpu=FAIL (the generator does not reproduce the registered FNVs)\n");
    return 1;
  }
  printf("d8_fixture_cpu=PASS (the generator reproduces the registered FNVs)\n");
  return 0;
}
