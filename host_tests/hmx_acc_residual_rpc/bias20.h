// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// 20-bit bias-cell encode/decode + Phase D payload table (modes 31..38)
// for the HMX residual probe.
//
// Layout source: Qualcomm, Hexagon V81 HMX PRM, 80-N2040-62 Rev AA,
// §2.3.1 "Bias register":
//   - Figure 13 (p.13): the 256-byte VTCM row of one bias register set
//     is 32 LOW words at bytes 0..127 followed by 32 HIGH words at bytes
//     128..255 (not interleaved uint64).
//   - Figure 15 + Table 4 (p.14): per-channel 64-bit register =
//       lo32 [31:16] = output bias FP16, lo32 [15:0] = scale FP16
//       hi32 [10:8]  = shape, hi32 [7:4] = output bias extra4 nibble,
//       hi32 [3:0]   = scale extra4 nibble.
//   - Numeric meaning of the 20-bit output-bias value (Fig.15):
//       value = (-1)^s * 2^(e-15) * (1 + fp16_frac/1024 + nibble/16384)
//     so on the negative -64.5 base (fp16 0xD408, e_field=21) nibble n
//     decodes to -(64.5 + n*2^-8). The scale extra4 nibble stays 0 in
//     every Phase D payload (scale is plain FP16 0x5400 = 64.0).
//
// Shared by probe_dsp.c (programs the cells), probe_host.c (double
// oracle) and probe_bias20_test.c (CPU-only spec test) so the three
// cannot drift apart.

#ifndef HEXATRAIN_BIAS20_H
#define HEXATRAIN_BIAS20_H

#include <stdint.h>

#include "fp16_util.h"

/* lo32 cell of one bias channel (first 128-byte vector, Fig.13). */
static inline uint32_t bias20_lo_cell(uint16_t scale_fp16,
                                      uint16_t out_bias_fp16) {
  return ((uint32_t)out_bias_fp16 << 16) | (uint32_t)scale_fp16;
}

/* hi32 cell of one bias channel (second 128-byte vector, Fig.13). */
static inline uint32_t bias20_hi_cell(unsigned shape,
                                      unsigned out_bias_nibble,
                                      unsigned scale_nibble) {
  return ((shape & 7u) << 8) | ((out_bias_nibble & 0xFu) << 4) |
         (scale_nibble & 0xFu);
}

/* Decode the signed 20-bit output-bias value from its FP16 half and the
 * extra4 nibble (Fig.15 formula above). Normal-range payloads only —
 * both probe encodings (-64.5, -64.0) are normal numbers. The nibble
 * adds |2^(e-15) * n/16384| on the magnitude side of the FP16 sign. */
static inline double bias20_decode_out_bias(uint16_t out_bias_fp16,
                                            unsigned nibble) {
  const double base = fp16_to_double(out_bias_fp16);
  const int exp_field = (out_bias_fp16 >> 10) & 31;
  const double binade = ldexp(1.0, exp_field - 15);
  const double extra = binade * ((double)(nibble & 0xFu) / 16384.0);
  return (out_bias_fp16 & 0x8000u) ? base - extra : base + extra;
}

/* Phase D extra4 nibble per mode; -1 outside modes 31..39.
 *   31..36 : sweep n = 0, 1, 2, 8, 15  (+ mode 36 border control n = 3)
 *   37     : exact-zero FALLBACK encoding uses nibble 0 (base -64.0);
 *            the intended n=1 alignment is off the 20-bit grid (see the
 *            README "Phase D" section and probe_bias20_test.c).
 *   38     : readback probe programs the n=1 sweep payload.
 *   39     : E-1 range discriminator, base -64.25 (0xD404), nibble 0;
 *            the convert target is 0.2509765625 = 64.25 grid units. */
static inline int phase_d_extra4_nibble(int mode) {
  switch (mode) {
    case 31: return 0;
    case 32: return 1;
    case 33: return 2;
    case 34: return 8;
    case 35: return 15;
    case 36: return 3;
    case 37: return 0;
    case 38: return 1;
    case 39: return 0;
    default: return -1;
  }
}

/* FP16 half of the Phase D output bias: -64.5 (0xD408) for the sweep and
 * the readback probe, -64.0 (0xD400) for the mode-37 exact-zero control,
 * -64.25 (0xD404) for the mode-39 E-1 range discriminator; 0 outside
 * modes 31..39. */
static inline uint16_t phase_d_out_bias_fp16(int mode) {
  if (mode == 37) return 0xD400u;
  if (mode == 39) return 0xD404u;
  if (mode >= 31 && mode <= 38) return 0xD408u;
  return 0u;
}

#endif /* HEXATRAIN_BIAS20_H */
