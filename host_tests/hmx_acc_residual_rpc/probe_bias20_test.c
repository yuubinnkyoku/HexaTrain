// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only spec test for the Phase D 20-bit output-bias encoding
// (bias20.h). No SDK, no device, no Hexagon toolchain — plain C11.
//
// This is the "independent calculation" for Phase D: every expectation
// is re-derived here directly from the spec formula of the README
// "Phase D" section — value = -(64.5 + n*2^-8) — using nothing but
// double arithmetic, and the shared encode/decode in bias20.h is checked
// against it for all nibbles n = 0..15. It also pins the byte order of
// the cells inside the 256-byte VTCM bias row (PRM 80-N2040-62 AA,
// §2.3.1 Fig.13 p.13: 32 low words at bytes 0..127, then 32 high words
// at bytes 128..255; cells are little-endian uint32 in memory) and
// justifies the mode-37 fallback by proving that the exact-zero target
// (1+2^-8)^2*64 = 64.5009765625 is not representable in the 20-bit
// output-bias format (Fig.15/Table 4 p.14).
//
// Build + run:
//   gcc -std=c11 -O2 -Wall -Wextra -Werror -o probe_bias20_test probe_bias20_test.c -lm
//   ./probe_bias20_test
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bias20.h"

static int failures = 0;
static long checks = 0;

static void check_i(long got, long want, const char *what) {
  ++checks;
  if (got != want) {
    ++failures;
    printf("FAIL %s got=%ld want=%ld\n", what, got, want);
  }
}

static void check_d(double got, double want, const char *what) {
  ++checks;
  if (got != want) {           /* exact double equality is the point */
    ++failures;
    printf("FAIL %s got=%.17g want=%.17g\n", what, got, want);
  }
}

int main(void) {
  const uint16_t obias_base = 0xD408u;   /* -64.5, sweep base */
  const uint16_t scale_fp16 = 0x5400u;   /* 64.0 */
  int sweep_cases = 0;
  int sweep_failures = 0;

  /* 1. Encode/decode sweep: n = 0..15 against the spec formula. */
  for (int n = 0; n <= 15; ++n) {
    const int before = failures;
    const uint32_t lo = bias20_lo_cell(scale_fp16, obias_base);
    const uint32_t hi = bias20_hi_cell(0u, (unsigned)n, 0u);
    /* Cell bit layout, PRM Table 4: lo[15:0]=scale, lo[31:16]=out_bias,
     * hi[7:4]=out_bias extra4, hi[10:8]=shape, hi[3:0]=scale extra4. */
    check_i((long)(lo & 0xFFFFu), 0x5400L, "lo[15:0] holds scale fp16");
    check_i((long)(lo >> 16), 0xD408L, "lo[31:16] holds out_bias fp16");
    check_i((long)((hi >> 4) & 0xFu), n, "hi[7:4] holds extra4 nibble");
    check_i((long)((hi >> 8) & 7u), 0, "hi[10:8] shape stays 0 in sweep");
    check_i((long)(hi & 0xFu), 0, "hi[3:0] scale extra4 stays 0");
    /* Independent double expectation from the spec formula:
     * value = -(64.5 + n*2^-8), exact in double for n = 0..15. */
    const double want = -(64.5 + (double)n * ldexp(1.0, -8));
    const double got =
        bias20_decode_out_bias((uint16_t)(lo >> 16),
                               (unsigned)((hi >> 4) & 0xFu));
    check_d(got, want, "decode equals -(64.5 + n*2^-8)");
    ++sweep_cases;
    if (failures != before) ++sweep_failures;
  }
  printf("encode_decode_sweep=PASS cases=%d failures=%d\n",
         sweep_cases, sweep_failures);

  /* 2. Scale / shape round trip: encode 0x5400 and shape 1, decode back. */
  {
    const uint32_t lo = bias20_lo_cell(scale_fp16, obias_base);
    const uint32_t hi = bias20_hi_cell(1u, 0u, 0u);
    check_d(fp16_to_double((uint16_t)(lo & 0xFFFFu)), 64.0,
            "scale cell 0x5400 decodes to 64.0");
    check_i((long)((hi >> 8) & 7u), 1, "shape=1 round trips via hi[10:8]");
    check_i((long)(hi & 0xFFu), 0, "shape must not touch hi[7:0]");
  }

  /* 3. Row byte order (Fig.13): lo cells at bytes 4*ch, hi cells at
   *    bytes 128 + 4*ch, little-endian uint32. Payload: shape=1, n=1. */
  {
    uint8_t row[256];
    uint32_t lo_rb = 0, hi_rb = 0;
    memset(row, 0, sizeof row);
    for (int ch = 0; ch < 32; ++ch) {
      const uint32_t lo = bias20_lo_cell(scale_fp16, obias_base);
      const uint32_t hi = bias20_hi_cell(1u, 1u, 0u);
      memcpy(row + 4 * ch, &lo, 4);
      memcpy(row + 128 + 4 * ch, &hi, 4);
    }
    check_i(row[0], 0x00, "row[0] = scale fp16 low byte (LE)");
    check_i(row[1], 0x54, "row[1] = scale fp16 high byte (LE)");
    check_i(row[2], 0x08, "row[2] = out_bias fp16 low byte (LE)");
    check_i(row[3], 0xD4, "row[3] = out_bias fp16 high byte (LE)");
    check_i(row[128], 0x10, "row[128] = hi cell byte0, extra4 n=1 in [7:4]");
    check_i(row[129], 0x01, "row[129] = hi cell byte1, shape=1 in [10:8]");
    check_i(row[130], 0x00, "row[130] = hi cell byte2 is zero");
    check_i(row[131], 0x00, "row[131] = hi cell byte3 is zero");
    /* Round trip: read the cells back from the row and decode them. */
    memcpy(&lo_rb, row, 4);
    memcpy(&hi_rb, row + 128, 4);
    check_i((long)lo_rb, 0xD4085400L, "lo cell round trip through row");
    check_i((long)hi_rb, 0x00000110L, "hi cell round trip through row");
    check_d(bias20_decode_out_bias((uint16_t)(lo_rb >> 16),
                                   (unsigned)((hi_rb >> 4) & 0xFu)),
            -(64.5 + ldexp(1.0, -8)), "row-decoded out bias n=1");
    check_d(fp16_to_double((uint16_t)(lo_rb & 0xFFFFu)), 64.0,
            "row-decoded scale");
  }

  /* 4. Mode-37 fallback justification: the exact-zero magnitude
   *    (1+2^-8)^2*64 = 64.5009765625 must be off the 20-bit grid
   *    base_fp16 + n*2^-8. Scanning every FP16 base of the [64,128)
   *    binade (sign=1, e_field=21) against n=0..15 is exhaustive here:
   *    the largest e_field=5 value plus nibble reach tops out below
   *    64.0 + 15*2^-8 - 2^-5 < target, and the smallest e_field=7 base
   *    is 128 > target. */
  {
    const double a = 1.0 + ldexp(1.0, -8);
    const double target = a * a * 64.0;
    int reachable = 0;
    for (uint32_t f = 0; f < 1024u && !reachable; ++f) {
      const uint16_t base = (uint16_t)(0xD400u | f); /* sign=1, e=6 */
      for (int n = 0; n <= 15; ++n) {
        if (bias20_decode_out_bias(base, (unsigned)n) == -target) {
          reachable = 1;
          break;
        }
      }
    }
    check_i(reachable, 0, "exact-zero target off the 20-bit grid");
    printf("exact_zero_target=%.17g reachable=%d (mode 37 fallback)\n",
           target, reachable);
    /* And the fallback itself: a = b = 1.0, scale 64, out_bias -64.0,
     * so residual = 64*1*1 + (-64) == 0 exactly in double. */
    check_d(64.0 * 1.0 * 1.0 + bias20_decode_out_bias(0xD400u, 0u), 0.0,
            "mode 37 fallback residual is exactly 0");
  }

  /* 4. Mode 39 (E-1 range discriminator) base: out_bias FP16 0xD404 =
   * -64.25. It shares the 0xD408 exponent field (e = 6), so the fine
   * grid stays 2^-8. Independent expectation:
   * value = -(64.25 + n*2^-8), exact in double for all n. */
  {
    int base_cases = 0, base_failures = 0;
    const long before = failures;
    check_i((long)(0xD404u & 0x7c00u), (long)(0xD408u & 0x7c00u),
            "0xD404 shares the 0xD408 exponent field (e=6)");
    check_i((long)(0xD404u >> 15), 1L, "0xD404 is negative");
    check_d(fp16_to_double(0xD404u), -64.25,
            "fp16_to_double(0xD404) = -64.25");
    for (unsigned n = 0; n <= 15; ++n) {
      const double want = -(64.25 + (double)n * ldexp(1.0, -8));
      check_d(bias20_decode_out_bias(0xD404u, n), want,
              "decode(0xD404,n) equals -(64.25 + n*2^-8)");
      ++base_cases;
      if (failures != before) ++base_failures;
    }
    /* And the discriminator arithmetic itself:
     * 64*1.00390625^2 - 64.25 = 64.5009765625 - 64.25 = 0.2509765625
     * = 64.25 grid units, exactly representable as FP16 0x3404. */
    check_d(64.0 * 1.00390625 * 1.00390625
                + bias20_decode_out_bias(0xD404u, 0u),
            0.2509765625, "mode 39 target is 64.25 grid units");
    printf("mode39_base_sweep=PASS cases=%d failures=%d\n",
           base_cases, base_failures);
  }

  if (failures == 0) {
    printf("probe_bias20_test=PASS checks=%ld\n", checks);
    return 0;
  }
  printf("probe_bias20_test=FAIL failures=%d checks=%ld\n", failures, checks);
  return 1;
}
