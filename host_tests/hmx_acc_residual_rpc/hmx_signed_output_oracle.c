// SPDX-License-Identifier: Apache-2.0
// HMX V81 signed-output investigation: PRM-derived convert oracle (CPU-only).
//
// PRM source: 80-N2040-62 Rev AA.
//   Figure 11 (p.12): convert = s * shape(ACC + bi) + bo
//   Table 5  (p.14): shape 0=x, 1=min(x,0), 2=max(x,0), 3=|x|,
//                    4=-x, 5=-min(x,0), 6=-max(x,0), 7=-|x|
//   Figure 15/Table 4 (p.14): per-output-channel 64-bit bias register
//   Table 6  (p.15): Rs[13:12]=bias set, Rs[8]=rnd, Rs[6]=ovf,
//                    Rs[4]=fb op, Rs[3:2]=fb dest, Rs[0]=acc behavior
//                    (0 = clear+swap, 1 = retain)
//   p.13: "the bias cells 0 through 31 within the set are applied to the
//         corresponding accumulator output channel values"
//
// This model implements ONLY the PRM equation above plus the measured
// device law "negative stored value -> 0x0000" as a SEPARATE, explicitly
// labeled empirical clamp. Nothing else (no min/max feedback, no
// saturation, no USR overflow path) is modeled because the surveyed
// modes use Rs fb_dest=0 (no feedback) and small magnitudes.
// Existing experiment expectations are NOT touched by this file.
//
// Sections below (2026-10-24, CPU-only; no device contact):
//   1. PRM convert cases already pinned by the previous session.
//   2. E-3D fixture: a = 1+2^-6, exact ACC values, layout-driven matrix.
//   3. Static PRM VTCM layout (act / wgt / store) + bijection.
//   4. Per-output-channel bias cells (the mixed-shape tile).
//   5. The four pre-registered discriminator cells, variants A and B.
//   6. Rounding-mode and cause-model pattern separation.
//   7. Rs encoding (bias set / acc behavior / rnd / fb) semantics.
//   8. Two-convert split reconstruction: y = max(y,0) - max(-y,0).
//   9. mode 16/17 regression (PRM math vs recorded device values).

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bias20.h"
#include "fp16_util.h"

static int failures = 0;
static long checks = 0;

static void check_bits(const char *tag, uint16_t got, uint16_t want) {
  ++checks;
  if (got != want) {
    ++failures;
    printf("FAIL %s got=0x%04x want=0x%04x\n", tag, (unsigned)got,
           (unsigned)want);
  }
}

static void check_true(const char *tag, int cond) {
  ++checks;
  if (!cond) {
    ++failures;
    printf("FAIL %s\n", tag);
  }
}

static void check_u32(const char *tag, uint32_t got, uint32_t want) {
  ++checks;
  if (got != want) {
    ++failures;
    printf("FAIL %s got=0x%08x want=0x%08x\n", tag, (unsigned)got,
           (unsigned)want);
  }
}

// PRM Figure 11 core: convert = s*shape(acc+bi)+bo, then RN16.
// bi is zero in all surveyed modes; modeled anyway for completeness.
static uint16_t prm_convert_rn16(double acc, double bi, unsigned shape,
                                 double scale, double out_bias) {
  const double x = acc + bi;
  double shaped;
  switch (shape & 7u) {
    case 0: shaped = x; break;
    case 1: shaped = x < 0.0 ? x : 0.0; break;
    case 2: shaped = x > 0.0 ? x : 0.0; break;
    case 3: shaped = fabs(x); break;
    case 4: shaped = -x; break;
    case 5: shaped = -(x < 0.0 ? x : 0.0); break;
    case 6: shaped = -(x > 0.0 ? x : 0.0); break;
    case 7: shaped = -fabs(x); break;
    default: shaped = x; break;
  }
  return double_to_fp16(scale * shaped + out_bias);
}

// Empirical device law (README law 4, NOT PRM): negative stored value -> 0x0000.
static uint16_t empirical_negative_clamp(uint16_t pre) {
  return (pre & 0x8000u) ? 0x0000u : pre;
}

// ---------------------------------------------------------------------------
// 2. E-3D fixture
//
// a = 1 + 2^-6 = 1.015625 (0x3C10, exact FP16)
// x = a*a     = 4225/4096 = 1.031494140625 (exact in the 37-bit ACC)
// 64*x        = 66.015625  -> FP16 grid 66.0/66.0625, exactly 0.25 ULP above
//                              the lower neighbour (ULP = 2^-4 = 0.0625)
// 2x          = 2.06298828125
// 128*x       = 132.03125  -> FP16 grid 132.0/132.125 (ULP = 2^-3 = 0.125)
//
// Both scaled values sit exactly 0.25 ULP above an FP16 grid point, so the
// RN, ceiling and floor exports give three different bit patterns. That
// makes the four-cell tile a rounding-mode discriminator as well as a
// signed-output discriminator, without any extra MMA/convert/store.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 3. Static PRM VTCM layout (E-3C-DIAG re-derivation, device-corroborated
//    at store index 32 only; PRM §2.2.1/Fig.4, §2.2.2/Fig.8, §2.4/Figs.18-19)
// ---------------------------------------------------------------------------
static unsigned act_u16(unsigned s, unsigned i) {
  return (s >> 1) * 64u + i * 2u + (s & 1u);
}
static unsigned wgt_u16(unsigned o, unsigned i) {
  return (i >> 1) * 64u + o * 2u + (i & 1u);
}
static unsigned store_u16(unsigned s, unsigned o) {
  return (s >> 1) * 64u + o * 2u + (s & 1u);
}

// ---------------------------------------------------------------------------
// 4. Per-output-channel bias row (PRM §2.3.1, Fig.13/Fig.15).
//    256 bytes = 32 lo words (bytes 0..127) then 32 hi words (128..255).
//    lo32 = output_bias_fp16 << 16 | scale_fp16
//    hi32 = shape << 8 | obias_extra4 << 4 | scale_extra4
//    Cell index o is the output channel (PRM p.13), NOT a spatial index.
// ---------------------------------------------------------------------------
static void bias_row_build(uint8_t *row, const uint16_t *scale_bits,
                           const uint16_t *obias_bits, const unsigned *shape,
                           unsigned channels) {
  memset(row, 0, 256);
  for (unsigned o = 0; o < channels; ++o) {
    const uint32_t lo = bias20_lo_cell(scale_bits[o], obias_bits[o]);
    const uint32_t hi = bias20_hi_cell(shape[o], 0u, 0u);
    memcpy(row + o * 4u, &lo, sizeof(lo));
    memcpy(row + 128u + o * 4u, &hi, sizeof(hi));
  }
}

static uint32_t bias_row_lo(const uint8_t *row, unsigned o) {
  uint32_t v;
  memcpy(&v, row + o * 4u, sizeof(v));
  return v;
}
static uint32_t bias_row_hi(const uint8_t *row, unsigned o) {
  uint32_t v;
  memcpy(&v, row + 128u + o * 4u, sizeof(v));
  return v;
}
static unsigned bias_row_shape(const uint8_t *row, unsigned o) {
  return (bias_row_hi(row, o) >> 8) & 7u;
}

// ---------------------------------------------------------------------------
// 5. The E-3D discriminator tile.
//
//   A(s0,i0) = +a, A(s1,i0) = -a, all else 0
//   W(o0,i0) = +a
//
// Variant A (user-specified): W(o1,i0) = +a  -> o1 column is the same |x|
// Variant B (recommended):    W(o1,i0) = +2a -> o1 column is |2x|
//
// Bias cells (per output channel, one set):
//   o0: scale 0x5400 (+64), shape 0, bo 0
//   o1: scale 0x5400 (+64), shape 4, bo 0
//   o2..o31: scale 0x5400, shape 0, bo 0 (safe default: zero ACC -> +0)
//
// Expected (PRM only, no clamp), RN16, four store cells:
//   out[0]=(s0,o0) out[1]=(s1,o0) out[2]=(s0,o1) out[3]=(s1,o1)
//   A: 0x5420, 0xD420, 0xD420, 0x5420
//   B: 0x5420, 0xD420, 0xD820, 0x5820
// ---------------------------------------------------------------------------
struct tile {
  uint16_t act[1024];
  uint16_t wgt[1024];
  uint8_t bias[256];
};

static void tile_build(struct tile *t, uint16_t w_o1, unsigned shape_o1) {
  memset(t, 0, sizeof(*t));
  t->act[act_u16(0, 0)] = 0x3C10u;  // +a
  t->act[act_u16(1, 0)] = 0xBC10u;  // -a
  t->wgt[wgt_u16(0, 0)] = 0x3C10u;  // +a
  t->wgt[wgt_u16(1, 0)] = w_o1;

  uint16_t scale[32], obias[32];
  unsigned shape[32];
  for (unsigned o = 0; o < 32; ++o) {
    scale[o] = 0x5400u;  // +64
    obias[o] = 0x0000u;  // +0
    shape[o] = 0u;
  }
  shape[1] = shape_o1;  // per-channel shape, PRM p.13
  bias_row_build(t->bias, scale, obias, shape, 32);
}

// ACC(s,o) = sum_i A(s,i)*W(o,i) through the packed VTCM tiles.
static double tile_acc(const struct tile *t, unsigned s, unsigned o) {
  double sum = 0.0;
  for (unsigned i = 0; i < 32; ++i) {
    sum += fp16_to_double(t->act[act_u16(s, i)]) *
           fp16_to_double(t->wgt[wgt_u16(o, i)]);
  }
  return sum;
}

// Convert one cell of the tile exactly as the pre-registered DSP would:
// the bias row is decoded per output channel, so a programming error in the
// row shows up as a wrong cell expectation instead of being papered over.
static uint16_t tile_convert(const struct tile *t, unsigned s, unsigned o) {
  const uint32_t lo = bias_row_lo(t->bias, o);
  const uint16_t scale_bits = (uint16_t)(lo & 0xFFFFu);
  const uint16_t obias_bits = (uint16_t)(lo >> 16);
  const unsigned shape = bias_row_shape(t->bias, o);
  const double scale = fp16_to_double(scale_bits);
  const double obias = bias20_decode_out_bias(obias_bits, 0u);
  return prm_convert_rn16(tile_acc(t, s, o), 0.0, shape, scale, obias);
}

// Directed exports used only to pin the *alternative* rounding predictions.
static uint16_t fp16_toward_plus_inf(uint16_t b) {
  if (b & 0x8000u) return b == 0x8000u ? 0x0000u : (uint16_t)(b - 1u);
  return (uint16_t)(b + 1u);
}
static uint16_t fp16_toward_minus_inf(uint16_t b) {
  if (b & 0x8000u) return (uint16_t)(b + 1u);
  return b == 0x0000u ? 0x8000u : (uint16_t)(b - 1u);
}
static uint16_t quantize_up(double v) {
  const uint16_t rn = double_to_fp16(v);
  return (fp16_to_double(rn) < v) ? fp16_toward_plus_inf(rn) : rn;
}
static uint16_t quantize_down(double v) {
  const uint16_t rn = double_to_fp16(v);
  return (fp16_to_double(rn) > v) ? fp16_toward_minus_inf(rn) : rn;
}

// ---------------------------------------------------------------------------
// 7. Rs encoding (PRM Table 6, p.15)
// ---------------------------------------------------------------------------
#define RS_ACC_RETAIN 0x0001u
#define RS_RND_EXTRA4 0x0100u
#define RS_FB_MIN 0x0000u
#define RS_FB_MAX 0x0010u
#define RS_FB_DEST_NONE 0x0000u
#define RS_FB_DEST_OBIAS 0x0004u
#define RS_FB_DEST_SCALE 0x0008u
#define RS_BIAS_SEL(n) (((uint32_t)(n) & 3u) << 12)

int main(void) {
  // -------------------------------------------------------------------------
  // 1. Previous session's PRM cases (unchanged expectations).
  // -------------------------------------------------------------------------
  const int q = 6;
  const double a = 1.0 + ldexp(1.0, -q);
  const double x = a * a;
  printf("fixture q=%d a=%.17g acc=x=%.17g\n", q, a, x);

  struct {
    const char *name;
    uint16_t scale_bits;
    uint16_t obias_bits;
    unsigned shape;
    const char *note;
  } cases[] = {
      {"mode12_neg_scale", 0xd400u, 0x0000u, 0u, "scale=-64 shape=x bias=0"},
      {"mode13_neg_bias", 0x0000u, 0xc400u, 0u, "scale=0 shape=x bias=-4"},
      {"mode16_neg_scale_pos_bias", 0xd400u, 0x5800u, 0u,
       "scale=-64 shape=x bias=+128"},
      {"mode17_pos_scale_neg_shape", 0x5400u, 0x5800u, 4u,
       "scale=+64 shape=-x bias=+128"},
      {"mode43_neg_acc", 0x5400u, 0x0000u, 0u, "scale=+64 shape=x bias=0, acc=-x"},
  };
  for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    const double scale = fp16_to_double(cases[c].scale_bits);
    const double obias = bias20_decode_out_bias(cases[c].obias_bits, 0u);
    const double acc = (strcmp(cases[c].name, "mode43_neg_acc") == 0) ? -x : x;
    const uint16_t prm_bits =
        prm_convert_rn16(acc, 0.0, cases[c].shape, scale, obias);
    const uint16_t clamped = empirical_negative_clamp(prm_bits);
    printf("%-24s acc=%+.17g scale=%+.17g shape=%u obias=%+.17g "
           "prm_derived_bits=0x%04x prm_derived_value=%+.17g "
           "with_empirical_clamp=0x%04x note=(%s)\n",
           cases[c].name, acc, scale, cases[c].shape, obias, (unsigned)prm_bits,
           fp16_to_double(prm_bits), (unsigned)clamped, cases[c].note);
  }

  // Pinned expectations: PRM-derived math vs device-observed regression bits.
  // q=6 single-dot fixture: a=1+2^-6, x=a*a=1.031494140625,
  // 64*x=66.015625, -64*x+128=64*x_neg+128=61.984375 (exact tie
  // between 0x53BF and 0x53C0, RN-even -> 0x53C0).
  // mode16: -64*x+128 = +61.984375 -> 0x53C0 mathematically (POSITIVE);
  //   device stored 0x0000. The empirical signbit clamp does NOT explain
  //   mode16 because the PRM value is already positive -- this is the key
  //   discriminator: the negative SCALE path itself fails, not the output.
  check_bits("mode16.prm_derived", prm_convert_rn16(x, 0.0, 0u, -64.0, 128.0),
             0x53c0u);
  check_bits("mode16.empirical_clamp_noop",
             empirical_negative_clamp(
                 prm_convert_rn16(x, 0.0, 0u, -64.0, 128.0)),
             0x53c0u);  // clamp keeps it: mode16 needs another cause
  // mode17: 64*(-x)+128 = +61.984375 -> 0x53c0; device stored 0x53c0.
  // Same PRM math as mode16, different device result.
  check_bits("mode17.prm_derived", prm_convert_rn16(x, 0.0, 4u, 64.0, 128.0),
             0x53c0u);
  // mode12: -64*x -> negative; device stored 0x0000.
  {
    const uint16_t pre = prm_convert_rn16(x, 0.0, 0u, -64.0, 0.0);
    printf("mode12 prm pre=0x%04x value=%+.17g sign=%s\n", (unsigned)pre,
           fp16_to_double(pre), (pre & 0x8000u) ? "neg" : "nonneg");
    check_bits("mode12.prm_sign_negative", (uint16_t)((pre & 0x8000u) ? 1 : 0),
               1u);
    check_bits("mode12.empirical_clamp", empirical_negative_clamp(pre),
               0x0000u);
  }
  // mode13: 0*x + (-4) = -4 -> 0xc400 mathematically; device stored 0x0000.
  check_bits("mode13.prm_derived", prm_convert_rn16(x, 0.0, 0u, 0.0, -4.0),
             0xc400u);
  check_bits("mode13.empirical_clamp",
             empirical_negative_clamp(
                 prm_convert_rn16(x, 0.0, 0u, 0.0, -4.0)),
             0x0000u);
  // mode43 slot1: 64*(-x) = -66.015625 -> 0xd420 mathematically (the
  // slot-1 cell is the negated s0 anchor, NOT -64.0); device stored 0x0000.
  check_bits("mode43.prm_derived", prm_convert_rn16(-x, 0.0, 0u, 64.0, 0.0),
             0xd420u);
  check_bits("mode43.empirical_clamp",
             empirical_negative_clamp(
                 prm_convert_rn16(-x, 0.0, 0u, 64.0, 0.0)),
             0x0000u);
  // Positive controls: mode14/15/17 positive paths are unaffected by clamp.
  check_bits("mode14_pos_control", prm_convert_rn16(x, 0.0, 0u, 64.0, 0.0),
             0x5420u);
  check_bits("mode17_pos_result", prm_convert_rn16(x, 0.0, 4u, 64.0, 128.0),
             0x53c0u);

  // Special-value sanity (scope: sign handling around zero/subnormal/NaN/Inf).
  printf("specials: fp16(-0)=0x%04x fp16(min_subnormal)=0x%04x "
         "fp16(+inf)=0x%04x fp16(-inf)=0x%04x fp16(nan)=0x%04x\n",
         (unsigned)double_to_fp16(-0.0),
         (unsigned)double_to_fp16(ldexp(1.0, -24)),
         (unsigned)double_to_fp16(INFINITY),
         (unsigned)double_to_fp16(-INFINITY),
         (unsigned)double_to_fp16(NAN));
  check_bits("special.neg_zero", double_to_fp16(-0.0), 0x8000u);
  check_bits("special.neg_zero_clamp_keeps_signbit",
             (uint16_t)(empirical_negative_clamp(0x8000u) == 0x0000u ? 1 : 0),
             1u);  // -0 also has the sign bit; device law as stated is signbit-based
  check_bits("special.subnormal_pos", double_to_fp16(ldexp(1.0, -24)), 0x0001u);
  check_bits("special.pos_inf", double_to_fp16(INFINITY), 0x7c00u);
  check_bits("special.neg_inf", double_to_fp16(-INFINITY), 0xfc00u);

  // -------------------------------------------------------------------------
  // 2. Fixture exactness (no rounding anywhere before the export).
  // -------------------------------------------------------------------------
  printf("E3D fixture: a=0x%04x 2a=0x%04x x=%.17g 64x=%.17g 128x=%.17g\n",
         (unsigned)double_to_fp16(a), (unsigned)double_to_fp16(2.0 * a), x,
         64.0 * x, 128.0 * x);
  check_bits("e3d.a_bits", double_to_fp16(a), 0x3c10u);
  check_true("e3d.a_exact", a == 1.015625);
  check_true("e3d.x_exact", x == 4225.0 / 4096.0);
  check_true("e3d.x_double_exact", x == 1.031494140625);
  check_bits("e3d.two_a_bits", double_to_fp16(2.0 * a), 0x4010u);
  check_true("e3d.64x_exact", 64.0 * x == 66.015625);
  check_true("e3d.128x_exact", 128.0 * x == 132.03125);
  // Both scaled values are 0.25 ULP above an FP16 grid point.
  check_true("e3d.64x_quarter_ulp", 66.015625 - 66.0 == ldexp(1.0, -4) / 4.0);
  check_true("e3d.128x_quarter_ulp",
             132.03125 - 132.0 == ldexp(1.0, -3) / 4.0);

  // -------------------------------------------------------------------------
  // 3. Static VTCM layout pins + bijection.
  // -------------------------------------------------------------------------
  check_u32("layout.act(0,0)", act_u16(0, 0), 0u);
  check_u32("layout.act(1,0)", act_u16(1, 0), 1u);
  check_u32("layout.act(0,1)", act_u16(0, 1), 2u);
  check_u32("layout.act(1,1)", act_u16(1, 1), 3u);
  check_u32("layout.wgt(o0,i0)", wgt_u16(0, 0), 0u);
  check_u32("layout.wgt(o1,i0)", wgt_u16(1, 0), 2u);
  check_u32("layout.wgt(o0,i1)", wgt_u16(0, 1), 1u);
  check_u32("layout.store(s0,o0)", store_u16(0, 0), 0u);
  check_u32("layout.store(s1,o0)", store_u16(1, 0), 1u);
  check_u32("layout.store(s0,o1)", store_u16(0, 1), 2u);
  check_u32("layout.store(s1,o1)", store_u16(1, 1), 3u);
  check_u32("layout.store(s0,o16)", store_u16(0, 16), 32u);
  {
    // Every one of the 1024 u16 cells is hit exactly once: a pack/decode
    // bijection, so an index error cannot masquerade as a numeric result.
    uint8_t hit[1024];
    memset(hit, 0, sizeof(hit));
    unsigned dup = 0;
    for (unsigned s = 0; s < 32; ++s)
      for (unsigned o = 0; o < 32; ++o) {
        const unsigned idx = store_u16(s, o);
        if (idx >= 1024u || hit[idx]) ++dup;
        else hit[idx] = 1;
      }
    check_u32("layout.store_dup_or_oob", dup, 0u);
    unsigned missing = 0;
    for (unsigned i = 0; i < 1024u; ++i) missing += (unsigned)(hit[i] == 0);
    check_u32("layout.store_missing", missing, 0u);
  }
  {
    uint8_t hit[1024];
    memset(hit, 0, sizeof(hit));
    unsigned dup = 0;
    for (unsigned s = 0; s < 32; ++s)
      for (unsigned i = 0; i < 32; ++i) {
        const unsigned idx = act_u16(s, i);
        if (idx >= 1024u || hit[idx]) ++dup;
        else hit[idx] = 1;
      }
    check_u32("layout.act_dup_or_oob", dup, 0u);
    memset(hit, 0, sizeof(hit));
    dup = 0;
    for (unsigned o = 0; o < 32; ++o)
      for (unsigned i = 0; i < 32; ++i) {
        const unsigned idx = wgt_u16(o, i);
        if (idx >= 1024u || hit[idx]) ++dup;
        else hit[idx] = 1;
      }
    check_u32("layout.wgt_dup_or_oob", dup, 0u);
  }

  // -------------------------------------------------------------------------
  // 4. Per-output-channel bias cells (mixed-shape tile).
  // -------------------------------------------------------------------------
  struct tile ta, tb;
  tile_build(&ta, 0x3C10u, 4u);  // variant A: W(o1) = +a,  shape 4
  tile_build(&tb, 0x4010u, 4u);  // variant B: W(o1) = +2a, shape 4
  {
    check_u32("biasrow.size", (uint32_t)sizeof(ta.bias), 256u);
    check_u32("bias.cell0.lo", bias_row_lo(ta.bias, 0), 0x00005400u);
    check_u32("bias.cell0.hi", bias_row_hi(ta.bias, 0), 0x00000000u);
    check_u32("bias.cell1.lo", bias_row_lo(ta.bias, 1), 0x00005400u);
    check_u32("bias.cell1.hi", bias_row_hi(ta.bias, 1), 0x00000400u);
    check_u32("bias.cell1.shape", bias_row_shape(ta.bias, 1), 4u);
    check_u32("bias.cell0.shape", bias_row_shape(ta.bias, 0), 0u);
    // The two cells of one 256-byte set differ, which no set-wide encoder
    // (init_bias20's shape argument) can express: per-channel programming
    // is required, and this test pins the byte offsets that carry it.
    check_true("bias.per_channel_distinct",
               bias_row_hi(ta.bias, 0) != bias_row_hi(ta.bias, 1));
    unsigned same = 0;
    for (unsigned o = 2; o < 32; ++o)
      same += (unsigned)(bias_row_hi(ta.bias, o) == bias_row_hi(ta.bias, 0));
    check_u32("bias.default_cells_identical", same, 30u);
    // Decoded values round-trip through the shared bias20 helpers.
    const uint32_t lo1 = bias_row_lo(ta.bias, 1);
    check_true("bias.cell1.scale_is_64",
               fp16_to_double((uint16_t)(lo1 & 0xFFFFu)) == 64.0);
    check_true("bias.cell1.obias_is_zero",
               bias20_decode_out_bias((uint16_t)(lo1 >> 16), 0u) == 0.0);
  }

  // -------------------------------------------------------------------------
  // 5. Four discriminator cells, layout-driven (variant A and B).
  // -------------------------------------------------------------------------
  printf("E3D variant A: acc(s0,o0)=%.17g (s1,o0)=%.17g (s0,o1)=%.17g "
         "(s1,o1)=%.17g\n",
         tile_acc(&ta, 0, 0), tile_acc(&ta, 1, 0), tile_acc(&ta, 0, 1),
         tile_acc(&ta, 1, 1));
  printf("E3D variant B: acc(s0,o0)=%.17g (s1,o0)=%.17g (s0,o1)=%.17g "
         "(s1,o1)=%.17g\n",
         tile_acc(&tb, 0, 0), tile_acc(&tb, 1, 0), tile_acc(&tb, 0, 1),
         tile_acc(&tb, 1, 1));
  check_true("e3dA.acc00", tile_acc(&ta, 0, 0) == x);
  check_true("e3dA.acc10", tile_acc(&ta, 1, 0) == -x);
  check_true("e3dA.acc01", tile_acc(&ta, 0, 1) == x);
  check_true("e3dA.acc11", tile_acc(&ta, 1, 1) == -x);
  check_true("e3dB.acc00", tile_acc(&tb, 0, 0) == x);
  check_true("e3dB.acc10", tile_acc(&tb, 1, 0) == -x);
  check_true("e3dB.acc01", tile_acc(&tb, 0, 1) == 2.0 * x);
  check_true("e3dB.acc11", tile_acc(&tb, 1, 1) == -2.0 * x);

  const uint16_t ea[4] = {tile_convert(&ta, 0, 0), tile_convert(&ta, 1, 0),
                          tile_convert(&ta, 0, 1), tile_convert(&ta, 1, 1)};
  const uint16_t eb[4] = {tile_convert(&tb, 0, 0), tile_convert(&tb, 1, 0),
                          tile_convert(&tb, 0, 1), tile_convert(&tb, 1, 1)};
  printf("E3D variant A expected bits: 0x%04x 0x%04x 0x%04x 0x%04x\n",
         (unsigned)ea[0], (unsigned)ea[1], (unsigned)ea[2], (unsigned)ea[3]);
  printf("E3D variant B expected bits: 0x%04x 0x%04x 0x%04x 0x%04x\n",
         (unsigned)eb[0], (unsigned)eb[1], (unsigned)eb[2], (unsigned)eb[3]);
  printf("E3D variant B values: %.17g %.17g %.17g %.17g\n",
         fp16_to_double(eb[0]), fp16_to_double(eb[1]), fp16_to_double(eb[2]),
         fp16_to_double(eb[3]));

  check_bits("e3dA.out0", ea[0], 0x5420u);
  check_bits("e3dA.out1", ea[1], 0xd420u);
  check_bits("e3dA.out2", ea[2], 0xd420u);
  check_bits("e3dA.out3", ea[3], 0x5420u);
  check_bits("e3dB.out0", eb[0], 0x5420u);
  check_bits("e3dB.out1", eb[1], 0xd420u);
  check_bits("e3dB.out2", eb[2], 0xd820u);
  check_bits("e3dB.out3", eb[3], 0x5820u);

  // Variant A is symmetric under a (slot1 <-> slot2) readback transposition:
  // exactly the layout gap the diagnosis doc flags ("indexes 1, 2 ... are
  // untested on device"). Variant B removes it.
  check_true("e3dA.transpose_degenerate", ea[1] == ea[2]);
  check_true("e3dB.transpose_distinguishable", eb[1] != eb[2]);
  check_true("e3dB.all_four_distinct",
             eb[0] != eb[1] && eb[0] != eb[2] && eb[0] != eb[3] &&
                 eb[1] != eb[2] && eb[1] != eb[3] && eb[2] != eb[3]);

  // -------------------------------------------------------------------------
  // 6. Rounding-mode and cause-model patterns.
  // -------------------------------------------------------------------------
  check_bits("round.64x.rn", double_to_fp16(66.015625), 0x5420u);
  check_bits("round.64x.ceil", quantize_up(66.015625), 0x5421u);
  check_bits("round.64x.floor", quantize_down(66.015625), 0x5420u);
  check_bits("round.128x.rn", double_to_fp16(132.03125), 0x5820u);
  check_bits("round.128x.ceil", quantize_up(132.03125), 0x5821u);
  check_bits("round.m64x.rn", double_to_fp16(-66.015625), 0xd420u);
  check_bits("round.m64x.floor", quantize_down(-66.015625), 0xd421u);
  check_bits("round.m64x.ceil", quantize_up(-66.015625), 0xd420u);
  check_bits("round.m128x.floor", quantize_down(-132.03125), 0xd821u);

  // Pre-registered outcome patterns for variant B (RN cells otherwise).
  const uint16_t m3_signed[4] = {0x5420u, 0xd420u, 0xd820u, 0x5820u};
  const uint16_t m1_late_clamp[4] = {0x5420u, 0x0000u, 0x0000u, 0x5820u};
  const uint16_t m2_early_clamp[4] = {0x5420u, 0x0000u, 0x0000u, 0x0000u};
  const uint16_t m5_shape_ignored[4] = {0x5420u, 0x0000u, 0x5820u, 0x0000u};
  const uint16_t m5b_shape_ignored_signed[4] = {0x5420u, 0xd420u, 0x5820u,
                                                0xd820u};
  const uint16_t m_ceiling[4] = {0x5421u, 0xd420u, 0xd820u, 0x5821u};
  const uint16_t m_floor[4] = {0x5420u, 0xd421u, 0xd821u, 0x5820u};
  // Mixed model: a signed store that also clamps a negative ACC *before* the
  // shaping function. out[1] = shape0(clamp(-x)) = +0; out[2] = shape4(+2x)
  // = -132 (proves the store keeps negatives); out[3] = shape4(clamp(-2x))
  // = -0, which is only reachable when the store does NOT clamp the sign bit.
  const uint16_t m_accclamp_signed[4] = {0x5420u, 0x0000u, 0xd820u, 0x8000u};
  // Store takes the magnitude (sign dropped, not clamped to zero).
  const uint16_t m_abs_store[4] = {0x5420u, 0x5420u, 0x5820u, 0x5820u};
  // Negative results saturate to the most-negative finite FP16 instead of +0.
  const uint16_t m_neg_maxnorm[4] = {0x5420u, 0xfbffu, 0xfbffu, 0x5820u};
  printf("E3D patterns: signed=0x%04x,0x%04x,0x%04x,0x%04x "
         "late=0x%04x,0x%04x,0x%04x,0x%04x "
         "early=0x%04x,0x%04x,0x%04x,0x%04x "
         "shapeign=0x%04x,0x%04x,0x%04x,0x%04x\n",
         (unsigned)m3_signed[0], (unsigned)m3_signed[1],
         (unsigned)m3_signed[2], (unsigned)m3_signed[3],
         (unsigned)m1_late_clamp[0], (unsigned)m1_late_clamp[1],
         (unsigned)m1_late_clamp[2], (unsigned)m1_late_clamp[3],
         (unsigned)m2_early_clamp[0], (unsigned)m2_early_clamp[1],
         (unsigned)m2_early_clamp[2], (unsigned)m2_early_clamp[3],
         (unsigned)m5_shape_ignored[0], (unsigned)m5_shape_ignored[1],
         (unsigned)m5_shape_ignored[2], (unsigned)m5_shape_ignored[3]);
  printf("E3D mixed patterns: accclamp_signed=0x%04x,0x%04x,0x%04x,0x%04x "
         "abs=0x%04x,0x%04x,0x%04x,0x%04x "
         "maxnorm=0x%04x,0x%04x,0x%04x,0x%04x "
         "ceiling=0x%04x,0x%04x,0x%04x,0x%04x "
         "floor=0x%04x,0x%04x,0x%04x,0x%04x\n",
         (unsigned)m_accclamp_signed[0], (unsigned)m_accclamp_signed[1],
         (unsigned)m_accclamp_signed[2], (unsigned)m_accclamp_signed[3],
         (unsigned)m_abs_store[0], (unsigned)m_abs_store[1],
         (unsigned)m_abs_store[2], (unsigned)m_abs_store[3],
         (unsigned)m_neg_maxnorm[0], (unsigned)m_neg_maxnorm[1],
         (unsigned)m_neg_maxnorm[2], (unsigned)m_neg_maxnorm[3],
         (unsigned)m_ceiling[0], (unsigned)m_ceiling[1], (unsigned)m_ceiling[2],
         (unsigned)m_ceiling[3], (unsigned)m_floor[0], (unsigned)m_floor[1],
         (unsigned)m_floor[2], (unsigned)m_floor[3]);

  // Each pre-registered cause must be separable from every other by at least
  // one of the four read cells; otherwise the trial cannot decide and must be
  // reported as INCONCLUSIVE rather than interpreted.
  {
    const uint16_t *models[] = {
        m3_signed,        m1_late_clamp,    m2_early_clamp,
        m5_shape_ignored, m5b_shape_ignored_signed, m_ceiling,
        m_floor,          m_accclamp_signed, m_abs_store,
        m_neg_maxnorm};
    const char *names[] = {"signed",       "late_clamp", "early_clamp",
                           "shape_ignored", "shape_ignored_signed",
                           "ceiling",      "floor",      "accclamp_signed",
                           "abs_store",    "neg_maxnorm"};
    unsigned collisions = 0;
    for (unsigned i = 0; i < 10u; ++i)
      for (unsigned j = i + 1u; j < 10u; ++j) {
        int differs = 0;
        for (unsigned k = 0; k < 4u; ++k) differs |= models[i][k] != models[j][k];
        if (!differs) {
          ++collisions;
          printf("COLLISION %s == %s\n", names[i], names[j]);
        }
      }
    check_u32("e3dB.model_collisions", collisions, 0u);
  }

  // Variant A is the authorised execution variant (user decision 2026-10-24).
  // The same ten cause models must stay pairwise distinct on A's cells as
  // well. A is transposition-degenerate at slots 1/2, so this check is weaker
  // than B's by construction; that limitation is recorded, not patched away.
  {
    const uint16_t a_signed[4] = {0x5420u, 0xd420u, 0xd420u, 0x5420u};
    const uint16_t a_late_clamp[4] = {0x5420u, 0x0000u, 0x0000u, 0x5420u};
    const uint16_t a_early_clamp[4] = {0x5420u, 0x0000u, 0x0000u, 0x0000u};
    const uint16_t a_shape_ignored[4] = {0x5420u, 0x0000u, 0x5420u, 0x0000u};
    const uint16_t a_shape_ignored_signed[4] = {0x5420u, 0xd420u, 0x5420u,
                                                0xd420u};
    const uint16_t a_accclamp_signed[4] = {0x5420u, 0x0000u, 0xd420u, 0x8000u};
    const uint16_t a_abs_store[4] = {0x5420u, 0x5420u, 0x5420u, 0x5420u};
    const uint16_t a_neg_maxnorm[4] = {0x5420u, 0xfbffu, 0xfbffu, 0x5420u};
    const uint16_t a_ceiling[4] = {0x5421u, 0xd420u, 0xd420u, 0x5421u};
    const uint16_t a_floor[4] = {0x5420u, 0xd421u, 0xd421u, 0x5420u};
    const uint16_t *amodels[] = {
        a_signed,        a_late_clamp,    a_early_clamp,
        a_shape_ignored, a_shape_ignored_signed, a_ceiling,
        a_floor,         a_accclamp_signed, a_abs_store,
        a_neg_maxnorm};
    const char *anames[] = {"signed",       "late_clamp", "early_clamp",
                            "shape_ignored", "shape_ignored_signed",
                            "ceiling",      "floor",      "accclamp_signed",
                            "abs_store",    "neg_maxnorm"};
    unsigned acollisions = 0;
    for (unsigned i = 0; i < 10u; ++i)
      for (unsigned j = i + 1u; j < 10u; ++j) {
        int differs = 0;
        for (unsigned k = 0; k < 4u; ++k)
          differs |= amodels[i][k] != amodels[j][k];
        if (!differs) {
          ++acollisions;
          printf("COLLISION_A %s == %s\n", anames[i], anames[j]);
        }
      }
    check_u32("e3dA.model_collisions", acollisions, 0u);
    // The authorised tuple must be the one the converter model produces.
    check_true("e3dA.authored_matches_model",
               a_signed[0] == ea[0] && a_signed[1] == ea[1] &&
                   a_signed[2] == ea[2] && a_signed[3] == ea[3]);
    check_true("e3dA.late_clamp_keeps_positive_anchor",
               a_late_clamp[0] == ea[0] && a_late_clamp[3] == ea[3]);
  }

  // -------------------------------------------------------------------------
  // 7. Rs encoding (PRM Table 6).
  // -------------------------------------------------------------------------
  check_u32("rs.no_feedback_baseline", RS_BIAS_SEL(0), 0x0000u);
  check_u32("rs.retain", RS_ACC_RETAIN, 0x0001u);
  check_u32("rs.set1_retain", RS_BIAS_SEL(1) | RS_ACC_RETAIN, 0x1001u);
  check_u32("rs.set1_clear", RS_BIAS_SEL(1), 0x1000u);
  check_u32("rs.rnd_bit", RS_RND_EXTRA4, 0x0100u);
  check_u32("rs.fb_dest_oobias", RS_FB_DEST_OBIAS, 0x0004u);
  check_u32("rs.fb_dest_scale", RS_FB_DEST_SCALE, 0x0008u);
  check_u32("rs.fb_op_max", RS_FB_MAX, 0x0010u);
  check_u32("rs.fb_none", RS_FB_DEST_NONE, 0x0000u);
  // Rs[0] is ACC retention, not feedback: the PRM "no feedback" recipe is
  // bias_sel<<12 with every other bit 0 (Rs[0]=0), while the existing probe
  // modes use Rs=1 (retain). Both are no-feedback converts, so no arithmetic
  // term of the convert equation can differ; only the post-convert ACC state.
  check_true("rs.baseline_differs_from_probe_only_in_acc",
             (RS_BIAS_SEL(0) ^ RS_ACC_RETAIN) == RS_ACC_RETAIN);
  // The two Rs settings may not be differenced numerically here: Rs is not an
  // operand of the convert equation (Fig.11), so the model has exactly one
  // value for both. The trial's job is to falsify that reading, not to
  // predict a difference.

  // -------------------------------------------------------------------------
  // 8. Two-convert split reconstruction: y = max(y,0) - max(-y,0).
  //
  // HMX realization: shape 2 = max(x,0) for the positive part and
  // shape 5 = -min(x,0) = max(-x,0) for the negative part. Both exports are
  // non-negative, so neither is affected by the measured negative clamp.
  // shape 6 = -max(x,0) is the trap: it is <= 0 and would be clamped away.
  // -------------------------------------------------------------------------
  {
    long both_nonzero = 0, mismatches = 0, cases_n = 0;
    for (int e = -8; e <= 8; ++e) {
      for (int k = 0; k < 64; ++k) {
        const double mant = 1.0 + (double)k / 64.0;
        for (int sign = -1; sign <= 1; sign += 2) {
          const double y = sign * ldexp(mant, e);
          const double scaled = 64.0 * y;
          const uint16_t p = double_to_fp16(scaled > 0.0 ? scaled : 0.0);
          const uint16_t n = double_to_fp16((-scaled) > 0.0 ? -scaled : 0.0);
          const double recon = fp16_to_double(p) - fp16_to_double(n);
          const double direct = fp16_to_double(double_to_fp16(scaled));
          if (fp16_to_double(p) != 0.0 && fp16_to_double(n) != 0.0)
            ++both_nonzero;
          if (recon != direct) ++mismatches;
          ++cases_n;
        }
      }
    }
    printf("split sweep: cases=%ld both_nonzero=%ld mismatches=%ld\n", cases_n,
           both_nonzero, mismatches);
    check_true("split.sweep_nonempty", cases_n > 0);
    check_u32("split.both_nonzero", (uint32_t)both_nonzero, 0u);
    check_u32("split.mismatches", (uint32_t)mismatches, 0u);
    // The split equals a signed RN16 store bit-for-bit: it recovers the sign,
    // never extra precision. Pinned on the E-3D magnitudes (-66.015625:
    // P=0, N=RN16(66.015625)=0x5420, P-N=-66.0 -> 0xD420).
    {
      const uint16_t p = double_to_fp16(0.0);
      const uint16_t n = double_to_fp16(66.015625);
      check_bits("split.part_negative_is_zero", p, 0x0000u);
      check_bits("split.part_magnitude", n, 0x5420u);
      check_bits("split.reconstruct_out1",
                 double_to_fp16(fp16_to_double(p) - fp16_to_double(n)),
                 0xd420u);
    }
    check_bits("split.neg_part_via_shape5",
               prm_convert_rn16(-x, 0.0, 5u, 64.0, 0.0), 0x5420u);
    check_bits("split.neg_part_via_shape6_is_zero",
               prm_convert_rn16(-x, 0.0, 6u, 64.0, 0.0), 0x0000u);
    check_bits("split.pos_part_via_shape2",
               prm_convert_rn16(x, 0.0, 2u, 64.0, 0.0), 0x5420u);
    check_bits("split.shape5_on_positive_acc_is_zero",
               prm_convert_rn16(x, 0.0, 5u, 64.0, 0.0), 0x0000u);
  }

  // -------------------------------------------------------------------------
  // 9. mode 16/17 regression: PRM math vs the recorded device values.
  //    Device observations are quoted from the commit bodies (413d872,
  //    d4d9935): mode 16 -> 0x0000, mode 17 -> 0x53c0. The PRM derives
  //    0x53c0 for BOTH, so mode 16 is a PRM contradiction that the
  //    negative-value clamp cannot explain (its PRM value is positive).
  // -------------------------------------------------------------------------
  check_bits("mode16.device_recorded", 0x0000u, 0x0000u);
  check_bits("mode17.device_recorded", 0x53c0u, 0x53c0u);
  check_true("mode16_prm_positive_but_device_zero",
             (prm_convert_rn16(x, 0.0, 0u, -64.0, 128.0) & 0x8000u) == 0u);

  printf("hmx_signed_output_oracle checks=%ld fail=%d\n", checks, failures);
  if (failures) {
    printf("hmx_signed_output_oracle=FAIL\n");
    return 1;
  }
  printf("hmx_signed_output_oracle=PASS\n");
  return 0;
}
