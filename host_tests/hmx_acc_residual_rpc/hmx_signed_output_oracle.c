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
//  10. mode-44 (E-3D) host contract classifier: every registered pattern,
//      the precondition, the single-cell fallback, the unmatched case and
//      the one-trial contract.
//  11. Sign-separated reconstruction with the E-3D shapes:
//      y = store(shape 0(y)) - store(shape 4(y)). Sweep, failure modes and
//      the clamp-law validity guard. CPU-only; no device behavior beyond
//      the E-3D four slots is assumed.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bias20.h"
#include "fp16_util.h"
#include "hmx_mode44.h"

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

// E-3D sign-separated store model. The same ACC is converted twice: shape 0
// exports it and shape 4 exports its negation. Under the measured clamp law
// exactly one of the two words survives when bo = 0, so the FP16 subtraction
// is exact and no second rounding enters the result. Returns the
// reconstructed bits; *s0 / *s4 receive the two stored words.
static uint16_t split04_store(double acc, double bi, double scale,
                              double out_bias, uint16_t *s0, uint16_t *s4) {
  *s0 = empirical_negative_clamp(prm_convert_rn16(acc, bi, 0u, scale, out_bias));
  *s4 = empirical_negative_clamp(prm_convert_rn16(acc, bi, 4u, scale, out_bias));
  return double_to_fp16(fp16_to_double(*s0) - fp16_to_double(*s4));
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

// ---------------------------------------------------------------------------
// 12. R2 state machine: one MMA, two converts over a retained ACC.
//
// PRM basis (80-N2040-62 Rev AA):
//   * 2.2.3 p.9-10: a pair of 32x32x37-bit accumulators; a multiply writes the
//     primary; a following convert may optionally clear and swap it; the
//     convert instructions are the only way to access the accumulators.
//   * 2.3 p.11-12: the convert reads every ACC cell, applies the bias-row
//     equation and writes the convert state; it then optionally clears the ACC.
//   * 2.3.1 p.13: four bias sets of 32 per-channel 64-bit cells; a convert
//     selects one set; the bias load can precede the multiply or sit between
//     the multiply and the convert.
//   * Fig.17/Table 6 p.15: Rs[13:12] = bias set; Rs[0]=1 retains the ACC,
//     Rs[0]=0 clears and swaps; no-feedback converts leave all other bits 0.
//   * 2.4 p.17: the store reads the convert state and does not modify it.
//
// The model is deliberately stateful (ACC bank/valid, convert state, previous
// convert state, bias sets, two output regions) so that a broken ACC, a missed
// bias-set switch, an unexpectedly enabled feedback path or overlapping stores
// cannot silently produce the expected bits. It models the PRM equation plus
// the measured negative-value clamp as a separate, labelled law.
//
// CPU arithmetic assumptions: IEEE binary16 RN-even helpers from fp16_util.h,
// double intermediates, no FMA contraction (the runner passes
// -ffp-contract=off), no FTZ/DAZ. HVX subtraction is modelled as IEEE FP16
// RN-even; the separate HVX caveat is documented, not hidden here.
// ---------------------------------------------------------------------------

#define R2_NS 32
#define R2_NO 32
#define R2_CELLS (R2_NS * R2_NO)
#define R2_BIAS_SETS 4
#define R2_REGIONS 2

typedef struct {
  uint16_t scale_bits;
  uint16_t obias_bits;
  unsigned shape;
} r2_bias_cell;

typedef struct {
  double scale;
  double obias;
  unsigned shape;
} r2_bias_value;

typedef struct {
  double acc[2][R2_CELLS]; /* two banks, primary/secondary */
  int primary;
  int acc_valid;

  double cvt[R2_CELLS];        /* current convert state, pre-RN16 unless
                                * clamp_in_convert is set */
  uint16_t cvt_bits[R2_CELLS]; /* current convert state, RN16 bits */
  double cvt_prev[R2_CELLS];   /* previous convert state for feedback */
  uint16_t cvt_prev_bits[R2_CELLS];
  int cvt_valid;
  int cvt_prev_valid;

  r2_bias_cell bias[R2_BIAS_SETS][R2_NO];
  int bias_loaded[R2_BIAS_SETS];

  uint16_t mem[R2_REGIONS][R2_CELLS];
  int mem_written[R2_REGIONS];
  long store_count;

  double acc_read[2][R2_CELLS]; /* snapshot per convert for the reuse check */
  int acc_read_count;

  long mma_count;
  long convert_count;
  int last_bias_sel;
  int last_fb_dest;
  int last_retain;

  /* Fault-injection switches. All default to the registered R2 behaviour. */
  int clamp_in_convert;     /* 0 = clamp at store (E-3D leaves the stage open) */
  int rounding_mode;        /* 0 = RN-even, 1 = toward +inf, 2 = toward -inf */
  int force_bias_sel;       /* -1 = use Rs[13:12], else override lookup */
  int force_feedback_dest;  /* -1 = use Rs[3:2], else override */
  int force_feedback_op;    /* used with force_feedback_dest */
  int bias_missing;
  int store_oob;
} r2_state;

static int r2_store_index(unsigned s, unsigned o) {
  return (int)((s >> 1) * 64u + o * 2u + (s & 1u));
}

static double r2_shape_value(double x, unsigned shape) {
  switch (shape & 7u) {
    case 0: return x;
    case 1: return x < 0.0 ? x : 0.0;
    case 2: return x > 0.0 ? x : 0.0;
    case 3: return fabs(x);
    case 4: return -x;
    case 5: return -(x < 0.0 ? x : 0.0);
    case 6: return -(x > 0.0 ? x : 0.0);
    case 7: return -fabs(x);
    default: return x;
  }
}

static void r2_bias_fill(r2_state *st, unsigned set, uint16_t scale_bits,
                         unsigned shape, uint16_t obias_bits) {
  for (unsigned o = 0; o < R2_NO; ++o) {
    st->bias[set][o].scale_bits = scale_bits;
    st->bias[set][o].obias_bits = obias_bits;
    st->bias[set][o].shape = shape;
  }
  st->bias_loaded[set] = 1;
}

static r2_bias_value r2_bias_decode(const r2_bias_cell *cell) {
  r2_bias_value v;
  v.scale = fp16_to_double(cell->scale_bits);
  v.obias = bias20_decode_out_bias(cell->obias_bits, 0u);
  v.shape = cell->shape;
  return v;
}

static void r2_init(r2_state *st) {
  memset(st, 0, sizeof(*st));
  st->primary = 0;
  st->acc_valid = 0;
  st->cvt_valid = 0;
  st->cvt_prev_valid = 0;
  st->acc_read_count = 0;
  st->clamp_in_convert = 0;
  st->rounding_mode = 0;
  st->force_bias_sel = -1;
  st->force_feedback_dest = -1;
  st->force_feedback_op = 0;
  st->bias_missing = 0;
  st->store_oob = 0;
  /* Registered R2 rows. Set 0: +64, shape 0, bo 0. Set 1: +64, shape 4,
   * bo 0. Set 2 is the retention/bias-switch control row (+32, shape 0). */
  r2_bias_fill(st, 0u, 0x5400u, 0u, 0x0000u);
  r2_bias_fill(st, 1u, 0x5400u, 4u, 0x0000u);
  r2_bias_fill(st, 2u, 0x5000u, 0u, 0x0000u);
}

static void r2_mma_begin(r2_state *st) {
  for (int i = 0; i < R2_CELLS; ++i) st->acc[st->primary][i] = 0.0;
}

static void r2_mma_cell(r2_state *st, unsigned s, unsigned o, double value) {
  st->acc[st->primary][s * R2_NO + o] = value;
}

static void r2_mma_end(r2_state *st) {
  st->acc_valid = 1;
  ++st->mma_count;
}

static uint16_t r2_round_value(const r2_state *st, double raw) {
  if (st->rounding_mode == 1) return quantize_up(raw);
  if (st->rounding_mode == 2) return quantize_down(raw);
  return double_to_fp16(raw);
}

/* One convert transfer. Writes the whole 1024-cell convert state. Feedback is
 * an explicitly modelled fault-injection path; the registered R2 Rs values
 * have Rs[3:2]=0. */
static void r2_convert(r2_state *st, unsigned rs) {
  const unsigned bias_sel = (rs >> 12) & 3u;
  const int retain = (int)(rs & 1u);
  unsigned fb_dest = (rs >> 2) & 3u;
  unsigned fb_op = (rs >> 4) & 1u;
  unsigned lookup = bias_sel;

  if (st->force_bias_sel >= 0) lookup = (unsigned)st->force_bias_sel;
  if (st->force_feedback_dest >= 0) {
    fb_dest = (unsigned)st->force_feedback_dest;
    fb_op = (unsigned)st->force_feedback_op;
  }
  if (!st->bias_loaded[lookup]) st->bias_missing = 1;

  if (st->acc_read_count < 2) {
    for (int i = 0; i < R2_CELLS; ++i)
      st->acc_read[st->acc_read_count][i] =
          st->acc_valid ? st->acc[st->primary][i] : 0.0;
  }
  ++st->acc_read_count;

  for (int i = 0; i < R2_CELLS; ++i) {
    const unsigned o = (unsigned)i & (R2_NO - 1u);
    const r2_bias_value bv = r2_bias_decode(&st->bias[lookup][o]);
    const double acc = st->acc_valid ? st->acc[st->primary][i] : 0.0;
    double scale = bv.scale;
    double obias = bv.obias;

    if (fb_dest != 0u && st->cvt_prev_valid) {
      const double fb = st->cvt_prev[i];
      if (fb_dest == 1u) {
        obias = fb_op ? fmax(obias, fb) : fmin(obias, fb);
      } else if (fb_dest == 2u) {
        scale = fb_op ? fmax(scale, fb) : fmin(scale, fb);
      }
    }

    const double raw = scale * r2_shape_value(acc, bv.shape) + obias;
    uint16_t bits = r2_round_value(st, raw);
    double state_value = raw;
    if (st->clamp_in_convert) {
      bits = empirical_negative_clamp(bits);
      state_value = fp16_to_double(bits);
    }
    st->cvt[i] = state_value;
    st->cvt_bits[i] = bits;
  }

  for (int i = 0; i < R2_CELLS; ++i) {
    st->cvt_prev[i] = st->cvt[i];
    st->cvt_prev_bits[i] = st->cvt_bits[i];
  }
  st->cvt_valid = 1;
  st->cvt_prev_valid = 1;
  ++st->convert_count;
  st->last_bias_sel = (int)bias_sel;
  st->last_fb_dest = (int)fb_dest;
  st->last_retain = retain;

  if (!retain) {
    /* The PRM clears the accumulator values and swaps the primary bank. Clear
     * the bank that was read, then swap; the old secondary becomes primary and
     * is stale, so acc_valid=0 prevents a convert from using it without a new
     * MMA. */
    for (int i = 0; i < R2_CELLS; ++i) st->acc[st->primary][i] = 0.0;
    st->primary ^= 1;
    st->acc_valid = 0;
  }
}

/* The store reads the current convert state and applies the measured clamp to
 * each FP16 word. It does not modify the ACC or the convert state. */
static void r2_store(r2_state *st, unsigned region) {
  if (region >= R2_REGIONS) {
    st->store_oob = 1;
    return;
  }
  /* Write in the PRM store layout (2 spatials per vector, output channel
   * fastest), not in the logical (s,o) order. This is the layout the future
   * host readback and the HVX subtract both see. */
  for (unsigned s = 0; s < R2_NS; ++s)
    for (unsigned o = 0; o < R2_NO; ++o)
      st->mem[region][r2_store_index(s, o)] =
          empirical_negative_clamp(st->cvt_bits[s * R2_NO + o]);
  ++st->mem_written[region];
  ++st->store_count;
}

static uint16_t r2_recon_cell(const r2_state *st, unsigned pos_region,
                              unsigned neg_region, int cell) {
  return double_to_fp16(fp16_to_double(st->mem[pos_region][cell]) -
                        fp16_to_double(st->mem[neg_region][cell]));
}

static void r2_force_acc_destroy(r2_state *st) {
  for (int b = 0; b < 2; ++b)
    for (int i = 0; i < R2_CELLS; ++i) st->acc[b][i] = 0.0;
  st->acc_valid = 0;
}

/* S2 tile: spatial 0 = +x, spatial 1 = -x, spatial 2 = 0, output channel 0.
 * All other cells are exact zero. */
static void r2_build_sign_tile(r2_state *st, double pos_x, double neg_x) {
  r2_init(st);
  r2_mma_begin(st);
  r2_mma_cell(st, 0u, 0u, pos_x);
  r2_mma_cell(st, 1u, 0u, neg_x);
  r2_mma_cell(st, 2u, 0u, 0.0);
  r2_mma_end(st);
}

/* S1 tile: spatial 0 = +x only. */
static void r2_build_control_tile(r2_state *st, double x) {
  r2_init(st);
  r2_mma_begin(st);
  r2_mma_cell(st, 0u, 0u, x);
  r2_mma_end(st);
}

static void r2_run_sign_tile_variant(r2_state *st, unsigned rs2) {
  r2_convert(st, 0x0001u); /* set 0, retain */
  r2_store(st, 0u);        /* positive buffer */
  r2_convert(st, rs2);     /* set 1, registered 0x1000 or retain 0x1001 */
  r2_store(st, 1u);        /* negative buffer */
}

static void r2_run_sign_tile(r2_state *st) {
  r2_run_sign_tile_variant(st, 0x1000u);
}

static void r2_run_control_tile(r2_state *st) {
  r2_convert(st, 0x0001u); /* set 0, retain */
  r2_store(st, 0u);        /* first store: +64x */
  r2_convert(st, 0x2000u); /* set 2 (+32), clear */
  r2_store(st, 1u);        /* second store: +32x */
}

/* ---------------------------------------------------------------------------
 * R2 future-decision procedure (CPU model only).
 *
 * The labels below are pre-registered future device verdicts. This section
 * tests the procedure by simulated observation; it does NOT establish a
 * device result. The oracle prints r2_device_verdict=NOT_RUN unconditionally.
 * ------------------------------------------------------------------------- */

typedef enum {
  R2_ACC_REUSE_SUPPORTED = 0,
  R2_ACC_DESTROYED,
  R2_BIAS_SWITCH_FAILED,
  R2_STORE_INTERFERENCE,
  R2_ROUNDING_MISMATCH,
  R2_CAUSE_INCONCLUSIVE,
  R2_UNEXPECTED_OUTPUT,
  R2_HOST_CONTRACT_FAILED
} r2_verdict_t;

static const char *r2_verdict_name(r2_verdict_t verdict) {
  switch (verdict) {
    case R2_ACC_REUSE_SUPPORTED: return "R2_ACC_REUSE_SUPPORTED";
    case R2_ACC_DESTROYED: return "R2_ACC_DESTROYED";
    case R2_BIAS_SWITCH_FAILED: return "R2_BIAS_SWITCH_FAILED";
    case R2_STORE_INTERFERENCE: return "R2_STORE_INTERFERENCE";
    case R2_ROUNDING_MISMATCH: return "R2_ROUNDING_MISMATCH";
    case R2_CAUSE_INCONCLUSIVE: return "R2_CAUSE_INCONCLUSIVE";
    case R2_UNEXPECTED_OUTPUT: return "R2_UNEXPECTED_OUTPUT";
    case R2_HOST_CONTRACT_FAILED: return "R2_HOST_CONTRACT_FAILED";
  }
  return "R2_HOST_CONTRACT_FAILED";
}

typedef struct {
  /* S1: retention + bias-set-selection control. */
  uint16_t s1_first;
  uint16_t s1_second;
  /* S2: sign-separated two-convert tile. */
  uint16_t s2_pos[3];
  uint16_t s2_neg[3];
  uint16_t s2_recon[3];
  /* Host contract: two subcases, one MMA + two converts + two stores each. */
  int mma_count;
  int convert_count;
  int store_count;
  int trials;
  int hmx_executed;
  int slot_count;
  int sentinel_hits;
  int build_id_match;
  int close_ok;
  int pos_region_writes;
  int neg_region_writes;
  unsigned rs_first;
  unsigned rs_s1_second;
  unsigned rs_second;
  int bias_loads;
  unsigned bias_sel_first;
  unsigned bias_sel_second;
} r2_observation_t;

typedef struct {
  r2_verdict_t verdict;
  const char *name;
  const char *cause;
} r2_report_t;

/* device=0 is the CPU fault-injection classifier: it knows the injected cause
 * and may return R2_ACC_DESTROYED / R2_BIAS_SWITCH_FAILED. device=1 is the
 * future device classifier: a bit pattern alone cannot establish those causes,
 * so the ambiguous S1/S2 patterns return R2_CAUSE_INCONCLUSIVE instead. */
static r2_report_t r2_classify_mode(const r2_observation_t *o, int device) {
  r2_report_t r;
  r.verdict = R2_UNEXPECTED_OUTPUT;
  r.name = r2_verdict_name(r.verdict);
  r.cause = "unmatched";

  /* 1. Host contract: the registered recipe and only that recipe. A feedback
   *    bit in Rs[3:2], a third MMA, a missing convert/store, a wrong slot
   *    count, an unwritten-store sentinel, a build-id mismatch or a failed
   *    close fails closed. */
  if (o->mma_count != 2 || o->convert_count != 4 || o->store_count != 4 ||
      o->trials != 2 || o->hmx_executed != 1 || o->slot_count != 11 ||
      o->sentinel_hits != 0 || !o->build_id_match || !o->close_ok ||
      o->rs_first != 0x0001u || o->rs_s1_second != 0x2000u ||
      (o->rs_second != 0x1000u && o->rs_second != 0x1001u) ||
      o->bias_loads != 3 || o->bias_sel_first != 0u ||
      o->bias_sel_second != 1u) {
    r.verdict = R2_HOST_CONTRACT_FAILED;
    r.name = r2_verdict_name(r.verdict);
    r.cause = "counts_or_rs";
    return r;
  }

  /* 2. Both stores must have landed in their registered, distinct regions. */
  if (o->pos_region_writes != 2 || o->neg_region_writes != 2) {
    r.verdict = R2_STORE_INTERFERENCE;
    r.name = r2_verdict_name(r.verdict);
    r.cause = "region_write_count";
    return r;
  }

  /* 3. Registered one-ULP alternatives are a rounding finding, not a generic
   *    unexpected value. 0x5421 = ceil(66.015625), 0x5021 = ceil(33.0078125). */
  if (o->s1_second == 0x5021u || o->s2_pos[0] == 0x5421u ||
      o->s2_neg[1] == 0x5421u) {
    r.verdict = R2_ROUNDING_MISMATCH;
    r.name = r2_verdict_name(r.verdict);
    r.cause = "one_ulp_ceiling";
    return r;
  }

  /* 4. S1 isolates ACC retention from the shape-4 path: the second convert
   *    selects a different bias set with a different positive scale. */
  if (o->s1_first != 0x5420u) {
    r.cause = "s1_first_anchor";
    return r;
  }
  if (o->s1_second == 0x0000u) {
    r.verdict = device ? R2_CAUSE_INCONCLUSIVE : R2_ACC_DESTROYED;
    r.name = r2_verdict_name(r.verdict);
    r.cause = device ? "s1_second_zero_ambiguous_acc_or_bias_or_convert_or_store"
                     : "s1_second_zero_after_retained_convert";
    return r;
  }
  if (o->s1_second == 0x5420u) {
    r.verdict = device ? R2_CAUSE_INCONCLUSIVE : R2_BIAS_SWITCH_FAILED;
    r.name = r2_verdict_name(r.verdict);
    r.cause = device ? "s1_second_set0_ambiguous_bias_load_or_rs_or_store"
                     : "s1_second_used_set0";
    return r;
  }
  if (o->s1_second != 0x5020u) {
    r.cause = "s1_second_unexpected";
    return r;
  }

  /* 5. S2: each sign must appear in exactly one buffer. */
  if (o->s2_pos[0] == 0x0000u && o->s2_pos[1] == 0x5420u) {
    r.verdict = R2_STORE_INTERFERENCE;
    r.name = r2_verdict_name(r.verdict);
    r.cause = "first_store_overwritten_by_second";
    return r;
  }
  if (o->s2_pos[0] != 0x5420u) {
    r.cause = "positive_half";
    return r;
  }
  /* The negative-ACC lane must not leak into the positive buffer. This is a
   * separate check from the first-store-overwrite pattern above: a non-zero
   * value here with a correct s2_pos[0] is an unexpected export, not an
   * interference signature. */
  if (o->s2_pos[1] != 0x0000u) {
    r.cause = "positive_buffer_negative_lane";
    return r;
  }
  if (o->s2_neg[0] != 0x0000u) {
    r.verdict = device ? R2_CAUSE_INCONCLUSIVE : R2_BIAS_SWITCH_FAILED;
    r.name = r2_verdict_name(r.verdict);
    r.cause = device ? "s2_neg_s0_nonzero_ambiguous_bias_shape_or_store"
                     : "shape4_not_applied";
    return r;
  }
  if (o->s2_neg[1] == 0x0000u) {
    r.verdict = device ? R2_CAUSE_INCONCLUSIVE : R2_UNEXPECTED_OUTPUT;
    r.name = r2_verdict_name(r.verdict);
    r.cause = device ? "s2_negative_half_missing_ambiguous"
                     : "negative_half_missing";
    return r;
  }
  if (o->s2_neg[1] != 0x5420u) {
    r.cause = "negative_half_unexpected";
    return r;
  }
  if (o->s2_pos[2] != 0x0000u || o->s2_neg[2] != 0x0000u) {
    r.cause = "zero_acc_lane";
    return r;
  }
  if (o->s2_recon[0] != 0x5420u || o->s2_recon[1] != 0xd420u ||
      o->s2_recon[2] != 0x0000u) {
    r.cause = "reconstruction";
    return r;
  }

  r.verdict = R2_ACC_REUSE_SUPPORTED;
  r.name = r2_verdict_name(r.verdict);
  r.cause = "all_registered_cells_match";
  return r;
}

static r2_report_t r2_classify(const r2_observation_t *o) {
  return r2_classify_mode(o, 0);
}

static r2_report_t r2_classify_device(const r2_observation_t *o) {
  return r2_classify_mode(o, 1);
}

static void r2_observe(const r2_state *s1, const r2_state *s2,
                       r2_observation_t *o) {
  static const int idx[3] = {0, 1, 64}; /* (s0,o0), (s1,o0), (s2,o0) */
  memset(o, 0, sizeof(*o));
  o->s1_first = s1->mem[0][0];
  o->s1_second = s1->mem[1][0];
  for (int k = 0; k < 3; ++k) {
    o->s2_pos[k] = s2->mem[0][idx[k]];
    o->s2_neg[k] = s2->mem[1][idx[k]];
    o->s2_recon[k] = r2_recon_cell(s2, 0u, 1u, idx[k]);
  }
  o->mma_count = (int)(s1->mma_count + s2->mma_count);
  o->convert_count = (int)(s1->convert_count + s2->convert_count);
  o->store_count = (int)(s1->store_count + s2->store_count);
  o->trials = 2;
  o->hmx_executed = 1;
  o->slot_count = 11;
  o->sentinel_hits = 0;
  o->build_id_match = 1;
  o->close_ok = 1;
  o->pos_region_writes = s1->mem_written[0] + s2->mem_written[0];
  o->neg_region_writes = s1->mem_written[1] + s2->mem_written[1];
  o->rs_first = 0x0001u;
  o->rs_s1_second = 0x2000u;
  o->rs_second = 0x1000u;
  o->bias_loads = 3;
  o->bias_sel_first = 0u;
  o->bias_sel_second = 1u;
}

/* The future host readback order. The current host buffer is four slots; the
 * device pre-registration widens it to eleven. This function is the CPU-side
 * contract for that order: S1 first/second, then S2 pos/neg/recon for the
 * (s0,o0), (s1,o0), (s2,o0) lanes. */
static void r2_pack_slots(const r2_observation_t *o, uint16_t slots[11]) {
  slots[0] = o->s1_first;
  slots[1] = o->s1_second;
  for (int k = 0; k < 3; ++k) {
    slots[2 + k] = o->s2_pos[k];
    slots[5 + k] = o->s2_neg[k];
    slots[8 + k] = o->s2_recon[k];
  }
}

static void r2_state_model_tests(void) {
  static r2_state s1, s2;
  r2_observation_t obs;
  r2_report_t rep;
  r2_report_t r_nominal, r_acc, r_bias, r_store, r_round, r_unexpected, r_contract;
  r2_report_t r_dev_nominal, r_dev_acc, r_dev_bias, r_dev_store, r_dev_round,
      r_dev_contract;
  const long checks_before = checks;
  const double a = 1.0 + ldexp(1.0, -6);
  const double x = a * a;
  const int i0 = r2_store_index(0u, 0u);
  const int i1 = r2_store_index(1u, 0u);
  const int i2 = r2_store_index(2u, 0u);

  /* 12.1 Positive ACC: only the shape-0 buffer is non-zero. */
  r2_build_sign_tile(&s2, x, 0.0);
  r2_run_sign_tile(&s2);
  check_bits("r2.pos.pos_s0", s2.mem[0][i0], 0x5420u);
  check_bits("r2.pos.neg_s0", s2.mem[1][i0], 0x0000u);
  check_bits("r2.pos.recon_s0", r2_recon_cell(&s2, 0u, 1u, i0), 0x5420u);
  check_bits("r2.pos.pos_s1", s2.mem[0][i1], 0x0000u);
  check_bits("r2.pos.neg_s1", s2.mem[1][i1], 0x0000u);

  /* 12.2 Negative ACC: only the shape-4 buffer is non-zero. */
  r2_build_sign_tile(&s2, 0.0, -x);
  r2_run_sign_tile(&s2);
  check_bits("r2.neg.pos_s1", s2.mem[0][i1], 0x0000u);
  check_bits("r2.neg.neg_s1", s2.mem[1][i1], 0x5420u);
  check_bits("r2.neg.recon_s1", r2_recon_cell(&s2, 0u, 1u, i1), 0xd420u);
  check_bits("r2.neg.pos_s0", s2.mem[0][i0], 0x0000u);
  check_bits("r2.neg.neg_s0", s2.mem[1][i0], 0x0000u);

  /* 12.3 Zero ACC: both buffers zero, reconstruction +0. */
  r2_build_sign_tile(&s2, 0.0, 0.0);
  r2_run_sign_tile(&s2);
  check_bits("r2.zero.pos_s2", s2.mem[0][i2], 0x0000u);
  check_bits("r2.zero.neg_s2", s2.mem[1][i2], 0x0000u);
  check_bits("r2.zero.recon_s2", r2_recon_cell(&s2, 0u, 1u, i2), 0x0000u);

  /* 12.4 Retention success model: both converts read the same 1024-cell ACC,
   *      and the ACC is untouched by either store. */
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  check_u32("r2.reuse.acc_read_count", (uint32_t)s2.acc_read_count, 2u);
  check_true("r2.reuse.same_acc",
             memcmp(s2.acc_read[0], s2.acc_read[1],
                    sizeof(s2.acc_read[0])) == 0);
  check_u32("r2.reuse.mma_count", (uint32_t)s2.mma_count, 1u);
  check_u32("r2.reuse.convert_count", (uint32_t)s2.convert_count, 2u);
  check_u32("r2.reuse.store_count", (uint32_t)s2.store_count, 2u);
  check_u32("r2.reuse.regions", (uint32_t)s2.mem_written[0], 1u);
  check_u32("r2.reuse.regions_neg", (uint32_t)s2.mem_written[1], 1u);
  check_true("r2.reuse.no_bias_missing", s2.bias_missing == 0);
  check_true("r2.reuse.no_store_oob", s2.store_oob == 0);
  check_true("r2.reuse.cleared_after_second", s2.acc_valid == 0);

  /* 12.5 The second convert may retain (0x1001) instead of clear (0x1000);
   *      the stored values must be identical. Only the post-convert ACC
   *      state differs. */
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile_variant(&s2, 0x1001u);
  check_bits("r2.retain.neg_s1", s2.mem[1][i1], 0x5420u);
  check_bits("r2.retain.recon_s1", r2_recon_cell(&s2, 0u, 1u, i1), 0xd420u);
  check_true("r2.retain.acc_valid", s2.acc_valid == 1);

  /* 12.6 S1 control: a positive second store distinguishes retention from ACC
   *      destruction and bias-set selection from a default-to-set-0 failure. */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  check_bits("r2.ctrl.first", s1.mem[0][i0], 0x5420u);
  check_bits("r2.ctrl.second", s1.mem[1][i0], 0x5020u);
  check_true("r2.ctrl.same_acc",
             memcmp(s1.acc_read[0], s1.acc_read[1],
                    sizeof(s1.acc_read[0])) == 0);

  /* 12.7 Decision-procedure nominal case (simulated; not a device result). */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  rep = r2_classify(&obs);
  r_nominal = rep;
  check_true("r2.cpu.nominal", rep.verdict == R2_ACC_REUSE_SUPPORTED);
  r_dev_nominal = r2_classify_device(&obs);
  check_true("r2.device.nominal",
             r_dev_nominal.verdict == R2_ACC_REUSE_SUPPORTED);

  /* 12.8 Fault injection: ACC destroyed after the first store. */
  r2_build_control_tile(&s1, x);
  r2_convert(&s1, 0x0001u);
  r2_store(&s1, 0u);
  r2_force_acc_destroy(&s1);
  r2_convert(&s1, 0x2000u);
  r2_store(&s1, 1u);
  r2_build_sign_tile(&s2, x, -x);
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  r2_force_acc_destroy(&s2);
  r2_convert(&s2, 0x1000u);
  r2_store(&s2, 1u);
  r2_observe(&s1, &s2, &obs);
  check_bits("r2.destroyed.ctrl_second", s1.mem[1][i0], 0x0000u);
  check_bits("r2.destroyed.neg_s1", s2.mem[1][i1], 0x0000u);
  rep = r2_classify(&obs);
  r_acc = rep;
  check_true("r2.destroyed.verdict", rep.verdict == R2_ACC_DESTROYED);
  r_dev_acc = r2_classify_device(&obs);
  check_true("r2.destroyed.device_inconclusive",
             r_dev_acc.verdict == R2_CAUSE_INCONCLUSIVE);

  /* 12.9 Fault injection: Rs selects set 1 but the hardware silently uses
   *      set 0. The S1 control exposes the switch failure with a positive
   *      second word (0x5420 instead of 0x5020). */
  r2_build_control_tile(&s1, x);
  r2_convert(&s1, 0x0001u);
  r2_store(&s1, 0u);
  s1.force_bias_sel = 0;
  r2_convert(&s1, 0x2000u);
  r2_store(&s1, 1u);
  r2_build_sign_tile(&s2, x, -x);
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  s2.force_bias_sel = 0;
  r2_convert(&s2, 0x1000u);
  r2_store(&s2, 1u);
  r2_observe(&s1, &s2, &obs);
  check_bits("r2.biasfail.ctrl_second", s1.mem[1][i0], 0x5420u);
  check_bits("r2.biasfail.neg_s0", s2.mem[1][i0], 0x5420u);
  check_bits("r2.biasfail.neg_s1", s2.mem[1][i1], 0x0000u);
  rep = r2_classify(&obs);
  r_bias = rep;
  check_true("r2.biasfail.verdict", rep.verdict == R2_BIAS_SWITCH_FAILED);
  r_dev_bias = r2_classify_device(&obs);
  check_true("r2.biasfail.device_inconclusive",
             r_dev_bias.verdict == R2_CAUSE_INCONCLUSIVE);

  /* 12.10 Fault injection: both stores target region 0. The second store
   *       overwrites the first, and the negative buffer is never written. */
  r2_build_control_tile(&s1, x);
  r2_convert(&s1, 0x0001u);
  r2_store(&s1, 0u);
  r2_convert(&s1, 0x2000u);
  r2_store(&s1, 0u);
  r2_build_sign_tile(&s2, x, -x);
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  r2_convert(&s2, 0x1000u);
  r2_store(&s2, 0u);
  r2_observe(&s1, &s2, &obs);
  check_u32("r2.store.pos_writes", (uint32_t)obs.pos_region_writes, 4u);
  check_u32("r2.store.neg_writes", (uint32_t)obs.neg_region_writes, 0u);
  check_bits("r2.store.overwritten_pos", s2.mem[0][i0], 0x0000u);
  rep = r2_classify(&obs);
  r_store = rep;
  check_true("r2.store.verdict", rep.verdict == R2_STORE_INTERFERENCE);
  r_dev_store = r2_classify_device(&obs);
  check_true("r2.store.device_interference",
             r_dev_store.verdict == R2_STORE_INTERFERENCE);

  /* 12.11 Fault injection: one-ULP ceiling rounding instead of RN16. */
  r2_build_control_tile(&s1, x);
  s1.rounding_mode = 1;
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  s2.rounding_mode = 1;
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  check_bits("r2.round.ctrl_second", s1.mem[1][i0], 0x5021u);
  check_bits("r2.round.pos_s0", s2.mem[0][i0], 0x5421u);
  check_bits("r2.round.neg_s1", s2.mem[1][i1], 0x5421u);
  rep = r2_classify(&obs);
  r_round = rep;
  check_true("r2.round.verdict", rep.verdict == R2_ROUNDING_MISMATCH);
  r_dev_round = r2_classify_device(&obs);
  check_true("r2.round.device_rounding",
             r_dev_round.verdict == R2_ROUNDING_MISMATCH);

  /* 12.12 Fault injection: non-zero output bias on the shape-4 row. */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_bias_fill(&s2, 1u, 0x5400u, 4u, 0x4400u); /* bo = +4 */
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  check_bits("r2.bo.neg_s1", s2.mem[1][i1], 0x5460u);
  rep = r2_classify(&obs);
  r_unexpected = rep;
  check_true("r2.bo.verdict", rep.verdict == R2_UNEXPECTED_OUTPUT);

  /* 12.13 Feedback: (a) an Rs with fb_dest != 0 fails the host contract before
   *       any output interpretation; (b) a hypothetical hardware feedback leak
   *       with the registered Rs is caught by the output when the clamp is at
   *       the store. If the clamp is inside the convert, the output detector is
   *       masked; the Rs contract remains the robust guard. That limitation is
   *       recorded explicitly, not patched away. */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  r2_convert(&s2, 0x1005u); /* set 1, fb_dest=1, min, clear */
  r2_store(&s2, 1u);
  r2_observe(&s1, &s2, &obs);
  obs.rs_second = 0x1005u;
  rep = r2_classify(&obs);
  r_contract = rep;
  check_true("r2.fb.rs_contract", rep.verdict == R2_HOST_CONTRACT_FAILED);
  r_dev_contract = r2_classify_device(&obs);
  check_true("r2.fb.device_contract",
             r_dev_contract.verdict == R2_HOST_CONTRACT_FAILED);

  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  s2.force_feedback_dest = 1;
  s2.force_feedback_op = 0;
  r2_convert(&s2, 0x1000u);
  r2_store(&s2, 1u);
  r2_observe(&s1, &s2, &obs);
  /* The feedback state is 20-bit (FP16 + extra bits), and Rs[8] selects the
   * precision; this oracle does not calibrate that payload. What it does
   * assert is that enabling feedback changes the registered output. */
  check_true("r2.fb.leak_differs", s2.mem[1][i1] != 0x5420u);
  printf("r2_feedback_leak_observed=0x%04x registered=0x5420\n",
         (unsigned)s2.mem[1][i1]);
  rep = r2_classify(&obs);
  check_true("r2.fb.leak_verdict", rep.verdict == R2_UNEXPECTED_OUTPUT);

  r2_build_sign_tile(&s2, x, -x);
  s2.clamp_in_convert = 1;
  r2_convert(&s2, 0x0001u);
  r2_store(&s2, 0u);
  s2.force_feedback_dest = 1;
  s2.force_feedback_op = 0;
  r2_convert(&s2, 0x1000u);
  r2_store(&s2, 1u);
  check_bits("r2.fb.clamp_masks_output", s2.mem[1][i1], 0x5420u);
  printf("r2_feedback_output_detector=%s (clamp_in_convert=1 masks the "
         "difference; Rs contract still guards)\n",
         s2.mem[1][i1] == 0x5420u ? "MASKED_AS_EXPECTED" : "DETECTED");

  /* 12.14 Corrupted word and host-contract failures. */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  s2.mem[1][i1] ^= 0x0100u; /* 0x5420 -> 0x5520, not a rounding neighbour */
  r2_observe(&s1, &s2, &obs);
  rep = r2_classify(&obs);
  check_true("r2.corrupt.verdict", rep.verdict == R2_UNEXPECTED_OUTPUT);

  /* 12.14b A negative-ACC lane leaked into the positive buffer while the
   *        positive anchor is still correct. This must not be mistaken for
   *        the first-store-overwrite pattern (which has s2_pos[0]=0). */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  obs.s2_pos[1] = 0x5420u;
  rep = r2_classify(&obs);
  check_true("r2.posleak.verdict", rep.verdict == R2_UNEXPECTED_OUTPUT);

  /* 12.14c A non-zero shape-4 buffer at the positive lane. The model knows
   *        the injected bias failure; a device observation alone cannot
   *        distinguish bias selection from shape or store failure. */
  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  obs.s2_neg[0] = 0x5420u;
  check_true("r2.s2neg0.model",
             r2_classify(&obs).verdict == R2_BIAS_SWITCH_FAILED);
  check_true("r2.s2neg0.device_inconclusive",
             r2_classify_device(&obs).verdict == R2_CAUSE_INCONCLUSIVE);

  r2_build_control_tile(&s1, x);
  r2_run_control_tile(&s1);
  r2_build_sign_tile(&s2, x, -x);
  r2_run_sign_tile(&s2);
  r2_observe(&s1, &s2, &obs);
  obs.mma_count = 3;
  rep = r2_classify(&obs);
  check_true("r2.contract.mma", rep.verdict == R2_HOST_CONTRACT_FAILED);
  r2_observe(&s1, &s2, &obs);
  obs.convert_count = 3;
  rep = r2_classify(&obs);
  check_true("r2.contract.convert", rep.verdict == R2_HOST_CONTRACT_FAILED);
  r2_observe(&s1, &s2, &obs);
  obs.store_count = 3;
  rep = r2_classify(&obs);
  check_true("r2.contract.store", rep.verdict == R2_HOST_CONTRACT_FAILED);
  r2_observe(&s1, &s2, &obs);
  obs.rs_second = 0x1004u; /* feedback destination set */
  rep = r2_classify(&obs);
  check_true("r2.contract.feedback_rs",
             rep.verdict == R2_HOST_CONTRACT_FAILED);
  r2_observe(&s1, &s2, &obs);
  obs.rs_s1_second = 0x0000u; /* S1 second convert lost its set-2 select */
  rep = r2_classify(&obs);
  check_true("r2.contract.s1_second_rs",
             rep.verdict == R2_HOST_CONTRACT_FAILED);
  r2_observe(&s1, &s2, &obs);
  obs.bias_loads = 2; /* one of the three prologue loads did not execute */
  rep = r2_classify(&obs);
  check_true("r2.contract.bias_loads",
             rep.verdict == R2_HOST_CONTRACT_FAILED);

  /* 12.15 VTCM layout: two 2-KiB-aligned output regions, two 256-B-aligned
   *       bias rows, no overlap, and a 1024-cell store bijection. */
  {
    const unsigned pos_off = 2u * 2048u;
    const unsigned neg_off = 3u * 2048u;
    const unsigned b0_off = 4u * 2048u;
    const unsigned b1_off = b0_off + 256u;
    check_true("r2.layout.pos_aligned", (pos_off % 2048u) == 0u);
    check_true("r2.layout.neg_aligned", (neg_off % 2048u) == 0u);
    check_true("r2.layout.bias0_aligned", (b0_off % 256u) == 0u);
    check_true("r2.layout.bias1_aligned", (b1_off % 256u) == 0u);
    check_true("r2.layout.disjoint",
               pos_off + 2048u <= neg_off && neg_off + 2048u <= b0_off &&
                   b0_off + 256u <= b1_off && b1_off + 256u <= 12288u);
    uint8_t hit[1024];
    memset(hit, 0, sizeof(hit));
    unsigned dup = 0;
    for (unsigned s = 0; s < R2_NS; ++s)
      for (unsigned o = 0; o < R2_NO; ++o) {
        const unsigned idx = (unsigned)r2_store_index(s, o);
        if (idx >= 1024u || hit[idx]) ++dup;
        else hit[idx] = 1;
      }
    check_u32("r2.layout.store_bijection_dup", dup, 0u);
  }

  /* 12.16 11-slot host readback and fail-closed metadata. */
  {
    uint16_t slots[11];
    static const uint16_t expected[11] = {
        0x5420u, 0x5020u, 0x5420u, 0x0000u, 0x0000u, 0x0000u,
        0x5420u, 0x0000u, 0x5420u, 0xd420u, 0x0000u};
    int slot_ok = 1, sentinel_ok = 1;
    r2_build_control_tile(&s1, x);
    r2_run_control_tile(&s1);
    r2_build_sign_tile(&s2, x, -x);
    r2_run_sign_tile(&s2);
    r2_observe(&s1, &s2, &obs);
    r2_pack_slots(&obs, slots);
    for (int k = 0; k < 11; ++k) {
      if (slots[k] != expected[k]) slot_ok = 0;
      if (slots[k] == 0xa5a5u) sentinel_ok = 0;
    }
    check_true("r2.slots.expected", slot_ok);
    check_true("r2.slots.no_sentinel", sentinel_ok);
    check_u32("r2.slots.count", (uint32_t)obs.slot_count, 11u);
    check_true("r2.slots.device_supported",
               r2_classify_device(&obs).verdict == R2_ACC_REUSE_SUPPORTED);

    r2_observe(&s1, &s2, &obs);
    obs.slot_count = 10;
    check_true("r2.host.slot_count",
               r2_classify_device(&obs).verdict == R2_HOST_CONTRACT_FAILED);
    r2_observe(&s1, &s2, &obs);
    obs.sentinel_hits = 1;
    check_true("r2.host.sentinel",
               r2_classify_device(&obs).verdict == R2_HOST_CONTRACT_FAILED);
    r2_observe(&s1, &s2, &obs);
    obs.build_id_match = 0;
    check_true("r2.host.build_id",
               r2_classify_device(&obs).verdict == R2_HOST_CONTRACT_FAILED);
    r2_observe(&s1, &s2, &obs);
    obs.close_ok = 0;
    check_true("r2.host.close",
               r2_classify_device(&obs).verdict == R2_HOST_CONTRACT_FAILED);
  }

  printf("r2_cpu_model nominal=%s acc_destroyed=%s bias_switch=%s "
         "store_interference=%s rounding=%s unexpected=%s contract=%s\n",
         r_nominal.name, r_acc.name, r_bias.name, r_store.name, r_round.name,
         r_unexpected.name, r_contract.name);
  printf("r2_device_classifier nominal=%s s1_zero=%s s1_set0=%s "
         "store=%s rounding=%s contract=%s\n",
         r_dev_nominal.name, r_dev_acc.name, r_dev_bias.name,
         r_dev_store.name, r_dev_round.name, r_dev_contract.name);
  printf("r2_slot_contract=11/11 sentinel=0 host_fail_closed=PASS\n");
  printf("r2_cpu_model_checks=%ld\n", checks - checks_before);
  /* This line is deliberately unconditional. No device was contacted and the
   * nominal label is a decision-procedure test, not an established result. */
  printf("r2_device_verdict=NOT_RUN (R2_ACC_REUSE_SUPPORTED above is a "
         "CPU-model classifier test only)\n");
}

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

  // -------------------------------------------------------------------------
  // 10. mode-44 (E-3D) host contract classifier.
  //     probe_host.c and this oracle must execute the same decision
  //     procedure; a mismatch here is a contract bug, not a device result.
  // -------------------------------------------------------------------------
  {
    const int pats[10][4] = {
        {0x5420, 0xd420, 0xd420, 0x5420}, /* signed (expected) */
        {0x5420, 0x0000, 0x0000, 0x5420}, /* late clamp */
        {0x5420, 0x0000, 0x0000, 0x0000}, /* early clamp */
        {0x5420, 0x0000, 0x5420, 0x0000}, /* shape ignored */
        {0x5420, 0xd420, 0x5420, 0xd420}, /* shape ignored, signed store */
        {0x5420, 0x0000, 0xd420, 0x8000}, /* ACC clamp + signed store */
        {0x5420, 0x5420, 0x5420, 0x5420}, /* magnitude store */
        {0x5420, 0xfbff, 0xfbff, 0x5420}, /* negative maxnorm */
        {0x5421, 0xd420, 0xd420, 0x5421}, /* round toward +inf */
        {0x5420, 0xd421, 0xd421, 0x5420}}; /* round toward -inf */
    const hmx_mode44_verdict_t want[10] = {
        HMX_MODE44_SIGNED_STORE_CONFIRMED,
        HMX_MODE44_CLAMP_DOWNSTREAM_OF_SHAPE,
        HMX_MODE44_CLAMP_AT_ACC_BEFORE_SHAPE,
        HMX_MODE44_SHAPE_NOT_APPLIED_PER_CHANNEL,
        HMX_MODE44_SHAPE_NOT_APPLIED_PER_CHANNEL,
        HMX_MODE44_SIGNED_STORE_WITH_ACC_CLAMP,
        HMX_MODE44_MAGNITUDE_STORE,
        HMX_MODE44_NEGATIVE_MAXNORM,
        HMX_MODE44_ROUNDING_NOT_RN16,
        HMX_MODE44_ROUNDING_NOT_RN16};
    const char *want_names[10] = {
        "E3D_SIGNED_STORE_CONFIRMED", "E3D_CLAMP_DOWNSTREAM_OF_SHAPE",
        "E3D_CLAMP_AT_ACC_BEFORE_SHAPE", "E3D_SHAPE_NOT_APPLIED_PER_CHANNEL",
        "E3D_SHAPE_NOT_APPLIED_PER_CHANNEL",
        "E3D_SIGNED_STORE_WITH_ACC_CLAMP", "E3D_MAGNITUDE_STORE",
        "E3D_NEGATIVE_MAXNORM", "E3D_ROUNDING_NOT_RN16",
        "E3D_ROUNDING_NOT_RN16"};
    for (int p = 0; p < 10; ++p) {
      const hmx_mode44_report_t r =
          hmx_mode44_classify(pats[p], HMX_MODE44_TRIALS, 1);
      ++checks;
      if (!r.matched || r.verdict != want[p] ||
          strcmp(r.verdict_name, want_names[p]) != 0) {
        ++failures;
        printf("FAIL e3d.classify[%d] verdict=%s matched=%d\n", p,
               r.verdict_name, r.matched);
      }
    }

    /* One-trial contract: 2 trials or no HMX execution must fail closed. */
    {
      const hmx_mode44_report_t ok =
          hmx_mode44_classify(pats[0], HMX_MODE44_TRIALS, 1);
      const hmx_mode44_report_t two = hmx_mode44_classify(pats[0], 2, 1);
      const hmx_mode44_report_t none = hmx_mode44_classify(pats[0], 0, 0);
      check_true("e3d.contract_ok", ok.contract_ok == 1);
      check_true("e3d.contract_rejects_two_trials", two.contract_ok == 0);
      check_true("e3d.contract_rejects_no_hmx", none.contract_ok == 0);
      check_true("e3d.expected_trials_is_one", ok.expected_trials == 1);
    }

    /* Precondition: out[0] must be the positive anchor. */
    {
      const int bad0[4] = {0x0000, 0xd420, 0xd420, 0x5420};
      const hmx_mode44_report_t r =
          hmx_mode44_classify(bad0, HMX_MODE44_TRIALS, 1);
      check_true("e3d.precondition_rejects_bad_out0",
                 r.verdict == HMX_MODE44_INCONCLUSIVE_TILE &&
                     r.out0_expected == 0);
    }

    /* Pre-registered bias_shape_misapplied observation: shape 4 landed on
     * channel 0 and swapped slots 0/2. Still INCONCLUSIVE_TILE. */
    {
      const int swap[4] = {0xd420, 0xd420, 0x5420, 0x5420};
      const hmx_mode44_report_t r =
          hmx_mode44_classify(swap, HMX_MODE44_TRIALS, 1);
      check_true("e3d.bias_shape_misapplied",
                 r.bias_shape_misapplied == 1 &&
                     r.verdict == HMX_MODE44_INCONCLUSIVE_TILE);
    }

    /* Single-cell fallback: a negative out[2] that matches no row still
     * proves a sign-preserving store, because out[2]'s ACC is positive. */
    {
      const int anomaly[4] = {0x5420, 0x0000, 0xd123, 0x5420};
      const hmx_mode44_report_t r =
          hmx_mode44_classify(anomaly, HMX_MODE44_TRIALS, 1);
      check_true("e3d.single_cell_fallback",
                 r.verdict ==
                         HMX_MODE44_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY &&
                     r.matched == 0);
    }

    /* Anything else is recorded verbatim and NOT interpreted. */
    {
      const int other[4] = {0x5420, 0x0000, 0x0000, 0x8000};
      const hmx_mode44_report_t r =
          hmx_mode44_classify(other, HMX_MODE44_TRIALS, 1);
      check_true("e3d.unmatched_is_inconclusive",
                 r.verdict == HMX_MODE44_INCONCLUSIVE_UNMATCHED &&
                     r.matched == 0);
    }
  }

  // -------------------------------------------------------------------------
  // 11. Sign-separated reconstruction with the E-3D-confirmed shapes:
  //       y_rec = store(shape 0(y)) - store(shape 4(y)).
  //
  // This section models ENCODING only; the ACC source is not changed. A
  // difference against a signed RN16 store here would therefore be a
  // re-encoding defect, never an arithmetic gain.
  //
  // Preconditions (PRM-derived or measured):
  //   * shape 0 = identity, shape 4 = -x     (Table 5; E-3D out[2] = 0x0000)
  //   * negative pre-store value -> 0x0000   (E-3D four slots)
  //   * out_bias bo = 0 on both channels     (E-3D bias row)
  //   * both duplicated channels share one positive scale
  //
  // With bo = 0 exactly one half survives, so there is one RN16 rounding per
  // output and the subtraction is exact. The method is bit-equivalent to a
  // signed RN16 store up to +/-0 and NaN payload; the sweep below is banded
  // (subnormal .. overflow) so a band-limited failure cannot hide.
  // -------------------------------------------------------------------------
  {
    long cases = 0, both_nonzero = 0, one_sided = 0, both_zero = 0;
    long numeric_mismatch = 0, neg_zero_bitdiff = 0;
    for (int e = -32; e <= 11; ++e) {
      for (int k = 0; k < 64; ++k) {
        const double mant = 1.0 + (double)k / 64.0;
        for (int sign = -1; sign <= 1; sign += 2) {
          const double y = sign * ldexp(mant, e);
          const uint16_t direct = double_to_fp16(64.0 * y);
          uint16_t s0, s4;
          const uint16_t recon = split04_store(y, 0.0, 64.0, 0.0, &s0, &s4);
          if (s0 != 0u && s4 != 0u) ++both_nonzero;
          if ((s0 != 0u) != (s4 != 0u)) ++one_sided;
          if (s0 == 0u && s4 == 0u) ++both_zero;
          if (fp16_to_double(recon) != fp16_to_double(direct)) {
            ++numeric_mismatch;
          } else if (recon != direct) {
            // Only expected bit difference: RN16(-0) = 0x8000 on the direct
            // path vs +0 on the split path. Numerically identical.
            ++neg_zero_bitdiff;
          }
          ++cases;
        }
      }
    }
    printf("split04 sweep: cases=%ld both_nonzero=%ld one_sided=%ld "
           "both_zero=%ld numeric_mismatch=%ld neg_zero_bitdiff=%ld\n",
           cases, both_nonzero, one_sided, both_zero, numeric_mismatch,
           neg_zero_bitdiff);
    check_true("split04.sweep_nonempty", cases > 0);
    check_u32("split04.both_nonzero", (uint32_t)both_nonzero, 0u);
    check_u32("split04.numeric_mismatch", (uint32_t)numeric_mismatch, 0u);
    check_true("split04.neg_zero_band_exercised", neg_zero_bitdiff > 0);
  }

  // The four E-3D slots reproduced from the measured ACC magnitudes.
  {
    uint16_t s0, s4;
    const uint16_t neg = split04_store(-x, 0.0, 64.0, 0.0, &s0, &s4);
    check_bits("split04.e3d_neg_shape0", s0, 0x0000u);
    check_bits("split04.e3d_neg_shape4", s4, 0x5420u);
    check_bits("split04.e3d_neg_recon", neg, 0xd420u);
    const uint16_t pos = split04_store(x, 0.0, 64.0, 0.0, &s0, &s4);
    check_bits("split04.e3d_pos_shape0", s0, 0x5420u);
    check_bits("split04.e3d_pos_shape4", s4, 0x0000u);
    check_bits("split04.e3d_pos_recon", pos, 0x5420u);
    const uint16_t neg2 = split04_store(-2.0 * x, 0.0, 64.0, 0.0, &s0, &s4);
    check_bits("split04.e3d_2x_shape4", s4, 0x5820u);
    check_bits("split04.e3d_2x_recon", neg2, 0xd820u);
  }

  // Validity guard. The split is correct only while the clamp law holds: if
  // a future mode preserved the sign, shape 0 would export -32 (not 0),
  // shape 4 would export +32, and the subtraction would return 2y. The
  // E-3D slots out[1] == 0x0000 and out[2] == 0x0000 are the re-test.
  {
    const double y = -0.5;
    const uint16_t signed0 = double_to_fp16(64.0 * y);   // 0xd000
    const uint16_t signed4 = double_to_fp16(-64.0 * y);  // 0x5000
    const uint16_t recon =
        double_to_fp16(fp16_to_double(signed0) - fp16_to_double(signed4));
    check_bits("split04.signed_store_shape0", signed0, 0xd000u);
    check_bits("split04.signed_store_shape4", signed4, 0x5000u);
    check_bits("split04.signed_store_recon_is_2y", recon, 0xd400u);
  }

  // bo != 0 breaks the pair. The bias is added AFTER the shape, so it
  // survives once (small bias -> y + bo) or twice (large bias -> 2y).
  {
    uint16_t s0, s4;
    const uint16_t two_y = split04_store(-0.5, 0.0, 64.0, 64.0, &s0, &s4);
    check_bits("split04.bo64_shape0", s0, 0x5000u);  // -32 + 64 = +32
    check_bits("split04.bo64_shape4", s4, 0x5600u);  // +32 + 64 = +96
    check_bits("split04.bo64_recon_is_2y", two_y, 0xd400u);
    check_true("split04.bo64_both_nonzero", s0 != 0u && s4 != 0u);

    const uint16_t y_plus = split04_store(0.5, 0.0, 64.0, 1.0, &s0, &s4);
    check_bits("split04.bo1_positive_shape0", s0, 0x5020u);  // 32 + 1
    check_bits("split04.bo1_positive_shape4", s4, 0x0000u);
    check_bits("split04.bo1_positive_recon", y_plus, 0x5020u);
    // The both-nonzero symptom does NOT cover this case, so a data-only
    // detector would miss a one-count bias error. Only a configuration
    // invariant (bo == 0) is sufficient.
    check_true("split04.bo1_one_sided_undetected", s4 == 0u);
  }

  // bi is applied BEFORE the shape and cancels exactly; only the post-shape
  // bo is forbidden.
  {
    uint16_t s0, s4;
    const uint16_t rec = split04_store(-1.0, 0.5, 64.0, 0.0, &s0, &s4);
    check_bits("split04.bi_pre_shape_cancels", rec, 0xd000u);
  }

  // Cross-check against section 8's pair. shape 2 / shape 5 give the same
  // reconstruction with bo = 0, and also with bo != 0 because the shape
  // zeroes the inactive half before the bias is added. That is the
  // bias-safe variant, but it relies on PRM shape 2/5 semantics that this
  // device configuration has not exercised (E-3D used shapes 0 and 4 only).
  {
    const uint16_t p = empirical_negative_clamp(
        prm_convert_rn16(-0.5, 0.0, 2u, 64.0, 64.0));
    const uint16_t n = empirical_negative_clamp(
        prm_convert_rn16(-0.5, 0.0, 5u, 64.0, 64.0));
    const uint16_t rec =
        double_to_fp16(fp16_to_double(p) - fp16_to_double(n));
    check_bits("split25.bo64_pos_part", p, 0x5400u);
    check_bits("split25.bo64_neg_part", n, 0x5600u);
    check_bits("split25.bo64_recon_is_y", rec, 0xd000u);
  }

  // Signed-zero caveat: a bitwise output comparator sees 0x8000 on the
  // direct path and 0x0000 on the split path. Both are numerically zero.
  {
    uint16_t s0, s4;
    const uint16_t direct = double_to_fp16(-0.0);
    const uint16_t rec = split04_store(-0.0, 0.0, 64.0, 0.0, &s0, &s4);
    check_bits("split04.neg_zero_direct", direct, 0x8000u);
    check_bits("split04.neg_zero_recon", rec, 0x0000u);
    check_true("split04.neg_zero_numeric",
               fp16_to_double(rec) == fp16_to_double(direct));
  }

  // -------------------------------------------------------------------------
  // 12. R2 state machine, fault injection and future decision procedure.
  // -------------------------------------------------------------------------
  r2_state_model_tests();

  printf("hmx_signed_output_oracle checks=%ld fail=%d\n", checks, failures);
  if (failures) {
    printf("hmx_signed_output_oracle=FAIL\n");
    return 1;
  }
  printf("hmx_signed_output_oracle=PASS\n");
  return 0;
}
