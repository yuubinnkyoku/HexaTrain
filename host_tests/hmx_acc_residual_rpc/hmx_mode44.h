// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// E-3D (mode 44) contract: can the V81 HMX convert path store a negative
// FP16 value?
//
// One 32x32x32 K32 tile, ONE MMA packet, ONE convert, ONE store, four read
// cells. Shared by probe_dsp.c (executes it), probe_host.c (classifies the
// device result) and hmx_signed_output_oracle.c (CPU-only contract test) so
// the decision procedure cannot drift between host and test.
//
// Fixture (variant A, the authorised literal mission fixture):
//   a = 1 + 2^-6 = 1.015625 = 0x3C10 (exact FP16)
//   x = a*a      = 1.031494140625 (exact in the ACC)
//   A(s0,i0) = +a, A(s1,i0) = -a, W(o0,i0) = +a, W(o1,i0) = +a
//   scale = +64, output bias = +0, shape 0 (channel 0) / 4 (channel 1),
//   convert scalar Rs = 0x00000001 (bias set 0, IEEE, no feedback, retain).
//
// Store cells, per PRM store_u16(s,o) = (s>>1)*64 + o*2 + (s&1):
//   results[0] = out[0] -> (s0,o0): ACC +x, shape 0 -> 0x5420
//   results[1] = out[1] -> (s1,o0): ACC -x, shape 0 -> 0xD420
//   results[2] = out[2] -> (s0,o1): ACC +x, shape 4 -> 0xD420 (DECISIVE)
//   results[3] = out[3] -> (s1,o1): ACC -x, shape 4 -> 0x5420
//
// out[2] is the cell the design exists for: its ACC is positive, so a
// negative stored value cannot be explained by clamping the accumulator.
//
// Expectations are frozen BEFORE the first device run (prereg doc
// docs/hmx-v81-signed-fp16-discriminator-prereg.md, section 8 variant A,
// section 15 execution selection). Do not retune after a measurement.

#ifndef HEXATRAIN_HMX_MODE44_H
#define HEXATRAIN_HMX_MODE44_H

#include <stdint.h>

#define HMX_MODE44_TRIALS 1
#define HMX_MODE44_SLOT_COUNT 4

/* Fixture constants (FP16 bit patterns, exact). */
#define HMX_MODE44_A_BITS 0x3c10u
#define HMX_MODE44_NEG_A_BITS 0xbc10u
#define HMX_MODE44_SCALE_BITS 0x5400u /* +64 */
#define HMX_MODE44_SHAPE_CH0 0u
#define HMX_MODE44_SHAPE_CH1 4u
#define HMX_MODE44_RS ((uint32_t)0x00000001u)

/* Signed (expected) store cells, variant A. */
#define HMX_MODE44_OUT0_BITS 0x5420u
#define HMX_MODE44_OUT1_BITS 0xd420u
#define HMX_MODE44_OUT2_BITS 0xd420u
#define HMX_MODE44_OUT3_BITS 0x5420u
/* Precondition anomaly: out[0] carries the negative cell while out[2] keeps
 * the positive one (bias row applied to the wrong output channel). */
#define HMX_MODE44_OUT0_SWAPPED_BITS 0xd420u

/* VTCM packing (PRM 2.2.1 / 2.2.2), device-corroborated at store 0/32. */
static inline int hmx_mode44_act_index(int s, int i) {
  return (s >> 1) * 64 + i * 2 + (s & 1);
}
static inline int hmx_mode44_wgt_index(int o, int i) {
  return (i >> 1) * 64 + o * 2 + (i & 1);
}
static inline int hmx_mode44_store_index(int s, int o) {
  return (s >> 1) * 64 + o * 2 + (s & 1);
}

typedef enum hmx_mode44_verdict {
  HMX_MODE44_SIGNED_STORE_CONFIRMED = 0,
  HMX_MODE44_CLAMP_DOWNSTREAM_OF_SHAPE,
  HMX_MODE44_CLAMP_AT_ACC_BEFORE_SHAPE,
  HMX_MODE44_SHAPE_NOT_APPLIED_PER_CHANNEL,
  HMX_MODE44_SIGNED_STORE_WITH_ACC_CLAMP,
  HMX_MODE44_MAGNITUDE_STORE,
  HMX_MODE44_NEGATIVE_MAXNORM,
  HMX_MODE44_ROUNDING_NOT_RN16,
  HMX_MODE44_INCONCLUSIVE_TILE,
  HMX_MODE44_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY,
  HMX_MODE44_INCONCLUSIVE_UNMATCHED
} hmx_mode44_verdict_t;

typedef struct hmx_mode44_report {
  /* Raw slot facts: always recorded, even when nothing matches. */
  uint16_t out[HMX_MODE44_SLOT_COUNT];
  int out0_expected; /* precondition: out[0] == 0x5420 */
  /* out[0] == 0xD420 and out[2] == 0x5420 is the pre-registered
   * "bias_shape_misapplied" observation from the decision procedure. */
  int bias_shape_misapplied;
  int matched; /* a registered pattern matched */
  const char *matched_name;
  int matched_store_signed; /* the shape-ignored row assumes a signed store */
  /* Host contract: mode 44 must be exactly one executed trial. */
  int expected_trials;
  int observed_trials;
  int trials_ok;
  int hmx_executed_ok;
  int contract_ok; /* trials_ok && hmx_executed_ok */
  hmx_mode44_verdict_t verdict;
  const char *verdict_name;
} hmx_mode44_report_t;

static inline const char *hmx_mode44_verdict_name(
    hmx_mode44_verdict_t verdict) {
  switch (verdict) {
    case HMX_MODE44_SIGNED_STORE_CONFIRMED:
      return "E3D_SIGNED_STORE_CONFIRMED";
    case HMX_MODE44_CLAMP_DOWNSTREAM_OF_SHAPE:
      return "E3D_CLAMP_DOWNSTREAM_OF_SHAPE";
    case HMX_MODE44_CLAMP_AT_ACC_BEFORE_SHAPE:
      return "E3D_CLAMP_AT_ACC_BEFORE_SHAPE";
    case HMX_MODE44_SHAPE_NOT_APPLIED_PER_CHANNEL:
      return "E3D_SHAPE_NOT_APPLIED_PER_CHANNEL";
    case HMX_MODE44_SIGNED_STORE_WITH_ACC_CLAMP:
      return "E3D_SIGNED_STORE_WITH_ACC_CLAMP";
    case HMX_MODE44_MAGNITUDE_STORE:
      return "E3D_MAGNITUDE_STORE";
    case HMX_MODE44_NEGATIVE_MAXNORM:
      return "E3D_NEGATIVE_MAXNORM";
    case HMX_MODE44_ROUNDING_NOT_RN16:
      return "E3D_ROUNDING_NOT_RN16";
    case HMX_MODE44_INCONCLUSIVE_TILE:
      return "E3D_INCONCLUSIVE_TILE";
    case HMX_MODE44_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY:
      return "E3D_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY";
    case HMX_MODE44_INCONCLUSIVE_UNMATCHED:
      return "E3D_INCONCLUSIVE_UNMATCHED";
  }
  return "E3D_INCONCLUSIVE_UNMATCHED";
}

/* Pure classifier: no I/O, no device state. `trials` and `hmx_executed`
 * are metadata[8]/metadata[7]. The verdict is only meaningful when
 * contract_ok is 1; the caller must fail closed otherwise and must not
 * consume the verdict.
 *
 * Order (prereg section 9 as amended in section 16):
 *   1. exact match against the ten variant-A rows
 *      (the ROUND_CEIL row legitimately has out[0] = 0x5421, so matching
 *      must precede the anchor precondition)
 *   2. no match and out[0] != 0x5420             -> INCONCLUSIVE_TILE
 *   3. no match, anchor ok, out[2] negative      -> SIGNED_STORE_WITH_
 *                                                   UNREGISTERED_ANOMALY
 *   4. otherwise                                 -> INCONCLUSIVE_UNMATCHED
 * No retry, no re-tuning, no after-the-fact explanation. */
static inline hmx_mode44_report_t hmx_mode44_classify(
    const int results[HMX_MODE44_SLOT_COUNT], int trials, int hmx_executed) {
  static const uint16_t patterns[10][HMX_MODE44_SLOT_COUNT] = {
      /* signed (expected) */
      {0x5420u, 0xd420u, 0xd420u, 0x5420u},
      /* late clamp: clamp at/after the convert */
      {0x5420u, 0x0000u, 0x0000u, 0x5420u},
      /* early clamp: clamp at the ACC input + store clamp */
      {0x5420u, 0x0000u, 0x0000u, 0x0000u},
      /* shape 4 not applied per channel */
      {0x5420u, 0x0000u, 0x5420u, 0x0000u},
      /* shape 4 not applied per channel, store signed */
      {0x5420u, 0xd420u, 0x5420u, 0xd420u},
      /* clamp at ACC, signed store, -0 kept */
      {0x5420u, 0x0000u, 0xd420u, 0x8000u},
      /* magnitude store: sign dropped, not zeroed */
      {0x5420u, 0x5420u, 0x5420u, 0x5420u},
      /* negative saturates to -maxnorm */
      {0x5420u, 0xfbffu, 0xfbffu, 0x5420u},
      /* rounding toward +inf */
      {0x5421u, 0xd420u, 0xd420u, 0x5421u},
      /* rounding toward -inf */
      {0x5420u, 0xd421u, 0xd421u, 0x5420u}};
  static const char *names[10] = {
      "signed",
      "late_clamp",
      "early_clamp",
      "shape_ignored",
      "shape_ignored_signed",
      "accclamp_signed",
      "abs_store",
      "neg_maxnorm",
      "round_ceiling",
      "round_floor"};
  static const hmx_mode44_verdict_t verdicts[10] = {
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

  hmx_mode44_report_t report;
  for (int k = 0; k < HMX_MODE44_SLOT_COUNT; ++k)
    report.out[k] = (uint16_t)results[k];
  report.out0_expected = report.out[0] == HMX_MODE44_OUT0_BITS;
  report.bias_shape_misapplied =
      report.out[0] == HMX_MODE44_OUT0_SWAPPED_BITS &&
      report.out[2] == HMX_MODE44_OUT0_BITS;
  report.matched = 0;
  report.matched_name = "none";
  report.matched_store_signed = 0;
  report.expected_trials = HMX_MODE44_TRIALS;
  report.observed_trials = trials;
  report.trials_ok = trials == HMX_MODE44_TRIALS;
  report.hmx_executed_ok = hmx_executed == 1;
  report.contract_ok = report.trials_ok && report.hmx_executed_ok;

  {
    int row = -1;
    for (int p = 0; p < 10; ++p) {
      int differs = 0;
      for (int k = 0; k < HMX_MODE44_SLOT_COUNT; ++k)
        differs |= patterns[p][k] != report.out[k];
      if (!differs) {
        row = p;
        break;
      }
    }
    if (row >= 0) {
      report.matched = 1;
      report.matched_name = names[row];
      report.matched_store_signed = row == 4;
      report.verdict = verdicts[row];
    } else if (!report.out0_expected) {
      report.verdict = HMX_MODE44_INCONCLUSIVE_TILE;
    } else if (report.out[2] != 0u && (report.out[2] & 0x8000u) != 0u) {
      /* Only a sign-preserving store can put a negative here: out[2]'s ACC
       * is positive and shape 4 is device-confirmed on a positive ACC. */
      report.verdict = HMX_MODE44_SIGNED_STORE_WITH_UNREGISTERED_ANOMALY;
    } else {
      report.verdict = HMX_MODE44_INCONCLUSIVE_UNMATCHED;
    }
  }
  report.verdict_name = hmx_mode44_verdict_name(report.verdict);
  return report;
}

#endif /* HEXATRAIN_HMX_MODE44_H */
