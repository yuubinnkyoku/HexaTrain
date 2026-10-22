// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Shared mode-43 (E-3C1R) contract: one plain trial, four corrected-packing
// store slots, and the pre-registered classification. Included by
// probe_host.c, probe_dsp.c and the CPU-only hmx_prm_layout_sim.c so the
// host gate and its tests cannot drift.
//
// Mode 43 reuses the exact mode-42 fixture, packet, bias=0 and scale=64.
// The ONLY intended data change is that the plain Ahi/Whi cells are packed
// at the PRM pair-interleaved VTCM offsets instead of row-major. One MMA
// packet, one convert, one store; the four readback slots come from that
// ONE converted plain tile:
//
//   results[0] = out[0]  -> device cell (s0, o0)
//   results[1] = out[1]  -> device cell (s1, o0)
//   results[2] = out[2]  -> device cell (s0, o1)
//   results[3] = out[32] -> device cell (s0, o16)
//
// Pre-registered expectations (fixed before the run; do not retune):
//   out[0]  = 0x5400
//   out[1]  = 0xd400 (signed store) or 0x0000 (clamp)
//   out[2]  = 0x5400
//   out[32] = 0x55e3
//
// Verdict policy: all three positive checks (out[0], out[2], out[32]) must
// pass before slot 1 is interpreted. A zero at out[1] only records the
// observed zero-store behavior; it does not locate the clamp (ACC, convert
// or store). 0xd401 is NOT a mode-43 candidate: mode 43 is plain-only, so
// the B3 1-ulp variant belongs to mode 42 only.

#ifndef HEXATRAIN_HMX_MODE43_H
#define HEXATRAIN_HMX_MODE43_H

#include <stdint.h>

#define HMX_MODE43_TRIALS 1
#define HMX_MODE43_SLOT_COUNT 4

#define HMX_MODE43_OUT0_BITS 0x5400u
#define HMX_MODE43_OUT1_SIGNED_BITS 0xd400u
#define HMX_MODE43_OUT1_CLAMP_BITS 0x0000u
#define HMX_MODE43_OUT2_BITS 0x5400u
#define HMX_MODE43_OUT32_BITS 0x55e3u

typedef enum hmx_mode43_verdict {
  HMX_MODE43_LAYOUT_CONFIRMED_SIGNED = 0,
  HMX_MODE43_LAYOUT_CONFIRMED_CLAMP = 1,
  HMX_MODE43_RESIDUAL_MISMATCH = 2
} hmx_mode43_verdict_t;

typedef struct hmx_mode43_report {
  /* Raw slot facts. */
  uint16_t out0;
  uint16_t out1;
  uint16_t out2;
  uint16_t out32;
  int out0_ok;
  int out2_ok;
  int out32_ok;
  int slot1_signed;
  int slot1_clamp;
  /* Positive checks gate slot-1 interpretation. */
  int positive_checks; /* 0..3 */
  int positive_ok;     /* all three positive checks passed */
  /* Host contract: mode 43 must be exactly one executed trial. */
  int expected_trials;
  int observed_trials;
  int trials_ok;
  int hmx_executed_ok;
  hmx_mode43_verdict_t verdict;
  const char *verdict_name;
} hmx_mode43_report_t;

static inline const char *hmx_mode43_verdict_name(hmx_mode43_verdict_t verdict) {
  switch (verdict) {
    case HMX_MODE43_LAYOUT_CONFIRMED_SIGNED:
      return "E3C1R_LAYOUT_CONFIRMED_SIGNED";
    case HMX_MODE43_LAYOUT_CONFIRMED_CLAMP:
      return "E3C1R_LAYOUT_CONFIRMED_CLAMP";
    case HMX_MODE43_RESIDUAL_MISMATCH:
      return "E3C1R_RESIDUAL_MISMATCH";
  }
  return "E3C1R_RESIDUAL_MISMATCH";
}

/* Pure classifier: no I/O and no device state. `trials` and
 * `hmx_executed` are metadata[8]/metadata[7]. The classification is only
 * meaningful when the host contract passes; the caller must fail closed
 * (and must not consume the verdict) when trials_ok/hmx_executed_ok are 0. */
static inline hmx_mode43_report_t hmx_mode43_classify(
    const int results[HMX_MODE43_SLOT_COUNT], int trials, int hmx_executed) {
  hmx_mode43_report_t report;
  report.out0 = (uint16_t)results[0];
  report.out1 = (uint16_t)results[1];
  report.out2 = (uint16_t)results[2];
  report.out32 = (uint16_t)results[3];
  report.out0_ok = report.out0 == HMX_MODE43_OUT0_BITS;
  report.out2_ok = report.out2 == HMX_MODE43_OUT2_BITS;
  report.out32_ok = report.out32 == HMX_MODE43_OUT32_BITS;
  report.slot1_signed = report.out1 == HMX_MODE43_OUT1_SIGNED_BITS;
  report.slot1_clamp = report.out1 == HMX_MODE43_OUT1_CLAMP_BITS;
  report.positive_checks =
      (report.out0_ok ? 1 : 0) + (report.out2_ok ? 1 : 0) +
      (report.out32_ok ? 1 : 0);
  report.positive_ok = report.positive_checks == 3;
  report.expected_trials = HMX_MODE43_TRIALS;
  report.observed_trials = trials;
  report.trials_ok = trials == HMX_MODE43_TRIALS;
  report.hmx_executed_ok = hmx_executed == 1;
  if (report.positive_ok && report.slot1_signed) {
    report.verdict = HMX_MODE43_LAYOUT_CONFIRMED_SIGNED;
  } else if (report.positive_ok && report.slot1_clamp) {
    report.verdict = HMX_MODE43_LAYOUT_CONFIRMED_CLAMP;
  } else {
    report.verdict = HMX_MODE43_RESIDUAL_MISMATCH;
  }
  report.verdict_name = hmx_mode43_verdict_name(report.verdict);
  return report;
}

#endif /* HEXATRAIN_HMX_MODE43_H */
