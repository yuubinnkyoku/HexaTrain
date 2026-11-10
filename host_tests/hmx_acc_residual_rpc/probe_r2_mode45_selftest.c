// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only self test for the frozen R2 S1/S2 (mode 45) contract and decision
// procedure in hmx_mode45.h. The fixture, the 11 frozen slots, the four Rs
// values, the execution counts and the decision ORDER come from
// docs/hmx-v81-r2-device-prereg.md sections 6-8 (and the matching sections
// 3-5 of docs/hmx-v81-r2-device-implementation-plan.md).
//
// It asserts: (1) the compile-time pins, (2) the baseline PASS verdict,
// (3) that every single-field corruption fails closed as
// R2_HOST_CONTRACT_FAILED, and (4) that the remaining decision steps produce
// exactly the pre-registered verdicts in the pre-registered order.
//
// No SDK, no FastRPC, no device. Build (host):
//   gcc -std=c11 -O2 -Wall -Wextra -Werror -I host_tests/hmx_acc_residual_rpc
//       host_tests/hmx_acc_residual_rpc/probe_r2_mode45_selftest.c
//       -o build/tmp/probe_r2_mode45_selftest.exe

#include <stdio.h>
#include <string.h>

#include "hmx_mode45.h"

static int checks = 0;
static int failures = 0;

static void expect_true(const char *name, int condition) {
  ++checks;
  if (!condition) {
    printf("FAIL %s\n", name);
    ++failures;
  } else {
    printf("ok   %s\n", name);
  }
}

static void expect_int(const char *name, int actual, int want) {
  ++checks;
  if (actual != want) {
    printf("FAIL %s: got %d want %d\n", name, actual, want);
    ++failures;
  } else {
    printf("ok   %s = %d\n", name, actual);
  }
}

static void expect_verdict(const char *name, const hmx_mode45_report_t *r,
                           hmx_mode45_verdict_t want) {
  ++checks;
  if (r->verdict != want) {
    printf("FAIL %s: got %s want %s (cause=%s)\n", name,
           hmx_mode45_verdict_name(r->verdict), hmx_mode45_verdict_name(want),
           r->cause ? r->cause : "(null)");
    ++failures;
  } else {
    printf("ok   %s = %s\n", name, hmx_mode45_verdict_name(r->verdict));
  }
}

static void expect_cause(const char *name, const hmx_mode45_report_t *r,
                         const char *want) {
  ++checks;
  if (r->cause == 0 || strcmp(r->cause, want) != 0) {
    printf("FAIL %s: got cause=%s want %s\n", name,
           r->cause ? r->cause : "(null)", want);
    ++failures;
  } else {
    printf("ok   %s = %s\n", name, want);
  }
}

/* ------------------------------------------------------------------ */
/* Section 1: compile-time pins (pre-registration sections 5-7).      */
/* ------------------------------------------------------------------ */

/* Frozen 11-slot pattern, pre-registration section 6 order. */
static const int kFrozenSlots[HMX_MODE45_RESULT_SLOTS] = {
    0x5420, 0x5020, 0x5420, 0x0000, 0x0000, 0x0000,
    0x5420, 0x0000, 0x5420, 0xD420, 0x0000};

_Static_assert(HMX_MODE45_RESULT_SLOTS == 11, "pin.result_slots");
_Static_assert(HMX_MODE45_METADATA_LEN == 32, "pin.metadata_len");
_Static_assert(HMX_MODE45_TRIALS == 2, "pin.trials");

/* Registered Rs values, in execution order. */
_Static_assert(HMX_MODE45_RS_S1_FIRST == 0x0001u, "pin.rs_s1_first");
_Static_assert(HMX_MODE45_RS_S1_SECOND == 0x2000u, "pin.rs_s1_second");
_Static_assert(HMX_MODE45_RS_S2_FIRST == 0x0001u, "pin.rs_s2_first");
_Static_assert(HMX_MODE45_RS_S2_SECOND == 0x1000u, "pin.rs_s2_second");

/* Registered execution counts: 2 / 2 / 4 / 4 / 3 / 2. */
_Static_assert(HMX_MODE45_MXCLRACC_COUNT == 2, "pin.mxclracc");
_Static_assert(HMX_MODE45_MMA_COUNT == 2, "pin.mma");
_Static_assert(HMX_MODE45_CONVERT_COUNT == 4, "pin.convert");
_Static_assert(HMX_MODE45_STORE_COUNT == 4, "pin.store");
_Static_assert(HMX_MODE45_BIAS_LOADS == 3, "pin.bias_loads");
_Static_assert(HMX_MODE45_RECON_LOOPS == 2, "pin.recon_loops");

/* VTCM map (bytes from the 12 KiB reserve base). */
_Static_assert(HMX_MODE45_ACT_OFFSET == 0, "pin.vtcm_act");
_Static_assert(HMX_MODE45_WGT_OFFSET == 2048, "pin.vtcm_wgt");
_Static_assert(HMX_MODE45_POS_OFFSET == 4096, "pin.vtcm_pos");
_Static_assert(HMX_MODE45_NEG_OFFSET == 6144, "pin.vtcm_neg");
_Static_assert(HMX_MODE45_RECON_OFFSET == 8192, "pin.vtcm_recon");
_Static_assert(HMX_MODE45_BIAS0_OFFSET == 10240, "pin.vtcm_bias0");
_Static_assert(HMX_MODE45_BIAS1_OFFSET == 10496, "pin.vtcm_bias1");
_Static_assert(HMX_MODE45_BIAS2_OFFSET == 10752, "pin.vtcm_bias2");
_Static_assert(HMX_MODE45_VTCM_USED == 11008, "pin.vtcm_used");

/* ------------------------------------------------------------------ */
/* Fixture: a contract that must always classify as supported.        */
/* ------------------------------------------------------------------ */

/* Arbitrary nonzero fixture id: the self-test checks the compare, not a real
 * build. Device runs use the stamped HEAD-derived id passed with
 * -DHMX_PROBE_BUILD_ID. */
#define EXPECTED_BUILD_ID 0x1234abcd

typedef struct fixture {
  int result[HMX_MODE45_RESULT_SLOTS];
  int meta[HMX_MODE45_METADATA_LEN];
} fixture_t;

static void fill_valid(fixture_t *f) {
  int i;
  memcpy(f->result, kFrozenSlots, sizeof(kFrozenSlots));
  for (i = 0; i < HMX_MODE45_METADATA_LEN; ++i) f->meta[i] = 0;
  f->meta[0] = 0x484d5831; /* magic  HMXM1 */
  f->meta[1] = 81;         /* arch   V81   */
  f->meta[2] = 12288;      /* vtcm bytes    */
  f->meta[3] = 0;          /* kernel us     */
  f->meta[4] = 0;          /* power         */
  f->meta[5] = 0;          /* lock A        */
  f->meta[6] = 0;          /* lock B        */
  f->meta[7] = 1;          /* hmx executed  */
  f->meta[8] = 2;          /* trials        */
  f->meta[9] = 1;          /* schema        */
  f->meta[10] = 2;         /* mxclracc      */
  f->meta[11] = 2;         /* mma           */
  f->meta[12] = 4;         /* convert       */
  f->meta[13] = 4;         /* store         */
  f->meta[14] = 3;         /* bias loads    */
  f->meta[15] = 2;         /* recon loops   */
  f->meta[16] = 0x0001;    /* Rs s1 first   */
  f->meta[17] = 0x2000;    /* Rs s1 second  */
  f->meta[18] = 0x0001;    /* Rs s2 first   */
  f->meta[19] = 0x1000;    /* Rs s2 second  */
  f->meta[20] = 7;         /* bias echo     */
  f->meta[21] = 0;         /* sentinel hits */
  f->meta[22] = 1;         /* region layout */
  f->meta[23] = 1;         /* recon source  */
  f->meta[24] = EXPECTED_BUILD_ID;
  f->meta[25] = 0;         /* hvx lock      */
  f->meta[26] = 2;         /* pos writes    */
  f->meta[27] = 2;         /* neg writes    */
  f->meta[28] = 1;         /* pos unchanged */
  f->meta[29] = 1;         /* neg unchanged */
  f->meta[30] = 32;        /* recon vectors */
  f->meta[31] = 0;         /* reserved      */
}

static hmx_mode45_report_t run_valid(const fixture_t *f) {
  return hmx_mode45_classify(f->result, HMX_MODE45_RESULT_SLOTS, f->meta,
                             HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);
}

/* Every fail-closed mutation must end as R2_HOST_CONTRACT_FAILED with
 * contract.all_ok == 0. */
static void expect_fail_closed(const char *name, const fixture_t *f,
                               int result_len, int meta_len,
                               int expected_build_id) {
  const hmx_mode45_report_t r = hmx_mode45_classify(
      f->result, result_len, f->meta, meta_len, expected_build_id);
  ++checks;
  if (r.contract.all_ok != 0 ||
      r.verdict != HMX_MODE45_HOST_CONTRACT_FAILED) {
    printf("FAIL %s: all_ok=%d verdict=%s cause=%s\n", name,
           r.contract.all_ok, hmx_mode45_verdict_name(r.verdict),
           r.cause ? r.cause : "(null)");
    ++failures;
  } else {
    printf("ok   %s = R2_HOST_CONTRACT_FAILED\n", name);
  }
}

/* ------------------------------------------------------------------ */
/* Section 1: pins with named runtime output.                         */
/* ------------------------------------------------------------------ */

static void test_pins(void) {
  fixture_t f;
  fill_valid(&f);
  expect_true("pins.frozen_slots",
              memcmp(f.result, kFrozenSlots, sizeof(kFrozenSlots)) == 0);
  expect_int("pins.metadata_len", HMX_MODE45_METADATA_LEN, 32);
  expect_int("pins.result_slots", HMX_MODE45_RESULT_SLOTS, 11);
  expect_int("pins.trials", HMX_MODE45_TRIALS, 2);
  expect_true("pins.rs_values",
              HMX_MODE45_RS_S1_FIRST == 0x0001u &&
                  HMX_MODE45_RS_S1_SECOND == 0x2000u &&
                  HMX_MODE45_RS_S2_FIRST == 0x0001u &&
                  HMX_MODE45_RS_S2_SECOND == 0x1000u);
  expect_true("pins.counts",
              HMX_MODE45_MXCLRACC_COUNT == 2 && HMX_MODE45_MMA_COUNT == 2 &&
                  HMX_MODE45_CONVERT_COUNT == 4 &&
                  HMX_MODE45_STORE_COUNT == 4 &&
                  HMX_MODE45_BIAS_LOADS == 3 &&
                  HMX_MODE45_RECON_LOOPS == 2);
  expect_true("pins.vtcm_offsets",
              HMX_MODE45_ACT_OFFSET == 0 && HMX_MODE45_WGT_OFFSET == 2048 &&
                  HMX_MODE45_POS_OFFSET == 4096 &&
                  HMX_MODE45_NEG_OFFSET == 6144 &&
                  HMX_MODE45_RECON_OFFSET == 8192 &&
                  HMX_MODE45_BIAS0_OFFSET == 10240 &&
                  HMX_MODE45_BIAS1_OFFSET == 10496 &&
                  HMX_MODE45_BIAS2_OFFSET == 10752 &&
                  HMX_MODE45_VTCM_USED == 11008);
}

/* ------------------------------------------------------------------ */
/* Section 2: baseline verdict.                                       */
/* ------------------------------------------------------------------ */

static void test_baseline(void) {
  fixture_t f;
  hmx_mode45_report_t r;
  fill_valid(&f);
  r = run_valid(&f);
  expect_int("baseline.all_ok", r.contract.all_ok, 1);
  expect_verdict("baseline.verdict", &r, HMX_MODE45_ACC_REUSE_SUPPORTED);
  expect_cause("baseline.cause", &r, "all_registered_cells_match");
}

/* ------------------------------------------------------------------ */
/* Section 3: fail-closed mutations (each = R2_HOST_CONTRACT_FAILED). */
/* ------------------------------------------------------------------ */

static void test_fail_closed(void) {
  fixture_t f;
  int i;
  char name[96];
  static const char *meta_names[6] = {
      "mxclracc", "mma", "convert", "store", "bias_loads", "recon_loops"};
  static const char *rs_names[4] = {"rs_s1_first", "rs_s1_second",
                                    "rs_s2_first", "rs_s2_second"};

  /* result_len 10. */
  fill_valid(&f);
  expect_fail_closed("fail.result_len_10", &f, 10,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  /* Every slot out of the zero-extended 16-bit range, both directions. */
  for (i = 0; i < HMX_MODE45_RESULT_SLOTS; ++i) {
    fill_valid(&f);
    f.result[i] = -1;
    snprintf(name, sizeof(name), "fail.slot%d_negative", i);
    expect_fail_closed(name, &f, HMX_MODE45_RESULT_SLOTS,
                       HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);
    fill_valid(&f);
    f.result[i] = 0x10000;
    snprintf(name, sizeof(name), "fail.slot%d_overflow", i);
    expect_fail_closed(name, &f, HMX_MODE45_RESULT_SLOTS,
                       HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);
  }

  /* Store sentinel word in a result slot. */
  fill_valid(&f);
  f.result[0] = HMX_MODE45_SENTINEL_WORD;
  expect_fail_closed("fail.sentinel_slot", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  /* Sentinel reported changed in metadata. */
  fill_valid(&f);
  f.meta[HMX_MODE45_META_SENTINEL_HITS] = 1;
  expect_fail_closed("fail.sentinel_meta", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  /* Execution counts, each off by one (metadata[10..15]). */
  for (i = 0; i < 6; ++i) {
    fill_valid(&f);
    f.meta[HMX_MODE45_META_MXCLRACC + i] += 1;
    snprintf(name, sizeof(name), "fail.count_%s", meta_names[i]);
    expect_fail_closed(name, &f, HMX_MODE45_RESULT_SLOTS,
                       HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);
  }

  /* The four Rs echo values, each changed (metadata[16..19]). */
  for (i = 0; i < 4; ++i) {
    fill_valid(&f);
    f.meta[HMX_MODE45_META_RS_S1_FIRST + i] += 1;
    snprintf(name, sizeof(name), "fail.%s_changed", rs_names[i]);
    expect_fail_closed(name, &f, HMX_MODE45_RESULT_SLOTS,
                       HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);
  }

  /* Remaining scalar metadata pins. */
  fill_valid(&f);
  f.meta[HMX_MODE45_META_BIAS_ECHO] = 6;
  expect_fail_closed("fail.bias_echo", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_REGION_LAYOUT] = 0;
  expect_fail_closed("fail.region_layout", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_RECON_SOURCE] = 0;
  expect_fail_closed("fail.recon_source", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_HVX_LOCK] = 1;
  expect_fail_closed("fail.hvx_lock", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_HVX_POS_UNCHANGED] = 0;
  expect_fail_closed("fail.pos_unchanged", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_HVX_NEG_UNCHANGED] = 0;
  expect_fail_closed("fail.neg_unchanged", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_RECON_VECTORS] = 31;
  expect_fail_closed("fail.recon_vectors", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[HMX_MODE45_META_RESERVED] = 1;
  expect_fail_closed("fail.reserved", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[8] = 1; /* trials must be 2 */
  expect_fail_closed("fail.trials", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  fill_valid(&f);
  f.meta[7] = 0; /* hmx_executed must be 1 */
  expect_fail_closed("fail.hmx_executed", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID);

  /* Nonzero expected build id that does not match the DSP echo. */
  fill_valid(&f);
  expect_fail_closed("fail.build_id_mismatch", &f, HMX_MODE45_RESULT_SLOTS,
                     HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID + 1);

  /* meta_len 31 (< HMX_MODE45_METADATA_LEN). */
  fill_valid(&f);
  expect_fail_closed("fail.meta_len_31", &f, HMX_MODE45_RESULT_SLOTS, 31,
                     EXPECTED_BUILD_ID);
}

/* ------------------------------------------------------------------ */
/* Section 4: decision-procedure order and verdicts.                  */
/* ------------------------------------------------------------------ */

static void test_decision(void) {
  fixture_t f;
  hmx_mode45_report_t r;

  /* Rounding alternatives win first (step 3). */
  fill_valid(&f);
  f.result[1] = 0x5021;
  r = run_valid(&f);
  expect_int("decision.rounding_s1.all_ok", r.contract.all_ok, 1);
  expect_verdict("decision.rounding_s1", &r, HMX_MODE45_ROUNDING_MISMATCH);

  fill_valid(&f);
  f.result[2] = 0x5421;
  r = run_valid(&f);
  expect_verdict("decision.rounding_s2_pos", &r, HMX_MODE45_ROUNDING_MISMATCH);

  fill_valid(&f);
  f.result[6] = 0x5421;
  r = run_valid(&f);
  expect_verdict("decision.rounding_s2_neg_wins_over_s1s2",
                 &r, HMX_MODE45_ROUNDING_MISMATCH);

  /* Region-write mismatch is decision step 2, NOT a contract failure. */
  fill_valid(&f);
  f.meta[HMX_MODE45_META_POS_WRITES] = 1;
  r = run_valid(&f);
  expect_int("decision.pos_writes.all_ok", r.contract.all_ok, 1);
  expect_verdict("decision.pos_writes", &r, HMX_MODE45_STORE_INTERFERENCE);
  expect_true("decision.pos_writes.not_host_contract_failed",
              r.verdict != HMX_MODE45_HOST_CONTRACT_FAILED);
  expect_cause("decision.pos_writes.cause", &r, "region_write_count");

  fill_valid(&f);
  f.meta[HMX_MODE45_META_NEG_WRITES] = 1;
  r = run_valid(&f);
  expect_int("decision.neg_writes.all_ok", r.contract.all_ok, 1);
  expect_verdict("decision.neg_writes", &r, HMX_MODE45_STORE_INTERFERENCE);
  expect_true("decision.neg_writes.not_host_contract_failed",
              r.verdict != HMX_MODE45_HOST_CONTRACT_FAILED);
  expect_cause("decision.neg_writes.cause", &r, "region_write_count");

  /* S1 first-store anchor failure (step 4). */
  fill_valid(&f);
  f.result[0] = 0x0000;
  r = run_valid(&f);
  expect_verdict("decision.s1_first_zero", &r, HMX_MODE45_UNEXPECTED_OUTPUT);
  expect_cause("decision.s1_first_zero.cause", &r, "s1_first_anchor");

  /* S1 second-store control outcomes (step 4). */
  fill_valid(&f);
  f.result[1] = 0x0000;
  r = run_valid(&f);
  expect_verdict("decision.s1_second_zero", &r, HMX_MODE45_CAUSE_INCONCLUSIVE);

  fill_valid(&f);
  f.result[1] = 0x5420;
  r = run_valid(&f);
  expect_verdict("decision.s1_second_set0", &r, HMX_MODE45_CAUSE_INCONCLUSIVE);

  fill_valid(&f);
  f.result[1] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s1_second_other", &r,
                 HMX_MODE45_UNEXPECTED_OUTPUT);

  /* S2 first-store overwrite signature (step 5). */
  fill_valid(&f);
  f.result[2] = 0x0000;
  f.result[3] = 0x5420;
  r = run_valid(&f);
  expect_verdict("decision.store_overwrite", &r,
                 HMX_MODE45_STORE_INTERFERENCE);
  expect_cause("decision.store_overwrite.cause", &r,
               "first_store_overwritten_by_second");

  /* S2 sign split (step 6). */
  fill_valid(&f);
  f.result[2] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_pos_other", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[3] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_pos_neg_lane", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[5] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_neg_s0_nonzero", &r,
                 HMX_MODE45_CAUSE_INCONCLUSIVE);

  fill_valid(&f);
  f.result[6] = 0x0000;
  r = run_valid(&f);
  expect_verdict("decision.s2_neg_missing", &r, HMX_MODE45_CAUSE_INCONCLUSIVE);

  fill_valid(&f);
  f.result[6] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_neg_other", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[4] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_lane4", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[7] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.s2_lane7", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[8] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.recon_slot8", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[9] = 0x5420;
  r = run_valid(&f);
  expect_verdict("decision.recon_slot9", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  fill_valid(&f);
  f.result[10] = 0x0001;
  r = run_valid(&f);
  expect_verdict("decision.recon_slot10", &r, HMX_MODE45_UNEXPECTED_OUTPUT);

  /* Build-id gate: 0 = unchecked, matching nonzero passes, mismatch fails. */
  fill_valid(&f);
  r = hmx_mode45_classify(f.result, HMX_MODE45_RESULT_SLOTS, f.meta,
                          HMX_MODE45_METADATA_LEN, 0);
  expect_int("decision.build_id_unchecked.all_ok", r.contract.all_ok, 1);
  expect_verdict("decision.build_id_unchecked", &r,
                 HMX_MODE45_ACC_REUSE_SUPPORTED);

  r = run_valid(&f);
  expect_int("decision.build_id_match.all_ok", r.contract.all_ok, 1);
  expect_verdict("decision.build_id_match", &r,
                 HMX_MODE45_ACC_REUSE_SUPPORTED);

  r = hmx_mode45_classify(f.result, HMX_MODE45_RESULT_SLOTS, f.meta,
                          HMX_MODE45_METADATA_LEN, EXPECTED_BUILD_ID + 1);
  expect_int("decision.build_id_mismatch.all_ok", r.contract.all_ok, 0);
  expect_verdict("decision.build_id_mismatch", &r,
                 HMX_MODE45_HOST_CONTRACT_FAILED);
}

int main(void) {
  test_pins();
  test_baseline();
  test_fail_closed();
  test_decision();
  if (failures) {
    printf("mode45_selftest checks=%d fail=%d\n", checks, failures);
    printf("mode45_selftest=FAIL\n");
    return 1;
  }
  printf("mode45_selftest checks=%d fail=0\n", checks);
  printf("mode45_selftest=PASS\n");
  return 0;
}
