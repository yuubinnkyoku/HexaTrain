// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only self test for the shared identity validator (hmx_identity.h).
// It feeds adversarial identity / metadata payloads and asserts that every
// single one of them fails closed, that the correct payload passes, and
// that build id 0 keeps the documented UNAVAILABLE semantics. No FastRPC,
// no Hexagon SDK and no device are required.
//
// Build (host, no Hexagon SDK required):
//   cc -std=c11 -O2 -o probe_identity_selftest probe_identity_selftest.c

#include <stdio.h>
#include <string.h>

#include "hmx_identity.h"

static int failures = 0;

static void expect_true(const char *name, int condition) {
  if (!condition) {
    printf("FAIL %s\n", name);
    failures++;
  } else {
    printf("ok   %s\n", name);
  }
}

static void expect_int(const char *name, int actual, int want) {
  if (actual != want) {
    printf("FAIL %s: got %d want %d\n", name, actual, want);
    failures++;
  } else {
    printf("ok   %s = %d\n", name, actual);
  }
}

/* A payload that must always pass: revision 3, echoed q/mode, arch 81,
 * magic metadata, HMX clear, trials clear, matching nonzero build id. */
static void fill_valid(int identity[4], int meta[9], int q, int mode,
                       int build_id) {
  identity[0] = HMX_IDENTITY_EXPECTED_REVISION;
  identity[1] = q;
  identity[2] = mode;
  identity[3] = HMX_IDENTITY_EXPECTED_ARCH;
  memset(meta, 0, 9 * sizeof(int));
  meta[0] = HMX_IDENTITY_MAGIC;
  meta[1] = HMX_IDENTITY_EXPECTED_ARCH;
  meta[2] = build_id; /* mode 30 build id echo slot */
  meta[7] = 0;
  meta[8] = 0;
}

static hmx_identity_report_t run_case(const int identity[4],
                                      const int meta[9], int rpc_status,
                                      int q, int mode, int build_id) {
  return hmx_identity_status(rpc_status, identity, meta, q, mode, build_id);
}

/* Correct payload: every flag ok, no failure reported. */
static void test_valid_payload_passes(void) {
  int identity[4];
  int meta[9];
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  const hmx_identity_report_t report =
      run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("valid.all_ok", report.all_ok, 1);
  expect_int("valid.first_failure", report.first_failure,
             HMX_IDENTITY_FAILURE_NONE);
  expect_int("valid.build_id_status", report.build_id_status,
             HMX_BUILD_ID_MATCH);
}

/* build id 0 on the DSP (legacy skel) or 0 on the host (build without the
 * define) must stay UNAVAILABLE and must not fail identity. */
static void test_build_id_unavailable_semantics(void) {
  int identity[4];
  int meta[9];

  fill_valid(identity, meta, 6, HMX_IDENTITY_MODE, 0);
  hmx_identity_report_t report =
      run_case(identity, meta, 0, 6, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("build_id.dsp_zero.all_ok", report.all_ok, 1);
  expect_int("build_id.dsp_zero.status", report.build_id_status,
             HMX_BUILD_ID_UNAVAILABLE);

  fill_valid(identity, meta, 6, HMX_IDENTITY_MODE, 0x11111111);
  report = run_case(identity, meta, 0, 6, HMX_IDENTITY_MODE, 0);
  expect_int("build_id.host_zero.all_ok", report.all_ok, 1);
  expect_int("build_id.host_zero.status", report.build_id_status,
             HMX_BUILD_ID_UNAVAILABLE);

  fill_valid(identity, meta, 6, HMX_IDENTITY_MODE, 0);
  report = run_case(identity, meta, 0, 6, HMX_IDENTITY_MODE, 0);
  expect_int("build_id.both_zero.all_ok", report.all_ok, 1);
  expect_int("build_id.both_zero.status", report.build_id_status,
             HMX_BUILD_ID_UNAVAILABLE);
}

/* Nonzero build id mismatch fails closed. */
static void test_build_id_mismatch_fails(void) {
  int identity[4];
  int meta[9];
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0xaaaa0001);
  const hmx_identity_report_t report =
      run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0xbbbb0002);
  expect_int("build_id.mismatch.all_ok", report.all_ok, 0);
  expect_int("build_id.mismatch.first_failure", report.first_failure,
             HMX_IDENTITY_FAILURE_BUILD_ID);
  expect_int("build_id.mismatch.flag", report.build_id_ok, 0);
}

/* One adversarial payload per individual constraint: every case must fail,
 * and must be attributed to its own flag (no masking of the first failure). */
static void test_adversarial_payloads(void) {
  int identity[4];
  int meta[9];
  hmx_identity_report_t report;

  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);

  /* FastRPC transport status non-zero. */
  report = run_case(identity, meta, 14006, 10, HMX_IDENTITY_MODE,
                    0x20261020);
  expect_int("adv.rpc_status.all_ok", report.all_ok, 0);
  expect_int("adv.rpc_status.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_STATUS);

  /* Stale skel: wrong DSP-side revision. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  identity[0] = 2;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.revision.all_ok", report.all_ok, 0);
  expect_int("adv.revision.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_REVISION);

  /* DSP echoed a different q. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  identity[1] = 6;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.q.all_ok", report.all_ok, 0);
  expect_int("adv.q.flag", report.first_failure, HMX_IDENTITY_FAILURE_Q);

  /* DSP echoed a different mode. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  identity[2] = 29;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.mode.all_ok", report.all_ok, 0);
  expect_int("adv.mode.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_MODE);

  /* Non-V81 DSP architecture echo. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  identity[3] = 80;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.arch.all_ok", report.all_ok, 0);
  expect_int("adv.arch.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_ARCH);

  /* Wrong metadata magic (stale or foreign skel). */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  meta[0] = 0x484d5830;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.meta_magic.all_ok", report.all_ok, 0);
  expect_int("adv.meta_magic.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_META_MAGIC);

  /* Wrong metadata architecture. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  meta[1] = 80;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.meta_arch.all_ok", report.all_ok, 0);
  expect_int("adv.meta_arch.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_META_ARCH);

  /* HMX executed flag set during identity mode. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  meta[7] = 1;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.meta_hmx.all_ok", report.all_ok, 0);
  expect_int("adv.meta_hmx.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_META_HMX);

  /* Trial count set during identity mode. */
  fill_valid(identity, meta, 10, HMX_IDENTITY_MODE, 0x20261020);
  meta[8] = 99;
  report = run_case(identity, meta, 0, 10, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("adv.meta_trials.all_ok", report.all_ok, 0);
  expect_int("adv.meta_trials.flag", report.first_failure,
             HMX_IDENTITY_FAILURE_META_TRIALS);
}

/* Multiple simultaneous failures: all_ok is 0, first_failure is the
 * earliest flag, and every individual flag reports itself. */
static void test_multiple_failures(void) {
  int identity[4];
  int meta[9];
  fill_valid(identity, meta, 8, HMX_IDENTITY_MODE, 0x20261020);
  identity[0] = 1;    /* revision */
  identity[1] = 9;    /* q */
  meta[7] = 1;        /* hmx executed */
  meta[8] = 5;        /* trials */
  const hmx_identity_report_t report =
      run_case(identity, meta, 0, 8, HMX_IDENTITY_MODE, 0x20261020);
  expect_int("multi.all_ok", report.all_ok, 0);
  expect_int("multi.first_failure", report.first_failure,
             HMX_IDENTITY_FAILURE_REVISION);
  expect_int("multi.flag_revision",
             hmx_identity_flag_failed(&report, HMX_IDENTITY_FAILURE_REVISION), 1);
  expect_int("multi.flag_q",
             hmx_identity_flag_failed(&report, HMX_IDENTITY_FAILURE_Q), 1);
  expect_int("multi.flag_meta_hmx",
             hmx_identity_flag_failed(&report, HMX_IDENTITY_FAILURE_META_HMX), 1);
  expect_int("multi.flag_meta_trials",
             hmx_identity_flag_failed(&report, HMX_IDENTITY_FAILURE_META_TRIALS), 1);
  expect_int("multi.flag_arch",
             hmx_identity_flag_failed(&report, HMX_IDENTITY_FAILURE_ARCH), 0);
  expect_true("multi.flag_names_stable",
              strcmp(hmx_identity_failure_name(HMX_IDENTITY_FAILURE_META_HMX),
                     "META_HMX_NOT_CLEAR") == 0 &&
                  strcmp(hmx_identity_failure_name(
                             HMX_IDENTITY_FAILURE_META_TRIALS),
                         "META_TRIALS_NOT_CLEAR") == 0);
}

int main(void) {
  test_valid_payload_passes();
  test_build_id_unavailable_semantics();
  test_build_id_mismatch_fails();
  test_adversarial_payloads();
  test_multiple_failures();
  if (failures) {
    printf("FAILED: %d check(s)\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
