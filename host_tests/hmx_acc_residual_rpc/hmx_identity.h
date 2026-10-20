// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Shared, dependency-free identity validation for the HMX residual probe.
//
// Included by the DSP skel (probe_dsp.c), the host runner (probe_host.c)
// and the CPU-only self test (probe_identity_selftest.c) so that the
// expected identity echo values and the build id cannot drift between the
// three. The header only uses plain C types: no stdint, no stdio.
//
// Build id protocol (ABI-compatible, mode 30 only):
//   The build system passes the same
//     -DHMX_PROBE_BUILD_ID=0x<32-bit int>
//   to the DSP skel build and the host runner build. In identity mode
//   (mode 30) the DSP echoes the value through metadata[2], which is
//   unused in mode 30 (normal runs keep their VTCM-bytes meaning there).
//   build id 0 means "unknown / legacy build": the host then prints
//   BUILD_ID=UNAVAILABLE and does NOT fail identity on it. Only a
//   mismatch where BOTH sides are nonzero fails the identity check.

#ifndef HEXATRAIN_HMX_IDENTITY_H
#define HEXATRAIN_HMX_IDENTITY_H

#ifndef HMX_PROBE_BUILD_ID
#define HMX_PROBE_BUILD_ID 0 /* legacy / unknown unless the build sets it */
#endif

enum {
  HMX_IDENTITY_MAGIC = 0x484d5831,        /* metadata[0] on every run */
  HMX_IDENTITY_EXPECTED_REVISION = 3,     /* results[0] echo */
  HMX_IDENTITY_EXPECTED_ARCH = 81,        /* results[3] / metadata[1] echo */
  HMX_IDENTITY_MODE = 30,                 /* identity mode number */
  HMX_IDENTITY_BUILD_ID_SLOT = 2          /* metadata[2] in mode 30 only */
};

/* Individual failure flags. Values are contiguous and start at 1 so the
 * host can iterate over every failing flag with hmx_identity_flag_failed().
 * The names are printed verbatim as identity_fail=<name>. */
typedef enum hmx_identity_failure {
  HMX_IDENTITY_FAILURE_NONE = 0,
  HMX_IDENTITY_FAILURE_STATUS = 1,        /* FastRPC pdu status != 0 */
  HMX_IDENTITY_FAILURE_REVISION = 2,      /* results[0] != expected revision */
  HMX_IDENTITY_FAILURE_Q = 3,             /* results[1] != requested q */
  HMX_IDENTITY_FAILURE_MODE = 4,          /* results[2] != expected mode */
  HMX_IDENTITY_FAILURE_ARCH = 5,          /* results[3] != 81 */
  HMX_IDENTITY_FAILURE_META_MAGIC = 6,    /* metadata[0] != HMX magic */
  HMX_IDENTITY_FAILURE_META_ARCH = 7,     /* metadata[1] != 81 */
  HMX_IDENTITY_FAILURE_META_HMX = 8,      /* metadata[7] != 0 (HMX off) */
  HMX_IDENTITY_FAILURE_META_TRIALS = 9,   /* metadata[8] != 0 (no trials) */
  HMX_IDENTITY_FAILURE_BUILD_ID = 10,     /* nonzero metadata[2] mismatch */
  HMX_IDENTITY_FAILURE_COUNT = 11
} hmx_identity_failure_t;

/* Build id comparison state. UNAVAILABLE means one side reported 0. */
typedef enum hmx_build_id_status {
  HMX_BUILD_ID_UNAVAILABLE = 0,
  HMX_BUILD_ID_MATCH = 1,
  HMX_BUILD_ID_MISMATCH = 2
} hmx_build_id_status_t;

typedef struct hmx_identity_report {
  /* Raw observations. */
  int rpc_status;
  int revision;        /* results[0] */
  int q;               /* results[1] */
  int mode;            /* results[2] */
  int arch;            /* results[3] */
  int meta_magic;      /* metadata[0] */
  int meta_arch;       /* metadata[1] */
  int meta_build_id;   /* metadata[2], mode 30 only */
  int meta_hmx;        /* metadata[7] */
  int meta_trials;     /* metadata[8] */
  int expected_build_id;
  /* Per-flag verdicts: 1 = ok, 0 = failed. */
  int status_ok;
  int revision_ok;
  int q_ok;
  int mode_ok;
  int arch_ok;
  int meta_magic_ok;
  int meta_arch_ok;
  int meta_hmx_ok;
  int meta_trials_ok;
  int build_id_ok;     /* also 1 when the build id is unavailable */
  hmx_build_id_status_t build_id_status;
  hmx_identity_failure_t first_failure;
  int all_ok;
} hmx_identity_report_t;

static inline const char *hmx_identity_failure_name(hmx_identity_failure_t flag) {
  switch (flag) {
    case HMX_IDENTITY_FAILURE_NONE: return "NONE";
    case HMX_IDENTITY_FAILURE_STATUS: return "RPC_STATUS";
    case HMX_IDENTITY_FAILURE_REVISION: return "REVISION";
    case HMX_IDENTITY_FAILURE_Q: return "Q";
    case HMX_IDENTITY_FAILURE_MODE: return "MODE";
    case HMX_IDENTITY_FAILURE_ARCH: return "ARCH";
    case HMX_IDENTITY_FAILURE_META_MAGIC: return "META_MAGIC";
    case HMX_IDENTITY_FAILURE_META_ARCH: return "META_ARCH";
    case HMX_IDENTITY_FAILURE_META_HMX: return "META_HMX_NOT_CLEAR";
    case HMX_IDENTITY_FAILURE_META_TRIALS: return "META_TRIALS_NOT_CLEAR";
    case HMX_IDENTITY_FAILURE_BUILD_ID: return "BUILD_ID";
    case HMX_IDENTITY_FAILURE_COUNT: break;
  }
  return "UNKNOWN";
}

static inline int hmx_identity_flag_failed(const hmx_identity_report_t *report,
                                           hmx_identity_failure_t flag) {
  switch (flag) {
    case HMX_IDENTITY_FAILURE_NONE: return 0;
    case HMX_IDENTITY_FAILURE_STATUS: return !report->status_ok;
    case HMX_IDENTITY_FAILURE_REVISION: return !report->revision_ok;
    case HMX_IDENTITY_FAILURE_Q: return !report->q_ok;
    case HMX_IDENTITY_FAILURE_MODE: return !report->mode_ok;
    case HMX_IDENTITY_FAILURE_ARCH: return !report->arch_ok;
    case HMX_IDENTITY_FAILURE_META_MAGIC: return !report->meta_magic_ok;
    case HMX_IDENTITY_FAILURE_META_ARCH: return !report->meta_arch_ok;
    case HMX_IDENTITY_FAILURE_META_HMX: return !report->meta_hmx_ok;
    case HMX_IDENTITY_FAILURE_META_TRIALS: return !report->meta_trials_ok;
    case HMX_IDENTITY_FAILURE_BUILD_ID: return !report->build_id_ok;
    case HMX_IDENTITY_FAILURE_COUNT: break;
  }
  return 1;
}

static inline const char *hmx_build_id_status_name(hmx_build_id_status_t status) {
  switch (status) {
    case HMX_BUILD_ID_UNAVAILABLE: return "UNAVAILABLE";
    case HMX_BUILD_ID_MATCH: return "MATCH";
    case HMX_BUILD_ID_MISMATCH: return "MISMATCH";
  }
  return "UNKNOWN";
}

/* Pure validation of one identity round trip. Fail closed: every individual
 * constraint must hold, otherwise all_ok is 0 and first_failure carries the
 * earliest failing flag (in enum order, i.e. the transport first).
 *
 * Constraints:
 *   rpc_status      == 0
 *   identity[0]     == HMX_IDENTITY_EXPECTED_REVISION (3)
 *   identity[1]     == q
 *   identity[2]     == expected_mode (30 in the real host flow)
 *   identity[3]     == HMX_IDENTITY_EXPECTED_ARCH (81)
 *   meta[0]         == HMX_IDENTITY_MAGIC
 *   meta[1]         == HMX_IDENTITY_EXPECTED_ARCH
 *   meta[7]         == 0 (HMX must not have executed)
 *   meta[8]         == 0 (no trials)
 *   meta[2]         == expected_build_id, unless either side is 0
 *                     (unknown/legacy build id -> UNAVAILABLE, not a failure)
 */
static inline hmx_identity_report_t hmx_identity_status(
    int rpc_status, const int identity[4], const int meta[9],
    int q, int expected_mode, int expected_build_id) {
  hmx_identity_report_t report;
  int index;

  report.rpc_status = rpc_status;
  report.revision = identity[0];
  report.q = identity[1];
  report.mode = identity[2];
  report.arch = identity[3];
  report.meta_magic = meta[0];
  report.meta_arch = meta[1];
  report.meta_build_id = meta[HMX_IDENTITY_BUILD_ID_SLOT];
  report.meta_hmx = meta[7];
  report.meta_trials = meta[8];
  report.expected_build_id = expected_build_id;

  report.status_ok = (rpc_status == 0);
  report.revision_ok = (identity[0] == HMX_IDENTITY_EXPECTED_REVISION);
  report.q_ok = (identity[1] == q);
  report.mode_ok = (identity[2] == expected_mode);
  report.arch_ok = (identity[3] == HMX_IDENTITY_EXPECTED_ARCH);
  report.meta_magic_ok = (meta[0] == HMX_IDENTITY_MAGIC);
  report.meta_arch_ok = (meta[1] == HMX_IDENTITY_EXPECTED_ARCH);
  report.meta_hmx_ok = (meta[7] == 0);
  report.meta_trials_ok = (meta[8] == 0);

  if (expected_build_id == 0 || meta[HMX_IDENTITY_BUILD_ID_SLOT] == 0) {
    report.build_id_status = HMX_BUILD_ID_UNAVAILABLE;
    report.build_id_ok = 1;
  } else if (meta[HMX_IDENTITY_BUILD_ID_SLOT] == expected_build_id) {
    report.build_id_status = HMX_BUILD_ID_MATCH;
    report.build_id_ok = 1;
  } else {
    report.build_id_status = HMX_BUILD_ID_MISMATCH;
    report.build_id_ok = 0;
  }

  report.all_ok = 1;
  report.first_failure = HMX_IDENTITY_FAILURE_NONE;
  for (index = HMX_IDENTITY_FAILURE_STATUS;
       index <= HMX_IDENTITY_FAILURE_BUILD_ID; ++index) {
    const hmx_identity_failure_t flag = (hmx_identity_failure_t)index;
    if (hmx_identity_flag_failed(&report, flag)) {
      report.all_ok = 0;
      if (report.first_failure == HMX_IDENTITY_FAILURE_NONE) {
        report.first_failure = flag;
      }
    }
  }
  return report;
}

/* Strict pre-experiment build-id gate for orchestrator-driven runs
 * (probe_host -strict-build-id). The device-harness semantics above stay
 * UNAVAILABLE-tolerant on purpose: a build id of 0 must not fail identity
 * for unstamped manual builds. This pure predicate is the ADDITIONAL
 * requirement used before any numerical RPC: it returns 1 only when the
 * build id was proven nonzero and matching on both sides. Nothing else in
 * the identity validator changes behavior. */
static inline int hmx_identity_strict_build_id_ok(
    const hmx_identity_report_t *report) {
  if (report->build_id_status != HMX_BUILD_ID_MATCH) return 0;
  if (report->expected_build_id == 0) return 0;
  if (report->meta_build_id == 0) return 0;
  if (report->meta_build_id != report->expected_build_id) return 0;
  return 1;
}

#endif /* HEXATRAIN_HMX_IDENTITY_H */
