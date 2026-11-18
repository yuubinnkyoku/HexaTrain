// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// CPU-only test of the D8 host gate and classifier (hmx_mode46_host.h).
//
// The device run is the only place the real arrays come from, but the gate and
// the classifier are pure functions of (rpc_status, slots, meta). This test
// drives every branch with synthetic inputs so a defect in the gate is found
// here rather than after the single registered device session has been spent.
//
// It does not prove anything about the device. It proves that:
//   - a well formed metadata block passes the gate;
//   - each individual field violation is caught, and the first failing field is
//     reported;
//   - the classifier maps each (condition, matched reference) pair to the
//     registered label, and never prints PROVEN;
//   - the host class follows the S0..S6 table;
//   - the observed-array digest matches the CPU model's serialisation.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hmx_d8_contract.h"
#include "hmx_mode46.h"
#include "hmx_mode46_host.h"

static int g_checks;
static int g_fail;

static void check(int cond, const char *tag) {
  g_checks++;
  if (!cond) {
    g_fail++;
    printf("FAIL %s\n", tag);
  }
}

/* Build a metadata block that satisfies the contract for one condition. */
static void make_valid(int *meta, int condition, int q, int build_id) {
  int i;
  for (i = 0; i < HMX_D8_METADATA_LEN; ++i) meta[i] = 0;
  meta[HMX_D8_META_GENERIC_MAGIC] = (int)HMX_D8_MAGIC;
  meta[HMX_D8_META_GENERIC_ARCH] = HMX_D8_ARCH;
  meta[HMX_D8_META_GENERIC_VTCM] = HMX_D8_VTCM_BYTES;
  meta[HMX_D8_META_GENERIC_HMX] = 1;
  meta[HMX_D8_META_GENERIC_TRIALS] = HMX_D8_TRIALS;
  meta[HMX_D8_META_BUILD_ID] = build_id;
  meta[HMX_D8_META_MXCLRACC] = 1;
  meta[HMX_D8_META_MMA] = 3;
  meta[HMX_D8_META_CONVERT] = 1;
  meta[HMX_D8_META_STORE] = 1;
  meta[HMX_D8_META_PACK] = 4;
  meta[HMX_D8_META_BIAS] = 2;
  meta[HMX_D8_META_FILL] =
      (int)hmx_mode46_fill_echo(hmx_d8_fill_mask_for_condition(condition));
  meta[HMX_D8_META_ADDRESS] = (int)hmx_mode46_address_echo(condition);
  meta[HMX_D8_META_SELECTOR] = (int)hmx_mode46_selector_echo(condition, q);
  meta[HMX_D8_META_FNV_AHI] = (int)0xC6231266u;
  meta[HMX_D8_META_FNV_ALO] = (int)0x3B2D22D2u;
  meta[HMX_D8_META_FNV_WHI] = (int)0x8B88609Fu;
  meta[HMX_D8_META_FNV_WLO] = (int)0xA66AF695u;
  meta[HMX_D8_META_GUARD_PRE] = (int)hmx_d8_guard_zero_fnv();
  meta[HMX_D8_META_GUARD_POST] = (int)hmx_d8_guard_zero_fnv();
  meta[HMX_D8_META_GUARD_STATUS] = (int)hmx_mode46_guard_status(
      HMX_D8_GUARD_STATUS_OK, HMX_D8_GUARD_FIRST_BAD_NONE,
      HMX_D8_GUARD_INIT_HVX12, HMX_D8_GUARD_CHECK_BYTE);
  meta[HMX_D8_META_FAILURE] = HMX_D8_FAIL_NONE;
  meta[HMX_D8_META_SCHEMA] = (int)HMX_D8_SCHEMA;
  meta[HMX_D8_META_READBACK] = (int)hmx_mode46_readback_echo(0);
  meta[HMX_D8_META_CLEANUP] = 0;
  meta[HMX_D8_META_RESERVED_29] = 0;
  meta[HMX_D8_META_RESERVED_31] = 0;
}

int main(void) {
  int meta[HMX_D8_METADATA_LEN];
  int slots[HMX_D8_RESULT_SLOTS];
  hmx_d8_contract_t c;
  int cond, q, i;

  /* ---- 1. a valid block passes for every condition ---------------------- */
  for (cond = 0; cond < 4; ++cond) {
    q = cond + HMX_D8_Q_FIRST;
    make_valid(meta, cond, q, 0x4D8B1121);
    c = hmx_d8_contract_check(meta, HMX_D8_METADATA_LEN, q, 0x4D8B1121);
    check(c.all_ok, "valid_block_passes");
    check(c.condition == cond, "valid_block_condition");
    check(c.q_echo == q, "valid_block_q_echo");
  }

  /* ---- 2. each field violation is caught ------------------------------- */
  struct {
    const char *tag;
    int word;
    int value;
  } faults[] = {
      {"fault_schema", HMX_D8_META_SCHEMA, 0},
      {"fault_build_id", HMX_D8_META_BUILD_ID, 0x11111111},
      {"fault_mxclracc", HMX_D8_META_MXCLRACC, 2},
      {"fault_mma", HMX_D8_META_MMA, 2},
      {"fault_convert", HMX_D8_META_CONVERT, 0},
      {"fault_store", HMX_D8_META_STORE, 0},
      {"fault_pack", HMX_D8_META_PACK, 3},
      {"fault_bias", HMX_D8_META_BIAS, 1},
      {"fault_fill", HMX_D8_META_FILL, 0},
      {"fault_address", HMX_D8_META_ADDRESS, 0},
      {"fault_selector", HMX_D8_META_SELECTOR, 0},
      {"fault_fnv_ahi", HMX_D8_META_FNV_AHI, 0},
      {"fault_fnv_alo", HMX_D8_META_FNV_ALO, 0},
      {"fault_fnv_whi", HMX_D8_META_FNV_WHI, 0},
      {"fault_fnv_wlo", HMX_D8_META_FNV_WLO, 0},
      {"fault_guard_pre", HMX_D8_META_GUARD_PRE, 0},
      {"fault_guard_post", HMX_D8_META_GUARD_POST, 0},
      {"fault_guard_status", HMX_D8_META_GUARD_STATUS, 0},
      {"fault_failure", HMX_D8_META_FAILURE, HMX_D8_FAIL_FNV},
      {"fault_readback", HMX_D8_META_READBACK, 0},
      {"fault_cleanup", HMX_D8_META_CLEANUP, 14007},
      {"fault_reserved_29", HMX_D8_META_RESERVED_29, 1},
      {"fault_reserved_31", HMX_D8_META_RESERVED_31, 1},
      {"fault_generic_magic", HMX_D8_META_GENERIC_MAGIC, 0},
      {"fault_generic_arch", HMX_D8_META_GENERIC_ARCH, 0},
      {"fault_generic_vtcm", HMX_D8_META_GENERIC_VTCM, 0},
      {"fault_generic_hmx", HMX_D8_META_GENERIC_HMX, 0},
      {"fault_generic_trials", HMX_D8_META_GENERIC_TRIALS, 0},
  };
  for (i = 0; i < (int)(sizeof(faults) / sizeof(faults[0])); ++i) {
    make_valid(meta, 1, 7, 0x4D8B1121);
    meta[faults[i].word] = faults[i].value;
    c = hmx_d8_contract_check(meta, HMX_D8_METADATA_LEN, 7, 0x4D8B1121);
    check(!c.all_ok, faults[i].tag);
    check(c.first_bad != NULL && strcmp(c.first_bad, "ok") != 0,
          "fault_reports_first_bad");
  }

  /* ---- 3. a wrong q is caught ----------------------------------------- */
  make_valid(meta, 1, 7, 0x4D8B1121);
  c = hmx_d8_contract_check(meta, HMX_D8_METADATA_LEN, 8, 0x4D8B1121);
  check(!c.all_ok, "fault_q_mismatch");

  /* ---- 4. the classifier maps every pair to the registered label ------- */
  {
    static const char *const expect[4][4] = {
        /* A: safe -> CONTROL_SAFE, anything else -> CONTROL_ANOMALY */
        {"D8_CONTROL_SAFE", "D8_CONTROL_ANOMALY", "D8_CONTROL_ANOMALY",
         "D8_CONTROL_ANOMALY"},
        /* B */
        {"D8_NO_VIOLATION_OBSERVED", "D8_VIOLATION_WLO_SUSPECT",
         "D8_VIOLATION_AHI_SUSPECT", "D8_VIOLATION_BOTH_SUSPECT"},
        /* C: only safe and wlo are meaningful */
        {"D8_NO_VIOLATION_OBSERVED", "D8_VIOLATION_WLO_SUSPECT",
         "D8_INCONCLUSIVE_NUMERIC", "D8_INCONCLUSIVE_NUMERIC"},
        /* D: only safe and ahi are meaningful */
        {"D8_NO_VIOLATION_OBSERVED", "D8_INCONCLUSIVE_NUMERIC",
         "D8_VIOLATION_AHI_SUSPECT", "D8_INCONCLUSIVE_NUMERIC"},
    };
    int m;
    for (cond = 0; cond < 4; ++cond) {
      for (m = 0; m < 4; ++m) {
        check(strcmp(d8_classify(cond, m), expect[cond][m]) == 0,
              "classify_table");
      }
      /* An unmatched array is INCONCLUSIVE for B/C/D. Condition A is the
       * numeric control, so anything other than the safe reference is an
       * anomaly by definition (prereg section 9). */
      if (cond == 0) {
        check(strcmp(d8_classify(cond, -1), "D8_CONTROL_ANOMALY") == 0,
              "classify_unmatched_control");
      } else {
        check(strcmp(d8_classify(cond, -1), "D8_INCONCLUSIVE_NUMERIC") == 0,
              "classify_unmatched");
      }
    }
  }

  /* ---- 5. the host class follows the S0..S6 table ---------------------- */
  make_valid(meta, 0, 6, 0x4D8B1121);
  check(strcmp(d8_host_class(0, meta, 1), "D8_CONTRACT_OK") == 0,
        "host_class_ok");
  check(strcmp(d8_host_class(0, meta, 0), "D8_HOST_EVIDENCE_FAILED") == 0,
        "host_class_no_capture");
  meta[HMX_D8_META_CLEANUP] = 14007;
  check(strcmp(d8_host_class(0, meta, 1), "D8_CONTRACT_FAILED_CLEANUP") == 0,
        "host_class_cleanup");
  meta[HMX_D8_META_CLEANUP] = 0;
  meta[HMX_D8_META_FAILURE] = HMX_D8_FAIL_FNV;
  check(strcmp(d8_host_class(0, meta, 1), "D8_CONTRACT_FAILED") == 0,
        "host_class_contract");
  meta[HMX_D8_META_FAILURE] = HMX_D8_FAIL_NONE;
  meta[HMX_D8_META_GENERIC_HMX] = 0;
  check(strcmp(d8_host_class(14001, meta, 1), "D8_CONTRACT_FAILED_PREEXEC") == 0,
        "host_class_preexec");
  meta[HMX_D8_META_GENERIC_HMX] = 1;
  check(strcmp(d8_host_class(14001, meta, 1), "D8_TRANSPORT_FAILED") == 0,
        "host_class_transport");

  /* ---- 6. the observed digest matches the CPU serialisation ------------ */
  {
    /* The CPU model hashes the array as u16 little endian. An all-zero array
     * therefore hashes the 2,048 zero bytes; check it against the same digest
     * computed here from a zero buffer. */
    char a[65], b[65];
    d8_sha256_t s;
    unsigned char zero[HMX_D8_RESULT_SLOTS * 2];
    memset(zero, 0, sizeof(zero));
    for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) slots[i] = 0;
    d8_observed_sha(slots, a);
    d8_sha256_init(&s);
    d8_sha256_update(&s, zero, sizeof(zero));
    d8_sha256_final(&s, b);
    check(strcmp(a, b) == 0, "digest_zero_array");
    /* And the standard vectors, so a broken SHA-256 cannot pass silently. */
    d8_sha256_init(&s);
    d8_sha256_final(&s, a);
    check(strcmp(a,
                 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == 0,
          "sha256_empty_vector");
    d8_sha256_init(&s);
    d8_sha256_update(&s, (const unsigned char *)"abc", 3);
    d8_sha256_final(&s, a);
    check(strcmp(a,
                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0,
          "sha256_abc_vector");
  }

  /* ---- 7. the results contract ---------------------------------------- */
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) slots[i] = i & 0xFFFF;
  check(hmx_d8_results_ok(slots, HMX_D8_RESULT_SLOTS), "results_ok");
  check(!hmx_d8_results_ok(slots, HMX_D8_RESULT_SLOTS - 1),
        "results_wrong_length");
  slots[0] = 0x10000;
  check(!hmx_d8_results_ok(slots, HMX_D8_RESULT_SLOTS), "results_out_of_range");

  /* ---- 8. the word PROVEN never appears in a label --------------------- */
  for (cond = 0; cond < 4; ++cond) {
    int m;
    for (m = -1; m < 4; ++m) {
      check(strstr(d8_classify(cond, m), "PROVEN") == NULL, "no_proven_word");
    }
  }

  printf("d8_host_gate_checks=%d fail=%d\n", g_checks, g_fail);
  if (g_fail != 0) {
    printf("d8_host_gate=FAIL\n");
    return 1;
  }
  printf("d8_host_gate=PASS\n");
  return 0;
}
