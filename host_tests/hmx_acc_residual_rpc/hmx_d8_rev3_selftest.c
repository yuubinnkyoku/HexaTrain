/* D8 revision-3 CPU contract self-test (CPU only).
 *
 * Exercises the metadata layout, the q selector, the guard evidence and the
 * host contract gate defined in hmx_d8_contract.h. It builds one valid
 * metadata array per condition, proves the gate accepts it, then injects one
 * defect at a time and proves the gate rejects it with the expected field.
 *
 * It also exposes --d8-contract-replay <file> so the independent Python audit
 * can feed cases and compare verdicts. The C side and the Python side decode
 * the same 32 words independently.
 *
 * No device, no DSP, no SDK, no ADB, no Mode 46. The metadata here is a
 * simulated host buffer, not a device observation.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hmx_d8_contract.h"

static int g_checks;
static int g_fail;

static void check(int cond, const char *tag) {
  g_checks++;
  if (!cond) {
    g_fail++;
    printf("FAIL %s\n", tag);
  }
}

static void put_valid(int *m, int condition, unsigned int build_id) {
  int q = hmx_d8_q_for_condition(condition);
  int i;
  for (i = 0; i < HMX_D8_METADATA_LEN; ++i) m[i] = 0;
  m[HMX_D8_META_GENERIC_MAGIC] = (int)HMX_D8_MAGIC;
  m[HMX_D8_META_GENERIC_ARCH] = HMX_D8_ARCH;
  m[HMX_D8_META_GENERIC_VTCM] = HMX_D8_VTCM_BYTES;
  m[HMX_D8_META_GENERIC_ELAPSED] = 4321; /* run()-owned, ignored by D8 */
  m[HMX_D8_META_GENERIC_POWER] = 0;
  m[HMX_D8_META_GENERIC_LOCK_A] = 0;
  m[HMX_D8_META_GENERIC_LOCK_B] = 0;
  m[HMX_D8_META_GENERIC_HMX] = 1;
  m[HMX_D8_META_GENERIC_TRIALS] = 1;

  m[HMX_D8_META_BUILD_ID] = (int)build_id;
  m[HMX_D8_META_MXCLRACC] = 1;
  m[HMX_D8_META_MMA] = 3;
  m[HMX_D8_META_CONVERT] = 1;
  m[HMX_D8_META_STORE] = 1;
  m[HMX_D8_META_PACK] = 4;
  m[HMX_D8_META_BIAS] = 2;
  m[HMX_D8_META_FILL] =
      hmx_d8_fill_mask_for_condition(condition) |
      (HMX_D8_FILL_WORD << HMX_D8_FILL_WORD_SHIFT) |
      (HMX_D8_FILL_FORM_16X128 << HMX_D8_FILL_FORM_SHIFT);
  m[HMX_D8_META_ADDRESS] =
      HMX_D8_ADDR_OUT |
      (((condition == 1 || condition == 2) ? HMX_D8_ADDR_WLO : 0)
       << HMX_D8_ADDR_WLO_SHIFT) |
      (HMX_D8_ADDR_AHI << HMX_D8_ADDR_AHI_SHIFT);
  m[HMX_D8_META_SELECTOR] = condition | (q << HMX_D8_SELECTOR_Q_SHIFT);
  m[HMX_D8_META_FNV_AHI] = (int)0xC6231266u;
  m[HMX_D8_META_FNV_ALO] = (int)0x3B2D22D2u;
  m[HMX_D8_META_FNV_WHI] = (int)0x8B88609Fu;
  m[HMX_D8_META_FNV_WLO] = (int)0xA66AF695u;
  m[HMX_D8_META_GUARD_PRE] = (int)hmx_d8_guard_zero_fnv();
  m[HMX_D8_META_GUARD_POST] = (int)hmx_d8_guard_zero_fnv();
  m[HMX_D8_META_GUARD_STATUS] =
      HMX_D8_GUARD_STATUS_OK |
      (HMX_D8_GUARD_FIRST_BAD_NONE << HMX_D8_GUARD_FIRST_BAD_SHIFT) |
      (HMX_D8_GUARD_INIT_HVX12 << HMX_D8_GUARD_INIT_SHIFT) |
      (HMX_D8_GUARD_CHECK_BYTE << HMX_D8_GUARD_CHECK_SHIFT);
  m[HMX_D8_META_FAILURE] = HMX_D8_FAIL_NONE;
  m[HMX_D8_META_SCHEMA] = (int)HMX_D8_SCHEMA;
  m[HMX_D8_META_READBACK] =
      (HMX_D8_TRACE_NOT_PROVIDED << HMX_D8_READBACK_TRACE_SHIFT) |
      (1 << HMX_D8_READBACK_DONE_SHIFT) |
      (0 << HMX_D8_READBACK_HVX_SHIFT) |
      (HMX_D8_RESULT_SLOTS << HMX_D8_READBACK_SLOTS_SHIFT);
  m[HMX_D8_META_RESERVED_29] = 0;
  m[HMX_D8_META_CLEANUP] = 0;
  m[HMX_D8_META_RESERVED_31] = 0;
}

static int expect_reject(const int *m, int q, unsigned int build,
                         const char *field, const char *tag) {
  hmx_d8_contract_t c = hmx_d8_contract_check(m, HMX_D8_METADATA_LEN, q,
                                              (int)build);
  int ok = (!c.all_ok) && strcmp(c.first_bad, field) == 0;
  if (!ok) {
    printf("d8_rev3_fault tag=%s first_bad=%s want=%s\n", tag, c.first_bad,
           field);
  }
  check(ok, tag);
  return ok;
}

static int replay(const char *path) {
  FILE *f = fopen(path, "r");
  char line[4096];
  int cases = 0, malformed = 0;
  if (!f) {
    fprintf(stderr, "d8_contract_replay=FAIL cannot open %s\n", path);
    return 2;
  }
  while (fgets(line, (int)sizeof(line), f)) {
    char *p = line;
    int m[HMX_D8_METADATA_LEN];
    int id = 0, q = 0, i;
    unsigned int build = 0;
    hmx_d8_contract_t c;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '#') continue;
    if (strncmp(p, "case ", 5) != 0) { malformed++; continue; }
    p += 5;
    id = (int)strtol(p, &p, 10);
    q = (int)strtol(p, &p, 10);
    {
      char *bend = 0;
      build = (unsigned int)strtoul(p, &bend, 16);
      if (bend == p) { malformed++; continue; }
      p = bend;
    }
    for (i = 0; i < HMX_D8_METADATA_LEN; ++i) {
      char *end = 0;
      unsigned int v;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '\0' || *p == '\n') { malformed++; break; }
      v = (unsigned int)strtoul(p, &end, 16);
      if (end == p) { malformed++; break; }
      m[i] = (int)v;
      p = end;
    }
    if (i != HMX_D8_METADATA_LEN) { malformed++; continue; }
    c = hmx_d8_contract_check(m, HMX_D8_METADATA_LEN, q, (int)build);
    printf("replay %d %s %s\n", id, c.all_ok ? "allow" : "deny", c.first_bad);
    cases++;
  }
  fclose(f);
  printf("d8_contract_replay cases=%d malformed=%d\n", cases, malformed);
  return malformed == 0 ? 0 : 2;
}

int main(int argc, char **argv) {
  int base[HMX_D8_METADATA_LEN];
  int m[HMX_D8_METADATA_LEN];
  int results[HMX_D8_RESULT_SLOTS];
  int i, cond;
  unsigned int build = 0xD8B01110u;

  if (argc == 3 && strcmp(argv[1], "--d8-contract-replay") == 0) {
    return replay(argv[2]);
  }
  if (argc != 1) {
    fprintf(stderr, "usage: %s [--d8-contract-replay <file>]\n", argv[0]);
    return 2;
  }

  printf("d8_rev3_layout=3 metadata_len=32 words=9..31 results=%d\n",
         HMX_D8_RESULT_SLOTS);
  printf("d8_rev3_schema=0x%08x guard_zero_fnv=0x%08x\n", HMX_D8_SCHEMA,
         hmx_d8_guard_zero_fnv());
  printf("d8_rev3_q A=%d B=%d C=%d D=%d\n",
         hmx_d8_q_for_condition(0), hmx_d8_q_for_condition(1),
         hmx_d8_q_for_condition(2), hmx_d8_q_for_condition(3));
  printf("d8_rev3_fill_mask A=%d B=%d C=%d D=%d\n",
         hmx_d8_fill_mask_for_condition(0),
         hmx_d8_fill_mask_for_condition(1),
         hmx_d8_fill_mask_for_condition(2),
         hmx_d8_fill_mask_for_condition(3));

  /* ---- q selector ---- */
  check(hmx_d8_condition_for_q(6) == 0, "q6_is_A");
  check(hmx_d8_condition_for_q(7) == 1, "q7_is_B");
  check(hmx_d8_condition_for_q(8) == 2, "q8_is_C");
  check(hmx_d8_condition_for_q(9) == 3, "q9_is_D");
  check(hmx_d8_condition_for_q(5) == -1, "q5_rejected");
  check(hmx_d8_condition_for_q(0) == -1, "q0_rejected");
  check(hmx_d8_condition_for_q(3) == -1, "q3_rejected");
  check(hmx_d8_condition_for_q(10) == -1, "q10_rejected");
  check(hmx_d8_condition_for_q(11) == -1, "q11_rejected");
  check(hmx_d8_q_for_condition(0) == 6 && hmx_d8_q_for_condition(3) == 9,
        "q_roundtrip");
  check(hmx_d8_q_for_condition(4) == -1, "condition4_rejected");
  check(hmx_d8_condition_for_q(hmx_d8_q_for_condition(2)) == 2,
        "q_condition_roundtrip");
  check(HMX_D8_Q_IDENTITY == HMX_D8_Q_FIRST, "identity_q_is_6");
  check(HMX_D8_IDENTITY_MODE == 30, "identity_mode_30");
  check(HMX_D8_MODE == 46, "mode_46");

  /* ---- valid layout per condition ---- */
  for (cond = 0; cond < 4; ++cond) {
    hmx_d8_contract_t c;
    int q = hmx_d8_q_for_condition(cond);
    put_valid(base, cond, build);
    c = hmx_d8_contract_check(base, HMX_D8_METADATA_LEN, q, (int)build);
    check(c.all_ok && strcmp(c.first_bad, "ok") == 0, "valid_condition");
    check(c.condition == cond && c.q_echo == q, "valid_selector_decode");
    check((base[HMX_D8_META_FILL] & HMX_D8_FILL_MASK_BITS) ==
              hmx_d8_fill_mask_for_condition(cond),
          "valid_fill_mask");
  }

  /* ---- generic words (run()-owned) ---- */
  put_valid(m, 1, build);
  m[HMX_D8_META_GENERIC_MAGIC] = 0;
  expect_reject(m, 7, build, "generic", "generic_magic");
  put_valid(m, 1, build);
  m[HMX_D8_META_GENERIC_ARCH] = 80;
  expect_reject(m, 7, build, "generic", "generic_arch");
  put_valid(m, 1, build);
  m[HMX_D8_META_GENERIC_VTCM] = 8192;
  expect_reject(m, 7, build, "generic", "generic_vtcm");
  put_valid(m, 1, build);
  m[HMX_D8_META_GENERIC_HMX] = 0;
  expect_reject(m, 7, build, "generic", "generic_hmx_flag");
  put_valid(m, 1, build);
  m[HMX_D8_META_GENERIC_TRIALS] = 2;
  expect_reject(m, 7, build, "generic", "generic_trials");
  put_valid(m, 1, build);
  check(hmx_d8_contract_check(m, 31, 7, (int)build).all_ok == 0,
        "meta_len_31_rejected");
  check(hmx_d8_contract_check(m, 33, 7, (int)build).all_ok == 0,
        "meta_len_33_rejected");

  /* ---- build id / condition confusion ---- */
  put_valid(m, 2, build);
  m[HMX_D8_META_BUILD_ID] = (int)build + 1;
  expect_reject(m, 8, build, "build_id", "build_id_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_BUILD_ID] = (int)(3u); /* condition value in build slot */
  expect_reject(m, 8, build, "build_id", "condition_in_build_slot");
  put_valid(m, 2, build);
  m[HMX_D8_META_SELECTOR] = (int)build; /* build id in selector slot */
  expect_reject(m, 8, build, "selector", "build_id_in_selector_slot");
  put_valid(m, 2, build);
  m[HMX_D8_META_SELECTOR] = 3 | (8 << HMX_D8_SELECTOR_Q_SHIFT); /* C->D */
  expect_reject(m, 8, build, "selector", "c_d_swap");
  put_valid(m, 2, build);
  m[HMX_D8_META_SELECTOR] = 2 | (9 << HMX_D8_SELECTOR_Q_SHIFT); /* q wrong */
  expect_reject(m, 8, build, "selector", "q_echo_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_SELECTOR] = 2 | (8 << HMX_D8_SELECTOR_Q_SHIFT) | 0x10;
  expect_reject(m, 8, build, "selector", "selector_reserved_bit");
  put_valid(m, 2, build);
  check(hmx_d8_contract_check(m, HMX_D8_METADATA_LEN, 5, (int)build).all_ok
            == 0,
        "q5_contract_rejected");
  check(hmx_d8_contract_check(m, HMX_D8_METADATA_LEN, 10, (int)build).all_ok
            == 0,
        "q10_contract_rejected");

  /* ---- counts ---- */
  put_valid(m, 3, build);
  m[HMX_D8_META_MXCLRACC] = 2;
  expect_reject(m, 9, build, "mxclracc", "count_mxclracc");
  put_valid(m, 3, build);
  m[HMX_D8_META_MMA] = 2;
  expect_reject(m, 9, build, "mma", "count_mma");
  put_valid(m, 3, build);
  m[HMX_D8_META_CONVERT] = 2;
  expect_reject(m, 9, build, "convert", "count_convert");
  put_valid(m, 3, build);
  m[HMX_D8_META_STORE] = 0;
  expect_reject(m, 9, build, "store", "count_store");
  put_valid(m, 3, build);
  m[HMX_D8_META_PACK] = 3;
  expect_reject(m, 9, build, "pack", "count_pack");
  put_valid(m, 3, build);
  m[HMX_D8_META_BIAS] = 3;
  expect_reject(m, 9, build, "bias", "count_bias");

  /* ---- fill evidence ---- */
  put_valid(m, 2, build);
  m[HMX_D8_META_FILL] = 5 | (HMX_D8_FILL_WORD << 8) | (1 << 24);
  expect_reject(m, 8, build, "fill_mask", "c_uses_d_mask");
  put_valid(m, 2, build);
  m[HMX_D8_META_FILL] = 3 | (0x5A5A << 8) | (1 << 24);
  expect_reject(m, 8, build, "fill_word", "fill_word_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_FILL] = 3 | (HMX_D8_FILL_WORD << 8) | (2 << 24);
  expect_reject(m, 8, build, "fill_form", "fill_form_scalar");
  put_valid(m, 2, build);
  m[HMX_D8_META_FILL] = 3 | (HMX_D8_FILL_WORD << 8) | (1 << 24) | 0x08;
  expect_reject(m, 8, build, "fill_form", "fill_reserved_bit");

  /* ---- address echo ---- */
  put_valid(m, 2, build);
  m[HMX_D8_META_ADDRESS] = 63 | (48 << 7);
  expect_reject(m, 8, build, "address", "store_target_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_ADDRESS] = 64 | (40 << 7);
  expect_reject(m, 8, build, "address", "wlo_target_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_ADDRESS] = 64 | (48 << 7) | (1 << 14);
  expect_reject(m, 8, build, "address", "ahi_target_wrong");
  put_valid(m, 2, build);
  m[HMX_D8_META_ADDRESS] = 64 | (48 << 7) | 0x100000;
  expect_reject(m, 8, build, "address", "address_reserved_bit");
  put_valid(m, 3, build);
  m[HMX_D8_META_ADDRESS] = 64 | (48 << 7); /* D must not echo a Wlo fill */
  expect_reject(m, 9, build, "address", "d_with_wlo_address");

  /* ---- operand FNV ---- */
  put_valid(m, 0, build);
  m[HMX_D8_META_FNV_WLO] = (int)0xA66AF694u;
  expect_reject(m, 6, build, "operand_fnv", "fnv_wlo_wrong");
  put_valid(m, 0, build);
  m[HMX_D8_META_FNV_AHI] = (int)0xC6231267u;
  expect_reject(m, 6, build, "operand_fnv", "fnv_ahi_wrong");

  /* ---- guard ---- */
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_PRE] = (int)hmx_d8_guard_zero_fnv() ^ 1u;
  expect_reject(m, 7, build, "guard_digest", "guard_pre_nonzero");
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_POST] = (int)hmx_d8_guard_zero_fnv() ^ 0x100u;
  expect_reject(m, 7, build, "guard_digest", "guard_post_changed");
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_STATUS] =
      HMX_D8_GUARD_STATUS_POST_CHANGED |
      (12 << HMX_D8_GUARD_FIRST_BAD_SHIFT) |
      (HMX_D8_GUARD_INIT_HVX12 << 20) | (HMX_D8_GUARD_CHECK_BYTE << 24);
  expect_reject(m, 7, build, "guard_status", "guard_status_changed");
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_STATUS] = HMX_D8_GUARD_STATUS_OK |
      (HMX_D8_GUARD_FIRST_BAD_NONE << 8) |
      (HMX_D8_GUARD_INIT_NONE << 20) | (HMX_D8_GUARD_CHECK_BYTE << 24);
  expect_reject(m, 7, build, "guard_status", "guard_init_missing");
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_STATUS] = HMX_D8_GUARD_STATUS_OK |
      (HMX_D8_GUARD_FIRST_BAD_NONE << 8) |
      (HMX_D8_GUARD_INIT_HVX12 << 20) | (HMX_D8_GUARD_CHECK_NONE << 24);
  expect_reject(m, 7, build, "guard_status", "guard_check_missing");
  put_valid(m, 1, build);
  m[HMX_D8_META_GUARD_STATUS] |= 0x8; /* status 8, undefined */
  expect_reject(m, 7, build, "guard_status", "guard_status_undefined");

  /* ---- failure / readback / cleanup / reserved ---- */
  put_valid(m, 2, build);
  m[HMX_D8_META_FAILURE] = HMX_D8_FAIL_GUARD_POST;
  expect_reject(m, 8, build, "failure_code", "failure_code_nonzero");
  put_valid(m, 2, build);
  m[HMX_D8_META_FAILURE] = HMX_D8_FAIL_COUNT + 5;
  expect_reject(m, 8, build, "failure_code", "failure_code_out_of_range");
  put_valid(m, 2, build);
  m[HMX_D8_META_READBACK] = 1 | (1 << 2) | (1024 << 11); /* trace=1 */
  expect_reject(m, 8, build, "readback", "trace_claimed");
  put_valid(m, 2, build);
  m[HMX_D8_META_READBACK] = (1 << 2) | (1023 << 11);
  expect_reject(m, 8, build, "readback", "slots_short");
  put_valid(m, 2, build);
  m[HMX_D8_META_READBACK] = (1 << 2) | (3 << 3) | (1024 << 11);
  expect_reject(m, 8, build, "readback", "hvx_lock_failed");
  put_valid(m, 2, build);
  m[HMX_D8_META_READBACK] = (1024 << 11); /* readback bit missing */
  expect_reject(m, 8, build, "readback", "readback_missing");
  put_valid(m, 2, build);
  m[HMX_D8_META_CLEANUP] = 14008;
  expect_reject(m, 8, build, "cleanup", "cleanup_failed");
  put_valid(m, 2, build);
  m[HMX_D8_META_RESERVED_31] = 1;
  expect_reject(m, 8, build, "reserved", "reserved31_nonzero");
  put_valid(m, 2, build);
  m[HMX_D8_META_RESERVED_29] = 1;
  expect_reject(m, 8, build, "reserved", "reserved29_nonzero");
  put_valid(m, 2, build);
  m[HMX_D8_META_SCHEMA] = (int)(HMX_D8_SCHEMA + 1u);
  expect_reject(m, 8, build, "schema", "schema_wrong");

  /* ---- results slots ---- */
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) results[i] = i & 0xFFFF;
  check(hmx_d8_results_ok(results, HMX_D8_RESULT_SLOTS) == 1,
        "results_all_u16");
  check(hmx_d8_results_ok(results, 1023) == 0, "results_short_rejected");
  check(hmx_d8_results_ok(results, 1025) == 0, "results_long_rejected");
  check(hmx_d8_results_ok(0, HMX_D8_RESULT_SLOTS) == 0,
        "results_null_rejected");
  {
    int copy[HMX_D8_RESULT_SLOTS];
    memcpy(copy, results, sizeof(copy));
    check(hmx_d8_results_equal(results, copy) == 1, "results_equal_self");
    copy[1023] ^= 1;
    check(hmx_d8_results_equal(results, copy) == 0, "results_equal_last_cell");
    results[0] = -1;
    check(hmx_d8_results_ok(results, HMX_D8_RESULT_SLOTS) == 0,
          "results_negative_rejected");
    results[0] = 0x10000;
    check(hmx_d8_results_ok(results, HMX_D8_RESULT_SLOTS) == 0,
          "results_overflow_rejected");
  }

  /* ---- failure namespace ---- */
  check(strcmp(hmx_d8_failure_name(HMX_D8_FAIL_NONE), "NONE") == 0,
        "failure_name_none");
  for (i = 0; i < HMX_D8_FAIL_COUNT; ++i) {
    check(strcmp(hmx_d8_failure_name(i), "UNKNOWN") != 0,
          "failure_name_defined");
  }

  printf("d8_rev3_checks=%d fail=%d\n", g_checks, g_fail);
  if (g_fail) {
    printf("d8_rev3_verdict=D8_REV3_CPU_CONTRACT_FAIL\n");
    return 1;
  }
  printf("d8_rev3_verdict=D8_REV3_CPU_CONTRACT_READY (CPU only, not device)\n");
  return 0;
}
