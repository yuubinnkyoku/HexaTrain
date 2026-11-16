// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// R2 S1/S2 (mode 45) frozen contract, shared by probe_dsp.c (executes it),
// probe_host.c (classifies the device result) and
// probe_r2_mode45_selftest.c (CPU-only contract test). The fixture, the 11
// slots, the four Rs values and the decision order come from
// docs/hmx-v81-r2-device-prereg.md and
// docs/hmx-v81-r2-device-implementation-plan.md and must not be retuned after
// the first measurement.
//
// This header is dependency-free (plain C types) and safe for the DSP skel.
//
// Fixture:
//   a = 1 + 2^-6 = 1.015625 = 0x3C10 (exact FP16)
//   x = a*a      = 1.031494140625
//   S1: ACC(s0,o0) = +x                       -> 64x = 66.015625
//   S2: ACC(s0,o0) = +x, ACC(s1,o0) = -x, ACC(s2,o0) = 0
//
// VTCM map (bytes from the 12 KiB reserve base):
//   0 act | 2048 wgt | 4096 out_pos | 6144 out_neg | 8192 out_recon
//   10240 bias set 0 | 10496 bias set 1 | 10752 bias set 2
//
// Store cells (PRM store_u16(s,o) = (s>>1)*64 + o*2 + (s&1)):
//   0 = (s0,o0), 1 = (s1,o0), 64 = (s2,o0)
#define HMX_MODE45_CELL_S0_O0 0
#define HMX_MODE45_CELL_S1_O0 1
#define HMX_MODE45_CELL_S2_O0 64

static inline int hmx_mode45_store_index(int s, int o) {
  return (s >> 1) * 64 + o * 2 + (s & 1);
}

#ifndef HEXATRAIN_HMX_MODE45_H
#define HEXATRAIN_HMX_MODE45_H

#define HMX_MODE45_MODE 45
#define HMX_MODE45_Q 6
#define HMX_MODE45_RESULT_SLOTS 11
#define HMX_MODE45_METADATA_LEN 32
#define HMX_MODE45_TRIALS 2

/* VTCM layout (frozen, pre-registration section 5). */
#define HMX_MODE45_ACT_OFFSET 0
#define HMX_MODE45_WGT_OFFSET 2048
#define HMX_MODE45_POS_OFFSET 4096
#define HMX_MODE45_NEG_OFFSET 6144
#define HMX_MODE45_RECON_OFFSET 8192
#define HMX_MODE45_BIAS0_OFFSET 10240
#define HMX_MODE45_BIAS1_OFFSET 10496
#define HMX_MODE45_BIAS2_OFFSET 10752
#define HMX_MODE45_VTCM_USED 11008

/* Fixture operands and registered bias payloads. */
#define HMX_MODE45_A_BITS 0x3C10
#define HMX_MODE45_NEG_A_BITS 0xBC10
#define HMX_MODE45_SCALE64_BITS 0x5400
#define HMX_MODE45_SCALE32_BITS 0x5000
#define HMX_MODE45_SHAPE0 0
#define HMX_MODE45_SHAPE4 4
#define HMX_MODE45_OUT_BIAS_BITS 0x0000

/* Registered Rs values, in execution order. */
#define HMX_MODE45_RS_S1_FIRST 0x0001u
#define HMX_MODE45_RS_S1_SECOND 0x2000u
#define HMX_MODE45_RS_S2_FIRST 0x0001u
#define HMX_MODE45_RS_S2_SECOND 0x1000u

/* Frozen slot pattern (pre-registration section 6). */
enum {
  HMX_MODE45_SLOT_S1_FIRST = 0x5420,
  HMX_MODE45_SLOT_S1_SECOND = 0x5020,
  HMX_MODE45_SLOT_S2_POS_S0 = 0x5420,
  HMX_MODE45_SLOT_S2_NEG_S1 = 0x5420,
  HMX_MODE45_SLOT_S2_RECON_S0 = 0x5420,
  HMX_MODE45_SLOT_S2_RECON_S1 = 0xD420,
  HMX_MODE45_SENTINEL_WORD = 0xA5A5
};

/* Registered conversion counts (pre-registration section 7). */
enum {
  HMX_MODE45_MXCLRACC_COUNT = 2,
  HMX_MODE45_MMA_COUNT = 2,
  HMX_MODE45_CONVERT_COUNT = 4,
  HMX_MODE45_STORE_COUNT = 4,
  HMX_MODE45_BIAS_LOADS = 3,
  HMX_MODE45_RECON_LOOPS = 2,
  HMX_MODE45_POS_REGION_WRITES = 2,
  HMX_MODE45_NEG_REGION_WRITES = 2
};

/* Metadata evidence indices. metadata[0..8] keep the existing meaning
 * (magic, arch, vtcm bytes, kernel us, power, lock A, lock B, hmx executed,
 * trials). Indices 9..31 are written only by mode 45. */
typedef enum hmx_mode45_meta_index {
  HMX_MODE45_META_SCHEMA = 9,
  HMX_MODE45_META_MXCLRACC = 10,
  HMX_MODE45_META_MMA = 11,
  HMX_MODE45_META_CONVERT = 12,
  HMX_MODE45_META_STORE = 13,
  HMX_MODE45_META_BIAS_LOADS = 14,
  HMX_MODE45_META_RECON_LOOPS = 15,
  HMX_MODE45_META_RS_S1_FIRST = 16,
  HMX_MODE45_META_RS_S1_SECOND = 17,
  HMX_MODE45_META_RS_S2_FIRST = 18,
  HMX_MODE45_META_RS_S2_SECOND = 19,
  HMX_MODE45_META_BIAS_ECHO = 20,   /* bit n set = bias set n load issued */
  HMX_MODE45_META_SENTINEL_HITS = 21,
  HMX_MODE45_META_REGION_LAYOUT = 22,
  HMX_MODE45_META_RECON_SOURCE = 23, /* 1 = DSP HVX vsub loop */
  HMX_MODE45_META_BUILD_ID = 24,
  HMX_MODE45_META_HVX_LOCK = 25,
  HMX_MODE45_META_POS_WRITES = 26,
  HMX_MODE45_META_NEG_WRITES = 27,
  HMX_MODE45_META_HVX_POS_UNCHANGED = 28,
  HMX_MODE45_META_HVX_NEG_UNCHANGED = 29,
  HMX_MODE45_META_RECON_VECTORS = 30,
  HMX_MODE45_META_RESERVED = 31
} hmx_mode45_meta_index_t;

enum {
  HMX_MODE45_SCHEMA_REVISION = 1,
  HMX_MODE45_BIAS_ECHO_ALL = 7,
  HMX_MODE45_RECON_SOURCE_DSP_HVX = 1,
  HMX_MODE45_RECON_VECTORS = 32 /* 2 loops x 16 vectors of 128 B */
};

typedef enum hmx_mode45_verdict {
  HMX_MODE45_ACC_REUSE_SUPPORTED = 0,
  HMX_MODE45_CAUSE_INCONCLUSIVE = 1,
  HMX_MODE45_STORE_INTERFERENCE = 2,
  HMX_MODE45_ROUNDING_MISMATCH = 3,
  HMX_MODE45_UNEXPECTED_OUTPUT = 4,
  HMX_MODE45_HOST_CONTRACT_FAILED = 5
} hmx_mode45_verdict_t;

static inline const char *hmx_mode45_verdict_name(hmx_mode45_verdict_t v) {
  switch (v) {
    case HMX_MODE45_ACC_REUSE_SUPPORTED: return "R2_ACC_REUSE_SUPPORTED";
    case HMX_MODE45_CAUSE_INCONCLUSIVE: return "R2_CAUSE_INCONCLUSIVE";
    case HMX_MODE45_STORE_INTERFERENCE: return "R2_STORE_INTERFERENCE";
    case HMX_MODE45_ROUNDING_MISMATCH: return "R2_ROUNDING_MISMATCH";
    case HMX_MODE45_UNEXPECTED_OUTPUT: return "R2_UNEXPECTED_OUTPUT";
    case HMX_MODE45_HOST_CONTRACT_FAILED: return "R2_HOST_CONTRACT_FAILED";
  }
  return "R2_UNKNOWN";
}

typedef struct hmx_mode45_contract {
  int slots_ok;
  int slot_values_ok;
  int sentinel_ok;
  int sentinel_meta_ok;
  int mxclracc_ok;
  int mma_ok;
  int convert_ok;
  int store_ok;
  int bias_loads_ok;
  int recon_loops_ok;
  int rs_ok;
  int bias_echo_ok;
  int region_layout_ok;
  int recon_source_ok;
  int hvx_lock_ok;
  int pos_writes_ok;
  int neg_writes_ok;
  int pos_unchanged_ok;
  int neg_unchanged_ok;
  int recon_vectors_ok;
  int reserved_ok;
  int trials_ok;
  int hmx_executed_ok;
  int build_id_ok;
  int all_ok;
} hmx_mode45_contract_t;

typedef struct hmx_mode45_report {
  hmx_mode45_contract_t contract;
  hmx_mode45_verdict_t verdict;
  const char *cause;
} hmx_mode45_report_t;

/* Step 1 of the pre-registered decision procedure: the host contract.
 * Every check is independent and reported so a failure names the field.
 * expected_build_id may be 0 on an unstamped manual build; the strict
 * orchestrator gate (hmx_identity_strict_build_id_ok) is what requires a
 * nonzero stamped build. */
static inline hmx_mode45_contract_t hmx_mode45_contract(
    const int *result, int result_len, const int *meta, int meta_len,
    int expected_build_id) {
  hmx_mode45_contract_t c;
  int i;

  c.slots_ok = (result != 0) && (result_len == HMX_MODE45_RESULT_SLOTS) &&
               (meta != 0) && (meta_len >= HMX_MODE45_METADATA_LEN);
  c.slot_values_ok = 1;
  c.sentinel_ok = 1;
  if (c.slots_ok) {
    for (i = 0; i < HMX_MODE45_RESULT_SLOTS; ++i) {
      if (result[i] < 0 || result[i] > 0xFFFF) c.slot_values_ok = 0;
      if (result[i] == HMX_MODE45_SENTINEL_WORD) c.sentinel_ok = 0;
    }
  } else {
    c.slot_values_ok = 0;
    c.sentinel_ok = 0;
  }
  c.sentinel_meta_ok =
      c.slots_ok && meta[HMX_MODE45_META_SENTINEL_HITS] == 0;
  c.mxclracc_ok = c.slots_ok &&
                  meta[HMX_MODE45_META_MXCLRACC] == HMX_MODE45_MXCLRACC_COUNT;
  c.mma_ok = c.slots_ok && meta[HMX_MODE45_META_MMA] == HMX_MODE45_MMA_COUNT;
  c.convert_ok = c.slots_ok &&
                 meta[HMX_MODE45_META_CONVERT] == HMX_MODE45_CONVERT_COUNT;
  c.store_ok = c.slots_ok &&
               meta[HMX_MODE45_META_STORE] == HMX_MODE45_STORE_COUNT;
  c.bias_loads_ok = c.slots_ok &&
                    meta[HMX_MODE45_META_BIAS_LOADS] == HMX_MODE45_BIAS_LOADS;
  c.recon_loops_ok = c.slots_ok &&
                     meta[HMX_MODE45_META_RECON_LOOPS] == HMX_MODE45_RECON_LOOPS;
  c.rs_ok = c.slots_ok &&
            (unsigned)meta[HMX_MODE45_META_RS_S1_FIRST] ==
                HMX_MODE45_RS_S1_FIRST &&
            (unsigned)meta[HMX_MODE45_META_RS_S1_SECOND] ==
                HMX_MODE45_RS_S1_SECOND &&
            (unsigned)meta[HMX_MODE45_META_RS_S2_FIRST] ==
                HMX_MODE45_RS_S2_FIRST &&
            (unsigned)meta[HMX_MODE45_META_RS_S2_SECOND] ==
                HMX_MODE45_RS_S2_SECOND;
  c.bias_echo_ok = c.slots_ok &&
                   meta[HMX_MODE45_META_BIAS_ECHO] == HMX_MODE45_BIAS_ECHO_ALL;
  c.region_layout_ok = c.slots_ok &&
                       meta[HMX_MODE45_META_REGION_LAYOUT] == 1;
  c.recon_source_ok =
      c.slots_ok &&
      meta[HMX_MODE45_META_RECON_SOURCE] == HMX_MODE45_RECON_SOURCE_DSP_HVX;
  c.hvx_lock_ok = c.slots_ok && meta[HMX_MODE45_META_HVX_LOCK] == 0;
  c.pos_writes_ok = c.slots_ok &&
                    meta[HMX_MODE45_META_POS_WRITES] ==
                        HMX_MODE45_POS_REGION_WRITES;
  c.neg_writes_ok = c.slots_ok &&
                    meta[HMX_MODE45_META_NEG_WRITES] ==
                        HMX_MODE45_NEG_REGION_WRITES;
  c.pos_unchanged_ok =
      c.slots_ok && meta[HMX_MODE45_META_HVX_POS_UNCHANGED] == 1;
  c.neg_unchanged_ok =
      c.slots_ok && meta[HMX_MODE45_META_HVX_NEG_UNCHANGED] == 1;
  c.recon_vectors_ok =
      c.slots_ok && meta[HMX_MODE45_META_RECON_VECTORS] ==
                        HMX_MODE45_RECON_VECTORS;
  c.reserved_ok = c.slots_ok && meta[HMX_MODE45_META_RESERVED] == 0;
  c.trials_ok = c.slots_ok && meta[8] == HMX_MODE45_TRIALS;
  c.hmx_executed_ok = c.slots_ok && meta[7] == 1;
  c.build_id_ok = 1;
  if (c.slots_ok && expected_build_id != 0) {
    c.build_id_ok = meta[HMX_MODE45_META_BUILD_ID] == expected_build_id;
  }

  /* pos_writes_ok / neg_writes_ok are deliberately NOT part of all_ok:
   * pre-registration section 8 puts a region-write mismatch in decision
   * step 2 (R2_STORE_INTERFERENCE), after the fail-closed contract. */
  c.all_ok = c.slots_ok && c.slot_values_ok && c.sentinel_ok &&
             c.sentinel_meta_ok && c.mxclracc_ok && c.mma_ok &&
             c.convert_ok && c.store_ok && c.bias_loads_ok &&
             c.recon_loops_ok && c.rs_ok && c.bias_echo_ok &&
             c.region_layout_ok && c.recon_source_ok && c.hvx_lock_ok &&
             c.pos_unchanged_ok && c.neg_unchanged_ok &&
             c.recon_vectors_ok && c.reserved_ok && c.trials_ok &&
             c.hmx_executed_ok && c.build_id_ok;
  return c;
}

/* The frozen device decision procedure, in the pre-registered order
 * (pre-registration section 8). Step 1 (host contract) is evaluated by the
 * caller with hmx_mode45_contract() and must fail closed first. */
static inline hmx_mode45_report_t hmx_mode45_classify(
    const int *result, int result_len, const int *meta, int meta_len,
    int expected_build_id) {
  hmx_mode45_report_t r;
  const int *s;
  r.contract = hmx_mode45_contract(result, result_len, meta, meta_len,
                                   expected_build_id);
  r.verdict = HMX_MODE45_UNEXPECTED_OUTPUT;
  r.cause = "unmatched";
  if (!r.contract.all_ok) {
    r.verdict = HMX_MODE45_HOST_CONTRACT_FAILED;
    r.cause = "contract";
    return r;
  }
  s = result;

  /* 2. Both stores of both subcases landed in their registered regions. */
  if (!r.contract.pos_writes_ok || !r.contract.neg_writes_ok) {
    r.verdict = HMX_MODE45_STORE_INTERFERENCE;
    r.cause = "region_write_count";
    return r;
  }

  /* 3. Registered one-ULP ceiling alternatives. */
  if (s[1] == 0x5021 || s[2] == 0x5421 || s[6] == 0x5421) {
    r.verdict = HMX_MODE45_ROUNDING_MISMATCH;
    r.cause = "one_ulp_ceiling";
    return r;
  }

  /* 4. S1 retention / bias-set control. */
  if (s[0] != HMX_MODE45_SLOT_S1_FIRST) {
    r.cause = "s1_first_anchor";
    return r;
  }
  if (s[1] == 0x0000) {
    r.verdict = HMX_MODE45_CAUSE_INCONCLUSIVE;
    r.cause = "s1_second_zero_ambiguous_acc_or_bias_or_convert_or_store";
    return r;
  }
  if (s[1] == 0x5420) {
    r.verdict = HMX_MODE45_CAUSE_INCONCLUSIVE;
    r.cause = "s1_second_set0_ambiguous_bias_load_or_rs_or_store";
    return r;
  }
  if (s[1] != HMX_MODE45_SLOT_S1_SECOND) {
    r.cause = "s1_second_unexpected";
    return r;
  }

  /* 5. Store-overwrite signature (second store clobbered the first). */
  if (s[2] == 0x0000 && s[3] == 0x5420) {
    r.verdict = HMX_MODE45_STORE_INTERFERENCE;
    r.cause = "first_store_overwritten_by_second";
    return r;
  }

  /* 6. S2 sign split. */
  if (s[2] != HMX_MODE45_SLOT_S2_POS_S0) {
    r.cause = "positive_half";
    return r;
  }
  if (s[3] != 0x0000) {
    r.cause = "positive_buffer_negative_lane";
    return r;
  }
  if (s[5] != 0x0000) {
    r.verdict = HMX_MODE45_CAUSE_INCONCLUSIVE;
    r.cause = "s2_neg_s0_nonzero_ambiguous_bias_shape_or_store";
    return r;
  }
  if (s[6] == 0x0000) {
    r.verdict = HMX_MODE45_CAUSE_INCONCLUSIVE;
    r.cause = "s2_negative_half_missing_ambiguous";
    return r;
  }
  if (s[6] != HMX_MODE45_SLOT_S2_NEG_S1) {
    r.cause = "negative_half_unexpected";
    return r;
  }
  if (s[4] != 0x0000 || s[7] != 0x0000) {
    r.cause = "zero_acc_lane";
    return r;
  }
  if (s[8] != HMX_MODE45_SLOT_S2_RECON_S0 ||
      s[9] != HMX_MODE45_SLOT_S2_RECON_S1 || s[10] != 0x0000) {
    r.cause = "reconstruction";
    return r;
  }

  r.verdict = HMX_MODE45_ACC_REUSE_SUPPORTED;
  r.cause = "all_registered_cells_match";
  return r;
}

#endif /* HEXATRAIN_HMX_MODE45_H */
