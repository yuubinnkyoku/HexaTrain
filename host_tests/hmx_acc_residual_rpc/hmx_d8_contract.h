// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// D8 (Candidate B minimal device experiment) revision-3 CPU contract.
//
// This header is the single machine-readable definition of the metadata
// layout, the q -> condition selector, the guard evidence and the host-side
// contract gate proposed in
//   docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md
//   docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md
//
// Status: PROPOSAL, CPU-ONLY. It authorises nothing: no DSP source, no DSP
// build, no QAIRT, no ADB, no device contact, no Mode 46 registration.
//
// Ownership rule (the reason revision 2 could not be implemented):
//   metadata[0..8] are written by the shared run() wrapper and keep their
//   existing meaning on every mode. D8 writes ONLY metadata[9..31].
//   metadata[3], [7] and [8] are (re)written by run() after the handler, so
//   D8 must never use them. The only exception is metadata[30] (cleanup),
//   which run() writes for mode 46 after the handler; see the rev3 contract.
//
// All 32-bit fields are transported as signed FastRPC `long` but compared as
// unsigned bit patterns. Packed fields use only low bits, so a value with the
// sign bit set (a Build ID or an FNV) cannot corrupt a packed field.
//
// Dependency-free plain C: no includes, no stdio. Safe for the DSP skel.

#ifndef HEXATRAIN_HMX_D8_CONTRACT_H
#define HEXATRAIN_HMX_D8_CONTRACT_H

/* ---------------------------------------------------------------- identity */

#define HMX_D8_MODE 46
#define HMX_D8_SCHEMA 0x0D8A5003u   /* positive; must not have bit31 set */
#define HMX_D8_LAYOUT_REVISION 3

/* q is not free: the shared run() gate accepts q in [6,10]. Mode 46 maps
 * q=6..9 to conditions A..D and rejects q=10 (and anything outside 6..9)
 * before execution. q=0..3 from revision 2 is impossible. */
#define HMX_D8_Q_FIRST 6
#define HMX_D8_Q_LAST 9
#define HMX_D8_Q_IDENTITY 6         /* mode 30 identity probe uses its own q */
#define HMX_D8_Q_COUNT 4

/* The identity gate is the existing mode 30 call: q=6, mode=30, arch=81,
 * metadata[7]=0, metadata[8]=0. It is independent of the run's q. */
#define HMX_D8_IDENTITY_MODE 30
#define HMX_D8_IDENTITY_Q 6

/* --------------------------------------------------------------- capacity */

#define HMX_D8_RESULT_SLOTS 1024    /* 4,096 B */
#define HMX_D8_RESULT_BYTES 4096
#define HMX_D8_METADATA_LEN 32      /* 128 B */
#define HMX_D8_TOTAL_BYTES 4224

/* ------------------------------------------------------------ fixed layout */

#define HMX_D8_VTCM_BYTES 12288
#define HMX_D8_AHI_OFFSET 0
#define HMX_D8_ALO_OFFSET 2048
#define HMX_D8_WHI_OFFSET 4096
#define HMX_D8_WLO_OFFSET 6144
#define HMX_D8_OUT_OFFSET 8192
#define HMX_D8_BIAS0_OFFSET 10240
#define HMX_D8_BIAS1_OFFSET 10496
#define HMX_D8_USED_BYTES 10752
#define HMX_D8_GUARD_OFFSET 10752
#define HMX_D8_GUARD_BYTES 1536
#define HMX_D8_GUARD_VECTORS 12     /* 12 x 128 B */
#define HMX_D8_VECTOR_BYTES 128
#define HMX_D8_TILE_VECTORS 16      /* 16 x 128 B per 2 KiB tile */

#define HMX_D8_RS_CVT1 0x0001u

/* Fill mask bits: bit0 out_pos (+8192), bit1 Wlo (+6144), bit2 Ahi (+0). */
#define HMX_D8_FILL_OUT 1
#define HMX_D8_FILL_WLO 2
#define HMX_D8_FILL_AHI 4

static inline int hmx_d8_condition_for_q(int q) {
  if (q < HMX_D8_Q_FIRST || q > HMX_D8_Q_LAST) return -1;
  return q - HMX_D8_Q_FIRST;
}

static inline int hmx_d8_q_for_condition(int condition) {
  if (condition < 0 || condition > 3) return -1;
  return condition + HMX_D8_Q_FIRST;
}

static inline int hmx_d8_fill_mask_for_condition(int condition) {
  switch (condition) {
    case 0: return HMX_D8_FILL_OUT;
    case 1: return HMX_D8_FILL_OUT | HMX_D8_FILL_WLO | HMX_D8_FILL_AHI;
    case 2: return HMX_D8_FILL_OUT | HMX_D8_FILL_WLO;
    case 3: return HMX_D8_FILL_OUT | HMX_D8_FILL_AHI;
    default: return 0;
  }
}

/* ------------------------------------------------------- metadata indices */

/* Words 0..8 are NOT listed here: they belong to run(). D8 never writes
 * them. A commentary copy of their meaning is below for the host gate. */
enum hmx_d8_meta_index {
  HMX_D8_META_BUILD_ID = 9,    /* stamped build id, one echo (unsigned cmp) */
  HMX_D8_META_MXCLRACC = 10,   /* exactly 1 */
  HMX_D8_META_MMA = 11,        /* exactly 3 */
  HMX_D8_META_CONVERT = 12,    /* exactly 1 */
  HMX_D8_META_STORE = 13,      /* exactly 1 */
  HMX_D8_META_PACK = 14,       /* exactly 4 */
  HMX_D8_META_BIAS = 15,       /* exactly 2 */
  HMX_D8_META_FILL = 16,       /* mask | word << 8 | form << 24 */
  HMX_D8_META_ADDRESS = 17,    /* out | Wlo << 7 | Ahi << 14, 128 B units */
  HMX_D8_META_SELECTOR = 18,   /* condition | q << 8 */
  HMX_D8_META_FNV_AHI = 19,
  HMX_D8_META_FNV_ALO = 20,
  HMX_D8_META_FNV_WHI = 21,
  HMX_D8_META_FNV_WLO = 22,
  HMX_D8_META_GUARD_PRE = 23,     /* FNV-1a 32 of the guard at entry */
  HMX_D8_META_GUARD_POST = 24,    /* FNV-1a 32 of the guard after Store #1 */
  HMX_D8_META_GUARD_STATUS = 25,  /* status | first_bad | methods */
  HMX_D8_META_FAILURE = 26,       /* hmx_d8_failure_t; 0 = none */
  HMX_D8_META_SCHEMA = 27,        /* HMX_D8_SCHEMA */
  HMX_D8_META_READBACK = 28,      /* trace | readback | hvx | slots */
  HMX_D8_META_RESERVED_29 = 29,   /* 0 */
  HMX_D8_META_CLEANUP = 30,       /* run() for mode 46: first cleanup error */
  HMX_D8_META_RESERVED_31 = 31    /* 0 */
};

/* Generic metadata words written by run() before dispatch. The D8 host gate
 * re-checks them but D8 never owns them. */
#define HMX_D8_META_GENERIC_MAGIC 0
#define HMX_D8_META_GENERIC_ARCH 1
#define HMX_D8_META_GENERIC_VTCM 2
#define HMX_D8_META_GENERIC_ELAPSED 3
#define HMX_D8_META_GENERIC_POWER 4
#define HMX_D8_META_GENERIC_LOCK_A 5
#define HMX_D8_META_GENERIC_LOCK_B 6
#define HMX_D8_META_GENERIC_HMX 7
#define HMX_D8_META_GENERIC_TRIALS 8

#define HMX_D8_MAGIC 0x484d5831
#define HMX_D8_ARCH 81
#define HMX_D8_TRIALS 1

/* ------------------------------------------------------------ field masks */

#define HMX_D8_FILL_MASK_BITS 0x7
#define HMX_D8_FILL_WORD_SHIFT 8
#define HMX_D8_FILL_WORD_BITS 0xFFFF
#define HMX_D8_FILL_FORM_SHIFT 24
#define HMX_D8_FILL_FORM_BITS 0xF
#define HMX_D8_FILL_FORM_16X128 1
#define HMX_D8_FILL_WORD 0xA5A5

#define HMX_D8_ADDR_BITS 0x7F
#define HMX_D8_ADDR_INVALID 0x7F     /* not representable / not 128 B aligned */
#define HMX_D8_ADDR_OUT 64          /* 8192 / 128 */
#define HMX_D8_ADDR_WLO 48          /* 6144 / 128 */
#define HMX_D8_ADDR_AHI 0           /* 0 / 128 */
#define HMX_D8_ADDR_WLO_SHIFT 7
#define HMX_D8_ADDR_AHI_SHIFT 14

#define HMX_D8_SELECTOR_COND_BITS 0xF
#define HMX_D8_SELECTOR_Q_SHIFT 8
#define HMX_D8_SELECTOR_Q_BITS 0x1F

#define HMX_D8_GUARD_STATUS_BITS 0xFF
#define HMX_D8_GUARD_FIRST_BAD_SHIFT 8
#define HMX_D8_GUARD_FIRST_BAD_BITS 0xFFF
#define HMX_D8_GUARD_FIRST_BAD_NONE 0xFFF
#define HMX_D8_GUARD_INIT_SHIFT 20
#define HMX_D8_GUARD_INIT_BITS 0xF
#define HMX_D8_GUARD_CHECK_SHIFT 24
#define HMX_D8_GUARD_CHECK_BITS 0xF
#define HMX_D8_GUARD_STATUS_OK 0
#define HMX_D8_GUARD_STATUS_PRE_NONZERO 1
#define HMX_D8_GUARD_STATUS_POST_CHANGED 2
#define HMX_D8_GUARD_STATUS_INIT_MISSING 3
#define HMX_D8_GUARD_STATUS_CHECK_MISSING 4
#define HMX_D8_GUARD_STATUS_INCONSISTENT 5
#define HMX_D8_GUARD_INIT_NONE 0
#define HMX_D8_GUARD_INIT_HVX12 1
#define HMX_D8_GUARD_CHECK_NONE 0
#define HMX_D8_GUARD_CHECK_BYTE 1

#define HMX_D8_READBACK_TRACE_SHIFT 0
#define HMX_D8_READBACK_TRACE_BITS 0x3
#define HMX_D8_READBACK_DONE_SHIFT 2
#define HMX_D8_READBACK_DONE_BITS 0x1
#define HMX_D8_READBACK_HVX_SHIFT 3
#define HMX_D8_READBACK_HVX_BITS 0xFF
#define HMX_D8_READBACK_SLOTS_SHIFT 11
#define HMX_D8_READBACK_SLOTS_BITS 0x7FF
#define HMX_D8_TRACE_NOT_PROVIDED 0

/* ------------------------------------------------------------- fail codes */

/* Codes 1..2 cannot appear in metadata: they are pre-execution failures, so
 * run() returns non-zero and rout data is not propagated. They exist so the
 * host has one reason namespace for the RPC status mapping. */
typedef enum hmx_d8_failure {
  HMX_D8_FAIL_NONE = 0,
  HMX_D8_FAIL_ARGS = 1,
  HMX_D8_FAIL_Q = 2,
  HMX_D8_FAIL_VTCM = 3,
  HMX_D8_FAIL_ALIGN = 4,
  HMX_D8_FAIL_HVX_LOCK = 5,
  HMX_D8_FAIL_GUARD_INIT = 6,
  HMX_D8_FAIL_GUARD_PRE = 7,
  HMX_D8_FAIL_GUARD_POST = 8,
  HMX_D8_FAIL_PACK = 9,
  HMX_D8_FAIL_BIAS = 10,
  HMX_D8_FAIL_MXCLRACC = 11,
  HMX_D8_FAIL_MMA = 12,
  HMX_D8_FAIL_CONVERT = 13,
  HMX_D8_FAIL_STORE = 14,
  HMX_D8_FAIL_FILL = 15,
  HMX_D8_FAIL_FNV = 16,
  HMX_D8_FAIL_READBACK = 17,
  HMX_D8_FAIL_CLEANUP = 18,
  HMX_D8_FAIL_INTERNAL = 19,
  HMX_D8_FAIL_COUNT = 20
} hmx_d8_failure_t;

static inline const char *hmx_d8_failure_name(int code) {
  switch (code) {
    case HMX_D8_FAIL_NONE: return "NONE";
    case HMX_D8_FAIL_ARGS: return "ARGS";
    case HMX_D8_FAIL_Q: return "Q";
    case HMX_D8_FAIL_VTCM: return "VTCM";
    case HMX_D8_FAIL_ALIGN: return "ALIGN";
    case HMX_D8_FAIL_HVX_LOCK: return "HVX_LOCK";
    case HMX_D8_FAIL_GUARD_INIT: return "GUARD_INIT";
    case HMX_D8_FAIL_GUARD_PRE: return "GUARD_PRE";
    case HMX_D8_FAIL_GUARD_POST: return "GUARD_POST";
    case HMX_D8_FAIL_PACK: return "PACK";
    case HMX_D8_FAIL_BIAS: return "BIAS";
    case HMX_D8_FAIL_MXCLRACC: return "MXCLRACC";
    case HMX_D8_FAIL_MMA: return "MMA";
    case HMX_D8_FAIL_CONVERT: return "CONVERT";
    case HMX_D8_FAIL_STORE: return "STORE";
    case HMX_D8_FAIL_FILL: return "FILL";
    case HMX_D8_FAIL_FNV: return "FNV";
    case HMX_D8_FAIL_READBACK: return "READBACK";
    case HMX_D8_FAIL_CLEANUP: return "CLEANUP";
    case HMX_D8_FAIL_INTERNAL: return "INTERNAL";
    default: return "UNKNOWN";
  }
}

/* ------------------------------------------------------------ guard digest */

/* FNV-1a 32 over the guard bytes. The guard is 1,536 B; the expected entry
 * value is the digest of 1,536 zero bytes. */
static inline unsigned int hmx_d8_fnv1a32(const unsigned char *p, int n) {
  unsigned int h = 2166136261u;
  int i;
  for (i = 0; i < n; ++i) {
    h ^= (unsigned int)p[i];
    h *= 16777619u;
  }
  return h;
}

static inline unsigned int hmx_d8_guard_zero_fnv(void) {
  unsigned int h = 2166136261u;
  int i;
  for (i = 0; i < HMX_D8_GUARD_BYTES; ++i) {
    h *= 16777619u; /* xor 0 */
  }
  return h;
}

/* --------------------------------------------------------- contract check */

typedef struct hmx_d8_contract {
  int meta_len_ok;
  int generic_ok;
  int schema_ok;
  int build_id_ok;
  int selector_ok;
  int counts_ok;
  int mxclracc_ok;
  int mma_ok;
  int convert_ok;
  int store_ok;
  int pack_ok;
  int bias_ok;
  int fill_mask_ok;
  int fill_word_ok;
  int fill_form_ok;
  int address_ok;
  int fnv_ok;
  int guard_ok;
  int guard_status_ok;
  int failure_ok;
  int readback_ok;
  int cleanup_ok;
  int reserved_ok;
  int all_ok;
  int condition;          /* decoded selector, or -1 */
  int q_echo;             /* decoded selector, or -1 */
  const char *first_bad;  /* first failing field, "ok" when all_ok */
} hmx_d8_contract_t;

static inline unsigned int hmx_d8_u32(int v) { return (unsigned int)v; }

/* expected_build_id may be 0 for an unstamped manual build; the strict
 * orchestrator gate still requires non-zero before any device run. */
static inline hmx_d8_contract_t hmx_d8_contract_check(
    const int *meta, int meta_len, int requested_q, int expected_build_id) {
  hmx_d8_contract_t c;
  unsigned int fill, addr, sel, gs, rb;
  int condition;

  c.meta_len_ok = (meta != 0) && (meta_len == HMX_D8_METADATA_LEN);
  c.generic_ok = 0;
  c.schema_ok = 0;
  c.build_id_ok = 0;
  c.selector_ok = 0;
  c.counts_ok = 0;
  c.mxclracc_ok = 0;
  c.mma_ok = 0;
  c.convert_ok = 0;
  c.store_ok = 0;
  c.pack_ok = 0;
  c.bias_ok = 0;
  c.fill_mask_ok = 0;
  c.fill_word_ok = 0;
  c.fill_form_ok = 0;
  c.address_ok = 0;
  c.fnv_ok = 0;
  c.guard_ok = 0;
  c.guard_status_ok = 0;
  c.failure_ok = 0;
  c.readback_ok = 0;
  c.cleanup_ok = 0;
  c.reserved_ok = 0;
  c.all_ok = 0;
  c.condition = -1;
  c.q_echo = -1;
  c.first_bad = "meta_len";
  if (!c.meta_len_ok) return c;

  condition = hmx_d8_condition_for_q(requested_q);
  c.generic_ok = hmx_d8_u32(meta[HMX_D8_META_GENERIC_MAGIC]) ==
                     hmx_d8_u32(HMX_D8_MAGIC) &&
                 meta[HMX_D8_META_GENERIC_ARCH] == HMX_D8_ARCH &&
                 meta[HMX_D8_META_GENERIC_VTCM] == HMX_D8_VTCM_BYTES &&
                 meta[HMX_D8_META_GENERIC_HMX] == 1 &&
                 meta[HMX_D8_META_GENERIC_TRIALS] == HMX_D8_TRIALS;
  if (!c.generic_ok) { c.first_bad = "generic"; return c; }

  c.schema_ok = hmx_d8_u32(meta[HMX_D8_META_SCHEMA]) ==
                hmx_d8_u32(HMX_D8_SCHEMA);
  if (!c.schema_ok) { c.first_bad = "schema"; return c; }

  c.build_id_ok = (expected_build_id == 0) ||
                  (hmx_d8_u32(meta[HMX_D8_META_BUILD_ID]) ==
                   hmx_d8_u32(expected_build_id));
  if (!c.build_id_ok) { c.first_bad = "build_id"; return c; }

  /* Selector: the echoed condition must be q-6 and the echoed q must be the
   * requested q. A build that writes the Build ID here, or a C/D swap in the
   * q table, fails both checks. */
  sel = hmx_d8_u32(meta[HMX_D8_META_SELECTOR]);
  c.condition = (int)(sel & HMX_D8_SELECTOR_COND_BITS);
  c.q_echo = (int)((sel >> HMX_D8_SELECTOR_Q_SHIFT) &
                   HMX_D8_SELECTOR_Q_BITS);
  c.selector_ok = (condition >= 0) && c.condition == condition &&
                  c.q_echo == requested_q &&
                  (sel & 0xFFFFE0F0u) == 0;
  if (!c.selector_ok) { c.first_bad = "selector"; return c; }

  c.mxclracc_ok = meta[HMX_D8_META_MXCLRACC] == 1;
  c.mma_ok = meta[HMX_D8_META_MMA] == 3;
  c.convert_ok = meta[HMX_D8_META_CONVERT] == 1;
  c.store_ok = meta[HMX_D8_META_STORE] == 1;
  c.pack_ok = meta[HMX_D8_META_PACK] == 4;
  c.bias_ok = meta[HMX_D8_META_BIAS] == 2;
  c.counts_ok = c.mxclracc_ok && c.mma_ok && c.convert_ok && c.store_ok &&
                c.pack_ok && c.bias_ok;
  if (!c.mxclracc_ok) { c.first_bad = "mxclracc"; return c; }
  if (!c.mma_ok) { c.first_bad = "mma"; return c; }
  if (!c.convert_ok) { c.first_bad = "convert"; return c; }
  if (!c.store_ok) { c.first_bad = "store"; return c; }
  if (!c.pack_ok) { c.first_bad = "pack"; return c; }
  if (!c.bias_ok) { c.first_bad = "bias"; return c; }

  fill = hmx_d8_u32(meta[HMX_D8_META_FILL]);
  c.fill_mask_ok = (int)(fill & HMX_D8_FILL_MASK_BITS) ==
                   hmx_d8_fill_mask_for_condition(condition);
  c.fill_word_ok = (int)((fill >> HMX_D8_FILL_WORD_SHIFT) &
                         HMX_D8_FILL_WORD_BITS) == HMX_D8_FILL_WORD;
  c.fill_form_ok = (int)((fill >> HMX_D8_FILL_FORM_SHIFT) &
                         HMX_D8_FILL_FORM_BITS) == HMX_D8_FILL_FORM_16X128 &&
                   (fill & 0xF00000F8u) == 0;
  if (!c.fill_mask_ok) { c.first_bad = "fill_mask"; return c; }
  if (!c.fill_word_ok) { c.first_bad = "fill_word"; return c; }
  if (!c.fill_form_ok) { c.first_bad = "fill_form"; return c; }

  addr = hmx_d8_u32(meta[HMX_D8_META_ADDRESS]);
  c.address_ok = ((addr & HMX_D8_ADDR_BITS) == HMX_D8_ADDR_OUT) &&
                 (((addr >> HMX_D8_ADDR_WLO_SHIFT) & HMX_D8_ADDR_BITS) ==
                  ((condition == 1 || condition == 2) ? HMX_D8_ADDR_WLO : 0)) &&
                 (((addr >> HMX_D8_ADDR_AHI_SHIFT) & HMX_D8_ADDR_BITS) ==
                  HMX_D8_ADDR_AHI) &&
                 (addr & 0xFFE00000u) == 0;
  if (!c.address_ok) { c.first_bad = "address"; return c; }

  c.fnv_ok = hmx_d8_u32(meta[HMX_D8_META_FNV_AHI]) == 0xC6231266u &&
             hmx_d8_u32(meta[HMX_D8_META_FNV_ALO]) == 0x3B2D22D2u &&
             hmx_d8_u32(meta[HMX_D8_META_FNV_WHI]) == 0x8B88609Fu &&
             hmx_d8_u32(meta[HMX_D8_META_FNV_WLO]) == 0xA66AF695u;
  if (!c.fnv_ok) { c.first_bad = "operand_fnv"; return c; }

  c.guard_ok = hmx_d8_u32(meta[HMX_D8_META_GUARD_PRE]) ==
                   hmx_d8_guard_zero_fnv() &&
               hmx_d8_u32(meta[HMX_D8_META_GUARD_POST]) ==
                   hmx_d8_u32(meta[HMX_D8_META_GUARD_PRE]);
  if (!c.guard_ok) { c.first_bad = "guard_digest"; return c; }

  gs = hmx_d8_u32(meta[HMX_D8_META_GUARD_STATUS]);
  c.guard_status_ok =
      (int)(gs & HMX_D8_GUARD_STATUS_BITS) == HMX_D8_GUARD_STATUS_OK &&
      (int)((gs >> HMX_D8_GUARD_FIRST_BAD_SHIFT) &
            HMX_D8_GUARD_FIRST_BAD_BITS) == HMX_D8_GUARD_FIRST_BAD_NONE &&
      (int)((gs >> HMX_D8_GUARD_INIT_SHIFT) & HMX_D8_GUARD_INIT_BITS) ==
          HMX_D8_GUARD_INIT_HVX12 &&
      (int)((gs >> HMX_D8_GUARD_CHECK_SHIFT) & HMX_D8_GUARD_CHECK_BITS) ==
          HMX_D8_GUARD_CHECK_BYTE &&
      (gs & 0xF0000000u) == 0;
  if (!c.guard_status_ok) { c.first_bad = "guard_status"; return c; }

  c.failure_ok = meta[HMX_D8_META_FAILURE] == HMX_D8_FAIL_NONE;
  if (!c.failure_ok) { c.first_bad = "failure_code"; return c; }

  rb = hmx_d8_u32(meta[HMX_D8_META_READBACK]);
  c.readback_ok =
      (int)((rb >> HMX_D8_READBACK_TRACE_SHIFT) &
            HMX_D8_READBACK_TRACE_BITS) == HMX_D8_TRACE_NOT_PROVIDED &&
      (int)((rb >> HMX_D8_READBACK_DONE_SHIFT) &
            HMX_D8_READBACK_DONE_BITS) == 1 &&
      (int)((rb >> HMX_D8_READBACK_HVX_SHIFT) &
            HMX_D8_READBACK_HVX_BITS) == 0 &&
      (int)((rb >> HMX_D8_READBACK_SLOTS_SHIFT) &
            HMX_D8_READBACK_SLOTS_BITS) == HMX_D8_RESULT_SLOTS &&
      (rb & 0xFFC00000u) == 0;
  if (!c.readback_ok) { c.first_bad = "readback"; return c; }

  c.cleanup_ok = meta[HMX_D8_META_CLEANUP] == 0;
  if (!c.cleanup_ok) { c.first_bad = "cleanup"; return c; }

  c.reserved_ok = meta[HMX_D8_META_RESERVED_29] == 0 &&
                  meta[HMX_D8_META_RESERVED_31] == 0;
  if (!c.reserved_ok) { c.first_bad = "reserved"; return c; }

  c.all_ok = 1;
  c.first_bad = "ok";
  return c;
}

/* Result-slot contract: 1,024 slots, each a zero-extended u16. */
static inline int hmx_d8_results_ok(const int *results, int len) {
  int i;
  if (!results || len != HMX_D8_RESULT_SLOTS) return 0;
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) {
    if (results[i] < 0 || results[i] > 0xFFFF) return 0;
  }
  return 1;
}

static inline int hmx_d8_results_equal(const int *a, const int *b) {
  int i;
  if (!a || !b) return 0;
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

#endif /* HEXATRAIN_HMX_D8_CONTRACT_H */
