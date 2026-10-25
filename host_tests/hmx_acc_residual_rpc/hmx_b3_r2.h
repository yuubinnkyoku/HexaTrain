// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Shared B3 + R2 integrated 32x32 full-tile contract header.
// Integrates:
//   - B3: FP16 hi/lo 3-product MMA accumulation (Ahi*Whi, Ahi*Wlo, Alo*Whi) into one ACC
//   - R2: ACC retention across 2 Convert operations (Convert #1 shape 0, Convert #2 shape 4)
//   - Downstream negative-clamp sign separation
//   - DSP HVX vsub.hf reconstruction (out_recon = out_pos - out_neg)
//
// VTCM 12 KiB allocation:
//   Offset     0 (2048 B): act buffer (Ahi then Alo)
//   Offset  2048 (2048 B): wgt buffer (Whi then Wlo then Whi)
//   Offset  4096 (2048 B): out_pos (Positive output buffer)
//   Offset  6144 (2048 B): out_neg (Negative output buffer)
//   Offset  8192 (2048 B): out_recon (HVX reconstructed output buffer)
//   Offset 10240 ( 256 B): Bias set 0 (scale +64, shape 0, bo 0)
//   Offset 10496 ( 256 B): Bias set 1 (scale +64, shape 4, bo 0)
//   Total used: 10752 B <= 12288 B (12 KiB reserve).
//
// Registered Rs values:
//   Convert #1: 0x0001 (set 0, retain ACC, no feedback, RN16)
//   Convert #2: 0x1000 (set 1, clear+swap after, no feedback, RN16)

#ifndef HEXATRAIN_HMX_B3_R2_H
#define HEXATRAIN_HMX_B3_R2_H

#include <stdint.h>
#include <stddef.h>

#define HMX_B3_R2_T8 32
#define HMX_B3_R2_CELLS (HMX_B3_R2_T8 * HMX_B3_R2_T8) /* 1024 */
#define HMX_B3_R2_TILE_BYTES (HMX_B3_R2_CELLS * 2)     /* 2048 */
#define HMX_B3_R2_VTCM_TOTAL 12288

/* VTCM layout offsets */
#define HMX_B3_R2_ACT_OFFSET   0
#define HMX_B3_R2_WGT_OFFSET   2048
#define HMX_B3_R2_POS_OFFSET   4096
#define HMX_B3_R2_NEG_OFFSET   6144
#define HMX_B3_R2_RECON_OFFSET 8192
#define HMX_B3_R2_BIAS0_OFFSET 10240
#define HMX_B3_R2_BIAS1_OFFSET 10496
#define HMX_B3_R2_VTCM_USED    10752

/* Bias parameters */
#define HMX_B3_R2_SCALE_BITS 0x5400u /* +64.0 */
#define HMX_B3_R2_SHAPE0     0u      /* x */
#define HMX_B3_R2_SHAPE4     4u      /* -x */
#define HMX_B3_R2_OUT_BIAS   0x0000u /* bo = 0 */

/* Rs control register encodings */
#define HMX_B3_R2_RS_CVT1 0x0001u /* bias set 0, retain ACC, no feedback */
#define HMX_B3_R2_RS_CVT2 0x1000u /* bias set 1, clear+swap, no feedback */

/* Sentinel byte/word for unwritten buffer detection */
#define HMX_B3_R2_SENTINEL_BYTE 0xA5u
#define HMX_B3_R2_SENTINEL_WORD 0xA5A5u

/* Registered execution counts for full test (Plain subcase + B3 subcase) */
#define HMX_B3_R2_PLAIN_MMA_COUNT   1
#define HMX_B3_R2_PLAIN_CVT_COUNT   2
#define HMX_B3_R2_PLAIN_STORE_COUNT 2
#define HMX_B3_R2_PLAIN_RECON_LOOPS 1

#define HMX_B3_R2_B3_MMA_COUNT      3
#define HMX_B3_R2_B3_CVT_COUNT      2
#define HMX_B3_R2_B3_STORE_COUNT    2
#define HMX_B3_R2_B3_RECON_LOOPS    1

#define HMX_B3_R2_TOTAL_MMA_COUNT   4
#define HMX_B3_R2_TOTAL_CVT_COUNT   4
#define HMX_B3_R2_TOTAL_STORE_COUNT 4
#define HMX_B3_R2_TOTAL_RECON_LOOPS 2
#define HMX_B3_R2_BIAS_SETS_LOADED  2
#define HMX_B3_R2_RECON_SOURCE_DSP_HVX 1

/* PRM pair-interleaved crouching indexing functions (80-N2040-62 Rev AA) */
static inline uint16_t hmx_b3_r2_act_u16(int s, int i) {
  return (uint16_t)(((s >> 1) * 64) + (i * 2) + (s & 1));
}

static inline uint16_t hmx_b3_r2_wgt_u16(int o, int i) {
  return (uint16_t)(((i >> 1) * 64) + (o * 2) + (i & 1));
}

static inline uint16_t hmx_b3_r2_store_u16(int s, int o) {
  return (uint16_t)(((s >> 1) * 64) + (o * 2) + (s & 1));
}

static inline int hmx_b3_r2_decode_pair(int idx) {
  return ((idx / 64) * 2) + (idx & 1);
}

static inline int hmx_b3_r2_decode_fast(int idx) {
  return (idx % 64) / 2;
}

/* Final mission verdict classifications */
typedef enum hmx_b3_r2_verdict {
  HMX_B3_R2_FULL_TILE_CPU_READY = 0,
  HMX_B3_R2_LAYOUT_OR_RESOURCE_BLOCKED = 1,
  HMX_B3_R2_NUMERICAL_BLOCKED = 2,
  HMX_B3_R2_INSUFFICIENT_EVIDENCE = 3,
  HMX_B3_R2_HOST_CONTRACT_FAILED = 4,
  HMX_B3_R2_FAULT_INJECTED_DETECTED = 5
} hmx_b3_r2_verdict_t;

static inline const char *hmx_b3_r2_verdict_name(hmx_b3_r2_verdict_t v) {
  switch (v) {
    case HMX_B3_R2_FULL_TILE_CPU_READY: return "B3_R2_FULL_TILE_CPU_READY";
    case HMX_B3_R2_LAYOUT_OR_RESOURCE_BLOCKED: return "B3_R2_LAYOUT_OR_RESOURCE_BLOCKED";
    case HMX_B3_R2_NUMERICAL_BLOCKED: return "B3_R2_NUMERICAL_BLOCKED";
    case HMX_B3_R2_INSUFFICIENT_EVIDENCE: return "B3_R2_INSUFFICIENT_EVIDENCE";
    case HMX_B3_R2_HOST_CONTRACT_FAILED: return "B3_R2_HOST_CONTRACT_FAILED";
    case HMX_B3_R2_FAULT_INJECTED_DETECTED: return "B3_R2_FAULT_INJECTED_DETECTED";
    default: return "UNKNOWN";
  }
}

/* Enumeration of fault injections for contract validation */
typedef enum hmx_b3_r2_fault {
  HMX_FAULT_NONE = 0,
  HMX_FAULT_MMA2_MISSING = 1,
  HMX_FAULT_MMA3_MISSING = 2,
  HMX_FAULT_MMA_ORDER = 3,
  HMX_FAULT_ACC_EARLY_CLEAR = 4,
  HMX_FAULT_OPERAND_SWAP = 5,
  HMX_FAULT_CROUTON_CORRUPT = 6,
  HMX_FAULT_SHAPE4_NOT_SELECTED = 7,
  HMX_FAULT_BO_NONZERO = 8,
  HMX_FAULT_BUFFER_OVERLAP = 9,
  HMX_FAULT_HVX_SUB_REVERSED = 10,
  HMX_FAULT_RECON_STORE_FAIL = 11,
  HMX_FAULT_HOST_FAKE_RECON = 12,
  HMX_FAULT_COUNTER_MISMATCH = 13
} hmx_b3_r2_fault_t;

static inline const char *hmx_b3_r2_fault_name(hmx_b3_r2_fault_t f) {
  switch (f) {
    case HMX_FAULT_NONE: return "NONE";
    case HMX_FAULT_MMA2_MISSING: return "MMA2_MISSING";
    case HMX_FAULT_MMA3_MISSING: return "MMA3_MISSING";
    case HMX_FAULT_MMA_ORDER: return "MMA_ORDER_REVERSED";
    case HMX_FAULT_ACC_EARLY_CLEAR: return "ACC_EARLY_CLEAR";
    case HMX_FAULT_OPERAND_SWAP: return "OPERAND_TILE_SWAPPED";
    case HMX_FAULT_CROUTON_CORRUPT: return "CROUTON_PACKING_CORRUPTED";
    case HMX_FAULT_SHAPE4_NOT_SELECTED: return "SHAPE4_NOT_SELECTED";
    case HMX_FAULT_BO_NONZERO: return "BO_NONZERO";
    case HMX_FAULT_BUFFER_OVERLAP: return "BUFFER_OVERLAP";
    case HMX_FAULT_HVX_SUB_REVERSED: return "HVX_SUB_REVERSED";
    case HMX_FAULT_RECON_STORE_FAIL: return "RECON_STORE_FAIL";
    case HMX_FAULT_HOST_FAKE_RECON: return "HOST_FAKE_RECON";
    case HMX_FAULT_COUNTER_MISMATCH: return "COUNTER_MISMATCH";
    default: return "UNKNOWN_FAULT";
  }
}

/* Structural execution facts passed to host contract classifier */
typedef struct hmx_b3_r2_facts {
  int mxclracc_count;
  int mma_count;
  int convert_count;
  int store_count;
  int recon_loops;
  int recon_source;       /* 1 = DSP HVX, 0 = host fallback */
  int mma_order_id;       /* 123 = #1 AhiWhi, #2 AhiWlo, #3 AloWhi */
  int sentinel_hits;
  int buffer_overlap;
  uint32_t rs_cvt1;
  uint32_t rs_cvt2;
  int disasm_mma_count;   /* from static disassembly audit */
} hmx_b3_r2_facts_t;

typedef struct hmx_b3_r2_contract {
  int mxclracc_ok;
  int mma_count_ok;
  int convert_count_ok;
  int store_count_ok;
  int recon_loops_ok;
  int recon_source_ok;
  int mma_order_ok;
  int rs_ok;
  int sentinel_ok;
  int layout_ok;
  int disasm_audit_ok;
  int all_ok;
} hmx_b3_r2_contract_t;

static inline hmx_b3_r2_contract_t hmx_b3_r2_check_contract(const hmx_b3_r2_facts_t *f) {
  hmx_b3_r2_contract_t c;
  c.mxclracc_ok = (f->mxclracc_count == 1 || f->mxclracc_count == 2);
  c.mma_count_ok = (f->mma_count == HMX_B3_R2_B3_MMA_COUNT || f->mma_count == HMX_B3_R2_TOTAL_MMA_COUNT);
  c.convert_count_ok = (f->convert_count == HMX_B3_R2_B3_CVT_COUNT || f->convert_count == HMX_B3_R2_TOTAL_CVT_COUNT);
  c.store_count_ok = (f->store_count == HMX_B3_R2_B3_STORE_COUNT || f->store_count == HMX_B3_R2_TOTAL_STORE_COUNT);
  c.recon_loops_ok = (f->recon_loops == HMX_B3_R2_B3_RECON_LOOPS || f->recon_loops == HMX_B3_R2_TOTAL_RECON_LOOPS);
  c.recon_source_ok = (f->recon_source == HMX_B3_R2_RECON_SOURCE_DSP_HVX);
  c.mma_order_ok = (f->mma_order_id == 123);
  c.rs_ok = (f->rs_cvt1 == HMX_B3_R2_RS_CVT1 && f->rs_cvt2 == HMX_B3_R2_RS_CVT2);
  c.sentinel_ok = (f->sentinel_hits == 0);
  c.layout_ok = (f->buffer_overlap == 0);
  c.disasm_audit_ok = (f->disasm_mma_count == f->mma_count);

  c.all_ok = c.mxclracc_ok && c.mma_count_ok && c.convert_count_ok &&
             c.store_count_ok && c.recon_loops_ok && c.recon_source_ok &&
             c.mma_order_ok && c.rs_ok && c.sentinel_ok && c.layout_ok &&
             c.disasm_audit_ok;
  return c;
}

#endif /* HEXATRAIN_HMX_B3_R2_H */
