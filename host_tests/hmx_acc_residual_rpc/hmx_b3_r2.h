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
//   Offset  2048 (2048 B): wgt buffer (Whi then Wlo then Whi restored from L2)
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
//
// Operand lifetime (who owns which tile, per MMA):
//   MMA #1: act = Ahi,     wgt = Whi
//   MMA #2: act = Ahi,     wgt = Wlo      (Whi in VTCM is destroyed here)
//   MMA #3: act = Alo,     wgt = Whi      (restored from the L2 stWhi tile)
// The wgt buffer is 2 KiB, so MMA #2 necessarily overwrites Whi. MMA #3
// therefore needs an explicit restore source. The only source that fits is a
// third 2 KiB hosted staging tile in DSP L2/TCM (outside the 12 KiB HMX VTCM
// window): Alo, Wlo and the Whi copy are 3 x 2 KiB = 6 KiB of L2, not 4 KiB.
//
// Static transfer accounting (counts only; NO cycle/ns/overhead estimate is
// registered - see docs/hmx-v81-b3-r2-full-tile-prereg.md section 4.4):
//   prologue: Ahi + Whi -> VTCM, Alo + Wlo + Whi -> L2 stage = 5 tiles
//   mid-kernel stream: Wlo, Alo, Whi(restore)              = 3 tiles = 48 vectors
//

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

/* DSP L2/TCM operand staging area (NOT part of the 12 KiB HMX VTCM window).
 * Three 2 KiB packed tiles hold the operands that are not resident in VTCM.
 * stWhi is written once in the prologue and is read-only afterwards; it is
 * the sole registered restore source for MMA #3. */
#define HMX_B3_R2_STAGE_TILE_BYTES 2048
#define HMX_B3_R2_STAGE_ALO_OFFSET 0
#define HMX_B3_R2_STAGE_WLO_OFFSET 2048
#define HMX_B3_R2_STAGE_WHI_OFFSET 4096
#define HMX_B3_R2_STAGE_TOTAL      6144 /* 3 tiles: Alo, Wlo, Whi copy */
#define HMX_B3_R2_STAGE_TILES      3

/* Static transfer accounting. A tile is 2,048 B = 16 vectors of 128 B. */
#define HMX_B3_R2_VECTOR_BYTES          128
#define HMX_B3_R2_VECTORS_PER_TILE      16
#define HMX_B3_R2_STREAM_TILES_MIDKERNEL 3 /* Wlo, Alo, Whi(restore) */
#define HMX_B3_R2_STREAM_VECTORS_MIDKERNEL 48
#define HMX_B3_R2_PROLOGUE_TILES          5 /* Ahi, Whi, Alo, Wlo, Whi */
#define HMX_B3_R2_PROLOGUE_VECTORS        80

/* Whi restore source enumeration for MMA #3. */
#define HMX_B3_R2_WHI_RESTORE_NONE        0 /* not needed (single-MMA paths) */
#define HMX_B3_R2_WHI_RESTORE_L2_STAGE    1 /* the only legal source */
#define HMX_B3_R2_WHI_RESTORE_VTCM_SHADOW 2 /* illegal: needs +2 KiB, no room */
#define HMX_B3_R2_WHI_RESTORE_HOST_REFETCH 3 /* illegal: extra mid-kernel RPC */

/* Operand identity log, so a wrong tile in act/wgt is detectable. */
#define HMX_B3_R2_OP_EMPTY 0
#define HMX_B3_R2_OP_AHI   1
#define HMX_B3_R2_OP_ALO   2
#define HMX_B3_R2_OP_WHI   3
#define HMX_B3_R2_OP_WLO   4

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
  HMX_B3_R2_FAULT_INJECTED_DETECTED = 5,
  /* Operand supply gates: device-only, see the pre-registration sections 4.3-4.5. */
  HMX_B3_R2_OPERAND_SUPPLY_BLOCKED = 6,
  HMX_B3_R2_SYNC_UNPROVEN = 7
} hmx_b3_r2_verdict_t;

static inline const char *hmx_b3_r2_verdict_name(hmx_b3_r2_verdict_t v) {
  switch (v) {
    case HMX_B3_R2_FULL_TILE_CPU_READY: return "B3_R2_FULL_TILE_CPU_READY";
    case HMX_B3_R2_LAYOUT_OR_RESOURCE_BLOCKED: return "B3_R2_LAYOUT_OR_RESOURCE_BLOCKED";
    case HMX_B3_R2_NUMERICAL_BLOCKED: return "B3_R2_NUMERICAL_BLOCKED";
    case HMX_B3_R2_INSUFFICIENT_EVIDENCE: return "B3_R2_INSUFFICIENT_EVIDENCE";
    case HMX_B3_R2_HOST_CONTRACT_FAILED: return "B3_R2_HOST_CONTRACT_FAILED";
    case HMX_B3_R2_FAULT_INJECTED_DETECTED: return "B3_R2_FAULT_INJECTED_DETECTED";
    case HMX_B3_R2_OPERAND_SUPPLY_BLOCKED: return "B3_R2_OPERAND_SUPPLY_BLOCKED";
    case HMX_B3_R2_SYNC_UNPROVEN: return "B3_R2_SYNC_UNPROVEN";
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
  HMX_FAULT_COUNTER_MISMATCH = 13,
  HMX_FAULT_WHI_RESTORE_MISSING = 14
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
    case HMX_FAULT_WHI_RESTORE_MISSING: return "WHI_RESTORE_SOURCE_LOST";
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
  /* Operand supply / lifetime facts (B3 3-product streaming). */
  int operand_log[6];     /* [MMA1 act, wgt, MMA2 act, wgt, MMA3 act, wgt] */
  int stream_tiles_midkernel;  /* tiles copied between MMAs (expect 3) */
  int whi_restore_source;     /* HMX_B3_R2_WHI_RESTORE_* */
  int whi_restore_reads;      /* restores of Whi into wgt (expect 1) */
  int whi_restore_payload_ok; /* staging checksum matched on restore */
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
  int operand_lifetime_ok;
  int whi_restore_ok;
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

  /* Ahi*Whi -> Ahi*Wlo -> Alo*Whi, with Whi restored from the L2 staging
   * tile exactly once after MMA #2 destroyed the VTCM copy. */
  c.operand_lifetime_ok =
      (f->mma_count == HMX_B3_R2_B3_MMA_COUNT) &&
      (f->operand_log[0] == HMX_B3_R2_OP_AHI &&
       f->operand_log[1] == HMX_B3_R2_OP_WHI &&
       f->operand_log[2] == HMX_B3_R2_OP_AHI &&
       f->operand_log[3] == HMX_B3_R2_OP_WLO &&
       f->operand_log[4] == HMX_B3_R2_OP_ALO &&
       f->operand_log[5] == HMX_B3_R2_OP_WHI);
  c.whi_restore_ok =
      (f->stream_tiles_midkernel == HMX_B3_R2_STREAM_TILES_MIDKERNEL) &&
      (f->whi_restore_source == HMX_B3_R2_WHI_RESTORE_L2_STAGE) &&
      (f->whi_restore_reads == 1) &&
      (f->whi_restore_payload_ok == 1);

  c.all_ok = c.mxclracc_ok && c.mma_count_ok && c.convert_count_ok &&
             c.store_count_ok && c.recon_loops_ok && c.recon_source_ok &&
             c.mma_order_ok && c.rs_ok && c.sentinel_ok && c.layout_ok &&
             c.disasm_audit_ok && c.operand_lifetime_ok && c.whi_restore_ok;
  return c;
}

#endif /* HEXATRAIN_HMX_B3_R2_H */
