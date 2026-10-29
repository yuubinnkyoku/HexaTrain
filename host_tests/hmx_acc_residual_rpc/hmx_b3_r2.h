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

/* ---------------- Phase-Reused VTCM layout (candidate B, 2026-10-27) --------
 *
 * Alternative to sequential operand streaming. The four operand tiles are all
 * resident for the whole MMA phase, because the three output tiles are only
 * needed *after* the last MMA. Output regions then reuse operand regions whose
 * last use has already passed.
 *
 * MMA phase (4 tiles + 2 bias sets = 8,704 B):
 *   +0     Ahi     +2048  Alo     +4096  Whi     +6144  Wlo
 *   +8192  bias 0  +8448  bias 1
 * Output phase (peak 10,752 B):
 *   out_pos   -> +8192 (fresh region, no operand ever lived there)
 *   out_neg   -> +6144 (reuses Wlo; Wlo's last use is MMA #2)
 *   out_recon -> +0     (reuses Ahi; Ahi's last use is MMA #2)
 *   +2048 Alo and +4096 Whi are NEVER written: their last use is MMA #3.
 *
 * Mid-kernel operand transfers: 0. L2 staging: 2,048 B (only the Plain
 * subcase's completed output, saved before the B3 subcase re-packs +0).
 *
 * The remaining hazard is narrower than streaming's: no operand is written
 * between MMAs, only out_neg/out_recon after MMA #3 and both converts. It is
 * still NOT proven; see HMX_B3_R2_PR_SYNC_NOTE below. */
#define HMX_B3_R2_PR_AHI_OFFSET   0
#define HMX_B3_R2_PR_ALO_OFFSET   2048
#define HMX_B3_R2_PR_WHI_OFFSET   4096
#define HMX_B3_R2_PR_WLO_OFFSET   6144
#define HMX_B3_R2_PR_POS_OFFSET   8192
#define HMX_B3_R2_PR_BIAS0_OFFSET 10240
#define HMX_B3_R2_PR_BIAS1_OFFSET 10496
#define HMX_B3_R2_PR_MMA_PHASE_USED 8704  /* 4 tiles + 2 bias sets */
#define HMX_B3_R2_PR_PEAK_USED     10752  /* same peak as the streaming map */
#define HMX_B3_R2_PR_NEG_ALIAS_OWNER   HMX_B3_R2_PR_WLO_OFFSET /* last use: MMA #2 */
#define HMX_B3_R2_PR_RECON_ALIAS_OWNER HMX_B3_R2_PR_AHI_OFFSET /* last use: MMA #2 */
#define HMX_B3_R2_PR_MIDKERNEL_TILES   0
#define HMX_B3_R2_PR_L2_SAVE_BYTES     2048 /* Plain output, not 6,144 B */

/* PR layout sync status: the D1-D3 overwrite hazard is removed by construction
 * (no operand is written between MMAs), but the reuse of Ahi/Wlo *after* MMA #3
 * still needs "MMA #2 operand reads are done", which remains undocumented. */
#define HMX_B3_R2_PR_SYNC_NOTE "reuse_after_mma3_unproven"

/* ---------------- Candidate C: single shared output buffer (2026-10-29) -----
 *
 * A third layout, evaluated CPU-only as a design hypothesis. The four operand
 * tiles are resident for the whole kernel and are NEVER written again; all
 * three output stages share ONE 2 KiB VTCM buffer.
 *
 * VTCM (12,288 B window):
 *   +0     Ahi      (2048)   written once, prologue
 *   +2048  Alo      (2048)   written once, prologue
 *   +4096  Whi      (2048)   written once, prologue
 *   +6144  Wlo      (2048)   written once, prologue
 *   +8192  OUT      (2048)   pos -> neg -> recon, in place
 *   +10240 bias set 0 (256)
 *   +10496 bias set 1 (256)
 *   used 10,752 B, margin 1,536 B
 *
 * Output staging lives in DSP L2, outside the VTCM window:
 *   Plain: pos, neg and recon are all copied out (the shared buffer is reused
 *          by the B3 subcase); the three raw tiles are retained for evidence.
 *   B3:    pos and neg are copied out (neg BEFORE the in-place recon destroys
 *          it); the recon stays in OUT and is read back over the RPC boundary.
 *
 * Status: CPU hypothesis only. NOT registered, NOT the preferred design, and
 * NOT a Mode 46 pre-registration. Candidate B keeps CPU_VALIDATED_PREFERRED_DESIGN.
 * See docs/hmx-v81-cand-c-single-output-buffer.md. */
#define HMX_B3_R2_CC_AHI_OFFSET   0
#define HMX_B3_R2_CC_ALO_OFFSET   2048
#define HMX_B3_R2_CC_WHI_OFFSET   4096
#define HMX_B3_R2_CC_WLO_OFFSET   6144
#define HMX_B3_R2_CC_OUT_OFFSET   8192 /* the single shared output buffer */
#define HMX_B3_R2_CC_BIAS0_OFFSET 10240
#define HMX_B3_R2_CC_BIAS1_OFFSET 10496
#define HMX_B3_R2_CC_USED          10752
#define HMX_B3_R2_CC_MARGIN        (HMX_B3_R2_VTCM_TOTAL - HMX_B3_R2_CC_USED)
#define HMX_B3_R2_CC_OPERAND_TILES 4
#define HMX_B3_R2_CC_OUTPUT_TILES  1
#define HMX_B3_R2_CC_MIDKERNEL_OPERAND_TILES 0 /* no operand is written after the prologue */

/* DSP L2 arena (outside VTCM). Lifetimes:
 *   Plain pos/neg/recon  written during the Plain subcase, retained to the end
 *   B3 cur pos/neg       written during the B3 subcase
 * Peak under raw-bit retention = 5 tiles = 10,240 B, reached when B3's neg
 * evidence is copied out. If the Plain pos/neg tiles are folded into the
 * mutual-exclusivity counters during the recon pass, only the Plain recon tile
 * survives and the peak drops to 6,144 B. Both numbers are computed from the
 * lifetime table in the simulator, not asserted from the layout. */
#define HMX_B3_R2_CC_L2_PLAIN_POS_OFFSET   0
#define HMX_B3_R2_CC_L2_PLAIN_NEG_OFFSET   2048
#define HMX_B3_R2_CC_L2_PLAIN_RECON_OFFSET 4096
#define HMX_B3_R2_CC_L2_CUR_POS_OFFSET     6144
#define HMX_B3_R2_CC_L2_CUR_NEG_OFFSET     8192
#define HMX_B3_R2_CC_L2_PEAK_RAW_BYTES     10240 /* 5 tiles, raw retention */
#define HMX_B3_R2_CC_L2_PEAK_FOLD_BYTES    6144  /* 3 tiles, Plain pos/neg folded */

/* Static copy accounting (counts only; NO cycle/ns/overhead claim). */
#define HMX_B3_R2_CC_L2_CLEAR_PLAIN 3 /* pos, neg, recon */
#define HMX_B3_R2_CC_L2_CLEAR_B3    2 /* pos, neg (recon stays in VTCM) */
#define HMX_B3_R2_CC_L2_CLEAR_TOTAL  5
#define HMX_B3_R2_CC_L2_CLEAR_VECTORS 80 /* 5 tiles x 16 vectors of 128 B */

/* Candidate C sync dependencies. Only C2 is a hardware question; C1/C3 are the
 * direction Mode 45 already exercised (MX store -> later HVX load), C4/C5 are
 * program-order properties inside the kernel. */
typedef enum hmx_b3_r2_cc_sync {
  HMX_CC_SYNC_C1 = 0, /* Convert #1 MX store -> HVX load of OUT for the pos save */
  HMX_CC_SYNC_C2 = 1, /* HVX pos save complete -> Convert #2 MX store over OUT */
  HMX_CC_SYNC_C3 = 2, /* Convert #2 MX store -> HVX load of OUT for the neg save */
  HMX_CC_SYNC_C4 = 3, /* in-place recon: per-vector load before store, same thread */
  HMX_CC_SYNC_C5 = 4  /* Plain evidence copied out before the B3 Convert #1 */
} hmx_b3_r2_cc_sync_t;

#define HMX_CC_SYNC_IMPL_SUPPORTED 1 /* direction exercised by Mode 45 */
#define HMX_CC_SYNC_DEVICE_ONLY    2 /* no spec, no binary, no device exercise */
#define HMX_CC_SYNC_PROGRAM_ORDER  3 /* enforceable by source order, hardware silent */

#define HMX_B3_R2_CC_STATUS "CAND_C_CPU_MODEL_CONSISTENT"
#define HMX_B3_R2_CC_PRIORITY_NOTE "candidate_b_preference_unchanged"
#define HMX_B3_R2_CC_REGISTERED 0 /* not a registered layout */

/* ---------------- Candidate C: C2 minimal experiment, revision 2 ----------------
 *
 * This is NOT the Candidate C layout above. It is the 1-MMA map used by the C2
 * device pre-registration (docs/hmx-v81-cand-c-c2-device-prereg.md): one operand
 * pair, one output buffer X, no operand reuse, no reconstruction on the critical
 * path. Revision 2 exists because revision 1 violated a hard invariant:
 *
 *   revision 1 put guard_lo at +3968, inside the packed weight tile
 *   (+2048..+4095). Filling the guard with 0x5A5A before the MMA destroys the
 *   last 128 B of W -- crouton ic 30/31 of every output channel -- so the
 *   experiment corrupts its own input before it can test anything.
 *
 * Revision 2 moves both guards next to X, clear of every operand. The revision 1
 * offsets are kept below so the CPU overlap checker can be shown to reject them
 * (HMX_FAULT_C2_GUARD_LO_WEIGHT_OVERLAP); they are not usable for a run. */
#define HMX_B3_R2_C2_GUARD_BYTES    128
#define HMX_B3_R2_C2_ACT_OFFSET     0
#define HMX_B3_R2_C2_WGT_OFFSET     2048
#define HMX_B3_R2_C2_GUARD_LO_OFFSET 6016 /* adjacent to X, 1,920 B clear of wgt */
#define HMX_B3_R2_C2_X_OFFSET       6144 /* 2 KiB aligned; the single output */
#define HMX_B3_R2_C2_GUARD_HI_OFFSET 8192
#define HMX_B3_R2_C2_BIAS0_OFFSET   8448 /* 256 B aligned, shape 0 (positive) */
#define HMX_B3_R2_C2_BIAS1_OFFSET   8704 /* 256 B aligned, shape 4 (negative) */
#define HMX_B3_R2_C2_USED           8960
#define HMX_B3_R2_C2_MARGIN        (HMX_B3_R2_VTCM_TOTAL - HMX_B3_R2_C2_USED)

/* Superseded revision 1 map (kept only for the overlap regression). */
#define HMX_B3_R2_C2_V1_GUARD_LO_OFFSET 3968
#define HMX_B3_R2_C2_V1_X_OFFSET        4096
#define HMX_B3_R2_C2_V1_GUARD_HI_OFFSET 6144
#define HMX_B3_R2_C2_V1_BIAS0_OFFSET    8192
#define HMX_B3_R2_C2_V1_BIAS1_OFFSET    8448
#define HMX_B3_R2_C2_V1_USED            8704

/* DSP L2 arena for the C2 experiment. The positive tile P is the only payload
 * that leaves VTCM, and it is guarded on both sides INSIDE the arena, so no
 * guard offset is negative and the whole reservation is a positive-size
 * range that can be checked structurally. Revision 1 wrote P at L2 +0 with its
 * lower guard at L2 -128, which is not a checkable allocation. */
#define HMX_B3_R2_C2_L2_ARENA_BYTES 2560
#define HMX_B3_R2_C2_L2_P_GUARD_LO  256  /* 256..383 */
#define HMX_B3_R2_C2_L2_P_OFFSET    384  /* 384..2431 */
#define HMX_B3_R2_C2_L2_P_GUARD_HI  2432 /* 2432..2559 */

/* C2 observation classes. Order is the evaluation order: the first matching
 * class wins, which is what makes the special values get classified before any
 * C1/C2 signature is claimed. */
typedef enum hmx_c2_class {
  HMX_C2_CLASS_CONTRACT_FAILED = 0, /* counts / Rs / guards / build mismatch */
  HMX_C2_CLASS_STORE2_PATH_BROKEN = 1, /* X is not the negative payload at all */
  HMX_C2_CLASS_COPY_NOT_EXECUTED = 2, /* any 0xDEAD residue, partial included */
  HMX_C2_CLASS_SUBNORMAL_OR_ZERO_LOSS = 3, /* FTZ/clamp zero where a payload was due */
  HMX_C2_CLASS_MIXED_SIGNATURE = 4, /* both C1 and C2 evidence present */
  HMX_C2_CLASS_C1_SIGNATURE = 5,   /* P saw X before Store #1 */
  HMX_C2_CLASS_C2_SIGNATURE = 6,   /* P saw X after Store #2 */
  HMX_C2_CLASS_CAUSE_INCONCLUSIVE = 7, /* content present but not attributable */
  HMX_C2_CLASS_SUPPORTED = 8       /* all 1,024 cells are the positive payload */
} hmx_c2_class_t;

#define HMX_C2_SENTINEL_X  0xA5A5u /* pre-Store-#1 fill of X (bit 15 set) */
#define HMX_C2_SENTINEL_P  0xDEADu /* fill of P (bit 15 set) */
#define HMX_C2_SENTINEL_G  0x5A5Au /* guard fill (bit 15 clear, not a payload) */
#define HMX_C2_COPY_VECTORS 16
#define HMX_C2_PREREG_REVISION 2

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
  HMX_FAULT_WHI_RESTORE_MISSING = 14,
  /* Phase-reused layout faults (candidate B). */
  HMX_FAULT_PR_REUSE_BEFORE_LAST_USE = 15,
  HMX_FAULT_PR_OPERAND_CLOBBER = 16,
  HMX_FAULT_PR_OUTPUT_OVERLAP = 17,
  /* Single shared output buffer faults (candidate C). */
  HMX_FAULT_CC_POS_SAVE_MISSING = 18,     /* pos never copied out before Convert #2 */
  HMX_FAULT_CC_RESTORE_BEFORE_SAVE = 19,  /* Convert #2 issued mid-save */
  HMX_FAULT_CC_NEG_EVIDENCE_LOST = 20,    /* neg raw bits lost to the in-place recon */
  HMX_FAULT_CC_INPLACE_RW_CONFLICT = 21,  /* vector stored before its own neg load */
  HMX_FAULT_CC_PLAIN_CLOBBERED = 22,     /* Plain evidence lost to the shared buffer */
  /* C2 minimal experiment pre-registration faults (revision 2). */
  HMX_FAULT_C2_GUARD_LO_WEIGHT_OVERLAP = 23, /* rev1 guard_lo lands in the W tile */
  HMX_FAULT_C2_COPY_PARTIAL = 24,            /* only half of X reaches P */
  HMX_FAULT_C2_MIXED_SIGNATURE = 25,         /* C1-looking and C2-looking cells */
  HMX_FAULT_C2_FTZ_ZERO_LOSS = 26,           /* FTZ zero read as a C2 signature */
  HMX_FAULT_C2_L2_GUARD_OVERRUN = 27         /* copy destination shifted one vector into the L2 P guard */
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
    case HMX_FAULT_PR_REUSE_BEFORE_LAST_USE: return "PR_REUSE_BEFORE_LAST_USE";
    case HMX_FAULT_PR_OPERAND_CLOBBER: return "PR_ALIVE_OPERAND_CLOBBERED";
    case HMX_FAULT_PR_OUTPUT_OVERLAP: return "PR_OUTPUT_REGIONS_OVERLAP";
    case HMX_FAULT_CC_POS_SAVE_MISSING: return "CC_POS_SAVE_MISSING";
    case HMX_FAULT_CC_RESTORE_BEFORE_SAVE: return "CC_RESTORE_BEFORE_SAVE";
    case HMX_FAULT_CC_NEG_EVIDENCE_LOST: return "CC_NEG_EVIDENCE_LOST";
    case HMX_FAULT_CC_INPLACE_RW_CONFLICT: return "CC_INPLACE_RW_CONFLICT";
    case HMX_FAULT_CC_PLAIN_CLOBBERED: return "CC_PLAIN_CLOBBERED";
    case HMX_FAULT_C2_GUARD_LO_WEIGHT_OVERLAP: return "C2_GUARD_LO_WEIGHT_OVERLAP";
    case HMX_FAULT_C2_COPY_PARTIAL: return "C2_COPY_PARTIAL";
    case HMX_FAULT_C2_MIXED_SIGNATURE: return "C2_MIXED_SIGNATURE";
    case HMX_FAULT_C2_FTZ_ZERO_LOSS: return "C2_FTZ_ZERO_LOSS";
    case HMX_FAULT_C2_L2_GUARD_OVERRUN: return "C2_L2_P_GUARD_OVERRUN";
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
