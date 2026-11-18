// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// D8 (Candidate B minimal device experiment) Mode 46 constants, shared by
// probe_dsp.c (executes it), probe_host.c (classifies the device result) and
// the CPU model. The machine-readable contract lives in hmx_d8_contract.h;
// this header only adds the DSP-side names that the contract does not need.
//
// Everything numeric here is either taken from hmx_d8_contract.h or is a
// register encoding already used by mode 43/44/45. No registered value is
// redefined: the offsets, the fill mask, the FNV expectations and the
// reference digests all come from the contract header.
//
// Dependency-free plain C: safe for the DSP skel.

#ifndef HEXATRAIN_HMX_MODE46_H
#define HEXATRAIN_HMX_MODE46_H

#include "hmx_d8_contract.h"

/* Mode identity. HMX_D8_MODE, HMX_D8_SCHEMA, the q range and the layout
 * offsets all come from hmx_d8_contract.h. */
#define HMX_MODE46_MODE HMX_D8_MODE
#define HMX_MODE46_RESULT_SLOTS HMX_D8_RESULT_SLOTS
#define HMX_MODE46_METADATA_LEN HMX_D8_METADATA_LEN
#define HMX_MODE46_TRIALS HMX_D8_TRIALS

/* VTCM layout (contract section "fixed layout"). */
#define HMX_MODE46_AHI_OFFSET HMX_D8_AHI_OFFSET
#define HMX_MODE46_ALO_OFFSET HMX_D8_ALO_OFFSET
#define HMX_MODE46_WHI_OFFSET HMX_D8_WHI_OFFSET
#define HMX_MODE46_WLO_OFFSET HMX_D8_WLO_OFFSET
#define HMX_MODE46_OUT_OFFSET HMX_D8_OUT_OFFSET
#define HMX_MODE46_BIAS0_OFFSET HMX_D8_BIAS0_OFFSET
#define HMX_MODE46_BIAS1_OFFSET HMX_D8_BIAS1_OFFSET
#define HMX_MODE46_GUARD_OFFSET HMX_D8_GUARD_OFFSET
#define HMX_MODE46_GUARD_BYTES HMX_D8_GUARD_BYTES
#define HMX_MODE46_GUARD_VECTORS HMX_D8_GUARD_VECTORS
#define HMX_MODE46_VECTOR_BYTES HMX_D8_VECTOR_BYTES
#define HMX_MODE46_TILE_VECTORS HMX_D8_TILE_VECTORS
#define HMX_MODE46_VTCM_BYTES HMX_D8_VTCM_BYTES

/* Bias payloads: scale 0x5400 (+64) for both sets, shape 0 (x) for set 0 and
 * shape 4 (-x) for set 1, output bias 0. Same encoding mode 43/44/45 use. */
#define HMX_MODE46_SCALE64_BITS 0x5400
#define HMX_MODE46_SHAPE0 0
#define HMX_MODE46_SHAPE4 4

/* Convert selector: Rs = 0x0001 (bias set 0, retain ACC, no feedback). */
#define HMX_MODE46_RS_CVT1 HMX_D8_RS_CVT1

/* Fill pattern: one 128 B vector of 0xA5A5, 16 vectors per 2 KiB tile. */
#define HMX_MODE46_FILL_WORD HMX_D8_FILL_WORD

/* Register encodings, identical to mode 43/44/45 so the packets are the same
 * shape the earlier modes already exercised on this device. */
#define HMX_MODE46_SPATIAL_MASK (14u << 7)
#define HMX_MODE46_ACT_RT (HMX_MODE46_SPATIAL_MASK | (31u << 2))
#define HMX_MODE46_WGT_RT ((uint32_t)(2048 - 1))
#define HMX_MODE46_WR_RT HMX_MODE46_SPATIAL_MASK

/* The four registered operand FNVs (contract section "inputs"). The DSP
 * recomputes them from the tiles it actually packed and echoes them; a
 * mismatch is HMX_D8_FAIL_FNV. */
#define HMX_MODE46_FNV_AHI 0xC6231266u
#define HMX_MODE46_FNV_ALO 0x3B2D22D2u
#define HMX_MODE46_FNV_WHI 0x8B88609Fu
#define HMX_MODE46_FNV_WLO 0xA66AF695u

/* Guard status packing helpers (contract section "field masks"). */
static inline unsigned int hmx_mode46_guard_status(int status, int first_bad,
                                                   int init_method,
                                                   int check_method) {
  return ((unsigned int)(status & HMX_D8_GUARD_STATUS_BITS)) |
         ((unsigned int)(first_bad & HMX_D8_GUARD_FIRST_BAD_BITS)
          << HMX_D8_GUARD_FIRST_BAD_SHIFT) |
         ((unsigned int)(init_method & HMX_D8_GUARD_INIT_BITS)
          << HMX_D8_GUARD_INIT_SHIFT) |
         ((unsigned int)(check_method & HMX_D8_GUARD_CHECK_BITS)
          << HMX_D8_GUARD_CHECK_SHIFT);
}

/* Fill echo packing: mask | word << 8 | form << 24. */
static inline unsigned int hmx_mode46_fill_echo(int mask) {
  return ((unsigned int)(mask & HMX_D8_FILL_MASK_BITS)) |
         ((unsigned int)(HMX_D8_FILL_WORD & HMX_D8_FILL_WORD_BITS)
          << HMX_D8_FILL_WORD_SHIFT) |
         ((unsigned int)(HMX_D8_FILL_FORM_16X128 & HMX_D8_FILL_FORM_BITS)
          << HMX_D8_FILL_FORM_SHIFT);
}

/* Address echo: out | Wlo << 7 | Ahi << 14, in 128 B units. The Wlo field is
 * non-zero only for the conditions that fill Wlo (B and C). */
static inline unsigned int hmx_mode46_address_echo(int condition) {
  const unsigned int wlo = (condition == 1 || condition == 2)
                               ? (unsigned int)HMX_D8_ADDR_WLO : 0u;
  return ((unsigned int)HMX_D8_ADDR_OUT) |
         (wlo << HMX_D8_ADDR_WLO_SHIFT) |
         ((unsigned int)HMX_D8_ADDR_AHI << HMX_D8_ADDR_AHI_SHIFT);
}

/* Selector echo: condition | q << 8. */
static inline unsigned int hmx_mode46_selector_echo(int condition, int q) {
  return ((unsigned int)(condition & HMX_D8_SELECTOR_COND_BITS)) |
         ((unsigned int)(q & HMX_D8_SELECTOR_Q_BITS)
          << HMX_D8_SELECTOR_Q_SHIFT);
}

/* Readback echo: trace | done << 2 | hvx << 3 | slots << 11. */
static inline unsigned int hmx_mode46_readback_echo(int hvx_errors) {
  return ((unsigned int)HMX_D8_TRACE_NOT_PROVIDED) |
         (1u << HMX_D8_READBACK_DONE_SHIFT) |
         (((unsigned int)hvx_errors & HMX_D8_READBACK_HVX_BITS)
          << HMX_D8_READBACK_HVX_SHIFT) |
         (((unsigned int)HMX_D8_RESULT_SLOTS & HMX_D8_READBACK_SLOTS_BITS)
          << HMX_D8_READBACK_SLOTS_SHIFT);
}

#endif /* HEXATRAIN_HMX_MODE46_H */
