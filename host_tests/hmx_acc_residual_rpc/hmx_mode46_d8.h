// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// D8 Mode 46 handler: the Candidate B minimal device experiment.
//
// This file is included by probe_dsp.c after the mode 43/44/45 handlers, so it
// can use init_bias20, e3c1_* and the fp16 helpers. It is not a standalone
// translation unit.
//
// Registered order (docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md
// section 3.1, machine-checked by --d8-sequence-rev3):
//
//   PACK AHI, PACK ALO, PACK WHI, PACK WLO, BIAS 0, BIAS 1,
//   GUARD INIT 12, GUARD PRE, MXCLRACC,
//   MMA 1 AHI WHI, MMA 2 AHI WLO, MMA 3 ALO WHI,
//   FILL OUT, FILL WLO B C, FILL AHI B D, CONVERT 1, STORE OUT, GUARD POST
//
// The critical interval is MMA #1 .. FILL AHI. Nothing in it may synchronise:
// no syncht, no barrier, no DMA sync, no dummy cvt wait, no checksum loop and
// no scalar read of +0/+6144/+8192. The operand FNVs are therefore taken
// BEFORE MMA #1, and the guard post hash is taken AFTER Store #1.
//
// Ownership: this handler writes metadata[9..31] only. metadata[0..8] belong to
// run(), and metadata[30] (cleanup) is written by run() for mode 46.

/* Copy one 2 KiB tile into VTCM with 16 x 128 B HVX stores.
 *
 * The registration lists PACK as "16 x 128 B" per operand, and the
 * disassembly audit checks the real object for exactly that form. Writing the
 * cells straight into VTCM lowers to scalar memd/memw stores, which is a
 * different experiment, so the tiles are built in a local buffer and copied
 * with vector stores. The loop count is a literal so the audit can expand it
 * statically. */
static void d8_pack_tile(uint8_t *dst_ptr, const uint16_t *src) {
  const uintptr_t dst = (uintptr_t)dst_ptr;
  int v;
  for (v = 0; v < HMX_MODE46_TILE_VECTORS; ++v) {
    const HVX_Vector chunk =
        *(const HVX_Vector *)((const uint8_t *)src +
                              (size_t)(v * HMX_MODE46_VECTOR_BYTES));
    asm volatile("vmem(%0) = %1\n"
                 : : "r"(dst + (uintptr_t)(v * HMX_MODE46_VECTOR_BYTES)),
                     "v"(chunk)
                 : "memory");
  }
}

/* ---- D8 operand fixture -------------------------------------------------
 *
 * The registered operands are the ones the CPU model packs from the mode 42
 * fixture (seed 0xE3C0117E5EEDF00D, the anchor rows and lanes, then the
 * float-arithmetic split). The DSP must reproduce exactly that, because the
 * four FNVs and the four reference digests are registered against it.
 *
 * build_d8_operands() fills the four 2 KiB tiles in the PRM layout the MMA
 * reads: activation cell (spatial, in_ch) at ((spatial>>1)*64 + in_ch*2 +
 * (spatial&1)), weight cell (out_ch, in_ch) at ((in_ch>>1)*64 + out_ch*2 +
 * (in_ch&1)). Same mapping as mode 43.
 */
/* The four packs must stay four distinct calls.
 *
 * At -O0 the object calls d8_pack_tile four times with +0, +0x800, +0x1000 and
 * +0x1800, which is the registered form. At -O2 the compiler inlines this
 * function into the handler and folds the third pack onto the out_pos slot: the
 * emitted object then writes whi_local to +8192 and never writes +4096, which
 * the disassembly audit catches as pack_4096_count 0/16. Keeping the function
 * out of line preserves the four destinations. */
__attribute__((noinline))
static void build_d8_operands(uint8_t *base, uint16_t *ahi, uint16_t *alo,
                              uint16_t *whi, uint16_t *wlo) {
  /* The tiles are built in local buffers and then copied into VTCM with the
   * registered 16 x 128 B vector stores; writing the cells straight into VTCM
   * would lower to scalar stores, which is a different experiment. */
  static uint16_t ahi_local[E3C1_T8_CELLS];
  static uint16_t alo_local[E3C1_T8_CELLS];
  static uint16_t whi_local[E3C1_T8_CELLS];
  static uint16_t wlo_local[E3C1_T8_CELLS];
  int idx;
  int s, o, i;

  /* ---- t8_build port: statement order verbatim (mode 42/43) ---- */
  e3c1_rng = 0xE3C0117E5EEDF00Dull;
  for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
    double u = e3c1_rng_unit();
    double v = u * 4.0;
    double val = v - 2.0;
    e3c1_w[idx] = val;
    u = e3c1_rng_unit();
    v = u * 4.0;
    val = v - 2.0;
    e3c1_x[idx] = val;
  }

  /* Special W rows: out-ch 0..3 case-class anchors, 28..31 s3 targets. */
  for (o = 0; o < E3C1_T8; ++o) {
    if (o <= 3 || o >= 28) {
      for (i = 0; i < E3C1_T8; ++i) e3c1_w[(long)o * E3C1_T8 + i] = 0.0;
    }
  }
  e3c1_w[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11;
  e3c1_w[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_w[0 * E3C1_T8 + 2] = 0x1p-13;
  e3c1_w[1 * E3C1_T8 + 0] = 1.0 - 0x1p-12;
  e3c1_w[2 * E3C1_T8 + 0] = 0x1p-9;
  e3c1_w[3 * E3C1_T8 + 0] = 1.0;
  e3c1_w[3 * E3C1_T8 + 1] = 1.0 + 0x1p-12;
  e3c1_w[28 * E3C1_T8 + 0] = 0x1p-10;
  e3c1_w[29 * E3C1_T8 + 0] = 0x1p-10 - 0x1p-21;
  e3c1_w[30 * E3C1_T8 + 0] = 3.0 * 0x1p-12;
  e3c1_w[31 * E3C1_T8 + 0] = 0x1p-13;

  /* Special A lanes 0..3; lanes 4..31 stay seeded realistic. */
  for (s = 0; s <= 3; ++s)
    for (i = 0; i < E3C1_T8; ++i) e3c1_x[(long)s * E3C1_T8 + i] = 0.0;
  e3c1_x[0 * E3C1_T8 + 0] = 1.0 + 0x1p-11;
  e3c1_x[0 * E3C1_T8 + 1] = 0x1p-11;
  e3c1_x[0 * E3C1_T8 + 2] = 0x1p-13;
  for (i = 0; i < E3C1_T8; ++i)
    e3c1_x[1 * E3C1_T8 + i] = -e3c1_x[i];
  e3c1_x[2 * E3C1_T8 + 0] = 1.0 - 0x1p-12;
  e3c1_x[2 * E3C1_T8 + 1] = -(1.0 - 0x1p-12);
  e3c1_x[3 * E3C1_T8 + 0] = 0x1p-10;

  /* Pack the four tiles. hi = RN16(value) is the plain operand; lo is the
   * float-arithmetic split residual, exactly e3c1_split_lo. */
  memset(ahi_local, 0, TILE_BYTES);
  memset(alo_local, 0, TILE_BYTES);
  memset(whi_local, 0, TILE_BYTES);
  memset(wlo_local, 0, TILE_BYTES);
  for (idx = 0; idx < E3C1_T8_CELLS; ++idx) {
    const int row = idx >> 5; /* activation spatial / weight output channel */
    const int col = idx & 31; /* input channel */
    const uint16_t ah = double_to_fp16(e3c1_x[idx]);
    const uint16_t wh = double_to_fp16(e3c1_w[idx]);
    const uint16_t al = e3c1_split_lo(e3c1_x[idx], ah);
    const uint16_t wl = e3c1_split_lo(e3c1_w[idx], wh);
    ahi_local[((row >> 1) * 64) + col * 2 + (row & 1)] = ah;
    alo_local[((row >> 1) * 64) + col * 2 + (row & 1)] = al;
    whi_local[((col >> 1) * 64) + row * 2 + (col & 1)] = wh;
    wlo_local[((col >> 1) * 64) + row * 2 + (col & 1)] = wl;
  }

  /* PACK AHI, PACK ALO, PACK WHI, PACK WLO: 16 x 128 B each, in that order.
   *
   * The third destination is derived from base at the call site rather than
   * from a pointer variable. The compiler otherwise copied the out_pos pointer
   * into the third pack's slot: the emitted object wrote whi_local to +8192 and
   * never wrote +4096, which the audit caught as pack_4096_count 0/16. */
  d8_pack_tile(base + 0u, ahi_local);
  d8_pack_tile(base + 2048u, alo_local);
  d8_pack_tile(base + 4096u, whi_local);
  d8_pack_tile(base + 6144u, wlo_local);
}

/* FNV-1a 32 over a packed tile, byte order as stored (little endian u16). */
static uint32_t d8_tile_fnv(const uint16_t *tile) {
  return mode45_fnv1a((const uint8_t *)tile, TILE_BYTES);
}

/* Fill one 2 KiB tile with 0xA5A5 using 16 x 128 B HVX stores.
 *
 * The registered form is HMX_D8_FILL_FORM_16X128, and the disassembly audit
 * checks that the real object contains 16 vector stores rather than a scalar
 * loop or a memset. The volatile asm keeps the compiler from turning this into
 * a library call. */
static void d8_fill_tile(uint16_t *tile) {
  const uintptr_t base = (uintptr_t)tile;
  const HVX_Vector pattern = Q6_V_vsplat_R(HMX_MODE46_FILL_WORD);
  int v;
  for (v = 0; v < HMX_MODE46_TILE_VECTORS; ++v) {
    asm volatile("vmem(%0) = %1\n"
                 : : "r"(base + (uintptr_t)(v * HMX_MODE46_VECTOR_BYTES)),
                     "v"(pattern)
                 : "memory");
  }
}

/* Initialise the 1,536 B guard with 12 x 128 B zero stores. */
static void d8_guard_init(uint8_t *guard) {
  const uintptr_t base = (uintptr_t)guard;
  const HVX_Vector zero = Q6_V_vzero();
  int v;
  for (v = 0; v < HMX_MODE46_GUARD_VECTORS; ++v) {
    asm volatile("vmem(%0) = %1\n"
                 : : "r"(base + (uintptr_t)(v * HMX_MODE46_VECTOR_BYTES)),
                     "v"(zero)
                 : "memory");
  }
}

/* Byte-wise guard check. Returns the first differing byte offset, or
 * HMX_D8_GUARD_FIRST_BAD_NONE. Reads only the guard, never an operand or
 * out_pos, so it cannot serialise the reuse. */
static int d8_guard_first_bad(const uint8_t *guard) {
  int i;
  for (i = 0; i < HMX_MODE46_GUARD_BYTES; ++i) {
    if (guard[i] != 0) return i;
  }
  return HMX_D8_GUARD_FIRST_BAD_NONE;
}

/* ---- the handler -------------------------------------------------------
 *
 * Returns 0 on success and on every contract failure that happens after
 * execution started (the reason goes to metadata[HMX_D8_META_FAILURE]).
 * Returns HMX_ERR_HVX only if the HVX lock fails before metadata[9] is
 * written, which is the S1 pre-execution case.
 */
static int run_mode46_d8_one_trial(uint8_t *base, int *results, int *metadata,
                                   int condition) {
  uint16_t *ahi = (uint16_t *)(base + HMX_MODE46_AHI_OFFSET);
  uint16_t *alo = (uint16_t *)(base + HMX_MODE46_ALO_OFFSET);
  uint16_t *whi = (uint16_t *)(base + HMX_MODE46_WHI_OFFSET);
  uint16_t *wlo = (uint16_t *)(base + HMX_MODE46_WLO_OFFSET);
  uint16_t *out = (uint16_t *)(base + HMX_MODE46_OUT_OFFSET);
  uint32_t *bias0 = (uint32_t *)(base + HMX_MODE46_BIAS0_OFFSET);
  uint32_t *bias1 = (uint32_t *)(base + HMX_MODE46_BIAS1_OFFSET);
  uint8_t *guard = (uint8_t *)(base + HMX_MODE46_GUARD_OFFSET);

  const uintptr_t ahi_reg = (uintptr_t)ahi;
  const uintptr_t alo_reg = (uintptr_t)alo;
  const uintptr_t whi_reg = (uintptr_t)whi;
  const uintptr_t wlo_reg = (uintptr_t)wlo;
  const uintptr_t out_reg = (uintptr_t)out;
  const uintptr_t bias0_reg = (uintptr_t)bias0;
  const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;

  const int fill_mask = hmx_d8_fill_mask_for_condition(condition);
  int hvx_lock;
  int i;
  uint32_t fnv_ahi, fnv_alo, fnv_whi, fnv_wlo;
  uint32_t guard_pre, guard_post;
  int first_bad;

  /* ---- prologue: operands, bias, guard (outside the critical interval) ----
   *
   * The HVX context is taken FIRST. build_d8_operands and d8_guard_init issue
   * vmem stores, and vmem is an HVX instruction: issued without the lock they
   * do not land. The device run showed exactly that - the guard hashed to the
   * zero-guard value because the guard init writes zeros anyway, and the four
   * operand tiles hashed to their untouched contents, so the handler stopped at
   * its own FNV gate before the critical interval.
   *
   * The lock is not an entry of the registered issued order (section 3.1); it
   * is a precondition for issuing any of them, so taking it first changes no
   * registered target. */
  hvx_lock = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (hvx_lock) {
    return HMX_ERR_HVX;
  }

  build_d8_operands(base, ahi, alo, whi, wlo);

  init_bias20(bias0, HMX_MODE46_SCALE64_BITS, 0, 0, HMX_MODE46_SHAPE0);
  init_bias20(bias1, HMX_MODE46_SCALE64_BITS, 0, 0, HMX_MODE46_SHAPE4);

  /* Load the two bias sets. The registration lists BIAS 0 and BIAS 1 as
   * issued operations before mxclracc; writing the payload without loading it
   * would leave the convert with no registered bias state. The low two address
   * bits select the set on V81, so set 1 is addressed with |1. */
  asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
  asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");

  d8_guard_init(guard);
  guard_pre = mode45_fnv1a(guard, HMX_MODE46_GUARD_BYTES);

  /* The operand FNVs must be taken BEFORE MMA #1: hashing 8 KiB inside the
   * critical interval would insert reads the registered experiment does not
   * have. */
  fnv_ahi = d8_tile_fnv(ahi);
  fnv_alo = d8_tile_fnv(alo);
  fnv_whi = d8_tile_fnv(whi);
  fnv_wlo = d8_tile_fnv(wlo);

  /* Execution starts here: metadata[9] is the first D8-owned word. */
  metadata[HMX_D8_META_BUILD_ID] = HMX_PROBE_BUILD_ID;
  metadata[HMX_D8_META_MXCLRACC] = 0;
  metadata[HMX_D8_META_MMA] = 0;
  metadata[HMX_D8_META_CONVERT] = 0;
  metadata[HMX_D8_META_STORE] = 0;
  metadata[HMX_D8_META_PACK] = 0;
  metadata[HMX_D8_META_BIAS] = 0;
  metadata[HMX_D8_META_FILL] = 0;
  metadata[HMX_D8_META_ADDRESS] = 0;
  metadata[HMX_D8_META_SELECTOR] = (int)hmx_mode46_selector_echo(condition,
                                                                 condition + HMX_D8_Q_FIRST);
  metadata[HMX_D8_META_FNV_AHI] = (int)fnv_ahi;
  metadata[HMX_D8_META_FNV_ALO] = (int)fnv_alo;
  metadata[HMX_D8_META_FNV_WHI] = (int)fnv_whi;
  metadata[HMX_D8_META_FNV_WLO] = (int)fnv_wlo;
  metadata[HMX_D8_META_GUARD_PRE] = (int)guard_pre;
  metadata[HMX_D8_META_GUARD_POST] = 0;
  metadata[HMX_D8_META_GUARD_STATUS] = 0;
  metadata[HMX_D8_META_FAILURE] = HMX_D8_FAIL_NONE;
  metadata[HMX_D8_META_SCHEMA] = (int)HMX_D8_SCHEMA;
  metadata[HMX_D8_META_READBACK] = 0;
  metadata[HMX_D8_META_RESERVED_29] = 0;
  metadata[HMX_D8_META_RESERVED_31] = 0;

  /* The four packs are the four tile writes above; the two bias loads are the
   * two init_bias20 calls. Report them as issued. */
  metadata[HMX_D8_META_PACK] = 4;
  metadata[HMX_D8_META_BIAS] = 2;

  /* Operand FNV gate: a mismatch means the tiles are not the registered
   * fixture, so the numeric result would be meaningless. */
  if (fnv_ahi != HMX_MODE46_FNV_AHI || fnv_alo != HMX_MODE46_FNV_ALO ||
      fnv_whi != HMX_MODE46_FNV_WHI || fnv_wlo != HMX_MODE46_FNV_WLO) {
    metadata[HMX_D8_META_FAILURE] = HMX_D8_FAIL_FNV;
    metadata[HMX_D8_META_GUARD_STATUS] = (int)hmx_mode46_guard_status(
        HMX_D8_GUARD_STATUS_OK, HMX_D8_GUARD_FIRST_BAD_NONE,
        HMX_D8_GUARD_INIT_HVX12, HMX_D8_GUARD_CHECK_BYTE);
    metadata[HMX_D8_META_GUARD_POST] = (int)guard_pre;
    metadata[HMX_D8_META_READBACK] = (int)hmx_mode46_readback_echo(0);
    qurt_hvx_unlock();
    return 0;
  }

  /* ===================== critical interval begins ===================== */
  /* No syncht, barrier, DMA sync, dummy cvt wait, checksum loop or scalar
   * read of +0/+6144/+8192 from here until FILL AHI. */

  asm volatile("mxclracc.hf\n" ::: "memory");
  metadata[HMX_D8_META_MXCLRACC] = 1;

  /* MMA #1: Ahi x Whi */
  asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3) }\n"
      : : "r"(ahi_reg), "r"(HMX_MODE46_ACT_RT),
          "r"(whi_reg), "r"(HMX_MODE46_WGT_RT)
      : "memory");
  metadata[HMX_D8_META_MMA] = 1;

  /* MMA #2: Ahi x Wlo */
  asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3) }\n"
      : : "r"(ahi_reg), "r"(HMX_MODE46_ACT_RT),
          "r"(wlo_reg), "r"(HMX_MODE46_WGT_RT)
      : "memory");
  metadata[HMX_D8_META_MMA] = 2;

  /* MMA #3: Alo x Whi */
  asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3) }\n"
      : : "r"(alo_reg), "r"(HMX_MODE46_ACT_RT),
          "r"(whi_reg), "r"(HMX_MODE46_WGT_RT)
      : "memory");
  metadata[HMX_D8_META_MMA] = 3;

  /* FILL OUT is mandatory for every condition. */
  d8_fill_tile(out);

  /* Condition fills: B and C fill Wlo, B and D fill Ahi, A fills neither. */
  if (fill_mask & HMX_D8_FILL_WLO) {
    d8_fill_tile(wlo);
  }
  if (fill_mask & HMX_D8_FILL_AHI) {
    d8_fill_tile(ahi);
  }

  /* Convert #1: bias set 0, retain ACC, no feedback. */
  asm volatile("cvt.hf = acc(%0)\n" : : "r"(HMX_MODE46_RS_CVT1) : "memory");
  metadata[HMX_D8_META_CONVERT] = 1;

  /* Store #1 into out_pos. */
  asm volatile("mxmem(%0, %1) = cvt\n"
               : : "r"(out_reg), "r"(HMX_MODE46_WR_RT) : "memory");
  metadata[HMX_D8_META_STORE] = 1;

  /* ====================== critical interval ends ====================== */

  /* Guard post hash and byte check, after Store #1 and before the readback. */
  guard_post = mode45_fnv1a(guard, HMX_MODE46_GUARD_BYTES);
  first_bad = d8_guard_first_bad(guard);

  metadata[HMX_D8_META_GUARD_POST] = (int)guard_post;
  metadata[HMX_D8_META_GUARD_STATUS] = (int)hmx_mode46_guard_status(
      (guard_pre == guard_post && first_bad == HMX_D8_GUARD_FIRST_BAD_NONE)
          ? HMX_D8_GUARD_STATUS_OK
          : HMX_D8_GUARD_STATUS_POST_CHANGED,
      first_bad, HMX_D8_GUARD_INIT_HVX12, HMX_D8_GUARD_CHECK_BYTE);

  metadata[HMX_D8_META_FILL] = (int)hmx_mode46_fill_echo(fill_mask);
  metadata[HMX_D8_META_ADDRESS] = (int)hmx_mode46_address_echo(condition);

  /* Readback: 1,024 slots, each the zero-extended u16 of out_pos. */
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) {
    results[i] = (int)out[i];
  }
  metadata[HMX_D8_META_READBACK] = (int)hmx_mode46_readback_echo(0);

  /* A dirty guard is a contract failure even when the numbers look right. */
  if (guard_pre != guard_post || first_bad != HMX_D8_GUARD_FIRST_BAD_NONE) {
    metadata[HMX_D8_META_FAILURE] = HMX_D8_FAIL_GUARD_POST;
  }

  qurt_hvx_unlock();
  return 0;
}
