/* D8 minimal experiment — CPU model and classifier (CPU only).
 *
 * D8 is the last device-only synchronization question of Candidate B: MMA #2
 * issues reads of Ahi (+0) and Wlo (+6144), and the output phase later writes
 * those same regions (out_recon -> +0, out_neg -> +6144). The CPU program order
 * is correct by construction; the question is whether the hardware completes
 * the operand reads before those writes land.
 *
 * This program does not answer that question. It does four things:
 *   1. computes the four numerical references a device run could produce
 *      (safe, Wlo-corrupt, Ahi-corrupt, both-corrupt) from the registered
 *      operand fixture, using the shipped model's arithmetic;
 *   2. checks the safe reference against the existing B3 subcase output
 *      (g_m_b3 style) so the minimal sequence is provably the same arithmetic;
 *   3. runs the D8 access-trace and issued-order validators and the
 *      fault/control families (01-16 numeric/access; 17-38 revision-3
 *      order/guard), including the metadata/q/failure contract model;
 *   4. exposes a classifier replay mode for the independent Python audit and
 *      a revision-3 issued-order signature mode.
 *
 * CPU only: no device, no DSP, no SDK. The references are outcomes of a
 * hypothetical read/write race, not a proposed program order for the device.
 */
#define main hmx_b3_r2_full_tile_sim_original_main
#include "hmx_b3_r2_full_tile_sim.c"
#undef main

#include "hmx_d8_contract.h"

#define D8_OUT_OFFSET HMX_B3_R2_PR_POS_OFFSET /* +8192, not an operand region */
#define D8_COND_A 0 /* no operand reuse (control) */
#define D8_COND_B 1 /* Candidate B: both Wlo and Ahi reused */
#define D8_COND_C 2 /* Wlo only */
#define D8_COND_D 3 /* Ahi only */
#define D8_TRACE_NOT_PROVIDED 0
#define D8_TRACE_VALID 1
#define D8_TRACE_INVALID 2

/* Registered fill-mask bits (evidence contract sections 3.2 and 3.3):
 *   bit0 = out_pos (+8192), bit1 = Wlo (+6144), bit2 = Ahi (+0).
 * Condition A fills only the fresh out_pos; B reuses both operands; C only
 * Wlo; D only Ahi. These four values are the contract, not a derived
 * convenience: revision 1 listed C and D swapped and made B ambiguous. */
#define D8_MASK_OUT 1
#define D8_MASK_WLO 2
#define D8_MASK_AHI 4

static int d8_condition_mask(int cond) {
  switch (cond) {
    case D8_COND_A: return D8_MASK_OUT;
    case D8_COND_B: return D8_MASK_OUT | D8_MASK_WLO | D8_MASK_AHI;
    case D8_COND_C: return D8_MASK_OUT | D8_MASK_WLO;
    case D8_COND_D: return D8_MASK_OUT | D8_MASK_AHI;
    default: return 0;
  }
}

typedef enum {
  D8_CLASS_CONTRACT_FAILED = 0,
  D8_CLASS_CONTROL_SAFE,
  D8_CLASS_CONTROL_ANOMALY,
  D8_CLASS_NO_VIOLATION,
  D8_CLASS_VIOLATION_WLO,
  D8_CLASS_VIOLATION_AHI,
  D8_CLASS_VIOLATION_BOTH,
  D8_CLASS_INCONCLUSIVE_NUMERIC,
  D8_CLASS_INCONCLUSIVE_TRACE
} d8_class_t;

static const char *d8_class_name(d8_class_t c) {
  switch (c) {
    case D8_CLASS_CONTRACT_FAILED: return "D8_CONTRACT_FAILED";
    case D8_CLASS_CONTROL_SAFE: return "D8_CONTROL_SAFE";
    case D8_CLASS_CONTROL_ANOMALY: return "D8_CONTROL_ANOMALY";
    case D8_CLASS_NO_VIOLATION: return "D8_NO_VIOLATION_OBSERVED";
    case D8_CLASS_VIOLATION_WLO: return "D8_VIOLATION_WLO_SUSPECT";
    case D8_CLASS_VIOLATION_AHI: return "D8_VIOLATION_AHI_SUSPECT";
    case D8_CLASS_VIOLATION_BOTH: return "D8_VIOLATION_BOTH_SUSPECT";
    case D8_CLASS_INCONCLUSIVE_NUMERIC: return "D8_INCONCLUSIVE_NUMERIC";
    case D8_CLASS_INCONCLUSIVE_TRACE: return "D8_INCONCLUSIVE_TRACE";
    default: return "D8_UNCLASSIFIED";
  }
}

static const char *d8_condition_name(int c) {
  switch (c) {
    case D8_COND_A: return "A_no_reuse";
    case D8_COND_B: return "B_candidate_reuse";
    case D8_COND_C: return "C_wlo_only";
    case D8_COND_D: return "D_ahi_only";
    default: return "UNKNOWN";
  }
}

typedef struct {
  uint16_t safe[T8_CELLS];
  uint16_t wlo[T8_CELLS];
  uint16_t ahi[T8_CELLS];
  uint16_t both[T8_CELLS];
} d8_refs_t;

static d8_refs_t g_d8_refs;
static int g_d8_checks;
static int g_d8_fail;

static void d8_check(int cond, const char *tag) {
  g_d8_checks++;
  if (!cond) {
    g_d8_fail++;
    printf("FAIL %s\n", tag);
  }
}

/* ---------------- program-order record (revision 3) ---------------- */

/* The shared model's access trace records memory accesses. It cannot show
 * mxclracc (no memory access), and it fuses Convert + Store into one HMX store
 * event. Revision 3 therefore records the *issued operation list* next to the
 * access trace and validates the two independently:
 *
 *   - the operation list fixes the order and the operands of mxclracc, the
 *     three MMA packets, the bias writes, Convert and Store;
 *   - the access trace fixes the actual byte ranges that were read/written.
 *
 * A fault that only edits one record is caught by the cross-check; a fault
 * that reorders real operations is caught by the list; a fault that skips an
 * operation is caught by the list and by the execution counters. This is
 * deliberately independent of the numeric result and of any fault flag.
 */

typedef enum {
  D8_OP_PACK = 1,
  D8_OP_BIAS,
  D8_OP_GUARD_INIT,
  D8_OP_GUARD_PRE,
  D8_OP_MXCLRACC,
  D8_OP_MMA,
  D8_OP_FILL,
  D8_OP_CONVERT,
  D8_OP_STORE,
  D8_OP_GUARD_POST
} d8_op_kind_t;

typedef struct {
  int kind;
  int a;
  int b;
  int c;
} d8_op_t;

#define D8_OPLOG_CAP 64

typedef struct {
  d8_op_t op[D8_OPLOG_CAP];
  int count;
  int overflow;
  int enabled;
} d8_oplog_t;

static d8_oplog_t g_d8_log;
static int g_d8_guard_pre_fnv;
static int g_d8_guard_pre_ok;

static void d8_log_begin(void) {
  memset(&g_d8_log, 0, sizeof(g_d8_log));
  g_d8_log.enabled = 1;
}

static void d8_log_add(int kind, int a, int b, int c) {
  int n;
  if (!g_d8_log.enabled) return;
  if (g_d8_log.count >= D8_OPLOG_CAP) {
    g_d8_log.overflow = 1;
    return;
  }
  n = g_d8_log.count++;
  g_d8_log.op[n].kind = kind;
  g_d8_log.op[n].a = a;
  g_d8_log.op[n].b = b;
  g_d8_log.op[n].c = c;
}

static const char *d8_op_name(int kind) {
  switch (kind) {
    case D8_OP_PACK: return "PACK";
    case D8_OP_BIAS: return "BIAS";
    case D8_OP_GUARD_INIT: return "GUARD_INIT";
    case D8_OP_GUARD_PRE: return "GUARD_PRE";
    case D8_OP_MXCLRACC: return "MXCLRACC";
    case D8_OP_MMA: return "MMA";
    case D8_OP_FILL: return "FILL";
    case D8_OP_CONVERT: return "CONVERT";
    case D8_OP_STORE: return "STORE";
    case D8_OP_GUARD_POST: return "GUARD_POST";
    default: return "?";
  }
}

/* ---------------- sequence primitives ---------------- */

/* One bias set at an explicit offset, logged. The registered pair is
 * (set 0 -> +10240, set 1 -> +10496); the offset parameter exists only so a
 * fault can write the wrong set at a registered offset. */
static void d8_bias_write(hmx_full_tile_state_t *st, int set, int off) {
  uint8_t *b = st->vtcm.raw + off;
  pr_event(st, PR_BIAS_WRITE, off, 256, 1, set);
  for (int o = 0; o < T8; ++o) {
    uint32_t lo = bias20_lo_cell(HMX_B3_R2_SCALE_BITS, HMX_B3_R2_OUT_BIAS);
    uint32_t hi = bias20_hi_cell(set ? HMX_B3_R2_SHAPE4 : HMX_B3_R2_SHAPE0,
                                 0, 0);
    memcpy(b + o * 4, &lo, 4);
    memcpy(b + 128 + o * 4, &hi, 4);
  }
  d8_log_add(D8_OP_BIAS, set, off, 1);
}

static void d8_bias_pair(hmx_full_tile_state_t *st) {
  d8_bias_write(st, 0, HMX_B3_R2_PR_BIAS0_OFFSET);
  d8_bias_write(st, 1, HMX_B3_R2_PR_BIAS1_OFFSET);
}

static void d8_pack_tiles(hmx_full_tile_state_t *st, const b3_r2_ops_t *ops) {
  pr_pack(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_OP_AHI, ops->ahi, 0);
  d8_log_add(D8_OP_PACK, HMX_B3_R2_OP_AHI, HMX_B3_R2_PR_AHI_OFFSET, 0);
  pr_pack(st, HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_OP_ALO, ops->alo, 0);
  d8_log_add(D8_OP_PACK, HMX_B3_R2_OP_ALO, HMX_B3_R2_PR_ALO_OFFSET, 0);
  pr_pack(st, HMX_B3_R2_PR_WHI_OFFSET, HMX_B3_R2_OP_WHI, ops->whi, 1);
  d8_log_add(D8_OP_PACK, HMX_B3_R2_OP_WHI, HMX_B3_R2_PR_WHI_OFFSET, 0);
  pr_pack(st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_OP_WLO, ops->wlo, 1);
  d8_log_add(D8_OP_PACK, HMX_B3_R2_OP_WLO, HMX_B3_R2_PR_WLO_OFFSET, 0);
}

static void d8_pack_operands(hmx_full_tile_state_t *st,
                             const b3_r2_ops_t *ops) {
  d8_pack_tiles(st, ops);
  d8_bias_pair(st);
}

static void d8_mma_packet(hmx_full_tile_state_t *st, int act_id, int wgt_id,
                          int act_off, int wgt_off) {
  st->act_id = act_id;
  st->wgt_id = wgt_id;
  exec_mma_packet_at(st, act_off, wgt_off);
  d8_log_add(D8_OP_MMA, act_off, wgt_off, 0);
}

static void d8_mxclracc(hmx_full_tile_state_t *st) {
  exec_mxclracc(st);
  d8_log_add(D8_OP_MXCLRACC, 0, 0, 0);
}

static void d8_mma_sequence(hmx_full_tile_state_t *st) {
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
}

static void d8_fill(hmx_full_tile_state_t *st, int off, int object) {
  pr_fill(st, off, HMX_B3_R2_TILE_BYTES, object);
  d8_log_add(D8_OP_FILL, off, object, HMX_B3_R2_TILE_BYTES);
}

static void d8_convert_store(hmx_full_tile_state_t *st) {
  d8_log_add(D8_OP_CONVERT, (int)HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET, 0);
  exec_convert_and_store_at(st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
  d8_log_add(D8_OP_STORE, D8_OUT_OFFSET, 0, 0);
}

/* ---------------- guard evidence (revision 3) ---------------- */

/* The guard is +10752..+12288 (1,536 B, 12 x 128 B). It is initialised to
 * zero before mxclracc, hashed immediately after the init, hashed again after
 * Store #1, and the two digests plus the first differing byte are echoed.
 * Both the init and the checks are outside the MMA #3 -> fill -> Convert
 * interval, so they cannot shorten the registered reuse distance. */
static void d8_guard_init(hmx_full_tile_state_t *st) {
  pr_event(st, PR_INIT_WRITE, HMX_D8_GUARD_OFFSET, HMX_D8_GUARD_BYTES, 1, 7);
  memset(st->vtcm.raw + HMX_D8_GUARD_OFFSET, 0, HMX_D8_GUARD_BYTES);
  d8_log_add(D8_OP_GUARD_INIT, HMX_D8_GUARD_VECTORS, HMX_D8_GUARD_BYTES,
             HMX_D8_GUARD_INIT_HVX12);
}

static void d8_guard_pre(hmx_full_tile_state_t *st) {
  unsigned int h = hmx_d8_fnv1a32(st->vtcm.raw + HMX_D8_GUARD_OFFSET,
                                  HMX_D8_GUARD_BYTES);
  g_d8_guard_pre_fnv = (int)h;
  g_d8_guard_pre_ok = (h == hmx_d8_guard_zero_fnv());
  d8_log_add(D8_OP_GUARD_PRE, (int)h, g_d8_guard_pre_ok, 0);
}

static void d8_guard_post(hmx_full_tile_state_t *st) {
  unsigned int h = hmx_d8_fnv1a32(st->vtcm.raw + HMX_D8_GUARD_OFFSET,
                                  HMX_D8_GUARD_BYTES);
  int first_bad = HMX_D8_GUARD_FIRST_BAD_NONE;
  int status = HMX_D8_GUARD_STATUS_OK;
  for (int i = 0; i < HMX_D8_GUARD_BYTES; ++i) {
    if (st->vtcm.raw[HMX_D8_GUARD_OFFSET + i] != 0) {
      first_bad = i;
      break;
    }
  }
  if (!g_d8_guard_pre_ok) {
    status = HMX_D8_GUARD_STATUS_PRE_NONZERO;
  } else if (h != (unsigned int)g_d8_guard_pre_fnv ||
             first_bad != HMX_D8_GUARD_FIRST_BAD_NONE) {
    status = HMX_D8_GUARD_STATUS_POST_CHANGED;
  }
  d8_log_add(D8_OP_GUARD_POST, (int)h, status, first_bad);
}

/* Numerical reference for "the read lost the race": the affected operand tile
 * already holds the sentinel when the MMA executes. This is an outcome model,
 * not a program order. */
static void d8_outcome(uint16_t *out, int corrupt_ahi, int corrupt_wlo) {
  static hmx_full_tile_state_t st;
  pr_state_init(&st);
  d8_pack_operands(&st, &g_ops_deploy);
  if (corrupt_ahi) {
    pr_fill(&st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_TILE_BYTES,
            HMX_B3_R2_OP_AHI);
  }
  if (corrupt_wlo) {
    pr_fill(&st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_TILE_BYTES,
            HMX_B3_R2_OP_WLO);
  }
  d8_mma_sequence(&st);
  exec_convert_and_store_at(&st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
  cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
}

/* Registered program order (revision 2): MMAs first, then the output phase.
 * The out_pos sentinel is the first write of that phase, exactly as
 * Candidate B's phase-reused subcase does (pos, then neg/Wlo, then
 * recon/Ahi), followed by the condition's operand fills and Convert #1.
 * Only the fills of the reuse conditions are scheduled; the full Candidate B
 * also stores Convert #2 into Wlo and the recon into Ahi, but those writes
 * are later than these fills, so the fills are the tightest reuse point. */
static void d8_sequence_with_mask(hmx_full_tile_state_t *st, int fill_mask) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mma_sequence(st);
  if (fill_mask & D8_MASK_OUT) {
    d8_fill(st, D8_OUT_OFFSET, 1); /* out_pos sentinel */
  }
  if (fill_mask & D8_MASK_WLO) {
    d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  }
  if (fill_mask & D8_MASK_AHI) {
    d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  }
  d8_convert_store(st);
}

/* Revision-3 candidate: the revision-2 order plus the guard init and the two
 * guard digests. Guard init/hash happen before mxclracc; the post hash happens
 * after Store #1. The critical interval is unchanged. */
static void d8_rev3_sequence(hmx_full_tile_state_t *st, int cond) {
  int fill_mask = d8_condition_mask(cond);
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_guard_init(st);
  d8_guard_pre(st);
  d8_mma_sequence(st);
  if (fill_mask & D8_MASK_OUT) d8_fill(st, D8_OUT_OFFSET, 1);
  if (fill_mask & D8_MASK_WLO) d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  if (fill_mask & D8_MASK_AHI) d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
  d8_guard_post(st);
}

static void d8_program_order(hmx_full_tile_state_t *st, int cond) {
  d8_sequence_with_mask(st, d8_condition_mask(cond));
}

/* Revision 1 sequence, kept only for the CPU comparison recorded in the
 * pre-registration. It is NOT the registered sequence: the out_pos fill runs
 * in the prologue, before mxclracc and the MMAs. Revision 2 rejects it with
 * "fill_before_last_mma_read"; the numeric result is identical because the
 * fill touches no operand region. Different instruction sequence, different
 * experiment, even when the numbers agree. */
static void d8_legacy_rev1_sequence(hmx_full_tile_state_t *st, int cond) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_fill(st, D8_OUT_OFFSET, 1); /* rev1: prologue */
  d8_mma_sequence(st);
  if (cond == D8_COND_B || cond == D8_COND_C) {
    d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  }
  if (cond == D8_COND_B || cond == D8_COND_D) {
    d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  }
  d8_convert_store(st);
}

/* ---------------- D8 access-trace validator ---------------- */

/* D8-specific version of the CB-FIX-1 rule. It derives the last MMA read of
 * each operand from the recorded events and rejects any later-overlapping
 * write that lands before that read. It also checks the registered output
 * phase order (out_pos, then Wlo, then Ahi), the condition's fill mask and the
 * event counts, so a truncated, padded, reordered or C/D-swapped trace fails
 * closed. Returns 1 = valid, 0 = invalid; `why` receives a short reason. */
static int d8_trace_only_valid(const hmx_full_tile_state_t *st, int cond,
                               const char **why) {
  const int ids[4] = {HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_ALO,
                      HMX_B3_R2_OP_WHI, HMX_B3_R2_OP_WLO};
  const int offsets[4] = {HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_ALO_OFFSET,
                          HMX_B3_R2_PR_WHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET};
  const int reads_expected[4] = {2, 1, 2, 1};
  int pack[4] = {0}, last[4] = {0}, reads[4] = {0};
  int sentinel = 0, hmx_stores = 0, hvx_stores = 0, bias_writes = 0;
  int out_fill = 0, wlo_fill = 0, ahi_fill = 0;
  int out_seq = 0, wlo_seq = 0, ahi_seq = 0;
  int first_fill_seq = 0, last_mma_read_seq = 0;
  if (why) *why = "unset";
  if (!st->pr_trace_active || st->pr_trace_overflow ||
      st->pr_trace_count < 1 || st->pr_trace_count > PR_TRACE_CAP) {
    if (why) *why = "trace_missing_or_overflow";
    return 0;
  }
  for (int i = 0; i < st->pr_trace_count; ++i) {
    const pr_event_t *e = &st->pr_trace[i];
    if (e->seq != i + 1 || e->kind < PR_PACK || e->kind > PR_INIT_WRITE ||
        e->offset < 0 || e->size <= 0 ||
        e->offset > HMX_B3_R2_VTCM_TOTAL ||
        e->size > HMX_B3_R2_VTCM_TOTAL - e->offset ||
        e->write != (e->kind != PR_MMA_READ)) {
      if (why) *why = "malformed_event";
      return 0;
    }
    if (e->kind == PR_PACK || e->kind == PR_MMA_READ) {
      int j;
      for (j = 0; j < 4; ++j) {
        if (e->object == ids[j]) {
          if (e->offset != offsets[j] ||
              e->size != HMX_B3_R2_TILE_BYTES) {
            if (why) *why = "operand_event_geometry";
            return 0;
          }
          if (e->kind == PR_PACK) {
            if (pack[j]) { if (why) *why = "double_pack"; return 0; }
            pack[j] = e->seq;
          } else {
            if (!pack[j]) { if (why) *why = "read_before_pack"; return 0; }
            reads[j]++;
            last[j] = e->seq;
            last_mma_read_seq = e->seq;
          }
          break;
        }
      }
      if (j == 4) { if (why) *why = "unknown_operand_object"; return 0; }
    } else if (e->kind == PR_SENTINEL) {
      sentinel++;
      if (first_fill_seq == 0) first_fill_seq = e->seq;
      if (e->offset == D8_OUT_OFFSET && e->size == HMX_B3_R2_TILE_BYTES) {
        out_fill++;
        out_seq = e->seq;
      } else if (e->offset == HMX_B3_R2_PR_WLO_OFFSET &&
                 e->size == HMX_B3_R2_TILE_BYTES) {
        wlo_fill++;
        wlo_seq = e->seq;
      } else if (e->offset == HMX_B3_R2_PR_AHI_OFFSET &&
                 e->size == HMX_B3_R2_TILE_BYTES) {
        ahi_fill++;
        ahi_seq = e->seq;
      } else {
        if (why) *why = "sentinel_outside_declared_regions";
        return 0;
      }
    } else if (e->kind == PR_HMX_STORE) {
      hmx_stores++;
      if (e->offset != D8_OUT_OFFSET) {
        if (why) *why = "store_outside_out";
        return 0;
      }
    } else if (e->kind == PR_HVX_STORE) {
      hvx_stores++;
    } else if (e->kind == PR_BIAS_WRITE) {
      bias_writes++;
    }
  }
  if (sentinel < 1 || out_fill != 1) { if (why) *why = "missing_out_fill"; return 0; }
  if (hmx_stores != 1) { if (why) *why = "hmx_store_count"; return 0; }
  if (hvx_stores != 0) { if (why) *why = "unexpected_hvx_store"; return 0; }
  if (bias_writes != 2) { if (why) *why = "bias_count"; return 0; }
  /* Revision 2: every registered fill belongs to the output phase, so none
   * may precede the last MMA operand read. Revision 1 placed the out_pos fill
   * in the prologue; that sequence is a different experiment and fails here. */
  if (first_fill_seq != 0 && last_mma_read_seq != 0 &&
      first_fill_seq < last_mma_read_seq) {
    if (why) *why = "fill_before_last_mma_read";
    return 0;
  }
  /* Candidate B output-phase order: out_pos first, then Wlo, then Ahi. */
  if (out_fill == 1 && wlo_fill == 1 && wlo_seq < out_seq) {
    if (why) *why = "fill_order_wlo_before_out";
    return 0;
  }
  if (ahi_fill == 1) {
    int prior = (wlo_fill == 1) ? wlo_seq : out_seq;
    if (ahi_seq < prior) { if (why) *why = "fill_order_ahi"; return 0; }
  }
  {
    /* The mask is derived from the observed fills, then compared with the
     * registered per-condition value. A C/D swap (C filling Ahi) and a B path
     * that drops one operand fill both fail here. */
    int observed = (out_fill ? 1 : 0) | (wlo_fill ? 2 : 0) | (ahi_fill ? 4 : 0);
    if (observed != d8_condition_mask(cond)) {
      if (why) *why = "condition_fill_mask";
      return 0;
    }
  }
  for (int j = 0; j < 4; ++j) {
    if (reads[j] != reads_expected[j]) {
      if (why) *why = "mma_read_count";
      return 0;
    }
    if (!pack[j]) { if (why) *why = "missing_pack"; return 0; }
    /* CB-FIX-1 rule: every write that overlaps this operand and lands after
     * its pack but before its last read is a violation. */
    for (int i = 0; i < st->pr_trace_count; ++i) {
      const pr_event_t *e = &st->pr_trace[i];
      if (e->write && e->seq > pack[j] && e->seq < last[j] &&
          e->offset < offsets[j] + HMX_B3_R2_TILE_BYTES &&
          offsets[j] < e->offset + e->size) {
        if (why) *why = "write_before_last_read";
        return 0;
      }
    }
  }
  if (why) *why = "ok";
  return 1;
}

/* ---------------- revision-3 program-order validator ---------------- */

/* The expected issued-operation list for a condition. GUARD_PRE/GUARD_POST
 * carry runtime values, so only their presence and kind are compared here;
 * their payload is validated against the memory image below. */
static int d8_expected_ops(int cond, int require_guard, d8_op_t *out) {
  int n = 0;
  const int mask = d8_condition_mask(cond);
#define D8_EXP(k, aa, bb, cc)                       \
  do {                                              \
    out[n].kind = (k);                              \
    out[n].a = (aa);                                \
    out[n].b = (bb);                                \
    out[n].c = (cc);                                \
    ++n;                                            \
  } while (0)
  D8_EXP(D8_OP_PACK, HMX_B3_R2_OP_AHI, HMX_B3_R2_PR_AHI_OFFSET, 0);
  D8_EXP(D8_OP_PACK, HMX_B3_R2_OP_ALO, HMX_B3_R2_PR_ALO_OFFSET, 0);
  D8_EXP(D8_OP_PACK, HMX_B3_R2_OP_WHI, HMX_B3_R2_PR_WHI_OFFSET, 0);
  D8_EXP(D8_OP_PACK, HMX_B3_R2_OP_WLO, HMX_B3_R2_PR_WLO_OFFSET, 0);
  D8_EXP(D8_OP_BIAS, 0, HMX_B3_R2_PR_BIAS0_OFFSET, 1);
  D8_EXP(D8_OP_BIAS, 1, HMX_B3_R2_PR_BIAS1_OFFSET, 1);
  if (require_guard) {
    D8_EXP(D8_OP_GUARD_INIT, HMX_D8_GUARD_VECTORS, HMX_D8_GUARD_BYTES,
           HMX_D8_GUARD_INIT_HVX12);
    D8_EXP(D8_OP_GUARD_PRE, 0, 0, 0);
  }
  D8_EXP(D8_OP_MXCLRACC, 0, 0, 0);
  D8_EXP(D8_OP_MMA, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET, 0);
  D8_EXP(D8_OP_MMA, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET, 0);
  D8_EXP(D8_OP_MMA, HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET, 0);
  if (mask & D8_MASK_OUT) D8_EXP(D8_OP_FILL, D8_OUT_OFFSET, 1, HMX_B3_R2_TILE_BYTES);
  if (mask & D8_MASK_WLO)
    D8_EXP(D8_OP_FILL, HMX_B3_R2_PR_WLO_OFFSET, 2, HMX_B3_R2_TILE_BYTES);
  if (mask & D8_MASK_AHI)
    D8_EXP(D8_OP_FILL, HMX_B3_R2_PR_AHI_OFFSET, 3, HMX_B3_R2_TILE_BYTES);
  D8_EXP(D8_OP_CONVERT, (int)HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET, 0);
  D8_EXP(D8_OP_STORE, D8_OUT_OFFSET, 0, 0);
  if (require_guard) D8_EXP(D8_OP_GUARD_POST, 0, 0, 0);
#undef D8_EXP
  return n;
}

static int d8_op_equal(const d8_op_t *a, const d8_op_t *b) {
  if (a->kind != b->kind) return 0;
  if (a->kind == D8_OP_GUARD_PRE || a->kind == D8_OP_GUARD_POST) return 1;
  return a->a == b->a && a->b == b->b && a->c == b->c;
}

static int d8_count_ops(int kind) {
  int n = 0;
  for (int i = 0; i < g_d8_log.count; ++i) {
    if (g_d8_log.op[i].kind == kind) n++;
  }
  return n;
}

static const d8_op_t *d8_find_op(int kind) {
  for (int i = 0; i < g_d8_log.count; ++i) {
    if (g_d8_log.op[i].kind == kind) return &g_d8_log.op[i];
  }
  return 0;
}

static int d8_validate_program(const hmx_full_tile_state_t *st, int cond,
                               int require_guard, const char **why) {
  d8_op_t expected[D8_OPLOG_CAP];
  int expected_n = d8_expected_ops(cond, require_guard, expected);
  int fills = 0;
  int first_mma_read = 0, last_mma_read = 0, last_fill = 0, store_seq = 0;
  int bias_n = 0, bias_off[2] = {0, 0}, bias_seq[2] = {0, 0};
  int guard_init_events = 0;
  int sentinel_n = 0;
  int sentinel_seq[4] = {0, 0, 0, 0};
  int sentinel_off[4] = {0, 0, 0, 0};
  int mma_seq[6] = {0, 0, 0, 0, 0, 0};
  int mma_off[6] = {0, 0, 0, 0, 0, 0};
  int mma_n = 0;

  if (!st->pr_trace_active || st->pr_trace_overflow ||
      st->pr_trace_count < 1 || st->pr_trace_count > PR_TRACE_CAP) {
    if (why) *why = "trace_missing_or_overflow";
    return 0;
  }
  /* Guard ownership before any other rule: only the model's whole-VTCM setup
   * write and (in revision 3) the registered 1,536 B zero init may touch
   * +10752..+12288. This makes a mode-45-style third bias set fail as a
   * region violation, not merely as a bias count. */
  for (int i = 0; i < st->pr_trace_count; ++i) {
    const pr_event_t *e = &st->pr_trace[i];
    if (e->offset < HMX_D8_GUARD_OFFSET + HMX_D8_GUARD_BYTES &&
        HMX_D8_GUARD_OFFSET < e->offset + e->size) {
      if (e->kind == PR_INIT_WRITE && e->offset == 0 &&
          e->size == HMX_B3_R2_VTCM_TOTAL) {
        continue;
      }
      if (require_guard && e->kind == PR_INIT_WRITE &&
          e->offset == HMX_D8_GUARD_OFFSET &&
          e->size == HMX_D8_GUARD_BYTES) {
        continue;
      }
      if (why) *why = "outside_declared_regions";
      return 0;
    }
  }
  if (!d8_trace_only_valid(st, cond, why)) return 0;

  for (int i = 0; i < st->pr_trace_count; ++i) {
    const pr_event_t *e = &st->pr_trace[i];
    if (e->kind == PR_MMA_READ) {
      if (!first_mma_read) first_mma_read = e->seq;
      last_mma_read = e->seq;
      if (mma_n < 6) {
        mma_seq[mma_n] = e->seq;
        mma_off[mma_n] = e->offset;
        mma_n++;
      }
    } else if (e->kind == PR_SENTINEL) {
      if (e->offset == D8_OUT_OFFSET || e->offset == HMX_B3_R2_PR_WLO_OFFSET ||
          e->offset == HMX_B3_R2_PR_AHI_OFFSET) {
        if (sentinel_n < 4) {
          sentinel_seq[sentinel_n] = e->seq;
          sentinel_off[sentinel_n] = e->offset;
          sentinel_n++;
        }
        last_fill = e->seq;
      }
    } else if (e->kind == PR_BIAS_WRITE) {
      if (bias_n < 2) {
        bias_off[bias_n] = e->offset;
        bias_seq[bias_n] = e->seq;
      }
      bias_n++;
    } else if (e->kind == PR_HMX_STORE) {
      store_seq = e->seq;
    }
    /* Guard-region ownership: only the model's whole-VTCM setup write (and,
     * in revision 3, the registered 1,536 B zero init) may touch the tail. */
    if (e->offset < HMX_D8_GUARD_OFFSET + HMX_D8_GUARD_BYTES &&
        HMX_D8_GUARD_OFFSET < e->offset + e->size) {
      if (e->kind == PR_INIT_WRITE && e->offset == 0 &&
          e->size == HMX_B3_R2_VTCM_TOTAL) {
        /* model initialisation, not a device instruction */
      } else if (require_guard && e->kind == PR_INIT_WRITE &&
                 e->offset == HMX_D8_GUARD_OFFSET &&
                 e->size == HMX_D8_GUARD_BYTES) {
        guard_init_events++;
      } else {
        if (why) *why = "outside_declared_regions";
        return 0;
      }
    }
  }

  fills = sentinel_n;
  if (fills != ((d8_condition_mask(cond) & 1) + ((d8_condition_mask(cond) >> 1) & 1) +
                ((d8_condition_mask(cond) >> 2) & 1))) {
    if (why) *why = "fill_event_count";
    return 0;
  }
  if (require_guard && guard_init_events != 1) {
    if (why) *why = "guard_init_event";
    return 0;
  }
  if (!require_guard && guard_init_events != 0) {
    if (why) *why = "unexpected_guard_init";
    return 0;
  }

  /* Bias writes: set 0 then set 1, at the registered offsets, before the
   * first MMA read. A bias write after mxclracc would still convert correctly
   * because Convert reads the bias, so only the order check catches it. */
  if (bias_n != 2 || bias_off[0] != HMX_B3_R2_PR_BIAS0_OFFSET ||
      bias_off[1] != HMX_B3_R2_PR_BIAS1_OFFSET) {
    if (why) *why = "bias_offset";
    return 0;
  }
  if (bias_seq[0] >= bias_seq[1]) {
    if (why) *why = "bias_order";
    return 0;
  }
  if (first_mma_read != 0 && bias_seq[1] > first_mma_read) {
    if (why) *why = "bias_after_mma";
    return 0;
  }

  /* Issued-operation record: present, not overflowed, then per-kind counts.
   * Checking counts before order means a skipped operation names the
   * operation ("convert_count") instead of a generic order failure. */
  if (!g_d8_log.enabled) { if (why) *why = "oplog_missing"; return 0; }
  if (g_d8_log.overflow) { if (why) *why = "oplog_overflow"; return 0; }
  if (st->count_mxclracc != 1 || d8_count_ops(D8_OP_MXCLRACC) != 1) {
    if (why) *why = "mxclracc_count";
    return 0;
  }
  if (st->count_mma != 3 || d8_count_ops(D8_OP_MMA) != 3) {
    if (why) *why = "mma_count";
    return 0;
  }
  if (st->count_convert != 1 || d8_count_ops(D8_OP_CONVERT) != 1) {
    if (why) *why = "convert_count";
    return 0;
  }
  if (st->count_store != 1 || d8_count_ops(D8_OP_STORE) != 1) {
    if (why) *why = "store_count";
    return 0;
  }
  if (d8_count_ops(D8_OP_PACK) != 4 || d8_count_ops(D8_OP_BIAS) != 2) {
    if (why) *why = "input_op_count";
    return 0;
  }

  /* Bias order and position in the issued list: both sets before mxclracc.
   * Convert reads the bias, so a bias write moved after the MMAs still
   * produces the safe numbers; only the order rule rejects it. */
  {
    int bias_i = 0, mx_i = -1, bias_set0_i = -1, bias_set1_i = -1;
    for (int i = 0; i < g_d8_log.count; ++i) {
      if (g_d8_log.op[i].kind == D8_OP_MXCLRACC) mx_i = i;
      if (g_d8_log.op[i].kind == D8_OP_BIAS) {
        if (g_d8_log.op[i].a == 0) bias_set0_i = i;
        if (g_d8_log.op[i].a == 1) bias_set1_i = i;
        bias_i++;
      }
    }
    if (bias_i != 2 || bias_set0_i < 0 || bias_set1_i < 0) {
      if (why) *why = "bias_op_count";
      return 0;
    }
    if (bias_set1_i < bias_set0_i) { if (why) *why = "bias_op_order"; return 0; }
    if (mx_i >= 0 && bias_set1_i > mx_i) {
      if (why) *why = "bias_after_mxclracc";
      return 0;
    }
  }

  /* Convert must follow every fill, and Store must follow Convert. The access
   * trace sees Convert+Store as one HMX store event, so use the issued list
   * for the Convert/Store order and the trace for the fill positions. */
  {
    const d8_op_t *cvt = d8_find_op(D8_OP_CONVERT);
    const d8_op_t *store = d8_find_op(D8_OP_STORE);
    int cvt_index = -1, store_index = -1, fill_last_index = -1;
    int fill_first_index = -1;
    for (int i = 0; i < g_d8_log.count; ++i) {
      if (g_d8_log.op[i].kind == D8_OP_CONVERT) cvt_index = i;
      if (g_d8_log.op[i].kind == D8_OP_STORE) store_index = i;
      if (g_d8_log.op[i].kind == D8_OP_FILL) {
        if (fill_first_index < 0) fill_first_index = i;
        fill_last_index = i;
      }
    }
    if (!cvt || !store || cvt_index < 0 || store_index < 0 ||
        fill_last_index < 0) {
      if (why) *why = "output_phase_missing";
      return 0;
    }
    if (fill_first_index < cvt_index && fill_last_index > cvt_index) {
      if (why) *why = "convert_between_fills";
      return 0;
    }
    if (last_fill > store_seq && store_seq != 0) {
      if (why) *why = "store_before_fill";
      return 0;
    }
    if (cvt_index > store_index) {
      if (why) *why = "store_before_convert";
      return 0;
    }
  }

  /* Guard payload: recompute from memory. A false self-report (a clean status
   * with a dirty tail) fails here, not at the status check. */
  if (require_guard) {
    const d8_op_t *pre = d8_find_op(D8_OP_GUARD_PRE);
    const d8_op_t *post = d8_find_op(D8_OP_GUARD_POST);
    unsigned int zero_fnv = hmx_d8_guard_zero_fnv();
    unsigned int now = hmx_d8_fnv1a32(st->vtcm.raw + HMX_D8_GUARD_OFFSET,
                                      HMX_D8_GUARD_BYTES);
    if (!pre || !post) { if (why) *why = "guard_op_missing"; return 0; }
    if ((unsigned int)pre->a != zero_fnv || pre->b != 1) {
      if (why) *why = "guard_pre_nonzero";
      return 0;
    }
    if (post->b != HMX_D8_GUARD_STATUS_OK ||
        post->c != HMX_D8_GUARD_FIRST_BAD_NONE ||
        (unsigned int)post->a != now) {
      if (why) *why = "guard_post_mismatch";
      return 0;
    }
    if (now != zero_fnv) { if (why) *why = "guard_tail_dirty"; return 0; }
  } else {
    if (d8_find_op(D8_OP_GUARD_INIT) || d8_find_op(D8_OP_GUARD_PRE) ||
        d8_find_op(D8_OP_GUARD_POST)) {
      if (why) *why = "unexpected_guard_op";
      return 0;
    }
  }

  /* Exact access-event count: a dropped, duplicated or padded event fails
   * even when every remaining event looks well formed. */
  {
    int expected_events = 6 + 4 + 2 + 6 + fills + 1 + (require_guard ? 1 : 0);
    if (st->pr_trace_count != expected_events) {
      if (why) *why = "event_count";
      return 0;
    }
  }

  /* Exact issued-operation list: count and every field, in order. */
  if (g_d8_log.count != expected_n) {
    if (why) *why = "oplog_count";
    return 0;
  }
  for (int i = 0; i < expected_n; ++i) {
    if (!d8_op_equal(&g_d8_log.op[i], &expected[i])) {
      if (why) *why = "oplog_order";
      return 0;
    }
  }

  /* Cross-check the two records: operation list vs access events. */
  {
    int fill_i = 0;
    for (int i = 0; i < g_d8_log.count; ++i) {
      if (g_d8_log.op[i].kind != D8_OP_FILL) continue;
      if (fill_i >= sentinel_n ||
          g_d8_log.op[i].a != sentinel_off[fill_i]) {
        if (why) *why = "fill_record_mismatch";
        return 0;
      }
      fill_i++;
    }
    if (fill_i != sentinel_n) { if (why) *why = "fill_record_mismatch"; return 0; }
  }
  {
    int mma_i = 0;
    for (int i = 0; i < g_d8_log.count; ++i) {
      if (g_d8_log.op[i].kind != D8_OP_MMA) continue;
      if (mma_i + 1 >= mma_n ||
          g_d8_log.op[i].a != mma_off[mma_i] ||
          g_d8_log.op[i].b != mma_off[mma_i + 1] ||
          mma_seq[mma_i] >= mma_seq[mma_i + 1]) {
        if (why) *why = "mma_record_mismatch";
        return 0;
      }
      mma_i += 2;
    }
    if (mma_i != mma_n) { if (why) *why = "mma_record_mismatch"; return 0; }
  }

  /* Every registered fill is in the output phase: the first fill event must
   * follow the last MMA read. trace_only_valid checks the same property from
   * its own last-read state; this duplicate is kept explicit because a future
   * change to that function must not silently drop the rule. */
  if (last_mma_read != 0 && sentinel_seq[0] != 0 &&
      sentinel_seq[0] < last_mma_read) {
    if (why) *why = "fill_before_last_mma_read";
    return 0;
  }

  /* Guard payload: recompute from memory. A false self-report (a clean status
   * with a dirty tail) fails here, not at the status check. */
  if (require_guard) {
    const d8_op_t *pre = d8_find_op(D8_OP_GUARD_PRE);
    const d8_op_t *post = d8_find_op(D8_OP_GUARD_POST);
    unsigned int zero_fnv = hmx_d8_guard_zero_fnv();
    unsigned int now = hmx_d8_fnv1a32(st->vtcm.raw + HMX_D8_GUARD_OFFSET,
                                      HMX_D8_GUARD_BYTES);
    if (!pre || !post) { if (why) *why = "guard_op_missing"; return 0; }
    if ((unsigned int)pre->a != zero_fnv || pre->b != 1) {
      if (why) *why = "guard_pre_nonzero";
      return 0;
    }
    if (post->b != HMX_D8_GUARD_STATUS_OK ||
        post->c != HMX_D8_GUARD_FIRST_BAD_NONE ||
        (unsigned int)post->a != now) {
      if (why) *why = "guard_post_mismatch";
      return 0;
    }
    if (now != zero_fnv) { if (why) *why = "guard_tail_dirty"; return 0; }
  } else {
    if (d8_find_op(D8_OP_GUARD_INIT) || d8_find_op(D8_OP_GUARD_PRE) ||
        d8_find_op(D8_OP_GUARD_POST)) {
      if (why) *why = "unexpected_guard_op";
      return 0;
    }
  }

  if (why) *why = "ok";
  return 1;
}

static int d8_trace_valid(const hmx_full_tile_state_t *st, int cond,
                          const char **why) {
  return d8_validate_program(st, cond, 0, why);
}

static int d8_rev3_valid(const hmx_full_tile_state_t *st, int cond,
                         const char **why) {
  return d8_validate_program(st, cond, 1, why);
}

/* ---------------- classifier ---------------- */

static int d8_array_equal(const uint16_t *a, const uint16_t *b) {
  for (int i = 0; i < T8_CELLS; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

/* Returns the index of the matching reference, or -1. */
static int d8_match_ref(const uint16_t *obs, const d8_refs_t *r) {
  if (d8_array_equal(obs, r->safe)) return 0;
  if (d8_array_equal(obs, r->wlo)) return 1;
  if (d8_array_equal(obs, r->ahi)) return 2;
  if (d8_array_equal(obs, r->both)) return 3;
  return -1;
}

/* Registered classification order (prereg section 6):
 *   contract -> trace -> numeric. A trace anomaly overrides a matching numeric
 *   result; an unmatched numeric result is never read as "no violation". The
 *   contract gate includes the echoed fill mask: a C run that reports D's mask
 *   (or a B run that dropped one operand fill) is a contract failure, never a
 *   numeric attribution. */
static d8_class_t d8_classify(int cond, const uint16_t *obs, int trace_status,
                              int evidence_ok, int ids_ok, int lengths_ok,
                              int fill_mask, const d8_refs_t *r,
                              int *matched_out) {
  int matched;
  if (!evidence_ok || !ids_ok || !lengths_ok) return D8_CLASS_CONTRACT_FAILED;
  if (fill_mask != d8_condition_mask(cond)) return D8_CLASS_CONTRACT_FAILED;
  if (trace_status == D8_TRACE_INVALID) return D8_CLASS_INCONCLUSIVE_TRACE;
  matched = d8_match_ref(obs, r);
  if (matched_out) *matched_out = matched;
  switch (cond) {
    case D8_COND_A:
      return matched == 0 ? D8_CLASS_CONTROL_SAFE : D8_CLASS_CONTROL_ANOMALY;
    case D8_COND_B:
      if (matched == 0) return D8_CLASS_NO_VIOLATION;
      if (matched == 1) return D8_CLASS_VIOLATION_WLO;
      if (matched == 2) return D8_CLASS_VIOLATION_AHI;
      if (matched == 3) return D8_CLASS_VIOLATION_BOTH;
      return D8_CLASS_INCONCLUSIVE_NUMERIC;
    case D8_COND_C:
      if (matched == 0) return D8_CLASS_NO_VIOLATION;
      if (matched == 1) return D8_CLASS_VIOLATION_WLO;
      return D8_CLASS_INCONCLUSIVE_NUMERIC;
    case D8_COND_D:
      if (matched == 0) return D8_CLASS_NO_VIOLATION;
      if (matched == 2) return D8_CLASS_VIOLATION_AHI;
      return D8_CLASS_INCONCLUSIVE_NUMERIC;
    default:
      return D8_CLASS_CONTRACT_FAILED;
  }
}

/* ---------------- registered event signature ---------------- */

/* The evidence contract fixes one output-phase order: out_pos, then Wlo, then
 * Ahi, all after the third MMA. The signature below is derived from the
 * recorded access events (not from the call sites), so a reordered or
 * truncated trace changes it. The Python audit extracts the same ordered list
 * from the pre-registration and compares it with `--d8-sequence`. */

static const char *d8_operand_name(int id) {
  switch (id) {
    case HMX_B3_R2_OP_AHI: return "AHI";
    case HMX_B3_R2_OP_ALO: return "ALO";
    case HMX_B3_R2_OP_WHI: return "WHI";
    case HMX_B3_R2_OP_WLO: return "WLO";
    default: return "?";
  }
}

static const char *d8_fill_region(int offset) {
  if (offset == D8_OUT_OFFSET) return "OUT";
  if (offset == HMX_B3_R2_PR_WLO_OFFSET) return "WLO";
  if (offset == HMX_B3_R2_PR_AHI_OFFSET) return "AHI";
  return "?";
}

static void d8_sig_append(char *buf, size_t cap, size_t *n, const char *token) {
  if (*n >= cap - 1) return;
  int written = snprintf(buf + *n, cap - *n, "%s%s", *n ? "," : "", token);
  if (written > 0 && (size_t)written < cap - *n) *n += (size_t)written;
}

static void d8_print_sequence(const hmx_full_tile_state_t *st, int cond) {
  char sig[1024];
  size_t n = 0;
  int reads = 0, act = 0;
  char token[64];
  sig[0] = '\0';
  for (int i = 0; i < st->pr_trace_count; ++i) {
    const pr_event_t *e = &st->pr_trace[i];
    switch (e->kind) {
      case PR_PACK:
        snprintf(token, sizeof(token), "PACK:%s", d8_operand_name(e->object));
        d8_sig_append(sig, sizeof(sig), &n, token);
        break;
      case PR_BIAS_WRITE:
        snprintf(token, sizeof(token), "BIAS:%d", e->object);
        d8_sig_append(sig, sizeof(sig), &n, token);
        break;
      case PR_MMA_READ:
        if ((reads & 1) == 0) {
          act = e->object;
        } else {
          snprintf(token, sizeof(token), "MMA:%d:%s:%s", reads / 2 + 1,
                   d8_operand_name(act), d8_operand_name(e->object));
          d8_sig_append(sig, sizeof(sig), &n, token);
        }
        reads++;
        break;
      case PR_SENTINEL:
        snprintf(token, sizeof(token), "FILL:%s", d8_fill_region(e->offset));
        d8_sig_append(sig, sizeof(sig), &n, token);
        break;
      case PR_HMX_STORE:
        /* The shared model records Convert + Store as one HMX store event;
         * the registered sequence names both operations. */
        d8_sig_append(sig, sizeof(sig), &n, "CONVERT:1");
        d8_sig_append(sig, sizeof(sig), &n, "STORE:OUT");
        break;
      default:
        break;
    }
  }
  printf("d8_seq cond=%c events=%s\n", 'A' + cond, sig);
}

static void d8_print_sequences(void) {
  static hmx_full_tile_state_t st;
  printf("d8_fill_mask A=%d B=%d C=%d D=%d\n",
         d8_condition_mask(D8_COND_A), d8_condition_mask(D8_COND_B),
         d8_condition_mask(D8_COND_C), d8_condition_mask(D8_COND_D));
  for (int c = 0; c < 4; ++c) {
    d8_sequence_with_mask(&st, d8_condition_mask(c));
    d8_print_sequence(&st, c);
  }
}

/* Revision-3 issued-operation signature, derived from the recorder (not from
 * the call sites). The Python audit parses the matching block in the rev3
 * pre-registration and requires all four condition signatures to be equal. */
static int d8_id_from_offset(int off) {
  switch (off) {
    case HMX_B3_R2_PR_AHI_OFFSET: return HMX_B3_R2_OP_AHI;
    case HMX_B3_R2_PR_ALO_OFFSET: return HMX_B3_R2_OP_ALO;
    case HMX_B3_R2_PR_WHI_OFFSET: return HMX_B3_R2_OP_WHI;
    case HMX_B3_R2_PR_WLO_OFFSET: return HMX_B3_R2_OP_WLO;
    default: return 0;
  }
}

static void d8_print_rev3_sequences(void) {
  static hmx_full_tile_state_t st;
  char sig[2048];
  for (int c = 0; c < 4; ++c) {
    size_t n = 0;
    int mma_index = 0;
    sig[0] = '\0';
    d8_rev3_sequence(&st, c);
    for (int i = 0; i < g_d8_log.count; ++i) {
      char token[80];
      const d8_op_t *op = &g_d8_log.op[i];
      switch (op->kind) {
        case D8_OP_PACK:
          snprintf(token, sizeof(token), "PACK:%s",
                   d8_operand_name(op->a));
          break;
        case D8_OP_BIAS:
          snprintf(token, sizeof(token), "BIAS:%d", op->a);
          break;
        case D8_OP_GUARD_INIT:
          snprintf(token, sizeof(token), "GUARD:INIT:%d", op->a);
          break;
        case D8_OP_GUARD_PRE:
          snprintf(token, sizeof(token), "GUARD:PRE");
          break;
        case D8_OP_GUARD_POST:
          snprintf(token, sizeof(token), "GUARD:POST");
          break;
        case D8_OP_MXCLRACC:
          snprintf(token, sizeof(token), "MXCLRACC");
          break;
        case D8_OP_MMA:
          snprintf(token, sizeof(token), "MMA:%d:%s:%s", ++mma_index,
                   d8_operand_name(d8_id_from_offset(op->a)),
                   d8_operand_name(d8_id_from_offset(op->b)));
          break;
        case D8_OP_FILL:
          snprintf(token, sizeof(token), "FILL:%s",
                   d8_fill_region(op->a));
          break;
        case D8_OP_CONVERT:
          snprintf(token, sizeof(token), "CONVERT:1");
          break;
        case D8_OP_STORE:
          snprintf(token, sizeof(token), "STORE:OUT");
          break;
        default:
          snprintf(token, sizeof(token), "?");
          break;
      }
      d8_sig_append(sig, sizeof(sig), &n, token);
    }
    printf("d8_rev3_seq cond=%c events=%s\n", 'A' + c, sig);
  }
}

/* ---------------- references ---------------- */

static void d8_build_references(void) {
  d8_outcome(g_d8_refs.safe, 0, 0);
  d8_outcome(g_d8_refs.wlo, 0, 1);
  d8_outcome(g_d8_refs.ahi, 1, 0);
  d8_outcome(g_d8_refs.both, 1, 1);
}

static int d8_count_diff(const uint16_t *a, const uint16_t *b) {
  int n = 0;
  for (int i = 0; i < T8_CELLS; ++i) {
    if (a[i] != b[i]) n++;
  }
  return n;
}

static void d8_sha(const uint16_t *a, char *out) {
  static uint8_t buf[T8_CELLS * 2];
  put_u16_le(buf, a, T8_CELLS);
  sha256_hex(buf, sizeof(buf), out);
}

/* FNV-1a over a packed operand tile, in the same layout the MMA reads. The
 * device can echo these four values cheaply; the host compares them with the
 * registered fixture so an input mix-up fails closed before any numeric
 * interpretation. */
static uint32_t d8_tile_fnv(const uint16_t *src, int weight) {
  static uint8_t buf[HMX_B3_R2_TILE_BYTES];
  if (weight) {
    pack_wgt_prm((uint16_t *)buf, src);
  } else {
    pack_act_prm((uint16_t *)buf, src);
  }
  return fnv1a32(buf, sizeof(buf));
}

static void d8_dump(const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    printf("d8_dump=not_written path=%s\n", path);
    return;
  }
  static uint8_t ops[T8_CELLS * 12];
  size_t ops_len = 0;
  {
    static uint8_t all[T8_CELLS * 12];
    put_opset(all, &g_ops_deploy);
    ops_len = sizeof(all);
    memcpy(ops, all, ops_len);
  }
  char sha_ops[65], sha_safe[65], sha_wlo[65], sha_ahi[65], sha_both[65];
  sha256_hex(ops, ops_len, sha_ops);
  d8_sha(g_d8_refs.safe, sha_safe);
  d8_sha(g_d8_refs.wlo, sha_wlo);
  d8_sha(g_d8_refs.ahi, sha_ahi);
  d8_sha(g_d8_refs.both, sha_both);
  fprintf(f, "# D8 minimal experiment CPU references (u16 LE, decoded cell order)\n");
  fprintf(f, "# sha256_operands_deploy=%s\n", sha_ops);
  fprintf(f, "# sha256_safe=%s\n", sha_safe);
  fprintf(f, "# sha256_wlo=%s\n", sha_wlo);
  fprintf(f, "# sha256_ahi=%s\n", sha_ahi);
  fprintf(f, "# sha256_both=%s\n", sha_both);
  fprintf(f, "# fnv_ahi=%08x\n", d8_tile_fnv(g_ops_deploy.ahi, 0));
  fprintf(f, "# fnv_alo=%08x\n", d8_tile_fnv(g_ops_deploy.alo, 0));
  fprintf(f, "# fnv_whi=%08x\n", d8_tile_fnv(g_ops_deploy.whi, 1));
  fprintf(f, "# fnv_wlo=%08x\n", d8_tile_fnv(g_ops_deploy.wlo, 1));
  fprintf(f, "section=operands_deploy\n");
  for (size_t i = 0; i < ops_len; ++i) fprintf(f, "%02x\n", ops[i]);
  {
    const uint16_t *arr[4] = {g_d8_refs.safe, g_d8_refs.wlo, g_d8_refs.ahi,
                              g_d8_refs.both};
    const char *names[4] = {"safe", "wlo", "ahi", "both"};
    for (int a = 0; a < 4; ++a) {
      fprintf(f, "section=%s\n", names[a]);
      for (int i = 0; i < T8_CELLS; ++i) fprintf(f, "%04x\n", arr[a][i]);
    }
  }
  fclose(f);
  printf("d8_dump=written path=%s sha_safe=%s sha_wlo=%s sha_ahi=%s "
         "sha_both=%s\n", path, sha_safe, sha_wlo, sha_ahi, sha_both);
}

/* ---------------- controls / fault injection ---------------- */

static void d8_reuse_fill(hmx_full_tile_state_t *st, int off, int byte,
                          int object) {
  pr_event(st, PR_SENTINEL, off, HMX_B3_R2_TILE_BYTES, 1, object);
  memset(st->vtcm.raw + off, byte, HMX_B3_R2_TILE_BYTES);
}

/* Full program order with a deliberately early reuse write (fault probe).
 * The early write is recorded exactly like the scheduled one, so the trace
 * validator has to catch it from the access order, not from a fault flag. */
static void d8_early_sequence(hmx_full_tile_state_t *st, int early_ahi,
                              int early_wlo, int byte_ahi, int byte_wlo,
                              int partial_byte) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  if (partial_byte) {
    /* One byte of the last Whi byte, written after the pack and before the
     * reads: the CB-FIX-1 rule must catch the overlap even though the write
     * is one byte and does not move the output. */
    pr_event(st, PR_INIT_WRITE, HMX_B3_R2_PR_WLO_OFFSET - 1, 1, 1, 9);
    st->vtcm.raw[HMX_B3_R2_PR_WLO_OFFSET - 1] = 0x00;
  }
  if (early_ahi) d8_reuse_fill(st, HMX_B3_R2_PR_AHI_OFFSET, byte_ahi, 3);
  exec_mxclracc(st);
  st->act_id = HMX_B3_R2_OP_AHI;
  st->wgt_id = HMX_B3_R2_OP_WHI;
  exec_mma_packet_at(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  if (early_wlo) d8_reuse_fill(st, HMX_B3_R2_PR_WLO_OFFSET, byte_wlo, 2);
  st->act_id = HMX_B3_R2_OP_AHI;
  st->wgt_id = HMX_B3_R2_OP_WLO;
  exec_mma_packet_at(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  st->act_id = HMX_B3_R2_OP_ALO;
  st->wgt_id = HMX_B3_R2_OP_WHI;
  exec_mma_packet_at(st, HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  pr_fill(st, D8_OUT_OFFSET, HMX_B3_R2_TILE_BYTES, 1);
  exec_convert_and_store_at(st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
}

/* Full program order with the two operand fills or the out_pos fill moved to
 * the wrong point. Used only to prove that the revision-2 order rule rejects a
 * reordered output phase; the numeric result is unchanged because the fills
 * still write the same regions. */
static void d8_reordered_sequence(hmx_full_tile_state_t *st, int variant) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mma_sequence(st);
  if (variant == 0) {
    /* Condition B with Ahi filled before Wlo. */
    pr_fill(st, D8_OUT_OFFSET, HMX_B3_R2_TILE_BYTES, 1);
    pr_fill(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_TILE_BYTES, 3);
    pr_fill(st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_TILE_BYTES, 2);
  } else {
    /* Condition B with the out_pos fill after both operand fills. */
    pr_fill(st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_TILE_BYTES, 2);
    pr_fill(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_TILE_BYTES, 3);
    pr_fill(st, D8_OUT_OFFSET, HMX_B3_R2_TILE_BYTES, 1);
  }
  exec_convert_and_store_at(st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
}

static int d8_run_controls(void) {
  static hmx_full_tile_state_t st;
  static uint16_t out[T8_CELLS];
  const char *why = NULL;
  int ok;

  /* 1. Wlo early rewrite (non-sentinel byte) before MMA #2's read. The trace
   *    must reject it from the access order and the numeric result must not
   *    be the safe array. */
  d8_early_sequence(&st, 0, 1, 0x00, 0x00, 0);
  ok = !d8_trace_valid(&st, D8_COND_C, &why);
  cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
  d8_check(ok && !d8_array_equal(out, g_d8_refs.safe),
           "d8_fault_01_wlo_early_trace");
  printf("d8_fault=01 wlo_early trace_invalid=%d why=%s numeric_safe=%d\n",
         ok, why, d8_array_equal(out, g_d8_refs.safe));

  /* 2. Ahi early rewrite before MMA #1. */
  d8_early_sequence(&st, 1, 0, 0x00, 0x00, 0);
  ok = !d8_trace_valid(&st, D8_COND_D, &why);
  cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
  d8_check(ok && !d8_array_equal(out, g_d8_refs.safe),
           "d8_fault_02_ahi_early_trace");
  printf("d8_fault=02 ahi_early trace_invalid=%d why=%s numeric_safe=%d\n",
         ok, why, d8_array_equal(out, g_d8_refs.safe));

  /* 3. Both early. */
  d8_early_sequence(&st, 1, 1, 0x00, 0x00, 0);
  ok = !d8_trace_valid(&st, D8_COND_B, &why);
  cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
  d8_check(ok && !d8_array_equal(out, g_d8_refs.safe),
           "d8_fault_03_both_early_trace");
  printf("d8_fault=03 both_early trace_invalid=%d why=%s numeric_safe=%d\n",
         ok, why, d8_array_equal(out, g_d8_refs.safe));

  /* 4. One-byte boundary overlap after the pack, before the last read. */
  d8_early_sequence(&st, 0, 0, 0x00, 0x00, 1);
  ok = !d8_trace_valid(&st, D8_COND_A, &why);
  d8_check(ok, "d8_fault_04_boundary_overlap_trace");
  printf("d8_fault=04 boundary_overlap trace_invalid=%d why=%s\n", ok, why);

  /* 5. The existing sentinel value used as the early write: the numeric
   *    outcome must equal the registered Wlo-corrupt reference. */
  d8_early_sequence(&st, 0, 1, 0x00, HMX_B3_R2_SENTINEL_BYTE, 0);
  ok = !d8_trace_valid(&st, D8_COND_C, &why);
  cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
  d8_check(ok && d8_array_equal(out, g_d8_refs.wlo),
           "d8_fault_05_sentinel_early_trace");
  printf("d8_fault=05 sentinel_early trace_invalid=%d why=%s numeric_wlo=%d\n",
         ok, why, d8_array_equal(out, g_d8_refs.wlo));

  /* 6. Normal post-last-use reuse: every condition's registered order is
   *    valid, the numeric result equals the safe reference, and the
   *    classifier accepts the registered mask. */
  for (int c = 0; c < 4; ++c) {
    int valid, m = -1;
    d8_class_t cls;
    d8_program_order(&st, c);
    valid = d8_trace_valid(&st, c, &why);
    cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
    cls = d8_classify(c, out, D8_TRACE_VALID, 1, 1, 1,
                      d8_condition_mask(c), &g_d8_refs, &m);
    d8_check(valid && d8_array_equal(out, g_d8_refs.safe) &&
                 cls == (c == D8_COND_A ? D8_CLASS_CONTROL_SAFE
                                        : D8_CLASS_NO_VIOLATION),
             "d8_fault_06_normal_order");
    printf("d8_fault=06 condition=%s trace_valid=%d mask=%d numeric_safe=%d "
           "class=%s\n",
           d8_condition_name(c), valid, d8_condition_mask(c),
           d8_array_equal(out, g_d8_refs.safe), d8_class_name(cls));
  }

  /* 7. Numeric safe + trace invalid: must not be read as no violation. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_C, g_d8_refs.safe, D8_TRACE_INVALID,
                               1, 1, 1, d8_condition_mask(D8_COND_C),
                               &g_d8_refs, &m);
    ok = (c == D8_CLASS_INCONCLUSIVE_TRACE);
    d8_check(ok, "d8_fault_07_safe_numeric_bad_trace");
    printf("d8_fault=07 safe_numeric_bad_trace class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 8. Corrupt numeric + trace valid: must be reported, not excused. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_C, g_d8_refs.wlo, D8_TRACE_VALID,
                               1, 1, 1, d8_condition_mask(D8_COND_C),
                               &g_d8_refs, &m);
    ok = (c == D8_CLASS_VIOLATION_WLO);
    d8_check(ok, "d8_fault_08_corrupt_numeric_good_trace");
    printf("d8_fault=08 corrupt_numeric_good_trace class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 9. Missing evidence array. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                               0, 1, 1, d8_condition_mask(D8_COND_A),
                               &g_d8_refs, &m);
    ok = (c == D8_CLASS_CONTRACT_FAILED);
    d8_check(ok, "d8_fault_09_missing_evidence");
    printf("d8_fault=09 missing_evidence class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 10. Length / identifier mismatch. */
  {
    int m = -1;
    d8_class_t c1 = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                                1, 0, 1, d8_condition_mask(D8_COND_A),
                                &g_d8_refs, &m);
    d8_class_t c2 = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                                1, 1, 0, d8_condition_mask(D8_COND_A),
                                &g_d8_refs, &m);
    ok = (c1 == D8_CLASS_CONTRACT_FAILED && c2 == D8_CLASS_CONTRACT_FAILED);
    d8_check(ok, "d8_fault_10_length_id_mismatch");
    printf("d8_fault=10 length_id_mismatch bad_id=%s bad_len=%s detected=%d\n",
           d8_class_name(c1), d8_class_name(c2), ok);
  }

  /* 11. Missing out_pos sentinel: a store cannot be distinguished from an
   *     unwritten region, so the run is a contract failure. */
  {
    int m = -1, trace_bad, contract;
    d8_class_t cls;
    d8_sequence_with_mask(&st, D8_MASK_WLO | D8_MASK_AHI); /* B without out */
    trace_bad = !d8_trace_valid(&st, D8_COND_B, &why);
    cls = d8_classify(D8_COND_B, g_d8_refs.safe, D8_TRACE_VALID, 1, 1, 1,
                      D8_MASK_WLO | D8_MASK_AHI, &g_d8_refs, &m);
    contract = (cls == D8_CLASS_CONTRACT_FAILED);
    ok = trace_bad && strcmp(why, "missing_out_fill") == 0 && contract;
    d8_check(ok, "d8_fault_11_missing_out_fill");
    printf("d8_fault=11 missing_out_fill trace_invalid=%d why=%s class=%s "
           "detected=%d\n", trace_bad, why, d8_class_name(cls), ok);
  }

  /* 12. C/D swap: condition C executed with D's mask (Ahi filled instead of
   *     Wlo) and condition D with C's mask. Both fail the mask contract. */
  {
    int m = -1, tb, contract;
    d8_class_t cls;
    d8_sequence_with_mask(&st, d8_condition_mask(D8_COND_D)); /* 5 */
    tb = !d8_trace_valid(&st, D8_COND_C, &why);
    cls = d8_classify(D8_COND_C, g_d8_refs.ahi, D8_TRACE_VALID, 1, 1, 1,
                      d8_condition_mask(D8_COND_D), &g_d8_refs, &m);
    contract = (cls == D8_CLASS_CONTRACT_FAILED);
    ok = tb && strcmp(why, "condition_fill_mask") == 0 && contract;
    d8_check(ok, "d8_fault_12_c_uses_d_mask");
    printf("d8_fault=12 c_uses_d_mask trace_invalid=%d why=%s class=%s "
           "detected=%d\n", tb, why, d8_class_name(cls), ok);

    d8_sequence_with_mask(&st, d8_condition_mask(D8_COND_C)); /* 3 */
    tb = !d8_trace_valid(&st, D8_COND_D, &why);
    cls = d8_classify(D8_COND_D, g_d8_refs.wlo, D8_TRACE_VALID, 1, 1, 1,
                      d8_condition_mask(D8_COND_C), &g_d8_refs, &m);
    contract = (cls == D8_CLASS_CONTRACT_FAILED);
    ok = tb && strcmp(why, "condition_fill_mask") == 0 && contract;
    d8_check(ok, "d8_fault_12_d_uses_c_mask");
    printf("d8_fault=12 d_uses_c_mask trace_invalid=%d why=%s class=%s "
           "detected=%d\n", tb, why, d8_class_name(cls), ok);
  }

  /* 13. B with one operand fill missing, either side. */
  {
    int m = -1, tb, contract;
    d8_class_t cls;
    d8_sequence_with_mask(&st, D8_MASK_OUT | D8_MASK_WLO); /* missing Ahi */
    tb = !d8_trace_valid(&st, D8_COND_B, &why);
    cls = d8_classify(D8_COND_B, g_d8_refs.wlo, D8_TRACE_VALID, 1, 1, 1,
                      D8_MASK_OUT | D8_MASK_WLO, &g_d8_refs, &m);
    contract = (cls == D8_CLASS_CONTRACT_FAILED);
    ok = tb && strcmp(why, "condition_fill_mask") == 0 && contract;
    d8_check(ok, "d8_fault_13_b_missing_ahi");
    printf("d8_fault=13 b_missing_ahi trace_invalid=%d why=%s class=%s "
           "detected=%d\n", tb, why, d8_class_name(cls), ok);

    d8_sequence_with_mask(&st, D8_MASK_OUT | D8_MASK_AHI); /* missing Wlo */
    tb = !d8_trace_valid(&st, D8_COND_B, &why);
    cls = d8_classify(D8_COND_B, g_d8_refs.ahi, D8_TRACE_VALID, 1, 1, 1,
                      D8_MASK_OUT | D8_MASK_AHI, &g_d8_refs, &m);
    contract = (cls == D8_CLASS_CONTRACT_FAILED);
    ok = tb && strcmp(why, "condition_fill_mask") == 0 && contract;
    d8_check(ok, "d8_fault_13_b_missing_wlo");
    printf("d8_fault=13 b_missing_wlo trace_invalid=%d why=%s class=%s "
           "detected=%d\n", tb, why, d8_class_name(cls), ok);
  }

  /* 14. Condition A (control) with an operand fill: the control must not
   *     reuse an operand. */
  {
    int m = -1;
    d8_class_t cls;
    int tb;
    d8_sequence_with_mask(&st, d8_condition_mask(D8_COND_B)); /* 7 on A */
    tb = !d8_trace_valid(&st, D8_COND_A, &why);
    cls = d8_classify(D8_COND_A, g_d8_refs.both, D8_TRACE_VALID, 1, 1, 1,
                      d8_condition_mask(D8_COND_B), &g_d8_refs, &m);
    ok = tb && strcmp(why, "condition_fill_mask") == 0 &&
         cls == D8_CLASS_CONTRACT_FAILED;
    d8_check(ok, "d8_fault_14_a_extra_fill");
    printf("d8_fault=14 a_extra_fill trace_invalid=%d why=%s class=%s "
           "detected=%d\n", tb, why, d8_class_name(cls), ok);
  }

  /* 15. Registered output-phase order: Ahi before Wlo, and the out_pos fill
   *     after the operand fills, are both rejected. */
  {
    int tb;
    d8_reordered_sequence(&st, 0);
    tb = !d8_trace_valid(&st, D8_COND_B, &why);
    ok = tb && strcmp(why, "fill_order_ahi") == 0;
    d8_check(ok, "d8_fault_15_ahi_before_wlo");
    printf("d8_fault=15 ahi_before_wlo trace_invalid=%d why=%s detected=%d\n",
           tb, why, ok);

    d8_reordered_sequence(&st, 1);
    tb = !d8_trace_valid(&st, D8_COND_B, &why);
    ok = tb && strcmp(why, "fill_order_wlo_before_out") == 0;
    d8_check(ok, "d8_fault_15_out_after_fills");
    printf("d8_fault=15 out_after_fills trace_invalid=%d why=%s detected=%d\n",
           tb, why, ok);
  }

  /* 16. Revision 1 sequence (out_pos fill in the prologue): numerically
   *     identical, but a different instruction sequence and therefore
   *     rejected by the revision-2 order rule. */
  {
    int tb, why_ok, safe;
    d8_legacy_rev1_sequence(&st, D8_COND_A);
    tb = !d8_trace_valid(&st, D8_COND_A, &why);
    why_ok = (strcmp(why, "fill_before_last_mma_read") == 0);
    cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
    safe = d8_array_equal(out, g_d8_refs.safe);
    d8_check(tb && why_ok, "d8_fault_16_legacy_rev1_rejected");
    d8_check(safe, "d8_fault_16_legacy_rev1_numeric_safe");
    printf("d8_fault=16 legacy_rev1 trace_invalid=%d why=%s numeric_safe=%d\n",
           tb, why, safe);
  }
  return g_d8_fail == 0;
}

/* ---------------- revision-3 order / guard faults ---------------- */

static int d8_out_is_safe(const hmx_full_tile_state_t *st) {
  static uint16_t out[T8_CELLS];
  cc_decode_tile((const uint16_t *)(st->vtcm.raw + D8_OUT_OFFSET), out);
  return d8_array_equal(out, g_d8_refs.safe);
}

static void d8_print_oplog(void) {
  printf("d8_oplog count=%d overflow=%d:", g_d8_log.count, g_d8_log.overflow);
  for (int i = 0; i < g_d8_log.count; ++i) {
    printf(" %s(%d,%d,%d)", d8_op_name(g_d8_log.op[i].kind),
           g_d8_log.op[i].a, g_d8_log.op[i].b, g_d8_log.op[i].c);
  }
  printf("\n");
}

static void d8_expect_reject(const hmx_full_tile_state_t *st, int cond,
                             int require_guard, const char *tag,
                             int expect_numeric_safe) {
  const char *why = NULL;
  char safe_tag[128];
  int valid = require_guard ? d8_rev3_valid(st, cond, &why)
                            : d8_trace_valid(st, cond, &why);
  d8_check(!valid, tag);
  snprintf(safe_tag, sizeof(safe_tag), "%s_numeric_safe", tag);
  if (expect_numeric_safe) d8_check(d8_out_is_safe(st), safe_tag);
  printf("d8_order_fault=%s guard=%d rejected=%d why=%s numeric_safe=%d\n",
         tag, require_guard, !valid, why ? why : "?", d8_out_is_safe(st));
  if (valid) d8_print_oplog();
}

static void d8_fault_bias_after_mma(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_tiles(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_bias_pair(st); /* wrong: bias belongs before mxclracc */
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_bias_swapped(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_tiles(st, &g_ops_deploy);
  d8_bias_write(st, 1, HMX_B3_R2_PR_BIAS0_OFFSET);
  d8_bias_write(st, 0, HMX_B3_R2_PR_BIAS1_OFFSET);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_bias_missing(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_tiles(st, &g_ops_deploy);
  d8_bias_write(st, 0, HMX_B3_R2_PR_BIAS0_OFFSET); /* set 1 omitted */
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_no_mxclracc(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_double_mxclracc(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mxclracc(st); /* duplicate */
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_two_mma(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_mma_swapped(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
}

static void d8_fault_no_convert(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  /* no Convert, no Store */
}

static void d8_fault_double_convert(hmx_full_tile_state_t *st) {
  d8_program_order(st, D8_COND_B);
  st->count_convert++;
  d8_log_add(D8_OP_CONVERT, (int)HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET, 0);
}

static void d8_fault_store_before_convert(hmx_full_tile_state_t *st) {
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_fill(st, D8_OUT_OFFSET, 1);
  d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_log_add(D8_OP_STORE, D8_OUT_OFFSET, 0, 0);
  d8_log_add(D8_OP_CONVERT, (int)HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET, 0);
  exec_convert_and_store_at(st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
}

static void d8_fault_drop_op_event(hmx_full_tile_state_t *st) {
  int victim = -1;
  d8_program_order(st, D8_COND_B);
  for (int i = 0; i < g_d8_log.count; ++i) {
    if (g_d8_log.op[i].kind == D8_OP_BIAS) { victim = i; break; }
  }
  if (victim >= 0 && victim + 1 < g_d8_log.count) {
    memmove(&g_d8_log.op[victim], &g_d8_log.op[victim + 1],
            sizeof(d8_op_t) * (size_t)(g_d8_log.count - victim - 1));
    g_d8_log.count--;
  }
}

static void d8_fault_drop_trace_event(hmx_full_tile_state_t *st) {
  d8_program_order(st, D8_COND_B);
  if (st->pr_trace_count > 1) st->pr_trace_count--;
}

static void d8_fault_duplicate_trace_event(hmx_full_tile_state_t *st) {
  d8_program_order(st, D8_COND_B);
  if (st->pr_trace_count < PR_TRACE_CAP) {
    st->pr_trace[st->pr_trace_count] = st->pr_trace[st->pr_trace_count - 1];
    st->pr_trace[st->pr_trace_count].seq = st->pr_trace_count + 1;
    st->pr_trace_count++;
  }
}

/* Revision-3 sequence with one guard/order defect. The defects are applied to
 * a real sequence, not to a metadata array, so the numeric result is produced
 * by the same arithmetic as the clean run. */
static void d8_rev3_fault_sequence(hmx_full_tile_state_t *st, int cond,
                                   int dirty_tail, int skip_init,
                                   int broken_init, int corrupt_at,
                                   int skip_post, int bias2, int false_post) {
  int mask = d8_condition_mask(cond);
  pr_state_init(st);
  d8_log_begin();
  d8_pack_operands(st, &g_ops_deploy);
  if (dirty_tail) {
    memset(st->vtcm.raw + HMX_D8_GUARD_OFFSET, 0x5A, HMX_D8_GUARD_BYTES);
  }
  if (bias2) {
    /* Mode 45's third bias set lives at +10752, inside the D8 guard. */
    pr_event(st, PR_BIAS_WRITE, HMX_D8_GUARD_OFFSET, 256, 1, 2);
    memset(st->vtcm.raw + HMX_D8_GUARD_OFFSET, 0x11, 256);
  }
  if (!skip_init) {
    if (broken_init) {
      /* Claim the init without zeroing: a false self-report. */
      pr_event(st, PR_INIT_WRITE, HMX_D8_GUARD_OFFSET, HMX_D8_GUARD_BYTES, 1,
               7);
      d8_log_add(D8_OP_GUARD_INIT, HMX_D8_GUARD_VECTORS,
                 HMX_D8_GUARD_BYTES, HMX_D8_GUARD_INIT_HVX12);
    } else {
      d8_guard_init(st);
    }
  }
  d8_guard_pre(st);
  d8_mxclracc(st);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_WLO,
                HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  d8_mma_packet(st, HMX_B3_R2_OP_ALO, HMX_B3_R2_OP_WHI,
                HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  if (mask & D8_MASK_OUT) d8_fill(st, D8_OUT_OFFSET, 1);
  if (mask & D8_MASK_WLO) d8_fill(st, HMX_B3_R2_PR_WLO_OFFSET, 2);
  if (mask & D8_MASK_AHI) d8_fill(st, HMX_B3_R2_PR_AHI_OFFSET, 3);
  d8_convert_store(st);
  if (corrupt_at >= 0) {
    st->vtcm.raw[HMX_D8_GUARD_OFFSET + corrupt_at] ^= 0xFF;
  }
  if (!skip_post) d8_guard_post(st);
  if (false_post) {
    for (int i = 0; i < g_d8_log.count; ++i) {
      if (g_d8_log.op[i].kind == D8_OP_GUARD_POST) {
        g_d8_log.op[i].a = g_d8_guard_pre_fnv;
        g_d8_log.op[i].b = HMX_D8_GUARD_STATUS_OK;
        g_d8_log.op[i].c = HMX_D8_GUARD_FIRST_BAD_NONE;
      }
    }
  }
}

static int d8_run_rev3_controls(void) {
  static hmx_full_tile_state_t st;
  const char *why = NULL;
  int valid;

  printf("=== D8 revision-3 order and guard controls ===\n");

  /* 17. Clean revision-3 sequence for every condition: guard ops, numbers and
   *     program order all accepted. */
  for (int c = 0; c < 4; ++c) {
    d8_rev3_sequence(&st, c);
    valid = d8_rev3_valid(&st, c, &why);
    d8_check(valid && d8_out_is_safe(&st), "d8_fault_17_rev3_clean");
    printf("d8_fault=17 condition=%s rev3_valid=%d numeric_safe=%d why=%s\n",
           d8_condition_name(c), valid, d8_out_is_safe(&st), why);
  }

  /* 18. Bias written after MMA #1: the converted numbers are unchanged
   *     because Convert reads the bias, so only the order rule can reject. */
  d8_fault_bias_after_mma(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_18_bias_after_mma", 1);

  /* 19. Bias sets 0 and 1 swapped between the two registered offsets. */
  d8_fault_bias_swapped(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_19_bias_swapped", 0);

  /* 20. One bias set omitted: Convert uses set 0, so omitting set 1 leaves
   *     the safe numbers. The count rule rejects. */
  d8_fault_bias_missing(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_20_bias_missing", 1);

  /* 21. mxclracc omitted. ACC starts zero in the model, so the first MMA adds
   *     to zero exactly as after mxclracc; the numbers are safe. */
  d8_fault_no_mxclracc(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_21_no_mxclracc", 1);

  /* 22. mxclracc duplicated: same numbers, wrong count. */
  d8_fault_double_mxclracc(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_22_double_mxclracc", 1);

  /* 23. MMA #3 omitted: the ALO x WHI contribution is missing. */
  d8_fault_two_mma(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_23_two_mma", 0);

  /* 24. MMA #1 and #2 swapped: same packet count, different order. */
  d8_fault_mma_swapped(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_24_mma_swapped", 0);

  /* 25. Convert omitted: no store event at all. */
  d8_fault_no_convert(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_25_no_convert", 0);

  /* 26. Convert duplicated: one store, two Convert records. */
  d8_fault_double_convert(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_26_double_convert", 1);

  /* 27. Store issued before Convert: the numbers are still produced by the
   *     same fused primitive, so only the recorded order rejects. */
  d8_fault_store_before_convert(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_27_store_before_convert", 1);

  /* 28. Event record only dropped (operation log loses one bias entry). */
  d8_fault_drop_op_event(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_28_drop_op_event", 1);

  /* 29. Access-trace record only dropped. */
  d8_fault_drop_trace_event(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_29_drop_trace_event", 1);

  /* 30. Access-trace record duplicated. */
  d8_fault_duplicate_trace_event(&st);
  d8_expect_reject(&st, D8_COND_B, 0, "d8_fault_30_duplicate_trace_event", 1);

  /* 31. Guard clean: dirty entry is repaired by the registered init. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 1, 0, 0, -1, 0, 0, 0);
  valid = d8_rev3_valid(&st, D8_COND_B, &why);
  d8_check(valid && d8_out_is_safe(&st), "d8_fault_31_guard_init_repairs");
  printf("d8_fault=31 dirty_entry_init validated=%d why=%s\n", valid, why);

  /* 32. Guard init omitted on a dirty entry. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 1, 1, 0, -1, 0, 0, 0);
  d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_32_guard_init_missing", 1);

  /* 33. Guard init claimed but not performed on a dirty entry: the pre digest
   *     exposes the false report. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 1, 0, 1, -1, 0, 0, 0);
  d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_33_guard_broken_init", 1);

  /* 34. One guard byte corrupted during the run (head, middle, tail). The
   *     output is unaffected, so the numeric comparison alone cannot see it. */
  for (int at = 0; at < 3; ++at) {
    int offset = at == 0 ? 0 : (at == 1 ? 700 : HMX_D8_GUARD_BYTES - 1);
    d8_rev3_fault_sequence(&st, D8_COND_B, 0, 0, 0, offset, 0, 0, 0);
    d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_34_guard_corrupt", 1);
  }

  /* 35. Guard post-check omitted. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 0, 0, 0, -1, 1, 0, 0);
  d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_35_guard_check_missing", 1);

  /* 36. Mode 45's third bias set written into +10752 (inside the guard). The
   *     numbers are unaffected; the region rule rejects. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 0, 0, 0, -1, 0, 1, 0);
  d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_36_bias2_in_guard", 1);

  /* 37. Corrupted guard with a clean-looking self-report: the validator
   *     recomputes the digest from memory and rejects. */
  d8_rev3_fault_sequence(&st, D8_COND_B, 0, 0, 0, 0, 0, 0, 1);
  d8_expect_reject(&st, D8_COND_B, 1, "d8_fault_37_guard_false_report", 1);

  /* 38. Guard corrupted but the numeric result matches the safe reference:
   *     the explicit numeric-safe check inside the loop above proves the
   *     point; this case records the same fact with a named fault. */
  d8_rev3_fault_sequence(&st, D8_COND_C, 0, 0, 0, 12, 0, 0, 0);
  d8_expect_reject(&st, D8_COND_C, 1, "d8_fault_38_guard_safe_numeric", 1);

  return g_d8_fail == 0;
}

/* ---------------- replay for the independent Python audit ---------------- */

static int d8_parse_words(const char *s, uint16_t *dst) {
  for (int i = 0; i < T8_CELLS; ++i) {
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 16);
    if (end == s) return 0;
    dst[i] = (uint16_t)(v & 0xFFFFu);
    s = end;
  }
  return 1;
}

static int d8_next_noncomment(char *buf, int cap, FILE *f) {
  while (fgets(buf, cap, f)) {
    char *p = buf;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '#') continue;
    if (p != buf) memmove(buf, p, strlen(p) + 1);
    return 1;
  }
  return 0;
}

static int d8_replay_run(const char *path) {
  FILE *f = fopen(path, "r");
  static char buf[8192];
  static uint16_t obs[T8_CELLS];
  int cases = 0, malformed = 0;
  if (!f) {
    fprintf(stderr, "d8_replay=FAIL cannot open %s\n", path);
    return 2;
  }
  while (d8_next_noncomment(buf, (int)sizeof(buf), f)) {
    int id = 0, cond = 0, trace = 0, evidence = 0, ids = 0, lengths = 0;
    int fill_mask = 0;
    if (sscanf(buf, "case %d %d %d %d %d %d %d", &id, &cond, &trace, &evidence,
               &ids, &lengths, &fill_mask) != 7) {
      malformed++;
      continue;
    }
    if (!d8_next_noncomment(buf, (int)sizeof(buf), f) ||
        strncmp(buf, "O ", 2) != 0 ||
        !d8_parse_words(buf + 2, obs)) {
      malformed++;
      continue;
    }
    printf("replay %d %s\n", id,
           d8_class_name(d8_classify(cond, obs, trace, evidence, ids, lengths,
                                     fill_mask, &g_d8_refs, NULL)));
    cases++;
  }
  fclose(f);
  printf("d8_replay cases=%d malformed=%d\n", cases, malformed);
  return malformed == 0 ? 0 : 2;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--d8-replay") == 0) {
    build_truth_fixture();
    build_fixture_conditions();
    d8_build_references();
    return d8_replay_run(argv[2]);
  }
  if (argc == 2 && strcmp(argv[1], "--d8-sequence") == 0) {
    build_truth_fixture();
    build_fixture_conditions();
    d8_build_references();
    d8_print_sequences();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--d8-sequence-rev3") == 0) {
    build_truth_fixture();
    build_fixture_conditions();
    d8_build_references();
    d8_print_rev3_sequences();
    return 0;
  }
  if (argc != 1) {
    fprintf(stderr, "usage: %s [--d8-replay <file>] [--d8-sequence] "
                    "[--d8-sequence-rev3]\n", argv[0]);
    return 2;
  }

  build_truth_fixture();
  build_fixture_conditions();
  d8_build_references();

  /* Registered fill masks. These four values are the contract; the trace
   * validator and the classifier derive their expectation from the same
   * function, so a change here is a change to the experiment. */
  d8_check(d8_condition_mask(D8_COND_A) == 1, "d8_fill_mask_A");
  d8_check(d8_condition_mask(D8_COND_B) == 7, "d8_fill_mask_B");
  d8_check(d8_condition_mask(D8_COND_C) == 3, "d8_fill_mask_C");
  d8_check(d8_condition_mask(D8_COND_D) == 5, "d8_fill_mask_D");
  printf("d8_fill_mask A=%d B=%d C=%d D=%d\n",
         d8_condition_mask(D8_COND_A), d8_condition_mask(D8_COND_B),
         d8_condition_mask(D8_COND_C), d8_condition_mask(D8_COND_D));

  printf("=== D8 minimal experiment CPU model ===\n");
  printf("d8_condition A=%s B=%s C=%s D=%s\n",
         d8_condition_name(D8_COND_A), d8_condition_name(D8_COND_B),
         d8_condition_name(D8_COND_C), d8_condition_name(D8_COND_D));

  /* Cross-check: the minimal safe sequence is the shipped B3 subcase's first
   * convert, so its 1,024 cells must be identical to that model's out_pos. */
  {
    static hmx_full_tile_state_t full;
    static uint16_t ref[T8_CELLS];
    state_init(&full);
    run_b3_r2_subcase(&full, &g_ops_deploy, HMX_FAULT_NONE);
    cc_decode_tile((const uint16_t *)(full.vtcm.raw + HMX_B3_R2_POS_OFFSET),
                   ref);
    {
      int diff = d8_count_diff(ref, g_d8_refs.safe);
      d8_check(diff == 0, "d8_safe_equals_shipped_b3_pos");
      printf("d8_safe_vs_shipped_b3_pos cells=%d diff=%d\n", T8_CELLS, diff);
    }
  }

  {
    char s_safe[65], s_wlo[65], s_ahi[65], s_both[65];
    int safe_nonzero = 0;
    for (int i = 0; i < T8_CELLS; ++i) {
      if (g_d8_refs.safe[i]) safe_nonzero++;
    }
    d8_sha(g_d8_refs.safe, s_safe);
    d8_sha(g_d8_refs.wlo, s_wlo);
    d8_sha(g_d8_refs.ahi, s_ahi);
    d8_sha(g_d8_refs.both, s_both);
    printf("d8_ref name=safe sha256=%s nonzero=%d\n", s_safe, safe_nonzero);
    printf("d8_ref name=wlo sha256=%s\n", s_wlo);
    printf("d8_ref name=ahi sha256=%s\n", s_ahi);
    printf("d8_ref name=both sha256=%s\n", s_both);
    printf("d8_operand_fnv ahi=%08x alo=%08x whi=%08x wlo=%08x\n",
           d8_tile_fnv(g_ops_deploy.ahi, 0), d8_tile_fnv(g_ops_deploy.alo, 0),
           d8_tile_fnv(g_ops_deploy.whi, 1), d8_tile_fnv(g_ops_deploy.wlo, 1));
    printf("d8_diff safe_vs_wlo=%d safe_vs_ahi=%d safe_vs_both=%d "
           "wlo_vs_ahi=%d wlo_vs_both=%d ahi_vs_both=%d\n",
           d8_count_diff(g_d8_refs.safe, g_d8_refs.wlo),
           d8_count_diff(g_d8_refs.safe, g_d8_refs.ahi),
           d8_count_diff(g_d8_refs.safe, g_d8_refs.both),
           d8_count_diff(g_d8_refs.wlo, g_d8_refs.ahi),
           d8_count_diff(g_d8_refs.wlo, g_d8_refs.both),
           d8_count_diff(g_d8_refs.ahi, g_d8_refs.both));
    /* The four references must be pairwise distinct or the experiment cannot
     * attribute a violation to D8a, D8b or both. */
    d8_check(d8_count_diff(g_d8_refs.safe, g_d8_refs.wlo) > 0,
             "d8_ref_safe_vs_wlo_distinct");
    d8_check(d8_count_diff(g_d8_refs.safe, g_d8_refs.ahi) > 0,
             "d8_ref_safe_vs_ahi_distinct");
    d8_check(d8_count_diff(g_d8_refs.safe, g_d8_refs.both) > 0,
             "d8_ref_safe_vs_both_distinct");
    d8_check(d8_count_diff(g_d8_refs.wlo, g_d8_refs.ahi) > 0,
             "d8_ref_wlo_vs_ahi_distinct");
    d8_check(d8_count_diff(g_d8_refs.wlo, g_d8_refs.both) > 0,
             "d8_ref_wlo_vs_both_distinct");
    d8_check(d8_count_diff(g_d8_refs.ahi, g_d8_refs.both) > 0,
             "d8_ref_ahi_vs_both_distinct");
  }

  {
    char path[512];
    const char *dir = "build/hmx_d8_reference_arrays.txt";
    snprintf(path, sizeof(path), "%s", dir);
    d8_dump(path);
  }

  d8_run_controls();
  d8_run_rev3_controls();

  printf("d8_checks=%d fail=%d\n", g_d8_checks, g_d8_fail);
  if (g_d8_fail != 0) {
    printf("d8_verdict=FAIL\n");
    return 1;
  }
  printf("d8_verdict=D8_CPU_READY\n");
  return 0;
}
