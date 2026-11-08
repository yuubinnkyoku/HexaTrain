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
 *   3. runs the D8 access-trace validator and ten fault/control cases;
 *   4. exposes a classifier replay mode for the independent Python audit.
 *
 * CPU only: no device, no DSP, no SDK. The references are outcomes of a
 * hypothetical read/write race, not a proposed program order for the device.
 */
#define main hmx_b3_r2_full_tile_sim_original_main
#include "hmx_b3_r2_full_tile_sim.c"
#undef main

#define D8_OUT_OFFSET HMX_B3_R2_PR_POS_OFFSET /* +8192, not an operand region */
#define D8_COND_A 0 /* no operand reuse (control) */
#define D8_COND_B 1 /* Candidate B: both Wlo and Ahi reused */
#define D8_COND_C 2 /* Wlo only */
#define D8_COND_D 3 /* Ahi only */
#define D8_TRACE_NOT_PROVIDED 0
#define D8_TRACE_VALID 1
#define D8_TRACE_INVALID 2

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

/* ---------------- sequence primitives ---------------- */

static void d8_pack_operands(hmx_full_tile_state_t *st,
                             const b3_r2_ops_t *ops) {
  pr_pack(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_OP_AHI, ops->ahi, 0);
  pr_pack(st, HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_OP_ALO, ops->alo, 0);
  pr_pack(st, HMX_B3_R2_PR_WHI_OFFSET, HMX_B3_R2_OP_WHI, ops->whi, 1);
  pr_pack(st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_OP_WLO, ops->wlo, 1);
  pr_bias_init(st);
}

static void d8_mma_sequence(hmx_full_tile_state_t *st) {
  exec_mxclracc(st);
  st->act_id = HMX_B3_R2_OP_AHI;
  st->wgt_id = HMX_B3_R2_OP_WHI;
  exec_mma_packet_at(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
  st->act_id = HMX_B3_R2_OP_AHI;
  st->wgt_id = HMX_B3_R2_OP_WLO;
  exec_mma_packet_at(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET);
  st->act_id = HMX_B3_R2_OP_ALO;
  st->wgt_id = HMX_B3_R2_OP_WHI;
  exec_mma_packet_at(st, HMX_B3_R2_PR_ALO_OFFSET, HMX_B3_R2_PR_WHI_OFFSET);
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

/* Program-order sequence for the condition: MMAs first, then the scheduled
 * reuse fills, then Convert #1. Only the fills of the reuse conditions are
 * scheduled; the full Candidate B also stores Convert #2 into Wlo and the
 * recon into Ahi, but those writes are later than these fills, so the fills
 * are the tightest reuse point. */
static void d8_program_order(hmx_full_tile_state_t *st, int cond) {
  pr_state_init(st);
  d8_pack_operands(st, &g_ops_deploy);
  d8_mma_sequence(st);
  pr_fill(st, D8_OUT_OFFSET, HMX_B3_R2_TILE_BYTES, 1); /* out_pos sentinel */
  if (cond == D8_COND_B || cond == D8_COND_C) {
    pr_fill(st, HMX_B3_R2_PR_WLO_OFFSET, HMX_B3_R2_TILE_BYTES, 2);
  }
  if (cond == D8_COND_B || cond == D8_COND_D) {
    pr_fill(st, HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_TILE_BYTES, 3);
  }
  exec_convert_and_store_at(st, HMX_B3_R2_RS_CVT1, D8_OUT_OFFSET,
                            HMX_B3_R2_PR_BIAS0_OFFSET,
                            HMX_B3_R2_PR_BIAS1_OFFSET);
}

/* ---------------- D8 access-trace validator ---------------- */

/* D8-specific version of the CB-FIX-1 rule. It derives the last MMA read of
 * each operand from the recorded events and rejects any later-overlapping
 * write that lands before that read. It also checks the condition's expected
 * fills and event counts, so a truncated or padded trace fails closed.
 * Returns 1 = valid, 0 = invalid; `why` receives a short reason. */
static int d8_trace_valid(const hmx_full_tile_state_t *st, int cond,
                          const char **why) {
  const int ids[4] = {HMX_B3_R2_OP_AHI, HMX_B3_R2_OP_ALO,
                      HMX_B3_R2_OP_WHI, HMX_B3_R2_OP_WLO};
  const int offsets[4] = {HMX_B3_R2_PR_AHI_OFFSET, HMX_B3_R2_PR_ALO_OFFSET,
                          HMX_B3_R2_PR_WHI_OFFSET, HMX_B3_R2_PR_WLO_OFFSET};
  const int reads_expected[4] = {2, 1, 2, 1};
  int pack[4] = {0}, last[4] = {0}, reads[4] = {0};
  int sentinel = 0, hmx_stores = 0, hvx_stores = 0, bias_writes = 0;
  int out_fill = 0, wlo_fill = 0, ahi_fill = 0;
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
          }
          break;
        }
      }
      if (j == 4) { if (why) *why = "unknown_operand_object"; return 0; }
    } else if (e->kind == PR_SENTINEL) {
      sentinel++;
      if (e->offset == D8_OUT_OFFSET && e->size == HMX_B3_R2_TILE_BYTES) {
        out_fill++;
      } else if (e->offset == HMX_B3_R2_PR_WLO_OFFSET &&
                 e->size == HMX_B3_R2_TILE_BYTES) {
        wlo_fill++;
      } else if (e->offset == HMX_B3_R2_PR_AHI_OFFSET &&
                 e->size == HMX_B3_R2_TILE_BYTES) {
        ahi_fill++;
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
  {
    int want_wlo = (cond == D8_COND_B || cond == D8_COND_C);
    int want_ahi = (cond == D8_COND_B || cond == D8_COND_D);
    if (wlo_fill != want_wlo || ahi_fill != want_ahi) {
      if (why) *why = "condition_fill_mismatch";
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
 *   result; an unmatched numeric result is never read as "no violation". */
static d8_class_t d8_classify(int cond, const uint16_t *obs, int trace_status,
                              int evidence_ok, int ids_ok, int lengths_ok,
                              const d8_refs_t *r, int *matched_out) {
  int matched;
  if (!evidence_ok || !ids_ok || !lengths_ok) return D8_CLASS_CONTRACT_FAILED;
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

  /* 6. Normal post-last-use reuse: every condition's program order is valid,
   *    and the numerical result equals the safe reference. */
  for (int c = 0; c < 4; ++c) {
    int valid;
    d8_program_order(&st, c);
    valid = d8_trace_valid(&st, c, &why);
    cc_decode_tile((const uint16_t *)(st.vtcm.raw + D8_OUT_OFFSET), out);
    d8_check(valid && d8_array_equal(out, g_d8_refs.safe),
             "d8_fault_06_normal_order");
    printf("d8_fault=06 condition=%s trace_valid=%d numeric_safe=%d\n",
           d8_condition_name(c), valid, d8_array_equal(out, g_d8_refs.safe));
  }

  /* 7. Numeric safe + trace invalid: must not be read as no violation. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_C, g_d8_refs.safe, D8_TRACE_INVALID,
                               1, 1, 1, &g_d8_refs, &m);
    ok = (c == D8_CLASS_INCONCLUSIVE_TRACE);
    d8_check(ok, "d8_fault_07_safe_numeric_bad_trace");
    printf("d8_fault=07 safe_numeric_bad_trace class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 8. Corrupt numeric + trace valid: must be reported, not excused. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_C, g_d8_refs.wlo, D8_TRACE_VALID,
                               1, 1, 1, &g_d8_refs, &m);
    ok = (c == D8_CLASS_VIOLATION_WLO);
    d8_check(ok, "d8_fault_08_corrupt_numeric_good_trace");
    printf("d8_fault=08 corrupt_numeric_good_trace class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 9. Missing evidence array. */
  {
    int m = -1;
    d8_class_t c = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                               0, 1, 1, &g_d8_refs, &m);
    ok = (c == D8_CLASS_CONTRACT_FAILED);
    d8_check(ok, "d8_fault_09_missing_evidence");
    printf("d8_fault=09 missing_evidence class=%s detected=%d\n",
           d8_class_name(c), ok);
  }

  /* 10. Length / identifier mismatch. */
  {
    int m = -1;
    d8_class_t c1 = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                                1, 0, 1, &g_d8_refs, &m);
    d8_class_t c2 = d8_classify(D8_COND_A, g_d8_refs.safe, D8_TRACE_VALID,
                                1, 1, 0, &g_d8_refs, &m);
    ok = (c1 == D8_CLASS_CONTRACT_FAILED && c2 == D8_CLASS_CONTRACT_FAILED);
    d8_check(ok, "d8_fault_10_length_id_mismatch");
    printf("d8_fault=10 length_id_mismatch bad_id=%s bad_len=%s detected=%d\n",
           d8_class_name(c1), d8_class_name(c2), ok);
  }
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
    if (sscanf(buf, "case %d %d %d %d %d %d", &id, &cond, &trace, &evidence,
               &ids, &lengths) != 6) {
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
                                     &g_d8_refs, NULL)));
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
  if (argc != 1) {
    fprintf(stderr, "usage: %s [--d8-replay <file>]\n", argv[0]);
    return 2;
  }

  build_truth_fixture();
  build_fixture_conditions();
  d8_build_references();

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

  printf("d8_checks=%d fail=%d\n", g_d8_checks, g_d8_fail);
  if (g_d8_fail != 0) {
    printf("d8_verdict=FAIL\n");
    return 1;
  }
  printf("d8_verdict=D8_CPU_READY\n");
  return 0;
}
