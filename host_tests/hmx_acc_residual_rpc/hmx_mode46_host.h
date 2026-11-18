// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// D8 (Mode 46) host-side evidence gate and classifier.
//
// Included by probe_host.c after the mode 45 block so it can use the same
// helpers. It reads the captured d8_slots/d8_meta and prints the d8_* keys.
//
// The classification order is the registered one
// (docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md section 9):
//
//   contract gate -> numeric comparison -> condition-specific label
//
// A contract failure stops before any numeric reading. The word PROVEN is
// never printed: a safe result is one build, one instruction distance, one
// run, and the report says so.

/* Registered reference digests (prereg section 5). */
static const char *const d8_ref_name[4] = {"safe", "wlo", "ahi", "both"};
static const char *const d8_ref_sha[4] = {
    "c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d",
    "f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb",
    "b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc",
    "5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d"};

/* SHA-256 of the 1,024 observed cells serialised as u16 little endian.
 * Implemented here so the host does not depend on an external digest tool. */
typedef struct {
  uint32_t h[8];
  uint64_t len;
  unsigned char buf[64];
  size_t used;
} d8_sha256_t;

static uint32_t d8_rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

static void d8_sha256_block(d8_sha256_t *s, const unsigned char *p) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  uint32_t a, b, c, d, e, f, g, h;
  int i;
  for (i = 0; i < 16; ++i) {
    w[i] = ((uint32_t)p[i * 4] << 24) | ((uint32_t)p[i * 4 + 1] << 16) |
           ((uint32_t)p[i * 4 + 2] << 8) | (uint32_t)p[i * 4 + 3];
  }
  for (i = 16; i < 64; ++i) {
    const uint32_t s0 = d8_rotr(w[i - 15], 7) ^ d8_rotr(w[i - 15], 18) ^
                        (w[i - 15] >> 3);
    const uint32_t s1 = d8_rotr(w[i - 2], 17) ^ d8_rotr(w[i - 2], 19) ^
                        (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3];
  e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];
  for (i = 0; i < 64; ++i) {
    const uint32_t S1 = d8_rotr(e, 6) ^ d8_rotr(e, 11) ^ d8_rotr(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + S1 + ch + k[i] + w[i];
    const uint32_t S0 = d8_rotr(a, 2) ^ d8_rotr(a, 13) ^ d8_rotr(a, 22);
    const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + mj;
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
  s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void d8_sha256_init(d8_sha256_t *s) {
  s->h[0] = 0x6a09e667u; s->h[1] = 0xbb67ae85u;
  s->h[2] = 0x3c6ef372u; s->h[3] = 0xa54ff53au;
  s->h[4] = 0x510e527fu; s->h[5] = 0x9b05688cu;
  s->h[6] = 0x1f83d9abu; s->h[7] = 0x5be0cd19u;
  s->len = 0;
  s->used = 0;
}

static void d8_sha256_update(d8_sha256_t *s, const unsigned char *p, size_t n) {
  s->len += (uint64_t)n;
  while (n > 0) {
    const size_t take = (s->used + n <= 64) ? n : (64 - s->used);
    memcpy(s->buf + s->used, p, take);
    s->used += take;
    p += take;
    n -= take;
    if (s->used == 64) {
      d8_sha256_block(s, s->buf);
      s->used = 0;
    }
  }
}

static void d8_sha256_final(d8_sha256_t *s, char *out_hex) {
  const uint64_t bits = s->len * 8u;
  unsigned char pad = 0x80;
  unsigned char zero = 0x00;
  unsigned char lenbuf[8];
  int i;
  d8_sha256_update(s, &pad, 1);
  while (s->used != 56) d8_sha256_update(s, &zero, 1);
  for (i = 0; i < 8; ++i) lenbuf[i] = (unsigned char)(bits >> (56 - 8 * i));
  d8_sha256_update(s, lenbuf, 8);
  for (i = 0; i < 8; ++i) {
    sprintf(out_hex + i * 8, "%08x", (unsigned)s->h[i]);
  }
  out_hex[64] = '\0';
}

/* SHA-256 of the observed array as u16 little endian, matching the CPU model's
 * serialisation (put_u16_le over T8_CELLS). */
static void d8_observed_sha(const int *slots, char *out_hex) {
  d8_sha256_t s;
  unsigned char row[2];
  int i;
  d8_sha256_init(&s);
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) {
    const unsigned v = (unsigned)slots[i] & 0xFFFFu;
    row[0] = (unsigned char)(v & 0xFFu);
    row[1] = (unsigned char)((v >> 8) & 0xFFu);
    d8_sha256_update(&s, row, 2);
  }
  d8_sha256_final(&s, out_hex);
}

/* Which registered reference the observation matches, or -1. */
static int d8_match_reference(const char *observed_sha) {
  int i;
  for (i = 0; i < 4; ++i) {
    if (strcmp(observed_sha, d8_ref_sha[i]) == 0) return i;
  }
  return -1;
}

/* Registered classification (prereg section 9). Returns a static string; the
 * word PROVEN never appears. */
static const char *d8_classify(int condition, int matched) {
  if (condition == 0) {
    return matched == 0 ? "D8_CONTROL_SAFE" : "D8_CONTROL_ANOMALY";
  }
  if (condition == 1) {
    if (matched == 0) return "D8_NO_VIOLATION_OBSERVED";
    if (matched == 1) return "D8_VIOLATION_WLO_SUSPECT";
    if (matched == 2) return "D8_VIOLATION_AHI_SUSPECT";
    if (matched == 3) return "D8_VIOLATION_BOTH_SUSPECT";
    return "D8_INCONCLUSIVE_NUMERIC";
  }
  if (condition == 2) {
    if (matched == 0) return "D8_NO_VIOLATION_OBSERVED";
    if (matched == 1) return "D8_VIOLATION_WLO_SUSPECT";
    return "D8_INCONCLUSIVE_NUMERIC";
  }
  if (condition == 3) {
    if (matched == 0) return "D8_NO_VIOLATION_OBSERVED";
    if (matched == 2) return "D8_VIOLATION_AHI_SUSPECT";
    return "D8_INCONCLUSIVE_NUMERIC";
  }
  return "D8_INCONCLUSIVE_NUMERIC";
}

/* Host class from the RPC status and the captured metadata (prereg section 8).
 * S0/S1 are pre-execution (no rout data), S2/S3 are contract failures after
 * start, S4 is the documented cleanup exception, S5/S6 are evidence loss. */
static const char *d8_host_class(int rpc_status, const int *meta, int captured) {
  if (!captured) return "D8_HOST_EVIDENCE_FAILED";
  if (rpc_status != 0) {
    /* A non-zero status with no execution flag is a pre-execution failure. */
    if (meta[HMX_D8_META_GENERIC_HMX] != 1) {
      return "D8_CONTRACT_FAILED_PREEXEC";
    }
    return "D8_TRANSPORT_FAILED";
  }
  if (meta[HMX_D8_META_CLEANUP] != 0) return "D8_CONTRACT_FAILED_CLEANUP";
  if (meta[HMX_D8_META_FAILURE] != HMX_D8_FAIL_NONE) return "D8_CONTRACT_FAILED";
  return "D8_CONTRACT_OK";
}

/* Print the whole D8 evidence block and return 1 when the run is a valid
 * scientific outcome (contract ok), 0 otherwise. */
static int d8_report(int rpc_status, const int *slots, const int *meta,
                     int captured, int requested_q, int expected_build_id) {
  const hmx_d8_contract_t c =
      hmx_d8_contract_check(meta, HMX_D8_METADATA_LEN, requested_q,
                            expected_build_id);
  const int results_ok = hmx_d8_results_ok(slots, HMX_D8_RESULT_SLOTS);
  char observed_sha[65];
  int matched;
  const char *label;
  const char *host_class = d8_host_class(rpc_status, meta, captured);
  int i;

  d8_observed_sha(slots, observed_sha);
  matched = d8_match_reference(observed_sha);

  printf("d8_rpc_status=%d\n", rpc_status);
  printf("d8_capture=%d\n", captured);
  printf("d8_host_class=%s\n", host_class);
  printf("d8_contract=%s first_bad=%s\n", c.all_ok ? "PASS" : "FAIL",
         c.first_bad);
  printf("d8_selector condition=%d q_echo=%d\n", c.condition, c.q_echo);
  printf("d8_counts mxclracc=%d mma=%d convert=%d store=%d pack=%d bias=%d\n",
         meta[HMX_D8_META_MXCLRACC], meta[HMX_D8_META_MMA],
         meta[HMX_D8_META_CONVERT], meta[HMX_D8_META_STORE],
         meta[HMX_D8_META_PACK], meta[HMX_D8_META_BIAS]);
  printf("d8_fill mask=%d word=0x%04x form=%d\n",
         meta[HMX_D8_META_FILL] & HMX_D8_FILL_MASK_BITS,
         (meta[HMX_D8_META_FILL] >> HMX_D8_FILL_WORD_SHIFT) &
             HMX_D8_FILL_WORD_BITS,
         (meta[HMX_D8_META_FILL] >> HMX_D8_FILL_FORM_SHIFT) &
             HMX_D8_FILL_FORM_BITS);
  printf("d8_address=0x%08x\n", (unsigned)meta[HMX_D8_META_ADDRESS]);
  printf("d8_fnv ahi=%08x alo=%08x whi=%08x wlo=%08x\n",
         (unsigned)meta[HMX_D8_META_FNV_AHI],
         (unsigned)meta[HMX_D8_META_FNV_ALO],
         (unsigned)meta[HMX_D8_META_FNV_WHI],
         (unsigned)meta[HMX_D8_META_FNV_WLO]);
  printf("d8_guard pre=%08x post=%08x status=0x%08x\n",
         (unsigned)meta[HMX_D8_META_GUARD_PRE],
         (unsigned)meta[HMX_D8_META_GUARD_POST],
         (unsigned)meta[HMX_D8_META_GUARD_STATUS]);
  printf("d8_failure=%d name=%s\n", meta[HMX_D8_META_FAILURE],
         hmx_d8_failure_name(meta[HMX_D8_META_FAILURE]));
  printf("d8_schema=0x%08x expected=0x%08x\n",
         (unsigned)meta[HMX_D8_META_SCHEMA], (unsigned)HMX_D8_SCHEMA);
  printf("d8_readback=0x%08x\n", (unsigned)meta[HMX_D8_META_READBACK]);
  printf("d8_cleanup=%d\n", meta[HMX_D8_META_CLEANUP]);
  printf("d8_build_id dsp=0x%08x host_expected=0x%08x\n",
         (unsigned)meta[HMX_D8_META_BUILD_ID], (unsigned)expected_build_id);
  printf("d8_results_ok=%d\n", results_ok);
  printf("d8_observed_sha256=%s\n", observed_sha);
  printf("d8_matched=%s\n", matched >= 0 ? d8_ref_name[matched] : "none");

  if (!c.all_ok) {
    /* Report every failing field, not only the first, so a single run
     * diagnoses as much as possible. */
    if (!c.meta_len_ok) printf("d8_contract_fail=meta_len\n");
    if (!c.generic_ok) printf("d8_contract_fail=generic\n");
    if (!c.schema_ok) printf("d8_contract_fail=schema\n");
    if (!c.build_id_ok) printf("d8_contract_fail=build_id\n");
    if (!c.selector_ok) printf("d8_contract_fail=selector\n");
    if (!c.mxclracc_ok) printf("d8_contract_fail=mxclracc\n");
    if (!c.mma_ok) printf("d8_contract_fail=mma\n");
    if (!c.convert_ok) printf("d8_contract_fail=convert\n");
    if (!c.store_ok) printf("d8_contract_fail=store\n");
    if (!c.pack_ok) printf("d8_contract_fail=pack\n");
    if (!c.bias_ok) printf("d8_contract_fail=bias\n");
    if (!c.fill_mask_ok) printf("d8_contract_fail=fill_mask\n");
    if (!c.fill_word_ok) printf("d8_contract_fail=fill_word\n");
    if (!c.fill_form_ok) printf("d8_contract_fail=fill_form\n");
    if (!c.address_ok) printf("d8_contract_fail=address\n");
    if (!c.fnv_ok) printf("d8_contract_fail=operand_fnv\n");
    if (!c.guard_ok) printf("d8_contract_fail=guard_digest\n");
    if (!c.guard_status_ok) printf("d8_contract_fail=guard_status\n");
    if (!c.failure_ok) printf("d8_contract_fail=failure_code\n");
    if (!c.readback_ok) printf("d8_contract_fail=readback\n");
    if (!c.cleanup_ok) printf("d8_contract_fail=cleanup\n");
    if (!c.reserved_ok) printf("d8_contract_fail=reserved\n");
    if (!results_ok) printf("d8_contract_fail=results_range\n");
  }

  label = c.all_ok ? d8_classify(c.condition, matched)
                   : "D8_CONTRACT_FAILED";
  printf("d8_label=%s\n", label);
  printf("d8_scope=one_build_one_distance_one_run no_proven_no_full_tile "
         "no_performance\n");

  /* The raw array, so the evidence survives even if the caller loses stdout.
   * 1,024 values on one line is long but greppable and exact. */
  printf("d8_slots=");
  for (i = 0; i < HMX_D8_RESULT_SLOTS; ++i) {
    printf("%s%d", i ? "," : "", slots[i]);
  }
  printf("\n");
  printf("d8_meta=");
  for (i = 0; i < HMX_D8_METADATA_LEN; ++i) {
    printf("%s%d", i ? "," : "", meta[i]);
  }
  printf("\n");

  return c.all_ok ? 1 : 0;
}

