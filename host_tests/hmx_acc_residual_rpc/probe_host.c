// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Standalone Android host-side runner for the research-only HMX RPC.
#include "hexatrain_hmx_acc_residual.h"

#include <remote.h>
#include <rpcmem.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double fp16_to_double(uint16_t bits) {
  const int sign = (bits & 0x8000u) ? -1 : 1;
  const int exp = (bits >> 10) & 31;
  const int frac = bits & 1023;
  if (exp == 31) return frac ? NAN : sign * INFINITY;
  if (!exp) return sign * ldexp((double)frac, -24);
  return sign * ldexp((double)(1024 + frac), exp - 25);
}

int main(int argc, char **argv) {
  int q = 10;
  int mode = 0;
  if (argc >= 2) q = atoi(argv[1]);
  if (argc >= 3) mode = atoi(argv[2]);
  if (q < 6 || q > 10 || mode < 0 || mode > 30) {
    fprintf(stderr, "usage: %s [q=6..10] [mode=0..30]\n", argv[0]);
    return 2;
  }
  // Same CDSP/unsigned-PD policy as the existing HexaTrain HVX device probe.
  struct remote_dsp_capability arch = {CDSP_DOMAIN_ID, 6, 0};
  int status = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch, sizeof(arch));
  printf("capability_status=%d raw_arch=0x%x\n", status, arch.capability);
  if (status || arch.capability != 0x8081) return 3;
  struct remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  status = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                  &unsigned_pd, sizeof(unsigned_pd));
  printf("unsigned_pd_status=%d\n", status);
  if (status) return 4;

  char uri[512];
  int size = snprintf(uri, sizeof(uri), "%s&_dom=cdsp",
                      hexatrain_hmx_acc_residual_URI);
  if (size < 0 || size >= (int)sizeof(uri)) return 5;
  remote_handle64 handle = 0;
  status = hexatrain_hmx_acc_residual_open(uri, &handle);
  printf("open_status=%d\n", status);
  if (status) return 6;

  int *result = (int *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                    RPCMEM_DEFAULT_FLAGS, 4 * sizeof(int));
  int *meta = (int *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                  RPCMEM_DEFAULT_FLAGS, 9 * sizeof(int));
  if (!result || !meta) {
    if (result) rpcmem_free(result);
    if (meta) rpcmem_free(meta);
    hexatrain_hmx_acc_residual_close(handle);
    return 7;
  }
  memset(result, 0, 4 * sizeof(int));
  memset(meta, 0, 9 * sizeof(int));
  /* Identity check first, on the same handle, for every run. */
  int identity[4] = {0, 0, 0, 0};
  const int identity_status =
      hexatrain_hmx_acc_residual_run(handle, identity, 4, meta, 9, q, 30);
  const int identity_ok = (identity_status == 0 && identity[0] == 3 &&
                           identity[1] == q && identity[2] == 30 &&
                           identity[3] == 81);
  printf("identity_status=%d dsp_revision=%d dsp_q=%d dsp_mode=%d dsp_arch=%d\n",
         identity_status, identity[0], identity[1], identity[2], identity[3]);
  printf("dsp_identity=%s\n", identity_ok ? "MATCH" : "MISMATCH");
  if (!identity_ok) {
    rpcmem_free(result);
    rpcmem_free(meta);
    const int close_status = hexatrain_hmx_acc_residual_close(handle);
    printf("close_status=%d\n", close_status);
    return 9;
  }

  if (mode == 30) {
    rpcmem_free(result);
    rpcmem_free(meta);
    const int close_status = hexatrain_hmx_acc_residual_close(handle);
    printf("close_status=%d\n", close_status);
    return 0;
  }

  status = hexatrain_hmx_acc_residual_run(handle, result, 4, meta, 9, q, mode);

  printf("rpc_status=%d magic=0x%x dsp_arch=%d vtcm_bytes=%d "
         "kernel_us=%d power=%d modeA_lock=%d modeB_lock=%d hmx_executed=%d trials=%d\n",
         status, meta[0], meta[1], meta[2], meta[3], meta[4],
         meta[5], meta[6], meta[7], meta[8]);

  const double a = 1.0 + ldexp(1.0, -q);
  const double oracle = a * a;
  printf("q=%d mode=%d a=%.17g oracle=%.17g\n", q, mode, a, oracle);

  /* Per-mode oracle. expected[0] is the first CVT output, expected[1] the
   * second CVT output (unused for single-convert modes).
   * expected_math[0..1] is the mathematically correct value where it is
   * known; observed_reproduced states only that the previous measurement
   * reproduced. Modes with neither expectation are reported as NO_ORACLE. */
  uint16_t expected[2] = {0, 0};
  uint16_t expected_math[2] = {0, 0};
  int have_expected = 0;
  int have_math = 0;
  int single_convert = 0;
  switch (mode) {
    case 0:
      // dynamic feedback: mathematically correct negative residual.
      if (q == 6) { expected[0] = 0x5420; expected[1] = 0xa400; expected_math[1] = 0xa400; }
      else if (q == 7) { expected[0] = 0x5410; expected[1] = 0x9c00; expected_math[1] = 0x9c00; }
      else if (q == 8) { expected[0] = 0x5408; expected[1] = 0x9400; expected_math[1] = 0x9400; }
      else if (q == 9) { expected[0] = 0x5404; expected[1] = 0x8c00; expected_math[1] = 0x8c00; }
      else if (q == 10) { expected[0] = 0x5402; expected[1] = 0x8400; expected_math[1] = 0x8400; }
      have_expected = 1; have_math = 1;
      break;
    case 7: case 8: expected[0] = 0x4400; have_expected = 1; single_convert = 1; break;
    case 12: case 13: case 16: case 19: expected[0] = 0x0000; have_expected = 1; single_convert = 1; break;
    case 14: expected[0] = 0x5420; have_expected = 1; single_convert = 1; break;
    case 15: case 17: expected[0] = 0x53c0; have_expected = 1; single_convert = 1; break;
    case 18: expected[0] = 0x2400; have_math = 1; single_convert = 1; break;
    case 20:
      // 2-CVT: first output is the plain hi, second is the remainder.
      if (q == 6) { expected[0] = 0x5420; expected[1] = 0x2400; expected_math[1] = 0x2400; }
      else if (q == 7) { expected[0] = 0x5410; expected[1] = 0x1c00; expected_math[1] = 0x1c00; }
      else if (q == 8) { expected[0] = 0x5408; expected[1] = 0x1c00; expected_math[1] = 0x1400; }
      else if (q == 9) { expected[0] = 0x5404; expected[1] = 0x1c00; expected_math[1] = 0x0c00; }
      else { expected[0] = 0x5402; expected[1] = 0x1c00; expected_math[1] = 0x0400; }
      have_expected = 1; have_math = 1;
      break;
    case 23:
      // single convert with known -hi bias: the output IS the remainder.
      if (q == 6) { expected[0] = 0x2400; expected_math[0] = 0x2400; }
      else if (q == 7) { expected[0] = 0x1c00; expected_math[0] = 0x1c00; }
      else if (q == 8) { expected[0] = 0x1c00; expected_math[0] = 0x1400; }
      else if (q == 9) { expected[0] = 0x1c00; expected_math[0] = 0x0c00; }
      else { expected[0] = 0x1c00; expected_math[0] = 0x0400; }
      have_expected = 1; have_math = 1; single_convert = 1;
      break;
    case 21: expected[0] = 0x0000; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 22: expected[0] = 0x5400; expected[1] = 0x0000; have_expected = 1; have_math = 1; break;
    case 24: expected[0] = 0x2000; expected_math[0] = 0x2000; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 25: expected[0] = 0x2400; expected_math[0] = 0x2400; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 26: expected[0] = 0x2800; expected_math[0] = 0x2800; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 27: expected[0] = 0x1800; expected_math[0] = 0x1000; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 28: expected[0] = 0x1c00; expected_math[0] = 0x1400; have_expected = 1; have_math = 1; single_convert = 1; break;
    case 29: expected[0] = 0x2000; expected_math[0] = 0x1800; have_expected = 1; have_math = 1; single_convert = 1; break;
    default: break;
  }

  int pass = status == 0 && meta[0] == 0x484d5831 &&
              meta[1] == 81 && meta[4] == 0 && meta[6] == 0 &&
              meta[7] == 1 && meta[8] == 2;
  for (int trial = 0; trial < 2; ++trial) {
    const uint16_t hi_bits = (uint16_t)result[2 * trial];
    const uint16_t residual_bits = (uint16_t)result[2 * trial + 1];
    const double hi = fp16_to_double(hi_bits);
    const double residual = fp16_to_double(residual_bits);
    /* mode 0 uses a negative residual (hi - r); the known-bias modes use a
     * positive remainder (hi + r). */
    const int positive_remainder = (mode == 20 || mode == 23);
    const double reconstructed =
        (positive_remainder ? (hi + residual) : (hi - residual)) / 64.0;
    const double basic = hi / 64.0;
    const double basic_error = fabs(basic - oracle);
    const double recovered_error = fabs(reconstructed - oracle);
    const uint16_t exp_hi = expected[0];
    const uint16_t exp_res = expected[1];
    const int reproduced = have_expected
        ? (hi_bits == exp_hi && (single_convert || residual_bits == exp_res))
        : 0;
    const int math_ok = have_math
        ? (single_convert
               ? (hi_bits == expected_math[0])
               : (hi_bits == expected_math[0] && residual_bits == expected_math[1]))
        : 0;
    printf("extra4=%d hi_bits=0x%04x residual_bits=0x%04x "
           "hi=%.17g residual=%.17g basic=%.17g "
           "reconstructed=%.17g oracle=%.17g "
           "basic_error=%.17g recovered_error=%.17g\n",
           trial, hi_bits, residual_bits, hi, residual,
           basic, reconstructed, oracle, basic_error, recovered_error);
    if (have_expected) {
      printf("expected_hi=0x%04x expected_res=0x%04x single_convert=%d "
             "observed_reproduced=%s\n",
             exp_hi, single_convert ? 0 : exp_res, single_convert,
             reproduced ? "REPRODUCED" : "DIFFERENT");
      if (!reproduced) pass = 0;
    } else {
      printf("oracle=NO_ORACLE\n");
    }
    if (have_math) {
      printf("math_expected_hi=0x%04x math_expected_res=0x%04x "
             "math_match=%s\n",
             expected_math[0], single_convert ? 0 : expected_math[1],
             math_ok ? "MATCH" : "MISMATCH");
    }
    if (mode == 0 && trial == 0) {
      const int exact = (hi_bits == expected[0] && residual_bits == expected[1]);
      printf("fp16_residual_recovery=%s\n", exact ? "SUPPORTED" : "NOT_CONFIRMED");
    }
  }
  rpcmem_free(result);
  rpcmem_free(meta);
  const int close_status = hexatrain_hmx_acc_residual_close(handle);
  printf("close_status=%d\n", close_status);
  if (close_status) pass = 0;
  return pass ? 0 : 8;
}
