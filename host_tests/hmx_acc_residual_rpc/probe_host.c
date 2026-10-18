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
  if (q < 6 || q > 10 || mode < 0 || mode > 6) {
    fprintf(stderr, "usage: %s [q=6..10] [mode=0..6]\n", argv[0]);
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
  status = hexatrain_hmx_acc_residual_run(handle, result, 4, meta, 9, q, mode);
  printf("rpc_status=%d magic=0x%x dsp_arch=%d vtcm_bytes=%d "
         "kernel_us=%d power=%d modeA_lock=%d modeB_lock=%d hmx_executed=%d trials=%d\n",
         status, meta[0], meta[1], meta[2], meta[3], meta[4],
         meta[5], meta[6], meta[7], meta[8]);

  const double a = 1.0 + ldexp(1.0, -q);
  const double oracle = a * a;
  printf("q=%d mode=%d a=%.17g oracle=%.17g\n", q, mode, a, oracle);
  int pass = status == 0 && meta[0] == 0x484d5831 &&
              meta[1] == 81 && meta[4] == 0 && meta[6] == 0 &&
              meta[7] == 1 && meta[8] == 2;
  // For non-zero modes, do not enforce a fixed residual expectation.
  for (int trial = 0; trial < 2; ++trial) {
    const uint16_t hi_bits = (uint16_t)result[2 * trial];
    const uint16_t residual_bits = (uint16_t)result[2 * trial + 1];
    const double hi = fp16_to_double(hi_bits);
    const double residual = fp16_to_double(residual_bits);
    const double reconstructed = (hi - residual) / 64.0;
    const double basic = hi / 64.0;
    const double basic_error = fabs(basic - oracle);
    const double recovered_error = fabs(reconstructed - oracle);
    printf("extra4=%d hi_bits=0x%04x residual_bits=0x%04x "
           "hi=%.17g residual=%.17g basic=%.17g "
           "reconstructed=%.17g oracle=%.17g "
           "basic_error=%.17g recovered_error=%.17g\n",
           trial, hi_bits, residual_bits, hi, residual,
           basic, reconstructed, oracle, basic_error, recovered_error);
    // Only trial 0 is the precise FP16 feedback hypothesis; trial 1
    // intentionally tests the effect of enabling four extra feedback bits.
    if (!trial) {
      const int exact = hi_bits == 0x5402 && residual_bits == 0x8400;
      printf("fp16_residual_recovery=%s\n", exact ? "SUPPORTED" : "NOT_CONFIRMED");
      if (!exact) pass = 0;
    }
  }
  rpcmem_free(result);
  rpcmem_free(meta);
  const int close_status = hexatrain_hmx_acc_residual_close(handle);
  printf("close_status=%d\n", close_status);
  if (close_status) pass = 0;
  return pass ? 0 : 8;
}
