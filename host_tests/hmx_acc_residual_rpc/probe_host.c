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

#include "fp16_util.h"
#include "hmx_identity.h"

/* Per-mode operand a. Must mirror run_one() in probe_dsp.c: the
 * zero-residual controls (21/22) run a = b = 1.0, the q=8 variants
 * (27/28/29) run a = b = 1 + 2^-8 regardless of the requested q, every
 * other mode runs a = b = 1 + 2^-q. The printed oracle derives from this,
 * so the zero-control and q=8 modes cannot show a stale q-based product. */
static double mode_operand_a(int q, int mode) {
  if (mode == 21 || mode == 22) return 1.0;
  if (mode >= 27 && mode <= 29) return 1.0 + ldexp(1.0, -8);
  return 1.0 + ldexp(1.0, -q);
}

/* Per-mode convert scale. Must mirror the bias tables in probe_dsp.c:
 * mode 24/27 use scale 32, mode 26/29 use scale 128, every other mode in
 * this experiment converts with scale 64 (including 20, 23, 25 and 28). */
static int mode_scale(int mode) {
  switch (mode) {
    case 24:
    case 27:
      return 32;
    case 26:
    case 29:
      return 128;
    default:
      return 64;
  }
}

static const char *dsp_status_name(int status) {
  switch (status) {
    case 0: return "OK";
    case 14001: return "ARG";
    case 14002: return "ALLOC";
    case 14003: return "VTCM";
    case 14004: return "LOCK";
    case 14005: return "UNLOCK";
    case 14006: return "POWER_UP";
    case 14007: return "CLEANUP_UNLOCK2";
    case 14008: return "CLEANUP_RELEASE";
    case 14009: return "CLEANUP_POWER_DOWN";
    default: return "UNKNOWN";
  }
}

/* Non-NULL when the DSP status is one of the cleanup-stage codes
 * (14007 unlock2, 14008 release, 14009 power-down). */
static const char *dsp_cleanup_stage(int status) {
  switch (status) {
    case 14007: return "unlock2";
    case 14008: return "release";
    case 14009: return "power_down";
    default: return NULL;
  }
}

/* Every exit goes through here so the machine-parseable
 * exit_code_description line always documents this run's exit code. */
static int exit_with(int code) {
  const char *meaning;
  switch (code) {
    case 0: meaning = "verdict pass+cleanup ok"; break;
    case 8: meaning = "cleanup/verdict failure after execution"; break;
    case 9: meaning = "identity fail"; break;
    case 7: meaning = "rpcmem"; break;
    case 6: meaning = "open"; break;
    case 5: meaning = "uri"; break;
    case 4: meaning = "pd"; break;
    case 3: meaning = "arch"; break;
    case 2: meaning = "args"; break;
    default: meaning = "unknown"; break;
  }
  printf("exit_code=%d exit_code_description=%s | legend: "
         "0=verdict pass+cleanup ok; 8=cleanup/verdict failure after "
         "execution; 9=identity fail; 7=rpcmem; 6=open; 5=uri; 4=pd; "
         "3=arch; 2=args\n",
         code, meaning);
  return code;
}

/* Independent verdicts, one per line. These are facts of different kinds
 * and must not be conflated with each other or with the exit code.
 * hmx_executed < 0 means "did not run / not applicable", with hmx_reason
 * explaining why. */
static void print_verdicts(int rpc_status, int identity_ok, int hmx_executed,
                           int hmx_trials, const char *hmx_reason,
                           int close_status, const char *observed,
                           const char *math) {
  printf("RPC_OK=%s rpc_status=%d dsp_status=%s\n",
         rpc_status == 0 ? "PASS" : "FAIL", rpc_status,
         dsp_status_name(rpc_status));
  printf("IDENTITY_OK=%s\n", identity_ok ? "PASS" : "FAIL");
  if (hmx_executed >= 0) {
    printf("HMX_EXECUTED=%s hmx_executed=%d trials=%d\n",
           hmx_executed == 1 ? "PASS" : "FAIL", hmx_executed, hmx_trials);
  } else {
    printf("HMX_EXECUTED=N/A reason=%s\n", hmx_reason ? hmx_reason : "not_run");
  }
  const int cleanup_ok = (close_status == 0 && rpc_status == 0);
  printf("RESOURCE_CLEANUP_OK=%s close_status=%d dsp_status=%s",
         cleanup_ok ? "PASS" : "FAIL", close_status,
         dsp_status_name(rpc_status));
  const char *stage = dsp_cleanup_stage(rpc_status);
  if (stage) printf(" dsp_cleanup_error=%d cleanup_stage=%s", rpc_status, stage);
  printf("\n");
  printf("OBSERVED_REGRESSION_MATCH=%s\n", observed);
  printf("MATH_ORACLE_MATCH=%s\n", math);
  printf("DYNAMIC_FEEDBACK_CONFIRMED=NOT_CONFIRMED\n");
}

int main(int argc, char **argv) {
  int q = 10;
  int mode = 0;
  if (argc >= 2) q = atoi(argv[1]);
  if (argc >= 3) mode = atoi(argv[2]);
  if (q < 6 || q > 10 || mode < 0 || mode > HMX_IDENTITY_MODE) {
    fprintf(stderr, "usage: %s [q=6..10] [mode=0..30]\n", argv[0]);
    return exit_with(2);
  }
  // Same CDSP/unsigned-PD policy as the existing HexaTrain HVX device probe.
  struct remote_dsp_capability arch = {CDSP_DOMAIN_ID, 6, 0};
  int status = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch, sizeof(arch));
  printf("capability_status=%d raw_arch=0x%x\n", status, arch.capability);
  if (status || arch.capability != 0x8081) return exit_with(3);
  struct remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  status = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                  &unsigned_pd, sizeof(unsigned_pd));
  printf("unsigned_pd_status=%d\n", status);
  if (status) return exit_with(4);

  char uri[512];
  int size = snprintf(uri, sizeof(uri), "%s&_dom=cdsp",
                      hexatrain_hmx_acc_residual_URI);
  if (size < 0 || size >= (int)sizeof(uri)) return exit_with(5);
  remote_handle64 handle = 0;
  status = hexatrain_hmx_acc_residual_open(uri, &handle);
  printf("open_status=%d\n", status);
  if (status) return exit_with(6);

  int *result = (int *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                    RPCMEM_DEFAULT_FLAGS, 4 * sizeof(int));
  int *meta = (int *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                  RPCMEM_DEFAULT_FLAGS, 9 * sizeof(int));
  if (!result || !meta) {
    if (result) rpcmem_free(result);
    if (meta) rpcmem_free(meta);
    hexatrain_hmx_acc_residual_close(handle);
    return exit_with(7);
  }
  memset(result, 0, 4 * sizeof(int));
  memset(meta, 0, 9 * sizeof(int));
  /* Identity check first, on the same handle, for every run. The pure
   * validation lives in hmx_identity.h so the CPU-only self test
   * (probe_identity_selftest.c) checks the exact same code. */
  int identity[4] = {0, 0, 0, 0};
  const int identity_status = hexatrain_hmx_acc_residual_run(
      handle, identity, 4, meta, 9, q, HMX_IDENTITY_MODE);
  const hmx_identity_report_t id_report = hmx_identity_status(
      identity_status, identity, meta, q, HMX_IDENTITY_MODE,
      HMX_PROBE_BUILD_ID);
  printf("identity_status=%d dsp_revision=%d dsp_q=%d dsp_mode=%d dsp_arch=%d\n",
         identity_status, id_report.revision, id_report.q, id_report.mode,
         id_report.arch);
  printf("identity_meta=0x%x arch=%d hmx_executed=%d trials=%d\n",
         id_report.meta_magic, id_report.meta_arch, id_report.meta_hmx,
         id_report.meta_trials);
  if (id_report.build_id_status == HMX_BUILD_ID_UNAVAILABLE) {
    printf("BUILD_ID=UNAVAILABLE dsp_build_id=0x%08x expected_build_id=0x%08x\n",
           (unsigned)id_report.meta_build_id,
           (unsigned)id_report.expected_build_id);
  } else {
    printf("build_id=0x%08x dsp_build_id=0x%08x build_id_status=%s\n",
           (unsigned)id_report.expected_build_id,
           (unsigned)id_report.meta_build_id,
           hmx_build_id_status_name(id_report.build_id_status));
  }
  printf("dsp_identity=%s\n", id_report.all_ok ? "MATCH" : "MISMATCH");
  if (!id_report.all_ok) {
    /* Fail closed: print every failing flag, release everything, close the
     * handle, and exit 9 before any numerical experiment starts. */
    int flag;
    for (flag = HMX_IDENTITY_FAILURE_STATUS;
         flag <= HMX_IDENTITY_FAILURE_BUILD_ID; ++flag) {
      const hmx_identity_failure_t failed_flag = (hmx_identity_failure_t)flag;
      if (hmx_identity_flag_failed(&id_report, failed_flag)) {
        printf("identity_fail=%s\n", hmx_identity_failure_name(failed_flag));
      }
    }
    rpcmem_free(result);
    rpcmem_free(meta);
    const int close_status = hexatrain_hmx_acc_residual_close(handle);
    printf("close_status=%d\n", close_status);
    print_verdicts(identity_status, 0, -1, 0, "identity_fail_closed",
                   close_status, "NOT_STARTED", "NOT_STARTED");
    return exit_with(9);
  }

  if (mode == HMX_IDENTITY_MODE) {
    rpcmem_free(result);
    rpcmem_free(meta);
    const int close_status = hexatrain_hmx_acc_residual_close(handle);
    printf("close_status=%d\n", close_status);
    /* A failing close is a failure, never a success. */
    print_verdicts(identity_status, 1, -1, 0, "identity_mode", close_status,
                   "NOT_APPLICABLE", "NOT_APPLICABLE");
    if (close_status) return exit_with(8);
    return exit_with(0);
  }

  status = hexatrain_hmx_acc_residual_run(handle, result, 4, meta, 9, q, mode);

  printf("rpc_status=%d magic=0x%x dsp_arch=%d vtcm_bytes=%d "
         "kernel_us=%d power=%d modeA_lock=%d modeB_lock=%d hmx_executed=%d trials=%d\n",
         status, meta[0], meta[1], meta[2], meta[3], meta[4],
         meta[5], meta[6], meta[7], meta[8]);
  const int hmx_executed_flag = meta[7];
  const int trials_flag = meta[8];

  const double a = mode_operand_a(q, mode);
  const int scale = mode_scale(mode);
  const double oracle = a * a;
  printf("q=%d mode=%d a=%.17g scale=%d oracle=%.17g\n", q, mode, a, scale,
         oracle);
  if (mode == 21 || mode == 22) {
    /* Zero-residual controls: the DSP forces a = b = 1.0, so the operand
     * is independent of q; mode 22 additionally requires q == 6 (fixed
     * bias values in the DSP argument check). */
    printf("zero_residual_control=1 operand=a_eq_b_eq_1.0 q_binding=%s\n",
           mode == 22 ? "6" : "none");
  }

  /* Per-mode oracle, derived from double arithmetic so that no constant
   * can drift out of sync with the experiment definition.
   *
   * expected[0]/expected[1]  : the previously observed raw FP16 outputs
   *                           (reproduction check only).
   * expected_math[0]/[1]     : the mathematically correct raw FP16 values.
   * expected_src[]           : "literal" when the expected value is a
   *                           hardcoded previously measured constant,
   *                           "double_derived" when this file computes it.
   * have_math==0             : the mode has no mathematical expectation.
   * single_convert==1        : result[0] is the CVT output itself, and the
   *                           second slot is unused (not a residual).
   */
  const double product = oracle;                     /* a*b, exact in double */
  const double scaled = product * (double)scale;     /* per-mode scale */
  /* The main value is the FP16 rounding of the scaled product. */
  const double hi_value = fp16_to_double(double_to_fp16(scaled));
  const double remainder = scaled - hi_value;  /* exact positive remainder */
  uint16_t expected[2] = {0, 0};              /* previously observed values */
  uint16_t expected_math[2] = {0, 0};         /* mathematically correct values */
  int have_expected = 0;
  int have_math = 0;
  int single_convert = 0;
  const char *expected_src[2] = {"n/a", "n/a"};
  const char *math_source = "none";
  int oracle_selfcheck = 0;          /* 1 = literal cross-check registered */
  int oracle_selfcheck_ok = 1;
  uint16_t oracle_selfcheck_literal = 0;
  switch (mode) {
    case 0:
      // dynamic feedback: mathematically correct negative residual.
      expected[0] = double_to_fp16(hi_value);
      expected_math[0] = expected[0];
      expected_math[1] = double_to_fp16(hi_value - scaled);  /* negative */
      expected[1] = expected_math[1];
      expected_src[0] = "double_derived";
      expected_src[1] = "double_derived";
      have_expected = 1; have_math = 1;
      math_source = "double_derived";
      break;
    case 7: case 8:
      expected[0] = 0x4400; have_expected = 1; single_convert = 1;
      expected_src[0] = "literal";
      break;
    case 12: case 13: case 16: case 19:
      expected[0] = 0x0000; have_expected = 1; single_convert = 1;
      expected_src[0] = "literal";
      break;
    case 14:
      expected[0] = 0x5420; have_expected = 1; single_convert = 1;
      expected_src[0] = "literal";
      break;
    case 15: case 17:
      expected[0] = 0x53c0; have_expected = 1; single_convert = 1;
      expected_src[0] = "literal";
      break;
    case 18:
      /* Hand-verified literal (bias -66, scale 64): NOT derived here. */
      expected[0] = 0x2400; expected_math[0] = 0x2400;
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "literal";
      break;
    case 20:
      // 2-CVT: first output is the plain hi, second is the remainder.
      expected[0] = double_to_fp16(hi_value);
      expected_math[0] = expected[0];
      expected_math[1] = double_to_fp16(remainder);   /* positive */
      /* observed: one fine ULP of the 20-bit format (previously measured) */
      expected[1] = (q == 6) ? 0x2400 : 0x1c00;
      expected_src[0] = "double_derived";
      expected_src[1] = "literal";
      have_expected = 1; have_math = 1;
      math_source = "double_derived";
      break;
    case 23:
      // single convert with known -hi bias: the output IS the remainder.
      expected_math[0] = double_to_fp16(remainder);
      expected[0] = (q == 6) ? 0x2400 : 0x1c00;  /* observed plateau */
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      break;
    case 21:
      /* Zero-residual control: a = b = 1.0, scale 64, bias -64. Literal
       * hand-verified values (not derived). */
      expected[0] = 0x0000; expected_math[0] = 0x0000;
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "literal";
      break;
    case 22:
      /* Zero-residual 2-CVT: a = b = 1.0, hi = 64.0, residual = 0.
       * Literal hand-verified values (not derived). */
      expected[0] = 0x5400; expected_math[0] = 0x5400;
      expected[1] = 0x0000; expected_math[1] = 0x0000;
      have_expected = 1; have_math = 1;
      expected_src[0] = "literal";
      expected_src[1] = "literal";
      math_source = "literal";
      break;
    case 24:
      // scale sweep scale=32, q=6: observed regression value (literal).
      expected[0] = 0x2000;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x2000;
      break;
    case 25:
      // scale sweep scale=64, q=6: observed regression value (literal).
      expected[0] = 0x2400;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x2400;
      break;
    case 26:
      // scale sweep scale=128, q=6: observed regression value (literal).
      expected[0] = 0x2800;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x2800;
      break;
    case 27:
      // q=8 scale=32: observed plateau (literal), math double-derived.
      expected[0] = 0x1800;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x1000;
      break;
    case 28:
      // q=8 scale=64: observed plateau (literal), math double-derived.
      expected[0] = 0x1c00;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x1400;
      break;
    case 29:
      // q=8 scale=128: observed plateau (literal), math double-derived.
      expected[0] = 0x2000;
      expected_math[0] = double_to_fp16(scaled - hi_value);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "literal";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x1800;
      break;
    default: break;
  }
  if (oracle_selfcheck) {
    oracle_selfcheck_ok = (expected_math[0] == oracle_selfcheck_literal);
    printf("oracle_selfcheck=%s literal_math_hi=0x%04x derived_math_hi=0x%04x\n",
           oracle_selfcheck_ok ? "OK" : "MISMATCH",
           oracle_selfcheck_literal, expected_math[0]);
  }

  int pass = status == 0 && meta[0] == HMX_IDENTITY_MAGIC &&
             meta[1] == HMX_IDENTITY_EXPECTED_ARCH && meta[4] == 0 &&
             meta[6] == 0 && meta[7] == 1 && meta[8] == 2;
  if (oracle_selfcheck && !oracle_selfcheck_ok) pass = 0;

  if (single_convert) {
    /* Output legend: single-CVT modes force the second result slot to 0 on
     * the DSP; it is not a residual and is labelled accordingly. */
    printf("slot2_label=unused_slot2 meaning=unused_dsp_forces_0\n");
  }

  int observed_any = 0;
  int observed_all = 1;
  int math_any = 0;
  int math_all = 1;
  for (int trial = 0; trial < 2; ++trial) {
    const uint16_t hi_bits = (uint16_t)result[2 * trial];
    const uint16_t slot2_bits = (uint16_t)result[2 * trial + 1];
    const double hi = fp16_to_double(hi_bits);
    const double slot2_value = fp16_to_double(slot2_bits);
    /* Reconstruction depends on what the slot actually holds:
     *  - mode 0  : slot0=hi, slot1=negative residual -> (hi - r)/scale
     *  - mode 2  : slot0=hi, slot1=the +4 output-bias-only convert ->
     *              (hi - r)/scale (measures the bias injection; it is
     *              expected NOT to equal the operand)
     *  - mode 20 : slot0=hi, slot1=positive remainder -> (hi + r)/scale
     *  - mode 23 : slot0=remainder only (DSP converts round(scale*x -
     *              known_hi), a POSITIVE remainder: 0x2400 = +2^-6 for
     *              q=6), slot1 unused -> (known_hi + slot0)/scale. The
     *              sign was verified against the DSP bias tables; a minus
     *              sign here would be wrong. known_hi is fp16(scale*x),
     *              which equals the DSP's per-q -hi bias table exactly.
     *  - other single-convert modes: slot 0 is an unrelated value -> no
     *    reconstruction is defined, printed as reconstruction=N/A. */
    double reconstructed = 0.0;
    int reconstruction_valid = 0;
    if (mode == 0 || mode == 2) {
      reconstructed = (hi - slot2_value) / (double)scale;
      reconstruction_valid = 1;
    } else if (mode == 20) {
      reconstructed = (hi + slot2_value) / (double)scale;
      reconstruction_valid = 1;
    } else if (mode == 23) {
      reconstructed = (hi_value + hi) / (double)scale;
      reconstruction_valid = 1;
    }
    const double basic = hi / (double)scale;   /* per-mode scale, not /64 */
    const double basic_error = fabs(basic - oracle);
    const double recovered_error = fabs(reconstructed - oracle);
    const uint16_t exp_hi = expected[0];
    const uint16_t exp_res = expected[1];
    const int reproduced = have_expected
        ? (hi_bits == exp_hi && (single_convert || slot2_bits == exp_res))
        : 0;
    const int math_ok = have_math
        ? (single_convert
               ? (hi_bits == expected_math[0])
               : (hi_bits == expected_math[0] && slot2_bits == expected_math[1]))
        : 0;
    const char *slot2_key = single_convert ? "unused_slot2" : "residual_bits";
    const char *slot2_value_key = single_convert ? "unused_val" : "residual";
    printf("extra4=%d hi_bits=0x%04x %s=0x%04x hi=%.17g %s=%.17g "
           "basic=%.17g scale=%d ",
           trial, hi_bits, slot2_key, slot2_bits, hi, slot2_value_key,
           slot2_value, basic, scale);
    if (reconstruction_valid) {
      printf("reconstruction=%.17g reconstruction_valid=1 "
             "recovered_error=%.17g ",
             reconstructed, recovered_error);
    } else {
      printf("reconstruction=N/A reconstruction_valid=0 recovered_error=N/A ");
    }
    printf("oracle=%.17g basic_error=%.17g\n", oracle, basic_error);
    if (have_expected) {
      observed_any = 1;
      if (!reproduced) {
        observed_all = 0;
        pass = 0;
      }
      printf("expected_hi=0x%04x expected_res=0x%04x single_convert=%d "
             "observed_reproduced=%s observed_source_hi=%s observed_source_res=%s\n",
             exp_hi, single_convert ? 0 : exp_res, single_convert,
             reproduced ? "REPRODUCED" : "DIFFERENT",
             expected_src[0], single_convert ? "unused" : expected_src[1]);
    } else {
      printf("oracle=NO_ORACLE\n");
    }
    if (have_math) {
      math_any = 1;
      if (!math_ok) math_all = 0;
      printf("math_expected_hi=0x%04x math_expected_res=0x%04x "
             "math_match=%s oracle_source=%s\n",
             expected_math[0], single_convert ? 0 : expected_math[1],
             math_ok ? "MATCH" : "MISMATCH", math_source);
    }
    if (mode == 0 && trial == 0) {
      const int exact = (hi_bits == expected[0] && slot2_bits == expected[1]);
      printf("fp16_residual_recovery=%s\n", exact ? "SUPPORTED" : "NOT_CONFIRMED");
    }
  }
  const char *observed_verdict =
      !observed_any ? "NO_ORACLE" : (observed_all ? "PASS" : "FAIL");
  const char *math_verdict =
      !math_any ? "NO_ORACLE" : (math_all ? "PASS" : "FAIL");

  rpcmem_free(result);
  rpcmem_free(meta);
  const int close_status = hexatrain_hmx_acc_residual_close(handle);
  printf("close_status=%d\n", close_status);
  if (close_status) pass = 0;
  print_verdicts(status, 1, hmx_executed_flag, trials_flag, NULL, close_status,
                 observed_verdict, math_verdict);
  return exit_with(pass ? 0 : 8);
}
