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
#include "bias20.h"
#include "hmx_identity.h"

/* Phase D mode range (extra4 nibble sweep + bias readback); must mirror
 * MODE_MAX in probe_dsp.c. */
enum { HMX_PHASE_D_MODE_MIN = 31, HMX_PHASE_D_MODE_MAX = 39 };

/* Per-mode operand a. Must mirror run_one() in probe_dsp.c: the
 * zero-residual controls (21/22) and the mode-37 exact-zero control run
 * a = b = 1.0, the q=8 variants (27/28/29) and the Phase D sweep modes
 * (31..36, 38, 39) run a = b = 1 + 2^-8 regardless of the requested q,
 * every other mode runs a = b = 1 + 2^-q. The printed oracle derives
 * from this, so the zero-control and q=8 modes cannot show a stale
 * q-based product. */
static double mode_operand_a(int q, int mode) {
  if (mode == 21 || mode == 22 || mode == 37) return 1.0;
  if (mode >= 27 && mode <= 29) return 1.0 + ldexp(1.0, -8);
  if ((mode >= HMX_PHASE_D_MODE_MIN && mode <= 36) || mode == 38 ||
      mode == 39) {
    return 1.0 + ldexp(1.0, -8);
  }
  return 1.0 + ldexp(1.0, -q);
}

/* Per-mode convert scale. Must mirror the bias tables in probe_dsp.c:
 * mode 24/27 use scale 32, mode 26/29 use scale 128, every other mode in
 * this experiment converts with scale 64 (including 20, 23, 25, 28 and
 * all Phase D modes 31..38: bias cell scale FP16 0x5400). */
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

/* Phase D research-note per mode; the tokens are defined verbatim in the
 * README "Phase D" mode table (printed as hypothesis_note=<token>). */
static const char *mode_hypothesis_note(int mode) {
  switch (mode) {
    case 31: return "extra4_sweep_n0_baseline";
    case 32: return "extra4_sweep_n1";
    case 33: return "extra4_sweep_n2";
    case 34: return "extra4_sweep_n8";
    case 35: return "extra4_sweep_n15";
    case 36: return "border_control_n3_between_2_and_8";
    case 37: return "exact_zero_control_fallback";
    case 38: return "bias_readback_probe";
    case 39: return "e1_range_discriminator_base64p25";
    default: return NULL;
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
  /* Optional flag: require a nonzero matching build id on the same handle
   * BEFORE any numerical RPC. Scanned across all argv so its position does
   * not disturb the positional q/mode contract (hand-run compatibility). */
  int strict_build_id = 0;
  int positional = 0;
  int arg_index;
  for (arg_index = 1; arg_index < argc; ++arg_index) {
    if (strcmp(argv[arg_index], "-strict-build-id") == 0) {
      strict_build_id = 1;
      continue;
    }
    if (positional == 0) {
      q = atoi(argv[arg_index]);
    } else if (positional == 1) {
      mode = atoi(argv[arg_index]);
    } else {
      fprintf(stderr,
              "usage: %s [q=6..10] [mode=0..39] [-strict-build-id]\n",
              argv[0]);
      return exit_with(2);
    }
    positional++;
  }
  if (q < 6 || q > 10 || mode < 0 || mode > HMX_PHASE_D_MODE_MAX) {
    fprintf(stderr,
            "usage: %s [q=6..10] [mode=0..39] [-strict-build-id]\n",
            argv[0]);
    return exit_with(2);
  }
  /* Phase D modes 31..38 are hard-bound to q=8 on the DSP (HMX_ERR_ARG);
   * mirror that gate host-side before touching any device state. */
  if (mode >= HMX_PHASE_D_MODE_MIN && mode <= HMX_PHASE_D_MODE_MAX &&
      q != 8) {
    fprintf(stderr, "phase D modes 31..38 require q=8 (got q=%d)\n", q);
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

  if (strict_build_id && !hmx_identity_strict_build_id_ok(&id_report)) {
    /* Orchestrator strict mode: all_ok above stays UNAVAILABLE-tolerant on
     * purpose, but -strict-build-id requires a proven nonzero matching
     * build id on the same handle BEFORE any numerical RPC. Fail closed
     * with the same cleanup shape as the all_ok block above. */
    const char *strict_type;
    if (id_report.build_id_status == HMX_BUILD_ID_UNAVAILABLE) {
      strict_type = "unavailable";
    } else if (id_report.build_id_status == HMX_BUILD_ID_MISMATCH) {
      strict_type = "mismatch";
    } else if (id_report.expected_build_id == 0 ||
               id_report.meta_build_id == 0) {
      strict_type = "zero";
    } else {
      strict_type = "status";
    }
    printf("strict_identity_fail=%s\n", strict_type);
    rpcmem_free(result);
    rpcmem_free(meta);
    const int close_status = hexatrain_hmx_acc_residual_close(handle);
    printf("close_status=%d\n", close_status);
    print_verdicts(identity_status, 0, -1, 0, "strict_build_id",
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

  /* Phase D payload provenance (modes 31..38). Every expectation below
   * is double arithmetic from the README "Phase D" formula:
   *   out_bias  = -(base + n * 2^-8)   (base 64.5 = 0xD408, mode 37: 64.0)
   *   residual  = scale * a^2 + out_bias = scaled - |out_bias|
   *   observed  = fp16_round(residual)  — the rounding to 16 output bits
   *     is part of the expected math: PRM 80-N2040-62 AA §2.4 says the
   *     write-convert-state instruction stores the FP16 data to VTCM, and
   *     Ch.5 (after Fig.25) says "HMX only stores FP16 to VTCM, the extra
   *     four bits are internal to the convert state and only used for
   *     feedback" — so a correct extra4 payload still lands as FP16. */
  const int phase_d = (mode >= HMX_PHASE_D_MODE_MIN &&
                       mode <= HMX_PHASE_D_MODE_MAX);
  const int bias_n = phase_d ? phase_d_extra4_nibble(mode) : -1;
  const uint16_t out_bias_fp16 =
      phase_d ? phase_d_out_bias_fp16(mode) : (uint16_t)0;
  const double out_bias_mag =
      phase_d ? -bias20_decode_out_bias(out_bias_fp16, (unsigned)bias_n)
              : 0.0;
  const double expected_residual = scaled - out_bias_mag;
  double hypothesis_out = 0.0;
  if (phase_d) {
    printf("bias_n=%d out_bias_fp16_raw=0x%04x extra4_nibble=%d "
           "decoded_out_bias_magnitude=%.17g ",
           bias_n, (unsigned)out_bias_fp16, bias_n, out_bias_mag);
    if (mode == 38) {
      /* Mode 38 returns the stored bias cells, not a convert output. */
      printf("expected_fp16_raw=N/A expected_value=N/A "
             "expected_returned=bias_readback_cells\n");
    } else {
      printf("expected_fp16_raw=0x%04x expected_value=%.17g\n",
             double_to_fp16(expected_residual), expected_residual);
      /* Device-2026-10-20 battery hypothesis oracle. Measured laws:
       *  - The convert output snaps onto the fine grid of the OUTPUT BIAS
       *    representation: 1 nibble LSB = 2^(e_obias-14) where e_obias is
       *    the FP16 exponent of the out-bias magnitude (64.5 -> e=6 ->
       *    2^-8; scale-32/128 sweep biases give 2^-9/2^-7, matching the
       *    plateau scaling of modes 24..29).
       *  - Positive targets round UP onto that grid (q=8 target +0.25
       *    units -> +1 unit = 0x1c00; ceiling, not nearest, since nearest
       *    would give 0 for 0.25).
       *  - Non-positive fine targets clamp to zero (battery modes 32..36).
       * This is a HYPOTHESIS gate for the regression verdict only; the
       * math verdict below stays independently double-derived. */
      int e_obias = -1;
      double grid = 0.0;
      e_obias = ((out_bias_fp16 >> 10) & 0x1f) - 15;
      grid = ldexp(1.0, e_obias - 14);
      hypothesis_out = (expected_residual > 0.0)
                           ? ceil(expected_residual / grid) * grid
                           : 0.0;
      printf("hypothesis_model=bias_grid_ceil_nonneg e_obias=%d grid=%.17g "
             "hypothesis_value=%.17g hypothesis_fp16_raw=0x%04x\n",
             e_obias, grid, hypothesis_out,
             double_to_fp16(hypothesis_out));
    }
    if (mode == 37) {
      /* Report the fallback fact: the intended n=1 alignment
       * (64.5 + 2^-10) is off the 20-bit grid (0.25 nibble steps), so
       * a=b=1.0 with out_bias=-64.0 gives an exact-zero target instead.
       * probe_bias20_test.c proves unreachability exhaustively. */
      printf("mode37_encoding=fallback_exact_zero n1_alignment=unreachable "
             "encoding=a_eq_b_eq_1.0_out_bias_eq_neg64.0 target=0\n");
    }
    const char *note = mode_hypothesis_note(mode);
    if (note) printf("hypothesis_note=%s\n", note);
  }
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
  uint16_t oracle_selfcheck_math_pin = 0;  /* optional extra pin on
                                            * expected_math[0] (mode 39) */
  int oracle_selfcheck_compare = 0;  /* 0: pin expected_math[0] (modes 24..29,
                                      * math-drift guard). 1: compare the
                                      * hypothesis oracle expected[0] against
                                      * the measured plateau literal (mode 31). */
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
    case 31: case 32: case 33: case 34: case 35: case 36:
      /* Phase D sweep modes: regression expectation uses the battery-derived
       * bias-grid hypothesis (positive ceiling / negative zero clamp),
       * printed as hypothesis_* lines above; the math expectation stays
       * double-derived. mode 31 additionally cross-checks the hypothesis
       * against the previously measured mode-23 plateau literal. */
      expected[0] = double_to_fp16(hypothesis_out);
      expected_math[0] = double_to_fp16(expected_residual);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "hypothesis_bias_grid";
      math_source = "double_derived";
      if (mode == 31) {
        oracle_selfcheck = 1; oracle_selfcheck_literal = 0x1c00;
        oracle_selfcheck_compare = 1;
      }
      break;
    case 39:
      /* E-1 range discriminator (review-corrected bits, PR #14): with
       * the bias base -64.25 (0xD404, nibble 0) the convert target is
       * 0.2509765625 = 64.25 grid units, exactly representable in FP16
       * as 0x3404; the global-ceiling hypothesis predicts 0x3410
       * (65 units = 0.25390625). The two differ, so the measured raw
       * bits decide whether the ceiling law is global or a
       * small-magnitude effect; 0x3404 is itself off the 2^-8 bias
       * grid, so a 0x3404 result falsifies the global-grid reading. */
      expected[0] = double_to_fp16(hypothesis_out);
      expected_math[0] = double_to_fp16(expected_residual);
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "hypothesis_bias_grid";
      math_source = "double_derived";
      oracle_selfcheck = 1; oracle_selfcheck_literal = 0x3410;
      oracle_selfcheck_compare = 1;
      oracle_selfcheck_math_pin = 0x3404;
      break;
    case 37:
      /* Exact-zero control: target 0 exactly; math and hypothesis agree. */
      expected[0] = 0x0000;
      expected_math[0] = 0x0000;
      have_expected = 1; have_math = 1; single_convert = 1;
      expected_src[0] = "double_derived";
      math_source = "double_derived";
      break;
    default: break;
  }
  if (oracle_selfcheck) {
    /* Modes 24..29 pin the double-derived math so it cannot drift silently;
     * mode 31 checks the hypothesis oracle against the independently
     * measured mode-23 plateau literal. Battery 3 finding: comparing
     * expected_math[0] (the math oracle, 0x1400) against the plateau
     * literal (0x1c00) could never pass, because the research result is
     * precisely that pure math does NOT derive the plateau. */
    const uint16_t derived = oracle_selfcheck_compare ? expected[0]
                                                      : expected_math[0];
    oracle_selfcheck_ok = (derived == oracle_selfcheck_literal);
    printf("oracle_selfcheck=%s compare=%s literal=0x%04x derived=0x%04x\n",
           oracle_selfcheck_ok ? "OK" : "MISMATCH",
           oracle_selfcheck_compare ? "hypothesis_vs_measured_plateau"
                                    : "double_derived_math",
           oracle_selfcheck_literal, derived);
  }

  int pass = status == 0 && meta[0] == HMX_IDENTITY_MAGIC &&
             meta[1] == HMX_IDENTITY_EXPECTED_ARCH && meta[4] == 0 &&
             meta[6] == 0 && meta[7] == 1 && meta[8] == 2;
  if (oracle_selfcheck_math_pin) {
    /* Mode 39 extra pin: the reviewer-corrected true-value FP16 bits
     * (0x3404 = 0.2509765625, exactly representable) are fixed here so
     * the discriminator pair cannot drift inside the oracle; the
     * hypothesis side is pinned by oracle_selfcheck_literal. The device
     * result is the experiment, not this check. */
    const int math_pin_ok = (expected_math[0] == oracle_selfcheck_math_pin);
    oracle_selfcheck_ok = oracle_selfcheck_ok && math_pin_ok;
    printf("oracle_selfcheck_math_pin=%s literal=0x%04x derived_math=0x%04x\n",
           math_pin_ok ? "OK" : "MISMATCH",
           oracle_selfcheck_math_pin, expected_math[0]);
  }
  if (oracle_selfcheck && !oracle_selfcheck_ok) pass = 0;

  if (single_convert) {
    /* Output legend: single-CVT modes force the second result slot to 0 on
     * the DSP; it is not a residual and is labelled accordingly. */
    printf("slot2_label=unused_slot2 meaning=unused_dsp_forces_0\n");
  }
  if (mode == 38) {
    printf("slot2_label=zeroed_by_contract meaning=mode38_readback_in_results_0_1\n");
  }

  int observed_any = 0;
  int observed_all = 1;
  int math_any = 0;
  int math_all = 1;
  if (mode == 38) {
    /* Mode 38 returns the stored bias cells, not FP16 convert outputs:
     * results[0] = lo(0) (bytes 0..3 of the row), results[1] = hi(0)
     * (bytes 128..131, PRM Fig.13), results[2..3] = 0 by contract. The
     * observed verdict compares the echo against the CONFIGURED payload;
     * the math verdict re-decodes the echo through bias20.h and checks
     * it against an independent double formula. */
    const uint32_t lo_cell = (uint32_t)result[0];
    const uint32_t hi_cell = (uint32_t)result[1];
    const uint32_t exp_lo = bias20_lo_cell(0x5400u, 0xD408u);
    const uint32_t exp_hi = bias20_hi_cell(0u, (unsigned)bias_n, 0u);
    const int readback_ok = (lo_cell == exp_lo && hi_cell == exp_hi);
    const uint16_t rb_fp16 = (uint16_t)(lo_cell >> 16);
    const unsigned rb_nibble = (unsigned)((hi_cell >> 4) & 0xFu);
    const uint16_t rb_scale = (uint16_t)(lo_cell & 0xFFFFu);
    const unsigned rb_shape = (unsigned)((hi_cell >> 8) & 7u);
    const double rb_out = bias20_decode_out_bias(rb_fp16, rb_nibble);
    const double rb_scale_value = fp16_to_double(rb_scale);
    /* Independent double expectation: -(64.5 + 1*2^-8), scale 64.0,
     * shape 0 — derived from the spec formula, not from bias20.h. */
    const double exp_out = -(64.5 + ldexp(1.0, -8));
    const int math_ok_readback =
        (rb_out == exp_out && rb_scale_value == 64.0 && rb_shape == 0 &&
         (hi_cell & 0xFu) == 0u);
    printf("readback_lo=0x%08x readback_hi=0x%08x "
           "expected_readback_lo=0x%08x expected_readback_hi=0x%08x "
           "readback_match=%s observed_source=configured_payload\n",
           (unsigned)lo_cell, (unsigned)hi_cell, (unsigned)exp_lo,
           (unsigned)exp_hi, readback_ok ? "MATCH" : "MISMATCH");
    printf("readback_decode_out_bias=%.17g readback_decode_scale=%.17g "
           "readback_extra4_nibble=%u readback_shape=%u "
           "math_match=%s oracle_source=double_derived\n",
           rb_out, rb_scale_value, rb_nibble, rb_shape,
           math_ok_readback ? "MATCH" : "MISMATCH");
    observed_any = 1;
    math_any = 1;
    if (!readback_ok) {
      observed_all = 0;
      pass = 0;
    }
    if (!math_ok_readback) math_all = 0;
  } else {
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
    /* "extra4" here is the Rs[8] rnd/feedback-round bit of the convert;
     * Phase D modes force it to 0 in both trials (feedback disabled), so
     * print the actual bit and keep the trial index as a separate field. */
    const int rnd_bit = phase_d ? 0 : trial;
    printf("extra4=%d trial=%d hi_bits=0x%04x %s=0x%04x hi=%.17g %s=%.17g "
           "basic=%.17g scale=%d ",
           rnd_bit, trial, hi_bits, slot2_key, slot2_bits, hi, slot2_value_key,
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
  } /* end generic trial loop (mode != 38) */
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
