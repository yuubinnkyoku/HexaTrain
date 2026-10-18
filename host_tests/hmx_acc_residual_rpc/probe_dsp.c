// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// V81 HMX convert-feedback EXPERIMENT, not an FP32 GEMM implementation.
// Derived from the public V81 HMX instruction specification (80-N2040-62 AA),
// not copied from any third-party kernel.
//
// Test x = (1 + 2^-10)^2 = 1 + 2^-9 + 2^-20.
// FP16 x FP16 -> HMX ACC; convert with scale 64:
//   hi = round_fp16(64*x)
//   residual = round_fp16(-64*x + min(+65504, previous_convert))
// Then compare whether (hi - residual)/64 recovers x.
//
// Two trials: first-convert Rs[8]=0 and Rs[8]=1 (extra feedback bits).
// No assertion that the undocumented intermediate rounding preserves residual.
#include "hexatrain_hmx_acc_residual.h"

#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__HEXAGON_ARCH__) || __HEXAGON_ARCH__ != 81
#error This probe must be built for the V81 DSP target.
#endif

enum {
  TILE_BYTES = 2048,
  VTCM_BYTES = 12288,
  BIAS_BYTES = 256,
  HMX_MAGIC = 0x484d5831,
  HMX_ERR_ARG = 14001,
  HMX_ERR_ALLOC = 14002,
  HMX_ERR_VTCM = 14003,
  HMX_ERR_LOCK = 14004,
  HMX_ERR_UNLOCK = 14005,
  HMX_ERR_POWER = 14006,
  PROBE_REVISION = 2,
};

typedef struct probe_context {
  uint32_t tag;
} probe_context_t;

static void init_bias(uint32_t *dst, uint16_t scale_bits,
                      uint16_t output_bias_bits, unsigned shape) {
  // V81 Table 4: 32 low words then 32 high words, not interleaved uint64.
  // Low[15:0] scale (FP16), low[31:16] output bias (FP16);
  // high[10:8] shape, all optional extra bits/input bias left at zero.
  memset(dst, 0, BIAS_BYTES);
  for (int ch = 0; ch < 32; ++ch) {
    dst[ch] = (uint32_t)scale_bits | ((uint32_t)output_bias_bits << 16);
    dst[32 + ch] = (shape & 7u) << 8;
  }
}

static void run_one(uint8_t *base, unsigned extra_four_bits, int q, int mode,
                    int *hi_bits, int *residual_bits) {
  uint16_t *act = (uint16_t *)(base + 0 * TILE_BYTES);
  uint16_t *wgt = (uint16_t *)(base + 1 * TILE_BYTES);
  uint16_t *hi = (uint16_t *)(base + 2 * TILE_BYTES);
  uint16_t *residual = (uint16_t *)(base + 3 * TILE_BYTES);
  uint32_t *bias0 = (uint32_t *)(base + 4 * TILE_BYTES);
  uint32_t *bias1 = (uint32_t *)((uint8_t *)bias0 + BIAS_BYTES);

  // a = 1 + 2^-q as exact FP16 (q in [6,10]).
  uint16_t a_bits;
  if (mode == 21 || mode == 22) {
    a_bits = 0x3c00u;  // 1.0 for zero-residual control
  } else {
    a_bits = (uint16_t)(0x3c00u + (1u << (10 - q)));
  }
  // (spatial=0, channel=0) and (channel=0, out=0) both map to tile[0].
  memset(act, 0, TILE_BYTES);
  memset(wgt, 0, TILE_BYTES);
  memset(hi, 0xa5, TILE_BYTES);
  memset(residual, 0xa5, TILE_BYTES);
  act[0] = a_bits;
  wgt[0] = a_bits;

  // 64.0 FP16 = 0x5400, +65504 FP16 = 0x7bff.
  // Bias set 0: 64*x. Bias set 1: -64*x + min(65504, feedback).
  init_bias(bias0, 0x5400, 0, 0);           // shape(x) = x
  // bias1 configuration depends on mode.
  if (mode == 0) {
    init_bias(bias1, 0x5400, 0x7bff, 4);   // original feedback
  } else if (mode == 1) {
    init_bias(bias1, 0x5400, 0, 4);        // shape-only, no output bias
  } else if (mode == 2) {
    init_bias(bias1, 0, 0x4400, 0);        // output-bias-only: +4
  } else if (mode == 3) {
    init_bias(bias1, 0, 0x7bff, 0);        // feedback-only: 65504, shape 0
  } else if (mode == 4) {
    init_bias(bias1, 0x5400, 0x5420, 4);  // constant bias: output_bias=66
  } else if (mode == 5) {
    init_bias(bias0, 0, 0x4400, 0);        // bias set0 also +4 for first
    init_bias(bias1, 0, 0x4400, 0);        // bias set1 +4
  } else if (mode == 6) {
    init_bias(bias0, 0x5400, 0, 0);        // same set0 twice
    init_bias(bias1, 0x5400, 0, 0);
  } else if (mode == 7) {
    init_bias(bias0, 0, 0x4400, 0);        // set0 alone +4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 8) {
    init_bias(bias0, 0, 0, 0);
    init_bias(bias1, 0, 0x4400, 0);        // set1 alone +4
  } else if (mode == 9) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0xd400, 0, 0);        // set1 negative scale -64
  } else if (mode == 10) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0x5400, 0x7bff, 4);   // set1 feedback (spec order)
  } else if (mode == 11) {
    init_bias(bias0, 0x5400, 0, 0);        // set0 +64
    init_bias(bias1, 0x5400, 0, 0);        // set1 +64 (positive switch)
  } else if (mode == 12) {
    init_bias(bias0, 0xd400, 0, 0);        // set0 -64
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 13) {
    init_bias(bias0, 0, 0xc400, 0);        // set0 output bias -4 (0xc400)
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 14) {
    init_bias(bias0, 0x5400, 0, 0);        // P: +64, bias 0
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 15) {
    init_bias(bias0, 0x5400, 0xc400, 0);   // NB: +64, bias -4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 16) {
    init_bias(bias0, 0xd400, 0x5800, 0);   // NS: -64, bias +128
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 17) {
    init_bias(bias0, 0x5400, 0x5800, 4);   // SH: +64, shape=4, bias +128
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 18) {
    init_bias(bias0, 0x5400, 0xd420, 0);   // R+: +64, bias -66
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 19) {
    init_bias(bias0, 0, 0xc400, 0);        // N: scale 0, bias -4
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 20) {
    // R+ generalized: bias1 = -hi (negative hi) as output bias, scale +64.
    uint16_t hi_bits = 0x5402;
    if (q == 6) hi_bits = 0x5420;
    else if (q == 7) hi_bits = 0x5410;
    else if (q == 8) hi_bits = 0x5408;
    else if (q == 9) hi_bits = 0x5404;
    else if (q == 10) hi_bits = 0x5402;
    init_bias(bias0, 0x5400, 0, 0);        // set0 original
    init_bias(bias1, 0x5400, hi_bits | 0x8000u, 0);  // set1 scale +64, bias -hi
  } else if (mode == 21) {
    init_bias(bias0, 0x5400, 0xd400, 0);   // zero-residual control: scale +64, bias -64
    init_bias(bias1, 0, 0, 0);
  } else if (mode == 22) {
    // zero residual 2-CVT: a=1.0, bias0 +64, bias1 +64/-64
    init_bias(bias0, 0x5400, 0, 0);
    init_bias(bias1, 0x5400, 0xd400, 0);
  } else if (mode == 23) {
    // single convert with known -hi bias
    uint16_t hi_bits = 0x5402;
    if (q == 6) hi_bits = 0x5420;
    else if (q == 7) hi_bits = 0x5410;
    else if (q == 8) hi_bits = 0x5408;
    else if (q == 9) hi_bits = 0x5404;
    else if (q == 10) hi_bits = 0x5402;
    init_bias(bias0, 0x5400, hi_bits | 0x8000u, 0);
    init_bias(bias1, 0, 0, 0);
  } else {
    init_bias(bias1, 0x5400, 0x7bff, 4);
  }

  // V81: bias load address must be 256B aligned; low two bits select set.
  const uintptr_t bias0_reg = (uintptr_t)bias0;
  const uintptr_t bias1_reg = ((uintptr_t)bias1) | 1u;
  const uintptr_t act_reg = (uintptr_t)act;
  const uintptr_t wgt_reg = (uintptr_t)wgt;
  // Spatial mask 0b11100 (YYYXX), all 32 input channels.
  const uint32_t spatial_mask = (14u << 7);
  const uint32_t act_rt = spatial_mask | (31u << 2);
  const uint32_t wr_rt = spatial_mask;

  // Intrinsics are not required: use the documented HMX assembly syntax.
  // activation and weight must appear in one instruction packet.
  asm volatile("mxclracc.hf\n" ::: "memory");
  asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3) }\n"
      : : "r"(act_reg), "r"(act_rt),
          "r"(wgt_reg), "r"((uint32_t)(TILE_BYTES - 1))
      : "memory");

  asm volatile("bias = mxmem2(%0)\n" : : "r"(bias0_reg) : "memory");
  if (mode != 10) {
    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");
  }

  // First convert retains ACC; optionally keeps 4 extra feedback bits.
  uint32_t first_rs = 1u | ((extra_four_bits & 1u) << 8);
  if (mode == 5 || mode == 8) first_rs |= (1u << 12);  // first convert uses bias set1
  asm volatile("cvt.hf = acc(%0)\n" : : "r"(first_rs) : "memory");
  asm volatile("mxmem(%0, %1) = cvt\n"
               : : "r"((uintptr_t)hi), "r"(wr_rt) : "memory");

  if (mode == 10) {
    asm volatile("bias = mxmem2(%0)\n" : : "r"(bias1_reg) : "memory");
  }

  if (mode == 7 || mode == 8 || mode == 12 || mode == 13 || mode == 14 || mode == 15 || mode == 16 || mode == 17 || mode == 18 || mode == 19 || mode == 21 || mode == 23) {
    *hi_bits = hi[0];
    *residual_bits = 0;
    return;
  }

  // Rs[3:2]=1 -> previous convert state used as output bias.
  // Rs[4]=0 -> min; Rs[0]=0 -> clear/swap after convert.
  // This is a *hypothesis test*, not proof that h-x survives rounding.
  uint32_t second_rs;
  if (mode == 6) {
    second_rs = 0;  // bias set 0, no fb
  } else {
    second_rs = (1u << 12);  // bias set 1
    if (mode == 0 || mode == 3 || mode == 10) second_rs |= (1u << 2);  // fb_dest=1
  }
  asm volatile("cvt.hf = acc(%0)\n" : : "r"(second_rs) : "memory");
  asm volatile("mxmem(%0, %1) = cvt\n"
               : : "r"((uintptr_t)residual), "r"(wr_rt) : "memory");

  *hi_bits = hi[0];
  *residual_bits = residual[0];
}

int hexatrain_hmx_acc_residual_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  if (!handle) return HMX_ERR_ARG;
  probe_context_t *context = (probe_context_t *)calloc(1, sizeof(*context));
  if (!context) return HMX_ERR_ALLOC;
  context->tag = HMX_MAGIC;
  *handle = (remote_handle64)(uintptr_t)context;
  return 0;
}

int hexatrain_hmx_acc_residual_close(remote_handle64 handle) {
  probe_context_t *context = (probe_context_t *)(uintptr_t)handle;
  if (!context || context->tag != HMX_MAGIC) return HMX_ERR_ARG;
  context->tag = 0;
  free(context);
  return 0;
}

int hexatrain_hmx_acc_residual_run(remote_handle64 handle,
                                   int *results, int resultsLen,
                                   int *metadata, int metadataLen,
                                   int q, int mode) {
  (void)mode;
  probe_context_t *context = (probe_context_t *)(uintptr_t)handle;
  if (!context || context->tag != HMX_MAGIC ||
       !results || resultsLen < 4 || !metadata || metadataLen < 9) {
    return HMX_ERR_ARG;
  }
  memset(results, 0, (size_t)resultsLen * sizeof(int));
  memset(metadata, 0, (size_t)metadataLen * sizeof(int));
  metadata[0] = HMX_MAGIC;
  metadata[1] = __HEXAGON_ARCH__;
  metadata[2] = VTCM_BYTES;
  metadata[3] = 0;  // DSP kernel microseconds
  metadata[4] = 0;  // power status
  metadata[5] = 0;  // mode A (HMX-only) lock status
  metadata[6] = 0;  // mode B (combined) lock status
  metadata[7] = 0;  // HMX executed flag
  metadata[8] = 0;  // trial count

  static int power_client = 0;
  HAP_power_request_t power_req;
  memset(&power_req, 0, sizeof(power_req));
  power_req.type = HAP_power_set_HMX_v2;
  power_req.hmx_v2.set_power = 1;
  power_req.hmx_v2.power_up = 1;
  power_req.hmx_v2.set_clock = 0;
  int power_status = HAP_power_set(&power_client, &power_req);
  metadata[4] = power_status;
  if (power_status) {
    HAP_power_request_t power_down;
    memset(&power_down, 0, sizeof(power_down));
    power_down.type = HAP_power_set_HMX_v2;
    power_down.hmx_v2.set_power = 1;
    power_down.hmx_v2.power_up = 0;
    power_down.hmx_v2.set_clock = 0;
    (void)HAP_power_set(&power_client, &power_down);
    return HMX_ERR_POWER;
  }

  int combined_lock_status = 0;
  int hmx_only_lock_status = 0;
  uint8_t *vtcm = NULL;
  int combined_resource = 0;
  int hmx_only_resource = 0;
  const uint64_t start = HAP_perf_get_time_us();

  /* Mode A: HMX-only resource (htp-ops-lib style). */
  {
    compute_res_attr_t attr;
    int status = HAP_compute_res_attr_init(&attr);
    if (!status) status = HAP_compute_res_attr_set_hmx_param(&attr, 1);
    if (!status) {
      hmx_only_resource = HAP_compute_res_acquire(&attr, 10000);
      if (hmx_only_resource) {
        hmx_only_lock_status = HAP_compute_res_hmx_lock2(
            hmx_only_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
        if (!hmx_only_lock_status) {
          (void)HAP_compute_res_hmx_unlock2(
              hmx_only_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
        }
        (void)HAP_compute_res_release(hmx_only_resource);
        hmx_only_resource = 0;
      } else {
        hmx_only_lock_status = -1;
      }
    } else {
      hmx_only_lock_status = status;
    }
  }
  metadata[5] = hmx_only_lock_status;

  /* Mode B: combined VTCM + HMX resource (current style). */
  {
    compute_res_attr_t attr;
    int status = HAP_compute_res_attr_init(&attr);
    if (!status) status = HAP_compute_res_attr_set_vtcm_param(&attr, VTCM_BYTES, 1);
    if (!status) status = HAP_compute_res_attr_set_hmx_param(&attr, 1);
    if (!status) {
      combined_resource = HAP_compute_res_acquire(&attr, 10000);
      if (combined_resource) {
        vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
        if (!vtcm || ((uintptr_t)vtcm & (TILE_BYTES - 1u))) {
          combined_lock_status = HMX_ERR_VTCM;
        } else {
          combined_lock_status = HAP_compute_res_hmx_lock2(
              combined_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
          metadata[6] = combined_lock_status;
          if (!combined_lock_status) {
            /* Execute HMX only if lock succeeded. */
            run_one(vtcm, 0, (int)q, mode, &results[0], &results[1]);
            run_one(vtcm, 1, (int)q, mode, &results[2], &results[3]);
            metadata[7] = 1;
            metadata[8] = 2;
            (void)HAP_compute_res_hmx_unlock2(
                combined_resource, HAP_COMPUTE_RES_HMX_NON_SHARED);
          }
        }
        (void)HAP_compute_res_release(combined_resource);
        combined_resource = 0;
      } else {
        combined_lock_status = -1;
        metadata[6] = combined_lock_status;
      }
    } else {
      combined_lock_status = status;
      metadata[6] = combined_lock_status;
    }
  }

  const uint64_t elapsed = HAP_perf_get_time_us() - start;
  metadata[3] = elapsed > 0x7fffffffULL ? 0x7fffffff : (int)elapsed;

  /* Power down. */
  {
    HAP_power_request_t power_down;
    memset(&power_down, 0, sizeof(power_down));
    power_down.type = HAP_power_set_HMX_v2;
    power_down.hmx_v2.set_power = 1;
    power_down.hmx_v2.power_up = 0;
    power_down.hmx_v2.set_clock = 0;
    (void)HAP_power_set(&power_client, &power_down);
  }

  if (metadata[7] == 1) return 0;
  return HMX_ERR_LOCK;
}
