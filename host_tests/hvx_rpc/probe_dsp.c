// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Research wrapper only; all QHL implementation is linked from the local SDK.
#include "hexatrain_hvx_probe.h"
#include "original_qhl.h"
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <qurt_hvx.h>
#include <qurt_error.h>
#include <qurt_thread.h>
#include <qhblas_hvx.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void add_qhl_profile(hexatrain_qhl_profile_t* total,
                            const hexatrain_qhl_profile_t* one) {
  total->total_us += one->total_us;
  total->momentum_nesterov_us += one->momentum_nesterov_us;
  total->norm_normalization_us += one->norm_normalization_us;
  total->vector_ops_us += one->vector_ops_us;
  total->transpose_us += one->transpose_us;
  total->final_update_us += one->final_update_us;
  total->gemm_calls += one->gemm_calls;
  total->matrix_count += one->matrix_count;
  for (int i = 0; i < 5; ++i) {
    total->ns_us[i] += one->ns_us[i];
    total->gemm_xxt_us[i] += one->gemm_xxt_us[i];
    total->gemm_a2_us[i] += one->gemm_a2_us[i];
    total->gemm_bx_us[i] += one->gemm_bx_us[i];
  }
}

static int profile_int(uint64_t value) {
  return value > 0x7fffffffULL ? 0x7fffffff : (int)value;
}

typedef struct probe_context {
  int high_performance;
} probe_context_t;

static int set_high_performance_vote(probe_context_t* context, int enabled) {
  HAP_power_request_t request;
  int status;
  if (!context) return 1001;
  memset(&request, 0, sizeof(request));
  if (enabled) {
    // Exact SDK profiling-example vote: compute client plus DCVS v3
    // PERFORMANCE mode at MAX core/bus corners with sleep disabled.
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    status = HAP_power_set(context, &request);
    if (status) return status;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = TRUE;
    status = HAP_power_set(context, &request);
    if (!status) context->high_performance = 1;
    return status;
  }
  // The SDK profiling example clears its DCVS v3 power-boost request this way.
  request.type = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_core_params = TRUE;
  status = HAP_power_set(context, &request);
  if (!status) context->high_performance = 0;
  return status;
}

typedef struct release_worker {
  const float* input;
  const float* hyper;
  float* output;
  int index;
  int count;
  void* scratch;
  size_t scratch_bytes;
  int status;
  int lock_status;
  int unlock_status;
} release_worker_t;

static void run_release_worker(release_worker_t* worker) {
  const size_t square_elements = (size_t)64 * 64;
  const size_t rect_elements = (size_t)64 * 128;
  const size_t square_input_count = (size_t)76 * 3 * square_elements;
  const size_t square_hyper_count = (size_t)2 * 76;
  const size_t square_output_count = (size_t)76 * 2 * square_elements;
  worker->lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (worker->lock_status) {
    worker->status = 1201;
    return;
  }
  for (int matrix = worker->index; matrix < 114 && !worker->status;
       matrix += worker->count) {
    if (matrix < 76) {
      worker->status = hexatrain_original_qhl_release(
          worker->input + (size_t)matrix * 3 * square_elements, 64,
          worker->hyper[2 * matrix], worker->hyper[2 * matrix + 1],
          worker->output + (size_t)matrix * 2 * square_elements,
          worker->scratch, worker->scratch_bytes);
    } else {
      const int rect = matrix - 76;
      worker->status = hexatrain_original_qhl_release(
          worker->input + square_input_count +
              (size_t)rect * 3 * rect_elements,
          128, worker->hyper[square_hyper_count + 2 * rect],
          worker->hyper[square_hyper_count + 2 * rect + 1],
          worker->output + square_output_count +
              (size_t)rect * 2 * rect_elements,
          worker->scratch, worker->scratch_bytes);
    }
  }
  worker->unlock_status = qurt_hvx_unlock();
  if (!worker->status && worker->unlock_status) worker->status = 1203;
}

static void release_worker_entry(void* opaque) {
  release_worker_t* worker = (release_worker_t*)opaque;
  run_release_worker(worker);
  qurt_thread_exit(worker->status);
}

static int run_release_parallel(const float* input, const float* hyper,
                                float* output, int workers,
                                int reported_units, int* locks_acquired,
                                int* threads_created) {
  const size_t scratch_bytes = hexatrain_original_qhl_scratch_bytes(128);
  const size_t stack_bytes = 32768;
  int result = 0;
  if (workers < 1 || workers > reported_units || workers > 114) return 1200;
  release_worker_t* args =
      (release_worker_t*)calloc((size_t)workers, sizeof(*args));
  qurt_thread_t* threads =
      (qurt_thread_t*)calloc((size_t)workers, sizeof(*threads));
  void** stacks = (void**)calloc((size_t)workers, sizeof(*stacks));
  int* created = (int*)calloc((size_t)workers, sizeof(*created));
  if (!args || !threads || !stacks || !created) {
    result = 1008;
    goto cleanup;
  }
  for (int i = 0; i < workers; ++i) {
    args[i].input = input;
    args[i].hyper = hyper;
    args[i].output = output;
    args[i].index = i;
    args[i].count = workers;
    args[i].lock_status = -1;
    args[i].scratch_bytes = scratch_bytes;
    args[i].scratch = memalign(128, scratch_bytes);
    if (!args[i].scratch) {
      result = 1008;
      goto cleanup;
    }
    if (i > 0) {
      stacks[i] = memalign(128, stack_bytes);
      if (!stacks[i]) {
        result = 1008;
        goto cleanup;
      }
    }
  }
  // Create at most W-1 threads after every shared resource is ready. The RPC
  // thread acts as worker zero, so W workers process many matrices each.
  for (int i = 1; i < workers; ++i) {
      qurt_thread_attr_t attr;
      char name[16];
      qurt_thread_attr_init(&attr);
      snprintf(name, sizeof(name), "muon%d", i);
      qurt_thread_attr_set_name(&attr, name);
      qurt_thread_attr_set_stack_addr(&attr, stacks[i]);
      qurt_thread_attr_set_stack_size(&attr, (unsigned int)stack_bytes);
      qurt_thread_attr_set_priority(&attr, 100);
      const int create_status = qurt_thread_create(
          &threads[i], &attr, release_worker_entry, &args[i]);
      if (create_status) {
        args[i].status = 1202;
        result = 1202;
        goto join;
      }
      created[i] = 1;
      ++*threads_created;
  }
  run_release_worker(&args[0]);

join:
  for (int i = 1; i < workers; ++i) {
    if (created[i]) {
      int thread_status = 0;
      const int join_status = qurt_thread_join(threads[i], &thread_status);
      if (!result && join_status != QURT_EOK &&
          join_status != QURT_ENOTHREAD) result = 1204;
      if (!result && join_status == QURT_EOK && thread_status)
        result = thread_status;
    }
  }
  for (int i = 0; i < workers; ++i) {
    if (!args[i].lock_status) ++*locks_acquired;
    if (!result && args[i].status) result = args[i].status;
  }

cleanup:
  if (args)
    for (int i = 0; i < workers; ++i) free(args[i].scratch);
  if (stacks)
    for (int i = 1; i < workers; ++i) free(stacks[i]);
  free(created);
  free(stacks);
  free(threads);
  free(args);
  return result;
}

static int run_original_group(const float* input, int columns, size_t count,
                              const float* hyper, float* output,
                              void* scratch, size_t scratch_bytes,
                              hexatrain_qhl_profile_t* group_profile) {
  const size_t stride = (size_t)3 * 64 * (size_t)columns;
  const size_t output_stride = (size_t)8 * 64 * (size_t)columns;
  int status = 0;
  for (size_t index = 0; index < count && !status; ++index) {
    hexatrain_qhl_profile_t one_profile;
    memset(&one_profile, 0, sizeof(one_profile));
    status = hexatrain_original_qhl_profiled(
        input + index * stride, columns, hyper[2 * index], hyper[2 * index + 1],
        output + index * output_stride, &one_profile, scratch, scratch_bytes);
    add_qhl_profile(group_profile, &one_profile);
  }
  return status;
}

#if __HEXAGON_ARCH__ != 81
#error V81 is required.
#endif

int hexatrain_hvx_probe_open(const char* uri, remote_handle64* handle) {
  (void)uri;
  if (!handle) return 1001;
  probe_context_t* context = (probe_context_t*)calloc(1, sizeof(*context));
  if (!context) return 1002;
  *handle = (remote_handle64)(uintptr_t)context;
  return 0;
}

int hexatrain_hvx_probe_close(remote_handle64 handle) {
  probe_context_t* context = (probe_context_t*)(uintptr_t)handle;
  int status = 0;
  if (!context) return 1001;
  if (context->high_performance)
    status = set_high_performance_vote(context, 0);
  free(context);
  return status;
}

int hexatrain_hvx_probe_configure(remote_handle64 handle,
                                  int high_performance,
                                  int* metadata, int metadataLen) {
  probe_context_t* context = (probe_context_t*)(uintptr_t)handle;
  if (!context || !metadata || metadataLen < 4 ||
      (high_performance != 0 && high_performance != 1))
    return 1003;
  memset(metadata, 0, (size_t)metadataLen * sizeof(int));
  const int hardware = qurt_hvx_get_units();
  const int units_128b = hardware > 0 ? (hardware >> 8) & 0xff : 0;
  int status = 0;
  if (high_performance != context->high_performance)
    status = set_high_performance_vote(context, high_performance);
  metadata[0] = hardware;
  metadata[1] = units_128b;
  metadata[2] = context->high_performance;
  metadata[3] = status;
  return status;
}

int hexatrain_hvx_probe_transport(remote_handle64 handle,
                                  const unsigned char* left, int leftLen,
                                  const unsigned char* right, int rightLen,
                                  unsigned char* result, int resultLen,
                                  int* metadata, int metadataLen) {
  if (!handle || !left || leftLen < 1 || !right || rightLen < 1 ||
      !result || resultLen < 1 || !metadata || metadataLen < 4)
    return 1003;
  const uint64_t start = HAP_perf_get_time_us();
  const unsigned int checksum = (unsigned int)left[0] + left[leftLen - 1] +
                                right[0] + right[rightLen - 1];
  result[0] = (unsigned char)checksum;
  result[resultLen - 1] = (unsigned char)(checksum ^ 0xa5u);
  metadata[0] = 0x54524e53;
  metadata[1] = (int)(HAP_perf_get_time_us() - start);
  metadata[2] = leftLen + rightLen + resultLen;
  metadata[3] = (int)checksum;
  return 0;
}

int hexatrain_hvx_probe_run(remote_handle64 handle, int mode, int m, int k, int n,
    const float* left, int leftLen, const float* right, int rightLen,
    float* result, int resultLen, int* metadata, int metadataLen) {
  if (!handle || !left || !right || !result || !metadata || metadataLen < 8)
    return 1003;
  memset(metadata, 0, (size_t)metadataLen * sizeof(int));
  const int wrapper_profile = mode == 6;
  const uint64_t validation_start = wrapper_profile ? HAP_perf_get_time_us() : 0;
  if (mode == 0) {
    if (m != 1 || k != 32 || n != 32 || leftLen != 32 || rightLen != 32 || resultLen != 32)
      return 1004;
  } else if (mode == 1) {
    if (m != 64 || (k != 64 && k != 128) || (n != 64 && n != 128) ||
        leftLen != m * k || rightLen != k * n || resultLen != m * n)
      return 1005;
  } else if (mode == 2 || mode == 3) {
    // Single matrix or a canonical batch sharing one shape. The matrix count
    // is carried by the buffer lengths so the IDL signature is unchanged and
    // one RPC still covers a whole square or rectangular group.
    if (m != 64 || (k != 64 && k != 128) || n != k || leftLen <= 0 ||
        leftLen % (3 * m * k) != 0 || leftLen / (3 * m * k) > 256 ||
        rightLen != 2 * (leftLen / (3 * m * k)) ||
        resultLen != 8 * leftLen / 3)
      return 1005;
  } else if (mode == 4 || mode == 5 || mode == 6) {
    const int square_left = 76 * 3 * 64 * 64;
    const int rect_left = 38 * 3 * 64 * 128;
    const int output_planes = (mode == 5 || mode == 6) ? 2 : 8;
    const int square_result = 76 * output_planes * 64 * 64;
    const int rect_result = 38 * output_planes * 64 * 128;
    if (((mode == 4 && (m != 64 || k != 64 || n != 64)) ||
         ((mode == 5 || mode == 6) && (m < 1 || k != 0 || n != 0))) ||
        leftLen != square_left + rect_left ||
        rightLen != 2 * 76 + 2 * 38 ||
        resultLen != square_result + rect_result)
      return 1005;
  } else return 1006;
  // The production probe's immutable input fixture is checked once while the
  // host packs it. Mode 6 retains the DSP-side scans for overhead attribution.
  if (mode != 5) {
    for (int i = 0; i < leftLen; ++i) if (!isfinite(left[i])) return 1007;
    for (int i = 0; i < rightLen; ++i) if (!isfinite(right[i])) return 1007;
  }
  if (wrapper_profile)
    metadata[16] = (int)(HAP_perf_get_time_us() - validation_start);
  const int direct_buffers = mode == 5 || mode == 6;
  float* a = direct_buffers ? (float*)left :
      memalign(128, (size_t)leftLen * sizeof(float));
  float* b = direct_buffers ? (float*)right :
      memalign(128, (size_t)rightLen * sizeof(float));
  float* c = direct_buffers ? result :
      memalign(128, (size_t)resultLen * sizeof(float));
  if (!a || !b || !c ||
      (direct_buffers && (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 127u))) {
    if (!direct_buffers) { free(a); free(b); free(c); }
    return 1008;
  }
  if (!direct_buffers) {
    memcpy(a, left, (size_t)leftLen * sizeof(float));
    memcpy(b, right, (size_t)rightLen * sizeof(float));
    for (int i = 0; i < resultLen; ++i) c[i] = NAN;
  }
  HAP_power_request_t power;
  memset(&power, 0, sizeof(power));
  power.type = HAP_power_set_HVX;
  power.hvx.power_up = TRUE;
  const uint64_t power_up_start = wrapper_profile ? HAP_perf_get_time_us() : 0;
  int status = HAP_power_set((void*)(uintptr_t)handle, &power);
  if (wrapper_profile)
    metadata[17] = (int)(HAP_perf_get_time_us() - power_up_start);
  int locked = 0;
  unsigned int vtcm_context = 0;
  void* vtcm_mem = NULL;
  int vtcm_status = 0;
  const size_t scratch_bytes =
      mode == 2 || mode == 3 ? hexatrain_original_qhl_scratch_bytes(k) :
      (mode == 4 ? hexatrain_original_qhl_scratch_bytes(128) : 0);
  if (!status && mode == 3) {
    compute_res_attr_t compute_res;
    memset(&compute_res, 0, sizeof(compute_res));
    vtcm_status = HAP_compute_res_attr_init(&compute_res);
    if (!vtcm_status)
      vtcm_status = HAP_compute_res_attr_set_serialize(&compute_res, 1);
    if (!vtcm_status)
      vtcm_status = HAP_compute_res_attr_set_vtcm_param(
          &compute_res, (unsigned int)scratch_bytes, 1);
    if (!vtcm_status) {
      vtcm_context = HAP_compute_res_acquire(&compute_res, 10000);
      if (!vtcm_context) vtcm_status = 1011;
    }
    if (!vtcm_status) {
      vtcm_mem = HAP_compute_res_attr_get_vtcm_ptr(&compute_res);
      if (!vtcm_mem || ((uintptr_t)vtcm_mem & 127u) != 0)
        vtcm_status = 1012;
    }
    if (vtcm_status && vtcm_context) {
      (void)HAP_compute_res_release(vtcm_context);
      vtcm_context = 0;
    }
    if (vtcm_status) status = vtcm_status;
  }
  if (!status && mode != 5 && mode != 6) {
    status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    locked = status == 0;
  }
  hexatrain_qhl_profile_t group_profile;
  memset(&group_profile, 0, sizeof(group_profile));
  if (!status) {
    const uint64_t start = HAP_perf_get_time_us();
    if (mode == 0) status = qhblas_hvx_vector_add_af(a, b, c, 32);
    else if (mode == 1)
      status = qhblas_hvx_matrix_matrix_mpy_af(a, b, c, (uint32_t)m, (uint32_t)k, (uint32_t)n);
    else if (mode == 2 || mode == 3) {
      // Sequential canonical matrices reuse one scratch allocation pattern
      // inside hexatrain_original_qhl; matrices never share buffers. Fail
      // closed on the first failing matrix instead of partial writes.
      const size_t stride = (size_t)3 * m * k;
      const size_t count = (size_t)leftLen / stride;
      (void)stride;
      status = run_original_group(
          a, k, count, b, c, mode == 3 ? vtcm_mem : NULL,
          mode == 3 ? scratch_bytes : 0, &group_profile);
    } else if (mode == 4) {
      const size_t square_left = (size_t)76 * 3 * 64 * 64;
      const size_t square_hyper = (size_t)2 * 76;
      const size_t square_result = (size_t)76 * 8 * 64 * 64;
      status = run_original_group(a, 64, 76, b, c, NULL, scratch_bytes,
                                  &group_profile);
      if (!status)
        status = run_original_group(a + square_left, 128, 38,
                                    b + square_hyper, c + square_result,
                                    NULL, scratch_bytes, &group_profile);
    } else {
      const int hardware = qurt_hvx_get_units();
      const int units_128b = hardware > 0 ? (hardware >> 8) & 0xff : 0;
      int locks_acquired = 0;
      int threads_created = 0;
      status = run_release_parallel(a, b, c, m, units_128b,
                                    &locks_acquired, &threads_created);
      metadata[8] = hardware;
      metadata[9] = units_128b;
      metadata[10] = m;
      metadata[11] = locks_acquired;
      metadata[12] = threads_created;
      metadata[13] = (int)((size_t)leftLen * sizeof(float));
      metadata[14] = (int)((size_t)rightLen * sizeof(float));
      metadata[15] = (int)((size_t)resultLen * sizeof(float));
    }
    metadata[3] = (int)(HAP_perf_get_time_us() - start);
    if (mode == 2 || mode == 3 || mode == 4) {
      metadata[8] = profile_int(group_profile.total_us);
      metadata[9] = profile_int(group_profile.momentum_nesterov_us);
      metadata[10] = profile_int(group_profile.norm_normalization_us);
      metadata[11] = profile_int(group_profile.ns_us[0]);
      metadata[12] = profile_int(group_profile.ns_us[1]);
      metadata[13] = profile_int(group_profile.ns_us[2]);
      metadata[14] = profile_int(group_profile.ns_us[3]);
      metadata[15] = profile_int(group_profile.ns_us[4]);
      metadata[16] = profile_int(group_profile.final_update_us);
      uint64_t gemm_total = 0;
      for (int i = 0; i < 5; ++i) {
        metadata[17] += profile_int(group_profile.gemm_xxt_us[i]);
        metadata[18] += profile_int(group_profile.gemm_a2_us[i]);
        metadata[19] += profile_int(group_profile.gemm_bx_us[i]);
        gemm_total += group_profile.gemm_xxt_us[i] +
                      group_profile.gemm_a2_us[i] +
                      group_profile.gemm_bx_us[i];
      }
      metadata[20] = profile_int(group_profile.vector_ops_us);
      metadata[21] = profile_int(group_profile.transpose_us);
      metadata[22] = (int)group_profile.gemm_calls;
      metadata[23] = (int)group_profile.matrix_count;
      metadata[24] = (int)scratch_bytes;
      metadata[25] = vtcm_status;
      metadata[26] = mode == 3 && !vtcm_status ? 1 : 0;
      metadata[27] = profile_int(gemm_total);
    }
    if (!status && mode != 5) {
      const uint64_t output_validation_start =
          wrapper_profile ? HAP_perf_get_time_us() : 0;
      for (int i = 0; i < resultLen; ++i) if (!isfinite(c[i])) { status = 1009; break; }
      if (wrapper_profile)
        metadata[18] = (int)(HAP_perf_get_time_us() - output_validation_start);
    }
    if (!status && !direct_buffers)
      memcpy(result, c, (size_t)resultLen * sizeof(float));
  }
  if (locked) {
    const int unlock_status = qurt_hvx_unlock();
    if (!status) status = unlock_status;
  }
  if (vtcm_context) {
    const int release_status = HAP_compute_res_release(vtcm_context);
    if (!status) status = release_status;
  }
  power.hvx.power_up = FALSE;
  const uint64_t power_down_start = wrapper_profile ? HAP_perf_get_time_us() : 0;
  const int power_down_status = HAP_power_set((void*)(uintptr_t)handle, &power);
  if (wrapper_profile)
    metadata[19] = (int)(HAP_perf_get_time_us() - power_down_start);
  if (!status) status = power_down_status;
  metadata[0] = 0x48565831;
  metadata[1] = __HEXAGON_ARCH__;
  metadata[2] = 128;
  metadata[4] = status;
  metadata[5] = mode == 3 && !vtcm_status ? 2 : 1;
  metadata[6] = (mode == 5 || mode == 6) ? (metadata[11] == m) : locked;
  metadata[7] = mode;
  if (mode == 2 || mode == 3 || mode == 4) {
    const size_t stride = (size_t)3 * m * k;
    const size_t batch_count = (size_t)leftLen / stride;
    const int alignment_pass =
        ((((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 127u) == 0) &&
        (scratch_bytes == 0 || (scratch_bytes & 127u) == 0);
    metadata[28] = alignment_pass;
    metadata[29] = (m % 32 == 0 && k % 32 == 0 && n % 32 == 0);
    metadata[30] = mode == 4 ||
                   (stride == (size_t)3 * m * k &&
                    (size_t)resultLen == batch_count * (size_t)8 * m * k);
    metadata[31] = scratch_bytes != 0;
  } else if (mode == 5) {
    metadata[28] = ((((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 127u) == 0);
    metadata[29] = 1;
    metadata[30] = 1;
    metadata[31] = 1;
  }
  if (!direct_buffers) { free(a); free(b); free(c); }
  return status;
}
