// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "hexatrain_hvx_probe.h"
#include <remote.h>
#include <rpcmem.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int read_exact(const char* path, float* buffer, int count) {
  FILE* file = fopen(path, "rb");
  if (!file) return 0;
  const size_t read_count = fread(buffer, sizeof(float), (size_t)count, file);
  const int extra = fgetc(file);
  const int closed = fclose(file);
  if (read_count != (size_t)count || extra != EOF || closed) return 0;
  for (int i = 0; i < count; ++i) if (!isfinite(buffer[i])) return 0;
  return 1;
}

static double time_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1000000.0 + t.tv_nsec / 1000.0;
}

static int compare_double(const void* left, const void* right) {
  const double a = *(const double*)left;
  const double b = *(const double*)right;
  return (a > b) - (a < b);
}

typedef struct transport_case {
  const char* name;
  int left_bytes;
  int right_bytes;
  int result_bytes;
} transport_case_t;

static int run_transport_microbench(remote_handle64 handle) {
  const transport_case_t cases[] = {
      {"tiny", 64, 4, 60},
      {"1MiB", 629056, 128, 419392},
      {"current", 7471104, 912, 4980736},
  };
  for (size_t case_index = 0;
       case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
    const transport_case_t* one = &cases[case_index];
    const double allocation_start = time_us();
    unsigned char* left = rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, one->left_bytes);
    unsigned char* right = rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, one->right_bytes);
    unsigned char* result = rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, one->result_bytes);
    int* metadata = rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 8 * sizeof(int));
    const double allocation_us = time_us() - allocation_start;
    if (!left || !right || !result || !metadata) {
      if (left) rpcmem_free(left);
      if (right) rpcmem_free(right);
      if (result) rpcmem_free(result);
      if (metadata) rpcmem_free(metadata);
      return 35;
    }
    const double initialization_start = time_us();
    memset(left, 0x31, (size_t)one->left_bytes);
    memset(right, 0x57, (size_t)one->right_bytes);
    memset(result, 0, (size_t)one->result_bytes);
    memset(metadata, 0, 8 * sizeof(int));
    const double initialization_us = time_us() - initialization_start;
    double rpc_samples[5] = {0}, dsp_samples[5] = {0};
    double rpc_sum = 0, dsp_sum = 0;
    int status = 0;
    for (int invocation = 0; invocation < 6; ++invocation) {
      const double start = time_us();
      status = hexatrain_hvx_probe_transport(
          handle, left, one->left_bytes, right, one->right_bytes,
          result, one->result_bytes, metadata, 8);
      const double rpc_us = time_us() - start;
      if (status || metadata[0] != 0x54524e53 ||
          metadata[2] != one->left_bytes + one->right_bytes +
                             one->result_bytes) {
        status = status ? status : 36;
        break;
      }
      printf("transport_repetition case=%s repetition=%d warmup=%d "
             "rpc_us=%.3f dsp_us=%d\n", one->name, invocation,
             invocation == 0, rpc_us, metadata[1]);
      if (invocation > 0) {
        rpc_samples[invocation - 1] = rpc_us;
        dsp_samples[invocation - 1] = metadata[1];
        rpc_sum += rpc_us;
        dsp_sum += metadata[1];
      }
    }
    if (!status) {
      double rpc_sorted[5], dsp_sorted[5];
      memcpy(rpc_sorted, rpc_samples, sizeof(rpc_sorted));
      memcpy(dsp_sorted, dsp_samples, sizeof(dsp_sorted));
      qsort(rpc_sorted, 5, sizeof(double), compare_double);
      qsort(dsp_sorted, 5, sizeof(double), compare_double);
      printf("transport_summary cache=CACHED case=%s payload_bytes=%d "
             "left_bytes=%d right_bytes=%d result_bytes=%d "
             "allocation_us=%.3f initialization_us=%.3f "
             "rpc_best_us=%.3f rpc_median_us=%.3f rpc_mean_us=%.3f "
             "dsp_best_us=%.3f dsp_median_us=%.3f dsp_mean_us=%.3f\n",
             one->name, one->left_bytes + one->right_bytes + one->result_bytes,
             one->left_bytes, one->right_bytes, one->result_bytes,
             allocation_us, initialization_us,
             rpc_sorted[0], rpc_sorted[2], rpc_sum / 5.0,
             dsp_sorted[0], dsp_sorted[2], dsp_sum / 5.0);
    }
    rpcmem_free(left);
    rpcmem_free(right);
    rpcmem_free(result);
    rpcmem_free(metadata);
    if (status) return status;
  }
  return 0;
}

int main(int argc, char** argv) {
  int mode = 0, m = 1, k = 32, n = 32, batches = 1;
  int combined = 0;
  int production = 0, performance_vote = 0;
  int overhead_profile = 0;
  int transport_microbench = 0;
  if (argc == 2 && strcmp(argv[1], "--transport-microbench") == 0) {
    transport_microbench = 1;
  } else if (argc == 10 &&
             (strcmp(argv[1], "--production-benchmark") == 0 ||
              strcmp(argv[1], "--overhead-profile") == 0)) {
    overhead_profile = strcmp(argv[1], "--overhead-profile") == 0;
    mode = overhead_profile ? 6 : 5;
    combined = 1;
    production = 1;
    m = atoi(argv[2]);
    performance_vote = atoi(argv[3]);
    if (m < 1 || (performance_vote != 0 && performance_vote != 1)) return 21;
    k = 0; n = 0; batches = 114;
    // Normalize file operands to the existing combined-fixture positions.
    argv[2] = argv[4]; argv[3] = argv[5];
    argv[4] = argv[6]; argv[5] = argv[7];
    argv[6] = argv[8]; argv[7] = argv[9];
  } else if (argc == 8 && strcmp(argv[1], "--original-combined") == 0) {
    mode = 4;
    combined = 1;
    m = 64; k = 64; n = 64; batches = 114;
  } else if (argc == 8 && (strcmp(argv[1], "--gemm") == 0 ||
                    strcmp(argv[1], "--original") == 0)) {
    if (strcmp(argv[2], "64") ||
        (strcmp(argv[3], "64") && strcmp(argv[3], "128")) ||
        (strcmp(argv[4], "64") && strcmp(argv[4], "128"))) return 21;
    mode = strcmp(argv[1], "--original") == 0 ? 2 : 1;
    m = 64; k = atoi(argv[3]); n = atoi(argv[4]);
    if (mode == 2 && k != n) return 21;
    batches = 1;
  } else if (argc == 9 &&
             (strcmp(argv[1], "--original-batch") == 0 ||
              strcmp(argv[1], "--original-profile") == 0 ||
              strcmp(argv[1], "--original-vtcm") == 0)) {
    if (strcmp(argv[2], "64") ||
        (strcmp(argv[3], "64") && strcmp(argv[3], "128")) ||
        (strcmp(argv[4], "64") && strcmp(argv[4], "128"))) return 21;
    mode = strcmp(argv[1], "--original-vtcm") == 0 ? 3 : 2;
    m = 64; k = atoi(argv[3]); n = atoi(argv[4]);
    if (k != n) return 21;
    batches = atoi(argv[5]);
    if (batches < 1 || batches > 256) return 21;
    // Shift the file operands by one relative to the single-matrix CLI.
    argv[5] = argv[6]; argv[6] = argv[7]; argv[7] = argv[8];
  } else if (argc != 1 && !(argc == 2 && strcmp(argv[1], "--capabilities") == 0)) return 21;
  const char* labels[] = {"domain_support", "unsigned_pd_support", "hvx_64b",
      "hvx_128b", "vtcm_page", "vtcm_count", "arch"};
  unsigned values[7] = {0};
  for (unsigned i = 0; i < 7; ++i) {
    struct remote_dsp_capability cap = {CDSP_DOMAIN_ID, i, 0};
    const int status = remote_handle_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
    printf("capability=%s status=%d value=%u\n", labels[i], status, cap.capability);
    if (status) return 20;
    values[i] = cap.capability;
  }
  printf("queried_domain=cdsp fallback=false\n");
  if (argc == 2 && strcmp(argv[1], "--capabilities") == 0) return 0;
  if (!values[0] || !values[1] || !values[3]) return 22;
  // Fail closed on the raw ARCH_VER observed during this target's bring-up.
  if (values[6] != 0x8081) return 22;
  struct remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  int status = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd, sizeof(unsigned_pd));
  printf("unsigned_pd_session_status=%d\n", status);
  if (status) return 23;
  remote_handle64 handle = 0;
  char uri[512];
  const int uri_length = snprintf(uri, sizeof(uri), "%s&_dom=cdsp", hexatrain_hvx_probe_URI);
  if (uri_length < 0 || uri_length >= (int)sizeof(uri)) return 24;
  const double setup_start = time_us();
  status = hexatrain_hvx_probe_open(uri, &handle);
  printf("dsp_open_status=%d rpc_setup_us=%.3f\n", status, time_us() - setup_start);
  if (status) return 25;
  int configure_metadata[4] = {0};
  if (transport_microbench) {
    status = hexatrain_hvx_probe_configure(handle, 1, configure_metadata, 4);
    printf("configure_status=%d power_vote=PERFORMANCE vote_active=%d\n",
           status, configure_metadata[2]);
    int result = status ? 37 : run_transport_microbench(handle);
    status = hexatrain_hvx_probe_close(handle);
    printf("dsp_close_status=%d\n", status);
    if (!result && status) result = 29;
    return result;
  }
  if (production) {
    status = hexatrain_hvx_probe_configure(
        handle, performance_vote, configure_metadata, 4);
    printf("configure_status=%d power_vote=%s hvx_hardware=0x%x reported_hvx_units_128b=%d vote_active=%d\n",
           status, performance_vote ? "PERFORMANCE" : "DEFAULT",
           configure_metadata[0], configure_metadata[1], configure_metadata[2]);
    if (status || configure_metadata[1] < m) {
      const int close_status = hexatrain_hvx_probe_close(handle);
      printf("dsp_close_status=%d\n", close_status);
      return status ? 33 : 34;
    }
  }
  const int a_count = (mode == 2 || mode == 3) ? batches * 3 * m * k :
                      ((mode == 4 || mode == 5 || mode == 6) ?
                       76 * 3 * 64 * 64 + 38 * 3 * 64 * 128 :
                       (mode ? m * k : 32));
  const int b_count = (mode == 2 || mode == 3) ? batches * 2 :
                      ((mode == 4 || mode == 5 || mode == 6) ?
                       2 * 76 + 2 * 38 : (mode ? k * n : 32));
  const int c_count = (mode == 2 || mode == 3) ? batches * 8 * m * k :
                      (mode == 4 ? 76 * 8 * 64 * 64 + 38 * 8 * 64 * 128 :
                       ((mode == 5 || mode == 6) ?
                        76 * 2 * 64 * 64 + 38 * 2 * 64 * 128 :
                       (mode ? m * n : 32)));
  const double pack_start = time_us();
  float* a = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, a_count * sizeof(float));
  float* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, b_count * sizeof(float));
  float* c = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, c_count * sizeof(float));
  int* metadata = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 32 * sizeof(int));
  int result = 0;
  if (!a || !b || !c || !metadata) { result = 26; goto cleanup; }
  if (mode) {
    if (combined) {
      const size_t square_input = (size_t)76 * 3 * 64 * 64;
      const size_t square_hyper = (size_t)2 * 76;
      if (!read_exact(argv[2], a, (int)square_input) ||
          !read_exact(argv[4], a + square_input,
                      (int)((size_t)38 * 3 * 64 * 128)) ||
          !read_exact(argv[3], b, (int)square_hyper) ||
          !read_exact(argv[5], b + square_hyper, 2 * 38)) {
        result = 30; goto cleanup;
      }
    } else if (!read_exact(argv[5], a, a_count) ||
               !read_exact(argv[6], b, b_count)) {
      result = 30; goto cleanup;
    }
  } else {
    for (int i = 0; i < 32; ++i) {
      a[i] = (float)(i - 16) / 8;
      b[i] = (float)(31 - i) / 16;
    }
  }
  for (int i = 0; i < c_count; ++i) c[i] = NAN;
  memset(metadata, 0, 32 * sizeof(int));
  const double host_pack_us = time_us() - pack_start;
  double rpc_us = 0.0;
  double rpc_samples[5] = {0}, kernel_samples[5] = {0};
  double prepare_samples[5] = {0}, postcheck_samples[5] = {0};
  double rpc_sum = 0.0, kernel_sum = 0.0;
  double prepare_sum = 0.0, postcheck_sum = 0.0;
  const int invocations = production ? 6 : 1;
  for (int invocation = 0; invocation < invocations; ++invocation) {
    const double prepare_start = time_us();
    for (int i = 0; i < c_count; ++i) c[i] = NAN;
    memset(metadata, 0, 32 * sizeof(int));
    const double prepare_us = time_us() - prepare_start;
    const double invoke_start = time_us();
    status = hexatrain_hvx_probe_run(
        handle, mode, m, k, n, a, a_count, b, b_count,
        c, c_count, metadata, 32);
    rpc_us = time_us() - invoke_start;
    if (status) break;
    if (production) {
      const double postcheck_start = time_us();
      int invocation_finite = 1;
      for (int i = 0; i < c_count; ++i)
        if (!isfinite(c[i])) invocation_finite = 0;
      const double postcheck_us = time_us() - postcheck_start;
      printf("benchmark_repetition=%d warmup=%d rpc_us=%.3f kernel_us=%d "
             "host_prepare_us=%.3f host_postcheck_us=%.3f finite=%d\n",
             invocation, invocation == 0, rpc_us, metadata[3], prepare_us,
             postcheck_us, invocation_finite);
      if (overhead_profile)
        printf("wrapper_repetition=%d warmup=%d input_validation_us=%d "
               "hvx_power_up_us=%d output_validation_us=%d "
               "hvx_power_down_us=%d\n",
               invocation, invocation == 0, metadata[16], metadata[17],
               metadata[18], metadata[19]);
      if (!invocation_finite) {
        result = 28;
        break;
      }
      if (invocation > 0) {
        rpc_samples[invocation - 1] = rpc_us;
        kernel_samples[invocation - 1] = metadata[3];
        prepare_samples[invocation - 1] = prepare_us;
        postcheck_samples[invocation - 1] = postcheck_us;
        rpc_sum += rpc_us;
        kernel_sum += metadata[3];
        prepare_sum += prepare_us;
        postcheck_sum += postcheck_us;
      }
    }
  }
  printf("host_pack_us=%.3f rpc_calls=1 rpc_status=%d rpc_us=%.3f\n",
         host_pack_us, status, rpc_us);
  if (status) { result = 27; goto cleanup; }
  if (result) goto cleanup;
  if (production) {
    double rpc_sorted[5], kernel_sorted[5], prepare_sorted[5], postcheck_sorted[5];
    memcpy(rpc_sorted, rpc_samples, sizeof(rpc_sorted));
    memcpy(kernel_sorted, kernel_samples, sizeof(kernel_sorted));
    memcpy(prepare_sorted, prepare_samples, sizeof(prepare_sorted));
    memcpy(postcheck_sorted, postcheck_samples, sizeof(postcheck_sorted));
    qsort(rpc_sorted, 5, sizeof(double), compare_double);
    qsort(kernel_sorted, 5, sizeof(double), compare_double);
    qsort(prepare_sorted, 5, sizeof(double), compare_double);
    qsort(postcheck_sorted, 5, sizeof(double), compare_double);
    printf("production_summary warmup_repetitions=1 measured_repetitions=5 rpc_best_us=%.3f rpc_median_us=%.3f rpc_mean_us=%.3f kernel_best_us=%.3f kernel_median_us=%.3f kernel_mean_us=%.3f\n",
           rpc_sorted[0], rpc_sorted[2], rpc_sum / 5.0,
           kernel_sorted[0], kernel_sorted[2], kernel_sum / 5.0);
    printf("host_validation_summary input_finite_prevalidated=1 "
           "output_finite_checked_each_invocation=1 "
           "prepare_best_us=%.3f prepare_median_us=%.3f prepare_mean_us=%.3f "
           "postcheck_best_us=%.3f postcheck_median_us=%.3f "
           "postcheck_mean_us=%.3f\n",
           prepare_sorted[0], prepare_sorted[2], prepare_sum / 5.0,
           postcheck_sorted[0], postcheck_sorted[2], postcheck_sum / 5.0);
  }
  int correct = metadata[0] == 0x48565831 && metadata[1] == 81 && metadata[2] == 128 &&
      metadata[4] == 0 && metadata[6] == 1 && metadata[7] == mode &&
      ((mode == 3) ? metadata[5] == 2 : metadata[5] == 1);
  int finite = 1;
  const double unpack_start = time_us();
  for (int i = 0; i < c_count; ++i) {
    if (!isfinite(c[i])) finite = 0;
    if (!mode && c[i] != a[i] + b[i]) correct = 0;
  }
  const double host_unpack_us = time_us() - unpack_start;
  printf("backend=HVX_QHL domain=cdsp vector_bytes=%d dsp_arch=%d kernel_us=%d kernel_status=%d scratch=%s finite=%d metadata_and_vector_check=%d fallback=false\n",
      metadata[2], metadata[1], metadata[3], metadata[4],
      metadata[5] == 2 ? "VTCM" : "DDR_SHARED", finite, correct);
  printf("rpc_overhead_us=%.3f host_unpack_us=%.3f rpc_kernel_timer_comparable=true\n",
         rpc_us - metadata[3], host_unpack_us);
  if (mode == 2 || mode == 3 || mode == 4) {
    printf("profile_matrix_count=%d profile_total_us=%d momentum_nesterov_us=%d norm_normalization_us=%d ns1_us=%d ns2_us=%d ns3_us=%d ns4_us=%d ns5_us=%d final_update_us=%d vector_ops_us=%d transpose_us=%d gemm_calls=%d gemm_xxt_us=%d gemm_a2_us=%d gemm_bx_us=%d gemm_total_us=%d scratch_bytes=%d vtcm_status=%d vtcm_used=%d\n",
        metadata[23], metadata[8], metadata[9], metadata[10], metadata[11],
        metadata[12], metadata[13], metadata[14], metadata[15], metadata[16],
        metadata[20], metadata[21], metadata[22], metadata[17], metadata[18],
        metadata[19], metadata[27], metadata[24], metadata[25], metadata[26]);
    const int host_alignment_pass =
        ((((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 127u) == 0);
    printf("qhl_fast_path_alignment=%s dsp_buffer_alignment=%s host_buffer_alignment=%s shape_conditions=%s contiguous_layout=%s\n",
        metadata[28] ? "PASS" : "FAIL", metadata[28] ? "PASS" : "FAIL",
        host_alignment_pass ? "PASS" : "FAIL",
        metadata[29] ? "PASS" : "FAIL", metadata[30] ? "PASS" : "FAIL");
  }
  if (mode == 5 || mode == 6) {
    printf("production_mode=true workers=%d reported_hvx_units_128b=%d hvx_locks_acquired=%d child_threads_created=%d input_bytes=%d hyper_bytes=%d output_bytes=%d bytes_per_update=%d diagnostic_taps=0 fine_grained_timers=%d intermediate_output_copies=0\n",
        metadata[10], metadata[9], metadata[11], metadata[12], metadata[13],
        metadata[14], metadata[15], metadata[13] + metadata[14] + metadata[15],
        overhead_profile);
  }
  if (mode == 6) {
    printf("wrapper_profile input_validation_us=%d hvx_power_up_us=%d "
           "output_validation_us=%d hvx_power_down_us=%d\n",
           metadata[16], metadata[17], metadata[18], metadata[19]);
  }
  if (!correct || !finite) result = 28;
  if (!result && mode) {
    if (combined) {
      const size_t square_result = (size_t)76 *
          ((mode == 5 || mode == 6) ? 2 : 8) * 64 * 64;
      FILE* square_file = fopen(argv[6], "wb");
      FILE* rect_file = fopen(argv[7], "wb");
      if (!square_file || !rect_file) {
        if (square_file) fclose(square_file);
        if (rect_file) fclose(rect_file);
        result = 31;
        goto cleanup;
      }
      const size_t square_written = fwrite(c, sizeof(float), square_result,
                                           square_file);
      const size_t rect_written = fwrite(c + square_result, sizeof(float),
                                         (size_t)c_count - square_result,
                                         rect_file);
      const int square_closed = fclose(square_file);
      const int rect_closed = fclose(rect_file);
      if (square_written != square_result ||
          rect_written != (size_t)c_count - square_result || square_closed ||
          rect_closed)
        result = 32;
    } else {
      FILE* file = fopen(argv[7], "wb");
      if (!file) { result = 31; goto cleanup; }
      const size_t written = fwrite(c, sizeof(float), (size_t)c_count, file);
      const int closed = fclose(file);
      if (written != (size_t)c_count || closed) result = 32;
    }
    printf("output_written=%d mode=%d m=%d k=%d n=%d batches=%d numeric_gate=HOST_COMPARISON_REQUIRED\n",
           !result, mode, m, k, n,
           (mode == 2 || mode == 3) ? batches :
           ((mode == 4 || mode == 5 || mode == 6) ? 114 : 1));
  }
cleanup:
  if (a) rpcmem_free(a);
  if (b) rpcmem_free(b);
  if (c) rpcmem_free(c);
  if (metadata) rpcmem_free(metadata);
  status = hexatrain_hvx_probe_close(handle);
  printf("dsp_close_status=%d\n", status);
  if (!result && status) result = 29;
  return result;
}
