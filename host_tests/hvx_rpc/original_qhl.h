// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <stddef.h>
#include <stdint.h>

// Canonical matrix only. Input planes: weights, gradient, previous momentum.
// Output planes: weights, momentum, normalized X, NS1, NS2, NS3, NS4, NS5.
// Execution belongs entirely to the DSP; no host arithmetic fallback exists.
int hexatrain_original_qhl(const float* input, int columns, float learning_rate,
                          float aspect_scale, float* output);

// Production-shaped entry point. Input planes are weights, gradient and
// previous momentum; output planes are only next weights and next momentum.
// The caller supplies private scratch so one allocation can be reused across
// all matrices assigned to a worker. No diagnostic taps or fine-grained
// timers are executed.
int hexatrain_original_qhl_release(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, void* scratch, size_t scratch_bytes);

// DSP-side timing accumulated with HAP_perf_get_time_us().  A profile is
// per canonical matrix unless the caller sums it across a group.
typedef struct hexatrain_qhl_profile {
  uint64_t total_us;
  uint64_t momentum_nesterov_us;
  uint64_t norm_normalization_us;
  uint64_t ns_us[5];
  uint64_t gemm_xxt_us[5];
  uint64_t gemm_a2_us[5];
  uint64_t gemm_bx_us[5];
  uint64_t vector_ops_us;
  uint64_t transpose_us;
  uint64_t final_update_us;
  uint32_t gemm_calls;
  uint32_t matrix_count;
} hexatrain_qhl_profile_t;

// Optional profiling/scratch entry point.  scratch_override is a caller-owned
// 128-byte-aligned buffer with at least the bytes returned by
// hexatrain_original_qhl_scratch_bytes(columns).  Passing NULL preserves the
// existing DDR allocation path.
size_t hexatrain_original_qhl_scratch_bytes(int columns);
int hexatrain_original_qhl_profiled(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, hexatrain_qhl_profile_t* profile,
    void* scratch_override, size_t scratch_bytes);
