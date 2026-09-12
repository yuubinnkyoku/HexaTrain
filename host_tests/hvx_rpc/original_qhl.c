// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "original_qhl.h"

#include <HAP_perf.h>
#include <qhblas_hvx.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int all_finite(const float* values, int count) {
  for (int i = 0; i < count; ++i) if (!isfinite(values[i])) return 0;
  return 1;
}

static uint64_t qhl_now_us(void) {
  return (uint64_t)HAP_perf_get_time_us();
}

size_t hexatrain_original_qhl_scratch_bytes(int columns) {
  if (columns != 64 && columns != 128) return 0;
  const size_t elements = (size_t)64 * (size_t)columns;
  return ((size_t)5 * elements + (size_t)3 * 4096u) * sizeof(float);
}

static int hexatrain_original_qhl_impl(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, hexatrain_qhl_profile_t* profile,
    void* scratch_override, size_t scratch_bytes, int diagnostic_outputs) {
  if (profile) memset(profile, 0, sizeof(*profile));
  if (!input || !output || (columns != 64 && columns != 128) ||
      !isfinite(learning_rate) || learning_rate <= 0 ||
      !isfinite(aspect_scale) || aspect_scale <= 0) return 1101;
  const int elements = 64 * columns;
  if (!all_finite(input, 3 * elements)) return 1102;
  const size_t required_scratch = hexatrain_original_qhl_scratch_bytes(columns);
  if (scratch_override && scratch_bytes < required_scratch) return 1103;
  const uint64_t total_start = profile ? qhl_now_us() : 0;
  const size_t count = (size_t)5 * (size_t)elements + (size_t)3 * 4096u;
  int owns_scratch = 0;
  float* scratch = (float*)scratch_override;
  if (!scratch) {
    scratch = (float*)memalign(128, required_scratch);
    owns_scratch = 1;
  }
  if (!scratch) return 1103;
  if (diagnostic_outputs)
    for (size_t i = 0; i < count; ++i) scratch[i] = NAN;
  float* x = scratch;
  float* xt = x + elements;
  float* bx = xt + elements;
  float* t1 = bx + elements;
  float* t2 = t1 + elements;
  float* a = t2 + elements;
  float* a2 = a + 4096;
  float* b = a2 + 4096;
  float* next_m = output + elements;
  const float momentum = .95f;
  const float gradient_coefficient = 1.0f - momentum;
  int status = 0;
  float norm = NAN;
  uint64_t stage_start = 0;
  uint64_t gemm_xxt_us[5] = {0};
  uint64_t gemm_a2_us[5] = {0};
  uint64_t gemm_bx_us[5] = {0};
  uint32_t gemm_calls = 0;

  // Immediate checks keep operation failure and nonfinite classifications
  // distinct. Vector and GEMM timers are separate from semantic stage timers
  // so their totals can be reported without double-counting.
#define CALL(expression) do { status = (expression); if (status) goto done; } while (0)
#define CALL_VECTOR(expression) do { \
    const uint64_t _vector_start = profile ? qhl_now_us() : 0; \
    status = (expression); \
    if (profile) profile->vector_ops_us += qhl_now_us() - _vector_start; \
    if (status) goto done; \
  } while (0)
#define CALL_GEMM(expression, bucket) do { \
    const uint64_t _gemm_start = profile ? qhl_now_us() : 0; \
    status = (expression); \
    if (profile) { \
      (bucket) += qhl_now_us() - _gemm_start; \
      ++gemm_calls; \
    } \
    if (status) goto done; \
  } while (0)
#define FINITE(values, size) do { \
    if (!all_finite((values), (size))) { status = 1104; goto done; } \
  } while (0)

  if (profile) stage_start = qhl_now_us();
  CALL_VECTOR(qhblas_hvx_vector_scaling_af(
      (float*)input + 2 * elements, momentum, t1, elements));
  CALL_VECTOR(qhblas_hvx_vector_scaling_af(
      (float*)input + elements, gradient_coefficient, t2, elements));
  CALL_VECTOR(qhblas_hvx_vector_add_af(t1, t2, next_m, elements));
  FINITE(next_m, elements);
  CALL_VECTOR(qhblas_hvx_vector_scaling_af(next_m, momentum, t1, elements));
  CALL_VECTOR(qhblas_hvx_vector_add_af(t2, t1, x, elements));
  FINITE(x, elements);
  if (profile) profile->momentum_nesterov_us += qhl_now_us() - stage_start;

  if (profile) stage_start = qhl_now_us();
  CALL_VECTOR(qhblas_hvx_f_vector_norm_af(x, &norm, elements));
  if (!isfinite(norm) || norm < 0 || !isfinite(norm + 1.0e-7f)) {
    status = 1105;
    goto done;
  }
  CALL_VECTOR(qhblas_hvx_vector_scaling_af(
      x, 1.0f / (norm + 1.0e-7f), t1, elements));
  memcpy(x, t1, (size_t)elements * sizeof(float));
  FINITE(x, elements);
  if (diagnostic_outputs)
    memcpy(output + 2 * elements, x, (size_t)elements * sizeof(float));
  if (profile) profile->norm_normalization_us += qhl_now_us() - stage_start;

  for (int iteration = 0; iteration < 5; ++iteration) {
    if (profile) stage_start = qhl_now_us();
    const uint64_t transpose_start = profile ? qhl_now_us() : 0;
    for (int row = 0; row < 64; ++row)
      for (int col = 0; col < columns; ++col)
        xt[col * 64 + row] = x[row * columns + col];
    if (profile) profile->transpose_us += qhl_now_us() - transpose_start;
    CALL_GEMM(qhblas_hvx_matrix_matrix_mpy_af(
                  x, xt, a, 64, columns, 64),
              gemm_xxt_us[iteration]);
    FINITE(a, 4096);
    // A @ A requires separate pointers to satisfy the SDK restrict contract.
    memcpy(b, a, 4096 * sizeof(float));
    CALL_GEMM(qhblas_hvx_matrix_matrix_mpy_af(
                  a, b, a2, 64, 64, 64),
              gemm_a2_us[iteration]);
    FINITE(a2, 4096);
    CALL_VECTOR(qhblas_hvx_vector_scaling_af(a, -4.7750f, t1, 4096));
    CALL_VECTOR(qhblas_hvx_vector_scaling_af(a2, 2.0315f, t2, 4096));
    CALL_VECTOR(qhblas_hvx_vector_add_af(t1, t2, b, 4096));
    FINITE(b, 4096);
    CALL_GEMM(qhblas_hvx_matrix_matrix_mpy_af(
                  b, x, bx, 64, 64, columns),
              gemm_bx_us[iteration]);
    FINITE(bx, elements);
    CALL_VECTOR(qhblas_hvx_vector_scaling_af(x, 3.4445f, t1, elements));
    CALL_VECTOR(qhblas_hvx_vector_add_af(t1, bx, x, elements));
    FINITE(x, elements);
    if (profile) profile->ns_us[iteration] += qhl_now_us() - stage_start;
    if (diagnostic_outputs)
      memcpy(output + (3 + iteration) * elements, x,
             (size_t)elements * sizeof(float));
  }

  if (profile) stage_start = qhl_now_us();
  CALL_VECTOR(qhblas_hvx_vector_scaling_af(
      x, learning_rate * aspect_scale, t1, elements));
  CALL_VECTOR(qhblas_hvx_vector_sub_af((float*)input, t1, output, elements));
  FINITE(output, elements);
  if (profile) profile->final_update_us += qhl_now_us() - stage_start;

done:
  if (profile) {
    profile->total_us = qhl_now_us() - total_start;
    memcpy(profile->gemm_xxt_us, gemm_xxt_us, sizeof(gemm_xxt_us));
    memcpy(profile->gemm_a2_us, gemm_a2_us, sizeof(gemm_a2_us));
    memcpy(profile->gemm_bx_us, gemm_bx_us, sizeof(gemm_bx_us));
    profile->gemm_calls = gemm_calls;
    profile->matrix_count = 1;
  }
  if (owns_scratch) free(scratch);
#undef CALL
#undef CALL_VECTOR
#undef CALL_GEMM
#undef FINITE
  return status;
}

int hexatrain_original_qhl_profiled(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, hexatrain_qhl_profile_t* profile,
    void* scratch_override, size_t scratch_bytes) {
  return hexatrain_original_qhl_impl(
      input, columns, learning_rate, aspect_scale, output, profile,
      scratch_override, scratch_bytes, 1);
}

int hexatrain_original_qhl_release(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, void* scratch, size_t scratch_bytes) {
  if (!scratch) return 1103;
  return hexatrain_original_qhl_impl(
      input, columns, learning_rate, aspect_scale, output, NULL,
      scratch, scratch_bytes, 0);
}

int hexatrain_original_qhl_release_profiled(
    const float* input, int columns, float learning_rate, float aspect_scale,
    float* output, hexatrain_qhl_profile_t* profile,
    void* scratch, size_t scratch_bytes) {
  if (!scratch || !profile) return 1103;
  return hexatrain_original_qhl_impl(
      input, columns, learning_rate, aspect_scale, output, profile,
      scratch, scratch_bytes, 0);
}

int hexatrain_original_qhl(const float* input, int columns, float learning_rate,
                           float aspect_scale, float* output) {
  return hexatrain_original_qhl_profiled(
      input, columns, learning_rate, aspect_scale, output, NULL, NULL, 0);
}
