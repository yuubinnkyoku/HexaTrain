// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Independent local research probe. QHL implementation remains in the SDK.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <hexagon_sim_timer.h>
#include <qhblas_hvx.h>

#if !defined(__HEXAGON_ARCH__) || __HEXAGON_ARCH__ != 81
#error This probe requires the explicitly selected V81 target.
#endif

#define CAPACITY (64 * 128)
static float left[CAPACITY] __attribute__((aligned(128)));
static float right[CAPACITY] __attribute__((aligned(128)));
static float output[CAPACITY] __attribute__((aligned(128)));

static int compare(uint32_t m, uint32_t k, uint32_t n, int use_float) {
  double max_abs = 0, abs_sum = 0, diff2 = 0, ref2 = 0, out2 = 0, dot = 0;
  int finite = 1;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      double sum = 0;
      float float_sum = 0;
      for (uint32_t p = 0; p < k; ++p) {
        if (use_float) {
          // Separate multiplication/addition; compiler contraction is disabled.
          float_sum += left[i * k + p] * right[p * n + j];
        } else {
          sum += (double)left[i * k + p] * right[p * n + j];
        }
      }
      const double ref = use_float ? float_sum : (float)sum;
      const double actual = output[i * n + j];
      const double difference = actual - ref;
      if (!isfinite(ref) || !isfinite(actual)) finite = 0;
      const double error = fabs(difference);
      if (error > max_abs) max_abs = error;
      abs_sum += error;
      diff2 += difference * difference;
      ref2 += ref * ref;
      out2 += actual * actual;
      dot += ref * actual;
    }
  }
  const double rel = ref2 > 0 ? sqrt(diff2 / ref2) : (diff2 == 0 ? 0 : INFINITY);
  const double cosine = ref2 > 0 && out2 > 0 ? dot / sqrt(ref2 * out2) : 0;
  const int pass = finite && max_abs <= .002 && rel <= .001 && cosine >= .99999;
  printf("reference=%s maxAbs=%.12g meanAbs=%.12g RMS=%.12g relativeL2=%.12g cosine=%.12g finite=%d gate=%s\n",
         use_float ? "CPU_FLOAT" : "CPU_DOUBLE", max_abs, abs_sum / (m * n),
         sqrt(diff2 / (m * n)), rel, cosine, finite, pass ? "PASS" : "FAIL");
  return pass;
}

int main(int argc, char** argv) {
  uint32_t m = 32, k = 32, n = 32;
  if (argc == 1) {
    // Exact identity product, including negative values and nonzero off-diagonals.
    for (uint32_t i = 0; i < 32; ++i) {
      left[i * 32 + i] = 1;
      for (uint32_t j = 0; j < 32; ++j)
        right[i * 32 + j] = (float)((int)((i * 7 + j * 3) % 31) - 15) / 32;
    }
  } else if (argc == 3) {
    m = k = n = 64;
    FILE* input = fopen(argv[1], "rb");
    if (!input) return 10;
    const size_t count = fread(left, sizeof(float), m * k, input);
    const int extra = fgetc(input);
    fclose(input);
    if (count != m * k || extra != EOF) return 11;
    for (uint32_t i = 0; i < m; ++i) {
      for (uint32_t j = 0; j < k; ++j) {
        if (!isfinite(left[i * k + j])) return 12;
        right[j * m + i] = left[i * k + j];
      }
    }
  } else {
    return 13;
  }
  for (uint32_t i = 0; i < m * n; ++i) output[i] = NAN;
  const int aligned = ((uintptr_t)left % 128 == 0) &&
      ((uintptr_t)right % 128 == 0) && ((uintptr_t)output % 128 == 0);
  if (!aligned || n % 32 != 0) return 14;
  const uint64_t begin = hexagon_sim_read_pcycles();
  const int32_t status = qhblas_hvx_matrix_matrix_mpy_af(left, right, output, m, k, n);
  const uint64_t cycles = hexagon_sim_read_pcycles() - begin;
  printf("backend=QHL_HVX_SIMULATOR target=v81 vector_bytes=128 internal_arithmetic=QFloat32 fallback=false\n");
  printf("m=%u k=%u n=%u fast_path_conditions=%d kernel_status=%ld cycles=%llu\n",
         (unsigned)m, (unsigned)k, (unsigned)n, aligned,
         (long)status, (unsigned long long)cycles);
  if (status != 0) return 15;
  const int double_pass = compare(m, k, n, 0);
  const int float_pass = compare(m, k, n, 1);
  if (argc == 1) {
    if (memcmp(output, right, 32 * 32 * sizeof(float)) != 0) return 16;
    printf("identity_product_bit_exact=true\n");
  } else {
    FILE* result = fopen(argv[2], "wb");
    if (!result) return 17;
    const size_t written = fwrite(output, sizeof(float), m * n, result);
    const int closed = fclose(result);
    if (written != m * n || closed != 0) return 18;
  }
  return double_pass && float_pass ? 0 : 19;
}
