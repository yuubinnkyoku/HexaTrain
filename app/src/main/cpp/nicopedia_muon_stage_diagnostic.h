// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

// Diagnostic-only Newton-Schulz single-iteration stage references.
// The CPU oracle (nicopedia_muon::zeropowerNewtonSchulzFp32) is unchanged.
// These helpers replicate the oracle evaluation order exactly:
//
//   A  = X0 @ X0^T
//   A2 = A @ A
//   B  = b*A + c*A2
//   BX = B @ X0
//   X1 = a*X0 + BX
//
// DOUBLE accumulates each dot product in double and rounds each stage to
// float (matching the oracle). FLOAT performs each multiply, add, and
// accumulation step in float (matching the HTP graph node order:
// b*A, c*A2, B=bA+cA2, BX=B@X, a*X, X1=aX+BX).

#include "nicopedia_muon_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace phonelm::nicopedia_muon_stage {

struct Stages {
  std::vector<float> x0;
  std::vector<float> a;
  std::vector<float> a2;
  std::vector<float> b;
  std::vector<float> bx;
  std::vector<float> x1;
};

inline bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

inline bool finiteVector(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

inline void matXXTDouble(const std::vector<float>& x, std::uint32_t rows,
                         std::uint32_t cols, std::vector<float>* a) {
  a->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double sum = 0.0;
      for (std::uint32_t k = 0; k < cols; ++k)
        sum += double(x[std::size_t(i) * cols + k]) *
               x[std::size_t(j) * cols + k];
      (*a)[std::size_t(i) * rows + j] = float(sum);
    }
}

inline void matSqDouble(const std::vector<float>& a, std::uint32_t rows,
                        std::vector<float>* a2) {
  a2->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      double sum = 0.0;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += double(a[std::size_t(i) * rows + k]) *
               a[std::size_t(k) * rows + j];
      (*a2)[std::size_t(i) * rows + j] = float(sum);
    }
}

inline void matXXTFloat(const std::vector<float>& x, std::uint32_t rows,
                        std::uint32_t cols, std::vector<float>* a) {
  a->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < cols; ++k)
        sum += x[std::size_t(i) * cols + k] * x[std::size_t(j) * cols + k];
      (*a)[std::size_t(i) * rows + j] = sum;
    }
}

inline void matSqFloat(const std::vector<float>& a, std::uint32_t rows,
                       std::vector<float>* a2) {
  a2->assign(std::size_t(rows) * rows, 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < rows; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += a[std::size_t(i) * rows + k] * a[std::size_t(k) * rows + j];
      (*a2)[std::size_t(i) * rows + j] = sum;
    }
}

inline Stages stepDoubleFromX0(const std::vector<float>& x0,
                               std::uint32_t rows, std::uint32_t cols) {
  using phonelm::nicopedia_muon::kNsA;
  using phonelm::nicopedia_muon::kNsB;
  using phonelm::nicopedia_muon::kNsC;
  Stages s;
  s.x0 = x0;
  matXXTDouble(x0, rows, cols, &s.a);
  matSqDouble(s.a, rows, &s.a2);
  s.b.assign(s.a.size(), 0.0f);
  for (std::size_t i = 0; i < s.b.size(); ++i)
    s.b[i] = float(double(kNsB) * s.a[i] + double(kNsC) * s.a2[i]);
  s.bx.assign(x0.size(), 0.0f);
  s.x1.assign(x0.size(), 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < cols; ++j) {
      double bx = 0.0;
      for (std::uint32_t k = 0; k < rows; ++k) {
        const double b = double(kNsB) * s.a[std::size_t(i) * rows + k] +
                         double(kNsC) * s.a2[std::size_t(i) * rows + k];
        bx += b * x0[std::size_t(k) * cols + j];
      }
      s.bx[std::size_t(i) * cols + j] = float(bx);
      s.x1[std::size_t(i) * cols + j] =
          float(double(kNsA) * x0[std::size_t(i) * cols + j] + bx);
    }
  return s;
}

inline Stages stepFloatFromX0(const std::vector<float>& x0,
                              std::uint32_t rows, std::uint32_t cols) {
  using phonelm::nicopedia_muon::kNsA;
  using phonelm::nicopedia_muon::kNsB;
  using phonelm::nicopedia_muon::kNsC;
  Stages s;
  s.x0 = x0;
  matXXTFloat(x0, rows, cols, &s.a);
  matSqFloat(s.a, rows, &s.a2);
  s.b.assign(s.a.size(), 0.0f);
  for (std::size_t i = 0; i < s.b.size(); ++i) {
    const float t1 = kNsB * s.a[i];
    const float t2 = kNsC * s.a2[i];
    s.b[i] = t1 + t2;
  }
  s.bx.assign(x0.size(), 0.0f);
  for (std::uint32_t i = 0; i < rows; ++i)
    for (std::uint32_t j = 0; j < cols; ++j) {
      float sum = 0.0f;
      for (std::uint32_t k = 0; k < rows; ++k)
        sum += s.b[std::size_t(i) * rows + k] * x0[std::size_t(k) * cols + j];
      s.bx[std::size_t(i) * cols + j] = sum;
    }
  s.x1.assign(x0.size(), 0.0f);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    const float ax = kNsA * x0[i];
    s.x1[i] = ax + s.bx[i];
  }
  return s;
}

inline bool normalizeFromNesterov(const std::vector<float>& nesterov,
                                  std::vector<float>* x0,
                                  std::string* error = nullptr) {
  if (!x0 || nesterov.empty()) return fail(error, "STAGE_INPUT_INVALID");
  if (!finiteVector(nesterov)) return fail(error, "STAGE_INPUT_NONFINITE");
  double normSquared = 0.0;
  for (float value : nesterov) normSquared += double(value) * value;
  const double denominator =
      std::sqrt(normSquared) + phonelm::nicopedia_muon::kNsEpsilon;
  if (!std::isfinite(denominator) || denominator <= 0.0)
    return fail(error, "STAGE_NORMALIZATION_INVALID");
  x0->assign(nesterov.size(), 0.0f);
  for (std::size_t i = 0; i < nesterov.size(); ++i)
    (*x0)[i] = float(double(nesterov[i]) / denominator);
  if (!finiteVector(*x0)) return fail(error, "STAGE_NORMALIZED_NONFINITE");
  return true;
}

inline bool stagesFromNesterovDouble(const std::vector<float>& nesterov,
                                     std::uint32_t rows, std::uint32_t cols,
                                     Stages* out,
                                     std::string* error = nullptr) {
  if (!out || rows == 0 || cols == 0 ||
      std::uint64_t(rows) * cols != nesterov.size())
    return fail(error, "STAGE_SHAPE_INVALID");
  if (rows > cols) return fail(error, "STAGE_TRANSPOSED_UNSUPPORTED");
  std::vector<float> x0;
  if (!normalizeFromNesterov(nesterov, &x0, error)) return false;
  *out = stepDoubleFromX0(x0, rows, cols);
  return true;
}

inline bool stagesFromNesterovFloat(const std::vector<float>& nesterov,
                                    std::uint32_t rows, std::uint32_t cols,
                                    Stages* out,
                                    std::string* error = nullptr) {
  if (!out || rows == 0 || cols == 0 ||
      std::uint64_t(rows) * cols != nesterov.size())
    return fail(error, "STAGE_SHAPE_INVALID");
  if (rows > cols) return fail(error, "STAGE_TRANSPOSED_UNSUPPORTED");
  std::vector<float> x0;
  if (!normalizeFromNesterov(nesterov, &x0, error)) return false;
  *out = stepFloatFromX0(x0, rows, cols);
  return true;
}

inline std::uint16_t f32ToF16Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::int32_t exp = std::int32_t((bits >> 23) & 0xffu) - 112;
  const std::uint32_t mant = bits & 0x7fffffu;
  if (exp >= 31) {
    if (mant == 0) return std::uint16_t(sign | 0x7c00u);
    return std::uint16_t(sign | 0x7c00u | (mant >> 13) | (mant ? 1u : 0u));
  }
  if (exp <= 0) {
    if (exp < -10) return std::uint16_t(sign);
    const std::uint32_t shifted = (mant | 0x800000u) >> (1 - exp);
    const std::uint32_t rounded =
        (shifted + 0xfffu + ((shifted >> 13) & 1u)) >> 13;
    return std::uint16_t(sign | rounded);
  }
  const std::uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
  if (rounded & 0x800000u) return std::uint16_t(sign | ((exp + 1) << 10));
  return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (rounded >> 13));
}

inline float f16BitsToF32(std::uint16_t value) {
  std::uint32_t sign = (std::uint32_t(value & 0x8000u)) << 16;
  std::uint32_t exp = (value >> 10) & 0x1fu;
  std::uint32_t mant = value & 0x3ffu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ffu;
      bits = sign | ((std::uint32_t(exp + 112)) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline float quantizeHalf(float value) {
  return f16BitsToF32(f32ToF16Bits(value));
}

inline void transposeMatrix(const std::vector<float>& input, std::uint32_t rows,
                            std::uint32_t cols, std::vector<float>* output) {
  output->assign(input.size(), 0.0f);
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t column = 0; column < cols; ++column)
      (*output)[std::size_t(column) * rows + row] =
          input[std::size_t(row) * cols + column];
}

inline std::uint64_t fnv1a64(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace phonelm::nicopedia_muon_stage
