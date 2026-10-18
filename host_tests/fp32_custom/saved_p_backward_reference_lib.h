// SPDX-License-Identifier: Apache-2.0
// Saved-P backward native-plan host reference (T=32, Dh=32).
// Reference A: independent double. Reference B: ordered FP32 replay of the
// exact QNN node order in qnn_runtime_transformer_training_generalized.inc.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace phonelm::saved_p_backward {
inline constexpr std::uint32_t kTokens = 32;
inline constexpr std::uint32_t kHeadDim = 32;
inline std::vector<double> SoftmaxBackwardDouble(const std::vector<float>& p, const std::vector<float>& dp) {
  std::vector<double> out(p.size(), 0.0);
  for (std::uint32_t r = 0; r < kTokens; ++r) {
    double dot = 0.0;
    for (std::uint32_t c = 0; c < kTokens; ++c) dot += double(p[r * kTokens + c]) * double(dp[r * kTokens + c]);
    for (std::uint32_t c = 0; c < kTokens; ++c) out[r * kTokens + c] = double(p[r * kTokens + c]) * (double(dp[r * kTokens + c]) - dot);
  }
  return out;
}
inline std::vector<double> DValueDouble(const std::vector<float>& p, const std::vector<float>& dctx) {
  std::vector<double> out(kTokens * kHeadDim, 0.0);
  for (std::uint32_t r = 0; r < kTokens; ++r)
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      double acc = 0.0;
      for (std::uint32_t c = 0; c < kTokens; ++c) acc += double(p[c * kTokens + r]) * double(dctx[c * kHeadDim + d]);
      out[r * kHeadDim + d] = acc;
    }
  return out;
}
inline std::vector<float> SoftmaxBackwardOrderedFp32(const std::vector<float>& p, const std::vector<float>& dp) {
  std::vector<float> product(p.size()), centered(p.size()), out(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) product[i] = dp[i] * p[i];
  for (std::uint32_t r = 0; r < kTokens; ++r) {
    float dot = 0.0f;
    for (std::uint32_t c = 0; c < kTokens; ++c) dot = dot + product[r * kTokens + c];
    for (std::uint32_t c = 0; c < kTokens; ++c) centered[r * kTokens + c] = dp[r * kTokens + c] - dot;
  }
  for (std::size_t i = 0; i < p.size(); ++i) out[i] = p[i] * centered[i];
  return out;
}
inline std::vector<float> DValueOrderedFp32(const std::vector<float>& p, const std::vector<float>& dctx) {
  std::vector<float> out(kTokens * kHeadDim, 0.0f);
  for (std::uint32_t r = 0; r < kTokens; ++r)
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      float acc = 0.0f;
      for (std::uint32_t c = 0; c < kTokens; ++c) acc = acc + (p[c * kTokens + r] * dctx[c * kHeadDim + d]);
      out[r * kHeadDim + d] = acc;
    }
  return out;
}
}  // namespace phonelm::saved_p_backward
