// SPDX-License-Identifier: Apache-2.0
// Self-test for the Saved-P backward host reference (no device, no fixtures).
#include "saved_p_backward_reference_lib.h"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
  using namespace phonelm::saved_p_backward;
  std::vector<float> p(kTokens * kTokens, 0.0f), dp(kTokens * kTokens, 0.0f), dctx(kTokens * kHeadDim, 0.0f);
  for (std::uint32_t r = 0; r < kTokens; ++r) {
    double row_sum = 0.0;
    for (std::uint32_t c = 0; c <= r; ++c) { p[r * kTokens + c] = float(1.0 / double(r + 1)); row_sum += p[r * kTokens + c]; }
    if (std::fabs(row_sum - 1.0) > 1e-5) return 1;
    for (std::uint32_t c = 0; c < kTokens; ++c) dp[r * kTokens + c] = float(int((r * 31 + c * 17) % 13) - 6) * 0.05f;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) dctx[r * kHeadDim + d] = float(int((r * 7 + d * 13) % 11) - 5) * 0.07f;
  }
  auto a = SoftmaxBackwardDouble(p, dp), v = DValueDouble(p, dctx);
  auto b = SoftmaxBackwardOrderedFp32(p, dp), w = DValueOrderedFp32(p, dctx);
  if (a.size() != b.size() || v.size() != w.size()) return 2;
  double smax = 0.0, vmax = 0.0, srowsum = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) smax = std::fmax(smax, std::fabs(double(b[i]) - a[i]));
  for (std::size_t i = 0; i < w.size(); ++i) vmax = std::fmax(vmax, std::fabs(double(w[i]) - v[i]));
  for (std::uint32_t r = 0; r < kTokens; ++r) {
    double s = 0.0;
    for (std::uint32_t c = 0; c < kTokens; ++c) s += b[r * kTokens + c];
    srowsum = std::fmax(srowsum, std::fabs(s));
  }
  std::printf("saved_p_ref_selftest softmax_fp32_vs_double=%.3g dvalue_fp32_vs_double=%.3g softmax_rowsum=%.3g\n", smax, vmax, srowsum);
  if (!(std::isfinite(smax) && std::isfinite(vmax) && std::isfinite(srowsum))) return 3;
  if (smax > 1e-5 || vmax > 1e-4 || srowsum > 1e-5) return 4;
  std::puts("saved_p_backward_reference_selftest=PASS");
  return 0;
}
