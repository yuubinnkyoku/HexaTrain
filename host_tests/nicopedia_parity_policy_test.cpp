// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// CI-safe host tests for the parity re-audit policy (protocol:
// docs/qnn-nicopedia-htp-parity-policy.md sections 4 and 6).
//
// The header app/src/main/cpp/nicopedia_generation.h is shared between the
// device runtime and this test, so the metric formulas and every candidate
// policy verdict (L/P/C/D/F) are exercised here without a device.  The fault
// battery asserts the protocol's expected verdicts; a wrong verdict fails the
// host test (CI).
#include "nicopedia_generation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngen = phonelm::nicopedia_gen;

namespace {

void require(bool condition, const char *what) {
  if (!condition) throw std::runtime_error(std::string("ASSERT_FAILED: ") + what);
}

// Collects the fault battery verdicts for the CSV export.
struct FaultRow {
  std::string name;
  bool legacy;
  bool full;
  bool ambiguous;
  double rawMaxAbs, centeredMaxAbs, logSoftmaxMaxAbs, probMaxAbs, probL1,
      jsDivergence, scaleRatio, marginCpu;
};

// Deterministic synthetic reference row `z0`: top-1 margin 0.3
// (top1 = 0.5, top2 = 0.2, tail decays linearly; smooth spectrum).
std::vector<float> referenceRow(uint32_t vocab) {
  std::vector<float> z(vocab, 0.0f);
  for (uint32_t i = 0; i < vocab; ++i) {
    if (i == 0) z[i] = 0.5f;
    else if (i == 1) z[i] = 0.2f;
    else z[i] = 0.05f - 0.001f * static_cast<float>(i - 2);
  }
  return z;
}

struct FaultCase {
  const char *name;
  const std::vector<float> &w;  // HTP-side row under test
  bool expectedLegacy;          // legacy policy L verdict
  bool expectedFull;            // candidate policy F verdict
  bool expectedAmbiguous;       // decision_ambiguous must be set (only when F)
};
std::vector<FaultRow> g_rows;

void runFault(const FaultCase &c, const std::vector<float> &z, uint32_t V) {
  const ngen::ParityRowMetrics m =
      ngen::computeParityRowMetrics(z.data(), c.w.data(), V);
  const ngen::ParityPolicies p = ngen::evaluateParityPolicies(m);
  std::printf("fault=%-32s legacy=%d full=%d ambiguous=%d raw=%.3e "
              "centered=%.3e lsm=%.3e prob=%.3e l1=%.3e js=%.3e scale=%.4f "
              "margin=%.4f\n",
              c.name, p.legacy ? 1 : 0, p.full ? 1 : 0,
              m.decisionAmbiguous ? 1 : 0, m.rawMaxAbs, m.centeredMaxAbs,
              m.logSoftmaxMaxAbs, m.probMaxAbs, m.probL1, m.jsDivergence,
              m.scaleRatio, m.marginCpu);
  g_rows.push_back(FaultRow{c.name, p.legacy, p.full, m.decisionAmbiguous,
                            m.rawMaxAbs, m.centeredMaxAbs, m.logSoftmaxMaxAbs,
                            m.probMaxAbs, m.probL1, m.jsDivergence,
                            m.scaleRatio, m.marginCpu});
  require(p.legacy == c.expectedLegacy, c.name);
  require(p.full == c.expectedFull, c.name);
  if (c.expectedFull)
    require(m.decisionAmbiguous == c.expectedAmbiguous, c.name);
}

void testFaultBattery() {
  const uint32_t V = 256;
  const std::vector<float> z = referenceRow(V);
  require(z[0] - z[1] > 0.29 && z[0] - z[1] < 0.31, "ref row margin");

  // f1 small common offset (gauge): both gates pass.
  {
    std::vector<float> w = z;
    for (float &x : w) x += 1e-3f;
    runFault(FaultCase{"f1_common_offset_1e-3", w, true, true, false}, z, V);
  }
  // f2 medium common offset: gauge-free policy F passes, legacy L fails
  // (raw max 3e-2 > 2e-2).  Distribution metrics are unchanged by design.
  {
    std::vector<float> w = z;
    for (float &x : w) x += 3e-2f;
    runFault(FaultCase{"f2_common_offset_3e-2", w, false, true, false}, z, V);
  }
  // f3 large common offset 0.5: fails the catastrophic raw bound in F.
  {
    std::vector<float> w = z;
    for (float &x : w) x += 5e-1f;
    runFault(FaultCase{"f3_common_offset_5e-1", w, false, false, false}, z, V);
  }
  // f4 single logit clamped: shapes worse; both gates fail.
  {
    std::vector<float> w = z;
    w[100] += 5e-2f;
    runFault(FaultCase{"f4_single_logit_5e-2", w, false, false, false}, z, V);
  }
  // f5 top1/top2 flip with small delta (raw 3e-2) at decisive CPU margin 0.4:
  // margin > 2 x raw -> decision rule FAILs; legacy also FAILs (raw >= 2e-2).
  {
    std::vector<float> z5 = z;
    z5[0] = 0.6f;  // reference with 0.4-deciding margin (0.6 vs 0.2)
    std::vector<float> w5 = z5;
    w5[1] = 0.63f;  // small perturbation flips the argmax (raw delta 3e-2)
    runFault(FaultCase{"f5_swap_margin_4e-1", w5, false, false, false}, z5, V);
  }
  // f6 top1/top2 swap at thin margin (5e-3): decision rule marks the row
  // ambiguous (margin <= 2 * raw) and F records decision_ambiguous instead
  // of FAILing (documented semantics).
  {
    std::vector<float> z6 = z;
    z6[0] = 0.505f;
    z6[1] = 0.5f;  // CPU top1-top2 margin 5e-3
    std::vector<float> w6 = z6;
    std::swap(w6[0], w6[1]);
    runFault(FaultCase{"f6_swap_margin_5e-3", w6, true, true, true}, z6, V);
  }
  // f7 scale x1.02: F fails on scale_ratio.
  {
    std::vector<float> w = z;
    for (float &x : w) x *= 1.02f;
    runFault(FaultCase{"f7_scale_1.02", w, true, false, false}, z, V);
  }
  // f8 scale x1.004: in-band, both pass.
  {
    std::vector<float> w = z;
    for (float &x : w) x *= 1.004f;
    runFault(FaultCase{"f8_scale_1.004", w, true, true, false}, z, V);
  }
  // f9 gaussian noise sigma 1e-3: both pass.
  {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1e-3f);
    std::vector<float> w = z;
    for (float &x : w) x += nd(rng);
    runFault(FaultCase{"f9_noise_sigma1e-3", w, true, true, false}, z, V);
  }
  // f10 gaussian noise sigma 1e-1: both fail (prob distortion).
  {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1e-1f);
    std::vector<float> w = z;
    for (float &x : w) x += nd(rng);
    runFault(FaultCase{"f10_noise_sigma1e-1", w, false, false, false}, z, V);
  }
  // f11 mass redistribution (interior logits scaled down): prob distorted.
  {
    std::vector<float> w = z;
    for (size_t i = 2; i < w.size(); ++i) w[i] *= 0.05f;
    runFault(FaultCase{"f11_mass_redistribution", w, false, false, false}, z,
             V);
  }
  // f12/f13/f14 non-finite: every policy fails.
  {
    std::vector<float> w = z;
    w[50] = std::numeric_limits<float>::quiet_NaN();
    runFault(FaultCase{"f12_nan_element", w, false, false, false}, z, V);
    w = z;
    w[50] = std::numeric_limits<float>::infinity();
    runFault(FaultCase{"f13_inf_element", w, false, false, false}, z, V);
    w = z;
    w[50] = -std::numeric_limits<float>::infinity();
    runFault(FaultCase{"f14_ninf_element", w, false, false, false}, z, V);
  }
  // f15 all-zero row: degenerate, both fail.
  {
    const std::vector<float> zero(V, 0.0f);
    runFault(FaultCase{"f15_all_zero_row", zero, false, false, false}, zero, V);
  }
}

void testMetricInvariants() {
  const uint32_t V = 256;
  const std::vector<float> z = referenceRow(V);
  std::vector<float> w = z;
  for (float &x : w) x += 0.02f;  // gauge offset 2e-2
  const ngen::ParityRowMetrics m1 =
      ngen::computeParityRowMetrics(z.data(), z.data(), V);
  const ngen::ParityRowMetrics m2 =
      ngen::computeParityRowMetrics(z.data(), w.data(), V);
  // Gauge-block invariance: prob/JS/logsoftmax/centered/scaling are all
  // unchanged by a pure common offset; raw max-abs moves exactly 2e-2.
  // Inputs are float32, so invariance holds to float rounding (~1e-10 for
  // softmax quantities at this scale), not double precision.
  const double gaugeTol = 1e-6;
  require(std::fabs(m2.probMaxAbs) < gaugeTol, "gauge inv prob_max_abs");
  require(std::fabs(m2.probL1) < gaugeTol, "gauge inv prob_l1");
  require(std::fabs(m2.jsDivergence) < gaugeTol, "gauge inv js");
  require(std::fabs(m2.logSoftmaxMaxAbs) < gaugeTol, "gauge inv lsm");
  require(std::fabs(m2.centeredMaxAbs) < gaugeTol, "gauge inv centered_max");
  require(std::fabs(m2.centeredRms) < gaugeTol, "gauge inv centered_rms");
  require(std::fabs(m2.scaleRatio - m1.scaleRatio) < 1e-4, "gauge inv scale");
  require(std::fabs(m1.scaleRatio - 1.0) < 1e-6, "identity scale ratio 1");
  require(std::fabs(m2.rawMaxAbs - 2e-2) < 1e-4, "raw moves with offset");
  require(std::fabs(m2.probMeanAbs) < gaugeTol, "gauge inv prob_mean");
  require(m2.argmaxMatch, "gauge inv argmax");
  require(!m2.decisionAmbiguous, "centsafe: no ambiguous on match");

  // Decision-margin algebra: guarantee holds when margin > 2*raw.
  {
    std::vector<float> w = z;
    w[100] += 1e-3f;  // raw <= 1e-3, margin 0.3 -> guaranteed
    const ngen::ParityRowMetrics m = ngen::computeParityRowMetrics(
        z.data(), w.data(), V);
    require(m.argmaxMatch && m.marginCpu > 2 * m.rawMaxAbs,
            "decision guarantee holds");
  }
  // Top-5 set overlap and order match on identity.
  require(m1.topkSetOverlap == 5 && m1.topkOrderMatch, "identity top-5 match");
  require(m1.topkSetSize == 5, "top-5 default size");
  // Prefix count for the device parity audit: 20 fixed prefixes.
  require(ngen::parityPrefixes().size() == 20, "prefix count 20");
  // Labels unique.
  {
    const auto &prefixes = ngen::parityPrefixes();
    for (size_t i = 0; i < prefixes.size(); ++i)
      for (size_t j = i + 1; j < prefixes.size(); ++j)
        require(std::string(prefixes[i].label) != std::string(prefixes[j].label),
                "prefix label uniqueness");
  }
  // Diagnostic tap configuration defaults must keep the production parity
  // path unchanged (gate semantics are independent of the private taps).
  {
    const ngen::GenerateConfig config;
    require(config.diagnosticTapScope == "NONE", "default tap scope NONE");
    require(config.diagnosticLayerIndex == 0xffffffffu,
            "default layer index sentinel");
    const ngen::NprtTapMetric metric;
    require(metric.name.empty() && metric.diffRms == 0 && metric.relRms == 0,
            "default tap metric zeroed");
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    testMetricInvariants();
    std::printf("=== parity policy candidate fault battery ===\n");
    testFaultBattery();
    // CSV summary (consumed by the public exporter; also printed).
    if (argc > 1) {
      std::FILE *out = std::fopen(argv[1], "w");
      require(out != nullptr, "open csv output");
      std::fprintf(out,
                   "fault,legacy_policy,candidate_full_policy,"
                   "decision_ambiguous,raw_max_abs,centered_max_abs,"
                   "logsoftmax_max_abs,prob_max_abs,prob_l1,js_divergence,"
                   "scale_ratio,margin_cpu\n");
      for (const auto &r : g_rows)
        std::fprintf(out,
                     "%s,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9f,%.6f\n",
                     r.name.c_str(), r.legacy ? 1 : 0, r.full ? 1 : 0,
                     r.ambiguous ? 1 : 0, r.rawMaxAbs, r.centeredMaxAbs,
                     r.logSoftmaxMaxAbs, r.probMaxAbs, r.probL1, r.jsDivergence,
                     r.scaleRatio, r.marginCpu);
      // Re-emit for the console (assertion of expected values still above).
      std::fflush(out);
      std::fclose(out);
    }
    std::printf("nicopedia_parity_policy_test=PASS\n");
    return 0;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "nicopedia_parity_policy_test=FAIL %s\n",
                 exception.what());
    return 1;
  }
}