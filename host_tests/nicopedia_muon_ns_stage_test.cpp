// SPDX-License-Identifier: Apache-2.0
// Diagnostic-only Newton-Schulz stage self-test (host side, no QNN).
// Verifies the shared stage header against the frozen CPU oracle and the
// frozen 64x64 normalized-input identity before any device run.
#include "nicopedia_muon_optimizer.h"
#include "nicopedia_muon_stage_diagnostic.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> makeValues(std::size_t count, float scale, int phase) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = scale *
        (std::sin(float(index + 1) * (0.013f + phase * 0.001f)) +
         float(int(index % 17) - 8) * 0.03125f);
  return result;
}

double relativeL2(const std::vector<float>& expected,
                  const std::vector<float>& actual) {
  require(expected.size() == actual.size() && !expected.empty(),
          "metric size");
  double squared = 0.0, expectedSquared = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double d = double(actual[i]) - expected[i];
    squared += d * d;
    expectedSquared += double(expected[i]) * expected[i];
  }
  return std::sqrt(squared) / std::max(std::sqrt(expectedSquared), 1.0e-30);
}

}  // namespace

int main() {
  try {
    using namespace phonelm;
    constexpr std::uint32_t kRows = 64, kColumns = 64;
    const auto gradient = makeValues(kRows * kColumns, 0.001f, 2);
    const auto prior = makeValues(kRows * kColumns, 0.0002f, 3);
    std::vector<float> nesterov(gradient.size());
    for (std::size_t i = 0; i < nesterov.size(); ++i) {
      const float nextMomentum = 0.95f * prior[i] + 0.05f * gradient[i];
      nesterov[i] = 0.05f * gradient[i] + 0.95f * nextMomentum;
    }
    nicopedia_muon_stage::Stages stages;
    std::string error;
    require(nicopedia_muon_stage::stagesFromNesterovDouble(
                nesterov, kRows, kColumns, &stages, &error),
            error.c_str());
    const std::uint64_t fnv = nicopedia_muon_stage::fnv1a64(
        stages.x0.data(), stages.x0.size() * sizeof(float));
    char fnvText[17];
    std::snprintf(fnvText, sizeof(fnvText), "%016llx",
                  static_cast<unsigned long long>(fnv));
    require(std::string(fnvText) == "70a2b57657bea1a0",
            "frozen input identity changed");
    std::vector<float> oracle1;
    require(nicopedia_muon::zeropowerNewtonSchulzFp32(
                nesterov, kRows, kColumns, 1, &oracle1, nullptr, &error),
            error.c_str());
    require(relativeL2(oracle1, stages.x1) <= 1.0e-9,
            "double stages diverge from oracle");
    nicopedia_muon_stage::Stages floatStages;
    require(nicopedia_muon_stage::stagesFromNesterovFloat(
                nesterov, kRows, kColumns, &floatStages, &error),
            error.c_str());
    // Sanity bounds from the frozen host diagnostic: float-vs-double X1 is
    // ~2.4e-07, so anything above 1e-5 here means the shared header drifted
    // from the audited reference implementation.
    require(relativeL2(stages.x1, floatStages.x1) <= 1.0e-5,
            "float stages diverge from double stages");
    require(relativeL2(stages.a, floatStages.a) <= 1.0e-5,
            "float A diverges from double A");
    require(relativeL2(stages.a2, floatStages.a2) <= 1.0e-5,
            "float A2 diverges from double A2");
    std::cout << "nicopedia_muon_ns_stage_test=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
