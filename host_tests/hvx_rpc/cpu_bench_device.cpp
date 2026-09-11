// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Same-device ARM64 CPU baseline for the frozen Original Muon workload.
//
// This intentionally reuses the production CPU optimizer and model fixture.
// One warm-up update is discarded; the measured repetitions stay in this
// process so JNI/process-launch cost is not part of the per-update sample.
#include "nicopedia_htp_muon.h"
#include "nicopedia_muon_optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    phonelm::tiny_lm::Config model{1024, 32, 64, 128, 1e-5f, 19, 2};
    const auto initial = phonelm::tiny_lm::initialParameters(model, 1);
    const auto gradients = initial;
    auto momentum = initial;
    for (const auto& entry : phonelm::tiny_lm::parameterRegistry(momentum))
      std::fill(const_cast<std::vector<float>*>(entry.values)->begin(),
                const_cast<std::vector<float>*>(entry.values)->end(), 0.0f);
    const auto auxM = momentum;
    const auto auxV = momentum;
    phonelm::nicopedia_muon::Config optimizer;
    optimizer.muonLearningRate = 0.01f;
    optimizer.momentum = 0.95f;
    optimizer.nesterov = true;
    optimizer.nsSteps = 5;

    // Warm up allocator/cache/code paths without recording the result.
    const auto warmup = phonelm::nicopedia_muon::update(
        initial, gradients, momentum, auxM, auxV, optimizer);
    if (!warmup.error.empty() || warmup.muonMatrixCount != 114 ||
        !warmup.health.parametersFinite || !warmup.health.momentumFinite)
      throw std::runtime_error("CPU Muon warmup failed");

    constexpr int kRepetitions = 5;
    double totalBest = 0.0;
    double muonBest = 0.0;
    double totalSum = 0.0;
    double muonSum = 0.0;
    double totalSamples[kRepetitions] = {};
    double muonSamples[kRepetitions] = {};
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
      const auto started = std::chrono::steady_clock::now();
      const auto result = phonelm::nicopedia_muon::update(
          initial, gradients, momentum, auxM, auxV, optimizer);
      const double totalUs = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - started).count();
      if (!result.error.empty() || result.muonMatrixCount != 114 ||
          result.muonParameterCount != 622592 ||
          !result.health.parametersFinite || !result.health.momentumFinite)
        throw std::runtime_error("CPU Muon update failed");
      const double muonUs = result.muonMicroseconds;
      std::cout << "repetition=" << repetition << " total_us=" << totalUs
                << " muon_us=" << muonUs << " matrices="
                << result.muonMatrixCount << " parameters="
                << result.muonParameterCount << " finite=true\n";
      totalSum += totalUs;
      muonSum += muonUs;
      totalSamples[repetition] = totalUs;
      muonSamples[repetition] = muonUs;
      if (repetition == 0 || totalUs < totalBest) totalBest = totalUs;
      if (repetition == 0 || muonUs < muonBest) muonBest = muonUs;
    }
    std::sort(totalSamples, totalSamples + kRepetitions);
    std::sort(muonSamples, muonSamples + kRepetitions);
    const double wallMedian = totalSamples[kRepetitions / 2];
    const double muonMedian = muonSamples[kRepetitions / 2];
    std::cout << "warmup_repetitions=1 measured_repetitions=" << kRepetitions
              // The authoritative comparator is Muon-only; auxiliary Adam is
              // reported separately so it cannot contaminate the 114-matrix
              // HVX comparison.
              << " device_cpu_total_us=" << muonSum / kRepetitions
              << " device_cpu_median_us=" << muonMedian
              << " device_cpu_best_us=" << muonBest
              << " device_cpu_update_wall_us=" << totalSum / kRepetitions
              << " device_cpu_update_wall_median_us=" << wallMedian
              << " device_cpu_update_wall_best_us=" << totalBest << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
