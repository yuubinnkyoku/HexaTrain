// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Standalone CPU benchmark for the frozen Original Muon full update.
#include "nicopedia_htp_muon.h"
#include "nicopedia_muon_optimizer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  try {
    phonelm::tiny_lm::Config config{1024, 32, 64, 128, 1e-5f, 19, 2};
    auto parameters = phonelm::tiny_lm::initialParameters(config, 1);
    auto gradients = parameters;
    auto momentum = parameters;
    for (const auto& e : phonelm::tiny_lm::parameterRegistry(momentum))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
    auto auxM = momentum, auxV = momentum;
    phonelm::nicopedia_muon::Config cpuConfig;
    double best = 0;
    for (int i = 0; i < 5; ++i) {
      const auto start = std::chrono::steady_clock::now();
      auto cpu = phonelm::nicopedia_muon::update(parameters, gradients,
                                                 momentum, auxM, auxV,
                                                 cpuConfig);
      const double us =
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (!cpu.error.empty()) throw std::runtime_error(cpu.error);
      std::cout << "iter=" << i << " total_us=" << us
                << " muon_us=" << cpu.muonMicroseconds << " matrices="
                << cpu.muonMatrixCount << "\n";
      if (!i || us < best) best = us;
    }
    std::cout << "cpu_best_us=" << best << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
