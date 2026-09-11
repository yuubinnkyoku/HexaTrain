// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "nicopedia_muon_optimizer.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void write(const char* path, const std::vector<float>& data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
  if (!stream) throw std::runtime_error("fixture write failed");
}

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("input cols payload reference hyperparameters");
    const std::string columns_arg = argv[2];
    if (columns_arg != "64" && columns_arg != "128") throw std::runtime_error("invalid columns");
    const int columns = std::stoi(columns_arg);
    const int elements = 64 * columns;
    std::vector<float> frozen(elements);
    std::ifstream stream(argv[1], std::ios::binary);
    stream.read(reinterpret_cast<char*>(frozen.data()), std::streamsize(elements * sizeof(float)));
    if (!stream || stream.peek() != EOF) throw std::runtime_error("frozen size mismatch");
    std::vector<float> payload(3 * elements, 0), reference(8 * elements, 0), nesterov(elements);
    const float beta = .95f, alpha = 1.0f - beta, lr = .01f, scale = 1.0f;
    double norm2 = 0;
    for (int i = 0; i < elements; ++i) {
      if (!std::isfinite(frozen[i])) throw std::runtime_error("nonfinite frozen input");
      // No libm regeneration or new random primary matrix: derive all states
      // directly from the frozen bytes. Momentum starts at zero.
      payload[i] = .02f * frozen[i];
      payload[elements + i] = frozen[i];
      reference[elements + i] = beta * payload[2 * elements + i] + alpha * payload[elements + i];
      nesterov[i] = alpha * payload[elements + i] + beta * reference[elements + i];
      norm2 += double(nesterov[i]) * nesterov[i];
    }
    const double denominator = std::sqrt(norm2) + phonelm::nicopedia_muon::kNsEpsilon;
    for (int i = 0; i < elements; ++i) reference[2 * elements + i] = float(double(nesterov[i]) / denominator);
    for (int step = 1; step <= 5; ++step) {
      std::vector<float> orthogonal;
      std::string error;
      if (!phonelm::nicopedia_muon::zeropowerNewtonSchulzFp32(nesterov, 64, columns, step, &orthogonal, nullptr, &error))
        throw std::runtime_error(error);
      std::copy(orthogonal.begin(), orthogonal.end(), reference.begin() + (2 + step) * elements);
      if (step == 5) for (int i = 0; i < elements; ++i) reference[i] = payload[i] - lr * scale * orthogonal[i];
    }
    write(argv[3], payload);
    write(argv[4], reference);
    write(argv[5], {lr, scale});
    std::cout << "fixture=FROZEN_DERIVED weights=0.02*X0 gradients=X0 momentum=0 ns=5 algorithm="
              << phonelm::nicopedia_muon::kAlgorithmIdentity << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
