// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "tiny_language_model_cpu.h"
#include <cstdint>
#include <string>
#include <vector>

namespace phonelm::nicopedia_muon {

inline constexpr float kNsA = 3.4445f;
inline constexpr float kNsB = -4.7750f;
inline constexpr float kNsC = 2.0315f;
inline constexpr float kNsEpsilon = 1.0e-7f;
inline constexpr char kAlgorithmIdentity[] = "keller_original_64560829_fp32";

struct Config {
  float muonLearningRate = 0.01f;
  float auxiliaryAdamLearningRate = 0.0022f;
  float momentum = 0.95f;
  bool nesterov = true;
  std::uint32_t nsSteps = 5;
  std::uint64_t optimizerStep = 1;
};

struct StageHealth {
  bool gradientFinite = true;
  bool momentumFinite = true;
  bool normalizedFinite = true;
  bool nsOutputFinite = true;
  bool updateFinite = true;
  bool parametersFinite = true;
};

struct Result {
  qnn::TinyTransformerParameters parameters;
  qnn::TinyTransformerParameters muonMomentum;
  qnn::TinyTransformerParameters auxiliaryAdamM;
  qnn::TinyTransformerParameters auxiliaryAdamV;
  StageHealth health;
  std::uint32_t muonMatrixCount = 0;
  std::uint64_t muonParameterCount = 0;
  std::uint64_t auxiliaryAdamParameterCount = 0;
  double muonMicroseconds = 0.0;
  double auxiliaryAdamMicroseconds = 0.0;
  std::string error;
};

bool zeropowerNewtonSchulzFp32(const std::vector<float>& input,
                               std::uint32_t rows, std::uint32_t columns,
                               std::uint32_t steps, std::vector<float>* output,
                               StageHealth* health = nullptr,
                               std::string* error = nullptr);

// Validate that a gradient or optimizer-state registry is an exact semantic
// match for the model registry.  This is intentionally exposed so host tests
// can exercise corrupt registry metadata (including a bad name) without
// relying on undefined mutation of the model's field layout.
bool validateOptimizerStateRegistry(
    const std::vector<tiny_lm::ParameterInfo>& expected,
    const std::vector<tiny_lm::ParameterInfo>& actual,
    const char* stateName,
    std::string* error = nullptr);

Result update(const qnn::TinyTransformerParameters& parameters,
              const qnn::TinyTransformerParameters& gradients,
              const qnn::TinyTransformerParameters& muonMomentum,
              const qnn::TinyTransformerParameters& auxiliaryAdamM,
              const qnn::TinyTransformerParameters& auxiliaryAdamV,
              const Config& config);

// Applies only the AUX_ADAM registry entries. MUON parameters and the supplied
// HTP-produced momentum are copied without optimizer arithmetic.
Result updateAuxiliaryAdamOnly(
    const qnn::TinyTransformerParameters& parameters,
    const qnn::TinyTransformerParameters& gradients,
    const qnn::TinyTransformerParameters& muonMomentum,
    const qnn::TinyTransformerParameters& auxiliaryAdamM,
    const qnn::TinyTransformerParameters& auxiliaryAdamV,
    const Config& config);

}  // namespace phonelm::nicopedia_muon
