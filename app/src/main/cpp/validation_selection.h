// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::validation_selection {

enum class Mode : std::uint32_t {
  FINAL_STEP = 0,
  BEST_VALIDATION_V1 = 1,
};

inline constexpr std::uint32_t kValidationSchemaVersion = 2;
inline constexpr std::uint32_t kParameterRegistryVersion = 2;
inline constexpr double kLossTieTolerance = 1.0e-7;

inline const char* modeName(std::uint32_t mode) {
  switch (mode) {
    case 0: return "FINAL_STEP";
    case 1: return "BEST_VALIDATION_V1";
    default: return "UNKNOWN_CHECKPOINT_SELECTION_MODE";
  }
}

inline const char* validateMode(std::uint32_t mode) {
  return mode <= std::uint32_t(Mode::BEST_VALIDATION_V1)
             ? nullptr
             : "UNKNOWN_CHECKPOINT_SELECTION_MODE";
}

inline const std::array<int, 23>& evaluationSteps() {
  static const std::array<int, 23> steps{
      0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48,
      56, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320};
  return steps;
}

inline bool isEvaluationStep(int step) {
  const auto& steps = evaluationSteps();
  return std::find(steps.begin(), steps.end(), step) != steps.end();
}

// Validation keeps the single-family structure used by held-out generation,
// but starts at phase 2. This makes every full input prefix distinct from TRAIN
// (phase 0), CURRENT_PHASE1_EVAL (phase 1), and formal generation (phase 0).
// The two-token family has no third distinct phase and is excluded rather than
// leaking a formal prefix. Token transitions intentionally remain in-domain.
inline const std::array<std::vector<std::uint32_t>, 4>& validationRules() {
  static const std::array<std::vector<std::uint32_t>, 4> rules{
      std::vector<std::uint32_t>{0, 1, 2, 3},
      std::vector<std::uint32_t>{4, 5, 6, 7},
      std::vector<std::uint32_t>{8, 9},
      std::vector<std::uint32_t>{10, 11, 12}};
  return rules;
}

struct TokenCase {
  std::string id;
  std::vector<std::uint32_t> input;
  std::vector<std::uint32_t> target;
};

inline std::vector<TokenCase> validationCases(std::uint32_t tokens) {
  std::vector<TokenCase> cases;
  const auto& rules = validationRules();
  for (const std::size_t pattern : {std::size_t(0), std::size_t(1),
                                    std::size_t(3)}) {
    TokenCase item;
    item.id = "validation_v2_rotated_pattern_" + std::to_string(pattern);
    item.input.resize(tokens);
    item.target.resize(tokens);
    for (std::uint32_t i = 0; i < tokens; ++i) {
      const std::size_t phase = (std::size_t(i) + 2) % rules[pattern].size();
      item.input[i] = rules[pattern][phase];
      item.target[i] = rules[pattern][(phase + 1) % rules[pattern].size()];
    }
    cases.push_back(std::move(item));
  }
  return cases;
}

inline std::uint64_t fnv1a(const void* data, std::size_t size,
                           std::uint64_t hash = 1469598103934665603ull) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

inline std::string validationSetHash(std::uint32_t tokens) {
  std::uint64_t hash = 1469598103934665603ull;
  const auto version = kValidationSchemaVersion;
  hash = fnv1a(&version, sizeof(version), hash);
  for (const auto& item : validationCases(tokens)) {
    hash = fnv1a(item.id.data(), item.id.size(), hash);
    for (const auto* values : {&item.input, &item.target}) {
      const std::uint64_t count = values->size();
      hash = fnv1a(&count, sizeof(count), hash);
      hash = fnv1a(values->data(), values->size() * sizeof((*values)[0]), hash);
    }
  }
  std::ostringstream text;
  text << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}

struct Metrics {
  double loss = std::numeric_limits<double>::infinity();
  double accuracy = 0.0;
  double targetMargin = 0.0;
  double targetProbability = 0.0;
};

inline bool finite(const Metrics& metrics) {
  return std::isfinite(metrics.loss) && std::isfinite(metrics.accuracy) &&
         std::isfinite(metrics.targetMargin) &&
         std::isfinite(metrics.targetProbability);
}

inline bool better(const Metrics& candidate, int candidateStep,
                   const Metrics& incumbent, int incumbentStep) {
  if (!finite(candidate)) return false;
  if (!finite(incumbent)) return true;
  if (candidate.loss < incumbent.loss - kLossTieTolerance) return true;
  if (std::abs(candidate.loss - incumbent.loss) > kLossTieTolerance)
    return false;
  if (candidate.accuracy > incumbent.accuracy) return true;
  if (candidate.accuracy < incumbent.accuracy) return false;
  return candidateStep < incumbentStep;
}

struct EarlyStopSimulation {
  int patience = 0;
  int stopStep = 0;
  int bestStep = 0;
  int savedTrainingSteps = 0;
};

inline EarlyStopSimulation simulateEarlyStop(
    const std::vector<std::pair<int, Metrics>>& trajectory, int patience,
    int totalSteps = 320) {
  EarlyStopSimulation result;
  result.patience = patience;
  result.stopStep = totalSteps;
  Metrics best;
  int stale = 0;
  for (const auto& entry : trajectory) {
    if (better(entry.second, entry.first, best, result.bestStep)) {
      best = entry.second;
      result.bestStep = entry.first;
      stale = 0;
    } else if (++stale >= patience) {
      result.stopStep = entry.first;
      break;
    }
  }
  result.savedTrainingSteps = std::max(0, totalSteps - result.stopStep);
  return result;
}

}  // namespace phonelm::validation_selection
