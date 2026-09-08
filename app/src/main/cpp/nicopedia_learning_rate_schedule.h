// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Exact, bounded learning-rate schedules for the Nicopedia HTP experiments.
#pragma once

#include <cmath>
#include <cstdint>

namespace phonelm::nicopedia_schedule {

enum class Kind : std::uint32_t {
  CONSTANT = 0,
  LINEAR_DECAY = 1,
  SQRT_DECAY = 2,
};

struct Config {
  Kind kind = Kind::CONSTANT;
  float peakLearningRate = 0.0f;
  float targetLearningRate = 0.0f;
  std::uint32_t decayStartStep = 0;
  std::uint32_t decayEndStep = 0;
  bool experimentFork = false;
};

inline bool validate(const Config& config, std::uint32_t totalSteps) {
  if (!std::isfinite(config.peakLearningRate) ||
      config.peakLearningRate <= 0.0f ||
      !std::isfinite(config.targetLearningRate) ||
      config.targetLearningRate < 0.0f)
    return false;
  if (config.kind == Kind::CONSTANT) {
    return config.decayStartStep == 0 && config.decayEndStep == 0 &&
           !config.experimentFork;
  }
  if (config.kind != Kind::LINEAR_DECAY && config.kind != Kind::SQRT_DECAY)
    return false;
  if (!config.experimentFork ||
      config.decayStartStep == 0 ||
      config.decayStartStep >= config.decayEndStep ||
      config.decayEndStep > totalSteps)
    return false;
  return true;
}

// The optimizer receives the LR for the 1-indexed global optimizer step.
// Boundary semantics are deliberately inclusive at both ends:
// step <= start is peak, start < step <= end is the selected cooldown shape,
// step >= end is target.
inline float at(const Config& config, std::uint32_t step) {
  if (config.kind == Kind::CONSTANT || step <= config.decayStartStep)
    return config.peakLearningRate;
  if (step >= config.decayEndStep) return config.targetLearningRate;
  const double progress =
      double(step - config.decayStartStep) /
      double(config.decayEndStep - config.decayStartStep);
  if (config.kind == Kind::SQRT_DECAY) {
    // Schedule-v2c uses shape = 1 - sqrt(p), not sqrt(1 - p).  Keep the
    // non-zero target in the affine scaling so the endpoint remains exact.
    const double shape = 1.0 - std::sqrt(progress);
    return static_cast<float>(double(config.targetLearningRate) +
                              (double(config.peakLearningRate) -
                               double(config.targetLearningRate)) *
                                  shape);
  }
  return static_cast<float>(double(config.peakLearningRate) +
                            progress *
                                (double(config.targetLearningRate) -
                                 double(config.peakLearningRate)));
}

inline const char* kindName(Kind kind) {
  if (kind == Kind::LINEAR_DECAY) return "linear_decay";
  if (kind == Kind::SQRT_DECAY) return "sqrt_decay";
  return "constant";
}

}  // namespace phonelm::nicopedia_schedule
