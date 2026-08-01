// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Explicit experimental training-stability modes for the generic QNN HTP
// tiny-LM trainer, and the paired-depth diagnostic initialization contract.
//
// LEGACY is, and remains, the default. Non-LEGACY modes are recorded in the
// device report and in private checkpoints, and mode-mismatched checkpoints
// are rejected fail-closed. None of these modes changes the numerics of the
// LayerNorm centered-scale fix; they are quality-side candidates only.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "tiny_language_model_cpu.h"

namespace phonelm {

enum class TrainingStabilityMode : std::uint32_t {
  LEGACY = 0,
  // A: learning-rate warmup, linear over the first 64 steps.
  WARMUP64 = 1,
  // B: learning-rate decay, linear from lr to 0 over all steps.
  DECAY_LINEAR = 2,
  // E: zero initialization for every layer's Wo (attention output projection)
  // and FFN W2, so residual branches start as identities.
  ZERO_OUTPUT_PROJ_BRANCH_INIT = 3,
  // F: depth-aware branch init variance: Wo, W1, W2 scales divided by
  // sqrt(2 * numLayers).
  DEPTH_SCALED_BRANCH_INIT = 4,
  // C (reserved): residual branch output scaling. Not supported by the
  // current HTP graphs; requests fail closed on device.
  RESIDUAL_BRANCH_SCALING = 5,
  // G: global gradient clipping at norm 1.0 (uses the established
  // gradient_scale Adam graph input).
  GRADIENT_CLIP_1 = 6,
};

enum class DepthPairInitMode : std::uint32_t {
  // Established phase-seeded initialization (unchanged).
  LEGACY = 0,
  // Diagnostic assertion that the phase-seeded scheme keeps the shared
  // prefix (token embedding, layers 0..L-2, output projection) identical to
  // the L-1 model for the same seed; mismatch fails closed. Recorded in
  // reports and checkpoints so diagnostics stay explicit.
  PAIRED_SHARED_PREFIX = 1,
};

inline const char* trainingStabilityModeName(std::uint32_t mode) {
  switch (mode) {
    case 0: return "LEGACY";
    case 1: return "WARMUP64";
    case 2: return "DECAY_LINEAR";
    case 3: return "ZERO_OUTPUT_PROJ_BRANCH_INIT";
    case 4: return "DEPTH_SCALED_BRANCH_INIT";
    case 5: return "RESIDUAL_BRANCH_SCALING";
    case 6: return "GRADIENT_CLIP_1";
  }
  return "UNKNOWN";
}

inline const char* depthPairInitModeName(std::uint32_t mode) {
  switch (mode) {
    case 0: return "LEGACY";
    case 1: return "PAIRED_SHARED_PREFIX";
  }
  return "UNKNOWN";
}

// nullptr when valid, else a stable failure tag.
inline const char* validateTrainingStabilityMode(std::uint32_t mode) {
  if (mode <= 6 && mode != 5) return nullptr;
  if (mode == 5) return "RESIDUAL_BRANCH_SCALING_UNSUPPORTED_ON_DEVICE";
  return "UNKNOWN_TRAINING_STABILITY_MODE";
}
inline const char* validateDepthPairInitMode(std::uint32_t mode) {
  if (mode <= 1) return nullptr;
  return "UNKNOWN_DEPTH_PAIR_INIT_MODE";
}

// Scheduled learning rate. LEGACY returns base for every step.
inline float stabilityLearningRate(std::uint32_t mode, float base,
                                   std::uint32_t step, std::uint32_t totalSteps) {
  if (mode == 1) {  // WARMUP64
    const double factor = std::min(1.0, double(step) / 64.0);
    return float(base * factor);
  }
  if (mode == 2) {  // DECAY_LINEAR
    const double done = totalSteps ? double(totalSteps) : 1.0;
    return float(base * (1.0 - double(step - 1) / done));
  }
  return base;
}

inline qnn::TinyTransformerLayerParameters& mutableLayerOf(
    qnn::TinyTransformerParameters& p, std::uint32_t i) {
  if (i == 0) return static_cast<qnn::TinyTransformerLayerParameters&>(p);
  return p.layers.at(i - 1);
}

// Init-side candidates applied to legacy initial parameters.
inline qnn::TinyTransformerParameters applyInitStability(
    const tiny_lm::Config& config, qnn::TinyTransformerParameters p,
    std::uint32_t mode) {
  if (mode == 3) {  // ZERO_OUTPUT_PROJ_BRANCH_INIT
    for (std::uint32_t i = 0; i < config.numLayers; ++i) {
      auto& l = mutableLayerOf(p, i);
      std::fill(l.wo.begin(), l.wo.end(), 0.0f);
      std::fill(l.w2.begin(), l.w2.end(), 0.0f);
    }
  } else if (mode == 4) {  // DEPTH_SCALED_BRANCH_INIT
    const float s = 1.0f / std::sqrt(2.0f * float(config.numLayers));
    for (std::uint32_t i = 0; i < config.numLayers; ++i) {
      auto& l = mutableLayerOf(p, i);
      for (float& x : l.wo) x *= s;
      for (float& x : l.w1) x *= s;
      for (float& x : l.w2) x *= s;
    }
  }
  return p;
}

// PAIRED_SHARED_PREFIX contract: every shared parameter of the (L-1)-layer
// model must be identical (same bytes) inside the L-layer model for the same
// configuration and seed. Name-keyed shared-prefix comparison for two
// prepared parameter sets.
inline bool sharedPrefixParametersEqual(
    const qnn::TinyTransformerParameters& shallow,
    const qnn::TinyTransformerParameters& deep) {
  const auto shallowRegistry = tiny_lm::parameterRegistry(shallow);
  const auto deepRegistry = tiny_lm::parameterRegistry(deep);
  size_t shared = 0, mismatched = 0;
  for (const auto& s : shallowRegistry) {
    const auto* match = static_cast<const std::vector<float>*>(nullptr);
    for (const auto& d : deepRegistry)
      if (d.name == s.name) match = d.values;
    if (!match) return false;
    ++shared;
    if (*match != *s.values) ++mismatched;
  }
  return shared > 0 && mismatched == 0;
}

}  // namespace phonelm
