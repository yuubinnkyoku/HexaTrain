// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

// Host-only causal interventions for the L19 Attention minimal-cause audit.
// This header intentionally exposes no final-holdout cases.

#include "seed_instability_diagnostics_lib.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::attention_minimal_cause {

namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace dq = phonelm::depth_quality;
namespace ma = phonelm::margin_analysis;
namespace rp = phonelm::readout_probe;
namespace si = phonelm::seed_instability;
namespace tiny = phonelm::tiny_lm;
namespace train = phonelm::critical_margin::train;

using P = qnn::TinyTransformerParameters;
using LP = qnn::TinyTransformerLayerParameters;

enum class Pattern { Learned, Self, Previous, UniformCausal };

inline const char* patternName(Pattern pattern) {
  switch (pattern) {
    case Pattern::Learned: return "LEARNED";
    case Pattern::Self: return "FIXED_SELF";
    case Pattern::Previous: return "FIXED_PREVIOUS";
    case Pattern::UniformCausal: return "FIXED_UNIFORM_CAUSAL";
  }
  return "UNKNOWN";
}

enum class Group { Attention, Qk, Vo, Ffn };

inline bool inGroup(const std::string& name, Group group) {
  const bool qk = si::hasSuffix(name, ".wq") || si::hasSuffix(name, ".wk");
  const bool vo = si::hasSuffix(name, ".wv") || si::hasSuffix(name, ".wo");
  if (group == Group::Attention) return qk || vo;
  if (group == Group::Qk) return qk;
  if (group == Group::Vo) return vo;
  return si::hasSuffix(name, ".ffn_w1") ||
         si::hasSuffix(name, ".ffn_w2");
}

inline std::vector<float>* mutableValues(P& params, const std::string& name) {
  if (name == "token_embedding") return &params.tokenEmbedding;
  if (name == "output_projection") return &params.outputProjection;
  if (name.rfind("layer_", 0) != 0 || name.size() < 10)
    throw std::invalid_argument("PARAMETER_NAME");
  const auto dot = name.find('.');
  if (dot == std::string::npos) throw std::invalid_argument("PARAMETER_NAME");
  const std::uint32_t layerIndex =
      static_cast<std::uint32_t>(std::stoul(name.substr(6, dot - 6)));
  auto& layer = train::layer(params, layerIndex);
  const std::string field = name.substr(dot + 1);
  if (field == "norm1_gamma") return &layer.gamma1;
  if (field == "norm1_beta") return &layer.beta1;
  if (field == "wq") return &layer.wq;
  if (field == "wk") return &layer.wk;
  if (field == "wv") return &layer.wv;
  if (field == "wo") return &layer.wo;
  if (field == "norm2_gamma") return &layer.gamma2;
  if (field == "norm2_beta") return &layer.beta2;
  if (field == "ffn_w1") return &layer.w1;
  if (field == "ffn_w2") return &layer.w2;
  throw std::invalid_argument("PARAMETER_FIELD");
}

inline void copyGroup(P& destination, const P& source, Group group) {
  const auto sourceRegistry = tiny::parameterRegistry(source);
  const auto destinationRegistry = tiny::parameterRegistry(destination);
  if (sourceRegistry.size() != destinationRegistry.size())
    throw std::invalid_argument("GROUP_REGISTRY_SIZE");
  for (std::size_t i = 0; i < sourceRegistry.size(); ++i) {
    if (sourceRegistry[i].name != destinationRegistry[i].name ||
        sourceRegistry[i].values->size() != destinationRegistry[i].values->size())
      throw std::invalid_argument("GROUP_REGISTRY_IDENTITY");
    if (inGroup(sourceRegistry[i].name, group))
      *mutableValues(destination, sourceRegistry[i].name) =
          *sourceRegistry[i].values;
  }
}

inline void zeroGroup(P& params, Group group) {
  for (const auto& item : tiny::parameterRegistry(params))
    if (inGroup(item.name, group))
      std::fill(mutableValues(params, item.name)->begin(),
                mutableValues(params, item.name)->end(), 0.0f);
}

inline bool sameGroup(const P& a, const P& b, Group group,
                      bool compareInside) {
  const auto ar = tiny::parameterRegistry(a);
  const auto br = tiny::parameterRegistry(b);
  if (ar.size() != br.size()) return false;
  for (std::size_t i = 0; i < ar.size(); ++i) {
    if (ar[i].name != br[i].name || ar[i].values->size() != br[i].values->size())
      return false;
    if (inGroup(ar[i].name, group) == compareInside &&
        *ar[i].values != *br[i].values)
      return false;
  }
  return true;
}

inline std::string groupHash(const P& params, Group group, bool inside = true) {
  std::uint64_t hash = 1469598103934665603ull;
  const auto registry = tiny::parameterRegistry(params);
  for (const auto& item : registry) {
    if (inGroup(item.name, group) != inside) continue;
    hash = ar::fnv1a(item.name.data(), item.name.size(), hash);
    const std::uint64_t size = item.values->size();
    hash = ar::fnv1a(&size, sizeof(size), hash);
    hash = ar::fnv1a(item.values->data(), item.values->size() * sizeof(float), hash);
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

inline std::string contentHash(const P& params) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto& item : tiny::parameterRegistry(params)) {
    hash = ar::fnv1a(item.name.data(), item.name.size(), hash);
    const std::uint64_t size = item.values->size();
    hash = ar::fnv1a(&size, sizeof(size), hash);
    hash = ar::fnv1a(item.values->data(), item.values->size() * sizeof(float),
                     hash);
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

inline P scaledBranch(const tiny::Config& config, const P& params,
                      float attentionAlpha, bool zeroFfn) {
  P result = params;
  for (std::uint32_t li = 0; li < config.numLayers; ++li) {
    auto& layer = train::layer(result, li);
    for (float& value : layer.wo) value *= attentionAlpha;
    if (zeroFfn) std::fill(layer.w2.begin(), layer.w2.end(), 0.0f);
  }
  return result;
}

// Learned is delegated to the canonical copy to guarantee a bitwise no-op.
// Fixed patterns retain V, O, the residual add, and downstream LN/FFN.
inline train::GF patternForward(const tiny::Config& c,
                                const std::vector<float>& oneHot,
                                const P& weights, Pattern pattern) {
  if (pattern == Pattern::Learned)
    return train::generalForward(c, oneHot, weights);
  train::GF g;
  g.embedded = train::mm(oneHot, weights.tokenEmbedding, c.tokens,
                         c.vocabularySize, c.dimension);
  std::vector<float> x = g.embedded;
  train::add(x, train::fixedPositionCpu(c));
  g.embedded = x;
  const std::uint32_t headDimension = c.dimension / c.numHeads;
  for (std::uint32_t li = 0; li < c.numLayers; ++li) {
    const LP& p = train::layer(weights, li);
    train::GL z;
    z.x = x;
    z.n1 = train::nf(c, x, p.gamma1, p.beta1);
    z.q.assign(static_cast<std::size_t>(c.tokens) * c.dimension, 0.0f);
    z.k = z.q;
    z.v = train::mm(z.n1.out, p.wv, c.tokens, c.dimension, c.dimension);
    z.prob.assign(static_cast<std::size_t>(c.numHeads) * c.tokens * c.tokens,
                  0.0f);
    z.ctx.assign(static_cast<std::size_t>(c.tokens) * c.dimension, 0.0f);
    for (std::uint32_t h = 0; h < c.numHeads; ++h) {
      for (std::uint32_t row = 0; row < c.tokens; ++row) {
        const std::size_t base =
            (static_cast<std::size_t>(h) * c.tokens + row) * c.tokens;
        for (std::uint32_t column = 0; column <= row; ++column) {
          float probability = 0.0f;
          if (pattern == Pattern::Self)
            probability = column == row ? 1.0f : 0.0f;
          else if (pattern == Pattern::Previous)
            probability = column == (row == 0 ? 0 : row - 1) ? 1.0f : 0.0f;
          else
            probability = 1.0f / static_cast<float>(row + 1);
          z.prob[base + column] = probability;
          for (std::uint32_t d = 0; d < headDimension; ++d)
            z.ctx[static_cast<std::size_t>(row) * c.dimension +
                  h * headDimension + d] +=
                probability * z.v[static_cast<std::size_t>(column) *
                                      c.dimension + h * headDimension + d];
        }
      }
    }
    z.r1 = x;
    train::add(z.r1, train::mm(z.ctx, p.wo, c.tokens, c.dimension, c.dimension));
    z.n2 = train::nf(c, z.r1, p.gamma2, p.beta2);
    z.f1 = train::mm(z.n2.out, p.w1, c.tokens, c.dimension,
                     c.feedForwardDimension);
    z.relu = z.f1;
    for (float& value : z.relu) value = std::max(0.0f, value);
    z.out = z.r1;
    train::add(z.out, train::mm(z.relu, p.w2, c.tokens,
                                c.feedForwardDimension, c.dimension));
    x = z.out;
    g.layers.push_back(std::move(z));
  }
  g.logits = train::mm(x, weights.outputProjection, c.tokens, c.dimension,
                       c.vocabularySize);
  g.prob.resize(g.logits.size());
  for (std::uint32_t row = 0; row < c.tokens; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * c.vocabularySize;
    const float maximum = *std::max_element(
        g.logits.begin() + base, g.logits.begin() + base + c.vocabularySize);
    double sum = 0.0;
    for (std::uint32_t token = 0; token < c.vocabularySize; ++token) {
      const float value = std::exp(g.logits[base + token] - maximum);
      g.prob[base + token] = value;
      sum += value;
    }
    for (std::uint32_t token = 0; token < c.vocabularySize; ++token)
      g.prob[base + token] /= static_cast<float>(sum);
  }
  return g;
}

struct PatternBackward {
  float loss = 0.0f;
  float accuracy = 0.0f;
  P gradients;
};

inline PatternBackward patternForwardBackward(
    const tiny::Config& config, const std::vector<float>& oneHot,
    const std::vector<float>& target, const P& params, Pattern pattern) {
  if (pattern == Pattern::Learned) {
    const auto canonical = tiny::forwardBackward(config, oneHot, target,
                                                  params, 0.0f);
    return {canonical.loss, canonical.accuracy, canonical.gradients};
  }
  const train::GF forward = patternForward(config, oneHot, params, pattern);
  std::vector<float> dLogits(forward.logits.size());
  double loss = 0.0;
  std::uint32_t correct = 0;
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    const std::size_t base =
        static_cast<std::size_t>(row) * config.vocabularySize;
    float maximum = forward.logits[base];
    std::uint32_t predicted = 0;
    std::uint32_t truth = 0;
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
      if (forward.logits[base + token] > maximum) {
        maximum = forward.logits[base + token];
        predicted = token;
      }
      if (target[base + token] > 0.5f) truth = token;
      dLogits[base + token] =
          (forward.prob[base + token] - target[base + token]) /
          static_cast<float>(config.tokens);
    }
    double sum = 0.0;
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token)
      sum += std::exp(static_cast<double>(forward.logits[base + token] - maximum));
    loss += maximum + std::log(sum) - forward.logits[base + truth];
    correct += predicted == truth ? 1u : 0u;
  }
  PatternBackward result;
  result.loss = static_cast<float>(loss / config.tokens);
  result.accuracy = static_cast<float>(correct) / config.tokens;
  result.gradients =
      train::generalBackwardGradients(config, forward, oneHot, params, dLogits);
  return result;
}

enum class TrainingMode {
  Canonical,
  FixedSelf,
  FixedPrevious,
  FixedUniform,
  FreezeAttentionInitial,
  FreezeQkInitial,
  FreezeVoInitial,
};

inline const char* trainingModeName(TrainingMode mode) {
  switch (mode) {
    case TrainingMode::Canonical: return "CANONICAL";
    case TrainingMode::FixedSelf: return "TRAIN_FIXED_SELF";
    case TrainingMode::FixedPrevious: return "TRAIN_FIXED_PREVIOUS";
    case TrainingMode::FixedUniform: return "TRAIN_FIXED_UNIFORM_CAUSAL";
    case TrainingMode::FreezeAttentionInitial: return "FREEZE_ATTENTION_INITIAL";
    case TrainingMode::FreezeQkInitial: return "FREEZE_QK_INITIAL";
    case TrainingMode::FreezeVoInitial: return "FREEZE_VO_INITIAL";
  }
  return "UNKNOWN";
}

inline Pattern trainingPattern(TrainingMode mode) {
  if (mode == TrainingMode::FixedSelf) return Pattern::Self;
  if (mode == TrainingMode::FixedPrevious) return Pattern::Previous;
  if (mode == TrainingMode::FixedUniform) return Pattern::UniformCausal;
  return Pattern::Learned;
}

inline bool finiteParameters(const P& params) {
  for (const auto& item : tiny::parameterRegistry(params))
    for (const float value : *item.values)
      if (!std::isfinite(value)) return false;
  return true;
}

inline si::TrainingState runTraining(const tiny::Config& config,
                                    std::uint32_t seed,
                                    TrainingMode mode, int endStep = 320) {
  si::TrainingState state;
  P initial = tiny::initialParameters(config, seed);
  initial = dq::applyInitStability(config, std::move(initial),
                                   dq::StabilityMode::LEGACY);
  state.params = initial;
  const Pattern pattern = trainingPattern(mode);
  if (pattern != Pattern::Learned) zeroGroup(state.params, Group::Qk);
  si::zeroLike(state.firstMoment, state.params);
  si::zeroLike(state.secondMoment, state.params);
  P fixed = state.params;
  for (int step = 1; step <= endStep; ++step) {
    const auto batch = dq::formalBatch(config,
        static_cast<std::uint32_t>((step - 1) % 4), 0);
    const auto backward = patternForwardBackward(
        config, batch.first, batch.second, state.params, pattern);
    const float correction1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float correction2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float learningRate = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(dq::StabilityMode::LEGACY), 0.003f,
        static_cast<std::uint32_t>(step),
        static_cast<std::uint32_t>(endStep));
    auto update = tiny::adamUpdate(
        state.params, backward.gradients, state.firstMoment,
        state.secondMoment, learningRate, 0.9f, 0.999f, 1e-8f,
        correction1, correction2);
    auto freeze = [&](Group group) {
      copyGroup(update.next, fixed, group);
      zeroGroup(update.firstMoment, group);
      zeroGroup(update.secondMoment, group);
    };
    if (pattern != Pattern::Learned) freeze(Group::Qk);
    if (mode == TrainingMode::FreezeAttentionInitial) freeze(Group::Attention);
    if (mode == TrainingMode::FreezeQkInitial) freeze(Group::Qk);
    if (mode == TrainingMode::FreezeVoInitial) freeze(Group::Vo);
    state.params = std::move(update.next);
    state.firstMoment = std::move(update.firstMoment);
    state.secondMoment = std::move(update.secondMoment);
    state.step = step;
    state.lastLoss = backward.loss;
    state.finite = state.finite && std::isfinite(backward.loss) &&
                   finiteParameters(state.params) &&
                   finiteParameters(state.firstMoment) &&
                   finiteParameters(state.secondMoment);
  }
  return state;
}

inline si::TrainingState trainHybridInitial(const tiny::Config& config,
                                            std::uint32_t attentionSeed,
                                            std::uint32_t restSeed,
                                            int endStep = 320) {
  P params = tiny::initialParameters(config, restSeed);
  P attention = tiny::initialParameters(config, attentionSeed);
  copyGroup(params, attention, Group::Attention);
  si::TrainingState state;
  state.params = std::move(params);
  si::zeroLike(state.firstMoment, state.params);
  si::zeroLike(state.secondMoment, state.params);
  for (int step = 1; step <= endStep; ++step) {
    const auto batch = dq::formalBatch(config,
        static_cast<std::uint32_t>((step - 1) % 4), 0);
    const auto fb = tiny::forwardBackward(config, batch.first, batch.second,
                                          state.params, 0.0f);
    const float c1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float c2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float lr = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(dq::StabilityMode::LEGACY), 0.003f,
        static_cast<std::uint32_t>(step),
        static_cast<std::uint32_t>(endStep));
    auto update = tiny::adamUpdate(state.params, fb.gradients,
        state.firstMoment, state.secondMoment, lr, 0.9f, 0.999f, 1e-8f,
        c1, c2);
    state.params = std::move(update.next);
    state.firstMoment = std::move(update.firstMoment);
    state.secondMoment = std::move(update.secondMoment);
    state.step = step;
    state.lastLoss = fb.loss;
    state.finite = state.finite && std::isfinite(fb.loss) &&
                   finiteParameters(state.params) &&
                   finiteParameters(state.firstMoment) &&
                   finiteParameters(state.secondMoment);
  }
  return state;
}

inline ma::Score patternScore(const tiny::Config& config, const P& params,
                              Pattern pattern,
                              const std::vector<std::uint32_t>& context,
                              std::uint32_t truth) {
  const auto oneHot = tiny::oneHot(context, config.vocabularySize);
  const auto forward = patternForward(config, oneHot, params, pattern);
  const std::size_t base =
      static_cast<std::size_t>(config.tokens - 1) * config.vocabularySize;
  std::vector<double> logits(config.vocabularySize);
  std::vector<double> probabilities(config.vocabularySize);
  for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
    logits[token] = forward.logits[base + token];
    probabilities[token] = forward.prob[base + token];
  }
  return rp::stableScoreFromLogits(logits, probabilities, truth);
}

struct Evaluation {
  rp::TokenMetrics teacherForced;
  cm::CheckpointMetrics freeRunning;
  double minimumMargin = std::numeric_limits<double>::infinity();
  std::uint64_t tieCount = 0;
};

inline Evaluation evaluate(const tiny::Config& config, const P& params,
                           Pattern pattern,
                           const std::vector<ar::Case>& cases) {
  Evaluation result;
  const auto rows = rp::teacherForcedRows(cases, config.tokens);
  std::vector<ma::Score> scores;
  scores.reserve(rows.size());
  for (const auto& row : rows) {
    const auto score = patternScore(config, params, pattern,
                                    row.context, row.truth);
    scores.push_back(score);
    result.minimumMargin = std::min(result.minimumMargin,
                                    score.expectedMinusTop1Margin);
    if (std::abs(score.expectedMinusTop1Margin) <= 1e-12) ++result.tieCount;
  }
  result.teacherForced = rp::aggregateTokenMetrics(scores, rows);
  result.freeRunning = rp::freeRunningRollout(
      cases, 320, [&](const std::vector<std::uint32_t>& context,
                      std::uint32_t truth) {
        return patternScore(config, params, pattern, context, truth);
      });
  return result;
}

inline bool sameCaseMetrics(const cm::CheckpointMetrics& a,
                            const ar::Metrics& b) {
  if (a.tokenExact != b.tokenExact || a.tokenTotal != b.tokenTotal ||
      a.sequenceExact != b.sequenceExact ||
      a.sequenceTotal != b.sequenceTotal || a.cases.size() != b.perCase.size())
    return false;
  for (std::size_t i = 0; i < a.cases.size(); ++i)
    if (a.cases[i].id != b.perCase[i].id ||
        a.cases[i].firstErrorPosition != b.perCase[i].firstErrorPosition ||
        a.cases[i].tokenExact != b.perCase[i].tokenExact ||
        a.cases[i].sequenceExact != b.perCase[i].sequenceExact)
      return false;
  return true;
}

inline bool selfTest(std::string* error = nullptr) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  tiny::Config config;
  config.vocabularySize = 8;
  config.tokens = 4;
  config.dimension = 4;
  config.feedForwardDimension = 8;
  config.numLayers = 2;
  config.numHeads = 2;
  const P initial = tiny::initialParameters(config, 9);
  const auto input = tiny::oneHot({0, 1, 2, 3}, config.vocabularySize);
  const auto target = tiny::oneHot({1, 2, 3, 4}, config.vocabularySize);
  const auto learned = patternForward(config, input, initial, Pattern::Learned);
  const auto canonical = train::generalForward(config, input, initial);
  if (learned.logits != canonical.logits || learned.prob != canonical.prob)
    return fail("NOOP_FORWARD_PARITY");
  const auto learnedBackward = patternForwardBackward(
      config, input, target, initial, Pattern::Learned);
  const auto canonicalBackward = tiny::forwardBackward(
      config, input, target, initial, 0.0f);
  if (learnedBackward.loss != canonicalBackward.loss ||
      !si::sameParameters(learnedBackward.gradients,
                          canonicalBackward.gradients))
    return fail("NOOP_BACKWARD_PARITY");
  for (const Pattern pattern : {Pattern::Self, Pattern::Previous,
                                Pattern::UniformCausal}) {
    const auto forward = patternForward(config, input, initial, pattern);
    for (const auto& layer : forward.layers) {
      for (std::uint32_t h = 0; h < config.numHeads; ++h) {
        for (std::uint32_t row = 0; row < config.tokens; ++row) {
          double sum = 0.0;
          for (std::uint32_t column = 0; column < config.tokens; ++column) {
            const float p = layer.prob[(static_cast<std::size_t>(h) *
                config.tokens + row) * config.tokens + column];
            if (column > row && p != 0.0f) return fail("CAUSAL_MASK");
            sum += p;
          }
          if (std::abs(sum - 1.0) > 1e-6) return fail("PATTERN_ROW_SUM");
          if (pattern == Pattern::Self &&
              layer.prob[(static_cast<std::size_t>(h) * config.tokens + row) *
                         config.tokens + row] != 1.0f)
            return fail("SELF_PATTERN");
          const std::uint32_t previous = row == 0 ? 0 : row - 1;
          if (pattern == Pattern::Previous &&
              layer.prob[(static_cast<std::size_t>(h) * config.tokens + row) *
                         config.tokens + previous] != 1.0f)
            return fail("PREVIOUS_PATTERN");
        }
      }
    }
    const auto backward = patternForwardBackward(
        config, input, target, initial, pattern);
    if (!std::isfinite(backward.loss) || !finiteParameters(backward.gradients))
      return fail("PATTERN_BACKWARD_FINITE");
    for (const auto& item : tiny::parameterRegistry(backward.gradients))
      if (inGroup(item.name, Group::Qk) &&
          std::any_of(item.values->begin(), item.values->end(),
                      [](float value) { return value != 0.0f; }))
        return fail("FIXED_PATTERN_QK_GRADIENT");
  }
  P donor = tiny::initialParameters(config, 10);
  P hybrid = initial;
  copyGroup(hybrid, donor, Group::Attention);
  if (!sameGroup(hybrid, donor, Group::Attention, true) ||
      !sameGroup(hybrid, initial, Group::Attention, false))
    return fail("GROUP_COPY_SCOPE");
  P self = initial;
  copyGroup(self, initial, Group::Attention);
  if (!si::sameParameters(self, initial)) return fail("SELF_SWAP_NOOP");
  const P scaledOne = scaledBranch(config, initial, 1.0f, false);
  if (!si::sameParameters(scaledOne, initial)) return fail("ALPHA_ONE_NOOP");
  const P scaledZero = scaledBranch(config, initial, 0.0f, false);
  for (std::uint32_t li = 0; li < config.numLayers; ++li)
    if (std::any_of(train::layer(scaledZero, li).wo.begin(),
                    train::layer(scaledZero, li).wo.end(),
                    [](float value) { return value != 0.0f; }))
      return fail("ALPHA_ZERO_BRANCH");
  const auto runA = runTraining(config, 9, TrainingMode::Canonical, 2);
  const auto runB = si::canonicalPrefix(config, 9, 2, 2);
  if (!si::sameParameters(runA.params, runB.params) ||
      !si::sameParameters(runA.firstMoment, runB.firstMoment) ||
      !si::sameParameters(runA.secondMoment, runB.secondMoment))
    return fail("CANONICAL_TRAINING_PARITY");
  const auto frozen = runTraining(
      config, 9, TrainingMode::FreezeAttentionInitial, 2);
  bool frozenMomentsZero = true;
  for (const auto* state : {&frozen.firstMoment, &frozen.secondMoment})
    for (const auto& item : tiny::parameterRegistry(*state))
      if (inGroup(item.name, Group::Attention))
        frozenMomentsZero = frozenMomentsZero &&
            std::all_of(item.values->begin(), item.values->end(),
                        [](float value) { return value == 0.0f; });
  if (!sameGroup(frozen.params, initial, Group::Attention, true) ||
      !frozenMomentsZero)
    return fail("FREEZE_SCOPE");
  return true;
}

}  // namespace phonelm::attention_minimal_cause
