// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

// Host-only data interventions for the L19 context-supervision audit.
// This file intentionally exposes no AR_FINAL_HOLDOUT_V3 examples.

#include "attention_minimal_cause_lib.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::context_supervision {

namespace amc = phonelm::attention_minimal_cause;
namespace ar = phonelm::autoregressive_validation;
namespace dq = phonelm::depth_quality;
namespace rp = phonelm::readout_probe;
namespace tiny = phonelm::tiny_lm;
namespace train = phonelm::critical_margin::train;

using P = qnn::TinyTransformerParameters;

enum class Condition {
  Canonical,
  HomogeneousMatched,
  MixedInvariant,
  MixedFirst,
  MixedLast,
};

inline const char* conditionName(Condition condition) {
  switch (condition) {
    case Condition::Canonical: return "CANONICAL_HOMOGENEOUS";
    case Condition::HomogeneousMatched: return "MATCHED_HOMOGENEOUS";
    case Condition::MixedInvariant: return "MIXED_INVARIANCE";
    case Condition::MixedFirst: return "CURRICULUM_MIXED_FIRST";
    case Condition::MixedLast: return "CURRICULUM_MIXED_LAST";
  }
  return "UNKNOWN";
}

struct Batch {
  std::vector<std::uint32_t> inputTokens;
  std::vector<std::uint32_t> targetTokens;
  std::vector<float> input;
  std::vector<float> target;
  bool special = false;
  std::uint32_t activeFamily = 0;
  std::uint32_t distractorFamily = 0;
  std::uint32_t block = 0;
};

inline std::uint32_t successor(std::uint32_t token) {
  for (const auto& rule : ar::rules())
    for (std::size_t i = 0; i < rule.size(); ++i)
      if (rule[i] == token) return rule[(i + 1) % rule.size()];
  throw std::invalid_argument("UNKNOWN_SUCCESSOR_TOKEN");
}

inline void materialize(Batch& batch, const tiny::Config& config) {
  if (batch.inputTokens.size() != config.tokens ||
      batch.targetTokens.size() != config.tokens)
    throw std::invalid_argument("BATCH_TOKEN_COUNT");
  batch.input = tiny::oneHot(batch.inputTokens, config.vocabularySize);
  batch.target = tiny::oneHot(batch.targetTokens, config.vocabularySize);
}

// A dose denominator D means four special batches per 4*D steps.  The four
// special batches are a complete family-balanced group.  In MIXED_INVARIANCE,
// each batch contains six distractor-family tokens followed by two active-
// family tokens.  Across the group, every family contributes one length-six
// segment and one length-two segment.  The latter starts six positions later,
// so its token/target histogram exactly completes the corresponding length-8
// homogeneous control batch.
inline Batch scheduledBatch(const tiny::Config& config, Condition condition,
                            int step, std::uint32_t doseDenominator = 4) {
  if (config.tokens != 8 || step < 1 || doseDenominator == 0)
    throw std::invalid_argument("SCHEDULE_ARGUMENT");
  const std::uint32_t period = 4u * doseDenominator;
  const std::uint32_t within = static_cast<std::uint32_t>(step - 1) % period;
  const bool special = within >= period - 4u;
  if (condition == Condition::Canonical || !special) {
    Batch batch;
    const auto formal = dq::formalBatch(
        config, static_cast<std::uint32_t>((step - 1) % 4), 0);
    batch.input = formal.first;
    batch.target = formal.second;
    batch.special = false;
    batch.activeFamily = static_cast<std::uint32_t>((step - 1) % 4);
    batch.distractorFamily = batch.activeFamily;
    const auto& rule = ar::rules().at(batch.activeFamily);
    for (std::uint32_t i = 0; i < config.tokens; ++i) {
      batch.inputTokens.push_back(rule[i % rule.size()]);
      batch.targetTokens.push_back(rule[(i + 1) % rule.size()]);
    }
    return batch;
  }

  Batch batch;
  batch.special = true;
  batch.block = static_cast<std::uint32_t>(step - 1) / period;
  batch.activeFamily = within - (period - 4u);
  const std::uint32_t distractorOffset = 1u + (batch.block % 2u);
  batch.distractorFamily =
      (batch.activeFamily + distractorOffset) % ar::rules().size();
  const auto basePhase = [&](std::uint32_t family) {
    return batch.block % static_cast<std::uint32_t>(ar::rules().at(family).size());
  };

  if (condition == Condition::HomogeneousMatched) {
    const auto& rule = ar::rules().at(batch.activeFamily);
    const std::uint32_t phase = basePhase(batch.activeFamily);
    for (std::uint32_t i = 0; i < config.tokens; ++i)
      batch.inputTokens.push_back(rule[(phase + i) % rule.size()]);
    batch.distractorFamily = batch.activeFamily;
  } else {
    const auto& distractor = ar::rules().at(batch.distractorFamily);
    const auto& active = ar::rules().at(batch.activeFamily);
    const std::uint32_t distractorPhase = basePhase(batch.distractorFamily);
    const std::uint32_t activePhase =
        (basePhase(batch.activeFamily) + 6u) % active.size();
    for (std::uint32_t i = 0; i < 6; ++i)
      batch.inputTokens.push_back(
          distractor[(distractorPhase + i) % distractor.size()]);
    for (std::uint32_t i = 0; i < 2; ++i)
      batch.inputTokens.push_back(active[(activePhase + i) % active.size()]);
  }
  for (const auto token : batch.inputTokens)
    batch.targetTokens.push_back(successor(token));
  materialize(batch, config);
  return batch;
}

inline std::vector<Batch> trainingSchedule(const tiny::Config& config,
                                           Condition condition,
                                           std::uint32_t doseDenominator,
                                           int steps = 320) {
  std::vector<Batch> source;
  source.reserve(static_cast<std::size_t>(steps));
  const bool curriculum = condition == Condition::MixedFirst ||
                          condition == Condition::MixedLast;
  const Condition sourceCondition =
      curriculum ? Condition::MixedInvariant : condition;
  for (int step = 1; step <= steps; ++step)
    source.push_back(
        scheduledBatch(config, sourceCondition, step, doseDenominator));
  if (!curriculum) return source;
  std::vector<Batch> special;
  std::vector<Batch> homogeneous;
  special.reserve(source.size());
  homogeneous.reserve(source.size());
  for (auto& batch : source) {
    if (batch.special)
      special.push_back(std::move(batch));
    else
      homogeneous.push_back(std::move(batch));
  }
  std::vector<Batch> ordered;
  ordered.reserve(source.size());
  auto append = [&](std::vector<Batch>& values) {
    for (auto& batch : values) ordered.push_back(std::move(batch));
  };
  if (condition == Condition::MixedFirst) {
    append(special);
    append(homogeneous);
  } else {
    append(homogeneous);
    append(special);
  }
  return ordered;
}

inline std::string scheduleHash(const tiny::Config& config,
                                Condition condition,
                                std::uint32_t doseDenominator,
                                int steps = 320) {
  std::uint64_t hash = 1469598103934665603ull;
  const std::uint32_t conditionValue = static_cast<std::uint32_t>(condition);
  hash = ar::fnv1a(&conditionValue, sizeof(conditionValue), hash);
  hash = ar::fnv1a(&doseDenominator, sizeof(doseDenominator), hash);
  for (const auto& batch :
       trainingSchedule(config, condition, doseDenominator, steps)) {
    hash = ar::hashTokens(batch.inputTokens, hash);
    hash = ar::hashTokens(batch.targetTokens, hash);
    hash = ar::fnv1a(&batch.special, sizeof(batch.special), hash);
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

struct Histogram {
  std::array<std::uint64_t, 32> input{};
  std::array<std::uint64_t, 32> target{};
  std::uint64_t examples = 0;
  std::uint64_t specialExamples = 0;
};

inline Histogram scheduleHistogram(const tiny::Config& config,
                                   Condition condition,
                                   std::uint32_t doseDenominator,
                                   int steps = 320) {
  Histogram result;
  for (const auto& batch :
       trainingSchedule(config, condition, doseDenominator, steps)) {
    ++result.examples;
    if (batch.special) ++result.specialExamples;
    for (const auto token : batch.inputTokens) ++result.input.at(token);
    for (const auto token : batch.targetTokens) ++result.target.at(token);
  }
  return result;
}

inline bool sameHistogram(const Histogram& a, const Histogram& b) {
  return a.input == b.input && a.target == b.target &&
         a.examples == b.examples &&
         a.specialExamples == b.specialExamples;
}

inline bool finiteParams(const P& params) {
  for (const auto& item : tiny::parameterRegistry(params))
    for (const auto value : *item.values)
      if (!std::isfinite(value)) return false;
  return true;
}

inline bool validProbabilities(const tiny::Config& config,
                               const std::vector<float>& probabilities) {
  if (probabilities.size() !=
      static_cast<std::size_t>(config.tokens) * config.vocabularySize)
    return false;
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    double sum = 0.0;
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
      const float value = probabilities[
          static_cast<std::size_t>(row) * config.vocabularySize + token];
      if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
      sum += value;
    }
    if (std::abs(sum - 1.0) > 1e-5) return false;
  }
  return true;
}

struct TrainingRun {
  P initial;
  P params;
  P firstMoment;
  P secondMoment;
  double finalLoss = 0.0;
  bool finite = true;
  std::uint64_t exampleCount = 0;
  std::uint64_t specialExampleCount = 0;
  std::string scheduleIdentity;
};

inline TrainingRun runTraining(const tiny::Config& config,
                               std::uint32_t seed,
                               Condition condition,
                               std::uint32_t doseDenominator = 4,
                               int steps = 320) {
  TrainingRun run;
  run.initial = tiny::initialParameters(config, seed);
  run.initial = dq::applyInitStability(config, std::move(run.initial),
                                       dq::StabilityMode::LEGACY);
  run.params = run.initial;
  phonelm::seed_instability::zeroLike(run.firstMoment, run.params);
  phonelm::seed_instability::zeroLike(run.secondMoment, run.params);
  run.scheduleIdentity = scheduleHash(config, condition, doseDenominator, steps);
  const auto schedule =
      trainingSchedule(config, condition, doseDenominator, steps);
  for (int step = 1; step <= steps; ++step) {
    const auto& batch = schedule.at(static_cast<std::size_t>(step - 1));
    const auto fb = tiny::forwardBackward(config, batch.input, batch.target,
                                          run.params, 0.0f);
    const float c1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float c2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float lr = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(dq::StabilityMode::LEGACY), 0.003f,
        static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(steps));
    run.finite = run.finite && std::isfinite(fb.loss) &&
                 finiteParams(fb.gradients) && finiteParams(run.params) &&
                 finiteParams(run.firstMoment) &&
                 finiteParams(run.secondMoment) &&
                 std::all_of(fb.logits.begin(), fb.logits.end(),
                             [](float value) { return std::isfinite(value); }) &&
                 validProbabilities(config, fb.probabilities);
    auto update = tiny::adamUpdate(
        run.params, fb.gradients, run.firstMoment, run.secondMoment, lr,
        0.9f, 0.999f, 1e-8f, c1, c2);
    run.params = std::move(update.next);
    run.firstMoment = std::move(update.firstMoment);
    run.secondMoment = std::move(update.secondMoment);
    run.finalLoss = fb.loss;
    ++run.exampleCount;
    if (batch.special) ++run.specialExampleCount;
    run.finite = run.finite && finiteParams(run.params) &&
                 finiteParams(run.firstMoment) &&
                 finiteParams(run.secondMoment);
  }
  return run;
}

enum class ParameterGroup { Qk, V, O, Ffn, Norm, EmbeddingHead, Other };

inline ParameterGroup parameterGroup(const std::string& name) {
  if (name.ends_with(".wq") || name.ends_with(".wk"))
    return ParameterGroup::Qk;
  if (name.ends_with(".wv")) return ParameterGroup::V;
  if (name.ends_with(".wo")) return ParameterGroup::O;
  if (name.find(".ffn_") != std::string::npos) return ParameterGroup::Ffn;
  if (name.find(".norm") != std::string::npos) return ParameterGroup::Norm;
  if (name == "token_embedding" || name == "output_projection")
    return ParameterGroup::EmbeddingHead;
  return ParameterGroup::Other;
}

inline double groupDeltaNorm(const P& before, const P& after,
                             ParameterGroup group) {
  const auto a = tiny::parameterRegistry(before);
  const auto b = tiny::parameterRegistry(after);
  if (a.size() != b.size()) throw std::runtime_error("REGISTRY_SIZE");
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].name != b[i].name || a[i].values->size() != b[i].values->size())
      throw std::runtime_error("REGISTRY_IDENTITY");
    if (parameterGroup(a[i].name) != group) continue;
    for (std::size_t j = 0; j < a[i].values->size(); ++j) {
      const double delta = static_cast<double>((*b[i].values)[j]) -
                           static_cast<double>((*a[i].values)[j]);
      sum += delta * delta;
    }
  }
  return std::sqrt(sum);
}

struct AttentionBehavior {
  double meanOutputNorm = 0.0;
  double meanEntropy = 0.0;
  double meanSelfMass = 0.0;
  double meanPreviousMass = 0.0;
  double meanFarMass = 0.0;
  double meanNonSelfMass = 0.0;
  std::uint64_t rows = 0;
  bool finite = true;
};

inline AttentionBehavior attentionBehavior(
    const tiny::Config& config, const P& params,
    const std::vector<ar::Case>& cases) {
  AttentionBehavior result;
  const auto rows = rp::teacherForcedRows(cases, config.tokens);
  for (const auto& item : rows) {
    const auto oneHot = tiny::oneHot(item.context, config.vocabularySize);
    const auto forward = train::generalForward(config, oneHot, params);
    for (std::uint32_t li = 0; li < config.numLayers; ++li) {
      const auto& layer = forward.layers.at(li);
      const auto& weights = train::layer(params, li);
      const auto output = train::mm(layer.ctx, weights.wo, config.tokens,
                                    config.dimension, config.dimension);
      double normSq = 0.0;
      for (const auto value : output) normSq += double(value) * value;
      result.meanOutputNorm += std::sqrt(normSq);
      for (std::uint32_t head = 0; head < config.numHeads; ++head) {
        for (std::uint32_t row = 0; row < config.tokens; ++row) {
          const std::size_t base =
              (static_cast<std::size_t>(head) * config.tokens + row) *
              config.tokens;
          double entropy = 0.0;
          double far = 0.0;
          for (std::uint32_t column = 0; column <= row; ++column) {
            const double probability = layer.prob[base + column];
            if (probability > 0.0) entropy -= probability * std::log(probability);
            if (column + 1 < row) far += probability;
          }
          const double self = layer.prob[base + row];
          const double previous = row == 0 ? 0.0 : layer.prob[base + row - 1];
          result.meanEntropy += entropy;
          result.meanSelfMass += self;
          result.meanPreviousMass += previous;
          result.meanFarMass += far;
          result.meanNonSelfMass += 1.0 - self;
          ++result.rows;
        }
      }
    }
  }
  if (result.rows == 0) {
    result.finite = false;
    return result;
  }
  const double denominator = static_cast<double>(result.rows);
  result.meanEntropy /= denominator;
  result.meanSelfMass /= denominator;
  result.meanPreviousMass /= denominator;
  result.meanFarMass /= denominator;
  result.meanNonSelfMass /= denominator;
  result.meanOutputNorm /=
      static_cast<double>(rows.size() * config.numLayers);
  result.finite = std::isfinite(result.meanOutputNorm) &&
                  std::isfinite(result.meanEntropy) &&
                  std::isfinite(result.meanSelfMass) &&
                  std::isfinite(result.meanPreviousMass) &&
                  std::isfinite(result.meanFarMass) &&
                  std::isfinite(result.meanNonSelfMass);
  return result;
}

inline bool selfTest(std::string* error = nullptr) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  tiny::Config config;
  config.vocabularySize = 32;
  config.tokens = 8;
  config.dimension = 8;
  config.feedForwardDimension = 16;
  config.numLayers = 2;
  config.numHeads = 2;
  for (const std::uint32_t denominator : {4u, 8u, 16u}) {
    const int steps = static_cast<int>(denominator * 4u * 2u);
    const auto homogeneous = scheduleHistogram(
        config, Condition::HomogeneousMatched, denominator, steps);
    const auto mixed = scheduleHistogram(
        config, Condition::MixedInvariant, denominator, steps);
    if (!sameHistogram(homogeneous, mixed))
      return fail("MATCHED_HISTOGRAM");
    if (homogeneous.specialExamples != 8 || mixed.specialExamples != 8)
      return fail("SPECIAL_EXAMPLE_COUNT");
  }
  if (scheduleHash(config, Condition::Canonical, 4, 320) !=
          "fnv1a64:e05aa08cad24a127" ||
      scheduleHash(config, Condition::HomogeneousMatched, 4, 320) !=
          "fnv1a64:ae785146103c9e87" ||
      scheduleHash(config, Condition::MixedInvariant, 4, 320) !=
          "fnv1a64:137bc62edb9199d0" ||
      scheduleHash(config, Condition::MixedFirst, 4, 320) !=
          "fnv1a64:4ef497bc24a1c641" ||
      scheduleHash(config, Condition::MixedLast, 4, 320) !=
          "fnv1a64:94f00a59ce4828e2" ||
      scheduleHash(config, Condition::MixedLast, 8, 320) !=
          "fnv1a64:6884f3404a78a015" ||
      scheduleHash(config, Condition::MixedLast, 16, 320) !=
          "fnv1a64:4d89f88e0ce0617c")
    return fail("SCHEDULE_GOLDEN_IDENTITY");
  const auto mixed = scheduledBatch(config, Condition::MixedInvariant, 13, 4);
  if (!mixed.special || mixed.inputTokens.size() != 8 ||
      mixed.activeFamily == mixed.distractorFamily)
    return fail("MIXED_PREFIX_CONTRACT");
  for (std::size_t i = 0; i < mixed.inputTokens.size(); ++i)
    if (mixed.targetTokens[i] != successor(mixed.inputTokens[i]))
      return fail("CURRENT_TOKEN_TARGET_CONTRACT");
  std::vector<float> valid(static_cast<std::size_t>(config.tokens) *
                               config.vocabularySize,
                           0.0f);
  for (std::uint32_t row = 0; row < config.tokens; ++row)
    valid[static_cast<std::size_t>(row) * config.vocabularySize] = 1.0f;
  if (!validProbabilities(config, valid))
    return fail("PROBABILITY_VALID_CONTROL");
  auto negative = valid;
  negative[0] = -0.1f;
  if (validProbabilities(config, negative))
    return fail("PROBABILITY_NEGATIVE_REJECTION");
  auto nonNormalized = valid;
  nonNormalized[0] = 0.9f;
  if (validProbabilities(config, nonNormalized))
    return fail("PROBABILITY_ROW_SUM_REJECTION");
  const auto canonical = runTraining(config, 9, Condition::Canonical, 4, 4);
  const auto reference = amc::runTraining(
      config, 9, amc::TrainingMode::Canonical, 4);
  if (!phonelm::seed_instability::sameParameters(canonical.params,
                                                 reference.params) ||
      !phonelm::seed_instability::sameParameters(canonical.firstMoment,
                                                 reference.firstMoment) ||
      !phonelm::seed_instability::sameParameters(canonical.secondMoment,
                                                 reference.secondMoment))
    return fail("CANONICAL_BITWISE_PARITY");
  const auto runA = runTraining(config, 9, Condition::MixedInvariant, 4, 8);
  const auto runB = runTraining(config, 9, Condition::MixedInvariant, 4, 8);
  if (!phonelm::seed_instability::sameParameters(runA.params, runB.params) ||
      runA.finalLoss != runB.finalLoss)
    return fail("DETERMINISTIC_TRAINING_RERUN");
  if (!runA.finite || groupDeltaNorm(runA.initial, runA.params,
                                    ParameterGroup::Qk) == 0.0 ||
      groupDeltaNorm(runA.initial, runA.params, ParameterGroup::V) == 0.0 ||
      groupDeltaNorm(runA.initial, runA.params, ParameterGroup::O) == 0.0)
    return fail("ORDINARY_ATTENTION_UPDATE");
  const auto interleaved = scheduleHistogram(
      config, Condition::MixedInvariant, 4, 32);
  const auto mixedFirst = scheduleHistogram(
      config, Condition::MixedFirst, 4, 32);
  const auto mixedLast = scheduleHistogram(
      config, Condition::MixedLast, 4, 32);
  if (!sameHistogram(interleaved, mixedFirst) ||
      !sameHistogram(interleaved, mixedLast) ||
      scheduleHash(config, Condition::MixedFirst, 4, 32) ==
          scheduleHash(config, Condition::MixedLast, 4, 32))
    return fail("CURRICULUM_SAME_MULTISET_ORDER_ONLY");
  return true;
}

}  // namespace phonelm::context_supervision
