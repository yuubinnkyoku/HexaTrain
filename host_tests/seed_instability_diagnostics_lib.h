// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

// Host-only helpers for the L19 seed-instability root-cause investigation.
// This library deliberately does not expose AR_FINAL_HOLDOUT_V3.

#include "readout_probe_lib.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::seed_instability {

namespace ar = phonelm::autoregressive_validation;
namespace rp = phonelm::readout_probe;
namespace train = phonelm::critical_margin::train;

struct TrainingRow {
  std::string caseId;
  std::vector<std::uint32_t> context;
  std::size_t featurePosition = 0;
  std::uint32_t truth = 0;
};

// TRAIN targets are per-position labels, not a continuation sequence.  Keep
// the original formal batch as the context and select the matching causal row.
inline std::vector<TrainingRow> correctedTrainingRows(
    const std::vector<ar::Case>& trainCases, std::uint32_t tokens = 8) {
  std::vector<TrainingRow> rows;
  for (const auto& item : trainCases) {
    if (item.domain != "HOMOGENEOUS_PHASE0" ||
        item.initialPrefix.size() != tokens || item.targets.size() != tokens)
      throw std::invalid_argument("INVALID_TRAIN_PROBE_CASE");
    for (std::size_t i = 0; i < item.targets.size(); ++i)
      rows.push_back({item.id, item.initialPrefix, i, item.targets[i]});
  }
  return rows;
}

inline std::size_t correctedCurrentTokenExact(
    const std::vector<TrainingRow>& rows) {
  const auto& rules = ar::rules();
  std::map<std::uint32_t, std::uint32_t> successor;
  for (const auto& rule : rules)
    for (std::size_t i = 0; i < rule.size(); ++i)
      successor.emplace(rule[i], rule[(i + 1) % rule.size()]);
  std::size_t exact = 0;
  for (const auto& row : rows) {
    if (row.featurePosition >= row.context.size())
      throw std::invalid_argument("TRAIN_FEATURE_POSITION_RANGE");
    const auto it = successor.find(row.context[row.featurePosition]);
    if (it != successor.end() && it->second == row.truth) ++exact;
  }
  return exact;
}

inline std::size_t legacyCurrentTokenExact(
    const std::vector<rp::ProbeRow>& rows) {
  const auto& rules = ar::rules();
  std::map<std::uint32_t, std::uint32_t> successor;
  for (const auto& rule : rules)
    for (std::size_t i = 0; i < rule.size(); ++i)
      successor.emplace(rule[i], rule[(i + 1) % rule.size()]);
  std::size_t exact = 0;
  for (const auto& row : rows) {
    const auto it = successor.find(row.context.back());
    if (it != successor.end() && it->second == row.truth) ++exact;
  }
  return exact;
}

struct ContextAudit {
  std::string contextKind;
  std::size_t occurrences = 0;
  std::size_t uniqueContexts = 0;
  std::size_t ambiguousContexts = 0;
  std::size_t ambiguousOccurrences = 0;
};

inline std::string tokenKey(const std::vector<std::uint32_t>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ':';
    out << values[i];
  }
  return out.str();
}

inline ContextAudit auditContexts(
    const std::vector<std::pair<std::vector<std::uint32_t>,
                                std::uint32_t>>& observations,
    std::size_t order, const std::string& name) {
  ContextAudit result;
  result.contextKind = name;
  std::map<std::string, std::set<std::uint32_t>> targets;
  std::map<std::string, std::size_t> counts;
  for (const auto& observation : observations) {
    if (order == 0) throw std::invalid_argument("CONTEXT_ORDER_RANGE");
    if (observation.first.size() < order) continue;
    ++result.occurrences;
    const std::vector<std::uint32_t> suffix(
        observation.first.end() - static_cast<std::ptrdiff_t>(order),
        observation.first.end());
    const std::string key = tokenKey(suffix);
    targets[key].insert(observation.second);
    ++counts[key];
  }
  result.uniqueContexts = targets.size();
  for (const auto& item : targets)
    if (item.second.size() > 1) {
      ++result.ambiguousContexts;
      result.ambiguousOccurrences += counts[item.first];
    }
  return result;
}

inline ContextAudit auditFullContexts(
    const std::vector<std::pair<std::vector<std::uint32_t>,
                                std::uint32_t>>& observations,
    const std::string& name) {
  ContextAudit result;
  result.contextKind = name;
  std::map<std::string, std::set<std::uint32_t>> targets;
  std::map<std::string, std::size_t> counts;
  for (const auto& observation : observations) {
    if (observation.first.empty()) continue;
    ++result.occurrences;
    const std::string key = tokenKey(observation.first);
    targets[key].insert(observation.second);
    ++counts[key];
  }
  result.uniqueContexts = targets.size();
  for (const auto& item : targets)
    if (item.second.size() > 1) {
      ++result.ambiguousContexts;
      result.ambiguousOccurrences += counts[item.first];
    }
  return result;
}

inline std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>>
evaluationObservations(const std::vector<ar::Case>& cases,
                       std::uint32_t tokens = 8) {
  std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>> result;
  for (const auto& row : rp::teacherForcedRows(cases, tokens))
    result.emplace_back(row.context, row.truth);
  return result;
}

inline std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>>
fullCausalPrefixObservations(const std::vector<ar::Case>& cases) {
  std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>> result;
  for (const auto& item : cases) {
    auto prefix = item.initialPrefix;
    for (const auto truth : item.targets) {
      result.emplace_back(prefix, truth);
      prefix.push_back(truth);
    }
  }
  return result;
}

inline std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>>
trainingObservations(const std::vector<TrainingRow>& rows) {
  std::vector<std::pair<std::vector<std::uint32_t>, std::uint32_t>> result;
  for (const auto& row : rows) {
    std::vector<std::uint32_t> causal(row.context.begin(),
                                     row.context.begin() + row.featurePosition + 1);
    result.emplace_back(std::move(causal), row.truth);
  }
  return result;
}

inline std::vector<ar::Case> homogeneousPhase0Cases(
    std::uint32_t tokens = 8, std::uint32_t rollout = 8) {
  std::vector<ar::Case> result;
  const auto& rules = ar::rules();
  for (std::uint32_t family = 0; family < rules.size(); ++family) {
    const auto& rule = rules[family];
    ar::Case item;
    item.id = "homogeneous_phase0_" + std::to_string(family);
    item.domain = "HOMOGENEOUS_PHASE0_CONTINUATION";
    item.activeFamily = family;
    item.distractorFamily = family;
    item.activePhase = 0;
    item.distractorPhase = 0;
    item.activeSuffixLength = tokens;
    item.rolloutLength = rollout;
    for (std::uint32_t i = 0; i < tokens; ++i)
      item.initialPrefix.push_back(rule[i % rule.size()]);
    for (std::uint32_t i = 0; i < rollout; ++i)
      item.targets.push_back(rule[(tokens + i) % rule.size()]);
    result.push_back(std::move(item));
  }
  return result;
}

struct ContextMetrics {
  std::string dataset;
  rp::TokenMetrics teacherForced;
  critical_margin::CheckpointMetrics freeRunning;
};

struct TrainingState {
  qnn::TinyTransformerParameters params;
  qnn::TinyTransformerParameters firstMoment;
  qnn::TinyTransformerParameters secondMoment;
  int step = 0;
  double lastLoss = 0.0;
  bool finite = true;
};

inline void zeroLike(qnn::TinyTransformerParameters& target,
                     const qnn::TinyTransformerParameters& like) {
  target = like;
  for (const auto& item : tiny_lm::parameterRegistry(target))
    std::fill(const_cast<std::vector<float>*>(item.values)->begin(),
              const_cast<std::vector<float>*>(item.values)->end(), 0.0f);
}

inline TrainingState canonicalPrefix(const tiny_lm::Config& config,
                                     std::uint32_t seed, int endStep,
                                     int totalSteps = 320) {
  TrainingState state;
  state.params = tiny_lm::initialParameters(config, seed);
  state.params = depth_quality::applyInitStability(
      config, std::move(state.params), depth_quality::StabilityMode::LEGACY);
  zeroLike(state.firstMoment, state.params);
  zeroLike(state.secondMoment, state.params);
  for (int step = 1; step <= endStep; ++step) {
    const auto batch = depth_quality::formalBatch(
        config, static_cast<std::uint32_t>((step - 1) % 4), 0);
    const auto fb = tiny_lm::forwardBackward(
        config, batch.first, batch.second, state.params, 0.0f);
    const float c1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float c2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float lr = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(depth_quality::StabilityMode::LEGACY),
        0.003f, static_cast<std::uint32_t>(step),
        static_cast<std::uint32_t>(totalSteps));
    const auto update = tiny_lm::adamUpdate(
        state.params, fb.gradients, state.firstMoment, state.secondMoment, lr,
        .9f, .999f, 1e-8f, c1, c2);
    state.params = update.next;
    state.firstMoment = update.firstMoment;
    state.secondMoment = update.secondMoment;
    state.step = step;
    state.lastLoss = fb.loss;
    state.finite = state.finite && std::isfinite(fb.loss);
  }
  return state;
}

enum class ResumeIntervention {
  FreezeOutputAfterBoundary,
  ResetOutputMomentsAtBoundary,
  ResetAttentionMomentsAtBoundary,
};

inline const char* interventionName(ResumeIntervention intervention) {
  switch (intervention) {
    case ResumeIntervention::FreezeOutputAfterBoundary:
      return "OUTPUT_FREEZE_AFTER_STEP24";
    case ResumeIntervention::ResetOutputMomentsAtBoundary:
      return "OUTPUT_MOMENTS_RESET_AT_STEP24";
    case ResumeIntervention::ResetAttentionMomentsAtBoundary:
      return "ATTENTION_MOMENTS_RESET_AT_STEP24";
  }
  return "UNKNOWN";
}

inline void zeroVector(std::vector<float>& values) {
  std::fill(values.begin(), values.end(), 0.0f);
}

inline void resetOutputMoments(TrainingState& state) {
  zeroVector(state.firstMoment.outputProjection);
  zeroVector(state.secondMoment.outputProjection);
}

inline void resetAttentionMoments(const tiny_lm::Config& config,
                                  TrainingState& state) {
  for (std::uint32_t li = 0; li < config.numLayers; ++li) {
    auto& m = train::layer(state.firstMoment, li);
    auto& v = train::layer(state.secondMoment, li);
    for (auto* values : {&m.wq, &m.wk, &m.wv, &m.wo, &v.wq, &v.wk,
                         &v.wv, &v.wo})
      zeroVector(*values);
  }
}

inline TrainingState continueCanonical(
    const tiny_lm::Config& config, TrainingState state, int endStep,
    ResumeIntervention intervention, int totalSteps = 320) {
  if (state.step < 0 || state.step >= endStep)
    throw std::invalid_argument("RESUME_STEP_RANGE");
  if (intervention == ResumeIntervention::ResetOutputMomentsAtBoundary)
    resetOutputMoments(state);
  if (intervention == ResumeIntervention::ResetAttentionMomentsAtBoundary)
    resetAttentionMoments(config, state);
  for (int step = state.step + 1; step <= endStep; ++step) {
    const auto batch = depth_quality::formalBatch(
        config, static_cast<std::uint32_t>((step - 1) % 4), 0);
    const auto fb = tiny_lm::forwardBackward(
        config, batch.first, batch.second, state.params, 0.0f);
    const float c1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float c2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float lr = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(depth_quality::StabilityMode::LEGACY),
        0.003f, static_cast<std::uint32_t>(step),
        static_cast<std::uint32_t>(totalSteps));
    auto update = tiny_lm::adamUpdate(
        state.params, fb.gradients, state.firstMoment, state.secondMoment, lr,
        .9f, .999f, 1e-8f, c1, c2);
    if (intervention == ResumeIntervention::FreezeOutputAfterBoundary) {
      update.next.outputProjection = state.params.outputProjection;
      update.firstMoment.outputProjection = state.firstMoment.outputProjection;
      update.secondMoment.outputProjection = state.secondMoment.outputProjection;
    }
    state.params = std::move(update.next);
    state.firstMoment = std::move(update.firstMoment);
    state.secondMoment = std::move(update.secondMoment);
    state.step = step;
    state.lastLoss = fb.loss;
    state.finite = state.finite && std::isfinite(fb.loss);
  }
  return state;
}

inline bool sameExceptOutput(const qnn::TinyTransformerParameters& a,
                             const qnn::TinyTransformerParameters& b) {
  const auto ar = tiny_lm::parameterRegistry(a);
  const auto br = tiny_lm::parameterRegistry(b);
  if (ar.size() != br.size()) return false;
  for (std::size_t i = 0; i < ar.size(); ++i) {
    if (ar[i].name != br[i].name) return false;
    if (ar[i].name == "output_projection") continue;
    if (*ar[i].values != *br[i].values) return false;
  }
  return true;
}

inline bool sameParameters(const qnn::TinyTransformerParameters& a,
                           const qnn::TinyTransformerParameters& b) {
  const auto ar = tiny_lm::parameterRegistry(a);
  const auto br = tiny_lm::parameterRegistry(b);
  if (ar.size() != br.size()) return false;
  for (std::size_t i = 0; i < ar.size(); ++i)
    if (ar[i].name != br[i].name || *ar[i].values != *br[i].values)
      return false;
  return true;
}

inline bool hasSuffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

enum class BranchAblation { AttentionZero, FfnZero };

inline bool isBranchParameter(const std::string& name,
                              BranchAblation ablation);

inline bool sameOutsideBranch(const qnn::TinyTransformerParameters& a,
                              const qnn::TinyTransformerParameters& b,
                              BranchAblation ablation) {
  const auto ar = tiny_lm::parameterRegistry(a);
  const auto br = tiny_lm::parameterRegistry(b);
  if (ar.size() != br.size()) return false;
  for (std::size_t i = 0; i < ar.size(); ++i) {
    if (ar[i].name != br[i].name) return false;
    if (!isBranchParameter(ar[i].name, ablation) &&
        *ar[i].values != *br[i].values)
      return false;
  }
  return true;
}

inline bool isBranchParameter(const std::string& name,
                              BranchAblation ablation) {
  if (ablation == BranchAblation::AttentionZero)
    return hasSuffix(name, ".wq") || hasSuffix(name, ".wk") ||
           hasSuffix(name, ".wv") || hasSuffix(name, ".wo");
  return hasSuffix(name, ".ffn_w1") || hasSuffix(name, ".ffn_w2");
}

inline const char* branchAblationName(BranchAblation ablation) {
  return ablation == BranchAblation::AttentionZero
             ? "ATTENTION_BRANCH_ZERO"
             : "FFN_BRANCH_ZERO";
}

inline void zeroBranch(const tiny_lm::Config& config,
                       qnn::TinyTransformerParameters& params,
                       BranchAblation ablation) {
  for (std::uint32_t li = 0; li < config.numLayers; ++li) {
    auto& layer = train::layer(params, li);
    if (ablation == BranchAblation::AttentionZero) {
      for (auto* values : {&layer.wq, &layer.wk, &layer.wv, &layer.wo})
        zeroVector(*values);
    } else {
      for (auto* values : {&layer.w1, &layer.w2}) zeroVector(*values);
    }
  }
}

inline bool branchIsZero(const tiny_lm::Config& config,
                         const qnn::TinyTransformerParameters& params,
                         BranchAblation ablation) {
  for (std::uint32_t li = 0; li < config.numLayers; ++li) {
    const auto& layer = train::layer(params, li);
    if (ablation == BranchAblation::AttentionZero) {
      for (const auto* values : {&layer.wq, &layer.wk, &layer.wv, &layer.wo})
        if (std::any_of(values->begin(), values->end(),
                        [](float value) { return value != 0.0f; }))
          return false;
    } else {
      for (const auto* values : {&layer.w1, &layer.w2})
        if (std::any_of(values->begin(), values->end(),
                        [](float value) { return value != 0.0f; }))
          return false;
    }
  }
  return true;
}

inline TrainingState trainWithBranchAblation(
    const tiny_lm::Config& config, std::uint32_t seed,
    BranchAblation ablation, int endStep = 320, int totalSteps = 320) {
  TrainingState state;
  state.params = tiny_lm::initialParameters(config, seed);
  state.params = depth_quality::applyInitStability(
      config, std::move(state.params), depth_quality::StabilityMode::LEGACY);
  zeroBranch(config, state.params, ablation);
  zeroLike(state.firstMoment, state.params);
  zeroLike(state.secondMoment, state.params);
  for (int step = 1; step <= endStep; ++step) {
    const auto batch = depth_quality::formalBatch(
        config, static_cast<std::uint32_t>((step - 1) % 4), 0);
    const auto fb = tiny_lm::forwardBackward(
        config, batch.first, batch.second, state.params, 0.0f);
    const float c1 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(step))));
    const float c2 = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(step))));
    const float lr = phonelm::stabilityLearningRate(
        static_cast<std::uint32_t>(depth_quality::StabilityMode::LEGACY),
        0.003f, static_cast<std::uint32_t>(step),
        static_cast<std::uint32_t>(totalSteps));
    auto update = tiny_lm::adamUpdate(
        state.params, fb.gradients, state.firstMoment, state.secondMoment, lr,
        .9f, .999f, 1e-8f, c1, c2);
    zeroBranch(config, update.next, ablation);
    zeroBranch(config, update.firstMoment, ablation);
    zeroBranch(config, update.secondMoment, ablation);
    state.params = std::move(update.next);
    state.firstMoment = std::move(update.firstMoment);
    state.secondMoment = std::move(update.secondMoment);
    state.step = step;
    state.lastLoss = fb.loss;
    state.finite = state.finite && std::isfinite(fb.loss);
  }
  state.finite = state.finite && branchIsZero(config, state.params, ablation);
  return state;
}

inline ContextMetrics evaluateContextDataset(
    const tiny_lm::Config& config, const qnn::TinyTransformerParameters& params,
    const std::string& dataset, const std::vector<ar::Case>& cases,
    int step = 320) {
  ContextMetrics result;
  result.dataset = dataset;
  const auto rows = rp::teacherForcedRows(cases, config.tokens);
  result.teacherForced = rp::headTokenMetrics(config, params, rows);
  result.freeRunning = rp::headFreeRunning(config, params, step, cases);
  return result;
}

inline bool selfTest(std::string* error = nullptr) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  const auto trainCases = ar::cases(ar::Partition::TRAIN, 8);
  const auto legacy = rp::teacherForcedRows(trainCases, 8);
  const auto corrected = correctedTrainingRows(trainCases, 8);
  if (legacy.size() != 32 || corrected.size() != 32)
    return fail("TRAIN_ROW_COUNT");
  if (legacyCurrentTokenExact(legacy) != 28)
    return fail("LEGACY_TRAIN_CONFLICT_ANCHOR");
  if (correctedCurrentTokenExact(corrected) != 32)
    return fail("CORRECTED_TRAIN_CONTRACT");
  const auto homogeneous = homogeneousPhase0Cases();
  if (homogeneous.size() != 4) return fail("HOMOGENEOUS_CASE_COUNT");
  for (const auto& item : homogeneous) {
    if (item.initialPrefix.size() != 8 || item.targets.size() != 8)
      return fail("HOMOGENEOUS_CASE_SHAPE");
    const auto& rule = ar::rules().at(item.activeFamily);
    if (item.targets.front() != rule[8 % rule.size()])
      return fail("HOMOGENEOUS_CONTINUATION");
  }
  const auto dev = evaluationObservations(
      ar::cases(ar::Partition::DEVELOPMENT, 8));
  const auto fullDev = fullCausalPrefixObservations(
      ar::cases(ar::Partition::DEVELOPMENT, 8));
  const auto current = auditContexts(dev, 1, "CURRENT_TOKEN");
  const auto prev2 = auditContexts(dev, 2, "PREVIOUS_2_TOKENS");
  if (current.ambiguousContexts != 0 || current.uniqueContexts != 13 ||
      prev2.ambiguousContexts != 0)
    return fail("DEVELOPMENT_CONTEXT_UNIQUENESS");
  if (auditFullContexts(fullDev, "FULL_CAUSAL_PREFIX").ambiguousContexts != 0)
    return fail("DEVELOPMENT_FULL_PREFIX_UNIQUENESS");
  for (const auto partition : {critical_margin::Partition::CALIBRATION,
                               critical_margin::Partition::DEVELOPMENT}) {
    const auto observations = evaluationObservations(
        critical_margin::cases(partition, 8));
    if (auditContexts(observations, 1, "CURRENT_TOKEN").ambiguousContexts != 0 ||
        auditContexts(observations, 2, "PREVIOUS_2_TOKENS").ambiguousContexts != 0 ||
        auditContexts(observations, 8, "FULL_PREFIX").ambiguousContexts != 0)
      return fail("MARGIN_CONTEXT_UNIQUENESS");
    if (auditFullContexts(fullCausalPrefixObservations(
            critical_margin::cases(partition, 8)),
            "FULL_CAUSAL_PREFIX").ambiguousContexts != 0)
      return fail("MARGIN_FULL_PREFIX_UNIQUENESS");
  }
  // Cheap resume parity fixture: an unmodified canonical prefix continued
  // with a no-op output reset must keep parameters intact at the boundary.
  tiny_lm::Config fixture;
  fixture.vocabularySize = 16;
  fixture.tokens = 4;
  fixture.dimension = 8;
  fixture.feedForwardDimension = 16;
  fixture.numLayers = 2;
  fixture.numHeads = 2;
  const auto prefix = canonicalPrefix(fixture, 7, 2, 4);
  const auto reference = depth_quality::runFormalCpu(
      fixture, 7, 4, 0.003f, depth_quality::StabilityMode::LEGACY, {2});
  if (!sameParameters(prefix.params, reference.checkpoints.at(2)) ||
      !sameParameters(prefix.firstMoment, reference.firstMoments.at(2)) ||
      !sameParameters(prefix.secondMoment, reference.secondMoments.at(2)))
    return fail("CANONICAL_PREFIX_FORMAL_RUN_PARITY");
  auto reset = prefix;
  resetOutputMoments(reset);
  if (!sameParameters(reset.params, prefix.params) ||
      !sameExceptOutput(reset.firstMoment, prefix.firstMoment) ||
      !sameExceptOutput(reset.secondMoment, prefix.secondMoment) ||
      std::any_of(reset.firstMoment.outputProjection.begin(),
                  reset.firstMoment.outputProjection.end(),
                  [](float value) { return value != 0.0f; }) ||
      std::any_of(reset.secondMoment.outputProjection.begin(),
                  reset.secondMoment.outputProjection.end(),
                  [](float value) { return value != 0.0f; }))
    return fail("INTERVENTION_BOUNDARY_PARAMETER_SCOPE");
  auto attentionReset = prefix;
  resetAttentionMoments(fixture, attentionReset);
  if (!sameParameters(attentionReset.params, prefix.params) ||
      !sameOutsideBranch(attentionReset.firstMoment, prefix.firstMoment,
                         BranchAblation::AttentionZero) ||
      !sameOutsideBranch(attentionReset.secondMoment, prefix.secondMoment,
                         BranchAblation::AttentionZero) ||
      !branchIsZero(fixture, attentionReset.firstMoment,
                    BranchAblation::AttentionZero) ||
      !branchIsZero(fixture, attentionReset.secondMoment,
                    BranchAblation::AttentionZero))
    return fail("ATTENTION_MOMENT_RESET_SCOPE");
  const auto frozenOutput = continueCanonical(
      fixture, prefix, 4, ResumeIntervention::FreezeOutputAfterBoundary, 4);
  if (frozenOutput.step != 4 ||
      frozenOutput.params.outputProjection != prefix.params.outputProjection ||
      frozenOutput.firstMoment.outputProjection !=
          prefix.firstMoment.outputProjection ||
      frozenOutput.secondMoment.outputProjection !=
          prefix.secondMoment.outputProjection ||
      sameExceptOutput(frozenOutput.params, prefix.params))
    return fail("OUTPUT_FREEZE_CONTINUATION_SCOPE");
  const auto initial = tiny_lm::initialParameters(fixture, 9);
  auto zeroAttention = initial;
  zeroBranch(fixture, zeroAttention, BranchAblation::AttentionZero);
  if (!branchIsZero(fixture, zeroAttention, BranchAblation::AttentionZero) ||
      !sameOutsideBranch(initial, zeroAttention,
                         BranchAblation::AttentionZero))
    return fail("ATTENTION_ZERO_SCOPE");
  auto zeroFfn = initial;
  zeroBranch(fixture, zeroFfn, BranchAblation::FfnZero);
  if (!branchIsZero(fixture, zeroFfn, BranchAblation::FfnZero) ||
      !sameOutsideBranch(initial, zeroFfn, BranchAblation::FfnZero))
    return fail("FFN_ZERO_SCOPE");
  const auto deterministicA = trainWithBranchAblation(
      fixture, 11, BranchAblation::AttentionZero, 4, 4);
  const auto deterministicB = trainWithBranchAblation(
      fixture, 11, BranchAblation::AttentionZero, 4, 4);
  if (!sameParameters(deterministicA.params, deterministicB.params) ||
      !sameParameters(deterministicA.firstMoment,
                      deterministicB.firstMoment) ||
      !sameParameters(deterministicA.secondMoment,
                      deterministicB.secondMoment) ||
      deterministicA.lastLoss != deterministicB.lastLoss)
    return fail("DETERMINISTIC_BRANCH_RERUN");
  if (!branchIsZero(fixture, deterministicA.params,
                    BranchAblation::AttentionZero) ||
      !branchIsZero(fixture, deterministicA.firstMoment,
                    BranchAblation::AttentionZero) ||
      !branchIsZero(fixture, deterministicA.secondMoment,
                    BranchAblation::AttentionZero))
    return fail("ATTENTION_BRANCH_STATE_SCOPE_AFTER_TRAINING");
  return true;
}

}  // namespace phonelm::seed_instability
