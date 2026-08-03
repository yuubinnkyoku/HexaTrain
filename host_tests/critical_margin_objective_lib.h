// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "autoregressive_validation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::critical_margin {

namespace ar = phonelm::autoregressive_validation;

inline constexpr std::uint32_t kDatasetSchemaVersion = 1;
inline constexpr const char* kDatasetSchemaName = "MARGIN_DATASET_V1";
inline constexpr const char* kCalibrationName = "MARGIN_CALIBRATION_V1";
inline constexpr const char* kDevelopmentName = "MARGIN_DEVELOPMENT_V1";
inline constexpr double kScoreTieTolerance = 1.0e-7;
inline constexpr std::array<int, 23> kEvaluationSteps{
    0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  48,
    56,  64,  80,  96,  128, 160, 192, 224, 256, 288, 320};

enum class Partition : std::uint32_t {
  CALIBRATION = 0,
  DEVELOPMENT = 1,
};

inline constexpr std::array<Partition, 2> kPartitions{
    Partition::CALIBRATION, Partition::DEVELOPMENT};

inline const char* partitionName(Partition partition) {
  switch (partition) {
    case Partition::CALIBRATION: return kCalibrationName;
    case Partition::DEVELOPMENT: return kDevelopmentName;
  }
  return "UNKNOWN_MARGIN_PARTITION";
}

inline const char* generatorDomain(Partition partition) {
  switch (partition) {
    case Partition::CALIBRATION:
      return "MIXED_PREFIX_SUFFIX_2_6_7_DISTRACTOR_OFFSET_1";
    case Partition::DEVELOPMENT:
      return "MIXED_PREFIX_SUFFIX_2_6_7_DISTRACTOR_OFFSET_2";
  }
  return "UNKNOWN_MARGIN_DOMAIN";
}

inline std::uint32_t distractorOffset(Partition partition) {
  return partition == Partition::CALIBRATION ? 1u : 2u;
}

inline constexpr std::array<std::uint32_t, 3> kActiveSuffixLengths{2, 6, 7};

inline std::vector<ar::Case> cases(Partition partition,
                                   std::uint32_t tokens = 8) {
  std::vector<ar::Case> result;
  if (tokens != 8) return result;
  const auto& families = ar::rules();
  const std::uint32_t offset = distractorOffset(partition);
  for (std::uint32_t active = 0; active < families.size(); ++active) {
    const auto& activeRule = families[active];
    const std::uint32_t distractor = (active + offset) % families.size();
    const auto& distractorRule = families[distractor];
    for (std::uint32_t suffixIndex = 0;
         suffixIndex < kActiveSuffixLengths.size(); ++suffixIndex) {
      const std::uint32_t activeSuffix = kActiveSuffixLengths[suffixIndex];
      for (std::uint32_t variant = 0; variant < 2; ++variant) {
        const std::uint32_t rollout = variant == 0 ? 4u : 8u;
        const std::uint32_t distractorLength = tokens - activeSuffix;
        const std::uint32_t activePhase =
            (active + suffixIndex + 2 * variant) % activeRule.size();
        const std::uint32_t distractorPhase =
            (active + suffixIndex + variant) % distractorRule.size();
        ar::Case item;
        item.id = std::string(partitionName(partition)) + "_active_" +
                  std::to_string(active) + "_suffix_" +
                  std::to_string(activeSuffix) + "_rollout_" +
                  std::to_string(rollout);
        item.domain = generatorDomain(partition);
        item.activeFamily = active;
        item.distractorFamily = distractor;
        item.activePhase = activePhase;
        item.distractorPhase = distractorPhase;
        item.activeSuffixLength = activeSuffix;
        item.rolloutLength = rollout;
        for (std::uint32_t i = 0; i < distractorLength; ++i)
          item.initialPrefix.push_back(
              distractorRule[(distractorPhase + i) % distractorRule.size()]);
        for (std::uint32_t i = 0; i < activeSuffix; ++i)
          item.initialPrefix.push_back(
              activeRule[(activePhase + i) % activeRule.size()]);
        for (std::uint32_t i = 0; i < rollout; ++i)
          item.targets.push_back(activeRule[
              (activePhase + activeSuffix + i) % activeRule.size()]);
        result.push_back(std::move(item));
      }
    }
  }
  return result;
}

inline std::string partitionHash(Partition partition,
                                 std::uint32_t tokens = 8) {
  std::uint64_t hash = 1469598103934665603ull;
  hash = ar::fnv1a(&kDatasetSchemaVersion, sizeof(kDatasetSchemaVersion), hash);
  hash = ar::hashString(kDatasetSchemaName, hash);
  hash = ar::hashString(partitionName(partition), hash);
  hash = ar::hashString(generatorDomain(partition), hash);
  for (const auto& item : cases(partition, tokens)) {
    hash = ar::hashString(item.id, hash);
    hash = ar::hashString(item.domain, hash);
    hash = ar::fnv1a(&item.activeFamily, sizeof(item.activeFamily), hash);
    hash = ar::fnv1a(&item.distractorFamily,
                     sizeof(item.distractorFamily), hash);
    hash = ar::fnv1a(&item.activePhase, sizeof(item.activePhase), hash);
    hash = ar::fnv1a(&item.distractorPhase,
                     sizeof(item.distractorPhase), hash);
    hash = ar::fnv1a(&item.activeSuffixLength,
                     sizeof(item.activeSuffixLength), hash);
    hash = ar::fnv1a(&item.rolloutLength, sizeof(item.rolloutLength), hash);
    hash = ar::hashTokens(item.initialPrefix, hash);
    hash = ar::hashTokens(item.targets, hash);
  }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex;
  output.width(16);
  output.fill('0');
  output << hash;
  return output.str();
}

// Pinned from the checked-in generator before any model evaluation ran.
// Computed by two independent audits of the unmodified generator (FNV-1a
// over schema version, names, domains, and every case field). FNV-1a is a
// deterministic identity check, not an authenticity primitive; changing the
// generator invalidates these values.
inline constexpr const char* kCalibrationHash = "fnv1a64:71806d5bf19c090a";
inline constexpr const char* kDevelopmentHash = "fnv1a64:f06fcc3e2d12ca99";

inline const char* pinnedPartitionHash(Partition partition) {
  return partition == Partition::CALIBRATION ? kCalibrationHash
                                             : kDevelopmentHash;
}

using Transition = ar::Transition;

inline std::vector<Transition> targetTransitions(
    const std::vector<ar::Case>& values, bool teacherForcedRows) {
  std::vector<Transition> result;
  for (const auto& item : values) {
    if (teacherForcedRows) {
      if (item.initialPrefix.size() != item.targets.size()) continue;
      for (std::size_t i = 0; i < item.targets.size(); ++i)
        result.push_back({item.initialPrefix[i], item.targets[i]});
      continue;
    }
    if (item.initialPrefix.empty()) continue;
    std::uint32_t previous = item.initialPrefix.back();
    for (const auto target : item.targets) {
      result.push_back({previous, target});
      previous = target;
    }
  }
  return result;
}

struct OverlapStats {
  std::size_t caseId = 0;
  std::size_t initialPrefix = 0;
  std::size_t fullSequence = 0;
  std::size_t uniqueTransitions = 0;
  std::size_t transitionOccurrences = 0;
};

inline OverlapStats overlap(const std::vector<ar::Case>& left,
                            bool leftTeacherForced,
                            const std::vector<ar::Case>& right,
                            bool rightTeacherForced) {
  OverlapStats result;
  std::set<std::string> ids, prefixes, sequences;
  for (const auto& item : left) {
    ids.insert(item.id);
    prefixes.insert(ar::tokenKey(item.initialPrefix));
    sequences.insert(ar::tokenKey(ar::fullSequence(item)));
  }
  for (const auto& item : right) {
    result.caseId += ids.count(item.id);
    result.initialPrefix += prefixes.count(ar::tokenKey(item.initialPrefix));
    result.fullSequence += sequences.count(ar::tokenKey(ar::fullSequence(item)));
  }
  const auto leftTransitions = targetTransitions(left, leftTeacherForced);
  const auto rightTransitions = targetTransitions(right, rightTeacherForced);
  const std::set<Transition> leftUnique(leftTransitions.begin(),
                                        leftTransitions.end());
  const std::set<Transition> rightUnique(rightTransitions.begin(),
                                         rightTransitions.end());
  for (const auto& transition : leftUnique)
    result.uniqueTransitions += rightUnique.count(transition);
  std::map<Transition, std::size_t> leftCounts, rightCounts;
  for (const auto& value : leftTransitions) ++leftCounts[value];
  for (const auto& value : rightTransitions) ++rightCounts[value];
  for (const auto& entry : leftCounts) {
    const auto found = rightCounts.find(entry.first);
    if (found != rightCounts.end())
      result.transitionOccurrences += std::min(entry.second, found->second);
  }
  return result;
}

inline bool validateDatasets(std::uint32_t tokens, std::string* error,
                             bool requirePinnedHashes = true) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  if (tokens != 8) return fail("MARGIN_DATASET_V1_REQUIRES_T8");
  std::string arError;
  if (!ar::validatePartitions(tokens, &arError))
    return fail("AR_V3_" + arError);
  for (const auto partition : kPartitions) {
    const auto generated = cases(partition, tokens);
    if (generated.size() != 24) return fail("MARGIN_CASE_COUNT");
    std::set<std::string> ids, prefixes, sequences;
    for (const auto& item : generated) {
      if (item.initialPrefix.size() != tokens || item.targets.empty())
        return fail("MARGIN_CASE_SHAPE");
      if (!ids.insert(item.id).second) return fail("MARGIN_CASE_ID_DUPLICATE");
      if (!prefixes.insert(ar::tokenKey(item.initialPrefix)).second)
        return fail("MARGIN_PREFIX_DUPLICATE");
      if (!sequences.insert(ar::tokenKey(ar::fullSequence(item))).second)
        return fail("MARGIN_SEQUENCE_DUPLICATE");
    }
    const auto transitions = targetTransitions(generated, false);
    const std::set<Transition> unique(transitions.begin(), transitions.end());
    // The fresh-partition transition contract is shared with the AR V3
    // generator; reference its pinned constants instead of re-hardcoding.
    if (transitions.size() != ar::kExpectedFreshTargetOccurrences ||
        unique.size() != ar::kExpectedTargetTransitionCount)
      return fail("MARGIN_TRANSITION_CONTRACT");
    if (requirePinnedHashes &&
        partitionHash(partition, tokens) != pinnedPartitionHash(partition))
      return fail("MARGIN_PARTITION_HASH_MISMATCH");
  }

  struct View {
    std::vector<ar::Case> values;
    bool teacherForced = false;
  };
  std::vector<View> views;
  for (const auto partition : ar::kPartitions)
    views.push_back({ar::cases(partition, tokens),
                     partition == ar::Partition::TRAIN});
  for (const auto partition : kPartitions)
    views.push_back({cases(partition, tokens), false});
  for (std::size_t i = 0; i < views.size(); ++i)
    for (std::size_t j = i + 1; j < views.size(); ++j) {
      const auto stats = overlap(views[i].values, views[i].teacherForced,
                                 views[j].values, views[j].teacherForced);
      if (stats.caseId != 0 || stats.initialPrefix != 0 ||
          stats.fullSequence != 0)
        return fail("MARGIN_GLOBAL_PARTITION_OVERLAP");
    }
  return true;
}

struct CaseTrace {
  std::string id;
  // Per-token expected-token margin: gold logit minus best-competitor
  // (non-gold) logit for the free-running rollout context at that position.
  // Positive means the gold token is strictly on top; this matches
  // margin_analysis_lib.h expectedMinusTop1Margin. All objectives below
  // minimize the corresponding deficit (max(0, delta - margin)).
  std::vector<double> margins;
  double autoregressiveNllSum = 0.0;
  std::uint64_t tokenExact = 0;
  bool sequenceExact = false;
  // One-based. -1 means no error.
  int firstErrorPosition = -1;
  bool finite = false;
};

inline double lowerMiddleMedian(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[(values.size() - 1) / 2];
}

inline double lowerTailMean(std::vector<double> values, double quantile) {
  if (values.empty() || !(quantile > 0.0 && quantile <= 1.0))
    return std::numeric_limits<double>::quiet_NaN();
  for (const double value : values)
    if (!std::isfinite(value))
      return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const std::size_t count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(quantile * values.size())));
  return std::accumulate(values.begin(), values.begin() + count, 0.0) /
         static_cast<double>(count);
}

struct CheckpointMetrics {
  int step = 0;
  std::vector<CaseTrace> cases;
  double autoregressiveNll = std::numeric_limits<double>::infinity();
  std::uint64_t tokenExact = 0;
  std::uint64_t tokenTotal = 0;
  std::uint64_t sequenceExact = 0;
  std::uint64_t sequenceTotal = 0;
  double medianFirstErrorSurvival = 0.0;
  double lowerTailMarginQ10 = -std::numeric_limits<double>::infinity();
  bool allFinite = false;
};

inline CheckpointMetrics summarizeCheckpoint(int step,
                                             std::vector<CaseTrace> traces) {
  CheckpointMetrics result;
  result.step = step;
  result.cases = std::move(traces);
  if (result.cases.empty()) return result;
  double nll = 0.0;
  std::vector<double> margins;
  std::vector<double> firstErrorSurvival;
  bool finite = true;
  for (const auto& item : result.cases) {
    const bool itemFinite = item.finite && !item.id.empty() &&
                            !item.margins.empty() &&
                            std::isfinite(item.autoregressiveNllSum) &&
                            item.tokenExact <= item.margins.size();
    finite = finite && itemFinite;
    if (!itemFinite) continue;
    nll += item.autoregressiveNllSum;
    result.tokenExact += item.tokenExact;
    result.tokenTotal += item.margins.size();
    result.sequenceExact += item.sequenceExact ? 1u : 0u;
    ++result.sequenceTotal;
    margins.insert(margins.end(), item.margins.begin(), item.margins.end());
    firstErrorSurvival.push_back(
        item.firstErrorPosition < 0
            ? static_cast<double>(item.margins.size() + 1)
            : static_cast<double>(item.firstErrorPosition));
  }
  result.allFinite = finite && result.tokenTotal > 0 &&
                     result.sequenceTotal == result.cases.size();
  if (!result.allFinite) return result;
  result.autoregressiveNll = nll / static_cast<double>(result.tokenTotal);
  result.medianFirstErrorSurvival = lowerMiddleMedian(firstErrorSurvival);
  result.lowerTailMarginQ10 = lowerTailMean(margins, 0.10);
  result.allFinite = std::isfinite(result.autoregressiveNll) &&
                     std::isfinite(result.medianFirstErrorSurvival) &&
                     std::isfinite(result.lowerTailMarginQ10);
  return result;
}

enum class ObjectiveFamily : std::uint32_t {
  MARGIN_DEFICIT_MEAN,
  LOWER_TAIL_MARGIN,
  SOFT_WORST_MARGIN,
  SEQUENCE_SURVIVAL_NLL,
  FIRST_ERROR_HAZARD,
};

struct ObjectiveSpec {
  const char* id;
  ObjectiveFamily family;
  double parameter;
  std::uint32_t priority;
};

inline constexpr std::array<ObjectiveSpec, 12> kObjectives{{
    {"MARGIN_DEFICIT_MEAN_D0", ObjectiveFamily::MARGIN_DEFICIT_MEAN,
     0.0, 0},
    {"MARGIN_DEFICIT_MEAN_D025", ObjectiveFamily::MARGIN_DEFICIT_MEAN,
     0.25, 1},
    {"MARGIN_DEFICIT_MEAN_D05", ObjectiveFamily::MARGIN_DEFICIT_MEAN,
     0.5, 2},
    {"LOWER_TAIL_MARGIN_Q10", ObjectiveFamily::LOWER_TAIL_MARGIN,
     0.10, 3},
    {"LOWER_TAIL_MARGIN_Q20", ObjectiveFamily::LOWER_TAIL_MARGIN,
     0.20, 4},
    {"SOFT_WORST_MARGIN_T025", ObjectiveFamily::SOFT_WORST_MARGIN,
     0.25, 5},
    {"SOFT_WORST_MARGIN_T05", ObjectiveFamily::SOFT_WORST_MARGIN,
     0.5, 6},
    {"SOFT_WORST_MARGIN_T1", ObjectiveFamily::SOFT_WORST_MARGIN,
     1.0, 7},
    {"SEQUENCE_SURVIVAL_NLL_T025",
     ObjectiveFamily::SEQUENCE_SURVIVAL_NLL, 0.25, 8},
    {"SEQUENCE_SURVIVAL_NLL_T05",
     ObjectiveFamily::SEQUENCE_SURVIVAL_NLL, 0.5, 9},
    {"SEQUENCE_SURVIVAL_NLL_T1",
     ObjectiveFamily::SEQUENCE_SURVIVAL_NLL, 1.0, 10},
    {"FIRST_ERROR_HAZARD_INV_POSITION",
     ObjectiveFamily::FIRST_ERROR_HAZARD, 0.0, 11},
}};

struct ObjectiveScore {
  double value = std::numeric_limits<double>::infinity();
  bool finite = false;
};

inline double softplus(double value) {
  if (value > 0.0) return value + std::log1p(std::exp(-value));
  return std::log1p(std::exp(value));
}

inline ObjectiveScore scoreObjective(const ObjectiveSpec& spec,
                                     const CheckpointMetrics& checkpoint) {
  ObjectiveScore result;
  if (!checkpoint.allFinite || checkpoint.cases.empty()) return result;
  std::vector<double> pooledMargins;
  for (const auto& item : checkpoint.cases)
    pooledMargins.insert(pooledMargins.end(), item.margins.begin(),
                         item.margins.end());
  if (pooledMargins.empty()) return result;
  double value = 0.0;
  switch (spec.family) {
    case ObjectiveFamily::MARGIN_DEFICIT_MEAN:
      for (const double margin : pooledMargins)
        value += std::max(0.0, spec.parameter - margin);
      value /= static_cast<double>(pooledMargins.size());
      break;
    case ObjectiveFamily::LOWER_TAIL_MARGIN:
      // All objectives are minimized. Negating the lower-tail mean makes a
      // larger (safer) tail better without changing its units.
      value = -lowerTailMean(pooledMargins, spec.parameter);
      break;
    case ObjectiveFamily::SOFT_WORST_MARGIN:
      for (const auto& item : checkpoint.cases) {
        std::vector<double> scaled;
        scaled.reserve(item.margins.size());
        for (const double margin : item.margins)
          scaled.push_back(std::max(0.0, -margin) / spec.parameter);
        const double maximum =
            *std::max_element(scaled.begin(), scaled.end());
        double exponentialSum = 0.0;
        for (const double itemValue : scaled)
          exponentialSum += std::exp(itemValue - maximum);
        value += spec.parameter *
                 (maximum + std::log(exponentialSum) -
                  std::log(static_cast<double>(scaled.size())));
      }
      value /= static_cast<double>(checkpoint.cases.size());
      break;
    case ObjectiveFamily::SEQUENCE_SURVIVAL_NLL:
      // Per the preregistered definition, sequence length is not normalized:
      // a length-8 sequence must survive twice as many decisions as length 4.
      for (const auto& item : checkpoint.cases)
        for (const double margin : item.margins)
          value += softplus(-margin / spec.parameter);
      value /= static_cast<double>(checkpoint.cases.size());
      break;
    case ObjectiveFamily::FIRST_ERROR_HAZARD:
      for (const auto& item : checkpoint.cases) {
        double weightSum = 0.0;
        for (std::size_t position = 0; position < item.margins.size(); ++position)
          weightSum += 1.0 / static_cast<double>(position + 1);
        for (std::size_t position = 0; position < item.margins.size(); ++position) {
          const double weight =
              (1.0 / static_cast<double>(position + 1)) / weightSum;
          value += weight * std::max(0.0, -item.margins[position]);
        }
      }
      value /= static_cast<double>(checkpoint.cases.size());
      break;
  }
  result.finite = std::isfinite(value);
  if (result.finite) result.value = value;
  return result;
}

inline bool betterCheckpoint(const ObjectiveSpec& objective,
                             const CheckpointMetrics& candidate,
                             const CheckpointMetrics& incumbent) {
  const auto candidateScore = scoreObjective(objective, candidate);
  const auto incumbentScore = scoreObjective(objective, incumbent);
  if (!candidateScore.finite) return false;
  if (!incumbentScore.finite) return true;
  if (candidateScore.value < incumbentScore.value - kScoreTieTolerance)
    return true;
  if (std::abs(candidateScore.value - incumbentScore.value) >
      kScoreTieTolerance)
    return false;
  if (candidate.autoregressiveNll <
      incumbent.autoregressiveNll - kScoreTieTolerance)
    return true;
  if (std::abs(candidate.autoregressiveNll - incumbent.autoregressiveNll) >
      kScoreTieTolerance)
    return false;
  if (candidate.tokenExact != incumbent.tokenExact)
    return candidate.tokenExact > incumbent.tokenExact;
  if (candidate.sequenceExact != incumbent.sequenceExact)
    return candidate.sequenceExact > incumbent.sequenceExact;
  return candidate.step < incumbent.step;
}

inline int selectCheckpoint(const ObjectiveSpec& objective,
                            const std::vector<CheckpointMetrics>& trajectory) {
  int selected = -1;
  CheckpointMetrics incumbent;
  for (std::size_t i = 0; i < trajectory.size(); ++i) {
    if (selected < 0 || betterCheckpoint(objective, trajectory[i], incumbent)) {
      incumbent = trajectory[i];
      selected = static_cast<int>(i);
    }
  }
  return selected;
}

struct Correlation {
  double value = 0.0;
  bool available = false;
};

inline std::vector<double> midranks(const std::vector<double>& values,
                                    double tolerance) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  for (const double value : values)
    if (!std::isfinite(value)) return {};
  std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                    std::size_t right) {
    return values[left] < values[right];
  });
  std::vector<double> result(values.size());
  std::size_t begin = 0;
  while (begin < order.size()) {
    std::size_t end = begin + 1;
    while (end < order.size() &&
           std::abs(values[order[end]] - values[order[end - 1]]) <= tolerance)
      ++end;
    const double rank =
        (static_cast<double>(begin + 1) + static_cast<double>(end)) / 2.0;
    for (std::size_t i = begin; i < end; ++i) result[order[i]] = rank;
    begin = end;
  }
  return result;
}

inline Correlation spearman(const std::vector<double>& left,
                            const std::vector<double>& right,
                            double leftTolerance = kScoreTieTolerance,
                            double rightTolerance = 0.0) {
  Correlation result;
  if (left.size() != right.size() || left.size() < 2) return result;
  const auto leftRanks = midranks(left, leftTolerance);
  const auto rightRanks = midranks(right, rightTolerance);
  if (leftRanks.size() != left.size() || rightRanks.size() != right.size())
    return result;
  const double leftMean = std::accumulate(leftRanks.begin(), leftRanks.end(), 0.0) /
                          static_cast<double>(leftRanks.size());
  const double rightMean =
      std::accumulate(rightRanks.begin(), rightRanks.end(), 0.0) /
      static_cast<double>(rightRanks.size());
  double covariance = 0.0, leftVariance = 0.0, rightVariance = 0.0;
  for (std::size_t i = 0; i < leftRanks.size(); ++i) {
    const double leftCentered = leftRanks[i] - leftMean;
    const double rightCentered = rightRanks[i] - rightMean;
    covariance += leftCentered * rightCentered;
    leftVariance += leftCentered * leftCentered;
    rightVariance += rightCentered * rightCentered;
  }
  if (!(leftVariance > 0.0 && rightVariance > 0.0)) return result;
  result.value = covariance / std::sqrt(leftVariance * rightVariance);
  result.available = std::isfinite(result.value);
  return result;
}

inline bool nonworseExact(const CheckpointMetrics& candidate,
                          const CheckpointMetrics& baseline) {
  return candidate.allFinite && baseline.allFinite &&
         candidate.tokenExact >= baseline.tokenExact &&
         candidate.sequenceExact >= baseline.sequenceExact;
}

inline bool strictlyImprovesExact(const CheckpointMetrics& candidate,
                                  const CheckpointMetrics& baseline) {
  return nonworseExact(candidate, baseline) &&
         (candidate.tokenExact > baseline.tokenExact ||
          candidate.sequenceExact > baseline.sequenceExact);
}

inline bool caseCollapse(const CheckpointMetrics& candidate,
                         const CheckpointMetrics& baseline) {
  if (candidate.cases.size() != baseline.cases.size()) return true;
  for (std::size_t i = 0; i < baseline.cases.size(); ++i) {
    if (candidate.cases[i].id != baseline.cases[i].id) return true;
    if (baseline.cases[i].tokenExact > 0 &&
        candidate.cases[i].tokenExact == 0)
      return true;
  }
  return false;
}

struct DevelopmentRecord {
  std::uint32_t seed = 0;
  CheckpointMetrics candidate;
  CheckpointMetrics finalStep;
  std::vector<CheckpointMetrics> adjacentCheckpoints;
};

struct DevelopmentGateResult {
  bool finite = false;
  bool seed2Strict = false;
  bool pooledTokenNonworse = false;
  bool pooledSequenceNonworse = false;
  bool controlNonworse = false;
  bool firstErrorMedianNonworse = false;
  std::uint32_t supportedSeeds = 0;
  std::uint32_t stableSupportedSeeds = 0;
  bool noCaseCollapse = false;
  bool pass = false;
};

inline double pooledFirstErrorMedian(
    const std::vector<const CheckpointMetrics*>& values) {
  std::vector<double> survival;
  for (const auto* value : values)
    for (const auto& item : value->cases)
      survival.push_back(
          item.firstErrorPosition < 0
              ? static_cast<double>(item.margins.size() + 1)
              : static_cast<double>(item.firstErrorPosition));
  return lowerMiddleMedian(std::move(survival));
}

inline DevelopmentGateResult developmentGate(
    const std::array<DevelopmentRecord, 3>& l19,
    const DevelopmentRecord& l18Control) {
  DevelopmentGateResult result;
  result.finite = l18Control.candidate.allFinite &&
                  l18Control.finalStep.allFinite;
  std::uint64_t candidateTokens = 0, finalTokens = 0;
  std::uint64_t candidateSequences = 0, finalSequences = 0;
  bool collapse = caseCollapse(l18Control.candidate, l18Control.finalStep);
  const DevelopmentRecord* seed2 = nullptr;
  std::vector<const CheckpointMetrics*> candidateMetrics, finalMetrics;
  for (const auto& record : l19) {
    result.finite = result.finite && record.candidate.allFinite &&
                    record.finalStep.allFinite;
    if (record.seed == 2) seed2 = &record;
    candidateTokens += record.candidate.tokenExact;
    finalTokens += record.finalStep.tokenExact;
    candidateSequences += record.candidate.sequenceExact;
    finalSequences += record.finalStep.sequenceExact;
    candidateMetrics.push_back(&record.candidate);
    finalMetrics.push_back(&record.finalStep);
    const bool supported = nonworseExact(record.candidate, record.finalStep);
    if (supported) ++result.supportedSeeds;
    bool stable = false;
    for (const auto& neighbor : record.adjacentCheckpoints)
      stable = stable || nonworseExact(neighbor, record.finalStep);
    if (supported && stable) ++result.stableSupportedSeeds;
    collapse = collapse || caseCollapse(record.candidate, record.finalStep);
  }
  result.seed2Strict = seed2 &&
                       strictlyImprovesExact(seed2->candidate,
                                             seed2->finalStep);
  result.pooledTokenNonworse = candidateTokens >= finalTokens;
  result.pooledSequenceNonworse = candidateSequences >= finalSequences;
  result.controlNonworse =
      nonworseExact(l18Control.candidate, l18Control.finalStep) &&
      l18Control.candidate.medianFirstErrorSurvival >=
          l18Control.finalStep.medianFirstErrorSurvival;
  result.firstErrorMedianNonworse =
      pooledFirstErrorMedian(candidateMetrics) >=
      pooledFirstErrorMedian(finalMetrics);
  result.noCaseCollapse = !collapse;
  result.pass = result.finite && result.seed2Strict &&
                result.pooledTokenNonworse &&
                result.pooledSequenceNonworse && result.controlNonworse &&
                result.firstErrorMedianNonworse &&
                result.supportedSeeds >= 2 &&
                result.stableSupportedSeeds >= 2 && result.noCaseCollapse;
  return result;
}

// ---------------------------------------------------------------------------
// Required API surface: aliases, per-metric helpers, selection, LOSO
// ---------------------------------------------------------------------------
// The margin generator only adds the MARGIN_DATASET_V1 partitions on top of
// the shared AR V3 case record, so the dataset types are aliases.
using DatasetCase = ar::Case;
using DatasetPartition = Partition;
using ObjectiveVariant = ObjectiveSpec;

inline std::vector<DatasetCase> buildMarginCalibrationV1(
    std::uint32_t tokens = 8) {
  return cases(DatasetPartition::CALIBRATION, tokens);
}

inline std::vector<DatasetCase> buildMarginDevelopmentV1(
    std::uint32_t tokens = 8) {
  return cases(DatasetPartition::DEVELOPMENT, tokens);
}

inline std::string hashPartition(DatasetPartition partition,
                                 std::uint32_t tokens = 8) {
  return partitionHash(partition, tokens);
}

inline std::size_t countCaseIdOverlap(const std::vector<DatasetCase>& left,
                                      const std::vector<DatasetCase>& right) {
  return overlap(left, false, right, false).caseId;
}

inline std::size_t countInitialPrefixOverlap(
    const std::vector<DatasetCase>& left,
    const std::vector<DatasetCase>& right) {
  return overlap(left, false, right, false).initialPrefix;
}

inline std::size_t countFullSequenceOverlap(
    const std::vector<DatasetCase>& left,
    const std::vector<DatasetCase>& right) {
  return overlap(left, false, right, false).fullSequence;
}

// Unique successor transitions shared between two partitions. The teacher-
// forced flag selects the TRAIN row-wise transition convention for either
// side; fresh partitions are free-running.
inline std::size_t countTransitionOverlap(
    const std::vector<DatasetCase>& left, bool leftTeacherForced,
    const std::vector<DatasetCase>& right, bool rightTeacherForced) {
  return overlap(left, leftTeacherForced, right, rightTeacherForced)
      .uniqueTransitions;
}

// Occurrence-level (multiset) transition overlap between two partitions.
inline std::size_t countTransitionOccurrenceOverlap(
    const std::vector<DatasetCase>& left, bool leftTeacherForced,
    const std::vector<DatasetCase>& right, bool rightTeacherForced) {
  return overlap(left, leftTeacherForced, right, rightTeacherForced)
      .transitionOccurrences;
}

// Per-token pooled autoregressive NLL of a finite checkpoint.
inline double computeMeanArNll(const CheckpointMetrics& checkpoint) {
  return checkpoint.allFinite
             ? checkpoint.autoregressiveNll
             : std::numeric_limits<double>::quiet_NaN();
}

inline constexpr double kNonFiniteScore = 0.0;  // placeholder guard

inline double scoreOrNaN(const ObjectiveScore& score) {
  return score.finite ? score.value : std::numeric_limits<double>::quiet_NaN();
}

// mean(max(0, delta - margin)) over all pooled tokens; minimized.
inline double computeMarginDeficitMean(const CheckpointMetrics& checkpoint,
                                       double delta) {
  const ObjectiveSpec spec{"MARGIN_DEFICIT_MEAN",
                           ObjectiveFamily::MARGIN_DEFICIT_MEAN, delta, 0};
  return scoreOrNaN(scoreObjective(spec, checkpoint));
}

// Lower-tail (q-quantile) mean of pooled margins; larger is safer. The
// LOWER_TAIL_MARGIN objective minimizes the negated value of this.
inline double computeLowerTailMargin(const CheckpointMetrics& checkpoint,
                                     double quantile) {
  if (!checkpoint.allFinite || !(quantile > 0.0 && quantile <= 1.0))
    return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> pooled;
  for (const auto& item : checkpoint.cases)
    pooled.insert(pooled.end(), item.margins.begin(), item.margins.end());
  return lowerTailMean(pooled, quantile);
}

// Per-case tau * logsumexp(max(0, -margin) / tau), averaged over cases;
// minimized.
inline double computeSoftWorstMargin(const CheckpointMetrics& checkpoint,
                                     double tau) {
  const ObjectiveSpec spec{"SOFT_WORST_MARGIN",
                           ObjectiveFamily::SOFT_WORST_MARGIN, tau, 0};
  return scoreOrNaN(scoreObjective(spec, checkpoint));
}

// Per-case sum(softplus(-margin / tau)), averaged over cases (sequence
// length is intentionally not normalized); minimized.
inline double computeSequenceSurvivalNll(const CheckpointMetrics& checkpoint,
                                         double tau) {
  const ObjectiveSpec spec{"SEQUENCE_SURVIVAL_NLL",
                           ObjectiveFamily::SEQUENCE_SURVIVAL_NLL, tau, 0};
  return scoreOrNaN(scoreObjective(spec, checkpoint));
}

// Per-case 1/position-normalized deficit average (position is 1-based);
// minimized.
inline double computeFirstErrorHazard(const CheckpointMetrics& checkpoint) {
  const ObjectiveSpec spec{"FIRST_ERROR_HAZARD",
                           ObjectiveFamily::FIRST_ERROR_HAZARD, 0.0, 0};
  return scoreOrNaN(scoreObjective(spec, checkpoint));
}

// -1: left strictly better, +1: right strictly better, 0: tie within
// kScoreTieTolerance. Non-finite scores lose; two non-finite scores tie.
inline int compareObjectiveScores(const ObjectiveScore& left,
                                  const ObjectiveScore& right) {
  if (left.finite && !right.finite) return -1;
  if (!left.finite && right.finite) return 1;
  if (!left.finite && !right.finite) return 0;
  if (left.value < right.value - kScoreTieTolerance) return -1;
  if (right.value < left.value - kScoreTieTolerance) return 1;
  return 0;
}

struct SelectionResult {
  std::string objectiveId;
  int trajectoryIndex = -1;  // index into the supplied trajectory
  int step = -1;
  CheckpointMetrics metrics;
  ObjectiveScore score;
  bool selected = false;
};

inline SelectionResult selectBestCheckpoint(
    const ObjectiveSpec& objective,
    const std::vector<CheckpointMetrics>& trajectory) {
  SelectionResult result;
  result.objectiveId = objective.id;
  for (std::size_t i = 0; i < trajectory.size(); ++i) {
    if (!result.selected ||
        betterCheckpoint(objective, trajectory[i], result.metrics)) {
      result.selected = true;
      result.trajectoryIndex = static_cast<int>(i);
      result.step = trajectory[i].step;
      result.metrics = trajectory[i];
    }
  }
  if (result.selected) result.score = scoreObjective(objective, result.metrics);
  return result;
}

// Pooled evidence of one objective variant over a set of seeds, used by the
// fixed pre-evaluation candidate priority (see docs/qnn-l19-critical-margin-
// stabilization.md): development sequence exact, development token exact,
// worst-seed token exact, first-error position, lower-tail margin,
// preregistered variant order, earlier checkpoint.
struct VariantEvidence {
  std::string objectiveId;
  std::uint32_t priority = 0;
  std::uint64_t sequenceExact = 0;
  std::uint64_t tokenExact = 0;
  std::uint64_t worstSeedTokenExact = 0;
  double firstErrorMedian = 0.0;   // lower-middle median of survival; larger better
  double lowerTailMargin = 0.0;    // pooled Q10 margin; larger better
  int selectedStep = 0;            // earliest selected step across seeds
};

inline bool betterVariantEvidence(const VariantEvidence& candidate,
                                  const VariantEvidence& incumbent) {
  if (candidate.sequenceExact != incumbent.sequenceExact)
    return candidate.sequenceExact > incumbent.sequenceExact;
  if (candidate.tokenExact != incumbent.tokenExact)
    return candidate.tokenExact > incumbent.tokenExact;
  if (candidate.worstSeedTokenExact != incumbent.worstSeedTokenExact)
    return candidate.worstSeedTokenExact > incumbent.worstSeedTokenExact;
  if (std::abs(candidate.firstErrorMedian - incumbent.firstErrorMedian) >
      kScoreTieTolerance)
    return candidate.firstErrorMedian > incumbent.firstErrorMedian;
  if (std::abs(candidate.lowerTailMargin - incumbent.lowerTailMargin) >
      kScoreTieTolerance)
    return candidate.lowerTailMargin > incumbent.lowerTailMargin;
  if (candidate.priority != incumbent.priority)
    return candidate.priority < incumbent.priority;
  return candidate.selectedStep < incumbent.selectedStep;
}

struct LeaveOneSeedOutFold {
  std::uint32_t heldOutSeed = 0;
  std::string chosenObjective;
  double chosenParameter = 0.0;
  std::uint32_t chosenPriority = 0;
  int selectedStep = -1;
  std::uint64_t tokenExact = 0;
  std::uint64_t tokenTotal = 0;
  std::uint64_t sequenceExact = 0;
  std::uint64_t sequenceTotal = 0;
  int finalStepTokenExactDelta = 0;  // selected minus FINAL_STEP token exact
  int finalStepSequenceExactDelta = 0;
  bool collapseFree = true;
  bool finite = false;
};

// For each held-out seed, choose the objective variant by the fixed priority
// applied to the remaining seeds' development evidence, select the held-out
// checkpoint on its calibration partition, and report the held-out
// development outcome. Calibration and development trajectories must share
// the same step cadence (same trajectory indices = same steps).
inline std::vector<LeaveOneSeedOutFold> runLeaveOneSeedOut(
    const std::vector<std::uint32_t>& seeds,
    const std::map<std::uint32_t, std::vector<CheckpointMetrics>>&
        calibrationTrajectoryBySeed,
    const std::map<std::uint32_t, std::vector<CheckpointMetrics>>&
        developmentTrajectoryBySeed) {
  std::vector<LeaveOneSeedOutFold> folds;
  for (const std::uint32_t heldOut : seeds) {
    LeaveOneSeedOutFold fold;
    fold.heldOutSeed = heldOut;
    std::vector<VariantEvidence> evidence;
    for (const auto& objective : kObjectives) {
      VariantEvidence item;
      item.objectiveId = objective.id;
      item.priority = objective.priority;
      std::uint64_t worst = std::numeric_limits<std::uint64_t>::max();
      std::vector<double> survivals, tail;
      bool seen = false;
      for (const std::uint32_t trainSeed : seeds) {
        if (trainSeed == heldOut) continue;
        const auto calibration = calibrationTrajectoryBySeed.find(trainSeed);
        const auto development = developmentTrajectoryBySeed.find(trainSeed);
        if (calibration == calibrationTrajectoryBySeed.end() ||
            development == developmentTrajectoryBySeed.end())
          continue;
        if (calibration->second.size() != development->second.size()) continue;
        const auto selection =
            selectBestCheckpoint(objective, calibration->second);
        if (!selection.selected) continue;
        const auto& dev = development->second[
            static_cast<std::size_t>(selection.trajectoryIndex)];
        if (!dev.allFinite) continue;
        seen = true;
        item.sequenceExact += dev.sequenceExact;
        item.tokenExact += dev.tokenExact;
        worst = std::min(worst, dev.tokenExact);
        for (const auto& trace : dev.cases)
          survivals.push_back(
              trace.firstErrorPosition < 0
                  ? static_cast<double>(trace.margins.size() + 1)
                  : static_cast<double>(trace.firstErrorPosition));
        for (const auto& trace : dev.cases)
          tail.insert(tail.end(), trace.margins.begin(), trace.margins.end());
        if (item.selectedStep == 0 || selection.step < item.selectedStep)
          item.selectedStep = selection.step;
      }
      if (!seen) continue;
      item.worstSeedTokenExact =
          worst == std::numeric_limits<std::uint64_t>::max() ? 0 : worst;
      item.firstErrorMedian = lowerMiddleMedian(std::move(survivals));
      item.lowerTailMargin = lowerTailMean(std::move(tail), 0.10);
      evidence.push_back(std::move(item));
    }
    if (evidence.empty()) {
      folds.push_back(std::move(fold));
      continue;
    }
    const VariantEvidence* best = &evidence.front();
    for (const auto& item : evidence)
      if (betterVariantEvidence(item, *best)) best = &item;
    const auto* chosen =
        std::find_if(kObjectives.begin(), kObjectives.end(),
                     [&](const ObjectiveSpec& spec) {
                       return std::string(spec.id) == best->objectiveId;
                     });
    if (chosen == kObjectives.end()) {
      folds.push_back(std::move(fold));
      continue;
    }
    fold.chosenObjective = chosen->id;
    fold.chosenParameter = chosen->parameter;
    fold.chosenPriority = chosen->priority;
    const auto heldOutCalibration = calibrationTrajectoryBySeed.find(heldOut);
    const auto heldOutDevelopment = developmentTrajectoryBySeed.find(heldOut);
    if (heldOutCalibration == calibrationTrajectoryBySeed.end() ||
        heldOutDevelopment == developmentTrajectoryBySeed.end()) {
      folds.push_back(std::move(fold));
      continue;
    }
    const auto selection =
        selectBestCheckpoint(*chosen, heldOutCalibration->second);
    if (!selection.selected) {
      folds.push_back(std::move(fold));
      continue;
    }
    fold.selectedStep = selection.step;
    const auto& development = heldOutDevelopment->second;
    if (static_cast<std::size_t>(selection.trajectoryIndex) <
        development.size()) {
      const auto& dev =
          development[static_cast<std::size_t>(selection.trajectoryIndex)];
      const auto& finalStep = development.back();
      fold.finite = dev.allFinite && finalStep.allFinite;
      fold.tokenExact = dev.tokenExact;
      fold.tokenTotal = dev.tokenTotal;
      fold.sequenceExact = dev.sequenceExact;
      fold.sequenceTotal = dev.sequenceTotal;
      fold.finalStepTokenExactDelta =
          static_cast<int>(dev.tokenExact) -
          static_cast<int>(finalStep.tokenExact);
      fold.finalStepSequenceExactDelta =
          static_cast<int>(dev.sequenceExact) -
          static_cast<int>(finalStep.sequenceExact);
      fold.collapseFree = !caseCollapse(dev, finalStep);
    }
    folds.push_back(std::move(fold));
  }
  return folds;
}

inline DevelopmentGateResult evaluateDevelopmentGate(
    const std::array<DevelopmentRecord, 3>& l19,
    const DevelopmentRecord& l18Control) {
  return developmentGate(l19, l18Control);
}

}  // namespace phonelm::critical_margin
