// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::autoregressive_validation {

// The objective name and partition schema are deliberately independent.  The
// former identifies the metric, while the latter identifies this fixed V3
// generator and is persisted with every checkpoint/report.
inline constexpr std::uint32_t kSchemaVersion = 3;
inline constexpr std::uint32_t kGeneratorSchemaVersion = kSchemaVersion;
inline constexpr const char* kObjectiveName = "AR_ROLLOUT_NLL_V1";
inline constexpr const char* kSchemaName = kObjectiveName;
inline constexpr const char* kValidationSchemaName = "AR_VALIDATION_V3";
inline constexpr const char* kDevelopmentSchemaName = "AR_DEVELOPMENT_V3";
inline constexpr const char* kFinalSchemaName = "AR_FINAL_HOLDOUT_V3";
inline constexpr double kNllTieTolerance = 1.0e-7;

enum class Partition : std::uint32_t {
  TRAIN = 0,
  VALIDATION = 1,
  DEVELOPMENT = 2,
  FINAL = 3,
  // Compatibility spelling used by the report/export layer.  Both names are
  // the same partition and therefore cannot diverge in hashes or ordering.
  FINAL_HOLDOUT = FINAL,
};

inline constexpr std::array<Partition, 4> kPartitions{
    Partition::TRAIN, Partition::VALIDATION, Partition::DEVELOPMENT,
    Partition::FINAL};

inline constexpr const char* schemaName() { return kObjectiveName; }
inline constexpr std::uint32_t schemaVersion() { return kSchemaVersion; }

inline const char* partitionName(Partition partition) {
  switch (partition) {
    case Partition::TRAIN: return "TRAIN";
    case Partition::VALIDATION: return "AR_VALIDATION_V3";
    case Partition::DEVELOPMENT: return "AR_DEVELOPMENT_V3";
    case Partition::FINAL: return "AR_FINAL_HOLDOUT_V3";
  }
  return "UNKNOWN_PARTITION";
}

inline const char* generatorDomain(Partition partition) {
  switch (partition) {
    case Partition::TRAIN: return "HOMOGENEOUS_PHASE0";
    case Partition::VALIDATION: return "MIXED_PREFIX_DISTRACTOR_OFFSET_1";
    case Partition::DEVELOPMENT: return "MIXED_PREFIX_DISTRACTOR_OFFSET_2";
    case Partition::FINAL: return "MIXED_PREFIX_DISTRACTOR_OFFSET_3";
  }
  return "UNKNOWN_DOMAIN";
}

// Short aliases keep call sites readable and make the persisted generator
// domain a single-source value.  They intentionally return string literals
// owned by this header.
inline const char* domain(Partition partition) {
  return generatorDomain(partition);
}

inline const char* partitionDomain(Partition partition) {
  return generatorDomain(partition);
}

inline const std::array<std::vector<std::uint32_t>, 4>& rules() {
  static const std::array<std::vector<std::uint32_t>, 4> value{
      std::vector<std::uint32_t>{0, 1, 2, 3},
      std::vector<std::uint32_t>{4, 5, 6, 7},
      std::vector<std::uint32_t>{8, 9},
      std::vector<std::uint32_t>{10, 11, 12}};
  return value;
}

inline const std::array<std::vector<std::uint32_t>, 4>& activeCycles() {
  return rules();
}

inline const std::array<std::vector<std::uint32_t>, 4>& cycles() {
  return rules();
}

struct Case {
  std::string id;
  std::string domain;
  std::uint32_t activeFamily = 0;
  std::uint32_t distractorFamily = 0;
  std::uint32_t activePhase = 0;
  std::uint32_t distractorPhase = 0;
  std::uint32_t activeSuffixLength = 0;
  std::uint32_t rolloutLength = 0;
  std::vector<std::uint32_t> initialPrefix;
  std::vector<std::uint32_t> targets;
};

using TokenCase = Case;

inline const std::vector<std::uint32_t>& prefix(const Case& item) {
  return item.initialPrefix;
}

inline const std::vector<std::uint32_t>& targetSequence(const Case& item) {
  return item.targets;
}

inline std::vector<std::uint32_t> fullSequence(const Case& item) {
  std::vector<std::uint32_t> sequence = item.initialPrefix;
  sequence.insert(sequence.end(), item.targets.begin(), item.targets.end());
  return sequence;
}

inline std::vector<Case> cases(Partition partition, std::uint32_t tokens = 8) {
  std::vector<Case> result;
  if (tokens < 6) return result;
  const auto& families = rules();
  if (partition == Partition::TRAIN) {
    for (std::uint32_t active = 0; active < families.size(); ++active) {
      const auto& rule = families[active];
      Case item;
      item.id = "train_pattern_" + std::to_string(active);
      item.domain = generatorDomain(partition);
      item.activeFamily = active;
      item.distractorFamily = active;
      item.activePhase = 0;
      item.distractorPhase = 0;
      item.activeSuffixLength = tokens;
      item.rolloutLength = tokens;
      for (std::uint32_t i = 0; i < tokens; ++i) {
        item.initialPrefix.push_back(rule[i % rule.size()]);
        // TRAIN stores the eight teacher-forced row targets from the formal
        // batch, not a free-running continuation.  Keeping them in the case
        // manifest makes its 32 transition occurrences auditable without a
        // second synthetic accounting path.
        item.targets.push_back(rule[(i + 1) % rule.size()]);
      }
      result.push_back(std::move(item));
    }
    return result;
  }

  const std::uint32_t distractorOffset = std::uint32_t(partition);
  for (std::uint32_t active = 0; active < families.size(); ++active) {
    const auto& activeRule = families[active];
    const std::uint32_t distractor = (active + distractorOffset) % 4;
    const auto& distractorRule = families[distractor];
    for (std::uint32_t suffixIndex = 0; suffixIndex < 3; ++suffixIndex) {
      const std::uint32_t activeSuffix = 3 + suffixIndex;
      for (std::uint32_t variant = 0; variant < 2; ++variant) {
        const std::uint32_t rollout = variant == 0 ? 4 : 8;
        const std::uint32_t distractorLength = tokens - activeSuffix;
        const std::uint32_t activePhase =
            (active + suffixIndex + 2 * variant) % activeRule.size();
        const std::uint32_t distractorPhase =
            (active + suffixIndex + variant) % distractorRule.size();
        Case item;
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

inline std::uint64_t fnv1a(const void* data, std::size_t size,
                           std::uint64_t hash = 1469598103934665603ull) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

inline std::uint64_t hashString(const std::string& text, std::uint64_t hash) {
  const std::uint64_t size = text.size();
  hash = fnv1a(&size, sizeof(size), hash);
  return fnv1a(text.data(), text.size(), hash);
}

inline std::uint64_t hashTokens(const std::vector<std::uint32_t>& values,
                                std::uint64_t hash) {
  const std::uint64_t size = values.size();
  hash = fnv1a(&size, sizeof(size), hash);
  return fnv1a(values.data(), values.size() * sizeof(values[0]), hash);
}

inline std::string partitionHash(Partition partition,
                                 std::uint32_t tokens = 8) {
  std::uint64_t hash = 1469598103934665603ull;
  hash = fnv1a(&kSchemaVersion, sizeof(kSchemaVersion), hash);
  hash = hashString(kObjectiveName, hash);
  hash = hashString(partitionName(partition), hash);
  hash = hashString(generatorDomain(partition), hash);
  for (const auto& item : cases(partition, tokens)) {
    hash = hashString(item.id, hash);
    hash = hashString(item.domain, hash);
    hash = fnv1a(&item.activeFamily, sizeof(item.activeFamily), hash);
    hash = fnv1a(&item.distractorFamily, sizeof(item.distractorFamily), hash);
    hash = fnv1a(&item.activePhase, sizeof(item.activePhase), hash);
    hash = fnv1a(&item.distractorPhase, sizeof(item.distractorPhase), hash);
    hash = fnv1a(&item.activeSuffixLength,
                 sizeof(item.activeSuffixLength), hash);
    hash = fnv1a(&item.rolloutLength, sizeof(item.rolloutLength), hash);
    hash = hashTokens(item.initialPrefix, hash);
    hash = hashTokens(item.targets, hash);
  }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << hash;
  return output.str();
}

inline constexpr const char* kTrainHash = "fnv1a64:5a64ca2d1aa7f29f";
inline constexpr const char* kValidationHash = "fnv1a64:aad785bd4dc88dc9";
inline constexpr const char* kDevelopmentHash = "fnv1a64:bd464d2a6e192d36";
inline constexpr const char* kFinalHoldoutHash = "fnv1a64:aa5081e6df658b4a";

inline const char* pinnedPartitionHash(Partition partition) {
  switch (partition) {
    case Partition::TRAIN: return kTrainHash;
    case Partition::VALIDATION: return kValidationHash;
    case Partition::DEVELOPMENT: return kDevelopmentHash;
    case Partition::FINAL: return kFinalHoldoutHash;
  }
  return "UNKNOWN_PARTITION_HASH";
}

inline bool hashMatchesPinned(Partition partition, std::uint32_t tokens = 8) {
  return tokens == 8 && partitionHash(partition, tokens) ==
                            pinnedPartitionHash(partition);
}

inline std::string tokenKey(const std::vector<std::uint32_t>& values) {
  std::ostringstream output;
  output << values.size() << ':';
  for (const auto value : values) output << value << ',';
  return output.str();
}

using Transition = std::pair<std::uint32_t, std::uint32_t>;

inline std::vector<Transition> targetTransitions(Partition partition,
                                                 std::uint32_t tokens = 8) {
  std::vector<Transition> transitions;
  if (partition == Partition::TRAIN) {
    for (const auto& item : cases(partition, tokens))
      for (std::size_t row = 0; row < item.targets.size(); ++row)
        transitions.push_back({item.initialPrefix[row], item.targets[row]});
    return transitions;
  }
  for (const auto& item : cases(partition, tokens)) {
    std::uint32_t previous = item.initialPrefix.back();
    for (const auto target : item.targets) {
      transitions.push_back({previous, target});
      previous = target;
    }
  }
  return transitions;
}

struct OverlapStats {
  std::size_t caseId = 0;
  std::size_t initialPrefix = 0;
  std::size_t fullSequence = 0;
  std::size_t uniqueTargetTransitions = 0;
  std::size_t targetTransitionOccurrences = 0;

  // Descriptive spellings used by CSV/report adapters.  They are populated
  // together with the compact fields above, so all consumers observe exactly
  // the same token-based counts.
  std::size_t caseIdOverlap = 0;
  std::size_t initialPrefixOverlap = 0;
  std::size_t fullSequenceOverlap = 0;
  std::size_t uniqueTransitionOverlap = 0;
  std::size_t transitionOccurrenceMultisetOverlap = 0;
};

inline OverlapStats overlap(Partition left, Partition right,
                            std::uint32_t tokens = 8) {
  OverlapStats result;
  std::set<std::string> leftIds, leftPrefixes, leftSequences;
  for (const auto& item : cases(left, tokens)) {
    leftIds.insert(item.id);
    leftPrefixes.insert(tokenKey(item.initialPrefix));
    leftSequences.insert(tokenKey(fullSequence(item)));
  }
  for (const auto& item : cases(right, tokens)) {
    result.caseId += leftIds.count(item.id);
    result.initialPrefix += leftPrefixes.count(tokenKey(item.initialPrefix));
    result.fullSequence += leftSequences.count(tokenKey(fullSequence(item)));
  }
  const auto leftTransitions = targetTransitions(left, tokens);
  const auto rightTransitions = targetTransitions(right, tokens);
  const std::set<Transition> leftUnique(leftTransitions.begin(),
                                        leftTransitions.end());
  const std::set<Transition> rightUnique(rightTransitions.begin(),
                                         rightTransitions.end());
  for (const auto& transition : leftUnique)
    result.uniqueTargetTransitions += rightUnique.count(transition);
  std::map<Transition, std::size_t> leftCounts, rightCounts;
  for (const auto& transition : leftTransitions) ++leftCounts[transition];
  for (const auto& transition : rightTransitions) ++rightCounts[transition];
  for (const auto& entry : leftCounts) {
    const auto found = rightCounts.find(entry.first);
    if (found != rightCounts.end())
      result.targetTransitionOccurrences +=
          std::min(entry.second, found->second);
  }
  result.caseIdOverlap = result.caseId;
  result.initialPrefixOverlap = result.initialPrefix;
  result.fullSequenceOverlap = result.fullSequence;
  result.uniqueTransitionOverlap = result.uniqueTargetTransitions;
  result.transitionOccurrenceMultisetOverlap =
      result.targetTransitionOccurrences;
  return result;
}

inline std::set<Transition> targetTransitionSet(Partition partition,
                                                std::uint32_t tokens = 8) {
  const auto values = targetTransitions(partition, tokens);
  return std::set<Transition>(values.begin(), values.end());
}

inline std::size_t uniqueTargetTransitionCount(Partition partition,
                                                std::uint32_t tokens = 8) {
  return targetTransitionSet(partition, tokens).size();
}

inline std::size_t targetTransitionOccurrenceCount(
    Partition partition, std::uint32_t tokens = 8) {
  return targetTransitions(partition, tokens).size();
}

inline std::size_t targetTransitionOccurrences(Partition partition,
                                               std::uint32_t tokens = 8) {
  return targetTransitionOccurrenceCount(partition, tokens);
}

inline constexpr std::size_t kExpectedTargetTransitionCount = 13;
inline constexpr std::size_t kExpectedFreshTargetOccurrences = 144;

inline bool verifyFreshTransitionContract(Partition partition,
                                          std::uint32_t tokens = 8) {
  if (partition == Partition::TRAIN) return false;
  return uniqueTargetTransitionCount(partition, tokens) ==
             kExpectedTargetTransitionCount &&
         targetTransitionOccurrenceCount(partition, tokens) ==
             kExpectedFreshTargetOccurrences;
}

inline std::size_t duplicateCaseIds(Partition partition,
                                    std::uint32_t tokens = 8) {
  std::set<std::string> seen;
  std::size_t duplicates = 0;
  for (const auto& item : cases(partition, tokens))
    if (!seen.insert(item.id).second) ++duplicates;
  return duplicates;
}

inline std::size_t duplicateInitialPrefixes(Partition partition,
                                            std::uint32_t tokens = 8) {
  std::set<std::string> seen;
  std::size_t duplicates = 0;
  for (const auto& item : cases(partition, tokens))
    if (!seen.insert(tokenKey(item.initialPrefix)).second) ++duplicates;
  return duplicates;
}

inline std::size_t duplicateFullSequences(Partition partition,
                                          std::uint32_t tokens = 8) {
  std::set<std::string> seen;
  std::size_t duplicates = 0;
  for (const auto& item : cases(partition, tokens))
    if (!seen.insert(tokenKey(fullSequence(item))).second) ++duplicates;
  return duplicates;
}

inline bool verifyGlobalUniqueness(std::uint32_t tokens = 8,
                                   std::string* error = nullptr) {
  auto fail = [&](const char* message) {
    if (error) *error = message;
    return false;
  };
  std::set<std::string> ids, prefixes, sequences;
  for (const auto partition : kPartitions) {
    for (const auto& item : cases(partition, tokens)) {
      if (!ids.insert(item.id).second) return fail("CASE_ID_OVERLAP");
      if (!prefixes.insert(tokenKey(item.initialPrefix)).second)
        return fail("INITIAL_PREFIX_OVERLAP");
      if (!sequences.insert(tokenKey(fullSequence(item))).second)
        return fail("FULL_SEQUENCE_OVERLAP");
    }
  }
  return true;
}

inline bool validateGlobalUniqueness(std::uint32_t tokens = 8,
                                     std::string* error = nullptr) {
  return verifyGlobalUniqueness(tokens, error);
}

inline bool verifyPartitionSeparation(std::uint32_t tokens = 8,
                                      std::string* error = nullptr) {
  return verifyGlobalUniqueness(tokens, error);
}

inline bool validatePartitions(std::uint32_t tokens, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  if (tokens != 8) return fail("AR_VALIDATION_V3_REQUIRES_T8");
  for (const auto partition : kPartitions) {
    const auto generated = cases(partition, tokens);
    const std::size_t expected = partition == Partition::TRAIN ? 4 : 24;
    if (generated.size() != expected) return fail("CASE_COUNT");
    if (duplicateCaseIds(partition, tokens) != 0) return fail("CASE_ID_DUPLICATE");
    if (duplicateInitialPrefixes(partition, tokens) != 0)
      return fail("INITIAL_PREFIX_DUPLICATE");
    if (duplicateFullSequences(partition, tokens) != 0)
      return fail("FULL_SEQUENCE_DUPLICATE");
    if (!hashMatchesPinned(partition, tokens))
      return fail("PARTITION_HASH_MISMATCH");
    for (const auto& item : generated) {
      if (item.initialPrefix.size() != tokens || item.targets.empty())
        return fail("CASE_SHAPE");
      if (item.activeFamily >= rules().size() ||
          item.distractorFamily >= rules().size())
        return fail("FAMILY_RANGE");
      const std::size_t expectedTargets =
          partition == Partition::TRAIN ? tokens : item.rolloutLength;
      if (item.targets.size() != expectedTargets)
        return fail("TARGET_COUNT");
    }
  }
  for (std::size_t i = 0; i < kPartitions.size(); ++i)
    for (std::size_t j = i + 1; j < kPartitions.size(); ++j) {
      const auto stats = overlap(kPartitions[i], kPartitions[j], tokens);
      if (stats.caseId != 0 || stats.initialPrefix != 0 ||
          stats.fullSequence != 0)
        return fail("PARTITION_OVERLAP");
    }
  if (!verifyGlobalUniqueness(tokens, error)) return false;
  for (const auto partition : {Partition::VALIDATION,
                               Partition::DEVELOPMENT, Partition::FINAL})
    if (!verifyFreshTransitionContract(partition, tokens))
      return fail("TARGET_TRANSITION_CONTRACT");
  return true;
}

struct CaseMetrics {
  std::string id;
  double autoregressiveNll = std::numeric_limits<double>::infinity();
  double teacherForcedNll = std::numeric_limits<double>::infinity();
  std::uint32_t tokenExact = 0;
  std::uint32_t tokenTotal = 0;
  bool sequenceExact = false;
  int firstErrorPosition = -1;
  std::uint32_t postErrorRecoveryTokens = 0;
  bool finite = false;
};

inline bool finite(const CaseMetrics& metrics) {
  return metrics.finite && std::isfinite(metrics.autoregressiveNll) &&
         std::isfinite(metrics.teacherForcedNll) && metrics.tokenTotal > 0 &&
         metrics.tokenExact <= metrics.tokenTotal &&
         (metrics.firstErrorPosition == -1 ||
          (metrics.firstErrorPosition >= 1 &&
           metrics.firstErrorPosition <=
               static_cast<int>(metrics.tokenTotal))) &&
         metrics.postErrorRecoveryTokens <= metrics.tokenTotal;
}

struct Metrics {
  double autoregressiveNll = std::numeric_limits<double>::infinity();
  double teacherForcedNll = std::numeric_limits<double>::infinity();
  double autoregressiveTeacherGap = std::numeric_limits<double>::infinity();
  std::uint64_t tokenExact = 0;
  std::uint64_t tokenTotal = 0;
  std::uint64_t sequenceExact = 0;
  std::uint64_t sequenceTotal = 0;
  double meanFirstErrorPosition = 0.0;
  std::uint64_t noErrorCases = 0;
  std::uint64_t postErrorRecoveryTokens = 0;
  bool allFinite = false;
  std::vector<CaseMetrics> perCase;
};

inline bool finite(const Metrics& metrics) {
  return metrics.allFinite && std::isfinite(metrics.autoregressiveNll) &&
         std::isfinite(metrics.teacherForcedNll) &&
         std::isfinite(metrics.autoregressiveTeacherGap) &&
         metrics.tokenTotal > 0 && metrics.sequenceTotal > 0;
}

inline bool isFinite(const Metrics& metrics) { return finite(metrics); }

inline double rolloutNll(const Metrics& metrics) {
  return metrics.autoregressiveNll;
}

inline double teacherNll(const Metrics& metrics) {
  return metrics.teacherForcedNll;
}

inline double teacherForcedNll(const Metrics& metrics) {
  return metrics.teacherForcedNll;
}

inline double rolloutTeacherGap(const Metrics& metrics) {
  return metrics.autoregressiveTeacherGap;
}

inline double tokenExactRate(const Metrics& metrics) {
  return metrics.tokenTotal == 0
             ? 0.0
             : static_cast<double>(metrics.tokenExact) /
                   static_cast<double>(metrics.tokenTotal);
}

inline double sequenceExactRate(const Metrics& metrics) {
  return metrics.sequenceTotal == 0
             ? 0.0
             : static_cast<double>(metrics.sequenceExact) /
                   static_cast<double>(metrics.sequenceTotal);
}

inline Metrics aggregate(const std::vector<CaseMetrics>& perCase) {
  Metrics result;
  result.perCase = perCase;
  if (perCase.empty()) return result;

  double rolloutNllSum = 0.0;
  double teacherNllSum = 0.0;
  double firstErrorSum = 0.0;
  std::uint64_t errorCases = 0;
  bool allFinite = true;
  for (const auto& item : perCase) {
    const bool itemFinite = finite(item);
    allFinite = allFinite && itemFinite;
    result.tokenTotal += item.tokenTotal;
    ++result.sequenceTotal;
    if (itemFinite) {
      result.tokenExact += item.tokenExact;
      result.sequenceExact += item.sequenceExact ? 1u : 0u;
      result.postErrorRecoveryTokens += item.postErrorRecoveryTokens;
      if (item.firstErrorPosition < 0) {
        ++result.noErrorCases;
      } else {
        firstErrorSum += static_cast<double>(item.firstErrorPosition);
        ++errorCases;
      }
      rolloutNllSum += item.autoregressiveNll * item.tokenTotal;
      teacherNllSum += item.teacherForcedNll * item.tokenTotal;
    }
  }
  result.allFinite = allFinite && result.tokenTotal != 0;
  if (!result.allFinite) return result;
  const double denominator = static_cast<double>(result.tokenTotal);
  result.autoregressiveNll = rolloutNllSum / denominator;
  result.teacherForcedNll = teacherNllSum / denominator;
  result.autoregressiveTeacherGap =
      result.autoregressiveNll - result.teacherForcedNll;
  result.meanFirstErrorPosition =
      errorCases == 0 ? 0.0 : firstErrorSum / static_cast<double>(errorCases);
  return result;
}

inline Metrics summarize(const std::vector<CaseMetrics>& perCase) {
  return aggregate(perCase);
}

inline bool nonworse(const Metrics& candidate, const Metrics& baseline) {
  return finite(candidate) && finite(baseline) &&
         candidate.autoregressiveNll <=
             baseline.autoregressiveNll + kNllTieTolerance &&
         candidate.tokenExact >= baseline.tokenExact &&
         candidate.sequenceExact >= baseline.sequenceExact;
}

inline bool strictImprovement(const Metrics& candidate,
                              const Metrics& baseline) {
  return nonworse(candidate, baseline) &&
         (candidate.autoregressiveNll <
              baseline.autoregressiveNll - kNllTieTolerance ||
          candidate.tokenExact > baseline.tokenExact ||
          candidate.sequenceExact > baseline.sequenceExact);
}

inline bool strict(const Metrics& candidate, const Metrics& baseline) {
  return strictImprovement(candidate, baseline);
}

inline bool better(const Metrics& candidate, int candidateStep,
                   const Metrics& incumbent, int incumbentStep) {
  if (!finite(candidate)) return false;
  if (!finite(incumbent)) return true;
  if (candidate.autoregressiveNll <
      incumbent.autoregressiveNll - kNllTieTolerance)
    return true;
  if (std::abs(candidate.autoregressiveNll - incumbent.autoregressiveNll) >
      kNllTieTolerance)
    return false;
  if (candidate.tokenExact != incumbent.tokenExact)
    return candidate.tokenExact > incumbent.tokenExact;
  if (candidate.sequenceExact != incumbent.sequenceExact)
    return candidate.sequenceExact > incumbent.sequenceExact;
  return candidateStep < incumbentStep;
}

}  // namespace phonelm::autoregressive_validation
