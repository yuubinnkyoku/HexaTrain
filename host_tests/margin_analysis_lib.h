// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace phonelm::margin_analysis {

// Pre-committed analysis constants.  They define the fixed decomposition
// conventions before any replay result is read, so no threshold is tuned on
// the outcome (see docs/agent/numerical-evidence.md).
inline constexpr double kMarginTieTolerance = 1.0e-7;
inline constexpr double kRecoveryRunLength = 2.0;
inline constexpr double kEasyProbabilityThreshold = 0.5;
inline constexpr double kPercentStabilityThreshold = 0.02;
inline constexpr double kLog2 = 0.6931471805599453;

struct Score {
  std::uint32_t predicted = 0;
  double expectedRank = 0.0;
  double expectedProbability = 0.0;
  double top1Probability = 0.0;
  double top2Probability = 0.0;
  double expectedMinusTop1Margin = 0.0;
  double top1MinusTop2Margin = 0.0;
  double entropy = 0.0;
  double tokenNll = 0.0;
  bool valid = false;
};

inline std::uint32_t argmaxFirst(const std::vector<double>& logits) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > logits[best]) best = static_cast<std::uint32_t>(i);
  return best;
}

// Midrank (average rank) of the expected token: 1 + #strictly greater
// + 0.5 * #equal competitors.  Rank 1.0 means a unique top position; ties at
// the top map to 1.5, a tie for second to 2.5, and so on.  The argmax path
// keeps the evaluator's strict-greater lowest-token-id rule, so correctness
// (exact counts) and the descriptive rank metric do not share tie artifacts.
inline double expectedRank(const std::vector<double>& logits,
                           std::uint32_t truth) {
  double rank = 1.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (i == truth) continue;
    if (logits[i] > logits[truth]) {
      rank += 1.0;
    } else if (logits[i] == logits[truth]) {
      rank += 0.5;
    }
  }
  return rank;
}

inline double expectedMinusTop1Margin(const std::vector<double>& logits,
                                      std::uint32_t truth) {
  double bestOther = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i)
    if (i != truth) bestOther = std::max(bestOther, logits[i]);
  return logits[truth] - bestOther;
}

inline double top1MinusTop2Margin(const std::vector<double>& logits) {
  double first = -std::numeric_limits<double>::infinity();
  double second = -std::numeric_limits<double>::infinity();
  for (const double z : logits) {
    if (z > first) {
      second = first;
      first = z;
    } else if (z > second) {
      second = z;
    }
  }
  if (!std::isfinite(first) || !std::isfinite(second)) return 0.0;
  return first - second;
}

inline std::pair<double, double> topTwoProbabilities(
    const std::vector<double>& probabilities) {
  double first = -1.0, second = -1.0;
  for (const double p : probabilities) {
    if (p > first) {
      second = first;
      first = p;
    } else if (p > second) {
      second = p;
    }
  }
  return {first, second};
}

inline double entropyOf(const std::vector<double>& probabilities) {
  double h = 0.0;
  for (const double p : probabilities)
    if (p > 0.0) h -= p * std::log(p);
  return h;
}

inline Score scoreFromLogits(const std::vector<double>& logits,
                             const std::vector<double>& probabilities,
                             std::uint32_t truth) {
  Score score;
  score.valid = logits.size() == probabilities.size() && !logits.empty() &&
                truth < logits.size();
  if (!score.valid) return score;
  score.predicted = argmaxFirst(logits);
  score.expectedRank = expectedRank(logits, truth);
  score.expectedProbability = probabilities[truth];
  const auto top = topTwoProbabilities(probabilities);
  score.top1Probability = top.first;
  score.top2Probability = top.second;
  score.expectedMinusTop1Margin = expectedMinusTop1Margin(logits, truth);
  score.top1MinusTop2Margin = top1MinusTop2Margin(logits);
  score.entropy = entropyOf(probabilities);
  score.tokenNll = score.expectedProbability > 0.0
                       ? -std::log(score.expectedProbability)
                       : std::numeric_limits<double>::infinity();
  return score;
}

enum class Bucket {
  BothCorrect,
  SelectedCorrectFinalWrong,
  SelectedWrongFinalCorrect,
  BothWrong,
};

inline const char* bucketName(Bucket bucket) {
  switch (bucket) {
    case Bucket::BothCorrect: return "BOTH_CORRECT";
    case Bucket::SelectedCorrectFinalWrong: return "SELECTED_CORRECT_FINAL_WRONG";
    case Bucket::SelectedWrongFinalCorrect: return "SELECTED_WRONG_FINAL_CORRECT";
    case Bucket::BothWrong: return "BOTH_WRONG";
  }
  return "UNKNOWN_BUCKET";
}

inline Bucket classifyBucket(bool selectedCorrect, bool finalCorrect) {
  if (selectedCorrect && finalCorrect) return Bucket::BothCorrect;
  if (selectedCorrect) return Bucket::SelectedCorrectFinalWrong;
  if (finalCorrect) return Bucket::SelectedWrongFinalCorrect;
  return Bucket::BothWrong;
}

struct FirstErrorInfo {
  int firstError = -1;  // 0-based; -1 means no error
  std::uint32_t wrongCount = 0;
  std::uint32_t postErrorExact = 0;
  bool recoveredK2 = false;
};

inline FirstErrorInfo firstErrorInfo(
    const std::vector<std::uint32_t>& predicted,
    const std::vector<std::uint32_t>& targets) {
  FirstErrorInfo info;
  const std::size_t length = std::min(predicted.size(), targets.size());
  for (std::size_t i = 0; i < length; ++i) {
    if (predicted[i] == targets[i]) {
      if (info.firstError >= 0) ++info.postErrorExact;
    } else {
      ++info.wrongCount;
      if (info.firstError < 0) info.firstError = static_cast<int>(i);
    }
  }
  if (info.firstError >= 0) {
    const std::size_t start = static_cast<std::size_t>(info.firstError + 1);
    for (std::size_t j = start; j + 1 < length; ++j)
      if (predicted[j] == targets[j] && predicted[j + 1] == targets[j + 1]) {
        info.recoveredK2 = true;
        break;
      }
  }
  return info;
}

enum class CaseClass {
  NoError,
  LateSingleError,
  ErrorWithRecovery,
  EarlyIrreversibleDivergence,
  MultipleLocalErrors,
};

inline const char* caseClassName(CaseClass value) {
  switch (value) {
    case CaseClass::NoError: return "NO_ERROR";
    case CaseClass::LateSingleError: return "LATE_SINGLE_ERROR";
    case CaseClass::ErrorWithRecovery: return "ERROR_WITH_RECOVERY";
    case CaseClass::EarlyIrreversibleDivergence:
      return "EARLY_IRREVERSIBLE_DIVERGENCE";
    case CaseClass::MultipleLocalErrors: return "MULTIPLE_LOCAL_ERRORS";
  }
  return "UNKNOWN_CASE_CLASS";
}

// Mutual-exclusive priority: NO_ERROR, LATE_SINGLE_ERROR (sole error within
// the last two positions), ERROR_WITH_RECOVERY (k=2 consecutive re-lock),
// EARLY_IRREVERSIBLE_DIVERGENCE (error in the first half with at most one
// later correct token), MULTIPLE_LOCAL_ERRORS otherwise.
inline CaseClass classifyCase(const FirstErrorInfo& info, std::size_t length) {
  if (info.firstError < 0) return CaseClass::NoError;
  const int fe = info.firstError;
  const int len = static_cast<int>(length);
  if (info.wrongCount == 1 && fe >= len - 2) return CaseClass::LateSingleError;
  if (info.recoveredK2) return CaseClass::ErrorWithRecovery;
  if (fe <= len / 2 - 1 && info.postErrorExact <= 1)
    return CaseClass::EarlyIrreversibleDivergence;
  return CaseClass::MultipleLocalErrors;
}

enum class Attribution {
  PerfectSelected,
  LocalRankingFailure,
  PrefixDriftAmplification,
  PrefixDriftUncorroborated,
  Mixed,
  NoClearAttribution,
};

inline const char* attributionName(Attribution value) {
  switch (value) {
    case Attribution::PerfectSelected: return "PERFECT_SELECTED";
    case Attribution::LocalRankingFailure: return "LOCAL_LOGIT_RANKING_FAILURE";
    case Attribution::PrefixDriftAmplification:
      return "PREFIX_DRIFT_AMPLIFICATION";
    case Attribution::PrefixDriftUncorroborated:
      return "PREFIX_DRIFT_AMPLIFICATION_UNCORROBORATED";
    case Attribution::Mixed: return "MIXED";
    case Attribution::NoClearAttribution: return "NO_CLEAR_ATTRIBUTION";
  }
  return "UNKNOWN_ATTRIBUTION";
}

struct AttributionInput {
  int selectedFirstError = -1;  // 0-based
  int finalFirstError = -1;
  bool selectedTieAtFirstErrorGold = false;
  bool finalTieAtFirstErrorGold = false;
  std::vector<double> selectedGoldRank;
  std::vector<double> finalGoldRank;
  std::vector<double> selectedSelfRank;
  std::vector<double> finalSelfRank;
  std::vector<double> selectedGoldMargin;
  std::vector<double> selectedSelfMargin;
  // Cross-prefix conditions D/E: selected evaluated under the final model's
  // rollout prefix (condition E) and final evaluated under the selected
  // model's rollout prefix (condition D).  They corroborate whether the
  // diverged prefix, not the local ranking, explains a final-exact gain.
  std::vector<double> selectedCrossRank;
  std::vector<double> finalCrossRank;
};

struct AttributionResult {
  Attribution attribution = Attribution::NoClearAttribution;
  bool localHit = false;
  bool driftHit = false;
  bool nearGold = false;
  double deltaRankMean = 0.0;
  double deltaMarginMean = 0.0;
  double rank1CrossFinalOnSelected = -1.0;  // -1 = not computed
  double rank1CrossSelectedOnFinal = -1.0;
  bool corroborationComputed = false;
};

inline AttributionResult attribute(const AttributionInput& input) {
  AttributionResult result;
  if (input.selectedFirstError < 0) {
    result.attribution = Attribution::PerfectSelected;
    return result;
  }
  const std::size_t fe = static_cast<std::size_t>(input.selectedFirstError);
  const std::size_t length = input.selectedGoldRank.size();
  if (fe >= length || length != input.finalGoldRank.size() ||
      length != input.selectedSelfRank.size() ||
      length != input.finalSelfRank.size() ||
      length != input.selectedGoldMargin.size() ||
      length != input.selectedSelfMargin.size()) {
    return result;
  }
  if (!input.selectedTieAtFirstErrorGold && !input.finalTieAtFirstErrorGold &&
      input.selectedGoldRank[fe] > input.finalGoldRank[fe])
    result.localHit = true;
  result.nearGold = !result.localHit;
  if (result.nearGold)
    for (std::size_t i = fe + 1; i < length; ++i)
      if (input.selectedGoldRank[i] > input.finalGoldRank[i])
        result.nearGold = false;
  std::size_t windowSize = 0;
  double rankSum = 0.0, marginSum = 0.0;
  for (std::size_t i = fe + 1; i < length; ++i) {
    ++windowSize;
    rankSum += input.selectedSelfRank[i] - input.selectedGoldRank[i];
    marginSum += input.selectedGoldMargin[i] - input.selectedSelfMargin[i];
  }
  if (windowSize > 0) {
    result.deltaRankMean = rankSum / static_cast<double>(windowSize);
    result.deltaMarginMean = marginSum / static_cast<double>(windowSize);
    if (result.deltaRankMean >= 1.0 ||
        (result.deltaRankMean > 0.0 && result.deltaMarginMean > 0.0))
      result.driftHit = true;
  }
  std::size_t corrSize = 0;
  double rank1D = 0.0, rank1B = 0.0;
  if (input.finalCrossRank.size() == length &&
      input.selectedCrossRank.size() == length) {
    for (std::size_t i = fe + 1; i < length; ++i) {
      ++corrSize;
      if (input.finalCrossRank[i] <= 1.0) rank1D += 1.0;
      if (input.selectedCrossRank[i] <= 1.0) rank1B += 1.0;
    }
  }
  if (corrSize > 0) {
    result.corroborationComputed = true;
    result.rank1CrossFinalOnSelected = rank1D / static_cast<double>(corrSize);
    result.rank1CrossSelectedOnFinal = rank1B / static_cast<double>(corrSize);
  }
  if (result.localHit && result.driftHit) {
    result.attribution = Attribution::Mixed;
  } else if (result.localHit) {
    result.attribution = Attribution::LocalRankingFailure;
  } else if (result.driftHit && result.nearGold) {
    if (result.corroborationComputed &&
        result.rank1CrossFinalOnSelected > result.rank1CrossSelectedOnFinal)
      result.attribution = Attribution::PrefixDriftAmplification;
    else
      result.attribution = Attribution::PrefixDriftUncorroborated;
  } else {
    result.attribution = Attribution::NoClearAttribution;
  }
  return result;
}

struct BucketStats {
  std::uint64_t tokenCount = 0;
  double nllSumSelected = 0.0;
  double nllSumFinal = 0.0;
  double rankSumSelected = 0.0;
  double rankSumFinal = 0.0;
  double marginSumSelected = 0.0;
  double marginSumFinal = 0.0;
  double entropySumSelected = 0.0;
  double entropySumFinal = 0.0;
  std::uint64_t firstErrorPositionsSelected = 0;
  std::uint64_t firstErrorPositionsFinal = 0;

  void add(bool selectedCorrect, bool finalCorrect, const Score& selected,
           const Score& final, bool selectedFirstError, bool finalFirstError) {
    classifyBucket(selectedCorrect, finalCorrect);
    ++tokenCount;
    nllSumSelected += selected.tokenNll;
    nllSumFinal += final.tokenNll;
    rankSumSelected += selected.expectedRank;
    rankSumFinal += final.expectedRank;
    marginSumSelected += selected.expectedMinusTop1Margin;
    marginSumFinal += final.expectedMinusTop1Margin;
    entropySumSelected += selected.entropy;
    entropySumFinal += final.entropy;
    if (selectedFirstError) ++firstErrorPositionsSelected;
    if (finalFirstError) ++firstErrorPositionsFinal;
  }

  double nllContribution() const { return nllSumSelected - nllSumFinal; }
  double meanRankSelected() const {
    return tokenCount == 0 ? 0.0
                           : rankSumSelected / static_cast<double>(tokenCount);
  }
  double meanRankFinal() const {
    return tokenCount == 0 ? 0.0
                           : rankSumFinal / static_cast<double>(tokenCount);
  }
  double meanMarginSelected() const {
    return tokenCount == 0
               ? 0.0
               : marginSumSelected / static_cast<double>(tokenCount);
  }
  double meanMarginFinal() const {
    return tokenCount == 0
               ? 0.0
               : marginSumFinal / static_cast<double>(tokenCount);
  }
  double meanEntropySelected() const {
    return tokenCount == 0
               ? 0.0
               : entropySumSelected / static_cast<double>(tokenCount);
  }
  double meanEntropyFinal() const {
    return tokenCount == 0
               ? 0.0
               : entropySumFinal / static_cast<double>(tokenCount);
  }
};

inline double medianLowerMiddle(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[(values.size() - 1) / 2];
}

// Percent of the total NLL improvement that comes from tokens the final
// checkpoint already predicted correctly.  numerator and denominator are
// nats; the ratio is only meaningful when the denominator is positive and the
// mean per-token difference clears the stability threshold.  Callers always
// report numerator/denominator next to this value.
inline double percentGainShare(double numerator, double denominator,
                               double tokenCount, bool* stable) {
  if (!stable) return 0.0;
  *stable = false;
  if (tokenCount <= 0.0) return 0.0;
  if (denominator <= 0.0 ||
      std::abs(denominator) / tokenCount < kPercentStabilityThreshold)
    return 0.0;
  *stable = true;
  return 100.0 * numerator / denominator;
}

}  // namespace phonelm::margin_analysis
