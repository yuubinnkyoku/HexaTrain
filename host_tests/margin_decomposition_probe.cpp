// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only CPU replay probe for the L19 first-error/margin decomposition.
// It regenerates the pinned CPU reference checkpoints deterministically,
// replays AR_DEVELOPMENT_V3 at every rollout position under five prefix
// conditions, and writes (a) private per-token records under build/ and
// (b) public-safe aggregate CSVs consumed by the allow-list exporter.
// AR_FINAL_HOLDOUT_V3 is never evaluated: the development partition is the
// only fresh partition read here.
#include "depth_quality_lib.h"
#include "margin_analysis_lib.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;

namespace {

constexpr int kSteps = 320;
constexpr int kTokens = 8;

struct RunSpec {
  const char* publicId;
  int layers;
  std::uint32_t seed;
  int pinnedSelectedStep;
};

const std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", 19, 1, 16},
    {"L19_SEED_2", 19, 2, 4},
    {"L19_SEED_4", 19, 4, 12},
    {"L18_SEED_2_CONTROL", 18, 2, 4},
}};

tiny::Config makeConfig(int layers) {
  tiny::Config c;
  c.tokens = kTokens;
  c.vocabularySize = 32;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = static_cast<std::uint32_t>(layers);
  c.numHeads = 2;
  std::string error;
  if (!tiny::validateConfig(c, &error))
    throw std::runtime_error("invalid tiny LM config: " + error);
  return c;
}

std::string csv(const std::string& value) {
  std::string result = "\"";
  for (const char ch : value) {
    if (ch == '"') result += "\"\"";
    else result += ch;
  }
  result += '"';
  return result;
}

template <typename T>
std::string text(T value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

std::string number(double value) {
  if (!std::isfinite(value)) return "NOT_FINITE";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void row(std::ofstream& output, const std::vector<std::string>& fields) {
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i) output << ',';
    output << csv(fields[i]);
  }
  output << '\n';
}

struct PositionRecord {
  std::vector<std::uint32_t> selfContext;
  ma::Score self;
  ma::Score gold;
  ma::Score cross;
  bool crossValid = false;
  bool prefixMatchesReference = false;
};

struct CaseRecord {
  std::string id;
  std::size_t length = 0;
  std::vector<std::uint32_t> expected;
  std::vector<std::uint32_t> predicted;
  std::vector<PositionRecord> positions;
  ma::FirstErrorInfo firstError;
  ma::CaseClass caseClass = ma::CaseClass::NoError;
};

struct CheckpointResult {
  int step = 0;
  std::string role;
  std::vector<CaseRecord> cases;
};

ma::Score scorePosition(const tiny::Config& config, const dq::Params& params,
                        const std::vector<std::uint32_t>& context,
                        std::uint32_t truth) {
  const std::size_t expectedTensorSize =
      std::size_t(config.tokens) * config.vocabularySize;
  const auto input = tiny::oneHot(context, config.vocabularySize);
  const auto target = tiny::oneHot(
      std::vector<std::uint32_t>(config.tokens, truth), config.vocabularySize);
  tiny::StepResult step;
  try {
    step = tiny::forwardBackward(config, input, target, params, 0.0f);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("forward failed: ") + error.what());
  }
  if (!dq::finiteTensor(step.logits, expectedTensorSize, nullptr) ||
      !dq::finiteTensor(step.probabilities, expectedTensorSize, nullptr)) {
    throw std::runtime_error("non-finite tensor in replay");
  }
  if (!dq::validProbabilityTensor(step.probabilities, config.tokens,
                                  config.vocabularySize)) {
    throw std::runtime_error("invalid probability tensor in replay");
  }
  const std::size_t base =
      std::size_t(config.tokens - 1) * config.vocabularySize;
  const double expectedProbability =
      static_cast<double>(step.probabilities[base + truth]);
  if (!(std::isfinite(expectedProbability) && expectedProbability > 0.0))
    throw std::runtime_error("invalid expected-token probability in replay");
  std::vector<double> logits(step.logits.begin() + base,
                             step.logits.begin() + base +
                                 config.vocabularySize);
  std::vector<double> probabilities(
      step.probabilities.begin() + base,
      step.probabilities.begin() + base + config.vocabularySize);
  return ma::scoreFromLogits(logits, probabilities, truth);
}

std::vector<std::uint32_t> slide(const std::vector<std::uint32_t>& context,
                                 std::uint32_t next) {
  std::vector<std::uint32_t> result(context.begin() + 1, context.end());
  result.push_back(next);
  return result;
}

CheckpointResult evaluateCheckpoint(const tiny::Config& config,
                                    const dq::Params& params, int step,
                                    const std::string& role,
                                    const std::vector<ar::Case>& cases) {
  CheckpointResult result;
  result.step = step;
  result.role = role;
  for (const auto& item : cases) {
    CaseRecord record;
    record.id = item.id;
    record.length = item.targets.size();
    record.expected = item.targets;
    std::vector<std::uint32_t> ownContext = item.initialPrefix;
    std::vector<std::uint32_t> goldContext = item.initialPrefix;
    record.positions.reserve(item.targets.size());
    for (std::size_t position = 0; position < item.targets.size(); ++position) {
      const std::uint32_t truth = item.targets[position];
      PositionRecord positionRecord;
      positionRecord.selfContext = ownContext;
      positionRecord.self = scorePosition(config, params, ownContext, truth);
      positionRecord.gold = scorePosition(config, params, goldContext, truth);
      positionRecord.prefixMatchesReference = ownContext == goldContext;
      record.predicted.push_back(positionRecord.self.predicted);
      ownContext = slide(ownContext, positionRecord.self.predicted);
      goldContext = slide(goldContext, truth);
      record.positions.push_back(std::move(positionRecord));
    }
    record.firstError = ma::firstErrorInfo(record.predicted, record.expected);
    record.caseClass = ma::classifyCase(record.firstError, record.length);
    result.cases.push_back(std::move(record));
  }
  return result;
}

void evaluateCross(const tiny::Config& config, const dq::Params& selected,
                   const dq::Params& final, CheckpointResult* selectedResult,
                   CheckpointResult* finalResult) {
  if (selectedResult->cases.size() != finalResult->cases.size())
    throw std::runtime_error("pair case count mismatch");
  for (std::size_t c = 0; c < selectedResult->cases.size(); ++c) {
    auto& selectedCase = selectedResult->cases[c];
    auto& finalCase = finalResult->cases[c];
    if (selectedCase.id != finalCase.id)
      throw std::runtime_error("pair case id mismatch");
    if (selectedCase.positions.size() != finalCase.positions.size())
      throw std::runtime_error("pair position count mismatch");
    for (std::size_t p = 0; p < selectedCase.positions.size(); ++p) {
      const std::uint32_t truth = selectedCase.expected[p];
      selectedCase.positions[p].cross = scorePosition(
          config, selected, finalCase.positions[p].selfContext, truth);
      selectedCase.positions[p].crossValid = true;
      finalCase.positions[p].cross = scorePosition(
          config, final, selectedCase.positions[p].selfContext, truth);
      finalCase.positions[p].crossValid = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Aggregation (deterministic, public-safe)
// ---------------------------------------------------------------------------

struct PairStats {
  std::array<ma::BucketStats, 4> buckets;
  std::array<std::uint64_t, 4> goldBucketCounts{};
  std::uint64_t tokenCount = 0;
  std::uint64_t selectedExact = 0;
  std::uint64_t finalExact = 0;
  double selectedNllSum = 0.0;
  double finalNllSum = 0.0;
  std::uint64_t easyTokenCount = 0;
  double easyNllGain = 0.0;
  std::uint64_t criticalTokenCount = 0;
  double criticalNllLoss = 0.0;
  std::vector<double> swfcMargin;
  std::vector<double> swfcRank;
  std::vector<double> swfcEntropySelected;
  std::vector<double> swfcEntropyFinal;

  double totalImprovement() const { return finalNllSum - selectedNllSum; }
  double gainFromBothCorrect() const {
    const auto& bucket = buckets[static_cast<std::size_t>(ma::Bucket::BothCorrect)];
    return bucket.nllSumFinal - bucket.nllSumSelected;
  }
  double lossFromSelectedWrongFinalCorrect() const {
    const auto& bucket =
        buckets[static_cast<std::size_t>(ma::Bucket::SelectedWrongFinalCorrect)];
    return bucket.nllSumSelected - bucket.nllSumFinal;
  }
  double selectedNll() const {
    return tokenCount == 0 ? 0.0 : selectedNllSum / static_cast<double>(tokenCount);
  }
  double finalNll() const {
    return tokenCount == 0 ? 0.0 : finalNllSum / static_cast<double>(tokenCount);
  }
};

PairStats pairStats(const CheckpointResult& selected,
                    const CheckpointResult& finalStep) {
  PairStats result;
  for (std::size_t c = 0; c < selected.cases.size(); ++c) {
    const auto& s = selected.cases[c];
    const auto& f = finalStep.cases[c];
    if (s.id != f.id) throw std::runtime_error("pair case id mismatch");
    for (std::size_t p = 0; p < s.positions.size(); ++p) {
      const std::uint32_t truth = s.expected[p];
      const bool sCorrect = s.positions[p].self.predicted == truth;
      const bool fCorrect = f.positions[p].self.predicted == truth;
      const bool sGoldCorrect = s.positions[p].gold.predicted == truth;
      const bool fGoldCorrect = f.positions[p].gold.predicted == truth;
      const ma::Bucket bucket = ma::classifyBucket(sCorrect, fCorrect);
      const ma::Bucket goldBucket =
          ma::classifyBucket(sGoldCorrect, fGoldCorrect);
      const bool sFirstError =
          s.firstError.firstError >= 0 &&
          static_cast<std::size_t>(s.firstError.firstError) == p;
      const bool fFirstError =
          f.firstError.firstError >= 0 &&
          static_cast<std::size_t>(f.firstError.firstError) == p;
      ++result.tokenCount;
      result.buckets[static_cast<std::size_t>(bucket)].add(
          sCorrect, fCorrect, s.positions[p].self, f.positions[p].self,
          sFirstError, fFirstError);
      ++result.goldBucketCounts[static_cast<std::size_t>(goldBucket)];
      if (sCorrect) ++result.selectedExact;
      if (fCorrect) ++result.finalExact;
      result.selectedNllSum += s.positions[p].self.tokenNll;
      result.finalNllSum += f.positions[p].self.tokenNll;
      const bool easy = s.positions[p].self.expectedRank <= 1.0 &&
                        f.positions[p].self.expectedRank <= 1.0 &&
                        s.positions[p].self.expectedProbability >=
                            ma::kEasyProbabilityThreshold &&
                        f.positions[p].self.expectedProbability >=
                            ma::kEasyProbabilityThreshold;
      if (easy) {
        ++result.easyTokenCount;
        result.easyNllGain += f.positions[p].self.tokenNll -
                              s.positions[p].self.tokenNll;
      }
      const bool critical =
          sFirstError || fFirstError ||
          bucket == ma::Bucket::SelectedWrongFinalCorrect;
      if (critical) {
        ++result.criticalTokenCount;
        result.criticalNllLoss += s.positions[p].self.tokenNll -
                                  f.positions[p].self.tokenNll;
      }
      if (bucket == ma::Bucket::SelectedWrongFinalCorrect) {
        result.swfcMargin.push_back(
            s.positions[p].self.expectedMinusTop1Margin);
        result.swfcRank.push_back(s.positions[p].self.expectedRank);
        result.swfcEntropySelected.push_back(s.positions[p].self.entropy);
        result.swfcEntropyFinal.push_back(f.positions[p].self.entropy);
      }
    }
  }
  return result;
}

struct AttributionSummary {
  std::uint64_t perfectSelected = 0;
  std::uint64_t local = 0;
  std::uint64_t drift = 0;
  std::uint64_t driftUncorroborated = 0;
  std::uint64_t mixed = 0;
  std::uint64_t noClear = 0;
};

struct AttributionRow {
  std::string caseId;
  std::size_t length = 0;
  int selectedFirstError = -1;  // 1-based
  int finalFirstError = -1;
  bool localHit = false;
  bool driftHit = false;
  bool nearGold = false;
  double deltaRankMean = 0.0;
  double deltaMarginMean = 0.0;
  double rank1CrossSelectedOnFinalPrefix = -1.0;
  double rank1CrossFinalOnSelectedPrefix = -1.0;
  bool corroborationComputed = false;
  ma::Attribution attribution = ma::Attribution::NoClearAttribution;
};

AttributionSummary attributionSummary(const CheckpointResult& selected,
                                      const CheckpointResult& finalStep,
                                      std::vector<AttributionRow>* rows) {
  AttributionSummary summary;
  for (std::size_t c = 0; c < selected.cases.size(); ++c) {
    const auto& s = selected.cases[c];
    const auto& f = finalStep.cases[c];
    ma::AttributionInput input;
    input.selectedFirstError = s.firstError.firstError;
    input.finalFirstError = f.firstError.firstError;
    const std::size_t length = s.positions.size();
    for (std::size_t p = 0; p < length; ++p) {
      input.selectedGoldRank.push_back(s.positions[p].gold.expectedRank);
      input.finalGoldRank.push_back(f.positions[p].gold.expectedRank);
      input.selectedSelfRank.push_back(s.positions[p].self.expectedRank);
      input.finalSelfRank.push_back(f.positions[p].self.expectedRank);
      input.selectedCrossRank.push_back(
          s.positions[p].crossValid ? s.positions[p].cross.expectedRank : 0.0);
      input.finalCrossRank.push_back(
          f.positions[p].crossValid ? f.positions[p].cross.expectedRank : 0.0);
      input.selectedGoldMargin.push_back(
          s.positions[p].gold.expectedMinusTop1Margin);
      input.selectedSelfMargin.push_back(
          s.positions[p].self.expectedMinusTop1Margin);
    }
    if (input.selectedFirstError >= 0) {
      const std::size_t fe = static_cast<std::size_t>(input.selectedFirstError);
      input.selectedTieAtFirstErrorGold =
          s.positions[fe].gold.top1MinusTop2Margin < ma::kMarginTieTolerance;
      input.finalTieAtFirstErrorGold =
          f.positions[fe].gold.top1MinusTop2Margin < ma::kMarginTieTolerance;
    }
    const ma::AttributionResult result = ma::attribute(input);
    switch (result.attribution) {
      case ma::Attribution::PerfectSelected: ++summary.perfectSelected; break;
      case ma::Attribution::LocalRankingFailure: ++summary.local; break;
      case ma::Attribution::PrefixDriftAmplification: ++summary.drift; break;
      case ma::Attribution::PrefixDriftUncorroborated:
        ++summary.driftUncorroborated;
        break;
      case ma::Attribution::Mixed: ++summary.mixed; break;
      case ma::Attribution::NoClearAttribution: ++summary.noClear; break;
    }
    if (rows) {
      AttributionRow item;
      item.caseId = s.id;
      item.length = length;
      item.selectedFirstError =
          input.selectedFirstError < 0 ? -1 : input.selectedFirstError + 1;
      item.finalFirstError =
          input.finalFirstError < 0 ? -1 : input.finalFirstError + 1;
      item.localHit = result.localHit;
      item.driftHit = result.driftHit;
      item.nearGold = result.nearGold;
      item.deltaRankMean = result.deltaRankMean;
      item.deltaMarginMean = result.deltaMarginMean;
      item.corroborationComputed = result.corroborationComputed;
      item.rank1CrossSelectedOnFinalPrefix =
          result.rank1CrossSelectedOnFinal;
      item.rank1CrossFinalOnSelectedPrefix =
          result.rank1CrossFinalOnSelected;
      item.attribution = result.attribution;
      rows->push_back(std::move(item));
    }
  }
  return summary;
}

struct ScalarStats {
  std::uint64_t tokenTotal = 0;
  std::uint64_t sequenceTotal = 0;
  std::uint64_t top1Exact = 0;
  std::uint64_t top2Inclusion = 0;
  std::uint64_t top3Inclusion = 0;
  std::uint64_t rank1Unique = 0;
  double rankSum = 0.0;
  double probabilitySum = 0.0;
  double entropySum = 0.0;
  double expectedTop1MarginSum = 0.0;
  double top1Top2MarginSum = 0.0;
  std::uint64_t marginNonnegative = 0;
  std::uint64_t marginAbsoluteBelowLog2 = 0;
  std::uint64_t noErrorCases = 0;
  std::uint64_t firstErrorCases = 0;
  double firstErrorPositionSum = 0.0;
  std::vector<double> firstErrorPositions;
};

ScalarStats scalarStats(const CheckpointResult& checkpoint) {
  ScalarStats stats;
  for (const auto& item : checkpoint.cases) {
    ++stats.sequenceTotal;
    if (item.firstError.firstError < 0) {
      ++stats.noErrorCases;
    } else {
      ++stats.firstErrorCases;
      const double position1 =
          static_cast<double>(item.firstError.firstError + 1);
      stats.firstErrorPositionSum += position1;
      stats.firstErrorPositions.push_back(position1);
    }
    for (std::size_t p = 0; p < item.positions.size(); ++p) {
      const auto& record = item.positions[p];
      const bool correct = record.self.predicted == item.expected[p];
      ++stats.tokenTotal;
      if (correct) ++stats.top1Exact;
      if (record.self.expectedRank <= 1.0) ++stats.rank1Unique;
      if (record.self.expectedRank <= 2.0) ++stats.top2Inclusion;
      if (record.self.expectedRank <= 3.0) ++stats.top3Inclusion;
      stats.rankSum += record.self.expectedRank;
      stats.probabilitySum += record.self.expectedProbability;
      stats.entropySum += record.self.entropy;
      stats.expectedTop1MarginSum += record.self.expectedMinusTop1Margin;
      stats.top1Top2MarginSum += record.self.top1MinusTop2Margin;
      if (record.self.expectedMinusTop1Margin >= 0.0)
        ++stats.marginNonnegative;
      if (std::abs(record.self.expectedMinusTop1Margin) < ma::kLog2)
        ++stats.marginAbsoluteBelowLog2;
    }
  }
  return stats;
}

struct ConfigEvaluation {
  RunSpec spec{};
  tiny::Config config;
  dq::AutoregressiveSelectedRun selected;
  int selectedStep = 0;
  CheckpointResult selectedResult;
  CheckpointResult finalResult;
  PairStats pair;
  AttributionSummary attribution;
  std::vector<AttributionRow> attributionRows;
  ScalarStats selectedScalars;
  ScalarStats finalScalars;
};

struct PooledPair {
  double totalImprovement = 0.0;
  double gainBothCorrect = 0.0;
  double lossSelectedWrongFinalCorrect = 0.0;
  double percentGain = 0.0;
  bool percentStable = false;
  std::uint64_t easyTokenCount = 0;
  std::uint64_t criticalTokenCount = 0;
  double easyNllGain = 0.0;
  double criticalNllLoss = 0.0;
  std::uint64_t selectedExact = 0;
  std::uint64_t finalExact = 0;
  std::uint64_t tokenTotal = 0;
  std::uint64_t selectedSequenceExact = 0;
  std::uint64_t finalSequenceExact = 0;
  std::uint64_t sequenceTotal = 0;
  std::uint64_t goldSwfcCount = 0;
  std::uint64_t selfSwfcCount = 0;
};

PooledPair pooledPair(const std::vector<ConfigEvaluation>& evaluations,
                      std::size_t start, std::size_t end) {
  PooledPair result;
  for (std::size_t i = start; i < end; ++i) {
    const auto& eval = evaluations[i];
    result.totalImprovement += eval.pair.totalImprovement();
    result.gainBothCorrect += eval.pair.gainFromBothCorrect();
    result.lossSelectedWrongFinalCorrect +=
        eval.pair.lossFromSelectedWrongFinalCorrect();
    result.easyTokenCount += eval.pair.easyTokenCount;
    result.criticalTokenCount += eval.pair.criticalTokenCount;
    result.easyNllGain += eval.pair.easyNllGain;
    result.criticalNllLoss += eval.pair.criticalNllLoss;
    result.selectedExact += eval.pair.selectedExact;
    result.finalExact += eval.pair.finalExact;
    result.tokenTotal += eval.pair.tokenCount;
    result.selectedSequenceExact += eval.selectedScalars.noErrorCases;
    result.finalSequenceExact += eval.finalScalars.noErrorCases;
    result.sequenceTotal += eval.selectedScalars.sequenceTotal;
    result.selfSwfcCount += eval.pair.buckets[static_cast<std::size_t>(
                               ma::Bucket::SelectedWrongFinalCorrect)]
                               .tokenCount;
    result.goldSwfcCount +=
        eval.pair.goldBucketCounts[static_cast<std::size_t>(
            ma::Bucket::SelectedWrongFinalCorrect)];
  }
  result.percentGain = ma::percentGainShare(
      result.gainBothCorrect, result.totalImprovement,
      static_cast<double>(result.tokenTotal), &result.percentStable);
  return result;
}

// ---------------------------------------------------------------------------
// Pre-committed hypothesis thresholds (fixed before measurement)
// ---------------------------------------------------------------------------
// H1 (EASY_TOKEN_NLL_DOMINANCE): at least half of the total NLL improvement
// must come from tokens the selected model already predicted correctly (the
// BOTH_CORRECT bucket), on a stable per-token basis, while easy tokens are
// the clear majority of the pool.
// H2 (CRITICAL_TOKEN_MARGIN_LOSS): the final checkpoint recovers enough
// selected-wrong tokens (>= kSwfcMinimumCount) that their median expected
// margin under the selected checkpoint is negative (the selected model ranked
// the wrong token first on exactly the tokens the final model corrects).
// H3 (PREFIX_DRIFT_AMPLIFICATION): drift attributions outnumber local
// ranking failures and mixed calls by a margin that cannot be a single case.
inline constexpr double kEasyTokenFractionThreshold = 0.5;
inline constexpr double kEasyGainShareThreshold = 0.50;
inline constexpr std::uint64_t kSwfcMinimumCount = 3;
inline constexpr std::uint64_t kDriftMinimumCount = 2;

std::string fromBool(bool value) { return value ? "true" : "false"; }

int countCorrect(const CaseRecord& item) {
  int correct = 0;
  for (std::size_t p = 0; p < item.expected.size(); ++p)
    if (item.positions[p].self.predicted == item.expected[p]) ++correct;
  return correct;
}

bool sequenceExact(const CaseRecord& item) {
  return item.firstError.firstError < 0;
}

double meanNll(const CaseRecord& item) {
  double sum = 0.0;
  for (const auto& record : item.positions) sum += record.self.tokenNll;
  return item.positions.empty()
             ? 0.0
             : sum / static_cast<double>(item.positions.size());
}

// Cross-check against the established ar::Metrics path (same contexts, same
// tie-breaking).  A mismatch is a fail-closed consistency error, not a report.
void verifyArConsistency(const tiny::Config& config, const dq::Params& params,
                         const CheckpointResult& result,
                         const std::string& label) {
  const auto metrics = dq::autoregressiveEvaluation(
      config, params, ar::Partition::DEVELOPMENT);
  if (metrics.perCase.size() != result.cases.size())
    throw std::runtime_error("consistency case count mismatch: " + label);
  for (std::size_t i = 0; i < result.cases.size(); ++i) {
    const auto& item = result.cases[i];
    const auto& expectedMetrics = metrics.perCase[i];
    if (item.id != expectedMetrics.id)
      throw std::runtime_error("consistency case id mismatch: " + label);
    if (static_cast<std::uint32_t>(countCorrect(item)) !=
        expectedMetrics.tokenExact)
      throw std::runtime_error("consistency token exact mismatch: " + label);
    if (sequenceExact(item) != expectedMetrics.sequenceExact)
      throw std::runtime_error("consistency sequence exact mismatch: " + label);
    const std::int64_t expectedFirstError =
        expectedMetrics.firstErrorPosition < 0
            ? -1
            : static_cast<std::int64_t>(expectedMetrics.firstErrorPosition - 1);
    if (static_cast<std::int64_t>(item.firstError.firstError) !=
        expectedFirstError)
      throw std::runtime_error("consistency first error mismatch: " + label);
    const double nll = meanNll(item);
    if (std::abs(nll - expectedMetrics.autoregressiveNll) > 1.0e-9)
      throw std::runtime_error("consistency NLL mismatch: " + label);
  }
}

void writePrivateTokens(const fs::path& root, const ConfigEvaluation& eval) {
  std::ofstream output(
      root / ("margin-tokens-" + std::string(eval.spec.publicId) + ".csv"));
  row(output, {"configuration_id", "case_id", "position", "length", "truth",
               "selected_predicted", "final_predicted", "selected_exact",
               "final_exact", "bucket", "selected_self_rank",
               "final_self_rank", "selected_gold_rank", "final_gold_rank",
               "selected_gold_margin", "selected_self_margin",
               "selected_top1_top2_margin", "final_top1_top2_margin",
               "selected_entropy", "final_entropy", "selected_nll",
               "final_nll", "selected_first_error", "final_first_error",
               "prefix_matches_reference", "selected_cross_rank_final_prefix",
               "final_cross_rank_selected_prefix"});
  for (std::size_t c = 0; c < eval.selectedResult.cases.size(); ++c) {
    const auto& s = eval.selectedResult.cases[c];
    const auto& f = eval.finalResult.cases[c];
    for (std::size_t p = 0; p < s.positions.size(); ++p) {
      const std::uint32_t truth = s.expected[p];
      const bool sCorrect = s.positions[p].self.predicted == truth;
      const bool fCorrect = f.positions[p].self.predicted == truth;
      row(output, {eval.spec.publicId, s.id, text(p + 1),
                   text(s.positions.size()), text(truth),
                   text(s.positions[p].self.predicted),
                   text(f.positions[p].self.predicted), fromBool(sCorrect),
                   fromBool(fCorrect),
                   ma::bucketName(ma::classifyBucket(sCorrect, fCorrect)),
                   number(s.positions[p].self.expectedRank),
                   number(f.positions[p].self.expectedRank),
                   number(s.positions[p].gold.expectedRank),
                   number(f.positions[p].gold.expectedRank),
                   number(s.positions[p].gold.expectedMinusTop1Margin),
                   number(s.positions[p].self.expectedMinusTop1Margin),
                   number(s.positions[p].self.top1MinusTop2Margin),
                   number(f.positions[p].self.top1MinusTop2Margin),
                   number(s.positions[p].self.entropy),
                   number(f.positions[p].self.entropy),
                   number(s.positions[p].self.tokenNll),
                   number(f.positions[p].self.tokenNll),
                   fromBool(s.firstError.firstError == static_cast<int>(p)),
                   fromBool(f.firstError.firstError == static_cast<int>(p)),
                   fromBool(s.positions[p].prefixMatchesReference),
                   s.positions[p].crossValid
                       ? number(s.positions[p].cross.expectedRank)
                       : "NOT_AVAILABLE",
                   f.positions[p].crossValid
                       ? number(f.positions[p].cross.expectedRank)
                       : "NOT_AVAILABLE"});
    }
  }
}

const char* dominantAttribution(const AttributionSummary& summary) {
  std::array<std::uint64_t, 6> counts{{
      summary.drift, summary.driftUncorroborated, summary.local, summary.mixed,
      summary.perfectSelected, summary.noClear}};
  std::size_t best = 0;
  for (std::size_t i = 1; i < counts.size(); ++i)
    if (counts[i] > counts[best]) best = i;
  switch (best) {
    case 0: return "PREFIX_DRIFT_AMPLIFICATION";
    case 1: return "PREFIX_DRIFT_AMPLIFICATION_UNCORROBORATED";
    case 2: return "LOCAL_LOGIT_RANKING_FAILURE";
    case 3: return "MIXED";
    case 4: return "PERFECT_SELECTED";
    default: return "NO_CLEAR_ATTRIBUTION";
  }
}

AttributionSummary mergeAttribution(const std::vector<AttributionSummary>& all) {
  AttributionSummary merged;
  for (const auto& item : all) {
    merged.perfectSelected += item.perfectSelected;
    merged.local += item.local;
    merged.drift += item.drift;
    merged.driftUncorroborated += item.driftUncorroborated;
    merged.mixed += item.mixed;
    merged.noClear += item.noClear;
  }
  return merged;
}

struct Hypothesis {
  bool h1 = false, h2 = false, h3 = false;
};

Hypothesis decideHypotheses(const PooledPair& pooled, double swfcMedianMargin,
                            const AttributionSummary& attribution) {
  Hypothesis result;
  const double easyFraction =
      pooled.tokenTotal == 0
          ? 0.0
          : static_cast<double>(pooled.easyTokenCount) /
                static_cast<double>(pooled.tokenTotal);
  result.h1 = pooled.percentStable &&
              pooled.gainBothCorrect >=
                  kEasyGainShareThreshold * pooled.totalImprovement &&
              easyFraction >= kEasyTokenFractionThreshold;
  result.h2 = pooled.selfSwfcCount >= kSwfcMinimumCount &&
              swfcMedianMargin < 0.0;
  const std::uint64_t driftCases = attribution.drift + attribution.driftUncorroborated;
  result.h3 = driftCases >= kDriftMinimumCount && driftCases > attribution.local &&
              driftCases > attribution.mixed;
  return result;
}

const char* conclude(const Hypothesis& hypothesis) {
  const std::uint64_t supported =
      (hypothesis.h1 ? 1u : 0u) + (hypothesis.h2 ? 1u : 0u) +
      (hypothesis.h3 ? 1u : 0u);
  if (supported == 0) return "INCONCLUSIVE";
  if (supported > 1) return "MIXED";
  if (hypothesis.h1) return "EASY_TOKEN_NLL_DOMINANCE";
  if (hypothesis.h2) return "CRITICAL_TOKEN_MARGIN_LOSS";
  return "PREFIX_DRIFT_AMPLIFICATION";
}

// ---------------------------------------------------------------------------
// Public-safe aggregate CSV writers
// ---------------------------------------------------------------------------
void writeConfiguration(const fs::path& root,
                        const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "configuration.csv");
  row(output, {"source", "configuration_id", "depth", "seed", "selected_step",
               "pinned_selected_step", "selected_step_matches_pinned",
               "best_token_exact_step", "best_token_exact_count",
               "best_sequence_exact_step", "best_sequence_exact_count",
               "validation_selected_ar_nll", "selected_nll", "final_nll",
               "nll_delta", "selected_token_exact", "final_token_exact",
               "token_total", "selected_sequence_exact",
               "final_sequence_exact", "sequence_total", "easy_token_count",
               "critical_token_count", "easy_nll_gain", "critical_nll_loss",
               "swfc_count", "swfc_median_margin", "swfc_median_rank",
               "swfc_median_entropy_selected", "swfc_median_entropy_final",
               "attribution_dominant"});
  for (const auto& eval : evaluations) {
    const auto& swfc = eval.pair.buckets[static_cast<std::size_t>(
        ma::Bucket::SelectedWrongFinalCorrect)];
    const double swfcMedian =
        eval.pair.swfcMargin.empty()
            ? 0.0
            : ma::medianLowerMiddle(eval.pair.swfcMargin);
    const double swfcMedianRank =
        eval.pair.swfcRank.empty() ? 0.0
                                   : ma::medianLowerMiddle(eval.pair.swfcRank);
    const double swfcMedianEntropySelected =
        eval.pair.swfcEntropySelected.empty()
            ? 0.0
            : ma::medianLowerMiddle(eval.pair.swfcEntropySelected);
    const double swfcMedianEntropyFinal =
        eval.pair.swfcEntropyFinal.empty()
            ? 0.0
            : ma::medianLowerMiddle(eval.pair.swfcEntropyFinal);
    row(output, {"CPU_REFERENCE_REGENERATION", eval.spec.publicId,
                 text(eval.spec.layers), text(eval.spec.seed),
                 text(eval.selectedStep), text(eval.spec.pinnedSelectedStep),
                 fromBool(eval.selectedStep == eval.spec.pinnedSelectedStep),
                 text(eval.selected.bestTokenExactStep),
                 text(eval.selected.bestTokenExactCount),
                 text(eval.selected.bestSequenceExactStep),
                 text(eval.selected.bestSequenceExactCount),
                 number(eval.selected.selectedValidation.autoregressiveNll),
                 number(eval.pair.selectedNll()), number(eval.pair.finalNll()),
                 number(eval.pair.totalImprovement()),
                 text(eval.pair.selectedExact), text(eval.pair.finalExact),
                 text(eval.pair.tokenCount),
                 text(eval.selectedScalars.noErrorCases),
                 text(eval.finalScalars.noErrorCases),
                 text(eval.selectedScalars.sequenceTotal),
                 text(eval.pair.easyTokenCount),
                 text(eval.pair.criticalTokenCount),
                 number(eval.pair.easyNllGain),
                 number(eval.pair.criticalNllLoss), text(swfc.tokenCount),
                 number(swfcMedian), number(swfcMedianRank),
                 number(swfcMedianEntropySelected),
                 number(swfcMedianEntropyFinal),
                 dominantAttribution(eval.attribution)});
  }
}

void writeCheckpointComparison(const fs::path& root,
                               const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "checkpoint-comparison.csv");
  row(output, {"configuration_id", "role", "step", "ar_nll", "token_exact",
               "token_total", "sequence_exact", "sequence_total",
               "mean_first_error_position", "no_error_cases",
               "first_error_cases", "mean_rank", "mean_probability",
               "mean_entropy", "mean_expected_margin",
               "mean_top1_top2_margin", "rank1_unique", "top2_inclusion",
               "top3_inclusion", "margin_nonnegative_fraction",
               "margin_abs_below_log2_fraction"});
  for (const auto& eval : evaluations) {
    const auto& scalarPairs = std::array<std::pair<const char*, const ScalarStats*>,
                                         2>{{{"SELECTED", &eval.selectedScalars},
                                             {"FINAL_STEP", &eval.finalScalars}}};
    for (const auto& entry : scalarPairs) {
      const auto& stats = *entry.second;
      const bool selected = entry.first[0] == 'S';
      const int step = selected ? eval.selectedStep : kSteps;
      const double fractionNonnegative =
          stats.tokenTotal == 0
              ? 0.0
              : static_cast<double>(stats.marginNonnegative) /
                    static_cast<double>(stats.tokenTotal);
      const double fractionBelowLog2 =
          stats.tokenTotal == 0
              ? 0.0
              : static_cast<double>(stats.marginAbsoluteBelowLog2) /
                    static_cast<double>(stats.tokenTotal);
      const double meanFirstError =
          stats.firstErrorCases == 0
              ? 0.0
              : stats.firstErrorPositionSum /
                    static_cast<double>(stats.firstErrorCases);
      row(output, {eval.spec.publicId, entry.first, text(step),
                   number(selected ? eval.pair.selectedNll()
                                   : eval.pair.finalNll()),
                   text(stats.top1Exact), text(stats.tokenTotal),
                   text(stats.noErrorCases), text(stats.sequenceTotal),
                   number(meanFirstError), text(stats.noErrorCases),
                   text(stats.firstErrorCases),
                   number(stats.tokenTotal == 0 ? 0.0 : stats.rankSum /
                     static_cast<double>(stats.tokenTotal)),
                   number(stats.tokenTotal == 0
                              ? 0.0
                              : stats.probabilitySum /
                                    static_cast<double>(stats.tokenTotal)),
                   number(stats.tokenTotal == 0
                              ? 0.0
                              : stats.entropySum /
                                    static_cast<double>(stats.tokenTotal)),
                   number(stats.tokenTotal == 0
                              ? 0.0
                              : stats.expectedTop1MarginSum /
                                    static_cast<double>(stats.tokenTotal)),
                   number(stats.tokenTotal == 0
                              ? 0.0
                              : stats.top1Top2MarginSum /
                                    static_cast<double>(stats.tokenTotal)),
                   text(stats.rank1Unique), text(stats.top2Inclusion),
                   text(stats.top3Inclusion), number(fractionNonnegative),
                   number(fractionBelowLog2)});
    }
  }
}

void writeTokenBuckets(const fs::path& root,
                       const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "token-buckets.csv");
  row(output, {"configuration_id", "bucket", "token_count", "percent_tokens",
               "selected_nll_mean", "final_nll_mean", "nll_contribution",
               "mean_rank_selected", "mean_rank_final", "mean_margin_selected",
               "mean_margin_final", "mean_entropy_selected",
               "mean_entropy_final", "first_error_positions_selected",
               "first_error_positions_final", "selected_exact_count",
               "final_exact_count"});
  for (const auto& eval : evaluations) {
    for (std::size_t b = 0; b < 4; ++b) {
      const ma::Bucket bucket = static_cast<ma::Bucket>(b);
      const auto& stats = eval.pair.buckets[b];
      std::uint64_t selectedExact = 0, finalExact = 0;
      switch (bucket) {
        case ma::Bucket::BothCorrect:
          selectedExact = finalExact = stats.tokenCount;
          break;
        case ma::Bucket::SelectedCorrectFinalWrong:
          selectedExact = stats.tokenCount;
          break;
        case ma::Bucket::SelectedWrongFinalCorrect:
          finalExact = stats.tokenCount;
          break;
        case ma::Bucket::BothWrong:
          break;
      }
      row(output, {eval.spec.publicId, ma::bucketName(bucket),
                   text(stats.tokenCount),
                   number(eval.pair.tokenCount == 0
                              ? 0.0
                              : 100.0 * static_cast<double>(stats.tokenCount) /
                                    static_cast<double>(eval.pair.tokenCount)),
                   number(stats.tokenCount == 0
                              ? 0.0
                              : stats.nllSumSelected /
                                    static_cast<double>(stats.tokenCount)),
                   number(stats.tokenCount == 0
                              ? 0.0
                              : stats.nllSumFinal /
                                    static_cast<double>(stats.tokenCount)),
                   number(stats.nllContribution()),
                   number(stats.meanRankSelected()),
                   number(stats.meanRankFinal()),
                   number(stats.meanMarginSelected()),
                   number(stats.meanMarginFinal()),
                   number(stats.meanEntropySelected()),
                   number(stats.meanEntropyFinal()),
                   text(stats.firstErrorPositionsSelected),
                   text(stats.firstErrorPositionsFinal),
                   text(selectedExact), text(finalExact)});
    }
  }
}

void writeFirstErrorSummary(const fs::path& root,
                            const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "first-error-summary.csv");
  row(output, {"configuration_id", "selected_no_error",
               "selected_late_single_error", "selected_error_with_recovery",
               "selected_early_irreversible_divergence",
               "selected_multiple_local_errors", "final_no_error",
               "final_late_single_error", "final_error_with_recovery",
               "final_early_irreversible_divergence",
               "final_multiple_local_errors",
               "selected_mean_first_error_position",
               "final_mean_first_error_position",
               "selected_median_first_error_position",
               "final_median_first_error_position",
               "selected_first_error_cases", "final_first_error_cases"});
  for (const auto& eval : evaluations) {
    std::array<std::uint64_t, 5> selectedCounts{}, finalCounts{};
    std::vector<double> selectedPositions, finalPositions;
    for (std::size_t c = 0; c < eval.selectedResult.cases.size(); ++c) {
      const auto& s = eval.selectedResult.cases[c];
      const auto& f = eval.finalResult.cases[c];
      ++selectedCounts[static_cast<std::size_t>(s.caseClass)];
      ++finalCounts[static_cast<std::size_t>(f.caseClass)];
      if (s.firstError.firstError >= 0)
        selectedPositions.push_back(s.firstError.firstError + 1.0);
      if (f.firstError.firstError >= 0)
        finalPositions.push_back(f.firstError.firstError + 1.0);
    }
    const auto sum = [](const std::array<std::uint64_t, 5>& values) {
      std::uint64_t total = 0;
      for (const auto value : values) total += value;
      return total;
    };
    double selectedMeanPosition = 0.0, finalMeanPosition = 0.0;
    if (!selectedPositions.empty())
      selectedMeanPosition = std::accumulate(
          selectedPositions.begin(), selectedPositions.end(), 0.0) /
                             static_cast<double>(selectedPositions.size());
    if (!finalPositions.empty())
      finalMeanPosition = std::accumulate(
          finalPositions.begin(), finalPositions.end(), 0.0) /
                          static_cast<double>(finalPositions.size());
    row(output, {eval.spec.publicId, text(selectedCounts[0]),
                 text(selectedCounts[1]), text(selectedCounts[2]),
                 text(selectedCounts[3]), text(selectedCounts[4]),
                 text(finalCounts[0]), text(finalCounts[1]),
                 text(finalCounts[2]), text(finalCounts[3]),
                 text(finalCounts[4]), number(selectedMeanPosition),
                 number(finalMeanPosition),
                 number(ma::medianLowerMiddle(selectedPositions)),
                 number(ma::medianLowerMiddle(finalPositions)),
                 text(sum(selectedCounts) - selectedCounts[0]),
                 text(sum(finalCounts) - finalCounts[0])});
  }
}

void writeMarginRankSummary(const fs::path& root,
                            const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "margin-rank-summary.csv");
  row(output, {"configuration_id", "mean_rank_selected", "mean_rank_final",
               "rank1_unique_selected", "rank1_unique_final",
               "rank1_unique_total", "top2_inclusion_selected",
               "top2_inclusion_final", "top3_inclusion_selected",
               "top3_inclusion_final", "mean_expected_probability_selected",
               "mean_expected_probability_final", "mean_entropy_selected",
               "mean_entropy_final", "mean_expected_minus_top1_margin_selected",
               "mean_expected_minus_top1_margin_final",
               "mean_top1_minus_top2_margin_selected",
               "mean_top1_minus_top2_margin_final",
               "fraction_margin_nonnegative_selected",
               "fraction_margin_nonnegative_final",
               "fraction_margin_abs_below_log2_selected",
               "fraction_margin_abs_below_log2_final", "swfc_token_count",
               "swfc_median_margin", "swfc_median_rank",
               "swfc_median_entropy_selected", "swfc_median_entropy_final"});
  for (const auto& eval : evaluations) {
    const auto& s = eval.selectedScalars;
    const auto& f = eval.finalScalars;
    const auto& swfc = eval.pair.buckets[static_cast<std::size_t>(
        ma::Bucket::SelectedWrongFinalCorrect)];
    row(output, {eval.spec.publicId,
                 number(s.tokenTotal == 0 ? 0.0 : s.rankSum /
                   static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0 ? 0.0 : f.rankSum /
                   static_cast<double>(f.tokenTotal)),
                 text(s.rank1Unique), text(f.rank1Unique),
                 text(s.tokenTotal),
                 text(s.top2Inclusion), text(f.top2Inclusion),
                 text(s.top3Inclusion), text(f.top3Inclusion),
                 number(s.tokenTotal == 0
                            ? 0.0
                            : s.probabilitySum /
                                  static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0
                            ? 0.0
                            : f.probabilitySum /
                                  static_cast<double>(f.tokenTotal)),
                 number(s.tokenTotal == 0 ? 0.0 : s.entropySum /
                   static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0 ? 0.0 : f.entropySum /
                   static_cast<double>(f.tokenTotal)),
                 number(s.tokenTotal == 0
                            ? 0.0
                            : s.expectedTop1MarginSum /
                                  static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0
                            ? 0.0
                            : f.expectedTop1MarginSum /
                                  static_cast<double>(f.tokenTotal)),
                 number(s.tokenTotal == 0
                            ? 0.0
                            : s.top1Top2MarginSum /
                                  static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0
                            ? 0.0
                            : f.top1Top2MarginSum /
                                  static_cast<double>(f.tokenTotal)),
                 number(s.tokenTotal == 0
                            ? 0.0
                            : static_cast<double>(s.marginNonnegative) /
                                  static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0
                            ? 0.0
                            : static_cast<double>(f.marginNonnegative) /
                                  static_cast<double>(f.tokenTotal)),
                 number(s.tokenTotal == 0
                            ? 0.0
                            : static_cast<double>(s.marginAbsoluteBelowLog2) /
                                  static_cast<double>(s.tokenTotal)),
                 number(f.tokenTotal == 0
                            ? 0.0
                            : static_cast<double>(f.marginAbsoluteBelowLog2) /
                                  static_cast<double>(f.tokenTotal)),
                 text(swfc.tokenCount),
                 number(eval.pair.swfcMargin.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(eval.pair.swfcMargin)),
                 number(eval.pair.swfcRank.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(eval.pair.swfcRank)),
                 number(eval.pair.swfcEntropySelected.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(
                                  eval.pair.swfcEntropySelected)),
                 number(eval.pair.swfcEntropyFinal.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(eval.pair.swfcEntropyFinal))});
  }
}

void writeCommonPrefixAttribution(const fs::path& root,
                                  const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "common-prefix-attribution.csv");
  row(output, {"configuration_id", "case_id", "length", "selected_first_error",
               "final_first_error", "local_hit", "drift_hit", "near_gold",
               "delta_rank_mean", "delta_margin_mean",
               "rank1_cross_selected_on_final_prefix",
               "rank1_cross_final_on_selected_prefix", "corroboration_computed",
               "attribution"});
  for (const auto& eval : evaluations) {
    for (const auto& item : eval.attributionRows) {
      row(output, {eval.spec.publicId, item.caseId, text(item.length),
                   text(item.selectedFirstError), text(item.finalFirstError),
                   fromBool(item.localHit), fromBool(item.driftHit),
                   fromBool(item.nearGold), number(item.deltaRankMean),
                   number(item.deltaMarginMean),
                   number(item.rank1CrossSelectedOnFinalPrefix),
                   number(item.rank1CrossFinalOnSelectedPrefix),
                   fromBool(item.corroborationComputed),
                   ma::attributionName(item.attribution)});
    }
  }
}

void writeSeedComparison(const fs::path& root,
                         const std::vector<ConfigEvaluation>& evaluations,
                         const Hypothesis& hypothesis) {
  std::ofstream output(root / "seed-comparison.csv");
  row(output, {"seed", "selected_step", "selected_nll", "final_nll",
               "nll_delta", "token_exact_selected", "token_exact_final",
               "sequence_exact_selected", "sequence_exact_final",
               "swfc_token_count", "swfc_median_margin",
               "attribution_dominant", "h1_supported", "h2_supported",
               "h3_supported"});
  for (const auto& eval : evaluations) {
    if (eval.spec.layers != 19) continue;
    const auto& swfc = eval.pair.buckets[static_cast<std::size_t>(
        ma::Bucket::SelectedWrongFinalCorrect)];
    row(output, {text(eval.spec.seed), text(eval.selectedStep),
                 number(eval.pair.selectedNll()), number(eval.pair.finalNll()),
                 number(eval.pair.totalImprovement()),
                 text(eval.pair.selectedExact), text(eval.pair.finalExact),
                 text(eval.selectedScalars.noErrorCases),
                 text(eval.finalScalars.noErrorCases), text(swfc.tokenCount),
                 number(eval.pair.swfcMargin.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(eval.pair.swfcMargin)),
                 dominantAttribution(eval.attribution),
                 fromBool(hypothesis.h1), fromBool(hypothesis.h2),
                 fromBool(hypothesis.h3)});
  }
}

void writeDepthControl(const fs::path& root,
                       const std::vector<ConfigEvaluation>& evaluations) {
  std::ofstream output(root / "depth-control.csv");
  row(output, {"configuration_id", "depth", "seed", "selected_step",
               "selected_nll", "final_nll", "nll_delta",
               "token_exact_selected", "token_exact_final",
               "sequence_exact_selected", "sequence_exact_final",
               "swfc_count", "swfc_median_margin", "attribution_dominant",
               "depth_control_observation"});
  const ConfigEvaluation* l18 = nullptr;
  const ConfigEvaluation* l19 = nullptr;
  for (const auto& eval : evaluations) {
    if (eval.spec.layers == 18 && eval.spec.seed == 2) l18 = &eval;
    if (eval.spec.layers == 19 && eval.spec.seed == 2) l19 = &eval;
  }
  if (!l18 || !l19) throw std::runtime_error("depth control pair missing");
  const auto& l18Swfc = l18->pair.buckets[static_cast<std::size_t>(
      ma::Bucket::SelectedWrongFinalCorrect)];
  const auto& l19Swfc = l19->pair.buckets[static_cast<std::size_t>(
      ma::Bucket::SelectedWrongFinalCorrect)];
  const char* observation;
  if (l18->finalScalars.top1Exact == l19->finalScalars.top1Exact &&
      l18->finalScalars.noErrorCases == l19->finalScalars.noErrorCases)
    observation = "DEPTH_CONTROL_EXACTLY_EQUIVALENT_FINAL_EXACT";
  else if (l18->finalScalars.top1Exact >= l19->finalScalars.top1Exact &&
           l18->finalScalars.noErrorCases >= l19->finalScalars.noErrorCases)
    observation = "CONTROL_FINAL_EXACT_AT_LEAST_L19";
  else
    observation = "L19_FINAL_EXACT_EXCEEDS_CONTROL";
  for (const ConfigEvaluation* eval : {l18, l19}) {
    const auto& swfc = eval->pair.buckets[static_cast<std::size_t>(
        ma::Bucket::SelectedWrongFinalCorrect)];
    row(output, {eval->spec.publicId, text(eval->spec.layers),
                 text(eval->spec.seed), text(eval->selectedStep),
                 number(eval->pair.selectedNll()),
                 number(eval->pair.finalNll()),
                 number(eval->pair.totalImprovement()),
                 text(eval->pair.selectedExact), text(eval->pair.finalExact),
                 text(eval->selectedScalars.noErrorCases),
                 text(eval->finalScalars.noErrorCases), text(swfc.tokenCount),
                 number(eval->pair.swfcMargin.empty()
                            ? 0.0
                            : ma::medianLowerMiddle(eval->pair.swfcMargin)),
                 dominantAttribution(eval->attribution), observation});
  }
  (void)l18Swfc;
  (void)l19Swfc;
}

void writeHypothesisDecision(const fs::path& root, const PooledPair& pooled,
                             double swfcMedianMargin,
                             const AttributionSummary& attribution,
                             const Hypothesis& hypothesis) {
  std::ofstream output(root / "hypothesis-decision.csv");
  row(output, {"hypothesis", "supported", "evidence_metric", "evidence_value",
               "threshold", "conclusion"});
  const double easyFraction =
      pooled.tokenTotal == 0
          ? 0.0
          : static_cast<double>(pooled.easyTokenCount) /
                static_cast<double>(pooled.tokenTotal);
  row(output, {"H1_EASY_TOKEN_NLL_DOMINANCE", fromBool(hypothesis.h1),
               "PCT_GAIN_FROM_BOTH_CORRECT", number(pooled.percentGain),
               number(100.0 * kEasyGainShareThreshold), "NOT_AVAILABLE"});
  row(output, {"H1_EASY_TOKEN_NLL_DOMINANCE", fromBool(hypothesis.h1),
               "EASY_TOKEN_FRACTION", number(easyFraction),
               number(kEasyTokenFractionThreshold), "NOT_AVAILABLE"});
  row(output, {"H2_CRITICAL_TOKEN_MARGIN_LOSS", fromBool(hypothesis.h2),
               "SWFC_TOKEN_COUNT", number(static_cast<double>(pooled.selfSwfcCount)),
               number(static_cast<double>(kSwfcMinimumCount)), "NOT_AVAILABLE"});
  row(output, {"H2_CRITICAL_TOKEN_MARGIN_LOSS", fromBool(hypothesis.h2),
               "SWFC_MEDIAN_MARGIN", number(swfcMedianMargin), "0",
               "NOT_AVAILABLE"});
  row(output, {"H3_PREFIX_DRIFT_AMPLIFICATION", fromBool(hypothesis.h3),
               "PREFIX_DRIFT_CASE_COUNT",
               number(static_cast<double>(attribution.drift +
                                         attribution.driftUncorroborated)),
               number(static_cast<double>(kDriftMinimumCount)), "NOT_AVAILABLE"});
  row(output, {"CONCLUSION", "true", "DOMINANT_HYPOTHESIS", "NOT_AVAILABLE",
               "NOT_AVAILABLE", conclude(hypothesis)});
}

void writeNextObjectives(const fs::path& root) {
  std::ofstream output(root / "next-objective-candidates.csv");
  row(output, {"candidate_objective", "addresses", "proposed_fix", "status"});
  row(output, {"FIRST_ERROR_MARGIN_OBJECTIVE",
               "exact degrades when the gold token sits just below the argmax "
               "at the first autoregressive error",
               "auxiliary training loss that pushes the gold-token expected "
               "margin positive at the first free-running error",
               "NOT_RUN_CANDIDATE"});
  row(output, {"MINIMUM_CORRECT_TOKEN_MARGIN",
               "already-correct tokens absorb most of the NLL improvement "
               "while the decisive tokens lose margin",
               "loss term on a margin floor for tokens that must stay exact",
               "NOT_RUN_CANDIDATE"});
  row(output, {"PAIRWISE_CORRECT_VS_TOP1_RANKING",
               "the selected checkpoint ranks a wrong token first on tokens "
               "the final checkpoint later corrects",
               "ranking loss preferring the correct token over the top-1 "
               "wrong token whenever they differ",
               "NOT_RUN_CANDIDATE"});
  row(output, {"FREE_RUNNING_ROLLOUT_NLL",
               "diverged-prefix attribution where the context, not the local "
               "ranking, is the failure",
               "train on own predictions so prefix drift is inside the "
               "optimization loop",
               "NOT_RUN_CANDIDATE"});
}

int run(const fs::path& outputRoot) {
  fs::create_directories(outputRoot);
  const auto developmentCases = ar::cases(ar::Partition::DEVELOPMENT, kTokens);
  std::vector<ConfigEvaluation> evaluations;
  evaluations.reserve(kRuns.size());
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    ConfigEvaluation eval;
    eval.spec = spec;
    eval.config = config;
    eval.selected = dq::runAutoregressiveSelectedCpu(
        config, spec.seed, kSteps,
        dq::AutoregressiveSelectionMode::BEST_AR_VALIDATION_V1);
    if (eval.selected.selectedStep != spec.pinnedSelectedStep)
      throw std::runtime_error(std::string("SELECTED_STEP_REGRESSION: ") +
                               spec.publicId);
    eval.selectedStep = eval.selected.selectedStep;
    const int finalStep = kSteps;
    eval.selectedResult = evaluateCheckpoint(
        config, eval.selected.selectedParameters, eval.selectedStep, "SELECTED",
        developmentCases);
    eval.finalResult = evaluateCheckpoint(
        config, eval.selected.training.finalParameters, finalStep,
        "FINAL_STEP", developmentCases);
    evaluateCross(config, eval.selected.selectedParameters,
                  eval.selected.training.finalParameters, &eval.selectedResult,
                  &eval.finalResult);
    verifyArConsistency(config, eval.selected.selectedParameters,
                        eval.selectedResult, std::string(spec.publicId) + ":selected");
    verifyArConsistency(config, eval.selected.training.finalParameters,
                        eval.finalResult, std::string(spec.publicId) + ":final");
    eval.pair = pairStats(eval.selectedResult, eval.finalResult);
    eval.attribution =
        attributionSummary(eval.selectedResult, eval.finalResult,
                           &eval.attributionRows);
    eval.selectedScalars = scalarStats(eval.selectedResult);
    eval.finalScalars = scalarStats(eval.finalResult);
    if (eval.pair.tokenCount != 144 || eval.pair.tokenCount == 0)
      throw std::runtime_error("unexpected token count for " +
                               std::string(spec.publicId));
    if (eval.pair.selectedExact != eval.selectedScalars.top1Exact ||
        eval.pair.finalExact != eval.finalScalars.top1Exact)
      throw std::runtime_error("bucket/scalar exact mismatch for " +
                               std::string(spec.publicId));
    if (eval.attributionRows.size() != 24)
      throw std::runtime_error("attribution row count mismatch for " +
                               std::string(spec.publicId));
    writePrivateTokens(outputRoot, eval);
    evaluations.push_back(std::move(eval));
  }

  const PooledPair pooled = pooledPair(evaluations, 0, 3);
  const AttributionSummary pooledAttribution =
      mergeAttribution({evaluations[0].attribution, evaluations[1].attribution,
                        evaluations[2].attribution});
  const std::vector<double> pooledSwfcMargins = [&] {
    std::vector<double> margins;
    for (std::size_t i = 0; i < 3; ++i)
      margins.insert(margins.end(), evaluations[i].pair.swfcMargin.begin(),
                     evaluations[i].pair.swfcMargin.end());
    return margins;
  }();
  const double swfcMedianMargin = pooledSwfcMargins.empty()
                                      ? 0.0
                                      : ma::medianLowerMiddle(pooledSwfcMargins);
  const Hypothesis hypothesis =
      decideHypotheses(pooled, swfcMedianMargin, pooledAttribution);

  writeConfiguration(outputRoot, evaluations);
  writeCheckpointComparison(outputRoot, evaluations);
  writeTokenBuckets(outputRoot, evaluations);
  writeFirstErrorSummary(outputRoot, evaluations);
  writeMarginRankSummary(outputRoot, evaluations);
  writeCommonPrefixAttribution(outputRoot, evaluations);
  writeSeedComparison(outputRoot, evaluations, hypothesis);
  writeDepthControl(outputRoot, evaluations);
  writeHypothesisDecision(outputRoot, pooled, swfcMedianMargin,
                          pooledAttribution, hypothesis);
  writeNextObjectives(outputRoot);
  return 0;
}

void selfTest() {
  std::string error;
  assert(ar::validatePartitions(8, &error));
  assert(ar::hashMatchesPinned(ar::Partition::DEVELOPMENT));
  assert(ar::cases(ar::Partition::DEVELOPMENT).size() == 24);
  assert(ar::targetTransitionOccurrenceCount(ar::Partition::DEVELOPMENT) == 144);

  const std::vector<double> logits{0.0, 1.0, 2.0, 0.5};
  const std::vector<double> probabilities{0.2, 0.3, 0.4, 0.1};
  assert(ma::argmaxFirst(logits) == 2);
  assert(ma::expectedRank(logits, 0) == 4.0);
  assert(ma::expectedRank(logits, 2) == 1.0);
  assert(ma::expectedRank(logits, 1) == 2.0);
  assert(std::abs(ma::expectedMinusTop1Margin(logits, 2) - 1.0) < 1e-12);
  assert(std::abs(ma::expectedMinusTop1Margin(logits, 0) + 2.0) < 1e-12);
  assert(std::abs(ma::top1MinusTop2Margin(logits) - 1.0) < 1e-12);
  const auto top = ma::topTwoProbabilities(probabilities);
  assert(std::abs(top.first - 0.4) < 1e-12 && std::abs(top.second - 0.3) < 1e-12);

  std::vector<double> tie{1.0, 1.0, 0.0};
  assert(ma::argmaxFirst(tie) == 0);
  assert(ma::expectedRank(tie, 0) == 1.5);
  assert(ma::expectedRank(tie, 2) == 3.0);

  const std::vector<double> uniform(32, 1.0 / 32.0);
  assert(std::abs(ma::entropyOf(uniform) - std::log(32.0)) < 1e-9);

  const auto score = ma::scoreFromLogits(logits, probabilities, 2);
  assert(score.valid && score.predicted == 2 && score.expectedRank == 1.0);
  assert(std::abs(score.tokenNll + std::log(0.4)) < 1e-12);

  assert(ma::classifyBucket(true, true) == ma::Bucket::BothCorrect);
  assert(ma::classifyBucket(true, false) == ma::Bucket::SelectedCorrectFinalWrong);
  assert(ma::classifyBucket(false, true) == ma::Bucket::SelectedWrongFinalCorrect);
  assert(ma::classifyBucket(false, false) == ma::Bucket::BothWrong);

  const std::vector<std::uint32_t> predicted{1, 2, 2, 3, 4, 4};
  const std::vector<std::uint32_t> targets{1, 2, 3, 3, 4, 5};
  const auto firstError = ma::firstErrorInfo(predicted, targets);
  assert(firstError.firstError == 2 && firstError.wrongCount == 2);
  assert(firstError.postErrorExact == 2 && firstError.recoveredK2);
  assert(ma::classifyCase(firstError, 6) == ma::CaseClass::ErrorWithRecovery);
  const auto noError = ma::firstErrorInfo(targets, targets);
  assert(noError.firstError == -1 && noError.wrongCount == 0);
  assert(ma::classifyCase(noError, 6) == ma::CaseClass::NoError);
  const auto late = ma::firstErrorInfo({1, 2, 3, 3, 4, 9}, targets);
  assert(ma::classifyCase(late, 6) == ma::CaseClass::LateSingleError);
  const auto early = ma::firstErrorInfo({1, 9, 8, 9, 4, 9}, targets);
  assert(ma::classifyCase(early, 6) == ma::CaseClass::EarlyIrreversibleDivergence);
  const auto multiple = ma::firstErrorInfo({1, 8, 9, 3, 9, 5}, targets);
  assert(ma::classifyCase(multiple, 6) == ma::CaseClass::MultipleLocalErrors);

  ma::AttributionInput input;
  input.selectedFirstError = 1;
  input.finalFirstError = 1;
  const std::vector<double> goldRanksLocal{2.0, 4.0, 2.0, 2.0};
  const std::vector<double> finalGoldRanksLocal{2.0, 3.0, 2.0, 2.0};
  const std::vector<double> margins(4, -1.0);
  input.selectedGoldRank = goldRanksLocal;
  input.finalGoldRank = finalGoldRanksLocal;
  input.selectedSelfRank = goldRanksLocal;
  input.finalSelfRank = goldRanksLocal;
  input.selectedGoldMargin = margins;
  input.selectedSelfMargin = margins;
  input.selectedCrossRank = std::vector<double>(4, 7.0);
  input.finalCrossRank = std::vector<double>(4, 2.0);
  const auto localOnly = ma::attribute(input);
  assert(localOnly.localHit && !localOnly.driftHit);
  assert(localOnly.attribution == ma::Attribution::LocalRankingFailure);

  ma::AttributionInput driftInput = input;
  driftInput.selectedFirstError = 1;
  const std::vector<double> goldRanks(4, 1.0);
  const std::vector<double> goldMargins(4, 1.0);
  driftInput.selectedGoldRank = goldRanks;
  driftInput.finalGoldRank = goldRanks;
  driftInput.selectedGoldMargin = goldMargins;
  driftInput.selectedSelfMargin = goldMargins;
  driftInput.selectedSelfRank = std::vector<double>(4, 4.0);
  driftInput.finalSelfRank = std::vector<double>(4, 4.0);
  driftInput.selectedCrossRank = std::vector<double>(4, 7.0);
  driftInput.finalCrossRank = std::vector<double>{7.0, 7.0, 1.0, 1.0};
  const auto drift = ma::attribute(driftInput);
  assert(!drift.localHit && drift.driftHit && drift.corroborationComputed);
  assert(drift.rank1CrossFinalOnSelected > drift.rank1CrossSelectedOnFinal);
  assert(drift.attribution == ma::Attribution::PrefixDriftAmplification);

  ma::AttributionInput perfect;
  perfect.selectedFirstError = -1;
  assert(ma::attribute(perfect).attribution == ma::Attribution::PerfectSelected);

  bool stable = false;
  assert(std::abs(ma::percentGainShare(4.0, 8.0, 144.0, &stable) - 50.0) < 1e-12);
  assert(stable);
  assert(ma::percentGainShare(4.0, 0.001, 144.0, &stable) == 0.0 && !stable);

  assert(std::abs(ma::medianLowerMiddle({3.0, 1.0, 2.0}) - 2.0) < 1e-12);
  assert(std::abs(ma::medianLowerMiddle({4.0, 2.0}) - 2.0) < 1e-12);

  PooledPair pooled;
  pooled.tokenTotal = 432;
  pooled.easyTokenCount = 300;
  pooled.percentStable = true;
  pooled.gainBothCorrect = 3.0;
  pooled.totalImprovement = 4.0;
  pooled.selfSwfcCount = 5;
  AttributionSummary empty;
  Hypothesis hypothesis = decideHypotheses(pooled, -0.5, empty);
  assert(hypothesis.h1 && hypothesis.h2 && !hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "MIXED");
  pooled.easyTokenCount = 100;
  hypothesis = decideHypotheses(pooled, -0.5, empty);
  assert(!hypothesis.h1 && hypothesis.h2 && !hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "CRITICAL_TOKEN_MARGIN_LOSS");
  pooled.selfSwfcCount = 0;
  hypothesis = decideHypotheses(pooled, -0.5, empty);
  assert(!hypothesis.h1 && !hypothesis.h2 && !hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "INCONCLUSIVE");
  pooled.selfSwfcCount = 4;
  pooled.easyTokenCount = 300;
  AttributionSummary driftOnly;
  driftOnly.drift = 5;
  hypothesis = decideHypotheses(pooled, -0.5, driftOnly);
  assert(hypothesis.h1 && hypothesis.h2 && hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "MIXED");
  pooled.easyTokenCount = 100;
  hypothesis = decideHypotheses(pooled, -0.5, driftOnly);
  assert(!hypothesis.h1 && hypothesis.h2 && hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "MIXED");
  pooled.selfSwfcCount = 0;
  hypothesis = decideHypotheses(pooled, -0.5, driftOnly);
  assert(!hypothesis.h1 && hypothesis.h2 == false && hypothesis.h3);
  assert(std::string(conclude(hypothesis)) == "PREFIX_DRIFT_AMPLIFICATION");
  std::cout << "margin_decomposition_probe_self_test=PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--self-test") {
      selfTest();
      return 0;
    }
    if (argc < 2 || std::string(argv[1]) != "--run") {
      std::cerr << "usage: margin_decomposition_probe --self-test | --run --output DIR\n";
      return 2;
    }
    fs::path output;
    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--output" && i + 1 < argc) output = argv[++i];
      else return 2;
    }
    if (output.empty()) return 2;
    return run(output);
  } catch (const std::exception& error) {
    std::cerr << "margin_decomposition_probe: " << error.what() << '\n';
    return 3;
  }
}
