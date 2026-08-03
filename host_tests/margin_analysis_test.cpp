// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host tests for the first-error/margin decomposition library
// (margin_analysis_lib.h) and the post-hoc best-token/best-sequence
// bookkeeping added to depth_quality_lib.h.
#include "depth_quality_lib.h"
#include "margin_analysis_lib.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace ma = phonelm::margin_analysis;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;

static void testRankAndMarginMath() {
  const std::vector<double> logits{0.0, 1.0, 2.0, 0.5};
  const std::vector<double> probabilities{0.2, 0.3, 0.4, 0.1};
  assert(ma::argmaxFirst(logits) == 2);
  assert(ma::expectedRank(logits, 0) == 4.0);
  assert(ma::expectedRank(logits, 1) == 2.0);
  assert(ma::expectedRank(logits, 2) == 1.0);
  assert(ma::expectedRank(logits, 3) == 3.0);
  assert(std::abs(ma::expectedMinusTop1Margin(logits, 2) - 1.0) < 1e-12);
  assert(std::abs(ma::expectedMinusTop1Margin(logits, 0) + 2.0) < 1e-12);
  assert(std::abs(ma::top1MinusTop2Margin(logits) - 1.0) < 1e-12);
  const auto top = ma::topTwoProbabilities(probabilities);
  assert(std::abs(top.first - 0.4) < 1e-12 && std::abs(top.second - 0.3) < 1e-12);

  std::vector<double> tie{1.0, 1.0, 0.0};
  assert(ma::argmaxFirst(tie) == 0);
  assert(ma::expectedRank(tie, 0) == 1.5);
  assert(ma::expectedRank(tie, 1) == 1.5);
  assert(ma::expectedRank(tie, 2) == 3.0);
  const std::vector<double> uniform(32, 1.0 / 32.0);
  assert(std::abs(ma::entropyOf(uniform) - std::log(32.0)) < 1e-9);
  const auto score = ma::scoreFromLogits(logits, probabilities, 2);
  assert(score.valid && score.predicted == 2 && score.expectedRank == 1.0);
  assert(std::abs(score.tokenNll + std::log(0.4)) < 1e-12);
  assert(std::abs(score.top1Probability - 0.4) < 1e-12);
  const auto invalid = ma::scoreFromLogits(logits, probabilities, 99);
  assert(!invalid.valid);
}

static void testBucketsAndFirstError() {
  assert(ma::classifyBucket(true, true) == ma::Bucket::BothCorrect);
  assert(ma::classifyBucket(true, false) ==
         ma::Bucket::SelectedCorrectFinalWrong);
  assert(ma::classifyBucket(false, true) ==
         ma::Bucket::SelectedWrongFinalCorrect);
  assert(ma::classifyBucket(false, false) == ma::Bucket::BothWrong);
  assert(std::string(ma::bucketName(ma::Bucket::SelectedWrongFinalCorrect)) ==
         "SELECTED_WRONG_FINAL_CORRECT");

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
  assert(std::string(ma::caseClassName(ma::CaseClass::NoError)) == "NO_ERROR");
}

static void testAttribution() {
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

  ma::AttributionInput tied = input;
  tied.selectedTieAtFirstErrorGold = true;
  const auto tiedResult = ma::attribute(tied);
  assert(!tiedResult.localHit);
  assert(tiedResult.attribution == ma::Attribution::NoClearAttribution);

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

  ma::AttributionInput uncorroborated = driftInput;
  uncorroborated.finalCrossRank = std::vector<double>(4, 7.0);
  const auto uncorrResult = ma::attribute(uncorroborated);
  assert(uncorrResult.driftHit && uncorrResult.corroborationComputed);
  assert(uncorrResult.attribution ==
         ma::Attribution::PrefixDriftUncorroborated);

  ma::AttributionInput mixed = input;
  mixed.selectedFirstError = 1;
  mixed.selectedGoldRank = goldRanksLocal;
  mixed.finalGoldRank = finalGoldRanksLocal;
  mixed.selectedGoldMargin = margins;
  mixed.selectedSelfMargin = margins;
  mixed.selectedSelfRank = std::vector<double>(4, 4.0);
  mixed.finalSelfRank = std::vector<double>(4, 4.0);
  const auto mixedResult = ma::attribute(mixed);
  assert(mixedResult.localHit && mixedResult.driftHit);
  assert(mixedResult.attribution == ma::Attribution::Mixed);

  ma::AttributionInput perfect;
  perfect.selectedFirstError = -1;
  assert(ma::attribute(perfect).attribution == ma::Attribution::PerfectSelected);

  ma::AttributionInput badSizes = driftInput;
  badSizes.selectedGoldRank.resize(2);
  const auto badResult = ma::attribute(badSizes);
  assert(badResult.attribution == ma::Attribution::NoClearAttribution);

  assert(std::string(ma::attributionName(ma::Attribution::Mixed)) == "MIXED");
}

static void testBucketStatsAndPercentShare() {
  ma::BucketStats stats;
  const auto selected = ma::scoreFromLogits(
      {0.0, 1.0, 2.0}, {0.2, 0.3, 0.5}, 2);
  const auto final = ma::scoreFromLogits(
      {0.0, 1.0, 2.5}, {0.15, 0.3, 0.55}, 2);
  stats.add(true, true, selected, final, false, false);
  stats.add(true, true, selected, final, false, false);
  assert(stats.tokenCount == 2);
  assert(std::abs(stats.nllContribution() -
                  (stats.nllSumSelected - stats.nllSumFinal)) < 1e-9);
  assert(std::abs(stats.nllContribution() - 2.0 * std::log(1.1)) < 1e-9);
  assert(stats.meanRankSelected() > 0.0 && stats.meanRankFinal() > 0.0);
  assert(stats.meanMarginSelected() > 0.0);
  assert(stats.meanEntropySelected() > 0.0);
  assert(stats.firstErrorPositionsSelected == 0);
  assert(stats.firstErrorPositionsFinal == 0);

  bool stable = false;
  assert(std::abs(ma::percentGainShare(4.0, 8.0, 144.0, &stable) - 50.0) < 1e-12);
  assert(stable);
  assert(ma::percentGainShare(4.0, 0.001, 144.0, &stable) == 0.0 && !stable);
  assert(ma::percentGainShare(4.0, -8.0, 144.0, &stable) == 0.0 && !stable);
  assert(ma::percentGainShare(4.0, 8.0, 0.0, &stable) == 0.0 && !stable);

  assert(std::abs(ma::medianLowerMiddle({3.0, 1.0, 2.0}) - 2.0) < 1e-12);
  assert(std::abs(ma::medianLowerMiddle({4.0, 2.0}) - 2.0) < 1e-12);
  assert(ma::medianLowerMiddle({}) == 0.0);
}

static void testBestTokenSequenceBookkeeping() {
  phonelm::tiny_lm::Config config;
  config.tokens = 8;
  config.vocabularySize = 32;
  config.dimension = 16;
  config.feedForwardDimension = 32;
  config.numLayers = 2;
  config.numHeads = 2;
  std::string error;
  assert(phonelm::tiny_lm::validateConfig(config, &error));

  const auto finalStep = dq::runAutoregressiveSelectedCpu(
      config, 1, 4, dq::AutoregressiveSelectionMode::FINAL_STEP);
  assert(finalStep.bestTokenExactStep == -1);
  assert(finalStep.bestSequenceExactStep == -1);
  assert(finalStep.bestTokenExactCount == 0);
  assert(finalStep.bestSequenceExactCount == 0);

  const auto best = dq::runAutoregressiveSelectedCpu(
      config, 1, 4, dq::AutoregressiveSelectionMode::BEST_AR_VALIDATION_V1);
  assert(best.bestTokenExactStep == -1 || best.bestTokenExactStep == 0 ||
         best.bestTokenExactStep == 4);
  assert(best.bestSequenceExactStep == -1 || best.bestSequenceExactStep == 0 ||
         best.bestSequenceExactStep == 4);
  if (best.bestTokenExactStep >= 0) {
    bool tokenExactMatch = false;
    for (const auto& entry : best.validationTrajectory) {
      if (entry.first == best.bestTokenExactStep)
        tokenExactMatch = entry.second.tokenExact == best.bestTokenExactCount;
    }
    assert(tokenExactMatch);
  }
  if (best.bestSequenceExactStep >= 0) {
    bool sequenceExactMatch = false;
    for (const auto& entry : best.validationTrajectory) {
      if (entry.first == best.bestSequenceExactStep)
        sequenceExactMatch =
            entry.second.sequenceExact == best.bestSequenceExactCount;
    }
    assert(sequenceExactMatch);
  }
}

int main() {
  testRankAndMarginMath();
  testBucketsAndFirstError();
  testAttribution();
  testBucketStatsAndPercentShare();
  testBestTokenSequenceBookkeeping();
  std::printf("margin_analysis_host_test=PASS\n");
  return 0;
}
