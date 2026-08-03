// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host tests for the critical-margin checkpoint objective library
// (critical_margin_objective_lib.h): dataset identity/separation, objective
// formulas and numeric stability, checkpoint selection tie-breaks, LOSO
// composition, and the development gate. All expected values are computed by
// hand or with independent arithmetic and pinned in comments.
#include "critical_margin_objective_lib.h"
#include "margin_analysis_lib.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace cm = phonelm::critical_margin;
namespace ma = phonelm::margin_analysis;
namespace ar = phonelm::autoregressive_validation;

using cm::CheckpointMetrics;
using cm::CaseTrace;

static CaseTrace trace(const std::string& id, std::vector<double> margins,
                       double nll, std::uint64_t tokenExact,
                       bool sequenceExact, int firstErrorPosition = -1) {
  CaseTrace result;
  result.id = id;
  result.margins = std::move(margins);
  result.autoregressiveNllSum = nll;
  result.tokenExact = tokenExact;
  result.sequenceExact = sequenceExact;
  result.firstErrorPosition = firstErrorPosition;
  result.finite = true;
  return result;
}

static CheckpointMetrics checkpoint(int step, std::vector<CaseTrace> cases) {
  return cm::summarizeCheckpoint(step, std::move(cases));
}

static void testPinnedDatasetIdentity() {
  const auto calibration = cm::buildMarginCalibrationV1();
  const auto development = cm::buildMarginDevelopmentV1();
  assert(calibration.size() == 24);
  assert(development.size() == 24);
  assert(cm::hashPartition(cm::Partition::CALIBRATION) ==
         std::string(cm::kCalibrationHash));
  assert(cm::hashPartition(cm::Partition::DEVELOPMENT) ==
         std::string(cm::kDevelopmentHash));
  // Generator is deterministic: hashing twice must give the same bytes.
  assert(cm::partitionHash(cm::Partition::CALIBRATION) ==
         cm::partitionHash(cm::Partition::CALIBRATION));
  assert(cm::partitionHash(cm::Partition::DEVELOPMENT) ==
         cm::partitionHash(cm::Partition::DEVELOPMENT));
  std::string error;
  assert(cm::validateDatasets(8, &error, /*requirePinnedHashes=*/true));
  assert(error.empty());
  assert(cm::kObjectives.size() <= 12);
  for (const auto& spec : cm::kObjectives) assert(spec.id != nullptr);
}

static void testDatasetSeparation() {
  const auto calibration = cm::buildMarginCalibrationV1();
  const auto development = cm::buildMarginDevelopmentV1();
  // Case, prefix, and full-sequence separation is absolute.
  assert(cm::countCaseIdOverlap(calibration, development) == 0);
  assert(cm::countInitialPrefixOverlap(calibration, development) == 0);
  assert(cm::countFullSequenceOverlap(calibration, development) == 0);
  // Successor transitions are shared structurally (same 4 family cycles):
  // 13 unique transitions, 144 free-running occurrences, exactly like the
  // AR V3 fresh-partition contract.
  assert(cm::countTransitionOverlap(calibration, false, development, false) ==
         ar::kExpectedTargetTransitionCount);
  assert(cm::countTransitionOccurrenceOverlap(calibration, false, development,
                                               false) ==
         ar::kExpectedFreshTargetOccurrences);
  // No case-id overlap with any AR V3 partition either.
  for (const auto arPartition : ar::kPartitions) {
    const auto arCases = ar::cases(arPartition, 8);
    assert(cm::countCaseIdOverlap(calibration, arCases) == 0);
    assert(cm::countCaseIdOverlap(development, arCases) == 0);
  }
}

static void testObjectiveFormulas() {
  // MARGIN_DEFICIT_MEAN: mean(max(0, delta - margin)).
  // margins {0.5, -1.5, 2.0, -0.25}, delta 0 -> (0 + 1.5 + 0 + 0.25)/4.
  CheckpointMetrics cp = checkpoint(
      4, {trace("c1", {0.5, -1.5, 2.0, -0.25}, 4.0, 2, false, 2)});
  assert(std::abs(cm::computeMarginDeficitMean(cp, 0.0) - 0.4375) < 1e-12);
  // delta 0.5 -> (0 + 2.0 + 0 + 0.75)/4 = 0.6875.
  assert(std::abs(cm::computeMarginDeficitMean(cp, 0.5) - 0.6875) < 1e-12);
  // LOWER_TAIL_MARGIN: lower-tail mean; sorted {-1.5, -0.25, 0.5, 2.0}.
  assert(std::abs(cm::computeLowerTailMargin(cp, 0.25) + 1.5) < 1e-12);
  assert(std::abs(cm::computeLowerTailMargin(cp, 0.5) + 0.875) < 1e-12);
  // SOFT_WORST_MARGIN tau=1 on margins {-3, -2, 1}: 3 + log(1 + e^-1 +
  // e^-3) - log(3) = 2.2503999281...
  const CheckpointMetrics worst = checkpoint(
      8, {trace("w1", {-3.0, -2.0, 1.0}, 3.0, 1, false, 1)});
  assert(std::abs(cm::computeSoftWorstMargin(worst, 1.0) - 2.2503999281000766) <
         1e-9);
  // SEQUENCE_SURVIVAL_NLL tau=1: softplus(3) + softplus(2) + softplus(-1)
  // = 3.0485873 + 2.1269280 + 0.3132617 = 5.4887770...
  assert(std::abs(cm::computeSequenceSurvivalNll(worst, 1.0) -
                  5.488777050134938) < 1e-9);
  // FIRST_ERROR_HAZARD on {-3, -2, 1}: weights 1/1, 1/2, 1/3 over deficits
  // {3, 2, 0}: (3 + 1) / (11/6) = 24/11 = 2.1818181...
  assert(std::abs(cm::computeFirstErrorHazard(worst) - 24.0 / 11.0) < 1e-12);
}

static void testNumericStability() {
  // softplus/logsumexp must not overflow on huge margins.
  const CheckpointMetrics huge = checkpoint(
      16, {trace("h1", {-1e6, 1e6}, 8.0, 1, false, 1)});
  const double softWorst = cm::computeSoftWorstMargin(huge, 1.0);
  assert(std::isfinite(softWorst));
  assert(std::abs(softWorst - (1e6 - std::log(2.0))) < 1e-6);
  const CheckpointMetrics hugeNeg = checkpoint(
      16, {trace("h2", {-1e6, -2e6}, 8.0, 1, false, 1)});
  const double survival = cm::computeSequenceSurvivalNll(hugeNeg, 0.25);
  assert(std::isfinite(survival));
  // softplus(4e6) + softplus(8e6) ~= 1.2e7 dominates the sum.
  assert(survival > 1.19e7 && survival < 1.21e7);
  // Invalid quantile is rejected, not silently accepted.
  const CheckpointMetrics cp = checkpoint(
      4, {trace("c1", {0.5, -1.5}, 2.0, 1, false, 1)});
  assert(std::isnan(cm::computeLowerTailMargin(cp, 0.0)));
  assert(std::isnan(cm::computeLowerTailMargin(cp, 1.5)));
}

static void testNonFiniteRejection() {
  // A NaN margin makes the whole checkpoint non-finite; every objective and
  // every selection path must fail closed on it.
  CaseTrace nanTrace = trace("n1", {0.5, -1.5}, 2.0, 1, false, 1);
  nanTrace.finite = true;
  nanTrace.margins[0] = std::numeric_limits<double>::quiet_NaN();
  const CheckpointMetrics nanCheckpoint =
      cm::summarizeCheckpoint(4, {nanTrace});
  assert(!nanCheckpoint.allFinite);
  for (const auto& spec : cm::kObjectives) {
    const auto score = cm::scoreObjective(spec, nanCheckpoint);
    assert(!score.finite);
  }
  const CheckpointMetrics good = checkpoint(
      4, {trace("g1", {0.5, -1.5}, 2.0, 1, false, 1)});
  const CheckpointMetrics bad = checkpoint(
      8, {trace("b1", {std::numeric_limits<double>::infinity()}, 2.0, 0, false)});
  assert(!bad.allFinite);
  // betterCheckpoint never accepts a non-finite candidate and the selection
  // index stays with the last finite checkpoint (or -1).
  assert(!cm::betterCheckpoint(cm::kObjectives[0], bad, good));
  const std::vector<CheckpointMetrics> trajectory{bad, good, bad};
  assert(cm::selectCheckpoint(cm::kObjectives[0], trajectory) == 1);
  // Empty traces are invalid: no finite checkpoint can be formed from them.
  const CheckpointMetrics empty = checkpoint(4, {});
  assert(!empty.allFinite);
}

static void testTieBreaks() {
  // Objective tie within kScoreTieTolerance -> NLL -> token exact ->
  // sequence exact -> earlier step.
  const auto a = checkpoint(4, {trace("a", {0.1, -0.2}, 1.5, 1, false, 1)});
  const auto bLowerNll = checkpoint(4, {trace("a", {0.1, -0.2}, 1.0, 1, false, 1)});
  assert(cm::betterCheckpoint(cm::kObjectives[0], bLowerNll, a));
  assert(!cm::betterCheckpoint(cm::kObjectives[0], a, bLowerNll));
  const auto moreTokens = checkpoint(4, {trace("a", {0.1, -0.2}, 1.0, 2, false, 1)});
  assert(cm::betterCheckpoint(cm::kObjectives[0], moreTokens, bLowerNll));
  const auto moreSequences =
      checkpoint(4, {trace("a", {0.1, -0.2}, 1.0, 2, true, 1)});
  assert(cm::betterCheckpoint(cm::kObjectives[0], moreSequences, moreTokens));
  const auto later = checkpoint(8, {trace("a", {0.1, -0.2}, 1.0, 2, true, 1)});
  assert(cm::betterCheckpoint(cm::kObjectives[0], moreSequences, later));
  assert(!cm::betterCheckpoint(cm::kObjectives[0], later, moreSequences));
}

static void testSelectionAndScoreComparison() {
  const std::vector<CheckpointMetrics> trajectory{
      checkpoint(4, {trace("s", {0.5, -1.5}, 2.0, 1, false, 2)}),
      checkpoint(16, {trace("s", {0.5, -0.2}, 1.0, 2, true, -1)}),
      checkpoint(64, {trace("s", {0.5, -0.1}, 1.2, 2, true, -1)}),
  };
  // Deficits: 0.75 (step 4), 0.10 (step 16), 0.05 (step 64) -> step 64.
  const int selected = cm::selectCheckpoint(cm::kObjectives[0], trajectory);
  assert(selected == 2);
  const auto result = cm::selectBestCheckpoint(cm::kObjectives[0], trajectory);
  assert(result.selected && result.trajectoryIndex == 2 && result.step == 64);
  assert(result.score.finite);
  assert(result.score.value <
         cm::scoreObjective(cm::kObjectives[0], trajectory[0]).value);
  // compareObjectiveScores: -1 left better, +1 right better, 0 tie; a
  // non-finite score loses to any finite one.
  const auto finiteScore = cm::scoreObjective(cm::kObjectives[0], trajectory[1]);
  cm::ObjectiveScore nonFinite;
  assert(cm::compareObjectiveScores(finiteScore, nonFinite) == -1);
  assert(cm::compareObjectiveScores(nonFinite, finiteScore) == 1);
  assert(cm::compareObjectiveScores(nonFinite, nonFinite) == 0);
  const auto sameScore = cm::scoreObjective(cm::kObjectives[0], trajectory[2]);
  // trajectory[1] and [2] differ by 0.1 > tie tolerance; left is better.
  assert(cm::compareObjectiveScores(sameScore, finiteScore) == -1);
}

static void testSpearmanAndMedians() {
  const std::vector<double> left{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> right{2.0, 4.0, 6.0, 8.0};
  const auto perfect = cm::spearman(left, right);
  assert(perfect.available && std::abs(perfect.value - 1.0) < 1e-12);
  const std::vector<double> inverted{8.0, 6.0, 4.0, 2.0};
  const auto anti = cm::spearman(left, inverted);
  assert(anti.available && std::abs(anti.value + 1.0) < 1e-12);
  // Tied values collapse to midranks: [1, 1, 2] ranks {1.5, 1.5, 3} vs
  // {1, 2, 3}: r = 1.5/sqrt(3) = 0.8660254.
  const std::vector<double> ties{1.0, 1.0, 2.0};
  const std::vector<double> monotone{1.0, 2.0, 3.0};
  const auto tied = cm::spearman(ties, monotone, 0.0);
  assert(tied.available &&
         std::abs(tied.value - 0.8660254037844386) < 1e-9);
  assert(cm::lowerMiddleMedian({3.0, 1.0, 2.0}) == 2.0);
  assert(cm::lowerMiddleMedian({4.0, 1.0, 2.0, 3.0}) == 2.0);
  assert(cm::lowerMiddleMedian({}) == 0.0);
}

static void testVariantEvidencePriority() {
  // Fixed candidate priority: development sequence exact, development token
  // exact, worst-seed token exact, first-error position, lower-tail margin,
  // preregistered variant order, earlier checkpoint.
  cm::VariantEvidence base{"A", 0, 10, 30, 25, 4.0, 0.5, 64};
  cm::VariantEvidence betterSequences{"B", 1, 11, 30, 25, 4.0, 0.5, 64};
  cm::VariantEvidence betterTokens{"C", 2, 10, 31, 25, 4.0, 0.5, 64};
  cm::VariantEvidence betterWorstSeed{"D", 3, 10, 30, 26, 4.0, 0.5, 64};
  cm::VariantEvidence betterFirstError{"E", 4, 10, 30, 25, 5.0, 0.5, 64};
  cm::VariantEvidence betterTail{"F", 5, 10, 30, 25, 4.0, 1.2, 64};
  cm::VariantEvidence simpler{"G", 6, 10, 30, 25, 4.0, 0.5, 64};
  cm::VariantEvidence earlier{"H", 0, 10, 30, 25, 4.0, 0.5, 32};
  assert(cm::betterVariantEvidence(betterSequences, base));
  assert(cm::betterVariantEvidence(betterTokens, base));
  assert(cm::betterVariantEvidence(betterWorstSeed, base));
  assert(cm::betterVariantEvidence(betterFirstError, base));
  assert(cm::betterVariantEvidence(betterTail, base));
  assert(cm::betterVariantEvidence(base, simpler));   // simpler = lower priority
  assert(cm::betterVariantEvidence(earlier, base));   // earlier step wins ties
  assert(!cm::betterVariantEvidence(base, base));
}

static std::map<std::uint32_t, std::vector<CheckpointMetrics>>
makeTrajectoryMap(std::uint32_t seed, const std::vector<int>& steps,
                  const std::vector<std::uint64_t>& tokenExactAtSteps) {
  std::map<std::uint32_t, std::vector<CheckpointMetrics>> result;
  std::vector<CheckpointMetrics> trajectory;
  for (std::size_t i = 0; i < steps.size(); ++i)
    trajectory.push_back(checkpoint(
        steps[i],
        {trace("seed" + std::to_string(seed) + "-case1",
               {0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.4, -0.4}, 2.0,
               tokenExactAtSteps[i], tokenExactAtSteps[i] == 6, 2)}));
  result[seed] = std::move(trajectory);
  return result;
}

static void testLeaveOneSeedOut() {
  // Seeds 1 and 4: MARGIN_DEFICIT_MEAN_D0 selects step 4 (dev token exact
  // 6/8); seed 2 behaves differently (its step-4 dev is 2, step-320 dev 5).
  // With seeds 2+4 training the held-out seed 1, D0 pools 5+6=11 tokens
  // while LOWER_TAIL_MARGIN_Q10 pools 5+6=11 too, so the preregistered
  // priority (D0 first) breaks the tie.
  const auto calib1 = makeTrajectoryMap(1, {4, 320}, {6, 3});
  const auto calib4 = makeTrajectoryMap(4, {4, 320}, {6, 3});
  const auto calib2 = makeTrajectoryMap(2, {4, 320}, {2, 5});
  auto dev1 = makeTrajectoryMap(1, {4, 320}, {6, 3});
  auto dev4 = makeTrajectoryMap(4, {4, 320}, {6, 3});
  auto dev2 = makeTrajectoryMap(2, {4, 320}, {2, 5});
  auto calib = calib1;
  calib.insert(calib2.begin(), calib2.end());
  calib.insert(calib4.begin(), calib4.end());
  auto dev = dev1;
  dev.insert(dev2.begin(), dev2.end());
  dev.insert(dev4.begin(), dev4.end());
  const auto folds = cm::runLeaveOneSeedOut({1, 2, 4}, calib, dev);
  assert(folds.size() == 3);
  const auto& fold1 = folds[0];
  assert(fold1.heldOutSeed == 1);
  assert(fold1.chosenObjective == "MARGIN_DEFICIT_MEAN_D0");
  assert(fold1.selectedStep == 4);
  assert(fold1.tokenExact == 6 && fold1.tokenTotal == 8);
  assert(fold1.finalStepTokenExactDelta == 3);
  assert(fold1.finite);
  // Collapse: the 320-step trace was constructed with the same case id and
  // positive tokenExact; the 4-step candidate keeps 20, so no collapse.
  assert(fold1.collapseFree);
  // Held-out seed 2: train seeds 1+4 both select 4 (20+20); D0 wins over
  // any variant that would pick the seed-2-specific step.
  const auto& fold2 = folds[1];
  assert(fold2.heldOutSeed == 2);
  assert(fold2.chosenObjective == "MARGIN_DEFICIT_MEAN_D0");
  // Held-out seed 2 selects the 320-step checkpoint: its calibration tie on
  // the objective is broken by token exact (5 > 2).
  assert(fold2.selectedStep == 320);
}

static void testCaseCollapseAndGateHelpers() {
  const auto good = checkpoint(4, {trace("g", {0.5, -0.5}, 1.0, 2, true, -1)});
  const auto worse = checkpoint(320, {trace("g", {0.5, -0.5}, 1.0, 1, false, 1)});
  const auto zeroed = checkpoint(320, {trace("g", {0.5, -0.5}, 1.0, 0, false, 1)});
  const auto renamed = checkpoint(320, {trace("x", {0.5, -0.5}, 1.0, 2, true, -1)});
  assert(!cm::caseCollapse(good, worse));
  assert(cm::caseCollapse(zeroed, good));  // candidate zeroes an exact case
  assert(cm::caseCollapse(good, renamed));
  assert(cm::nonworseExact(good, worse));
  assert(!cm::nonworseExact(worse, good));
  assert(cm::strictlyImprovesExact(good, worse));
  assert(!cm::strictlyImprovesExact(worse, good));
}

static cm::DevelopmentRecord record(std::uint32_t seed,
                                    const CheckpointMetrics& candidate,
                                    const CheckpointMetrics& finalStep,
                                    std::vector<CheckpointMetrics> adjacent) {
  cm::DevelopmentRecord result;
  result.seed = seed;
  result.candidate = candidate;
  result.finalStep = finalStep;
  result.adjacentCheckpoints = std::move(adjacent);
  return result;
}

static void testDevelopmentGate() {
  const std::string caseId = "gate-case";
  const auto make = [&](int step, bool sequence, double medianSurvival,
                        std::uint64_t exact) {
    CheckpointMetrics cp = cm::summarizeCheckpoint(
        step, {trace(caseId, {0.6, -0.6, 0.5, -0.5, 0.4, -0.4, 0.3, -0.3}, 2.0,
                     exact, sequence, 2)});
    cp.medianFirstErrorSurvival = medianSurvival;
    cp.allFinite = std::isfinite(cp.lowerTailMarginQ10);
    return cp;
  };
  const auto candidate = make(16, false, 3.0, 3);
  const auto finalStep = make(320, false, 2.0, 1);
  const auto neighbor = make(12, false, 3.0, 2);
  // All three L19 seeds improve or hold, seed 2 strictly improves, control
  // (L18 seed 2) is not worse, and the first-error median is not worse.
  const std::array<cm::DevelopmentRecord, 3> l19{
      record(1, candidate, finalStep, {neighbor}),
      record(2, candidate, finalStep, {neighbor}),
      record(4, candidate, finalStep, {neighbor}),
  };
  const auto control = record(2, candidate, finalStep, {neighbor});
  const auto result = cm::developmentGate(l19, control);
  assert(result.pass);
  assert(result.seed2Strict && result.pooledTokenNonworse &&
         result.pooledSequenceNonworse && result.controlNonworse &&
         result.firstErrorMedianNonworse);
  assert(result.supportedSeeds == 3 && result.stableSupportedSeeds == 3);
  assert(result.noCaseCollapse);

  // seed 2 strictly worse -> reject.
  const auto seed2Worse = record(2, finalStep, candidate, {neighbor});
  std::array<cm::DevelopmentRecord, 3> broken{record(1, candidate, finalStep,
                                                     {neighbor}),
                                              seed2Worse,
                                              record(4, candidate, finalStep,
                                                     {neighbor})};
  const auto rejectSeed2 = cm::developmentGate(broken, control);
  assert(!rejectSeed2.pass && !rejectSeed2.seed2Strict);

  // Control better than its final step -> reject.
  const auto betterControl = record(2, candidate, make(320, true, 4.0, 3),
                                    {neighbor});
  const auto rejectControl = cm::developmentGate(l19, betterControl);
  assert(!rejectControl.pass && !rejectControl.controlNonworse);

  // Only one supported seed -> reject.
  std::array<cm::DevelopmentRecord, 3> unsupported{
      record(1, finalStep, candidate, {neighbor}),
      record(2, candidate, finalStep, {neighbor}),
      record(4, finalStep, candidate, {neighbor})};
  const auto rejectSupport = cm::developmentGate(unsupported, control);
  assert(!rejectSupport.pass && rejectSupport.supportedSeeds == 1);

  // No stable neighbor -> reject.
  const auto unstable =
      record(2, candidate, finalStep, {make(12, false, 1.0, 0)});
  const auto rejectStable = cm::developmentGate(
      {record(1, candidate, finalStep, {}), unstable,
       record(4, candidate, finalStep, {})},
      control);
  assert(!rejectStable.pass && rejectStable.stableSupportedSeeds < 2);

  // Case collapse (candidate zeroes a case that was exact at the final
  // step) -> reject.
  const auto collapsedCandidate = cm::summarizeCheckpoint(
      16, {trace(caseId, {0.6, -0.6, 0.5, -0.5, 0.4, -0.4, 0.3, -0.3}, 2.0, 0,
                 false, 1)});
  assert(collapsedCandidate.allFinite);
  assert(cm::caseCollapse(collapsedCandidate, finalStep));
  const auto rejectCollapse = cm::developmentGate(
      {record(1, collapsedCandidate, finalStep, {neighbor}),
       record(2, candidate, finalStep, {neighbor}),
       record(4, candidate, finalStep, {neighbor})},
      control);
  assert(!rejectCollapse.pass && !rejectCollapse.noCaseCollapse);
}

static void testMarginConventionMatchesMarginAnalysis() {
  // The margin convention is expectedMinusTop1Margin from
  // margin_analysis_lib.h: gold logit minus best non-gold logit, positive
  // exactly when the gold token is strictly the argmax.
  const std::vector<double> logits{1.0, 3.0, 2.0, 0.5};
  std::vector<double> probabilities;
  probabilities.reserve(4);
  for (const double value : logits) probabilities.push_back(std::exp(value));
  const double sum = std::accumulate(probabilities.begin(),
                                     probabilities.end(), 0.0);
  for (double& value : probabilities) value /= sum;
  const auto score = ma::scoreFromLogits(logits, probabilities, 1);
  assert(score.valid);
  assert(score.predicted == 1);
  assert(std::abs(score.expectedMinusTop1Margin - 1.0) < 1e-12);
  assert(std::abs(score.expectedRank - 1.0) < 1e-12);
  assert(std::abs(score.tokenNll + std::log(probabilities[1])) < 1e-12);
  // Gold not argmax: margin negative, rank > 1.
  const auto other = ma::scoreFromLogits(logits, probabilities, 2);
  assert(std::abs(other.expectedMinusTop1Margin + 1.0) < 1e-12);
  assert(std::abs(other.expectedRank - 2.0) < 1e-12);
  // Gold is strictly argmax iff expected margin >= 0 (no competitor ties the
  // gold logit in this setup).
  for (int gold = 0; gold < 4; ++gold) {
    const auto s = ma::scoreFromLogits(logits, probabilities, gold);
    assert(s.valid);
    assert((s.expectedMinusTop1Margin >= 0.0) ==
           (s.predicted == static_cast<std::uint32_t>(gold)));
  }
}

int main() {
  testPinnedDatasetIdentity();
  testDatasetSeparation();
  testObjectiveFormulas();
  testNumericStability();
  testNonFiniteRejection();
  testTieBreaks();
  testSelectionAndScoreComparison();
  testSpearmanAndMedians();
  testVariantEvidencePriority();
  testLeaveOneSeedOut();
  testCaseCollapseAndGateHelpers();
  testDevelopmentGate();
  testMarginConventionMatchesMarginAnalysis();
  std::printf("critical_margin_objective_host_test=PASS\n");
  return 0;
}
