// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only CPU checkpoint-objective probe for the L19 critical-margin
// stabilization investigation. It deterministically regenerates the pinned
// CPU reference trajectories (L19 seeds 1/2/4 plus the L18 seed-2 control),
// evaluates the MARGIN_CALIBRATION_V1 / MARGIN_DEVELOPMENT_V1 partitions at
// every evaluation step, scores all 12 preregistered checkpoint objectives,
// runs the leave-one-seed-out protocol and the development gate, verifies
// the canonical AR_DEVELOPMENT_V3 anchors, and recomputes per-step
// gradient/loss attribution with bitwise parity against the training loop.
//
// Everything written here is private evidence under build/reports/; the
// public bundle is produced separately by the allow-list exporter.
// AR_FINAL_HOLDOUT_V3 is never evaluated.
#include "critical_margin_objective_lib.h"
#include "critical_margin_training_lib.h"
#include "depth_quality_lib.h"
#include "margin_analysis_lib.h"

#include <algorithm>
#include <array>
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
namespace tiny = phonelm::tiny_lm;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;
namespace ma = phonelm::margin_analysis;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;

namespace {

constexpr int kSteps = 320;
constexpr int kTokens = 8;
constexpr double kUpdateImbalanceFactor = 3.0;
constexpr double kUpdateImbalanceFraction = 0.25;
constexpr double kDepthRatioThreshold = 3.0;
constexpr double kTrainRankOverfitThreshold = 1.05;
constexpr double kCriticalShareShortfall = -0.05;
constexpr double kCriticalTokenShareFloor = 0.10;

struct RunSpec {
  const char* publicId;
  int layers;
  std::uint32_t seed;
  int pinnedSelectedStep;
  std::uint64_t selectedTokenExact;
  std::uint64_t finalTokenExact;
  std::uint64_t selectedSequenceExact;
  std::uint64_t finalSequenceExact;
};

// Canonical anchors from docs/results/qnn-htp-autoregressive-validation-2026-08
// and docs/results/qnn-htp-l19-first-error-margin-2026-08 (AR_DEVELOPMENT_V3
// counts at the pinned AR-selected step and at step 320).
const std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", 19, 1, 16, 14, 30, 0, 2},
    {"L19_SEED_2", 19, 2, 4, 20, 63, 0, 6},
    {"L19_SEED_4", 19, 4, 12, 22, 46, 0, 6},
    {"L18_SEED_2_CONTROL", 18, 2, 4, 18, 65, 0, 8},
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

// Free-running rollout over one margin partition at one checkpoint.
cm::CheckpointMetrics evaluatePartition(const tiny::Config& config,
                                        const dq::Params& params, int step,
                                        const std::vector<cm::DatasetCase>& cases) {
  std::vector<cm::CaseTrace> traces;
  traces.reserve(cases.size());
  for (const auto& item : cases) {
    cm::CaseTrace trace;
    trace.id = item.id;
    std::vector<std::uint32_t> context = item.initialPrefix;
    std::vector<std::uint32_t> predicted;
    predicted.reserve(item.targets.size());
    bool finite = true;
    for (const std::uint32_t truth : item.targets) {
      ma::Score score;
      try {
        score = scorePosition(config, params, context, truth);
      } catch (const std::exception&) {
        finite = false;
        break;
      }
      trace.margins.push_back(score.expectedMinusTop1Margin);
      trace.autoregressiveNllSum += score.tokenNll;
      predicted.push_back(score.predicted);
      context = slide(context, score.predicted);
    }
    if (!finite) {
      trace.finite = false;
      traces.push_back(std::move(trace));
      continue;
    }
    const auto firstError = ma::firstErrorInfo(predicted, item.targets);
    trace.tokenExact = static_cast<std::uint64_t>(predicted.size()) -
                       firstError.wrongCount;
    trace.sequenceExact = firstError.wrongCount == 0;
    // The library contract is one-based; -1 means no error.
    trace.firstErrorPosition =
        firstError.firstError < 0 ? -1 : firstError.firstError + 1;
    trace.finite = true;
    traces.push_back(std::move(trace));
  }
  return cm::summarizeCheckpoint(step, std::move(traces));
}

// ---------------------------------------------------------------------------
// Gradient attribution: bitwise-parity retraining pass
// ---------------------------------------------------------------------------
struct GradientStepRow {
  int step = 0;
  double loss = 0.0;
  double accuracy = 0.0;
  double meanMargin = 0.0;
  double meanRank = 0.0;
  double meanNll = 0.0;
  double criticalTokenShare = 0.0;
  double criticalLossShare = 0.0;
  double easyTokenShare = 0.0;
  double gradNormTotal = 0.0;
  double gradNormEmbedding = 0.0;
  double gradNormOutput = 0.0;
  double gradNormLayerMean = 0.0;
  double gradNormFirstLayer = 0.0;
  double gradNormLastLayer = 0.0;
  double gradNormAttnShare = 0.0;
  double gradNormFfnShare = 0.0;
  double gradNormNormShare = 0.0;
  double gradNormDepthRatio = 0.0;
  double gradNormOutputShare = 0.0;
  bool lossParity = false;
  bool gradientParity = false;
};

struct GradientRun {
  std::vector<GradientStepRow> steps;
  double finalTrainMeanRank = 0.0;
};

// Re-runs the exact training loop of dq::runFormalCpu and records per-step
// per-token attribution. Every intermediate value (params, batches, Adam
// updates) is identical by construction; parity with the recorded StepMetric
// rows is asserted per step.
bool isAttn(const std::string& name) {
  return name.find(".wq") != std::string::npos ||
         name.find(".wk") != std::string::npos ||
         name.find(".wv") != std::string::npos ||
         name.find(".wo") != std::string::npos;
}
bool isFfn(const std::string& name) {
  return name.find("ffn_w1") != std::string::npos ||
         name.find("ffn_w2") != std::string::npos;
}
bool isNorm(const std::string& name) {
  return name.find("norm1_") != std::string::npos ||
         name.find("norm2_") != std::string::npos;
}
bool isLayer(const std::string& name, int layer) {
  const std::string prefix = "layer_" + std::to_string(layer) + ".";
  return name.rfind(prefix, 0) == 0;
}

// Re-runs the exact training loop of dq::runFormalCpu and records per-step
// per-token attribution. Every intermediate value (params, batches, Adam
// updates) is identical by construction; parity with the recorded StepMetric
// rows is asserted per step.
GradientRun gradientAttribution(const tiny::Config& config, std::uint32_t seed,
                                const dq::FormRun& reference) {
  dq::Params params = tiny::initialParameters(config, seed);
  params = dq::applyInitStability(config, std::move(params),
                                  dq::StabilityMode::LEGACY);
  dq::Params m, v;
  const auto zeroOut = [](dq::Params& t, const dq::Params& like) {
    t = like;
    for (const auto& e : tiny::parameterRegistry(t))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
  };
  zeroOut(m, params);
  zeroOut(v, params);
  GradientRun run;
  const int layers = static_cast<int>(config.numLayers);
  std::uint64_t rankSum = 0;
  std::uint64_t rankCount = 0;
  for (int step = 1; step <= kSteps; ++step) {
    const std::uint32_t pattern = std::uint32_t((step - 1) % 4);
    const auto batch = dq::formalBatch(config, pattern, 0);
    const auto fb = tiny::forwardBackward(config, batch.first, batch.second,
                                          params, 0.0f);
    const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
    const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
    const float lr = phonelm::stabilityLearningRate(
        std::uint32_t(dq::StabilityMode::LEGACY), 0.003f,
        std::uint32_t(step), std::uint32_t(kSteps));
    const auto update = tiny::adamUpdate(params, fb.gradients, m, v, lr, .9f,
                                         .999f, 1e-8f, c1, c2);
    GradientStepRow item;
    item.step = step;
    item.loss = static_cast<double>(fb.loss);
    item.accuracy = static_cast<double>(fb.accuracy);
    const std::size_t vocab = config.vocabularySize;
    double marginSum = 0.0, rankSumStep = 0.0, nllSum = 0.0;
    double criticalLoss = 0.0, criticalCount = 0.0, easyCount = 0.0;
    for (std::uint32_t r = 0; r < config.tokens; ++r) {
      const std::size_t base = std::size_t(r) * vocab;
      std::uint32_t truth = 0;
      for (std::uint32_t j = 0; j < vocab; ++j)
        if (batch.second[base + j] > .5f) truth = j;
      std::vector<double> logits(fb.logits.begin() + base,
                                 fb.logits.begin() + base + vocab);
      std::vector<double> probabilities(fb.probabilities.begin() + base,
                                        fb.probabilities.begin() + base +
                                            vocab);
      const auto score = ma::scoreFromLogits(logits, probabilities, truth);
      if (!score.valid) continue;
      marginSum += score.expectedMinusTop1Margin;
      rankSumStep += score.expectedRank;
      nllSum += score.tokenNll;
      const bool critical = score.expectedMinusTop1Margin < 0.0;
      if (critical) {
        ++criticalCount;
        criticalLoss += score.tokenNll;
      }
      if (score.expectedProbability >= ma::kEasyProbabilityThreshold)
        ++easyCount;
    }
    const double tokens = static_cast<double>(config.tokens);
    item.meanMargin = marginSum / tokens;
    item.meanRank = rankSumStep / tokens;
    item.meanNll = nllSum / tokens;
    item.criticalTokenShare = criticalCount / tokens;
    item.criticalLossShare =
        nllSum > 0.0 ? criticalLoss / nllSum : 0.0;
    item.easyTokenShare = easyCount / tokens;
    double totalSq = 0.0, embeddingSq = 0.0, outputSq = 0.0;
    std::vector<double> layerSq(static_cast<std::size_t>(layers), 0.0);
    double attnSq = 0.0, ffnSq = 0.0, normSq = 0.0;
    for (const auto& e : tiny::parameterRegistry(fb.gradients)) {
      const std::string name(e.name);
      double groupSq = 0.0;
      for (const float g : *e.values) groupSq += double(g) * double(g);
      totalSq += groupSq;
      if (name == "token_embedding") embeddingSq += groupSq;
      if (name == "output_projection") outputSq += groupSq;
      if (isAttn(name)) attnSq += groupSq;
      if (isFfn(name)) ffnSq += groupSq;
      if (isNorm(name)) normSq += groupSq;
      for (int layer = 0; layer < layers; ++layer)
        if (isLayer(name, layer)) layerSq[static_cast<std::size_t>(layer)] += groupSq;
    }
    item.gradNormTotal = std::sqrt(totalSq);
    item.gradNormEmbedding = std::sqrt(embeddingSq);
    item.gradNormOutput = std::sqrt(outputSq);
    item.gradNormLayerMean = std::sqrt(totalSq / double(layers));
    item.gradNormFirstLayer = std::sqrt(layerSq.front());
    item.gradNormLastLayer = std::sqrt(layerSq.back());
    item.gradNormAttnShare = totalSq > 0.0 ? attnSq / totalSq : 0.0;
    item.gradNormFfnShare = totalSq > 0.0 ? ffnSq / totalSq : 0.0;
    item.gradNormNormShare = totalSq > 0.0 ? normSq / totalSq : 0.0;
    item.gradNormOutputShare = totalSq > 0.0 ? outputSq / totalSq : 0.0;
    item.gradNormDepthRatio =
        item.gradNormFirstLayer > 0.0
            ? item.gradNormLastLayer / item.gradNormFirstLayer
            : 0.0;
    const auto& metric = reference.steps.at(static_cast<std::size_t>(step - 1));
    item.lossParity = std::abs(item.loss - static_cast<double>(metric.loss)) < 1e-12;
    const double gradDiff =
        std::abs(item.gradNormTotal - metric.gradientNorm);
    item.gradientParity =
        gradDiff < 1e-9 * std::max(1.0, item.gradNormTotal);
    if (step > kSteps - 8) {
      rankSum += static_cast<std::uint64_t>(std::llround(rankSumStep));
      rankCount += 8;
    }
    run.steps.push_back(std::move(item));
    params = update.next;
    m = update.firstMoment;
    v = update.secondMoment;
  }
  run.finalTrainMeanRank =
      rankCount > 0 ? static_cast<double>(rankSum) / static_cast<double>(rankCount)
                    : 0.0;
  return run;
}

// ---------------------------------------------------------------------------
// Attribution classification (preregistered thresholds above)
// ---------------------------------------------------------------------------
struct Attribution {
  bool criticalUnderweight = false;
  bool depthAmplification = false;
  bool outputRankDrift = false;
  bool seedImbalance = false;

  std::string classification() const {
    if (!criticalUnderweight && !depthAmplification && !outputRankDrift &&
        !seedImbalance)
      return "NO_SINGLE_TRAINING_MECHANISM";
    std::string result;
    if (criticalUnderweight) result += "UNIFORM_CE_UNDERWEIGHTS_CRITICAL_POSITIONS";
    if (depthAmplification) {
      if (!result.empty()) result += "+";
      result += "RESIDUAL_DEPTH_AMPLIFICATION";
    }
    if (outputRankDrift) {
      if (!result.empty()) result += "+";
      result += "OUTPUT_HEAD_RANKING_DRIFT";
    }
    if (seedImbalance) {
      if (!result.empty()) result += "+";
      result += "SEED_SPECIFIC_UPDATE_IMBALANCE";
    }
    return result;
  }

  std::string recommendedFamilies() const {
    if (criticalUnderweight || outputRankDrift)
      return "PAIRWISE_MARGIN_CE_V1;SEQUENCE_WORST_MARGIN_CE_V1";
    return "NONE_WITH_DIRECT_EVIDENCE";
  }
};

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------
void writeDatasetPartitions(std::ofstream& output) {
  row(output, {"partition", "domain", "hash", "case_count", "token_count"});
  const auto write = [&](const char* name, const char* domain,
                         const std::string& hash, const auto& cases) {
    std::uint64_t tokens = 0;
    for (const auto& item : cases) tokens += item.targets.size();
    row(output, {name, domain, hash, text(cases.size()), text(tokens)});
  };
  write(cm::kCalibrationName, cm::generatorDomain(cm::Partition::CALIBRATION),
        cm::kCalibrationHash, cm::buildMarginCalibrationV1());
  write(cm::kDevelopmentName, cm::generatorDomain(cm::Partition::DEVELOPMENT),
        cm::kDevelopmentHash, cm::buildMarginDevelopmentV1());
  for (const auto partition : ar::kPartitions)
    write(ar::partitionName(partition), ar::domain(partition),
          ar::partitionHash(partition, 8), ar::cases(partition, 8));
}

void writeDatasetOverlap(std::ofstream& output) {
  row(output, {"left_partition", "right_partition", "case_id_overlap",
               "initial_prefix_overlap", "full_sequence_overlap",
               "unique_transition_overlap",
               "transition_occurrence_multiset_overlap"});
  const auto cal = cm::buildMarginCalibrationV1();
  const auto dev = cm::buildMarginDevelopmentV1();
  const auto overlap = [&](const std::vector<cm::DatasetCase>& left,
                           const std::vector<cm::DatasetCase>& right) {
    const auto stats = cm::overlap(left, false, right, false);
    return std::vector<std::string>{text(stats.caseId),
                                    text(stats.initialPrefix),
                                    text(stats.fullSequence),
                                    text(stats.uniqueTransitions),
                                    text(stats.transitionOccurrences)};
  };
  auto fields = overlap(cal, dev);
  fields.insert(fields.begin(), cm::kDevelopmentName);
  fields.insert(fields.begin(), cm::kCalibrationName);
  row(output, fields);
  for (const auto partition : ar::kPartitions) {
    const auto arCases = ar::cases(partition, 8);
    auto f1 = overlap(cal, arCases);
    f1.insert(f1.begin(), ar::partitionName(partition));
    f1.insert(f1.begin(), cm::kCalibrationName);
    row(output, f1);
    auto f2 = overlap(dev, arCases);
    f2.insert(f2.begin(), ar::partitionName(partition));
    f2.insert(f2.begin(), cm::kDevelopmentName);
    row(output, f2);
  }
}

void writeDatasetHashes(std::ofstream& output) {
  row(output, {"partition", "hash"});
  row(output, {cm::kCalibrationName, cm::kCalibrationHash});
  row(output, {cm::kDevelopmentName, cm::kDevelopmentHash});
  for (const auto partition : ar::kPartitions)
    row(output, {ar::partitionName(partition), ar::partitionHash(partition, 8)});
}

// ---------------------------------------------------------------------------
// Main run
// ---------------------------------------------------------------------------
struct RunResult {
  const RunSpec* spec = nullptr;
  dq::AutoregressiveSelectedRun run;
  std::vector<cm::CheckpointMetrics> calibration;
  std::vector<cm::CheckpointMetrics> development;
  ar::Metrics selectedDevelopment;
  ar::Metrics finalDevelopment;
  GradientRun gradients;
  bool anchorsMatch = false;
};

std::vector<RunResult> runAll() {
  std::vector<RunResult> results;
  const auto calibrationCases = cm::buildMarginCalibrationV1();
  const auto developmentCases = cm::buildMarginDevelopmentV1();
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    RunResult result;
    result.spec = &spec;
    std::cerr << "[" << spec.publicId << "] regenerating CPU trajectory..."
              << std::endl;
    result.run = dq::runAutoregressiveSelectedCpu(
        config, spec.seed, kSteps,
        dq::AutoregressiveSelectionMode::BEST_AR_VALIDATION_V1);
    if (result.run.selectedStep != spec.pinnedSelectedStep) {
      throw std::runtime_error(
          std::string("CANONICAL_ANCHOR_MISMATCH: ") + spec.publicId +
          " CPU regeneration selected step " +
          std::to_string(result.run.selectedStep) + " but pinned step is " +
          std::to_string(spec.pinnedSelectedStep));
    }
    std::cerr << "[" << spec.publicId << "] evaluating margin partitions..."
              << std::endl;
    for (const int step : cm::kEvaluationSteps) {
      const auto& params = result.run.training.checkpoints.at(step);
      result.calibration.push_back(
          evaluatePartition(config, params, step, calibrationCases));
      result.development.push_back(
          evaluatePartition(config, params, step, developmentCases));
    }
    std::cerr << "[" << spec.publicId << "] verifying AR anchors..."
              << std::endl;
    result.selectedDevelopment = dq::autoregressiveEvaluation(
        config, result.run.training.checkpoints.at(result.run.selectedStep),
        ar::Partition::DEVELOPMENT);
    result.finalDevelopment = dq::autoregressiveEvaluation(
        config, result.run.training.checkpoints.at(kSteps),
        ar::Partition::DEVELOPMENT);
    result.anchorsMatch =
        result.selectedDevelopment.tokenExact == spec.selectedTokenExact &&
        result.finalDevelopment.tokenExact == spec.finalTokenExact &&
        result.selectedDevelopment.sequenceExact ==
            spec.selectedSequenceExact &&
        result.finalDevelopment.sequenceExact == spec.finalSequenceExact;
    if (!result.anchorsMatch) {
      throw std::runtime_error(
          std::string("CANONICAL_ANCHOR_MISMATCH: ") + spec.publicId +
          " AR_DEVELOPMENT_V3 counts differ from the canonical bundle "
          "(selected token/sequence " +
          std::to_string(result.selectedDevelopment.tokenExact) + "/" +
          std::to_string(result.selectedDevelopment.sequenceExact) +
          " vs pinned " + std::to_string(spec.selectedTokenExact) + "/" +
          std::to_string(spec.selectedSequenceExact) + "; final token/sequence " +
          std::to_string(result.finalDevelopment.tokenExact) + "/" +
          std::to_string(result.finalDevelopment.sequenceExact) + " vs pinned " +
          std::to_string(spec.finalTokenExact) + "/" +
          std::to_string(spec.finalSequenceExact) + ")");
    }
    std::cerr << "[" << spec.publicId
              << "] gradient attribution (320 steps)..." << std::endl;
    result.gradients =
        gradientAttribution(config, spec.seed, result.run.training);
    for (const auto& step : result.gradients.steps) {
      if (!step.lossParity || !step.gradientParity) {
        throw std::runtime_error(
            std::string("GRADIENT_PARITY_MISMATCH at step ") +
            std::to_string(step.step) + " for " + spec.publicId);
      }
    }
    results.push_back(std::move(result));
  }
  return results;
}

void writeConfiguration(std::ofstream& output,
                        const std::vector<RunResult>& results) {
  row(output, {"source", "configuration_id", "depth", "seed", "steps",
               "evaluation_step_count", "calibration_partition",
               "development_partition", "calibration_hash",
               "development_hash", "pinned_selected_step",
               "selected_step_matches_pinned"});
  for (const auto& result : results) {
    const auto& spec = *result.spec;
    row(output, {"CPU_REFERENCE_REGENERATION", spec.publicId,
                 text(spec.layers), text(spec.seed), text(kSteps),
                 text(cm::kEvaluationSteps.size()), cm::kCalibrationName,
                 cm::kDevelopmentName, cm::kCalibrationHash,
                 cm::kDevelopmentHash, text(spec.pinnedSelectedStep),
                 result.run.selectedStep == spec.pinnedSelectedStep
                     ? "true"
                     : "false"});
  }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
struct VariantSelection {
  const cm::ObjectiveSpec* spec = nullptr;
  cm::SelectionResult selection;
  const cm::CheckpointMetrics* selectedDevelopment = nullptr;
  const cm::CheckpointMetrics* finalDevelopment = nullptr;
  int selectedDevIndex = -1;
};

struct ConfigAnalysis {
  const RunResult* run = nullptr;
  std::array<VariantSelection, cm::kObjectives.size()> variants;
};

std::vector<ConfigAnalysis> analyze(const std::vector<RunResult>& results) {
  std::vector<ConfigAnalysis> analysis;
  for (const auto& result : results) {
    ConfigAnalysis item;
    item.run = &result;
    for (std::size_t v = 0; v < cm::kObjectives.size(); ++v) {
      auto& variant = item.variants[v];
      variant.spec = &cm::kObjectives[v];
      variant.selection =
          cm::selectBestCheckpoint(cm::kObjectives[v], result.calibration);
      if (!variant.selection.selected) continue;
      variant.selectedDevIndex = variant.selection.trajectoryIndex;
      variant.selectedDevelopment =
          &result.development[static_cast<std::size_t>(variant.selectedDevIndex)];
      variant.finalDevelopment = &result.development.back();
    }
    analysis.push_back(std::move(item));
  }
  return analysis;
}

void writeCheckpointMetrics(
    std::ofstream& output, const std::vector<RunResult>& results) {
  row(output, {"configuration_id", "partition", "step", "token_exact",
               "token_total", "sequence_exact", "sequence_total",
               "autoregressive_nll", "median_first_error_survival",
               "lower_tail_margin_q10", "all_finite"});
  for (const auto& result : results) {
    const auto write = [&](const char* partition,
                           const std::vector<cm::CheckpointMetrics>& metrics) {
      for (const auto& item : metrics) {
        row(output, {result.spec->publicId, partition, text(item.step),
                     text(item.tokenExact), text(item.tokenTotal),
                     text(item.sequenceExact), text(item.sequenceTotal),
                     number(item.autoregressiveNll),
                     number(item.medianFirstErrorSurvival),
                     number(item.lowerTailMarginQ10),
                     item.allFinite ? "true" : "false"});
      }
    };
    write(cm::kCalibrationName, result.calibration);
    write(cm::kDevelopmentName, result.development);
  }
}

void writeObjectiveScores(std::ofstream& output,
                          const std::vector<RunResult>& results) {
  row(output, {"configuration_id", "objective", "priority", "step", "score",
               "finite"});
  for (const auto& result : results)
    for (const auto& spec : cm::kObjectives)
      for (const auto& metrics : result.calibration) {
        const auto score = cm::scoreObjective(spec, metrics);
        row(output, {result.spec->publicId, spec.id, text(spec.priority),
                     text(metrics.step),
                     score.finite ? number(score.value) : "NOT_FINITE",
                     score.finite ? "true" : "false"});
      }
}

void writeObjectiveCorrelations(std::ofstream& output,
                                const std::vector<ConfigAnalysis>& analysis) {
  row(output, {"configuration_id", "objective",
               "spearman_objective_vs_development_token_exact",
               "spearman_objective_vs_development_sequence_exact",
               "spearman_objective_vs_development_first_error_survival",
               "spearman_objective_vs_calibration_token_exact",
               "selection_regret_token_exact",
               "selection_regret_sequence_exact", "near_tie_step_count",
               "delta_token_exact_vs_final", "delta_sequence_exact_vs_final"});
  for (const auto& item : analysis) {
    const auto& run = *item.run;
    std::vector<double> devTokens, devSequences, devSurvival, calibTokens;
    for (std::size_t i = 0; i < run.development.size(); ++i) {
      devTokens.push_back(static_cast<double>(run.development[i].tokenExact));
      devSequences.push_back(
          static_cast<double>(run.development[i].sequenceExact));
      devSurvival.push_back(run.development[i].medianFirstErrorSurvival);
      calibTokens.push_back(
          static_cast<double>(run.calibration[i].tokenExact));
    }
    std::uint64_t bestDevTokens = 0, bestDevSequences = 0;
    for (const auto& m : run.development) {
      bestDevTokens = std::max(bestDevTokens, m.tokenExact);
      bestDevSequences = std::max(bestDevSequences, m.sequenceExact);
    }
    for (const auto& variant : item.variants) {
      if (!variant.selection.selected || !variant.selectedDevelopment) continue;
      std::vector<double> objectiveValues;
      for (const auto& m : run.calibration) {
        const auto score = cm::scoreObjective(*variant.spec, m);
        objectiveValues.push_back(score.finite ? score.value
                                               : std::numeric_limits<double>::quiet_NaN());
      }
      const auto spearmanTokens =
          cm::spearman(objectiveValues, devTokens);
      const auto spearmanSequences =
          cm::spearman(objectiveValues, devSequences);
      const auto spearmanSurvival =
          cm::spearman(objectiveValues, devSurvival);
      const auto spearmanCalibration =
          cm::spearman(objectiveValues, calibTokens);
      const auto minObjective =
          *std::min_element(objectiveValues.begin(), objectiveValues.end());
      std::size_t nearTies = 0;
      for (const double value : objectiveValues)
        if (std::isfinite(value) &&
            std::abs(value - minObjective) <= cm::kScoreTieTolerance)
          ++nearTies;
      const std::uint64_t selectedTokens =
          variant.selectedDevelopment->tokenExact;
      const std::uint64_t selectedSequences =
          variant.selectedDevelopment->sequenceExact;
      row(output, {run.spec->publicId, variant.spec->id,
                   number(spearmanTokens.available ? spearmanTokens.value : 0.0),
                   number(spearmanSequences.available ? spearmanSequences.value : 0.0),
                   number(spearmanSurvival.available ? spearmanSurvival.value : 0.0),
                   number(spearmanCalibration.available ? spearmanCalibration.value : 0.0),
                   text(static_cast<std::int64_t>(bestDevTokens) -
                        static_cast<std::int64_t>(selectedTokens)),
                   text(static_cast<std::int64_t>(bestDevSequences) -
                        static_cast<std::int64_t>(selectedSequences)),
                   text(nearTies),
                   text(static_cast<std::int64_t>(selectedTokens) -
                        static_cast<std::int64_t>(run.development.back().tokenExact)),
                   text(static_cast<std::int64_t>(selectedSequences) -
                        static_cast<std::int64_t>(run.development.back().sequenceExact))});
    }
  }
}

void writeCheckpointSelection(std::ofstream& output,
                              const std::vector<ConfigAnalysis>& analysis) {
  row(output, {"configuration_id", "objective", "priority", "selected_step",
               "score", "calibration_token_exact", "calibration_sequence_exact",
               "development_token_exact", "development_sequence_exact",
               "development_nll", "development_median_survival",
               "development_lower_tail_margin_q10",
               "delta_token_exact_vs_final", "delta_sequence_exact_vs_final"});
  for (const auto& item : analysis)
    for (const auto& variant : item.variants) {
      if (!variant.selection.selected || !variant.selectedDevelopment) continue;
      const auto& dev = *variant.selectedDevelopment;
      const auto& final = *variant.finalDevelopment;
      row(output, {item.run->spec->publicId, variant.spec->id,
                   text(variant.spec->priority), text(variant.selection.step),
                   variant.selection.score.finite
                       ? number(variant.selection.score.value)
                       : "NOT_FINITE",
                   text(variant.selection.metrics.tokenExact),
                   text(variant.selection.metrics.sequenceExact),
                   text(dev.tokenExact), text(dev.sequenceExact),
                   number(dev.autoregressiveNll),
                   number(dev.medianFirstErrorSurvival),
                   number(dev.lowerTailMarginQ10),
                   text(static_cast<std::int64_t>(dev.tokenExact) -
                        static_cast<std::int64_t>(final.tokenExact)),
                   text(static_cast<std::int64_t>(dev.sequenceExact) -
                        static_cast<std::int64_t>(final.sequenceExact))});
    }
}

std::vector<cm::LeaveOneSeedOutFold> runLoso(
    const std::vector<RunResult>& results) {
  std::map<std::uint32_t, std::vector<cm::CheckpointMetrics>> calibrationBySeed,
      developmentBySeed;
  for (const auto& result : results) {
    if (result.spec->layers != 19) continue;
    calibrationBySeed[result.spec->seed] = result.calibration;
    developmentBySeed[result.spec->seed] = result.development;
  }
  return cm::runLeaveOneSeedOut({1, 2, 4}, calibrationBySeed,
                                developmentBySeed);
}

void writeLeaveOneSeedOut(std::ofstream& output,
                          const std::vector<cm::LeaveOneSeedOutFold>& folds) {
  row(output, {"held_out_seed", "chosen_objective", "chosen_parameter",
               "chosen_priority", "selected_step", "token_exact",
               "token_total", "sequence_exact", "sequence_total",
               "delta_token_exact_vs_final", "delta_sequence_exact_vs_final",
               "collapse_free", "finite"});
  for (const auto& fold : folds)
    row(output, {text(fold.heldOutSeed), fold.chosenObjective,
                 number(fold.chosenParameter), text(fold.chosenPriority),
                 text(fold.selectedStep), text(fold.tokenExact),
                 text(fold.tokenTotal), text(fold.sequenceExact),
                 text(fold.sequenceTotal),
                 text(fold.finalStepTokenExactDelta),
                 text(fold.finalStepSequenceExactDelta),
                 fold.collapseFree ? "true" : "false",
                 fold.finite ? "true" : "false"});
}

cm::DevelopmentRecord buildRecord(
    std::uint32_t seed, const VariantSelection& variant,
    const std::vector<cm::CheckpointMetrics>& developmentTrajectory) {
  cm::DevelopmentRecord record;
  record.seed = seed;
  if (!variant.selection.selected || !variant.selectedDevelopment ||
      !variant.finalDevelopment)
    return record;
  record.candidate = *variant.selectedDevelopment;
  record.finalStep = *variant.finalDevelopment;
  const std::size_t index = static_cast<std::size_t>(variant.selectedDevIndex);
  if (index < developmentTrajectory.size()) {
    if (index > 0)
      record.adjacentCheckpoints.push_back(developmentTrajectory[index - 1]);
    if (index + 1 < developmentTrajectory.size())
      record.adjacentCheckpoints.push_back(developmentTrajectory[index + 1]);
  }
  return record;
}

struct GateRow {
  cm::DevelopmentGateResult gate;
  bool losoCollapseFree = false;
  double losoMeanTokenDelta = 0.0;
  bool pass = false;
};

std::map<std::string, GateRow> developmentGates(
    const std::vector<ConfigAnalysis>& analysis, const RunResult& control,
    const std::vector<cm::LeaveOneSeedOutFold>& folds) {
  std::map<std::string, GateRow> result;
  const auto& controlAnalysis =
      *std::find_if(analysis.begin(), analysis.end(),
                    [&](const ConfigAnalysis& item) {
                      return item.run->spec == control.spec;
                    });
  bool losoCollapseFree = true;
  double losoDeltaSum = 0.0;
  std::size_t losoCount = 0;
  for (const auto& fold : folds) {
    losoCollapseFree = losoCollapseFree && fold.finite && fold.collapseFree;
    losoDeltaSum += static_cast<double>(fold.finalStepTokenExactDelta);
    ++losoCount;
  }
  for (std::size_t v = 0; v < cm::kObjectives.size(); ++v) {
    const auto& spec = cm::kObjectives[v];
    std::array<cm::DevelopmentRecord, 3> l19{};
    for (const auto& item : analysis) {
      if (item.run->spec->layers != 19) continue;
      const std::uint32_t seed = item.run->spec->seed;
      const auto& variant = item.variants[v];
      if (seed == 1) l19[0] = buildRecord(seed, variant, item.run->development);
      if (seed == 2) l19[1] = buildRecord(seed, variant, item.run->development);
      if (seed == 4) l19[2] = buildRecord(seed, variant, item.run->development);
    }
    const auto& controlVariant = controlAnalysis.variants[v];
    const auto controlRecord =
        buildRecord(control.spec->seed, controlVariant,
                    controlAnalysis.run->development);
    GateRow gateRow;
    gateRow.gate = cm::developmentGate(l19, controlRecord);
    gateRow.losoCollapseFree = losoCollapseFree;
    gateRow.losoMeanTokenDelta =
        losoCount > 0 ? losoDeltaSum / static_cast<double>(losoCount) : 0.0;
    gateRow.pass = gateRow.gate.pass && gateRow.losoCollapseFree;
    result[spec.id] = gateRow;
  }
  return result;
}

void writeDevelopmentGate(std::ofstream& output,
                          const std::map<std::string, GateRow>& gates) {
  row(output, {"objective", "priority", "finite", "seed2_strict",
               "pooled_token_nonworse", "pooled_sequence_nonworse",
               "control_nonworse", "first_error_median_nonworse",
               "supported_seeds", "stable_supported_seeds",
               "no_case_collapse", "gate_pass", "loso_collapse_free",
               "loso_mean_token_delta", "pass"});
  for (const auto& spec : cm::kObjectives) {
    const auto& item = gates.at(spec.id);
    const auto& gate = item.gate;
    row(output, {spec.id, text(spec.priority),
                 gate.finite ? "true" : "false",
                 gate.seed2Strict ? "true" : "false",
                 gate.pooledTokenNonworse ? "true" : "false",
                 gate.pooledSequenceNonworse ? "true" : "false",
                 gate.controlNonworse ? "true" : "false",
                 gate.firstErrorMedianNonworse ? "true" : "false",
                 text(gate.supportedSeeds), text(gate.stableSupportedSeeds),
                 gate.noCaseCollapse ? "true" : "false",
                 gate.pass ? "true" : "false",
                 item.losoCollapseFree ? "true" : "false",
                 number(item.losoMeanTokenDelta),
                 item.pass ? "true" : "false"});
  }
}

// Pooled variant evidence across the three L19 seeds (the preregistered
// freeze rule of the goal protocol).
cm::VariantEvidence pooledEvidence(
    const cm::ObjectiveSpec& spec,
    const std::vector<ConfigAnalysis>& analysis) {
  cm::VariantEvidence item;
  item.objectiveId = spec.id;
  item.priority = spec.priority;
  std::uint64_t worst = std::numeric_limits<std::uint64_t>::max();
  std::vector<double> survival, tail;
  bool seen = false;
  int minStep = 0;
  for (const auto& config : analysis) {
    if (config.run->spec->layers != 19) continue;
    const auto* specPtr = &spec;
    const auto found = std::find_if(
        config.variants.begin(), config.variants.end(),
        [&](const VariantSelection& variant) {
          return variant.spec == specPtr;
        });
    if (found == config.variants.end()) continue;
    const auto& variant = *found;
    if (!variant.selection.selected || !variant.selectedDevelopment) continue;
    const auto& dev = *variant.selectedDevelopment;
    if (!dev.allFinite) continue;
    seen = true;
    item.sequenceExact += dev.sequenceExact;
    item.tokenExact += dev.tokenExact;
    worst = std::min(worst, dev.tokenExact);
    for (const auto& trace : dev.cases)
      survival.push_back(trace.firstErrorPosition < 0
                             ? static_cast<double>(trace.margins.size() + 1)
                             : static_cast<double>(trace.firstErrorPosition));
    for (const auto& trace : dev.cases)
      tail.insert(tail.end(), trace.margins.begin(), trace.margins.end());
    if (!seen || minStep == 0 || variant.selection.step < minStep)
      minStep = variant.selection.step;
  }
  if (!seen) return item;
  item.worstSeedTokenExact =
      worst == std::numeric_limits<std::uint64_t>::max() ? 0 : worst;
  item.firstErrorMedian = cm::lowerMiddleMedian(std::move(survival));
  item.lowerTailMargin = cm::lowerTailMean(std::move(tail), 0.10);
  item.selectedStep = minStep;
  return item;
}

std::string bestVariant(const std::vector<cm::VariantEvidence>& evidence) {
  if (evidence.empty()) return "NONE";
  const cm::VariantEvidence* best = &evidence.front();
  for (const auto& item : evidence)
    if (cm::betterVariantEvidence(item, *best)) best = &item;
  return best->objectiveId;
}

Attribution classifyAttribution(
    const std::vector<RunResult>& results,
    const std::vector<ConfigAnalysis>& analysis) {
  Attribution attribution;
  double criticalShareSum = 0.0, criticalLossSum = 0.0;
  std::size_t count = 0;
  double depthRatioSum = 0.0;
  std::size_t depthCount = 0;
  std::size_t rankDriftConfigs = 0;
  const auto* seed2 = static_cast<const RunResult*>(nullptr);
  std::array<const RunResult*, 2> others{};
  std::size_t otherCount = 0;
  for (const auto& result : results) {
    for (const auto& step : result.gradients.steps) {
      criticalShareSum += step.criticalTokenShare;
      criticalLossSum += step.criticalLossShare;
      ++count;
      if (step.step > 256) {
        depthRatioSum += step.gradNormDepthRatio;
        ++depthCount;
      }
    }
    if (result.spec->layers == 19 && result.spec->seed == 2) seed2 = &result;
    else if (result.spec->layers == 19 && otherCount < 2)
      others[otherCount++] = &result;
  }
  const double criticalTokenMean =
      count > 0 ? criticalShareSum / static_cast<double>(count) : 0.0;
  const double criticalLossMean =
      count > 0 ? criticalLossSum / static_cast<double>(count) : 0.0;
  attribution.criticalUnderweight =
      criticalTokenMean > kCriticalTokenShareFloor &&
      (criticalLossMean - criticalTokenMean) < kCriticalShareShortfall;
  const double depthMean =
      depthCount > 0 ? depthRatioSum / static_cast<double>(depthCount) : 0.0;
  attribution.depthAmplification = depthMean > kDepthRatioThreshold;
  for (const auto& config : analysis) {
    const auto& result = *config.run;
    const bool trainOverfit =
        result.gradients.finalTrainMeanRank < kTrainRankOverfitThreshold;
    const auto& finalDev = result.development.back();
    const auto selectedIdx =
        static_cast<std::size_t>(
            std::find_if(result.development.begin(), result.development.end(),
                         [&](const cm::CheckpointMetrics& m) {
                           return m.step == result.run.selectedStep;
                         }) -
            result.development.begin());
    const bool freshRankingWorse =
        selectedIdx < result.development.size() &&
        finalDev.lowerTailMarginQ10 <
            result.development[selectedIdx].lowerTailMarginQ10;
    if (trainOverfit && freshRankingWorse) ++rankDriftConfigs;
  }
  attribution.outputRankDrift = rankDriftConfigs >= 3;
  if (seed2 && otherCount == 2) {
    std::size_t imbalancedSteps = 0;
    for (std::size_t i = 0; i < seed2->run.training.steps.size(); ++i) {
      const double otherMedian =
          (others[0]->run.training.steps[i].updateNorm +
           others[1]->run.training.steps[i].updateNorm) /
          2.0;
      if (seed2->run.training.steps[i].updateNorm >
          kUpdateImbalanceFactor * std::max(1e-12, otherMedian))
        ++imbalancedSteps;
    }
    const double fraction =
        static_cast<double>(imbalancedSteps) /
        static_cast<double>(seed2->run.training.steps.size());
    attribution.seedImbalance = fraction > kUpdateImbalanceFraction;
  }
  return attribution;
}

void writeGradientAttribution(std::ofstream& output,
                              const std::vector<RunResult>& results) {
  row(output, {"configuration_id", "step", "loss", "accuracy",
               "mean_target_margin", "mean_target_rank", "mean_target_nll",
               "critical_token_share", "critical_loss_share",
               "easy_token_share", "grad_norm_total", "grad_norm_embedding",
               "grad_norm_output_projection", "grad_norm_layer_mean",
               "grad_norm_first_layer", "grad_norm_last_layer",
               "grad_norm_attn_share", "grad_norm_ffn_share",
               "grad_norm_norm_share", "grad_norm_depth_ratio",
               "grad_norm_output_share", "loss_parity", "gradient_parity"});
  for (const auto& result : results)
    for (const auto& step : result.gradients.steps)
      row(output, {result.spec->publicId, text(step.step),
                   number(step.loss), number(step.accuracy),
                   number(step.meanMargin), number(step.meanRank),
                   number(step.meanNll), number(step.criticalTokenShare),
                   number(step.criticalLossShare), number(step.easyTokenShare),
                   number(step.gradNormTotal),
                   number(step.gradNormEmbedding),
                   number(step.gradNormOutput),
                   number(step.gradNormLayerMean),
                   number(step.gradNormFirstLayer),
                   number(step.gradNormLastLayer),
                   number(step.gradNormAttnShare),
                   number(step.gradNormFfnShare),
                   number(step.gradNormNormShare),
                   number(step.gradNormDepthRatio),
                   number(step.gradNormOutputShare),
                   step.lossParity ? "true" : "false",
                   step.gradientParity ? "true" : "false"});
}

// ---------------------------------------------------------------------------
// Training families (margin-aware CE, host-only; preregistered hyper-
// parameters). The families are implemented on top of the verbatim
// CPU-reference arithmetic in critical_margin_training_lib.h; every training
// run asserts bitwise backward parity against tiny_lm::forwardBackward on a
// parity cadence and the whole loop is validated end-to-end against
// runAutoregressiveSelectedCpu with lambda=0 in the self-test.
// ---------------------------------------------------------------------------

struct FamilySpec {
  const char* id;
  train::MarginLossSpec::Family family;
  float delta;  // pairwise margin floor (logit units)
  float tau;    // sequence-worst temperature
  float lambda;
  const char* parameterDescription;
};

// Preregistered hyperparameters (fixed, no grid): modest margin floor and
// worst-margin emphasis at the scale of the fresh-partition margins.
const std::array<FamilySpec, 2> kFamilies{{
    {"PAIRWISE_MARGIN_CE_V1",
     train::MarginLossSpec::Family::PairwiseMarginCe, 0.5f, 0.0f, 1.0f,
     "delta=0.5,lambda=1.0"},
    {"SEQUENCE_WORST_MARGIN_CE_V1",
     train::MarginLossSpec::Family::SequenceWorstMarginCe, 0.0f, 0.5f, 1.0f,
     "tau=0.5,lambda=1.0"},
}};

struct BaselineMetrics {
  bool present = false;
  std::uint64_t arDevTokenExact = 0;
  std::uint64_t arDevSequenceExact = 0;
  double arDevNll = 0.0;
  double marginDevLtm = 0.0;
};

std::map<std::string, BaselineMetrics> readBaseline(const fs::path& dir) {
  std::map<std::string, BaselineMetrics> result;
  std::ifstream file(dir / "baseline-summary.csv");
  if (!file)
    throw std::runtime_error("missing baseline-summary.csv in " +
                             dir.string());
  const auto strip = [](const std::string& s) {
    return s.size() >= 2 ? s.substr(1, s.size() - 2) : s;
  };
  std::string line;
  std::getline(file, line);  // header
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) fields.push_back(field);
    if (fields.size() < 7) continue;
    BaselineMetrics m;
    m.present = true;
    m.arDevTokenExact = std::stoull(strip(fields[1]));
    m.arDevSequenceExact = std::stoull(strip(fields[2]));
    m.arDevNll = std::stod(strip(fields[3]));
    m.marginDevLtm = std::stod(strip(fields[4]));
    result[strip(fields[0])] = m;
  }
  return result;
}

void writeBaselineSummary(std::ofstream& output,
                          const std::vector<RunResult>& results) {
  row(output, {"configuration_id", "final_ar_dev_token_exact",
               "final_ar_dev_sequence_exact", "final_ar_dev_nll",
               "final_margin_dev_lower_tail_margin_q10",
               "final_margin_dev_median_first_error_survival",
               "final_margin_calib_lower_tail_margin_q10", "finite"});
  for (const auto& result : results) {
    const auto& dev = result.development.back();
    const auto& cal = result.calibration.back();
    row(output, {result.spec->publicId,
                 text(result.finalDevelopment.tokenExact),
                 text(result.finalDevelopment.sequenceExact),
                 number(result.finalDevelopment.autoregressiveNll),
                 number(dev.lowerTailMarginQ10),
                 number(dev.medianFirstErrorSurvival),
                 number(cal.lowerTailMarginQ10),
                 dev.allFinite && result.finalDevelopment.allFinite
                     ? "true"
                     : "false"});
  }
}

// Host-only self-test for the margin-aware training path: (1) bitwise
// backward parity against tiny_lm::forwardBackward with the CE dLogits,
// (2) exact check of the augmented dLogits against the margin formula,
// (3) finite-difference spot check of the margin-aware total loss, and
// (4) an end-to-end lambda==0 run that must reproduce the canonical
// 64-step trajectory of runAutoregressiveSelectedCpu bitwise.
void selfTestTraining() {
  tiny::Config c;
  c.tokens = 3;
  c.vocabularySize = 32;  // formalBatch uses token ids up to 12
  c.dimension = 4;
  c.feedForwardDimension = 8;
  c.numLayers = 2;
  c.numHeads = 2;
  std::string error;
  if (!tiny::validateConfig(c, &error))
    throw std::runtime_error("invalid tiny LM config: " + error);
  const auto x = tiny::oneHot({0, 1, 2}, 32);
  const auto y = tiny::oneHot({1, 2, 3}, 32);
  const auto params = tiny::initialParameters(c, 7);

  // (1) backward-copy parity (multi-head branch, lambda=0)
  train::MarginLossSpec zeroSpec;
  zeroSpec.lambda = 0.0f;
  const std::vector<int> noSteps;
  const auto parityRun = train::runMarginTraining(c, 7, zeroSpec, 8, noSteps,
                                                  1, nullptr, nullptr);
  if (parityRun.lastParityStep != 8)
    throw std::runtime_error("backward parity failed in self-test");

  // (2) augmented dLogits matches the pairwise margin formula
  train::MarginLossSpec pairwise;
  pairwise.family = train::MarginLossSpec::Family::PairwiseMarginCe;
  pairwise.delta = 0.5f;
  pairwise.lambda = 1.0f;
  const auto forward = train::marginForwardBackward(c, x, y, params, pairwise);
  for (std::uint32_t row = 0; row < c.tokens; ++row) {
    const std::size_t b = std::size_t(row) * c.vocabularySize;
    std::uint32_t truth = 0;
    for (std::uint32_t j = 0; j < c.vocabularySize; ++j)
      if (y[b + j] > .5f) truth = j;
    float best = -std::numeric_limits<float>::infinity();
    std::uint32_t bestIdx = 0;
    for (std::uint32_t j = 0; j < c.vocabularySize; ++j)
      if (j != truth && forward.forward.logits[b + j] > best) {
        best = forward.forward.logits[b + j];
        bestIdx = j;
      }
    const float margin = forward.forward.logits[b + truth] - best;
    const float expectedCe =
        (forward.forward.prob[b + truth] - 1.0f) / float(c.tokens);
    const float expectedDelta =
        margin < pairwise.delta ? -pairwise.lambda / float(c.tokens) : 0.0f;
    const float actualDelta =
        forward.dLogits[b + truth] - expectedCe;
    if (std::abs(actualDelta - expectedDelta) > 1e-6f)
      throw std::runtime_error("pairwise dLogits mismatch in self-test");
    if (bestIdx != truth &&
        std::abs((forward.dLogits[b + bestIdx] -
                  (forward.forward.prob[b + bestIdx] - 0.0f) /
                      float(c.tokens)) -
                 (margin < pairwise.delta
                      ? pairwise.lambda / float(c.tokens)
                      : 0.0f)) > 1e-6f)
      throw std::runtime_error("pairwise competitor dLogits mismatch");
  }

  // (3) finite-difference spot check of the margin-aware total loss
  train::MarginLossSpec sequence;
  sequence.family = train::MarginLossSpec::Family::SequenceWorstMarginCe;
  sequence.tau = 0.5f;
  sequence.lambda = 1.0f;
  constexpr float kEps = 1e-3f;
  const auto reference = train::marginForwardBackward(c, x, y, params, sequence);
  const auto perturb = [&](const std::string& name, std::size_t index,
                           float delta) {
    auto copy = params;
    std::vector<float>* target = nullptr;
    for (const auto& e : tiny::parameterRegistry(copy))
      if (e.name == name) target = const_cast<std::vector<float>*>(e.values);
    if (!target) throw std::runtime_error("parameter group not found");
    (*target)[index] += delta;
    return copy;
  };
  for (const auto& e : tiny::parameterRegistry(params)) {
    if (e.name != "output_projection" && e.name != "layer_000.wq") continue;
    const std::size_t index = e.values->size() / 2;
    const auto plus = train::marginForwardBackward(
        c, x, y, perturb(e.name, index, kEps), sequence);
    const auto minus = train::marginForwardBackward(
        c, x, y, perturb(e.name, index, -kEps), sequence);
    const double numeric =
        (plus.summary.total - minus.summary.total) / (2.0 * double(kEps));
    double analyticValue = 0.0;
    for (const auto& g : tiny::parameterRegistry(reference.gradients))
      if (g.name == e.name) {
        analyticValue = double((*g.values)[index]);
        break;
      }
    const double tolerance =
        2e-2 * std::max(1.0, std::abs(analyticValue));
    if (std::abs(numeric - analyticValue) > tolerance)
      throw std::runtime_error("finite-difference mismatch for " +
                               std::string(e.name) + ": numeric=" +
                               std::to_string(numeric) + " analytic=" +
                               std::to_string(analyticValue));
  }

  // (4) end-to-end lambda==0 vs the canonical 64-step trajectory
  const auto config = makeConfig(19);
  const auto canonical = dq::runAutoregressiveSelectedCpu(
      config, 2, 64, dq::AutoregressiveSelectionMode::BEST_AR_VALIDATION_V1);
  const auto mine =
      train::runMarginTraining(config, 2, zeroSpec, 64, noSteps, 4, nullptr,
                               nullptr);
  if (canonical.training.steps.size() != mine.stepSummaries.size())
    throw std::runtime_error("trajectory length mismatch in self-test");
  for (std::size_t i = 0; i < mine.stepSummaries.size(); ++i) {
    const float myLoss = float(mine.stepSummaries[i].nll);
    if (myLoss != canonical.training.steps[i].loss)
      throw std::runtime_error("lambda=0 loss parity failed at step " +
                               std::to_string(i + 1));
  }
  const auto myFinal = dq::autoregressiveEvaluation(
      config, mine.finalParameters, ar::Partition::VALIDATION);
  if (myFinal.tokenExact != canonical.finalStepValidation.tokenExact ||
      myFinal.sequenceExact != canonical.finalStepValidation.sequenceExact)
    throw std::runtime_error("lambda=0 final checkpoint mismatch");
}

struct TrainRun {
  const FamilySpec* family = nullptr;
  const RunSpec* spec = nullptr;
  train::MarginTrainingRun training;
  ar::Metrics finalDevelopment;      // AR dev at `steps`
  ar::Metrics stabilityDevelopment;  // AR dev at steps-16
  std::vector<cm::CheckpointMetrics> calibration;
  std::vector<cm::CheckpointMetrics> development;
};

struct TrainContext {
  const tiny::Config* config = nullptr;
  std::vector<cm::DatasetCase> calibrationCases;
  std::vector<cm::DatasetCase> developmentCases;
  TrainRun* run = nullptr;
};

void trainCheckpoint(int step, const dq::Params& params,
                     void* opaque) {
  TrainContext* ctx = static_cast<TrainContext*>(opaque);
  ctx->run->calibration.push_back(
      evaluatePartition(*ctx->config, params, step, ctx->calibrationCases));
  ctx->run->development.push_back(
      evaluatePartition(*ctx->config, params, step, ctx->developmentCases));
}

std::vector<TrainRun> runTrainingFamilies(bool micro) {
  const int steps = micro ? 64 : 320;
  const auto calibrationCases = cm::buildMarginCalibrationV1();
  const auto developmentCases = cm::buildMarginDevelopmentV1();
  std::vector<int> evaluationSteps;
  for (const int step : cm::kEvaluationSteps)
    if (step <= steps) evaluationSteps.push_back(step);
  std::vector<TrainRun> runs;
  for (const auto& family : kFamilies) {
    const std::size_t specCount = micro ? 2 : 4;
    for (std::size_t s = 0; s < specCount; ++s) {
      const RunSpec& spec = kRuns[s];
      std::cerr << "[" << spec.publicId << "] training " << family.id
                << (micro ? " (micro " : " (") << steps << " steps)..."
                << std::endl;
      const auto config = makeConfig(spec.layers);
      TrainRun run;
      run.family = &family;
      run.spec = &spec;
      TrainContext context;
      context.config = &config;
      context.calibrationCases = calibrationCases;
      context.developmentCases = developmentCases;
      context.run = &run;
      train::MarginLossSpec lossSpec;
      lossSpec.family = family.family;
      lossSpec.delta = family.delta;
      lossSpec.tau = family.tau;
      lossSpec.lambda = family.lambda;
      run.training = train::runMarginTraining(
          config, spec.seed, lossSpec, steps, evaluationSteps,
          micro ? 4 : 32, &trainCheckpoint, &context);
      run.finalDevelopment = dq::autoregressiveEvaluation(
          config, run.training.finalParameters, ar::Partition::DEVELOPMENT);
      if (steps >= 32)
        run.stabilityDevelopment = dq::autoregressiveEvaluation(
            config, run.training.stabilityParameters,
            ar::Partition::DEVELOPMENT);
      runs.push_back(std::move(run));
    }
  }
  return runs;
}

struct FamilyGate {
  bool finite = true;
  int improvedSeeds = 0;
  bool controlNonworse = true;
  bool marginImproved = false;
  bool stabilityPass = true;
  bool pass = false;
  double pooledTokenDelta = 0.0;
  double pooledSequenceDelta = 0.0;
  double pooledLtmDelta = 0.0;
};

std::map<std::string, FamilyGate> trainingGates(
    const std::vector<TrainRun>& runs,
    const std::map<std::string, BaselineMetrics>& baselines) {
  std::map<std::string, FamilyGate> gates;
  for (const auto& family : kFamilies) {
    FamilyGate gate;
    double ltmSum = 0.0, baseLtmSum = 0.0;
    int ltmCount = 0;
    bool seenControl = false;
    for (const auto& run : runs) {
      if (run.family != &family) continue;
      const auto& spec = *run.spec;
      const auto baseIt = baselines.find(spec.publicId);
      if (baseIt == baselines.end() || !baseIt->second.present)
        throw std::runtime_error(std::string("baseline missing for ") +
                           spec.publicId);
      const auto& base = baseIt->second;
      gate.finite = gate.finite && run.training.finite &&
                    run.finalDevelopment.allFinite;
      gate.stabilityPass = gate.stabilityPass &&
                           run.finalDevelopment.tokenExact >=
                               run.stabilityDevelopment.tokenExact;
      const std::int64_t tokenDelta =
          static_cast<std::int64_t>(run.finalDevelopment.tokenExact) -
          static_cast<std::int64_t>(base.arDevTokenExact);
      if (spec.layers == 19) {
        if (tokenDelta > 0) ++gate.improvedSeeds;
        gate.pooledTokenDelta += static_cast<double>(tokenDelta);
        gate.pooledSequenceDelta +=
            static_cast<double>(
                static_cast<std::int64_t>(
                    run.finalDevelopment.sequenceExact) -
                static_cast<std::int64_t>(base.arDevSequenceExact));
        if (!run.development.empty()) {
          ltmSum += run.development.back().lowerTailMarginQ10;
          baseLtmSum += base.marginDevLtm;
          ++ltmCount;
        }
      } else {
        seenControl = true;
        gate.controlNonworse = gate.controlNonworse && tokenDelta >= 0;
      }
    }
    if (ltmCount > 0) {
      gate.pooledLtmDelta = (ltmSum - baseLtmSum) / double(ltmCount);
      gate.marginImproved = gate.pooledLtmDelta > 0.0;
    }
    if (!seenControl) gate.controlNonworse = false;
    gate.pass = gate.finite && gate.improvedSeeds >= 2 &&
                gate.controlNonworse && gate.marginImproved &&
                gate.stabilityPass;
    gates[family.id] = gate;
  }
  return gates;
}

std::string bestTrainingFamily(const std::map<std::string, FamilyGate>& gates) {
  const FamilySpec* best = nullptr;
  const FamilyGate* bestGate = nullptr;
  for (const auto& family : kFamilies) {
    const auto found = gates.find(family.id);
    if (found == gates.end() || !found->second.pass) continue;
    if (!best || found->second.pooledTokenDelta > bestGate->pooledTokenDelta ||
        (found->second.pooledTokenDelta == bestGate->pooledTokenDelta &&
         found->second.pooledLtmDelta > bestGate->pooledLtmDelta)) {
      best = &family;
      bestGate = &found->second;
    }
  }
  return best ? best->id : "NONE";
}

void writeTrainingMetrics(std::ofstream& output,
                          const std::vector<TrainRun>& runs,
                          const std::map<std::string, BaselineMetrics>& baselines) {
  row(output, {"family_id", "configuration_id", "seed", "layers", "steps",
               "family_parameter", "final_ar_dev_token_exact",
               "final_ar_dev_sequence_exact", "final_ar_dev_nll",
               "stability_ar_dev_token_exact",
               "stability_ar_dev_sequence_exact",
               "final_margin_dev_lower_tail_margin_q10",
               "final_margin_dev_median_survival",
               "final_margin_calib_lower_tail_margin_q10", "final_nll",
               "final_margin_term", "final_total_loss", "final_mean_margin",
               "final_critical_share", "final_gradient_norm",
               "last_parity_step", "finite", "delta_token_exact_vs_baseline",
               "delta_sequence_exact_vs_baseline", "delta_ltm_vs_baseline"});
  for (const auto& run : runs) {
    const auto& spec = *run.spec;
    const auto baseIt = baselines.find(spec.publicId);
    const bool hasBase = baseIt != baselines.end() && baseIt->second.present;
    const auto& summary = run.training.stepSummaries.back();
    const auto& dev =
        run.development.empty() ? cm::CheckpointMetrics{} : run.development.back();
    const auto& cal =
        run.calibration.empty() ? cm::CheckpointMetrics{} : run.calibration.back();
    const std::int64_t tokenDelta =
        static_cast<std::int64_t>(run.finalDevelopment.tokenExact) -
        static_cast<std::int64_t>(baseIt->second.arDevTokenExact);
    const std::int64_t sequenceDelta =
        static_cast<std::int64_t>(run.finalDevelopment.sequenceExact) -
        static_cast<std::int64_t>(baseIt->second.arDevSequenceExact);
    row(output, {run.family->id, spec.publicId, text(spec.seed),
                 text(spec.layers), text(run.training.steps),
                 run.family->parameterDescription,
                 text(run.finalDevelopment.tokenExact),
                 text(run.finalDevelopment.sequenceExact),
                 number(run.finalDevelopment.autoregressiveNll),
                 text(run.stabilityDevelopment.tokenExact),
                 text(run.stabilityDevelopment.sequenceExact),
                 number(dev.lowerTailMarginQ10),
                 number(dev.medianFirstErrorSurvival),
                 number(cal.lowerTailMarginQ10), number(summary.nll),
                 number(summary.marginTerm), number(summary.total),
                 number(summary.meanMargin), number(summary.criticalShare),
                 number(summary.gradientNorm),
                 text(run.training.lastParityStep),
                 run.training.finite && run.finalDevelopment.allFinite
                     ? "true"
                     : "false",
                 hasBase ? text(tokenDelta) : "NA",
                 hasBase ? text(sequenceDelta) : "NA",
                 hasBase ? number(dev.lowerTailMarginQ10 -
                                  baseIt->second.marginDevLtm)
                         : "NA"});
  }
}

void writeTrainingTrajectory(std::ofstream& output,
                             const std::vector<TrainRun>& runs) {
  row(output, {"family_id", "configuration_id", "partition", "step",
               "token_exact", "token_total", "sequence_exact",
               "sequence_total", "autoregressive_nll",
               "median_first_error_survival", "lower_tail_margin_q10",
               "all_finite"});
  for (const auto& run : runs) {
    for (const auto& m : run.calibration)
      row(output, {run.family->id, run.spec->publicId, "MARGIN_CALIBRATION_V1",
                   text(m.step), text(m.tokenExact), text(m.tokenTotal),
                   text(m.sequenceExact), text(m.sequenceTotal),
                   number(m.autoregressiveNll),
                   number(m.medianFirstErrorSurvival),
                   number(m.lowerTailMarginQ10),
                   m.allFinite ? "true" : "false"});
    for (const auto& m : run.development)
      row(output, {run.family->id, run.spec->publicId, "MARGIN_DEVELOPMENT_V1",
                   text(m.step), text(m.tokenExact), text(m.tokenTotal),
                   text(m.sequenceExact), text(m.sequenceTotal),
                   number(m.autoregressiveNll),
                   number(m.medianFirstErrorSurvival),
                   number(m.lowerTailMarginQ10),
                   m.allFinite ? "true" : "false"});
  }
}

void writeTrainingGate(std::ofstream& output,
                       const std::map<std::string, FamilyGate>& gates) {
  row(output, {"family_id", "finite", "improved_seeds_count",
               "control_nonworse", "margin_improved", "stability_pass",
               "pooled_token_delta", "pooled_sequence_delta",
               "pooled_ltm_delta", "pass"});
  for (const auto& family : kFamilies) {
    const auto& gate = gates.at(family.id);
    row(output, {family.id, gate.finite ? "true" : "false",
                 text(gate.improvedSeeds),
                 gate.controlNonworse ? "true" : "false",
                 gate.marginImproved ? "true" : "false",
                 gate.stabilityPass ? "true" : "false",
                 number(gate.pooledTokenDelta),
                 number(gate.pooledSequenceDelta),
                 number(gate.pooledLtmDelta),
                 gate.pass ? "true" : "false"});
  }
}

void writeTrainingDecision(std::ofstream& output,
                           const std::map<std::string, FamilyGate>& gates,
                           const std::string& best, int steps) {
  int passCount = 0;
  for (const auto& entry : gates)
    if (entry.second.pass) ++passCount;
  row(output, {"steps", "passing_family_count", "best_family",
               "training_development_gate", "decision"});
  row(output, {text(steps), text(passCount), best,
               passCount > 0 ? "PASS" : "REJECT",
               best != "NONE"
                   ? std::string("FREEZE_FAMILY_") + best
                   : "NO_TRAINING_FAMILY_ACCEPTED"});
}

int runTrain(const fs::path& output, const fs::path& baselineDir, bool micro) {
  std::cerr << "building datasets..." << std::endl;
  std::string error;
  if (!cm::validateDatasets(8, &error, /*requirePinnedHashes=*/true))
    throw std::runtime_error("margin dataset validation failed: " + error);
  if (!ar::validatePartitions(8, &error))
    throw std::runtime_error("AR dataset validation failed: " + error);
  const auto baselines = readBaseline(baselineDir);
  const int steps = micro ? 64 : 320;
  std::cerr << "running training families ("
            << (micro ? "micro smoke, 64 steps" : "full gate, 320 steps")
            << ")..." << std::endl;
  const auto runs = runTrainingFamilies(micro);
  const auto gates = trainingGates(runs, baselines);
  const auto best = bestTrainingFamily(gates);
  std::cerr << "writing reports..." << std::endl;
  {
    std::ofstream file(output / "training-family-metrics.csv");
    writeTrainingMetrics(file, runs, baselines);
  }
  {
    std::ofstream file(output / "training-trajectory.csv");
    writeTrainingTrajectory(file, runs);
  }
  {
    std::ofstream file(output / "training-family-gate.csv");
    writeTrainingGate(file, gates);
  }
  {
    std::ofstream file(output / "training-family-decision.csv");
    writeTrainingDecision(file, gates, best, steps);
  }
  std::cout << "critical_margin_training_probe=PASS" << std::endl;
  std::cout << "training_steps=" << steps << std::endl;
  std::cout << "best_training_family=" << best << std::endl;
  int passCount = 0;
  for (const auto& entry : gates)
    if (entry.second.pass) ++passCount;
  std::cout << "passing_family_count=" << passCount << std::endl;
  std::cout << "training_development_gate="
            << (passCount > 0 ? "PASS" : "REJECT") << std::endl;
  return 0;
}

void writeConsistency(std::ofstream& output,
                      const std::vector<RunResult>& results) {
  row(output, {"configuration_id", "pinned_selected_step",
               "regenerated_selected_step", "selected_token_exact",
               "final_token_exact", "selected_sequence_exact",
               "final_sequence_exact", "selected_step_matches_pinned",
               "anchors_match", "loss_parity_all_steps",
               "gradient_parity_all_steps"});
  for (const auto& result : results) {
    bool lossParity = true, gradientParity = true;
    for (const auto& step : result.gradients.steps) {
      lossParity = lossParity && step.lossParity;
      gradientParity = gradientParity && step.gradientParity;
    }
    row(output, {result.spec->publicId,
                 text(result.spec->pinnedSelectedStep),
                 text(result.run.selectedStep),
                 text(result.selectedDevelopment.tokenExact),
                 text(result.finalDevelopment.tokenExact),
                 text(result.selectedDevelopment.sequenceExact),
                 text(result.finalDevelopment.sequenceExact),
                 result.run.selectedStep == result.spec->pinnedSelectedStep
                     ? "true"
                     : "false",
                 result.anchorsMatch ? "true" : "false",
                 lossParity ? "true" : "false",
                 gradientParity ? "true" : "false"});
  }
}

void writeDecision(
    std::ofstream& output, const std::map<std::string, GateRow>& gates,
    const std::string& best, const Attribution& attribution,
    std::size_t passingCount) {
  const bool hasBest = gates.count(best) > 0;
  const bool developmentGatePass = passingCount > 0 && hasBest;
  const bool losoCollapseFree = hasBest && gates.at(best).losoCollapseFree;
  const bool candidatePass = developmentGatePass && losoCollapseFree;
  row(output, {"hypothesis_prior", "passing_variant_count", "best_variant",
               "development_gate", "loso_collapse_free",
               "gradient_attribution", "recommended_training_family",
               "recommended_training_family_evidence",
               "checkpoint_objective_conclusion"});
  row(output, {"CRITICAL_TOKEN_MARGIN_LOSS", text(passingCount), best,
               developmentGatePass ? "PASS" : "REJECT",
               losoCollapseFree ? "PASS" : "REJECT",
               attribution.classification(),
               attribution.recommendedFamilies(),
               attribution.classification() == "NO_SINGLE_TRAINING_MECHANISM"
                   ? "NO_DIRECT_EVIDENCE"
                   : "GRADIENT_ATTRIBUTION",
               candidatePass ? "CHECKPOINT_OBJECTIVE_DEVELOPMENT_PASS"
                             : "CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT"});
}

}  // namespace

int run(const fs::path& output) {
  std::cerr << "building datasets..." << std::endl;
  std::string error;
  if (!cm::validateDatasets(8, &error, /*requirePinnedHashes=*/true))
    throw std::runtime_error("margin dataset validation failed: " + error);
  if (!ar::validatePartitions(8, &error))
    throw std::runtime_error("AR dataset validation failed: " + error);

  const auto results = runAll();
  const auto analysis = analyze(results);
  const auto folds = runLoso(results);
  const auto& control =
      *std::find_if(results.begin(), results.end(), [](const RunResult& r) {
        return r.spec->layers == 18;
      });
  const auto gates = developmentGates(analysis, control, folds);
  const auto attribution = classifyAttribution(results, analysis);
  std::vector<cm::VariantEvidence> evidence;
  for (const auto& spec : cm::kObjectives)
    evidence.push_back(pooledEvidence(spec, analysis));
  const auto best = bestVariant(evidence);
  std::size_t passingCount = 0;
  for (const auto& entry : gates)
    if (entry.second.pass) ++passingCount;

  std::cerr << "writing reports..." << std::endl;
  {
    std::ofstream file(output / "configuration.csv");
    writeConfiguration(file, results);
  }
  {
    std::ofstream file(output / "dataset-partitions.csv");
    writeDatasetPartitions(file);
  }
  {
    std::ofstream file(output / "dataset-overlap.csv");
    writeDatasetOverlap(file);
  }
  {
    std::ofstream file(output / "dataset-hashes.csv");
    writeDatasetHashes(file);
  }
  {
    std::ofstream file(output / "checkpoint-metrics.csv");
    writeCheckpointMetrics(file, results);
  }
  {
    std::ofstream file(output / "baseline-summary.csv");
    writeBaselineSummary(file, results);
  }
  {
    std::ofstream file(output / "objective-scores.csv");
    writeObjectiveScores(file, results);
  }
  {
    std::ofstream file(output / "objective-correlations.csv");
    writeObjectiveCorrelations(file, analysis);
  }
  {
    std::ofstream file(output / "checkpoint-selection.csv");
    writeCheckpointSelection(file, analysis);
  }
  {
    std::ofstream file(output / "leave-one-seed-out.csv");
    writeLeaveOneSeedOut(file, folds);
  }
  {
    std::ofstream file(output / "development-gate.csv");
    writeDevelopmentGate(file, gates);
  }
  {
    std::ofstream file(output / "gradient-attribution.csv");
    writeGradientAttribution(file, results);
  }
  {
    std::ofstream file(output / "consistency.csv");
    writeConsistency(file, results);
  }
  {
    std::ofstream file(output / "decision.csv");
    writeDecision(file, gates, best, attribution, passingCount);
  }

  std::cout << "critical_margin_objective_probe=PASS" << std::endl;
  std::cout << "best_variant=" << best << std::endl;
  std::cout << "passing_variant_count=" << passingCount << std::endl;
  std::cout << "development_gate="
            << (passingCount > 0 && gates.count(best) > 0 ? "PASS" : "REJECT")
            << std::endl;
  std::cout << "gradient_attribution=" << attribution.classification()
            << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--self-test") {
      std::string error;
      if (!cm::validateDatasets(8, &error, /*requirePinnedHashes=*/true))
        throw std::runtime_error(error);
      const auto cal = cm::buildMarginCalibrationV1();
      const auto dev = cm::buildMarginDevelopmentV1();
      if (cal.size() != 24 || dev.size() != 24)
        throw std::runtime_error("unexpected margin case count");
      if (cm::countCaseIdOverlap(cal, dev) != 0)
        throw std::runtime_error("unexpected margin case overlap");
      cm::CheckpointMetrics cp = cm::summarizeCheckpoint(
          4, {cm::CaseTrace{"self", {0.5, -1.5}, 2.0, 1, false, 1, true}});
      const auto score = cm::scoreObjective(cm::kObjectives[0], cp);
      if (!score.finite || std::abs(score.value - 0.75) > 1e-12)
        throw std::runtime_error("unexpected objective score");
      selfTestTraining();
      std::cout << "critical_margin_objective_probe_self_test=PASS"
                << std::endl;
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--train") {
      bool micro = false;
      fs::path baselineDir, output;
      for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--micro") micro = true;
        if (std::string(argv[i]) == "--baseline" && i + 1 < argc) {
          baselineDir = argv[i + 1];
          ++i;
        }
        if (std::string(argv[i]) == "--output" && i + 1 < argc) {
          output = argv[i + 1];
          ++i;
        }
      }
      if (output.empty() || baselineDir.empty()) {
        std::cerr << "usage: critical_margin_objective_probe --train "
                     "[--micro] --baseline DIR --output DIR"
                  << std::endl;
        return 2;
      }
      fs::create_directories(output);
      return runTrain(output, baselineDir, micro);
    }
    if (argc < 2 || std::string(argv[1]) != "--run") {
      std::cerr << "usage: critical_margin_objective_probe --run --output DIR"
                << std::endl;
      return 2;
    }
    fs::path output;
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "--output" && i + 1 < argc) {
        output = argv[i + 1];
        ++i;
      }
    }
    if (output.empty()) {
      std::cerr << "missing --output DIR" << std::endl;
      return 2;
    }
    fs::create_directories(output);
    return run(output);
  } catch (const std::exception& err) {
    std::cerr << "critical_margin_objective_probe: " << err.what() << std::endl;
    return 1;
  }
}
