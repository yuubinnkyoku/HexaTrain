// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Post-training CPU analysis for the fixed validation-selection candidate.
#include "depth_quality_lib.h"
#include <iomanip>
#include <iostream>

namespace dq = phonelm::depth_quality;
namespace vs = phonelm::validation_selection;

static dq::Config config(std::uint32_t layers) {
  dq::Config c;
  c.tokens = 8;
  c.vocabularySize = 32;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = layers;
  c.numHeads = 2;
  return c;
}

static const dq::StepMetric* stepMetric(const dq::FormRun& run, int step) {
  if (step == 0) return nullptr;
  return &run.steps.at(std::size_t(step - 1));
}

static void emit(std::uint32_t layers, std::uint32_t seed) {
  const auto c = config(layers);
  const auto selected = dq::runValidationSelectedCpu(
      c, seed, 320, vs::Mode::BEST_VALIDATION_V1);
  const auto finalGeneration = dq::generationQuality(c, selected.training.finalParameters);
  const auto selectedEval = dq::phase1Evaluation(c, selected.selectedParameters);
  const auto finalEval = dq::phase1Evaluation(c, selected.training.finalParameters);
  const auto initialBatch = dq::formalBatch(c, 0, 0);
  const auto initialTraining = phonelm::tiny_lm::forwardBackward(
      c, initialBatch.first, initialBatch.second,
      selected.training.checkpoints.at(0), 0.0f);
  std::cout << "SUMMARY," << layers << ',' << seed << ",true," << selected.selectedStep
            << ',' << selected.bestValidation.loss << ','
            << selected.bestValidation.accuracy << ','
            << selected.finalStepValidation.loss << ','
            << selected.finalStepValidation.accuracy << ','
            << selected.generation.oracleExact << ','
            << selected.generation.freeExact << ',' << finalGeneration.oracleExact
            << ',' << finalGeneration.freeExact << ',' << selectedEval.loss << ','
            << selectedEval.accuracy << ',' << finalEval.loss << ','
            << finalEval.accuracy << '\n';
  for (const auto& entry : selected.validationTrajectory) {
    const auto generation = dq::generationQuality(c, selected.training.checkpoints.at(entry.first));
    const auto* training = stepMetric(selected.training, entry.first);
    const double trainingLoss = training ? training->loss : initialTraining.loss;
    const double trainingAccuracy = training ? training->accuracy : initialTraining.accuracy;
    double initialGradientSquared = 0.0;
    if (!training)
      for (const auto& item : phonelm::tiny_lm::parameterRegistry(initialTraining.gradients))
        for (float value : *item.values) initialGradientSquared += double(value) * value;
    std::cout << "TRAJECTORY," << layers << ',' << seed << ',' << entry.first
              << ',' << trainingLoss << ',' << trainingAccuracy << ',' << entry.second.loss
              << ',' << entry.second.accuracy << ',' << entry.second.targetMargin
              << ',' << entry.second.targetProbability << ','
              << (training ? training->parameterNorm : dq::registryNorm(selected.training.checkpoints.at(0)))
              << ',' << (training ? training->gradientNorm : std::sqrt(initialGradientSquared)) << ','
              << (training ? training->updateToParameter : 0.0) << ','
              << generation.oracleExact << ',' << generation.freeExact << '\n';
  }
  for (int patience : {2, 3, 4}) {
    const auto simulation = vs::simulateEarlyStop(
        selected.validationTrajectory, patience, 320);
    std::cout << "EARLY_STOP," << layers << ',' << seed << ',' << patience << ','
              << simulation.stopStep << ',' << simulation.bestStep << ','
              << simulation.savedTrainingSteps << '\n';
  }
}

int main() {
  std::cout << std::setprecision(17);
  for (std::uint32_t seed : {1u, 2u, 4u}) emit(19, seed);
  for (std::uint32_t seed : {1u, 2u}) emit(18, seed);
  return 0;
}
