// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Params = phonelm::qnn::TinyTransformerParameters;
using Member = std::vector<float> Params::*;

struct Field { Member member; };
const std::array<Field, 12>& fields() {
  static const std::array<Field, 12> value{{
      {&Params::tokenEmbedding}, {&Params::outputProjection}, {&Params::wq},
      {&Params::wk}, {&Params::wv}, {&Params::wo}, {&Params::gamma1},
      {&Params::beta1}, {&Params::gamma2}, {&Params::beta2}, {&Params::w1},
      {&Params::w2}}};
  return value;
}
const std::array<std::vector<uint32_t>, 4>& patterns() {
  static const std::array<std::vector<uint32_t>, 4> value{{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}, {10, 11, 12}}};
  return value;
}
struct Sample { uint32_t pattern, phase; };
struct Batch { std::vector<uint32_t> input, target; };

Batch makeBatch(const phonelm::tiny_lm::Config& config, uint32_t pattern, uint32_t phase) {
  const auto& sequence = patterns().at(pattern);
  Batch batch{std::vector<uint32_t>(config.tokens), std::vector<uint32_t>(config.tokens)};
  for (uint32_t position = 0; position < config.tokens; ++position) {
    batch.input[position] = sequence[(position + phase) % sequence.size()];
    batch.target[position] = sequence[(position + phase + 1) % sequence.size()];
  }
  return batch;
}
std::vector<Sample> samplingPlan(const std::string& mode) {
  std::vector<Sample> result;
  if (mode == "phase01_round_robin") {
    // A four-pattern cycle at phase 0 followed by the same four-pattern cycle
    // at phase 1 makes every pattern equally frequent and alternates its phase
    // on each occurrence.  Pattern 2 thereby covers both of its phases exactly.
    for (uint32_t phase = 0; phase < 2; ++phase)
      for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern)
        result.push_back({pattern, phase});
    return result;
  }
  if (mode != "phase0_round_robin" && mode != "all_phases_round_robin")
    throw std::invalid_argument("mode must be phase0_round_robin, phase01_round_robin, or all_phases_round_robin");
  for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern) {
    const uint32_t phases = mode == "phase0_round_robin" ? 1u : uint32_t(patterns()[pattern].size());
    for (uint32_t phase = 0; phase < phases; ++phase) result.push_back({pattern, phase});
  }
  return result;
}
Params zerosLike(const Params& source) {
  Params result;
  for (const auto field : fields()) (result.*field.member).assign((source.*field.member).size(), 0.0f);
  return result;
}
bool finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}
bool finite(const Params& parameters) {
  for (const auto field : fields()) if (!finite(parameters.*field.member)) return false;
  return true;
}
bool finite(const phonelm::tiny_lm::StepResult& step) {
  return std::isfinite(step.loss) && std::isfinite(step.accuracy) &&
      finite(step.embeddedInput) && finite(step.transformerOutput) && finite(step.logits) &&
      finite(step.probabilities) && finite(step.dLogits) && finite(step.gradients) && finite(step.next);
}
bool same(const Params& left, const Params& right) {
  for (const auto field : fields()) if (left.*field.member != right.*field.member) return false;
  return true;
}
double clipGlobal(Params& gradients, double threshold) {
  double sum = 0.0;
  for (const auto field : fields()) for (const float value : gradients.*field.member) sum += double(value) * value;
  const double norm = std::sqrt(sum);
  if (norm > threshold) {
    const float scale = float(threshold / norm);
    for (const auto field : fields()) for (float& value : gradients.*field.member) value *= scale;
  }
  return norm;
}
uint32_t argmax(const std::vector<float>& values, size_t base, uint32_t vocabulary) {
  uint32_t best = 0;
  for (uint32_t token = 1; token < vocabulary; ++token)
    if (values[base + token] > values[base + best]) best = token;
  return best;
}
double entropy(const std::vector<float>& probabilities, size_t base, uint32_t vocabulary) {
  double result = 0.0;
  for (uint32_t token = 0; token < vocabulary; ++token) {
    const double probability = std::max(double(probabilities[base + token]), 1e-30);
    result -= probability * std::log(probability);
  }
  return result;
}
double margin(const phonelm::tiny_lm::StepResult& step, size_t base, uint32_t target,
              uint32_t vocabulary) {
  float bestOther = -std::numeric_limits<float>::infinity();
  for (uint32_t token = 0; token < vocabulary; ++token)
    if (token != target) bestOther = std::max(bestOther, step.logits[base + token]);
  return double(step.logits[base + target]) - bestOther;
}
std::array<uint32_t, 3> top3(const phonelm::tiny_lm::StepResult& step, size_t base,
                             uint32_t vocabulary) {
  std::array<uint32_t, 3> result{{0, 1, 2}};
  std::sort(result.begin(), result.end(), [&](uint32_t left, uint32_t right) {
    return step.probabilities[base + left] > step.probabilities[base + right];
  });
  for (uint32_t token = 3; token < vocabulary; ++token) {
    if (step.probabilities[base + token] > step.probabilities[base + result[2]]) {
      result[2] = token;
      std::sort(result.begin(), result.end(), [&](uint32_t left, uint32_t right) {
        return step.probabilities[base + left] > step.probabilities[base + right];
      });
    }
  }
  return result;
}
struct Evaluation {
  double loss = 0.0, accuracy = 0.0, correctProbability = 0.0, meanMargin = 0.0, entropy = 0.0;
  uint32_t rows = 0, correct = 0;
  std::array<std::array<uint32_t, 8>, 4> positionCount{};
  std::array<std::array<uint32_t, 8>, 4> positionCorrect{};
  std::array<std::array<double, 8>, 4> positionMargin{};
  std::array<uint32_t, 32> targetFrequency{};
  std::array<uint32_t, 32> predictionFrequency{};
  std::array<std::array<uint32_t, 32>, 32> confusion{};
};
Evaluation evaluate(const phonelm::tiny_lm::Config& config, const Params& parameters) {
  Evaluation result;
  for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern) {
    for (uint32_t phase = 0; phase < patterns()[pattern].size(); ++phase) {
      const Batch batch = makeBatch(config, pattern, phase);
      const auto step = phonelm::tiny_lm::forwardBackward(
          config, phonelm::tiny_lm::oneHot(batch.input, config.vocabularySize),
          phonelm::tiny_lm::oneHot(batch.target, config.vocabularySize), parameters, 0.0f);
      result.loss += step.loss;
      for (uint32_t position = 0; position < config.tokens; ++position) {
        const size_t base = size_t(position) * config.vocabularySize;
        const uint32_t target = batch.target[position];
        const uint32_t prediction = argmax(step.logits, base, config.vocabularySize);
        const double rowMargin = margin(step, base, target, config.vocabularySize);
        ++result.rows; result.correct += prediction == target;
        ++result.positionCount[pattern][position];
        result.positionCorrect[pattern][position] += prediction == target;
        result.positionMargin[pattern][position] += rowMargin;
        ++result.targetFrequency[target]; ++result.predictionFrequency[prediction];
        ++result.confusion[target][prediction];
        result.correctProbability += step.probabilities[base + target];
        result.meanMargin += rowMargin;
        result.entropy += entropy(step.probabilities, base, config.vocabularySize);
      }
    }
  }
  const double batches = 13.0;
  result.loss /= batches;
  result.accuracy = double(result.correct) / result.rows;
  result.correctProbability /= result.rows;
  result.meanMargin /= result.rows;
  result.entropy /= result.rows;
  return result;
}
void printEvaluation(uint32_t seed, const Evaluation& evaluation) {
  std::cout << "teacher_forced=seed," << seed << ",loss," << evaluation.loss
            << ",accuracy," << evaluation.accuracy << ",correct_probability,"
            << evaluation.correctProbability << ",mean_margin," << evaluation.meanMargin
            << ",entropy," << evaluation.entropy << ",rows," << evaluation.rows
            << ",correct," << evaluation.correct << '\n';
  for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern) for (uint32_t position = 0; position < 8; ++position) {
    const auto count = evaluation.positionCount[pattern][position];
    std::cout << "pattern_position=seed," << seed << ",pattern," << pattern << ",position," << position
              << ",count," << count << ",accuracy," << double(evaluation.positionCorrect[pattern][position]) / count
              << ",mean_margin," << evaluation.positionMargin[pattern][position] / count << '\n';
  }
  for (uint32_t token = 0; token < 32; ++token) {
    if (evaluation.targetFrequency[token]) std::cout << "target_frequency=seed," << seed << ",token," << token << ",count," << evaluation.targetFrequency[token] << '\n';
    if (evaluation.predictionFrequency[token]) std::cout << "prediction_frequency=seed," << seed << ",token," << token << ",count," << evaluation.predictionFrequency[token] << '\n';
  }
  for (uint32_t target = 0; target < 32; ++target) for (uint32_t prediction = 0; prediction < 32; ++prediction)
    if (evaluation.confusion[target][prediction]) std::cout << "confusion=seed," << seed << ",target," << target << ",prediction," << prediction << ",count," << evaluation.confusion[target][prediction] << '\n';
}
void printRolloutStep(const char* mode, uint32_t seed, uint32_t pattern, uint32_t position,
                      uint32_t target, const phonelm::tiny_lm::StepResult& step,
                      uint32_t vocabulary) {
  const size_t base = (step.logits.size() / vocabulary - 1) * vocabulary;
  const uint32_t prediction = argmax(step.logits, base, vocabulary);
  const auto ranking = top3(step, base, vocabulary);
  std::cout << "rollout_step=seed," << seed << ",mode," << mode << ",pattern," << pattern
            << ",position," << position << ",target," << target << ",prediction," << prediction
            << ",correct_probability," << step.probabilities[base + target] << ",margin,"
            << margin(step, base, target, vocabulary) << ",entropy,"
            << entropy(step.probabilities, base, vocabulary) << ",top3," << ranking[0] << ':'
            << step.probabilities[base + ranking[0]] << '|' << ranking[1] << ':'
            << step.probabilities[base + ranking[1]] << '|' << ranking[2] << ':'
            << step.probabilities[base + ranking[2]] << '\n';
}
void printRollouts(uint32_t seed, const phonelm::tiny_lm::Config& config, const Params& parameters) {
  for (uint32_t pattern = 0; pattern < patterns().size(); ++pattern) {
    const auto& rule = patterns()[pattern];
    std::vector<uint32_t> oracleContext(config.tokens), freeContext(config.tokens);
    for (uint32_t position = 0; position < config.tokens; ++position)
      oracleContext[position] = rule[position % rule.size()];
    freeContext = oracleContext;
    int oracleFirstError = -1, freeFirstError = -1;
    uint32_t oracleCorrect = 0, freeCorrect = 0;
    for (uint32_t position = 0; position < config.tokens; ++position) {
      const uint32_t target =
          rule[(size_t(config.tokens) + position) % rule.size()];
      const auto targetRows = phonelm::tiny_lm::oneHot(
          std::vector<uint32_t>(config.tokens, target), config.vocabularySize);
      const auto oracleStep = phonelm::tiny_lm::forwardBackward(
          config, phonelm::tiny_lm::oneHot(oracleContext, config.vocabularySize),
          targetRows, parameters, 0.0f);
      const auto freeStep = phonelm::tiny_lm::forwardBackward(
          config, phonelm::tiny_lm::oneHot(freeContext, config.vocabularySize),
          targetRows, parameters, 0.0f);
      const size_t base = size_t(config.tokens - 1) * config.vocabularySize;
      const uint32_t oraclePrediction =
          argmax(oracleStep.logits, base, config.vocabularySize);
      const uint32_t freePrediction =
          argmax(freeStep.logits, base, config.vocabularySize);
      oracleCorrect += oraclePrediction == target;
      freeCorrect += freePrediction == target;
      if (oraclePrediction != target && oracleFirstError < 0)
        oracleFirstError = int(position);
      if (freePrediction != target && freeFirstError < 0)
        freeFirstError = int(position);
      printRolloutStep("oracle", seed, pattern, position, target, oracleStep,
                       config.vocabularySize);
      printRolloutStep("free", seed, pattern, position, target, freeStep,
                       config.vocabularySize);
      oracleContext.erase(oracleContext.begin());
      oracleContext.push_back(target);
      freeContext.erase(freeContext.begin());
      freeContext.push_back(freePrediction);
    }
    std::cout << "rollout_summary=seed," << seed << ",mode,oracle,pattern," << pattern
              << ",correct," << oracleCorrect << ",total," << config.tokens << ",first_error," << oracleFirstError << '\n';
    std::cout << "rollout_summary=seed," << seed << ",mode,free,pattern," << pattern
              << ",correct," << freeCorrect << ",total," << config.tokens << ",first_error," << freeFirstError << '\n';
  }
}
struct Result { Params parameters; Evaluation evaluation; bool finite = true; int clipped = 0; double maximumGradientNorm = 0.0; std::vector<uint32_t> visits; };
Result train(const std::string& mode, uint32_t seed, int steps) {
  phonelm::tiny_lm::Config config{32, 8, 16, 32, 1e-5f};
  Params parameters = phonelm::tiny_lm::initialParameters(config, seed);
  Params firstMoment = zerosLike(parameters), secondMoment = zerosLike(parameters);
  const auto plan = samplingPlan(mode);
  Result result; result.visits.assign(plan.size(), 0);
  for (int index = 1; index <= steps; ++index) {
    const auto sample = plan[size_t(index - 1) % plan.size()];
    ++result.visits[size_t(index - 1) % plan.size()];
    const Batch batch = makeBatch(config, sample.pattern, sample.phase);
    auto step = phonelm::tiny_lm::forwardBackward(
        config, phonelm::tiny_lm::oneHot(batch.input, config.vocabularySize),
        phonelm::tiny_lm::oneHot(batch.target, config.vocabularySize), parameters, 0.0f);
    const double gradientNorm = clipGlobal(step.gradients, 10.0);
    result.maximumGradientNorm = std::max(result.maximumGradientNorm, gradientNorm);
    result.clipped += gradientNorm > 10.0;
    const auto update = phonelm::tiny_lm::adamUpdate(parameters, step.gradients, firstMoment, secondMoment,
        0.0003f, 0.9f, 0.999f, 1e-8f, float(1.0 / (1.0 - std::pow(0.9, index))),
        float(1.0 / (1.0 - std::pow(0.999, index))));
    parameters = update.next; firstMoment = update.firstMoment; secondMoment = update.secondMoment;
    result.finite = result.finite && finite(step) && finite(parameters) && finite(firstMoment) && finite(secondMoment);
    if (!result.finite) break;
  }
  result.parameters = parameters;
  result.evaluation = evaluate(config, parameters);
  result.finite = result.finite && finite(parameters);
  return result;
}
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: tiny-lm-autoregressive-probe phase0_round_robin|phase01_round_robin|all_phases_round_robin first_seed seed_count steps\n";
    return 2;
  }
  const std::string mode = argv[1]; const uint32_t firstSeed = uint32_t(std::stoul(argv[2]));
  const uint32_t seedCount = uint32_t(std::stoul(argv[3])); const int steps = std::stoi(argv[4]);
  if (seedCount == 0 || steps <= 0 || steps > 2000) { std::cerr << "seed_count must be positive and steps must be 1..2000\n"; return 2; }
  const auto plan = samplingPlan(mode);
  std::cout << std::setprecision(10) << "shape=B1_T8_V32_D16_H1_L1_F32\noptimizer=ADAM\nlearning_rate=0.0003\nglobal_gradient_clip_threshold=10\nsteps=" << steps
            << "\nsampling=" << mode << "\nsampling_plan_count=" << plan.size()
            << "\nprefix_length_candidates=not_evaluated_fixed_T8_api_has_no_padding_or_loss_mask\n";
  std::vector<Result> results;
  for (uint32_t offset = 0; offset < seedCount; ++offset) {
    const uint32_t seed = firstSeed + offset;
    results.push_back(train(mode, seed, steps));
    const Result& result = results.back();
    std::cout << "seed=" << seed << "\ntraining_visit_counts=";
    for (size_t index = 0; index < result.visits.size(); ++index) {
      if (index) std::cout << '|';
      std::cout << plan[index].pattern << ':' << plan[index].phase << ':' << result.visits[index];
    }
    std::cout << "\ngradient_clip_events=" << result.clipped << "\nmaximum_preclip_gradient_norm=" << result.maximumGradientNorm
              << "\nfinite=" << (result.finite ? "true" : "false") << '\n';
    printEvaluation(seed, result.evaluation);
    printRollouts(seed, phonelm::tiny_lm::Config{32, 8, 16, 32, 1e-5f}, result.parameters);
  }
  const Result replayA = train(mode, firstSeed, steps), replayB = train(mode, firstSeed, steps);
  double loss = 0.0, accuracy = 0.0, marginValue = 0.0; bool allFinite = true;
  for (const Result& result : results) { loss += result.evaluation.loss; accuracy += result.evaluation.accuracy; marginValue += result.evaluation.meanMargin; allFinite = allFinite && result.finite; }
  std::cout << "summary_mode=" << mode << "\nsummary_seed_count=" << seedCount
            << "\nsummary_mean_teacher_forced_loss=" << loss / seedCount
            << "\nsummary_mean_teacher_forced_accuracy=" << accuracy / seedCount
            << "\nsummary_mean_teacher_forced_margin=" << marginValue / seedCount
            << "\nsummary_all_finite=" << (allFinite ? "true" : "false")
            << "\ndeterministic_replay=" << ((replayA.finite && replayB.finite && same(replayA.parameters, replayB.parameters)) ? "true" : "false") << '\n';
  return allFinite ? 0 : 1;
}
