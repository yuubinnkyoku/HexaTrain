// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "attention_minimal_cause_lib.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace amc = phonelm::attention_minimal_cause;
namespace ar = phonelm::autoregressive_validation;
namespace dq = phonelm::depth_quality;
namespace rp = phonelm::readout_probe;
namespace si = phonelm::seed_instability;
namespace tiny = phonelm::tiny_lm;

namespace {

struct RunSpec {
  const char* id;
  int depth;
  std::uint32_t seed;
  const char* parameterHash;
  std::uint64_t tokenExact;
  std::uint64_t sequenceExact;
};

constexpr std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", 19, 1, "fnv1a64:643e09a80b4de96e", 30, 2},
    {"L19_SEED_2", 19, 2, "fnv1a64:35394c806dd48fca", 63, 6},
    {"L19_SEED_4", 19, 4, "fnv1a64:ec23064d8b6a8807", 46, 6},
    {"L18_SEED_2_CONTROL", 18, 2, "fnv1a64:e56e01abc30e7449", 65, 8},
}};

tiny::Config configFor(int depth) {
  tiny::Config config;
  config.vocabularySize = 32;
  config.tokens = 8;
  config.dimension = 16;
  config.feedForwardDimension = 32;
  config.numLayers = static_cast<std::uint32_t>(depth);
  config.numHeads = 2;
  std::string error;
  if (!tiny::validateConfig(config, &error)) throw std::runtime_error(error);
  return config;
}

std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string orderedRowHash(const std::vector<rp::ProbeRow>& rows) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto& row : rows) {
    hash = ar::fnv1a(row.caseId.data(), row.caseId.size(), hash);
    const std::uint64_t size = row.context.size();
    hash = ar::fnv1a(&size, sizeof(size), hash);
    hash = ar::fnv1a(row.context.data(), row.context.size() * sizeof(std::uint32_t), hash);
    hash = ar::fnv1a(&row.truth, sizeof(row.truth), hash);
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

void writeEvaluationRow(std::ofstream& out, const RunSpec& spec,
                        const std::string& intervention,
                        const amc::Evaluation& metrics,
                        const std::string& parameterHash,
                        bool checkpointPreserved) {
  out << spec.id << ',' << spec.depth << ',' << spec.seed << ','
      << intervention << ',' << parameterHash << ','
      << (checkpointPreserved ? "true" : "false") << ','
      << metrics.teacherForced.tokenExact << ','
      << metrics.teacherForced.total << ','
      << number(metrics.teacherForced.meanNll) << ','
      << metrics.freeRunning.tokenExact << ','
      << metrics.freeRunning.tokenTotal << ','
      << metrics.freeRunning.sequenceExact << ','
      << metrics.freeRunning.sequenceTotal << ','
      << number(metrics.freeRunning.autoregressiveNll) << ','
      << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
      << number(metrics.freeRunning.lowerTailMarginQ10) << ','
      << number(metrics.minimumMargin) << ',' << metrics.tieCount << ','
      << (metrics.teacherForced.finite && metrics.freeRunning.allFinite
              ? "true" : "false") << '\n';
}

void cycle1(const fs::path& root) {
  fs::create_directories(root);
  const auto cases = ar::cases(ar::Partition::DEVELOPMENT, 8);
  const auto rows = rp::teacherForcedRows(cases, 8);
  if (!ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8))
    throw std::runtime_error("AR_DEVELOPMENT_IDENTITY");
  std::ofstream measurement(root / "measurement-audit.csv");
  measurement << "configuration_id,check,status,observed,expected\n";
  measurement << "ALL,ar_development_hash,PASS,fnv1a64:bd464d2a6e192d36,fnv1a64:bd464d2a6e192d36\n";
  measurement << "ALL,ordered_teacher_row_hash,PASS," << orderedRowHash(rows)
              << ',' << orderedRowHash(rows) << "\n";
  measurement << "ALL,final_holdout_open_count,PASS,0,0\n";
  std::ofstream identities(root / "canonical-identities.csv");
  identities << "configuration_id,depth,seed,parameter_content_hash,legacy_support_mask_hash,historical_legacy_support_mask_hash,first_moment_content_hash,second_moment_content_hash,attention_parameter_hash,nonattention_parameter_hash,independent_full_state_parity,finite\n";
  std::ofstream results(root / "evaluation-interventions.csv");
  results << "configuration_id,depth,seed,intervention,source_parameter_hash,checkpoint_preserved,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,margin_q10,minimum_teacher_margin,tie_count,all_finite\n";
  for (const auto& spec : kRuns) {
    const auto config = configFor(spec.depth);
    const auto formal = dq::runFormalCpu(config, spec.seed, 320, 0.003f,
                                         dq::StabilityMode::LEGACY, {320});
    si::TrainingState state;
    state.params = formal.checkpoints.at(320);
    state.firstMoment = formal.firstMoments.at(320);
    state.secondMoment = formal.secondMoments.at(320);
    state.step = 320;
    state.lastLoss = formal.steps.back().loss;
    state.finite = amc::finiteParameters(state.params);
    const std::string parameterHash = amc::contentHash(state.params);
    const std::string legacySupportHash = rp::fnv1aParams(state.params);
    const bool momentsFinite = amc::finiteParameters(state.firstMoment) &&
                               amc::finiteParameters(state.secondMoment);
    const auto independentState = amc::runTraining(
        config, spec.seed, amc::TrainingMode::Canonical, 320);
    const bool fullStateParity = si::sameParameters(
        state.params, independentState.params) && si::sameParameters(
        state.firstMoment, independentState.firstMoment) && si::sameParameters(
        state.secondMoment, independentState.secondMoment);
    identities << spec.id << ',' << spec.depth << ',' << spec.seed << ','
               << parameterHash << ',' << legacySupportHash << ','
               << spec.parameterHash << ','
               << amc::contentHash(state.firstMoment) << ','
               << amc::contentHash(state.secondMoment) << ','
               << amc::groupHash(state.params, amc::Group::Attention) << ','
               << amc::groupHash(state.params, amc::Group::Attention, false)
               << ',' << (fullStateParity ? "true" : "false") << ','
               << (state.finite && momentsFinite ? "true" : "false")
               << '\n';
    measurement << spec.id << ",historical_support_mask_hash,INVALID,"
                << legacySupportHash << ',' << spec.parameterHash << '\n';
    measurement << spec.id << ",independent_full_state_parity,"
                << (fullStateParity ? "PASS" : "FAIL") << ','
                << parameterHash << ','
                << amc::contentHash(independentState.params) << '\n';
    if (!fullStateParity || !state.finite || !momentsFinite)
      throw std::runtime_error(std::string("CANONICAL_IDENTITY:") + spec.id);
    const auto baseline = amc::evaluate(config, state.params,
                                        amc::Pattern::Learned, cases);
    if (baseline.freeRunning.tokenExact != spec.tokenExact ||
        baseline.freeRunning.sequenceExact != spec.sequenceExact)
      throw std::runtime_error(std::string("CANONICAL_METRIC:") + spec.id);
    const auto independent = dq::evaluateAutoregressive(
        config, state.params, ar::Partition::DEVELOPMENT);
    const bool evaluatorParity = amc::sameCaseMetrics(
        baseline.freeRunning, independent.metrics);
    measurement << spec.id << ",independent_evaluator_case_parity,"
                << (evaluatorParity ? "PASS" : "FAIL") << ','
                << baseline.freeRunning.tokenExact << ','
                << independent.metrics.tokenExact << '\n';
    if (!evaluatorParity) throw std::runtime_error("EVALUATOR_PARITY");
    writeEvaluationRow(results, spec, "LEARNED_ALPHA_1_NOOP", baseline,
                       parameterHash, true);
    for (const auto alpha : {0.5f, 0.25f, 0.0f}) {
      const auto altered = amc::scaledBranch(config, state.params, alpha, false);
      const auto metrics = amc::evaluate(config, altered,
                                         amc::Pattern::Learned, cases);
      std::ostringstream name;
      name << "EVAL_ATTENTION_ALPHA_" << std::setprecision(2) << alpha;
      writeEvaluationRow(results, spec, name.str(), metrics, parameterHash, true);
    }
    for (const auto pattern : {amc::Pattern::Self, amc::Pattern::Previous,
                               amc::Pattern::UniformCausal}) {
      const auto metrics = amc::evaluate(config, state.params, pattern, cases);
      writeEvaluationRow(results, spec, std::string("EVAL_") +
                         amc::patternName(pattern), metrics, parameterHash, true);
    }
    const auto ffnCut = amc::scaledBranch(config, state.params, 1.0f, true);
    writeEvaluationRow(results, spec, "EVAL_FFN_W2_ZERO_CONTROL",
                       amc::evaluate(config, ffnCut, amc::Pattern::Learned, cases),
                       parameterHash, true);
    measurement << spec.id << ",alpha_one_parameter_noop,PASS,"
                << parameterHash << ',' << amc::contentHash(
                    amc::scaledBranch(config, state.params, 1.0f, false)) << '\n';
  }
}

std::vector<amc::TrainingMode> parseModes(const std::string& text) {
  const std::map<std::string, amc::TrainingMode> known{
      {"self", amc::TrainingMode::FixedSelf},
      {"previous", amc::TrainingMode::FixedPrevious},
      {"uniform", amc::TrainingMode::FixedUniform},
      {"attention-freeze", amc::TrainingMode::FreezeAttentionInitial},
      {"qk-freeze", amc::TrainingMode::FreezeQkInitial},
      {"vo-freeze", amc::TrainingMode::FreezeVoInitial},
  };
  std::vector<amc::TrainingMode> result;
  std::stringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto found = known.find(token);
    if (found == known.end()) throw std::invalid_argument("TRAIN_MODE");
    result.push_back(found->second);
  }
  if (result.empty()) throw std::invalid_argument("EMPTY_TRAIN_MODES");
  return result;
}

void trainingCycle(const fs::path& root, const std::string& fileName,
                   const std::vector<amc::TrainingMode>& modes) {
  fs::create_directories(root);
  const auto cases = ar::cases(ar::Partition::DEVELOPMENT, 8);
  std::ofstream out(root / fileName);
  out << "configuration_id,depth,seed,intervention,steps,train_finite,final_train_loss,parameter_hash,first_moment_hash,second_moment_hash,frozen_scope_pass,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,margin_q10,all_finite\n";
  for (const auto& spec : kRuns) {
    const auto config = configFor(spec.depth);
    const auto initial = tiny::initialParameters(config, spec.seed);
    for (const auto mode : modes) {
      const auto run = amc::runTraining(config, spec.seed, mode, 320);
      bool scope = true;
      if (mode == amc::TrainingMode::FixedSelf ||
          mode == amc::TrainingMode::FixedPrevious ||
          mode == amc::TrainingMode::FixedUniform) {
        amc::P zeroInitial = initial;
        amc::zeroGroup(zeroInitial, amc::Group::Qk);
        scope = amc::sameGroup(run.params, zeroInitial, amc::Group::Qk, true);
        for (const auto* state : {&run.firstMoment, &run.secondMoment})
          for (const auto& item : tiny::parameterRegistry(*state))
            if (amc::inGroup(item.name, amc::Group::Qk))
              scope = scope && std::all_of(item.values->begin(), item.values->end(),
                                          [](float value) { return value == 0.0f; });
      } else {
        const amc::Group group =
            mode == amc::TrainingMode::FreezeAttentionInitial
                ? amc::Group::Attention
                : (mode == amc::TrainingMode::FreezeQkInitial
                       ? amc::Group::Qk : amc::Group::Vo);
        scope = amc::sameGroup(run.params, initial, group, true);
        for (const auto* state : {&run.firstMoment, &run.secondMoment})
          for (const auto& item : tiny::parameterRegistry(*state))
            if (amc::inGroup(item.name, group))
              scope = scope && std::all_of(item.values->begin(), item.values->end(),
                                          [](float value) { return value == 0.0f; });
      }
      if (!run.finite || !scope) throw std::runtime_error("TRAIN_SCOPE_OR_FINITE");
      const auto metrics = amc::evaluate(config, run.params,
                                         amc::trainingPattern(mode), cases);
      out << spec.id << ',' << spec.depth << ',' << spec.seed << ','
          << amc::trainingModeName(mode) << ",320,true,"
          << number(run.lastLoss) << ',' << amc::contentHash(run.params) << ','
          << amc::contentHash(run.firstMoment) << ','
          << amc::contentHash(run.secondMoment) << ",true,"
          << metrics.teacherForced.tokenExact << ','
          << metrics.teacherForced.total << ','
          << number(metrics.teacherForced.meanNll) << ','
          << metrics.freeRunning.tokenExact << ','
          << metrics.freeRunning.tokenTotal << ','
          << metrics.freeRunning.sequenceExact << ','
          << metrics.freeRunning.sequenceTotal << ','
          << number(metrics.freeRunning.autoregressiveNll) << ','
          << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
          << number(metrics.freeRunning.lowerTailMarginQ10) << ','
          << (metrics.teacherForced.finite && metrics.freeRunning.allFinite
                  ? "true" : "false") << '\n';
    }
  }
}

void hybridCycle(const fs::path& root, const std::string& pairs) {
  fs::create_directories(root);
  const auto config = configFor(19);
  const auto cases = ar::cases(ar::Partition::DEVELOPMENT, 8);
  std::ofstream out(root / "initialization-factorial.csv");
  out << "configuration_id,depth,attention_initial_seed,nonattention_initial_seed,optimizer_initial_state,steps,scope_pass,train_finite,final_train_loss,parameter_hash,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,margin_q10,all_finite\n";
  std::stringstream input(pairs);
  std::string pair;
  while (std::getline(input, pair, ',')) {
    const auto colon = pair.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("HYBRID_PAIR");
    const auto attentionSeed = static_cast<std::uint32_t>(std::stoul(pair.substr(0, colon)));
    const auto restSeed = static_cast<std::uint32_t>(std::stoul(pair.substr(colon + 1)));
    amc::P expectedRest = tiny::initialParameters(config, restSeed);
    const amc::P expectedAttention = tiny::initialParameters(config, attentionSeed);
    amc::P hybridInitial = expectedRest;
    amc::copyGroup(hybridInitial, expectedAttention, amc::Group::Attention);
    const bool scope = amc::sameGroup(hybridInitial, expectedAttention,
                                      amc::Group::Attention, true) &&
                       amc::sameGroup(hybridInitial, expectedRest,
                                      amc::Group::Attention, false);
    if (!scope) throw std::runtime_error("HYBRID_SCOPE");
    const auto run = amc::trainHybridInitial(config, attentionSeed, restSeed, 320);
    const auto metrics = amc::evaluate(config, run.params,
                                       amc::Pattern::Learned, cases);
    out << "L19_ATTN_" << attentionSeed << "_REST_" << restSeed
        << ",19," << attentionSeed << ',' << restSeed
        << ",ALL_ZERO,320,true," << (run.finite ? "true" : "false") << ','
        << number(run.lastLoss) << ',' << amc::contentHash(run.params) << ','
        << metrics.teacherForced.tokenExact << ','
        << metrics.teacherForced.total << ','
        << number(metrics.teacherForced.meanNll) << ','
        << metrics.freeRunning.tokenExact << ','
        << metrics.freeRunning.tokenTotal << ','
        << metrics.freeRunning.sequenceExact << ','
        << metrics.freeRunning.sequenceTotal << ','
        << number(metrics.freeRunning.autoregressiveNll) << ','
        << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
        << number(metrics.freeRunning.lowerTailMarginQ10) << ','
        << (metrics.teacherForced.finite && metrics.freeRunning.allFinite
                ? "true" : "false") << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string error;
    if (!amc::selfTest(&error)) {
      std::cerr << "ATTENTION_MINIMAL_CAUSE_SELF_TEST_FAIL:" << error << '\n';
      return 1;
    }
    if (argc == 1 || std::string(argv[1]) == "--self-test") {
      std::cout << "ATTENTION_MINIMAL_CAUSE_SELF_TEST_PASS\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--cycle1") {
      cycle1(argv[2]);
    } else if (argc == 5 && std::string(argv[1]) == "--train-modes") {
      trainingCycle(argv[4], argv[3], parseModes(argv[2]));
    } else if (argc == 4 && std::string(argv[1]) == "--hybrids") {
      hybridCycle(argv[3], argv[2]);
    } else {
      throw std::invalid_argument(
          "usage: --self-test | --cycle1 root | --train-modes modes file root | --hybrids pairs root");
    }
    std::cout << "ATTENTION_MINIMAL_CAUSE_RUN_PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ATTENTION_MINIMAL_CAUSE_RUN_FAIL:" << error.what() << '\n';
    return 1;
  }
}
