// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "context_supervision_stability_lib.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace amc = phonelm::attention_minimal_cause;
namespace ar = phonelm::autoregressive_validation;
namespace cs = phonelm::context_supervision;
namespace si = phonelm::seed_instability;
namespace tiny = phonelm::tiny_lm;

namespace {

struct RunSpec {
  const char* id;
  int depth;
  std::uint32_t seed;
  const char* canonicalParameterHash;
  std::uint64_t canonicalDevTokenExact;
  std::uint64_t canonicalDevSequenceExact;
  std::uint64_t canonicalHomogeneousTokenExact;
};

constexpr std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", 19, 1, "fnv1a64:55b7058fd81b186f", 30, 2, 32},
    {"L19_SEED_2", 19, 2, "fnv1a64:106556f181be6634", 63, 6, 24},
    {"L19_SEED_4", 19, 4, "fnv1a64:6536f953c8dd77c9", 46, 6, 32},
    {"L18_SEED_2_CONTROL", 18, 2, "fnv1a64:6a9354c52237f851", 65, 8, 26},
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

std::string counts(const std::array<std::uint64_t, 32>& values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ';';
    out << i << ':' << values[i];
  }
  return out.str();
}

void writeMetricRow(std::ofstream& out, const RunSpec& spec,
                    cs::Condition condition, const char* dataset,
                    const amc::Evaluation& metrics,
                    const cs::TrainingRun& run,
                    const cs::AttentionBehavior& behavior,
                    double dosePercent = 25.0) {
  const double qk = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::Qk);
  const double v = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::V);
  const double o = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::O);
  const double ffn = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::Ffn);
  const double norm = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::Norm);
  const double embeddingHead = cs::groupDeltaNorm(
      run.initial, run.params, cs::ParameterGroup::EmbeddingHead);
  const bool attentionActive = qk > 0.0 && v > 0.0 && o > 0.0 &&
      behavior.meanOutputNorm > 1e-3 && behavior.meanNonSelfMass > 0.05;
  out << spec.id << ',' << spec.depth << ',' << spec.seed << ','
      << cs::conditionName(condition) << ',' << number(dosePercent) << ','
      << dataset << ",320,"
      << run.exampleCount << ',' << run.specialExampleCount << ','
      << run.scheduleIdentity << ',' << amc::contentHash(run.initial) << ','
      << amc::contentHash(run.params) << ','
      << amc::contentHash(run.firstMoment) << ','
      << amc::contentHash(run.secondMoment) << ','
      << (run.finite ? "true" : "false") << ',' << number(run.finalLoss) << ','
      << metrics.teacherForced.tokenExact << ','
      << metrics.teacherForced.total << ','
      << number(metrics.teacherForced.meanNll) << ','
      << number(metrics.teacherForced.marginQ10) << ','
      << metrics.freeRunning.tokenExact << ','
      << metrics.freeRunning.tokenTotal << ','
      << metrics.freeRunning.sequenceExact << ','
      << metrics.freeRunning.sequenceTotal << ','
      << number(metrics.freeRunning.autoregressiveNll) << ','
      << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
      << number(metrics.freeRunning.lowerTailMarginQ10) << ','
      << number(qk) << ',' << number(v) << ',' << number(o) << ','
      << number(ffn) << ',' << number(norm) << ',' << number(embeddingHead) << ','
      << number(behavior.meanOutputNorm) << ','
      << number(behavior.meanEntropy) << ','
      << number(behavior.meanSelfMass) << ','
      << number(behavior.meanPreviousMass) << ','
      << number(behavior.meanFarMass) << ','
      << number(behavior.meanNonSelfMass) << ','
      << (attentionActive ? "true" : "false") << ','
      << (metrics.teacherForced.finite && metrics.freeRunning.allFinite &&
                  behavior.finite
              ? "true" : "false")
      << '\n';
}

void runCycle1(const fs::path& root) {
  fs::create_directories(root);
  if (!ar::hashMatchesPinned(ar::Partition::TRAIN, 8) ||
      !ar::hashMatchesPinned(ar::Partition::VALIDATION, 8) ||
      !ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8) ||
      !ar::hashMatchesPinned(ar::Partition::FINAL, 8))
    throw std::runtime_error("PINNED_DATASET_IDENTITY");
  const auto development = ar::cases(ar::Partition::DEVELOPMENT, 8);
  const auto homogeneous = si::homogeneousPhase0Cases();
  if (development.size() != 24 || homogeneous.size() != 4)
    throw std::runtime_error("EVALUATION_CASE_COUNT");

  std::ofstream measurement(root / "measurement-audit.csv");
  measurement << "configuration_id,check,status,observed,expected\n";
  measurement << "ALL,train_hash,PASS," << ar::partitionHash(ar::Partition::TRAIN)
              << ',' << ar::pinnedPartitionHash(ar::Partition::TRAIN) << '\n';
  measurement << "ALL,validation_hash,PASS,"
              << ar::partitionHash(ar::Partition::VALIDATION) << ','
              << ar::pinnedPartitionHash(ar::Partition::VALIDATION) << '\n';
  measurement << "ALL,development_hash,PASS,"
              << ar::partitionHash(ar::Partition::DEVELOPMENT) << ','
              << ar::pinnedPartitionHash(ar::Partition::DEVELOPMENT) << '\n';
  measurement << "ALL,final_hash_only,PASS,"
              << ar::partitionHash(ar::Partition::FINAL) << ','
              << ar::pinnedPartitionHash(ar::Partition::FINAL) << '\n';
  measurement << "ALL,final_holdout_evaluations,PASS,0,0\n";
  measurement << "ALL,development_case_count,PASS," << development.size()
              << ",24\n";
  measurement << "ALL,development_teacher_rows,PASS,"
              << phonelm::readout_probe::teacherForcedRows(development).size()
              << ",144\n";

  const auto auditConfig = configFor(19);
  const auto controlHistogram = cs::scheduleHistogram(
      auditConfig, cs::Condition::HomogeneousMatched, 4);
  const auto mixedHistogram = cs::scheduleHistogram(
      auditConfig, cs::Condition::MixedInvariant, 4);
  if (!cs::sameHistogram(controlHistogram, mixedHistogram))
    throw std::runtime_error("MATCHED_SCHEDULE_HISTOGRAM");
  std::ofstream schedule(root / "schedule-audit.csv");
  schedule << "condition,dose_percent,steps,supervised_rows,special_steps,"
              "schedule_hash,input_histogram,target_histogram,matched_histogram\n";
  for (const auto condition : {cs::Condition::Canonical,
                               cs::Condition::HomogeneousMatched,
                               cs::Condition::MixedInvariant}) {
    const auto histogram = cs::scheduleHistogram(auditConfig, condition, 4);
    const bool matched = condition == cs::Condition::Canonical ||
        cs::sameHistogram(histogram, condition == cs::Condition::MixedInvariant
                                        ? controlHistogram : mixedHistogram);
    schedule << cs::conditionName(condition) << ",25,320,2560,"
             << histogram.specialExamples << ','
             << cs::scheduleHash(auditConfig, condition, 4) << ','
             << counts(histogram.input) << ',' << counts(histogram.target) << ','
             << (matched ? "true" : "false") << '\n';
  }

  std::ofstream results(root / "training-results.csv");
  results << "configuration_id,depth,seed,condition,dose_percent,dataset,steps,"
             "example_count,special_example_count,schedule_hash,initial_parameter_hash,"
             "final_parameter_hash,first_moment_hash,second_moment_hash,train_finite,"
             "final_train_loss,teacher_token_exact,teacher_token_total,teacher_nll,"
             "teacher_margin_q10,free_token_exact,free_token_total,free_sequence_exact,"
             "free_sequence_total,free_nll,median_first_error,free_margin_q10,"
             "qk_update_l2,v_update_l2,o_update_l2,ffn_update_l2,norm_update_l2,"
             "embedding_head_update_l2,attention_output_norm,attention_entropy,"
             "attention_self_mass,attention_previous_mass,attention_far_mass,"
             "attention_nonself_mass,ordinary_attention_active,all_finite\n";

  for (const auto& spec : kRuns) {
    const auto config = configFor(spec.depth);
    for (const auto condition : {cs::Condition::Canonical,
                                 cs::Condition::HomogeneousMatched,
                                 cs::Condition::MixedInvariant}) {
      const auto run = cs::runTraining(config, spec.seed, condition, 4, 320);
      const auto parameterHash = amc::contentHash(run.params);
      const auto devMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, development);
      const auto homogeneousMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, homogeneous);
      const auto behavior = cs::attentionBehavior(config, run.params, development);
      if (!run.finite || !devMetrics.teacherForced.finite ||
          !devMetrics.freeRunning.allFinite || !behavior.finite)
        throw std::runtime_error(std::string("NONFINITE_RUN:") + spec.id + ":" +
                                 cs::conditionName(condition));
      if (condition == cs::Condition::Canonical &&
          (parameterHash != spec.canonicalParameterHash ||
           devMetrics.freeRunning.tokenExact != spec.canonicalDevTokenExact ||
           devMetrics.freeRunning.sequenceExact != spec.canonicalDevSequenceExact ||
           homogeneousMetrics.freeRunning.tokenExact !=
               spec.canonicalHomogeneousTokenExact))
        throw std::runtime_error(
            std::string("CANONICAL_ANCHOR_MISMATCH:") + spec.id + ":hash=" +
            parameterHash + ":dev_token=" +
            std::to_string(devMetrics.freeRunning.tokenExact) + ":dev_seq=" +
            std::to_string(devMetrics.freeRunning.sequenceExact) + ":hom_token=" +
            std::to_string(homogeneousMetrics.freeRunning.tokenExact));
      writeMetricRow(results, spec, condition, "AR_DEVELOPMENT_V3",
                     devMetrics, run, behavior);
      writeMetricRow(results, spec, condition, "HOMOGENEOUS_PHASE0",
                     homogeneousMetrics, run, behavior);
    }
  }
}

void runCycle2(const fs::path& root) {
  fs::create_directories(root);
  if (!ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8) ||
      !ar::hashMatchesPinned(ar::Partition::FINAL, 8))
    throw std::runtime_error("PINNED_DATASET_IDENTITY");
  const auto development = ar::cases(ar::Partition::DEVELOPMENT, 8);
  const auto homogeneous = si::homogeneousPhase0Cases();
  const auto auditConfig = configFor(19);
  const auto interleaved = cs::scheduleHistogram(
      auditConfig, cs::Condition::MixedInvariant, 4);

  std::ofstream schedule(root / "schedule-audit.csv");
  schedule << "condition,dose_percent,steps,supervised_rows,special_steps,"
              "schedule_hash,input_histogram,target_histogram,same_multiset_as_interleaved\n";
  for (const auto condition : {cs::Condition::MixedFirst,
                               cs::Condition::MixedLast}) {
    const auto histogram = cs::scheduleHistogram(auditConfig, condition, 4);
    if (!cs::sameHistogram(interleaved, histogram))
      throw std::runtime_error("CURRICULUM_MULTISET_MISMATCH");
    schedule << cs::conditionName(condition) << ",25,320,2560,"
             << histogram.specialExamples << ','
             << cs::scheduleHash(auditConfig, condition, 4) << ','
             << counts(histogram.input) << ',' << counts(histogram.target)
             << ",true\n";
  }

  std::ofstream results(root / "training-results.csv");
  results << "configuration_id,depth,seed,condition,dose_percent,dataset,steps,"
             "example_count,special_example_count,schedule_hash,initial_parameter_hash,"
             "final_parameter_hash,first_moment_hash,second_moment_hash,train_finite,"
             "final_train_loss,teacher_token_exact,teacher_token_total,teacher_nll,"
             "teacher_margin_q10,free_token_exact,free_token_total,free_sequence_exact,"
             "free_sequence_total,free_nll,median_first_error,free_margin_q10,"
             "qk_update_l2,v_update_l2,o_update_l2,ffn_update_l2,norm_update_l2,"
             "embedding_head_update_l2,attention_output_norm,attention_entropy,"
             "attention_self_mass,attention_previous_mass,attention_far_mass,"
             "attention_nonself_mass,ordinary_attention_active,all_finite\n";
  for (const auto& spec : kRuns) {
    const auto config = configFor(spec.depth);
    for (const auto condition : {cs::Condition::MixedFirst,
                                 cs::Condition::MixedLast}) {
      const auto run = cs::runTraining(config, spec.seed, condition, 4, 320);
      const auto devMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, development);
      const auto homogeneousMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, homogeneous);
      const auto behavior = cs::attentionBehavior(config, run.params, development);
      if (!run.finite || !devMetrics.teacherForced.finite ||
          !devMetrics.freeRunning.allFinite || !behavior.finite)
        throw std::runtime_error(std::string("NONFINITE_RUN:") + spec.id + ":" +
                                 cs::conditionName(condition));
      writeMetricRow(results, spec, condition, "AR_DEVELOPMENT_V3",
                     devMetrics, run, behavior);
      writeMetricRow(results, spec, condition, "HOMOGENEOUS_PHASE0",
                     homogeneousMetrics, run, behavior);
    }
  }
}

void runCycle3(const fs::path& root) {
  fs::create_directories(root);
  if (!ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8) ||
      !ar::hashMatchesPinned(ar::Partition::FINAL, 8))
    throw std::runtime_error("PINNED_DATASET_IDENTITY");
  const auto development = ar::cases(ar::Partition::DEVELOPMENT, 8);
  const auto homogeneous = si::homogeneousPhase0Cases();
  const auto auditConfig = configFor(19);
  std::ofstream schedule(root / "schedule-audit.csv");
  schedule << "condition,dose_percent,dose_denominator,steps,supervised_rows,"
              "special_steps,schedule_hash,input_histogram,target_histogram\n";
  for (const std::uint32_t denominator : {8u, 16u}) {
    const auto histogram = cs::scheduleHistogram(
        auditConfig, cs::Condition::MixedLast, denominator);
    const std::uint64_t expectedSpecial = denominator == 8u ? 40u : 20u;
    if (histogram.specialExamples != expectedSpecial)
      throw std::runtime_error("SMALL_DOSE_SPECIAL_COUNT");
    schedule << cs::conditionName(cs::Condition::MixedLast) << ','
             << number(100.0 / static_cast<double>(denominator)) << ','
             << denominator << ",320,2560," << histogram.specialExamples << ','
             << cs::scheduleHash(auditConfig, cs::Condition::MixedLast,
                                 denominator)
             << ',' << counts(histogram.input) << ',' << counts(histogram.target)
             << '\n';
  }

  std::ofstream results(root / "training-results.csv");
  results << "configuration_id,depth,seed,condition,dose_percent,dataset,steps,"
             "example_count,special_example_count,schedule_hash,initial_parameter_hash,"
             "final_parameter_hash,first_moment_hash,second_moment_hash,train_finite,"
             "final_train_loss,teacher_token_exact,teacher_token_total,teacher_nll,"
             "teacher_margin_q10,free_token_exact,free_token_total,free_sequence_exact,"
             "free_sequence_total,free_nll,median_first_error,free_margin_q10,"
             "qk_update_l2,v_update_l2,o_update_l2,ffn_update_l2,norm_update_l2,"
             "embedding_head_update_l2,attention_output_norm,attention_entropy,"
             "attention_self_mass,attention_previous_mass,attention_far_mass,"
             "attention_nonself_mass,ordinary_attention_active,all_finite\n";
  for (const auto& spec : kRuns) {
    const auto config = configFor(spec.depth);
    for (const std::uint32_t denominator : {8u, 16u}) {
      const auto run = cs::runTraining(
          config, spec.seed, cs::Condition::MixedLast, denominator, 320);
      const auto devMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, development);
      const auto homogeneousMetrics = amc::evaluate(
          config, run.params, amc::Pattern::Learned, homogeneous);
      const auto behavior = cs::attentionBehavior(config, run.params, development);
      if (!run.finite || !devMetrics.teacherForced.finite ||
          !devMetrics.freeRunning.allFinite || !behavior.finite)
        throw std::runtime_error(std::string("NONFINITE_RUN:") + spec.id);
      const double dose = 100.0 / static_cast<double>(denominator);
      writeMetricRow(results, spec, cs::Condition::MixedLast,
                     "AR_DEVELOPMENT_V3", devMetrics, run, behavior, dose);
      writeMetricRow(results, spec, cs::Condition::MixedLast,
                     "HOMOGENEOUS_PHASE0", homogeneousMetrics, run, behavior,
                     dose);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      std::string error;
      if (!cs::selfTest(&error)) {
        std::cerr << "FAIL " << error << '\n';
        return 1;
      }
      std::cout << "PASS context_supervision_stability_self_test\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--cycle1") {
      runCycle1(fs::path(argv[2]));
      std::cout << "PASS context_supervision_stability_cycle1\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--cycle2") {
      runCycle2(fs::path(argv[2]));
      std::cout << "PASS context_supervision_stability_cycle2\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--cycle3") {
      runCycle3(fs::path(argv[2]));
      std::cout << "PASS context_supervision_stability_cycle3\n";
      return 0;
    }
    std::cerr << "usage: context_supervision_stability --self-test | --cycle1 <report-root> | --cycle2 <report-root> | --cycle3 <report-root>\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
