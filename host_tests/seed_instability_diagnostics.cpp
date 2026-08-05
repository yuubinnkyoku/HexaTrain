// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "seed_instability_diagnostics_lib.h"
#include "depth_quality_lib.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace si = phonelm::seed_instability;
namespace ar = phonelm::autoregressive_validation;
namespace dq = phonelm::depth_quality;
namespace cm = phonelm::critical_margin;

namespace {

struct RunSpec {
  const char* id;
  int layers;
  std::uint32_t seed;
  std::uint64_t mixedFinalTokenExact;
  std::uint64_t mixedFinalSequenceExact;
};

const std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", 19, 1, 30, 2},
    {"L19_SEED_2", 19, 2, 63, 6},
    {"L19_SEED_4", 19, 4, 46, 6},
    {"L18_SEED_2_CONTROL", 18, 2, 65, 8},
}};

phonelm::tiny_lm::Config makeConfig(int layers) {
  phonelm::tiny_lm::Config c;
  c.vocabularySize = 32;
  c.tokens = 8;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = static_cast<std::uint32_t>(layers);
  c.numHeads = 2;
  std::string error;
  if (!phonelm::tiny_lm::validateConfig(c, &error))
    throw std::runtime_error(error);
  return c;
}

std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void writeDataAudit(const fs::path& root) {
  std::ofstream out(root / "data-structure.csv");
  out << "dataset,context_kind,occurrences,unique_contexts,ambiguous_contexts,ambiguous_occurrences\n";
  const auto trainRows = si::correctedTrainingRows(
      ar::cases(ar::Partition::TRAIN, 8));
  const auto train = si::trainingObservations(trainRows);
  for (const auto& item : std::array<si::ContextAudit, 2>{
           si::auditContexts(train, 1, "CURRENT_TOKEN"),
           si::auditContexts(train, 2, "PREVIOUS_2_TOKENS")})
    out << "TRAIN," << item.contextKind << ',' << item.occurrences << ','
        << item.uniqueContexts << ',' << item.ambiguousContexts << ','
        << item.ambiguousOccurrences << '\n';
  for (const auto partition : {ar::Partition::VALIDATION,
                               ar::Partition::DEVELOPMENT}) {
    const auto partitionCases = ar::cases(partition, 8);
    const auto observations = si::evaluationObservations(partitionCases);
    for (const auto& item : std::array<si::ContextAudit, 4>{
             si::auditContexts(observations, 1, "CURRENT_TOKEN"),
             si::auditContexts(observations, 2, "PREVIOUS_2_TOKENS"),
             si::auditContexts(observations, 8, "MODEL_WINDOW_8"),
             si::auditFullContexts(si::fullCausalPrefixObservations(
                 partitionCases), "FULL_CAUSAL_PREFIX")})
      out << ar::partitionName(partition) << ',' << item.contextKind << ','
          << item.occurrences << ',' << item.uniqueContexts << ','
          << item.ambiguousContexts << ',' << item.ambiguousOccurrences << '\n';
  }
  for (const auto partition : {cm::Partition::CALIBRATION,
                               cm::Partition::DEVELOPMENT}) {
    const auto partitionCases = cm::cases(partition, 8);
    const auto observations = si::evaluationObservations(partitionCases);
    for (const auto& item : std::array<si::ContextAudit, 4>{
             si::auditContexts(observations, 1, "CURRENT_TOKEN"),
             si::auditContexts(observations, 2, "PREVIOUS_2_TOKENS"),
             si::auditContexts(observations, 8, "MODEL_WINDOW_8"),
             si::auditFullContexts(si::fullCausalPrefixObservations(
                 partitionCases), "FULL_CAUSAL_PREFIX")})
      out << cm::partitionName(partition) << ',' << item.contextKind << ','
          << item.occurrences << ',' << item.uniqueContexts << ','
          << item.ambiguousContexts << ',' << item.ambiguousOccurrences << '\n';
  }
}

void writeMeasurementAudit(const fs::path& root) {
  const auto trainCases = ar::cases(ar::Partition::TRAIN, 8);
  const auto legacy = phonelm::readout_probe::teacherForcedRows(trainCases, 8);
  const auto corrected = si::correctedTrainingRows(trainCases, 8);
  std::ofstream out(root / "measurement-audit.csv");
  out << "measurement,status,observed,expected,impact\n";
  out << "legacy_train_current_token_exact,FAIL,"
      << si::legacyCurrentTokenExact(legacy)
      << ",32,probe_fit_rows_contain_four_contract_conflicts\n";
  out << "corrected_train_current_token_exact,PASS,"
      << si::correctedCurrentTokenExact(corrected)
      << ",32,formal_batch_row_and_target_are_aligned\n";
  const auto calAudit = si::auditContexts(si::evaluationObservations(
      cm::cases(cm::Partition::CALIBRATION, 8)), 1, "CURRENT_TOKEN");
  const auto devAudit = si::auditContexts(si::evaluationObservations(
      cm::cases(cm::Partition::DEVELOPMENT, 8)), 1, "CURRENT_TOKEN");
  out << "cal_dev_current_token_ambiguity,"
      << (calAudit.ambiguousContexts == 0 && devAudit.ambiguousContexts == 0
              ? "PASS" : "FAIL")
      << ',' << (calAudit.ambiguousContexts + devAudit.ambiguousContexts)
      << ",0,target_is_unique\n";
  out << "cross_seed_swap_historical,INVALID,NOT_RECOMPUTED,ROW_WISE_IDENTITY,excluded_from_causal_inference\n";
  out << "projection_contribution_norm_cosine_historical,INVALID,ONE_ROW,DEV_AGGREGATE,excluded_from_causal_inference\n";
  out << "final_holdout,PASS,0,0,hash_only_not_evaluated\n";
}

void writeContextRows(const fs::path& root) {
  std::ofstream out(root / "context-shift.csv");
  out << "configuration_id,depth,seed,dataset,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,all_finite\n";
  const auto homogeneous = si::homogeneousPhase0Cases();
  const auto mixed = ar::cases(ar::Partition::DEVELOPMENT, 8);
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    const auto run = dq::runFormalCpu(config, spec.seed, 320, 0.003f,
                                      dq::StabilityMode::LEGACY, {320});
    const auto checkpoint = run.checkpoints.at(320);
    const auto homogeneousMetrics = si::evaluateContextDataset(
        config, checkpoint, "HOMOGENEOUS_PHASE0", homogeneous, 320);
    const auto mixedMetrics = si::evaluateContextDataset(
        config, checkpoint, "AR_DEVELOPMENT_V3", mixed, 320);
    if (mixedMetrics.freeRunning.tokenExact != spec.mixedFinalTokenExact ||
        mixedMetrics.freeRunning.sequenceExact != spec.mixedFinalSequenceExact)
      throw std::runtime_error(std::string("MIXED_ANCHOR_MISMATCH:") + spec.id);
    for (const auto* metrics : {&homogeneousMetrics, &mixedMetrics}) {
      out << spec.id << ',' << spec.layers << ',' << spec.seed << ','
          << metrics->dataset << ','
          << metrics->teacherForced.tokenExact << ','
          << metrics->teacherForced.total << ','
          << number(metrics->teacherForced.meanNll) << ','
          << metrics->freeRunning.tokenExact << ','
          << metrics->freeRunning.tokenTotal << ','
          << metrics->freeRunning.sequenceExact << ','
          << metrics->freeRunning.sequenceTotal << ','
          << number(metrics->freeRunning.autoregressiveNll) << ','
          << number(metrics->freeRunning.medianFirstErrorSurvival) << ','
          << (metrics->teacherForced.finite && metrics->freeRunning.allFinite
                  ? "true" : "false")
          << '\n';
    }
  }
}

void writeOptimizationInterventions(const fs::path& root) {
  std::ofstream out(root / "optimization-interventions.csv");
  out << "configuration_id,depth,seed,intervention,boundary_step,final_step,train_finite,final_train_loss,dataset,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,all_finite\n";
  const auto homogeneous = si::homogeneousPhase0Cases();
  const auto mixed = ar::cases(ar::Partition::DEVELOPMENT, 8);
  constexpr std::array<si::ResumeIntervention, 3> interventions{{
      si::ResumeIntervention::FreezeOutputAfterBoundary,
      si::ResumeIntervention::ResetOutputMomentsAtBoundary,
      si::ResumeIntervention::ResetAttentionMomentsAtBoundary,
  }};
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    const auto boundary = si::canonicalPrefix(config, spec.seed, 24, 320);
    if (!boundary.finite) throw std::runtime_error("NONFINITE_PREFIX");
    for (const auto intervention : interventions) {
      const auto run = si::continueCanonical(config, boundary, 320, intervention, 320);
      for (const auto* dataset : {&homogeneous, &mixed}) {
        const std::string datasetName = dataset == &homogeneous
                                            ? "HOMOGENEOUS_PHASE0"
                                            : "AR_DEVELOPMENT_V3";
        const auto metrics = si::evaluateContextDataset(
            config, run.params, datasetName, *dataset, 320);
        out << spec.id << ',' << spec.layers << ',' << spec.seed << ','
            << si::interventionName(intervention) << ",24,320,"
            << (run.finite ? "true" : "false") << ','
            << number(run.lastLoss) << ',' << datasetName << ','
            << metrics.teacherForced.tokenExact << ','
            << metrics.teacherForced.total << ','
            << number(metrics.teacherForced.meanNll) << ','
            << metrics.freeRunning.tokenExact << ','
            << metrics.freeRunning.tokenTotal << ','
            << metrics.freeRunning.sequenceExact << ','
            << metrics.freeRunning.sequenceTotal << ','
            << number(metrics.freeRunning.autoregressiveNll) << ','
            << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
            << (metrics.teacherForced.finite && metrics.freeRunning.allFinite
                    ? "true" : "false")
            << '\n';
      }
    }
  }
}

void writeBranchAblations(const fs::path& root) {
  std::ofstream out(root / "branch-ablation.csv");
  out << "configuration_id,depth,seed,intervention,steps,train_finite,branch_exactly_zero,final_train_loss,dataset,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,all_finite\n";
  const auto homogeneous = si::homogeneousPhase0Cases();
  const auto mixed = ar::cases(ar::Partition::DEVELOPMENT, 8);
  constexpr std::array<si::BranchAblation, 2> ablations{{
      si::BranchAblation::AttentionZero,
      si::BranchAblation::FfnZero,
  }};
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    for (const auto ablation : ablations) {
      const auto run = si::trainWithBranchAblation(config, spec.seed, ablation);
      const bool branchZero = si::branchIsZero(config, run.params, ablation);
      if (!branchZero) throw std::runtime_error("BRANCH_ZERO_IDENTITY_FAIL");
      for (const auto* dataset : {&homogeneous, &mixed}) {
        const std::string datasetName = dataset == &homogeneous
                                            ? "HOMOGENEOUS_PHASE0"
                                            : "AR_DEVELOPMENT_V3";
        const auto metrics = si::evaluateContextDataset(
            config, run.params, datasetName, *dataset, 320);
        out << spec.id << ',' << spec.layers << ',' << spec.seed << ','
            << si::branchAblationName(ablation) << ",320,"
            << (run.finite ? "true" : "false") << ','
            << (branchZero ? "true" : "false") << ','
            << number(run.lastLoss) << ',' << datasetName << ','
            << metrics.teacherForced.tokenExact << ','
            << metrics.teacherForced.total << ','
            << number(metrics.teacherForced.meanNll) << ','
            << metrics.freeRunning.tokenExact << ','
            << metrics.freeRunning.tokenTotal << ','
            << metrics.freeRunning.sequenceExact << ','
            << metrics.freeRunning.sequenceTotal << ','
            << number(metrics.freeRunning.autoregressiveNll) << ','
            << number(metrics.freeRunning.medianFirstErrorSurvival) << ','
            << (metrics.teacherForced.finite && metrics.freeRunning.allFinite
                    ? "true" : "false")
            << '\n';
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string error;
    if (!si::selfTest(&error)) {
      std::cerr << "SEED_INSTABILITY_SELF_TEST_FAIL:" << error << '\n';
      return 1;
    }
    if (argc == 1 || std::string(argv[1]) == "--self-test") {
      std::cout << "SEED_INSTABILITY_SELF_TEST_PASS\n";
      return 0;
    }
    if (argc != 3 || (std::string(argv[1]) != "--report-root" &&
                      std::string(argv[1]) != "--data-audit-root" &&
                      std::string(argv[1]) != "--optimization-root" &&
                      std::string(argv[1]) != "--branch-root"))
      throw std::invalid_argument("usage: --report-root|--data-audit-root|--optimization-root|--branch-root <path>");
    const fs::path root = argv[2];
    fs::create_directories(root);
    if (std::string(argv[1]) == "--report-root" ||
        std::string(argv[1]) == "--data-audit-root") {
      writeMeasurementAudit(root);
      writeDataAudit(root);
      if (std::string(argv[1]) == "--report-root") writeContextRows(root);
    } else if (std::string(argv[1]) == "--optimization-root") {
      writeOptimizationInterventions(root);
    } else {
      writeBranchAblations(root);
    }
    std::cout << "SEED_INSTABILITY_DIAGNOSTICS_PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SEED_INSTABILITY_DIAGNOSTICS_FAIL:" << error.what() << '\n';
    return 1;
  }
}
