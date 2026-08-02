// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only CPU replay and autoregressive validation probe.  This executable
// intentionally writes only aggregate, public-safe diagnostics.  Checkpoint
// bytes, tensor values, device identifiers and source paths never enter a
// report.
#include "depth_quality_lib.h"
#include "qnn/qnn_first_nonfinite_diagnostics.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;
namespace ff = phonelm::qnn::first_nonfinite;
namespace tiny = phonelm::tiny_lm;
namespace vs = phonelm::validation_selection;

namespace {

constexpr int kSteps = 320;
const std::vector<int> kEvaluationSteps(vs::evaluationSteps().begin(),
                                        vs::evaluationSteps().end());

struct RunSpec {
  const char* publicId;
  const char* checkpointDirectory;
  int layers;
  std::uint32_t seed;
};

const std::array<RunSpec, 4> kRuns{{
    {"L19_SEED_1", "l19s1", 19, 1},
    {"L19_SEED_2", "l19s2", 19, 2},
    {"L19_SEED_4", "l19s4", 19, 4},
    {"L18_SEED_2_CONTROL", "l18s2", 18, 2},
}};

struct CpuRecord {
  RunSpec spec{};
  dq::AutoregressiveSelectedRun selected;
  std::map<int, ar::Metrics> validation;
  ar::Metrics developmentSelected;
  ar::Metrics developmentFinal;
  ar::Metrics finalSelected;
  ar::Metrics finalFinal;
  bool developmentEvaluated = false;
  bool finalEvaluated = false;
};

tiny::Config makeConfig(int layers) {
  tiny::Config c;
  c.tokens = 8;
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
  std::string escaped = value;
  std::string result = "\"";
  for (const char ch : escaped) {
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

std::vector<std::uint8_t> readBytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t shapeElements(const std::vector<std::uint32_t>& shape) {
  std::size_t count = 1;
  for (const auto dimension : shape) count *= dimension;
  return count;
}

bool bindCheckpoint(const ff::Checkpoint& checkpoint, const tiny::Config& config,
                    dq::Params* parameters, std::string* error) {
  if (!parameters) {
    if (error) *error = "NULL_PARAMETERS";
    return false;
  }
  *parameters = tiny::initialParameters(config, checkpoint.seed);
  const auto expected = tiny::parameterRegistry(*parameters);
  if (checkpoint.registry.size() != expected.size()) {
    if (error) *error = "REGISTRY_COUNT";
    return false;
  }
  if (checkpoint.registryHash != ff::hashRegistry(checkpoint.registry)) {
    if (error) *error = "REGISTRY_HASH";
    return false;
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& source = checkpoint.registry[i];
    const auto& target = expected[i];
    if (source.name != target.name) {
      if (error) *error = "REGISTRY_NAME";
      return false;
    }
    std::vector<std::uint32_t> expectedShape;
    if (source.name == "token_embedding") {
      expectedShape = {config.vocabularySize, config.dimension};
    } else if (source.name == "output_projection") {
      expectedShape = {config.dimension, config.vocabularySize};
    } else if (source.name.find("norm") != std::string::npos) {
      expectedShape = {config.dimension};
    } else if (source.name.find("ffn_w1") != std::string::npos) {
      expectedShape = {config.dimension, config.feedForwardDimension};
    } else if (source.name.find("ffn_w2") != std::string::npos) {
      expectedShape = {config.feedForwardDimension, config.dimension};
    } else {
      expectedShape = {config.dimension, config.dimension};
    }
    const std::size_t count = target.values->size();
    if (source.shape != expectedShape || shapeElements(source.shape) != count) {
      if (error) *error = "REGISTRY_SHAPE";
      return false;
    }
    if (offset > checkpoint.parameters.size() ||
        count > checkpoint.parameters.size() - offset) {
      if (error) *error = "PARAMETER_LENGTH";
      return false;
    }
    std::copy_n(checkpoint.parameters.begin() + offset, count,
                const_cast<std::vector<float>*>(target.values)->begin());
    offset += count;
  }
  if (offset != checkpoint.parameters.size()) {
    if (error) *error = "PARAMETER_TRAILING";
    return false;
  }
  const auto finiteValues = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finiteValues(checkpoint.parameters) || !finiteValues(checkpoint.adamM) ||
      !finiteValues(checkpoint.adamV)) {
    if (error) *error = "NONFINITE_CHECKPOINT_STATE";
    return false;
  }
  return true;
}

bool checkpointConfigMatches(const ff::Checkpoint& checkpoint,
                             const tiny::Config& expected) {
  const auto& c = checkpoint.config;
  return c.tokens == expected.tokens && c.vocabularySize == expected.vocabularySize &&
         c.dimension == expected.dimension &&
         c.feedForwardDimension == expected.feedForwardDimension &&
         c.numLayers == expected.numLayers && c.numHeads == expected.numHeads &&
         c.epsilon == expected.epsilon && c.learningRate == 0.003f &&
         c.beta1 == 0.9f && c.beta2 == 0.999f && c.adamEpsilon == 1.0e-8f &&
         c.clipThreshold == 0.0f &&
         c.trainingStabilityMode == 0 && c.depthPairInitMode == 0 &&
         c.totalSteps == kSteps;
}

void writeMetrics(std::ofstream& output, const std::string& source,
                  const RunSpec& spec, int step, const std::string& role,
                  ar::Partition partition, const ar::Metrics& metrics) {
  row(output, {source, spec.publicId, text(spec.layers), text(spec.seed), text(step),
               role, ar::partitionName(partition), metrics.allFinite ? "true" : "false",
               number(metrics.autoregressiveNll), number(metrics.teacherForcedNll),
               number(metrics.autoregressiveTeacherGap), text(metrics.tokenExact),
               text(metrics.tokenTotal), text(metrics.sequenceExact),
               text(metrics.sequenceTotal), number(metrics.meanFirstErrorPosition),
               text(metrics.postErrorRecoveryTokens)});
}

void writeFinalMetrics(std::ofstream& output, const std::string& source,
                       const RunSpec& spec, int step, const std::string& role,
                       const ar::Metrics& metrics) {
  row(output, {source, spec.publicId, text(spec.layers), text(spec.seed), text(step),
               role, ar::partitionName(ar::Partition::FINAL), "EVALUATED",
               metrics.allFinite ? "true" : "false",
               number(metrics.autoregressiveNll), number(metrics.teacherForcedNll),
               number(metrics.autoregressiveTeacherGap), text(metrics.tokenExact),
               text(metrics.tokenTotal), text(metrics.sequenceExact),
               text(metrics.sequenceTotal), number(metrics.meanFirstErrorPosition),
               text(metrics.postErrorRecoveryTokens)});
}

void writeReplayMetrics(std::ofstream& output, const RunSpec& spec, int step,
                        const std::string& status, const ar::Metrics& metrics,
                        const std::string& parameterNorm = "NOT_AVAILABLE") {
  row(output, {"LEGACY_HTP_CHECKPOINT_REPLAY", spec.publicId, text(spec.layers),
               text(spec.seed), text(step), status,
               metrics.allFinite ? "true" : "false",
               number(metrics.autoregressiveNll), number(metrics.teacherForcedNll),
               number(metrics.autoregressiveTeacherGap), text(metrics.tokenExact),
               text(metrics.tokenTotal), text(metrics.sequenceExact),
               text(metrics.sequenceTotal), number(metrics.meanFirstErrorPosition),
               text(metrics.postErrorRecoveryTokens), "NOT_AVAILABLE",
               "NOT_AVAILABLE", parameterNorm, "NOT_AVAILABLE"});
}

bool caseCollapse(const ar::Metrics& selected, const ar::Metrics& final) {
  if (selected.perCase.size() != final.perCase.size()) return true;
  for (std::size_t i = 0; i < final.perCase.size(); ++i) {
    if (final.perCase[i].tokenExact > 0 && selected.perCase[i].tokenExact == 0)
      return true;
  }
  return false;
}

bool allValidationFinite(const CpuRecord& record) {
  if (record.validation.size() != kEvaluationSteps.size()) return false;
  for (const int step : kEvaluationSteps) {
    const auto found = record.validation.find(step);
    if (found == record.validation.end() || !ar::finite(found->second))
      return false;
  }
  return true;
}

ar::Metrics pool(const std::vector<ar::Metrics>& values) {
  return dq::poolAutoregressiveMetrics(values);
}

bool developmentGate(const std::vector<CpuRecord>& records) {
  const auto l19 = std::vector<const CpuRecord*>{&records[0], &records[1], &records[2]};
  const CpuRecord* l18 = &records[3];
  bool finiteRuns = true;
  for (const auto* record : l19)
    finiteRuns = finiteRuns && allValidationFinite(*record) &&
                 ar::finite(record->developmentSelected) &&
                 ar::finite(record->developmentFinal);
  finiteRuns = finiteRuns && allValidationFinite(*l18) &&
               ar::finite(l18->developmentSelected) &&
               ar::finite(l18->developmentFinal);
  if (!finiteRuns) return false;
  if (!ar::strict(l19[1]->developmentSelected, l19[1]->developmentFinal))
    return false;
  if (!ar::nonworse(pool({l19[0]->developmentSelected,
                          l19[1]->developmentSelected,
                          l19[2]->developmentSelected}),
                    pool({l19[0]->developmentFinal, l19[1]->developmentFinal,
                          l19[2]->developmentFinal})))
    return false;
  if (!ar::nonworse(l18->developmentSelected, l18->developmentFinal)) return false;
  int earlyNonworse = 0;
  for (const auto* record : l19) {
    if (record->selected.selectedStep < kSteps) {
      if (!ar::nonworse(record->developmentSelected, record->developmentFinal))
        return false;
      ++earlyNonworse;
    }
    if (caseCollapse(record->developmentSelected, record->developmentFinal))
      return false;
  }
  return earlyNonworse >= 2 &&
         !caseCollapse(l18->developmentSelected, l18->developmentFinal);
}

bool finalGate(const std::vector<CpuRecord>& records) {
  const auto& s1 = records[0];
  const auto& s2 = records[1];
  const auto& s4 = records[2];
  const auto& l18 = records[3];
  if (!s1.finalEvaluated || !s2.finalEvaluated || !s4.finalEvaluated ||
      !l18.finalEvaluated)
    return false;
  if (!ar::strict(s2.finalSelected, s2.finalFinal)) return false;
  if (!ar::strict(pool({s1.finalSelected, s2.finalSelected, s4.finalSelected}),
                  pool({s1.finalFinal, s2.finalFinal, s4.finalFinal})))
    return false;
  return ar::nonworse(l18.finalSelected, l18.finalFinal);
}

std::string trajectoryClass(const CpuRecord& record) {
  if (!allValidationFinite(record)) return "INCONCLUSIVE";
  ar::Metrics best;
  int bestStep = kSteps;
  for (const auto& entry : record.validation) {
    if (ar::better(entry.second, entry.first, best, bestStep)) {
      best = entry.second;
      bestStep = entry.first;
    }
  }
  const auto& final = record.validation.at(kSteps);
  if (!ar::finite(best)) return "INCONCLUSIVE";
  int signChanges = 0;
  int comparableDeltas = 0;
  double previousDelta = 0.0;
  bool havePreviousDelta = false;
  for (std::size_t i = 1; i < kEvaluationSteps.size(); ++i) {
    const double delta =
        record.validation.at(kEvaluationSteps[i]).autoregressiveNll -
        record.validation.at(kEvaluationSteps[i - 1]).autoregressiveNll;
    if (std::abs(delta) <= ar::kNllTieTolerance) continue;
    if (havePreviousDelta) {
      ++comparableDeltas;
      if (delta * previousDelta < 0.0) ++signChanges;
    }
    previousDelta = delta;
    havePreviousDelta = true;
  }
  if (comparableDeltas > 0 &&
      static_cast<double>(signChanges) / comparableDeltas > 0.6)
    return "AR_OSCILLATES";
  if (bestStep == kSteps && final.tokenExact == 0) return "AR_NEVER_GENERALIZES";
  if (bestStep < kSteps && !ar::nonworse(final, best))
    return "AR_GENERALIZES_THEN_REGRESSES";
  if (bestStep < kSteps) return "AR_GENERALIZES_AND_STAYS";
  if (final.tokenExact == 0) return "AR_NEVER_GENERALIZES";
  return "TRAIN_AR_GAP_ONLY";
}

void writeDataset(const fs::path& root) {
  std::ofstream partitions(root / "dataset-partitions.csv");
  row(partitions, {"partition", "case_id", "domain", "active_family",
                   "distractor_family", "active_phase", "distractor_phase",
                   "active_suffix_length", "rollout_length", "initial_prefix",
                   "targets"});
  for (const auto partition : ar::kPartitions) {
    for (const auto& item : ar::cases(partition, 8)) {
      auto list = [](const std::vector<std::uint32_t>& values) {
        std::ostringstream out;
        for (std::size_t i = 0; i < values.size(); ++i) {
          if (i) out << ':';
          out << values[i];
        }
        return out.str();
      };
      row(partitions, {ar::partitionName(partition), item.id, item.domain,
                       text(item.activeFamily), text(item.distractorFamily),
                       text(item.activePhase), text(item.distractorPhase),
                       text(item.activeSuffixLength), text(item.rolloutLength),
                       list(item.initialPrefix), list(item.targets)});
    }
  }
  std::ofstream overlaps(root / "dataset-overlap.csv");
  row(overlaps, {"left", "right", "case_id_overlap", "initial_prefix_overlap",
                 "full_sequence_overlap", "unique_transition_overlap",
                 "transition_occurrence_multiset_overlap"});
  for (std::size_t i = 0; i < ar::kPartitions.size(); ++i)
    for (std::size_t j = i + 1; j < ar::kPartitions.size(); ++j) {
      const auto stats = ar::overlap(ar::kPartitions[i], ar::kPartitions[j], 8);
      row(overlaps, {ar::partitionName(ar::kPartitions[i]),
                     ar::partitionName(ar::kPartitions[j]), text(stats.caseId),
                     text(stats.initialPrefix), text(stats.fullSequence),
                     text(stats.uniqueTargetTransitions),
                     text(stats.targetTransitionOccurrences)});
    }
  std::ofstream hashes(root / "dataset-hashes.csv");
  row(hashes, {"partition", "schema", "generator_domain", "hash", "case_count",
               "target_transition_occurrences", "unique_target_transitions"});
  for (const auto partition : ar::kPartitions)
    row(hashes, {ar::partitionName(partition), ar::schemaName(), ar::domain(partition),
                 ar::partitionHash(partition, 8),
                 text(ar::cases(partition, 8).size()),
                 text(ar::targetTransitionOccurrenceCount(partition, 8)),
                 text(ar::uniqueTargetTransitionCount(partition, 8))});
}

void writeManifest(const fs::path& root, bool developmentPass,
                   bool finalPass) {
  std::ofstream output(root / "manifest.json");
  output << "{\n"
         << "  \"schema\": \"" << ar::schemaName() << "\",\n"
         << "  \"schema_version\": " << ar::schemaVersion() << ",\n"
         << "  \"candidate_selection_mode\": \"BEST_AR_VALIDATION_V1\",\n"
         << "  \"adopted_selection_mode\": \""
         << (finalPass ? "BEST_AR_VALIDATION_V1" : "NONE") << "\",\n"
         << "  \"development_gate\": \""
         << (developmentPass ? "PASS" : "REJECT") << "\",\n"
         << "  \"final_holdout_gate\": \""
         << (developmentPass ? (finalPass ? "PASS" : "REJECT") : "NOT_RUN")
         << "\",\n"
         << "  \"selection_tolerance\": " << number(ar::kNllTieTolerance) << ",\n"
         << "  \"configurations\": [\"T8/D16/FFN32/L19/H2\","
            "\"T8/D16/FFN32/L18/H2_CONTROL\"],\n"
         << "  \"host_checkpoint_retention\": "
            "\"EXPERIMENTAL_ALL_23_NOT_PHASE9_NATIVE\",\n"
         << "  \"legacy_replay_optimizer_metrics\": "
            "\"NOT_AVAILABLE_CHECKPOINT_SCHEMA\",\n"
         << "  \"evaluation_steps\": [";
  for (std::size_t i = 0; i < kEvaluationSteps.size(); ++i) {
    if (i) output << ',';
    output << kEvaluationSteps[i];
  }
  output << "],\n  \"final_holdout_policy\": \"NOT_OPENED_UNLESS_DEVELOPMENT_PASS\"\n}\n";
}

void selfTest() {
  std::string error;
  assert(ar::validatePartitions(8, &error));
  assert(ar::hashMatchesPinned(ar::Partition::TRAIN));
  assert(ar::hashMatchesPinned(ar::Partition::VALIDATION));
  assert(ar::hashMatchesPinned(ar::Partition::DEVELOPMENT));
  assert(ar::hashMatchesPinned(ar::Partition::FINAL));
  assert(ar::cases(ar::Partition::TRAIN).size() == 4);
  assert(ar::cases(ar::Partition::VALIDATION).size() == 24);
  assert(ar::cases(ar::Partition::DEVELOPMENT).size() == 24);
  assert(ar::cases(ar::Partition::FINAL).size() == 24);
  assert(ar::targetTransitionOccurrenceCount(ar::Partition::TRAIN) == 32);
  assert(ar::verifyFreshTransitionContract(ar::Partition::VALIDATION));
  assert(ar::verifyFreshTransitionContract(ar::Partition::DEVELOPMENT));
  assert(ar::verifyFreshTransitionContract(ar::Partition::FINAL));
  assert(kEvaluationSteps.size() == 23 && kEvaluationSteps.front() == 0 &&
         kEvaluationSteps.back() == 320);
  const std::vector<int> expectedSteps{
      0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48,
      56, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320};
  assert(kEvaluationSteps == expectedSteps);

  ar::Metrics incumbent, candidate;
  incumbent.allFinite = candidate.allFinite = true;
  incumbent.autoregressiveNll = candidate.autoregressiveNll = 1.0;
  incumbent.teacherForcedNll = candidate.teacherForcedNll = 1.0;
  incumbent.autoregressiveTeacherGap = candidate.autoregressiveTeacherGap = 0.0;
  incumbent.tokenTotal = candidate.tokenTotal = 10;
  incumbent.sequenceTotal = candidate.sequenceTotal = 2;
  candidate.tokenExact = 8;
  incumbent.tokenExact = 7;
  candidate.sequenceExact = incumbent.sequenceExact = 1;
  assert(ar::better(candidate, 4, incumbent, 8));
  assert(ar::strictImprovement(candidate, incumbent));
  assert(!ar::strictImprovement(incumbent, candidate));
  auto syntheticRecord = [](const char* publicId, int layers,
                            std::uint32_t seed) {
    CpuRecord record;
    record.spec = {publicId, "unused", layers, seed};
    return record;
  };
  const std::vector<CpuRecord> synthetic{
      syntheticRecord("l19s1", 19, 1), syntheticRecord("l19s2", 19, 2),
      syntheticRecord("l19s4", 19, 4), syntheticRecord("l18s2", 18, 2)};
  assert(!finalGate(synthetic));
  // This is the fail-closed guard used by the normal path: a rejected
  // development gate must not invoke the final evaluator at all.
  assert(!developmentGate(synthetic));
  std::cout << "autoregressive_validation_probe_self_test=PASS\n";
}

int run(const fs::path& outputRoot, const fs::path& checkpointRoot) {
  fs::create_directories(outputRoot);
  writeDataset(outputRoot);
  std::ofstream replay(outputRoot / "checkpoint-replay.csv");
  row(replay, {"source", "configuration_id", "depth", "seed", "step", "status",
               "all_finite",
               "ar_rollout_nll", "teacher_forced_nll", "teacher_forced_gap",
               "token_exact", "token_total", "sequence_exact", "sequence_total",
               "mean_first_error_position", "post_error_recovery_tokens",
               "training_loss", "gradient_norm", "parameter_norm",
               "update_to_parameter"});
  std::ofstream validation(outputRoot / "ar-validation-metrics.csv");
  row(validation, {"source", "configuration_id", "depth", "seed", "step", "role", "partition",
                    "all_finite", "ar_rollout_nll", "teacher_forced_nll",
                    "teacher_forced_gap", "token_exact", "token_total", "sequence_exact",
                    "sequence_total", "mean_first_error_position",
                    "post_error_recovery_tokens"});
  std::ofstream trajectory(outputRoot / "checkpoint-trajectory.csv");
  row(trajectory, {"source", "configuration_id", "depth", "seed", "step", "loss", "accuracy",
                   "gradient_norm", "parameter_norm", "update_norm",
                   "update_to_parameter"});
  std::vector<CpuRecord> records;
  std::size_t replayAvailable = 0;
  std::size_t replayMissing = 0;
  for (const auto& spec : kRuns) {
    const auto config = makeConfig(spec.layers);
    const fs::path dir = checkpointRoot / spec.checkpointDirectory / "checkpoints";
    for (const int step : kEvaluationSteps) {
      const fs::path path = dir / ("ckpt_seed" + text(spec.seed) + "_step" +
                                   text(step) + ".bin");
      if (!fs::is_regular_file(path)) {
        ++replayMissing;
        ar::Metrics unavailable;
        writeReplayMetrics(replay, spec, step, "NOT_AVAILABLE", unavailable);
        continue;
      }
      const auto bytes = readBytes(path);
      if (bytes.empty())
        throw std::runtime_error("checkpoint file is empty");
      ff::Checkpoint checkpoint;
      std::string error;
      if (!ff::decodeCheckpoint(bytes, &checkpoint, &error))
        throw std::runtime_error("checkpoint decode failed: " + error);
      if (checkpoint.seed != spec.seed)
        throw std::runtime_error("checkpoint seed mismatch");
      if (checkpoint.completedStep != static_cast<std::uint32_t>(step))
        throw std::runtime_error("checkpoint step mismatch");
      if (checkpoint.nextOptimizerStep != static_cast<std::uint32_t>(step + 1) ||
          checkpoint.deterministicState !=
              "fixed_language_batch=" + text(step % 4))
        throw std::runtime_error("checkpoint deterministic state mismatch");
      if (!checkpointConfigMatches(checkpoint, config))
        throw std::runtime_error("checkpoint configuration mismatch");
      dq::Params parameters;
      if (!bindCheckpoint(checkpoint, config, &parameters, &error))
        throw std::runtime_error("checkpoint bind failed: " + error);
      const auto expectedBatch = dq::formalBatch(
          config, static_cast<std::uint32_t>(step % 4), 0);
      if (checkpoint.input != expectedBatch.first ||
          checkpoint.target != expectedBatch.second)
        throw std::runtime_error("checkpoint deterministic batch mismatch");
      const auto metrics = dq::autoregressiveEvaluation(
          config, parameters, ar::Partition::VALIDATION);
      ++replayAvailable;
      writeReplayMetrics(replay, spec, step, "REPLAY", metrics,
                         number(dq::registryNorm(parameters)));
      writeMetrics(validation, "LEGACY_HTP_CHECKPOINT_REPLAY", spec, step, "REPLAY",
                   ar::Partition::VALIDATION, metrics);
    }
  }
  std::ofstream cpuSmoke(outputRoot / "cpu-smoke.csv");
  row(cpuSmoke, {"source", "configuration_id", "depth", "seed", "selected_step",
                 "validation_ar_nll", "development_gate", "final_gate", "classification"});
  std::ofstream decisions(outputRoot / "selection-decisions.csv");
  row(decisions, {"configuration_id", "depth", "seed", "mode", "selected_step",
                  "validation_ar_nll", "validation_token_exact", "validation_sequence_exact",
                  "classification"});

  for (const auto& spec : kRuns) {
    CpuRecord record;
    record.spec = spec;
    const auto config = makeConfig(spec.layers);
    record.selected = dq::runAutoregressiveSelectedCpu(config, spec.seed,
                                                       kSteps,
                                                       dq::AutoregressiveSelectionMode::BEST_AR_VALIDATION_V1);
    for (const auto& entry : record.selected.validationTrajectory) {
      record.validation[entry.first] = entry.second;
      writeMetrics(validation, "CPU_REFERENCE_REGENERATION", spec, entry.first,
                   entry.first == record.selected.selectedStep ? "SELECTED" : "TRAJECTORY",
                   ar::Partition::VALIDATION, entry.second);
    }
    for (const int step : kEvaluationSteps) {
      const auto checkpoint = record.selected.training.checkpoints.find(step);
      if (checkpoint == record.selected.training.checkpoints.end())
        throw std::runtime_error("CPU checkpoint missing at requested step");
      if (step == 0) {
        const auto batch = dq::formalBatch(config, 0, 0);
        const auto initial = tiny::forwardBackward(
            config, batch.first, batch.second, checkpoint->second, 0.0f);
        row(trajectory, {"CPU_REFERENCE_REGENERATION", spec.publicId, text(spec.layers),
                         text(spec.seed), "0", number(initial.loss),
                         number(initial.accuracy),
                         number(dq::registryNorm(initial.gradients)),
                         number(dq::registryNorm(checkpoint->second)), "0", "0"});
        continue;
      }
      const auto& metric = record.selected.training.steps.at(
          static_cast<std::size_t>(step - 1));
      row(trajectory, {"CPU_REFERENCE_REGENERATION", spec.publicId, text(spec.layers),
                       text(spec.seed), text(step), number(metric.loss),
                       number(metric.accuracy), number(metric.gradientNorm),
                       number(metric.parameterNorm),
                       number(metric.updateNorm), number(metric.updateToParameter)});
    }
    const auto selectedDev = dq::autoregressiveEvaluation(
        config, record.selected.selectedParameters, ar::Partition::DEVELOPMENT);
    const auto finalDev = dq::autoregressiveEvaluation(
        config, record.selected.training.finalParameters, ar::Partition::DEVELOPMENT);
    record.developmentSelected = selectedDev;
    record.developmentFinal = finalDev;
    record.developmentEvaluated = true;
    row(decisions, {spec.publicId, text(spec.layers), text(spec.seed), "BEST_AR_VALIDATION_V1",
                    text(record.selected.selectedStep), number(record.selected.selectedValidation.autoregressiveNll),
                    text(record.selected.selectedValidation.tokenExact),
                    text(record.selected.selectedValidation.sequenceExact), trajectoryClass(record)});
    records.push_back(std::move(record));
  }

  const bool devPass = developmentGate(records);
  std::ofstream development(outputRoot / "ar-development-metrics.csv");
  row(development, {"source", "configuration_id", "depth", "seed", "step", "role", "partition",
                    "all_finite", "ar_rollout_nll", "teacher_forced_nll",
                    "teacher_forced_gap", "token_exact", "token_total", "sequence_exact",
                    "sequence_total", "mean_first_error_position",
                    "post_error_recovery_tokens"});
  for (auto& record : records) {
    writeMetrics(development, "CPU_REFERENCE_REGENERATION", record.spec, record.selected.selectedStep,
                 "SELECTED", ar::Partition::DEVELOPMENT, record.developmentSelected);
    writeMetrics(development, "CPU_REFERENCE_REGENERATION", record.spec, kSteps, "FINAL_STEP",
                 ar::Partition::DEVELOPMENT, record.developmentFinal);
  }
  std::ofstream final(outputRoot / "ar-final-holdout-metrics.csv");
  row(final, {"source", "configuration_id", "depth", "seed", "step", "role", "partition", "status",
              "all_finite", "ar_rollout_nll", "teacher_forced_nll", "teacher_forced_gap",
              "token_exact", "token_total", "sequence_exact", "sequence_total",
              "mean_first_error_position", "post_error_recovery_tokens"});
  if (!devPass) {
    std::vector<std::string> notRun(18);
    notRun[0] = "GATE";
    notRun[1] = "ALL";
    notRun[6] = ar::partitionName(ar::Partition::FINAL);
    notRun[7] = "NOT_RUN_GATE_REJECTED";
    row(final, notRun);
  } else {
    for (auto& record : records) {
      const auto config = makeConfig(record.spec.layers);
      record.finalSelected = dq::autoregressiveEvaluation(
          config, record.selected.selectedParameters, ar::Partition::FINAL);
      record.finalFinal = dq::autoregressiveEvaluation(
          config, record.selected.training.finalParameters, ar::Partition::FINAL);
      record.finalEvaluated = true;
      writeFinalMetrics(final, "CPU_REFERENCE_REGENERATION", record.spec,
                        record.selected.selectedStep, "SELECTED", record.finalSelected);
      writeFinalMetrics(final, "CPU_REFERENCE_REGENERATION", record.spec, kSteps,
                        "FINAL_STEP", record.finalFinal);
    }
  }
  const bool finalPass = devPass && finalGate(records);
  writeManifest(outputRoot, devPass, finalPass);
  std::ofstream decision(outputRoot / "decision.csv");
  row(decision, {"development_gate", "final_holdout_gate", "checkpoint_selection_mode",
                 "stabilizer", "replay_available", "replay_missing", "classification"});
  row(decision, {devPass ? "PASS" : "REJECT",
                 devPass ? (finalPass ? "PASS" : "REJECT") : "NOT_RUN",
                 finalPass ? "BEST_AR_VALIDATION_V1" : "NONE", "NONE",
                 text(replayAvailable), text(replayMissing),
                 finalPass ? "AUTOREGRESSIVE_VALIDATION_SELECTED_QUALITY_IMPROVED"
                           : "AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE"});
  for (const auto& record : records)
    row(cpuSmoke, {"CPU_REFERENCE_REGENERATION", record.spec.publicId, text(record.spec.layers),
                   text(record.spec.seed), text(record.selected.selectedStep),
                   number(record.selected.selectedValidation.autoregressiveNll),
                   devPass ? "PASS" : "REJECT",
                   devPass ? (finalPass ? "PASS" : "REJECT") : "NOT_RUN",
                   trajectoryClass(record)});
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--self-test") {
      selfTest();
      return 0;
    }
    if (argc < 2 || std::string(argv[1]) != "--run") {
      std::cerr << "usage: autoregressive_validation_probe --self-test | --run --output DIR --checkpoint-root DIR\n";
      return 2;
    }
    fs::path output, checkpoints;
    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--output" && i + 1 < argc) output = argv[++i];
      else if (argument == "--checkpoint-root" && i + 1 < argc) checkpoints = argv[++i];
      else return 2;
    }
    if (output.empty() || checkpoints.empty()) return 2;
    return run(output, checkpoints);
  } catch (const std::exception& error) {
    std::cerr << "autoregressive_validation_probe: " << error.what() << '\n';
    return 3;
  }
}
