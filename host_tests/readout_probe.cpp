// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// READOUT_PROBE_V1 host-only runner: layer-wise linear readout probes and
// frozen-output-head retraining for the L19 readout/representation
// diagnosis. No device, QAIRT, ADB, QNN graph, or Android involvement;
// production training code is not modified (features come from the VERBATIM
// forward copy in critical_margin_training_lib.h).
//
// Trajectories are regenerated host-only with the canonical CPU training
// loop (dq::runFormalCpu, lr 0.003 LEGACY, 320 steps) and every regenerated
// checkpoint is asserted against pinned canonical anchors before use:
//   AR_DEVELOPMENT_V3 (dq::autoregressiveEvaluation): selected 14/20/22/18
//     token exact (0 seq), final-320 30/63/46/65 token exact (2/6/6/8 seq),
//     final AR NLL 8.1239203249880703/4.1834252619661516/7.5872917441801651/
//     5.3026052051209884 (first-error bundle configuration.csv)
//   MARGIN_DEVELOPMENT_V1 free-running at 320: token 50/65/86/60, seq
//     8/9/12/9, AR NLL 7.2793924123677254/4.5271806692690921/
//     4.5741163228215562/5.6041387784705003 (canonical checkpoint-metrics.csv)
//   MARGIN_CALIBRATION_V1 free-running at 320: token 50/59/92/54, seq
//     7/8/11/8, AR NLL 7.1173910296005136/6.0751862806397474/
//     4.030170295310362/6.7455517100879607
// Any mismatch is fatal: the canonical trajectory is not reproduced.
//
// AR_FINAL_HOLDOUT_V3 is only hash-verified, never opened or evaluated.
#include "readout_probe_lib.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rp = phonelm::readout_probe;
namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;
namespace dq = phonelm::depth_quality;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;

namespace {

struct ConfigSpec {
  const char* publicId = "";
  std::uint32_t seed = 0;
  std::uint32_t layers = 19;
  int arSelectedStep = 0;
  int bestExactStep = 0;
  int finalStep = 320;
  // AR_DEVELOPMENT_V3 anchors.
  std::uint64_t arDevSelectedTokenExact = 0;
  std::uint64_t arDevSelectedSeqExact = 0;
  std::uint64_t arDevFinalTokenExact = 0;
  std::uint64_t arDevFinalSeqExact = 0;
  double arDevFinalNll = 0.0;
  // MARGIN_DEVELOPMENT_V1 free-running anchors.
  std::uint64_t marginDevSelectedTokenExact = 0;
  std::uint64_t marginDevFinalTokenExact = 0;
  std::uint64_t marginDevFinalSeqExact = 0;
  double marginDevFinalNll = 0.0;
  double marginDevSelectedNll = 0.0;
  // MARGIN_CALIBRATION_V1 free-running anchors.
  std::uint64_t marginCalFinalTokenExact = 0;
  std::uint64_t marginCalFinalSeqExact = 0;
  double marginCalFinalNll = 0.0;
};

const std::vector<ConfigSpec> kSpecs{
    {"L19_SEED_1", 1, 19, 16, 32, 320, 14, 0, 30, 2, 8.1239203249880703,
     16, 50, 8, 7.2793924123677254, 3.1434760002758511, 50, 7,
     7.1173910296005136},
    {"L19_SEED_2", 2, 19, 4, 128, 320, 20, 0, 63, 6, 4.1834252619661516,
     20, 65, 9, 4.5271806692690921, 3.2025826990955695, 59, 8,
     6.0751862806397474},
    {"L19_SEED_4", 4, 19, 12, 80, 320, 22, 0, 46, 6, 7.5872917441801651,
     21, 86, 12, 4.5741163228215562, 2.9437165421652174, 92, 11,
     4.030170295310362},
    {"L18_SEED_2_CONTROL", 2, 18, 4, 160, 320, 18, 0, 65, 8,
     5.3026052051209884, 29, 60, 9, 5.6041387784705003, 3.2197527055581228,
     54, 8, 6.7455517100879607},
};

tiny::Config modelConfig(std::uint32_t layers) {
  tiny::Config config;
  config.vocabularySize = 32;
  config.tokens = 8;
  config.dimension = 16;
  config.feedForwardDimension = 32;
  config.epsilon = 1e-5f;
  config.numLayers = layers;
  config.numHeads = 2;
  return config;
}

struct CheckpointRun {
  int step = 0;
  bool finalCheckpoint = false;
  bool fullReps = false;
  train::P params;
  std::string paramHash;
  rp::TokenMetrics headTrainTf, headCalTf, headDevTf;
  cm::CheckpointMetrics headCalFr, headDevFr;
  std::vector<ma::Score> headDevScores;
  std::vector<int> reps;
  // per rep:
  std::map<int, rp::ProbeTrainResult> probes;
  std::map<int, rp::TokenMetrics> probeTrainTf, probeCalTf, probeDevTf;
  std::map<int, cm::CheckpointMetrics> probeDevFr, probeDevHeadCtx, probeCalFr;
  std::map<int, rp::RepMetrics> repMetricsDev, repMetricsTrain;
  std::vector<double> repAlignment;
};

struct RunRecord {
  const ConfigSpec* spec = nullptr;
  dq::FormRun run;
  std::vector<CheckpointRun> checkpoints;
  // head retraining on the final checkpoint:
  std::vector<rp::HeadTrainResult> headRetrains;
  // head geometry (final checkpoint):
  std::vector<double> headRowNorms;
  std::vector<double> headSingularValues;
  double headEffectiveRank = 0.0;
};

std::string anchorError(const ConfigSpec& spec, const std::string& name,
                        std::uint64_t got, std::uint64_t expected) {
  std::ostringstream out;
  out << "ANCHOR_MISMATCH " << spec.publicId << " " << name << " got=" << got
      << " expected=" << expected;
  return out.str();
}

std::string anchorError(const ConfigSpec& spec, const std::string& name,
                        double got, double expected) {
  std::ostringstream out;
  out << "ANCHOR_MISMATCH " << spec.publicId << " " << name
      << " got=" << std::setprecision(16) << got
      << " expected=" << std::setprecision(16) << expected;
  return out.str();
}

void validateDatasets() {
  std::string error;
  if (!cm::validateDatasets(8, &error)) {
    throw std::runtime_error("MARGIN_DATASET_INVALID: " + error);
  }
  if (!ar::hashMatchesPinned(ar::Partition::TRAIN, 8))
    throw std::runtime_error("AR_TRAIN_HASH_MISMATCH");
  if (!ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8))
    throw std::runtime_error("AR_DEVELOPMENT_HASH_MISMATCH");
  // Holdout: hash-check only, never evaluated.
  if (!ar::hashMatchesPinned(ar::Partition::FINAL, 8))
    throw std::runtime_error("AR_FINAL_HOLDOUT_HASH_MISMATCH");
}

// Build all teacher-forced rows and the combined feature set layout.
struct DataSet {
  std::vector<ar::Case> trainCases;
  std::vector<ar::Case> calCases;
  std::vector<ar::Case> devCases;
  std::vector<rp::ProbeRow> trainRows;
  std::vector<rp::ProbeRow> calRows;
  std::vector<rp::ProbeRow> devRows;
  std::vector<rp::ProbeRow> allRows;
  std::size_t trainBegin = 0;
  std::size_t calBegin = 0;
  std::size_t devBegin = 0;
  std::string combinedHash;
};

DataSet buildDataSet() {
  DataSet ds;
  ds.trainCases = ar::cases(ar::Partition::TRAIN, 8);
  ds.calCases = cm::cases(cm::Partition::CALIBRATION, 8);
  ds.devCases = cm::cases(cm::Partition::DEVELOPMENT, 8);
  ds.trainRows = rp::teacherForcedRows(ds.trainCases);
  ds.calRows = rp::teacherForcedRows(ds.calCases);
  ds.devRows = rp::teacherForcedRows(ds.devCases);
  if (ds.trainRows.size() != 32 || ds.calRows.size() != 144 ||
      ds.devRows.size() != 144)
    throw std::runtime_error("PROBE_ROW_COUNT_MISMATCH");
  ds.allRows = ds.trainRows;
  ds.allRows.insert(ds.allRows.end(), ds.calRows.begin(), ds.calRows.end());
  ds.allRows.insert(ds.allRows.end(), ds.devRows.begin(), ds.devRows.end());
  ds.trainBegin = 0;
  ds.calBegin = ds.trainRows.size();
  ds.devBegin = ds.calBegin + ds.calRows.size();
  std::ostringstream hash;
  hash << "TRAIN=" << ar::partitionHash(ar::Partition::TRAIN, 8)
       << ";CAL=" << cm::partitionHash(cm::Partition::CALIBRATION, 8)
       << ";DEV=" << cm::partitionHash(cm::Partition::DEVELOPMENT, 8);
  ds.combinedHash = hash.str();
  return ds;
}

void runAnchors(const ConfigSpec& spec, const dq::FormRun& run,
                const DataSet& ds) {
  const auto selected = run.checkpoints.at(spec.arSelectedStep);
  const auto final = run.checkpoints.at(spec.finalStep);
  const auto arSelected =
      dq::autoregressiveEvaluation(modelConfig(spec.layers), selected,
                                   ar::Partition::DEVELOPMENT);
  if (arSelected.tokenExact != spec.arDevSelectedTokenExact)
    throw std::runtime_error(anchorError(
        spec, "AR_DEV_SELECTED_TOKEN", arSelected.tokenExact,
        spec.arDevSelectedTokenExact));
  if (arSelected.sequenceExact != spec.arDevSelectedSeqExact)
    throw std::runtime_error(anchorError(
        spec, "AR_DEV_SELECTED_SEQ", arSelected.sequenceExact,
        spec.arDevSelectedSeqExact));
  const auto arFinal =
      dq::autoregressiveEvaluation(modelConfig(spec.layers), final,
                                   ar::Partition::DEVELOPMENT);
  if (arFinal.tokenExact != spec.arDevFinalTokenExact)
    throw std::runtime_error(anchorError(spec, "AR_DEV_FINAL_TOKEN",
                                         arFinal.tokenExact,
                                         spec.arDevFinalTokenExact));
  if (arFinal.sequenceExact != spec.arDevFinalSeqExact)
    throw std::runtime_error(anchorError(spec, "AR_DEV_FINAL_SEQ",
                                         arFinal.sequenceExact,
                                         spec.arDevFinalSeqExact));
  if (std::abs(arFinal.autoregressiveNll - spec.arDevFinalNll) > 1e-6)
    throw std::runtime_error(anchorError(spec, "AR_DEV_FINAL_NLL",
                                         arFinal.autoregressiveNll,
                                         spec.arDevFinalNll));
  const auto config = modelConfig(spec.layers);
  const auto marginDevSelected =
      rp::headFreeRunning(config, selected, spec.arSelectedStep, ds.devCases);
  if (marginDevSelected.tokenExact != spec.marginDevSelectedTokenExact)
    throw std::runtime_error(anchorError(
        spec, "MARGIN_DEV_SELECTED_TOKEN", marginDevSelected.tokenExact,
        spec.marginDevSelectedTokenExact));
  if (marginDevSelected.sequenceExact != 0)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_SELECTED_SEQ",
                                         marginDevSelected.sequenceExact, 0));
  
  // NLL anchors are float32-limited (params are float; NLL sums 32 rows);
  // tolerance 1e-6, integers must match exactly.
  if (std::abs(marginDevSelected.autoregressiveNll -
               spec.marginDevSelectedNll) > 1e-6)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_SELECTED_NLL",
                                         marginDevSelected.autoregressiveNll,
                                         spec.marginDevSelectedNll));
  const auto marginDevFinal =
      rp::headFreeRunning(config, final, spec.finalStep, ds.devCases);
  if (marginDevFinal.tokenExact != spec.marginDevFinalTokenExact)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_FINAL_TOKEN",
                                         marginDevFinal.tokenExact,
                                         spec.marginDevFinalTokenExact));
  if (marginDevFinal.sequenceExact != spec.marginDevFinalSeqExact)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_FINAL_SEQ",
                                         marginDevFinal.sequenceExact,
                                         spec.marginDevFinalSeqExact));
  if (std::abs(marginDevFinal.autoregressiveNll - spec.marginDevFinalNll) >
      1e-6)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_FINAL_NLL",
                                         marginDevFinal.autoregressiveNll,
                                         spec.marginDevFinalNll));
  const auto marginCalFinal =
      rp::headFreeRunning(config, final, spec.finalStep, ds.calCases);
  if (marginCalFinal.tokenExact != spec.marginCalFinalTokenExact)
    throw std::runtime_error(anchorError(spec, "MARGIN_CAL_FINAL_TOKEN",
                                         marginCalFinal.tokenExact,
                                         spec.marginCalFinalTokenExact));
  if (marginCalFinal.sequenceExact != spec.marginCalFinalSeqExact)
    throw std::runtime_error(anchorError(spec, "MARGIN_CAL_FINAL_SEQ",
                                         marginCalFinal.sequenceExact,
                                         spec.marginCalFinalSeqExact));
  if (std::abs(marginCalFinal.autoregressiveNll - spec.marginCalFinalNll) >
      1e-6)
    throw std::runtime_error(anchorError(spec, "MARGIN_CAL_FINAL_NLL",
                                         marginCalFinal.autoregressiveNll,
                                         spec.marginCalFinalNll));
}

// Probes per checkpoint; representative reps for non-final checkpoints.
std::vector<int> repsFor(const ConfigSpec& spec, int step) {
  const int repCount = rp::representationCount(static_cast<int>(spec.layers));
  if (step == spec.finalStep) {
    std::vector<int> reps;
    for (int rep = 0; rep < repCount; ++rep) reps.push_back(rep);
    return reps;
  }
  std::vector<int> reps;
  for (const int proto : rp::kRepresentativeReps)
    reps.push_back(rp::resolveRep(static_cast<int>(spec.layers), proto));
  return reps;
}

CheckpointRun evaluateCheckpoint(const ConfigSpec& spec, const DataSet& ds,
                                 const train::P& params, int step,
                                 const std::filesystem::path& cacheDir) {
  const auto config = modelConfig(spec.layers);
  CheckpointRun out;
  out.step = step;
  out.finalCheckpoint = step == spec.finalStep;
  out.fullReps = out.finalCheckpoint;
  out.params = params;
  out.paramHash = rp::fnv1aParams(params);

  // Baseline: current head on teacher-forced rows + free-running rollouts.
  out.headTrainTf = rp::headTokenMetrics(config, params, ds.trainRows);
  out.headCalTf = rp::headTokenMetrics(config, params, ds.calRows);
  out.headDevTf = rp::headTokenMetrics(config, params, ds.devRows);
  out.headCalFr = rp::headFreeRunning(config, params, step, ds.calCases);
  out.headDevFr = rp::headFreeRunning(config, params, step, ds.devCases);
  out.headDevScores = rp::headRowScores(config, params, ds.devRows);

  // Hidden features (all reps, all rows) with cache identity.
  rp::CacheIdentity identity;
  identity.protocol = rp::kProbeProtocolId;
  identity.config = spec.publicId;
  identity.seed = spec.seed;
  identity.step = step;
  identity.datasetHash = ds.combinedHash;
  identity.depth = static_cast<int>(spec.layers);
  identity.repCount =
      rp::representationCount(static_cast<int>(spec.layers));
  identity.dim = static_cast<int>(config.dimension);
  identity.rows = ds.allRows.size();
  identity.paramHash = out.paramHash;

  rp::LayerSet set;
  if (!rp::readHiddenCache(cacheDir, identity, set)) {
    set = rp::extractFeatures(config, params, ds.allRows);
    if (!rp::writeHiddenCache(cacheDir, identity, set))
      throw std::runtime_error("HIDDEN_CACHE_WRITE_FAILED");
  } else {
    const std::string hash = rp::fnv1aParams(params);
    if (hash != out.paramHash)
      throw std::runtime_error("HIDDEN_CACHE_PARAM_HASH_MISMATCH");
  }

  out.reps = repsFor(spec, step);
  const auto headVectors =
      rp::headClassVectors(params, static_cast<int>(config.dimension),
                           static_cast<int>(config.vocabularySize));

  for (const int rep : out.reps) {
    const rp::ZStats stats =
        rp::computeZStats(set, rep, ds.trainRows.size());  // TRAIN rows only
    auto probeResult = rp::trainProbe(
        set, rep, stats, ds.allRows, ds.trainBegin, ds.calBegin,
        ds.allRows, ds.calBegin, ds.devBegin);
    out.probes[rep] = probeResult;
    if (!probeResult.finite) continue;
    out.probeTrainTf[rep] = rp::probeTokenMetrics(
        probeResult.probe, stats, set, rep, ds.trainRows, 0);
    out.probeCalTf[rep] = rp::probeTokenMetrics(
        probeResult.probe, stats, set, rep, ds.calRows, ds.calBegin);
    out.probeDevTf[rep] = rp::probeTokenMetrics(
        probeResult.probe, stats, set, rep, ds.devRows, ds.devBegin);
    out.probeDevFr[rep] =
        rp::probeFreeRunning(config, params, step, ds.devCases,
                             probeResult.probe, stats, set, rep);
    out.probeDevHeadCtx[rep] =
        rp::probeOnHeadContexts(config, params, step, ds.devCases,
                                probeResult.probe, stats, set, rep);
    out.probeCalFr[rep] =
        rp::probeFreeRunning(config, params, step, ds.calCases,
                             probeResult.probe, stats, set, rep);
    // Representation metrics (dev rows; train rows for the gap).
    const auto devMeans = rp::classMeans(set, rep, ds.devRows, ds.devBegin);
    out.repMetricsDev[rep] = rp::computeRepMetrics(
        set, rep, ds.devRows,
        rp::embeddedNorms(set, ds.devBegin, ds.devBegin + ds.devRows.size()),
        out.headDevScores, devMeans, ds.devBegin);
    out.repMetricsDev[rep].alignmentCosine =
        rp::meanAlignmentCosine(set, rep, ds.devRows, headVectors, ds.devBegin);
    const auto trainMeans = rp::classMeans(set, rep, ds.trainRows, 0);
    const auto trainHeadScores = rp::headRowScores(config, params, ds.trainRows);
    out.repMetricsTrain[rep] = rp::computeRepMetrics(
        set, rep, ds.trainRows,
        rp::embeddedNorms(set, 0, ds.trainRows.size()), trainHeadScores,
        trainMeans, 0);
  }
  return out;
}

std::vector<rp::HeadTrainResult> retrainHeads(const ConfigSpec& spec,
                                              const DataSet& ds,
                                              const train::P& finalParams) {
  const auto config = modelConfig(spec.layers);
  std::vector<rp::HeadTrainResult> results;
  for (const auto candidate :
       {rp::HeadCandidate::kWarmStart, rp::HeadCandidate::kReinit,
        rp::HeadCandidate::kBiasOnly}) {
    results.push_back(rp::retrainHead(
        config, spec.seed, finalParams, candidate, ds.trainRows, ds.calRows,
        ds.devRows, ds.calCases, ds.devCases));
  }
  return results;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------
struct ReportPaths {
  std::filesystem::path root;
  std::filesystem::path hidden;
};

std::string tokenSummary(const rp::TokenMetrics& m) {
  std::ostringstream out;
  out << m.tokenExact << "/" << m.total << " rank=" << std::setprecision(4)
      << m.meanRank << " nll=" << std::setprecision(6) << m.meanNll
      << " margin=" << std::setprecision(6) << m.meanMargin
      << " q10=" << std::setprecision(6) << m.marginQ10
      << " top2=" << std::setprecision(4) << m.top2
      << " top3=" << std::setprecision(4) << m.top3;
  return out.str();
}

std::string frSummary(const cm::CheckpointMetrics& m) {
  std::ostringstream out;
  out << m.tokenExact << "/" << m.tokenTotal << " seq=" << m.sequenceExact
      << "/" << m.sequenceTotal << " nll=" << std::setprecision(6)
      << m.autoregressiveNll << " surv=" << std::setprecision(4)
      << m.medianFirstErrorSurvival << " q10=" << std::setprecision(6)
      << m.lowerTailMarginQ10;
  return out.str();
}

void writeDatasetAnchors(const std::filesystem::path& path,
                         const DataSet& ds) {
  rp::CsvWriter csv(path);
  csv.header({"dataset", "role", "hash", "rows"});
  csv.row({"TRAIN", "probe_and_head_learning",
           ar::partitionHash(ar::Partition::TRAIN, 8),
           rp::text(std::uint64_t(ds.trainRows.size()))});
  csv.row({"MARGIN_CALIBRATION_V1", "step_selection_only",
           cm::partitionHash(cm::Partition::CALIBRATION, 8),
           rp::text(std::uint64_t(ds.calRows.size()))});
  csv.row({"MARGIN_DEVELOPMENT_V1", "final_evaluation_only",
           cm::partitionHash(cm::Partition::DEVELOPMENT, 8),
           rp::text(std::uint64_t(ds.devRows.size()))});
  csv.row({"AR_FINAL_HOLDOUT_V3", "UNOPENED_hash_verified_only",
           ar::partitionHash(ar::Partition::FINAL, 8), "0"});
}

void writeTrajectoryAnchors(const std::filesystem::path& path,
                            const std::vector<RunRecord>& runs,
                            const DataSet& ds) {
  (void)ds;
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint", "step", "metric", "value",
              "pinned_anchor", "match"});
  for (const auto& run : runs) {
    const auto& spec = *run.spec;
    const auto& selected = run.run.checkpoints.at(spec.arSelectedStep);
    const auto& final = run.run.checkpoints.at(spec.finalStep);
    const auto arSelected = dq::autoregressiveEvaluation(
        modelConfig(spec.layers), selected, ar::Partition::DEVELOPMENT);
    const auto arFinal = dq::autoregressiveEvaluation(
        modelConfig(spec.layers), final, ar::Partition::DEVELOPMENT);
    struct Row {
      std::string ckpt;
      int step = 0;
      std::string metric;
      std::string value;
      std::string anchor;
      bool match = false;
    };
    std::vector<Row> rows;
    auto add = [&](const std::string& ckpt, int step,
                   const std::string& metric, std::uint64_t value,
                   std::uint64_t anchor) {
      rows.push_back({ckpt, step, metric, rp::text(value), rp::text(anchor),
                      value == anchor});
    };
    auto addD = [&](const std::string& ckpt, int step,
                    const std::string& metric, double value, double anchor) {
      // NLL anchors are float32-limited; match within 1e-6.
      rows.push_back({ckpt, step, metric, rp::text(value, 16),
                      rp::text(anchor, 16),
                      std::abs(value - anchor) <= 1e-6});
    };
    add("AR_DEV_SELECTED", spec.arSelectedStep, "token_exact",
        arSelected.tokenExact, spec.arDevSelectedTokenExact);
    add("AR_DEV_SELECTED", spec.arSelectedStep, "sequence_exact",
        arSelected.sequenceExact, spec.arDevSelectedSeqExact);
    add("AR_DEV_FINAL", spec.finalStep, "token_exact", arFinal.tokenExact,
        spec.arDevFinalTokenExact);
    add("AR_DEV_FINAL", spec.finalStep, "sequence_exact",
        arFinal.sequenceExact, spec.arDevFinalSeqExact);
    addD("AR_DEV_FINAL", spec.finalStep, "autoregressive_nll",
         arFinal.autoregressiveNll, spec.arDevFinalNll);
    for (const auto& row : rows)
      csv.row({spec.publicId, row.ckpt, rp::text(row.step), row.metric,
               row.value, row.anchor, row.match ? "true" : "false"});
  }
}

void writeBaseline(const std::filesystem::path& path,
                   const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "is_final",
              "head_train_tf_token_exact", "head_cal_tf_token_exact",
              "head_dev_tf_token_exact", "head_cal_fr_token_exact",
              "head_cal_fr_sequence_exact", "head_dev_fr_token_exact",
              "head_dev_fr_sequence_exact", "head_dev_fr_nll",
              "head_dev_fr_median_survival", "head_dev_fr_margin_q10",
              "head_dev_tf_mean_rank", "head_dev_tf_mean_nll"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints)
      csv.row({run.spec->publicId, rp::text(cp.step),
               cp.finalCheckpoint ? "true" : "false",
               rp::text(cp.headTrainTf.tokenExact),
               rp::text(cp.headCalTf.tokenExact),
               rp::text(cp.headDevTf.tokenExact),
               rp::text(cp.headCalFr.tokenExact),
               rp::text(cp.headCalFr.sequenceExact),
               rp::text(cp.headDevFr.tokenExact),
               rp::text(cp.headDevFr.sequenceExact),
               rp::text(cp.headDevFr.autoregressiveNll),
               rp::text(cp.headDevFr.medianFirstErrorSurvival),
               rp::text(cp.headDevFr.lowerTailMarginQ10),
               rp::text(cp.headDevTf.meanRank), rp::text(cp.headDevTf.meanNll)});
}

void writeProbeSelection(const std::filesystem::path& path,
                         const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "rep", "rep_name",
              "selected_step", "finite", "nonfinite_step", "nonfinite_what",
              "train_ce", "train_token_exact", "cal_ce", "cal_token_exact",
              "max_logit_abs", "dev_tf_token_exact", "dev_tf_mean_rank",
              "dev_tf_mean_nll", "dev_fr_token_exact", "dev_fr_sequence_exact",
              "dev_fr_nll", "dev_head_ctx_token_exact", "dev_head_ctx_seq_exact",
              "step2000_train_ce", "step2000_train_token_exact"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints)
      for (const int rep : cp.reps) {
        const auto& probe = cp.probes.at(rep);
        std::string devTf = "", devRank = "", devNll = "", devFr = "",
                    devFrSeq = "", devFrNll = "", headCtx = "", headCtxSeq = "";
        if (probe.finite) {
          const auto& t = cp.probeDevTf.at(rep);
          const auto& f = cp.probeDevFr.at(rep);
          const auto& h = cp.probeDevHeadCtx.at(rep);
          devTf = rp::text(t.tokenExact);
          devRank = rp::text(t.meanRank);
          devNll = rp::text(t.meanNll);
          devFr = rp::text(f.tokenExact);
          devFrSeq = rp::text(f.sequenceExact);
          devFrNll = rp::text(f.autoregressiveNll);
          headCtx = rp::text(h.tokenExact);
          headCtxSeq = rp::text(h.sequenceExact);
        }
        csv.row({run.spec->publicId, rp::text(cp.step), rp::text(rep),
                 rp::representationName(static_cast<int>(run.spec->layers), rep),
                 rp::text(probe.selectedStep),
                 probe.finite ? "true" : "false",
                 rp::text(probe.nonfiniteStep), probe.nonfiniteWhat,
                 rp::text(probe.trainCe), rp::text(probe.trainExact),
                 rp::text(probe.calCe), rp::text(probe.calExact),
                 rp::text(probe.maxLogitAbs), devTf, devRank, devNll, devFr,
                 devFrSeq, devFrNll, headCtx, headCtxSeq,
                 rp::text(probe.trainCe2000),
                 rp::text(probe.trainExact2000)});
      }
}

void writeLayerCurve(const std::filesystem::path& path,
                     const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "rep", "rep_name",
              "probe_dev_fr_token_exact", "head_dev_fr_token_exact",
              "probe_minus_head_fr", "probe_dev_tf_token_exact",
              "probe_train_tf_token_exact", "probe_cal_fr_token_exact"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints)
      for (const int rep : cp.reps) {
        const auto& probe = cp.probes.at(rep);
        if (!probe.finite) continue;
        const auto& fr = cp.probeDevFr.at(rep);
        const auto& tf = cp.probeDevTf.at(rep);
        const auto& tr = cp.probeTrainTf.at(rep);
        const auto& cf = cp.probeCalFr.find(rep) == cp.probeCalFr.end()
                             ? cm::CheckpointMetrics{}
                             : cp.probeCalFr.at(rep);
        const std::int64_t delta =
            static_cast<std::int64_t>(fr.tokenExact) -
            static_cast<std::int64_t>(cp.headDevFr.tokenExact);
        csv.row({run.spec->publicId, rp::text(cp.step), rp::text(rep),
                 rp::representationName(static_cast<int>(run.spec->layers), rep),
                 rp::text(fr.tokenExact), rp::text(cp.headDevFr.tokenExact),
                 rp::text(static_cast<std::int64_t>(delta)),
                 rp::text(tf.tokenExact), rp::text(tr.tokenExact),
                 rp::text(cf.tokenExact)});
      }
}

void writeProbeGrid(const std::filesystem::path& path,
                    const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "rep", "grid_step",
              "train_ce", "cal_ce", "train_token_exact", "cal_token_exact",
              "is_selected"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints)
      for (const int rep : cp.reps) {
        const auto& probe = cp.probes.at(rep);
        if (!probe.finite) continue;
        for (const auto& point : probe.grid)
          csv.row({run.spec->publicId, rp::text(cp.step), rp::text(rep),
                   rp::text(point.step), rp::text(point.trainCe),
                   rp::text(point.calCe), rp::text(point.trainExact),
                   rp::text(point.calExact),
                   point.step == probe.selectedStep ? "true" : "false"});
      }
}

void writeHeadRetraining(const std::filesystem::path& path,
                         const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "candidate", "selected_step", "finite",
              "frozen_unchanged", "train_ce", "train_token_exact", "cal_ce",
              "cal_token_exact", "dev_tf_token_exact", "dev_tf_mean_nll",
              "dev_fr_token_exact", "dev_fr_sequence_exact", "dev_fr_nll",
              "dev_fr_median_survival", "dev_fr_margin_q10",
              "dev_tf320_token_exact", "dev_fr320_token_exact",
              "dev_fr320_sequence_exact"});
  for (const auto& run : runs)
    for (const auto& head : run.headRetrains)
      csv.row({run.spec->publicId, rp::headCandidateName(head.candidate),
               rp::text(head.selectedStep), head.finite ? "true" : "false",
               head.frozenUnchanged ? "true" : "false",
               rp::text(head.trainCe), rp::text(head.trainExact),
               rp::text(head.calCe), rp::text(head.calExact),
               rp::text(head.devTf.tokenExact), rp::text(head.devTf.meanNll),
               rp::text(head.devFr.tokenExact),
               rp::text(head.devFr.sequenceExact),
               rp::text(head.devFr.autoregressiveNll),
               rp::text(head.devFr.medianFirstErrorSurvival),
               rp::text(head.devFr.lowerTailMarginQ10),
               rp::text(head.devTf320.tokenExact),
               rp::text(head.devFr320.tokenExact),
               rp::text(head.devFr320.sequenceExact)});
}

void writeRepresentationMetrics(const std::filesystem::path& path,
                                const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "rep", "rep_name",
              "dataset", "eta2", "effective_rank", "norm_ratio",
              "hidden_margin_midmedian", "sign_agreement", "alignment_cosine"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints)
      for (const int rep : cp.reps) {
        if (!cp.probes.at(rep).finite) continue;
        const auto& dev = cp.repMetricsDev.at(rep);
        const auto& tr = cp.repMetricsTrain.at(rep);
        const std::string name =
            rp::representationName(static_cast<int>(run.spec->layers), rep);
        for (const auto* dataset : {&dev, &tr}) {
          const char* ds = dataset == &dev ? "MARGIN_DEVELOPMENT_V1" : "TRAIN";
          csv.row({run.spec->publicId, rp::text(cp.step), rp::text(rep), name,
                   ds, rp::text(dataset->eta2),
                   rp::text(dataset->effectiveRank),
                   rp::text(dataset->normRatio),
                   rp::text(dataset->hiddenMarginMidmedian),
                   rp::text(dataset->signAgreement),
                   rp::text(dataset->alignmentCosine)});
        }
      }
}

void writeHeadGeometry(const std::filesystem::path& path,
                       const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "item", "index", "value"});
  for (const auto& run : runs) {
    for (std::size_t i = 0; i < run.headRowNorms.size(); ++i)
      csv.row({run.spec->publicId, "class_row_norm", rp::text(int(i)),
               rp::text(run.headRowNorms[i])});
    for (std::size_t i = 0; i < run.headSingularValues.size(); ++i)
      csv.row({run.spec->publicId, "singular_value", rp::text(int(i)),
               rp::text(run.headSingularValues[i])});
    csv.row({run.spec->publicId, "effective_rank", "0",
             rp::text(run.headEffectiveRank)});
  }
}

// ---------------------------------------------------------------------------
// Cause classification (thresholds fixed before results; never tuned)
// ---------------------------------------------------------------------------
struct Decision {
  std::string verdict;
  std::string reasons;
};

Decision classify(const std::vector<RunRecord>& runs,
                  const std::vector<std::pair<const ConfigSpec*, const CheckpointRun*>>&
                      finalCps) {
  // final-layer rep index per spec.
  const std::vector<std::int64_t> headFr = {
      static_cast<std::int64_t>(finalCps[0].second->headDevFr.tokenExact),
      static_cast<std::int64_t>(finalCps[1].second->headDevFr.tokenExact),
      static_cast<std::int64_t>(finalCps[2].second->headDevFr.tokenExact)};
  std::vector<std::int64_t> probeFr;
  for (const auto& entry : finalCps) {
    if (entry.first->layers != 19) continue;
    const int depth = static_cast<int>(entry.first->layers);
    const int finalRep = depth;  // final block out == HEAD_IN
    const auto& probe = entry.second->probes.at(finalRep);
    if (!probe.finite) throw std::runtime_error("FINAL_REP_PROBE_NONFINITE");
    probeFr.push_back(
        static_cast<std::int64_t>(entry.second->probeDevFr.at(finalRep).tokenExact));
  }
  std::int64_t headPooled = 0, probePooled = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    headPooled += headFr[i];
    probePooled += probeFr[i];
  }
  std::int64_t seedWins = 0;
  for (std::size_t i = 0; i < 3; ++i)
    if (probeFr[i] - headFr[i] >= 5) ++seedWins;

  // Best intermediate (excluding the final-layer rep) per seed, pooled.
  std::vector<std::int64_t> bestIntermediate(3, 0);
  std::int64_t bestIntermediatePooled = 0;
  for (std::size_t s = 0; s < 3; ++s) {
    const auto* cp = finalCps[s].second;
    for (const int rep : cp->reps) {
      if (rep >= static_cast<int>(finalCps[s].first->layers)) continue;
      const auto& probe = cp->probes.at(rep);
      if (!probe.finite) continue;
      const std::int64_t fr =
          static_cast<std::int64_t>(cp->probeDevFr.at(rep).tokenExact);
      bestIntermediate[s] = std::max(bestIntermediate[s], fr);
    }
    bestIntermediatePooled += bestIntermediate[s];
  }
  std::int64_t intermediateWins = 0;
  for (std::size_t i = 0; i < 3; ++i)
    if (bestIntermediate[i] - probeFr[i] >= 5) ++intermediateWins;

  // Deep degradation must not be a single-checkpoint artifact: confirm on a
  // non-final checkpoint as well (any non-final checkpoint with reps).
  bool deepMultiCheckpoint = false;
  for (const auto& run : runs) {
    if (run.spec->layers != 19) continue;
    for (const auto& cp : run.checkpoints) {
      if (cp.finalCheckpoint) continue;
      std::int64_t bestMid = 0, finalRepFr = -1;
      for (const int rep : cp.reps) {
        if (rep >= static_cast<int>(run.spec->layers)) continue;
        const auto& probe = cp.probes.at(rep);
        if (!probe.finite) continue;
        bestMid = std::max(bestMid, static_cast<std::int64_t>(
                                        cp.probeDevFr.at(rep).tokenExact));
      }
      const auto& probe = cp.probes.at(static_cast<int>(run.spec->layers));
      if (probe.finite)
        finalRepFr = static_cast<std::int64_t>(
            cp.probeDevFr.at(static_cast<int>(run.spec->layers)).tokenExact);
      if (finalRepFr >= 0 && bestMid - finalRepFr >= 5) {
        deepMultiCheckpoint = true;
        break;
      }
    }
  }

  // Train vs dev gap (probe TF, pooled L19, final checkpoints).
  std::int64_t trainPooled = 0, devTfPooled = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    const int depth = static_cast<int>(finalCps[i].first->layers);
    trainPooled += static_cast<std::int64_t>(
        finalCps[i].second->probeTrainTf.at(depth).tokenExact);
    devTfPooled += static_cast<std::int64_t>(
        finalCps[i].second->probeDevTf.at(depth).tokenExact);
  }

  // L18 control: final probe vs head, must not worsen by more than 2.
  const auto* control = finalCps[3].second;
  const int cDepth = static_cast<int>(finalCps[3].first->layers);
  const std::int64_t controlHead =
      static_cast<std::int64_t>(control->headDevFr.tokenExact);
  const std::int64_t controlProbe = static_cast<std::int64_t>(
      control->probeDevFr.at(cDepth).tokenExact);
  const std::int64_t controlDelta = controlProbe - controlHead;

  const bool readoutFailure =
      probePooled - headPooled >= 5 && seedWins >= 2 &&
      controlDelta >= -2 &&
      // dev (not train-only): probe dev TF also improves pooled.
      devTfPooled > 0;
  const bool deepDegradation =
      bestIntermediatePooled - probePooled >= 5 && intermediateWins >= 2 &&
      deepMultiCheckpoint;
  const bool generalizationGap =
      trainPooled - devTfPooled >= 40 && !readoutFailure && !deepDegradation;

  Decision decision;
  if (readoutFailure && deepDegradation)
    decision.verdict = "MIXED_READOUT_FAILURE_AND_DEEP_DEGRADATION";
  else if (readoutFailure)
    decision.verdict = "READOUT_FAILURE";
  else if (deepDegradation)
    decision.verdict = "DEEP_DEGRADATION";
  else if (generalizationGap)
    decision.verdict = "GENERALIZATION_GAP";
  else
    decision.verdict = "UNDETERMINED";

  std::ostringstream out;
  out << "pooled_head_fr=" << headPooled << " pooled_final_probe_fr=" << probePooled
      << " seed_wins_ge5=" << seedWins << " control_delta=" << controlDelta
      << " best_intermediate_pooled=" << bestIntermediatePooled
      << " intermediate_wins_ge5=" << intermediateWins
      << " deep_multi_checkpoint=" << (deepMultiCheckpoint ? "true" : "false")
      << " pooled_train_tf=" << trainPooled << " pooled_dev_tf=" << devTfPooled;
  decision.reasons = out.str();
  return decision;
}

void writeDecision(const std::filesystem::path& path,
                   const Decision& decision) {
  rp::CsvWriter csv(path);
  csv.header({"verdict", "reasons", "thresholds_fixed_before_results"});
  csv.row({decision.verdict, decision.reasons, "true"});
}

void writeSummary(const std::filesystem::path& path,
                  const std::vector<RunRecord>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint_step", "is_final", "scope",
              "metric", "value"});
  for (const auto& run : runs)
    for (const auto& cp : run.checkpoints) {
      const std::string finalFlag = cp.finalCheckpoint ? "true" : "false";
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_tf",
               "train_token_exact", rp::text(cp.headTrainTf.tokenExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_tf",
               "cal_token_exact", rp::text(cp.headCalTf.tokenExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_tf",
               "dev_token_exact", rp::text(cp.headDevTf.tokenExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_fr",
               "cal_token_exact", rp::text(cp.headCalFr.tokenExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_fr",
               "dev_token_exact", rp::text(cp.headDevFr.tokenExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_fr",
               "dev_sequence_exact", rp::text(cp.headDevFr.sequenceExact)});
      csv.row({run.spec->publicId, rp::text(cp.step), finalFlag, "head_fr",
               "dev_nll", rp::text(cp.headDevFr.autoregressiveNll)});
    }
}

// ---------------------------------------------------------------------------
// Self-test (no canonical anchors, no dev partition decisions)
// ---------------------------------------------------------------------------
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error("SELF_TEST_FAILED: " + message);
}

void selfTest() {
  const auto config = modelConfig(2);
  // 1. Row counts and dataset hashes.
  validateDatasets();
  const auto ds = buildDataSet();
  require(ds.trainRows.size() == 32, "TRAIN row count");
  require(ds.calRows.size() == 144, "CAL row count");
  require(ds.devRows.size() == 144, "DEV row count");
  require(ar::partitionHash(ar::Partition::TRAIN, 8) ==
              "fnv1a64:5a64ca2d1aa7f29f",
          "TRAIN hash pin");
  require(cm::partitionHash(cm::Partition::CALIBRATION, 8) ==
              "fnv1a64:71806d5bf19c090a",
          "CAL hash pin");
  require(cm::partitionHash(cm::Partition::DEVELOPMENT, 8) ==
              "fnv1a64:f06fcc3e2d12ca99",
          "DEV hash pin");
  require(ar::partitionHash(ar::Partition::FINAL, 8) ==
              "fnv1a64:aa5081e6df658b4a",
          "FINAL holdout hash pin");

  // 2. LCG init parity with production initialParameters (phase 8, scale .16).
  {
    const auto fresh = tiny::initialParameters(config, 7);
    std::vector<double> lcg;
    rp::lcgFill(lcg, fresh.outputProjection.size(), 7, 8, 0.16);
    bool equal = true;
    for (std::size_t i = 0; i < fresh.outputProjection.size(); ++i)
      if (static_cast<float>(lcg[i]) != fresh.outputProjection[i]) equal = false;
    require(equal, "LCG init parity with initialParameters");
  }

  // 3. Softmax stability at extreme logits.
  {
    double logits[32];
    for (int i = 0; i < 32; ++i) logits[i] = (i % 2 == 0) ? 1e6 : -1e6;
    const auto p = rp::softmaxRow(logits, 32);
    double sum = 0.0;
    for (const double x : p) {
      require(std::isfinite(x) && x >= 0.0, "softmax finite non-negative");
      sum += x;
    }
    require(std::abs(sum - 1.0) < 1e-12, "softmax sums to one");
  }

  // 4. Deterministic training on a small real trajectory.
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    // Combined train+cal rows so probe evals stay in-bounds; z-stats use
    // TRAIN rows only (32).
    std::vector<rp::ProbeRow> combined = ds.trainRows;
    combined.insert(combined.end(), ds.calRows.begin(), ds.calRows.end());
    const auto set = rp::extractFeatures(config, params, combined);
    const rp::ZStats stats = rp::computeZStats(set, 1, ds.trainRows.size());
    const auto a = rp::trainProbe(set, 1, stats, combined, 0,
                                  ds.trainRows.size(), combined,
                                  ds.trainRows.size(), combined.size());
    const auto b = rp::trainProbe(set, 1, stats, combined, 0,
                                  ds.trainRows.size(), combined,
                                  ds.trainRows.size(), combined.size());
    require(a.finite && b.finite, "small probe finite");
    // Regression guard: with correct z-stats the probe must actually learn
    // (a broken stats plumbing silently degenerates every probe into a
    // constant predictor at the class-frequency CE ~ 2.5 with ~4/32 exact).
    require(a.trainExact >= 20 && b.trainExact >= 20,
            "small probe learns (train exact)");
    require(a.trainCe < 1.0 && b.trainCe < 1.0,
            "small probe learns (train CE)");
    require(a.selectedStep == b.selectedStep, "deterministic selection");
    require(a.probe.w == b.probe.w && a.probe.b == b.probe.b,
            "deterministic probe weights");
    require(a.grid.size() == b.grid.size(), "deterministic grid size");
    for (std::size_t i = 0; i < a.grid.size(); ++i) {
      require(a.grid[i].trainCe == b.grid[i].trainCe &&
                  a.grid[i].calCe == b.grid[i].calCe,
              "deterministic grid values");
    }
    // Finite-difference gradient check on the probe CE.
    const double eps = 1e-5;
    rp::Probe probe = a.probe;
    // Loss at +eps on w[0].
    rp::Probe p1 = probe, p2 = probe;
    p1.w[0] += eps;
    p2.w[0] -= eps;
    const auto loss = [&](const rp::Probe& p) {
      double ce = 0.0;
      const auto& f = set.features[1];
      std::vector<double> logits(32);
      for (std::size_t r = 0; r < ds.trainRows.size(); ++r) {
        rp::probeForward(p, stats, f.data() + r * set.dim, logits.data());
        ce += rp::ceFromLogits(logits.data(), 32, ds.trainRows[r].truth);
      }
      return ce / static_cast<double>(ds.trainRows.size());
    };
    const double fd = (loss(p1) - loss(p2)) / (2.0 * eps);
    // Analytic gradient at w[0].
    double analytic = 0.0;
    const auto& f = set.features[1];
    std::vector<double> logits(32);
    for (std::size_t r = 0; r < ds.trainRows.size(); ++r) {
      rp::probeForward(probe, stats, f.data() + r * set.dim, logits.data());
      const auto probs = rp::softmaxRow(logits.data(), 32);
      const int cls = 0;  // w[0] belongs to class 0, dim 0
      analytic += (probs[0] - (ds.trainRows[r].truth == 0 ? 1.0 : 0.0)) *
                  rp::zScore(stats, f.data() + r * set.dim, 0) /
                  static_cast<double>(ds.trainRows.size());
      (void)cls;
    }
    require(std::abs(fd - analytic) < 1e-4, "probe gradient finite-difference");
  }

  // 5. Freeze verification for all three head candidates (small config).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    for (const auto candidate :
         {rp::HeadCandidate::kWarmStart, rp::HeadCandidate::kReinit,
          rp::HeadCandidate::kBiasOnly}) {
      const auto result = rp::retrainHead(
          config, 5, params, candidate, ds.trainRows, ds.calRows, ds.devRows,
          ds.calCases, ds.devCases);
      require(result.finite, "head retrain finite");
      require(result.frozenUnchanged, "head retrain freeze");
      require(result.selectedStep >= 0 && result.selectedStep <= 320,
              "head selection in range");
      require(result.devFr.allFinite, "head retrain dev FR finite");
      if (candidate == rp::HeadCandidate::kBiasOnly) {
        bool biasChanged = false;
        for (const double x : result.bias)
          if (x != 0.0) biasChanged = true;
        require(biasChanged, "bias-only learned a nonzero bias");
      } else {
        bool headChanged = false;
        for (std::size_t i = 0; i < params.outputProjection.size(); ++i)
          if (result.trained.outputProjection[i] != params.outputProjection[i])
            headChanged = true;
        require(headChanged, "head retrain changed output projection");
      }
    }
  }

  // 6. Cache round-trip and identity rejection.
  {
    const auto run = dq::runFormalCpu(config, 5, 8, 0.003f,
                                     dq::StabilityMode::LEGACY, {8});
    const auto& params = run.checkpoints.at(8);
    const auto set = rp::extractFeatures(config, params, ds.trainRows);
    rp::CacheIdentity identity;
    identity.protocol = rp::kProbeProtocolId;
    identity.config = "SELF_TEST";
    identity.seed = 5;
    identity.step = 8;
    identity.datasetHash = ds.combinedHash;
    identity.depth = static_cast<int>(config.numLayers);
    identity.repCount = rp::representationCount(2);
    identity.dim = static_cast<int>(config.dimension);
    identity.rows = ds.trainRows.size();
    identity.paramHash = rp::fnv1aParams(params);
    const auto dir = std::filesystem::temp_directory_path() / "rp-cache-test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    require(rp::writeHiddenCache(dir, identity, set), "cache write");
    rp::LayerSet loaded;
    require(rp::readHiddenCache(dir, identity, loaded), "cache read");
    require(loaded.dim == set.dim && loaded.repCount == set.repCount &&
                loaded.features == set.features,
            "cache bitwise round-trip");
    rp::CacheIdentity corrupt = identity;
    corrupt.paramHash = "fnv1a64:0000000000000000";
    require(!rp::readHiddenCache(dir, corrupt, loaded),
            "cache identity rejection");
    std::filesystem::remove_all(dir, ec);
  }

  // 7. Rollout determinism and finite-ness.
  {
    const auto run = dq::runFormalCpu(config, 5, 8, 0.003f,
                                     dq::StabilityMode::LEGACY, {8});
    const auto& params = run.checkpoints.at(8);
    const auto a = rp::headFreeRunning(config, params, 8, ds.devCases);
    const auto b = rp::headFreeRunning(config, params, 8, ds.devCases);
    require(a.allFinite && b.allFinite, "rollout finite");
    require(a.tokenExact == b.tokenExact && a.sequenceExact == b.sequenceExact &&
                a.autoregressiveNll == b.autoregressiveNll,
            "rollout determinism");
  }

  std::cout << "SELF_TEST_PASS" << std::endl;
}

// ---------------------------------------------------------------------------
// Main run
// ---------------------------------------------------------------------------
int runProduction(const ReportPaths& paths) {
  validateDatasets();
  const DataSet ds = buildDataSet();
  std::filesystem::create_directories(paths.root);
  std::filesystem::create_directories(paths.hidden);

  std::vector<RunRecord> runs;
  runs.reserve(kSpecs.size());
  const auto started = std::chrono::steady_clock::now();
  for (const auto& spec : kSpecs) {
    const auto config = modelConfig(spec.layers);
    const auto specStart = std::chrono::steady_clock::now();
    std::cerr << "[" << spec.publicId << "] regenerating canonical trajectory "
              << spec.arSelectedStep << "/" << spec.bestExactStep << "/"
              << spec.finalStep << " ..." << std::endl;
    RunRecord record;
    record.spec = &spec;
    record.run = dq::runFormalCpu(config, spec.seed, spec.finalStep, 0.003f,
                                  dq::StabilityMode::LEGACY,
                                  {spec.arSelectedStep, spec.bestExactStep,
                                   spec.finalStep});
    std::cerr << "[" << spec.publicId << "] verifying canonical anchors..."
              << std::endl;
    runAnchors(spec, record.run, ds);
    std::cerr << "[" << spec.publicId << "] anchors OK" << std::endl;
    for (const int step :
         {spec.arSelectedStep, spec.bestExactStep, spec.finalStep}) {
      std::cerr << "[" << spec.publicId << "] evaluating checkpoint step "
                << step << " ..." << std::endl;
      record.checkpoints.push_back(
          evaluateCheckpoint(spec, ds, record.run.checkpoints.at(step), step,
                             paths.hidden));
    }
    // Head retraining on the final checkpoint only.
    const auto& finalParams = record.run.checkpoints.at(spec.finalStep);
    std::cerr << "[" << spec.publicId << "] retraining output heads ..."
              << std::endl;
    record.headRetrains = retrainHeads(spec, ds, finalParams);

    // Head geometry on the final checkpoint.
    const auto headVectors = rp::headClassVectors(
        finalParams, static_cast<int>(config.dimension),
        static_cast<int>(config.vocabularySize));
    for (const auto& row : headVectors) {
      double sq = 0.0;
      for (const double x : row) sq += x * x;
      record.headRowNorms.push_back(std::sqrt(sq));
    }
    {
      // Singular values via W W^T (16x16 symmetric) Jacobi.
      const int dim = static_cast<int>(config.dimension);
      const int vocab = static_cast<int>(config.vocabularySize);
      std::vector<std::vector<double>> wwt(
          static_cast<std::size_t>(dim),
          std::vector<double>(static_cast<std::size_t>(dim), 0.0));
      for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b) {
          double acc = 0.0;
          for (int c = 0; c < vocab; ++c)
            acc += static_cast<double>(
                       finalParams.outputProjection[a * vocab + c]) *
                   static_cast<double>(
                       finalParams.outputProjection[b * vocab + c]);
          wwt[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = acc;
        }
      auto values = rp::symmetricEigenvalues(wwt);
      for (double& v : values) v = std::sqrt(std::max(0.0, v));
      record.headSingularValues = values;
      record.headEffectiveRank = rp::participationRatio(values);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - specStart);
    std::cerr << "[" << spec.publicId << "] done in " << elapsed.count()
              << " s" << std::endl;
    runs.push_back(std::move(record));
  }

  // Decision (fixed thresholds).
  std::vector<std::pair<const ConfigSpec*, const CheckpointRun*>> finalCps;
  for (const auto& run : runs)
    finalCps.push_back({run.spec, &run.checkpoints.back()});
  const Decision decision = classify(runs, finalCps);

  // Reports.
  writeDatasetAnchors(paths.root / "dataset-anchors.csv", ds);
  writeTrajectoryAnchors(paths.root / "trajectory-anchors.csv", runs, ds);
  writeBaseline(paths.root / "baseline-current-head.csv", runs);
  writeProbeSelection(paths.root / "probe-selection.csv", runs);
  writeLayerCurve(paths.root / "probe-layer-curve.csv", runs);
  writeProbeGrid(paths.root / "probe-training-grid.csv", runs);
  writeHeadRetraining(paths.root / "head-retraining.csv", runs);
  writeRepresentationMetrics(paths.root / "representation-metrics.csv", runs);
  writeHeadGeometry(paths.root / "head-geometry.csv", runs);
  writeDecision(paths.root / "decision.csv", decision);
  writeSummary(paths.root / "summary.csv", runs);

  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started);
  std::cout << "=== READOUT_PROBE_V1 RUN COMPLETE ===" << std::endl;
  std::cout << "elapsed_seconds=" << elapsed.count() << std::endl;
  std::cout << "verdict=" << decision.verdict << std::endl;
  std::cout << "reasons=" << decision.reasons << std::endl;
  for (const auto& run : runs) {
    const auto& cp = run.checkpoints.back();
    std::cout << run.spec->publicId << " final: head_dev_fr="
              << frSummary(cp.headDevFr) << std::endl;
    for (const int rep : cp.reps) {
      const auto& probe = cp.probes.at(rep);
      if (!probe.finite) continue;
      std::cout << "  rep " << rep << " "
                << rp::representationName(static_cast<int>(run.spec->layers),
                                          rep)
                << " probe_dev_fr="
                << frSummary(cp.probeDevFr.at(rep))
                << " train_tf=" << tokenSummary(cp.probeTrainTf.at(rep))
                << " dev_tf=" << tokenSummary(cp.probeDevTf.at(rep))
                << std::endl;
    }
    for (const auto& head : run.headRetrains)
      std::cout << run.spec->publicId << " head " << rp::headCandidateName(head.candidate)
                << " sel=" << head.selectedStep
                << " dev_fr=" << frSummary(head.devFr)
                << " dev_tf=" << tokenSummary(head.devTf)
                << " frozen=" << (head.frozenUnchanged ? "true" : "false")
                << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool selfTestMode = false;
    std::filesystem::path root = "build/reports/qnn-readout-representation-diagnosis";
    std::filesystem::path hidden =
        "build/reports/qnn-readout-probe/private-hidden";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") selfTestMode = true;
      else if (arg == "--run") selfTestMode = false;
      else if (arg == "--report-root" && i + 1 < argc)
        root = argv[++i];
      else if (arg == "--hidden-root" && i + 1 < argc)
        hidden = argv[++i];
      else if (arg == "--dump-rows") {
        const DataSet ds = buildDataSet();
        for (std::size_t r = 0; r < ds.allRows.size(); ++r)
          std::cout << r << "," << ds.allRows[r].truth << "\n";
        return 0;
      }
      else
        throw std::invalid_argument("UNKNOWN_ARG: " + arg);
    }
    if (selfTestMode) {
      selfTest();
      return 0;
    }
    return runProduction({root, hidden});
  } catch (const std::exception& error) {
    std::cerr << "FATAL: " << error.what() << std::endl;
    return 1;
  }
}
