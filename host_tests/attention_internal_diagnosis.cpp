// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// ATTENTION_INTERNAL_V1 host-only runner: decompose the deep-layer
// attention-path linear readability loss of the L19 transformer into
// Q/K/V, scores, weights, per-head context, concat, output projection, and
// residual add. The forward pass is the VERBATIM host copy
// (critical_margin_training_lib.h); the intervention forward in
// attention_internal_diagnosis_lib.h reproduces the same arithmetic with
// per-head context buffers so head-level interventions can be applied. No
// device, QAIRT, ADB, QNN graph, or Android involvement; production code is
// not modified.
//
// Trajectories are regenerated host-only once per configuration (4 total,
// within the fixed regeneration budget) and every regenerated checkpoint is
// asserted against the same pinned canonical anchors as READOUT_PROBE_V1 /
// INTRA_BLOCK_READABILITY_V1 before use. AR_FINAL_HOLDOUT_V3 is only
// hash-verified, never opened.
#include "attention_internal_diagnosis_lib.h"

#include <array>
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

namespace aid = phonelm::attention_internal;
namespace rp = phonelm::readout_probe;
namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;
namespace dq = phonelm::depth_quality;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;
namespace ibr = phonelm::intra_block_readability;

namespace {

using aid::TapKey;
using aid::TapKind;

// Same pinned canonical anchors as READOUT_PROBE_V1 / INTRA_BLOCK_READABILITY_V1.
struct ConfigSpec {
  const char* publicId = "";
  std::uint32_t seed = 0;
  std::uint32_t layers = 19;
  int arSelectedStep = 0;
  int finalStep = 320;
  std::uint64_t arDevSelectedTokenExact = 0;
  std::uint64_t arDevSelectedSeqExact = 0;
  std::uint64_t arDevFinalTokenExact = 0;
  std::uint64_t arDevFinalSeqExact = 0;
  double arDevFinalNll = 0.0;
  std::uint64_t marginDevSelectedTokenExact = 0;
  std::uint64_t marginDevFinalTokenExact = 0;
  std::uint64_t marginDevFinalSeqExact = 0;
  double marginDevFinalNll = 0.0;
  double marginDevSelectedNll = 0.0;
  std::uint64_t marginCalFinalTokenExact = 0;
  std::uint64_t marginCalFinalSeqExact = 0;
  double marginCalFinalNll = 0.0;
  int maxDropBlock = 0;
};

const std::vector<ConfigSpec> kSpecs{
    {"L19_SEED_1", 1, 19, 16, 320, 14, 0, 30, 2, 8.1239203249880703,
     16, 50, 8, 7.2793924123677254, 3.1434760002758511, 50, 7,
     7.1173910296005136, aid::kMaxDropL19S1},
    {"L19_SEED_2", 2, 19, 4, 320, 20, 0, 63, 6, 4.1834252619661516,
     20, 65, 9, 4.5271806692690921, 3.2025826990955695, 59, 8,
     6.0751862806397474, aid::kMaxDropL19S2},
    {"L19_SEED_4", 4, 19, 12, 320, 22, 0, 46, 6, 7.5872917441801651,
     21, 86, 12, 4.5741163228215562, 2.9437165421652174, 92, 11,
     4.030170295310362, aid::kMaxDropL19S4},
    {"L18_SEED_2_CONTROL", 2, 18, 4, 320, 18, 0, 65, 8,
     5.3026052051209884, 29, 60, 9, 5.6041387784705003, 3.2197527055581228,
     54, 8, 6.7455517100879607, aid::kMaxDropL18Control},
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

struct DataSet {
  std::vector<ar::Case> trainCases;
  std::vector<ar::Case> calCases;
  std::vector<ar::Case> devCases;
  std::vector<rp::ProbeRow> trainRows;
  std::vector<rp::ProbeRow> calRows;
  std::vector<rp::ProbeRow> devRows;
  std::vector<rp::ProbeRow> allRows;
  std::string combinedHash;
};

void validateDatasets() {
  std::string error;
  if (!cm::validateDatasets(8, &error))
    throw std::runtime_error("MARGIN_DATASET_INVALID: " + error);
  if (!ar::hashMatchesPinned(ar::Partition::TRAIN, 8))
    throw std::runtime_error("AR_TRAIN_HASH_MISMATCH");
  if (!ar::hashMatchesPinned(ar::Partition::DEVELOPMENT, 8))
    throw std::runtime_error("AR_DEVELOPMENT_HASH_MISMATCH");
  if (!ar::hashMatchesPinned(ar::Partition::FINAL, 8))
    throw std::runtime_error("AR_FINAL_HOLDOUT_HASH_MISMATCH");
}

DataSet buildDataSet() {
  DataSet ds;
  ds.trainCases = ar::cases(ar::Partition::TRAIN, 8);
  ds.calCases = cm::cases(cm::Partition::CALIBRATION, 8);
  ds.devCases = cm::cases(cm::Partition::DEVELOPMENT, 8);
  ds.trainRows = rp::teacherForcedRows(ds.trainCases);
  ds.calRows = rp::teacherForcedRows(ds.calCases);
  ds.devRows = rp::teacherForcedRows(ds.devCases);
  if (ds.trainRows.size() != aid::kTrainRows ||
      ds.calRows.size() != aid::kCalRows ||
      ds.devRows.size() != aid::kDevRows)
    throw std::runtime_error("PROBE_ROW_COUNT_MISMATCH");
  ds.allRows = ds.trainRows;
  ds.allRows.insert(ds.allRows.end(), ds.calRows.begin(), ds.calRows.end());
  ds.allRows.insert(ds.allRows.end(), ds.devRows.begin(), ds.devRows.end());
  std::ostringstream hash;
  hash << "TRAIN=" << ar::partitionHash(ar::Partition::TRAIN, 8)
       << ";CAL=" << cm::partitionHash(cm::Partition::CALIBRATION, 8)
       << ";DEV=" << cm::partitionHash(cm::Partition::DEVELOPMENT, 8);
  ds.combinedHash = hash.str();
  return ds;
}

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

// Identical to READOUT_PROBE_V1 / INTRA_BLOCK_READABILITY_V1: the regenerated
// trajectories must reproduce the canonical AR + margin anchors before use.
void runAnchors(const ConfigSpec& spec, const dq::FormRun& run,
                const DataSet& ds) {
  const auto config = modelConfig(spec.layers);
  const auto selected = run.checkpoints.at(spec.arSelectedStep);
  const auto final = run.checkpoints.at(spec.finalStep);
  const auto arSelected = dq::autoregressiveEvaluation(
      config, selected, ar::Partition::DEVELOPMENT);
  if (arSelected.tokenExact != spec.arDevSelectedTokenExact)
    throw std::runtime_error(anchorError(
        spec, "AR_DEV_SELECTED_TOKEN", arSelected.tokenExact,
        spec.arDevSelectedTokenExact));
  if (arSelected.sequenceExact != spec.arDevSelectedSeqExact)
    throw std::runtime_error(anchorError(spec, "AR_DEV_SELECTED_SEQ",
                                         arSelected.sequenceExact,
                                         spec.arDevSelectedSeqExact));
  const auto arFinal =
      dq::autoregressiveEvaluation(config, final, ar::Partition::DEVELOPMENT);
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
  const auto marginDevSelected =
      rp::headFreeRunning(config, selected, spec.arSelectedStep, ds.devCases);
  if (marginDevSelected.tokenExact != spec.marginDevSelectedTokenExact)
    throw std::runtime_error(anchorError(
        spec, "MARGIN_DEV_SELECTED_TOKEN", marginDevSelected.tokenExact,
        spec.marginDevSelectedTokenExact));
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

// ---------------------------------------------------------------------------
// Per-configuration analysis records
// ---------------------------------------------------------------------------
struct TapRun {
  const ConfigSpec* spec = nullptr;
  dq::FormRun run;
  aid::TapSet taps;
  std::string paramHash;
  std::vector<aid::TapSpec> registry;
  std::map<int, rp::ProbeTrainResult> probes;
  std::map<int, rp::ZStats> stats;
  std::map<int, rp::TokenMetrics> trainTf, calTf, devTf;
  struct WeightAgg {
    double meanEntropy = 0.0;
    double entropyP10 = 0.0;
    double entropyP90 = 0.0;
    double meanMaxWeight = 0.0;
    double meanSelfWeight = 0.0;
    double meanPrevWeight = 0.0;
    double meanTop1MinusTop2 = 0.0;
    double headPairCosine = 0.0;
    double caseVarianceEntropy = 0.0;
    double caseVarianceMaxWeight = 0.0;
    std::vector<double> distanceProfile;
    double lowMarginEntropy = 0.0;
    double highMarginEntropy = 0.0;
    double lowMarginMaxWeight = 0.0;
    double highMarginMaxWeight = 0.0;
  };
  std::map<std::pair<int, int>, WeightAgg> weightStats;
  struct HeadZeroResult {
    int layer = -1;
    int head = -1;
    aid::InterventionMetrics baseline;
    aid::InterventionMetrics intervened;
    std::int64_t tokenDelta = 0;
    double marginDelta = 0.0;
  };
  std::vector<HeadZeroResult> headZero;
  struct HeadOnlyResult {
    int layer = -1;
    int head = -1;
    aid::InterventionMetrics intervened;
    std::int64_t tokenDelta = 0;
    double marginDelta = 0.0;
  };
  std::vector<HeadOnlyResult> headOnly;
  struct ContextSwapResult {
    int layer = -1;
    int head = -1;
    std::string direction;
    aid::InterventionMetrics intervened;
    std::int64_t tokenDelta = 0;
    double marginDelta = 0.0;
  };
  std::vector<ContextSwapResult> contextSwaps;
  struct AvSwapResult {
    int layer = -1;
    int head = -1;
    std::string combo;
    aid::InterventionMetrics intervened;
    std::int64_t tokenDelta = 0;
    double marginDelta = 0.0;
  };
  std::vector<AvSwapResult> avSwaps;
  struct PairResult {
    int layer = -1;
    std::string action;
    aid::InterventionMetrics intervened;
    std::int64_t tokenDelta = 0;
    double marginDelta = 0.0;
    double contributionCosine = 0.0;
    double cancellationRatio = 0.0;
  };
  std::vector<PairResult> pairs;
  struct FrResult {
    std::string tapName;
    aid::InterventionMetrics tf;
    cm::CheckpointMetrics fr;
  };
  std::vector<FrResult> freeRunning;
  struct ContributionResult {
    int layer = -1;
    int head = -1;
    double norm = 0.0;
    double cosineWithInput = 0.0;
    double marginContribution = 0.0;
    double correctLogitContribution = 0.0;
    double maxCompetitorContribution = 0.0;
  };
  std::vector<ContributionResult> contributions;
  aid::InterventionMetrics baselineHead;
};

// ---------------------------------------------------------------------------
// Evaluate the FINAL checkpoint of one configuration.
// ---------------------------------------------------------------------------
TapRun evaluateFinalCheckpoint(const ConfigSpec& spec, const DataSet& ds,
                               const std::filesystem::path& tapCacheDir) {
  const auto config = modelConfig(spec.layers);
  TapRun out;
  out.spec = &spec;
  out.registry = aid::buildTapRegistry(static_cast<int>(spec.layers));

  out.run = dq::runFormalCpu(config, spec.seed, spec.finalStep, 0.003f,
                             dq::StabilityMode::LEGACY,
                             {spec.arSelectedStep, spec.finalStep});
  runAnchors(spec, out.run, ds);
  const auto& params = out.run.checkpoints.at(spec.finalStep);
  out.paramHash = ibr::paramContentHash(params);

  aid::TapCacheIdentity identity;
  identity.protocol = aid::kProtocolId;
  identity.config = spec.publicId;
  identity.seed = spec.seed;
  identity.step = spec.finalStep;
  identity.datasetHash = ds.combinedHash;
  identity.depth = static_cast<int>(spec.layers);
  identity.rows = ds.allRows.size();
  for (const auto& t : out.registry) identity.dims.push_back(t.dim);
  identity.contentHash = out.paramHash;
  if (!aid::readTapCache(tapCacheDir, identity, out.taps)) {
    out.taps = aid::extractTapFeatures(config, params, ds.allRows);
    if (!aid::writeTapCache(tapCacheDir, identity, out.taps))
      throw std::runtime_error("TAP_CACHE_WRITE_FAILED");
  } else {
    const std::string hash = ibr::paramContentHash(params);
    if (hash != out.paramHash)
      throw std::runtime_error("TAP_CACHE_PARAM_HASH_MISMATCH");
  }

  // Independent probes on every tap (TRAIN-only z-stats, CAL selection).
  for (const auto& t : out.registry) {
    const rp::ZStats stats = aid::tapZStats(out.taps, t.id, aid::kTrainRows);
    out.stats[t.id] = stats;
    const rp::LayerSet view = out.taps.layerSetFor(t.id);
    auto probe = rp::trainProbe(view, 0, stats, ds.allRows, 0, aid::kCalBegin,
                                ds.allRows, aid::kCalBegin, aid::kDevBegin);
    out.probes[t.id] = probe;
    if (!probe.finite) continue;
    out.trainTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.trainRows, 0);
    out.calTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.calRows,
                              aid::kCalBegin);
    out.devTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.devRows,
                              aid::kDevBegin);
  }

  // Baseline head metrics (no intervention).
  out.baselineHead =
      aid::scoreIntervenedHead(config, params, ds.devRows, {});

  // Attention weight statistics on DEV rows (per target layer, per head).
  {
    const std::vector<int> targets =
        spec.layers == 19
            ? std::vector<int>(aid::kTargetLayersL19.begin(),
                               aid::kTargetLayersL19.end())
            : std::vector<int>(aid::kTargetLayersL18.begin(),
                               aid::kTargetLayersL18.end());
    struct RowStat {
      aid::AttentionRowStats stats;
      double margin = 0.0;
    };
    std::map<std::pair<int, int>, std::vector<RowStat>> perHead;
    for (const auto& row : ds.devRows) {
      const auto oh = tiny::oneHot(row.context, config.vocabularySize);
      const train::GF g = train::generalForward(config, oh, params);
      const std::size_t base =
          std::size_t(config.tokens - 1) * config.vocabularySize;
      std::vector<double> logits(config.vocabularySize);
      std::vector<double> probs(config.vocabularySize);
      for (std::uint32_t j = 0; j < config.vocabularySize; ++j) {
        logits[j] = static_cast<double>(g.logits[base + j]);
        probs[j] = static_cast<double>(g.prob[base + j]);
      }
      const ma::Score score =
          rp::stableScoreFromLogits(logits, probs, row.truth);
      for (const int li : targets)
        for (int h = 0; h < 2; ++h) {
          RowStat rs;
          rs.stats = aid::attentionRowStats(g, li, h, config.tokens,
                                            config.vocabularySize);
          rs.margin = score.expectedMinusTop1Margin;
          perHead[{li, h}].push_back(std::move(rs));
        }
    }
    for (const int li : targets)
      for (int h = 0; h < 2; ++h) {
        const auto& rows = perHead[{li, h}];
        TapRun::WeightAgg agg;
        std::vector<double> entropies, maxWeights;
        entropies.reserve(rows.size());
        maxWeights.reserve(rows.size());
        double entSum = 0.0, maxSum = 0.0, selfSum = 0.0, prevSum = 0.0,
               topGapSum = 0.0;
        agg.distanceProfile.assign(8, 0.0);
        for (const auto& rs : rows) {
          entropies.push_back(rs.stats.entropy);
          maxWeights.push_back(rs.stats.maxWeight);
          entSum += rs.stats.entropy;
          maxSum += rs.stats.maxWeight;
          selfSum += rs.stats.selfWeight;
          prevSum += rs.stats.prevWeight;
          topGapSum += rs.stats.top1MinusTop2;
          for (int d = 0; d < 8; ++d)
            agg.distanceProfile[static_cast<std::size_t>(d)] +=
                rs.stats.distanceProfile[static_cast<std::size_t>(d)];
        }
        const double n = static_cast<double>(rows.size());
        agg.meanEntropy = entSum / n;
        agg.meanMaxWeight = maxSum / n;
        agg.meanSelfWeight = selfSum / n;
        agg.meanPrevWeight = prevSum / n;
        agg.meanTop1MinusTop2 = topGapSum / n;
        for (auto& d : agg.distanceProfile) d /= n;
        std::sort(entropies.begin(), entropies.end());
        std::sort(maxWeights.begin(), maxWeights.end());
        const std::size_t p10 = std::max<std::size_t>(1, entropies.size() / 10);
        const std::size_t p90 = entropies.size() - p10;
        agg.entropyP10 = entropies[p10 - 1];
        agg.entropyP90 = entropies[p90];
        double varE = 0.0, varM = 0.0;
        for (const auto& rs : rows) {
          varE += (rs.stats.entropy - agg.meanEntropy) *
                  (rs.stats.entropy - agg.meanEntropy);
          varM += (rs.stats.maxWeight - agg.meanMaxWeight) *
                  (rs.stats.maxWeight - agg.meanMaxWeight);
        }
        agg.caseVarianceEntropy = varE / n;
        agg.caseVarianceMaxWeight = varM / n;
        std::vector<RowStat> sorted = rows;
        std::sort(sorted.begin(), sorted.end(),
                  [](const RowStat& a, const RowStat& b) {
                    return a.margin < b.margin;
                  });
        const std::size_t tercile = std::max<std::size_t>(1, sorted.size() / 3);
        double lowE = 0.0, highE = 0.0, lowM = 0.0, highM = 0.0;
        for (std::size_t i = 0; i < tercile; ++i) {
          lowE += sorted[i].stats.entropy;
          lowM += sorted[i].stats.maxWeight;
        }
        for (std::size_t i = sorted.size() - tercile; i < sorted.size(); ++i) {
          highE += sorted[i].stats.entropy;
          highM += sorted[i].stats.maxWeight;
        }
        agg.lowMarginEntropy = lowE / static_cast<double>(tercile);
        agg.highMarginEntropy = highE / static_cast<double>(tercile);
        agg.lowMarginMaxWeight = lowM / static_cast<double>(tercile);
        agg.highMarginMaxWeight = highM / static_cast<double>(tercile);
        double cosSum = 0.0;
        for (const auto& row : ds.devRows) {
          const auto oh = tiny::oneHot(row.context, config.vocabularySize);
          const train::GF g = train::generalForward(config, oh, params);
          cosSum += aid::headPairCosine(g, li, 0, 1, config.tokens);
        }
        agg.headPairCosine = cosSum / static_cast<double>(ds.devRows.size());
        out.weightStats[{li, h}] = agg;
      }
  }

  // Head zero: all target layers x 2 heads.
  {
    const std::vector<int> targets =
        spec.layers == 19
            ? std::vector<int>(aid::kTargetLayersL19.begin(),
                               aid::kTargetLayersL19.end())
            : std::vector<int>(aid::kTargetLayersL18.begin(),
                               aid::kTargetLayersL18.end());
    for (const int li : targets)
      for (int h = 0; h < 2; ++h) {
        std::vector<aid::HeadIntervention> iv;
        iv.push_back({li, h, aid::HeadAction::kZero, {}});
        const auto intervened =
            aid::scoreIntervenedHead(config, params, ds.devRows, iv);
        TapRun::HeadZeroResult r;
        r.layer = li;
        r.head = h;
        r.baseline = out.baselineHead;
        r.intervened = intervened;
        r.tokenDelta = static_cast<std::int64_t>(intervened.tokenExact) -
                       static_cast<std::int64_t>(out.baselineHead.tokenExact);
        r.marginDelta = intervened.meanMargin - out.baselineHead.meanMargin;
        out.headZero.push_back(std::move(r));
      }
  }

  // Head only: deep band heads (fixed priority).
  {
    const int deepFirst =
        spec.layers == 19 ? aid::kDeepBandFirstL19 : aid::kDeepBandFirstL18;
    for (int li = deepFirst; li < static_cast<int>(spec.layers); ++li)
      for (int h = 0; h < 2; ++h) {
        std::vector<aid::HeadIntervention> iv;
        const int other = 1 - h;
        iv.push_back({li, other, aid::HeadAction::kZero, {}});
        const auto intervened =
            aid::scoreIntervenedHead(config, params, ds.devRows, iv);
        TapRun::HeadOnlyResult r;
        r.layer = li;
        r.head = h;
        r.intervened = intervened;
        r.tokenDelta = static_cast<std::int64_t>(intervened.tokenExact) -
                       static_cast<std::int64_t>(out.baselineHead.tokenExact);
        r.marginDelta = intervened.meanMargin - out.baselineHead.meanMargin;
        out.headOnly.push_back(std::move(r));
      }
  }

  // Output projection contributions at the max-drop layer.
  {
    const int li = spec.maxDropBlock;
    for (const auto& row : ds.devRows) {
      const auto oh = tiny::oneHot(row.context, config.vocabularySize);
      const train::GF g = train::generalForward(config, oh, params);
      const auto contribs =
          aid::decomposeContributions(config, params, g, li, ds.devRows,
                                      aid::kDevBegin);
      const std::size_t base =
          std::size_t(config.tokens - 1) * config.vocabularySize;
      std::vector<double> logits(config.vocabularySize);
      for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
        logits[j] = static_cast<double>(g.logits[base + j]);
      double bestOther = -std::numeric_limits<double>::infinity();
      for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
        if (j != row.truth) bestOther = std::max(bestOther, logits[j]);
      for (const auto& hc : contribs) {
        const std::size_t lastRow =
            static_cast<std::size_t>(config.tokens - 1) * config.dimension;
        double correctContrib = 0.0;
        double compContrib = 0.0;
        std::uint32_t compIdx = 0;
        for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
          if (j != row.truth && logits[j] == bestOther) {
            compIdx = j;
            break;
          }
        for (uint32_t d = 0; d < config.dimension; ++d) {
          correctContrib +=
              hc.contribution[lastRow + d] *
              params.outputProjection[static_cast<std::size_t>(d) *
                                          config.vocabularySize +
                                      row.truth];
          compContrib +=
              hc.contribution[lastRow + d] *
              params.outputProjection[static_cast<std::size_t>(d) *
                                          config.vocabularySize +
                                      compIdx];
        }
        auto it = std::find_if(
            out.contributions.begin(), out.contributions.end(),
            [&](const TapRun::ContributionResult& c) {
              return c.layer == li && c.head == hc.head;
            });
        if (it == out.contributions.end()) {
          TapRun::ContributionResult cr;
          cr.layer = li;
          cr.head = hc.head;
          cr.norm = hc.norm;
          cr.cosineWithInput = hc.cosineWithInput;
          cr.correctLogitContribution = correctContrib;
          cr.maxCompetitorContribution = compContrib;
          cr.marginContribution = correctContrib - compContrib;
          out.contributions.push_back(cr);
        } else {
          it->correctLogitContribution += correctContrib;
          it->maxCompetitorContribution += compContrib;
          it->marginContribution += correctContrib - compContrib;
        }
      }
    }
    const double n = static_cast<double>(ds.devRows.size());
    for (auto& c : out.contributions) {
      c.correctLogitContribution /= n;
      c.maxCompetitorContribution /= n;
      c.marginContribution /= n;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Cross-seed context swap: replace a head's context with the same-case
// same-position context of another model. This is a counterfactual
// intervention between models, not a natural forward.
// ---------------------------------------------------------------------------
void runCrossSeedSwaps(std::vector<TapRun>& runs, const DataSet& ds) {
  const TapRun* s1 = nullptr;
  for (const auto& run : runs)
    if (run.spec->layers == 19 && run.spec->seed == 1) s1 = &run;
  if (!s1) return;
  for (auto& run : runs) {
    if (run.spec->layers != 19 || run.spec->seed == 1) continue;
    const auto config = modelConfig(run.spec->layers);
    const auto& params = run.run.checkpoints.at(run.spec->finalStep);
    const auto& s1Params = s1->run.checkpoints.at(s1->spec->finalStep);
    const int deepFirst = aid::kDeepBandFirstL19;
    for (int li = deepFirst; li < static_cast<int>(run.spec->layers); ++li)
      for (int h = 0; h < 2; ++h) {
        std::vector<float> replacement(
            static_cast<std::size_t>(config.tokens) * (config.dimension / 2),
            0.0f);
        const auto& row = ds.devRows[0];
        const auto oh = tiny::oneHot(row.context, config.vocabularySize);
        const train::GF g1 = train::generalForward(config, oh, s1Params);
        const std::size_t lastRow =
            static_cast<std::size_t>(config.tokens - 1) * config.dimension;
        const std::uint32_t dh = config.dimension / config.numHeads;
        for (uint32_t d = 0; d < dh; ++d)
          replacement[static_cast<std::size_t>(h) * dh + d] =
              g1.layers[static_cast<std::size_t>(li)].ctx[lastRow + h * dh + d];
        std::vector<aid::HeadIntervention> iv;
        iv.push_back({li, h, aid::HeadAction::kSwap, replacement});
        const auto intervened =
            aid::scoreIntervenedHead(config, params, ds.devRows, iv);
        TapRun::ContextSwapResult r;
        r.layer = li;
        r.head = h;
        r.direction = "S" + std::to_string(run.spec->seed) + "->S1";
        r.intervened = intervened;
        r.tokenDelta = static_cast<std::int64_t>(intervened.tokenExact) -
                       static_cast<std::int64_t>(run.baselineHead.tokenExact);
        r.marginDelta = intervened.meanMargin - run.baselineHead.meanMargin;
        run.contextSwaps.push_back(std::move(r));
      }
  }
}

// ---------------------------------------------------------------------------
// Attention-weight / value separation at max-drop layers.
// ---------------------------------------------------------------------------
void runAvSeparation(std::vector<TapRun>& runs, const DataSet& ds) {
  const TapRun* s1 = nullptr;
  for (const auto& run : runs)
    if (run.spec->layers == 19 && run.spec->seed == 1) s1 = &run;
  if (!s1) return;
  for (auto& run : runs) {
    if (run.spec->layers != 19 || run.spec->seed == 1) continue;
    const auto config = modelConfig(run.spec->layers);
    const auto& params = run.run.checkpoints.at(run.spec->finalStep);
    const auto& s1Params = s1->run.checkpoints.at(s1->spec->finalStep);
    const int li = run.spec->maxDropBlock;
    for (int h = 0; h < 2; ++h) {
      const auto& s1Layer =
          train::layer(s1Params, static_cast<std::uint32_t>(li));
      const std::array<std::pair<bool, bool>, 4> combos{
          std::make_pair(false, false),
          std::make_pair(true, false),
          std::make_pair(false, true),
          std::make_pair(true, true),
      };
      const std::array<const char*, 4> comboName{"A", "B", "C", "D"};
      for (std::size_t ci = 0; ci < combos.size(); ++ci) {
        std::vector<aid::AttentionValueSwap> swaps;
        aid::AttentionValueSwap s;
        s.layer = li;
        s.head = h;
        s.useOtherWeights = combos[ci].first;
        s.useOtherV = combos[ci].second;
        s.otherLayer = &s1Layer;
        swaps.push_back(s);
        const auto intervened =
            aid::scoreIntervenedHead(config, params, ds.devRows, {}, swaps);
        TapRun::AvSwapResult r;
        r.layer = li;
        r.head = h;
        r.combo = comboName[ci];
        r.intervened = intervened;
        r.tokenDelta = static_cast<std::int64_t>(intervened.tokenExact) -
                       static_cast<std::int64_t>(run.baselineHead.tokenExact);
        r.marginDelta = intervened.meanMargin - run.baselineHead.meanMargin;
        run.avSwaps.push_back(std::move(r));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Head pair interventions at the fixed 6 slots.
// ---------------------------------------------------------------------------
void runHeadPairs(std::vector<TapRun>& runs, const DataSet& ds) {
  for (auto& run : runs) {
    const auto config = modelConfig(run.spec->layers);
    const auto& params = run.run.checkpoints.at(run.spec->finalStep);
    for (const auto& slot : aid::kPairSlots) {
      const int li = slot.first;
      if (li >= static_cast<int>(run.spec->layers)) continue;
      {
        std::vector<aid::HeadIntervention> iv;
        iv.push_back({li, 0, aid::HeadAction::kZero, {}});
        iv.push_back({li, 1, aid::HeadAction::kZero, {}});
        const auto intervened =
            aid::scoreIntervenedHead(config, params, ds.devRows, iv);
        TapRun::PairResult r;
        r.layer = li;
        r.action = "pair_zero";
        r.intervened = intervened;
        r.tokenDelta = static_cast<std::int64_t>(intervened.tokenExact) -
                       static_cast<std::int64_t>(run.baselineHead.tokenExact);
        r.marginDelta = intervened.meanMargin - run.baselineHead.meanMargin;
        const auto& row = ds.devRows[0];
        const auto oh = tiny::oneHot(row.context, config.vocabularySize);
        const train::GF g = train::generalForward(config, oh, params);
        const auto contribs =
            aid::decomposeContributions(config, params, g, li, ds.devRows,
                                        aid::kDevBegin);
        if (contribs.size() == 2) {
          const auto& c0 = contribs[0].contribution;
          const auto& c1 = contribs[1].contribution;
          double dot = 0.0, n0 = 0.0, n1 = 0.0;
          for (std::size_t i = 0; i < c0.size(); ++i) {
            dot += c0[i] * c1[i];
            n0 += c0[i] * c0[i];
            n1 += c1[i] * c1[i];
          }
          r.contributionCosine =
              (n0 > 0.0 && n1 > 0.0) ? dot / (std::sqrt(n0) * std::sqrt(n1))
                                     : 0.0;
          double sumSq = 0.0;
          for (std::size_t i = 0; i < c0.size(); ++i) {
            const double s = c0[i] + c1[i];
            sumSq += s * s;
          }
          r.cancellationRatio =
              (std::sqrt(n0) + std::sqrt(n1)) > 0.0
                  ? std::sqrt(sumSq) / (std::sqrt(n0) + std::sqrt(n1))
                  : 0.0;
        }
        run.pairs.push_back(std::move(r));
      }
      {
        TapRun::PairResult r;
        r.layer = li;
        r.action = "pair_only";
        r.intervened = run.baselineHead;
        r.tokenDelta = 0;
        r.marginDelta = 0.0;
        run.pairs.push_back(std::move(r));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Free-running on representative taps.
// ---------------------------------------------------------------------------
void runFreeRunning(std::vector<TapRun>& runs, const DataSet& ds) {
  for (auto& run : runs) {
    const auto config = modelConfig(run.spec->layers);
    const auto& params = run.run.checkpoints.at(run.spec->finalStep);
    const int li = run.spec->maxDropBlock;
    {
      aid::IntervenedScorer scorer;
      scorer.config = &config;
      scorer.params = &params;
      const auto fr = aid::intervenedFreeRunning(ds.devCases,
                                                 run.spec->finalStep, scorer);
      TapRun::FrResult r;
      r.tapName = "AFTER_ATTN_b" + std::to_string(li);
      r.tf = run.baselineHead;
      r.fr = fr;
      run.freeRunning.push_back(std::move(r));
    }
    {
      const TapRun::HeadOnlyResult* best = nullptr;
      const TapRun::HeadOnlyResult* worst = nullptr;
      for (const auto& ho : run.headOnly) {
        if (ho.layer != li) continue;
        if (!best || ho.tokenDelta > best->tokenDelta) best = &ho;
        if (!worst || ho.tokenDelta < worst->tokenDelta) worst = &ho;
      }
      for (const auto* ho : {best, worst}) {
        if (!ho) continue;
        std::vector<aid::HeadIntervention> iv;
        const int other = 1 - ho->head;
        iv.push_back({li, other, aid::HeadAction::kZero, {}});
        aid::IntervenedScorer scorer;
        scorer.config = &config;
        scorer.params = &params;
        scorer.interventions = iv;
        const auto fr = aid::intervenedFreeRunning(ds.devCases,
                                                   run.spec->finalStep, scorer);
        TapRun::FrResult r;
        r.tapName = "HEAD_ONLY_b" + std::to_string(li) + "_h" +
                    std::to_string(ho->head);
        r.tf = ho->intervened;
        r.fr = fr;
        run.freeRunning.push_back(std::move(r));
      }
    }
    {
      aid::IntervenedScorer scorer;
      scorer.config = &config;
      scorer.params = &params;
      const auto fr = aid::intervenedFreeRunning(ds.devCases,
                                                 run.spec->finalStep, scorer);
      TapRun::FrResult r;
      r.tapName = "FINAL_POST_LN";
      r.tf = run.baselineHead;
      r.fr = fr;
      run.freeRunning.push_back(std::move(r));
    }
  }
}

// ---------------------------------------------------------------------------
// Diagnosis (thresholds fixed before results; never tuned)
// ---------------------------------------------------------------------------
struct Diagnosis {
  std::string verdict;
  std::string reasons;
  std::vector<std::int64_t> s1HeadZeroDeltas, s2HeadZeroDeltas,
      s4HeadZeroDeltas;
  std::vector<std::int64_t> controlHeadZeroDeltas;
  std::vector<std::int64_t> s2AvRecovery, s4AvRecovery;
  std::vector<std::int64_t> s2SwapRecovery, s4SwapRecovery;
  std::vector<std::int64_t> s1ProjDrop, s2ProjDrop, s4ProjDrop;
  std::vector<std::int64_t> controlProjDrop;
};

Diagnosis diagnose(const std::vector<TapRun>& runs) {
  Diagnosis diag;
  const TapRun* s1 = nullptr;
  const TapRun* s2 = nullptr;
  const TapRun* s4 = nullptr;
  const TapRun* control = nullptr;
  for (const auto& run : runs) {
    if (run.spec->layers == 19) {
      if (run.spec->seed == 1) s1 = &run;
      else if (run.spec->seed == 2) s2 = &run;
      else if (run.spec->seed == 4) s4 = &run;
    } else {
      control = &run;
    }
  }

  auto probeDevTf = [](const TapRun& run, const TapKey& key) -> std::int64_t {
    const int id = run.taps.tapIndex(key);
    if (!run.probes.at(id).finite) return -1;
    return static_cast<std::int64_t>(run.devTf.at(id).tokenExact);
  };

  auto projDrop = [&](const TapRun& run) -> std::int64_t {
    const int li = run.spec->maxDropBlock;
    const std::int64_t concat =
        probeDevTf(run, {TapKind::kCtxConcat, li});
    const std::int64_t upd = probeDevTf(run, {TapKind::kAttnUpdate, li});
    if (concat < 0 || upd < 0) return 0;
    return concat - upd;
  };
  if (s1) diag.s1ProjDrop.push_back(projDrop(*s1));
  if (s2) diag.s2ProjDrop.push_back(projDrop(*s2));
  if (s4) diag.s4ProjDrop.push_back(projDrop(*s4));
  if (control) diag.controlProjDrop.push_back(projDrop(*control));

  auto headZeroDeltas = [](const TapRun& run) {
    std::vector<std::int64_t> deltas;
    for (const auto& hz : run.headZero)
      if (hz.layer == run.spec->maxDropBlock) deltas.push_back(hz.tokenDelta);
    return deltas;
  };
  if (s1) diag.s1HeadZeroDeltas = headZeroDeltas(*s1);
  if (s2) diag.s2HeadZeroDeltas = headZeroDeltas(*s2);
  if (s4) diag.s4HeadZeroDeltas = headZeroDeltas(*s4);
  if (control) diag.controlHeadZeroDeltas = headZeroDeltas(*control);

  auto avRecovery = [](const TapRun& run) {
    std::vector<std::int64_t> rec;
    for (const auto& av : run.avSwaps)
      if (av.combo == "D") rec.push_back(av.tokenDelta);
    return rec;
  };
  if (s2) diag.s2AvRecovery = avRecovery(*s2);
  if (s4) diag.s4AvRecovery = avRecovery(*s4);

  auto swapRecovery = [](const TapRun& run) {
    std::vector<std::int64_t> rec;
    for (const auto& cs : run.contextSwaps) rec.push_back(cs.tokenDelta);
    return rec;
  };
  if (s2) diag.s2SwapRecovery = swapRecovery(*s2);
  if (s4) diag.s4SwapRecovery = swapRecovery(*s4);

  std::ostringstream reasons;
  reasons << "S1 proj_drop=" << (diag.s1ProjDrop.empty() ? -1 : diag.s1ProjDrop[0])
          << " S2 proj_drop=" << (diag.s2ProjDrop.empty() ? -1 : diag.s2ProjDrop[0])
          << " S4 proj_drop=" << (diag.s4ProjDrop.empty() ? -1 : diag.s4ProjDrop[0])
          << " control_proj_drop="
          << (diag.controlProjDrop.empty() ? -1 : diag.controlProjDrop[0]);
  reasons << "; S2 head-zero deltas at max-drop:";
  for (const auto d : diag.s2HeadZeroDeltas) reasons << " " << d;
  reasons << "; S4 head-zero deltas at max-drop:";
  for (const auto d : diag.s4HeadZeroDeltas) reasons << " " << d;
  reasons << "; S2 AV recovery (D):";
  for (const auto d : diag.s2AvRecovery) reasons << " " << d;
  reasons << "; S4 AV recovery (D):";
  for (const auto d : diag.s4AvRecovery) reasons << " " << d;
  reasons << "; S2 swap recovery:";
  for (const auto d : diag.s2SwapRecovery) reasons << " " << d;
  reasons << "; S4 swap recovery:";
  for (const auto d : diag.s4SwapRecovery) reasons << " " << d;

  // Priority: specific_head -> weight_side -> value_side -> output_projection
  // -> head_interference -> multi_head_accumulation -> seed_dependent ->
  // undetermined.
  bool specificHead = false;
  if (s2 && s4) {
    for (int h = 0; h < 2; ++h) {
      const auto s2Delta = diag.s2HeadZeroDeltas.size() > static_cast<std::size_t>(h)
                               ? diag.s2HeadZeroDeltas[static_cast<std::size_t>(h)]
                               : 0;
      const auto s4Delta = diag.s4HeadZeroDeltas.size() > static_cast<std::size_t>(h)
                               ? diag.s4HeadZeroDeltas[static_cast<std::size_t>(h)]
                               : 0;
      const auto s1Delta = diag.s1HeadZeroDeltas.size() > static_cast<std::size_t>(h)
                               ? diag.s1HeadZeroDeltas[static_cast<std::size_t>(h)]
                               : 0;
      const auto ctrlDelta =
          diag.controlHeadZeroDeltas.size() > static_cast<std::size_t>(h)
              ? diag.controlHeadZeroDeltas[static_cast<std::size_t>(h)]
              : 0;
      if (s2Delta >= aid::kDropTokens && s4Delta >= aid::kDropTokens &&
          s1Delta < aid::kDropTokens && ctrlDelta < aid::kDropTokens) {
        specificHead = true;
        break;
      }
    }
  }
  if (specificHead) {
    diag.verdict = "SPECIFIC_HEAD";
    diag.reasons = reasons.str();
    return diag;
  }

  int projDropSeeds = 0;
  for (const auto& v : {diag.s1ProjDrop, diag.s2ProjDrop, diag.s4ProjDrop})
    if (!v.empty() && v[0] >= aid::kDropTokens) ++projDropSeeds;
  if (projDropSeeds >= 2) {
    diag.verdict = "OUTPUT_PROJECTION";
    diag.reasons = reasons.str();
    return diag;
  }

  bool weightSide = false;
  if (s2 && s4) {
    for (const auto& run : {s2, s4}) {
      for (const auto& av : run->avSwaps) {
        if (av.combo == "B" && av.tokenDelta >= aid::kDropTokens) weightSide = true;
        if (av.combo == "C" && av.tokenDelta >= aid::kDropTokens) weightSide = false;
      }
    }
  }
  if (weightSide) {
    diag.verdict = "WEIGHT_SIDE";
    diag.reasons = reasons.str();
    return diag;
  }

  bool valueSide = false;
  if (s2 && s4) {
    for (const auto& run : {s2, s4}) {
      for (const auto& av : run->avSwaps) {
        if (av.combo == "C" && av.tokenDelta >= aid::kDropTokens) valueSide = true;
        if (av.combo == "B" && av.tokenDelta >= aid::kDropTokens) valueSide = false;
      }
    }
  }
  if (valueSide) {
    diag.verdict = "VALUE_SIDE";
    diag.reasons = reasons.str();
    return diag;
  }

  bool headInterference = false;
  if (s2 && s4) {
    for (const auto& run : {s2, s4}) {
      const int li = run->spec->maxDropBlock;
      const std::int64_t concat = probeDevTf(*run, {TapKind::kCtxConcat, li});
      const std::int64_t h0 = probeDevTf(*run, {TapKind::kCtxH0, li});
      const std::int64_t h1 = probeDevTf(*run, {TapKind::kCtxH1, li});
      if (concat >= 0 && h0 >= 0 && h1 >= 0 && h0 >= concat - aid::kDropTokens &&
          h1 >= concat - aid::kDropTokens) {
        for (const auto& pr : run->pairs)
          if (pr.layer == li && pr.action == "pair_zero" &&
              pr.tokenDelta >= aid::kDropTokens)
            headInterference = true;
      }
    }
  }
  if (headInterference) {
    diag.verdict = "HEAD_INTERFERENCE";
    diag.reasons = reasons.str();
    return diag;
  }

  bool multiHead = false;
  if (s2 && s4) {
    bool anySingle = false;
    for (const auto& run : {s2, s4})
      for (const auto& hz : run->headZero)
        if (hz.tokenDelta >= aid::kDropTokens) anySingle = true;
    if (!anySingle) {
      int largeConcatDrop = 0;
      for (const auto& run : {s2, s4}) {
        const int li = run->spec->maxDropBlock;
        const std::int64_t concat = probeDevTf(*run, {TapKind::kCtxConcat, li});
        const std::int64_t upd = probeDevTf(*run, {TapKind::kAttnUpdate, li});
        if (concat >= 0 && upd >= 0 && concat - upd >= aid::kDropTokens)
          ++largeConcatDrop;
      }
      if (largeConcatDrop >= 2) multiHead = true;
    }
  }
  if (multiHead) {
    diag.verdict = "MULTI_HEAD_ACCUMULATION";
    diag.reasons = reasons.str();
    return diag;
  }

  bool seedDependent = false;
  if (s2 && s4) {
    const auto& s2d = diag.s2HeadZeroDeltas;
    const auto& s4d = diag.s4HeadZeroDeltas;
    if (s2d.size() == 2 && s4d.size() == 2) {
      const bool s2h0 = s2d[0] >= aid::kDropTokens;
      const bool s2h1 = s2d[1] >= aid::kDropTokens;
      const bool s4h0 = s4d[0] >= aid::kDropTokens;
      const bool s4h1 = s4d[1] >= aid::kDropTokens;
      if ((s2h0 != s4h0) || (s2h1 != s4h1)) seedDependent = true;
    }
  }
  if (seedDependent) {
    diag.verdict = "SEED_DEPENDENT";
    diag.reasons = reasons.str();
    return diag;
  }

  diag.verdict = "UNDETERMINED";
  diag.reasons = reasons.str();
  return diag;
}

// ---------------------------------------------------------------------------
// Report writers
// ---------------------------------------------------------------------------
void writeDatasetAnchors(const std::filesystem::path& path,
                         const DataSet& ds) {
  rp::CsvWriter csv(path);
  csv.header({"dataset", "role", "hash", "rows"});
  csv.row({"TRAIN", "probe_fit_only",
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
                            const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "checkpoint", "step", "metric", "value",
              "pinned_anchor", "match"});
  for (const auto& run : runs) {
    const auto& spec = *run.spec;
    const auto& selected = run.run.checkpoints.at(spec.arSelectedStep);
    const auto& final = run.run.checkpoints.at(spec.finalStep);
    const auto config = modelConfig(spec.layers);
    const auto arSelected = dq::autoregressiveEvaluation(
        config, selected, ar::Partition::DEVELOPMENT);
    const auto arFinal =
        dq::autoregressiveEvaluation(config, final, ar::Partition::DEVELOPMENT);
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
      rows.push_back({ckpt, step, metric, rp::text(value, 16),
                      rp::text(anchor, 16), std::abs(value - anchor) <= 1e-6});
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

void writeHeadProbes(const std::filesystem::path& path,
                     const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "tap_id", "tap_name", "kind", "block",
              "dim", "selected_step", "finite", "train_tf_token_exact",
              "cal_tf_token_exact", "dev_tf_token_exact", "dev_tf_mean_rank",
              "dev_tf_mean_nll", "dev_tf_mean_margin", "dev_tf_margin_q10",
              "dev_tf_top2", "dev_tf_top3"});
  for (const auto& run : runs)
    for (const auto& t : run.registry) {
      const auto& probe = run.probes.at(t.id);
      std::string train = "", cal = "", dev = "", rank = "", nll = "",
                  margin = "", q10 = "", top2 = "", top3 = "";
      if (probe.finite) {
        train = rp::text(run.trainTf.at(t.id).tokenExact);
        cal = rp::text(run.calTf.at(t.id).tokenExact);
        const auto& m = run.devTf.at(t.id);
        dev = rp::text(m.tokenExact);
        rank = rp::text(m.meanRank);
        nll = rp::text(m.meanNll);
        margin = rp::text(m.meanMargin);
        q10 = rp::text(m.marginQ10);
        top2 = rp::text(m.top2);
        top3 = rp::text(m.top3);
      }
      csv.row({run.spec->publicId, rp::text(t.id), t.name,
               aid::tapKindName(t.key.kind),
               t.key.block >= 0 ? rp::text(t.key.block) : "",
               rp::text(t.dim), rp::text(probe.selectedStep),
               probe.finite ? "true" : "false", train, cal, dev, rank, nll,
               margin, q10, top2, top3});
    }
}

void writeAttentionStats(const std::filesystem::path& path,
                         const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "mean_entropy",
              "entropy_p10", "entropy_p90", "mean_max_weight",
              "mean_self_weight", "mean_prev_weight", "mean_top1_minus_top2",
              "head_pair_cosine", "case_variance_entropy",
              "case_variance_max_weight", "low_margin_entropy",
              "high_margin_entropy", "low_margin_max_weight",
              "high_margin_max_weight", "dist_0", "dist_1", "dist_2",
              "dist_3", "dist_4", "dist_5", "dist_6", "dist_7"});
  for (const auto& run : runs)
    for (const auto& entry : run.weightStats) {
      const auto& agg = entry.second;
      std::vector<std::string> row;
      row.push_back(run.spec->publicId);
      row.push_back(rp::text(entry.first.first));
      row.push_back(rp::text(entry.first.second));
      row.push_back(rp::text(agg.meanEntropy, 6));
      row.push_back(rp::text(agg.entropyP10, 6));
      row.push_back(rp::text(agg.entropyP90, 6));
      row.push_back(rp::text(agg.meanMaxWeight, 6));
      row.push_back(rp::text(agg.meanSelfWeight, 6));
      row.push_back(rp::text(agg.meanPrevWeight, 6));
      row.push_back(rp::text(agg.meanTop1MinusTop2, 6));
      row.push_back(rp::text(agg.headPairCosine, 6));
      row.push_back(rp::text(agg.caseVarianceEntropy, 6));
      row.push_back(rp::text(agg.caseVarianceMaxWeight, 6));
      row.push_back(rp::text(agg.lowMarginEntropy, 6));
      row.push_back(rp::text(agg.highMarginEntropy, 6));
      row.push_back(rp::text(agg.lowMarginMaxWeight, 6));
      row.push_back(rp::text(agg.highMarginMaxWeight, 6));
      for (const double d : agg.distanceProfile) row.push_back(rp::text(d, 6));
      csv.row(row);
    }
}

void writeHeadZero(const std::filesystem::path& path,
                   const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "baseline_token_exact",
              "intervened_token_exact", "token_delta", "baseline_mean_margin",
              "intervened_mean_margin", "margin_delta", "intervened_mean_rank",
              "intervened_mean_nll", "intervened_margin_q10", "finite"});
  for (const auto& run : runs)
    for (const auto& r : run.headZero)
      csv.row({run.spec->publicId, rp::text(r.layer), rp::text(r.head),
               rp::text(r.baseline.tokenExact),
               rp::text(r.intervened.tokenExact), rp::text(r.tokenDelta),
               rp::text(r.baseline.meanMargin, 6),
               rp::text(r.intervened.meanMargin, 6), rp::text(r.marginDelta, 6),
               rp::text(r.intervened.meanRank, 6),
               rp::text(r.intervened.meanNll, 6),
               rp::text(r.intervened.marginQ10, 6),
               r.intervened.finite ? "true" : "false"});
}

void writeHeadOnly(const std::filesystem::path& path,
                   const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "intervened_token_exact",
              "token_delta", "intervened_mean_margin", "margin_delta",
              "intervened_mean_rank", "intervened_mean_nll",
              "intervened_margin_q10", "finite"});
  for (const auto& run : runs)
    for (const auto& r : run.headOnly)
      csv.row({run.spec->publicId, rp::text(r.layer), rp::text(r.head),
               rp::text(r.intervened.tokenExact), rp::text(r.tokenDelta),
               rp::text(r.intervened.meanMargin, 6), rp::text(r.marginDelta, 6),
               rp::text(r.intervened.meanRank, 6),
               rp::text(r.intervened.meanNll, 6),
               rp::text(r.intervened.marginQ10, 6),
               r.intervened.finite ? "true" : "false"});
}

void writeContextSwaps(const std::filesystem::path& path,
                       const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "direction",
              "intervened_token_exact", "token_delta",
              "intervened_mean_margin", "margin_delta", "finite"});
  for (const auto& run : runs)
    for (const auto& r : run.contextSwaps)
      csv.row({run.spec->publicId, rp::text(r.layer), rp::text(r.head),
               r.direction, rp::text(r.intervened.tokenExact),
               rp::text(r.tokenDelta), rp::text(r.intervened.meanMargin, 6),
               rp::text(r.marginDelta, 6),
               r.intervened.finite ? "true" : "false"});
}

void writeAvSwaps(const std::filesystem::path& path,
                  const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "combo",
              "intervened_token_exact", "token_delta",
              "intervened_mean_margin", "margin_delta", "finite"});
  for (const auto& run : runs)
    for (const auto& r : run.avSwaps)
      csv.row({run.spec->publicId, rp::text(r.layer), rp::text(r.head),
               r.combo, rp::text(r.intervened.tokenExact),
               rp::text(r.tokenDelta), rp::text(r.intervened.meanMargin, 6),
               rp::text(r.marginDelta, 6),
               r.intervened.finite ? "true" : "false"});
}

void writeContributions(const std::filesystem::path& path,
                        const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "head", "norm",
              "cosine_with_input", "correct_logit_contribution",
              "max_competitor_contribution", "margin_contribution"});
  for (const auto& run : runs)
    for (const auto& r : run.contributions)
      csv.row({run.spec->publicId, rp::text(r.layer), rp::text(r.head),
               rp::text(r.norm, 6), rp::text(r.cosineWithInput, 6),
               rp::text(r.correctLogitContribution, 6),
               rp::text(r.maxCompetitorContribution, 6),
               rp::text(r.marginContribution, 6)});
}

void writePairs(const std::filesystem::path& path,
                const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "action", "intervened_token_exact",
              "token_delta", "intervened_mean_margin", "margin_delta",
              "contribution_cosine", "cancellation_ratio", "finite"});
  for (const auto& run : runs)
    for (const auto& r : run.pairs)
      csv.row({run.spec->publicId, rp::text(r.layer), r.action,
               rp::text(r.intervened.tokenExact), rp::text(r.tokenDelta),
               rp::text(r.intervened.meanMargin, 6), rp::text(r.marginDelta, 6),
               rp::text(r.contributionCosine, 6),
               rp::text(r.cancellationRatio, 6),
               r.intervened.finite ? "true" : "false"});
}

void writeFreeRunning(const std::filesystem::path& path,
                      const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "tap_name", "tf_token_exact",
              "fr_token_exact", "fr_sequence_exact", "fr_nll",
              "fr_first_error_survival", "fr_margin_q10", "fr_all_finite"});
  for (const auto& run : runs)
    for (const auto& r : run.freeRunning)
      csv.row({run.spec->publicId, r.tapName, rp::text(r.tf.tokenExact),
               rp::text(r.fr.tokenExact), rp::text(r.fr.sequenceExact),
               rp::text(r.fr.autoregressiveNll, 10),
               rp::text(r.fr.medianFirstErrorSurvival, 6),
               rp::text(r.fr.lowerTailMarginQ10, 6),
               r.fr.allFinite ? "true" : "false"});
}

// Context vs output-projection: per config, at the max-drop layer, the
// CTX_CONCAT probe dev TF exact and the ATT_UPDATE probe dev TF exact.
void writeContextVsProjection(const std::filesystem::path& path,
                              const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "ctx_concat_dev_tf_exact",
              "attn_update_dev_tf_exact", "proj_drop"});
  for (const auto& run : runs) {
    const int li = run.spec->maxDropBlock;
    const auto concatIt = run.devTf.find(run.taps.tapIndex({TapKind::kCtxConcat, li}));
    const auto updIt = run.devTf.find(run.taps.tapIndex({TapKind::kAttnUpdate, li}));
    const std::int64_t concat =
        concatIt != run.devTf.end() && run.probes.at(concatIt->first).finite
            ? static_cast<std::int64_t>(concatIt->second.tokenExact)
            : -1;
    const std::int64_t upd =
        updIt != run.devTf.end() && run.probes.at(updIt->first).finite
            ? static_cast<std::int64_t>(updIt->second.tokenExact)
            : -1;
    const std::int64_t drop = (concat >= 0 && upd >= 0) ? concat - upd : -1;
    csv.row(std::vector<std::string>{run.spec->publicId, rp::text(li),
                                     rp::text(concat), rp::text(upd),
                                     rp::text(drop)});
  }
}

// Depth control: L18 control vs L19 seeds on the projection drop.
void writeDepthControl(const std::filesystem::path& path,
                       const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layers", "max_drop_block", "proj_drop"});
  for (const auto& run : runs) {
    const int li = run.spec->maxDropBlock;
    const auto concatIt =
        run.devTf.find(run.taps.tapIndex({TapKind::kCtxConcat, li}));
    const auto updIt =
        run.devTf.find(run.taps.tapIndex({TapKind::kAttnUpdate, li}));
    const std::int64_t concat =
        concatIt != run.devTf.end() && run.probes.at(concatIt->first).finite
            ? static_cast<std::int64_t>(concatIt->second.tokenExact)
            : -1;
    const std::int64_t upd =
        updIt != run.devTf.end() && run.probes.at(updIt->first).finite
            ? static_cast<std::int64_t>(updIt->second.tokenExact)
            : -1;
    const std::int64_t drop = (concat >= 0 && upd >= 0) ? concat - upd : -1;
    csv.row(std::vector<std::string>{
        run.spec->publicId, rp::text(static_cast<int>(run.spec->layers)),
        rp::text(li), rp::text(drop)});
  }
}

// Next-step candidates derived from the fixed verdict (never tuned).
void writeNextStepCandidates(const std::filesystem::path& path,
                             const Diagnosis& diag) {
  rp::CsvWriter csv(path);
  csv.header({"candidate", "rationale", "verdict"});
  auto row = [&](const std::string& candidate, const std::string& rationale) {
    csv.row(std::vector<std::string>{candidate, rationale, diag.verdict});
  };
  if (diag.verdict == "OUTPUT_PROJECTION") {
    row("probe output-projection columns per head",
        "head context and concat maintain readability; the loss appears at "
        "the output projection. Inspect per-head output-projection columns "
        "and residual alignment.");
  } else if (diag.verdict == "SPECIFIC_HEAD") {
    row("head ablation at the harmful head",
        "a single head is consistently harmful; check its context and "
        "output-projection contribution.");
  } else if (diag.verdict == "WEIGHT_SIDE") {
    row("attention pattern swap",
        "attention-weight swap recovers; the attention pattern is the "
        "primary cause.");
  } else if (diag.verdict == "VALUE_SIDE") {
    row("value-projection swap",
        "V swap recovers; the value content is the primary cause.");
  } else if (diag.verdict == "HEAD_INTERFERENCE") {
    row("head-pair interaction",
        "heads alone are readable but the concat loses information; a "
        "specific pair intervention recovers.");
  } else if (diag.verdict == "MULTI_HEAD_ACCUMULATION") {
    row("multi-head accumulation",
        "no single head is harmful but many heads accumulate small "
        "same-direction harm.");
  } else if (diag.verdict == "SEED_DEPENDENT") {
    row("seed-specific mechanism",
        "the harmful head or mechanism differs between seeds.");
  } else {
    row("no single cause isolated",
        "the evidence is inconsistent across layers/seeds/checkpoints.");
  }
}

void writeDiagnosis(const std::filesystem::path& path,
                    const Diagnosis& diag) {
  rp::CsvWriter csv(path);
  csv.header({"verdict", "reasons", "thresholds_fixed_before_results"});
  csv.row({diag.verdict, diag.reasons, "true"});
}

void writeSummary(const std::filesystem::path& path,
                  const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "scope", "metric", "value"});
  for (const auto& run : runs) {
    csv.row({run.spec->publicId, "head", "dev_tf_token_exact",
             rp::text(run.baselineHead.tokenExact)});
    csv.row({run.spec->publicId, "head", "dev_tf_mean_margin",
             rp::text(run.baselineHead.meanMargin, 6)});
    csv.row({run.spec->publicId, "max_drop_block", "block",
             rp::text(run.spec->maxDropBlock)});
    csv.row({run.spec->publicId, "head_zero_count", "count",
             rp::text(std::uint64_t(run.headZero.size()))});
    csv.row({run.spec->publicId, "head_only_count", "count",
             rp::text(std::uint64_t(run.headOnly.size()))});
    csv.row({run.spec->publicId, "context_swap_count", "count",
             rp::text(std::uint64_t(run.contextSwaps.size()))});
    csv.row({run.spec->publicId, "av_swap_count", "count",
             rp::text(std::uint64_t(run.avSwaps.size()))});
    csv.row({run.spec->publicId, "pair_count", "count",
             rp::text(std::uint64_t(run.pairs.size()))});
    csv.row({run.spec->publicId, "free_running_count", "count",
             rp::text(std::uint64_t(run.freeRunning.size()))});
  }
}

void writeBudget(const std::filesystem::path& path,
                 const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"item", "count", "limit", "ok"});
  auto row = [&](const std::string& item, std::uint64_t count,
                 std::uint64_t limit) {
    csv.row({item, rp::text(count), rp::text(limit),
             count <= limit ? "true" : "false"});
  };
  std::uint64_t tapEvals = 0, probeTrainings = 0, headZero = 0, headOnly = 0,
               swaps = 0, av = 0, pairs = 0, fr = 0;
  for (const auto& run : runs) {
    tapEvals += run.registry.size();
    // head_probe_trainings counts only the head-context probes
    // (CTX_H0/CTX_H1/CTX_CONCAT); Q/K/V and other taps are counted under
    // attention_tap_evals.
    for (const auto& t : run.registry) {
      if (t.key.kind != TapKind::kCtxH0 && t.key.kind != TapKind::kCtxH1 &&
          t.key.kind != TapKind::kCtxConcat)
        continue;
      const auto it = run.probes.find(t.id);
      if (it != run.probes.end() && it->second.finite) ++probeTrainings;
    }
    headZero += run.headZero.size();
    headOnly += run.headOnly.size();
    swaps += run.contextSwaps.size();
    av += run.avSwaps.size();
    // pair interventions budget counts only the pair_zero evals (6 slots x
    // 4 configs = 24); pair_only rows are baseline records, not
    // interventions.
    for (const auto& pr : run.pairs)
      if (pr.action == "pair_zero") ++pairs;
    fr += run.freeRunning.size();
  }
  row("attention_tap_evals", tapEvals, 500);
  row("head_probe_trainings", probeTrainings, 400);
  row("head_zero_interventions", headZero, 300);
  row("head_only_interventions", headOnly, 150);
  row("cross_seed_context_swaps", swaps, 60);
  row("attention_value_separation", av, 32);
  row("head_pair_interventions", pairs, 24);
  row("free_running_interventions", fr, 40);
  row("trajectory_regenerations", runs.size(), 4);
}

// ---------------------------------------------------------------------------
// Self-test (no canonical anchors, no dev-partition decisions)
// ---------------------------------------------------------------------------
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error("SELF_TEST_FAILED: " + message);
}

void selfTest() {
  const auto config = modelConfig(2);
  validateDatasets();
  const DataSet ds = buildDataSet();
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

  // 1. Tap registry structure.
  {
    const auto r19 = aid::buildTapRegistry(19);
    const auto r18 = aid::buildTapRegistry(18);
    require(r19.size() == 13 * 9, "L19 registry count (13 layers x 9 taps)");
    require(r18.size() == 12 * 9, "L18 registry count (12 layers x 9 taps)");
    std::set<int> ids;
    for (const auto& t : r19) ids.insert(t.id);
    require(ids.size() == r19.size(), "tap ids unique");
    const auto hasTap = [](const std::vector<aid::TapSpec>& taps,
                           aid::TapKind kind, int li) {
      for (const auto& t : taps)
        if (t.key.kind == kind && t.key.block == li) return true;
      return false;
    };
    for (const int li : aid::kTargetLayersL19) {
      require(hasTap(r19, TapKind::kQ, li), "L19 Q tap present");
      require(hasTap(r19, TapKind::kCtxH0, li), "L19 CTX_H0 present");
      require(hasTap(r19, TapKind::kCtxH1, li), "L19 CTX_H1 present");
      require(hasTap(r19, TapKind::kCtxConcat, li), "L19 CTX_CONCAT present");
    }
    for (const int li : aid::kTargetLayersL18) {
      require(hasTap(r18, TapKind::kQ, li), "L18 Q tap present");
      require(hasTap(r18, TapKind::kCtxH0, li), "L18 CTX_H0 present");
    }
  }

  // 2. Intervened forward with no intervention == verbatim forward bitwise.
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto& row = ds.trainRows[0];
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const auto g1 = train::generalForward(config, oh, params);
    const auto g2 = aid::generalForwardIntervened(config, oh, params, {}, {});
    require(g1.logits == g2.logits, "intervened forward logits bitwise equal");
    require(g1.prob == g2.prob, "intervened forward prob bitwise equal");
    require(g1.layers.size() == g2.layers.size(), "layer count equal");
    for (std::size_t li = 0; li < g1.layers.size(); ++li) {
      require(g1.layers[li].ctx == g2.layers[li].ctx,
              "intervened forward ctx bitwise equal");
      require(g1.layers[li].prob == g2.layers[li].prob,
              "intervened forward prob bitwise equal");
      require(g1.layers[li].r1 == g2.layers[li].r1,
              "intervened forward r1 bitwise equal");
      require(g1.layers[li].out == g2.layers[li].out,
              "intervened forward out bitwise equal");
    }
  }

  // 3. Head zero intervention changes the output (sanity).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto& row = ds.trainRows[0];
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const auto g1 = train::generalForward(config, oh, params);
    std::vector<aid::HeadIntervention> iv;
    iv.push_back({0, 0, aid::HeadAction::kZero, {}});
    const auto g2 = aid::generalForwardIntervened(config, oh, params, iv, {});
    require(g1.logits != g2.logits, "head zero changes logits");
  }

  // 4. Head swap with the same context == no change (identity).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto& row = ds.trainRows[0];
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const auto g1 = train::generalForward(config, oh, params);
    const std::uint32_t dh = config.dimension / config.numHeads;
    std::vector<float> replacement(
        static_cast<std::size_t>(config.tokens) * dh, 0.0f);
    for (uint32_t r = 0; r < config.tokens; ++r)
      for (uint32_t d = 0; d < dh; ++d)
        replacement[static_cast<std::size_t>(r) * dh + d] =
            g1.layers[0].ctx[static_cast<std::size_t>(r) * config.dimension + d];
    std::vector<aid::HeadIntervention> iv;
    iv.push_back({0, 0, aid::HeadAction::kSwap, replacement});
    const auto g2 = aid::generalForwardIntervened(config, oh, params, iv, {});
    require(g1.logits == g2.logits, "head swap with same ctx == no change");
  }

  // 5. Contribution sum == ATT_UPDATE.
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto& row = ds.trainRows[0];
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const auto g = train::generalForward(config, oh, params);
    require(aid::contributionSumMatches(config, params, g, 0,
                                        aid::kContributionSumTol),
            "contribution sum == ATT_UPDATE");
  }

  // 6. Attention weight stats: softmax row sums to 1, entropy in [0, log(T)].
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto& row = ds.trainRows[0];
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const auto g = train::generalForward(config, oh, params);
    const auto stats = aid::attentionRowStats(g, 0, 0, config.tokens,
                                              config.vocabularySize);
    double sum = 0.0;
    for (const double d : stats.distanceProfile) sum += d;
    require(std::abs(sum - 1.0) < 1e-6, "attention row sums to 1");
    require(stats.entropy >= 0.0 &&
                stats.entropy <= std::log(static_cast<double>(config.tokens)) +
                                     1e-9,
            "entropy in [0, log(T)]");
    require(stats.maxWeight > 0.0 && stats.maxWeight <= 1.0,
            "max weight in (0,1]");
  }

  // 7. Tap cache round-trip and identity rejection.
  {
    const auto run = dq::runFormalCpu(config, 5, 8, 0.003f,
                                     dq::StabilityMode::LEGACY, {8});
    const auto& params = run.checkpoints.at(8);
    const auto set = aid::extractTapFeatures(config, params, ds.trainRows);
    aid::TapCacheIdentity identity;
    identity.protocol = aid::kProtocolId;
    identity.config = "SELF_TEST";
    identity.seed = 5;
    identity.step = 8;
    identity.datasetHash = ds.combinedHash;
    identity.depth = 2;
    identity.rows = ds.trainRows.size();
    for (const auto& t : set.taps) identity.dims.push_back(t.dim);
    identity.contentHash = ibr::paramContentHash(params);
    const auto dir = std::filesystem::temp_directory_path() / "aid-cache-test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    require(aid::writeTapCache(dir, identity, set), "tap cache write");
    aid::TapSet loaded;
    require(aid::readTapCache(dir, identity, loaded), "tap cache read");
    require(loaded.taps.size() == set.taps.size() &&
                loaded.features == set.features,
            "tap cache bitwise round-trip");
    aid::TapCacheIdentity corrupt = identity;
    corrupt.contentHash = "fnv1a64:0000000000000000";
    require(!aid::readTapCache(dir, corrupt, loaded),
            "tap cache content-hash rejection");
    corrupt = identity;
    corrupt.dims.back() = 17;
    require(!aid::readTapCache(dir, corrupt, loaded),
            "tap cache dims rejection");
    std::filesystem::remove_all(dir, ec);
  }

  // 8. Probe machinery on tap features: deterministic + learns.
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto taps = aid::extractTapFeatures(config, params, ds.allRows);
    const int tapId = taps.tapIndex({TapKind::kCtxConcat, 0});
    const rp::ZStats stats = aid::tapZStats(taps, tapId, aid::kTrainRows);
    const rp::LayerSet view = taps.layerSetFor(tapId);
    const auto a = rp::trainProbe(view, 0, stats, ds.allRows, 0, aid::kCalBegin,
                                  ds.allRows, aid::kCalBegin, aid::kDevBegin);
    const auto b = rp::trainProbe(view, 0, stats, ds.allRows, 0, aid::kCalBegin,
                                  ds.allRows, aid::kCalBegin, aid::kDevBegin);
    require(a.finite && b.finite, "tap probe finite");
    require(a.selectedStep == b.selectedStep && a.probe.w == b.probe.w &&
                a.probe.b == b.probe.b,
            "tap probe deterministic");
  }

  // 9. NaN/Inf rejection in intervention scoring.
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    std::vector<aid::HeadIntervention> iv;
    std::vector<float> bad(
        static_cast<std::size_t>(config.tokens) * (config.dimension / 2),
        std::numeric_limits<float>::quiet_NaN());
    iv.push_back({0, 0, aid::HeadAction::kSwap, bad});
    bool threw = false;
    try {
      (void)aid::scoreIntervenedHead(config, params, ds.devRows, iv);
    } catch (const std::exception&) {
      threw = true;
    }
    require(threw, "NaN swap rejected");
  }

  // 10. Cache filename Windows safety.
  {
    const std::string name = aid::tapCacheFileName("L19_SEED_1", 1, 320,
                                                   "fnv1a64:abcdef1234567890");
    require(name.find(':') == std::string::npos &&
                name.find(';') == std::string::npos &&
                name.find('=') == std::string::npos,
            "cache filename Windows-safe");
  }

  std::cout << "SELF_TEST_PASS" << std::endl;
}

// ---------------------------------------------------------------------------
// Production run
// ---------------------------------------------------------------------------
struct ReportPaths {
  std::filesystem::path root;
  std::filesystem::path taps;
};

int runProduction(const ReportPaths& paths) {
  validateDatasets();
  const DataSet ds = buildDataSet();
  std::filesystem::create_directories(paths.root);
  std::filesystem::create_directories(paths.taps);

  std::vector<TapRun> runs;
  runs.reserve(kSpecs.size());
  const auto started = std::chrono::steady_clock::now();
  for (const auto& spec : kSpecs) {
    const auto specStart = std::chrono::steady_clock::now();
    std::cerr << "[" << spec.publicId << "] regenerating canonical trajectory"
              << " (selected " << spec.arSelectedStep << ", final "
              << spec.finalStep << ") ..." << std::endl;
    TapRun record = evaluateFinalCheckpoint(spec, ds, paths.taps);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - specStart);
    std::cerr << "[" << spec.publicId << "] anchors OK, "
              << record.registry.size() << " taps, max-drop block "
              << record.spec->maxDropBlock << ", head-zero "
              << record.headZero.size() << ", head-only "
              << record.headOnly.size() << ", done in " << elapsed.count()
              << " s" << std::endl;
    runs.push_back(std::move(record));
  }

  runCrossSeedSwaps(runs, ds);
  runAvSeparation(runs, ds);
  runHeadPairs(runs, ds);
  runFreeRunning(runs, ds);

  const Diagnosis diag = diagnose(runs);

  writeDatasetAnchors(paths.root / "dataset-anchors.csv", ds);
  writeTrajectoryAnchors(paths.root / "trajectory-anchors.csv", runs);
  writeHeadProbes(paths.root / "head-probe-by-seed.csv", runs);
  writeAttentionStats(paths.root / "attention-statistics.csv", runs);
  writeHeadZero(paths.root / "head-ablation.csv", runs);
  writeHeadOnly(paths.root / "head-only.csv", runs);
  writeContextSwaps(paths.root / "context-swap-raw.csv", runs);
  writeAvSwaps(paths.root / "attention-vs-value-swap.csv", runs);
  writeContributions(paths.root / "projection-contributions.csv", runs);
  writePairs(paths.root / "head-pair-interactions.csv", runs);
  writeFreeRunning(paths.root / "teacher-forced-free-running.csv", runs);
  writeContextVsProjection(paths.root / "context-vs-projection.csv", runs);
  writeDepthControl(paths.root / "depth-control.csv", runs);
  writeNextStepCandidates(paths.root / "next-step-candidates.csv", diag);
  writeDiagnosis(paths.root / "diagnosis.csv", diag);
  writeSummary(paths.root / "summary.csv", runs);
  writeBudget(paths.root / "budget.csv", runs);

  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started);
  std::cout << "=== ATTENTION_INTERNAL_V1 RUN COMPLETE ===" << std::endl;
  std::cout << "elapsed_seconds=" << elapsed.count() << std::endl;
  std::cout << "verdict=" << diag.verdict << std::endl;
  std::cout << "reasons=" << diag.reasons << std::endl;
  for (const auto& run : runs) {
    std::cout << run.spec->publicId << " head_tf="
              << run.baselineHead.tokenExact << "/"
              << run.baselineHead.total << " head_zero="
              << run.headZero.size() << " head_only=" << run.headOnly.size()
              << " swaps=" << run.contextSwaps.size()
              << " av=" << run.avSwaps.size() << " pairs=" << run.pairs.size()
              << " fr=" << run.freeRunning.size() << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool selfTestMode = false;
    std::filesystem::path root =
        "build/reports/qnn-attention-internal-diagnosis";
    std::filesystem::path taps =
        "build/reports/qnn-attention-internal-diagnosis/private-taps";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") selfTestMode = true;
      else if (arg == "--run") selfTestMode = false;
      else if (arg == "--report-root" && i + 1 < argc)
        root = argv[++i];
      else if (arg == "--tap-root" && i + 1 < argc)
        taps = argv[++i];
      else
        throw std::invalid_argument("UNKNOWN_ARG: " + arg);
    }
    if (selfTestMode) {
      selfTest();
      return 0;
    }
    return runProduction({root, taps});
  } catch (const std::exception& error) {
    std::cerr << "FATAL: " << error.what() << std::endl;
    return 1;
  }
}