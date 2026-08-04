// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// INTRA_BLOCK_READABILITY_V1 host-only runner: per-block ("intra-block")
// linear-readability diagnosis of the L19/L18 quantized transformer. The
// forward pass is the VERBATIM host copy (critical_margin_training_lib.h);
// taps observe tensors already kept by that forward (plus the two update
// tensors recomputed with the same mm arithmetic, bitwise identical to what
// is added to the residual stream). No device, QAIRT, ADB, QNN graph, or
// Android involvement; production code is not modified.
//
// Trajectories are regenerated host-only once per configuration (4 total,
// within the fixed regeneration budget) and every regenerated checkpoint is
// asserted against the same pinned canonical anchors as READOUT_PROBE_V1
// before use. AR_FINAL_HOLDOUT_V3 is only hash-verified, never opened.
//
// Budgets (fixed before results, see protocol.json):
//   independent tap probes   498  (L19 3x126 + L18 1x120) <= 500
//   cross-tap transfers      444  (coarse raw+norm all blocks, fine raw at
//                                  representative blocks)        <= 500
//   alignment fits           150  (coarse pairs, LS + Procrustes) <= 300
//   free-running rollouts     12  (max-drop block in/out + POST_LN_FINAL,
//                                  per configuration)            <= 16
//   trajectory regenerations   4  (one per configuration)        <= 4
//   token baselines            3  (A/B/C, token-only)
// Head-clone parity (head-input tap, zero-step clone probe) and the
// classification thresholds are fixed before results; the exporter only
// publishes allow-listed files.
#include "intra_block_readability_lib.h"

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

namespace ibr = phonelm::intra_block_readability;
namespace rp = phonelm::readout_probe;
namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;
namespace dq = phonelm::depth_quality;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;

namespace {

using ibr::TapKey;
using ibr::TapKind;

// Same pinned canonical anchors as READOUT_PROBE_V1 (identical trajectories).
struct ConfigSpec {
  const char* publicId = "";
  std::uint32_t seed = 0;
  std::uint32_t layers = 19;
  int arSelectedStep = 0;
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
    {"L19_SEED_1", 1, 19, 16, 320, 14, 0, 30, 2, 8.1239203249880703,
     16, 50, 8, 7.2793924123677254, 3.1434760002758511, 50, 7,
     7.1173910296005136},
    {"L19_SEED_2", 2, 19, 4, 320, 20, 0, 63, 6, 4.1834252619661516,
     20, 65, 9, 4.5271806692690921, 3.2025826990955695, 59, 8,
     6.0751862806397474},
    {"L19_SEED_4", 4, 19, 12, 320, 22, 0, 46, 6, 7.5872917441801651,
     21, 86, 12, 4.5741163228215562, 2.9437165421652174, 92, 11,
     4.030170295310362},
    {"L18_SEED_2_CONTROL", 2, 18, 4, 320, 18, 0, 65, 8,
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
  if (ds.trainRows.size() != ibr::kTrainRows || ds.calRows.size() != ibr::kCalRows ||
      ds.devRows.size() != ibr::kDevRows)
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

// Identical to READOUT_PROBE_V1: the regenerated trajectories must reproduce
// the canonical AR + margin anchors before any tap analysis is used.
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
  if (marginDevSelected.sequenceExact != 0)
    throw std::runtime_error(anchorError(spec, "MARGIN_DEV_SELECTED_SEQ",
                                         marginDevSelected.sequenceExact, 0));
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
struct TransferOutcome {
  TapKey src, dst;
  std::string pairKind;  // coarse | fine
  std::string variant;   // raw | norm
  std::uint64_t srcProbeDevTf = 0;
  std::uint64_t transferDevTf = 0;
  std::uint64_t dstProbeDevTf = 0;
  std::int64_t deltaVsSrc = 0;
  std::int64_t deltaVsDst = 0;
};

enum class PairVerdict { kCoordinateTransform, kInformationLoss, kMixed };

inline const char* pairVerdictName(PairVerdict v) {
  switch (v) {
    case PairVerdict::kCoordinateTransform: return "COORDINATE_TRANSFORM";
    case PairVerdict::kInformationLoss: return "INFORMATION_LOSS";
    case PairVerdict::kMixed: return "MIXED";
  }
  return "UNKNOWN";
}

struct AlignOutcome {
  TapKey src, dst;
  bool finite = true;
  int fitRank = 0;
  double cond = 0.0;
  double relResidualLs = 0.0;
  double relResidualOrth = 0.0;
  std::uint64_t nativeDevTf = 0;    // probe trained on dst (i side)
  std::uint64_t transferDevTf = 0;  // probe(src) with dst stats on dst
  std::uint64_t alignedLsDevTf = 0;
  std::uint64_t alignedOrthDevTf = 0;
  double recoveryLs = 0.0;
  double recoveryOrth = 0.0;
  std::int64_t residualLossLs = 0;   // transferDevTf - alignedLsDevTf
  std::int64_t residualLossOrth = 0; // transferDevTf - alignedOrthDevTf
  PairVerdict verdict = PairVerdict::kMixed;
};

struct TapRun {
  const ConfigSpec* spec = nullptr;
  dq::FormRun run;
  ibr::TapSet taps;
  std::string paramHash;
  std::vector<ibr::TapSpec> registry;
  // per tap id:
  std::map<int, rp::ProbeTrainResult> probes;
  std::map<int, rp::ZStats> stats;
  std::map<int, rp::TokenMetrics> trainTf, calTf, devTf;
  std::map<int, ibr::TapAux> aux;
  std::vector<ibr::BlockGeometry> geometry;
  std::map<std::tuple<int, int, std::string>, TransferOutcome> transfers;  // key: (src, dst, variant)
  std::map<std::pair<int, int>, AlignOutcome> alignments;
  ibr::CloneParity cloneParity;
  int maxDropBlock = -1;
  std::int64_t maxDropValue = 0;
  std::vector<int> frTapIds;  // chosen FR taps (3 per config)
  std::map<int, cm::CheckpointMetrics> fr;
};

std::string tapName(const ibr::TapSet& set, int tapId) {
  return set.tap(tapId).name;
}

// ---------------------------------------------------------------------------
// Evaluate the FINAL checkpoint of one configuration.
// ---------------------------------------------------------------------------
TapRun evaluateFinalCheckpoint(const ConfigSpec& spec, const DataSet& ds,
                               const std::filesystem::path& tapCacheDir) {
  const auto config = modelConfig(spec.layers);
  TapRun out;
  out.spec = &spec;
  out.registry = ibr::buildTapRegistry(static_cast<int>(spec.layers));

  out.run = dq::runFormalCpu(config, spec.seed, spec.finalStep, 0.003f,
                             dq::StabilityMode::LEGACY,
                             {spec.arSelectedStep, spec.finalStep});
  runAnchors(spec, out.run, ds);
  const auto& params = out.run.checkpoints.at(spec.finalStep);
  out.paramHash = ibr::paramContentHash(params);

  ibr::TapCacheIdentity identity;
  identity.protocol = ibr::kProtocolId;
  identity.config = spec.publicId;
  identity.seed = spec.seed;
  identity.step = spec.finalStep;
  identity.datasetHash = ds.combinedHash;
  identity.depth = static_cast<int>(spec.layers);
  identity.rows = ds.allRows.size();
  for (const auto& t : out.registry) identity.dims.push_back(t.dim);
  identity.contentHash = out.paramHash;
  if (!ibr::readTapCache(tapCacheDir, identity, out.taps)) {
    out.taps = ibr::extractTapFeatures(config, params, ds.allRows);
    if (!ibr::writeTapCache(tapCacheDir, identity, out.taps))
      throw std::runtime_error("TAP_CACHE_WRITE_FAILED");
  } else {
    const std::string hash = ibr::paramContentHash(params);
    if (hash != out.paramHash)
      throw std::runtime_error("TAP_CACHE_PARAM_HASH_MISMATCH");
  }

  // Independent probes on every tap (TRAIN-only z-stats, CAL selection).
  for (const auto& t : out.registry) {
    const rp::ZStats stats = ibr::tapZStats(out.taps, t.id, ibr::kTrainRows);
    out.stats[t.id] = stats;
    const rp::LayerSet view = out.taps.layerSetFor(t.id);
    auto probe = rp::trainProbe(view, 0, stats, ds.allRows, 0, ibr::kCalBegin,
                                ds.allRows, ibr::kCalBegin, ibr::kDevBegin);
    out.probes[t.id] = probe;
    if (!probe.finite) continue;
    out.trainTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.trainRows, 0);
    out.calTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.calRows,
                              ibr::kCalBegin);
    out.devTf[t.id] =
        rp::probeTokenMetrics(probe.probe, stats, view, 0, ds.devRows,
                              ibr::kDevBegin);
    out.aux[t.id] =
        ibr::tapAuxMetrics(out.taps, t.id, ds.devRows, ibr::kDevBegin,
                           ibr::kDevBegin + ibr::kDevRows, 0, ibr::kTrainRows);
  }

  // Head-clone parity on the head-input tap (final block AFTER_FFN).
  {
    const int headTap =
        out.taps.tapIndex({TapKind::kAfterFfn, static_cast<int>(spec.layers) - 1});
    out.cloneParity = ibr::headCloneParity(config, params, out.taps, headTap,
                                           ds.devRows, ibr::kDevBegin);
  }

  // Residual geometry (dev rows).
  out.geometry =
      ibr::blockGeometry(out.taps, ibr::kDevBegin, ibr::kDevBegin + ibr::kDevRows);

  // Transfers: coarse (BLOCK_INPUT->AFTER_ATTN, AFTER_ATTN->AFTER_FFN) for
  // every block, raw + norm; fine (6 intra-block pairs) at representative
  // blocks {1,4,8,12,16} plus the last block, raw only.
  auto probeIdOf = [&](const TapKey& key) {
    return out.taps.tapIndex(key);
  };
  const int depth = static_cast<int>(spec.layers);
  for (int li = 0; li < depth; ++li) {
    const TapKey in = ibr::blockInputKey(li);
    const TapKey attn = {TapKind::kAfterAttn, li};
    const TapKey ffn = {TapKind::kAfterFfn, li};
    for (const auto& pair : {std::make_pair(in, attn), std::make_pair(attn, ffn)}) {
      const int src = probeIdOf(pair.first);
      const int dst = probeIdOf(pair.second);
      if (!out.probes.at(src).finite || !out.probes.at(dst).finite) continue;
      for (const auto& variant : {std::string("raw"), std::string("norm")}) {
        TransferOutcome outcome;
        outcome.src = pair.first;
        outcome.dst = pair.second;
        outcome.pairKind = "coarse";
        outcome.variant = variant;
        outcome.srcProbeDevTf = out.devTf.at(src).tokenExact;
        outcome.dstProbeDevTf = out.devTf.at(dst).tokenExact;
        const rp::TokenMetrics transfer =
            variant == "raw"
                ? ibr::transferEval(out.probes.at(src).probe, out.stats.at(src),
                                    out.taps, dst, ds.devRows, ibr::kDevBegin)
                : ibr::transferEval(out.probes.at(src).probe, out.stats.at(dst),
                                    out.taps, dst, ds.devRows, ibr::kDevBegin);
        outcome.transferDevTf = transfer.tokenExact;
        outcome.deltaVsSrc = static_cast<std::int64_t>(transfer.tokenExact) -
                             static_cast<std::int64_t>(outcome.srcProbeDevTf);
        outcome.deltaVsDst = static_cast<std::int64_t>(transfer.tokenExact) -
                             static_cast<std::int64_t>(outcome.dstProbeDevTf);
        out.transfers[{src, dst, variant}] = outcome;
      }
    }
  }
  // Fine pairs at representative + last blocks, raw only.
  std::set<int> fineBlocks;
  for (const int b : ibr::kRepBlocks) fineBlocks.insert(b);
  fineBlocks.insert(depth - 1);
  for (const int li : fineBlocks) {
    const TapKey in = ibr::blockInputKey(li);
    const TapKey norm1 = {TapKind::kNorm1, li};
    const TapKey attnUpd = {TapKind::kAttnUpdate, li};
    const TapKey attn = {TapKind::kAfterAttn, li};
    const TapKey norm2 = {TapKind::kNorm2, li};
    const TapKey ffnUpd = {TapKind::kFfnUpdate, li};
    const TapKey ffn = {TapKind::kAfterFfn, li};
    const std::vector<std::pair<TapKey, TapKey>> finePairs = {
        {in, norm1},       {norm1, attnUpd},  {attnUpd, attn},
        {attn, norm2},     {norm2, ffnUpd},   {ffnUpd, ffn}};
    for (const auto& pair : finePairs) {
      const int src = probeIdOf(pair.first);
      const int dst = probeIdOf(pair.second);
      if (!out.probes.at(src).finite || !out.probes.at(dst).finite) continue;
      TransferOutcome outcome;
      outcome.src = pair.first;
      outcome.dst = pair.second;
      outcome.pairKind = "fine";
      outcome.variant = "raw";
      outcome.srcProbeDevTf = out.devTf.at(src).tokenExact;
      outcome.dstProbeDevTf = out.devTf.at(dst).tokenExact;
      const rp::TokenMetrics transfer =
          ibr::transferEval(out.probes.at(src).probe, out.stats.at(src),
                            out.taps, dst, ds.devRows, ibr::kDevBegin);
      outcome.transferDevTf = transfer.tokenExact;
      outcome.deltaVsSrc = static_cast<std::int64_t>(transfer.tokenExact) -
                           static_cast<std::int64_t>(outcome.srcProbeDevTf);
      outcome.deltaVsDst = static_cast<std::int64_t>(transfer.tokenExact) -
                           static_cast<std::int64_t>(outcome.dstProbeDevTf);
      out.transfers[{src, dst, std::string("raw")}] = outcome;
    }
  }

  // Alignment on coarse pairs (LS + Procrustes in one fit; function-level
  // probe alignment evaluated on dev).
  for (int li = 0; li < depth; ++li) {
    const TapKey in = ibr::blockInputKey(li);
    const TapKey attn = {TapKind::kAfterAttn, li};
    const TapKey ffn = {TapKind::kAfterFfn, li};
    for (const auto& pair : {std::make_pair(in, attn), std::make_pair(attn, ffn)}) {
      const int src = probeIdOf(pair.first);
      const int dst = probeIdOf(pair.second);
      if (!out.probes.at(src).finite || !out.probes.at(dst).finite) continue;
      AlignOutcome outcome;
      outcome.src = pair.first;
      outcome.dst = pair.second;
      const ibr::AlignResult al =
          ibr::fitTapAlignment(out.taps, src, dst, 0, ibr::kTrainRows);
      outcome.finite = al.finite;
      outcome.fitRank = al.fitRank;
      outcome.cond = al.cond;
      outcome.relResidualLs = al.relResidualLs;
      outcome.relResidualOrth = al.relResidualOrth;
      outcome.nativeDevTf = out.devTf.at(dst).tokenExact;
      // Transfer with dst stats = the "initial" performance i.
      outcome.transferDevTf =
          ibr::transferEval(out.probes.at(src).probe, out.stats.at(dst),
                            out.taps, dst, ds.devRows, ibr::kDevBegin)
              .tokenExact;
      const rp::Probe lsAligned =
          ibr::alignProbeFunctionLevel(out.probes.at(src).probe, out.taps, src,
                                       dst, 0, ibr::kTrainRows);
      const rp::Probe orthAligned = ibr::alignProbeProcrustes(
          out.probes.at(src).probe, al);
      outcome.alignedLsDevTf =
          ibr::transferEval(lsAligned, out.stats.at(dst), out.taps, dst,
                            ds.devRows, ibr::kDevBegin)
              .tokenExact;
      outcome.alignedOrthDevTf =
          ibr::transferEval(orthAligned, out.stats.at(dst), out.taps, dst,
                            ds.devRows, ibr::kDevBegin)
              .tokenExact;
      const auto recovery = [&](std::uint64_t aligned) {
        const std::int64_t i = static_cast<std::int64_t>(outcome.transferDevTf);
        const std::int64_t n = static_cast<std::int64_t>(outcome.nativeDevTf);
        const std::int64_t a = static_cast<std::int64_t>(aligned);
        if (i <= n) return 0.0;
        const double r = static_cast<double>(a - n) / static_cast<double>(i - n);
        return std::min(1.0, std::max(0.0, r));
      };
      outcome.recoveryLs = recovery(outcome.alignedLsDevTf);
      outcome.recoveryOrth = recovery(outcome.alignedOrthDevTf);
      outcome.residualLossLs = static_cast<std::int64_t>(outcome.transferDevTf) -
                               static_cast<std::int64_t>(outcome.alignedLsDevTf);
      outcome.residualLossOrth =
          static_cast<std::int64_t>(outcome.transferDevTf) -
          static_cast<std::int64_t>(outcome.alignedOrthDevTf);
      // Fixed verdict rules (LS alignment; R and residual loss as defined):
      if (outcome.recoveryLs >= 0.75 && outcome.residualLossLs <= 1)
        outcome.verdict = PairVerdict::kCoordinateTransform;
      else if (outcome.recoveryLs <= 0.25 && outcome.residualLossLs >= 3)
        outcome.verdict = PairVerdict::kInformationLoss;
      else
        outcome.verdict = PairVerdict::kMixed;
      out.alignments[{src, dst}] = outcome;
    }
  }

  // Free-running taps: selected after the max-drop block is known (fixed
  // rule). maxDrop = argmax over blocks of tf(block_input) - tf(after_ffn),
  // earliest block on ties.
  out.maxDropValue = -1;
  out.maxDropBlock = 0;
  for (int li = 0; li < depth; ++li) {
    const int inId = probeIdOf(ibr::blockInputKey(li));
    const int ffnId = probeIdOf({TapKind::kAfterFfn, li});
    if (!out.probes.at(inId).finite || !out.probes.at(ffnId).finite) continue;
    const std::int64_t drop =
        static_cast<std::int64_t>(out.devTf.at(inId).tokenExact) -
        static_cast<std::int64_t>(out.devTf.at(ffnId).tokenExact);
    if (drop > out.maxDropValue) {
      out.maxDropValue = drop;
      out.maxDropBlock = li;
    }
  }
  std::set<int> frTaps;
  frTaps.insert(probeIdOf(ibr::blockInputKey(out.maxDropBlock)));
  frTaps.insert(probeIdOf({TapKind::kAfterFfn, out.maxDropBlock}));
  frTaps.insert(probeIdOf({TapKind::kNorm2, depth - 1}));  // POST_LN_FINAL
  out.frTapIds.assign(frTaps.begin(), frTaps.end());
  for (const int tapId : out.frTapIds) {
    if (!out.probes.at(tapId).finite) continue;
    ibr::TapScorer scorer;
    scorer.config = &config;
    scorer.params = &params;
    scorer.probe = out.probes.at(tapId).probe;
    scorer.stats = out.stats.at(tapId);
    scorer.key = out.taps.tap(tapId).key;
    out.fr[tapId] =
        ibr::tapFreeRunning(ds.devCases, spec.finalStep, scorer);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Diagnosis (thresholds fixed before results; never tuned)
// ---------------------------------------------------------------------------
struct Diagnosis {
  std::string verdict;
  std::string reasons;
  // per-L19-seed deep-band counts (blocks 11..18):
  std::vector<int> ctCounts, ilCounts, attnDrops, ffnDrops, norm1Drops,
      norm2Drops, attnOverwrites, ffnOverwrites;
  // L18 control:
  int controlIlCount = 0;
  std::string controlNote;
  bool controlGuardOk = true;
};

constexpr int kDeepFirstBlock = 11;  // 0-based, blocks 11..18 on L19

Diagnosis diagnose(const std::vector<TapRun>& runs, const DataSet& ds) {
  (void)ds;
  Diagnosis diag;
  // Order: L19_SEED_1, L19_SEED_2, L19_SEED_4, L18 control.
  std::vector<const TapRun*> l19;
  const TapRun* control = nullptr;
  for (const auto& run : runs) {
    if (run.spec->layers == 19) l19.push_back(&run);
    else control = &run;
  }

  auto probeDevTf = [](const TapRun& run, const TapKey& key) -> std::int64_t {
    const int id = run.taps.tapIndex(key);
    if (!run.probes.at(id).finite) return -1;
    return static_cast<std::int64_t>(run.devTf.at(id).tokenExact);
  };

  for (const auto* run : l19) {
    int ct = 0, il = 0, attnDrop = 0, ffnDrop = 0, norm1Drop = 0, norm2Drop = 0,
        attnOv = 0, ffnOv = 0;
    const int depth = static_cast<int>(run->spec->layers);
    for (int li = kDeepFirstBlock; li <= depth - 1; ++li) {
      const std::int64_t tfIn = probeDevTf(*run, ibr::blockInputKey(li));
      const std::int64_t tfNorm1 = probeDevTf(*run, {TapKind::kNorm1, li});
      const std::int64_t tfAttn = probeDevTf(*run, {TapKind::kAfterAttn, li});
      const std::int64_t tfNorm2 = probeDevTf(*run, {TapKind::kNorm2, li});
      const std::int64_t tfFfn = probeDevTf(*run, {TapKind::kAfterFfn, li});
      if (tfIn < 0 || tfNorm1 < 0 || tfAttn < 0 || tfNorm2 < 0 || tfFfn < 0)
        continue;
      if (tfNorm1 - tfAttn >= ibr::kDropTokens) ++attnDrop;
      if (tfNorm2 - tfFfn >= ibr::kDropTokens) ++ffnDrop;
      if (tfIn - tfNorm1 >= ibr::kDropTokens) ++norm1Drop;
      if (tfAttn - tfNorm2 >= ibr::kDropTokens) ++norm2Drop;
      const auto& g = run->geometry.at(static_cast<std::size_t>(li));
      if (g.attnRatio > ibr::kResidualRatioThreshold &&
          g.cosAttn < ibr::kCosineThreshold)
        ++attnOv;
      if (g.ffnRatio > ibr::kResidualRatioThreshold &&
          g.cosFfn < ibr::kCosineThreshold)
        ++ffnOv;
      // The block's two coarse pairs (block_input->after_attn and
      // after_attn->after_ffn) both live in the deep-band pool.
      const int inId = run->taps.tapIndex(ibr::blockInputKey(li));
      const int attnId = run->taps.tapIndex({TapKind::kAfterAttn, li});
      const int ffnId = run->taps.tapIndex({TapKind::kAfterFfn, li});
      auto verdictOf = [&](int s, int d) {
        const auto it = run->alignments.find({s, d});
        return it == run->alignments.end() ? PairVerdict::kMixed
                                           : it->second.verdict;
      };
      for (const auto v : {verdictOf(inId, attnId), verdictOf(attnId, ffnId)}) {
        if (v == PairVerdict::kCoordinateTransform) ++ct;
        else if (v == PairVerdict::kInformationLoss) ++il;
      }
    }
    diag.ctCounts.push_back(ct);
    diag.ilCounts.push_back(il);
    diag.attnDrops.push_back(attnDrop);
    diag.ffnDrops.push_back(ffnDrop);
    diag.norm1Drops.push_back(norm1Drop);
    diag.norm2Drops.push_back(norm2Drop);
    diag.attnOverwrites.push_back(attnOv);
    diag.ffnOverwrites.push_back(ffnOv);
  }

  // L18 control: deep band = blocks 10..16 (7 blocks, 14 pairs).
  if (control) {
    int il = 0;
    const int depth = static_cast<int>(control->spec->layers);
    for (int li = 10; li <= depth - 1; ++li) {
      const int inId = control->taps.tapIndex(ibr::blockInputKey(li));
      const int attnId = control->taps.tapIndex({TapKind::kAfterAttn, li});
      const int ffnId = control->taps.tapIndex({TapKind::kAfterFfn, li});
      auto verdictOf = [&](int s, int d) {
        const auto it = control->alignments.find({s, d});
        return it == control->alignments.end() ? PairVerdict::kMixed
                                               : it->second.verdict;
      };
      for (const auto v : {verdictOf(inId, attnId), verdictOf(attnId, ffnId)})
        if (v == PairVerdict::kInformationLoss) ++il;
    }
    diag.controlIlCount = il;
    diag.controlGuardOk = il < 9;
    diag.controlNote = diag.controlGuardOk
                           ? "L18 deep-band IL below majority (guard OK)"
                           : "L18 deep-band IL at/above majority: degradation "
                             "is NOT depth-specific";
  }

  // Majority helpers: deep band is 8 blocks x 2 pairs = 16 for L19.
  auto majority = [](const std::vector<int>& v, int threshold) {
    int n = 0;
    for (const int x : v)
      if (x >= threshold) ++n;
    return n;
  };

  std::ostringstream reasons;
  reasons << "L19 deep-band CT/16 per seed: " << diag.ctCounts[0] << "/"
          << diag.ctCounts[1] << "/" << diag.ctCounts[2] << "; IL/16: "
          << diag.ilCounts[0] << "/" << diag.ilCounts[1] << "/"
          << diag.ilCounts[2] << "; attn_drop_blocks/8: "
          << diag.attnDrops[0] << "/" << diag.attnDrops[1] << "/"
          << diag.attnDrops[2] << "; ffn_drop_blocks/8: " << diag.ffnDrops[0]
          << "/" << diag.ffnDrops[1] << "/" << diag.ffnDrops[2]
          << "; norm1_drop_blocks/8: " << diag.norm1Drops[0] << "/"
          << diag.norm1Drops[1] << "/" << diag.norm1Drops[2]
          << "; norm2_drop_blocks/8: " << diag.norm2Drops[0] << "/"
          << diag.norm2Drops[1] << "/" << diag.norm2Drops[2]
          << "; attn_overwrite_blocks/8: " << diag.attnOverwrites[0] << "/"
          << diag.attnOverwrites[1] << "/" << diag.attnOverwrites[2]
          << "; ffn_overwrite_blocks/8: " << diag.ffnOverwrites[0] << "/"
          << diag.ffnOverwrites[1] << "/" << diag.ffnOverwrites[2]
          << "; control_il=" << diag.controlIlCount << " ("
          << diag.controlNote << ")";

  // Priority (fixed): overwrite -> attention -> ffn -> ln -> coordinate ->
  // information -> cumulative -> undetermined.
  const int ctSeeds = majority(diag.ctCounts, ibr::kDeepBandMajority);
  const int ilSeeds = majority(diag.ilCounts, ibr::kDeepBandMajority);
  const int attnSeeds = majority(diag.attnDrops, 6);
  const int ffnSeeds = majority(diag.ffnDrops, 6);
  const int n1Seeds = majority(diag.norm1Drops, 6);
  const int n2Seeds = majority(diag.norm2Drops, 6);
  const int attnOvSeeds = majority(diag.attnOverwrites, 6);
  const int ffnOvSeeds = majority(diag.ffnOverwrites, 6);

  if ((attnSeeds >= 2 && attnOvSeeds >= 2) ||
      (ffnSeeds >= 2 && ffnOvSeeds >= 2))
    diag.verdict = "RESIDUAL_OVERWRITE";
  else if (attnSeeds >= 2)
    diag.verdict = "ATTENTION";
  else if (ffnSeeds >= 2)
    diag.verdict = "FFN";
  else if (n1Seeds >= 2 || n2Seeds >= 2)
    diag.verdict = "LAYERNORM";
  else if (ctSeeds >= 2)
    diag.verdict = "COORDINATE_DRIFT";
  else if (ilSeeds >= 2)
    diag.verdict = "LINEAR_INFO_LOSS";
  else if (ctSeeds == 1 || ilSeeds == 1)
    diag.verdict = "MIXED";
  else
    diag.verdict = "CUMULATIVE";
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
  csv.row({"TRAIN", "probe_and_alignment_learning",
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

void writeTapProbes(const std::filesystem::path& path,
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
               ibr::tapKindName(t.key.kind),
               t.key.block >= 0 ? rp::text(t.key.block) : "",
               rp::text(t.dim), rp::text(probe.selectedStep),
               probe.finite ? "true" : "false", train, cal, dev, rank, nll,
               margin, q10, top2, top3});
    }
}

void writeTransfers(const std::filesystem::path& path,
                    const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "src_tap", "dst_tap", "pair_kind",
              "variant", "src_probe_dev_tf_exact", "transfer_dev_tf_exact",
              "delta_vs_src", "dst_probe_dev_tf_exact", "delta_vs_dst"});
  for (const auto& run : runs)
    for (const auto& entry : run.transfers) {
      const auto& x = entry.second;
      csv.row({run.spec->publicId, tapName(run.taps, std::get<0>(entry.first)),
               tapName(run.taps, std::get<1>(entry.first)), x.pairKind,
               x.variant,
               rp::text(x.srcProbeDevTf), rp::text(x.transferDevTf),
               rp::text(x.deltaVsSrc), rp::text(x.dstProbeDevTf),
               rp::text(x.deltaVsDst)});
    }
}

// Format a condition number explicitly: rank-deficient Gram yields +inf
// (documented in protocol.json), true NaN stays distinct from inf.
inline std::string condText(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return "inf";
  return rp::text(value, 6);
}

void writeAlignments(const std::filesystem::path& path,
                     const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "src_tap", "dst_tap", "finite", "fit_rank",
              "cond", "rel_residual_ls", "rel_residual_orth",
              "native_dev_tf_exact", "transfer_dev_tf_exact",
              "aligned_ls_dev_tf_exact", "aligned_orth_dev_tf_exact",
              "recovery_ls", "recovery_orth", "residual_loss_ls",
              "residual_loss_orth", "verdict"});
  for (const auto& run : runs)
    for (const auto& entry : run.alignments) {
      const auto& x = entry.second;
      csv.row({run.spec->publicId, tapName(run.taps, entry.first.first),
               tapName(run.taps, entry.first.second),
               x.finite ? "true" : "false", rp::text(x.fitRank),
               condText(x.cond), rp::text(x.relResidualLs, 6),
               rp::text(x.relResidualOrth, 6), rp::text(x.nativeDevTf),
               rp::text(x.transferDevTf), rp::text(x.alignedLsDevTf),
               rp::text(x.alignedOrthDevTf), rp::text(x.recoveryLs, 6),
               rp::text(x.recoveryOrth, 6), rp::text(x.residualLossLs),
               rp::text(x.residualLossOrth), pairVerdictName(x.verdict)});
    }
}

void writeGeometry(const std::filesystem::path& path,
                   const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "block", "dataset", "residual_norm",
              "attn_update_norm", "attn_ratio", "cos_attn", "after_attn_norm",
              "ffn_update_norm", "ffn_ratio", "cos_ffn", "after_ffn_norm",
              "attn_overwrite", "ffn_overwrite"});
  for (const auto& run : runs)
    for (const auto& g : run.geometry) {
      csv.row({run.spec->publicId, rp::text(g.block), "MARGIN_DEVELOPMENT_V1",
               rp::text(g.residualNorm, 6), rp::text(g.attnUpdateNorm, 6),
               rp::text(g.attnRatio, 6), rp::text(g.cosAttn, 6),
               rp::text(g.afterAttnNorm, 6), rp::text(g.ffnUpdateNorm, 6),
               rp::text(g.ffnRatio, 6), rp::text(g.cosFfn, 6),
               rp::text(g.afterFfnNorm, 6),
               g.attnRatio > ibr::kResidualRatioThreshold &&
                       g.cosAttn < ibr::kCosineThreshold
                   ? "true"
                   : "false",
               g.ffnRatio > ibr::kResidualRatioThreshold &&
                       g.cosFfn < ibr::kCosineThreshold
                   ? "true"
                   : "false"});
    }
}

void writeAux(const std::filesystem::path& path,
              const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "tap_id", "tap_name", "dim", "eta2_dev",
              "effective_rank_train", "between_within_dev",
              "mean_pairwise_cosine_dev", "cond_train"});
  for (const auto& run : runs)
    for (const auto& entry : run.aux) {
      const auto& a = entry.second;
      csv.row({run.spec->publicId, rp::text(a.tapId),
               tapName(run.taps, entry.first), rp::text(run.taps.tap(entry.first).dim),
               rp::text(a.eta2, 6), rp::text(a.effectiveRank, 6),
               rp::text(a.betweenWithin, 6), rp::text(a.meanPairwiseCosine, 6),
               condText(a.cond)});
    }
}

void writeCloneParity(const std::filesystem::path& path,
                      const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "tap_id", "tap_name", "max_logit_delta",
              "max_nll_delta", "max_margin_delta", "argmax_flips",
              "rank_flips", "exact_mismatch", "pass"});
  for (const auto& run : runs) {
    const int headTap =
        run.taps.tapIndex({TapKind::kAfterFfn,
                           static_cast<int>(run.spec->layers) - 1});
    const auto& p = run.cloneParity;
    csv.row({run.spec->publicId, rp::text(headTap),
             tapName(run.taps, headTap), rp::text(p.maxLogitDelta, 12),
             rp::text(p.maxNllDelta, 12), rp::text(p.maxMarginDelta, 12),
             rp::text(p.argmaxFlips), rp::text(p.rankFlips),
             rp::text(p.exactMismatch), p.pass ? "true" : "false"});
  }
}

void writeBaselines(const std::filesystem::path& path, const DataSet& ds) {
  const auto results = ibr::tokenBaselines(ds.trainRows, ds.calRows, ds.devRows);
  rp::CsvWriter csv(path);
  csv.header({"baseline", "dataset", "total", "seen", "unseen", "correct",
              "correct_seen", "correct_unseen", "fallback_used",
              "unique_keys"});
  for (const auto& r : results)
    csv.row({r.baseline, r.dataset, rp::text(r.total), rp::text(r.seen),
             rp::text(r.unseen), rp::text(r.correct), rp::text(r.correctSeen),
             rp::text(r.correctUnseen), rp::text(r.fallbackUsed),
             rp::text(r.uniqueKeys)});
}

void writeFreeRunning(const std::filesystem::path& path,
                      const std::vector<TapRun>& runs) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "tap_id", "tap_name", "step",
              "token_exact", "token_total", "sequence_exact",
              "sequence_total", "nll", "median_first_error_survival",
              "lower_tail_margin_q10", "all_finite"});
  for (const auto& run : runs)
    for (const auto& entry : run.fr) {
      const auto& m = entry.second;
      csv.row({run.spec->publicId, rp::text(entry.first),
               tapName(run.taps, entry.first), rp::text(run.spec->finalStep),
               rp::text(m.tokenExact), rp::text(m.tokenTotal),
               rp::text(m.sequenceExact), rp::text(m.sequenceTotal),
               rp::text(m.autoregressiveNll, 10),
               rp::text(m.medianFirstErrorSurvival, 6),
               rp::text(m.lowerTailMarginQ10, 6),
               m.allFinite ? "true" : "false"});
    }
}

void writeDiagnosis(const std::filesystem::path& path,
                    const Diagnosis& diag) {
  rp::CsvWriter csv(path);
  csv.header({"verdict", "reasons", "thresholds_fixed_before_results"});
  csv.row({diag.verdict, diag.reasons, "true"});
}

void writeSummary(const std::filesystem::path& path,
                  const std::vector<TapRun>& runs, const DataSet& ds) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "scope", "metric", "value"});
  for (const auto& run : runs) {
    const auto config = modelConfig(run.spec->layers);
    const auto& finalParams = run.run.checkpoints.at(run.spec->finalStep);
    const auto headDevTf = rp::headTokenMetrics(config, finalParams, ds.devRows);
    const auto headDevFr = rp::headFreeRunning(config, finalParams,
                                               run.spec->finalStep, ds.devCases);
    const int embId = run.taps.tapIndex({TapKind::kEmbedding, -1});
    const int headTap =
        run.taps.tapIndex({TapKind::kAfterFfn,
                           static_cast<int>(run.spec->layers) - 1});
    csv.row({run.spec->publicId, "head", "dev_tf_token_exact",
             rp::text(headDevTf.tokenExact)});
    csv.row({run.spec->publicId, "head", "dev_fr_token_exact",
             rp::text(headDevFr.tokenExact)});
    csv.row({run.spec->publicId, "head", "dev_fr_sequence_exact",
             rp::text(headDevFr.sequenceExact)});
    csv.row({run.spec->publicId, "head", "dev_fr_nll",
             rp::text(headDevFr.autoregressiveNll, 10)});
    csv.row({run.spec->publicId, "head_input_tap", "tap_name",
             tapName(run.taps, headTap)});
    csv.row({run.spec->publicId, "embedding_tap", "dev_tf_token_exact",
             run.probes.at(embId).finite
                 ? rp::text(run.devTf.at(embId).tokenExact)
                 : "nonfinite"});
    csv.row({run.spec->publicId, "max_drop_block", "block",
             rp::text(run.maxDropBlock)});
    csv.row({run.spec->publicId, "max_drop_block", "drop_tokens",
             rp::text(run.maxDropValue)});
    csv.row({run.spec->publicId, "clone_parity", "pass",
             run.cloneParity.pass ? "true" : "false"});
    csv.row({run.spec->publicId, "clone_parity", "max_logit_delta",
             rp::text(run.cloneParity.maxLogitDelta, 12)});
    csv.row({run.spec->publicId, "clone_parity", "argmax_flips",
             rp::text(run.cloneParity.argmaxFlips)});
    csv.row({run.spec->publicId, "clone_parity", "exact_mismatch",
             rp::text(run.cloneParity.exactMismatch)});
  }
}

void writeBudget(const std::filesystem::path& path, const DataSet& ds) {
  (void)ds;
  rp::CsvWriter csv(path);
  csv.header({"item", "count", "limit", "ok"});
  auto row = [&](const std::string& item, std::uint64_t count,
                 std::uint64_t limit) {
    csv.row({item, rp::text(count), rp::text(limit),
             count <= limit ? "true" : "false"});
  };
  // Counts are fixed by the protocol; the runner enforces them structurally
  // (the registry and pair lists are generated from the same constants).
  row("tap_probe_trainings", 498, 500);
  row("cross_tap_transfer_evals", 444, 500);
  row("alignment_fits", 150, 300);
  row("free_running_rollouts", 12, 16);
  row("trajectory_regenerations", 4, 4);
  row("token_baselines", 3, 3);
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
    const auto r19 = ibr::buildTapRegistry(19);
    const auto r18 = ibr::buildTapRegistry(18);
    require(r19.size() == 126, "L19 registry count");
    require(r18.size() == 120, "L18 registry count");
    std::set<int> ids;
    for (const auto& t : r19) ids.insert(t.id);
    require(ids.size() == r19.size(), "tap ids unique");
    int relu = 0, ctx = 0;
    for (const auto& t : r19) {
      if (t.key.kind == TapKind::kRelu) ++relu;
      if (t.key.kind == TapKind::kCtx) ++ctx;
    }
    require(relu == 5, "L19 RELU count");
    require(ctx == 6, "L19 CTX count (rep blocks + last)");
    require(r19.back().key.kind == TapKind::kAfterFfn, "L19 last tap");
    // Block-input aliasing: block li input == AFTER_FFN(li-1) (or embedding).
    require(ibr::blockInputKey(0) == TapKey{TapKind::kEmbedding, -1},
            "block 0 input alias");
    require(ibr::blockInputKey(5) == TapKey{TapKind::kAfterFfn, 4},
            "block 5 input alias");
  }

  // 2. Tap extraction matches the readout LayerSet bitwise on shared tensors
  // and the residual identities hold bitwise (r1 = x + attn_update in float,
  // out = r1 + ffn_update in float).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    std::vector<rp::ProbeRow> rows = ds.trainRows;
    rows.insert(rows.end(), ds.calRows.begin(), ds.calRows.end());
    const auto taps = ibr::extractTapFeatures(config, params, rows);
    const auto readout = rp::extractFeatures(config, params, rows);
    require(taps.taps.size() == 15, "depth-2 tap count (1 + 6 + 8)");
    const int dim = static_cast<int>(config.dimension);
    const std::size_t n = rows.size() * dim;
    const auto& emb = taps.features.at(static_cast<std::size_t>(taps.tapIndex(
        {TapKind::kEmbedding, -1})));
    require(emb == readout.features[0], "EMBEDDING tap == rep 0");
    const auto& ffn0 = taps.features.at(static_cast<std::size_t>(
        taps.tapIndex({TapKind::kAfterFfn, 0})));
    require(ffn0 == readout.features[1], "AFTER_FFN(0) == rep 1");
    const auto& ffn1 = taps.features.at(static_cast<std::size_t>(
        taps.tapIndex({TapKind::kAfterFfn, 1})));
    require(ffn1 == readout.features[2], "AFTER_FFN(1) == rep 2 (HEAD_IN)");
    const auto& attn1 = taps.features.at(static_cast<std::size_t>(
        taps.tapIndex({TapKind::kAfterAttn, 1})));
    require(attn1 == readout.features[3], "AFTER_ATTN(1) == PRE_LN_FINAL");
    const auto& norm2_1 = taps.features.at(static_cast<std::size_t>(
        taps.tapIndex({TapKind::kNorm2, 1})));
    require(norm2_1 == readout.features[4], "NORM2(1) == POST_LN_FINAL");
    // Residual identities: the recomputed updates plus the stored tensors
    // reproduce the stored residual streams exactly (float addition).
    for (int li = 0; li < 2; ++li) {
      const auto& x = taps.features.at(static_cast<std::size_t>(
          taps.tapIndex(ibr::blockInputKey(li))));
      const auto& upd = taps.features.at(static_cast<std::size_t>(
          taps.tapIndex({TapKind::kAttnUpdate, li})));
      const auto& r1 = taps.features.at(static_cast<std::size_t>(
          taps.tapIndex({TapKind::kAfterAttn, li})));
      for (std::size_t i = 0; i < n; ++i)
        require(r1[i] == static_cast<float>(static_cast<double>(x[i]) +
                                            static_cast<double>(upd[i])),
                "r1 == x + attn_update bitwise");
      const auto& ffn = taps.features.at(static_cast<std::size_t>(
          taps.tapIndex({TapKind::kFfnUpdate, li})));
      const auto& out = taps.features.at(static_cast<std::size_t>(
          taps.tapIndex({TapKind::kAfterFfn, li})));
      for (std::size_t i = 0; i < n; ++i)
        require(out[i] == static_cast<float>(static_cast<double>(r1[i]) +
                                             static_cast<double>(ffn[i])),
                "out == r1 + ffn_update bitwise");
    }
    require(ibr::paramContentHash(params) == ibr::paramContentHash(params),
            "content hash deterministic");
  }

  // 3. Tap cache round-trip and identity rejection.
  {
    const auto run = dq::runFormalCpu(config, 5, 8, 0.003f,
                                     dq::StabilityMode::LEGACY, {8});
    const auto& params = run.checkpoints.at(8);
    const auto set = ibr::extractTapFeatures(config, params, ds.trainRows);
    ibr::TapCacheIdentity identity;
    identity.protocol = ibr::kProtocolId;
    identity.config = "SELF_TEST";
    identity.seed = 5;
    identity.step = 8;
    identity.datasetHash = ds.combinedHash;
    identity.depth = 2;
    identity.rows = ds.trainRows.size();
    for (const auto& t : set.taps) identity.dims.push_back(t.dim);
    identity.contentHash = ibr::paramContentHash(params);
    const auto dir = std::filesystem::temp_directory_path() / "ibr-cache-test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    require(ibr::writeTapCache(dir, identity, set), "tap cache write");
    ibr::TapSet loaded;
    require(ibr::readTapCache(dir, identity, loaded), "tap cache read");
    require(loaded.taps.size() == set.taps.size() &&
                loaded.features == set.features,
            "tap cache bitwise round-trip");
    ibr::TapCacheIdentity corrupt = identity;
    corrupt.contentHash = "fnv1a64:0000000000000000";
    require(!ibr::readTapCache(dir, corrupt, loaded),
            "tap cache content-hash rejection");
    corrupt = identity;
    corrupt.dims.back() = 17;
    require(!ibr::readTapCache(dir, corrupt, loaded),
            "tap cache dims rejection");
    std::filesystem::remove_all(dir, ec);
  }

  // 4. Probe machinery on tap features: deterministic + learns (regression
  // guard for the z-stats row-count bug).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto taps = ibr::extractTapFeatures(config, params, ds.allRows);
    const int tapId = taps.tapIndex({TapKind::kAfterFfn, 0});
    const rp::ZStats stats = ibr::tapZStats(taps, tapId, ibr::kTrainRows);
    const rp::LayerSet view = taps.layerSetFor(tapId);
    const auto a = rp::trainProbe(view, 0, stats, ds.allRows, 0, ibr::kCalBegin,
                                  ds.allRows, ibr::kCalBegin, ibr::kDevBegin);
    const auto b = rp::trainProbe(view, 0, stats, ds.allRows, 0, ibr::kCalBegin,
                                  ds.allRows, ibr::kCalBegin, ibr::kDevBegin);
    require(a.finite && b.finite, "tap probe finite");
    require(a.trainExact >= 20 && b.trainExact >= 20,
            "tap probe learns (train exact)");
    require(a.trainCe < 1.0 && b.trainCe < 1.0,
            "tap probe learns (train CE)");
    require(a.selectedStep == b.selectedStep && a.probe.w == b.probe.w &&
                a.probe.b == b.probe.b,
            "tap probe deterministic");
  }

  // 5. Head-clone parity on a small configuration (head-input tap).
  {
    const auto run = dq::runFormalCpu(config, 5, 16, 0.003f,
                                     dq::StabilityMode::LEGACY, {16});
    const auto& params = run.checkpoints.at(16);
    const auto taps = ibr::extractTapFeatures(config, params, ds.allRows);
    const int headTap = taps.tapIndex({TapKind::kAfterFfn, 1});
    const auto parity = ibr::headCloneParity(config, params, taps, headTap,
                                             ds.devRows, ibr::kDevBegin);
    require(parity.pass, "head clone parity within tolerance");
    require(parity.argmaxFlips == 0 && parity.exactMismatch == 0,
            "head clone parity exact");
    require(parity.maxLogitDelta < 1e-4, "head clone logit delta within protocol");
  }

  // 6. Deterministic eigendecomposition / pinv on known matrices.
  {
    const std::vector<std::vector<double>> diag = {
        {4.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
    const auto pin = ibr::symmetricPinv(diag, 1e-9);
    require(std::abs(pin[0][0] - 0.25) < 1e-12 &&
                std::abs(pin[1][1] - 1.0) < 1e-12 &&
                std::abs(pin[2][2] - 0.0) < 1e-15,
            "pinv known diagonal");
    const auto eig = ibr::symmetricEigen(diag);
    require(std::abs(eig.values[0] - 4.0) < 1e-12 &&
                std::abs(eig.values[1] - 1.0) < 1e-12 &&
                std::abs(eig.values[2] - 0.0) < 1e-15,
            "eig known diagonal (descending)");
  }

  // 7. Alignment recovers a rotation exactly on synthetic taps.
  {
    const auto run = dq::runFormalCpu(config, 5, 8, 0.003f,
                                     dq::StabilityMode::LEGACY, {8});
    const auto& params = run.checkpoints.at(8);
    auto set = ibr::extractTapFeatures(config, params, ds.trainRows);
    // 32 rows x 16 dims; build B = A * R^T for a fixed rotation R.
    const int dim = 16;
    std::vector<std::vector<double>> r(
        static_cast<std::size_t>(dim), std::vector<double>(static_cast<std::size_t>(dim), 0.0));
    for (int i = 0; i < dim; ++i) r[i][i] = 1.0;
    const double c = std::cos(0.7), s = std::sin(0.7);
    r[0][0] = c; r[0][1] = -s; r[1][0] = s; r[1][1] = c;
    const int srcId = set.tapIndex({TapKind::kNorm1, 0});
    const int dstId = set.tapIndex({TapKind::kAfterAttn, 0});
    auto& a = set.features[static_cast<std::size_t>(srcId)];
    auto& b = set.features[static_cast<std::size_t>(dstId)];
    for (std::size_t row = 0; row < 32; ++row)
      for (int j = 0; j < dim; ++j) {
        double acc = 0.0;
        for (int i = 0; i < dim; ++i)
          acc += static_cast<double>(a[row * dim + i]) * r[i][j];
        b[row * dim + j] = static_cast<float>(acc);
      }
    const auto al = ibr::fitTapAlignment(set, srcId, dstId, 0, 32);
    require(al.finite, "synthetic alignment finite");
    require(al.relResidualOrth < 1e-8 && al.relResidualLs < 1e-8,
            "synthetic rotation recovered");
    // Full-rank synthetic source (DCT-II basis rows, nearly orthonormal) so
    // the Procrustes map is uniquely determined: it must be orthogonal and
    // exactly recover the applied rotation.
    for (std::size_t row = 0; row < 32; ++row)
      for (int i = 0; i < dim; ++i)
        a[row * dim + i] = static_cast<float>(
            std::cos(3.14159265358979323846 * static_cast<double>(i) *
                     (static_cast<double>(row) + 0.5) / 32.0));
    for (std::size_t row = 0; row < 32; ++row)
      for (int j = 0; j < dim; ++j) {
        double acc = 0.0;
        for (int i = 0; i < dim; ++i)
          acc += static_cast<double>(a[row * dim + i]) * r[i][j];
        b[row * dim + j] = static_cast<float>(acc);
      }
    const auto al2 = ibr::fitTapAlignment(set, srcId, dstId, 0, 32);
    require(al2.finite && al2.relResidualLs < 1e-8 && al2.relResidualOrth < 1e-8,
            "full-rank synthetic recovered");
    require(al2.fitRank == dim, "full-rank synthetic fit rank");
    double off = 0.0;
    double mapErrSq = 0.0, mapNrmSq = 0.0;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        double acc = 0.0;
        for (int k = 0; k < dim; ++k)
          acc += al2.mapOrth[i][k] * al2.mapOrth[j][k];
        off += std::abs(acc - (i == j ? 1.0 : 0.0));
        const double e = al2.mapOrth[i][j] - r[i][j];
        mapErrSq += e * e;
        mapNrmSq += r[i][j] * r[i][j];
      }
    const double mapErrRel = std::sqrt(mapErrSq / mapNrmSq);
    require(off < 1e-8, "procrustes map orthogonal (full rank)");
    // Float32 feature storage floors exact map recovery at ~1e-7 accumulated;
    // require the map to match the applied rotation well below any real bug.
    require(mapErrRel < 1e-6, "procrustes map equals applied rotation");
  }

  // 8. Token baselines: structural invariants + Explorer3-verified pins.
  // Explorer3-verified facts: A/C see every row (TRAIN=32, CAL=144, DEV=144);
  // B keys (last token, position%8) are unique per TRAIN row and 83/144
  // CAL+DEV rows are unseen (measured 83, deterministic; the pre-registered
  // "unseen>=100" estimate was corrected to the measured value). Correct
  // counts are data-dependent, not pinned.
  {
    const auto results = ibr::tokenBaselines(ds.trainRows, ds.calRows, ds.devRows);
    const auto& a = results[0];
    const auto& b = results[3];
    const auto& c = results[6];
    require(a.baseline == "BASELINE_A" && a.dataset == "TRAIN" &&
                a.total == 32 && a.seen == 32 && a.unseen == 0,
            "baseline A TRAIN all seen");
    require(results[1].total == 144 && results[1].seen == 144 &&
                results[2].total == 144 && results[2].seen == 144,
            "baseline A CAL/DEV all seen");
    require(c.total == 32 && c.seen == 32 && c.unseen == 0 &&
                results[7].total == 144 && results[7].seen == 144 &&
                results[8].total == 144 && results[8].seen == 144,
            "baseline C 32/144/144 all seen");
    require(b.seen == 32 && b.unseen == 0, "baseline B TRAIN all seen");
    require(results[4].unseen == 83 && results[5].unseen == 83,
            "baseline B CAL/DEV unseen 83 (deterministic)");
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
              << record.maxDropBlock << " (" << record.maxDropValue
              << " tokens), done in " << elapsed.count() << " s"
              << std::endl;
    runs.push_back(std::move(record));
  }

  const Diagnosis diag = diagnose(runs, ds);

  writeDatasetAnchors(paths.root / "dataset-anchors.csv", ds);
  writeTrajectoryAnchors(paths.root / "trajectory-anchors.csv", runs);
  writeTapProbes(paths.root / "tap-probes.csv", runs);
  writeTransfers(paths.root / "tap-transfers.csv", runs);
  writeAlignments(paths.root / "tap-alignments.csv", runs);
  writeGeometry(paths.root / "tap-geometry.csv", runs);
  writeAux(paths.root / "tap-aux.csv", runs);
  writeCloneParity(paths.root / "clone-parity.csv", runs);
  writeBaselines(paths.root / "token-baselines.csv", ds);
  writeFreeRunning(paths.root / "tap-free-running.csv", runs);
  writeDiagnosis(paths.root / "diagnosis.csv", diag);
  writeSummary(paths.root / "summary.csv", runs, ds);
  writeBudget(paths.root / "budget.csv", ds);

  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started);
  std::cout << "=== INTRA_BLOCK_READABILITY_V1 RUN COMPLETE ===" << std::endl;
  std::cout << "elapsed_seconds=" << elapsed.count() << std::endl;
  std::cout << "verdict=" << diag.verdict << std::endl;
  std::cout << "reasons=" << diag.reasons << std::endl;
  for (const auto& run : runs) {
    std::cout << run.spec->publicId << " max_drop_block=" << run.maxDropBlock
              << " drop=" << run.maxDropValue
              << " clone_parity=" << (run.cloneParity.pass ? "PASS" : "FAIL")
              << std::endl;
    for (const auto& entry : run.fr) {
      const auto& m = entry.second;
      std::cout << "  fr " << tapName(run.taps, entry.first) << ": "
                << m.tokenExact << "/" << m.tokenTotal << " seq="
                << m.sequenceExact << "/" << m.sequenceTotal
                << " nll=" << m.autoregressiveNll << std::endl;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool selfTestMode = false;
    std::filesystem::path root =
        "build/reports/qnn-intra-block-readability";
    std::filesystem::path taps =
        "build/reports/qnn-intra-block-readability/private-taps";
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

