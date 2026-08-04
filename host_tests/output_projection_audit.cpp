// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// OUTPUT_PROJECTION_AUDIT_V1 host-only runner.
//
// See output_projection_audit_lib.h for the mathematical orientation and
// transport derivation. No device, QAIRT, ADB, QNN graph, or Android
// involvement; production code is not modified.
#include "output_projection_audit_lib.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace opa = phonelm::output_projection_audit;
namespace aid = phonelm::attention_internal;
namespace rp = phonelm::readout_probe;
namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;
namespace dq = phonelm::depth_quality;
namespace tiny = phonelm::tiny_lm;
namespace ibr = phonelm::intra_block_readability;

constexpr const char* kProtocolId = "OUTPUT_PROJECTION_AUDIT_V1";
constexpr const char* kProtocolHash = "fnv1a64:c35a2e6ae3102772";

namespace {

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
     7.1173910296005136, 9},
    {"L19_SEED_2", 2, 19, 4, 320, 20, 0, 63, 6, 4.1834252619661516,
     20, 65, 9, 4.5271806692690921, 3.2025826990955695, 59, 8,
     6.0751862806397474, 7},
    {"L19_SEED_4", 4, 19, 12, 320, 22, 0, 46, 6, 7.5872917441801651,
     21, 86, 12, 4.5741163228215562, 2.9437165421652174, 92, 11,
     4.030170295310362, 12},
    {"L18_SEED_2_CONTROL", 2, 18, 4, 320, 18, 0, 65, 8,
     5.3026052051209884, 29, 60, 9, 5.6041387784705003, 3.2197527055581228,
     54, 8, 6.7455517100879607, 6},
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
  if (std::abs(marginDevSelected.autoregressiveNll - spec.marginDevSelectedNll) > 1e-6)
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
  if (std::abs(marginDevFinal.autoregressiveNll - spec.marginDevFinalNll) > 1e-6)
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
  if (std::abs(marginCalFinal.autoregressiveNll - spec.marginCalFinalNll) > 1e-6)
    throw std::runtime_error(anchorError(spec, "MARGIN_CAL_FINAL_NLL",
                                         marginCalFinal.autoregressiveNll,
                                         spec.marginCalFinalNll));
}

aid::TapSet loadTapSet(const ConfigSpec& spec, const train::P& params,
                       const DataSet& ds,
                       const std::filesystem::path& tapCacheDir) {
  aid::TapSet set;
  const auto registry = aid::buildTapRegistry(static_cast<int>(spec.layers));
  aid::TapCacheIdentity identity;
  identity.protocol = aid::kProtocolId;
  identity.config = spec.publicId;
  identity.seed = spec.seed;
  identity.step = spec.finalStep;
  identity.datasetHash = ds.combinedHash;
  identity.depth = static_cast<int>(spec.layers);
  identity.rows = ds.allRows.size();
  for (const auto& t : registry) identity.dims.push_back(t.dim);
  identity.contentHash = ibr::paramContentHash(params);
  if (aid::readTapCache(tapCacheDir, identity, set)) return set;

  std::cerr << "[" << spec.publicId
            << "] tap cache miss; extracting (this is expected if the cache"
               " file is absent or the parameter hash changed)"
            << std::endl;
  set = aid::extractTapFeatures(modelConfig(spec.layers), params, ds.allRows);
  return set;
}

struct LayerAudit {
  int layer = -1;
  opa::MatrixStats stats;
  rp::ProbeTrainResult ctxProbe;
  rp::ProbeTrainResult projProbe;
  opa::TransportResult transport;
  rp::ProbeTrainResult warmStartProbe;
  std::map<std::string, opa::LogitParity> parity;  // TRAIN / CALIBRATION / DEVELOPMENT
  std::map<std::string, opa::LogitParity> floatParity;  // double-vs-float transport
  std::map<std::string, rp::TokenMetrics> ctxMetrics;
  std::map<std::string, rp::TokenMetrics> projMetrics;
  std::map<std::string, rp::TokenMetrics> transportMetrics;
  std::map<std::string, rp::TokenMetrics> warmStartMetrics;
  bool fullAudit = false;
};

rp::TokenMetrics evalOnPartition(const rp::Probe& probe,
                                 const rp::ZStats& stats,
                                 const rp::LayerSet& set,
                                 const std::vector<rp::ProbeRow>& rows,
                                 std::size_t offset) {
  return rp::probeTokenMetrics(probe, stats, set, 0, rows, offset);
}

inline void csvRow(rp::CsvWriter& w,
                   std::initializer_list<std::string> vals) {
  w.row(std::vector<std::string>(vals.begin(), vals.end()));
}

LayerAudit auditLayer(const train::P& params, const tiny::Config& config,
                      const aid::TapSet& taps, const DataSet& ds, int layer,
                      bool full) {
  LayerAudit out;
  out.layer = layer;
  out.fullAudit = full;

  const auto& p = train::layer(params, static_cast<std::uint32_t>(layer));
  const auto w = opa::outputProjectionMatrix(p.wo, static_cast<int>(config.dimension));
  out.stats = opa::analyzeMatrixWithDetSign(w);

  if (!full) return out;

  const int ctxTapId = taps.tapIndex({aid::TapKind::kCtxConcat, layer});
  const int projTapId = taps.tapIndex({aid::TapKind::kAttnUpdate, layer});
  const rp::ZStats ctxStats = aid::tapZStats(taps, ctxTapId, aid::kTrainRows);
  const rp::ZStats projStats = aid::tapZStats(taps, projTapId, aid::kTrainRows);
  const rp::LayerSet ctxSet = taps.layerSetFor(ctxTapId);
  const rp::LayerSet projSet = taps.layerSetFor(projTapId);

  out.ctxProbe = rp::trainProbe(ctxSet, 0, ctxStats, ds.allRows, 0,
                                aid::kCalBegin, ds.allRows, aid::kCalBegin,
                                aid::kDevBegin);
  out.projProbe = rp::trainProbe(projSet, 0, projStats, ds.allRows, 0,
                                 aid::kCalBegin, ds.allRows, aid::kCalBegin,
                                 aid::kDevBegin);

  if (out.ctxProbe.finite) {
    out.transport = opa::transportProbe(out.ctxProbe.probe, ctxStats, projStats,
                                        w, out.stats.doubleTol);
    if (out.transport.finite) {
      out.warmStartProbe = opa::trainProbeFromInit(
          projSet, 0, projStats, out.transport.probe, ds.allRows, 0,
          aid::kCalBegin, ds.allRows, aid::kCalBegin, aid::kDevBegin);
    }
  }

  const std::vector<std::pair<std::string, std::size_t>> partitions{
      {"TRAIN", 0},
      {"CALIBRATION", aid::kCalBegin},
      {"DEVELOPMENT", aid::kDevBegin},
  };

  for (const auto& part : partitions) {
    const auto& rows = part.first == "TRAIN" ? ds.trainRows
                        : part.first == "CALIBRATION" ? ds.calRows
                                                       : ds.devRows;
    out.ctxMetrics[part.first] = evalOnPartition(
        out.ctxProbe.probe, ctxStats, ctxSet, rows, part.second);
    out.projMetrics[part.first] = evalOnPartition(
        out.projProbe.probe, projStats, projSet, rows, part.second);
    if (out.transport.finite) {
      out.transportMetrics[part.first] = evalOnPartition(
          out.transport.probe, projStats, projSet, rows, part.second);
      out.parity[part.first] = opa::compareProbeLogits(
          out.ctxProbe.probe, ctxStats, ctxSet, 0, out.transport.probe,
          projStats, projSet, 0, rows, part.second);
      // Double-vs-float transport.
      rp::Probe floatProbe = opa::transportProbeFloat(out.ctxProbe.probe, ctxStats,
                                                      projStats, w);
      out.floatParity[part.first] = opa::compareProbeLogits(
          out.transport.probe, projStats, projSet, 0, floatProbe, projStats,
          projSet, 0, rows, part.second);
    }
    if (out.warmStartProbe.finite) {
      out.warmStartMetrics[part.first] = evalOnPartition(
          out.warmStartProbe.probe, projStats, projSet, rows, part.second);
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Self-test: synthetic matrices.
// ---------------------------------------------------------------------------
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("SELF_TEST_FAILED: " + message);
}

template <typename T>
inline std::vector<std::vector<T>> randomMatrix(int rows, int cols,
                                                 std::uint32_t seed,
                                                 T scale) {
  std::vector<std::vector<T>> m(static_cast<std::size_t>(rows),
                                std::vector<T>(static_cast<std::size_t>(cols)));
  std::uint32_t s = seed * 747796405u + 2891336453u;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      s = s * 1664525u + 1013904223u;
      const T draw = static_cast<T>(
          (static_cast<double>(static_cast<int>((s >> 8) & 65535u)) / 32767.5 - 1.0) *
          static_cast<double>(scale));
      m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = draw;
    }
  return m;
}

inline opa::Mat<double> orthogonalMatrix(int n, std::uint32_t seed) {
  // Gram-Schmidt QR on a random matrix.
  auto a = randomMatrix<double>(n, n, seed, 1.0);
  opa::Mat<double> q = opa::zerosMat<double>(n, n);
  for (int j = 0; j < n; ++j) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    for (int k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < n; ++i) dot += v[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
      for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] -= dot * q[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
    }
    double norm = 0.0;
    for (double x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (int i = 0; i < n; ++i) q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(i)] / norm;
  }
  return q;
}

inline opa::Mat<double> diagMatrix(const std::vector<double>& diag) {
  const int n = static_cast<int>(diag.size());
  auto m = opa::zerosMat<double>(n, n);
  for (int i = 0; i < n; ++i) m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = diag[static_cast<std::size_t>(i)];
  return m;
}

void selfTest() {
  const int dim = 16;
  const int classes = 32;
  const std::uint32_t seed = 20260805u;

  // 1. SVD reconstruction and pseudoinverse identities.
  {
    auto w = randomMatrix<double>(dim, dim, seed, 0.3);
    const auto svd = opa::computeSvd(w);
    auto sm = opa::zerosMat<double>(dim, dim);
    for (int i = 0; i < dim; ++i) sm[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = svd.s[static_cast<std::size_t>(i)];
    auto recon = opa::mulMat(opa::mulMat(svd.u, sm), svd.vt);
    double err = 0.0;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        err = std::max(err, std::abs(recon[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));
    require(err < 1e-12, "svd reconstruction");

    const auto stats = opa::analyzeMatrix(w);
    const auto pinv = opa::pseudoInverse(w, stats.doubleTol);
    auto wwpw = opa::mulMat(opa::mulMat(w, pinv), w);
    err = 0.0;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        err = std::max(err, std::abs(wwpw[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - w[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));
    require(err < 1e-10, "pinv identity W W^+ W = W");
  }

  // 2. Rank detection on known matrices.
  {
    auto id = opa::identityMat<double>(dim);
    auto stats = opa::analyzeMatrix(id);
    require(stats.mathRank == dim, "identity full rank");
    require(std::abs(stats.conditionDouble - 1.0) < 1e-9, "identity condition");

    std::vector<double> diag(dim, 1.0);
    diag.back() = 1e-9;
    auto ill = diagMatrix(diag);
    stats = opa::analyzeMatrix(ill);
    require(stats.mathRank == dim, "ill-conditioned full rank in double");
    require(stats.floatRank < dim, "ill-conditioned rank deficient in float");

    std::vector<double> rank15(dim, 1.0);
    rank15.back() = 0.0;
    auto r15 = diagMatrix(rank15);
    stats = opa::analyzeMatrix(r15);
    require(stats.mathRank == dim - 1, "rank 15 detection");

    std::vector<double> rank8(dim, 0.0);
    for (int i = 0; i < 8; ++i) rank8[static_cast<std::size_t>(i)] = 1.0;
    auto r8 = diagMatrix(rank8);
    stats = opa::analyzeMatrix(r8);
    require(stats.mathRank == 8, "rank 8 detection");
  }

  // 3. Orthogonal matrix transport preserves information.
  {
    auto q = orthogonalMatrix(dim, seed);
    auto stats = opa::analyzeMatrix(q);
    require(stats.mathRank == dim, "orthogonal full rank");

    // Build a fake context probe and stats.
    rp::Probe ctxProbe = rp::initialProbe(dim);
    rp::ZStats ctxStats, projStats;
    ctxStats.dim = dim;
    ctxStats.mean.assign(dim, 0.1);
    ctxStats.std.assign(dim, 1.0);
    projStats.dim = dim;
    projStats.mean.assign(dim, -0.2);
    projStats.std.assign(dim, 0.7);

    auto tr = opa::transportProbe(ctxProbe, ctxStats, projStats, q,
                                  stats.doubleTol);
    require(tr.finite, "orthogonal transport finite");
    require(tr.nullspaceFraction < 1e-12, "orthogonal transport no nullspace");

    // Check a few random x vectors. Use float features and float y = x W
    // to mirror the actual tap pipeline; tolerance matches the protocol.
    auto xs = randomMatrix<double>(100, dim, seed + 1, 2.0);
    double maxDiff = 0.0;
    std::vector<double> logCtx(classes), logProj(classes);
    for (const auto& x : xs) {
      std::vector<float> fx(dim);
      for (int d = 0; d < dim; ++d) fx[d] = static_cast<float>(x[d]);
      rp::probeForward(ctxProbe, ctxStats, fx.data(), logCtx.data());
      std::vector<float> fy(dim, 0.0f);
      for (int j = 0; j < dim; ++j)
        for (int i = 0; i < dim; ++i)
          fy[j] += fx[i] * static_cast<float>(q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
      rp::probeForward(tr.probe, projStats, fy.data(), logProj.data());
      for (int c = 0; c < classes; ++c)
        maxDiff = std::max(maxDiff, std::abs(logCtx[c] - logProj[c]));
    }
    require(maxDiff < 1e-5, "orthogonal transport logit parity");
  }

  // 4. Rank-deficient matrix with probe inside/outside nullspace.
  {
    // W = [I_8 0; 0 0] in 16x16.
    std::vector<double> diag(dim, 0.0);
    for (int i = 0; i < 8; ++i) diag[static_cast<std::size_t>(i)] = 1.0;
    auto w = diagMatrix(diag);
    auto stats = opa::analyzeMatrix(w);
    require(stats.mathRank == 8, "block rank 8 detection");

    rp::Probe ctxProbe = rp::initialProbe(dim);
    rp::ZStats ctxStats, projStats;
    ctxStats.dim = dim;
    ctxStats.mean.assign(dim, 0.0);
    ctxStats.std.assign(dim, 1.0);
    projStats.dim = dim;
    projStats.mean.assign(dim, 0.0);
    projStats.std.assign(dim, 1.0);

    // Set all weights in first 8 dims to zero so C lies in nullspace.
    for (int c = 0; c < classes; ++c)
      for (int d = 0; d < 8; ++d)
        ctxProbe.w[static_cast<std::size_t>(c) * dim + d] = 0.0;
    auto tr = opa::transportProbe(ctxProbe, ctxStats, projStats, w,
                                  stats.doubleTol);
    require(tr.nullspaceFraction > 0.99, "nullspace probe high loss fraction");
  }

  std::cout << "SELF_TEST_PASS" << std::endl;
}

// ---------------------------------------------------------------------------
// Report writers.
// ---------------------------------------------------------------------------
std::vector<int> targetLayersFor(std::uint32_t layers) {
  if (layers == 19)
    return std::vector<int>(aid::kTargetLayersL19.begin(),
                            aid::kTargetLayersL19.end());
  if (layers == 18)
    return std::vector<int>(aid::kTargetLayersL18.begin(),
                            aid::kTargetLayersL18.end());
  std::vector<int> v;
  for (int i = 0; i < static_cast<int>(layers); ++i) v.push_back(i);
  return v;
}

void writeDatasetUsage(const std::filesystem::path& path, const DataSet& ds) {
  rp::CsvWriter csv(path);
  csv.header({"dataset", "role", "hash", "rows"});
  csvRow(csv, {"TRAIN", "probe_fit_and_z_stats",
           ar::partitionHash(ar::Partition::TRAIN, 8),
           rp::text(std::uint64_t(ds.trainRows.size()))});
  csvRow(csv, {"MARGIN_CALIBRATION_V1", "probe_step_selection",
           cm::partitionHash(cm::Partition::CALIBRATION, 8),
           rp::text(std::uint64_t(ds.calRows.size()))});
  csvRow(csv, {"MARGIN_DEVELOPMENT_V1", "final_evaluation",
           cm::partitionHash(cm::Partition::DEVELOPMENT, 8),
           rp::text(std::uint64_t(ds.devRows.size()))});
  csvRow(csv, {"AR_FINAL_HOLDOUT_V3", "UNOPENED_hash_verified_only",
           ar::partitionHash(ar::Partition::FINAL, 8), "0"});
}

void writeConfiguration(const std::filesystem::path& path) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "seed", "layers", "final_step",
              "max_drop_block", "ar_selected_step"});
  for (const auto& spec : kSpecs)
    csvRow(csv, {spec.publicId, rp::text(std::uint64_t(spec.seed)),
             rp::text(std::uint64_t(spec.layers)), rp::text(spec.finalStep),
             rp::text(spec.maxDropBlock), rp::text(spec.arSelectedStep)});
}

void writeSingularValues(const std::filesystem::path& path,
                         const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "index", "singular_value"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    for (std::size_t i = 0; i < la.stats.singularValues.size(); ++i)
      csvRow(csv, {spec, rp::text(la.layer), rp::text(std::uint64_t(i)),
               rp::text(la.stats.singularValues[i], 12)});
  }
}

void writeProjectionMatrixSummary(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "sigma_min", "sigma_max",
              "condition_double", "condition_float", "math_rank",
              "float_rank", "effective_rank", "participation_ratio",
              "frobenius_norm", "spectral_norm", "determinant_sign",
              "log_abs_determinant"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    csvRow(csv, {spec, rp::text(la.layer),
             rp::text(la.stats.sigmaMin, 12),
             rp::text(la.stats.sigmaMax, 12),
             rp::text(la.stats.conditionDouble, 6),
             rp::text(la.stats.conditionFloat, 6),
             rp::text(la.stats.mathRank),
             rp::text(la.stats.floatRank),
             rp::text(la.stats.effectiveRank, 6),
             rp::text(la.stats.participationRatio, 6),
             rp::text(la.stats.frobeniusNorm, 12),
             rp::text(la.stats.spectralNorm, 12),
             rp::text(la.stats.determinantSign),
             rp::text(la.stats.logAbsDeterminant, 12)});
  }
}

void writeSingularValueSummary(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "sigma_min", "sigma_max",
              "condition_double", "math_rank", "float_rank",
              "effective_rank"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    csvRow(csv, {spec, rp::text(la.layer),
             rp::text(la.stats.sigmaMin, 12),
             rp::text(la.stats.sigmaMax, 12),
             rp::text(la.stats.conditionDouble, 6),
             rp::text(la.stats.mathRank),
             rp::text(la.stats.floatRank),
             rp::text(la.stats.effectiveRank, 6)});
  }
}

void writeRankAndConditioning(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  // Same data as projection-matrix-summary, kept separate for readability.
  writeProjectionMatrixSummary(path, audits);
}

void writeProbeTransportSummary(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "partition",
              "context_token_exact", "from_scratch_token_exact",
              "transport_token_exact", "warm_start_token_exact",
              "max_logit_diff", "mean_logit_diff", "rms_logit_diff",
              "argmax_flips", "token_exact_diff"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    for (const auto& part : {"TRAIN", "CALIBRATION", "DEVELOPMENT"}) {
      const std::string pstr = part;
      const auto& par = la.parity.at(pstr);
      csvRow(csv, {spec, rp::text(la.layer), pstr,
               rp::text(la.ctxMetrics.at(pstr).tokenExact),
               rp::text(la.projMetrics.at(pstr).tokenExact),
               rp::text(la.transportMetrics.at(pstr).tokenExact),
               rp::text(la.warmStartMetrics.at(pstr).tokenExact),
               rp::text(par.maxAbsDiff, 12),
               rp::text(par.meanAbsDiff, 12),
               rp::text(par.rmsDiff, 12),
               rp::text(par.argmaxFlips),
               rp::text(par.tokenExactDiff)});
    }
  }
}

void writeProbeTransportBySeed(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  writeProbeTransportSummary(path, audits);
}

void writeFloatDoubleComparison(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "partition",
              "max_logit_diff", "mean_logit_diff", "rms_logit_diff",
              "argmax_flips", "token_exact_diff"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    for (const auto& part : {"TRAIN", "CALIBRATION", "DEVELOPMENT"}) {
      const std::string pstr = part;
      const auto& par = la.floatParity.at(pstr);
      csvRow(csv, {spec, rp::text(la.layer), pstr,
               rp::text(par.maxAbsDiff, 12),
               rp::text(par.meanAbsDiff, 12),
               rp::text(par.rmsDiff, 12),
               rp::text(par.argmaxFlips),
               rp::text(par.tokenExactDiff)});
    }
  }
}

void writeNullspaceSummary(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "overall_lost_fraction"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    csvRow(csv, {spec, rp::text(la.layer),
             rp::text(la.transport.nullspaceFraction, 12)});
  }
}

void writeNullspacePrivate(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "class", "c_norm",
              "c_lost_norm", "lost_fraction"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    const int classes = static_cast<int>(la.transport.c.size());
    for (int c = 0; c < classes; ++c) {
      double cNorm = 0.0, lNorm = 0.0;
      for (std::size_t d = 0; d < la.transport.c[0].size(); ++d) {
        const double v = la.transport.c[static_cast<std::size_t>(c)][d];
        const double lv = la.transport.cLost[static_cast<std::size_t>(c)][d];
        cNorm += v * v;
        lNorm += lv * lv;
      }
      cNorm = std::sqrt(cNorm);
      lNorm = std::sqrt(lNorm);
      csvRow(csv, {spec, rp::text(la.layer), rp::text(c),
               rp::text(cNorm, 12), rp::text(lNorm, 12),
               rp::text(cNorm > 0.0 ? lNorm / cNorm : 0.0, 12)});
    }
  }
}

void writeFromScratchVsTransport(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layer", "partition",
              "from_scratch_exact", "transport_exact",
              "warm_start_exact", "transport_argmax_flips"});
  for (const auto& entry : audits) {
    const auto& spec = entry.first;
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    for (const auto& part : {"TRAIN", "CALIBRATION", "DEVELOPMENT"}) {
      const std::string pstr = part;
      csvRow(csv, {spec, rp::text(la.layer), pstr,
               rp::text(la.projMetrics.at(pstr).tokenExact),
               rp::text(la.transportMetrics.at(pstr).tokenExact),
               rp::text(la.warmStartMetrics.at(pstr).tokenExact),
               rp::text(la.parity.at(pstr).argmaxFlips)});
    }
  }
}

void writeDepthControl(const std::filesystem::path& path,
                       const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"configuration_id", "layers", "max_drop_block",
              "math_rank", "condition_double", "nullspace_fraction"});
  for (const auto& entry : audits) {
    const auto& specName = entry.first;
    const auto specIt =
        std::find_if(kSpecs.begin(), kSpecs.end(),
                     [&](const ConfigSpec& s) { return s.publicId == specName; });
    if (specIt == kSpecs.end()) continue;
    const ConfigSpec& specRef = *specIt;
    const auto& la = entry.second;
    if (la.layer != specRef.maxDropBlock) continue;
    csvRow(csv, {specName, rp::text(std::uint64_t(specRef.layers)),
             rp::text(la.layer), rp::text(la.stats.mathRank),
             rp::text(la.stats.conditionDouble, 6),
             rp::text(la.fullAudit ? la.transport.nullspaceFraction : 0.0, 12)});
  }
}

std::string computeVerdict(
    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  int preservedCount = 0, illCount = 0, lostCount = 0, layerCount = 0;
  for (const auto& entry : audits) {
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    ++layerCount;
    const bool rankFull = la.stats.mathRank == la.stats.dim;
    const bool transportMatches =
        la.parity.at("DEVELOPMENT").maxAbsDiff <= opa::kTransportLogitTol &&
        la.parity.at("DEVELOPMENT").argmaxFlips == 0 &&
        la.parity.at("DEVELOPMENT").tokenExactDiff == 0;
    const bool floatMatches =
        la.floatParity.at("DEVELOPMENT").maxAbsDiff <= 1e-3 &&
        la.floatParity.at("DEVELOPMENT").argmaxFlips == 0;
    const bool nullspace = la.transport.nullspaceFraction > 1e-6;
    if (rankFull && transportMatches) ++preservedCount;
    if (rankFull && transportMatches && !floatMatches) ++illCount;
    if (!rankFull && nullspace && !transportMatches) ++lostCount;
  }

  if (lostCount > 0 && preservedCount == 0)
    return "OUTPUT_PROJECTION_LOSES_INFORMATION";
  if (preservedCount > 0 && lostCount == 0)
    return "OUTPUT_PROJECTION_PRESERVES_INFORMATION";
  if (illCount > 0) return "OUTPUT_PROJECTION_ILL_CONDITIONED";
  return "UNDETERMINED";
}

std::string verdictReason(const std::string& verdict) {
  if (verdict == "OUTPUT_PROJECTION_PRESERVES_INFORMATION")
    return "All audited matrices are full rank and the transported context probe matches the projection probe at step 0 with no argmax flips; the observed previous drop is a learning/standardization/optimization artifact.";
  if (verdict == "OUTPUT_PROJECTION_ILL_CONDITIONED")
    return "Matrices are full rank and double transport matches, but float transport diverges; the projection maps information to a numerically ill-conditioned coordinate system.";
  if (verdict == "OUTPUT_PROJECTION_LOSES_INFORMATION")
    return "At least one audited matrix is rank-deficient and the lost null-space component corresponds to the observed probe drop.";
  if (verdict == "SEED_LAYER_DEPENDENT")
    return "Some seeds/layers preserve information while others lose it.";
  return "Results are inconsistent or inconclusive; verify W orientation, z-score fold, and pseudoinverse tolerance.";
}

void writeDiagnosis(const std::filesystem::path& path,
                    const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  int preservedCount = 0, illCount = 0, lostCount = 0, layerCount = 0;
  for (const auto& entry : audits) {
    const auto& la = entry.second;
    if (!la.fullAudit) continue;
    ++layerCount;
    const bool rankFull = la.stats.mathRank == la.stats.dim;
    const bool transportMatches =
        la.parity.at("DEVELOPMENT").maxAbsDiff <= opa::kTransportLogitTol &&
        la.parity.at("DEVELOPMENT").argmaxFlips == 0 &&
        la.parity.at("DEVELOPMENT").tokenExactDiff == 0;
    const bool floatMatches =
        la.floatParity.at("DEVELOPMENT").maxAbsDiff <= 1e-3 &&
        la.floatParity.at("DEVELOPMENT").argmaxFlips == 0;
    const bool nullspace = la.transport.nullspaceFraction > 1e-6;
    if (rankFull && transportMatches) ++preservedCount;
    if (rankFull && transportMatches && !floatMatches) ++illCount;
    if (!rankFull && nullspace && !transportMatches) ++lostCount;
  }
  const std::string verdict = computeVerdict(audits);
  const std::string reason = verdictReason(verdict);

  rp::CsvWriter csv(path);
  csv.header({"verdict", "preserved_layers", "ill_conditioned_layers",
              "lost_layers", "audited_layers", "criteria_fixed_before_results",
              "reason"});
  csvRow(csv, {verdict, rp::text(preservedCount), rp::text(illCount),
           rp::text(lostCount), rp::text(layerCount), "true", reason});
}

void writePreviousResultCorrection(const std::filesystem::path& path,
                                   const std::string& diagnosisVerdict) {
  rp::CsvWriter csv(path);
  csv.header({"previous_claim", "previous_evidence", "this_audit_verdict",
              "correction_required"});
  csvRow(csv, {"Attention output projection causes information loss",
           "CTX_CONCAT dev TF exact >> ATT_UPDATE dev TF exact in ATTENTION_INTERNAL_V1",
           diagnosisVerdict,
           diagnosisVerdict == "OUTPUT_PROJECTION_PRESERVES_INFORMATION"
               ? "yes: rephrase as optimization/standardization/numerical difficulty"
               : (diagnosisVerdict == "OUTPUT_PROJECTION_LOSES_INFORMATION"
                      ? "no: continue to support"
                      : "partial: add nuance")});
}

void writeNextStepCandidates(const std::filesystem::path& path,
                             const std::string& diagnosisVerdict) {
  rp::CsvWriter csv(path);
  csv.header({"candidate", "rationale", "verdict"});
  if (diagnosisVerdict == "OUTPUT_PROJECTION_PRESERVES_INFORMATION") {
    csvRow(csv, {"Probe initialization / standardization audit",
             "The projection preserves linear information; the observed drop is"
             " a learning/standardization artifact.",
             diagnosisVerdict});
    csvRow(csv, {"Alternative optimizer or longer warm-start",
             "Transported probe matches at step 0; investigate why from-scratch"
             " optimization does not find the same solution.",
             diagnosisVerdict});
  } else if (diagnosisVerdict == "OUTPUT_PROJECTION_ILL_CONDITIONED") {
    csvRow(csv, {"Mixed-precision sensitivity audit",
             "Double transport works but float transport does not; the matrix"
             " is numerically ill-conditioned for FP32.",
             diagnosisVerdict});
  } else if (diagnosisVerdict == "OUTPUT_PROJECTION_LOSES_INFORMATION") {
    csvRow(csv, {"Residual / FFN downstream recovery",
             "The output projection genuinely loses linear class information;"
             " check whether the residual add or later layers recover it.",
             diagnosisVerdict});
  } else {
    csvRow(csv, {"Re-examine identity and transport implementation",
             "Results are inconsistent; verify W orientation, z-score fold, and"
             " pseudoinverse tolerance.",
             diagnosisVerdict});
  }
}

void writeBudget(const std::filesystem::path& path,
                 const std::vector<std::pair<std::string, LayerAudit>>& audits) {
  rp::CsvWriter csv(path);
  csv.header({"item", "count", "limit", "ok"});
  std::uint64_t matrixStats = audits.size();
  std::uint64_t fullAudits = 0;
  for (const auto& e : audits)
    if (e.second.fullAudit) ++fullAudits;
  auto row = [&](const std::string& item, std::uint64_t count,
                 std::uint64_t limit) {
    csvRow(csv, {item, rp::text(count), rp::text(limit),
             count <= limit ? "true" : "false"});
  };
  row("matrix_decompositions", matrixStats, 60);
  row("full_probe_transports", fullAudits, 24);
  row("transport_warm_start_trainings", fullAudits, 24);
  row("cpu_trajectory_regenerations", static_cast<std::uint64_t>(kSpecs.size()), 4);
}

void writeRunIdentity(const std::filesystem::path& path,
                      const std::string& protocol,
                      const std::string& protocolHash) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write run-identity.json");
  out << "{\n"
      << "  \"protocol\": \"" << protocol << "\",\n"
      << "  \"protocol_hash\": \"" << protocolHash << "\",\n"
      << "  \"start_head\": \"1bf296a01e0962419c34fff0af1c9aedf4f1d7c3\",\n"
      << "  \"final_holdout_opened\": false,\n"
      << "  \"device_runs\": 0,\n"
      << "  \"htp_runs\": 0\n"
      << "}\n";
}

// ---------------------------------------------------------------------------
// Production run.
// ---------------------------------------------------------------------------
int runProduction(const std::filesystem::path& reportRoot,
                  const std::filesystem::path& tapCacheDir) {
  validateDatasets();
  const DataSet ds = buildDataSet();
  std::filesystem::create_directories(reportRoot);

  std::vector<std::pair<std::string, LayerAudit>> allAudits;
  allAudits.reserve(60);

  for (const auto& spec : kSpecs) {
    const auto config = modelConfig(spec.layers);
    std::cerr << "[" << spec.publicId << "] regenerating trajectory ..."
              << std::endl;
    const auto run = dq::runFormalCpu(config, spec.seed, spec.finalStep, 0.003f,
                                      dq::StabilityMode::LEGACY,
                                      {spec.arSelectedStep, spec.finalStep});
    runAnchors(spec, run, ds);
    const auto& params = run.checkpoints.at(spec.finalStep);
    const auto taps = loadTapSet(spec, params, ds, tapCacheDir);

    const auto targets = targetLayersFor(spec.layers);
    for (int li : targets) {
      const bool full = (li == spec.maxDropBlock);
      std::cerr << "[" << spec.publicId << "] auditing layer " << li
                << (full ? " (full)" : " (stats)") << std::endl;
      auto audit = auditLayer(params, config, taps, ds, li, full);
      allAudits.emplace_back(spec.publicId, std::move(audit));
    }
  }

  writeDatasetUsage(reportRoot / "dataset-usage.csv", ds);
  writeConfiguration(reportRoot / "configuration.csv");
  writeSingularValues(reportRoot / "singular-values-raw.csv", allAudits);
  writeProjectionMatrixSummary(reportRoot / "projection-matrix-summary.csv",
                               allAudits);
  writeSingularValueSummary(reportRoot / "singular-value-summary.csv", allAudits);
  writeRankAndConditioning(reportRoot / "rank-and-conditioning.csv", allAudits);
  writeProbeTransportSummary(reportRoot / "probe-transport-summary.csv",
                             allAudits);
  writeProbeTransportBySeed(reportRoot / "probe-transport-by-seed.csv",
                            allAudits);
  writeFloatDoubleComparison(reportRoot / "float-double-comparison.csv",
                             allAudits);
  writeNullspaceSummary(reportRoot / "nullspace-summary.csv", allAudits);
  writeNullspacePrivate(reportRoot / "nullspace-private.csv", allAudits);
  writeFromScratchVsTransport(reportRoot / "from-scratch-vs-transport.csv",
                              allAudits);
  writeDepthControl(reportRoot / "depth-control.csv", allAudits);
  writeBudget(reportRoot / "budget.csv", allAudits);
  writeRunIdentity(reportRoot / "run-identity.json", kProtocolId,
                   kProtocolHash);

  writeDiagnosis(reportRoot / "diagnosis.csv", allAudits);
  const std::string diagVerdict = computeVerdict(allAudits);

  writePreviousResultCorrection(reportRoot / "previous-result-correction.csv",
                                diagVerdict);
  writeNextStepCandidates(reportRoot / "next-step-candidates.csv", diagVerdict);

  std::cout << "=== OUTPUT_PROJECTION_AUDIT_V1 RUN COMPLETE ===" << std::endl;
  std::cout << "verdict=" << diagVerdict << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool selfTestMode = false;
    std::filesystem::path reportRoot =
        "build/reports/qnn-output-projection-audit";
    std::filesystem::path tapCacheDir =
        "build/reports/qnn-attention-internal-diagnosis/private-taps";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") selfTestMode = true;
      else if (arg == "--run") selfTestMode = false;
      else if (arg == "--report-root" && i + 1 < argc)
        reportRoot = argv[++i];
      else if (arg == "--tap-root" && i + 1 < argc)
        tapCacheDir = argv[++i];
      else
        throw std::invalid_argument("UNKNOWN_ARG: " + arg);
    }
    if (selfTestMode) {
      selfTest();
      return 0;
    }
    return runProduction(reportRoot, tapCacheDir);
  } catch (const std::exception& error) {
    std::cerr << "FATAL: " << error.what() << std::endl;
    return 1;
  }
}
