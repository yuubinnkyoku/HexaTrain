// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only intra-block readability analysis for the L19 transformer.
//
// The forward pass is the VERBATIM host copy train::generalForward
// (critical_margin_training_lib.h, itself a bitwise copy of
// app/src/main/cpp/tiny_language_model_cpu.cpp). This library only OBSERVES
// the tensors already kept in the returned GF structure (plus the two update
// tensors recomputed with the same mm arithmetic), so no production code is
// touched and the existing CPU results stay bitwise invariant. The probe
// machinery (trainProbe / z-score / free-running rollouts / cache / CSV) is
// reused from readout_probe_lib.h unchanged.
//
// Protocol: INTRA_BLOCK_READABILITY_V1 (fixed before results, see
// protocol.json). Tap registry per configuration depth:
//   EMBEDDING                                (dim 16)
//   per block li in [0, depth):
//     NORM1        = LN1 output            (dim 16)
//     ATT_UPDATE   = ctx @ Wo              (dim 16)
//     AFTER_ATTN   = residual stream after attention add (r1)
//     NORM2        = LN2 output            (dim 16)
//     FFN_UPDATE   = relu @ W2             (dim 16)
//     AFTER_FFN    = residual stream after FFN add (out)
//   representative blocks {1,4,8,12,16} (0-based):
//     CTX          = attention context     (dim 16)
//     RELU         = post-activation FFN   (dim 32)
//   the LAST block also gets CTX.
// AFTER_FFN(li) is the SAME tensor as the next block's input; the final
// block's AFTER_FFN is the head input (== rep L of the readout protocol),
// its NORM2 is POST_LN_FINAL (== rep L+2) and its AFTER_ATTN is
// PRE_LN_FINAL (== rep L+1). Probes on those taps must therefore reproduce
// the previous readout bundle exactly (cross-bundle identity check).
#ifndef INTRA_BLOCK_READABILITY_LIB_H
#define INTRA_BLOCK_READABILITY_LIB_H

#include "readout_probe_lib.h"
#include "critical_margin_training_lib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::intra_block_readability {

namespace train = phonelm::critical_margin::train;
namespace rp = phonelm::readout_probe;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;
namespace cm = phonelm::critical_margin;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;

inline constexpr const char* kProtocolId = "INTRA_BLOCK_READABILITY_V1";
// Row layout (identical to READOUT_PROBE_V1): TRAIN 32 rows (4 cases x 8
// windows), CAL 144, DEV 144; row 0 is always a TRAIN row.
inline constexpr std::size_t kTrainRows = 32;
inline constexpr std::size_t kCalBegin = 32;
inline constexpr std::size_t kCalRows = 144;
inline constexpr std::size_t kDevBegin = 176;
inline constexpr std::size_t kDevRows = 144;
inline constexpr std::size_t kTotalRows = 320;
// Alignment / parity constants (fixed before results).
inline constexpr double kAlignRcond = 1.0e-9;
inline constexpr int kAlignSweeps = 24;
inline constexpr double kParityLogitTol = 1.0e-4;
inline constexpr double kParityNllTol = 1.0e-4;
inline constexpr double kParityMarginTol = 1.0e-4;
// Diagnostic thresholds (fixed before results; never tuned).
inline constexpr int kDropTokens = 5;               // of 144 dev tokens
inline constexpr double kResidualRatioThreshold = 1.0;
inline constexpr double kCosineThreshold = -0.5;
inline constexpr double kRecoveryHigh = 0.75;       // coordinate-transform
inline constexpr double kRecoveryLow = 0.25;        // information-loss
inline constexpr std::int64_t kInfoLossGapMin = 3;  // native - aligned
inline constexpr std::int64_t kCoordinateGapMax = 1;
// Deep-band pool verdicts (blocks 11..18 on L19, coarse pairs only):
// 8 blocks x 2 pairs = 16 pairs; majority >= 10 (>= 62.5%).
inline constexpr int kDeepBandPairs = 16;
inline constexpr int kDeepBandMajority = 10;
// Representative blocks (0-based) for fine taps CTX/RELU and fine transfers.
inline constexpr std::array<int, 5> kRepBlocks{1, 4, 8, 12, 16};

// ---------------------------------------------------------------------------
// Tap registry
// ---------------------------------------------------------------------------
enum class TapKind : int {
  kEmbedding = 0,
  kNorm1 = 1,
  kCtx = 2,
  kAttnUpdate = 3,
  kAfterAttn = 4,
  kNorm2 = 5,
  kFfnUpdate = 6,
  kAfterFfn = 7,
  kRelu = 8,
};

inline const char* tapKindName(TapKind kind) {
  switch (kind) {
    case TapKind::kEmbedding: return "EMBEDDING";
    case TapKind::kNorm1: return "NORM1";
    case TapKind::kCtx: return "CTX";
    case TapKind::kAttnUpdate: return "ATT_UPDATE";
    case TapKind::kAfterAttn: return "AFTER_ATTN";
    case TapKind::kNorm2: return "NORM2";
    case TapKind::kFfnUpdate: return "FFN_UPDATE";
    case TapKind::kAfterFfn: return "AFTER_FFN";
    case TapKind::kRelu: return "RELU";
  }
  return "UNKNOWN_TAP";
}

struct TapKey {
  TapKind kind = TapKind::kEmbedding;
  int block = -1;  // 0-based; -1 for embedding
};

inline bool operator<(const TapKey& a, const TapKey& b) {
  if (static_cast<int>(a.kind) != static_cast<int>(b.kind))
    return static_cast<int>(a.kind) < static_cast<int>(b.kind);
  return a.block < b.block;
}

inline bool operator==(const TapKey& a, const TapKey& b) {
  return a.kind == b.kind && a.block == b.block;
}

struct TapSpec {
  int id = -1;
  TapKey key;
  std::string name;
  int dim = 0;
};

// Fixed registry for a depth: EMBEDDING + per-block {NORM1, ATT_UPDATE,
// AFTER_ATTN, NORM2, FFN_UPDATE, AFTER_FFN} + CTX/RELU at representative
// blocks {1,4,8,12,16} + CTX at the last block.
inline std::vector<TapSpec> buildTapRegistry(int depth) {
  std::vector<TapSpec> taps;
  int id = 0;
  taps.push_back(TapSpec{id++, {TapKind::kEmbedding, -1}, "EMBEDDING", 16});
  for (int li = 0; li < depth; ++li) {
    const bool rep = std::find(kRepBlocks.begin(), kRepBlocks.end(), li) !=
                     kRepBlocks.end();
    const bool last = li == depth - 1;
    auto push = [&](TapKind kind, int dim, const std::string& suffix) {
      std::ostringstream name;
      name << "b" << (li < 10 ? "0" : "") << li << "_" << suffix;
      taps.push_back(TapSpec{id++, {kind, li}, name.str(), dim});
    };
    push(TapKind::kNorm1, 16, "NORM1");
    if (rep || last) push(TapKind::kCtx, 16, "CTX");
    push(TapKind::kAttnUpdate, 16, "ATT_UPDATE");
    push(TapKind::kAfterAttn, 16, "AFTER_ATTN");
    push(TapKind::kNorm2, 16, "NORM2");
    push(TapKind::kFfnUpdate, 16, "FFN_UPDATE");
    push(TapKind::kAfterFfn, 16, "AFTER_FFN");
    if (rep) push(TapKind::kRelu, 32, "RELU");
  }
  return taps;
}

// Block input aliases: block 0 input == EMBEDDING; block li input ==
// AFTER_FFN(li-1). These are tensor identities, not separate features.
inline TapKey blockInputKey(int li) {
  return li == 0 ? TapKey{TapKind::kEmbedding, -1}
                 : TapKey{TapKind::kAfterFfn, li - 1};
}

// ---------------------------------------------------------------------------
// TapSet: per-tap features over the 320 teacher-forced rows.
// ---------------------------------------------------------------------------
struct TapSet {
  int depth = 0;
  std::size_t rows = 0;
  std::vector<TapSpec> taps;
  std::vector<std::vector<float>> features;  // [tapId][row*dim + d]

  int tapIndex(const TapKey& key) const {
    for (std::size_t i = 0; i < taps.size(); ++i)
      if (taps[i].key == key) return static_cast<int>(i);
    throw std::invalid_argument("TAP_NOT_IN_REGISTRY");
  }
  bool hasTap(const TapKey& key) const {
    for (const auto& t : taps)
      if (t.key == key) return true;
    return false;
  }
  const TapSpec& tap(int tapId) const {
    return taps.at(static_cast<std::size_t>(tapId));
  }
  // Single-representation view for the reused rp:: probe machinery.
  rp::LayerSet layerSetFor(int tapId) const {
    rp::LayerSet set;
    set.depth = depth;
    set.repCount = 1;
    set.dim = tap(tapId).dim;
    set.features = {features.at(static_cast<std::size_t>(tapId))};
    return set;
  }
};

// One full forward per row (verbatim train::generalForward); taps are read
// from the returned tensors at the last token position. ATT_UPDATE /
// FFN_UPDATE are recomputed with the same mm (double-accumulate -> float)
// that the forward applies internally, so they equal the values that were
// added to the residual stream bit-for-bit.
inline TapSet extractTapFeatures(const tiny::Config& config,
                                 const train::P& params,
                                 const std::vector<rp::ProbeRow>& rows) {
  TapSet set;
  set.depth = static_cast<int>(config.numLayers);
  set.rows = rows.size();
  set.taps = buildTapRegistry(set.depth);
  set.features.clear();
  set.features.reserve(set.taps.size());
  for (const auto& t : set.taps)
    set.features.push_back(
        std::vector<float>(rows.size() * static_cast<std::size_t>(t.dim), 0.0f));
  const std::uint32_t dim = config.dimension;
  const std::uint32_t fdim = config.feedForwardDimension;
  const std::size_t lastRow = dim * (config.tokens - 1);
  for (std::size_t ri = 0; ri < rows.size(); ++ri) {
    const auto oh = tiny::oneHot(rows[ri].context, config.vocabularySize);
    const train::GF g = train::generalForward(config, oh, params);
    for (std::size_t ti = 0; ti < set.taps.size(); ++ti) {
      const TapSpec& t = set.taps[ti];
      float* out = set.features[ti].data() + ri * static_cast<std::size_t>(t.dim);
      switch (t.key.kind) {
        case TapKind::kEmbedding:
          std::memcpy(out, g.embedded.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kNorm1:
          std::memcpy(out, g.layers[static_cast<std::size_t>(t.key.block)].n1.out.data() + lastRow,
                      dim * sizeof(float));
          break;
        case TapKind::kCtx: {
          const auto& ctx = g.layers[static_cast<std::size_t>(t.key.block)].ctx;
          std::memcpy(out, ctx.data() + lastRow, dim * sizeof(float));
          break;
        }
        case TapKind::kAttnUpdate: {
          const auto& z = g.layers[static_cast<std::size_t>(t.key.block)];
          const auto& p = train::layer(params, static_cast<std::uint32_t>(t.key.block));
          const auto upd = train::mm(z.ctx, p.wo, config.tokens, dim, dim);
          std::memcpy(out, upd.data() + lastRow, dim * sizeof(float));
          break;
        }
        case TapKind::kAfterAttn:
          std::memcpy(out, g.layers[static_cast<std::size_t>(t.key.block)].r1.data() + lastRow,
                      dim * sizeof(float));
          break;
        case TapKind::kNorm2:
          std::memcpy(out, g.layers[static_cast<std::size_t>(t.key.block)].n2.out.data() + lastRow,
                      dim * sizeof(float));
          break;
        case TapKind::kFfnUpdate: {
          const auto& z = g.layers[static_cast<std::size_t>(t.key.block)];
          const auto& p = train::layer(params, static_cast<std::uint32_t>(t.key.block));
          const auto upd =
              train::mm(z.relu, p.w2, config.tokens, fdim, dim);
          std::memcpy(out, upd.data() + lastRow, dim * sizeof(float));
          break;
        }
        case TapKind::kAfterFfn:
          std::memcpy(out, g.layers[static_cast<std::size_t>(t.key.block)].out.data() + lastRow,
                      dim * sizeof(float));
          break;
        case TapKind::kRelu: {
          const auto& relu = g.layers[static_cast<std::size_t>(t.key.block)].relu;
          std::memcpy(out, relu.data() + lastRow, fdim * sizeof(float));
          break;
        }
      }
    }
  }
  return set;
}

// Real parameter content hash (fnv1a64 over all float bits), unlike the
// zero-pattern hash used by the readout cache. Strong cache identity.
inline std::string paramContentHash(const train::P& params) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto& e : tiny::parameterRegistry(params)) {
    const std::vector<float>& values = *e.values;
    const float* raw = values.data();
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint32_t bits;
      std::memcpy(&bits, raw + i, sizeof(bits));
      hash ^= bits;
      hash *= 1099511628211ull;
    }
  }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << hash;
  return output.str();
}

// ---------------------------------------------------------------------------
// Tap feature cache (private; never committed). New magic; the tap layout
// differs from the readout cache by design.
// ---------------------------------------------------------------------------
struct TapCacheIdentity {
  std::string protocol;
  std::string config;
  std::uint32_t seed = 0;
  int step = 0;
  std::string datasetHash;
  int depth = 0;
  std::size_t rows = 0;
  std::vector<int> dims;
  std::string contentHash;
};

inline std::string tapCacheFileName(const std::string& config,
                                    std::uint32_t seed, int step,
                                    const std::string& datasetHash) {
  std::ostringstream output;
  output << config << "_s" << seed << "_step" << step << "_"
         << datasetHash.substr(8, 16) << ".bin";
  std::string name = output.str();
  // ':' in a Windows file name silently creates an NTFS alternate data
  // stream (0-byte base file). Keep cache files normal files.
  for (char& ch : name)
    if (ch == ':' || ch == ';' || ch == '=') ch = '_';
  return name;
}

inline bool writeTapCache(const std::filesystem::path& dir,
                          const TapCacheIdentity& identity,
                          const TapSet& set) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return false;
  const auto file =
      dir / tapCacheFileName(identity.config, identity.seed, identity.step,
                             identity.datasetHash);
  std::ofstream out(file, std::ios::binary);
  if (!out) return false;
  const std::uint64_t magic = 0x5441505345543131ull;  // "TAPSET11"
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  char contentHash[33] = {};
  std::memcpy(contentHash, identity.contentHash.c_str(),
              std::min<std::size_t>(32, identity.contentHash.size()));
  out.write(contentHash, 32);
  const std::uint32_t depth = static_cast<std::uint32_t>(identity.depth);
  const std::uint32_t tapCount = static_cast<std::uint32_t>(set.taps.size());
  const std::uint64_t rows = static_cast<std::uint64_t>(identity.rows);
  out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
  out.write(reinterpret_cast<const char*>(&tapCount), sizeof(tapCount));
  out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  for (const int d : identity.dims) {
    const std::uint32_t dd = static_cast<std::uint32_t>(d);
    out.write(reinterpret_cast<const char*>(&dd), sizeof(dd));
  }
  for (const auto& f : set.features)
    out.write(reinterpret_cast<const char*>(f.data()),
              static_cast<std::streamsize>(f.size() * sizeof(float)));
  out.close();
  return static_cast<bool>(out);
}

inline bool readTapCache(const std::filesystem::path& dir,
                         const TapCacheIdentity& identity, TapSet& set) {
  const auto file =
      dir / tapCacheFileName(identity.config, identity.seed, identity.step,
                             identity.datasetHash);
  std::ifstream in(file, std::ios::binary);
  if (!in) return false;
  std::uint64_t magic = 0;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != 0x5441505345543131ull) return false;
  char contentHash[33] = {};
  in.read(contentHash, 32);
  const std::string stored(contentHash);
  if (stored != identity.contentHash) return false;
  std::uint32_t depth = 0, tapCount = 0;
  std::uint64_t rows = 0;
  in.read(reinterpret_cast<char*>(&depth), sizeof(depth));
  in.read(reinterpret_cast<char*>(&tapCount), sizeof(tapCount));
  in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  if (static_cast<int>(depth) != identity.depth ||
      tapCount != identity.dims.size() || rows != identity.rows)
    return false;
  std::vector<int> dims(identity.dims.size(), 0);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    std::uint32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    dims[i] = static_cast<int>(d);
  }
  if (dims != identity.dims) return false;
  set.depth = identity.depth;
  set.rows = static_cast<std::size_t>(rows);
  set.taps = buildTapRegistry(identity.depth);
  if (set.taps.size() != dims.size()) return false;
  set.features.assign(dims.size(),
                      std::vector<float>(static_cast<std::size_t>(rows) * dims[0], 0.0f));
  for (std::size_t i = 0; i < dims.size(); ++i) {
    set.features[i].resize(static_cast<std::size_t>(rows) *
                           static_cast<std::size_t>(dims[i]));
    in.read(reinterpret_cast<char*>(set.features[i].data()),
            static_cast<std::streamsize>(set.features[i].size() *
                                         sizeof(float)));
  }
  if (!in) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Probe training on a tap (identical protocol to READOUT_PROBE_V1; the
// LayerSet view is single-rep so rep index is always 0).
// ---------------------------------------------------------------------------
inline rp::ZStats tapZStats(const TapSet& set, int tapId,
                            std::size_t trainRows) {
  const rp::LayerSet view = set.layerSetFor(tapId);
  return rp::computeZStats(view, 0, trainRows);
}

// ---------------------------------------------------------------------------
// Current-head clone: fold the z-score affine into the head's linear map so
// the probe reproduces the head logits on z-scored features. With no head
// bias (the architecture has none) the fold is exact in double:
//   w[c*D+d] = head_w[d*V+c] * std[d]
//   b[c]     = sum_d head_w[d*V+c] * mean[d]
// Dead dims (std < floor) then contribute head_w*mean via b, exactly as the
// head's own raw dot product.
// ---------------------------------------------------------------------------
inline rp::Probe cloneHeadProbe(const train::P& params, const rp::ZStats& stats,
                                int dim, int classes) {
  rp::Probe probe = rp::zeroProbe(dim);
  for (int c = 0; c < classes; ++c) {
    double bias = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double hw = static_cast<double>(
          params.outputProjection[static_cast<std::size_t>(d) * classes + c]);
      probe.w[static_cast<std::size_t>(c) * dim + d] = hw * stats.std[d];
      bias += hw * stats.mean[d];
    }
    probe.b[static_cast<std::size_t>(c)] = bias;
  }
  return probe;
}

struct CloneParity {
  double maxLogitDelta = 0.0;
  double maxNllDelta = 0.0;
  double maxMarginDelta = 0.0;
  std::uint64_t argmaxFlips = 0;
  std::uint64_t rankFlips = 0;
  std::uint64_t exactMismatch = 0;
  bool pass = false;
};

// Zero-step clone evaluation on development rows: the clone probe evaluated
// on the head-input tap must reproduce the current head's scores within the
// fixed tolerances (float32 head storage vs double probe arithmetic).
inline CloneParity headCloneParity(const tiny::Config& config,
                                   const train::P& params,
                                   const TapSet& set, int tapId,
                                   const std::vector<rp::ProbeRow>& devRows,
                                   std::size_t devBegin) {
  CloneParity result;
  const int dim = static_cast<int>(config.dimension);
  const rp::ZStats stats = tapZStats(set, tapId, kTrainRows);
  const rp::Probe clone =
      cloneHeadProbe(params, stats, dim, static_cast<int>(config.vocabularySize));
  const auto headScores = rp::headRowScores(config, params, devRows);
  const auto probeScores =
      rp::probeRowScores(clone, stats, set.layerSetFor(tapId), 0, devRows,
                         devBegin);
  if (headScores.size() != probeScores.size())
    throw std::runtime_error("CLONE_PARITY_ROW_COUNT");
  bool ok = true;
  for (std::size_t r = 0; r < headScores.size(); ++r) {
    const auto& h = headScores[r];
    const auto& p = probeScores[r];
    if (!h.valid || !p.valid) {
      ok = false;
      continue;
    }
    // Logits are not part of ma::Score; recompute both logit rows.
    std::vector<double> hLogits(config.vocabularySize);
    {
      const auto oh = tiny::oneHot(devRows[r].context, config.vocabularySize);
      const train::GF g = train::generalForward(config, oh, params);
      const std::size_t base = std::size_t(config.tokens - 1) * config.vocabularySize;
      for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
        hLogits[j] = static_cast<double>(g.logits[base + j]);
    }
    const auto& f = set.features[static_cast<std::size_t>(tapId)];
    std::vector<double> pLogits(probeScores[r].valid ? config.vocabularySize : 0);
    if (p.valid) {
      rp::probeForward(clone, stats,
                       f.data() + (devBegin + r) * static_cast<std::size_t>(dim),
                       pLogits.data());
    }
    double md = 0.0;
    for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
      md = std::max(md, std::abs(hLogits[j] - pLogits[j]));
    result.maxLogitDelta = std::max(result.maxLogitDelta, md);
    result.maxNllDelta =
        std::max(result.maxNllDelta, std::abs(h.tokenNll - p.tokenNll));
    result.maxMarginDelta = std::max(
        result.maxMarginDelta,
        std::abs(h.expectedMinusTop1Margin - p.expectedMinusTop1Margin));
    if (h.predicted != p.predicted) {
      ++result.argmaxFlips;
      ok = false;
    }
    if (std::abs(h.expectedRank - p.expectedRank) > 1e-9) {
      ++result.rankFlips;
      ok = false;
    }
    if (h.predicted != devRows[r].truth && p.predicted == devRows[r].truth)
      ++result.exactMismatch;
    if (h.predicted == devRows[r].truth && p.predicted != devRows[r].truth)
      ++result.exactMismatch;
    if (md > kParityLogitTol) ok = false;
    if (std::abs(h.tokenNll - p.tokenNll) > kParityNllTol) ok = false;
    if (std::abs(h.expectedMinusTop1Margin - p.expectedMinusTop1Margin) >
        kParityMarginTol)
      ok = false;
  }
  result.pass = ok;
  return result;
}

// ---------------------------------------------------------------------------
// Cross-tap transfer: apply a probe trained on tap A to tap B features.
//   raw  = A's z-stats on B's features (strict coordinate change test)
//   norm = B's z-stats (per-dim affine absorbed; probes the non-diagonal
//          linear gap + information loss)
// ---------------------------------------------------------------------------
inline rp::TokenMetrics transferEval(const rp::Probe& probe,
                                     const rp::ZStats& stats,
                                     const TapSet& set, int tapId,
                                     const std::vector<rp::ProbeRow>& rows,
                                     std::size_t rowOffset) {
  return rp::probeTokenMetrics(probe, stats, set.layerSetFor(tapId), 0, rows,
                               rowOffset);
}

// ---------------------------------------------------------------------------
// Deterministic linear algebra (double, fixed sweeps).
// ---------------------------------------------------------------------------
struct SymEig {
  std::vector<double> values;                  // descending
  std::vector<std::vector<double>> vectors;    // d x d, column j = eigenvector of values[j]
};

// Cyclic Jacobi on a symmetric matrix with a fixed sweep count (same flow
// as rp::symmetricEigenvalues but with eigenvectors accumulated).
inline SymEig symmetricEigen(const std::vector<std::vector<double>>& matrix) {
  const int n = static_cast<int>(matrix.size());
  std::vector<std::vector<double>> a = matrix;
  std::vector<std::vector<double>> v(
      static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
  for (int sweep = 0; sweep < kAlignSweeps; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n - 1; ++p)
      for (int q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        if (std::abs(a[p][q]) < 1e-18) continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = std::copysign(1.0, theta) /
                         (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        // Symmetric update A' = J^T A J (columns/rows p,q), eigenvectors V' = V J.
        for (int k = 0; k < n; ++k) {
          if (k == p || k == q) continue;
          const double akp = a[k][p];
          const double akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p];
          a[q][k] = a[k][q];
        }
        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (int k = 0; k < n; ++k) {
          const double vkp = v[k][p];
          const double vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  SymEig result;
  result.values.resize(static_cast<std::size_t>(n));
  result.vectors.resize(static_cast<std::size_t>(n));
  for (auto& row : result.vectors)
    row.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double val = a[i][i];
    if (std::abs(val) < 1e-15) val = 0.0;
    result.values[static_cast<std::size_t>(i)] = val;
  }
  // Sort descending and carry the vectors.
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::vector<double> raw(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double val = a[i][i];
    if (std::abs(val) < 1e-15) val = 0.0;
    raw[static_cast<std::size_t>(i)] = val;
  }
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    return raw[static_cast<std::size_t>(x)] >
           raw[static_cast<std::size_t>(y)];
  });
  for (int j = 0; j < n; ++j) {
    const int src = order[static_cast<std::size_t>(j)];
    result.values[static_cast<std::size_t>(j)] = raw[static_cast<std::size_t>(src)];
    for (int k = 0; k < n; ++k)
      result.vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
          v[static_cast<std::size_t>(k)][static_cast<std::size_t>(src)];
  }
  return result;
}

// Moore-Penrose pseudo-inverse of a symmetric PSD matrix via its
// eigendecomposition with a fixed relative cutoff.
inline std::vector<std::vector<double>> symmetricPinv(
    const std::vector<std::vector<double>>& g, double rcond) {
  const int n = static_cast<int>(g.size());
  const SymEig eig = symmetricEigen(g);
  double maxV = 0.0;
  for (const double x : eig.values) maxV = std::max(maxV, x);
  const double cutoff = maxV > 0.0 ? rcond * maxV : 0.0;
  std::vector<std::vector<double>> pin(
      static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j) {
    if (eig.values[static_cast<std::size_t>(j)] <= cutoff) continue;
    const double inv = 1.0 / eig.values[static_cast<std::size_t>(j)];
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b)
        pin[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            inv * eig.vectors[static_cast<std::size_t>(a)][static_cast<std::size_t>(j)] *
            eig.vectors[static_cast<std::size_t>(b)][static_cast<std::size_t>(j)];
  }
  return pin;
}

struct AlignResult {
  bool finite = true;
  std::vector<std::vector<double>> mapLs;    // d x d: z_B ~= z_A * mapLs (LS)
  std::vector<std::vector<double>> mapOrth;  // d x d: z_B ~= z_A * mapOrth (Procrustes)
  std::vector<double> sv;                    // singular values of Z_A (desc)
  int fitRank = 0;
  double relResidualLs = 0.0;
  double relResidualOrth = 0.0;
  double cond = 0.0;  // condition number of Z_A^T Z_A (with floor)
};

// Train-row features of a tap as a rows x dim double matrix.
inline std::vector<std::vector<double>> tapTrainFeatures(
    const TapSet& set, int tapId, std::size_t trainBegin,
    std::size_t trainEnd) {
  const int dim = set.tap(tapId).dim;
  const auto& f = set.features[static_cast<std::size_t>(tapId)];
  std::vector<std::vector<double>> m(
      trainEnd - trainBegin, std::vector<double>(static_cast<std::size_t>(dim), 0.0));
  for (std::size_t r = trainBegin; r < trainEnd; ++r)
    for (int d = 0; d < dim; ++d)
      m[r - trainBegin][static_cast<std::size_t>(d)] =
          static_cast<double>(f[r * static_cast<std::size_t>(dim) + d]);
  return m;
}

// Fit both maps on TRAIN rows only. mapLs: Z_B ~= Z_A mapLs via
// mapLs = pinv(Z_A^T Z_A) Z_A^T Z_B. mapOrth: Procrustes rotation X = U V^T
// from the SVD of Z_A^T Z_B (via the eigendecomposition of its Gram matrices,
// fixed sweeps), z_B ~= z_A mapOrth. All double, deterministic.
inline AlignResult fitTapAlignment(const TapSet& set, int tapA, int tapB,
                                   std::size_t trainBegin,
                                   std::size_t trainEnd) {
  AlignResult result;
  const int d = set.tap(tapB).dim;
  if (set.tap(tapA).dim != d) throw std::invalid_argument("ALIGN_DIM_MISMATCH");
  const std::vector<std::vector<double>> za =
      tapTrainFeatures(set, tapA, trainBegin, trainEnd);
  const std::vector<std::vector<double>> zb =
      tapTrainFeatures(set, tapB, trainBegin, trainEnd);
  const std::size_t n = za.size();
  // G_a = Z_A^T Z_A, G_ab = Z_A^T Z_B, G_b = Z_B^T Z_B.
  std::vector<std::vector<double>> ga(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  std::vector<std::vector<double>> gab(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  std::vector<std::vector<double>> gb(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  for (std::size_t r = 0; r < n; ++r)
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        ga[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            za[r][static_cast<std::size_t>(a)] * za[r][static_cast<std::size_t>(b)];
        gab[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            za[r][static_cast<std::size_t>(a)] * zb[r][static_cast<std::size_t>(b)];
        gb[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            zb[r][static_cast<std::size_t>(a)] * zb[r][static_cast<std::size_t>(b)];
      }
  const std::vector<std::vector<double>> pinA = symmetricPinv(ga, kAlignRcond);
  // mapLs = pinv(Z_A^T Z_A) Z_A^T Z_B so that predLs[b] = sum_a z_A[a]*mapLs[a][b]
  // is the LS prediction of z_B (z_B ~= z_A * mapLs).
  result.mapLs.assign(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += pinA[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] *
               gab[static_cast<std::size_t>(k)][static_cast<std::size_t>(b)];
      result.mapLs[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = acc;
    }
  // Singular values of Z_A from eig(ga).
  const SymEig eigA = symmetricEigen(ga);
  double maxV = 0.0;
  for (const double x : eigA.values) maxV = std::max(maxV, x);
  result.cond = maxV > 0.0 && eigA.values.back() > 0.0
                    ? maxV / eigA.values.back()
                    : std::numeric_limits<double>::infinity();
  result.fitRank = 0;
  result.sv.clear();
  for (const double x : eigA.values) {
    if (x > kAlignRcond * maxV) {
      ++result.fitRank;
      result.sv.push_back(std::sqrt(std::max(0.0, x)));
    } else {
      result.sv.push_back(0.0);
    }
  }
  // Procrustes: SVD of M = Z_B^T Z_A = gab^T; V from eig(M^T M) = eig(gab gab^T)
  // (gab here is Z_A^T Z_B). With gab = U0 S0 V0^T, eig vectors of M^T M are
  // U0 and u = M V S^{-1} = V0; the minimizer of ||Z_A X - Z_B|| is X = U0 V0^T,
  // so mapOrth[a][b] = sum_k V_eigM[a][k] * u[b][k] (z_B ~= z_A * mapOrth).
  std::vector<std::vector<double>> mtm(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += gab[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] *
               gab[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)];
      mtm[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = acc;
    }
  const SymEig eigM = symmetricEigen(mtm);
  // U = M V diag(1/sigma); R = U V^T.
  std::vector<std::vector<double>> u(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  result.mapOrth.assign(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  double sMax = 0.0;
  for (const double x : eigM.values) sMax = std::max(sMax, x);
  for (int j = 0; j < d; ++j) {
    const double s = std::sqrt(std::max(0.0, eigM.values[static_cast<std::size_t>(j)]));
    if (s <= kAlignRcond * std::max(sMax, 1e-300)) continue;
    for (int a = 0; a < d; ++a) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += gab[static_cast<std::size_t>(k)][static_cast<std::size_t>(a)] *
               eigM.vectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
      u[static_cast<std::size_t>(a)][static_cast<std::size_t>(j)] = acc / s;
    }
  }
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += eigM.vectors[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] *
               u[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)];
      result.mapOrth[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = acc;
    }
  // Relative residuals on TRAIN rows.
  double nrmZb = 0.0, resLs = 0.0, resOrth = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    for (int b = 0; b < d; ++b) {
      nrmZb += zb[r][static_cast<std::size_t>(b)] * zb[r][static_cast<std::size_t>(b)];
      double predLs = 0.0, predOrth = 0.0;
      for (int a = 0; a < d; ++a) {
        predLs += za[r][static_cast<std::size_t>(a)] *
                  result.mapLs[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)];
        predOrth += za[r][static_cast<std::size_t>(a)] *
                    result.mapOrth[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)];
      }
      const double eLs = predLs - zb[r][static_cast<std::size_t>(b)];
      const double eOrth = predOrth - zb[r][static_cast<std::size_t>(b)];
      resLs += eLs * eLs;
      resOrth += eOrth * eOrth;
    }
  }
  result.relResidualLs = nrmZb > 0.0 ? std::sqrt(resLs / nrmZb) : 0.0;
  result.relResidualOrth = nrmZb > 0.0 ? std::sqrt(resOrth / nrmZb) : 0.0;
  for (const auto& row : result.mapLs)
    for (const double y : row)
      if (!std::isfinite(y)) result.finite = false;
  for (const auto& row : result.mapOrth)
    for (const double y : row)
      if (!std::isfinite(y)) result.finite = false;
  return result;
}

// Function-level least-squares alignment: w_c' = pinv(Z_B^T Z_B) Z_B^T Z_A
// w_{A,c} (min-norm; no matrix inversion of the source map). Returns a probe
// in tap-B coordinates with tap-A's bias.
inline rp::Probe alignProbeFunctionLevel(const rp::Probe& probeA,
                                         const TapSet& set, int tapA,
                                         int tapB, std::size_t trainBegin,
                                         std::size_t trainEnd) {
  const int d = set.tap(tapB).dim;
  const std::vector<std::vector<double>> za =
      tapTrainFeatures(set, tapA, trainBegin, trainEnd);
  const std::vector<std::vector<double>> zb =
      tapTrainFeatures(set, tapB, trainBegin, trainEnd);
  std::vector<std::vector<double>> gb(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  std::vector<std::vector<double>> gba(
      static_cast<std::size_t>(d), std::vector<double>(static_cast<std::size_t>(d), 0.0));
  for (std::size_t r = 0; r < zb.size(); ++r)
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        gb[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            zb[r][static_cast<std::size_t>(a)] * zb[r][static_cast<std::size_t>(b)];
        gba[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] +=
            zb[r][static_cast<std::size_t>(a)] * za[r][static_cast<std::size_t>(b)];
      }
  const std::vector<std::vector<double>> pinB = symmetricPinv(gb, kAlignRcond);
  rp::Probe out = probeA;
  out.dim = d;
  out.w.assign(static_cast<std::size_t>(probeA.classes) * static_cast<std::size_t>(d), 0.0);
  for (int c = 0; c < probeA.classes; ++c) {
    for (int b = 0; b < d; ++b) {
      double acc = 0.0;
      for (int a = 0; a < d; ++a) {
        double inner = 0.0;
        for (int k = 0; k < d; ++k)
          inner += pinB[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] *
                   gba[static_cast<std::size_t>(k)][static_cast<std::size_t>(b)];
        acc += inner * probeA.w[static_cast<std::size_t>(c) * d + a];
      }
      out.w[static_cast<std::size_t>(c) * d + b] = acc;
    }
  }
  return out;
}

// Procrustes-aligned probe: w_B = R w_A where R = mapOrth = U V^T maps
// z_B-coordinates back to z_A (z_B ~= z_A R^T); bias kept.
inline rp::Probe alignProbeProcrustes(const rp::Probe& probeA,
                                      const AlignResult& al) {
  const int d = static_cast<int>(al.mapOrth.size());
  rp::Probe out = probeA;
  out.dim = d;
  out.w.assign(static_cast<std::size_t>(probeA.classes) * static_cast<std::size_t>(d), 0.0);
  for (int c = 0; c < probeA.classes; ++c)
    for (int a = 0; a < d; ++a) {
      double acc = 0.0;
      for (int b = 0; b < d; ++b)
        acc += al.mapOrth[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
               probeA.w[static_cast<std::size_t>(c) * d + b];
      out.w[static_cast<std::size_t>(c) * d + a] = acc;
    }
  return out;
}

// ---------------------------------------------------------------------------
// Block geometry: residual/update norms, ratios, cosines (dev rows).
// ---------------------------------------------------------------------------
struct BlockGeometry {
  int block = 0;
  double residualNorm = 0.0;    // mean ||block_input||
  double attnUpdateNorm = 0.0;  // mean ||ATT_UPDATE||
  double attnRatio = 0.0;       // attnUpdateNorm / residualNorm
  double cosAttn = 0.0;         // mean cos(block_input, ATT_UPDATE)
  double afterAttnNorm = 0.0;   // mean ||AFTER_ATTN||
  double ffnUpdateNorm = 0.0;   // mean ||FFN_UPDATE||
  double ffnRatio = 0.0;        // ffnUpdateNorm / afterAttnNorm
  double cosFfn = 0.0;          // mean cos(after_attn, FFN_UPDATE)
  double afterFfnNorm = 0.0;    // mean ||AFTER_FFN||
};

inline double rowNorm(const float* f, int dim) {
  double sq = 0.0;
  for (int d = 0; d < dim; ++d) sq += static_cast<double>(f[d]) * f[d];
  return std::sqrt(sq);
}

inline double rowDot(const float* a, const float* b, int dim) {
  double acc = 0.0;
  for (int d = 0; d < dim; ++d)
    acc += static_cast<double>(a[d]) * static_cast<double>(b[d]);
  return acc;
}

inline std::vector<BlockGeometry> blockGeometry(
    const TapSet& set, std::size_t rowBegin, std::size_t rowEnd) {
  const int dim = 16;
  std::vector<BlockGeometry> out;
  for (int li = 0; li < set.depth; ++li) {
    BlockGeometry g;
    g.block = li;
    const TapKey inKey = blockInputKey(li);
    const int inId = set.tapIndex(inKey);
    const int attnId = set.tapIndex({TapKind::kAttnUpdate, li});
    const int afterAttnId = set.tapIndex({TapKind::kAfterAttn, li});
    const int ffnId = set.tapIndex({TapKind::kFfnUpdate, li});
    const int afterFfnId = set.tapIndex({TapKind::kAfterFfn, li});
    const auto& fin = set.features[static_cast<std::size_t>(inId)];
    const auto& fattn = set.features[static_cast<std::size_t>(attnId)];
    const auto& faa = set.features[static_cast<std::size_t>(afterAttnId)];
    const auto& fffn = set.features[static_cast<std::size_t>(ffnId)];
    const auto& faf = set.features[static_cast<std::size_t>(afterFfnId)];
    const std::size_t n = rowEnd - rowBegin;
    double sumRes = 0.0, sumAttn = 0.0, sumAfterAttn = 0.0, sumFfn = 0.0,
           sumAfterFfn = 0.0, sumCosA = 0.0, sumCosF = 0.0;
    std::size_t cosAn = 0, cosFn = 0;
    for (std::size_t r = rowBegin; r < rowEnd; ++r) {
      const float* x = fin.data() + r * dim;
      const float* a = fattn.data() + r * dim;
      const float* aa = faa.data() + r * dim;
      const float* f = fffn.data() + r * dim;
      const float* af = faf.data() + r * dim;
      const double nx = rowNorm(x, dim), na = rowNorm(a, dim),
                   naa = rowNorm(aa, dim), nf = rowNorm(f, dim),
                   naf = rowNorm(af, dim);
      sumRes += nx;
      sumAttn += na;
      sumAfterAttn += naa;
      sumFfn += nf;
      sumAfterFfn += naf;
      if (nx > 0.0 && na > 0.0) {
        sumCosA += rowDot(x, a, dim) / (nx * na);
        ++cosAn;
      }
      if (naa > 0.0 && nf > 0.0) {
        sumCosF += rowDot(aa, f, dim) / (naa * nf);
        ++cosFn;
      }
    }
    g.residualNorm = n > 0 ? sumRes / n : 0.0;
    g.attnUpdateNorm = n > 0 ? sumAttn / n : 0.0;
    g.attnRatio =
        g.residualNorm > 0.0 ? g.attnUpdateNorm / g.residualNorm : 0.0;
    g.cosAttn = cosAn > 0 ? sumCosA / cosAn : 0.0;
    g.afterAttnNorm = n > 0 ? sumAfterAttn / n : 0.0;
    g.ffnUpdateNorm = n > 0 ? sumFfn / n : 0.0;
    g.ffnRatio =
        g.afterAttnNorm > 0.0 ? g.ffnUpdateNorm / g.afterAttnNorm : 0.0;
    g.cosFfn = cosFn > 0 ? sumCosF / cosFn : 0.0;
    g.afterFfnNorm = n > 0 ? sumAfterFfn / n : 0.0;
    out.push_back(g);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Auxiliary metrics per tap (dev separability + train geometry).
// ---------------------------------------------------------------------------
struct TapAux {
  int tapId = -1;
  double eta2 = 0.0;               // dev rows, class separability
  double effectiveRank = 0.0;      // dev rows, participation ratio
  double betweenWithin = 0.0;      // dev rows, tr(SB)/tr(SW)
  double meanPairwiseCosine = 0.0; // dev rows
  double cond = 0.0;               // TRAIN rows, Z^T Z condition number
};

inline TapAux tapAuxMetrics(const TapSet& set, int tapId,
                            const std::vector<rp::ProbeRow>& devRows,
                            std::size_t devBegin, std::size_t devEnd,
                            std::size_t trainBegin, std::size_t trainEnd) {
  TapAux result;
  result.tapId = tapId;
  const int dim = set.tap(tapId).dim;
  const auto& f = set.features[static_cast<std::size_t>(tapId)];

  // Class means on dev rows (needed for eta2 / between-within).
  std::vector<std::vector<double>> means(
      rp::kClasses, std::vector<double>(static_cast<std::size_t>(dim), 0.0));
  std::vector<std::size_t> counts(rp::kClasses, 0);
  std::vector<double> global(static_cast<std::size_t>(dim), 0.0);
  for (std::size_t r = devBegin; r < devEnd; ++r) {
    const std::uint32_t cls = devRows[r - devBegin].truth;
    ++counts[cls];
    for (int d = 0; d < dim; ++d) {
      const double x = static_cast<double>(f[r * static_cast<std::size_t>(dim) + d]);
      means[cls][static_cast<std::size_t>(d)] += x;
      global[static_cast<std::size_t>(d)] += x;
    }
  }
  const std::size_t n = devEnd - devBegin;
  for (int d = 0; d < dim; ++d) global[static_cast<std::size_t>(d)] /= static_cast<double>(n);
  for (std::uint32_t c = 0; c < rp::kClasses; ++c)
    if (counts[c] > 0)
      for (int d = 0; d < dim; ++d)
        means[c][static_cast<std::size_t>(d)] /= static_cast<double>(counts[c]);

  double trSt = 0.0, trSb = 0.0, trSw = 0.0;
  for (std::size_t r = devBegin; r < devEnd; ++r) {
    const std::uint32_t cls = devRows[r - devBegin].truth;
    for (int d = 0; d < dim; ++d) {
      const double x = static_cast<double>(f[r * static_cast<std::size_t>(dim) + d]);
      const double zt = x - global[static_cast<std::size_t>(d)];
      trSt += zt * zt;
      const double zw = x - means[cls][static_cast<std::size_t>(d)];
      trSw += zw * zw;
    }
  }
  for (std::uint32_t c = 0; c < rp::kClasses; ++c) {
    if (counts[c] == 0) continue;
    for (int d = 0; d < dim; ++d) {
      const double zb = means[c][static_cast<std::size_t>(d)] -
                        global[static_cast<std::size_t>(d)];
      trSb += static_cast<double>(counts[c]) * zb * zb;
    }
  }
  result.eta2 = trSt > 0.0 ? trSb / trSt : 0.0;
  result.betweenWithin = trSw > 0.0 ? trSb / trSw : 0.0;

  // Effective rank from TRAIN covariance eigenvalues (participation ratio).
  std::vector<std::vector<double>> st(
      static_cast<std::size_t>(dim), std::vector<double>(static_cast<std::size_t>(dim), 0.0));
  std::vector<double> tmean(static_cast<std::size_t>(dim), 0.0);
  for (std::size_t r = trainBegin; r < trainEnd; ++r)
    for (int d = 0; d < dim; ++d)
      tmean[static_cast<std::size_t>(d)] +=
          static_cast<double>(f[r * static_cast<std::size_t>(dim) + d]);
  const std::size_t tn = trainEnd - trainBegin;
  for (int d = 0; d < dim; ++d) tmean[static_cast<std::size_t>(d)] /= static_cast<double>(tn);
  for (std::size_t r = trainBegin; r < trainEnd; ++r)
    for (int a = 0; a < dim; ++a)
      for (int b = 0; b < dim; ++b) {
        const double za = static_cast<double>(f[r * static_cast<std::size_t>(dim) + a]) -
                          tmean[static_cast<std::size_t>(a)];
        const double zb = static_cast<double>(f[r * static_cast<std::size_t>(dim) + b]) -
                          tmean[static_cast<std::size_t>(b)];
        st[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] += za * zb;
      }
  const SymEig eig = symmetricEigen(st);
  double sum = 0.0, sq = 0.0, maxV = 0.0;
  for (const double x : eig.values) {
    sum += x;
    sq += x * x;
    maxV = std::max(maxV, x);
  }
  result.effectiveRank = sq > 0.0 ? sum * sum / sq : 0.0;
  result.cond = maxV > 0.0 && eig.values.back() > 0.0
                    ? maxV / eig.values.back()
                    : std::numeric_limits<double>::infinity();

  // Mean pairwise cosine over dev rows (full 144 x 144/2 pairs).
  std::vector<double> norms(n, 0.0);
  for (std::size_t r = 0; r < n; ++r) norms[r] = rowNorm(f.data() + (devBegin + r) * dim, dim);
  double cosSum = 0.0;
  std::size_t cosN = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (norms[i] <= 0.0) continue;
    for (std::size_t j = i + 1; j < n; ++j) {
      if (norms[j] <= 0.0) continue;
      cosSum += rowDot(f.data() + (devBegin + i) * dim,
                       f.data() + (devBegin + j) * dim, dim) /
                (norms[i] * norms[j]);
      ++cosN;
    }
  }
  result.meanPairwiseCosine = cosN > 0 ? cosSum / cosN : 0.0;
  return result;
}

// ---------------------------------------------------------------------------
// Token-only baselines (no model forward). Learned from TRAIN rows only.
//   A: key = current token (context.back())
//   B: key = (current token, row index within its case)
//   C: key = (context[-2], context[-1])
// Prediction = argmax over observed next tokens (tie-break smallest id);
// unseen keys fall back to the globally most frequent next token.
// ---------------------------------------------------------------------------
struct BaselineResult {
  std::string baseline;
  std::string dataset;
  std::uint64_t total = 0;
  std::uint64_t seen = 0;
  std::uint64_t unseen = 0;
  std::uint64_t correct = 0;
  std::uint64_t correctSeen = 0;
  std::uint64_t correctUnseen = 0;
  std::uint64_t fallbackUsed = 0;
  std::uint64_t uniqueKeys = 0;
};

inline std::string baselineKey(const std::vector<rp::ProbeRow>& rows,
                               std::size_t r, const char* kind) {
  const auto& row = rows[r];
  std::ostringstream out;
  switch (kind[0]) {
    case 'A':
      out << row.context.back();
      break;
    case 'B':
      out << row.context.back() << "," << (r % 8);
      break;
    case 'C':
      out << row.context[row.context.size() - 2] << ","
          << row.context.back();
      break;
  }
  return out.str();
}

inline std::vector<BaselineResult> tokenBaselines(
    const std::vector<rp::ProbeRow>& trainRows,
    const std::vector<rp::ProbeRow>& calRows,
    const std::vector<rp::ProbeRow>& devRows) {
  std::vector<BaselineResult> results;
  for (const char* kind : {"A", "B", "C"}) {
    std::map<std::string, std::map<std::uint32_t, std::uint64_t>> table;
    for (std::size_t r = 0; r < trainRows.size(); ++r)
      ++table[baselineKey(trainRows, r, kind)][trainRows[r].truth];
    // Global fallback: most frequent next token (tie-break smallest id).
    std::map<std::uint32_t, std::uint64_t> globalCounts;
    for (std::size_t r = 0; r < trainRows.size(); ++r)
      ++globalCounts[trainRows[r].truth];
    std::uint32_t fallback = 0;
    std::uint64_t bestCount = 0;
    for (const auto& entry : globalCounts)
      if (entry.second > bestCount) {
        bestCount = entry.second;
        fallback = entry.first;
      }
    const std::array<const std::vector<rp::ProbeRow>*, 3> datasets = {
        &trainRows, &calRows, &devRows};
    const std::array<const char*, 3> datasetNames = {"TRAIN", "CAL",
                                                     "DEV"};
    for (std::size_t di = 0; di < datasets.size(); ++di) {
      BaselineResult result;
      result.baseline = std::string("BASELINE_") + kind;
      result.dataset = datasetNames[di];
      const auto& rows = *datasets[di];
      result.total = rows.size();
      for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::string key = baselineKey(rows, r, kind);
        const auto it = table.find(key);
        std::uint32_t pred = fallback;
        if (it == table.end()) {
          ++result.unseen;
          ++result.fallbackUsed;
        } else {
          ++result.seen;
          std::uint64_t best = 0;
          for (const auto& entry : it->second)
            if (entry.second > best) {
              best = entry.second;
              pred = entry.first;
            }
        }
        if (pred == rows[r].truth) {
          ++result.correct;
          if (it == table.end())
            ++result.correctUnseen;
          else
            ++result.correctSeen;
        }
      }
      result.uniqueKeys = table.size();
      results.push_back(result);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Free-running scoring on a tap (secondary evidence). The context slides
// with the probe's own argmax; every position re-runs the verbatim forward
// and scores the tap feature at the last token position.
// ---------------------------------------------------------------------------
struct TapScorer {
  const tiny::Config* config = nullptr;
  const train::P* params = nullptr;
  rp::Probe probe;
  rp::ZStats stats;
  TapKey key;

  inline ma::Score operator()(const std::vector<std::uint32_t>& context,
                              std::uint32_t truth) const {
    if (!config || !params) throw std::invalid_argument("TAP_SCORER_UNSET");
    const auto oh = tiny::oneHot(context, config->vocabularySize);
    const train::GF g = train::generalForward(*config, oh, *params);
    const std::uint32_t dim = config->dimension;
    const std::uint32_t fdim = config->feedForwardDimension;
    const std::size_t lastRow = std::size_t(config->tokens - 1) * dim;
    std::vector<float> buf(static_cast<std::size_t>(std::max(dim, fdim)), 0.0f);
    const float* feature = nullptr;
    switch (key.kind) {
      case TapKind::kEmbedding:
        feature = g.embedded.data() + lastRow;
        break;
      case TapKind::kNorm1:
        feature = g.layers[static_cast<std::size_t>(key.block)].n1.out.data() + lastRow;
        break;
      case TapKind::kCtx:
        feature = g.layers[static_cast<std::size_t>(key.block)].ctx.data() + lastRow;
        break;
      case TapKind::kAttnUpdate: {
        const auto& p = train::layer(*params, static_cast<std::uint32_t>(key.block));
        const auto upd = train::mm(g.layers[static_cast<std::size_t>(key.block)].ctx, p.wo,
                                   config->tokens, dim, dim);
        std::memcpy(buf.data(), upd.data() + lastRow, dim * sizeof(float));
        feature = buf.data();
        break;
      }
      case TapKind::kAfterAttn:
        feature = g.layers[static_cast<std::size_t>(key.block)].r1.data() + lastRow;
        break;
      case TapKind::kNorm2:
        feature = g.layers[static_cast<std::size_t>(key.block)].n2.out.data() + lastRow;
        break;
      case TapKind::kFfnUpdate: {
        const auto& p = train::layer(*params, static_cast<std::uint32_t>(key.block));
        const auto upd = train::mm(g.layers[static_cast<std::size_t>(key.block)].relu, p.w2,
                                   config->tokens, fdim, dim);
        std::memcpy(buf.data(), upd.data() + lastRow, dim * sizeof(float));
        feature = buf.data();
        break;
      }
      case TapKind::kAfterFfn:
        feature = g.layers[static_cast<std::size_t>(key.block)].out.data() + lastRow;
        break;
      case TapKind::kRelu:
        feature = g.layers[static_cast<std::size_t>(key.block)].relu.data() + lastRow;
        break;
    }
    std::vector<double> logits(probe.classes);
    rp::probeForward(probe, stats, feature, logits.data());
    if (!std::isfinite(logits[truth]))
      throw std::runtime_error("NON_FINITE_TAP_PROBE_LOGITS");
    const auto p = rp::softmaxRow(logits.data(), probe.classes);
    return rp::stableScoreFromLogits(logits, p, truth);
  }
};

inline cm::CheckpointMetrics tapFreeRunning(
    const std::vector<ar::Case>& cases, int step, const TapScorer& scorer) {
  return rp::freeRunningRollout(cases, step, scorer);
}

// ---------------------------------------------------------------------------
// Diagnosis helpers (fixed thresholds above; never tuned).
// ---------------------------------------------------------------------------
inline int tapDevTfExact(const rp::TokenMetrics& m) {
  return static_cast<int>(m.tokenExact);
}

}  // namespace phonelm::intra_block_readability

#endif  // INTRA_BLOCK_READABILITY_LIB_H
