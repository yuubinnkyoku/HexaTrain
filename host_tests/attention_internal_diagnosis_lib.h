// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only attention-internal diagnosis for the L19 transformer
// (ATTENTION_INTERNAL_V1 protocol, fixed before results; see
// build/private-diagnostics/attention-internal-goal/protocol.json).
//
// The forward pass is the VERBATIM host copy train::generalForward
// (critical_margin_training_lib.h, itself a bitwise copy of
// app/src/main/cpp/tiny_language_model_cpu.cpp). This library adds an
// INTERVENTION forward that reproduces the same arithmetic but keeps each
// head's context in a separate buffer so head-level interventions (head
// zero, head only, cross-seed context swap, attention-weight/value swap,
// head pair) can be applied before the concat and output projection. With an
// empty intervention the intervened forward must be bitwise identical to
// train::generalForward (asserted in the self-test). No production code is
// touched and the existing CPU results stay bitwise invariant.
//
// Protocol: ATTENTION_INTERNAL_V1 (fixed before results). Tap kinds per
// target layer:
//   NORM1        = LN1 output            (dim 16)  [reused from intra-block]
//   Q            = Q projection          (dim 16)
//   K            = K projection          (dim 16)
//   V            = V projection          (dim 16)
//   CTX_H0       = head 0 context        (dim 8)
//   CTX_H1       = head 1 context        (dim 8)
//   CTX_CONCAT   = concat of head ctx    (dim 16)
//   ATT_UPDATE   = ctx @ Wo              (dim 16)  [reused from intra-block]
//   AFTER_ATTN   = residual after attn   (dim 16)  [reused from intra-block]
// NORM1/ATT_UPDATE/AFTER_ATTN probes are reused from the INTRA_BLOCK_READABILITY_V1
// run under paramHash identity; recomputed only if cache identity fails.
#ifndef ATTENTION_INTERNAL_DIAGNOSIS_LIB_H
#define ATTENTION_INTERNAL_DIAGNOSIS_LIB_H

#include "readout_probe_lib.h"
#include "critical_margin_training_lib.h"
#include "intra_block_readability_lib.h"

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::attention_internal {

namespace train = phonelm::critical_margin::train;
namespace rp = phonelm::readout_probe;
namespace tiny = phonelm::tiny_lm;
namespace ma = phonelm::margin_analysis;
namespace cm = phonelm::critical_margin;
namespace dq = phonelm::depth_quality;
namespace ar = phonelm::autoregressive_validation;
namespace ibr = phonelm::intra_block_readability;

inline constexpr const char* kProtocolId = "ATTENTION_INTERNAL_V1";
// Row layout (identical to READOUT_PROBE_V1 / INTRA_BLOCK_READABILITY_V1).
inline constexpr std::size_t kTrainRows = 32;
inline constexpr std::size_t kCalBegin = 32;
inline constexpr std::size_t kCalRows = 144;
inline constexpr std::size_t kDevBegin = 176;
inline constexpr std::size_t kDevRows = 144;
inline constexpr std::size_t kTotalRows = 320;
// Fixed diagnostic thresholds (never tuned after results).
inline constexpr int kDropTokens = 5;  // of 144 dev tokens
inline constexpr double kParityLogitTol = 1.0e-4;
inline constexpr double kContributionSumTol = 1.0e-4;  // relative
// Target layers (0-based) fixed before results (see protocol.json).
inline constexpr std::array<int, 13> kTargetLayersL19{1, 4, 7, 8, 9, 11, 12,
                                                      13, 14, 15, 16, 17, 18};
inline constexpr std::array<int, 12> kTargetLayersL18{1, 4, 6, 8, 10, 11, 12,
                                                      13, 14, 15, 16, 17};
// Deep band (0-based) for head-only / swap priority.
inline constexpr int kDeepBandFirstL19 = 11;
inline constexpr int kDeepBandFirstL18 = 10;
// Max-drop layers per config (0-based, from INTRA_BLOCK_READABILITY_V1).
inline constexpr int kMaxDropL19S1 = 9;
inline constexpr int kMaxDropL19S2 = 7;
inline constexpr int kMaxDropL19S4 = 12;
inline constexpr int kMaxDropL18Control = 6;
// Head-pair slots (layer, pair) fixed before results.
inline constexpr std::array<std::pair<int, int>, 6> kPairSlots{{
    {7, 0},   // S2 b7 head0+head1
    {12, 0},  // S4 b12 head0+head1
    {9, 0},   // S1 b9 head0+head1
    {6, 0},   // L18 b6 head0+head1
    {16, 0},  // S2 deep layer 16
    {17, 0},  // S4 deep layer 17
}};

// ---------------------------------------------------------------------------
// Attention intervention
// ---------------------------------------------------------------------------
// Per (layer, head) action applied to the head context before concat.
enum class HeadAction : int {
  kKeep = 0,   // keep the computed head context
  kZero = 1,   // zero the head context
  kSwap = 2,   // replace with a supplied context (cross-seed swap)
};

struct HeadIntervention {
  int layer = -1;   // 0-based; -1 = no intervention
  int head = -1;    // 0-based
  HeadAction action = HeadAction::kKeep;
  // For kSwap: the replacement context (tokens x head_dim, row-major).
  std::vector<float> replacement;
};

// Attention-weight / value separation: per (layer, head) choose which model's
// attention weights and which model's V to use. The "own" model is the one
// being forwarded; the "other" model supplies weights or V.
struct AttentionValueSwap {
  int layer = -1;
  int head = -1;
  bool useOtherWeights = false;  // use other model's Q/K (attention pattern)
  bool useOtherV = false;        // use other model's V
  // The other model's layer parameters (wq/wk/wv) for this layer.
  const train::LP* otherLayer = nullptr;
};

// ---------------------------------------------------------------------------
// Intervened forward: bitwise identical to train::generalForward when no
// intervention is active. Head contexts are accumulated per-head into a
// temporary buffer (same float add order as the original direct accumulation
// into the head block of ctx), then the intervention is applied, then the
// head blocks are copied into the concat ctx buffer.
// ---------------------------------------------------------------------------
inline train::GF generalForwardIntervened(
    const tiny::Config& c, const std::vector<float>& oh, const train::P& w,
    const std::vector<HeadIntervention>& interventions = {},
    const std::vector<AttentionValueSwap>& swaps = {}) {
  train::GF g;
  g.embedded = train::mm(oh, w.tokenEmbedding, c.tokens, c.vocabularySize,
                         c.dimension);
  std::vector<float> x = g.embedded;
  train::add(x, train::fixedPositionCpu(c));
  g.embedded = x;
  const uint32_t dh = c.dimension / c.numHeads;
  for (uint32_t li = 0; li < c.numLayers; ++li) {
    const train::LP& p = train::layer(w, li);
    train::GL z;
    z.x = x;
    z.n1 = train::nf(c, x, p.gamma1, p.beta1);
    z.q = train::mm(z.n1.out, p.wq, c.tokens, c.dimension, c.dimension);
    z.k = train::mm(z.n1.out, p.wk, c.tokens, c.dimension, c.dimension);
    z.v = train::mm(z.n1.out, p.wv, c.tokens, c.dimension, c.dimension);
    z.prob.assign(static_cast<std::size_t>(c.numHeads) * c.tokens * c.tokens,
                  0);
    z.ctx.assign(static_cast<std::size_t>(c.tokens) * c.dimension, 0);
    const float scale = 1 / std::sqrt(float(dh));
    // Per-head context buffers (tokens x head_dim each).
    std::vector<std::vector<float>> headCtx(
        static_cast<std::size_t>(c.numHeads),
        std::vector<float>(static_cast<std::size_t>(c.tokens) * dh, 0.0f));
    // Attention-weight / V swap: choose the effective Q/K/V per head.
    // The swap only changes the attention pattern (Q/K) and the value (V)
    // for the specified head; the arithmetic order is preserved.
    // Per-head effective Q/K/V tensors (recomputed only when swapped).
    std::vector<std::vector<float>> effQ, effK, effV;
    effQ.resize(static_cast<std::size_t>(c.numHeads));
    effK.resize(static_cast<std::size_t>(c.numHeads));
    effV.resize(static_cast<std::size_t>(c.numHeads));
    for (uint32_t h = 0; h < c.numHeads; ++h) {
      const AttentionValueSwap* swap = nullptr;
      for (const auto& s : swaps)
        if (s.layer == static_cast<int>(li) && s.head == static_cast<int>(h)) {
          swap = &s;
          break;
        }
      if (swap && swap->otherLayer) {
        if (swap->useOtherWeights) {
          effQ[static_cast<std::size_t>(h)] =
              train::mm(z.n1.out, swap->otherLayer->wq, c.tokens, c.dimension,
                        c.dimension);
          effK[static_cast<std::size_t>(h)] =
              train::mm(z.n1.out, swap->otherLayer->wk, c.tokens, c.dimension,
                        c.dimension);
        }
        if (swap->useOtherV) {
          effV[static_cast<std::size_t>(h)] =
              train::mm(z.n1.out, swap->otherLayer->wv, c.tokens, c.dimension,
                        c.dimension);
        }
      }
    }
    for (uint32_t h = 0; h < c.numHeads; ++h) {
      const float* qRow = z.q.data();
      const float* kRow = z.k.data();
      const float* vRow = z.v.data();
      const AttentionValueSwap* swap = nullptr;
      for (const auto& s : swaps)
        if (s.layer == static_cast<int>(li) && s.head == static_cast<int>(h)) {
          swap = &s;
          break;
        }
      if (swap && swap->otherLayer) {
        if (swap->useOtherWeights) {
          qRow = effQ[static_cast<std::size_t>(h)].data();
          kRow = effK[static_cast<std::size_t>(h)].data();
        }
        if (swap->useOtherV) vRow = effV[static_cast<std::size_t>(h)].data();
      }
      for (uint32_t r = 0; r < c.tokens; ++r) {
        const std::size_t base =
            (static_cast<std::size_t>(h) * c.tokens + r) * c.tokens;
        float mx = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j <= r; ++j) {
          double s = 0;
          for (uint32_t d = 0; d < dh; ++d)
            s += double(qRow[static_cast<std::size_t>(r) * c.dimension +
                             h * dh + d]) *
                 kRow[static_cast<std::size_t>(j) * c.dimension + h * dh + d];
          mx = std::max(mx, float(s) * scale);
        }
        double sum = 0;
        for (uint32_t j = 0; j <= r; ++j) {
          double s = 0;
          for (uint32_t d = 0; d < dh; ++d)
            s += double(qRow[static_cast<std::size_t>(r) * c.dimension +
                             h * dh + d]) *
                 kRow[static_cast<std::size_t>(j) * c.dimension + h * dh + d];
          const float e = std::exp(float(s) * scale - mx);
          z.prob[base + j] = e;
          sum += e;
        }
        for (uint32_t j = 0; j <= r; ++j) {
          const float a = z.prob[base + j] / float(sum);
          z.prob[base + j] = a;
          for (uint32_t d = 0; d < dh; ++d)
            headCtx[static_cast<std::size_t>(h)]
                   [static_cast<std::size_t>(r) * dh + d] +=
                a * vRow[static_cast<std::size_t>(j) * c.dimension + h * dh +
                         d];
        }
      }
    }
    // Apply head interventions.
    for (const auto& iv : interventions) {
      if (iv.layer != static_cast<int>(li) || iv.head < 0 ||
          iv.head >= static_cast<int>(c.numHeads))
        continue;
      auto& hc = headCtx[static_cast<std::size_t>(iv.head)];
      if (iv.action == HeadAction::kZero) {
        std::fill(hc.begin(), hc.end(), 0.0f);
      } else if (iv.action == HeadAction::kSwap) {
        if (iv.replacement.size() !=
            static_cast<std::size_t>(c.tokens) * dh)
          throw std::invalid_argument("HEAD_SWAP_CONTEXT_SIZE");
        for (const float value : iv.replacement)
          if (!std::isfinite(value))
            throw std::invalid_argument("HEAD_SWAP_NONFINITE");
        hc = iv.replacement;
      }
    }
    // Copy per-head contexts into the concat ctx buffer.
    for (uint32_t h = 0; h < c.numHeads; ++h)
      for (uint32_t r = 0; r < c.tokens; ++r)
        for (uint32_t d = 0; d < dh; ++d)
          z.ctx[static_cast<std::size_t>(r) * c.dimension + h * dh + d] =
              headCtx[static_cast<std::size_t>(h)]
                     [static_cast<std::size_t>(r) * dh + d];
    z.r1 = x;
    train::add(z.r1, train::mm(z.ctx, p.wo, c.tokens, c.dimension, c.dimension));
    z.n2 = train::nf(c, z.r1, p.gamma2, p.beta2);
    z.f1 = train::mm(z.n2.out, p.w1, c.tokens, c.dimension,
                     c.feedForwardDimension);
    z.relu = z.f1;
    for (float& v : z.relu) v = std::max(0.f, v);
    z.out = z.r1;
    train::add(z.out, train::mm(z.relu, p.w2, c.tokens,
                                c.feedForwardDimension, c.dimension));
    x = z.out;
    g.layers.push_back(std::move(z));
  }
  g.logits = train::mm(x, w.outputProjection, c.tokens, c.dimension,
                       c.vocabularySize);
  g.prob.resize(g.logits.size());
  for (uint32_t r = 0; r < c.tokens; ++r) {
    const std::size_t b = static_cast<std::size_t>(r) * c.vocabularySize;
    const float mx = *std::max_element(g.logits.begin() + b,
                                       g.logits.begin() + b + c.vocabularySize);
    double s = 0;
    for (uint32_t j = 0; j < c.vocabularySize; ++j) {
      const float e = std::exp(g.logits[b + j] - mx);
      g.prob[b + j] = e;
      s += e;
    }
    for (uint32_t j = 0; j < c.vocabularySize; ++j)
      g.prob[b + j] /= float(s);
  }
  return g;
}

// ---------------------------------------------------------------------------
// Attention observer taps
// ---------------------------------------------------------------------------
enum class TapKind : int {
  kNorm1 = 0,
  kQ = 1,
  kK = 2,
  kV = 3,
  kCtxH0 = 4,
  kCtxH1 = 5,
  kCtxConcat = 6,
  kAttnUpdate = 7,
  kAfterAttn = 8,
};

inline const char* tapKindName(TapKind kind) {
  switch (kind) {
    case TapKind::kNorm1: return "NORM1";
    case TapKind::kQ: return "Q";
    case TapKind::kK: return "K";
    case TapKind::kV: return "V";
    case TapKind::kCtxH0: return "CTX_H0";
    case TapKind::kCtxH1: return "CTX_H1";
    case TapKind::kCtxConcat: return "CTX_CONCAT";
    case TapKind::kAttnUpdate: return "ATT_UPDATE";
    case TapKind::kAfterAttn: return "AFTER_ATTN";
  }
  return "UNKNOWN_TAP";
}

struct TapKey {
  TapKind kind = TapKind::kNorm1;
  int block = -1;  // 0-based
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

// Fixed registry for a depth: per target layer {NORM1, Q, K, V, CTX_H0,
// CTX_H1, CTX_CONCAT, ATT_UPDATE, AFTER_ATTN}. For small depths (self-test)
// only the target layers that fit within the depth are kept so the registry
// never references a layer beyond the model.
inline std::vector<TapSpec> buildTapRegistry(int depth) {
  std::vector<TapSpec> taps;
  int id = 0;
  std::vector<int> targets;
  if (depth == 19) {
    targets.assign(kTargetLayersL19.begin(), kTargetLayersL19.end());
  } else if (depth == 18) {
    targets.assign(kTargetLayersL18.begin(), kTargetLayersL18.end());
  } else {
    // Small self-test depth: every layer 0..depth-1 is a target.
    for (int li = 0; li < depth; ++li) targets.push_back(li);
  }
  for (const int li : targets) {
    auto push = [&](TapKind kind, int dim, const std::string& suffix) {
      std::ostringstream name;
      name << "b" << (li < 10 ? "0" : "") << li << "_" << suffix;
      taps.push_back(TapSpec{id++, {kind, li}, name.str(), dim});
    };
    push(TapKind::kNorm1, 16, "NORM1");
    push(TapKind::kQ, 16, "Q");
    push(TapKind::kK, 16, "K");
    push(TapKind::kV, 16, "V");
    push(TapKind::kCtxH0, 8, "CTX_H0");
    push(TapKind::kCtxH1, 8, "CTX_H1");
    push(TapKind::kCtxConcat, 16, "CTX_CONCAT");
    push(TapKind::kAttnUpdate, 16, "ATT_UPDATE");
    push(TapKind::kAfterAttn, 16, "AFTER_ATTN");
  }
  return taps;
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
  rp::LayerSet layerSetFor(int tapId) const {
    rp::LayerSet set;
    set.depth = depth;
    set.repCount = 1;
    set.dim = tap(tapId).dim;
    set.features = {features.at(static_cast<std::size_t>(tapId))};
    return set;
  }
};

// Extract attention-internal taps for the target layers. One full forward per
// row (verbatim train::generalForward); taps are read from the returned
// tensors at the last token position. CTX_H0/CTX_H1 are the head blocks of
// the concat ctx; CTX_CONCAT is the full ctx. ATT_UPDATE is recomputed with
// the same mm arithmetic (bitwise equal to what is added to the residual).
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
        std::vector<float>(rows.size() * static_cast<std::size_t>(t.dim),
                           0.0f));
  const std::uint32_t dim = config.dimension;
  const std::uint32_t dh = dim / config.numHeads;
  const std::size_t lastRow = dim * (config.tokens - 1);
  for (std::size_t ri = 0; ri < rows.size(); ++ri) {
    const auto oh = tiny::oneHot(rows[ri].context, config.vocabularySize);
    const train::GF g = train::generalForward(config, oh, params);
    for (std::size_t ti = 0; ti < set.taps.size(); ++ti) {
      const TapSpec& t = set.taps[ti];
      float* out =
          set.features[ti].data() + ri * static_cast<std::size_t>(t.dim);
      const auto& z = g.layers[static_cast<std::size_t>(t.key.block)];
      switch (t.key.kind) {
        case TapKind::kNorm1:
          std::memcpy(out, z.n1.out.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kQ:
          std::memcpy(out, z.q.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kK:
          std::memcpy(out, z.k.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kV:
          std::memcpy(out, z.v.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kCtxH0:
          std::memcpy(out, z.ctx.data() + lastRow, dh * sizeof(float));
          break;
        case TapKind::kCtxH1:
          std::memcpy(out, z.ctx.data() + lastRow + dh, dh * sizeof(float));
          break;
        case TapKind::kCtxConcat:
          std::memcpy(out, z.ctx.data() + lastRow, dim * sizeof(float));
          break;
        case TapKind::kAttnUpdate: {
          const auto& p = train::layer(params,
                                       static_cast<std::uint32_t>(t.key.block));
          const auto upd = train::mm(z.ctx, p.wo, config.tokens, dim, dim);
          std::memcpy(out, upd.data() + lastRow, dim * sizeof(float));
          break;
        }
        case TapKind::kAfterAttn:
          std::memcpy(out, z.r1.data() + lastRow, dim * sizeof(float));
          break;
      }
    }
  }
  return set;
}

// ---------------------------------------------------------------------------
// Tap feature cache (private; never committed). New magic; the tap layout
// differs from the intra-block cache by design.
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
  const std::uint64_t magic = 0x415454494E544531ull;  // "ATTINTE1"
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
  if (magic != 0x415454494E544531ull) return false;
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
                      std::vector<float>(static_cast<std::size_t>(rows) *
                                             static_cast<std::size_t>(dims[0]),
                                         0.0f));
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
// Probe training on a tap (identical protocol to READOUT_PROBE_V1).
// ---------------------------------------------------------------------------
inline rp::ZStats tapZStats(const TapSet& set, int tapId,
                            std::size_t trainRows) {
  const rp::LayerSet view = set.layerSetFor(tapId);
  return rp::computeZStats(view, 0, trainRows);
}

// ---------------------------------------------------------------------------
// Attention weight statistics (per head per row, DEV rows only).
// ---------------------------------------------------------------------------
struct AttentionRowStats {
  double entropy = 0.0;
  double maxWeight = 0.0;
  double selfWeight = 0.0;
  double prevWeight = 0.0;
  double top1MinusTop2 = 0.0;
  int top1Position = -1;
  std::vector<double> distanceProfile;  // 8 bins (distance 0..7)
};

// Compute per-row attention statistics for a head at a layer. The attention
// probability layout is head x query x key (causal j<=r). The row is the last
// query position (tokens-1).
inline AttentionRowStats attentionRowStats(const train::GF& g, int layer,
                                           int head, int tokens, int vocab) {
  (void)vocab;
  AttentionRowStats stats;
  stats.distanceProfile.assign(8, 0.0);
  const std::size_t r = static_cast<std::size_t>(tokens - 1);
  const std::size_t base =
      (static_cast<std::size_t>(head) * tokens + r) * tokens;
  std::vector<double> probs(static_cast<std::size_t>(tokens), 0.0);
  double sum = 0.0;
  for (int j = 0; j <= static_cast<int>(r); ++j) {
    const double p = static_cast<double>(
        g.layers[static_cast<std::size_t>(layer)].prob[base + j]);
    probs[static_cast<std::size_t>(j)] = p;
    sum += p;
    if (p > 0.0) stats.entropy -= p * std::log(p);
    stats.maxWeight = std::max(stats.maxWeight, p);
    if (j == static_cast<int>(r)) stats.selfWeight = p;
    if (j == static_cast<int>(r) - 1) stats.prevWeight = p;
    const int dist = static_cast<int>(r) - j;
    if (dist >= 0 && dist < 8) stats.distanceProfile[static_cast<std::size_t>(dist)] += p;
  }
  // top-1 position and top1-top2 gap.
  int top1 = -1;
  double top1v = -1.0, top2v = -1.0;
  for (int j = 0; j <= static_cast<int>(r); ++j) {
    const double p = probs[static_cast<std::size_t>(j)];
    if (p > top1v) {
      top2v = top1v;
      top1 = j;
      top1v = p;
    } else if (p > top2v) {
      top2v = p;
    }
  }
  stats.top1Position = top1;
  stats.top1MinusTop2 = top1v - top2v;
  return stats;
}

// Head-pair cosine between two heads' attention rows (DEV rows).
inline double headPairCosine(const train::GF& g, int layer, int headA,
                             int headB, int tokens) {
  const std::size_t r = static_cast<std::size_t>(tokens - 1);
  const std::size_t baseA =
      (static_cast<std::size_t>(headA) * tokens + r) * tokens;
  const std::size_t baseB =
      (static_cast<std::size_t>(headB) * tokens + r) * tokens;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int j = 0; j <= static_cast<int>(r); ++j) {
    const double a = static_cast<double>(
        g.layers[static_cast<std::size_t>(layer)].prob[baseA + j]);
    const double b = static_cast<double>(
        g.layers[static_cast<std::size_t>(layer)].prob[baseB + j]);
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ---------------------------------------------------------------------------
// Output projection contribution decomposition.
// contribution_h = ctx_h @ Wo[:, h*8:(h+1)*8]; sum_h contribution_h ==
// ATT_UPDATE bit-for-bit (verified in self-test).
// ---------------------------------------------------------------------------
struct HeadContribution {
  int layer = -1;
  int head = -1;
  std::vector<float> contribution;  // tokens x dim
  double norm = 0.0;
  double cosineWithInput = 0.0;
  double marginContribution = 0.0;
  double correctLogitContribution = 0.0;
  double maxCompetitorContribution = 0.0;
};

inline std::vector<HeadContribution> decomposeContributions(
    const tiny::Config& config, const train::P& params, const train::GF& g,
    int layer, const std::vector<rp::ProbeRow>& devRows,
    std::size_t devBegin) {
  (void)devBegin;
  const std::uint32_t dim = config.dimension;
  const std::uint32_t dh = dim / config.numHeads;
  const auto& p = train::layer(params, static_cast<std::uint32_t>(layer));
  const auto& z = g.layers[static_cast<std::size_t>(layer)];
  std::vector<HeadContribution> out;
  out.reserve(config.numHeads);
  for (uint32_t h = 0; h < config.numHeads; ++h) {
    HeadContribution hc;
    hc.layer = layer;
    hc.head = static_cast<int>(h);
    hc.contribution.assign(static_cast<std::size_t>(config.tokens) * dim, 0.0f);
    // contribution[r][d] = sum_{dd in head block} ctx[r][h*dh+dd] * Wo[(h*dh+dd)*dim + d]
    for (uint32_t r = 0; r < config.tokens; ++r)
      for (uint32_t d = 0; d < dim; ++d) {
        double s = 0.0;
        for (uint32_t dd = 0; dd < dh; ++dd)
          s += double(z.ctx[static_cast<std::size_t>(r) * dim + h * dh + dd]) *
               p.wo[static_cast<std::size_t>(h * dh + dd) * dim + d];
        hc.contribution[static_cast<std::size_t>(r) * dim + d] = float(s);
      }
    // Norm over DEV rows (last token position).
    double normSq = 0.0;
    const std::size_t lastRow = static_cast<std::size_t>(config.tokens - 1) * dim;
    for (std::size_t i = 0; i < devRows.size(); ++i) {
      for (uint32_t d = 0; d < dim; ++d) {
        const double v = hc.contribution[lastRow + d];
        normSq += v * v;
      }
    }
    hc.norm = std::sqrt(normSq / static_cast<double>(devRows.size()));
    // Cosine with block input x (residual before attention).
    double dot = 0.0, nx = 0.0, nc = 0.0;
    for (std::size_t i = 0; i < devRows.size(); ++i) {
      for (uint32_t d = 0; d < dim; ++d) {
        const double c = hc.contribution[lastRow + d];
        const double x = z.x[lastRow + d];
        dot += c * x;
        nc += c * c;
        nx += x * x;
      }
    }
    hc.cosineWithInput =
        (nc > 0.0 && nx > 0.0) ? dot / (std::sqrt(nc) * std::sqrt(nx)) : 0.0;
    out.push_back(std::move(hc));
  }
  return out;
}

// Verify sum_h contribution_h == ATT_UPDATE (relative tolerance).
inline bool contributionSumMatches(const tiny::Config& config,
                                   const train::P& params, const train::GF& g,
                                   int layer, double tol) {
  const std::uint32_t dim = config.dimension;
  const auto contribs = decomposeContributions(config, params, g, layer, {}, 0);
  const auto& p = train::layer(params, static_cast<std::uint32_t>(layer));
  const auto& z = g.layers[static_cast<std::size_t>(layer)];
  const auto upd = train::mm(z.ctx, p.wo, config.tokens, dim, dim);
  double maxRel = 0.0;
  for (std::size_t i = 0; i < upd.size(); ++i) {
    double sum = 0.0;
    for (const auto& hc : contribs) sum += hc.contribution[i];
    const double denom =
        std::max(static_cast<double>(std::abs(upd[i])), 1.0e-12);
    maxRel = std::max(maxRel, std::abs(sum - upd[i]) / denom);
  }
  return maxRel <= tol;
}

// ---------------------------------------------------------------------------
// Head-level intervention scoring on DEV rows (teacher-forced).
// ---------------------------------------------------------------------------
struct InterventionMetrics {
  std::uint64_t tokenExact = 0;
  std::uint64_t total = 0;
  double meanMargin = 0.0;
  double meanRank = 0.0;
  double meanNll = 0.0;
  double marginQ10 = 0.0;
  int firstErrorPosition = -1;
  bool finite = true;
};

// Score the current head (final logits) on DEV rows after an intervention.
inline InterventionMetrics scoreIntervenedHead(
    const tiny::Config& config, const train::P& params,
    const std::vector<rp::ProbeRow>& devRows,
    const std::vector<HeadIntervention>& interventions,
    const std::vector<AttentionValueSwap>& swaps = {}) {
  InterventionMetrics metrics;
  metrics.total = devRows.size();
  std::vector<double> margins;
  margins.reserve(devRows.size());
  for (const auto& row : devRows) {
    const auto oh = tiny::oneHot(row.context, config.vocabularySize);
    const train::GF g =
        generalForwardIntervened(config, oh, params, interventions, swaps);
    const std::size_t base =
        std::size_t(config.tokens - 1) * config.vocabularySize;
    std::vector<double> logits(config.vocabularySize);
    std::vector<double> probs(config.vocabularySize);
    for (std::uint32_t j = 0; j < config.vocabularySize; ++j) {
      logits[j] = static_cast<double>(g.logits[base + j]);
      probs[j] = static_cast<double>(g.prob[base + j]);
    }
    if (!std::isfinite(logits[row.truth])) {
      metrics.finite = false;
      continue;
    }
    const ma::Score score = rp::stableScoreFromLogits(logits, probs, row.truth);
    if (score.predicted == row.truth) ++metrics.tokenExact;
    metrics.meanMargin += score.expectedMinusTop1Margin;
    metrics.meanRank += score.expectedRank;
    metrics.meanNll += score.tokenNll;
    margins.push_back(score.expectedMinusTop1Margin);
  }
  const double n = static_cast<double>(devRows.size());
  if (n > 0.0) {
    metrics.meanMargin /= n;
    metrics.meanRank /= n;
    metrics.meanNll /= n;
  }
  if (!margins.empty()) {
    std::sort(margins.begin(), margins.end());
    const std::size_t cut = std::max<std::size_t>(1, margins.size() / 10);
    double sum = 0.0;
    for (std::size_t i = 0; i < cut; ++i) sum += margins[i];
    metrics.marginQ10 = sum / static_cast<double>(cut);
  }
  return metrics;
}

// ---------------------------------------------------------------------------
// Free-running scoring on an intervened forward (secondary evidence).
// ---------------------------------------------------------------------------
struct IntervenedScorer {
  const tiny::Config* config = nullptr;
  const train::P* params = nullptr;
  std::vector<HeadIntervention> interventions;
  std::vector<AttentionValueSwap> swaps;

  inline ma::Score operator()(const std::vector<std::uint32_t>& context,
                              std::uint32_t truth) const {
    if (!config || !params) throw std::invalid_argument("INTERVENED_SCORER_UNSET");
    const auto oh = tiny::oneHot(context, config->vocabularySize);
    const train::GF g =
        generalForwardIntervened(*config, oh, *params, interventions, swaps);
    const std::size_t base =
        std::size_t(config->tokens - 1) * config->vocabularySize;
    std::vector<double> logits(config->vocabularySize);
    std::vector<double> probs(config->vocabularySize);
    for (std::uint32_t j = 0; j < config->vocabularySize; ++j) {
      logits[j] = static_cast<double>(g.logits[base + j]);
      probs[j] = static_cast<double>(g.prob[base + j]);
    }
    if (!std::isfinite(logits[truth]))
      throw std::runtime_error("NON_FINITE_INTERVENED_LOGITS");
    return rp::stableScoreFromLogits(logits, probs, truth);
  }
};

inline cm::CheckpointMetrics intervenedFreeRunning(
    const std::vector<ar::Case>& cases, int step,
    const IntervenedScorer& scorer) {
  return rp::freeRunningRollout(cases, step, scorer);
}

}  // namespace phonelm::attention_internal

#endif  // ATTENTION_INTERNAL_DIAGNOSIS_LIB_H