// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only linear readout probe library for the L19 readout/representation
// diagnosis (READOUT_PROBE_V1 protocol). Everything here is host-only and
// deterministic; no Android, QAIRT, ADB, QNN graph, or device path is
// involved. The frozen-model features come from the VERBATIM forward copy in
// critical_margin_training_lib.h (phonelm::critical_margin::train), which is
// asserted bitwise against the production forward by the margin probe
// self-test; this library never modifies production code.
//
// Protocol (fixed before any result was read; see protocol.json):
//  - rows: teacher-forced sliding 8-token windows over gold sequences
//    (TRAIN 32 rows, MARGIN_CALIBRATION_V1 144, MARGIN_DEVELOPMENT_V1 144)
//  - probe: logits = W z + b, z = TRAIN-only z-scored features (double)
//  - init: house LCG seed 20260804 / phase 101 / scale 0.1, b = 0
//  - optimizer: full-batch Adam, lr 0.01, beta1 .9, beta2 .999, eps 1e-8,
//    bias-corrected, m = v = 0, no weight decay, 2000 steps, grid every 25
//  - step selection: min calibration CE -> higher calibration token exact
//    -> earlier step (tolerance 1e-7); development is final evaluation only
//  - NaN/Inf: immediate per-probe abort with PROBE_NONFINITE, continue
//  - head retraining: candidates A (warm start) / B (reinit phase 8 scale
//    .16) / C (bias only), Adam lr 0.003 LEGACY, 320 steps, batches
//    formalBatch((step-1)%4), selection grid = canonical 23-point cadence
#ifndef READOUT_PROBE_LIB_H
#define READOUT_PROBE_LIB_H

#include "autoregressive_validation.h"
#include "critical_margin_objective_lib.h"
#include "critical_margin_training_lib.h"
#include "depth_quality_lib.h"
#include "margin_analysis_lib.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::readout_probe {

namespace ar = phonelm::autoregressive_validation;
namespace cm = phonelm::critical_margin;
namespace train = phonelm::critical_margin::train;
namespace ma = phonelm::margin_analysis;
namespace dq = phonelm::depth_quality;
namespace tiny = phonelm::tiny_lm;

// ---------------------------------------------------------------------------
// Pinned protocol constants (READOUT_PROBE_V1)
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kProbeSeed = 20260804u;
inline constexpr std::uint32_t kProbePhase = 101u;
inline constexpr double kProbeInitScale = 0.1;
inline constexpr double kProbeLr = 0.01;
inline constexpr int kProbeSteps = 2000;
inline constexpr int kProbeGridStep = 25;
inline constexpr double kProbeStdFloor = 1.0e-6;
inline constexpr double kTieTolerance = 1.0e-7;
inline constexpr double kHeadLr = 0.003;
inline constexpr int kHeadSteps = 320;
inline constexpr std::uint32_t kTokens = 8;
inline constexpr std::uint32_t kClasses = 32;
inline constexpr const char* kProbeProtocolId = "READOUT_PROBE_V1";
// Representative layer set used for non-final checkpoints:
// embedded, L4, L8, L12, L16, final block out, PRE_LN_FINAL, POST_LN_FINAL.
inline constexpr std::array<int, 8> kRepresentativeReps{0, 4, 8, 12, 16, -1,
                                                        -2, -3};
// Sentinel meaning "final block out" / "PRE_LN_FINAL" / "POST_LN_FINAL".
inline constexpr int kRepFinalBlockOut = -1;
inline constexpr int kRepPreLnFinal = -2;
inline constexpr int kRepPostLnFinal = -3;

// ---------------------------------------------------------------------------
// Probe rows: teacher-forced sliding windows over gold sequences.
// Row i of a case: context = sequence[i..i+7], truth = targets[i].
// TRAIN cases store 8 batch rows; margin cases store rollout rows (4 or 8).
// ---------------------------------------------------------------------------
struct ProbeRow {
  std::string caseId;
  std::vector<std::uint32_t> context;
  std::uint32_t truth = 0;
};

inline std::uint32_t rowsTruth(const std::vector<ProbeRow>& rows,
                               std::size_t index) {
  return rows.at(index).truth;
}

inline std::vector<ProbeRow> teacherForcedRows(
    const std::vector<ar::Case>& cases, std::uint32_t tokens = kTokens) {
  std::vector<ProbeRow> rows;
  rows.reserve(cases.size() * 8);
  for (const auto& item : cases) {
    std::vector<std::uint32_t> sequence = item.initialPrefix;
    sequence.insert(sequence.end(), item.targets.begin(), item.targets.end());
    for (std::size_t i = 0; i < item.targets.size(); ++i) {
      if (i + tokens > sequence.size())
        throw std::invalid_argument("PROBE_ROW_WINDOW_OUT_OF_RANGE");
      ProbeRow row;
      row.caseId = item.id;
      row.context.assign(sequence.begin() + i, sequence.begin() + i + tokens);
      row.truth = item.targets[i];
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Feature extraction. One verbatim forward per row yields every
// representation at once. Rep index mapping for depth L:
//   0         = embedded (position-add applied)
//   1..L      = block k output (layers[k-1].out); rep L == HEAD_IN
//   L+1       = PRE_LN_FINAL (last block r1, LN2 input)
//   L+2       = POST_LN_FINAL (last block n2.out, LN2 output)
// The architecture has NO global final LayerNorm; the tensor entering the
// output projection is the last block's out (== rep L).
// ---------------------------------------------------------------------------
inline int representationCount(int depth) { return depth + 3; }

inline const char* representationName(int depth, int rep) {
  if (rep == 0) return "L0_EMBEDDED";
  if (rep >= 1 && rep <= depth) {
    static thread_local std::string buffer;
    std::ostringstream out;
    out << "L" << rep << "_BLOCK_OUT";
    buffer = out.str();
    return buffer.c_str();
  }
  if (rep == depth + 1) return "PRE_LN_FINAL";
  if (rep == depth + 2) return "POST_LN_FINAL";
  return "UNKNOWN_REP";
}

struct LayerSet {
  int depth = 0;
  int repCount = 0;
  int dim = 0;
  std::vector<std::vector<float>> features;  // [rep][row*dim + d]
};

inline LayerSet extractFeatures(const tiny::Config& config,
                                const train::P& params,
                                const std::vector<ProbeRow>& rows) {
  const int depth = static_cast<int>(config.numLayers);
  const int repCount = representationCount(depth);
  const std::uint32_t dim = config.dimension;
  LayerSet set;
  set.depth = depth;
  set.repCount = repCount;
  set.dim = static_cast<int>(dim);
  set.features.assign(
      static_cast<std::size_t>(repCount),
      std::vector<float>(rows.size() * dim, 0.0f));
  const std::size_t lastRow = dim * (config.tokens - 1);
  for (std::size_t ri = 0; ri < rows.size(); ++ri) {
    const auto oh = tiny::oneHot(rows[ri].context, config.vocabularySize);
    const train::GF g = train::generalForward(config, oh, params);
    auto& embedded = set.features[0];
    std::memcpy(embedded.data() + ri * dim, g.embedded.data() + lastRow,
                dim * sizeof(float));
    for (int li = 0; li < depth; ++li) {
      auto& out = set.features[static_cast<std::size_t>(li) + 1];
      std::memcpy(out.data() + ri * dim, g.layers[static_cast<std::size_t>(li)].out.data() + lastRow,
                  dim * sizeof(float));
    }
    auto& pre = set.features[static_cast<std::size_t>(depth) + 1];
    std::memcpy(pre.data() + ri * dim,
                g.layers.back().r1.data() + lastRow, dim * sizeof(float));
    auto& post = set.features[static_cast<std::size_t>(depth) + 2];
    std::memcpy(post.data() + ri * dim,
                g.layers.back().n2.out.data() + lastRow, dim * sizeof(float));
  }
  return set;
}

// Rep resolution: protocol rep id -> concrete rep index for a depth.
// Handles the kRepresentativeReps sentinels.
inline int resolveRep(int depth, int rep) {
  switch (rep) {
    case kRepFinalBlockOut: return depth;
    case kRepPreLnFinal: return depth + 1;
    case kRepPostLnFinal: return depth + 2;
    default:
      if (rep < 0 || rep > representationCount(depth) - 1)
        throw std::invalid_argument("REP_OUT_OF_RANGE");
      return rep;
  }
}

// ---------------------------------------------------------------------------
// Z-score transform. Statistics from TRAIN rows only (leak-free); dead
// dimensions (std < floor) map to 0.
// ---------------------------------------------------------------------------
struct ZStats {
  int dim = 0;
  std::vector<double> mean;
  std::vector<double> std;
};

inline ZStats computeZStats(const LayerSet& set, int rep,
                            std::size_t trainRows) {
  ZStats stats;
  stats.dim = set.dim;
  stats.mean.assign(static_cast<std::size_t>(set.dim), 0.0);
  stats.std.assign(static_cast<std::size_t>(set.dim), 0.0);
  if (trainRows == 0) return stats;
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  for (std::size_t d = 0; d < static_cast<std::size_t>(set.dim); ++d) {
    double sum = 0.0;
    for (std::size_t r = 0; r < trainRows; ++r)
      sum += static_cast<double>(f[r * set.dim + d]);
    const double mean = sum / static_cast<double>(trainRows);
    double sq = 0.0;
    for (std::size_t r = 0; r < trainRows; ++r) {
      const double z = static_cast<double>(f[r * set.dim + d]) - mean;
      sq += z * z;
    }
    double var = sq / static_cast<double>(trainRows);
    if (var < 0.0) var = 0.0;
    stats.mean[d] = mean;
    stats.std[d] = var > 0.0 ? std::sqrt(var) : 0.0;
  }
  return stats;
}

inline double zScore(const ZStats& stats, const float* feature, int d) {
  if (stats.std[static_cast<std::size_t>(d)] < kProbeStdFloor) return 0.0;
  return (static_cast<double>(feature[d]) - stats.mean[static_cast<std::size_t>(d)]) /
         stats.std[static_cast<std::size_t>(d)];
}

// ---------------------------------------------------------------------------
// Linear softmax probe (double arithmetic, full-batch Adam)
// ---------------------------------------------------------------------------
struct Probe {
  int classes = static_cast<int>(kClasses);
  int dim = 0;
  std::vector<double> w;  // class-major: [class*dim + d]
  std::vector<double> b;  // [class]
};

inline Probe zeroProbe(int dim) {
  Probe probe;
  probe.dim = dim;
  probe.w.assign(static_cast<std::size_t>(kClasses) * static_cast<std::size_t>(dim), 0.0);
  probe.b.assign(kClasses, 0.0);
  return probe;
}

// House LCG fill, bit-identical to tiny_lm::initialParameters.
inline void lcgFill(std::vector<double>& target, std::size_t count,
                    std::uint32_t seed, std::uint32_t phase, double scale) {
  target.resize(count);
  std::uint32_t s = seed * 747796405u + phase * 2891336453u;
  for (std::size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    const float draw = (float(int((s >> 8) & 65535u)) / 32767.5f - 1.0f) *
                       static_cast<float>(scale);
    target[i] = static_cast<double>(draw);
  }
}

inline Probe initialProbe(int dim) {
  Probe probe = zeroProbe(dim);
  lcgFill(probe.w, probe.w.size(), kProbeSeed, kProbePhase, kProbeInitScale);
  return probe;
}

// Stable softmax over a logit row (double). Returns probabilities.
inline std::vector<double> softmaxRow(const double* logits, int classes) {
  std::vector<double> p(static_cast<std::size_t>(classes));
  double mx = -std::numeric_limits<double>::infinity();
  for (int j = 0; j < classes; ++j) mx = std::max(mx, logits[j]);
  double sum = 0.0;
  for (int j = 0; j < classes; ++j) {
    p[static_cast<std::size_t>(j)] = std::exp(logits[j] - mx);
    sum += p[static_cast<std::size_t>(j)];
  }
  if (!(sum > 0.0) || !std::isfinite(sum))
    throw std::runtime_error("PROBE_SOFTMAX_NONFINITE");
  for (int j = 0; j < classes; ++j) p[static_cast<std::size_t>(j)] /= sum;
  return p;
}

// log(sum(exp(logits))) with the max shift; always finite for finite
// logits. CE in the form logsumexp(l) - l[truth] is mathematically
// identical to -log(softmax(l)[truth]) but never underflows when the
// model saturates.
inline double logSumExp(const double* logits, int classes) {
  double mx = -std::numeric_limits<double>::infinity();
  for (int j = 0; j < classes; ++j) mx = std::max(mx, logits[j]);
  double sum = 0.0;
  for (int j = 0; j < classes; ++j) sum += std::exp(logits[j] - mx);
  if (!(sum > 0.0) || !std::isfinite(sum))
    throw std::runtime_error("PROBE_LOGSUMEXP_NONFINITE");
  return mx + std::log(sum);
}

inline double ceFromLogits(const double* logits, int classes,
                           std::uint32_t truth) {
  return logSumExp(logits, classes) - logits[truth];
}

// Score like ma::scoreFromLogits but with the underflow-safe token NLL.
inline ma::Score stableScoreFromLogits(const std::vector<double>& logits,
                                       const std::vector<double>& probabilities,
                                       std::uint32_t truth) {
  ma::Score score = ma::scoreFromLogits(logits, probabilities, truth);
  if (score.valid) score.tokenNll = ceFromLogits(logits.data(),
                                                 static_cast<int>(logits.size()),
                                                 truth);
  return score;
}

inline void probeForward(const Probe& probe, const ZStats& stats,
                         const float* feature, double* logits) {
  for (int c = 0; c < probe.classes; ++c) {
    double acc = probe.b[static_cast<std::size_t>(c)];
    const double* row =
        probe.w.data() + static_cast<std::size_t>(c) * probe.dim;
    for (int d = 0; d < probe.dim; ++d)
      acc += row[d] * zScore(stats, feature, d);
    logits[c] = acc;
  }
}

struct ProbeGridPoint {
  int step = 0;
  double trainCe = 0.0;
  double calCe = 0.0;
  std::uint64_t trainExact = 0;
  std::uint64_t calExact = 0;
};

struct ProbeTrainResult {
  Probe probe;  // at selected step
  int selectedStep = 0;
  bool finite = true;
  int nonfiniteStep = -1;
  std::string nonfiniteWhat;
  std::vector<ProbeGridPoint> grid;
  // Primary (selected step):
  double trainCe = 0.0;
  std::uint64_t trainExact = 0;
  double calCe = 0.0;
  std::uint64_t calExact = 0;
  double maxLogitAbs = 0.0;
  // Supplementary (step 2000, non-selecting):
  double trainCe2000 = 0.0;
  std::uint64_t trainExact2000 = 0;
};

// Full-batch cross entropy (mean over rows) for a probe.
struct ProbeEval {
  double ce = 0.0;
  std::uint64_t exact = 0;
  bool finite = true;
};

inline ProbeEval evaluateProbeRows(const Probe& probe, const ZStats& stats,
                                   const LayerSet& set, int rep,
                                   const std::vector<ProbeRow>& rows,
                                   std::size_t rowBegin,
                                   std::size_t rowEnd) {
  ProbeEval result;
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  std::vector<double> logits(kClasses);
  double loss = 0.0;
  for (std::size_t r = rowBegin; r < rowEnd; ++r) {
    probeForward(probe, stats, f.data() + r * set.dim, logits.data());
    for (const double l : logits)
      if (!std::isfinite(l)) {
        result.finite = false;
        return result;
      }
    loss += ceFromLogits(logits.data(), probe.classes, rows[r].truth);
    result.exact +=
        ma::argmaxFirst(std::vector<double>(logits.begin(), logits.end())) ==
                rows[r].truth
            ? 1u
            : 0u;
  }
  const double n = static_cast<double>(rowEnd - rowBegin);
  result.ce = n > 0.0 ? loss / n : 0.0;
  return result;
}

// Adam (double) step on probe parameters; returns false on non-finite.
inline bool adamStepProbe(Probe& probe, std::vector<double>& m,
                          std::vector<double>& v, const std::vector<double>& gradW,
                          const std::vector<double>& gradB, int step,
                          double lr) {
  const double c1 = 1.0 / (1.0 - std::pow(0.9, static_cast<double>(step)));
  const double c2 = 1.0 / (1.0 - std::pow(0.999, static_cast<double>(step)));
  const std::size_t wn = probe.w.size();
  for (std::size_t i = 0; i < wn; ++i) {
    m[i] = 0.9 * m[i] + 0.1 * gradW[i];
    v[i] = 0.999 * v[i] + 0.001 * gradW[i] * gradW[i];
    const double mh = m[i] * c1;
    const double vh = v[i] * c2;
    probe.w[i] = probe.w[i] - lr * mh / (std::sqrt(vh) + 1e-8);
    if (!std::isfinite(probe.w[i])) return false;
  }
  for (int c = 0; c < probe.classes; ++c) {
    const std::size_t i = static_cast<std::size_t>(c);
    m[wn + i] = 0.9 * m[wn + i] + 0.1 * gradB[i];
    v[wn + i] = 0.999 * v[wn + i] + 0.001 * gradB[i] * gradB[i];
    const double mh = m[wn + i] * c1;
    const double vh = v[wn + i] * c2;
    probe.b[i] = probe.b[i] - lr * mh / (std::sqrt(vh) + 1e-8);
    if (!std::isfinite(probe.b[i])) return false;
  }
  return true;
}

// Train one probe on TRAIN rows (rowBegin..rowEnd of the row list),
// select the step on CALIBRATION rows, never touch development rows.
inline ProbeTrainResult trainProbe(const LayerSet& set, int rep,
                                   const ZStats& stats,
                                   const std::vector<ProbeRow>& trainRows,
                                   std::size_t trainBegin,
                                   std::size_t trainEnd,
                                   const std::vector<ProbeRow>& calRows,
                                   std::size_t calBegin, std::size_t calEnd) {
  ProbeTrainResult result;
  result.probe = initialProbe(set.dim);
  const std::size_t wn = result.probe.w.size();
  std::vector<double> m(wn + kClasses, 0.0), v(wn + kClasses, 0.0);
  std::vector<double> gradW(wn, 0.0), gradB(kClasses, 0.0);
  std::vector<double> logits(kClasses);
  const auto& f = set.features.at(static_cast<std::size_t>(rep));

  const auto checkFinite = [&](int step, const char* what) {
    for (const double x : result.probe.w)
      if (!std::isfinite(x)) return false;
    for (const double x : result.probe.b)
      if (!std::isfinite(x)) return false;
    for (const double x : m)
      if (!std::isfinite(x)) return false;
    for (const double x : v)
      if (!std::isfinite(x)) return false;
    (void)step;
    (void)what;
    return true;
  };

  const auto evaluate = [&](const std::vector<ProbeRow>& rows, std::size_t begin,
                            std::size_t end) {
    return evaluateProbeRows(result.probe, stats, set, rep, rows, begin, end);
  };

  // Grid point at step 0 (init).
  {
    ProbeGridPoint point;
    point.step = 0;
    const auto t = evaluate(trainRows, trainBegin, trainEnd);
    const auto c = evaluate(calRows, calBegin, calEnd);
    point.trainCe = t.ce;
    point.calCe = c.ce;
    point.trainExact = t.exact;
    point.calExact = c.exact;
    if (!t.finite || !c.finite) result.finite = false;
    result.grid.push_back(point);
  }

  const double lr = kProbeLr;
  for (int step = 1; step <= kProbeSteps; ++step) {
    std::fill(gradW.begin(), gradW.end(), 0.0);
    std::fill(gradB.begin(), gradB.end(), 0.0);
    double loss = 0.0;
    const double invN = 1.0 / static_cast<double>(trainEnd - trainBegin);
    for (std::size_t r = trainBegin; r < trainEnd; ++r) {
      probeForward(result.probe, stats, f.data() + r * set.dim, logits.data());
      for (const double l : logits)
        if (!std::isfinite(l)) {
          result.finite = false;
          result.nonfiniteStep = step;
          result.nonfiniteWhat = "train_logits";
          return result;
        }
      const auto p = softmaxRow(logits.data(), result.probe.classes);
      loss += ceFromLogits(logits.data(), result.probe.classes,
                           rowsTruth(trainRows, r));
      for (int c = 0; c < result.probe.classes; ++c) {
        const double delta = p[static_cast<std::size_t>(c)] -
                             (c == static_cast<int>(rowsTruth(trainRows, r)) ? 1.0 : 0.0);
        gradB[static_cast<std::size_t>(c)] += delta * invN;
        for (int d = 0; d < result.probe.dim; ++d)
          gradW[static_cast<std::size_t>(c) * result.probe.dim + d] +=
              delta * zScore(stats, f.data() + r * set.dim, d) * invN;
      }
    }
    if (!adamStepProbe(result.probe, m, v, gradW, gradB, step, lr)) {
      result.finite = false;
      result.nonfiniteStep = step;
      result.nonfiniteWhat = "adam_update";
      return result;
    }
    if (!checkFinite(step, "params")) {
      result.finite = false;
      result.nonfiniteStep = step;
      result.nonfiniteWhat = "params";
      return result;
    }
    if (step % kProbeGridStep == 0 || step == kProbeSteps) {
      ProbeGridPoint point;
      point.step = step;
      const auto t = evaluate(trainRows, trainBegin, trainEnd);
      const auto c = evaluate(calRows, calBegin, calEnd);
      point.trainCe = t.ce;
      point.calCe = c.ce;
      point.trainExact = t.exact;
      point.calExact = c.exact;
      if (!t.finite || !c.finite) {
        result.finite = false;
        result.nonfiniteStep = step;
        result.nonfiniteWhat = "grid_eval";
        return result;
      }
      result.grid.push_back(point);
    }
  }

  // Selection: min calibration CE -> higher calibration token exact ->
  // earlier step (tolerance 1e-7).
  const ProbeGridPoint* best = nullptr;
  for (const auto& point : result.grid) {
    if (!best) {
      best = &point;
      continue;
    }
    const double delta = point.calCe - best->calCe;
    bool better = false;
    if (delta < -kTieTolerance) {
      better = true;
    } else if (std::abs(delta) <= kTieTolerance) {
      if (point.calExact > best->calExact) better = true;
      else if (point.calExact == best->calExact && point.step < best->step)
        better = true;
    }
    if (better) best = &point;
  }
  if (!best) throw std::runtime_error("PROBE_EMPTY_GRID");
  result.selectedStep = best->step;
  result.trainCe = best->trainCe;
  result.trainExact = best->trainExact;
  result.calCe = best->calCe;
  result.calExact = best->calExact;
  if (best->step != kProbeSteps) {
    // Recompute probe at the selected step (grid points do not retain W).
    // Re-training to the selected step is deterministic.
    Probe replay = initialProbe(set.dim);
    std::vector<double> rm(wn + kClasses, 0.0), rv(wn + kClasses, 0.0);
    for (int step = 1; step <= best->step; ++step) {
      std::fill(gradW.begin(), gradW.end(), 0.0);
      std::fill(gradB.begin(), gradB.end(), 0.0);
      const double invN = 1.0 / static_cast<double>(trainEnd - trainBegin);
      for (std::size_t r = trainBegin; r < trainEnd; ++r) {
        probeForward(replay, stats, f.data() + r * set.dim, logits.data());
        const auto p = softmaxRow(logits.data(), replay.classes);
        for (int c = 0; c < replay.classes; ++c) {
          const double delta =
              p[static_cast<std::size_t>(c)] -
              (c == static_cast<int>(rowsTruth(trainRows, r)) ? 1.0 : 0.0);
          gradB[static_cast<std::size_t>(c)] += delta * invN;
          for (int d = 0; d < replay.dim; ++d)
            gradW[static_cast<std::size_t>(c) * replay.dim + d] +=
                delta * zScore(stats, f.data() + r * set.dim, d) * invN;
        }
      }
      if (!adamStepProbe(replay, rm, rv, gradW, gradB, step, lr)) {
        result.finite = false;
        result.nonfiniteStep = step;
        result.nonfiniteWhat = "replay_adam";
        return result;
      }
    }
    result.probe = std::move(replay);
  }
  // Sanity: recompute selected-step metrics on the final probe.
  {
    const auto t = evaluate(trainRows, trainBegin, trainEnd);
    const auto c = evaluate(calRows, calBegin, calEnd);
    result.trainCe = t.ce;
    result.trainExact = t.exact;
    result.calCe = c.ce;
    result.calExact = c.exact;
    if (!t.finite || !c.finite) {
      result.finite = false;
      result.nonfiniteStep = best->step;
      result.nonfiniteWhat = "selected_eval";
    }
    double maxAbs = 0.0;
    for (const double x : result.probe.w) maxAbs = std::max(maxAbs, std::abs(x));
    for (const double x : result.probe.b) maxAbs = std::max(maxAbs, std::abs(x));
    result.maxLogitAbs = maxAbs;
  }
  // Supplementary (non-selecting) metrics at step 2000.
  if (!result.grid.empty()) {
    const auto& last = result.grid.back();
    result.trainCe2000 = last.trainCe;
    result.trainExact2000 = last.trainExact;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Current-head scoring / rollouts (host-only, verbatim forward copy)
// ---------------------------------------------------------------------------
inline ma::Score headScore(const tiny::Config& config, const train::P& params,
                           const std::vector<std::uint32_t>& context,
                           std::uint32_t truth) {
  const auto oh = tiny::oneHot(context, config.vocabularySize);
  const train::GF g = train::generalForward(config, oh, params);
  const std::size_t base =
      std::size_t(config.tokens - 1) * config.vocabularySize;
  std::vector<double> logits(config.vocabularySize);
  std::vector<double> probabilities(config.vocabularySize);
  for (std::uint32_t j = 0; j < config.vocabularySize; ++j) {
    logits[j] = static_cast<double>(g.logits[base + j]);
    probabilities[j] = static_cast<double>(g.prob[base + j]);
  }
  if (!std::isfinite(logits[truth]) || !(probabilities[truth] > 0.0) ||
      !std::isfinite(probabilities[truth]))
    throw std::runtime_error("NON_FINITE_HEAD_SCORE");
  return stableScoreFromLogits(logits, probabilities, truth);
}

inline std::vector<std::uint32_t> slideContext(
    const std::vector<std::uint32_t>& context, std::uint32_t next) {
  std::vector<std::uint32_t> result(context.begin() + 1, context.end());
  result.push_back(next);
  return result;
}

// Free-running rollout with a generic per-position scorer. The scorer may
// produce the head logits or probe logits; contexts always slide with the
// scorer's own argmax (free-running), matching the canonical evaluator.
template <typename Scorer>
cm::CheckpointMetrics freeRunningRollout(
    const std::vector<ar::Case>& cases, int step, Scorer&& scorer) {
  std::vector<cm::CaseTrace> traces;
  traces.reserve(cases.size());
  for (const auto& item : cases) {
    cm::CaseTrace trace;
    trace.id = item.id;
    std::vector<std::uint32_t> context = item.initialPrefix;
    std::vector<std::uint32_t> predicted;
    predicted.reserve(item.targets.size());
    bool finite = true;
    for (const std::uint32_t truth : item.targets) {
      ma::Score score;
      try {
        score = scorer(context, truth);
      } catch (const std::exception&) {
        finite = false;
        break;
      }
      trace.margins.push_back(score.expectedMinusTop1Margin);
      trace.autoregressiveNllSum += score.tokenNll;
      predicted.push_back(score.predicted);
      context = slideContext(context, score.predicted);
    }
    if (!finite) {
      trace.finite = false;
      traces.push_back(std::move(trace));
      continue;
    }
    const auto firstError = ma::firstErrorInfo(predicted, item.targets);
    trace.tokenExact = static_cast<std::uint64_t>(predicted.size()) -
                       firstError.wrongCount;
    trace.sequenceExact = firstError.wrongCount == 0;
    trace.firstErrorPosition =
        firstError.firstError < 0 ? -1 : firstError.firstError + 1;
    trace.finite = true;
    traces.push_back(std::move(trace));
  }
  return cm::summarizeCheckpoint(step, std::move(traces));
}

inline cm::CheckpointMetrics headFreeRunning(
    const tiny::Config& config, const train::P& params, int step,
    const std::vector<ar::Case>& cases) {
  return freeRunningRollout(cases, step, [&](const std::vector<std::uint32_t>& context,
                                             std::uint32_t truth) {
    return headScore(config, params, context, truth);
  });
}

inline ma::Score probeScore(const tiny::Config& config, const train::P& params,
                            const Probe& probe, const ZStats& stats,
                            const LayerSet& set, int rep,
                            const std::vector<std::uint32_t>& context,
                            std::uint32_t truth) {
  (void)set;
  const auto oh = tiny::oneHot(context, config.vocabularySize);
  const train::GF g = train::generalForward(config, oh, params);
  const std::size_t lastRow =
      std::size_t(config.tokens - 1) * config.dimension;
  std::vector<double> logits(kClasses);
  const float* feature = nullptr;
  if (rep == 0) {
    feature = g.embedded.data() + lastRow;
  } else if (rep <= static_cast<int>(config.numLayers)) {
    feature = g.layers[static_cast<std::size_t>(rep) - 1].out.data() + lastRow;
  } else if (rep == static_cast<int>(config.numLayers) + 1) {
    feature = g.layers.back().r1.data() + lastRow;
  } else if (rep == static_cast<int>(config.numLayers) + 2) {
    feature = g.layers.back().n2.out.data() + lastRow;
  } else {
    throw std::invalid_argument("REP_OUT_OF_RANGE");
  }
  probeForward(probe, stats, feature, logits.data());
  const auto p = softmaxRow(logits.data(), probe.classes);
  if (!std::isfinite(logits[truth]))
    throw std::runtime_error("NON_FINITE_PROBE_LOGITS");
  return stableScoreFromLogits(logits, p, truth);
}

inline cm::CheckpointMetrics probeFreeRunning(
    const tiny::Config& config, const train::P& params, int step,
    const std::vector<ar::Case>& cases, const Probe& probe,
    const ZStats& stats, const LayerSet& set, int rep) {
  return freeRunningRollout(cases, step,
                            [&](const std::vector<std::uint32_t>& context,
                                std::uint32_t truth) {
                              return probeScore(config, params, probe, stats,
                                                set, rep, context, truth);
                            });
}

// Probe evaluated on the contexts the CURRENT HEAD would produce (drift
// analysis). Contexts slide with the head argmax; exactness/margins/nll are
// the probe's on those contexts.
inline cm::CheckpointMetrics probeOnHeadContexts(
    const tiny::Config& config, const train::P& params, int step,
    const std::vector<ar::Case>& cases, const Probe& probe,
    const ZStats& stats, const LayerSet& set, int rep) {
  std::vector<cm::CaseTrace> traces;
  traces.reserve(cases.size());
  for (const auto& item : cases) {
    cm::CaseTrace trace;
    trace.id = item.id;
    std::vector<std::uint32_t> context = item.initialPrefix;
    std::vector<std::uint32_t> predicted;
    predicted.reserve(item.targets.size());
    bool finite = true;
    for (const std::uint32_t truth : item.targets) {
      ma::Score head;
      ma::Score probeScore2;
      try {
        head = headScore(config, params, context, truth);
        probeScore2 = probeScore(config, params, probe, stats, set, rep,
                                 context, truth);
      } catch (const std::exception&) {
        finite = false;
        break;
      }
      trace.margins.push_back(probeScore2.expectedMinusTop1Margin);
      trace.autoregressiveNllSum += probeScore2.tokenNll;
      predicted.push_back(probeScore2.predicted);
      context = slideContext(context, head.predicted);
    }
    if (!finite) {
      trace.finite = false;
      traces.push_back(std::move(trace));
      continue;
    }
    const auto firstError = ma::firstErrorInfo(predicted, item.targets);
    trace.tokenExact = static_cast<std::uint64_t>(predicted.size()) -
                       firstError.wrongCount;
    trace.sequenceExact = firstError.wrongCount == 0;
    trace.firstErrorPosition =
        firstError.firstError < 0 ? -1 : firstError.firstError + 1;
    trace.finite = true;
    traces.push_back(std::move(trace));
  }
  return cm::summarizeCheckpoint(step, std::move(traces));
}

// ---------------------------------------------------------------------------
// Token-level aggregate metrics over teacher-forced rows (head or probe)
// ---------------------------------------------------------------------------
struct TokenMetrics {
  std::uint64_t tokenExact = 0;
  std::uint64_t total = 0;
  double meanRank = 0.0;
  double meanNll = 0.0;
  double meanMargin = 0.0;
  double marginQ10 = 0.0;
  double top2 = 0.0;
  double top3 = 0.0;
  double meanEntropy = 0.0;
  bool finite = true;
};

inline TokenMetrics aggregateTokenMetrics(
    const std::vector<ma::Score>& scores,
    const std::vector<ProbeRow>& rows = {}) {
  TokenMetrics metrics;
  metrics.total = scores.size();
  std::vector<double> margins;
  margins.reserve(scores.size());
  for (std::size_t i = 0; i < scores.size(); ++i) {
    const auto& score = scores[i];
    if (!score.valid || !std::isfinite(score.tokenNll)) {
      metrics.finite = false;
      continue;
    }
    if (!rows.empty()) {
      if (score.predicted == rows[i].truth) ++metrics.tokenExact;
    } else {
      ++metrics.tokenExact;
    }
    metrics.meanRank += score.expectedRank;
    metrics.meanNll += score.tokenNll;
    metrics.meanMargin += score.expectedMinusTop1Margin;
    metrics.meanEntropy += score.entropy;
    margins.push_back(score.expectedMinusTop1Margin);
    const double rank = score.expectedRank;
    metrics.top2 += rank <= 2.0 + 1e-9 ? 1.0 : 0.0;
    metrics.top3 += rank <= 3.0 + 1e-9 ? 1.0 : 0.0;
  }
  const double n = static_cast<double>(scores.size());
  if (n > 0.0) {
    metrics.meanRank /= n;
    metrics.meanNll /= n;
    metrics.meanMargin /= n;
    metrics.meanEntropy /= n;
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

inline std::vector<ma::Score> headRowScores(
    const tiny::Config& config, const train::P& params,
    const std::vector<ProbeRow>& rows) {
  std::vector<ma::Score> scores;
  scores.reserve(rows.size());
  for (const auto& row : rows)
    scores.push_back(headScore(config, params, row.context, row.truth));
  return scores;
}

inline TokenMetrics headTokenMetrics(const tiny::Config& config,
                                     const train::P& params,
                                     const std::vector<ProbeRow>& rows) {
  return aggregateTokenMetrics(headRowScores(config, params, rows), rows);
}

inline std::vector<ma::Score> probeRowScores(
    const Probe& probe, const ZStats& stats, const LayerSet& set, int rep,
    const std::vector<ProbeRow>& rows, std::size_t rowOffset = 0) {
  std::vector<ma::Score> scores;
  scores.reserve(rows.size());
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  std::vector<double> logits(kClasses);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    probeForward(probe, stats,
                 f.data() + (rowOffset + r) * static_cast<std::size_t>(set.dim),
                 logits.data());
    const auto p = softmaxRow(logits.data(), probe.classes);
    scores.push_back(stableScoreFromLogits(logits, p, rows[r].truth));
  }
  return scores;
}

inline TokenMetrics probeTokenMetrics(const Probe& probe, const ZStats& stats,
                                      const LayerSet& set, int rep,
                                      const std::vector<ProbeRow>& rows,
                                      std::size_t rowOffset = 0) {
  return aggregateTokenMetrics(
      probeRowScores(probe, stats, set, rep, rows, rowOffset), rows);
}

// ---------------------------------------------------------------------------
// Head-only retraining (host-only; production training code untouched)
// Candidates:
//   A: warm start from the checkpoint head, train outputProjection only
//   B: re-init outputProjection via initialParameters(config, seed), train it
//   C: bias-only: frozen head logits + learned bias vector
// Optimizer: Adam lr 0.003 LEGACY, fresh m/v, batches formalBatch((step-1)%4)
// Selection: min calibration CE over canonical 23-point cadence, tie-break
// higher calibration token exact, then earlier step (tol 1e-7).
// ---------------------------------------------------------------------------
enum class HeadCandidate { kWarmStart = 0, kReinit = 1, kBiasOnly = 2 };

inline const char* headCandidateName(HeadCandidate candidate) {
  switch (candidate) {
    case HeadCandidate::kWarmStart: return "A_WARM_START";
    case HeadCandidate::kReinit: return "B_REINIT";
    case HeadCandidate::kBiasOnly: return "C_BIAS_ONLY";
  }
  return "UNKNOWN";
}

struct HeadGridPoint {
  int step = 0;
  double trainCe = 0.0;
  double calCe = 0.0;
  std::uint64_t trainExact = 0;
  std::uint64_t calExact = 0;
};

struct HeadTrainResult {
  HeadCandidate candidate = HeadCandidate::kWarmStart;
  bool finite = true;
  int nonfiniteStep = -1;
  std::string nonfiniteWhat;
  int selectedStep = 0;
  train::P trained;  // A/B: params with trained head; C: frozen params
  std::vector<double> bias;  // C only
  bool frozenUnchanged = true;
  std::vector<HeadGridPoint> grid;
  // Primary metrics at the selected step:
  double trainCe = 0.0;
  std::uint64_t trainExact = 0;
  double calCe = 0.0;
  std::uint64_t calExact = 0;
  TokenMetrics trainTf, calTf, devTf;
  cm::CheckpointMetrics calFr, devFr;
  // Supplementary at step 320 (non-selecting):
  TokenMetrics devTf320;
  cm::CheckpointMetrics devFr320;
};

inline bool parametersEqual(const train::P& a, const train::P& b) {
  const auto ra = tiny::parameterRegistry(a);
  const auto rb = tiny::parameterRegistry(b);
  if (ra.size() != rb.size()) return false;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    if (ra[i].name != rb[i].name) return false;
    if (ra[i].values->size() != rb[i].values->size()) return false;
    for (std::size_t j = 0; j < ra[i].values->size(); ++j)
      if ((*ra[i].values)[j] != (*rb[i].values)[j]) return false;
  }
  return true;
}

// Restrict a full gradient to the output projection only.
inline void zeroAllButOutputProjection(train::P& gradients) {
  for (auto& e : tiny::parameterRegistry(gradients)) {
    if (e.name == "output_projection") continue;
    auto* values = const_cast<std::vector<float>*>(e.values);
    std::fill(values->begin(), values->end(), 0.0f);
  }
}

inline void setHeadFrom(train::P& target, const train::P& source) {
  target.outputProjection = source.outputProjection;
}

inline std::vector<double> headLogitsFromForward(
    const tiny::Config& config, const train::P& params,
    const std::vector<std::uint32_t>& context) {
  const auto oh = tiny::oneHot(context, config.vocabularySize);
  const train::GF g = train::generalForward(config, oh, params);
  const std::size_t base =
      std::size_t(config.tokens - 1) * config.vocabularySize;
  std::vector<double> logits(config.vocabularySize);
  for (std::uint32_t j = 0; j < config.vocabularySize; ++j)
    logits[j] = static_cast<double>(g.logits[base + j]);
  return logits;
}

// Logits with an added bias vector (candidate C).
inline ma::Score biasScoreFromLogits(const std::vector<double>& logits,
                                     const std::vector<double>& bias,
                                     std::uint32_t truth) {
  std::vector<double> shifted(logits.size());
  for (std::size_t j = 0; j < logits.size(); ++j)
    shifted[j] = logits[j] + bias[j];
  const auto p = softmaxRow(shifted.data(), static_cast<int>(shifted.size()));
  if (!std::isfinite(shifted[truth]))
    throw std::runtime_error("NON_FINITE_BIAS_SCORE");
  return stableScoreFromLogits(shifted, p, truth);
}

inline TokenMetrics tokenMetricsWithBias(const tiny::Config& config,
                                         const train::P& params,
                                         const std::vector<double>& bias,
                                         const std::vector<ProbeRow>& rows) {
  std::vector<ma::Score> scores;
  scores.reserve(rows.size());
  for (const auto& row : rows) {
    const auto logits = headLogitsFromForward(config, params, row.context);
    scores.push_back(biasScoreFromLogits(logits, bias, row.truth));
  }
  return aggregateTokenMetrics(scores, rows);
}

inline cm::CheckpointMetrics freeRunningBias(
    const tiny::Config& config, const train::P& params,
    const std::vector<double>& bias, int step,
    const std::vector<ar::Case>& cases) {
  return freeRunningRollout(
      cases, step,
      [&](const std::vector<std::uint32_t>& context, std::uint32_t truth) {
        const auto logits = headLogitsFromForward(config, params, context);
        return biasScoreFromLogits(logits, bias, truth);
      });
}

inline HeadTrainResult retrainHead(const tiny::Config& config,
                                   std::uint32_t seed, const train::P& base,
                                   HeadCandidate candidate,
                                   const std::vector<ProbeRow>& trainRows,
                                   const std::vector<ProbeRow>& calRows,
                                   const std::vector<ProbeRow>& devRows,
                                   const std::vector<ar::Case>& calCases,
                                   const std::vector<ar::Case>& devCases) {
  HeadTrainResult result;
  result.candidate = candidate;
  const train::P frozen = base;
  train::P params = base;
  std::vector<double> bias(kClasses, 0.0);
  if (candidate == HeadCandidate::kReinit) {
    const auto fresh = tiny::initialParameters(config, seed);
    setHeadFrom(params, fresh);
  }

  train::P m = params;
  train::P v = params;
  for (const auto& e : tiny::parameterRegistry(m))
    std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
              const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
  for (const auto& e : tiny::parameterRegistry(v))
    std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
              const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
  std::vector<double> mb(kClasses, 0.0), vb(kClasses, 0.0);
  const bool biasOnly = candidate == HeadCandidate::kBiasOnly;
  if (biasOnly) {
    // m/v remain zero-size placeholders; only bias is trained.
  }

  const auto ceAndExact = [&](const std::vector<ProbeRow>& rows,
                              double& ce, std::uint64_t& exact) {
    ce = 0.0;
    exact = 0;
    double loss = 0.0;
    for (const auto& row : rows) {
      std::vector<double> logits;
      if (biasOnly) {
        logits = headLogitsFromForward(config, params, row.context);
        for (std::size_t j = 0; j < logits.size(); ++j)
          logits[j] += bias[j];
      } else {
        logits = headLogitsFromForward(config, params, row.context);
      }
      for (const double l : logits)
        if (!std::isfinite(l)) {
          ce = std::numeric_limits<double>::infinity();
          return;
        }
      loss += ceFromLogits(logits.data(), static_cast<int>(logits.size()),
                           row.truth);
      exact += ma::argmaxFirst(logits) == row.truth ? 1u : 0u;
    }
    ce = loss / static_cast<double>(rows.size());
  };

  // Grid point at step 0.
  {
    HeadGridPoint point;
    point.step = 0;
    ceAndExact(trainRows, point.trainCe, point.trainExact);
    ceAndExact(calRows, point.calCe, point.calExact);
    result.grid.push_back(point);
  }

  for (int step = 1; step <= kHeadSteps; ++step) {
    const std::uint32_t pattern = std::uint32_t((step - 1) % 4);
    const auto batch = dq::formalBatch(config, pattern, 0);
    if (biasOnly) {
      // Full-batch CE gradient w.r.t. bias: dL/db[c] = sum_rows (p[c] - y[c])/N
      std::vector<double> gradB(kClasses, 0.0);
      const double invN = 1.0 / static_cast<double>(trainRows.size());
      for (const auto& row : trainRows) {
        const auto logits = headLogitsFromForward(config, params, row.context);
        std::vector<double> shifted(logits.size());
        for (std::size_t j = 0; j < logits.size(); ++j)
          shifted[j] = logits[j] + bias[j];
        const auto p = softmaxRow(shifted.data(), static_cast<int>(shifted.size()));
        for (std::uint32_t c = 0; c < kClasses; ++c)
          gradB[c] += (p[c] - (c == row.truth ? 1.0 : 0.0)) * invN;
      }
      const double c1 = 1.0 / (1.0 - std::pow(0.9, static_cast<double>(step)));
      const double c2 = 1.0 / (1.0 - std::pow(0.999, static_cast<double>(step)));
      for (std::uint32_t c = 0; c < kClasses; ++c) {
        mb[c] = 0.9 * mb[c] + 0.1 * gradB[c];
        vb[c] = 0.999 * vb[c] + 0.001 * gradB[c] * gradB[c];
        bias[c] =
            bias[c] - kHeadLr * (mb[c] * c1) / (std::sqrt(vb[c] * c2) + 1e-8);
        if (!std::isfinite(bias[c])) {
          result.finite = false;
          result.nonfiniteStep = step;
          result.nonfiniteWhat = "bias_adam";
          return result;
        }
      }
    } else {
      const auto fb =
          tiny::forwardBackward(config, batch.first, batch.second, params, 0.0f);
      if (!dq::finiteTensor(fb.logits, config.tokens * config.vocabularySize,
                            nullptr)) {
        result.finite = false;
        result.nonfiniteStep = step;
        result.nonfiniteWhat = "forward_nonfinite";
        return result;
      }
      train::P gradients = fb.gradients;
      zeroAllButOutputProjection(gradients);
      const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
      const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
      const float lr = phonelm::stabilityLearningRate(
          std::uint32_t(dq::StabilityMode::LEGACY), kHeadLr,
          std::uint32_t(step), std::uint32_t(kHeadSteps));
      const auto update =
          tiny::adamUpdate(params, gradients, m, v, lr, .9f, .999f, 1e-8f, c1, c2);
      params = update.next;
      m = update.firstMoment;
      v = update.secondMoment;
      for (const auto& e : tiny::parameterRegistry(params)) {
        if (e.name == "output_projection") continue;
        for (const float x : *e.values)
          if (!std::isfinite(x)) {
            result.finite = false;
            result.nonfiniteStep = step;
            result.nonfiniteWhat = "param_nonfinite";
            return result;
          }
      }
    }
    if (step % 4 == 0 || step == kHeadSteps) {
      HeadGridPoint point;
      point.step = step;
      ceAndExact(trainRows, point.trainCe, point.trainExact);
      ceAndExact(calRows, point.calCe, point.calExact);
      result.grid.push_back(point);
    }
  }

  // Selection: min cal CE -> higher cal exact -> earlier step.
  const HeadGridPoint* best = nullptr;
  for (const auto& point : result.grid) {
    if (!best) {
      best = &point;
      continue;
    }
    const double delta = point.calCe - best->calCe;
    bool better = false;
    if (delta < -kTieTolerance) {
      better = true;
    } else if (std::abs(delta) <= kTieTolerance) {
      if (point.calExact > best->calExact) better = true;
      else if (point.calExact == best->calExact && point.step < best->step)
        better = true;
    }
    if (better) best = &point;
  }
  if (!best) throw std::runtime_error("HEAD_EMPTY_GRID");
  result.selectedStep = best->step;
  result.trainCe = best->trainCe;
  result.trainExact = best->trainExact;
  result.calCe = best->calCe;
  result.calExact = best->calExact;

  // Keep the full-run (step 320) parameters for the supplementary eval.
  const train::P fullRunParams = params;
  const std::vector<double> fullRunBias = bias;

  // Replay to the selected step for the primary params (deterministic).
  if (best->step != kHeadSteps) {
    train::P replay = base;
    std::vector<double> rbias(kClasses, 0.0);
    if (candidate == HeadCandidate::kReinit) {
      const auto fresh = tiny::initialParameters(config, seed);
      setHeadFrom(replay, fresh);
    }
    train::P rm = replay;
    train::P rv = replay;
    for (const auto& e : tiny::parameterRegistry(rm))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
    for (const auto& e : tiny::parameterRegistry(rv))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
    std::vector<double> rmb(kClasses, 0.0), rvb(kClasses, 0.0);
    for (int step = 1; step <= best->step; ++step) {
      const std::uint32_t pattern = std::uint32_t((step - 1) % 4);
      const auto batch = dq::formalBatch(config, pattern, 0);
      if (biasOnly) {
        std::vector<double> gradB(kClasses, 0.0);
        const double invN = 1.0 / static_cast<double>(trainRows.size());
        for (const auto& row : trainRows) {
          const auto logits = headLogitsFromForward(config, replay, row.context);
          std::vector<double> shifted(logits.size());
          for (std::size_t j = 0; j < logits.size(); ++j)
            shifted[j] = logits[j] + rbias[j];
          const auto p =
              softmaxRow(shifted.data(), static_cast<int>(shifted.size()));
          for (std::uint32_t c = 0; c < kClasses; ++c)
            gradB[c] += (p[c] - (c == row.truth ? 1.0 : 0.0)) * invN;
        }
        const double c1 =
            1.0 / (1.0 - std::pow(0.9, static_cast<double>(step)));
        const double c2 =
            1.0 / (1.0 - std::pow(0.999, static_cast<double>(step)));
        for (std::uint32_t c = 0; c < kClasses; ++c) {
          rmb[c] = 0.9 * rmb[c] + 0.1 * gradB[c];
          rvb[c] = 0.999 * rvb[c] + 0.001 * gradB[c] * gradB[c];
          rbias[c] =
              rbias[c] - kHeadLr * (rmb[c] * c1) / (std::sqrt(rvb[c] * c2) + 1e-8);
        }
      } else {
        const auto fb =
            tiny::forwardBackward(config, batch.first, batch.second, replay, 0.0f);
        train::P gradients = fb.gradients;
        zeroAllButOutputProjection(gradients);
        const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
        const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
        const float lr = phonelm::stabilityLearningRate(
            std::uint32_t(dq::StabilityMode::LEGACY), kHeadLr,
            std::uint32_t(step), std::uint32_t(kHeadSteps));
        const auto update =
            tiny::adamUpdate(replay, gradients, rm, rv, lr, .9f, .999f, 1e-8f, c1, c2);
        replay = update.next;
        rm = update.firstMoment;
        rv = update.secondMoment;
      }
    }
    params = replay;
    bias = rbias;
  }
  result.trained = params;
  result.bias = bias;

  // Freeze verification.
  if (biasOnly) {
    result.frozenUnchanged = parametersEqual(params, frozen);
  } else {
    train::P a = params;
    train::P b = frozen;
    a.outputProjection = b.outputProjection;
    result.frozenUnchanged = parametersEqual(a, b);
  }

  // Primary evaluation at the selected step (replayed params).
  result.trainTf =
      biasOnly ? tokenMetricsWithBias(config, params, bias, trainRows)
               : headTokenMetrics(config, params, trainRows);
  result.calTf = biasOnly ? tokenMetricsWithBias(config, params, bias, calRows)
                          : headTokenMetrics(config, params, calRows);
  result.devTf = biasOnly ? tokenMetricsWithBias(config, params, bias, devRows)
                          : headTokenMetrics(config, params, devRows);
  result.calFr =
      biasOnly
          ? freeRunningBias(config, params, bias, result.selectedStep, calCases)
          : headFreeRunning(config, params, result.selectedStep, calCases);
  result.devFr =
      biasOnly
          ? freeRunningBias(config, params, bias, result.selectedStep, devCases)
          : headFreeRunning(config, params, result.selectedStep, devCases);
  // Supplementary at step 320 (full-run params; non-selecting).
  result.devTf320 = biasOnly
                        ? tokenMetricsWithBias(config, fullRunParams, fullRunBias,
                                               devRows)
                        : headTokenMetrics(config, fullRunParams, devRows);
  result.devFr320 =
      biasOnly
          ? freeRunningBias(config, fullRunParams, fullRunBias, kHeadSteps,
                            devCases)
          : headFreeRunning(config, fullRunParams, kHeadSteps, devCases);
  return result;
}

// ---------------------------------------------------------------------------
// Representation metrics (dev TF rows by default; TRAIN rows for gap)
// ---------------------------------------------------------------------------
struct RepMetrics {
  double eta2 = 0.0;             // tr(SB)/tr(ST) over class means
  double effectiveRank = 0.0;    // participation ratio of ST eigenvalues
  double normRatio = 0.0;        // mean||h|| / mean||h_embedded||
  double hiddenMarginMidmedian = 0.0;  // midmedian of (d_wrong - d_correct)
  double signAgreement = 0.0;    // fraction sign(hidden margin)==sign(head margin)
  double alignmentCosine = 0.0;  // mean cos(h, head row of truth class)
};

// Deterministic cyclic Jacobi (double, 16x16) with a fixed sweep count.
inline std::vector<double> symmetricEigenvalues(
    const std::vector<std::vector<double>>& matrix) {
  const int n = static_cast<int>(matrix.size());
  std::vector<std::vector<double>> a = matrix;
  for (int sweep = 0; sweep < 24; ++sweep) {
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
        for (int k = 0; k < n; ++k) {
          const double akp = a[k][p];
          const double akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = a[p][k];
          const double aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  std::vector<double> values(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    double v = a[i][i];
    if (std::abs(v) < 1e-15) v = 0.0;
    values[static_cast<std::size_t>(i)] = v;
  }
  std::sort(values.begin(), values.end(), std::greater<double>());
  return values;
}

inline double participationRatio(const std::vector<double>& eigenvalues) {
  double sum = 0.0, sq = 0.0;
  for (const double x : eigenvalues) {
    sum += x;
    sq += x * x;
  }
  return sq > 0.0 ? sum * sum / sq : 0.0;
}

inline double midMedian(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline RepMetrics computeRepMetrics(
    const LayerSet& set, int rep, const std::vector<ProbeRow>& rows,
    const std::vector<double>& embeddedNorms,
    const std::vector<ma::Score>& headScores,
    const std::vector<std::vector<double>>& classMeans,
    std::size_t rowOffset = 0) {
  RepMetrics result;
  const int dim = set.dim;
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  const std::size_t n = rows.size();
  if (n == 0) return result;

  // Scatter matrices (double).
  std::vector<double> st(static_cast<std::size_t>(dim) * dim, 0.0);
  std::vector<double> globalMean(static_cast<std::size_t>(dim), 0.0);
  for (std::size_t r = 0; r < n; ++r)
    for (int d = 0; d < dim; ++d)
      globalMean[static_cast<std::size_t>(d)] +=
          static_cast<double>(f[(rowOffset + r) * dim + d]);
  for (int d = 0; d < dim; ++d)
    globalMean[static_cast<std::size_t>(d)] /= static_cast<double>(n);
  for (std::size_t r = 0; r < n; ++r) {
    for (int a = 0; a < dim; ++a) {
      const double za =
          static_cast<double>(f[(rowOffset + r) * dim + a]) -
          globalMean[static_cast<std::size_t>(a)];
      for (int b = 0; b < dim; ++b) {
        const double zb =
            static_cast<double>(f[(rowOffset + r) * dim + b]) -
            globalMean[static_cast<std::size_t>(b)];
        st[static_cast<std::size_t>(a) * dim + b] += za * zb;
      }
    }
  }
  std::vector<double> sb(static_cast<std::size_t>(dim) * dim, 0.0);
  std::map<std::uint32_t, std::size_t> classCounts;
  for (const auto& row : rows) ++classCounts[row.truth];
  for (const auto& entry : classCounts) {
    const std::uint32_t cls = entry.first;
    const std::size_t count = entry.second;
    const auto& mean = classMeans[cls];
    for (int a = 0; a < dim; ++a) {
      const double za = mean[static_cast<std::size_t>(a)] -
                        globalMean[static_cast<std::size_t>(a)];
      for (int b = 0; b < dim; ++b) {
        const double zb = mean[static_cast<std::size_t>(b)] -
                          globalMean[static_cast<std::size_t>(b)];
        sb[static_cast<std::size_t>(a) * dim + b] +=
            static_cast<double>(count) * za * zb;
      }
    }
  }
  double trSt = 0.0, trSb = 0.0;
  for (int d = 0; d < dim; ++d) {
    trSt += st[static_cast<std::size_t>(d) * dim + d];
    trSb += sb[static_cast<std::size_t>(d) * dim + d];
  }
  result.eta2 = trSt > 0.0 ? trSb / trSt : 0.0;

  // Effective rank from ST eigenvalues.
  std::vector<std::vector<double>> stMat(
      static_cast<std::size_t>(dim), std::vector<double>(static_cast<std::size_t>(dim)));
  for (int a = 0; a < dim; ++a)
    for (int b = 0; b < dim; ++b)
      stMat[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] =
          st[static_cast<std::size_t>(a) * dim + b];
  result.effectiveRank =
      participationRatio(symmetricEigenvalues(stMat));

  // Norm ratio.
  double normSum = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    double sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double x = static_cast<double>(f[(rowOffset + r) * dim + d]);
      sq += x * x;
    }
    normSum += std::sqrt(sq);
  }
  double embeddedMean = 0.0;
  for (const double x : embeddedNorms) embeddedMean += x;
  embeddedMean /= static_cast<double>(embeddedNorms.size());
  result.normRatio = embeddedMean > 0.0
                         ? (normSum / static_cast<double>(n)) / embeddedMean
                         : 0.0;

  // Hidden margin: nearest-wrong-class centroid distance minus correct-class.
  std::vector<double> hiddenMargins;
  hiddenMargins.reserve(n);
  std::uint64_t signAgree = 0;
  std::uint64_t signTotal = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const std::uint32_t truth = rows[r].truth;
    const float* h = f.data() + (rowOffset + r) * dim;
    double dCorrect = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double z = static_cast<double>(h[d]) - classMeans[truth][static_cast<std::size_t>(d)];
      dCorrect += z * z;
    }
    dCorrect = std::sqrt(dCorrect);
    double dWrong = std::numeric_limits<double>::infinity();
    for (const auto& entry : classCounts) {
      if (entry.first == truth) continue;
      const auto& mean = classMeans[entry.first];
      double acc = 0.0;
      for (int d = 0; d < dim; ++d) {
        const double z = static_cast<double>(h[d]) - mean[static_cast<std::size_t>(d)];
        acc += z * z;
      }
      dWrong = std::min(dWrong, std::sqrt(acc));
    }
    const double margin = dWrong - dCorrect;
    hiddenMargins.push_back(margin);
    if (r < headScores.size()) {
      const double g = headScores[r].expectedMinusTop1Margin;
      const bool mPos = margin > 0.0, gPos = g > 0.0;
      const bool mNeg = margin < 0.0, gNeg = g < 0.0;
      if ((mPos && gPos) || (mNeg && gNeg)) ++signAgree;
      ++signTotal;
    }
  }
  result.hiddenMarginMidmedian = midMedian(std::move(hiddenMargins));
  result.signAgreement = signTotal > 0
                             ? static_cast<double>(signAgree) /
                                   static_cast<double>(signTotal)
                             : 0.0;
  return result;
}

// Class-mean / head-alignment helpers shared by the runner.
inline std::vector<std::vector<double>> classMeans(
    const LayerSet& set, int rep, const std::vector<ProbeRow>& rows,
    std::size_t rowOffset = 0) {
  std::vector<std::vector<double>> means(kClasses,
                                         std::vector<double>(set.dim, 0.0));
  std::vector<std::size_t> counts(kClasses, 0);
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const std::uint32_t cls = rows[r].truth;
    ++counts[cls];
    for (int d = 0; d < set.dim; ++d)
      means[cls][static_cast<std::size_t>(d)] +=
          static_cast<double>(f[(rowOffset + r) * set.dim + d]);
  }
  for (std::uint32_t c = 0; c < kClasses; ++c)
    if (counts[c] > 0)
      for (int d = 0; d < set.dim; ++d)
        means[c][static_cast<std::size_t>(d)] /=
            static_cast<double>(counts[c]);
  return means;
}

inline std::vector<double> embeddedNorms(const LayerSet& set,
                                         std::size_t rowBegin,
                                         std::size_t rowEnd) {
  std::vector<double> norms(rowEnd - rowBegin, 0.0);
  const auto& f = set.features[0];
  for (std::size_t r = rowBegin; r < rowEnd; ++r) {
    double sq = 0.0;
    for (int d = 0; d < set.dim; ++d) {
      const double x = static_cast<double>(f[r * set.dim + d]);
      sq += x * x;
    }
    norms[r - rowBegin] = std::sqrt(sq);
  }
  return norms;
}

inline std::vector<std::vector<double>> headClassVectors(
    const train::P& params, int dim, int classes) {
  // outputProjection is [dim x vocab] row-major; class c vector = column c.
  std::vector<std::vector<double>> rows(
      static_cast<std::size_t>(classes), std::vector<double>(static_cast<std::size_t>(dim)));
  for (int d = 0; d < dim; ++d)
    for (int c = 0; c < classes; ++c)
      rows[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] =
          static_cast<double>(
              params.outputProjection[static_cast<std::size_t>(d) * classes + c]);
  return rows;
}

inline double meanAlignmentCosine(
    const LayerSet& set, int rep, const std::vector<ProbeRow>& rows,
    const std::vector<std::vector<double>>& headVectors,
    std::size_t rowOffset = 0) {
  const auto& f = set.features.at(static_cast<std::size_t>(rep));
  double sum = 0.0;
  std::size_t total = 0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto& w = headVectors[rows[r].truth];
    double hd = 0.0, wd = 0.0, dot = 0.0;
    for (int d = 0; d < set.dim; ++d) {
      const double h =
          static_cast<double>(f[(rowOffset + r) * set.dim + d]);
      dot += h * w[static_cast<std::size_t>(d)];
      hd += h * h;
      wd += w[static_cast<std::size_t>(d)] * w[static_cast<std::size_t>(d)];
    }
    const double hh = std::sqrt(hd), ww = std::sqrt(wd);
    if (hh > 0.0 && ww > 0.0) {
      sum += dot / (hh * ww);
      ++total;
    }
  }
  return total > 0 ? sum / static_cast<double>(total) : 0.0;
}

// ---------------------------------------------------------------------------
// Hidden-state cache (private, never committed)
// ---------------------------------------------------------------------------
struct CacheIdentity {
  std::string protocol;
  std::string config;
  std::uint32_t seed = 0;
  int step = 0;
  std::string datasetHash;
  int depth = 0;
  int repCount = 0;
  int dim = 0;
  std::size_t rows = 0;
  std::string paramHash;
};

inline std::string fnv1aParams(const train::P& params) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto& e : tiny::parameterRegistry(params))
    for (const float x : *e.values) {
      hash ^= static_cast<std::uint8_t>(x == 0.0f ? 0 : 1);
      hash *= 1099511628211ull;
    }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << hash;
  return output.str();
}

inline std::string cacheFileName(const std::string& config,
                                 std::uint32_t seed, int step,
                                 const std::string& datasetHash) {
  std::ostringstream output;
  output << config << "_s" << seed << "_step" << step << "_"
         << datasetHash.substr(8, 16) << ".bin";
  std::string name = output.str();
  // datasetHash contains ':' (and ';'/'=') — invalid in Windows file names;
  // an unsanitized ':' would silently create an NTFS alternate data stream
  // with a 0-byte base file. Keep the cache file a normal file.
  for (char& ch : name)
    if (ch == ':' || ch == ';' || ch == '=') ch = '_';
  return name;
}

inline bool writeHiddenCache(const std::filesystem::path& dir,
                             const CacheIdentity& identity,
                             const LayerSet& set) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return false;
  const auto file = dir / cacheFileName(identity.config, identity.seed,
                                        identity.step, identity.datasetHash);
  std::ofstream out(file, std::ios::binary);
  if (!out) return false;
  const std::uint64_t magic = 0x524550524F424531ull;  // "REPROBE1"
  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  char paramHash[33] = {};
  std::memcpy(paramHash, identity.paramHash.c_str(),
              std::min<std::size_t>(32, identity.paramHash.size()));
  out.write(paramHash, 32);
  const std::uint32_t dim = static_cast<std::uint32_t>(set.dim);
  const std::uint32_t repCount = static_cast<std::uint32_t>(set.repCount);
  const std::uint64_t rows = static_cast<std::uint64_t>(set.features[0].size() / set.dim);
  out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char*>(&repCount), sizeof(repCount));
  out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  for (const auto& rep : set.features)
    out.write(reinterpret_cast<const char*>(rep.data()),
              static_cast<std::streamsize>(rep.size() * sizeof(float)));
  out.close();
  return static_cast<bool>(out);
}

inline bool readHiddenCache(const std::filesystem::path& dir,
                            const CacheIdentity& identity, LayerSet& set) {
  const auto file = dir / cacheFileName(identity.config, identity.seed,
                                        identity.step, identity.datasetHash);
  std::ifstream in(file, std::ios::binary);
  if (!in) return false;
  std::uint64_t magic = 0;
  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != 0x524550524F424531ull) return false;
  char paramHash[33] = {};
  in.read(paramHash, 32);
  const std::string stored(paramHash);
  if (stored != identity.paramHash) return false;
  std::uint32_t dim = 0, repCount = 0;
  std::uint64_t rows = 0;
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  in.read(reinterpret_cast<char*>(&repCount), sizeof(repCount));
  in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  if (dim != static_cast<std::uint32_t>(identity.dim) ||
      repCount != static_cast<std::uint32_t>(identity.repCount) ||
      rows != identity.rows)
    return false;
  set.depth = identity.depth;
  set.repCount = static_cast<int>(repCount);
  set.dim = static_cast<int>(dim);
  set.features.assign(
      static_cast<std::size_t>(repCount),
      std::vector<float>(static_cast<std::size_t>(rows) * dim, 0.0f));
  for (auto& rep : set.features)
    in.read(reinterpret_cast<char*>(rep.data()),
            static_cast<std::streamsize>(rep.size() * sizeof(float)));
  if (!in) return false;
  return true;
}

// ---------------------------------------------------------------------------
// CSV output helper
// ---------------------------------------------------------------------------
class CsvWriter {
 public:
  explicit CsvWriter(const std::filesystem::path& path) : file_(path) {
    if (!file_) throw std::runtime_error("CSV_OPEN_FAILED: " + path.string());
  }
  void header(const std::vector<std::string>& names) { writeRow(names); }
  void row(const std::vector<std::string>& values) { writeRow(values); }
  void row(const std::vector<std::string>& names,
           const std::vector<std::string>& values) {
    (void)names;
    writeRow(values);
  }
  bool good() const { return static_cast<bool>(file_); }

 private:
  void writeRow(const std::vector<std::string>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i > 0) file_ << ',';
      const std::string& v = values[i];
      const bool quote = v.find_first_of(",\n\"") != std::string::npos;
      if (quote) {
        file_ << '"';
        for (const char ch : v) {
          if (ch == '"') file_ << '"';  // escape
          file_ << ch;
        }
        file_ << '"';
      } else {
        file_ << v;
      }
    }
    file_ << '\n';
  }
  std::ofstream file_;
};

inline std::string text(double value, int precision = 12) {
  if (!std::isfinite(value)) return "nan";
  std::ostringstream output;
  output << std::setprecision(precision) << value;
  return output.str();
}

inline std::string text(std::uint64_t value) {
  return std::to_string(value);
}

inline std::string text(std::int64_t value) {
  return std::to_string(value);
}

inline std::string text(int value) { return std::to_string(value); }

inline std::string text(bool value) { return value ? "true" : "false"; }

}  // namespace phonelm::readout_probe

#endif  // READOUT_PROBE_LIB_H
