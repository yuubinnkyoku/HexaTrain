// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Depth-quality diagnostics shared library (host-only tooling).
//
// Replicates the established CPU formal protocol of the on-device generic
// trainer so diagnostics run without a device:
//   batches: pattern=(step-1)%4, phase=0 (matches the POST_FIX formal path)
//   Adam: beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.003, bias corrections
//         recomputed per step, exactly as tiny_lm::adamUpdate receives them.
// Forward internals (attention scores, layer activations) are captured by a
// structurally identical reimplementation whose logits are asserted bitwise
// equal to tiny_lm::forwardBackward before any metric is trusted.
#pragma once
#include "tiny_language_model_cpu.h"
#include "training_stability.h"
#include "validation_selection.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::depth_quality {

using Config = tiny_lm::Config;
using Params = qnn::TinyTransformerParameters;
using LayerParams = qnn::TinyTransformerLayerParameters;

// ---------------------------------------------------------------------------
// Batches (mirrors languageBatch in qnn_transformer_training.cpp)
// ---------------------------------------------------------------------------
inline std::pair<std::vector<float>, std::vector<float>> formalBatch(
    const Config& config, uint32_t patternIndex, uint32_t phase = 0) {
  static const std::vector<std::vector<uint32_t>> patterns{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}, {10, 11, 12}};
  const auto& pattern = patterns.at(patternIndex % patterns.size());
  std::vector<uint32_t> input(config.tokens), target(config.tokens);
  for (uint32_t i = 0; i < config.tokens; ++i) {
    input[i] = pattern[(i + phase) % pattern.size()];
    target[i] = pattern[(i + phase + 1) % pattern.size()];
  }
  return {tiny_lm::oneHot(input, config.vocabularySize),
          tiny_lm::oneHot(target, config.vocabularySize)};
}

inline const LayerParams& layerAt(const Params& p, uint32_t i);
inline LayerParams& layerAt(Params& p, uint32_t i) {
  if (i == 0) return static_cast<LayerParams&>(p);
  return p.layers.at(i - 1);
}
inline const LayerParams& layerAt(const Params& p, uint32_t i) {
  return layerAt(const_cast<Params&>(p), i);
}

// ---------------------------------------------------------------------------
// Statistic helpers (double accumulation, stable)
// ---------------------------------------------------------------------------
struct Moments {
  double rms = 0.0, maxAbs = 0.0, mean = 0.0, variance = 0.0;
};
inline Moments moments(const std::vector<float>& v) {
  Moments m;
  if (v.empty()) return m;
  double sum = 0.0, sq = 0.0;
  for (float x : v) {
    sum += double(x);
    sq += double(x) * double(x);
    m.maxAbs = std::max(m.maxAbs, std::abs(double(x)));
  }
  m.mean = sum / v.size();
  m.variance = sq / v.size() - m.mean * m.mean;
  if (m.variance < 0) m.variance = 0;
  m.rms = std::sqrt(m.variance + m.mean * m.mean);
  return m;
}
inline double l2(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += double(x) * double(x);
  return std::sqrt(s);
}
inline std::vector<float> differences(const std::vector<float>& a,
                                      const std::vector<float>& b) {
  std::vector<float> d(std::min(a.size(), b.size()));
  for (size_t i = 0; i < d.size(); ++i) d[i] = a[i] - b[i];
  return d;
}

// ---------------------------------------------------------------------------
// Depth-pair initialization contract
// ---------------------------------------------------------------------------
// initialParameters reseeds every tensor from (seed, phase). Shared tensors
// therefore agree between any two depths that share (V, D, FFN, seed); the
// extra layer of the deeper model is seeded from its own phase window.
inline bool sharedPrefixParametersEqual(const Params& shallow,
                                        const Params& deep) {
  const auto registryShallow = tiny_lm::parameterRegistry(shallow);
  const auto registryDeep = tiny_lm::parameterRegistry(deep);
  std::map<std::string, const std::vector<float>*> deepMap;
  for (const auto& e : registryDeep) deepMap[e.name] = e.values;
  size_t shared = 0, mismatched = 0;
  for (const auto& e : registryShallow) {
    auto it = deepMap.find(e.name);
    if (it == deepMap.end()) return false;
    ++shared;
    if (*it->second != *e.values) ++mismatched;
  }
  return shared > 0 && mismatched == 0;
}

// ---------------------------------------------------------------------------
// Training stability modes (explicit, LEGACY default)
// ---------------------------------------------------------------------------
// The canonical enum and helpers live in training_stability.h so device and
// host code share one definition.
using StabilityMode = phonelm::TrainingStabilityMode;
inline const char* stabilityModeName(StabilityMode mode) {
  return phonelm::trainingStabilityModeName(std::uint32_t(mode));
}
inline bool parseStabilityMode(const std::string& name, StabilityMode& mode) {
  for (std::uint32_t candidate = 0; candidate <= 6; ++candidate) {
    if (name == phonelm::trainingStabilityModeName(candidate)) {
      mode = StabilityMode(candidate);
      return true;
    }
  }
  return false;
}
inline Params applyInitStability(const Config& config, Params p,
                                 StabilityMode mode) {
  return phonelm::applyInitStability(config, std::move(p), std::uint32_t(mode));
}

// ---------------------------------------------------------------------------
// Instrumented forward (parity-checked against the CPU reference)
// ---------------------------------------------------------------------------
struct NormStats {
  double inputVariance = 0.0, centeredRms = 0.0, centeredMaxAbs = 0.0,
         normalizedRms = 0.0, normalizedMaxAbs = 0.0, gammaNorm = 0.0,
         betaNorm = 0.0;
};
struct AttentionStats {
  double scoreRms = 0.0, scoreMaxAbs = 0.0;
  // Per head, in [layer,head] order
  std::vector<double> headEntropyMean, headMaxProbability,
      headMinNonzeroProbability;
  double outputNorm = 0.0, residualInputNorm = 0.0, branchToInputRatio = 0.0;
};
struct FfnStats {
  double w1OutputRms = 0.0, w1OutputMaxAbs = 0.0, activationZeroFraction = 0.0,
         activationRms = 0.0, w2OutputNorm = 0.0, residualInputNorm = 0.0,
         branchToInputRatio = 0.0;
};
struct LayerMetrics {
  double inputRms = 0.0, inputMaxAbs = 0.0;
  NormStats n1, n2;
  AttentionStats attention;
  double residual1Rms = 0.0, residual1MaxAbs = 0.0;
  FfnStats ffn;
  double outputRms = 0.0, outputMaxAbs = 0.0;
};
struct ReplayForward {
  std::vector<float> logits, probabilities;
  float loss = 0.0f, accuracy = 0.0f;
  std::vector<LayerMetrics> layers;
};

namespace detail {
inline std::vector<float> mm(const std::vector<float>& a,
                             const std::vector<float>& b, uint32_t r, uint32_t k,
                             uint32_t c) {
  std::vector<float> o(size_t(r) * c);
  for (uint32_t i = 0; i < r; ++i)
    for (uint32_t j = 0; j < c; ++j) {
      double s = 0;
      for (uint32_t z = 0; z < k; ++z) s += double(a[size_t(i) * k + z]) * b[size_t(z) * c + j];
      o[size_t(i) * c + j] = float(s);
    }
  return o;
}
inline void add(std::vector<float>& a, const std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}
}  // namespace detail

// LayerNorm capture for one (tokens, dimension) buffer.
inline std::vector<float> layerNormWithStats(const Config& c,
                                             const std::vector<float>& x,
                                             const std::vector<float>& gamma,
                                             const std::vector<float>& beta,
                                             NormStats& stats) {
  std::vector<float> out(x.size());
  std::vector<float> xhat(x.size());
  double varSum = 0;
  for (uint32_t r = 0; r < c.tokens; ++r) {
    double mean = 0, var = 0;
    for (uint32_t d = 0; d < c.dimension; ++d)
      mean += x[size_t(r) * c.dimension + d];
    mean /= c.dimension;
    for (uint32_t d = 0; d < c.dimension; ++d) {
      double z = x[size_t(r) * c.dimension + d] - mean;
      var += z * z;
      stats.centeredRms += z * z;
      stats.centeredMaxAbs = std::max(stats.centeredMaxAbs, std::abs(z));
    }
    var /= c.dimension;
    varSum += var;
    const float inv = float(1 / std::sqrt(var + c.epsilon));
    for (uint32_t d = 0; d < c.dimension; ++d) {
      size_t i = size_t(r) * c.dimension + d;
      xhat[i] = (x[i] - float(mean)) * inv;
      out[i] = xhat[i] * gamma[d] + beta[d];
    }
  }
  const size_t n = x.size();
  stats.inputVariance = varSum / c.tokens;
  stats.centeredRms = std::sqrt(stats.centeredRms / n);
  stats.normalizedRms = moments(xhat).rms;
  stats.normalizedMaxAbs = moments(xhat).maxAbs;
  stats.gammaNorm = l2(gamma);
  stats.betaNorm = l2(beta);
  return out;
}

inline ReplayForward instrumentedForward(
    const Config& c, const std::vector<float>& oh,
    const std::vector<float>& target, const Params& w,
    StabilityMode stability = StabilityMode::LEGACY) {
  ReplayForward replay;
  replay.layers.resize(c.numLayers);
  std::vector<float> x = detail::mm(oh, w.tokenEmbedding, c.tokens,
                                    c.vocabularySize, c.dimension);
  detail::add(x, tiny_lm::fixedPosition(c));
  const uint32_t dh = c.dimension / c.numHeads;
  const float branchScale =
      stability == StabilityMode::RESIDUAL_BRANCH_SCALING
          ? float(1.0 / std::sqrt(2.0))
          : 1.0f;
  for (uint32_t li = 0; li < c.numLayers; ++li) {
    const LayerParams& p = layerAt(w, li);
    LayerMetrics& m = replay.layers[li];
    const auto inputStats = moments(x);
    m.inputRms = inputStats.rms;
    m.inputMaxAbs = inputStats.maxAbs;
    // Attention sub-block (pre-norm). The computation below is a verbatim
    // copy of the single/multi-head branches of the CPU reference
    // (generalForward) so logits stay bitwise identical; metrics are computed
    // afterwards from the same intermediates without changing any value.
    const std::vector<float> n1out =
        layerNormWithStats(c, x, p.gamma1, p.beta1, m.n1);
    auto q = detail::mm(n1out, p.wq, c.tokens, c.dimension, c.dimension);
    auto k = detail::mm(n1out, p.wk, c.tokens, c.dimension, c.dimension);
    auto v = detail::mm(n1out, p.wv, c.tokens, c.dimension, c.dimension);
    std::vector<float> prob(size_t(c.numHeads) * c.tokens * c.tokens, 0);
    std::vector<float> ctx(size_t(c.tokens) * c.dimension, 0);
    const float scoreScale = 1.0f / std::sqrt(float(dh));
    double scoreSq = 0, scoreMax = 0;
    if (c.numHeads == 1) {
      auto scores = [&] {
        std::vector<float> o(size_t(c.tokens) * c.tokens);
        for (uint32_t i = 0; i < c.tokens; ++i)
          for (uint32_t j = 0; j < c.tokens; ++j) {
            double s = 0;
            for (uint32_t z = 0; z < c.dimension; ++z)
              s += double(q[size_t(i) * c.dimension + z]) * double(k[size_t(j) * c.dimension + z]);
            o[size_t(i) * c.tokens + j] = float(s);
          }
        return o;
      }();
      for (uint32_t r = 0; r < c.tokens; ++r) {
        float mx = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j <= r; ++j)
          mx = std::max(mx, scores[size_t(r) * c.tokens + j] * scoreScale);
        double sum = 0;
        for (uint32_t j = 0; j <= r; ++j) {
          float e = std::exp(scores[size_t(r) * c.tokens + j] * scoreScale - mx);
          prob[size_t(r) * c.tokens + j] = e;
          sum += e;
        }
        for (uint32_t j = 0; j <= r; ++j)
          prob[size_t(r) * c.tokens + j] /= float(sum);
        for (uint32_t j = 0; j <= r; ++j) {
          const double s0 = scores[size_t(r) * c.tokens + j] * scoreScale;
          scoreSq += s0 * s0;
          scoreMax = std::max(scoreMax, std::abs(s0));
        }
      }
      ctx = detail::mm(prob, v, c.tokens, c.tokens, c.dimension);
    } else {
      for (uint32_t h = 0; h < c.numHeads; ++h)
        for (uint32_t r = 0; r < c.tokens; ++r) {
          const size_t base = (size_t(h) * c.tokens + r) * c.tokens;
          float mx = -std::numeric_limits<float>::infinity();
          for (uint32_t j = 0; j <= r; ++j) {
            double s = 0;
            for (uint32_t d = 0; d < dh; ++d)
              s += double(q[size_t(r) * c.dimension + h * dh + d]) *
                   k[size_t(j) * c.dimension + h * dh + d];
            const float scaled = float(s) * scoreScale;
            mx = std::max(mx, scaled);
            scoreSq += double(scaled) * scaled;
            scoreMax = std::max(scoreMax, std::abs(double(scaled)));
          }
          double sum = 0;
          for (uint32_t j = 0; j <= r; ++j) {
            double s = 0;
            for (uint32_t d = 0; d < dh; ++d)
              s += double(q[size_t(r) * c.dimension + h * dh + d]) *
                   k[size_t(j) * c.dimension + h * dh + d];
            float e = std::exp(float(s) * scoreScale - mx);
            prob[base + j] = e;
            sum += e;
          }
          for (uint32_t j = 0; j <= r; ++j) {
            float a = prob[base + j] / float(sum);
            prob[base + j] = a;
            for (uint32_t d = 0; d < dh; ++d)
              ctx[size_t(r) * c.dimension + h * dh + d] +=
                  a * v[size_t(j) * c.dimension + h * dh + d];
          }
        }
    }
    // Metrics from the already-computed probabilities (no numeric impact).
    m.attention.headEntropyMean.assign(c.numHeads, 0.0);
    m.attention.headMaxProbability.assign(c.numHeads, 0.0);
    m.attention.headMinNonzeroProbability.assign(c.numHeads, 1.0);
    for (uint32_t h = 0; h < c.numHeads; ++h)
      for (uint32_t r = 0; r < c.tokens; ++r) {
        double entropy = 0, maxP = 0, minP = 1.0;
        for (uint32_t j = 0; j <= r; ++j) {
          const double a = prob[(size_t(h) * c.tokens + r) * c.tokens + j];
          if (a > 0) {
            entropy -= a * std::log(a);
            minP = std::min(minP, a);
          }
          maxP = std::max(maxP, a);
        }
        m.attention.headEntropyMean[h] += entropy / c.tokens;
        m.attention.headMaxProbability[h] = std::max(m.attention.headMaxProbability[h], maxP);
        m.attention.headMinNonzeroProbability[h] =
            std::min(m.attention.headMinNonzeroProbability[h], minP);
      }
    m.attention.scoreRms = std::sqrt(scoreSq / (size_t(c.numHeads) * c.tokens * (c.tokens + 1) / 2));
    m.attention.scoreMaxAbs = scoreMax;
    auto attnOut = detail::mm(ctx, p.wo, c.tokens, c.dimension, c.dimension);
    for (float& value : attnOut) value *= branchScale;
    m.attention.outputNorm = l2(attnOut);
    m.attention.residualInputNorm = l2(x);
    m.attention.branchToInputRatio =
        m.attention.residualInputNorm > 0
            ? m.attention.outputNorm / m.attention.residualInputNorm
            : 0.0;
    auto r1 = x;
    detail::add(r1, attnOut);
    const auto r1Stats = moments(r1);
    m.residual1Rms = r1Stats.rms;
    m.residual1MaxAbs = r1Stats.maxAbs;
    // FFN sub-block (pre-norm).
    const std::vector<float> n2out =
        layerNormWithStats(c, r1, p.gamma2, p.beta2, m.n2);
    auto f1 = detail::mm(n2out, p.w1, c.tokens, c.dimension,
                         c.feedForwardDimension);
    const auto f1Stats = moments(f1);
    m.ffn.w1OutputRms = f1Stats.rms;
    m.ffn.w1OutputMaxAbs = f1Stats.maxAbs;
    auto relu = f1;
    size_t zeros = 0;
    double actSq = 0;
    for (float& value : relu) {
      value = std::max(0.0f, value);
      zeros += value == 0.0f;
      actSq += double(value) * double(value);
    }
    m.ffn.activationZeroFraction = double(zeros) / relu.size();
    m.ffn.activationRms = std::sqrt(actSq / relu.size());
    auto ffnOut = detail::mm(relu, p.w2, c.tokens, c.feedForwardDimension,
                             c.dimension);
    for (float& value : ffnOut) value *= branchScale;
    m.ffn.w2OutputNorm = l2(ffnOut);
    m.ffn.residualInputNorm = l2(r1);
    m.ffn.branchToInputRatio =
        m.ffn.residualInputNorm > 0
            ? m.ffn.w2OutputNorm / m.ffn.residualInputNorm
            : 0.0;
    x = r1;
    detail::add(x, ffnOut);
    const auto outStats = moments(x);
    m.outputRms = outStats.rms;
    m.outputMaxAbs = outStats.maxAbs;
  }
  replay.logits = detail::mm(x, w.outputProjection, c.tokens, c.dimension,
                             c.vocabularySize);
  replay.probabilities.resize(replay.logits.size());
  double loss = 0;
  uint32_t correct = 0;
  for (uint32_t r = 0; r < c.tokens; ++r) {
    const size_t base = size_t(r) * c.vocabularySize;
    float mx = *std::max_element(replay.logits.begin() + base,
                                 replay.logits.begin() + base + c.vocabularySize);
    uint32_t pred = 0, truth = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (uint32_t j = 0; j < c.vocabularySize; ++j) {
      if (replay.logits[base + j] > best) { best = replay.logits[base + j]; pred = j; }
      if (target[base + j] > .5f) truth = j;
    }
    double s = 0;
    for (uint32_t j = 0; j < c.vocabularySize; ++j) {
      double e = std::exp(double(replay.logits[base + j] - mx));
      replay.probabilities[base + j] = float(e);
      s += e;
    }
    for (uint32_t j = 0; j < c.vocabularySize; ++j)
      replay.probabilities[base + j] /= float(s);
    loss += mx + std::log(s) - replay.logits[base + truth];
    correct += pred == truth;
  }
  replay.loss = float(loss / c.tokens);
  replay.accuracy = float(correct) / c.tokens;
  return replay;
}

// ---------------------------------------------------------------------------
// Per-step trajectory record and summary/classification
// ---------------------------------------------------------------------------
struct StepMetric {
  int step = 0;
  float loss = 0, accuracy = 0;
  double gradientNorm = 0, parameterNorm = 0, updateNorm = 0,
         updateToParameter = 0, logitMaxAbs = 0, targetMargin = 0,
         targetProbability = 0;
};
struct TrajectorySummary {
  double initialLoss = 0, minimumLoss = 1e30, finalLoss = 0;
  int minimumStep = 0, lastImprovementStep = 0;
  double movingAverageFinalWindow = 0, maximumWorseningWindow = 0;
  std::string classification = "UNKNOWN";
};

// Rules (fixed thresholds, defined before measurement):
//   NEVER_LEARNS:           minimumLoss >= 0.95 * initialLoss
//   LATE_COLLAPSE:          minimumLoss < 0.95*initial, lastImprovement >= 80%
//                           of steps, finalLoss > minimumLoss + 0.5
//   LEARNS_THEN_REGRESSES:  minimumLoss < 0.95*initial AND
//                           finalLoss > minimumLoss + 0.5 AND final worse than
//                           0.95*initial
//   OSCILLATES:             sign changes of step deltas in the final half
//                           exceed 60% and finalLoss > minimumLoss + 0.1*initial
//   PLATEAUS:               learns below 0.7*initial then |final-min| small and
//                           no improvement in the last 30% of steps
//   GENERATION_ONLY_SHORTFALL: training converged (final < 0.5*initial and
//                           final <= minimum + 0.05*initial); generation must
//                           be judged elsewhere
inline std::string classifyTrajectory(const std::vector<StepMetric>& steps) {
  if (steps.empty()) return "UNKNOWN";
  const double initial = steps.front().loss;
  double minimum = initial;
  int lastImprovement = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    const double l = steps[i].loss;
    if (l < minimum - 1e-9) { minimum = l; }
    if (i > 0 && l < steps[i - 1].loss - 1e-9) lastImprovement = int(i);
  }
  const double finalLoss = steps.back().loss;
  const size_t n = steps.size();
  if (minimum >= 0.95 * initial) return "NEVER_LEARNS";
  if (lastImprovement >= int(0.8 * (n - 1)) && finalLoss > minimum + 0.5)
    return "LATE_COLLAPSE";
  if (finalLoss > std::max(minimum + 0.5, 0.95 * initial))
    return "LEARNS_THEN_REGRESSES";
  // oscillation: direction changes within the second half
  int signChanges = 0, deltas = 0;
  for (size_t i = n / 2 + 1; i < n; ++i) {
    const double d = steps[i].loss - steps[i - 1].loss;
    const double p = steps[i - 1].loss - steps[i - 2].loss;
    if (d * p < 0) ++signChanges;
    ++deltas;
  }
  if (deltas > 0 && double(signChanges) / deltas > 0.6 &&
      finalLoss > minimum + 0.1 * initial)
    return "OSCILLATES";
  if (finalLoss < 0.5 * initial && finalLoss <= minimum + 0.05 * initial)
    return "GENERATION_ONLY_SHORTFALL";
  if (lastImprovement < int(0.7 * (n - 1))) return "PLATEAUS";
  return "LEARNS";
}

// First "meaningful" divergence between two same-length loss trajectories.
// Meaningful: |a-b| > absDelta AND > relDelta * max(|a|,|b|), sustained for at
// least `persist` consecutive steps. Returns the 1-based step or -1.
inline int firstTrajectoryDivergence(const std::vector<StepMetric>& a,
                                     const std::vector<StepMetric>& b,
                                     double absDelta = 0.05,
                                     double relDelta = 0.02,
                                     int persist = 3) {
  const size_t n = std::min(a.size(), b.size());
  int streak = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = a[i].loss, y = b[i].loss;
    const double limit = std::max(absDelta, relDelta * std::max(std::abs(x), std::abs(y)));
    if (std::abs(x - y) > limit) {
      if (++streak >= persist) return int(i + 1) - (persist - 1);
    } else {
      streak = 0;
    }
  }
  return -1;
}

inline TrajectorySummary summarizeTrajectory(const std::vector<StepMetric>& steps) {
  TrajectorySummary s;
  if (steps.empty()) return s;
  s.initialLoss = steps.front().loss;
  s.finalLoss = steps.back().loss;
  s.minimumLoss = steps.front().loss;
  double minimum = s.initialLoss;
  for (size_t i = 0; i < steps.size(); ++i) {
    if (steps[i].loss < minimum - 1e-9) {
      minimum = steps[i].loss;
      s.minimumStep = int(i);
    }
    if (i > 0 && steps[i].loss < steps[i - 1].loss - 1e-9)
      s.lastImprovementStep = int(i);
  }
  s.minimumLoss = minimum;
  const size_t window = std::min<size_t>(32, steps.size());
  double ma = 0;
  for (size_t i = steps.size() - window; i < steps.size(); ++i) ma += steps[i].loss;
  s.movingAverageFinalWindow = ma / window;
  // maximum worsening over any 32-step window (sum of deltas)
  double worst = 0;
  for (size_t start = 0; start + window <= steps.size(); ++start) {
    const double delta = steps[start + window - 1].loss - steps[start].loss;
    worst = std::max(worst, delta);
  }
  s.maximumWorseningWindow = worst;
  s.classification = classifyTrajectory(steps);
  return s;
}

// Final evaluation contract of the formal protocol: mean loss and accuracy
// over the four batches at phase 1 (matches cpuLanguageQuality on device).
struct Phase1Evaluation {
  double loss = 0, accuracy = 0;
};
inline Phase1Evaluation phase1Evaluation(const Config& config,
                                         const Params& parameters) {
  Phase1Evaluation result;
  double correct = 0, rows = 0;
  for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    const auto batch = formalBatch(config, pattern, 1);
    const auto step = tiny_lm::forwardBackward(config, batch.first,
                                               batch.second, parameters, 0.0f);
    result.loss += step.loss;
    correct += step.accuracy * config.tokens;
    rows += config.tokens;
  }
  result.loss /= 4;
  result.accuracy = correct / rows;
  return result;
}

inline validation_selection::Metrics validationEvaluation(
    const Config& config, const Params& parameters) {
  validation_selection::Metrics result;
  result.loss = 0.0;
  double correct = 0.0, rows = 0.0, margin = 0.0, probability = 0.0;
  const auto cases = validation_selection::validationCases(config.tokens);
  for (const auto& item : cases) {
    const auto input = tiny_lm::oneHot(item.input, config.vocabularySize);
    const auto target = tiny_lm::oneHot(item.target, config.vocabularySize);
    const auto step = tiny_lm::forwardBackward(config, input, target,
                                               parameters, 0.0f);
    const std::size_t base = std::size_t(config.tokens - 1) * config.vocabularySize;
    const std::uint32_t truth = item.target.back();
    std::uint32_t prediction = 0;
    float other = -std::numeric_limits<float>::infinity();
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
      if (step.logits[base + token] > step.logits[base + prediction]) prediction = token;
      if (token != truth) other = std::max(other, step.logits[base + token]);
    }
    const double p = std::max(1.0e-30, double(step.probabilities[base + truth]));
    result.loss -= std::log(p);
    correct += prediction == truth;
    rows += 1.0;
    margin += step.logits[base + truth] - other;
    probability += p;
  }
  result.loss /= cases.size();
  result.accuracy = correct / rows;
  result.targetMargin = margin / rows;
  result.targetProbability = probability / rows;
  return result;
}

// ---------------------------------------------------------------------------
// Full CPU formal run with per-step metrics and state for diagnostics
// ---------------------------------------------------------------------------
struct FormRun {
  std::vector<StepMetric> steps;
  std::map<int, Params> checkpoints;       // step -> parameters
  std::map<int, Params> firstMoments;
  std::map<int, Params> secondMoments;
  std::map<int, std::pair<std::vector<float>, std::vector<float>>> checkpointBatches;
  Params finalParameters, finalM, finalV;
  TrajectorySummary summary;
  Phase1Evaluation finalEvaluation;
};

inline FormRun runFormalCpu(const Config& config, uint32_t seed, int stepCount,
                            float learningRate = 0.003f,
                            StabilityMode stability = StabilityMode::LEGACY,
                            const std::vector<int>& checkpointSteps = {}) {
  Params params = tiny_lm::initialParameters(config, seed);
  params = applyInitStability(config, std::move(params), stability);
  Params m, v;
  auto zeroOut = [](Params& t, const Params& like) {
    t = like;
    for (const auto& e : tiny_lm::parameterRegistry(t))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
  };
  zeroOut(m, params);
  zeroOut(v, params);
  FormRun run;
  const auto wants = [&](int step) {
    return std::find(checkpointSteps.begin(), checkpointSteps.end(), step) !=
           checkpointSteps.end();
  };
  if (wants(0)) {
    run.checkpoints[0] = params;
    run.firstMoments[0] = m;
    run.secondMoments[0] = v;
    run.checkpointBatches[0] = formalBatch(config, 0);
  }
  for (int step = 1; step <= stepCount; ++step) {
    const uint32_t pattern = uint32_t((step - 1) % 4);
    const auto batch = formalBatch(config, pattern, 0);
    const auto fb = tiny_lm::forwardBackward(config, batch.first, batch.second,
                                             params, 0.0f);
    const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
    const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
    const float lr = phonelm::stabilityLearningRate(
        std::uint32_t(stability), learningRate, std::uint32_t(step),
        std::uint32_t(stepCount));
    Params gradients = fb.gradients;
    if (stability == StabilityMode::GRADIENT_CLIP_1) {
      double normSq = 0;
      for (const auto& e : tiny_lm::parameterRegistry(gradients))
        for (float g : *e.values) normSq += double(g) * double(g);
      const double norm = std::sqrt(normSq);
      if (norm > 1.0 && std::isfinite(norm)) {
        const float scale = float(1.0 / (norm + 1e-6));
        for (const auto& e : tiny_lm::parameterRegistry(gradients))
          for (float& g : *const_cast<std::vector<float>*>(e.values)) g *= scale;
      }
    }
    const auto update = tiny_lm::adamUpdate(params, gradients, m, v, lr, .9f,
                                            .999f, 1e-8f, c1, c2);
    StepMetric metric;
    metric.step = step;
    metric.loss = fb.loss;
    metric.accuracy = fb.accuracy;
    double gradSq = 0, paramSq = 0;
    for (const auto& e : tiny_lm::parameterRegistry(gradients))
      for (float g : *e.values) gradSq += double(g) * double(g);
    for (const auto& e : tiny_lm::parameterRegistry(params))
      for (float x : *e.values) paramSq += double(x) * double(x);
    metric.gradientNorm = std::sqrt(gradSq);
    metric.parameterNorm = std::sqrt(paramSq);
    // update norm & per-token margin/probability
    double updSq = 0, logitMax = 0, marginSum = 0, probSum = 0;
    for (const auto& e : tiny_lm::parameterRegistry(params)) {
      const auto nextIt = std::find_if(
          tiny_lm::parameterRegistry(update.next).begin(),
          tiny_lm::parameterRegistry(update.next).end(),
          [&](const tiny_lm::ParameterInfo& n) { return n.name == e.name; });
      for (size_t i = 0; i < e.values->size(); ++i) {
        const double d = double((*nextIt->values)[i]) - double((*e.values)[i]);
        updSq += d * d;
      }
    }
    metric.updateNorm = std::sqrt(updSq);
    metric.updateToParameter =
        metric.parameterNorm > 0 ? metric.updateNorm / metric.parameterNorm : 0;
    const size_t vocab = config.vocabularySize;
    for (uint32_t r = 0; r < config.tokens; ++r) {
      const size_t base = size_t(r) * vocab;
      uint32_t truth = 0;
      float other = -std::numeric_limits<float>::infinity();
      for (uint32_t j = 0; j < vocab; ++j) {
        logitMax = std::max(logitMax, std::abs(double(fb.logits[base + j])));
        if (batch.second[base + j] > .5f) truth = j;
      }
      for (uint32_t j = 0; j < vocab; ++j)
        if (j != truth) other = std::max(other, fb.logits[base + j]);
      marginSum += fb.logits[base + truth] - other;
      probSum += fb.probabilities[base + truth];
    }
    metric.logitMaxAbs = logitMax;
    metric.targetMargin = marginSum / config.tokens;
    metric.targetProbability = probSum / config.tokens;
    run.steps.push_back(metric);
    params = update.next;
    m = update.firstMoment;
    v = update.secondMoment;
    if (wants(step)) {
      run.checkpoints[step] = params;
      run.firstMoments[step] = m;
      run.secondMoments[step] = v;
      run.checkpointBatches[step] = formalBatch(config, uint32_t(step % 4), 0);
    }
  }
  run.finalParameters = params;
  run.finalM = m;
  run.finalV = v;
  run.summary = summarizeTrajectory(run.steps);
  run.finalEvaluation = phase1Evaluation(config, params);
  return run;
}

// ---------------------------------------------------------------------------
// Optimizer state metrics at a checkpoint
// ---------------------------------------------------------------------------
struct OptimizerStats {
  double parameterNorm = 0, gradientNorm = 0, firstMomentNorm = 0,
         secondMomentNorm = 0, updateNorm = 0, updateToParameter = 0;
};
inline double registryNorm(const Params& p) {
  double s = 0;
  for (const auto& e : tiny_lm::parameterRegistry(p))
    for (float x : *e.values) s += double(x) * double(x);
  return std::sqrt(s);
}
inline double registryDifferenceNorm(const Params& a, const Params& b) {
  const auto ra = tiny_lm::parameterRegistry(a);
  const auto rb = tiny_lm::parameterRegistry(b);
  double s = 0;
  for (size_t i = 0; i < ra.size(); ++i)
    for (size_t j = 0; j < ra[i].values->size(); ++j) {
      const double d = double((*ra[i].values)[j]) - double((*rb[i].values)[j]);
      s += d * d;
    }
  return std::sqrt(s);
}
inline double relativeL2(const Params& a, const Params& b) {
  const double denom = registryNorm(a);
  return denom > 0 ? registryDifferenceNorm(a, b) / denom : 0.0;
}

// ---------------------------------------------------------------------------
// Generation evaluation proxy (CPU rollout, mirrors the on-device protocol:
// four patterns, eight steps, oracle/free-running context, argmax at the last
// position). Not a replacement for the HTP generation contract — a screening
// signal for candidate comparison.
// ---------------------------------------------------------------------------
struct GenerationQuality {
  int oracleExact = 0, freeExact = 0;
};
inline GenerationQuality generationQuality(const Config& config,
                                           const Params& parameters) {
  static const std::vector<std::vector<uint32_t>> rules{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}, {10, 11, 12}};
  GenerationQuality q;
  auto argmax = [&](const std::vector<float>& logits) {
    const size_t base = size_t(config.tokens - 1) * config.vocabularySize;
    uint32_t pred = 0;
    for (uint32_t j = 1; j < config.vocabularySize; ++j)
      if (logits[base + j] > logits[base + pred]) pred = j;
    return pred;
  };
  for (const auto& rule : rules) {
    for (int oracle = 0; oracle < 2; ++oracle) {
      std::vector<uint32_t> context(config.tokens);
      for (uint32_t i = 0; i < config.tokens; ++i) context[i] = rule[i % rule.size()];
      int correct = 0;
      for (int step = 0; step < 8; ++step) {
        const uint32_t expected =
            rule[(size_t(config.tokens) + size_t(step)) % rule.size()];
        const auto input = tiny_lm::oneHot(context, config.vocabularySize);
        const auto target = tiny_lm::oneHot(
            std::vector<uint32_t>(config.tokens, expected), config.vocabularySize);
        const auto fb = tiny_lm::forwardBackward(config, input, target,
                                                 parameters, 0.0f);
        const uint32_t predicted = argmax(fb.logits);
        correct += predicted == expected;
        context.erase(context.begin());
        context.push_back(oracle ? expected : predicted);
      }
      if (correct == 8) {
        if (oracle) ++q.oracleExact;
        else ++q.freeExact;
      }
    }
  }
  return q;
}

struct ValidationSelectedRun {
  FormRun training;
  int selectedStep = 0;
  validation_selection::Metrics bestValidation;
  validation_selection::Metrics finalStepValidation;
  Params selectedParameters, selectedM, selectedV;
  GenerationQuality generation;
  std::vector<std::pair<int, validation_selection::Metrics>> validationTrajectory;
};

inline ValidationSelectedRun runValidationSelectedCpu(
    const Config& config, std::uint32_t seed, int stepCount = 320,
    validation_selection::Mode mode =
        validation_selection::Mode::BEST_VALIDATION_V1) {
  std::vector<int> checkpoints;
  for (int step : validation_selection::evaluationSteps())
    if (step <= stepCount) checkpoints.push_back(step);
  if (std::find(checkpoints.begin(), checkpoints.end(), stepCount) ==
      checkpoints.end())
    checkpoints.push_back(stepCount);
  ValidationSelectedRun result;
  result.training = runFormalCpu(config, seed, stepCount, 0.003f,
                                 StabilityMode::LEGACY, checkpoints);
  result.selectedStep = mode == validation_selection::Mode::FINAL_STEP
                            ? stepCount
                            : 0;
  for (int step : checkpoints) {
    const auto metrics = validationEvaluation(config,
                                               result.training.checkpoints.at(step));
    result.validationTrajectory.push_back({step, metrics});
    if (step == stepCount) result.finalStepValidation = metrics;
    if (mode == validation_selection::Mode::FINAL_STEP ||
        validation_selection::better(metrics, step, result.bestValidation,
                                     result.selectedStep)) {
      if (mode == validation_selection::Mode::FINAL_STEP && step != stepCount)
        continue;
      result.selectedStep = step;
      result.bestValidation = metrics;
    }
  }
  result.selectedParameters = result.training.checkpoints.at(result.selectedStep);
  result.selectedM = result.training.firstMoments.at(result.selectedStep);
  result.selectedV = result.training.secondMoments.at(result.selectedStep);
  result.generation = generationQuality(config, result.selectedParameters);
  return result;
}

// Per-layer gradient norms split by sub-block from a StepResult.
struct LayerGradientStats {
  std::vector<double> inputGradientNorms;      // per layer, layerInputGradients
  std::vector<double> attentionParameterNorms; // wq+wk+wv+wo
  std::vector<double> ffnParameterNorms;       // w1+w2
  std::vector<double> normParameterNorms;      // gamma1+beta1+gamma2+beta2
};
inline LayerGradientStats gradientStats(const Config& config,
                                        const tiny_lm::StepResult& fb) {
  LayerGradientStats stats;
  stats.inputGradientNorms.resize(config.numLayers);
  stats.attentionParameterNorms.assign(config.numLayers, 0.0);
  stats.ffnParameterNorms.assign(config.numLayers, 0.0);
  stats.normParameterNorms.assign(config.numLayers, 0.0);
  for (uint32_t li = 0; li < config.numLayers; ++li) {
    stats.inputGradientNorms[li] = l2(fb.layerInputGradients[li]);
    const LayerParams& g = layerAt(fb.gradients, li);
    stats.attentionParameterNorms[li] =
        std::sqrt(l2(g.wq) * l2(g.wq) + l2(g.wk) * l2(g.wk) +
                  l2(g.wv) * l2(g.wv) + l2(g.wo) * l2(g.wo));
    stats.ffnParameterNorms[li] =
        std::sqrt(l2(g.w1) * l2(g.w1) + l2(g.w2) * l2(g.w2));
    stats.normParameterNorms[li] =
        std::sqrt(l2(g.gamma1) * l2(g.gamma1) + l2(g.beta1) * l2(g.beta1) +
                  l2(g.gamma2) * l2(g.gamma2) + l2(g.beta2) * l2(g.beta2));
  }
  return stats;
}

}  // namespace phonelm::depth_quality
