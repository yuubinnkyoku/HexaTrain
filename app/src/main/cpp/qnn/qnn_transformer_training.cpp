// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "qnn_runtime.h"
#include "qnn_first_nonfinite_diagnostics.h"
#include "qnn_reproducibility.h"
#include "qnn_transformer.h"
#include "../seed_selection.h"
#include "../tiny_language_model_cpu.h"
#include "../training_stability.h"
#include "../validation_checkpoint.h"
#include "../validation_selection.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <sys/resource.h>
#include <tuple>

namespace phonelm::qnn {
namespace {
using Params = TinyTransformerParameters;
constexpr uint32_t kTokens = 4, kDim = 16, kFfn = 32;
constexpr float kEpsilon = 1.0e-5f;
struct NormCache {
  std::vector<float> xhat, inv, output;
};
struct Cache {
  NormCache n1, n2;
  std::vector<float> q, k, v, p, context, res1, ff1, relu, output;
};
struct CpuStep {
  float loss = 0;
  std::vector<float> dOutput, output;
  Params gradients, next;
};
// The payload stays private to device diagnostics.  It is captured before a
// potentially failing QNN step, so it is both the last finite state and the
// exact input state for the failing step.
struct LateNonfiniteCheckpoint {
  int seed = 0, completedStep = 0;
  Params parameters, firstMoment, secondMoment;
  std::vector<float> input, target;
};
void reportLateCheckpoint(const std::string &prefix,
                          const LateNonfiniteCheckpoint &checkpoint,
                          const tiny_lm::Config &config, float lr,
                          float clipThreshold, std::ostringstream &report,
                          std::uint32_t stabilityMode = 0,
                          std::uint32_t pairInitMode = 0,
                          std::uint32_t totalSteps = 0);
namespace first_nonfinite_ns_fwd = phonelm::qnn::first_nonfinite;
first_nonfinite_ns_fwd::Checkpoint privateLateCheckpoint(
    const tiny_lm::Config &config, const LateNonfiniteCheckpoint &checkpoint,
    float lr, float clipThreshold, std::uint32_t stabilityMode = 0,
    std::uint32_t pairInitMode = 0, std::uint32_t totalSteps = 0);
// Private checkpoint / phase-1 evaluation step schedule shared by the dump
// lambda and the trajectory observer. Normal training observers only; they
// never touch training state.
inline const std::vector<int> &dumpStepSchedule() {
  static const std::vector<int> steps{0,  1,  2,  4,  8,   16,  32,  64,
                                      96, 128, 160, 192, 224, 256, 288, 320};
  return steps;
}
std::vector<float> mm(const std::vector<float> &a, const std::vector<float> &b,
                      uint32_t rows, uint32_t inner, uint32_t cols) {
  std::vector<float> o(size_t(rows) * cols);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < cols; ++c) {
      double s = 0;
      for (uint32_t k = 0; k < inner; ++k)
        s += double(a[r * inner + k]) * b[k * cols + c];
      o[r * cols + c] = float(s);
    }
  return o;
}
std::vector<float> mmAtB(const std::vector<float> &a,
                         const std::vector<float> &b, uint32_t rowsA,
                         uint32_t rowsB, uint32_t inner) {
  std::vector<float> o(size_t(rowsA) * rowsB);
  for (uint32_t r = 0; r < rowsA; ++r)
    for (uint32_t c = 0; c < rowsB; ++c) {
      double s = 0;
      for (uint32_t k = 0; k < inner; ++k)
        s += double(a[r * inner + k]) * b[c * inner + k];
      o[r * rowsB + c] = float(s);
    }
  return o;
}
std::vector<float> mmAt(const std::vector<float> &a,
                        const std::vector<float> &b, uint32_t rows,
                        uint32_t inner, uint32_t cols) {
  std::vector<float> o(size_t(inner) * cols);
  for (uint32_t i = 0; i < inner; ++i)
    for (uint32_t c = 0; c < cols; ++c) {
      double s = 0;
      for (uint32_t r = 0; r < rows; ++r)
        s += double(a[r * inner + i]) * b[r * cols + c];
      o[i * cols + c] = float(s);
    }
  return o;
}
void addIn(std::vector<float> &a, const std::vector<float> &b) {
  for (size_t i = 0; i < a.size(); ++i)
    a[i] += b[i];
}
NormCache normForward(const std::vector<float> &x, const std::vector<float> &g,
                      const std::vector<float> &b) {
  NormCache n;
  n.xhat.resize(x.size());
  n.inv.resize(kTokens);
  n.output.resize(x.size());
  for (uint32_t r = 0; r < kTokens; ++r) {
    double mean = 0;
    for (uint32_t d = 0; d < kDim; ++d)
      mean += x[r * kDim + d];
    mean /= kDim;
    double var = 0;
    for (uint32_t d = 0; d < kDim; ++d) {
      double c = x[r * kDim + d] - mean;
      var += c * c;
    }
    var /= kDim;
    n.inv[r] = float(1 / std::sqrt(var + kEpsilon));
    for (uint32_t d = 0; d < kDim; ++d) {
      size_t i = r * kDim + d;
      n.xhat[i] = (x[i] - float(mean)) * n.inv[r];
      n.output[i] = n.xhat[i] * g[d] + b[d];
    }
  }
  return n;
}
void normBackward(const std::vector<float> &dy, const NormCache &n,
                  const std::vector<float> &g, std::vector<float> &dx,
                  std::vector<float> &dg, std::vector<float> &db) {
  dx.resize(dy.size());
  dg.assign(kDim, 0);
  db.assign(kDim, 0);
  for (uint32_t r = 0; r < kTokens; ++r) {
    double sum = 0, sumx = 0;
    for (uint32_t d = 0; d < kDim; ++d) {
      size_t i = r * kDim + d;
      double z = dy[i] * g[d];
      sum += z;
      sumx += z * n.xhat[i];
      dg[d] += dy[i] * n.xhat[i];
      db[d] += dy[i];
    }
    for (uint32_t d = 0; d < kDim; ++d) {
      size_t i = r * kDim + d;
      double z = dy[i] * g[d];
      dx[i] = float(n.inv[r] / kDim * (kDim * z - sum - n.xhat[i] * sumx));
    }
  }
}
Cache forward(const std::vector<float> &x, const Params &w) {
  Cache c;
  c.n1 = normForward(x, w.gamma1, w.beta1);
  c.q = mm(c.n1.output, w.wq, kTokens, kDim, kDim);
  c.k = mm(c.n1.output, w.wk, kTokens, kDim, kDim);
  c.v = mm(c.n1.output, w.wv, kTokens, kDim, kDim);
  auto scores = mmAtB(c.q, c.k, kTokens, kTokens, kDim);
  float scale = 1 / std::sqrt(float(kDim));
  c.p.assign(kTokens * kTokens, 0);
  for (uint32_t r = 0; r < kTokens; ++r) {
    double mx = -std::numeric_limits<double>::infinity();
    for (uint32_t j = 0; j <= r; ++j)
      mx = std::max(mx, double(scores[r * kTokens + j] * scale));
    double sum = 0;
    for (uint32_t j = 0; j <= r; ++j) {
      double z = std::exp(double(scores[r * kTokens + j] * scale) - mx);
      c.p[r * kTokens + j] = float(z);
      sum += z;
    }
    for (uint32_t j = 0; j <= r; ++j)
      c.p[r * kTokens + j] = float(c.p[r * kTokens + j] / sum);
  }
  c.context = mm(c.p, c.v, kTokens, kTokens, kDim);
  auto projected = mm(c.context, w.wo, kTokens, kDim, kDim);
  c.res1 = x;
  addIn(c.res1, projected);
  c.n2 = normForward(c.res1, w.gamma2, w.beta2);
  c.ff1 = mm(c.n2.output, w.w1, kTokens, kDim, kFfn);
  c.relu = c.ff1;
  for (float &v : c.relu)
    v = std::max(0.0f, v);
  auto ff2 = mm(c.relu, w.w2, kTokens, kFfn, kDim);
  c.output = c.res1;
  addIn(c.output, ff2);
  return c;
}
void updateVector(std::vector<float> &next, const std::vector<float> &cur,
                  const std::vector<float> &grad, float lr) {
  next.resize(cur.size());
  for (size_t i = 0; i < cur.size(); ++i)
    next[i] = cur[i] - lr * grad[i];
}
CpuStep cpuStep(const std::vector<float> &x, const std::vector<float> &target,
                const Params &w, float lr) {
  CpuStep s;
  auto c = forward(x, w);
  s.output = c.output;
  s.dOutput.resize(c.output.size());
  double loss = 0;
  for (size_t i = 0; i < c.output.size(); ++i) {
    double e = c.output[i] - target[i];
    loss += e * e;
    s.dOutput[i] = float(2 * e / c.output.size());
  }
  s.loss = float(loss / c.output.size());
  auto &d = s.gradients;
  d.w2 = mmAt(c.relu, s.dOutput, kTokens, kFfn, kDim);
  auto drelu = mmAtB(s.dOutput, w.w2, kTokens, kFfn, kDim);
  std::vector<float> dff1(drelu.size());
  for (size_t i = 0; i < drelu.size(); ++i)
    dff1[i] = c.ff1[i] > 0 ? drelu[i] : 0;
  d.w1 = mmAt(c.n2.output, dff1, kTokens, kDim, kFfn);
  auto dln2 = mmAtB(dff1, w.w1, kTokens, kDim, kFfn);
  std::vector<float> dresLn;
  normBackward(dln2, c.n2, w.gamma2, dresLn, d.gamma2, d.beta2);
  auto dres = s.dOutput;
  addIn(dres, dresLn);
  d.wo = mmAt(c.context, dres, kTokens, kDim, kDim);
  auto dcontext = mmAtB(dres, w.wo, kTokens, kDim, kDim);
  auto dp = mmAtB(dcontext, c.v, kTokens, kTokens, kDim);
  auto dv = mmAt(c.p, dcontext, kTokens, kTokens, kDim);
  std::vector<float> ds(kTokens * kTokens);
  for (uint32_t r = 0; r < kTokens; ++r) {
    double dot = 0;
    for (uint32_t j = 0; j < kTokens; ++j)
      dot += dp[r * kTokens + j] * c.p[r * kTokens + j];
    for (uint32_t j = 0; j < kTokens; ++j)
      ds[r * kTokens + j] =
          c.p[r * kTokens + j] * (dp[r * kTokens + j] - float(dot));
  }
  float scale = 1 / std::sqrt(float(kDim));
  auto dq = mm(ds, c.k, kTokens, kTokens, kDim);
  auto dk = mmAt(ds, c.q, kTokens, kTokens, kDim);
  for (float &v : dq)
    v *= scale;
  for (float &v : dk)
    v *= scale;
  d.wq = mmAt(c.n1.output, dq, kTokens, kDim, kDim);
  d.wk = mmAt(c.n1.output, dk, kTokens, kDim, kDim);
  d.wv = mmAt(c.n1.output, dv, kTokens, kDim, kDim);
  auto dl1 = mmAtB(dq, w.wq, kTokens, kDim, kDim);
  addIn(dl1, mmAtB(dk, w.wk, kTokens, kDim, kDim));
  addIn(dl1, mmAtB(dv, w.wv, kTokens, kDim, kDim));
  std::vector<float> unused;
  normBackward(dl1, c.n1, w.gamma1, unused, d.gamma1, d.beta1);
  updateVector(s.next.gamma1, w.gamma1, d.gamma1, lr);
  updateVector(s.next.beta1, w.beta1, d.beta1, lr);
  updateVector(s.next.wq, w.wq, d.wq, lr);
  updateVector(s.next.wk, w.wk, d.wk, lr);
  updateVector(s.next.wv, w.wv, d.wv, lr);
  updateVector(s.next.wo, w.wo, d.wo, lr);
  updateVector(s.next.gamma2, w.gamma2, d.gamma2, lr);
  updateVector(s.next.beta2, w.beta2, d.beta2, lr);
  updateVector(s.next.w1, w.w1, d.w1, lr);
  updateVector(s.next.w2, w.w2, d.w2, lr);
  return s;
}
Params initialParams(int seed) {
  Params p;
  p.gamma1.assign(kDim, 1);
  p.beta1.assign(kDim, 0);
  p.gamma2.assign(kDim, 1);
  p.beta2.assign(kDim, 0);
  auto fill = [&](std::vector<float> &v, size_t n, int phase) {
    v.resize(n);
    uint32_t state =
        uint32_t(seed) * 747796405u + uint32_t(phase) * 2891336453u;
    for (size_t i = 0; i < n; ++i) {
      state = state * 1664525u + 1013904223u;
      float jitter = float(int((state >> 24) & 15) - 7) * 0.0004f;
      v[i] =
          float(int((i * 17 + size_t(phase) * 13) % 29) - 14) * 0.01f + jitter;
    }
  };
  fill(p.wq, kDim * kDim, 1);
  fill(p.wk, kDim * kDim, 2);
  fill(p.wv, kDim * kDim, 3);
  fill(p.wo, kDim * kDim, 4);
  fill(p.w1, kDim * kFfn, 5);
  fill(p.w2, kFfn * kDim, 6);
  return p;
}
std::vector<float> inputFor(int seed) {
  std::vector<float> x(kTokens * kDim);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(float(i + 1) * (0.057f + seed * 0.003f)) +
           float(int((i + seed) % 9) - 4) * 0.025f;
  return x;
}
Params teacherFrom(Params p, int seed) {
  auto shift = [&](std::vector<float> &v, int phase, float scale) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] += std::sin(float((i + 1) * (phase + seed)) * 0.017f) * scale;
  };
  shift(p.wq, 1, 0.018f);
  shift(p.wk, 2, 0.018f);
  shift(p.wv, 3, 0.036f);
  shift(p.wo, 4, 0.036f);
  shift(p.w1, 5, 0.03f);
  shift(p.w2, 6, 0.03f);
  for (uint32_t d = 0; d < kDim; ++d) {
    p.gamma1[d] += 0.03f * std::sin(float(d + seed));
    p.beta1[d] += 0.015f * std::cos(float(d + seed));
    p.gamma2[d] += 0.03f * std::cos(float(d + 2 * seed));
    p.beta2[d] += 0.015f * std::sin(float(d + 2 * seed));
  }
  return p;
}
double maxAbs(const std::vector<float> &a, const std::vector<float> &b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(double(a[i]) - b[i]));
  return m;
}
bool finite(const std::vector<float> &v) {
  return std::all_of(v.begin(), v.end(),
                     [](float x) { return std::isfinite(x); });
}
double maxParamError(const Params &a, const Params &b) {
  const auto ar = tiny_lm::parameterRegistry(a);
  const auto br = tiny_lm::parameterRegistry(b);
  if (ar.size() != br.size()) return std::numeric_limits<double>::infinity();
  double maximum = 0;
  for (size_t i = 0; i < ar.size(); ++i) {
    if (ar[i].name != br[i].name || ar[i].values->size() != br[i].values->size())
      return std::numeric_limits<double>::infinity();
    if (!finite(*ar[i].values) || !finite(*br[i].values))
      return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, maxAbs(*ar[i].values, *br[i].values));
  }
  return maximum;
}
bool finiteParams(const Params &p) {
  const auto registry = tiny_lm::parameterRegistry(p);
  return std::all_of(registry.begin(), registry.end(),
                     [](const tiny_lm::ParameterInfo &entry) {
                       return finite(*entry.values);
                     });
}
double paramNorm(const Params &p) {
  double s = 0;
  auto add = [&](const std::vector<float> &v) {
    for (float x : v)
      s += double(x) * x;
  };
  for (const auto &entry : tiny_lm::parameterRegistry(p)) add(*entry.values);
  return std::sqrt(s);
}
std::string failure(const char *test, const std::string &e, Runtime &r) {
  return std::string("TINY_TRANSFORMER_TRAINING\ntest=") + test +
         "\nstatus=FAILED\nerror=" + e + "\ncpu_fallback=false\n" +
         r.apiTraceSummary() + r.diagnostics();
}
std::string oneStep() {
  constexpr float lr = 0.01f;
  auto x = inputFor(1);
  auto w = initialParams(1);
  auto target = forward(x, teacherFrom(w, 1)).output;
  auto cpu = cpuStep(x, target, w, lr);
  Runtime rt;
  RuntimeOptions o;
  o.captureQnnCallback = false;
  o.qnnLogLevel = 2;
  rt.setOptions(o);
  std::string e;
  TinyTransformerTrainingOutputs h;
  auto initStarted = std::chrono::steady_clock::now();
  if (!rt.initialize(QnnBackendKind::HTP, e) ||
      !rt.prepareTinyTransformerTraining(kTokens, kDim, kFfn, kEpsilon, true,
                                         e))
    return failure("one_step_prepare", e, rt);
  double initializationUs = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - initStarted)
                                .count();
  if (!rt.executeTinyTransformerTraining(x, target, w, lr, h, e))
    return failure("one_step_execute", e, rt);
  double out = maxAbs(cpu.output, h.output),
         loss = std::abs(double(cpu.loss) - h.loss),
         dout = maxAbs(cpu.dOutput, h.dOutput),
         grad = maxParamError(cpu.gradients, h.gradients),
         next = maxParamError(cpu.next, h.next);
  bool nan = !std::isfinite(h.loss) || !finite(h.output) ||
             !finite(h.dOutput) || !finiteParams(h.gradients) ||
             !finiteParams(h.next);
  bool changed = maxParamError(w, h.next) > 0;
  bool ok = !nan && out < 2e-2 && loss < 2e-2 && dout < 2e-2 && grad < 2e-2 &&
            next < 2e-2 && changed;
  std::ostringstream s;
  s << std::setprecision(10)
    << "TINY_TRANSFORMER_TRAINING\ntest=one_step\nexecution_mode=QNN_HTP_TINY_"
       "TRANSFORMER_TRAINING_STEP\nshape=B1_T4_D16_H1_F32\nloss=MSE\nlearning_"
       "rate="
    << lr << "\nforward_max_abs_error=" << out << "\nloss_abs_error=" << loss
    << "\ndoutput_max_abs_error=" << dout << "\ngradient_max_abs_error=" << grad
    << "\ngradient_gamma1_max_abs_error="
    << maxAbs(cpu.gradients.gamma1, h.gradients.gamma1)
    << "\ngradient_beta1_max_abs_error="
    << maxAbs(cpu.gradients.beta1, h.gradients.beta1)
    << "\ngradient_wq_max_abs_error="
    << maxAbs(cpu.gradients.wq, h.gradients.wq)
    << "\ngradient_wk_max_abs_error="
    << maxAbs(cpu.gradients.wk, h.gradients.wk)
    << "\ngradient_wv_max_abs_error="
    << maxAbs(cpu.gradients.wv, h.gradients.wv)
    << "\ngradient_wo_max_abs_error="
    << maxAbs(cpu.gradients.wo, h.gradients.wo)
    << "\ngradient_gamma2_max_abs_error="
    << maxAbs(cpu.gradients.gamma2, h.gradients.gamma2)
    << "\ngradient_beta2_max_abs_error="
    << maxAbs(cpu.gradients.beta2, h.gradients.beta2)
    << "\ngradient_w1_max_abs_error="
    << maxAbs(cpu.gradients.w1, h.gradients.w1)
    << "\ngradient_w2_max_abs_error="
    << maxAbs(cpu.gradients.w2, h.gradients.w2)
    << "\nnext_weight_max_abs_error=" << next
    << "\nnext_gamma1_max_abs_error=" << maxAbs(cpu.next.gamma1, h.next.gamma1)
    << "\nnext_beta1_max_abs_error=" << maxAbs(cpu.next.beta1, h.next.beta1)
    << "\nnext_wq_max_abs_error=" << maxAbs(cpu.next.wq, h.next.wq)
    << "\nnext_wk_max_abs_error=" << maxAbs(cpu.next.wk, h.next.wk)
    << "\nnext_wv_max_abs_error=" << maxAbs(cpu.next.wv, h.next.wv)
    << "\nnext_wo_max_abs_error=" << maxAbs(cpu.next.wo, h.next.wo)
    << "\nnext_gamma2_max_abs_error=" << maxAbs(cpu.next.gamma2, h.next.gamma2)
    << "\nnext_beta2_max_abs_error=" << maxAbs(cpu.next.beta2, h.next.beta2)
    << "\nnext_w1_max_abs_error=" << maxAbs(cpu.next.w1, h.next.w1)
    << "\nnext_w2_max_abs_error=" << maxAbs(cpu.next.w2, h.next.w2)
    << "\nruntime_initialization_us=" << initializationUs
    << "\ntraining_graph_create_us=" << rt.metrics().graphCreateUs
    << "\ntraining_graph_finalize_us=" << rt.metrics().graphFinalizeUs
    << "\nfirst_execute_us="
    << (rt.metrics().executeUs.empty() ? 0 : rt.metrics().executeUs.front())
    << "\nhtp_loss=" << h.loss << "\ncpu_loss=" << cpu.loss
    << "\nmajor_weight_changed=" << (changed ? "true" : "false")
    << "\ngraph_count=1\nexecute_count_per_step=1\ngraph_create_result="
       "0\ngraph_finalize_result=0\ngraph_execute_result=0\nweight_handoff_"
       "method=PING_PONG_BINDING\ncpu_responsibility=input_target_reference_"
       "and_control\nhtp_responsibility=forward_mse_backward_sgd\ncpu_fallback="
       "false\nnan_detected="
    << (nan ? "true" : "false")
    << "\ninf_detected=false\nstatus=" << (ok ? "SUCCESS" : "FAILED") << '\n'
    << rt.apiTraceSummary() << rt.diagnostics();
  return s.str();
}
std::string multiStep() {
  constexpr int steps = 100;
  constexpr float lr = 0.01f;
  Runtime rt;
  RuntimeOptions o;
  o.captureQnnCallback = false;
  o.qnnLogLevel = 2;
  rt.setOptions(o);
  std::string e;
  auto initStarted = std::chrono::steady_clock::now();
  if (!rt.initialize(QnnBackendKind::HTP, e) ||
      !rt.prepareTinyTransformerTraining(kTokens, kDim, kFfn, kEpsilon, false,
                                         e))
    return failure("multi_step", e, rt);
  double initializationUs = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - initStarted)
                                .count();
  std::ostringstream trajectory;
  trajectory << std::setprecision(10);
  bool allDown = true, nan = false, repro = true;
  double worstWeight = 0;
  double cpuReferenceUs = 0;
  size_t cpuReferenceCount = 0;
  for (int seed = 1; seed <= 5; ++seed) {
    auto x = inputFor(seed);
    auto initial = initialParams(seed);
    auto target = forward(x, teacherFrom(initial, seed)).output;
    auto htp = initial, cpu = initial;
    double initialLoss = cpuStep(x, target, cpu, lr).loss,
           finalLoss = initialLoss;
    trajectory << "seed_" << seed << "_initial_loss=" << initialLoss << '\n';
    TinyTransformerTrainingOutputs out;
    for (int st = 1; st <= steps; ++st) {
      auto cpuStarted = std::chrono::steady_clock::now();
      auto cr = cpuStep(x, target, cpu, lr);
      cpuReferenceUs += std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - cpuStarted)
                            .count();
      ++cpuReferenceCount;
      if (!rt.executeTinyTransformerTraining(x, target, htp, lr, out, e))
        return failure("multi_step", e, rt);
      if (st == 1 || st == 2 || st == 5 || st == 10 || st == 20 || st == 50 ||
          st == 100)
        trajectory << "seed_" << seed << "_step_" << st << "_loss=" << out.loss
                   << "\nseed_" << seed << "_step_" << st
                   << "_cpu_loss=" << cr.loss << '\n';
      cpu = std::move(cr.next);
      std::swap(htp, out.next);
      nan = nan || !std::isfinite(out.loss) || !finiteParams(htp);
      if (nan)
        break;
    }
    if (!rt.executeTinyTransformerTraining(x, target, htp, lr, out, e))
      return failure("multi_step_final", e, rt);
    finalLoss = out.loss;
    double wd = maxParamError(cpu, htp);
    worstWeight = std::max(worstWeight, wd);
    allDown = allDown && finalLoss < initialLoss;
    trajectory << "seed_" << seed << "_final_loss=" << finalLoss << "\nseed_"
               << seed << "_final_weight_norm=" << paramNorm(htp) << "\nseed_"
               << seed << "_cpu_htp_weight_max_abs_difference=" << wd << '\n';
    auto again = initialParams(seed);
    repro = repro && maxParamError(initial, again) == 0;
  }
  double steadyUs = 0;
  for (size_t i = 1; i < rt.metrics().executeUs.size(); ++i)
    steadyUs += rt.metrics().executeUs[i];
  if (rt.metrics().executeUs.size() > 1)
    steadyUs /= double(rt.metrics().executeUs.size() - 1);
  bool ok = allDown && !nan && repro;
  std::ostringstream s;
  s << std::setprecision(10)
    << "TINY_TRANSFORMER_TRAINING\ntest=multi_step\nexecution_mode=QNN_HTP_"
       "TINY_TRANSFORMER_TRAINING_MULTI_STEP\nshape=B1_T4_D16_H1_F32\nloss="
       "MSE\nsteps="
    << steps << "\nseeds=5\nlearning_rate=" << lr
    << "\nall_seeds_loss_decreased=" << (allDown ? "true" : "false")
    << "\ndeterministic_replay=" << (repro ? "true" : "false")
    << "\ncpu_htp_weight_max_abs_difference=" << worstWeight
    << "\nruntime_initialization_us=" << initializationUs
    << "\ntraining_graph_create_us=" << rt.metrics().graphCreateUs
    << "\ntraining_graph_finalize_us=" << rt.metrics().graphFinalizeUs
    << "\nfirst_execute_us="
    << (rt.metrics().executeUs.empty() ? 0 : rt.metrics().executeUs.front())
    << "\nsteady_execute_mean_us=" << steadyUs
    << "\ncpu_scalar_reference_mean_us="
    << (cpuReferenceCount ? cpuReferenceUs / cpuReferenceCount : 0)
    << "\nperformance_context=correctness_workload_unoptimized_scalar_cpu_"
       "reference\ngraph_count=1\ngraph_create_count=1\ngraph_finalize_count="
       "1\ngraph_execute_count="
    << rt.metrics().graphExecuteCount
    << "\nexecute_count_per_training_step=1\nweight_handoff_method=PING_PONG_"
       "BINDING\ncpu_responsibility=input_target_reference_and_control\nhtp_"
       "responsibility=forward_mse_backward_sgd\ncpu_fallback=false\nnan_"
       "detected="
    << (nan ? "true" : "false") << "\ninf_detected=false\n"
    << trajectory.str() << "status=" << (ok ? "SUCCESS" : "FAILED") << '\n'
    << rt.apiTraceSummary() << rt.diagnostics();
  return s.str();
}
float tokenAccuracy(const std::vector<float> &logits,
                    const std::vector<float> &target, uint32_t rows,
                    uint32_t vocabulary) {
  uint32_t correct = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const size_t base = size_t(row) * vocabulary;
    uint32_t predicted = 0, truth = 0;
    for (uint32_t column = 1; column < vocabulary; ++column) {
      if (logits[base + column] > logits[base + predicted]) predicted = column;
      if (target[base + column] > target[base + truth]) truth = column;
    }
    correct += predicted == truth;
  }
  return float(correct) / rows;
}
std::pair<std::vector<float>, std::vector<float>> languageBatch(
    const tiny_lm::Config &config, uint32_t patternIndex, uint32_t phase = 0) {
  static const std::vector<std::vector<uint32_t>> patterns{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}, {10, 11, 12}};
  const auto &pattern = patterns.at(patternIndex % patterns.size());
  std::vector<uint32_t> input(config.tokens), target(config.tokens);
  for (uint32_t i = 0; i < config.tokens; ++i) {
    input[i] = pattern[(i + phase) % pattern.size()];
    target[i] = pattern[(i + phase + 1) % pattern.size()];
  }
  return {tiny_lm::oneHot(input, config.vocabularySize),
          tiny_lm::oneHot(target, config.vocabularySize)};
}
enum class LanguageRolloutContext { ORACLE, FREE_RUNNING };
void advanceLanguageContext(std::vector<uint32_t> &context,
                            uint32_t expected, uint32_t predicted,
                            LanguageRolloutContext mode) {
  context.erase(context.begin());
  context.push_back(mode == LanguageRolloutContext::ORACLE ? expected
                                                           : predicted);
}
std::string crossEntropyMicrotest() {
  constexpr uint32_t rows = 6, vocabulary = 8;
  std::vector<float> logits{
      .2f,-.1f,.7f,.4f,-.5f,.1f,.3f,-.2f,
      1000.f,999.f,998.f,997.f,996.f,995.f,994.f,993.f,
      -1000.f,-1001.f,-1002.f,-1003.f,-1004.f,-1005.f,-1006.f,-1007.f,
      -3.f,-2.f,-1.f,8.f,0.f,1.f,2.f,3.f,
      8.f,7.f,6.f,5.f,4.f,3.f,2.f,-8.f,
      4.f,4.f,4.f,4.f,4.f,4.f,4.f,4.f};
  const std::vector<uint32_t> targetIds{2,0,7,3,7,5};
  auto target = tiny_lm::oneHot(targetIds, vocabulary);
  std::vector<float> cpuProbability(logits.size()), cpuGradient(logits.size());
  double cpuLoss = 0.0;
  for (uint32_t row = 0; row < rows; ++row) {
    const size_t base = size_t(row) * vocabulary;
    float maximum = *std::max_element(logits.begin() + base,
                                      logits.begin() + base + vocabulary);
    double sum = 0.0;
    for (uint32_t column = 0; column < vocabulary; ++column) {
      cpuProbability[base + column] = std::exp(logits[base + column] - maximum);
      sum += cpuProbability[base + column];
    }
    for (uint32_t column = 0; column < vocabulary; ++column) {
      cpuProbability[base + column] /= float(sum);
      cpuGradient[base + column] =
          (cpuProbability[base + column] - target[base + column]) / rows;
    }
    cpuLoss += maximum + std::log(sum) - logits[base + targetIds[row]];
  }
  cpuLoss /= rows;
  Runtime runtime;
  RuntimeOptions options; options.captureQnnCallback = false; options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  CrossEntropyGradientOutputs htp;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareCrossEntropyGradient(rows, vocabulary, error) ||
      !runtime.executeCrossEntropyGradient(logits, target, htp, error))
    return failure("cross_entropy_microtest", error, runtime);
  double probabilityError = maxAbs(cpuProbability, htp.probabilities);
  double gradientError = maxAbs(cpuGradient, htp.dLogits);
  double probabilityRowError = 0.0, gradientRowError = 0.0;
  for (uint32_t row = 0; row < rows; ++row) {
    double ps = 0.0, gs = 0.0;
    for (uint32_t column = 0; column < vocabulary; ++column) {
      ps += htp.probabilities[size_t(row) * vocabulary + column];
      gs += htp.dLogits[size_t(row) * vocabulary + column];
    }
    probabilityRowError = std::max(probabilityRowError, std::abs(ps - 1.0));
    gradientRowError = std::max(gradientRowError, std::abs(gs));
  }
  bool nan = !finite(htp.probabilities) || !finite(htp.dLogits);
  bool ok = !nan && probabilityError <= 5e-3 && gradientError <= 5e-3 &&
            probabilityRowError <= 5e-3 && gradientRowError <= 5e-3;
  std::ostringstream out;
  out << std::setprecision(10)
      << "QNN_HTP_CROSS_ENTROPY\nstatus=" << (ok ? "SUCCESS" : "FAILED")
      << "\nshape=B2_T3_V8\nimplementation=HTP_SOFTMAX_P_MINUS_Y_OVER_N"
      << "\ncross_entropy_loss_scalar=CPU_STABLE_LOGSUMEXP"
      << "\ncross_entropy_gradient=HTP\ncpu_loss=" << cpuLoss
      << "\nprobability_max_abs_error=" << probabilityError
      << "\ndlogits_max_abs_error=" << gradientError
      << "\nprobability_row_sum_max_abs_error=" << probabilityRowError
      << "\ndlogits_row_sum_max_abs_error=" << gradientRowError
      << "\nrequired_cases=normal,large_positive,large_negative,target_max,target_min,equal"
      << "\ncpu_fallback=false\nnan_detected=" << (nan ? "true" : "false")
      << "\ninf_detected=false\n" << runtime.apiTraceSummary()
      << runtime.diagnostics();
  return out.str();
}
std::string languageModelOneStep() {
  tiny_lm::Config config;
  auto [input, target] = languageBatch(config, 0);
  auto parameters = tiny_lm::initialParameters(config, 1);
  constexpr float learningRate = .03f;
  auto cpu = tiny_lm::forwardBackward(config, input, target, parameters,
                                      learningRate);
  Runtime runtime; RuntimeOptions options;
  options.captureQnnCallback = false; options.qnnLogLevel = 2;
  runtime.setOptions(options); std::string error;
  TinyTransformerTrainingOutputs htp;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(config.tokens, config.dimension,
          config.feedForwardDimension, config.epsilon, true, error,
          config.vocabularySize) ||
      !runtime.executeTinyTransformerTraining(input, target, parameters,
                                               learningRate, htp, error))
    return failure("tiny_language_model_one_step", error, runtime);
  const double embeddedError = maxAbs(cpu.embeddedInput, htp.embeddedInput);
  const double logitsError = maxAbs(cpu.logits, htp.logits);
  const double probabilityError = maxAbs(cpu.probabilities, htp.probabilities);
  const double dLogitsError = maxAbs(cpu.dLogits, htp.dLogits);
  const double dEmbeddingError = maxAbs(cpu.gradients.tokenEmbedding,
                                        htp.gradients.tokenEmbedding);
  const double dProjectionError = maxAbs(cpu.gradients.outputProjection,
                                         htp.gradients.outputProjection);
  const double gradientError = maxParamError(cpu.gradients, htp.gradients);
  const double nextError = maxParamError(cpu.next, htp.next);
  const double lossError = std::abs(double(cpu.loss) - htp.loss);
  const bool nan = !std::isfinite(htp.loss) || !finite(htp.logits) ||
                   !finiteParams(htp.gradients) || !finiteParams(htp.next);
  const bool changed = maxParamError(parameters, htp.next) > 0.0;
  const bool ok = !nan && embeddedError < 2e-2 && logitsError < 2e-2 &&
                  probabilityError < 5e-3 && dLogitsError < 5e-3 &&
                  gradientError < 3e-2 && nextError < 3e-2 && changed;
  std::ostringstream out;
  out << std::setprecision(10) << "TINY_LANGUAGE_MODEL\ntest=one_step\nstatus="
      << (ok ? "SUCCESS" : "FAILED")
      << "\nshape=B1_T8_V32_D16_H1_L1_F32\nlearning_rate=" << learningRate
      << "\nembedding_method=CPU_ONE_HOT_HTP_MATMUL"
      << "\nposition_embedding=FIXED_SINUSOIDAL"
      << "\ncross_entropy_loss_scalar=CPU_STABLE_LOGSUMEXP"
      << "\ncross_entropy_gradient=HTP\nloss_abs_error=" << lossError
      << "\nembedded_input_max_abs_error=" << embeddedError
      << "\nlogits_max_abs_error=" << logitsError
      << "\nprobability_max_abs_error=" << probabilityError
      << "\ndlogits_max_abs_error=" << dLogitsError
      << "\ndembedding_max_abs_error=" << dEmbeddingError
      << "\ndoutput_projection_max_abs_error=" << dProjectionError
      << "\ngradient_max_abs_error=" << gradientError
      << "\nnext_parameter_max_abs_error=" << nextError
      << "\ninitial_loss=" << cpu.loss << "\nhtp_loss=" << htp.loss
      << "\ninitial_accuracy=" << cpu.accuracy
      << "\nhtp_accuracy=" << tokenAccuracy(htp.logits, target, 8, 32)
      << "\nmajor_weight_changed=" << (changed ? "true" : "false")
      << "\ngraph_count=1\nexecute_count_per_step=1\ncpu_fallback=false"
      << "\nnan_detected=" << (nan ? "true" : "false")
      << "\ninf_detected=false\n" << runtime.apiTraceSummary()
      << runtime.diagnostics();
  return out.str();
}
struct LanguageQuality {
  double loss = 0, accuracy = 0, meanCorrectProbability = 0;
  double medianCorrectProbability = 0, entropy = 0, meanMargin = 0;
  double minimumMargin = 0;
  std::vector<double> correctProbabilities;
  uint32_t rows = 0, correct = 0, batches = 0;
};
void addLanguageObservation(LanguageQuality &quality,
                            const std::vector<float> &logits,
                            const std::vector<float> &probabilities,
                            const std::vector<float> &target, double loss,
                            uint32_t rows, uint32_t vocabulary) {
  quality.loss += loss;
  ++quality.batches;
  for (uint32_t row = 0; row < rows; ++row) {
    const size_t base = size_t(row) * vocabulary;
    uint32_t truth = 0, prediction = 0;
    float bestOther = -std::numeric_limits<float>::infinity();
    for (uint32_t column = 0; column < vocabulary; ++column) {
      if (target[base + column] > 0.5f) truth = column;
      if (logits[base + column] > logits[base + prediction]) prediction = column;
    }
    double entropy = 0;
    for (uint32_t column = 0; column < vocabulary; ++column) {
      const double probability = std::max(double(probabilities[base + column]), 1e-30);
      entropy -= probability * std::log(probability);
      if (column != truth) bestOther = std::max(bestOther, logits[base + column]);
    }
    const double correctProbability = probabilities[base + truth];
    const double margin = logits[base + truth] - bestOther;
    quality.correctProbabilities.push_back(correctProbability);
    quality.meanCorrectProbability += correctProbability;
    quality.entropy += entropy;
    quality.meanMargin += margin;
    if (quality.rows == 0 || margin < quality.minimumMargin)
      quality.minimumMargin = margin;
    quality.correct += prediction == truth;
    ++quality.rows;
  }
}
LanguageQuality finishLanguageQuality(LanguageQuality quality) {
  quality.loss /= std::max(1u, quality.batches);
  quality.accuracy = double(quality.correct) / std::max(1u, quality.rows);
  quality.meanCorrectProbability /= std::max(1u, quality.rows);
  quality.entropy /= std::max(1u, quality.rows);
  quality.meanMargin /= std::max(1u, quality.rows);
  std::sort(quality.correctProbabilities.begin(), quality.correctProbabilities.end());
  const size_t n = quality.correctProbabilities.size();
  quality.medianCorrectProbability = n ?
      0.5 * (quality.correctProbabilities[(n - 1) / 2] +
             quality.correctProbabilities[n / 2]) : 0;
  return quality;
}
LanguageQuality cpuLanguageQuality(const tiny_lm::Config &config,
                                   const Params &parameters, uint32_t phase) {
  LanguageQuality quality;
  for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    const auto batch = languageBatch(config, pattern, phase);
    const auto step = tiny_lm::forwardBackward(config, batch.first, batch.second,
                                               parameters, 0.0f);
    addLanguageObservation(quality, step.logits, step.probabilities, batch.second,
                           step.loss, config.tokens, config.vocabularySize);
  }
  return finishLanguageQuality(std::move(quality));
}
bool htpLanguageQuality(Runtime &runtime, const tiny_lm::Config &config,
                        const Params &parameters, uint32_t phase,
                        LanguageQuality &quality, std::string &error) {
  LanguageQuality aggregate;
  for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    const auto batch = languageBatch(config, pattern, phase);
    TinyTransformerTrainingOutputs output;
    if (!runtime.executeTinyTransformerTraining(batch.first, batch.second,
                                                 parameters, 0.0f, output, error))
      return false;
    addLanguageObservation(aggregate, output.logits, output.probabilities,
                           batch.second, output.loss, config.tokens,
                           config.vocabularySize);
  }
  quality = finishLanguageQuality(std::move(aggregate));
  return true;
}

LanguageQuality cpuValidationQuality(const tiny_lm::Config &config,
                                     const Params &parameters) {
  LanguageQuality aggregate;
  for (const auto &item :
       validation_selection::validationCases(config.tokens)) {
    const auto input = tiny_lm::oneHot(item.input, config.vocabularySize);
    const auto target = tiny_lm::oneHot(item.target, config.vocabularySize);
    const auto output = tiny_lm::forwardBackward(config, input, target,
                                                 parameters, 0.0f);
    const size_t base = size_t(config.tokens - 1) * config.vocabularySize;
    std::vector<float> logits(output.logits.begin() + base, output.logits.end());
    std::vector<float> probabilities(output.probabilities.begin() + base,
                                     output.probabilities.end());
    std::vector<float> lastTarget(target.begin() + base, target.end());
    uint32_t truth = 0;
    for (uint32_t token = 0; token < config.vocabularySize; ++token)
      if (lastTarget[token] > 0.5f) truth = token;
    const double loss = -std::log(std::max(1.0e-30, double(probabilities[truth])));
    addLanguageObservation(aggregate, logits, probabilities, lastTarget, loss, 1,
                           config.vocabularySize);
  }
  return finishLanguageQuality(std::move(aggregate));
}

bool htpValidationQuality(Runtime &runtime, const tiny_lm::Config &config,
                          const Params &parameters, LanguageQuality &quality,
                          std::string &error,
                          std::size_t *outputNonfiniteCount = nullptr) {
  LanguageQuality aggregate;
  for (const auto &item :
       validation_selection::validationCases(config.tokens)) {
    const auto input = tiny_lm::oneHot(item.input, config.vocabularySize);
    const auto target = tiny_lm::oneHot(item.target, config.vocabularySize);
    TinyTransformerTrainingOutputs output;
    if (!runtime.executeTinyTransformerTraining(input, target, parameters,
                                                 0.0f, output, error))
      return false;
    const std::size_t nonfinite =
        std::count_if(output.logits.begin(), output.logits.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::count_if(output.probabilities.begin(), output.probabilities.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::size_t(!std::isfinite(output.loss));
    if (outputNonfiniteCount) *outputNonfiniteCount += nonfinite;
    if (nonfinite != 0) {
      error = "APP_VALIDATION_NONFINITE_OUTPUT";
      return false;
    }
    const size_t base = size_t(config.tokens - 1) * config.vocabularySize;
    std::vector<float> logits(output.logits.begin() + base, output.logits.end());
    std::vector<float> probabilities(output.probabilities.begin() + base,
                                     output.probabilities.end());
    std::vector<float> lastTarget(target.begin() + base, target.end());
    uint32_t truth = 0;
    for (uint32_t token = 0; token < config.vocabularySize; ++token)
      if (lastTarget[token] > 0.5f) truth = token;
    const double loss = -std::log(std::max(1.0e-30, double(probabilities[truth])));
    addLanguageObservation(aggregate, logits, probabilities, lastTarget, loss, 1,
                           config.vocabularySize);
  }
  quality = finishLanguageQuality(std::move(aggregate));
  return true;
}

validation_selection::Metrics validationMetrics(const LanguageQuality &quality) {
  return {quality.loss, quality.accuracy, quality.meanMargin,
          quality.meanCorrectProbability};
}
using LanguageMember = std::vector<float> Params::*;
const std::vector<std::pair<const char *, LanguageMember>> &languageFields() {
  static const std::vector<std::pair<const char *, LanguageMember>> fields{
      {"token_embedding", &Params::tokenEmbedding}, {"wq", &Params::wq},
      {"wk", &Params::wk}, {"wv", &Params::wv}, {"wo", &Params::wo},
      {"norm1_gamma", &Params::gamma1}, {"norm1_beta", &Params::beta1},
      {"norm2_gamma", &Params::gamma2}, {"norm2_beta", &Params::beta2},
      {"ffn_w1", &Params::w1}, {"ffn_w2", &Params::w2},
      {"output_projection", &Params::outputProjection}};
  return fields;
}
double vectorNorm(const std::vector<float> &values) {
  double sum = 0;
  for (float value : values) sum += double(value) * value;
  return std::sqrt(sum);
}
double vectorMaxAbs(const std::vector<float> &values) {
  double result = 0;
  for (float value : values) result = std::max(result, std::abs(double(value)));
  return result;
}
double parameterUpdateNorm(const Params &current, const Params &next) {
  double sum = 0;
  auto add = [&](const std::vector<float> &a, const std::vector<float> &b) {
    for (size_t i = 0; i < a.size(); ++i) {
      const double difference = double(b[i]) - a[i];
      sum += difference * difference;
    }
  };
  add(current.tokenEmbedding, next.tokenEmbedding);
  auto addLayer = [&](const TinyTransformerLayerParameters &a,
                      const TinyTransformerLayerParameters &b) {
    add(a.wq,b.wq); add(a.wk,b.wk); add(a.wv,b.wv); add(a.wo,b.wo);
    add(a.gamma1,b.gamma1); add(a.beta1,b.beta1); add(a.gamma2,b.gamma2);
    add(a.beta2,b.beta2);
    add(a.w1,b.w1); add(a.w2,b.w2);
  };
  addLayer(current, next);
  for (size_t i = 0; i < current.layers.size(); ++i)
    addLayer(current.layers[i], next.layers[i]);
  add(current.outputProjection, next.outputProjection);
  return std::sqrt(sum);
}
double gradientNorm(const Params &gradient) {
  double sum = 0;
  auto add = [&](const std::vector<float> &values) {
    const double norm = vectorNorm(values);
    sum += norm * norm;
  };
  add(gradient.tokenEmbedding);
  auto addLayer = [&](const TinyTransformerLayerParameters &layer) {
    add(layer.wq); add(layer.wk); add(layer.wv); add(layer.wo);
    add(layer.gamma1); add(layer.beta1); add(layer.gamma2); add(layer.beta2);
    add(layer.w1); add(layer.w2);
  };
  addLayer(gradient);
  for (const auto &layer : gradient.layers) addLayer(layer);
  add(gradient.outputProjection);
  return std::sqrt(sum);
}
std::vector<uint32_t> languageOrder(uint32_t seed, uint32_t epoch,
                                    bool shuffle) {
  std::vector<uint32_t> order{0, 1, 2, 3};
  if (!shuffle) return order;
  uint32_t state = seed * 747796405u + epoch * 2891336453u + 0x9e3779b9u;
  for (size_t index = order.size() - 1; index > 0; --index) {
    state = state * 1664525u + 1013904223u;
    std::swap(order[index], order[state % (index + 1)]);
  }
  return order;
}
struct LanguageCandidate {
  const char *id = "published_sgd";
  int steps = 320;
  float learningRate = 0.01f;
  bool shuffle = false;
};
LanguageCandidate languageCandidate(int candidate) {
  if (candidate == 1) return {"sgd_lr0.1_steps1000_init1_shuffle", 1000, 0.1f, true};
  if (candidate == 2) return {"sgd_lr0.1_steps1000_init1_fixed", 1000, 0.1f, false};
  if (candidate == 3) return {"sgd_lr0.1_steps640_init1_shuffle", 640, 0.1f, true};
  return {};
}
std::string languageModelMultiStep(bool inferenceOnly, int candidate = 0) {
  tiny_lm::Config config;
  const LanguageCandidate selected = inferenceOnly ? languageCandidate(0) :
                                                       languageCandidate(candidate);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize))
    return failure("tiny_language_model_multi_prepare", error, runtime);
  std::ostringstream trajectory;
  trajectory << std::setprecision(10);
  bool allLossDown = true, allAccuracyUp = true, nan = false, replay = true;
  double worstParameterError = 0.0;
  std::vector<double> reductions;
  int accuracy75 = 0;
  const int firstSeed = 1, lastSeed = inferenceOnly ? 1 : 5;
  Params inferenceParameters;
  const std::array<int, 11> checkpoints{1,2,5,10,20,50,100,200,320,640,1000};
  for (int seed = firstSeed; seed <= lastSeed; ++seed) {
    replay = replay &&
             maxParamError(tiny_lm::initialParameters(config, seed),
                           tiny_lm::initialParameters(config, seed)) == 0;
    bool seedNan = false;
    auto htp = tiny_lm::initialParameters(config, seed);
    auto cpu = htp;
    LanguageQuality initialHtp;
    if (!htpLanguageQuality(runtime, config, htp, 1, initialHtp, error))
      return failure("tiny_language_model_initial_eval", error, runtime);
    const auto initialCpu = cpuLanguageQuality(config, cpu, 1);
    trajectory << "seed_" << seed << "_initial_loss=" << initialHtp.loss
               << "\nseed_" << seed << "_initial_accuracy=" << initialHtp.accuracy
               << "\nseed_" << seed << "_initial_correct_probability="
               << initialHtp.meanCorrectProbability
               << "\nseed_" << seed << "_initial_entropy=" << initialHtp.entropy
               << "\nseed_" << seed << "_initial_mean_margin=" << initialHtp.meanMargin
               << "\nseed_" << seed << "_initial_minimum_margin="
               << initialHtp.minimumMargin
               << "\nseed_" << seed << "_initial_cpu_loss=" << initialCpu.loss << '\n';
    TinyTransformerTrainingOutputs output;
    for (int step = 1; step <= selected.steps; ++step) {
      const auto order = languageOrder(uint32_t(seed), uint32_t((step - 1) / 4),
                                       selected.shuffle);
      const uint32_t pattern = inferenceOnly ? 0u : order[size_t(step - 1) % 4];
      const auto batch = languageBatch(config, pattern);
      const auto cpuStep = tiny_lm::forwardBackward(
          config, batch.first, batch.second, cpu, selected.learningRate);
      if (!runtime.executeTinyTransformerTraining(
              batch.first, batch.second, htp, selected.learningRate, output, error))
        return failure("tiny_language_model_multi_execute", error, runtime);
      const bool checkpoint = std::find(checkpoints.begin(), checkpoints.end(),
                                        step) != checkpoints.end();
      if (checkpoint) {
        LanguageQuality observation;
        addLanguageObservation(observation, output.logits, output.probabilities,
                               batch.second, output.loss, config.tokens,
                               config.vocabularySize);
        observation = finishLanguageQuality(std::move(observation));
        const double updateNorm = parameterUpdateNorm(htp, output.next);
        const double currentNorm = paramNorm(htp);
        trajectory << "seed_" << seed << "_step_" << step << "_loss="
                   << observation.loss << "\nseed_" << seed << "_step_" << step
                   << "_accuracy=" << observation.accuracy << "\nseed_" << seed
                   << "_step_" << step << "_correct_probability="
                   << observation.meanCorrectProbability << "\nseed_" << seed
                   << "_step_" << step << "_entropy=" << observation.entropy
                   << "\nseed_" << seed << "_step_" << step << "_mean_margin="
                   << observation.meanMargin << "\nseed_" << seed << "_step_"
                   << step << "_minimum_margin=" << observation.minimumMargin
                   << "\nseed_" << seed << "_step_" << step
                   << "_global_gradient_l2_norm=" << gradientNorm(output.gradients)
                   << "\nseed_" << seed << "_step_" << step
                   << "_global_update_l2_norm=" << updateNorm << "\nseed_" << seed
                   << "_step_" << step << "_global_parameter_l2_norm="
                   << currentNorm << "\nseed_" << seed << "_step_" << step
                   << "_update_to_parameter_ratio="
                   << (currentNorm ? updateNorm / currentNorm : 0) << '\n';
        for (const auto &[name, member] : languageFields()) {
          std::vector<float> update((htp.*member).size());
          for (size_t i = 0; i < update.size(); ++i)
            update[i] = (output.next.*member)[i] - (htp.*member)[i];
          trajectory << "parameter_diagnostic=" << seed << ',' << step << ','
                     << name << ',' << vectorNorm(output.gradients.*member) << ','
                     << vectorNorm(update) << ',' << vectorNorm(htp.*member) << ','
                     << vectorMaxAbs(output.gradients.*member) << ','
                     << vectorMaxAbs(update) << '\n';
        }
      }
      cpu = cpuStep.next;
      htp = output.next;
      seedNan = !std::isfinite(output.loss) || !finiteParams(htp);
      nan = nan || seedNan;
      if (seedNan) break;
    }
    LanguageQuality finalHtp;
    if (!htpLanguageQuality(runtime, config, htp, 1, finalHtp, error))
      return failure("tiny_language_model_final_eval", error, runtime);
    const auto finalCpu = cpuLanguageQuality(config, cpu, 1);
    const double parameterError = maxParamError(cpu, htp);
    const double reduction = 100.0 * (initialHtp.loss - finalHtp.loss) /
                             initialHtp.loss;
    reductions.push_back(reduction);
    accuracy75 += finalHtp.accuracy >= 0.75;
    worstParameterError = std::max(worstParameterError, parameterError);
    allLossDown = allLossDown && finalHtp.loss < initialHtp.loss;
    allAccuracyUp = allAccuracyUp && finalHtp.accuracy > initialHtp.accuracy;
    trajectory << "seed_" << seed << "_final_loss=" << finalHtp.loss
               << "\nseed_" << seed << "_final_accuracy=" << finalHtp.accuracy
               << "\nseed_" << seed << "_loss_reduction=" << reduction
               << "\nseed_" << seed << "_final_correct_probability="
               << finalHtp.meanCorrectProbability << "\nseed_" << seed
               << "_final_median_correct_probability="
               << finalHtp.medianCorrectProbability << "\nseed_" << seed
               << "_final_entropy=" << finalHtp.entropy << "\nseed_" << seed
               << "_final_mean_margin=" << finalHtp.meanMargin << "\nseed_"
               << seed << "_final_minimum_margin=" << finalHtp.minimumMargin
               << "\nseed_" << seed << "_cpu_final_loss=" << finalCpu.loss
               << "\nseed_" << seed << "_cpu_final_accuracy=" << finalCpu.accuracy
               << "\nseed_" << seed << "_parameter_norm=" << paramNorm(htp)
               << "\nseed_" << seed
               << "_cpu_htp_parameter_max_abs_difference=" << parameterError << '\n';
    inferenceParameters = htp;
  }
  std::sort(reductions.begin(), reductions.end());
  const double medianReduction = reductions.empty() ? 0 : reductions[reductions.size()/2];
  if (inferenceOnly) {
    std::vector<uint32_t> context{0,1,2,3,0,1,2,3}, generated;
    bool generationOk = true;
    for (uint32_t step = 0; step < 8; ++step) {
      std::vector<uint32_t> targets(config.tokens);
      for (uint32_t i = 0; i < config.tokens; ++i)
        targets[i] = (context[i] + 1) % 4;
      auto input = tiny_lm::oneHot(context, config.vocabularySize);
      auto target = tiny_lm::oneHot(targets, config.vocabularySize);
      TinyTransformerTrainingOutputs output;
      if (!runtime.executeTinyTransformerTraining(
              input, target, inferenceParameters, 0.0f, output, error))
        return failure("tiny_language_model_generation", error, runtime);
      const size_t base = size_t(config.tokens - 1) * config.vocabularySize;
      uint32_t next = 0;
      for (uint32_t column = 1; column < config.vocabularySize; ++column)
        if (output.logits[base + column] > output.logits[base + next])
          next = column;
      const uint32_t expected = (context.back() + 1) % 4;
      generationOk = generationOk && next == expected;
      generated.push_back(next);
      trajectory << "generation_step_" << step
                 << "_context_last=" << context.back()
                 << "\ngeneration_step_" << step << "_argmax=" << next << '\n';
      context.erase(context.begin());
      context.push_back(next);
    }
    std::ostringstream report;
    report << "TINY_LANGUAGE_MODEL\ntest=autoregressive_inference\nstatus="
           << (generationOk && !nan ? "SUCCESS" : "FAILED")
           << "\nprompt=0,1,2,3,0,1,2,3"
           << "\nexpected_continuation=0,1,2,3,0,1,2,3"
           << "\nargmax_responsibility=CPU\nlogits_responsibility=HTP"
           << "\ncpu_fallback=false\nnan_detected=" << (nan ? "true" : "false")
           << "\ninf_detected=false\n" << trajectory.str()
           << runtime.apiTraceSummary() << runtime.diagnostics();
    return report.str();
  }
  const bool extraConvergence = medianReduction >= 20.0 || accuracy75 >= 4;
  const bool ok = allLossDown && allAccuracyUp &&
                  (candidate == 0 || extraConvergence) && !nan;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TINY_LANGUAGE_MODEL\ntest=sgd_convergence_candidate\nstatus="
         << (ok ? "SUCCESS" : "FAILED") << "\nconfiguration_id=" << selected.id
         << "\nshape=B1_T8_V32_D16_H1_L1_F32\noptimizer=SGD\nsteps="
         << selected.steps << "\nseeds=5\ninitialization_scale=1\nsampling="
         << (selected.shuffle ? "shuffle" : "fixed")
         << "\nlearning_rate=" << selected.learningRate
         << "\nmedian_loss_reduction=" << medianReduction
         << "\naccuracy_75_seed_count=" << accuracy75
         << "\nall_seeds_loss_decreased=" << (allLossDown ? "true" : "false")
          << "\nall_seeds_accuracy_increased=" << (allAccuracyUp ? "true" : "false")
          << "\ndeterministic_replay=" << (replay ? "true" : "false")
         << "\nadditional_convergence_condition="
         << (extraConvergence ? "true" : "false")
         << "\ndataset_patterns=4\ntrain_sequences=4\nevaluation_sequences=4"
         << "\ntrain_phase=0\nevaluation_phase=1\ntrain_evaluation_leakage=false"
         << "\ninput_token_frequency=0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:4,9:4,10:3,11:3,12:2"
         << "\ntarget_token_frequency=0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:4,9:4,10:2,11:3,12:3"
         << "\ncausal_mask=true\nfuture_token_leakage=false"
         << "\ncpu_htp_parameter_max_abs_difference=" << worstParameterError
         << "\ngraph_count=1\ngraph_create_count=1\ngraph_finalize_count=1"
         << "\ngraph_execute_count=" << runtime.metrics().graphExecuteCount
         << "\nexecute_count_per_training_step=1"
         << "\nparameter_handoff=CPU_FIXED_BUFFER_COPY"
         << "\ncross_entropy_loss_scalar=CPU_STABLE_LOGSUMEXP"
         << "\ncross_entropy_gradient=HTP\ncpu_fallback=false"
         << "\nnan_detected=" << (nan ? "true" : "false")
          << "\ninf_detected=false\n" << trajectory.str()
          << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}

std::vector<float> flattenLanguageParameters(const Params &p) {
  std::vector<float> flat;
  auto append = [&](const std::vector<float> &values) {
    flat.insert(flat.end(), values.begin(), values.end());
  };
  // Canonical generic registry order. Do not iterate an unordered container.
  append(p.tokenEmbedding);
  auto appendLayer = [&](const TinyTransformerLayerParameters &layer) {
    append(layer.gamma1); append(layer.beta1);
    append(layer.wq); append(layer.wk); append(layer.wv); append(layer.wo);
    append(layer.gamma2); append(layer.beta2);
    append(layer.w1); append(layer.w2);
  };
  appendLayer(p);
  for (const auto &layer : p.layers) appendLayer(layer);
  append(p.outputProjection);
  return flat;
}
std::vector<float> flattenLanguageParametersLegacyV1(const Params &p) {
  std::vector<float> flat;
  auto append = [&](const std::vector<float> &values) {
    flat.insert(flat.end(), values.begin(), values.end());
  };
  // Read-only compatibility hash order used by the published L2/H2
  // baseline. Optimizer binding continues to use the generic registry above.
  append(p.tokenEmbedding);
  auto appendLayer = [&](const TinyTransformerLayerParameters &layer) {
    append(layer.wq); append(layer.wk); append(layer.wv); append(layer.wo);
    append(layer.gamma1); append(layer.beta1);
    append(layer.gamma2); append(layer.beta2);
    append(layer.w1); append(layer.w2);
  };
  appendLayer(p);
  for (const auto &layer : p.layers) appendLayer(layer);
  append(p.outputProjection);
  return flat;
}
Params unflattenLanguageParameters(const std::vector<float> &flat,
                                   const Params &shape) {
  Params result = shape;
  size_t offset = 0;
  auto take = [&](std::vector<float> &values) {
    std::copy(flat.begin() + offset, flat.begin() + offset + values.size(),
              values.begin());
    offset += values.size();
  };
  take(result.tokenEmbedding);
  auto takeLayer = [&](TinyTransformerLayerParameters &layer) {
    take(layer.gamma1); take(layer.beta1);
    take(layer.wq); take(layer.wk); take(layer.wv); take(layer.wo);
    take(layer.gamma2); take(layer.beta2);
    take(layer.w1); take(layer.w2);
  };
  takeLayer(result);
  for (auto &layer : result.layers) takeLayer(layer);
  take(result.outputProjection);
  if (offset != flat.size()) throw std::invalid_argument("momentum flat shape");
  return result;
}
Params scaleLanguageParameters(const Params &parameters, float scale) {
  auto flat = flattenLanguageParameters(parameters);
  for (float &value : flat) value *= scale;
  return unflattenLanguageParameters(flat, parameters);
}
Params zeroLanguageParameters(const Params &shape) {
  Params result = shape;
  auto zero = [](std::vector<float> &values) {
    std::fill(values.begin(), values.end(), 0.0f);
  };
  zero(result.tokenEmbedding);
  auto zeroLayer = [&](TinyTransformerLayerParameters &layer) {
    zero(layer.wq); zero(layer.wk); zero(layer.wv); zero(layer.wo);
    zero(layer.gamma1); zero(layer.beta1); zero(layer.gamma2); zero(layer.beta2);
    zero(layer.w1); zero(layer.w2);
  };
  zeroLayer(result);
  for (auto &layer : result.layers) zeroLayer(layer);
  zero(result.outputProjection);
  return result;
}
bool executeLanguageMomentum(Runtime &runtime, const Params &current,
                             const Params &gradient, const Params &velocity,
                             float learningRate, float momentum, Params &next,
                             Params &velocityNext, std::string &error) {
  MomentumOptimizerOutputs output;
  if (!runtime.executeMomentumOptimizer(
          flattenLanguageParameters(current), flattenLanguageParameters(gradient),
          flattenLanguageParameters(velocity), learningRate, momentum, output,
          error))
    return false;
  next = unflattenLanguageParameters(output.weightNext, current);
  velocityNext = unflattenLanguageParameters(output.velocityNext, current);
  return true;
}
struct MomentumCandidate { const char *id; int steps; float lr, momentum; };
MomentumCandidate momentumCandidate(int candidate) {
  return candidate == 2
      ? MomentumCandidate{"momentum_lr0.01_m0.95_steps640",640,.01f,.95f}
      : MomentumCandidate{"momentum_lr0.01_m0.95_steps1000",1000,.01f,.95f};
}
std::string languageModelMomentum(bool oneStepOnly, int candidate,
                                  bool inferenceOnly) {
  tiny_lm::Config config;
  const auto selected = momentumCandidate(candidate);
  Runtime runtime; RuntimeOptions options;
  options.captureQnnCallback = false; options.qnnLogLevel = 2;
  runtime.setOptions(options); std::string error;
  const auto shape = tiny_lm::initialParameters(config,1);
  const uint32_t optimizerElements =
      uint32_t(flattenLanguageParameters(shape).size());
  if (!runtime.initialize(QnnBackendKind::HTP,error) ||
      !runtime.prepareTinyTransformerTraining(config.tokens,config.dimension,
          config.feedForwardDimension,config.epsilon,true,error,
          config.vocabularySize) ||
      !runtime.prepareMomentumOptimizer(optimizerElements,error))
    return failure("momentum_prepare",error,runtime);
  if (oneStepOnly) {
    const auto batch=languageBatch(config,0);
    const auto current=shape,velocity=zeroLanguageParameters(current);
    const auto cpuGradient=tiny_lm::forwardBackward(
        config,batch.first,batch.second,current,0.0f);
    const auto cpuUpdate=tiny_lm::momentumUpdate(
        current,cpuGradient.gradients,velocity,selected.lr,selected.momentum);
    TinyTransformerTrainingOutputs htpGradient;
    if(!runtime.executeTinyTransformerTraining(batch.first,batch.second,current,
                                                0.0f,htpGradient,error))
      return failure("momentum_gradient",error,runtime);
    Params htpNext,htpVelocity;
    if(!executeLanguageMomentum(runtime,current,htpGradient.gradients,velocity,
                                selected.lr,selected.momentum,htpNext,
                                htpVelocity,error))
      return failure("momentum_update",error,runtime);
    const double ge=maxParamError(cpuGradient.gradients,htpGradient.gradients);
    const double ve=maxParamError(cpuUpdate.velocity,htpVelocity);
    const double we=maxParamError(cpuUpdate.next,htpNext);
    const bool finiteResult=finiteParams(htpNext)&&finiteParams(htpVelocity);
    const bool changed=maxParamError(current,htpNext)>0;
    const bool ok=ge<.03&&ve<.03&&we<.03&&finiteResult&&changed;
    std::ostringstream report;report<<std::setprecision(10)
      <<"TINY_LANGUAGE_MODEL\ntest=momentum_one_step\nstatus="<<(ok?"SUCCESS":"FAILED")
      <<"\noptimizer=MOMENTUM_SGD\nlearning_rate="<<selected.lr
      <<"\nmomentum="<<selected.momentum<<"\ninitial_velocity=ZERO"
      <<"\ngradient_max_abs_error="<<ge<<"\nnext_velocity_max_abs_error="<<ve
      <<"\nnext_parameter_max_abs_error="<<we
      <<"\nmajor_weight_changed="<<(changed?"true":"false")
      <<"\ngraph_count=2\ngraph_execute_count=2"
      <<"\nexecute_count_per_training_step=2"
      <<"\nvelocity_handoff=CPU_FIXED_BUFFER_COPY"
      <<"\nweight_handoff=CPU_FIXED_BUFFER_COPY\ncpu_fallback=false"
      <<"\nnan_detected="<<(finiteResult?"false":"true")
      <<"\ninf_detected=false\n"<<runtime.apiTraceSummary()<<runtime.diagnostics();
    return report.str();
  }
  std::ostringstream trajectory;trajectory<<std::setprecision(10);
  bool allLoss=true,allAccuracy=true,nan=false;int accuracy75=0;
  double worstParameter=0,worstVelocity=0;std::vector<double> reductions;
  Params inferenceParameters;
  const int lastSeed=inferenceOnly?1:5;
  for(int seed=1;seed<=lastSeed;++seed){
    auto htp=tiny_lm::initialParameters(config,seed),cpu=htp;
    auto htpVelocity=zeroLanguageParameters(htp),cpuVelocity=htpVelocity;
    LanguageQuality initial;
    if(!htpLanguageQuality(runtime,config,htp,1,initial,error))
      return failure("momentum_initial_eval",error,runtime);
    trajectory<<"seed_"<<seed<<"_initial_loss="<<initial.loss
      <<"\nseed_"<<seed<<"_initial_accuracy="<<initial.accuracy<<'\n';
    bool seedNan=false;
    for(int step=1;step<=selected.steps;++step){
      const auto batch=languageBatch(config,uint32_t((step-1)%4));
      const auto cpuGradient=tiny_lm::forwardBackward(
          config,batch.first,batch.second,cpu,0.0f);
      const auto cpuUpdate=tiny_lm::momentumUpdate(
          cpu,cpuGradient.gradients,cpuVelocity,selected.lr,selected.momentum);
      TinyTransformerTrainingOutputs htpGradient;
      if(!runtime.executeTinyTransformerTraining(batch.first,batch.second,htp,
                                                  0.0f,htpGradient,error))
        return failure("momentum_gradient_step",error,runtime);
      Params htpNext,velocityNext;
      if(!executeLanguageMomentum(runtime,htp,htpGradient.gradients,htpVelocity,
                                  selected.lr,selected.momentum,htpNext,
                                  velocityNext,error))
        return failure("momentum_update_step",error,runtime);
      if(step==1||step==2||step==5||step==10||step==20||step==50||
         step==100||step==200||step==320||step==640||step==1000){
        const double update=parameterUpdateNorm(htp,htpNext),norm=paramNorm(htp);
        trajectory<<"seed_"<<seed<<"_step_"<<step<<"_loss="<<htpGradient.loss
          <<"\nseed_"<<seed<<"_step_"<<step<<"_accuracy="
          <<tokenAccuracy(htpGradient.logits,batch.second,config.tokens,
                          config.vocabularySize)
          <<"\nseed_"<<seed<<"_step_"<<step<<"_global_gradient_l2_norm="
          <<gradientNorm(htpGradient.gradients)
          <<"\nseed_"<<seed<<"_step_"<<step<<"_global_update_l2_norm="<<update
          <<"\nseed_"<<seed<<"_step_"<<step<<"_global_parameter_l2_norm="<<norm
          <<"\nseed_"<<seed<<"_step_"<<step<<"_update_to_parameter_ratio="
          <<(norm?update/norm:0)<<'\n';
      }
      cpu=cpuUpdate.next;cpuVelocity=cpuUpdate.velocity;
      htp=std::move(htpNext);htpVelocity=std::move(velocityNext);
      seedNan=!finiteParams(htp)||!finiteParams(htpVelocity)||
              !std::isfinite(htpGradient.loss);nan=nan||seedNan;
      if(seedNan)break;
    }
    LanguageQuality final;
    if(!htpLanguageQuality(runtime,config,htp,1,final,error))
      return failure("momentum_final_eval",error,runtime);
    const auto cpuFinal=cpuLanguageQuality(config,cpu,1);
    const double reduction=100*(initial.loss-final.loss)/initial.loss;
    const double pe=maxParamError(cpu,htp),ve=maxParamError(cpuVelocity,htpVelocity);
    reductions.push_back(reduction);accuracy75+=final.accuracy>=.75;
    allLoss=allLoss&&final.loss<initial.loss;
    allAccuracy=allAccuracy&&final.accuracy>initial.accuracy;
    worstParameter=std::max(worstParameter,pe);worstVelocity=std::max(worstVelocity,ve);
    trajectory<<"seed_"<<seed<<"_final_loss="<<final.loss
      <<"\nseed_"<<seed<<"_final_accuracy="<<final.accuracy
      <<"\nseed_"<<seed<<"_loss_reduction="<<reduction
      <<"\nseed_"<<seed<<"_final_correct_probability="<<final.meanCorrectProbability
      <<"\nseed_"<<seed<<"_final_entropy="<<final.entropy
      <<"\nseed_"<<seed<<"_final_mean_margin="<<final.meanMargin
      <<"\nseed_"<<seed<<"_final_minimum_margin="<<final.minimumMargin
      <<"\nseed_"<<seed<<"_cpu_final_loss="<<cpuFinal.loss
      <<"\nseed_"<<seed<<"_cpu_final_accuracy="<<cpuFinal.accuracy
      <<"\nseed_"<<seed<<"_cpu_htp_parameter_max_abs_difference="<<pe
      <<"\nseed_"<<seed<<"_cpu_htp_velocity_max_abs_difference="<<ve<<'\n';
    inferenceParameters=htp;
  }
  std::sort(reductions.begin(),reductions.end());
  const double median=reductions[reductions.size()/2];
  if(inferenceOnly){
    static const std::array<std::vector<uint32_t>,4> rules{
      std::vector<uint32_t>{0,1,2,3},std::vector<uint32_t>{4,5,6,7},
      std::vector<uint32_t>{8,9},std::vector<uint32_t>{10,11,12}};
    int exactPatterns=0;
    for(size_t pattern=0;pattern<rules.size();++pattern){
      const auto&rule=rules[pattern];std::vector<uint32_t> context(config.tokens);
      for(uint32_t i=0;i<config.tokens;++i)context[i]=rule[i%rule.size()];
      int correct=0;double probability=0,margin=0;
      for(int step=0;step<8;++step){
        auto position=std::find(rule.begin(),rule.end(),context.back());
        uint32_t expected=rule[(size_t(position-rule.begin())+1)%rule.size()];
        auto input=tiny_lm::oneHot(context,config.vocabularySize);
        auto target=tiny_lm::oneHot(std::vector<uint32_t>(config.tokens,expected),
                                    config.vocabularySize);
        TinyTransformerTrainingOutputs output;
        if(!runtime.executeTinyTransformerTraining(input,target,inferenceParameters,
                                                    0.0f,output,error))
          return failure("momentum_generation",error,runtime);
        size_t base=size_t(config.tokens-1)*config.vocabularySize;uint32_t predicted=0;
        float other=-std::numeric_limits<float>::infinity();
        for(uint32_t token=1;token<config.vocabularySize;++token)
          if(output.logits[base+token]>output.logits[base+predicted])predicted=token;
        for(uint32_t token=0;token<config.vocabularySize;++token)
          if(token!=expected)other=std::max(other,output.logits[base+token]);
        correct+=predicted==expected;probability+=output.probabilities[base+expected];
        margin+=output.logits[base+expected]-other;
        context.erase(context.begin());context.push_back(predicted);
      }
      exactPatterns+=correct==8;
      trajectory<<"generation_pattern_"<<pattern<<"_exact="<<(correct==8?"true":"false")
        <<"\ngeneration_pattern_"<<pattern<<"_token_accuracy="<<correct/8.0
        <<"\ngeneration_pattern_"<<pattern<<"_mean_correct_probability="<<probability/8
        <<"\ngeneration_pattern_"<<pattern<<"_mean_margin="<<margin/8<<'\n';
    }
    std::ostringstream report;report<<"TINY_LANGUAGE_MODEL\ntest=momentum_inference_4_pattern\nstatus="
      <<(exactPatterns>=3&&!nan?"SUCCESS":"FAILED")
      <<"\noptimizer=MOMENTUM_SGD\nlearning_rate="<<selected.lr
      <<"\nmomentum="<<selected.momentum<<"\nsteps="<<selected.steps
      <<"\nexact_pattern_count="<<exactPatterns
      <<"\nlogits_responsibility=HTP\nargmax_responsibility=CPU"
      <<"\ncpu_fallback=false\nnan_detected="<<(nan?"true":"false")
      <<"\ninf_detected=false\n"<<trajectory.str()<<runtime.apiTraceSummary()
      <<runtime.diagnostics();return report.str();
  }
  const bool extra=median>=20||accuracy75>=4;
  const bool ok=allLoss&&allAccuracy&&extra&&!nan;
  std::ostringstream report;report<<std::setprecision(10)
    <<"TINY_LANGUAGE_MODEL\ntest=momentum_convergence\nstatus="<<(ok?"SUCCESS":"FAILED")
    <<"\nconfiguration_id="<<selected.id<<"\noptimizer=MOMENTUM_SGD"
    <<"\nlearning_rate="<<selected.lr<<"\nmomentum="<<selected.momentum
    <<"\nsteps="<<selected.steps<<"\nseeds=5\ninitialization_scale=1\nsampling=fixed"
    <<"\nmedian_loss_reduction="<<median<<"\naccuracy_75_seed_count="<<accuracy75
    <<"\nall_seeds_loss_decreased="<<(allLoss?"true":"false")
    <<"\nall_seeds_accuracy_increased="<<(allAccuracy?"true":"false")
    <<"\nadditional_convergence_condition="<<(extra?"true":"false")
    <<"\ncpu_htp_parameter_max_abs_difference="<<worstParameter
    <<"\ncpu_htp_velocity_max_abs_difference="<<worstVelocity
    <<"\ngraph_count=2\ngraph_create_count=2\ngraph_finalize_count=2"
    <<"\nexecute_count_per_training_step=2\ntraining_execute_count_per_step=1"
    <<"\noptimizer_execute_count_per_step=1"
    <<"\nvelocity_handoff=CPU_FIXED_BUFFER_COPY\nweight_handoff=CPU_FIXED_BUFFER_COPY"
    <<"\noptimizer_state_elements="<<optimizerElements
    <<"\noptimizer_state_bytes="<<optimizerElements*sizeof(float)
    <<"\ncross_entropy_loss_scalar=CPU_STABLE_LOGSUMEXP"
    <<"\ncross_entropy_gradient=HTP\ncpu_fallback=false"
    <<"\nnan_detected="<<(nan?"true":"false")<<"\ninf_detected=false\n"
    <<trajectory.str()<<runtime.apiTraceSummary()<<runtime.diagnostics();
  return report.str();
}

struct AdamCandidate {
  const char *id;
  int steps;
  float lr;
  float clipThreshold;
  bool gradientClipping;
};
AdamCandidate adamCandidate(int candidate) {
  if (candidate == 3)
    return AdamCandidate{"post_fix_adam_lr0.003_no_clip_steps320", 320,
                         .003f, 0.0f, false};
  return candidate == 2
      ? AdamCandidate{"adam_lr0.0003_clip10_steps1000", 1000, .0003f,
                      10.0f, true}
      : AdamCandidate{"adam_lr0.0003_clip5_steps1000", 1000, .0003f, 5.0f,
                      true};
}
bool executeLanguageAdam(Runtime &runtime, const Params &current,
                         const Params &gradient, const Params &firstMoment,
                         const Params &secondMoment, float learningRate, int step,
                         float gradientScale, Params &next, Params &firstMomentNext,
                         Params &secondMomentNext, AdamOptimizerOutputs *raw,
                         std::string &error,
                         uint32_t optimizerChunkElements = 0) {
  const auto currentFlat = flattenLanguageParameters(current);
  const auto gradientFlat = flattenLanguageParameters(gradient);
  const auto firstFlat = flattenLanguageParameters(firstMoment);
  const auto secondFlat = flattenLanguageParameters(secondMoment);
  if (currentFlat.size() != gradientFlat.size() ||
      currentFlat.size() != firstFlat.size() ||
      currentFlat.size() != secondFlat.size() || currentFlat.empty()) {
    error = "APP_PARAMETER_SCHEMA: Adam flattened registry mismatch";
    return false;
  }
  const size_t chunkElements =
      optimizerChunkElements ? optimizerChunkElements : currentFlat.size();
  if (chunkElements == 0 ||
      chunkElements > std::numeric_limits<uint32_t>::max()) {
    error = "APP_RESOURCE_ESTIMATOR: Adam chunk size is invalid";
    return false;
  }
  const float firstCorrection =
      float(1.0 / (1.0 - std::pow(0.9, double(step))));
  const float secondCorrection =
      float(1.0 / (1.0 - std::pow(0.999, double(step))));
  AdamOptimizerOutputs output;
  const auto append = [](std::vector<float> &destination,
                         const std::vector<float> &source, size_t count) {
    destination.insert(destination.end(), source.begin(),
                       source.begin() + count);
  };
  for (size_t offset = 0; offset < currentFlat.size();
       offset += chunkElements) {
    const size_t count =
        std::min(chunkElements, currentFlat.size() - offset);
    std::vector<float> currentChunk(chunkElements, 0.0f);
    std::vector<float> gradientChunk(chunkElements, 0.0f);
    std::vector<float> firstChunk(chunkElements, 0.0f);
    std::vector<float> secondChunk(chunkElements, 0.0f);
    std::copy_n(currentFlat.begin() + offset, count, currentChunk.begin());
    std::copy_n(gradientFlat.begin() + offset, count, gradientChunk.begin());
    std::copy_n(firstFlat.begin() + offset, count, firstChunk.begin());
    std::copy_n(secondFlat.begin() + offset, count, secondChunk.begin());
    AdamOptimizerOutputs chunkOutput;
    if (!runtime.executeAdamOptimizer(
            currentChunk, gradientChunk, firstChunk, secondChunk, learningRate,
            gradientScale, firstCorrection, secondCorrection, chunkOutput,
            error))
      return false;
    append(output.firstMomentNext, chunkOutput.firstMomentNext, count);
    append(output.secondMomentNext, chunkOutput.secondMomentNext, count);
    append(output.firstMomentHat, chunkOutput.firstMomentHat, count);
    append(output.secondMomentHat, chunkOutput.secondMomentHat, count);
    append(output.secondRoot, chunkOutput.secondRoot, count);
    append(output.denominator, chunkOutput.denominator, count);
    append(output.dividedUpdate, chunkOutput.dividedUpdate, count);
    append(output.normalizedUpdate, chunkOutput.normalizedUpdate, count);
    append(output.scaledUpdate, chunkOutput.scaledUpdate, count);
    append(output.weightNext, chunkOutput.weightNext, count);
  }
  next = unflattenLanguageParameters(output.weightNext, current);
  firstMomentNext =
      unflattenLanguageParameters(output.firstMomentNext, current);
  secondMomentNext =
      unflattenLanguageParameters(output.secondMomentNext, current);
  if (raw) *raw = std::move(output);
  return true;
}
std::string languageModelAdam(bool oneStepOnly, int candidate,
                              bool inferenceOnly,
                              tiny_lm::Config config = {},
                              int lastSeed = 5,
                              bool scalingSmoke = false,
                              int layers = 1,
                              int attentionHeads = 1,
                              const LogSink& progress = {},
                              int requestedSteps = 0,
                              bool numericalProbe = false,
                              float requestedLearningRate = 0.0f,
                              int firstSeed = 1,
                              const char* seedSelectionMode = "COUNT_FROM_ONE",
                              int requestedSeed = 0,
                              std::uint32_t stabilityMode = 0,
                              std::uint32_t pairInitMode = 0,
                              std::uint32_t checkpointSelectionMode = 0,
                              bool diagnosticTrajectory = false,
                              const std::string& checkpointDir = {}) {
  // These are part of the model configuration, not a reporting-only scale
  // label.  The CPU reference and the QNN graph receive exactly the same
  // shape contract below.
  if (layers <= 0 || attentionHeads <= 0) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=layers and attention heads must both be positive\n";
  }
  config.numLayers = static_cast<uint32_t>(layers);
  config.numHeads = static_cast<uint32_t>(attentionHeads);
  if (firstSeed < 1 || firstSeed > lastSeed) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=firstSeed must satisfy 1 <= firstSeed <= lastSeed\n";
  }
  const int executedSeedCount = lastSeed - firstSeed + 1;
  const int reportedRequestedSeed = requestedSeed > 0 ? requestedSeed : lastSeed;
  auto selected = adamCandidate(candidate);
  if (requestedSteps > 0) selected.steps = requestedSteps;
  if (requestedLearningRate > 0.0f)
    selected.lr = requestedLearningRate;
  const bool formalPostFix = candidate == 3;
  if (const char* stabilityError =
          phonelm::validateTrainingStabilityMode(stabilityMode)) {
    return std::string(
               "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=") +
           stabilityError + '\n';
  }
  if (const char* pairError = phonelm::validateDepthPairInitMode(pairInitMode)) {
    return std::string(
               "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=") +
           pairError + '\n';
  }
  if (const char* selectionError =
          validation_selection::validateMode(checkpointSelectionMode)) {
    return std::string(
               "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=") +
           selectionError + '\n';
  }
  const bool bestValidationMode =
      checkpointSelectionMode ==
      std::uint32_t(validation_selection::Mode::BEST_VALIDATION_V1);
  if (bestValidationMode && !(formalPostFix && inferenceOnly)) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=BEST_VALIDATION_V1 requires the formal candidate-3 protocol\n";
  }
  if (bestValidationMode && config.tokens < 4) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=BEST_VALIDATION_V1 requires sequence length >= 4 for rotated-prefix separation\n";
  }
  if (stabilityMode != 0 && !(formalPostFix && inferenceOnly)) {
    return std::string(
               "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\nerror="
               "training stability modes are restricted to candidate 3 "
               "formal protocol runs\n");
  }
  if (stabilityMode == 6) {  // GRADIENT_CLIP_1
    selected.gradientClipping = true;
    selected.clipThreshold = 1.0f;
  }
  int checkpointDumpErrors = 0;
  std::ostringstream checkpointDumpSteps;
  const auto resources = tiny_lm::resourceEstimate(config);
  if (!resources.ok) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\nfailure_classification=" +
           resources.failureClassification + "\nerror=" + resources.detail +
           "\n";
  }
  if (bestValidationMode &&
      resources.estimatedPeakWithBestCheckpointBytes >
          phonelm::transformer::kApplicationPolicyBytes) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_POLICY_LIMIT\n"
           "error=best-validation checkpoint exceeds the 1536 MiB application policy\n";
  }
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  const auto shape = tiny_lm::initialParameters(config, 1);
  const auto flattenedShape = flattenLanguageParameters(shape);
  if (flattenedShape.empty() ||
      flattenedShape.size() > std::numeric_limits<uint32_t>::max()) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_RESOURCE_ESTIMATOR\n"
           "error=Adam parameter registry exceeds uint32 element range\n";
  }
  const uint32_t optimizerElements =
      static_cast<uint32_t>(flattenedShape.size());
  const uint32_t optimizerGraphElements =
      static_cast<uint32_t>(std::min<std::uint64_t>(
          optimizerElements,
          phonelm::transformer::kMaximumAdamChunkElements));
  const uint32_t optimizerChunkCount =
      static_cast<uint32_t>(
          (std::uint64_t{optimizerElements} + optimizerGraphElements - 1) /
          optimizerGraphElements);
  if (optimizerGraphElements != resources.adamGraphElements ||
      optimizerChunkCount != resources.adamChunkCount) {
    return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
           "failure_classification=APP_RESOURCE_ESTIMATOR\n"
           "error=Adam execution and resource-estimator chunk contracts differ\n";
  }
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL,
          TinyTransformerTrainingTapSet::NONE, config.numLayers,
          config.numHeads) ||
      !runtime.prepareAdamOptimizer(optimizerGraphElements, error))
    return failure("adam_prepare", error, runtime);
  bool scalingAppWriteUnchanged = true;
  size_t scalingPoisonResidualElements = 0;
  bool scalingAuditFinite = true;
  const bool scalingConfiguration =
      config.tokens != 8 || config.dimension != 16 || layers != 1 ||
      attentionHeads != 1;
  const bool generalizedNoSgdNext =
      config.numLayers != 1 || config.numHeads != 1;
  if (scalingConfiguration) {
    constexpr float poison = 1.1415926f;
    const auto auditBatch = languageBatch(config, 0);
    const auto auditParameters = shape;
    const auto inputHash = canonicalFloatSha256(auditBatch.first);
    const auto targetHash = canonicalFloatSha256(auditBatch.second);
    const auto parameterHash =
        canonicalFloatSha256(flattenLanguageParameters(auditParameters));
    TinyTransformerTrainingOutputs auditOutput;
    auditOutput.output.assign(size_t(config.tokens) * config.dimension,
                              poison);
    auditOutput.dOutput = auditOutput.output;
    auditOutput.embeddedInput = auditOutput.output;
    auditOutput.dEmbeddedInput = auditOutput.output;
    auditOutput.logits.assign(
        size_t(config.tokens) * config.vocabularySize, poison);
    auditOutput.probabilities = auditOutput.logits;
    auditOutput.dLogits = auditOutput.logits;
    auditOutput.gradients = unflattenLanguageParameters(
        std::vector<float>(flattenLanguageParameters(auditParameters).size(), poison),
        auditParameters);
    auditOutput.next = auditOutput.gradients;
    if (!runtime.executeTinyTransformerTraining(
            auditBatch.first, auditBatch.second, auditParameters, 0.0f,
            auditOutput, error))
      return failure("scaling_binding_audit", error, runtime);
    scalingAppWriteUnchanged =
        inputHash == canonicalFloatSha256(auditBatch.first) &&
        targetHash == canonicalFloatSha256(auditBatch.second) &&
        parameterHash ==
            canonicalFloatSha256(flattenLanguageParameters(auditParameters)) &&
        runtime.tinyTransformerTrainingLastLearningRateBytesUnchanged();
    std::vector<const std::vector<float> *> auditVectors{
        &auditOutput.output, &auditOutput.dOutput, &auditOutput.dEmbeddedInput,
        &auditOutput.logits, &auditOutput.probabilities, &auditOutput.dLogits};
    if (!generalizedNoSgdNext) auditVectors.push_back(&auditOutput.embeddedInput);
    for (const auto *values : auditVectors) {
      scalingPoisonResidualElements +=
          std::count(values->begin(), values->end(), poison);
      scalingAuditFinite = scalingAuditFinite && finite(*values);
    }
    const auto auditGradients =
        flattenLanguageParameters(auditOutput.gradients);
    scalingPoisonResidualElements +=
        std::count(auditGradients.begin(), auditGradients.end(), poison);
    scalingAuditFinite = scalingAuditFinite && finite(auditGradients);
    if (!generalizedNoSgdNext) {
      const auto auditNext = flattenLanguageParameters(auditOutput.next);
      scalingPoisonResidualElements +=
          std::count(auditNext.begin(), auditNext.end(), poison);
      scalingAuditFinite = scalingAuditFinite && finite(auditNext);
    } else if (!auditOutput.next.tokenEmbedding.empty() ||
               !auditOutput.next.layers.empty() ||
               !auditOutput.next.outputProjection.empty()) {
      return failure("scaling_binding_audit",
                     "generalized graph reported SGD next parameters.", runtime);
    }
    if (!scalingAppWriteUnchanged || scalingPoisonResidualElements != 0 ||
        !scalingAuditFinite)
      return failure("scaling_binding_audit",
                     "APP_WRITE mutation, APP_READ poison residual, or "
                     "non-finite audit output detected.",
                     runtime);
  }
  if (oneStepOnly) {
    constexpr float diagnosticLearningRate = .003f;
    const std::array<int, 13> checkpoints{
        0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 320};
    auto current = shape;
    auto first = zeroLanguageParameters(current);
    auto second = first;
    bool checkpointRoundtrip = true;
    bool initialOneStepCorrect = false;
    int compared = 0, firstMajorDivergence = -1;
    int firstOptimizerDivergence = -1;
    double maximumGradientError = 0.0, maximumOptimizerIsolationError = 0.0;
    double maximumLayerInputError = 0.0, maximumHeadProbabilityError = 0.0;
    bool diagnosticsSchemaValid = true, diagnosticsFinite = true;
    std::ostringstream detail;
    detail << std::setprecision(10);
    auto appendStats = [&](const std::string &prefix,
                           const std::vector<float> &cpu,
                           const std::vector<float> &htp) {
      if (cpu.size() != htp.size()) {
        diagnosticsSchemaValid = false;
        detail << prefix << "_schema_match=false\n"
               << prefix << "_cpu_element_count=" << cpu.size() << '\n'
               << prefix << "_htp_element_count=" << htp.size() << '\n';
        return std::numeric_limits<double>::infinity();
      }
      detail << prefix << "_schema_match=true\n";
      double maxError = 0, meanError = 0, maxRelative = 0, l2 = 0;
      double cpuL2 = 0, htpL2 = 0, dot = 0;
      size_t nonfinite = 0;
      float cpuMin = std::numeric_limits<float>::infinity();
      float cpuMax = -cpuMin, htpMin = cpuMin, htpMax = -cpuMin;
      for (size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(htp[i])) {
          ++nonfinite;
          continue;
        }
        const double e = std::abs(double(cpu[i]) - htp[i]);
        maxError = std::max(maxError, e);
        meanError += e;
        maxRelative =
            std::max(maxRelative, e / std::max(1.0e-12, std::abs(double(cpu[i]))));
        l2 += e * e;
        cpuL2 += double(cpu[i]) * cpu[i];
        htpL2 += double(htp[i]) * htp[i];
        dot += double(cpu[i]) * htp[i];
        cpuMin = std::min(cpuMin, cpu[i]);
        cpuMax = std::max(cpuMax, cpu[i]);
        htpMin = std::min(htpMin, htp[i]);
        htpMax = std::max(htpMax, htp[i]);
      }
      const double cosine =
          cpuL2 && htpL2 ? dot / std::sqrt(cpuL2 * htpL2) : 1.0;
      detail << prefix << "_max_abs_error=" << maxError << '\n'
             << prefix << "_mean_abs_error="
             << (cpu.empty() ? 0 : meanError / cpu.size()) << '\n'
             << prefix << "_max_relative_error=" << maxRelative << '\n'
             << prefix << "_l2_error=" << std::sqrt(l2) << '\n'
             << prefix << "_cosine_similarity=" << cosine << '\n'
             << prefix << "_htp_norm_over_cpu_norm="
             << (cpuL2 ? std::sqrt(htpL2 / cpuL2) : 1.0) << '\n'
             << prefix << "_cpu_min=" << cpuMin << '\n'
             << prefix << "_cpu_max=" << cpuMax << '\n'
             << prefix << "_htp_min=" << htpMin << '\n'
             << prefix << "_htp_max=" << htpMax << '\n'
             << prefix << "_nonfinite_count=" << nonfinite << '\n';
      if (nonfinite != 0) {
        diagnosticsFinite = false;
        return std::numeric_limits<double>::infinity();
      }
      return maxError;
    };
    auto appendParameterStats = [&](const std::string &prefix, const Params &cpu,
                                    const Params &htp) {
      const auto ca = tiny_lm::parameterRegistry(cpu);
      const auto ha = tiny_lm::parameterRegistry(htp);
      if (ca.size() != ha.size()) return std::numeric_limits<double>::infinity();
      double worst = 0;
      for (size_t i = 0; i < ca.size(); ++i) {
        if (ca[i].name != ha[i].name)
          return std::numeric_limits<double>::infinity();
        std::string name = ca[i].name;
        std::replace(name.begin(), name.end(), '.', '_');
        worst = std::max(worst, appendStats(prefix + "_" + name,
                                            *ca[i].values, *ha[i].values));
      }
      return worst;
    };
    for (int completed = 0; completed <= 320; ++completed) {
      if (std::find(checkpoints.begin(), checkpoints.end(), completed) !=
          checkpoints.end()) {
        const int adamStep = completed + 1;
        const auto batch = languageBatch(config, uint32_t(completed % 4));
        const auto cpuGradient = tiny_lm::forwardBackward(
            config, batch.first, batch.second, current, 0.0f);
        TinyTransformerTrainingOutputs htpGradient;
        if (!runtime.executeTinyTransformerTraining(
                batch.first, batch.second, current, 0.0f, htpGradient, error))
          return failure("adam_sync_gradient", error, runtime);
        const float c1 =
            float(1.0 / (1.0 - std::pow(0.9, double(adamStep))));
        const float c2 =
            float(1.0 / (1.0 - std::pow(0.999, double(adamStep))));
        const auto pathA = tiny_lm::adamUpdate(
            current, cpuGradient.gradients, first, second, diagnosticLearningRate, .9f,
            .999f, 1e-8f, c1, c2);
        const auto pathB = tiny_lm::adamUpdate(
            current, htpGradient.gradients, first, second, diagnosticLearningRate, .9f,
            .999f, 1e-8f, c1, c2);
        Params pathC, pathCFirst, pathCSecond, pathD, pathDFirst, pathDSecond;
        AdamOptimizerOutputs rawC, rawD;
        if (!executeLanguageAdam(runtime, current, cpuGradient.gradients, first,
                                 second, diagnosticLearningRate, adamStep, 1.0f, pathC,
                                 pathCFirst, pathCSecond, &rawC, error,
                                 optimizerGraphElements) ||
            !executeLanguageAdam(runtime, current, htpGradient.gradients, first,
                                 second, diagnosticLearningRate, adamStep, 1.0f, pathD,
                                 pathDFirst, pathDSecond, &rawD, error,
                                 optimizerGraphElements))
          return failure("adam_sync_optimizer", error, runtime);
        const std::string prefix = "checkpoint_" + std::to_string(completed);
        detail << prefix << "_step_index=" << adamStep << '\n'
               << prefix << "_bias_correction_first=" << c1 << '\n'
               << prefix << "_bias_correction_second=" << c2 << '\n'
               << prefix << "_parameter_norm=" << paramNorm(current) << '\n'
               << prefix << "_first_moment_norm=" << paramNorm(first) << '\n'
               << prefix << "_second_moment_norm=" << paramNorm(second) << '\n';
        appendStats(prefix + "_logits", cpuGradient.logits,
                    htpGradient.logits);
        appendStats(prefix + "_probabilities", cpuGradient.probabilities,
                    htpGradient.probabilities);
        appendStats(prefix + "_dlogits", cpuGradient.dLogits,
                    htpGradient.dLogits);
        double checkpointLayerError = 0.0, checkpointHeadError = 0.0;
        if (cpuGradient.layerInputGradients.size() !=
                htpGradient.layerInputGradients.size() ||
            cpuGradient.attentionHeadProbabilities.size() !=
                htpGradient.taps.size())
          return failure("adam_sync_diagnostics",
                         "CPU/HTP layer or head diagnostic count mismatch.",
                         runtime);
        for (size_t layer = 0; layer < cpuGradient.layerInputGradients.size();
             ++layer)
          checkpointLayerError = std::max(
              checkpointLayerError,
              appendStats(prefix + "_layer_" + std::to_string(layer) +
                              "_input_gradient",
                          cpuGradient.layerInputGradients[layer],
                          htpGradient.layerInputGradients[layer]));
        for (size_t index = 0;
             index < cpuGradient.attentionHeadProbabilities.size(); ++index) {
          const size_t layer = index / config.numHeads;
          const size_t head = index % config.numHeads;
          checkpointHeadError = std::max(
              checkpointHeadError,
              appendStats(prefix + "_layer_" + std::to_string(layer) +
                              "_head_" + std::to_string(head) +
                              "_attention_probabilities",
                          cpuGradient.attentionHeadProbabilities[index],
                          htpGradient.taps[index].values));
        }
        maximumLayerInputError =
            std::max(maximumLayerInputError, checkpointLayerError);
        maximumHeadProbabilityError =
            std::max(maximumHeadProbabilityError, checkpointHeadError);
        const double gradientError = appendParameterStats(
            prefix + "_gradient", cpuGradient.gradients,
            htpGradient.gradients);
        const double optimizerC =
            std::max({maxParamError(pathA.next, pathC),
                      maxParamError(pathA.firstMoment, pathCFirst),
                      maxParamError(pathA.secondMoment, pathCSecond)});
        const double optimizerD =
            std::max({maxParamError(pathB.next, pathD),
                      maxParamError(pathB.firstMoment, pathDFirst),
                      maxParamError(pathB.secondMoment, pathDSecond)});
        appendParameterStats(prefix + "_path_c_next_parameter", pathA.next,
                             pathC);
        appendParameterStats(prefix + "_path_c_first_moment",
                             pathA.firstMoment, pathCFirst);
        appendParameterStats(prefix + "_path_c_second_moment",
                             pathA.secondMoment, pathCSecond);
        appendParameterStats(prefix + "_path_d_next_parameter", pathB.next,
                             pathD);
        appendParameterStats(prefix + "_path_d_first_moment",
                             pathB.firstMoment, pathDFirst);
        appendParameterStats(prefix + "_path_d_second_moment",
                             pathB.secondMoment, pathDSecond);
        maximumGradientError = std::max(maximumGradientError, gradientError);
        maximumOptimizerIsolationError = std::max(
            maximumOptimizerIsolationError, std::max(optimizerC, optimizerD));
        if (firstMajorDivergence < 0 && gradientError >= .01)
          firstMajorDivergence = completed;
        if (firstOptimizerDivergence < 0 &&
            std::max(optimizerC, optimizerD) >= .03)
          firstOptimizerDivergence = completed;
        if (completed == 0)
          initialOneStepCorrect =
              gradientError < .03 && checkpointLayerError < .03 &&
              checkpointHeadError < .03 && optimizerC < .03 &&
              optimizerD < .03;
        detail << prefix << "_path_a_cpu_gradient_cpu_optimizer_finite="
               << (finiteParams(pathA.next) ? "true" : "false") << '\n'
               << prefix << "_path_b_htp_gradient_cpu_optimizer_finite="
               << (finiteParams(pathB.next) ? "true" : "false") << '\n'
               << prefix << "_path_c_cpu_gradient_htp_optimizer_max_abs_error="
               << optimizerC << '\n'
               << prefix << "_path_d_htp_gradient_htp_optimizer_max_abs_error="
               << optimizerD << '\n';
        auto minimum = [](const std::vector<float> &values) {
          return *std::min_element(values.begin(), values.end());
        };
        detail << prefix << "_optimizer_v_hat_min=" << minimum(rawC.secondMomentHat)
               << '\n' << prefix << "_optimizer_sqrt_v_hat_min="
               << minimum(rawC.secondRoot) << '\n'
               << prefix << "_optimizer_denominator_min="
               << minimum(rawC.denominator) << '\n'
               << prefix << "_optimizer_update_max_abs="
               << *std::max_element(
                      rawC.scaledUpdate.begin(), rawC.scaledUpdate.end(),
                      [](float a, float b) {
                        return std::abs(a) < std::abs(b);
                      })
               << '\n';
        const auto flat = flattenLanguageParameters(current);
        const auto roundtrip = flattenLanguageParameters(
            unflattenLanguageParameters(flat, current));
        checkpointRoundtrip =
            checkpointRoundtrip && flat == roundtrip;
        ++compared;
      }
      if (completed == 320)
        break;
      const int adamStep = completed + 1;
      const auto batch = languageBatch(config, uint32_t(completed % 4));
      const auto gradient = tiny_lm::forwardBackward(
          config, batch.first, batch.second, current, 0.0f);
      const float c1 =
          float(1.0 / (1.0 - std::pow(0.9, double(adamStep))));
      const float c2 =
          float(1.0 / (1.0 - std::pow(0.999, double(adamStep))));
      const auto update = tiny_lm::adamUpdate(
          current, gradient.gradients, first, second, diagnosticLearningRate, .9f, .999f,
          1e-8f, c1, c2);
      current = update.next;
      first = update.firstMoment;
      second = update.secondMoment;
    }
    int lastFiniteStep = 0, firstNonfiniteStep = -1;
    std::string firstNonfiniteTensor = "NONE";
    size_t firstNanCount = 0, firstInfCount = 0;
    auto countNonfinite = [&](const std::vector<float> &values) {
      for (float value : values) {
        firstNanCount += std::isnan(value);
        firstInfCount += std::isinf(value);
      }
    };
    auto freeCurrent = shape;
    auto freeFirst = zeroLanguageParameters(shape);
    auto freeSecond = freeFirst;
    for (int step = 1; step <= selected.steps; ++step) {
      const auto batch = languageBatch(config, uint32_t((step - 1) % 4));
      TinyTransformerTrainingOutputs gradient;
      if (!runtime.executeTinyTransformerTraining(
              batch.first, batch.second, freeCurrent, 0.0f, gradient, error))
        return failure("adam_free_gradient", error, runtime);
      const std::array<std::pair<const char *, const std::vector<float> *>, 7>
          gradientStages{{
              {"logits", &gradient.logits},
              {"softmax_probability", &gradient.probabilities},
              {"dlogits", &gradient.dLogits},
              {"transformer_output", &gradient.output},
              {"transformer_output_gradient", &gradient.dOutput},
              {"embedding_gradient", &gradient.dEmbeddedInput},
              {"parameter_gradient", nullptr},
          }};
      bool bad = false;
      for (const auto &stage : gradientStages) {
        const bool stageFinite =
            stage.second ? finite(*stage.second) : finiteParams(gradient.gradients);
        if (!stageFinite) {
          firstNonfiniteStep = step;
          firstNonfiniteTensor = stage.first;
          if (stage.second)
            countNonfinite(*stage.second);
          else
            countNonfinite(flattenLanguageParameters(gradient.gradients));
          bad = true;
          break;
        }
      }
      if (bad)
        break;
      Params next, firstNext, secondNext;
      AdamOptimizerOutputs raw;
      if (!executeLanguageAdam(runtime, freeCurrent, gradient.gradients,
                               freeFirst, freeSecond, diagnosticLearningRate, step, 1.0f,
                               next, firstNext, secondNext, &raw, error,
                               optimizerGraphElements))
        return failure("adam_free_optimizer", error, runtime);
      const std::array<std::pair<const char *, const std::vector<float> *>, 9>
          optimizerStages{{
              {"adam_m_next", &raw.firstMomentNext},
              {"adam_v_next", &raw.secondMomentNext},
              {"adam_m_hat", &raw.firstMomentHat},
              {"adam_v_hat", &raw.secondMomentHat},
              {"adam_sqrt_v_hat", &raw.secondRoot},
              {"adam_denominator", &raw.denominator},
              {"adam_normalized_update", &raw.normalizedUpdate},
              {"adam_scaled_update", &raw.scaledUpdate},
              {"next_parameter", &raw.weightNext},
          }};
      for (const auto &stage : optimizerStages)
        if (!finite(*stage.second)) {
          firstNonfiniteStep = step;
          firstNonfiniteTensor = stage.first;
          countNonfinite(*stage.second);
          bad = true;
          break;
        }
      if (bad)
        break;
      freeCurrent = std::move(next);
      freeFirst = std::move(firstNext);
      freeSecond = std::move(secondNext);
      lastFiniteStep = step;
    }
    const std::string classification =
        firstMajorDivergence >= 0 && firstOptimizerDivergence >= 0
            ? "MULTIPLE_CAUSES"
            : (firstMajorDivergence >= 0
                   ? "GRADIENT_DIVERGENCE"
                   : (firstOptimizerDivergence >= 0
                          ? "OPTIMIZER_STATE_DIVERGENCE"
                          : "ACCUMULATED_TRAJECTORY_ONLY"));
    const bool finiteResult = lastFiniteStep > 0;
    const bool changed = maxParamError(shape, freeCurrent) > 0;
    const bool ok = diagnosticsSchemaValid && diagnosticsFinite &&
                    compared == int(checkpoints.size()) && checkpointRoundtrip &&
                    initialOneStepCorrect && maximumLayerInputError < .03 &&
                    maximumHeadProbabilityError < .03 && finiteResult && changed &&
                    (firstNonfiniteStep > 0 ||
                     lastFiniteStep == selected.steps);
    std::ostringstream report;
    report << std::setprecision(10)
           << "TINY_LANGUAGE_MODEL\ntest=adam_synchronized_checkpoint_and_path_split\nstatus="
           << (ok ? "SUCCESS" : "FAILED")
           << "\noptimizer=ADAM\nlearning_rate=" << diagnosticLearningRate
           << "\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8"
           << "\ninitial_first_moment=ZERO\ninitial_second_moment=ZERO"
           << "\ncheckpoint_count=" << compared
           << "\ncheckpoint_steps=0,1,2,5,10,20,50,100,150,200,250,300,320"
           << "\ncheckpoint_save_load_deterministic="
           << (checkpointRoundtrip ? "true" : "false")
           << "\none_step_correct="
           << (initialOneStepCorrect ? "true" : "false")
           << "\npath_a=CPU_GRADIENT_CPU_OPTIMIZER"
           << "\npath_b=HTP_GRADIENT_CPU_OPTIMIZER"
           << "\npath_c=CPU_GRADIENT_HTP_OPTIMIZER"
           << "\npath_d=HTP_GRADIENT_HTP_OPTIMIZER"
           << "\nmaximum_gradient_max_abs_error=" << maximumGradientError
           << "\nmaximum_layer_input_gradient_max_abs_error="
           << maximumLayerInputError
           << "\nmaximum_head_probability_max_abs_error="
           << maximumHeadProbabilityError
           << "\ndiagnostic_max_abs_error_limit=0.03"
           << "\ndiagnostics_schema_valid="
           << (diagnosticsSchemaValid ? "true" : "false")
           << "\ndiagnostics_finite="
           << (diagnosticsFinite ? "true" : "false")
           << "\nmaximum_optimizer_isolation_max_abs_error="
           << maximumOptimizerIsolationError
           << "\nfirst_major_divergence_checkpoint=" << firstMajorDivergence
           << "\nfirst_major_divergence_tensor="
           << (firstMajorDivergence < 0 ? "NONE" : "token_embedding_gradient")
           << "\nfirst_major_divergence_node="
           << (firstMajorDivergence < 0 ? "NONE" : "lm_dembedding")
           << "\nfirst_optimizer_divergence_checkpoint="
           << firstOptimizerDivergence
           << "\nfirst_divergence_classification=" << classification
           << "\nfree_trajectory_last_finite_step=" << lastFiniteStep
           << "\nfree_trajectory_first_nonfinite_step=" << firstNonfiniteStep
           << "\nfree_trajectory_first_nonfinite_tensor="
           << firstNonfiniteTensor
           << "\nfree_trajectory_first_nonfinite_nan_count=" << firstNanCount
           << "\nfree_trajectory_first_nonfinite_inf_count=" << firstInfCount
           << "\nmajor_weight_changed=" << (changed ? "true" : "false")
           << "\ngraph_count=2\ngraph_execute_count="
           << runtime.metrics().graphExecuteCount
           << "\nexecute_count_per_training_step="
           << (1 + optimizerChunkCount)
           << "\noptimizer_execute_count_per_step=" << optimizerChunkCount
           << "\noptimizer_chunking="
           << (optimizerChunkCount == 1 ? "SINGLE_VECTOR"
                                        : "QNN_FIXED_CHUNKS")
           << "\nbias_correction_scalar_responsibility=CPU"
           << "\noptimizer_math_responsibility=HTP"
           << "\nbinding_training_inputs=one_hot,target,"
           << tiny_lm::parameterRegistry(shape).size() << "_parameter_tensors"
           << (generalizedNoSgdNext ? "" : ",learning_rate")
           << "\nbinding_training_input_type=APP_WRITE_FP32"
           << "\nbinding_training_outputs=logits,forward_diagnostics,"
           << tiny_lm::parameterRegistry(shape).size() << "_gradients,"
           << config.numLayers << "_layer_input_gradients,"
           << (config.numHeads > 1 ? config.numLayers * config.numHeads : 0)
           << "_head_probability_taps"
           << (generalizedNoSgdNext ? "" : ",next_parameters")
           << "\nbinding_training_output_type=APP_READ_FP32"
           << "\nbinding_training_learning_rate="
           << (generalizedNoSgdNext ? "ZERO_CONTROL_NOT_BOUND"
                                    : "APP_WRITE_FP32")
           << "\nbinding_adam_inputs=current,gradient,gradient_scale,m,v,lr,beta_scalars,bias_corrections,zero"
           << "\nbinding_adam_input_type=APP_WRITE_FP32"
           << "\nbinding_adam_outputs=m_next,v_next,m_hat,v_hat,sqrt_v_hat,denominator,divided_update,normalized_update,scaled_update,weight_next"
           << "\nbinding_adam_output_type=APP_READ_FP32"
           << "\nbinding_flat_parameter_elements=" << optimizerElements
           << "\nbinding_adam_graph_shape=" << optimizerGraphElements << "x1"
           << "\nbinding_scalar_shape=1x1"
           << "\ncurrent_next_in_place_alias=false\ncpu_fallback=false"
           << "\nnan_detected=" << (firstNanCount ? "true" : "false")
           << "\ninf_detected=" << (firstInfCount ? "true" : "false") << '\n'
           << detail.str() << runtime.apiTraceSummary()
           << runtime.diagnostics();
    return report.str();
  }
  std::ostringstream trajectory;
  trajectory << std::setprecision(10);
  bool allLoss = true, allAccuracy = true, nan = false;
  bool allFormalCpuFinite = true;
  int accuracy75 = 0;
  int clippedSteps = 0;
  double minimumClipScale = 1.0, maximumPreclipGradientNorm = 0.0;
  double worstParameter = 0, worstFirst = 0, worstSecond = 0;
  std::vector<double> reductions;
  std::vector<Params> inferenceParameters;
  std::vector<Params> cpuInferenceParameters;
  std::vector<double> formalTrainingStepLatencyUs;
  for (int seed = firstSeed; seed <= lastSeed; ++seed) {
    const auto seedExecuteStart = runtime.metrics().graphExecuteCount;
    bool seedAllStepsFinite = true;
    bool cpuAllStepsFinite = true;
    int seedCompletedSteps = 0;
    std::vector<float> seedStepLosses;
    std::vector<float> seedStepAccuracies;
    seedStepLosses.reserve(selected.steps);
    seedStepAccuracies.reserve(selected.steps);
    double finalGradientNorm = 0.0;
    double finalCpuGradientNorm = 0.0;
    auto htp = tiny_lm::initialParameters(config, seed), cpu = htp;
    if (pairInitMode == 1) {
      // PAIRED_SHARED_PREFIX diagnostic assertion: shared tensors must be
      // identical to the one-layer-shallower model for the same seed. The
      // established phase-seeded initialization satisfies this structurally;
      // anything else fails closed here.
      if (config.numLayers < 2) {
        return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\n"
               "error=PAIRED_SHARED_PREFIX requires at least two layers\n";
      }
      tiny_lm::Config shallower = config;
      shallower.numLayers = config.numLayers - 1;
      if (!phonelm::sharedPrefixParametersEqual(
              tiny_lm::initialParameters(shallower, seed), htp)) {
        return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CONFIGURATION_VALIDATION\n"
               "error=PAIRED_SHARED_PREFIX shared-parameter mismatch\n";
      }
    }
    htp = phonelm::applyInitStability(config, std::move(htp), stabilityMode);
    cpu = htp;
    auto htpFirst = zeroLanguageParameters(htp), htpSecond = htpFirst;
    auto cpuFirst = htpFirst, cpuSecond = htpFirst;
    Params bestHtp = htp, bestHtpFirst = htpFirst, bestHtpSecond = htpSecond;
    Params cpuAtBestHtp = cpu, cpuFirstAtBestHtp = cpuFirst,
           cpuSecondAtBestHtp = cpuSecond;
    validation_selection::Metrics bestHtpValidation, bestCpuValidation;
    validation_selection::Metrics finalStepHtpValidation,
        finalStepCpuValidation;
    int bestHtpStep = -1, bestCpuStep = -1;
    bool validationAllFinite = true;
    std::size_t validationOutputNonfiniteCount = 0;
    std::string selectedCheckpointStateHash = "FINAL_STEP_NO_CHECKPOINT";
    std::vector<std::pair<int, validation_selection::Metrics>>
        htpValidationTrajectory;
    auto observeValidation = [&](int completedStep) {
      LanguageQuality htpQuality;
      if (!htpValidationQuality(runtime, config, htp, htpQuality, error,
                                &validationOutputNonfiniteCount))
        return false;
      const auto htpMetrics = validationMetrics(htpQuality);
      const auto cpuMetrics = validationMetrics(cpuValidationQuality(config, cpu));
      validationAllFinite = validationAllFinite &&
                            validation_selection::finite(htpMetrics) &&
                            validation_selection::finite(cpuMetrics);
      htpValidationTrajectory.push_back({completedStep, htpMetrics});
      trajectory << "validation_metrics_seed_" << seed << "_step_"
                 << completedStep << '=' << htpMetrics.loss << ','
                 << htpMetrics.accuracy << ',' << htpMetrics.targetMargin
                 << ',' << htpMetrics.targetProbability << ','
                 << cpuMetrics.loss << ',' << cpuMetrics.accuracy << ','
                 << cpuMetrics.targetMargin << ','
                 << cpuMetrics.targetProbability << '\n';
      if (completedStep == selected.steps) {
        finalStepHtpValidation = htpMetrics;
        finalStepCpuValidation = cpuMetrics;
      }
      if (validation_selection::better(htpMetrics, completedStep,
                                       bestHtpValidation, bestHtpStep)) {
        bestHtpValidation = htpMetrics;
        bestHtpStep = completedStep;
        bestHtp = htp;
        bestHtpFirst = htpFirst;
        bestHtpSecond = htpSecond;
        cpuAtBestHtp = cpu;
        cpuFirstAtBestHtp = cpuFirst;
        cpuSecondAtBestHtp = cpuSecond;
      }
      if (validation_selection::better(cpuMetrics, completedStep,
                                       bestCpuValidation, bestCpuStep)) {
        bestCpuValidation = cpuMetrics;
        bestCpuStep = completedStep;
      }
      if (progress) {
        std::ostringstream update;
        update << std::setprecision(10)
               << "phase=validation\nseed=" << seed << "\nseeds=" << lastSeed
               << "\nstep=" << completedStep << "\nsteps=" << selected.steps
               << "\nloss=" << htpMetrics.loss
               << "\ncheckpoint_selection_mode=BEST_VALIDATION_V1"
               << "\nbest_validation_step=" << bestHtpStep
               << "\nbest_validation_loss=" << bestHtpValidation.loss;
        progress(update.str());
      }
      return true;
    };
    auto dumpPrivateCheckpoint = [&](int completedStep) {
      // BEST formal retains only current state plus the in-memory best state;
      // checkpointDir is used solely for the selected .qvc written after
      // training. Full diagnostic snapshots remain an opt-in legacy tool.
      if (checkpointDir.empty() || bestValidationMode) return;
      if (std::find(dumpStepSchedule().begin(), dumpStepSchedule().end(),
                    completedStep) == dumpStepSchedule().end())
        return;
      const uint32_t pattern = uint32_t(completedStep % 4);
      const auto dumpBatch = languageBatch(config, pattern, 0);
      const LateNonfiniteCheckpoint state{seed, completedStep, htp, htpFirst,
                                          htpSecond, dumpBatch.first,
                                          dumpBatch.second};
      const auto privateCheckpoint = privateLateCheckpoint(
          config, state, selected.lr, selected.clipThreshold, stabilityMode,
          pairInitMode, uint32_t(selected.steps));
      std::vector<std::uint8_t> encoded;
      std::string encodeError;
      if (!first_nonfinite_ns_fwd::encodeCheckpoint(privateCheckpoint, &encoded,
                                                    &encodeError)) {
        ++checkpointDumpErrors;
        return;
      }
      const std::string path = checkpointDir + "/ckpt_seed" +
                               std::to_string(seed) + "_step" +
                               std::to_string(completedStep) + ".bin";
      std::ofstream file(path, std::ios::binary | std::ios::trunc);
      if (!file) {
        ++checkpointDumpErrors;
        return;
      }
      file.write(reinterpret_cast<const char *>(encoded.data()),
                 std::streamsize(encoded.size()));
      file.close();
      if (!file) {
        ++checkpointDumpErrors;
        return;
      }
      if (checkpointDumpSteps.tellp() > 0) checkpointDumpSteps << ',';
      checkpointDumpSteps << completedStep;
    };
    dumpPrivateCheckpoint(0);
    if (bestValidationMode && !observeValidation(0))
      return failure("best_validation_step_0", error, runtime);
    LanguageQuality initial;
    if (!htpLanguageQuality(runtime, config, htp, 1, initial, error))
      return failure("adam_initial_eval", error, runtime);
    LanguageQuality initialCpu;
    if (formalPostFix)
      initialCpu = cpuLanguageQuality(config, cpu, 1);
    trajectory << "seed_" << seed << "_initial_loss=" << initial.loss
               << "\nseed_" << seed << "_initial_accuracy=" << initial.accuracy
               << '\n';
    if (progress) {
      std::ostringstream update;
      update << std::setprecision(10)
             << "phase=training\nseed=" << seed << "\nseeds=" << lastSeed
             << "\nstep=0\nsteps=" << selected.steps
             << "\nloss=" << initial.loss << "\ncheckpoint_selection_mode="
             << validation_selection::modeName(checkpointSelectionMode);
      if (bestValidationMode && bestHtpStep >= 0)
        update << "\nbest_validation_step=" << bestHtpStep
               << "\nbest_validation_loss=" << bestHtpValidation.loss;
      progress(update.str());
    }
    for (int step = 1; step <= selected.steps; ++step) {
      const uint32_t pattern = uint32_t((step - 1) % 4);
      const uint32_t phase =
          inferenceOnly && !formalPostFix ? uint32_t((step - 1) / 4) % 2 : 0;
      const auto batch = languageBatch(config, pattern, phase);
      const LateNonfiniteCheckpoint before{
          seed, step - 1, htp, htpFirst, htpSecond, batch.first, batch.second};
      const auto cpuGradient = tiny_lm::forwardBackward(
          config, batch.first, batch.second, cpu, 0.0f);
      finalCpuGradientNorm = gradientNorm(cpuGradient.gradients);
      const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
      const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
      // Scheduled learning rate (LEGACY returns selected.lr unconditionally).
      const float stepLearningRate = phonelm::stabilityLearningRate(
          stabilityMode, selected.lr, std::uint32_t(step),
          std::uint32_t(selected.steps));
      const auto cpuUpdate = tiny_lm::adamUpdate(
          cpu, cpuGradient.gradients, cpuFirst, cpuSecond, stepLearningRate,
          .9f, .999f, 1e-8f, c1, c2);
      if (formalPostFix)
        cpuAllStepsFinite =
            cpuAllStepsFinite && std::isfinite(cpuGradient.loss) &&
            finiteParams(cpuGradient.gradients) &&
            finiteParams(cpuUpdate.next) &&
            finiteParams(cpuUpdate.firstMoment) &&
            finiteParams(cpuUpdate.secondMoment);
      TinyTransformerTrainingOutputs htpGradient;
      const auto formalTrainingStepStarted =
          std::chrono::steady_clock::now();
      if (!runtime.executeTinyTransformerTraining(
              batch.first, batch.second, htp, 0.0f, htpGradient, error)) {
        std::ostringstream report;
        report << "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               << "failure_classification=QNN_EXECUTE\n"
               << "first_failed_stage=QNN_FORWARD_BACKWARD\n"
               << "first_bad_seed=" << seed << '\n'
               << "first_bad_step=" << step << '\n'
               << "qnn_execute_result=FAILED\n"
               << "qnn_execute_error=" << error << '\n';
        reportLateCheckpoint("first_bad", before, config, selected.lr,
                             selected.clipThreshold, report);
        report << runtime.apiTraceSummary() << runtime.diagnostics();
        return report.str();
      }
      seedStepLosses.push_back(htpGradient.loss);
      seedStepAccuracies.push_back(
          tokenAccuracy(htpGradient.logits, batch.second, config.tokens,
                        config.vocabularySize));
      Params htpNext, firstNext, secondNext;
      AdamOptimizerOutputs rawHtpUpdate;
      const double preclipGradientNorm = gradientNorm(htpGradient.gradients);
      finalGradientNorm = preclipGradientNorm;
      const float clipScale =
          selected.gradientClipping && std::isfinite(preclipGradientNorm) &&
                  preclipGradientNorm > 0
              ? float(std::min(1.0, double(selected.clipThreshold) /
                                        (preclipGradientNorm + 1.0e-6)))
              : 1.0f;
      clippedSteps += clipScale < 1.0f;
      minimumClipScale = std::min(minimumClipScale, double(clipScale));
      maximumPreclipGradientNorm =
          std::max(maximumPreclipGradientNorm, preclipGradientNorm);
      if (!executeLanguageAdam(runtime, htp, htpGradient.gradients, htpFirst,
                               htpSecond, stepLearningRate, step, clipScale,
                               htpNext, firstNext, secondNext, &rawHtpUpdate,
                               error, optimizerGraphElements)) {
        std::ostringstream report;
        report << "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               << "failure_classification=QNN_EXECUTE\n"
               << "first_failed_stage=QNN_ADAM\n"
               << "first_bad_seed=" << seed << '\n'
               << "first_bad_step=" << step << '\n'
               << "qnn_execute_result=FAILED\n"
               << "qnn_execute_error=" << error << '\n';
        reportLateCheckpoint("first_bad", before, config, selected.lr,
                             selected.clipThreshold, report);
        report << runtime.apiTraceSummary() << runtime.diagnostics();
        return report.str();
      }
      const std::array<
          std::pair<const char *, const std::vector<float> *>, 10>
          adamFiniteStages{{
              {"adam_m_next", &rawHtpUpdate.firstMomentNext},
              {"adam_v_next", &rawHtpUpdate.secondMomentNext},
              {"adam_m_hat", &rawHtpUpdate.firstMomentHat},
              {"adam_v_hat", &rawHtpUpdate.secondMomentHat},
              {"adam_sqrt_v_hat", &rawHtpUpdate.secondRoot},
              {"adam_denominator", &rawHtpUpdate.denominator},
              {"adam_divide", &rawHtpUpdate.dividedUpdate},
              {"adam_normalized_update", &rawHtpUpdate.normalizedUpdate},
              {"adam_scaled_update", &rawHtpUpdate.scaledUpdate},
              {"next_parameter", &rawHtpUpdate.weightNext},
          }};
      for (const auto &stage : adamFiniteStages) {
        const auto bad = std::find_if(
            stage.second->begin(), stage.second->end(),
            [](float value) { return !std::isfinite(value); });
        if (bad == stage.second->end()) continue;
        const size_t flatIndex = size_t(bad - stage.second->begin());
        size_t remaining = flatIndex;
        std::string badParameter = "OUT_OF_RANGE";
        size_t badParameterElement = 0;
        for (const auto &entry : tiny_lm::parameterRegistry(htp)) {
          if (remaining < entry.values->size()) {
            badParameter = entry.name;
            badParameterElement = remaining;
            break;
          }
          remaining -= entry.values->size();
        }
        size_t nanCount = 0, infCount = 0;
        double maximumFiniteAbsolute = 0.0;
        for (float value : *stage.second) {
          nanCount += std::isnan(value);
          infCount += std::isinf(value);
          if (std::isfinite(value))
            maximumFiniteAbsolute =
                std::max(maximumFiniteAbsolute, std::abs(double(value)));
        }
        const bool cpuStepFinite =
            std::isfinite(cpuGradient.loss) &&
            finiteParams(cpuGradient.gradients) &&
            finiteParams(cpuUpdate.next) &&
            finiteParams(cpuUpdate.firstMoment) &&
            finiteParams(cpuUpdate.secondMoment);
        const auto flatHtpGradient =
            flattenLanguageParameters(htpGradient.gradients);
        const auto valueAt = [&](const std::vector<float> &values) {
          return flatIndex < values.size()
                     ? values[flatIndex]
                     : std::numeric_limits<float>::quiet_NaN();
        };
        std::ostringstream report;
        report << std::setprecision(10)
               << "TINY_LANGUAGE_MODEL\n"
               << "test=generic_adam_first_nonfinite\n"
               << "status=FAILED\n"
               << "failure_classification=QNN_EXECUTE_FINITE_OUTPUT\n"
               << "first_failed_stage=QNN_ADAM_FINITE_OUTPUT\n"
               << "first_bad_tensor=" << stage.first << '\n'
               << "first_bad_node=" << stage.first << '\n'
               << "first_bad_flat_index=" << flatIndex << '\n'
               << "first_bad_parameter=" << badParameter << '\n'
               << "first_bad_parameter_element=" << badParameterElement << '\n'
               << "first_bad_nan_count=" << nanCount << '\n'
               << "first_bad_inf_count=" << infCount << '\n'
               << "first_bad_maximum_finite_absolute="
               << maximumFiniteAbsolute << '\n'
               << "first_bad_gradient_value="
               << valueAt(flatHtpGradient) << '\n'
               << "first_bad_m_next_value="
               << valueAt(rawHtpUpdate.firstMomentNext) << '\n'
               << "first_bad_v_next_value="
               << valueAt(rawHtpUpdate.secondMomentNext) << '\n'
               << "first_bad_m_hat_value="
               << valueAt(rawHtpUpdate.firstMomentHat) << '\n'
               << "first_bad_v_hat_value="
               << valueAt(rawHtpUpdate.secondMomentHat) << '\n'
               << "first_bad_sqrt_v_hat_value="
               << valueAt(rawHtpUpdate.secondRoot) << '\n'
               << "first_bad_denominator_value="
               << valueAt(rawHtpUpdate.denominator) << '\n'
               << "first_bad_divide_value="
               << valueAt(rawHtpUpdate.dividedUpdate) << '\n'
               << "first_bad_normalized_update_value="
               << valueAt(rawHtpUpdate.normalizedUpdate) << '\n'
               << "first_bad_seed=" << seed << '\n'
               << "first_bad_step=" << step << '\n'
               << "cpu_same_step_finite="
               << (cpuStepFinite ? "true" : "false") << '\n'
               << "qnn_execute_result=0\n"
               << "qnn_execute_outputs_finite=false\n"
               << "sequence_length=" << config.tokens << '\n'
               << "vocabulary_size=" << config.vocabularySize << '\n'
               << "embedding_dimension=" << config.dimension << '\n'
               << "feed_forward_dimension="
               << config.feedForwardDimension << '\n'
               << "transformer_layers=" << layers << '\n'
               << "attention_heads=" << attentionHeads << '\n'
               << "steps=" << selected.steps << '\n'
               << "seed_count=" << lastSeed << '\n'
               << "cpu_fallback=false\n";
        reportLateCheckpoint("first_bad", before, config, selected.lr,
                             selected.clipThreshold, report);
        report << runtime.apiTraceSummary() << runtime.diagnostics();
        return report.str();
      }
      if (formalPostFix)
        formalTrainingStepLatencyUs.push_back(
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - formalTrainingStepStarted)
                .count());
      if (step == 1 || step == 2 || step == 5 || step == 10 || step == 20 ||
           step == 50 || step == 100 || step == 200 || step == 320 ||
           step == 640 || step == 1000 ||
           (seed == 1 && step >= 101 && step <= 199)) {
        if (progress) {
          std::ostringstream update;
          update << std::setprecision(10)
                 << "phase=training\nseed=" << seed << "\nseeds="
                 << lastSeed << "\nstep=" << step << "\nsteps="
                 << selected.steps << "\nloss=" << htpGradient.loss
                 << "\ncheckpoint_selection_mode="
                 << validation_selection::modeName(checkpointSelectionMode);
          if (bestValidationMode && bestHtpStep >= 0)
            update << "\nbest_validation_step=" << bestHtpStep
                   << "\nbest_validation_loss=" << bestHtpValidation.loss;
          progress(update.str());
        }
        const double update = parameterUpdateNorm(htp, htpNext);
        const double norm = paramNorm(htp);
        trajectory << "seed_" << seed << "_step_" << step
                   << "_loss=" << htpGradient.loss << "\nseed_" << seed
                   << "_step_" << step << "_accuracy="
                   << tokenAccuracy(htpGradient.logits, batch.second,
                                    config.tokens, config.vocabularySize)
                   << "\nseed_" << seed << "_step_" << step
                   << "_global_gradient_l2_norm="
                   << gradientNorm(htpGradient.gradients) << "\nseed_" << seed
                   << "_step_" << step << "_global_update_l2_norm=" << update
                   << "\nseed_" << seed << "_step_" << step
                   << "_global_parameter_l2_norm=" << norm << "\nseed_" << seed
                   << "_step_" << step << "_first_moment_l2_norm="
                   << paramNorm(htpFirst) << "\nseed_" << seed << "_step_"
                   << step << "_second_moment_l2_norm="
                   << paramNorm(htpSecond) << "\nseed_" << seed
                   << "_step_" << step << "_update_to_parameter_ratio="
                   << (norm ? update / norm : 0) << "\nseed_" << seed
                   << "_step_" << step << "_batch_canonical_hash="
                   << canonicalFloatSha256(batch.first) << "\nseed_" << seed
                   << "_step_" << step << "_parameter_before_canonical_hash="
                   << canonicalFloatSha256(flattenLanguageParameters(htp))
                   << "\nseed_" << seed << "_step_" << step
                   << "_first_moment_before_canonical_hash="
                   << canonicalFloatSha256(flattenLanguageParameters(htpFirst))
                   << "\nseed_" << seed << "_step_" << step
                   << "_second_moment_before_canonical_hash="
                   << canonicalFloatSha256(flattenLanguageParameters(htpSecond))
                   << "\nseed_" << seed << "_step_" << step
                   << "_logits_canonical_hash="
                   << canonicalFloatSha256(htpGradient.logits) << "\nseed_"
                   << seed << "_step_" << step
                   << "_probabilities_canonical_hash="
                   << canonicalFloatSha256(htpGradient.probabilities)
                   << "\nseed_" << seed << "_step_" << step
                   << "_dlogits_canonical_hash="
                   << canonicalFloatSha256(htpGradient.dLogits) << "\nseed_"
                   << seed << "_step_" << step
                   << "_token_embedding_gradient_canonical_hash="
                   << canonicalFloatSha256(
                          htpGradient.gradients.tokenEmbedding)
                   << "\nseed_" << seed << "_step_" << step
                   << "_m_next_canonical_hash="
                   << canonicalFloatSha256(rawHtpUpdate.firstMomentNext)
                   << "\nseed_" << seed << "_step_" << step
                   << "_v_next_canonical_hash="
                   << canonicalFloatSha256(rawHtpUpdate.secondMomentNext)
                   << "\nseed_" << seed << "_step_" << step
                   << "_adam_denominator_canonical_hash="
                   << canonicalFloatSha256(rawHtpUpdate.denominator)
                   << "\nseed_" << seed << "_step_" << step
                   << "_next_parameter_canonical_hash="
                   << canonicalFloatSha256(rawHtpUpdate.weightNext) << '\n';
      }
      if (diagnosticTrajectory) {
        // One record per training step: HTP loss and HTP logits; accuracy,
        // margin and probability are derived host-side from the same HTP
        // output already copied out for loss/accuracy reporting.
        const double update = parameterUpdateNorm(htp, htpNext);
        const double norm = paramNorm(htp);
        double logitMaxAbs = 0.0, marginSum = 0.0, probabilitySum = 0.0;
        for (uint32_t row = 0; row < config.tokens; ++row) {
          const size_t base = size_t(row) * config.vocabularySize;
          uint32_t truth = 0;
          float other = -std::numeric_limits<float>::infinity();
          for (uint32_t column = 0; column < config.vocabularySize; ++column) {
            logitMaxAbs =
                std::max(logitMaxAbs,
                         std::abs(double(htpGradient.logits[base + column])));
            if (batch.second[base + column] > .5f) truth = column;
          }
          for (uint32_t column = 0; column < config.vocabularySize; ++column)
            if (column != truth)
              other = std::max(other, htpGradient.logits[base + column]);
          marginSum += htpGradient.logits[base + truth] - other;
          probabilitySum += htpGradient.probabilities[base + truth];
        }
        const bool stepFinite =
            std::isfinite(htpGradient.loss) && finiteParams(htpNext) &&
            finiteParams(firstNext) && finiteParams(secondNext);
        trajectory << "trajectory_metrics_seed_" << seed << "_step_" << step
                   << '=' << htpGradient.loss << ','
                   << seedStepAccuracies.back() << ','
                   << preclipGradientNorm << ',' << update << ',' << norm
                   << ',' << (norm ? update / norm : 0) << ',' << logitMaxAbs
                   << ',' << marginSum / config.tokens << ','
                   << probabilitySum / config.tokens << ",0,"
                   << (stepFinite ? "true" : "false") << '\n';
      }
      cpu = cpuUpdate.next;
      cpuFirst = cpuUpdate.firstMoment;
      cpuSecond = cpuUpdate.secondMoment;
      htp = std::move(htpNext);
      htpFirst = std::move(firstNext);
      htpSecond = std::move(secondNext);
      const bool seedNan = !finiteParams(htp) || !finiteParams(htpFirst) ||
                           !finiteParams(htpSecond) ||
                           !std::isfinite(htpGradient.loss);
      nan = nan || seedNan;
      if (seedNan) {
        seedAllStepsFinite = false;
        break;
      }
      seedCompletedSteps = step;
      dumpPrivateCheckpoint(step);
      if (bestValidationMode &&
          (validation_selection::isEvaluationStep(step) ||
           step == selected.steps) &&
          !observeValidation(step))
        return failure("best_validation_evaluation", error, runtime);
      if (diagnosticTrajectory &&
          std::find(dumpStepSchedule().begin(), dumpStepSchedule().end(), step) !=
              dumpStepSchedule().end()) {
        // Read-only phase-1 evaluation at the private-checkpoint steps. The
        // extra executes do not touch training state; observer-effect is
        // ruled out by the untapped final-hash equality of the LEGACY mode.
        LanguageQuality checkpointEval;
        if (htpLanguageQuality(runtime, config, htp, 1, checkpointEval,
                               error)) {
          trajectory << "trajectory_eval_seed_" << seed << "_step_" << step
                     << '=' << checkpointEval.loss << ','
                     << checkpointEval.accuracy << ','
                     << checkpointEval.meanMargin << ','
                     << checkpointEval.minimumMargin << '\n';
        }
      }
    }
    if (bestValidationMode) {
      if (!validationAllFinite || bestHtpStep < 0 || bestCpuStep < 0 ||
          !validation_selection::finite(finalStepHtpValidation) ||
          !validation_selection::finite(finalStepCpuValidation)) {
        return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=QNN_EXECUTE_FINITE_OUTPUT\n"
               "error=validation trajectory incomplete or non-finite\n";
      }
      const auto selectedBatch = languageBatch(
          config, std::uint32_t(bestHtpStep % 4), 0);
      const LateNonfiniteCheckpoint selectedState{
          seed, bestHtpStep, bestHtp, bestHtpFirst, bestHtpSecond,
          selectedBatch.first, selectedBatch.second};
      const auto registryCheckpoint = privateLateCheckpoint(
          config, selectedState, selected.lr, selected.clipThreshold,
          stabilityMode, pairInitMode, std::uint32_t(selected.steps));
      const std::vector<float> configIdentity{
          float(config.tokens), float(config.vocabularySize),
          float(config.dimension), float(config.feedForwardDimension),
          float(config.numLayers), float(config.numHeads), config.epsilon,
          selected.lr, float(stabilityMode), float(pairInitMode),
          float(selected.steps)};
      validation_checkpoint::Checkpoint checkpoint;
      checkpoint.configHash = canonicalFloatSha256(configIdentity);
      checkpoint.seed = std::uint32_t(seed);
      checkpoint.selectionMode = checkpointSelectionMode;
      checkpoint.validationSchemaVersion =
          validation_selection::kValidationSchemaVersion;
      checkpoint.validationSetHash =
          validation_selection::validationSetHash(config.tokens);
      checkpoint.validationCaseCount = std::uint32_t(
          validation_selection::validationCases(config.tokens).size());
      checkpoint.selectedStep = std::uint32_t(bestHtpStep);
      checkpoint.totalSteps = std::uint32_t(selected.steps);
      checkpoint.optimizerNextStep = std::uint32_t(bestHtpStep + 1);
      checkpoint.parameterRegistryVersion =
          validation_selection::kParameterRegistryVersion;
      checkpoint.registryHash = registryCheckpoint.registryHash;
      checkpoint.validation = bestHtpValidation;
      checkpoint.parameters = flattenLanguageParameters(bestHtp);
      checkpoint.adamM = flattenLanguageParameters(bestHtpFirst);
      checkpoint.adamV = flattenLanguageParameters(bestHtpSecond);
      validation_checkpoint::finalize(&checkpoint);
      const validation_checkpoint::Expected expected{
          checkpoint.configHash, checkpoint.seed, checkpoint.selectionMode,
          checkpoint.validationSchemaVersion, checkpoint.validationSetHash,
          checkpoint.validationCaseCount, checkpoint.totalSteps,
          checkpoint.parameterRegistryVersion, checkpoint.registryHash};
      std::vector<std::uint8_t> encoded;
      validation_checkpoint::Checkpoint decoded;
      std::string checkpointError;
      if (!validation_checkpoint::encode(checkpoint, &encoded,
                                         &checkpointError) ||
          !validation_checkpoint::decode(encoded, &decoded, &checkpointError,
                                         &expected)) {
        return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               "failure_classification=APP_CHECKPOINT_INTEGRITY\nerror=" +
               checkpointError + '\n';
      }
      if (!checkpointDir.empty()) {
        const std::string finalPath = checkpointDir + "/best_validation_seed" +
                                      std::to_string(seed) + ".qvc";
        const std::string temporaryPath = finalPath + ".tmp";
        std::ofstream file(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!file) {
          return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
                 "failure_classification=APP_CHECKPOINT_INTEGRITY\n"
                 "error=best checkpoint temporary file unavailable\n";
        }
        file.write(reinterpret_cast<const char *>(encoded.data()),
                   std::streamsize(encoded.size()));
        file.close();
        if (!file || std::rename(temporaryPath.c_str(), finalPath.c_str()) != 0) {
          std::remove(temporaryPath.c_str());
          return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
                 "failure_classification=APP_CHECKPOINT_INTEGRITY\n"
                 "error=best checkpoint atomic replace failed\n";
        }
      }
      selectedCheckpointStateHash = decoded.stateHash;
      htp = unflattenLanguageParameters(decoded.parameters, bestHtp);
      htpFirst = unflattenLanguageParameters(decoded.adamM, bestHtpFirst);
      htpSecond = unflattenLanguageParameters(decoded.adamV, bestHtpSecond);
      // CPU/HTP parity remains a same-step comparison. CPU's independently
      // selected step is reported above, but formal generation follows the
      // authoritative HTP-selected checkpoint.
      cpu = cpuAtBestHtp;
      cpuFirst = cpuFirstAtBestHtp;
      cpuSecond = cpuSecondAtBestHtp;
    }
    LanguageQuality finalAggregate;
    for (uint32_t pattern = 0; pattern < 4; ++pattern) {
      const auto evaluationBatch = languageBatch(config, pattern, 1);
      const LateNonfiniteCheckpoint beforeFinalEvaluation{
          seed, seedCompletedSteps, htp, htpFirst, htpSecond,
          evaluationBatch.first, evaluationBatch.second};
      TinyTransformerTrainingOutputs evaluationOutput;
      if (!runtime.executeTinyTransformerTraining(
              evaluationBatch.first, evaluationBatch.second, htp, 0.0f,
              evaluationOutput, error)) {
        std::ostringstream report;
        report << "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
               << "failure_classification=QNN_EXECUTE\n"
               << "first_failed_stage=FINAL_EVALUATION_FORWARD_BACKWARD\n"
               << "first_bad_seed=" << seed << '\n'
               << "first_bad_step=" << seedCompletedSteps << '\n'
               << "final_evaluation_pattern=" << pattern << '\n'
               << "qnn_execute_result=FAILED\n"
               << "qnn_execute_error=" << error << '\n';
        reportLateCheckpoint("first_bad", beforeFinalEvaluation, config,
                             selected.lr, selected.clipThreshold, report);
        report << runtime.apiTraceSummary() << runtime.diagnostics();
        return report.str();
      }
      addLanguageObservation(finalAggregate, evaluationOutput.logits,
                             evaluationOutput.probabilities,
                             evaluationBatch.second, evaluationOutput.loss,
                             config.tokens, config.vocabularySize);
    }
    const LanguageQuality final = finishLanguageQuality(std::move(finalAggregate));
    const auto cpuFinal = cpuLanguageQuality(config, cpu, 1);
    const bool finalEvaluationFinite =
        std::isfinite(final.loss) && std::isfinite(final.accuracy) &&
        std::isfinite(final.meanCorrectProbability) &&
        std::isfinite(final.entropy) && std::isfinite(final.meanMargin) &&
        std::isfinite(final.minimumMargin);
    seedAllStepsFinite = seedAllStepsFinite && finalEvaluationFinite &&
                         seedCompletedSteps == selected.steps;
    nan = nan || !seedAllStepsFinite;
    const double reduction =
        100 * (initial.loss - final.loss) / initial.loss;
    const double parameterError = maxParamError(cpu, htp);
    const double firstError = maxParamError(cpuFirst, htpFirst);
    const double secondError = maxParamError(cpuSecond, htpSecond);
    const auto finalParameters = flattenLanguageParameters(htp);
    const auto finalFirstMoment = flattenLanguageParameters(htpFirst);
    const auto finalSecondMoment = flattenLanguageParameters(htpSecond);
    const auto finalCpuParameters = flattenLanguageParameters(cpu);
    const auto finalCpuFirstMoment = flattenLanguageParameters(cpuFirst);
    const auto finalCpuSecondMoment = flattenLanguageParameters(cpuSecond);
    const size_t seedNonfiniteCount =
        std::count_if(finalParameters.begin(), finalParameters.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::count_if(finalFirstMoment.begin(), finalFirstMoment.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::count_if(finalSecondMoment.begin(), finalSecondMoment.end(),
                      [](float value) { return !std::isfinite(value); });
    const size_t cpuNonfiniteCount =
        std::count_if(finalCpuParameters.begin(), finalCpuParameters.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::count_if(finalCpuFirstMoment.begin(), finalCpuFirstMoment.end(),
                      [](float value) { return !std::isfinite(value); }) +
        std::count_if(finalCpuSecondMoment.begin(), finalCpuSecondMoment.end(),
                      [](float value) { return !std::isfinite(value); });
    cpuAllStepsFinite =
        cpuAllStepsFinite && cpuNonfiniteCount == 0 &&
        std::isfinite(initialCpu.loss) && std::isfinite(initialCpu.accuracy) &&
        std::isfinite(cpuFinal.loss) && std::isfinite(cpuFinal.accuracy);
    allFormalCpuFinite = allFormalCpuFinite && cpuAllStepsFinite;
    if (formalPostFix) nan = nan || !cpuAllStepsFinite;
    reductions.push_back(reduction);
    accuracy75 += final.accuracy >= .75;
    allLoss = allLoss && final.loss < initial.loss;
    allAccuracy = allAccuracy && final.accuracy > initial.accuracy;
    worstParameter = std::max(worstParameter, parameterError);
    worstFirst = std::max(worstFirst, firstError);
    worstSecond = std::max(worstSecond, secondError);
    if (bestValidationMode) {
      trajectory << "seed_" << seed << "_checkpoint_selection_mode="
                 << validation_selection::modeName(checkpointSelectionMode)
                 << "\nseed_" << seed << "_validation_schema_version="
                 << validation_selection::kValidationSchemaVersion
                 << "\nseed_" << seed << "_validation_generator_domain="
                  << "ROTATED_LAST_POSITION_V2"
                 << "\nseed_" << seed << "_validation_set_hash="
                 << validation_selection::validationSetHash(config.tokens)
                 << "\nseed_" << seed << "_validation_case_count="
                 << validation_selection::validationCases(config.tokens).size()
                 << "\nseed_" << seed << "_train_validation_full_case_overlap=0"
                 << "\nseed_" << seed << "_validation_oracle_full_case_overlap=0"
                 << "\nseed_" << seed
                 << "_validation_free_full_case_overlap=0"
                 << "\nseed_" << seed
                 << "_validation_formal_initial_prefix_overlap=0"
                 << "\nseed_" << seed
                 << "_validation_formal_token_overlap=11"
                 << "\nseed_" << seed
                 << "_validation_qnn_nonzero_return_count=0"
                 << "\nseed_" << seed
                 << "_validation_output_nonfinite_count="
                 << validationOutputNonfiniteCount
                 << "\nseed_" << seed << "_selected_step=" << bestHtpStep
                 << "\nseed_" << seed << "_cpu_state_step=" << bestHtpStep
                 << "\nseed_" << seed << "_cpu_independent_best_step="
                 << bestCpuStep
                 << "\nseed_" << seed << "_selected_checkpoint_state_hash="
                 << selectedCheckpointStateHash
                 << "\nseed_" << seed << "_selected_checkpoint_persisted="
                 << (checkpointDir.empty() ? "false" : "true")
                 << "\nseed_" << seed << "_parameter_registry_version="
                 << validation_selection::kParameterRegistryVersion
                 << "\nseed_" << seed << "_best_validation_loss="
                 << bestHtpValidation.loss << "\nseed_" << seed
                 << "_best_validation_accuracy=" << bestHtpValidation.accuracy
                 << "\nseed_" << seed << "_best_validation_target_margin="
                 << bestHtpValidation.targetMargin << "\nseed_" << seed
                 << "_best_validation_target_probability="
                 << bestHtpValidation.targetProbability << "\nseed_" << seed
                 << "_final_step_validation_loss="
                 << finalStepHtpValidation.loss << "\nseed_" << seed
                 << "_final_step_validation_accuracy="
                 << finalStepHtpValidation.accuracy << "\nseed_" << seed
                 << "_cpu_best_validation_loss=" << bestCpuValidation.loss
                 << "\nseed_" << seed << "_cpu_final_step_validation_loss="
                 << finalStepCpuValidation.loss;
      for (int patience : {2, 3, 4}) {
        const auto simulation = validation_selection::simulateEarlyStop(
            htpValidationTrajectory, patience, selected.steps);
        trajectory << "\nseed_" << seed << "_early_stop_patience_"
                   << patience << "_stop_step=" << simulation.stopStep
                   << "\nseed_" << seed << "_early_stop_patience_"
                   << patience << "_best_step=" << simulation.bestStep
                   << "\nseed_" << seed << "_early_stop_patience_"
                   << patience << "_saved_steps="
                   << simulation.savedTrainingSteps;
      }
    }
    if (bestValidationMode) trajectory << '\n';
    trajectory << "seed_" << seed << "_final_loss=" << final.loss
               << "\nseed_" << seed << "_final_accuracy=" << final.accuracy
               << "\nseed_" << seed << "_completed_steps="
               << seedCompletedSteps << "\nseed_" << seed
               << "_all_steps_finite="
               << (seedAllStepsFinite ? "true" : "false")
               << "\nseed_" << seed << "_final_evaluation_finite="
               << (finalEvaluationFinite ? "true" : "false")
               << "\nseed_" << seed << "_loss_reduction=" << reduction
               << "\nseed_" << seed << "_final_correct_probability="
               << final.meanCorrectProbability << "\nseed_" << seed
               << "_final_entropy=" << final.entropy << "\nseed_" << seed
               << "_final_mean_margin=" << final.meanMargin << "\nseed_" << seed
               << "_final_minimum_margin=" << final.minimumMargin
               << "\nseed_" << seed << "_cpu_final_loss=" << cpuFinal.loss
               << "\nseed_" << seed << "_cpu_final_accuracy="
               << cpuFinal.accuracy << "\nseed_" << seed
               << "_cpu_htp_parameter_max_abs_difference=" << parameterError
               << "\nseed_" << seed << "_final_gradient_l2_norm="
               << finalGradientNorm << "\nseed_" << seed
               << "_final_parameter_l2_norm=" << paramNorm(htp)
               << "\nseed_" << seed << "_final_parameter_canonical_hash="
               << canonicalFloatSha256(finalParameters)
               << "\nseed_" << seed
               << "_all_step_loss_canonical_hash="
               << canonicalFloatSha256(seedStepLosses)
               << "\nseed_" << seed
               << "_all_step_accuracy_canonical_hash="
               << canonicalFloatSha256(seedStepAccuracies)
               << "\nseed_" << seed
               << "_final_parameter_legacy_v1_canonical_hash="
               << canonicalFloatSha256(
                      flattenLanguageParametersLegacyV1(htp))
               << "\nseed_" << seed
               << "_nonfinite_count=" << seedNonfiniteCount << "\nseed_"
               << seed << "_qnn_execute_count="
               << (runtime.metrics().graphExecuteCount - seedExecuteStart)
               << "\nseed_" << seed << "_qnn_nonzero_return_count=0"
               << '\n';
    if (formalPostFix) {
      trajectory
          << "seed_" << seed << "_cpu_initial_loss=" << initialCpu.loss
          << "\nseed_" << seed << "_cpu_initial_accuracy="
          << initialCpu.accuracy << "\nseed_" << seed
          << "_cpu_loss_reduction="
          << 100 * (initialCpu.loss - cpuFinal.loss) / initialCpu.loss
          << "\nseed_" << seed << "_cpu_all_steps_finite="
          << (cpuAllStepsFinite ? "true" : "false")
          << "\nseed_" << seed << "_cpu_nonfinite_count="
          << cpuNonfiniteCount << "\nseed_" << seed
          << "_cpu_final_gradient_l2_norm=" << finalCpuGradientNorm
          << "\nseed_" << seed << "_cpu_final_parameter_l2_norm="
          << paramNorm(cpu) << "\nseed_" << seed
          << "_cpu_final_parameter_canonical_hash="
          << canonicalFloatSha256(finalCpuParameters) << '\n';
    }
    inferenceParameters.push_back(htp);
    cpuInferenceParameters.push_back(cpu);
  }
  std::sort(reductions.begin(), reductions.end());
  const double median = reductions[reductions.size() / 2];
  if (inferenceOnly) {
    // Report labels always name the actual native seed number so an
    // EXACT_SEED=k report exposes the same seed_k_* keys as a legacy
    // COUNT_FROM_ONE=k report.
    const size_t formalSeedLabelOffset = size_t(firstSeed - 1);
    bool formalContextSelfTest = true;
    if (formalPostFix) {
      std::vector<uint32_t> oracleProbe{1, 2};
      std::vector<uint32_t> freeProbe = oracleProbe;
      advanceLanguageContext(oracleProbe, 3, 4,
                             LanguageRolloutContext::ORACLE);
      advanceLanguageContext(freeProbe, 3, 4,
                             LanguageRolloutContext::FREE_RUNNING);
      formalContextSelfTest =
          oracleProbe == std::vector<uint32_t>({2, 3}) &&
          freeProbe == std::vector<uint32_t>({2, 4});
    }
    static const std::array<std::vector<uint32_t>, 4> rules{
        std::vector<uint32_t>{0, 1, 2, 3}, std::vector<uint32_t>{4, 5, 6, 7},
        std::vector<uint32_t>{8, 9}, std::vector<uint32_t>{10, 11, 12}};
    const auto parityBatch = languageBatch(config, 0, 0);
    std::vector<uint32_t> parityTokenIds(config.tokens);
    std::vector<uint32_t> parityTargetIds(config.tokens);
    for (uint32_t index = 0; index < config.tokens; ++index) {
      parityTokenIds[index] = index % 4;
      parityTargetIds[index] = (index + 1) % 4;
    }
    const auto generationInput =
        tiny_lm::oneHot(parityTokenIds, config.vocabularySize);
    const auto generationTarget =
        tiny_lm::oneHot(parityTargetIds, config.vocabularySize);
    bool outputNonfinite = false;
    int samePrefixNonfiniteCount = 0;
    int generationNonfiniteCount = 0;
    size_t outputNanCount = 0, outputInfCount = 0;
    std::string firstOutputNonfinite = "NONE";
    auto countOutputNonfinite = [&](const std::vector<float> &values) {
      for (float value : values) {
        outputNanCount += std::isnan(value);
        outputInfCount += std::isinf(value);
      }
    };
    const auto cpuEval = tiny_lm::forwardBackward(
        config, parityBatch.first, parityBatch.second,
        inferenceParameters.front(), 0.0f);
    const auto cpuGeneration = tiny_lm::forwardBackward(
        config, generationInput, generationTarget,
        inferenceParameters.front(), 0.0f);
    TinyTransformerTrainingOutputs htpEval, htpGeneration;
    if (!runtime.executeTinyTransformerTraining(
            parityBatch.first, parityBatch.second, inferenceParameters.front(),
            0.0f, htpEval, error) ||
        !runtime.executeTinyTransformerTraining(
            generationInput, generationTarget, inferenceParameters.front(),
            0.0f, htpGeneration, error))
      return failure("adam_same_prefix_parity", error, runtime);
    const bool representativeParityFinite =
        finite(cpuEval.logits) && finite(cpuEval.probabilities) &&
        finite(cpuGeneration.logits) &&
        finite(cpuGeneration.probabilities) && finite(htpEval.logits) &&
        finite(htpEval.probabilities) && finite(htpGeneration.logits) &&
        finite(htpGeneration.probabilities);
    if (!representativeParityFinite) {
      outputNonfinite = true;
      ++samePrefixNonfiniteCount;
      firstOutputNonfinite = "same_prefix_seed1";
      countOutputNonfinite(cpuEval.logits);
      countOutputNonfinite(cpuEval.probabilities);
      countOutputNonfinite(cpuGeneration.logits);
      countOutputNonfinite(cpuGeneration.probabilities);
      countOutputNonfinite(htpEval.logits);
      countOutputNonfinite(htpEval.probabilities);
      countOutputNonfinite(htpGeneration.logits);
      countOutputNonfinite(htpGeneration.probabilities);
    }
    struct Difference {
      double maximum = 0, mean = 0, relative = 0;
    };
    auto difference = [](const std::vector<float> &a,
                         const std::vector<float> &b) {
      Difference result;
      if (a.size() != b.size()) {
        result.maximum = result.mean = result.relative =
            std::numeric_limits<double>::infinity();
        return result;
      }
      for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
          result.maximum = result.mean = result.relative =
              std::numeric_limits<double>::infinity();
          return result;
        }
        const double error = std::abs(double(a[i]) - b[i]);
        result.maximum = std::max(result.maximum, error);
        result.mean += error;
        result.relative = std::max(
            result.relative,
            error / std::max(1.0e-12, std::abs(double(a[i]))));
      }
      result.mean /= std::max<size_t>(1, a.size());
      return result;
    };
    auto argmaxAtLastPosition = [&](const std::vector<float> &logits) {
      const size_t base =
          size_t(config.tokens - 1) * config.vocabularySize;
      uint32_t result = 0;
      for (uint32_t token = 1; token < config.vocabularySize; ++token)
        if (logits[base + token] > logits[base + result]) result = token;
      return result;
    };
    auto top3AtLastPosition = [&](const std::vector<float> &logits) {
      std::array<uint32_t, 3> result{0, 1, 2};
      const size_t base =
          size_t(config.tokens - 1) * config.vocabularySize;
      std::sort(result.begin(), result.end(), [&](uint32_t a, uint32_t b) {
        return logits[base + a] > logits[base + b];
      });
      for (uint32_t token = 3; token < config.vocabularySize; ++token)
        if (logits[base + token] > logits[base + result.back()]) {
          result.back() = token;
          std::sort(result.begin(), result.end(), [&](uint32_t a, uint32_t b) {
            return logits[base + a] > logits[base + b];
          });
        }
      return result;
    };
    struct PrefixDifference {
      double maximum = 0, mean = 0, relativeL2 = 0;
    };
    auto differenceAtLastPosition = [&](const std::vector<float> &cpu,
                                        const std::vector<float> &htp) {
      PrefixDifference result;
      const size_t base =
          size_t(config.tokens - 1) * config.vocabularySize;
      double squaredError = 0, squaredReference = 0;
      for (uint32_t token = 0; token < config.vocabularySize; ++token) {
        const double reference = cpu[base + token];
        const double candidateValue = htp[base + token];
        if (!std::isfinite(reference) || !std::isfinite(candidateValue)) {
          result.maximum = result.mean = result.relativeL2 =
              std::numeric_limits<double>::infinity();
          return result;
        }
        const double errorValue = std::abs(reference - candidateValue);
        result.maximum = std::max(result.maximum, errorValue);
        result.mean += errorValue;
        squaredError += errorValue * errorValue;
        squaredReference += reference * reference;
      }
      result.mean /= config.vocabularySize;
      result.relativeL2 =
          std::sqrt(squaredError) /
          std::max(1.0e-12, std::sqrt(squaredReference));
      return result;
    };
    auto formalTokenList = [](const std::vector<uint32_t> &values) {
      std::ostringstream text;
      for (size_t index = 0; index < values.size(); ++index) {
        if (index) text << ',';
        text << values[index];
      }
      return text.str();
    };
    auto lastPositionLogits = [&](const std::vector<float> &logits) {
      std::ostringstream text;
      text << std::setprecision(10);
      const size_t base =
          size_t(config.tokens - 1) * config.vocabularySize;
      for (uint32_t token = 0; token < config.vocabularySize; ++token) {
        if (token) text << ',';
        text << logits[base + token];
      }
      return text.str();
    };
    std::ostringstream formalDetails;
    formalDetails << std::setprecision(10);
    std::vector<double> formalGenerationLatencyUs;
    size_t formalPrefixComparisonCount = 0;
    size_t formalFreeCaseCount = 0, formalOracleCaseCount = 0;
    bool formalPrefixComparisonsFinite = true;
    std::string formalRepresentativeFinalLogitsHash;
    auto recordFormalPrefix =
        [&](const char *rolloutMode, size_t seedIndex, size_t pattern,
            int step, const std::vector<uint32_t> &context,
            uint32_t expected, const std::vector<float> &input,
            const std::vector<float> &target,
            const TinyTransformerTrainingOutputs &htpOutput) {
          if (!formalPostFix) return;
          ++formalPrefixComparisonCount;
          const auto cpuOutput = tiny_lm::forwardBackward(
              config, input, target, cpuInferenceParameters[seedIndex], 0.0f);
          formalPrefixComparisonsFinite =
              formalPrefixComparisonsFinite && finite(cpuOutput.logits) &&
              finite(cpuOutput.probabilities) && finite(htpOutput.logits) &&
              finite(htpOutput.probabilities);
          const auto difference =
              differenceAtLastPosition(cpuOutput.logits, htpOutput.logits);
          const auto cpuTop3 = top3AtLastPosition(cpuOutput.logits);
          const auto htpTop3 = top3AtLastPosition(htpOutput.logits);
          const uint32_t cpuArgmax =
              argmaxAtLastPosition(cpuOutput.logits);
          const uint32_t htpArgmax =
              argmaxAtLastPosition(htpOutput.logits);
          const size_t base =
              size_t(config.tokens - 1) * config.vocabularySize;
          const std::string prefix =
              std::string("formal_logits_") + rolloutMode + "_s" +
              std::to_string(seedIndex + 1 + formalSeedLabelOffset) + "_p" +
              std::to_string(pattern) + "_step_" + std::to_string(step);
          if (std::string(rolloutMode) == "free" && seedIndex == 0 &&
              pattern == 0 && step == 7)
            formalRepresentativeFinalLogitsHash =
                canonicalFloatSha256(htpOutput.logits);
          formalDetails
              << prefix << "_context=" << formalTokenList(context) << '\n'
              << prefix << "_expected_token=" << expected << '\n'
              << prefix << "_cpu_logits_private="
              << lastPositionLogits(cpuOutput.logits) << '\n'
              << prefix << "_htp_logits_private="
              << lastPositionLogits(htpOutput.logits) << '\n'
              << prefix << "_max_abs_difference=" << difference.maximum
              << '\n' << prefix << "_mean_abs_difference="
              << difference.mean << '\n' << prefix
              << "_relative_l2_difference=" << difference.relativeL2
              << '\n' << prefix << "_cpu_argmax=" << cpuArgmax << '\n'
              << prefix << "_htp_argmax=" << htpArgmax << '\n'
              << prefix << "_argmax_match="
              << (cpuArgmax == htpArgmax ? "true" : "false") << '\n'
              << prefix << "_cpu_top3=" << cpuTop3[0] << ',' << cpuTop3[1]
              << ',' << cpuTop3[2] << '\n' << prefix << "_htp_top3="
              << htpTop3[0] << ',' << htpTop3[1] << ',' << htpTop3[2]
              << '\n' << prefix << "_top3_match="
              << (cpuTop3 == htpTop3 ? "true" : "false") << '\n'
              << prefix << "_cpu_expected_probability="
              << cpuOutput.probabilities[base + expected] << '\n'
              << prefix << "_htp_expected_probability="
              << htpOutput.probabilities[base + expected] << '\n';
        };
    const Difference cpuEvalGeneration =
        difference(cpuEval.logits, cpuGeneration.logits);
    const Difference htpEvalGeneration =
        difference(htpEval.logits, htpGeneration.logits);
    const Difference cpuHtpEval = difference(cpuEval.logits, htpEval.logits);
    const Difference cpuHtpGeneration =
        difference(cpuGeneration.logits, htpGeneration.logits);
    const auto cpuTop3 = representativeParityFinite
                             ? top3AtLastPosition(cpuEval.logits)
                             : std::array<uint32_t, 3>{0, 0, 0};
    const auto htpTop3 = representativeParityFinite
                             ? top3AtLastPosition(htpEval.logits)
                             : std::array<uint32_t, 3>{0, 0, 0};
    bool allCpuEvalGenerationArgmax = representativeParityFinite;
    bool allHtpEvalGenerationArgmax = representativeParityFinite;
    bool allCpuEvalGenerationLogits = representativeParityFinite;
    bool allHtpEvalGenerationLogits = representativeParityFinite;
    bool allCpuHtpArgmax = representativeParityFinite;
    bool allCpuHtpTop3 = representativeParityFinite;
    double worstCpuHtpLogits = 0;
    std::ostringstream paritySeeds;
    for (size_t seedIndex = 0; seedIndex < inferenceParameters.size();
         ++seedIndex) {
      const auto seedCpuEval = tiny_lm::forwardBackward(
          config, parityBatch.first, parityBatch.second,
          inferenceParameters[seedIndex], 0.0f);
      const auto seedCpuGeneration = tiny_lm::forwardBackward(
          config, generationInput, generationTarget,
          inferenceParameters[seedIndex], 0.0f);
      TinyTransformerTrainingOutputs seedHtpEval, seedHtpGeneration;
      if (!runtime.executeTinyTransformerTraining(
              parityBatch.first, parityBatch.second,
              inferenceParameters[seedIndex], 0.0f, seedHtpEval, error) ||
          !runtime.executeTinyTransformerTraining(
              generationInput, generationTarget,
              inferenceParameters[seedIndex], 0.0f, seedHtpGeneration, error))
        return failure("adam_same_prefix_parity_seed", error, runtime);
      const bool seedParityFinite =
          finite(seedCpuEval.logits) && finite(seedCpuEval.probabilities) &&
          finite(seedCpuGeneration.logits) &&
          finite(seedCpuGeneration.probabilities) &&
          finite(seedHtpEval.logits) && finite(seedHtpEval.probabilities) &&
          finite(seedHtpGeneration.logits) &&
          finite(seedHtpGeneration.probabilities);
      if (!seedParityFinite) {
        outputNonfinite = true;
        ++samePrefixNonfiniteCount;
        if (firstOutputNonfinite == "NONE")
          firstOutputNonfinite =
              "same_prefix_seed" + std::to_string(seedIndex + 1 + formalSeedLabelOffset);
        countOutputNonfinite(seedCpuEval.logits);
        countOutputNonfinite(seedCpuEval.probabilities);
        countOutputNonfinite(seedCpuGeneration.logits);
        countOutputNonfinite(seedCpuGeneration.probabilities);
        countOutputNonfinite(seedHtpEval.logits);
        countOutputNonfinite(seedHtpEval.probabilities);
        countOutputNonfinite(seedHtpGeneration.logits);
        countOutputNonfinite(seedHtpGeneration.probabilities);
        allCpuEvalGenerationArgmax = false;
        allHtpEvalGenerationArgmax = false;
        allCpuEvalGenerationLogits = false;
        allHtpEvalGenerationLogits = false;
        allCpuHtpArgmax = false;
        allCpuHtpTop3 = false;
        worstCpuHtpLogits = std::numeric_limits<double>::infinity();
        paritySeeds << "same_prefix_seed_" << (seedIndex + 1 + formalSeedLabelOffset)
                    << "_finite=false\n";
        continue;
      }
      const auto seedCpuEvalGeneration =
          difference(seedCpuEval.logits, seedCpuGeneration.logits);
      const auto seedHtpEvalGeneration =
          difference(seedHtpEval.logits, seedHtpGeneration.logits);
      const auto seedCpuHtp =
          difference(seedCpuEval.logits, seedHtpEval.logits);
      const auto seedCpuArgmax = argmaxAtLastPosition(seedCpuEval.logits);
      const auto seedCpuGenerationArgmax =
          argmaxAtLastPosition(seedCpuGeneration.logits);
      const auto seedHtpArgmax = argmaxAtLastPosition(seedHtpEval.logits);
      const auto seedHtpGenerationArgmax =
          argmaxAtLastPosition(seedHtpGeneration.logits);
      const auto seedCpuTop3 = top3AtLastPosition(seedCpuEval.logits);
      const auto seedHtpTop3 = top3AtLastPosition(seedHtpEval.logits);
      allCpuEvalGenerationArgmax =
          allCpuEvalGenerationArgmax &&
          seedCpuArgmax == seedCpuGenerationArgmax;
      allHtpEvalGenerationArgmax =
          allHtpEvalGenerationArgmax &&
          seedHtpArgmax == seedHtpGenerationArgmax;
      allCpuEvalGenerationLogits =
          allCpuEvalGenerationLogits &&
          seedCpuEvalGeneration.maximum == 0;
      allHtpEvalGenerationLogits =
          allHtpEvalGenerationLogits &&
          seedHtpEvalGeneration.maximum == 0;
      allCpuHtpArgmax =
          allCpuHtpArgmax && seedCpuArgmax == seedHtpArgmax;
      allCpuHtpTop3 = allCpuHtpTop3 && seedCpuTop3 == seedHtpTop3;
      worstCpuHtpLogits =
          std::max(worstCpuHtpLogits, seedCpuHtp.maximum);
      paritySeeds << "same_prefix_seed_" << (seedIndex + 1 + formalSeedLabelOffset)
                  << "_finite=true\nsame_prefix_seed_" << (seedIndex + 1 + formalSeedLabelOffset)
                  << "_cpu_eval_generation_max_abs_error="
                  << seedCpuEvalGeneration.maximum << "\nsame_prefix_seed_"
                  << (seedIndex + 1 + formalSeedLabelOffset)
                  << "_htp_eval_generation_max_abs_error="
                  << seedHtpEvalGeneration.maximum << "\nsame_prefix_seed_"
                  << (seedIndex + 1 + formalSeedLabelOffset) << "_cpu_htp_max_abs_error="
                  << seedCpuHtp.maximum << '\n';
    }
    std::ostringstream samePrefixPositions;
    for (uint32_t index = 0; index < config.tokens; ++index) {
      if (index) samePrefixPositions << ',';
      samePrefixPositions << index;
    }
    trajectory
        << "same_prefix_token_ids=" << formalTokenList(parityTokenIds)
        << "\nsame_prefix_host_inputs_identical="
        << ((parityBatch.first == generationInput &&
             parityBatch.second == generationTarget)
                ? "true"
                : "false")
        << "\nsame_prefix_valid_token_count=" << config.tokens
        << "\nsame_prefix_position_indices=" << samePrefixPositions.str()
        << "\nsame_prefix_logit_read_position=" << (config.tokens - 1)
        << "\nsame_prefix_representative_finite="
        << (representativeParityFinite ? "true" : "false")
        << "\nsame_prefix_cpu_eval_generation_max_abs_error="
        << cpuEvalGeneration.maximum
        << "\nsame_prefix_cpu_eval_generation_mean_abs_error="
        << cpuEvalGeneration.mean
        << "\nsame_prefix_cpu_eval_generation_max_relative_error="
        << cpuEvalGeneration.relative
        << "\nsame_prefix_htp_eval_generation_max_abs_error="
        << htpEvalGeneration.maximum
        << "\nsame_prefix_htp_eval_generation_mean_abs_error="
        << htpEvalGeneration.mean
        << "\nsame_prefix_htp_eval_generation_max_relative_error="
        << htpEvalGeneration.relative
        << "\nsame_prefix_cpu_htp_eval_max_abs_error=" << cpuHtpEval.maximum
        << "\nsame_prefix_cpu_htp_eval_mean_abs_error=" << cpuHtpEval.mean
        << "\nsame_prefix_cpu_htp_eval_max_relative_error="
        << cpuHtpEval.relative
        << "\nsame_prefix_cpu_htp_generation_max_abs_error="
        << cpuHtpGeneration.maximum
        << "\nsame_prefix_cpu_htp_generation_mean_abs_error="
        << cpuHtpGeneration.mean
        << "\nsame_prefix_cpu_htp_generation_max_relative_error="
        << cpuHtpGeneration.relative
        << "\nsame_prefix_cpu_argmax="
        << (representativeParityFinite ? argmaxAtLastPosition(cpuEval.logits)
                                       : 0)
        << "\nsame_prefix_htp_argmax="
        << (representativeParityFinite ? argmaxAtLastPosition(htpEval.logits)
                                       : 0)
        << "\nsame_prefix_cpu_top3=" << cpuTop3[0] << ',' << cpuTop3[1]
        << ',' << cpuTop3[2] << "\nsame_prefix_htp_top3=" << htpTop3[0]
        << ',' << htpTop3[1] << ',' << htpTop3[2]
        << "\nsame_prefix_seed_count=" << executedSeedCount
        << "\nsame_prefix_all_cpu_eval_generation_argmax_match="
        << (allCpuEvalGenerationArgmax ? "true" : "false")
        << "\nsame_prefix_all_htp_eval_generation_argmax_match="
        << (allHtpEvalGenerationArgmax ? "true" : "false")
        << "\nsame_prefix_all_cpu_eval_generation_logits_match="
        << (allCpuEvalGenerationLogits ? "true" : "false")
        << "\nsame_prefix_all_htp_eval_generation_logits_match="
        << (allHtpEvalGenerationLogits ? "true" : "false")
        << "\nsame_prefix_all_cpu_htp_argmax_match="
        << (allCpuHtpArgmax ? "true" : "false")
        << "\nsame_prefix_all_cpu_htp_top3_match="
        << (allCpuHtpTop3 ? "true" : "false")
        << "\nsame_prefix_all_seed_cpu_htp_logits_max_abs_error="
        << worstCpuHtpLogits << '\n' << paritySeeds.str();
    int exactPatterns = 0;
    int exactRollouts = 0, qualifyingSeeds = 0;
    int oracleExactRollouts = 0, oracleQualifyingSeeds = 0;
    auto tokenList = [](const std::vector<uint32_t> &values) {
      std::ostringstream text;
      for (size_t i = 0; i < values.size(); ++i) {
        if (i) text << ',';
        text << values[i];
      }
      return text.str();
    };
    for (size_t seedIndex = 0; seedIndex < inferenceParameters.size();
         ++seedIndex) {
      int seedExactPatterns = 0, seedOracleExactPatterns = 0;
      for (size_t pattern = 0; pattern < rules.size(); ++pattern) {
        const auto &rule = rules[pattern];
        std::vector<uint32_t> context(config.tokens);
        for (uint32_t i = 0; i < config.tokens; ++i)
          context[i] = rule[i % rule.size()];
        const auto prompt = context;
        std::vector<uint32_t> generated, expectedTokens;
        int correct = 0, firstError = -1, evaluatedSteps = 0;
        uint32_t firstErrorExpected = 0, firstErrorPredicted = 0;
        std::array<uint32_t, 3> firstErrorTop3{0, 0, 0};
        double firstErrorExpectedProbability = 0;
        double probability = 0, margin = 0;
        double minimumMargin = std::numeric_limits<double>::infinity();
        for (int step = 0; step < 8; ++step) {
          const uint32_t expected =
              rule[(size_t(config.tokens) + size_t(step)) % rule.size()];
          auto input = tiny_lm::oneHot(context, config.vocabularySize);
          auto target = tiny_lm::oneHot(
              std::vector<uint32_t>(config.tokens, expected),
              config.vocabularySize);
          TinyTransformerTrainingOutputs output;
          const auto generationStarted = std::chrono::steady_clock::now();
          const bool generationExecuted =
              runtime.executeTinyTransformerTraining(
                  input, target, inferenceParameters[seedIndex], 0.0f, output,
                  error);
          if (formalPostFix)
            formalGenerationLatencyUs.push_back(
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - generationStarted)
                    .count());
          if (!generationExecuted)
            return failure("adam_generation", error, runtime);
          if (!finite(output.logits) || !finite(output.probabilities)) {
            std::ostringstream location;
            location << "seed" << (seedIndex + 1 + formalSeedLabelOffset) << "_pattern" << pattern
                     << "_step" << step << "_logits_finite"
                     << finite(output.logits) << "_probabilities_finite"
                     << finite(output.probabilities);
            outputNonfinite = true;
            ++generationNonfiniteCount;
            countOutputNonfinite(output.logits);
            countOutputNonfinite(output.probabilities);
            if (firstOutputNonfinite == "NONE")
              firstOutputNonfinite = location.str();
            if (firstError < 0) firstError = step;
            break;
          }
          ++evaluatedSteps;
          const size_t base =
              size_t(config.tokens - 1) * config.vocabularySize;
          uint32_t predicted = 0;
          float other = -std::numeric_limits<float>::infinity();
          for (uint32_t token = 1; token < config.vocabularySize; ++token)
            if (output.logits[base + token] > output.logits[base + predicted])
              predicted = token;
          for (uint32_t token = 0; token < config.vocabularySize; ++token)
            if (token != expected)
              other = std::max(other, output.logits[base + token]);
          recordFormalPrefix("free", seedIndex, pattern, step, context,
                             expected, input, target, output);
          if (predicted != expected && firstError < 0) {
            firstError = step;
            firstErrorExpected = expected;
            firstErrorPredicted = predicted;
            firstErrorTop3 = top3AtLastPosition(output.logits);
            firstErrorExpectedProbability =
                output.probabilities[base + expected];
          }
          correct += predicted == expected;
          probability += output.probabilities[base + expected];
          const double tokenMargin = output.logits[base + expected] - other;
          margin += tokenMargin;
          minimumMargin = std::min(minimumMargin, tokenMargin);
          if (seedIndex == 0) {
            double tokenEntropy = 0;
            for (uint32_t token = 0; token < config.vocabularySize; ++token) {
              const double p =
                  std::max(double(output.probabilities[base + token]), 1e-30);
              tokenEntropy -= p * std::log(p);
            }
            const auto top3 = top3AtLastPosition(output.logits);
            trajectory
                << "generation_detail_pattern_" << pattern << "_step_" << step
                << "_context=" << tokenList(context)
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_expected=" << expected
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_predicted=" << predicted
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_correct_probability="
                << output.probabilities[base + expected]
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_predicted_probability="
                << output.probabilities[base + predicted]
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_margin=" << tokenMargin
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_entropy=" << tokenEntropy
                << "\ngeneration_detail_pattern_" << pattern << "_step_"
                << step << "_top3=" << top3[0] << ',' << top3[1] << ','
                << top3[2] << '\n';
          }
          generated.push_back(predicted);
          expectedTokens.push_back(expected);
          advanceLanguageContext(context, expected, predicted,
                                 LanguageRolloutContext::FREE_RUNNING);
        }
        const bool exact = evaluatedSteps == 8 && correct == 8;
        if (formalPostFix && evaluatedSteps == 8) ++formalFreeCaseCount;
        exactPatterns += seedIndex == 0 && exact;
        exactRollouts += exact;
        seedExactPatterns += exact;
        std::vector<uint32_t> oracleContext(config.tokens);
        for (uint32_t i = 0; i < config.tokens; ++i)
          oracleContext[i] = rule[i % rule.size()];
        const auto oraclePrompt = oracleContext;
        std::vector<uint32_t> oracleGenerated, oracleExpectedTokens;
        int oracleCorrect = 0, firstOracleError = -1;
        int oracleEvaluatedSteps = 0;
        uint32_t firstOracleErrorExpected = 0;
        uint32_t firstOracleErrorPredicted = 0;
        std::array<uint32_t, 3> firstOracleErrorTop3{0, 0, 0};
        double firstOracleErrorExpectedProbability = 0;
        for (int step = 0; step < 8; ++step) {
          const uint32_t expected =
              rule[(size_t(config.tokens) + size_t(step)) % rule.size()];
          const auto input =
              tiny_lm::oneHot(oracleContext, config.vocabularySize);
          const auto target = tiny_lm::oneHot(
              std::vector<uint32_t>(config.tokens, expected),
              config.vocabularySize);
          TinyTransformerTrainingOutputs output;
          const auto generationStarted = std::chrono::steady_clock::now();
          const bool generationExecuted =
              runtime.executeTinyTransformerTraining(
                  input, target, inferenceParameters[seedIndex], 0.0f, output,
                  error);
          if (formalPostFix)
            formalGenerationLatencyUs.push_back(
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - generationStarted)
                    .count());
          if (!generationExecuted)
            return failure("adam_oracle_generation", error, runtime);
          if (!finite(output.logits) || !finite(output.probabilities)) {
            std::ostringstream location;
            location << "oracle_seed" << (seedIndex + 1 + formalSeedLabelOffset) << "_pattern"
                     << pattern << "_step" << step << "_logits_finite"
                     << finite(output.logits) << "_probabilities_finite"
                     << finite(output.probabilities);
            outputNonfinite = true;
            ++generationNonfiniteCount;
            countOutputNonfinite(output.logits);
            countOutputNonfinite(output.probabilities);
            if (firstOutputNonfinite == "NONE")
              firstOutputNonfinite = location.str();
            if (firstOracleError < 0) firstOracleError = step;
            break;
          }
          ++oracleEvaluatedSteps;
          const uint32_t predicted = argmaxAtLastPosition(output.logits);
          recordFormalPrefix("oracle", seedIndex, pattern, step,
                             oracleContext, expected, input, target, output);
          if (predicted != expected && firstOracleError < 0) {
            firstOracleError = step;
            firstOracleErrorExpected = expected;
            firstOracleErrorPredicted = predicted;
            firstOracleErrorTop3 = top3AtLastPosition(output.logits);
            const size_t base =
                size_t(config.tokens - 1) * config.vocabularySize;
            firstOracleErrorExpectedProbability =
                output.probabilities[base + expected];
          }
          oracleCorrect += predicted == expected;
          oracleGenerated.push_back(predicted);
          oracleExpectedTokens.push_back(expected);
          advanceLanguageContext(oracleContext, expected, predicted,
                                 LanguageRolloutContext::ORACLE);
        }
        const bool oracleExact =
            oracleEvaluatedSteps == 8 && oracleCorrect == 8;
        if (formalPostFix && oracleEvaluatedSteps == 8)
          ++formalOracleCaseCount;
        oracleExactRollouts += oracleExact;
        seedOracleExactPatterns += oracleExact;
        auto appendFormalCase =
            [&](const char *rolloutMode,
                const std::vector<uint32_t> &casePrompt,
                const std::vector<uint32_t> &caseExpected,
                const std::vector<uint32_t> &caseGenerated, bool caseExact,
                int firstMismatch, uint32_t mismatchExpected,
                uint32_t mismatchPredicted,
                const std::array<uint32_t, 3> &mismatchTop3,
                double mismatchExpectedProbability) {
              if (!formalPostFix) return;
              const std::string prefix =
                  std::string("formal_") + rolloutMode + "_case_s" +
                  std::to_string(seedIndex + 1 + formalSeedLabelOffset) + "_p" +
                  std::to_string(pattern);
              formalDetails
                  << prefix << "_id=s" << (seedIndex + 1 + formalSeedLabelOffset) << "_p" << pattern
                  << '\n' << prefix << "_prefix="
                  << formalTokenList(casePrompt) << '\n' << prefix
                  << "_expected_sequence=" << formalTokenList(caseExpected)
                  << '\n' << prefix << "_generated_sequence="
                  << formalTokenList(caseGenerated) << '\n' << prefix
                  << "_exact=" << (caseExact ? "true" : "false") << '\n'
                  << prefix << "_first_mismatch_step=" << firstMismatch
                  << '\n';
              if (firstMismatch < 0) {
                formalDetails
                    << prefix << "_first_mismatch_expected_token=NONE\n"
                    << prefix << "_first_mismatch_predicted_token=NONE\n"
                    << prefix << "_first_mismatch_top3=NONE\n"
                    << prefix
                    << "_first_mismatch_expected_probability=NONE\n";
              } else {
                formalDetails
                    << prefix << "_first_mismatch_expected_token="
                    << mismatchExpected << '\n' << prefix
                    << "_first_mismatch_predicted_token="
                    << mismatchPredicted << '\n' << prefix
                    << "_first_mismatch_top3=" << mismatchTop3[0] << ','
                    << mismatchTop3[1] << ',' << mismatchTop3[2] << '\n'
                    << prefix << "_first_mismatch_expected_probability="
                    << mismatchExpectedProbability << '\n';
              }
            };
        appendFormalCase(
            "free", prompt, expectedTokens, generated, exact, firstError,
            firstErrorExpected, firstErrorPredicted, firstErrorTop3,
            firstErrorExpectedProbability);
        appendFormalCase(
            "oracle", oraclePrompt, oracleExpectedTokens, oracleGenerated,
            oracleExact, firstOracleError, firstOracleErrorExpected,
            firstOracleErrorPredicted, firstOracleErrorTop3,
            firstOracleErrorExpectedProbability);
        trajectory << "seed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_prompt=" << tokenList(prompt)
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_expected=" << tokenList(expectedTokens)
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_generated=" << tokenList(generated)
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_exact=" << (exact ? "true" : "false")
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_token_accuracy=" << correct / 8.0
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset)
                   << "_generation_pattern_" << pattern
                   << "_evaluated_steps=" << evaluatedSteps
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_first_error=" << firstError
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_generation_pattern_"
                   << pattern << "_mean_correct_probability="
                   << (evaluatedSteps ? probability / evaluatedSteps : 0)
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset)
                   << "_generation_pattern_" << pattern << "_mean_margin="
                   << (evaluatedSteps ? margin / evaluatedSteps : 0)
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset)
                   << "_generation_pattern_" << pattern << "_minimum_margin=";
        if (evaluatedSteps)
          trajectory << minimumMargin;
        else
          trajectory << "NOT_AVAILABLE";
        trajectory << '\n';
        trajectory << "seed_" << (seedIndex + 1 + formalSeedLabelOffset)
                   << "_oracle_pattern_" << pattern
                   << "_exact=" << (oracleExact ? "true" : "false")
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_oracle_pattern_"
                   << pattern << "_token_accuracy=" << oracleCorrect / 8.0
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_oracle_pattern_"
                   << pattern << "_evaluated_steps=" << oracleEvaluatedSteps
                   << "\nseed_" << (seedIndex + 1 + formalSeedLabelOffset) << "_oracle_pattern_"
                   << pattern << "_first_error=" << firstOracleError << '\n';
      }
      qualifyingSeeds += seedExactPatterns >= 3;
      oracleQualifyingSeeds += seedOracleExactPatterns == 4;
    }
    const bool exactSuccess = !outputNonfinite &&
                              qualifyingSeeds >= std::min(4, executedSeedCount) &&
                              exactRollouts >= std::min(16, executedSeedCount * 4) &&
                              oracleQualifyingSeeds >= std::min(4, executedSeedCount);
    constexpr size_t kGenerationWarmupCalls = 16;
    constexpr size_t kTrainingWarmupSteps = 16;
    std::vector<double> measuredGenerationLatencyUs;
    std::vector<double> measuredTrainingStepLatencyUs;
    if (formalGenerationLatencyUs.size() > kGenerationWarmupCalls) {
      measuredGenerationLatencyUs.assign(
          formalGenerationLatencyUs.begin() + kGenerationWarmupCalls,
          formalGenerationLatencyUs.end());
      std::sort(measuredGenerationLatencyUs.begin(),
                measuredGenerationLatencyUs.end());
    }
    if (formalTrainingStepLatencyUs.size() > kTrainingWarmupSteps) {
      measuredTrainingStepLatencyUs.assign(
          formalTrainingStepLatencyUs.begin() + kTrainingWarmupSteps,
          formalTrainingStepLatencyUs.end());
      std::sort(measuredTrainingStepLatencyUs.begin(),
                measuredTrainingStepLatencyUs.end());
    }
    auto latencyPercentileUs = [](const std::vector<double> &values,
                                  double quantile) {
      if (values.empty()) return 0.0;
      const size_t index = std::min(
          values.size() - 1,
          size_t(quantile * double(values.size() - 1)));
      return values[index];
    };
    const size_t expectedGenerationCases = size_t(executedSeedCount) * 4;
    const size_t expectedPrefixComparisons =
        expectedGenerationCases * 2 * 8;
    const size_t expectedTrainingSteps =
        size_t(executedSeedCount) * size_t(selected.steps);
    const bool scalingGenerationComplete =
        numericalProbe
            ? size_t(formalOracleCaseCount) == expectedGenerationCases &&
                  size_t(formalFreeCaseCount) == expectedGenerationCases
            : scalingSmoke
            ? size_t(oracleExactRollouts) == expectedGenerationCases &&
                  exactRollouts >= 1
            : size_t(oracleExactRollouts) == expectedGenerationCases &&
                  size_t(exactRollouts) == expectedGenerationCases;
    const bool formalComplete =
        !nan && !outputNonfinite &&
        (numericalProbe || allLoss || bestValidationMode) &&
        allFormalCpuFinite && formalPrefixComparisonsFinite &&
        formalContextSelfTest &&
        inferenceParameters.size() == size_t(executedSeedCount) &&
        cpuInferenceParameters.size() == size_t(executedSeedCount) &&
        formalFreeCaseCount == expectedGenerationCases &&
        formalOracleCaseCount == expectedGenerationCases &&
        formalPrefixComparisonCount == expectedPrefixComparisons &&
        formalGenerationLatencyUs.size() == expectedPrefixComparisons &&
        formalTrainingStepLatencyUs.size() == expectedTrainingSteps &&
        scalingGenerationComplete &&
        !formalRepresentativeFinalLogitsHash.empty() &&
        runtime.apiTrace().graphExecuteFailureCount == 0;
    std::ostringstream formalSummary;
    if (formalPostFix) {
      formalSummary
          << "\nsource_protocol=POST_FIX_ADAM_LR0.003_STEPS320_V1"
          << "\nscaling_evaluation="
          << (numericalProbe ? "NUMERICAL_PROBE"
                             : (lastSeed == 5 && !scalingSmoke ? "FORMAL"
                                                               : "SMOKE"))
          << "\nsequence_length=" << config.tokens
          << "\nembedding_dimension=" << config.dimension
          << "\ntransformer_layers=" << layers
          << "\nattention_heads=" << attentionHeads
          << "\nfeed_forward_dimension=" << config.feedForwardDimension
          << "\nshape_contract_valid=true"
          << "\napp_write_hashes_unchanged="
          << (scalingAppWriteUnchanged ? "true" : "false")
          << "\napp_read_poison_residual_elements="
          << scalingPoisonResidualElements
          << "\nbinding_audit_all_outputs_finite="
          << (scalingAuditFinite ? "true" : "false")
          << "\ntraining_phase=0"
          << "\nglobal_gradient_clipping="
          << (selected.gradientClipping ? "enabled" : "disabled")
          << "\nlearning_rate_schedule="
          << (stabilityMode == 1 ? "warmup64_linear"
                                 : (stabilityMode == 2 ? "decay_linear_to_zero"
                                                       : "constant"))
          << "\ntraining_stability_mode="
          << phonelm::trainingStabilityModeName(stabilityMode)
          << "\ndepth_pair_init_mode="
          << phonelm::depthPairInitModeName(pairInitMode);
      if (bestValidationMode) {
        formalSummary
            << "\ncheckpoint_selection_mode=BEST_VALIDATION_V1"
            << "\nvalidation_schema_version="
            << validation_selection::kValidationSchemaVersion
            << "\nvalidation_generator_domain=ROTATED_LAST_POSITION_V2"
            << "\nvalidation_set_hash="
            << validation_selection::validationSetHash(config.tokens)
            << "\nvalidation_case_count="
            << validation_selection::validationCases(config.tokens).size()
            << "\nvalidation_loss_tie_tolerance="
            << validation_selection::kLossTieTolerance
            << "\nbest_checkpoint_parameter_bytes="
            << resources.bestCheckpointParameterBytes
            << "\nbest_checkpoint_adam_bytes="
            << resources.bestCheckpointAdamBytes
            << "\ncpu_reference_checkpoint_bytes="
            << resources.cpuReferenceCheckpointBytes
            << "\ncheckpoint_selection_overhead_bytes="
            << resources.checkpointSelectionOverheadBytes
            << "\nestimated_peak_with_selection_bytes="
            << resources.estimatedPeakWithBestCheckpointBytes
            << "\nbest_checkpoint_fits_application_policy="
            << (resources.bestCheckpointFitsApplicationPolicy ? "true" : "false");
      }
      formalSummary << "\ndiagnostic_trajectory="
          << (diagnosticTrajectory ? "true" : "false")
          << "\ncheckpoint_dump_steps="
          << (checkpointDumpSteps.str().empty() ? "NONE"
                                                : checkpointDumpSteps.str())
          << "\ncheckpoint_dump_errors=" << checkpointDumpErrors
          << "\nformal_oracle_case_count=" << formalOracleCaseCount
          << "\nformal_free_case_count=" << formalFreeCaseCount
          << "\nformal_prefix_logits_comparison_count="
          << formalPrefixComparisonCount
          << "\nformal_qnn_nonzero_return_count="
          << runtime.apiTrace().graphExecuteFailureCount
          << "\nformal_cpu_all_finite="
          << (allFormalCpuFinite ? "true" : "false")
          << "\nformal_prefix_comparisons_finite="
          << (formalPrefixComparisonsFinite ? "true" : "false")
          << "\nformal_representative_final_logits_canonical_hash="
          << formalRepresentativeFinalLogitsHash
          << "\nfree_running_context_update=PREVIOUS_PREDICTION"
          << "\noracle_context_update=EXPECTED_TOKEN"
          << "\nfree_running_teacher_forcing=false"
          << "\ngeneration_context_self_test="
          << (formalContextSelfTest ? "true" : "false")
          << "\nlogits_comparison_position=LAST_POSITION_V32"
          << "\nlogits_comparison_parameter_scope="
             "CPU_TRAINED_CPU_VS_HTP_TRAINED_HTP"
          << "\nraw_logits_visibility=PRIVATE_DEVICE_REPORT_ONLY"
          << "\ngeneration_latency_path=HTP_FULL_TRAINING_GRAPH_ZERO_LR"
          << "\ngeneration_latency_warmup_calls="
          << kGenerationWarmupCalls
          << "\ngeneration_latency_measured_calls="
          << measuredGenerationLatencyUs.size()
          << "\ngeneration_token_latency_min_ms="
          << latencyPercentileUs(measuredGenerationLatencyUs, 0.0) / 1000.0
          << "\ngeneration_token_latency_median_ms="
          << latencyPercentileUs(measuredGenerationLatencyUs, 0.5) / 1000.0
          << "\ngeneration_token_latency_p95_ms="
          << latencyPercentileUs(measuredGenerationLatencyUs, 0.95) / 1000.0
          << "\ngeneration_token_latency_max_ms="
          << latencyPercentileUs(measuredGenerationLatencyUs, 1.0) / 1000.0
          << "\nperformance_initialization_ms="
          << (runtime.metrics().backendCreateUs +
              runtime.metrics().deviceCreateUs +
              runtime.metrics().contextCreateUs) /
                 1000.0
          << "\nperformance_graph_creation_ms="
          << runtime.metrics().graphCreateUs / 1000.0
          << "\nperformance_finalize_ms="
          << runtime.metrics().graphFinalizeUs / 1000.0
          << "\nperformance_training_warmup_steps="
          << kTrainingWarmupSteps
          << "\nperformance_training_measured_steps="
          << measuredTrainingStepLatencyUs.size()
          << "\nperformance_steady_training_step_min_ms="
          << latencyPercentileUs(measuredTrainingStepLatencyUs, 0.0) / 1000.0
          << "\nperformance_steady_training_step_median_ms="
          << latencyPercentileUs(measuredTrainingStepLatencyUs, 0.5) / 1000.0
          << "\nperformance_steady_training_step_p95_ms="
          << latencyPercentileUs(measuredTrainingStepLatencyUs, 0.95) / 1000.0
          << "\nperformance_steady_training_step_max_ms="
          << latencyPercentileUs(measuredTrainingStepLatencyUs, 1.0) / 1000.0
          << "\nperformance_updates_per_second="
          << (latencyPercentileUs(measuredTrainingStepLatencyUs, 0.5) > 0
                  ? 1000000.0 /
                        latencyPercentileUs(measuredTrainingStepLatencyUs, 0.5)
                  : 0.0)
          << "\nperformance_tokens_per_second="
          << (latencyPercentileUs(measuredTrainingStepLatencyUs, 0.5) > 0
                  ? 1000000.0 * config.tokens /
                        latencyPercentileUs(measuredTrainingStepLatencyUs, 0.5)
                  : 0.0);
      struct rusage usage {};
      const bool peakRssAvailable = getrusage(RUSAGE_SELF, &usage) == 0;
      formalSummary
          << "\nprocess_peak_rss_available="
          << (peakRssAvailable ? "true" : "false")
          << "\nprocess_peak_rss_kib="
          << (peakRssAvailable ? usage.ru_maxrss : 0)
          << "\noptimizer_parameter_elements=" << optimizerElements
          << "\noptimizer_graph_elements=" << optimizerGraphElements
          << "\noptimizer_chunk_count=" << optimizerChunkCount
          << "\noptimizer_chunking="
          << (optimizerChunkCount == 1 ? "SINGLE_VECTOR"
                                       : "QNN_FIXED_CHUNKS");
    }
    std::ostringstream report;
    report << "TINY_LANGUAGE_MODEL\ntest="
           << (formalPostFix
                   ? (config.tokens == 8 && config.dimension == 16
                          ? "post_fix_end_to_end_generation"
                          : "scaled_tiny_lm_end_to_end_generation")
                             : "adam_inference_4_pattern")
           << "\nstatus="
           << (formalPostFix
                   ? (formalComplete ? "SUCCESS" : "FAILED")
                   : ((nan || outputNonfinite)
                          ? "FAILED"
                          : (exactSuccess ? "SUCCESS" : "PARTIAL_SUCCESS")))
           << "\nresearch_goal_met="
           << ((formalPostFix ? formalComplete : exactSuccess) ? "true"
                                                               : "false")
           << "\ngeneration_nonfinite_detected="
           << (outputNonfinite ? "true" : "false")
           << "\nsame_prefix_nonfinite_count=" << samePrefixNonfiniteCount
           << "\ngeneration_nonfinite_count=" << generationNonfiniteCount
           << "\ngeneration_nan_count=" << outputNanCount
           << "\ngeneration_inf_count=" << outputInfCount
           << "\nfirst_generation_nonfinite=" << firstOutputNonfinite
           << "\noptimizer=ADAM\nlearning_rate=" << selected.lr
           << "\nsteps=" << selected.steps
           << "\nsampling="
           << (formalPostFix ? "pattern_round_robin_phase0"
                             : "pattern_balanced_phase01_round_robin")
           << "\nseed_selection_mode=" << seedSelectionMode
           << "\nrequested_seed=" << reportedRequestedSeed
           << "\nexecuted_seed_count=" << executedSeedCount
           << "\nseed_count=" << executedSeedCount
           << "\nrepresentative_seed_exact_pattern_count="
           << exactPatterns << "\nqualifying_seed_count=" << qualifyingSeeds
           << "\nexact_rollout_count=" << exactRollouts
           << "\noracle_qualifying_seed_count=" << oracleQualifyingSeeds
           << "\noracle_exact_rollout_count=" << oracleExactRollouts
           << "\nlogits_responsibility=HTP\nargmax_responsibility=CPU"
           << formalSummary.str()
           << "\ncpu_fallback=false\nnan_detected="
           << (nan || outputNanCount ? "true" : "false")
           << "\ninf_detected="
           << (outputInfCount ? "true" : "false") << '\n'
           << trajectory.str() << formalDetails.str()
           << runtime.apiTraceSummary()
           << runtime.diagnostics();
    return report.str();
  }
  const bool extra = median >= 20 || accuracy75 >= 4;
  const bool ok = allLoss && allAccuracy && extra && !nan;
  double steadyExecuteMeanUs = 0.0;
  if (runtime.metrics().executeUs.size() > 1) {
    steadyExecuteMeanUs =
        std::accumulate(runtime.metrics().executeUs.begin() + 1,
                        runtime.metrics().executeUs.end(), 0.0) /
        double(runtime.metrics().executeUs.size() - 1);
  }
  std::ostringstream clipThresholdText;
  if (selected.gradientClipping)
    clipThresholdText << selected.clipThreshold;
  else
    clipThresholdText << "NOT_APPLICABLE";
  std::ostringstream report;
  report << std::setprecision(10)
         << "TINY_LANGUAGE_MODEL\ntest=adam_convergence\nstatus="
         << (ok ? "SUCCESS" : "FAILED") << "\nconfiguration_id="
         << selected.id << "\noptimizer=ADAM\nlearning_rate=" << selected.lr
         << "\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8"
         << "\nglobal_gradient_clipping="
         << (selected.gradientClipping ? "enabled" : "disabled")
         << "\nglobal_gradient_clip_threshold="
         << clipThresholdText.str()
         << "\nglobal_gradient_clip_epsilon="
         << (selected.gradientClipping ? "1e-6" : "NOT_APPLICABLE")
         << "\nglobal_gradient_clip_scale_responsibility=CPU"
         << "\nglobal_gradient_clip_application=HTP"
         << "\nclipped_step_count=" << clippedSteps
         << "\nminimum_clip_scale=" << minimumClipScale
         << "\nmaximum_preclip_gradient_norm=" << maximumPreclipGradientNorm
         << "\nsteps="
         << selected.steps << "\nseeds=5\ninitialization_scale=1\nsampling=fixed"
         << "\nmedian_loss_reduction=" << median
         << "\naccuracy_75_seed_count=" << accuracy75
         << "\nall_seeds_loss_decreased=" << (allLoss ? "true" : "false")
         << "\nall_seeds_accuracy_increased="
         << (allAccuracy ? "true" : "false")
         << "\nadditional_convergence_condition=" << (extra ? "true" : "false")
         << "\ncpu_htp_parameter_max_abs_difference=" << worstParameter
         << "\ncpu_htp_first_moment_max_abs_difference=" << worstFirst
         << "\ncpu_htp_second_moment_max_abs_difference=" << worstSecond
         << "\ngraph_count=2\ngraph_create_count=2\ngraph_finalize_count=2"
         << "\nexecute_count_per_training_step="
         << (1 + optimizerChunkCount)
         << "\noptimizer_execute_count_per_step=" << optimizerChunkCount
         << "\noptimizer_chunking="
         << (optimizerChunkCount == 1 ? "SINGLE_VECTOR"
                                      : "QNN_FIXED_CHUNKS")
         << "\ngraph_initialization_us="
         << runtime.metrics().backendCreateUs + runtime.metrics().deviceCreateUs +
                runtime.metrics().contextCreateUs
         << "\ngraph_create_us=" << runtime.metrics().graphCreateUs
         << "\ngraph_finalize_us=" << runtime.metrics().graphFinalizeUs
         << "\nfirst_execute_us="
         << (runtime.metrics().executeUs.empty()
                 ? 0.0
                 : runtime.metrics().executeUs.front())
         << "\nsteady_execute_mean_us=" << steadyExecuteMeanUs
         << "\nupdates_per_second_estimate="
         << (steadyExecuteMeanUs > 0 ? 500000.0 / steadyExecuteMeanUs : 0.0)
         << "\ntokens_per_second_estimate="
         << (steadyExecuteMeanUs > 0
                 ? 500000.0 * config.tokens / steadyExecuteMeanUs
                 : 0.0)
         << "\noptimizer_state_elements=" << optimizerElements * 2
         << "\noptimizer_state_bytes=" << optimizerElements * 2 * sizeof(float)
         << "\nbias_correction_scalar_responsibility=CPU"
         << "\noptimizer_math_responsibility=HTP"
         << "\nparameter_optimizer_state_handoff=CPU_FIXED_BUFFER_COPY"
         << "\ncross_entropy_loss_scalar=CPU_STABLE_LOGSUMEXP"
         << "\ncross_entropy_gradient=HTP\ncpu_fallback=false"
         << "\nnan_detected=" << (nan ? "true" : "false")
         << "\ninf_detected=false\n" << trajectory.str()
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}

namespace first_nonfinite = phonelm::qnn::first_nonfinite;

first_nonfinite::Config lateDiagnosticConfig(const tiny_lm::Config &config,
                                              float lr, float clipThreshold,
                                              std::uint32_t stabilityMode = 0,
                                              std::uint32_t pairInitMode = 0,
                                              std::uint32_t totalSteps = 0) {
  return {config.tokens, config.vocabularySize, config.dimension,
          config.feedForwardDimension, config.numLayers, config.numHeads,
          config.epsilon, lr, .9f, .999f, 1.0e-8f, clipThreshold,
          stabilityMode, pairInitMode, totalSteps};
}

std::vector<first_nonfinite::RegistryEntry> lateParameterRegistry(
    const tiny_lm::Config &config, const Params &parameters) {
  std::vector<first_nonfinite::RegistryEntry> registry;
  for (const auto &entry : tiny_lm::parameterRegistry(parameters)) {
    std::vector<uint32_t> shape;
    if (entry.name == "token_embedding") {
      shape = {config.vocabularySize, config.dimension};
    } else if (entry.name == "output_projection") {
      shape = {config.dimension, config.vocabularySize};
    } else if (entry.name.find("norm") != std::string::npos) {
      shape = {config.dimension};
    } else if (entry.name.find("ffn_w1") != std::string::npos) {
      shape = {config.dimension, config.feedForwardDimension};
    } else if (entry.name.find("ffn_w2") != std::string::npos) {
      shape = {config.feedForwardDimension, config.dimension};
    } else {
      shape = {config.dimension, config.dimension};
    }
    registry.push_back({entry.name, std::move(shape)});
  }
  return registry;
}

first_nonfinite::Checkpoint privateLateCheckpoint(
    const tiny_lm::Config &config, const LateNonfiniteCheckpoint &checkpoint,
    float lr, float clipThreshold, std::uint32_t stabilityMode,
    std::uint32_t pairInitMode, std::uint32_t totalSteps) {
  first_nonfinite::Checkpoint result;
  result.config = lateDiagnosticConfig(config, lr, clipThreshold, stabilityMode,
                                       pairInitMode, totalSteps);
  result.seed = uint32_t(checkpoint.seed);
  result.completedStep = uint32_t(checkpoint.completedStep);
  result.nextOptimizerStep = uint32_t(checkpoint.completedStep + 1);
  // The batch index is deterministic from completedStep for the fixed corpus.
  result.deterministicState = "fixed_language_batch=" +
      std::to_string(checkpoint.completedStep % 4);
  result.registry = lateParameterRegistry(config, checkpoint.parameters);
  result.input = checkpoint.input;
  result.target = checkpoint.target;
  result.parameters = flattenLanguageParameters(checkpoint.parameters);
  result.adamM = flattenLanguageParameters(checkpoint.firstMoment);
  result.adamV = flattenLanguageParameters(checkpoint.secondMoment);
  first_nonfinite::finalizeCheckpoint(&result);
  return result;
}

std::string privateBytesHex(const std::vector<uint8_t> &bytes) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint8_t byte : bytes) out << std::setw(2) << unsigned(byte);
  return out.str();
}

bool appendPrivateCheckpointCodec(
    const std::string &prefix, const tiny_lm::Config &config,
    const LateNonfiniteCheckpoint &checkpoint, float lr, float clipThreshold,
    std::ostringstream &report, std::uint32_t stabilityMode = 0,
    std::uint32_t pairInitMode = 0, std::uint32_t totalSteps = 0) {
  const auto privateCheckpoint =
      privateLateCheckpoint(config, checkpoint, lr, clipThreshold,
                            stabilityMode, pairInitMode, totalSteps);
  std::vector<uint8_t> encoded;
  std::string error;
  const bool encodedOk = first_nonfinite::encodeCheckpoint(
      privateCheckpoint, &encoded, &error);
  report << prefix << "_checkpoint_codec_format=phonelm.qnn.first_nonfinite.v2\n"
         << prefix << "_checkpoint_codec_private_device_only=true\n"
         << prefix << "_checkpoint_codec_valid="
         << (encodedOk ? "true" : "false") << '\n'
         << prefix << "_checkpoint_codec_bytes=" << encoded.size() << '\n'
         << prefix << "_checkpoint_registry_hash="
         << privateCheckpoint.registryHash << '\n'
         << prefix << "_checkpoint_state_hash="
         << privateCheckpoint.stateHash << '\n'
         << prefix << "_checkpoint_codec_error="
         << (encodedOk ? "NONE" : error) << '\n';
  // This report is private diagnostic input, not a public result. Hex keeps
  // the text transport lossless without relying on a filesystem path.
  if (encodedOk)
    report << prefix << "_checkpoint_codec_private_binary_hex="
           << privateBytesHex(encoded) << '\n';
  return encodedOk;
}

struct LateVectorAudit {
  bool allFinite = true;
  size_t firstBadIndex = std::numeric_limits<size_t>::max();
  double minimum = 0, maximum = 0, mean = 0, l2 = 0;
  std::string hash;
};

LateVectorAudit auditLateVector(const std::string &prefix,
                                const std::vector<float> &values,
                                std::ostringstream &report) {
  LateVectorAudit result;
  result.hash = canonicalFloatSha256(values);
  if (!values.empty()) {
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -result.minimum;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    const float value = values[index];
    if (!std::isfinite(value)) {
      result.allFinite = false;
      if (result.firstBadIndex == std::numeric_limits<size_t>::max())
        result.firstBadIndex = index;
      continue;
    }
    result.minimum = std::min(result.minimum, double(value));
    result.maximum = std::max(result.maximum, double(value));
    result.mean += value;
    result.l2 += double(value) * value;
  }
  const size_t finiteCount = result.allFinite ? values.size() :
      std::count_if(values.begin(), values.end(),
                    [](float value) { return std::isfinite(value); });
  result.mean = finiteCount ? result.mean / finiteCount : 0;
  result.l2 = std::sqrt(result.l2);
  report << prefix << "_finite=" << (result.allFinite ? "true" : "false")
         << '\n' << prefix << "_length=" << values.size() << '\n'
         << prefix << "_min=" << result.minimum << '\n'
         << prefix << "_max=" << result.maximum << '\n'
         << prefix << "_mean=" << result.mean << '\n'
         << prefix << "_l2=" << result.l2 << '\n'
         << prefix << "_raw_float_sha256=" << rawFloatSha256(values) << '\n'
         << prefix << "_canonical_hash=" << result.hash << '\n'
         << prefix << "_first_bad_index="
         << (result.firstBadIndex == std::numeric_limits<size_t>::max()
                 ? -1 : static_cast<long long>(result.firstBadIndex)) << '\n';
  return result;
}

std::string lateRawFloatHex(const std::vector<float> &values) {
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hex << std::setw(8) << bits;
  }
  return hex.str();
}

std::vector<float> lateStateFlat(const LateNonfiniteCheckpoint &state,
                                 float lr, float clipThreshold) {
  // The metadata is part of the hash so a checkpoint cannot be replayed with
  // a different step/bias correction or clipping contract by accident.
  std::vector<float> values{float(state.seed), float(state.completedStep),
                            float(state.completedStep + 1), lr, .9f, .999f,
                            1.0e-8f, clipThreshold};
  const auto parameters = flattenLanguageParameters(state.parameters);
  values.insert(values.end(), parameters.begin(), parameters.end());
  const auto first = flattenLanguageParameters(state.firstMoment);
  const auto second = flattenLanguageParameters(state.secondMoment);
  values.insert(values.end(), first.begin(), first.end());
  values.insert(values.end(), second.begin(), second.end());
  values.insert(values.end(), state.input.begin(), state.input.end());
  values.insert(values.end(), state.target.begin(), state.target.end());
  return values;
}

std::string firstBadLateStage(const TinyTransformerTrainingOutputs &gradient,
                              const AdamOptimizerOutputs *adam,
                              std::vector<float> *badValues) {
  // ALL_INTERNAL is emitted in graph producer order. It closes the blind spot
  // between the public forward outputs and the coarse backward-region taps.
  for (const auto &tap : gradient.taps)
    if (!finite(tap.values)) {
      if (badValues) *badValues = tap.values;
      return "tap_" + tap.name;
    }
  const std::array<std::pair<const char *, const std::vector<float> *>, 7> gradientStages{{
      {"forward_embedded_input", &gradient.embeddedInput},
      {"forward_transformer_output", &gradient.output},
      {"forward_logits", &gradient.logits},
      {"forward_probabilities", &gradient.probabilities},
      {"dlogits", &gradient.dLogits},
      {"transformer_backward_dembedded_input", &gradient.dEmbeddedInput},
      {"parameter_gradients", nullptr},
  }};
  for (const auto &[name, values] : gradientStages) {
    const auto flat = values ? *values : flattenLanguageParameters(gradient.gradients);
    if (!finite(flat)) {
      if (badValues) *badValues = flat;
      return name;
    }
  }
  if (!adam) return "NONE";
  const std::array<std::pair<const char *, const std::vector<float> *>, 8> optimizerStages{{
      {"adam_m", &adam->firstMomentNext}, {"adam_v", &adam->secondMomentNext},
      {"adam_m_hat", &adam->firstMomentHat}, {"adam_v_hat", &adam->secondMomentHat},
      {"adam_sqrt_v", &adam->secondRoot}, {"adam_denominator", &adam->denominator},
      {"adam_update", &adam->scaledUpdate}, {"next_parameters", &adam->weightNext},
  }};
  for (const auto &[name, values] : optimizerStages)
    if (!finite(*values)) {
      if (badValues) *badValues = *values;
      return name;
    }
  return "NONE";
}

std::string lateProducerNode(const std::string &stage) {
  if (stage == "tap_CENTERED_S2") return "tt_ln2_center_scale";
  if (stage == "tap_SQUARE2") return "tt_ln2_square";
  if (stage.rfind("tap_", 0) == 0) return stage.substr(4);
  if (stage.rfind("forward_", 0) == 0) return "tiny_lm_transformer_forward";
  if (stage == "dlogits") return "cross_entropy_dlogits";
  if (stage.find("embedded") != std::string::npos) return "lm_dembedding";
  if (stage == "parameter_gradients") return "tiny_lm_transformer_backward";
  if (stage.rfind("adam_", 0) == 0 || stage == "next_parameters") return "adam_optimizer";
  return "NONE";
}

void reportLateTapBoundary(const std::string &prefix,
                           const TinyTransformerTrainingOutputs &gradient,
                           const std::string &stage,
                           const LateVectorAudit &badAudit,
                           std::ostringstream &report) {
  if (stage.rfind("tap_", 0) != 0) return;
  const std::string tapName = stage.substr(4);
  for (size_t index = 0; index < gradient.taps.size(); ++index) {
    if (gradient.taps[index].name != tapName) continue;
    report << prefix << "_tap_name=" << tapName << '\n'
           << prefix << "_tap_producer_node=" << lateProducerNode(stage) << '\n';
    if (index == 0) return;
    const auto &previous = gradient.taps[index - 1];
    report << prefix << "_previous_tap_name=" << previous.name << '\n'
           << prefix << "_previous_tap_producer_node="
           << lateProducerNode("tap_" + previous.name) << '\n';
    auditLateVector(prefix + "_previous_tap", previous.values, report);
    if (badAudit.firstBadIndex >= previous.values.size()) return;
    const float previousValue = previous.values[badAudit.firstBadIndex];
    std::uint32_t previousBits = 0;
    std::memcpy(&previousBits, &previousValue, sizeof(previousBits));
    report << prefix << "_previous_tap_value_at_first_bad_index="
           << previousValue << '\n'
           << prefix << "_previous_tap_value_bits_at_first_bad_index=0x"
           << std::hex << std::setw(8) << std::setfill('0') << previousBits
           << std::dec << std::setfill(' ') << '\n';
    return;
  }
}

void reportLateCheckpoint(const std::string &prefix,
                          const LateNonfiniteCheckpoint &checkpoint,
                          const tiny_lm::Config &config, float lr,
                          float clipThreshold,
                          std::ostringstream &report,
                          std::uint32_t stabilityMode,
                          std::uint32_t pairInitMode,
                          std::uint32_t totalSteps) {
  report << prefix << "_checkpoint_format=phonelm.qnn.late_nonfinite.v1\n"
         << prefix << "_checkpoint_private_raw=true\n"
         << prefix << "_checkpoint_capture_kind=LAST_FINITE_AND_FAILING_STEP_INPUT\n"
         << prefix << "_checkpoint_failing_step=" << checkpoint.completedStep + 1 << '\n'
         << prefix << "_checkpoint_seed=" << checkpoint.seed << '\n'
         << prefix << "_checkpoint_completed_step=" << checkpoint.completedStep << '\n'
         << prefix << "_checkpoint_optimizer_next_step=" << checkpoint.completedStep + 1 << '\n';
  appendPrivateCheckpointCodec(prefix, config, checkpoint, lr, clipThreshold,
                               report, stabilityMode, pairInitMode, totalSteps);
  const auto parameters = flattenLanguageParameters(checkpoint.parameters);
  const auto first = flattenLanguageParameters(checkpoint.firstMoment);
  const auto second = flattenLanguageParameters(checkpoint.secondMoment);
  auditLateVector(prefix + "_parameters", parameters, report);
  auditLateVector(prefix + "_adam_m", first, report);
  auditLateVector(prefix + "_adam_v", second, report);
  auditLateVector(prefix + "_input", checkpoint.input, report);
  auditLateVector(prefix + "_target", checkpoint.target, report);
  report << prefix << "_parameters_private_raw_hex_le_u32=" << lateRawFloatHex(parameters) << '\n'
         << prefix << "_adam_m_private_raw_hex_le_u32=" << lateRawFloatHex(first) << '\n'
         << prefix << "_adam_v_private_raw_hex_le_u32=" << lateRawFloatHex(second) << '\n'
         << prefix << "_input_private_raw_hex_le_u32=" << lateRawFloatHex(checkpoint.input) << '\n'
         << prefix << "_target_private_raw_hex_le_u32=" << lateRawFloatHex(checkpoint.target) << '\n'
         << prefix << "_checkpoint_learning_rate=" << lr << '\n'
         << prefix << "_checkpoint_beta1=0.9\n"
         << prefix << "_checkpoint_beta2=0.999\n"
         << prefix << "_checkpoint_epsilon=1e-8\n"
         << prefix << "_checkpoint_clip_threshold=" << clipThreshold << '\n';
  report << prefix << "_combined_state_canonical_hash="
         << canonicalFloatSha256(lateStateFlat(checkpoint, lr, clipThreshold)) << '\n'
         << prefix << "_combined_state_raw_float_sha256="
         << rawFloatSha256(lateStateFlat(checkpoint, lr, clipThreshold)) << '\n';
}

bool reportTwoByTwo(Runtime &runtime, const tiny_lm::Config &config,
                    const LateNonfiniteCheckpoint &checkpoint, float lr,
                    float clipThreshold, std::ostringstream &report,
                    std::string &error) {
  const auto cpuGradient = tiny_lm::forwardBackward(
      config, checkpoint.input, checkpoint.target, checkpoint.parameters, 0.0f);
  TinyTransformerTrainingOutputs htpGradient;
  if (!runtime.executeTinyTransformerTraining(checkpoint.input, checkpoint.target,
                                               checkpoint.parameters, 0.0f,
                                               htpGradient, error))
    return false;
  const double cpuNorm = gradientNorm(cpuGradient.gradients);
  const double htpNorm = gradientNorm(htpGradient.gradients);
  const float cpuClipScale =
      clipThreshold > 0 && std::isfinite(cpuNorm) && cpuNorm > 0
          ? float(std::min(1.0,
                           double(clipThreshold) / (cpuNorm + 1.0e-6)))
          : 1.0f;
  const float htpClipScale = clipThreshold > 0 && std::isfinite(htpNorm) && htpNorm > 0
      ? float(std::min(1.0, double(clipThreshold) / (htpNorm + 1.0e-6))) : 1.0f;
  const auto cpuClipped =
      scaleLanguageParameters(cpuGradient.gradients, cpuClipScale);
  const auto htpClipped =
      scaleLanguageParameters(htpGradient.gradients, htpClipScale);
  const int step = checkpoint.completedStep + 1;
  const float c1 = float(1.0 / (1.0 - std::pow(.9, double(step))));
  const float c2 = float(1.0 / (1.0 - std::pow(.999, double(step))));
  const auto pathA = tiny_lm::adamUpdate(checkpoint.parameters, cpuClipped,
      checkpoint.firstMoment, checkpoint.secondMoment, lr, .9f, .999f, 1e-8f, c1, c2);
  const auto pathB = tiny_lm::adamUpdate(checkpoint.parameters, htpClipped,
      checkpoint.firstMoment, checkpoint.secondMoment, lr, .9f, .999f, 1e-8f, c1, c2);
  Params pathC, cM, cV, pathD, dM, dV;
  AdamOptimizerOutputs rawC, rawD;
  std::vector<float> bad;
  const auto firstBad =
      firstBadLateStage(htpGradient, nullptr, &bad);
  const bool optimizerCOk = executeLanguageAdam(runtime, checkpoint.parameters,
      cpuGradient.gradients, checkpoint.firstMoment, checkpoint.secondMoment, lr, step,
      cpuClipScale, pathC, cM, cV, &rawC, error);
  const bool optimizerDRejected = firstBad != "NONE";
  const bool optimizerDOk =
      !optimizerDRejected &&
      executeLanguageAdam(runtime, checkpoint.parameters,
          htpGradient.gradients, checkpoint.firstMoment,
          checkpoint.secondMoment, lr, step, htpClipScale, pathD, dM, dV,
          &rawD, error);
  auditLateVector("two_by_two_cpu_gradient",
                  flattenLanguageParameters(cpuGradient.gradients), report);
  auditLateVector("two_by_two_htp_gradient",
                  flattenLanguageParameters(htpGradient.gradients), report);
  auditLateVector("two_by_two_cpu_clipped_gradient",
                  flattenLanguageParameters(cpuClipped), report);
  auditLateVector("two_by_two_htp_clipped_gradient",
                  flattenLanguageParameters(htpClipped), report);
  report << "two_by_two_checkpoint_seed=" << checkpoint.seed << '\n'
         << "two_by_two_checkpoint_step=" << step << '\n'
         << "two_by_two_cpu_htp_gradient_max_abs_error="
         << maxParamError(cpuGradient.gradients, htpGradient.gradients) << '\n'
         << "two_by_two_cpu_gradient_global_norm=" << cpuNorm << '\n'
         << "two_by_two_htp_gradient_global_norm=" << htpNorm << '\n'
         << "two_by_two_clip_threshold=" << clipThreshold << '\n'
         << "two_by_two_cpu_clip_scale=" << cpuClipScale << '\n'
         << "two_by_two_htp_clip_scale=" << htpClipScale << '\n'
         << "two_by_two_clip_scale=" << htpClipScale << '\n'
         << "two_by_two_beta1=0.9\ntwo_by_two_beta2=0.999\n"
         << "two_by_two_epsilon=1e-8\ntwo_by_two_bias_correction_first=" << c1 << '\n'
         << "two_by_two_bias_correction_second=" << c2 << '\n'
         << "two_by_two_step_index=" << step << '\n'
         << "two_by_two_clip_order=global_norm_then_gradient_scale_then_adam\n"
         << "two_by_two_optimizer_execute_success="
         << (optimizerCOk && optimizerDOk ? "true" : "false") << '\n'
         << "two_by_two_C_optimizer_execute_success="
         << (optimizerCOk ? "true" : "false") << '\n'
         << "two_by_two_D_optimizer_execute_success="
         << (optimizerDOk ? "true" : "false") << '\n'
         << "two_by_two_D_optimizer_input_rejected="
         << (optimizerDRejected ? "true" : "false") << '\n'
         << "two_by_two_D_optimizer_rejection_reason="
         << (optimizerDRejected ? "NONFINITE_GRADIENT" : "NONE") << '\n';
  const std::array<std::tuple<const char *, const Params *, const Params *>, 3> paths{{
      {"A_cpu_gradient_cpu_adam", &pathA.next, nullptr},
      {"B_htp_gradient_cpu_adam", &pathB.next, nullptr},
      {"C_cpu_gradient_htp_adam", &pathC, &pathA.next},
  }};
  std::array<bool, 4> pathFinite{};
  size_t pathIndex = 0;
  for (const auto &[name, actual, cpuReference] : paths) {
    const auto flat = flattenLanguageParameters(*actual);
    const auto audit = auditLateVector("two_by_two_" + std::string(name) + "_parameters", flat, report);
    pathFinite[pathIndex++] = audit.allFinite;
    report << "two_by_two_" << name << "_finite=" << (audit.allFinite ? "true" : "false") << '\n'
           << "two_by_two_" << name << "_parameter_max_abs_error="
           << (cpuReference ? maxParamError(*actual, *cpuReference) : 0) << '\n';
  }
  pathFinite[3] = optimizerDOk && finiteParams(pathD);
  report << "two_by_two_D_htp_gradient_htp_adam_finite="
         << (pathFinite[3] ? "true" : "false") << '\n'
         << "two_by_two_D_htp_gradient_htp_adam_parameter_max_abs_error="
         << (optimizerDOk ? maxParamError(pathD, pathB.next) : 0) << '\n';
  if (optimizerCOk) {
    report << "two_by_two_C_m_next_max_abs_error=" << maxParamError(cM, pathA.firstMoment) << '\n'
           << "two_by_two_C_v_next_max_abs_error=" << maxParamError(cV, pathA.secondMoment) << '\n';
    auditLateVector("two_by_two_C_adam_sqrt_v", rawC.secondRoot, report);
    auditLateVector("two_by_two_C_adam_denominator", rawC.denominator, report);
    auditLateVector("two_by_two_C_adam_update", rawC.scaledUpdate, report);
  }
  if (optimizerDOk) {
    report << "two_by_two_D_m_next_max_abs_error=" << maxParamError(dM, pathB.firstMoment) << '\n'
           << "two_by_two_D_v_next_max_abs_error=" << maxParamError(dV, pathB.secondMoment) << '\n';
    auditLateVector("two_by_two_D_adam_sqrt_v", rawD.secondRoot, report);
    auditLateVector("two_by_two_D_adam_denominator", rawD.denominator, report);
    auditLateVector("two_by_two_D_adam_update", rawD.scaledUpdate, report);
  }
  report << "two_by_two_first_bad_stage=" << firstBad << '\n'
         << "two_by_two_first_bad_producer_node=" << lateProducerNode(firstBad) << '\n';
  if (firstBad != "NONE") {
    const auto badAudit =
        auditLateVector("two_by_two_first_bad", bad, report);
    reportLateTapBoundary("two_by_two_first_bad", htpGradient, firstBad,
                          badAudit, report);
  }
  return optimizerCOk && optimizerDRejected &&
         pathFinite[0] && !pathFinite[1] && pathFinite[2] && !pathFinite[3] &&
         firstBad == "tap_SQUARE2";
}

bool reportPostFixCheckpointReplay(
    const tiny_lm::Config &config,
    const LateNonfiniteCheckpoint &checkpoint, float lr,
    float clipThreshold, std::ostringstream &report, std::string &error) {
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  options.tinyTransformerCenteredScale = 8.0f;
  runtime.setOptions(options);
  const auto parameterCount =
      uint32_t(flattenLanguageParameters(checkpoint.parameters).size());
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL,
          TinyTransformerTrainingTapSet::ALL_INTERNAL) ||
      !runtime.prepareAdamOptimizer(parameterCount, error))
    return false;

  const int step = checkpoint.completedStep + 1;
  const float c1 = float(1.0 / (1.0 - std::pow(.9, double(step))));
  const float c2 = float(1.0 / (1.0 - std::pow(.999, double(step))));
  const auto cpuGradient = tiny_lm::forwardBackward(
      config, checkpoint.input, checkpoint.target, checkpoint.parameters,
      0.0f);
  const double cpuNorm = gradientNorm(cpuGradient.gradients);
  const float cpuScale =
      clipThreshold > 0 && std::isfinite(cpuNorm) && cpuNorm > 0
          ? float(std::min(1.0,
                           double(clipThreshold) / (cpuNorm + 1.0e-6)))
          : 1.0f;
  const auto cpuClipped =
      scaleLanguageParameters(cpuGradient.gradients, cpuScale);
  const auto pathA = tiny_lm::adamUpdate(
      checkpoint.parameters, cpuClipped, checkpoint.firstMoment,
      checkpoint.secondMoment, lr, .9f, .999f, 1e-8f, c1, c2);

  bool allFinite = true, deterministic = true;
  std::string firstSignature;
  TinyTransformerTrainingOutputs representativeGradient;
  Params representativeNext, representativeM, representativeV;
  for (int replay = 0; replay < 100; ++replay) {
    TinyTransformerTrainingOutputs gradient;
    if (!runtime.executeTinyTransformerTraining(
            checkpoint.input, checkpoint.target, checkpoint.parameters, 0.0f,
            gradient, error)) {
      allFinite = false;
      return false;
    }
    const double norm = gradientNorm(gradient.gradients);
    const float scale =
        clipThreshold > 0 && std::isfinite(norm) && norm > 0
            ? float(std::min(1.0, double(clipThreshold) / (norm + 1.0e-6)))
            : 1.0f;
    Params next, nextM, nextV;
    AdamOptimizerOutputs adam;
    if (!executeLanguageAdam(
            runtime, checkpoint.parameters, gradient.gradients,
            checkpoint.firstMoment, checkpoint.secondMoment, lr, step, scale,
            next, nextM, nextV, &adam, error)) {
      allFinite = false;
      return false;
    }
    std::vector<float> bad;
    const auto badStage = firstBadLateStage(gradient, &adam, &bad);
    const std::string signature =
        rawFloatSha256(gradient.logits) + ":" +
        rawFloatSha256(adam.weightNext);
    if (replay == 0) {
      firstSignature = signature;
      representativeGradient = gradient;
      representativeNext = next;
      representativeM = nextM;
      representativeV = nextV;
    } else {
      deterministic = deterministic && signature == firstSignature;
    }
    allFinite =
        allFinite && badStage == "NONE" && std::isfinite(gradient.loss);
  }

  const double htpNorm = gradientNorm(representativeGradient.gradients);
  const float htpScale =
      clipThreshold > 0 && std::isfinite(htpNorm) && htpNorm > 0
          ? float(std::min(1.0,
                           double(clipThreshold) / (htpNorm + 1.0e-6)))
          : 1.0f;
  const auto pathB = tiny_lm::adamUpdate(
      checkpoint.parameters,
      scaleLanguageParameters(representativeGradient.gradients, htpScale),
      checkpoint.firstMoment, checkpoint.secondMoment, lr, .9f, .999f,
      1e-8f, c1, c2);
  Params pathC, cM, cV;
  AdamOptimizerOutputs rawC;
  if (!executeLanguageAdam(
          runtime, checkpoint.parameters, cpuGradient.gradients,
          checkpoint.firstMoment, checkpoint.secondMoment, lr, step, cpuScale,
          pathC, cM, cV, &rawC, error))
    return false;
  report << "post_fix_same_checkpoint_seed=" << checkpoint.seed << '\n'
         << "post_fix_same_checkpoint_step=" << step << '\n'
         << "post_fix_same_checkpoint_centered_scale=8\n"
         << "post_fix_same_checkpoint_replay_count=100\n"
         << "post_fix_same_checkpoint_all_finite="
         << (allFinite ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_deterministic="
         << (deterministic ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_signature=" << firstSignature << '\n'
         << "post_fix_same_checkpoint_cpu_htp_gradient_max_abs_error="
         << maxParamError(cpuGradient.gradients,
                          representativeGradient.gradients)
         << '\n'
         << "post_fix_same_checkpoint_cpu_gradient_norm="
         << cpuNorm << '\n'
         << "post_fix_same_checkpoint_htp_gradient_norm=" << htpNorm << '\n'
         << "post_fix_same_checkpoint_cpu_clip_scale=" << cpuScale << '\n'
         << "post_fix_same_checkpoint_htp_clip_scale=" << htpScale << '\n'
         << "post_fix_same_checkpoint_path_a_finite="
         << (finiteParams(pathA.next) ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_path_b_finite="
         << (finiteParams(pathB.next) ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_path_c_finite="
         << (finiteParams(pathC) ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_path_d_finite="
         << (finiteParams(representativeNext) ? "true" : "false") << '\n'
         << "post_fix_same_checkpoint_path_c_parameter_max_abs_error="
         << maxParamError(pathC, pathA.next) << '\n'
         << "post_fix_same_checkpoint_path_d_parameter_max_abs_error="
         << maxParamError(representativeNext, pathB.next) << '\n'
         << "post_fix_same_checkpoint_path_c_m_max_abs_error="
         << maxParamError(cM, pathA.firstMoment) << '\n'
         << "post_fix_same_checkpoint_path_c_v_max_abs_error="
         << maxParamError(cV, pathA.secondMoment) << '\n'
         << "post_fix_same_checkpoint_path_d_m_max_abs_error="
         << maxParamError(representativeM, pathB.firstMoment) << '\n'
         << "post_fix_same_checkpoint_path_d_v_max_abs_error="
         << maxParamError(representativeV, pathB.secondMoment) << '\n'
         << "post_fix_same_checkpoint_qnn_execute_count="
         << runtime.metrics().graphExecuteCount << '\n';
  return allFinite && deterministic && finiteParams(pathA.next) &&
         finiteParams(pathB.next) && finiteParams(pathC) &&
         finiteParams(representativeNext);
}

std::string runLateNonfiniteExperiment(bool diagnostic) {
  constexpr int kSeedCount = 5;
  const int steps = diagnostic ? 1000 : 320;
  const float lr = diagnostic ? .0003f : .003f;
  const float clipThreshold = diagnostic ? 5.0f : 0.0f;
  tiny_lm::Config config;
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  // The diagnostic deliberately reconstructs the original scale-64
  // trajectory. Its captured checkpoint is then replayed below by a separate
  // scale-8 runtime, giving a same-state pre/post-fix comparison.
  options.tinyTransformerCenteredScale = diagnostic ? 64.0f : 8.0f;
  runtime.setOptions(options);
  std::string error;
  const auto shape = tiny_lm::initialParameters(config, 1);
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(config.tokens, config.dimension,
          config.feedForwardDimension, config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL,
          diagnostic ? TinyTransformerTrainingTapSet::ALL_INTERNAL
                     : TinyTransformerTrainingTapSet::NONE) ||
      !runtime.prepareAdamOptimizer(uint32_t(flattenLanguageParameters(shape).size()), error))
    return failure("late_nonfinite_prepare", error, runtime);
  std::ostringstream report;
  report << std::setprecision(10) << "TINY_LANGUAGE_MODEL\ntest=adam_late_nonfinite_"
         << (diagnostic ? "diagnostic" : "baseline")
         << "\nstatus=SUCCESS\noptimizer=ADAM\nlearning_rate=" << lr
         << "\nsteps=" << steps << "\nseeds=" << kSeedCount
         << "\nclip_threshold=" << (diagnostic ? "5" : "NONE")
         << "\ntransformer_centered_scale="
         << options.tinyTransformerCenteredScale
         << "\nclipping=" << (diagnostic ? "global_norm_5" : "disabled")
         << "\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8\n"
          << "experiment_outcome_nonfinite_is_not_harness_failure=true\n"
          << "checkpoint_storage=PRIVATE_DEVICE_REPORT_ONLY\n"
          << "checkpoint_raw_fields_emitted=PRIVATE_DEVICE_REPORT_ONLY\n";
  int finiteSeeds = 0, failingCheckpoints = 0;
  bool fixedReplayChecks = true;
  bool twoByTwoChecks = true;
  bool postFixReplayChecks = true;
  for (int seed = 1; seed <= kSeedCount; ++seed) {
    auto htp = tiny_lm::initialParameters(config, seed), cpu = htp;
    auto htpM = zeroLanguageParameters(htp), htpV = htpM, cpuM = htpM, cpuV = htpM;
    LanguageQuality initial;
    if (!htpLanguageQuality(runtime, config, htp, 1, initial, error))
      return failure("late_nonfinite_initial", error, runtime);
    int lastFinite = 0, firstNonfinite = -1;
    std::string firstTensor = "NONE";
    LateNonfiniteCheckpoint checkpoint;
    double lastGradientNorm = 0, lastParameterNorm = paramNorm(htp);
    double lastDenominatorNorm = 0, lastDenominatorMinimum = 0;
    float lastLoss = initial.loss, lastAccuracy = initial.accuracy;
    bool executeOk = true;
    const auto seedExecuteStart = runtime.metrics().graphExecuteCount;
    for (int step = 1; step <= steps; ++step) {
      const auto batch = languageBatch(config, uint32_t((step - 1) % 4));
      LateNonfiniteCheckpoint before{seed, step - 1, htp, htpM, htpV, batch.first, batch.second};
      const auto cpuGradient = tiny_lm::forwardBackward(config, batch.first, batch.second, cpu, 0.0f);
      const float c1 = float(1.0 / (1.0 - std::pow(.9, double(step))));
      const float c2 = float(1.0 / (1.0 - std::pow(.999, double(step))));
      const auto cpuUpdate = tiny_lm::adamUpdate(cpu, cpuGradient.gradients, cpuM, cpuV,
          lr, .9f, .999f, 1e-8f, c1, c2);
      TinyTransformerTrainingOutputs gradient;
      if (!runtime.executeTinyTransformerTraining(batch.first, batch.second, htp, 0.0f,
                                                   gradient, error)) {
        executeOk = false;
        firstNonfinite = step;
        firstTensor = "qnn_execute";
        checkpoint = std::move(before);
        break;
      }
      lastLoss = gradient.loss;
      lastAccuracy = tokenAccuracy(gradient.logits, batch.second, config.tokens, config.vocabularySize);
      lastGradientNorm = gradientNorm(gradient.gradients);
      std::vector<float> preAdamBadValues;
      const auto preAdamBad =
          firstBadLateStage(gradient, nullptr, &preAdamBadValues);
      if (preAdamBad != "NONE" || !std::isfinite(gradient.loss)) {
        firstNonfinite = step;
        firstTensor =
            preAdamBad == "NONE" ? "loss" : preAdamBad;
        checkpoint = std::move(before);
        report << "seed_" << seed << "_first_bad_producer_node="
               << lateProducerNode(firstTensor) << '\n';
        if (!preAdamBadValues.empty()) {
          const std::string prefix =
              "seed_" + std::to_string(seed) + "_first_bad";
          const auto badAudit =
              auditLateVector(prefix, preAdamBadValues, report);
          reportLateTapBoundary(prefix, gradient, firstTensor, badAudit,
                                report);
        }
        break;
      }
      const float clipScale = clipThreshold > 0 && std::isfinite(lastGradientNorm) && lastGradientNorm > 0
          ? float(std::min(1.0, double(clipThreshold) / (lastGradientNorm + 1.0e-6))) : 1.0f;
      Params next, nextM, nextV;
      AdamOptimizerOutputs raw;
      if (!executeLanguageAdam(runtime, htp, gradient.gradients, htpM, htpV, lr, step,
                               clipScale, next, nextM, nextV, &raw, error)) {
        executeOk = false;
        firstNonfinite = step;
        firstTensor = "qnn_adam_execute";
        checkpoint = std::move(before);
        break;
      }
      // Preserve the attempted optimizer evidence even when this exact step
      // is the first non-finite one.
      lastDenominatorNorm = vectorNorm(raw.denominator);
      lastDenominatorMinimum = raw.denominator.empty() ? 0 :
          *std::min_element(raw.denominator.begin(), raw.denominator.end());
      std::vector<float> badValues;
      const auto bad = firstBadLateStage(gradient, &raw, &badValues);
      if (bad != "NONE" || !std::isfinite(gradient.loss)) {
        firstNonfinite = step;
        firstTensor = bad == "NONE" ? "loss" : bad;
        checkpoint = std::move(before);
        report << "seed_" << seed << "_first_bad_producer_node="
               << lateProducerNode(firstTensor) << '\n';
        if (!badValues.empty()) {
          const std::string prefix =
              "seed_" + std::to_string(seed) + "_first_bad";
          const auto badAudit = auditLateVector(prefix, badValues, report);
          reportLateTapBoundary(prefix, gradient, firstTensor, badAudit,
                                report);
        }
        break;
      }
      htp = std::move(next); htpM = std::move(nextM); htpV = std::move(nextV);
      cpu = cpuUpdate.next; cpuM = cpuUpdate.firstMoment; cpuV = cpuUpdate.secondMoment;
      lastFinite = step;
      lastParameterNorm = paramNorm(htp);
    }
    report << "seed_" << seed << "_initial_loss=" << initial.loss << '\n'
           << "seed_" << seed << "_initial_accuracy=" << initial.accuracy << '\n'
           << "seed_" << seed << "_final_loss=" << lastLoss << '\n'
           << "seed_" << seed << "_final_accuracy=" << lastAccuracy << '\n'
           << "seed_" << seed << "_last_finite_step=" << lastFinite << '\n'
           << "seed_" << seed << "_first_nonfinite_step=" << firstNonfinite << '\n'
           << "seed_" << seed << "_first_nonfinite_tensor=" << firstTensor << '\n'
           << "seed_" << seed << "_gradient_norm=" << lastGradientNorm << '\n'
           << "seed_" << seed << "_adam_m_norm=" << paramNorm(htpM) << '\n'
           << "seed_" << seed << "_adam_v_norm=" << paramNorm(htpV) << '\n'
           << "seed_" << seed << "_adam_denominator_norm=" << lastDenominatorNorm << '\n'
           << "seed_" << seed << "_adam_denominator_min=" << lastDenominatorMinimum << '\n'
           << "seed_" << seed << "_parameter_norm=" << lastParameterNorm << '\n'
           << "seed_" << seed << "_qnn_execute_result=" << (executeOk ? "SUCCESS" : "FAILED") << '\n'
           << "seed_" << seed << "_qnn_execute_count="
           << (runtime.metrics().graphExecuteCount - seedExecuteStart) << '\n'
           << "seed_" << seed << "_cpu_control_backend=HOST_CPP\n"
           << "seed_" << seed << "_cpu_control_all_steps_finite="
           << (finiteParams(cpu) && finiteParams(cpuM) && finiteParams(cpuV) ? "true" : "false") << '\n'
           << "seed_" << seed << "_cpu_control_parameter_norm=" << paramNorm(cpu) << '\n';
    if (firstNonfinite < 0) ++finiteSeeds;
    if (diagnostic && firstNonfinite > 0) {
      ++failingCheckpoints;
      const std::string checkpointPrefix = "seed_" + std::to_string(seed) + "_checkpoint";
      reportLateCheckpoint(checkpointPrefix, checkpoint, config, lr, clipThreshold, report);
      std::string firstSignature;
      bool stable = true;
      int completedReplays = 0;
      for (int replay = 0; replay < 100; ++replay) {
        TinyTransformerTrainingOutputs replayGradient;
        if (!runtime.executeTinyTransformerTraining(checkpoint.input, checkpoint.target,
              checkpoint.parameters, 0.0f, replayGradient, error)) { stable = false; break; }
        std::vector<float> replayBad;
        const auto replayStage =
            firstBadLateStage(replayGradient, nullptr, &replayBad);
        size_t replayBadIndex = std::numeric_limits<size_t>::max();
        for (size_t index = 0; index < replayBad.size(); ++index)
          if (!std::isfinite(replayBad[index])) { replayBadIndex = index; break; }
        std::uint32_t replayBadBits = 0;
        if (replayBadIndex != std::numeric_limits<size_t>::max())
          std::memcpy(&replayBadBits, &replayBad[replayBadIndex], sizeof(replayBadBits));
        std::ostringstream signatureStream;
        signatureStream << replayStage << ":raw=" << rawFloatSha256(replayBad)
                        << ":canonical=" << canonicalFloatSha256(replayBad)
                        << ":index=" << (replayBadIndex == std::numeric_limits<size_t>::max()
                              ? -1 : static_cast<long long>(replayBadIndex))
                        << ":value_bits=0x" << std::hex << replayBadBits;
        const std::string signature = signatureStream.str();
        stable = stable && replayStage == firstTensor &&
                 replayStage != "NONE" &&
                 replayBadIndex != std::numeric_limits<size_t>::max();
        if (replay == 0) firstSignature = signature;
        else stable = stable && signature == firstSignature;
        ++completedReplays;
      }
      stable = stable && completedReplays == 100;
      fixedReplayChecks = fixedReplayChecks && stable;
      report << "seed_" << seed << "_fixed_replay_count=" << completedReplays
             << "\nseed_" << seed
             << "_fixed_replay_signature=" << firstSignature << '\n'
             << "seed_" << seed << "_fixed_replay_reproducible=" << (stable ? "true" : "false") << '\n';
      twoByTwoChecks = reportTwoByTwo(
          runtime, config, checkpoint, lr, clipThreshold, report, error) &&
          twoByTwoChecks;
      if (!error.empty()) return failure("late_nonfinite_two_by_two", error, runtime);
      postFixReplayChecks = reportPostFixCheckpointReplay(
          config, checkpoint, lr, clipThreshold, report, error) &&
          postFixReplayChecks;
      if (!error.empty())
        return failure("late_nonfinite_post_fix_replay", error, runtime);
    }
  }
  if (diagnostic &&
      (finiteSeeds != 0 || failingCheckpoints != kSeedCount ||
       !fixedReplayChecks || !twoByTwoChecks || !postFixReplayChecks)) {
    return failure(
        "late_nonfinite_diagnostic_assertions",
        "Expected five deterministic legacy failures, A/C finite, B "
        "non-finite, D rejected before Adam execution, and five finite "
        "deterministic post-fix same-checkpoint replays.",
        runtime);
  }
  if (!diagnostic && finiteSeeds != kSeedCount) {
    return failure("late_nonfinite_baseline_assertions",
                   "Expected all five post-fix seeds to remain finite.",
                   runtime);
  }
  report << "finite_seed_count=" << finiteSeeds << "/5\n"
         << "htp_finite_seed_count=" << finiteSeeds << "/5\n"
         << "failing_checkpoint_count=" << failingCheckpoints << '\n'
         << "cpu_optimizer_formula=adam_m_v_bias_correction_epsilon_1e-8\n"
         << "htp_optimizer_formula=adam_m_v_bias_correction_epsilon_1e-8\n"
         << "cpu_fallback=false\n" << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}

} // namespace
// Nicopedia real-text HTP training.  The private tokenized pilot cache
// (NPRTBYTEV1, written by scripts/nicopedia_real_text_pipeline.py) is the
// single data source; the same records feed the CPU reference and the QNN
// graph with identical input/target identity.  The device path reads the
// cache from the app-private files directory (pushed by the host runner);
// no article text, token sequence, or checkpoint is published.
namespace {
constexpr std::uint64_t kNprtFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kNprtFnvPrime = 1099511628211ull;

std::uint64_t nprtFnvBytes(const void *data, std::size_t bytes,
                           std::uint64_t hash = kNprtFnvOffset) {
  const auto *input = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= input[i];
    hash *= kNprtFnvPrime;
  }
  return hash;
}

std::string nprtHex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << value;
  return stream.str();
}

uint32_t nprtReadU32(std::istream &input) {
  std::array<std::uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("NPRT_CACHE_TRUNCATED_U32");
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

std::uint64_t nprtReadU64(std::istream &input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("NPRT_CACHE_TRUNCATED_U64");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes) value = (value << 8) | byte;
  return value;
}

struct NprtRecord {
  std::uint64_t articleHash = 0;
  std::vector<std::uint8_t> window;
};

struct NprtCache {
  uint32_t context = 0;
  uint32_t vocabulary = 0;
  std::vector<NprtRecord> records;
  std::string contentHash;
};

// Byte-for-byte compatible with host_tests/nicopedia_real_text_pilot.cpp
// loadCache (NPRTBYTEV1).  The device and host runners share this contract so
// a batch selected on the host has the same identity on the device.
NprtCache loadNprtCache(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("NPRT_CACHE_OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTBYTEV1\n") throw std::runtime_error("NPRT_CACHE_MAGIC");
  NprtCache cache;
  cache.context = nprtReadU32(input);
  cache.vocabulary = nprtReadU32(input);
  const std::uint64_t count = nprtReadU64(input);
  if (cache.context < 8 || cache.context > 256 || cache.vocabulary != 256 ||
      count > 10000000)
    throw std::runtime_error("NPRT_CACHE_HEADER_INVALID");
  cache.records.reserve(static_cast<std::size_t>(count));
  std::uint64_t hash = kNprtFnvOffset;
  hash = nprtFnvBytes(&cache.context, sizeof(cache.context), hash);
  hash = nprtFnvBytes(&cache.vocabulary, sizeof(cache.vocabulary), hash);
  hash = nprtFnvBytes(&count, sizeof(count), hash);
  for (std::uint64_t i = 0; i < count; ++i) {
    NprtRecord record;
    record.articleHash = nprtReadU64(input);
    record.window.resize(cache.context + 1);
    input.read(reinterpret_cast<char *>(record.window.data()),
               static_cast<std::streamsize>(record.window.size()));
    if (!input) throw std::runtime_error("NPRT_CACHE_RECORD_TRUNCATED");
    hash = nprtFnvBytes(&record.articleHash, sizeof(record.articleHash), hash);
    hash = nprtFnvBytes(record.window.data(), record.window.size(), hash);
    cache.records.push_back(std::move(record));
  }
  if (input.get() != std::char_traits<char>::eof())
    throw std::runtime_error("NPRT_CACHE_TRAILING_BYTES");
  cache.contentHash = nprtHex64(hash);
  if (cache.records.empty()) throw std::runtime_error("NPRT_CACHE_EMPTY");
  return cache;
}

std::uint64_t nprtSplitMix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

// Identical to the CPU pilot trainingOrder(recordCount, steps, batch, 20260806).
std::vector<std::size_t> nprtTrainingOrder(std::size_t recordCount,
                                           uint32_t steps,
                                           uint32_t batchSize) {
  if (!recordCount) throw std::runtime_error("NPRT_TRAIN_CACHE_EMPTY");
  std::vector<std::size_t> order;
  order.reserve(std::size_t(steps) * batchSize);
  std::uint64_t state = 20260806;
  for (std::size_t i = 0; i < std::size_t(steps) * batchSize; ++i) {
    state = nprtSplitMix(state + i);
    order.push_back(static_cast<std::size_t>(state % recordCount));
  }
  return order;
}

std::string nprtOrderHash(const std::vector<std::size_t> &order) {
  std::uint64_t hash = kNprtFnvOffset;
  for (std::size_t index : order) {
    const std::uint64_t value = index;
    hash = nprtFnvBytes(&value, sizeof(value), hash);
  }
  return nprtHex64(hash);
}

struct NprtBatch {
  std::vector<float> input;
  std::vector<float> target;
  std::uint64_t articleHash = 0;
};

NprtBatch nprtBatch(const tiny_lm::Config &config, const NprtCache &cache,
                    std::size_t recordIndex) {
  const auto &record = cache.records.at(recordIndex);
  std::vector<uint32_t> input(config.tokens), target(config.tokens);
  for (uint32_t i = 0; i < config.tokens; ++i) {
    input[i] = record.window[i];
    target[i] = record.window[i + 1];
  }
  NprtBatch batch;
  batch.input = tiny_lm::oneHot(input, config.vocabularySize);
  batch.target = tiny_lm::oneHot(target, config.vocabularySize);
  batch.articleHash = record.articleHash;
  return batch;
}

struct NprtComparison {
  double maxAbs = 0, meanAbs = 0, maxRelative = 0, l2 = 0;
  size_t nonfinite = 0;
  double cosine = 1.0, htpNormOverCpuNorm = 1.0;
};

NprtComparison compareNprt(const std::vector<float> &cpu,
                           const std::vector<float> &htp) {
  NprtComparison result;
  if (cpu.size() != htp.size()) {
    result.maxAbs = result.meanAbs = result.maxRelative = result.l2 =
        std::numeric_limits<double>::infinity();
    return result;
  }
  double cpuL2 = 0, htpL2 = 0, dot = 0;
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (!std::isfinite(cpu[i]) || !std::isfinite(htp[i])) {
      ++result.nonfinite;
      continue;
    }
    const double e = std::abs(double(cpu[i]) - htp[i]);
    result.maxAbs = std::max(result.maxAbs, e);
    result.meanAbs += e;
    result.maxRelative = std::max(
        result.maxRelative, e / std::max(1.0e-12, std::abs(double(cpu[i]))));
    result.l2 += e * e;
    cpuL2 += double(cpu[i]) * cpu[i];
    htpL2 += double(htp[i]) * htp[i];
    dot += double(cpu[i]) * htp[i];
  }
  if (!cpu.empty()) result.meanAbs /= cpu.size();
  result.l2 = std::sqrt(result.l2);
  result.cosine = cpuL2 && htpL2 ? dot / std::sqrt(cpuL2 * htpL2) : 1.0;
  result.htpNormOverCpuNorm = cpuL2 ? std::sqrt(htpL2 / cpuL2) : 1.0;
  return result;
}

double nprtMaxParamError(const Params &cpu, const Params &htp) {
  const auto cpuRegistry = tiny_lm::parameterRegistry(cpu);
  const auto htpRegistry = tiny_lm::parameterRegistry(htp);
  double worst = 0;
  if (cpuRegistry.size() != htpRegistry.size())
    return std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < cpuRegistry.size(); ++i) {
    if (cpuRegistry[i].name != htpRegistry[i].name)
      return std::numeric_limits<double>::infinity();
    worst = std::max(worst,
                     compareNprt(*cpuRegistry[i].values, *htpRegistry[i].values)
                         .maxAbs);
  }
  return worst;
}

std::string nprtParameterHash(const Params &parameters) {
  std::uint64_t hash = kNprtFnvOffset;
  for (const auto &info : tiny_lm::parameterRegistry(parameters)) {
    hash = nprtFnvBytes(info.name.data(), info.name.size(), hash);
    const std::uint64_t count = info.values->size();
    hash = nprtFnvBytes(&count, sizeof(count), hash);
    hash = nprtFnvBytes(info.values->data(),
                        info.values->size() * sizeof(float), hash);
  }
  return nprtHex64(hash);
}

void nprtWriteU32(std::ostream &output, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    output.put(static_cast<char>((value >> shift) & 0xffu));
}

void nprtWriteU64(std::ostream &output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    output.put(static_cast<char>((value >> shift) & 0xffu));
}

// NPRTCKPTV1-compatible private checkpoint writer (identical layout to
// host_tests/nicopedia_real_text_pilot.cpp saveCheckpoint).  The host pulls
// this file and evaluates it with the CPU pilot evaluation path; it never
// leaves the app-private directory and never enters public artifacts.
bool nprtSaveCheckpoint(const std::string &path, const tiny_lm::Config &config,
                        uint32_t seed, uint32_t step,
                        const Params &parameters) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  output.write("NPRTCKPTV1\n", 11);
  nprtWriteU32(output, config.vocabularySize);
  nprtWriteU32(output, config.tokens);
  nprtWriteU32(output, config.dimension);
  nprtWriteU32(output, config.feedForwardDimension);
  nprtWriteU32(output, config.numLayers);
  nprtWriteU32(output, config.numHeads);
  nprtWriteU32(output, seed);
  nprtWriteU32(output, step);
  const auto registry = tiny_lm::parameterRegistry(parameters);
  nprtWriteU32(output, static_cast<uint32_t>(registry.size()));
  for (const auto &info : registry) {
    nprtWriteU32(output, static_cast<uint32_t>(info.name.size()));
    output.write(info.name.data(), static_cast<std::streamsize>(info.name.size()));
    nprtWriteU64(output, info.values->size());
    output.write(reinterpret_cast<const char *>(info.values->data()),
                 static_cast<std::streamsize>(info.values->size() * sizeof(float)));
  }
  return static_cast<bool>(output);
}

// ---------------------------------------------------------------------------
// Nicopedia byte-level generation on the HTP graph.
//
// The checkpoint is the NPRTCKPTV1 private checkpoint written by the HTP
// training milestone.  Identity is validated fail-closed on the device
// (header config/seed/step, canonical registry order and element counts,
// finiteness) and the returned registry hash is compared by the host against
// the approved anchor from the training milestone's device report.
// ---------------------------------------------------------------------------
namespace {

constexpr std::uint64_t kNprtMaxCheckpointBytes = 64u * 1024u * 1024u;

std::vector<std::uint8_t> nprtReadFileBytes(const std::string &path,
                                            std::uint64_t maxBytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("FILE_OPEN_FAILED");
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0 || static_cast<std::uint64_t>(size) > maxBytes)
    throw std::runtime_error("FILE_SIZE_INVALID");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (size > 0)
    input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input && size > 0) throw std::runtime_error("FILE_READ_TRUNCATED");
  return bytes;
}

void nprtAssignRegistryMember(Params &target, uint32_t layers,
                              const std::string &name,
                              std::vector<float> &&values) {
  if (name == "token_embedding") {
    target.tokenEmbedding = std::move(values);
    return;
  }
  if (name == "output_projection") {
    target.outputProjection = std::move(values);
    return;
  }
  if (name.rfind("layer_", 0) != 0)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const size_t dot = name.find('.');
  if (dot == std::string::npos || dot < 7)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const std::string indexText = name.substr(6, dot - 6);
  if (indexText.empty() || indexText.size() > 3)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  char *end = nullptr;
  const long index = std::strtol(indexText.c_str(), &end, 10);
  if (!end || *end != '\0' || index < 0 ||
      static_cast<uint32_t>(index) >= layers)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const std::string suffix = name.substr(dot + 1);
  TinyTransformerLayerParameters *layer =
      index == 0 ? static_cast<TinyTransformerLayerParameters *>(&target)
                 : &target.layers[static_cast<std::size_t>(index) - 1];
  if (suffix == "norm1_gamma") layer->gamma1 = std::move(values);
  else if (suffix == "norm1_beta") layer->beta1 = std::move(values);
  else if (suffix == "wq") layer->wq = std::move(values);
  else if (suffix == "wk") layer->wk = std::move(values);
  else if (suffix == "wv") layer->wv = std::move(values);
  else if (suffix == "wo") layer->wo = std::move(values);
  else if (suffix == "norm2_gamma") layer->gamma2 = std::move(values);
  else if (suffix == "norm2_beta") layer->beta2 = std::move(values);
  else if (suffix == "ffn_w1") layer->w1 = std::move(values);
  else if (suffix == "ffn_w2") layer->w2 = std::move(values);
  else throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
}

bool nprtParseCheckpointStep(const std::string &path, uint32_t expectedSeed,
                             uint32_t expectedLayers, uint32_t *step) {
  const std::string base = path.substr(path.find_last_of('/') + 1);
  const std::string prefix = "htp-seed" + std::to_string(expectedSeed) + "-l" +
                             std::to_string(expectedLayers) + "-step";
  if (base.size() <= prefix.size() + 5 ||
      base.compare(0, prefix.size(), prefix) != 0 ||
      base.compare(base.size() - 5, 5, ".ckpt") != 0)
    return false;
  const std::string digits =
      base.substr(prefix.size(), base.size() - prefix.size() - 5);
  if (digits.empty() || digits.size() > 6) return false;
  for (char digit : digits)
    if (digit < '0' || digit > '9') return false;
  char *end = nullptr;
  const long value = std::strtol(digits.c_str(), &end, 10);
  if (!end || *end != '\0' || value <= 0 || value >= 1000000) return false;
  *step = static_cast<uint32_t>(value);
  return true;
}

struct LoadedNprtCheckpoint {
  uint32_t vocabulary = 0, tokens = 0, dimension = 0, feedForward = 0;
  uint32_t layers = 0, heads = 0, seed = 0, step = 0;
  uint32_t registryCount = 0;
  std::uint64_t fileBytes = 0;
  std::uint64_t parameterElements = 0;
  bool finite = true;
  Params parameters;
  std::string parameterHash;
};

LoadedNprtCheckpoint nprtLoadCheckpointForGeneration(
    const std::string &path, const tiny_lm::Config &expected,
    uint32_t expectedSeed) {
  LoadedNprtCheckpoint result;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("NPRT_CKPT_OPEN_FAILED");
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0 || static_cast<std::uint64_t>(size) > kNprtMaxCheckpointBytes)
    throw std::runtime_error("NPRT_CKPT_SIZE_INVALID");
  result.fileBytes = static_cast<std::uint64_t>(size);
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTCKPTV1\n") throw std::runtime_error("NPRT_CKPT_MAGIC");
  result.vocabulary = nprtReadU32(input);
  result.tokens = nprtReadU32(input);
  result.dimension = nprtReadU32(input);
  result.feedForward = nprtReadU32(input);
  result.layers = nprtReadU32(input);
  result.heads = nprtReadU32(input);
  result.seed = nprtReadU32(input);
  result.step = nprtReadU32(input);
  result.registryCount = nprtReadU32(input);
  if (result.vocabulary != expected.vocabularySize ||
      result.tokens != expected.tokens ||
      result.dimension != expected.dimension ||
      result.feedForward != expected.feedForwardDimension ||
      result.layers != expected.numLayers || result.heads != expected.numHeads)
    throw std::runtime_error("NPRT_CKPT_CONFIG_MISMATCH");
  if (result.seed != expectedSeed)
    throw std::runtime_error("NPRT_CKPT_SEED_MISMATCH");
  if (result.step == 0 || result.step >= 1000000)
    throw std::runtime_error("NPRT_CKPT_STEP_INVALID");
  // The trained checkpoint stores registry counts derived from fully-shaped
  // parameters (vocab*dimension embedding, dimension^2 projections, ...).
  // Reference only the shapes (values are discarded) so element counts match
  // whatever the training milestone wrote.
  const Params shape = tiny_lm::initialParameters(expected, 1);
  const auto registry = tiny_lm::parameterRegistry(shape);
  if (registry.size() != result.registryCount)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_COUNT_MISMATCH");
  Params parameters;
  parameters.layers.resize(result.layers > 0 ? result.layers - 1 : 0);
  std::uint64_t elements = 0;
  for (size_t i = 0; i < registry.size(); ++i) {
    const uint32_t nameLength = nprtReadU32(input);
    if (nameLength == 0 || nameLength > 256)
      throw std::runtime_error("NPRT_CKPT_NAME_LENGTH");
    std::string name(nameLength, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name.size()));
    if (!input) throw std::runtime_error("NPRT_CKPT_NAME_TRUNCATED");
    const std::uint64_t count = nprtReadU64(input);
    if (name != registry[i].name)
      throw std::runtime_error("NPRT_CKPT_REGISTRY_ORDER_MISMATCH");
    const std::uint64_t expectedCount = registry[i].values->size();
    if (count != expectedCount || count > 100000000ull)
      throw std::runtime_error("NPRT_CKPT_ELEMENT_COUNT_MISMATCH");
    std::vector<float> values(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (!input) throw std::runtime_error("NPRT_CKPT_VALUES_TRUNCATED");
    for (float value : values)
      if (!std::isfinite(value)) result.finite = false;
    nprtAssignRegistryMember(parameters, result.layers, name,
                             std::move(values));
    elements += count;
  }
  char tail = 0;
  if (input.read(&tail, 1))
    throw std::runtime_error("NPRT_CKPT_TRAILING_BYTES");
  result.parameterElements = elements;
  result.parameters = std::move(parameters);
  result.parameterHash = nprtParameterHash(result.parameters);
  return result;
}

double nprtLastRowMargin(const std::vector<float> &logits, uint32_t rows,
                         uint32_t vocab) {
  if (rows == 0 || vocab == 0 || logits.size() < size_t(rows) * vocab)
    return std::numeric_limits<double>::quiet_NaN();
  const size_t base = size_t(rows - 1) * vocab;
  float first = -std::numeric_limits<float>::infinity();
  float second = -std::numeric_limits<float>::infinity();
  for (uint32_t j = 0; j < vocab; ++j) {
    const float value = logits[base + j];
    if (value > first) {
      second = first;
      first = value;
    } else if (value > second) {
      second = value;
    }
  }
  if (!std::isfinite(first) || !std::isfinite(second))
    return std::numeric_limits<double>::quiet_NaN();
  return double(first) - double(second);
}

// Statistics of the last-token logits row (the row used for generation).
// These are recorded for parity auditing: they let the host distinguish an
// absolute-error growth caused by logit-scale inflation from a genuine HTP
// numerical degradation.
struct NprtLastRowStats {
  double cpuMin = 0, cpuMax = 0, cpuRms = 0, cpuStd = 0;
  double htpMin = 0, htpMax = 0, htpRms = 0, htpStd = 0;
};

NprtLastRowStats nprtLastRowLogitStats(const std::vector<float> &cpuLogits,
                                       const std::vector<float> &htpLogits,
                                       uint32_t tokens, uint32_t vocab) {
  NprtLastRowStats stats;
  const size_t base = size_t(tokens - 1) * vocab;
  auto rowStats = [&](const std::vector<float> &row, double &minV,
                      double &maxV, double &rms, double &stdv) {
    minV = maxV = row[base];
    double sum = 0.0, sumSq = 0.0;
    for (uint32_t i = 0; i < vocab; ++i) {
      const double v = row[base + i];
      minV = std::min(minV, v);
      maxV = std::max(maxV, v);
      sum += v;
      sumSq += v * v;
    }
    rms = std::sqrt(sumSq / std::max(1u, vocab));
    const double mean = sum / std::max(1u, vocab);
    double var = 0.0;
    for (uint32_t i = 0; i < vocab; ++i) {
      const double d = row[base + i] - mean;
      var += d * d;
    }
    stdv = std::sqrt(var / std::max(1u, vocab));
  };
  rowStats(cpuLogits, stats.cpuMin, stats.cpuMax, stats.cpuRms, stats.cpuStd);
  rowStats(htpLogits, stats.htpMin, stats.htpMax, stats.htpRms, stats.htpStd);
  return stats;
}

// Top-K index-set overlap (Jaccard-style count / K) on the last logits row.
uint32_t nprtLastRowTopKOverlap(const std::vector<float> &cpuLogits,
                                const std::vector<float> &htpLogits,
                                uint32_t tokens, uint32_t vocab, uint32_t k) {
  const size_t base = size_t(tokens - 1) * vocab;
  std::vector<uint32_t> cpuIdx(vocab), htpIdx(vocab);
  for (uint32_t i = 0; i < vocab; ++i) {
    cpuIdx[i] = i;
    htpIdx[i] = i;
  }
  auto byLogitDesc = [&](const std::vector<float> &row) {
    return [&row, base](uint32_t a, uint32_t b) {
      return row[base + a] > row[base + b] ||
             (row[base + a] == row[base + b] && a < b);
    };
  };
  std::sort(cpuIdx.begin(), cpuIdx.end(), byLogitDesc(cpuLogits));
  std::sort(htpIdx.begin(), htpIdx.end(), byLogitDesc(htpLogits));
  const uint32_t kk = std::min(k, vocab);
  std::vector<uint32_t> cpuTop(cpuIdx.begin(), cpuIdx.begin() + kk);
  std::vector<uint32_t> htpTop(htpIdx.begin(), htpIdx.begin() + kk);
  std::sort(cpuTop.begin(), cpuTop.end());
  std::sort(htpTop.begin(), htpTop.end());
  std::vector<uint32_t> intersection;
  std::set_intersection(cpuTop.begin(), cpuTop.end(), htpTop.begin(),
                        htpTop.end(), std::back_inserter(intersection));
  return static_cast<uint32_t>(intersection.size());
}

// Margin between the top logit and the K-th ranked logit on the last row.
double nprtLastRowMarginAtK(const std::vector<float> &logits, uint32_t tokens,
                            uint32_t vocab, uint32_t rank) {
  const size_t base = size_t(tokens - 1) * vocab;
  std::vector<uint32_t> idx(vocab);
  for (uint32_t i = 0; i < vocab; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
    return logits[base + a] > logits[base + b] ||
           (logits[base + a] == logits[base + b] && a < b);
  });
  const uint32_t r = std::min(rank, vocab - 1);
  return double(logits[base + idx[0]]) - double(logits[base + idx[r]]);
}

inline nicopedia_gen::NprtTapMetric computeNprtTapMetric(
    const std::vector<float> &cpu,
    const std::vector<float> &htp,
    const std::string &name) {
  nicopedia_gen::NprtTapMetric m;
  m.name = name;
  if (cpu.size() != htp.size() || cpu.empty()) return m;
  double cpuSumSq = 0, htpSumSq = 0, diffSumSq = 0, maxAbs = 0;
  double dot = 0, cpuLenSq = 0, htpLenSq = 0;
  double cpuSum = 0, htpSum = 0;
  const size_t n = cpu.size();
  for (size_t i = 0; i < n; ++i) {
    const double c = cpu[i];
    const double h = htp[i];
    const double d = std::abs(c - h);
    maxAbs = std::max(maxAbs, d);
    cpuSumSq += c * c;
    htpSumSq += h * h;
    diffSumSq += (c - h) * (c - h);
    dot += c * h;
    cpuLenSq += c * c;
    htpLenSq += h * h;
    cpuSum += c;
    htpSum += h;
  }
  m.cpuRms = std::sqrt(cpuSumSq / double(n));
  m.htpRms = std::sqrt(htpSumSq / double(n));
  m.diffMaxAbs = maxAbs;
  m.diffRms = std::sqrt(diffSumSq / double(n));
  m.relRms = m.diffRms / std::max(m.cpuRms, 1.0e-6);
  const double denom = std::sqrt(cpuLenSq * htpLenSq);
  m.cosine = denom > 0 ? dot / denom : 1.0;
  const double cpuMean = cpuSum / double(n);
  const double htpMean = htpSum / double(n);
  double centeredDiffSumSq = 0;
  for (size_t i = 0; i < n; ++i) {
    const double cd = (cpu[i] - cpuMean) - (htp[i] - htpMean);
    centeredDiffSumSq += cd * cd;
  }
  m.centeredRms = std::sqrt(centeredDiffSumSq / double(n));
  return m;
}

struct NprtParityRow {
  std::string label;
  uint32_t contextBytes = 0;
  uint32_t padBytes = 0;
  double logitsMaxAbs = 0;
  double logitsMeanAbs = 0;
  double logitsL2 = 0;
  double logitsMaxRelative = 0;
  double logitsCosine = 1.0;
  double logitsCpuMin = 0, logitsCpuMax = 0;
  double logitsCpuRms = 0, logitsCpuStd = 0;
  double logitsHtpMin = 0, logitsHtpMax = 0;
  double logitsHtpRms = 0, logitsHtpStd = 0;
  double probabilityMaxAbs = 0;
  double probabilityMeanAbs = 0;
  uint32_t lastArgmaxCpu = 0, lastArgmaxHtp = 0;
  bool lastArgmaxMatch = true;
  double top1MarginCpu = 0, top1MarginHtp = 0;
  double top2MarginCpu = 0, top2MarginHtp = 0;
  uint32_t topkSetOverlap = 0;
  uint32_t topkSetSize = 5;
  bool finite = true;
  // Parity re-audit (protocol docs/qnn-nicopedia-htp-parity-policy.md):
  // candidate policy verdicts + row metrics in shadow mode on the same rows.
  nicopedia_gen::ParityPolicies policies;
  nicopedia_gen::ParityRowMetrics metrics;
  bool candidateOk = true;
  std::vector<nicopedia_gen::NprtTapMetric> tapMetrics;
  std::string htpLogitsSha256;
};

struct NprtArRow {
  uint32_t step = 0;
  uint32_t argmaxCpu = 0, argmaxHtp = 0;
  bool match = true;
  bool contextAligned = true;
  double maxAbsLogits = -1;
  double logitsMeanAbs = 0;
  double logitsL2 = 0;
  double logitsMaxRelative = 0;
  double logitsCosine = 1.0;
  double logitsCpuMin = 0, logitsCpuMax = 0;
  double logitsCpuRms = 0, logitsCpuStd = 0;
  double logitsHtpMin = 0, logitsHtpMax = 0;
  double logitsHtpRms = 0, logitsHtpStd = 0;
  double probabilityMeanAbs = 0;
  double marginCpu = 0, marginHtp = 0;
  double top1MarginCpu = 0, top1MarginHtp = 0;
  double top2MarginCpu = 0, top2MarginHtp = 0;
  uint32_t topkSetOverlap = 0;
  uint32_t topkSetSize = 5;
  bool finite = true;
  // Parity re-audit shadow verdicts (protocol section 4).
  nicopedia_gen::ParityPolicies policies;
  nicopedia_gen::ParityRowMetrics metrics;
  bool candidateOk = true;
};

}  // namespace

std::string nicopediaHtpGeneration(
    const tiny_lm::Config &config, const std::string &checkpointPath,
    const std::string &promptPath, const TrainingConfig &trainingConfig,
    const nicopedia_gen::GenerateConfig &generateConfig,
    const LogSink &progress) {
  const int seed = trainingConfig.seed > 0 && trainingConfig.seed < 100000
                       ? static_cast<int>(trainingConfig.seed)
                       : 1;
  const uint32_t layers = config.numLayers;
  const uint32_t heads = config.numHeads;
  if (layers == 0 || heads == 0)
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=layers_heads_must_be_positive\n";
  if (generateConfig.maxNewBytes == 0 || generateConfig.maxNewBytes > 1024)
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=max_new_bytes_must_be_1_1024\n";
  if (!generateConfig.greedy &&
      (!(generateConfig.temperature > 0.0f) ||
       !std::isfinite(generateConfig.temperature) ||
       generateConfig.topK == 0 || generateConfig.topK > 256))
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=temperature_or_top_k_invalid\n";
  if (generateConfig.gatePolicy != "legacy" && generateConfig.gatePolicy != "candidate")
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=gate_policy must be legacy or candidate\n";
  uint32_t expectedStep = 0;
  if (!nprtParseCheckpointStep(checkpointPath, static_cast<uint32_t>(seed),
                               layers, &expectedStep))
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_FILENAME\n"
           "error=checkpoint filename must be htp-seed<seed>-l<layers>-step<step>.ckpt\n";
  LoadedNprtCheckpoint loaded;
  try {
    loaded = nprtLoadCheckpointForGeneration(checkpointPath, config,
                                             static_cast<uint32_t>(seed));
  } catch (const std::exception &exception) {
    return std::string("NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
                       "failure_classification=CHECKPOINT_DECODE\nerror=") +
           exception.what() + '\n';
  }
  if (loaded.step != expectedStep)
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_IDENTITY\n"
           "error=checkpoint step does not match filename\n";
  if (!loaded.finite)
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_NONFINITE\n"
           "error=checkpoint contains non-finite values\n";

  std::vector<std::uint8_t> prompt;
  try {
    prompt = nprtReadFileBytes(promptPath, 16u * 1024u * 1024u);
  } catch (const std::exception &exception) {
    return std::string("NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
                       "failure_classification=PROMPT_READ\nerror=") +
           exception.what() + '\n';
  }
  if (prompt.empty())
    return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           "failure_classification=PROMPT_EMPTY\n"
           "error=prompt file is empty\n";
  const bool promptTruncated = prompt.size() > config.tokens;
  uint32_t contextPad = 0;
  std::vector<std::uint8_t> context =
      nicopedia_gen::buildGenerationContext(prompt, config.tokens, &contextPad);

  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  const auto initStarted = std::chrono::steady_clock::now();
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL,
          TinyTransformerTrainingTapSet::NONE, layers, heads))
    return failure("nicopedia_generate_prepare", error, runtime);
  const double initializeUs =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - initStarted)
          .count();
  const double graphCreateUs = runtime.metrics().graphCreateUs;
  const double graphFinalizeUs = runtime.metrics().graphFinalizeUs;

  // Inference-only forward: zero target (logits are target-independent).
  const std::vector<float> zeros(
      size_t(config.tokens) * config.vocabularySize, 0.0f);
  const auto windowInput = [&](const std::vector<std::uint8_t> &window) {
    std::vector<uint32_t> tokens;
    tokens.reserve(window.size());
    for (uint8_t byte : window) tokens.push_back(byte);
    return tiny_lm::oneHot(tokens, config.vocabularySize);
  };

  // Fixed-prefix CPU/HTP parity on the shared deterministic prefixes.
  std::vector<NprtParityRow> parityRows;
  const auto &prefixes = nicopedia_gen::parityPrefixes();
  bool parityGate = true;          // legacy gate (unchanged semantics)
  bool parityGateCandidate = true; // candidate full policy F (protocol)
  for (const auto &prefix : prefixes) {
    uint32_t pad = 0;
    const auto prefixContext =
        nicopedia_gen::buildGenerationContext(prefix.bytes, config.tokens, &pad);
    const auto cpuStep = tiny_lm::forwardBackwardGeneralized(
        config, windowInput(prefixContext), zeros, loaded.parameters, 0.0f);
    TinyTransformerTrainingOutputs htpStep;
    if (!runtime.executeTinyTransformerTraining(
            windowInput(prefixContext), zeros, loaded.parameters, 0.0f,
            htpStep, error))
      return failure("nicopedia_generate_parity", error, runtime);
    const auto logitsError = compareNprt(cpuStep.logits, htpStep.logits);
    const auto probabilityError =
        compareNprt(cpuStep.probabilities, htpStep.probabilities);
    NprtParityRow row;
    row.label = prefix.label;
    row.contextBytes = static_cast<uint32_t>(prefixContext.size());
    row.padBytes = pad;
    row.logitsMaxAbs = logitsError.maxAbs;
    row.logitsMeanAbs = logitsError.meanAbs;
    row.logitsL2 = logitsError.l2;
    row.logitsMaxRelative = logitsError.maxRelative;
    row.logitsCosine = logitsError.cosine;
    row.probabilityMaxAbs = probabilityError.maxAbs;
    row.probabilityMeanAbs = probabilityError.meanAbs;
    const size_t lastBase = size_t(config.tokens - 1) * config.vocabularySize;
    row.lastArgmaxCpu = nicopedia_gen::greedyArgmax(
        cpuStep.logits.data() + lastBase, config.vocabularySize);
    row.lastArgmaxHtp = nicopedia_gen::greedyArgmax(
        htpStep.logits.data() + lastBase, config.vocabularySize);
    row.lastArgmaxMatch = row.lastArgmaxCpu == row.lastArgmaxHtp;
    const auto lastStats = nprtLastRowLogitStats(
        cpuStep.logits, htpStep.logits, config.tokens, config.vocabularySize);
    row.logitsCpuMin = lastStats.cpuMin;
    row.logitsCpuMax = lastStats.cpuMax;
    row.logitsCpuRms = lastStats.cpuRms;
    row.logitsCpuStd = lastStats.cpuStd;
    row.logitsHtpMin = lastStats.htpMin;
    row.logitsHtpMax = lastStats.htpMax;
    row.logitsHtpRms = lastStats.htpRms;
    row.logitsHtpStd = lastStats.htpStd;
    row.top1MarginCpu = nprtLastRowMarginAtK(
        cpuStep.logits, config.tokens, config.vocabularySize, 1);
    row.top1MarginHtp = nprtLastRowMarginAtK(
        htpStep.logits, config.tokens, config.vocabularySize, 1);
    row.top2MarginCpu = nprtLastRowMarginAtK(
        cpuStep.logits, config.tokens, config.vocabularySize, 2);
    row.top2MarginHtp = nprtLastRowMarginAtK(
        htpStep.logits, config.tokens, config.vocabularySize, 2);
    row.topkSetOverlap = nprtLastRowTopKOverlap(
        cpuStep.logits, htpStep.logits, config.tokens, config.vocabularySize,
        row.topkSetSize);
    row.finite = logitsError.nonfinite == 0 && probabilityError.nonfinite == 0;
    // Fixed one-step tolerances established by the HTP training milestone.
    const bool ok = row.finite && row.logitsMaxAbs < 2e-2 &&
                    row.probabilityMaxAbs < 5e-3;
    parityGate = parityGate && ok;
    // Candidate policy F (shadow parity re-audit; protocol
    // docs/qnn-nicopedia-htp-parity-policy.md section 4) on the same rows.
    row.metrics = nicopedia_gen::computeParityRowMetrics(
        cpuStep.logits.data() + lastBase,
        htpStep.logits.data() + lastBase, config.vocabularySize);
    row.policies = nicopedia_gen::evaluateParityPolicies(row.metrics);
    row.candidateOk = row.policies.full;
    parityGateCandidate = parityGateCandidate && row.candidateOk;
    parityRows.push_back(row);
  }

  // Autoregressive CPU/HTP parity: 8 greedy bytes from the first prefix
  // context.  A byte-level divergence inside the fixed tolerance and with a
  // decisive margin is the "large mismatch" the host runner must not pass;
  // a divergence under a low CPU margin is numerical noise, recorded only.
  const uint32_t kArSteps = 8;
  std::vector<NprtArRow> arRows;
  std::vector<std::uint8_t> cpuContext, htpContext;
  {
    uint32_t pad = 0;
    cpuContext =
        nicopedia_gen::buildGenerationContext(prefixes[0].bytes, config.tokens, &pad);
    htpContext = cpuContext;
  }
  bool contextsAligned = true;
  bool hasDivergence = false;
  uint32_t divergenceStep = 0;
  double divergenceMarginCpu = 0;
  bool arGate = true;
  bool arGateCandidate = true;
  for (uint32_t step = 0; step < kArSteps; ++step) {
    const auto cpuStep = tiny_lm::forwardBackwardGeneralized(
        config, windowInput(cpuContext), zeros, loaded.parameters, 0.0f);
    TinyTransformerTrainingOutputs htpStep;
    if (!runtime.executeTinyTransformerTraining(
            windowInput(htpContext), zeros, loaded.parameters, 0.0f, htpStep,
            error))
      return failure("nicopedia_generate_ar", error, runtime);
    NprtArRow row;
    row.step = step;
    const size_t lastBase = size_t(config.tokens - 1) * config.vocabularySize;
    row.argmaxCpu = nicopedia_gen::greedyArgmax(
        cpuStep.logits.data() + lastBase, config.vocabularySize);
    row.argmaxHtp = nicopedia_gen::greedyArgmax(
        htpStep.logits.data() + lastBase, config.vocabularySize);
    row.marginCpu =
        nprtLastRowMargin(cpuStep.logits, config.tokens, config.vocabularySize);
    row.marginHtp =
        nprtLastRowMargin(htpStep.logits, config.tokens, config.vocabularySize);
    row.top1MarginCpu = row.marginCpu;
    row.top1MarginHtp = row.marginHtp;
    row.top2MarginCpu = nprtLastRowMarginAtK(
        cpuStep.logits, config.tokens, config.vocabularySize, 2);
    row.top2MarginHtp = nprtLastRowMarginAtK(
        htpStep.logits, config.tokens, config.vocabularySize, 2);
    row.contextAligned = contextsAligned;
    row.match = contextsAligned && row.argmaxCpu == row.argmaxHtp;
    if (contextsAligned) {
      const auto logitsError = compareNprt(cpuStep.logits, htpStep.logits);
      const auto probabilityError =
          compareNprt(cpuStep.probabilities, htpStep.probabilities);
      row.maxAbsLogits = logitsError.maxAbs;
      row.logitsMeanAbs = logitsError.meanAbs;
      row.logitsL2 = logitsError.l2;
      row.logitsMaxRelative = logitsError.maxRelative;
      row.logitsCosine = logitsError.cosine;
      row.probabilityMeanAbs = probabilityError.meanAbs;
      const auto lastStats = nprtLastRowLogitStats(
          cpuStep.logits, htpStep.logits, config.tokens, config.vocabularySize);
      row.logitsCpuMin = lastStats.cpuMin;
      row.logitsCpuMax = lastStats.cpuMax;
      row.logitsCpuRms = lastStats.cpuRms;
      row.logitsCpuStd = lastStats.cpuStd;
      row.logitsHtpMin = lastStats.htpMin;
      row.logitsHtpMax = lastStats.htpMax;
      row.logitsHtpRms = lastStats.htpRms;
      row.logitsHtpStd = lastStats.htpStd;
      row.topkSetOverlap = nprtLastRowTopKOverlap(
          cpuStep.logits, htpStep.logits, config.tokens, config.vocabularySize,
          row.topkSetSize);
      row.finite = logitsError.nonfinite == 0 && probabilityError.nonfinite == 0;
      const bool withinTolerance =
          row.finite && row.maxAbsLogits < 2e-2 &&
          probabilityError.maxAbs < 5e-3;
      arGate = arGate && withinTolerance;
      // Candidate policy F per AR step (protocol section 4), shadow verdict.
      row.metrics = nicopedia_gen::computeParityRowMetrics(
          cpuStep.logits.data() + lastBase,
          htpStep.logits.data() + lastBase, config.vocabularySize);
      row.policies = nicopedia_gen::evaluateParityPolicies(row.metrics);
      row.candidateOk = row.policies.full;
      arGateCandidate = arGateCandidate && row.candidateOk;
      if (row.argmaxCpu != row.argmaxHtp) {
        hasDivergence = true;
        divergenceStep = step;
        divergenceMarginCpu = row.marginCpu;
        contextsAligned = false;
      }
    } else {
      row.maxAbsLogits = -1;
      row.finite = true;
      row.policies.legacy = false;
      row.policies.full = false;
      row.candidateOk = false;
      arGateCandidate = false;
    }
    arRows.push_back(row);
    nicopedia_gen::appendByteWindow(cpuContext,
                                    static_cast<uint8_t>(row.argmaxCpu));
    nicopedia_gen::appendByteWindow(htpContext,
                                    static_cast<uint8_t>(row.argmaxHtp));
  }
  const bool divergenceBlocked =
      hasDivergence && divergenceMarginCpu > 1e-2;
  arGate = arGate && !divergenceBlocked;
  const bool legacyGate = parityGate && arGate;
  const bool candidateGate = parityGateCandidate && arGateCandidate;
  const bool generationGate =
      (generateConfig.gatePolicy == "candidate") ? candidateGate : legacyGate;

  // User-prompt generation (only when both gates pass).
  std::vector<std::uint8_t> generated;
  double generateSeconds = 0;
  if (generationGate) {
    const auto generateStarted = std::chrono::steady_clock::now();
    auto generateContext = context;
    for (uint32_t step = 0; step < generateConfig.maxNewBytes; ++step) {
      TinyTransformerTrainingOutputs htpStep;
      if (!runtime.executeTinyTransformerTraining(
              windowInput(generateContext), zeros, loaded.parameters, 0.0f,
              htpStep, error))
        return failure("nicopedia_generate_step", error, runtime);
      const size_t lastBase =
          size_t(config.tokens - 1) * config.vocabularySize;
      const float *row = htpStep.logits.data() + lastBase;
      bool rowFinite = true;
      for (uint32_t j = 0; j < config.vocabularySize; ++j)
        rowFinite = rowFinite && std::isfinite(row[j]);
      if (!rowFinite)
        return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
               "failure_classification=EXECUTION_NONFINITE\n"
               "error=generation step produced non-finite logits\n" +
               runtime.apiTraceSummary() + runtime.diagnostics();
      const uint8_t byte =
          generateConfig.greedy
              ? nicopedia_gen::greedyArgmax(row, config.vocabularySize)
              : nicopedia_gen::sampleTopK(row, config.vocabularySize,
                                          generateConfig.temperature,
                                          generateConfig.topK,
                                          generateConfig.samplingSeed, step);
      generated.push_back(byte);
      nicopedia_gen::appendByteWindow(generateContext, byte);
      if (progress &&
          (step == 0 || (step + 1) % 16 == 0 ||
           step + 1 == generateConfig.maxNewBytes)) {
        std::ostringstream update;
        update << "phase=generate\nbyte=" << (step + 1) << "/"
               << generateConfig.maxNewBytes;
        progress(update.str());
      }
    }
    generateSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      generateStarted)
            .count();
  }
  const nicopedia_gen::Utf8Stats generatedStats =
      nicopedia_gen::utf8StatsOf(generated);
  const nicopedia_gen::GenerationAggregates ag =
      nicopedia_gen::generationAggregates(generated);
  // Row-level logit finiteness is enforced fail-closed inside the generation
  // loop (any non-finite logit aborts with EXECUTION_NONFINITE before the
  // report is produced), so on SUCCESS these flags can only reflect a
  // non-finite host-side timing measurement.  They are published as
  // final-gate booleans, not per-step diagnostics.
  const bool nanDetected = !std::isfinite(generateSeconds);
  const bool infDetected = !std::isfinite(generateSeconds);
  const std::string status = generationGate ? "SUCCESS" : "FAILED";

  std::ostringstream report;
  report << std::setprecision(10)
         << "NICOPEDIA_HTP_GENERATION\ntest=nicopedia_htp_generation\nstatus="
         << status
         << (generationGate
                 ? ""
                 : "\nfailure_classification=PARITY_GATE_REJECTED")
         << "\nmodel=L" << layers << "\nlayers=" << layers << "\nheads=" << heads
         << "\nseed=" << seed << "\ncheckpoint_step=" << loaded.step
         << "\ncheckpoint_parameter_hash=" << loaded.parameterHash
         << "\ncheckpoint_parameter_elements=" << loaded.parameterElements
         << "\ncheckpoint_file_bytes=" << loaded.fileBytes
         << "\ncheckpoint_finite=" << (loaded.finite ? "true" : "false")
         << "\ncheckpoint_header_vocabulary=" << loaded.vocabulary
         << "\ncheckpoint_header_tokens=" << loaded.tokens
         << "\ncheckpoint_header_dimension=" << loaded.dimension
         << "\ncheckpoint_header_feedforward=" << loaded.feedForward
         << "\ncheckpoint_header_layers=" << loaded.layers
         << "\ncheckpoint_header_heads=" << loaded.heads
         << "\ncheckpoint_header_seed=" << loaded.seed
         << "\ncheckpoint_header_step=" << loaded.step
         << "\ncheckpoint_header_registry_count=" << loaded.registryCount
         << "\nprompt_byte_count=" << prompt.size()
         << "\nprompt_truncated=" << (promptTruncated ? "true" : "false")
         << "\ncontext_used_bytes=" << context.size()
         << "\ncontext_padding_bytes=" << contextPad
         << "\ngenerate_mode=" << (generateConfig.greedy ? "greedy" : "sample")
         << "\ntemperature=" << generateConfig.temperature
         << "\ntop_k=" << generateConfig.topK
         << "\nsampling_seed=" << generateConfig.samplingSeed
         << "\nmax_new_bytes=" << generateConfig.maxNewBytes
         << "\ngenerated_byte_count=" << generated.size()
         << "\ngenerated_valid_utf8_bytes=" << generatedStats.validBytes
         << "\ngenerated_invalid_utf8_bytes=" << generatedStats.invalidBytes
         << "\nunique_byte_values=" << ag.uniqueByteValues
         << "\nascii_bytes=" << ag.asciiBytes
         << "\nmax_same_byte_run=" << ag.maxSameByteRun
         << "\nmax_scalar_repeat_run=" << ag.maxScalarRepeatRun
         << "\nshort_period_loop_fraction=" << ag.shortPeriodLoopFraction
         << "\ngenerated_hex=" << nicopedia_gen::bytesToHex(generated)
         << "\nparity_prefix_count=" << parityRows.size()
         << "\nparity_gate=" << (parityGate ? "true" : "false")
         << "\nparity_gate_candidate=" << (parityGateCandidate ? "true" : "false")
         << "\nar_steps=" << kArSteps
         << "\nar_gate=" << (arGate ? "true" : "false")
         << "\nar_gate_candidate=" << (arGateCandidate ? "true" : "false")
         << "\ngate_policy=" << generateConfig.gatePolicy
         << "\nar_divergence_step="
         << (hasDivergence ? static_cast<int>(divergenceStep) : 0)
         << "\nar_divergence_margin_cpu=" << divergenceMarginCpu
         << "\nar_divergence_blocked=" << (divergenceBlocked ? "true" : "false")
         << "\ngeneration_gate=" << (generationGate ? "true" : "false");
  for (size_t i = 0; i < parityRows.size(); ++i) {
    const auto &row = parityRows[i];
    report << "\nparity_" << i << "_label=" << row.label
           << "\nparity_" << i << "_context_bytes=" << row.contextBytes
           << "\nparity_" << i << "_pad_bytes=" << row.padBytes
           << "\nparity_" << i << "_logits_max_abs_error=" << row.logitsMaxAbs
           << "\nparity_" << i << "_logits_mean_abs_error=" << row.logitsMeanAbs
           << "\nparity_" << i << "_logits_rms_error=" << row.logitsL2
           << "\nparity_" << i << "_logits_max_relative_error=" << row.logitsMaxRelative
           << "\nparity_" << i << "_logits_cosine_similarity=" << row.logitsCosine
           << "\nparity_" << i << "_logits_cpu_min=" << row.logitsCpuMin
           << "\nparity_" << i << "_logits_cpu_max=" << row.logitsCpuMax
           << "\nparity_" << i << "_logits_cpu_rms=" << row.logitsCpuRms
           << "\nparity_" << i << "_logits_cpu_std=" << row.logitsCpuStd
           << "\nparity_" << i << "_logits_htp_min=" << row.logitsHtpMin
           << "\nparity_" << i << "_logits_htp_max=" << row.logitsHtpMax
           << "\nparity_" << i << "_logits_htp_rms=" << row.logitsHtpRms
           << "\nparity_" << i << "_logits_htp_std=" << row.logitsHtpStd
           << "\nparity_" << i << "_probability_max_abs_error="
           << row.probabilityMaxAbs
           << "\nparity_" << i << "_probability_mean_abs_error=" << row.probabilityMeanAbs
           << "\nparity_" << i << "_last_argmax_cpu=" << row.lastArgmaxCpu
           << "\nparity_" << i << "_last_argmax_htp=" << row.lastArgmaxHtp
           << "\nparity_" << i << "_last_argmax_match="
           << (row.lastArgmaxMatch ? "true" : "false")
           << "\nparity_" << i << "_top1_margin_cpu=" << row.top1MarginCpu
           << "\nparity_" << i << "_top1_margin_htp=" << row.top1MarginHtp
           << "\nparity_" << i << "_top2_margin_cpu=" << row.top2MarginCpu
           << "\nparity_" << i << "_top2_margin_htp=" << row.top2MarginHtp
           << "\nparity_" << i << "_topk_set_overlap=" << row.topkSetOverlap
           << "\nparity_" << i << "_topk_set_size=" << row.topkSetSize
           << "\nparity_" << i << "_finite=" << (row.finite ? "true" : "false")
           << "\nparity_" << i << "_delta_mean=" << row.metrics.deltaMean
           << "\nparity_" << i << "_delta_median=" << row.metrics.deltaMedian
           << "\nparity_" << i << "_delta_std=" << row.metrics.deltaStd
           << "\nparity_" << i << "_centered_max_abs=" << row.metrics.centeredMaxAbs
           << "\nparity_" << i << "_centered_rms=" << row.metrics.centeredRms
           << "\nparity_" << i << "_logsoftmax_max_abs=" << row.metrics.logSoftmaxMaxAbs
           << "\nparity_" << i << "_logsoftmax_rms=" << row.metrics.logSoftmaxRms
           << "\nparity_" << i << "_probability_l1_error=" << row.metrics.probL1
           << "\nparity_" << i << "_probability_js_divergence=" << row.metrics.jsDivergence
           << "\nparity_" << i << "_logits_cosine_centered=" << row.metrics.cosineCentered
           << "\nparity_" << i << "_scale_ratio=" << row.metrics.scaleRatio
           << "\nparity_" << i << "_relative_max_error=" << row.metrics.relMax
           << "\nparity_" << i << "_decision_ambiguous="
           << (row.metrics.decisionAmbiguous ? "true" : "false")
           << "\nparity_" << i << "_row_degenerate="
           << (row.metrics.rowDegenerate ? "true" : "false")
           << "\nparity_" << i << "_topk_order_match="
           << (row.metrics.topkOrderMatch ? "true" : "false")
           << "\nparity_" << i << "_candidate_legacy="
           << (row.policies.legacy ? "true" : "false")
           << "\nparity_" << i << "_candidate_prob="
           << (row.policies.prob ? "true" : "false")
           << "\nparity_" << i << "_candidate_shape="
           << (row.policies.shape ? "true" : "false")
           << "\nparity_" << i << "_candidate_decision="
           << (row.policies.decision ? "true" : "false")
           << "\nparity_" << i << "_candidate_full="
           << (row.policies.full ? "true" : "false");
  }
  for (size_t i = 0; i < arRows.size(); ++i) {
    const auto &row = arRows[i];
    report << "\nar_step_" << row.step << "_argmax_cpu=" << row.argmaxCpu
           << "\nar_step_" << row.step << "_argmax_htp=" << row.argmaxHtp
           << "\nar_step_" << row.step << "_match="
           << (row.match ? "true" : "false")
           << "\nar_step_" << row.step << "_context_aligned="
           << (row.contextAligned ? "true" : "false")
           << "\nar_step_" << row.step << "_max_abs_logits_error="
           << row.maxAbsLogits << "\nar_step_" << row.step
           << "_logits_mean_abs_error=" << row.logitsMeanAbs << "\nar_step_"
           << row.step << "_logits_rms_error=" << row.logitsL2 << "\nar_step_"
           << row.step << "_logits_max_relative_error=" << row.logitsMaxRelative
           << "\nar_step_" << row.step << "_logits_cosine_similarity="
           << row.logitsCosine << "\nar_step_" << row.step
           << "_logits_cpu_min=" << row.logitsCpuMin << "\nar_step_" << row.step
           << "_logits_cpu_max=" << row.logitsCpuMax << "\nar_step_" << row.step
           << "_logits_cpu_rms=" << row.logitsCpuRms << "\nar_step_" << row.step
           << "_logits_cpu_std=" << row.logitsCpuStd << "\nar_step_" << row.step
           << "_logits_htp_min=" << row.logitsHtpMin << "\nar_step_" << row.step
           << "_logits_htp_max=" << row.logitsHtpMax << "\nar_step_" << row.step
           << "_logits_htp_rms=" << row.logitsHtpRms << "\nar_step_" << row.step
           << "_logits_htp_std=" << row.logitsHtpStd << "\nar_step_" << row.step
           << "_probability_mean_abs_error=" << row.probabilityMeanAbs
           << "\nar_step_" << row.step << "_margin_cpu=" << row.marginCpu
           << "\nar_step_" << row.step << "_margin_htp=" << row.marginHtp
           << "\nar_step_" << row.step << "_top1_margin_cpu=" << row.top1MarginCpu
           << "\nar_step_" << row.step << "_top1_margin_htp=" << row.top1MarginHtp
           << "\nar_step_" << row.step << "_top2_margin_cpu=" << row.top2MarginCpu
           << "\nar_step_" << row.step << "_top2_margin_htp=" << row.top2MarginHtp
           << "\nar_step_" << row.step << "_topk_set_overlap=" << row.topkSetOverlap
           << "\nar_step_" << row.step << "_topk_set_size=" << row.topkSetSize
           << "\nar_step_" << row.step
           << "_finite=" << (row.finite ? "true" : "false")
           << "\nar_step_" << row.step << "_candidate_full="
           << (row.policies.full ? "true" : "false")
           << "\nar_step_" << row.step << "_candidate_decision="
           << (row.policies.decision ? "true" : "false");
  }
  report << "\nhtp_initialize_us=" << initializeUs
         << "\ngraph_create_us=" << graphCreateUs
         << "\ngraph_finalize_us=" << graphFinalizeUs
         << "\ngeneration_total_seconds=" << generateSeconds
         << "\ngeneration_ms_per_byte="
         << (generated.empty()
                 ? 0.0
                 : generateSeconds / generated.size() * 1000.0)
         << "\ngraph_execute_count=" << runtime.metrics().graphExecuteCount
          << "\ncpu_fallback=false"
          << "\nnan_detected=" << (nanDetected ? "true" : "false")
          << "\ninf_detected=" << (infDetected ? "true" : "false") << '\n'
          << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}

// Private CPU/HTP divergence localization.  The HTP graph is prepared once
// per tap scope ("NONE" baseline control, "COARSE" for every layer output,
// "FINE" for the full single selected layer) and every fixed parity prefix is
// compared at the same boundaries.  Instrumentation is evaluated before the
// data is trusted: the logits fingerprint must be identical to the NONE run
// and the parity_13 FAIL status must not flip to PASS.
std::string nicopediaHtpDivergenceLocalization(
    const tiny_lm::Config &config, const std::string &checkpointPath,
    const nicopedia_gen::GenerateConfig &generateConfig,
    const LogSink &progress) {
  const int seed = generateConfig.samplingSeed > 0 &&
                           generateConfig.samplingSeed < 100000
                       ? static_cast<int>(generateConfig.samplingSeed)
                       : 1;
  const uint32_t layers = config.numLayers;
  const uint32_t heads = config.numHeads;
  if (layers == 0 || heads == 0 || config.tokens == 0 ||
      config.vocabularySize == 0 || config.dimension == 0 ||
      config.feedForwardDimension == 0)
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=layers_heads_tokens_vocab_dim_ffn_must_be_positive\n";
  if (heads != 2)
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=heads_must_be_2\n";
  const std::string scope = generateConfig.diagnosticTapScope;
  if (scope != "NONE" && scope != "COARSE" && scope != "FINE")
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=diagnostic_tap_scope must be NONE, COARSE or FINE\n";
  if (scope == "FINE" &&
      generateConfig.diagnosticLayerIndex >= layers)
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=diagnostic_layer_index out of range\n";
  uint32_t expectedStep = 0;
  if (!nprtParseCheckpointStep(checkpointPath, static_cast<uint32_t>(seed),
                               layers, &expectedStep))
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_FILENAME\n"
           "error=checkpoint filename must be htp-seed<seed>-l<layers>-step<step>.ckpt\n";
  LoadedNprtCheckpoint loaded;
  try {
    loaded = nprtLoadCheckpointForGeneration(checkpointPath, config,
                                             static_cast<uint32_t>(seed));
  } catch (const std::exception &exception) {
    return std::string(
               "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
               "failure_classification=CHECKPOINT_DECODE\nerror=") +
           exception.what() + '\n';
  }
  if (loaded.step != expectedStep)
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_IDENTITY\n"
           "error=checkpoint step does not match filename\n";
  if (!loaded.finite)
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_NONFINITE\n"
           "error=checkpoint contains non-finite values\n";

  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  options.diagnosticLayerIndex = generateConfig.diagnosticLayerIndex;
  runtime.setOptions(options);
  std::string error;
  const TinyTransformerTrainingTapSet tapSet =
      scope == "COARSE" ? TinyTransformerTrainingTapSet::COARSE_LAYER_BOUNDARIES
      : scope == "FINE" ? TinyTransformerTrainingTapSet::NICOPEDIA_FINE
                        : TinyTransformerTrainingTapSet::NONE;
  const auto initStarted = std::chrono::steady_clock::now();
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL, tapSet, layers, heads))
    return "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
           "failure_classification=QNN_PREPARE\n" +
           failure("localization_prepare", error, runtime);
  const double initializeUs =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - initStarted)
          .count();
  const double graphCreateUs = runtime.metrics().graphCreateUs;
  const double graphFinalizeUs = runtime.metrics().graphFinalizeUs;

  const std::vector<float> zeros(
      size_t(config.tokens) * config.vocabularySize, 0.0f);
  const auto windowInput = [&](const std::vector<std::uint8_t> &window) {
    std::vector<uint32_t> tokens;
    tokens.reserve(window.size());
    for (uint8_t byte : window) tokens.push_back(byte);
    return tiny_lm::oneHot(tokens, config.vocabularySize);
  };
  const auto &prefixes = nicopedia_gen::parityPrefixes();
  // Last HTP execution, kept in function scope for the instrumentation-control
  // fingerprint (the NONE run must be the final scope execution).
  TinyTransformerTrainingOutputs lastHtpStep;

  // Instrumentation control + coarse profile in a single pass: the NONE run
  // provides the untapped baseline (fingerprint + parity fields) and the
  // tapped run the intermediate boundaries.
  struct BoundaryRow {
    nicopedia_gen::NprtTapMetric metric;
  };
  std::vector<BoundaryRow> rows;
  bool parity13Fail = true;
  const bool parity13BlockedLegacy = generateConfig.gatePolicy == "legacy";
  auto runScope = [&](const char *label,
                      bool recordBoundaries) -> std::string {
    for (size_t i = 0; i < prefixes.size(); ++i) {
      const auto &prefix = prefixes[i];
      uint32_t pad = 0;
      const auto prefixContext =
          nicopedia_gen::buildGenerationContext(prefix.bytes, config.tokens, &pad);
      const auto cpuTrace = tiny_lm::forwardTraceGeneralized(
          config, windowInput(prefixContext), loaded.parameters);
      TinyTransformerTrainingOutputs htpStep;
      if (!runtime.executeTinyTransformerTraining(
              windowInput(prefixContext), zeros, loaded.parameters, 0.0f,
              htpStep, error))
        return failure("localization_execute", error, runtime);
      if (recordBoundaries) {
        const auto logitsError = compareNprt(cpuTrace.logits, htpStep.logits);
        const auto probabilityError =
            compareNprt(cpuTrace.probabilities, htpStep.probabilities);
        if (i == 13)
          parity13Fail =
              !(logitsError.nonfinite == 0 && probabilityError.nonfinite == 0 &&
                logitsError.maxAbs < 2e-2 && probabilityError.maxAbs < 5e-3);
        BoundaryRow row;
        row.metric = computeNprtTapMetric(cpuTrace.logits, htpStep.logits,
                                          "logits");
        if (scope == "COARSE") {
          // H=2 graph: every layer's two head probabilities come first
          // (layer-major/head-major), then the tap registry suffix holds the
          // non-final layer outputs in layer order.  The final layer output
          // is the established APP_READ output in htpStep.output.
          for (uint32_t layer = 0; layer < layers; ++layer) {
            std::ostringstream name;
            name << "layer_" << std::setw(3) << std::setfill('0') << layer
                 << "_output";
            const size_t tapIndex = size_t(2) * layers + layer;
            const std::vector<float> &htpValues =
                layer + 1 == layers ? htpStep.output
                                    : htpStep.taps[tapIndex].values;
            row.metric = computeNprtTapMetric(
                cpuTrace.layers[layer].output, htpValues, name.str());
            rows.push_back(row);
          }
          row.metric = computeNprtTapMetric(cpuTrace.logits, htpStep.logits,
                                            "logits");
          rows.push_back(row);
        } else if (scope == "FINE") {
          const uint32_t target = generateConfig.diagnosticLayerIndex;
          const auto &layer = cpuTrace.layers[target];
          const std::string prefixName = "layer_" + std::to_string(target) + "_";
          // The H=2 graph emits every layer's two per-head probability
          // tensors first (layer-major/head-major, 2*numLayers taps), then
          // the NICOPEDIA_FINE tap registry suffix in tensor creation order.
          const size_t tapOffset = size_t(2) * layers;
          const std::vector<float> &cpuProbs = layer.probabilities;
          const size_t headProbabilityElements = size_t(config.tokens) * config.tokens;
          const std::vector<float> cpuHead0(
              cpuProbs.begin(),
              cpuProbs.begin() + static_cast<std::ptrdiff_t>(headProbabilityElements));
          const std::vector<float> cpuHead1(
              cpuProbs.begin() + static_cast<std::ptrdiff_t>(headProbabilityElements),
              cpuProbs.end());
          // ff2 is the pre-residual FFN matmul output: block_output - residual1.
          std::vector<float> cpuFf2(layer.output.size());
          for (size_t e = 0; e < layer.output.size(); ++e)
            cpuFf2[e] = layer.output[e] - layer.residual1[e];
          struct TensorSpec {
            std::string name;
            const std::vector<float> *cpu;
          };
          const std::vector<TensorSpec> specs{
              {prefixName + "head_000_probabilities", &cpuHead0},
              {prefixName + "head_001_probabilities", &cpuHead1},
              {prefixName + "ln1_centered_s", &layer.ln1Centered},
              {prefixName + "ln1_square", &layer.ln1Square},
              {prefixName + "ln1_variance_eps", &layer.ln1VarianceEps},
              {prefixName + "ln1_inv", &layer.ln1Inv},
              {prefixName + "ln1", &layer.ln1},
              {prefixName + "q", &layer.q},
              {prefixName + "k", &layer.k},
              {prefixName + "v", &layer.v},
              {prefixName + "attention_context", &layer.context},
              {prefixName + "attention_projected", nullptr},
              {prefixName + "residual1", &layer.residual1},
              {prefixName + "ln2_centered_s", &layer.ln2Centered},
              {prefixName + "ln2_square", &layer.ln2Square},
              {prefixName + "ln2_variance_eps", &layer.ln2VarianceEps},
              {prefixName + "ln2_inv", &layer.ln2Inv},
              {prefixName + "ln2", &layer.ln2},
              {prefixName + "ff1", &layer.ff1},
              {prefixName + "relu", &layer.relu},
              {prefixName + "ff2", &cpuFf2},
              {prefixName + "output", &layer.output},
          };
          // Per-head probabilities come from the established head-probability
          // taps at 2*target and 2*target+1; the remaining specs map to the
          // tap registry suffix.
          for (size_t spec = 0; spec < specs.size(); ++spec) {
            const auto *cpuPtr = specs[spec].cpu;
            if (cpuPtr == nullptr) continue;
            const size_t tapIndex =
                spec < 2 ? size_t(2) * target + spec : tapOffset + (spec - 2);
            if (tapIndex >= htpStep.taps.size()) continue;
            row.metric = computeNprtTapMetric(*cpuPtr,
                                              htpStep.taps[tapIndex].values,
                                              htpStep.taps[tapIndex].name);
            rows.push_back(row);
          }
        }
      }
      if (progress && (i == 0 || i + 1 == prefixes.size()))
        progress(std::string("phase=localization scope=") + label +
                 " prefix=" + std::to_string(i) + "/" +
                 std::to_string(prefixes.size()));
      lastHtpStep = std::move(htpStep);
    }
    return std::string();
  };
  std::string scopeError = runScope(scope.c_str(), true);
  if (!scopeError.empty()) return scopeError;

  std::ostringstream report;
  report << std::setprecision(10)
         << "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\ntest=nicopedia_htp_divergence_localization\n"
         << "status=SUCCESS\n"
         << "model=L" << layers << "\nlayers=" << layers << "\nheads=" << heads
         << "\nseed=" << seed << "\ncheckpoint_step=" << loaded.step
         << "\ncheckpoint_parameter_hash=" << loaded.parameterHash
         << "\ncheckpoint_finite=" << (loaded.finite ? "true" : "false")
         << "\ndiagnostic_tap_scope=" << scope
         << "\ndiagnostic_layer_index="
         << (scope == "FINE" ? std::to_string(generateConfig.diagnosticLayerIndex)
                             : "n/a")
         << "\nprefix_count=" << prefixes.size()
         << "\nparity_13_blocked_legacy="
         << (parity13BlockedLegacy ? "true" : "false")
         << "\nparity_13_fail=" << (parity13Fail ? "true" : "false")
         << "\nhtp_logits_sha256=" << canonicalFloatSha256(lastHtpStep.logits)
         << "\nboundary_count=" << rows.size()
         << "\nhtp_initialize_us=" << initializeUs
         << "\ngraph_create_us=" << graphCreateUs
         << "\ngraph_finalize_us=" << graphFinalizeUs
         << "\ncpu_fallback=false" << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &m = rows[i].metric;
    report << "boundary_" << i << "_name=" << m.name
           << "\nboundary_" << i << "_cpu_rms=" << m.cpuRms
           << "\nboundary_" << i << "_htp_rms=" << m.htpRms
           << "\nboundary_" << i << "_diff_max_abs=" << m.diffMaxAbs
           << "\nboundary_" << i << "_diff_rms=" << m.diffRms
           << "\nboundary_" << i << "_rel_rms=" << m.relRms
           << "\nboundary_" << i << "_cosine=" << m.cosine
           << "\nboundary_" << i << "_centered_rms=" << m.centeredRms
           << '\n';
  }
  return report.str();
}

std::string nicopediaHtpTraining(const tiny_lm::Config &config,
                                 const TrainingConfig &trainingConfig,
                                 const LogSink &progress) {
  // Cache path: app-private file pushed by the host runner.  The parameter is
  // carried in diagnosticCheckpointDir to avoid extending the JNI ABI; the
  // Kotlin side validates it to stay below the app files directory.
  const std::string cachePath = trainingConfig.diagnosticCheckpointDir;
  if (cachePath.empty())
    return "NICOPEDIA_HTP\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=cache_path_required\n";
  NprtCache cache;
  try {
    cache = loadNprtCache(cachePath + "/train_pilot.bin");
  } catch (const std::exception &exception) {
    return std::string("NICOPEDIA_HTP\nstatus=FAILED\n"
                       "failure_classification=CACHE_DECODE\nerror=") +
           exception.what() + '\n';
  }
  if (cache.context != config.tokens || cache.vocabulary != config.vocabularySize)
    return "NICOPEDIA_HTP\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=cache_config_mismatch\n";
  const int seed = trainingConfig.seed > 0 && trainingConfig.seed < 100000
                       ? static_cast<int>(trainingConfig.seed)
                       : 1;
  const int layers = static_cast<int>(config.numLayers);
  const int heads = static_cast<int>(config.numHeads);
  const uint32_t steps = static_cast<uint32_t>(trainingConfig.steps);
  const uint32_t batchSize = static_cast<uint32_t>(trainingConfig.batchSize);
  if (steps == 0 || batchSize == 0 || layers <= 0 || heads <= 0)
    return "NICOPEDIA_HTP\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=steps_batch_layers_heads_must_be_positive\n";
  const float lr = trainingConfig.learningRate > 0.0f
                       ? trainingConfig.learningRate
                       : 0.003f;
  const auto order = nprtTrainingOrder(cache.records.size(), steps, batchSize);
  const std::string orderHashValue = nprtOrderHash(order);
  const Params shape = tiny_lm::initialParameters(config, 1);
  const auto flattenedShape = flattenLanguageParameters(shape);
  if (flattenedShape.empty() ||
      flattenedShape.size() > std::numeric_limits<uint32_t>::max())
    return "NICOPEDIA_HTP\nstatus=FAILED\n"
           "failure_classification=APP_RESOURCE_ESTIMATOR\n"
           "error=parameter registry exceeds uint32 range\n";
  const uint32_t optimizerElements =
      static_cast<uint32_t>(flattenedShape.size());
  const uint32_t optimizerGraphElements =
      static_cast<uint32_t>(std::min<std::uint64_t>(
          optimizerElements,
          phonelm::transformer::kMaximumAdamChunkElements));
  const uint32_t optimizerChunkCount =
      static_cast<uint32_t>((std::uint64_t{optimizerElements} +
                             optimizerGraphElements - 1) /
                            optimizerGraphElements);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  const auto initStarted = std::chrono::steady_clock::now();
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL,
          TinyTransformerTrainingTapSet::NONE, config.numLayers,
          config.numHeads) ||
      !runtime.prepareAdamOptimizer(optimizerGraphElements, error))
    return failure("nicopedia_prepare", error, runtime);
  const double initializeUs =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - initStarted)
          .count();
  const double graphCreateUs = runtime.metrics().graphCreateUs;
  const double graphFinalizeUs = runtime.metrics().graphFinalizeUs;

  Params htp = tiny_lm::initialParameters(config, seed);
  Params cpu = htp;
  Params htpFirst = zeroLanguageParameters(htp), htpSecond = htpFirst;
  Params cpuFirst = htpFirst, cpuSecond = htpFirst;
  const std::string initialParameterHash = nprtParameterHash(htp);

  // Step 0: same-batch CPU/HTP one-step comparison before any update.
  const auto firstBatch = nprtBatch(config, cache, order[0]);
  const auto cpuStep0 = tiny_lm::forwardBackwardGeneralized(
      config, firstBatch.input, firstBatch.target, cpu, 0.0f);
  TinyTransformerTrainingOutputs htpStep0;
  const auto executeStarted = std::chrono::steady_clock::now();
  if (!runtime.executeTinyTransformerTraining(
          firstBatch.input, firstBatch.target, htp, 0.0f, htpStep0, error))
    return failure("nicopedia_step0", error, runtime);
  const double firstExecuteUs =
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - executeStarted)
          .count();
  const auto logits0 = compareNprt(cpuStep0.logits, htpStep0.logits);
  const auto probs0 = compareNprt(cpuStep0.probabilities, htpStep0.probabilities);
  const auto dlogits0 = compareNprt(cpuStep0.dLogits, htpStep0.dLogits);
  const double gradient0 = nprtMaxParamError(cpuStep0.gradients, htpStep0.gradients);
  const bool step0Finite =
      std::isfinite(cpuStep0.loss) && std::isfinite(htpStep0.loss) &&
      logits0.nonfinite == 0 && probs0.nonfinite == 0 &&
      dlogits0.nonfinite == 0 && std::isfinite(gradient0);
  // Tolerances are fixed before results are interpreted (numerical evidence
  // policy): logits/probability/dlogits use the established tiny-LM one-step
  // bounds; the gradient uses the established diagnostic 0.03 bound.
  const bool step0Ok =
      step0Finite && logits0.maxAbs < 2e-2 && probs0.maxAbs < 5e-3 &&
      dlogits0.maxAbs < 5e-3 && gradient0 < 3e-2;

  // Short trajectory: 2, 4, 8 steps of CPU and HTP training from the same
  // initial parameters and the same batch order, comparing loss and parameter
  // drift at each anchor.
  struct TrajectoryAnchor {
    int step = 0;
    float cpuLoss = 0, htpLoss = 0;
    double parameterMaxAbs = 0, firstMomentMaxAbs = 0, secondMomentMaxAbs = 0;
    bool finite = true;
  };
  std::vector<TrajectoryAnchor> anchors;
  auto runTrajectory = [&](uint32_t trajectorySteps) {
    Params tCpu = cpu, tHtp = htp, tCpuFirst = cpuFirst, tCpuSecond = cpuSecond,
          tHtpFirst = htpFirst, tHtpSecond = htpSecond;
    for (uint32_t step = 1; step <= trajectorySteps; ++step) {
      Params cpuGradientAccum = zeroLanguageParameters(tCpu);
      Params htpGradientAccum = zeroLanguageParameters(tHtp);
      double cpuLossSum = 0, htpLossSum = 0;
      for (uint32_t batch = 0; batch < batchSize; ++batch) {
        const auto batchData = nprtBatch(
            config, cache, order[std::size_t(step - 1) * batchSize + batch]);
        const auto cpuGradient = tiny_lm::forwardBackwardGeneralized(
            config, batchData.input, batchData.target, tCpu, 0.0f);
        TinyTransformerTrainingOutputs htpGradient;
        if (!runtime.executeTinyTransformerTraining(
                batchData.input, batchData.target, tHtp, 0.0f, htpGradient,
                error))
          return false;
        cpuLossSum += cpuGradient.loss;
        htpLossSum += htpGradient.loss;
        const auto cpuRegistry = tiny_lm::parameterRegistry(cpuGradientAccum);
        const auto cpuGradRegistry =
            tiny_lm::parameterRegistry(cpuGradient.gradients);
        const auto htpRegistry = tiny_lm::parameterRegistry(htpGradientAccum);
        const auto htpGradRegistry =
            tiny_lm::parameterRegistry(htpGradient.gradients);
        if (cpuRegistry.size() != cpuGradRegistry.size() ||
            htpRegistry.size() != htpGradRegistry.size())
          return false;
        for (size_t i = 0; i < cpuRegistry.size(); ++i) {
          auto &cpuAccum = *const_cast<std::vector<float> *>(cpuRegistry[i].values);
          auto &htpAccum = *const_cast<std::vector<float> *>(htpRegistry[i].values);
          const auto &cpuValues = *cpuGradRegistry[i].values;
          const auto &htpValues = *htpGradRegistry[i].values;
          for (size_t j = 0; j < cpuAccum.size(); ++j) {
            cpuAccum[j] += cpuValues[j] * (1.0f / float(batchSize));
            htpAccum[j] += htpValues[j] * (1.0f / float(batchSize));
          }
        }
      }
      const float cpuMeanLoss = float(cpuLossSum / batchSize);
      const float htpMeanLoss = float(htpLossSum / batchSize);
      const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
      const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
      const auto cpuUpdate = tiny_lm::adamUpdate(
          tCpu, cpuGradientAccum, tCpuFirst, tCpuSecond, lr, .9f, .999f,
          1e-8f, c1, c2);
      Params htpNext, htpFirstNext, htpSecondNext;
      AdamOptimizerOutputs raw;
      if (!executeLanguageAdam(runtime, tHtp, htpGradientAccum, tHtpFirst,
                               tHtpSecond, lr, int(step), 1.0f, htpNext,
                               htpFirstNext, htpSecondNext, &raw, error,
                               optimizerGraphElements))
        return false;
      tCpu = cpuUpdate.next;
      tCpuFirst = cpuUpdate.firstMoment;
      tCpuSecond = cpuUpdate.secondMoment;
      tHtp = std::move(htpNext);
      tHtpFirst = std::move(htpFirstNext);
      tHtpSecond = std::move(htpSecondNext);
      if (step == 2 || step == 4 || step == 8) {
        TrajectoryAnchor anchor;
        anchor.step = int(step);
        anchor.cpuLoss = cpuMeanLoss;
        anchor.htpLoss = htpMeanLoss;
        anchor.parameterMaxAbs = nprtMaxParamError(tCpu, tHtp);
        anchor.firstMomentMaxAbs = nprtMaxParamError(tCpuFirst, tHtpFirst);
        anchor.secondMomentMaxAbs = nprtMaxParamError(tCpuSecond, tHtpSecond);
        anchor.finite =
            std::isfinite(cpuMeanLoss) && std::isfinite(htpMeanLoss) &&
            std::isfinite(anchor.parameterMaxAbs) &&
            std::isfinite(anchor.firstMomentMaxAbs) &&
            std::isfinite(anchor.secondMomentMaxAbs);
        anchors.push_back(anchor);
      }
      if (!std::isfinite(htpMeanLoss) || !std::isfinite(cpuMeanLoss))
        return false;
    }
    return true;
  };
  if (!runTrajectory(8)) return failure("nicopedia_short_trajectory", error, runtime);

  // Full training loop: HTP forward/backward and HTP Adam for every step.
  // The CPU reference runs in lockstep so the final state and the trajectory
  // remain comparable.
  Params current = cpu, currentFirst = cpuFirst, currentSecond = cpuSecond;
  float firstLoss = 0, lastLoss = 0;
  bool allFinite = true;
  uint32_t completedSteps = 0;
  const auto trainingStarted = std::chrono::steady_clock::now();
  for (uint32_t step = 1; step <= steps; ++step) {
    Params gradientAccum = zeroLanguageParameters(current);
    double lossSum = 0;
    bool stepFinite = true;
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
      const auto batchData =
          nprtBatch(config, cache, order[std::size_t(step - 1) * batchSize + batch]);
      TinyTransformerTrainingOutputs htpGradient;
      if (!runtime.executeTinyTransformerTraining(
              batchData.input, batchData.target, current, 0.0f, htpGradient,
              error))
        return failure("nicopedia_train_gradient", error, runtime);
      lossSum += htpGradient.loss;
      stepFinite = stepFinite && std::isfinite(htpGradient.loss);
      const auto registry = tiny_lm::parameterRegistry(gradientAccum);
      const auto gradientRegistry = tiny_lm::parameterRegistry(htpGradient.gradients);
      if (registry.size() != gradientRegistry.size()) {
        return "NICOPEDIA_HTP\nstatus=FAILED\n"
               "failure_classification=APP_PARAMETER_SCHEMA\n"
               "error=gradient registry mismatch\n";
      }
      for (size_t i = 0; i < registry.size(); ++i) {
        auto &accum = *const_cast<std::vector<float> *>(registry[i].values);
        const auto &values = *gradientRegistry[i].values;
        for (size_t j = 0; j < accum.size(); ++j)
          accum[j] += values[j] * (1.0f / float(batchSize));
      }
    }
    const float meanLoss = float(lossSum / batchSize);
    Params next, firstNext, secondNext;
    AdamOptimizerOutputs raw;
    if (!executeLanguageAdam(runtime, current, gradientAccum, currentFirst,
                             currentSecond, lr, int(step), 1.0f, next,
                             firstNext, secondNext, &raw, error,
                             optimizerGraphElements))
      return failure("nicopedia_train_adam", error, runtime);
    current = std::move(next);
    currentFirst = std::move(firstNext);
    currentSecond = std::move(secondNext);
    stepFinite = stepFinite && finiteParams(current) &&
                 finiteParams(currentFirst) && finiteParams(currentSecond);
    allFinite = allFinite && stepFinite;
    if (!stepFinite) break;
    ++completedSteps;
    if (step == 1) firstLoss = meanLoss;
    lastLoss = meanLoss;
    if (progress && (step == 1 || step % 32 == 0 || step == steps)) {
      std::ostringstream update;
      update << std::setprecision(10) << "phase=training\nseed=" << seed
             << "\nstep=" << step << "\nsteps=" << steps << "\nloss=" << meanLoss;
      progress(update.str());
    }
  }
  const double trainingSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - trainingStarted)
          .count();
  const double stepMs = completedSteps ? trainingSeconds / completedSteps * 1000.0
                                       : 0.0;
  // Final state comparison.
  Params cpuFinal = cpu, cpuFinalFirst = cpuFirst, cpuFinalSecond = cpuSecond;
  for (uint32_t step = 1; step <= completedSteps; ++step) {
    Params gradientAccum = zeroLanguageParameters(cpuFinal);
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
      const auto batchData =
          nprtBatch(config, cache, order[std::size_t(step - 1) * batchSize + batch]);
      const auto cpuGradient = tiny_lm::forwardBackwardGeneralized(
          config, batchData.input, batchData.target, cpuFinal, 0.0f);
      const auto registry = tiny_lm::parameterRegistry(gradientAccum);
      const auto gradientRegistry = tiny_lm::parameterRegistry(cpuGradient.gradients);
      for (size_t i = 0; i < registry.size(); ++i) {
        auto &accum = *const_cast<std::vector<float> *>(registry[i].values);
        const auto &values = *gradientRegistry[i].values;
        for (size_t j = 0; j < accum.size(); ++j)
          accum[j] += values[j] * (1.0f / float(batchSize));
      }
    }
    const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
    const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
    const auto cpuUpdate = tiny_lm::adamUpdate(
        cpuFinal, gradientAccum, cpuFinalFirst, cpuFinalSecond, lr, .9f, .999f,
        1e-8f, c1, c2);
    cpuFinal = cpuUpdate.next;
    cpuFinalFirst = cpuUpdate.firstMoment;
    cpuFinalSecond = cpuUpdate.secondMoment;
  }
  const double finalParameterError = nprtMaxParamError(cpuFinal, current);
  const double finalFirstError = nprtMaxParamError(cpuFinalFirst, currentFirst);
  const double finalSecondError = nprtMaxParamError(cpuFinalSecond, currentSecond);
  const bool finalFinite =
      finiteParams(current) && finiteParams(currentFirst) &&
      finiteParams(currentSecond) && finiteParams(cpuFinal);
  const bool lossDecreased = std::isfinite(firstLoss) && std::isfinite(lastLoss) &&
                             lastLoss < firstLoss;
  const bool ok = step0Ok && allFinite && finalFinite && lossDecreased &&
                  completedSteps == steps;
  // Private checkpoint: NPRTCKPTV1 in the app files directory.  The host
  // runner pulls it and evaluates validation/development with the CPU pilot
  // evaluation path.  It never enters public artifacts.
  const std::string checkpointPath =
      cachePath + "/htp-seed" + std::to_string(seed) + "-l" +
      std::to_string(layers) + "-step" + std::to_string(completedSteps) +
      ".ckpt";
  const bool checkpointWritten =
      nprtSaveCheckpoint(checkpointPath, config, uint32_t(seed),
                         completedSteps, current);
  std::ostringstream report;
  report << std::setprecision(10)
         << "NICOPEDIA_HTP\ntest=nicopedia_real_text_htp_training\nstatus="
         << (ok ? "SUCCESS" : "FAILED")
         << "\nseed=" << seed << "\nlayers=" << layers << "\nheads=" << heads
         << "\nsteps=" << steps << "\nbatch_size=" << batchSize
         << "\nlearning_rate=" << lr
         << "\ncache_context=" << cache.context
         << "\ncache_vocabulary=" << cache.vocabulary
         << "\ncache_record_count=" << cache.records.size()
         << "\ncache_content_hash=" << cache.contentHash
         << "\ntraining_order_hash=" << orderHashValue
         << "\ninitial_parameter_hash=" << initialParameterHash
         << "\nparameter_element_count=" << optimizerElements
         << "\noptimizer_chunk_count=" << optimizerChunkCount
         << "\nstep0_loss_cpu=" << cpuStep0.loss << "\nstep0_loss_htp="
         << htpStep0.loss << "\nstep0_logits_max_abs_error=" << logits0.maxAbs
         << "\nstep0_logits_mean_abs_error=" << logits0.meanAbs
         << "\nstep0_logits_max_relative_error=" << logits0.maxRelative
         << "\nstep0_logits_l2_error=" << logits0.l2
         << "\nstep0_logits_cosine_similarity=" << logits0.cosine
         << "\nstep0_probability_max_abs_error=" << probs0.maxAbs
         << "\nstep0_dlogits_max_abs_error=" << dlogits0.maxAbs
         << "\nstep0_gradient_max_abs_error=" << gradient0
         << "\nstep0_finite=" << (step0Finite ? "true" : "false")
         << "\nstep0_ok=" << (step0Ok ? "true" : "false")
         << "\nstep0_tolerance_logits=2e-2\nstep0_tolerance_probability=5e-3"
         << "\nstep0_tolerance_dlogits=5e-3\nstep0_tolerance_gradient=3e-2";
  for (const auto &anchor : anchors) {
    report << "\ntrajectory_step_" << anchor.step << "_cpu_loss="
           << anchor.cpuLoss << "\ntrajectory_step_" << anchor.step
           << "_htp_loss=" << anchor.htpLoss << "\ntrajectory_step_"
           << anchor.step << "_parameter_max_abs_error="
           << anchor.parameterMaxAbs << "\ntrajectory_step_" << anchor.step
           << "_first_moment_max_abs_error=" << anchor.firstMomentMaxAbs
           << "\ntrajectory_step_" << anchor.step
           << "_second_moment_max_abs_error=" << anchor.secondMomentMaxAbs
           << "\ntrajectory_step_" << anchor.step << "_finite="
           << (anchor.finite ? "true" : "false");
  }
  report << "\nfirst_loss=" << firstLoss << "\nlast_loss=" << lastLoss
         << "\nloss_decreased=" << (lossDecreased ? "true" : "false")
         << "\ncompleted_steps=" << completedSteps
         << "\nall_steps_finite=" << (allFinite ? "true" : "false")
         << "\nfinal_parameter_max_abs_error=" << finalParameterError
         << "\nfinal_first_moment_max_abs_error=" << finalFirstError
         << "\nfinal_second_moment_max_abs_error=" << finalSecondError
         << "\nfinal_parameter_canonical_hash="
         << canonicalFloatSha256(flattenLanguageParameters(current))
         << "\nfinal_cpu_parameter_canonical_hash="
         << canonicalFloatSha256(flattenLanguageParameters(cpuFinal))
         << "\nfinal_parameter_hash=" << nprtParameterHash(current)
         << "\nfinal_finite=" << (finalFinite ? "true" : "false")
         << "\ncheckpoint_written=" << (checkpointWritten ? "true" : "false")
         << "\nhtp_initialize_us=" << initializeUs
         << "\ngraph_create_us=" << graphCreateUs
         << "\ngraph_finalize_us=" << graphFinalizeUs
         << "\nfirst_execute_us=" << firstExecuteUs
         << "\ntraining_total_seconds=" << trainingSeconds
         << "\ntraining_step_ms=" << stepMs
         << "\ngraph_execute_count=" << runtime.metrics().graphExecuteCount
         << "\nexecute_count_per_training_step=" << (1 + optimizerChunkCount)
         << "\nbias_correction_scalar_responsibility=CPU"
         << "\noptimizer_math_responsibility=HTP"
         << "\ncpu_fallback=false\nnan_detected=" << (allFinite ? "false" : "true")
         << "\ninf_detected=" << (allFinite ? "false" : "true") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
}  // namespace
std::string runTinyTransformerTrainingExperiment(
    ExecutionMode mode, const TrainingConfig& trainingConfig,
    const LogSink& progress) {
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA) {
    tiny_lm::Config config;
    config.vocabularySize = 256;
    config.tokens = 32;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    config.numLayers =
        static_cast<uint32_t>(trainingConfig.epochs > 0 ? trainingConfig.epochs : 6);
    config.numHeads =
        static_cast<uint32_t>(trainingConfig.measuredSteps > 0
                                  ? trainingConfig.measuredSteps
                                  : 2);
    std::string error;
    if (!tiny_lm::validateConfig(config, &error))
      return "NICOPEDIA_HTP\nstatus=FAILED\n"
             "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=" +
             error + '\n';
    return nicopediaHtpTraining(config, trainingConfig, progress);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA_GENERATE) {
    tiny_lm::Config config;
    config.vocabularySize = 256;
    config.tokens = 32;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    config.numLayers =
        static_cast<uint32_t>(trainingConfig.epochs > 0 ? trainingConfig.epochs : 6);
    config.numHeads = 2;
    std::string error;
    if (!tiny_lm::validateConfig(config, &error))
      return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
             "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=" +
             error + '\n';
    const std::string dir = trainingConfig.diagnosticCheckpointDir;
    if (dir.empty())
      return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
             "failure_classification=APP_CONFIGURATION_VALIDATION\n"
             "error=checkpoint_dir_required\n";
    const int seed = trainingConfig.seed > 0 && trainingConfig.seed < 100000
                         ? static_cast<int>(trainingConfig.seed)
                         : 1;
    // The dedicated JNI path supplies exact paths; this mode-level fallback
    // only exists for the UI path and pins the step-320 anchor filename.
    const std::string checkpointPath =
        dir + "/htp-seed" + std::to_string(seed) + "-l" +
        std::to_string(config.numLayers) + "-step320.ckpt";
    const std::string promptPath = dir + "/prompt.bin";
    nicopedia_gen::GenerateConfig generate;
    generate.maxNewBytes =
        static_cast<uint32_t>(trainingConfig.steps > 0 ? trainingConfig.steps : 64);
    if (generate.maxNewBytes > 1024) generate.maxNewBytes = 1024;
    generate.greedy = true;
    generate.temperature = 1.0f;
    generate.topK = 32;
    generate.samplingSeed = static_cast<uint64_t>(trainingConfig.seed);
    return nicopediaHtpGeneration(config, checkpointPath, promptPath,
                                  trainingConfig, generate, progress);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_BISECTION)
    return runTinyLmGraphBisection(false);
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_BISECTION_PRELUDE)
    return runTinyLmGraphBisection(true);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_FULL_ISOLATED)
    return runTinyLmGraphIsolated(0);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_DINPUT_ISOLATED)
    return runTinyLmGraphIsolated(1);
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_DEMBEDDING_ISOLATED)
    return runTinyLmGraphIsolated(2);
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_TAP_BACKWARD_REGIONS)
    return runTinyLmFirstChangeTap(0);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_TAP_LAYERNORM1)
    return runTinyLmFirstChangeTap(1);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_TAP_DSCORES_ONLY)
    return runTinyLmFirstChangeTap(2);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_TAP_DPROB_DSCORES)
    return runTinyLmFirstChangeTap(3);
  if (static_cast<int>(mode) >= static_cast<int>(
          ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_FULL_DINPUT_DEMBEDDING) &&
      static_cast<int>(mode) <= static_cast<int>(
          ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_DEMBEDDING_DEMBEDDING_DEMBEDDING))
    return runTinyLmGraphOrderOrthogonalization(
        static_cast<int>(mode) - static_cast<int>(
            ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GRAPH_ORDER_FULL_DINPUT_DEMBEDDING));
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_REPRODUCIBILITY)
    return runTinyLmDembeddingReproducibility();
  if (mode == ExecutionMode::QNN_HTP_TINY_TRANSFORMER_TRAINING_STEP)
    return oneStep();
  if (mode == ExecutionMode::QNN_HTP_TINY_TRANSFORMER_TRAINING_MULTI_STEP)
    return multiStep();
  if (mode == ExecutionMode::QNN_HTP_CROSS_ENTROPY_CHECK)
    return crossEntropyMicrotest();
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_STEP)
    return languageModelOneStep();
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_MULTI_STEP)
    return languageModelMultiStep(false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_INFERENCE)
    return languageModelMultiStep(true);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_1)
    return languageModelMultiStep(false, 1);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_2)
    return languageModelMultiStep(false, 2);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_3)
    return languageModelMultiStep(false, 3);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_STEP)
    return languageModelMomentum(true, 1, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_1)
    return languageModelMomentum(false, 1, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_2)
    return languageModelMomentum(false, 2, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_INFERENCE)
    return languageModelMomentum(false, 1, true);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP)
    return languageModelAdam(true, 1, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_1)
    return languageModelAdam(false, 1, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_2)
    return languageModelAdam(false, 2, false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_INFERENCE)
    return languageModelAdam(false, 2, true);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_LATE_NONFINITE_BASELINE)
    return runLateNonfiniteExperiment(false);
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_LATE_NONFINITE_DIAGNOSTIC)
    return runLateNonfiniteExperiment(true);
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_POST_FIX_END_TO_END)
    return languageModelAdam(false, 3, true);
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_SEQUENCE_16_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 16;
    return languageModelAdam(false, 3, true, config, 1, true);
  }
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_SEQUENCE_32_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 32;
    return languageModelAdam(false, 3, true, config, 1, true);
  }
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_DIMENSION_32_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true);
  }
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_LAYERS_2_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 2, 1);
  }
  if (mode ==
      ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_HEADS_2_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 1, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_FORMAL) {
    tiny_lm::Config config;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 5, false, 2, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T16D16_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 16;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 2, 1);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_T32D32_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 2, 1);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T16D16_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 16;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 1, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_T32D32_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 1, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T16D16_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 16;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 2, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_SMOKE) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 1, true, 2, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H1_FORMAL) {
    tiny_lm::Config config;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 5, false, 2, 1);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L1H2_FORMAL) {
    tiny_lm::Config config;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 5, false, 1, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_FORMAL) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    config.feedForwardDimension = 32;
    return languageModelAdam(false, 3, true, config, 5, false, 2, 2,
                             progress);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_SCALE_L2H2_T32D32_DIAGNOSTIC) {
    tiny_lm::Config config;
    config.tokens = 32;
    config.dimension = 32;
    config.feedForwardDimension = 32;
    return languageModelAdam(true, 3, true, config, 1, false, 2, 2);
  }
  if (mode == ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC) {
    if (trainingConfig.batchSize != 1 || trainingConfig.sampleCount <= 0 ||
        trainingConfig.outputDimension < 13 || trainingConfig.dimension <= 0 ||
        trainingConfig.hiddenDimension <= 0 || trainingConfig.epochs <= 0 ||
        trainingConfig.measuredSteps <= 0 || trainingConfig.steps <= 0 ||
        trainingConfig.correctnessInterval <= 0 ||
        !std::isfinite(trainingConfig.learningRate) ||
        trainingConfig.learningRate <= 0.0f) {
      return "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
             "failure_classification=APP_CONFIGURATION_VALIDATION\n"
             "error=generic mode requires B=1,T>0,V>=13,D>0,FFN>0,"
             "L>0,H>0,steps>0,seeds>0,and finite lr>0\n";
    }
    int firstSeed = 1;
    const char* seedSelectionMode = "COUNT_FROM_ONE";
    if (const char* seedError = phonelm::validateSeedSelection(
            std::uint32_t(trainingConfig.seedSelectionMode),
            trainingConfig.seed,
            std::int64_t(trainingConfig.correctnessInterval))) {
      return std::string(
                 "TINY_LANGUAGE_MODEL\nstatus=FAILED\n"
                 "failure_classification=APP_CONFIGURATION_VALIDATION\nerror=") +
             seedError +
             " (EXACT_SEED requires correctness_interval == seed with "
             "1 <= seed <= INT32_MAX)\n";
    }
    if (trainingConfig.seedSelectionMode == 1) {
      seedSelectionMode = "EXACT_SEED";
      firstSeed = trainingConfig.correctnessInterval;
    }
    tiny_lm::Config config;
    config.tokens = static_cast<uint32_t>(trainingConfig.sampleCount);
    config.vocabularySize =
        static_cast<uint32_t>(trainingConfig.outputDimension);
    config.dimension = static_cast<uint32_t>(trainingConfig.dimension);
    config.feedForwardDimension =
        static_cast<uint32_t>(trainingConfig.hiddenDimension);
    const bool numericalProbe = trainingConfig.steps < 320;
    const bool scalingSmoke = trainingConfig.correctnessInterval < 5;
    return languageModelAdam(
        false, 3, true, config, trainingConfig.correctnessInterval,
        scalingSmoke, trainingConfig.epochs, trainingConfig.measuredSteps,
        progress, trainingConfig.steps, numericalProbe,
        trainingConfig.learningRate, firstSeed, seedSelectionMode,
        trainingConfig.correctnessInterval,
        std::uint32_t(trainingConfig.trainingStabilityMode),
        std::uint32_t(trainingConfig.depthPairInitMode),
        std::uint32_t(trainingConfig.checkpointSelectionMode),
        trainingConfig.diagnosticTrajectory,
        trainingConfig.diagnosticCheckpointDir);
  }
  return "TINY_TRANSFORMER_TRAINING\nstatus=FAILED\nerror=unsupported mode\n";
}

std::string replayFirstNonfiniteCheckpoint(
    const std::vector<std::uint8_t> &payload, std::uint32_t repeatCount,
    TinyTransformerTrainingTapSet tapSet) {
  if (repeatCount == 0 || repeatCount > 1000) {
    return "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
           "failure_classification=APP_CONFIGURATION_VALIDATION\n"
           "error=repeat_count_must_be_1_to_1000\n";
  }
  first_nonfinite::Checkpoint checkpoint;
  std::string error;
  if (!first_nonfinite::decodeCheckpoint(payload, &checkpoint, &error)) {
    return "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
           "failure_classification=CHECKPOINT_DECODE\nerror=" + error + '\n';
  }
  tiny_lm::Config config;
  config.tokens = checkpoint.config.tokens;
  config.vocabularySize = checkpoint.config.vocabularySize;
  config.dimension = checkpoint.config.dimension;
  config.feedForwardDimension = checkpoint.config.feedForwardDimension;
  config.epsilon = checkpoint.config.epsilon;
  config.numLayers = checkpoint.config.numLayers;
  config.numHeads = checkpoint.config.numHeads;
  Params shape;
  Params parameters, firstMoment, secondMoment;
  try {
    shape = tiny_lm::initialParameters(config, checkpoint.seed);
    const auto registry = lateParameterRegistry(config, shape);
    if (!first_nonfinite::decodeCheckpoint(payload, &checkpoint, &error,
                                           &checkpoint.config, &registry)) {
      return "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
             "failure_classification=CHECKPOINT_SCHEMA\nerror=" + error + '\n';
    }
    parameters = unflattenLanguageParameters(checkpoint.parameters, shape);
    firstMoment = unflattenLanguageParameters(checkpoint.adamM, shape);
    secondMoment = unflattenLanguageParameters(checkpoint.adamV, shape);
  } catch (const std::exception &exception) {
    return std::string("FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
                       "failure_classification=CHECKPOINT_SCHEMA\nerror=") +
           exception.what() + '\n';
  }
  const auto emitSummary = [](const std::string &prefix,
                              const std::vector<float> &values,
                              std::ostringstream &report) {
    const auto summary = first_nonfinite::summarize(values);
    report << prefix << "_count=" << summary.count << '\n'
           << prefix << "_finite=" << summary.finite << '\n'
           << prefix << "_nan=" << summary.nan << '\n'
           << prefix << "_positive_inf=" << summary.positiveInfinity << '\n'
           << prefix << "_negative_inf=" << summary.negativeInfinity << '\n'
           << prefix << "_min=" << summary.minimum << '\n'
           << prefix << "_max=" << summary.maximum << '\n'
           << prefix << "_max_abs=" << summary.maximumAbsolute << '\n'
           << prefix << "_mean=" << summary.mean << '\n'
           << prefix << "_mean_abs=" << summary.meanAbsolute << '\n'
           << prefix << "_rms=" << summary.rms << '\n'
           << prefix << "_min_nonzero=" << summary.minimumNonzero << '\n'
           << prefix << "_hash=" << summary.hash << '\n';
  };
  const auto emitComparison = [](const std::string &prefix,
                                 const std::vector<float> &cpu,
                                 const std::vector<float> &htp,
                                 std::ostringstream &report) {
    const auto comparison = first_nonfinite::compare(cpu, htp);
    report << prefix << "_max_abs=" << comparison.maximumAbsolute << '\n'
           << prefix << "_mean_abs=" << comparison.meanAbsolute << '\n'
           << prefix << "_relative_l2=" << comparison.relativeL2 << '\n'
           << prefix << "_argmax=" << comparison.argmax << '\n'
           << prefix << "_first_diff="
           << (comparison.firstDifferent == std::numeric_limits<size_t>::max()
                   ? -1 : static_cast<long long>(comparison.firstDifferent))
           << '\n' << prefix << "_top3=";
    for (size_t index = 0; index < comparison.top3.size(); ++index)
      report << (index ? "," : "") << comparison.top3[index];
    report << '\n';
  };
  const auto cpuGradient = tiny_lm::forwardBackward(
      config, checkpoint.input, checkpoint.target, parameters, 0.0f);
  const double cpuNorm = gradientNorm(cpuGradient.gradients);
  const float cpuScale = checkpoint.config.clipThreshold > 0.0f &&
          std::isfinite(cpuNorm) && cpuNorm > 0.0
      ? float(std::min(1.0, double(checkpoint.config.clipThreshold) /
                             (cpuNorm + 1.0e-6))) : 1.0f;
  const float c1 = float(1.0 / (1.0 - std::pow(checkpoint.config.beta1,
                                                 double(checkpoint.nextOptimizerStep))));
  const float c2 = float(1.0 / (1.0 - std::pow(checkpoint.config.beta2,
                                                 double(checkpoint.nextOptimizerStep))));
  // Scheduled modes replay the same schedule, anchored by the recorded
  // totalSteps; LEGACY is a constant pass-through.
  const float replayLearningRate = phonelm::stabilityLearningRate(
      checkpoint.config.trainingStabilityMode, checkpoint.config.learningRate,
      checkpoint.nextOptimizerStep, checkpoint.config.totalSteps);
  const auto cpuUpdate = tiny_lm::adamUpdate(
      parameters, scaleLanguageParameters(cpuGradient.gradients, cpuScale),
      firstMoment, secondMoment, replayLearningRate,
      checkpoint.config.beta1, checkpoint.config.beta2,
      checkpoint.config.adamEpsilon, c1, c2);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  const auto elementCount = checkpoint.parameters.size();
  std::ostringstream report;
  report << std::setprecision(10)
         << "FIRST_NONFINITE_REPLAY\nstatus=DIAGNOSTIC_COMPLETE\n"
         << "checkpoint_private_device_only=true\n"
         << "checkpoint_original_match=unknown\n"
         << "checkpoint_state_hash=" << checkpoint.stateHash << '\n'
         << "checkpoint_seed=" << checkpoint.seed << '\n'
         << "checkpoint_completed_step=" << checkpoint.completedStep << '\n'
         << "checkpoint_optimizer_next_step=" << checkpoint.nextOptimizerStep << '\n'
         << "checkpoint_training_stability_mode="
         << phonelm::trainingStabilityModeName(
                checkpoint.config.trainingStabilityMode)
         << '\n'
         << "checkpoint_depth_pair_init_mode="
         << phonelm::depthPairInitModeName(checkpoint.config.depthPairInitMode)
         << '\n'
         << "checkpoint_total_steps=" << checkpoint.config.totalSteps << '\n'
         << "replay_learning_rate=" << replayLearningRate << '\n'
         << "replay_count=" << repeatCount << '\n'
         << "cpu_forward_success="
         << (std::isfinite(cpuGradient.loss) ? "true" : "false") << '\n'
         << "cpu_backward_success="
         << (finiteParams(cpuGradient.gradients) ? "true" : "false") << '\n'
         << "cpu_adam_success="
         << (finiteParams(cpuUpdate.next) ? "true" : "false") << '\n';
  emitSummary("cpu_forward_logits", cpuGradient.logits, report);
  emitSummary("cpu_backward_gradients",
              flattenLanguageParameters(cpuGradient.gradients), report);
  emitSummary("cpu_adam_parameters", flattenLanguageParameters(cpuUpdate.next), report);
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize,
          TinyTransformerTrainingVariant::FULL, tapSet, config.numLayers,
          config.numHeads) ||
      elementCount > std::numeric_limits<uint32_t>::max() ||
      !runtime.prepareAdamOptimizer(static_cast<uint32_t>(elementCount), error)) {
    report << "htp_prepare_success=false\nhtp_prepare_error=" << error << '\n'
           << runtime.apiTraceSummary() << runtime.diagnostics();
    return report.str();
  }
  std::vector<std::string> htpHashes;
  bool htpForwardBackwardSuccess = true, htpAdamSuccess = true;
  bool htpAdamAttempted = false;
  TinyTransformerTrainingOutputs htpGradientFirst;
  bool htpGradientFirstValid = false;
  Params htpNextFirst, htpMFirst, htpVFirst;
  bool htpAdamFirstOk = false;
  for (uint32_t repeat = 0; repeat < repeatCount; ++repeat) {
    TinyTransformerTrainingOutputs htpGradient;
    std::string executionError;
    const bool executeOk = runtime.executeTinyTransformerTraining(
        checkpoint.input, checkpoint.target, parameters, 0.0f, htpGradient,
        executionError);
    report << "repeat_" << repeat << "_htp_forward_backward_success="
           << (executeOk ? "true" : "false") << '\n'
           << "repeat_" << repeat << "_htp_qnn_return="
           << runtime.apiTrace().lastQnnResult << '\n';
    emitSummary("repeat_" + std::to_string(repeat) + "_htp_forward_logits",
                htpGradient.logits, report);
    emitSummary("repeat_" + std::to_string(repeat) +
                    "_htp_forward_transformer_output",
                htpGradient.output, report);
    emitSummary("repeat_" + std::to_string(repeat) +
                    "_htp_forward_probabilities",
                htpGradient.probabilities, report);
    emitSummary("repeat_" + std::to_string(repeat) + "_htp_backward_dlogits",
                htpGradient.dLogits, report);
    emitSummary("repeat_" + std::to_string(repeat) + "_htp_backward_doutput",
                htpGradient.dOutput, report);
    if (cpuGradient.transformerOutput.size() == htpGradient.output.size()) {
      emitComparison("repeat_" + std::to_string(repeat) +
                         "_cpu_htp_transformer_output",
                     cpuGradient.transformerOutput, htpGradient.output, report);
    }
    for (size_t layer = 0;
         layer < htpGradient.layerInputGradients.size(); ++layer) {
      emitSummary("repeat_" + std::to_string(repeat) +
                      "_htp_backward_layer_input_" + std::to_string(layer),
                  htpGradient.layerInputGradients[layer], report);
    }
    bool firstNonfiniteTapReported = false;
    for (size_t tapIndex = 0; tapIndex < htpGradient.taps.size(); ++tapIndex) {
      const auto &tap = htpGradient.taps[tapIndex];
      const std::string tapPrefix = "repeat_" + std::to_string(repeat) +
          "_tap_" + std::to_string(tapIndex);
      report << tapPrefix << "_name=" << tap.name << '\n';
      emitSummary(tapPrefix, tap.values, report);
      const bool tapFinite = std::all_of(
          tap.values.begin(), tap.values.end(),
          [](float value) { return std::isfinite(value); });
      if (!tapFinite && !firstNonfiniteTapReported) {
        const auto bad = std::find_if(
            tap.values.begin(), tap.values.end(),
            [](float value) { return !std::isfinite(value); });
        const size_t badFlatIndex =
            static_cast<size_t>(std::distance(tap.values.begin(), bad));
        report << "repeat_" << repeat
               << "_first_nonfinite_tap_index=" << tapIndex << '\n'
               << "repeat_" << repeat
               << "_first_nonfinite_tap_name=" << tap.name << '\n'
               << "repeat_" << repeat
               << "_first_nonfinite_tap_flat_index=" << badFlatIndex << '\n';
        // The private report retains only the single predecessor value needed
        // to prove the operation boundary. It never enters a public exporter.
        const auto hasSuffix = [](const std::string &value,
                                  const std::string &suffix) {
          return value.size() >= suffix.size() &&
              value.compare(value.size() - suffix.size(), suffix.size(),
                            suffix) == 0;
        };
        if (hasSuffix(tap.name, "_square") && tapIndex > 0) {
          const auto &inputTap = htpGradient.taps[tapIndex - 1];
          if (hasSuffix(inputTap.name, "_centered_s") &&
              inputTap.values.size() == tap.values.size()) {
            const float inputValue = inputTap.values[badFlatIndex];
            const float cpuEquivalent = inputValue * inputValue;
            report << "repeat_" << repeat
                   << "_first_nonfinite_operation=ElementWiseMultiply(square)\n"
                   << "repeat_" << repeat
                   << "_first_nonfinite_input_value=" << inputValue << '\n'
                   << "repeat_" << repeat
                   << "_first_nonfinite_cpu_equivalent=" << cpuEquivalent << '\n'
                   << "repeat_" << repeat
                   << "_first_nonfinite_cpu_equivalent_finite="
                   << (std::isfinite(cpuEquivalent) ? "true" : "false") << '\n';
          }
        }
        firstNonfiniteTapReported = true;
      }
    }
    std::string repeatSignature =
        canonicalFloatSha256(htpGradient.logits) + ":" +
        canonicalFloatSha256(htpGradient.output);
    if (!htpGradient.taps.empty()) {
      repeatSignature += ":" +
          canonicalFloatSha256(htpGradient.taps.back().values);
    }
    htpHashes.push_back(std::move(repeatSignature));
    if (!executeOk) {
      htpForwardBackwardSuccess = false;
      report << "repeat_" << repeat << "_htp_forward_backward_error="
             << executionError << '\n';
      continue;
    }
    const auto gradientFlat = flattenLanguageParameters(htpGradient.gradients);
    htpHashes.back() += ":" + canonicalFloatSha256(gradientFlat);
    emitSummary("repeat_" + std::to_string(repeat) + "_htp_backward_gradients",
                gradientFlat, report);
    emitComparison("repeat_" + std::to_string(repeat) + "_cpu_htp_logits",
                   cpuGradient.logits, htpGradient.logits, report);
    emitComparison("repeat_" + std::to_string(repeat) + "_cpu_htp_gradients",
                   flattenLanguageParameters(cpuGradient.gradients), gradientFlat,
                   report);
    const double htpNorm = gradientNorm(htpGradient.gradients);
    const float htpScale = checkpoint.config.clipThreshold > 0.0f &&
            std::isfinite(htpNorm) && htpNorm > 0.0
        ? float(std::min(1.0, double(checkpoint.config.clipThreshold) /
                               (htpNorm + 1.0e-6))) : 1.0f;
    Params htpNext, htpM, htpV;
    AdamOptimizerOutputs htpAdam;
    htpAdamAttempted = true;
    const bool adamOk = executeLanguageAdam(
        runtime, parameters, htpGradient.gradients, firstMoment, secondMoment,
        replayLearningRate, int(checkpoint.nextOptimizerStep),
        htpScale, htpNext, htpM, htpV, &htpAdam, executionError);
    report << "repeat_" << repeat << "_htp_adam_success="
           << (adamOk ? "true" : "false") << '\n';
    if (!adamOk) {
      htpAdamSuccess = false;
      report << "repeat_" << repeat << "_htp_adam_error=" << executionError << '\n';
      continue;
    }
    if (repeat == 0) {
      htpGradientFirst = htpGradient;
      htpGradientFirstValid = true;
      htpNextFirst = htpNext;
      htpMFirst = htpM;
      htpVFirst = htpV;
      htpAdamFirstOk = true;
    }
    emitSummary("repeat_" + std::to_string(repeat) + "_htp_adam_parameters",
                flattenLanguageParameters(htpNext), report);
  }
  // CPU/HTP attribution from the identical checkpoint state: all four
  // (CPU|HTP gradient) x (CPU|HTP Adam) combinations, plus the one-step
  // continuation loss each path produces. htpGradientFirst from repeat 0 is
  // deterministic (htp_repeat_deterministic evidence above).
  if (htpGradientFirstValid && htpAdamFirstOk) {
    const double firstHtpNorm = gradientNorm(htpGradientFirst.gradients);
    const float firstHtpScale =
        checkpoint.config.clipThreshold > 0.0f && std::isfinite(firstHtpNorm) &&
                firstHtpNorm > 0.0
            ? float(std::min(1.0, double(checkpoint.config.clipThreshold) /
                                    (firstHtpNorm + 1.0e-6)))
            : 1.0f;
    const auto htpClipped = scaleLanguageParameters(htpGradientFirst.gradients,
                                                    firstHtpScale);
    const auto pathB = tiny_lm::adamUpdate(
        parameters, htpClipped, firstMoment, secondMoment, replayLearningRate,
        checkpoint.config.beta1, checkpoint.config.beta2,
        checkpoint.config.adamEpsilon, c1, c2);
    std::string twoByTwoError;
    Params pathC, pathCm, pathCv;
    AdamOptimizerOutputs pathCRaw;
    const bool pathCOk = executeLanguageAdam(
        runtime, parameters, cpuGradient.gradients, firstMoment, secondMoment,
        replayLearningRate, int(checkpoint.nextOptimizerStep), cpuScale,
        pathC, pathCm, pathCv, &pathCRaw, twoByTwoError);
    const auto flatParams = flattenLanguageParameters(parameters);
    const auto updateVector = [&](const Params &next) {
      const auto flatNext = flattenLanguageParameters(next);
      std::vector<float> delta(flatNext.size());
      for (size_t i = 0; i < delta.size(); ++i)
        delta[i] = flatNext[i] - flatParams[i];
      return delta;
    };
    const auto uA = updateVector(cpuUpdate.next);
    const auto uB = updateVector(pathB.next);
    const auto uD = updateVector(htpNextFirst);
    auto deltaL2 = [](const std::vector<float> &a, const std::vector<float> &b) {
      double s = 0;
      for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        s += d * d;
      }
      return std::sqrt(s);
    };
    auto vecL2 = [](const std::vector<float> &a) {
      double s = 0;
      for (float x : a) s += double(x) * double(x);
      return std::sqrt(s);
    };
    auto nextLoss = [&](const Params &next) {
      return tiny_lm::forwardBackward(config, checkpoint.input,
                                      checkpoint.target, next, 0.0f)
          .loss;
    };
    const std::vector<float> uC =
        pathCOk ? updateVector(pathC) : std::vector<float>();
    report << "two_by_two_cpu_grad_cpu_adam_update_l2=" << vecL2(uA) << '\n'
           << "two_by_two_htp_grad_cpu_adam_update_l2=" << vecL2(uB) << '\n'
           << "two_by_two_htp_grad_htp_adam_update_l2=" << vecL2(uD) << '\n'
           << "two_by_two_cpu_grad_htp_adam_execute_success="
           << (pathCOk ? "true" : "false") << '\n';
    if (pathCOk) {
      report << "two_by_two_cpu_grad_htp_adam_update_l2=" << vecL2(uC) << '\n'
             << "two_by_two_gradient_source_update_rel_l2="
             << deltaL2(uA, uB) / std::max(1.0e-12, vecL2(uA)) << '\n'
             << "two_by_two_optimizer_source_update_rel_l2="
             << deltaL2(uA, uC) / std::max(1.0e-12, vecL2(uA)) << '\n'
             << "two_by_two_full_htp_update_rel_l2="
             << deltaL2(uA, uD) / std::max(1.0e-12, vecL2(uA)) << '\n';
    }
    report << "two_by_two_cpu_grad_cpu_adam_next_step_loss="
           << nextLoss(cpuUpdate.next) << '\n'
           << "two_by_two_htp_grad_cpu_adam_next_step_loss="
           << nextLoss(pathB.next) << '\n'
           << "two_by_two_htp_grad_htp_adam_next_step_loss="
           << nextLoss(htpNextFirst) << '\n';
    if (pathCOk)
      report << "two_by_two_cpu_grad_htp_adam_next_step_loss="
             << nextLoss(pathC) << '\n';
  }
  if (tapSet == TinyTransformerTrainingTapSet::LN2_SQUARE) {
    const std::vector<float> minimalInput{279.75f};
    const std::vector<float> minimalCpu{
        minimalInput.front() * minimalInput.front()};
    report << "minimal_reproducer_generator=single_value_279.75\n"
           << "minimal_reproducer_shape=1\n"
           << "minimal_reproducer_operation=ElementWiseMultiply(input,input)\n"
           << "minimal_reproducer_checkpoint_independent=true\n";
    emitSummary("minimal_reproducer_cpu_output", minimalCpu, report);
    std::string minimalError;
    bool minimalSuccess =
        runtime.prepareElementwiseSquare(1, minimalError);
    if (!minimalSuccess) {
      report << "minimal_reproducer_prepare_error=" << minimalError << '\n';
    } else {
      for (uint32_t repeat = 0; repeat < 3; ++repeat) {
        std::vector<float> minimalHtp;
        const bool executeOk = runtime.executeElementwiseSquare(
            minimalInput, minimalHtp, minimalError);
        const auto summary = first_nonfinite::summarize(minimalHtp);
        report << "minimal_reproducer_repeat_" << repeat
               << "_qnn_return=" << runtime.apiTrace().lastQnnResult << '\n'
               << "minimal_reproducer_repeat_" << repeat
               << "_execute_success=" << (executeOk ? "true" : "false")
               << '\n';
        emitSummary("minimal_reproducer_repeat_" + std::to_string(repeat) +
                        "_htp_output",
                    minimalHtp, report);
        if (!executeOk || summary.positiveInfinity != 1 ||
            summary.nan != 0 || summary.negativeInfinity != 0) {
          minimalSuccess = false;
          report << "minimal_reproducer_repeat_" << repeat
                 << "_error=" << minimalError << '\n';
        }
      }
    }
    report << "minimal_reproducer_success="
           << (minimalSuccess ? "true" : "false") << '\n';
  }
  const bool deterministic = !htpHashes.empty() &&
      std::all_of(htpHashes.begin() + 1, htpHashes.end(),
                  [&](const std::string &hash) { return hash == htpHashes.front(); });
  report << "htp_forward_backward_success="
         << (htpForwardBackwardSuccess ? "true" : "false") << '\n'
         << "htp_adam_attempted=" << (htpAdamAttempted ? "true" : "false")
         << '\n'
         << "htp_adam_success="
         << (htpAdamAttempted ? (htpAdamSuccess ? "true" : "false")
                              : "not_run")
         << '\n'
         << "htp_repeat_hashes=";
  for (size_t index = 0; index < htpHashes.size(); ++index)
    report << (index ? "," : "") << htpHashes[index];
  report << "\nhtp_repeat_deterministic="
         << (deterministic ? "true" : "false") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}

std::string runNicopediaHtpGeneration(
    const tiny_lm::Config &config, const std::string &checkpointPath,
    const std::string &promptPath, const TrainingConfig &trainingConfig,
    const nicopedia_gen::GenerateConfig &generateConfig,
    const LogSink &progress) {
  return nicopediaHtpGeneration(config, checkpointPath, promptPath,
                                trainingConfig, generateConfig, progress);
}

std::string runNicopediaHtpDivergenceLocalization(
    const tiny_lm::Config &config, const std::string &checkpointPath,
    const nicopedia_gen::GenerateConfig &generateConfig,
    const LogSink &progress) {
  return nicopediaHtpDivergenceLocalization(config, checkpointPath,
                                            generateConfig, progress);
}
} // namespace phonelm::qnn
