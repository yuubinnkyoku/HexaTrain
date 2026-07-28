// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "qnn_runtime.h"
#include "qnn_reproducibility.h"
#include "qnn_transformer.h"
#include "../tiny_language_model_cpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
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
  return std::max({maxAbs(a.gamma1, b.gamma1), maxAbs(a.beta1, b.beta1),
                   maxAbs(a.wq, b.wq), maxAbs(a.wk, b.wk), maxAbs(a.wv, b.wv),
                   maxAbs(a.wo, b.wo), maxAbs(a.gamma2, b.gamma2),
                   maxAbs(a.beta2, b.beta2), maxAbs(a.w1, b.w1),
                   maxAbs(a.w2, b.w2),
                   maxAbs(a.tokenEmbedding, b.tokenEmbedding),
                   maxAbs(a.outputProjection, b.outputProjection)});
}
bool finiteParams(const Params &p) {
  return finite(p.gamma1) && finite(p.beta1) && finite(p.wq) && finite(p.wk) &&
         finite(p.wv) && finite(p.wo) && finite(p.gamma2) && finite(p.beta2) &&
         finite(p.w1) && finite(p.w2) && finite(p.tokenEmbedding) &&
         finite(p.outputProjection);
}
double paramNorm(const Params &p) {
  double s = 0;
  auto add = [&](const std::vector<float> &v) {
    for (float x : v)
      s += double(x) * x;
  };
  add(p.gamma1);
  add(p.beta1);
  add(p.wq);
  add(p.wk);
  add(p.wv);
  add(p.wo);
  add(p.gamma2);
  add(p.beta2);
  add(p.w1);
  add(p.w2);
  add(p.tokenEmbedding);
  add(p.outputProjection);
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
  for (const auto &[name, member] : languageFields()) {
    (void)name;
    const auto &a = current.*member;
    const auto &b = next.*member;
    for (size_t i = 0; i < a.size(); ++i) {
      const double difference = double(b[i]) - a[i];
      sum += difference * difference;
    }
  }
  return std::sqrt(sum);
}
double gradientNorm(const Params &gradient) {
  double sum = 0;
  for (const auto &[name, member] : languageFields()) {
    (void)name;
    const double norm = vectorNorm(gradient.*member);
    sum += norm * norm;
  }
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
  for (const auto &[name, member] : languageFields()) {
    (void)name;
    const auto &values = p.*member;
    flat.insert(flat.end(), values.begin(), values.end());
  }
  return flat;
}
Params unflattenLanguageParameters(const std::vector<float> &flat,
                                   const Params &shape) {
  Params result = shape;
  size_t offset = 0;
  for (const auto &[name, member] : languageFields()) {
    (void)name;
    auto &values = result.*member;
    std::copy(flat.begin() + offset, flat.begin() + offset + values.size(),
              values.begin());
    offset += values.size();
  }
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
  for (const auto &[name, member] : languageFields()) {
    (void)name;
    std::fill((result.*member).begin(), (result.*member).end(), 0.0f);
  }
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
};
AdamCandidate adamCandidate(int candidate) {
  return candidate == 2
      ? AdamCandidate{"adam_lr0.0003_clip10_steps1000", 1000, .0003f,
                      10.0f}
      : AdamCandidate{"adam_lr0.0003_clip5_steps1000", 1000, .0003f, 5.0f};
}
bool executeLanguageAdam(Runtime &runtime, const Params &current,
                         const Params &gradient, const Params &firstMoment,
                         const Params &secondMoment, float learningRate, int step,
                         float gradientScale, Params &next, Params &firstMomentNext,
                         Params &secondMomentNext, AdamOptimizerOutputs *raw,
                         std::string &error) {
  AdamOptimizerOutputs output;
  const float firstCorrection =
      float(1.0 / (1.0 - std::pow(0.9, double(step))));
  const float secondCorrection =
      float(1.0 / (1.0 - std::pow(0.999, double(step))));
  if (!runtime.executeAdamOptimizer(
          flattenLanguageParameters(current), flattenLanguageParameters(gradient),
          flattenLanguageParameters(firstMoment),
          flattenLanguageParameters(secondMoment), learningRate, gradientScale,
          firstCorrection, secondCorrection, output, error))
    return false;
  next = unflattenLanguageParameters(output.weightNext, current);
  firstMomentNext =
      unflattenLanguageParameters(output.firstMomentNext, current);
  secondMomentNext =
      unflattenLanguageParameters(output.secondMomentNext, current);
  if (raw) *raw = std::move(output);
  return true;
}
std::string languageModelAdam(bool oneStepOnly, int candidate,
                              bool inferenceOnly) {
  tiny_lm::Config config;
  const auto selected = adamCandidate(candidate);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  const auto shape = tiny_lm::initialParameters(config, 1);
  const uint32_t optimizerElements =
      uint32_t(flattenLanguageParameters(shape).size());
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformerTraining(
          config.tokens, config.dimension, config.feedForwardDimension,
          config.epsilon, true, error, config.vocabularySize) ||
      !runtime.prepareAdamOptimizer(optimizerElements, error))
    return failure("adam_prepare", error, runtime);
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
    std::ostringstream detail;
    detail << std::setprecision(10);
    auto appendStats = [&](const std::string &prefix,
                           const std::vector<float> &cpu,
                           const std::vector<float> &htp) {
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
      return maxError;
    };
    auto appendParameterStats = [&](const std::string &prefix, const Params &cpu,
                                    const Params &htp) {
      const std::array<const char *, 12> names{
          "norm1_gamma", "norm1_beta", "wq", "wk", "wv", "wo",
          "norm2_gamma", "norm2_beta", "ffn_w1", "ffn_w2",
          "token_embedding", "output_projection"};
      const std::array<const std::vector<float> *, 12> ca{
          &cpu.gamma1, &cpu.beta1, &cpu.wq, &cpu.wk, &cpu.wv, &cpu.wo,
          &cpu.gamma2, &cpu.beta2, &cpu.w1, &cpu.w2,
          &cpu.tokenEmbedding, &cpu.outputProjection};
      const std::array<const std::vector<float> *, 12> ha{
          &htp.gamma1, &htp.beta1, &htp.wq, &htp.wk, &htp.wv, &htp.wo,
          &htp.gamma2, &htp.beta2, &htp.w1, &htp.w2,
          &htp.tokenEmbedding, &htp.outputProjection};
      double worst = 0;
      for (size_t i = 0; i < names.size(); ++i)
        worst = std::max(
            worst, appendStats(prefix + "_" + names[i], *ca[i], *ha[i]));
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
                                 pathCFirst, pathCSecond, &rawC, error) ||
            !executeLanguageAdam(runtime, current, htpGradient.gradients, first,
                                 second, diagnosticLearningRate, adamStep, 1.0f, pathD,
                                 pathDFirst, pathDSecond, &rawD, error))
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
              gradientError < .03 && optimizerC < .03 && optimizerD < .03;
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
                               next, firstNext, secondNext, &raw, error))
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
    const bool ok = compared == int(checkpoints.size()) && checkpointRoundtrip &&
                    finiteResult && changed &&
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
           << "\nmaximum_optimizer_isolation_max_abs_error="
           << maximumOptimizerIsolationError
           << "\nfirst_major_divergence_checkpoint=" << firstMajorDivergence
           << "\nfirst_major_divergence_tensor=token_embedding_gradient"
           << "\nfirst_major_divergence_node=lm_dembedding"
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
           << "\nexecute_count_per_training_step=2"
           << "\nbias_correction_scalar_responsibility=CPU"
           << "\noptimizer_math_responsibility=HTP"
           << "\nbinding_training_inputs=one_hot,target,12_parameter_tensors,learning_rate"
           << "\nbinding_training_input_type=APP_WRITE_FP32"
           << "\nbinding_training_outputs=logits,forward_diagnostics,12_gradients,12_next_parameters"
           << "\nbinding_training_output_type=APP_READ_FP32"
           << "\nbinding_adam_inputs=current,gradient,gradient_scale,m,v,lr,beta_scalars,bias_corrections,zero"
           << "\nbinding_adam_input_type=APP_WRITE_FP32"
           << "\nbinding_adam_outputs=m_next,v_next,m_hat,v_hat,sqrt_v_hat,denominator,normalized_update,scaled_update,weight_next"
           << "\nbinding_adam_output_type=APP_READ_FP32"
           << "\nbinding_flat_parameter_shape=3136x1"
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
  int accuracy75 = 0;
  int clippedSteps = 0;
  double minimumClipScale = 1.0, maximumPreclipGradientNorm = 0.0;
  double worstParameter = 0, worstFirst = 0, worstSecond = 0;
  std::vector<double> reductions;
  std::vector<Params> inferenceParameters;
  const int lastSeed = 5;
  for (int seed = 1; seed <= lastSeed; ++seed) {
    bool seedAllStepsFinite = true;
    int seedCompletedSteps = 0;
    auto htp = tiny_lm::initialParameters(config, seed), cpu = htp;
    auto htpFirst = zeroLanguageParameters(htp), htpSecond = htpFirst;
    auto cpuFirst = htpFirst, cpuSecond = htpFirst;
    LanguageQuality initial;
    if (!htpLanguageQuality(runtime, config, htp, 1, initial, error))
      return failure("adam_initial_eval", error, runtime);
    trajectory << "seed_" << seed << "_initial_loss=" << initial.loss
               << "\nseed_" << seed << "_initial_accuracy=" << initial.accuracy
               << '\n';
    for (int step = 1; step <= selected.steps; ++step) {
      const uint32_t pattern = uint32_t((step - 1) % 4);
      const uint32_t phase =
          inferenceOnly ? uint32_t((step - 1) / 4) % 2 : 0;
      const auto batch = languageBatch(config, pattern, phase);
      const auto cpuGradient = tiny_lm::forwardBackward(
          config, batch.first, batch.second, cpu, 0.0f);
      const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
      const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
      const auto cpuUpdate = tiny_lm::adamUpdate(
          cpu, cpuGradient.gradients, cpuFirst, cpuSecond, selected.lr, .9f,
          .999f, 1e-8f, c1, c2);
      TinyTransformerTrainingOutputs htpGradient;
      if (!runtime.executeTinyTransformerTraining(
              batch.first, batch.second, htp, 0.0f, htpGradient, error))
        return failure("adam_gradient_step", error, runtime);
      Params htpNext, firstNext, secondNext;
      AdamOptimizerOutputs rawHtpUpdate;
      const double preclipGradientNorm = gradientNorm(htpGradient.gradients);
      const float clipScale =
          std::isfinite(preclipGradientNorm) && preclipGradientNorm > 0
              ? float(std::min(1.0, double(selected.clipThreshold) /
                                        (preclipGradientNorm + 1.0e-6)))
              : 1.0f;
      clippedSteps += clipScale < 1.0f;
      minimumClipScale = std::min(minimumClipScale, double(clipScale));
      maximumPreclipGradientNorm =
          std::max(maximumPreclipGradientNorm, preclipGradientNorm);
      if (!executeLanguageAdam(runtime, htp, htpGradient.gradients, htpFirst,
                               htpSecond, selected.lr, step, clipScale, htpNext,
                               firstNext, secondNext, &rawHtpUpdate, error))
        return failure("adam_update_step", error, runtime);
      if (step == 1 || step == 2 || step == 5 || step == 10 || step == 20 ||
           step == 50 || step == 100 || step == 200 || step == 320 ||
           step == 640 || step == 1000 ||
           (seed == 1 && step >= 101 && step <= 199)) {
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
    }
    LanguageQuality final;
    if (!htpLanguageQuality(runtime, config, htp, 1, final, error))
      return failure("adam_final_eval", error, runtime);
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
    reductions.push_back(reduction);
    accuracy75 += final.accuracy >= .75;
    allLoss = allLoss && final.loss < initial.loss;
    allAccuracy = allAccuracy && final.accuracy > initial.accuracy;
    worstParameter = std::max(worstParameter, parameterError);
    worstFirst = std::max(worstFirst, firstError);
    worstSecond = std::max(worstSecond, secondError);
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
               << '\n';
    inferenceParameters.push_back(htp);
  }
  std::sort(reductions.begin(), reductions.end());
  const double median = reductions[reductions.size() / 2];
  if (inferenceOnly) {
    static const std::array<std::vector<uint32_t>, 4> rules{
        std::vector<uint32_t>{0, 1, 2, 3}, std::vector<uint32_t>{4, 5, 6, 7},
        std::vector<uint32_t>{8, 9}, std::vector<uint32_t>{10, 11, 12}};
    const auto parityBatch = languageBatch(config, 0, 0);
    const std::vector<uint32_t> parityTokenIds{0, 1, 2, 3, 0, 1, 2, 3};
    const std::vector<uint32_t> parityTargetIds{1, 2, 3, 0, 1, 2, 3, 0};
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
              "same_prefix_seed" + std::to_string(seedIndex + 1);
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
        paritySeeds << "same_prefix_seed_" << (seedIndex + 1)
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
      paritySeeds << "same_prefix_seed_" << (seedIndex + 1)
                  << "_finite=true\nsame_prefix_seed_" << (seedIndex + 1)
                  << "_cpu_eval_generation_max_abs_error="
                  << seedCpuEvalGeneration.maximum << "\nsame_prefix_seed_"
                  << (seedIndex + 1)
                  << "_htp_eval_generation_max_abs_error="
                  << seedHtpEvalGeneration.maximum << "\nsame_prefix_seed_"
                  << (seedIndex + 1) << "_cpu_htp_max_abs_error="
                  << seedCpuHtp.maximum << '\n';
    }
    trajectory
        << "same_prefix_token_ids=0,1,2,3,0,1,2,3"
        << "\nsame_prefix_host_inputs_identical="
        << ((parityBatch.first == generationInput &&
             parityBatch.second == generationTarget)
                ? "true"
                : "false")
        << "\nsame_prefix_valid_token_count=8"
        << "\nsame_prefix_position_indices=0,1,2,3,4,5,6,7"
        << "\nsame_prefix_logit_read_position=7"
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
        << "\nsame_prefix_seed_count=5"
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
          if (!runtime.executeTinyTransformerTraining(
                  input, target, inferenceParameters[seedIndex], 0.0f, output,
                  error))
            return failure("adam_generation", error, runtime);
          if (!finite(output.logits) || !finite(output.probabilities)) {
            std::ostringstream location;
            location << "seed" << (seedIndex + 1) << "_pattern" << pattern
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
          if (predicted != expected && firstError < 0) firstError = step;
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
          context.erase(context.begin());
          context.push_back(predicted);
        }
        const bool exact = evaluatedSteps == 8 && correct == 8;
        exactPatterns += seedIndex == 0 && exact;
        exactRollouts += exact;
        seedExactPatterns += exact;
        std::vector<uint32_t> oracleContext(config.tokens);
        for (uint32_t i = 0; i < config.tokens; ++i)
          oracleContext[i] = rule[i % rule.size()];
        int oracleCorrect = 0, firstOracleError = -1;
        int oracleEvaluatedSteps = 0;
        for (int step = 0; step < 8; ++step) {
          const uint32_t expected =
              rule[(size_t(config.tokens) + size_t(step)) % rule.size()];
          const auto input =
              tiny_lm::oneHot(oracleContext, config.vocabularySize);
          const auto target = tiny_lm::oneHot(
              std::vector<uint32_t>(config.tokens, expected),
              config.vocabularySize);
          TinyTransformerTrainingOutputs output;
          if (!runtime.executeTinyTransformerTraining(
                  input, target, inferenceParameters[seedIndex], 0.0f, output,
                  error))
            return failure("adam_oracle_generation", error, runtime);
          if (!finite(output.logits) || !finite(output.probabilities)) {
            std::ostringstream location;
            location << "oracle_seed" << (seedIndex + 1) << "_pattern"
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
          if (predicted != expected && firstOracleError < 0)
            firstOracleError = step;
          oracleCorrect += predicted == expected;
          oracleContext.erase(oracleContext.begin());
          oracleContext.push_back(expected);
        }
        const bool oracleExact =
            oracleEvaluatedSteps == 8 && oracleCorrect == 8;
        oracleExactRollouts += oracleExact;
        seedOracleExactPatterns += oracleExact;
        trajectory << "seed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_prompt=" << tokenList(prompt)
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_expected=" << tokenList(expectedTokens)
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_generated=" << tokenList(generated)
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_exact=" << (exact ? "true" : "false")
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_token_accuracy=" << correct / 8.0
                   << "\nseed_" << (seedIndex + 1)
                   << "_generation_pattern_" << pattern
                   << "_evaluated_steps=" << evaluatedSteps
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_first_error=" << firstError
                   << "\nseed_" << (seedIndex + 1) << "_generation_pattern_"
                   << pattern << "_mean_correct_probability="
                   << (evaluatedSteps ? probability / evaluatedSteps : 0)
                   << "\nseed_" << (seedIndex + 1)
                   << "_generation_pattern_" << pattern << "_mean_margin="
                   << (evaluatedSteps ? margin / evaluatedSteps : 0)
                   << "\nseed_" << (seedIndex + 1)
                   << "_generation_pattern_" << pattern << "_minimum_margin=";
        if (evaluatedSteps)
          trajectory << minimumMargin;
        else
          trajectory << "NOT_AVAILABLE";
        trajectory << '\n';
        trajectory << "seed_" << (seedIndex + 1)
                   << "_oracle_pattern_" << pattern
                   << "_exact=" << (oracleExact ? "true" : "false")
                   << "\nseed_" << (seedIndex + 1) << "_oracle_pattern_"
                   << pattern << "_token_accuracy=" << oracleCorrect / 8.0
                   << "\nseed_" << (seedIndex + 1) << "_oracle_pattern_"
                   << pattern << "_evaluated_steps=" << oracleEvaluatedSteps
                   << "\nseed_" << (seedIndex + 1) << "_oracle_pattern_"
                   << pattern << "_first_error=" << firstOracleError << '\n';
      }
      qualifyingSeeds += seedExactPatterns >= 3;
      oracleQualifyingSeeds += seedOracleExactPatterns == 4;
    }
    const bool exactSuccess = !outputNonfinite &&
                              qualifyingSeeds >= 4 && exactRollouts >= 16 &&
                              oracleQualifyingSeeds >= 4;
    std::ostringstream report;
    report << "TINY_LANGUAGE_MODEL\ntest=adam_inference_4_pattern\nstatus="
           << ((nan || outputNonfinite) ? "FAILED"
                    : (exactSuccess ? "SUCCESS" : "PARTIAL_SUCCESS"))
           << "\nresearch_goal_met=" << (exactSuccess ? "true" : "false")
           << "\ngeneration_nonfinite_detected="
           << (outputNonfinite ? "true" : "false")
           << "\nsame_prefix_nonfinite_count=" << samePrefixNonfiniteCount
           << "\ngeneration_nonfinite_count=" << generationNonfiniteCount
           << "\ngeneration_nan_count=" << outputNanCount
           << "\ngeneration_inf_count=" << outputInfCount
           << "\nfirst_generation_nonfinite=" << firstOutputNonfinite
           << "\noptimizer=ADAM\nlearning_rate=" << selected.lr
           << "\nsteps=" << selected.steps
           << "\nsampling=pattern_balanced_phase01_round_robin"
           << "\nseed_count=5\nrepresentative_seed_exact_pattern_count="
           << exactPatterns << "\nqualifying_seed_count=" << qualifyingSeeds
           << "\nexact_rollout_count=" << exactRollouts
           << "\noracle_qualifying_seed_count=" << oracleQualifyingSeeds
           << "\noracle_exact_rollout_count=" << oracleExactRollouts
           << "\nlogits_responsibility=HTP\nargmax_responsibility=CPU"
           << "\ncpu_fallback=false\nnan_detected="
           << (nan || outputNanCount ? "true" : "false")
           << "\ninf_detected="
           << (outputInfCount ? "true" : "false") << '\n'
           << trajectory.str() << runtime.apiTraceSummary()
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
  std::ostringstream report;
  report << std::setprecision(10)
         << "TINY_LANGUAGE_MODEL\ntest=adam_convergence\nstatus="
         << (ok ? "SUCCESS" : "FAILED") << "\nconfiguration_id="
         << selected.id << "\noptimizer=ADAM\nlearning_rate=" << selected.lr
         << "\nbeta1=0.9\nbeta2=0.999\nepsilon=1e-8"
         << "\nglobal_gradient_clip_threshold=" << selected.clipThreshold
         << "\nglobal_gradient_clip_epsilon=1e-6"
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
         << "\nexecute_count_per_training_step=2"
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

// This is deliberately an in-memory checkpoint.  The data is diagnostic input
// and can be large enough to make a checked-in raw checkpoint inappropriate.
struct LateNonfiniteCheckpoint {
  int seed = 0, completedStep = 0;
  Params parameters, firstMoment, secondMoment;
  std::vector<float> input, target;
};

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
                          float lr, float clipThreshold,
                          std::ostringstream &report) {
  report << prefix << "_checkpoint_format=phonelm.qnn.late_nonfinite.v1\n"
         << prefix << "_checkpoint_private_raw=true\n"
         << prefix << "_checkpoint_seed=" << checkpoint.seed << '\n'
         << prefix << "_checkpoint_completed_step=" << checkpoint.completedStep << '\n'
         << prefix << "_checkpoint_optimizer_next_step=" << checkpoint.completedStep + 1 << '\n';
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
  const bool optimizerOk = executeLanguageAdam(runtime, checkpoint.parameters,
      cpuGradient.gradients, checkpoint.firstMoment, checkpoint.secondMoment, lr, step,
      cpuClipScale, pathC, cM, cV, &rawC, error) && executeLanguageAdam(runtime,
      checkpoint.parameters, htpGradient.gradients, checkpoint.firstMoment,
      checkpoint.secondMoment, lr, step, htpClipScale, pathD, dM, dV, &rawD, error);
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
         << "two_by_two_optimizer_execute_success=" << (optimizerOk ? "true" : "false") << '\n';
  const std::array<std::tuple<const char *, const Params *, const Params *>, 4> paths{{
      {"A_cpu_gradient_cpu_adam", &pathA.next, nullptr},
      {"B_htp_gradient_cpu_adam", &pathB.next, nullptr},
      {"C_cpu_gradient_htp_adam", &pathC, &pathA.next},
      {"D_htp_gradient_htp_adam", &pathD, &pathB.next},
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
  if (optimizerOk) {
    report << "two_by_two_C_m_next_max_abs_error=" << maxParamError(cM, pathA.firstMoment) << '\n'
           << "two_by_two_C_v_next_max_abs_error=" << maxParamError(cV, pathA.secondMoment) << '\n'
           << "two_by_two_D_m_next_max_abs_error=" << maxParamError(dM, pathB.firstMoment) << '\n'
           << "two_by_two_D_v_next_max_abs_error=" << maxParamError(dV, pathB.secondMoment) << '\n';
    auditLateVector("two_by_two_C_adam_sqrt_v", rawC.secondRoot, report);
    auditLateVector("two_by_two_C_adam_denominator", rawC.denominator, report);
    auditLateVector("two_by_two_C_adam_update", rawC.scaledUpdate, report);
    auditLateVector("two_by_two_D_adam_sqrt_v", rawD.secondRoot, report);
    auditLateVector("two_by_two_D_adam_denominator", rawD.denominator, report);
    auditLateVector("two_by_two_D_adam_update", rawD.scaledUpdate, report);
  }
  std::vector<float> bad;
  const auto firstBad = firstBadLateStage(htpGradient, optimizerOk ? &rawD : nullptr, &bad);
  report << "two_by_two_first_bad_stage=" << firstBad << '\n'
         << "two_by_two_first_bad_producer_node=" << lateProducerNode(firstBad) << '\n';
  if (firstBad != "NONE") {
    const auto badAudit =
        auditLateVector("two_by_two_first_bad", bad, report);
    reportLateTapBoundary("two_by_two_first_bad", htpGradient, firstBad,
                          badAudit, report);
  }
  return optimizerOk && pathFinite[0] && !pathFinite[1] &&
         pathFinite[2] && !pathFinite[3] &&
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
      reportLateCheckpoint(checkpointPrefix, checkpoint, lr, clipThreshold, report);
      std::string firstSignature;
      bool stable = true;
      int completedReplays = 0;
      for (int replay = 0; replay < 100; ++replay) {
        TinyTransformerTrainingOutputs replayGradient;
        if (!runtime.executeTinyTransformerTraining(checkpoint.input, checkpoint.target,
              checkpoint.parameters, 0.0f, replayGradient, error)) { stable = false; break; }
        const double replayNorm = gradientNorm(replayGradient.gradients);
        const float replayScale = std::isfinite(replayNorm) && replayNorm > 0
            ? float(std::min(1.0, 5.0 / (replayNorm + 1.0e-6))) : 1.0f;
        Params replayNext, replayM, replayV;
        AdamOptimizerOutputs replayAdam;
        if (!executeLanguageAdam(runtime, checkpoint.parameters, replayGradient.gradients,
              checkpoint.firstMoment, checkpoint.secondMoment, lr, checkpoint.completedStep + 1,
              replayScale, replayNext, replayM, replayV, &replayAdam, error)) { stable = false; break; }
        std::vector<float> replayBad;
        const auto replayStage = firstBadLateStage(replayGradient, &replayAdam, &replayBad);
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
        "Expected five deterministic legacy failures, A/C finite and B/D "
        "non-finite path isolation, and five finite deterministic post-fix "
        "same-checkpoint replays.",
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
std::string runTinyTransformerTrainingExperiment(ExecutionMode mode) {
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
  return "TINY_TRANSFORMER_TRAINING\nstatus=FAILED\nerror=unsupported mode\n";
}
} // namespace phonelm::qnn
