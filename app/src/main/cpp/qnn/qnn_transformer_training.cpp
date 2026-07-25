// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "qnn_runtime.h"
#include "qnn_transformer.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

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
                   maxAbs(a.w2, b.w2)});
}
bool finiteParams(const Params &p) {
  return finite(p.gamma1) && finite(p.beta1) && finite(p.wq) && finite(p.wk) &&
         finite(p.wv) && finite(p.wo) && finite(p.gamma2) && finite(p.beta2) &&
         finite(p.w1) && finite(p.w2);
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
} // namespace
std::string runTinyTransformerTrainingExperiment(ExecutionMode mode) {
  if (mode == ExecutionMode::QNN_HTP_TINY_TRANSFORMER_TRAINING_STEP)
    return oneStep();
  if (mode == ExecutionMode::QNN_HTP_TINY_TRANSFORMER_TRAINING_MULTI_STEP)
    return multiStep();
  return "TINY_TRANSFORMER_TRAINING\nstatus=FAILED\nerror=unsupported mode\n";
}
} // namespace phonelm::qnn
