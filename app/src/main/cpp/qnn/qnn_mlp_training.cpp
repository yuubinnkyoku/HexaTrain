#include "qnn_mlp_training.h"
#include "../cpu_reference_training.h"
#include "qnn_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

namespace phonelm::qnn {
namespace {
using cpu::Matrix;
using Clock = std::chrono::steady_clock;
struct Data {
  Matrix x, y, tw1, tw2, w1, w2;
};
struct Pass {
  Matrix z1, h, p, dp, dw2, dh, dz1, dw1;
  float loss = 0;
};
float normal(std::mt19937_64 &g, float s) {
  std::normal_distribution<float> d(0, s);
  return d(g);
}
Data makeData(const TrainingConfig &c) {
  std::mt19937_64 g(c.seed);
  Data d{Matrix(c.sampleCount, c.dimension),
         Matrix(c.sampleCount, c.outputDimension),
         Matrix(c.dimension, c.hiddenDimension),
         Matrix(c.hiddenDimension, c.outputDimension),
         Matrix(c.dimension, c.hiddenDimension),
         Matrix(c.hiddenDimension, c.outputDimension)};
  std::uniform_real_distribution<float> u(-0.5f, 0.5f);
  for (float &v : d.x.values)
    v = u(g);
  for (float &v : d.tw1.values)
    v = normal(g, std::sqrt(2.0f / c.dimension));
  for (float &v : d.tw2.values)
    v = normal(g, std::sqrt(2.0f / (c.hiddenDimension + c.outputDimension)));
  for (float &v : d.w1.values)
    v = normal(g, std::sqrt(2.0f / c.dimension));
  for (float &v : d.w2.values)
    v = normal(g, std::sqrt(2.0f / (c.hiddenDimension + c.outputDimension)));
  auto z = cpu::matMul(d.x, d.tw1);
  for (float &v : z.values)
    v = std::max(0.0f, v);
  d.y = cpu::matMul(z, d.tw2);
  return d;
}
Pass pass(const Matrix &x, const Matrix &y, const Matrix &w1,
          const Matrix &w2) {
  Pass r{cpu::matMul(x, w1),          Matrix(x.rows, w1.columns),
         Matrix(x.rows, w2.columns),  Matrix(x.rows, w2.columns),
         Matrix(w2.rows, w2.columns), Matrix(x.rows, w2.rows),
         Matrix(x.rows, w1.columns),  Matrix(w1.rows, w1.columns)};
  for (size_t i = 0; i < r.z1.values.size(); ++i)
    r.h.values[i] = std::max(0.0f, r.z1.values[i]);
  r.p = cpu::matMul(r.h, w2);
  r.loss = cpu::meanSquaredError(r.p, y);
  float s = 2.0f / static_cast<float>(x.rows * y.columns);
  for (size_t i = 0; i < r.dp.values.size(); ++i)
    r.dp.values[i] = (r.p.values[i] - y.values[i]) * s;
  r.dw2 = cpu::matMul(cpu::transpose(r.h), r.dp);
  r.dh = cpu::matMul(r.dp, cpu::transpose(w2));
  for (size_t i = 0; i < r.dz1.values.size(); ++i)
    r.dz1.values[i] = r.h.values[i] > 0 ? r.dh.values[i] : 0;
  r.dw1 = cpu::matMul(cpu::transpose(x), r.dz1);
  return r;
}
bool finite(const Matrix &m) {
  return std::all_of(m.values.begin(), m.values.end(),
                     [](float v) { return std::isfinite(v); });
}
double maxAbs(const Matrix &a, const Matrix &b) {
  double v = 0;
  for (size_t i = 0; i < a.values.size(); ++i)
    v = std::max(v, std::abs(double(a.values[i]) - b.values[i]));
  return v;
}
double meanAbs(const Matrix &a, const Matrix &b) {
  double v = 0;
  for (size_t i = 0; i < a.values.size(); ++i)
    v += std::abs(double(a.values[i]) - b.values[i]);
  return v / a.values.size();
}
double maxRel(const Matrix &a, const Matrix &b) {
  double v = 0;
  for (size_t i = 0; i < a.values.size(); ++i) {
    double den = std::max(
        {1e-7, std::abs(double(a.values[i])), std::abs(double(b.values[i]))});
    v = std::max(v, std::abs(double(a.values[i]) - b.values[i]) / den);
  }
  return v;
}
Matrix batch(const Matrix &m, int start, int count) {
  Matrix o(count, m.columns);
  for (int r = 0; r < count; ++r)
    std::copy_n(m.values.data() + ((start + r) % m.rows) * m.columns, m.columns,
                o.values.data() + r * m.columns);
  return o;
}
std::string gradientCheck() {
  TrainingConfig c;
  c.batchSize = 2;
  c.sampleCount = 2;
  c.dimension = 4;
  c.hiddenDimension = 5;
  c.outputDimension = 3;
  c.seed = 20260710;
  auto d = makeData(c);
  auto a = pass(d.x, d.y, d.w1, d.w2);
  const float e = 1e-3f;
  double a1 = 0, r1 = 0, a2 = 0, r2 = 0;
  int skip = 0;
  for (float z : a.z1.values)
    if (std::abs(z) < 5 * e)
      ++skip;
  auto check = [&](Matrix &w, const Matrix &analytic, double &ma, double &mr) {
    for (size_t i = 0; i < w.values.size(); ++i) {
      float old = w.values[i];
      w.values[i] = old + e;
      float p = pass(d.x, d.y, d.w1, d.w2).loss;
      w.values[i] = old - e;
      float n = pass(d.x, d.y, d.w1, d.w2).loss;
      w.values[i] = old;
      double num = (p - n) / (2 * e), an = analytic.values[i],
             ab = std::abs(num - an),
             re = ab / std::max({1e-6, std::abs(num), std::abs(an)});
      ma = std::max(ma, ab);
      mr = std::max(mr, re);
    }
  };
  check(d.w1, a.dw1, a1, r1);
  check(d.w2, a.dw2, a2, r2);
  bool ok = a1 < 2e-4 && a2 < 2e-4;
  std::ostringstream s;
  s << std::setprecision(9)
    << "MLP_GRADIENT_CHECK\nexecution_mode=QNN_MLP_GRADIENT_CHECK\nepsilon="
    << e << "\nnear_relu_zero_threshold=" << 5 * e
    << "\nskipped_near_relu_zero_count=" << skip << "\ndw1_max_abs_error=" << a1
    << "\ndw1_max_relative_error=" << r1 << "\ndw2_max_abs_error=" << a2
    << "\ndw2_max_relative_error=" << r2
    << "\nstatus=" << (ok ? "SUCCESS" : "FAILED")
    << "\ncpu_fallback=false\nnan_detected=false\ninf_detected=false\nnan_inf="
       "false";
  return s.str();
}
std::string dxCheck(const TrainingConfig &) {
  Runtime rt;
  std::string e;
  if (!rt.initialize(QnnBackendKind::HTP, e))
    return "DX_CHECK\nexecution_mode=QNN_HTP_DX_CHECK\nstatus=FAILED\nfailed_"
           "api=initialize\nerror=" +
           e + "\ncpu_fallback=false";
  if (!rt.prepareInputGradientMatMul(2, 4, 3, e))
    return "DX_CHECK\nexecution_mode=QNN_HTP_DX_CHECK\nstatus=FAILED\nfailed_"
           "api=prepareInputGradientMatMul\nerror=" +
           e + "\ncpu_fallback=false";
  Matrix dp(2, 3), w(4, 3);
  for (size_t i = 0; i < dp.values.size(); ++i)
    dp.values[i] = float(i + 1) * .07f;
  for (size_t i = 0; i < w.values.size(); ++i)
    w.values[i] = float(int(i % 5) - 2) * .11f;
  Matrix ref = cpu::matMul(dp, cpu::transpose(w));
  std::vector<float> out;
  if (!rt.executeInputGradient(dp.values, w.values, out, e))
    return "DX_CHECK\nexecution_mode=QNN_HTP_DX_CHECK\nstatus=FAILED\nfailed_"
           "api=graphExecute\nerror=" +
           e + "\ncpu_fallback=false";
  Matrix got(2, 4, out);
  double ma = maxAbs(ref, got), me = meanAbs(ref, got), mr = maxRel(ref, got);
  bool ok = ma < 1e-4 && finite(got);
  std::ostringstream s;
  s << std::setprecision(9)
    << "DX_CHECK\nexecution_mode=QNN_HTP_DX_CHECK\nbatch_size=2\ninput_dim="
       "4\noutput_dim=3\ntranspose_in1=true\nmax_abs_error="
    << ma << "\nmean_abs_error=" << me << "\nmax_relative_error=" << mr
    << "\ngraph_create_result=0\ngraph_finalize_result=0\ngraph_execute_result="
       "0\ncpu_fallback=false\nnan_inf="
    << (!finite(got) ? "true" : "false")
    << "\nstatus=" << (ok ? "SUCCESS" : "FAILED") << "\n"
    << rt.diagnostics();
  return s.str();
}
std::string reluBackwardCheck() {
  Runtime rt;
  std::string e;
  const std::string prefix =
      "RELU_BACKWARD_CHECK\nexecution_mode=QNN_HTP_RELU_BACKWARD_CHECK\n";
  if (!rt.initialize(QnnBackendKind::HTP, e))
    return prefix + "status=FAILED\nfailed_api=initialize\nerror=" + e +
           "\ncpu_fallback=false";
  if (!rt.prepareReluBackward(2, 5, e))
    return prefix + "status=FAILED\nfailed_api=prepareReluBackward\nerror=" +
           e + "\ncpu_fallback=false\n" + rt.diagnostics();
  Matrix activation(2, 5), dh(2, 5), ref(2, 5);
  activation.values = {-1.0f, -0.1f, -1e-6f, 0.0f, 1e-6f,
                       0.1f,  1.0f,  -0.0f, -1e-4f, 1e-4f};
  dh.values = {0.25f, -0.5f, 0.75f, -1.0f, 1.25f,
               -1.5f, 1.75f, -2.0f, 2.25f, -2.5f};
  for (size_t k = 0; k < ref.values.size(); ++k)
    ref.values[k] = activation.values[k] > 0.0f ? dh.values[k] : 0.0f;
  std::vector<std::uint8_t> mask;
  std::vector<float> output;
  if (!rt.executeReluBackward(activation.values, dh.values, mask, output, e))
    return prefix + "status=FAILED\nfailed_api=graphExecute\nerror=" + e +
           "\ncpu_fallback=false\n" + rt.diagnostics();
  Matrix got(2, 5, output);
  bool maskMatches = mask.size() == ref.values.size();
  for (size_t k = 0; maskMatches && k < mask.size(); ++k)
    maskMatches = (mask[k] != 0) == (activation.values[k] > 0.0f);
  const double ma = maxAbs(ref, got), me = meanAbs(ref, got),
               mr = maxRel(ref, got);
  const bool ok = ma <= 1e-7 && maskMatches && finite(got);
  std::ostringstream out;
  out << std::setprecision(9) << prefix
      << "relu_backward_op_method=ElementWiseGreater+ElementWiseSelect\n"
      << "tensor_shape=2x5\nmask_datatype=BOOL_8\noutput_datatype=FLOAT_32\n"
      << "zero_boundary_rule=Z1_GT_0\nmax_abs_error=" << ma
      << "\nmean_abs_error=" << me << "\nmax_relative_error=" << mr
      << "\nmask_matches_cpu=" << (maskMatches ? "true" : "false")
      << "\ngraph_create_result=0\ngraph_finalize_result=0\n"
      << "graph_execute_result=0\ncpu_fallback=false\nnan_inf="
      << (!finite(got) ? "true" : "false") << "\nstatus="
      << (ok ? "SUCCESS" : "FAILED") << '\n'
      << rt.diagnostics();
  return out.str();
}
struct Summary {
  double min = 0, median = 0, mean = 0, sd = 0, p90 = 0, p95 = 0, max = 0;
};
Summary summarize(std::vector<double> v) {
  Summary r;
  if (v.empty())
    return r;
  std::sort(v.begin(), v.end());
  r.min = v.front();
  r.max = v.back();
  r.median = v[v.size() / 2];
  r.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double q = 0;
  for (double x : v)
    q += (x - r.mean) * (x - r.mean);
  r.sd = std::sqrt(q / v.size());
  auto pct = [&](double p) {
    return v[std::min(v.size() - 1, size_t(std::ceil(p * v.size()) - 1))];
  };
  r.p90 = pct(.90);
  r.p95 = pct(.95);
  return r;
}
void emitStats(std::ostringstream &s, const char *n,
               const std::vector<double> &v) {
  auto r = summarize(v);
  s << '\n'
    << n << "_min_us=" << r.min << '\n'
    << n << "_median_us=" << r.median << '\n'
    << n << "_mean_us=" << r.mean << '\n'
    << n << "_stddev_us=" << r.sd << '\n'
    << n << "_p90_us=" << r.p90 << '\n'
    << n << "_p95_us=" << r.p95 << '\n'
    << n << "_max_us=" << r.max;
}
std::string train(ExecutionMode mode, const TrainingConfig &c,
                  std::atomic_bool &stop) {
  const bool cpuOnly = mode == ExecutionMode::QNN_CPU_MLP_TRAINING;
  const bool fused = mode == ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD ||
                     mode ==
                         ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK;
  const bool htpBw = fused ||
                     mode == ExecutionMode::QNN_HTP_MLP_HTP_LINEAR_BACKWARD ||
                     mode == ExecutionMode::QNN_HTP_MLP_BENCHMARK;
  const bool fusedDiagnostic = fused && !c.benchmarkMode;
  auto d = makeData(c);
  const Matrix initialW1 = d.w1, initialW2 = d.w2;
  const float initial = pass(d.x, d.y, d.w1, d.w2).loss;
  Runtime rt;
  std::string e;
  const auto initializationStart = Clock::now();
  if (!cpuOnly &&
      (!rt.initialize(QnnBackendKind::HTP, e) ||
       !rt.prepareMlp(c.batchSize, c.dimension, c.hiddenDimension,
                      c.outputDimension, e, !fused) ||
       (fused && !rt.prepareMlpFusedBackward(fusedDiagnostic, e)) ||
       !rt.setMlpWeights(d.w1.values, d.w2.values, e)))
    return "MLP_TRAINING\nexecution_mode=" +
           std::string(executionModeName(mode)) +
           "\nstatus=FAILED\nfailed_api=HTP_prepare\nerror=" + e +
           "\ncpu_fallback=false";
  const double initializationUs =
      cpuOnly ? 0.0
              : std::chrono::duration<double, std::micro>(Clock::now() -
                                                          initializationStart)
                    .count();
  const int steps =
      c.epochs > 0 ? (c.sampleCount / c.batchSize) * c.epochs : c.steps;
  bool nan = false, dw1Nonzero = false, dw2Nonzero = false,
       dhNonzero = false, dz1Nonzero = false;
  double dw1err = 0, dw2err = 0, dherr = 0, dz1err = 0, dw1mean = 0,
         dw2mean = 0, dhmean = 0, dz1mean = 0, forwardErr = 0,
         updatedForwardErr = 0;
  Matrix lastX, lastY;
  std::vector<double> fullT, forwardT, lossDpT, backward2T, reluT, dw1T,
      optimizerT, updateT;
  for (int st = 0; st < steps && !stop.load(); ++st) {
    auto fullStart = Clock::now();
    Matrix x = batch(d.x, (st * c.batchSize) % c.sampleCount, c.batchSize),
           y = batch(d.y, (st * c.batchSize) % c.sampleCount, c.batchSize);
    lastX = x;
    lastY = y;
    Matrix h(c.batchSize, c.hiddenDimension), p(c.batchSize, c.outputDimension),
        dp(c.batchSize, c.outputDimension),
        dw2(c.hiddenDimension, c.outputDimension),
        dh(c.batchSize, c.hiddenDimension), dz1(c.batchSize, c.hiddenDimension),
        dw1(c.dimension, c.hiddenDimension);
    Pass ref;
    bool check = !c.benchmarkMode && (c.correctnessInterval <= 1 || st == 0 ||
                                      st % c.correctnessInterval == 0);
    if (cpuOnly) {
      ref = pass(x, y, d.w1, d.w2);
      h = ref.h;
      p = ref.p;
      dp = ref.dp;
      dw2 = ref.dw2;
      dh = ref.dh;
      dz1 = ref.dz1;
      dw1 = ref.dw1;
    } else {
      auto t = Clock::now();
      std::vector<float> hv, pv;
      if (!rt.executeMlpForward(x.values, hv, pv, e))
        return "MLP_TRAINING\nexecution_mode=" +
               std::string(executionModeName(mode)) +
               "\nstatus=FAILED\nfailed_api=forward_execute\nerror=" + e +
               "\ncpu_fallback=false";
      forwardT.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - t).count());
      h = Matrix(c.batchSize, c.hiddenDimension, hv);
      p = Matrix(c.batchSize, c.outputDimension, pv);
      t = Clock::now();
      float sc = 2.0f / (c.batchSize * c.outputDimension);
      for (size_t i = 0; i < dp.values.size(); ++i)
        dp.values[i] = (p.values[i] - y.values[i]) * sc;
      volatile float loss = cpu::meanSquaredError(p, y);
      (void)loss;
      lossDpT.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - t).count());
      if (check) {
        ref = pass(x, y, d.w1, d.w2);
        forwardErr =
            std::max(forwardErr, std::max(maxAbs(ref.h, h), maxAbs(ref.p, p)));
      }
      if (htpBw) {
        t = Clock::now();
        std::vector<float> w2v, dhv, dzv, w1v;
        std::vector<std::uint8_t> mask;
        if (fused) {
          if (!rt.executeMlpFusedBackward(x.values, h.values, dp.values, w2v,
                                          dhv, mask, dzv, w1v, e))
            return "MLP_TRAINING\nexecution_mode=" +
                   std::string(executionModeName(mode)) +
                   "\nstatus=FAILED\nfailed_api=fused_backward_execute\nerror=" +
                   e + "\ncpu_fallback=false\nhtp_relu_backward_used=false\n"
                       "htp_fused_backward_used=false";
          backward2T.push_back(
              std::chrono::duration<double, std::micro>(Clock::now() - t)
                  .count());
          dw2 = Matrix(c.hiddenDimension, c.outputDimension, w2v);
          dw1 = Matrix(c.dimension, c.hiddenDimension, w1v);
          if (fusedDiagnostic) {
            dh = Matrix(c.batchSize, c.hiddenDimension, dhv);
            dz1 = Matrix(c.batchSize, c.hiddenDimension, dzv);
          }
        } else {
          if (!rt.executeMlpSecondBackward(h.values, dp.values, w2v, dhv, e))
            return "MLP_TRAINING\nexecution_mode=" +
                   std::string(executionModeName(mode)) +
                   "\nstatus=FAILED\nfailed_api=backward2_execute\nerror=" + e +
                   "\ncpu_fallback=false";
          backward2T.push_back(
              std::chrono::duration<double, std::micro>(Clock::now() - t)
                  .count());
          dw2 = Matrix(c.hiddenDimension, c.outputDimension, w2v);
          dh = Matrix(c.batchSize, c.hiddenDimension, dhv);
          t = Clock::now();
          for (size_t i = 0; i < dz1.values.size(); ++i)
            dz1.values[i] = h.values[i] > 0 ? dh.values[i] : 0;
          reluT.push_back(
              std::chrono::duration<double, std::micro>(Clock::now() - t)
                  .count());
          t = Clock::now();
          if (!rt.executeMlpFirstBackward(x.values, dz1.values, w1v, e))
            return "MLP_TRAINING\nexecution_mode=" +
                   std::string(executionModeName(mode)) +
                   "\nstatus=FAILED\nfailed_api=backward1_execute\nerror=" + e +
                   "\ncpu_fallback=false";
          dw1T.push_back(
              std::chrono::duration<double, std::micro>(Clock::now() - t)
                  .count());
          dw1 = Matrix(c.dimension, c.hiddenDimension, w1v);
        }
        if (check) {
          dw1err = std::max(dw1err, maxAbs(ref.dw1, dw1));
          dw2err = std::max(dw2err, maxAbs(ref.dw2, dw2));
          dw1mean = std::max(dw1mean, meanAbs(ref.dw1, dw1));
          dw2mean = std::max(dw2mean, meanAbs(ref.dw2, dw2));
          if (!fused || fusedDiagnostic) {
            dherr = std::max(dherr, maxAbs(ref.dh, dh));
            dz1err = std::max(dz1err, maxAbs(ref.dz1, dz1));
            dhmean = std::max(dhmean, meanAbs(ref.dh, dh));
            dz1mean = std::max(dz1mean, meanAbs(ref.dz1, dz1));
          }
        }
      } else {
        auto t2 = Clock::now();
        dw2 = cpu::matMul(cpu::transpose(h), dp);
        dh = cpu::matMul(dp, cpu::transpose(d.w2));
        for (size_t i = 0; i < dz1.values.size(); ++i)
          dz1.values[i] = h.values[i] > 0 ? dh.values[i] : 0;
        dw1 = cpu::matMul(cpu::transpose(x), dz1);
        backward2T.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - t2)
                .count());
      }
    }
    dw1Nonzero = dw1Nonzero || std::any_of(dw1.values.begin(), dw1.values.end(),
                                           [](float v) { return v != 0.0f; });
    dw2Nonzero = dw2Nonzero || std::any_of(dw2.values.begin(), dw2.values.end(),
                                           [](float v) { return v != 0.0f; });
    dhNonzero = dhNonzero || std::any_of(dh.values.begin(), dh.values.end(),
                                         [](float v) { return v != 0.0f; });
    dz1Nonzero = dz1Nonzero ||
                 std::any_of(dz1.values.begin(), dz1.values.end(),
                             [](float v) { return v != 0.0f; });
    auto t = Clock::now();
    cpu::sgdUpdate(d.w1, dw1, c.learningRate);
    cpu::sgdUpdate(d.w2, dw2, c.learningRate);
    optimizerT.push_back(
        std::chrono::duration<double, std::micro>(Clock::now() - t).count());
    if (!cpuOnly) {
      t = Clock::now();
      if (!rt.setMlpWeights(d.w1.values, d.w2.values, e))
        return "MLP_TRAINING\nexecution_mode=" +
               std::string(executionModeName(mode)) +
               "\nstatus=FAILED\nfailed_api=weight_update\nerror=" + e +
               "\ncpu_fallback=false";
      updateT.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - t).count());
    }
    nan = nan || !finite(d.w1) || !finite(d.w2);
    fullT.push_back(
        std::chrono::duration<double, std::micro>(Clock::now() - fullStart)
            .count());
    if (nan)
      break;
  }
  Pass finalPass = pass(d.x, d.y, d.w1, d.w2);
  const float final = finalPass.loss;
  double predictionMax = maxAbs(finalPass.p, d.y);
  if (!cpuOnly && !lastX.values.empty()) {
    std::vector<float> h, p;
    if (!rt.executeMlpForward(lastX.values, h, p, e))
      return "MLP_TRAINING\nexecution_mode=" +
             std::string(executionModeName(mode)) +
             "\nstatus=FAILED\nfailed_api=updated_forward_execute\nerror=" + e +
             "\ncpu_fallback=false";
    auto ref = pass(lastX, lastY, d.w1, d.w2);
    updatedForwardErr =
        maxAbs(ref.p, Matrix(c.batchSize, c.outputDimension, p));
  }
  const bool ok =
      !nan && final < initial &&
      (!htpBw || (dw1err < 1e-2 && dw2err < 1e-2 && dherr < 1e-2)) &&
      updatedForwardErr < 1e-3;
  std::ostringstream s;
  s << std::setprecision(9)
    << "MLP_TRAINING\nexecution_mode=" << executionModeName(mode)
    << "\nseed=" << c.seed << "\nsample_count=" << c.sampleCount
    << "\nbatch_size=" << c.batchSize << "\ninput_dim=" << c.dimension
    << "\nhidden_dim=" << c.hiddenDimension
    << "\noutput_dim=" << c.outputDimension << "\nepochs=" << c.epochs
    << "\nsteps=" << steps << "\nlearning_rate=" << c.learningRate
    << "\ninitial_loss=" << initial << "\nfinal_loss=" << final
    << "\nruntime_initialization_us=" << initializationUs
    << "\nprediction_mse=" << final
    << "\nprediction_max_abs_error=" << predictionMax
    << "\nforward_max_abs_error=" << forwardErr
    << "\ndw1_max_abs_error=" << dw1err << "\ndw2_max_abs_error=" << dw2err
    << "\ndh_max_abs_error=" << dherr << "\ndz1_max_abs_error=" << dz1err
    << "\ndw1_mean_abs_error=" << dw1mean
    << "\ndw2_mean_abs_error=" << dw2mean << "\ndh_mean_abs_error=" << dhmean
    << "\ndz1_mean_abs_error=" << dz1mean
    << "\nupdated_forward_max_abs_error=" << updatedForwardErr
    << "\nforward_backend=" << (cpuOnly ? "CPU" : "HTP")
    << "\ndw2_backend=" << (htpBw ? "HTP" : "CPU")
    << "\ndh_backend=" << (htpBw ? "HTP" : "CPU")
    << "\nrelu_backward_backend=" << (fused ? "HTP" : "CPU")
    << "\ndw1_backend=" << (htpBw ? "HTP" : "CPU")
    << "\noptimizer_backend=CPU\nforward_graph_create_count="
    << (cpuOnly ? 0 : 1)
    << "\nforward_graph_finalize_count=" << (cpuOnly ? 0 : 1)
    << "\nforward_execute_count=" << (cpuOnly ? 0 : steps + 1)
    << "\ndw2_graph_create_count=" << (!cpuOnly && !fused ? 1 : 0)
    << "\ndh_graph_create_count=" << (!cpuOnly && !fused ? 1 : 0)
    << "\ndw1_graph_create_count=" << (!cpuOnly && !fused ? 1 : 0)
    << "\nfused_backward_graph_create_count=" << (fused ? 1 : 0)
    << "\nbackward_graph_finalize_count=" << (cpuOnly ? 0 : (fused ? 1 : 3))
    << "\ndw2_execute_count=" << (htpBw && !fused ? steps : 0)
    << "\ndh_execute_count=" << (htpBw && !fused ? steps : 0)
    << "\ndw1_execute_count=" << (htpBw && !fused ? steps : 0)
    << "\nfused_backward_execute_count=" << (fused ? steps : 0)
    << "\nbackward_execute_count_per_step=" << (fused ? 1 : (htpBw ? 3 : 0))
    << "\nw1_initial_bind_count=" << (cpuOnly ? 0 : 1)
    << "\nw2_initial_bind_count=" << (cpuOnly ? 0 : 1)
    << "\nw1_update_count=" << (cpuOnly ? 0 : steps)
    << "\nw2_update_count=" << (cpuOnly ? 0 : steps)
    << "\nw2_copies_per_step=" << (cpuOnly ? 0 : 1)
    << "\nw1_changed=" << (maxAbs(initialW1, d.w1) > 0 ? "true" : "false")
    << "\nw2_changed=" << (maxAbs(initialW2, d.w2) > 0 ? "true" : "false")
    << "\ndw2_nonzero=" << (dw2Nonzero ? "true" : "false")
    << "\ndh_nonzero=" << (dhNonzero ? "true" : "false")
    << "\ndw1_nonzero=" << (dw1Nonzero ? "true" : "false")
    << "\ndz1_nonzero=" << (dz1Nonzero ? "true" : "false")
    << "\nshared_w2_app_buffer=true\ncross_graph_w2_sync=true";
  emitStats(s, "forward", forwardT);
  emitStats(s, "loss_dp", lossDpT);
  emitStats(s, "second_layer_backward", backward2T);
  emitStats(s, "relu_backward", reluT);
  emitStats(s, "first_layer_dw", dw1T);
  emitStats(s, "optimizer", optimizerT);
  emitStats(s, "weight_update", updateT);
  emitStats(s, "full_step", fullT);
  s << "\ncpu_fallback=false\nhtp_linear_backward_used="
    << (htpBw ? "true" : "false")
    << "\nhtp_relu_backward_used=" << (fused ? "true" : "false")
    << "\nhtp_fused_backward_used=" << (fused ? "true" : "false")
    << "\nnan_detected=" << (nan ? "true" : "false")
    << "\ninf_detected=false\nnan_inf=" << (nan ? "true" : "false")
    << "\nstatus=" << (ok ? "SUCCESS" : "FAILED");
  if (!cpuOnly)
    s << '\n' << rt.diagnostics();
  return s.str();
}
std::string trainingOpsMicroCheck(ExecutionMode mode) {
  Runtime rt;
  std::string e;
  const char* modeName = executionModeName(mode);
  const std::string prefix = std::string("TRAINING_OPS_MICRO\nexecution_mode=") + modeName + "\n";
  if (!rt.initialize(QnnBackendKind::HTP, e))
    return prefix + "status=FAILED\nfailed_api=initialize\nerror=" + e + "\ncpu_fallback=false";
  if (!rt.prepareTrainingOpsMicro(2, 3, 2, 3, e))
    return prefix + "status=FAILED\nfailed_api=prepareTrainingOpsMicro\nerror=" + e + "\ncpu_fallback=false\n" + rt.diagnostics();
  Matrix p(2, 3), y(2, 3), w(2, 3), dw(2, 3);
  p.values = {0.25f, -0.5f, 0.75f, 1.0f, -1.25f, 1.5f};
  y.values = {-0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f};
  w.values = {0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f};
  dw.values = {-0.11f, 0.12f, -0.13f, 0.14f, -0.15f, 0.16f};
  const float lr = 0.125f;
  float loss = 0.0f;
  std::vector<float> dpv, nextv;
  if (!rt.executeTrainingOpsMicro(p.values, y.values, w.values, dw.values, lr,
                                  loss, dpv, nextv, e))
    return prefix + "status=FAILED\nfailed_api=graphExecute\nerror=" + e + "\ncpu_fallback=false\n" + rt.diagnostics();
  Matrix dp(2, 3), next(2, 3), dpRef(2, 3), nextRef(2, 3);
  dp.values = dpv; next.values = nextv;
  const float scale = 2.0f / 6.0f;
  for (size_t k = 0; k < p.values.size(); ++k)
    dpRef.values[k] = (p.values[k] - y.values[k]) * scale;
  for (size_t k = 0; k < w.values.size(); ++k)
    nextRef.values[k] = w.values[k] - lr * dw.values[k];
  const float lossRef = cpu::meanSquaredError(p, y);
  const double lossError = std::abs(double(loss) - lossRef);
  const double dpMax = maxAbs(dpRef, dp), dpMean = meanAbs(dpRef, dp);
  const double sgdMax = maxAbs(nextRef, next), sgdMean = meanAbs(nextRef, next);
  const bool finiteOk = std::isfinite(loss) && finite(dp) && finite(next);
  const bool ok = lossError < 1e-5 && dpMax < 5e-4 && sgdMax < 5e-4 && finiteOk;
  std::ostringstream s;
  s << std::setprecision(9) << prefix
    << "mse_op_method=ElementWiseSubtract+ElementWiseMultiply+ReduceMean\n"
    << "dp_op_method=ElementWiseMultiply\n"
    << "sgd_op_method=ElementWiseMultiply+ElementWiseSubtract\n"
    << "input_shape=2x3\nloss_output_shape=1x1\nreduction_axes=0,1\n"
    << "reduce_keep_dims=true\nscalar_broadcast=true\n"
    << "mse_cpu=" << lossRef << "\nmse_htp=" << loss
    << "\nmse_abs_error=" << lossError
    << "\ndp_max_abs_error=" << dpMax << "\ndp_mean_abs_error=" << dpMean
    << "\ndp_max_relative_error=" << maxRel(dpRef, dp)
    << "\nsgd_max_abs_error=" << sgdMax << "\nsgd_mean_abs_error=" << sgdMean
    << "\nsgd_max_relative_error=" << maxRel(nextRef, next)
    << "\ngraph_create_result=0\ngraph_finalize_result=0\ngraph_execute_result=0\n"
    << "loss_backend=HTP\ndp_backend=HTP\noptimizer_backend=HTP\n"
    << "cpu_fallback=false\nnan_inf=" << (finiteOk ? "false" : "true")
    << "\nstatus=" << (ok ? "SUCCESS" : "FAILED") << '\n' << rt.diagnostics();
  return s.str();
}

double checksum(const Matrix& m) {
  double v = 0.0;
  for (size_t k = 0; k < m.values.size(); ++k)
    v += double(m.values[k]) * double((k % 997) + 1);
  return v;
}
double l2(const Matrix& m) {
  double v = 0.0;
  for (float x : m.values) v += double(x) * x;
  return std::sqrt(v);
}

std::string trainFullStep(ExecutionMode mode, const TrainingConfig& c,
                          std::atomic_bool& stop) {
  const bool captureBenchmark =
      mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE;
  const bool executeFailure =
      mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE;
  const bool finalizeFailure =
      mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE;
  const bool benchmark = mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK ||
                         captureBenchmark || executeFailure || finalizeFailure ||
                         c.benchmarkMode;
  const bool diagnostic = !benchmark;
  auto d = makeData(c);
  Matrix cpuW1 = d.w1, cpuW2 = d.w2;
  std::vector<float> w1Current = d.w1.values, w2Current = d.w2.values;
  const Matrix initialW1 = d.w1, initialW2 = d.w2;
  const float initialLoss = pass(d.x, d.y, d.w1, d.w2).loss;
  Runtime rt;
  RuntimeOptions runtimeOptions;
  runtimeOptions.captureQnnCallback = !benchmark || captureBenchmark ||
                                      executeFailure || finalizeFailure;
  runtimeOptions.qnnLogLevel = runtimeOptions.captureQnnCallback ? 4 : 2;
  runtimeOptions.failGraphExecuteAt = executeFailure ? 37 : -1;
  runtimeOptions.failGraphFinalize = finalizeFailure;
  rt.setOptions(runtimeOptions);
  std::string e;
  const auto initStart = Clock::now();
  if (!rt.initialize(QnnBackendKind::HTP, e) ||
      !rt.prepareMlpFullStep(c.batchSize, c.dimension, c.hiddenDimension,
                            c.outputDimension, diagnostic, e))
    return "MLP_FULL_STEP\nexecution_mode=" + std::string(executionModeName(mode)) +
           "\nstatus=FAILED\nfailed_api=HTP_prepare\nerror=" + e +
           "\ncpu_fallback=false\nhtp_full_step_used=false\n" +
           rt.apiTraceSummary() + rt.qnnCallbackCaptureSummary() + rt.diagnostics();
  const double initializationUs =
      std::chrono::duration<double, std::micro>(Clock::now() - initStart).count();
  const int steps = c.epochs > 0 ? (c.sampleCount / c.batchSize) * c.epochs : c.steps;
  double hiddenError=0, predictionError=0, errorTensorError=0, lossError=0, dpError=0, dw2Error=0, dhError=0,
         dz1Error=0, dw1Error=0, w1NextError=0, w2NextError=0,
         optimizerW1Error=0, optimizerW2Error=0, trajectoryW1Error=0,
         trajectoryW2Error=0;
  uint64_t cpuMaskMismatch=0, internalMaskMismatch=0;
  double maskMismatchMaxAbsZ1=0;
  bool nan=false, dpNonzero=false, dw1Nonzero=false, dw2Nonzero=false,
       w1Changed=false, w2Changed=false, nextPredictionChanged=false;
  std::vector<double> fullTimes, handoffTimes, batchPreparationTimes;
  std::ostringstream trajectory;
  trajectory << std::setprecision(12)
             << "trajectory_0_w1_checksum=" << checksum(initialW1)
             << "\ntrajectory_0_w2_checksum=" << checksum(initialW2)
             << "\ntrajectory_0_w1_l2_norm=" << l2(initialW1)
             << "\ntrajectory_0_w2_l2_norm=" << l2(initialW2)
             << "\ntrajectory_0_cpu_htp_w1_max_abs_difference=0"
             << "\ntrajectory_0_cpu_htp_w2_max_abs_difference=0"
             << "\ntrajectory_0_loss=" << initialLoss << '\n';
  auto checkpoint = [&](int step) {
    static const int values[] = {1,2,5,10,20,50,100};
    if (step == steps) return true;
    return std::find(std::begin(values), std::end(values), step) != std::end(values);
  };
  std::vector<float> firstPrediction;
  MlpFullStepOutputs out;
  for (int st=0; st<steps && !stop.load(); ++st) {
    const auto fullStart=Clock::now();
    const auto batchStart=Clock::now();
    Matrix x=batch(d.x,(st*c.batchSize)%c.sampleCount,c.batchSize);
    Matrix y=batch(d.y,(st*c.batchSize)%c.sampleCount,c.batchSize);
    batchPreparationTimes.push_back(
        std::chrono::duration<double,std::micro>(Clock::now()-batchStart).count());
    Matrix currentW1, currentW2;
    Pass ref;
    if (diagnostic) {
      currentW1=Matrix(c.dimension,c.hiddenDimension,w1Current);
      currentW2=Matrix(c.hiddenDimension,c.outputDimension,w2Current);
      ref=pass(x,y,currentW1,currentW2);
    }
    if (!rt.executeMlpFullStep(x.values,y.values,w1Current,w2Current,
                               c.learningRate,out,e))
      return "MLP_FULL_STEP\nexecution_mode="+std::string(executionModeName(mode))+
             "\nstatus=FAILED\nfailed_api=full_step_execute\nerror="+e+
             "\ncpu_fallback=false\nhtp_full_step_used=false\n"+
             rt.apiTraceSummary()+rt.qnnCallbackCaptureSummary()+rt.diagnostics();
    Matrix nextW1(c.dimension,c.hiddenDimension,out.w1Next);
    Matrix nextW2(c.hiddenDimension,c.outputDimension,out.w2Next);
    if (diagnostic) {
      Matrix gotH(c.batchSize,c.hiddenDimension,out.hidden);
      Matrix gotP(c.batchSize,c.outputDimension,out.prediction);
      Matrix gotE(c.batchSize,c.outputDimension,out.error);
      Matrix gotDp(c.batchSize,c.outputDimension,out.dPrediction);
      Matrix gotDw2(c.hiddenDimension,c.outputDimension,out.dW2);
      Matrix gotDh(c.batchSize,c.hiddenDimension,out.dHidden);
      Matrix gotDz(c.batchSize,c.hiddenDimension,out.dZ1);
      Matrix gotDw1(c.dimension,c.hiddenDimension,out.dW1);
      hiddenError=std::max(hiddenError,maxAbs(ref.h,gotH));
      predictionError=std::max(predictionError,maxAbs(ref.p,gotP));
      Matrix refE(c.batchSize,c.outputDimension);
      for(size_t k=0;k<refE.values.size();++k) refE.values[k]=ref.p.values[k]-y.values[k];
      errorTensorError=std::max(errorTensorError,maxAbs(refE,gotE));
      lossError=std::max(lossError,std::abs(double(ref.loss)-out.loss));
      dpError=std::max(dpError,maxAbs(ref.dp,gotDp));
      dw2Error=std::max(dw2Error,maxAbs(ref.dw2,gotDw2));
      dhError=std::max(dhError,maxAbs(ref.dh,gotDh));
      dz1Error=std::max(dz1Error,maxAbs(ref.dz1,gotDz));
      dw1Error=std::max(dw1Error,maxAbs(ref.dw1,gotDw1));
      for(size_t k=0;k<out.mask.size();++k) {
        if ((out.mask[k]!=0)!=(out.hidden[k]>0.0f)) ++internalMaskMismatch;
        if ((out.mask[k]!=0)!=(ref.h.values[k]>0.0f)) {
          ++cpuMaskMismatch;
          maskMismatchMaxAbsZ1=std::max(maskMismatchMaxAbsZ1,std::abs(double(ref.z1.values[k])));
        }
      }
      Matrix expectedW1=currentW1, expectedW2=currentW2;
      cpu::sgdUpdate(expectedW1,ref.dw1,c.learningRate);
      cpu::sgdUpdate(expectedW2,ref.dw2,c.learningRate);
      w1NextError=std::max(w1NextError,maxAbs(expectedW1,nextW1));
      w2NextError=std::max(w2NextError,maxAbs(expectedW2,nextW2));
      Matrix opExpectedW1=currentW1,opExpectedW2=currentW2;
      cpu::sgdUpdate(opExpectedW1,gotDw1,c.learningRate);
      cpu::sgdUpdate(opExpectedW2,gotDw2,c.learningRate);
      optimizerW1Error=std::max(optimizerW1Error,maxAbs(opExpectedW1,nextW1));
      optimizerW2Error=std::max(optimizerW2Error,maxAbs(opExpectedW2,nextW2));
      dpNonzero=dpNonzero||std::any_of(out.dPrediction.begin(),out.dPrediction.end(),[](float v){return v!=0;});
      dw1Nonzero=dw1Nonzero||std::any_of(out.dW1.begin(),out.dW1.end(),[](float v){return v!=0;});
      dw2Nonzero=dw2Nonzero||std::any_of(out.dW2.begin(),out.dW2.end(),[](float v){return v!=0;});
      if (firstPrediction.empty()) firstPrediction=out.prediction;
      else nextPredictionChanged=nextPredictionChanged||out.prediction!=firstPrediction;
    }
    nan=nan||!std::isfinite(out.loss)||!finite(nextW1)||!finite(nextW2);
    w1Changed=w1Changed||maxAbs(initialW1,nextW1)>0;
    w2Changed=w2Changed||maxAbs(initialW2,nextW2)>0;
    if (diagnostic) {
      Pass cpuPass=pass(x,y,cpuW1,cpuW2);
      cpu::sgdUpdate(cpuW1,cpuPass.dw1,c.learningRate);
      cpu::sgdUpdate(cpuW2,cpuPass.dw2,c.learningRate);
    }
    const auto handoffStart=Clock::now();
    w1Current.swap(out.w1Next);
    w2Current.swap(out.w2Next);
    handoffTimes.push_back(std::chrono::duration<double,std::micro>(Clock::now()-handoffStart).count());
    const int completed=st+1;
    if (diagnostic) {
      Matrix htpW1(c.dimension,c.hiddenDimension,w1Current);
      Matrix htpW2(c.hiddenDimension,c.outputDimension,w2Current);
      trajectoryW1Error=std::max(trajectoryW1Error,maxAbs(cpuW1,htpW1));
      trajectoryW2Error=std::max(trajectoryW2Error,maxAbs(cpuW2,htpW2));
      if(checkpoint(completed)||completed==steps) {
        const float checkpointLoss=pass(d.x,d.y,htpW1,htpW2).loss;
        trajectory << "trajectory_"<<completed<<"_w1_checksum="<<checksum(htpW1)
                   <<"\ntrajectory_"<<completed<<"_w2_checksum="<<checksum(htpW2)
                   <<"\ntrajectory_"<<completed<<"_w1_l2_norm="<<l2(htpW1)
                   <<"\ntrajectory_"<<completed<<"_w2_l2_norm="<<l2(htpW2)
                   <<"\ntrajectory_"<<completed<<"_cpu_htp_w1_max_abs_difference="<<maxAbs(cpuW1,htpW1)
                   <<"\ntrajectory_"<<completed<<"_cpu_htp_w2_max_abs_difference="<<maxAbs(cpuW2,htpW2)
                   <<"\ntrajectory_"<<completed<<"_loss="<<checkpointLoss<<'\n';
      }
    }
    fullTimes.push_back(std::chrono::duration<double,std::micro>(Clock::now()-fullStart).count());
    if(nan) break;
  }
  Matrix finalW1(c.dimension,c.hiddenDimension,w1Current), finalW2(c.hiddenDimension,c.outputDimension,w2Current);
  Pass finalPass=pass(d.x,d.y,finalW1,finalW2);
  const double predictionMax=maxAbs(finalPass.p,d.y);
  double finalCpuHtpLossDifference=0, finalCpuHtpPredictionDifference=0;
  double finalCpuHtpW1Difference=0, finalCpuHtpW2Difference=0;
  if (diagnostic) {
    const Pass cpuFinalPass=pass(d.x,d.y,cpuW1,cpuW2);
    finalCpuHtpLossDifference=std::abs(double(cpuFinalPass.loss)-finalPass.loss);
    finalCpuHtpPredictionDifference=maxAbs(cpuFinalPass.p,finalPass.p);
    finalCpuHtpW1Difference=maxAbs(cpuW1,finalW1);
    finalCpuHtpW2Difference=maxAbs(cpuW2,finalW2);
  }
  const bool correctnessOk=!diagnostic || (hiddenError<1e-3&&predictionError<1e-3&&errorTensorError<2e-3&&lossError<1e-3&&dpError<1e-3&&
      dw2Error<1e-2&&dhError<1e-2&&dz1Error<1e-2&&dw1Error<1e-2&&
      w1NextError<1e-2&&w2NextError<1e-2&&optimizerW1Error<1e-3&&optimizerW2Error<1e-3&&internalMaskMismatch==0&&
      (cpuMaskMismatch==0||maskMismatchMaxAbsZ1<1e-3));
  const bool ok=!nan&&finalPass.loss<initialLoss&&correctnessOk&&w1Changed&&w2Changed;
  std::ostringstream s;
  s<<std::setprecision(9)<<"MLP_FULL_STEP\nexecution_mode="<<executionModeName(mode)
   <<"\nseed="<<c.seed<<"\nsample_count="<<c.sampleCount<<"\nbatch_size="<<c.batchSize
   <<"\ninput_dim="<<c.dimension<<"\nhidden_dim="<<c.hiddenDimension
   <<"\noutput_dim="<<c.outputDimension<<"\nepochs="<<c.epochs<<"\nsteps="<<steps
   <<"\nlearning_rate="<<c.learningRate<<"\ninitial_loss="<<initialLoss
   <<"\nfinal_loss="<<finalPass.loss<<"\nloss_reduction_ratio="<<(finalPass.loss/initialLoss)
   <<"\nprediction_mse="<<finalPass.loss<<"\nprediction_max_abs_error="<<predictionMax
   <<"\nhidden_max_abs_error="<<hiddenError<<"\nprediction_max_abs_difference="<<predictionError
   <<"\nerror_tensor_max_abs_error="<<errorTensorError<<"\nloss_abs_error="<<lossError
   <<"\ndp_max_abs_error="<<dpError<<"\ndw2_max_abs_error="<<dw2Error
   <<"\ndh_max_abs_error="<<dhError<<"\nmask_mismatch_count="<<cpuMaskMismatch
   <<"\ninternal_mask_mismatch_count="<<internalMaskMismatch
   <<"\nmask_mismatch_max_abs_cpu_z1="<<maskMismatchMaxAbsZ1
   <<"\ndz1_max_abs_error="<<dz1Error<<"\ndw1_max_abs_error="<<dw1Error
   <<"\nw1_next_max_abs_error="<<w1NextError<<"\nw2_next_max_abs_error="<<w2NextError
   <<"\nw1_update_max_abs_error="<<optimizerW1Error<<"\nw2_update_max_abs_error="<<optimizerW2Error
   <<"\ncpu_htp_final_loss_difference="<<finalCpuHtpLossDifference
   <<"\ncpu_htp_prediction_max_abs_difference="<<finalCpuHtpPredictionDifference
   <<"\ncpu_htp_w1_max_abs_difference="<<finalCpuHtpW1Difference
   <<"\ncpu_htp_w2_max_abs_difference="<<finalCpuHtpW2Difference
   <<"\ncpu_htp_w1_trajectory_max_abs_difference="<<trajectoryW1Error
   <<"\ncpu_htp_w2_trajectory_max_abs_difference="<<trajectoryW2Error
   <<"\ntraining_graph_create_count=1\ntraining_graph_finalize_count=1"
   <<"\ntraining_graph_execute_count="<<steps<<"\nexecute_count_per_step=1"
   <<"\nold_execute_count_per_step=2\nnew_execute_count_per_step=1\nexecute_reduction=1"
   <<"\nweight_handoff_method=PING_PONG_BINDING\nweight_handoff_backend=CPU"
   <<"\ninput_output_alias=false\ncopied_bytes_per_step=0"
   <<"\nforward_backend=HTP\nloss_backend=HTP\ndp_backend=HTP"
   <<"\nbackward_backend=HTP\ndw2_backend=HTP\ndh_backend=HTP"
   <<"\nrelu_backward_backend=HTP\ndw1_backend=HTP\noptimizer_backend=HTP"
   <<"\nruntime_initialization_us="<<initializationUs
   <<"\nw1_changed="<<(w1Changed?"true":"false")<<"\nw2_changed="<<(w2Changed?"true":"false")
   <<"\ndp_nonzero="<<(dpNonzero?"true":"false")<<"\ndw1_nonzero="<<(dw1Nonzero?"true":"false")
   <<"\ndw2_nonzero="<<(dw2Nonzero?"true":"false")
   <<"\nnext_step_prediction_changed="<<(nextPredictionChanged?"true":"false");
  s<<"\ntraining_graph_create_us="<<rt.metrics().graphCreateUs
   <<"\ntraining_graph_finalize_us="<<rt.metrics().graphFinalizeUs;
  emitStats(s,"batch_preparation",batchPreparationTimes);
  emitStats(s,"input_bind",rt.metrics().inputBindUs);
  emitStats(s,"output_access_bind",rt.metrics().outputBindUs);
  emitStats(s,"training_graph_execute",rt.metrics().executeUs);
  emitStats(s,"weight_buffer_handoff",handoffTimes);
  emitStats(s,"full_step",fullTimes);
  if(!rt.metrics().executeUs.empty()) {
    s<<"\nfirst_execute_us="<<rt.metrics().executeUs.front();
    std::vector<double> steady(rt.metrics().executeUs.begin()+1,rt.metrics().executeUs.end());
    emitStats(s,"steady_execute",steady);
  }
  s<<'\n'<<trajectory.str()
   <<"cpu_fallback=false\nhtp_loss_used=true\nhtp_dp_used=true\nhtp_optimizer_used=true"
   <<"\nhtp_full_step_used=true\nnan_detected="<<(nan?"true":"false")
   <<"\ninf_detected=false\nnan_inf="<<(nan?"true":"false")
   <<"\nstatus="<<(ok?"SUCCESS":"FAILED")<<'\n'<<rt.diagnostics()
   <<rt.apiTraceSummary()<<rt.qnnCallbackCaptureSummary();
  return s.str();
}
} // namespace
std::string runMlpExperiment(ExecutionMode mode, const TrainingConfig &c,
                             std::atomic_bool &stop, const LogSink &log) {
  std::string r;
  if (mode == ExecutionMode::QNN_MLP_GRADIENT_CHECK)
    r = gradientCheck();
  else if (mode == ExecutionMode::QNN_HTP_DX_CHECK)
    r = dxCheck(c);
  else if (mode == ExecutionMode::QNN_HTP_RELU_BACKWARD_CHECK)
    r = reluBackwardCheck();
  else if (mode == ExecutionMode::QNN_HTP_MSE_CHECK ||
           mode == ExecutionMode::QNN_HTP_SGD_CHECK)
    r = trainingOpsMicroCheck(mode);
  else if (mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP ||
           mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK ||
           mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE ||
           mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE ||
           mode == ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE)
    r = trainFullStep(mode, c, stop);
  else
    r = train(mode, c, stop);
  if (log)
    log(r);
  return r;
}
} // namespace phonelm::qnn