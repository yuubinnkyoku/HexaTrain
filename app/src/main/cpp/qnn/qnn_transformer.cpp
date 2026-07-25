// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku

#include "qnn_transformer.h"
#include "qnn_runtime.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

namespace phonelm::qnn {
namespace {
bool finite(const std::vector<float> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}
bool anyNan(const std::vector<float> &values) {
  return std::any_of(values.begin(), values.end(),
                     [](float value) { return std::isnan(value); });
}
bool anyInf(const std::vector<float> &values) {
  return std::any_of(values.begin(), values.end(),
                     [](float value) { return std::isinf(value); });
}
double maxAbs(const std::vector<float> &a, const std::vector<float> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    result = std::max(result, std::abs(double(a[i]) - b[i]));
  return result;
}
double meanAbs(const std::vector<float> &a, const std::vector<float> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    result += std::abs(double(a[i]) - b[i]);
  return result / a.size();
}
double maxRelative(const std::vector<float> &reference,
                   const std::vector<float> &actual) {
  double result = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    result =
        std::max(result, std::abs(double(reference[i]) - actual[i]) /
                             std::max(1.0e-3, std::abs(double(reference[i]))));
  }
  return result;
}
std::vector<float> layerNormReference(const std::vector<float> &input,
                                      size_t dimension, float epsilon) {
  std::vector<float> output(input.size());
  for (size_t start = 0; start < input.size(); start += dimension) {
    double mean = 0.0;
    for (size_t j = 0; j < dimension; ++j)
      mean += input[start + j];
    mean /= dimension;
    double variance = 0.0;
    for (size_t j = 0; j < dimension; ++j) {
      double d = input[start + j] - mean;
      variance += d * d;
    }
    variance /= dimension;
    const double inverse = 1.0 / std::sqrt(variance + epsilon);
    for (size_t j = 0; j < dimension; ++j)
      output[start + j] = float((input[start + j] - mean) * inverse);
  }
  return output;
}
std::vector<float> softmaxReference(const std::vector<float> &input,
                                    size_t columns) {
  std::vector<float> output(input.size());
  for (size_t start = 0; start < input.size(); start += columns) {
    const float maximum = *std::max_element(input.begin() + start,
                                            input.begin() + start + columns);
    double sum = 0.0;
    for (size_t j = 0; j < columns; ++j) {
      output[start + j] = std::exp(input[start + j] - maximum);
      sum += output[start + j];
    }
    for (size_t j = 0; j < columns; ++j)
      output[start + j] = float(output[start + j] / sum);
  }
  return output;
}
std::string failure(const char *test, const std::string &error,
                    Runtime &runtime) {
  return std::string("TRANSFORMER_MICRO\ntest=") + test +
         "\nstatus=FAILED\nerror=" + error +
         "\ncpu_fallback=false\nnan_inf=false\n" + runtime.apiTraceSummary() +
         runtime.diagnostics();
}
std::string layerNormCheck() {
  constexpr uint32_t b = 2, t = 3, d = 8;
  constexpr float epsilon = 1.0e-5f;
  std::vector<float> input(b * t * d);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = float(int(i % 17) - 8) * 0.125f + float(i / 8) * 0.01f;
  const auto reference = layerNormReference(input, d, epsilon);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  std::vector<float> output;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareLayerNorm(b, t, d, epsilon, error) ||
      !runtime.executeLayerNorm(input, output, error))
    return failure("layer_norm", error, runtime);
  const double maximum = maxAbs(reference, output),
               mean = meanAbs(reference, output),
               relative = maxRelative(reference, output);
  const bool ok = finite(output) && maximum < 2.0e-3;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_MICRO\ntest=layer_norm\nmethod=DIRECT_OP\nshape="
            "2x3x8\nepsilon="
         << epsilon << "\nmax_abs_error=" << maximum
         << "\nmean_abs_error=" << mean << "\nmax_relative_error=" << relative
         << "\nnan_inf=" << (finite(output) ? "false" : "true")
         << "\ngraph_finalize=SUCCESS\ngraph_execute=SUCCESS\ncpu_fallback="
            "false\nstatus="
         << (ok ? "SUCCESS" : "FAILED") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
struct LayerNormCpuResult {
  std::vector<float> output;
  std::vector<float> normalized;
  std::vector<float> dInput;
  std::vector<float> dGamma;
  std::vector<float> dBeta;
};

LayerNormCpuResult layerNormBackwardReference(
    const std::vector<float> &input, const std::vector<float> &upstream,
    const std::vector<float> &gamma, const std::vector<float> &beta,
    size_t dimension, double epsilon) {
  LayerNormCpuResult result;
  result.output.resize(input.size());
  result.normalized.resize(input.size());
  result.dInput.resize(input.size());
  result.dGamma.assign(dimension, 0.0f);
  result.dBeta.assign(dimension, 0.0f);
  for (size_t start = 0; start < input.size(); start += dimension) {
    double mean = 0.0;
    for (size_t column = 0; column < dimension; ++column)
      mean += input[start + column];
    mean /= dimension;
    double variance = 0.0;
    for (size_t column = 0; column < dimension; ++column) {
      const double centered = input[start + column] - mean;
      variance += centered * centered;
    }
    variance /= dimension;
    const double inverseStd = 1.0 / std::sqrt(variance + epsilon);
    double sumDNormalized = 0.0, sumDNormalizedXhat = 0.0;
    for (size_t column = 0; column < dimension; ++column) {
      const size_t index = start + column;
      result.normalized[index] = float((input[index] - mean) * inverseStd);
      result.output[index] =
          result.normalized[index] * gamma[column] + beta[column];
      const double dNormalized = upstream[index] * gamma[column];
      sumDNormalized += dNormalized;
      sumDNormalizedXhat += dNormalized * result.normalized[index];
      result.dGamma[column] += upstream[index] * result.normalized[index];
      result.dBeta[column] += upstream[index];
    }
    for (size_t column = 0; column < dimension; ++column) {
      const size_t index = start + column;
      const double dNormalized = upstream[index] * gamma[column];
      result.dInput[index] = float(
          inverseStd / dimension *
          (dimension * dNormalized - sumDNormalized -
           result.normalized[index] * sumDNormalizedXhat));
    }
  }
  return result;
}

std::vector<float> layerNormNumericGradient(
    const std::vector<float> &input, const std::vector<float> &upstream,
    const std::vector<float> &gamma, const std::vector<float> &beta,
    size_t dimension, double normalizationEpsilon, int variable,
    double finiteDifferenceEpsilon) {
  std::vector<double> x(input.begin(), input.end()), g(gamma.begin(), gamma.end()),
      b(beta.begin(), beta.end());
  auto loss = [&]() {
    double total = 0.0;
    for (size_t start = 0; start < x.size(); start += dimension) {
      double mean = 0.0;
      for (size_t column = 0; column < dimension; ++column)
        mean += x[start + column];
      mean /= dimension;
      double variance = 0.0;
      for (size_t column = 0; column < dimension; ++column) {
        const double centered = x[start + column] - mean;
        variance += centered * centered;
      }
      variance /= dimension;
      const double inverseStd =
          1.0 / std::sqrt(variance + normalizationEpsilon);
      for (size_t column = 0; column < dimension; ++column) {
        const size_t index = start + column;
        const double normalized = (x[index] - mean) * inverseStd;
        total += (normalized * g[column] + b[column]) * upstream[index];
      }
    }
    return total;
  };
  std::vector<double> *selected = variable == 0 ? &x : (variable == 1 ? &g : &b);
  std::vector<float> gradient(selected->size());
  for (size_t index = 0; index < selected->size(); ++index) {
    const double original = (*selected)[index];
    (*selected)[index] = original + finiteDifferenceEpsilon;
    const double positive = loss();
    (*selected)[index] = original - finiteDifferenceEpsilon;
    const double negative = loss();
    (*selected)[index] = original;
    gradient[index] = float((positive - negative) /
                            (2.0 * finiteDifferenceEpsilon));
  }
  return gradient;
}

std::string layerNormBackwardCheck() {
  constexpr uint32_t batch = 2, tokens = 3, dimension = 8, rows = batch * tokens;
  constexpr float normalizationEpsilon = 1.0e-5f;
  constexpr double finiteDifferenceEpsilon = 1.0e-5;
  struct TestCase {
    std::string name;
    std::vector<float> input;
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> upstream;
  };
  std::vector<float> normal(rows * dimension), lowVariance(rows * dimension);
  for (size_t index = 0; index < normal.size(); ++index) {
    normal[index] = float(int(index % 17) - 8) * 0.125f +
                    float(index / dimension) * 0.01f;
    lowVariance[index] = 0.25f + float(int(index % dimension) - 3) * 1.0e-3f +
                         float(index / dimension) * 2.0e-4f;
  }
  std::vector<float> gammaOne(dimension, 1.0f), betaZero(dimension, 0.0f);
  std::vector<float> gammaArbitrary(dimension), betaArbitrary(dimension);
  for (uint32_t column = 0; column < dimension; ++column) {
    gammaArbitrary[column] = 0.7f + float(column) * 0.08f;
    betaArbitrary[column] = float(int(column) - 3) * 0.04f;
  }
  std::vector<float> upstream(rows * dimension),
      lowVarianceUpstream(rows * dimension);
  for (size_t index = 0; index < upstream.size(); ++index) {
    upstream[index] = std::sin(float(index + 2) * 0.19f) * 0.35f +
                      float(int(index % 5) - 2) * 0.03f;
    lowVarianceUpstream[index] = upstream[index] * 0.005f;
  }
  const std::vector<TestCase> cases = {
      {"gamma_one_beta_zero", normal, gammaOne, betaZero, upstream},
      {"arbitrary_gamma_beta_low_variance", lowVariance, gammaArbitrary,
       betaArbitrary, lowVarianceUpstream}};
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareLayerNormBackward(rows, dimension, normalizationEpsilon,
                                        error))
    return failure("layer_norm_backward", error, runtime);
  double analyticNumericMax = 0.0, analyticNumericMeanSum = 0.0;
  double htpCpuMax = 0.0, htpCpuMeanSum = 0.0, htpCpuRelative = 0.0;
  double dxMax = 0.0, dGammaMax = 0.0, dBetaMax = 0.0;
  double outputMax = 0.0, normalizedMax = 0.0;
  bool nanDetected = false, infDetected = false;
  for (const auto &testCase : cases) {
    const auto cpu = layerNormBackwardReference(
        testCase.input, testCase.upstream, testCase.gamma, testCase.beta,
        dimension, normalizationEpsilon);
    const auto numericX = layerNormNumericGradient(
        testCase.input, testCase.upstream, testCase.gamma, testCase.beta,
        dimension, normalizationEpsilon, 0, finiteDifferenceEpsilon);
    const auto numericGamma = layerNormNumericGradient(
        testCase.input, testCase.upstream, testCase.gamma, testCase.beta,
        dimension, normalizationEpsilon, 1, finiteDifferenceEpsilon);
    const auto numericBeta = layerNormNumericGradient(
        testCase.input, testCase.upstream, testCase.gamma, testCase.beta,
        dimension, normalizationEpsilon, 2, finiteDifferenceEpsilon);
    LayerNormBackwardOutputs htp;
    if (!runtime.executeLayerNormBackward(testCase.input, testCase.upstream,
                                          testCase.gamma, testCase.beta, htp,
                                          error))
      return failure("layer_norm_backward", error, runtime);
    analyticNumericMax = std::max(
        {analyticNumericMax, maxAbs(cpu.dInput, numericX),
         maxAbs(cpu.dGamma, numericGamma), maxAbs(cpu.dBeta, numericBeta)});
    analyticNumericMeanSum +=
        (meanAbs(cpu.dInput, numericX) + meanAbs(cpu.dGamma, numericGamma) +
         meanAbs(cpu.dBeta, numericBeta)) /
        3.0;
    dxMax = std::max(dxMax, maxAbs(cpu.dInput, htp.dInput));
    dGammaMax = std::max(dGammaMax, maxAbs(cpu.dGamma, htp.dGamma));
    dBetaMax = std::max(dBetaMax, maxAbs(cpu.dBeta, htp.dBeta));
    htpCpuMax = std::max({htpCpuMax, dxMax, dGammaMax, dBetaMax});
    htpCpuMeanSum +=
        (meanAbs(cpu.dInput, htp.dInput) +
         meanAbs(cpu.dGamma, htp.dGamma) + meanAbs(cpu.dBeta, htp.dBeta)) /
        3.0;
    htpCpuRelative =
        std::max({htpCpuRelative, maxRelative(cpu.dInput, htp.dInput),
                  maxRelative(cpu.dGamma, htp.dGamma),
                  maxRelative(cpu.dBeta, htp.dBeta)});
    outputMax = std::max(outputMax, maxAbs(cpu.output, htp.output));
    normalizedMax =
        std::max(normalizedMax, maxAbs(cpu.normalized, htp.normalized));
    nanDetected = nanDetected || anyNan(htp.output) || anyNan(htp.normalized) ||
                  anyNan(htp.dInput) || anyNan(htp.dGamma) || anyNan(htp.dBeta);
    infDetected = infDetected || anyInf(htp.output) || anyInf(htp.normalized) ||
                  anyInf(htp.dInput) || anyInf(htp.dGamma) || anyInf(htp.dBeta);
  }
  const double analyticNumericMean = analyticNumericMeanSum / cases.size();
  const double htpCpuMean = htpCpuMeanSum / cases.size();
  const bool ok = !nanDetected && !infDetected && analyticNumericMax < 5.0e-4 &&
                  dxMax <= 2.0e-2 && dGammaMax <= 2.0e-2 &&
                  dBetaMax <= 2.0e-2;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_BACKWARD_MICRO\ntest=layer_norm_backward\n"
            "execution_mode=QNN_HTP_LAYER_NORM_BACKWARD_CHECK\n"
            "method=PRIMITIVE_COMPOSITION\nshape=B2_T3_D8\n"
            "qnn_tensor_shape=6x8\nnorm_axis=D\nepsilon="
         << normalizationEpsilon
         << "\nfinite_difference_epsilon=" << finiteDifferenceEpsilon
         << "\nvariance_rescale=64\nlow_variance_upstream_scale=0.005"
            "\ncases=gamma_one_beta_zero,arbitrary_gamma_beta_low_variance\n"
            "cpu_analytic_vs_numeric_max_abs_error="
         << analyticNumericMax
         << "\ncpu_analytic_vs_numeric_mean_abs_error=" << analyticNumericMean
         << "\nhtp_vs_cpu_max_abs_error=" << htpCpuMax
         << "\nhtp_vs_cpu_mean_abs_error=" << htpCpuMean
         << "\nhtp_vs_cpu_max_relative_error=" << htpCpuRelative
         << "\nhtp_vs_cpu_dx_max_abs_error=" << dxMax
         << "\nhtp_vs_cpu_dgamma_max_abs_error=" << dGammaMax
         << "\nhtp_vs_cpu_dbeta_max_abs_error=" << dBetaMax
         << "\nhtp_vs_cpu_forward_max_abs_error=" << outputMax
         << "\nhtp_vs_cpu_xhat_max_abs_error=" << normalizedMax
         << "\ngraph_create_result=0\ngraph_finalize_result=0\n"
            "graph_execute_result=0\ngraph_create=SUCCESS\n"
            "graph_finalize=SUCCESS\ngraph_execute=SUCCESS\n"
            "cpu_fallback=false\nnan_detected="
         << (nanDetected ? "true" : "false") << "\ninf_detected="
         << (infDetected ? "true" : "false") << "\nnan_inf="
         << (nanDetected || infDetected ? "true" : "false")
         << "\nhtp_graph_execute_count=" << runtime.metrics().graphExecuteCount
         << "\nstatus=" << (ok ? "SUCCESS" : "FAILED") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
std::string softmaxCheck() {
  constexpr uint32_t rows = 3, columns = 4;
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareSoftmax(rows, columns, error))
    return failure("softmax", error, runtime);
  const std::vector<std::vector<float>> cases = {
      {-1, 0, 1, 2, 2, 1, 0, -1, 0.5f, -0.5f, 1.5f, -1.5f},
      {1000, 999, 998, 997, 1004, 1003, 1002, 1001, 1010, 1000, 990, 980},
      {-1000, -1001, -1002, -1003, -996, -997, -998, -999, -990, -1000, -1010,
       -1020}};
  double maximum = 0.0, mean = 0.0, rowSumError = 0.0;
  bool allFinite = true;
  size_t count = 0;
  for (const auto &input : cases) {
    std::vector<float> output;
    if (!runtime.executeSoftmax(input, output, error))
      return failure("softmax", error, runtime);
    const auto reference = softmaxReference(input, columns);
    maximum = std::max(maximum, maxAbs(reference, output));
    mean += meanAbs(reference, output);
    ++count;
    allFinite = allFinite && finite(output);
    for (size_t start = 0; start < output.size(); start += columns) {
      double sum = std::accumulate(output.begin() + start,
                                   output.begin() + start + columns, 0.0);
      rowSumError = std::max(rowSumError, std::abs(sum - 1.0));
    }
  }
  mean /= count;
  const bool ok = allFinite && maximum < 2.0e-3 && rowSumError < 2.0e-3;
  std::ostringstream report;
  report
      << std::setprecision(10)
      << "TRANSFORMER_MICRO\ntest=softmax\nmethod=DIRECT_OP\nshape=3x4\naxis="
         "last\ncases=normal,large_positive,large_negative\nmax_abs_error="
      << maximum << "\nmean_abs_error=" << mean
      << "\nmax_row_sum_error=" << rowSumError
      << "\nnan_inf=" << (allFinite ? "false" : "true")
      << "\ngraph_finalize=SUCCESS\ngraph_execute=SUCCESS\ncpu_fallback="
         "false\nstatus="
      << (ok ? "SUCCESS" : "FAILED") << '\n'
      << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
std::vector<float> softmaxBackwardReference(
    const std::vector<float> &probabilities,
    const std::vector<float> &upstream, size_t columns) {
  std::vector<float> gradient(probabilities.size());
  for (size_t start = 0; start < probabilities.size(); start += columns) {
    double dot = 0.0;
    for (size_t j = 0; j < columns; ++j)
      dot += double(upstream[start + j]) * probabilities[start + j];
    for (size_t j = 0; j < columns; ++j)
      gradient[start + j] = float(probabilities[start + j] *
                                  (upstream[start + j] - dot));
  }
  return gradient;
}

std::vector<float> softmaxBackwardNumeric(const std::vector<float> &input,
                                           const std::vector<float> &upstream,
                                           size_t columns, float epsilon) {
  std::vector<double> perturbed(input.begin(), input.end());
  std::vector<float> gradient(input.size());
  auto loss = [&](const std::vector<double> &x) {
    double result = 0.0;
    for (size_t start = 0; start < x.size(); start += columns) {
      const double maximum = *std::max_element(x.begin() + start,
                                                x.begin() + start + columns);
      double denominator = 0.0;
      for (size_t j = 0; j < columns; ++j)
        denominator += std::exp(x[start + j] - maximum);
      for (size_t j = 0; j < columns; ++j)
        result += std::exp(x[start + j] - maximum) / denominator *
                  upstream[start + j];
    }
    return result;
  };
  for (size_t i = 0; i < input.size(); ++i) {
    perturbed[i] = double(input[i]) + epsilon;
    const double positive = loss(perturbed);
    perturbed[i] = double(input[i]) - epsilon;
    const double negative = loss(perturbed);
    perturbed[i] = input[i];
    gradient[i] = float((positive - negative) / (2.0 * epsilon));
  }
  return gradient;
}

std::string softmaxBackwardCheck() {
  constexpr uint32_t rows = 4, columns = 4;
  constexpr float epsilon = 1.0e-3f;
  const std::vector<float> upstream = {
      0.25f, -0.5f, 0.75f, -0.125f, -0.3f, 0.2f, 0.6f, -0.4f,
      0.9f, -0.7f, 0.1f, 0.35f, -0.8f, 0.55f, -0.2f, 0.45f};
  const std::vector<std::pair<std::string, std::vector<float>>> cases = {
      {"normal", {-1.0f, 0.0f, 1.0f, 2.0f, 0.2f, -0.3f, 0.7f, 0.1f,
                  -0.4f, 0.9f, -0.8f, 0.3f, 1.2f, -1.1f, 0.4f, -0.2f}},
      {"large_positive", {1000, 999, 998, 997, 1004, 1003, 1002, 1001,
                          1010, 1000, 990, 980, 1020, 1019, 1018, 1017}},
      {"large_negative", {-1000, -1001, -1002, -1003, -996, -997, -998,
                          -999, -990, -1000, -1010, -1020, -1010, -1011,
                          -1012, -1013}},
      {"equal_rows", {3, 3, 3, 3, -2, -2, -2, -2, 0.5f, 0.5f, 0.5f,
                      0.5f, 12, 12, 12, 12}},
      {"causal_masked", {0.4f, -10000, -10000, -10000, 0.2f, -0.1f,
                          -10000, -10000, -0.3f, 0.8f, 0.1f, -10000,
                          0.7f, -0.2f, 0.5f, 0.0f}}};
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareSoftmaxBackward(rows, columns, error))
    return failure("softmax_backward", error, runtime);
  double analyticNumericMax = 0.0, analyticNumericSum = 0.0;
  double htpCpuMax = 0.0, htpCpuSum = 0.0, htpCpuRelative = 0.0;
  double rowSumMax = 0.0;
  size_t compared = 0;
  bool allFinite = true, nanDetected = false, infDetected = false;
  for (const auto &testCase : cases) {
    const auto probabilities = softmaxReference(testCase.second, columns);
    const auto analytic =
        softmaxBackwardReference(probabilities, upstream, columns);
    const auto numeric = softmaxBackwardNumeric(testCase.second, upstream,
                                                 columns, epsilon);
    std::vector<float> htp;
    if (!runtime.executeSoftmaxBackward(probabilities, upstream, htp, error))
      return failure("softmax_backward", error, runtime);
    analyticNumericMax = std::max(analyticNumericMax, maxAbs(analytic, numeric));
    htpCpuMax = std::max(htpCpuMax, maxAbs(analytic, htp));
    htpCpuRelative = std::max(htpCpuRelative, maxRelative(analytic, htp));
    for (size_t i = 0; i < analytic.size(); ++i) {
      analyticNumericSum += std::abs(double(analytic[i]) - numeric[i]);
      htpCpuSum += std::abs(double(analytic[i]) - htp[i]);
      ++compared;
    }
    for (size_t start = 0; start < htp.size(); start += columns) {
      const double sum = std::accumulate(htp.begin() + start,
                                         htp.begin() + start + columns, 0.0);
      rowSumMax = std::max(rowSumMax, std::abs(sum));
    }
    allFinite = allFinite && finite(probabilities) && finite(analytic) &&
                finite(numeric) && finite(htp);
    nanDetected = nanDetected || anyNan(probabilities) || anyNan(analytic) ||
                  anyNan(numeric) || anyNan(htp);
    infDetected = infDetected || anyInf(probabilities) || anyInf(analytic) ||
                  anyInf(numeric) || anyInf(htp);
  }
  const double analyticNumericMean = analyticNumericSum / compared;
  const double htpCpuMean = htpCpuSum / compared;
  const bool ok = allFinite && analyticNumericMax < 2.0e-4 &&
                  htpCpuMax <= 5.0e-3 && rowSumMax <= 5.0e-3;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_BACKWARD_MICRO\ntest=softmax_backward\n"
            "execution_mode=QNN_HTP_SOFTMAX_BACKWARD_CHECK\n"
            "method=PRIMITIVE_COMPOSITION\nshape=B1_H1_T4_D4\n"
            "qnn_tensor_shape=4x4\naxis=last\nepsilon="
         << epsilon
         << "\ncases=normal,large_positive,large_negative,equal_rows,causal_"
            "masked\ncpu_analytic_vs_numeric_max_abs_error="
         << analyticNumericMax
         << "\ncpu_analytic_vs_numeric_mean_abs_error=" << analyticNumericMean
         << "\nhtp_vs_cpu_max_abs_error=" << htpCpuMax
         << "\nhtp_vs_cpu_mean_abs_error=" << htpCpuMean
         << "\nhtp_vs_cpu_max_relative_error=" << htpCpuRelative
         << "\nmax_row_gradient_sum_abs=" << rowSumMax
         << "\ngraph_create_result=0\ngraph_finalize_result=0\n"
            "graph_execute_result=0\ngraph_create=SUCCESS\n"
            "graph_finalize=SUCCESS\ngraph_execute=SUCCESS\ncpu_fallback=false\n"
            "nan_detected="
         << (nanDetected ? "true" : "false") << "\ninf_detected="
         << (infDetected ? "true" : "false") << "\nnan_inf="
         << (allFinite ? "false" : "true")
         << "\nhtp_graph_execute_count=" << runtime.metrics().graphExecuteCount
         << "\nstatus=" << (ok ? "SUCCESS" : "FAILED") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
std::string attentionCheck() {
  constexpr uint32_t tokens = 4, dimension = 8;
  std::vector<float> query(tokens * dimension), key(tokens * dimension),
      value(tokens * dimension);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = std::sin(float(i + 1) * 0.17f);
    key[i] = std::cos(float(i + 2) * 0.13f);
    value[i] = float(int(i % 11) - 5) * 0.09f;
  }
  std::vector<float> mask(tokens * tokens, 0.0f);
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t column = row + 1; column < tokens; ++column)
      mask[row * tokens + column] = -10000.0f;
  const double scale = 1.0 / std::sqrt(double(dimension));
  std::vector<float> referenceProb(tokens * tokens),
      referenceOut(tokens * dimension, 0.0f);
  for (uint32_t row = 0; row < tokens; ++row) {
    double maximum = -1.0e300;
    for (uint32_t column = 0; column < tokens; ++column) {
      double score = 0.0;
      for (uint32_t d = 0; d < dimension; ++d)
        score +=
            double(query[row * dimension + d]) * key[column * dimension + d];
      score = score * scale + mask[row * tokens + column];
      referenceProb[row * tokens + column] = float(score);
      maximum = std::max(maximum, score);
    }
    double sum = 0.0;
    for (uint32_t column = 0; column < tokens; ++column) {
      referenceProb[row * tokens + column] = float(
          std::exp(double(referenceProb[row * tokens + column]) - maximum));
      sum += referenceProb[row * tokens + column];
    }
    for (uint32_t column = 0; column < tokens; ++column)
      referenceProb[row * tokens + column] =
          float(referenceProb[row * tokens + column] / sum);
    for (uint32_t d = 0; d < dimension; ++d)
      for (uint32_t column = 0; column < tokens; ++column)
        referenceOut[row * dimension + d] +=
            referenceProb[row * tokens + column] *
            value[column * dimension + d];
  }
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  std::vector<float> output, probabilities;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareAttention(tokens, dimension, error) ||
      !runtime.executeAttention(query, key, value, mask, output, probabilities,
                                error))
    return failure("attention_forward", error, runtime);
  const double outputError = maxAbs(referenceOut, output),
               probabilityError = maxAbs(referenceProb, probabilities);
  double futureMax = 0.0, rowSumError = 0.0;
  for (uint32_t row = 0; row < tokens; ++row) {
    double sum = 0.0;
    for (uint32_t column = 0; column < tokens; ++column) {
      sum += probabilities[row * tokens + column];
      if (column > row)
        futureMax = std::max(
            futureMax, double(std::abs(probabilities[row * tokens + column])));
    }
    rowSumError = std::max(rowSumError, std::abs(sum - 1.0));
  }
  const bool allFinite = finite(output) && finite(probabilities);
  const bool ok = allFinite && outputError < 5.0e-3 &&
                  probabilityError < 3.0e-3 && futureMax < 2.0e-3 &&
                  rowSumError < 3.0e-3;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_MICRO\ntest=attention_forward\nmethod=COMPOSED_QNN_"
            "OPS\nshape=B1_H1_T4_D8\ncausal_mask=true\nscale=1/"
            "sqrt(8)\nmax_abs_error="
         << outputError << "\nprobability_max_abs_error=" << probabilityError
         << "\nfuture_probability_max=" << futureMax
         << "\nmax_row_sum_error=" << rowSumError
         << "\nnan_inf=" << (allFinite ? "false" : "true")
         << "\ngraph_finalize=SUCCESS\ngraph_execute=SUCCESS\nhtp_graph_"
            "execute_count="
         << runtime.metrics().graphExecuteCount
         << "\ncpu_fallback=false\nstatus=" << (ok ? "SUCCESS" : "FAILED")
         << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
struct AttentionCpuResult {
  std::vector<float> probabilities;
  std::vector<float> output;
  std::vector<float> dScores;
  std::vector<float> dQuery;
  std::vector<float> dKey;
  std::vector<float> dValue;
};

AttentionCpuResult attentionBackwardReference(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value, const std::vector<float> &upstream,
    uint32_t tokens, uint32_t dimension) {
  AttentionCpuResult result;
  result.probabilities.resize(size_t(tokens) * tokens);
  result.output.assign(size_t(tokens) * dimension, 0.0f);
  result.dScores.resize(size_t(tokens) * tokens);
  result.dQuery.assign(size_t(tokens) * dimension, 0.0f);
  result.dKey.assign(size_t(tokens) * dimension, 0.0f);
  result.dValue.assign(size_t(tokens) * dimension, 0.0f);
  const double scale = 1.0 / std::sqrt(double(dimension));
  std::vector<float> scores(size_t(tokens) * tokens);
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t column = 0; column < tokens; ++column) {
      double sum = 0.0;
      for (uint32_t d = 0; d < dimension; ++d)
        sum += double(query[row * dimension + d]) *
               key[column * dimension + d];
      scores[row * tokens + column] =
          float(sum * scale + (column > row ? -10000.0 : 0.0));
    }
  result.probabilities = softmaxReference(scores, tokens);
  std::vector<float> dProbabilities(size_t(tokens) * tokens, 0.0f);
  for (uint32_t row = 0; row < tokens; ++row) {
    for (uint32_t d = 0; d < dimension; ++d)
      for (uint32_t column = 0; column < tokens; ++column)
        result.output[row * dimension + d] +=
            result.probabilities[row * tokens + column] *
            value[column * dimension + d];
    for (uint32_t column = 0; column < tokens; ++column) {
      double dp = 0.0;
      for (uint32_t d = 0; d < dimension; ++d)
        dp += double(upstream[row * dimension + d]) *
              value[column * dimension + d];
      dProbabilities[row * tokens + column] = float(dp);
    }
  }
  result.dScores =
      softmaxBackwardReference(result.probabilities, dProbabilities, tokens);
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t d = 0; d < dimension; ++d) {
      double dq = 0.0;
      for (uint32_t column = 0; column < tokens; ++column) {
        dq += double(result.dScores[row * tokens + column]) *
              key[column * dimension + d];
        result.dKey[column * dimension + d] +=
            float(result.dScores[row * tokens + column] *
                  query[row * dimension + d] * scale);
        result.dValue[column * dimension + d] +=
            result.probabilities[row * tokens + column] *
            upstream[row * dimension + d];
      }
      result.dQuery[row * dimension + d] = float(dq * scale);
    }
  return result;
}

std::vector<float> attentionNumericGradient(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value, const std::vector<float> &upstream,
    uint32_t tokens, uint32_t dimension, int variable, double epsilon) {
  std::vector<double> q(query.begin(), query.end()), k(key.begin(), key.end()),
      v(value.begin(), value.end());
  auto loss = [&]() {
    const double scale = 1.0 / std::sqrt(double(dimension));
    double total = 0.0;
    for (uint32_t row = 0; row < tokens; ++row) {
      std::vector<double> probabilities(tokens);
      double maximum = -1.0e300;
      for (uint32_t column = 0; column < tokens; ++column) {
        double score = 0.0;
        for (uint32_t d = 0; d < dimension; ++d)
          score += q[row * dimension + d] * k[column * dimension + d];
        probabilities[column] =
            score * scale + (column > row ? -10000.0 : 0.0);
        maximum = std::max(maximum, probabilities[column]);
      }
      double denominator = 0.0;
      for (double &probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (double &probability : probabilities)
        probability /= denominator;
      for (uint32_t d = 0; d < dimension; ++d) {
        double output = 0.0;
        for (uint32_t column = 0; column < tokens; ++column)
          output += probabilities[column] * v[column * dimension + d];
        total += output * upstream[row * dimension + d];
      }
    }
    return total;
  };
  std::vector<double> *selected = variable == 0 ? &q : (variable == 1 ? &k : &v);
  std::vector<float> gradient(selected->size());
  for (size_t index = 0; index < selected->size(); ++index) {
    const double original = (*selected)[index];
    (*selected)[index] = original + epsilon;
    const double positive = loss();
    (*selected)[index] = original - epsilon;
    const double negative = loss();
    (*selected)[index] = original;
    gradient[index] = float((positive - negative) / (2.0 * epsilon));
  }
  return gradient;
}

std::string attentionBackwardCheck() {
  constexpr uint32_t tokens = 4, dimension = 8;
  constexpr double epsilon = 1.0e-3;
  std::vector<float> query(tokens * dimension), key(tokens * dimension),
      value(tokens * dimension), upstream(tokens * dimension);
  for (size_t index = 0; index < query.size(); ++index) {
    query[index] = std::sin(float(index + 1) * 0.17f) * 0.7f;
    key[index] = std::cos(float(index + 2) * 0.13f) * 0.6f;
    value[index] = float(int(index % 11) - 5) * 0.09f;
    upstream[index] = std::sin(float(index + 3) * 0.11f) * 0.4f;
  }
  std::vector<float> mask(tokens * tokens, 0.0f);
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t column = row + 1; column < tokens; ++column)
      mask[row * tokens + column] = -10000.0f;
  const auto cpu = attentionBackwardReference(query, key, value, upstream,
                                               tokens, dimension);
  const auto numericQ = attentionNumericGradient(
      query, key, value, upstream, tokens, dimension, 0, epsilon);
  const auto numericK = attentionNumericGradient(
      query, key, value, upstream, tokens, dimension, 1, epsilon);
  const auto numericV = attentionNumericGradient(
      query, key, value, upstream, tokens, dimension, 2, epsilon);
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  AttentionBackwardOutputs htp;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareAttentionBackward(tokens, dimension, error) ||
      !runtime.executeAttentionBackward(query, key, value, upstream, mask, htp,
                                        error))
    return failure("attention_backward", error, runtime);
  const double numericQMax = maxAbs(cpu.dQuery, numericQ);
  const double numericKMax = maxAbs(cpu.dKey, numericK);
  const double numericVMax = maxAbs(cpu.dValue, numericV);
  const double analyticNumericMax =
      std::max({numericQMax, numericKMax, numericVMax});
  const double analyticNumericMean =
      (meanAbs(cpu.dQuery, numericQ) + meanAbs(cpu.dKey, numericK) +
       meanAbs(cpu.dValue, numericV)) /
      3.0;
  const double dqMax = maxAbs(cpu.dQuery, htp.dQuery);
  const double dkMax = maxAbs(cpu.dKey, htp.dKey);
  const double dvMax = maxAbs(cpu.dValue, htp.dValue);
  const double dsMax = maxAbs(cpu.dScores, htp.dScores);
  const double probabilityMax = maxAbs(cpu.probabilities, htp.probabilities);
  const double htpCpuMax = std::max({dqMax, dkMax, dvMax});
  const double htpCpuMean =
      (meanAbs(cpu.dQuery, htp.dQuery) + meanAbs(cpu.dKey, htp.dKey) +
       meanAbs(cpu.dValue, htp.dValue)) /
      3.0;
  const double htpCpuRelative =
      std::max({maxRelative(cpu.dQuery, htp.dQuery),
                maxRelative(cpu.dKey, htp.dKey),
                maxRelative(cpu.dValue, htp.dValue)});
  double futureProbabilityMax = 0.0, futureDScoreMax = 0.0;
  for (uint32_t row = 0; row < tokens; ++row)
    for (uint32_t column = row + 1; column < tokens; ++column) {
      futureProbabilityMax = std::max(
          futureProbabilityMax,
          double(std::abs(htp.probabilities[row * tokens + column])));
      futureDScoreMax = std::max(
          futureDScoreMax, double(std::abs(htp.dScores[row * tokens + column])));
    }
  const bool nanDetected = anyNan(htp.probabilities) || anyNan(htp.dScores) ||
                           anyNan(htp.dQuery) || anyNan(htp.dKey) ||
                           anyNan(htp.dValue);
  const bool infDetected = anyInf(htp.probabilities) || anyInf(htp.dScores) ||
                           anyInf(htp.dQuery) || anyInf(htp.dKey) ||
                           anyInf(htp.dValue);
  const bool ok = !nanDetected && !infDetected && analyticNumericMax < 2.0e-4 &&
                  dqMax <= 1.0e-2 && dkMax <= 1.0e-2 && dvMax <= 1.0e-2 &&
                  futureProbabilityMax <= 2.0e-3 && futureDScoreMax <= 2.0e-3;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_BACKWARD_MICRO\ntest=attention_backward\n"
            "execution_mode=QNN_HTP_ATTENTION_BACKWARD_CHECK\n"
            "method=COMPOSED_QNN_OPS\nshape=B1_H1_T4_D8\nepsilon="
         << epsilon
         << "\ncausal_mask=true\nmask_gradient=false\n"
            "cpu_analytic_vs_numeric_max_abs_error="
         << analyticNumericMax
         << "\ncpu_analytic_vs_numeric_mean_abs_error=" << analyticNumericMean
         << "\ncpu_analytic_vs_numeric_dq_max_abs_error=" << numericQMax
         << "\ncpu_analytic_vs_numeric_dk_max_abs_error=" << numericKMax
         << "\ncpu_analytic_vs_numeric_dv_max_abs_error=" << numericVMax
         << "\nhtp_vs_cpu_max_abs_error=" << htpCpuMax
         << "\nhtp_vs_cpu_mean_abs_error=" << htpCpuMean
         << "\nhtp_vs_cpu_max_relative_error=" << htpCpuRelative
         << "\nhtp_vs_cpu_dq_max_abs_error=" << dqMax
         << "\nhtp_vs_cpu_dk_max_abs_error=" << dkMax
         << "\nhtp_vs_cpu_dv_max_abs_error=" << dvMax
         << "\nhtp_vs_cpu_dscores_max_abs_error=" << dsMax
         << "\nhtp_vs_cpu_probability_max_abs_error=" << probabilityMax
         << "\nfuture_probability_max=" << futureProbabilityMax
         << "\nfuture_dscores_max=" << futureDScoreMax
         << "\ngraph_create_result=0\ngraph_finalize_result=0\n"
            "graph_execute_result=0\ngraph_create=SUCCESS\n"
            "graph_finalize=SUCCESS\ngraph_execute=SUCCESS\n"
            "cpu_fallback=false\nnan_detected="
         << (nanDetected ? "true" : "false") << "\ninf_detected="
         << (infDetected ? "true" : "false") << "\nnan_inf="
         << (nanDetected || infDetected ? "true" : "false")
         << "\nhtp_graph_execute_count=" << runtime.metrics().graphExecuteCount
         << "\nstatus=" << (ok ? "SUCCESS" : "FAILED") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
std::string tinyTransformerCheck() {
  constexpr uint32_t tokens = 4, dimension = 16, feedForward = 32;
  constexpr float epsilon = 1.0e-5f;
  std::vector<float> input(tokens * dimension);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = std::sin(float(i + 1) * 0.071f) + float(int(i % 7) - 3) * 0.03f;
  auto weights = [](size_t count, int phase) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
      values[i] = float(int((i * 17 + size_t(phase) * 13) % 29) - 14) * 0.01f;
    return values;
  };
  const auto wq = weights(dimension * dimension, 1),
             wk = weights(dimension * dimension, 2),
             wv = weights(dimension * dimension, 3),
             wo = weights(dimension * dimension, 4),
             w1 = weights(dimension * feedForward, 5),
             w2 = weights(feedForward * dimension, 6);
  auto matmul = [](const std::vector<float> &a, const std::vector<float> &b,
                   uint32_t rows, uint32_t inner, uint32_t columns) {
    std::vector<float> out(size_t(rows) * columns, 0.0f);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < columns; ++c) {
        double sum = 0.0;
        for (uint32_t k = 0; k < inner; ++k)
          sum += double(a[r * inner + k]) * b[k * columns + c];
        out[r * columns + c] = float(sum);
      }
    return out;
  };
  const auto ln1 = layerNormReference(input, dimension, epsilon),
             q = matmul(ln1, wq, tokens, dimension, dimension),
             k = matmul(ln1, wk, tokens, dimension, dimension),
             v = matmul(ln1, wv, tokens, dimension, dimension);
  std::vector<float> scores(tokens * tokens);
  const double scale = 1.0 / std::sqrt(double(dimension));
  for (uint32_t r = 0; r < tokens; ++r)
    for (uint32_t c = 0; c < tokens; ++c) {
      double sum = 0.0;
      for (uint32_t d = 0; d < dimension; ++d)
        sum += double(q[r * dimension + d]) * k[c * dimension + d];
      scores[r * tokens + c] = float(sum * scale + (c > r ? -10000.0 : 0.0));
    }
  const auto probabilities = softmaxReference(scores, tokens);
  const auto context = matmul(probabilities, v, tokens, tokens, dimension),
             projected = matmul(context, wo, tokens, dimension, dimension);
  std::vector<float> residual1(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    residual1[i] = input[i] + projected[i];
  const auto ln2 = layerNormReference(residual1, dimension, epsilon);
  auto ff1 = matmul(ln2, w1, tokens, dimension, feedForward);
  for (float &value : ff1)
    value = std::max(0.0f, value);
  const auto ff2 = matmul(ff1, w2, tokens, feedForward, dimension);
  std::vector<float> reference(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    reference[i] = residual1[i] + ff2[i];
  Runtime runtime;
  RuntimeOptions options;
  options.captureQnnCallback = false;
  options.qnnLogLevel = 2;
  runtime.setOptions(options);
  std::string error;
  std::vector<float> output;
  if (!runtime.initialize(QnnBackendKind::HTP, error) ||
      !runtime.prepareTinyTransformer(tokens, dimension, feedForward, epsilon,
                                      error) ||
      !runtime.executeTinyTransformer(input, output, error))
    return failure("tiny_transformer_forward", error, runtime);
  const double maximum = maxAbs(reference, output),
               mean = meanAbs(reference, output);
  const bool ok = finite(output) && maximum < 2.0e-2 && mean < 5.0e-3;
  std::ostringstream report;
  report << std::setprecision(10)
         << "TRANSFORMER_MICRO\ntest=tiny_transformer_forward\nmethod=FUSED_"
            "QNN_GRAPH\nshape=B1_T4_D16_F32_H1\nblock=pre_ln_attention_"
            "residual_pre_ln_ffn_residual\nmax_abs_error="
         << maximum << "\nmean_abs_error=" << mean
         << "\nnan_inf=" << (finite(output) ? "false" : "true")
         << "\ngraph_finalize=SUCCESS\ngraph_execute=SUCCESS\nhtp_graph_"
            "boundary_count=1\nhtp_graph_execute_count="
         << runtime.metrics().graphExecuteCount
         << "\ncpu_responsibility=input_and_reference_only\nhtp_responsibility="
            "all_transformer_block_ops\ncpu_fallback=false\nstatus="
         << (ok ? "SUCCESS" : "FAILED") << '\n'
         << runtime.apiTraceSummary() << runtime.diagnostics();
  return report.str();
}
} // namespace
std::string runTransformerExperiment(ExecutionMode mode) {
  if (mode == ExecutionMode::QNN_HTP_LAYER_NORM_CHECK)
    return layerNormCheck();
  if (mode == ExecutionMode::QNN_HTP_SOFTMAX_CHECK)
    return softmaxCheck();
  if (mode == ExecutionMode::QNN_HTP_ATTENTION_FORWARD_CHECK)
    return attentionCheck();
  if (mode == ExecutionMode::QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK)
    return tinyTransformerCheck();
  if (mode == ExecutionMode::QNN_HTP_SOFTMAX_BACKWARD_CHECK)
    return softmaxBackwardCheck();
  if (mode == ExecutionMode::QNN_HTP_ATTENTION_BACKWARD_CHECK)
    return attentionBackwardCheck();
  if (mode == ExecutionMode::QNN_HTP_LAYER_NORM_BACKWARD_CHECK)
    return layerNormBackwardCheck();
  return "TRANSFORMER_MICRO\nstatus=FAILED\nerror=unsupported transformer "
         "mode\n";
}
} // namespace phonelm::qnn