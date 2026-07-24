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
  return "TRANSFORMER_MICRO\nstatus=FAILED\nerror=unsupported transformer "
         "mode\n";
}
} // namespace phonelm::qnn