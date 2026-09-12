// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "nicopedia_muon_optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace phonelm::nicopedia_muon { namespace {

using Params = qnn::TinyTransformerParameters;
using Clock = std::chrono::steady_clock;

double elapsedUs(Clock::time_point started) {
  return std::chrono::duration<double, std::micro>(Clock::now() - started)
      .count();
}

bool finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

std::string statePrefix(const char* stateName) {
  return std::string("MUON_") +
      (stateName && *stateName ? stateName : "STATE") + "_";
}

bool validateStateRegistry(
    const std::vector<tiny_lm::ParameterInfo>& expected,
    const std::vector<tiny_lm::ParameterInfo>& actual,
    const char* stateName, std::string* error) {
  const std::string prefix = statePrefix(stateName);
  std::string registryError;
  if (!tiny_lm::validateParameterRegistry(expected, &registryError))
    return fail(error, prefix + "EXPECTED_REGISTRY_INVALID:" + registryError);
  if (!tiny_lm::validateParameterRegistry(actual, &registryError))
    return fail(error, prefix + "REGISTRY_INVALID:" + registryError);
  if (actual.size() != expected.size())
    return fail(error, prefix + "COUNT_MISMATCH");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& expectedEntry = expected[index];
    const auto& actualEntry = actual[index];
    if (actualEntry.name != expectedEntry.name)
      return fail(error, prefix + "NAME_MISMATCH:" + expectedEntry.name);
    if (actualEntry.role != expectedEntry.role)
      return fail(error, prefix + "ROLE_MISMATCH:" + expectedEntry.name);
    if (actualEntry.shape != expectedEntry.shape)
      return fail(error, prefix + "SHAPE_MISMATCH:" + expectedEntry.name);
    if (actualEntry.fanOut != expectedEntry.fanOut ||
        actualEntry.fanIn != expectedEntry.fanIn)
      return fail(error, prefix + "AXES_MISMATCH:" + expectedEntry.name);
    if (!actualEntry.values || !expectedEntry.values ||
        actualEntry.values->size() != expectedEntry.values->size())
      return fail(error, prefix + "ELEMENT_COUNT_MISMATCH:" +
                           expectedEntry.name);
  }
  return true;
}

bool finiteRegistry(const std::vector<tiny_lm::ParameterInfo>& registry,
                    const char* stateName, std::string* error) {
  const std::string prefix = statePrefix(stateName);
  for (const auto& entry : registry) {
    if (!entry.values || !finite(*entry.values))
      return fail(error, prefix + "NONFINITE:" + entry.name);
  }
  return true;
}

bool finiteRegistryForRole(
    const std::vector<tiny_lm::ParameterInfo>& registry,
    tiny_lm::ParameterRole role, const char* stateName, std::string* error) {
  const std::string prefix = statePrefix(stateName);
  for (const auto& entry : registry) {
    if (entry.role != role) continue;
    if (!entry.values || !finite(*entry.values))
      return fail(error, prefix + "NONFINITE:" + entry.name);
  }
  return true;
}

void transpose(const std::vector<float>& input, std::uint32_t rows,
               std::uint32_t columns, std::vector<float>* output) {
  output->assign(input.size(), 0.0f);
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t column = 0; column < columns; ++column)
      (*output)[std::size_t(column) * rows + row] =
          input[std::size_t(row) * columns + column];
}

bool applyAuxiliaryAdamEntry(
    const tiny_lm::ParameterInfo& parameterInfo,
    const std::vector<float>& gradient, const std::vector<float>& oldM,
    const std::vector<float>& oldV, double correction1, double correction2,
    float learningRate, std::vector<float>* parameter, std::vector<float>* m,
    std::vector<float>* v, StageHealth* health, std::string* error,
    double* arithmeticUs = nullptr,
    double* postUpdateFiniteValidationUs = nullptr) {
  if (!parameter || !m || !v || parameter->size() != gradient.size() ||
      m->size() != gradient.size() || v->size() != gradient.size() ||
      oldM.size() != gradient.size() || oldV.size() != gradient.size())
    return fail(error, "MUON_AUX_ADAM_SIZE_MISMATCH:" + parameterInfo.name);
  Clock::time_point arithmeticStarted;
  if (arithmeticUs) arithmeticStarted = Clock::now();
  for (std::size_t i = 0; i < parameter->size(); ++i) {
    (*m)[i] = 0.9f * oldM[i] + 0.1f * gradient[i];
    (*v)[i] = 0.999f * oldV[i] + 0.001f * gradient[i] * gradient[i];
    const double updateValue = (double((*m)[i]) * correction1) /
        (std::sqrt(double((*v)[i]) * correction2) + 1.0e-8);
    (*parameter)[i] -= learningRate * float(updateValue);
  }
  if (arithmeticUs) *arithmeticUs += elapsedUs(arithmeticStarted);
  Clock::time_point validationStarted;
  if (postUpdateFiniteValidationUs) validationStarted = Clock::now();
  if (!finite(*m) || !finite(*v) || !finite(*parameter)) {
    if (postUpdateFiniteValidationUs)
      *postUpdateFiniteValidationUs += elapsedUs(validationStarted);
    if (health) health->updateFinite = false;
    return fail(error, "MUON_AUX_ADAM_STATE_NONFINITE:" + parameterInfo.name);
  }
  if (postUpdateFiniteValidationUs)
    *postUpdateFiniteValidationUs += elapsedUs(validationStarted);
  return true;
}

}  // namespace

bool validateOptimizerStateRegistry(
    const std::vector<tiny_lm::ParameterInfo>& expected,
    const std::vector<tiny_lm::ParameterInfo>& actual,
    const char* stateName, std::string* error) {
  return validateStateRegistry(expected, actual, stateName, error);
}

bool zeropowerNewtonSchulzFp32(const std::vector<float>& input,
                               std::uint32_t rows, std::uint32_t columns,
                               std::uint32_t steps, std::vector<float>* output,
                               StageHealth* health, std::string* error) {
  if (!output || rows == 0 || columns == 0 || steps == 0 || steps >= 100 ||
      std::uint64_t(rows) * columns != input.size())
    return fail(error, "MUON_NS_SHAPE_OR_STEPS_INVALID");
  if (!finite(input)) {
    if (health) health->gradientFinite = false;
    return fail(error, "MUON_GRADIENT_NONFINITE");
  }
  bool transposed = rows > columns;
  std::vector<float> x;
  if (transposed) {
    transpose(input, rows, columns, &x);
    std::swap(rows, columns);
  } else {
    x = input;
  }
  double normSquared = 0.0;
  for (const float value : x) normSquared += double(value) * value;
  const double denominator = std::sqrt(normSquared) + kNsEpsilon;
  if (!std::isfinite(denominator) || denominator <= 0.0)
    return fail(error, "MUON_NORMALIZATION_INVALID");
  for (float& value : x) value = float(double(value) / denominator);
  if (!finite(x)) {
    if (health) health->normalizedFinite = false;
    return fail(error, "MUON_NORMALIZED_NONFINITE");
  }
  std::vector<float> a(std::size_t(rows) * rows);
  std::vector<float> a2(a.size());
  std::vector<float> next(x.size());
  for (std::uint32_t iteration = 0; iteration < steps; ++iteration) {
    for (std::uint32_t i = 0; i < rows; ++i)
      for (std::uint32_t j = 0; j < rows; ++j) {
        double sum = 0.0;
        for (std::uint32_t k = 0; k < columns; ++k)
          sum += double(x[std::size_t(i) * columns + k]) *
                 x[std::size_t(j) * columns + k];
        a[std::size_t(i) * rows + j] = float(sum);
      }
    for (std::uint32_t i = 0; i < rows; ++i)
      for (std::uint32_t j = 0; j < rows; ++j) {
        double sum = 0.0;
        for (std::uint32_t k = 0; k < rows; ++k)
          sum += double(a[std::size_t(i) * rows + k]) *
                 a[std::size_t(k) * rows + j];
        a2[std::size_t(i) * rows + j] = float(sum);
      }
    for (std::uint32_t i = 0; i < rows; ++i)
      for (std::uint32_t j = 0; j < columns; ++j) {
        double bx = 0.0;
        for (std::uint32_t k = 0; k < rows; ++k) {
          const double b = double(kNsB) * a[std::size_t(i) * rows + k] +
                           double(kNsC) * a2[std::size_t(i) * rows + k];
          bx += b * x[std::size_t(k) * columns + j];
        }
        next[std::size_t(i) * columns + j] =
            float(double(kNsA) * x[std::size_t(i) * columns + j] + bx);
      }
    x.swap(next);
    if (!finite(x)) {
      if (health) health->nsOutputFinite = false;
      return fail(error, "MUON_NS_OUTPUT_NONFINITE");
    }
  }
  if (transposed) transpose(x, rows, columns, output); else *output = std::move(x);
  return true;
}

Result update(const Params& parameters, const Params& gradients,
              const Params& muonMomentum, const Params& auxiliaryAdamM,
              const Params& auxiliaryAdamV, const Config& config) {
  Result result;
  result.parameters = parameters;
  result.muonMomentum = muonMomentum;
  result.auxiliaryAdamM = auxiliaryAdamM;
  result.auxiliaryAdamV = auxiliaryAdamV;
  if (!(config.muonLearningRate > 0.0f) ||
      !(config.auxiliaryAdamLearningRate > 0.0f) ||
      !(config.momentum >= 0.0f && config.momentum < 1.0f) ||
      config.nsSteps == 0 || config.nsSteps >= 100 || config.optimizerStep == 0) {
    result.error = "MUON_CONFIG_INVALID";
    return result;
  }
  std::string registryError;
  const auto p = tiny_lm::parameterRegistry(parameters);
  const auto g = tiny_lm::parameterRegistry(gradients);
  const auto oldMomentum = tiny_lm::parameterRegistry(muonMomentum);
  const auto oldM = tiny_lm::parameterRegistry(auxiliaryAdamM);
  const auto oldV = tiny_lm::parameterRegistry(auxiliaryAdamV);
  auto next = tiny_lm::parameterRegistry(result.parameters);
  auto nextMomentum = tiny_lm::parameterRegistry(result.muonMomentum);
  auto nextM = tiny_lm::parameterRegistry(result.auxiliaryAdamM);
  auto nextV = tiny_lm::parameterRegistry(result.auxiliaryAdamV);
  if (!tiny_lm::validateParameterRegistry(parameters, &registryError) ||
      !validateStateRegistry(p, g, "GRADIENT", &registryError) ||
      !validateStateRegistry(p, oldMomentum, "MOMENTUM", &registryError) ||
      !validateStateRegistry(p, oldM, "AUX_ADAM_M", &registryError) ||
      !validateStateRegistry(p, oldV, "AUX_ADAM_V", &registryError) ||
      !validateStateRegistry(p, next, "PARAMETERS", &registryError) ||
      !validateStateRegistry(p, nextMomentum, "NEXT_MOMENTUM",
                             &registryError) ||
      !validateStateRegistry(p, nextM, "NEXT_AUX_ADAM_M", &registryError) ||
      !validateStateRegistry(p, nextV, "NEXT_AUX_ADAM_V", &registryError)) {
    result.error = "MUON_REGISTRY_INVALID:" + registryError;
    return result;
  }
  if (!finiteRegistry(p, "PARAMETERS", &registryError)) {
    result.health.parametersFinite = false;
    result.error = registryError;
    return result;
  }
  if (!finiteRegistry(g, "GRADIENT", &registryError)) {
    result.health.gradientFinite = false;
    result.error = registryError;
    return result;
  }
  if (!finiteRegistry(oldMomentum, "MOMENTUM", &registryError)) {
    result.health.momentumFinite = false;
    result.error = registryError;
    return result;
  }
  if (!finiteRegistry(oldM, "AUX_ADAM_M", &registryError) ||
      !finiteRegistry(oldV, "AUX_ADAM_V", &registryError)) {
    result.health.updateFinite = false;
    result.error = registryError;
    return result;
  }
  const double correction1 = 1.0 / (1.0 - std::pow(0.9, double(config.optimizerStep)));
  const double correction2 = 1.0 / (1.0 - std::pow(0.999, double(config.optimizerStep)));
  for (std::size_t index = 0; index < p.size(); ++index) {
    if (p[index].name != g[index].name || p[index].name != next[index].name ||
        p[index].role != g[index].role || p[index].shape != g[index].shape) {
      result.error = "MUON_REGISTRY_MISMATCH";
      return result;
    }
    const auto& grad = *g[index].values;
    if (!finite(grad)) {
      result.health.gradientFinite = false;
      result.error = "MUON_GRADIENT_NONFINITE:" + p[index].name;
      return result;
    }
    auto& parameter = *const_cast<std::vector<float>*>(next[index].values);
    if (p[index].role == tiny_lm::ParameterRole::MUON) {
      const auto started = std::chrono::steady_clock::now();
      auto& momentum = *const_cast<std::vector<float>*>(nextMomentum[index].values);
      const auto& prior = *oldMomentum[index].values;
      std::vector<float> nesterov(grad.size());
      for (std::size_t i = 0; i < grad.size(); ++i) {
        momentum[i] = config.momentum * prior[i] + (1.0f - config.momentum) * grad[i];
        nesterov[i] = config.nesterov
            ? (1.0f - config.momentum) * grad[i] + config.momentum * momentum[i]
            : momentum[i];
      }
      if (!finite(momentum)) {
        result.health.momentumFinite = false;
        result.error = "MUON_MOMENTUM_NONFINITE:" + p[index].name;
        return result;
      }
      std::vector<float> orthogonal;
      if (!zeropowerNewtonSchulzFp32(nesterov, p[index].shape[0],
                                     p[index].shape[1], config.nsSteps,
                                     &orthogonal, &result.health, &result.error)) {
        result.error += ":" + p[index].name;
        return result;
      }
      const float scale = std::sqrt(std::max(
          1.0f, float(p[index].fanOut) / float(p[index].fanIn)));
      for (std::size_t i = 0; i < parameter.size(); ++i)
        parameter[i] -= config.muonLearningRate * scale * orthogonal[i];
      ++result.muonMatrixCount;
      result.muonParameterCount += parameter.size();
      result.muonMicroseconds += std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - started).count();
    } else if (p[index].role == tiny_lm::ParameterRole::AUX_ADAM) {
      const auto started = std::chrono::steady_clock::now();
      auto& m = *const_cast<std::vector<float>*>(nextM[index].values);
      auto& v = *const_cast<std::vector<float>*>(nextV[index].values);
      const auto& oldMValues = *oldM[index].values;
      const auto& oldVValues = *oldV[index].values;
      if (!applyAuxiliaryAdamEntry(
              p[index], grad, oldMValues, oldVValues, correction1, correction2,
              config.auxiliaryAdamLearningRate, &parameter, &m, &v,
              &result.health, &result.error))
        return result;
      result.auxiliaryAdamParameterCount += parameter.size();
      result.auxiliaryAdamMicroseconds += std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - started).count();
    } else {
      result.error = "MUON_PARAMETER_UNCLASSIFIED:" + p[index].name;
      return result;
    }
    if (!finite(parameter)) {
      result.health.updateFinite = false;
      result.health.parametersFinite = false;
      result.error = "MUON_PARAMETER_UPDATE_NONFINITE:" + p[index].name;
      return result;
    }
  }
  return result;
}

Result updateAuxiliaryAdamOnly(
    const Params& parameters, const Params& gradients,
    const Params& muonMomentum, const Params& auxiliaryAdamM,
    const Params& auxiliaryAdamV, const Config& config) {
  Result result;
  auto timingStarted = Clock::now();
  result.parameters = parameters;
  result.auxiliaryAdamTimings.parameterCopyUs = elapsedUs(timingStarted);
  timingStarted = Clock::now();
  result.muonMomentum = muonMomentum;
  result.auxiliaryAdamTimings.muonMomentumCopyUs = elapsedUs(timingStarted);
  timingStarted = Clock::now();
  result.auxiliaryAdamM = auxiliaryAdamM;
  result.auxiliaryAdamTimings.auxiliaryAdamMCopyUs = elapsedUs(timingStarted);
  timingStarted = Clock::now();
  result.auxiliaryAdamV = auxiliaryAdamV;
  result.auxiliaryAdamTimings.auxiliaryAdamVCopyUs = elapsedUs(timingStarted);
  (void)updateAuxiliaryAdamCandidateInPlace(
      gradients, auxiliaryAdamM, auxiliaryAdamV, config, false, &result);
  return result;
}

bool updateAuxiliaryAdamCandidateInPlace(
    const Params& gradients, const Params& oldAuxiliaryAdamM,
    const Params& oldAuxiliaryAdamV, const Config& config,
    bool muonInputsAlreadyFinite, Result* result) {
  if (!result) return false;
  if (!(config.auxiliaryAdamLearningRate > 0.0f) ||
      config.optimizerStep == 0) {
    result->error = "MUON_CONFIG_INVALID";
    return false;
  }
  std::string registryError;
  auto timingStarted = Clock::now();
  const auto p = tiny_lm::parameterRegistry(result->parameters);
  const auto g = tiny_lm::parameterRegistry(gradients);
  const auto mu = tiny_lm::parameterRegistry(result->muonMomentum);
  const auto oldM = tiny_lm::parameterRegistry(oldAuxiliaryAdamM);
  const auto oldV = tiny_lm::parameterRegistry(oldAuxiliaryAdamV);
  auto next = tiny_lm::parameterRegistry(result->parameters);
  auto nextM = tiny_lm::parameterRegistry(result->auxiliaryAdamM);
  auto nextV = tiny_lm::parameterRegistry(result->auxiliaryAdamV);
  result->auxiliaryAdamTimings.registryConstructionUs =
      elapsedUs(timingStarted);
  timingStarted = Clock::now();
  if (!tiny_lm::validateParameterRegistry(result->parameters, &registryError) ||
      !validateStateRegistry(p, g, "GRADIENT", &registryError) ||
      !validateStateRegistry(p, mu, "MOMENTUM", &registryError) ||
      !validateStateRegistry(p, oldM, "AUX_ADAM_M", &registryError) ||
      !validateStateRegistry(p, oldV, "AUX_ADAM_V", &registryError) ||
      !validateStateRegistry(p, next, "PARAMETERS", &registryError) ||
      !validateStateRegistry(p, nextM, "NEXT_AUX_ADAM_M", &registryError) ||
      !validateStateRegistry(p, nextV, "NEXT_AUX_ADAM_V", &registryError)) {
    result->auxiliaryAdamTimings.registryValidationUs =
        elapsedUs(timingStarted);
    result->error = "MUON_REGISTRY_INVALID:" + registryError;
    return false;
  }
  result->auxiliaryAdamTimings.registryValidationUs = elapsedUs(timingStarted);
  const auto preValidationStarted = Clock::now();
  timingStarted = Clock::now();
  const auto finiteRequired = [&](const auto& registry,
                                  const char* stateName) {
    return muonInputsAlreadyFinite
        ? finiteRegistryForRole(registry, tiny_lm::ParameterRole::AUX_ADAM,
                                stateName, &registryError)
        : finiteRegistry(registry, stateName, &registryError);
  };
  if (!finiteRequired(p, "PARAMETERS")) {
    result->auxiliaryAdamTimings.preParameterFiniteValidationUs =
        elapsedUs(timingStarted);
    result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
        elapsedUs(preValidationStarted);
    result->health.parametersFinite = false;
    result->error = registryError;
    return false;
  }
  result->auxiliaryAdamTimings.preParameterFiniteValidationUs =
      elapsedUs(timingStarted);
  timingStarted = Clock::now();
  if (!finiteRequired(g, "GRADIENT")) {
    result->auxiliaryAdamTimings.preGradientFiniteValidationUs =
        elapsedUs(timingStarted);
    result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
        elapsedUs(preValidationStarted);
    result->health.gradientFinite = false;
    result->error = registryError;
    return false;
  }
  result->auxiliaryAdamTimings.preGradientFiniteValidationUs =
      elapsedUs(timingStarted);
  timingStarted = Clock::now();
  if (!muonInputsAlreadyFinite &&
      !finiteRegistry(mu, "MOMENTUM", &registryError)) {
    result->auxiliaryAdamTimings.preMomentumFiniteValidationUs =
        elapsedUs(timingStarted);
    result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
        elapsedUs(preValidationStarted);
    result->health.momentumFinite = false;
    result->error = registryError;
    return false;
  }
  result->auxiliaryAdamTimings.preMomentumFiniteValidationUs =
      elapsedUs(timingStarted);
  timingStarted = Clock::now();
  if (!finiteRequired(oldM, "AUX_ADAM_M")) {
    result->auxiliaryAdamTimings.preAdamMFiniteValidationUs =
        elapsedUs(timingStarted);
    result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
        elapsedUs(preValidationStarted);
    result->health.updateFinite = false;
    result->error = registryError;
    return false;
  }
  result->auxiliaryAdamTimings.preAdamMFiniteValidationUs =
      elapsedUs(timingStarted);
  timingStarted = Clock::now();
  if (!finiteRequired(oldV, "AUX_ADAM_V")) {
    result->auxiliaryAdamTimings.preAdamVFiniteValidationUs =
        elapsedUs(timingStarted);
    result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
        elapsedUs(preValidationStarted);
    result->health.updateFinite = false;
    result->error = registryError;
    return false;
  }
  result->auxiliaryAdamTimings.preAdamVFiniteValidationUs =
      elapsedUs(timingStarted);
  result->auxiliaryAdamTimings.preUpdateFiniteValidationUs =
      elapsedUs(preValidationStarted);
  const double correction1 =
      1.0 / (1.0 - std::pow(0.9, double(config.optimizerStep)));
  const double correction2 =
      1.0 / (1.0 - std::pow(0.999, double(config.optimizerStep)));
  for (std::size_t index = 0; index < p.size(); ++index) {
    if (p[index].role == tiny_lm::ParameterRole::MUON) continue;
    if (p[index].role != tiny_lm::ParameterRole::AUX_ADAM) {
      result->error = "MUON_PARAMETER_UNCLASSIFIED:" + p[index].name;
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    auto& parameter = *const_cast<std::vector<float>*>(next[index].values);
    auto& m = *const_cast<std::vector<float>*>(nextM[index].values);
    auto& v = *const_cast<std::vector<float>*>(nextV[index].values);
    if (!applyAuxiliaryAdamEntry(
            p[index], *g[index].values, *oldM[index].values,
            *oldV[index].values, correction1, correction2,
            config.auxiliaryAdamLearningRate, &parameter, &m, &v,
            &result->health, &result->error,
            &result->auxiliaryAdamTimings.arithmeticUs,
            &result->auxiliaryAdamTimings.postUpdateFiniteValidationUs))
      return false;
    result->auxiliaryAdamParameterCount += parameter.size();
    result->auxiliaryAdamMicroseconds +=
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - started).count();
  }
  return true;
}

}  // namespace phonelm::nicopedia_muon
