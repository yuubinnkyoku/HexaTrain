// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "nicopedia_hvx_muon.h"

#include "nicopedia_htp_muon.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

#if PHONELM_ENABLE_HVX_MUON
#include "hexatrain_hvx_probe.h"
#include <remote.h>
#include <rpcmem.h>
#endif

namespace phonelm::nicopedia_hvx_muon { namespace {

using Clock = std::chrono::steady_clock;
double elapsedUs(Clock::time_point started) {
  return std::chrono::duration<double, std::micro>(Clock::now() - started).count();
}

constexpr std::size_t kSquareElements = 64 * 64;
constexpr std::size_t kRectElements = 64 * 128;
constexpr std::size_t kInputFloats = 76 * 3 * kSquareElements + 38 * 3 * kRectElements;
constexpr std::size_t kHyperFloats = 2 * (76 + 38);
constexpr std::size_t kOutputFloats = 76 * 2 * kSquareElements + 38 * 2 * kRectElements;

template <typename T> bool allFinite(const T* values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

#if PHONELM_ENABLE_HVX_MUON
void appendInterleavedGroup(const std::vector<float>& current,
                            const std::vector<float>& gradient,
                            const std::vector<float>& momentum,
                            std::size_t elements, float** destination) {
  const std::size_t matrices = current.size() / elements;
  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    const std::size_t offset = matrix * elements;
    std::memcpy(*destination, current.data() + offset, elements * sizeof(float));
    *destination += elements;
    std::memcpy(*destination, gradient.data() + offset, elements * sizeof(float));
    *destination += elements;
    std::memcpy(*destination, momentum.data() + offset, elements * sizeof(float));
    *destination += elements;
  }
}

class Session {
 public:
  ~Session() { reset(); }

  void reset() {
    if (handle_) (void)hexatrain_hvx_probe_close(handle_);
    if (input_) rpcmem_free(input_);
    if (hyper_) rpcmem_free(hyper_);
    if (output_) rpcmem_free(output_);
    if (metadata_) rpcmem_free(metadata_);
    handle_ = 0;
    input_ = nullptr;
    hyper_ = nullptr;
    output_ = nullptr;
    metadata_ = nullptr;
    initialized_ = false;
  }

  bool initialize(std::string* error) {
    if (initialized_) return true;
    input_ = static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS, kInputFloats * sizeof(float)));
    hyper_ = static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS, kHyperFloats * sizeof(float)));
    output_ = static_cast<float*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS, kOutputFloats * sizeof(float)));
    metadata_ = static_cast<int*>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS, 32 * sizeof(int)));
    if (!input_ || !hyper_ || !output_ || !metadata_) {
      *error = "HVX_RPCMEM_ALLOCATION_FAILED";
      reset();
      return false;
    }
    struct remote_rpc_control_unsigned_module unsignedPd = {CDSP_DOMAIN_ID, 1};
    if (remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsignedPd,
                               sizeof(unsignedPd)) != 0) {
      *error = "HVX_UNSIGNED_PD_CONFIGURATION_FAILED";
      reset();
      return false;
    }
    const std::string uri = std::string(hexatrain_hvx_probe_URI) + "&_dom=cdsp";
    int status = hexatrain_hvx_probe_open(uri.c_str(), &handle_);
    if (status) {
      *error = "HVX_RPC_OPEN_FAILED:" + std::to_string(status);
      reset();
      return false;
    }
    int configure[4] = {};
    status = hexatrain_hvx_probe_configure(handle_, 1, configure, 4);
    if (status || configure[1] < 8 || configure[2] != 1) {
      *error = "HVX_PERFORMANCE_CONFIGURATION_FAILED:" + std::to_string(status);
      reset();
      return false;
    }
    initialized_ = true;
    return true;
  }

  std::mutex mutex;
  float* input_ = nullptr;
  float* hyper_ = nullptr;
  float* output_ = nullptr;
  int* metadata_ = nullptr;
  remote_handle64 handle_ = 0;
  bool initialized_ = false;
};

Session& session() { static Session value; return value; }
#endif

template <std::size_t N> double median(std::array<double, N> values) {
  std::sort(values.begin(), values.end());
  return values[N / 2];
}
template <std::size_t N> double best(const std::array<double, N>& values) {
  return *std::min_element(values.begin(), values.end());
}
template <std::size_t N> double mean(const std::array<double, N>& values) {
  double sum = 0.0; for (double value : values) sum += value; return sum / N;
}

struct Difference {
  double maxAbs = 0.0;
  double relativeL2 = 0.0;
  double cosine = 1.0;
  bool finite = true;
};

Difference difference(const std::vector<float>& reference,
                      const std::vector<float>& actual) {
  Difference metric;
  if (reference.size() != actual.size() || reference.empty()) {
    metric.finite = false;
    return metric;
  }
  double diff2 = 0.0, reference2 = 0.0, actual2 = 0.0, dot = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
      metric.finite = false;
      continue;
    }
    const double delta = double(reference[i]) - actual[i];
    metric.maxAbs = std::max(metric.maxAbs, std::abs(delta));
    diff2 += delta * delta;
    reference2 += double(reference[i]) * reference[i];
    actual2 += double(actual[i]) * actual[i];
    dot += double(reference[i]) * actual[i];
  }
  metric.relativeL2 = reference2 > 0.0
      ? std::sqrt(diff2 / reference2)
      : (diff2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
  if (reference2 > 0.0 && actual2 > 0.0)
    metric.cosine = dot / std::sqrt(reference2 * actual2);
  else
    metric.cosine = reference2 == actual2 ? 1.0 : 0.0;
  return metric;
}

bool parityPass(const Difference& metric) {
  return metric.finite && metric.maxAbs <= 0.002 &&
      metric.relativeL2 <= 0.001 && metric.cosine >= 0.99999;
}

}  // namespace

Result update(const qnn::TinyTransformerParameters& parameters,
              const qnn::TinyTransformerParameters& gradients,
              const qnn::TinyTransformerParameters& muonMomentum,
              const qnn::TinyTransformerParameters& auxiliaryAdamM,
              const qnn::TinyTransformerParameters& auxiliaryAdamV,
              const nicopedia_muon::Config& config) {
  Result result;
  const auto totalStarted = Clock::now();
#if !PHONELM_ENABLE_HVX_MUON
  (void)parameters; (void)gradients; (void)muonMomentum;
  (void)auxiliaryAdamM; (void)auxiliaryAdamV; (void)config;
  result.update.error = "HVX_MUON_NOT_BUILT";
  result.rpcStatus = -1;
  result.timings.totalUs = elapsedUs(totalStarted);
  return result;
#else
  if (!(config.muonLearningRate > 0.0f) ||
      !std::isfinite(config.muonLearningRate) ||
      !(config.auxiliaryAdamLearningRate > 0.0f) ||
      !std::isfinite(config.auxiliaryAdamLearningRate) ||
      config.momentum != 0.95f || !config.nesterov || config.nsSteps != 5 ||
      config.optimizerStep == 0) {
    result.update.error = "HVX_MUON_CONFIG_IDENTITY_MISMATCH";
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  Session& rpc = session();
  std::lock_guard<std::mutex> lock(rpc.mutex);
  if (!rpc.initialize(&result.update.error)) {
    result.rpcStatus = -1;
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  nicopedia_htp_muon::PackedInputs packed;
  auto phase = Clock::now();
  if (!nicopedia_htp_muon::packForValidatedRpc(
          parameters, gradients, muonMomentum, &packed, &result.update.error)) {
    result.timings.packUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  float* input = rpc.input_;
  appendInterleavedGroup(packed.currentSquare, packed.gradientSquare,
                         packed.momentumSquare, kSquareElements, &input);
  appendInterleavedGroup(packed.currentRectangular,
                         packed.gradientRectangular,
                         packed.momentumRectangular, kRectElements, &input);
  float* hyper = rpc.hyper_;
  for (float scale : packed.scaleSquare) {
    *hyper++ = config.muonLearningRate; *hyper++ = scale;
  }
  for (float scale : packed.scaleRectangular) {
    *hyper++ = config.muonLearningRate; *hyper++ = scale;
  }
  result.timings.packUs = elapsedUs(phase);

  phase = Clock::now();
  if (!allFinite(rpc.input_, kInputFloats) ||
      !allFinite(rpc.hyper_, kHyperFloats)) {
    if (result.update.error.empty()) result.update.error = "HVX_RPC_INPUT_NONFINITE";
    result.timings.inputValidationUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  std::fill_n(rpc.output_, kOutputFloats,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(rpc.metadata_, 32, 0);
  result.timings.inputValidationUs = elapsedUs(phase);

  phase = Clock::now();
  result.rpcStatus = hexatrain_hvx_probe_run(
      rpc.handle_, 5, 8, 0, 0, rpc.input_, static_cast<int>(kInputFloats),
      rpc.hyper_, static_cast<int>(kHyperFloats), rpc.output_,
      static_cast<int>(kOutputFloats), rpc.metadata_, 32);
  result.timings.rpcUs = elapsedUs(phase);
  result.timings.kernelUs = rpc.metadata_[3];
  if (result.rpcStatus || rpc.metadata_[0] != 0x48565831 ||
      rpc.metadata_[4] || rpc.metadata_[6] != 1 || rpc.metadata_[7] != 5 ||
      rpc.metadata_[10] != 8 || rpc.metadata_[11] != 8) {
    result.update.error = "HVX_RPC_EXECUTE_FAILED:" +
        std::to_string(result.rpcStatus) + ":kernel=" +
        std::to_string(rpc.metadata_[4]);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }

  phase = Clock::now();
  result.outputFinite = allFinite(rpc.output_, kOutputFloats);
  result.timings.outputValidationUs = elapsedUs(phase);
  if (!result.outputFinite) {
    result.update.error = "HVX_RPC_OUTPUT_NONFINITE";
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }

  phase = Clock::now();
  std::vector<float> squareWeights(76 * kSquareElements);
  std::vector<float> squareMomentum(76 * kSquareElements);
  std::vector<float> rectWeights(38 * kRectElements);
  std::vector<float> rectMomentum(38 * kRectElements);
  for (std::size_t i = 0; i < 76; ++i) {
    const float* source = rpc.output_ + i * 2 * kSquareElements;
    std::copy_n(source, kSquareElements, squareWeights.begin() + i * kSquareElements);
    std::copy_n(source + kSquareElements, kSquareElements,
                squareMomentum.begin() + i * kSquareElements);
  }
  const float* rectOutput = rpc.output_ + 76 * 2 * kSquareElements;
  for (std::size_t i = 0; i < 38; ++i) {
    const float* source = rectOutput + i * 2 * kRectElements;
    std::copy_n(source, kRectElements, rectWeights.begin() + i * kRectElements);
    std::copy_n(source + kRectElements, kRectElements,
                rectMomentum.begin() + i * kRectElements);
  }
  auto candidateParameters = parameters;
  auto candidateMomentum = muonMomentum;
  if (!nicopedia_htp_muon::unpack(packed, squareWeights, squareMomentum,
          rectWeights, rectMomentum, &candidateParameters, &candidateMomentum,
          &result.update.error)) {
    result.timings.unpackApplyUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  result.update = nicopedia_muon::updateAuxiliaryAdamOnly(
      candidateParameters, gradients, candidateMomentum, auxiliaryAdamM,
      auxiliaryAdamV, config);
  if (result.update.error.empty()) {
    result.update.muonMatrixCount = 114;
    result.update.muonParameterCount = 622592;
  }
  result.timings.unpackApplyUs = elapsedUs(phase);
  result.timings.totalUs = elapsedUs(totalStarted);
  if (result.update.error.empty()) {
    result.update.muonMicroseconds = std::max(
        0.0, result.timings.totalUs - result.update.auxiliaryAdamMicroseconds);
  }
  return result;
#endif
}

std::string benchmarkActualOptimizerPath() {
  tiny_lm::Config model{1024, 32, 64, 128, 1e-5f, 19, 2};
  const auto parameters = tiny_lm::initialParameters(model, 1);
  const auto gradients = parameters;
  auto momentum = parameters;
  for (const auto& entry : tiny_lm::parameterRegistry(momentum))
    std::fill(const_cast<std::vector<float>*>(entry.values)->begin(),
              const_cast<std::vector<float>*>(entry.values)->end(), 0.0f);
  const auto adamM = momentum, adamV = momentum;
  nicopedia_muon::Config config;
  config.muonLearningRate = 0.01f;
  config.auxiliaryAdamLearningRate = 0.0022f;
  config.optimizerStep = 1;

  constexpr std::size_t kMeasured = 5;
  std::array<double, kMeasured> cpu{}, hvx{}, pack{}, inputValidation{}, rpc{},
      kernel{}, outputValidation{}, unpackApply{};
  nicopedia_muon::Result cpuReference;
  Result hvxReference;
  for (std::size_t repetition = 0; repetition <= kMeasured; ++repetition) {
    auto started = Clock::now();
    auto cpuResult = nicopedia_muon::update(
        parameters, gradients, momentum, adamM, adamV, config);
    const double cpuUs = elapsedUs(started);
    auto hvxResult = nicopedia_hvx_muon::update(
        parameters, gradients, momentum, adamM, adamV, config);
    if (!cpuResult.error.empty() || !hvxResult.update.error.empty())
      return "HVX_MUON_OPTIMIZER_BENCHMARK\nstatus=FAILED\nerror=" +
          (cpuResult.error.empty() ? hvxResult.update.error : cpuResult.error) +
          "\nfallback=false\n";
    if (repetition == kMeasured) {
      cpuReference = std::move(cpuResult);
      hvxReference = std::move(hvxResult);
    }
    if (repetition == 0) continue;
    const std::size_t index = repetition - 1;
    cpu[index] = cpuUs; hvx[index] = hvxResult.timings.totalUs;
    pack[index] = hvxResult.timings.packUs;
    inputValidation[index] = hvxResult.timings.inputValidationUs;
    rpc[index] = hvxResult.timings.rpcUs; kernel[index] = hvxResult.timings.kernelUs;
    outputValidation[index] = hvxResult.timings.outputValidationUs;
    unpackApply[index] = hvxResult.timings.unpackApplyUs;
  }
  const auto cpuP = tiny_lm::parameterRegistry(cpuReference.parameters);
  const auto hvxP = tiny_lm::parameterRegistry(hvxReference.update.parameters);
  const auto cpuM = tiny_lm::parameterRegistry(cpuReference.muonMomentum);
  const auto hvxM = tiny_lm::parameterRegistry(hvxReference.update.muonMomentum);
  std::size_t parityMatrices = 0;
  double maxAbs = 0.0, worstRelativeL2 = 0.0, minimumCosine = 1.0;
  bool parity = cpuP.size() == hvxP.size() && cpuP.size() == cpuM.size() &&
      cpuP.size() == hvxM.size();
  for (std::size_t entry = 0; entry < cpuP.size(); ++entry) {
    if (cpuP[entry].role != tiny_lm::ParameterRole::MUON) continue;
    const Difference weight = difference(*cpuP[entry].values, *hvxP[entry].values);
    const Difference state = difference(*cpuM[entry].values, *hvxM[entry].values);
    const bool identity = cpuP[entry].name == hvxP[entry].name &&
        cpuM[entry].name == hvxM[entry].name;
    const bool matrixPass = identity && parityPass(weight) && parityPass(state);
    if (matrixPass) ++parityMatrices;
    parity = parity && matrixPass;
    maxAbs = std::max({maxAbs, weight.maxAbs, state.maxAbs});
    worstRelativeL2 = std::max(
        {worstRelativeL2, weight.relativeL2, state.relativeL2});
    minimumCosine = std::min({minimumCosine, weight.cosine, state.cosine});
  }
  parity = parity && parityMatrices == 114;
  std::ostringstream report;
  report << std::setprecision(10)
         << "HVX_MUON_OPTIMIZER_BENCHMARK\nstatus=" << (parity ? "SUCCESS" : "FAILED")
         << "\nwarmup_repetitions=1\nmeasured_repetitions=5"
         << "\ncpu_optimizer_step_best_us=" << best(cpu)
         << "\ncpu_optimizer_step_median_us=" << median(cpu)
         << "\ncpu_optimizer_step_mean_us=" << mean(cpu)
         << "\nhvx_optimizer_step_best_us=" << best(hvx)
         << "\nhvx_optimizer_step_median_us=" << median(hvx)
         << "\nhvx_optimizer_step_mean_us=" << mean(hvx)
         << "\npack_best_us=" << best(pack) << "\npack_median_us=" << median(pack)
         << "\npack_mean_us=" << mean(pack)
         << "\ninput_validation_best_us=" << best(inputValidation)
         << "\ninput_validation_median_us=" << median(inputValidation)
         << "\ninput_validation_mean_us=" << mean(inputValidation)
         << "\nrpc_best_us=" << best(rpc) << "\nrpc_median_us=" << median(rpc)
         << "\nrpc_mean_us=" << mean(rpc)
         << "\nkernel_best_us=" << best(kernel) << "\nkernel_median_us=" << median(kernel)
         << "\nkernel_mean_us=" << mean(kernel)
         << "\noutput_validation_best_us=" << best(outputValidation)
         << "\noutput_validation_median_us=" << median(outputValidation)
         << "\noutput_validation_mean_us=" << mean(outputValidation)
         << "\nunpack_apply_best_us=" << best(unpackApply)
         << "\nunpack_apply_median_us=" << median(unpackApply)
         << "\nunpack_apply_mean_us=" << mean(unpackApply)
         << "\nend_to_end_speedup_best=" << best(cpu) / best(hvx)
         << "\nend_to_end_speedup_median=" << median(cpu) / median(hvx)
         << "\nend_to_end_speedup_mean=" << mean(cpu) / mean(hvx)
         << "\nfull_parity_matrices=" << parityMatrices
         << "\nfull_parity_total_matrices=114\nfull_parity_max_abs=" << maxAbs
         << "\nfull_parity_worst_relative_l2=" << worstRelativeL2
         << "\nfull_parity_minimum_cosine=" << minimumCosine
         << "\nrpc_status_success=true\noutput_tensors_finite="
         << (hvxReference.outputFinite ? "true" : "false")
         << "\nfallback=false\n";
  return report.str();
}

}  // namespace phonelm::nicopedia_hvx_muon
