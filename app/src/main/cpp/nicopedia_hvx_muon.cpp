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
constexpr std::size_t kMetadataInts = 64;

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
    packed_ = {};
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
        RPCMEM_DEFAULT_FLAGS, kMetadataInts * sizeof(int)));
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
  nicopedia_htp_muon::PackedInputs packed_;
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
  auto phase = Clock::now();
  std::lock_guard<std::mutex> lock(rpc.mutex);
  result.timings.mutexWaitUs = elapsedUs(phase);
  phase = Clock::now();
  if (!rpc.initialize(&result.update.error)) {
    result.rpcStatus = -1;
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  result.timings.sessionSetupUs = elapsedUs(phase);
  auto& packed = rpc.packed_;
  nicopedia_htp_muon::PackTimings packTimings;
  phase = Clock::now();
  if (!nicopedia_htp_muon::packForValidatedRpc(
          parameters, gradients, muonMomentum, &packed, &result.update.error,
          &packTimings)) {
    result.timings.packUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  result.timings.packRegistryTraversalUs = packTimings.registryTraversalUs;
  result.timings.packAllocationResizeUs = packTimings.allocationResizeUs;
  result.timings.packActualReallocationUs = packTimings.actualReallocationUs;
  result.timings.packResizeGrowthInitializationUs =
      packTimings.resizeGrowthInitializationUs;
  result.timings.packResizeOtherUs = packTimings.resizeOtherUs;
  result.timings.packActualReallocationCount =
      packTimings.actualReallocationCount;
  result.timings.packResizeGrowthInitializationCount =
      packTimings.resizeGrowthInitializationCount;
  result.timings.packMetadataSetupUs = packTimings.metadataSetupUs;
  result.timings.packSquareWeightCopyUs = packTimings.squareWeightCopyUs;
  result.timings.packW1WeightCopyUs = packTimings.w1WeightCopyUs;
  result.timings.packW2WeightTransposeUs = packTimings.w2WeightTransposeUs;
  result.timings.packGradientCopyUs = packTimings.gradientCopyUs;
  result.timings.packMomentumCopyUs = packTimings.momentumCopyUs;
  auto subphase = Clock::now();
  float* input = rpc.input_;
  appendInterleavedGroup(packed.currentSquare, packed.gradientSquare,
                         packed.momentumSquare, kSquareElements, &input);
  appendInterleavedGroup(packed.currentRectangular,
                          packed.gradientRectangular,
                          packed.momentumRectangular, kRectElements, &input);
  result.timings.packFlatRpcCopyUs = elapsedUs(subphase);
  subphase = Clock::now();
  float* hyper = rpc.hyper_;
  for (float scale : packed.scaleSquare) {
    *hyper++ = config.muonLearningRate; *hyper++ = scale;
  }
  for (float scale : packed.scaleRectangular) {
    *hyper++ = config.muonLearningRate; *hyper++ = scale;
  }
  result.timings.packHyperSetupUs = elapsedUs(subphase);
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
  std::fill_n(rpc.metadata_, kMetadataInts, 0);
  result.timings.inputValidationUs = elapsedUs(phase);

  phase = Clock::now();
  result.rpcStatus = hexatrain_hvx_probe_run(
      rpc.handle_, 5, 8, 0, 0, rpc.input_, static_cast<int>(kInputFloats),
      rpc.hyper_, static_cast<int>(kHyperFloats), rpc.output_,
      static_cast<int>(kOutputFloats), rpc.metadata_,
      static_cast<int>(kMetadataInts));
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
  subphase = Clock::now();
  nicopedia_muon::Result candidateUpdate;
  auto candidateCopyStarted = Clock::now();
  candidateUpdate.parameters = parameters;
  result.timings.unpackCandidateParameterCopyUs =
      elapsedUs(candidateCopyStarted);
  candidateCopyStarted = Clock::now();
  candidateUpdate.muonMomentum = muonMomentum;
  result.timings.unpackCandidateMomentumCopyUs =
      elapsedUs(candidateCopyStarted);
  result.timings.unpackCandidateGenerationUs = elapsedUs(subphase);
  nicopedia_htp_muon::UnpackTimings unpackTimings;
  subphase = Clock::now();
  if (!nicopedia_htp_muon::unpackValidatedRpcOutput(
          packed, rpc.output_, kOutputFloats, &candidateUpdate.parameters,
          &candidateUpdate.muonMomentum, &result.update.error,
          &unpackTimings)) {
    result.timings.unpackRpcOutputDecodeUs = elapsedUs(subphase);
    result.timings.unpackApplyUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  result.timings.unpackRpcOutputDecodeUs = elapsedUs(subphase);
  result.timings.unpackDecodedValidationUs = unpackTimings.decodedValidationUs;
  result.timings.unpackRegistryTraversalUs = unpackTimings.registryTraversalUs;
  result.timings.unpackSquareOutputCopyUs = unpackTimings.squareOutputCopyUs;
  result.timings.unpackW1OutputCopyUs = unpackTimings.w1OutputCopyUs;
  result.timings.unpackW2TransposeBackUs = unpackTimings.w2TransposeBackUs;
  subphase = Clock::now();
  auto stateCopyStarted = Clock::now();
  candidateUpdate.auxiliaryAdamM = auxiliaryAdamM;
  candidateUpdate.auxiliaryAdamTimings.auxiliaryAdamMCopyUs =
      elapsedUs(stateCopyStarted);
  stateCopyStarted = Clock::now();
  candidateUpdate.auxiliaryAdamV = auxiliaryAdamV;
  candidateUpdate.auxiliaryAdamTimings.auxiliaryAdamVCopyUs =
      elapsedUs(stateCopyStarted);
  const bool auxiliarySuccess =
      nicopedia_muon::updateAuxiliaryAdamCandidateInPlace(
          // Pack validation covers MUON gradients. Full RPC-output validation
          // plus direct unpack covers candidate MUON parameters and momentum.
          gradients, auxiliaryAdamM, auxiliaryAdamV, config, true,
          &candidateUpdate);
  result.timings.unpackAuxAdamUs = elapsedUs(subphase);
  result.timings.auxParameterCopyUs =
      candidateUpdate.auxiliaryAdamTimings.parameterCopyUs;
  result.timings.auxMuonMomentumCopyUs =
      candidateUpdate.auxiliaryAdamTimings.muonMomentumCopyUs;
  result.timings.auxAdamMCopyUs =
      candidateUpdate.auxiliaryAdamTimings.auxiliaryAdamMCopyUs;
  result.timings.auxAdamVCopyUs =
      candidateUpdate.auxiliaryAdamTimings.auxiliaryAdamVCopyUs;
  result.timings.auxRegistryConstructionUs =
      candidateUpdate.auxiliaryAdamTimings.registryConstructionUs;
  result.timings.auxRegistryValidationUs =
      candidateUpdate.auxiliaryAdamTimings.registryValidationUs;
  result.timings.auxPreUpdateFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preUpdateFiniteValidationUs;
  result.timings.auxPreParameterFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preParameterFiniteValidationUs;
  result.timings.auxPreGradientFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preGradientFiniteValidationUs;
  result.timings.auxPreMomentumFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preMomentumFiniteValidationUs;
  result.timings.auxPreAdamMFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preAdamMFiniteValidationUs;
  result.timings.auxPreAdamVFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.preAdamVFiniteValidationUs;
  result.timings.auxArithmeticUs =
      candidateUpdate.auxiliaryAdamTimings.arithmeticUs;
  result.timings.auxPostUpdateFiniteValidationUs =
      candidateUpdate.auxiliaryAdamTimings.postUpdateFiniteValidationUs;
  if (!auxiliarySuccess) {
    result.update.error = candidateUpdate.error;
    result.update.health = candidateUpdate.health;
    result.timings.unpackApplyUs = elapsedUs(phase);
    result.timings.totalUs = elapsedUs(totalStarted);
    return result;
  }
  subphase = Clock::now();
  result.update = std::move(candidateUpdate);
  result.timings.unpackFinalCommitUs = elapsedUs(subphase);
  if (result.update.error.empty()) {
    // Derive Muon partition counts from the live ParameterDefinition registry
    // rather than restating current-table constants. Aux Adam counts are
    // already produced by updateAuxiliaryAdamCandidateInPlace.
    tiny_lm::ParameterPartition partition;
    std::string partitionError;
    if (!tiny_lm::splitParameterRegistry(result.update.parameters, &partition,
                                         &partitionError)) {
      result.update.error = "HVX_MUON_PARTITION_UNAVAILABLE:" + partitionError;
    } else {
      result.update.muonMatrixCount = 0;
      result.update.muonParameterCount = 0;
      for (const auto& entry : partition.muon) {
        ++result.update.muonMatrixCount;
        result.update.muonParameterCount += entry.values->size();
      }
    }
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
      kernel{}, rpcKernelExternal{}, outputValidation{}, unpackApply{};
  std::array<double, kMeasured> packRegistry{}, packAllocation{}, packMetadata{},
      packSquareWeight{}, packW1Weight{}, packW2Transpose{}, packGradient{},
      packMomentum{}, packFlatRpc{}, packHyper{}, unpackDecode{},
      unpackCandidate{}, unpackCandidateParameterCopy{},
      unpackCandidateMomentumCopy{}, unpackCandidateOther{}, unpackRegistry{},
      unpackSquare{}, unpackW1{}, unpackW2{},
      unpackAuxAdam{}, unpackFinalCommit{}, unpackDecodedValidation{}, mutexWait{},
      sessionSetup{}, packActualReallocation{}, packResizeGrowthInitialization{},
      packResizeOther{}, packActualReallocationCount{},
      packResizeGrowthInitializationCount{}, auxParameterCopy{},
      auxMuonMomentumCopy{}, auxAdamMCopy{}, auxAdamVCopy{},
      auxRegistryConstruction{}, auxRegistryValidation{},
      auxPreUpdateFiniteValidation{}, auxPreParameterFiniteValidation{},
      auxPreGradientFiniteValidation{}, auxPreMomentumFiniteValidation{},
      auxPreAdamMFiniteValidation{}, auxPreAdamVFiniteValidation{},
      auxArithmetic{}, auxPostUpdateFiniteValidation{}, auxOther{};
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
    rpcKernelExternal[index] = std::max(0.0, rpc[index] - kernel[index]);
    outputValidation[index] = hvxResult.timings.outputValidationUs;
    unpackApply[index] = hvxResult.timings.unpackApplyUs;
    packRegistry[index] = hvxResult.timings.packRegistryTraversalUs;
    packAllocation[index] = hvxResult.timings.packAllocationResizeUs;
    packActualReallocation[index] =
        hvxResult.timings.packActualReallocationUs;
    packResizeGrowthInitialization[index] =
        hvxResult.timings.packResizeGrowthInitializationUs;
    packResizeOther[index] = hvxResult.timings.packResizeOtherUs;
    packActualReallocationCount[index] =
        double(hvxResult.timings.packActualReallocationCount);
    packResizeGrowthInitializationCount[index] =
        double(hvxResult.timings.packResizeGrowthInitializationCount);
    packMetadata[index] = hvxResult.timings.packMetadataSetupUs;
    packSquareWeight[index] = hvxResult.timings.packSquareWeightCopyUs;
    packW1Weight[index] = hvxResult.timings.packW1WeightCopyUs;
    packW2Transpose[index] = hvxResult.timings.packW2WeightTransposeUs;
    packGradient[index] = hvxResult.timings.packGradientCopyUs;
    packMomentum[index] = hvxResult.timings.packMomentumCopyUs;
    packFlatRpc[index] = hvxResult.timings.packFlatRpcCopyUs;
    packHyper[index] = hvxResult.timings.packHyperSetupUs;
    unpackDecode[index] = hvxResult.timings.unpackRpcOutputDecodeUs;
    unpackDecodedValidation[index] = hvxResult.timings.unpackDecodedValidationUs;
    unpackCandidate[index] = hvxResult.timings.unpackCandidateGenerationUs;
    unpackCandidateParameterCopy[index] =
        hvxResult.timings.unpackCandidateParameterCopyUs;
    unpackCandidateMomentumCopy[index] =
        hvxResult.timings.unpackCandidateMomentumCopyUs;
    unpackCandidateOther[index] = std::max(0.0,
        unpackCandidate[index] - unpackCandidateParameterCopy[index] -
            unpackCandidateMomentumCopy[index]);
    unpackRegistry[index] = hvxResult.timings.unpackRegistryTraversalUs;
    unpackSquare[index] = hvxResult.timings.unpackSquareOutputCopyUs;
    unpackW1[index] = hvxResult.timings.unpackW1OutputCopyUs;
    unpackW2[index] = hvxResult.timings.unpackW2TransposeBackUs;
    unpackAuxAdam[index] = hvxResult.timings.unpackAuxAdamUs;
    auxParameterCopy[index] = hvxResult.timings.auxParameterCopyUs;
    auxMuonMomentumCopy[index] = hvxResult.timings.auxMuonMomentumCopyUs;
    auxAdamMCopy[index] = hvxResult.timings.auxAdamMCopyUs;
    auxAdamVCopy[index] = hvxResult.timings.auxAdamVCopyUs;
    auxRegistryConstruction[index] =
        hvxResult.timings.auxRegistryConstructionUs;
    auxRegistryValidation[index] = hvxResult.timings.auxRegistryValidationUs;
    auxPreUpdateFiniteValidation[index] =
        hvxResult.timings.auxPreUpdateFiniteValidationUs;
    auxPreParameterFiniteValidation[index] =
        hvxResult.timings.auxPreParameterFiniteValidationUs;
    auxPreGradientFiniteValidation[index] =
        hvxResult.timings.auxPreGradientFiniteValidationUs;
    auxPreMomentumFiniteValidation[index] =
        hvxResult.timings.auxPreMomentumFiniteValidationUs;
    auxPreAdamMFiniteValidation[index] =
        hvxResult.timings.auxPreAdamMFiniteValidationUs;
    auxPreAdamVFiniteValidation[index] =
        hvxResult.timings.auxPreAdamVFiniteValidationUs;
    auxArithmetic[index] = hvxResult.timings.auxArithmeticUs;
    auxPostUpdateFiniteValidation[index] =
        hvxResult.timings.auxPostUpdateFiniteValidationUs;
    auxOther[index] = std::max(0.0,
        unpackAuxAdam[index] - auxParameterCopy[index] -
            auxMuonMomentumCopy[index] - auxAdamMCopy[index] -
            auxAdamVCopy[index] - auxRegistryConstruction[index] -
            auxRegistryValidation[index] -
            auxPreUpdateFiniteValidation[index] - auxArithmetic[index] -
            auxPostUpdateFiniteValidation[index]);
    unpackFinalCommit[index] = hvxResult.timings.unpackFinalCommitUs;
    mutexWait[index] = hvxResult.timings.mutexWaitUs;
    sessionSetup[index] = hvxResult.timings.sessionSetupUs;
  }
  std::array<double, kMeasured> profileKernel{}, profileTotalWork{},
      profileMomentum{}, profileNorm{}, profileFinalUpdate{}, profileGemm{},
      profileNonGemm{}, profileVector{}, profileTranspose{}, profileLongest{},
      profileShortest{}, profileImbalance{};
  std::array<std::array<double, kMeasured>, 5> profileNs{};
  std::array<std::array<double, kMeasured>, 8> profileWorkerUs{},
      profileWorkerPcycles{}, profileWorkerMatrices{};
#if PHONELM_ENABLE_HVX_MUON
  {
    Session& rpc = session();
    std::lock_guard<std::mutex> lock(rpc.mutex);
    for (std::size_t repetition = 0; repetition <= kMeasured; ++repetition) {
      std::fill_n(rpc.output_, kOutputFloats,
                  std::numeric_limits<float>::quiet_NaN());
      std::fill_n(rpc.metadata_, kMetadataInts, 0);
      const int status = hexatrain_hvx_probe_run(
          rpc.handle_, 7, 8, 0, 0, rpc.input_, static_cast<int>(kInputFloats),
          rpc.hyper_, static_cast<int>(kHyperFloats), rpc.output_,
          static_cast<int>(kOutputFloats), rpc.metadata_,
          static_cast<int>(kMetadataInts));
      if (status || rpc.metadata_[4] || rpc.metadata_[7] != 7 ||
          rpc.metadata_[10] != 8 || rpc.metadata_[11] != 8 ||
          rpc.metadata_[30] != 114 || !allFinite(rpc.output_, kOutputFloats)) {
        return "HVX_MUON_OPTIMIZER_BENCHMARK\nstatus=FAILED\nerror="
            "HVX_W8_PROFILE_FAILED:" + std::to_string(status) +
            ":kernel=" + std::to_string(rpc.metadata_[4]) +
            "\nfallback=false\n";
      }
      if (repetition == 0) continue;
      const std::size_t index = repetition - 1;
      profileKernel[index] = rpc.metadata_[3];
      profileTotalWork[index] = rpc.metadata_[16];
      profileMomentum[index] = rpc.metadata_[17];
      profileNorm[index] = rpc.metadata_[18];
      for (std::size_t stage = 0; stage < 5; ++stage)
        profileNs[stage][index] = rpc.metadata_[19 + stage];
      profileFinalUpdate[index] = rpc.metadata_[24];
      profileGemm[index] = rpc.metadata_[25];
      profileVector[index] = rpc.metadata_[26];
      profileTranspose[index] = rpc.metadata_[27];
      profileNonGemm[index] = rpc.metadata_[28];
      profileLongest[index] = rpc.metadata_[56];
      profileShortest[index] = rpc.metadata_[57];
      profileImbalance[index] = rpc.metadata_[58];
      for (std::size_t worker = 0; worker < 8; ++worker) {
        profileWorkerUs[worker][index] = rpc.metadata_[32 + worker];
        profileWorkerPcycles[worker][index] = rpc.metadata_[40 + worker];
        profileWorkerMatrices[worker][index] = rpc.metadata_[48 + worker];
      }
    }
  }
#endif
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
  const auto appendSummary = [&](const char* name,
                                 const std::array<double, kMeasured>& values) {
    report << name << "_best_us=" << best(values) << '\n'
           << name << "_median_us=" << median(values) << '\n'
           << name << "_mean_us=" << mean(values) << '\n';
  };
  const auto appendUnitlessSummary = [&](const char* name,
      const std::array<double, kMeasured>& values) {
    report << name << "_best=" << best(values) << '\n'
           << name << "_median=" << median(values) << '\n'
           << name << "_mean=" << mean(values) << '\n';
  };
  appendSummary("rpc_kernel_external", rpcKernelExternal);
  appendSummary("pack_registry_traversal", packRegistry);
  appendSummary("pack_allocation_resize", packAllocation);
  appendSummary("pack_actual_reallocation", packActualReallocation);
  appendSummary("pack_resize_growth_initialization",
                packResizeGrowthInitialization);
  appendSummary("pack_resize_other", packResizeOther);
  report << "pack_actual_reallocation_count_median="
         << median(packActualReallocationCount) << '\n'
         << "pack_resize_growth_initialization_count_median="
         << median(packResizeGrowthInitializationCount) << '\n';
  appendSummary("pack_metadata_setup", packMetadata);
  appendSummary("pack_square_weight_copy", packSquareWeight);
  appendSummary("pack_w1_weight_copy", packW1Weight);
  appendSummary("pack_w2_weight_transpose", packW2Transpose);
  appendSummary("pack_gradient_copy", packGradient);
  appendSummary("pack_momentum_copy", packMomentum);
  appendSummary("pack_flat_rpc_copy", packFlatRpc);
  appendSummary("pack_hyper_setup", packHyper);
  appendSummary("unpack_rpc_output_decode", unpackDecode);
  appendSummary("unpack_decoded_validation", unpackDecodedValidation);
  appendSummary("unpack_candidate_generation", unpackCandidate);
  appendSummary("unpack_candidate_parameter_copy",
                unpackCandidateParameterCopy);
  appendSummary("unpack_candidate_momentum_copy",
                unpackCandidateMomentumCopy);
  appendSummary("unpack_candidate_other", unpackCandidateOther);
  appendSummary("unpack_registry_traversal", unpackRegistry);
  appendSummary("unpack_square_output_copy", unpackSquare);
  appendSummary("unpack_w1_output_copy", unpackW1);
  appendSummary("unpack_w2_transpose_back", unpackW2);
  appendSummary("unpack_aux_adam", unpackAuxAdam);
  appendSummary("aux_parameter_deep_copy", auxParameterCopy);
  appendSummary("aux_muon_momentum_deep_copy", auxMuonMomentumCopy);
  appendSummary("aux_adam_m_deep_copy", auxAdamMCopy);
  appendSummary("aux_adam_v_deep_copy", auxAdamVCopy);
  appendSummary("aux_registry_construction", auxRegistryConstruction);
  appendSummary("aux_registry_validation", auxRegistryValidation);
  appendSummary("aux_pre_update_finite_validation",
                auxPreUpdateFiniteValidation);
  appendSummary("aux_pre_parameter_finite_validation",
                auxPreParameterFiniteValidation);
  appendSummary("aux_pre_gradient_finite_validation",
                auxPreGradientFiniteValidation);
  appendSummary("aux_pre_momentum_finite_validation",
                auxPreMomentumFiniteValidation);
  appendSummary("aux_pre_adam_m_finite_validation",
                auxPreAdamMFiniteValidation);
  appendSummary("aux_pre_adam_v_finite_validation",
                auxPreAdamVFiniteValidation);
  appendSummary("aux_actual_arithmetic", auxArithmetic);
  appendSummary("aux_post_update_finite_validation",
                auxPostUpdateFiniteValidation);
  appendSummary("aux_other", auxOther);
  appendSummary("unpack_final_commit", unpackFinalCommit);
  appendSummary("mutex_wait", mutexWait);
  appendSummary("session_setup", sessionSetup);
  appendSummary("dsp_profile_kernel", profileKernel);
  appendSummary("dsp_profile_total_worker_work", profileTotalWork);
  appendSummary("dsp_profile_momentum_nesterov", profileMomentum);
  appendSummary("dsp_profile_frobenius_normalization", profileNorm);
  for (std::size_t stage = 0; stage < 5; ++stage) {
    const std::string name = "dsp_profile_ns" + std::to_string(stage + 1);
    appendSummary(name.c_str(), profileNs[stage]);
  }
  appendSummary("dsp_profile_final_parameter_update", profileFinalUpdate);
  appendSummary("dsp_profile_qhl_gemm_total", profileGemm);
  appendSummary("dsp_profile_non_gemm", profileNonGemm);
  appendSummary("dsp_profile_vector_ops", profileVector);
  appendSummary("dsp_profile_transpose", profileTranspose);
  appendSummary("dsp_profile_longest_worker", profileLongest);
  appendSummary("dsp_profile_shortest_worker", profileShortest);
  appendSummary("dsp_profile_worker_imbalance", profileImbalance);
  for (std::size_t worker = 0; worker < 8; ++worker) {
    const std::string prefix = "dsp_profile_worker_" +
        std::to_string(worker);
    appendSummary((prefix + "_elapsed").c_str(), profileWorkerUs[worker]);
    appendUnitlessSummary((prefix + "_pcycles").c_str(),
                          profileWorkerPcycles[worker]);
    appendUnitlessSummary((prefix + "_matrices").c_str(),
                          profileWorkerMatrices[worker]);
  }
  for (std::size_t i = 0; i < kMeasured; ++i) {
    const auto run = [&](const char* name, double value) {
      report << "measured_run_" << (i + 1) << '_' << name << "_us="
             << value << '\n';
    };
    run("optimizer_total", hvx[i]);
    run("pack", pack[i]);
    run("input_validation", inputValidation[i]);
    run("rpc", rpc[i]);
    run("kernel", kernel[i]);
    run("rpc_kernel_external", rpcKernelExternal[i]);
    run("output_validation", outputValidation[i]);
    run("pack_registry_traversal", packRegistry[i]);
    run("pack_allocation_resize", packAllocation[i]);
    run("pack_actual_reallocation", packActualReallocation[i]);
    run("pack_resize_growth_initialization",
        packResizeGrowthInitialization[i]);
    run("pack_resize_other", packResizeOther[i]);
    report << "measured_run_" << (i + 1)
           << "_pack_actual_reallocation_count="
           << packActualReallocationCount[i] << '\n'
           << "measured_run_" << (i + 1)
           << "_pack_resize_growth_initialization_count="
           << packResizeGrowthInitializationCount[i] << '\n';
    run("pack_metadata_setup", packMetadata[i]);
    run("pack_square_weight_copy", packSquareWeight[i]);
    run("pack_w1_weight_copy", packW1Weight[i]);
    run("pack_w2_weight_transpose", packW2Transpose[i]);
    run("pack_gradient_copy", packGradient[i]);
    run("pack_momentum_copy", packMomentum[i]);
    run("pack_flat_rpc_copy", packFlatRpc[i]);
    run("pack_hyper_setup", packHyper[i]);
    run("unpack_apply", unpackApply[i]);
    run("unpack_rpc_output_decode", unpackDecode[i]);
    run("unpack_decoded_validation", unpackDecodedValidation[i]);
    run("unpack_candidate_generation", unpackCandidate[i]);
    run("unpack_candidate_parameter_copy", unpackCandidateParameterCopy[i]);
    run("unpack_candidate_momentum_copy", unpackCandidateMomentumCopy[i]);
    run("unpack_candidate_other", unpackCandidateOther[i]);
    run("unpack_registry_traversal", unpackRegistry[i]);
    run("unpack_square_output_copy", unpackSquare[i]);
    run("unpack_w1_output_copy", unpackW1[i]);
    run("unpack_w2_transpose_back", unpackW2[i]);
    run("unpack_aux_adam", unpackAuxAdam[i]);
    run("aux_parameter_deep_copy", auxParameterCopy[i]);
    run("aux_muon_momentum_deep_copy", auxMuonMomentumCopy[i]);
    run("aux_adam_m_deep_copy", auxAdamMCopy[i]);
    run("aux_adam_v_deep_copy", auxAdamVCopy[i]);
    run("aux_registry_construction", auxRegistryConstruction[i]);
    run("aux_registry_validation", auxRegistryValidation[i]);
    run("aux_pre_update_finite_validation",
        auxPreUpdateFiniteValidation[i]);
    run("aux_pre_parameter_finite_validation",
        auxPreParameterFiniteValidation[i]);
    run("aux_pre_gradient_finite_validation",
        auxPreGradientFiniteValidation[i]);
    run("aux_pre_momentum_finite_validation",
        auxPreMomentumFiniteValidation[i]);
    run("aux_pre_adam_m_finite_validation",
        auxPreAdamMFiniteValidation[i]);
    run("aux_pre_adam_v_finite_validation",
        auxPreAdamVFiniteValidation[i]);
    run("aux_actual_arithmetic", auxArithmetic[i]);
    run("aux_post_update_finite_validation",
        auxPostUpdateFiniteValidation[i]);
    run("aux_other", auxOther[i]);
    run("unpack_final_commit", unpackFinalCommit[i]);
    run("mutex_wait", mutexWait[i]);
    run("session_setup", sessionSetup[i]);
    run("dsp_profile_kernel", profileKernel[i]);
    run("dsp_profile_total_worker_work", profileTotalWork[i]);
    run("dsp_profile_momentum_nesterov", profileMomentum[i]);
    run("dsp_profile_frobenius_normalization", profileNorm[i]);
    for (std::size_t stage = 0; stage < 5; ++stage) {
      const std::string name = "dsp_profile_ns" + std::to_string(stage + 1);
      run(name.c_str(), profileNs[stage][i]);
    }
    run("dsp_profile_final_parameter_update", profileFinalUpdate[i]);
    run("dsp_profile_qhl_gemm_total", profileGemm[i]);
    run("dsp_profile_non_gemm", profileNonGemm[i]);
    run("dsp_profile_vector_ops", profileVector[i]);
    run("dsp_profile_transpose", profileTranspose[i]);
    run("dsp_profile_longest_worker", profileLongest[i]);
    run("dsp_profile_shortest_worker", profileShortest[i]);
    run("dsp_profile_worker_imbalance", profileImbalance[i]);
    for (std::size_t worker = 0; worker < 8; ++worker) {
      const std::string prefix = "dsp_profile_worker_" +
          std::to_string(worker);
      run((prefix + "_elapsed").c_str(), profileWorkerUs[worker][i]);
      report << "measured_run_" << (i + 1) << '_' << prefix
             << "_pcycles=" << profileWorkerPcycles[worker][i] << '\n'
             << "measured_run_" << (i + 1) << '_' << prefix
             << "_matrices=" << profileWorkerMatrices[worker][i] << '\n';
    }
  }
  return report.str();
}

}  // namespace phonelm::nicopedia_hvx_muon
