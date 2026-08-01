// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

namespace phonelm::transformer {

// QNN's elementwise Adam graph is executed in bounded pieces. Keeping this
// limit here makes the resource estimate and execution contract identical.
// It is an application safety limit, not a QNN/HTP hardware limit.
inline constexpr std::uint64_t kMaximumAdamChunkElements = 32768;
inline constexpr std::uint64_t kApplicationPolicyBytes =
    std::uint64_t{1536} * 1024 * 1024;

// Conservative accounting for application-visible Transformer training
// storage. These values do not include DSP/runtime-internal allocations and
// must not be presented as HTP memory usage.
struct ResourceEstimate {
  bool ok = false;
  std::string failureClassification;
  std::string detail;

  std::uint64_t parameterElements = 0;
  std::uint64_t parameterBytes = 0;
  std::uint64_t gradientElements = 0;
  std::uint64_t gradientBytes = 0;
  std::uint64_t adamMomentElements = 0;  // first and second moments together
  std::uint64_t adamMomentBytes = 0;
  std::uint64_t forwardActivationElements = 0;
  std::uint64_t forwardActivationBytes = 0;
  std::uint64_t backwardActivationElements = 0;
  std::uint64_t backwardActivationBytes = 0;
  std::uint64_t attentionElements = 0;  // scores and probabilities
  std::uint64_t attentionBytes = 0;
  std::uint64_t adamGraphElements = 0;
  std::uint64_t adamChunkCount = 0;
  std::uint64_t adamApplicationVisibleBytes = 0;
  std::uint64_t persistentApplicationTensorBytes = 0;
  std::uint64_t estimatedPeakApplicationTensorBytes = 0;
  std::uint64_t bestCheckpointParameterBytes = 0;
  std::uint64_t bestCheckpointAdamBytes = 0;
  std::uint64_t checkpointSelectionOverheadBytes = 0;
  std::uint64_t cpuReferenceCheckpointBytes = 0;
  std::uint64_t estimatedPeakWithBestCheckpointBytes = 0;
  bool bestCheckpointFitsApplicationPolicy = false;
  std::uint64_t nodeCount = 0;
  std::uint64_t tensorCount = 0;
};

inline ResourceEstimate estimateTrainingResources(
    std::uint64_t sequenceLength, std::uint64_t vocabularySize,
    std::uint64_t embeddingDimension, std::uint64_t feedForwardDimension,
    std::uint64_t numLayers, std::uint64_t numHeads) {
  ResourceEstimate result;
  const auto fail = [&](const char* classification, const char* detail) {
    result.failureClassification = classification;
    result.detail = detail;
    return result;
  };
  if (!sequenceLength || !vocabularySize || !embeddingDimension ||
      !feedForwardDimension || !numLayers || !numHeads) {
    return fail("APP_CONFIGURATION_VALIDATION",
                "all Transformer dimensions and counts must be >= 1");
  }
  if (embeddingDimension % numHeads != 0) {
    return fail("APP_CONFIGURATION_VALIDATION",
                "embeddingDimension must be divisible by numHeads");
  }

  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t kFloatBytes = sizeof(float);
  // This is a PhoneLM host-allocation safety policy, not a QNN/runtime limit.
  const auto add = [&](std::uint64_t a, std::uint64_t b,
                       std::uint64_t* out) {
    if (b > kMax - a) return false;
    *out = a + b;
    return true;
  };
  const auto multiply = [&](std::uint64_t a, std::uint64_t b,
                            std::uint64_t* out) {
    if (a && b > kMax / a) return false;
    *out = a * b;
    return true;
  };
  const auto sum = [&](std::initializer_list<std::uint64_t> values,
                       std::uint64_t* out) {
    std::uint64_t total = 0;
    for (const auto value : values)
      if (!add(total, value, &total)) return false;
    *out = total;
    return true;
  };
  const auto bytes = [&](std::uint64_t elements, std::uint64_t* out) {
    return multiply(elements, kFloatBytes, out);
  };

  const std::uint64_t t = sequenceLength;
  const std::uint64_t v = vocabularySize;
  const std::uint64_t d = embeddingDimension;
  const std::uint64_t f = feedForwardDimension;
  const std::uint64_t l = numLayers;
  const std::uint64_t h = numHeads;
  const std::uint64_t dh = d / h;
  std::uint64_t td = 0, tf = 0, tv = 0, tt = 0, dd = 0, df = 0;
  if (!multiply(t, d, &td) || !multiply(t, f, &tf) ||
      !multiply(t, v, &tv) || !multiply(t, t, &tt) ||
      !multiply(d, d, &dd) || !multiply(d, f, &df)) {
    return fail("APP_RESOURCE_ESTIMATOR", "derived tensor element overflow");
  }

  std::uint64_t layerParameterElements = 0, globalParameterElements = 0;
  std::uint64_t fourDd = 0, fourD = 0, twoDf = 0, vd = 0;
  if (!multiply(4, dd, &fourDd) || !multiply(4, d, &fourD) ||
      !multiply(2, df, &twoDf) ||
      !sum({fourDd, fourD, twoDf}, &layerParameterElements) ||
      !multiply(v, d, &vd) || !multiply(2, vd, &globalParameterElements)) {
    return fail("APP_RESOURCE_ESTIMATOR", "parameter element overflow");
  }
  std::uint64_t allLayerParameters = 0;
  if (!multiply(l, layerParameterElements, &allLayerParameters) ||
      !add(globalParameterElements, allLayerParameters,
           &result.parameterElements) ||
      !bytes(result.parameterElements, &result.parameterBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR", "parameter byte overflow");
  }
  result.gradientElements = result.parameterElements;
  result.gradientBytes = result.parameterBytes;
  if (!multiply(2, result.parameterElements, &result.adamMomentElements) ||
      !bytes(result.adamMomentElements, &result.adamMomentBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR", "Adam m/v byte overflow");
  }
  result.adamGraphElements =
      result.parameterElements < kMaximumAdamChunkElements
          ? result.parameterElements
          : kMaximumAdamChunkElements;
  if (!result.adamGraphElements ||
      result.parameterElements >
          kMax - (result.adamGraphElements - 1)) {
    return fail("APP_RESOURCE_ESTIMATOR", "Adam chunk count overflow");
  }
  result.adamChunkCount =
      (result.parameterElements + result.adamGraphElements - 1) /
      result.adamGraphElements;
  // Ten APP_READ stage outputs plus one zero-padded input staging vector.
  std::uint64_t adamVisibleElements = 0;
  if (!multiply(11, result.adamGraphElements, &adamVisibleElements) ||
      !bytes(adamVisibleElements, &result.adamApplicationVisibleBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "Adam application-visible buffer byte overflow");
  }

  // Forward accounting follows the actual QNN builder, including the
  // full-width Q/K/V tensors and, for H>1, selector/scatter tensors.
  std::uint64_t baseForward = 0, perLayerForward = 0;
  std::uint64_t twentyTd = 0, tenT = 0, fourTt = 0, twoTf = 0;
  if (!multiply(20, td, &twentyTd) || !multiply(10, t, &tenT) ||
      !multiply(4, tt, &fourTt) || !multiply(2, tf, &twoTf) ||
      !sum({twentyTd, tenT, fourTt, twoTf}, &baseForward)) {
    return fail("APP_RESOURCE_ESTIMATOR", "forward activation overflow");
  }
  perLayerForward = baseForward;
  if (h > 1) {
    std::uint64_t tdh = 0, selector = 0, headForward = 0;
    std::uint64_t allHeads = 0, contextAccumulators = 0;
    std::uint64_t fourTdh = 0;
    if (!multiply(t, dh, &tdh) || !multiply(d, dh, &selector) ||
        !multiply(4, tdh, &fourTdh) ||
        !sum({selector, fourTdh, fourTt, td}, &headForward) ||
        !multiply(h, headForward, &allHeads) ||
        !multiply(h - 1, td, &contextAccumulators) ||
        !sum({baseForward, allHeads, contextAccumulators},
             &perLayerForward)) {
      return fail("APP_RESOURCE_ESTIMATOR",
                  "multi-head forward activation overflow");
    }
  }
  if (!multiply(l, perLayerForward, &result.forwardActivationElements) ||
      !bytes(result.forwardActivationElements,
             &result.forwardActivationBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR", "forward activation byte overflow");
  }

  std::uint64_t perLayerBackward = 0;
  std::uint64_t thirtySixTd = 0, sevenT = 0, threeTf = 0;
  if (!multiply(36, td, &thirtySixTd) || !multiply(7, t, &sevenT) ||
      !multiply(3, tf, &threeTf) ||
      !sum({thirtySixTd, sevenT, fourTt, threeTf},
           &perLayerBackward)) {
    return fail("APP_RESOURCE_ESTIMATOR", "backward activation overflow");
  }
  if (h > 1) {
    std::uint64_t tdh = 0, headBackward = 0, allHeads = 0;
    std::uint64_t gradientAccumulators = 0;
    std::uint64_t sixTdh = 0, threeTd = 0, accumulatorCount = 0;
    if (!multiply(t, dh, &tdh) ||
        !multiply(6, tdh, &sixTdh) || !multiply(3, td, &threeTd) ||
        !sum({sixTdh, fourTt, t, threeTd}, &headBackward) ||
        !multiply(h, headBackward, &allHeads) ||
        !multiply(3, h - 1, &accumulatorCount) ||
        !multiply(accumulatorCount, td, &gradientAccumulators) ||
        !sum({perLayerBackward, allHeads, gradientAccumulators},
             &perLayerBackward)) {
      return fail("APP_RESOURCE_ESTIMATOR",
                  "multi-head backward activation overflow");
    }
  }
  if (!multiply(l, perLayerBackward, &result.backwardActivationElements) ||
      !bytes(result.backwardActivationElements,
             &result.backwardActivationBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR", "backward activation byte overflow");
  }

  if (!multiply(2, tt, &result.attentionElements) ||
      !multiply(result.attentionElements, h, &result.attentionElements) ||
      !multiply(result.attentionElements, l, &result.attentionElements) ||
      !bytes(result.attentionElements, &result.attentionBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR", "attention tensor byte overflow");
  }

  // Inputs + parameters + all APP_READ buffers emitted by the training graph.
  std::uint64_t inputBytes = 0, outputElements = 0, outputBytes = 0;
  std::uint64_t layerInputElements = 0, headProbabilityElements = 0;
  std::uint64_t fourTv = 0, twoTd = 0;
  if (!multiply(2, tv, &inputBytes) || !bytes(inputBytes, &inputBytes) ||
      !multiply(l, td, &layerInputElements) ||
      !multiply(l, h, &headProbabilityElements) ||
      !multiply(headProbabilityElements, tt, &headProbabilityElements) ||
      !multiply(4, tv, &fourTv) || !multiply(2, td, &twoTd) ||
      !sum({fourTv, twoTd, result.gradientElements, layerInputElements,
            h > 1 ? headProbabilityElements : 0},
           &outputElements) ||
      !bytes(outputElements, &outputBytes) ||
      !sum({inputBytes, result.parameterBytes, outputBytes},
           &result.persistentApplicationTensorBytes) ||
      !sum({result.persistentApplicationTensorBytes,
            result.adamMomentBytes, result.forwardActivationBytes,
            result.backwardActivationBytes,
            result.adamApplicationVisibleBytes},
           &result.estimatedPeakApplicationTensorBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "application-visible tensor byte overflow");
  }

  // Exact tensor count for the current generalized training builder plus the
  // separately prepared flattened Adam graph.
  std::uint64_t headSlots = 0, layerSlots = 100;
  if (h > 1) {
    std::uint64_t accumulatorSlots = 0;
    if (!multiply(24, h, &headSlots) ||
        !multiply(4, h - 1, &accumulatorSlots) ||
        !add(headSlots, accumulatorSlots, &headSlots) ||
        !add(layerSlots, headSlots, &layerSlots)) {
      return fail("APP_RESOURCE_ESTIMATOR", "tensor count overflow");
    }
  }
  if (!multiply(l, layerSlots, &result.tensorCount) ||
      !add(24, result.tensorCount, &result.tensorCount) ||
      !add(35, result.tensorCount, &result.tensorCount)) {
    return fail("APP_RESOURCE_ESTIMATOR", "tensor count overflow");
  }

  std::uint64_t forwardNodes = 36, backwardNodes = 54;
  std::uint64_t perLayerNodes = 0;
  if (h > 1 &&
      (!multiply(10, h, &forwardNodes) ||
       !add(30, forwardNodes, &forwardNodes) ||
       !multiply(17, h, &backwardNodes) ||
       !add(41, backwardNodes, &backwardNodes))) {
    return fail("APP_RESOURCE_ESTIMATOR", "node count overflow");
  }
  if (!add(forwardNodes, backwardNodes, &perLayerNodes) ||
      !multiply(l, perLayerNodes, &result.nodeCount) ||
      !add(9 + 20, result.nodeCount, &result.nodeCount)) {
    return fail("APP_RESOURCE_ESTIMATOR", "node count overflow");
  }
  if (result.tensorCount > std::numeric_limits<std::uint32_t>::max() ||
      result.nodeCount > std::numeric_limits<std::uint32_t>::max()) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "node or tensor count exceeds QNN uint32 registry range");
  }
  if (result.estimatedPeakApplicationTensorBytes >
      kApplicationPolicyBytes) {
    return fail("APP_POLICY_LIMIT",
                "estimated application-visible allocation exceeds 1536 MiB");
  }
  result.bestCheckpointParameterBytes = result.parameterBytes;
  result.bestCheckpointAdamBytes = result.adamMomentBytes;
  if (!add(result.bestCheckpointParameterBytes,
           result.bestCheckpointAdamBytes,
           &result.checkpointSelectionOverheadBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "best-validation checkpoint byte overflow");
  }
  result.cpuReferenceCheckpointBytes = result.checkpointSelectionOverheadBytes;
  if (!add(result.checkpointSelectionOverheadBytes,
           result.cpuReferenceCheckpointBytes,
           &result.checkpointSelectionOverheadBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "CPU reference checkpoint byte overflow");
  }
  if (!add(result.estimatedPeakApplicationTensorBytes,
           result.checkpointSelectionOverheadBytes,
           &result.estimatedPeakWithBestCheckpointBytes)) {
    return fail("APP_RESOURCE_ESTIMATOR",
                "selection peak byte overflow");
  }
  result.bestCheckpointFitsApplicationPolicy =
      result.estimatedPeakWithBestCheckpointBytes <= kApplicationPolicyBytes;
  result.ok = true;
  result.failureClassification = "NONE";
  result.detail = "application-visible estimate; excludes QNN/DSP internal memory";
  return result;
}

}  // namespace phonelm::transformer
