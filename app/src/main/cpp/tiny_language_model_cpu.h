// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once
#include "qnn/qnn_runtime.h"
#include "transformer_resource_estimator.h"
#include <cstdint>
#include <string>
#include <vector>
namespace phonelm::tiny_lm {
struct Config {
  uint32_t vocabularySize=32,tokens=8,dimension=16,feedForwardDimension=32;
  float epsilon=1e-5f;
  uint32_t numLayers=1,numHeads=1;
};
struct ParameterInfo { std::string name; const std::vector<float>* values=nullptr; };
// Checks all derived element and byte counts before graph or CPU work starts.
bool validateConfig(const Config&, std::string* error=nullptr);
transformer::ResourceEstimate resourceEstimate(const Config&);
uint32_t headDim(const Config&);
std::vector<ParameterInfo> parameterRegistry(const qnn::TinyTransformerParameters&);
size_t parameterElementCount(const qnn::TinyTransformerParameters&);
bool storageRangesHaveNoAliases(const std::vector<ParameterInfo>&);
bool parameterStorageHasNoAliases(const qnn::TinyTransformerParameters&);
struct StepResult {
  float loss=0,accuracy=0;
  std::vector<float> embeddedInput,transformerOutput,logits,probabilities,dLogits,dEmbeddedInput;
  // Diagnostic ordering is layer-major; attention probabilities are
  // [layer * numHeads + head], each with [T,T] elements.
  std::vector<std::vector<float>> layerInputGradients;
  std::vector<std::vector<float>> attentionHeadProbabilities;
  qnn::TinyTransformerParameters gradients,next;
};
struct GradientCheckResult { bool passed=false; float maximumAbsoluteError=0,maximumRelativeError=0; std::string report; };
struct MomentumResult { qnn::TinyTransformerParameters velocity,next; };
struct AdamResult {
  qnn::TinyTransformerParameters firstMoment, secondMoment;
  qnn::TinyTransformerParameters firstMomentHat, secondMomentHat, next;
};
qnn::TinyTransformerParameters initialParameters(const Config&,uint32_t);
std::vector<float> oneHot(const std::vector<uint32_t>&,uint32_t);
std::vector<float> fixedPosition(const Config&);
StepResult forwardBackward(const Config&,const std::vector<float>&,const std::vector<float>&,const qnn::TinyTransformerParameters&,float);
// Test/diagnostic entry point: always uses the generalized layer/head path.
StepResult forwardBackwardGeneralized(const Config&,const std::vector<float>&,const std::vector<float>&,const qnn::TinyTransformerParameters&,float);
struct GeneralizedCpuTrace {
  std::vector<float> embeddedInput;
  std::vector<float> logits;
  std::vector<float> probabilities;
  struct Layer {
    std::vector<float> input;
    std::vector<float> ln1;
    std::vector<float> ln1Centered, ln1Square, ln1VarianceEps, ln1Inv;
    std::vector<float> q, k, v;
    std::vector<float> probabilities;
    std::vector<float> context;
    std::vector<float> residual1;
    std::vector<float> ln2;
    std::vector<float> ln2Centered, ln2Square, ln2VarianceEps, ln2Inv;
    std::vector<float> ff1;
    std::vector<float> relu;
    std::vector<float> output;
  };
  std::vector<Layer> layers;
};
GeneralizedCpuTrace forwardTraceGeneralized(const Config& config, const std::vector<float>& oneHotInput, const qnn::TinyTransformerParameters& parameters);
MomentumResult momentumUpdate(const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,float,float);
AdamResult adamUpdate(const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,
                      const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,
                      float,float,float,float,float,float);
GradientCheckResult gradientCheck(uint32_t seed=20260725,float epsilon=1e-3f);
}
