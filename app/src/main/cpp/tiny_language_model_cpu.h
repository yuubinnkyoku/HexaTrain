// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once
#include "qnn/qnn_runtime.h"
#include <cstdint>
#include <string>
#include <vector>
namespace phonelm::tiny_lm {
struct Config { uint32_t vocabularySize=32,tokens=8,dimension=16,feedForwardDimension=32; float epsilon=1e-5f; };
struct StepResult { float loss=0,accuracy=0; std::vector<float> embeddedInput,transformerOutput,logits,probabilities,dLogits,dEmbeddedInput; qnn::TinyTransformerParameters gradients,next; };
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
MomentumResult momentumUpdate(const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,float,float);
AdamResult adamUpdate(const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,
                      const qnn::TinyTransformerParameters&,const qnn::TinyTransformerParameters&,
                      float,float,float,float,float,float);
GradientCheckResult gradientCheck(uint32_t seed=20260725,float epsilon=1e-3f);
}
