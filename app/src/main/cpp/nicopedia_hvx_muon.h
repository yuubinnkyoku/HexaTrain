// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "nicopedia_muon_optimizer.h"

#include <cstddef>
#include <string>

namespace phonelm::nicopedia_hvx_muon {

struct Timings {
  double mutexWaitUs = 0.0;
  double sessionSetupUs = 0.0;
  double packUs = 0.0;
  double packRegistryTraversalUs = 0.0;
  double packAllocationResizeUs = 0.0;
  double packActualReallocationUs = 0.0;
  double packResizeGrowthInitializationUs = 0.0;
  double packResizeOtherUs = 0.0;
  std::size_t packActualReallocationCount = 0;
  std::size_t packResizeGrowthInitializationCount = 0;
  double packMetadataSetupUs = 0.0;
  double packSquareWeightCopyUs = 0.0;
  double packW1WeightCopyUs = 0.0;
  double packW2WeightTransposeUs = 0.0;
  double packGradientCopyUs = 0.0;
  double packMomentumCopyUs = 0.0;
  double packFlatRpcCopyUs = 0.0;
  double packHyperSetupUs = 0.0;
  double inputValidationUs = 0.0;
  double rpcUs = 0.0;
  double kernelUs = 0.0;
  double outputValidationUs = 0.0;
  double unpackRpcOutputDecodeUs = 0.0;
  double unpackDecodedValidationUs = 0.0;
  double unpackCandidateGenerationUs = 0.0;
  double unpackRegistryTraversalUs = 0.0;
  double unpackSquareOutputCopyUs = 0.0;
  double unpackW1OutputCopyUs = 0.0;
  double unpackW2TransposeBackUs = 0.0;
  double unpackAuxAdamUs = 0.0;
  double unpackFinalCommitUs = 0.0;
  double unpackApplyUs = 0.0;
  double totalUs = 0.0;
};

struct Result {
  nicopedia_muon::Result update;
  Timings timings;
  int rpcStatus = 0;
  bool outputFinite = false;
  bool fallback = false;
};

Result update(const qnn::TinyTransformerParameters& parameters,
              const qnn::TinyTransformerParameters& gradients,
              const qnn::TinyTransformerParameters& muonMomentum,
              const qnn::TinyTransformerParameters& auxiliaryAdamM,
              const qnn::TinyTransformerParameters& auxiliaryAdamV,
              const nicopedia_muon::Config& config);

std::string benchmarkActualOptimizerPath();

}  // namespace phonelm::nicopedia_hvx_muon
