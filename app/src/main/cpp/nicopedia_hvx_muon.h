// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "nicopedia_muon_optimizer.h"

#include <string>

namespace phonelm::nicopedia_hvx_muon {

struct Timings {
  double packUs = 0.0;
  double inputValidationUs = 0.0;
  double rpcUs = 0.0;
  double kernelUs = 0.0;
  double outputValidationUs = 0.0;
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
