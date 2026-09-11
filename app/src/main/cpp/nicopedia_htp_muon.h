// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "tiny_language_model_cpu.h"

#include <cstddef>
#include <string>
#include <vector>

namespace phonelm::nicopedia_htp_muon {

inline constexpr std::size_t kSquareBatch = 76;
inline constexpr std::size_t kRectangularBatch = 38;
inline constexpr std::size_t kRows = 64;
inline constexpr std::size_t kSquareColumns = 64;
inline constexpr std::size_t kRectangularColumns = 128;

struct MatrixBinding {
  std::size_t registryIndex = 0;
  std::string name;
  std::uint32_t originalRows = 0;
  std::uint32_t originalColumns = 0;
  bool transposed = false;
  float updateScale = 1.0f;
};

struct PackedInputs {
  std::vector<float> currentSquare;
  std::vector<float> gradientSquare;
  std::vector<float> momentumSquare;
  std::vector<float> scaleSquare;
  std::vector<float> currentRectangular;
  std::vector<float> gradientRectangular;
  std::vector<float> momentumRectangular;
  std::vector<float> scaleRectangular;
  std::vector<MatrixBinding> squareBindings;
  std::vector<MatrixBinding> rectangularBindings;
};

struct PackTimings {
  double registryTraversalUs = 0.0;
  double allocationResizeUs = 0.0;
  double actualReallocationUs = 0.0;
  double resizeGrowthInitializationUs = 0.0;
  double resizeOtherUs = 0.0;
  std::size_t actualReallocationCount = 0;
  std::size_t resizeGrowthInitializationCount = 0;
  double metadataSetupUs = 0.0;
  double squareWeightCopyUs = 0.0;
  double w1WeightCopyUs = 0.0;
  double w2WeightTransposeUs = 0.0;
  double gradientCopyUs = 0.0;
  double momentumCopyUs = 0.0;
};

struct UnpackTimings {
  double decodedValidationUs = 0.0;
  double registryTraversalUs = 0.0;
  double squareOutputCopyUs = 0.0;
  double w1OutputCopyUs = 0.0;
  double w2TransposeBackUs = 0.0;
};

bool pack(const qnn::TinyTransformerParameters& parameters,
          const qnn::TinyTransformerParameters& gradients,
          const qnn::TinyTransformerParameters& momentum,
          PackedInputs* packed, std::string* error = nullptr);

// Structural packing used by the HVX optimizer path. Finiteness is checked in
// a separate, timed pass immediately before RPC; callers must not dispatch a
// result of this function without validateFinite().
bool packForValidatedRpc(const qnn::TinyTransformerParameters& parameters,
                         const qnn::TinyTransformerParameters& gradients,
                         const qnn::TinyTransformerParameters& momentum,
                          PackedInputs* packed,
                          std::string* error = nullptr,
                          PackTimings* timings = nullptr);

bool validateFinite(const PackedInputs& packed,
                    std::string* error = nullptr);

bool unpack(const PackedInputs& packed,
            const std::vector<float>& nextSquareWeights,
            const std::vector<float>& nextSquareMomentum,
            const std::vector<float>& nextRectangularWeights,
            const std::vector<float>& nextRectangularMomentum,
            qnn::TinyTransformerParameters* parameters,
            qnn::TinyTransformerParameters* momentum,
            std::string* error = nullptr,
            UnpackTimings* timings = nullptr);

}  // namespace phonelm::nicopedia_htp_muon
