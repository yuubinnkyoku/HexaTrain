// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "nicopedia_htp_muon.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace phonelm::nicopedia_htp_muon { namespace {

using Clock = std::chrono::steady_clock;
double elapsedUs(Clock::time_point started) {
  return std::chrono::duration<double, std::micro>(Clock::now() - started).count();
}

bool fail(std::string* error, const std::string& message) {
  if (error) *error = "HTP_MUON_" + message;
  return false;
}

bool finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

void ensurePlaneSize(std::vector<float>* destination, std::size_t size,
                     PackTimings* timings) {
  if (destination->size() == size) return;
  const std::size_t oldSize = destination->size();
  const std::size_t oldCapacity = destination->capacity();
  auto phase = Clock::now();
  destination->resize(size);
  const double resizeUs = elapsedUs(phase);
  if (timings) {
    timings->allocationResizeUs += resizeUs;
    if (size > oldCapacity) {
      timings->actualReallocationUs += resizeUs;
      ++timings->actualReallocationCount;
    } else if (size > oldSize) {
      timings->resizeGrowthInitializationUs += resizeUs;
      ++timings->resizeGrowthInitializationCount;
    } else {
      timings->resizeOtherUs += resizeUs;
    }
  }
}

bool writeMatrixAtOffset(const std::vector<float>& source, std::uint32_t rows,
                         std::uint32_t columns, bool transpose,
                         std::size_t offset,
                         std::vector<float>* destination, double* copyUs) {
  if (offset > destination->size() ||
      destination->size() - offset < source.size())
    return false;
  const auto phase = Clock::now();
  if (!transpose) {
    std::copy(source.begin(), source.end(), destination->begin() + offset);
    if (copyUs) *copyUs += elapsedUs(phase);
    return true;
  }
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t column = 0; column < columns; ++column)
      (*destination)[offset + std::size_t(column) * rows + row] =
          source[std::size_t(row) * columns + column];
  if (copyUs) *copyUs += elapsedUs(phase);
  return true;
}

bool appendBinding(const tiny_lm::ParameterInfo& parameter,
                   const tiny_lm::ParameterInfo& gradient,
                   const tiny_lm::ParameterInfo& momentum,
                   std::size_t registryIndex, bool transpose,
                   std::vector<float>* current, std::vector<float>* gradients,
                    std::vector<float>* momenta, std::vector<float>* scales,
                    std::vector<MatrixBinding>* bindings, bool checkFinite,
                    bool square, PackTimings* timings, std::string* error) {
  if (!parameter.values || !gradient.values || !momentum.values ||
      parameter.values->size() != gradient.values->size() ||
      parameter.values->size() != momentum.values->size())
    return fail(error, "PACK_ELEMENT_COUNT_MISMATCH:" + parameter.name);
  if (checkFinite && (!finite(*parameter.values) || !finite(*gradient.values) ||
      !finite(*momentum.values)))
    return fail(error, "PACK_NONFINITE:" + parameter.name);
  const float scale = std::sqrt(std::max(
      1.0f, float(parameter.fanOut) / float(parameter.fanIn)));
  double* weightCopyUs = nullptr;
  if (timings) {
    weightCopyUs = square ? &timings->squareWeightCopyUs
                          : (transpose ? &timings->w2WeightTransposeUs
                                       : &timings->w1WeightCopyUs);
  }
  const std::size_t offset = bindings->size() * parameter.values->size();
  if (!writeMatrixAtOffset(*parameter.values, parameter.shape[0],
                           parameter.shape[1], transpose, offset, current,
                           weightCopyUs) ||
      !writeMatrixAtOffset(*gradient.values, parameter.shape[0],
                           parameter.shape[1], transpose, offset, gradients,
                           timings ? &timings->gradientCopyUs : nullptr) ||
      !writeMatrixAtOffset(*momentum.values, parameter.shape[0],
                           parameter.shape[1], transpose, offset, momenta,
                           timings ? &timings->momentumCopyUs : nullptr))
    return fail(error, "PACK_DESTINATION_BOUNDS:" + parameter.name);
  const auto metadataStarted = Clock::now();
  scales->push_back(scale);
  bindings->push_back({registryIndex, parameter.name, parameter.shape[0],
                       parameter.shape[1], transpose, scale});
  if (timings) timings->metadataSetupUs += elapsedUs(metadataStarted);
  return true;
}

bool writeMatrix(const std::vector<float>& source, std::size_t matrixIndex,
                 const MatrixBinding& binding, std::vector<float>* destination,
                 std::string* error) {
  if (!destination) return fail(error, "UNPACK_NULL_DESTINATION");
  const std::size_t elements = std::size_t(binding.originalRows) *
                               binding.originalColumns;
  const std::size_t offset = matrixIndex * elements;
  if (offset > source.size() || source.size() - offset < elements ||
      destination->size() != elements)
    return fail(error, "UNPACK_ELEMENT_COUNT_MISMATCH:" + binding.name);
  if (!binding.transposed) {
    std::copy_n(source.begin() + offset, elements, destination->begin());
    return true;
  }
  const std::uint32_t canonicalRows = binding.originalColumns;
  const std::uint32_t canonicalColumns = binding.originalRows;
  for (std::uint32_t row = 0; row < canonicalRows; ++row)
    for (std::uint32_t column = 0; column < canonicalColumns; ++column)
      (*destination)[std::size_t(column) * canonicalRows + row] =
          source[offset + std::size_t(row) * canonicalColumns + column];
  return true;
}

}  // namespace

bool packImpl(const qnn::TinyTransformerParameters& parameters,
              const qnn::TinyTransformerParameters& gradients,
              const qnn::TinyTransformerParameters& momentum,
               bool checkFinite, PackedInputs* packed, std::string* error,
               PackTimings* timings = nullptr) {
  if (!packed) return fail(error, "PACK_NULL_OUTPUT");
  if (timings) *timings = {};
  const auto totalStarted = Clock::now();
  constexpr std::size_t kSquarePlaneElements =
      kSquareBatch * kRows * kSquareColumns;
  constexpr std::size_t kRectangularPlaneElements =
      kRectangularBatch * kRows * kRectangularColumns;
  ensurePlaneSize(&packed->currentSquare, kSquarePlaneElements, timings);
  ensurePlaneSize(&packed->gradientSquare, kSquarePlaneElements, timings);
  ensurePlaneSize(&packed->momentumSquare, kSquarePlaneElements, timings);
  ensurePlaneSize(&packed->currentRectangular, kRectangularPlaneElements,
                  timings);
  ensurePlaneSize(&packed->gradientRectangular, kRectangularPlaneElements,
                  timings);
  ensurePlaneSize(&packed->momentumRectangular, kRectangularPlaneElements,
                  timings);
  packed->scaleSquare.clear();
  packed->scaleRectangular.clear();
  packed->squareBindings.clear();
  packed->rectangularBindings.clear();
  std::string registryError;
  const auto p = tiny_lm::parameterRegistry(parameters);
  const auto g = tiny_lm::parameterRegistry(gradients);
  const auto m = tiny_lm::parameterRegistry(momentum);
  if (!tiny_lm::validateParameterRegistry(parameters, &registryError) ||
      !tiny_lm::validateParameterRegistry(gradients, &registryError) ||
      !tiny_lm::validateParameterRegistry(momentum, &registryError) ||
      p.size() != g.size() || p.size() != m.size())
    return fail(error, "PACK_REGISTRY_INVALID:" + registryError);
  std::vector<std::size_t> rectangularWide;
  std::vector<std::size_t> rectangularTall;
  for (std::size_t index = 0; index < p.size(); ++index) {
    if (p[index].name != g[index].name || p[index].name != m[index].name ||
        p[index].role != g[index].role || p[index].role != m[index].role ||
        p[index].shape != g[index].shape || p[index].shape != m[index].shape)
      return fail(error, "PACK_REGISTRY_MISMATCH");
    if (p[index].role != tiny_lm::ParameterRole::MUON) continue;
    if (p[index].shape == std::vector<std::uint32_t>{64, 64}) {
      if (!appendBinding(p[index], g[index], m[index], index, false,
                         &packed->currentSquare, &packed->gradientSquare,
                          &packed->momentumSquare, &packed->scaleSquare,
                          &packed->squareBindings, checkFinite, true, timings,
                          error))
        return false;
    } else if (p[index].shape == std::vector<std::uint32_t>{64, 128} &&
               p[index].fanOut == 128 && p[index].fanIn == 64) {
      rectangularWide.push_back(index);
    } else if (p[index].shape == std::vector<std::uint32_t>{128, 64} &&
               p[index].fanOut == 64 && p[index].fanIn == 128) {
      rectangularTall.push_back(index);
    } else {
      return fail(error, "PACK_UNSUPPORTED_MUON_SHAPE:" + p[index].name);
    }
  }
  const auto appendGroup = [&](const std::vector<std::size_t>& indices,
                               bool transpose) {
    for (const std::size_t index : indices)
      if (!appendBinding(p[index], g[index], m[index], index, transpose,
                         &packed->currentRectangular,
                         &packed->gradientRectangular,
                         &packed->momentumRectangular,
                          &packed->scaleRectangular,
                          &packed->rectangularBindings, checkFinite, false,
                          timings, error))
        return false;
    return true;
  };
  if (!appendGroup(rectangularWide, false) ||
      !appendGroup(rectangularTall, true))
    return false;
  if (packed->squareBindings.size() != kSquareBatch ||
      rectangularWide.size() != 19 || rectangularTall.size() != 19 ||
      packed->rectangularBindings.size() != kRectangularBatch)
    return fail(error, "PACK_PARTITION_COUNT_MISMATCH");
  if (timings) {
    timings->registryTraversalUs = elapsedUs(totalStarted) -
        timings->allocationResizeUs - timings->metadataSetupUs -
        timings->squareWeightCopyUs - timings->w1WeightCopyUs -
        timings->w2WeightTransposeUs - timings->gradientCopyUs -
        timings->momentumCopyUs;
  }
  return true;
}

bool pack(const qnn::TinyTransformerParameters& parameters,
          const qnn::TinyTransformerParameters& gradients,
          const qnn::TinyTransformerParameters& momentum,
          PackedInputs* packed, std::string* error) {
  return packImpl(parameters, gradients, momentum, true, packed, error);
}

bool packForValidatedRpc(const qnn::TinyTransformerParameters& parameters,
                         const qnn::TinyTransformerParameters& gradients,
                         const qnn::TinyTransformerParameters& momentum,
                         PackedInputs* packed, std::string* error,
                         PackTimings* timings) {
  return packImpl(parameters, gradients, momentum, false, packed, error,
                  timings);
}

bool validateFinite(const PackedInputs& packed, std::string* error) {
  const bool valid = finite(packed.currentSquare) &&
      finite(packed.gradientSquare) && finite(packed.momentumSquare) &&
      finite(packed.scaleSquare) && finite(packed.currentRectangular) &&
      finite(packed.gradientRectangular) &&
      finite(packed.momentumRectangular) &&
      finite(packed.scaleRectangular);
  if (!valid) return fail(error, "RPC_INPUT_NONFINITE");
  return true;
}

bool unpack(const PackedInputs& packed,
            const std::vector<float>& nextSquareWeights,
            const std::vector<float>& nextSquareMomentum,
            const std::vector<float>& nextRectangularWeights,
            const std::vector<float>& nextRectangularMomentum,
            qnn::TinyTransformerParameters* parameters,
            qnn::TinyTransformerParameters* momentum,
            std::string* error, UnpackTimings* timings) {
  if (!parameters || !momentum)
    return fail(error, "UNPACK_NULL_OUTPUT");
  if (timings) *timings = {};
  const auto validationStarted = Clock::now();
  if (!finite(nextSquareWeights) || !finite(nextSquareMomentum) ||
      !finite(nextRectangularWeights) || !finite(nextRectangularMomentum))
    return fail(error, "UNPACK_NONFINITE");
  if (timings) timings->decodedValidationUs = elapsedUs(validationStarted);
  const auto registryStarted = Clock::now();
  auto p = tiny_lm::parameterRegistry(*parameters);
  auto m = tiny_lm::parameterRegistry(*momentum);
  if (timings) timings->registryTraversalUs = elapsedUs(registryStarted);
  if (p.size() != m.size()) return fail(error, "UNPACK_REGISTRY_MISMATCH");
  const auto unpackGroup = [&](const std::vector<MatrixBinding>& bindings,
                               const std::vector<float>& weights,
                               const std::vector<float>& momenta) {
    for (std::size_t index = 0; index < bindings.size(); ++index) {
      const auto& binding = bindings[index];
      if (binding.registryIndex >= p.size() ||
          p[binding.registryIndex].name != binding.name ||
          m[binding.registryIndex].name != binding.name)
        return fail(error, "UNPACK_IDENTITY_MISMATCH:" + binding.name);
      auto* weight = const_cast<std::vector<float>*>(
          p[binding.registryIndex].values);
      auto* nextMomentum = const_cast<std::vector<float>*>(
          m[binding.registryIndex].values);
      const auto copyStarted = Clock::now();
      if (!writeMatrix(weights, index, binding, weight, error) ||
          !writeMatrix(momenta, index, binding, nextMomentum, error))
        return false;
      if (timings) {
        double* target = binding.originalRows == 64 &&
                                 binding.originalColumns == 64
            ? &timings->squareOutputCopyUs
            : (binding.transposed ? &timings->w2TransposeBackUs
                                  : &timings->w1OutputCopyUs);
        *target += elapsedUs(copyStarted);
      }
    }
    return true;
  };
  return unpackGroup(packed.squareBindings, nextSquareWeights,
                     nextSquareMomentum) &&
         unpackGroup(packed.rectangularBindings, nextRectangularWeights,
                     nextRectangularMomentum);
}

}  // namespace phonelm::nicopedia_htp_muon
