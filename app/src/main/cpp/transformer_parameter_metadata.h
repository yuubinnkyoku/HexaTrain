// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "qnn/qnn_runtime.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace phonelm::tiny_lm {

enum class ParameterRole : std::uint8_t {
  UNKNOWN = 0,
  MUON = 1,
  AUX_ADAM = 2,
  kUnknown = UNKNOWN,
  kMuon = MUON,
  kAuxAdam = AUX_ADAM,
};
using ParameterSemanticRole = ParameterRole;

enum class ParameterPlacement : std::uint8_t {
  GLOBAL_PREFIX,
  PER_LAYER,
  GLOBAL_SUFFIX,
};

enum class ParameterCondition : std::uint8_t {
  ALWAYS,
  HEADWISE_G1,
};

enum class ParameterDimension : std::uint8_t {
  NONE,
  VOCABULARY,
  MODEL,
  FEED_FORWARD,
  HEADS,
};

struct ParameterInfo {
  std::string name;
  const std::vector<float>* values = nullptr;
  ParameterRole role = ParameterRole::UNKNOWN;
  std::vector<std::uint32_t> shape;
  // Semantic linear-map axes used by Keller-original Muon LR adjustment.
  // Storage is [input, output] in this model, while Muon defines the ratio as
  // fan_out/fan_in; keeping both avoids name-based inference.
  std::uint32_t fanOut = 0;
  std::uint32_t fanIn = 0;
  std::string_view suffix;
  ParameterPlacement placement = ParameterPlacement::PER_LAYER;
  ParameterCondition condition = ParameterCondition::ALWAYS;
};

struct ParameterDimensions {
  std::uint64_t vocabulary = 0;
  std::uint64_t model = 0;
  std::uint64_t feedForward = 0;
  std::uint64_t layers = 0;
  std::uint64_t heads = 0;
  bool headwiseG1 = false;
};

using LayerParameterMember =
    std::vector<float> qnn::TinyTransformerLayerParameters::*;
using GlobalParameterMember =
    std::vector<float> qnn::TinyTransformerParameters::*;

// One declaration owns parameter identity, canonical suffix, ordering group,
// shape semantics, optimizer role, optionality, storage binding, and Muon
// fan axes. No numeric ParameterId is needed: the member pointer is identity.
struct ParameterDefinition {
  const char* suffix;
  ParameterRole role;
  ParameterPlacement placement;
  ParameterCondition condition;
  std::array<ParameterDimension, 2> shape;
  std::uint8_t rank;
  std::int8_t fanOutAxis;
  std::int8_t fanInAxis;
  LayerParameterMember layerMember;
  GlobalParameterMember globalMember;
};

inline const std::array<ParameterDefinition, 13>& parameterDefinitions() {
  using Layer = qnn::TinyTransformerLayerParameters;
  using Parameters = qnn::TinyTransformerParameters;
  static const std::array<ParameterDefinition, 13> definitions{{
      {"token_embedding", ParameterRole::AUX_ADAM,
       ParameterPlacement::GLOBAL_PREFIX, ParameterCondition::ALWAYS,
       {ParameterDimension::VOCABULARY, ParameterDimension::MODEL}, 2, -1, -1,
       nullptr, &Parameters::tokenEmbedding},
      {"norm1_gamma", ParameterRole::AUX_ADAM,
       ParameterPlacement::PER_LAYER, ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::NONE}, 1, -1, -1,
       &Layer::gamma1, nullptr},
      {"norm1_beta", ParameterRole::AUX_ADAM,
       ParameterPlacement::PER_LAYER, ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::NONE}, 1, -1, -1,
       &Layer::beta1, nullptr},
      {"wq", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::MODEL}, 2, 1, 0,
       &Layer::wq, nullptr},
      {"wk", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::MODEL}, 2, 1, 0,
       &Layer::wk, nullptr},
      {"wv", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::MODEL}, 2, 1, 0,
       &Layer::wv, nullptr},
      {"wo", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::MODEL}, 2, 1, 0,
       &Layer::wo, nullptr},
      {"attention_gate_weight", ParameterRole::AUX_ADAM,
       ParameterPlacement::PER_LAYER, ParameterCondition::HEADWISE_G1,
       {ParameterDimension::MODEL, ParameterDimension::HEADS}, 2, -1, -1,
       &Layer::attentionGateWeight, nullptr},
      {"norm2_gamma", ParameterRole::AUX_ADAM,
       ParameterPlacement::PER_LAYER, ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::NONE}, 1, -1, -1,
       &Layer::gamma2, nullptr},
      {"norm2_beta", ParameterRole::AUX_ADAM,
       ParameterPlacement::PER_LAYER, ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::NONE}, 1, -1, -1,
       &Layer::beta2, nullptr},
      {"ffn_w1", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::FEED_FORWARD}, 2, 1, 0,
       &Layer::w1, nullptr},
      {"ffn_w2", ParameterRole::MUON, ParameterPlacement::PER_LAYER,
       ParameterCondition::ALWAYS,
       {ParameterDimension::FEED_FORWARD, ParameterDimension::MODEL}, 2, 1, 0,
       &Layer::w2, nullptr},
      {"output_projection", ParameterRole::AUX_ADAM,
       ParameterPlacement::GLOBAL_SUFFIX, ParameterCondition::ALWAYS,
       {ParameterDimension::MODEL, ParameterDimension::VOCABULARY}, 2, -1, -1,
       nullptr, &Parameters::outputProjection},
  }};
  return definitions;
}

inline bool validParameterDefinition(const ParameterDefinition& definition) {
  if (!definition.suffix || definition.suffix[0] == '\0' ||
      definition.rank == 0 || definition.rank > definition.shape.size() ||
      (definition.role != ParameterRole::MUON &&
       definition.role != ParameterRole::AUX_ADAM) ||
      (definition.placement != ParameterPlacement::GLOBAL_PREFIX &&
       definition.placement != ParameterPlacement::PER_LAYER &&
       definition.placement != ParameterPlacement::GLOBAL_SUFFIX) ||
      (definition.condition != ParameterCondition::ALWAYS &&
       definition.condition != ParameterCondition::HEADWISE_G1))
    return false;
  for (std::size_t axis = 0; axis < definition.shape.size(); ++axis) {
    const bool active = axis < definition.rank;
    if ((definition.shape[axis] == ParameterDimension::NONE) == active)
      return false;
  }
  const auto validAxis = [&](std::int8_t axis) {
    return axis < 0 || static_cast<std::uint8_t>(axis) < definition.rank;
  };
  if (!validAxis(definition.fanOutAxis) ||
      !validAxis(definition.fanInAxis))
    return false;
  if (definition.role == ParameterRole::MUON &&
      (definition.rank != 2 || definition.fanOutAxis < 0 ||
       definition.fanInAxis < 0))
    return false;
  if (definition.role == ParameterRole::AUX_ADAM &&
      (definition.fanOutAxis >= 0 || definition.fanInAxis >= 0))
    return false;
  if (definition.placement == ParameterPlacement::PER_LAYER)
    return definition.layerMember != nullptr && definition.globalMember == nullptr;
  return definition.layerMember == nullptr && definition.globalMember != nullptr;
}

inline bool parameterDefinitionEnabled(const ParameterDefinition& definition,
                                       const ParameterDimensions& dimensions) {
  return definition.condition == ParameterCondition::ALWAYS ||
         (definition.condition == ParameterCondition::HEADWISE_G1 &&
          dimensions.headwiseG1);
}

inline std::uint64_t parameterDimensionExtent(
    ParameterDimension dimension, const ParameterDimensions& dimensions) {
  switch (dimension) {
    case ParameterDimension::VOCABULARY:
      return dimensions.vocabulary;
    case ParameterDimension::MODEL:
      return dimensions.model;
    case ParameterDimension::FEED_FORWARD:
      return dimensions.feedForward;
    case ParameterDimension::HEADS:
      return dimensions.heads;
    case ParameterDimension::NONE:
      return 0;
  }
  return 0;
}

inline bool checkedParameterElementCount(const ParameterDimensions& dimensions,
                                         std::uint64_t* result) {
  if (!result || !dimensions.vocabulary || !dimensions.model ||
      !dimensions.feedForward || !dimensions.layers || !dimensions.heads)
    return false;
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t total = 0;
  for (const auto& definition : parameterDefinitions()) {
    if (!validParameterDefinition(definition)) return false;
    if (!parameterDefinitionEnabled(definition, dimensions)) continue;
    std::uint64_t elements = 1;
    for (std::uint8_t axis = 0; axis < definition.rank; ++axis) {
      const std::uint64_t extent =
          parameterDimensionExtent(definition.shape[axis], dimensions);
      if (!extent || elements > kMax / extent) return false;
      elements *= extent;
    }
    const std::uint64_t instances =
        definition.placement == ParameterPlacement::PER_LAYER
            ? dimensions.layers
            : 1;
    if (elements > kMax / instances) return false;
    elements *= instances;
    if (elements > kMax - total) return false;
    total += elements;
  }
  *result = total;
  return true;
}

inline std::vector<ParameterInfo> parameterMetadata(
    const ParameterDimensions& dimensions,
    const qnn::TinyTransformerParameters* parameters = nullptr) {
  std::vector<ParameterInfo> result;
  if (!dimensions.layers || dimensions.layers > 999 ||
      dimensions.vocabulary > std::numeric_limits<std::uint32_t>::max() ||
      dimensions.model > std::numeric_limits<std::uint32_t>::max() ||
      dimensions.feedForward > std::numeric_limits<std::uint32_t>::max() ||
      dimensions.heads > std::numeric_limits<std::uint32_t>::max())
    return result;

  const auto append = [&](const ParameterDefinition& definition,
                          std::uint32_t layerIndex,
                          std::vector<ParameterInfo>* output) {
    if (!validParameterDefinition(definition)) return false;
    if (!parameterDefinitionEnabled(definition, dimensions)) return true;
    ParameterInfo info;
    info.suffix = definition.suffix;
    info.placement = definition.placement;
    info.condition = definition.condition;
    info.role = definition.role;
    if (definition.placement == ParameterPlacement::PER_LAYER) {
      std::ostringstream index;
      index << std::setw(3) << std::setfill('0') << layerIndex;
      info.name = "layer_" + index.str() + "." + definition.suffix;
      if (parameters && layerIndex < dimensions.layers &&
          parameters->layers.size() + 1 == dimensions.layers) {
        const auto& layer =
            layerIndex == 0
                ? static_cast<const qnn::TinyTransformerLayerParameters&>(
                      *parameters)
                : parameters->layers[layerIndex - 1];
        info.values = &(layer.*(definition.layerMember));
      }
    } else {
      info.name = definition.suffix;
      if (parameters) info.values = &((*parameters).*(definition.globalMember));
    }
    for (std::uint8_t axis = 0; axis < definition.rank; ++axis)
      info.shape.push_back(static_cast<std::uint32_t>(
          parameterDimensionExtent(definition.shape[axis], dimensions)));
    if (definition.fanOutAxis >= 0)
      info.fanOut = info.shape[static_cast<std::size_t>(definition.fanOutAxis)];
    if (definition.fanInAxis >= 0)
      info.fanIn = info.shape[static_cast<std::size_t>(definition.fanInAxis)];
    output->push_back(std::move(info));
    return true;
  };

  for (const auto& definition : parameterDefinitions())
    if (definition.placement == ParameterPlacement::GLOBAL_PREFIX)
      if (!append(definition, 0, &result)) return {};
  for (std::uint32_t layer = 0; layer < dimensions.layers; ++layer)
    for (const auto& definition : parameterDefinitions())
      if (definition.placement == ParameterPlacement::PER_LAYER)
        if (!append(definition, layer, &result)) return {};
  for (const auto& definition : parameterDefinitions())
    if (definition.placement == ParameterPlacement::GLOBAL_SUFFIX)
      if (!append(definition, 0, &result)) return {};
  return result;
}

}  // namespace phonelm::tiny_lm
