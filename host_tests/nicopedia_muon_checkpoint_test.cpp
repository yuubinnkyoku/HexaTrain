// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host-only tests for the research-only NPRTCKPTV4 mixed Muon/Aux-Adam
// checkpoint codec.  No private checkpoint or corpus is read or written.

#include "nicopedia_muon_checkpoint.h"
#include "nicopedia_muon_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mc = phonelm::nicopedia_muon_checkpoint;
namespace tl = phonelm::tiny_lm;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void writeU32(std::vector<std::uint8_t>* bytes, std::size_t offset,
              std::uint32_t value) {
  require(bytes && offset <= bytes->size() && bytes->size() - offset >= 4,
          "serialized u32 offset out of range");
  for (int shift = 24; shift >= 0; shift -= 8)
    (*bytes)[offset + static_cast<std::size_t>((24 - shift) / 8)] =
        static_cast<std::uint8_t>((value >> shift) & 0xffu);
}

void writeU64(std::vector<std::uint8_t>* bytes, std::size_t offset,
              std::uint64_t value) {
  require(bytes && offset <= bytes->size() && bytes->size() - offset >= 8,
          "serialized u64 offset out of range");
  for (int shift = 56; shift >= 0; shift -= 8)
    (*bytes)[offset + static_cast<std::size_t>((56 - shift) / 8)] =
        static_cast<std::uint8_t>((value >> shift) & 0xffu);
}

void writeFloat(std::vector<std::uint8_t>* bytes, std::size_t offset,
                float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  writeU32(bytes, offset, bits);
}

std::uint32_t readU32(const std::vector<std::uint8_t>& bytes,
                      std::size_t offset) {
  require(offset <= bytes.size() && bytes.size() - offset >= 4,
          "serialized read u32 offset out of range");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i)
    value = (value << 8) | bytes[offset + static_cast<std::size_t>(i)];
  return value;
}

std::size_t skipString(const std::vector<std::uint8_t>& bytes,
                       std::size_t offset) {
  const std::uint32_t length = readU32(bytes, offset);
  const std::size_t next = offset + 4ull + length;
  require(next >= offset && next <= bytes.size(),
          "serialized string offset out of range");
  return next;
}

struct SerializedOffsets {
  std::size_t muonMomentum = 0;
  std::size_t nesterov = 0;
  std::size_t nsSteps = 0;
  std::size_t auxAdamBeta1 = 0;
  std::size_t auxAdamBeta2 = 0;
  std::size_t auxAdamEpsilon = 0;
  std::size_t muonWeightDecay = 0;
  std::size_t auxAdamWeightDecay = 0;
  std::size_t parameterCount = 0;
  std::size_t firstParameterRows = 0;
  std::size_t firstValuesCount = 0;
};

SerializedOffsets serializedOffsets(const std::vector<std::uint8_t>& bytes) {
  // Keep this layout mirror deliberately local to the host negative tests.
  // Every offset is derived from the encoded string lengths rather than from
  // a hard-coded sample hash/name length.
  std::size_t offset = std::strlen(mc::kMagic);
  offset += 6 * sizeof(std::uint32_t) + sizeof(float);  // model config
  offset += sizeof(std::uint32_t) + sizeof(std::uint64_t);  // seed/step
  offset = skipString(bytes, offset);  // tokenizer kind
  offset = skipString(bytes, offset);  // tokenizer hash
  offset = skipString(bytes, offset);  // dataset hash
  offset += 5 * sizeof(std::uint64_t);  // data cursor
  offset = skipString(bytes, offset);  // optimizer identity

  offset += 4 * sizeof(float);  // learning rates
  SerializedOffsets result;
  result.muonMomentum = offset;
  offset += sizeof(float);
  result.nesterov = offset;
  offset += sizeof(std::uint32_t);
  result.nsSteps = offset;
  offset += sizeof(std::uint32_t);
  result.auxAdamBeta1 = offset;
  offset += sizeof(float);
  result.auxAdamBeta2 = offset;
  offset += sizeof(float);
  result.auxAdamEpsilon = offset;
  offset += sizeof(float);
  result.muonWeightDecay = offset;
  offset += sizeof(float);
  result.auxAdamWeightDecay = offset;
  offset += sizeof(float);
  offset += 3 * sizeof(std::uint32_t);  // schedule
  offset += 2 * sizeof(std::uint32_t);  // schema/registry versions
  result.parameterCount = offset;
  offset += sizeof(std::uint32_t);
  offset = skipString(bytes, offset);  // first parameter name
  offset += 3 * sizeof(std::uint32_t);  // role and 2-D shape
  result.firstParameterRows = offset - 2 * sizeof(std::uint32_t);
  result.firstValuesCount = offset;
  return result;
}

bool sameConfig(const tl::Config& a, const tl::Config& b) {
  return a.vocabularySize == b.vocabularySize && a.tokens == b.tokens &&
         a.dimension == b.dimension &&
         a.feedForwardDimension == b.feedForwardDimension &&
         a.numLayers == b.numLayers && a.numHeads == b.numHeads &&
         a.epsilon == b.epsilon;
}

bool same(const mc::Checkpoint& a, const mc::Checkpoint& b) {
  if (a.optimizerIdentity != b.optimizerIdentity ||
      !sameConfig(a.identity.config, b.identity.config) ||
      a.identity.seed != b.identity.seed ||
      a.identity.globalStep != b.identity.globalStep ||
      a.identity.tokenizerKind != b.identity.tokenizerKind ||
      a.identity.tokenizerHash != b.identity.tokenizerHash ||
      a.identity.dataCursor.datasetHash != b.identity.dataCursor.datasetHash ||
      a.identity.dataCursor.recordIndex != b.identity.dataCursor.recordIndex ||
      a.identity.dataCursor.tokenOffset != b.identity.dataCursor.tokenOffset ||
      a.identity.dataCursor.epoch != b.identity.dataCursor.epoch ||
      a.identity.dataCursor.exposedTokens !=
          b.identity.dataCursor.exposedTokens ||
      a.identity.dataCursor.orderSeed != b.identity.dataCursor.orderSeed)
    return false;
  const auto& ah = a.hyperparameters;
  const auto& bh = b.hyperparameters;
  if (ah.muonLearningRate != bh.muonLearningRate ||
      ah.auxAdamLearningRate != bh.auxAdamLearningRate ||
      ah.muonTargetLearningRate != bh.muonTargetLearningRate ||
      ah.auxAdamTargetLearningRate != bh.auxAdamTargetLearningRate ||
      ah.muonMomentum != bh.muonMomentum ||
      ah.muonNesterov != bh.muonNesterov || ah.muonNsSteps != bh.muonNsSteps ||
      ah.auxAdamBeta1 != bh.auxAdamBeta1 ||
      ah.auxAdamBeta2 != bh.auxAdamBeta2 ||
      ah.auxAdamEpsilon != bh.auxAdamEpsilon ||
      ah.muonWeightDecay != bh.muonWeightDecay ||
      ah.auxAdamWeightDecay != bh.auxAdamWeightDecay ||
      ah.decayStartStep != bh.decayStartStep ||
      ah.decayEndStep != bh.decayEndStep ||
      ah.scheduleTotalSteps != bh.scheduleTotalSteps)
    return false;
  if (a.parameters.size() != b.parameters.size()) return false;
  for (std::size_t i = 0; i < a.parameters.size(); ++i) {
    const auto& x = a.parameters[i];
    const auto& y = b.parameters[i];
    if (x.name != y.name || x.role != y.role ||
        x.shape.rows != y.shape.rows || x.shape.columns != y.shape.columns ||
        x.values != y.values || x.momentum != y.momentum ||
        x.adamM != y.adamM || x.adamV != y.adamV)
      return false;
  }
  return true;
}

mc::Checkpoint makeCheckpoint() {
  mc::Checkpoint checkpoint;
  auto& identity = checkpoint.identity;
  identity.config.vocabularySize = 8;
  identity.config.tokens = 3;
  identity.config.dimension = 4;
  identity.config.feedForwardDimension = 6;
  identity.config.epsilon = 1.0e-5f;
  identity.config.numLayers = 2;
  identity.config.numHeads = 2;
  identity.seed = 1;
  identity.globalStep = 1;
  identity.tokenizerKind = "byte_bpe";
  identity.tokenizerHash = "sha256:" + std::string(64, 'a');
  identity.dataCursor.datasetHash = "fnv1a64:" + std::string(16, 'b');
  identity.dataCursor.recordIndex = 7;
  identity.dataCursor.tokenOffset = 13;
  identity.dataCursor.epoch = 2;
  identity.dataCursor.exposedTokens = 192;
  identity.dataCursor.orderSeed = 0x1020304050607080ull;

  auto& hp = checkpoint.hyperparameters;
  hp.muonLearningRate = 0.005f;
  hp.auxAdamLearningRate = 0.0022f;
  hp.muonTargetLearningRate = 0.005f * (0.0001f / 0.0022f);
  hp.auxAdamTargetLearningRate = 0.0001f;
  hp.muonMomentum = 0.95f;
  hp.muonNesterov = true;
  hp.muonNsSteps = 5;
  hp.auxAdamBeta1 = 0.9f;
  hp.auxAdamBeta2 = 0.999f;
  hp.auxAdamEpsilon = 1.0e-8f;
  hp.muonWeightDecay = 0.0f;
  hp.auxAdamWeightDecay = 0.0f;
  hp.decayStartStep = 4000;
  hp.decayEndStep = 8000;
  hp.scheduleTotalSteps = 8000;

  const auto registry = mc::expectedRegistry(identity.config);
  checkpoint.parameters.reserve(registry.size());
  for (std::size_t index = 0; index < registry.size(); ++index) {
    const auto& entry = registry[index];
    mc::ParameterState parameter;
    parameter.name = entry.name;
    parameter.role = entry.role;
    parameter.shape = entry.shape;
    const std::size_t count = static_cast<std::size_t>(
        std::uint64_t(entry.shape.rows) * entry.shape.columns);
    parameter.values.resize(count);
    for (std::size_t i = 0; i < count; ++i)
      parameter.values[i] = 0.001f * float(index + 1) +
                           0.00001f * float(i + 1);
    if (entry.role == mc::ParameterRole::MUON) {
      parameter.momentum.resize(count);
      for (std::size_t i = 0; i < count; ++i)
        parameter.momentum[i] = -0.0003f * float(index + 1) +
                               0.000001f * float(i + 1);
    } else {
      parameter.adamM.resize(count);
      parameter.adamV.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        parameter.adamM[i] = 0.0002f * float(index + 1) +
                             0.000002f * float(i + 1);
        parameter.adamV[i] = 0.00001f * float(index + 1) +
                             0.0000001f * float(i + 1);
      }
    }
    checkpoint.parameters.push_back(std::move(parameter));
  }
  return checkpoint;
}

void testRegistry() {
  tl::Config config;
  config.vocabularySize = 1024;
  config.tokens = 32;
  config.dimension = 64;
  config.feedForwardDimension = 128;
  config.epsilon = 1.0e-5f;
  config.numLayers = 19;
  config.numHeads = 2;
  const auto registry = mc::expectedRegistry(config);
  require(registry.size() == 192, "V1024 registry count mismatch");
  std::size_t muonCount = 0;
  std::size_t auxCount = 0;
  for (const auto& entry : registry) {
    if (entry.role == mc::ParameterRole::MUON) {
      ++muonCount;
      require(entry.shape.rows == 64 || entry.shape.rows == 128,
              "Muon row shape mismatch");
      require(entry.shape.columns == 64 || entry.shape.columns == 128,
              "Muon column shape mismatch");
    } else {
      ++auxCount;
      require(entry.role == mc::ParameterRole::AUX_ADAM,
              "unknown auxiliary role");
    }
  }
  require(muonCount == 114, "V1024 Muon count mismatch");
  require(auxCount == 78, "V1024 auxiliary count mismatch");
  mc::ParameterRole role = mc::ParameterRole::MUON;
  mc::MatrixShape shape;
  std::string error;
  require(mc::classifyParameter(config, "layer_000.wq", &role, &shape,
                                &error),
          "exact Muon classification failed");
  require(role == mc::ParameterRole::MUON && shape.rows == 64 &&
              shape.columns == 64,
          "Wq classification shape mismatch");
  require(mc::classifyParameter(config, "layer_000.ffn_w1", &role, &shape,
                                &error),
          "exact FFN classification failed");
  require(role == mc::ParameterRole::MUON && shape.rows == 64 &&
              shape.columns == 128,
          "FFN W1 classification shape mismatch");
  require(mc::classifyParameter(config, "layer_000.norm1_gamma", &role,
                                &shape, &error) &&
              role == mc::ParameterRole::AUX_ADAM && shape.rows == 1 &&
              shape.columns == 64,
          "norm classification mismatch");
  require(!mc::classifyParameter(config, "some_random_wq_suffix", &role,
                                 &shape, &error),
          "unclassified name accepted");
}

void testRoundTripAndRejections() {
  const mc::Checkpoint original = makeCheckpoint();
  std::vector<std::uint8_t> bytes;
  std::string error;
  require(mc::validateCheckpoint(original, &error),
          "valid Muon checkpoint rejected");
  require(mc::encodeCheckpoint(original, &bytes, &error),
          "Muon checkpoint encode failed");
  mc::Checkpoint decoded;
  require(mc::decodeCheckpoint(bytes, &decoded, &error),
          "Muon checkpoint decode failed");
  require(same(original, decoded), "V4 roundtrip changed state");
  std::vector<std::uint8_t> reencoded;
  require(mc::encodeCheckpoint(decoded, &reencoded, &error),
          "V4 re-encode failed");
  require(bytes == reencoded, "V4 encoding is not deterministic");

  tl::Config config = original.identity.config;
  phonelm::qnn::TinyTransformerParameters parameters;
  require(mc::extractParameters(original, config, original.identity.seed,
                                &parameters, &error),
          "V4 parameter extraction failed");
  const auto destination = tl::parameterRegistry(parameters);
  require(destination.size() == original.parameters.size(),
          "extracted registry count mismatch");
  for (std::size_t i = 0; i < destination.size(); ++i)
    require(*destination[i].values == original.parameters[i].values,
            "extracted parameter mismatch");
  require(!mc::extractParameters(original, config, 2, &parameters, &error) &&
              error == "NPRT_CKPT_V4_SEED_MISMATCH",
          "wrong seed extraction accepted");

  mc::Checkpoint adamIdentity = original;
  adamIdentity.optimizerIdentity = "adam";
  require(!mc::encodeCheckpoint(adamIdentity, &reencoded, &error) &&
              error == "NPRT_CKPT_V4_OPTIMIZER_IDENTITY",
          "Adam identity accepted by Muon codec");
  const std::string legacyMagic = "NPRTCKPTV3\n";
  const std::vector<std::uint8_t> legacy(legacyMagic.begin(), legacyMagic.end());
  require(!mc::decodeCheckpoint(legacy, &decoded, &error) &&
              error == "NPRT_CKPT_V4_MAGIC",
          "Adam/V3 checkpoint accepted as V4");
  require(!mc::decodeCheckpoint(bytes, &decoded, &error, "adam") &&
              error == "NPRT_CKPT_V4_OPTIMIZER_MISMATCH",
          "Muon checkpoint accepted for Adam resume");

  const auto offsets = serializedOffsets(bytes);
  auto malformedConfig = bytes;
  writeU32(&malformedConfig, std::strlen(mc::kMagic),
           std::numeric_limits<std::uint32_t>::max());
  require(!mc::decodeCheckpoint(malformedConfig, &decoded, &error) &&
              error == "NPRT_CKPT_V4_CONFIG_IDENTITY",
          "oversized config reached allocation path");

  auto malformedParameterCount = bytes;
  writeU32(&malformedParameterCount, offsets.parameterCount,
           std::numeric_limits<std::uint32_t>::max());
  require(!mc::decodeCheckpoint(malformedParameterCount, &decoded, &error) &&
              error == "NPRT_CKPT_V4_REGISTRY_COUNT",
          "oversized registry count accepted");

  auto malformedStateCount = bytes;
  writeU64(&malformedStateCount, offsets.firstValuesCount,
           std::numeric_limits<std::uint64_t>::max());
  require(!mc::decodeCheckpoint(malformedStateCount, &decoded, &error) &&
              error == "NPRT_CKPT_V4_STATE_SHAPE",
          "oversized optimizer state count accepted");

  auto malformedShape = bytes;
  writeU32(&malformedShape, offsets.firstParameterRows,
           std::numeric_limits<std::uint32_t>::max());
  require(!mc::decodeCheckpoint(malformedShape, &decoded, &error) &&
              error == "NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY",
          "oversized parameter shape accepted");

  auto nonV1 = bytes;
  writeFloat(&nonV1, offsets.muonMomentum, 0.90f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 Muon momentum accepted");
  nonV1 = bytes;
  writeU32(&nonV1, offsets.nesterov, 0);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-Nesterov Muon accepted");
  nonV1 = bytes;
  writeU32(&nonV1, offsets.nsSteps, 4);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 Newton-Schulz step count accepted");
  nonV1 = bytes;
  writeFloat(&nonV1, offsets.auxAdamBeta1, 0.8f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 auxiliary Adam beta1 accepted");
  nonV1 = bytes;
  writeFloat(&nonV1, offsets.auxAdamBeta2, 0.99f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 auxiliary Adam beta2 accepted");
  nonV1 = bytes;
  writeFloat(&nonV1, offsets.auxAdamEpsilon, 1.0e-7f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 auxiliary Adam epsilon accepted");
  nonV1 = bytes;
  writeFloat(&nonV1, offsets.muonWeightDecay, 1.0e-4f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 Muon weight decay accepted");
  nonV1 = bytes;
  writeFloat(&nonV1, offsets.auxAdamWeightDecay, 1.0e-4f);
  require(!mc::decodeCheckpoint(nonV1, &decoded, &error) &&
              error == "NPRT_CKPT_V4_HYPERPARAMETERS",
          "non-v1 auxiliary Adam weight decay accepted");

  auto trailing = bytes;
  trailing.push_back(0x5a);
  require(!mc::decodeCheckpoint(trailing, &decoded, &error) &&
              error == "NPRT_CKPT_V4_TRAILING_BYTES",
          "trailing V4 bytes accepted");

  mc::Checkpoint nonfinite = original;
  nonfinite.parameters.front().values.front() =
      std::numeric_limits<float>::quiet_NaN();
  require(!mc::validateCheckpoint(nonfinite, &error) &&
              error == "NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY",
          "non-finite parameter accepted");
  nonfinite = original;
  nonfinite.parameters[3].momentum.front() =
      std::numeric_limits<float>::infinity();
  require(!mc::encodeCheckpoint(nonfinite, &reencoded, &error) &&
              error == "NPRT_CKPT_V4_MUON_STATE_IDENTITY",
          "non-finite Muon state accepted");

  mc::Checkpoint wrongShape = original;
  wrongShape.parameters[3].shape.columns++;
  require(!mc::validateCheckpoint(wrongShape, &error) &&
              error == "NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY",
          "wrong matrix shape accepted");
  mc::Checkpoint missing = original;
  missing.parameters.pop_back();
  require(!mc::validateCheckpoint(missing, &error) &&
              error == "NPRT_CKPT_V4_REGISTRY_COUNT",
          "registry count mismatch accepted");
  mc::Checkpoint doubleClassified = original;
  doubleClassified.parameters[3].role = mc::ParameterRole::AUX_ADAM;
  require(!mc::validateCheckpoint(doubleClassified, &error) &&
              error == "NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY",
          "double/misclassified parameter accepted");

  // Zero optimizer state is valid and must remain finite after a roundtrip.
  mc::Checkpoint zeroState = original;
  for (auto& parameter : zeroState.parameters) {
    std::fill(parameter.momentum.begin(), parameter.momentum.end(), 0.0f);
    std::fill(parameter.adamM.begin(), parameter.adamM.end(), 0.0f);
    std::fill(parameter.adamV.begin(), parameter.adamV.end(), 0.0f);
  }
  require(mc::encodeCheckpoint(zeroState, &reencoded, &error),
          "zero state encode failed");
  require(mc::decodeCheckpoint(reencoded, &decoded, &error) &&
              same(zeroState, decoded),
          "zero state roundtrip failed");
}

using Params = phonelm::qnn::TinyTransformerParameters;

void zeroParameters(Params* parameters) {
  require(parameters != nullptr, "null parameters in zeroParameters");
  const auto registry = tl::parameterRegistry(*parameters);
  for (const auto& entry : registry)
    const_cast<std::vector<float>*>(entry.values)->assign(
        entry.values->size(), 0.0f);
}

Params makeGradients(const tl::Config& config, std::uint32_t seed,
                     std::uint64_t step) {
  Params gradients = tl::initialParameters(config, seed);
  const auto registry = tl::parameterRegistry(gradients);
  for (std::size_t pi = 0; pi < registry.size(); ++pi) {
    auto& values = *const_cast<std::vector<float>*>(registry[pi].values);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = 0.0001f * float((pi % 7) + 1) +
                  0.000001f * float((i % 11) + 1) +
                  0.0000001f * float(step % 17);
  }
  return gradients;
}

struct ProductionTrajectory {
  mc::Checkpoint checkpoint;
  Params parameters;
  Params momentum;
  Params adamM;
  Params adamV;
};

ProductionTrajectory makeProductionTrajectory(const mc::Checkpoint& checkpoint) {
  ProductionTrajectory trajectory;
  trajectory.checkpoint = checkpoint;
  std::string error;
  require(mc::extractParameters(checkpoint, checkpoint.identity.config,
                                checkpoint.identity.seed,
                                &trajectory.parameters, &error),
          "production trajectory parameter extraction failed");
  trajectory.momentum = tl::initialParameters(checkpoint.identity.config,
                                              checkpoint.identity.seed);
  trajectory.adamM = trajectory.momentum;
  trajectory.adamV = trajectory.momentum;
  zeroParameters(&trajectory.momentum);
  zeroParameters(&trajectory.adamM);
  zeroParameters(&trajectory.adamV);
  const auto momentum = tl::parameterRegistry(trajectory.momentum);
  const auto adamM = tl::parameterRegistry(trajectory.adamM);
  const auto adamV = tl::parameterRegistry(trajectory.adamV);
  require(momentum.size() == checkpoint.parameters.size() &&
              adamM.size() == checkpoint.parameters.size() &&
              adamV.size() == checkpoint.parameters.size(),
          "production trajectory state registry mismatch");
  for (std::size_t index = 0; index < checkpoint.parameters.size(); ++index) {
    const auto& source = checkpoint.parameters[index];
    if (source.role == mc::ParameterRole::MUON)
      *const_cast<std::vector<float>*>(momentum[index].values) =
          source.momentum;
    else {
      *const_cast<std::vector<float>*>(adamM[index].values) = source.adamM;
      *const_cast<std::vector<float>*>(adamV[index].values) = source.adamV;
    }
  }
  return trajectory;
}

void syncCheckpoint(ProductionTrajectory* trajectory) {
  const auto parameters = tl::parameterRegistry(trajectory->parameters);
  const auto momentum = tl::parameterRegistry(trajectory->momentum);
  const auto adamM = tl::parameterRegistry(trajectory->adamM);
  const auto adamV = tl::parameterRegistry(trajectory->adamV);
  require(parameters.size() == trajectory->checkpoint.parameters.size() &&
              momentum.size() == parameters.size() &&
              adamM.size() == parameters.size() && adamV.size() == parameters.size(),
          "production trajectory sync registry mismatch");
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    auto& destination = trajectory->checkpoint.parameters[index];
    destination.values = *parameters[index].values;
    if (destination.role == mc::ParameterRole::MUON)
      destination.momentum = *momentum[index].values;
    else {
      destination.adamM = *adamM[index].values;
      destination.adamV = *adamV[index].values;
    }
  }
}

void advanceProduction(ProductionTrajectory* trajectory, std::uint64_t step) {
  const auto& hp = trajectory->checkpoint.hyperparameters;
  phonelm::nicopedia_muon::Config config;
  config.muonLearningRate = hp.muonLearningRate;
  config.auxiliaryAdamLearningRate = hp.auxAdamLearningRate;
  config.momentum = hp.muonMomentum;
  config.nesterov = hp.muonNesterov;
  config.nsSteps = hp.muonNsSteps;
  config.optimizerStep = step;
  const Params gradients = makeGradients(trajectory->checkpoint.identity.config,
                                         trajectory->checkpoint.identity.seed + 17,
                                         step);
  auto result = phonelm::nicopedia_muon::update(
      trajectory->parameters, gradients, trajectory->momentum,
      trajectory->adamM, trajectory->adamV, config);
  require(result.error.empty(), "production Muon update failed");
  require(result.health.gradientFinite && result.health.momentumFinite &&
              result.health.normalizedFinite && result.health.nsOutputFinite &&
              result.health.updateFinite && result.health.parametersFinite,
          "production Muon update reported non-finite state");
  require(result.muonMatrixCount == 12 && result.muonParameterCount == 224 &&
              result.auxiliaryAdamParameterCount == 96,
          "production Muon partition count mismatch");
  trajectory->parameters = std::move(result.parameters);
  trajectory->momentum = std::move(result.muonMomentum);
  trajectory->adamM = std::move(result.auxiliaryAdamM);
  trajectory->adamV = std::move(result.auxiliaryAdamV);
  trajectory->checkpoint.identity.globalStep = step;
  trajectory->checkpoint.identity.dataCursor.recordIndex = step * 7;
  trajectory->checkpoint.identity.dataCursor.tokenOffset = step * 13;
  trajectory->checkpoint.identity.dataCursor.exposedTokens = step * 192;
  syncCheckpoint(trajectory);
}

void testResumeParity() {
  const mc::Checkpoint initial = makeCheckpoint();
  ProductionTrajectory uninterrupted = makeProductionTrajectory(initial);
  for (std::uint64_t step = 1; step <= 8; ++step)
    advanceProduction(&uninterrupted, step);

  ProductionTrajectory beforeResume = makeProductionTrajectory(initial);
  for (std::uint64_t step = 1; step <= 4; ++step)
    advanceProduction(&beforeResume, step);
  std::vector<std::uint8_t> bytes;
  std::string error;
  require(mc::encodeCheckpoint(beforeResume.checkpoint, &bytes, &error),
          "resume checkpoint encode failed");
  mc::Checkpoint resumedCheckpoint;
  require(mc::decodeCheckpoint(bytes, &resumedCheckpoint, &error),
          "resume checkpoint decode failed");
  require(resumedCheckpoint.identity.globalStep == 4 &&
              resumedCheckpoint.identity.dataCursor.recordIndex == 28,
          "resume cursor identity mismatch");
  ProductionTrajectory resumed = makeProductionTrajectory(resumedCheckpoint);
  for (std::uint64_t step = 5; step <= 8; ++step)
    advanceProduction(&resumed, step);
  require(same(uninterrupted.checkpoint, resumed.checkpoint),
          "fresh 8-step and 4+resume+4 production trajectories differ");
}

}  // namespace

int main() {
  try {
    testRegistry();
    testRoundTripAndRejections();
    testResumeParity();
    std::cout << "nicopedia_muon_checkpoint=PASS\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "nicopedia_muon_checkpoint=FAIL\nerror="
              << exception.what() << '\n';
    return 1;
  }
}
