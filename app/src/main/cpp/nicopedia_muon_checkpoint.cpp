// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku

#include "nicopedia_muon_checkpoint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace phonelm::nicopedia_muon_checkpoint {
namespace {

constexpr std::size_t kMagicBytes = sizeof(kMagic) - 1;
constexpr std::size_t kMaxCheckpointBytes = 256u * 1024u * 1024u;
constexpr std::uint32_t kMaxRegistryEntries = 10000;
constexpr std::uint64_t kMaxParameterElements = 100000000ull;
constexpr std::uint32_t kMaxStringBytes = 4096;

bool fail(std::string* error, const char* message) {
  if (error) *error = message;
  return false;
}

bool validConfig(const tiny_lm::Config& config) {
  // This function is called before expectedRegistry()/initialParameters() in
  // the decoder. Keep the cheap shape checks here and reject any config that
  // cannot pass the model's allocation/resource policy before constructing a
  // registry or parameter storage. The explicit element budget also keeps a
  // syntactically valid but unreasonably large V4 config from reaching the
  // model registry path.
  if (config.vocabularySize == 0 || config.tokens == 0 ||
      config.dimension == 0 || config.feedForwardDimension == 0 ||
      config.numLayers == 0 || config.numLayers > 999 ||
      config.numHeads == 0 || config.dimension % config.numHeads != 0 ||
      !std::isfinite(config.epsilon) || config.epsilon <= 0.0f)
    return false;

  std::uint64_t total = 0;
  if (!tiny_lm::checkedParameterElementCount(
          {config.vocabularySize, config.dimension,
           config.feedForwardDimension, config.numLayers, config.numHeads,
           config.attentionGate ==
               tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID},
          &total) ||
      total > kMaxParameterElements)
    return false;

  std::string modelError;
  return tiny_lm::validateConfig(config, &modelError);
}

bool finiteValues(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

bool validSha256(const std::string& value) {
  if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0)
    return false;
  return std::all_of(value.begin() + 7, value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool validDataHash(const std::string& value) {
  if (validSha256(value)) return true;
  if (value.size() != 24 || value.compare(0, 8, "fnv1a64:") != 0)
    return false;
  return std::all_of(value.begin() + 8, value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::uint64_t shapeElements(const MatrixShape& shape) {
  return std::uint64_t(shape.rows) * std::uint64_t(shape.columns);
}

bool sameConfig(const tiny_lm::Config& a, const tiny_lm::Config& b) {
  return a.vocabularySize == b.vocabularySize && a.tokens == b.tokens &&
         a.dimension == b.dimension &&
         a.feedForwardDimension == b.feedForwardDimension &&
         a.numLayers == b.numLayers && a.numHeads == b.numHeads &&
         a.epsilon == b.epsilon && a.attentionGate == b.attentionGate;
}

bool modelRegistryMatchesConfig(const tiny_lm::Config& config,
                                const std::vector<RegistryEntry>& expected,
                                std::string* error) {
  try {
    // Use the model's explicit semantic registry as the source of truth.  The
    // local V4 registry is a serialized copy with a compact 2-D shape, so
    // normalize rank-1 vector entries to 1 x D before comparing it.
    const auto shaped = tiny_lm::initialParameters(config, 1);
    const auto model = tiny_lm::parameterRegistry(shaped);
    std::string modelError;
    if (!tiny_lm::validateParameterRegistry(model, &modelError) ||
        model.size() != expected.size())
      return fail(error, "NPRT_CKPT_V4_MODEL_REGISTRY_INVALID");
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const auto& source = model[index];
      MatrixShape shape;
      if (source.shape.size() == 1) {
        shape = {1, source.shape.front()};
      } else if (source.shape.size() == 2) {
        shape = {source.shape[0], source.shape[1]};
      } else {
        return fail(error, "NPRT_CKPT_V4_MODEL_REGISTRY_RANK");
      }
      ParameterRole role;
      if (source.role == tiny_lm::ParameterRole::MUON) {
        role = ParameterRole::MUON;
      } else if (source.role == tiny_lm::ParameterRole::AUX_ADAM) {
        role = ParameterRole::AUX_ADAM;
      } else {
        return fail(error, "NPRT_CKPT_V4_MODEL_REGISTRY_ROLE");
      }
      if (source.name != expected[index].name || role != expected[index].role ||
          shape.rows != expected[index].shape.rows ||
          shape.columns != expected[index].shape.columns || !source.values ||
          source.values->size() != shapeElements(shape))
        return fail(error, "NPRT_CKPT_V4_MODEL_REGISTRY_IDENTITY");
    }
    return true;
  } catch (const std::exception&) {
    return fail(error, "NPRT_CKPT_V4_MODEL_REGISTRY_EXCEPTION");
  }
}

void appendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
}

void appendU64(std::vector<std::uint8_t>& output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
}

void appendFloat(std::vector<std::uint8_t>& output, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  appendU32(output, bits);
}

void appendString(std::vector<std::uint8_t>& output, const std::string& value) {
  appendU32(output, static_cast<std::uint32_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

  std::uint32_t u32() {
    if (remaining() < 4) throw std::runtime_error("NPRT_CKPT_V4_TRUNCATED_U32");
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value = (value << 8) | bytes_[offset_++];
    return value;
  }

  std::uint64_t u64() {
    if (remaining() < 8) throw std::runtime_error("NPRT_CKPT_V4_TRUNCATED_U64");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value = (value << 8) | bytes_[offset_++];
    return value;
  }

  float floating() {
    const std::uint32_t bits = u32();
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  std::string string(std::uint32_t maxBytes = kMaxStringBytes) {
    const std::uint32_t length = u32();
    if (length > maxBytes || remaining() < length)
      throw std::runtime_error("NPRT_CKPT_V4_STRING_INVALID");
    std::string value(reinterpret_cast<const char*>(bytes_.data() + offset_),
                      length);
    offset_ += length;
    return value;
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

  bool atEnd() const { return offset_ == bytes_.size(); }

  bool requireMagic() {
    if (remaining() < kMagicBytes)
      throw std::runtime_error("NPRT_CKPT_V4_MAGIC");
    const bool gated =
        std::memcmp(bytes_.data() + offset_, kGatedMagic, kMagicBytes) == 0;
    if (!gated && std::memcmp(bytes_.data() + offset_, kMagic, kMagicBytes) != 0)
      throw std::runtime_error("NPRT_CKPT_V4_MAGIC");
    offset_ += kMagicBytes;
    return gated;
  }

 private:
  const std::vector<std::uint8_t>& bytes_;
  std::size_t offset_ = 0;
};

void appendValues(std::vector<std::uint8_t>& output,
                  const std::vector<float>& values) {
  appendU64(output, static_cast<std::uint64_t>(values.size()));
  for (float value : values) appendFloat(output, value);
}

std::vector<float> readValues(Reader& reader, std::uint64_t expectedCount) {
  const std::uint64_t count = reader.u64();
  if (count != expectedCount || count > kMaxParameterElements)
    throw std::runtime_error("NPRT_CKPT_V4_STATE_SHAPE");
  if (count > reader.remaining() / sizeof(float))
    throw std::runtime_error("NPRT_CKPT_V4_VALUES_TRUNCATED");
  std::vector<float> values(static_cast<std::size_t>(count));
  for (float& value : values) value = reader.floating();
  return values;
}

bool validHyperparameters(const Hyperparameters& hyperparameters) {
  const auto finite = [](float value) { return std::isfinite(value); };
  if (!finite(hyperparameters.muonLearningRate) ||
      hyperparameters.muonLearningRate <= 0.0f ||
      !finite(hyperparameters.auxAdamLearningRate) ||
      hyperparameters.auxAdamLearningRate <= 0.0f ||
      !finite(hyperparameters.muonTargetLearningRate) ||
      hyperparameters.muonTargetLearningRate < 0.0f ||
      !finite(hyperparameters.auxAdamTargetLearningRate) ||
      hyperparameters.auxAdamTargetLearningRate < 0.0f ||
      !finite(hyperparameters.muonMomentum) ||
      hyperparameters.muonMomentum < 0.0f ||
      hyperparameters.muonMomentum > 1.0f ||
      hyperparameters.muonNsSteps == 0 ||
      hyperparameters.muonNsSteps > 64 ||
      !finite(hyperparameters.auxAdamBeta1) ||
      hyperparameters.auxAdamBeta1 < 0.0f ||
      hyperparameters.auxAdamBeta1 >= 1.0f ||
      !finite(hyperparameters.auxAdamBeta2) ||
      hyperparameters.auxAdamBeta2 < 0.0f ||
      hyperparameters.auxAdamBeta2 >= 1.0f ||
      !finite(hyperparameters.auxAdamEpsilon) ||
      hyperparameters.auxAdamEpsilon <= 0.0f ||
      !finite(hyperparameters.muonWeightDecay) ||
      hyperparameters.muonWeightDecay < 0.0f ||
      !finite(hyperparameters.auxAdamWeightDecay) ||
      hyperparameters.auxAdamWeightDecay < 0.0f ||
      // V4 is the frozen Muon v1 pilot format. Do not allow a checkpoint to
      // silently select a different optimizer variant on resume.
      hyperparameters.muonMomentum != 0.95f ||
      !hyperparameters.muonNesterov || hyperparameters.muonNsSteps != 5 ||
      hyperparameters.auxAdamBeta1 != 0.9f ||
      hyperparameters.auxAdamBeta2 != 0.999f ||
      hyperparameters.auxAdamEpsilon != 1.0e-8f ||
      hyperparameters.muonWeightDecay != 0.0f ||
      hyperparameters.auxAdamWeightDecay != 0.0f)
    return false;
  if (hyperparameters.scheduleTotalSteps == 0)
    return hyperparameters.decayStartStep == 0 &&
           hyperparameters.decayEndStep == 0;
  return hyperparameters.decayStartStep > 0 &&
         hyperparameters.decayStartStep < hyperparameters.decayEndStep &&
         hyperparameters.decayEndStep <= hyperparameters.scheduleTotalSteps;
}

}  // namespace

std::vector<RegistryEntry> expectedRegistry(const tiny_lm::Config& config) {
  std::vector<RegistryEntry> registry;
  if (config.numLayers == 0 || config.numLayers > 999) return registry;
  const auto metadata = tiny_lm::parameterMetadata(
      {config.vocabularySize, config.dimension, config.feedForwardDimension,
       config.numLayers, config.numHeads,
       config.attentionGate == tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID});
  registry.reserve(metadata.size());
  for (const auto& source : metadata) {
    ParameterRole role;
    if (source.role == tiny_lm::ParameterRole::MUON) {
      role = ParameterRole::MUON;
    } else if (source.role == tiny_lm::ParameterRole::AUX_ADAM) {
      role = ParameterRole::AUX_ADAM;
    } else {
      return {};
    }
    if (source.shape.empty() || source.shape.size() > 2) return {};
    const MatrixShape shape =
        source.shape.size() == 1
            ? MatrixShape{1, source.shape[0]}
            : MatrixShape{source.shape[0], source.shape[1]};
    registry.push_back({source.name, role, shape});
  }
  return registry;
}

const char* parameterRoleName(ParameterRole role) {
  if (role == ParameterRole::MUON) return "MUON";
  if (role == ParameterRole::AUX_ADAM) return "AUX_ADAM";
  return "UNKNOWN";
}

bool classifyParameter(const tiny_lm::Config& config, const std::string& name,
                       ParameterRole* role, MatrixShape* shape,
                       std::string* error) {
  if (!role || !shape) return fail(error, "NPRT_CKPT_V4_CLASSIFY_OUTPUT_NULL");
  for (const auto& entry : expectedRegistry(config)) {
    if (entry.name == name) {
      *role = entry.role;
      *shape = entry.shape;
      return true;
    }
  }
  return fail(error, "NPRT_CKPT_V4_PARAMETER_UNCLASSIFIED");
}

bool validateCheckpoint(const Checkpoint& checkpoint, std::string* error) {
  if (checkpoint.optimizerIdentity != kOptimizerIdentity)
    return fail(error, "NPRT_CKPT_V4_OPTIMIZER_IDENTITY");
  if (!validConfig(checkpoint.identity.config))
    return fail(error, "NPRT_CKPT_V4_CONFIG_IDENTITY");
  if (checkpoint.identity.seed == 0 || checkpoint.identity.globalStep == 0)
    return fail(error, "NPRT_CKPT_V4_STEP_OR_SEED_INVALID");
  if (checkpoint.identity.tokenizerKind != "byte_bpe" ||
      !validSha256(checkpoint.identity.tokenizerHash))
    return fail(error, "NPRT_CKPT_V4_TOKENIZER_IDENTITY");
  if (!validDataHash(checkpoint.identity.dataCursor.datasetHash))
    return fail(error, "NPRT_CKPT_V4_DATASET_IDENTITY");
  if (!validHyperparameters(checkpoint.hyperparameters))
    return fail(error, "NPRT_CKPT_V4_HYPERPARAMETERS");

  const auto registry = expectedRegistry(checkpoint.identity.config);
  if (registry.empty() || registry.size() > kMaxRegistryEntries ||
      checkpoint.parameters.size() != registry.size())
    return fail(error, "NPRT_CKPT_V4_REGISTRY_COUNT");
  if (!modelRegistryMatchesConfig(checkpoint.identity.config, registry, error))
    return false;
  std::uint64_t totalElements = 0;
  for (std::size_t index = 0; index < registry.size(); ++index) {
    const auto& expected = registry[index];
    const auto& actual = checkpoint.parameters[index];
    const std::uint64_t expectedElements = shapeElements(expected.shape);
    if (expectedElements == 0 || expectedElements > kMaxParameterElements ||
        actual.name != expected.name || actual.role != expected.role ||
        actual.shape.rows != expected.shape.rows ||
        actual.shape.columns != expected.shape.columns ||
        actual.values.size() != expectedElements ||
        !finiteValues(actual.values))
      return fail(error, "NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY");
    if (totalElements > kMaxParameterElements - expectedElements)
      return fail(error, "NPRT_CKPT_V4_PARAMETER_ELEMENTS_LIMIT");
    totalElements += expectedElements;
    if (actual.role == ParameterRole::MUON) {
      if (actual.momentum.size() != expectedElements ||
          !actual.adamM.empty() || !actual.adamV.empty() ||
          !finiteValues(actual.momentum))
        return fail(error, "NPRT_CKPT_V4_MUON_STATE_IDENTITY");
    } else if (actual.role == ParameterRole::AUX_ADAM) {
      if (!actual.momentum.empty() || actual.adamM.size() != expectedElements ||
          actual.adamV.size() != expectedElements ||
          !finiteValues(actual.adamM) || !finiteValues(actual.adamV))
        return fail(error, "NPRT_CKPT_V4_AUX_ADAM_STATE_IDENTITY");
    } else {
      return fail(error, "NPRT_CKPT_V4_PARAMETER_ROLE");
    }
  }
  return true;
}

bool encodeCheckpoint(const Checkpoint& checkpoint,
                      std::vector<std::uint8_t>* bytes,
                      std::string* error) {
  if (!bytes) return fail(error, "NPRT_CKPT_V4_ENCODE_OUTPUT_NULL");
  if (!validateCheckpoint(checkpoint, error)) return false;

  std::vector<std::uint8_t> encoded;
  const bool gated = checkpoint.identity.config.attentionGate ==
      tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
  const char* magic = gated ? kGatedMagic : kMagic;
  encoded.insert(encoded.end(), magic, magic + kMagicBytes);
  const auto& identity = checkpoint.identity;
  const auto& config = identity.config;
  appendU32(encoded, config.vocabularySize);
  appendU32(encoded, config.tokens);
  appendU32(encoded, config.dimension);
  appendU32(encoded, config.feedForwardDimension);
  appendU32(encoded, config.numLayers);
  appendU32(encoded, config.numHeads);
  appendFloat(encoded, config.epsilon);
  if (gated) appendU32(encoded, static_cast<std::uint32_t>(config.attentionGate));
  appendU32(encoded, identity.seed);
  appendU64(encoded, identity.globalStep);
  appendString(encoded, identity.tokenizerKind);
  appendString(encoded, identity.tokenizerHash);
  appendString(encoded, identity.dataCursor.datasetHash);
  appendU64(encoded, identity.dataCursor.recordIndex);
  appendU64(encoded, identity.dataCursor.tokenOffset);
  appendU64(encoded, identity.dataCursor.epoch);
  appendU64(encoded, identity.dataCursor.exposedTokens);
  appendU64(encoded, identity.dataCursor.orderSeed);

  appendString(encoded, checkpoint.optimizerIdentity);
  const auto& hp = checkpoint.hyperparameters;
  appendFloat(encoded, hp.muonLearningRate);
  appendFloat(encoded, hp.auxAdamLearningRate);
  appendFloat(encoded, hp.muonTargetLearningRate);
  appendFloat(encoded, hp.auxAdamTargetLearningRate);
  appendFloat(encoded, hp.muonMomentum);
  appendU32(encoded, hp.muonNesterov ? 1u : 0u);
  appendU32(encoded, hp.muonNsSteps);
  appendFloat(encoded, hp.auxAdamBeta1);
  appendFloat(encoded, hp.auxAdamBeta2);
  appendFloat(encoded, hp.auxAdamEpsilon);
  appendFloat(encoded, hp.muonWeightDecay);
  appendFloat(encoded, hp.auxAdamWeightDecay);
  appendU32(encoded, hp.decayStartStep);
  appendU32(encoded, hp.decayEndStep);
  appendU32(encoded, hp.scheduleTotalSteps);
  appendU32(encoded, gated ? kGatedSchemaVersion : kSchemaVersion);
  appendU32(encoded, gated ? kGatedParameterRegistryVersion
                           : kParameterRegistryVersion);
  appendU32(encoded, static_cast<std::uint32_t>(checkpoint.parameters.size()));

  for (const auto& parameter : checkpoint.parameters) {
    appendString(encoded, parameter.name);
    appendU32(encoded, static_cast<std::uint32_t>(parameter.role));
    appendU32(encoded, parameter.shape.rows);
    appendU32(encoded, parameter.shape.columns);
    appendValues(encoded, parameter.values);
    if (parameter.role == ParameterRole::MUON) {
      appendValues(encoded, parameter.momentum);
    } else {
      appendValues(encoded, parameter.adamM);
      appendValues(encoded, parameter.adamV);
    }
    if (encoded.size() > kMaxCheckpointBytes)
      return fail(error, "NPRT_CKPT_V4_SIZE_LIMIT");
  }
  *bytes = std::move(encoded);
  return true;
}

bool decodeCheckpoint(const std::vector<std::uint8_t>& bytes,
                      Checkpoint* checkpoint, std::string* error,
                      const std::string& expectedOptimizerIdentity) {
  if (!checkpoint) return fail(error, "NPRT_CKPT_V4_DECODE_OUTPUT_NULL");
  if (bytes.empty() || bytes.size() > kMaxCheckpointBytes)
    return fail(error, "NPRT_CKPT_V4_SIZE_INVALID");
  try {
    Reader reader(bytes);
    const bool gated = reader.requireMagic();
    Checkpoint decoded;
    auto& config = decoded.identity.config;
    config.vocabularySize = reader.u32();
    config.tokens = reader.u32();
    config.dimension = reader.u32();
    config.feedForwardDimension = reader.u32();
    config.numLayers = reader.u32();
    config.numHeads = reader.u32();
    config.epsilon = reader.floating();
    if (gated) {
      const auto gate = reader.u32();
      if (gate != static_cast<std::uint32_t>(
                      tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID))
        throw std::runtime_error("NPRT_CKPT_V5_ATTENTION_GATE");
      config.attentionGate = tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
    }
    // Do not derive a registry (or reserve parameter/state storage) until all
    // serialized dimensions have passed the same allocation policy as the
    // live model. This is intentionally before reading/allocating any state.
    if (!validConfig(config))
      throw std::runtime_error("NPRT_CKPT_V4_CONFIG_IDENTITY");
    decoded.identity.seed = reader.u32();
    decoded.identity.globalStep = reader.u64();
    decoded.identity.tokenizerKind = reader.string();
    decoded.identity.tokenizerHash = reader.string();
    decoded.identity.dataCursor.datasetHash = reader.string();
    decoded.identity.dataCursor.recordIndex = reader.u64();
    decoded.identity.dataCursor.tokenOffset = reader.u64();
    decoded.identity.dataCursor.epoch = reader.u64();
    decoded.identity.dataCursor.exposedTokens = reader.u64();
    decoded.identity.dataCursor.orderSeed = reader.u64();

    decoded.optimizerIdentity = reader.string();
    auto& hp = decoded.hyperparameters;
    hp.muonLearningRate = reader.floating();
    hp.auxAdamLearningRate = reader.floating();
    hp.muonTargetLearningRate = reader.floating();
    hp.auxAdamTargetLearningRate = reader.floating();
    hp.muonMomentum = reader.floating();
    const std::uint32_t nesterov = reader.u32();
    if (nesterov > 1) throw std::runtime_error("NPRT_CKPT_V4_BOOLEAN");
    hp.muonNesterov = nesterov != 0;
    hp.muonNsSteps = reader.u32();
    hp.auxAdamBeta1 = reader.floating();
    hp.auxAdamBeta2 = reader.floating();
    hp.auxAdamEpsilon = reader.floating();
    hp.muonWeightDecay = reader.floating();
    hp.auxAdamWeightDecay = reader.floating();
    hp.decayStartStep = reader.u32();
    hp.decayEndStep = reader.u32();
    hp.scheduleTotalSteps = reader.u32();
    if (reader.u32() != (gated ? kGatedSchemaVersion : kSchemaVersion))
      throw std::runtime_error("NPRT_CKPT_V4_SCHEMA_VERSION");
    if (reader.u32() != (gated ? kGatedParameterRegistryVersion
                               : kParameterRegistryVersion))
      throw std::runtime_error("NPRT_CKPT_V4_REGISTRY_VERSION");
    const std::uint32_t parameterCount = reader.u32();
    const auto registry = expectedRegistry(config);
    if (parameterCount > kMaxRegistryEntries || registry.empty() ||
        parameterCount != registry.size())
      throw std::runtime_error("NPRT_CKPT_V4_REGISTRY_COUNT");

    // A valid config still needs enough bytes for every value and optimizer
    // state array. Reject impossible state budgets before reserve() and,
    // importantly, before readValues() can allocate a large vector from a
    // hostile count. The serialized float payload is a strict lower bound on
    // the required byte count; string/count overhead only makes it larger.
    std::uint64_t stateElements = 0;
    for (const auto& entry : registry) {
      const std::uint64_t elements = shapeElements(entry.shape);
      const std::uint64_t arrays =
          entry.role == ParameterRole::MUON ? 2ull : 3ull;
      if (elements == 0 || elements > kMaxParameterElements / arrays ||
          stateElements > kMaxParameterElements - elements * arrays)
        throw std::runtime_error("NPRT_CKPT_V4_STATE_BUDGET");
      stateElements += elements * arrays;
    }
    if (stateElements > bytes.size() / sizeof(float))
      throw std::runtime_error("NPRT_CKPT_V4_STATE_BUDGET");
    decoded.parameters.reserve(parameterCount);
    std::uint64_t totalElements = 0;
    for (std::uint32_t index = 0; index < parameterCount; ++index) {
      ParameterState parameter;
      parameter.name = reader.string();
      const std::uint32_t role = reader.u32();
      if (role != static_cast<std::uint32_t>(ParameterRole::MUON) &&
          role != static_cast<std::uint32_t>(ParameterRole::AUX_ADAM))
        throw std::runtime_error("NPRT_CKPT_V4_PARAMETER_ROLE");
      parameter.role = static_cast<ParameterRole>(role);
      parameter.shape.rows = reader.u32();
      parameter.shape.columns = reader.u32();
      const auto& expected = registry[index];
      const std::uint64_t elements = shapeElements(expected.shape);
      if (parameter.name != expected.name || parameter.role != expected.role ||
          parameter.shape.rows != expected.shape.rows ||
          parameter.shape.columns != expected.shape.columns || elements == 0 ||
          elements > kMaxParameterElements ||
          totalElements > kMaxParameterElements - elements)
        throw std::runtime_error("NPRT_CKPT_V4_PARAMETER_REGISTRY_IDENTITY");
      totalElements += elements;
      parameter.values = readValues(reader, elements);
      if (parameter.role == ParameterRole::MUON) {
        parameter.momentum = readValues(reader, elements);
      } else {
        parameter.adamM = readValues(reader, elements);
        parameter.adamV = readValues(reader, elements);
      }
      decoded.parameters.push_back(std::move(parameter));
    }
    if (!reader.atEnd()) throw std::runtime_error("NPRT_CKPT_V4_TRAILING_BYTES");
    if (decoded.optimizerIdentity != expectedOptimizerIdentity)
      throw std::runtime_error("NPRT_CKPT_V4_OPTIMIZER_MISMATCH");
    if (!validateCheckpoint(decoded, error)) return false;
    *checkpoint = std::move(decoded);
    return true;
  } catch (const std::exception& exception) {
    return fail(error, exception.what());
  }
}

bool extractParameters(const Checkpoint& checkpoint,
                       const tiny_lm::Config& expectedConfig,
                       std::uint32_t expectedSeed,
                       qnn::TinyTransformerParameters* parameters,
                       std::string* error) {
  if (!parameters) return fail(error, "NPRT_CKPT_V4_EXTRACT_OUTPUT_NULL");
  if (!validateCheckpoint(checkpoint, error)) return false;
  if (!sameConfig(checkpoint.identity.config, expectedConfig))
    return fail(error, "NPRT_CKPT_V4_CONFIG_MISMATCH");
  if (checkpoint.identity.seed != expectedSeed)
    return fail(error, "NPRT_CKPT_V4_SEED_MISMATCH");
  try {
    qnn::TinyTransformerParameters extracted =
        tiny_lm::initialParameters(expectedConfig, expectedSeed);
    const auto destination = tiny_lm::parameterRegistry(extracted);
    const auto registry = expectedRegistry(expectedConfig);
    if (destination.size() != registry.size())
      return fail(error, "NPRT_CKPT_V4_EXTRACT_REGISTRY_COUNT");
    for (std::size_t index = 0; index < registry.size(); ++index) {
      if (destination[index].name != checkpoint.parameters[index].name)
        return fail(error, "NPRT_CKPT_V4_EXTRACT_REGISTRY_ORDER");
      *const_cast<std::vector<float>*>(destination[index].values) =
          checkpoint.parameters[index].values;
    }
    *parameters = std::move(extracted);
    return true;
  } catch (const std::exception& exception) {
    return fail(error, exception.what());
  }
}

}  // namespace phonelm::nicopedia_muon_checkpoint
