// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once
//
// Shared, dependency-light loader for NPRTCKPTV1/V2/V3 private generation
// checkpoints.  Extracted from qnn_transformer_training.cpp so both the cold
// generation path and the process-local prepared-generation engine load and
// validate checkpoints through one implementation (fail-closed identity,
// registry order, finiteness, parameter hash).
//
// The header has no QNN/Android dependencies.

#include "../tiny_language_model_cpu.h"
#include "../nicopedia_checkpoint_policy.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phonelm::qnn {

// Checkpoint filename identity check (seed/layers/step + T/D/FFN tags).
inline bool nprtParseCheckpointStep(
    const std::string &path, uint32_t expectedSeed, uint32_t expectedLayers,
    uint32_t *step, uint32_t expectedTokens = 32,
    uint32_t expectedDimension = 32,
    uint32_t expectedFeedForwardDimension = 32) {
  return ::phonelm::nicopedia_checkpoint::parseCheckpointStep(
      path, expectedSeed, expectedLayers, step, expectedTokens,
      expectedDimension, expectedFeedForwardDimension);
}

inline constexpr std::uint64_t kNprtLoaderMaxCheckpointBytes =
    64u * 1024u * 1024u;

// FNV-1a 64-bit offset basis 0xCBF29CE484222325 = 14695981039346656037.
inline constexpr std::uint64_t kNprtFnvOffset = 14695981039346656037ull;
inline constexpr std::uint64_t kNprtFnvPrime = 1099511628211ull;

inline std::uint64_t nprtFnvBytes(const void *data, std::size_t bytes,
                                  std::uint64_t hash = kNprtFnvOffset) {
  const auto *input = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= input[i];
    hash *= kNprtFnvPrime;
  }
  return hash;
}

inline std::string nprtHex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << value;
  return stream.str();
}

inline uint32_t nprtReadU32(std::istream &input) {
  std::array<std::uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("NPRT_CACHE_TRUNCATED_U32");
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

inline std::uint64_t nprtReadU64(std::istream &input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("NPRT_CACHE_TRUNCATED_U64");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes) value = (value << 8) | byte;
  return value;
}

inline std::vector<std::uint8_t> nprtReadFileBytes(const std::string &path,
                                                   std::uint64_t maxBytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("FILE_OPEN_FAILED");
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0 || static_cast<std::uint64_t>(size) > maxBytes)
    throw std::runtime_error("FILE_SIZE_INVALID");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (size > 0)
    input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input && size > 0) throw std::runtime_error("FILE_READ_TRUNCATED");
  return bytes;
}

inline void nprtAssignRegistryMember(TinyTransformerParameters &target,
                                     uint32_t layers, const std::string &name,
                                     std::vector<float> &&values) {
  using Params = TinyTransformerParameters;
  using Layer = TinyTransformerLayerParameters;
  if (name == "token_embedding") {
    target.tokenEmbedding = std::move(values);
    return;
  }
  if (name == "output_projection") {
    target.outputProjection = std::move(values);
    return;
  }
  if (name.rfind("layer_", 0) != 0)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const size_t dot = name.find('.');
  if (dot == std::string::npos || dot < 7)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const std::string indexText = name.substr(6, dot - 6);
  if (indexText.empty() || indexText.size() > 3)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  char *end = nullptr;
  const long index = std::strtol(indexText.c_str(), &end, 10);
  if (!end || *end != '\0' || index < 0 ||
      static_cast<uint32_t>(index) >= layers)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
  const std::string suffix = name.substr(dot + 1);
  Layer *layer =
      index == 0 ? static_cast<Layer *>(&target) : &target.layers[static_cast<std::size_t>(index) - 1];
  if (suffix == "norm1_gamma") layer->gamma1 = std::move(values);
  else if (suffix == "norm1_beta") layer->beta1 = std::move(values);
  else if (suffix == "wq") layer->wq = std::move(values);
  else if (suffix == "wk") layer->wk = std::move(values);
  else if (suffix == "wv") layer->wv = std::move(values);
  else if (suffix == "wo") layer->wo = std::move(values);
  else if (suffix == "norm2_gamma") layer->gamma2 = std::move(values);
  else if (suffix == "norm2_beta") layer->beta2 = std::move(values);
  else if (suffix == "ffn_w1") layer->w1 = std::move(values);
  else if (suffix == "ffn_w2") layer->w2 = std::move(values);
  else throw std::runtime_error("NPRT_CKPT_REGISTRY_NAME");
}

struct LoadedNprtCheckpoint {
  uint32_t vocabulary = 0, tokens = 0, dimension = 0, feedForward = 0;
  uint32_t layers = 0, heads = 0, seed = 0, step = 0;
  uint32_t registryCount = 0;
  std::uint64_t fileBytes = 0;
  std::uint64_t parameterElements = 0;
  bool finite = true;
  bool hasAdam = false;
  bool v2 = false;
  bool v3 = false;
  std::string tokenizerKind;
  std::string tokenizerHash;
  TinyTransformerParameters parameters;
  std::vector<std::pair<std::string, std::vector<float>>> adamFirst;
  std::vector<std::pair<std::string, std::vector<float>>> adamSecond;
  std::string parameterHash;
};

inline std::string nprtParameterHash(
    const TinyTransformerParameters &parameters) {
  std::uint64_t hash = kNprtFnvOffset;
  for (const auto &info : tiny_lm::parameterRegistry(parameters)) {
    hash = nprtFnvBytes(info.name.data(), info.name.size(), hash);
    const std::uint64_t count = info.values->size();
    hash = nprtFnvBytes(&count, sizeof(count), hash);
    hash = nprtFnvBytes(info.values->data(),
                        info.values->size() * sizeof(float), hash);
  }
  return nprtHex64(hash);
}

// Loads and validates a generation checkpoint.  Throws std::runtime_error on
// any identity/format/finiteness failure; callers must not weaken the checks.
inline LoadedNprtCheckpoint nprtLoadCheckpointForGeneration(
    const std::string &path, const tiny_lm::Config &expected,
    uint32_t expectedSeed, const std::string &expectedTokenizerHash = {}) {
  LoadedNprtCheckpoint result;
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("NPRT_CKPT_OPEN_FAILED");
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0 ||
      static_cast<std::uint64_t>(size) > kNprtLoaderMaxCheckpointBytes)
    throw std::runtime_error("NPRT_CKPT_SIZE_INVALID");
  result.fileBytes = static_cast<std::uint64_t>(size);
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTCKPTV1\n" && magic != "NPRTCKPTV2\n" &&
      magic != "NPRTCKPTV3\n")
    throw std::runtime_error("NPRT_CKPT_MAGIC");
  const bool v2 = magic == "NPRTCKPTV2\n";
  const bool v3 = magic == "NPRTCKPTV3\n";
  result.v2 = v2;
  result.v3 = v3;
  result.vocabulary = nprtReadU32(input);
  result.tokens = nprtReadU32(input);
  result.dimension = nprtReadU32(input);
  result.feedForward = nprtReadU32(input);
  result.layers = nprtReadU32(input);
  result.heads = nprtReadU32(input);
  result.seed = nprtReadU32(input);
  result.step = nprtReadU32(input);
  if (v3) {
    const uint32_t kindLength = nprtReadU32(input);
    if (kindLength == 0 || kindLength > 32)
      throw std::runtime_error("NPRT_CKPT_TOKENIZER_KIND_LENGTH");
    result.tokenizerKind.resize(kindLength);
    input.read(result.tokenizerKind.data(),
               static_cast<std::streamsize>(kindLength));
    const uint32_t hashLength = nprtReadU32(input);
    if (hashLength != 71)
      throw std::runtime_error("NPRT_CKPT_TOKENIZER_HASH_LENGTH");
    result.tokenizerHash.resize(hashLength);
    input.read(result.tokenizerHash.data(),
               static_cast<std::streamsize>(hashLength));
    if (!input || result.tokenizerKind != "byte_bpe" ||
        result.tokenizerHash.compare(0, 7, "sha256:") != 0)
      throw std::runtime_error("NPRT_CKPT_TOKENIZER_IDENTITY");
  }
  result.registryCount = nprtReadU32(input);
  if (result.vocabulary != expected.vocabularySize ||
      result.tokens != expected.tokens ||
      result.dimension != expected.dimension ||
      result.feedForward != expected.feedForwardDimension ||
      result.layers != expected.numLayers ||
      result.heads != expected.numHeads)
    throw std::runtime_error("NPRT_CKPT_CONFIG_MISMATCH");
  if (result.seed != expectedSeed)
    throw std::runtime_error("NPRT_CKPT_SEED_MISMATCH");
  constexpr std::uint32_t kBpeVocabulary = 1024;
  if (expected.vocabularySize == kBpeVocabulary) {
    if (!v3) throw std::runtime_error("NPRT_CKPT_V3_REQUIRED_FOR_BPE");
    if (expectedTokenizerHash.empty() ||
        result.tokenizerHash != expectedTokenizerHash)
      throw std::runtime_error("NPRT_CKPT_TOKENIZER_HASH_MISMATCH");
  } else if (v3) {
    throw std::runtime_error("NPRT_CKPT_V3_REQUIRES_BPE_MODEL");
  }
  if (result.step == 0 || result.step >= 1000000)
    throw std::runtime_error("NPRT_CKPT_STEP_INVALID");
  // The trained checkpoint stores registry counts derived from fully-shaped
  // parameters (vocab*dimension embedding, dimension^2 projections, ...).
  // Reference only the shapes (values are discarded) so element counts match
  // whatever the training milestone wrote.
  const auto shape = tiny_lm::initialParameters(expected, 1);
  const auto registry = tiny_lm::parameterRegistry(shape);
  if (registry.size() != result.registryCount)
    throw std::runtime_error("NPRT_CKPT_REGISTRY_COUNT_MISMATCH");
  TinyTransformerParameters parameters;
  parameters.layers.resize(result.layers > 0 ? result.layers - 1 : 0);
  std::uint64_t elements = 0;
  for (size_t i = 0; i < registry.size(); ++i) {
    const uint32_t nameLength = nprtReadU32(input);
    if (nameLength == 0 || nameLength > 256)
      throw std::runtime_error("NPRT_CKPT_NAME_LENGTH");
    std::string name(nameLength, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name.size()));
    if (!input) throw std::runtime_error("NPRT_CKPT_NAME_TRUNCATED");
    const std::uint64_t count = nprtReadU64(input);
    if (name != registry[i].name)
      throw std::runtime_error("NPRT_CKPT_REGISTRY_ORDER_MISMATCH");
    const std::uint64_t expectedCount = registry[i].values->size();
    if (count != expectedCount || count > 100000000ull)
      throw std::runtime_error("NPRT_CKPT_ELEMENT_COUNT_MISMATCH");
    std::vector<float> values(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (!input) throw std::runtime_error("NPRT_CKPT_VALUES_TRUNCATED");
    for (float value : values)
      if (!std::isfinite(value)) result.finite = false;
    nprtAssignRegistryMember(parameters, result.layers, name,
                             std::move(values));
    elements += count;
  }
  if (v2 || v3) {
    const auto readMomentRegistry =
        [&](std::vector<std::pair<std::string, std::vector<float>>> &target) {
          const uint32_t count = nprtReadU32(input);
          if (count != registry.size())
            throw std::runtime_error("NPRT_CKPT_ADAM_REGISTRY_COUNT");
          target.reserve(count);
          for (size_t i = 0; i < count; ++i) {
            const uint32_t nameLength = nprtReadU32(input);
            if (nameLength == 0 || nameLength > 256)
              throw std::runtime_error("NPRT_CKPT_NAME_LENGTH");
            std::string name(nameLength, '\0');
            input.read(name.data(), static_cast<std::streamsize>(name.size()));
            if (!input) throw std::runtime_error("NPRT_CKPT_NAME_TRUNCATED");
            const std::uint64_t countValues = nprtReadU64(input);
            if (name != registry[i].name ||
                countValues != registry[i].values->size() ||
                countValues > 100000000ull)
              throw std::runtime_error("NPRT_CKPT_ADAM_REGISTRY_IDENTITY");
            std::vector<float> values(static_cast<std::size_t>(countValues));
            input.read(reinterpret_cast<char *>(values.data()),
                       static_cast<std::streamsize>(values.size() *
                                                    sizeof(float)));
            if (!input) throw std::runtime_error("NPRT_CKPT_VALUES_TRUNCATED");
            for (float value : values)
              if (!std::isfinite(value)) result.finite = false;
            target.emplace_back(std::move(name), std::move(values));
          }
        };
    result.hasAdam = true;
    readMomentRegistry(result.adamFirst);
    readMomentRegistry(result.adamSecond);
  }
  char tail = 0;
  if (input.read(&tail, 1)) throw std::runtime_error("NPRT_CKPT_TRAILING_BYTES");
  result.parameterElements = elements;
  result.parameters = std::move(parameters);
  result.parameterHash = nprtParameterHash(result.parameters);
  return result;
}

}  // namespace phonelm::qnn
