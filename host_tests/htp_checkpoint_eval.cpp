// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host-only evaluation helper for the private Nicopedia HTP checkpoints.
// Loads an NPRTCKPTV1 checkpoint (as written by the device NICOPEDIA mode)
// and evaluates it on the fixed validation/development caches with the same
// CPU reference used by the pilot.  Outputs aggregate NLL only; the private
// checkpoint itself never enters the repository.
#include "tiny_language_model_cpu.h"
#include "nicopedia_byte_bpe.h"
#include "nicopedia_muon_checkpoint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// FNV-1a 64-bit offset basis 0xCBF29CE484222325 = 14695981039346656037.
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t fnvBytes(const void* data, std::size_t bytes, std::uint64_t hash = kFnvOffset) {
  const auto* input = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= input[i];
    hash *= kFnvPrime;
  }
  return hash;
}

std::uint32_t readU32(std::istream& input) {
  std::array<std::uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("TRUNCATED_U32");
  return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
         (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

std::uint64_t readU64(std::istream& input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("TRUNCATED_U64");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes) value = (value << 8) | byte;
  return value;
}

struct CacheRecord {
  std::uint64_t articleHash = 0;
  std::vector<std::uint16_t> window;
};

struct Cache {
  std::uint32_t context = 0;
  std::uint32_t vocabulary = 0;
  std::vector<CacheRecord> records;
  std::string contentHash;
};

Cache loadCache(const std::string& path,
                const phonelm::nicopedia_bpe::Model* bpeModel = nullptr) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("CACHE_OPEN_FAILED");
  std::string prefix(10, '\0');
  input.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  if (prefix == "NPRTBPEV1\n") {
    if (!bpeModel) throw std::runtime_error("BPE_MODEL_REQUIRED");
    input.close();
    const auto source = phonelm::nicopedia_bpe::loadCache(path, *bpeModel);
    Cache cache;
    cache.context = source.context; cache.vocabulary = source.vocabulary;
    for (const auto& item : source.records)
      cache.records.push_back({item.articleHash, item.window});
    cache.contentHash = source.tokenizerHash;
    return cache;
  }
  char last = 0; input.get(last);
  if (prefix + last != "NPRTBYTEV1\n") throw std::runtime_error("CACHE_MAGIC_MISMATCH");
  Cache cache;
  cache.context = readU32(input);
  cache.vocabulary = readU32(input);
  const std::uint64_t count = readU64(input);
  if (cache.context < 8 || cache.context > 256 || cache.vocabulary != 256 || count > 10000000)
    throw std::runtime_error("CACHE_HEADER_INVALID");
  cache.records.reserve(static_cast<std::size_t>(count));
  std::uint64_t hash = kFnvOffset;
  hash = fnvBytes(&cache.context, sizeof(cache.context), hash);
  hash = fnvBytes(&cache.vocabulary, sizeof(cache.vocabulary), hash);
  hash = fnvBytes(&count, sizeof(count), hash);
  for (std::uint64_t i = 0; i < count; ++i) {
    CacheRecord record;
    record.articleHash = readU64(input);
    std::vector<std::uint8_t> byteWindow(cache.context + 1);
    input.read(reinterpret_cast<char*>(byteWindow.data()), static_cast<std::streamsize>(byteWindow.size()));
    if (!input) throw std::runtime_error("CACHE_RECORD_TRUNCATED");
    record.window.assign(byteWindow.begin(), byteWindow.end());
    hash = fnvBytes(&record.articleHash, sizeof(record.articleHash), hash);
    hash = fnvBytes(byteWindow.data(), byteWindow.size(), hash);
    cache.records.push_back(std::move(record));
  }
  if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("CACHE_TRAILING_BYTES");
  std::ostringstream text;
  text << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  cache.contentHash = text.str();
  return cache;
}

struct Checkpoint {
  phonelm::tiny_lm::Config config;
  std::uint32_t seed = 0;
  std::uint32_t step = 0;
  bool v3 = false;
  std::string tokenizerKind;
  std::string tokenizerHash;
  phonelm::qnn::TinyTransformerParameters parameters;
};

Checkpoint loadCheckpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("CHECKPOINT_OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic == "NPRTCKPTV4\n" || magic == "NPRTCKPTV5\n") {
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    phonelm::nicopedia_muon_checkpoint::Checkpoint mixed;
    std::string error;
    if (!phonelm::nicopedia_muon_checkpoint::decodeCheckpoint(bytes, &mixed,
                                                               &error))
      throw std::runtime_error(error);
    Checkpoint checkpoint;
    checkpoint.config = mixed.identity.config;
    checkpoint.seed = mixed.identity.seed;
    checkpoint.step = static_cast<std::uint32_t>(mixed.identity.globalStep);
    checkpoint.tokenizerKind = mixed.identity.tokenizerKind;
    checkpoint.tokenizerHash = mixed.identity.tokenizerHash;
    if (!phonelm::nicopedia_muon_checkpoint::extractParameters(
            mixed, checkpoint.config, checkpoint.seed, &checkpoint.parameters,
            &error))
      throw std::runtime_error(error);
    return checkpoint;
  }
  if (magic != "NPRTCKPTV1\n" && magic != "NPRTCKPTV2\n" && magic != "NPRTCKPTV3\n")
    throw std::runtime_error("CHECKPOINT_MAGIC");
  const bool v2 = magic == "NPRTCKPTV2\n";
  const bool v3 = magic == "NPRTCKPTV3\n";
  Checkpoint checkpoint;
  checkpoint.v3 = v3;
  checkpoint.config.vocabularySize = readU32(input);
  checkpoint.config.tokens = readU32(input);
  checkpoint.config.dimension = readU32(input);
  checkpoint.config.feedForwardDimension = readU32(input);
  checkpoint.config.numLayers = readU32(input);
  checkpoint.config.numHeads = readU32(input);
  checkpoint.seed = readU32(input);
  checkpoint.step = readU32(input);
  if (v3) {
    const std::uint32_t kindLength = readU32(input);
    if (kindLength == 0 || kindLength > 32) throw std::runtime_error("CHECKPOINT_TOKENIZER_KIND_LENGTH");
    checkpoint.tokenizerKind.resize(kindLength);
    input.read(checkpoint.tokenizerKind.data(), static_cast<std::streamsize>(kindLength));
    const std::uint32_t hashLength = readU32(input);
    if (hashLength != 71) throw std::runtime_error("CHECKPOINT_TOKENIZER_HASH_LENGTH");
    checkpoint.tokenizerHash.resize(hashLength);
    input.read(checkpoint.tokenizerHash.data(), static_cast<std::streamsize>(hashLength));
    if (!input || checkpoint.tokenizerKind != "byte_bpe" ||
        checkpoint.tokenizerHash.compare(0, 7, "sha256:") != 0)
      throw std::runtime_error("CHECKPOINT_TOKENIZER_IDENTITY");
  }
  if (checkpoint.config.vocabularySize == 1024 && !v3)
    throw std::runtime_error("CHECKPOINT_V3_REQUIRED_FOR_BPE");
  if (checkpoint.config.vocabularySize != 1024 && v3)
    throw std::runtime_error("CHECKPOINT_V3_REQUIRES_BPE");
  checkpoint.parameters = phonelm::tiny_lm::initialParameters(checkpoint.config, checkpoint.seed);
  auto registry = phonelm::tiny_lm::parameterRegistry(checkpoint.parameters);
  const std::uint32_t count = readU32(input);
  if (count != registry.size()) throw std::runtime_error("CHECKPOINT_REGISTRY_COUNT");
  for (std::uint32_t group = 0; group < count; ++group) {
    const std::uint32_t nameLength = readU32(input);
    if (nameLength > 256) throw std::runtime_error("CHECKPOINT_NAME_LENGTH");
    std::string name(nameLength, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name.size()));
    const std::uint64_t values = readU64(input);
    if (!input || name != registry[group].name || values != registry[group].values->size())
      throw std::runtime_error("CHECKPOINT_REGISTRY_IDENTITY");
    auto& destination = *const_cast<std::vector<float>*>(registry[group].values);
    input.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(destination.size() * sizeof(float)));
    if (!input) throw std::runtime_error("CHECKPOINT_VALUES_TRUNCATED");
    if (!std::all_of(destination.begin(), destination.end(),
                     [](float value) { return std::isfinite(value); }))
      throw std::runtime_error("CHECKPOINT_VALUES_NONFINITE");
  }
  if (v2 || v3) {
    // NPRTCKPTV2 appends the Adam first/second moment registries (same
    // registry layout as the parameters). The CPU evaluation only uses the
    // parameters, so the moments are read and discarded.
    for (int moment = 0; moment < 2; ++moment) {
      const std::uint32_t momentCount = readU32(input);
      if (momentCount != registry.size())
        throw std::runtime_error("CHECKPOINT_ADAM_REGISTRY_COUNT");
      std::vector<float> scratch;
      for (std::uint32_t group = 0; group < momentCount; ++group) {
        const std::uint32_t nameLength = readU32(input);
        if (nameLength > 256) throw std::runtime_error("CHECKPOINT_NAME_LENGTH");
        std::string name(nameLength, '\0');
        input.read(name.data(), static_cast<std::streamsize>(name.size()));
        const std::uint64_t values = readU64(input);
        if (!input || name != registry[group].name || values != registry[group].values->size())
          throw std::runtime_error("CHECKPOINT_ADAM_REGISTRY_IDENTITY");
        scratch.resize(static_cast<std::size_t>(values));
        input.read(reinterpret_cast<char*>(scratch.data()), static_cast<std::streamsize>(scratch.size() * sizeof(float)));
        if (!input) throw std::runtime_error("CHECKPOINT_ADAM_VALUES_TRUNCATED");
        if (!std::all_of(scratch.begin(), scratch.end(),
                         [](float value) { return std::isfinite(value); }))
          throw std::runtime_error("CHECKPOINT_ADAM_VALUES_NONFINITE");
      }
    }
  }
  if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("CHECKPOINT_TRAILING_BYTES");
  return checkpoint;
}

bool allFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool parametersFinite(const phonelm::qnn::TinyTransformerParameters& parameters) {
  for (const auto& info : phonelm::tiny_lm::parameterRegistry(parameters)) {
    if (!allFinite(*info.values)) return false;
  }
  return true;
}

std::uint64_t parameterHash(const phonelm::qnn::TinyTransformerParameters& parameters) {
  std::uint64_t hash = kFnvOffset;
  for (const auto& info : phonelm::tiny_lm::parameterRegistry(parameters)) {
    hash = fnvBytes(info.name.data(), info.name.size(), hash);
    const std::uint64_t count = info.values->size();
    hash = fnvBytes(&count, sizeof(count), hash);
    hash = fnvBytes(info.values->data(), info.values->size() * sizeof(float), hash);
  }
  return hash;
}

struct Metrics {
  double nll = 0, perplexity = 0, top1 = 0, top5 = 0, meanRank = 0, meanMargin = 0;
  std::uint64_t tokens = 0, targetBytes = 0, chunks = 0;
  bool finite = true;
};

struct Inference {
  std::vector<float> logits;
  std::vector<float> probabilities;
};

Inference infer(const phonelm::tiny_lm::Config& config, const std::vector<std::uint32_t>& tokens,
                const phonelm::qnn::TinyTransformerParameters& parameters) {
  if (tokens.size() != config.tokens) throw std::runtime_error("INFERENCE_TOKEN_COUNT");
  for (std::uint32_t row = 0; row < config.tokens; ++row)
    if (tokens[row] >= config.vocabularySize) throw std::runtime_error("INFERENCE_TOKEN_RANGE");
  // Production CPU forward is the single evaluation semantics source: gated
  // and ungated models both go through generalForward (LN1 → QKV → SDPA →
  // optional G1 gate → Wo), matching QNN training and FORWARD_ONLY graphs.
  const auto trace = phonelm::tiny_lm::forwardTraceGeneralized(
      config, phonelm::tiny_lm::oneHot(tokens, config.vocabularySize), parameters);
  Inference result;
  result.logits = trace.logits;
  result.probabilities = trace.probabilities;
  return result;
}

Metrics evaluate(const Cache& cache, const phonelm::tiny_lm::Config& config,
                 const phonelm::qnn::TinyTransformerParameters& parameters,
                 std::size_t maximumChunks,
                 const phonelm::nicopedia_bpe::Model* bpeModel = nullptr) {
  Metrics metrics;
  if (cache.records.size() < maximumChunks)
    throw std::runtime_error("CACHE_CAPACITY_MISMATCH");
  const std::size_t count = maximumChunks;
  metrics.chunks = count;
  double top1 = 0, top5 = 0, rankSum = 0, marginSum = 0, lossSum = 0;
  for (std::size_t chunk = 0; chunk < count; ++chunk) {
    const auto& record = cache.records[chunk];
    std::vector<std::uint32_t> input(config.tokens), targets(config.tokens);
    for (std::uint32_t i = 0; i < config.tokens; ++i) {
      input[i] = record.window[i];
      targets[i] = record.window[i + 1];
    }
    const Inference output = infer(config, input, parameters);
    metrics.finite = metrics.finite && allFinite(output.logits) && allFinite(output.probabilities);
    for (std::uint32_t row = 0; row < config.tokens; ++row) {
      const std::size_t base = std::size_t(row) * config.vocabularySize;
      const std::uint32_t truth = targets[row];
      const float truthLogit = output.logits[base + truth];
      float maximum = output.logits[base];
      float maximumOther = -std::numeric_limits<float>::infinity();
      std::uint32_t prediction = 0;
      std::uint32_t rank = 1;
      for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
        const float value = output.logits[base + token];
        if (value > maximum) { maximum = value; prediction = token; }
        if (token != truth) maximumOther = std::max(maximumOther, value);
        if (value > truthLogit) ++rank;
      }
      double exponentialSum = 0;
      for (std::uint32_t token = 0; token < config.vocabularySize; ++token)
        exponentialSum += std::exp(double(output.logits[base + token] - maximum));
      lossSum += maximum + std::log(exponentialSum) - truthLogit;
      top1 += prediction == truth;
      top5 += rank <= 5;
      rankSum += rank;
      marginSum += truthLogit - maximumOther;
      metrics.targetBytes += bpeModel
          ? bpeModel->tokenByteLength(static_cast<std::uint16_t>(truth)) : 1u;
      ++metrics.tokens;
    }
  }
  metrics.nll = lossSum / std::max<std::uint64_t>(1, metrics.tokens);
  metrics.perplexity = std::exp(std::min(50.0, metrics.nll));
  metrics.top1 = top1 / std::max<std::uint64_t>(1, metrics.tokens);
  metrics.top5 = top5 / std::max<std::uint64_t>(1, metrics.tokens);
  metrics.meanRank = rankSum / std::max<std::uint64_t>(1, metrics.tokens);
  metrics.meanMargin = marginSum / std::max<std::uint64_t>(1, metrics.tokens);
  metrics.finite = metrics.finite && std::isfinite(metrics.nll) && std::isfinite(metrics.perplexity);
  return metrics;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: htp_checkpoint_eval CHECKPOINT VALIDATION DEVELOPMENT MAX_VALIDATION_CHUNKS MAX_DEV_CHUNKS\n";
      return 2;
    }
    const Checkpoint checkpoint = loadCheckpoint(argv[1]);
    std::unique_ptr<phonelm::nicopedia_bpe::Model> bpeModel;
    if (checkpoint.config.vocabularySize == 1024) {
      const std::string checkpointPath = argv[1];
      const std::size_t separator = checkpointPath.find_last_of("/\\");
      const std::string directory = separator == std::string::npos ? "." : checkpointPath.substr(0, separator);
      bpeModel = std::make_unique<phonelm::nicopedia_bpe::Model>(
          phonelm::nicopedia_bpe::loadModel(directory + "/byte-bpe-v1024.model"));
      if (checkpoint.tokenizerHash != bpeModel->identity())
        throw std::runtime_error("CHECKPOINT_TOKENIZER_HASH_MISMATCH");
    }
    const Cache validation = loadCache(argv[2], bpeModel.get());
    const Cache development = loadCache(argv[3], bpeModel.get());
    const std::size_t validationChunks = std::stoull(argv[4]);
    const std::size_t developmentChunks = std::stoull(argv[5]);
    // Fail closed: a cache whose context length differs from the checkpoint's
    // tokens would silently misread the window (record.window has
    // context+1 bytes but evaluate reads config.tokens). Reject up front.
    if (validation.context != checkpoint.config.tokens ||
        development.context != checkpoint.config.tokens)
      throw std::runtime_error("CACHE_CONTEXT_MISMATCH");
    if (!parametersFinite(checkpoint.parameters))
      throw std::runtime_error("CHECKPOINT_PARAMETERS_NONFINITE");
    const Metrics validationMetrics =
        evaluate(validation, checkpoint.config, checkpoint.parameters, validationChunks, bpeModel.get());
    const Metrics developmentMetrics =
        evaluate(development, checkpoint.config, checkpoint.parameters, developmentChunks, bpeModel.get());
    if (!validationMetrics.finite || !developmentMetrics.finite)
      throw std::runtime_error("EVALUATION_NONFINITE");
    std::cout << std::setprecision(10)
              << "htp_checkpoint_eval\nseed=" << checkpoint.seed
              << "\nlayers=" << checkpoint.config.numLayers
              << "\ndimension=" << checkpoint.config.dimension
              << "\nfeed_forward_dimension="
              << checkpoint.config.feedForwardDimension
              << "\nstep=" << checkpoint.step
              << "\nparameter_hash=fnv1a64:" << std::hex << std::setw(16)
              << std::setfill('0') << parameterHash(checkpoint.parameters) << std::dec
              << "\nvalidation_nll=" << validationMetrics.nll
              << "\nvalidation_perplexity=" << validationMetrics.perplexity
              << "\nvalidation_top1=" << validationMetrics.top1
              << "\nvalidation_top5=" << validationMetrics.top5
              << "\nvalidation_mean_rank=" << validationMetrics.meanRank
              << "\nvalidation_margin=" << validationMetrics.meanMargin
              << "\nvalidation_chunks=" << validationMetrics.chunks
              << "\nvalidation_tokens=" << validationMetrics.tokens
              << "\nvalidation_target_utf8_bytes=" << validationMetrics.targetBytes
              << "\nvalidation_nats_per_utf8_byte="
              << validationMetrics.nll * validationMetrics.tokens / std::max<std::uint64_t>(1, validationMetrics.targetBytes)
              << "\nvalidation_bits_per_utf8_byte="
              << validationMetrics.nll * validationMetrics.tokens /
                    std::max<std::uint64_t>(1, validationMetrics.targetBytes) / std::log(2.0)
              << "\ndevelopment_nll=" << developmentMetrics.nll
              << "\ndevelopment_perplexity=" << developmentMetrics.perplexity
              << "\ndevelopment_top1=" << developmentMetrics.top1
              << "\ndevelopment_top5=" << developmentMetrics.top5
              << "\ndevelopment_chunks=" << developmentMetrics.chunks
              << "\ndevelopment_tokens=" << developmentMetrics.tokens
              << "\ndevelopment_target_utf8_bytes=" << developmentMetrics.targetBytes
              << "\ndevelopment_nats_per_utf8_byte="
              << developmentMetrics.nll * developmentMetrics.tokens / std::max<std::uint64_t>(1, developmentMetrics.targetBytes)
              << "\ndevelopment_bits_per_utf8_byte="
              << developmentMetrics.nll * developmentMetrics.tokens /
                    std::max<std::uint64_t>(1, developmentMetrics.targetBytes) / std::log(2.0)
              << "\nfinite=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "htp_checkpoint_eval_error=" << error.what() << '\n';
    return 1;
  }
}
