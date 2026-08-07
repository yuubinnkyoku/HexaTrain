// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host-only evaluation helper for the private Nicopedia HTP checkpoints.
// Loads an NPRTCKPTV1 checkpoint (as written by the device NICOPEDIA mode)
// and evaluates it on the fixed validation/development caches with the same
// CPU reference used by the pilot.  Outputs aggregate NLL only; the private
// checkpoint itself never enters the repository.
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
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
  std::vector<std::uint8_t> window;
};

struct Cache {
  std::uint32_t context = 0;
  std::uint32_t vocabulary = 0;
  std::vector<CacheRecord> records;
  std::string contentHash;
};

Cache loadCache(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("CACHE_OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTBYTEV1\n") throw std::runtime_error("CACHE_MAGIC_MISMATCH");
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
    record.window.resize(cache.context + 1);
    input.read(reinterpret_cast<char*>(record.window.data()), static_cast<std::streamsize>(record.window.size()));
    if (!input) throw std::runtime_error("CACHE_RECORD_TRUNCATED");
    hash = fnvBytes(&record.articleHash, sizeof(record.articleHash), hash);
    hash = fnvBytes(record.window.data(), record.window.size(), hash);
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
  phonelm::qnn::TinyTransformerParameters parameters;
};

Checkpoint loadCheckpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("CHECKPOINT_OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTCKPTV1\n") throw std::runtime_error("CHECKPOINT_MAGIC");
  Checkpoint checkpoint;
  checkpoint.config.vocabularySize = readU32(input);
  checkpoint.config.tokens = readU32(input);
  checkpoint.config.dimension = readU32(input);
  checkpoint.config.feedForwardDimension = readU32(input);
  checkpoint.config.numLayers = readU32(input);
  checkpoint.config.numHeads = readU32(input);
  checkpoint.seed = readU32(input);
  checkpoint.step = readU32(input);
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
  std::uint64_t tokens = 0;
  bool finite = true;
};

std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b,
                            std::uint32_t rows, std::uint32_t inner, std::uint32_t columns) {
  std::vector<float> result(std::size_t(rows) * columns);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      double sum = 0;
      for (std::uint32_t index = 0; index < inner; ++index)
        sum += double(a[std::size_t(row) * inner + index]) * b[std::size_t(index) * columns + column];
      result[std::size_t(row) * columns + column] = float(sum);
    }
  }
  return result;
}

std::vector<float> normalize(const phonelm::tiny_lm::Config& config, const std::vector<float>& input,
                             const std::vector<float>& gamma, const std::vector<float>& beta) {
  std::vector<float> result(input.size());
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    double mean = 0;
    for (std::uint32_t d = 0; d < config.dimension; ++d) mean += input[std::size_t(row) * config.dimension + d];
    mean /= config.dimension;
    double variance = 0;
    for (std::uint32_t d = 0; d < config.dimension; ++d) {
      const double centered = input[std::size_t(row) * config.dimension + d] - mean;
      variance += centered * centered;
    }
    variance /= config.dimension;
    const double inverse = 1.0 / std::sqrt(variance + config.epsilon);
    for (std::uint32_t d = 0; d < config.dimension; ++d) {
      const std::size_t index = std::size_t(row) * config.dimension + d;
      result[index] = float((input[index] - mean) * inverse) * gamma[d] + beta[d];
    }
  }
  return result;
}

void addInPlace(std::vector<float>& target, const std::vector<float>& addition) {
  if (target.size() != addition.size()) throw std::runtime_error("INFERENCE_SHAPE_MISMATCH");
  for (std::size_t i = 0; i < target.size(); ++i) target[i] += addition[i];
}

struct Inference {
  std::vector<float> logits;
  std::vector<float> probabilities;
};

const phonelm::qnn::TinyTransformerLayerParameters& layer(const phonelm::qnn::TinyTransformerParameters& parameters, std::uint32_t index) {
  if (index == 0) return static_cast<const phonelm::qnn::TinyTransformerLayerParameters&>(parameters);
  return parameters.layers.at(index - 1);
}

Inference infer(const phonelm::tiny_lm::Config& config, const std::vector<std::uint32_t>& tokens,
                const phonelm::qnn::TinyTransformerParameters& parameters) {
  if (tokens.size() != config.tokens) throw std::runtime_error("INFERENCE_TOKEN_COUNT");
  std::vector<float> input(std::size_t(config.tokens) * config.dimension);
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    if (tokens[row] >= config.vocabularySize) throw std::runtime_error("INFERENCE_TOKEN_RANGE");
    const std::size_t source = std::size_t(tokens[row]) * config.dimension;
    const std::size_t destination = std::size_t(row) * config.dimension;
    std::copy_n(parameters.tokenEmbedding.begin() + static_cast<std::ptrdiff_t>(source),
                config.dimension, input.begin() + static_cast<std::ptrdiff_t>(destination));
  }
  addInPlace(input, phonelm::tiny_lm::fixedPosition(config));
  const std::uint32_t headDimension = config.dimension / config.numHeads;
  const float scale = 1.0f / std::sqrt(float(headDimension));
  for (std::uint32_t layerIndex = 0; layerIndex < config.numLayers; ++layerIndex) {
    const auto& weights = layer(parameters, layerIndex);
    const auto norm1 = normalize(config, input, weights.gamma1, weights.beta1);
    const auto q = multiply(norm1, weights.wq, config.tokens, config.dimension, config.dimension);
    const auto k = multiply(norm1, weights.wk, config.tokens, config.dimension, config.dimension);
    const auto v = multiply(norm1, weights.wv, config.tokens, config.dimension, config.dimension);
    std::vector<float> context(std::size_t(config.tokens) * config.dimension);
    for (std::uint32_t head = 0; head < config.numHeads; ++head) {
      for (std::uint32_t row = 0; row < config.tokens; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(config.tokens, -std::numeric_limits<float>::infinity());
        for (std::uint32_t column = 0; column <= row; ++column) {
          double score = 0;
          for (std::uint32_t d = 0; d < headDimension; ++d)
            score += double(q[std::size_t(row) * config.dimension + head * headDimension + d]) *
                     k[std::size_t(column) * config.dimension + head * headDimension + d];
          scores[column] = float(score) * scale;
          maximum = std::max(maximum, scores[column]);
        }
        double sum = 0;
        for (std::uint32_t column = 0; column <= row; ++column) {
          scores[column] = std::exp(scores[column] - maximum);
          sum += scores[column];
        }
        for (std::uint32_t column = 0; column <= row; ++column) {
          const float probability = scores[column] / float(sum);
          for (std::uint32_t d = 0; d < headDimension; ++d)
            context[std::size_t(row) * config.dimension + head * headDimension + d] +=
                probability * v[std::size_t(column) * config.dimension + head * headDimension + d];
        }
      }
    }
    auto residual = input;
    addInPlace(residual, multiply(context, weights.wo, config.tokens, config.dimension, config.dimension));
    const auto norm2 = normalize(config, residual, weights.gamma2, weights.beta2);
    auto hidden = multiply(norm2, weights.w1, config.tokens, config.dimension, config.feedForwardDimension);
    for (float& value : hidden) value = std::max(0.0f, value);
    input = residual;
    addInPlace(input, multiply(hidden, weights.w2, config.tokens, config.feedForwardDimension, config.dimension));
  }
  Inference result;
  result.logits = multiply(input, parameters.outputProjection, config.tokens, config.dimension, config.vocabularySize);
  result.probabilities.resize(result.logits.size());
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    const std::size_t base = std::size_t(row) * config.vocabularySize;
    const float maximum = *std::max_element(result.logits.begin() + static_cast<std::ptrdiff_t>(base),
                                             result.logits.begin() + static_cast<std::ptrdiff_t>(base + config.vocabularySize));
    double sum = 0;
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
      result.probabilities[base + token] = std::exp(result.logits[base + token] - maximum);
      sum += result.probabilities[base + token];
    }
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token)
      result.probabilities[base + token] /= float(sum);
  }
  return result;
}

Metrics evaluate(const Cache& cache, const phonelm::tiny_lm::Config& config,
                 const phonelm::qnn::TinyTransformerParameters& parameters,
                 std::size_t maximumChunks) {
  Metrics metrics;
  const std::size_t count = std::min(maximumChunks, cache.records.size());
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
    const Cache validation = loadCache(argv[2]);
    const Cache development = loadCache(argv[3]);
    const std::size_t validationChunks = std::stoull(argv[4]);
    const std::size_t developmentChunks = std::stoull(argv[5]);
    if (!parametersFinite(checkpoint.parameters))
      throw std::runtime_error("CHECKPOINT_PARAMETERS_NONFINITE");
    const Metrics validationMetrics =
        evaluate(validation, checkpoint.config, checkpoint.parameters, validationChunks);
    const Metrics developmentMetrics =
        evaluate(development, checkpoint.config, checkpoint.parameters, developmentChunks);
    if (!validationMetrics.finite || !developmentMetrics.finite)
      throw std::runtime_error("EVALUATION_NONFINITE");
    std::cout << std::setprecision(10)
              << "htp_checkpoint_eval\nseed=" << checkpoint.seed
              << "\nlayers=" << checkpoint.config.numLayers
              << "\nstep=" << checkpoint.step
              << "\nparameter_hash=fnv1a64:" << std::hex << std::setw(16)
              << std::setfill('0') << parameterHash(checkpoint.parameters) << std::dec
              << "\nvalidation_nll=" << validationMetrics.nll
              << "\nvalidation_perplexity=" << validationMetrics.perplexity
              << "\nvalidation_top1=" << validationMetrics.top1
              << "\nvalidation_top5=" << validationMetrics.top5
              << "\nvalidation_mean_rank=" << validationMetrics.meanRank
              << "\nvalidation_margin=" << validationMetrics.meanMargin
              << "\nvalidation_tokens=" << validationMetrics.tokens
              << "\ndevelopment_nll=" << developmentMetrics.nll
              << "\ndevelopment_perplexity=" << developmentMetrics.perplexity
              << "\ndevelopment_top1=" << developmentMetrics.top1
              << "\ndevelopment_top5=" << developmentMetrics.top5
              << "\ndevelopment_tokens=" << developmentMetrics.tokens
              << "\nfinite=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "htp_checkpoint_eval_error=" << error.what() << '\n';
    return 1;
  }
}
