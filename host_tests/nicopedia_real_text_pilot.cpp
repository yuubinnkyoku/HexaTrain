// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;
namespace lm = phonelm::tiny_lm;
using Parameters = phonelm::qnn::TinyTransformerParameters;
using Layer = phonelm::qnn::TinyTransformerLayerParameters;

namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

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

struct Inference {
  std::vector<float> logits;
  std::vector<float> probabilities;
  std::vector<float> lastAttention;
};

struct Metrics {
  double nll = 0;
  double perplexity = 0;
  double top1 = 0;
  double top5 = 0;
  double meanRank = 0;
  double meanMargin = 0;
  std::uint64_t tokens = 0;
  bool finite = true;
};

struct Checkpoint {
  lm::Config config;
  std::uint32_t seed = 0;
  std::uint32_t step = 0;
  Parameters parameters;
};

std::uint64_t fnvBytes(const void* data, std::size_t bytes, std::uint64_t hash = kFnvOffset) {
  const auto* input = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= input[i];
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
std::uint64_t fnvValue(const T& value, std::uint64_t hash) {
  return fnvBytes(&value, sizeof(value), hash);
}

std::string hex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

std::uint32_t readU32(std::istream& input) {
  std::array<std::uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("CACHE_TRUNCATED_U32");
  return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
         (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

std::uint64_t readU64(std::istream& input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("CACHE_TRUNCATED_U64");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes) value = (value << 8) | byte;
  return value;
}

void writeU32(std::ostream& output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) output.put(static_cast<char>((value >> shift) & 0xffu));
}

void writeU64(std::ostream& output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) output.put(static_cast<char>((value >> shift) & 0xffu));
}

Cache loadCache(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("CACHE_OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != "NPRTBYTEV1\n") throw std::runtime_error("CACHE_MAGIC_MISMATCH");
  Cache cache;
  cache.context = readU32(input);
  cache.vocabulary = readU32(input);
  const std::uint64_t count = readU64(input);
  if (cache.context < 8 || cache.context > 256 || cache.vocabulary != 256 || count > 10000000) {
    throw std::runtime_error("CACHE_HEADER_INVALID");
  }
  cache.records.reserve(static_cast<std::size_t>(count));
  std::uint64_t hash = kFnvOffset;
  hash = fnvValue(cache.context, hash);
  hash = fnvValue(cache.vocabulary, hash);
  hash = fnvValue(count, hash);
  for (std::uint64_t i = 0; i < count; ++i) {
    CacheRecord record;
    record.articleHash = readU64(input);
    record.window.resize(cache.context + 1);
    input.read(reinterpret_cast<char*>(record.window.data()), static_cast<std::streamsize>(record.window.size()));
    if (!input) throw std::runtime_error("CACHE_RECORD_TRUNCATED");
    hash = fnvValue(record.articleHash, hash);
    hash = fnvBytes(record.window.data(), record.window.size(), hash);
    cache.records.push_back(std::move(record));
  }
  if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("CACHE_TRAILING_BYTES");
  cache.contentHash = hex64(hash);
  if (cache.records.empty()) throw std::runtime_error("CACHE_EMPTY");
  return cache;
}

const Layer& layer(const Parameters& parameters, std::uint32_t index) {
  if (index == 0) return static_cast<const Layer&>(parameters);
  return parameters.layers.at(index - 1);
}

std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b,
                            std::uint32_t rows, std::uint32_t inner, std::uint32_t columns) {
  std::vector<float> result(std::size_t(rows) * columns);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      double sum = 0;
      for (std::uint32_t index = 0; index < inner; ++index) {
        sum += double(a[std::size_t(row) * inner + index]) * b[std::size_t(index) * columns + column];
      }
      result[std::size_t(row) * columns + column] = float(sum);
    }
  }
  return result;
}

std::vector<float> normalize(const lm::Config& config, const std::vector<float>& input,
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

Inference infer(const lm::Config& config, const std::vector<std::uint32_t>& tokens,
                const Parameters& parameters) {
  if (tokens.size() != config.tokens) throw std::runtime_error("INFERENCE_TOKEN_COUNT");
  std::vector<float> input(std::size_t(config.tokens) * config.dimension);
  for (std::uint32_t row = 0; row < config.tokens; ++row) {
    if (tokens[row] >= config.vocabularySize) throw std::runtime_error("INFERENCE_TOKEN_RANGE");
    const std::size_t source = std::size_t(tokens[row]) * config.dimension;
    const std::size_t destination = std::size_t(row) * config.dimension;
    std::copy_n(parameters.tokenEmbedding.begin() + static_cast<std::ptrdiff_t>(source),
                config.dimension, input.begin() + static_cast<std::ptrdiff_t>(destination));
  }
  addInPlace(input, lm::fixedPosition(config));
  std::vector<float> lastAttention;
  const std::uint32_t headDimension = config.dimension / config.numHeads;
  const float scale = 1.0f / std::sqrt(float(headDimension));
  for (std::uint32_t layerIndex = 0; layerIndex < config.numLayers; ++layerIndex) {
    const Layer& weights = layer(parameters, layerIndex);
    const auto norm1 = normalize(config, input, weights.gamma1, weights.beta1);
    const auto q = multiply(norm1, weights.wq, config.tokens, config.dimension, config.dimension);
    const auto k = multiply(norm1, weights.wk, config.tokens, config.dimension, config.dimension);
    const auto v = multiply(norm1, weights.wv, config.tokens, config.dimension, config.dimension);
    std::vector<float> attention(std::size_t(config.numHeads) * config.tokens * config.tokens);
    std::vector<float> context(std::size_t(config.tokens) * config.dimension);
    for (std::uint32_t head = 0; head < config.numHeads; ++head) {
      for (std::uint32_t row = 0; row < config.tokens; ++row) {
        const std::size_t base = (std::size_t(head) * config.tokens + row) * config.tokens;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t column = 0; column <= row; ++column) {
          double score = 0;
          for (std::uint32_t d = 0; d < headDimension; ++d) {
            score += double(q[std::size_t(row) * config.dimension + head * headDimension + d]) *
                     k[std::size_t(column) * config.dimension + head * headDimension + d];
          }
          attention[base + column] = float(score) * scale;
          maximum = std::max(maximum, attention[base + column]);
        }
        double sum = 0;
        for (std::uint32_t column = 0; column <= row; ++column) {
          attention[base + column] = std::exp(attention[base + column] - maximum);
          sum += attention[base + column];
        }
        for (std::uint32_t column = 0; column <= row; ++column) {
          const float probability = attention[base + column] / float(sum);
          attention[base + column] = probability;
          for (std::uint32_t d = 0; d < headDimension; ++d) {
            context[std::size_t(row) * config.dimension + head * headDimension + d] +=
                probability * v[std::size_t(column) * config.dimension + head * headDimension + d];
          }
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
    if (layerIndex + 1 == config.numLayers) lastAttention = std::move(attention);
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
    for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
      result.probabilities[base + token] /= float(sum);
    }
  }
  result.lastAttention = std::move(lastAttention);
  return result;
}

std::vector<std::uint32_t> inputTokens(const CacheRecord& record, std::uint32_t context) {
  std::vector<std::uint32_t> result(context);
  for (std::uint32_t i = 0; i < context; ++i) result[i] = record.window[i];
  return result;
}

std::vector<std::uint32_t> targetTokens(const CacheRecord& record, std::uint32_t context) {
  std::vector<std::uint32_t> result(context);
  for (std::uint32_t i = 0; i < context; ++i) result[i] = record.window[i + 1];
  return result;
}

bool allFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool parametersFinite(const Parameters& parameters) {
  for (const auto& info : lm::parameterRegistry(parameters)) {
    if (!allFinite(*info.values)) return false;
  }
  return true;
}

bool betterValidation(double candidateNll, std::uint32_t candidateStep,
                      double bestNll, std::uint32_t bestStep) {
  return candidateNll < bestNll - 1e-7 ||
         (std::abs(candidateNll - bestNll) <= 1e-7 && (!bestStep || candidateStep < bestStep));
}

Metrics evaluate(const Cache& cache, const lm::Config& config, const Parameters& parameters,
                 std::size_t maximumChunks = std::numeric_limits<std::size_t>::max()) {
  Metrics metrics;
  const std::size_t count = std::min(maximumChunks, cache.records.size());
  double top1 = 0, top5 = 0, rankSum = 0, marginSum = 0, lossSum = 0;
  for (std::size_t chunk = 0; chunk < count; ++chunk) {
    const auto input = inputTokens(cache.records[chunk], config.tokens);
    const auto targets = targetTokens(cache.records[chunk], config.tokens);
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
      for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
        exponentialSum += std::exp(double(output.logits[base + token] - maximum));
      }
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

Parameters zerosLike(const Parameters& source) {
  Parameters result = source;
  for (const auto& info : lm::parameterRegistry(result)) {
    auto& values = *const_cast<std::vector<float>*>(info.values);
    std::fill(values.begin(), values.end(), 0.0f);
  }
  return result;
}

void addScaled(Parameters& destination, const Parameters& source, float scale) {
  const auto destinationRegistry = lm::parameterRegistry(destination);
  const auto sourceRegistry = lm::parameterRegistry(source);
  if (destinationRegistry.size() != sourceRegistry.size()) throw std::runtime_error("PARAMETER_REGISTRY_SIZE");
  for (std::size_t group = 0; group < destinationRegistry.size(); ++group) {
    if (destinationRegistry[group].name != sourceRegistry[group].name ||
        destinationRegistry[group].values->size() != sourceRegistry[group].values->size()) {
      throw std::runtime_error("PARAMETER_REGISTRY_IDENTITY");
    }
    auto& target = *const_cast<std::vector<float>*>(destinationRegistry[group].values);
    const auto& values = *sourceRegistry[group].values;
    for (std::size_t i = 0; i < target.size(); ++i) target[i] += scale * values[i];
  }
}

std::string parameterHash(const Parameters& parameters) {
  std::uint64_t hash = kFnvOffset;
  for (const auto& info : lm::parameterRegistry(parameters)) {
    hash = fnvBytes(info.name.data(), info.name.size(), hash);
    const std::uint64_t count = info.values->size();
    hash = fnvValue(count, hash);
    hash = fnvBytes(info.values->data(), info.values->size() * sizeof(float), hash);
  }
  return hex64(hash);
}

std::uint64_t splitMix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

std::vector<std::size_t> trainingOrder(std::size_t recordCount, std::uint32_t steps,
                                       std::uint32_t batchSize, std::uint64_t orderSeed) {
  if (!recordCount) throw std::runtime_error("TRAIN_CACHE_EMPTY");
  std::vector<std::size_t> order;
  order.reserve(std::size_t(steps) * batchSize);
  std::uint64_t state = orderSeed;
  for (std::size_t i = 0; i < std::size_t(steps) * batchSize; ++i) {
    state = splitMix(state + i);
    order.push_back(static_cast<std::size_t>(state % recordCount));
  }
  return order;
}

std::string orderHash(const std::vector<std::size_t>& order) {
  std::uint64_t hash = kFnvOffset;
  for (std::size_t index : order) {
    const std::uint64_t value = index;
    hash = fnvValue(value, hash);
  }
  return hex64(hash);
}

void saveCheckpoint(const fs::path& path, const Checkpoint& checkpoint) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("CHECKPOINT_OPEN_FAILED");
  output.write("NPRTCKPTV1\n", 11);
  writeU32(output, checkpoint.config.vocabularySize);
  writeU32(output, checkpoint.config.tokens);
  writeU32(output, checkpoint.config.dimension);
  writeU32(output, checkpoint.config.feedForwardDimension);
  writeU32(output, checkpoint.config.numLayers);
  writeU32(output, checkpoint.config.numHeads);
  writeU32(output, checkpoint.seed);
  writeU32(output, checkpoint.step);
  const auto registry = lm::parameterRegistry(checkpoint.parameters);
  writeU32(output, static_cast<std::uint32_t>(registry.size()));
  for (const auto& info : registry) {
    writeU32(output, static_cast<std::uint32_t>(info.name.size()));
    output.write(info.name.data(), static_cast<std::streamsize>(info.name.size()));
    writeU64(output, info.values->size());
    output.write(reinterpret_cast<const char*>(info.values->data()),
                 static_cast<std::streamsize>(info.values->size() * sizeof(float)));
  }
  if (!output) throw std::runtime_error("CHECKPOINT_WRITE_FAILED");
}

Checkpoint loadCheckpoint(const fs::path& path) {
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
  checkpoint.parameters = lm::initialParameters(checkpoint.config, checkpoint.seed);
  auto registry = lm::parameterRegistry(checkpoint.parameters);
  const std::uint32_t count = readU32(input);
  if (count != registry.size()) throw std::runtime_error("CHECKPOINT_REGISTRY_COUNT");
  for (std::uint32_t group = 0; group < count; ++group) {
    const std::uint32_t nameLength = readU32(input);
    if (nameLength > 256) throw std::runtime_error("CHECKPOINT_NAME_LENGTH");
    std::string name(nameLength, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name.size()));
    const std::uint64_t values = readU64(input);
    if (!input || name != registry[group].name || values != registry[group].values->size()) {
      throw std::runtime_error("CHECKPOINT_REGISTRY_IDENTITY");
    }
    auto& destination = *const_cast<std::vector<float>*>(registry[group].values);
    input.read(reinterpret_cast<char*>(destination.data()),
               static_cast<std::streamsize>(destination.size() * sizeof(float)));
    if (!input) throw std::runtime_error("CHECKPOINT_VALUES_TRUNCATED");
  }
  if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("CHECKPOINT_TRAILING_BYTES");
  return checkpoint;
}

std::uint64_t peakWorkingSetBytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  }
  return 0;
#else
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ull;
  return 0;
#endif
}

void writeMetrics(std::ostream& output, const Metrics& metrics) {
  output << std::setprecision(10) << metrics.nll << ',' << metrics.perplexity << ','
         << metrics.top1 << ',' << metrics.top5 << ',' << metrics.meanRank << ','
         << metrics.meanMargin << ',' << metrics.tokens << ',' << (metrics.finite ? "true" : "false");
}

lm::Config makeConfig(std::uint32_t context, std::uint32_t layers) {
  lm::Config config;
  config.vocabularySize = 256;
  config.tokens = context;
  config.dimension = 16;
  config.feedForwardDimension = 32;
  config.numLayers = layers;
  config.numHeads = 2;
  std::string error;
  if (!lm::validateConfig(config, &error)) throw std::runtime_error("CONFIG_INVALID:" + error);
  return config;
}

void runBenchmark(const fs::path& cachePath, const fs::path& outputPath, std::uint32_t layers,
                  std::uint32_t measuredSteps) {
  if (!measuredSteps) throw std::runtime_error("BENCHMARK_STEPS_MUST_BE_POSITIVE");
  const Cache cache = loadCache(cachePath);
  const lm::Config config = makeConfig(cache.context, layers);
  Parameters parameters = lm::initialParameters(config, 1);
  Parameters first = zerosLike(parameters), second = zerosLike(parameters);
  const auto order = trainingOrder(cache.records.size(), measuredSteps, 1, 20260806);
  const auto started = std::chrono::steady_clock::now();
  for (std::uint32_t step = 1; step <= measuredSteps; ++step) {
    const auto& record = cache.records[order[step - 1]];
    const auto result = lm::forwardBackwardGeneralized(
        config, lm::oneHot(inputTokens(record, config.tokens), config.vocabularySize),
        lm::oneHot(targetTokens(record, config.tokens), config.vocabularySize), parameters, 0.0f);
    const float firstCorrection = 1.0f / (1.0f - std::pow(0.9f, float(step)));
    const float secondCorrection = 1.0f / (1.0f - std::pow(0.999f, float(step)));
    auto update = lm::adamUpdate(parameters, result.gradients, first, second, 0.003f, 0.9f, 0.999f,
                                 1e-8f, firstCorrection, secondCorrection);
    parameters = std::move(update.next);
    first = std::move(update.firstMoment);
    second = std::move(update.secondMoment);
    if (!std::isfinite(result.loss)) throw std::runtime_error("BENCHMARK_NONFINITE");
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  fs::create_directories(outputPath.parent_path());
  std::ofstream output(outputPath, std::ios::trunc);
  output << "layers,context,vocabulary,dimension,ffn,heads,parameter_count,measured_steps,seconds,step_seconds,estimated_100_steps_seconds,estimated_320_steps_seconds,peak_working_set_bytes,cache_bytes,checkpoint_write_seconds\n";
  const fs::path checkpointPath = outputPath.parent_path() / "benchmark-private.ckpt";
  const auto checkpointStarted = std::chrono::steady_clock::now();
  saveCheckpoint(checkpointPath, Checkpoint{config, 1, measuredSteps, parameters});
  const double checkpointSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - checkpointStarted).count();
  const double perStep = seconds / measuredSteps;
  output << layers << ',' << config.tokens << ',' << config.vocabularySize << ',' << config.dimension << ','
         << config.feedForwardDimension << ',' << config.numHeads << ',' << lm::parameterElementCount(parameters) << ','
         << measuredSteps << ',' << seconds << ',' << perStep << ',' << perStep * 100 << ',' << perStep * 320 << ','
         << peakWorkingSetBytes() << ',' << fs::file_size(cachePath) << ',' << checkpointSeconds << '\n';
  std::cout << "benchmark_status=PASS layers=" << layers << " step_seconds=" << perStep << '\n';
}

void runTraining(const fs::path& trainPath, const fs::path& validationPath, const fs::path& outputDirectory,
                 std::uint32_t seed, std::uint32_t layers, std::uint32_t steps,
                 std::uint32_t batchSize, std::uint32_t checkpointInterval) {
  const Cache train = loadCache(trainPath);
  const Cache validation = loadCache(validationPath);
  if (train.context != validation.context || train.vocabulary != validation.vocabulary) {
    throw std::runtime_error("TRAIN_VALIDATION_CACHE_CONFIG_MISMATCH");
  }
  const lm::Config config = makeConfig(train.context, layers);
  Parameters parameters = lm::initialParameters(config, seed);
  const std::string initialHash = parameterHash(parameters);
  Parameters first = zerosLike(parameters), second = zerosLike(parameters);
  const auto order = trainingOrder(train.records.size(), steps, batchSize, 20260806);
  const std::string fixedOrderHash = orderHash(order);
  fs::create_directories(outputDirectory);
  std::ofstream trajectory(outputDirectory / "training-trajectory.csv", std::ios::trunc);
  trajectory << "seed,layers,step,train_nll,validation_nll,validation_perplexity,validation_top1,validation_top5,validation_mean_rank,validation_margin,validation_tokens,finite,parameter_hash\n";
  double bestNll = std::numeric_limits<double>::infinity();
  std::uint32_t bestStep = 0;
  Parameters bestParameters;
  Metrics finalValidation;
  double lastTrainLoss = 0;
  const auto started = std::chrono::steady_clock::now();
  for (std::uint32_t step = 1; step <= steps; ++step) {
    Parameters gradient = zerosLike(parameters);
    double loss = 0;
    bool finite = true;
    for (std::uint32_t batch = 0; batch < batchSize; ++batch) {
      const auto& record = train.records[order[std::size_t(step - 1) * batchSize + batch]];
      const auto result = lm::forwardBackwardGeneralized(
          config, lm::oneHot(inputTokens(record, config.tokens), config.vocabularySize),
          lm::oneHot(targetTokens(record, config.tokens), config.vocabularySize), parameters, 0.0f);
      loss += result.loss;
      finite = finite && std::isfinite(result.loss) && parametersFinite(result.gradients);
      addScaled(gradient, result.gradients, 1.0f / batchSize);
    }
    lastTrainLoss = loss / batchSize;
    const float firstCorrection = 1.0f / (1.0f - std::pow(0.9f, float(step)));
    const float secondCorrection = 1.0f / (1.0f - std::pow(0.999f, float(step)));
    auto update = lm::adamUpdate(parameters, gradient, first, second, 0.003f, 0.9f, 0.999f,
                                 1e-8f, firstCorrection, secondCorrection);
    parameters = std::move(update.next);
    first = std::move(update.firstMoment);
    second = std::move(update.secondMoment);
    finite = finite && parametersFinite(parameters) && parametersFinite(first) && parametersFinite(second);
    if (!finite) throw std::runtime_error("TRAINING_NONFINITE");
    if (step % checkpointInterval == 0 || step == steps) {
      const Metrics validationMetrics = evaluate(validation, config, parameters, 256);
      trajectory << seed << ',' << layers << ',' << step << ',' << lastTrainLoss << ',';
      writeMetrics(trajectory, validationMetrics);
      trajectory << ',' << parameterHash(parameters) << '\n';
      if (!validationMetrics.finite) throw std::runtime_error("VALIDATION_NONFINITE");
      if (betterValidation(validationMetrics.nll, step, bestNll, bestStep)) {
        bestNll = validationMetrics.nll;
        bestStep = step;
        bestParameters = parameters;
      }
      finalValidation = validationMetrics;
    }
  }
  const double runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  if (!bestStep) throw std::runtime_error("NO_SELECTED_CHECKPOINT");
  const fs::path checkpointPath = outputDirectory / "selected-private.ckpt";
  const auto checkpointStarted = std::chrono::steady_clock::now();
  saveCheckpoint(checkpointPath, Checkpoint{config, seed, bestStep, bestParameters});
  const double checkpointSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - checkpointStarted).count();
  std::ofstream summary(outputDirectory / "run-summary.csv", std::ios::trunc);
  summary << "seed,layers,steps,batch_size,checkpoint_interval,initial_parameter_hash,train_cache_hash,validation_cache_hash,training_order_hash,best_step,best_validation_nll,final_validation_nll,last_train_nll,finite,runtime_seconds,peak_working_set_bytes,checkpoint_write_seconds,checkpoint_parameter_hash,development_used_for_selection,final_test_used\n";
  summary << seed << ',' << layers << ',' << steps << ',' << batchSize << ',' << checkpointInterval << ','
          << initialHash << ',' << train.contentHash << ',' << validation.contentHash << ',' << fixedOrderHash << ','
          << bestStep << ',' << std::setprecision(10) << bestNll << ',' << finalValidation.nll << ',' << lastTrainLoss
          << ",true," << runtime << ',' << peakWorkingSetBytes() << ',' << checkpointSeconds << ','
          << parameterHash(bestParameters) << ",false,false\n";
  std::cout << "training_status=PASS seed=" << seed << " layers=" << layers << " best_step=" << bestStep
            << " best_validation_nll=" << bestNll << '\n';
}

double jsDivergence(const float* a, const float* b, std::uint32_t vocabulary) {
  double result = 0;
  for (std::uint32_t token = 0; token < vocabulary; ++token) {
    const double left = std::max(1e-30, double(a[token]));
    const double right = std::max(1e-30, double(b[token]));
    const double middle = 0.5 * (left + right);
    result += 0.5 * left * std::log(left / middle) + 0.5 * right * std::log(right / middle);
  }
  return result;
}

std::uint32_t argmaxRow(const Inference& inference, const lm::Config& config, std::uint32_t row) {
  const auto begin = inference.logits.begin() + std::ptrdiff_t(std::size_t(row) * config.vocabularySize);
  return static_cast<std::uint32_t>(std::max_element(begin, begin + config.vocabularySize) - begin);
}

bool validUtf8(const std::vector<std::uint8_t>& bytes) {
  std::size_t i = 0;
  while (i < bytes.size()) {
    const std::uint8_t first = bytes[i++];
    if (first < 0x80) continue;
    int remaining = 0;
    std::uint32_t code = 0;
    if ((first & 0xe0) == 0xc0) { remaining = 1; code = first & 0x1f; if (code < 2) return false; }
    else if ((first & 0xf0) == 0xe0) { remaining = 2; code = first & 0x0f; }
    else if ((first & 0xf8) == 0xf0) { remaining = 3; code = first & 0x07; }
    else return false;
    if (i + remaining > bytes.size()) return false;
    for (int j = 0; j < remaining; ++j) {
      const std::uint8_t next = bytes[i++];
      if ((next & 0xc0) != 0x80) return false;
      code = (code << 6) | (next & 0x3f);
    }
    if ((remaining == 2 && code < 0x800) || (remaining == 3 && code < 0x10000) ||
        code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
  }
  return true;
}

void runCompare(const fs::path& trainPath, const fs::path& developmentPath,
                const fs::path& outputDirectory, const std::vector<fs::path>& checkpointPaths) {
  if (checkpointPaths.size() != 3) throw std::runtime_error("COMPARE_REQUIRES_THREE_CHECKPOINTS");
  const Cache train = loadCache(trainPath);
  const Cache development = loadCache(developmentPath);
  std::vector<Checkpoint> checkpoints;
  for (const auto& path : checkpointPaths) checkpoints.push_back(loadCheckpoint(path));
  const lm::Config config = checkpoints.front().config;
  const std::array<std::uint32_t, 3> expectedSeeds{1, 2, 4};
  for (std::size_t checkpointIndex = 0; checkpointIndex < checkpoints.size(); ++checkpointIndex) {
    const auto& checkpoint = checkpoints[checkpointIndex];
    if (checkpoint.config.vocabularySize != config.vocabularySize || checkpoint.config.tokens != config.tokens ||
        checkpoint.config.dimension != config.dimension ||
        checkpoint.config.feedForwardDimension != config.feedForwardDimension ||
        checkpoint.config.numLayers != config.numLayers || checkpoint.config.numHeads != config.numHeads ||
        checkpoint.config.epsilon != config.epsilon) throw std::runtime_error("COMPARE_CONFIG_MISMATCH");
    if (checkpoint.seed != expectedSeeds[checkpointIndex]) throw std::runtime_error("COMPARE_SEED_ORDER_MISMATCH");
  }
  if (config.vocabularySize != 256 || config.tokens != 32 || config.dimension != 16 ||
      config.feedForwardDimension != 32 || config.numHeads != 2 ||
      (config.numLayers != 6 && config.numLayers != 19)) throw std::runtime_error("COMPARE_FORMAL_CONFIG_MISMATCH");
  if (config.tokens != train.context || config.vocabularySize != train.vocabulary ||
      config.tokens != development.context || config.vocabularySize != development.vocabulary) {
    throw std::runtime_error("COMPARE_CACHE_CONFIG_MISMATCH");
  }
  fs::create_directories(outputDirectory);
  std::ofstream teacher(outputDirectory / "development-teacher-forced.csv", std::ios::trunc);
  teacher << "seed,layers,selected_step,nll,perplexity,top1,top5,mean_rank,mean_margin,tokens,finite,used_for_selection\n";
  std::vector<Metrics> seedMetrics;
  for (const auto& checkpoint : checkpoints) {
    const Metrics metrics = evaluate(development, config, checkpoint.parameters, 512);
    if (!metrics.finite || !std::isfinite(metrics.nll) || !std::isfinite(metrics.perplexity) ||
        !std::isfinite(metrics.top1) || !std::isfinite(metrics.top5) ||
        !std::isfinite(metrics.meanRank) || !std::isfinite(metrics.meanMargin)) {
      throw std::runtime_error("DEVELOPMENT_EVALUATION_NONFINITE");
    }
    seedMetrics.push_back(metrics);
    teacher << checkpoint.seed << ',' << config.numLayers << ',' << checkpoint.step << ',';
    writeMetrics(teacher, metrics);
    teacher << ",false\n";
  }

  const std::size_t alignedChunks = std::min<std::size_t>(128, development.records.size());
  double teacherJs = 0, teacherAgreement = 0;
  std::uint64_t alignedTokens = 0;
  for (std::size_t chunk = 0; chunk < alignedChunks; ++chunk) {
    const auto input = inputTokens(development.records[chunk], config.tokens);
    std::array<Inference, 3> outputs{infer(config, input, checkpoints[0].parameters),
                                     infer(config, input, checkpoints[1].parameters),
                                     infer(config, input, checkpoints[2].parameters)};
    for (std::uint32_t row = 0; row < config.tokens; ++row) {
      const std::size_t base = std::size_t(row) * config.vocabularySize;
      teacherJs += (jsDivergence(outputs[0].probabilities.data() + base, outputs[1].probabilities.data() + base, config.vocabularySize) +
                    jsDivergence(outputs[0].probabilities.data() + base, outputs[2].probabilities.data() + base, config.vocabularySize) +
                    jsDivergence(outputs[1].probabilities.data() + base, outputs[2].probabilities.data() + base, config.vocabularySize)) / 3.0;
      const auto a = argmaxRow(outputs[0], config, row), b = argmaxRow(outputs[1], config, row), c = argmaxRow(outputs[2], config, row);
      teacherAgreement += (a == b && b == c);
      ++alignedTokens;
    }
  }

  const std::size_t promptCount = std::min<std::size_t>(16, development.records.size());
  constexpr std::uint32_t generationSteps = 16;
  double freeJs = 0, freeAgreement = 0, repeatCount = 0, loopCount = 0, invalidUtf8 = 0;
  double firstDivergenceSum = 0;
  std::uint64_t freePositions = 0;
  for (std::size_t prompt = 0; prompt < promptCount; ++prompt) {
    std::array<std::vector<std::uint32_t>, 3> contexts;
    std::array<std::vector<std::uint8_t>, 3> generated;
    for (auto& context : contexts) context = inputTokens(development.records[prompt], config.tokens);
    std::uint32_t firstDivergence = generationSteps;
    for (std::uint32_t step = 0; step < generationSteps; ++step) {
      std::array<Inference, 3> outputs{infer(config, contexts[0], checkpoints[0].parameters),
                                       infer(config, contexts[1], checkpoints[1].parameters),
                                       infer(config, contexts[2], checkpoints[2].parameters)};
      std::array<std::uint32_t, 3> predictions{};
      const std::size_t base = std::size_t(config.tokens - 1) * config.vocabularySize;
      for (std::size_t model = 0; model < 3; ++model) predictions[model] = argmaxRow(outputs[model], config, config.tokens - 1);
      freeJs += (jsDivergence(outputs[0].probabilities.data() + base, outputs[1].probabilities.data() + base, config.vocabularySize) +
                 jsDivergence(outputs[0].probabilities.data() + base, outputs[2].probabilities.data() + base, config.vocabularySize) +
                 jsDivergence(outputs[1].probabilities.data() + base, outputs[2].probabilities.data() + base, config.vocabularySize)) / 3.0;
      const bool agree = predictions[0] == predictions[1] && predictions[1] == predictions[2];
      freeAgreement += agree;
      if (!agree && firstDivergence == generationSteps) firstDivergence = step;
      for (std::size_t model = 0; model < 3; ++model) {
        const auto token = static_cast<std::uint8_t>(predictions[model]);
        repeatCount += !generated[model].empty() && generated[model].back() == token;
        loopCount += generated[model].size() >= 2 && generated[model][generated[model].size() - 2] == token;
        generated[model].push_back(token);
        std::rotate(contexts[model].begin(), contexts[model].begin() + 1, contexts[model].end());
        contexts[model].back() = token;
      }
      ++freePositions;
    }
    firstDivergenceSum += firstDivergence;
    for (const auto& bytes : generated) invalidUtf8 += !validUtf8(bytes);
  }

  struct SuffixStats { std::uint32_t count = 0; std::array<std::uint32_t, 256> targets{}; };
  std::unordered_map<std::string, SuffixStats> suffixStats;
  constexpr std::uint32_t suffix = 8;
  for (const auto& record : train.records) {
    std::string key(reinterpret_cast<const char*>(record.window.data() + config.tokens - suffix), suffix);
    auto& stats = suffixStats[key];
    ++stats.count;
    ++stats.targets[record.window[config.tokens]];
  }
  struct PairedMetrics {
    double logitDifference = 0;
    double nllDifference = 0;
    double attentionDifference = 0;
    std::uint64_t count = 0;
    std::uint64_t flips = 0;
    double identityMaximumLogitDifference = 0;
  };
  std::array<PairedMetrics, 3> pairedMetrics{};
  for (std::size_t model = 0; model < checkpoints.size(); ++model) {
    auto& metrics = pairedMetrics[model];
    for (std::size_t index = 0; index < development.records.size() && metrics.count < 128; ++index) {
      const auto& originalRecord = development.records[index];
      std::string key(reinterpret_cast<const char*>(originalRecord.window.data() + config.tokens - suffix), suffix);
      const auto found = suffixStats.find(key);
      if (found == suffixStats.end() || found->second.count < 4) continue;
      const auto modal = std::max_element(found->second.targets.begin(), found->second.targets.end());
      if (double(*modal) / found->second.count < 0.95 ||
          std::size_t(modal - found->second.targets.begin()) != originalRecord.window[config.tokens]) continue;
      const CacheRecord* donor = nullptr;
      for (std::size_t offset = 1; offset < development.records.size(); ++offset) {
        const auto& candidate = development.records[(index + offset) % development.records.size()];
        if (candidate.articleHash != originalRecord.articleHash) { donor = &candidate; break; }
      }
      if (!donor) continue;
      auto original = inputTokens(originalRecord, config.tokens);
      auto changed = original;
      for (std::uint32_t position = 0; position < config.tokens - suffix; ++position) changed[position] = donor->window[position];
      const auto originalOutput = infer(config, original, checkpoints[model].parameters);
      const auto identityOutput = infer(config, original, checkpoints[model].parameters);
      const auto changedOutput = infer(config, changed, checkpoints[model].parameters);
      const std::size_t base = std::size_t(config.tokens - 1) * config.vocabularySize;
      const std::uint32_t truth = originalRecord.window[config.tokens];
      double logitDiff = 0;
      for (std::uint32_t token = 0; token < config.vocabularySize; ++token) {
        logitDiff += std::abs(originalOutput.logits[base + token] - changedOutput.logits[base + token]);
        metrics.identityMaximumLogitDifference = std::max(
            metrics.identityMaximumLogitDifference,
            std::abs(double(originalOutput.logits[base + token]) - identityOutput.logits[base + token]));
      }
      metrics.logitDifference += logitDiff / config.vocabularySize;
      metrics.nllDifference += std::abs(-std::log(std::max(1e-30f, originalOutput.probabilities[base + truth])) +
                                        std::log(std::max(1e-30f, changedOutput.probabilities[base + truth])));
      metrics.flips += argmaxRow(originalOutput, config, config.tokens - 1) !=
                       argmaxRow(changedOutput, config, config.tokens - 1);
      double originalFar = 0, changedFar = 0;
      for (std::uint32_t head = 0; head < config.numHeads; ++head) {
        const std::size_t attentionBase = (std::size_t(head) * config.tokens + config.tokens - 1) * config.tokens;
        for (std::uint32_t position = 0; position < config.tokens - suffix; ++position) {
          originalFar += originalOutput.lastAttention[attentionBase + position];
          changedFar += changedOutput.lastAttention[attentionBase + position];
        }
      }
      metrics.attentionDifference += std::abs(originalFar - changedFar) / config.numHeads;
      ++metrics.count;
    }
    if (metrics.identityMaximumLogitDifference != 0) throw std::runtime_error("PAIRED_PREFIX_IDENTITY_CONTROL_FAILED");
  }

  std::ofstream stability(outputDirectory / "seed-stability.csv", std::ios::trunc);
  const auto nllValues = std::array<double, 3>{seedMetrics[0].nll, seedMetrics[1].nll, seedMetrics[2].nll};
  const double nllMinimum = *std::min_element(nllValues.begin(), nllValues.end());
  const double nllMaximum = *std::max_element(nllValues.begin(), nllValues.end());
  for (double value : {nllMinimum, nllMaximum, teacherAgreement, teacherJs, freeAgreement, freeJs,
                       firstDivergenceSum, repeatCount, loopCount, invalidUtf8}) {
    if (!std::isfinite(value)) throw std::runtime_error("COMPARE_AGGREGATE_NONFINITE");
  }
  stability << "layers,seeds,development_nll_min,development_nll_max,development_nll_range,teacher_argmax_agreement,teacher_pairwise_js,free_argmax_agreement,free_pairwise_js,mean_first_divergence_position,repeat_rate,short_loop_rate,invalid_utf8_generation_rate,generation_completion_rate,aligned_teacher_tokens,free_positions,final_test_used\n";
  stability << config.numLayers << ",1;2;4," << nllMinimum << ',' << nllMaximum << ',' << nllMaximum - nllMinimum << ','
            << teacherAgreement / std::max<std::uint64_t>(1, alignedTokens) << ',' << teacherJs / std::max<std::uint64_t>(1, alignedTokens) << ','
            << freeAgreement / std::max<std::uint64_t>(1, freePositions) << ',' << freeJs / std::max<std::uint64_t>(1, freePositions) << ','
            << firstDivergenceSum / std::max<std::size_t>(1, promptCount) << ','
            << repeatCount / std::max<double>(1, freePositions * 3.0) << ',' << loopCount / std::max<double>(1, freePositions * 3.0) << ','
            << invalidUtf8 / std::max<double>(1, promptCount * 3.0) << ",1," << alignedTokens << ',' << freePositions << ",false\n";
  std::ofstream paired(outputDirectory / "paired-prefix.csv", std::ios::trunc);
  paired << "seed,layers,suffix_bytes,eligibility_train_count,eligibility_target_concentration,pairs,mean_absolute_logit_difference,argmax_flip_rate,mean_absolute_nll_difference,mean_far_attention_mass_difference,identity_maximum_logit_difference,interpretation,final_test_used\n";
  for (std::size_t model = 0; model < checkpoints.size(); ++model) {
    const auto& metrics = pairedMetrics[model];
    paired << checkpoints[model].seed << ',' << config.numLayers << ',' << suffix << ",4,0.95," << metrics.count << ','
           << metrics.logitDifference / std::max<std::uint64_t>(1, metrics.count) << ','
           << double(metrics.flips) / std::max<std::uint64_t>(1, metrics.count) << ','
           << metrics.nllDifference / std::max<std::uint64_t>(1, metrics.count) << ','
           << metrics.attentionDifference / std::max<std::uint64_t>(1, metrics.count) << ','
           << metrics.identityMaximumLogitDifference << ",exploratory_semantic_invariance_not_assumed,false\n";
  }
  std::cout << "compare_status=PASS layers=" << config.numLayers << " development_nll_range=" << nllMaximum - nllMinimum
            << " paired_prefix_pairs_per_seed=" << pairedMetrics.front().count << '\n';
}

void runCheckpointAudit(const fs::path& outputPath, const std::vector<fs::path>& checkpointPaths) {
  if (checkpointPaths.size() != 3) throw std::runtime_error("CHECKPOINT_AUDIT_REQUIRES_THREE");
  const std::array<std::uint32_t, 3> expectedSeeds{1, 2, 4};
  std::ofstream output(outputPath, std::ios::trunc);
  if (!output) throw std::runtime_error("CHECKPOINT_AUDIT_OUTPUT_OPEN_FAILED");
  output << "seed,layers,step,parameter_hash,finite\n";
  std::uint32_t expectedLayers = 0;
  for (std::size_t index = 0; index < checkpointPaths.size(); ++index) {
    const Checkpoint checkpoint = loadCheckpoint(checkpointPaths[index]);
    if (checkpoint.seed != expectedSeeds[index]) throw std::runtime_error("CHECKPOINT_AUDIT_SEED_ORDER");
    if (!expectedLayers) expectedLayers = checkpoint.config.numLayers;
    if (checkpoint.config.vocabularySize != 256 || checkpoint.config.tokens != 32 ||
        checkpoint.config.dimension != 16 || checkpoint.config.feedForwardDimension != 32 ||
        checkpoint.config.numHeads != 2 || checkpoint.config.numLayers != expectedLayers ||
        (expectedLayers != 6 && expectedLayers != 19) || !parametersFinite(checkpoint.parameters)) {
      throw std::runtime_error("CHECKPOINT_AUDIT_CONFIG_OR_FINITE");
    }
    output << checkpoint.seed << ',' << checkpoint.config.numLayers << ',' << checkpoint.step << ','
           << parameterHash(checkpoint.parameters) << ",true\n";
  }
  std::cout << "checkpoint_audit=PASS layers=" << expectedLayers << " checkpoints=3\n";
}

void writeFixtureCache(const fs::path& path, std::uint32_t context) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write("NPRTBYTEV1\n", 11);
  writeU32(output, context);
  writeU32(output, 256);
  writeU64(output, 8);
  for (std::uint64_t record = 0; record < 8; ++record) {
    writeU64(output, record + 1);
    for (std::uint32_t token = 0; token <= context; ++token) {
      output.put(static_cast<char>((record * 17 + token * 3) & 255));
    }
  }
}

void selfTest() {
  const fs::path directory = fs::temp_directory_path() / "phonelm-nicopedia-real-text-self-test";
  fs::remove_all(directory);
  fs::create_directories(directory);
  const fs::path cachePath = directory / "fixture.bin";
  writeFixtureCache(cachePath, 8);
  const Cache cache = loadCache(cachePath);
  if (cache.context != 8 || cache.vocabulary != 256 || cache.records.size() != 8) throw std::runtime_error("SELFTEST_CACHE");
  const auto input = inputTokens(cache.records.front(), cache.context);
  const auto target = targetTokens(cache.records.front(), cache.context);
  for (std::size_t i = 1; i < input.size(); ++i) if (input[i] != target[i - 1]) throw std::runtime_error("SELFTEST_TARGET_SHIFT");
  const lm::Config config = makeConfig(8, 2);
  const Parameters parameters = lm::initialParameters(config, 1);
  const auto fast = infer(config, input, parameters);
  const auto reference = lm::forwardBackwardGeneralized(config, lm::oneHot(input, 256), lm::oneHot(target, 256), parameters, 0.0f);
  if (fast.logits.size() != reference.logits.size()) throw std::runtime_error("SELFTEST_INFERENCE_SIZE");
  double maximumDifference = 0;
  for (std::size_t i = 0; i < fast.logits.size(); ++i) maximumDifference = std::max(maximumDifference, std::abs(double(fast.logits[i]) - reference.logits[i]));
  if (maximumDifference > 1e-5 || !allFinite(fast.logits) || !allFinite(reference.logits)) throw std::runtime_error("SELFTEST_INFERENCE_PARITY");
  if (parameterHash(parameters) == parameterHash(lm::initialParameters(config, 2))) throw std::runtime_error("SELFTEST_SEED_IDENTITY");
  const auto firstOrder = trainingOrder(cache.records.size(), 10, 2, 20260806);
  const auto secondOrder = trainingOrder(cache.records.size(), 10, 2, 20260806);
  if (firstOrder != secondOrder || orderHash(firstOrder) != orderHash(secondOrder)) throw std::runtime_error("SELFTEST_ORDER_IDENTITY");
  const fs::path checkpointPath = directory / "private.ckpt";
  saveCheckpoint(checkpointPath, Checkpoint{config, 1, 3, parameters});
  const Checkpoint replay = loadCheckpoint(checkpointPath);
  if (replay.step != 3 || replay.seed != 1 || parameterHash(replay.parameters) != parameterHash(parameters)) throw std::runtime_error("SELFTEST_CHECKPOINT_IDENTITY");
  const Metrics metrics = evaluate(cache, config, replay.parameters, 2);
  if (!metrics.finite || !std::isfinite(metrics.nll) || metrics.tokens != 16) throw std::runtime_error("SELFTEST_METRICS");
  if (!betterValidation(4.0, 40, 5.0, 20) || betterValidation(4.0, 60, 4.0, 40)) {
    throw std::runtime_error("SELFTEST_VALIDATION_SELECTION");
  }
  fs::remove_all(directory);
  std::cout << "nicopedia_real_text_pilot_self_test=PASS\n";
}

std::uint32_t parseU32(const char* text, const char* label) {
  try {
    const unsigned long value = std::stoul(text);
    if (value > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("range");
    return static_cast<std::uint32_t>(value);
  } catch (...) {
    throw std::runtime_error(std::string("INVALID_") + label);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      selfTest();
      return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "--benchmark") {
      runBenchmark(argv[2], argv[3], parseU32(argv[4], "LAYERS"), parseU32(argv[5], "STEPS"));
      return 0;
    }
    if (argc == 10 && std::string(argv[1]) == "--train") {
      runTraining(argv[2], argv[3], argv[4], parseU32(argv[5], "SEED"), parseU32(argv[6], "LAYERS"),
                  parseU32(argv[7], "STEPS"), parseU32(argv[8], "BATCH"), parseU32(argv[9], "INTERVAL"));
      return 0;
    }
    if (argc == 8 && std::string(argv[1]) == "--compare") {
      runCompare(argv[2], argv[3], argv[4], {argv[5], argv[6], argv[7]});
      return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "--checkpoint-audit") {
      runCheckpointAudit(argv[2], {argv[3], argv[4], argv[5]});
      return 0;
    }
    std::cerr << "usage: --self-test | --benchmark CACHE OUT LAYERS STEPS | "
                 "--train TRAIN VALIDATION OUT SEED LAYERS STEPS BATCH INTERVAL | "
                 "--compare TRAIN DEVELOPMENT OUT CHECKPOINT1 CHECKPOINT2 CHECKPOINT4 | "
                 "--checkpoint-audit OUT CHECKPOINT1 CHECKPOINT2 CHECKPOINT4\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "nicopedia_real_text_pilot_error=" << error.what() << '\n';
    return 1;
  }
}
