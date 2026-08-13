// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host-only validation of the NPRTCKPTV2 canonical-resume contract used by
// the HTP NICOPEDIA training mode:
//
//  1. V2 save/load round-trip preserves parameters, Adam first/second moments
//     and the step bit-for-bit; V1 loads as parameters-only (hasAdam=false).
//  2. A run split at step k (scratch 0..k, checkpoint, resume k+1..steps) is
//     bit-identical to a single from-scratch run 1..steps when every
//     optimizer call uses the global step (bias corrections and the record
//     order are functions of the global step; the record cursor is re-derived
//     as order[(globalStep-1)*batch + batch]).
//  3. Resuming from a V1 checkpoint (no Adam moments) fails closed.
//  4. Step/seed mismatches are rejected.
//  5. A failed first resumed step is ineligible for a final write, preserving
//     the existing canonical resume checkpoint byte-for-byte.
//
// The C++ mirrors the device implementation in
// app/src/main/cpp/qnn/qnn_transformer_training.cpp
// (nprtSaveCheckpointV2, nprtLoadCheckpointForGeneration, nicopediaHtpTraining);
// the reference math is tiny_lm::adamUpdate, the same function the device's
// CPU replay uses.
#include "nicopedia_checkpoint_policy.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using phonelm::tiny_lm::Config;
using Params = phonelm::qnn::TinyTransformerParameters;

std::uint64_t splitMix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

// Mirrors nprtTrainingOrder: the record order is a pure function of the
// global step index, so a resume re-derives the cursor from the step.
std::vector<std::size_t> trainingOrder(std::size_t recordCount, uint32_t steps,
                                       uint32_t batchSize) {
  std::vector<std::size_t> order;
  order.reserve(std::size_t(steps) * batchSize);
  std::uint64_t state = 20260806;
  for (std::size_t i = 0; i < std::size_t(steps) * batchSize; ++i) {
    state = splitMix(state + i);
    order.push_back(static_cast<std::size_t>(state % recordCount));
  }
  return order;
}

struct Cache {
  std::uint32_t context = 0;
  std::vector<std::vector<std::uint8_t>> windows;
};

Cache buildCache(std::uint32_t context, std::uint32_t vocabulary,
                 std::size_t recordCount) {
  Cache cache;
  cache.context = context;
  std::uint64_t state = 0x1234567890ABCDEFull;
  for (std::size_t r = 0; r < recordCount; ++r) {
    std::vector<std::uint8_t> window(context + 1);
    for (auto& byte : window) {
      state = splitMix(state + 1);
      byte = std::uint8_t(state % vocabulary);
    }
    cache.windows.push_back(std::move(window));
  }
  return cache;
}

Params zeroParams(const Params& source) {
  Params result = source;
  for (const auto& info : phonelm::tiny_lm::parameterRegistry(result)) {
    auto& values = *const_cast<std::vector<float>*>(info.values);
    std::fill(values.begin(), values.end(), 0.0f);
  }
  return result;
}

void writeU32(std::ostream& output, std::uint32_t value) {
  std::array<std::uint8_t, 4> bytes{std::uint8_t(value >> 24),
                                    std::uint8_t(value >> 16),
                                    std::uint8_t(value >> 8),
                                    std::uint8_t(value)};
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

void writeU64(std::ostream& output, std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (int i = 7; i >= 0; --i) {
    bytes[std::size_t(i)] = std::uint8_t(value & 0xFF);
    value >>= 8;
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::uint32_t readU32(std::istream& input) {
  std::array<std::uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("TRUNCATED_U32");
  return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
         (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

std::uint64_t readU64(std::istream& input) {
  std::array<std::uint8_t, 8> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("TRUNCATED_U64");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes) value = (value << 8) | byte;
  return value;
}

// Mirrors nprtSaveCheckpoint / nprtSaveCheckpointV2 layout exactly.
void saveCheckpoint(const std::string& path, const Config& config,
                    std::uint32_t seed, std::uint32_t step,
                    const Params& parameters, bool withAdam, const Params& first,
                    const Params& second) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("OPEN_FAILED");
  output.write(withAdam ? "NPRTCKPTV2\n" : "NPRTCKPTV1\n", 11);
  writeU32(output, config.vocabularySize);
  writeU32(output, config.tokens);
  writeU32(output, config.dimension);
  writeU32(output, config.feedForwardDimension);
  writeU32(output, config.numLayers);
  writeU32(output, config.numHeads);
  writeU32(output, seed);
  writeU32(output, step);
  const auto registry = phonelm::tiny_lm::parameterRegistry(parameters);
  const auto firstRegistry = phonelm::tiny_lm::parameterRegistry(first);
  const auto secondRegistry = phonelm::tiny_lm::parameterRegistry(second);
  if (withAdam && (registry.size() != firstRegistry.size() ||
                   registry.size() != secondRegistry.size()))
    throw std::runtime_error("REGISTRY_MISMATCH");
  auto writeRegistry = [&output](const auto& entries) {
    writeU32(output, static_cast<std::uint32_t>(entries.size()));
    for (const auto& info : entries) {
      writeU32(output, static_cast<std::uint32_t>(info.name.size()));
      output.write(info.name.data(), static_cast<std::streamsize>(info.name.size()));
      writeU64(output, info.values->size());
      output.write(reinterpret_cast<const char*>(info.values->data()),
                   static_cast<std::streamsize>(info.values->size() * sizeof(float)));
    }
  };
  writeRegistry(registry);
  if (withAdam) {
    writeRegistry(firstRegistry);
    writeRegistry(secondRegistry);
  }
  if (!output) throw std::runtime_error("WRITE_FAILED");
}

struct Loaded {
  Config config;
  std::uint32_t seed = 0, step = 0;
  bool hasAdam = false;
  Params parameters;
  Params first;
  Params second;
};

// Mirrors nprtLoadCheckpointForGeneration (which accepts both magics).
Loaded loadCheckpoint(const std::string& path, const Config& expected,
                      std::uint32_t expectedSeed, std::uint32_t expectedStep,
                      bool requireAdam) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("OPEN_FAILED");
  std::string magic(11, '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  const bool v2 = magic == "NPRTCKPTV2\n";
  if (magic != "NPRTCKPTV1\n" && !v2) throw std::runtime_error("MAGIC");
  Loaded loaded;
  loaded.config.vocabularySize = readU32(input);
  loaded.config.tokens = readU32(input);
  loaded.config.dimension = readU32(input);
  loaded.config.feedForwardDimension = readU32(input);
  loaded.config.numLayers = readU32(input);
  loaded.config.numHeads = readU32(input);
  loaded.seed = readU32(input);
  loaded.step = readU32(input);
  if (loaded.config.vocabularySize != expected.vocabularySize ||
      loaded.config.tokens != expected.tokens ||
      loaded.config.dimension != expected.dimension ||
      loaded.config.feedForwardDimension != expected.feedForwardDimension ||
      loaded.config.numLayers != expected.numLayers ||
      loaded.config.numHeads != expected.numHeads)
    throw std::runtime_error("CONFIG_MISMATCH");
  if (loaded.seed != expectedSeed) throw std::runtime_error("SEED_MISMATCH");
  if (loaded.step != expectedStep) throw std::runtime_error("STEP_MISMATCH");
  loaded.parameters = phonelm::tiny_lm::initialParameters(expected, loaded.seed);
  loaded.first = zeroParams(loaded.parameters);
  loaded.second = zeroParams(loaded.parameters);
  // Shape registry is only consulted for names/counts (device behavior);
  // decoded groups are assigned to the destination by order afterwards.
  const Params shape = phonelm::tiny_lm::initialParameters(expected, 1);
  const auto shapeRegistry = phonelm::tiny_lm::parameterRegistry(shape);
  auto readRaw = [&]() -> std::vector<std::vector<float>> {
    const std::uint32_t count = readU32(input);
    if (count != shapeRegistry.size()) throw std::runtime_error("REGISTRY_COUNT");
    std::vector<std::vector<float>> groups;
    groups.reserve(count);
    for (std::uint32_t group = 0; group < count; ++group) {
      const std::uint32_t nameLength = readU32(input);
      if (nameLength > 256) throw std::runtime_error("NAME_LENGTH");
      std::string name(nameLength, '\0');
      input.read(name.data(), static_cast<std::streamsize>(name.size()));
      const std::uint64_t values = readU64(input);
      if (!input || name != shapeRegistry[group].name ||
          values != shapeRegistry[group].values->size())
        throw std::runtime_error("REGISTRY_IDENTITY");
      std::vector<float> raw(values);
      input.read(reinterpret_cast<char*>(raw.data()),
                 static_cast<std::streamsize>(raw.size() * sizeof(float)));
      if (!input) throw std::runtime_error("VALUES_TRUNCATED");
      if (!std::all_of(raw.begin(), raw.end(),
                       [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("VALUES_NONFINITE");
      groups.push_back(std::move(raw));
    }
    return groups;
  };
  const auto assignTo = [&](Params& destination,
                            const std::vector<std::vector<float>>& groups) {
    const auto destinationRegistry =
        phonelm::tiny_lm::parameterRegistry(destination);
    for (std::size_t i = 0; i < groups.size(); ++i)
      *const_cast<std::vector<float>*>(destinationRegistry[i].values) =
          groups[i];
  };
  assignTo(loaded.parameters, readRaw());
  loaded.hasAdam = v2;
  if (v2) {
    assignTo(loaded.first, readRaw());
    assignTo(loaded.second, readRaw());
  }
  if (input.get() != std::char_traits<char>::eof())
    throw std::runtime_error("TRAILING_BYTES");
  if (requireAdam && !loaded.hasAdam) throw std::runtime_error("ADAM_MISSING");
  return loaded;
}

bool paramsBitEqual(const Params& a, const Params& b) {
  const auto ra = phonelm::tiny_lm::parameterRegistry(a);
  const auto rb = phonelm::tiny_lm::parameterRegistry(b);
  if (ra.size() != rb.size()) return false;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    if (ra[i].name != rb[i].name || ra[i].values->size() != rb[i].values->size())
      return false;
    if (std::memcmp(ra[i].values->data(), rb[i].values->data(),
                    ra[i].values->size() * sizeof(float)) != 0)
      return false;
  }
  return true;
}

std::vector<char> readBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("OPEN_FAILED");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

struct TrainState {
  Params parameters;
  Params first;
  Params second;
};

// Mirrors the device training loop (nicopediaHtpTraining): global steps
// resumeStep+1..steps, per-batch gradient averages with 1/batch weighting and
// Adam with global-step bias corrections. A resume changes the initial state,
// never the per-step formulas.
TrainState runTrain(const Config& config, const Cache& cache,
                    const std::vector<std::size_t>& order, uint32_t steps,
                    uint32_t batchSize, float lr, TrainState state,
                    uint32_t resumeStep) {
  for (uint32_t step = resumeStep + 1; step <= steps; ++step) {
    Params gradientAccum = zeroParams(state.parameters);
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
      const std::size_t recordIndex =
          order[std::size_t(step - 1) * batchSize + batch];
      const auto& window = cache.windows.at(recordIndex);
      std::vector<std::uint32_t> input(config.tokens), target(config.tokens);
      for (std::uint32_t i = 0; i < config.tokens; ++i) {
        input[i] = window[i];
        target[i] = window[i + 1];
      }
      const auto stepResult = phonelm::tiny_lm::forwardBackwardGeneralized(
          config, phonelm::tiny_lm::oneHot(input, config.vocabularySize),
          phonelm::tiny_lm::oneHot(target, config.vocabularySize),
          state.parameters, 0.0f);
      const auto accumulator = phonelm::tiny_lm::parameterRegistry(gradientAccum);
      const auto gradientRegistry =
          phonelm::tiny_lm::parameterRegistry(stepResult.gradients);
      for (std::size_t i = 0; i < accumulator.size(); ++i) {
        auto& accum = *const_cast<std::vector<float>*>(accumulator[i].values);
        const auto& values = *gradientRegistry[i].values;
        for (std::size_t j = 0; j < accum.size(); ++j)
          accum[j] += values[j] * (1.0f / float(batchSize));
      }
    }
    const float c1 = float(1.0 / (1.0 - std::pow(0.9, double(step))));
    const float c2 = float(1.0 / (1.0 - std::pow(0.999, double(step))));
    const auto result = phonelm::tiny_lm::adamUpdate(
        state.parameters, gradientAccum, state.first, state.second, lr, 0.9f,
        0.999f, 1e-8f, c1, c2);
    state.parameters = result.next;
    state.first = result.firstMoment;
    state.second = result.secondMoment;
  }
  return state;
}

}  // namespace

int main() {
  int failures = 0;
  try {
    // Checkpoint filename identity contract.
    //
    // T32/D32/FFN32 is the production anchor and exclusively owns the
    // canonical untagged filename. All non-anchor configurations, including
    // historical/research D16, must carry explicit T/D/FFN tags.
    {
      using phonelm::nicopedia_checkpoint::checkpointName;
      using phonelm::nicopedia_checkpoint::parseCheckpointStep;

      const std::string d32Canonical =
          checkpointName(1, 19, 32, 32, 32, 8000);
      if (d32Canonical != "htp-seed1-l19-step8000.ckpt") {
        std::cerr << "D32 canonical checkpoint name mismatch: "
                  << d32Canonical << '\n';
        ++failures;
      }

      const std::string d16Tagged =
          checkpointName(1, 19, 32, 16, 32, 8000);
      if (d16Tagged !=
          "htp-seed1-l19-t32-d16-f32-step8000.ckpt") {
        std::cerr << "D16 tagged checkpoint name mismatch: "
                  << d16Tagged << '\n';
        ++failures;
      }

      std::uint32_t parsedStep = 0;

      // D32 canonical untagged => ACCEPT.
      if (!parseCheckpointStep(
              "htp-seed1-l19-step8000.ckpt",
              1, 19, &parsedStep,
              32, 32, 32) ||
          parsedStep != 8000) {
        std::cerr << "D32 canonical checkpoint must parse\n";
        ++failures;
      }

      // Default parser identity must also be T32/D32/FFN32.
      parsedStep = 0;
      if (!parseCheckpointStep(
              "htp-seed1-l19-step8000.ckpt",
              1, 19, &parsedStep) ||
          parsedStep != 8000) {
        std::cerr << "default checkpoint identity must be D32 anchor\n";
        ++failures;
      }

      // D16 explicitly tagged => ACCEPT.
      parsedStep = 0;
      if (!parseCheckpointStep(
              "htp-seed1-l19-t32-d16-f32-step8000.ckpt",
              1, 19, &parsedStep,
              32, 16, 32) ||
          parsedStep != 8000) {
        std::cerr << "D16 tagged checkpoint must parse\n";
        ++failures;
      }

      // D16 untagged => REJECT.
      if (parseCheckpointStep(
              "htp-seed1-l19-step8000.ckpt",
              1, 19, &parsedStep,
              32, 16, 32)) {
        std::cerr << "D16 untagged checkpoint must be rejected\n";
        ++failures;
      }

      // T mismatch => REJECT.
      if (parseCheckpointStep(
              "htp-seed1-l19-t16-d16-f32-step8000.ckpt",
              1, 19, &parsedStep,
              32, 16, 32)) {
        std::cerr << "checkpoint token mismatch must be rejected\n";
        ++failures;
      }

      // D mismatch => REJECT.
      if (parseCheckpointStep(
              "htp-seed1-l19-t32-d32-f32-step8000.ckpt",
              1, 19, &parsedStep,
              32, 16, 32)) {
        std::cerr << "checkpoint dimension mismatch must be rejected\n";
        ++failures;
      }

      // FFN mismatch => REJECT.
      if (parseCheckpointStep(
              "htp-seed1-l19-t32-d16-f64-step8000.ckpt",
              1, 19, &parsedStep,
              32, 16, 32)) {
        std::cerr << "checkpoint FFN mismatch must be rejected\n";
        ++failures;
      }
    }

    Config config;
    config.vocabularySize = 16;
    config.tokens = 8;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    config.numLayers = 2;
    config.numHeads = 2;
    const std::uint32_t seed = 1;
    const Cache cache = buildCache(config.tokens, config.vocabularySize, 512);
    const std::uint32_t steps = 40, batchSize = 2;
    const float lr = 0.003f;
    const std::vector<std::size_t> order =
        trainingOrder(cache.windows.size(), steps, batchSize);

    TrainState initialState;
    initialState.parameters = phonelm::tiny_lm::initialParameters(config, seed);
    initialState.first = zeroParams(initialState.parameters);
    initialState.second = zeroParams(initialState.parameters);

    // One-shot from-scratch run 1..steps.
    const TrainState full =
        runTrain(config, cache, order, steps, batchSize, lr, initialState, 0);

    // Scratch 0..12 -> V2 checkpoint -> resume 13..steps.
    TrainState splitState =
        runTrain(config, cache, order, 12, batchSize, lr, initialState, 0);
    const std::string v2Path = "nprt_resume_checkpoint_v2.ckpt";
    saveCheckpoint(v2Path, config, seed, 12, splitState.parameters, true,
                   splitState.first, splitState.second);
    const Loaded loadedOnce = loadCheckpoint(v2Path, config, seed, 12, true);
    if (!loadedOnce.hasAdam) {
      std::cerr << "V2 must load hasAdam=true\n";
      ++failures;
    }
    if (!paramsBitEqual(loadedOnce.parameters, splitState.parameters) ||
        !paramsBitEqual(loadedOnce.first, splitState.first) ||
        !paramsBitEqual(loadedOnce.second, splitState.second)) {
      std::cerr << "V2 round trip is not bit-identical\n";
      ++failures;
    }

    // Fault boundary: resume step 13 returned but was non-finite, so the
    // completed cursor remains 12 while the in-memory state may already have
    // changed. The canonical step-12 input must not be replaced under either
    // a non-finite-state or non-finite-loss failure.
    const auto canonicalBefore = readBytes(v2Path);
    const bool nonfiniteStateWrite =
        phonelm::nicopedia_checkpoint::finalWriteAllowed(
            /*allStepsFinite=*/false, /*finalStateFinite=*/false,
            /*lastCompletedStep=*/12, /*requestedStep=*/steps);
    const bool nonfiniteLossWrite =
        phonelm::nicopedia_checkpoint::finalWriteAllowed(
            /*allStepsFinite=*/false, /*finalStateFinite=*/true,
            /*lastCompletedStep=*/12, /*requestedStep=*/steps);
    if (nonfiniteStateWrite || nonfiniteLossWrite) {
      std::cerr << "failed first resumed step must not be checkpoint-eligible\n";
      ++failures;
    }
    // The production final block calls save only when the shared policy says
    // true. Simulating the false branch therefore leaves the existing file
    // untouched; compare the complete private payload, not only parameters.
    const auto canonicalAfter = readBytes(v2Path);
    if (canonicalAfter != canonicalBefore) {
      std::cerr << "failed resume changed the canonical checkpoint bytes\n";
      ++failures;
    }
    if (!phonelm::nicopedia_checkpoint::finalWriteAllowed(
            /*allStepsFinite=*/true, /*finalStateFinite=*/true,
            /*lastCompletedStep=*/steps, /*requestedStep=*/steps)) {
      std::cerr << "completed finite target must remain checkpoint-eligible\n";
      ++failures;
    }
    TrainState resumed = {loadedOnce.parameters, loadedOnce.first,
                          loadedOnce.second};
    resumed = runTrain(config, cache, order, steps, batchSize, lr, resumed, 12);
    if (!paramsBitEqual(resumed.parameters, full.parameters) ||
        !paramsBitEqual(resumed.first, full.first) ||
        !paramsBitEqual(resumed.second, full.second)) {
      std::cerr << "split+resume diverges from the one-shot run\n";
      ++failures;
    }

    // Chain: scratch 0..12 -> resume to 25 -> checkpoint -> resume to steps.
    TrainState chainedState =
        runTrain(config, cache, order, 12, batchSize, lr, initialState, 0);
    chainedState =
        runTrain(config, cache, order, 25, batchSize, lr, chainedState, 12);
    const std::string v2bPath = "nprt_resume_checkpoint_v2b.ckpt";
    saveCheckpoint(v2bPath, config, seed, 25, chainedState.parameters, true,
                   chainedState.first, chainedState.second);
    const Loaded loadedChain = loadCheckpoint(v2bPath, config, seed, 25, true);
    TrainState chained = {loadedChain.parameters, loadedChain.first,
                          loadedChain.second};
    chained =
        runTrain(config, cache, order, steps, batchSize, lr, chained, 25);
    if (!paramsBitEqual(chained.parameters, full.parameters) ||
        !paramsBitEqual(chained.first, full.first) ||
        !paramsBitEqual(chained.second, full.second)) {
      std::cerr << "chained resume diverges from the one-shot run\n";
      ++failures;
    }

    // V1 (parameters only) loads but must fail closed for resume.
    const std::string v1Path = "nprt_resume_checkpoint_v1.ckpt";
    saveCheckpoint(v1Path, config, seed, 12, splitState.parameters, false,
                   zeroParams(splitState.parameters),
                   zeroParams(splitState.parameters));
    const Loaded v1Loaded =
        loadCheckpoint(v1Path, config, seed, 12, /*requireAdam=*/false);
    if (v1Loaded.hasAdam) {
      std::cerr << "V1 must load hasAdam=false\n";
      ++failures;
    }
    bool adamRejected = false;
    try {
      loadCheckpoint(v1Path, config, seed, 12, /*requireAdam=*/true);
    } catch (const std::exception&) {
      adamRejected = true;
    }
    if (!adamRejected) {
      std::cerr << "resuming from V1 must be rejected\n";
      ++failures;
    }

    // A V2 header is insufficient resume evidence: non-finite Adam state is
    // rejected even when the parameter registry itself is finite.
    Params nonfiniteFirst = splitState.first;
    auto nonfiniteRegistry = phonelm::tiny_lm::parameterRegistry(nonfiniteFirst);
    (*const_cast<std::vector<float>*>(nonfiniteRegistry.front().values)).front() =
        std::numeric_limits<float>::quiet_NaN();
    const std::string nonfiniteV2Path = "nprt_resume_checkpoint_v2_nonfinite.ckpt";
    saveCheckpoint(nonfiniteV2Path, config, seed, 12, splitState.parameters,
                   true, nonfiniteFirst, splitState.second);
    bool nonfiniteAdamRejected = false;
    try {
      loadCheckpoint(nonfiniteV2Path, config, seed, 12, true);
    } catch (const std::exception&) {
      nonfiniteAdamRejected = true;
    }
    if (!nonfiniteAdamRejected) {
      std::cerr << "V2 non-finite Adam state must be rejected\n";
      ++failures;
    }

    // Step mismatch must be rejected.
    bool stepRejected = false;
    try {
      loadCheckpoint(v1Path, config, seed, 99, false);
    } catch (const std::exception&) {
      stepRejected = true;
    }
    if (!stepRejected) {
      std::cerr << "step mismatch must be rejected\n";
      ++failures;
    }

    // Seed mismatch must be rejected.
    bool seedRejected = false;
    try {
      loadCheckpoint(v1Path, config, seed + 1, 12, false);
    } catch (const std::exception&) {
      seedRejected = true;
    }
    if (!seedRejected) {
      std::cerr << "seed mismatch must be rejected\n";
      ++failures;
    }

    // Fail-closed cross-load: a checkpoint built with one context length
    // (config.tokens) must be rejected when validated against an expected
    // config with a DIFFERENT tokens value, never silently accepted under
    // the wrong context window.
    bool tokensRejected = false;
    {
      Config wrongTokens = config;
      wrongTokens.tokens = config.tokens + 8;  // deliberately different
      try {
        loadCheckpoint(v2Path, wrongTokens, seed, 12, /*requireAdam=*/true);
      } catch (const std::runtime_error& error) {
        tokensRejected = std::string(error.what()) == "CONFIG_MISMATCH";
      }
    }
    if (!tokensRejected) {
      std::cerr << "tokens mismatch must be rejected fail-closed\n";
      ++failures;
    }

    // D and FFN are checkpoint identity fields too.  A research checkpoint
    // must never be resumed under a same-T/L/H configuration with a different
    // embedding or FFN width.
    bool dimensionRejected = false;
    {
      Config wrongDimension = config;
      wrongDimension.dimension = config.dimension + 2;
      try {
        loadCheckpoint(v2Path, wrongDimension, seed, 12, /*requireAdam=*/true);
      } catch (const std::runtime_error& error) {
        dimensionRejected = std::string(error.what()) == "CONFIG_MISMATCH";
      }
    }
    if (!dimensionRejected) {
      std::cerr << "dimension mismatch must be rejected fail-closed\n";
      ++failures;
    }
    bool ffnRejected = false;
    {
      Config wrongFfn = config;
      wrongFfn.feedForwardDimension = config.feedForwardDimension + 8;
      try {
        loadCheckpoint(v2Path, wrongFfn, seed, 12, /*requireAdam=*/true);
      } catch (const std::runtime_error& error) {
        ffnRejected = std::string(error.what()) == "CONFIG_MISMATCH";
      }
    }
    if (!ffnRejected) {
      std::cerr << "FFN mismatch must be rejected fail-closed\n";
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "nicopedia_resume_test_error=" << error.what() << '\n';
    return 1;
  }
  std::cout << "nicopedia_resume_test=" << (failures == 0 ? "PASS" : "FAIL")
            << "\nresume_bit_identity=" << (failures == 0 ? "true" : "false")
            << '\n';
  return failures == 0 ? 0 : 1;
}
