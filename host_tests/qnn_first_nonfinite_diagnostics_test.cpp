// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host contract for private qnn_first_nonfinite diagnostics tooling:
// checkpoint codec fail-closed behavior, tensor summaries, tap plans,
// observer-effect classification, and CPU replay determinism.
// Coverage moved from qnn_sdk_independent_test.cpp and depth_quality_test.cpp
// without changing fixture values or assertion semantics.
#include "qnn/qnn_first_nonfinite_diagnostics.h"
#include "tiny_language_model_cpu.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fnd = phonelm::qnn::first_nonfinite;

namespace {

using phonelm::tiny_lm::Config;

static Config smallConfig(std::uint32_t layers = 2, std::uint32_t heads = 2) {
  Config c;
  c.tokens = 8;
  c.vocabularySize = 32;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = layers;
  c.numHeads = heads;
  return c;
}

static fnd::Checkpoint makeCheckpoint(const Config& config, std::uint32_t seed,
                                      std::uint32_t step) {
  const auto params = phonelm::tiny_lm::initialParameters(config, seed);
  fnd::Checkpoint result;
  result.config = {config.tokens, config.vocabularySize, config.dimension,
                   config.feedForwardDimension, config.numLayers,
                   config.numHeads, config.epsilon, 0.003f, .9f, .999f, 1e-8f,
                   0.0f, 0, 0, 320};
  result.seed = seed;
  result.completedStep = step;
  result.nextOptimizerStep = step + 1;
  result.deterministicState = "fixed_language_batch=" + std::to_string(step % 4);
  for (const auto& e : phonelm::tiny_lm::parameterRegistry(params)) {
    result.registry.push_back({e.name, {std::uint32_t(e.values->size())}});
    result.parameters.insert(result.parameters.end(), e.values->begin(),
                             e.values->end());
  }
  result.adamM.assign(result.parameters.size(), 0.0f);
  result.adamV.assign(result.parameters.size(), 0.0f);
  result.input.assign(size_t(config.tokens) * config.vocabularySize, 0.0f);
  result.target = result.input;
  result.input[0] = 1.0f;
  result.target[config.vocabularySize + 1] = 1.0f;
  fnd::finalizeCheckpoint(&result);
  return result;
}

// Moved from qnn_sdk_independent_test.cpp (testFirstNonfiniteDiagnosticCodecAndSummaries).
void testDiagnosticCodecAndSummaries() {
    fnd::Checkpoint checkpoint;
    checkpoint.config = {2, 3, 2, 4, 1, 1, 1.0e-5f, 0.003f,
                         0.9f, 0.999f, 1.0e-8f, 0.0f};
    checkpoint.seed = 2;
    checkpoint.completedStep = 31;
    checkpoint.nextOptimizerStep = 32;
    checkpoint.deterministicState = "fixed_batch=3";
    checkpoint.registry = {{"layer_000.norm1_gamma", {2}},
                           {"layer_000.ffn_w1", {2, 4}}};
    checkpoint.input = {1, 0, 0, 0, 1, 0};
    checkpoint.target = {0, 1, 0, 1, 0, 0};
    checkpoint.parameters.resize(10);
    checkpoint.adamM.resize(10);
    checkpoint.adamV.resize(10);
    for (std::size_t i = 0; i < checkpoint.parameters.size(); ++i) {
        checkpoint.parameters[i] = float(i + 1);
        checkpoint.adamM[i] = float(i) * 0.1f;
        checkpoint.adamV[i] = float(i) * 0.01f;
    }
    fnd::finalizeCheckpoint(&checkpoint);
    std::string error;
    assert(fnd::validateCheckpoint(checkpoint, &error));
    std::vector<std::uint8_t> encoded;
    assert(fnd::encodeCheckpoint(checkpoint, &encoded, &error));
    fnd::Checkpoint decoded;
    assert(fnd::decodeCheckpoint(encoded, &decoded, &error,
                                 &checkpoint.config, &checkpoint.registry));
    assert(decoded.stateHash == checkpoint.stateHash);
    encoded[12] ^= 1;
    assert(!fnd::decodeCheckpoint(encoded, &decoded, &error));
    assert(error == "checkpoint checksum");
    encoded[12] ^= 1;
    auto mismatched = checkpoint.config;
    ++mismatched.dimension;
    assert(!fnd::decodeCheckpoint(encoded, &decoded, &error, &mismatched));
    assert(error == "checkpoint configuration mismatch");

    const std::vector<float> values{1.0f, std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(), -2.0f};
    const auto summary = fnd::summarize(values, {5});
    assert(summary.count == 5 && summary.finite == 2 && summary.nan == 1);
    assert(summary.positiveInfinity == 1 && summary.negativeInfinity == 1);
    assert(summary.minimum == -2.0 && summary.maximum == 1.0);
    const auto first = fnd::firstBad({{"finite", {1}, &checkpoint.input},
                                      {"bad", {5}, &values}});
    assert(first.name == "bad" && first.flatIndex == 1);
    const auto comparison = fnd::compare({1.0f, 2.0f, 3.0f}, {1.0f, 9.0f, 1.0f});
    assert(comparison.argmax == 1 && comparison.firstDifferent == 1);
    assert(comparison.top3.size() == 3 && comparison.top3[0] == 1);
    const fnd::TapPlan valid{fnd::TapScope::COARSE_LAYER_BOUNDARIES,
                             {{"layer_002_output", {2, 2}}}, 10, 1};
    assert(fnd::validateTapPlan(valid, &error));
    const fnd::TapPlan invalid{fnd::TapScope::COARSE_LAYER_BOUNDARIES,
                               {{"duplicate", {2}}, {"duplicate", {2}}}, 0, 2};
    assert(!fnd::validateTapPlan(invalid, &error));
    const auto observer = fnd::classifyObserverEffect({1.0f}, {2.0f}, true);
    assert(observer.originalMatch && observer.observerEffect);
}

// Moved from qnn_sdk_independent_test.cpp (testFirstNonfiniteCpuReplayDeterminism).
void testCpuReplayDeterminism() {
    phonelm::tiny_lm::Config config;
    config.tokens = 2;
    config.vocabularySize = 4;
    config.dimension = 4;
    config.feedForwardDimension = 6;
    const auto parameters = phonelm::tiny_lm::initialParameters(config, 7);
    const auto input = phonelm::tiny_lm::oneHot({0, 1}, config.vocabularySize);
    const auto target = phonelm::tiny_lm::oneHot({1, 2}, config.vocabularySize);
    const auto first = phonelm::tiny_lm::forwardBackward(
        config, input, target, parameters, 0.0f);
    const auto replay = phonelm::tiny_lm::forwardBackward(
        config, input, target, parameters, 0.0f);
    assert(first.logits == replay.logits);
    assert(first.dLogits == replay.dLogits);
}

// Moved from depth_quality_test.cpp (makeCheckpoint + testCheckpointCodecV2).
void testCheckpointCodecV2() {
  const auto config = smallConfig(2, 2);
  const auto checkpoint = makeCheckpoint(config, 2, 32);
  std::vector<std::uint8_t> bytes;
  std::string error;
  assert(fnd::encodeCheckpoint(checkpoint, &bytes, &error));
  fnd::Checkpoint decoded;
  assert(fnd::decodeCheckpoint(bytes, &decoded, &error));
  assert(decoded.config.trainingStabilityMode == 0);
  assert(decoded.config.depthPairInitMode == 0);
  assert(decoded.config.totalSteps == 320);
  assert(decoded.stateHash == checkpoint.stateHash);

  // seeded checkpoint carries a different hash; same-seed roundtrip passes
  const auto otherSeed = makeCheckpoint(config, 3, 32);
  assert(otherSeed.stateHash != checkpoint.stateHash);

  // registry mismatch (different depth) rejected fail-closed
  const auto deep = makeCheckpoint(smallConfig(3, 2), 2, 32);
  fnd::Checkpoint out;
  assert(!fnd::decodeCheckpoint(bytes, &out, &error, &deep.config, &deep.registry));

  // stability mode out of range rejected fail-closed
  auto badMode = checkpoint;
  badMode.config.trainingStabilityMode = 7;
  fnd::finalizeCheckpoint(&badMode);
  std::vector<std::uint8_t> badBytes;
  assert(!fnd::encodeCheckpoint(badMode, &badBytes, &error));

  // checkpoint version 1 payload rejected fail-closed
  std::vector<std::uint8_t> downgraded = bytes;
  // version is the uint32 after the 4-byte magic header
  std::memcpy(downgraded.data() + 4, "\x01\x00\x00\x00", 4);
  fnd::Checkpoint any;
  assert(!fnd::decodeCheckpoint(downgraded, &any, &error));
}

}  // namespace

int main() {
    testDiagnosticCodecAndSummaries();
    testCpuReplayDeterminism();
    testCheckpointCodecV2();
    std::cout << "qnn_first_nonfinite_diagnostics_test=PASS\n";
    return 0;
}
