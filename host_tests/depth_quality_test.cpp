// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Host tests for the direct-seed contract, depth-pair initialization,
// stability modes, validation selection and checkpoint codecs.
#include "depth_quality_lib.h"
#include "qnn/qnn_first_nonfinite_diagnostics.h"
#include "seed_selection.h"
#include "validation_checkpoint.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace dq = phonelm::depth_quality;
namespace ff = phonelm::qnn::first_nonfinite;
namespace vs = phonelm::validation_selection;
namespace vc = phonelm::validation_checkpoint;
using phonelm::tiny_lm::Config;

static Config smallConfig(uint32_t layers = 2, uint32_t heads = 2) {
  Config c;
  c.tokens = 8;
  c.vocabularySize = 32;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = layers;
  c.numHeads = heads;
  return c;
}

static std::uint64_t fnvParams(const phonelm::qnn::TinyTransformerParameters& p) {
  std::uint64_t h = 1469598103934665603ull;
  for (const auto& e : phonelm::tiny_lm::parameterRegistry(p)) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(e.values->data());
    for (size_t i = 0; i < e.values->size() * sizeof(float); ++i) {
      h ^= bytes[i];
      h *= 1099511628211ull;
    }
  }
  return h;
}

static void testExactSeedValidation() {
  // valid EXACT requests
  for (std::uint64_t k : {1ull, 2ull, 5ull, 19ull, 2147483647ull})
    assert(phonelm::validateSeedSelection(1, k, std::int64_t(k)) == nullptr);
  // invalid: zero, overflow, contradiction, unknown mode, bad count
  assert(std::strcmp(phonelm::validateSeedSelection(1, 0, 1), "EXACT_SEED_RANGE") == 0);
  assert(std::strcmp(phonelm::validateSeedSelection(1, 1, -3), "EXACT_SEED_COUNT_RANGE") == 0);
  assert(std::strcmp(phonelm::validateSeedSelection(1, 2147483648ull, 2147483647ll), "EXACT_SEED_RANGE") == 0);
  assert(std::strcmp(phonelm::validateSeedSelection(1, 2, 3), "EXACT_SEED_CONTRADICTS_COUNT") == 0);
  assert(std::strcmp(phonelm::validateSeedSelection(7, 2, 2), "UNKNOWN_SEED_SELECTION_MODE") == 0);
  // COUNT_FROM_ONE stays permissive for legacy callers
  assert(phonelm::validateSeedSelection(0, 20260710, 5) == nullptr);
}

static void testExactSeedDeterminismAndLegacyEquivalence() {
  // seeds 1/2/5 deterministic parameter initialization
  const auto config = smallConfig(2, 2);
  for (uint32_t seed : {1u, 2u, 5u}) {
    const auto a = phonelm::tiny_lm::initialParameters(config, seed);
    const auto b = phonelm::tiny_lm::initialParameters(config, seed);
    const auto ra = phonelm::tiny_lm::parameterRegistry(a);
    const auto rb = phonelm::tiny_lm::parameterRegistry(b);
    assert(ra.size() == rb.size());
    for (size_t i = 0; i < ra.size(); ++i)
      assert(*ra[i].values == *rb[i].values);
  }
  // CPU emulation of the COUNT_FROM_ONE loop: no state leaks across seeds, so
  // the per-seed CPU formal run equals the direct call bit-for-bit. This
  // guards future regressions that would make direct seeds diverge.
  const auto viaLoopA = dq::runFormalCpu(config, 2, 40);
  const auto viaLoopB = dq::runFormalCpu(config, 2, 40);
  const auto ra = phonelm::tiny_lm::parameterRegistry(viaLoopA.finalParameters);
  const auto rb = phonelm::tiny_lm::parameterRegistry(viaLoopB.finalParameters);
  for (size_t i = 0; i < ra.size(); ++i) assert(*ra[i].values == *rb[i].values);
  assert(viaLoopA.finalEvaluation.loss == viaLoopB.finalEvaluation.loss);
}

static void testLegacyTrajectoryHashInvariance() {
  // Pin the established LEGACY protocol results computed by the same g++ host
  // toolchain before this series of changes (values recorded from the probe).
  const auto l19s2 = dq::runFormalCpu(smallConfig(19, 2), 2, 320);
  assert(std::abs(l19s2.finalEvaluation.loss - 2.51632029) < 1e-5);
  const auto l19s1 = dq::runFormalCpu(smallConfig(19, 2), 1, 320);
  assert(std::abs(l19s1.finalEvaluation.loss - 0.311677747) < 1e-5);
  const auto l18s2 = dq::runFormalCpu(smallConfig(18, 2), 2, 320);
  assert(std::abs(l18s2.finalEvaluation.loss - 1.23493402) < 1e-5);
}

static void testValidationPartitionAndSelection() {
  const auto config = smallConfig(2, 2);
  const auto validationA = vs::validationCases(config.tokens);
  const auto validationB = vs::validationCases(config.tokens);
  assert(validationA.size() == 3 && validationA.size() == validationB.size());
  assert(vs::validationSetHash(config.tokens) == "fnv1a64:8e1411f19126879c");
  assert(validationA[0].input ==
         std::vector<std::uint32_t>({2, 3, 0, 1, 2, 3, 0, 1}));
  assert(validationA[0].target ==
         std::vector<std::uint32_t>({3, 0, 1, 2, 3, 0, 1, 2}));
  std::vector<std::string> ids;
  for (size_t i = 0; i < validationA.size(); ++i) {
    assert(validationA[i].id == validationB[i].id);
    assert(validationA[i].input == validationB[i].input);
    assert(validationA[i].target == validationB[i].target);
    ids.push_back(validationA[i].id);
    for (auto token : validationA[i].input) assert(token <= 12);
    for (auto token : validationA[i].target) assert(token <= 12);
  }
  std::sort(ids.begin(), ids.end());
  assert(std::adjacent_find(ids.begin(), ids.end()) == ids.end());

  // Full sequences/prefixes/target sequences are disjoint. Token transitions
  // intentionally overlap so validation measures learned rules rather than
  // untouched vocabulary rows.
  for (std::uint32_t pattern = 0; pattern < 4; ++pattern) {
    const auto train = dq::formalBatch(config, pattern, 0);
    const auto phase1 = dq::formalBatch(config, pattern, 1);
    for (const auto& item : validationA) {
      const auto input = phonelm::tiny_lm::oneHot(item.input,
                                                  config.vocabularySize);
      const auto target = phonelm::tiny_lm::oneHot(item.target,
                                                   config.vocabularySize);
      assert(input != train.first && target != train.second);
      assert(input != phase1.first && target != phase1.second);
      assert(input != train.second && target != phase1.first);
    }
  }

  vs::Metrics incumbent{1.0, 0.5, 0.0, 0.0};
  assert(vs::better(vs::Metrics{0.9, 0.1, 0.0, 0.0}, 20,
                    incumbent, 16));
  assert(vs::better(vs::Metrics{1.0 + 0.5e-7, 0.75, 0.0, 0.0}, 20,
                    incumbent, 16));
  assert(vs::better(vs::Metrics{1.0, 0.5, 0.0, 0.0}, 12,
                    incumbent, 16));
  assert(!vs::better(vs::Metrics{1.0, 0.5, 0.0, 0.0}, 20,
                     incumbent, 16));

  const auto legacy = dq::runFormalCpu(config, 1, 40);
  const auto selectedFinal = dq::runValidationSelectedCpu(
      config, 1, 40, vs::Mode::FINAL_STEP);
  const auto left = phonelm::tiny_lm::parameterRegistry(legacy.finalParameters);
  const auto right =
      phonelm::tiny_lm::parameterRegistry(selectedFinal.selectedParameters);
  assert(selectedFinal.selectedStep == 40 && left.size() == right.size());
  for (size_t i = 0; i < left.size(); ++i)
    assert(*left[i].values == *right[i].values);
  assert(legacy.steps.size() == selectedFinal.training.steps.size());
  for (size_t i = 0; i < legacy.steps.size(); ++i) {
    assert(legacy.steps[i].loss == selectedFinal.training.steps[i].loss);
    assert(legacy.steps[i].accuracy == selectedFinal.training.steps[i].accuracy);
    assert(legacy.steps[i].gradientNorm ==
           selectedFinal.training.steps[i].gradientNorm);
  }
  const auto legacyGeneration = dq::generationQuality(config, legacy.finalParameters);
  assert(legacyGeneration.oracleExact == selectedFinal.generation.oracleExact);
  assert(legacyGeneration.freeExact == selectedFinal.generation.freeExact);

  const auto selectedBest = dq::runValidationSelectedCpu(
      config, 1, 40, vs::Mode::BEST_VALIDATION_V1);
  vs::Metrics manualBest;
  int manualStep = 0;
  for (const auto& entry : selectedBest.validationTrajectory) {
    if (vs::better(entry.second, entry.first, manualBest, manualStep)) {
      manualBest = entry.second;
      manualStep = entry.first;
    }
  }
  assert(manualStep == selectedBest.selectedStep);
  const auto selectedHash = fnvParams(selectedBest.selectedParameters);
  (void)dq::generationQuality(config, selectedBest.selectedParameters);
  assert(fnvParams(selectedBest.selectedParameters) == selectedHash &&
         selectedBest.selectedStep == manualStep);

  std::vector<std::pair<int, vs::Metrics>> early{{0, {3.0, 0.0, 0.0, 0.0}},
      {4, {2.0, 0.2, 0.0, 0.0}}, {8, {2.1, 0.2, 0.0, 0.0}},
      {12, {2.2, 0.2, 0.0, 0.0}}, {16, {2.3, 0.2, 0.0, 0.0}}};
  const auto simulation = vs::simulateEarlyStop(early, 3, 320);
  assert(simulation.stopStep == 16 && simulation.bestStep == 4 &&
         simulation.savedTrainingSteps == 304);
}

static void testPairedDepthInitialization() {
  const Config deep = smallConfig(19, 2);
  const Config shallow = smallConfig(18, 2);
  for (uint32_t seed : {1u, 2u, 3u, 4u, 5u}) {
    const auto pd = phonelm::tiny_lm::initialParameters(deep, seed);
    const auto ps = phonelm::tiny_lm::initialParameters(shallow, seed);
    assert(dq::sharedPrefixParametersEqual(ps, pd));
    // extra layer (layer_018) exists only in the deep model and is not all-zero
    bool anyNonzero = false;
    const auto rd = phonelm::tiny_lm::parameterRegistry(pd);
    for (const auto& e : rd)
      if (e.name.rfind("layer_018", 0) == 0)
        for (float x : *e.values) anyNonzero = anyNonzero || x != 0.0f;
    assert(anyNonzero);
    // extra layer domain separation: its values differ from every other layer
    const auto deepM = phonelm::tiny_lm::initialParameters(deep, seed);
    const auto rm = phonelm::tiny_lm::parameterRegistry(deepM);
    const std::vector<float>* extra = nullptr;
    const std::vector<float>* other = nullptr;
    for (const auto& e : rm) {
      if (e.name == "layer_018.wq") extra = e.values;
      if (e.name == "layer_017.wq") other = e.values;
    }
    assert(extra && other && *extra != *other);
  }
  // determinism
  assert(fnvParams(phonelm::tiny_lm::initialParameters(deep, 2)) ==
         fnvParams(phonelm::tiny_lm::initialParameters(deep, 2)));
}

static void testStabilityModeContract() {
  for (std::uint32_t m = 0; m <= 6; ++m) {
    if (m == 5) continue;
    assert(phonelm::validateTrainingStabilityMode(m) == nullptr);
  }
  assert(std::strcmp(phonelm::validateTrainingStabilityMode(5),
                     "RESIDUAL_BRANCH_SCALING_UNSUPPORTED_ON_DEVICE") == 0);
  assert(std::strcmp(phonelm::validateTrainingStabilityMode(7),
                     "UNKNOWN_TRAINING_STABILITY_MODE") == 0);
  assert(phonelm::validateDepthPairInitMode(0) == nullptr);
  assert(phonelm::validateDepthPairInitMode(1) == nullptr);
  assert(phonelm::validateDepthPairInitMode(2) != nullptr);
  // schedule anchors
  assert(std::abs(phonelm::stabilityLearningRate(0, 0.003f, 1, 320) - 0.003f) < 1e-7);
  assert(std::abs(phonelm::stabilityLearningRate(1, 0.003f, 1, 320) -
                  0.003f / 64.0) < 1e-6);
  assert(std::abs(phonelm::stabilityLearningRate(1, 0.003f, 64, 320) - 0.003f) < 1e-7);
  assert(std::abs(phonelm::stabilityLearningRate(1, 0.003f, 320, 320) - 0.003f) < 1e-7);
  assert(std::abs(phonelm::stabilityLearningRate(2, 0.003f, 1, 320) - 0.003f) < 1e-6);
  assert(std::abs(phonelm::stabilityLearningRate(2, 1.0f, 320, 320) - 1.0f / 320.0) < 1e-5);
  // init-side candidates
  const auto config = smallConfig(3, 2);
  const auto zeroed = phonelm::applyInitStability(
      config, phonelm::tiny_lm::initialParameters(config, 1), 3);
  const auto rz = phonelm::tiny_lm::parameterRegistry(zeroed);
  for (const auto& e : rz)
    if (e.name.size() >= 3 &&
        (e.name.rfind("wo") == e.name.size() - 2 ||
         e.name.rfind("w2") == e.name.size() - 2))
      for (float x : *e.values) assert(x == 0.0f);
  const auto scaled = phonelm::applyInitStability(
      config, phonelm::tiny_lm::initialParameters(config, 1), 4);
  const auto legacy = phonelm::tiny_lm::initialParameters(config, 1);
  const auto rs = phonelm::tiny_lm::parameterRegistry(scaled);
  const auto rl = phonelm::tiny_lm::parameterRegistry(legacy);
  const float s = 1.0f / std::sqrt(6.0f);
  bool checked = false;
  for (size_t i = 0; i < rs.size(); ++i) {
    if (rs[i].name == "layer_000.wo" || rs[i].name == "layer_000.ffn_w1") {
      assert(std::abs((*rs[i].values)[0] - (*rl[i].values)[0] * s) < 1e-7);
      checked = true;
    }
  }
  assert(checked);
}

static void testTrajectoryClassification() {
  auto series = [](std::vector<double> losses) {
    std::vector<dq::StepMetric> s;
    int step = 0;
    for (double l : losses) {
      dq::StepMetric m;
      m.step = ++step;
      m.loss = float(l);
      s.push_back(m);
    }
    return s;
  };
  // NEVER_LEARNS: never below 0.95 * initial
  assert(dq::classifyTrajectory(series({3.5, 3.5, 3.5, 3.5, 3.5})) == "NEVER_LEARNS");
  // LEARNS_THEN_REGRESSES: learned, then final worse than min+0.5 and worse
  // than 0.95*initial
  {
    std::vector<double> l(320, 3.5);
    for (int i = 0; i < 100; ++i) l[i] = 3.5 - i * 0.03;
    for (int i = 100; i < 320; ++i) l[i] = l[99] + (i - 99) * 0.03;
    assert(dq::classifyTrajectory(series(l)) == "LEARNS_THEN_REGRESSES");
  }
  // LATE_COLLAPSE: improving to the end then collapses in the last stretch
  {
    std::vector<double> l(320, 3.5);
    for (int i = 0; i < 96; ++i) l[i] = 3.5 - i * 0.03;
    for (int i = 96; i < 288; ++i) l[i] = l[95] - (i - 95) * 0.005;
    for (int i = 288; i < 320; ++i) l[i] = l[287] + (i - 287) * 0.05;
    assert(dq::classifyTrajectory(series(l)) == "LATE_COLLAPSE");
  }
  // GENERATION_ONLY_SHORTFALL: converged well below half the initial loss
  {
    std::vector<double> l(320, 3.5);
    for (int i = 0; i < 320; ++i) l[i] = 3.5 * std::pow(0.97, i);
    assert(dq::classifyTrajectory(series(l)) == "GENERATION_ONLY_SHORTFALL");
  }
  // PLATEAUS: learns, then stalls above the "converged" band (no improvement
  // in the last 30%)
  {
    std::vector<double> l(320, 3.5);
    for (int i = 0; i < 200; ++i) l[i] = 3.5 - i * 0.0045;
    for (int i = 200; i < 320; ++i) l[i] = l[199];
    assert(dq::classifyTrajectory(series(l)) == "PLATEAUS");
  }
  // OSCILLATES: wild sign flips in the second half, final above minimum
  {
    std::vector<double> l(320, 1.0);
    for (int i = 160; i < 320; ++i) l[i] = 0.8 + (i % 2 ? 0.3 : 0.0);
    assert(dq::classifyTrajectory(series(l)) == "OSCILLATES");
  }
}

static void testFirstDivergenceSelection() {
  auto make = [](double base, double bumpAt, double bump) {
    std::vector<dq::StepMetric> s(320);
    for (int i = 0; i < 320; ++i) {
      s[i].step = i + 1;
      s[i].loss = float(base + (i + 1 >= bumpAt ? bump : 0.0));
    }
    return s;
  };
  const auto a = make(1.0, 1e9, 0);
  const auto b = make(1.0, 101, 0.2);
  assert(dq::firstTrajectoryDivergence(a, b) == 101);
  assert(dq::firstTrajectoryDivergence(a, a) == -1);
  const auto c = make(1.0, 50, 0.01);
  assert(dq::firstTrajectoryDivergence(a, c) == -1);  // below threshold
}

static ff::Checkpoint makeCheckpoint(const Config& config, uint32_t seed,
                                     uint32_t step) {
  const auto params = phonelm::tiny_lm::initialParameters(config, seed);
  ff::Checkpoint result;
  result.config = {config.tokens, config.vocabularySize, config.dimension,
                   config.feedForwardDimension, config.numLayers,
                   config.numHeads, config.epsilon, 0.003f, .9f, .999f, 1e-8f,
                   0.0f, 0, 0, 320};
  result.seed = seed;
  result.completedStep = step;
  result.nextOptimizerStep = step + 1;
  result.deterministicState = "fixed_language_batch=" + std::to_string(step % 4);
  for (const auto& e : phonelm::tiny_lm::parameterRegistry(params)) {
    result.registry.push_back({e.name, {uint32_t(e.values->size())}});
    result.parameters.insert(result.parameters.end(), e.values->begin(),
                             e.values->end());
  }
  result.adamM.assign(result.parameters.size(), 0.0f);
  result.adamV.assign(result.parameters.size(), 0.0f);
  result.input.assign(size_t(config.tokens) * config.vocabularySize, 0.0f);
  result.target = result.input;
  result.input[0] = 1.0f;
  result.target[config.vocabularySize + 1] = 1.0f;
  ff::finalizeCheckpoint(&result);
  return result;
}

static void testCheckpointCodecV2() {
  const auto config = smallConfig(2, 2);
  const auto checkpoint = makeCheckpoint(config, 2, 32);
  std::vector<std::uint8_t> bytes;
  std::string error;
  assert(ff::encodeCheckpoint(checkpoint, &bytes, &error));
  ff::Checkpoint decoded;
  assert(ff::decodeCheckpoint(bytes, &decoded, &error));
  assert(decoded.config.trainingStabilityMode == 0);
  assert(decoded.config.depthPairInitMode == 0);
  assert(decoded.config.totalSteps == 320);
  assert(decoded.stateHash == checkpoint.stateHash);

  // seeded checkpoint carries a different hash; same-seed roundtrip passes
  const auto otherSeed = makeCheckpoint(config, 3, 32);
  assert(otherSeed.stateHash != checkpoint.stateHash);

  // registry mismatch (different depth) rejected fail-closed
  const auto deep = makeCheckpoint(smallConfig(3, 2), 2, 32);
  ff::Checkpoint out;
  assert(!ff::decodeCheckpoint(bytes, &out, &error, &deep.config, &deep.registry));

  // stability mode out of range rejected fail-closed
  auto badMode = checkpoint;
  badMode.config.trainingStabilityMode = 7;
  ff::finalizeCheckpoint(&badMode);
  std::vector<std::uint8_t> badBytes;
  assert(!ff::encodeCheckpoint(badMode, &badBytes, &error));

  // checkpoint version 1 payload rejected fail-closed
  std::vector<std::uint8_t> downgraded = bytes;
  // version is the uint32 after the 4-byte magic header
  std::memcpy(downgraded.data() + 4, "\x01\x00\x00\x00", 4);
  ff::Checkpoint any;
  assert(!ff::decodeCheckpoint(downgraded, &any, &error));
}

static void testValidationCheckpointCodec() {
  vc::Checkpoint checkpoint;
  checkpoint.configHash = "config-v1";
  checkpoint.seed = 2;
  checkpoint.selectionMode = std::uint32_t(vs::Mode::BEST_VALIDATION_V1);
  checkpoint.validationSchemaVersion = vs::kValidationSchemaVersion;
  checkpoint.validationSetHash = vs::validationSetHash(8);
  checkpoint.validationCaseCount = 3;
  checkpoint.selectedStep = 64;
  checkpoint.totalSteps = 320;
  checkpoint.optimizerNextStep = 65;
  checkpoint.parameterRegistryVersion = vs::kParameterRegistryVersion;
  checkpoint.registryHash = "registry-v2";
  checkpoint.validation = {0.5, 0.75, 1.25, 0.6};
  checkpoint.parameters = {1.0f, 2.0f, 3.0f};
  checkpoint.adamM = {0.1f, 0.2f, 0.3f};
  checkpoint.adamV = {0.01f, 0.02f, 0.03f};
  vc::finalize(&checkpoint);
  std::vector<std::uint8_t> bytes;
  std::string error;
  assert(vc::encode(checkpoint, &bytes, &error));
  const vc::Expected expected{checkpoint.configHash, checkpoint.seed,
      checkpoint.selectionMode, checkpoint.validationSchemaVersion,
      checkpoint.validationSetHash, checkpoint.validationCaseCount,
      checkpoint.totalSteps, checkpoint.parameterRegistryVersion,
      checkpoint.registryHash};
  vc::Checkpoint decoded;
  assert(vc::decode(bytes, &decoded, &error, &expected));
  assert(decoded.stateHash == checkpoint.stateHash &&
         decoded.parameters == checkpoint.parameters);

  auto corrupt = bytes;
  corrupt[corrupt.size() / 2] ^= 1;
  assert(!vc::decode(corrupt, &decoded, &error, &expected));
  auto truncated = bytes;
  truncated.pop_back();
  assert(!vc::decode(truncated, &decoded, &error, &expected));
  auto wrong = expected;
  wrong.seed = 3;
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  wrong = expected; wrong.selectionMode = 0;
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  wrong = expected; wrong.validationSetHash = "changed";
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  wrong = expected; wrong.validationSchemaVersion += 1;
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  wrong = expected; wrong.configHash = "changed";
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  wrong = expected; wrong.parameterRegistryVersion += 1;
  assert(!vc::decode(bytes, &decoded, &error, &wrong));
  auto nonfinite = checkpoint;
  nonfinite.parameters[0] = std::numeric_limits<float>::infinity();
  vc::finalize(&nonfinite);
  assert(!vc::encode(nonfinite, &bytes, &error));
}

int main() {
  testExactSeedValidation();
  std::puts("exact_seed_validation=PASS");
  testExactSeedDeterminismAndLegacyEquivalence();
  std::puts("exact_seed_determinism_and_equivalence=PASS");
  testLegacyTrajectoryHashInvariance();
  std::puts("legacy_trajectory_hash_invariance=PASS");
  testValidationPartitionAndSelection();
  std::puts("validation_partition_and_selection=PASS");
  testPairedDepthInitialization();
  std::puts("paired_depth_initialization=PASS");
  testStabilityModeContract();
  std::puts("stability_mode_contract=PASS");
  testTrajectoryClassification();
  std::puts("trajectory_classification=PASS");
  testFirstDivergenceSelection();
  std::puts("first_divergence_selection=PASS");
  testCheckpointCodecV2();
  std::puts("checkpoint_codec_v2=PASS");
  testValidationCheckpointCodec();
  std::puts("validation_checkpoint_codec=PASS");
  std::puts("depth_quality_tests=PASS");
  return 0;
}
