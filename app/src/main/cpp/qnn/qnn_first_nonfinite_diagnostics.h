// Private, fail-closed diagnostics for reproducing a first non-finite value.
// This format is intentionally not a public experiment artifact.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace phonelm::qnn::first_nonfinite {

constexpr std::uint32_t kCheckpointVersion = 1;

struct Config {
  std::uint32_t tokens = 0, vocabularySize = 0, dimension = 0;
  std::uint32_t feedForwardDimension = 0, numLayers = 0, numHeads = 0;
  float epsilon = 0.0f, learningRate = 0.0f, beta1 = 0.9f, beta2 = 0.999f;
  float adamEpsilon = 1.0e-8f, clipThreshold = 0.0f;
};

struct RegistryEntry {
  std::string name;
  std::vector<std::uint32_t> shape;
};

// values are stored in canonical registry order.  Keeping the registry beside
// each state makes a stale checkpoint fail instead of silently binding a new
// graph's parameter layout.
struct Checkpoint {
  std::uint32_t version = kCheckpointVersion;
  Config config;
  std::uint32_t seed = 0, completedStep = 0, nextOptimizerStep = 1;
  std::string deterministicState;
  std::vector<RegistryEntry> registry;
  std::vector<float> input, target, parameters, adamM, adamV;
  std::string registryHash, stateHash;
};

struct TensorSummary {
  std::vector<std::uint32_t> shape;
  std::size_t count = 0, finite = 0, nan = 0, positiveInfinity = 0;
  std::size_t negativeInfinity = 0;
  double minimum = 0.0, maximum = 0.0, maximumAbsolute = 0.0;
  double mean = 0.0, meanAbsolute = 0.0, rms = 0.0, minimumNonzero = 0.0;
  std::string hash;
};

struct TensorComparison {
  double maximumAbsolute = 0.0, meanAbsolute = 0.0, relativeL2 = 0.0;
  std::size_t argmax = 0, firstDifferent = static_cast<std::size_t>(-1);
  std::vector<std::size_t> top3;
};

struct NamedTensor {
  std::string name;
  std::vector<std::uint32_t> shape;
  const std::vector<float>* values = nullptr;
};

struct FirstBadTensor {
  std::string name = "NONE";
  std::size_t flatIndex = static_cast<std::size_t>(-1);
  TensorSummary summary;
};

// These scopes grow observation from a small graph-preserving boundary set to
// subblocks and finally operator-level values. NONE must retain the old ABI.
enum class TapScope : std::uint32_t {
  NONE = 0,
  COARSE_LAYER_BOUNDARIES = 1,
  LAYER_SUBBLOCKS = 2,
  LAYER_OPS = 3,
};

struct TapPlan {
  TapScope scope = TapScope::NONE;
  std::vector<RegistryEntry> taps;
  std::size_t untappedNodeCount = 0, tappedNodeCount = 0;
};

struct ObserverEffect {
  bool originalMatch = false;
  bool observerEffect = false;
  std::string untappedFinalHash, tappedFinalHash;
};

std::string hashRegistry(const std::vector<RegistryEntry>& registry);
std::string hashCheckpointState(const Checkpoint& checkpoint);
void finalizeCheckpoint(Checkpoint* checkpoint);
bool validateCheckpoint(const Checkpoint& checkpoint, std::string* error);
bool encodeCheckpoint(const Checkpoint& checkpoint, std::vector<std::uint8_t>* bytes,
                      std::string* error);
bool decodeCheckpoint(const std::vector<std::uint8_t>& bytes, Checkpoint* checkpoint,
                      std::string* error, const Config* expectedConfig = nullptr,
                      const std::vector<RegistryEntry>* expectedRegistry = nullptr);
std::string checkpointText(const Checkpoint& checkpoint);

TensorSummary summarize(const std::vector<float>& values,
                        const std::vector<std::uint32_t>& shape = {});
TensorComparison compare(const std::vector<float>& expected,
                         const std::vector<float>& actual);
FirstBadTensor firstBad(const std::vector<NamedTensor>& ordered);
bool validateTapPlan(const TapPlan& plan, std::string* error);
ObserverEffect classifyObserverEffect(const std::vector<float>& untappedFinal,
                                      const std::vector<float>& tappedFinal,
                                      bool originalMatch);

}  // namespace phonelm::qnn::first_nonfinite
