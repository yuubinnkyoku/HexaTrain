#include "qnn_first_nonfinite_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace phonelm::qnn::first_nonfinite {
namespace {
constexpr std::uint32_t kMagic = 0x31464e51u;  // QNF1, little endian.
constexpr std::size_t kMaxBytes = size_t(1) << 30;
constexpr std::size_t kMaxString = size_t(1) << 20;
constexpr std::size_t kMaxEntries = 4096;

std::uint64_t fnv(const void* data, std::size_t size, std::uint64_t seed = 1469598103934665603ull) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) { seed ^= bytes[i]; seed *= 1099511628211ull; }
  return seed;
}
template <typename T> void hashValue(std::uint64_t* hash, const T& value) {
  *hash = fnv(&value, sizeof(value), *hash);
}
void hashString(std::uint64_t* hash, const std::string& value) {
  const std::uint64_t size = value.size(); hashValue(hash, size);
  *hash = fnv(value.data(), value.size(), *hash);
}
std::string hex(std::uint64_t value) {
  std::ostringstream stream; stream << std::hex << std::setw(16) << std::setfill('0') << value; return stream.str();
}
bool sameConfig(const Config& a, const Config& b) {
  return a.tokens == b.tokens && a.vocabularySize == b.vocabularySize &&
      a.dimension == b.dimension && a.feedForwardDimension == b.feedForwardDimension &&
      a.numLayers == b.numLayers && a.numHeads == b.numHeads &&
      a.epsilon == b.epsilon && a.learningRate == b.learningRate &&
      a.beta1 == b.beta1 && a.beta2 == b.beta2 &&
      a.adamEpsilon == b.adamEpsilon && a.clipThreshold == b.clipThreshold;
}
bool sameRegistry(const std::vector<RegistryEntry>& a, const std::vector<RegistryEntry>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) if (a[i].name != b[i].name || a[i].shape != b[i].shape) return false;
  return true;
}
bool elementCount(const std::vector<RegistryEntry>& registry, std::size_t* count) {
  *count = 0;
  for (const auto& entry : registry) {
    std::size_t local = 1;
    for (std::uint32_t dimension : entry.shape) {
      if (dimension == 0 || local > std::numeric_limits<std::size_t>::max() / dimension) return false;
      local *= dimension;
    }
    if (*count > std::numeric_limits<std::size_t>::max() - local) return false;
    *count += local;
  }
  return true;
}
class Writer {
 public:
  explicit Writer(std::vector<std::uint8_t>* bytes) : bytes_(bytes) {}
  template <typename T> void pod(const T& value) { const auto* p = reinterpret_cast<const std::uint8_t*>(&value); bytes_->insert(bytes_->end(), p, p + sizeof(value)); }
  void string(const std::string& value) { const std::uint64_t size = value.size(); pod(size); bytes_->insert(bytes_->end(), value.begin(), value.end()); }
  void floats(const std::vector<float>& values) { const std::uint64_t size = values.size(); pod(size); const auto* p = reinterpret_cast<const std::uint8_t*>(values.data()); bytes_->insert(bytes_->end(), p, p + values.size() * sizeof(float)); }
 private: std::vector<std::uint8_t>* bytes_;
};
class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
  template <typename T> bool pod(T* value) { if (remaining() < sizeof(T)) return false; std::memcpy(value, bytes_.data() + offset_, sizeof(T)); offset_ += sizeof(T); return true; }
  bool string(std::string* value) { std::uint64_t size = 0; if (!pod(&size) || size > kMaxString || size > remaining()) return false; value->assign(reinterpret_cast<const char*>(bytes_.data() + offset_), size); offset_ += size; return true; }
  bool floats(std::vector<float>* values) { std::uint64_t count = 0; if (!pod(&count) || count > kMaxBytes / sizeof(float) || count > remaining() / sizeof(float)) return false; values->resize(size_t(count)); std::memcpy(values->data(), bytes_.data() + offset_, values->size() * sizeof(float)); offset_ += values->size() * sizeof(float); return true; }
  std::size_t remaining() const { return offset_ <= bytes_.size() ? bytes_.size() - offset_ : 0; }
 private: const std::vector<std::uint8_t>& bytes_; std::size_t offset_ = 0;
};
void writeConfig(Writer* writer, const Config& c) {
  writer->pod(c.tokens); writer->pod(c.vocabularySize); writer->pod(c.dimension);
  writer->pod(c.feedForwardDimension); writer->pod(c.numLayers); writer->pod(c.numHeads);
  writer->pod(c.epsilon); writer->pod(c.learningRate); writer->pod(c.beta1); writer->pod(c.beta2);
  writer->pod(c.adamEpsilon); writer->pod(c.clipThreshold);
}
bool readConfig(Reader* reader, Config* c) {
  return reader->pod(&c->tokens) && reader->pod(&c->vocabularySize) &&
      reader->pod(&c->dimension) && reader->pod(&c->feedForwardDimension) &&
      reader->pod(&c->numLayers) && reader->pod(&c->numHeads) &&
      reader->pod(&c->epsilon) && reader->pod(&c->learningRate) &&
      reader->pod(&c->beta1) && reader->pod(&c->beta2) &&
      reader->pod(&c->adamEpsilon) && reader->pod(&c->clipThreshold);
}
void hashConfig(std::uint64_t* hash, const Config& c) {
  hashValue(hash, c.tokens); hashValue(hash, c.vocabularySize); hashValue(hash, c.dimension);
  hashValue(hash, c.feedForwardDimension); hashValue(hash, c.numLayers); hashValue(hash, c.numHeads);
  hashValue(hash, c.epsilon); hashValue(hash, c.learningRate); hashValue(hash, c.beta1); hashValue(hash, c.beta2);
  hashValue(hash, c.adamEpsilon); hashValue(hash, c.clipThreshold);
}
bool fail(std::string* error, const char* message) { if (error) *error = message; return false; }
}

std::string hashRegistry(const std::vector<RegistryEntry>& registry) {
  std::uint64_t hash = 1469598103934665603ull; const std::uint64_t count = registry.size(); hashValue(&hash, count);
  for (const auto& entry : registry) { hashString(&hash, entry.name); const std::uint64_t rank = entry.shape.size(); hashValue(&hash, rank); for (std::uint32_t value : entry.shape) hashValue(&hash, value); }
  return hex(hash);
}
std::string hashCheckpointState(const Checkpoint& checkpoint) {
  std::uint64_t hash = 1469598103934665603ull; hashValue(&hash, checkpoint.version); hashConfig(&hash, checkpoint.config);
  hashValue(&hash, checkpoint.seed); hashValue(&hash, checkpoint.completedStep); hashValue(&hash, checkpoint.nextOptimizerStep); hashString(&hash, checkpoint.deterministicState); hashString(&hash, hashRegistry(checkpoint.registry));
  for (const auto* values : {&checkpoint.input, &checkpoint.target, &checkpoint.parameters, &checkpoint.adamM, &checkpoint.adamV}) { const std::uint64_t count = values->size(); hashValue(&hash, count); hash = fnv(values->data(), values->size() * sizeof(float), hash); }
  return hex(hash);
}
void finalizeCheckpoint(Checkpoint* checkpoint) {
  if (!checkpoint) return;
  checkpoint->registryHash = hashRegistry(checkpoint->registry);
  checkpoint->stateHash = hashCheckpointState(*checkpoint);
}
bool validateCheckpoint(const Checkpoint& checkpoint, std::string* error) {
  if (checkpoint.version != kCheckpointVersion ||
      checkpoint.completedStep == std::numeric_limits<std::uint32_t>::max() ||
      checkpoint.nextOptimizerStep != checkpoint.completedStep + 1 ||
      checkpoint.registry.empty() || checkpoint.registry.size() > kMaxEntries) {
    return fail(error, "checkpoint version or step schema");
  }
  const Config& c = checkpoint.config;
  if (!c.tokens || !c.vocabularySize || !c.dimension || !c.feedForwardDimension || !c.numLayers || !c.numHeads || c.dimension % c.numHeads || !std::isfinite(c.epsilon) || c.epsilon <= 0.0f || !std::isfinite(c.learningRate) || !std::isfinite(c.beta1) || !std::isfinite(c.beta2) || !std::isfinite(c.adamEpsilon) || c.adamEpsilon <= 0.0f || !std::isfinite(c.clipThreshold)) return fail(error, "checkpoint configuration");
  std::unordered_set<std::string> names;
  for (const auto& entry : checkpoint.registry) if (entry.name.empty() || !names.insert(entry.name).second || entry.shape.empty()) return fail(error, "checkpoint registry");
  std::size_t expected = 0;
  if (!elementCount(checkpoint.registry, &expected) || checkpoint.parameters.size() != expected || checkpoint.adamM.size() != expected || checkpoint.adamV.size() != expected) return fail(error, "checkpoint parameter lengths");
  if (c.tokens > std::numeric_limits<std::size_t>::max() / c.vocabularySize)
    return fail(error, "checkpoint input length overflow");
  const std::size_t sequence = size_t(c.tokens) * c.vocabularySize;
  if (checkpoint.input.size() != sequence || checkpoint.target.size() != sequence) return fail(error, "checkpoint input lengths");
  if (checkpoint.registryHash != hashRegistry(checkpoint.registry) || checkpoint.stateHash != hashCheckpointState(checkpoint)) return fail(error, "checkpoint hash");
  return true;
}
bool encodeCheckpoint(const Checkpoint& checkpoint, std::vector<std::uint8_t>* bytes, std::string* error) {
  if (!bytes || !validateCheckpoint(checkpoint, error)) return false;
  bytes->clear(); Writer writer(bytes); writer.pod(kMagic); writer.pod(checkpoint.version); writeConfig(&writer, checkpoint.config); writer.pod(checkpoint.seed); writer.pod(checkpoint.completedStep); writer.pod(checkpoint.nextOptimizerStep); writer.string(checkpoint.deterministicState);
  const std::uint64_t count = checkpoint.registry.size(); writer.pod(count);
  for (const auto& entry : checkpoint.registry) { writer.string(entry.name); const std::uint64_t rank = entry.shape.size(); writer.pod(rank); for (auto dimension : entry.shape) writer.pod(dimension); }
  writer.floats(checkpoint.input); writer.floats(checkpoint.target); writer.floats(checkpoint.parameters); writer.floats(checkpoint.adamM); writer.floats(checkpoint.adamV); writer.string(checkpoint.registryHash); writer.string(checkpoint.stateHash);
  const std::uint64_t checksum = fnv(bytes->data(), bytes->size()); writer.pod(checksum); return true;
}
bool decodeCheckpoint(const std::vector<std::uint8_t>& bytes, Checkpoint* checkpoint, std::string* error, const Config* expectedConfig, const std::vector<RegistryEntry>* expectedRegistry) {
  if (!checkpoint || bytes.size() < sizeof(std::uint64_t) || bytes.size() > kMaxBytes) return fail(error, "checkpoint length");
  std::uint64_t stored = 0; std::memcpy(&stored, bytes.data() + bytes.size() - sizeof(stored), sizeof(stored)); if (fnv(bytes.data(), bytes.size() - sizeof(stored)) != stored) return fail(error, "checkpoint checksum");
  Reader reader(bytes); std::uint32_t magic = 0; Checkpoint result; if (!reader.pod(&magic) || !reader.pod(&result.version) || magic != kMagic || result.version != kCheckpointVersion || !readConfig(&reader, &result.config) || !reader.pod(&result.seed) || !reader.pod(&result.completedStep) || !reader.pod(&result.nextOptimizerStep) || !reader.string(&result.deterministicState)) return fail(error, "checkpoint header");
  std::uint64_t count = 0; if (!reader.pod(&count) || count == 0 || count > kMaxEntries) return fail(error, "checkpoint registry count"); result.registry.resize(size_t(count));
  for (auto& entry : result.registry) { std::uint64_t rank = 0; if (!reader.string(&entry.name) || !reader.pod(&rank) || rank == 0 || rank > 8) return fail(error, "checkpoint registry entry"); entry.shape.resize(size_t(rank)); for (auto& dimension : entry.shape) if (!reader.pod(&dimension)) return fail(error, "checkpoint registry shape"); }
  if (!reader.floats(&result.input) || !reader.floats(&result.target) || !reader.floats(&result.parameters) || !reader.floats(&result.adamM) || !reader.floats(&result.adamV) || !reader.string(&result.registryHash) || !reader.string(&result.stateHash) || reader.remaining() != sizeof(std::uint64_t)) return fail(error, "checkpoint payload");
  if (expectedConfig && !sameConfig(result.config, *expectedConfig)) return fail(error, "checkpoint configuration mismatch");
  if (expectedRegistry && !sameRegistry(result.registry, *expectedRegistry)) return fail(error, "checkpoint registry mismatch");
  if (!validateCheckpoint(result, error)) return false;
  *checkpoint = std::move(result);
  return true;
}
std::string checkpointText(const Checkpoint& checkpoint) {
  std::ostringstream out; out << "checkpoint_format=phonelm.qnn.first_nonfinite.v" << checkpoint.version << '\n' << "checkpoint_private_raw=true\n" << "checkpoint_seed=" << checkpoint.seed << '\n' << "checkpoint_completed_step=" << checkpoint.completedStep << '\n' << "checkpoint_optimizer_next_step=" << checkpoint.nextOptimizerStep << '\n' << "checkpoint_registry_hash=" << checkpoint.registryHash << '\n' << "checkpoint_state_hash=" << checkpoint.stateHash << '\n' << "checkpoint_registry_count=" << checkpoint.registry.size() << '\n';
  for (std::size_t i = 0; i < checkpoint.registry.size(); ++i) { out << "checkpoint_registry_" << i << "_name=" << checkpoint.registry[i].name << '\n' << "checkpoint_registry_" << i << "_shape="; for (std::size_t j = 0; j < checkpoint.registry[i].shape.size(); ++j) out << (j ? "x" : "") << checkpoint.registry[i].shape[j]; out << '\n'; } return out.str();
}
TensorSummary summarize(const std::vector<float>& values, const std::vector<std::uint32_t>& shape) {
  TensorSummary summary; summary.shape = shape; summary.count = values.size(); std::uint64_t hash = fnv(values.data(), values.size() * sizeof(float)); summary.hash = hex(hash); bool hasFinite = false, hasNonzero = false; double squares = 0.0;
  for (float value : values) { if (std::isnan(value)) { ++summary.nan; continue; } if (std::isinf(value)) { value > 0 ? ++summary.positiveInfinity : ++summary.negativeInfinity; continue; } const double d = value, absolute = std::fabs(d); if (!hasFinite) { summary.minimum = summary.maximum = d; hasFinite = true; } else { summary.minimum = std::min(summary.minimum, d); summary.maximum = std::max(summary.maximum, d); } summary.maximumAbsolute = std::max(summary.maximumAbsolute, absolute); summary.mean += d; summary.meanAbsolute += absolute; squares += d * d; ++summary.finite; if (absolute != 0.0 && (!hasNonzero || absolute < summary.minimumNonzero)) { summary.minimumNonzero = absolute; hasNonzero = true; } }
  if (summary.finite) { summary.mean /= summary.finite; summary.meanAbsolute /= summary.finite; summary.rms = std::sqrt(squares / summary.finite); } return summary;
}
TensorComparison compare(const std::vector<float>& expected, const std::vector<float>& actual) {
  TensorComparison result; const std::size_t count = std::min(expected.size(), actual.size()); std::vector<std::pair<double, std::size_t>> errors; double referenceSquares = 0.0, errorSquares = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t a = 0, b = 0;
    std::memcpy(&a, &expected[i], sizeof(a));
    std::memcpy(&b, &actual[i], sizeof(b));
    if (a != b && result.firstDifferent == static_cast<std::size_t>(-1))
      result.firstDifferent = i;
    const bool bothFinite = std::isfinite(expected[i]) && std::isfinite(actual[i]);
    const double error = bothFinite ? std::fabs(double(expected[i]) - actual[i]) :
        (a == b ? 0.0 : std::numeric_limits<double>::infinity());
    errors.push_back({error, i});
    result.meanAbsolute += error;
    errorSquares += error * error;
    if (std::isfinite(expected[i])) referenceSquares += double(expected[i]) * expected[i];
    if (error > result.maximumAbsolute) { result.maximumAbsolute = error; result.argmax = i; }
  }
  if (expected.size() != actual.size() &&
      result.firstDifferent == static_cast<std::size_t>(-1)) {
    result.firstDifferent = count;
  }
  if (count) result.meanAbsolute /= count;
  result.relativeL2 = std::sqrt(errorSquares) /
      std::max(std::sqrt(referenceSquares), 1.0e-30);
  std::sort(errors.begin(), errors.end(), [](const auto& a, const auto& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  });
  for (std::size_t i = 0; i < std::min<std::size_t>(3, errors.size()); ++i)
    result.top3.push_back(errors[i].second);
  return result;
}
FirstBadTensor firstBad(const std::vector<NamedTensor>& ordered) { FirstBadTensor result; for (const auto& item : ordered) { if (!item.values) continue; result.summary = summarize(*item.values, item.shape); if (result.summary.nan || result.summary.positiveInfinity || result.summary.negativeInfinity) { result.name = item.name; for (std::size_t i = 0; i < item.values->size(); ++i) if (!std::isfinite((*item.values)[i])) { result.flatIndex = i; break; } return result; } } return result; }
bool validateTapPlan(const TapPlan& plan, std::string* error) { if (plan.scope == TapScope::NONE && !plan.taps.empty()) return fail(error, "tap scope NONE has taps"); if (plan.scope != TapScope::NONE && plan.taps.empty()) return fail(error, "tap scope requires taps"); std::unordered_set<std::string> names; for (const auto& entry : plan.taps) if (entry.name.empty() || entry.shape.empty() || !names.insert(entry.name).second) return fail(error, "tap registry uniqueness or shape"); return true; }
ObserverEffect classifyObserverEffect(const std::vector<float>& untappedFinal, const std::vector<float>& tappedFinal, bool originalMatch) { ObserverEffect result; result.originalMatch = originalMatch; result.untappedFinalHash = summarize(untappedFinal).hash; result.tappedFinalHash = summarize(tappedFinal).hash; result.observerEffect = result.untappedFinalHash != result.tappedFinalHash; return result; }
}  // namespace phonelm::qnn::first_nonfinite
