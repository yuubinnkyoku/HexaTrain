// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#include "validation_checkpoint.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace phonelm::validation_checkpoint {
namespace {
constexpr std::uint32_t kMagic = 0x31435651u;  // QVC1
constexpr std::size_t kMaximumBytes = std::size_t{1} << 30;
constexpr std::size_t kMaximumString = std::size_t{1} << 20;

std::uint64_t hashBytes(const void* data, std::size_t size,
                        std::uint64_t hash = 1469598103934665603ull) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}
template <typename T>
void hashValue(std::uint64_t* hash, const T& value) {
  *hash = hashBytes(&value, sizeof(value), *hash);
}
void hashString(std::uint64_t* hash, const std::string& value) {
  const std::uint64_t size = value.size();
  hashValue(hash, size);
  *hash = hashBytes(value.data(), value.size(), *hash);
}
void hashFloats(std::uint64_t* hash, const std::vector<float>& values) {
  const std::uint64_t size = values.size();
  hashValue(hash, size);
  *hash = hashBytes(values.data(), values.size() * sizeof(float), *hash);
}
std::string hex(std::uint64_t value) {
  std::ostringstream text;
  text << std::hex << std::setw(16) << std::setfill('0') << value;
  return text.str();
}
bool fail(std::string* error, const char* message) {
  if (error) *error = message;
  return false;
}

class Writer {
 public:
  explicit Writer(std::vector<std::uint8_t>* bytes) : bytes_(bytes) {}
  template <typename T> void pod(const T& value) {
    const auto* data = reinterpret_cast<const std::uint8_t*>(&value);
    bytes_->insert(bytes_->end(), data, data + sizeof(value));
  }
  void string(const std::string& value) {
    const std::uint64_t size = value.size();
    pod(size);
    bytes_->insert(bytes_->end(), value.begin(), value.end());
  }
  void floats(const std::vector<float>& values) {
    const std::uint64_t size = values.size();
    pod(size);
    const auto* data = reinterpret_cast<const std::uint8_t*>(values.data());
    bytes_->insert(bytes_->end(), data, data + values.size() * sizeof(float));
  }
 private:
  std::vector<std::uint8_t>* bytes_;
};

class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
  template <typename T> bool pod(T* value) {
    if (remaining() < sizeof(T)) return false;
    std::memcpy(value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return true;
  }
  bool string(std::string* value) {
    std::uint64_t size = 0;
    if (!pod(&size) || size > kMaximumString || size > remaining()) return false;
    value->assign(reinterpret_cast<const char*>(bytes_.data() + offset_),
                  std::size_t(size));
    offset_ += std::size_t(size);
    return true;
  }
  bool floats(std::vector<float>* values) {
    std::uint64_t size = 0;
    if (!pod(&size) || size > kMaximumBytes / sizeof(float) ||
        size > remaining() / sizeof(float)) return false;
    values->resize(std::size_t(size));
    std::memcpy(values->data(), bytes_.data() + offset_,
                values->size() * sizeof(float));
    offset_ += values->size() * sizeof(float);
    return true;
  }
  std::size_t remaining() const { return bytes_.size() - offset_; }
 private:
  const std::vector<std::uint8_t>& bytes_;
  std::size_t offset_ = 0;
};
}  // namespace

std::string hashState(const Checkpoint& c) {
  std::uint64_t hash = 1469598103934665603ull;
  hashValue(&hash, c.version); hashString(&hash, c.configHash);
  hashValue(&hash, c.seed); hashValue(&hash, c.selectionMode);
  hashValue(&hash, c.validationSchemaVersion); hashString(&hash, c.validationSetHash);
  hashValue(&hash, c.validationCaseCount); hashValue(&hash, c.selectedStep);
  hashValue(&hash, c.totalSteps); hashValue(&hash, c.optimizerNextStep);
  hashValue(&hash, c.parameterRegistryVersion); hashString(&hash, c.registryHash);
  hashValue(&hash, c.validation.loss); hashValue(&hash, c.validation.accuracy);
  hashValue(&hash, c.validation.targetMargin);
  hashValue(&hash, c.validation.targetProbability);
  hashFloats(&hash, c.parameters); hashFloats(&hash, c.adamM); hashFloats(&hash, c.adamV);
  return hex(hash);
}

void finalize(Checkpoint* checkpoint) {
  if (checkpoint) checkpoint->stateHash = hashState(*checkpoint);
}

bool validate(const Checkpoint& c, std::string* error, const Expected* expected) {
  if (c.version != kVersion || c.seed == 0 ||
      c.selectionMode != std::uint32_t(validation_selection::Mode::BEST_VALIDATION_V1) ||
      c.validationSchemaVersion != validation_selection::kValidationSchemaVersion ||
      c.parameterRegistryVersion != validation_selection::kParameterRegistryVersion ||
      c.optimizerNextStep != c.selectedStep + 1 || c.configHash.empty() ||
      c.validationSetHash.empty() || c.validationCaseCount == 0 ||
      c.totalSteps == 0 || c.selectedStep > c.totalSteps || c.registryHash.empty())
    return fail(error, "validation checkpoint header");
  if (!validation_selection::finite(c.validation))
    return fail(error, "validation checkpoint metrics");
  if (c.validation.loss < 0.0 || c.validation.accuracy < 0.0 ||
      c.validation.accuracy > 1.0 || c.validation.targetProbability < 0.0 ||
      c.validation.targetProbability > 1.0)
    return fail(error, "validation checkpoint metric range");
  if (c.parameters.empty() || c.parameters.size() != c.adamM.size() ||
      c.parameters.size() != c.adamV.size())
    return fail(error, "validation checkpoint state lengths");
  for (const auto* values : {&c.parameters, &c.adamM, &c.adamV})
    for (float value : *values)
      if (!std::isfinite(value))
        return fail(error, "validation checkpoint non-finite state");
  if (c.stateHash != hashState(c))
    return fail(error, "validation checkpoint state hash");
  if (expected &&
      (c.configHash != expected->configHash || c.seed != expected->seed ||
       c.selectionMode != expected->selectionMode ||
       c.validationSchemaVersion != expected->validationSchemaVersion ||
       c.validationSetHash != expected->validationSetHash ||
       c.validationCaseCount != expected->validationCaseCount ||
       c.totalSteps != expected->totalSteps ||
       c.parameterRegistryVersion != expected->parameterRegistryVersion ||
       c.registryHash != expected->registryHash))
    return fail(error, "validation checkpoint identity mismatch");
  return true;
}

bool encode(const Checkpoint& c, std::vector<std::uint8_t>* bytes,
            std::string* error) {
  if (!bytes || !validate(c, error)) return false;
  bytes->clear();
  Writer w(bytes);
  w.pod(kMagic); w.pod(c.version); w.string(c.configHash); w.pod(c.seed);
  w.pod(c.selectionMode); w.pod(c.validationSchemaVersion);
  w.string(c.validationSetHash); w.pod(c.validationCaseCount);
  w.pod(c.selectedStep); w.pod(c.totalSteps); w.pod(c.optimizerNextStep);
  w.pod(c.parameterRegistryVersion); w.string(c.registryHash);
  w.pod(c.validation.loss); w.pod(c.validation.accuracy);
  w.pod(c.validation.targetMargin); w.pod(c.validation.targetProbability);
  w.floats(c.parameters); w.floats(c.adamM); w.floats(c.adamV); w.string(c.stateHash);
  const std::uint64_t checksum = hashBytes(bytes->data(), bytes->size());
  w.pod(checksum);
  return true;
}

bool decode(const std::vector<std::uint8_t>& bytes, Checkpoint* checkpoint,
            std::string* error, const Expected* expected) {
  if (!checkpoint || bytes.size() < 16 || bytes.size() > kMaximumBytes)
    return fail(error, "validation checkpoint length");
  std::uint64_t checksum = 0;
  std::memcpy(&checksum, bytes.data() + bytes.size() - sizeof(checksum),
              sizeof(checksum));
  if (hashBytes(bytes.data(), bytes.size() - sizeof(checksum)) != checksum)
    return fail(error, "validation checkpoint checksum");
  Reader r(bytes);
  Checkpoint c;
  std::uint32_t magic = 0;
  if (!r.pod(&magic) || magic != kMagic || !r.pod(&c.version) ||
      !r.string(&c.configHash) || !r.pod(&c.seed) || !r.pod(&c.selectionMode) ||
      !r.pod(&c.validationSchemaVersion) || !r.string(&c.validationSetHash) ||
      !r.pod(&c.validationCaseCount) || !r.pod(&c.selectedStep) ||
      !r.pod(&c.totalSteps) || !r.pod(&c.optimizerNextStep) ||
      !r.pod(&c.parameterRegistryVersion) || !r.string(&c.registryHash) ||
      !r.pod(&c.validation.loss) || !r.pod(&c.validation.accuracy) ||
      !r.pod(&c.validation.targetMargin) ||
      !r.pod(&c.validation.targetProbability) || !r.floats(&c.parameters) ||
      !r.floats(&c.adamM) || !r.floats(&c.adamV) || !r.string(&c.stateHash) ||
      r.remaining() != sizeof(std::uint64_t))
    return fail(error, "validation checkpoint payload");
  if (!validate(c, error, expected)) return false;
  *checkpoint = std::move(c);
  return true;
}

}  // namespace phonelm::validation_checkpoint
