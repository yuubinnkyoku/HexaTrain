// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include "validation_selection.h"
#include <cstdint>
#include <string>
#include <vector>

namespace phonelm::validation_checkpoint {

inline constexpr std::uint32_t kVersion = 1;

struct Checkpoint {
  std::uint32_t version = kVersion;
  std::string configHash;
  std::uint32_t seed = 0;
  std::uint32_t selectionMode = 0;
  std::uint32_t validationSchemaVersion = 0;
  std::string validationSetHash;
  std::uint32_t validationCaseCount = 0;
  std::uint32_t selectedStep = 0;
  std::uint32_t totalSteps = 0;
  std::uint32_t optimizerNextStep = 1;
  std::uint32_t parameterRegistryVersion = 0;
  std::string registryHash;
  validation_selection::Metrics validation;
  std::vector<float> parameters, adamM, adamV;
  std::string stateHash;
};

struct Expected {
  std::string configHash;
  std::uint32_t seed = 0;
  std::uint32_t selectionMode = 0;
  std::uint32_t validationSchemaVersion = 0;
  std::string validationSetHash;
  std::uint32_t validationCaseCount = 0;
  std::uint32_t totalSteps = 0;
  std::uint32_t parameterRegistryVersion = 0;
  std::string registryHash;
};

std::string hashState(const Checkpoint& checkpoint);
void finalize(Checkpoint* checkpoint);
bool validate(const Checkpoint& checkpoint, std::string* error,
              const Expected* expected = nullptr);
bool encode(const Checkpoint& checkpoint, std::vector<std::uint8_t>* bytes,
            std::string* error);
bool decode(const std::vector<std::uint8_t>& bytes, Checkpoint* checkpoint,
            std::string* error, const Expected* expected = nullptr);

}  // namespace phonelm::validation_checkpoint
