// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Seed selection contract for the generic QNN HTP tiny-LM trainer.
// Shared between the device native path and host-side tests so the
// fail-closed validation rule has exactly one definition.
#pragma once

#include <cstdint>
#include <limits>

namespace phonelm {

enum class SeedSelectionMode : std::uint32_t {
  // Legacy: the process executes seeds 1..correctnessInterval.
  COUNT_FROM_ONE = 0,
  // Execute exactly one established seed k. The seed value must equal
  // correctnessInterval so derived protocol flags match the legacy seed-k
  // process slice; earlier seeds are not initialized, trained, or generated.
  EXACT_SEED = 1,
};

inline const char* seedSelectionModeName(std::uint32_t mode) {
  switch (mode) {
    case 0: return "COUNT_FROM_ONE";
    case 1: return "EXACT_SEED";
  }
  return "UNKNOWN";
}

// Returns nullptr when valid, otherwise a stable failure tag.
inline const char* validateSeedSelection(std::uint32_t mode,
                                         std::uint64_t exactSeed,
                                         std::int64_t correctnessInterval) {
  if (mode == 0) return nullptr;
  if (mode != 1) return "UNKNOWN_SEED_SELECTION_MODE";
  if (correctnessInterval < 1 ||
      correctnessInterval > std::numeric_limits<int>::max())
    return "EXACT_SEED_COUNT_RANGE";
  if (exactSeed < 1 || exactSeed > std::numeric_limits<int>::max())
    return "EXACT_SEED_RANGE";
  if (exactSeed != static_cast<std::uint64_t>(correctnessInterval))
    return "EXACT_SEED_CONTRADICTS_COUNT";
  return nullptr;
}

}  // namespace phonelm
