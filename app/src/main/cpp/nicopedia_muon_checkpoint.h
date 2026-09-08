// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

// Research-only mixed-optimizer checkpoint codec for the Nicopedia Muon pilot.
//
// This is deliberately a new, self-contained format.  NPRTCKPTV1/V2/V3 are
// generation and Adam-resume formats and their semantics must not be changed
// to make room for Muon state. Generation/evaluation callers may accept V4
// only through this codec's full identity validation and extractParameters();
// the Adam resume path must continue to reject it.

#include "tiny_language_model_cpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace phonelm::nicopedia_muon_checkpoint {

inline constexpr char kMagic[] = "NPRTCKPTV4\n";
inline constexpr std::uint32_t kSchemaVersion = 4;
inline constexpr char kOptimizerIdentity[] = "muon_aux_adam";
inline constexpr std::uint32_t kParameterRegistryVersion = 1;

enum class ParameterRole : std::uint32_t {
  MUON = 1,
  AUX_ADAM = 2,
};

struct MatrixShape {
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
};

struct RegistryEntry {
  std::string name;
  ParameterRole role = ParameterRole::AUX_ADAM;
  MatrixShape shape;
};

// The cursor is part of the experiment identity.  A global optimizer step is
// not sufficient to prove that a resumed run sees the same article/chunk
// order, especially when a data cache is regenerated independently.
struct DataCursor {
  std::string datasetHash;
  std::uint64_t recordIndex = 0;
  std::uint64_t tokenOffset = 0;
  std::uint64_t epoch = 0;
  std::uint64_t exposedTokens = 0;
  std::uint64_t orderSeed = 0;
};

struct Identity {
  tiny_lm::Config config;
  std::uint32_t seed = 0;
  std::uint64_t globalStep = 0;
  std::string tokenizerKind;
  std::string tokenizerHash;
  DataCursor dataCursor;
};

// All fields that affect the mixed optimizer update are serialized.  The
// schedule fields are retained even though the first pilot only varies the
// Muon peak LR; this prevents a resume from silently changing a schedule.
struct Hyperparameters {
  float muonLearningRate = 0.0f;
  float auxAdamLearningRate = 0.0f;
  float muonTargetLearningRate = 0.0f;
  float auxAdamTargetLearningRate = 0.0f;
  float muonMomentum = 0.95f;
  bool muonNesterov = true;
  std::uint32_t muonNsSteps = 5;
  float auxAdamBeta1 = 0.9f;
  float auxAdamBeta2 = 0.999f;
  float auxAdamEpsilon = 1.0e-8f;
  float muonWeightDecay = 0.0f;
  float auxAdamWeightDecay = 0.0f;
  std::uint32_t decayStartStep = 0;
  std::uint32_t decayEndStep = 0;
  std::uint32_t scheduleTotalSteps = 0;
};

struct ParameterState {
  std::string name;
  ParameterRole role = ParameterRole::AUX_ADAM;
  MatrixShape shape;
  std::vector<float> values;

  // Exactly one optimizer state branch is populated according to role:
  // MUON -> momentum; AUX_ADAM -> adamM and adamV.
  std::vector<float> momentum;
  std::vector<float> adamM;
  std::vector<float> adamV;
};

struct Checkpoint {
  std::string optimizerIdentity = kOptimizerIdentity;
  Identity identity;
  Hyperparameters hyperparameters;
  std::vector<ParameterState> parameters;
};

// Returns the canonical registry in the same order as tiny_lm::parameterRegistry.
// Classification is based on exact semantic roles and expected shapes, never on
// an arbitrary substring match.
std::vector<RegistryEntry> expectedRegistry(const tiny_lm::Config& config);

const char* parameterRoleName(ParameterRole role);

bool classifyParameter(const tiny_lm::Config& config, const std::string& name,
                       ParameterRole* role, MatrixShape* shape,
                       std::string* error = nullptr);

bool validateCheckpoint(const Checkpoint& checkpoint,
                        std::string* error = nullptr);

bool encodeCheckpoint(const Checkpoint& checkpoint,
                      std::vector<std::uint8_t>* bytes,
                      std::string* error = nullptr);

// expectedOptimizerIdentity defaults to the only supported V4 identity.  An
// explicit different value is useful for a fail-closed Muon->Adam resume test.
bool decodeCheckpoint(const std::vector<std::uint8_t>& bytes,
                      Checkpoint* checkpoint, std::string* error = nullptr,
                      const std::string& expectedOptimizerIdentity =
                          kOptimizerIdentity);

// Extracts only model parameters for an explicit research/evaluation caller.
// This does not alter the existing NPRTCKPTV3 generation loader and requires
// an exact config identity match before assigning any destination storage.
bool extractParameters(const Checkpoint& checkpoint,
                       const tiny_lm::Config& expectedConfig,
                       std::uint32_t expectedSeed,
                       qnn::TinyTransformerParameters* parameters,
                       std::string* error = nullptr);

}  // namespace phonelm::nicopedia_muon_checkpoint
