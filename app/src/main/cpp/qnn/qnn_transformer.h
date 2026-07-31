// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku

#pragma once
#include "../training_engine.h"
#include "qnn_runtime.h"
#include <cstdint>
#include <string>
#include <vector>
namespace phonelm::qnn {
std::string runTransformerExperiment(ExecutionMode mode);
std::string runTinyTransformerTrainingExperiment(
    ExecutionMode mode, const TrainingConfig& trainingConfig,
    const LogSink& progress);
// Private-device diagnostic entry. The payload is the fail-closed
// phonelm.qnn.first_nonfinite codec, never a public result artifact.
std::string replayFirstNonfiniteCheckpoint(
    const std::vector<std::uint8_t>& payload, std::uint32_t repeatCount = 2,
    TinyTransformerTrainingTapSet tapSet = TinyTransformerTrainingTapSet::NONE);
}
