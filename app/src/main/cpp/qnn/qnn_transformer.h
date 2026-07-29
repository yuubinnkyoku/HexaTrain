// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku

#pragma once
#include "../training_engine.h"
#include <string>
namespace phonelm::qnn {
std::string runTransformerExperiment(ExecutionMode mode);
std::string runTinyTransformerTrainingExperiment(
    ExecutionMode mode, const TrainingConfig& trainingConfig);
}
