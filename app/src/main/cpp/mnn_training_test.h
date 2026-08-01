#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace phonelm {

enum class BackendKind : int {
    CPU = 0,
    OPENCL = 1,
    VULKAN = 2,
};

struct TrainingConfig {
    BackendKind backend = BackendKind::CPU;
    int batchSize = 8;
    int dimension = 128;
    int hiddenDimension = 128;
    int outputDimension = 64;
    int steps = 100;
    int warmupSteps = 0;
    float learningRate = 0.1f;
    std::uint64_t seed = 20260710;
    int sampleCount = 512;
    int epochs = 0;
    int measuredSteps = 0;
    int correctnessInterval = 0;
    bool benchmarkMode = false;
    // Seed selection for QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC:
    //   0 = COUNT_FROM_ONE (legacy): run seeds 1..correctnessInterval.
    //   1 = EXACT_SEED: run exactly one seed; the seed value is carried by
    //       `seed`, must be >= 1, and must equal correctnessInterval so that
    //       derived protocol flags match the legacy seed-k process slice.
    int seedSelectionMode = 0;
    // Explicit experimental training-stability mode (training_stability.h).
    // LEGACY is the default and keeps every established hash.
    int trainingStabilityMode = 0;
    // Paired-depth diagnostic initialization assertion (training_stability.h).
    int depthPairInitMode = 0;
    // Diagnostics: when true, the formal loop emits one trajectory_metrics
    // record per training step.
    bool diagnosticTrajectory = false;
    // Diagnostics: when non-empty, private checkpoints are written to this
    // app-private directory at the established checkpoint steps. The path is
    // validated on the Kotlin side to stay below the app files directory;
    // contents are never part of public artifacts.
    std::string diagnosticCheckpointDir;
};

struct TrainingOutcome {
    std::string backendRequested;
    std::string backendActual = "UNINITIALIZED";
    std::string executedBackends = "NONE";
    std::string status = "FAILED";
    std::string error;
    std::string fallbackOperations;
    float initialLoss = 0.0f;
    float finalLoss = 0.0f;
    double averageStepTimeMs = 0.0;
    double medianStepTimeMs = 0.0;
    double p95StepTimeMs = 0.0;
    double totalTimeMs = 0.0;
    int completedSteps = 0;
    bool lossDecreased = false;
    bool nanDetected = false;
    bool weightsChanged = false;
    bool fallbackDetected = false;
    bool requestedBackendObserved = false;
    // MNN 3.5.0 SGD::step intentionally materializes gradients and parameters
    // through readMap()/input(). This is reported instead of being presented as
    // a device-resident optimizer.
    bool optimizerHostSync = true;
    std::vector<double> measuredStepTimesMs;
};

using LogSink = std::function<void(const std::string&)>;

const char* backendName(BackendKind backend);
bool validateTrainingConfig(const TrainingConfig& config, std::string& error);
TrainingOutcome runMnnTraining(const TrainingConfig& config,
                               std::atomic_bool& stopRequested,
                               const LogSink& log);
std::string mnnEnvironmentReport();

}  // namespace phonelm
