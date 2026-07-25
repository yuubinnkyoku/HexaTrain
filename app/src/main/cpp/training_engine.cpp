#include "training_engine.h"

#include "benchmark_runner.h"
#include "cpu_reference_training.h"
#include "qnn/qnn_backend_info.h"
#include "qnn/qnn_linear_training.h"
#include "qnn/qnn_mlp_training.h"
#include "qnn/qnn_transformer.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace phonelm {
namespace {

std::string runCpuReference(const TrainingConfig& config, const LogSink& log) {
    std::string error;
    if (!validateTrainingConfig(config, error)) {
        const std::string report =
            "CPU_REFERENCE_RESULT\nexecution_mode=CPU_REFERENCE\nbackend_actual=HOST_CPP\n"
            "status=FAILED\nerror=" + error;
        if (log) log(report);
        return report;
    }

    const auto check = cpu::gradientCheck();
    const auto result = cpu::trainLinearRegression(config.batchSize,
                                                   config.dimension,
                                                   config.steps + config.warmupSteps,
                                                   config.learningRate,
                                                   config.seed);
    const bool success = check.passed && result.lossDecreased && result.weightsChanged &&
                         !result.nanDetected;
    std::ostringstream stream;
    stream << std::setprecision(9)
           << "CPU_REFERENCE_RESULT\n"
           << "execution_mode=CPU_REFERENCE\n"
           << "backend_actual=HOST_CPP\n"
           << "batch_size=" << config.batchSize << '\n'
           << "dimension=" << config.dimension << '\n'
           << "steps=" << (config.steps + config.warmupSteps) << '\n'
           << "initial_loss=" << result.initialLoss << '\n'
           << "final_loss=" << result.finalLoss << '\n'
           << "loss_decreased=" << (result.lossDecreased ? "true" : "false") << '\n'
           << "weights_changed=" << (result.weightsChanged ? "true" : "false") << '\n'
           << "nan_detected=" << (result.nanDetected ? "true" : "false") << '\n'
           << "gradient_check_passed=" << (check.passed ? "true" : "false") << '\n'
           << "gradient_check_max_abs_dw=" << check.maxAbsoluteErrorWeight << '\n'
           << "gradient_check_max_rel_dw=" << check.maxRelativeErrorWeight << '\n'
           << "gradient_check_max_abs_dx=" << check.maxAbsoluteErrorInput << '\n'
           << "gradient_check_max_rel_dx=" << check.maxRelativeErrorInput << '\n'
           << "cpu_operations=forward,loss,dP,dW,dX,sgd_update\n"
           << "npu_operations=none\n"
           << "status=" << (success ? "SUCCESS" : "FAILED") << '\n'
           << "error=" << (success ? "none" : "CPU reference correctness check failed");
    const auto report = stream.str();
    if (log) log(report);
    return report;
}

}  // namespace

const char* executionModeName(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::CPU_REFERENCE: return "CPU_REFERENCE";
        case ExecutionMode::MNN_CPU: return "MNN_CPU";
        case ExecutionMode::MNN_OPENCL: return "MNN_OPENCL";
        case ExecutionMode::MNN_VULKAN: return "MNN_VULKAN";
        case ExecutionMode::QNN_CPU_FORWARD: return "QNN_CPU_FORWARD";
        case ExecutionMode::QNN_HTP_FORWARD: return "QNN_HTP_FORWARD";
        case ExecutionMode::QNN_HTP_FORWARD_CPU_BACKWARD:
            return "QNN_HTP_FORWARD_CPU_BACKWARD";
        case ExecutionMode::QNN_HTP_FORWARD_DW: return "QNN_HTP_FORWARD_DW";
        case ExecutionMode::QNN_HTP_FORWARD_DW_DX: return "QNN_HTP_FORWARD_DW_DX";
        case ExecutionMode::QNN_HTP_FULL_STEP: return "QNN_HTP_FULL_STEP";
        case ExecutionMode::QNN_HTP_DEVICE_PROBE: return "QNN_HTP_DEVICE_PROBE";
        case ExecutionMode::QNN_CPU_LINEAR_TRAINING: return "QNN_CPU_LINEAR_TRAINING";
        case ExecutionMode::QNN_HTP_LINEAR_TRAINING: return "QNN_HTP_LINEAR_TRAINING";
        case ExecutionMode::QNN_LINEAR_GRADIENT_CHECK: return "QNN_LINEAR_GRADIENT_CHECK";
        case ExecutionMode::QNN_CPU_MULTIBATCH_TRAINING: return "QNN_CPU_MULTIBATCH_TRAINING";
        case ExecutionMode::QNN_HTP_MULTIBATCH_TRAINING: return "QNN_HTP_MULTIBATCH_TRAINING";
        case ExecutionMode::QNN_CPU_TRAINING_BENCHMARK: return "QNN_CPU_TRAINING_BENCHMARK";
        case ExecutionMode::QNN_HTP_TRAINING_BENCHMARK: return "QNN_HTP_TRAINING_BENCHMARK";
        case ExecutionMode::QNN_HTP_DW_CHECK: return "QNN_HTP_DW_CHECK";
        case ExecutionMode::QNN_HTP_FORWARD_HTP_DW_TRAINING: return "QNN_HTP_FORWARD_HTP_DW_TRAINING";
        case ExecutionMode::QNN_HTP_FORWARD_HTP_DW_BENCHMARK: return "QNN_HTP_FORWARD_HTP_DW_BENCHMARK";        case ExecutionMode::QNN_HTP_DX_CHECK: return "QNN_HTP_DX_CHECK";
        case ExecutionMode::QNN_CPU_MLP_TRAINING: return "QNN_CPU_MLP_TRAINING";
        case ExecutionMode::QNN_HTP_MLP_CPU_BACKWARD: return "QNN_HTP_MLP_CPU_BACKWARD";
        case ExecutionMode::QNN_HTP_MLP_HTP_LINEAR_BACKWARD: return "QNN_HTP_MLP_HTP_LINEAR_BACKWARD";
        case ExecutionMode::QNN_HTP_MLP_BENCHMARK: return "QNN_HTP_MLP_BENCHMARK";
        case ExecutionMode::QNN_MLP_GRADIENT_CHECK: return "QNN_MLP_GRADIENT_CHECK";
        case ExecutionMode::QNN_HTP_RELU_BACKWARD_CHECK: return "QNN_HTP_RELU_BACKWARD_CHECK";
        case ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD: return "QNN_HTP_MLP_FUSED_BACKWARD";
        case ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK: return "QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK";
        case ExecutionMode::QNN_HTP_MSE_CHECK: return "QNN_HTP_MSE_CHECK";
        case ExecutionMode::QNN_HTP_SGD_CHECK: return "QNN_HTP_SGD_CHECK";
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP: return "QNN_HTP_MLP_FULL_STEP";
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK: return "QNN_HTP_MLP_FULL_STEP_BENCHMARK";
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE: return "QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE";
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE: return "QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE";
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE: return "QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE";
        case ExecutionMode::QNN_HTP_LAYER_NORM_CHECK: return "QNN_HTP_LAYER_NORM_CHECK";
        case ExecutionMode::QNN_HTP_SOFTMAX_CHECK: return "QNN_HTP_SOFTMAX_CHECK";
        case ExecutionMode::QNN_HTP_ATTENTION_FORWARD_CHECK: return "QNN_HTP_ATTENTION_FORWARD_CHECK";
        case ExecutionMode::QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK: return "QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK";
        case ExecutionMode::QNN_HTP_SOFTMAX_BACKWARD_CHECK: return "QNN_HTP_SOFTMAX_BACKWARD_CHECK";
        default: return "UNKNOWN";
    }
}

std::string TrainingEngine::run(ExecutionMode mode,
                                const TrainingConfig& config,
                                std::atomic_bool& stopRequested,
                                const LogSink& log) {
    switch (mode) {
        case ExecutionMode::CPU_REFERENCE:
            return runCpuReference(config, log);
        case ExecutionMode::MNN_CPU: {
            auto mnnConfig = config;
            mnnConfig.backend = BackendKind::CPU;
            return BenchmarkRunner::run(mnnConfig, stopRequested, log);
        }
        case ExecutionMode::MNN_OPENCL: {
            auto mnnConfig = config;
            mnnConfig.backend = BackendKind::OPENCL;
            return BenchmarkRunner::run(mnnConfig, stopRequested, log);
        }
        case ExecutionMode::MNN_VULKAN: {
            auto mnnConfig = config;
            mnnConfig.backend = BackendKind::VULKAN;
            return BenchmarkRunner::run(mnnConfig, stopRequested, log);
        }
        case ExecutionMode::QNN_CPU_FORWARD:
        case ExecutionMode::QNN_HTP_FORWARD:
        case ExecutionMode::QNN_HTP_FORWARD_CPU_BACKWARD:
        case ExecutionMode::QNN_HTP_FORWARD_DW:
        case ExecutionMode::QNN_HTP_FORWARD_DW_DX:
        case ExecutionMode::QNN_HTP_FULL_STEP:
        case ExecutionMode::QNN_HTP_DEVICE_PROBE:
        case ExecutionMode::QNN_CPU_LINEAR_TRAINING:
        case ExecutionMode::QNN_HTP_LINEAR_TRAINING:
        case ExecutionMode::QNN_LINEAR_GRADIENT_CHECK:
        case ExecutionMode::QNN_CPU_MULTIBATCH_TRAINING:
        case ExecutionMode::QNN_HTP_MULTIBATCH_TRAINING:
        case ExecutionMode::QNN_CPU_TRAINING_BENCHMARK:
        case ExecutionMode::QNN_HTP_TRAINING_BENCHMARK:
        case ExecutionMode::QNN_HTP_DW_CHECK:
        case ExecutionMode::QNN_HTP_FORWARD_HTP_DW_TRAINING:
        case ExecutionMode::QNN_HTP_FORWARD_HTP_DW_BENCHMARK:
            return qnn::runLinearExperiment(mode, config, log);
        case ExecutionMode::QNN_HTP_DX_CHECK:
        case ExecutionMode::QNN_CPU_MLP_TRAINING:
        case ExecutionMode::QNN_HTP_MLP_CPU_BACKWARD:
        case ExecutionMode::QNN_HTP_MLP_HTP_LINEAR_BACKWARD:
        case ExecutionMode::QNN_HTP_MLP_BENCHMARK:
        case ExecutionMode::QNN_MLP_GRADIENT_CHECK:
        case ExecutionMode::QNN_HTP_RELU_BACKWARD_CHECK:
        case ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD:
        case ExecutionMode::QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK:
        case ExecutionMode::QNN_HTP_MSE_CHECK:
        case ExecutionMode::QNN_HTP_SGD_CHECK:
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP:
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK:
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE:
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE:
        case ExecutionMode::QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE:
            return qnn::runMlpExperiment(mode, config, stopRequested, log);
        case ExecutionMode::QNN_HTP_LAYER_NORM_CHECK:
        case ExecutionMode::QNN_HTP_SOFTMAX_CHECK:
        case ExecutionMode::QNN_HTP_ATTENTION_FORWARD_CHECK:
        case ExecutionMode::QNN_HTP_TINY_TRANSFORMER_FORWARD_CHECK:
        case ExecutionMode::QNN_HTP_SOFTMAX_BACKWARD_CHECK:
            return qnn::runTransformerExperiment(mode);
        default: {
            const std::string report =
                "status=NOT_IMPLEMENTED\nerror=unknown execution mode";
            if (log) log(report);
            return report;
        }
    }
}

std::string TrainingEngine::capabilityReport() {
    return qnn::queryBackendInfo().toLogString();
}

}  // namespace phonelm
