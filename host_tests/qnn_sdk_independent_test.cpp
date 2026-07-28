#include "qnn/qnn_host_quantization.h"
#include "qnn/qnn_hybrid_training.h"
#include "qnn/qnn_runtime.h"
#include "qnn/qnn_graph_shape_validator.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void near(double actual, double expected, double tolerance = 1.0e-6) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void testSymmetricQuantizeAndDequantize() {
    using namespace phonelm::qnn::host;
    const std::vector<float> values{-1.0f, -0.25f, 0.0f, 0.25f, 1.0f};
    const auto parameters = chooseSignedSymmetricParameters(values, 8);
    assert(parameters.valid());
    assert(parameters.minimumCode == -127);
    assert(parameters.maximumCode == 127);
    assert(parameters.zeroPoint == 0);
    near(parameters.scale, 1.0 / 127.0, 1.0e-8);

    const auto quantized = quantize(values, parameters);
    assert(quantized.values.front() == -127);
    assert(quantized.values[2] == 0);
    assert(quantized.values.back() == 127);
    near(quantized.saturatedValueRatio, 0.4);
    const auto restored = dequantize(quantized.values, parameters);
    for (std::size_t index = 0; index < values.size(); ++index) {
        assert(std::fabs(restored[index] - values[index]) <= parameters.scale * 0.51f);
    }
}

void testAffineScaleAndZeroPoint() {
    using namespace phonelm::qnn::host;
    const std::vector<float> values{-0.25f, 0.0f, 0.5f};
    const auto parameters = chooseSignedAffineParameters(values, 8);
    assert(parameters.valid());
    assert(parameters.minimumCode == -128);
    assert(parameters.maximumCode == 127);
    assert(parameters.zeroPoint >= parameters.minimumCode);
    assert(parameters.zeroPoint <= parameters.maximumCode);
    const auto restored = dequantize(quantize(values, parameters).values, parameters);
    for (std::size_t index = 0; index < values.size(); ++index) {
        assert(std::fabs(restored[index] - values[index]) <= parameters.scale * 0.51f);
    }
}

void testGradientAndSaturationMetrics() {
    using namespace phonelm::qnn::host;
    const AffineQuantizationParameters parameters{0.1f, 0, -2, 2};
    const std::vector<float> gradients{0.0f, 0.01f, -0.01f, 0.11f, -0.5f, 0.5f};
    const auto quantized = quantize(gradients, parameters);
    near(zeroGradientRatio(gradients, quantized.values, parameters), 2.0 / 5.0);
    near(quantized.saturatedValueRatio, 2.0 / 6.0);
}

void testShapeAndBufferChecks() {
    using namespace phonelm::qnn::host;
    const auto elements = checkedElementCount({2, 4});
    const auto bytes = checkedBufferSize({2, 4}, sizeof(float));
    assert(elements && *elements == 8);
    assert(bytes && *bytes == 32);
    assert(!checkedElementCount({2, 0, 4}));
    assert(!checkedElementCount({std::numeric_limits<std::size_t>::max(), 2}));
    assert(!checkedBufferSize({std::numeric_limits<std::size_t>::max()}, 2));
}

void testGraphShapeValidator() {
    using namespace phonelm::qnn::shape;
    const auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    const Tensor product{"SOFTMAX_PRODUCT", {8, 8}};
    Node broken{"tt_smd", Op::ReduceSum, {product},
                {{"SOFTMAX_DOT", {8, 8}}}, {1}, true};
    const auto rejected = validate(broken);
    require(!rejected.ok, "shape validator accepted bad Softmax reduction");
    require(rejected.error.find("node=tt_smd") != std::string::npos,
            "shape validator error omitted node name");
    require(rejected.error.find("expected=[8,1]") != std::string::npos,
            "shape validator error omitted inferred shape");
    broken.outputs[0].shape = {8, 1};
    require(validate(broken).ok, "shape validator rejected valid Softmax reduction");
    const std::vector<Node> validNodes{
        {"mean", Op::ReduceMean, {product}, {{"out", {8}}}, {1}, false},
        {"matmul", Op::MatMul, {{"a", {2, 3}}, {"b", {3, 4}}}, {{"out", {2, 4}}}},
        {"transpose", Op::Transpose, {{"a", {2, 3}}}, {{"out", {3, 2}}}, {}, false, false, false, {1, 0}},
        {"broadcast", Op::ElementWiseBinary, {{"a", {8, 8}}, {"b", {8, 1}}}, {{"out", {8, 8}}}},
        {"softmax", Op::Softmax, {{"a", {8, 8}}}, {{"out", {8, 8}}}, {}, false, false, false, {}, {}, 0, {}, 1},
        {"reshape", Op::Reshape, {{"a", {2, 4}}}, {{"out", {8}}}, {}, false, false, false, {}, {8}},
        {"concat", Op::Concat, {{"a", {2, 3}}, {"b", {2, 4}}}, {{"out", {2, 7}}}, {}, false, false, false, {}, {}, 1},
        {"split", Op::Split, {{"a", {2, 7}}}, {{"left", {2, 3}}, {"right", {2, 4}}}, {}, false, false, false, {}, {}, 1, {3, 4}},
        {"layernorm", Op::LayerNorm, {{"x", {8, 16}}, {"g", {16}}, {"b", {16}}}, {{"out", {8, 16}}}, {1}, true},
    };
    for (const auto& node : validNodes)
        require(validate(node).ok, "shape validator rejected valid op");
}

void testMockGraphReuseAndRuntimeWeight() {
    using phonelm::cpu::Matrix;
    using phonelm::qnn::host::MockFixedShapeMatMulExecutor;
    MockFixedShapeMatMulExecutor executor;
    const Matrix x(2, 4, {
        0.1f, -0.2f, 0.3f, -0.4f,
        -0.1f, 0.2f, -0.3f, 0.4f,
    });
    Matrix firstWeight(4, 4, {
        0.1f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.1f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.1f,
    });
    Matrix firstOutput;
    Matrix secondOutput;
    std::string error;
    assert(executor.execute(x, firstWeight, firstOutput, error));
    firstWeight.at(0, 0) = 0.2f;
    assert(executor.execute(x, firstWeight, secondOutput, error));
    assert(executor.graphFinalized());
    assert(executor.graphBuildCount() == 1);
    assert(executor.executionCount() == 2);
    assert(firstOutput.at(0, 0) != secondOutput.at(0, 0));

    const Matrix wrongShape(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});
    Matrix ignored;
    assert(!executor.execute(wrongShape, firstWeight, ignored, error));
    assert(error == "MOCK_FIXED_SHAPE_GRAPH_REQUIRES_REBUILD");
}

void testMockDWeightAndHybridLossDecrease() {
    using phonelm::cpu::Matrix;
    using phonelm::qnn::host::MockFixedShapeMatMulExecutor;
    const Matrix x(2, 4, {
        0.1f, -0.2f, 0.3f, -0.4f,
        -0.1f, 0.2f, -0.3f, 0.4f,
    });
    const Matrix dPrediction(2, 4, {
        0.01f, -0.02f, 0.03f, -0.04f,
        -0.01f, 0.02f, -0.03f, 0.04f,
    });
    const Matrix transposedX = phonelm::cpu::transpose(x);
    const Matrix expected = phonelm::cpu::matMul(transposedX, dPrediction);
    MockFixedShapeMatMulExecutor dWeightExecutor;
    Matrix actual;
    std::string error;
    assert(dWeightExecutor.execute(transposedX, dPrediction, actual, error));
    assert(actual.sameShape(expected));
    for (std::size_t index = 0; index < actual.values.size(); ++index) {
        near(actual.values[index], expected.values[index]);
    }

    MockFixedShapeMatMulExecutor forwardTrainingExecutor;
    MockFixedShapeMatMulExecutor dWeightTrainingExecutor;
    const auto result = phonelm::qnn::host::runHybridTraining(
        forwardTrainingExecutor,
        dWeightTrainingExecutor,
        2,
        4,
        20,
        0.1f,
        20'260'710ULL);
    const auto cpuBaseline = phonelm::cpu::trainLinearRegression(
        2, 4, 20, 0.1f, 20'260'710ULL);
    const double initialLossAbsoluteError =
        std::fabs(result.initialLoss - cpuBaseline.initialLoss);
    const double finalLossAbsoluteError =
        std::fabs(result.finalLoss - cpuBaseline.finalLoss);
    std::cout << "mock_hybrid_initial_loss=" << result.initialLoss << '\n'
              << "mock_hybrid_final_loss=" << result.finalLoss << '\n'
              << "mock_hybrid_forward_steps=" << result.forwardSteps << '\n'
              << "mock_hybrid_dw_steps=" << result.dWeightSteps << '\n'
              << "mock_vs_cpu_initial_loss_abs_error=" << initialLossAbsoluteError << '\n'
              << "mock_vs_cpu_final_loss_abs_error=" << finalLossAbsoluteError << '\n';
    assert(result.error.empty());
    assert(result.forwardBackend == "MOCK_HOST_CPP");
    assert(result.dWeightBackend == "MOCK_HOST_CPP");
    assert(result.transposeLocation == "HOST_CPP");
    assert(result.stepsCompleted == 20);
    assert(result.forwardSteps == 20);
    assert(result.dWeightSteps == 20);
    assert(forwardTrainingExecutor.graphBuildCount() == 1);
    assert(dWeightTrainingExecutor.graphBuildCount() == 1);
    assert(result.lossDecreased);
    assert(result.weightsChanged);
    assert(!result.nanDetected);
    near(initialLossAbsoluteError, 0.0);
    near(finalLossAbsoluteError, 0.0);
}

void testQnnDisabledRuntimeIsExplicitlyBlocked() {
    using namespace phonelm::qnn;
    const auto info = queryBackendInfo();
    assert(!info.qnnBuildEnabled);
    assert(!info.sdkDetected);
    assert(!info.implementationReady);
    assert(info.status == "BLOCKED_BY_QAIRT_SDK_NOT_INSTALLED");
    assert(std::string(backendKindName(QnnBackendKind::CPU)) == "CPU");
    assert(std::string(backendKindName(QnnBackendKind::HTP)) == "HTP");

    Runtime runtime;
    std::string error;
    assert(!runtime.initialize(QnnBackendKind::HTP, error));
    assert(error.find("QNN_SDK_NOT_FOUND") != std::string::npos);
    assert(error.find("HTP") != std::string::npos);
}

}  // namespace

int main() {
    testSymmetricQuantizeAndDequantize();
    testAffineScaleAndZeroPoint();
    testGradientAndSaturationMetrics();
    testShapeAndBufferChecks();
    testGraphShapeValidator();
    testMockGraphReuseAndRuntimeWeight();
    testMockDWeightAndHybridLossDecrease();
    testQnnDisabledRuntimeIsExplicitlyBlocked();
    std::cout << "qnn_sdk_independent_tests=PASS\n";
    return 0;
}
