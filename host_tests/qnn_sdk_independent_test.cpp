#include <cmath>

#include "qnn/qnn_host_quantization.h"
#include "qnn/qnn_hybrid_training.h"
#include "qnn/qnn_runtime.h"
#include "qnn/qnn_graph_shape_validator.h"
#include "qnn/nicopedia_checkpoint_loader.h"
#include "tiny_language_model_cpu.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void near(double actual, double expected, double tolerance = 1.0e-6) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void writeU32(std::ostream& output, std::uint32_t value) {
    const std::uint8_t bytes[] = {
        static_cast<std::uint8_t>(value >> 24),
        static_cast<std::uint8_t>(value >> 16),
        static_cast<std::uint8_t>(value >> 8),
        static_cast<std::uint8_t>(value),
    };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void writeU64(std::ostream& output, std::uint64_t value) {
    const std::uint8_t bytes[] = {
        static_cast<std::uint8_t>(value >> 56),
        static_cast<std::uint8_t>(value >> 48),
        static_cast<std::uint8_t>(value >> 40),
        static_cast<std::uint8_t>(value >> 32),
        static_cast<std::uint8_t>(value >> 24),
        static_cast<std::uint8_t>(value >> 16),
        static_cast<std::uint8_t>(value >> 8),
        static_cast<std::uint8_t>(value),
    };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void writeGenerationCheckpoint(const std::string& path,
                               const phonelm::tiny_lm::Config& config,
                               std::uint32_t seed, std::uint32_t step) {
    const auto parameters = phonelm::tiny_lm::initialParameters(config, seed);
    const auto registry = phonelm::tiny_lm::parameterRegistry(parameters);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output);
    output.write("NPRTCKPTV1\n", 11);
    writeU32(output, config.vocabularySize);
    writeU32(output, config.tokens);
    writeU32(output, config.dimension);
    writeU32(output, config.feedForwardDimension);
    writeU32(output, config.numLayers);
    writeU32(output, config.numHeads);
    writeU32(output, seed);
    writeU32(output, step);
    writeU32(output, static_cast<std::uint32_t>(registry.size()));
    for (const auto& entry : registry) {
        writeU32(output, static_cast<std::uint32_t>(entry.name.size()));
        output.write(entry.name.data(),
                     static_cast<std::streamsize>(entry.name.size()));
        writeU64(output, static_cast<std::uint64_t>(entry.values->size()));
        output.write(reinterpret_cast<const char*>(entry.values->data()),
                     static_cast<std::streamsize>(entry.values->size() * sizeof(float)));
    }
    assert(output);
}

phonelm::qnn::shape::Node makeShapeNode(
        std::string name,
        phonelm::qnn::shape::Op op,
        std::vector<phonelm::qnn::shape::Tensor> inputs,
        std::vector<phonelm::qnn::shape::Tensor> outputs) {
    phonelm::qnn::shape::Node node{};
    node.name = std::move(name);
    node.op = op;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

template <typename Configure>
phonelm::qnn::shape::Node makeShapeNode(
        std::string name,
        phonelm::qnn::shape::Op op,
        std::vector<phonelm::qnn::shape::Tensor> inputs,
        std::vector<phonelm::qnn::shape::Tensor> outputs,
        Configure&& configure) {
    auto node = makeShapeNode(
            std::move(name), op, std::move(inputs), std::move(outputs));
    std::forward<Configure>(configure)(node);
    return node;
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
    Node broken = makeShapeNode(
            "tt_smd", Op::ReduceSum, {product}, {{"SOFTMAX_DOT", {8, 8}}},
            [](Node& node) {
                node.axes = {1};
                node.keepDims = true;
            });
    const auto rejected = validate(broken);
    require(!rejected.ok, "shape validator accepted bad Softmax reduction");
    require(rejected.error.find("node=tt_smd") != std::string::npos,
            "shape validator error omitted node name");
    require(rejected.error.find("expected=[8,1]") != std::string::npos,
            "shape validator error omitted inferred shape");
    broken.outputs[0].shape = {8, 1};
    require(validate(broken).ok, "shape validator rejected valid Softmax reduction");
    const std::vector<Node> validNodes{
        makeShapeNode("mean", Op::ReduceMean, {product}, {{"out", {8}}},
                [](Node& node) { node.axes = {1}; }),
        makeShapeNode("matmul", Op::MatMul,
                {{"a", {2, 3}}, {"b", {3, 4}}}, {{"out", {2, 4}}}),
        makeShapeNode("transpose", Op::Transpose,
                {{"a", {2, 3}}}, {{"out", {3, 2}}},
                [](Node& node) { node.permutation = {1, 0}; }),
        makeShapeNode("broadcast", Op::ElementWiseBinary,
                {{"a", {8, 8}}, {"b", {8, 1}}}, {{"out", {8, 8}}}),
        makeShapeNode("diagnostic_elementwise_square", Op::ElementWiseBinary,
                {{"input", {1}}, {"input", {1}}}, {{"output", {1}}}),
        makeShapeNode("softmax", Op::Softmax,
                {{"a", {8, 8}}}, {{"out", {8, 8}}},
                [](Node& node) { node.softmaxAxis = 1; }),
        makeShapeNode("reshape", Op::Reshape,
                {{"a", {2, 4}}}, {{"out", {8}}},
                [](Node& node) { node.reshape = {8}; }),
        makeShapeNode("concat", Op::Concat,
                {{"a", {2, 3}}, {"b", {2, 4}}}, {{"out", {2, 7}}},
                [](Node& node) { node.concatAxis = 1; }),
        makeShapeNode("split", Op::Split,
                {{"a", {2, 7}}}, {{"left", {2, 3}}, {"right", {2, 4}}},
                [](Node& node) {
                    node.concatAxis = 1;
                    node.splitSizes = {3, 4};
                }),
        makeShapeNode("layernorm", Op::LayerNorm,
                {{"x", {8, 16}}, {"g", {16}}, {"b", {16}}},
                {{"out", {8, 16}}},
                [](Node& node) {
                    node.axes = {1};
                    node.keepDims = true;
                }),
    };
    for (const auto& node : validNodes)
        require(validate(node).ok, "shape validator rejected valid op");
    const float minimalSquareInput = 279.75f;
    const float minimalSquareCpu =
            minimalSquareInput * minimalSquareInput;
    require(std::isfinite(minimalSquareCpu) &&
                    minimalSquareCpu == 78260.0625f,
            "minimal square CPU fixture changed");
    const std::vector<double> centered{-34.96875, -1.25, 0.5, 35.71875};
    const auto transformedInverse = [&](double scale) {
        double varianceScaled = 0.0;
        for (double value : centered)
            varianceScaled += (scale * value) * (scale * value);
        varianceScaled /= centered.size();
        return std::sqrt(1.0 / (varianceScaled + 1.0e-5 * scale * scale)) *
                scale;
    };
    require(std::fabs(transformedInverse(1.0) -
                      transformedInverse(8.0)) < 1.0e-12,
            "LayerNorm centered-scale transform is not equivalent");
    const auto attentionContext = makeShapeNode(
            "layer_00_attention_context", Op::MatMul,
            {{"probabilities", {8, 8}}, {"value", {8, 16}}},
            {{"context", {8, 16}}});
    require(validate(attentionContext).ok,
            "shape validator rejected [T,T] probabilities times [T,D] values");
    const auto attentionProjected = makeShapeNode(
            "layer_00_attention_projected", Op::MatMul,
            {{"context", {8, 16}}, {"wo", {16, 16}}},
            {{"projected", {8, 16}}});
    require(validate(attentionProjected).ok,
            "shape validator rejected [T,D] context times [D,D] output projection");
    auto badAttentionContext = attentionContext;
    badAttentionContext.outputs[0].shape = {8, 8};
    require(!validate(badAttentionContext).ok,
            "shape validator accepted attention context declared as [T,T]");
    auto badAttentionProjected = attentionProjected;
    badAttentionProjected.outputs[0].shape = {8, 8};
    require(!validate(badAttentionProjected).ok,
            "shape validator accepted attention projected declared as [T,T]");
    const auto headQuery = makeShapeNode(
            "layer_00_head_00_query", Op::MatMul,
            {{"q", {8, 16}}, {"selector", {16, 8}}}, {{"head_q", {8, 8}}});
    require(validate(headQuery).ok, "shape validator rejected [D,Dh] head selector");
    auto badHeadSelector = headQuery;
    badHeadSelector.inputs[1].shape = {16, 7};
    require(!validate(badHeadSelector).ok,
            "shape validator accepted wrong head selector dimension");
    const auto headContext = makeShapeNode(
            "layer_00_head_00_context", Op::MatMul,
            {{"probabilities", {8, 8}}, {"head_v", {8, 8}}}, {{"context", {8, 8}}});
    require(validate(headContext).ok, "shape validator rejected head context [T,Dh]");
    auto badHeadScores = headContext;
    badHeadScores.inputs[0].shape = {8, 7};
    require(!validate(badHeadScores).ok,
            "shape validator accepted wrong head score shape");
    const auto headScatter = makeShapeNode(
            "layer_00_head_00_context_scatter", Op::MatMul,
            {{"context", {8, 8}}, {"selector", {16, 8}}}, {{"scatter", {8, 16}}},
            [](Node& node) { node.transposeInput1 = true; });
    require(validate(headScatter).ok, "shape validator rejected head context scatter [T,D]");
    auto badHeadScatter = headScatter;
    badHeadScatter.outputs[0].shape = {8, 8};
    require(!validate(badHeadScatter).ok,
            "shape validator accepted head scatter instead of [T,D]");
    const auto headSum = makeShapeNode(
            "layer_00_head_context_sum_1", Op::ElementWiseBinary,
            {{"head_00_scatter", {8, 16}}, {"head_01_scatter", {8, 16}}},
            {{"full_context", {8, 16}}});
    require(validate(headSum).ok, "shape validator rejected full head context sum");
    auto badHeadSum = headSum;
    badHeadSum.inputs[1].shape = {8, 8};
    require(!validate(badHeadSum).ok,
            "shape validator accepted head binding with incompatible scatter shape");

    const auto makeLayer = [](std::size_t layer, std::size_t tokens,
                              std::size_t dimension, std::size_t heads,
                              std::size_t feedForward = 32) {
        const std::size_t headDim = dimension / heads;
        TransformerLayerTopology topology{};
        topology.layerIndex = layer;
        topology.parameterLayerIndex = layer;
        topology.gradientLayerIndex = layer;
        topology.input = {tokens, dimension};
        topology.output = {tokens, dimension};
        topology.query = topology.key = topology.value = {tokens, dimension};
        topology.queryHeads = topology.keyHeads = topology.valueHeads = {heads, tokens, headDim};
        topology.scores = topology.probabilities = {heads, tokens, tokens};
        topology.context = {heads, tokens, headDim};
        topology.concat = {tokens, dimension};
        topology.inputGradient = {tokens, dimension};
        topology.residualAfterAttention = topology.residualAfterFfn = {tokens, dimension};
        topology.norm1Mean = topology.norm1Variance = topology.norm2Mean = topology.norm2Variance = {tokens, 1};
        topology.norm1Gamma = topology.norm1Beta = topology.norm2Gamma = topology.norm2Beta = {dimension};
        topology.wq = topology.wk = topology.wv = topology.wo = {dimension, dimension};
        topology.ffnW1 = {dimension, feedForward};
        topology.ffnW2 = {feedForward, dimension};
        topology.dNorm1Gamma = topology.dNorm1Beta = topology.dNorm2Gamma = topology.dNorm2Beta = {dimension};
        topology.dWq = topology.dWk = topology.dWv = topology.dWo = {dimension, dimension};
        topology.dFfnW1 = {dimension, feedForward};
        topology.dFfnW2 = {feedForward, dimension};
        topology.parameterElements = 4 * dimension * dimension + 4 * dimension + 2 * dimension * feedForward;
        topology.optimizerElements = 2 * topology.parameterElements;
        return topology;
    };
    const auto requireTopology = [&](std::size_t layers, std::size_t heads) {
        TransformerTopologyConfig config{8, 16, 32, layers, heads, 32};
        config.parameterElements = layers * (4 * 16 * 16 + 4 * 16 + 2 * 16 * 32) + 2 * 32 * 16;
        config.optimizerElements = 2 * config.parameterElements;
        std::vector<TransformerLayerTopology> topology;
        for (std::size_t i = 0; i < layers; ++i) topology.push_back(makeLayer(i, 8, 16, heads));
        require(validateTransformerTopology(config, topology).ok,
                "shape validator rejected valid Transformer topology");
        return topology;
    };
    requireTopology(1, 1);
    requireTopology(2, 1);
    requireTopology(1, 2);
    requireTopology(3, 2);
    requireTopology(4, 2);
    requireTopology(2, 4);
    requireTopology(3, 4);
    requireTopology(4, 4);
    requireTopology(3, 8);
    // Nicopedia real-text configuration: V=256, T=32, D=16, FFN=32, H=2.
    {
      TransformerTopologyConfig nicopedia{32, 16, 32, 6, 2, 256};
      nicopedia.parameterElements =
          6 * (4 * 16 * 16 + 4 * 16 + 2 * 16 * 32) + 2 * 256 * 16;
      nicopedia.optimizerElements = 2 * nicopedia.parameterElements;
      std::vector<TransformerLayerTopology> topology;
      for (std::size_t i = 0; i < 6; ++i)
        topology.push_back(makeLayer(i, 32, 16, 2));
      require(validateTransformerTopology(nicopedia, topology).ok,
              "shape validator rejected Nicopedia V256/T32/L6/H2 topology");
      TransformerTopologyConfig nicopediaL19 = nicopedia;
      nicopediaL19.numLayers = 19;
      nicopediaL19.parameterElements =
          19 * (4 * 16 * 16 + 4 * 16 + 2 * 16 * 32) + 2 * 256 * 16;
      nicopediaL19.optimizerElements = 2 * nicopediaL19.parameterElements;
      std::vector<TransformerLayerTopology> topology19;
      for (std::size_t i = 0; i < 19; ++i)
        topology19.push_back(makeLayer(i, 32, 16, 2));
      require(validateTransformerTopology(nicopediaL19, topology19).ok,
              "shape validator rejected Nicopedia V256/T32/L19/H2 topology");
      // V256/T32 parameter element contract for L6.
      require(nicopedia.parameterElements == 20864,
              "Nicopedia L6 parameter element count is not 20864");
      require(nicopediaL19.parameterElements == 48320,
              "Nicopedia L19 parameter element count is not 48320");
      TransformerTopologyConfig bpeV1024{32, 32, 32, 19, 2, 1024};
      bpeV1024.parameterElements =
          19 * (4 * 32 * 32 + 4 * 32 + 2 * 32 * 32) + 2 * 1024 * 32;
      bpeV1024.optimizerElements = 2 * bpeV1024.parameterElements;
      std::vector<TransformerLayerTopology> bpeTopology;
      for (std::size_t i = 0; i < 19; ++i)
        bpeTopology.push_back(makeLayer(i, 32, 32, 2, 32));
      require(validateTransformerTopology(bpeV1024, bpeTopology).ok,
              "shape validator rejected Nicopedia V1024/T32/D32/FFN32/L19/H2 topology");
      require(bpeV1024.parameterElements == 184704,
              "Nicopedia V1024 parameter element count is not 184704");
    }
    // Width-search contract: the generalized graph shape validator must accept
    // both single-axis and joint D/FFN candidates, rather than silently
    // assuming the anchor's FFN32 shapes.
    {
      const std::size_t widthCases[][3] = {
          {32, 32, 135552}, {16, 64, 67776}, {32, 64, 174464}};
      for (const auto& widthCase : widthCases) {
        const std::size_t dimension = widthCase[0];
        const std::size_t feedForward = widthCase[1];
        TransformerTopologyConfig wide{
            32, dimension, feedForward, 19, 2, 256};
        wide.parameterElements =
            19 * (4 * dimension * dimension + 4 * dimension +
                  2 * dimension * feedForward) +
            2 * 256 * dimension;
        wide.optimizerElements = 2 * wide.parameterElements;
        std::vector<TransformerLayerTopology> topology;
        for (std::size_t i = 0; i < 19; ++i) {
          topology.push_back(makeLayer(
              i, 32, dimension, 2, feedForward));
        }
        require(validateTransformerTopology(wide, topology).ok,
                "shape validator rejected Nicopedia width-search topology");
        require(wide.parameterElements == widthCase[2],
                "Nicopedia width-search parameter element count mismatch");
      }
    }
    auto twoByTwo = requireTopology(2, 2);
    TransformerTopologyConfig twoByTwoConfig{8, 16, 32, 2, 2, 32};
    twoByTwoConfig.parameterElements = 2 * (4 * 16 * 16 + 4 * 16 + 2 * 16 * 32) + 2 * 32 * 16;
    twoByTwoConfig.optimizerElements = 2 * twoByTwoConfig.parameterElements;
    twoByTwo[0].queryHeads = {2, 8, 7};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong head dimension");
    twoByTwo[0].queryHeads = {2, 8, 8};
    twoByTwo[1].scores = {2, 8, 7};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong scores shape");
    twoByTwo[1].scores = {2, 8, 8};
    twoByTwo[1].concat = {8, 15};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong concat shape");
    twoByTwo[1].concat = {8, 16};
    twoByTwo[1].input = {7, 16};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted broken layer chaining");
    twoByTwo[1].input = {8, 16};
    twoByTwo[1].gradientLayerIndex = 0;
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted another layer's gradient binding");
    twoByTwo[1].gradientLayerIndex = 1;
    twoByTwo[0].probabilities = {2, 8, 7};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong attention probability shape");
    twoByTwo[0].probabilities = {2, 8, 8};
    twoByTwo[0].context = {2, 8, 7};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong attention context shape");
    twoByTwo[0].context = {2, 8, 8};
    twoByTwo[0].norm1Mean = {8};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong LayerNorm reduction shape");
    twoByTwo[0].norm1Mean = {8, 1};
    twoByTwo[0].residualAfterFfn = {8, 15};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong residual add shape");
    twoByTwo[0].residualAfterFfn = {8, 16};
    twoByTwo[0].dWq = {15, 16};
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong backward parameter gradient shape");
    twoByTwo[0].dWq = {16, 16};
    twoByTwo[0].parameterElements -= 1;
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong parameter element count");
    twoByTwo[0].parameterElements += 1;
    twoByTwo[0].optimizerElements -= 1;
    require(!validateTransformerTopology(twoByTwoConfig, twoByTwo).ok,
            "topology accepted wrong optimizer element count");
    TransformerTopologyConfig invalidHeads{8, 16, 32, 2, 3, 32};
    invalidHeads.parameterElements = twoByTwoConfig.parameterElements;
    invalidHeads.optimizerElements = twoByTwoConfig.optimizerElements;
    require(!validateTransformerTopology(invalidHeads, requireTopology(2, 2)).ok,
            "topology accepted non-divisible embedding/head configuration");
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

// FORWARD_ONLY is a generation-only variant: the stub runtime must reject
// both its prepare and its execute fail-closed (never silently succeed), and
// the resource estimator must expose a per-layer forward node budget so the
// generalized builder can derive forward-only expectations.
void testForwardOnlyVariantFailsClosedWithoutQnn() {
    using namespace phonelm::qnn;
    Runtime runtime;
    std::string error;
    assert(!runtime.prepareTinyTransformerTraining(
        32, 32, 32, 1.0e-5f, true, error, 256,
        TinyTransformerTrainingVariant::FORWARD_ONLY,
        TinyTransformerTrainingTapSet::NONE, 2, 2));
    assert(!runtime.initialize(QnnBackendKind::HTP, error));
    TinyTransformerParameters parameters = phonelm::tiny_lm::initialParameters(
        phonelm::tiny_lm::Config{256, 32, 32, 32, 1.0e-5f, 2, 2}, 1);
    TinyTransformerTrainingOutputs outputs;
    assert(!runtime.executeTinyTransformerForwardOnly({}, parameters, outputs, error));
    assert(!runtime.executeTinyTransformerTraining({}, {}, parameters, 0.0f, outputs, error));

    const auto estimate = phonelm::transformer::estimateTrainingResources(
        32, 256, 32, 32, 19, 2);
    assert(estimate.ok);
    // H=2 per-layer forward budget: 30 + 10*2 = 50 nodes.
    assert(estimate.perLayerForwardNodes == 50);
}

void testPreparedGenerationUsesHeaderStepInsteadOfFilename() {
    namespace qnn = phonelm::qnn;
    phonelm::tiny_lm::Config config;
    config.vocabularySize = 4;
    config.tokens = 2;
    config.dimension = 2;
    config.feedForwardDimension = 4;
    config.numLayers = 1;
    config.numHeads = 1;
    constexpr std::uint32_t kSeed = 7;
    constexpr std::uint32_t kHeaderStep = 37;
    const std::string importedPath = "model.ckpt";
    const std::string canonicalPath = "htp-seed7-l1-t2-d2-f4-step37.ckpt";
    std::remove(importedPath.c_str());
    std::remove(canonicalPath.c_str());

    writeGenerationCheckpoint(importedPath, config, kSeed, kHeaderStep);
    writeGenerationCheckpoint(canonicalPath, config, kSeed, kHeaderStep);
    const auto validatePreparedIdentity =
        [&](const std::string& path, std::uint32_t keyStep) {
            const auto loaded = qnn::nprtLoadCheckpointForGeneration(
                path, config, kSeed);
            if (loaded.step != keyStep) {
                throw std::runtime_error("CHECKPOINT_STEP_MISMATCH");
            }
            return loaded;
        };

    // Prepared generation compares the registry-validated header step with
    // PreparedGenerationKey::step, so a generic import filename is accepted.
    const auto imported = validatePreparedIdentity(importedPath, kHeaderStep);
    const auto canonical = validatePreparedIdentity(canonicalPath, kHeaderStep);
    assert(imported.step == kHeaderStep && canonical.step == kHeaderStep);
    // A mismatched key must still be rejected after the same full loader
    // validation; a filename cannot make this comparison pass.
    try {
        static_cast<void>(validatePreparedIdentity(importedPath, kHeaderStep + 1));
        assert(false && "header/key step mismatch must reject");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) == "CHECKPOINT_STEP_MISMATCH");
    }

    assert(std::remove(importedPath.c_str()) == 0);
    assert(std::remove(canonicalPath.c_str()) == 0);
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
    testForwardOnlyVariantFailsClosedWithoutQnn();
    testPreparedGenerationUsesHeaderStepInsteadOfFilename();
    std::cout << "qnn_sdk_independent_tests=PASS\n";
    return 0;
}
