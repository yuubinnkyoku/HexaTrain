#include "cpu_reference_training.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>

namespace {

void near(float actual, float expected, float tolerance = 1.0e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void testMatMulAndTranspose() {
    using phonelm::cpu::Matrix;
    const Matrix x(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    const Matrix w(2, 2, {0.5f, -1.0f, 2.0f, 0.25f});
    const auto product = phonelm::cpu::matMul(x, w);
    near(product.at(0, 0), 4.5f);
    near(product.at(0, 1), -0.5f);
    near(product.at(1, 0), 9.5f);
    near(product.at(1, 1), -2.0f);

    const auto transposed = phonelm::cpu::transpose(x);
    near(transposed.at(0, 1), 3.0f);
    near(transposed.at(1, 0), 2.0f);
}

void testLossGradientsAndSgd() {
    using phonelm::cpu::Matrix;
    const Matrix x(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix w(2, 2, {0.5f, -1.0f, 2.0f, 0.25f});
    const Matrix target(2, 2, {4.0f, 0.0f, 10.0f, -1.0f});
    const auto state = phonelm::cpu::forwardBackward(x, target, w);

    // E = [[0.5, -0.5], [-0.5, -1.0]], mean(E^2) = 0.4375.
    near(state.loss, 0.4375f);
    // dP = 2E/(B*D) = E/2.
    near(state.dPrediction.at(0, 0), 0.25f);
    near(state.dPrediction.at(1, 1), -0.5f);
    // dW = X^T dP.
    near(state.dWeight.at(0, 0), -0.5f);
    near(state.dWeight.at(0, 1), -1.75f);
    near(state.dWeight.at(1, 0), -0.5f);
    near(state.dWeight.at(1, 1), -2.5f);
    // dX = dP W^T.
    near(state.dInput.at(0, 0), 0.375f);
    near(state.dInput.at(0, 1), 0.4375f);
    near(state.dInput.at(1, 0), 0.375f);
    near(state.dInput.at(1, 1), -0.625f);

    phonelm::cpu::sgdUpdate(w, state.dWeight, 0.1f);
    near(w.at(0, 0), 0.55f);
    near(w.at(0, 1), -0.825f);
    near(w.at(1, 0), 2.05f);
    near(w.at(1, 1), 0.5f);
}

void testGradientCheck() {
    const auto check = phonelm::cpu::gradientCheck(2, 4, 1.0e-3f);
    std::cout << "gradient_check_max_abs_dw=" << check.maxAbsoluteErrorWeight << '\n'
              << "gradient_check_max_rel_dw=" << check.maxRelativeErrorWeight << '\n'
              << "gradient_check_max_abs_dx=" << check.maxAbsoluteErrorInput << '\n'
              << "gradient_check_max_rel_dx=" << check.maxRelativeErrorInput << '\n';
    assert(check.passed);
}

void testLossDecrease() {
    const auto result = phonelm::cpu::trainLinearRegression(8, 128, 100, 0.1f, 20'260'710ULL);
    std::cout << "cpu_initial_loss=" << result.initialLoss << '\n'
              << "cpu_final_loss=" << result.finalLoss << '\n';
    assert(result.lossDecreased);
    assert(result.weightsChanged);
    assert(!result.nanDetected);
    assert(result.lossHistory.size() == 101);
}

void testTinyLanguageModelGradientCheck() {
    const auto check = phonelm::tiny_lm::gradientCheck();
    std::cout << check.report
              << "tiny_lm_gradient_check_max_abs=" << check.maximumAbsoluteError << '\n'
              << "tiny_lm_gradient_check_max_rel=" << check.maximumRelativeError << '\n';
    assert(check.passed);
}

void testTinyLanguageModelLearning() {
    const phonelm::tiny_lm::Config config{};
    const std::vector<uint32_t> sequence{0, 1, 2, 3, 0, 1, 2, 3, 0};
    const std::vector<uint32_t> inputTokens(sequence.begin(), sequence.end() - 1);
    const std::vector<uint32_t> targetTokens(sequence.begin() + 1, sequence.end());
    const auto input = phonelm::tiny_lm::oneHot(inputTokens, config.vocabularySize);
    const auto target = phonelm::tiny_lm::oneHot(targetTokens, config.vocabularySize);
    for (uint32_t seed = 1; seed <= 5; ++seed) {
        auto parameters = phonelm::tiny_lm::initialParameters(config, seed);
        const auto initial = phonelm::tiny_lm::forwardBackward(
            config, input, target, parameters, 0.0f);
        auto current = parameters;
        for (int step = 0; step < 320; ++step)
            current = phonelm::tiny_lm::forwardBackward(
                config, input, target, current, 0.03f).next;
        const auto final = phonelm::tiny_lm::forwardBackward(
            config, input, target, current, 0.0f);
        std::cout << "tiny_lm_seed=" << seed
                  << " initial_loss=" << initial.loss
                  << " final_loss=" << final.loss
                  << " initial_accuracy=" << initial.accuracy
                  << " final_accuracy=" << final.accuracy << '\n';
        assert(final.loss < initial.loss);
        assert(final.accuracy > initial.accuracy);
    }
}

void testTinyLanguageModelMultiLayerMultiHead() {
    using namespace phonelm::tiny_lm;
    Config config{};
    config.tokens = 4;
    config.dimension = 8;
    config.feedForwardDimension = 16;
    config.numLayers = 2;
    config.numHeads = 2;
    std::string error;
    assert(validateConfig(config, &error));
    auto parameters = initialParameters(config, 7);
    assert(parameters.layers.size() == 1);
    assert(parameterStorageHasNoAliases(parameters));
    const auto registry = parameterRegistry(parameters);
    assert(registry.front().name == "token_embedding");
    assert(registry[1].name == "layer_000.norm1_gamma");
    assert(registry[3].name == "layer_000.wq");
    assert(registry[11].name == "layer_001.norm1_gamma");
    const auto input = oneHot({0, 1, 2, 3}, config.vocabularySize);
    const auto target = oneHot({1, 2, 3, 4}, config.vocabularySize);
    const auto step = forwardBackward(config, input, target, parameters, 0.003f);
    assert(std::isfinite(step.loss));
    assert(step.gradients.layers.size() == 1);
    assert(step.gradients.layers[0].wq.size() == parameters.layers[0].wq.size());
    auto invalid = config;
    invalid.numHeads = 3;
    assert(!validateConfig(invalid, &error));
}

void testTinyLanguageModelGeneralizedCoverage() {
    using namespace phonelm::tiny_lm;
    auto exercise = [](uint32_t layers, uint32_t heads) {
        Config c{};
        c.tokens = 4; c.dimension = 8; c.feedForwardDimension = 16;
        c.numLayers = layers; c.numHeads = heads;
        const auto p = initialParameters(c, 13);
        const auto x = oneHot({0, 1, 2, 3}, c.vocabularySize);
        const auto y = oneHot({1, 2, 3, 4}, c.vocabularySize);
        const auto step = forwardBackward(c, x, y, p, 0.003f);
        assert(std::isfinite(step.loss));
        assert(step.gradients.layers.size() == layers - 1);
        const auto expected = parameterRegistry(p);
        const auto gradients = parameterRegistry(step.gradients);
        const auto next = parameterRegistry(step.next);
        assert(expected.size() == gradients.size() && expected.size() == next.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(expected[i].name == gradients[i].name && expected[i].name == next[i].name);
            assert(expected[i].values->size() == gradients[i].values->size());
            assert(expected[i].values->size() == next[i].values->size());
        }
        assert(parameterStorageHasNoAliases(step.gradients));
        assert(parameterStorageHasNoAliases(step.next));
        return std::make_tuple(c, p, x, y, step);
    };
    exercise(1, 1);
    exercise(2, 2);
    exercise(3, 2);
    exercise(4, 2);
    exercise(2, 4);
    exercise(3, 4);
    exercise(4, 4);
    exercise(3, 8);
    auto [c, p, x, y, step] = exercise(2, 2);
    auto zero = p;
    auto clear = [](phonelm::qnn::TinyTransformerLayerParameters& l) {
        for (auto* v : {&l.gamma1,&l.beta1,&l.wq,&l.wk,&l.wv,&l.wo,&l.gamma2,&l.beta2,&l.w1,&l.w2})
            std::fill(v->begin(), v->end(), 0.0f);
    };
    clear(zero); for (auto& l : zero.layers) clear(l);
    std::fill(zero.tokenEmbedding.begin(), zero.tokenEmbedding.end(), 0.0f);
    std::fill(zero.outputProjection.begin(), zero.outputProjection.end(), 0.0f);
    const auto adam = adamUpdate(p, step.gradients, zero, zero, 0.003f, .9f, .999f, 1e-8f, 10.0f, 1000.0f);
    assert(adam.next.layers.size() == 1);
    assert(parameterStorageHasNoAliases(adam.firstMoment));
    assert(parameterStorageHasNoAliases(adam.secondMoment));
    assert(parameterStorageHasNoAliases(adam.firstMomentHat));
    assert(parameterStorageHasNoAliases(adam.secondMomentHat));
    assert(parameterStorageHasNoAliases(adam.next));
    const size_t expected = size_t(2) * c.vocabularySize * c.dimension + c.numLayers *
        (size_t(4) * c.dimension * c.dimension + size_t(4) * c.dimension + size_t(2) * c.dimension * c.feedForwardDimension);
    assert(parameterElementCount(p) == expected);
    const auto originalLayer0 = p.wq;
    const auto originalLayer1 = p.layers[0].wq;
    auto changed0 = p; changed0.wq[0] += .1f;
    auto changed1 = p; changed1.layers[0].wq[0] += .1f;
    const auto s0 = forwardBackward(c, x, y, changed0, 0.0f);
    const auto s1 = forwardBackward(c, x, y, changed1, 0.0f);
    assert(s0.loss != step.loss && s1.loss != step.loss);
    assert(p.wq == originalLayer0 && p.layers[0].wq == originalLayer1);
    assert(s0.gradients.wq != step.gradients.wq);
    assert(s1.gradients.layers[0].wq != step.gradients.layers[0].wq);
    assert(adam.firstMoment.wq.data() != adam.firstMoment.layers[0].wq.data());
    assert(adam.secondMoment.wq.data() != adam.secondMoment.layers[0].wq.data());
    const auto estimate = resourceEstimate(c);
    assert(estimate.ok);
    assert(estimate.parameterElements == expected);
    assert(estimate.gradientElements == expected);
    assert(estimate.adamMomentElements == 2 * expected);
    assert(estimate.forwardActivationBytes > 0);
    assert(estimate.backwardActivationBytes > 0);
    assert(estimate.attentionBytes ==
           size_t(2) * c.numLayers * c.numHeads * c.tokens * c.tokens *
               sizeof(float));
    assert(estimate.adamGraphElements ==
           std::min<std::uint64_t>(
               estimate.parameterElements,
               phonelm::transformer::kMaximumAdamChunkElements));
    assert(estimate.adamChunkCount >= 1);
    assert(estimate.adamApplicationVisibleBytes ==
           size_t(11) * estimate.adamGraphElements * sizeof(float));
    assert(estimate.nodeCount > 0 && estimate.tensorCount > 0);
}

void testTinyLanguageModelSchemaFailClosedAndRegistry() {
    using namespace phonelm::tiny_lm;
    Config base{}; std::string error;
    auto invalid = base; invalid.numLayers = 0; assert(!validateConfig(invalid, &error));
    invalid = base; invalid.numHeads = 0; assert(!validateConfig(invalid, &error));
    invalid = base; invalid.dimension = 17; invalid.numHeads = 2; assert(!validateConfig(invalid, &error));
    invalid = base; invalid.dimension = std::numeric_limits<uint32_t>::max(); invalid.feedForwardDimension = std::numeric_limits<uint32_t>::max(); assert(!validateConfig(invalid, &error));
    Config c{}; c.tokens = 4; c.dimension = 8; c.feedForwardDimension = 16; c.numLayers = 2; c.numHeads = 2;
    const auto a = initialParameters(c, 77), b = initialParameters(c, 77);
    const auto ar = parameterRegistry(a), br = parameterRegistry(b);
    assert(ar.size() == br.size()); std::set<std::string> names;
    for (size_t i = 0; i < ar.size(); ++i) { assert(names.insert(ar[i].name).second); assert(ar[i].name == br[i].name); assert(*ar[i].values == *br[i].values); }
    assert(ar.front().name == "token_embedding" && ar.back().name == "output_projection");
    assert(ar[1].name == "layer_000.norm1_gamma");
    assert(ar[3].name == "layer_000.wq");
    assert(ar[11].name == "layer_001.norm1_gamma");
    assert(ar[13].name == "layer_001.wq");
    const std::vector<ParameterInfo> duplicateRanges{
        {"first", ar[0].values}, {"duplicate", ar[0].values}};
    assert(!storageRangesHaveNoAliases(duplicateRanges));
    Config policy = base;
    policy.tokens = std::numeric_limits<uint32_t>::max();
    assert(!validateConfig(policy, &error));
    assert(error.find("APP_POLICY_LIMIT") != std::string::npos ||
           error.find("APP_RESOURCE_ESTIMATOR") != std::string::npos);
}

void testTinyLanguageModelGenericFiniteDifference() {
    using namespace phonelm::tiny_lm;
    Config c{};
    c.vocabularySize = 8;
    c.tokens = 3;
    c.dimension = 8;
    c.feedForwardDimension = 8;
    c.numLayers = 3;
    c.numHeads = 4;
    auto parameters = initialParameters(c, 20260731);
    const auto input = oneHot({0, 1, 2}, c.vocabularySize);
    const auto target = oneHot({1, 2, 3}, c.vocabularySize);
    auto analytic =
        forwardBackwardGeneralized(c, input, target, parameters, 0.0f);
    auto logicalLayer = [](phonelm::qnn::TinyTransformerParameters& p,
                           uint32_t index)
        -> phonelm::qnn::TinyTransformerLayerParameters& {
        return index == 0
            ? static_cast<phonelm::qnn::TinyTransformerLayerParameters&>(p)
            : p.layers.at(index - 1);
    };
    constexpr float epsilon = 1.0e-3f;
    const auto check = [&](std::vector<float>& values, float gradient,
                           size_t index) {
        const float saved = values[index];
        values[index] = saved + epsilon;
        const float plus =
            forwardBackwardGeneralized(c, input, target, parameters, 0.0f).loss;
        values[index] = saved - epsilon;
        const float minus =
            forwardBackwardGeneralized(c, input, target, parameters, 0.0f).loss;
        values[index] = saved;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        assert(std::abs(gradient - numeric) <= 3.0e-3f);
    };
    check(parameters.tokenEmbedding, analytic.gradients.tokenEmbedding[0], 0);
    check(parameters.outputProjection, analytic.gradients.outputProjection[0], 0);
    for (const uint32_t layerIndex : {0u, 1u, 2u}) {
        auto& layer = logicalLayer(parameters, layerIndex);
        auto& gradient = logicalLayer(analytic.gradients, layerIndex);
        // Wq columns 0 and D-1 exercise the first and last head without
        // introducing per-head parameter tensors.
        check(layer.wq, gradient.wq[0], 0);
        check(layer.wq, gradient.wq[c.dimension - 1], c.dimension - 1);
    }
}

void testTinyLanguageModelLegacyGeneralizedRegression() {
    using namespace phonelm::tiny_lm;
    Config c{}; c.tokens = 4; c.dimension = 8; c.feedForwardDimension = 16;
    const auto p = initialParameters(c, 19);
    const auto x = oneHot({0, 1, 2, 3}, c.vocabularySize);
    const auto y = oneHot({1, 2, 3, 4}, c.vocabularySize);
    const auto legacy = forwardBackward(c, x, y, p, .003f);
    const auto generic = forwardBackwardGeneralized(c, x, y, p, .003f);
    assert(legacy.loss == generic.loss && legacy.accuracy == generic.accuracy);
    for (size_t i = 0; i < legacy.logits.size(); ++i) if (legacy.logits[i] != generic.logits[i]) {
        std::cerr << "tiny_lm_generalized_first_logit_divergence index=" << i << " legacy=" << legacy.logits[i] << " generalized=" << generic.logits[i] << '\n';
        assert(false);
    }
    assert(legacy.dLogits == generic.dLogits);
    const auto a = parameterRegistry(legacy.gradients), b = parameterRegistry(generic.gradients);
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < a[i].values->size(); ++j) if ((*a[i].values)[j] != (*b[i].values)[j]) {
        std::cerr << "tiny_lm_generalized_first_gradient_divergence parameter=" << a[i].name << " index=" << j << " legacy=" << (*a[i].values)[j] << " generalized=" << (*b[i].values)[j] << '\n';
        assert(false);
    }
    const auto na = parameterRegistry(legacy.next), nb = parameterRegistry(generic.next);
    for (size_t i = 0; i < na.size(); ++i) assert(*na[i].values == *nb[i].values);
}
}  // namespace

int main() {
    testMatMulAndTranspose();
    testLossGradientsAndSgd();
    testGradientCheck();
    testLossDecrease();
    testTinyLanguageModelGradientCheck();
    testTinyLanguageModelLearning();
    testTinyLanguageModelMultiLayerMultiHead();
    testTinyLanguageModelGeneralizedCoverage();
    testTinyLanguageModelLegacyGeneralizedRegression();
    testTinyLanguageModelSchemaFailClosedAndRegistry();
    testTinyLanguageModelGenericFiniteDifference();
    std::cout << "cpu_reference_tests=PASS\n";
    return 0;
}
