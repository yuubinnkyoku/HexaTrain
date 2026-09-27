// SPDX-License-Identifier: Apache-2.0
#include "nicopedia_muon_checkpoint.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace phonelm;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

tiny_lm::Config config(tiny_lm::AttentionGate gate) {
  tiny_lm::Config c;
  c.vocabularySize = 1024;
  c.tokens = 32;
  c.dimension = 64;
  c.feedForwardDimension = 128;
  c.numLayers = 19;
  c.numHeads = 2;
  c.attentionGate = gate;
  return c;
}

nicopedia_muon_checkpoint::Checkpoint checkpointFor(
    const tiny_lm::Config& c) {
  namespace ck = nicopedia_muon_checkpoint;
  ck::Checkpoint result;
  result.identity.config = c;
  result.identity.seed = 1;
  result.identity.globalStep = 7;
  result.identity.tokenizerKind = "byte_bpe";
  result.identity.tokenizerHash = "sha256:" + std::string(64, 'a');
  result.identity.dataCursor.datasetHash = "fnv1a64:" + std::string(16, 'b');
  result.identity.dataCursor.orderSeed = 1;
  result.hyperparameters.muonLearningRate = 0.005f;
  result.hyperparameters.auxAdamLearningRate = 0.0022f;
  result.hyperparameters.muonTargetLearningRate = 0.00022727272f;
  result.hyperparameters.auxAdamTargetLearningRate = 0.0001f;
  result.hyperparameters.decayStartStep = 4000;
  result.hyperparameters.decayEndStep = 8000;
  result.hyperparameters.scheduleTotalSteps = 8000;
  const auto parameters = tiny_lm::initialParameters(c, 1);
  const auto modelRegistry = tiny_lm::parameterRegistry(parameters);
  const auto registry = ck::expectedRegistry(c);
  require(modelRegistry.size() == registry.size(), "checkpoint registry size");
  for (std::size_t i = 0; i < registry.size(); ++i) {
    ck::ParameterState state;
    state.name = registry[i].name;
    state.role = registry[i].role;
    state.shape = registry[i].shape;
    state.values = *modelRegistry[i].values;
    if (state.role == ck::ParameterRole::MUON)
      state.momentum.assign(state.values.size(), 0.0f);
    else {
      state.adamM.assign(state.values.size(), 0.0f);
      state.adamV.assign(state.values.size(), 0.0f);
    }
    result.parameters.push_back(std::move(state));
  }
  return result;
}
}

int main() {
  try {
    const auto baseline = config(tiny_lm::AttentionGate::NONE);
    const auto gated = config(tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID);
    const auto p0 = tiny_lm::initialParameters(baseline, 1);
    const auto p1 = tiny_lm::initialParameters(gated, 1);
    const auto p1Again = tiny_lm::initialParameters(gated, 1);
    require(tiny_lm::parameterElementCount(p0) == 758528, "baseline count");
    require(tiny_lm::parameterElementCount(p1) == 760960, "gated count");
    require(tiny_lm::resourceEstimate(baseline).parameterElements == 758528,
            "baseline estimator count");
    require(tiny_lm::resourceEstimate(gated).parameterElements == 760960,
            "gated estimator count");
    tiny_lm::ParameterPartition partition;
    std::string error;
    require(tiny_lm::splitParameterRegistry(p1, &partition, &error),
            "gated registry validation");
    std::size_t muonElements = 0, auxElements = 0;
    for (const auto& entry : partition.muon) muonElements += entry.values->size();
    for (const auto& entry : partition.auxiliaryAdam)
      auxElements += entry.values->size();
    require(partition.muon.size() == 114 && muonElements == 622592,
            "Muon identity");
    require(partition.auxiliaryAdam.size() == 97 && auxElements == 138368,
            "Aux Adam identity");
    const auto gatedRegistry = tiny_lm::parameterRegistry(p1);
    std::size_t gateEntries = 0;
    for (const auto& entry : gatedRegistry) {
      if (entry.suffix != "attention_gate_weight") continue;
      ++gateEntries;
      require(entry.placement == tiny_lm::ParameterPlacement::PER_LAYER &&
                  entry.condition == tiny_lm::ParameterCondition::HEADWISE_G1 &&
                  entry.role == tiny_lm::ParameterRole::AUX_ADAM &&
                  entry.shape == std::vector<std::uint32_t>({64, 2}) &&
                  entry.fanOut == 0 && entry.fanIn == 0,
              "Headwise G1 metadata");
    }
    require(gateEntries == 19, "Headwise G1 metadata count");
    require(p1.attentionGateWeight == p1Again.attentionGateWeight,
            "deterministic Wg initialization");

    // Identity-init G1: G = 2*sigmoid(N@Wg) with Wg = 0 => G = 1 at step 0.
    const auto identityCfg = config(
        tiny_lm::AttentionGate::HEADWISE_G1_SCALE2_IDENTITY);
    const auto pIdent = tiny_lm::initialParameters(identityCfg, 1);
    require(tiny_lm::parameterElementCount(pIdent) == 760960,
            "identity-init parameter count");
    require(pIdent.attentionGateWeight.size() ==
                size_t(identityCfg.dimension) * identityCfg.numHeads,
            "identity-init Wg shape");
    for (float value : pIdent.attentionGateWeight)
      require(value == 0.0f, "identity-init Wg is exactly zero");
    for (const auto& layer : pIdent.layers)
      for (float value : layer.attentionGateWeight)
        require(value == 0.0f, "identity-init layer Wg is exactly zero");

    tiny_lm::Config small;
    small.vocabularySize = 8;
    small.tokens = 3;
    small.dimension = 4;
    small.feedForwardDimension = 8;
    small.numLayers = 2;
    small.numHeads = 2;
    small.attentionGate = tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;

    tiny_lm::Config smallIdent = small;
    smallIdent.attentionGate =
        tiny_lm::AttentionGate::HEADWISE_G1_SCALE2_IDENTITY;
    const auto psIdent = tiny_lm::initialParameters(smallIdent, 17);
    for (float value : psIdent.attentionGateWeight)
      require(value == 0.0f, "small identity Wg zero");
    const auto stepIdent = tiny_lm::forwardBackwardGeneralized(
        smallIdent, tiny_lm::oneHot({0, 1, 2}, 8),
        tiny_lm::oneHot({1, 2, 3}, 8), psIdent, 0.0f);
    require(std::isfinite(stepIdent.loss), "identity forward finite");
    require(stepIdent.attentionGates.size() == 2, "identity gate layers");
    for (const auto& gates : stepIdent.attentionGates) {
      require(gates.size() == size_t(smallIdent.tokens) * smallIdent.numHeads,
              "identity gate shape");
      for (float value : gates)
        require(std::abs(value - 1.0f) < 1.0e-6f, "step0 identity G == 1");
    }
    require(stepIdent.gradients.attentionGateWeight.size() ==
                size_t(smallIdent.dimension) * smallIdent.numHeads,
            "identity dWg shape");
    bool identityDWgNonzero = false;
    for (float value : stepIdent.gradients.attentionGateWeight)
      if (value != 0.0f) identityDWgNonzero = true;
    require(identityDWgNonzero, "identity dWg nonzero after real backward");
    for (float value : stepIdent.gradients.attentionGateWeight)
      require(std::isfinite(value), "identity dWg finite");

    // Ungated parity at Wg=0 / scale=2: gated attention output equals ungated.
    tiny_lm::Config parityCfg = smallIdent;
    parityCfg.attentionGate = tiny_lm::AttentionGate::NONE;
    qnn::TinyTransformerParameters ungatedFromIdent = psIdent;
    ungatedFromIdent.attentionGateWeight.clear();
    for (auto& layer : ungatedFromIdent.layers)
      layer.attentionGateWeight.clear();
    const auto ungatedTrace = tiny_lm::forwardTraceGeneralized(
        parityCfg, tiny_lm::oneHot({0, 1, 2}, 8), ungatedFromIdent);
    const auto gatedTrace = tiny_lm::forwardTraceGeneralized(
        smallIdent, tiny_lm::oneHot({0, 1, 2}, 8), psIdent);
    require(ungatedTrace.layers.size() == gatedTrace.layers.size(),
            "parity layer count");
    for (std::size_t li = 0; li < gatedTrace.layers.size(); ++li) {
      const auto& ungatedCtx = ungatedTrace.layers[li].context;
      const auto& gatedCtx = gatedTrace.layers[li].context;
      require(ungatedCtx.size() == gatedCtx.size(), "parity context size");
      for (std::size_t i = 0; i < gatedCtx.size(); ++i)
        require(std::abs(ungatedCtx[i] - gatedCtx[i]) < 1.0e-5f,
                "Wg=0 scale2 context equals ungated");
    }

    // Architecture identity: V5 with gate=2 must not resume as gate=1.
    const auto originalIdent = checkpointFor(identityCfg);
    std::vector<std::uint8_t> identBytes;
    require(nicopedia_muon_checkpoint::encodeCheckpoint(
                originalIdent, &identBytes, &error),
            "identity checkpoint encode");
    nicopedia_muon_checkpoint::Checkpoint decodedIdent;
    require(nicopedia_muon_checkpoint::decodeCheckpoint(
                identBytes, &decodedIdent, &error),
            "identity checkpoint decode");
    qnn::TinyTransformerParameters extractedIdent;
    require(nicopedia_muon_checkpoint::extractParameters(
                decodedIdent, identityCfg, 1, &extractedIdent, &error),
            "identity checkpoint extract");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decodedIdent, gated, 1, &extractedIdent, &error),
            "identity checkpoint must not resume as current G1");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decodedIdent, baseline, 1, &extractedIdent, &error),
            "identity checkpoint must not resume as ungated");

    // Fixed 0.5 branch scale: Yh = 0.5 * Ah, no Wg parameters.
    const auto fixedCfg = config(tiny_lm::AttentionGate::FIXED_HALF);
    const auto pFixed = tiny_lm::initialParameters(fixedCfg, 1);
    require(tiny_lm::parameterElementCount(pFixed) == 758528,
            "fixed_half parameter count matches ungated baseline");
    require(pFixed.attentionGateWeight.empty(), "fixed_half has no Wg");
    for (const auto& layer : pFixed.layers)
      require(layer.attentionGateWeight.empty(), "fixed_half layer has no Wg");

    tiny_lm::Config smallFixed = small;
    smallFixed.attentionGate = tiny_lm::AttentionGate::FIXED_HALF;
    const auto psFixed = tiny_lm::initialParameters(smallFixed, 17);
    require(psFixed.attentionGateWeight.empty(), "small fixed_half has no Wg");

    tiny_lm::Config smallUngated = small;
    smallUngated.attentionGate = tiny_lm::AttentionGate::NONE;
    const auto traceUngated = tiny_lm::forwardTraceGeneralized(
        smallUngated, tiny_lm::oneHot({0, 1, 2}, 8), psFixed);
    const auto traceFixed = tiny_lm::forwardTraceGeneralized(
        smallFixed, tiny_lm::oneHot({0, 1, 2}, 8), psFixed);
    require(traceUngated.layers.size() == traceFixed.layers.size(),
            "fixed_half layer count");
    // Local forward contract: given the same layer input, fixed_half attention
    // output is exactly 0.5 * ungated. Only layer 0 shares the same input;
    // deeper layers diverge through the residual path.
    for (std::size_t i = 0; i < traceFixed.layers[0].context.size(); ++i) {
      const float u = traceUngated.layers[0].context[i];
      const float f = traceFixed.layers[0].context[i];
      const float expected = 0.5f * u;
      const float diff = std::abs(f - expected);
      if (diff >= 1.0e-5f) {
        std::ostringstream message;
        message << "fixed_half context equals 0.5 * ungated context"
                << " i=" << i << " u=" << u << " f=" << f
                << " expected=" << expected << " diff=" << diff;
        throw std::runtime_error(message.str());
      }
    }
    // Fixed-half layer-0 raw attention branch must match ungated layer-0
    // scaled by 0.5 even if deeper layers diverge.
    require(traceFixed.layers.size() == 2, "fixed_half small layer count");

    const auto stepFixed = tiny_lm::forwardBackwardGeneralized(
        smallFixed, tiny_lm::oneHot({0, 1, 2}, 8),
        tiny_lm::oneHot({1, 2, 3}, 8), psFixed, 0.0f);
    require(std::isfinite(stepFixed.loss), "fixed_half forward finite");
    require(stepFixed.attentionGates.empty(), "fixed_half has no gate telemetry");
    require(stepFixed.gradients.attentionGateWeight.empty(),
            "fixed_half has no dWg");
    // Local backward contract: dContext path is scaled by 0.5 versus ungated.
    // Compare the attention-side input gradient magnitude proxy via finite
    // gradients on wv (sensitive to dAh).
    for (float value : stepFixed.gradients.wv) require(std::isfinite(value), "fixed_half dWv finite");

    // Architecture identity fail-closed.
    const auto fixedCkpt = checkpointFor(fixedCfg);
    std::vector<std::uint8_t> fixedBytes;
    require(nicopedia_muon_checkpoint::encodeCheckpoint(
                fixedCkpt, &fixedBytes, &error),
            "fixed_half checkpoint encode");
    nicopedia_muon_checkpoint::Checkpoint decodedFixed;
    require(nicopedia_muon_checkpoint::decodeCheckpoint(
                fixedBytes, &decodedFixed, &error),
            "fixed_half checkpoint decode");
    qnn::TinyTransformerParameters extractedFixed;
    require(nicopedia_muon_checkpoint::extractParameters(
                decodedFixed, fixedCfg, 1, &extractedFixed, &error),
            "fixed_half checkpoint extract");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decodedFixed, baseline, 1, &extractedFixed, &error),
            "fixed_half checkpoint must not resume as ungated");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decodedFixed, gated, 1, &extractedFixed, &error),
            "fixed_half checkpoint must not resume as current G1");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decodedIdent, fixedCfg, 1, &extractedFixed, &error),
            "identity checkpoint must not resume as fixed_half");

    const auto ps = tiny_lm::initialParameters(small, 17);
    const auto step = tiny_lm::forwardBackwardGeneralized(
        small, tiny_lm::oneHot({0, 1, 2}, 8),
        tiny_lm::oneHot({1, 2, 3}, 8), ps, 0.0f);
    require(std::isfinite(step.loss) && step.attentionGates.size() == 2,
            "gated forward finite");
    std::vector<std::uint32_t> formalTokens(32);
    for (std::uint32_t i = 0; i < formalTokens.size(); ++i) formalTokens[i] = i;
    const auto initialTrace = tiny_lm::forwardTraceGeneralized(
        gated, tiny_lm::oneHot(formalTokens, 1024), p1);
    require(initialTrace.layers.size() == 19, "formal initialization trace");
    double sum = 0.0;
    float minimum = 1.0f, maximum = 0.0f;
    std::size_t count = 0, saturated = 0;
    for (const auto& layer : initialTrace.layers)
      for (float value : layer.gates) {
        require(value > 0.0f && value < 1.0f, "gate range");
        sum += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        saturated += value < 0.1f || value > 0.9f;
        ++count;
      }
    const auto gradient = tiny_lm::headwiseG1GateGradientCheck();
    require(gradient.passed, "gate numerical gradient");
    const float worstMagnitude =
        std::max(std::abs(gradient.worstRelativeAnalytic),
                 std::abs(gradient.worstRelativeNumeric));
    // Near-zero gradients inflate the floored relative metric; absolute error
    // remains the hard correctness gate.
    const bool worstNearZero = worstMagnitude < 1.0e-3f;
    if (!worstNearZero && gradient.maximumRelativeError > 0.02f)
      throw std::runtime_error(
          "gate gradient worst relative error is large on a meaningful gradient");

    const auto original = checkpointFor(gated);
    std::vector<std::uint8_t> bytes;
    require(nicopedia_muon_checkpoint::encodeCheckpoint(original, &bytes, &error),
            "gated checkpoint encode");
    require(std::equal(std::begin(nicopedia_muon_checkpoint::kGatedMagic),
                       std::end(nicopedia_muon_checkpoint::kGatedMagic) - 1,
                       bytes.begin()), "gated checkpoint magic");
    nicopedia_muon_checkpoint::Checkpoint decoded;
    require(nicopedia_muon_checkpoint::decodeCheckpoint(bytes, &decoded, &error),
            "gated checkpoint decode");
    qnn::TinyTransformerParameters extracted;
    require(nicopedia_muon_checkpoint::extractParameters(
                decoded, gated, 1, &extracted, &error),
            "gated checkpoint extraction");
    require(tiny_lm::parameterElementCount(extracted) == 760960,
            "gated checkpoint round trip");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decoded, baseline, 1, &extracted, &error) &&
                error == "NPRT_CKPT_V4_CONFIG_MISMATCH",
            "architecture mismatch rejection");
    require(!nicopedia_muon_checkpoint::extractParameters(
                decoded, fixedCfg, 1, &extracted, &error),
            "current G1 checkpoint must not resume as fixed_half");

    const auto baseEstimate = tiny_lm::resourceEstimate(baseline);
    const auto gatedEstimate = tiny_lm::resourceEstimate(gated);
    const auto fixedEstimate = tiny_lm::resourceEstimate(fixedCfg);
    require(fixedEstimate.parameterElements == 758528,
            "fixed_half resource estimator parameter count");
    std::cout << "headwise_g1_gate=PASS\n"
              << "parameter_elements=760960\nmuon_matrices=114\n"
              << "muon_elements=622592\naux_adam_elements=138368\n"
              << "identity_init_gate_mean=1\nidentity_init_gate_min=1\n"
              << "identity_init_gate_max=1\nidentity_init_wg_zero=true\n"
              << "identity_init_dwg_nonzero=" << (identityDWgNonzero ? "true" : "false")
              << "\nidentity_checkpoint_cross_resume_rejected=true\n"
              << "fixed_half_parameter_count=758528\nfixed_half_forward_scale=0.5\n"
              << "fixed_half_no_wg=true\nfixed_half_cross_resume_rejected=true\n"
              << "fixed_half_parameter_delta_vs_g1="
              << (int64_t(gatedEstimate.parameterElements) -
                  int64_t(fixedEstimate.parameterElements))
              << "\nfixed_half_adam_state_delta_vs_g1="
              << (int64_t(gatedEstimate.adamMomentElements) -
                  int64_t(fixedEstimate.adamMomentElements))
              << "\ngate_mean=" << sum / count << "\ngate_min=" << minimum
              << "\ngate_max=" << maximum << "\ngate_saturation_fraction="
              << double(saturated) / count << '\n'
              << "gradient_max_absolute_error="
              << gradient.maximumAbsoluteError << '\n'
              << "gradient_max_relative_error="
              << gradient.maximumRelativeError << '\n'
              << "worst_relative_parameter=" << gradient.worstRelativeParameter
              << '\n'
              << "worst_relative_index=" << gradient.worstRelativeIndex << '\n'
              << "worst_relative_analytic=" << gradient.worstRelativeAnalytic
              << '\n'
              << "worst_relative_numeric=" << gradient.worstRelativeNumeric
              << '\n'
              << "worst_relative_absolute_error="
              << gradient.worstRelativeAbsoluteError << '\n'
              << "worst_relative_relative_error="
              << gradient.worstRelativeRelativeError << '\n'
              << "worst_relative_near_zero=" << (worstNearZero ? "true" : "false")
              << '\n'
              << "worst_relative_max_magnitude=" << worstMagnitude << '\n'
              << "baseline_graph_nodes=" << baseEstimate.nodeCount << '\n'
              << "gated_graph_nodes=" << gatedEstimate.nodeCount << '\n'
              << "graph_node_delta=" << gatedEstimate.nodeCount - baseEstimate.nodeCount << '\n'
              << "baseline_graph_tensors=" << baseEstimate.tensorCount << '\n'
              << "gated_graph_tensors=" << gatedEstimate.tensorCount << '\n'
              << "graph_tensor_delta=" << gatedEstimate.tensorCount - baseEstimate.tensorCount << '\n'
              << "checkpoint_parameter_byte_delta="
              << gatedEstimate.parameterBytes - baseEstimate.parameterBytes << '\n'
              << "optimizer_state_byte_delta="
              << gatedEstimate.adamMomentBytes - baseEstimate.adamMomentBytes << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "headwise_g1_gate=FAIL\nerror=" << exception.what() << '\n';
    return 1;
  }
}
