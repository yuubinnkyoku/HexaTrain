// SPDX-License-Identifier: Apache-2.0
#include "nicopedia_muon_checkpoint.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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

    tiny_lm::Config small;
    small.vocabularySize = 8;
    small.tokens = 3;
    small.dimension = 4;
    small.feedForwardDimension = 8;
    small.numLayers = 2;
    small.numHeads = 2;
    small.attentionGate = tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID;
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

    const auto baseEstimate = tiny_lm::resourceEstimate(baseline);
    const auto gatedEstimate = tiny_lm::resourceEstimate(gated);
    std::cout << "headwise_g1_gate=PASS\n"
              << "parameter_elements=760960\nmuon_matrices=114\n"
              << "muon_elements=622592\naux_adam_elements=138368\n"
              << "gate_mean=" << sum / count << "\ngate_min=" << minimum
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
