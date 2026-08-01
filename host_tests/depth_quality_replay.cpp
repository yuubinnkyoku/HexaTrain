// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Checkpoint replay (decode + CPU replay diagnostics) for the private
// phase-B checkpoint files. Raw tensors stay private; this tool prints only
// aggregate metrics.
//
// Usage:
//   depth_quality_replay <checkpoint.bin>...
// Prints one REPLAY line per layer plus a META line per checkpoint.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "depth_quality_lib.h"
#include "qnn/qnn_first_nonfinite_diagnostics.h"

namespace dq = phonelm::depth_quality;
namespace ff = phonelm::qnn::first_nonfinite;
namespace tiny_lm = phonelm::tiny_lm;
namespace qnn = phonelm::qnn;

static std::vector<std::uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: depth_quality_replay <checkpoint.bin>...\n");
    return 2;
  }
  std::printf("replay_csv_header=seed,step,layer,metrics...\n");
  for (int i = 1; i < argc; ++i) {
    const auto bytes = readFile(argv[i]);
    ff::Checkpoint checkpoint;
    std::string error;
    if (!ff::decodeCheckpoint(bytes, &checkpoint, &error)) {
      std::fprintf(stderr, "decode failed for %s: %s\n", argv[i], error.c_str());
      return 3;
    }
    tiny_lm::Config config;
    config.tokens = checkpoint.config.tokens;
    config.vocabularySize = checkpoint.config.vocabularySize;
    config.dimension = checkpoint.config.dimension;
    config.feedForwardDimension = checkpoint.config.feedForwardDimension;
    config.epsilon = checkpoint.config.epsilon;
    config.numLayers = checkpoint.config.numLayers;
    config.numHeads = checkpoint.config.numHeads;
    // Rebuild shapes from the registry order embedded in the checkpoint.
    qnn::TinyTransformerParameters shape;
    try {
      shape = tiny_lm::initialParameters(config, checkpoint.seed);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "shape rebuild failed: %s\n", e.what());
      return 3;
    }
    // Bind checkpoint fields into the canonical parameter structure.
    const auto registry = tiny_lm::parameterRegistry(shape);
    size_t offset = 0;
    for (const auto& entry : registry) {
      std::memcpy(const_cast<float*>(entry.values->data()),
                  checkpoint.parameters.data() + offset,
                  entry.values->size() * sizeof(float));
      offset += entry.values->size();
    }
    const auto replay = dq::instrumentedForward(config, checkpoint.input,
                                                checkpoint.target, shape);
    const auto fb = tiny_lm::forwardBackward(config, checkpoint.input,
                                             checkpoint.target, shape, 0.0f);
    if (fb.logits != replay.logits) {
      std::fprintf(stderr, "instrumented forward parity failure\n");
      return 4;
    }
    const auto gs = dq::gradientStats(config, fb);
    std::vector<float> flatGradients;
    for (const auto& e : tiny_lm::parameterRegistry(fb.gradients))
      flatGradients.insert(flatGradients.end(), e.values->begin(),
                           e.values->end());
    std::printf("META,seed=%u,step=%u,training_stability_mode=%s,"
                "depth_pair_init_mode=%s,loss=%.9g,"
                "param_l2=%.9g,grad_l2=%.9g,m_l2=%.9g,v_l2=%.9g\n",
                checkpoint.seed, checkpoint.completedStep,
                phonelm::trainingStabilityModeName(
                    checkpoint.config.trainingStabilityMode),
                phonelm::depthPairInitModeName(
                    checkpoint.config.depthPairInitMode),
                double(replay.loss),
                dq::registryNorm(shape), dq::l2(flatGradients),
                dq::l2(checkpoint.adamM), dq::l2(checkpoint.adamV));
    for (size_t li = 0; li < replay.layers.size(); ++li) {
      const auto& m = replay.layers[li];
      std::ostringstream heads;
      for (size_t h = 0; h < m.attention.headEntropyMean.size(); ++h) {
        heads << (h ? ";" : "") << "h" << h << "_ent="
              << m.attention.headEntropyMean[h] << "_maxp="
              << m.attention.headMaxProbability[h] << "_minp="
              << m.attention.headMinNonzeroProbability[h];
      }
      std::printf("LAYER,seed=%u,step=%u,layer=%zu,input_rms=%.6g,"
                  "input_max=%.6g,r1_rms=%.6g,out_rms=%.6g,out_max=%.6g,"
                  "n1_var=%.6g,n1_gamma=%.6g,n1_beta=%.6g,n2_var=%.6g,"
                  "n2_gamma=%.6g,n2_beta=%.6g,score_rms=%.6g,score_max=%.6g,"
                  "%s,attn_out_norm=%.6g,attn_ratio=%.6g,f1_rms=%.6g,"
                  "f1_max=%.6g,ffn_zero=%.4g,act_rms=%.6g,ffn_out_norm=%.6g,"
                  "ffn_ratio=%.6g,din=%.6g,dattn=%.6g,dffn=%.6g,dnorm=%.6g\n",
                  checkpoint.seed, checkpoint.completedStep, li, m.inputRms,
                  m.inputMaxAbs, m.residual1Rms, m.outputRms, m.outputMaxAbs,
                  m.n1.inputVariance, m.n1.gammaNorm, m.n1.betaNorm,
                  m.n2.inputVariance, m.n2.gammaNorm, m.n2.betaNorm,
                  m.attention.scoreRms, m.attention.scoreMaxAbs,
                  heads.str().c_str(), m.attention.outputNorm,
                  m.attention.branchToInputRatio, m.ffn.w1OutputRms,
                  m.ffn.w1OutputMaxAbs, m.ffn.activationZeroFraction,
                  m.ffn.activationRms, m.ffn.w2OutputNorm,
                  m.ffn.branchToInputRatio, gs.inputGradientNorms[li],
                  gs.attentionParameterNorms[li], gs.ffnParameterNorms[li],
                  gs.normParameterNorms[li]);
    }
  }
  return 0;
}
