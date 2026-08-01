// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Depth-quality diagnostics probe (host-only, no device required).
//
// Usage:
//   depth_quality_probe trajectory <L> <H> <steps> <seed...> [mode]
//       CPU formal run per seed; prints one per-step CSV block and a summary
//       block per (config,seed[,mode]).
//   depth_quality_probe replay <L> <H> <steps> <seed...>
//       Formal run with checkpoints at 0,1,2,4,8,16,32,64,96,128,160,192,
//       224,256,288,320; prints layer/attention/FFN/gradient/optimizer metric
//       CSV rows per checkpoint.
//   depth_quality_probe paired <shallowL> <deepL> <H> <seed...>
//       Verifies shared-parameter equality across depths.
#include "depth_quality_lib.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace phonelm;
namespace dq = phonelm::depth_quality;

static tiny_lm::Config makeConfig(int layers, int heads) {
  tiny_lm::Config c;
  c.tokens = 8;
  c.vocabularySize = 32;
  c.dimension = 16;
  c.feedForwardDimension = 32;
  c.numLayers = uint32_t(layers);
  c.numHeads = uint32_t(heads);
  std::string error;
  if (!tiny_lm::validateConfig(c, &error)) {
    std::fprintf(stderr, "invalid config: %s\n", error.c_str());
    std::exit(2);
  }
  return c;
}

static void verifyForwardParity(const tiny_lm::Config& c) {
  // Assert the instrumented forward is value-identical to the CPU reference
  // for seeds 1..3 and every batch pattern before trusting any metric.
  for (uint32_t seed = 1; seed <= 3; ++seed) {
    const auto p = tiny_lm::initialParameters(c, seed);
    for (uint32_t pattern = 0; pattern < 4; ++pattern) {
      const auto b = dq::formalBatch(c, pattern, 0);
      const auto ref = tiny_lm::forwardBackward(c, b.first, b.second, p, 0.0f);
      const auto inst = dq::instrumentedForward(c, b.first, b.second, p);
      if (ref.logits != inst.logits || ref.loss != inst.loss ||
          ref.accuracy != inst.accuracy) {
        std::fprintf(stderr,
                     "PARITY_FAILURE seed=%u pattern=%u (instrumented forward "
                     "differs from CPU reference)\n",
                     seed, pattern);
        std::exit(3);
      }
    }
  }
  std::printf("forward_parity=BITWISE_IDENTICAL\n");
}

static int runTrajectory(int layers, int heads, int steps,
                         const std::vector<uint32_t>& seeds,
                         dq::StabilityMode mode) {
  const auto c = makeConfig(layers, heads);
  verifyForwardParity(c);
  std::printf("trajectory_csv_header=configuration,mode,seed,step,loss,accuracy,"
              "gradient_l2,parameter_l2,update_l2,update_to_parameter,"
              "logit_max_abs,target_margin,target_probability\n");
  const std::string id = "t" + std::to_string(c.tokens) + "_d" +
                         std::to_string(c.dimension) + "_f" +
                         std::to_string(c.feedForwardDimension) + "_l" +
                         std::to_string(layers) + "_h" + std::to_string(heads);
  for (uint32_t seed : seeds) {
    const auto run = dq::runFormalCpu(c, seed, steps, 0.003f, mode);
    for (const auto& s : run.steps)
      std::printf("TRAJ,%s,%s,%u,%d,%.9g,%.6g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                  id.c_str(), dq::stabilityModeName(mode), seed, s.step, s.loss,
                  s.accuracy, s.gradientNorm, s.parameterNorm, s.updateNorm,
                  s.updateToParameter, s.logitMaxAbs, s.targetMargin,
                  s.targetProbability);
    const auto& s = run.summary;
    std::printf("SUMMARY,%s,%s,%u,initial=%.9g,min=%.9g,min_step=%d,final=%.9g,"
                "ma_final32=%.9g,max_worsening32=%.9g,last_improvement=%d,"
                "classification=%s\n",
                id.c_str(), dq::stabilityModeName(mode), seed, s.initialLoss,
                s.minimumLoss, s.minimumStep, s.finalLoss,
                s.movingAverageFinalWindow, s.maximumWorseningWindow,
                s.lastImprovementStep, s.classification.c_str());
    std::printf("PHASE1_EVAL,%s,%s,%u,eval_loss=%.9g,eval_accuracy=%.6g\n",
                id.c_str(), dq::stabilityModeName(mode), seed,
                run.finalEvaluation.loss, run.finalEvaluation.accuracy);
    const auto gen = dq::generationQuality(c, run.finalParameters);
    std::printf("GENERATION,%s,%s,%u,oracle_exact=%d/4,free_exact=%d/4\n",
                id.c_str(), dq::stabilityModeName(mode), seed, gen.oracleExact,
                gen.freeExact);
  }
  return 0;
}

static int runReplay(int layers, int heads, int steps,
                     const std::vector<uint32_t>& seeds) {
  const auto c = makeConfig(layers, heads);
  verifyForwardParity(c);
  static const int checkpointSteps[] = {0,  1,  2,  4,  8,  16, 32,  64,
                                        96, 128, 160, 192, 224, 256, 288, 320};
  const std::vector<int> ck(std::begin(checkpointSteps), std::end(checkpointSteps));
  const std::string id = "l" + std::to_string(layers) + "_h" + std::to_string(heads);
  std::printf("replay_csv_header=configuration,seed,step,layer,metric\n");
  for (uint32_t seed : seeds) {
    const auto run = dq::runFormalCpu(c, seed, steps, 0.003f,
                                      dq::StabilityMode::LEGACY, ck);
    for (int step : ck) {
      if (step > steps) continue;
      const auto pIt = run.checkpoints.find(step);
      const auto bIt = run.checkpointBatches.find(step);
      const auto mIt = run.firstMoments.find(step);
      const auto vIt = run.secondMoments.find(step);
      if (pIt == run.checkpoints.end() || bIt == run.checkpointBatches.end())
        continue;
      const auto replay = dq::instrumentedForward(c, bIt->second.first,
                                                  bIt->second.second, pIt->second);
      const auto fb = tiny_lm::forwardBackward(c, bIt->second.first,
                                               bIt->second.second, pIt->second, 0.0f);
      const auto gs = dq::gradientStats(c, fb);
      const auto eval = dq::phase1Evaluation(c, pIt->second);
      std::printf("REPLAY_META,%s,%u,%d,loss=%.9g,accuracy=%.6g,"
                  "eval_loss=%.9g,eval_accuracy=%.6g,"
                  "param_l2=%.9g,grad_l2=%.9g,m_l2=%.9g,v_l2=%.9g\n",
                  id.c_str(), seed, step, replay.loss, replay.accuracy,
                  eval.loss, eval.accuracy,
                  dq::registryNorm(pIt->second), dq::registryNorm(fb.gradients),
                  mIt != run.firstMoments.end() ? dq::registryNorm(mIt->second) : 0.0,
                  vIt != run.secondMoments.end() ? dq::registryNorm(vIt->second) : 0.0);
      for (size_t li = 0; li < replay.layers.size(); ++li) {
        const auto& m = replay.layers[li];
        double headEntropyMean = 0, headMaxP = 0, headMinP = 1.0;
        for (size_t h = 0; h < m.attention.headEntropyMean.size(); ++h) {
          headEntropyMean += m.attention.headEntropyMean[h];
          headMaxP = std::max(headMaxP, m.attention.headMaxProbability[h]);
          headMinP = std::min(headMinP, m.attention.headMinNonzeroProbability[h]);
        }
        if (!m.attention.headEntropyMean.empty())
          headEntropyMean /= m.attention.headEntropyMean.size();
        std::printf(
            "REPLAY,%s,%u,%d,%zu,input_rms=%.6g,input_max=%.6g,"
            "r1_rms=%.6g,out_rms=%.6g,out_max=%.6g,"
            "n1_var=%.6g,n1_gamma=%.6g,n1_beta=%.6g,n2_var=%.6g,n2_gamma=%.6g,n2_beta=%.6g,"
            "score_rms=%.6g,score_max=%.6g,entropy=%.6g,maxp=%.6g,minp=%.3e,"
            "attn_out_norm=%.6g,attn_ratio=%.6g,"
            "f1_rms=%.6g,f1_max=%.6g,ffn_zero=%.4g,act_rms=%.6g,ffn_out_norm=%.6g,ffn_ratio=%.6g,"
            "din_norm=%.6g,dattn_norm=%.6g,dffn_norm=%.6g,dnorm_norm=%.6g\n",
            id.c_str(), seed, step, li, m.inputRms, m.inputMaxAbs, m.residual1Rms,
            m.outputRms, m.outputMaxAbs, m.n1.inputVariance, m.n1.gammaNorm,
            m.n1.betaNorm, m.n2.inputVariance, m.n2.gammaNorm, m.n2.betaNorm,
            m.attention.scoreRms, m.attention.scoreMaxAbs, headEntropyMean,
            headMaxP, headMinP, m.attention.outputNorm,
            m.attention.branchToInputRatio, m.ffn.w1OutputRms,
            m.ffn.w1OutputMaxAbs, m.ffn.activationZeroFraction,
            m.ffn.activationRms, m.ffn.w2OutputNorm, m.ffn.branchToInputRatio,
            gs.inputGradientNorms[li], gs.attentionParameterNorms[li],
            gs.ffnParameterNorms[li], gs.normParameterNorms[li]);
      }
    }
    const auto& s = run.summary;
    std::printf("SUMMARY,%s,%s,%u,initial=%.9g,min=%.9g,min_step=%d,final=%.9g,"
                "ma_final32=%.9g,max_worsening32=%.9g,last_improvement=%d,"
                "classification=%s\n",
                id.c_str(), "LEGACY", seed, s.initialLoss, s.minimumLoss,
                s.minimumStep, s.finalLoss, s.movingAverageFinalWindow,
                s.maximumWorseningWindow, s.lastImprovementStep,
                s.classification.c_str());
  }
  return 0;
}

static int runPaired(int shallowL, int deepL, int heads,
                     const std::vector<uint32_t>& seeds) {
  const auto shallow = makeConfig(shallowL, heads);
  const auto deep = makeConfig(deepL, heads);
  bool ok = true;
  for (uint32_t seed : seeds) {
    const auto ps = tiny_lm::initialParameters(shallow, seed);
    const auto pd = tiny_lm::initialParameters(deep, seed);
    const bool equal = dq::sharedPrefixParametersEqual(ps, pd);
    ok = ok && equal;
    std::printf("PAIRED,seed=%u,shared_prefix_equal=%s\n", seed,
                equal ? "true" : "false");
  }
  std::printf("paired_shared_prefix_all_equal=%s\n", ok ? "true" : "false");
  return ok ? 0 : 4;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: trajectory|replay|paired ...\n");
    return 2;
  }
  const std::string mode = argv[1];
  std::vector<uint32_t> seeds;
  int i = 5;
  dq::StabilityMode stability = dq::StabilityMode::LEGACY;
  if (mode == "trajectory") {
    if (argc < 6) { std::fprintf(stderr, "trajectory <L> <H> <steps> <seed...> [mode]\n"); return 2; }
    for (; i < argc; ++i) {
      dq::StabilityMode parsed;
      if (dq::parseStabilityMode(argv[i], parsed)) stability = parsed;
      else seeds.push_back(uint32_t(std::strtoul(argv[i], nullptr, 10)));
    }
    if (seeds.empty()) seeds = {1, 2, 3, 4, 5};
    return runTrajectory(std::atoi(argv[2]), std::atoi(argv[3]),
                         std::atoi(argv[4]), seeds, stability);
  }
  if (mode == "replay") {
    if (argc < 6) { std::fprintf(stderr, "replay <L> <H> <steps> <seed...>\n"); return 2; }
    for (; i < argc; ++i) seeds.push_back(uint32_t(std::strtoul(argv[i], nullptr, 10)));
    if (seeds.empty()) seeds = {1, 2, 4};
    return runReplay(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), seeds);
  }
  if (mode == "paired") {
    if (argc < 5) { std::fprintf(stderr, "paired <shallowL> <deepL> <H> <seed...>\n"); return 2; }
    for (int j = 5; j < argc; ++j) seeds.push_back(uint32_t(std::strtoul(argv[j], nullptr, 10)));
    if (seeds.empty()) seeds = {1, 2, 3, 4, 5};
    return runPaired(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), seeds);
  }
  std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
  return 2;
}
