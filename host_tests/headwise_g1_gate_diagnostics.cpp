// SPDX-License-Identifier: Apache-2.0
// Checkpoint-static Headwise G1 gate diagnostics on fixed evaluation windows.
// Uses production CPU forward (LN1 -> G = sigmoid(N @ Wg)) so gates match
// training/QNN semantics. Not a training-trajectory aggregate.
#include "nicopedia_byte_bpe.h"
#include "nicopedia_muon_checkpoint.h"
#include "tiny_language_model_cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace phonelm;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

struct WindowSet {
  std::vector<std::vector<std::uint16_t>> windows;
  std::string contentHash;
};

WindowSet loadWindows(const std::string& cachePath,
                      const nicopedia_bpe::Model& model,
                      std::uint32_t tokens,
                      std::uint32_t limit) {
  const auto source = nicopedia_bpe::loadCache(cachePath, model);
  WindowSet out;
  out.contentHash = source.tokenizerHash;
  const std::uint32_t n =
      std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(source.records.size()));
  out.windows.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto& rec = source.records[i];
    if (rec.window.size() < tokens) continue;
    out.windows.emplace_back(rec.window.end() - static_cast<std::ptrdiff_t>(tokens),
                             rec.window.end());
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      std::cerr << "usage: headwise_g1_gate_diagnostics CHECKPOINT "
                   "TOKENIZER VALIDATION CACHE_TOKENS LIMIT\n";
      return 2;
    }
    const std::string checkpointPath = argv[1];
    const std::string tokenizerPath = argv[2];
    const std::string validationPath = argv[3];
    const std::uint32_t tokens = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const std::uint32_t limit = static_cast<std::uint32_t>(std::stoul(argv[5]));

    std::string error;
    nicopedia_muon_checkpoint::Checkpoint decoded;
    {
      std::ifstream input(checkpointPath, std::ios::binary);
      require(bool(input), "CHECKPOINT_OPEN_FAILED");
      input.seekg(0, std::ios::end);
      const auto size = input.tellg();
      input.seekg(0);
      std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
      input.read(reinterpret_cast<char*>(bytes.data()), size);
      require(nicopedia_muon_checkpoint::decodeCheckpoint(bytes, &decoded, &error),
              error.c_str());
    }
    const auto& cfg = decoded.identity.config;
    require(cfg.attentionGate == tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID,
            "CHECKPOINT_NOT_GATED");
    qnn::TinyTransformerParameters parameters;
    require(nicopedia_muon_checkpoint::extractParameters(
                decoded, cfg, decoded.identity.seed, &parameters, &error),
            error.c_str());

    const auto model = nicopedia_bpe::loadModel(tokenizerPath);
    const auto windows = loadWindows(validationPath, model, tokens, limit);
    require(!windows.windows.empty(), "NO_VALIDATION_WINDOWS");

    const std::uint32_t layers = cfg.numLayers;
    const std::uint32_t heads = cfg.numHeads;
    struct Agg {
      double sum = 0, sumSq = 0, minV = 1.0, maxV = 0.0;
      std::uint64_t count = 0, below = 0, above = 0;
    };
    std::vector<Agg> aggs(std::size_t(layers) * heads);

    std::uint64_t windowsUsed = 0;
    for (const auto& window : windows.windows) {
      tiny_lm::Config local = cfg;
      local.tokens = tokens;
      const auto trace = tiny_lm::forwardTraceGeneralized(
          local, tiny_lm::oneHot(std::vector<std::uint32_t>(window.begin(), window.end()),
                                 local.vocabularySize),
          parameters);
      require(trace.layers.size() == layers, "LAYER_COUNT_MISMATCH");
      for (std::uint32_t li = 0; li < layers; ++li) {
        const auto& gates = trace.layers[li].gates;
        require(gates.size() == std::size_t(tokens) * heads, "GATE_SHAPE_MISMATCH");
        for (std::uint32_t t = 0; t < tokens; ++t) {
          for (std::uint32_t h = 0; h < heads; ++h) {
            const float g = gates[std::size_t(t) * heads + h];
            auto& a = aggs[std::size_t(li) * heads + h];
            a.sum += g;
            a.sumSq += double(g) * g;
            a.minV = std::min<double>(a.minV, g);
            a.maxV = std::max<double>(a.maxV, g);
            if (g < 0.1f) ++a.below;
            if (g > 0.9f) ++a.above;
            ++a.count;
          }
        }
      }
      ++windowsUsed;
    }

    std::cout << "headwise_g1_gate_diagnostics=PASS\n"
              << "gate_diagnostics_kind=checkpoint_static\n"
              << "checkpoint_step=" << decoded.identity.globalStep << '\n'
              << "checkpoint_format="
              << (cfg.attentionGate == tiny_lm::AttentionGate::HEADWISE_G1_SIGMOID
                      ? "NPRTCKPTV5"
                      : "unknown")
              << '\n'
              << "windows_used=" << windowsUsed << '\n'
              << "tokens_per_window=" << tokens << '\n'
              << "validation_cache=" << windows.contentHash << '\n'
              << "layers=" << layers << '\n'
              << "heads=" << heads << '\n';

    double sumHeadMeans = 0;
    double minHeadMean = 1.0;
    double maxHeadMean = 0.0;
    std::size_t headCount = 0;
    for (std::uint32_t li = 0; li < layers; ++li) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        const auto& a = aggs[std::size_t(li) * heads + h];
        require(a.count > 0, "EMPTY_AGG");
        const double mean = a.sum / a.count;
        const double var = std::max(0.0, a.sumSq / a.count - mean * mean);
        sumHeadMeans += mean;
        minHeadMean = std::min(minHeadMean, mean);
        maxHeadMean = std::max(maxHeadMean, mean);
        ++headCount;
        const std::string p = "gate_static_l" + std::to_string(li) + "_h" + std::to_string(h);
        std::cout << p << "_mean=" << mean << '\n'
                  << p << "_stddev=" << std::sqrt(var) << '\n'
                  << p << "_min=" << a.minV << '\n'
                  << p << "_max=" << a.maxV << '\n'
                  << p << "_below_0_1_fraction=" << (double(a.below) / a.count) << '\n'
                  << p << "_above_0_9_fraction=" << (double(a.above) / a.count) << '\n';
      }
    }
    std::cout << "gate_static_mean_of_head_means=" << (sumHeadMeans / headCount) << '\n'
              << "gate_static_min_head_mean=" << minHeadMean << '\n'
              << "gate_static_max_head_mean=" << maxHeadMean << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "headwise_g1_gate_diagnostics=FAIL\nerror=" << e.what() << '\n';
    return 1;
  }
}
