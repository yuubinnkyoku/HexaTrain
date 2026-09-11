// SPDX-License-Identifier: Apache-2.0
#include "nicopedia_htp_muon.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    using namespace phonelm;
    tiny_lm::Config config{1024, 32, 64, 128, 1e-5f, 19, 2};
    auto parameters = tiny_lm::initialParameters(config, 1);
    auto gradients = parameters;
    auto momentum = parameters;
    auto original = parameters;
    auto registry = tiny_lm::parameterRegistry(parameters);
    auto gradientRegistry = tiny_lm::parameterRegistry(gradients);
    auto momentumRegistry = tiny_lm::parameterRegistry(momentum);
    for (std::size_t i = 0; i < registry.size(); ++i) {
      auto* p = const_cast<std::vector<float>*>(registry[i].values);
      auto* g = const_cast<std::vector<float>*>(gradientRegistry[i].values);
      auto* m = const_cast<std::vector<float>*>(momentumRegistry[i].values);
      for (std::size_t j = 0; j < p->size(); ++j) {
        (*p)[j] = float(i * 100000 + j);
        (*g)[j] = float(-int(i * 100000 + j));
        (*m)[j] = float(i * 100000 + j) + 0.25f;
      }
    }
    original = parameters;
    nicopedia_htp_muon::PackedInputs packed;
    std::string error;
    require(nicopedia_htp_muon::pack(parameters, gradients, momentum, &packed,
                                     &error), error.c_str());
    require(packed.squareBindings.size() == 76, "square batch count");
    require(packed.rectangularBindings.size() == 38, "rect batch count");
    require(packed.scaleRectangular.size() == 38, "rect scale count");
    for (std::size_t i = 0; i < 19; ++i)
      require(std::abs(packed.scaleRectangular[i] - std::sqrt(2.0f)) < 1e-7f,
              "W1 scale");
    for (std::size_t i = 19; i < 38; ++i)
      require(packed.scaleRectangular[i] == 1.0f, "W2 scale");
    require(!packed.rectangularBindings.front().transposed,
            "first W1 orientation");
    require(packed.rectangularBindings[19].transposed,
            "first W2 orientation");
    require(packed.rectangularBindings.front().name == "layer_000.ffn_w1",
            "first W1 identity");
    require(packed.rectangularBindings[18].name == "layer_018.ffn_w1",
            "last W1 identity");
    require(packed.rectangularBindings[19].name == "layer_000.ffn_w2",
            "first W2 identity");
    require(packed.rectangularBindings[37].name == "layer_018.ffn_w2",
            "last W2 identity");
    auto restored = original;
    auto restoredMomentum = momentum;
    require(nicopedia_htp_muon::unpack(
                packed, packed.currentSquare, packed.momentumSquare,
                packed.currentRectangular, packed.momentumRectangular,
                &restored, &restoredMomentum, &error), error.c_str());
    const auto expected = tiny_lm::parameterRegistry(original);
    const auto actual = tiny_lm::parameterRegistry(restored);
    const auto expectedMomentum = tiny_lm::parameterRegistry(momentum);
    const auto actualMomentum = tiny_lm::parameterRegistry(restoredMomentum);
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (expected[i].role != tiny_lm::ParameterRole::MUON) continue;
      require(*expected[i].values == *actual[i].values, "weight roundtrip");
      require(*expectedMomentum[i].values == *actualMomentum[i].values,
              "momentum roundtrip");
    }
    gradients.wq.front() = std::numeric_limits<float>::quiet_NaN();
    require(!nicopedia_htp_muon::pack(parameters, gradients, momentum, &packed,
                                      &error) &&
                error.find("HTP_MUON_PACK_NONFINITE") != std::string::npos,
            "nonfinite accepted");
    error.clear();
    require(nicopedia_htp_muon::packForValidatedRpc(
                parameters, gradients, momentum, &packed, &error),
            "structural RPC packing rejected before timed validation");
    require(!nicopedia_htp_muon::validateFinite(packed, &error) &&
                error.find("HTP_MUON_RPC_INPUT_NONFINITE") != std::string::npos,
            "separate pre-RPC finite validation accepted NaN input");
    std::cout << "nicopedia_htp_muon_pack_test=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
