// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Verifies device full-group outputs through the semantic pack/unpack path:
// slices plane0/plane1 per matrix, unpacks into live parameters, and compares
// MUON entries against the immutable CPU oracle update(). Also checks the
// W1/W2 orientation sentinel and semantic aspect scales.
#include "nicopedia_htp_muon.h"
#include "nicopedia_muon_optimizer.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> readBin(const std::string& path,
                                  std::size_t expect) {
  std::ifstream s(path, std::ios::binary);
  if (!s) throw std::runtime_error("open: " + path);
  std::vector<float> v(expect);
  s.read(reinterpret_cast<char*>(v.data()),
         std::streamsize(expect * sizeof(float)));
  if (!s || s.peek() != EOF) throw std::runtime_error("size: " + path);
  return v;
}

static void compare(const std::vector<float>& actual,
                    const std::vector<float>& expected, const char* name,
                    double& worstRel, double& worstAbs, std::string& worstName,
                    int& fails) {
  if (actual.size() != expected.size())
    throw std::runtime_error("count: " + std::string(name));
  double e2 = 0, r2 = 0, a2 = 0, maxAbs = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      throw std::runtime_error("nonfinite: " + std::string(name));
    const double d = double(actual[i]) - expected[i];
    e2 += d * d;
    r2 += double(expected[i]) * expected[i];
    a2 += double(actual[i]) * actual[i];
    maxAbs = std::max(maxAbs, std::abs(d));
  }
  const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
  const double cos =
      (a2 > 0 && r2 > 0)
          ? (double([&] {
              double dot = 0;
              for (std::size_t i = 0; i < actual.size(); ++i)
                dot += double(actual[i]) * expected[i];
              return dot;
            }()) / std::sqrt(a2 * r2))
          : 1.0;
  const bool pass = maxAbs <= 2e-3 && rel <= 1e-3 && cos >= 0.99999;
  if (!pass) {
    ++fails;
    std::cout << "FAIL " << name << " maxAbs=" << maxAbs << " relL2=" << rel
              << " cos=" << cos << "\n";
  }
  if (rel > worstRel) {
    worstRel = rel;
    worstAbs = maxAbs;
    worstName = name;
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("outDir");
    const std::string out = argv[1];
    phonelm::tiny_lm::Config config{1024, 32, 64, 128, 1e-5f, 19, 2};
    auto parameters = phonelm::tiny_lm::initialParameters(config, 1);
    auto gradients = parameters;
    auto momentum = parameters;
    for (const auto& e : phonelm::tiny_lm::parameterRegistry(momentum))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
    auto auxM = momentum, auxV = momentum;
    phonelm::nicopedia_htp_muon::PackedInputs packed;
    std::string error;
    if (!phonelm::nicopedia_htp_muon::pack(parameters, gradients, momentum,
                                           &packed, &error))
      throw std::runtime_error(error);
    if (packed.squareBindings.size() != 76 ||
        packed.rectangularBindings.size() != 38)
      throw std::runtime_error("partition");
    if (packed.rectangularBindings.front().name != "layer_000.ffn_w1" ||
        packed.rectangularBindings[18].name != "layer_018.ffn_w1" ||
        packed.rectangularBindings[19].name != "layer_000.ffn_w2" ||
        packed.rectangularBindings[37].name != "layer_018.ffn_w2")
      throw std::runtime_error("sentinel identity");
    if (packed.rectangularBindings.front().transposed ||
        !packed.rectangularBindings[19].transposed)
      throw std::runtime_error("sentinel orientation");
    for (std::size_t i = 0; i < 19; ++i)
      if (std::abs(packed.scaleRectangular[i] - std::sqrt(2.0f)) > 1e-7f)
        throw std::runtime_error("W1 scale");
    for (std::size_t i = 19; i < 38; ++i)
      if (packed.scaleRectangular[i] != 1.0f)
        throw std::runtime_error("W2 scale");
    std::cout << "sentinel=IDENTITY_ORIENTATION_SCALE_OK\n";

    const auto sqOut = readBin(out + "/full-square-output.bin", 76 * 8 * 4096);
    const auto rcOut = readBin(out + "/full-rect-output.bin", 38 * 8 * 8192);
    std::vector<float> sqW, sqM, rcW, rcM;
    for (int i = 0; i < 76; ++i) {
      sqW.insert(sqW.end(), sqOut.begin() + (i * 8) * 4096,
                 sqOut.begin() + (i * 8 + 1) * 4096);
      sqM.insert(sqM.end(), sqOut.begin() + (i * 8 + 1) * 4096,
                 sqOut.begin() + (i * 8 + 2) * 4096);
    }
    for (int i = 0; i < 38; ++i) {
      rcW.insert(rcW.end(), rcOut.begin() + (i * 8) * 8192,
                 rcOut.begin() + (i * 8 + 1) * 8192);
      rcM.insert(rcM.end(), rcOut.begin() + (i * 8 + 1) * 8192,
                 rcOut.begin() + (i * 8 + 2) * 8192);
    }
    auto devParams = parameters, devMom = momentum;
    if (!phonelm::nicopedia_htp_muon::unpack(packed, sqW, sqM, rcW, rcM,
                                             &devParams, &devMom, &error))
      throw std::runtime_error(error);

    phonelm::nicopedia_muon::Config cpuConfig;
    auto cpu = phonelm::nicopedia_muon::update(parameters, gradients, momentum,
                                               auxM, auxV, cpuConfig);
    if (!cpu.error.empty()) throw std::runtime_error(cpu.error);
    if (cpu.muonMatrixCount != 114 || cpu.muonParameterCount != 622592)
      throw std::runtime_error("cpu counts");
    const auto devP = phonelm::tiny_lm::parameterRegistry(devParams);
    const auto devMo = phonelm::tiny_lm::parameterRegistry(devMom);
    const auto cpuP = phonelm::tiny_lm::parameterRegistry(cpu.parameters);
    const auto cpuMo = phonelm::tiny_lm::parameterRegistry(cpu.muonMomentum);
    double worstRel = 0, worstAbs = 0;
    std::string worstName;
    int fails = 0, muonCount = 0;
    for (std::size_t i = 0; i < devP.size(); ++i) {
      if (devP[i].role != phonelm::tiny_lm::ParameterRole::MUON) continue;
      ++muonCount;
      compare(*devP[i].values, *cpuP[i].values,
              ("param:" + devP[i].name).c_str(), worstRel, worstAbs, worstName,
              fails);
      compare(*devMo[i].values, *cpuMo[i].values,
              ("momentum:" + devP[i].name).c_str(), worstRel, worstAbs,
              worstName, fails);
    }
    if (muonCount != 114) throw std::runtime_error("muon count");
    std::cout << "unpack_vs_cpu_update matrices=" << muonCount
              << " fails=" << fails << " worst=" << worstName
              << " relL2=" << worstRel << " maxAbs=" << worstAbs << "\n";
    if (fails) throw std::runtime_error("unpack parity failed");
    std::cout << "unpack_orientation_parity=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
