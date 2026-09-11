// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// Full 114-matrix Original fixture. CPU oracle is immutable; this only packs
// the semantic registry (ParameterRole::MUON) and evaluates the frozen
// keller_original_64560829_fp32 recipe per canonical matrix.
#include "nicopedia_htp_muon.h"
#include "nicopedia_muon_optimizer.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void write(const char* path, const std::vector<float>& data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data.data()),
               std::streamsize(data.size() * sizeof(float)));
  if (!stream) throw std::runtime_error("fixture write failed");
}

static void append(std::vector<float>& out, const std::vector<float>& in,
                   std::size_t offset, std::size_t count) {
  out.insert(out.end(), in.begin() + offset, in.begin() + offset + count);
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("outDir lr");
    const std::string out = argv[1];
    const float lr = std::stof(argv[2]);
    if (!std::isfinite(lr) || lr <= 0) throw std::runtime_error("bad lr");
    phonelm::tiny_lm::Config config{1024, 32, 64, 128, 1e-5f, 19, 2};
    auto parameters = phonelm::tiny_lm::initialParameters(config, 1);
    auto gradients = parameters;
    auto momentum = parameters;
    for (const auto& e : phonelm::tiny_lm::parameterRegistry(momentum))
      std::fill(const_cast<std::vector<float>*>(e.values)->begin(),
                const_cast<std::vector<float>*>(e.values)->end(), 0.0f);
    phonelm::nicopedia_htp_muon::PackedInputs packed;
    std::string error;
    if (!phonelm::nicopedia_htp_muon::pack(parameters, gradients, momentum,
                                           &packed, &error))
      throw std::runtime_error(error);
    if (packed.squareBindings.size() != 76 ||
        packed.rectangularBindings.size() != 38)
      throw std::runtime_error("partition count");
    const auto buildGroup = [&](const std::vector<float>& cur,
                                const std::vector<float>& grad,
                                const std::vector<float>& mom,
                                const std::vector<phonelm::nicopedia_htp_muon::MatrixBinding>& bindings,
                                std::size_t elements, const std::string& tag) {
      std::vector<float> input, hyper, reference;
      input.reserve(bindings.size() * 3 * elements);
      hyper.reserve(bindings.size() * 2);
      reference.reserve(bindings.size() * 8 * elements);
      std::string names;
      for (std::size_t b = 0; b < bindings.size(); ++b) {
        const auto& binding = bindings[b];
        const std::size_t off = b * elements;
        append(input, cur, off, elements);
        append(input, grad, off, elements);
        append(input, mom, off, elements);
        hyper.push_back(lr);
        hyper.push_back(binding.updateScale);
        std::vector<float> w(cur.begin() + off, cur.begin() + off + elements);
        std::vector<float> g(grad.begin() + off, grad.begin() + off + elements);
        std::vector<float> m(mom.begin() + off, mom.begin() + off + elements);
        const float beta = .95f, alpha = 1.0f - beta;
        std::vector<float> next_m(elements), nesterov(elements);
        for (std::size_t i = 0; i < elements; ++i) {
          next_m[i] = beta * m[i] + alpha * g[i];
          nesterov[i] = alpha * g[i] + beta * next_m[i];
        }
        double norm2 = 0;
        for (float v : nesterov) norm2 += double(v) * v;
        const double den = std::sqrt(norm2) +
                           phonelm::nicopedia_muon::kNsEpsilon;
        std::vector<float> x0(elements);
        for (std::size_t i = 0; i < elements; ++i)
          x0[i] = float(double(nesterov[i]) / den);
        const std::uint32_t cols =
            (elements == 4096) ? 64u : 128u;
        std::vector<std::vector<float>> stages;
        stages.push_back(x0);
        for (std::uint32_t step = 1; step <= 5; ++step) {
          std::vector<float> o;
          if (!phonelm::nicopedia_muon::zeropowerNewtonSchulzFp32(
                  nesterov, 64, cols, step, &o, nullptr, &error))
            throw std::runtime_error(error);
          stages.push_back(o);
        }
        std::vector<float> wnext(elements);
        for (std::size_t i = 0; i < elements; ++i)
          wnext[i] = w[i] - lr * binding.updateScale * stages[5][i];
        append(reference, wnext, 0, elements);
        append(reference, next_m, 0, elements);
        for (int s = 0; s < 6; ++s) append(reference, stages[s], 0, elements);
        names += binding.name + (binding.transposed ? " T" : " N") + " " +
                 std::to_string(binding.updateScale) + "\n";
      }
      write((out + "/full-" + tag + "-input.bin").c_str(), input);
      write((out + "/full-" + tag + "-hyper.bin").c_str(), hyper);
      write((out + "/full-" + tag + "-reference.bin").c_str(), reference);
      std::ofstream n(out + "/full-" + tag + "-bindings.txt");
      n << names;
      std::cout << "group=" << tag << " matrices=" << bindings.size()
                << " input=" << input.size()
                << " reference=" << reference.size()
                << " algorithm="
                << phonelm::nicopedia_muon::kAlgorithmIdentity << '\n';
    };
    buildGroup(packed.currentSquare, packed.gradientSquare,
               packed.momentumSquare, packed.squareBindings, 4096, "square");
    buildGroup(packed.currentRectangular, packed.gradientRectangular,
               packed.momentumRectangular, packed.rectangularBindings, 8192,
               "rect");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
