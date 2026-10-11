// SPDX-License-Identifier: Apache-2.0
// Read-only private checkpoint/state and fixed training-input reference probe.
// Usage: g1_forensic_checkpoint_inspect ckpt cache tokenizer update microbatch
#include "nicopedia_muon_checkpoint.h"
#include "nicopedia_byte_bpe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
std::string rawHash(const std::vector<float>& values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * 4);
  for (float value : values) {
    std::uint32_t bits = 0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8)
      bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
  }
  return phonelm::nicopedia_bpe::detail::sha256(bytes);
}
void summary(const std::string& name, const std::vector<float>& values) {
  double norm2 = 0, maxAbs = 0;
  std::size_t bad = 0;
  for (float value : values) {
    if (!std::isfinite(value)) { ++bad; continue; }
    norm2 += double(value) * value;
    maxAbs = std::max(maxAbs, std::abs(double(value)));
  }
  std::cout << name << ',' << values.size() << ',' << bad << ',' << maxAbs
            << ',' << std::sqrt(norm2) << ','
            << rawHash(values) << '\n';
}
std::uint64_t splitMix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}
std::string datasetHash(const phonelm::nicopedia_bpe::Cache& cache) {
  std::uint64_t hash = 14695981039346656037ull;
  const auto byte = [&](std::uint8_t value) { hash = (hash ^ value) * 1099511628211ull; };
  const auto integer = [&](std::uint64_t value, int count) {
    for (int i = 0; i < count; ++i) byte(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  integer(cache.context, 4); integer(cache.vocabulary, 4);
  for (unsigned char c : cache.tokenizerHash) byte(c);
  integer(cache.recordCount(), 8);
  for (std::size_t r = 0; r < cache.recordCount(); ++r) {
    integer(cache.articleHashAt(r), 8);
    for (std::size_t t = 0; t <= cache.context; ++t) integer(cache.tokenAt(r, t), 2);
  }
  std::ostringstream text;
  text << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return text.str();
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("ARGS: checkpoint cache tokenizer update microbatch");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("CHECKPOINT_OPEN");
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    phonelm::nicopedia_muon_checkpoint::Checkpoint checkpoint;
    std::string error;
    if (!phonelm::nicopedia_muon_checkpoint::decodeCheckpoint(bytes, &checkpoint, &error))
      throw std::runtime_error(error);
    const auto& id = checkpoint.identity;
    const auto update = std::stoull(argv[4]);
    const auto batch = std::stoull(argv[5]);
    if (update != id.globalStep + 1 || batch >= 8 ||
        id.dataCursor.recordIndex != id.globalStep * 8 ||
        id.dataCursor.exposedTokens != id.globalStep * 8 * id.config.tokens ||
        id.dataCursor.orderSeed != 20260806 || id.dataCursor.tokenOffset || id.dataCursor.epoch)
      throw std::runtime_error("CURSOR_IDENTITY");
    phonelm::qnn::TinyTransformerParameters parameters;
    if (!phonelm::nicopedia_muon_checkpoint::extractParameters(checkpoint, id.config, id.seed, &parameters, &error))
      throw std::runtime_error(error);
    const auto tokenizer = phonelm::nicopedia_bpe::loadModel(argv[3]);
    const auto cache = phonelm::nicopedia_bpe::loadCache(argv[2], tokenizer, 15000000);
    if (cache.tokenizerHash != id.tokenizerHash || cache.context != id.config.tokens ||
        cache.vocabulary != id.config.vocabularySize || datasetHash(cache) != id.dataCursor.datasetHash)
      throw std::runtime_error("CACHE_IDENTITY");
    const std::uint64_t selection = (update - 1) * 8 + batch;
    std::uint64_t state = id.dataCursor.orderSeed;
    for (std::uint64_t i = 0; i <= selection; ++i) state = splitMix(state + i);
    const std::size_t record = state % cache.recordCount();
    std::vector<std::uint32_t> tokens, labels;
    for (std::size_t t = 0; t < cache.context; ++t) {
      tokens.push_back(cache.tokenAt(record, t));
      labels.push_back(cache.tokenAt(record, t + 1));
    }
    const auto oneHot = phonelm::tiny_lm::oneHot(tokens, cache.vocabulary);
    std::cout << std::setprecision(17)
              << "checkpoint_step=" << id.globalStep << "\nnext_update=" << update
              << "\nmicrobatch=" << batch << "\nselection_index=" << selection
              << "\nrecord_index=" << record << "\ndataset_identity=" << id.dataCursor.datasetHash
              << "\ntensor,count,nonfinite,max_abs,l2,raw_f32_le_sha256\n";
    std::vector<float> p, mu, m, v;
    for (const auto& entry : checkpoint.parameters) {
      summary("parameter/" + entry.name, entry.values);
      summary("momentum/" + entry.name, entry.momentum);
      summary("adam_m/" + entry.name, entry.adamM);
      summary("adam_v/" + entry.name, entry.adamV);
      p.insert(p.end(), entry.values.begin(), entry.values.end());
      mu.insert(mu.end(), entry.momentum.begin(), entry.momentum.end());
      m.insert(m.end(), entry.adamM.begin(), entry.adamM.end());
      v.insert(v.end(), entry.adamV.begin(), entry.adamV.end());
    }
    summary("all_parameters", p); summary("all_momentum", mu);
    summary("all_adam_m", m); summary("all_adam_v", v);
    summary("input", oneHot);
    summary("target", phonelm::tiny_lm::oneHot(labels, cache.vocabulary));
    const auto trace = phonelm::tiny_lm::forwardTraceGeneralized(id.config, oneHot, parameters);
    summary("cpu/embedding", trace.embeddedInput);
    for (std::size_t i = 0; i < trace.layers.size(); ++i) {
      const auto& layer = trace.layers[i];
      const std::string prefix = "cpu/layer_" + std::to_string(i) + "/";
      summary(prefix + "input", layer.input);
      summary(prefix + "ln1", layer.ln1);
      summary(prefix + "q", layer.q); summary(prefix + "k", layer.k);
      summary(prefix + "v", layer.v); summary(prefix + "gates", layer.gates);
      summary(prefix + "probabilities", layer.probabilities);
      summary(prefix + "context", layer.context);
      summary(prefix + "residual1", layer.residual1);
      summary(prefix + "ln2", layer.ln2);
      summary(prefix + "ff1", layer.ff1);
      summary(prefix + "output", layer.output);
    }
    summary("cpu/logits", trace.logits);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FORENSIC_INSPECT_FAILED: " << exception.what() << '\n';
    return 1;
  }
}
