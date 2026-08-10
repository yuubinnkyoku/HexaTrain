// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once
// Shared, host-testable core for Nicopedia byte-level generation.
//
// This header has no QNN / Android / MNN dependencies: the same sources are
// compiled into the device runtime (qnn/qnn_transformer_training.cpp) and into
// the CI-safe host test (host_tests/nicopedia_htp_generation_test.cpp).
//
// Contracts (fixed by the HTP training milestone):
//  - The model predicts the byte after a rolling 32-byte window (T=32,
//    V=256: every byte value is a token, no EOS).  Generation stops only at
//    maxNewBytes.
//  - Context windows are the LAST `tokens` bytes of history; shorter
//    histories are NUL-padded at the front (documented edge-case contract,
//    training windows are always full 32-byte chunks).
//  - Display is lossless: valid UTF-8 passes through, every invalid or
//    truncated byte is emitted as a literal \xNN escape; bytes are never
//    silently dropped or replaced.
//  - Sampling is deterministic within a platform for (seed, step): splitmix64
//    state, fp64 softmax over the temperature-scaled top-K logits, inverse
//    CDF draw.  Cross-platform bit-exact sampling is NOT claimed (libm exp
//    may differ in the last ulp).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace phonelm::nicopedia_gen {

struct GenerateConfig {
  uint32_t maxNewBytes = 64;
  bool greedy = true;
  float temperature = 1.0f;
  // 1..256; values above the vocabulary are clamped to the vocabulary.
  uint32_t topK = 256;
  uint64_t samplingSeed = 0;
  // Parity policy that gates generation ("legacy" | "candidate"; fixed by
  // the parity re-audit protocol docs/qnn-nicopedia-htp-parity-policy.md).
  // "candidate" is the full policy F and is only usable after independent
  // review approval; the runner enforces this with -CandidatePolicyApproved.
  std::string gatePolicy = "legacy";
  // Diagnostic tap configuration ("NONE" | "COARSE" | "FINE").  These fields
  // are private diagnostics only: they never change parity-gate semantics.
  std::string diagnosticTapScope = "NONE";
  uint32_t diagnosticLayerIndex = 0xffffffffu;
  // HTP graph precision control (private diagnostics; numeric-path-only).
  // 0 = unset (backend default, established behavior), 1 = FLOAT16,
  // 2 = FLOAT32.  Mirrors RuntimeOptions::htpGraphPrecisionMode.
  uint32_t htpGraphPrecisionMode = 0;
  // QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION. 0 = unset,
  // 1 = false, 2 = true.
  uint32_t htpGraphPrecisionCompensation = 0;
  // QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING. 0 = unset, 1 = false,
  // 2 = true.
  uint32_t htpGraphWeightsPacking = 0;
  // QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION. 0 = unset,
  // 1 = false, 2 = true.
  uint32_t htpGraphAdvancedActivationFusion = 0;
  // QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED. 0 = unset
  // (established nullptr context config), 1 = false, 2 = true.
  uint32_t htpContextGraphSplitting = 0;
  // Private diagnostic: declare NATIVE float tensors as FP16 (probe whether
  // the backend already executes FP32-declared graphs in FP16).  False by
  // default (established behavior).
  bool htpNativeTensorFp16 = false;
};

// Per-tensor CPU-vs-HTP comparison result for divergence localization.
// Names are the QNN tap tensor names ("layer_000_ln1", ...).  The values are
// private diagnostics and never enter public artifacts.
struct NprtTapMetric {
  std::string name;
  double cpuRms = 0;
  double htpRms = 0;
  double diffMaxAbs = 0;
  double diffRms = 0;
  double relRms = 0;
  double cosine = 1.0;
  double centeredRms = 0;
};

struct Utf8Stats {
  size_t validBytes = 0;
  size_t invalidBytes = 0;
  size_t escapeSpans = 0;  // number of \xNN escapes emitted
};

// Lossless display form (see header comment).  `stats` receives the byte
// statistics of the input (valid/invalid counts, escape spans).
std::string safeUtf8Display(const std::vector<uint8_t>& bytes,
                            Utf8Stats* stats = nullptr);

// Byte statistics without building the display string.
Utf8Stats utf8StatsOf(const std::vector<uint8_t>& bytes);

// Rolling context window per the training contract.  Returns the last
// `tokens` bytes right-aligned; a shorter history is NUL-padded at the front
// and the padding count is written to `padBytes` (nullable).  An empty
// history yields `tokens` NUL bytes (padding == tokens).
std::vector<uint8_t> buildGenerationContext(const std::vector<uint8_t>& history,
                                            uint32_t tokens,
                                            uint32_t* padBytes = nullptr);

// Deterministic argmax over a logits row (first maximum on ties).
uint8_t greedyArgmax(const float* logits, uint32_t vocab);

// Deterministic temperature + top-K sampling (see header comment).
// `topK == 0` means the full vocabulary; a non-positive or non-finite
// temperature falls back to greedy argmax.
uint8_t sampleTopK(const float* logits, uint32_t vocab, float temperature,
                   uint32_t topK, uint64_t seed, uint64_t step);

// Sampling diagnostics used by the HTP-native generation health gate.  The
// legacy sampleTopK() ABI remains unchanged; callers that need a fail-closed
// probability audit use sampleTopKChecked() and inspect every finite/sum flag.
struct TopKSamplingResult {
  uint8_t value = 0;
  bool ok = true;
  bool logitsFinite = true;
  bool weightsFinite = true;
  bool weightSumFinite = true;
  bool weightSumPositive = true;
  bool probabilitiesFinite = true;
  bool probabilitySumFinite = true;
  bool probabilitySumPositive = true;
  double weightSum = 0.0;
  double probabilitySum = 0.0;
};

TopKSamplingResult sampleTopKChecked(const float* logits, uint32_t vocab,
                                     float temperature, uint32_t topK,
                                     uint64_t seed, uint64_t step);

std::string bytesToHex(const std::vector<uint8_t>& bytes);

// Parses hex (both cases accepted); throws std::invalid_argument on
// malformed input (odd length or non-hex digit).
std::vector<uint8_t> hexToBytes(const std::string& hex);

// Slide the rolling window: append one generated byte, drop the oldest.
inline void appendByteWindow(std::vector<uint8_t>& window, uint8_t byte) {
  window.erase(window.begin());
  window.push_back(byte);
}

// Fixed synthetic parity prefixes (deterministic; no licensed corpus text).
// These exact byte sequences are compiled into both the device run and the
// host test so CPU/HTP parity is measured on identical inputs.
struct ParityPrefix {
  const char* label;
  std::vector<uint8_t> bytes;
};
const std::vector<ParityPrefix>& parityPrefixes();

// ---------------------------------------------------------------------------
// Implementation (header-only: compiled into the device and the host test).
// ---------------------------------------------------------------------------

namespace detail {

inline bool isContinuation(uint8_t b) { return (b & 0xc0u) == 0x80u; }

// Returns the expected total sequence length when `b` is a lead byte, or 0
// when the lead byte is invalid.
inline uint32_t leadLength(uint8_t b) {
  if (b < 0x80u) return 1;
  if (b >= 0xc2u && b <= 0xdfu) return 2;
  if (b >= 0xe0u && b <= 0xefu) return 3;
  if (b >= 0xf0u && b <= 0xf4u) return 4;
  return 0;
}

// Returns the number of bytes consumed at `pos` and fills `sequence` (only
// meaningful when the return value is > 0).  0 means an invalid byte.
inline size_t decodeUtf8(const std::vector<uint8_t>& bytes, size_t pos,
                         size_t* length) {
  const uint8_t b = bytes[pos];
  const uint32_t lead = leadLength(b);
  if (lead == 1) {
    if (b < 0x80u) { *length = 1; return 1; }
    return 0;
  }
  if (lead == 0) return 0;
  if (pos + lead > bytes.size()) return 0;  // truncated at end of input
  // Overlong / surrogate / > U+10FFFF exclusion on the second byte.
  const uint8_t b1 = bytes[pos + 1];
  if (!isContinuation(b1)) return 0;
  if (lead == 2) {
    // C0/C1 are rejected by leadLength; C2-DF + continuation is valid.
    *length = 2;
    return 2;
  }
  if (lead == 3) {
    if (pos + 2 >= bytes.size()) return 0;
    const uint8_t b2 = bytes[pos + 2];
    if (!isContinuation(b2)) return 0;
    if (b == 0xe0u && b1 < 0xa0u) return 0;   // overlong E0 80-9F
    if (b == 0xedu && b1 >= 0xa0u) return 0;  // UTF-16 surrogate half
    *length = 3;
    return 3;
  }
  if (lead == 4) {
    if (pos + 3 >= bytes.size()) return 0;
    const uint8_t b2 = bytes[pos + 2];
    const uint8_t b3 = bytes[pos + 3];
    if (!isContinuation(b2) || !isContinuation(b3)) return 0;
    if (b == 0xf0u && b1 < 0x90u) return 0;   // overlong F0 80-8F
    if (b == 0xf4u && b1 >= 0x90u) return 0;  // > U+10FFFF
    *length = 4;
    return 4;
  }
  return 0;
}

inline void appendHexByte(std::string& out, uint8_t b) {
  static const char kHex[] = "0123456789abcdef";
  out.push_back('\\');
  out.push_back('x');
  out.push_back(kHex[(b >> 4) & 0x0fu]);
  out.push_back(kHex[b & 0x0fu]);
}

}  // namespace detail

inline Utf8Stats utf8StatsOf(const std::vector<uint8_t>& bytes) {
  Utf8Stats stats;
  size_t pos = 0;
  while (pos < bytes.size()) {
    size_t length = 0;
    if (detail::decodeUtf8(bytes, pos, &length) > 0) {
      stats.validBytes += length;
      pos += length;
    } else {
      ++stats.invalidBytes;
      ++stats.escapeSpans;
      pos += 1;
    }
  }
  return stats;
}

inline std::string safeUtf8Display(const std::vector<uint8_t>& bytes,
                                   Utf8Stats* stats) {
  Utf8Stats local;
  std::string out;
  out.reserve(bytes.size());
  size_t pos = 0;
  while (pos < bytes.size()) {
    size_t length = 0;
    if (detail::decodeUtf8(bytes, pos, &length) > 0) {
      out.append(reinterpret_cast<const char*>(bytes.data() + pos), length);
      local.validBytes += length;
      pos += length;
    } else {
      detail::appendHexByte(out, bytes[pos]);
      ++local.invalidBytes;
      ++local.escapeSpans;
      pos += 1;
    }
  }
  if (stats) *stats = local;
  return out;
}

inline std::vector<uint8_t> buildGenerationContext(
    const std::vector<uint8_t>& history, uint32_t tokens,
    uint32_t* padBytes) {
  std::vector<uint8_t> context;
  if (history.size() >= tokens) {
    context.assign(history.end() - tokens, history.end());
    if (padBytes) *padBytes = 0;
  } else {
    const size_t pad = tokens - history.size();
    context.assign(pad, 0x00);
    context.insert(context.end(), history.begin(), history.end());
    if (padBytes) *padBytes = static_cast<uint32_t>(pad);
  }
  return context;
}

inline uint8_t greedyArgmax(const float* logits, uint32_t vocab) {
  // uint8_t return contracts vocab <= 256 (the byte vocabulary is fixed at
  // 256); larger vocabularies would silently wrap indices.
  if (vocab > 256) throw std::invalid_argument("vocab_gt_256");
  uint8_t best = 0;
  if (vocab == 0) return best;
  float bestValue = logits[0];
  for (uint32_t i = 1; i < vocab; ++i) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      best = static_cast<uint8_t>(i);
    }
  }
  return best;
}

namespace detail {

// splitmix64 finalizer over a (seed, step) state; deterministic per platform.
inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Top-53-bit uniform sample in [0, 1).
inline double uniform01(uint64_t mixed) {
  return static_cast<double>(mixed >> 11) * 0x1.0p-53;
}

}  // namespace detail

inline TopKSamplingResult sampleTopKChecked(const float* logits, uint32_t vocab,
                                            float temperature, uint32_t topK,
                                            uint64_t seed, uint64_t step) {
  TopKSamplingResult result;
  if (vocab > 256) throw std::invalid_argument("vocab_gt_256");
  if (vocab == 0) return result;
  for (uint32_t i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i])) {
      result.logitsFinite = false;
      result.ok = false;
    }
  }
  if (!result.ok) return result;
  if (!(temperature > 0.0f) || !std::isfinite(temperature))
  {
    result.value = greedyArgmax(logits, vocab);
    return result;
  }
  const uint32_t k = (topK == 0 || topK >= vocab) ? vocab : topK;
  // Unique total order (logit desc, index asc) makes the top-K selection
  // deterministic across any conforming sort implementation.
  std::vector<uint32_t> indices(vocab);
  for (uint32_t i = 0; i < vocab; ++i) indices[i] = i;
  std::sort(indices.begin(), indices.end(),
            [&](uint32_t a, uint32_t b) {
              const float la = logits[a], lb = logits[b];
              if (la != lb) return la > lb;
              return a < b;
            });
  // Temperature-scaled softmax over the top-K entries in double precision.
  double maxLogit = static_cast<double>(logits[indices[0]]);
  std::vector<double> weights(k);
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) {
    const double w = std::exp((static_cast<double>(logits[indices[i]]) -
                               maxLogit) /
                              static_cast<double>(temperature));
    weights[i] = w;
    sum += w;
    if (!std::isfinite(w)) result.weightsFinite = false;
  }
  result.weightSum = sum;
  result.weightSumFinite = std::isfinite(sum);
  result.weightSumPositive = sum > 0.0;
  if (!result.weightsFinite || !result.weightSumFinite ||
      !result.weightSumPositive) {
    result.ok = false;
    return result;
  }
  const double u = detail::uniform01(
      detail::mix64(seed + static_cast<uint64_t>(step) * 0x9E3779B97F4A7C15ull));
  double cumulative = 0.0;
  double probabilitySum = 0.0;
  bool selected = false;
  for (uint32_t i = 0; i < k; ++i) {
    const double probability = weights[i] / sum;
    if (!std::isfinite(probability)) result.probabilitiesFinite = false;
    probabilitySum += probability;
    if (!selected && u <= cumulative + probability) {
      result.value = static_cast<uint8_t>(indices[i]);
      selected = true;
    }
    cumulative += probability;
  }
  if (!std::isfinite(cumulative) || !std::isfinite(probabilitySum))
    result.probabilitiesFinite = false;
  result.probabilitySum = probabilitySum;
  result.probabilitySumFinite = std::isfinite(probabilitySum);
  result.probabilitySumPositive = probabilitySum > 0.0;
  if (!result.probabilitiesFinite || !result.probabilitySumFinite ||
      !result.probabilitySumPositive) {
    result.ok = false;
    return result;
  }
  // Rounding can leave u just above the cumulative sum; preserve the legacy
  // deterministic fallback while still publishing the finite/sum audit.
  if (!selected)
    result.value = static_cast<uint8_t>(indices[k - 1]);
  return result;
}

inline uint8_t sampleTopK(const float* logits, uint32_t vocab,
                          float temperature, uint32_t topK, uint64_t seed,
                          uint64_t step) {
  return sampleTopKChecked(logits, vocab, temperature, topK, seed, step).value;
}

inline std::string bytesToHex(const std::vector<uint8_t>& bytes) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    out.push_back(kHex[(b >> 4) & 0x0fu]);
    out.push_back(kHex[b & 0x0fu]);
  }
  return out;
}

inline std::vector<uint8_t> hexToBytes(const std::string& hex) {
  if (hex.size() % 2 != 0)
    throw std::invalid_argument("hex length must be even");
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    auto nibble = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      throw std::invalid_argument("invalid hex digit");
    };
    out.push_back(
        static_cast<uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
  }
  return out;
}

inline const std::vector<ParityPrefix>& parityPrefixes() {
  // Fixed deterministic prefixes (no licensed corpus text).  Prefixes shorter
  // than 32 bytes exercise the NUL-padding branch of the context window.
  static const std::vector<ParityPrefix> kPrefixes = {
      {"japanese_utf8_21",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x93), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xE3), uint8_t(0x81), uint8_t(0xA1),
        uint8_t(0xE3), uint8_t(0x81), uint8_t(0xAF), uint8_t(0xE3),
        uint8_t(0x80), uint8_t(0x80), uint8_t(0xE4), uint8_t(0xB8),
        uint8_t(0x96)}},  // こんにちは世界
      {"ascii_26",
       {'H', 'e', 'l', 'l', 'o', ',', ' ', 'P', 'h', 'o', 'n', 'e', 'L',
        'M', '!', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}},
      {"byte_edges_32",
       {uint8_t(0x00), uint8_t(0x01), uint8_t(0x7f), uint8_t(0x80),
        uint8_t(0xff), uint8_t(0xfe), uint8_t(0x0a), uint8_t(0x09), 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'}},
      {"truncated_utf8_32",
       {uint8_t(0xE6), uint8_t(0x97), uint8_t(0xA5), uint8_t(0xE6),
        uint8_t(0x9C), uint8_t(0xAC), uint8_t(0xE8), uint8_t(0xAA),
        uint8_t(0x9E), uint8_t(0xE3), 'a', 'b', 'c', '1', '2', '3', '4', '5',
        '6', '7', '8', '9', '0', '1', '2', '3', '4', '5', '6', '7', '8',
        '9'}},
      // --- 16 additional prefixes added by the parity re-audit (2026-08).
      // Deterministic byte vectors (generated once, fixed in code; no
      // licensed corpus text).  Label = byte-count convention used by the
      // parity re-audit protocol docs/qnn-nicopedia-htp-parity-policy.md.
      {"ascii_short_5", {'A', 'b', 'C', 'd', '!'}},
      {"japanese_13",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x93), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xC3), uint8_t(0xA9), uint8_t(0xC3),
        uint8_t(0xA9)}},  // こんにéé (13B: 3+3+3+2+2)
      {"japanese_32_exact",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x93), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xE3), uint8_t(0x81), uint8_t(0xA1),
        uint8_t(0xE3), uint8_t(0x81), uint8_t(0xAF), uint8_t(0xE3),
        uint8_t(0x81), uint8_t(0x9B), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0x8B), uint8_t(0xE3), uint8_t(0x81), uint8_t(0x84),
        '1', '2', '3', '4', '5', '6', '7', '8'}},  // こんにちはせかい + digits
      {"japanese_long_48",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x93), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xE3), uint8_t(0x81), uint8_t(0xA1),
        uint8_t(0xE3), uint8_t(0x81), uint8_t(0xAF), uint8_t(0xE3),
        uint8_t(0x81), uint8_t(0x9B), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0x8B), uint8_t(0xE3), uint8_t(0x81), uint8_t(0x84),
        uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x8C), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAF), uint8_t(0xE3), uint8_t(0x81), uint8_t(0x9F),
        uint8_t(0xE3), uint8_t(0x81), uint8_t(0x99), uint8_t(0xE3),
        uint8_t(0x81), uint8_t(0xA6), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0x8D), uint8_t(0xE3), uint8_t(0x81), uint8_t(0xAA)}},
      // こんにちはせかいこれはたすてきな (48B; context window keeps last 32)
      {"mixed_13_punct",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0x93), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xE3), uint8_t(0x81), uint8_t(0xA1),
        '!'}},  // こんにち! (12B + 1B ASCII)
      {"punctuation_24",
       {uint8_t(0xE3), uint8_t(0x80), uint8_t(0x81), uint8_t(0xE3),
        uint8_t(0x80), uint8_t(0x82), uint8_t(0xE3), uint8_t(0x80),
        uint8_t(0x8D), uint8_t(0xEF), uint8_t(0xBC), uint8_t(0x81),
        uint8_t(0xEF), uint8_t(0xBC), uint8_t(0x9F), uint8_t(0xE2),
        uint8_t(0x80), uint8_t(0xA6), uint8_t(0xEF), uint8_t(0xBC),
        uint8_t(0x88), uint8_t(0xEF), uint8_t(0xBC), uint8_t(0x89)}},
      // 、。」！？…（）
      {"control_whitespace_12",
       {0x09, 0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x20, 0x1D, 0x7F, 0x1E, 0x1F,
        0x0A}},
      {"digits_punct_32",
       {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', ',', ';', ':',
        '!', '?', '#', '@', '$', '%', '&', '*', '(', ')', '-', '_', '=', '+',
        '[', ']', '{', '}'}},
      {"utf8_emoji_16",
       {uint8_t(0xF0), uint8_t(0x9F), uint8_t(0x98), uint8_t(0x80),
        uint8_t(0xF0), uint8_t(0x9F), uint8_t(0x98), uint8_t(0x81),
        uint8_t(0xF0), uint8_t(0x9F), uint8_t(0x98), uint8_t(0x82),
        uint8_t(0xF0), uint8_t(0x9F), uint8_t(0x98), uint8_t(0x83)}},  // 😀😁😂😃
      {"katakana_12",
       {uint8_t(0xE3), uint8_t(0x82), uint8_t(0xA2), uint8_t(0xE3),
        uint8_t(0x82), uint8_t(0xA4), uint8_t(0xE3), uint8_t(0x82),
        uint8_t(0xA6), uint8_t(0xE3), uint8_t(0x82), uint8_t(0xAB)}},  // アイウカ
      {"halfwidth_16",
       {uint8_t(0xEF), uint8_t(0xBD), uint8_t(0xB1), uint8_t(0xEF),
        uint8_t(0xBD), uint8_t(0xB2), uint8_t(0xEF), uint8_t(0xBD),
        uint8_t(0xB3), uint8_t(0xEF), uint8_t(0xBD), uint8_t(0xB4),
        uint8_t(0xEF), uint8_t(0xBD), uint8_t(0xB5), 0x20}},  // ｱｲｳｴｵ + space
      {"pseudo_random_32",
       {uint8_t(0x5b), uint8_t(0x4f), uint8_t(0x19), uint8_t(0x16),
        uint8_t(0x69), uint8_t(0x7c), uint8_t(0x8f), uint8_t(0xce),
        uint8_t(0x83), uint8_t(0x96), uint8_t(0x1b), uint8_t(0x98),
        uint8_t(0x26), uint8_t(0x11), uint8_t(0x68), uint8_t(0xb7),
        uint8_t(0x45), uint8_t(0xbe), uint8_t(0x3a), uint8_t(0x1a),
        uint8_t(0x1b), uint8_t(0x45), uint8_t(0x7e), uint8_t(0x26),
        uint8_t(0xf1), uint8_t(0x85), uint8_t(0x4a), uint8_t(0x29),
        uint8_t(0xab), uint8_t(0x48), uint8_t(0x04), uint8_t(0x90)}},
      // splitmix64(0x3141592653589793) high bytes, first 32 outputs
      {"hiragana_9",
       {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x82), uint8_t(0xE3),
        uint8_t(0x81), uint8_t(0x84), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0x86)}},  // あいう
      {"lowercase_31p",
       {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
        '2', '3', '4'}},  // 31B; context pads 1 NUL at front
      {"invalid_utf8_34",
       {uint8_t(0xFF), uint8_t(0xFE), uint8_t(0x80), uint8_t(0xC0),
        uint8_t(0x80), uint8_t(0xC1), uint8_t(0xBF), uint8_t(0xE3),
        uint8_t(0x81), uint8_t(0x93), uint8_t(0xED), uint8_t(0xA0),
        uint8_t(0x80), uint8_t(0xE3), uint8_t(0x81), 'A', 'B', uint8_t(0xF0),
        uint8_t(0x9F), uint8_t(0x98), uint8_t(0xF4), uint8_t(0x90),
        uint8_t(0x80), uint8_t(0x80), uint8_t(0xE3), uint8_t(0x81),
        uint8_t(0xAB), uint8_t(0xE3), uint8_t(0x81), uint8_t(0x82),
        uint8_t(0xC2), 0x28, uint8_t(0x80), uint8_t(0xF5)}},
      // overlong, surrogate half, truncated seqs, >U+10FFFF, stray cont.
      {"leading_ff_8",
       {uint8_t(0xFF), uint8_t(0xF4), uint8_t(0xF0), uint8_t(0xE0),
        uint8_t(0xC2), uint8_t(0x80), uint8_t(0xFF), uint8_t(0xFF)}},
  };
  return kPrefixes;
}

// ---------------------------------------------------------------------------
// Aggregate quality metrics over generated bytes (host-testable, no QNN/Android
// dependencies).  These mirror the aggregate fields the PowerShell runner
// computes for the private report; only the aggregates are publishable.
// ---------------------------------------------------------------------------

struct GenerationAggregates {
  size_t totalBytes = 0;
  size_t validUtf8Bytes = 0;
  size_t invalidUtf8Bytes = 0;
  size_t validScalars = 0;       // number of well-formed UTF-8 sequences
  size_t asciiBytes = 0;        // 0x00..0x7F bytes
  size_t uniqueByteValues = 0;  // distinct byte values seen
  size_t maxSameByteRun = 0;    // longest run of one identical byte value
  // Longest run of a repeated single character decoded as UTF-8 (a scalar
  // repeated back to back); 0 when the byte stream contains no two identical
  // adjacent scalars.
  size_t maxScalarRepeatRun = 0;
  // Fraction (0..1) of the tail (last 32 bytes) that repeats with a period
  // of 1..4; 0 when no short-period repetition is detected.
  double shortPeriodLoopFraction = 0.0;
};

inline GenerationAggregates generationAggregates(
    const std::vector<uint8_t>& bytes) {
  GenerationAggregates ag;
  ag.totalBytes = bytes.size();
  // Byte-level stats over every byte (independent of UTF-8 decoding).
  bool seen[256] = {};
  size_t sameRun = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    const uint8_t b = bytes[i];
    if (!seen[b]) { seen[b] = true; ++ag.uniqueByteValues; }
    if (b < 0x80u) ++ag.asciiBytes;
    if (i > 0 && b == bytes[i - 1]) {
      ++sameRun;
    } else {
      sameRun = 1;
    }
    if (sameRun > ag.maxSameByteRun) ag.maxSameByteRun = sameRun;
  }
  // UTF-8 decoding pass: valid scalar counts and scalar-level repetitions.
  size_t pos = 0;
  std::vector<uint8_t> prevScalar;
  bool havePrevScalar = false;
  size_t scalarRepetitions = 0;
  while (pos < bytes.size()) {
    size_t length = 0;
    const size_t consumed = detail::decodeUtf8(bytes, pos, &length);
    if (consumed > 0) {
      ag.validUtf8Bytes += consumed;
      ++ag.validScalars;
      const std::vector<uint8_t> scalar(bytes.begin() + pos,
                                        bytes.begin() + pos + consumed);
      if (havePrevScalar && scalar == prevScalar) {
        ++scalarRepetitions;
        const size_t run = static_cast<size_t>(scalarRepetitions) + 1;
        if (run > ag.maxScalarRepeatRun) ag.maxScalarRepeatRun = run;
      } else {
        scalarRepetitions = 0;
      }
      prevScalar = scalar;
      havePrevScalar = true;
      pos += consumed;
    } else {
      ag.invalidUtf8Bytes += 1;
      const std::vector<uint8_t> scalar(1, bytes[pos]);
      if (havePrevScalar && scalar == prevScalar) {
        ++scalarRepetitions;
        const size_t run = static_cast<size_t>(scalarRepetitions) + 1;
        if (run > ag.maxScalarRepeatRun) ag.maxScalarRepeatRun = run;
      } else {
        scalarRepetitions = 0;
      }
      prevScalar = scalar;
      havePrevScalar = true;
      pos += 1;
    }
  }

  // Short-period loop detection on the trailing window (last 32 bytes).
  const size_t tailStart = bytes.size() > 32 ? bytes.size() - 32 : 0;
  const size_t tail = bytes.size() - tailStart;
  double bestPeriodFraction = 0.0;
  for (size_t period = 1; period <= 3; ++period) {
    if (tail - period > 0) {
      size_t matches = 0;
      size_t compared = 0;
      for (size_t i = tailStart; i + period < bytes.size(); ++i) {
        ++compared;
        if (bytes[i] == bytes[i + period]) ++matches;
      }
      const double fraction =
          compared > 0 ? static_cast<double>(matches) / compared : 0.0;
      if (fraction > bestPeriodFraction) bestPeriodFraction = fraction;
    }
  }
  ag.shortPeriodLoopFraction = bestPeriodFraction;
  return ag;
}

}  // namespace phonelm::nicopedia_gen

// ---------------------------------------------------------------------------
// CPU/HTP parity metrics (parity re-audit, protocol:
// docs/qnn-nicopedia-htp-parity-policy.md).  Header-only and dependency-free
// so the device and the host fault-injection test share one implementation.
// Thresholds below are FIXED by the protocol; changing them invalidates the
// audit (they must not be tuned to device results).
// ---------------------------------------------------------------------------

namespace phonelm::nicopedia_gen {

// Fixed thresholds from docs/qnn-nicopedia-htp-parity-policy.md section 4.
constexpr double kParityRawLogitsMaxAbs = 2e-2;   // legacy gate
constexpr double kParityProbMaxAbs = 5e-3;        // legacy + prob dimension
constexpr double kParityProbL1 = 2e-2;            // TV <= 1e-2
constexpr double kParityJsDivergence = 5e-3;
constexpr double kParityCenteredMaxAbs = 2e-2;    // gauge-free shape bound
constexpr double kParityCenteredRms = 5e-3;
constexpr double kParityLogSoftmaxMaxAbs = 1e-1;
constexpr double kParityLogSoftmaxRms = 5e-2;
constexpr double kParityScaleRatioLo = 0.995;
constexpr double kParityScaleRatioHi = 1.005;
constexpr double kParityRawCatastrophic = 5e-1;   // common-offset overflow guard
constexpr double kParityRowDegenerateCpuStd = 1e-6;

struct ParityRowMetrics {
  double deltaMean = 0;
  double deltaMedian = 0;
  double deltaStd = 0;
  double rawMaxAbs = 0;
  double rawRms = 0;
  double centeredMaxAbs = 0;
  double centeredRms = 0;
  double logSoftmaxMaxAbs = 0;
  double logSoftmaxRms = 0;
  double probMaxAbs = 0;
  double probMeanAbs = 0;
  double probL1 = 0;
  double jsDivergence = 0;
  double cosineRaw = 1.0;
  double cosineCentered = 1.0;
  double cpuRms = 0;
  double cpuStd = 0;
  double htpRms = 0;
  double htpStd = 0;
  double scaleRatio = 1.0;   // htpStd / cpuStd (both mean-centered)
  double relMax = 0;         // rawMaxAbs / cpuRms (reporting only)
  uint32_t argmaxCpu = 0;
  uint32_t argmaxHtp = 0;
  double marginCpu = 0;      // top1 - top2 gap on CPU row
  double marginHtp = 0;
  uint32_t topkSetOverlap = 0;
  uint32_t topkSetSize = 5;
  bool topkOrderMatch = false;
  bool finite = true;
  bool rowDegenerate = false;    // cpuStd <= 1e-6 (constant row)
  bool argmaxMatch = true;
  bool decisionAmbiguous = false;  // argmax mismatch and margin <= 2*raw
};

struct ParityPolicies {
  bool legacy = true;   // L: raw_max_abs < 2e-2 && prob_max_abs < 5e-3
  bool prob = true;     // P: prob_max_abs < 5e-3 && prob_l1 < 2e-2 && js < 5e-3
  bool shape = true;    // C: P && centered/logsoftmax/scale/raw-catastrophic
  bool decision = true; // D: P && (argmax match or decision_ambiguous)
  bool full = true;     // F: C && D (candidate ACTIVE policy)
};

// Compute the full metric set over one logits row (both sides already
// extracted; `vocab` = number of elements).  CPU row is the reference.
inline ParityRowMetrics computeParityRowMetrics(const float* cpuLogits,
                                                const float* htpLogits,
                                                uint32_t vocab) {
  ParityRowMetrics m;
  if (vocab == 0) return m;
  std::vector<double> cpu(vocab), htp(vocab), delta(vocab);
  bool cpuFinite = true, htpFinite = true;
  double cpuSum = 0, htpSum = 0, cpuSq = 0, htpSq = 0, dSum = 0, dSq = 0;
  double maxAbs = 0;
  for (uint32_t i = 0; i < vocab; ++i) {
    const double c = static_cast<double>(cpuLogits[i]);
    const double h = static_cast<double>(htpLogits[i]);
    cpu[i] = c; htp[i] = h;
    cpuFinite = cpuFinite && std::isfinite(c);
    htpFinite = htpFinite && std::isfinite(h);
    const double d = h - c;
    delta[i] = d;
    cpuSum += c; htpSum += h;
    cpuSq += c * c; htpSq += h * h;
    dSum += d; dSq += d * d;
    maxAbs = std::max(maxAbs, std::abs(d));
  }
  m.finite = cpuFinite && htpFinite;
  const double n = static_cast<double>(vocab);
  const double cpuMean = cpuSum / n, htpMean = htpSum / n;
  const double deltaMean = dSum / n;
  m.deltaMean = deltaMean;
  m.cpuRms = std::sqrt(cpuSq / n);
  m.htpRms = std::sqrt(htpSq / n);
  m.relMax = m.cpuRms > 0 ? maxAbs / m.cpuRms : 0;
  m.rawMaxAbs = maxAbs;
  m.rawRms = std::sqrt(dSq / n);
  // CPU row spread (population std) and degeneracy check.
  double cpuVar = 0;
  for (uint32_t i = 0; i < vocab; ++i) {
    const double c = cpu[i] - cpuMean;
    cpuVar += c * c;
  }
  m.cpuStd = std::sqrt(cpuVar / n);
  m.rowDegenerate = m.cpuStd <= kParityRowDegenerateCpuStd;
  // delta stats.
  std::vector<double> sorted(delta);
  std::sort(sorted.begin(), sorted.end());
  m.deltaMedian = (vocab % 2 == 1)
                      ? sorted[vocab / 2]
                      : 0.5 * (sorted[vocab / 2 - 1] + sorted[vocab / 2]);
  double dVar = 0;
  double centeredMax = 0, centeredSq = 0;
  for (uint32_t i = 0; i < vocab; ++i) {
    const double c = delta[i] - deltaMean;
    dVar += c * c;
    centeredSq += c * c;
    centeredMax = std::max(centeredMax, std::abs(c));
  }
  m.deltaStd = std::sqrt(dVar / n);
  m.centeredMaxAbs = centeredMax;
  m.centeredRms = std::sqrt(centeredSq / n);
  // htp std.
  double htpVar = 0;
  for (uint32_t i = 0; i < vocab; ++i) {
    const double h = htp[i] - htpMean;
    htpVar += h * h;
  }
  m.htpStd = std::sqrt(htpVar / n);
  m.scaleRatio = (m.cpuStd > 0) ? m.htpStd / m.cpuStd : 0;
  // softmax rows (double precision) for probability metrics.
  const auto softmax = [&](const std::vector<double>& row, double* lseOut) {
    std::vector<double> p(vocab);
    double maxV = row[0];
    for (uint32_t i = 1; i < vocab; ++i) maxV = std::max(maxV, row[i]);
    double sum = 0;
    for (uint32_t i = 0; i < vocab; ++i) {
      p[i] = std::exp(row[i] - maxV);
      sum += p[i];
    }
    if (lseOut) *lseOut = std::log(sum) + maxV;
    for (uint32_t i = 0; i < vocab; ++i) p[i] /= sum;
    return p;
  };
  double lseCpu = 0, lseHtp = 0;
  const std::vector<double> p = softmax(cpu, &lseCpu);
  const std::vector<double> q = softmax(htp, &lseHtp);
  double probMax = 0, probMean = 0, probL1 = 0, js = 0;
  double logSmMax = 0, logSmSq = 0;
  for (uint32_t i = 0; i < vocab; ++i) {
    const double dp = std::abs(p[i] - q[i]);
    probMax = std::max(probMax, dp);
    probMean += dp;
    probL1 += dp;
    const double lsCpu = cpu[i] - lseCpu;
    const double lsHtp = htp[i] - lseHtp;
    const double dl = std::abs(lsHtp - lsCpu);
    logSmMax = std::max(logSmMax, dl);
    logSmSq += dl * dl;
    // JS: 0.5 * (KL(p||mid) + KL(q||mid)), mid = (p+q)/2, 1e-30 clamp.
    const double pc = std::max(p[i], 1e-30);
    const double qc = std::max(q[i], 1e-30);
    if (pc > 0) js += pc * std::log(2.0 * pc / (pc + qc));
    if (qc > 0) js += qc * std::log(2.0 * qc / (pc + qc));
  }
  m.probMaxAbs = probMax;
  m.probMeanAbs = probMean / n;
  m.probL1 = probL1;
  m.jsDivergence = 0.5 * js;
  m.logSoftmaxMaxAbs = logSmMax;
  m.logSoftmaxRms = std::sqrt(logSmSq / n);
  // cosines (raw and centered).
  const auto cosine = [&](const std::vector<double>& a,
                          const std::vector<double>& b, double am, double bm) {
    double dot = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < vocab; ++i) {
      const double x = a[i] - am, y = b[i] - bm;
      dot += x * y; na += x * x; nb += y * y;
    }
    return (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1.0;
  };
  m.cosineRaw = cosine(cpu, htp, 0.0, 0.0);
  m.cosineCentered = cosine(cpu, htp, cpuMean, htpMean);
  // decision info.
  m.argmaxCpu = greedyArgmax(cpuLogits, vocab);
  m.argmaxHtp = greedyArgmax(htpLogits, vocab);
  m.argmaxMatch = m.argmaxCpu == m.argmaxHtp;
  // margins: top1 - top2 gap per side (top1 - top(k+1) unused; k=0 only).
  const auto topGap = [&](const float* row) {
    double first = -std::numeric_limits<double>::infinity();
    double second = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < vocab; ++i) {
      const double v = static_cast<double>(row[i]);
      if (v > first) { second = first; first = v; }
      else if (v > second) { second = v; }
    }
    return first - second;
  };
  m.marginCpu = topGap(cpuLogits);
  m.marginHtp = topGap(htpLogits);
  m.decisionAmbiguous =
      !m.argmaxMatch && m.marginCpu <= 2.0 * m.rawMaxAbs;
  // top-k set overlap and order match (k = min(topkSetSize, vocab)).
  const auto topkIndices = [&](const float* row, uint32_t k) {
    std::vector<uint32_t> idx(vocab);
    for (uint32_t i = 0; i < vocab; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
      const float la = row[a], lb = row[b];
      if (la != lb) return la > lb;
      return a < b;
    });
    idx.resize(std::min<uint32_t>(k, vocab));
    return idx;
  };
  const uint32_t k = std::min<uint32_t>(m.topkSetSize, vocab);
  const std::vector<uint32_t> cpuTop = topkIndices(cpuLogits, k);
  const std::vector<uint32_t> htpTop = topkIndices(htpLogits, k);
  std::vector<uint32_t> inter;
  for (uint32_t i : cpuTop)
    if (std::find(htpTop.begin(), htpTop.end(), i) != htpTop.end())
      inter.push_back(i);
  m.topkSetOverlap = static_cast<uint32_t>(inter.size());
  m.topkOrderMatch = cpuTop == htpTop;
  return m;
}

// Evaluate all policies on one row.  Protocol section 4.
inline ParityPolicies evaluateParityPolicies(const ParityRowMetrics& m) {
  ParityPolicies v;
  const bool probOk = m.finite && !m.rowDegenerate &&
                      m.probMaxAbs < kParityProbMaxAbs &&
                      m.probL1 < kParityProbL1 &&
                      m.jsDivergence < kParityJsDivergence;
  v.prob = probOk;
  v.shape = probOk && m.centeredMaxAbs < kParityCenteredMaxAbs &&
            m.centeredRms < kParityCenteredRms &&
            m.logSoftmaxMaxAbs < kParityLogSoftmaxMaxAbs &&
            m.logSoftmaxRms < kParityLogSoftmaxRms &&
            m.scaleRatio >= kParityScaleRatioLo &&
            m.scaleRatio <= kParityScaleRatioHi &&
            m.rawMaxAbs < kParityRawCatastrophic;
  v.decision = probOk && (m.argmaxMatch || m.decisionAmbiguous);
  v.full = v.shape && v.decision;
  v.legacy = m.finite && !m.rowDegenerate && m.rawMaxAbs < kParityRawLogitsMaxAbs &&
             m.probMaxAbs < kParityProbMaxAbs;
  return v;
}

}  // namespace phonelm::nicopedia_gen
