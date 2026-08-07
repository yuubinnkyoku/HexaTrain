// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
// CI-safe host tests for the Nicopedia byte-level generation core
// (app/src/main/cpp/nicopedia_generation.h).  The header is shared between
// the device runtime and this test and has no QNN/Android dependencies.
#include "nicopedia_generation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngen = phonelm::nicopedia_gen;

namespace {

void require(bool condition, const char *what) {
  if (!condition) throw std::runtime_error(std::string("ASSERT_FAILED: ") + what);
}

std::vector<uint8_t> prefixes_bytes() {
  const auto &prefixes = ngen::parityPrefixes();
  return prefixes.front().bytes;
}

// Deterministic synthetic forward provider: logits are a weighted sum over
// context positions so that (context, position) uniquely shapes the argmax.
// The last-row logit for byte b is sum_i (i + 1) * context[i] plus a per-byte
// constant so greedy picks a byte that depends on the whole window.
std::vector<float> syntheticLogits(const std::vector<uint8_t> &context,
                                   uint32_t vocab) {
  std::vector<float> logits(vocab, 0.0f);
  float positionSum = 0.0f;
  for (size_t i = 0; i < context.size(); ++i)
    positionSum += float(i + 1) * float(context[i]);
  for (uint32_t b = 0; b < vocab; ++b)
    logits[b] = positionSum + float(b) * 1e-3f;
  return logits;
}

void testContextWindow() {
  {
    uint32_t pad = 999;
    const std::vector<uint8_t> empty;
    const auto context = ngen::buildGenerationContext(empty, 32, &pad);
    require(context.size() == 32, "empty history pads to 32");
    require(pad == 32, "empty history pad count");
    for (uint8_t byte : context) require(byte == 0x00, "pad byte is NUL");
  }
  {
    uint32_t pad = 999;
    const std::vector<uint8_t> shortHistory = {'a', 'b', 'c'};
    const auto context = ngen::buildGenerationContext(shortHistory, 32, &pad);
    require(context.size() == 32, "short history pads to 32");
    require(pad == 29, "short history pad count");
    require(context[0] == 0x00 && context[28] == 0x00, "pad region is NUL");
    require(context[29] == 'a' && context[30] == 'b' && context[31] == 'c',
            "history right-aligned");
  }
  {
    uint32_t pad = 999;
    std::vector<uint8_t> longHistory(64);
    for (size_t i = 0; i < longHistory.size(); ++i) longHistory[i] = uint8_t(i);
    const auto context = ngen::buildGenerationContext(longHistory, 32, &pad);
    require(context.size() == 32, "long history takes last 32");
    require(pad == 0, "long history no padding");
    for (size_t i = 0; i < 32; ++i)
      require(context[i] == longHistory[32 + i], "window is the tail");
  }
  {
    uint32_t pad = 999;
    std::vector<uint8_t> exact(32, 7);
    const auto context = ngen::buildGenerationContext(exact, 32, &pad);
    require(context.size() == 32 && pad == 0, "exact window unchanged");
    require(context == exact, "exact window content");
  }
}

void testGreedy() {
  const float logits[6] = {1.0f, 3.0f, 3.0f, 2.0f, 0.5f, -1.0f};
  require(ngen::greedyArgmax(logits, 6) == 1, "greedy takes first max on ties");
  const float all[4] = {-2.0f, -2.0f, -2.0f, -2.0f};
  require(ngen::greedyArgmax(all, 4) == 0, "greedy on all-equal picks index 0");
  const float high[3] = {0.0f, 0.0f, 5.0f};
  require(ngen::greedyArgmax(high, 3) == 2, "greedy picks unique max");
}

void testSampling() {
  // topK=1 must equal greedy regardless of temperature/seed.
  const std::vector<float> logits = [&] {
    std::vector<float> values(256);
    for (uint32_t b = 0; b < 256; ++b) values[b] = float(int(b % 7) - 3);
    return values;
  }();
  const uint8_t greedy = ngen::greedyArgmax(logits.data(), 256);
  for (uint64_t seed : {0ull, 1ull, 42ull, 0xffffffffffffffffull})
    require(ngen::sampleTopK(logits.data(), 256, 0.7f, 1, seed, 3) == greedy,
            "topK=1 sampling equals greedy");

  // Determinism: same (seed, step) -> same byte; different step -> spread.
  const uint8_t a1 = ngen::sampleTopK(logits.data(), 256, 1.0f, 8, 1234, 0);
  const uint8_t a2 = ngen::sampleTopK(logits.data(), 256, 1.0f, 8, 1234, 0);
  require(a1 == a2, "sampling deterministic for (seed, step)");
  bool spread = false;
  for (uint32_t step = 1; step < 64; ++step) {
    if (ngen::sampleTopK(logits.data(), 256, 1.0f, 8, 1234, step) != a1) {
      spread = true;
      break;
    }
  }
  require(spread, "sampling varies with step on crafted logits");

  // Sample only within the top-K set (logit-desc, index-asc order).
  // The crafted logits have ties at every value; the top-4 set is the unique
  // index-ascending first four of value 3, 3, 2, 2 patterns.
  std::vector<uint8_t> seen;
  for (uint32_t step = 0; step < 256; ++step) {
    const uint8_t byte =
        ngen::sampleTopK(logits.data(), 256, 1.0f, 4, 7, step);
    if (std::find(seen.begin(), seen.end(), byte) == seen.end())
      seen.push_back(byte);
  }
  require(!seen.empty() && seen.size() <= 4, "sampling confined to top-K");
  for (uint8_t byte : seen) {
    const float value = logits[byte];
    require(value >= logits[3], "sampled byte is within top-K values");
  }

  // Temperature scaling: a very small temperature must collapse to the
  // unique argmax when topK is full.
  std::vector<float> distinct(256);
  for (uint32_t b = 0; b < 256; ++b) distinct[b] = float(b);
  const uint8_t distinctGreedy = ngen::greedyArgmax(distinct.data(), 256);
  require(distinctGreedy == 255, "distinct logits argmax is the last byte");
  const uint8_t cold = ngen::sampleTopK(distinct.data(), 256, 1e-4f, 256, 5, 0);
  require(cold == distinctGreedy, "near-zero temperature collapses to argmax");
  // With ties at the max, collapse keeps the sampler inside the tie group
  // (value equal to the argmax value), never below it.
  const uint8_t coldTied =
      ngen::sampleTopK(logits.data(), 256, 1e-4f, 256, 5, 0);
  require(logits[coldTied] == logits[6],
          "tie collapse stays inside the argmax value band");
}

void testDisplay() {
  {
    const std::vector<uint8_t> text = {'h', 'i'};
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 2 && stats.invalidBytes == 0,
            "ASCII is valid UTF-8");
    require(ngen::safeUtf8Display(text) == "hi", "ASCII passes through");
  }
  {
    const std::vector<uint8_t> text = {uint8_t(0xE3), uint8_t(0x81), uint8_t(0x93)};  // こ
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 3 && stats.invalidBytes == 0,
            "3-byte Japanese sequence is valid");
    require(ngen::safeUtf8Display(text) == "\xE3\x81\x93",
            "valid multi-byte passes through");
  }
  {
    const std::vector<uint8_t> text = {uint8_t(0xE3), uint8_t(0x81)};  // truncated lead
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 0 && stats.invalidBytes == 2,
            "truncated lead counts as invalid bytes");
    require(ngen::safeUtf8Display(text) == "\\xe3\\x81",
            "truncated bytes escaped, never dropped");
  }
  {
    const std::vector<uint8_t> text = {'a', uint8_t(0xFF), uint8_t(0x80), 'b'};
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 2 && stats.invalidBytes == 2,
            "invalid continuation bytes counted");
    require(ngen::safeUtf8Display(text) == "a\\xff\\x80b",
            "invalid bytes escaped in place");
  }
  {
    const std::vector<uint8_t> text = {uint8_t(0xC2), uint8_t(0xA2)};  // ¢
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 2 && stats.invalidBytes == 0,
            "2-byte sequence is valid");
    require(ngen::safeUtf8Display(text) == "\xC2\xA2", "2-byte passes through");
  }
  {
    const std::vector<uint8_t> text;
    require(ngen::safeUtf8Display(text).empty(), "empty input displays empty");
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 0 && stats.invalidBytes == 0,
            "empty input stats zero");
  }
  {
    const std::vector<uint8_t> text = {uint8_t(0xED), uint8_t(0xA0), uint8_t(0x80)};  // surrogate
    const auto stats = ngen::utf8StatsOf(text);
    require(stats.validBytes == 0 && stats.invalidBytes == 3,
            "surrogate range rejected");
    require(ngen::safeUtf8Display(text) == "\\xed\\xa0\\x80",
            "surrogate escaped");
  }
}

void testHex() {
  const std::vector<uint8_t> bytes = {0x00, 0x01, 0x7f, 0x80, 0xff};
  require(ngen::bytesToHex(bytes) == "00017f80ff", "hex lower-case, no prefix");
  require(ngen::bytesToHex({}) == "", "empty hex");
  require(ngen::bytesToHex({0xAB, 0xCD}) == "abcd", "hex formatting");
}

void testWindowAppend() {
  std::vector<uint8_t> window(32, 0);
  for (uint32_t i = 0; i < 40; ++i) ngen::appendByteWindow(window, uint8_t(i));
  require(window.size() == 32, "window stays 32");
  require(window[0] == 8 && window[31] == 39, "window slides and drops first");
}

void testParityPrefixes() {
  const auto &prefixes = ngen::parityPrefixes();
  require(prefixes.size() == 4, "four fixed parity prefixes");
  std::vector<std::string> labels;
  for (const auto &prefix : prefixes) {
    require(prefix.bytes.size() <= 32, "prefix fits the 32-byte window");
    require(prefix.label && *prefix.label, "prefix label present");
    labels.emplace_back(prefix.label);
  }
  require(labels.size() == 4, "four labels");
  for (size_t i = 0; i < labels.size(); ++i)
    for (size_t j = i + 1; j < labels.size(); ++j)
      require(labels[i] != labels[j], "labels unique");
  bool sawNull = false, sawHigh = false, sawAscii = false;
  for (const auto &prefix : prefixes)
    for (uint8_t byte : prefix.bytes) {
      if (byte == 0x00) sawNull = true;
      if (byte == 0xFF) sawHigh = true;
      if (byte == 'a' || byte == 'A') sawAscii = true;
    }
  require(sawNull, "byte-edge prefix covers 0x00");
  require(sawHigh, "byte-edge prefix covers 0xFF");
  require(sawAscii, "ASCII prefix present");
  bool truncated = false;
  for (uint8_t byte : prefixes.back().bytes)
    if (byte == 0xE3) truncated = true;
  require(truncated, "truncated UTF-8 prefix exercises the lead byte");
}

void testAutoregressiveLoopWithSyntheticProvider() {
  // Simulate the device AR loop: greedy bytes from the synthetic provider,
  // context slides, and the argmax is stable for identical context.
  std::vector<uint8_t> context =
      ngen::buildGenerationContext(prefixes_bytes(), 32, nullptr);
  std::vector<uint8_t> generated;
  for (uint32_t step = 0; step < 8; ++step) {
    const auto logits = syntheticLogits(context, 256);
    const uint8_t byte = ngen::greedyArgmax(logits.data(), 256);
    generated.push_back(byte);
    ngen::appendByteWindow(context, byte);
  }
  require(generated.size() == 8, "eight AR steps");
  // The synthetic provider is a pure function of the window: replaying the
  // same loop must reproduce the same bytes exactly.
  std::vector<uint8_t> replayContext =
      ngen::buildGenerationContext(prefixes_bytes(), 32, nullptr);
  for (uint32_t step = 0; step < 8; ++step) {
    const auto logits = syntheticLogits(replayContext, 256);
    const uint8_t byte = ngen::greedyArgmax(logits.data(), 256);
    require(byte == generated[step], "AR replay deterministic");
    ngen::appendByteWindow(replayContext, byte);
  }
}

}  // namespace

int main() {
  try {
    testContextWindow();
    testGreedy();
    testSampling();
    testDisplay();
    testHex();
    testWindowAppend();
    testParityPrefixes();
    testAutoregressiveLoopWithSyntheticProvider();
    std::printf("nicopedia_htp_generation_test=PASS\n");
    return 0;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "nicopedia_htp_generation_test=FAIL %s\n",
                 exception.what());
    return 1;
  }
}
