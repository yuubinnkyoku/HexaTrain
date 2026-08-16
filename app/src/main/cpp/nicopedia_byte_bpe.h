// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phonelm::nicopedia_bpe {

inline constexpr std::uint32_t kVocabulary = 1024;
inline constexpr std::uint32_t kMergeCount = 768;
inline constexpr char kModelMagic[] = "NPRTBPEM1\n";
inline constexpr char kCacheMagic[] = "NPRTBPEV1\n";

namespace detail {

inline std::uint32_t rotateRight(std::uint32_t value, std::uint32_t bits) {
  return (value >> bits) | (value << (32u - bits));
}

inline std::string sha256(const std::vector<std::uint8_t>& input) {
  static constexpr std::array<std::uint32_t, 64> k = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
  std::vector<std::uint8_t> bytes = input;
  const std::uint64_t bitLength = static_cast<std::uint64_t>(bytes.size()) * 8u;
  bytes.push_back(0x80u);
  while ((bytes.size() % 64u) != 56u) bytes.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(bitLength >> shift));
  std::array<std::uint32_t, 8> h = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t p = offset + i * 4;
      w[i] = (std::uint32_t(bytes[p]) << 24) | (std::uint32_t(bytes[p + 1]) << 16) |
             (std::uint32_t(bytes[p + 2]) << 8) | bytes[p + 3];
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotateRight(w[i-15],7) ^ rotateRight(w[i-15],18) ^ (w[i-15] >> 3);
      const std::uint32_t s1 = rotateRight(w[i-2],17) ^ rotateRight(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    std::uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1=rotateRight(e,6)^rotateRight(e,11)^rotateRight(e,25);
      const std::uint32_t ch=(e&f)^((~e)&g);
      const std::uint32_t t1=hh+s1+ch+k[i]+w[i];
      const std::uint32_t s0=rotateRight(a,2)^rotateRight(a,13)^rotateRight(a,22);
      const std::uint32_t maj=(a&b)^(a&c)^(b&c);
      const std::uint32_t t2=s0+maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint32_t value : h) output << std::setw(8) << value;
  return output.str();
}

inline std::uint32_t readU32(const std::vector<std::uint8_t>& bytes,
                             std::size_t* offset) {
  if (*offset + 4 > bytes.size()) throw std::runtime_error("BPE_MODEL_TRUNCATED");
  const std::uint32_t value = (std::uint32_t(bytes[*offset]) << 24) |
      (std::uint32_t(bytes[*offset + 1]) << 16) |
      (std::uint32_t(bytes[*offset + 2]) << 8) | bytes[*offset + 3];
  *offset += 4;
  return value;
}

inline std::uint16_t readU16(const std::vector<std::uint8_t>& bytes,
                             std::size_t* offset) {
  if (*offset + 2 > bytes.size()) throw std::runtime_error("BPE_MODEL_TRUNCATED");
  const std::uint16_t value = static_cast<std::uint16_t>(
      (std::uint16_t(bytes[*offset]) << 8) | bytes[*offset + 1]);
  *offset += 2;
  return value;
}

inline std::uint64_t readU64(const std::vector<std::uint8_t>& bytes,
                             std::size_t* offset) {
  if (*offset + 8 > bytes.size()) throw std::runtime_error("BPE_CACHE_TRUNCATED");
  std::uint64_t value = 0;
  for (int index = 0; index < 8; ++index) value = (value << 8) | bytes[(*offset)++];
  return value;
}

inline std::string hexOf(const std::uint8_t* bytes, std::size_t count) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(count * 2, '0');
  for (std::size_t index = 0; index < count; ++index) {
    result[index * 2] = hex[bytes[index] >> 4];
    result[index * 2 + 1] = hex[bytes[index] & 15u];
  }
  return result;
}

inline std::uint32_t pairKey(std::uint16_t left, std::uint16_t right) {
  return (std::uint32_t(left) << 16) | right;
}

}  // namespace detail

struct Model {
  std::array<std::pair<std::uint16_t, std::uint16_t>, kMergeCount> merges{};
  std::array<std::vector<std::uint8_t>, kVocabulary> expansions{};
  std::string sha256;

  std::string identity() const { return "sha256:" + sha256; }

  std::vector<std::uint16_t> encode(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.empty()) return {};
    std::vector<std::uint16_t> token(bytes.begin(), bytes.end());
    std::vector<std::int32_t> previous(token.size()), next(token.size());
    for (std::size_t i = 0; i < token.size(); ++i) {
      previous[i] = i == 0 ? -1 : static_cast<std::int32_t>(i - 1);
      next[i] = i + 1 == token.size() ? -1 : static_cast<std::int32_t>(i + 1);
    }
    std::unordered_map<std::uint32_t, std::uint16_t> rank;
    rank.reserve(kMergeCount * 2);
    for (std::uint16_t i = 0; i < kMergeCount; ++i)
      rank.emplace(detail::pairKey(merges[i].first, merges[i].second), i);
    using Candidate = std::pair<std::uint16_t, std::uint32_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    const auto enqueue = [&](std::int32_t position) {
      if (position < 0 || next[static_cast<std::size_t>(position)] < 0) return;
      const auto found = rank.find(detail::pairKey(token[static_cast<std::size_t>(position)],
          token[static_cast<std::size_t>(next[static_cast<std::size_t>(position)])]));
      if (found != rank.end()) queue.emplace(found->second, static_cast<std::uint32_t>(position));
    };
    for (std::uint32_t i = 0; i + 1 < token.size(); ++i) enqueue(static_cast<std::int32_t>(i));
    while (!queue.empty()) {
      const auto [mergeRank, leftPosition] = queue.top(); queue.pop();
      const std::int32_t rightPosition = next[leftPosition];
      if (rightPosition < 0 || token[leftPosition] != merges[mergeRank].first ||
          token[static_cast<std::size_t>(rightPosition)] != merges[mergeRank].second) continue;
      token[leftPosition] = static_cast<std::uint16_t>(256u + mergeRank);
      const std::int32_t after = next[static_cast<std::size_t>(rightPosition)];
      next[leftPosition] = after;
      if (after >= 0) previous[static_cast<std::size_t>(after)] = static_cast<std::int32_t>(leftPosition);
      next[static_cast<std::size_t>(rightPosition)] = -2;
      enqueue(previous[leftPosition]); enqueue(static_cast<std::int32_t>(leftPosition));
    }
    std::vector<std::uint16_t> result;
    for (std::int32_t position = 0; position >= 0; position = next[static_cast<std::size_t>(position)])
      result.push_back(token[static_cast<std::size_t>(position)]);
    return result;
  }

  std::vector<std::uint8_t> decode(const std::vector<std::uint16_t>& tokens) const {
    std::vector<std::uint8_t> output;
    for (std::uint16_t token : tokens) {
      if (token >= kVocabulary) throw std::runtime_error("BPE_TOKEN_ID_RANGE");
      output.insert(output.end(), expansions[token].begin(), expansions[token].end());
    }
    return output;
  }

  std::size_t tokenByteLength(std::uint16_t token) const {
    if (token >= kVocabulary) throw std::runtime_error("BPE_TOKEN_ID_RANGE");
    return expansions[token].size();
  }
};

inline Model parseModel(const std::vector<std::uint8_t>& bytes) {
  constexpr std::size_t magicSize = sizeof(kModelMagic) - 1;
  constexpr std::size_t expectedSize = magicSize + 8 + kMergeCount * 4;
  if (bytes.size() != expectedSize) throw std::runtime_error("BPE_MODEL_SIZE");
  if (!std::equal(bytes.begin(), bytes.begin() + magicSize, kModelMagic))
    throw std::runtime_error("BPE_MODEL_MAGIC");
  std::size_t offset = magicSize;
  if (detail::readU32(bytes, &offset) != kVocabulary ||
      detail::readU32(bytes, &offset) != kMergeCount)
    throw std::runtime_error("BPE_MODEL_HEADER");
  Model model;
  for (std::uint16_t value = 0; value < 256; ++value)
    model.expansions[value] = {static_cast<std::uint8_t>(value)};
  for (std::uint16_t index = 0; index < kMergeCount; ++index) {
    const std::uint16_t left = detail::readU16(bytes, &offset);
    const std::uint16_t right = detail::readU16(bytes, &offset);
    const std::uint16_t token = static_cast<std::uint16_t>(256u + index);
    if (left >= token || right >= token) throw std::runtime_error("BPE_MODEL_FORWARD_REFERENCE");
    model.merges[index] = {left, right};
    model.expansions[token] = model.expansions[left];
    model.expansions[token].insert(model.expansions[token].end(),
                                   model.expansions[right].begin(), model.expansions[right].end());
  }
  model.sha256 = detail::sha256(bytes);
  return model;
}

inline Model loadModel(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("BPE_MODEL_OPEN");
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) throw std::runtime_error("BPE_MODEL_READ");
  return parseModel(bytes);
}

struct CacheRecord {
  std::uint64_t articleHash = 0;
  std::vector<std::uint16_t> window;
};

struct Cache {
  std::uint32_t context = 0;
  std::uint32_t vocabulary = 0;
  std::string tokenizerHash;
  std::vector<CacheRecord> records;
};

inline Cache parseCache(const std::vector<std::uint8_t>& bytes,
                        const Model& model) {
  constexpr std::size_t magicSize = sizeof(kCacheMagic) - 1;
  constexpr std::size_t fixedHeader = magicSize + 8 + 32 + 8;
  if (bytes.size() < fixedHeader ||
      !std::equal(bytes.begin(), bytes.begin() + magicSize, kCacheMagic))
    throw std::runtime_error("BPE_CACHE_MAGIC");
  std::size_t offset = magicSize;
  Cache cache;
  cache.context = detail::readU32(bytes, &offset);
  cache.vocabulary = detail::readU32(bytes, &offset);
  if (cache.context == 0 || cache.context > 256 || cache.vocabulary != kVocabulary)
    throw std::runtime_error("BPE_CACHE_HEADER");
  cache.tokenizerHash = "sha256:" + detail::hexOf(bytes.data() + offset, 32);
  offset += 32;
  if (cache.tokenizerHash != model.identity())
    throw std::runtime_error("BPE_CACHE_TOKENIZER_HASH");
  const std::uint64_t count = detail::readU64(bytes, &offset);
  if (count > 10000000u) throw std::runtime_error("BPE_CACHE_COUNT");
  const std::uint64_t recordSize = 8u + 2u * (std::uint64_t(cache.context) + 1u);
  if (count > (std::numeric_limits<std::size_t>::max() - offset) / recordSize ||
      bytes.size() != offset + static_cast<std::size_t>(count * recordSize))
    throw std::runtime_error("BPE_CACHE_SIZE");
  cache.records.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t recordIndex = 0; recordIndex < count; ++recordIndex) {
    CacheRecord record;
    record.articleHash = detail::readU64(bytes, &offset);
    record.window.reserve(cache.context + 1);
    for (std::uint32_t tokenIndex = 0; tokenIndex <= cache.context; ++tokenIndex) {
      const std::uint16_t token = detail::readU16(bytes, &offset);
      if (token >= cache.vocabulary) throw std::runtime_error("BPE_CACHE_TOKEN_RANGE");
      record.window.push_back(token);
    }
    cache.records.push_back(std::move(record));
  }
  return cache;
}

inline Cache loadCache(const std::string& path, const Model& model) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("BPE_CACHE_OPEN");
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) throw std::runtime_error("BPE_CACHE_READ");
  return parseCache(bytes, model);
}

inline std::vector<std::uint16_t> buildTokenContext(
    const std::vector<std::uint16_t>& history, std::uint32_t context,
    std::uint32_t* padTokens = nullptr) {
  if (context == 0) throw std::runtime_error("BPE_CONTEXT_ZERO");
  if (history.size() >= context) {
    if (padTokens) *padTokens = 0;
    return {history.end() - context, history.end()};
  }
  const std::size_t pad = context - history.size();
  std::vector<std::uint16_t> result(pad, 0);
  result.insert(result.end(), history.begin(), history.end());
  if (padTokens) *padTokens = static_cast<std::uint32_t>(pad);
  return result;
}

inline bool appendTokenWithinByteLimit(
    const Model* model, std::uint32_t token, std::size_t maximumBytes,
    std::vector<std::uint8_t>* rawBytes,
    std::vector<std::uint16_t>* generatedTokens) {
  if (!rawBytes || !generatedTokens) throw std::runtime_error("BPE_OUTPUT_REQUIRED");
  if (token >= (model ? kVocabulary : 256u)) throw std::runtime_error("BPE_TOKEN_ID_RANGE");
  const std::vector<std::uint8_t> literal = {static_cast<std::uint8_t>(token)};
  const auto& decoded = model ? model->expansions[token] : literal;
  if (rawBytes->size() + decoded.size() > maximumBytes) return false;
  rawBytes->insert(rawBytes->end(), decoded.begin(), decoded.end());
  generatedTokens->push_back(static_cast<std::uint16_t>(token));
  return true;
}

struct CrossTokenizerMetrics {
  double tokenNll = 0;
  double natsPerUtf8Byte = 0;
  double bitsPerUtf8Byte = 0;
};

inline CrossTokenizerMetrics crossTokenizerMetrics(
    double totalNegativeLogLikelihood, std::uint64_t targetTokens,
    std::uint64_t targetUtf8Bytes) {
  if (!std::isfinite(totalNegativeLogLikelihood) || targetTokens == 0 ||
      targetUtf8Bytes == 0)
    throw std::runtime_error("BPE_METRIC_DENOMINATOR");
  CrossTokenizerMetrics result;
  result.tokenNll = totalNegativeLogLikelihood / targetTokens;
  result.natsPerUtf8Byte = totalNegativeLogLikelihood / targetUtf8Bytes;
  result.bitsPerUtf8Byte = result.natsPerUtf8Byte / std::log(2.0);
  return result;
}

}  // namespace phonelm::nicopedia_bpe
