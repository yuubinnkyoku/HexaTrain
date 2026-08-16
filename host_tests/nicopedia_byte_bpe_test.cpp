#include "nicopedia_byte_bpe.h"

#include <iostream>
#include <stdexcept>

namespace bpe = phonelm::nicopedia_bpe;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t> hex(const std::string& value) {
  std::vector<std::uint8_t> result;
  for (std::size_t index = 0; index < value.size(); index += 2)
    result.push_back(static_cast<std::uint8_t>(std::stoul(value.substr(index, 2), nullptr, 16)));
  return result;
}

std::vector<std::uint8_t> modelBytes() {
  std::vector<std::uint8_t> bytes(bpe::kModelMagic,
                                  bpe::kModelMagic + sizeof(bpe::kModelMagic) - 1);
  appendU32(bytes, bpe::kVocabulary);
  appendU32(bytes, bpe::kMergeCount);
  for (std::uint16_t index = 0; index < bpe::kMergeCount; ++index) {
    if (index == 0) { appendU16(bytes, 65); appendU16(bytes, 66); }
    else if (index == 1) { appendU16(bytes, 256); appendU16(bytes, 67); }
    else { appendU16(bytes, static_cast<std::uint16_t>(255 + index)); appendU16(bytes, 65); }
  }
  return bytes;
}

std::vector<std::uint8_t> cacheBytes(const bpe::Model& model) {
  std::vector<std::uint8_t> bytes(bpe::kCacheMagic,
                                  bpe::kCacheMagic + sizeof(bpe::kCacheMagic) - 1);
  appendU32(bytes, 4); appendU32(bytes, bpe::kVocabulary);
  const auto digest = hex(model.sha256);
  bytes.insert(bytes.end(), digest.begin(), digest.end());
  appendU64(bytes, 1); appendU64(bytes, 0x123456789abcdef0ull);
  for (std::uint16_t token : {std::uint16_t(65), std::uint16_t(66), std::uint16_t(256),
                              std::uint16_t(257), std::uint16_t(1023)}) appendU16(bytes, token);
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto canonical = modelBytes();
    const auto model = bpe::parseModel(canonical);
    require(model.identity() == "sha256:6ee0da7a526b8f7a3f24ef831ed23db08802657c6d003052d73a2f82954b7711",
            "cross-language canonical hash");
    require(model.expansions[0] == std::vector<std::uint8_t>{0}, "literal byte zero");
    require(model.expansions[255] == std::vector<std::uint8_t>{255}, "literal byte 255");
    const std::vector<std::uint8_t> payload = {65,66,67,65,66,0,255};
    const auto encoded = model.encode(payload);
    require(encoded == std::vector<std::uint16_t>({257,256,0,255}), "merge rank semantics");
    require(model.decode(encoded) == payload, "exact byte round trip");
    auto context = bpe::buildTokenContext(encoded, 3);
    require(context == std::vector<std::uint16_t>({256,0,255}), "last three tokens context");
    std::vector<std::uint8_t> cappedBytes;
    std::vector<std::uint16_t> cappedTokens;
    require(bpe::appendTokenWithinByteLimit(&model, 257, 3, &cappedBytes, &cappedTokens),
            "token exactly at MaxNewBytes");
    require(!bpe::appendTokenWithinByteLimit(&model, 256, 4, &cappedBytes, &cappedTokens),
            "overflow token discarded without resample");
    require(cappedBytes == std::vector<std::uint8_t>({65,66,67}) && cappedTokens.size() == 1,
            "MaxNewBytes state unchanged after rejection");
    const auto byteMetric = bpe::crossTokenizerMetrics(8.0, 4, 4);
    const auto bpeMetric = bpe::crossTokenizerMetrics(8.0, 4, 8);
    require(byteMetric.tokenNll == byteMetric.natsPerUtf8Byte,
            "V256 token NLL equals nats per byte");
    require(bpeMetric.tokenNll == 2.0 && bpeMetric.natsPerUtf8Byte == 1.0 &&
                std::abs(bpeMetric.bitsPerUtf8Byte - 1.0 / std::log(2.0)) < 1e-12,
            "BPE nats/bits per byte fixture");

    const auto cache = bpe::parseCache(cacheBytes(model), model);
    require(cache.context == 4 && cache.vocabulary == 1024 && cache.records.size() == 1,
            "cache header");
    require(cache.records[0].window.back() == 1023, "uint16 token round trip");
    auto truncated = cacheBytes(model); truncated.pop_back();
    try { (void)bpe::parseCache(truncated, model); throw std::runtime_error("truncated accepted"); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()) == "BPE_CACHE_SIZE", "truncated rejected");
    }
    auto wrongHash = cacheBytes(model);
    wrongHash[sizeof(bpe::kCacheMagic) - 1 + 8] ^= 1;
    try { (void)bpe::parseCache(wrongHash, model); throw std::runtime_error("wrong hash accepted"); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()) == "BPE_CACHE_TOKENIZER_HASH", "wrong hash rejected");
    }
    auto outOfRange = cacheBytes(model);
    outOfRange[outOfRange.size() - 2] = 4; outOfRange[outOfRange.size() - 1] = 0;
    try { (void)bpe::parseCache(outOfRange, model); throw std::runtime_error("token range accepted"); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()) == "BPE_CACHE_TOKEN_RANGE", "token range rejected");
    }
    auto malformed = canonical;
    malformed[sizeof(bpe::kModelMagic) - 1 + 8] = 1;
    try { (void)bpe::parseModel(malformed); throw std::runtime_error("forward reference accepted"); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()) == "BPE_MODEL_FORWARD_REFERENCE", "malformed model rejected");
    }
    std::size_t privateRecords = 0;
    if (argc > 1) {
      require(argc >= 3, "private model requires at least one cache");
      const auto privateModel = bpe::loadModel(argv[1]);
      for (int index = 2; index < argc; ++index)
        privateRecords += bpe::loadCache(argv[index], privateModel).records.size();
      std::cout << "private_bpe_artifact_audit=PASS cache_count=" << (argc - 2)
                << " record_count=" << privateRecords
                << " tokenizer_hash=" << privateModel.identity() << '\n';
    }
    std::cout << "nicopedia_byte_bpe_host_test=PASS tokenizer_hash=" << model.identity() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "nicopedia_byte_bpe_host_test=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
