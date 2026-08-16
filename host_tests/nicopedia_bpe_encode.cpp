#include "nicopedia_byte_bpe.h"

#include <fstream>
#include <iostream>

namespace {

std::uint32_t readU32(std::istream& input) {
  std::uint32_t value = 0;
  for (int index = 0; index < 4; ++index) {
    const int byte = input.get();
    if (byte < 0) throw std::runtime_error("ENCODE_INPUT_TRUNCATED");
    value = (value << 8) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

std::uint64_t readU64(std::istream& input) {
  std::uint64_t value = 0;
  for (int index = 0; index < 8; ++index) {
    const int byte = input.get();
    if (byte < 0) throw std::runtime_error("ENCODE_INPUT_TRUNCATED");
    value = (value << 8) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

void writeU32(std::ostream& output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) output.put(static_cast<char>(value >> shift));
}

void writeU64(std::ostream& output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) output.put(static_cast<char>(value >> shift));
}

void writeU16(std::ostream& output, std::uint16_t value) {
  output.put(static_cast<char>(value >> 8));
  output.put(static_cast<char>(value));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: nicopedia_bpe_encode INPUT MODEL OUTPUT\n";
      return 2;
    }
    const auto model = phonelm::nicopedia_bpe::loadModel(argv[2]);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("ENCODE_INPUT_OPEN");
    std::string magic(11, '\0');
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != "NPRTBPEEN1\n") throw std::runtime_error("ENCODE_INPUT_MAGIC");
    const std::uint32_t articleCount = readU32(input);
    if (articleCount == 0 || articleCount > 1000000) throw std::runtime_error("ENCODE_ARTICLE_COUNT");
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("ENCODE_OUTPUT_OPEN");
    output.write("NPRTBPEEO1\n", 11);
    writeU32(output, articleCount);
    std::uint64_t totalBytes = 0, totalTokens = 0;
    for (std::uint32_t article = 0; article < articleCount; ++article) {
      const std::uint64_t articleId = readU64(input);
      const std::uint64_t articleHash = readU64(input);
      const std::uint64_t byteCount = readU64(input);
      if (byteCount == 0 || byteCount > 16u * 1024u * 1024u)
        throw std::runtime_error("ENCODE_ARTICLE_LENGTH");
      std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byteCount));
      input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      if (!input) throw std::runtime_error("ENCODE_ARTICLE_TRUNCATED");
      const auto tokens = model.encode(bytes);
      if (model.decode(tokens) != bytes) throw std::runtime_error("ENCODE_ROUND_TRIP");
      writeU64(output, articleId);
      writeU64(output, articleHash);
      writeU64(output, byteCount);
      writeU64(output, tokens.size());
      for (const auto token : tokens) writeU16(output, token);
      totalBytes += byteCount;
      totalTokens += tokens.size();
    }
    if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("ENCODE_INPUT_TRAILING");
    if (!output) throw std::runtime_error("ENCODE_OUTPUT_WRITE");
    std::cout << "bpe_encode_status=PASS articles=" << articleCount
              << " bytes=" << totalBytes << " tokens=" << totalTokens
              << " tokenizer_hash=" << model.identity() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bpe_encode_status=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
