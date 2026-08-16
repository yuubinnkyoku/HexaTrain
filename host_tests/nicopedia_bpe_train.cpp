#include "nicopedia_byte_bpe.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace {

std::uint32_t readU32(std::istream& input) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const int byte = input.get();
    if (byte < 0) throw std::runtime_error("TRAIN_INPUT_TRUNCATED");
    value = (value << 8) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

std::uint64_t readU64(std::istream& input) {
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    const int byte = input.get();
    if (byte < 0) throw std::runtime_error("TRAIN_INPUT_TRUNCATED");
    value = (value << 8) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

std::uint32_t pairKey(std::uint16_t left, std::uint16_t right) {
  return (std::uint32_t(left) << 16) | right;
}

struct Candidate {
  std::int64_t count = 0;
  std::uint16_t left = 0, right = 0;
};

struct CandidateLess {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.count != b.count) return a.count < b.count;
    if (a.left != b.left) return a.left > b.left;
    return a.right > b.right;
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: nicopedia_bpe_train TRAIN_ARTICLES OUTPUT_MODEL\n";
      return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("TRAIN_INPUT_OPEN");
    std::string magic(11, '\0'); input.read(magic.data(), 11);
    if (magic != "NPRTBPETR1\n") throw std::runtime_error("TRAIN_INPUT_MAGIC");
    const std::uint32_t articleCount = readU32(input);
    if (articleCount == 0 || articleCount > 1000000) throw std::runtime_error("TRAIN_ARTICLE_COUNT");
    std::vector<std::uint16_t> tokens;
    std::vector<std::int32_t> previous, next;
    std::unordered_map<std::uint32_t, std::int64_t> counts;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> occurrences;
    counts.reserve(1 << 20); occurrences.reserve(1 << 20);
    for (std::uint32_t article = 0; article < articleCount; ++article) {
      const std::uint64_t length64 = readU64(input);
      if (length64 == 0 || length64 > 16u * 1024u * 1024u ||
          length64 > std::numeric_limits<std::uint32_t>::max() - tokens.size())
        throw std::runtime_error("TRAIN_ARTICLE_LENGTH");
      const std::uint32_t length = static_cast<std::uint32_t>(length64);
      std::vector<std::uint8_t> bytes(length);
      input.read(reinterpret_cast<char*>(bytes.data()), length);
      if (!input) throw std::runtime_error("TRAIN_ARTICLE_TRUNCATED");
      const std::uint32_t base = static_cast<std::uint32_t>(tokens.size());
      tokens.reserve(tokens.size() + length);
      previous.reserve(previous.size() + length); next.reserve(next.size() + length);
      for (std::uint32_t index = 0; index < length; ++index) {
        tokens.push_back(bytes[index]);
        previous.push_back(index == 0 ? -1 : static_cast<std::int32_t>(base + index - 1));
        next.push_back(index + 1 == length ? -1 : static_cast<std::int32_t>(base + index + 1));
      }
      for (std::uint32_t index = 0; index + 1 < length; ++index) {
        const std::uint32_t key = pairKey(bytes[index], bytes[index + 1]);
        ++counts[key]; occurrences[key].push_back(base + index);
      }
    }
    if (input.get() != std::char_traits<char>::eof()) throw std::runtime_error("TRAIN_INPUT_TRAILING");
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> queue;
    for (const auto& [key, count] : counts)
      if (count > 0) queue.push({count, static_cast<std::uint16_t>(key >> 16), static_cast<std::uint16_t>(key)});
    std::array<std::pair<std::uint16_t, std::uint16_t>, phonelm::nicopedia_bpe::kMergeCount> merges{};
    for (std::uint16_t mergeIndex = 0; mergeIndex < phonelm::nicopedia_bpe::kMergeCount; ++mergeIndex) {
      Candidate selected;
      for (;;) {
        if (queue.empty()) throw std::runtime_error("TRAIN_INSUFFICIENT_PAIRS");
        selected = queue.top(); queue.pop();
        const auto found = counts.find(pairKey(selected.left, selected.right));
        if (found != counts.end() && found->second == selected.count && selected.count > 0) break;
      }
      const std::uint32_t selectedKey = pairKey(selected.left, selected.right);
      const std::uint16_t newToken = static_cast<std::uint16_t>(256u + mergeIndex);
      std::unordered_set<std::uint32_t> changed;
      changed.reserve(1024); changed.insert(selectedKey);
      const auto& positions = occurrences.at(selectedKey);
      for (std::uint32_t leftPosition : positions) {
        const std::int32_t rightPosition = next[leftPosition];
        if (rightPosition < 0 || tokens[leftPosition] != selected.left ||
            tokens[static_cast<std::size_t>(rightPosition)] != selected.right) continue;
        const std::int32_t before = previous[leftPosition];
        const std::int32_t after = next[static_cast<std::size_t>(rightPosition)];
        --counts[selectedKey];
        if (before >= 0) {
          const auto old = pairKey(tokens[static_cast<std::size_t>(before)], tokens[leftPosition]);
          --counts[old]; changed.insert(old);
        }
        if (after >= 0) {
          const auto old = pairKey(tokens[static_cast<std::size_t>(rightPosition)], tokens[static_cast<std::size_t>(after)]);
          --counts[old]; changed.insert(old);
        }
        tokens[leftPosition] = newToken; next[leftPosition] = after;
        if (after >= 0) previous[static_cast<std::size_t>(after)] = static_cast<std::int32_t>(leftPosition);
        next[static_cast<std::size_t>(rightPosition)] = -2;
        previous[static_cast<std::size_t>(rightPosition)] = -2;
        if (before >= 0) {
          const auto key = pairKey(tokens[static_cast<std::size_t>(before)], newToken);
          ++counts[key]; occurrences[key].push_back(static_cast<std::uint32_t>(before)); changed.insert(key);
        }
        if (after >= 0) {
          const auto key = pairKey(newToken, tokens[static_cast<std::size_t>(after)]);
          ++counts[key]; occurrences[key].push_back(leftPosition); changed.insert(key);
        }
      }
      merges[mergeIndex] = {selected.left, selected.right};
      for (std::uint32_t key : changed) {
        const std::int64_t count = counts[key];
        if (count > 0) queue.push({count, static_cast<std::uint16_t>(key >> 16), static_cast<std::uint16_t>(key)});
      }
      if ((mergeIndex + 1) % 64 == 0)
        std::cerr << "bpe_merges_completed=" << (mergeIndex + 1) << '\n';
    }
    std::vector<std::uint8_t> canonical(phonelm::nicopedia_bpe::kModelMagic,
        phonelm::nicopedia_bpe::kModelMagic + sizeof(phonelm::nicopedia_bpe::kModelMagic) - 1);
    for (const std::uint32_t value : {phonelm::nicopedia_bpe::kVocabulary,
                                      phonelm::nicopedia_bpe::kMergeCount}) {
      for (int shift = 24; shift >= 0; shift -= 8)
        canonical.push_back(static_cast<std::uint8_t>(value >> shift));
    }
    for (const auto& [left, right] : merges) {
      canonical.push_back(static_cast<std::uint8_t>(left >> 8)); canonical.push_back(static_cast<std::uint8_t>(left));
      canonical.push_back(static_cast<std::uint8_t>(right >> 8)); canonical.push_back(static_cast<std::uint8_t>(right));
    }
    const auto model = phonelm::nicopedia_bpe::parseModel(canonical);
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(canonical.data()), static_cast<std::streamsize>(canonical.size()));
    if (!output) throw std::runtime_error("MODEL_WRITE");
    std::cout << "bpe_train_status=PASS articles=" << articleCount
              << " bytes=" << tokens.size() << " tokenizer_hash=" << model.identity() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bpe_train_status=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
