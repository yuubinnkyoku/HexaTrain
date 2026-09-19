// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only deterministic exporter and contract validator for PhoneLM's
// single source of truth: transformer_parameter_metadata.h.
//
// Usage:
//   export_transformer_parameter_metadata                -> emit JSON to stdout
//   export_transformer_parameter_metadata --self-test   -> run contract tests
//   export_transformer_parameter_metadata --check <file> -> fail if file differs
//   export_transformer_parameter_metadata --write <file> -> write JSON to file

#include "transformer_parameter_metadata.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace phonelm::tools {

inline const char* roleString(tiny_lm::ParameterRole role) {
  switch (role) {
    case tiny_lm::ParameterRole::MUON:
      return "MUON";
    case tiny_lm::ParameterRole::AUX_ADAM:
      return "AUX_ADAM";
    case tiny_lm::ParameterRole::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

inline const char* placementString(tiny_lm::ParameterPlacement placement) {
  switch (placement) {
    case tiny_lm::ParameterPlacement::GLOBAL_PREFIX:
      return "GLOBAL_PREFIX";
    case tiny_lm::ParameterPlacement::PER_LAYER:
      return "PER_LAYER";
    case tiny_lm::ParameterPlacement::GLOBAL_SUFFIX:
      return "GLOBAL_SUFFIX";
  }
  return "UNKNOWN";
}

inline const char* conditionString(tiny_lm::ParameterCondition condition) {
  switch (condition) {
    case tiny_lm::ParameterCondition::ALWAYS:
      return "ALWAYS";
    case tiny_lm::ParameterCondition::HEADWISE_G1:
      return "HEADWISE_G1";
  }
  return "UNKNOWN";
}

inline const char* dimensionString(tiny_lm::ParameterDimension dimension) {
  switch (dimension) {
    case tiny_lm::ParameterDimension::VOCABULARY:
      return "VOCABULARY";
    case tiny_lm::ParameterDimension::MODEL:
      return "MODEL";
    case tiny_lm::ParameterDimension::FEED_FORWARD:
      return "FEED_FORWARD";
    case tiny_lm::ParameterDimension::HEADS:
      return "HEADS";
    case tiny_lm::ParameterDimension::NONE:
      return "NONE";
  }
  return "UNKNOWN";
}

inline std::string generateParameterMetadataJson() {
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema_version\": 1,\n";
  out << "  \"generator\": \"export_transformer_parameter_metadata\",\n";
  out << "  \"parameter_definitions\": [\n";

  const auto& definitions = tiny_lm::parameterDefinitions();
  for (std::size_t i = 0; i < definitions.size(); ++i) {
    const auto& def = definitions[i];
    out << "    {\n";
    out << "      \"suffix\": \"" << def.suffix << "\",\n";
    out << "      \"role\": \"" << roleString(def.role) << "\",\n";
    out << "      \"placement\": \"" << placementString(def.placement) << "\",\n";
    out << "      \"condition\": \"" << conditionString(def.condition) << "\",\n";
    out << "      \"shape\": [";
    for (std::uint8_t axis = 0; axis < def.rank; ++axis) {
      if (axis > 0) out << ", ";
      out << "\"" << dimensionString(def.shape[axis]) << "\"";
    }
    out << "],\n";
    out << "      \"rank\": " << static_cast<int>(def.rank) << ",\n";
    out << "      \"fan_out_axis\": " << static_cast<int>(def.fanOutAxis) << ",\n";
    out << "      \"fan_in_axis\": " << static_cast<int>(def.fanInAxis) << "\n";
    out << "    }" << (i + 1 < definitions.size() ? "," : "") << "\n";
  }

  out << "  ]\n";
  out << "}\n";
  return out.str();
}

inline bool readFileContent(const std::string& path, std::string* content) {
  if (!content) return false;
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  std::ostringstream ss;
  ss << file.rdbuf();
  *content = ss.str();
  return true;
}

inline bool writeFileContent(const std::string& path, const std::string& content) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file.write(content.data(), static_cast<std::streamsize>(content.size()));
  return file.good();
}

inline bool runContractTests() {
  const auto& defs = tiny_lm::parameterDefinitions();

  // 1. Definition count and validity
  if (defs.size() != 13) {
    std::cerr << "CONTRACT_FAIL: expected 13 definitions, got " << defs.size() << "\n";
    return false;
  }
  for (const auto& def : defs) {
    if (!tiny_lm::validParameterDefinition(def)) {
      std::cerr << "CONTRACT_FAIL: invalid definition: " << (def.suffix ? def.suffix : "null") << "\n";
      return false;
    }
  }

  // 2. Canonical ordering checks
  if (std::string_view(defs[0].suffix) != "token_embedding" ||
      defs[0].placement != tiny_lm::ParameterPlacement::GLOBAL_PREFIX) {
    std::cerr << "CONTRACT_FAIL: definition 0 must be token_embedding GLOBAL_PREFIX\n";
    return false;
  }
  if (std::string_view(defs[12].suffix) != "output_projection" ||
      defs[12].placement != tiny_lm::ParameterPlacement::GLOBAL_SUFFIX) {
    std::cerr << "CONTRACT_FAIL: definition 12 must be output_projection GLOBAL_SUFFIX\n";
    return false;
  }

  // 3. Placement breakdown
  std::size_t globalPrefixCount = 0;
  std::size_t perLayerCount = 0;
  std::size_t globalSuffixCount = 0;
  for (const auto& def : defs) {
    if (def.placement == tiny_lm::ParameterPlacement::GLOBAL_PREFIX) ++globalPrefixCount;
    if (def.placement == tiny_lm::ParameterPlacement::PER_LAYER) ++perLayerCount;
    if (def.placement == tiny_lm::ParameterPlacement::GLOBAL_SUFFIX) ++globalSuffixCount;
  }
  if (globalPrefixCount != 1 || perLayerCount != 11 || globalSuffixCount != 1) {
    std::cerr << "CONTRACT_FAIL: placement count mismatch (prefix=" << globalPrefixCount
              << ", per_layer=" << perLayerCount << ", suffix=" << globalSuffixCount << ")\n";
    return false;
  }

  // 4. Gated condition identity
  std::size_t gatedCount = 0;
  for (const auto& def : defs) {
    if (def.condition == tiny_lm::ParameterCondition::HEADWISE_G1) {
      ++gatedCount;
      if (std::string_view(def.suffix) != "attention_gate_weight") {
        std::cerr << "CONTRACT_FAIL: HEADWISE_G1 applied to unexpected parameter: " << def.suffix << "\n";
        return false;
      }
    }
  }
  if (gatedCount != 1) {
    std::cerr << "CONTRACT_FAIL: expected exactly 1 HEADWISE_G1 parameter, got " << gatedCount << "\n";
    return false;
  }

  // 5. Distinct identity for same-shaped matrices (wq, wk, wv, wo)
  std::vector<std::string_view> attentionMatrices;
  for (const auto& def : defs) {
    if (def.role == tiny_lm::ParameterRole::MUON &&
        def.shape[0] == tiny_lm::ParameterDimension::MODEL &&
        def.shape[1] == tiny_lm::ParameterDimension::MODEL) {
      attentionMatrices.push_back(def.suffix);
    }
  }
  if (attentionMatrices.size() != 4 ||
      attentionMatrices[0] != "wq" || attentionMatrices[1] != "wk" ||
      attentionMatrices[2] != "wv" || attentionMatrices[3] != "wo") {
    std::cerr << "CONTRACT_FAIL: attention matrix order/identity mismatch\n";
    return false;
  }

  // 6. Determinism check
  const std::string json1 = generateParameterMetadataJson();
  const std::string json2 = generateParameterMetadataJson();
  if (json1 != json2 || json1.empty()) {
    std::cerr << "CONTRACT_FAIL: generator is not deterministic\n";
    return false;
  }

  // 7. Verify JSON string contains expected keys and suffixes
  if (json1.find("\"schema_version\": 1") == std::string::npos ||
      json1.find("\"suffix\": \"token_embedding\"") == std::string::npos ||
      json1.find("\"suffix\": \"attention_gate_weight\"") == std::string::npos ||
      json1.find("\"suffix\": \"output_projection\"") == std::string::npos) {
    std::cerr << "CONTRACT_FAIL: JSON missing expected content\n";
    return false;
  }

  return true;
}

}  // namespace phonelm::tools

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    if (!phonelm::tools::runContractTests()) {
      std::cerr << "export_transformer_parameter_metadata contract tests: FAILED\n";
      return 1;
    }
    std::cout << "export_transformer_parameter_metadata contract tests: PASS\n";
    return 0;
  }

  if (argc == 3 && std::string_view(argv[1]) == "--check") {
    const std::string targetPath = argv[2];
    std::string existingContent;
    if (!phonelm::tools::readFileContent(targetPath, &existingContent)) {
      std::cerr << "ERROR: unable to read target file for staleness check: " << targetPath << "\n";
      return 1;
    }
    const std::string generated = phonelm::tools::generateParameterMetadataJson();
    // Normalize CRLF to LF for portable comparison across Windows/Linux git checkouts
    std::string normalizedExisting;
    normalizedExisting.reserve(existingContent.size());
    for (char c : existingContent) {
      if (c != '\r') normalizedExisting.push_back(c);
    }
    if (normalizedExisting != generated) {
      std::cerr << "STALENESS_CHECK_FAILED: " << targetPath
                << " differs from C++ SSOT (transformer_parameter_metadata.h).\n"
                << "Run 'pwsh scripts/generate_parameter_metadata.ps1' to update the generated artifact.\n";
      return 2;
    }
    std::cout << "metadata staleness check: PASS (" << targetPath << " matches C++ SSOT)\n";
    return 0;
  }

  if (argc == 3 && std::string_view(argv[1]) == "--write") {
    const std::string targetPath = argv[2];
    const std::string generated = phonelm::tools::generateParameterMetadataJson();
    if (!phonelm::tools::writeFileContent(targetPath, generated)) {
      std::cerr << "ERROR: failed to write generated metadata to: " << targetPath << "\n";
      return 1;
    }
    std::cout << "wrote parameter metadata to: " << targetPath << "\n";
    return 0;
  }

  if (argc == 1) {
    std::cout << phonelm::tools::generateParameterMetadataJson();
    return 0;
  }

  std::cerr << "Usage: export_transformer_parameter_metadata [--self-test | --check <file> | --write <file>]\n";
  return 1;
}
