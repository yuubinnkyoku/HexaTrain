// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Host-only deterministic exporter and contract validator for PhoneLM's
// single source of truth: transformer_parameter_metadata.h.
//
// Usage:
//   export_transformer_parameter_metadata                                  -> emit JSON to stdout
//   export_transformer_parameter_metadata --self-test                     -> run generic contract tests
//   export_transformer_parameter_metadata --check <json-file> [kt-file]   -> fail if files differ
//   export_transformer_parameter_metadata --write <json-file> [kt-file]   -> write artifacts to files

#include "transformer_parameter_metadata.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
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

// Lossless string emitters for exporter artifacts. validParameterDefinition()
// only requires a non-empty suffix; the exporters must therefore escape
// quote/backslash (and other controls) rather than narrow the vocabulary.
inline std::string escapeJsonString(std::string_view input) {
  std::ostringstream out;
  for (unsigned char c : input) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

inline std::string escapeKotlinString(std::string_view input) {
  std::ostringstream out;
  for (unsigned char c : input) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      case '$': out << "\\$"; break;
      case '\'': out << "\\'"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
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
    out << "      \"suffix\": \"" << escapeJsonString(def.suffix) << "\",\n";
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

inline std::string generateParameterMetadataKotlin() {
  std::ostringstream out;
  out << "// SPDX-License-Identifier: Apache-2.0\n";
  out << "// Auto-generated by export_transformer_parameter_metadata. DO NOT EDIT.\n";
  out << "package com.yuubinnkyoku.phonelm\n\n";

  out << "enum class GeneratedParameterRole { MUON, AUX_ADAM }\n";
  out << "enum class GeneratedParameterPlacement { GLOBAL_PREFIX, PER_LAYER, GLOBAL_SUFFIX }\n";
  out << "enum class GeneratedParameterCondition { ALWAYS, HEADWISE_G1 }\n";
  out << "enum class GeneratedParameterDimension { VOCABULARY, MODEL, FEED_FORWARD, HEADS }\n\n";

  out << "data class GeneratedParameterDefinition(\n";
  out << "    val suffix: String,\n";
  out << "    val role: GeneratedParameterRole,\n";
  out << "    val placement: GeneratedParameterPlacement,\n";
  out << "    val condition: GeneratedParameterCondition,\n";
  out << "    val shape: List<GeneratedParameterDimension>,\n";
  out << "    val rank: Int,\n";
  out << "    val fanOutAxis: Int,\n";
  out << "    val fanInAxis: Int,\n";
  out << ")\n\n";

  out << "object GeneratedTransformerParameterMetadata {\n";
  out << "    const val SCHEMA_VERSION: Int = 1\n\n";
  out << "    val DEFINITIONS: List<GeneratedParameterDefinition> = listOf(\n";

  const auto& definitions = tiny_lm::parameterDefinitions();
  for (std::size_t i = 0; i < definitions.size(); ++i) {
    const auto& def = definitions[i];
    out << "        GeneratedParameterDefinition(\n";
    out << "            suffix = \"" << escapeKotlinString(def.suffix) << "\",\n";
    out << "            role = GeneratedParameterRole." << roleString(def.role) << ",\n";
    out << "            placement = GeneratedParameterPlacement." << placementString(def.placement) << ",\n";
    out << "            condition = GeneratedParameterCondition." << conditionString(def.condition) << ",\n";
    out << "            shape = listOf(";
    for (std::uint8_t axis = 0; axis < def.rank; ++axis) {
      if (axis > 0) out << ", ";
      out << "GeneratedParameterDimension." << dimensionString(def.shape[axis]);
    }
    out << "),\n";
    out << "            rank = " << static_cast<int>(def.rank) << ",\n";
    out << "            fanOutAxis = " << static_cast<int>(def.fanOutAxis) << ",\n";
    out << "            fanInAxis = " << static_cast<int>(def.fanInAxis) << ",\n";
    out << "        )" << (i + 1 < definitions.size() ? "," : "") << "\n";
  }
  out << "    )\n\n";

  out << "    fun calculateParameterCount(\n";
  out << "        vocabularySize: Long,\n";
  out << "        dimension: Long,\n";
  out << "        feedForwardDimension: Long,\n";
  out << "        layers: Long,\n";
  out << "        heads: Long,\n";
  out << "        headwiseG1: Boolean = false,\n";
  out << "    ): Long {\n";
  out << "        var total = 0L\n";
  out << "        for (def in DEFINITIONS) {\n";
  out << "            if (def.condition == GeneratedParameterCondition.HEADWISE_G1 && !headwiseG1) continue\n";
  out << "            var elements = 1L\n";
  out << "            for (dim in def.shape) {\n";
  out << "                val extent = when (dim) {\n";
  out << "                    GeneratedParameterDimension.VOCABULARY -> vocabularySize\n";
  out << "                    GeneratedParameterDimension.MODEL -> dimension\n";
  out << "                    GeneratedParameterDimension.FEED_FORWARD -> feedForwardDimension\n";
  out << "                    GeneratedParameterDimension.HEADS -> heads\n";
  out << "                }\n";
  out << "                elements = Math.multiplyExact(elements, extent)\n";
  out << "            }\n";
  out << "            val instances = when (def.placement) {\n";
  out << "                GeneratedParameterPlacement.PER_LAYER -> layers\n";
  out << "                GeneratedParameterPlacement.GLOBAL_PREFIX, GeneratedParameterPlacement.GLOBAL_SUFFIX -> 1L\n";
  out << "            }\n";
  out << "            total = Math.addExact(total, Math.multiplyExact(elements, instances))\n";
  out << "        }\n";
  out << "        return total\n";
  out << "    }\n";
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

inline std::string normalizeCrLf(const std::string& input) {
  std::string normalized;
  normalized.reserve(input.size());
  for (char c : input) {
    if (c != '\r') normalized.push_back(c);
  }
  return normalized;
}

inline bool checkFileContent(const std::string& path, const std::string& expected, const char* label) {
  std::string existing;
  if (!readFileContent(path, &existing)) {
    std::cerr << "STALENESS_CHECK_FAILED: unable to read " << label << " from " << path << "\n";
    return false;
  }
  if (normalizeCrLf(existing) != normalizeCrLf(expected)) {
    std::cerr << "STALENESS_CHECK_FAILED: " << path << " differs from C++ SSOT.\n"
              << "Run 'pwsh scripts/generate_parameter_metadata.ps1' to update.\n";
    return false;
  }
  return true;
}

inline bool runGenericContractTests() {
  const auto& defs = tiny_lm::parameterDefinitions();

  // 1. Definition list is non-empty
  if (defs.empty()) {
    std::cerr << "CONTRACT_FAIL: definition list is empty\n";
    return false;
  }

  std::unordered_set<std::string> suffixes;
  std::size_t prefixCount = 0;
  std::size_t perLayerCount = 0;
  std::size_t suffixCount = 0;

  for (const auto& def : defs) {
    // 2. Each definition satisfies the canonical validity predicate
    if (!tiny_lm::validParameterDefinition(def)) {
      std::cerr << "CONTRACT_FAIL: invalid definition: " << (def.suffix ? def.suffix : "null") << "\n";
      return false;
    }

    // 3. Suffix is non-empty and unique across definitions
    if (!def.suffix || def.suffix[0] == '\0') {
      std::cerr << "CONTRACT_FAIL: empty suffix\n";
      return false;
    }
    if (!suffixes.insert(def.suffix).second) {
      std::cerr << "CONTRACT_FAIL: duplicate suffix: " << def.suffix << "\n";
      return false;
    }

    // 4. Rank and shape dimensions consistency
    if (def.rank == 0 || def.rank > def.shape.size()) {
      std::cerr << "CONTRACT_FAIL: rank out of range for " << def.suffix << "\n";
      return false;
    }
    for (std::size_t axis = 0; axis < def.shape.size(); ++axis) {
      if (axis < def.rank) {
        if (def.shape[axis] == tiny_lm::ParameterDimension::NONE) {
          std::cerr << "CONTRACT_FAIL: active dimension is NONE for " << def.suffix << "\n";
          return false;
        }
      } else {
        if (def.shape[axis] != tiny_lm::ParameterDimension::NONE) {
          std::cerr << "CONTRACT_FAIL: inactive dimension is not NONE for " << def.suffix << "\n";
          return false;
        }
      }
    }

    // 5. Muon axes consistency
    if (def.role == tiny_lm::ParameterRole::MUON) {
      if (def.rank != 2 || def.fanOutAxis < 0 || def.fanInAxis < 0 ||
          def.fanOutAxis >= def.rank || def.fanInAxis >= def.rank) {
        std::cerr << "CONTRACT_FAIL: invalid Muon axes for " << def.suffix << "\n";
        return false;
      }
    } else {
      if (def.fanOutAxis != -1 || def.fanInAxis != -1) {
        std::cerr << "CONTRACT_FAIL: non-sentinel fan axes for AUX_ADAM parameter: " << def.suffix << "\n";
        return false;
      }
    }

    // Placement group accounting
    if (def.placement == tiny_lm::ParameterPlacement::GLOBAL_PREFIX) ++prefixCount;
    else if (def.placement == tiny_lm::ParameterPlacement::PER_LAYER) ++perLayerCount;
    else if (def.placement == tiny_lm::ParameterPlacement::GLOBAL_SUFFIX) ++suffixCount;
  }

  if (prefixCount == 0 || perLayerCount == 0 || suffixCount == 0) {
    std::cerr << "CONTRACT_FAIL: missing at least one definition in a placement group\n";
    return false;
  }

  // 6. Determinism: multiple invocations yield byte-identical output
  const std::string json1 = generateParameterMetadataJson();
  const std::string json2 = generateParameterMetadataJson();
  if (json1 != json2 || json1.empty()) {
    std::cerr << "CONTRACT_FAIL: JSON generator not deterministic\n";
    return false;
  }
  const std::string kt1 = generateParameterMetadataKotlin();
  const std::string kt2 = generateParameterMetadataKotlin();
  if (kt1 != kt2 || kt1.empty()) {
    std::cerr << "CONTRACT_FAIL: Kotlin generator not deterministic\n";
    return false;
  }

  // 7. Lossless representation: generated artifacts carry each definition's
  // suffix after JSON/Kotlin escaping. Synthetic quote/backslash cases are
  // asserted separately so the current table is not copied as a fixture.
  for (const auto& def : defs) {
    const std::string needle =
        std::string("\"suffix\": \"") + escapeJsonString(def.suffix) + "\"";
    if (json1.find(needle) == std::string::npos) {
      std::cerr << "CONTRACT_FAIL: JSON missing definition " << def.suffix << "\n";
      return false;
    }
    const std::string ktNeedle =
        std::string("suffix = \"") + escapeKotlinString(def.suffix) + "\"";
    if (kt1.find(ktNeedle) == std::string::npos) {
      std::cerr << "CONTRACT_FAIL: Kotlin missing definition " << def.suffix << "\n";
      return false;
    }
  }

  // 7b. Synthetic suffix serialization contract (quote / backslash / mixed).
  // validParameterDefinition allows any non-empty C-string suffix; the
  // exporters must emit lossless JSON and Kotlin strings for those too.
  {
    struct SyntheticCase {
      const char* raw;
      const char* json;
      const char* kotlin;
    };
    const SyntheticCase cases[] = {
        {"synthetic\\backslash", "synthetic\\\\backslash", "synthetic\\\\backslash"},
        {"synthetic\"quote", "synthetic\\\"quote", "synthetic\\\"quote"},
        {"synthetic\"\\mixed", "synthetic\\\"\\\\mixed", "synthetic\\\"\\\\mixed"},
        {"synthetic\tdollar$", "synthetic\\tdollar$", "synthetic\\tdollar\\$"},
    };
    for (const auto& item : cases) {
      const std::string raw = item.raw;
      tiny_lm::ParameterDefinition synthetic{};
      synthetic.suffix = raw.c_str();
      synthetic.role = tiny_lm::ParameterRole::AUX_ADAM;
      synthetic.placement = tiny_lm::ParameterPlacement::PER_LAYER;
      synthetic.condition = tiny_lm::ParameterCondition::ALWAYS;
      synthetic.shape = {tiny_lm::ParameterDimension::MODEL,
                         tiny_lm::ParameterDimension::NONE};
      synthetic.rank = 1;
      synthetic.fanOutAxis = -1;
      synthetic.fanInAxis = -1;
      synthetic.layerMember = &qnn::TinyTransformerLayerParameters::gamma1;
      synthetic.globalMember = nullptr;
      if (!tiny_lm::validParameterDefinition(synthetic)) {
        std::cerr << "CONTRACT_FAIL: synthetic suffix rejected: " << raw << "\n";
        return false;
      }
      const std::string jsonEscaped = escapeJsonString(raw);
      const std::string kotlinEscaped = escapeKotlinString(raw);
      if (jsonEscaped != item.json || kotlinEscaped != item.kotlin) {
        std::cerr << "CONTRACT_FAIL: suffix escaping mismatch for synthetic="
                  << raw << " json=" << jsonEscaped
                  << " kotlin=" << kotlinEscaped << "\n";
        return false;
      }
      const std::string jsonFragment =
          std::string("\"suffix\": \"") + jsonEscaped + "\"";
      const std::string kotlinFragment =
          std::string("suffix = \"") + kotlinEscaped + "\"";
      if (raw.find('"') != std::string::npos) {
        if (jsonFragment.find("\\\"") == std::string::npos ||
            kotlinFragment.find("\\\"") == std::string::npos) {
          std::cerr << "CONTRACT_FAIL: synthetic quote not escaped\n";
          return false;
        }
      }
      if (raw.find('\\') != std::string::npos) {
        if (jsonFragment.find("\\\\") == std::string::npos ||
            kotlinFragment.find("\\\\") == std::string::npos) {
          std::cerr << "CONTRACT_FAIL: synthetic backslash not escaped\n";
          return false;
        }
      }
    }
  }

  // 7c. Role-count helpers must agree with per-definition traversal without
  // restating the current numeric table.
  {
    const std::vector<tiny_lm::ParameterDimensions> sampleDims = {
        {256, 32, 64, 2, 2, false},
        {256, 32, 64, 2, 2, true},
        {1024, 64, 128, 19, 2, false},
        {1024, 64, 128, 19, 2, true},
    };
    for (const auto& dims : sampleDims) {
      tiny_lm::ParameterRoleCounts roles;
      if (!tiny_lm::checkedParameterRoleCounts(dims, &roles)) {
        std::cerr << "CONTRACT_FAIL: checkedParameterRoleCounts failed\n";
        return false;
      }
      std::uint64_t total = 0, muon = 0, aux = 0, matrices = 0;
      for (const auto& def : defs) {
        if (!tiny_lm::parameterDefinitionEnabled(def, dims)) continue;
        std::uint64_t elements = 0;
        if (!tiny_lm::parameterDefinitionElementCount(def, dims, &elements)) {
          std::cerr << "CONTRACT_FAIL: role traversal element count failed\n";
          return false;
        }
        const std::uint64_t instances =
            def.placement == tiny_lm::ParameterPlacement::PER_LAYER ? dims.layers
                                                                    : 1;
        const std::uint64_t count = elements * instances;
        total += count;
        if (def.role == tiny_lm::ParameterRole::MUON) {
          muon += count;
          matrices += instances;
        } else if (def.role == tiny_lm::ParameterRole::AUX_ADAM) {
          aux += count;
        }
      }
      if (roles.totalParameterCount != total ||
          roles.muonParameterCount != muon ||
          roles.auxiliaryAdamParameterCount != aux ||
          roles.muonMatrixCount != matrices ||
          roles.muonParameterCount + roles.auxiliaryAdamParameterCount !=
              roles.totalParameterCount) {
        std::cerr << "CONTRACT_FAIL: role counts disagree with SSOT traversal\n";
        return false;
      }
    }
  }

  // 8. Calculation consistency with C++ SSOT helpers over sample dimensions
  const std::vector<tiny_lm::ParameterDimensions> sampleDims = {
      {256, 32, 64, 2, 2, false},
      {256, 32, 64, 2, 2, true},
      {1024, 64, 128, 19, 2, false},
      {1024, 64, 128, 19, 2, true},
  };
  for (const auto& dims : sampleDims) {
    std::size_t expectedInstances = 0;
    std::uint64_t expectedElements = 0;
    if (!tiny_lm::checkedParameterInstanceCount(dims, &expectedInstances) ||
        !tiny_lm::checkedParameterElementCount(dims, &expectedElements)) {
      std::cerr << "CONTRACT_FAIL: SSOT helper failed for sample dimensions\n";
      return false;
    }

    std::size_t testInstances = 0;
    std::uint64_t testElements = 0;
    for (const auto& def : defs) {
      if (!tiny_lm::parameterDefinitionEnabled(def, dims)) continue;
      const std::size_t inst = (def.placement == tiny_lm::ParameterPlacement::PER_LAYER) ? dims.layers : 1;
      testInstances += inst;
      std::uint64_t count = 0;
      if (!tiny_lm::parameterDefinitionElementCount(def, dims, &count)) {
        std::cerr << "CONTRACT_FAIL: element count computation failed for " << def.suffix << "\n";
        return false;
      }
      testElements += count * inst;
    }

    if (testInstances != expectedInstances || testElements != expectedElements) {
      std::cerr << "CONTRACT_FAIL: instance or element count mismatch with SSOT helper\n";
      return false;
    }
  }

  return true;
}

}  // namespace phonelm::tools

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    if (!phonelm::tools::runGenericContractTests()) {
      std::cerr << "export_transformer_parameter_metadata contract tests: FAILED\n";
      return 1;
    }
    std::cout << "export_transformer_parameter_metadata contract tests: PASS\n";
    return 0;
  }

  if (argc >= 3 && std::string_view(argv[1]) == "--check") {
    const std::string jsonPath = argv[2];
    const std::string jsonContent = phonelm::tools::generateParameterMetadataJson();
    if (!phonelm::tools::checkFileContent(jsonPath, jsonContent, "JSON metadata")) {
      return 2;
    }
    if (argc >= 4) {
      const std::string ktPath = argv[3];
      const std::string ktContent = phonelm::tools::generateParameterMetadataKotlin();
      if (!phonelm::tools::checkFileContent(ktPath, ktContent, "Kotlin metadata")) {
        return 2;
      }
    }
    std::cout << "metadata staleness check: PASS\n";
    return 0;
  }

  if (argc >= 3 && std::string_view(argv[1]) == "--write") {
    const std::string jsonPath = argv[2];
    const std::string jsonContent = phonelm::tools::generateParameterMetadataJson();
    if (!phonelm::tools::writeFileContent(jsonPath, jsonContent)) {
      std::cerr << "ERROR: failed to write JSON metadata to: " << jsonPath << "\n";
      return 1;
    }
    std::cout << "wrote parameter metadata JSON to: " << jsonPath << "\n";
    if (argc >= 4) {
      const std::string ktPath = argv[3];
      const std::string ktContent = phonelm::tools::generateParameterMetadataKotlin();
      if (!phonelm::tools::writeFileContent(ktPath, ktContent)) {
        std::cerr << "ERROR: failed to write Kotlin metadata to: " << ktPath << "\n";
        return 1;
      }
      std::cout << "wrote parameter metadata Kotlin to: " << ktPath << "\n";
    }
    return 0;
  }

  if (argc == 1) {
    std::cout << phonelm::tools::generateParameterMetadataJson();
    return 0;
  }

  std::cerr << "Usage: export_transformer_parameter_metadata [--self-test | --check <json-file> [kt-file] | --write <json-file> [kt-file]]\n";
  return 1;
}
