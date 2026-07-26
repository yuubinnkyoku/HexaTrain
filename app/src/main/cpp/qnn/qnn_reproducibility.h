#pragma once

#include <string>
#include <vector>

namespace phonelm::qnn {

// Runs the fixed-input OneHot^T @ dX diagnostic used to isolate lm_dembedding.
// The report intentionally contains aggregate hashes and statistics only.
std::string runTinyLmDembeddingReproducibility();
// Compares the fixed E checkpoint on the full graph and two graph-prefix
// variants in one headless fresh process.
std::string runTinyLmGraphBisection(bool standalonePrelude);
std::string runTinyLmGraphIsolated(int variantCode);
std::string rawFloatSha256(const std::vector<float>& values);
std::string canonicalFloatSha256(const std::vector<float>& values);

}  // namespace phonelm::qnn
