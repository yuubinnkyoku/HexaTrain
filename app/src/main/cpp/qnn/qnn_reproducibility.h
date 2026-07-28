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
// Runs one graph-preserving internal-tensor tap set on the fixed E state.
std::string runTinyLmFirstChangeTap(int tapCode);
// Runs one selected three-variant order with a fresh Runtime/context per slot.
// planCode 0..5 are permutations; 6..8 are homogeneous controls.
std::string runTinyLmGraphOrderOrthogonalization(int planCode);
std::string rawFloatSha256(const std::vector<float>& values);
std::string canonicalFloatSha256(const std::vector<float>& values);

}  // namespace phonelm::qnn
