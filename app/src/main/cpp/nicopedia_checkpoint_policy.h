// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace phonelm::nicopedia_checkpoint {

// Production anchor identity.  The canonical untagged name
// htp-seed<S>-l<L>-step<N>.ckpt is reserved for T32/D32/FFN32; every other
// configuration must carry explicit -t<T>-d<D>-f<FFN> tags.
inline constexpr std::uint32_t kAnchorTokens = 32;
inline constexpr std::uint32_t kAnchorDimension = 32;
inline constexpr std::uint32_t kAnchorFeedForwardDimension = 32;

// A failed or incomplete segment must never replace its canonical input
// checkpoint. In particular, a failure in resumeStep+1 leaves
// lastCompletedStep == resumeStep while the in-memory state may already hold
// the failed update. Only a fully completed finite target state is eligible
// for the final atomic write.
inline bool finalWriteAllowed(bool allStepsFinite, bool finalStateFinite,
                              std::uint32_t lastCompletedStep,
                              std::uint32_t requestedStep) {
  return allStepsFinite && finalStateFinite && requestedStep > 0 &&
         lastCompletedStep == requestedStep;
}

// A user stop is accepted only at a completed optimizer-step boundary.  The
// last finite V2 state may be committed so an explicit Resume can continue
// from the canonical next step, but an in-flight partial update is never
// serialized as a checkpoint.
inline bool interruptedWriteAllowed(bool allStepsFinite, bool finalStateFinite,
                                    std::uint32_t lastCompletedStep,
                                    std::uint32_t resumeStep) {
  return allStepsFinite && finalStateFinite && lastCompletedStep > resumeStep;
}

// Shared digit validation for the step segment (and, in the -t<T> variant,
// the token/dimension/ffn segments): 1..6 ASCII digits parsing to a value in
// 1..999999, mirroring the legacy checkpoint step rule.
inline bool parseCheckpointDigits(const std::string &digits,
                                  std::uint32_t *value) {
  if (digits.empty() || digits.size() > 6) return false;
  for (char digit : digits)
    if (digit < '0' || digit > '9') return false;
  char *end = nullptr;
  const long parsed = std::strtol(digits.c_str(), &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed >= 1000000) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

// Build the canonical checkpoint filename from model identity + step.
inline std::string checkpointName(std::uint32_t seed, std::uint32_t layers,
                                  std::uint32_t tokens,
                                  std::uint32_t dimension,
                                  std::uint32_t feedForwardDimension,
                                  std::uint32_t step) {
  return "htp-seed" + std::to_string(seed) + "-l" + std::to_string(layers) +
         ((tokens == kAnchorTokens && dimension == kAnchorDimension &&
           feedForwardDimension == kAnchorFeedForwardDimension)
              ? ""
              : "-t" + std::to_string(tokens) + "-d" +
                    std::to_string(dimension) + "-f" +
                    std::to_string(feedForwardDimension)) + "-step" +
         std::to_string(step) + ".ckpt";
}

// Parse a checkpoint filename for seed/layers/step and validate that the
// embedded T/D/FFN identity matches the caller's expected model identity.
//
// Contract:
//   - T32/D32/FFN32 canonical untagged name => ACCEPT (when expectations match)
//   - D16 or any non-anchor tagged name (-t<T>-d<D>-f<FFN>) => ACCEPT
//   - D16 untagged name => REJECT (untagged is reserved for the anchor)
//   - Any T/D/FFN mismatch => REJECT
inline bool parseCheckpointStep(const std::string &path,
                                std::uint32_t expectedSeed,
                                std::uint32_t expectedLayers,
                                std::uint32_t *step,
                                std::uint32_t expectedTokens = kAnchorTokens,
                                std::uint32_t expectedDimension = kAnchorDimension,
                                std::uint32_t expectedFeedForwardDimension =
                                    kAnchorFeedForwardDimension) {
  const std::string base = path.substr(path.find_last_of("/\\") + 1);
  const std::string stepMarker = "-step";
  const std::string suffix = ".ckpt";
  const std::string prefix = "htp-seed" + std::to_string(expectedSeed) + "-l" +
                             std::to_string(expectedLayers) + stepMarker;
  // The legacy untagged name is exclusively the T32/D32/FFN32 anchor.
  if (base.size() > prefix.size() + suffix.size() &&
      base.compare(0, prefix.size(), prefix) == 0 &&
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
    if (expectedTokens != kAnchorTokens ||
        expectedDimension != kAnchorDimension ||
        expectedFeedForwardDimension != kAnchorFeedForwardDimension)
      return false;
    return parseCheckpointDigits(
        base.substr(prefix.size(),
                    base.size() - prefix.size() - suffix.size()),
        step);
  }
  // The extended name must carry and match T, D, and FFN exactly.
  const std::string tPrefix = "htp-seed" + std::to_string(expectedSeed) +
                              "-l" + std::to_string(expectedLayers) + "-t";
  if (base.size() <= tPrefix.size() ||
      base.compare(0, tPrefix.size(), tPrefix) != 0 ||
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) != 0)
    return false;
  const std::size_t dPos = base.find("-d", tPrefix.size());
  const std::size_t fPos = dPos == std::string::npos
                               ? std::string::npos
                               : base.find("-f", dPos + 2);
  const std::size_t stepPos = fPos == std::string::npos
                                  ? std::string::npos
                                  : base.find(stepMarker, fPos + 2);
  if (dPos == std::string::npos || fPos == std::string::npos ||
      stepPos == std::string::npos)
    return false;
  std::uint32_t tokens = 0;
  std::uint32_t dimension = 0;
  std::uint32_t feedForwardDimension = 0;
  if (!parseCheckpointDigits(
          base.substr(tPrefix.size(), dPos - tPrefix.size()), &tokens) ||
      !parseCheckpointDigits(base.substr(dPos + 2, fPos - dPos - 2),
                             &dimension) ||
      !parseCheckpointDigits(base.substr(fPos + 2, stepPos - fPos - 2),
                             &feedForwardDimension))
    return false;
  if (tokens != expectedTokens || dimension != expectedDimension ||
      feedForwardDimension != expectedFeedForwardDimension)
    return false;
  return parseCheckpointDigits(
      base.substr(stepPos + stepMarker.size(),
                  base.size() - stepPos - stepMarker.size() - suffix.size()),
      step);
}

}  // namespace phonelm::nicopedia_checkpoint
