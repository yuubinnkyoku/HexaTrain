// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
#pragma once

#include <cstdint>

namespace phonelm::nicopedia_checkpoint {

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

}  // namespace phonelm::nicopedia_checkpoint
