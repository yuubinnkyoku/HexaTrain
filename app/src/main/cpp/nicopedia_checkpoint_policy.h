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

// A user stop is accepted only at a completed optimizer-step boundary.  The
// last finite V2 state may be committed so an explicit Resume can continue
// from the canonical next step, but an in-flight partial update is never
// serialized as a checkpoint.
inline bool interruptedWriteAllowed(bool allStepsFinite, bool finalStateFinite,
                                    std::uint32_t lastCompletedStep,
                                    std::uint32_t resumeStep) {
  return allStepsFinite && finalStateFinite && lastCompletedStep > resumeStep;
}

}  // namespace phonelm::nicopedia_checkpoint
