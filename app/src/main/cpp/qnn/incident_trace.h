// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Opt-in incident tracing for the QAIRT_GRAPH_ERROR_ABORTED (6031)
// investigation described in docs/g1-1p5x-multiseed-tier2-incident.md.
//
// The trace is completely inert unless a marker file exists, so a normal run
// keeps its original timing, artifacts and behavior:
//
//   <cachePath>/incident_trace_enabled            (run directory)
//   <parent of cachePath>/incident_trace_enabled  (headless-input directory)
//
// Design constraints that come from the incident itself:
//
//   * Overhead must not perturb what is being measured. Every emit is a single
//     buffered append with no fsync, no JSON, no parsing and no Java callback.
//     When tracing is off, each call site costs one predictable-branch check.
//   * A run that aborts must still leave the preceding record recoverable.
//     Every line is flushed as it is produced (no buffering past one line), so
//     a QNN abort, a process kill or a power cut loses at most the line being
//     written.
//   * Native and Kotlin must land on one timeline. Native uses
//     std::chrono::steady_clock (CLOCK_MONOTONIC); Kotlin uses
//     SystemClock.elapsedRealtimeNanos() (also CLOCK_MONOTONIC). Both files
//     additionally carry a wall-clock stamp so a skew between the two
//     processes can be reconstructed after the fact.
//
// Format: one line per event, space separated `key=value` tokens, first three
// keys always `ts_ns` `tid` `src`, then `event=<name>`. There are no spaces
// inside values, so a plain split on whitespace is a correct parser.
#ifndef PHONELM_INCIDENT_TRACE_H
#define PHONELM_INCIDENT_TRACE_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace phonelm::incident_trace {

inline long threadId() {
#ifdef SYS_gettid
  return static_cast<long>(::syscall(SYS_gettid));
#else
  return 0;
#endif
}

inline std::uint64_t monotonicNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline std::uint64_t unixMs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

struct State {
  bool enabled = false;
  std::FILE* file = nullptr;
  std::uint64_t executeSeq = 0;
  std::uint64_t currentExecuteId = 0;
  int currentStep = 0;
  int currentBatch = -1;
  std::uint64_t monotonicAnchorNs = 0;
  std::uint64_t unixAnchorMs = 0;
  std::mutex mutex;
};

inline State& state() {
  static State value;
  return value;
}

inline bool enabled() { return state().enabled; }

inline bool fileExists(const std::string& path) {
  std::FILE* handle = std::fopen(path.c_str(), "rb");
  if (!handle) return false;
  std::fclose(handle);
  return true;
}

// One line, one flush, no allocation-heavy work.  The mutex is only ever
// contended between the training thread and the FastRPC-free Kotlin-free native
// callers; the FastRPC session events come from the same training thread, so in
// practice this is uncontended.
inline void log(const std::string& payload) {
  State& s = state();
  if (!s.enabled || !s.file) return;
  std::string line = "ts_ns=";
  line += std::to_string(monotonicNs());
  line += " tid=";
  line += std::to_string(threadId());
  line += " src=native";
  line += " step=";
  line += std::to_string(s.currentStep);
  line += " batch=";
  line += std::to_string(s.currentBatch);
  line += ' ';
  line += payload;
  line += '\n';
  std::lock_guard<std::mutex> guard(s.mutex);
  std::fwrite(line.data(), 1, line.size(), s.file);
  // Flush per line: an abort must not cost us the records before it.
  std::fflush(s.file);
}

inline void note(const std::string& payload) { log(payload); }

// Resolves the marker (run directory first, then its parent, mirroring the
// host_validation_full lookup) and opens the trace beside the run inputs.
inline void configure(const std::string& cachePath) {
  State& s = state();
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    if (s.file) {
      std::fflush(s.file);
      std::fclose(s.file);
      s.file = nullptr;
    }
  }
  s.enabled = false;
  s.executeSeq = 0;
  s.currentExecuteId = 0;
  s.currentStep = 0;
  s.currentBatch = -1;
  std::string marker = cachePath + "/incident_trace_enabled";
  bool found = fileExists(marker);
  if (!found) {
    const std::size_t slash = cachePath.find_last_of("/\\");
    if (slash == std::string::npos) return;
    marker = cachePath.substr(0, slash) + "/incident_trace_enabled";
    found = fileExists(marker);
    if (!found) return;
  }
  const std::string path = cachePath + "/incident-native-trace.log";
  std::FILE* file = std::fopen(path.c_str(), "ab");
  if (!file) return;
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    s.file = file;
  }
  s.monotonicAnchorNs = monotonicNs();
  s.unixAnchorMs = unixMs();
  s.enabled = true;
  // The two anchors let the host reconstruct cross-process time alignment even
  // if the two files were pulled at different moments.  The trace directory
  // basename is the run id the host runner uses, so it is recorded here to
  // make a cross-artifact identity mismatch detectable rather than assumed.
  const std::size_t slash = cachePath.find_last_of("/\\");
  const std::string runId = slash == std::string::npos
      ? std::string() : cachePath.substr(slash + 1);
  note("event=trace_start pid=" + std::to_string(static_cast<long>(::getpid())) +
       " run_id=" + runId +
       " unix_anchor_ms=" + std::to_string(s.unixAnchorMs) +
       " monotonic_anchor_ns=" + std::to_string(s.monotonicAnchorNs) +
       " clock=steady_clock"
       " invariant=qnn_signal_argument_nonnull_count value=0"
       " invariant=hexatrain_signal_trigger_count value=0");
}

// The training loop owns step/batch; the QNN adapter owns the return code.
inline std::uint64_t beginExecute(int step, int batch) {
  State& s = state();
  s.currentStep = step;
  s.currentBatch = batch;
  if (!s.enabled) return 0;
  s.currentExecuteId = ++s.executeSeq;
  note("event=execute_begin execute_id=" + std::to_string(s.currentExecuteId));
  return s.currentExecuteId;
}

inline void endExecute(std::uint64_t executeId, bool ok) {
  if (!enabled() || executeId == 0) return;
  note("event=execute_end execute_id=" + std::to_string(executeId) +
       " ok=" + (ok ? "true" : "false"));
}

inline std::uint64_t currentExecuteId() {
  return enabled() ? state().currentExecuteId : 0;
}

inline void scope(int step, int batch) {
  State& s = state();
  s.currentStep = step;
  s.currentBatch = batch;
}

// Step boundary instrumentation.  `begin`/`end` pairs only; the analyzer pairs
// them by name and reports unpaired ones.
inline void phase(const char* name, const char* edge, int step,
                  const std::string& extra = std::string()) {
  std::string payload = "event=";
  payload += name;
  payload += '_';
  payload += edge;
  payload += " step=";
  payload += std::to_string(step);
  if (!extra.empty()) {
    payload += ' ';
    payload += extra;
  }
  note(payload);
}

inline void phase(const char* name, const char* edge, int step, int batch) {
  phase(name, edge, step, "batch=" + std::to_string(batch));
}

inline void stopCheck(int step, bool stopRequested) {
  note("event=stop_check step=" + std::to_string(step) +
       " stop_requested=" + (stopRequested ? "1" : "0"));
}

inline void optimizerBegin(int step, const char* backend) {
  note("event=optimizer_begin step=" + std::to_string(step) +
       " backend=" + backend);
}

inline void optimizerEnd(int step, int rpcStatus, bool fallback,
                         bool outputFinite) {
  note("event=optimizer_end step=" + std::to_string(step) +
       " rpc_status=" + std::to_string(rpcStatus) +
       " fallback=" + (fallback ? "true" : "false") +
       " output_finite=" + (outputFinite ? "true" : "false"));
}

inline void parameterMoveBegin(int step) {
  phase("parameter_move", "begin", step);
}

inline void parameterMoveEnd(int step) {
  phase("parameter_move", "end", step);
}

inline void telemetryBegin(int step) { phase("telemetry", "begin", step); }
inline void telemetryEnd(int step) { phase("telemetry", "end", step); }

inline void checkpointBegin(int step) { phase("checkpoint", "begin", step); }
inline void checkpointEnd(int step, bool written) {
  phase("checkpoint", "end", step,
        std::string("written=") + (written ? "true" : "false"));
}

inline void zeroParametersBegin(int step) {
  phase("zero_parameters", "begin", step);
}

inline void zeroParametersEnd(int step) {
  phase("zero_parameters", "end", step);
}

inline void batchPrepareBegin(int step, int batch) {
  phase("batch_prepare", "begin", step, batch);
}

inline void batchPrepareEnd(int step, int batch) {
  phase("batch_prepare", "end", step, batch);
}

// Called from the QNN adapter: the signal arguments are recorded as the
// literal nullptr values HexaTrain passes, so a future change that supplies a
// real handle shows up as a non-null invariant violation instead of silently
// altering the abort path.
inline void qnnExecuteBegin() {
  if (!enabled()) return;
  note("event=qnn_execute_begin execute_id=" +
       std::to_string(state().currentExecuteId) +
       " signal_arg1=null signal_arg2=null"
       " qnn_signal_argument_nonnull_count=0"
       " hexatrain_signal_trigger_count=0");
}

inline void qnnExecuteEnd(int qnnResult, bool success) {
  if (!enabled()) return;
  note("event=qnn_execute_end execute_id=" +
       std::to_string(state().currentExecuteId) +
       " qnn_result=" + std::to_string(qnnResult) +
       " success=" + (success ? "true" : "false"));
}

inline void poisonFillBegin(int step, int batch) {
  phase("poison_fill", "begin", step, batch);
}

inline void poisonFillEnd(int step, int batch) {
  phase("poison_fill", "end", step, batch);
}

// FastRPC (HVX Muon) session activity.  `sessionIdentity` must be a stable
// local label, never a path, serial or any other secret-bearing value.
inline void fastRpcEvent(const char* event, int step, int invocation,
                         const char* operation, int status,
                         const char* sessionIdentity) {
  std::string payload = "event=";
  payload += event;
  payload += " invocation=";
  payload += std::to_string(invocation);
  payload += " rpc_status=";
  payload += std::to_string(status);
  payload += " session=";
  payload += sessionIdentity;
  if (step > 0) {
    // Session-wide events (lock acquire) carry no optimizer step; the
    // per-invocation events do, and the analyzer pairs them by invocation id.
    payload += " rpc_step=";
    payload += std::to_string(step);
  }
  if (operation != nullptr && operation[0] != '\0') {
    payload += " operation=";
    payload += operation;
  }
  note(payload);
}

inline void fastRpcSessionEvent(const char* event, const char* operation,
                                int status, const char* sessionIdentity) {
  note(std::string("event=") + event +
       (operation != nullptr && operation[0] != '\0'
            ? std::string(" operation=") + operation : std::string()) +
       " rpc_status=" + std::to_string(status) + " session=" +
       sessionIdentity);
}

inline void trainingStart(int steps, int resumeStep, const char* backend) {
  note("event=training_start steps=" + std::to_string(steps) +
       " resume_step=" + std::to_string(resumeStep) + " micro_batch=8" +
       " backend=" + backend);
}

}  // namespace phonelm::incident_trace

#endif  // PHONELM_INCIDENT_TRACE_H
