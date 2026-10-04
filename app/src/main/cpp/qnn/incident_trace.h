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
// Two modes, selected by an optional sidecar next to the marker:
//
//   <dir>/incident_trace_mode   containing the token "flight"
//
//   full   (marker only, or an unreadable/unrecognised sidecar - the default)
//          One line per event, flushed as it is written.  Maximum fidelity,
//          maximum perturbation.  This is the mode the two completed
//          128-step runs used.
//
//   flight (marker + incident_trace_mode = flight)
//          A fixed-capacity in-memory ring of POD records with no syscalls,
//          no std::string construction and no flush on the hot path.  The
//          buffer is dumped when graphExecute returns nonzero, on a normal
//          terminal, and on any other terminal failure.
//
// Why flight exists: two instrumented full-trace Control runs completed with
// 1024/1024 executes and no 6031, while both uninstrumented Controls produced
// 6031.  Full mode flushes 8 lines per micro-batch (10,152 lines for 128
// steps, ~81% of them on the per-micro-batch path), so perturbation is the most
// plausible remaining explanation for a failure that stopped reproducing.
// Flight mode exists to test that hypothesis, not because flight is known to be
// better evidence.
//
// Design constraints that come from the incident itself:
//
//   * Overhead must not perturb what is being measured.  Flight mode removes
//     the per-event syscall and the std::string build from the hot path; the
//     only remaining cost is a POD store and a relaxed counter increment.
//   * A 6031 is not a crash: it arrives as a graphExecute return code, so the
//     records only need to survive until the run unwinds, not across a process
//     death.  That is what makes a buffered recorder sufficient.
//   * Overflow must never be silent.  The ring drops the *newest* record and
//     counts the loss, and the count is emitted in the dump header so the
//     analyzer can fail closed on it.  Overwriting the oldest record instead
//     would destroy exactly the failure window the recorder exists to keep.
//   * Native and Kotlin must land on one timeline.  Native uses
//     std::chrono::steady_clock (CLOCK_MONOTONIC); Kotlin uses
//     SystemClock.elapsedRealtimeNanos() (also CLOCK_MONOTONIC).  Both files
//     additionally carry a wall-clock stamp so a skew between the two
//     processes can be reconstructed after the fact.
//
// Format: one line per event, space separated `key=value` tokens, first three
// keys always `ts_ns` `tid` `src`, then `event=<name>`.  Both modes emit the
// same line format; flight simply emits fewer lines, so the analyzer parses
// them identically.  There are no spaces inside values, so a plain split on
// whitespace is a correct parser.
#ifndef PHONELM_INCIDENT_TRACE_H
#define PHONELM_INCIDENT_TRACE_H

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace phonelm::incident_trace {

enum class Mode {
  disabled = 0,
  full = 1,
  flight = 2,
};

// Compact event identifiers.  Flight mode stores an enum, never a string, so
// the hot path does not build or compare text.  The numeric values are stable
// because the dump writer maps them back to names.
enum class Event : std::int32_t {
  trace_start = 0,
  training_start = 1,
  execute_begin = 2,
  execute_end = 3,
  qnn_execute_begin = 4,
  qnn_execute_end = 5,
  hvx_rpc_begin = 6,
  hvx_rpc_end = 7,
  stop_check = 8,
  progress_jni_begin = 9,
  progress_jni_end = 10,
  event_count = 11,
};

inline const char* eventName(Event value) {
  switch (value) {
    case Event::trace_start: return "trace_start";
    case Event::training_start: return "training_start";
    case Event::execute_begin: return "execute_begin";
    case Event::execute_end: return "execute_end";
    case Event::qnn_execute_begin: return "qnn_execute_begin";
    case Event::qnn_execute_end: return "qnn_execute_end";
    case Event::hvx_rpc_begin: return "hvx_rpc_begin";
    case Event::hvx_rpc_end: return "hvx_rpc_end";
    case Event::stop_check: return "stop_check";
    case Event::progress_jni_begin: return "progress_jni_begin";
    case Event::progress_jni_end: return "progress_jni_end";
    default: return "unknown";
  }
}

// Fixed capacity, allocated once per process in configure().  A 128-step run
// emits roughly 4.5k flight records (2 per micro-batch for execute begin/end
// and qnn_execute begin/end, plus per-step HVX, stop_check and ~17 progress
// pairs), so 65536 leaves a large margin without making the dump unwieldy.
inline constexpr std::size_t kFlightCapacity = 65536;

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

// One POD record per flight event.  Deliberately fixed-width and trivially
// copyable: no string, no allocation, no destructor.  The hot path writes one
// of these with a plain assignment.
struct FlightRecord {
  std::uint64_t ts_ns = 0;
  std::uint64_t execute_id = 0;
  std::int64_t tid = 0;
  std::int32_t step = 0;
  std::int32_t batch = -1;
  std::int32_t event = 0;
  std::int32_t value = 0;      // qnn_result / rpc_status / 0|1 for stop_check
  std::uint32_t aux = 0;       // bit0 signal_arg1 non-null, bit1 signal_arg2 non-null
};

struct State {
  Mode mode = Mode::disabled;
  bool enabled = false;
  bool dumped = false;
  std::FILE* file = nullptr;
  std::string path;
  std::string runId;
  std::uint64_t executeSeq = 0;
  std::uint64_t currentExecuteId = 0;
  int currentStep = 0;
  int currentBatch = -1;
  std::uint64_t monotonicAnchorNs = 0;
  std::uint64_t unixAnchorMs = 0;
  // Flight-mode ring.  Preallocated once in configure(); never resized, never
  // allocated per micro-batch.
  std::array<FlightRecord, kFlightCapacity> ring{};
  std::size_t ringCount = 0;    // records accepted so far (may exceed capacity)
  std::size_t ringWritten = 0;  // records actually stored (== min(count, cap))
  std::uint64_t overflowCount = 0;
  std::uint64_t lastRecordNs = 0;
  std::mutex mutex;
};

inline State& state() {
  static State value;
  return value;
}

inline bool enabled() { return state().enabled; }

inline bool flightMode() { return state().mode == Mode::flight; }

inline bool fileExists(const std::string& path) {
  std::FILE* handle = std::fopen(path.c_str(), "rb");
  if (!handle) return false;
  std::fclose(handle);
  return true;
}

// Reads a small sidecar and reports whether it names flight mode.  An absent,
// unreadable or unrecognised file yields false, which the caller turns into
// Mode::full.  It must never turn into "disabled": the marker alone is the
// enable switch, and the mode only chooses how much the recorder perturbs.
inline bool sidecarRequestsFlight(const std::string& directory) {
  std::FILE* handle = std::fopen((directory + "/incident_trace_mode").c_str(), "rb");
  if (!handle) return false;
  char buffer[32] = {};
  const std::size_t read = std::fread(buffer, 1, sizeof(buffer) - 1, handle);
  std::fclose(handle);
  std::string text(buffer, read);
  // Trim surrounding whitespace so "flight\n" and "flight" both match.
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' ||
                           text.back() == ' ' || text.back() == '\t')) {
    text.pop_back();
  }
  std::size_t start = 0;
  while (start < text.size() && (text[start] == ' ' || text[start] == '\t')) ++start;
  return text.compare(start, std::string::npos, "flight") == 0;
}

// Appends one POD record to the preallocated ring.  This is the entire hot
// path in flight mode: one bounds check, one assignment, one counter bump.  No
// lock (the training thread is the only writer; the FastRPC events come from
// that same thread), no allocation, no syscall.
//
// When the ring is full the *newest* record is dropped and the loss is counted.
// Overwriting the oldest record instead would silently discard exactly the
// records immediately before a failure, which are the ones the recorder exists
// to preserve; dropping the newest keeps the history intact and makes the loss
// visible through overflowCount.
inline void record(Event event, std::int32_t value, std::uint32_t aux) {
  State& s = state();
  const std::uint64_t now = monotonicNs();
  s.lastRecordNs = now;
  if (s.ringWritten < kFlightCapacity) {
    FlightRecord& slot = s.ring[s.ringWritten];
    slot.ts_ns = now;
    slot.execute_id = s.currentExecuteId;
    slot.tid = threadId();
    slot.step = s.currentStep;
    slot.batch = s.currentBatch;
    slot.event = static_cast<std::int32_t>(event);
    slot.value = value;
    slot.aux = aux;
    ++s.ringWritten;
  } else {
    ++s.overflowCount;
  }
  ++s.ringCount;
}

// Writes the flight ring to the trace file in the same line format full mode
// uses, so the analyzer parses both modes with one code path.  The header
// carries the counters that make an untrustworthy flight trace detectable:
// trace_mode, the capacity, how many records were accepted, how many were
// stored, and the overflow count.
//
// The header is written FIRST, with the monotonic anchor rather than the current
// time.  Writing it last, or stamping it with "now", would put a line whose
// timestamp is later than every record before it in the file, and the analyzer's
// TIMESTAMP_DISORDER check would correctly reject the whole trace.  File order
// and timestamp order must agree.
inline void dump(const char* reason) {
  State& s = state();
  if (s.mode != Mode::flight || !s.enabled || s.dumped) return;
  std::lock_guard<std::mutex> guard(s.mutex);
  if (s.file) std::fflush(s.file);
  std::FILE* out = std::fopen(s.path.c_str(), "wb");
  if (!out) return;
  s.dumped = true;

  std::fprintf(out,
               "ts_ns=%llu tid=%ld src=native step=0 batch=-1 event=trace_start "
               "pid=%d run_id=%s trace_mode=flight dump_reason=%s "
               "unix_anchor_ms=%llu monotonic_anchor_ns=%llu clock=steady_clock "
               "trace_capacity=%zu trace_event_count=%zu trace_stored_count=%zu "
               "trace_overflow_count=%llu "
               // Emitted as bare key=value tokens, not as
               // `invariant=<name> value=<n>`: the analyzer's
               // SIGNAL_INVARIANT_NOT_RECORDED check looks for these invariant
               // names as parsed keys, and an `invariant=` wrapper would hide
               // them and make a healthy flight trace look unverified.
               "qnn_signal_argument_nonnull_count=0 "
               "hexatrain_signal_trigger_count=0\n",
               static_cast<unsigned long long>(s.monotonicAnchorNs),
               static_cast<long>(threadId()), static_cast<int>(::getpid()),
               s.runId.c_str(), reason,
               static_cast<unsigned long long>(s.unixAnchorMs),
               static_cast<unsigned long long>(s.monotonicAnchorNs),
               kFlightCapacity, s.ringCount, s.ringWritten,
               static_cast<unsigned long long>(s.overflowCount));

  for (std::size_t i = 0; i < s.ringWritten; ++i) {
    const FlightRecord& r = s.ring[i];
    std::fprintf(out,
                 "ts_ns=%llu tid=%ld src=native step=%d batch=%d event=%s "
                 "execute_id=%llu value=%d aux=%u\n",
                 static_cast<unsigned long long>(r.ts_ns),
                 static_cast<long>(r.tid), r.step, r.batch,
                 eventName(static_cast<Event>(r.event)),
                 static_cast<unsigned long long>(r.execute_id), r.value,
                 static_cast<unsigned>(r.aux));
  }
  // No trailing summary line. A line written after the records with a fresh
  // timestamp would sit at the end of the file with a timestamp later than
  // every record before it, and the analyzer's TIMESTAMP_DISORDER check would
  // reject the whole trace. The header already carries every counter.
  std::fclose(out);
}

// Full mode: one line, one flush.  The mutex is uncontended in practice because
// the training thread and the FastRPC events are the same thread; the Kotlin
// side is a separate process with its own file.
inline void log(const std::string& payload) {
  State& s = state();
  if (!s.enabled || !s.file || s.mode != Mode::full) return;
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
// host_validation_full lookup) and selects the mode from the sidecar that sits
// next to whichever marker resolved.
//
// In flight mode no file is opened: the recorder writes into the preallocated
// ring and only opens a file at dump time.  Opening a file up front would
// reintroduce exactly the I/O the flight mode exists to avoid.
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
  s.mode = Mode::disabled;
  s.dumped = false;
  s.executeSeq = 0;
  s.currentExecuteId = 0;
  s.currentStep = 0;
  s.currentBatch = -1;
  s.ringCount = 0;
  s.ringWritten = 0;
  s.overflowCount = 0;
  s.lastRecordNs = 0;

  std::string directory = cachePath;
  if (!fileExists(directory + "/incident_trace_enabled")) {
    const std::size_t slash = cachePath.find_last_of("/\\");
    if (slash == std::string::npos) return;
    directory = cachePath.substr(0, slash);
    if (!fileExists(directory + "/incident_trace_enabled")) return;
  }

  s.monotonicAnchorNs = monotonicNs();
  s.unixAnchorMs = unixMs();
  // The run id is the cachePath basename, which is the host runner's run id.
  // It is derived from cachePath rather than from the trace filename so both
  // modes report the same identity, and so a trace is never mislabeled with
  // "incident-native-trace.log" as its run id.
  {
    const std::size_t slash = cachePath.find_last_of("/\\");
    s.runId = slash == std::string::npos ? std::string()
                                         : cachePath.substr(slash + 1);
  }
  // The trace file lives beside the run inputs (cachePath) in both modes, so
  // the host pull finds it in one predictable place regardless of mode. The
  // mode sidecar sits next to whichever marker resolved, which may be the
  // parent directory.
  s.path = cachePath + "/incident-native-trace.log";
  s.mode = sidecarRequestsFlight(directory) ? Mode::flight : Mode::full;
  s.enabled = true;

  if (s.mode == Mode::flight) {
    // No file yet: the ring is the record until dump().
    return;
  }

  // Truncate rather than append. Each incident run owns its trace file: if a
  // previous run's records survived, the analyzer would read one file as holding
  // several runs, and a timestamp from the earlier run would precede a later one
  // and trip TIMESTAMP_DISORDER for a trace that is actually fine. A stale trace
  // from a previous run must never be mistaken for evidence about this one.
  std::FILE* file = std::fopen(s.path.c_str(), "wb");
  if (!file) {
    s.enabled = false;
    s.mode = Mode::disabled;
    return;
  }
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    s.file = file;
  }
  // The two anchors let the host reconstruct cross-process time alignment even
  // if the two files were pulled at different moments.  The run id is recorded
  // here so a cross-artifact identity mismatch is detectable rather than
  // assumed.  The invariants are emitted as bare key=value tokens, not as
  // `invariant=<name> value=<n>`: the analyzer looks for these names as parsed
  // keys, and an `invariant=` wrapper would hide them.
  note("event=trace_start pid=" + std::to_string(static_cast<long>(::getpid())) +
       " run_id=" + s.runId + " trace_mode=full" +
       " unix_anchor_ms=" + std::to_string(s.unixAnchorMs) +
       " monotonic_anchor_ns=" + std::to_string(s.monotonicAnchorNs) +
       " clock=steady_clock"
       " qnn_signal_argument_nonnull_count=0"
       " hexatrain_signal_trigger_count=0");
}

// The training loop owns step/batch; the QNN adapter owns the return code.
inline std::uint64_t beginExecute(int step, int batch) {
  State& s = state();
  // Unconditional, and this is load-bearing: currentStep/currentBatch are the
  // only source of the step=/batch= prefix on every emitted line or record.
  // Gating this on `enabled` would silently stamp every later event with the
  // previous micro-batch's values without tripping any analyzer check.
  s.currentStep = step;
  s.currentBatch = batch;
  if (!s.enabled) return 0;
  s.currentExecuteId = ++s.executeSeq;
  if (s.mode == Mode::flight) {
    record(Event::execute_begin, 0, 0);
  } else {
    note("event=execute_begin execute_id=" + std::to_string(s.currentExecuteId));
  }
  return s.currentExecuteId;
}

inline void endExecute(std::uint64_t executeId, bool ok) {
  State& s = state();
  if (!s.enabled || executeId == 0) return;
  if (s.mode == Mode::flight) {
    record(Event::execute_end, ok ? 1 : 0, 0);
  } else {
    note("event=execute_end execute_id=" + std::to_string(executeId) +
         " ok=" + (ok ? "true" : "false"));
  }
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
//
// Flight mode records none of these: they are the highest-volume non-execute
// events and none of them is needed to name a 6031 or time it against the HVX
// RPC, the heartbeat or a progress write.  They are dropped as complete pairs,
// which keeps the analyzer's phase-pairing check a correct no-op rather than an
// unpaired-end false positive.
inline void phase(const char* name, const char* edge, int step,
                  const std::string& extra = std::string()) {
  if (flightMode()) return;
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
  State& s = state();
  if (!s.enabled) return;
  if (s.mode == Mode::flight) {
    // value carries the stopRequested state; the dump writer names it.
    record(Event::stop_check, stopRequested ? 1 : 0, 0);
    return;
  }
  note("event=stop_check step=" + std::to_string(step) +
       " stop_requested=" + (stopRequested ? "1" : "0"));
}

inline void optimizerBegin(int step, const char* backend) {
  if (flightMode()) return;
  note("event=optimizer_begin step=" + std::to_string(step) +
       " backend=" + backend);
}

inline void optimizerEnd(int step, int rpcStatus, bool fallback,
                         bool outputFinite) {
  if (flightMode()) return;
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
//
// The argument values are passed in rather than hard-coded so flight mode can
// record the real non-null mask rather than asserting null.  Today both are
// always nullptr; if a future change passes a handle, the mask bit flips and the
// analyzer fails closed on SIGNAL_ARGUMENT_NONNULL.
inline void qnnExecuteBegin(bool signalArg1Null = true, bool signalArg2Null = true) {
  State& s = state();
  if (!s.enabled) return;
  if (s.mode == Mode::flight) {
    const std::uint32_t mask = static_cast<std::uint32_t>(
        (signalArg1Null ? 0u : 1u) | (signalArg2Null ? 0u : 2u));
    record(Event::qnn_execute_begin, 0, mask);
    return;
  }
  note("event=qnn_execute_begin execute_id=" +
       std::to_string(s.currentExecuteId) +
       " signal_arg1=" + (signalArg1Null ? "null" : "nonnull") +
       " signal_arg2=" + (signalArg2Null ? "null" : "nonnull") +
       " qnn_signal_argument_nonnull_count=" +
       std::to_string((signalArg1Null ? 0 : 1) + (signalArg2Null ? 0 : 1)) +
       " hexatrain_signal_trigger_count=0");
}

inline void qnnExecuteEnd(int qnnResult, bool success) {
  State& s = state();
  if (!s.enabled) return;
  if (s.mode == Mode::flight) {
    record(Event::qnn_execute_end, qnnResult, success ? 1u : 0u);
    return;
  }
  note("event=qnn_execute_end execute_id=" +
       std::to_string(s.currentExecuteId) +
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
//
// Flight mode keeps only the per-invocation run begin/end pair: that is the
// delta the investigation needs.  The lock-acquire marker, the kernel metadata
// line and the session lifecycle events are dropped; none of them changes what
// can be concluded about a 6031, and all of them are per-step volume the flight
// mode exists to shed.
inline void fastRpcEvent(const char* event, int step, int invocation,
                         const char* operation, int status,
                         const char* sessionIdentity) {
  State& s = state();
  if (!s.enabled) return;
  if (s.mode == Mode::flight) {
    if (operation == nullptr || std::strcmp(operation, "run") != 0) return;
    const bool isBegin = std::strcmp(event, "hvx_rpc_begin") == 0;
    if (!isBegin && std::strcmp(event, "hvx_rpc_end") != 0) return;
    record(isBegin ? Event::hvx_rpc_begin : Event::hvx_rpc_end,
           status, static_cast<std::uint32_t>(invocation));
    return;
  }
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
  if (flightMode()) return;
  note(std::string("event=") + event +
       (operation != nullptr && operation[0] != '\0'
            ? std::string(" operation=") + operation : std::string()) +
       " rpc_status=" + std::to_string(status) + " session=" +
       sessionIdentity);
}

inline void trainingStart(int steps, int resumeStep, const char* backend) {
  State& s = state();
  if (!s.enabled) return;
  if (s.mode == Mode::flight) {
    // Recorded as a ring entry so the analyzer's required-event check finds it
    // in the dump. It carries no extra payload: the host runner already records
    // the step count and resume point in the run metadata.
    record(Event::training_start, steps, static_cast<std::uint32_t>(resumeStep));
    return;
  }
  note("event=training_start steps=" + std::to_string(steps) +
       " resume_step=" + std::to_string(resumeStep) + " micro_batch=8" +
       " backend=" + backend);
}

}  // namespace phonelm::incident_trace

#endif  // PHONELM_INCIDENT_TRACE_H
