#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Incident analyzer for QAIRT_GRAPH_ERROR_ABORTED (6031).

Investigation record: docs/g1-1p5x-multiseed-tier2-incident.md

This analyzer answers one question and refuses to answer any other: *what was
observed around the failing graphExecute?*  It never claims that proximity is
causation.  Every "nearest event" it prints is labelled as an observation, and
the report carries an explicit statement that correlation is not attribution.

Two trace modes are parsed by the same code path, because both emit the same
`key=value` line format and differ only in how many lines they emit:

  full    every event, flushed as it is written (maximum perturbation)
  flight  a minimal event set, buffered in memory and dumped at a terminal or
          at a nonzero graphExecute (the mode used to test whether the
          instrumentation itself was suppressing the abort)

`trace_mode` is read from the trace itself rather than assumed from the
filename, so a mode can never be misreported. A flight trace that lost records
raises TRACE_OVERFLOW: "no failure observed" from a trace with holes in it is
not evidence of absence.

Inputs (all optional except the native trace in incident mode):

  --native   incident-native-trace.log   (training thread; CLOCK_MONOTONIC)
  --kotlin   incident-kotlin-trace.log   (heartbeat / progress / state writes)
  --report   <runId>-result.txt          (device primary report)
  --status   status.json                 (device run identity)
  --logcat   logcat.txt                  (raw device log, not filtered)
  --hostlog  host runner log
  --legacy   a pre-instrumentation result.txt, analyzed as legacy evidence

Outputs:

  <out>/incident-timeline.csv    one row per event, merged and sorted
  <out>/incident-findings.json   machine-readable findings + problems
  <out>/incident-report.md       human-readable timeline and deltas

Fail-closed rules are in `check_invariants`; any of them produces a non-empty
`problems` list and a non-zero exit, because a trace we cannot trust must not be
read as an absence of evidence.

Modes:

  --selftest   run the synthetic fixture battery (no device, no network)
  --legacy DIR analyze a directory of pre-instrumentation evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ABORTED_QNN_RESULT = 6031

# Micro-batches per training step in the L19 Muon loop.  This is a structural
# constant of that loop (qnn_transformer_training.cpp), not a tunable: one
# graphExecute per micro-batch.  Legacy step/batch arithmetic uses it to turn a
# recorded execute index into a step and batch.
LEGACY_MICRO_BATCH = 8

# Event names the instrumentation guarantees.  Anything else is unexpected and
# is reported rather than silently ignored.
NATIVE_REQUIRED_EVENTS = ("trace_start", "training_start")
KOTLIN_REQUIRED_EVENTS = ("trace_start",)

# Flight mode emits a deliberately minimal set: the execute pair (which names a
# failure and carries the QNN return code), the HVX RPC pair (the delta the
# investigation needs), stop_check (the stopRequested state), the progress pair,
# and the two anchors.  It emits no step phases at all, which is why the phase
# pairing check is a correct no-op for a flight trace rather than a source of
# unpaired-end false positives.
FLIGHT_REQUIRED_EVENTS = ("trace_start", "execute_begin", "execute_end",
                          "qnn_execute_begin", "qnn_execute_end")

# Step-boundary phases that must appear as a begin/end pair per training step.
# Flight mode emits none of these, so pairing checks are skipped there; full
# mode emits them as complete pairs.
STEP_PHASES = (
    "zero_parameters",
    "optimizer",
    "parameter_move",
    "telemetry",
    "checkpoint",
)

MODE_FULL = "full"
MODE_FLIGHT = "flight"

# Fields that make a trace line parseable.  Values never contain whitespace,
# so a split on whitespace is exact.
TOKEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# An execute is "failed" if the QNN return code is nonzero.
FAILURE_RESULT_MIN = 1


class ParseProblem(Exception):
    pass


def parse_trace(path: Path) -> list[dict[str, Any]]:
    """Parse a trace file into event dicts, preserving file order.

    File order is meaningful: each emit is an append + flush, so a record that
    appears later with an *earlier* timestamp means either the clock or the
    writer is not behaving.  The merged timeline sorts later, but the
    monotonicity check below runs on this preserved order.

    Raises ParseProblem only for structural corruption (a line with no
    `event=` token, or no `ts_ns`).  Individual unparsable fields are kept as
    strings so the analyzer can report them instead of losing the record.
    """
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            record: dict[str, Any] = {"_lineno": lineno, "_raw": line}
            for token in line.split():
                match = TOKEN_RE.match(token)
                if match:
                    record[match.group(1)] = match.group(2)
                else:
                    record.setdefault("_unparsed", []).append(token)
            if "event" not in record:
                raise ParseProblem(f"{path.name}:{lineno}: no event= token")
            if "ts_ns" not in record:
                raise ParseProblem(f"{path.name}:{lineno}: no ts_ns token")
            events.append(record)
    return events


def as_int(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def detect_mode(timeline: "Timeline") -> str:
    """Read trace_mode from the trace itself, defaulting to full.

    Reading it from the data rather than the filename means a mode can never be
    reported as something the recorder did not actually use.  A trace with no
    trace_mode token predates flight mode, which is full by definition.
    """
    for event in timeline.events:
        mode = event.get("trace_mode")
        if mode in (MODE_FLIGHT, MODE_FULL):
            return mode
    return MODE_FULL


def trace_counters(timeline: "Timeline") -> dict[str, int | None]:
    """Collect the flight-mode capacity/overflow counters from the trace.

    `trace_overflow_count` is the field that makes a sparse flight trace
    trustworthy or not: a nonzero value means records were dropped, so the
    trace has holes and its silence is not evidence.
    """
    overflow: int | None = None
    events_seen: int | None = None
    bytes_hint: int | None = None
    for event in timeline.events:
        for key in ("trace_overflow_count", "kotlin_trace_overflow_count"):
            value = as_int(event, key)
            if value is not None:
                overflow = value if overflow is None else max(overflow, value)
        for key in ("trace_event_count", "kotlin_trace_event_count"):
            value = as_int(event, key)
            if value is not None:
                events_seen = value if events_seen is None else max(events_seen, value)
        for key in ("trace_bytes", "kotlin_trace_bytes"):
            value = as_int(event, key)
            if value is not None:
                bytes_hint = value if bytes_hint is None else max(bytes_hint, value)
    return {
        "trace_overflow_count": overflow,
        "trace_event_count": events_seen,
        "trace_bytes": bytes_hint,
    }


def load_kv_report(path: Path) -> dict[str, str]:
    """Parse a device primary report (`key=value` per line).

    The report repeats some keys (generalized_tiny_training_qnn_return appears
    once per execute in the incident case), so repeated keys are collected into
    a list under `<key>__all` while the last value wins for `<key>`.
    """
    result: dict[str, str] = {}
    repeated: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        repeated.setdefault(key, []).append(value)
        result[key] = value
    for key, values in repeated.items():
        if len(values) > 1:
            result[f"{key}__all"] = values
    return result


class Timeline:
    """Merged native + kotlin event list with the delta helpers the report needs.

    `events` is sorted by timestamp for timeline arithmetic.  `native_in_order`
    and `kotlin_in_order` keep each file's original write order, because the
    merged sort destroys the only evidence that a clock or writer went
    backwards.
    """

    def __init__(self, events: list[dict[str, Any]],
                 native_in_order: list[dict[str, Any]] | None = None,
                 kotlin_in_order: list[dict[str, Any]] | None = None):
        self.events = events
        self.native_in_order = (
            native_in_order if native_in_order is not None
            else [e for e in events if e.get("src") == "native"])
        self.kotlin_in_order = (
            kotlin_in_order if kotlin_in_order is not None
            else [e for e in events if e.get("src") == "kotlin"])

    def failures(self) -> list[dict[str, Any]]:
        """Failing executes, identified from either mode's encoding.

        Full mode writes `qnn_result=N`; flight mode packs the same number into
        the POD record's `value` field. Both are read here so one code path
        serves both modes.
        """
        out = []
        for event in self.events:
            if event.get("event") != "qnn_execute_end":
                continue
            result = as_int(event, "qnn_result")
            if result is None:
                result = as_int(event, "value")
            if result is not None and result >= FAILURE_RESULT_MIN:
                out.append(event)
        return out

    def ends_named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]

    def nearest_before(self, target_ns: int, predicate) -> dict[str, Any] | None:
        """Nearest strictly-earlier event satisfying `predicate`, by timestamp.

        "Nearest" is proximity only.  The caller must present it as an
        observation; the report template labels every such row as such.
        """
        best: dict[str, Any] | None = None
        best_delta: int | None = None
        for event in self.events:
            ts = as_int(event, "ts_ns")
            if ts is None or ts >= target_ns:
                continue
            if not predicate(event):
                continue
            delta = target_ns - ts
            if best_delta is None or delta < best_delta:
                best, best_delta = event, delta
        return best

    def window(self, target_ns: int, before_ns: int, after_ns: int,
               nearest: int = 12) -> list[dict[str, Any]]:
        """Events within the window, trimmed to the `nearest` closest on each side.

        Trimming keeps the report about the failure's immediate neighbourhood
        instead of dumping an entire training run around it.
        """
        before: list[dict[str, Any]] = []
        after: list[dict[str, Any]] = []
        for event in self.events:
            ts = as_int(event, "ts_ns")
            if ts is None:
                continue
            if target_ns - before_ns <= ts <= target_ns + after_ns:
                if ts <= target_ns:
                    before.append(event)
                else:
                    after.append(event)
        before.sort(key=lambda e: as_int(e, "ts_ns") or 0, reverse=True)
        after.sort(key=lambda e: as_int(e, "ts_ns") or 0)
        kept = list(reversed(before[:nearest])) + after[:nearest]
        kept.sort(key=lambda e: as_int(e, "ts_ns") or 0)
        return kept


def describe(event: dict[str, Any] | None) -> str:
    if event is None:
        return "(none)"
    parts = [f"event={event.get('event', '?')}"]
    for key in ("src", "thread", "reason", "operation", "session", "step", "batch"):
        if key in event:
            parts.append(f"{key}={event[key]}")
    return " ".join(parts)


def check_invariants(
    timeline: Timeline,
    report: dict[str, str] | None,
    status: dict[str, Any] | None,
    incident_mode: bool,
) -> list[dict[str, str]]:
    """Every condition here is fail-closed: each returns a problem entry.

    The list is intentionally aggressive.  A trace that trips any of these is
    not evidence of absence, it is evidence of an unusable measurement.
    """
    problems: list[dict[str, str]] = []

    def problem(code: str, detail: str) -> None:
        problems.append({"code": code, "detail": detail})

    events = timeline.events
    mode = detect_mode(timeline)
    is_flight = mode == MODE_FLIGHT

    # --- native trace structural requirements -------------------------
    native = [e for e in events if e.get("src") == "native"]
    kotlin = [e for e in events if e.get("src") == "kotlin"]
    if incident_mode and not native:
        problem("NATIVE_TRACE_MISSING",
                "incident mode but no native-sourced events were parsed")
    required_native = FLIGHT_REQUIRED_EVENTS if is_flight else NATIVE_REQUIRED_EVENTS
    for required in required_native:
        if native and not any(e.get("event") == required for e in native):
            problem("NATIVE_EVENT_MISSING",
                    f"required native event {required} absent in {mode} mode")
    for required in KOTLIN_REQUIRED_EVENTS:
        if kotlin and not any(e.get("event") == required for e in kotlin):
            problem("KOTLIN_EVENT_MISSING", f"required kotlin event {required} absent")

    # --- timestamp monotonicity, in per-file write order ----------------
    # The merged timeline is sorted by timestamp, so it cannot detect a clock
    # or writer that went backwards.  Both raw per-file sequences are carried
    # on the Timeline for exactly this check.
    for source, subset in (("native", timeline.native_in_order),
                           ("kotlin", timeline.kotlin_in_order)):
        previous_ts: int | None = None
        previous_line = 0
        for event in subset:
            ts = as_int(event, "ts_ns")
            if ts is None:
                problem("TIMESTAMP_UNPARSABLE",
                        f"{source} line {event.get('_lineno')}: ts_ns not an integer")
                continue
            if previous_ts is not None and ts < previous_ts:
                problem("TIMESTAMP_DISORDER",
                        f"{source} line {event.get('_lineno')}: ts_ns {ts} precedes "
                        f"{previous_ts} from line {previous_line}")
            previous_ts, previous_line = ts, event.get("_lineno", 0)

    # --- execute id integrity -----------------------------------------
    seen_begins: dict[int, int] = {}
    seen_ends: set[int] = set()
    for event in events:
        if event.get("event") != "execute_begin":
            continue
        execute_id = as_int(event, "execute_id")
        if execute_id is None:
            problem("EXECUTE_ID_MISSING",
                    f"line {event.get('_lineno')}: execute_begin without execute_id")
            continue
        if execute_id in seen_begins:
            problem("EXECUTE_ID_DUPLICATE",
                    f"execute_id={execute_id} begins at lines "
                    f"{seen_begins[execute_id]} and {event.get('_lineno')}")
        else:
            seen_begins[execute_id] = event.get("_lineno", 0)
    for event in events:
        if event.get("event") != "execute_end":
            continue
        execute_id = as_int(event, "execute_id")
        if execute_id is None:
            problem("EXECUTE_ID_MISSING",
                    f"line {event.get('_lineno')}: execute_end without execute_id")
            continue
        if execute_id in seen_ends:
            problem("EXECUTE_ID_DUPLICATE",
                    f"execute_id={execute_id} ends more than once")
        seen_ends.add(execute_id)

    failures = timeline.failures()
    failed_ids = set()
    for event in failures:
        execute_id = as_int(event, "execute_id")
        if execute_id is not None:
            failed_ids.add(execute_id)

    # --- begin/end pairing --------------------------------------------
    # A failure execute is allowed to lack an execute_end: the run aborts
    # inside graphExecute, so the end record may not exist.  Every other
    # unpaired begin is a problem.
    for execute_id, lineno in sorted(seen_begins.items()):
        if execute_id in seen_ends:
            continue
        if execute_id in failed_ids:
            continue
        problem("EXECUTE_BEGIN_WITHOUT_END",
                f"execute_id={execute_id} (line {lineno}) has no execute_end "
                f"and is not the failing execute")
    for execute_id in sorted(seen_ends - set(seen_begins)):
        if execute_id in failed_ids:
            continue
        problem("EXECUTE_END_WITHOUT_BEGIN", f"execute_id={execute_id} has no execute_begin")

    # --- step / batch coherence ---------------------------------------
    for event in events:
        if event.get("event") != "execute_begin":
            continue
        step = as_int(event, "step")
        batch = as_int(event, "batch")
        if step is None or step <= 0:
            problem("STEP_MISSING",
                    f"execute_begin (line {event.get('_lineno')}) has step={event.get('step')}")
        if batch is None or not (0 <= batch < LEGACY_MICRO_BATCH):
            problem("BATCH_OUT_OF_RANGE",
                    f"execute_begin (line {event.get('_lineno')}) has "
                    f"batch={event.get('batch')} outside "
                    f"0..{LEGACY_MICRO_BATCH - 1}")

    # --- step phase pairing -------------------------------------------
    # Flight mode emits no step phases at all, so this check is skipped rather
    # than run against a set that is empty by design. Running it there would be
    # harmless (an empty phase list pairs nothing) but stating the intent is
    # clearer than relying on that.
    open_phase: dict[str, int] = {}
    for event in events:
        name = event.get("event", "")
        for phase in (() if is_flight else STEP_PHASES):
            if name == f"{phase}_begin":
                if phase in open_phase:
                    problem("PHASE_REOPENED",
                            f"{phase} begins again at line {event.get('_lineno')} "
                            f"while an earlier begin is still open")
                open_phase[phase] = event.get("_lineno", 0)
            elif name == f"{phase}_end":
                if phase not in open_phase:
                    problem("PHASE_END_WITHOUT_BEGIN",
                            f"{phase}_end at line {event.get('_lineno')} has no begin")
                else:
                    open_phase.pop(phase)
    for phase, lineno in sorted(open_phase.items()):
        # A phase left open at the abort is legitimate only for the failing
        # step; report it as an observation with a problem if it is not there.
        problem("PHASE_BEGIN_WITHOUT_END",
                f"{phase} begins at line {lineno} and never ends "
                f"(expected when the run aborts inside it)")

    # --- signal invariants --------------------------------------------
    # Full mode writes the arguments as `signal_arg1=null signal_arg2=null`.
    # Flight mode writes the same fact as a packed bitmask in `aux`
    # (bit0 = arg1 non-null, bit1 = arg2 non-null) because it cannot afford a
    # string on the hot path. Both encodings are checked here so the invariant
    # holds in either mode rather than being assumed in one of them.
    nonnull_signal = 0
    for event in events:
        for key in ("signal_arg1", "signal_arg2"):
            value = event.get(key)
            if value is not None and value != "null":
                nonnull_signal += 1
                problem("SIGNAL_ARGUMENT_NONNULL",
                        f"line {event.get('_lineno')}: {key}={value}")
        if event.get("event") == "qnn_execute_begin":
            aux = as_int(event, "aux")
            if aux is not None and aux != 0:
                nonnull_signal += bin(aux).count("1")
                problem("SIGNAL_ARGUMENT_NONNULL",
                        f"line {event.get('_lineno')}: flight aux mask={aux} "
                        f"marks a non-null signal argument")
        counted = as_int(event, "qnn_signal_argument_nonnull_count")
        if counted is not None and counted != 0:
            problem("SIGNAL_INVARIANT_VIOLATION",
                    f"line {event.get('_lineno')}: "
                    f"qnn_signal_argument_nonnull_count={counted}")
        triggers = as_int(event, "hexatrain_signal_trigger_count")
        if triggers is not None and triggers != 0:
            problem("SIGNAL_TRIGGER_INVARIANT_VIOLATION",
                    f"line {event.get('_lineno')}: "
                    f"hexatrain_signal_trigger_count={triggers}")
    if nonnull_signal == 0 and native:
        # The invariant must be positively recorded, not merely unviolated.
        # Full mode records it on trace_start and on every qnn_execute_begin;
        # flight mode records it once on the dump header. All three spellings
        # are the bare `name=0` key form, so one check covers both modes.
        if not any("qnn_signal_argument_nonnull_count" in e for e in native):
            problem("SIGNAL_INVARIANT_NOT_RECORDED",
                    "no native event carries qnn_signal_argument_nonnull_count; "
                    "the invariant was not observed, only assumed")

    # --- flight-mode ring overflow -------------------------------------
    # A flight trace that dropped records has holes in it. Its silence is then
    # not evidence of absence, so this is fail-closed rather than a note.
    counters = trace_counters(timeline)
    overflow = counters.get("trace_overflow_count")
    if overflow is not None and overflow != 0:
        problem("TRACE_OVERFLOW",
                f"trace_overflow_count={overflow}: flight-mode records were "
                f"dropped, so this trace cannot support an absence-of-failure "
                f"claim")
    if is_flight and overflow is None:
        problem("TRACE_OVERFLOW_UNKNOWN",
                "flight-mode trace carries no trace_overflow_count, so whether "
                "records were lost cannot be established")

    # --- device identity ----------------------------------------------
    # The native trace_start carries the run id (the trace directory basename,
    # which the host runner sets from the device run id).  A mismatch against
    # status.json means the trace and the device record describe different
    # runs, which invalidates every cross-artifact conclusion.
    trace_run_ids = {
        e.get("run_id") for e in native
        if e.get("event") == "trace_start" and e.get("run_id")}
    status_run_id = status.get("run_id") if status else None
    if trace_run_ids and status_run_id:
        if status_run_id not in trace_run_ids:
            problem("RUN_IDENTITY_MISMATCH",
                    f"native trace run_id={sorted(trace_run_ids)} but "
                    f"status.json run_id={status_run_id}")
    if incident_mode and native and not trace_run_ids:
        problem("RUN_IDENTITY_ABSENT",
                "native trace has no run_id on trace_start; a cross-artifact "
                "identity check is impossible, so this is reported rather than "
                "assumed to match")
    if incident_mode and status and not status_run_id:
        problem("RUN_IDENTITY_ABSENT",
                "status.json carries no run_id; device identity cannot be "
                "cross-checked against the trace")

    return problems


def failure_analysis(timeline: Timeline, window_ns: int = 2_000_000_000,
                     nearest: int = 12) -> dict[str, Any]:
    """For each failing execute, report the observations around it.

    Nothing here is causal.  Each `nearest_*` entry is proximity on a shared
    monotonic clock; the caller must not read it as attribution.  The window
    list is bounded to the `nearest` closest events on each side so a report
    stays readable instead of dumping the whole run.

    The QNN return code is read from `qnn_result` in full mode and from `value`
    in flight mode, because flight mode packs the return code into the POD
    record's value field instead of formatting a string.
    """
    results = []
    for failure in timeline.failures():
        ts = as_int(failure, "ts_ns")
        execute_id = as_int(failure, "execute_id")
        entry: dict[str, Any] = {
            "execute_id": execute_id,
            "step": as_int(failure, "step"),
            "batch": as_int(failure, "batch"),
            # Full mode writes qnn_result=N; flight mode packs the same number
            # into value. failures() already resolved which encoding applies,
            # so read it the same way here.
            "qnn_result": (as_int(failure, "qnn_result")
                           if as_int(failure, "qnn_result") is not None
                           else as_int(failure, "value")),
            "ts_ns": ts,
            "tid": failure.get("tid"),
        }
        if ts is None:
            entry["observations"] = {}
            results.append(entry)
            continue

        hvx_end = timeline.nearest_before(
            ts, lambda e: e.get("event") == "hvx_rpc_end")
        heartbeat_end = timeline.nearest_before(
            ts, lambda e: e.get("event") == "heartbeat_write_end")
        progress_end = timeline.nearest_before(
            ts, lambda e: e.get("event") == "progress_status_write_end")
        progress_jni_end = timeline.nearest_before(
            ts, lambda e: e.get("event") == "progress_jni_end")
        checkpoint_end = timeline.nearest_before(
            ts, lambda e: e.get("event") == "checkpoint_end")
        stop_check = timeline.nearest_before(
            ts, lambda e: e.get("event") == "stop_check")

        # Full mode writes stop_requested=0|1; flight mode packs it into value.
        if stop_check is not None and "stop_requested" not in stop_check:
            packed = as_int(stop_check, "value")
            if packed is not None:
                stop_check = dict(stop_check, stop_requested=str(packed))

        def delta(event: dict[str, Any] | None) -> int | None:
            if event is None:
                return None
            other = as_int(event, "ts_ns")
            return None if other is None else ts - other

        entry["observations"] = {
            "nearest_hvx_rpc_end": {
                "delta_ns": delta(hvx_end), "event": describe(hvx_end)},
            "nearest_heartbeat_write_end": {
                "delta_ns": delta(heartbeat_end), "event": describe(heartbeat_end)},
            "nearest_progress_status_write_end": {
                "delta_ns": delta(progress_end), "event": describe(progress_end)},
            "nearest_progress_jni_end": {
                "delta_ns": delta(progress_jni_end), "event": describe(progress_jni_end)},
            "nearest_checkpoint_end": {
                "delta_ns": delta(checkpoint_end), "event": describe(checkpoint_end)},
            "nearest_stop_check": {
                "delta_ns": delta(stop_check), "event": describe(stop_check),
                "stop_requested": stop_check.get("stop_requested") if stop_check else None},
        }
        entry["window_events"] = [
            describe(e) for e in timeline.window(ts, window_ns, window_ns, nearest)]
        results.append(entry)
    return {"failures": results}


def ms(delta_ns: int | None) -> str:
    if delta_ns is None:
        return "n/a"
    return f"{delta_ns / 1e6:+.3f} ms"


def us(delta_ns: int | None) -> str:
    if delta_ns is None:
        return "n/a"
    return f"{delta_ns / 1e3:+.1f} us"


def render_markdown(findings: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 6031 incident analysis")
    add("")
    add(f"- schema_version: {findings['schema_version']}")
    add(f"- incident_mode: {findings['incident_mode']}")
    add(f"- evidence_class: {findings['evidence_class']}")
    add(f"- trace_mode: {findings.get('trace_mode', 'unknown')}")
    add(f"- native events: {findings['counts']['native']}")
    add(f"- kotlin events: {findings['counts']['kotlin']}")
    add(f"- failing executes: {findings['counts']['failures']}")
    add("")
    add("> All deltas below are *proximity on a shared monotonic clock*, not")
    add("> evidence of causation. `near == cause` is never asserted by this tool.")
    add("")

    overhead = findings.get("trace_overhead") or {}
    if overhead:
        add("## Trace overhead")
        add("")
        add("Diagnostic instrumentation cost only. This is **not** a quality")
        add("measurement and must not be compared against any G1 run.")
        add("")
        add(f"- `trace_event_count` = {overhead.get('trace_event_count')}")
        add(f"- `trace_bytes` = {overhead.get('trace_bytes')}")
        overflow = overhead.get("trace_overflow_count")
        add(f"- `trace_overflow_count` = {overflow}")
        if overflow is not None and overflow == 0:
            add("")
            add("No records were lost, so this trace supports an")
            add("absence-of-failure claim.")
        elif overflow is None:
            add("")
            add("No overflow counter was present; for a flight trace that")
            add("means the claim could not be established.")
        else:
            add("")
            add("Records were lost. This trace has holes and cannot support an")
            add("absence-of-failure claim.")
        add("")

    problems = findings["problems"]
    add("## Fail-closed problems")
    add("")
    if not problems:
        add("None. Every structural and invariant check passed.")
    else:
        add("| code | detail |")
        add("| --- | --- |")
        for problem in problems:
            add(f"| `{problem['code']}` | {problem['detail']} |")
    add("")

    add("## Signal invariants")
    add("")
    for key, value in findings["signal_invariants"].items():
        add(f"- `{key}` = {value}")
    add("")

    add("## Failing executes")
    add("")
    if not findings["failures"]:
        add("No failing `qnn_execute_end` was found in this trace.")
    for failure in findings["failures"]:
        add(f"### execute_id={failure['execute_id']} "
            f"step={failure['step']} batch={failure['batch']} "
            f"qnn_result={failure['qnn_result']}")
        add("")
        observations = failure["observations"]
        add("| observation | delta to failure | nearest event |")
        add("| --- | --- | --- |")
        add(f"| HVX RPC end | {us(observations['nearest_hvx_rpc_end']['delta_ns'])} "
            f"| `{observations['nearest_hvx_rpc_end']['event']}` |")
        add(f"| heartbeat write end | {ms(observations['nearest_heartbeat_write_end']['delta_ns'])} "
            f"| `{observations['nearest_heartbeat_write_end']['event']}` |")
        add(f"| progress status write end | {ms(observations['nearest_progress_status_write_end']['delta_ns'])} "
            f"| `{observations['nearest_progress_status_write_end']['event']}` |")
        add(f"| progress JNI end | {ms(observations['nearest_progress_jni_end']['delta_ns'])} "
            f"| `{observations['nearest_progress_jni_end']['event']}` |")
        add(f"| checkpoint end | {ms(observations['nearest_checkpoint_end']['delta_ns'])} "
            f"| `{observations['nearest_checkpoint_end']['event']}` |")
        stop = observations["nearest_stop_check"]
        add(f"| stop_check | {ms(stop['delta_ns'])} | `{stop['event']}` |")
        add("")
        add(f"stopRequested state at the failure: "
            f"`{stop.get('stop_requested')}`")
        add("")
        add("#### Events within the analysis window")
        add("")
        for description in failure["window_events"]:
            add(f"- `{description}`")
        add("")

    if findings.get("legacy"):
        add("## Legacy evidence (pre-instrumentation)")
        add("")
        add("These runs predate the incident trace. Only information recoverable")
        add("from the saved primary report is reported; nothing is reconstructed")
        add("or inferred about events the run did not record.")
        add("")
        legacy = findings["legacy"]
        items = legacy if isinstance(legacy, list) else [legacy]
        for item in items:
            add(f"- **{item.get('label', 'legacy run')}**:")
            for key, value in sorted(item["facts"].items()):
                add(f"  - `{key}` = {value}")
            for key, value in sorted(item.get("derived", {}).items()):
                add(f"  - `{key}` (derived from the recorded execute counts) = {value}")
            for note in item.get("unknown", []):
                add(f"  - NOT RECORDED: {note}")
        add("")

    if findings.get("logcat_terms"):
        add("## logcat keyword hits")
        add("")
        add("| term | lines |")
        add("| --- | --- |")
        for term, count in sorted(findings["logcat_terms"].items()):
            add(f"| `{term}` | {count} |")
        add("")
        add("The raw logcat is preserved alongside this report; these counts are")
        add("a navigation aid, not an interpretation.")
        add("")

    return "\n".join(lines) + "\n"


LOGCAT_TERMS = (
    "qnn", "qairt", "htp", "adsprpc", "fastrpc", "dsp", "ssr", "signal",
    "abort", "graph", "6031", "fatal", "error", "reset", "domain", "skel",
)


def scan_logcat(path: Path) -> dict[str, int]:
    counts = {term: 0 for term in LOGCAT_TERMS}
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    for term in LOGCAT_TERMS:
        counts[term] = text.count(term)
    return counts


def analyze(args: argparse.Namespace) -> int:
    events: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    incident_mode = bool(args.native)
    native_in_order: list[dict[str, Any]] = []
    kotlin_in_order: list[dict[str, Any]] = []

    if args.native:
        try:
            native_in_order = parse_trace(args.native)
            events.extend(native_in_order)
        except ParseProblem as error:
            problems.append({"code": "TRACE_PARSE_ERROR", "detail": str(error)})
    if args.kotlin:
        try:
            kotlin_in_order = parse_trace(args.kotlin)
            events.extend(kotlin_in_order)
        except ParseProblem as error:
            problems.append({"code": "TRACE_PARSE_ERROR", "detail": str(error)})

    timeline = Timeline(
        sorted(events, key=lambda e: as_int(e, "ts_ns") or 0),
        native_in_order=native_in_order,
        kotlin_in_order=kotlin_in_order,
    )
    counters = trace_counters(timeline)

    report = load_kv_report(args.report) if args.report else None
    if report:
        first = args.report.read_text(encoding="utf-8", errors="replace").splitlines()
        report["__first_line"] = first[0] if first else ""
    status = None
    if args.status:
        try:
            status = json.loads(args.status.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append({"code": "STATUS_PARSE_ERROR", "detail": str(error)})

    problems.extend(check_invariants(timeline, report, status, incident_mode))

    # Signal invariants are reported even when the trace is untrustworthy.
    nonnull = sum(
        1 for e in timeline.events
        for key in ("signal_arg1", "signal_arg2")
        if e.get(key) not in (None, "null"))
    trigger_values = [
        as_int(e, "hexatrain_signal_trigger_count") for e in timeline.events
        if "hexatrain_signal_trigger_count" in e]
    signal_invariants = {
        "qnn_signal_argument_nonnull_count": nonnull,
        "hexatrain_signal_trigger_count": max(trigger_values) if trigger_values else None,
    }

    analysis = failure_analysis(timeline)
    logcat_terms = scan_logcat(args.logcat) if args.logcat else None

    legacy = None
    if args.legacy_dir:
        # Every primary report in the directory is analyzed; nothing is assumed
        # about which runs are present.
        legacy = [analyze_legacy(path) for path in sorted(args.legacy_dir.glob("*-result.txt"))]
        if not legacy:
            legacy = [analyze_legacy(path)
                      for path in sorted(args.legacy_dir.glob("*result.txt"))]
    elif args.legacy:
        legacy = analyze_legacy(args.legacy)

    findings = {
        "schema_version": SCHEMA_VERSION,
        "incident_mode": incident_mode,
        "evidence_class": "instrumented" if incident_mode else "legacy",
        "trace_mode": detect_mode(timeline) if incident_mode else "legacy",
        "trace_overhead": {
            "trace_overflow_count": counters.get("trace_overflow_count"),
            "trace_event_count": counters.get("trace_event_count"),
            "trace_bytes": counters.get("trace_bytes"),
        },
        "counts": {
            "native": sum(1 for e in timeline.events if e.get("src") == "native"),
            "kotlin": sum(1 for e in timeline.events if e.get("src") == "kotlin"),
            "failures": len(analysis["failures"]),
        },
        "signal_invariants": signal_invariants,
        "problems": problems,
        "failures": analysis["failures"],
        "logcat_terms": logcat_terms,
        "legacy": legacy,
    }

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "incident-findings.json").write_text(
            json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (out_dir / "incident-report.md").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown(findings))
        with (out_dir / "incident-timeline.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ts_ns", "src", "tid", "thread", "event", "step",
                             "batch", "detail"])
            for event in timeline.events:
                detail = " ".join(
                    f"{k}={v}" for k, v in sorted(event.items())
                    if k not in ("ts_ns", "src", "tid", "thread", "event", "step",
                                 "batch", "_lineno", "_raw", "_unparsed"))
                writer.writerow([
                    event.get("ts_ns"), event.get("src"), event.get("tid"),
                    event.get("thread"), event.get("event"), event.get("step"),
                    event.get("batch"), detail])

    print(render_markdown(findings))
    return 1 if problems else 0


def analyze_legacy(path: Path) -> dict[str, Any]:
    """Extract only what a pre-instrumentation primary report actually records.

    This function must never invent instrumentation-equivalent values.  Facts
    that the legacy run did not record are listed under `unknown` so a reader
    cannot mistake their absence for a zero.  The step/batch derivation uses
    only the counts the report itself contains; the micro-batch size is a
    documented constant of the L19 Muon loop (8 micro-batches per step), not an
    inference from data the run did not save.
    """
    report = load_kv_report(path)
    attempts = report.get("api_trace_graph_execute_attempt_count")
    successes = report.get("api_trace_graph_execute_success_count")
    failures = report.get("api_trace_graph_execute_failure_count")
    first_failure = report.get("api_trace_graph_execute_first_failure_call")
    facts = {
        "status": report.get("status"),
        "error": report.get("error"),
        "test": report.get("test"),
        "cpu_fallback": report.get("cpu_fallback"),
        "api_trace_graph_execute_attempt_count": attempts,
        "api_trace_graph_execute_success_count": successes,
        "api_trace_graph_execute_failure_count": failures,
        "api_trace_graph_execute_first_failure_call": first_failure,
        "api_trace_last_qnn_result": report.get("api_trace_last_qnn_result"),
        "compile_time_sdk_build_id": report.get("compile_time_sdk_build_id"),
        "backend_build_id_match": report.get("backend_build_id_match"),
        "qnn_skel_action": report.get("qnn_skel_action"),
        "api_trace_fallback_attempted": report.get("api_trace_fallback_attempted"),
        "focus_takeover_count": report.get("focus_takeover_count"),
    }
    derived: dict[str, Any] = {}
    if first_failure is not None and attempts is not None:
        call = int(first_failure)
        derived["derived_first_failure_step"] = call // LEGACY_MICRO_BATCH + 1
        derived["derived_first_failure_batch"] = call % LEGACY_MICRO_BATCH
        derived["derived_micro_batch_size"] = LEGACY_MICRO_BATCH
        derived["derived_successful_executes_before_failure"] = successes
    unknown = [
        "per-execute timeline",
        "monotonic timestamps",
        "HVX FastRPC timing relative to the failure",
        "heartbeat / progress write timing relative to the failure",
        "stopRequested state at the failure",
        "thread identity at the failure",
        "QnnSignal argument state per execute (statically verified to be "
        "nullptr at all 24 call sites, but not recorded per execute)",
    ]
    return {
        "label": path.name,
        "facts": {k: v for k, v in facts.items() if v is not None},
        "derived": derived,
        "unknown": unknown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--kotlin", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--logcat", type=Path)
    parser.add_argument("--hostlog", type=Path)
    parser.add_argument("--legacy", type=Path,
                        help="a pre-instrumentation primary report, analyzed as "
                             "legacy evidence (never merged with instrumented data)")
    parser.add_argument("--legacy-dir", type=Path,
                        help="directory of pre-instrumentation primary reports")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        import incident_trace_selftest
        return incident_trace_selftest.run()
    if not any((args.native, args.legacy, args.legacy_dir)):
        parser.error("one of --native, --legacy or --legacy-dir is required")
    return analyze(args)


if __name__ == "__main__":
    sys.exit(main())
