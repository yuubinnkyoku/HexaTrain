#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Synthetic fixture battery for scripts/incident_6031_analyze.py.

These fixtures are hand-built traces, not recordings.  They exist so the
analyzer's fail-closed behaviour and its timeline arithmetic are pinned on the
host, without a device.  Every case asserts a specific expectation; a case that
starts passing for the wrong reason is a bug in this file, not progress.

Cases cover the incident scenarios that matter plus every fail-closed rule:

  normal_success            clean run, no problems
  failure_step4_batch0      6031 at step 4 batch 0 (incident A)
  failure_step77_batch0     6031 at step 77 batch 0 (incident B)
  failure_missing_end       execute_begin with no execute_end and no failure end
  failure_after_heartbeat   heartbeat write immediately before the failure
  failure_after_progress    progress status write immediately before
  failure_after_hvx_rpc     HVX FastRPC end immediately before
  failure_no_external_event failure with nothing nearby (negative evidence)
  timestamp_disorder        ts_ns goes backwards
  duplicate_execute_id      the same execute_id begins twice
  signal_nonnull            a non-null signal argument
  signal_trigger_violation  hexatrain_signal_trigger_count != 0
  identity_mismatch         status.json run_id differs from the trace
  missing_trace_file        incident mode with an empty native trace
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import incident_6031_analyze as analyzer

MS = 1_000_000
US = 1_000
# The synthetic failing execute stays "in QNN" for this long before returning
# 6031, so every delta measured to `qnn_execute_end` carries this offset.
FAIL_EXECUTE_US = 900


def _native(ts: int, tid: int, step: int, batch: int, payload: str) -> str:
    return (f"ts_ns={ts} tid={tid} src=native step={step} batch={batch} {payload}\n")


def _kotlin(ts: int, tid: int, payload: str) -> str:
    return f"ts_ns={ts} tid={tid} src=kotlin step=0 batch=-1 {payload}\n"


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(lines), encoding="utf-8")
    return path


class Fixture:
    """A synthetic incident run under a temporary directory.

    Every fixture carries a Kotlin `trace_start` because the analyzer treats a
    Kotlin trace file that exists but has no anchor as unusable.  Fixtures that
    model a run with no heartbeat/progress activity still write the anchor and
    nothing else, which is exactly what a quiet run produces.
    """

    def __init__(self, root: Path, native_lines: list[str], kotlin_lines: list[str]):
        # Anchor the Kotlin side at ts 0 so every fixture has a valid Kotlin
        # trace regardless of what activity it models.
        if not any("event=trace_start" in line for line in kotlin_lines):
            kotlin_lines = [_kotlin(0, 299, "event=trace_start")] + kotlin_lines
        self.native = write(root / "incident-native-trace.log", native_lines)
        self.kotlin = write(root / "incident-kotlin-trace.log", kotlin_lines)
        self.root = root

    def parse(self) -> analyzer.Timeline:
        native = analyzer.parse_trace(self.native)
        kotlin = analyzer.parse_trace(self.kotlin)
        return analyzer.Timeline(
            sorted(native + kotlin,
                   key=lambda e: analyzer.as_int(e, "ts_ns") or 0),
            native_in_order=native,
            kotlin_in_order=kotlin)

    def problems(self) -> list[dict[str, str]]:
        return analyzer.check_invariants(self.parse(), None, None, True)

    def codes(self) -> set[str]:
        return {p["code"] for p in self.problems()}


FIXTURE_RUN_ID = "20261001-000000-000"


def _header(step_total: int, first_step: int = 1) -> list[str]:
    return [
        _native(0, 100, 0, -1, "event=trace_start pid=4242 run_id=%s "
                                "unix_anchor_ms=1 monotonic_anchor_ns=0 "
                                "clock=steady_clock" % FIXTURE_RUN_ID),
        _native(1, 100, 0, -1, "event=training_start steps=%d resume_step=0 "
                                "micro_batch=8 backend=HVX_W8" % step_total),
    ]


def _successful_execute(ts: int, tid: int, step: int, batch: int,
                        execute_id: int, gap_us: int = 1000) -> int:
    begin = ts
    end = ts + gap_us * US
    return begin, [
        _native(begin, tid, step, batch, f"event=execute_begin execute_id={execute_id}"),
        _native(begin + 1, tid, step, batch, "event=qnn_execute_begin "
                f"execute_id={execute_id} signal_arg1=null signal_arg2=null "
                f"qnn_signal_argument_nonnull_count=0 "
                f"hexatrain_signal_trigger_count=0"),
        _native(end, tid, step, batch, f"event=qnn_execute_end execute_id={execute_id} "
                "qnn_result=0 success=true"),
        _native(end + 1, tid, step, batch, f"event=execute_end execute_id={execute_id} ok=true"),
    ], end + 2


HEAD_OFFSET_NS = 3  # stop_check, zero_parameters_begin, zero_parameters_end


def _step_frame(ts: int, step: int, body: list[str]) -> tuple[int, list[str]]:
    """Wraps `body` in a well-formed step boundary so phase pairing closes.

    The leading markers occupy `ts .. ts+2`, so `body` must already start at or
    after `ts + HEAD_OFFSET_NS`; the trailing markers land strictly after the
    last body timestamp.  Callers pass `ts` as the step start and generate the
    body from `ts + HEAD_OFFSET_NS`.
    """
    head = [
        f"event=stop_check step={step} stop_requested=0",
        f"event=zero_parameters_begin step={step}",
        f"event=zero_parameters_end step={step}",
    ]
    tail = [
        f"event=optimizer_begin step={step} backend=HVX_W8",
        f"event=optimizer_end step={step} rpc_status=0 fallback=false "
        f"output_finite=true",
        f"event=parameter_move_begin step={step}",
        f"event=parameter_move_end step={step}",
        f"event=telemetry_begin step={step}",
        f"event=telemetry_end step={step}",
        f"event=checkpoint_begin step={step}",
        f"event=checkpoint_end step={step} written=false",
    ]
    lines = [_native(ts + offset, 200, step, -1, payload)
             for offset, payload in enumerate(head)]
    body_end = ts + HEAD_OFFSET_NS
    for line in body:
        ts_of_line = analyzer.as_int(
            {"ts_ns": line.split()[0].split("=", 1)[1]}, "ts_ns")
        body_end = max(body_end, ts_of_line)
    lines.extend(body)
    for offset, payload in enumerate(tail, start=1):
        lines.append(_native(body_end + offset, 200, step, -1, payload))
    return ts, lines


def _hvx(ts: int, step: int, invocation: int) -> list[str]:
    return [
        _native(ts, 200, step, -1, f"event=hvx_lock_acquired invocation=0 "
                "operation=mutex rpc_status=0 session=hvx_muon_session"),
        _native(ts + 10 * US, 200, step, -1, f"event=hvx_rpc_begin "
                f"invocation={invocation} operation=run rpc_status=-1 "
                "session=hvx_muon_session"),
        _native(ts + 500 * US, 200, step, -1, f"event=hvx_rpc_end "
                f"invocation={invocation} operation=run rpc_status=0 "
                "session=hvx_muon_session"),
    ]


def _heartbeat(ts: int) -> list[str]:
    return [
        _kotlin(ts, 300, "event=heartbeat_wake reason=heartbeat"),
        _kotlin(ts + 5 * MS, 300, "event=heartbeat_write_begin reason=heartbeat"),
        _kotlin(ts + 8 * MS, 300, "event=heartbeat_write_end reason=heartbeat"),
    ]


def _progress(ts: int, step: int) -> list[str]:
    return [
        _native(ts, 200, step, -1, f"event=progress_jni_begin step={step}"),
        _native(ts + 3 * MS, 200, step, -1, f"event=progress_jni_end step={step}"),
        _kotlin(ts + 3 * MS, 310, "event=progress_callback_enter"),
        _kotlin(ts + 3 * MS + 10, 310, "event=progress_callback_exit"),
        _kotlin(ts + 4 * MS, 310, "event=progress_status_write_begin reason=progress"),
        _kotlin(ts + 6 * MS, 310, "event=progress_status_write_end reason=progress"),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def fixture_normal_success(root: Path) -> Fixture:
    lines = _header(2)
    ts = 1000
    execute_id = 0
    for step in (1, 2):
        step_start = ts
        ts = step_start + HEAD_OFFSET_NS
        body: list[str] = []
        for batch in range(8):
            execute_id += 1
            begin, produced, ts = _successful_execute(
                ts, 200, step, batch, execute_id)
            body.extend(produced)
        base, framed = _step_frame(step_start, step, body)
        lines.extend(framed)
        # The frame tail ends at the last checkpoint_end; per-step DSP and JNI
        # activity must be placed strictly after it or file order goes backwards.
        step_end = max(
            analyzer.as_int({"ts_ns": line.split()[0].split("=", 1)[1]}, "ts_ns")
            for line in framed)
        lines.extend(_hvx(step_end + 1000, step, step))
        lines.extend(_progress(step_end + 700 * US, step))
        ts = step_end + 10 * MS
    lines.extend(_heartbeat(ts))
    return Fixture(root, lines, [])


def _failure_fixture(root: Path, fail_step: int, preceding_kotlin: list[str],
                     preceding_native: list[str], gap_ns: int) -> Fixture:
    """Builds a run whose first `fail_step-1` steps succeed, then aborts."""
    lines = _header(fail_step)
    ts = 1000
    execute_id = 0
    for step in range(1, fail_step):
        step_start = ts
        ts = step_start + HEAD_OFFSET_NS
        body = []
        for batch in range(8):
            execute_id += 1
            begin, produced, ts = _successful_execute(
                ts, 200, step, batch, execute_id)
            body.extend(produced)
        base, framed = _step_frame(step_start, step, body)
        lines.extend(framed)
        # The frame tail ends at the last checkpoint_end; per-step DSP and JNI
        # activity must be placed strictly after it or file order goes backwards.
        step_end = max(
            analyzer.as_int({"ts_ns": line.split()[0].split("=", 1)[1]}, "ts_ns")
            for line in framed)
        lines.extend(_hvx(step_end + 1000, step, step))
        lines.extend(_progress(step_end + 700 * US, step))
        ts = step_end + 10 * MS
    # The failing step: step boundary closes, then the next batch-0 execute
    # returns 6031 with no execute_end, exactly as a real abort does.
    base = ts
    failing_frame = _step_frame(base, fail_step, [])[1]
    lines.extend(failing_frame)
    frame_end = max(
        analyzer.as_int({"ts_ns": line.split()[0].split("=", 1)[1]}, "ts_ns")
        for line in failing_frame)
    # `preceding_native` (e.g. a FastRPC) is placed after the frame so the
    # delta assertion in the case table is meaningful; the fixtures that need
    # a specific delta rewrite the file themselves.
    lines.extend(preceding_native)
    fail_ts = frame_end + gap_ns
    lines.append(_native(fail_ts, 200, fail_step, 0,
                         "event=execute_begin execute_id=%d" % (execute_id + 1)))
    lines.append(_native(fail_ts + 1, 200, fail_step, 0,
                         "event=qnn_execute_begin execute_id=%d signal_arg1=null "
                         "signal_arg2=null qnn_signal_argument_nonnull_count=0 "
                         "hexatrain_signal_trigger_count=0" % (execute_id + 1)))
    lines.append(_native(fail_ts + FAIL_EXECUTE_US * US, 200, fail_step, 0,
                         "event=qnn_execute_end execute_id=%d qnn_result=6031 "
                         "success=false" % (execute_id + 1)))
    lines.extend(preceding_kotlin)
    return Fixture(root, lines, [])


def fixture_failure_step4_batch0(root: Path) -> Fixture:
    return _failure_fixture(root, 4, [], [], 5 * MS)


def fixture_failure_step77_batch0(root: Path) -> Fixture:
    return _failure_fixture(root, 77, [], [], 5 * MS)


def fixture_failure_missing_end(root: Path) -> Fixture:
    """execute_begin + qnn_execute_begin with no end record at all.

    The analyzer must not invent a failure from a missing end; it reports the
    unpaired begin, which is the honest description.
    """
    lines = _header(1)
    lines.extend(_step_frame(1000, 1, [])[1])
    lines.append(_native(2000, 200, 1, 0, "event=execute_begin execute_id=1"))
    lines.append(_native(2001, 200, 1, 0, "event=qnn_execute_begin execute_id=1 "
                    "signal_arg1=null signal_arg2=null "
                    "qnn_signal_argument_nonnull_count=0 "
                    "hexatrain_signal_trigger_count=0"))
    return Fixture(root, lines, [])


def _failure_begin_ts(fixture: Fixture) -> int:
    """Timestamp of the failing execute_begin (the instant QNN was entered)."""
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    index = max(i for i, line in enumerate(lines) if "event=execute_begin" in line)
    return analyzer.as_int(
        {"ts_ns": lines[index].split()[0].split("=", 1)[1]}, "ts_ns")


def fixture_failure_after_heartbeat(root: Path) -> Fixture:
    """heartbeat_write_end lands 4 ms before the failing execute begins."""
    fixture = _failure_fixture(root, 2, [], [], 12 * MS)
    # _heartbeat(base) emits write_end at base+8ms.
    base = _failure_begin_ts(fixture) - 4 * MS - 8 * MS
    fixture.kotlin = write(fixture.root / "incident-kotlin-trace.log",
                           [_kotlin(0, 299, "event=trace_start")] + _heartbeat(base))
    return fixture


def fixture_failure_after_progress(root: Path) -> Fixture:
    """progress_status_write_end lands 2 ms before the failing execute begins."""
    fixture = _failure_fixture(root, 2, [], [], 9 * MS)
    base = _failure_begin_ts(fixture) - 9 * MS
    progress = [
        _kotlin(0, 299, "event=trace_start"),
        _kotlin(base + 4 * MS, 310, "event=progress_callback_enter"),
        _kotlin(base + 4 * MS + 10, 310, "event=progress_callback_exit"),
        _kotlin(base + 5 * MS, 310, "event=progress_status_write_begin reason=progress"),
        _kotlin(base + 7 * MS, 310, "event=progress_status_write_end reason=progress"),
    ]
    fixture.kotlin = write(fixture.root / "incident-kotlin-trace.log", progress)
    return fixture


def fixture_failure_after_hvx_rpc(root: Path) -> Fixture:
    """HVX RPC end 200 us before the failing execute.

    The RPC block is inserted immediately before the failing execute_begin so
    the file stays in timestamp order; a trace whose writes go backwards is
    exactly what the monotonicity check exists to catch.
    """
    fixture = _failure_fixture(root, 2, [], [], 5 * MS)
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    # The failing execute is the last execute_begin in the file.
    failure_begin_index = max(
        index for index, line in enumerate(lines)
        if "event=execute_begin" in line)
    failure_ts = analyzer.as_int(
        {"ts_ns": lines[failure_begin_index].split()[0].split("=", 1)[1]}, "ts_ns")
    rpc_block = _hvx(failure_ts - 700 * US, 1, 99)
    fixture.native = write(
        fixture.root / "incident-native-trace.log",
        lines[:failure_begin_index] + rpc_block + lines[failure_begin_index:])
    return fixture


def fixture_failure_no_external_event(root: Path) -> Fixture:
    """Failure with a 10 s gap from every external event.

    This is the negative-evidence case: the analyzer must still produce the
    failure entry with n/a deltas rather than inventing a nearby cause.
    """
    return _failure_fixture(root, 2, [], [], 10_000 * MS)


def fixture_timestamp_disorder(root: Path) -> Fixture:
    fixture = fixture_normal_success(root)
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    # Move one late record's timestamp before an earlier one.
    for index, line in enumerate(lines):
        if "event=telemetry_end" in line:
            lines[index] = line.replace("ts_ns=", "ts_ns=1", 1)
            break
    fixture.native = write(fixture.root / "incident-native-trace.log", lines)
    return fixture


def fixture_duplicate_execute_id(root: Path) -> Fixture:
    fixture = fixture_normal_success(root)
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    extra = [line for line in lines if "event=execute_begin execute_id=1" in line]
    lines.append(extra[0])
    fixture.native = write(fixture.root / "incident-native-trace.log", lines)
    return fixture


def fixture_signal_nonnull(root: Path) -> Fixture:
    fixture = fixture_normal_success(root)
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    patched = []
    for line in lines:
        if "signal_arg1=null" in line:
            line = line.replace("signal_arg1=null", "signal_arg1=0x7f00")
        patched.append(line)
    fixture.native = write(fixture.root / "incident-native-trace.log", patched)
    return fixture


def fixture_signal_trigger_violation(root: Path) -> Fixture:
    fixture = fixture_normal_success(root)
    lines = fixture.native.read_text(encoding="utf-8").splitlines(keepends=True)
    patched = []
    for line in lines:
        if "hexatrain_signal_trigger_count=0" in line:
            line = line.replace("hexatrain_signal_trigger_count=0",
                                "hexatrain_signal_trigger_count=1")
        patched.append(line)
    fixture.native = write(fixture.root / "incident-native-trace.log", patched)
    return fixture


def fixture_identity_mismatch(root: Path) -> Fixture:
    fixture = fixture_normal_success(root)
    status = root / "status.json"
    status.write_text(json.dumps({
        "schema_version": 1, "run_id": "20261001-999999-999",
        "suite": "nicopedia-long-training", "status": "FAILED",
    }) + "\n", encoding="utf-8")
    report = root / "result.txt"
    report.write_text("NICOPEDIA_HTP\nstatus=FAILED\nerror=x\n", encoding="utf-8")
    problems = analyzer.check_invariants(
        fixture.parse(), analyzer.load_kv_report(report),
        json.loads(status.read_text(encoding="utf-8")), True)
    return fixture, problems


def fixture_missing_trace_file(root: Path) -> Fixture:
    empty = write(root / "incident-native-trace.log", [])
    events = analyzer.parse_trace(empty)
    timeline = analyzer.Timeline(events)
    problems = analyzer.check_invariants(timeline, None, None, True)
    return Fixture(root, [], []), problems


# ---------------------------------------------------------------------------
# Case table
# ---------------------------------------------------------------------------

CLEAN_CASES = ("normal_success",)
FAILURE_CASES = (
    "failure_step4_batch0",
    "failure_step77_batch0",
    "failure_after_heartbeat",
    "failure_after_progress",
    "failure_after_hvx_rpc",
    "failure_no_external_event",
)
EXPECTED_PROBLEM_CASES = {
    "failure_missing_end": "EXECUTE_BEGIN_WITHOUT_END",
    "timestamp_disorder": "TIMESTAMP_DISORDER",
    "duplicate_execute_id": "EXECUTE_ID_DUPLICATE",
    "signal_nonnull": "SIGNAL_ARGUMENT_NONNULL",
    "signal_trigger_violation": "SIGNAL_TRIGGER_INVARIANT_VIOLATION",
    "identity_mismatch": "RUN_IDENTITY_MISMATCH",
    "missing_trace_file": "NATIVE_TRACE_MISSING",
}

FIXTURES: dict[str, Callable[[Path], Any]] = {
    "normal_success": fixture_normal_success,
    "failure_step4_batch0": fixture_failure_step4_batch0,
    "failure_step77_batch0": fixture_failure_step77_batch0,
    "failure_missing_end": fixture_failure_missing_end,
    "failure_after_heartbeat": fixture_failure_after_heartbeat,
    "failure_after_progress": fixture_failure_after_progress,
    "failure_after_hvx_rpc": fixture_failure_after_hvx_rpc,
    "failure_no_external_event": fixture_failure_no_external_event,
    "timestamp_disorder": fixture_timestamp_disorder,
    "duplicate_execute_id": fixture_duplicate_execute_id,
    "signal_nonnull": fixture_signal_nonnull,
    "signal_trigger_violation": fixture_signal_trigger_violation,
    "identity_mismatch": fixture_identity_mismatch,
    "missing_trace_file": fixture_missing_trace_file,
}


class SelfTestFailure(Exception):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def run() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="incident-selftest-") as tmp:
        for name, builder in FIXTURES.items():
            root = Path(tmp) / name
            root.mkdir(parents=True, exist_ok=True)
            try:
                result = builder(root)
                if name == "identity_mismatch":
                    _, problems = result
                    codes = {p["code"] for p in problems}
                elif name == "missing_trace_file":
                    _, problems = result
                    codes = {p["code"] for p in problems}
                else:
                    fixture = result
                    problems = fixture.problems()
                    codes = {p["code"] for p in problems}

                if name in EXPECTED_PROBLEM_CASES:
                    expected = EXPECTED_PROBLEM_CASES[name]
                    _check(expected in codes,
                           f"{name}: expected problem {expected}, got {sorted(codes)}")
                else:
                    _check(not codes,
                           f"{name}: expected a clean trace, got {sorted(codes)}")

                if name in FAILURE_CASES:
                    fixture = result
                    timeline = fixture.parse()
                    analysis = analyzer.failure_analysis(timeline)
                    entries = analysis["failures"]
                    _check(len(entries) == 1,
                           f"{name}: expected exactly one failure, got {len(entries)}")
                    entry = entries[0]
                    _check(entry["qnn_result"] == analyzer.ABORTED_QNN_RESULT,
                           f"{name}: expected 6031, got {entry['qnn_result']}")
                    _check(entry["batch"] == 0,
                           f"{name}: expected batch 0, got {entry['batch']}")
                    expected_step = 4 if name == "failure_step4_batch0" else (
                        77 if name == "failure_step77_batch0" else 2)
                    _check(entry["step"] == expected_step,
                           f"{name}: expected step {expected_step}, got {entry['step']}")
                    if name == "failure_after_hvx_rpc":
                        # Deltas are measured to the qnn_execute_end record,
                        # which lands FAIL_EXECUTE_US after execute_begin.
                        delta = entry["observations"]["nearest_hvx_rpc_end"]["delta_ns"]
                        expected = 200 * US + FAIL_EXECUTE_US * US
                        _check(delta is not None and abs(delta - expected) < 2 * US,
                               f"{name}: hvx delta {delta} ns not ~{expected} ns")
                    if name == "failure_after_heartbeat":
                        delta = entry["observations"]["nearest_heartbeat_write_end"]["delta_ns"]
                        expected = 4 * MS + FAIL_EXECUTE_US * US
                        _check(delta is not None and abs(delta - expected) < 100 * US,
                               f"{name}: heartbeat delta {delta} ns not ~{expected} ns")
                    if name == "failure_after_progress":
                        delta = entry["observations"]["nearest_progress_status_write_end"]["delta_ns"]
                        expected = 2 * MS + FAIL_EXECUTE_US * US
                        _check(delta is not None and abs(delta - expected) < 100 * US,
                               f"{name}: progress delta {delta} ns not ~{expected} ns")
                    if name == "failure_no_external_event":
                        observations = entry["observations"]
                        _check(observations["nearest_heartbeat_write_end"]["delta_ns"] is None,
                               f"{name}: expected no nearby heartbeat, got one")
                        _check(observations["nearest_hvx_rpc_end"]["event"] != "(none)",
                               f"{name}: an earlier step's RPC should still exist")
                print(f"  ok  {name}")
            except SelfTestFailure as error:
                failures.append(f"{name}: {error}")
                print(f"  FAIL {name}: {error}")
            except Exception as error:  # noqa: BLE001 - report, do not mask
                failures.append(f"{name}: unexpected {type(error).__name__}: {error}")
                print(f"  FAIL {name}: unexpected {type(error).__name__}: {error}")

    if failures:
        print(f"\nincident selftest FAILED ({len(failures)} case(s))")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nincident selftest PASS ({len(FIXTURES)} cases)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run())
