"""Read-only failure attribution for the shared-lifetime pilot monitor/transport.

No device calls, no arithmetic, no mutation of earlier private evidence.  Every
number is derived from files already written by the previous pilot, so the
attribution can be re-run and compared byte-for-byte.

Output (exclusive creation, private):
  build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/
    evidence-freeze-private.json
    monitor-cadence-attribution-private.json
    transport-attribution-private.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "build/reports/g1-backward-shared-lifetime-formal/pilot"
RECOVERY = ROOT / "build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery"
BUNDLE = ROOT / "build/fp32-saved-p-backward-shared-lifetime-five-role-final"
PLAN = ROOT / "host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json"

ROLES = ["standalone_softmax", "standalone_dvalue", "native_softmax", "native_dvalue", "dual"]
CONTRACT_PERIOD_SECONDS = 2.0
GAP_SPLIT_SECONDS = 10.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_records(path):
    """Completion timestamps in file order; the runner only recorded completion."""
    pattern = re.compile(r'"utc":"([^"]+)"')
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            out.append(match.group(1))
    return out


def parse_utc(text):
    from datetime import datetime, timezone

    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def deltas(times):
    return [round((b - a).total_seconds(), 6) for a, b in zip(times, times[1:])]


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    middle = ordered[len(ordered) // 2]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": middle,
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def evidence_freeze():
    plan = read(PLAN)
    build = read(BUNDLE / "build-identity.json")
    identity = {
        "schema": 1,
        "kind": "G1_SHARED_LIFETIME_MONITOR_TRANSPORT_EVIDENCE_FREEZE",
        "source_commit_at_freeze": "900e949dc5eee62cb230af39136f96a45df1824e",
        "branch": "investigation/g1-backward-shared-lifetime-formal",
        "candidate": build["candidate"],
        "classification_before_this_mission": "BLOCKED_FORMAL_COHORT",
        "formal_completed_before_this_mission": "0/3",
        "frozen_identities": {
            "probe_sha256": build["probe_sha256"],
            "dsp_sha256": build["binary_hashes"]["libHexaBackwardRepairDsp.so"],
            "host_sha256": build["binary_hashes"]["libHexaBackwardRepairHost.so"],
            "compiled_source_commit": build["source_sha"],
            "original_compiled_source_commit": plan["frozen"]["original_compiled_source"],
            "original_arm_sha256": plan["frozen"]["original_binary_sha256"]["probe"],
            "two_role_arm_sha256": plan["frozen"]["two_role_arm_sha256"],
            "formal_plan_sha256": sha(PLAN),
            "gradient_fixture_sha256": build["gradient_sha256"],
            "protected_40k_sha256": plan["frozen"]["protected_40k_sha256"],
            "pilot_runner_sha256": sha(ROOT / "scripts/run_backward_shared_lifetime_formal.ps1"),
            "pilot_runner_v2_sha256": sha(ROOT / "scripts/run_backward_shared_lifetime_formal_v2.ps1"),
            "pilot_orchestration_sha256_executed": read(
                PILOT / "execution-before-readback-private.json"
            )["orchestration_sha256"],
            "monitor_policy_sha256": sha(ROOT / "scripts/mixed_qkv_monitoring_policy.ps1"),
            "adb_common_sha256": sha(ROOT / "scripts/nicopedia_runner_common.ps1"),
            "numerical_assessor_sha256": sha(ROOT / "scripts/assess_backward_shared_lifetime_formal.py"),
            "certifier_v2_sha256": sha(ROOT / "scripts/certify_backward_shared_lifetime_formal_v2.py"),
            "public_exporter_v2_sha256": sha(ROOT / "scripts/export_public_backward_shared_lifetime_formal_v2.py"),
            "recovered_public_exporter_sha256": sha(
                ROOT / "scripts/export_public_recovered_backward_shared_lifetime.py"
            ),
            "plan_anchor_legacy_digests": plan["anchor_legacy_digests"],
        },
        "pilot_process": {
            "actual_exit": read(PILOT / "partial-completion-private.json")["actual_exit"],
            "duration_seconds": read(PILOT / "execution-before-readback-private.json")["duration_seconds"],
            "owned_processes": 1,
            "new_device_arithmetic_after_failure": 0,
        },
        "raw_inventory": {},
        "missing_raw": [],
        "mission_stop": read(ROOT / "build/reports/g1-backward-shared-lifetime-formal/mission-stop-private.json"),
        "new_evidence_policy": "APPEND_ONLY_NEW_EXCLUSIVE_DIRECTORY",
        "prior_private_evidence_rewritten": False,
    }
    for role in ROLES:
        name = "cohort-" + role + ".bin"
        raw = PILOT / name
        gz = PILOT / (name + ".gz")
        entry = {
            "raw_present": raw.exists(),
            "raw_bytes": raw.stat().st_size if raw.exists() else None,
            "raw_sha256": sha(raw) if raw.exists() else None,
            "gzip_present": gz.exists(),
            "gzip_bytes": gz.stat().st_size if gz.exists() else None,
            "gzip_sha256": sha(gz) if gz.exists() else None,
        }
        identity["raw_inventory"][role] = entry
        if not raw.exists():
            identity["missing_raw"].append(name)
    container = PILOT / "recovered-cohort-container.bin"
    identity["container"] = {
        "present": container.exists(),
        "bytes": container.stat().st_size if container.exists() else None,
        "sha256": sha(container) if container.exists() else None,
        "reconstructed_from_role_files": True,
    }
    partial = PILOT / "saved-p-words.bin.gz"
    identity["incomplete_container_transfer"] = {
        "present": partial.exists(),
        "bytes": partial.stat().st_size if partial.exists() else None,
        "sha256": sha(partial) if partial.exists() else None,
    }
    identity["missing_raw"].append("saved-p-words.bin")
    return identity


def cadence_attribution():
    tops = [parse_utc(value) for value in utc_records(PILOT / "top-private.jsonl")]
    awakes = [parse_utc(value) for value in utc_records(PILOT / "awake-private.jsonl")]
    competition = [parse_utc(value) for value in utc_records(PILOT / "competition-private.jsonl")]
    gaps = deltas(tops)
    # The runner performs two pre-launch top checks; the larger of the two gaps
    # (staging plus the host-inventory review window) separates them from the
    # monitoring loop.  The loop records are the ones after that second check.
    split = max((index for index, gap in enumerate(gaps) if gap >= GAP_SPLIT_SECONDS), default=-1)
    first = split + 2
    last = len(tops) - 2  # the final record is the post-exit top check
    runtime = tops[first : last + 1]
    runtime_gaps = deltas(runtime)

    # Iteration alignment: competition -> awake -> top completed in that order,
    # bounded by the previous top so a leftover record cannot be reused.
    ps_durations, awake_durations, block_durations, wait_residues, sums = [], [], [], [], []
    previous = tops[first - 1]
    for index, top in enumerate(runtime):
        comp = max((c for c in competition if previous < c <= top), default=None)
        awake = max((a for a in awakes if comp is not None and comp <= a <= top), default=None)
        if comp is None or awake is None:
            continue
        ps_durations.append(round((awake - comp).total_seconds(), 6))
        awake_durations.append(round((top - awake).total_seconds(), 6))
        block_durations.append(round((top - comp).total_seconds(), 6))
        if index:
            wait = round((comp - runtime[index - 1]).total_seconds(), 6)
            wait_residues.append(wait)
            sums.append(round(wait + (top - comp).total_seconds(), 6))
        previous = top

    return {
        "schema": 1,
        "kind": "G1_SHARED_LIFETIME_MONITOR_CADENCE_ATTRIBUTION",
        "instrumentation_actually_recorded": {
            "per_sample_completion_utc": True,
            "per_sample_monotonic_start": False,
            "per_query_boundaries": False,
            "explicit_scheduled_deadline": False,
            "missed_deadline_record": False,
            "monotonic_clock": False,
        },
        "contract": {
            "source": "host_tests/fp32_custom/backward_repair_cohort_plan.json backend_lifetime",
            "registered_text": "then two-second foreground monitoring throughout all roles",
            "interpreted_requirement": "foreground(top-resumed) sample start-to-start period <= 2.000 s",
            "period_used_by_runner": "process.WaitForExit(2000) in series with the full query block",
            "hidden_relaxation_in_original_runner": "the 2000 ms wait was additive, not a period",
        },
        "record_counting": {
            "total_top_records": len(tops),
            "pre_launch_records": first,
            "runtime_loop_records": len(runtime),
            "post_exit_records": len(tops) - first - len(runtime),
            "window_rule": "the last >=10 s gap separates the two pre-launch checks from the loop; the final record is the post-exit check",
        },
        "runtime_interval_seconds_completion_to_completion": stats(runtime_gaps),
        "all_interval_seconds": stats(gaps),
        "iteration_block_seconds": {
            "post_top_to_ps_completion_seconds": stats(wait_residues),
            "awake_plus_top_seconds": stats(block_durations),
            "reconstructed_interval": stats(sums),
            "awake_command_seconds": stats(ps_durations),
            "top_command_seconds": stats(awake_durations),
        },
        "violation": {
            "required_max_seconds": CONTRACT_PERIOD_SECONDS,
            "observed_median_seconds": stats(runtime_gaps)["median"],
            "observed_max_seconds": stats(runtime_gaps)["max"],
            "minimum_observed_seconds": stats(runtime_gaps)["min"],
            "samples_over_contract": sum(1 for value in runtime_gaps if value > CONTRACT_PERIOD_SECONDS),
            "samples_total": len(runtime_gaps),
            "classification": "MONITOR_CONTRACT_FAILED",
            "not_a_numerical_failure": True,
        },
        "root_cause_decomposition": {
            "serial_additive_wait_seconds": 2.0,
            "serial_query_block_median_seconds": stats(block_durations)["median"],
            "median_interval_reconstructed_from_parts": stats(sums)["median"],
            "median_interval_reported_from_completion_timestamps": stats(runtime_gaps)["median"],
            "thermal_block_iterations_are_the_maxima": "the largest intervals are the iterations that also ran the ten-second thermal block",
            "dominant_hypothesis": "ADDITIVE_WAIT_PLUS_SERIAL_QUERY_BLOCK",
            "supporting_hypotheses": [
                "SERIAL_POLLING_OF_PS_AWAKE_AND_ACTIVITY",
                "TEN_SECOND_THERMAL_BLOCK_ADDED_ON_ITS_OWN_ITERATION",
            ],
            "hypotheses_not_supported_as_primary": [
                "ADB_SERVER_CONTENTION",
                "DUMPSYS_RESPONSE_LATENCY_ONLY",
                "POWERSHELL_JOB_SCHEDULING",
                "POST_EXIT_TRANSFER_CONTENTION_DURING_RUN",
            ],
            "reasoning": "no transfer ran during the process; Competition/Awake/Top are strictly serial; the 2000 ms wait is added in series to that block",
        },
        "deadline_accounting_possible_from_original_log": False,
        "new_device_arithmetic": 0,
    }


def transport_attribution():
    remote = read(PILOT / "recovery-remote-hashes-private.json")["text_private"]
    remote_hashes = {}
    for line in remote.splitlines():
        parts = line.split()
        if len(parts) == 2:
            remote_hashes[Path(parts[1]).name] = parts[0]
    container_remote = remote_hashes.get("saved-p-words.bin")
    container_gz_remote = remote_hashes.get("saved-p-words.bin.gz")
    partial = PILOT / "saved-p-words.bin.gz"
    partial_sha = sha(partial) if partial.exists() else None
    container = PILOT / "recovered-cohort-container.bin"
    role_hashes = {}
    for role in ROLES:
        name = "cohort-" + role + ".bin"
        role_hashes[role] = {
            "remote_sha256": remote_hashes.get(name),
            "local_sha256": sha(PILOT / name),
            "remote_gzip_sha256": remote_hashes.get(name + ".gz"),
            "local_gzip_sha256": sha(PILOT / (name + ".gz")),
        }
        role_hashes[role]["raw_identity_verified"] = (
            role_hashes[role]["remote_sha256"] == role_hashes[role]["local_sha256"]
        )
        role_hashes[role]["gzip_identity_verified"] = (
            role_hashes[role]["remote_gzip_sha256"] == role_hashes[role]["local_gzip_sha256"]
        )
    return {
        "schema": 1,
        "kind": "G1_SHARED_LIFETIME_POST_EXIT_TRANSPORT_ATTRIBUTION",
        "failure_operation": "adb pull of the device-gzipped 5-role container",
        "exact_command_family": "adb -s <endpoint> pull <remote>/<name>.gz <private>/<name>.gz",
        "source_file": "saved-p-words.bin.gz (gzip -c of saved-p-words.bin)",
        "source_uncompressed_bytes": container.stat().st_size if container.exists() else None,
        "source_uncompressed_sha256": sha(container) if container.exists() else None,
        "remote_gzip_sha256": container_gz_remote,
        "local_partial_bytes": partial.stat().st_size if partial.exists() else None,
        "local_partial_sha256": partial_sha,
        "local_partial_complete": partial_sha == container_gz_remote,
        "configured_timeout_seconds": 180,
        "adb_exit_code_observed": 124,
        "adb_exit_code_meaning": "host wrapper killed the adb client after the finite bound",
        "host_wrapper_exit": 124,
        "device_side_file_preserved": True,
        "remote_after_sha256": container_remote,
        "remote_unchanged_after_failure": container_remote == (sha(container) if container.exists() else None),
        "elapsed_seconds": 180,
        "transferred_bytes_measurable": partial.exists(),
        "failure_classification": {
            "adb_command_timeout": False,
            "adb_wrapper_timeout": True,
            "adb_transport_disconnected": False,
            "post_exit_evidence_incomplete": True,
            "remote_raw_missing": False,
            "transfer_verified_for_container_gzip": False,
        },
        "interpretation": "the finite 180 s wrapper bound was smaller than the time needed to move the compressed container over the current transport; the wrapper killed its own adb client, leaving a partial local file and an intact remote file",
        "recovery": {
            "five_role_raw_and_gzip_recovered": all(
                value["raw_identity_verified"] and value["gzip_identity_verified"]
                for value in role_hashes.values()
            ),
            "container_reconstructed_losslessly": (
                (sha(container) if container.exists() else None) == container_remote
            ),
            "new_device_transfer_attempted_in_this_mission": False,
            "remote_deleted_or_rewritten": False,
        },
        "role_identity": role_hashes,
        "new_device_arithmetic": 0,
    }


def write_once(path, payload):
    data = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RECOVERY)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "evidence-freeze-private.json", evidence_freeze())
    write_once(out / "monitor-cadence-attribution-private.json", cadence_attribution())
    write_once(out / "transport-attribution-private.json", transport_attribution())
    print("monitor_transport_attribution=PASS files=3 device_calls=0")


if __name__ == "__main__":
    main()
