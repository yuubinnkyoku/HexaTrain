#!/usr/bin/env python3
"""Normalize private critical-path evidence and summarize prepared device audits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

PHASE_FIELDS = {
    "qnn_execute_ms": "fwd_backward_ms_per_update",
    "dsp_span_ms": "muon_kernel_ms_per_update",
    "state_move_ms": "optimizer_result_move_ms_per_update",
    "registry_validation_ms": "gradient_registry_validation_ms_per_update",
    "optimizer_wall_ms": "optimizer_update_wall_ms_per_update",
    "training_step_ms": "training_step_ms",
}


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def number(fields: dict[str, Any], name: str) -> float | None:
    try:
        value = float(fields[name])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_median_ci(values: list[float], seed: int, samples: int = 50_000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    lo, hi = percentile(draws, .025), percentile(draws, .975)
    return [round(lo, 8), round(hi, 8)] if lo is not None and hi is not None else None


def derive_registry_ms(fields: dict[str, str], steps: int) -> float | None:
    value = number(fields, "gradient_registry_validation_us")
    if value is None or steps <= 0:
        return None
    return value / steps / 1000.0


def derive_dsp_ms(fields: dict[str, str], steps: int) -> float | None:
    value = number(fields, "muon_kernel_us")
    if value is None or steps <= 0:
        return None
    return value / steps / 1000.0


def normalize_legacy(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [("main", "before"), ("registry-on", "candidate")]
    for prefix, arm in specs:
        for model in ("control", "g1"):
            for repetition in (0, 2, 3, 4, 5):
                folder = root / f"{prefix}-{model}-r{repetition}"
                results = list(folder.glob("*-result.txt"))
                if not results:
                    continue
                fields = parse_kv(results[0])
                steps = int(fields.get("steps", "0") or 0)
                qnn = number(fields, "qnn_return_code_success")
                finite = fields.get("output_tensors_finite") == "true" and fields.get("all_steps_finite") == "true" and fields.get("final_finite") == "true"
                fallback = fields.get("cpu_fallback") == "true"
                row: dict[str, Any] = {
                    "dataset": "initial_profile" if repetition == 0 else "matched",
                    "arm": arm,
                    "model": model,
                    "repetition": repetition,
                    "pair_id": None if repetition == 0 else f"r{repetition}-{model}",
                    "run_id": folder.name,
                    "run_order": None,
                    "steps": steps,
                    "training_step_ms": number(fields, "training_step_ms"),
                    "training_total_seconds": number(fields, "training_total_seconds"),
                    "qnn_execute_ms": number(fields, "fwd_backward_ms_per_update"),
                    "dsp_span_ms": derive_dsp_ms(fields, steps),
                    "state_move_ms": number(fields, "optimizer_result_move_ms_per_update"),
                    "registry_validation_ms": derive_registry_ms(fields, steps),
                    "optimizer_wall_ms": number(fields, "optimizer_update_wall_ms_per_update"),
                    "battery_temperature_c_before": number(fields, "battery_temperature_c_before"),
                    "battery_temperature_c_after": number(fields, "battery_temperature_c_after"),
                    "thermal_status_before": number(fields, "android_thermal_status_before"),
                    "thermal_status_after": number(fields, "android_thermal_status_after"),
                    "status": fields.get("status", "UNKNOWN"),
                    "qnn_return_code_success": fields.get("qnn_return_code_success", "UNKNOWN"),
                    "output_tensors_finite": fields.get("output_tensors_finite", "UNKNOWN"),
                    "cpu_fallback": fields.get("cpu_fallback", "UNKNOWN"),
                    "health_ok": fields.get("status") == "SUCCESS" and fields.get("qnn_return_code_success") == "true" and finite and not fallback,
                }
                rows.append(row)
    return rows


def load_device_rows(audit_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for manifest_path in sorted(audit_root.rglob("run-manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            excluded.append({"run_id": manifest_path.parent.name, "reason": "INCOMPLETE_MANIFEST"})
            continue
        run = manifest_path.parent
        report_name = manifest.get("result_file")
        result_path = run / report_name if report_name else None
        fields = parse_kv(result_path) if result_path else {}
        reasons: list[str] = []
        if manifest.get("status") != "COMPLETED":
            reasons.append(str(manifest.get("failure_class") or "INCOMPLETE_RUN"))
        if not result_path or not result_path.exists():
            reasons.append("INCOMPLETE_REPORT")
        expected_steps = int(manifest.get("steps", 0) or 0)
        completed = int(fields.get("completed_steps", fields.get("run_completed_steps", "0")) or 0)
        if fields.get("status") != "SUCCESS" or expected_steps <= 0 or completed != expected_steps:
            reasons.append("INCOMPLETE_REPORT")
        if fields.get("qnn_return_code_success") != "true" or fields.get("output_tensors_finite") != "true":
            reasons.append("QNN_OR_TENSOR_HEALTH_FAILURE")
        if fields.get("all_steps_finite") != "true" or fields.get("final_finite") != "true":
            reasons.append("NONFINITE_OR_MISSING_FINITE_EVIDENCE")
        if fields.get("cpu_fallback") != "false":
            reasons.append("FALLBACK_OR_UNKNOWN")
        if fields.get("optimizer_muon_backend", "").upper() not in ("HVX_W8", "HVX"):
            reasons.append("HVX_BACKEND_MISMATCH_OR_UNKNOWN")
        artifact = manifest.get("artifact", {})
        if not artifact.get("app_apk_sha256") or not artifact.get("android_test_apk_sha256"):
            reasons.append("IDENTITY_MISMATCH_OR_MISSING")
        row = {
            "dataset": manifest.get("audit_mode", "device_audit"),
            "arm": manifest.get("arm", "UNKNOWN"),
            "model": manifest.get("model", "UNKNOWN"),
            "repetition": manifest.get("repetition"),
            "pair_id": manifest.get("pair_id"),
            "run_id": manifest.get("run_id", run.name),
            "run_order": manifest.get("run_order"),
            "steps": expected_steps,
            "training_step_ms": number(fields, "training_step_ms"),
            "training_total_seconds": number(fields, "training_total_seconds"),
            "qnn_execute_ms": number(fields, "fwd_backward_ms_per_update"),
            "dsp_span_ms": derive_dsp_ms(fields, completed or expected_steps),
            "state_move_ms": number(fields, "optimizer_result_move_ms_per_update"),
            "registry_validation_ms": derive_registry_ms(fields, completed or expected_steps),
            "optimizer_wall_ms": number(fields, "optimizer_update_wall_ms_per_update"),
            "app_apk_sha256": artifact.get("app_apk_sha256"),
            "android_test_apk_sha256": artifact.get("android_test_apk_sha256"),
            "battery_temperature_c_before": number(fields, "battery_temperature_c_before"),
            "battery_temperature_c_after": number(fields, "battery_temperature_c_after"),
            "thermal_status_before": number(fields, "android_thermal_status_before"),
            "thermal_status_after": number(fields, "android_thermal_status_after"),
            "status": fields.get("status", manifest.get("status", "UNKNOWN")),
            "health_ok": not reasons,
            "exclusion_reasons": sorted(set(reasons)),
            "manifest_path": str(manifest_path),
        }
        rows.append(row)
        if reasons:
            excluded.append({"run_id": row["run_id"], "pair_id": row["pair_id"], "model": row["model"], "arm": row["arm"], "reasons": row["exclusion_reasons"]})
    return validate_matched_apk_identities(rows, excluded)


def validate_matched_apk_identities(rows: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched_rows = [row for row in rows if row.get("dataset") == "MatchedAB"]
    arm_rows = {arm: [row for row in matched_rows if row.get("arm") == arm] for arm in ("before", "candidate")}
    before_app_hashes = {row.get("app_apk_sha256") for row in arm_rows["before"] if row.get("app_apk_sha256")}
    candidate_app_hashes = {row.get("app_apk_sha256") for row in arm_rows["candidate"] if row.get("app_apk_sha256")}
    before_test_hashes = {row.get("android_test_apk_sha256") for row in arm_rows["before"] if row.get("android_test_apk_sha256")}
    candidate_test_hashes = {row.get("android_test_apk_sha256") for row in arm_rows["candidate"] if row.get("android_test_apk_sha256")}
    invalid_arm_rows: list[dict[str, Any]] = []
    if len(before_app_hashes) > 1 or len(candidate_app_hashes) > 1:
        invalid_arm_rows.extend(matched_rows)
    if before_app_hashes and candidate_app_hashes and before_app_hashes == candidate_app_hashes:
        invalid_arm_rows.extend(matched_rows)
    if before_test_hashes and candidate_test_hashes and before_test_hashes != candidate_test_hashes:
        invalid_arm_rows.extend(matched_rows)
    for row in invalid_arm_rows:
        row["health_ok"] = False
        row["exclusion_reasons"] = sorted(set(row["exclusion_reasons"] + ["CROSS_RUN_APK_IDENTITY_MISMATCH"]))
        if not any(item.get("run_id") == row["run_id"] for item in excluded):
            excluded.append({"run_id": row.get("run_id"), "pair_id": row.get("pair_id"), "model": row.get("model"), "arm": row.get("arm"), "reasons": row["exclusion_reasons"]})
    return rows, excluded


def audit_summary(rows: list[dict[str, Any]], excluded: list[dict[str, Any]], seed: int = 20261007) -> dict[str, Any]:
    valid = [r for r in rows if r.get("health_ok") and r.get("training_step_ms") is not None]
    before_only = bool(rows) and all(r.get("dataset") == "BeforeOnly300vs400" for r in rows)
    pair_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in valid:
        if before_only or row.get("arm") not in ("before", "candidate") or not row.get("pair_id"):
            continue
        pair_groups.setdefault((str(row.get("model")), str(row.get("pair_id"))), {})[str(row["arm"])] = row
    pairs: dict[str, list[dict[str, Any]]] = {"control": [], "g1": []}
    incomplete_pairs: list[dict[str, Any]] = []
    for (model, pair_id), arms in sorted(pair_groups.items()):
        if "before" not in arms or "candidate" not in arms:
            incomplete_pairs.append({"model": model, "pair_id": pair_id, "reason": "MISSING_ARM"})
            continue
        before = arms["before"]["training_step_ms"]
        after = arms["candidate"]["training_step_ms"]
        pairs.setdefault(model, []).append({"pair_id": pair_id, "before_ms": before, "candidate_ms": after,
                                            "paired_speedup": before / after,
                                            "gain_percent": 100.0 * (1.0 - after / before),
                                            "before_run_id": arms["before"]["run_id"], "candidate_run_id": arms["candidate"]["run_id"]})
    summaries: dict[str, Any] = {}
    for model, model_pairs in pairs.items():
        gains = [float(p["gain_percent"]) for p in model_pairs]
        ci = bootstrap_median_ci(gains, seed + (0 if model == "control" else 1))
        summaries[model] = {
            "valid_pairs": len(model_pairs), "positive_pairs": sum(g > 0 for g in gains), "pairs": model_pairs,
            "median_gain_percent": median(gains), "median_paired_speedup": median([float(p["paired_speedup"]) for p in model_pairs]),
            "bootstrap_95_median_gain_percent": ci,
            "classification": "INSUFFICIENT_VALID_PAIRS" if len(model_pairs) < 3 else
                ("LAND" if median(gains) is not None and median(gains) >= 5 and ci and ci[0] > 0 else
                 ("CONDITION_DEPENDENT_GAIN" if median(gains) is not None and median(gains) > 0 else "NO_REPRODUCIBLE_GAIN")),
        }
    overhead: dict[str, float | None] = {}
    for arm in ("before", "candidate"):
        c = [r["training_step_ms"] for r in valid if r.get("arm") == arm and r.get("model") == "control"]
        g = [r["training_step_ms"] for r in valid if r.get("arm") == arm and r.get("model") == "g1"]
        cm, gm = median(c), median(g)
        overhead[arm] = 100 * (gm / cm - 1) if cm and gm else None
    model_classes = [summaries[model]["classification"] for model in ("control", "g1")]
    if any(label == "INSUFFICIENT_VALID_PAIRS" for label in model_classes):
        overall_classification = "INSUFFICIENT_VALID_PAIRS"
    elif all(label == "LAND" for label in model_classes):
        overall_classification = "LAND"
    elif any((summaries[model]["median_gain_percent"] or 0) > 0 for model in ("control", "g1")):
        overall_classification = "CONDITION_DEPENDENT_GAIN"
    else:
        overall_classification = "NO_REPRODUCIBLE_GAIN"
    phase_distributions: dict[str, Any] = {}
    phase_names = ("qnn_execute_ms", "dsp_span_ms", "state_move_ms", "registry_validation_ms", "optimizer_wall_ms", "training_step_ms")
    for arm in ("before", "candidate"):
        phase_distributions[arm] = {}
        for model in ("control", "g1"):
            selected = [r for r in valid if r.get("arm") == arm and r.get("model") == model]
            phase_distributions[arm][model] = {}
            for phase in phase_names:
                values = [float(r[phase]) for r in selected if r.get(phase) is not None]
                phase_distributions[arm][model][phase] = {
                    "n": len(values), "median": median(values), "p25": percentile(values, .25), "p75": percentile(values, .75)
                }
    step_count_comparison = {}
    before_only_pairs: list[dict[str, Any]] = []
    if before_only:
        step_phases = ("qnn_execute_ms", "dsp_span_ms", "state_move_ms", "registry_validation_ms", "optimizer_wall_ms", "training_step_ms")
        for steps in sorted({int(r["steps"]) for r in valid if r.get("steps")}):
            selected = [r for r in valid if int(r.get("steps", 0)) == steps and r.get("training_step_ms") is not None]
            step_count_comparison[str(steps)] = {
                "n": len(selected),
                "phase_medians_ms_per_update": {
                    phase: median([float(r[phase]) for r in selected if r.get(phase) is not None])
                    for phase in step_phases
                },
                "run_ids_in_order": [r["run_id"] for r in sorted(selected, key=lambda x: x.get("run_order") or 0)],
            }
        step_pair_groups: dict[str, dict[int, dict[str, Any]]] = {}
        for row in valid:
            steps = int(row.get("steps", 0))
            pair_id = row.get("pair_id")
            if steps in (300, 400) and pair_id:
                step_pair_groups.setdefault(str(pair_id), {})[steps] = row
        for pair_id, step_rows in sorted(step_pair_groups.items()):
            if 300 not in step_rows or 400 not in step_rows:
                continue
            phase_comparison = {}
            for phase in step_phases:
                low, high = step_rows[300].get(phase), step_rows[400].get(phase)
                phase_comparison[phase] = {
                    "300_ms": low,
                    "400_ms": high,
                    "400_over_300_ratio": (high / low) if low is not None and high is not None and low != 0 else None,
                    "change_percent": (100.0 * (high / low - 1.0)) if low is not None and high is not None and low != 0 else None,
                }
            before_only_pairs.append({
                "pair_id": pair_id,
                "300_run_id": step_rows[300]["run_id"],
                "400_run_id": step_rows[400]["run_id"],
                "phase_comparison": phase_comparison,
            })
    temp_values: list[float] = []
    thermal_values: list[float] = []
    for row in rows:
        for key in ("battery_temperature_c_before", "battery_temperature_c_after"):
            if row.get(key) is not None:
                temp_values.append(float(row[key]))
        for key in ("thermal_status_before", "thermal_status_after"):
            if row.get(key) is not None:
                thermal_values.append(float(row[key]))
        manifest_path = row.get("manifest_path")
        if manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            for phase in ("pre", "mid", "post"):
                telemetry_path = Path(manifest_path).parent / "telemetry" / f"{phase}.json"
                if not telemetry_path.exists():
                    continue
                try:
                    tele = json.loads(telemetry_path.read_text(encoding="utf-8-sig"))
                    fields = tele.get("fields", {})
                    for k, v in fields.items():
                        if "battery_temperature" in k and str(v).replace('.', '', 1).isdigit():
                            temp_values.append(float(v) / (10.0 if k.endswith("deci_c") else 1.0))
                        if "thermal_status" in k:
                            m = re.search(r"\b([0-6])\b", str(v))
                            if m:
                                thermal_values.append(float(m.group(1)))
                except (OSError, json.JSONDecodeError):
                    pass
    correlation_samples: dict[str, list[tuple[float, float]]] = {"battery_temperature_c": [], "cpu_frequency_khz": [], "android_thermal_status": [], "run_order": [], "arm": [], "wall_elapsed_seconds": []}
    apk_identity_groups: dict[str, Any] = {}
    for arm in ("before", "candidate"):
        selected = [r for r in valid if r.get("arm") == arm]
        identities = {str(r.get("app_apk_sha256")) for r in selected if r.get("app_apk_sha256")}
        test_identities = {str(r.get("android_test_apk_sha256")) for r in selected if r.get("android_test_apk_sha256")}
        apk_identity_groups[arm] = {
            "run_count": len(selected), "distinct_app_apk_hashes": len(identities) if identities else "UNKNOWN",
            "distinct_android_test_apk_hashes": len(test_identities) if test_identities else "UNKNOWN",
            "median_training_step_ms": median([float(r["training_step_ms"]) for r in selected if r.get("training_step_ms") is not None]),
            "identity_hashes_published": False,
        }
    for row in valid:
        training = row.get("training_step_ms")
        if training is None:
            continue
        battery_values: list[float] = []
        frequency_values: list[float] = []
        thermal_values_for_run: list[float] = []
        manifest_path = row.get("manifest_path")
        if manifest_path:
            run_root = Path(manifest_path).parent
            for phase in ("pre", "mid", "post"):
                telemetry_path = run_root / "telemetry" / f"{phase}.json"
                if not telemetry_path.exists():
                    continue
                try:
                    fields = json.loads(telemetry_path.read_text(encoding="utf-8-sig")).get("fields", {})
                except (OSError, json.JSONDecodeError):
                    continue
                for key, value in fields.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        if "thermal_status" in key:
                            match = re.search(r"\b([0-6])\b", str(value))
                            if match:
                                thermal_values_for_run.append(float(match.group(1)))
                        continue
                    if not math.isfinite(numeric):
                        continue
                    if "battery_temperature_deci_c" in key:
                        battery_values.append(numeric / 10.0)
                    elif "battery_temperature_c" in key:
                        battery_values.append(numeric)
                    elif key.endswith("_cur_freq_khz"):
                        frequency_values.append(numeric)
                    elif "thermal_status" in key:
                        thermal_values_for_run.append(numeric)
        if not battery_values:
            for key in ("battery_temperature_c_before", "battery_temperature_c_after"):
                if row.get(key) is not None:
                    battery_values.append(float(row[key]))
        if battery_values:
            correlation_samples["battery_temperature_c"].append((statistics.mean(battery_values), float(training)))
        if frequency_values:
            correlation_samples["cpu_frequency_khz"].append((statistics.mean(frequency_values), float(training)))
        for key in ("thermal_status_before", "thermal_status_after"):
            if row.get(key) is not None:
                thermal_values_for_run.append(float(row[key]))
        if thermal_values_for_run:
            correlation_samples["android_thermal_status"].append((statistics.mean(thermal_values_for_run), float(training)))
        if row.get("run_order") is not None:
            correlation_samples["run_order"].append((float(row["run_order"]), float(training)))
        if row.get("arm") in ("before", "candidate"):
            correlation_samples["arm"].append((0.0 if row["arm"] == "before" else 1.0, float(training)))
        if manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
                started, finished = manifest.get("started_utc"), manifest.get("finished_utc")
                if started and finished:
                    elapsed = (datetime.fromisoformat(finished.replace("Z", "+00:00")) - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds()
                    if elapsed > 0:
                        correlation_samples["wall_elapsed_seconds"].append((elapsed, float(training)))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    correlations = {}
    for label, pairs_xy in correlation_samples.items():
        correlations[label] = {"n": len(pairs_xy), "pearson_r": pearson([p[0] for p in pairs_xy], [p[1] for p in pairs_xy])}
    return {
        "schema_version": 1,
        "headline_status": "DEVICE_AUDIT_PENDING" if not rows else "DEVICE_AUDIT_RESULTS_AVAILABLE",
        "performance_based_exclusion_allowed": False,
        "acceptance_rule": "At least 3 valid pairs per model, median gain >= 5%, and lower 95% bootstrap bound > 0 for both models => LAND; positive median with failed gate => CONDITION_DEPENDENT_GAIN; otherwise NO_REPRODUCIBLE_GAIN. This classification is evidence-local, not a causal claim.",
        "overall_classification": overall_classification,
        "classification_by_model": summaries,
        "phase_distributions_ms_per_update": phase_distributions,
        "before_only_step_count_comparison": step_count_comparison,
        "before_only_balanced_pairs": before_only_pairs,
        "exploratory_correlations": correlations,
        "apk_identity_groups": apk_identity_groups,
        "g1_vs_control_overhead_percent": overhead,
        "thermal_range": {"battery_temperature_c": [min(temp_values), max(temp_values)] if temp_values else "NOT_AVAILABLE",
                          "android_thermal_status": [min(thermal_values), max(thermal_values)] if thermal_values else "NOT_AVAILABLE"},
        "excluded_runs": excluded,
        "incomplete_pairs": incomplete_pairs,
        "run_count": len(rows),
        "valid_run_count": len(valid),
        "bootstrap_samples": 50000,
        "bootstrap_seed": seed,
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def legacy_summary(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for dataset in ("initial_profile", "matched"):
        for arm in ("before", "candidate"):
            for model in ("control", "g1"):
                selected = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm and r["model"] == model]
                summaries[f"{dataset}:{arm}:{model}"] = {phase: median([float(r[phase]) for r in selected if r.get(phase) is not None]) for phase in PHASE_FIELDS}
    ratios: dict[str, Any] = {}
    for model in ("control", "g1"):
        ratios[model] = {}
        for phase in PHASE_FIELDS:
            initial = summaries[f"initial_profile:before:{model}"].get(phase)
            matched = summaries[f"matched:before:{model}"].get(phase)
            ratios[model][phase] = matched / initial if initial and matched else None
    family_path = root / "family-final-uncertainty.json"
    negative = json.loads(family_path.read_text(encoding="utf-8-sig")) if family_path.exists() else None
    current_corr_rows = [r for r in rows if r.get("dataset") == "matched" and r.get("training_step_ms") is not None]
    elapsed_x = [float(r["training_total_seconds"]) for r in current_corr_rows if r.get("training_total_seconds") is not None]
    elapsed_y = [float(r["training_step_ms"]) for r in current_corr_rows if r.get("training_total_seconds") is not None]
    temps = [((float(r["battery_temperature_c_before"]) + float(r["battery_temperature_c_after"])) / 2)
             for r in current_corr_rows if r.get("battery_temperature_c_before") is not None and r.get("battery_temperature_c_after") is not None]
    temp_y = [float(r["training_step_ms"]) for r in current_corr_rows if r.get("battery_temperature_c_before") is not None and r.get("battery_temperature_c_after") is not None]
    return {
        "source": "existing saved reports; no new device run",
        "phase_medians_ms_per_update": summaries,
        "matched_before_over_initial_main_ratios": ratios,
        "negative_result_summary": negative,
        "exploratory_correlations": {"n_elapsed": len(elapsed_x), "training_step_vs_training_total_seconds": pearson(elapsed_x, elapsed_y),
                                     "elapsed_independence": "NOT_INDEPENDENT; legacy training_total_seconds and training_step_ms come from the same report timer and step count",
                                     "n_battery_temperature": len(temps), "training_step_vs_battery_temperature_c": pearson(temps, temp_y),
                                     "cpu_frequency_correlation": "NOT_AVAILABLE; no per-run CPU frequency telemetry"},
        "causal_claim": False,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    assert percentile([1.0, 3.0], .5) == 2.0
    assert bootstrap_median_ci([50.0, 55.0, 60.0], 1, 100)[0] <= 55.0
    fields = {"gradient_registry_validation_us": "800000", "muon_kernel_us": "160000"}
    assert derive_registry_ms(fields, 400) == 2.0
    assert derive_dsp_ms(fields, 400) == .4
    data = [{"arm":"before","model":"control","pair_id":f"{i}","run_id":f"b{i}","training_step_ms":100.0,"health_ok":True} for i in range(3)]
    data += [{"arm":"candidate","model":"control","pair_id":f"{i}","run_id":f"c{i}","training_step_ms":50.0,"health_ok":True} for i in range(3)]
    summary = audit_summary(data, [])
    assert summary["classification_by_model"]["control"]["classification"] == "LAND"
    assert summary["overall_classification"] == "INSUFFICIENT_VALID_PAIRS"
    complete_data = data + [{"arm":"before","model":"g1","pair_id":f"g{i}","run_id":f"gb{i}","training_step_ms":110.0,"health_ok":True} for i in range(3)]
    complete_data += [{"arm":"candidate","model":"g1","pair_id":f"g{i}","run_id":f"gc{i}","training_step_ms":55.0,"health_ok":True} for i in range(3)]
    assert audit_summary(complete_data, [])["overall_classification"] == "LAND"
    count_rows = [
        {"dataset":"BeforeOnly300vs400","arm":"before","model":"control","pair_id":"steps-pair-1","run_id":"s300","steps":300,"run_order":1,"training_step_ms":100.0,"health_ok":True,"qnn_execute_ms":30.0},
        {"dataset":"BeforeOnly300vs400","arm":"before","model":"control","pair_id":"steps-pair-1","run_id":"s400","steps":400,"run_order":2,"training_step_ms":105.0,"health_ok":True,"qnn_execute_ms":31.0},
    ]
    count_summary = audit_summary(count_rows, [])
    assert math.isclose(count_summary["before_only_balanced_pairs"][0]["phase_comparison"]["training_step_ms"]["change_percent"], 5.0)
    identity_rows = [
        {"dataset":"MatchedAB","arm":"before","app_apk_sha256":"before-hash","android_test_apk_sha256":"same-test","run_id":"id-b","health_ok":True,"exclusion_reasons":[]},
        {"dataset":"MatchedAB","arm":"candidate","app_apk_sha256":"candidate-hash","android_test_apk_sha256":"different-test","run_id":"id-c","health_ok":True,"exclusion_reasons":[]},
    ]
    identity_rows, identity_excluded = validate_matched_apk_identities(identity_rows, [])
    assert len(identity_excluded) == 2 and not any(row["health_ok"] for row in identity_rows)
    print("audit_analyzer_self_test=PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-root", type=Path)
    ap.add_argument("--legacy-root", type=Path)
    ap.add_argument("--output-dir", type=Path, required=False)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.output_dir:
        ap.error("--output-dir is required unless --self-test is used")
    if not args.audit_root and not args.legacy_root:
        ap.error("provide --audit-root and/or --legacy-root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    output: dict[str, Any] = {"schema_version": 1}
    if args.audit_root:
        device_rows, exclusions = load_device_rows(args.audit_root)
        normalized.extend(device_rows)
        output["device_audit"] = audit_summary(device_rows, exclusions)
    if args.legacy_root:
        legacy_rows = normalize_legacy(args.legacy_root)
        normalized.extend(legacy_rows)
        output["legacy_evidence"] = legacy_summary(legacy_rows, args.legacy_root)
    write_rows(args.output_dir / "normalized_runs.csv", normalized)
    (args.output_dir / "audit-summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "audit-summary.md").write_text(render_markdown(output), encoding="utf-8")
    print(f"normalized_runs={len(normalized)} excluded_runs={len(exclusions)} output={args.output_dir}")
    return 0


def render_markdown(output: dict[str, Any]) -> str:
    lines = ["# Training critical-path audit", "", "Private evidence summary. Performance alone never excludes a run.", ""]
    legacy = output.get("legacy_evidence")
    if legacy:
        lines += ["## Existing evidence", "", "| Phase (ms/update) | Initial main Control | Matched before Control | Ratio |", "| --- | ---: | ---: | ---: |"]
        phases = legacy["phase_medians_ms_per_update"]
        ratios = legacy["matched_before_over_initial_main_ratios"]["control"]
        for phase, label in (("qnn_execute_ms","QNN execute"),("dsp_span_ms","DSP span"),("state_move_ms","State move"),("registry_validation_ms","Registry validation"),("optimizer_wall_ms","Optimizer wall"),("training_step_ms","Training step")):
            initial = phases.get(f"initial_profile:before:control", {}).get(phase)
            matched = phases.get(f"matched:before:control", {}).get(phase)
            ratio = ratios.get(phase)
            lines.append(f"| {label} | {initial if initial is not None else 'UNKNOWN'} | {matched if matched is not None else 'UNKNOWN'} | {ratio if ratio is not None else 'UNKNOWN'} |")
        corr = legacy["exploratory_correlations"]
        lines += ["", f"Exploratory correlations only (small n): legacy total elapsed n={corr['n_elapsed']}, r={corr['training_step_vs_training_total_seconds']} ({corr['elapsed_independence']}); battery temperature n={corr['n_battery_temperature']}, r={corr['training_step_vs_battery_temperature_c']}. CPU frequency: {corr['cpu_frequency_correlation']}.", ""]
        negative = legacy.get("negative_result_summary")
        if negative:
            lines += ["Reverted validation family remains a saved negative result; no new experiment was run.", ""]
    device = output.get("device_audit")
    if device:
        if device.get("before_only_step_count_comparison"):
            lines += ["## Before-only 300/400 update audit", "", "| Updates | Runs | Median phase time (ms/update) |", "| ---: | ---: | --- |"]
            for steps, row in sorted(device["before_only_step_count_comparison"].items(), key=lambda item: int(item[0])):
                medians = row["phase_medians_ms_per_update"]
                lines.append(f"| {steps} | {row['n']} | {medians} |")
            lines += ["", "Balanced adjacent-pair phase changes (400 vs 300):", ""]
            for pair in device.get("before_only_balanced_pairs", []):
                changes = {phase: values["change_percent"] for phase, values in pair["phase_comparison"].items()}
                lines.append(f"- `{pair['pair_id']}` (`{pair['300_run_id']}` → `{pair['400_run_id']}`): `{changes}`")
            lines.append("")
        lines += ["## Device audit", "", f"Headline status: **{device['headline_status']}**", f"Acceptance classification: **{device['overall_classification']}**", "", "| Model | Valid pairs | Median gain | Paired speedup | Bootstrap CI | Classification |", "| --- | ---: | ---: | ---: | --- | --- |"]
        for model, row in device["classification_by_model"].items():
            ci = row.get("bootstrap_95_median_gain_percent")
            lines.append(f"| {model} | {row['valid_pairs']} | {row['median_gain_percent']} | {row['median_paired_speedup']} | {ci} | {row['classification']} |")
        lines += ["", f"G1 vs Control overhead (%): `{device['g1_vs_control_overhead_percent']}`", f"Thermal range: `{device['thermal_range']}`", f"Excluded runs: {len(device['excluded_runs'])}; incomplete pairs: {len(device['incomplete_pairs'])}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
