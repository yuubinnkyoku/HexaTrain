#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Split-level paired analyzer for the G1 1.5x multi-seed replication.

Design contract: docs/g1-1p5x-multiseed-3000.md

Inputs are primary reports only:

  <tree>/seed<N>/<arm>/eval256-step<step>-htp.txt
  <tree>/seed<N>/<arm>/seed<N>-l19-*-steps<steps>-result.txt
  <tree>/seed<N>/g1/gate-static-step<step>.txt

Outputs (aggregate, never primary evidence):

  quality-split-level.csv  gate-trajectory.csv  run-health.csv
  run-identity.csv         verdicts.csv         markdown summary on stdout

Modes:

  --selftest   recompute the committed seed-1 trees (stress grid 20 cells and
               the 1.5x full-8000 run) and assert they match the committed
               aggregates.  This is the regression guard for this analyzer.
  --tree DIR   analyze a multi-seed tree produced by
               scripts/run_g1_1p5x_multiseed.ps1.

The primary judgment is the per-seed split-level trajectory (R1-R5), not a
single horizon metric.  No significance test is attempted: the seed count is
at most three, so only sign consistency is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nicopedia_real_text_pipeline as pipeline  # noqa: E402  (splitmix / FNV SSOT)

SPLITS = ("validation", "development")
EVAL_METRICS = ("nll", "bits_per_utf8_byte", "top1", "top5", "mean_rank")
CANONICAL_ORDER_SEED = 20260806

ARM_IDENTITY = {
    "control": {"attention_gate": "none", "parameter_count": "758528",
                "checkpoint_format": "NPRTCKPTV4"},
    "g1": {"attention_gate": "headwise_g1_sigmoid", "parameter_count": "760960",
           "checkpoint_format": "NPRTCKPTV5"},
}

# The pre-registered 1.5x triple as the device reports it (float32 rendering).
# Steps 1-3000 must sit on the constant peak of the 8000-step schedule.
LR_IDENTITY = {
    "muon_lr": "0.007499999832",
    "aux_adam_lr": "0.003299999982",
    "learning_rate_peak": "0.003299999982",
    "learning_rate_target": "0.0001500000071",
    "learning_rate_schedule": "linear_decay",
    "learning_rate_decay_start_step": "4000",
    "learning_rate_decay_end_step": "8000",
    "learning_rate_schedule_total_steps": "8000",
}

GATE_HEAD_RE = re.compile(
    r"^(?P<prefix>[a-z_]+)_l(?P<layer>\d+)_h(?P<head>\d+)_"
    r"(?P<stat>mean|stddev|min|max|below_0_1_fraction|above_0_9_fraction)$")


class EvidenceError(RuntimeError):
    """A primary report is missing, malformed, or contradicts its identity."""


# --------------------------------------------------------------------------- #
# primary report parsing
# --------------------------------------------------------------------------- #

def parse_report(path: Path) -> dict[str, str]:
    """Read a ``key=value`` report.  First occurrence of a key wins: the
    trailing backend / api-trace dumps repeat some keys."""
    if not path.is_file():
        raise EvidenceError(f"REPORT_MISSING: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in values:
            continue
        values[key] = value.strip()
    if not values:
        raise EvidenceError(f"REPORT_EMPTY: {path}")
    return values


def require_str(values: dict[str, str], key: str, where: str) -> str:
    if key not in values:
        raise EvidenceError(f"FIELD_MISSING: {key} in {where}")
    return values[key]


def require_float(values: dict[str, str], key: str, where: str) -> float:
    raw = require_str(values, key, where)
    try:
        return float(raw)
    except ValueError as error:
        raise EvidenceError(f"FIELD_NOT_NUMBER: {key}={raw!r} in {where}") from error


def require_int(values: dict[str, str], key: str, where: str) -> int:
    raw = require_str(values, key, where)
    try:
        return int(raw)
    except ValueError as error:
        raise EvidenceError(f"FIELD_NOT_INTEGER: {key}={raw!r} in {where}") from error


def read_eval(path: Path) -> dict[str, object]:
    where = str(path)
    values = parse_report(path)
    splits: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        entry: dict[str, float] = {
            name: require_float(values, f"{split}_{name}", where)
            for name in EVAL_METRICS
        }
        entry["tokens"] = float(require_int(values, f"{split}_tokens", where))
        entry["utf8_bytes"] = float(
            require_int(values, f"{split}_target_utf8_bytes", where))
        entry["nonfinite_chunks"] = float(
            require_int(values, f"{split}_nonfinite_chunks", where))
        splits[split] = entry
    return {
        "path": path,
        "status": require_str(values, "status", where),
        "seed": require_int(values, "seed", where),
        "step": require_int(values, "checkpoint_step", where),
        "checkpoint_format": require_str(values, "checkpoint_format", where),
        "splits": splits,
        "balanced_bpb": 0.5 * (splits["validation"]["bits_per_utf8_byte"]
                               + splits["development"]["bits_per_utf8_byte"]),
        "health": {
            "qnn_return_code_success": require_str(
                values, "qnn_return_code_success", where),
            "output_tensors_finite": require_str(
                values, "output_tensors_finite", where),
            "checkpoint_finite": require_str(values, "checkpoint_finite", where),
            "cpu_fallback": require_str(values, "cpu_fallback", where),
            "nan_detected": require_str(values, "nan_detected", where),
            "inf_detected": require_str(values, "inf_detected", where),
            "graph_execute_failures": require_int(
                values, "api_trace_graph_execute_failure_count", where),
            "focus_takeover_count": require_int(values, "focus_takeover_count", where),
            "thermal_before": require_int(values, "android_thermal_status_before", where),
            "thermal_after": require_int(values, "android_thermal_status_after", where),
            "battery_before": require_int(values, "battery_level_before", where),
            "battery_after": require_int(values, "battery_level_after", where),
            "total_seconds": require_float(values, "evaluation_total_seconds", where),
        },
    }


def eval_at(arms: dict[str, dict[int, dict[str, object]]], arm: str, step: int) -> dict:
    if arm not in arms:
        raise EvidenceError(f"ARM_MISSING: {arm}")
    if step not in arms[arm]:
        raise EvidenceError(f"EVAL_MISSING: {arm} step {step}")
    return arms[arm][step]


def read_training_result(path: Path) -> dict[str, object]:
    where = str(path)
    values = parse_report(path)
    identity_keys = (
        "status", "seed", "steps", "completed_steps", "run_completed_steps", "batch_size",
        "dataset_hash", "training_order_seed", "training_order_hash",
        "attention_gate", "parameter_count", "checkpoint_format",
        "muon_lr", "aux_adam_lr", "learning_rate_peak", "learning_rate_target",
        "learning_rate_schedule", "learning_rate_decay_start_step",
        "learning_rate_decay_end_step", "learning_rate_schedule_total_steps",
        "muon_momentum", "muon_ns_steps", "optimizer_muon_backend",
        "initial_parameter_hash", "final_parameter_hash",
    )
    identity = {key: require_str(values, key, where) for key in identity_keys}
    identity["path"] = path
    identity["health"] = {
        "all_steps_finite": require_str(values, "all_steps_finite", where),
        "final_finite": require_str(values, "final_finite", where),
        "output_tensors_finite": require_str(values, "output_tensors_finite", where),
        "qnn_return_code_success": require_str(values, "qnn_return_code_success", where),
        "cpu_fallback": require_str(values, "cpu_fallback", where),
        "nan_detected": require_str(values, "nan_detected", where),
        "inf_detected": require_str(values, "inf_detected", where),
        "graph_execute_failures": require_int(
            values, "api_trace_graph_execute_failure_count", where),
        "hvx_rpc_failure_count": require_int(values, "hvx_rpc_failure_count", where),
        "hvx_fallback_count": require_int(values, "hvx_fallback_count", where),
        "hvx_nonfinite_count": require_int(values, "hvx_nonfinite_count", where),
        "focus_takeover_count": require_int(values, "focus_takeover_count", where),
        "thermal_before": require_int(values, "android_thermal_status_before", where),
        "thermal_after": require_int(values, "android_thermal_status_after", where),
        "battery_before": require_int(values, "battery_level_before", where),
        "battery_after": require_int(values, "battery_level_after", where),
        "training_total_seconds": require_float(values, "training_total_seconds", where),
    }
    return identity


def aggregate_gate(path: Path, prefix: str) -> dict[str, object]:
    """Collapse the per-head gate statistics of one report into one row."""
    where = str(path)
    values = parse_report(path)
    heads: dict[tuple[int, int], dict[str, float]] = {}
    for key, raw in values.items():
        match = GATE_HEAD_RE.match(key)
        if not match or match.group("prefix") != prefix:
            continue
        slot = (int(match.group("layer")), int(match.group("head")))
        heads.setdefault(slot, {})[match.group("stat")] = float(raw)
    if not heads:
        raise EvidenceError(f"GATE_HEADS_MISSING: {prefix} in {where}")
    for slot_id, slot in heads.items():
        for name in ("mean", "below_0_1_fraction", "above_0_9_fraction"):
            if name not in slot:
                raise EvidenceError(
                    f"GATE_STAT_MISSING: {prefix}_l{slot_id[0]}_h{slot_id[1]}_"
                    f"{name} in {where}")
    means = [slot["mean"] for slot in heads.values()]
    below = [slot["below_0_1_fraction"] for slot in heads.values()]
    above = [slot["above_0_9_fraction"] for slot in heads.values()]
    return {
        "path": path,
        "heads": len(heads),
        "mean_of_head_means": sum(means) / len(means),
        "min_head_mean": min(means),
        "max_head_mean": max(means),
        "mean_below_0_1": sum(below) / len(below),
        "max_above_0_9": max(above),
        "heads_fully_suppressed": sum(1 for value in below if value >= 0.999),
    }



# --------------------------------------------------------------------------- #
# tree loading and paired rows
# --------------------------------------------------------------------------- #

EVAL_FILE_RE = re.compile(r"^eval256-step\d+-htp\.txt$")
RESULT_FILE_RE = re.compile(r"^seed\d+-l\d+-.*-result\.txt$")
GATE_FILE_RE = re.compile(r"^gate-static-step(\d+)\.txt$")

# Pre-registered fresh seeds of docs/g1-1p5x-multiseed-3000.md.  Seed 1 is the
# committed reference trajectory, not a replication sample.  Any other seed is
# exploratory and must never reach the pre-registered decision.
REFERENCE_SEED = 1
PREREGISTERED_SEEDS = (2, 4)
SEED_REGISTRY_FILE = "seed-registry.json"
SEED_ROLES = ("preregistered", "exploratory", "reference")


def default_seed_role(seed: int) -> str:
    if seed == REFERENCE_SEED:
        return "reference"
    return "preregistered" if seed in PREREGISTERED_SEEDS else "exploratory"


def load_seed_roles(tree: Path) -> tuple[dict[int, dict[str, str]], str]:
    """Classify every seed directory as pre-registered / exploratory / reference.

    The run registry written by scripts/run_g1_1p5x_multiseed.ps1 is the source
    of truth once it exists.  A seed directory that the registry does not list is
    treated as exploratory (fail closed on provenance): adding a seed is a
    protocol event, so it has to be recorded, not inferred.  Without a registry
    only the pre-registered pair 2 / 4 is trusted.
    """
    registry_path = tree / SEED_REGISTRY_FILE
    roles: dict[int, dict[str, str]] = {}
    source = "default"
    if registry_path.is_file():
        source = "registry"
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(
                f"SEED_REGISTRY_INVALID: {registry_path}: {error}") from error
        entries = data.get("seeds") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            raise EvidenceError(
                f"SEED_REGISTRY_INVALID: {registry_path} has no seeds list")
        for entry in entries:
            if not isinstance(entry, dict) or "seed" not in entry or "role" not in entry:
                raise EvidenceError(
                    f"SEED_REGISTRY_INVALID: {registry_path} entry {entry!r}")
            role = str(entry["role"])
            if role not in SEED_ROLES:
                raise EvidenceError(
                    f"SEED_REGISTRY_INVALID: {registry_path} role {role!r}")
            seed = int(entry["seed"])
            roles[seed] = {"role": role, "source": source}
    for seed in discover_seeds(tree):
        if seed in roles:
            continue
        if source == "registry":
            role = "reference" if seed == REFERENCE_SEED else "exploratory"
            roles[seed] = {"role": role, "source": "unlisted"}
        else:
            roles[seed] = {"role": default_seed_role(seed), "source": source}
    return roles, source


def discover_seeds(tree: Path) -> list[int]:
    seeds: list[int] = []
    for entry in sorted(tree.iterdir()) if tree.is_dir() else []:
        match = re.fullmatch(r"seed(\d+)", entry.name)
        if entry.is_dir() and match:
            seeds.append(int(match.group(1)))
    if not seeds:
        raise EvidenceError(f"NO_SEED_DIRECTORIES: {tree}")
    return seeds


def load_arm_evals(arm_dir: Path) -> dict[int, dict[str, object]]:
    if not arm_dir.is_dir():
        raise EvidenceError(f"ARM_DIR_MISSING: {arm_dir}")
    evals: dict[int, dict[str, object]] = {}
    for path in sorted(arm_dir.iterdir()):
        if not EVAL_FILE_RE.match(path.name):
            continue
        record = read_eval(path)
        step = int(record["step"])
        if step in evals:
            raise EvidenceError(f"DUPLICATE_CHECKPOINT_STEP: {step} in {arm_dir}")
        evals[step] = record
    if not evals:
        raise EvidenceError(f"NO_EVAL_REPORTS: {arm_dir}")
    return evals


def load_result(arm_dir: Path) -> dict[str, object]:
    matches = [p for p in sorted(arm_dir.iterdir()) if RESULT_FILE_RE.match(p.name)]
    if len(matches) != 1:
        raise EvidenceError(
            f"TRAINING_RESULT_CARDINALITY: expected 1 got {len(matches)} in {arm_dir}")
    return read_training_result(matches[0])


def load_gate_series(g1_dir: Path) -> dict[int, dict[str, object]]:
    series: dict[int, dict[str, object]] = {}
    for path in sorted(g1_dir.iterdir()):
        match = GATE_FILE_RE.match(path.name)
        if not match:
            continue
        record = aggregate_gate(path, "gate_static")
        record["step"] = int(match.group(1))
        series[record["step"]] = record
    return series


def paired_row(seed: int, control: dict[str, object], g1: dict[str, object]) -> dict[str, object]:
    """One G1 - control comparison at one checkpoint, per split."""
    if control["step"] != g1["step"]:
        raise EvidenceError(
            f"STEP_MISMATCH: {control['step']} vs {g1['step']}")
    row: dict[str, object] = {"seed": seed, "step": control["step"]}
    for split, tag in (("validation", "val"), ("development", "dev")):
        base = control["splits"][split]  # type: ignore[index]
        other = g1["splits"][split]  # type: ignore[index]
        if base["tokens"] != other["tokens"]:
            raise EvidenceError(
                f"SPLIT_SUPPORT_MISMATCH: {split} step {control['step']} "
                f"tokens {base['tokens']} vs {other['tokens']}")
        if base["utf8_bytes"] != other["utf8_bytes"]:
            raise EvidenceError(
                f"SPLIT_SUPPORT_MISMATCH: {split} step {control['step']} bytes "
                f"{base['utf8_bytes']} vs {other['utf8_bytes']}")
        for metric, key in (("bits_per_utf8_byte", "bpb"), ("nll", "nll")):
            row[f"{tag}_{key}_control"] = base[metric]
            row[f"{tag}_{key}_g1"] = other[metric]
            row[f"{tag}_{key}_delta"] = other[metric] - base[metric]
        row[f"{tag}_top1_token_delta"] = round((other["top1"] - base["top1"]) * base["tokens"])
        row[f"{tag}_top5_token_delta"] = round((other["top5"] - base["top5"]) * base["tokens"])
        row[f"{tag}_mean_rank_delta"] = other["mean_rank"] - base["mean_rank"]
    row["balanced_bpb_delta"] = g1["balanced_bpb"] - control["balanced_bpb"]  # type: ignore[operator]
    return row


def paired_series(seed: int, control: dict[int, dict], g1: dict[int, dict]) -> list[dict]:
    common = sorted(set(control) & set(g1))
    if not common:
        raise EvidenceError(f"NO_COMMON_CHECKPOINTS: seed {seed}")
    return [paired_row(seed, control[step], g1[step]) for step in common]



# --------------------------------------------------------------------------- #
# identity / health consistency
# --------------------------------------------------------------------------- #

SHARED_IDENTITY_KEYS = (
    "batch_size", "dataset_hash", "training_order_seed", "training_order_hash",
    "muon_lr", "aux_adam_lr", "learning_rate_peak", "learning_rate_target",
    "learning_rate_schedule", "learning_rate_decay_start_step",
    "learning_rate_decay_end_step", "learning_rate_schedule_total_steps",
    "muon_momentum", "muon_ns_steps", "optimizer_muon_backend",
)

HEALTH_EXPECTATIONS = {
    "qnn_return_code_success": {"true"},
    "output_tensors_finite": {"true"},
    "cpu_fallback": {"false", "0"},
    "nan_detected": {"false", "0"},
    "inf_detected": {"false", "0"},
}


def identity_problems(runs: list[dict[str, object]]) -> list[str]:
    """Cross-run identity assertions: runs may differ only in seed."""
    problems: list[str] = []
    for run in runs:
        arm, seed = str(run["arm"]), int(run["seed_dir"])
        result = run["result"]  # type: ignore[assignment]
        if int(result["seed"]) != seed:
            problems.append(
                f"{arm}(seed {seed}): report seed {result['seed']} != directory seed")
        if str(result["status"]).upper() != "SUCCESS":
            problems.append(f"{arm}(seed {seed}): status {result['status']}")
        for key, expected in LR_IDENTITY.items():
            if str(result[key]) != expected:
                problems.append(
                    f"{arm}(seed {seed}): {key}={result[key]} expected {expected}")
        if int(result["steps"]) < REVERSAL_MAX_STEP:
            problems.append(
                f"{arm}(seed {seed}): steps {result['steps']} < "
                f"{REVERSAL_MAX_STEP}, the R2 band cannot be observed")
        if int(result["completed_steps"]) != int(result["steps"]):
            problems.append(
                f"{arm}(seed {seed}): completed_steps {result['completed_steps']} "
                f"!= steps {result['steps']}")
        for key, expected in ARM_IDENTITY[arm].items():
            if str(result[key]) != expected:
                problems.append(
                    f"{arm}(seed {seed}): {key}={result[key]} expected {expected}")
        health = result["health"]
        if str(health["all_steps_finite"]) != "true":
            problems.append(f"{arm}(seed {seed}): all_steps_finite != true")
        if int(health["hvx_rpc_failure_count"]) > 0 or int(health["hvx_nonfinite_count"]) > 0:
            problems.append(
                f"{arm}(seed {seed}): hvx_rpc_failure_count="
                f"{health['hvx_rpc_failure_count']} hvx_nonfinite_count="
                f"{health['hvx_nonfinite_count']}")
        if int(health["focus_takeover_count"]) > 0:
            problems.append(f"{arm}(seed {seed}): focus_takeover_count > 0")
        if health["thermal_before"] != health["thermal_after"]:
            problems.append(
                f"{arm}(seed {seed}): thermal {health['thermal_before']} -> "
                f"{health['thermal_after']}")
    reference = runs[0]["result"]
    for run in runs[1:]:
        result = run["result"]  # type: ignore[assignment]
        for key in SHARED_IDENTITY_KEYS:
            if str(result[key]) != str(reference[key]):
                problems.append(
                    f"{run['arm']}(seed {run['seed_dir']}): {key}={result[key]} "
                    f"differs from {runs[0]['arm']}(seed {runs[0]['seed_dir']}) "
                    f"{reference[key]}")
    if str(reference["training_order_seed"]) != str(CANONICAL_ORDER_SEED):
        problems.append(
            f"training_order_seed {reference['training_order_seed']} != "
            f"canonical {CANONICAL_ORDER_SEED}")
    return problems


def health_problems(runs: list[dict[str, object]]) -> list[str]:
    problems: list[str] = []
    for run in runs:
        arm, seed = str(run["arm"]), run["seed_dir"]
        for step, record in sorted(run["evals"].items()):  # type: ignore[union-attr]
            health = record["health"]
            if record["status"].upper() != "SUCCESS":
                problems.append(
                    f"{arm}(seed {seed}) step {step}: eval status {record['status']}")
            for key, allowed in HEALTH_EXPECTATIONS.items():
                if str(health[key]).lower() not in allowed:
                    problems.append(
                        f"{arm}(seed {seed}) step {step}: {key}={health[key]}")
            for split in SPLITS:
                if record["splits"][split]["nonfinite_chunks"] != 0:
                    problems.append(
                        f"{arm}(seed {seed}) step {step}: {split} nonfinite_chunks="
                        f"{record['splits'][split]['nonfinite_chunks']}")
            if int(health["graph_execute_failures"]) > 0:
                problems.append(
                    f"{arm}(seed {seed}) step {step}: graph_execute_failures="
                    f"{health['graph_execute_failures']}")
    return problems


# --------------------------------------------------------------------------- #
# pre-registered criteria R1 - R5
# --------------------------------------------------------------------------- #

EARLY_MAX_STEP = 1000
REVERSAL_MIN_STEP = 1500
REVERSAL_MAX_STEP = 3000


def verdicts_for_seed(rows: list[dict], gate: dict[int, dict]) -> list[dict[str, object]]:
    """Apply the pre-registered criteria of docs/g1-1p5x-multiseed-3000.md.

    The criteria are computed for every seed so that an exploratory seed is still
    fully readable, but each row carries the seed role and decision() consumes the
    pre-registered rows only.
    """
    if not rows:
        raise EvidenceError("NO_PAIRED_ROWS")
    seed = rows[0]["seed"]
    role = str(rows[0].get("seed_role", "unknown"))
    early = [row for row in rows if row["step"] <= EARLY_MAX_STEP]
    late = [row for row in rows
            if REVERSAL_MIN_STEP <= row["step"] <= REVERSAL_MAX_STEP]
    if not early:
        raise EvidenceError(f"NO_EARLY_CHECKPOINTS: seed {seed}")
    if not late:
        raise EvidenceError(
            f"NO_LATE_CHECKPOINTS: seed {seed} has no paired evaluation inside "
            f"the R2 band {REVERSAL_MIN_STEP}-{REVERSAL_MAX_STEP}; R2 is left "
            "undefined instead of defaulting to not_reproduced")

    r1_hits = [row["step"] for row in early
               if row["val_bpb_delta"] < 0 and row["dev_bpb_delta"] < 0]
    r1_pass = len(r1_hits) * 5 >= len(early) * 3  # >= 60 % of early checkpoints
    reversals = [row["step"] for row in late if row["dev_nll_delta"] > 0]

    out: list[dict[str, object]] = [
        {
            "seed": seed,
            "criterion": "R1",
            "verdict": "pass" if r1_pass else "fail",
            "value": f"{len(r1_hits)}/{len(early)}",
            "detail": f"steps {sorted(r1_hits)} lower both Val and Dev bpb; "
                      f"threshold 60% of steps <= {EARLY_MAX_STEP}",
        },
        {
            "seed": seed,
            "criterion": "R2",
            "verdict": "reproduced" if reversals else "not_reproduced",
            "value": f"{len(reversals)}/{len(late)}",
            "detail": f"Dev NLL delta > 0 at steps {reversals}; band "
                      f"{REVERSAL_MIN_STEP}-{REVERSAL_MAX_STEP}",
        },
        {
            "seed": seed,
            "criterion": "R3",
            "verdict": "located" if reversals else "none",
            "value": ",".join(str(step) for step in reversals) or "none",
            "detail": "reversal step set; the R1 early band must stay unaffected",
        },
    ]

    for step in reversals:
        row = next(item for item in rows if item["step"] == step)
        both_lower = (row["val_top1_token_delta"] < 0
                      and row["dev_top1_token_delta"] < 0)
        out.append({
            "seed": seed,
            "criterion": "R4",
            "verdict": "quality_degraded" if both_lower else "mixed",
            "value": f"val{row['val_top1_token_delta']:+d} "
                     f"dev{row['dev_top1_token_delta']:+d} top-1 tokens",
            "detail": f"step {step}: exact top-k token counts, no rounding threshold",
        })
        suppressed = gate.get(step, {}).get("heads_fully_suppressed")
        lower = [adj for adj in gate if adj < step]
        upper = [adj for adj in gate if adj > step]
        neighbours, neighbour_steps = [], []
        if lower:
            neighbour_steps.append(max(lower))
            neighbours.append(gate[max(lower)]["heads_fully_suppressed"])
        if upper:
            neighbour_steps.append(min(upper))
            neighbours.append(gate[min(upper)]["heads_fully_suppressed"])
        if suppressed is None or not neighbours:
            r5_verdict, r5_value = "insufficient_data", "gate series incomplete"
        elif suppressed > max(neighbours):
            r5_verdict, r5_value = "peak_at_reversal", f"{suppressed} > {max(neighbours)}"
        else:
            r5_verdict, r5_value = "no_peak", f"{suppressed} <= {max(neighbours)}"
        out.append({
            "seed": seed,
            "criterion": "R5",
            "verdict": r5_verdict,
            "value": r5_value,
            "detail": f"step {step}: heads with g<0.1 fraction >= 0.999 against the "
                      f"nearest gate checkpoints {sorted(neighbour_steps)} "
                      "(descriptive, never causal)",
        })
    for row in out:
        row["seed_role"] = role
    return out


def decision(verdict_rows: list[dict]) -> tuple[str, str]:
    """R2 replication over the pre-registered fresh seeds drives the lane decision.

    Verdict rows carry the seed role.  Exploratory seeds (anything outside the
    pre-registered 2 / 4 pair) and the seed-1 reference are reported but never
    enter this decision: adding a seed after seeing results must not be able to
    move the pre-registered conclusion.
    """
    usable = [row for row in verdict_rows
              if row.get("seed_role") == "preregistered"]
    if not usable:
        return ("preregistered_incomplete",
                "no pre-registered seed in this tree, so the R1-R5 decision is "
                f"withheld; the pre-registered seeds are {PREREGISTERED_SEEDS}. "
                "Exploratory seeds are reported separately and are not evidence "
                "for the pre-registered criteria.")
    seeds = sorted({row["seed"] for row in usable})
    hits = sorted({row["seed"] for row in usable
                   if row["criterion"] == "R2" and row["verdict"] == "reproduced"})
    total = len(seeds)
    if len(hits) == total:
        return ("close_high_lr_quality_lane",
                f"R2 reproduced in {len(hits)}/{total} pre-registered fresh seeds: "
                "G1 at 1.5x is an early sample-efficiency component inside this "
                "split pair; the high-LR quality lane closes and the Val/Dev "
                "disagreement becomes a documented selection hazard.")
    if not hits:
        return ("close_split_lane",
                f"R2 reproduced in 0/{total} pre-registered fresh seeds: the seed-1 "
                "reversal was a single-seed trajectory; the committed G1 1.5x gains "
                "stand and the split-disagreement lane closes.")
    return ("ambiguous_tie_breaker",
            f"R2 reproduced in {len(hits)}/{total} pre-registered fresh seeds "
            f"(seeds {hits}): ambiguous.  The documented tie-breaker is one seed 3 "
            "run via -AllowExploratorySeed; the analyzer reports it as exploratory "
            "and keeps it outside this decision, so the tie is broken by an "
            "explicit human call, not by an automatic merge.")




# --------------------------------------------------------------------------- #
# tree analysis
# --------------------------------------------------------------------------- #

QUALITY_FIELDS = (
    "seed", "seed_role", "step", "val_bpb_control", "val_bpb_g1", "val_bpb_delta",
    "dev_bpb_control", "dev_bpb_g1", "dev_bpb_delta", "balanced_bpb_delta",
    "val_nll_control", "val_nll_g1", "val_nll_delta",
    "dev_nll_control", "dev_nll_g1", "dev_nll_delta",
    "val_top1_token_delta", "val_top5_token_delta",
    "dev_top1_token_delta", "dev_top5_token_delta",
    "val_mean_rank_delta", "dev_mean_rank_delta",
)
GATE_FIELDS = (
    "seed", "seed_role", "step", "heads", "mean_of_head_means", "min_head_mean",
    "max_head_mean", "mean_below_0_1", "max_above_0_9", "heads_fully_suppressed",
)
HEALTH_FIELDS = (
    "kind", "seed", "seed_role", "arm", "step", "status", "qnn_return_code_success",
    "output_tensors_finite", "checkpoint_finite", "all_steps_finite", "final_finite",
    "cpu_fallback", "nan_detected", "inf_detected", "graph_execute_failures",
    "hvx_rpc_failure_count", "hvx_fallback_count", "hvx_nonfinite_count",
    "focus_takeover_count", "thermal_before", "thermal_after",
    "battery_before", "battery_after", "total_seconds",
)
IDENTITY_FIELDS = (
    "seed", "seed_role", "arm", "steps", "completed_steps", "run_completed_steps", "batch_size",
    "dataset_hash", "training_order_seed", "training_order_hash",
    "muon_lr", "aux_adam_lr", "learning_rate_peak", "learning_rate_target",
    "learning_rate_schedule", "learning_rate_decay_start_step",
    "learning_rate_decay_end_step", "learning_rate_schedule_total_steps",
    "muon_momentum", "muon_ns_steps", "optimizer_muon_backend",
    "attention_gate", "parameter_count", "checkpoint_format",
    "initial_parameter_hash", "final_parameter_hash",
)
VERDICT_FIELDS = ("seed", "seed_role", "criterion", "verdict", "value", "detail")


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_runs(tree: Path, roles: dict[int, dict[str, str]]) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for seed in discover_seeds(tree):
        role = roles[seed]
        for arm in ("control", "g1"):
            arm_dir = tree / f"seed{seed}" / arm
            runs.append({
                "seed_dir": seed,
                "arm": arm,
                "role": role["role"],
                "role_source": role["source"],
                "evals": load_arm_evals(arm_dir),
                "result": load_result(arm_dir),
                "gate": load_gate_series(arm_dir) if arm == "g1" else {},
            })
    if not runs:
        raise EvidenceError(f"NO_RUNS_FOUND: {tree}")
    return runs


def health_rows(runs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        result = run["result"]
        training = dict(result["health"])
        training["total_seconds"] = training.pop("training_total_seconds")
        rows.append({"kind": "training", "seed": run["seed_dir"],
                     "seed_role": run["role"], "arm": run["arm"],
                     "step": result["completed_steps"],
                     "status": result["status"], **training})
        for step, record in sorted(run["evals"].items()):
            rows.append({"kind": "eval", "seed": run["seed_dir"],
                         "seed_role": run["role"], "arm": run["arm"],
                         "step": step, "status": record["status"],
                         **record["health"]})
    return rows


def identity_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        row = {key: value for key, value in run["result"].items()
               if key not in ("path", "health")}
        row["seed"] = run["seed_dir"]
        row["seed_role"] = run["role"]
        row["arm"] = run["arm"]
        rows.append(row)
    return rows



def print_paired_table(rows: list[dict], gates: dict[int, dict[int, dict]],
                       with_role: bool) -> None:
    role_head = " | role" if with_role else ""
    role_sep = " | :---" if with_role else ""
    print(f"| seed{role_head} | step | d Val bpb | d Dev bpb | d Val NLL | d Dev NLL | "
          "d Val top-1 | d Dev top-1 | sup. heads |")
    print(f"| ---:{role_sep} | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        gate = gates.get(row["seed"], {})
        suppressed = gate.get(row["step"], {}).get("heads_fully_suppressed", "")
        role_cell = f" {row['seed_role']} |" if with_role else ""
        print(f"| {row['seed']}{role_cell} {row['step']} | {row['val_bpb_delta']:+.6f} | "
              f"{row['dev_bpb_delta']:+.6f} | {row['val_nll_delta']:+.6f} | "
              f"{row['dev_nll_delta']:+.6f} | {row['val_top1_token_delta']:+d} | "
              f"{row['dev_top1_token_delta']:+d} | {suppressed} |")


def print_markdown(rows: list[dict], gates: dict[int, dict[int, dict]],
                   verdicts: list[dict], code: str, text: str,
                   roles: dict[int, dict[str, str]]) -> None:
    preregistered = [row for row in rows if row["seed_role"] == "preregistered"]
    other = [row for row in rows if row["seed_role"] != "preregistered"]
    covered = sorted({row["seed"] for row in preregistered})
    if preregistered:
        print("### pre-registered fresh seeds")
        print_paired_table(preregistered, gates, with_role=False)
        print()
        print(f"pre-registered coverage: seeds {covered} of {list(PREREGISTERED_SEEDS)}")
    else:
        print("### pre-registered fresh seeds")
        print(f"none present; the pre-registered seeds are {list(PREREGISTERED_SEEDS)}")
        print("R1-R5 decision withheld: a replication claim needs a pre-registered seed.")
    if other:
        described = ", ".join(
            f"seed {seed} ({roles[seed]['role']}, source={roles[seed]['source']})"
            for seed in sorted({row["seed"] for row in other}))
        print()
        print("### exploratory / reference seeds (excluded from the R1-R5 decision)")
        print(described)
        print_paired_table(other, gates, with_role=True)
    print()
    print("| seed | role | criterion | verdict | value |")
    print("| ---: | :--- | :--- | :--- | :--- |")
    for row in verdicts:
        print(f"| {row['seed']} | {row['seed_role']} | {row['criterion']} | "
              f"{row['verdict']} | {row['value']} |")
    print()
    print(f"decision: {code} (pre-registered seeds only)")
    print(text)


def analyze_tree(tree: Path, out: Path, strict: bool) -> int:
    roles, role_source = load_seed_roles(tree)
    runs = load_runs(tree, roles)
    problems = identity_problems(runs) + health_problems(runs)

    all_rows: list[dict] = []
    gate_rows: list[dict] = []
    verdict_rows: list[dict] = []
    gates: dict[int, dict[int, dict]] = {}
    for seed in sorted({run["seed_dir"] for run in runs}):
        role = roles[seed]["role"]
        control = next(run for run in runs
                       if run["seed_dir"] == seed and run["arm"] == "control")
        g1 = next(run for run in runs
                  if run["seed_dir"] == seed and run["arm"] == "g1")
        rows = paired_series(seed, control["evals"], g1["evals"])  # type: ignore[arg-type]
        for row in rows:
            row["seed_role"] = role
        all_rows += rows
        gate = g1["gate"]  # type: ignore[assignment]
        gates[seed] = gate
        for step, aggregate in sorted(gate.items()):
            gate_rows.append({"seed": seed, "seed_role": role, "step": step,
                              **aggregate})
        verdict_rows += verdicts_for_seed(rows, gate)

    preregistered = sorted({row["seed"] for row in verdict_rows
                            if row["seed_role"] == "preregistered"})
    exploratory = sorted({row["seed"] for row in verdict_rows
                          if row["seed_role"] == "exploratory"})
    notes = [f"seed_role_source={role_source}",
             f"preregistered_seeds={list(PREREGISTERED_SEEDS)}",
             f"preregistered_coverage={preregistered}"]
    if exploratory:
        notes.append(
            f"EXPLORATORY_SEED_EXCLUDED seeds={exploratory} — outside the "
            "pre-registered R1-R5 decision")
    missing = [seed for seed in PREREGISTERED_SEEDS if seed not in preregistered]
    if missing:
        notes.append(f"PREREGISTERED_SEED_MISSING seeds={missing} — decision is "
                     "based on the pre-registered seeds present above")

    code, text = decision(verdict_rows)
    write_csv(out / "quality-split-level.csv", all_rows, QUALITY_FIELDS)
    write_csv(out / "gate-trajectory.csv", gate_rows, GATE_FIELDS)
    write_csv(out / "run-health.csv", health_rows(runs), HEALTH_FIELDS)
    write_csv(out / "run-identity.csv", identity_rows(runs), IDENTITY_FIELDS)
    write_csv(out / "verdicts.csv", verdict_rows + [{
        "seed": "", "seed_role": "", "criterion": "decision", "verdict": code,
        "value": "", "detail": text}], VERDICT_FIELDS)
    print_markdown(all_rows, gates, verdict_rows, code, text, roles)
    print(f"\nrows={len(all_rows)} gate={len(gate_rows)} "
          f"verdicts={len(verdict_rows)} problems={len(problems)}")
    for note in notes:
        print(f"NOTE: {note}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems and strict:
        print("STRICT: identity or health problems -> result not claimable")
        return 2
    return 0


# --------------------------------------------------------------------------- #
# training-order derivation regression (host side, no device needed)
# --------------------------------------------------------------------------- #

MASK64 = (1 << 64) - 1


def order_indices(record_count: int, steps: int, batch_size: int,
                  order_seed: int = CANONICAL_ORDER_SEED) -> list[int]:
    """Mirror of pipeline.training_order_identity; used to prove prefix stability."""
    state = order_seed
    indices: list[int] = []
    for index in range(steps * batch_size):
        state = pipeline.split_mix((state + index) & MASK64)
        indices.append(state % record_count)
    return indices


def order_hash(indices: list[int]) -> str:
    value = pipeline.FNV_OFFSET
    for selected in indices:
        value = pipeline.fnv_update(value, selected.to_bytes(8, "little"))
    return f"fnv1a64:{value:016x}"


def order_prefix_checks() -> list[str]:
    """A shorter horizon must be a prefix of a longer one: a fresh 3000-step run
    replays the same batch sequence as a longer run, independent of horizon."""
    problems: list[str] = []
    for record_count in (64000, 46616):
        short = order_indices(record_count, 300, 8)
        long = order_indices(record_count, 900, 8)
        if long[:len(short)] != short:
            problems.append(f"ORDER_NOT_PREFIX: record_count={record_count}")
        if order_hash(long[:500 * 8]) != pipeline.training_order_identity(
                record_count, 500, 8, CANONICAL_ORDER_SEED):
            problems.append(f"ORDER_HASH_MISMATCH: record_count={record_count}")
        if pipeline.training_order_identity(
                record_count, 3000, 8, CANONICAL_ORDER_SEED) != order_hash(
                    order_indices(record_count, 3000, 8)):
            problems.append(f"ORDER_3000_MISMATCH: record_count={record_count}")
    return problems


def load_committed_pairs(path: Path,
                         use_lr: bool = False) -> dict[tuple[str, int], dict[str, float]]:
    pairs: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["lr_multiplier"] if use_lr else "", int(row["step"]))
            pairs[key] = {name: float(row[name]) for name in
                          ("delta_val", "delta_dev", "delta_balanced")}
    return pairs


def recompute_pairs(tree: Path,
                    lr: str = "") -> dict[tuple[str, int], dict[str, float]]:
    rows = paired_series(1, load_arm_evals(tree / "control"), load_arm_evals(tree / "g1"))
    return {(lr, row["step"]): {"delta_val": row["val_bpb_delta"],
                                "delta_dev": row["dev_bpb_delta"],
                                "delta_balanced": row["balanced_bpb_delta"]}
            for row in rows}


def check_close(problems: list[str], label: str, got: float, want: float,
                tolerance: float) -> None:
    if abs(got - want) > tolerance:
        problems.append(f"{label}: computed {got!r} committed {want!r}")



GATE_STEP500_ANCHORS = {
    "lr1.0": (0.1541, 0.0291, 0.5290, 0.4655, 0),
    "lr1.25": (0.1311, 0.0322, 0.4759, 0.5626, 1),
    "lr1.5": (0.1064, 0.0201, 0.3256, 0.6576, 12),
    "lr2.0": (0.0776, 0.0162, 0.2997, 0.7532, 18),
}

NLL_TOP1_ANCHORS = {
    1750: {"val_nll": -0.030889, "dev_nll": 0.018780, "val_top1": -76, "dev_top1": -53},
    3000: {"val_nll": -0.053659, "dev_nll": 0.025505, "val_top1": -45, "dev_top1": -16},
}


def seed_contract_checks() -> list[str]:
    """Pin the seed-role contract that the runner and this analyzer share.

    Only the pre-registered pair 2 / 4 may drive R1-R5.  A seed directory the
    registry does not list is exploratory (fail closed on provenance), and a
    malformed registry is an evidence error rather than a silent downgrade.
    """
    problems: list[str] = []
    expected = {1: "reference", 2: "preregistered", 3: "exploratory",
                4: "preregistered", 5: "exploratory"}
    for seed, role in expected.items():
        got = default_seed_role(seed)
        if got != role:
            problems.append(f"SEED_ROLE {seed}: {got} != {role}")

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        for seed in (2, 4, 5):
            (tree / f"seed{seed}").mkdir()
        registry = {
            "protocol": "docs/g1-1p5x-multiseed-3000.md",
            "preregistered_seeds": [2, 4],
            "seeds": [{"seed": 2, "role": "preregistered"},
                      {"seed": 5, "role": "exploratory"}],
        }
        (tree / SEED_REGISTRY_FILE).write_text(json.dumps(registry),
                                              encoding="utf-8")
        roles, source = load_seed_roles(tree)
        if source != "registry":
            problems.append(f"SEED_REGISTRY source={source} != registry")
        for seed, role, role_source in ((2, "preregistered", "registry"),
                                        (4, "exploratory", "unlisted"),
                                        (5, "exploratory", "registry")):
            got = roles.get(seed, {})
            if got.get("role") != role or got.get("source") != role_source:
                problems.append(f"SEED_REGISTRY seed {seed}: {got} != "
                                f"{role}/{role_source}")

        (tree / SEED_REGISTRY_FILE).write_text(
            json.dumps({"seeds": [{"seed": 2, "role": "convenient"}]}),
            encoding="utf-8")
        try:
            load_seed_roles(tree)
            problems.append("SEED_REGISTRY accepted an unknown role")
        except EvidenceError:
            pass

    def rows(*seed_verdicts: tuple[int, str]) -> list[dict]:
        return [{"seed": seed, "seed_role": default_seed_role(seed),
                 "criterion": "R2", "verdict": verdict, "value": "", "detail": ""}
                for seed, verdict in seed_verdicts]

    for label, built, want in (
            ("exploratory only", rows((3, "reproduced")), "preregistered_incomplete"),
            ("reference only", rows((1, "reproduced")), "preregistered_incomplete"),
            ("2/2 with exploratory miss",
             rows((2, "reproduced"), (4, "reproduced"), (3, "none")),
             "close_high_lr_quality_lane"),
            ("1/2 with exploratory hit",
             rows((2, "reproduced"), (4, "none"), (3, "reproduced")),
             "ambiguous_tie_breaker"),
            ("0/2 with exploratory hit",
             rows((2, "none"), (4, "none"), (3, "reproduced")),
             "close_split_lane")):
        code, _ = decision(built)
        if code != want:
            problems.append(f"DECISION {label}: {code} != {want}")
    return problems


def selftest() -> int:
    """Recompute committed seed-1 evidence; a failure means this analyzer drifted."""
    problems: list[str] = order_prefix_checks()
    contract_problems = seed_contract_checks()
    problems += contract_problems
    if not contract_problems:
        print("seed-role contract verified: pre-registered 2/4 only, an unlisted "
              "seed fails closed as exploratory, exploratory / reference verdicts "
              "cannot move the R1-R5 decision")

    grid = REPO_ROOT / "docs/results/g1-lr-stress-2026-09"
    committed_grid = load_committed_pairs(grid / "quality-paired.csv", use_lr=True)
    cells = 0
    for lr, lr_dir in (("1.0", "lr1.0"), ("1.25", "lr1.25"),
                       ("1.5", "lr1.5"), ("2.0", "lr2.0")):
        for key, values in sorted(recompute_pairs(grid / lr_dir, lr).items()):
            for name in ("delta_val", "delta_dev", "delta_balanced"):
                check_close(problems, f"grid {lr_dir} step {key[1]} {name}",
                            values[name], committed_grid[key][name], 1e-9)
            cells += 1
    print(f"grid paired cells verified: {cells}")

    full = REPO_ROOT / "docs/results/g1-1p5x-full-8000-2026-09"
    committed_full = load_committed_pairs(full / "quality-paired.csv")
    full_rows = {row["step"]: row for row in paired_series(
        1, load_arm_evals(full / "control"), load_arm_evals(full / "g1"))}
    for key, values in sorted(recompute_pairs(full).items()):
        for name in ("delta_val", "delta_dev", "delta_balanced"):
            check_close(problems, f"full-8000 step {key[1]} {name}",
                        values[name], committed_full[key][name], 1e-9)
    print(f"full-8000 paired checkpoints verified: {len(full_rows)}")

    for step, expected in NLL_TOP1_ANCHORS.items():
        row = full_rows[step]
        check_close(problems, f"step {step} Val NLL delta",
                    row["val_nll_delta"], expected["val_nll"], 5e-7)
        check_close(problems, f"step {step} Dev NLL delta",
                    row["dev_nll_delta"], expected["dev_nll"], 5e-7)
        for split, name in (("val", "val_top1"), ("dev", "dev_top1")):
            got = row[f"{split}_top1_token_delta"]
            if got != expected[name]:
                problems.append(
                    f"step {step} {split} top-1 token delta {got} != "
                    f"committed {expected[name]}")
        print(f"step {step}: Val NLL {row['val_nll_delta']:+.6f} / "
              f"Dev NLL {row['dev_nll_delta']:+.6f} / top-1 "
              f"{row['val_top1_token_delta']:+d} Val vs "
              f"{row['dev_top1_token_delta']:+d} Dev matches committed README")

    for lr_dir, anchors in GATE_STEP500_ANCHORS.items():
        mean, low, high, below, suppressed = anchors
        aggregate = aggregate_gate(grid / lr_dir / "g1" / "gate-static-step500.txt",
                                   "gate_static")
        if aggregate["heads"] != 38:
            problems.append(f"gate {lr_dir}: heads={aggregate['heads']} != 38")
        for label, got, want in (
                ("mean of head means", aggregate["mean_of_head_means"], mean),
                ("min head mean", aggregate["min_head_mean"], low),
                ("max head mean", aggregate["max_head_mean"], high),
                ("mean below 0.1", aggregate["mean_below_0_1"], below)):
            check_close(problems, f"gate {lr_dir} {label}", got, want, 5e-5)
        if aggregate["heads_fully_suppressed"] != suppressed:
            problems.append(
                f"gate {lr_dir} fully suppressed heads "
                f"{aggregate['heads_fully_suppressed']} != committed {suppressed}")
        print(f"gate {lr_dir}: mean {aggregate['mean_of_head_means']:.4f}, "
              f"fully-suppressed {aggregate['heads_fully_suppressed']}/38 "
              f"matches committed README")

    for problem in problems:
        print(f"SELFTEST FAIL: {problem}")
    if problems:
        return 1
    print("SELFTEST PASS: order prefix stability, seed-role contract, grid and "
          "full-8000 paired deltas, NLL / top-1 anchors, gate aggregates")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split-level paired analyzer for the G1 1.5x multi-seed run")
    parser.add_argument("--tree", type=Path,
                        help="multi-seed tree with seed<N>/{control,g1} children")
    parser.add_argument("--out", type=Path,
                        help="directory for aggregate CSVs (default: --tree)")
    parser.add_argument("--selftest", action="store_true",
                        help="recompute committed seed-1 evidence and compare")
    parser.add_argument("--allow-problems", action="store_true",
                        help="report identity or health problems without failing")
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        if args.tree is None:
            parser.error("--tree or --selftest is required")
        return analyze_tree(args.tree, args.out or args.tree, not args.allow_problems)
    except EvidenceError as error:
        print(f"EVIDENCE_ERROR: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

