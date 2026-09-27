#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Aggregate G1 1.5x full-horizon 8000-step Control vs G1 artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ARMS = ["control", "g1"]
TARGETS = [2.90, 2.80, 2.70, 2.65, 2.60, 2.55, 2.50, 2.45, 2.40, 2.35]
CANONICAL_8000_STEP_UTF8_BYTES = 5_491_256
CANONICAL_8000_STEPS = 8000
HISTORICAL_1P5X = {
    100: -0.140503,
    200: -0.057896,
    300: -0.037254,
    400: -0.063141,
    500: -0.029986,
    750: -0.033073,
    1000: -0.041670,
    1250: -0.046216,
    1500: -0.033268,
    1750: -0.002872,
    2000: -0.025141,
}
# Historical 1.0x full-horizon (seed1) for 1.0x-vs-1.5x comparison.
HISTORICAL_1P0X = {
    500: -0.032712,
    1000: -0.036686,
    1500: -0.021864,
    2000: -0.007113,
    2500: -0.016147,
    3000: -0.017483,
    3500: -0.007219,
    4000: -0.007490,
    5000: 0.004985,
    6000: 0.008981,
    7000: 0.009498,
    8000: -0.000367,
}


def parse_kv(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def fnum(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def balanced(val: float, dev: float) -> float:
    return (val + dev) / 2.0


def load_eval(arm_path: Path, step: int) -> Optional[Dict[str, str]]:
    path = arm_path / f"eval256-step{step}-htp.txt"
    if not path.is_file():
        eval_dir = arm_path / f"eval256-step{step}"
        if eval_dir.is_dir():
            hits = list(eval_dir.glob("*-htp.txt"))
            if hits:
                path = hits[0]
            else:
                return None
        else:
            return None
    return parse_kv(path)


def estimated_bytes(step: int) -> str:
    if step <= 0:
        return "NOT_MEASURED"
    return f"{CANONICAL_8000_STEP_UTF8_BYTES * step / CANONICAL_8000_STEPS:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--eval-steps", default="2000,2500,3000,3500,4000,4500,5000,6000,7000,8000")
    ap.add_argument("--parent-results-root", default="docs/results/g1-1p5x-long-2000-2026-09")
    args = ap.parse_args()
    report_root = Path(args.report_root)
    results_root = Path(args.results_root)
    eval_steps = [int(x) for x in args.eval_steps.split(",") if x]
    results_root.mkdir(parents=True, exist_ok=True)

    quality_rows: List[dict] = []
    paired_rows: List[dict] = []
    gate_rows: List[dict] = []
    runtime_rows: List[dict] = []
    health_rows: List[dict] = []
    t2b_rows: List[dict] = []
    telemetry_rows: List[dict] = []
    arm_evals: Dict[str, Dict[int, Tuple[float, float, float]]] = {}

    parent_root = Path(args.parent_results_root)
    # Historical context steps stay in the parent tree; merge them so the phase
    # tables (500->2000) and the 1.0x-vs-1.5x table can be emitted from one CSV.
    history_steps = [500, 750, 1000, 1250, 1500, 1750]
    for arm in ARMS:
        arm_path = report_root / arm
        # New-eval dir may hold eval256-stepN/ subdirs; results_root holds flat copies.
        for step in eval_steps:
            ev = load_eval(arm_path, step)
            if not ev:
                ev = load_eval(results_root / arm, step)
            if not ev:
                continue
            val = fnum(ev.get("validation_bits_per_utf8_byte"))
            dev = fnum(ev.get("development_bits_per_utf8_byte"))
            bal = balanced(val, dev)
            arm_evals[arm] = arm_evals.get(arm, {})
            arm_evals[arm][step] = (val, dev, bal)
            quality_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "validation_bpb": f"{val:.9f}",
                    "development_bpb": f"{dev:.9f}",
                    "balanced_bpb": f"{bal:.9f}",
                    "validation_chunks": ev.get("validation_chunks", "256"),
                    "development_chunks": ev.get("development_chunks", "256"),
                    "final_split_used": "false",
                }
            )
    for arm in ARMS:
        for step in history_steps:
            ev = load_eval(parent_root / arm, step)
            if not ev:
                continue
            val = fnum(ev.get("validation_bits_per_utf8_byte"))
            dev = fnum(ev.get("development_bits_per_utf8_byte"))
            bal = balanced(val, dev)
            arm_evals[arm][step] = (val, dev, bal)
            quality_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "validation_bpb": f"{val:.9f}",
                    "development_bpb": f"{dev:.9f}",
                    "balanced_bpb": f"{bal:.9f}",
                    "validation_chunks": ev.get("validation_chunks", "256"),
                    "development_chunks": ev.get("development_chunks", "256"),
                    "final_split_used": "false",
                }
            )
    for arm in ARMS:
        arm_path = report_root / arm
        result_path = arm_path / f"seed1-l19-v1024-t32-d64-f128-steps{args.steps}-result.txt"
        result = parse_kv(result_path) if result_path.is_file() else {}
        runtime_rows.append(
            {
                "arm": arm,
                "training_wall_s": result.get("training_total_seconds", ""),
                "training_step_ms": result.get("training_step_ms", ""),
                "fwd_backward_ms_per_update": result.get("fwd_backward_ms_per_update", ""),
                "muon_ms_per_update": result.get("muon_ms_per_update", ""),
                "aux_adam_ms_per_update": result.get("aux_adam_ms_per_update", ""),
                "completed_steps": result.get("completed_steps", ""),
                "status": result.get("status", ""),
            }
        )
        health_rows.append(
            {
                "arm": arm,
                "qnn_return_code_success": result.get("qnn_return_code_success", ""),
                "hvx_rpc_failure_count": result.get("hvx_rpc_failure_count", ""),
                "hvx_fallback_count": result.get("hvx_fallback_count", ""),
                "hvx_nonfinite_count": result.get("hvx_nonfinite_count", ""),
                "cpu_fallback": result.get("cpu_fallback", ""),
                "nan_detected": result.get("nan_detected", ""),
                "inf_detected": result.get("inf_detected", ""),
                "thermal_status_before": result.get("android_thermal_status_before", ""),
                "thermal_status_after": result.get("android_thermal_status_after", ""),
                "focus_takeover_count": result.get("focus_takeover_count", ""),
            }
        )
        telemetry_rows.append(
            {
                "arm": arm,
                "first_loss": result.get("first_loss", ""),
                "last_loss": result.get("last_loss", ""),
                "parameter_count": result.get("parameter_count", ""),
                "checkpoint_format": result.get("checkpoint_format", ""),
                "attention_gate": result.get("attention_gate", ""),
            }
        )
        if arm == "g1":
            for step in eval_steps + history_steps:
                diag = arm_path / f"gate-static-step{step}.txt"
                if not diag.is_file():
                    diag = results_root / arm / f"gate-static-step{step}.txt"
                if not diag.is_file():
                    diag = parent_root / arm / f"gate-static-step{step}.txt"
                if not diag.is_file():
                    continue
                for line in diag.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.startswith("gate_static_l") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    parts = k.strip().split("_")
                    if len(parts) < 5:
                        continue
                    try:
                        layer = int(parts[2][1:] if parts[2].startswith("l") else parts[2])
                        head = int(parts[3][1:] if parts[3].startswith("h") else parts[3])
                    except ValueError:
                        continue
                    field = "_".join(parts[4:])
                    row = next(
                        (r for r in gate_rows if r["step"] == step and r["layer"] == layer and r["head"] == head),
                        None,
                    )
                    if row is None:
                        row = {"step": step, "layer": layer, "head": head}
                        gate_rows.append(row)
                    row[field] = v.strip()

    deltas: List[float] = []
    all_steps = sorted(set(eval_steps) | set(history_steps) | set(arm_evals.get("control", {})) | set(arm_evals.get("g1", {})))
    for step in all_steps:
        ctrl = arm_evals.get("control", {}).get(step)
        g1 = arm_evals.get("g1", {}).get(step)
        if not ctrl or not g1:
            continue
        delta_bal = g1[2] - ctrl[2]
        deltas.append(delta_bal)
        paired_rows.append(
            {
                "step": step,
                "control_val": f"{ctrl[0]:.9f}",
                "g1_val": f"{g1[0]:.9f}",
                "delta_val": f"{g1[0] - ctrl[0]:.9f}",
                "control_dev": f"{ctrl[1]:.9f}",
                "g1_dev": f"{g1[1]:.9f}",
                "delta_dev": f"{g1[1] - ctrl[1]:.9f}",
                "control_balanced": f"{ctrl[2]:.9f}",
                "g1_balanced": f"{g1[2]:.9f}",
                "delta_balanced": f"{delta_bal:.9f}",
                "historical_delta_balanced": (
                    f"{HISTORICAL_1P5X[step]:.6f}" if step in HISTORICAL_1P5X else ""
                ),
                "delta_balanced_1p0x": (
                    f"{HISTORICAL_1P0X[step]:.6f}" if step in HISTORICAL_1P0X else ""
                ),
            }
        )

    mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
    # Trapezoid area over step, normalized by span.
    area = float("nan")
    if len(deltas) >= 2:
        pts = [(s, arm_evals["g1"][s][2] - arm_evals["control"][s][2])
               for s in all_steps
               if s in arm_evals.get("g1", {}) and s in arm_evals.get("control", {})]
        if len(pts) >= 2:
            acc = 0.0
            for i in range(1, len(pts)):
                acc += 0.5 * (pts[i][1] + pts[i - 1][1]) * (pts[i][0] - pts[i - 1][0])
            span = pts[-1][0] - pts[0][0]
            area = acc / span if span else float("nan")

    phase_rows: List[dict] = []
    phase_defs = {
        "500-2000": [500, 750, 1000, 1250, 1500, 1750, 2000],
        "2000-4000": [2000, 2500, 3000, 3500, 4000],
        "4000-8000": [4000, 4500, 5000, 6000, 7000, 8000],
        "500-8000": all_steps,
    }
    for name, steps in phase_defs.items():
        vals = [arm_evals["g1"][s][2] - arm_evals["control"][s][2]
                for s in steps
                if s in arm_evals.get("g1", {}) and s in arm_evals.get("control", {})]
        wins = sum(1 for v in vals if v < 0)
        ties = sum(1 for v in vals if v == 0)
        losses = sum(1 for v in vals if v > 0)
        mean = sum(vals) / len(vals) if vals else float("nan")
        spts = sorted(s for s in steps if s in arm_evals.get("g1", {}) and s in arm_evals.get("control", {}))
        acc = 0.0
        span = 0.0
        if len(spts) >= 2:
            for i in range(1, len(spts)):
                d0 = arm_evals["g1"][spts[i - 1]][2] - arm_evals["control"][spts[i - 1]][2]
                d1 = arm_evals["g1"][spts[i]][2] - arm_evals["control"][spts[i]][2]
                acc += 0.5 * (d0 + d1) * (spts[i] - spts[i - 1])
            span = spts[-1] - spts[0]
        phase_rows.append(
            {
                "phase": name,
                "n": len(vals),
                "mean_delta_balanced": f"{mean:.9f}" if math.isfinite(mean) else "nan",
                "normalized_area_delta_balanced": f"{acc / span:.9f}" if span else "nan",
                "g1_wins": wins,
                "ties": ties,
                "control_wins": losses,
            }
        )
    compare_rows: List[dict] = []
    for step in all_steps:
        if step not in HISTORICAL_1P0X and step not in HISTORICAL_1P5X:
            continue
        ctrl = arm_evals.get("control", {}).get(step)
        g1 = arm_evals.get("g1", {}).get(step)
        cur = (g1[2] - ctrl[2]) if (ctrl and g1) else float("nan")
        v10 = HISTORICAL_1P0X.get(step)
        imp = (cur - v10) if (v10 is not None and math.isfinite(cur)) else float("nan")
        compare_rows.append(
            {
                "step": step,
                "delta_balanced_1p0x": f"{v10:.6f}" if v10 is not None else "",
                "delta_balanced_1p5x": f"{cur:.9f}" if math.isfinite(cur) else "",
                "improvement_from_1p5x": f"{imp:.9f}" if math.isfinite(imp) else "",
            }
        )
    for target in TARGETS:
        best = {}
        for arm in ARMS:
            cands = []
            for step, (val, dev, bal) in arm_evals.get(arm, {}).items():
                if math.isfinite(bal) and bal <= target:
                    cands.append(step)
            if cands:
                best[arm] = min(cands)
        ctrl = best.get("control")
        g1 = best.get("g1")
        t2b_rows.append(
            {
                "target_bpb": f"{target:.2f}",
                "control_first_step": ctrl if ctrl else ">8000",
                "g1_first_step": g1 if g1 else ">8000",
                "delta_step": (g1 - ctrl) if ctrl and g1 else "CENSORED",
                "control_estimated_original_bytes_to_target": estimated_bytes(ctrl) if ctrl else "NOT_REACHED",
                "g1_estimated_original_bytes_to_target": estimated_bytes(g1) if g1 else "NOT_REACHED",
                "control_checkpoint_training_wall_s": "NOT_MEASURED",
                "g1_checkpoint_training_wall_s": "NOT_MEASURED",
            }
        )

    write_csv(results_root / "quality.csv", quality_rows,
              ["arm", "step", "validation_bpb", "development_bpb", "balanced_bpb",
               "validation_chunks", "development_chunks", "final_split_used"])
    write_csv(results_root / "quality-paired.csv", paired_rows,
              ["step", "control_val", "g1_val", "delta_val",
               "control_dev", "g1_dev", "delta_dev",
               "control_balanced", "g1_balanced", "delta_balanced",
               "historical_delta_balanced", "delta_balanced_1p0x"])
    write_csv(results_root / "quality-1p0-vs-1p5.csv", compare_rows,
              ["step", "delta_balanced_1p0x", "delta_balanced_1p5x",
               "improvement_from_1p5x"])
    write_csv(results_root / "phase-summary.csv", phase_rows,
              ["phase", "n", "mean_delta_balanced",
               "normalized_area_delta_balanced", "g1_wins", "ties",
               "control_wins"])
    write_csv(results_root / "time-to-bpb.csv", t2b_rows,
              ["target_bpb", "control_first_step", "g1_first_step", "delta_step",
               "control_estimated_original_bytes_to_target",
               "g1_estimated_original_bytes_to_target",
               "control_checkpoint_training_wall_s", "g1_checkpoint_training_wall_s"])
    write_csv(results_root / "gate-static.csv", gate_rows,
              ["step", "layer", "head", "mean", "stddev", "min", "max",
               "below_0_1_fraction", "above_0_9_fraction"])
    write_csv(results_root / "training-telemetry.csv", telemetry_rows,
              ["arm", "first_loss", "last_loss", "parameter_count",
               "checkpoint_format", "attention_gate"])
    write_csv(results_root / "runtime.csv", runtime_rows,
              ["arm", "training_wall_s", "training_step_ms", "fwd_backward_ms_per_update",
               "muon_ms_per_update", "aux_adam_ms_per_update", "completed_steps", "status"])
    write_csv(results_root / "health.csv", health_rows,
              ["arm", "qnn_return_code_success", "hvx_rpc_failure_count",
               "hvx_fallback_count", "hvx_nonfinite_count", "cpu_fallback",
               "nan_detected", "inf_detected", "thermal_status_before",
               "thermal_status_after", "focus_takeover_count"])
    (results_root / "trajectory-summary.json").write_text(
        json.dumps(
            {
                "mean_delta_balanced": mean_delta,
                "normalized_area_delta_balanced": area,
                "eval_steps": eval_steps,
                "all_steps": all_steps,
                "phases": {r["phase"]: r for r in phase_rows},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"analysis_written={results_root}")
    print(f"mean_delta_balanced={mean_delta}")
    print(f"normalized_area_delta_balanced={area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
