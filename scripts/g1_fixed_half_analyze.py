#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Aggregate Current G1 vs Fixed 0.5 branch-scale A/B artifacts."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ARMS = ["current", "fixedhalf"]
TARGETS = [3.10, 3.00, 2.95, 2.90, 2.85]
CANONICAL_8000_STEP_UTF8_BYTES = 5_491_256
CANONICAL_8000_STEPS = 8000


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
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-steps", default="100,200,300,400,500")
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

    for arm in ARMS:
        arm_path = report_root / arm
        result_path = arm_path / f"seed1-l19-v1024-t32-d64-f128-steps{args.steps}-result.txt"
        result = parse_kv(result_path) if result_path.is_file() else {}
        arm_evals[arm] = {}
        for step in eval_steps:
            ev = load_eval(arm_path, step)
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
            }
        )
        telemetry_rows.append(
            {
                "arm": arm,
                "first_loss": result.get("first_loss", ""),
                "last_loss": result.get("last_loss", ""),
                "parameter_count": result.get("parameter_count", ""),
                "muon_parameter_count": result.get("muon_parameter_count", ""),
                "aux_adam_parameter_count": result.get("aux_adam_parameter_count", ""),
                "checkpoint_format": result.get("checkpoint_format", ""),
                "attention_gate": result.get("attention_gate", ""),
            }
        )
        if arm == "current":
            for step in eval_steps:
                diag = arm_path / f"gate-static-step{step}.txt"
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
                        row = {"arm": "current", "step": step, "layer": layer, "head": head}
                        gate_rows.append(row)
                    row[field] = v.strip()
            # Theoretical fixed_half gate: constant 0.5.
            for step in eval_steps:
                gate_rows.append(
                    {
                        "arm": "fixedhalf",
                        "step": step,
                        "layer": "all",
                        "head": "all",
                        "mean": "0.5",
                        "stddev": "0",
                        "min": "0.5",
                        "max": "0.5",
                        "below_0_1_fraction": "0",
                        "above_0_9_fraction": "0",
                        "note": "constant_theoretical",
                    }
                )

    for step in eval_steps:
        cur = arm_evals.get("current", {}).get(step)
        fx = arm_evals.get("fixedhalf", {}).get(step)
        if not cur or not fx:
            continue
        paired_rows.append(
            {
                "step": step,
                "current_val": f"{cur[0]:.9f}",
                "fixed_val": f"{fx[0]:.9f}",
                "delta_val": f"{fx[0] - cur[0]:.9f}",
                "current_dev": f"{cur[1]:.9f}",
                "fixed_dev": f"{fx[1]:.9f}",
                "delta_dev": f"{fx[1] - cur[1]:.9f}",
                "current_balanced": f"{cur[2]:.9f}",
                "fixed_balanced": f"{fx[2]:.9f}",
                "delta_balanced": f"{fx[2] - cur[2]:.9f}",
            }
        )

    for target in TARGETS:
        best = {}
        for arm in ARMS:
            cands = []
            for step, (val, dev, bal) in arm_evals.get(arm, {}).items():
                if math.isfinite(bal) and bal <= target:
                    cands.append({"step": step, "balanced": bal})
            if cands:
                cands.sort(key=lambda x: (x["step"], x["balanced"]))
                best[arm] = cands[0]
        cur = best.get("current")
        fx = best.get("fixedhalf")
        t2b_rows.append(
            {
                "target_bpb": f"{target:.2f}",
                "current_first_step": cur["step"] if cur else ">500",
                "fixed_first_step": fx["step"] if fx else ">500",
                "delta_step": (fx["step"] - cur["step"]) if cur and fx else "CENSORED",
                "current_estimated_original_bytes_to_target": estimated_bytes(cur["step"]) if cur else "NOT_REACHED",
                "fixed_estimated_original_bytes_to_target": estimated_bytes(fx["step"]) if fx else "NOT_REACHED",
                "current_checkpoint_training_wall_s": "NOT_MEASURED" if cur else "NOT_REACHED",
                "fixed_checkpoint_training_wall_s": "NOT_MEASURED" if fx else "NOT_REACHED",
            }
        )

    write_csv(results_root / "quality.csv", quality_rows,
              ["arm", "step", "validation_bpb", "development_bpb", "balanced_bpb",
               "validation_chunks", "development_chunks", "final_split_used"])
    write_csv(results_root / "quality-paired.csv", paired_rows,
              ["step", "current_val", "fixed_val", "delta_val",
               "current_dev", "fixed_dev", "delta_dev",
               "current_balanced", "fixed_balanced", "delta_balanced"])
    write_csv(results_root / "time-to-bpb.csv", t2b_rows,
              ["target_bpb", "current_first_step", "fixed_first_step", "delta_step",
               "current_estimated_original_bytes_to_target",
               "fixed_estimated_original_bytes_to_target",
               "current_checkpoint_training_wall_s", "fixed_checkpoint_training_wall_s"])
    write_csv(results_root / "current-gate-static.csv", gate_rows,
              ["arm", "step", "layer", "head", "mean", "stddev", "min", "max",
               "below_0_1_fraction", "above_0_9_fraction", "note"])
    write_csv(results_root / "training-telemetry.csv", telemetry_rows,
              ["arm", "first_loss", "last_loss", "parameter_count",
               "muon_parameter_count", "aux_adam_parameter_count",
               "checkpoint_format", "attention_gate"])
    write_csv(results_root / "runtime.csv", runtime_rows,
              ["arm", "training_wall_s", "training_step_ms", "fwd_backward_ms_per_update",
               "muon_ms_per_update", "aux_adam_ms_per_update", "completed_steps", "status"])
    write_csv(results_root / "health.csv", health_rows,
              ["arm", "qnn_return_code_success", "hvx_rpc_failure_count",
               "hvx_fallback_count", "hvx_nonfinite_count", "cpu_fallback",
               "nan_detected", "inf_detected", "thermal_status_before", "thermal_status_after"])
    print(f"analysis_written={results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
