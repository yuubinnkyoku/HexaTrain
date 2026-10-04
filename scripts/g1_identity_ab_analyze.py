#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Aggregate Current G1 vs Identity-init G1 500-step A/B artifacts.

Read-only over report/results roots. Does not train and does not touch the
device.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ARMS = ["current", "identity"]
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
    wgate_rows: List[dict] = []
    runtime_rows: List[dict] = []
    health_rows: List[dict] = []
    t2b_rows: List[dict] = []
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
                "completed_steps": result.get("completed_steps", ""),
                "checkpoint_count": result.get("checkpoint_count", ""),
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
        for step in eval_steps:
            diag = arm_path / f"gate-static-step{step}.txt"
            if not diag.is_file():
                continue
            for line in diag.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith("gate_static_l") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                # gate_static_l{L}_h{H}_{field}
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
                    (r for r in gate_rows if r["arm"] == arm and r["step"] == step
                     and r["layer"] == layer and r["head"] == head),
                    None,
                )
                if row is None:
                    row = {"arm": arm, "step": step, "layer": layer, "head": head}
                    gate_rows.append(row)
                row[field] = v.strip()
                if field == "mean":
                    mean = fnum(v)
                    row["mean_abs_from_identity"] = f"{abs(mean - 1.0):.6f}"
        # Wg telemetry from final result fields if present.
        for key in (
            "wg_weight_rms", "wg_grad_rms", "wg_update_rms",
            "attention_gate_weight_rms",
        ):
            if key in result:
                wgate_rows.append({"arm": arm, "step": args.steps, "metric": key, "value": result[key]})

    for step in eval_steps:
        cur = arm_evals.get("current", {}).get(step)
        ident = arm_evals.get("identity", {}).get(step)
        if not cur or not ident:
            continue
        paired_rows.append(
            {
                "step": step,
                "current_val": f"{cur[0]:.9f}",
                "identity_val": f"{ident[0]:.9f}",
                "delta_val": f"{ident[0] - cur[0]:.9f}",
                "current_dev": f"{cur[1]:.9f}",
                "identity_dev": f"{ident[1]:.9f}",
                "delta_dev": f"{ident[1] - cur[1]:.9f}",
                "current_balanced": f"{cur[2]:.9f}",
                "identity_balanced": f"{ident[2]:.9f}",
                "delta_balanced": f"{ident[2] - cur[2]:.9f}",
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
        ident = best.get("identity")
        t2b_rows.append(
            {
                "target_bpb": f"{target:.2f}",
                "current_first_step": cur["step"] if cur else ">500",
                "identity_first_step": ident["step"] if ident else ">500",
                "delta_step": (ident["step"] - cur["step"]) if cur and ident else "CENSORED",
                "current_estimated_original_bytes_to_target": (
                    estimated_bytes(cur["step"]) if cur else "NOT_REACHED"
                ),
                "identity_estimated_original_bytes_to_target": (
                    estimated_bytes(ident["step"]) if ident else "NOT_REACHED"
                ),
                "current_checkpoint_training_wall_s": "NOT_MEASURED" if cur else "NOT_REACHED",
                "identity_checkpoint_training_wall_s": "NOT_MEASURED" if ident else "NOT_REACHED",
            }
        )

    write_csv(
        results_root / "quality.csv",
        quality_rows,
        ["arm", "step", "validation_bpb", "development_bpb", "balanced_bpb",
         "validation_chunks", "development_chunks", "final_split_used"],
    )
    write_csv(
        results_root / "quality-paired.csv",
        paired_rows,
        ["step", "current_val", "identity_val", "delta_val",
         "current_dev", "identity_dev", "delta_dev",
         "current_balanced", "identity_balanced", "delta_balanced"],
    )
    write_csv(
        results_root / "time-to-bpb.csv",
        t2b_rows,
        ["target_bpb", "current_first_step", "identity_first_step", "delta_step",
         "current_estimated_original_bytes_to_target",
         "identity_estimated_original_bytes_to_target",
         "current_checkpoint_training_wall_s",
         "identity_checkpoint_training_wall_s"],
    )
    write_csv(
        results_root / "gate-static.csv",
        gate_rows,
        ["arm", "step", "layer", "head", "mean", "stddev", "min", "max",
         "below_0_1_fraction", "above_0_9_fraction", "mean_abs_from_identity"],
    )
    write_csv(
        results_root / "wgate-telemetry.csv",
        wgate_rows,
        ["arm", "step", "metric", "value"],
    )
    write_csv(
        results_root / "runtime.csv",
        runtime_rows,
        ["arm", "training_wall_s", "training_step_ms", "completed_steps",
         "checkpoint_count", "status"],
    )
    write_csv(
        results_root / "health.csv",
        health_rows,
        ["arm", "qnn_return_code_success", "hvx_rpc_failure_count",
         "hvx_fallback_count", "hvx_nonfinite_count", "cpu_fallback",
         "nan_detected", "inf_detected", "thermal_status_before",
         "thermal_status_after"],
    )
    print(f"analysis_written={results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
