#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Aggregate G1 high-LR stress grid artifacts into canonical CSVs.

Read-only over build/g1-lr-stress and docs/results/g1-lr-stress-2026-09.
Does not train and does not touch the device.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TARGETS = [3.10, 3.00, 2.95, 2.90, 2.85]
MULTS = ["1.0", "1.25", "1.5", "2.0"]
ARMS = ["Control", "G1"]
# Canonical 8000-step original UTF-8 exposure. Muon-hybrid reports do not
# emit a per-step byte counter, so time-to-target bytes are step-proportional
# estimates only (same DataCursor / batch / step across arms at equal step).
CANONICAL_8000_STEP_UTF8_BYTES = 5_491_256
CANONICAL_8000_STEPS = 8000
MAGIC_V4 = b"NPRTCKPTV4\n"
MAGIC_V5 = b"NPRTCKPTV5\n"
ROLE_MUON = 1
ROLE_AUX_ADAM = 2


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


def inum(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return -1


class ByteReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def u32(self) -> int:
        if self.remaining() < 4:
            raise ValueError("TRUNCATED_U32")
        value = struct.unpack_from(">I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def u64(self) -> int:
        if self.remaining() < 8:
            raise ValueError("TRUNCATED_U64")
        value = struct.unpack_from(">Q", self.data, self.offset)[0]
        self.offset += 8
        return value

    def f32(self) -> float:
        if self.remaining() < 4:
            raise ValueError("TRUNCATED_F32")
        value = struct.unpack_from(">f", self.data, self.offset)[0]
        self.offset += 4
        return float(value)

    def string(self) -> str:
        length = self.u32()
        if length < 1 or length > 4096 or self.remaining() < length:
            raise ValueError("STRING_INVALID")
        value = self.data[self.offset : self.offset + length].decode("utf-8")
        self.offset += length
        return value

    def floats(self, count: int) -> np.ndarray:
        actual = self.u64()
        if actual != count:
            raise ValueError(f"VALUES_COUNT_MISMATCH:{count}:{actual}")
        nbytes = count * 4
        raw = self.data[self.offset : self.offset + nbytes]
        self.offset += nbytes
        return np.frombuffer(raw, dtype=">f4").astype(np.float32)


@dataclass
class Parameter:
    name: str
    role: int
    rows: int
    columns: int
    values: np.ndarray


def decode_checkpoint(path: Path) -> Tuple[dict, List[Parameter]]:
    data = path.read_bytes()
    reader = ByteReader(data)
    if data.startswith(MAGIC_V4):
        gated = False
        reader.offset = len(MAGIC_V4)
    elif data.startswith(MAGIC_V5):
        gated = True
        reader.offset = len(MAGIC_V5)
    else:
        raise ValueError(f"CHECKPOINT_MAGIC:{path}")
    meta = {
        "vocabulary": reader.u32(),
        "tokens": reader.u32(),
        "dimension": reader.u32(),
        "feed_forward": reader.u32(),
        "layers": reader.u32(),
        "heads": reader.u32(),
    }
    meta["epsilon"] = reader.f32()
    if gated:
        meta["attention_gate"] = reader.u32()
    else:
        meta["attention_gate"] = 0
    meta["seed"] = reader.u32()
    meta["step"] = reader.u64()
    meta["tokenizer_kind"] = reader.string()
    meta["tokenizer_hash"] = reader.string()
    meta["dataset_hash"] = reader.string()
    meta["record_index"] = reader.u64()
    meta["token_offset"] = reader.u64()
    meta["epoch"] = reader.u64()
    meta["exposed"] = reader.u64()
    meta["order_seed"] = reader.u64()
    meta["optimizer_identity"] = reader.string()
    meta["muon_lr"] = reader.f32()
    meta["aux_adam_lr"] = reader.f32()
    meta["muon_momentum"] = reader.f32()
    # remaining schedule floats
    for _ in range(2):
        reader.f32()
    reader.u32()  # nesterov
    meta["ns_steps"] = reader.u32()
    for _ in range(5):
        reader.f32()
    meta["decay_start"] = reader.u32()
    meta["decay_end"] = reader.u32()
    meta["schedule_total"] = reader.u32()
    meta["schema_version"] = reader.u32()
    meta["registry_version"] = reader.u32()
    registry_count = reader.u32()
    params: List[Parameter] = []
    for _ in range(registry_count):
        name = reader.string()
        role = reader.u32()
        rows = reader.u32()
        columns = reader.u32()
        elements = rows * columns
        values = reader.floats(elements)
        if role == ROLE_MUON:
            reader.floats(elements)
        elif role == ROLE_AUX_ADAM:
            reader.floats(elements)
            reader.floats(elements)
        else:
            raise ValueError(f"ROLE_INVALID:{name}")
        params.append(Parameter(name, role, rows, columns, values.reshape((rows, columns))))
    return meta, params


def shared_parameter_hash(params: Sequence[Parameter]) -> str:
    h = hashlib.sha256()
    for p in sorted(params, key=lambda x: x.name):
        if "attention_gate" in p.name or "attentionGate" in p.name:
            continue
        h.update(p.name.encode("utf-8"))
        h.update(p.values.astype(">f4").tobytes())
    return h.hexdigest()


def wg_geometry(params: Sequence[Parameter]) -> dict:
    rows = []
    for p in params:
        if "attention_gate" not in p.name and "attentionGate" not in p.name:
            continue
        w = p.values.astype(np.float64)
        norm = float(np.linalg.norm(w))
        rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
        rows.append(
            {
                "name": p.name,
                "weight_norm": norm,
                "weight_rms": rms,
                "min": float(np.min(w)) if w.size else 0.0,
                "max": float(np.max(w)) if w.size else 0.0,
                "mean": float(np.mean(w)) if w.size else 0.0,
            }
        )
    if not rows:
        return {
            "wg_weight_norm_mean": "",
            "wg_weight_norm_max": "",
            "wg_weight_rms_mean": "",
            "wg_count": 0,
        }
    return {
        "wg_weight_norm_mean": float(np.mean([r["weight_norm"] for r in rows])),
        "wg_weight_norm_max": float(np.max([r["weight_norm"] for r in rows])),
        "wg_weight_rms_mean": float(np.mean([r["weight_rms"] for r in rows])),
        "wg_count": len(rows),
    }


def parse_gate_static(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: List[dict] = []
    aggregates: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("gate_static_l"):
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            m = re.match(r"gate_static_l(\d+)_h(\d+)_(.+)", k.strip())
            if not m:
                continue
            layer, head, field = int(m.group(1)), int(m.group(2)), m.group(3)
            found = next((r for r in rows if r["layer"] == layer and r["head"] == head), None)
            if found is None:
                found = {"layer": layer, "head": head}
                rows.append(found)
            found[field] = fnum(v)
        elif line.startswith("gate_static_") and "=" in line:
            k, v = line.split("=", 1)
            aggregates[k.strip()] = fnum(v)
    for row in rows:
        for key in ("mean_of_head_means", "min_head_mean", "max_head_mean"):
            row[key] = aggregates.get(key, "")
    return rows


def parse_gate_trajectory(result: Dict[str, str]) -> List[dict]:
    rows: List[dict] = []
    by_key: Dict[Tuple[int, int], dict] = {}
    for key, value in result.items():
        m = re.match(r"gate_training_trajectory_l(\d+)_h(\d+)_(.+)", key)
        if not m:
            continue
        layer, head, field = int(m.group(1)), int(m.group(2)), m.group(3)
        item = by_key.setdefault((layer, head), {"layer": layer, "head": head})
        item[field] = fnum(value)
    rows.extend(by_key.values())
    return rows


def balanced(val: float, dev: float) -> float:
    return (val + dev) / 2.0


def arm_dir(report_root: Path, mult: str, arm: str) -> Path:
    return report_root / f"lr{mult}" / arm.lower()


def find_result(arm_path: Path, steps: int) -> Optional[Path]:
    pattern = f"seed1-l19-v1024-t32-d64-f128-steps{steps}-result.txt"
    direct = arm_path / pattern
    if direct.is_file():
        return direct
    matches = list(arm_path.glob("seed1-l19-v1024-t32-d64-f128-steps*-result.txt"))
    return matches[0] if matches else None


def load_eval(arm_path: Path, step: int) -> Optional[Dict[str, str]]:
    path = arm_path / f"eval256-step{step}-htp.txt"
    if not path.is_file():
        # fallback to directory form used by the training-time copy
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
    runtime_rows: List[dict] = []
    stability_rows: List[dict] = []
    gate_static_rows: List[dict] = []
    gate_traj_rows: List[dict] = []
    wgate_rows: List[dict] = []
    telemetry_rows: List[dict] = []
    device_rows: List[dict] = []
    shared_rows: List[dict] = []
    arm_summaries: Dict[Tuple[str, str], dict] = {}

    for mult in MULTS:
        for arm in ARMS:
            arm_path = arm_dir(report_root, mult, arm)
            result_path = find_result(arm_path, args.steps)
            result = parse_kv(result_path) if result_path else {}
            hard_flags: List[str] = []
            soft_flags: List[str] = []
            status = "MISSING"
            if result:
                status = result.get("status", "UNKNOWN")
                if result.get("all_steps_finite") != "true":
                    hard_flags.append("all_steps_finite")
                if result.get("cpu_fallback") == "true" or result.get("fallback") == "true":
                    hard_flags.append("cpu_fallback")
                if result.get("nan_detected") == "true":
                    hard_flags.append("nan_detected")
                if result.get("inf_detected") == "true":
                    hard_flags.append("inf_detected")
                if result.get("qnn_return_code_success") != "true":
                    hard_flags.append("qnn_return_code_success")
                for key in (
                    "api_trace_graph_execute_failure_count",
                    "hvx_rpc_failure_count",
                    "hvx_fallback_count",
                    "hvx_nonfinite_count",
                ):
                    if key in result and inum(result[key]) != 0:
                        hard_flags.append(f"{key}={result[key]}")
                thermal_after = result.get("android_thermal_status_after", "")
                thermal_before = result.get("android_thermal_status_before", "")
                if thermal_after and inum(thermal_after) >= 3:
                    soft_flags.append(f"thermal_status_after={thermal_after}")
                if thermal_before and inum(thermal_before) >= 3:
                    soft_flags.append(f"thermal_status_before={thermal_before}")
                if inum(result.get("focus_takeover_count", "0")) != 0:
                    soft_flags.append("focus_takeover")
            elif arm_path.exists():
                status = "INCOMPLETE"

            if hard_flags:
                stability = "UNSTABLE"
            elif soft_flags or status != "SUCCESS":
                stability = "MARGINAL" if status == "SUCCESS" else status
            else:
                stability = "STABLE"

            # quality trajectory
            eval_meta = []
            for step in eval_steps:
                ev = load_eval(arm_path, step)
                if not ev:
                    continue
                val = fnum(ev.get("validation_bits_per_utf8_byte"))
                dev = fnum(ev.get("development_bits_per_utf8_byte"))
                bal = balanced(val, dev)
                eval_meta.append((step, val, dev, bal, ev))

            for step, val, dev, bal, ev in eval_meta:
                quality_rows.append(
                    {
                        "lr_multiplier": mult,
                        "arm": arm,
                        "step": step,
                        "validation_bpb": f"{val:.9f}",
                        "development_bpb": f"{dev:.9f}",
                        "balanced_bpb": f"{bal:.9f}",
                        "status": "OK" if math.isfinite(bal) else "NONFINITE",
                        "validation_chunks": ev.get("validation_chunks", "256"),
                        "development_chunks": ev.get("development_chunks", "256"),
                        "final_split_used": "false",
                    }
                )

            # runtime / telemetry from result
            training_seconds = fnum(result.get("training_total_seconds", ""))
            step_ms = fnum(result.get("training_step_ms", ""))
            run_bytes = inum(result.get("run_target_utf8_bytes_seen", ""))
            completed_steps = inum(result.get("completed_steps", ""))
            runtime_rows.append(
                {
                    "lr_multiplier": mult,
                    "arm": arm,
                    "training_wall_s": training_seconds,
                    "training_step_ms": step_ms,
                    "fwd_backward_ms_per_update": fnum(result.get("fwd_backward_ms_per_update", "")),
                    "muon_ms_per_update": fnum(result.get("muon_ms_per_update", "")),
                    "aux_adam_ms_per_update": fnum(result.get("aux_adam_ms_per_update", "")),
                    "run_target_utf8_bytes_seen": run_bytes,
                    "completed_steps": inum(result.get("completed_steps", "")),
                    "checkpoint_count": inum(result.get("checkpoint_count", "")),
                    "status": status,
                }
            )
            device_rows.append(
                {
                    "lr_multiplier": mult,
                    "arm": arm,
                    "qnn_return_code_success": result.get("qnn_return_code_success", ""),
                    "hvx_rpc_failure_count": result.get("hvx_rpc_failure_count", ""),
                    "hvx_fallback_count": result.get("hvx_fallback_count", ""),
                    "hvx_nonfinite_count": result.get("hvx_nonfinite_count", ""),
                    "cpu_fallback": result.get("cpu_fallback", ""),
                    "nan_detected": result.get("nan_detected", ""),
                    "inf_detected": result.get("inf_detected", ""),
                    "api_trace_graph_execute_failure_count": result.get("api_trace_graph_execute_failure_count", ""),
                    "focus_takeover_count": result.get("focus_takeover_count", ""),
                    "thermal_status_before": result.get("android_thermal_status_before", ""),
                    "thermal_status_after": result.get("android_thermal_status_after", ""),
                    "battery_temperature_c_before": result.get("battery_temperature_c_before", ""),
                    "battery_temperature_c_after": result.get("battery_temperature_c_after", ""),
                }
            )
            stability_rows.append(
                {
                    "lr_multiplier": mult,
                    "arm": arm,
                    "status": status,
                    "stability": stability,
                    "hard_flags": "|".join(hard_flags),
                    "soft_flags": "|".join(soft_flags),
                    "completed_steps": inum(result.get("completed_steps", "")),
                }
            )
            telemetry_rows.append(
                {
                    "lr_multiplier": mult,
                    "arm": arm,
                    "training_loss_first": fnum(result.get("first_loss", result.get("step0_loss_htp", ""))),
                    "training_loss_last": fnum(result.get("last_loss", "")),
                    "parameter_count": inum(result.get("parameter_element_count", "")),
                    "initial_parameter_hash": result.get("initial_parameter_hash", ""),
                    "final_parameter_hash": result.get("final_parameter_hash", ""),
                    "muon_lr": result.get("muon_lr", result.get("learning_rate", "")),
                    "aux_adam_lr": result.get("aux_adam_lr", result.get("learning_rate", "")),
                    "checkpoint_format": result.get("checkpoint_format", ""),
                }
            )

            # shared hash + wg geometry from final checkpoint if present
            ckpt_path = arm_path / f"htp-seed1-l19-t32-d64-f128-step{args.steps}.ckpt"
            if ckpt_path.is_file():
                try:
                    meta, params = decode_checkpoint(ckpt_path)
                    shared_rows.append(
                        {
                            "lr_multiplier": mult,
                            "arm": arm,
                            "seed": meta.get("seed", ""),
                            "step": meta.get("step", ""),
                            "attention_gate": meta.get("attention_gate", ""),
                            "muon_lr_runtime": meta.get("muon_lr", ""),
                            "aux_adam_lr_runtime": meta.get("aux_adam_lr", ""),
                            "shared_parameter_hash": shared_parameter_hash(params),
                            "parameter_count": sum(int(p.rows) * int(p.columns) for p in params),
                        }
                    )
                    wg = wg_geometry(params)
                    wgate_rows.append({"lr_multiplier": mult, "arm": arm, "step": args.steps, **wg})
                except Exception as exc:  # noqa: BLE001
                    shared_rows.append(
                        {
                            "lr_multiplier": mult,
                            "arm": arm,
                            "error": str(exc),
                        }
                    )

            if arm == "G1":
                for step in eval_steps:
                    diag = arm_path / f"gate-static-step{step}.txt"
                    rows = parse_gate_static(diag)
                    for row in rows:
                        gate_static_rows.append({"lr_multiplier": mult, "step": step, **row})
                traj = parse_gate_trajectory(result)
                for row in traj:
                    gate_traj_rows.append({"lr_multiplier": mult, **row})

            arm_summaries[(mult, arm)] = {
                "eval": eval_meta,
                "training_seconds": training_seconds,
                "run_bytes": run_bytes,
                "completed_steps": completed_steps,
                "stability": stability,
                "result": result,
            }

    # paired quality table
    paired_rows: List[dict] = []
    for mult in MULTS:
        for step in eval_steps:
            ctrl = next((r for r in quality_rows if r["lr_multiplier"] == mult and r["arm"] == "Control" and r["step"] == step), None)
            g1 = next((r for r in quality_rows if r["lr_multiplier"] == mult and r["arm"] == "G1" and r["step"] == step), None)
            if not ctrl or not g1:
                paired_rows.append(
                    {
                        "lr_multiplier": mult,
                        "step": step,
                        "control_val": "CENSORED" if not ctrl else ctrl["validation_bpb"],
                        "g1_val": "CENSORED" if not g1 else g1["validation_bpb"],
                        "delta_val": "CENSORED",
                        "control_dev": "CENSORED" if not ctrl else ctrl["development_bpb"],
                        "g1_dev": "CENSORED" if not g1 else g1["development_bpb"],
                        "delta_dev": "CENSORED",
                        "control_balanced": "CENSORED" if not ctrl else ctrl["balanced_bpb"],
                        "g1_balanced": "CENSORED" if not g1 else g1["balanced_bpb"],
                        "delta_balanced": "CENSORED",
                    }
                )
                continue
            cv, gv = fnum(ctrl["validation_bpb"]), fnum(g1["validation_bpb"])
            cd, gd = fnum(ctrl["development_bpb"]), fnum(g1["development_bpb"])
            cb, gb = fnum(ctrl["balanced_bpb"]), fnum(g1["balanced_bpb"])
            paired_rows.append(
                {
                    "lr_multiplier": mult,
                    "step": step,
                    "control_val": f"{cv:.9f}",
                    "g1_val": f"{gv:.9f}",
                    "delta_val": f"{gv - cv:.9f}",
                    "control_dev": f"{cd:.9f}",
                    "g1_dev": f"{gd:.9f}",
                    "delta_dev": f"{gd - cd:.9f}",
                    "control_balanced": f"{cb:.9f}",
                    "g1_balanced": f"{gb:.9f}",
                    "delta_balanced": f"{gb - cb:.9f}",
                }
            )

    # time-to-bpb (primary = first observed eval checkpoint at or below target)
    # Wall semantics: Muon-hybrid reports emit only full-run training totals.
    # Checkpoint-file mtimes are not distributed across training, so cumulative
    # training wall at an intermediate checkpoint cannot be recovered. Use the
    # measured full-run wall only when the first-hit step equals completed
    # steps; otherwise emit NOT_MEASURED. Never reuse the 500-step total as a
    # step-400 target wall.
    def estimated_original_bytes_to_target(step: int) -> str:
        if step <= 0 or CANONICAL_8000_STEPS <= 0:
            return "NOT_MEASURED"
        est = CANONICAL_8000_STEP_UTF8_BYTES * step / CANONICAL_8000_STEPS
        return f"{est:.1f}"

    t2b_rows: List[dict] = []
    for target in TARGETS:
        best: Dict[str, dict] = {}
        for arm in ARMS:
            candidates = []
            for mult in MULTS:
                summary = arm_summaries.get((mult, arm))
                if not summary:
                    continue
                completed = summary.get("completed_steps", -1)
                for step, val, dev, bal, _ev in summary["eval"]:
                    if math.isfinite(bal) and bal <= target:
                        wall = (
                            summary["training_seconds"]
                            if completed == step and math.isfinite(summary["training_seconds"])
                            else float("nan")
                        )
                        candidates.append(
                            {
                                "multiplier": mult,
                                "step": step,
                                "balanced": bal,
                                "checkpoint_training_wall_s": wall,
                                "estimated_original_bytes_to_target": estimated_original_bytes_to_target(step),
                            }
                        )
            if candidates:
                candidates.sort(key=lambda x: (x["step"], x["balanced"]))
                best[arm] = candidates[0]
        ctrl = best.get("Control")
        g1 = best.get("G1")
        ctrl_wall = ctrl["checkpoint_training_wall_s"] if ctrl else float("nan")
        g1_wall = g1["checkpoint_training_wall_s"] if g1 else float("nan")
        t2b_rows.append(
            {
                "target_bpb": f"{target:.2f}",
                "control_best_lr": ctrl["multiplier"] if ctrl else "NOT_REACHED",
                "control_step": ctrl["step"] if ctrl else ">500",
                "control_checkpoint_training_wall_s": (
                    f"{ctrl_wall:.3f}" if ctrl and math.isfinite(ctrl_wall) else "NOT_MEASURED"
                ) if ctrl else "NOT_REACHED",
                "control_estimated_original_bytes_to_target": (
                    ctrl["estimated_original_bytes_to_target"] if ctrl else "NOT_REACHED"
                ),
                "g1_best_lr": g1["multiplier"] if g1 else "NOT_REACHED",
                "g1_step": g1["step"] if g1 else ">500",
                "g1_checkpoint_training_wall_s": (
                    f"{g1_wall:.3f}" if g1 and math.isfinite(g1_wall) else "NOT_MEASURED"
                ) if g1 else "NOT_REACHED",
                "g1_estimated_original_bytes_to_target": (
                    g1["estimated_original_bytes_to_target"] if g1 else "NOT_REACHED"
                ),
                "delta_step": (g1["step"] - ctrl["step"]) if ctrl and g1 else "CENSORED",
                "delta_checkpoint_training_wall_s": (
                    f"{g1_wall - ctrl_wall:.3f}"
                    if ctrl and g1 and math.isfinite(ctrl_wall) and math.isfinite(g1_wall)
                    else "CENSORED"
                ),
                "delta_estimated_bytes": (
                    f"{float(g1['estimated_original_bytes_to_target']) - float(ctrl['estimated_original_bytes_to_target']):.1f}"
                    if ctrl and g1
                    else "CENSORED"
                ),
            }
        )

    # max stable LR
    max_stable = {}
    for arm in ARMS:
        stable_mults = []
        for mult in MULTS:
            row = next((r for r in stability_rows if r["lr_multiplier"] == mult and r["arm"] == arm), None)
            if row and row["stability"] == "STABLE":
                stable_mults.append(mult)
            elif row and row["stability"] == "MARGINAL":
                stable_mults.append(mult)  # still usable for max stable region with flags
        # max stable = highest mult classified STABLE or MARGINAL without UNSTABLE
        usable = []
        for mult in MULTS:
            row = next((r for r in stability_rows if r["lr_multiplier"] == mult and r["arm"] == arm), None)
            if row and row["stability"] in ("STABLE", "MARGINAL"):
                usable.append(mult)
        max_stable[arm] = usable[-1] if usable else "NONE"

    # write all CSVs
    write_csv(results_root / "quality.csv", quality_rows, [
        "lr_multiplier", "arm", "step", "validation_bpb", "development_bpb", "balanced_bpb",
        "status", "validation_chunks", "development_chunks", "final_split_used",
    ])
    write_csv(results_root / "quality-paired.csv", paired_rows, [
        "lr_multiplier", "step", "control_val", "g1_val", "delta_val", "control_dev", "g1_dev",
        "delta_dev", "control_balanced", "g1_balanced", "delta_balanced",
    ])
    write_csv(results_root / "time-to-bpb.csv", t2b_rows, [
        "target_bpb", "control_best_lr", "control_step",
        "control_checkpoint_training_wall_s", "control_estimated_original_bytes_to_target",
        "g1_best_lr", "g1_step",
        "g1_checkpoint_training_wall_s", "g1_estimated_original_bytes_to_target",
        "delta_step", "delta_checkpoint_training_wall_s", "delta_estimated_bytes",
    ])
    write_csv(results_root / "runtime.csv", runtime_rows, [
        "lr_multiplier", "arm", "training_wall_s", "training_step_ms", "fwd_backward_ms_per_update",
        "muon_ms_per_update", "aux_adam_ms_per_update", "run_target_utf8_bytes_seen",
        "completed_steps", "checkpoint_count", "status",
    ])
    write_csv(results_root / "stability.csv", stability_rows, [
        "lr_multiplier", "arm", "status", "stability", "hard_flags", "soft_flags", "completed_steps",
    ])
    write_csv(results_root / "training-telemetry.csv", telemetry_rows, [
        "lr_multiplier", "arm", "training_loss_first", "training_loss_last", "parameter_count",
        "initial_parameter_hash", "final_parameter_hash", "muon_lr", "aux_adam_lr", "checkpoint_format",
    ])
    write_csv(results_root / "device-health.csv", device_rows, [
        "lr_multiplier", "arm", "qnn_return_code_success", "hvx_rpc_failure_count", "hvx_fallback_count",
        "hvx_nonfinite_count", "cpu_fallback", "nan_detected", "inf_detected",
        "api_trace_graph_execute_failure_count", "focus_takeover_count",
        "thermal_status_before", "thermal_status_after",
        "battery_temperature_c_before", "battery_temperature_c_after",
    ])
    write_csv(results_root / "shared-parameter-hash.csv", shared_rows, [
        "lr_multiplier", "arm", "seed", "step", "attention_gate", "muon_lr_runtime",
        "aux_adam_lr_runtime", "shared_parameter_hash", "parameter_count", "error",
    ])
    write_csv(results_root / "gate-static.csv", gate_static_rows, [
        "lr_multiplier", "step", "layer", "head", "mean", "stddev", "min", "max",
        "below_0_1_fraction", "above_0_9_fraction", "mean_of_head_means",
        "min_head_mean", "max_head_mean",
    ])
    write_csv(results_root / "gate-trajectory.csv", gate_traj_rows, [
        "lr_multiplier", "layer", "head", "mean", "stddev", "min", "max",
        "below_0_1_fraction", "above_0_9_fraction",
    ])
    write_csv(results_root / "wgate-geometry.csv", wgate_rows, [
        "lr_multiplier", "arm", "step", "wg_weight_norm_mean", "wg_weight_norm_max",
        "wg_weight_rms_mean", "wg_count",
    ])

    summary = {
        "schema": "G1_LR_STRESS_SUMMARY_V1",
        "steps": args.steps,
        "eval_steps": eval_steps,
        "targets": TARGETS,
        "max_stable_multiplier": max_stable,
        "historical_1x_sanity_balanced": {
            "control_step500": 2.896905,
            "g1_step500": 2.864193,
        },
    }
    (results_root / "stress-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"analysis_written={results_root}")
    print(f"max_stable_control={max_stable.get('Control')}")
    print(f"max_stable_g1={max_stable.get('G1')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
