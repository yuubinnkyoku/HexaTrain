#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Phase-2 Muon row-geometry audit over existing NPRTCKPTV4/V5 checkpoints.

Reads only already-trained quality checkpoints. Does not train, does not
touch the device, and does not modify optimizer identity.

Semantic row direction follows transformer_parameter_metadata fan_out_axis:
storage is [input, output], so fan_out_axis=1 means each output neuron is a
column of the stored matrix. W1/W2 therefore use output-neuron vectors, not
raw storage rows.

Metrics per Muon matrix / step:
  max / median / RMS semantic row norm
  spectral norm (largest singular value of the stored matrix)
  effective rank (participation ratio of singular values)
  row coherence (max |cos| among sampled semantic rows)
  angular update vs previous step (mean / max radians)

A simple drift verdict compares step-order correlations of max row norm and
spectral norm. Weak drift lowers Muown implementation priority.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC_V4 = b"NPRTCKPTV4\n"
MAGIC_V5 = b"NPRTCKPTV5\n"
ROLE_MUON = 1
ROLE_AUX_ADAM = 2


@dataclass
class Parameter:
    name: str
    role: int
    rows: int
    columns: int
    values: np.ndarray  # shape (rows, columns), storage [input, output]
    fan_out_axis: int  # 1 for all current Muon matrices


@dataclass
class Checkpoint:
    path: Path
    magic: str
    step: int
    seed: int
    vocabulary: int
    tokens: int
    dimension: int
    feed_forward: int
    layers: int
    heads: int
    parameters: List[Parameter]


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
        return value

    def string(self) -> str:
        length = self.u32()
        if length < 1 or length > 4096 or self.remaining() < length:
            raise ValueError("STRING_INVALID")
        value = self.data[self.offset : self.offset + length].decode("utf-8")
        self.offset += length
        return value

    def floats(self, count: int) -> np.ndarray:
        # NPRTCKPTV4 value arrays are length-prefixed with a big-endian u64.
        actual = self.u64()
        if actual != count:
            raise ValueError(f"VALUES_COUNT_MISMATCH:expected={count}:actual={actual}")
        nbytes = count * 4
        if count < 0 or self.remaining() < nbytes:
            raise ValueError("VALUES_TRUNCATED")
        raw = self.data[self.offset : self.offset + nbytes]
        self.offset += nbytes
        return np.frombuffer(raw, dtype=">f4").astype(np.float32)


def load_metadata_fan_out(metadata_path: Path) -> Dict[str, int]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: Dict[str, int] = {}
    for entry in payload.get("parameter_definitions", []):
        if entry.get("role") == "MUON":
            result[entry["suffix"]] = int(entry["fan_out_axis"])
    return result


def decode_checkpoint(path: Path, fan_out: Dict[str, int]) -> Checkpoint:
    data = path.read_bytes()
    reader = ByteReader(data)
    if data.startswith(MAGIC_V4):
        magic = "NPRTCKPTV4"
        gated = False
        reader.offset = len(MAGIC_V4)
    elif data.startswith(MAGIC_V5):
        magic = "NPRTCKPTV5"
        gated = True
        reader.offset = len(MAGIC_V5)
    else:
        raise ValueError(f"CHECKPOINT_MAGIC_MISMATCH:{path}")

    vocabulary = reader.u32()
    tokens = reader.u32()
    dimension = reader.u32()
    feed_forward = reader.u32()
    layers = reader.u32()
    heads = reader.u32()
    _epsilon = reader.f32()
    if gated:
        _gate = reader.u32()
    seed = reader.u32()
    step = reader.u64()
    _tokenizer_kind = reader.string()
    _tokenizer_hash = reader.string()
    _dataset_hash = reader.string()
    _record_index = reader.u64()
    _token_offset = reader.u64()
    _epoch = reader.u64()
    _exposed = reader.u64()
    _order_seed = reader.u64()
    _optimizer_identity = reader.string()
    for _ in range(5):
        reader.f32()  # muon/adam learning rates and muon momentum
    reader.u32()  # nesterov
    reader.u32()  # ns steps
    for _ in range(5):
        reader.f32()  # adam betas/epsilon + weight decays
    reader.u32()  # decay start
    reader.u32()  # decay end
    reader.u32()  # schedule total
    schema_version = reader.u32()
    _registry_version = reader.u32()
    registry_count = reader.u32()
    expected_schema = 5 if gated else 4
    if schema_version != expected_schema:
        raise ValueError(f"SCHEMA_VERSION_MISMATCH:{path}")

    parameters: List[Parameter] = []
    for _ in range(registry_count):
        name = reader.string()
        role = reader.u32()
        rows = reader.u32()
        columns = reader.u32()
        elements = rows * columns
        values = reader.floats(elements)
        if role == ROLE_MUON:
            reader.floats(elements)  # momentum
        elif role == ROLE_AUX_ADAM:
            reader.floats(elements)  # adam m
            reader.floats(elements)  # adam v
        else:
            raise ValueError(f"PARAMETER_ROLE_INVALID:{path}:{name}")
        matrix = values.reshape((rows, columns))
        suffix = name.split(".", 1)[-1] if name.startswith("layer_") else name
        axis = fan_out.get(suffix, 1 if role == ROLE_MUON else -1)
        parameters.append(
            Parameter(
                name=name,
                role=role,
                rows=rows,
                columns=columns,
                values=matrix,
                fan_out_axis=axis,
            )
        )

    if reader.remaining() != 0:
        raise ValueError(f"TRAILING_BYTES:{path}")

    return Checkpoint(
        path=path,
        magic=magic,
        step=step,
        seed=seed,
        vocabulary=vocabulary,
        tokens=tokens,
        dimension=dimension,
        feed_forward=feed_forward,
        layers=layers,
        heads=heads,
        parameters=parameters,
    )


def semantic_rows(matrix: np.ndarray, fan_out_axis: int) -> np.ndarray:
    """Return shape (n_rows_semantic, dim) with output-neuron vectors as rows."""
    if fan_out_axis == 1:
        return matrix.T
    if fan_out_axis == 0:
        return matrix
    raise ValueError(f"UNSUPPORTED_FAN_OUT_AXIS:{fan_out_axis}")


def effective_rank(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values, dtype=np.float64)
    energy = s * s
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    p = energy / total
    # Participation ratio: (sum s^2)^2 / sum s^4
    denom = float(np.sum(p * p))
    return float(1.0 / denom) if denom > 0.0 else 0.0


def row_coherence(rows: np.ndarray, max_rows: int = 32, seed: int = 0) -> float:
    if rows.shape[0] < 2:
        return 0.0
    if rows.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(rows.shape[0], size=max_rows, replace=False)
        sample = rows[idx]
    else:
        sample = rows
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = sample / norms
    gram = unit @ unit.T
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


def angular_update(prev: np.ndarray, curr: np.ndarray) -> Tuple[float, float]:
    if prev.shape != curr.shape:
        raise ValueError("SHAPE_MISMATCH_ANGULAR")
    # Per semantic-row angle between weight vectors.
    dot = np.sum(prev * curr, axis=1)
    nprev = np.linalg.norm(prev, axis=1)
    ncurr = np.linalg.norm(curr, axis=1)
    denom = np.maximum(nprev * ncurr, 1e-12)
    cos = np.clip(dot / denom, -1.0, 1.0)
    angles = np.arccos(cos)
    return float(np.mean(angles)), float(np.max(angles))


def matrix_metrics(
    parameter: Parameter,
    previous: Optional[np.ndarray] = None,
    coherence_rows: int = 32,
) -> Dict[str, float]:
    rows = semantic_rows(parameter.values, parameter.fan_out_axis)
    norms = np.linalg.norm(rows, axis=1)
    # SVD on stored matrix: singular values are orientation-invariant.
    try:
        singular = np.linalg.svd(parameter.values.astype(np.float64), compute_uv=False)
    except np.linalg.LinAlgError:
        singular = np.array([float(np.linalg.norm(parameter.values))])
    out: Dict[str, float] = {
        "row_norm_max": float(np.max(norms)),
        "row_norm_median": float(np.median(norms)),
        "row_norm_rms": float(np.sqrt(np.mean(norms * norms))),
        "spectral_norm": float(singular[0]) if singular.size else 0.0,
        "effective_rank": effective_rank(singular),
        "row_coherence": row_coherence(rows, max_rows=coherence_rows),
    }
    if previous is not None:
        mean_ang, max_ang = angular_update(previous, rows)
        out["angular_update_mean"] = mean_ang
        out["angular_update_max"] = max_ang
    return out


def collect_checkpoints(paths: Sequence[Path]) -> List[Checkpoint]:
    return sorted(paths, key=lambda p: p.name)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="NPRTCKPTV4/V5 checkpoint path (repeatable)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata/transformer_parameter_metadata.json"),
        help="Parameter metadata SSOT (fan_out_axis)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/muon-geometry-audit"),
        help="Directory for CSV/JSON evidence",
    )
    parser.add_argument(
        "--coherence-rows",
        type=int,
        default=32,
        help="Max semantic rows sampled for coherence",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Optional comma-separated step filter, e.g. 500,1000,2000,4000,6000,8000",
    )
    args = parser.parse_args(argv)

    step_filter: Optional[set[int]] = None
    if args.steps.strip():
        step_filter = {int(part.strip()) for part in args.steps.split(",") if part.strip()}

    fan_out = load_metadata_fan_out(args.metadata)
    if not fan_out:
        raise SystemExit("METADATA_HAS_NO_MUON_DEFINITIONS")

    checkpoints: List[Checkpoint] = []
    for path in args.checkpoint:
        if not path.is_file():
            raise SystemExit(f"CHECKPOINT_MISSING:{path}")
        ckpt = decode_checkpoint(path, fan_out)
        if step_filter is not None and ckpt.step not in step_filter:
            continue
        checkpoints.append(ckpt)

    if not checkpoints:
        raise SystemExit("NO_CHECKPOINTS_SELECTED")

    checkpoints = sorted(checkpoints, key=lambda c: (c.seed, c.step))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # previous semantic rows keyed by parameter name (same seed chain)
    previous_rows: Dict[Tuple[int, str], np.ndarray] = {}
    detail_rows: List[Dict[str, object]] = []

    for ckpt in checkpoints:
        muon_params = [p for p in ckpt.parameters if p.role == ROLE_MUON]
        if len(muon_params) != 114:
            raise SystemExit(
                f"UNEXPECTED_MUON_MATRIX_COUNT:{ckpt.path}:expected=114:actual={len(muon_params)}"
            )
        for parameter in muon_params:
            prev_key = (ckpt.seed, parameter.name)
            prev = previous_rows.get(prev_key)
            metrics = matrix_metrics(parameter, prev, coherence_rows=args.coherence_rows)
            rows = semantic_rows(parameter.values, parameter.fan_out_axis)
            previous_rows[prev_key] = rows.copy()
            detail_rows.append(
                {
                    "checkpoint": str(ckpt.path).replace("\\", "/"),
                    "seed": ckpt.seed,
                    "step": ckpt.step,
                    "parameter": parameter.name,
                    "rows": parameter.rows,
                    "columns": parameter.columns,
                    "fan_out_axis": parameter.fan_out_axis,
                    **metrics,
                }
            )

    detail_path = args.out_dir / "muon-row-geometry-detail.csv"
    fieldnames = list(detail_rows[0].keys())
    for row in detail_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)

    # Aggregate per seed/step and per seed/parameter for drift analysis.
    by_step: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    by_param: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_step[(int(row["seed"]), int(row["step"]))].append(row)
        by_param[(int(row["seed"]), str(row["parameter"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (seed, step), items in sorted(by_step.items()):
        max_rows = [float(item["row_norm_max"]) for item in items]
        spectral = [float(item["spectral_norm"]) for item in items]
        coherence = [float(item["row_coherence"]) for item in items]
        eff_rank = [float(item["effective_rank"]) for item in items]
        ang_mean = [
            float(item["angular_update_mean"])
            for item in items
            if "angular_update_mean" in item and item["angular_update_mean"] is not None
        ]
        summary_rows.append(
            {
                "seed": seed,
                "step": step,
                "matrix_count": len(items),
                "row_norm_max_mean": float(np.mean(max_rows)),
                "row_norm_max_max": float(np.max(max_rows)),
                "row_norm_rms_mean": float(
                    np.mean([float(item["row_norm_rms"]) for item in items])
                ),
                "spectral_norm_mean": float(np.mean(spectral)),
                "spectral_norm_max": float(np.max(spectral)),
                "row_coherence_mean": float(np.mean(coherence)),
                "row_coherence_max": float(np.max(coherence)),
                "effective_rank_mean": float(np.mean(eff_rank)),
                "angular_update_mean_mean": float(np.mean(ang_mean)) if ang_mean else "",
                "angular_update_mean_max": float(np.max(ang_mean)) if ang_mean else "",
            }
        )

    summary_path = args.out_dir / "muon-row-geometry-summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Drift verdict: across steps within each seed, correlate max-row-norm
    # growth with spectral-norm growth per matrix, then summarize.
    drift_path = args.out_dir / "muon-row-geometry-drift.csv"
    drift_rows: List[Dict[str, object]] = []
    for (seed, name), items in sorted(by_param.items()):
        items = sorted(items, key=lambda r: int(r["step"]))
        steps = [int(item["step"]) for item in items]
        max_norm = [float(item["row_norm_max"]) for item in items]
        spectral = [float(item["spectral_norm"]) for item in items]
        if len(steps) >= 3:
            corr = pearson(max_norm, spectral)
            max_norm_ratio = max_norm[-1] / max_norm[0] if max_norm[0] > 0 else float("nan")
            spectral_ratio = spectral[-1] / spectral[0] if spectral[0] > 0 else float("nan")
        else:
            corr = float("nan")
            max_norm_ratio = float("nan")
            spectral_ratio = float("nan")
        drift_rows.append(
            {
                "seed": seed,
                "parameter": name,
                "steps": ";".join(str(s) for s in steps),
                "corr_max_row_vs_spectral": corr,
                "max_row_norm_ratio_last_over_first": max_norm_ratio,
                "spectral_norm_ratio_last_over_first": spectral_ratio,
                "max_row_norm_first": max_norm[0] if max_norm else "",
                "max_row_norm_last": max_norm[-1] if max_norm else "",
                "spectral_norm_first": spectral[0] if spectral else "",
                "spectral_norm_last": spectral[-1] if spectral else "",
            }
        )

    if drift_rows:
        with drift_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(drift_rows[0].keys()))
            writer.writeheader()
            writer.writerows(drift_rows)

    correlations = [
        float(row["corr_max_row_vs_spectral"])
        for row in drift_rows
        if not math.isnan(float(row["corr_max_row_vs_spectral"]))
    ]
    ratios = [
        float(row["max_row_norm_ratio_last_over_first"])
        for row in drift_rows
        if not math.isnan(float(row["max_row_norm_ratio_last_over_first"]))
    ]
    spectral_ratios = [
        float(row["spectral_norm_ratio_last_over_first"])
        for row in drift_rows
        if not math.isnan(float(row["spectral_norm_ratio_last_over_first"]))
    ]

    verdict = {
        "checkpoint_count": len(checkpoints),
        "muon_matrix_count_per_checkpoint": 114,
        "detail_rows": len(detail_rows),
        "seeds": sorted({int(row["seed"]) for row in detail_rows}),
        "steps": sorted({int(row["step"]) for row in detail_rows}),
        "corr_max_row_vs_spectral_mean": float(np.mean(correlations)) if correlations else None,
        "corr_max_row_vs_spectral_median": float(np.median(correlations)) if correlations else None,
        "max_row_norm_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "max_row_norm_ratio_median": float(np.median(ratios)) if ratios else None,
        "spectral_norm_ratio_mean": float(np.mean(spectral_ratios)) if spectral_ratios else None,
        "spectral_norm_ratio_median": float(np.median(spectral_ratios)) if spectral_ratios else None,
        "muown_priority_hint": (
            "LOWER_IF_DRIFT_WEAK"
            if ratios and float(np.median(ratios)) < 1.25
            else "KEEP_OR_RAISE_IF_DRIFT_STRONG"
        ),
        "notes": [
            "semantic row = fan_out_axis vector (output neuron), not storage row",
            "spectral norm uses singular values of stored [input, output] matrix",
            "single-run geometry is diagnostic only, not independent reproduction evidence",
        ],
    }
    verdict_path = args.out_dir / "muon-row-geometry-verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(verdict, indent=2))
    print(f"Wrote {detail_path}", file=sys.stderr)
    print(f"Wrote {summary_path}", file=sys.stderr)
    print(f"Wrote {drift_path}", file=sys.stderr)
    print(f"Wrote {verdict_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
