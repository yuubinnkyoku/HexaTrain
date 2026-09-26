#!/usr/bin/env python3
"""V4.1 optimizer-split pre-implementation diagnostic (checkpoint-only + offline replay).

Reads existing NPRTCKPTV4/V5 quality checkpoints. Does not train, does not
touch the device, and does not change production optimizer semantics.

SSOT for parameter orientation is metadata/transformer_parameter_metadata.json
(generated from app/src/main/cpp/transformer_parameter_metadata.h):
  storage is [input, output]; fan_out_axis=1 means semantic output neurons are
  columns. Head-wise split of Wq/Wk/Wv therefore slices the fan_out axis into
  [MODEL, MODEL/HEADS] blocks.

Muon update math matches nicopedia_muon_optimizer (Original Muon):
  momentum' = m * momentum + (1-m) * grad
  nesterov  = (1-m) * grad + m * momentum'   (nesterov=true)
  orthogonal = zeropowerNewtonSchulzFp32(nesterov, rows, cols, NS5)
  scale = sqrt(max(1, fanOut / fanIn))
  param -= muon_lr * scale * orthogonal
NS coefficients (3.4445, -4.7750, 2.0315), epsilon 1e-7 are unchanged.

Offline 1-step replay feeds one shared input matrix into full-matrix and
head-wise Newton-Schulz so only the preconditioner partition differs. When a
true DataCursor batch gradient is not available on host, the stored Muon
momentum is used as the shared input (momentum-proxy replay). Production
training path is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC_V4 = b"NPRTCKPTV4\n"
MAGIC_V5 = b"NPRTCKPTV5\n"
ROLE_MUON = 1
ROLE_AUX_ADAM = 2

NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315
NS_EPS = 1.0e-7
NS_STEPS = 5
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True


@dataclass
class Parameter:
    name: str
    role: int
    rows: int
    columns: int
    values: np.ndarray  # (rows, columns) storage [input, output]
    momentum: Optional[np.ndarray] = None
    adam_m: Optional[np.ndarray] = None
    adam_v: Optional[np.ndarray] = None
    fan_out_axis: int = 1


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
    muon_lr: float
    aux_adam_lr: float
    muon_momentum: float
    muon_nesterov: int
    muon_ns_steps: int
    dataset_hash: str
    record_index: int
    token_offset: int
    epoch: int
    exposed_tokens: int
    order_seed: int
    parameters: List[Parameter] = field(default_factory=list)


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
            raise ValueError(f"VALUES_COUNT_MISMATCH:expected={count}:actual={actual}")
        nbytes = count * 4
        if count < 0 or self.remaining() < nbytes:
            raise ValueError("VALUES_TRUNCATED")
        raw = self.data[self.offset : self.offset + nbytes]
        self.offset += nbytes
        return np.frombuffer(raw, dtype=">f4").astype(np.float32)


def load_metadata(metadata_path: Path) -> Dict[str, dict]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: Dict[str, dict] = {}
    for entry in payload.get("parameter_definitions", []):
        result[entry["suffix"]] = entry
    return result


def suffix_of(name: str) -> str:
    return name.split(".", 1)[-1] if name.startswith("layer_") else name


def decode_checkpoint(path: Path, metadata: Dict[str, dict]) -> Checkpoint:
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
    dataset_hash = reader.string()
    record_index = reader.u64()
    token_offset = reader.u64()
    epoch = reader.u64()
    exposed_tokens = reader.u64()
    order_seed = reader.u64()
    _optimizer_identity = reader.string()
    muon_lr = reader.f32()
    aux_adam_lr = reader.f32()
    _muon_target = reader.f32()
    _adam_target = reader.f32()
    muon_momentum = reader.f32()
    muon_nesterov = reader.u32()
    muon_ns_steps = reader.u32()
    for _ in range(5):
        reader.f32()
    reader.u32()
    reader.u32()
    reader.u32()
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
        values = reader.floats(elements).reshape((rows, columns))
        momentum = None
        adam_m = None
        adam_v = None
        if role == ROLE_MUON:
            momentum = reader.floats(elements).reshape((rows, columns))
        elif role == ROLE_AUX_ADAM:
            adam_m = reader.floats(elements).reshape((rows, columns))
            adam_v = reader.floats(elements).reshape((rows, columns))
        else:
            raise ValueError(f"PARAMETER_ROLE_INVALID:{path}:{name}")
        suffix = suffix_of(name)
        meta = metadata.get(suffix, {})
        fan_out = int(meta.get("fan_out_axis", 1 if role == ROLE_MUON else -1))
        parameters.append(
            Parameter(
                name=name,
                role=role,
                rows=rows,
                columns=columns,
                values=values,
                momentum=momentum,
                adam_m=adam_m,
                adam_v=adam_v,
                fan_out_axis=fan_out,
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
        muon_lr=muon_lr,
        aux_adam_lr=aux_adam_lr,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_ns_steps=muon_ns_steps,
        dataset_hash=dataset_hash,
        record_index=record_index,
        token_offset=token_offset,
        epoch=epoch,
        exposed_tokens=exposed_tokens,
        order_seed=order_seed,
        parameters=parameters,
    )


def find_param(ckpt: Checkpoint, name: str) -> Parameter:
    for p in ckpt.parameters:
        if p.name == name:
            return p
    raise KeyError(f"PARAMETER_NOT_FOUND:{ckpt.path}:{name}")


def head_blocks(matrix: np.ndarray, heads: int, fan_out_axis: int) -> List[np.ndarray]:
    """Slice semantic fan_out axis into head blocks.

    Storage is [input, output]; fan_out_axis=1 => split columns.
    Each block keeps the full fan_in axis: [MODEL, MODEL/HEADS].
    """
    if fan_out_axis != 1:
        raise ValueError(f"SUPPORTED_FAN_OUT_AXIS_1_ONLY:{fan_out_axis}")
    rows, cols = matrix.shape
    if cols % heads != 0:
        raise ValueError(f"HEAD_SPLIT_INVALID:{rows}x{cols}:heads={heads}")
    width = cols // heads
    return [matrix[:, h * width : (h + 1) * width] for h in range(heads)]


def mat_norms(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    fro = float(np.linalg.norm(a))
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    try:
        spectral = float(np.linalg.norm(a, 2))
    except Exception:
        spectral = float("nan")
    max_abs = float(np.max(np.abs(a))) if a.size else 0.0
    return {
        "frobenius": fro,
        "rms": rms,
        "spectral": spectral,
        "max_abs": max_abs,
        "elements": int(a.size),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def angular_step(weight: np.ndarray, update: np.ndarray) -> float:
    """Angle (radians) between weight and update directions."""
    w = np.asarray(weight, dtype=np.float64).ravel()
    u = np.asarray(update, dtype=np.float64).ravel()
    nw = float(np.linalg.norm(w))
    nu = float(np.linalg.norm(u))
    if nw == 0.0 or nu == 0.0:
        return float("nan")
    c = max(-1.0, min(1.0, float(np.dot(w, u) / (nw * nu))))
    return float(math.acos(c))


def zeropower_newton_schulz(
    matrix: np.ndarray, steps: int = NS_STEPS
) -> np.ndarray:
    """Match nicopedia_muon::zeropowerNewtonSchulzFp32 exactly (float32)."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("NS_SHAPE")
    rows, cols = x.shape
    transposed = rows > cols
    if transposed:
        x = np.ascontiguousarray(x.T)
        rows, cols = cols, rows
    else:
        x = np.ascontiguousarray(x)
    norm_sq = float(np.sum(x.astype(np.float64) ** 2))
    denom = math.sqrt(norm_sq) + NS_EPS
    x = (x / np.float32(denom)).astype(np.float32)
    for _ in range(steps):
        a = (x.astype(np.float64) @ x.astype(np.float64).T).astype(np.float32)
        a2 = (a.astype(np.float64) @ a.astype(np.float64)).astype(np.float32)
        b = (NS_B * a.astype(np.float64) + NS_C * a2.astype(np.float64)).astype(
            np.float32
        )
        next_x = (
            NS_A * x.astype(np.float64) + b.astype(np.float64) @ x.astype(np.float64)
        ).astype(np.float32)
        x = next_x
    if transposed:
        x = np.ascontiguousarray(x.T)
    return x


def muon_scale(rows: int, cols: int, fan_out_axis: int = 1) -> float:
    if fan_out_axis == 1:
        fan_out, fan_in = cols, rows
    else:
        fan_out, fan_in = rows, cols
    return float(math.sqrt(max(1.0, float(fan_out) / float(fan_in))))


def muon_update_from_input(
    weight: np.ndarray,
    shared_input: np.ndarray,
    fan_out_axis: int,
    muon_lr: float,
) -> np.ndarray:
    """Offline Muon update (the applied delta, not the new weight)."""
    ortho = zeropower_newton_schulz(shared_input, NS_STEPS)
    scale = muon_scale(weight.shape[0], weight.shape[1], fan_out_axis)
    return muon_lr * scale * ortho.astype(np.float64)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def collect_checkpoints(paths: Iterable[Path], metadata: Dict[str, dict]) -> List[Checkpoint]:
    checkpoints = [decode_checkpoint(p, metadata) for p in paths]
    checkpoints.sort(key=lambda c: (c.seed, c.step))
    return checkpoints


def checkpoint_inventory_rows(ckpts: Sequence[Checkpoint]) -> List[dict]:
    rows = []
    for c in ckpts:
        muon = [p for p in c.parameters if p.role == ROLE_MUON]
        aux = [p for p in c.parameters if p.role == ROLE_AUX_ADAM]
        rows.append(
            {
                "path": str(c.path).replace("\\", "/"),
                "magic": c.magic,
                "seed": c.seed,
                "step": c.step,
                "vocabulary": c.vocabulary,
                "tokens": c.tokens,
                "dimension": c.dimension,
                "feed_forward": c.feed_forward,
                "layers": c.layers,
                "heads": c.heads,
                "head_dim": c.dimension // c.heads if c.heads else 0,
                "muon_lr": c.muon_lr,
                "aux_adam_lr": c.aux_adam_lr,
                "muon_momentum": c.muon_momentum,
                "muon_nesterov": c.muon_nesterov,
                "muon_ns_steps": c.muon_ns_steps,
                "muon_matrix_count": len(muon),
                "aux_adam_count": len(aux),
                "registry_count": len(c.parameters),
                "dataset_hash": c.dataset_hash,
                "record_index": c.record_index,
                "token_offset": c.token_offset,
                "epoch": c.epoch,
                "exposed_tokens": c.exposed_tokens,
                "order_seed": c.order_seed,
            }
        )
    return rows


def head_checkpoint_rows(
    ckpts: Sequence[Checkpoint], suffixes: Sequence[str]
) -> Tuple[List[dict], List[dict]]:
    raw: List[dict] = []
    summary: List[dict] = []
    for c in ckpts:
        for suffix in suffixes:
            params = [p for p in c.parameters if suffix_of(p.name) == suffix and p.role == ROLE_MUON]
            if not params:
                continue
            for p in params:
                blocks = head_blocks(p.values, c.heads, p.fan_out_axis)
                mom_blocks = (
                    head_blocks(p.momentum, c.heads, p.fan_out_axis)
                    if p.momentum is not None
                    else [None] * c.heads
                )
                head_rms = []
                head_fro = []
                for h, block in enumerate(blocks):
                    w = mat_norms(block)
                    head_rms.append(w["rms"])
                    head_fro.append(w["frobenius"])
                    row = {
                        "seed": c.seed,
                        "step": c.step,
                        "layer_param": p.name,
                        "suffix": suffix,
                        "rows": p.rows,
                        "columns": p.columns,
                        "head": h,
                        "head_rows": block.shape[0],
                        "head_cols": block.shape[1],
                        "weight_frobenius": w["frobenius"],
                        "weight_rms": w["rms"],
                        "weight_spectral": w["spectral"],
                        "weight_max_abs": w["max_abs"],
                        "weight_elements": w["elements"],
                    }
                    mb = mom_blocks[h]
                    if mb is not None:
                        m = mat_norms(mb)
                        row.update(
                            {
                                "momentum_frobenius": m["frobenius"],
                                "momentum_rms": m["rms"],
                                "momentum_spectral": m["spectral"],
                                "momentum_max_abs": m["max_abs"],
                                "weight_momentum_cosine": cosine(block, mb),
                            }
                        )
                    else:
                        row.update(
                            {
                                "momentum_frobenius": "",
                                "momentum_rms": "",
                                "momentum_spectral": "",
                                "momentum_max_abs": "",
                                "weight_momentum_cosine": "",
                            }
                        )
                    raw.append(row)
                if c.heads == 2:
                    r0, r1 = head_rms[0], head_rms[1]
                    f0, f1 = head_fro[0], head_fro[1]
                    ratio_rms = r1 / r0 if r0 else float("nan")
                    ratio_fro = f1 / f0 if f0 else float("nan")
                    cross = cosine(blocks[0], blocks[1]) if blocks[0].shape == blocks[1].shape else float("nan")
                    # Head blocks share fan_in but are different fan_out slices;
                    # cosine between flattened blocks is still a geometry probe
                    # when shapes match (square 64x32 will not). Use pooled stats.
                    pooled = np.concatenate([blocks[0].ravel(), blocks[1].ravel()])
                    mom0 = mom_blocks[0]
                    mom1 = mom_blocks[1]
                    mom_ratio = float("nan")
                    if mom0 is not None and mom1 is not None:
                        m0 = mat_norms(mom0)["rms"]
                        m1 = mat_norms(mom1)["rms"]
                        mom_ratio = m1 / m0 if m0 else float("nan")
                    summary.append(
                        {
                            "seed": c.seed,
                            "step": c.step,
                            "layer_param": p.name,
                            "suffix": suffix,
                            "head0_weight_rms": r0,
                            "head1_weight_rms": r1,
                            "head_rms_ratio_h1_over_h0": ratio_rms,
                            "head0_weight_frobenius": f0,
                            "head1_weight_frobenius": f1,
                            "head_fro_ratio_h1_over_h0": ratio_fro,
                            "head_cosine_flattened": cross,
                            "momentum_rms_ratio_h1_over_h0": mom_ratio,
                            "pooled_weight_rms": mat_norms(pooled.reshape(-1, 1))["rms"],
                        }
                    )
    return raw, summary


def aggregate_head_summary(summary: Sequence[dict]) -> List[dict]:
    """Layer-aggregate imbalance stats per seed/step/suffix."""
    buckets: Dict[Tuple[int, int, str], List[dict]] = {}
    for row in summary:
        key = (int(row["seed"]), int(row["step"]), str(row["suffix"]))
        buckets.setdefault(key, []).append(row)
    out: List[dict] = []
    for (seed, step, suffix), rows in sorted(buckets.items()):
        ratios = np.array(
            [float(r["head_rms_ratio_h1_over_h0"]) for r in rows], dtype=np.float64
        )
        ratios = ratios[np.isfinite(ratios)]
        mom_ratios = np.array(
            [
                float(r["momentum_rms_ratio_h1_over_h0"])
                for r in rows
                if r["momentum_rms_ratio_h1_over_h0"] != ""
                and math.isfinite(float(r["momentum_rms_ratio_h1_over_h0"]))
            ],
            dtype=np.float64,
        )
        if ratios.size == 0:
            continue
        # Imbalance as max(r, 1/r) so both directions count.
        imb = np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-12))
        out.append(
            {
                "seed": seed,
                "step": step,
                "suffix": suffix,
                "layer_count": len(rows),
                "median_rms_ratio_h1_over_h0": float(np.median(ratios)),
                "p90_rms_ratio_h1_over_h0": float(np.percentile(ratios, 90)),
                "max_rms_ratio_h1_over_h0": float(np.max(ratios)),
                "median_imbalance_max_ratio": float(np.median(imb)),
                "p90_imbalance_max_ratio": float(np.percentile(imb, 90)),
                "max_imbalance_max_ratio": float(np.max(imb)),
                "median_momentum_rms_ratio_h1_over_h0": (
                    float(np.median(mom_ratios)) if mom_ratios.size else ""
                ),
                "layers": ",".join(str(r["layer_param"]) for r in rows[:3]) + (
                    f"...(+{len(rows)-3})" if len(rows) > 3 else ""
                ),
            }
        )
    return out


def checkpoint_displacement_rows(
    ckpts: Sequence[Checkpoint], suffixes: Sequence[str]
) -> List[dict]:
    """Consecutive-checkpoint weight / momentum displacement per head."""
    by_seed: Dict[int, List[Checkpoint]] = {}
    for c in ckpts:
        by_seed.setdefault(c.seed, []).append(c)
    rows: List[dict] = []
    for seed, series in sorted(by_seed.items()):
        series = sorted(series, key=lambda c: c.step)
        for prev, curr in zip(series, series[1:]):
            for suffix in suffixes:
                params = [
                    p
                    for p in curr.parameters
                    if suffix_of(p.name) == suffix and p.role == ROLE_MUON
                ]
                for p in params:
                    prev_p = find_param(prev, p.name)
                    dW = p.values.astype(np.float64) - prev_p.values.astype(np.float64)
                    blocks_dW = head_blocks(dW, curr.heads, p.fan_out_axis)
                    blocks_w0 = head_blocks(prev_p.values, curr.heads, p.fan_out_axis)
                    blocks_w1 = head_blocks(p.values, curr.heads, p.fan_out_axis)
                    for h in range(curr.heads):
                        dw = mat_norms(blocks_dW[h])
                        w0 = mat_norms(blocks_w0[h])
                        w1 = mat_norms(blocks_w1[h])
                        rel = dw["frobenius"] / w0["frobenius"] if w0["frobenius"] else float("nan")
                        ang = angular_step(blocks_w0[h], blocks_dW[h])
                        rows.append(
                            {
                                "seed": seed,
                                "step_from": prev.step,
                                "step_to": curr.step,
                                "layer_param": p.name,
                                "suffix": suffix,
                                "head": h,
                                "delta_weight_frobenius": dw["frobenius"],
                                "delta_weight_rms": dw["rms"],
                                "weight_frobenius_from": w0["frobenius"],
                                "weight_frobenius_to": w1["frobenius"],
                                "relative_update_proxy": rel,
                                "angular_displacement_rad": ang,
                                "weight_from_rms": w0["rms"],
                                "weight_to_rms": w1["rms"],
                            }
                        )
    return rows


def replay_rows(
    ckpts: Sequence[Checkpoint],
    suffixes: Sequence[str],
    input_mode: str = "momentum_proxy",
) -> List[dict]:
    """1-step offline Muon: full-matrix vs head-wise, shared input matrix."""
    rows: List[dict] = []
    for c in ckpts:
        for suffix in suffixes:
            params = [
                p
                for p in c.parameters
                if suffix_of(p.name) == suffix and p.role == ROLE_MUON
            ]
            for p in params:
                if input_mode == "momentum_proxy":
                    if p.momentum is None:
                        continue
                    shared = p.momentum.astype(np.float64)
                elif input_mode == "synthetic_grad":
                    rng = np.random.default_rng(10_000 * c.seed + c.step + hash(p.name) % 10_000)
                    shared = rng.standard_normal(p.values.shape).astype(np.float64)
                else:
                    raise ValueError(f"INPUT_MODE:{input_mode}")

                full_update = muon_update_from_input(
                    p.values, shared, p.fan_out_axis, c.muon_lr
                )
                full_block = full_update
                blocks_shared = head_blocks(shared, c.heads, p.fan_out_axis)
                blocks_weight = head_blocks(p.values, c.heads, p.fan_out_axis)
                split_update = np.zeros_like(full_update)
                head_updates = []
                for h, (sw, ww) in enumerate(zip(blocks_shared, blocks_weight)):
                    hu = muon_update_from_input(ww, sw, p.fan_out_axis, c.muon_lr)
                    head_updates.append(hu)
                    width = p.columns // c.heads
                    split_update[:, h * width : (h + 1) * width] = hu

                # Per-head metrics comparing full slice vs split slice.
                width = p.columns // c.heads
                for h in range(c.heads):
                    full_h = full_block[:, h * width : (h + 1) * width]
                    split_h = head_updates[h]
                    w_h = blocks_weight[h]
                    rows.append(
                        {
                            "seed": c.seed,
                            "step": c.step,
                            "layer_param": p.name,
                            "suffix": suffix,
                            "arm": "wqwk" if suffix in ("wq", "wk") else "exploratory",
                            "input_mode": input_mode,
                            "head": h,
                            "update_rms_full": mat_norms(full_h)["rms"],
                            "update_rms_split": mat_norms(split_h)["rms"],
                            "update_norm_full": mat_norms(full_h)["frobenius"],
                            "update_norm_split": mat_norms(split_h)["frobenius"],
                            "update_weight_ratio_full": (
                                mat_norms(full_h)["frobenius"]
                                / mat_norms(w_h)["frobenius"]
                                if mat_norms(w_h)["frobenius"]
                                else float("nan")
                            ),
                            "update_weight_ratio_split": (
                                mat_norms(split_h)["frobenius"]
                                / mat_norms(w_h)["frobenius"]
                                if mat_norms(w_h)["frobenius"]
                                else float("nan")
                            ),
                            "cosine_full_split": cosine(full_h, split_h),
                            "relative_l2_full_split": relative_l2(full_h, split_h),
                            "max_abs_diff": float(np.max(np.abs(full_h - split_h))),
                            "angular_step_full_rad": angular_step(w_h, full_h),
                            "angular_step_split_rad": angular_step(w_h, split_h),
                            "weight_rms": mat_norms(w_h)["rms"],
                            "momentum_rms": mat_norms(blocks_shared[h])["rms"],
                        }
                    )

                # Matrix-level aggregate for this parameter.
                if c.heads == 2:
                    h0 = rows[-2]
                    h1 = rows[-1]
                    n0 = h0["update_norm_split"]
                    n1 = h1["update_norm_split"]
                    a0 = h0["angular_step_split_rad"]
                    a1 = h1["angular_step_split_rad"]
                    rows.append(
                        {
                            "seed": c.seed,
                            "step": c.step,
                            "layer_param": p.name,
                            "suffix": suffix,
                            "arm": h0["arm"],
                            "input_mode": input_mode,
                            "head": "head_ratio",
                            "update_rms_full": mat_norms(full_block)["rms"],
                            "update_rms_split": mat_norms(split_update)["rms"],
                            "update_norm_full": mat_norms(full_block)["frobenius"],
                            "update_norm_split": mat_norms(split_update)["frobenius"],
                            "update_weight_ratio_full": (
                                mat_norms(full_block)["frobenius"]
                                / mat_norms(p.values)["frobenius"]
                                if mat_norms(p.values)["frobenius"]
                                else float("nan")
                            ),
                            "update_weight_ratio_split": (
                                mat_norms(split_update)["frobenius"]
                                / mat_norms(p.values)["frobenius"]
                                if mat_norms(p.values)["frobenius"]
                                else float("nan")
                            ),
                            "cosine_full_split": cosine(full_block, split_update),
                            "relative_l2_full_split": relative_l2(full_block, split_update),
                            "max_abs_diff": float(np.max(np.abs(full_block - split_update))),
                            "angular_step_full_rad": angular_step(p.values, full_block),
                            "angular_step_split_rad": angular_step(p.values, split_update),
                            "weight_rms": mat_norms(p.values)["rms"],
                            "momentum_rms": (
                                mat_norms(shared)["rms"]
                                if input_mode == "momentum_proxy"
                                else mat_norms(shared)["rms"]
                            ),
                            "head0_update_norm_split": n0,
                            "head1_update_norm_split": n1,
                            "head_update_norm_ratio_h1_over_h0": (
                                n1 / n0 if n0 else float("nan")
                            ),
                            "head0_angular_step_split_rad": a0,
                            "head1_angular_step_split_rad": a1,
                            "head_angular_step_ratio_h1_over_h0": (
                                a1 / a0 if a0 else float("nan")
                            ),
                        }
                    )
    return rows


def replay_aggregate_rows(replay: Sequence[dict]) -> List[dict]:
    """Aggregate full-vs-split geometry across layers per seed/step/suffix/arm."""
    buckets: Dict[Tuple[int, int, str, str], List[dict]] = {}
    for row in replay:
        if row.get("head") not in (0, 1, "0", "1"):
            continue
        key = (
            int(row["seed"]),
            int(row["step"]),
            str(row["suffix"]),
            str(row["arm"]),
        )
        buckets.setdefault(key, []).append(row)
    out: List[dict] = []
    for (seed, step, suffix, arm), rows in sorted(buckets.items()):
        cos = np.array([float(r["cosine_full_split"]) for r in rows], dtype=np.float64)
        rel = np.array(
            [float(r["relative_l2_full_split"]) for r in rows], dtype=np.float64
        )
        mad = np.array([float(r["max_abs_diff"]) for r in rows], dtype=np.float64)
        ang_f = np.array(
            [float(r["angular_step_full_rad"]) for r in rows], dtype=np.float64
        )
        ang_s = np.array(
            [float(r["angular_step_split_rad"]) for r in rows], dtype=np.float64
        )
        ratio_n = np.array(
            [float(r["update_norm_split"]) / max(float(r["update_norm_full"]), 1e-12) for r in rows],
            dtype=np.float64,
        )
        out.append(
            {
                "seed": seed,
                "step": step,
                "suffix": suffix,
                "arm": arm,
                "layer_head_count": len(rows),
                "median_cosine_full_split": float(np.median(cos)),
                "p10_cosine_full_split": float(np.percentile(cos, 10)),
                "min_cosine_full_split": float(np.min(cos)),
                "median_relative_l2_full_split": float(np.median(rel)),
                "p90_relative_l2_full_split": float(np.percentile(rel, 90)),
                "max_relative_l2_full_split": float(np.max(rel)),
                "median_max_abs_diff": float(np.median(mad)),
                "median_update_norm_ratio_split_over_full": float(np.median(ratio_n)),
                "median_angular_step_full_rad": float(np.median(ang_f)),
                "median_angular_step_split_rad": float(np.median(ang_s)),
                "median_angular_step_ratio_split_over_full": float(
                    np.median(ang_s / np.maximum(ang_f, 1e-12))
                ),
            }
        )
    return out


def embedding_audit_rows(ckpts: Sequence[Checkpoint]) -> List[dict]:
    rows: List[dict] = []
    for c in ckpts:
        for suffix, expected in (
            ("token_embedding", ("VOCABULARY", "MODEL")),
            ("output_projection", ("MODEL", "VOCABULARY")),
        ):
            p = find_param(c, suffix)
            # Semantic vocabulary axis from SSOT:
            #   token_embedding shape [VOCABULARY, MODEL] => vocab axis 0
            #   output_projection shape [MODEL, VOCABULARY] => vocab axis 1
            if suffix == "token_embedding":
                vocab_axis = 0
                model_axis = 1
                vocab_extent = p.rows
                model_extent = p.columns
            else:
                vocab_axis = 1
                model_axis = 0
                vocab_extent = p.columns
                model_extent = p.rows

            n = p.values.size
            adam_m_bytes = p.adam_m.nbytes if p.adam_m is not None else 0
            adam_v_bytes = p.adam_v.nbytes if p.adam_v is not None else 0
            total_opt_bytes = adam_m_bytes + adam_v_bytes

            w = mat_norms(p.values)
            if p.adam_m is not None:
                m_rms = mat_norms(p.adam_m)["rms"]
                m_fro = mat_norms(p.adam_m)["frobenius"]
            else:
                m_rms = m_fro = float("nan")
            if p.adam_v is not None:
                v_rms = mat_norms(p.adam_v)["rms"]
            else:
                v_rms = float("nan")

            # Approximate Adam update magnitude: m / (sqrt(v)+eps) without bias correction.
            if p.adam_m is not None and p.adam_v is not None:
                upd = p.adam_m / (np.sqrt(p.adam_v) + 1.0e-8)
                u = mat_norms(upd)
            else:
                upd = None
                u = {"rms": float("nan"), "frobenius": float("nan")}

            # Token/vocab-direction norm distribution.
            if vocab_axis == 0:
                # each row is a token
                axis_norms = np.linalg.norm(p.values.astype(np.float64), axis=1)
            else:
                axis_norms = np.linalg.norm(p.values.astype(np.float64), axis=0)
            rows.append(
                {
                    "seed": c.seed,
                    "step": c.step,
                    "suffix": suffix,
                    "shape_rows": p.rows,
                    "shape_cols": p.columns,
                    "semantic_shape": f"{expected[0]}x{expected[1]}",
                    "vocab_axis": vocab_axis,
                    "model_axis": model_axis,
                    "vocab_extent": vocab_extent,
                    "model_extent": model_extent,
                    "parameter_count": n,
                    "adam_m_bytes": adam_m_bytes,
                    "adam_v_bytes": adam_v_bytes,
                    "total_optimizer_state_bytes": total_opt_bytes,
                    "weight_rms": w["rms"],
                    "weight_frobenius": w["frobenius"],
                    "weight_spectral": w["spectral"],
                    "momentum_rms": m_rms,
                    "momentum_frobenius": m_fro,
                    "adam_v_rms": v_rms,
                    "update_rms_approx": u["rms"],
                    "update_frobenius_approx": u["frobenius"],
                    "update_weight_ratio_approx": (
                        u["frobenius"] / w["frobenius"] if w["frobenius"] else float("nan")
                    ),
                    "vocab_dir_norm_mean": float(np.mean(axis_norms)),
                    "vocab_dir_norm_median": float(np.median(axis_norms)),
                    "vocab_dir_norm_p90": float(np.percentile(axis_norms, 90)),
                    "vocab_dir_norm_max": float(np.max(axis_norms)),
                    "vocab_dir_norm_min": float(np.min(axis_norms)),
                    "vocab_dir_norm_rms": float(np.sqrt(np.mean(axis_norms**2))),
                    "vocab_dir_norm_cv": (
                        float(np.std(axis_norms) / np.mean(axis_norms))
                        if np.mean(axis_norms)
                        else float("nan")
                    ),
                }
            )
    return rows


def sinkhorn_balance(
    matrix: np.ndarray, iterations: int = 20, eps: float = 1e-8
) -> np.ndarray:
    """Offline Sinkhorn-style row/column norm balancing (diagnostic only)."""
    x = np.maximum(np.abs(np.asarray(matrix, dtype=np.float64)), eps)
    for _ in range(iterations):
        row = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        x = x / np.maximum(row, eps)
        col = np.sqrt(np.sum(x * x, axis=0, keepdims=True))
        x = x / np.maximum(col, eps)
    # Keep original signs / relative phase approximately via elementwise scale of |update|.
    signed = np.sign(np.asarray(matrix, dtype=np.float64)) * x
    return signed


def sinkhorn_rows(ckpts: Sequence[Checkpoint]) -> List[dict]:
    rows: List[dict] = []
    for c in ckpts:
        for suffix in ("token_embedding", "output_projection"):
            p = find_param(c, suffix)
            if p.adam_m is None or p.adam_v is None:
                continue
            update = p.adam_m / (np.sqrt(p.adam_v) + 1.0e-8)
            balanced = sinkhorn_balance(update, iterations=20)
            before_row = np.sqrt(np.sum(update.astype(np.float64) ** 2, axis=1))
            after_row = np.sqrt(np.sum(balanced.astype(np.float64) ** 2, axis=1))
            before_col = np.sqrt(np.sum(update.astype(np.float64) ** 2, axis=0))
            after_col = np.sqrt(np.sum(balanced.astype(np.float64) ** 2, axis=0))
            rows.append(
                {
                    "seed": c.seed,
                    "step": c.step,
                    "suffix": suffix,
                    "rows": p.rows,
                    "columns": p.columns,
                    "update_matrix_bytes": int(update.nbytes),
                    "working_bytes_estimate_2x": int(2 * update.nbytes),
                    "before_row_norm_mean": float(np.mean(before_row)),
                    "before_row_norm_cv": float(np.std(before_row) / (np.mean(before_row) + 1e-12)),
                    "after_row_norm_mean": float(np.mean(after_row)),
                    "after_row_norm_cv": float(np.std(after_row) / (np.mean(after_row) + 1e-12)),
                    "before_col_norm_mean": float(np.mean(before_col)),
                    "before_col_norm_cv": float(np.std(before_col) / (np.mean(before_col) + 1e-12)),
                    "after_col_norm_mean": float(np.mean(after_col)),
                    "after_col_norm_cv": float(np.std(after_col) / (np.mean(after_col) + 1e-12)),
                    "cosine_to_original_update": cosine(update, balanced),
                    "relative_l2_to_original_update": relative_l2(update, balanced),
                    "note": "diagnostic_only_not_applied_to_training",
                }
            )
    return rows


def resolve_checkpoint_paths(root: Path, seeds: Sequence[int]) -> List[Path]:
    """Locate formal HVX Muon quality checkpoints (seed1 primary, seed2 optional)."""
    candidates: List[Path] = []
    search_roots = [
        root / "build" / "reports" / "hvx-promotion" / "quality-hvx-seed1-step8000",
        root / "build" / "reports" / "hvx-promotion" / "quality-hvx-seed2-step8000",
        root / "build" / "reports" / "hvx-promotion" / "quality-hvx-step1000",
        root / "build" / "reports" / "hvx-promotion" / "quality-hvx-step100",
    ]
    for directory in search_roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.ckpt")):
            # seed encoded in filename
            name = path.name
            seed = None
            if "seed1" in name:
                seed = 1
            elif "seed2" in name:
                seed = 2
            if seed is not None and seed in seeds:
                candidates.append(path)
    # De-duplicate by (seed, step) preferring the first match.
    seen: Dict[Tuple[int, int], Path] = {}
    for path in candidates:
        data = path.read_bytes()[:1]  # touch
        # cheap step parse from filename
        stem = path.name
        if "step" not in stem:
            continue
        step = int(stem.split("step")[-1].split(".")[0])
        seed = 1 if "seed1" in stem else 2
        key = (seed, step)
        if key not in seen:
            seen[key] = path
    return [seen[k] for k in sorted(seen)]


def pick_steps(
    available: Sequence[Tuple[int, int]], mode: str
) -> List[Tuple[int, int]]:
    """Select early/mid/late (or all) checkpoints for diagnostics."""
    if mode == "all":
        return list(available)
    # Prefer seed1 primary.
    seed1 = sorted(s for s in available if s[0] == 1)
    seed2 = sorted(s for s in available if s[0] == 2)
    if mode == "key3":
        def nearest(series, target):
            if not series:
                return None
            return min(series, key=lambda s: abs(s[1] - target))

        picks = []
        for target in (500, 2000, 8000):
            hit = nearest(seed1, target)
            if hit and hit not in picks:
                picks.append(hit)
        return picks
    if mode == "key3+seed2":
        def nearest(series, target):
            if not series:
                return None
            return min(series, key=lambda s: abs(s[1] - target))

        picks = []
        for target in (500, 2000, 8000):
            hit = nearest(seed1, target)
            if hit and hit not in picks:
                picks.append(hit)
            hit2 = nearest(seed2, target)
            if hit2 and hit2 not in picks:
                picks.append(hit2)
        return picks
    raise ValueError(mode)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="transformer_parameter_metadata.json (default: metadata/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: docs/results/v41-optimizer-split-diagnostic-2026-09)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="explicit checkpoint path (repeatable); overrides auto-discovery",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2],
    )
    parser.add_argument(
        "--step-mode",
        choices=["key3", "key3+seed2", "all"],
        default="key3+seed2",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=["wq", "wk", "wv"],
    )
    parser.add_argument(
        "--input-mode",
        choices=["momentum_proxy", "synthetic_grad"],
        default="momentum_proxy",
    )
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    metadata_path = args.metadata or (root / "metadata" / "transformer_parameter_metadata.json")
    output = args.output or (
        root / "docs" / "results" / "v41-optimizer-split-diagnostic-2026-09"
    )
    output.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    # Validate SSOT head-related shapes.
    for suffix in ("wq", "wk", "wv", "token_embedding", "output_projection"):
        if suffix not in metadata:
            raise SystemExit(f"SSOT_MISSING:{suffix}")

    if args.checkpoint:
        paths = [p.resolve() for p in args.checkpoint]
    else:
        discovered = resolve_checkpoint_paths(root, args.seeds)
        if not discovered:
            raise SystemExit("NO_CHECKPOINTS_FOUND")
        # Decode cheaply for step selection using filename.
        available: List[Tuple[int, int, Path]] = []
        for path in discovered:
            seed = 1 if "seed1" in path.name else 2
            step = int(path.name.split("step")[-1].split(".")[0])
            available.append((seed, step, path))
        wanted = set(pick_steps([(s, st) for s, st, _ in available], args.step_mode))
        paths = [p for s, st, p in available if (s, st) in wanted]
        if not paths:
            paths = discovered

    print(f"checkpoints_selected={len(paths)}")
    ckpts = collect_checkpoints(paths, metadata)
    for c in ckpts:
        print(f"  seed={c.seed} step={c.step} magic={c.magic} params={len(c.parameters)}")

    # 1) inventory
    inv = checkpoint_inventory_rows(ckpts)
    write_csv(output / "checkpoint-inventory.csv", inv)

    # 2) checkpoint-only head diagnostic
    raw, summary = head_checkpoint_rows(ckpts, args.suffixes)
    write_csv(output / "headwise-muon-checkpoint.csv", raw)
    write_csv(output / "headwise-muon-checkpoint-summary.csv", summary)
    agg = aggregate_head_summary(summary)
    write_csv(output / "headwise-muon-checkpoint-aggregate.csv", agg)

    # 3) checkpoint displacement
    disp = checkpoint_displacement_rows(ckpts, args.suffixes)
    write_csv(output / "headwise-muon-displacement.csv", disp)

    # 4) 1-step replay full vs split
    replay = replay_rows(ckpts, args.suffixes, input_mode=args.input_mode)
    write_csv(output / "headwise-muon-replay.csv", replay)
    replay_agg = replay_aggregate_rows(replay)
    write_csv(output / "headwise-muon-replay-aggregate.csv", replay_agg)

    # 5) embedding / output projection audit
    emb = embedding_audit_rows(ckpts)
    write_csv(output / "embedding-head-optimizer.csv", emb)

    # 6) Sinkhorn feasibility (offline diagnostic only)
    sink = sinkhorn_rows(ckpts)
    write_csv(output / "sinkhorn-feasibility.csv", sink)

    # Machine-readable verdict skeleton (numbers only; human judgment in md).
    verdict = {
        "checkpoints": len(ckpts),
        "seeds": sorted({c.seed for c in ckpts}),
        "steps": sorted({c.step for c in ckpts}),
        "head_suffixes": args.suffixes,
        "input_mode": args.input_mode,
        "output_dir": str(output).replace("\\", "/"),
        "files": sorted(p.name for p in output.glob("*.csv")),
    }
    (output / "diagnostic-run.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(f"output_dir={output}")
    print(f"files={verdict['files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
