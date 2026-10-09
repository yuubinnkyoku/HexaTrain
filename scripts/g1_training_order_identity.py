# SPDX-License-Identifier: Apache-2.0
"""Compute the canonical G1 training-order hash without reading corpus bytes."""

from __future__ import annotations

import argparse
import json


MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
ORDER_SEED = 20260806
BATCH_SIZE = 8
LEGACY_STEPS = 100_000
LEGACY_HASH = "fnv1a64:be3269f541236dc2"
LONG_LINEAR_STEPS = 200_000
LONG_LINEAR_HASH = "fnv1a64:83f9c96cf10261a0"


def split_mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def order_hash(record_count: int, steps: int) -> tuple[str, str | None]:
    if record_count <= 0 or steps <= 0:
        raise ValueError("record_count and steps must be positive")
    state = ORDER_SEED
    value = FNV_OFFSET
    prefix_hash: str | None = None
    selection_count = steps * BATCH_SIZE
    legacy_selection_count = LEGACY_STEPS * BATCH_SIZE
    for index in range(selection_count):
        state = split_mix((state + index) & MASK64)
        selected = state % record_count
        for byte in selected.to_bytes(8, "little", signed=False):
            value ^= byte
            value = (value * FNV_PRIME) & MASK64
        if index + 1 == legacy_selection_count:
            prefix_hash = f"fnv1a64:{value:016x}"
    return f"fnv1a64:{value:016x}", prefix_hash


def self_test() -> None:
    if split_mix(0) != 0xE220A8397B1DCDAF:
        raise AssertionError("SplitMix64 reference vector mismatch")
    legacy, _ = order_hash(12_741_465, LEGACY_STEPS)
    if legacy != LEGACY_HASH:
        raise AssertionError("100k canonical order hash mismatch")
    extended, prefix = order_hash(12_741_465, LONG_LINEAR_STEPS)
    if prefix != LEGACY_HASH or extended != LONG_LINEAR_HASH:
        raise AssertionError("200k extended order identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-count", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("g1_training_order_identity_self_test=PASS")
        return 0
    if args.record_count is None or args.steps is None:
        parser.error("--record-count and --steps are required unless --self-test is set")
    digest, legacy_prefix = order_hash(args.record_count, args.steps)
    print(
        json.dumps(
            {
                "steps": args.steps,
                "batch_size": BATCH_SIZE,
                "order_seed": ORDER_SEED,
                "record_count": args.record_count,
                "selection_count": args.steps * BATCH_SIZE,
                "hash": digest,
                "prefix_100k_hash": legacy_prefix,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
