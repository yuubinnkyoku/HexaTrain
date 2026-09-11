"""Host-only comparisons for raw, little-endian FP32 diagnostic GEMMs.

No training data, optimizer mutation, or device execution is performed here.
"""

import argparse
import json
import math
from pathlib import Path
import struct


def floats(path):
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ValueError("FP32 byte length is not a multiple of four")
    return struct.unpack(f"<{len(data) // 4}f", data)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def gemm_reference(a, b, m, k, n, float_accumulation=False):
    if min(m, k, n) <= 0 or len(a) != m * k or len(b) != k * n:
        raise ValueError("GEMM shape mismatch")
    if not all(math.isfinite(v) for v in (*a, *b)):
        raise ValueError("Nonfinite GEMM input")
    result = []
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                product = a[i * k + p] * b[p * n + j]
                total = f32(total + f32(product)) if float_accumulation else total + product
            result.append(f32(total))
    return result


def compare(actual, reference):
    if not actual or len(actual) != len(reference):
        raise ValueError("Comparison size mismatch")
    if not all(math.isfinite(v) for v in (*actual, *reference)):
        return {"finite": False, "gate": False}
    delta = [a - r for a, r in zip(actual, reference)]
    error2 = math.fsum(d * d for d in delta)
    actual2 = math.fsum(a * a for a in actual)
    reference2 = math.fsum(r * r for r in reference)
    # Zero/zero is exact; a nonzero update vs zero reference fails explicitly.
    relative = math.sqrt(error2 / reference2) if reference2 else (0 if not error2 else None)
    cosine = math.fsum(a * r for a, r in zip(actual, reference)) / math.sqrt(actual2 * reference2) if actual2 and reference2 else (1 if not actual2 and not reference2 else 0)
    result = {
        "maxAbs": max(map(abs, delta)),
        "meanAbs": math.fsum(map(abs, delta)) / len(delta),
        "RMS": math.sqrt(error2 / len(delta)),
        "relativeL2": relative,
        "cosine": cosine,
        "finite": True,
    }
    result["gate"] = relative is not None and result["maxAbs"] <= .002 and relative <= .001 and cosine >= .99999
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--shape", nargs=3, type=int, required=True)
    parser.add_argument("--simulator", type=Path)
    args = parser.parse_args()
    a, b, actual = floats(args.left), floats(args.right), floats(args.actual)
    results = {
        name: compare(actual, gemm_reference(a, b, *args.shape, use_float))
        for name, use_float in (("CPU_DOUBLE", False), ("CPU_FLOAT", True))
    }
    if args.simulator:
        results["SIMULATOR"] = compare(actual, floats(args.simulator))
    args.report.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if all(value["gate"] for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
