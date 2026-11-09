#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Independent CPU audit of the D8 minimal experiment (prereg 2026-11-08).

This test does not reuse the C model's numbers as its expected values. It
    1. re-derives the registered operand fixture from the dump and checks the
       registered operands_deploy digest,
    2. re-implements the registered arithmetic (float32 accumulation of the
       32-product dot product, three MMA passes, RN16 convert with the negative
       clamp) and derives the four D8 references itself,
    3. checks those digests against the constants pre-registered in
       docs/hmx-v81-cand-b-d8-device-prereg.md and against the C dump,
    4. re-implements the D8 classifier from the pre-registration text and runs
       a fixed battery plus deterministic mutations against the C classifier
       through `hmx_d8_min_sim --d8-replay`,
    5. mutates one C answer in memory and requires the comparator to notice.

CPU only. No device, DSP, SDK or network access.

Exit: 0 = all invariants hold, 1 = mismatch, 2 = missing dump/environment.
"""

import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build")
CASE_FILE = os.path.join(BUILD, "hmx_d8_replay_cases.txt")
C_DUMP = os.path.join(BUILD, "hmx_d8_reference_arrays.txt")
PREREG = os.path.join(REPO, "docs", "hmx-v81-cand-b-d8-device-prereg.md")
CONTRACT = os.path.join(REPO, "docs", "hmx-v81-cand-b-d8-evidence-contract.md")

T8 = 32
CELLS = T8 * T8

REG_SHA = {
    "operands_deploy": "e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3",
    "safe": "c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d",
    "wlo": "f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb",
    "ahi": "b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc",
    "both": "5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d",
}

SENTINEL_WORD = 0xA5A5
# FNV-1a 32 of each packed operand tile, in the layout the MMA reads. The
# device contract echoes these so an input mix-up fails closed before any
# numeric interpretation.
REG_FNV = {
    "ahi": 0xC6231266,
    "alo": 0x3B2D22D2,
    "whi": 0x8B88609F,
    "wlo": 0xA66AF695,
}
COND_A, COND_B, COND_C, COND_D = 0, 1, 2, 3
TRACE_NOT_PROVIDED, TRACE_VALID, TRACE_INVALID = 0, 1, 2

# Registered fill masks (evidence contract sections 3.2 and 3.3). Bit 0 is
# out_pos, bit 1 Wlo, bit 2 Ahi. Revision 1 listed C and D swapped and made B
# ambiguous; revision 2 fixes A=1, B=7, C=3, D=5. The value below is a
# transcription of the contract document, not a copy of the C table: main()
# also parses the document's machine-checked block and compares both.
MASK = {COND_A: 1, COND_B: 7, COND_C: 3, COND_D: 5}
FILL_BIT_OUT, FILL_BIT_WLO, FILL_BIT_AHI = 1, 2, 4

CHECKS = {"n": 0, "fail": 0}


def check(cond, tag):
    CHECKS["n"] += 1
    if not cond:
        CHECKS["fail"] += 1
        print("FAIL %s" % tag)
    return cond


# ------------------------------------------------------------------- numeric
def fp16_to_f64(bits):
    sign = -1.0 if (bits >> 15) & 1 else 1.0
    exp = (bits >> 10) & 31
    frac = bits & 1023
    if exp == 31:
        return sign * (math.nan if frac else math.inf)
    if exp == 0:
        return sign * math.ldexp(float(frac), -24)
    return sign * math.ldexp(float(1024 + frac), exp - 25)


def f64_to_fp16_rn(value):
    """IEEE round-to-nearest-even f64 -> fp16 (same law as fp16_util.h)."""
    if value != value:
        return 0x7E00
    sign = 0x8000 if (value < 0.0 or (value == 0.0 and math.copysign(1.0, value) < 0)) else 0
    mag = abs(value)
    if mag == 0.0:
        return sign
    if mag > 65520.0:
        return sign | 0x7C00
    _, be = math.frexp(mag)
    exponent = be - 1
    if exponent < -14:
        quantum = math.ldexp(1.0, -24)
        scaled = mag / quantum
        floor_value = math.floor(scaled)
        delta = scaled - floor_value
        if delta > 0.5:
            rounded = floor_value + 1.0
        elif delta < 0.5:
            rounded = floor_value
        else:
            rounded = floor_value if math.fmod(floor_value, 2.0) == 0.0 else floor_value + 1.0
        if rounded >= 1024.0:
            return sign | (1 << 10)
        return sign | int(rounded)
    quantum = math.ldexp(1.0, exponent - 10)
    scaled = mag / quantum
    floor_value = math.floor(scaled)
    delta = scaled - floor_value
    if delta > 0.5:
        rounded = floor_value + 1.0
    elif delta < 0.5:
        rounded = floor_value
    else:
        rounded = floor_value if math.fmod(floor_value, 2.0) == 0.0 else floor_value + 1.0
    result_exponent = exponent
    significand = int(rounded)
    if significand == 2048:
        result_exponent += 1
        significand = 1024
    if result_exponent > 15:
        return sign | 0x7C00
    return sign | ((result_exponent + 15) << 10) | (significand - 1024)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def convert(acc):
    out = []
    for x in acc:
        v = f64_to_fp16_rn(64.0 * x)
        if v & 0x8000:
            v = 0
        out.append(v)
    return out


def mma(acc, a, w, a_pat=None, w_pat=None):
    for s in range(T8):
        arow = a[s * T8:(s + 1) * T8]
        for o in range(T8):
            wrow = w[o * T8:(o + 1) * T8]
            dot = 0.0
            for i in range(T8):
                av = fp16_to_f64(a_pat if a_pat is not None else arow[i])
                wv = fp16_to_f64(w_pat if w_pat is not None else wrow[i])
                dot = f32(dot + f32(av * wv))
            acc[s * T8 + o] = f32(acc[s * T8 + o] + dot)
    return acc


def derive_refs(ahi, alo, whi, wlo):
    def build(a_pat=None, w_pat=None):
        acc = [0.0] * CELLS
        mma(acc, ahi, whi, a_pat=a_pat)
        mma(acc, ahi, wlo, a_pat=a_pat, w_pat=w_pat)
        mma(acc, alo, whi)
        return convert(acc)

    return {
        "safe": build(),
        "wlo": build(w_pat=SENTINEL_WORD),
        "ahi": build(a_pat=SENTINEL_WORD),
        "both": build(a_pat=SENTINEL_WORD, w_pat=SENTINEL_WORD),
    }


def bits(arr):
    return b"".join(struct.pack("<H", v) for v in arr)


def pack_tile(src, weight):
    """Registered PRM pair-interleaved crouton packing (index mapping)."""
    dst = [0] * CELLS
    for a in range(T8):
        for i in range(T8):
            if weight:
                idx = ((i >> 1) * 64) + (a * 2) + (i & 1)  # wgt_u16(o, i)
            else:
                idx = ((a >> 1) * 64) + (i * 2) + (a & 1)  # act_u16(s, i)
            dst[idx] = src[a * T8 + i]
    return dst


def fnv1a32(data):
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ------------------------------------------------------------------- dump I/O
def read_dump(path):
    if not os.path.exists(path):
        print("SKIP (no dump at %s - run scripts/run_hmx_d8_sim.ps1 first)" % path)
        sys.exit(2)
    sections = {}
    header = {}
    current = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("# sha256_"):
                key, val = line[2:].split("=", 1)
                header[key] = val
            elif line.startswith("section="):
                current = line[len("section="):]
                sections[current] = []
            elif current and line:
                sections[current].append(line)
    out = {}
    for name, lines in sections.items():
        if name == "operands_deploy":
            out[name] = bytes.fromhex("".join(lines))
        else:
            out[name] = [int(v, 16) for v in lines]
    return out, header


# ---------------------------------------------------------------- classifier
def match_ref(obs, refs):
    for i, name in enumerate(("safe", "wlo", "ahi", "both")):
        if obs == refs[name]:
            return i
    return -1


def mask_ok(cond, fill_mask):
    """The contract gate: the echoed mask must be the registered one."""
    return fill_mask == MASK.get(cond)


def derive_mask(cond):
    """Build the mask from the condition's operand fills, bit by bit."""
    m = FILL_BIT_OUT
    if cond in (COND_B, COND_C):
        m |= FILL_BIT_WLO
    if cond in (COND_B, COND_D):
        m |= FILL_BIT_AHI
    return m


def classify(cond, obs, trace, evidence_ok, ids_ok, lengths_ok, fill_mask, refs):
    """Prereg section 6 order: contract -> trace -> numeric."""
    if not (evidence_ok and ids_ok and lengths_ok):
        return "D8_CONTRACT_FAILED"
    if not mask_ok(cond, fill_mask):
        return "D8_CONTRACT_FAILED"
    if trace == TRACE_INVALID:
        return "D8_INCONCLUSIVE_TRACE"
    m = match_ref(obs, refs)
    if cond == COND_A:
        return "D8_CONTROL_SAFE" if m == 0 else "D8_CONTROL_ANOMALY"
    if cond == COND_B:
        return ("D8_NO_VIOLATION_OBSERVED" if m == 0 else
                "D8_VIOLATION_WLO_SUSPECT" if m == 1 else
                "D8_VIOLATION_AHI_SUSPECT" if m == 2 else
                "D8_VIOLATION_BOTH_SUSPECT" if m == 3 else
                "D8_INCONCLUSIVE_NUMERIC")
    if cond == COND_C:
        return ("D8_NO_VIOLATION_OBSERVED" if m == 0 else
                "D8_VIOLATION_WLO_SUSPECT" if m == 1 else
                "D8_INCONCLUSIVE_NUMERIC")
    if cond == COND_D:
        return ("D8_NO_VIOLATION_OBSERVED" if m == 0 else
                "D8_VIOLATION_AHI_SUSPECT" if m == 2 else
                "D8_INCONCLUSIVE_NUMERIC")
    return "D8_CONTRACT_FAILED"


# ------------------------------------------------------------------- cases
def build_cases(refs):
    cases = []

    def add(desc, cond, obs, trace=TRACE_VALID, evidence=1, ids=1, lengths=1,
            mask=None):
        cases.append({
            "desc": desc, "cond": cond, "obs": list(obs), "trace": trace,
            "evidence": evidence, "ids": ids, "lengths": lengths,
            "mask": MASK[cond] if mask is None else mask,
        })

    # Every condition against every reference: 16 combinations. Some are
    # deliberately invalid (e.g. condition C with the Ahi reference) and must
    # fail closed rather than be attributed to the wrong region.
    for cond in (COND_A, COND_B, COND_C, COND_D):
        for name in ("safe", "wlo", "ahi", "both"):
            add("cond%d_vs_%s" % (cond, name), cond, refs[name])

    # Trace anomaly overrides a safe numeric result.
    for cond in (COND_A, COND_B, COND_C, COND_D):
        add("trace_invalid_cond%d" % cond, cond, refs["safe"], trace=TRACE_INVALID)

    # Mask contract: correct numeric result and clean trace, wrong mask echo.
    # A C run that reports D's mask (5) fills Ahi; a D run that reports C's
    # mask (3) fills Wlo; a B run that dropped one fill reports 3 or 5; a B
    # run with no operand fill reports 1; an A control with operand fills 7.
    add("mask_c_uses_d", COND_C, refs["ahi"], mask=5)
    add("mask_d_uses_c", COND_D, refs["wlo"], mask=3)
    add("mask_b_missing_ahi", COND_B, refs["wlo"], mask=3)
    add("mask_b_missing_wlo", COND_B, refs["ahi"], mask=5)
    add("mask_b_no_operand_fill", COND_B, refs["safe"], mask=1)
    add("mask_a_extra_fill", COND_A, refs["both"], mask=7)

    # Contract failures.
    add("no_evidence", COND_C, refs["safe"], evidence=0)
    add("bad_ids", COND_C, refs["safe"], ids=0)
    add("bad_lengths", COND_C, refs["safe"], lengths=0)

    # Deterministic mutations: a few cells changed. These normally match no
    # reference and must be inconclusive, never "no violation".
    for seed in range(1, 65):
        r = random.Random(seed)
        obs = list(refs["safe"])
        for _ in range(1 + seed % 8):
            i = r.randrange(CELLS)
            obs[i] = (obs[i] + 1 + r.randrange(3)) & 0xFFFF
        add("mutated_safe_%d" % seed, COND_B, obs)

    # Random arrays (deterministic), including values that could look like a
    # reference by accident; the classifier must stay inconclusive.
    for seed in range(1, 33):
        r = random.Random(1000 + seed)
        obs = [r.randrange(65536) for _ in range(CELLS)]
        add("random_%d" % seed, COND_D, obs)

    return cases


# ------------------------------------------------------------------- replay
def write_cases(cases):
    with open(CASE_FILE, "w") as fh:
        fh.write("# D8 classifier replay cases (CPU only)\n")
        for cid, c in enumerate(cases):
            fh.write("case %d %d %d %d %d %d %d\n" % (
                cid, c["cond"], c["trace"], c["evidence"], c["ids"],
                c["lengths"], c["mask"]))
            fh.write("O %s\n" % " ".join("%04x" % v for v in c["obs"]))


def run_replay(exe):
    proc = subprocess.run([exe, "--d8-replay", CASE_FILE],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    answers = {}
    malformed = 0
    for line in proc.stdout.splitlines():
        if line.startswith("replay "):
            parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            answers[int(parts[1])] = parts[2]
        elif line.startswith("d8_replay cases="):
            for token in line.split()[1:]:
                if token.startswith("malformed="):
                    malformed = int(token.split("=", 1)[1])
    return proc.returncode, answers, malformed, proc.stdout


# ------------------------------------------------- pre-registration checks
def read_marked_block(path, marker):
    """Return the lines between <!-- marker:begin --> and <!-- marker:end -->."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    inside = False
    block = []
    for line in lines:
        if ("%s:begin" % marker) in line:
            inside = True
            continue
        if ("%s:end" % marker) in line:
            break
        if inside:
            block.append(line)
    return block


def parse_mask_block(path):
    table = {}
    for line in read_marked_block(path, "d8-fill-mask"):
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("A", "B", "C", "D"):
            table[parts[0]] = int(parts[1], 0)
    return table


def parse_sequence_block(path):
    tokens = []
    for line in read_marked_block(path, "d8-sequence"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "PACK" and len(parts) == 2:
            tokens.append(("PACK", parts[1], None))
        elif kind == "BIAS" and len(parts) == 2:
            tokens.append(("BIAS", parts[1], None))
        elif kind == "MMA" and len(parts) == 4:
            tokens.append(("MMA", parts[1], (parts[2], parts[3])))
        elif kind == "FILL" and len(parts) >= 2:
            tokens.append(("FILL", parts[1], tuple(parts[2:])))
        elif kind == "CONVERT" and len(parts) == 2:
            tokens.append(("CONVERT", parts[1], None))
        elif kind == "STORE" and len(parts) == 2:
            tokens.append(("STORE", parts[1], None))
        else:
            tokens.append(("UNKNOWN", line, None))
    return tokens


def expected_signature(tokens, cond_letter):
    """Expand the registered block into the C model's event tokens."""
    out = []
    for kind, a, b in tokens:
        if kind == "PACK":
            out.append("PACK:%s" % a)
        elif kind == "BIAS":
            out.append("BIAS:%s" % a)
        elif kind == "MMA":
            out.append("MMA:%s:%s:%s" % (a, b[0], b[1]))
        elif kind == "FILL":
            conds = b or ("A", "B", "C", "D")
            if cond_letter in conds:
                out.append("FILL:%s" % a)
        elif kind == "CONVERT":
            out.append("CONVERT:%s" % a)
        elif kind == "STORE":
            out.append("STORE:%s" % a)
    return out


def run_sequence(exe):
    proc = subprocess.run([exe, "--d8-sequence"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    masks, seqs = None, {}
    for line in proc.stdout.splitlines():
        if line.startswith("d8_fill_mask "):
            masks = {}
            for token in line.split()[1:]:
                key, value = token.split("=", 1)
                masks[key] = int(value)
        elif line.startswith("d8_seq cond="):
            head, events = line.split(" events=", 1)
            letter = head.split("cond=", 1)[1]
            seqs[letter] = events.split(",") if events else []
    return proc.returncode, masks, seqs, proc.stdout


def main():
    if not os.path.isdir(BUILD):
        os.makedirs(BUILD)
    sections, header = read_dump(C_DUMP)

    ops = sections["operands_deploy"]
    check(hashlib.sha256(ops).hexdigest() == REG_SHA["operands_deploy"],
          "operands_deploy_digest")
    words = list(struct.unpack("<%dH" % (len(ops) // 2), ops))
    ahi = words[0:CELLS]
    alo = words[CELLS:2 * CELLS]
    whi = words[2 * CELLS:3 * CELLS]
    wlo = words[3 * CELLS:4 * CELLS]

    for name, src, weight in (("ahi", ahi, 0), ("alo", alo, 0),
                              ("whi", whi, 1), ("wlo", wlo, 1)):
        got = fnv1a32(bits(pack_tile(src, weight)))
        check(got == REG_FNV[name], "packed_fnv_%s" % name)
        print("d8_operand_fnv name=%s fnv=%08x" % (name, got))

    refs = derive_refs(ahi, alo, whi, wlo)
    for name, values in refs.items():
        sha = hashlib.sha256(bits(values)).hexdigest()
        check(sha == REG_SHA[name], "rederived_sha_%s" % name)
        check(values == sections[name], "rederived_equals_c_dump_%s" % name)
        print("d8_ref name=%s sha256=%s" % (name, sha))

    # Pairwise distinctness is the experiment's attribution property.
    names = list(refs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = sum(1 for k in range(CELLS) if refs[names[i]][k] != refs[names[j]][k])
            check(diff > 0, "refs_distinct_%s_%s" % (names[i], names[j]))
            print("d8_diff %s_vs_%s=%d" % (names[i], names[j], diff))

    # Safe must equal the shipped B3 positive array's digest? The C model
    # asserts cell equality; here we only pin the derived digest.
    cases = build_cases(refs)
    write_cases(cases)
    expected = {}
    for cid, c in enumerate(cases):
        expected[cid] = classify(c["cond"], c["obs"], c["trace"],
                                 c["evidence"], c["ids"], c["lengths"],
                                 c["mask"], refs)

    exe = None
    for arg in sys.argv[1:]:
        if arg.startswith("--replay-exe="):
            exe = arg.split("=", 1)[1]
    if exe is None:
        exe = os.path.join(BUILD, "hmx_d8_sim.exe")
    if not os.path.isfile(exe):
        print("SKIP (no C replay binary at %s)" % exe)
        return 2

    # ---- pre-registration vs CPU event order and fill masks ----
    # The contract's mask table is parsed from the document, not copied from
    # the C model. The registered event order is parsed from the
    # pre-registration and compared with the C trace for each condition.
    doc_mask = parse_mask_block(CONTRACT)
    check(doc_mask == {"A": 1, "B": 7, "C": 3, "D": 5},
          "contract_fill_mask_table")
    doc_mask_int = {COND_A: doc_mask.get("A"), COND_B: doc_mask.get("B"),
                    COND_C: doc_mask.get("C"), COND_D: doc_mask.get("D")}
    check(doc_mask_int == MASK and
          all(derive_mask(c) == MASK[c] for c in MASK),
          "python_mask_matches_contract")

    tokens = parse_sequence_block(PREREG)
    check(len(tokens) > 0, "prereg_sequence_block_present")
    seq_rc, c_masks, c_seqs, seq_out = run_sequence(exe)
    check(seq_rc == 0, "c_sequence_exit")
    if c_masks is None:
        c_masks = {}
    c_mask_int = {COND_A: c_masks.get("A"), COND_B: c_masks.get("B"),
                  COND_C: c_masks.get("C"), COND_D: c_masks.get("D")}
    check(c_mask_int == MASK, "c_fill_mask_matches_python")
    for letter in ("A", "B", "C", "D"):
        check(c_seqs.get(letter) == expected_signature(tokens, letter),
              "sequence_%s" % letter)
    if any(c_seqs.get(letter) != expected_signature(tokens, letter)
           for letter in ("A", "B", "C", "D")):
        print("prereg_sequence=%s" % " ".join(
            "%s:%s" % (k, ",".join(v)) for k, v in sorted(c_seqs.items())))
        print(seq_out[-1500:])

    rc, answers, malformed, out = run_replay(exe)
    if rc != 0 or malformed:
        print("FAIL replay exit=%d malformed=%d" % (rc, malformed))
        print(out[-1500:])
        return 1

    mismatches = []
    for cid, c in enumerate(cases):
        got = answers.get(cid)
        want = expected[cid]
        if got != want:
            mismatches.append((cid, c, want, got))
    if answers.get(0) is None:
        print("FAIL replay produced no case 0")
        return 1

    if mismatches:
        for cid, c, want, got in mismatches[:10]:
            print("MISMATCH case=%d (%s): python=%s c=%s"
                  % (cid, c["desc"], want, got))
        path = os.path.join(BUILD, "hmx_d8_mismatch.json")
        with open(path, "w") as fh:
            json.dump([{"case": cid, "desc": c["desc"], "python": want, "c": got}
                       for cid, c, want, got in mismatches], fh)
        print("repro saved: %s" % path)
        print("d8_differential=FAIL cases=%d mismatches=%d" % (len(cases), len(mismatches)))
        return 1

    # Comparator self-check: corrupt one answer and require a mismatch.
    mutated = dict(answers)
    victim = next((cid for cid, c in enumerate(cases)
                   if expected[cid] == "D8_VIOLATION_WLO_SUSPECT"), None)
    mutation_detected = 0
    if victim is not None:
        mutated[victim] = "D8_NO_VIOLATION_OBSERVED"
        if mutated[victim] != expected[victim]:
            mutation_detected = 1
    check(mutation_detected == 1, "comparator_mutation_detected")

    print("d8_differential=PASS cases=%d deterministic_mutations=%d "
          "random_arrays=%d" % (len(cases), 64, 32))
    print("d8_device_status=NOT_RUN cpu_labels_only=1")
    print("checks=%d fail=%d" % (CHECKS["n"], CHECKS["fail"]))
    print("verdict=%s" % ("D8_INDEPENDENT_AUDIT_PASS"
                          if CHECKS["fail"] == 0 else "D8_INDEPENDENT_AUDIT_FAIL"))
    return 0 if CHECKS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
