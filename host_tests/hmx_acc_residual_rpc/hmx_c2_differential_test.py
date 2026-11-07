#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Machine differential test: the C2 classifier in C vs the Python re-implementation.

Why this exists
---------------
`hmx_c2_independent_classifier_test.py` re-implements the pre-registered
decision procedure and pins every branch to the class section 9 names, but it
never feeds the *same observations* to the C classifier and compares the two
answers. The C model's own controls are a second, separately written battery.
Two batteries can agree with the text and still disagree with each other on an
input neither lists. This file closes that gap mechanically:

    1. it derives the registered 1,024-cell Plain payloads from the digest-pinned
       operands (the same independent arithmetic the classifier audit uses),
    2. it builds a case battery -- the fixed cases, the classification
       boundaries, multi-fault combinations, and deterministic-seed mixtures --
       and computes the Python answer for every case BEFORE any C code runs,
    3. it writes all cases to a replay file and runs the shipped
       `c2_observe()`/`c2_classify()` (`--c2-replay` mode of the simulator) over
       exactly those bytes under several compiler/optimisation configurations,
    4. it compares the two class streams mechanically and saves a repro on any
       mismatch,
    5. it proves the comparator itself is not vacuous by corrupting one C answer
       and requiring the comparison to notice.

CPU only. Nothing here builds or touches a DSP binary, QAIRT, ADB or a device.
The Python side never reads a C classification result to decide what the answer
should be: expectations are computed first and stored, then C is executed.

Exit: 0 = all cases agree and the mutation check fired, 1 = mismatch / mutation
not detected, 2 = missing dump or build environment.
"""

import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build")
sys.path.insert(0, HERE)

import hmx_c2_independent_classifier_test as indep  # noqa: E402

T8 = 32
CELLS = T8 * T8
SIM_SOURCE = os.path.join(HERE, "hmx_b3_r2_full_tile_sim.c")
CASE_FILE = os.path.join(BUILD, "c2_differential_cases.txt")
REPLAY_ARGS = ["--c2-replay", CASE_FILE]

MUTATION_CLASS = "C2_ORDERING_VIOLATED"
MUTATION_REPLACEMENT = "C2_CAUSE_INCONCLUSIVE"


def log(line):
    print(line)
    sys.stdout.flush()


# ------------------------------------------------------------------ fixture
def load_fixture():
    sections, _header = indep.read_dump(indep.DUMP)
    ops = sections["operands_deploy"]
    words = list(struct.unpack("<%dH" % (len(ops) // 2), ops))
    a_plain = words[4 * CELLS:5 * CELLS]
    w_plain = words[5 * CELLS:6 * CELLS]
    pos, neg = indep.derive_plain_pos_neg(a_plain, w_plain)
    sha_pos = hashlib.sha256(indep.pack_u16_le(pos)).hexdigest()
    sha_neg = hashlib.sha256(indep.pack_u16_le(neg)).hexdigest()
    if sha_pos != indep.REG_SHA["cand_c_plain_pos"] or \
            sha_neg != indep.REG_SHA["cand_c_plain_neg"]:
        raise SystemExit("derived payloads do not match the registered digests")
    return pos, neg


def is_subnormal(bits):
    return ((bits >> 10) & 31) == 0 and (bits & 1023) != 0


# ------------------------------------------------------------------- cases
class Case(object):
    __slots__ = ("cid", "kind", "desc", "contract", "p", "x", "counts")

    def __init__(self, cid, kind, desc, contract, p=None, x=None, counts=None):
        self.cid = cid
        self.kind = kind
        self.desc = desc
        self.contract = contract
        self.p = p
        self.x = x
        self.counts = counts


def scaled_store_index(s, o):
    """The registered store crouton bijection (independent copy)."""
    return ((s >> 1) * 64) + (o * 2) + (s & 1)


def scramble(arr):
    raw = [0] * CELLS
    for s in range(T8):
        for o in range(T8):
            raw[scaled_store_index(s, o)] = arr[s * T8 + o]
    return raw


def first_nonzero_negative(exp_pos, exp_neg, start=0):
    for i in range(start, CELLS):
        if exp_pos[i] != exp_neg[i] and exp_neg[i] != 0:
            return i
    return -1


def subnormal_cells(exp_pos):
    return [i for i in range(CELLS) if exp_pos[i] != 0 and is_subnormal(exp_pos[i])]


def weak_cells(exp_pos, exp_neg):
    return [i for i in range(CELLS)
            if exp_neg[i] == 0 and exp_pos[i] != 0 and not is_subnormal(exp_pos[i])]


def unknown_value(exp_pos, exp_neg, i, salt):
    v = (0x0001 + salt) & 0xFFFF
    while v in (0, 0xA5A5, 0xDEAD, 0x5A5A, exp_pos[i], exp_neg[i]):
        v = (v + 1) & 0xFFFF
    return v


def build_array_cases(exp_pos, exp_neg):
    """Fixed battery + boundaries + multi-fault combinations."""
    cases = []
    sub = subnormal_cells(exp_pos)
    weak = weak_cells(exp_pos, exp_neg)
    strong = first_nonzero_negative(exp_pos, exp_neg)
    assert sub and weak and strong >= 0

    def add(desc, p, x, contract=1):
        cases.append(Case(len(cases), "arrays", desc, contract, list(p), list(x)))

    nominal_p = list(exp_pos)
    nominal_x = list(exp_neg)

    add("nominal", nominal_p, nominal_x)

    # Whole-tile post-Store-#2 read: also carries the two ambiguous subnormal
    # cells, so it pins "signature before special values" at run level.
    add("full_store2", list(exp_neg), nominal_x)

    add("all_dead", [0xDEAD] * CELLS, nominal_x)

    p_partial = list(exp_pos)
    for i in range(512, CELLS):
        p_partial[i] = 0xDEAD
    add("partial_copy", p_partial, nominal_x)

    p_dead_one = list(exp_pos)
    p_dead_one[CELLS - 1] = 0xDEAD
    add("single_dead_cell", p_dead_one, nominal_x)

    p_c1 = list(exp_pos)
    p_c1[0] = 0xA5A5
    add("single_c1", p_c1, nominal_x)

    p_c1_all = [0xA5A5] * CELLS
    add("all_c1", p_c1_all, nominal_x)

    x_broken = list(nominal_x)
    x_broken[0] ^= 0x0001
    add("x_broken", nominal_p, x_broken)
    add("x_broken_plus_dead", p_partial, x_broken)
    add("x_broken_plus_signature", list(exp_neg), x_broken)

    p_c1_sub = list(exp_pos)
    p_c1_sub[0] = 0xA5A5
    for i in sub:
        p_c1_sub[i] = 0
    add("c1_plus_subnormal", p_c1_sub, nominal_x)

    p_c1_unknown = list(exp_pos)
    p_c1_unknown[0] = 0xA5A5
    p_c1_unknown[1] = unknown_value(exp_pos, exp_neg, 1, 0)
    add("c1_plus_unknown", p_c1_unknown, nominal_x)

    p_c1_zero = list(exp_pos)
    p_c1_zero[0] = 0xA5A5
    for i in weak:
        p_c1_zero[i] = 0
    add("c1_plus_zero_agreement", p_c1_zero, nominal_x)

    p_zero = list(exp_pos)
    for i in weak:
        p_zero[i] = 0
    add("zero_agreement_only", p_zero, nominal_x)

    p_ftz = list(exp_pos)
    for i in sub:
        p_ftz[i] = 0
    add("ftz_two_cells", p_ftz, nominal_x)

    p_ftz1 = list(exp_pos)
    p_ftz1[sub[0]] = 0
    add("ftz_one_cell", p_ftz1, nominal_x)

    # FTZ zeros AND a specific non-zero negative pattern: the signature wins.
    p_ftz_sig = list(p_ftz)
    p_ftz_sig[strong] = exp_neg[strong]
    add("ftz_plus_signature", p_ftz_sig, nominal_x)

    # One strong cell plus the weak zeros a genuine post-Store-#2 read also
    # produces: attribution still holds.
    p_sig_zero = list(exp_pos)
    p_sig_zero[strong] = exp_neg[strong]
    for i in weak:
        p_sig_zero[i] = 0
    add("signature_plus_zero_agreement", p_sig_zero, nominal_x)

    p_sig_only = list(exp_pos)
    p_sig_only[strong] = exp_neg[strong]
    add("single_nonzero_c2_cell", p_sig_only, nominal_x)

    p_unknown = list(exp_pos)
    p_unknown[0] = unknown_value(exp_pos, exp_neg, 0, 0)
    add("unknown_bit", p_unknown, nominal_x)

    p_unknown_all = [unknown_value(exp_pos, exp_neg, i, i) for i in range(CELLS)]
    add("all_unknown", p_unknown_all, nominal_x)

    add("p_x_swapped", list(nominal_x), list(nominal_p))
    add("decode_swap_both", scramble(nominal_x), scramble(nominal_p))
    add("decode_swap_p_only", scramble(nominal_p), nominal_x)

    add("contract_failed", nominal_p, nominal_x, contract=0)
    add("contract_failed_plus_dead", p_partial, x_broken, contract=0)

    return cases


def build_random_cases(exp_pos, exp_neg, seeds=256):
    """Deterministic-seed mixtures reaching every classifier branch."""
    profiles = {
        0: (0.80, 0.05, 0.00, 0.00, 0.10, 0.05),  # payload/zero/unknown only
        1: (0.85, 0.05, 0.02, 0.02, 0.04, 0.02),  # sentinel-bearing
        2: (0.97, 0.00, 0.00, 0.03, 0.00, 0.00),  # C1-only
        3: (0.10, 0.70, 0.00, 0.02, 0.15, 0.03),  # C2-heavy / mixed
        4: (0.10, 0.00, 0.00, 0.70, 0.17, 0.03),  # C1 + zero/unknown
    }
    cases = []
    for seed in range(1, seeds + 1):
        r = random.Random(seed)
        keep, negw, deadw, c1w, zerow, unk = profiles[seed % 5]
        p = list(exp_pos)
        x = list(exp_neg)
        for i in range(CELLS):
            k = r.random()
            if k < keep:
                pass
            elif k < keep + negw:
                p[i] = exp_neg[i]
            elif k < keep + negw + deadw:
                p[i] = 0xDEAD
            elif k < keep + negw + deadw + c1w:
                p[i] = 0xA5A5
            elif k < keep + negw + deadw + c1w + zerow:
                p[i] = 0
            else:
                p[i] = 0x3100 | r.randrange(1024)
        if r.random() < 0.30:
            x[r.randrange(CELLS)] ^= 0x0001
        cases.append(Case(len(cases), "arrays", "seed%d" % seed, 1, p, x))
    return cases


def build_count_cases():
    """Exhaustive total decision-table sweep over the nine counters.

    These are not observations of the fixture; they pin the classifier as a
    total function, including combinations the fixture cannot produce (which is
    exactly where a future refactor could drift).
    """
    cases = []

    def add(counts, contract=1):
        cases.append(Case(len(cases), "counts", "counts_%d" % len(cases),
                          contract, counts=dict(counts)))

    keys = ("dead", "c1", "c2_nonzero", "c2_zero", "ambig", "zero_loss",
            "match", "unknown", "x_mismatch")
    for mask in range(1 << len(keys)):
        counts = {k: ((mask >> i) & 1) for i, k in enumerate(keys)}
        add(counts)
    for match in (1023, CELLS):
        sub_keys = keys[:6] + keys[7:]
        for mask in range(1 << len(sub_keys)):
            counts = {k: 0 for k in keys}
            counts["match"] = match
            for i, k in enumerate(sub_keys):
                counts[k] = (mask >> i) & 1
            add(counts)
    for mask in range(1 << len(keys)):
        counts = {k: ((mask >> i) & 1) for i, k in enumerate(keys)}
        add(counts, contract=0)
    return cases


# -------------------------------------------------------------- expectations
def python_expectation(case, exp_pos, exp_neg):
    if case.kind == "counts":
        obs = {k: case.counts[k] for k in case.counts}
        return indep.classify(obs, case.contract), obs
    obs = indep.observe_cells(case.p, case.x, exp_pos, exp_neg)
    return indep.classify(obs, case.contract), obs


# ------------------------------------------------------------------- replay
def write_cases(cases, exp_pos, exp_neg):
    with open(CASE_FILE, "w") as fh:
        fh.write("# C2 classifier differential replay cases (CPU only)\n")
        fh.write("# generated by hmx_c2_differential_test.py; do not hand edit\n")
        for c in cases:
            if c.kind == "counts":
                fh.write("counts %d %d %d %d %d %d %d %d %d %d %d\n" % (
                    c.cid, c.contract,
                    c.counts["dead"], c.counts["c1"], c.counts["c2_nonzero"],
                    c.counts["c2_zero"], c.counts["ambig"], c.counts["zero_loss"],
                    c.counts["match"], c.counts["unknown"],
                    c.counts["x_mismatch"]))
            else:
                fh.write("arrays %d %d\n" % (c.cid, c.contract))
                for tag, arr in (("P", c.p), ("X", c.x),
                                 ("EP", exp_pos), ("EN", exp_neg)):
                    fh.write("%s %s\n" % (tag, " ".join("%04x" % v for v in arr)))


def compile_replay(compiler, opt, out_path):
    cmd = [compiler, "-std=gnu11", "-O" + str(opt), "-Wall", "-Wextra",
           "-Werror", "-ffp-contract=off", "-I", HERE, "-o", out_path,
           SIM_SOURCE, "-lm"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    if proc.returncode != 0:
        return None, "exit %d: %s" % (proc.returncode, proc.stdout.strip()[-800:])
    return out_path, None


def run_replay(exe):
    proc = subprocess.run([exe] + REPLAY_ARGS, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    answers = {}
    malformed = 0
    for line in proc.stdout.splitlines():
        if line.startswith("replay "):
            parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            answers[int(parts[1])] = parts[2]
        elif line.startswith("c2_replay cases="):
            for token in line.split()[1:]:
                if token.startswith("malformed="):
                    malformed = int(token.split("=", 1)[1])
    return proc.returncode, answers, malformed, proc.stdout


def compare(cases, expected, answers):
    mismatches = []
    for c in cases:
        got = answers.get(c.cid)
        want = expected[c.cid]
        if got != want:
            mismatches.append((c, want, got))
    missing = [c.cid for c in cases if c.cid not in answers]
    return mismatches, missing


def save_repro(cases, mismatches, impl):
    saved = []
    for case, want, got in mismatches[:8]:
        name = "c2_differential_mismatch_%s_%d.json" % (impl.replace("/", "_"), case.cid)
        path = os.path.join(BUILD, name)
        payload = {
            "implementation": impl,
            "case_id": case.cid,
            "kind": case.kind,
            "description": case.desc,
            "contract_ok": case.contract,
            "python_class": want,
            "c_class": got,
            "counts": case.counts,
            "p": case.p,
            "x": case.x,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        saved.append(path)
    return saved


# --------------------------------------------------------------------- main
def main():
    if not os.path.isdir(BUILD):
        os.makedirs(BUILD)
    exp_pos, exp_neg = load_fixture()

    cases = build_array_cases(exp_pos, exp_neg)
    cases += build_random_cases(exp_pos, exp_neg, seeds=256)
    cases += build_count_cases()
    # The three builders number their own lists from zero; renumber globally so
    # an id names exactly one case in the replay file.
    for i, case in enumerate(cases):
        case.cid = i
    array_cases = sum(1 for c in cases if c.kind == "arrays")
    count_cases = sum(1 for c in cases if c.kind == "counts")

    # Python answers are computed BEFORE any C process runs.
    expected = {}
    for c in cases:
        expected[c.cid], _obs = python_expectation(c, exp_pos, exp_neg)

    write_cases(cases, exp_pos, exp_neg)

    log("=== C2 classifier differential audit (CPU only) ===")
    log("cases: arrays=%d counts=%d total=%d" % (array_cases, count_cases,
                                                len(cases)))
    log("expectations computed from the Python implementation before C ran")
    histogram = {}
    for cls in expected.values():
        histogram[cls] = histogram.get(cls, 0) + 1
    log("python class distribution: %s"
        % json.dumps(histogram, sort_keys=True))

    runs = []
    # Do NOT honour $CC: on the Windows development machines CC is commonly
    # pinned to cl.exe, whose flags are incompatible with the gcc-style build
    # this source requires. An explicit --cc= argument is the only override.
    gcc = "gcc"
    clang_exe = None
    for arg in sys.argv[1:]:
        if arg.startswith("--cc="):
            gcc = arg.split("=", 1)[1]
        elif arg.startswith("--clang-exe="):
            clang_exe = arg.split("=", 1)[1]
    if shutil.which(gcc) is None:
        log("SKIP (no C compiler %r on PATH)" % gcc)
        return 2
    for opt in (2, 0):
        exe = os.path.join(BUILD, "hmx_c2_replay_gcc_o%d.exe" % opt)
        path, err = compile_replay(gcc, opt, exe)
        if path is None:
            log("FAIL gcc-O%d compile %s" % (opt, err))
            return 2
        runs.append(("gcc-O%d" % opt, path))

    if clang_exe:
        if not os.path.isfile(clang_exe):
            log("SKIP clang replay (no binary at %s)" % clang_exe)
        else:
            runs.append(("clang-O2", clang_exe))

    total_mismatch = 0
    verified_ids = None
    for name, exe in runs:
        rc, answers, malformed, out = run_replay(exe)
        if rc != 0 or malformed:
            log("FAIL %s replay exit=%d malformed=%s" % (name, rc, malformed))
            log(out[-1500:])
            return 1
        mismatches, missing = compare(cases, expected, answers)
        if missing:
            log("FAIL %s: %d cases produced no answer" % (name, len(missing)))
            total_mismatch += len(missing)
            continue
        if mismatches:
            total_mismatch += len(mismatches)
            for case, want, got in mismatches[:10]:
                log("MISMATCH %s case=%d (%s): python=%s c=%s"
                    % (name, case.cid, case.desc, want, got))
            for path in save_repro(cases, mismatches, name):
                log("repro saved: %s" % path)
        log("c2_diff_impl=%s cases=%d mismatches=%d answers=%d"
            % (name, len(cases), len(mismatches), len(answers)))
        if verified_ids is None:
            verified_ids = set(answers)
        else:
            if set(answers) != verified_ids:
                log("FAIL %s answered a different case set" % name)
                total_mismatch += 1

    # The comparator must not be vacuous: corrupt one C answer and require a
    # mismatch. This runs on the first implementation's in-memory output only.
    rc, answers, malformed, out = run_replay(runs[0][1])
    victim = next((c.cid for c in cases
                   if expected[c.cid] == MUTATION_CLASS), None)
    if victim is None:
        log("FAIL mutation self-check: no case classified %s" % MUTATION_CLASS)
        total_mismatch += 1
    else:
        mutated = dict(answers)
        mutated[victim] = MUTATION_REPLACEMENT
        mut_mism, _ = compare(cases, expected, mutated)
        if len(mut_mism) != 1 or mut_mism[0][0].cid != victim:
            log("FAIL mutation self-check: comparator missed the corruption")
            total_mismatch += 1
        else:
            log("c2_diff_mutation_detected=1 case=%d %s->%s"
                % (victim, MUTATION_CLASS, MUTATION_REPLACEMENT))

    checks = len(cases) * len(runs)
    log("c2_diff_checks=%d fail=%d implementations=%d"
        % (checks, total_mismatch, len(runs)))
    verdict = "C2_DIFFERENTIAL_PASS" if total_mismatch == 0 else "C2_DIFFERENTIAL_FAIL"
    log("c2_diff_verdict=%s" % verdict)
    return 0 if total_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
