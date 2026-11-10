#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""Independent CPU audit of the D8 revision-3 contract proposal (2026-11-10).

This test does not call the C contract implementation. It
    1. parses the machine-readable blocks in
       docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md and
       docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md,
    2. parses the metadata indices, masks and constants from
       host_tests/hmx_acc_residual_rpc/hmx_d8_contract.h,
    3. re-derives the 32-word sample vector for each condition and requires the
       document's sample vectors to match,
    4. re-implements the host contract gate and runs curated defects plus
       deterministic random mutations through the C replay mode
       `hmx_d8_rev3_selftest --d8-contract-replay`,
    5. compares the C revision-3 issued-order signatures with the document
       block and with an independent expansion,
    6. checks the q map, the 23-word ownership, signed/unsigned handling and
       the failure-reason namespace.

CPU only. No device, DSP, SDK, network or Mode 46.

Exit: 0 = all invariants hold, 1 = mismatch, 2 = missing environment.
"""

import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build")
HEADER = os.path.join(HERE, "hmx_d8_contract.h")
PREREG = os.path.join(REPO, "docs",
                      "hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md")
CONTRACT = os.path.join(REPO, "docs",
                        "hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md")
FAILURE = os.path.join(REPO, "docs",
                       "hmx-v81-d8-fastrpc-failure-contract.md")
CASES = os.path.join(BUILD, "hmx_d8_rev3_cases.txt")

CHECKS = {"n": 0, "fail": 0}


def check(cond, tag):
    CHECKS["n"] += 1
    if not cond:
        CHECKS["fail"] += 1
        print("FAIL %s" % tag)
    return cond


def marked_block(path, marker):
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    inside = False
    out = []
    for line in lines:
        if ("%s:begin" % marker) in line:
            inside = True
            continue
        if ("%s:end" % marker) in line:
            break
        if inside:
            out.append(line)
    return out


# --------------------------------------------------------------- header parse
def parse_header(path):
    text = open(path, encoding="utf-8").read()
    enum = {}
    for name, value in re.findall(
            r"HMX_D8_META_([A-Z0-9_]+)\s*=\s*(\d+)", text):
        enum[name] = int(value)
    defines = {}
    for name, value in re.findall(
            r"#define\s+(HMX_D8_[A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+|\d+)[uU]?",
            text):
        defines[name] = int(value, 0)
    for name, value in re.findall(
            r"HMX_D8_FAIL_([A-Z0-9_]+)\s*=\s*(\d+)", text):
        defines["HMX_D8_FAIL_" + name] = int(value)
    return enum, defines


def fnv1a32(data):
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def u32(v):
    return v & 0xFFFFFFFF


# -------------------------------------------------------------- contract gate
class Contract:
    def __init__(self, enum, defines):
        self.e = enum
        self.d = defines
        self.guard_zero = fnv1a32(bytes(1536))
        self.fill_bits = {
            0: defines["HMX_D8_FILL_OUT"],
            1: (defines["HMX_D8_FILL_OUT"] | defines["HMX_D8_FILL_WLO"] |
                defines["HMX_D8_FILL_AHI"]),
            2: defines["HMX_D8_FILL_OUT"] | defines["HMX_D8_FILL_WLO"],
            3: defines["HMX_D8_FILL_OUT"] | defines["HMX_D8_FILL_AHI"],
        }

    def word(self, name):
        return self.e[name]

    def condition_for_q(self, q):
        return q - 6 if 6 <= q <= 9 else -1

    def valid_vector(self, cond, build=0xD8B01110, elapsed=4321):
        d = self.d
        q = 6 + cond
        m = [0] * 32
        m[0] = d["HMX_D8_MAGIC"]
        m[1] = d["HMX_D8_ARCH"]
        m[2] = d["HMX_D8_VTCM_BYTES"]
        m[3] = elapsed
        m[4] = 0
        m[5] = 0
        m[6] = 0
        m[7] = 1
        m[8] = 1
        m[self.word("BUILD_ID")] = build
        m[self.word("MXCLRACC")] = 1
        m[self.word("MMA")] = 3
        m[self.word("CONVERT")] = 1
        m[self.word("STORE")] = 1
        m[self.word("PACK")] = 4
        m[self.word("BIAS")] = 2
        m[self.word("FILL")] = (
            self.fill_bits[cond] |
            (d["HMX_D8_FILL_WORD"] << d["HMX_D8_FILL_WORD_SHIFT"]) |
            (d["HMX_D8_FILL_FORM_16X128"] << d["HMX_D8_FILL_FORM_SHIFT"]))
        m[self.word("ADDRESS")] = (
            d["HMX_D8_ADDR_OUT"] |
            ((d["HMX_D8_ADDR_WLO"] if cond in (1, 2) else 0)
             << d["HMX_D8_ADDR_WLO_SHIFT"]) |
            (d["HMX_D8_ADDR_AHI"] << d["HMX_D8_ADDR_AHI_SHIFT"]))
        m[self.word("SELECTOR")] = (
            cond | (q << d["HMX_D8_SELECTOR_Q_SHIFT"]))
        m[self.word("FNV_AHI")] = 0xC6231266
        m[self.word("FNV_ALO")] = 0x3B2D22D2
        m[self.word("FNV_WHI")] = 0x8B88609F
        m[self.word("FNV_WLO")] = 0xA66AF695
        m[self.word("GUARD_PRE")] = self.guard_zero
        m[self.word("GUARD_POST")] = self.guard_zero
        m[self.word("GUARD_STATUS")] = (
            d["HMX_D8_GUARD_STATUS_OK"] |
            (d["HMX_D8_GUARD_FIRST_BAD_NONE"] <<
             d["HMX_D8_GUARD_FIRST_BAD_SHIFT"]) |
            (d["HMX_D8_GUARD_INIT_HVX12"] << d["HMX_D8_GUARD_INIT_SHIFT"]) |
            (d["HMX_D8_GUARD_CHECK_BYTE"] << d["HMX_D8_GUARD_CHECK_SHIFT"]))
        m[self.word("FAILURE")] = d["HMX_D8_FAIL_NONE"]
        m[self.word("SCHEMA")] = d["HMX_D8_SCHEMA"]
        m[self.word("READBACK")] = (
            (d["HMX_D8_TRACE_NOT_PROVIDED"] <<
             d["HMX_D8_READBACK_TRACE_SHIFT"]) |
            (1 << d["HMX_D8_READBACK_DONE_SHIFT"]) |
            (0 << d["HMX_D8_READBACK_HVX_SHIFT"]) |
            (d["HMX_D8_RESULT_SLOTS"] << d["HMX_D8_READBACK_SLOTS_SHIFT"]))
        m[self.word("RESERVED_29")] = 0
        m[self.word("CLEANUP")] = 0
        m[self.word("RESERVED_31")] = 0
        return m

    def contract(self, meta, q, build):
        """Mirror of hmx_d8_contract_check(); returns (ok, first_bad)."""
        d = self.d
        if meta is None or len(meta) != d["HMX_D8_METADATA_LEN"]:
            return False, "meta_len"
        w = lambda name: u32(meta[self.word(name)])
        if not (u32(meta[0]) == d["HMX_D8_MAGIC"] and
                meta[1] == d["HMX_D8_ARCH"] and
                meta[2] == d["HMX_D8_VTCM_BYTES"] and
                meta[7] == 1 and meta[8] == 1):
            return False, "generic"
        if w("SCHEMA") != d["HMX_D8_SCHEMA"]:
            return False, "schema"
        if build != 0 and w("BUILD_ID") != u32(build):
            return False, "build_id"
        cond = self.condition_for_q(q)
        sel = w("SELECTOR")
        scond = sel & d["HMX_D8_SELECTOR_COND_BITS"]
        sq = (sel >> d["HMX_D8_SELECTOR_Q_SHIFT"]) & \
            d["HMX_D8_SELECTOR_Q_BITS"]
        if not (cond >= 0 and scond == cond and sq == q and
                (sel & 0xFFFFE0F0) == 0):
            return False, "selector"
        if meta[self.word("MXCLRACC")] != 1:
            return False, "mxclracc"
        if meta[self.word("MMA")] != 3:
            return False, "mma"
        if meta[self.word("CONVERT")] != 1:
            return False, "convert"
        if meta[self.word("STORE")] != 1:
            return False, "store"
        if meta[self.word("PACK")] != 4:
            return False, "pack"
        if meta[self.word("BIAS")] != 2:
            return False, "bias"
        fill = w("FILL")
        if (fill & d["HMX_D8_FILL_MASK_BITS"]) != self.fill_bits[cond]:
            return False, "fill_mask"
        if ((fill >> d["HMX_D8_FILL_WORD_SHIFT"]) &
                d["HMX_D8_FILL_WORD_BITS"]) != d["HMX_D8_FILL_WORD"]:
            return False, "fill_word"
        if not (((fill >> d["HMX_D8_FILL_FORM_SHIFT"]) &
                 d["HMX_D8_FILL_FORM_BITS"]) ==
                d["HMX_D8_FILL_FORM_16X128"] and
                (fill & 0xF00000F8) == 0):
            return False, "fill_form"
        addr = w("ADDRESS")
        wlo = d["HMX_D8_ADDR_WLO"] if cond in (1, 2) else 0
        if not ((addr & d["HMX_D8_ADDR_BITS"]) == d["HMX_D8_ADDR_OUT"] and
                ((addr >> d["HMX_D8_ADDR_WLO_SHIFT"]) &
                 d["HMX_D8_ADDR_BITS"]) == wlo and
                ((addr >> d["HMX_D8_ADDR_AHI_SHIFT"]) &
                 d["HMX_D8_ADDR_BITS"]) == d["HMX_D8_ADDR_AHI"] and
                (addr & 0xFFE00000) == 0):
            return False, "address"
        if not (w("FNV_AHI") == 0xC6231266 and w("FNV_ALO") == 0x3B2D22D2 and
                w("FNV_WHI") == 0x8B88609F and w("FNV_WLO") == 0xA66AF695):
            return False, "operand_fnv"
        if not (w("GUARD_PRE") == self.guard_zero and
                w("GUARD_POST") == w("GUARD_PRE")):
            return False, "guard_digest"
        gs = w("GUARD_STATUS")
        if not ((gs & d["HMX_D8_GUARD_STATUS_BITS"]) ==
                d["HMX_D8_GUARD_STATUS_OK"] and
                ((gs >> d["HMX_D8_GUARD_FIRST_BAD_SHIFT"]) &
                 d["HMX_D8_GUARD_FIRST_BAD_BITS"]) ==
                d["HMX_D8_GUARD_FIRST_BAD_NONE"] and
                ((gs >> d["HMX_D8_GUARD_INIT_SHIFT"]) &
                 d["HMX_D8_GUARD_INIT_BITS"]) ==
                d["HMX_D8_GUARD_INIT_HVX12"] and
                ((gs >> d["HMX_D8_GUARD_CHECK_SHIFT"]) &
                 d["HMX_D8_GUARD_CHECK_BITS"]) ==
                d["HMX_D8_GUARD_CHECK_BYTE"] and
                (gs & 0xF0000000) == 0):
            return False, "guard_status"
        if meta[self.word("FAILURE")] != d["HMX_D8_FAIL_NONE"]:
            return False, "failure_code"
        rb = w("READBACK")
        if not (((rb >> d["HMX_D8_READBACK_TRACE_SHIFT"]) &
                 d["HMX_D8_READBACK_TRACE_BITS"]) ==
                d["HMX_D8_TRACE_NOT_PROVIDED"] and
                ((rb >> d["HMX_D8_READBACK_DONE_SHIFT"]) &
                 d["HMX_D8_READBACK_DONE_BITS"]) == 1 and
                ((rb >> d["HMX_D8_READBACK_HVX_SHIFT"]) &
                 d["HMX_D8_READBACK_HVX_BITS"]) == 0 and
                ((rb >> d["HMX_D8_READBACK_SLOTS_SHIFT"]) &
                 d["HMX_D8_READBACK_SLOTS_BITS"]) ==
                d["HMX_D8_RESULT_SLOTS"] and
                (rb & 0xFFC00000) == 0):
            return False, "readback"
        if meta[self.word("CLEANUP")] != 0:
            return False, "cleanup"
        if meta[self.word("RESERVED_29")] != 0 or \
                meta[self.word("RESERVED_31")] != 0:
            return False, "reserved"
        return True, "ok"


# --------------------------------------------------------------- doc parsing
def parse_layout(path):
    rows = []
    for line in marked_block(path, "d8-meta-layout"):
        parts = line.split()
        if len(parts) == 2:
            rows.append((int(parts[0]), parts[1]))
    return rows


def parse_vectors(path):
    vectors = {}
    for line in marked_block(path, "d8-meta-values"):
        parts = line.split()
        if len(parts) != 33:
            continue
        vectors[parts[0]] = [int(v, 16) for v in parts[1:]]
    return vectors


def parse_q_map(path):
    table = {}
    for line in marked_block(path, "d8-q-map"):
        parts = line.split()
        if len(parts) == 2:
            table[int(parts[0])] = parts[1]
    return table


def parse_sequence_block(path):
    tokens = []
    for line in marked_block(path, "d8-rev3-sequence"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "PACK" and len(parts) == 2:
            tokens.append(("PACK", parts[1], None))
        elif kind == "BIAS" and len(parts) == 2:
            tokens.append(("BIAS", parts[1], None))
        elif kind == "GUARD" and len(parts) >= 2:
            tokens.append(("GUARD", parts[1], parts[2:] or None))
        elif kind == "MXCLRACC" and len(parts) == 1:
            tokens.append(("MXCLRACC", None, None))
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


def expand_sequence(tokens, letter):
    out = []
    for kind, a, b in tokens:
        if kind == "PACK":
            out.append("PACK:%s" % a)
        elif kind == "BIAS":
            out.append("BIAS:%s" % a)
        elif kind == "GUARD":
            if a == "PRE":
                out.append("GUARD:PRE")
            elif a == "POST":
                out.append("GUARD:POST")
            else:
                out.append("GUARD:%s:%s" % (a, b[0] if b else "?"))
        elif kind == "MXCLRACC":
            out.append("MXCLRACC")
        elif kind == "MMA":
            out.append("MMA:%s:%s:%s" % (a, b[0], b[1]))
        elif kind == "FILL":
            conds = b or ("A", "B", "C", "D")
            if letter in conds:
                out.append("FILL:%s" % a)
        elif kind == "CONVERT":
            out.append("CONVERT:%s" % a)
        elif kind == "STORE":
            out.append("STORE:%s" % a)
        else:
            out.append("UNKNOWN")
    return out


def run_c_sequence(exe):
    proc = subprocess.run([exe, "--d8-sequence-rev3"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    seqs = {}
    for line in proc.stdout.splitlines():
        if line.startswith("d8_rev3_seq cond="):
            head, events = line.split(" events=", 1)
            letter = head.split("cond=", 1)[1]
            seqs[letter] = events.split(",") if events else []
    return proc.returncode, seqs, proc.stdout


# ------------------------------------------------------------- curated faults
def curated_cases(c, base_cond=1):
    """Return (name, words, q, build, expected_first_bad)."""
    d = c.d
    B = c.word("BUILD_ID")
    SEL = c.word("SELECTOR")
    FILL = c.word("FILL")
    ADDR = c.word("ADDRESS")
    GS = c.word("GUARD_STATUS")
    RB = c.word("READBACK")
    cases = []

    def add(name, cond, mutate, expected):
        words = c.valid_vector(cond)
        q = 6 + cond
        build = words[B]
        mutate(words)
        cases.append((name, words, q, build, expected))

    add("generic_magic", 1, lambda w: w.__setitem__(0, 0), "generic")
    add("generic_arch", 1, lambda w: w.__setitem__(1, 80), "generic")
    add("generic_vtcm", 1, lambda w: w.__setitem__(2, 8192), "generic")
    add("generic_hmx", 1, lambda w: w.__setitem__(7, 0), "generic")
    add("generic_trials", 1, lambda w: w.__setitem__(8, 2), "generic")
    add("schema_wrong", 1,
        lambda w: w.__setitem__(c.word("SCHEMA"), d["HMX_D8_SCHEMA"] + 1),
        "schema")
    add("build_id_wrong", 1, lambda w: w.__setitem__(B, 0xD8B01111), "build_id")
    add("condition_in_build_slot", 2, lambda w: w.__setitem__(B, 3), "build_id")
    add("build_id_in_selector_slot", 2,
        lambda w: w.__setitem__(SEL, 0xD8B01110), "selector")
    add("c_d_swap", 2,
        lambda w: w.__setitem__(SEL, 3 | (8 << d["HMX_D8_SELECTOR_Q_SHIFT"])),
        "selector")
    add("q_echo_wrong", 2,
        lambda w: w.__setitem__(SEL, 2 | (9 << d["HMX_D8_SELECTOR_Q_SHIFT"])),
        "selector")
    add("selector_reserved_bit", 2, lambda w: w.__setitem__(SEL, w[SEL] | 0x10),
        "selector")
    add("count_mxclracc", 3,
        lambda w: w.__setitem__(c.word("MXCLRACC"), 2), "mxclracc")
    add("count_mma", 3, lambda w: w.__setitem__(c.word("MMA"), 2), "mma")
    add("count_convert", 3, lambda w: w.__setitem__(c.word("CONVERT"), 2),
        "convert")
    add("count_store", 3, lambda w: w.__setitem__(c.word("STORE"), 0), "store")
    add("count_pack", 3, lambda w: w.__setitem__(c.word("PACK"), 3), "pack")
    add("count_bias", 3, lambda w: w.__setitem__(c.word("BIAS"), 3), "bias")
    add("c_uses_d_mask", 2,
        lambda w: w.__setitem__(FILL,
                                (w[FILL] & ~0x7) | 5), "fill_mask")
    add("fill_word_wrong", 2,
        lambda w: w.__setitem__(FILL,
                                (w[FILL] & ~0xFFFF00) | (0x5A5A << 8)),
        "fill_word")
    add("fill_form_scalar", 2,
        lambda w: w.__setitem__(FILL,
                                (w[FILL] & ~0xF000000) | (2 << 24)),
        "fill_form")
    add("fill_reserved_bit", 2,
        lambda w: w.__setitem__(FILL, w[FILL] | 0x08), "fill_form")
    add("store_target_wrong", 2,
        lambda w: w.__setitem__(ADDR, 63 | (48 << 7)), "address")
    add("wlo_target_wrong", 2,
        lambda w: w.__setitem__(ADDR, 64 | (40 << 7)), "address")
    add("ahi_target_wrong", 2,
        lambda w: w.__setitem__(ADDR, 64 | (48 << 7) | (1 << 14)), "address")
    add("address_reserved_bit", 2,
        lambda w: w.__setitem__(ADDR, w[ADDR] | 0x100000), "address")
    add("d_with_wlo_address", 3,
        lambda w: w.__setitem__(ADDR, 64 | (48 << 7)), "address")
    add("fnv_wlo_wrong", 0,
        lambda w: w.__setitem__(c.word("FNV_WLO"), 0xA66AF694), "operand_fnv")
    add("fnv_ahi_wrong", 0,
        lambda w: w.__setitem__(c.word("FNV_AHI"), 0xC6231267), "operand_fnv")
    add("guard_pre_nonzero", 1,
        lambda w: w.__setitem__(c.word("GUARD_PRE"),
                                c.guard_zero ^ 1), "guard_digest")
    add("guard_post_changed", 1,
        lambda w: w.__setitem__(c.word("GUARD_POST"),
                                c.guard_zero ^ 0x100), "guard_digest")
    add("guard_status_changed", 1,
        lambda w: w.__setitem__(
            GS, d["HMX_D8_GUARD_STATUS_POST_CHANGED"] |
            (12 << d["HMX_D8_GUARD_FIRST_BAD_SHIFT"]) |
            (d["HMX_D8_GUARD_INIT_HVX12"] << d["HMX_D8_GUARD_INIT_SHIFT"]) |
            (d["HMX_D8_GUARD_CHECK_BYTE"] << d["HMX_D8_GUARD_CHECK_SHIFT"])),
        "guard_status")
    add("guard_init_missing", 1,
        lambda w: w.__setitem__(
            GS, (w[GS] & ~(0xF << d["HMX_D8_GUARD_INIT_SHIFT"]))),
        "guard_status")
    add("guard_check_missing", 1,
        lambda w: w.__setitem__(
            GS, (w[GS] & ~(0xF << d["HMX_D8_GUARD_CHECK_SHIFT"]))),
        "guard_status")
    add("guard_status_undefined", 1,
        lambda w: w.__setitem__(GS, w[GS] | 0x8), "guard_status")
    add("failure_code_nonzero", 2,
        lambda w: w.__setitem__(c.word("FAILURE"),
                                d["HMX_D8_FAIL_GUARD_POST"]), "failure_code")
    add("failure_code_out_of_range", 2,
        lambda w: w.__setitem__(c.word("FAILURE"),
                                d["HMX_D8_FAIL_COUNT"] + 5), "failure_code")
    add("trace_claimed", 2,
        lambda w: w.__setitem__(
            RB, 1 | (1 << d["HMX_D8_READBACK_DONE_SHIFT"]) |
            (1024 << d["HMX_D8_READBACK_SLOTS_SHIFT"])), "readback")
    add("slots_short", 2,
        lambda w: w.__setitem__(
            RB, (1 << d["HMX_D8_READBACK_DONE_SHIFT"]) |
            (1023 << d["HMX_D8_READBACK_SLOTS_SHIFT"])), "readback")
    add("hvx_lock_failed", 2,
        lambda w: w.__setitem__(
            RB, (1 << d["HMX_D8_READBACK_DONE_SHIFT"]) |
            (3 << d["HMX_D8_READBACK_HVX_SHIFT"]) |
            (1024 << d["HMX_D8_READBACK_SLOTS_SHIFT"])), "readback")
    add("readback_missing", 2,
        lambda w: w.__setitem__(
            RB, (1024 << d["HMX_D8_READBACK_SLOTS_SHIFT"])), "readback")
    add("cleanup_failed", 2,
        lambda w: w.__setitem__(c.word("CLEANUP"), 14008), "cleanup")
    add("reserved31_nonzero", 2,
        lambda w: w.__setitem__(c.word("RESERVED_31"), 1), "reserved")
    add("reserved29_nonzero", 2,
        lambda w: w.__setitem__(c.word("RESERVED_29"), 1), "reserved")
    return cases


# ------------------------------------------------------------------ replay IO
def write_cases(cases, path):
    with open(path, "w") as fh:
        fh.write("# D8 revision-3 contract replay cases (CPU only)\n")
        for cid, case in enumerate(cases):
            name, words, q, build, expected = case
            fh.write("case %d %d %08x %s\n" % (
                cid, q, build, " ".join("%08x" % u32(v) for v in words)))


def run_replay(exe, path):
    proc = subprocess.run([exe, "--d8-contract-replay", path],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    answers = {}
    malformed = 0
    for line in proc.stdout.splitlines():
        if line.startswith("replay "):
            parts = line.split()
            if len(parts) != 4:
                malformed += 1
                continue
            answers[int(parts[1])] = (parts[2], parts[3])
        elif line.startswith("d8_contract_replay cases="):
            for token in line.split()[1:]:
                if token.startswith("malformed="):
                    malformed = int(token.split("=", 1)[1])
    return proc.returncode, answers, malformed, proc.stdout


def main():
    if not os.path.isdir(BUILD):
        os.makedirs(BUILD)
    enum, defines = parse_header(HEADER)
    c = Contract(enum, defines)

    # 1. Layout: 23 words, 9..31, no overlap with 0..8.
    rows = parse_layout(CONTRACT)
    check(len(rows) == 23, "layout_23_words")
    check([i for i, _ in rows] == list(range(9, 32)), "layout_indices_9_31")
    check(len({i for i, _ in rows}) == 23, "layout_no_duplicate")
    check(all(i >= 9 for i, _ in rows), "layout_no_generic_words")
    doc_names = {name: i for i, name in rows}
    header_names = {name.lower(): i for name, i in enum.items()}
    check(doc_names == header_names, "layout_matches_header_enum")
    check(defines["HMX_D8_METADATA_LEN"] == 32, "metadata_len_32")
    check(defines["HMX_D8_RESULT_SLOTS"] == 1024, "result_slots_1024")
    check(defines["HMX_D8_MODE"] == 46, "mode_46")
    check(defines["HMX_D8_SCHEMA"] < 0x80000000, "schema_positive_signed")

    # 2. Sample vectors re-derived independently and compared with the doc.
    vectors = parse_vectors(CONTRACT)
    check(len(vectors) == 4, "sample_vectors_present")
    for cond, letter in enumerate("ABCD"):
        want = c.valid_vector(cond)
        got = vectors.get(letter)
        check(got == [u32(v) for v in want], "sample_vector_%s" % letter)
        ok, first = c.contract(want, 6 + cond, want[c.word("BUILD_ID")])
        check(ok and first == "ok", "sample_vector_accepts_%s" % letter)

    # 3. Guard zero digest is independent of the C helper.
    check(c.guard_zero == 0xF5EBD5C5, "guard_zero_digest")
    check(re.search(r"0xf5ebd5c5", open(CONTRACT).read(), re.I) is not None,
          "guard_zero_digest_in_doc")

    # 4. q map.
    qmap = parse_q_map(PREREG)
    check(qmap == {6: "A", 7: "B", 8: "C", 9: "D"}, "q_map_block")
    check(c.condition_for_q(6) == 0 and c.condition_for_q(9) == 3,
          "q_map_python")
    check(c.condition_for_q(5) == -1 and c.condition_for_q(10) == -1,
          "q_out_of_range")
    check(defines["HMX_D8_Q_FIRST"] == 6 and defines["HMX_D8_Q_LAST"] == 9,
          "q_defines")
    check(defines["HMX_D8_Q_IDENTITY"] == 6, "identity_q_is_6")
    check(defines["HMX_D8_IDENTITY_MODE"] == 30, "identity_mode_30")

    # The current DSP does not dispatch mode 46 and rejects q<6. The proposal
    # must not pretend that the mode already exists.
    probe = open(os.path.join(HERE, "probe_dsp.c"), encoding="utf-8").read()
    host = open(os.path.join(HERE, "probe_host.c"), encoding="utf-8").read()
    mode_max = re.search(r"MODE_MAX\s*=\s*(\d+)", probe)
    check(mode_max is not None and int(mode_max.group(1)) == 45,
          "dsp_mode_max_45")
    check(defines["HMX_D8_MODE"] > int(mode_max.group(1)),
          "mode46_not_dispatched")
    check("MODE46" not in probe and "mode == 46" not in probe,
          "mode46_absent_in_dsp")
    check("q < 6 || q > 10" in probe, "dsp_q_gate")
    check("q < 6 || q > 10" in host, "host_q_gate")

    # 5. Field decodes for each condition.
    for cond, letter in enumerate("ABCD"):
        v = c.valid_vector(cond)
        fill = u32(v[c.word("FILL")])
        addr = u32(v[c.word("ADDRESS")])
        sel = u32(v[c.word("SELECTOR")])
        check((fill & 7) == c.fill_bits[cond], "decode_fill_%s" % letter)
        check(((fill >> 8) & 0xFFFF) == 0xA5A5, "decode_word_%s" % letter)
        check(((fill >> 24) & 0xF) == 1, "decode_form_%s" % letter)
        check((addr & 0x7F) == 64, "decode_out_%s" % letter)
        check(((addr >> 7) & 0x7F) == (48 if cond in (1, 2) else 0),
              "decode_wlo_%s" % letter)
        check(((sel >> 8) & 0x1F) == 6 + cond, "decode_q_%s" % letter)
        check((sel & 0xF) == cond, "decode_cond_%s" % letter)

    # 6. Signed/unsigned handling.
    high = c.valid_vector(1, build=0xF0000001)
    ok, _ = c.contract(high, 7, 0xF0000001)
    check(ok, "build_id_high_bit")
    fnv_high = c.valid_vector(1)
    fnv_high[c.word("FNV_WHI")] = 0xFFFFFFFF
    ok, first = c.contract(fnv_high, 7, 0xD8B01110)
    check(not ok and first == "operand_fnv", "fnv_high_bit_compare")
    bad_addr = c.valid_vector(1)
    bad_addr[c.word("ADDRESS")] = (bad_addr[c.word("ADDRESS")] & ~0x7F) | 0x7F
    ok, first = c.contract(bad_addr, 7, 0xD8B01110)
    check(not ok and first == "address", "address_invalid_sentinel")

    # 7. Curated defects: Python gate must reject with the expected field.
    cases = curated_cases(c)
    for name, words, q, build, expected in cases:
        got_ok, got_first = c.contract(words, q, build)
        check(not got_ok and got_first == expected,
              "curated_%s" % name)

    # 8. Words 3..6 are run()-owned and must not affect the D8 gate.
    for w in (3, 4, 5, 6):
        vec = c.valid_vector(2)
        vec[w] ^= 0xFFFF
        ok, _ = c.contract(vec, 8, 0xD8B01110)
        check(ok, "generic_owned_ignored_%d" % w)

    # 9. Deterministic random mutations.
    rng = random.Random(20261110)
    random_cases = []
    for i in range(200):
        cond = rng.randrange(4)
        vec = c.valid_vector(cond)
        n = 1 + rng.randrange(3)
        for _ in range(n):
            idx = rng.randrange(32)
            vec[idx] = u32(vec[idx] ^ (1 << rng.randrange(32)))
        q = 6 + cond
        ok, _ = c.contract(vec, q, 0xD8B01110)
        random_cases.append(("random_%d" % i, vec, q, 0xD8B01110,
                             "ok" if ok else "any"))
    all_cases = cases + random_cases

    # 10. Replay through the C implementation.
    exe = None
    for arg in sys.argv[1:]:
        if arg.startswith("--exe="):
            exe = arg.split("=", 1)[1]
    if exe is None:
        exe = os.path.join(BUILD, "hmx_d8_rev3_selftest.exe")
    if not os.path.isfile(exe):
        print("SKIP (no C contract binary at %s)" % exe)
        return 2

    write_cases(all_cases, CASES)
    rc, answers, malformed, out = run_replay(exe, CASES)
    if rc != 0 or malformed:
        print("FAIL replay exit=%d malformed=%d" % (rc, malformed))
        print(out[-1500:])
        return 1
    check(len(answers) == len(all_cases), "replay_case_count")
    mismatches = 0
    for cid, case in enumerate(all_cases):
        name, words, q, build, expected = case
        got = answers.get(cid)
        if got is None:
            mismatches += 1
            continue
        got_ok = got[0] == "allow"
        want_ok, want_first = c.contract(words, q, build)
        if got_ok != want_ok:
            mismatches += 1
            print("MISMATCH case=%d (%s) c=%s python=%s"
                  % (cid, name, got, (want_ok, want_first)))
            continue
        if expected not in ("ok", "any") and got[1] != expected:
            mismatches += 1
            print("FIELD case=%d (%s) c=%s want=%s"
                  % (cid, name, got[1], expected))
    check(mismatches == 0, "replay_differential")
    print("d8_rev3_differential cases=%d mismatches=%d" %
          (len(all_cases), mismatches))

    # 11. Revision-3 issued order: document == C simulator.
    tokens = parse_sequence_block(PREREG)
    check(len(tokens) > 0, "rev3_sequence_block_present")
    sim = os.path.join(BUILD, "hmx_d8_sim.exe")
    if os.path.isfile(sim):
        rc, seqs, seq_out = run_c_sequence(sim)
        check(rc == 0, "sim_rev3_sequence_exit")
        for letter in "ABCD":
            check(seqs.get(letter) == expand_sequence(tokens, letter),
                  "sim_rev3_sequence_%s" % letter)
        if any(seqs.get(letter) != expand_sequence(tokens, letter)
               for letter in "ABCD"):
            print(seq_out[-1500:])
    else:
        print("SKIP (no simulator for the issued-order check)")

    # 12. FastRPC failure-state table: every state must be explicit and must
    #     forbid a numeric reading.
    states = []
    for line in marked_block(FAILURE, "d8-failure-states"):
        parts = line.split()
        if len(parts) == 6:
            states.append(parts)
    check(len(states) == 7, "failure_state_count")
    check([s[0] for s in states] == ["S0", "S1", "S2", "S3", "S4", "S5", "S6"],
          "failure_state_ids")
    check(all(s[4].startswith("D8_") for s in states), "failure_state_classes")
    check(all(s[5] == "none" for s in states), "failure_state_no_numeric")
    check(all(s[2] in ("0", "nonzero", "14001", "unknown") for s in states),
          "failure_state_return_kinds")
    ftext = open(FAILURE, encoding="utf-8").read()
    check("return 0" in ftext and "metadata[30]" in ftext,
          "failure_return_zero_rule")
    check("mode-46-only exception" in ftext, "failure_cleanup_exception")
    check("D8_EVIDENCE_LOST" in ftext, "failure_evidence_lost")

    # 13. Host decision order simulation (S0..S6). This is a table test, not a
    #     transport test: no RPC is made.
    def host_decide(rpc_status, have_meta, cleanup=0, failure=0,
                    guard_ok=True, slots_ok=True):
        preexec = (14001, 14002, 14003, 14004, 14006, 14010)
        if rpc_status != 0:
            return ("D8_CONTRACT_FAILED_PREEXEC" if rpc_status in preexec
                    else "D8_EVIDENCE_LOST")
        if not have_meta:
            return "D8_EVIDENCE_LOST"
        if not slots_ok:
            return "D8_CONTRACT_FAILED"
        if cleanup != 0:
            return "D8_CONTRACT_FAILED_CLEANUP"
        if failure != 0:
            return "D8_CONTRACT_FAILED"
        if not guard_ok:
            return "D8_CONTRACT_FAILED"
        return "NUMERIC"

    decisions = [
        ((14001, False), "D8_CONTRACT_FAILED_PREEXEC"),
        ((14006, False), "D8_CONTRACT_FAILED_PREEXEC"),
        ((14010, False), "D8_CONTRACT_FAILED_PREEXEC"),
        ((14008, True), "D8_EVIDENCE_LOST"),
        ((0, False), "D8_EVIDENCE_LOST"),
        ((0, True), "NUMERIC"),
    ]
    for args, want in decisions:
        check(host_decide(*args) == want, "decision_%s_%s" % args)
    check(host_decide(0, True, cleanup=14008) == "D8_CONTRACT_FAILED_CLEANUP",
          "decision_cleanup")
    check(host_decide(0, True, failure=8) == "D8_CONTRACT_FAILED",
          "decision_failure_code")
    check(host_decide(0, True, guard_ok=False) == "D8_CONTRACT_FAILED",
          "decision_guard")
    check(host_decide(0, True, slots_ok=False) == "D8_CONTRACT_FAILED",
          "decision_slots")

    # 14. Failure-reason namespace and ownership text.
    fail_codes = sorted(v for k, v in defines.items()
                        if k.startswith("HMX_D8_FAIL_"))
    check(fail_codes == list(range(21)), "failure_reason_codes_0_20")
    check(len(set(fail_codes)) == 21, "failure_reason_unique")
    ctext = open(CONTRACT, encoding="utf-8").read()
    check("0..8" in ctext and "9..31" in ctext, "ownership_text_0_8_9_31")
    ptext = open(PREREG, encoding="utf-8").read()
    check(re.search(r"Mode 45", ptext) is not None and
          "bias set 2" in ptext, "mode45_overlap_documented")

    print("d8_rev3_device_status=NOT_RUN cpu_labels_only=1")
    print("checks=%d fail=%d" % (CHECKS["n"], CHECKS["fail"]))
    print("verdict=%s" % ("D8_REV3_INDEPENDENT_AUDIT_PASS"
                          if CHECKS["fail"] == 0
                          else "D8_REV3_INDEPENDENT_AUDIT_FAIL"))
    return 0 if CHECKS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
