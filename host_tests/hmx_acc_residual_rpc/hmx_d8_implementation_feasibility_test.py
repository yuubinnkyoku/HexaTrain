#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
"""D8 implementation-feasibility static audit (CPU only).

This verifier answers a different question from hmx_d8_independent_test.py:

  * the existing test proves the D8 *numbers*, classifier and registered event
    order against the CPU model;
  * this audit checks whether the registered revision-2 experiment can be
    carried by the *existing* DSP/host interface (probe_dsp.c, probe_host.c,
    hexatrain_hmx_acc_residual.idl), and records every pre-implementation gap
    that the source and the documents can establish without a DSP build.

It does not build DSP code, contact a device, run FastRPC, or change a
registered value. It reads source and documentation only.

Exit 0: the audit ran and every encoded invariant still holds.
Exit 1: an encoded invariant is violated (layout / mask / sequence / interface
        drift). The audit document and this verifier must then be updated.
"""

import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RPC = os.path.join(ROOT, "host_tests", "hmx_acc_residual_rpc")
PREREG = os.path.join(ROOT, "docs", "hmx-v81-cand-b-d8-device-prereg.md")
CONTRACT = os.path.join(ROOT, "docs", "hmx-v81-cand-b-d8-evidence-contract.md")
PROBE_DSP = os.path.join(RPC, "probe_dsp.c")
PROBE_HOST = os.path.join(RPC, "probe_host.c")
IDL = os.path.join(RPC, "hexatrain_hmx_acc_residual.idl")
MODE45_H = os.path.join(RPC, "hmx_mode45.h")
B3R2_H = os.path.join(RPC, "hmx_b3_r2.h")
IDENTITY_H = os.path.join(RPC, "hmx_identity.h")
D8_SIM = os.path.join(RPC, "hmx_d8_min_sim.c")
DISASM_PS1 = os.path.join(ROOT, "scripts", "audit_hmx_mode45_disasm.ps1")

# Registered revision-2 constants. These are the values the audit checks; it
# never rewrites them.
CELLS = 1024
T8 = 32
TILE_BYTES = 2048
VTCM_BYTES = 12288
BIAS_BYTES = 256
VECTOR_BYTES = 128
VECTORS_PER_TILE = TILE_BYTES // VECTOR_BYTES  # 16
REG = {
    "AHI": 0,
    "ALO": 2048,
    "WHI": 4096,
    "WLO": 6144,
    "OUT": 8192,
    "BIAS0": 10240,
    "BIAS1": 10496,
}
USED = 10752
TAIL = VTCM_BYTES - USED  # 1536
MASK = {"A": 1, "B": 7, "C": 3, "D": 5}
FILL_WORD = 0xA5A5
FNV = {
    "ahi": 0xC6231266,
    "alo": 0x3B2D22D2,
    "whi": 0x8B88609F,
    "wlo": 0xA66AF695,
}

CHECKS = {"n": 0, "fail": 0, "failures": []}
FINDINGS = []


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def check(cond, tag, detail=""):
    CHECKS["n"] += 1
    if not cond:
        CHECKS["fail"] += 1
        CHECKS["failures"].append(tag)
        print("FAIL %s%s" % (tag, (" " + detail) if detail else ""))


def finding(fid, klass, text):
    FINDINGS.append((fid, klass, text))


def read_marked(text, marker):
    begin = "<!-- %s:begin -->" % marker
    end = "<!-- %s:end -->" % marker
    if begin not in text or end not in text:
        return []
    body = text.split(begin, 1)[1].split(end, 1)[0]
    return [line.strip() for line in body.splitlines()]


def parse_mask_block(text):
    out = {}
    for line in read_marked(text, "d8-fill-mask"):
        parts = line.split()
        if len(parts) == 2 and parts[0] in "ABCD":
            out[parts[0]] = int(parts[1])
    return out


def parse_pre_reg_mask_table(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*\*\*([ABCD])\*\*\s*\|[^|]*\|\s*`(\d+)`\s*\|", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_layout_table(text):
    """Parse the section-0 VTCM map: | `+0` | `Ahi` | `out_recon` | MMA #2 |."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*`\+(\d+)`\s*\|([^|]*)\|([^|]*)\|([^|]*)\|", line)
        if m:
            out[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip(),
                                    m.group(4).strip())
    return out


def parse_sequence_block(text):
    tokens = []
    for line in read_marked(text, "d8-sequence"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "PACK" and len(parts) == 2:
            tokens.append(("PACK", parts[1], None))
        elif parts[0] == "BIAS" and len(parts) == 2:
            tokens.append(("BIAS", parts[1], None))
        elif parts[0] == "MMA" and len(parts) == 4:
            tokens.append(("MMA", parts[1], (parts[2], parts[3])))
        elif parts[0] == "FILL" and len(parts) >= 2:
            tokens.append(("FILL", parts[1], tuple(parts[2:])))
        elif parts[0] == "CONVERT" and len(parts) == 2:
            tokens.append(("CONVERT", parts[1], None))
        elif parts[0] == "STORE" and len(parts) == 2:
            tokens.append(("STORE", parts[1], None))
        else:
            tokens.append(("UNKNOWN", line, None))
    return tokens


def parse_metadata_table(text):
    """Return {word_index: (content, expected_text)} and ranges covered."""
    start = text.index("### 3.2 Metadata words")
    end = text.index("Word 15 cannot distinguish", start)
    rows = {}
    for line in text[start:end].splitlines():
        m = re.match(r"\|\s*(\d+)(?:\s*[\u2013-]\s*(\d+))?\s*\|([^|]*)\|([^|]*)\|",
                     line)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        for idx in range(lo, hi + 1):
            rows[idx] = (m.group(3).strip(), m.group(4).strip())
    return rows


def fp16_is_normal_negative(word):
    sign = (word >> 15) & 1
    exp = (word >> 10) & 0x1F
    frac = word & 0x3FF
    return sign == 1 and 0 < exp < 31 and (exp != 0 or frac != 0)


def fnv1a32(data):
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def fp16_to_f64(bits):
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        return sign * (frac / 1024.0) * 2.0 ** -14
    if exp == 31:
        return sign * float("inf") if frac == 0 else float("nan")
    return sign * (1.0 + frac / 1024.0) * 2.0 ** (exp - 15)


def f64_to_fp16_rn(value):
    """Round-to-nearest-even binary64 -> binary16 (same law as the CPU model)."""
    if value != value:
        return 0x7E00
    sign = 0x8000 if (value < 0 or (value == 0 and str(value)[0] == "-")) else 0
    a = abs(value)
    if a == 0.0:
        return sign
    if a >= 65520.0:
        return sign | 0x7C00
    if a < 2.0 ** -14:
        # subnormal: value = m * 2^-24
        m = int(round(a * 2.0 ** 24))
        if m >= 1024:
            return sign | 0x0400
        return sign | m
    exp = 0
    norm = a
    while norm >= 2.0:
        norm /= 2.0
        exp += 1
    while norm < 1.0:
        norm *= 2.0
        exp -= 1
    e = exp + 15
    if e >= 31:
        return sign | 0x7C00
    mant = norm - 1.0
    m = int(round(mant * 1024.0))
    if m == 1024:
        m = 0
        e += 1
        if e >= 31:
            return sign | 0x7C00
    return sign | (e << 10) | m


def pack_tile(src, weight):
    dst = [0] * CELLS
    for a in range(T8):
        for i in range(T8):
            if weight:
                idx = ((i >> 1) * 64) + (a * 2) + (i & 1)
            else:
                idx = ((a >> 1) * 64) + (i * 2) + (a & 1)
            dst[idx] = src[a * T8 + i]
    return dst


def bytes_u16_le(arr):
    return b"".join(struct.pack("<H", v) for v in arr)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def split_lo_float(v, hi_bits):
    xf = f32(v)
    hf = f32(fp16_to_f64(hi_bits))
    return f64_to_fp16_rn(f32(xf - hf))


def mode42_generator_tiles():
    """Re-derive the mode-42 DSP tile generator and return the four tiles.

    probe_dsp.c's e3c1 generator (mode 42, B3 trial) draws W then X from seed
    0xE3C0117E5EEDF00D, applies the same anchor rows/lanes as the registered
    G8/deploy fixture, and splits hi/lo with float32 arithmetic. If this
    re-derivation reproduces the registered packed-tile FNVs, the existing
    DSP source already contains a bit-exact generator for the D8 operands.
    """
    state = 0xE3C0117E5EEDF00D
    mask64 = (1 << 64) - 1
    x = [0.0] * CELLS
    w = [0.0] * CELLS
    for i in range(CELLS):
        state ^= (state << 13) & mask64
        state ^= state >> 7
        state ^= (state << 17) & mask64
        state &= mask64
        w[i] = 4.0 * ((state >> 11) * (1.0 / 9007199254740992.0)) - 2.0
        state ^= (state << 13) & mask64
        state ^= state >> 7
        state ^= (state << 17) & mask64
        state &= mask64
        x[i] = 4.0 * ((state >> 11) * (1.0 / 9007199254740992.0)) - 2.0
    for o in range(T8):
        if o <= 3 or o >= 28:
            for i in range(T8):
                w[o * T8 + i] = 0.0
    w[0 * T8 + 0] = 1.0 + 2.0 ** -11
    w[0 * T8 + 1] = 2.0 ** -11
    w[0 * T8 + 2] = 2.0 ** -13
    w[1 * T8 + 0] = 1.0 - 2.0 ** -12
    w[2 * T8 + 0] = 2.0 ** -9
    w[3 * T8 + 0] = 1.0
    w[3 * T8 + 1] = 1.0 + 2.0 ** -12
    w[28 * T8 + 0] = 2.0 ** -10
    w[29 * T8 + 0] = 2.0 ** -10 - 2.0 ** -21
    w[30 * T8 + 0] = 3.0 * 2.0 ** -12
    w[31 * T8 + 0] = 2.0 ** -13
    for s in range(4):
        for i in range(T8):
            x[s * T8 + i] = 0.0
    x[0 * T8 + 0] = 1.0 + 2.0 ** -11
    x[0 * T8 + 1] = 2.0 ** -11
    x[0 * T8 + 2] = 2.0 ** -13
    for i in range(T8):
        x[1 * T8 + i] = -x[i]
    x[2 * T8 + 0] = 1.0 - 2.0 ** -12
    x[2 * T8 + 1] = -(1.0 - 2.0 ** -12)
    x[3 * T8 + 0] = 2.0 ** -10
    ahi = [f64_to_fp16_rn(v) for v in x]
    whi = [f64_to_fp16_rn(v) for v in w]
    alo = [split_lo_float(x[i], ahi[i]) for i in range(CELLS)]
    wlo = [split_lo_float(w[i], whi[i]) for i in range(CELLS)]
    return ahi, alo, whi, wlo


def main():
    prereg = read(PREREG)
    contract = read(CONTRACT)
    dsp = read(PROBE_DSP)
    host = read(PROBE_HOST)
    idl = read(IDL)
    mode45 = read(MODE45_H)
    b3r2 = read(B3R2_H)
    identity = read(IDENTITY_H)
    sim = read(D8_SIM)

    print("d8_feas_scope=static_only device=NOT_RUN dsp_build=NOT_RUN")
    print("d8_feas_sources=prereg,contract,probe_dsp,probe_host,idl,mode45,b3r2,identity,d8_sim,disasm_script")

    # ------------------------------------------------------------------ A. layout
    layout = parse_layout_table(prereg)
    check(set(layout) == set(REG.values()), "layout_table_offsets",
          "got %s" % sorted(layout))
    expected_use = {
        0: "Ahi", 2048: "Alo", 4096: "Whi", 6144: "Wlo",
        8192: "out_pos", 10240: "bias set 0", 10496: "bias set 1",
    }
    for off, name in expected_use.items():
        cells = " ".join(layout.get(off, ("", "", ""))).lower()
        check(off in layout and name.lower() in cells,
              "layout_table_%d" % off, str(layout.get(off)))

    def define_int(src, name):
        m = re.search(r"#define\s+" + name + r"\s+(-?0x[0-9a-fA-F]+|\d+)", src)
        return int(m.group(1), 0) if m else None

    pr = {
        "AHI": define_int(b3r2, "HMX_B3_R2_PR_AHI_OFFSET"),
        "ALO": define_int(b3r2, "HMX_B3_R2_PR_ALO_OFFSET"),
        "WHI": define_int(b3r2, "HMX_B3_R2_PR_WHI_OFFSET"),
        "WLO": define_int(b3r2, "HMX_B3_R2_PR_WLO_OFFSET"),
        "OUT": define_int(b3r2, "HMX_B3_R2_PR_POS_OFFSET"),
        "BIAS0": define_int(b3r2, "HMX_B3_R2_PR_BIAS0_OFFSET"),
        "BIAS1": define_int(b3r2, "HMX_B3_R2_PR_BIAS1_OFFSET"),
    }
    check(pr == REG, "layout_header_offsets", str(pr))
    check("#define HMX_B3_R2_CELLS (HMX_B3_R2_T8 * HMX_B3_R2_T8)" in b3r2 and
          "#define HMX_B3_R2_TILE_BYTES (HMX_B3_R2_CELLS * 2)" in b3r2 and
          "/* 2048 */" in b3r2, "layout_tile_bytes")
    check(define_int(b3r2, "HMX_B3_R2_VTCM_TOTAL") == VTCM_BYTES,
          "layout_vtcm_bytes")
    check(define_int(b3r2, "HMX_B3_R2_PR_MMA_PHASE_USED") == 8704,
          "layout_mma_phase_used")

    # Pairwise disjoint byte ranges.
    ranges = [(off, off + TILE_BYTES) for off in
              (REG["AHI"], REG["ALO"], REG["WHI"], REG["WLO"], REG["OUT"])]
    ranges += [(REG["BIAS0"], REG["BIAS0"] + BIAS_BYTES),
               (REG["BIAS1"], REG["BIAS1"] + BIAS_BYTES)]
    disjoint = True
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a0, a1 = ranges[i]
            b0, b1 = ranges[j]
            if a0 < b1 and b0 < a1:
                disjoint = False
    check(disjoint, "layout_disjoint")
    check(USED == REG["BIAS1"] + BIAS_BYTES, "layout_used_bytes", str(USED))
    check(USED <= VTCM_BYTES, "layout_fits")
    check(TAIL == 1536 and TAIL % VECTOR_BYTES == 0, "layout_tail_1536")
    check(all(off % TILE_BYTES == 0 for off in
              (REG["AHI"], REG["ALO"], REG["WHI"], REG["WLO"], REG["OUT"])),
          "layout_tile_alignment_2048")
    check(REG["BIAS0"] % 256 == 0 and REG["BIAS1"] % 256 == 0,
          "layout_bias_alignment_256")
    check(VECTOR_BYTES == 128 and TILE_BYTES // VECTOR_BYTES == 16,
          "layout_vector_granularity")
    check("VTCM_BYTES = 12288" in dsp and "TILE_BYTES = 2048" in dsp and
          "BIAS_BYTES = 256" in dsp, "layout_dsp_enum")
    check("(uintptr_t)vtcm & (TILE_BYTES - 1u)" in dsp,
          "layout_dsp_pointer_alignment_2048")

    # Mode 45's third bias row starts exactly at the D8 guard tail.
    bias2 = define_int(mode45, "HMX_MODE45_BIAS2_OFFSET")
    check(bias2 == 10752, "layout_mode45_bias2_offset", str(bias2))
    check(bias2 >= USED and bias2 + BIAS_BYTES <= VTCM_BYTES,
          "layout_mode45_bias2_in_d8_tail")
    finding("mode45_bias2_tail", "IMPLEMENTATION_CHANGE_REQUIRED",
            "mode 45 writes bias set 2 at +10752..+11008, which is inside the "
            "D8 1,536 B guard tail (+10752..+12288); copying the mode-45 "
            "three-bias prologue would violate D8 contract section 2 item 7")

    # ------------------------------------------------------------- B. fill ledger
    fills = {
        "out": (REG["OUT"], TILE_BYTES),
        "wlo": (REG["WLO"], TILE_BYTES),
        "ahi": (REG["AHI"], TILE_BYTES),
    }
    for name, (off, size) in fills.items():
        check(size == VECTORS_PER_TILE * VECTOR_BYTES,
              "fill_%s_16x128" % name)
        check(off % VECTOR_BYTES == 0, "fill_%s_128_aligned" % name)
        check(off + size <= USED, "fill_%s_inside_used" % name)
        check(off + size <= VTCM_BYTES, "fill_%s_inside_vtcm" % name)
    check(sum(s for _, s in fills.values()) == 3 * TILE_BYTES,
          "fill_total_bytes")
    check(len(fills) * VECTORS_PER_TILE == 48, "fill_total_vectors")
    # The three fill ranges must not overlap each other.
    fr = [(off, off + size) for off, size in fills.values()]
    overlap = any(fr[i][0] < fr[j][1] and fr[j][0] < fr[i][1]
                  for i in range(3) for j in range(i + 1, 3))
    check(not overlap, "fill_ranges_disjoint")

    doc_mask = parse_mask_block(contract)
    pre_mask = parse_pre_reg_mask_table(prereg)
    check(doc_mask == MASK, "mask_contract_block", str(doc_mask))
    check(pre_mask == MASK, "mask_prereg_table", str(pre_mask))

    # Derive the mask from the registered per-condition fills.
    ledger = {
        "A": ["out"],
        "B": ["out", "wlo", "ahi"],
        "C": ["out", "wlo"],
        "D": ["out", "ahi"],
    }
    derived = {}
    for cond, names in ledger.items():
        m = 0
        for name in names:
            m |= {"out": 1, "wlo": 2, "ahi": 4}[name]
        derived[cond] = m
        check(len(names) == len(set(names)), "ledger_%s_unique" % cond)
    check(derived == MASK, "mask_derived_from_ledger", str(derived))
    check(ledger["A"] == ["out"], "ledger_A_out_only")
    check(ledger["B"] == ["out", "wlo", "ahi"], "ledger_B_all")
    check(ledger["C"] == ["out", "wlo"], "ledger_C_wlo")
    check(ledger["D"] == ["out", "ahi"], "ledger_D_ahi")
    check(fp16_is_normal_negative(FILL_WORD), "fill_word_normal_negative_fp16")
    check((FILL_WORD >> 8) == 0xA5 and (FILL_WORD & 0xFF) == 0xA5,
          "fill_word_bytes_a5")
    check("HMX_B3_R2_SENTINEL_BYTE 0xA5u" in b3r2, "fill_word_header_byte")

    # -------------------------------------------------------------- C. sequence
    tokens = parse_sequence_block(prereg)
    expected_tokens = [
        ("PACK", "AHI", None), ("PACK", "ALO", None),
        ("PACK", "WHI", None), ("PACK", "WLO", None),
        ("BIAS", "0", None), ("BIAS", "1", None),
        ("MMA", "1", ("AHI", "WHI")), ("MMA", "2", ("AHI", "WLO")),
        ("MMA", "3", ("ALO", "WHI")),
        ("FILL", "OUT", ()), ("FILL", "WLO", ("B", "C")),
        ("FILL", "AHI", ("B", "D")),
        ("CONVERT", "1", None), ("STORE", "OUT", None),
    ]
    check(tokens == expected_tokens, "sequence_block_tokens",
          str(tokens))
    check(not any(k == "UNKNOWN" for k, _, _ in tokens),
          "sequence_block_no_unknown")

    # Expands to the per-condition operation list the DSP must emit.
    for cond in "ABCD":
        got = []
        for kind, a, b in tokens:
            if kind == "FILL":
                conds = b or ("A", "B", "C", "D")
                if cond in conds:
                    got.append("FILL:%s" % a)
            else:
                got.append(kind)
        want_fills = {"A": ["FILL:OUT"],
                      "B": ["FILL:OUT", "FILL:WLO", "FILL:AHI"],
                      "C": ["FILL:OUT", "FILL:WLO"],
                      "D": ["FILL:OUT", "FILL:AHI"]}[cond]
        check([g for g in got if g.startswith("FILL:")] == want_fills,
              "sequence_%s_fills" % cond, str(got))

    # Existing DSP template: mode 42's B3 trial uses the registered operand
    # addresses and the three-packet ACC chain. Verify the source order.
    m42 = dsp.index("if (mode == 42)")
    m43 = dsp.index("static void run_mode43_one_trial")
    block42 = dsp[m42:m43]
    tuples = re.findall(
        r'"r"\((act_\w+_reg)\), "r"\(act_rt\), "r"\((wgt_\w+_reg)\), '
        r'"r"\(wgt_rt\)', block42)
    check(tuples[:3] == [("act_hi_reg", "wgt_hi_reg"),
                         ("act_hi_reg", "wgt_lo_reg"),
                         ("act_lo_reg", "wgt_hi_reg")] and
          tuples[3] == ("act_hi_reg", "wgt_hi_reg"),
          "template_mode42_mma_order", str(tuples))
    check(block42.count('"{ activation.hf = mxmem(%0, %1)') >= 4,
          "template_mode42_packet_forms")
    offsets42 = re.findall(
        r"uint16_t \*(act_hi|act_lo|wgt_hi|wgt_lo) = "
        r"\(uint16_t \*\)\(base \+ (\d+) \* TILE_BYTES\)", block42)
    check(dict(offsets42) == {"act_hi": "0", "act_lo": "1",
                              "wgt_hi": "2", "wgt_lo": "3"},
          "template_mode42_tile_offsets", str(offsets42))

    # Bias position: mode 42 loads bias after MMA #3; mode 45 before mxclracc.
    m42_mma = block42.rindex('"{ activation.hf = mxmem(%0, %1)')
    m42_bias = block42.index('bias = mxmem2', m42_mma)
    check(m42_bias > m42_mma, "template_mode42_bias_after_mma")
    m45 = dsp.index("static int run_mode45_one_trial")
    block45 = dsp[m45:dsp.index("int hexatrain_hmx_acc_residual_open")]
    check(block45.index('bias = mxmem2') < block45.index("mxclracc.hf"),
          "template_mode45_bias_before_mxclracc")
    finding("bias_position_template", "IMPLEMENTATION_CHANGE_REQUIRED",
            "the nearest 3-MMA template (mode 42) loads the bias sets after "
            "MMA #3, while D8 revision 2 registers BIAS 0/1 before mxclracc "
            "and before MMA #1; the target order is the mode-45 prologue, so "
            "the mode-42 block must not be copied verbatim")

    check("asm volatile(\"\" ::: \"memory\")" in dsp,
          "template_compiler_barrier_present")
    for banned in ("syncht", "dmsyncht", "dmtlbsynch", "mxswapacc"):
        check(banned not in dsp, "template_no_%s" % banned)
    check("qurt_hvx_lock(QURT_HVX_MODE_128B)" in dsp, "template_hvx_lock")
    check("for (int v = 0; v < 16; ++v)" in dsp, "template_hvx_16_loop")
    check("*(HVX_Vector *)" in dsp, "template_hvx_vector_store")

    # The D8 CPU trace records one 2048 B fill event; it cannot prove the
    # registered 16 x 128 B machine form.
    check("pr_fill(st, D8_OUT_OFFSET, HMX_B3_R2_TILE_BYTES" in sim,
          "model_fill_is_range_event")
    check(not any(("fill" in line and "128" in line) for line in sim.splitlines()),
          "model_fill_has_no_128b_events")
    finding("fill_form_unproven", "INSUFFICIENT_EVIDENCE",
            "the CPU model records each fill as one 2,048 B event; the "
            "registered 16 x 128 B HVX vector-store form is a future "
            "disassembly property, and a compiler-generated loop/memset would "
            "not be the registered experiment")

    # ----------------------------------------------------------- D. ABI/evidence
    check("long run(rout sequence<long> results, rout sequence<long> metadata, "
          "in long q, in long mode);" in idl, "abi_idl_signature")
    check(idl.count("rout sequence<long>") == 2, "abi_idl_two_rout_sequences")
    check("int *results, int resultsLen" in dsp and
          "int *metadata, int metadataLen" in dsp,
          "abi_dsp_length_parameters")
    check("resultsLen < 4" in dsp and
          "resultsLen < HMX_MODE45_RESULT_SLOTS" in dsp and
          "metadataLen < 9" in dsp and
          "metadataLen < HMX_MODE45_METADATA_LEN" in dsp,
          "abi_dsp_min_length_gates")
    check(define_int(mode45, "HMX_MODE45_RESULT_SLOTS") == 11 and
          define_int(mode45, "HMX_MODE45_METADATA_LEN") == 32,
          "abi_mode45_sizes")
    result_bytes = CELLS * 4
    metadata_bytes = 32 * 4
    check(result_bytes == 4096, "abi_results_4096")
    check(metadata_bytes == 128, "abi_metadata_128")
    check(result_bytes + metadata_bytes == 4224, "abi_total_4224")
    check("HMX_MODE45_RESULT_SLOTS * sizeof(int)" in host and
          "HMX_MODE45_METADATA_LEN * sizeof(int)" in host,
          "abi_host_current_allocation")
    check("(mode == HMX_MODE45_MODE)" in host and
          "? HMX_MODE45_RESULT_SLOTS" in host,
          "abi_host_current_result_len")
    finding("results_length_unobservable", "IMPLEMENTATION_CHANGE_REQUIRED",
            "rout sequence<long> maps to (int* results, int resultsLen): the "
            "client fixes the maximum and the callee cannot return a different "
            "length, so evidence-contract section 3.3's 'short or long return' "
            "is not an observable; a DSP-side 'readbacks' word would be a "
            "self-report, and the host length is a host allocation fact")

    # q selector baseline.
    check("if (q < 6 || q > 10 || mode < 0 || mode > MODE_MAX)" in dsp,
          "abi_dsp_q_gate_6_10")
    check("MODE_MAX = 45" in dsp, "abi_dsp_mode_max_45")
    check("if (q < 6 || q > 10 || mode < 0 || mode > HMX_PROBE_MODE_MAX)" in host,
          "abi_host_q_gate_6_10")
    check("HMX_PROBE_MODE_MAX = 45" in host, "abi_host_mode_max_45")
    check("!report->q_ok" in identity and
          "report.q_ok = (identity[1] == q)" in identity,
          "abi_identity_q_echo")
    check("HMX_IDENTITY_FAILURE_Q = 3" in identity, "abi_identity_q_failure")
    host_flat = re.sub(r"\s+", " ", host)
    check("handle, identity, 4, meta, 9, q, HMX_IDENTITY_MODE" in host_flat,
          "abi_identity_uses_same_q")
    finding("q_selector_encoding", "IMPLEMENTATION_CHANGE_REQUIRED",
            "the contract proposes q=0..3, but probe_dsp.c and probe_host.c "
            "both reject q<6, and the identity pre-check runs with the same q; "
            "q=6..9 with condition=q-6 keeps the gate, the identity flow and "
            "the word-3 echo (0..3), but the selector encoding must be "
            "registered before implementation")

    # Metadata table completeness and values.
    meta = parse_metadata_table(contract)
    check(sorted(meta) == list(range(32)), "metadata_table_covers_32",
          str(sorted(meta)))
    check(len(meta) == 32, "metadata_table_unique")
    for word, tag in ((0, "build_id"), (1, "arch"), (2, "mode"), (3, "cond"),
                      (4, "mxclracc"), (5, "mma"), (6, "convert"),
                      (7, "store"), (8, "packs"), (9, "bias"),
                      (10, "mask"), (11, "fill_word"), (12, "store_target"),
                      (13, "out_pos"), (14, "wlo_fill"), (15, "ahi_fill"),
                      (16, "trace"), (21, "readbacks"), (22, "cleanup"),
                      (23, "failure"), (24, "build id")):
        check(word in meta, "metadata_word_%02d_present" % word)
        check(tag.replace("_", " ") in meta.get(word, ("", ""))[0].lower() or
              tag in meta.get(word, ("", ""))[0].lower(),
              "metadata_word_%02d_content" % word, str(meta.get(word)))
    check("A=1, B=7, C=3, D=5" in meta[10][1], "metadata_word10_masks")
    check("0x0000A5A5" in meta[11][1], "metadata_word11_fill_word")
    check("8192" in meta[12][1] and "8192" in meta[13][1],
          "metadata_word12_13_8192")
    check("6144" in meta[14][1], "metadata_word14_6144")
    check(meta[15][1].startswith("0;"), "metadata_word15_0")
    check(meta[16][1].startswith("0"), "metadata_word16_0")
    for word, name in ((17, "c6231266"), (18, "3b2d22d2"),
                       (19, "8b88609f"), (20, "a66af695")):
        check(name in meta[word][1], "metadata_word%02d_fnv" % word)
    check("same as word 0" in meta[24][1], "metadata_word24_build_id2")
    check(all(meta[w][1] == "0" for w in range(25, 32)),
          "metadata_reserved_zero")

    # Per-condition expected words are internally consistent with the mask.
    for cond, cond_id in (("A", 0), ("B", 1), ("C", 2), ("D", 3)):
        mask = MASK[cond]
        wlo = 6144 if (mask & 2) else 0
        check(wlo == (6144 if cond in "BC" else 0),
              "metadata_%s_wlo_offset" % cond)
        check(bool(mask & 4) == (cond in "BD"),
              "metadata_%s_ahi_presence" % cond)
        check(bool(mask & 1), "metadata_%s_out_always" % cond)
        check(cond_id in range(4), "metadata_%s_cond_echo" % cond)

    # Metadata ownership: run() writes [0..8] before dispatch and [3]/[7]/[8]
    # after the handler, so the contract's words 3/7/8 are not handler-owned.
    s45 = dsp.index("const int s45 = run_mode45_one_trial")
    w3 = dsp.index("metadata[3] = elapsed", s45)
    w7 = dsp.index("metadata[7] = 1", s45)
    w8 = dsp.index("metadata[8] = HMX_MODE45_TRIALS", s45)
    check(w3 > s45 and w7 > s45 and w8 > s45, "abi_metadata_post_dispatch")
    check("metadata[0] = HMX_MAGIC;" in dsp, "abi_metadata0_magic")
    check("metadata[2] = VTCM_BYTES;" in dsp, "abi_metadata2_vtcm")
    finding("metadata_ownership_collision", "IMPLEMENTATION_CHANGE_REQUIRED",
            "the D8 table reuses words 0..8 (build id/arch/mode/condition/"
            "counts/mask...), but run() writes magic at 0, VTCM bytes at 2 and "
            "overwrites 3 (elapsed) and 7/8 (hmx flag/trials) after the "
            "handler returns; the D8 evidence must be renumbered into 9..31 "
            "(23 free words, 25 fields -> drop the derivable Ahi offset and "
            "the second build id) or run() must be made mode-46 aware")

    # Build identity.
    check("-DHMX_PROBE_BUILD_ID" in host or "-DHMX_PROBE_BUILD_ID" in read(
        os.path.join(ROOT, "host_tests", "hmx_acc_residual_rpc",
                     "run_experiment.ps1")), "abi_build_id_stamp")
    check("hmx_identity_strict_build_id_ok" in identity and
          "-strict-build-id" in host, "abi_strict_build_id")
    check("HMX_PROBE_BUILD_ID" in dsp, "abi_dsp_build_id_echo")
    check("// 0 means \"unknown / legacy build\"" in identity or
          "UNAVAILABLE" in identity, "abi_build_id_unavailable_semantics")

    # Error-path evidence. QAIC discards rout data on a nonzero return; the
    # host reads the status before any slot, so a nonzero status leaves no raw
    # data to preserve.
    check("rpc_status=%d magic=0x%x" in host, "abi_host_status_before_slots")
    finding("error_path_raw_loss", "IMPLEMENTATION_CHANGE_REQUIRED",
            "FastRPC/QAIC does not propagate rout data when the method returns "
            "non-zero; the D8 contract asks to preserve raw bytes on failure, "
            "so the run must return 0 after execution starts and carry the "
            "first-failure reason in metadata, reserving non-zero for "
            "pre-execution acquisition failures")

    # ---------------------------------------------------- E. environment facts
    check("if(NOT HEXAGON_ARCH STREQUAL \"v81\")" in read(
        os.path.join(RPC, "CMakeLists.txt")), "env_cmake_v81_guard")
    check("-mhmx" in read(os.path.join(RPC, "CMakeLists.txt")), "env_mhmx")
    sdk = os.environ.get("HEXAGON_SDK_ROOT", "")
    check(not sdk or os.path.isdir(sdk), "env_sdk_path_if_set")
    finding("dsp_build_env_absent", "INSUFFICIENT_EVIDENCE",
            "no Hexagon SDK / hexagon-llvm-objdump / built skel .so is present "
            "in this checkout, so no DSP build, object inspection or "
            "disassembly can be performed in this audit")

    disasm = read(DISASM_PS1)
    check("activation.hf = mxmem" in disasm and "mxclracc" in disasm and
          "cvt.hf = acc" in disasm, "disasm_template_mnemonics")
    check("vmem" not in disasm and "FILL" not in disasm and
          "ExpectedFill" not in disasm,
          "disasm_template_has_no_fill_audit")
    finding("disasm_audit_fill_gap", "INSUFFICIENT_EVIDENCE",
            "audit_hmx_mode45_disasm.ps1 counts mnemonics and a fixed order "
            "but does not parse HVX store addresses, fill extents, branch "
            "targets or the CFG; a D8 audit must add fill-address/extent and "
            "per-condition path checks before it can accept a future object")

    # Mode 46 is unregistered and unauthorised.
    check("MODE_MAX = 45" in dsp and "HMX_PROBE_MODE_MAX = 45" in host,
          "mode_max_45")
    check("mode == 46" not in dsp and "HMX_MODE46" not in dsp and
          "HMX_MODE46" not in host, "mode46_absent")
    check("NOT_RUN" in prereg and "NOT GRANTED" in prereg,
          "mode46_status_not_run")
    finding("mode46_unregistered", "BLOCKED",
            "the registered experiment has no trigger mode; Mode 46 remains "
            "unregistered/NOT_RUN and the pre-registration itself lists a "
            "registered mode as a prerequisite (section 11 item 7)")

    # Fixture generation: the mode-42 DSP generator reproduces the registered
    # packed-tile FNVs exactly (same seed/anchors/split).
    gen = mode42_generator_tiles()
    gen_fnv = {}
    for name, arr, weight in (("ahi", gen[0], 0), ("alo", gen[1], 0),
                              ("whi", gen[2], 1), ("wlo", gen[3], 1)):
        gen_fnv[name] = fnv1a32(bytes_u16_le(pack_tile(arr, weight)))
    check(gen_fnv == FNV, "fixture_mode42_generator_matches_registered_fnv",
          str({k: "%08x" % v for k, v in gen_fnv.items()}))
    check("0xE3C0117E5EEDF00Dull" in dsp, "fixture_seed_in_dsp")
    check("act[((row >> 1) * 64) + col * 2 + (row & 1)]" in dsp,
          "fixture_prm_pack_loop_in_dsp")
    finding("fixture_packing_new", "IMPLEMENTATION_CHANGE_REQUIRED",
            "the DSP already generates the registered operand values (mode 42 "
            "seed/anchors/split) and mode 43 has the PRM pack loop, but no "
            "existing mode writes the packed lo tiles (+2048/+6144); the D8 "
            "build must add that composition and keep the four runtime tile "
            "FNVs")

    # The Candidate B full-tile model computes its operand FNVs after MMA #3;
    # copied into the device path that would be a read inside the critical
    # interval (contract section 2 item 6).
    check("crc_ahi = fnv1a32(st->vtcm.raw + HMX_B3_R2_PR_AHI_OFFSET" in
          read(os.path.join(RPC, "hmx_b3_r2_full_tile_sim.c")),
          "model_fnv_before_fill_but_after_mma")
    finding("fnv_timing", "IMPLEMENTATION_CHANGE_REQUIRED",
            "Candidate B's model hashes the four operands after MMA #3 and "
            "before the fills; that is CPU instrumentation. On the device the "
            "FNV echo (words 17..20) must be taken before MMA #1, otherwise it "
            "inserts 8 KiB of reads into the registered critical interval")

    # Guard tail: no evidence channel in the 32-word table.
    check(not any("tail" in meta[w][0].lower() or "guard" in meta[w][0].lower()
                  for w in meta), "metadata_no_tail_word")
    finding("guard_tail_no_channel", "IMPLEMENTATION_CHANGE_REQUIRED",
            "contract section 2 item 7 requires the 1,536 B tail to stay 0x00, "
            "but no metadata word carries a tail/guard status and VTCM "
            "zero-initialisation is not established; the 32 words are also "
            "fully assigned, so the guard needs an explicit channel and a "
            "pre-run baseline")

    # -------------------------------------------------------------- F. verdict
    print("d8_feas_layout=%s" % ("PASS" if not any(
        t.startswith("layout") for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_fill=%s" % ("PASS" if not any(
        t.startswith("fill") for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_mask=%s" % ("PASS" if not any(
        t.startswith("mask") or t.startswith("ledger")
        for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_sequence=%s" % ("PASS" if not any(
        t.startswith("sequence") or t.startswith("template")
        for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_metadata=%s" % ("PASS" if not any(
        t.startswith("metadata") for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_abi=%s" % ("PASS" if not any(
        t.startswith("abi") for t in CHECKS["failures"]) else "FAIL"))
    print("d8_feas_fixture=%s" % ("PASS" if not any(
        t.startswith("fixture") for t in CHECKS["failures"]) else "FAIL"))

    for fid, klass, text in FINDINGS:
        print("d8_feas_finding=%s %s %s" % (fid, klass, text))
    counts = {}
    for _, klass, _ in FINDINGS:
        counts[klass] = counts.get(klass, 0) + 1
    print("d8_feas_findings=%d %s" % (
        len(FINDINGS),
        " ".join("%s=%d" % (k, counts[k]) for k in sorted(counts))))
    print("d8_feas_checks=%d fail=%d" % (CHECKS["n"], CHECKS["fail"]))
    print("d8_device_status=NOT_RUN dsp_build=NOT_RUN")
    if CHECKS["fail"]:
        print("d8_feas_gate=FAIL")
        return 1
    print("d8_feas_gate=PASS")
    if counts.get("BLOCKED"):
        print("d8_feas_verdict=D8_IMPLEMENTATION_BLOCKED")
    elif counts.get("IMPLEMENTATION_CHANGE_REQUIRED"):
        print("d8_feas_verdict=D8_IMPLEMENTATION_INSUFFICIENT_EVIDENCE")
    else:
        print("d8_feas_verdict=D8_IMPLEMENTATION_STATIC_READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
