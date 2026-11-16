# D8 Disassembly Audit Specification (Revision 3 Proposal)

Status: **`D8_DISASM_RULES_ONLY`**, 2026-11-10. Part of
`D8_REV3_PROPOSED_CPU_ONLY`. This specification and its script authorise
nothing: no DSP source, no DSP build, no QAIRT, no ADB, no device contact, no
Mode 46. No V81 binary exists in this checkout; the rules have been exercised
only on a synthetic text fixture.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md`
- `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md`

## 1. Purpose

The D8 evidence contract relies on two independent kinds of evidence: the
1,024-cell numeric outcome and the static structure of the stamped object. This
specification defines the second. It exists because the numeric outcome alone
cannot distinguish a correct experiment from one that skipped a fill, filled
the wrong address, used a scalar store, wrote the guard, or inserted a
synchronisation that changed the question.

The audit reads **machine code text**, not the C source. A clean source file
proves nothing about the emitted instructions.

## 2. What is audited

| Item | Required |
| --- | --- |
| Symbol | one D8 handler symbol (proposed `run_mode46_one_trial`), one stamped object |
| Build | one clean stamped DSP build, build id recorded, object SHA-256 recorded |
| Text | `hexagon-llvm-objdump -d --disassemble-symbols=<symbol> <skel.so>` output |
| Conditions | the four condition paths in the same object (A/B/C/D) |
| Order | object/instruction order, per condition path |
| Timing | nothing; this is a static audit |

## 3. Input grammar

The auditor accepts the objdump text and ignores lines it does not recognise.
Recognised forms:

```text
  10c0:	{ activation.hf = mxmem(##0) weight.hf = mxmem(##4096) }
  10c4:	{ bias = mxmem2(##10240) }
  10c8:	{ mxclracc.hf }
  10cc:	{ r0 = ##0x0001 }
  10d0:	{ cvt.hf = acc(r0) }
  10d4:	{ mxmem(##8192) = cvt }
  10d8:	{ vmem(##8192) = v0 }
```

Address expressions may be:

- `##N` or `#N` (decimal or `0x` hex);
- `rN+##M` / `rN+#M`, where `rN` was assigned a constant earlier in the same
  symbol;
- `rN`, where `rN` was assigned a constant.

An expression that cannot be resolved is a **failure**, not a skip. The audit
does not guess, and it does not accept a register whose value is unknown.

A packet may hold the activation and the weight on one line, or on two
consecutive lines. Other packet forms must be normalized before the audit; an
unrecognized MMA form fails the count check.

## 4. Required checks

### 4.1 Counts

| Construct | Required |
| --- | ---: |
| `mxclracc.hf` | 1 |
| MMA packets | 3 |
| bias loads (`bias = mxmem2`) | 2 |
| Convert (`cvt.hf = acc`) | 1 |
| HMX Store (`mxmem(...) = cvt`) | 1 |
| operand packs | 16 × 128 B at each of `+0`, `+2048`, `+4096`, `+6144` (64 stores) |
| guard init | 12 × 128 B at `+10752 + k*128` |
| fills | A 16, C/D 32, B 48 (see 4.4) |

### 4.2 Fixed order

1. packs, then bias set 0 (`+10240`), then bias set 1 (`+10496`);
2. guard init, then `mxclracc`;
3. MMA #1 `+0/+4096`, MMA #2 `+0/+6144`, MMA #3 `+2048/+4096`;
4. fills after MMA #3, in the order out_pos → Wlo → Ahi;
5. Convert, then Store at `+8192`;
6. no guard write after the init;
7. no vector/HMX store in the `[mxclracc, Store]` interval other than the
   registered fills.

### 4.3 Convert and Store

- the Convert `Rs` immediate is resolved from an earlier constant assignment
  and must be `0x0001`;
- the Store target must be `+8192`;
- Convert must precede Store;
- every fill must precede Convert.

### 4.4 Fill form and mask

| Condition | Fill blocks (16 × 128 B each) | Mask |
| --- | --- | ---: |
| A | out `+8192` | 1 |
| B | out `+8192`, Wlo `+6144`, Ahi `+0` | 7 |
| C | out `+8192`, Wlo `+6144` | 3 |
| D | out `+8192`, Ahi `+0` | 5 |

Each block must contain exactly 16 stores at `base + k*128`, `k = 0..15`. A
single 2,048 B store, a narrower store (`vN.b`, `vN.h`), a block at the wrong
base, or an extra block fails. The auditor reports the observed count.

### 4.5 Forbidden constructs

Inside `[mxclracc, Store]`:

- `syncht`, `barrier`, `dmwait`, `dmstart`, `dmb`, `isync`;
- any store other than the registered fills;
- any read other than the six MMA operands and the two bias loads;
- a dummy `cvt.hf = acc` is caught by the Convert count (exactly one).

Outside that interval, a guard write after the init, a store above `+12288`,
or an unregistered `mxmem`/`vmem` access also fails.

### 4.6 Fail-closed rules

The audit fails, never skips, on:

- an address expression it cannot resolve;
- an unrecognized MMA/Convert/Store/pack form;
- a count or order mismatch;
- a store narrower than 128 B;
- a guard write after the init;
- a forbidden instruction in the critical interval;
- an unregistered memory access.

## 5. Synthetic test

`scripts/test_hmx_d8_disasm_audit.ps1` generates a synthetic text fixture for
each condition, runs `scripts/audit_hmx_d8_disasm.ps1`, and mutates it:

| Case | Expected |
| --- | --- |
| clean A/B/C/D | pass |
| one MMA missing | fail |
| MMA #1/#2 swapped | fail |
| a fill before MMA #3 | fail |
| out fill base `+8064` | fail |
| one 2,048 B store instead of 16 × 128 B | fail |
| guard write after the init | fail |
| `syncht` after MMA #3 | fail |
| unresolved MMA address | fail |
| Convert before the fills | fail |
| Store target `+8200` | fail |
| condition C with an Ahi fill | fail |
| sub-128 B fill store | fail |

The 2026-11-10 record is `d8_disasm_audit_test=PASS clean=4 mutants=12`.

**What the synthetic test proves:** the parser and the rules are executable and
reject the listed structural defects.

**What it does not prove:** that the Hexagon compiler emits the assumed forms,
that `hexagon-llvm-objdump` prints them exactly like the fixture, that the fill
is 16 × 128 B in the real object, or that the real object has no unresolved
addresses. Those are properties of a future stamped build. The synthetic test
must not be cited as a passed D8 disassembly audit.

## 6. Procedure for a future stamped object

```powershell
# 1. record the object identity
Get-FileHash -Algorithm SHA256 <skel.so>
# 2. disassemble the D8 symbol, once, from the stamped object
hexagon-llvm-objdump -d --disassemble-symbols=run_mode46_one_trial <skel.so> `
    > build\d8_disasm\stamped.txt
# 3. audit every condition path
foreach ($c in 'A','B','C','D') {
    .\scripts\audit_hmx_d8_disasm.ps1 -DisasmPath build\d8_disasm\stamped.txt -Condition $c
}
```

Requirements:

- the four invocations must all print `d8_disasm_audit=PASS`;
- the object SHA-256, the symbol, the tool version and the four outputs are
  preserved with the run record;
- if the disassembly contains a loop form the parser cannot normalize, the
  audit **fails**; the implementer must provide a normalizable form or the
  experiment is re-registered;
- a merged or relinked object invalidates the audit.

## 7. Limitations

1. **Static structure is not execution order.** A clean disassembly shows the
   sequence is present and no forbidden construct exists; it cannot prove how
   the hardware completes the MMA read.
2. **Compiler freedom.** The expected forms are a contract, not a prediction.
   A different but semantically equivalent lowering is a different experiment
   until it is audited and registered.
3. **No V81 tool here.** This checkout has no Hexagon SDK, `objdump` or skel;
   the audit rules have not seen a real V81 object.
4. **Loop forms.** The current rules accept straight-line blocks. A loop-based
   fill or pack must be normalized to the same address sequence (or the rules
   extended and re-tested) before it can pass.
5. **No timing, no performance, no device behaviour.**
