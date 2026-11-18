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

---

## Appendix A — Real-object extension (2026-11-19)

Status: **`D8_DISASM_RULES_ONLY`**, extended. This appendix records the three
rules added so the audit can read an actual V81 object, and states explicitly
that **no registered verification target changed**. The order, the offsets, the
fill form, the Convert `Rs` and the synchronisation ban are exactly as
registered; only the way the text is read changed.

### A.1 Why the rules had to change

The 2026-11-10 rules were written against a synthetic fixture that spells every
address as an absolute `##N`. A real object cannot: VTCM is allocated at run
time by `HAP_compute_res_attr_get_vtcm_ptr`, so the handler receives the base in
a register and derives every operand pointer from it. The compiler also lowers
`16 x 128 B` to `loop0(L,#0x10)` around a single `vmem` store, and it reuses
registers after their last use.

Applied unchanged to the stamped object, the rules failed 19 checks. Three of
those were properties of the audit, not of the object:

| Symptom | Cause |
| --- | --- |
| `convert_rs: 0x0fff/0x0001` | the Rs register was read from the final map, after the handler reused it |
| `mma_address_unresolved`, `store_address_unresolved` | the rules required absolute addresses, which a real object cannot contain |
| `pack_*_count: 0/16`, `guard_init_count: 1/12`, `fill_count: 3/48` | the rules counted stores statically, but 16 stores lower to one loop |

### A.2 The three rules

**A.2.1 `-BaseRegister rN` — base-relative resolution.**

The declared register resolves to offset 0. A register derived from it by a
constant add, directly or through a stack slot, inherits the offset. Any other
expression is unresolved and **fails**; nothing is skipped or warned.

In this mode a bare `##N` immediate is **not** an address. Without that rule the
payload constants the handler writes would resolve as addresses and pass:
`0x5400` (the bias scale) and `0xA5A5` (the fill pattern) were both observed
being read as addresses before the rule was added.

**A.2.2 Loop expansion.**

`loop0(L,#N)` … `:endloop0` around a single 128 B `vmem` store stands for `N`
stores at `base + k*128`. The count must be a literal immediate; a register
count is not statically known and fails. The body is every segment up to the one
carrying `:endloop0`, not just the next segment, because the compiler reloads
the pointer and builds the pattern inside the body. The stride is checked to be
128 B.

A loop whose body holds no vector store — the handler's FNV and copy loops — is
not part of the registered fill/pack form and is ignored.

**A.2.3 Per-segment derivation snapshots.**

Every address is resolved against the derivation map **as it stood at that
segment**, not against the final map. A register that holds an operand at the
MMA is routinely reused afterwards: the handler reloads `r5` for its FNV loops
and increments `r19` inside a fill loop. Resolving late read both at their later
values and produced wrong MMA and fill addresses.

The loop expansion additionally writes back the address it resolved, because it
uses the map at the loop entry rather than at the store, and that value is the
authoritative one.

### A.3 Two smaller corrections

- **64-bit `memd` reload.** `rM:rN = memd(r29+#off)` restores `rN` from `+off`
  and `rM` from `+off+4`, mirroring the spill. The compiler uses it to restore
  the callee-saved registers and to reload pointer pairs; without it `r19` lost
  its derivation in the epilogue.
- **`.new` suffix.** A stack spill may be spelled `memw(r29+#off) = rN.new`.

### A.4 What did NOT change

| Registered target | Status |
| --- | --- |
| issued-operation order (18 entries) | **unchanged**; still checked against `--d8-sequence-rev3` |
| VTCM offsets `+0/+2048/+4096/+6144/+8192/+10240/+10496/+10752` | **unchanged** |
| fill form `16 x 128 B`, mask A=1 B=7 C=3 D=5 | **unchanged** |
| Convert `Rs = 0x0001`, Store target `+8192` | **unchanged** |
| synchronisation ban in `[mxclracc, Store]` | **unchanged** |
| guard init `12 x 128 B`, no write after the init | **unchanged** |
| fail-closed policy | **unchanged**; an unresolved expression is still a FAIL |

### A.5 Test record (2026-11-19)

| Suite | Result |
| --- | --- |
| synthetic, original 16 cases | **unchanged, all pass** |
| synthetic, added 9 cases (base-relative clean A/B/C/D + 5 mutations) | **all pass** |
| total synthetic | `d8_disasm_audit_test=PASS clean=4 mutants=12 base_clean=4 base_mutants=5` (25 cases) |
| stamped object, conditions A/B/C/D | `d8_disasm_audit=PASS packs=64 guard=12 fills=48 mma=3 convert=1 store=1` |

The added mutations are: MMA #3 reading `+0` instead of `+2048`; a loop count of
8 instead of 16; a real `Rs` change; a payload constant used as an address; and
the same base-relative text with no `-BaseRegister` declared, which must fail
closed.

### A.6 What the real-object pass does and does not prove

**Proves:** the stamped object contains the registered sequence — one
`mxclracc`, three MMA packets with the registered operand pairs, the three fill
blocks of 16 x 128 B at the registered bases, one Convert with `Rs = 1`, one
Store at `+8192`, a 12 x 128 B guard init with no later guard write, and no
synchronisation in the critical interval.

**Does not prove:** that the hardware completes the MMA read before the fill
writes, that the numeric outcome matches a reference, or anything about timing.
Those are the device questions the four runs exist to answer.

### A.7 Two implementation defects this audit caught

Recorded because they show the audit is load-bearing, not decorative.

1. **`bias = mxmem2` was never issued.** The handler wrote the two bias payloads
   to memory but never loaded them into the bias register, so the Convert had no
   registered bias state. The registration lists `BIAS 0` and `BIAS 1` as issued
   operations.
2. **`-O2` folded the third pack onto the out_pos slot.** At `-O0` the object
   calls the pack helper four times with `+0`, `+0x800`, `+0x1000` and
   `+0x1800`. At `-O2` the helper was inlined into the handler and the third
   pack was folded onto `+0x2000`: `whi_local` was written to `+8192` and
   `+4096` was never written. Four different spellings of the destination all
   produced the same fold, so the fix is `__attribute__((noinline))` on the
   helper, not a rewording of the call.
