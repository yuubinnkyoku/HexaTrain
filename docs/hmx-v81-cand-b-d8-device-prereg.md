# Candidate B — D8 Minimal Device Experiment Pre-Registration

Status: **PRE-REGISTERED, NOT AUTHORISED, CPU-ONLY**, 2026-11-08.
Scope: the minimal device experiment that isolates **D8**, the last
design-specific synchronization question of Candidate B (phase-reused VTCM).

This document authorises nothing. No DSP source was written or modified, no DSP
build was made, no QAIRT invocation, no ADB, no device contact, no Mode 46
registration. §9 is the authorisation gate and it stays closed.

Labels unchanged by this document: `B3_R2_FULL_TILE_CPU_READY`,
`B3_R2_SYNC_UNPROVEN`, `CPU_VALIDATED_PREFERRED_DESIGN` (Candidate B), Candidate
C CPU-only-consistent and device-unconfirmed, `C2 NOT_RUN`, Mode 46 `NOT_RUN`,
`R2_ACC_REUSE_SUPPORTED` (Mode 45 only). The new label
`D8_CPU_PREREG_READY` (see §10) is a CPU-preparation verdict only.

The existing C2 pre-registration (`docs/hmx-v81-cand-c-c2-device-prereg.md`) is
not modified by this document.

## Revision history

| Revision | Date | State | What changed |
| --- | --- | --- | --- |
| 1 | 2026-11-08 | current | first registration; four conditions A–D, CPU references, trace validator, contract gate |

---

## 0. The one question

Candidate B keeps four operand tiles resident and writes its outputs into two
of them after the last MMA:

| VTCM offset | MMA phase | Output phase | Owner's last use |
| ---: | --- | --- | --- |
| `+0` | `Ahi` | `out_recon` | MMA #2 |
| `+2048` | `Alo` | — | MMA #3 |
| `+4096` | `Whi` | — | MMA #3 |
| `+6144` | `Wlo` | `out_neg` | MMA #2 |
| `+8192` | — | `out_pos` | none (fresh) |
| `+10240` | bias set 0 | bias set 0 | — |
| `+10496` | bias set 1 | bias set 1 | — |

The CPU model proves the **program order** (`last_use < first_write`) and the
numbers. It cannot prove that the hardware finished MMA #2's operand reads
before the later stores changed those bytes.

**D8a (Wlo):** has MMA #2 finished reading `Wlo` at `+6144` before the output
phase writes `Wlo` (sentinel fill and, later, `out_neg`)?

**D8b (Ahi):** has MMA #1/#2 finished reading `Ahi` at `+0` before the output
phase writes `Ahi` (sentinel fill and, later, `out_recon`)?

The experiment observes the **numerical consequence** of a lost read. It does
not attempt to observe the internal MX pipeline, and it does not rely on the
program order being a completion order.

What is deliberately *not* asked: the arithmetic (frozen), the VTCM budget
(proven), the ACC retention (Mode 45), the Convert→HVX direction (Mode 45), or
the streaming design's D1–D3 (Candidate B removes those operations).

---

## 1. What "read completed" means here

Three orders must be kept apart:

1. **issue order** — the MMA packet is issued before the fill packet;
2. **execution order** — the MX unit consumes the operand tile at some time
   after issue;
3. **completion order** — the last byte the MMA reads is read.

Only (3) matters. The experiment is designed so that a violation of (3) changes
the 1,024-cell output in a way the CPU references can identify. It cannot
measure *when* the read happened; it only bounds whether the fill beat it.

The experiment does **not** assume "in-order MX execution" (assumption A1 of
`docs/hmx-v81-b3-r2-sync-static-audit.md` §7.3). A1 would be a hypothesis, not a
measurement; this design tests the observation instead of leaning on it.

---

## 2. Existing synchronization evidence, re-evaluated (2026-11-08)

The audit `docs/hmx-v81-b3-r2-sync-static-audit.md` is the primary source. This
section repeats its four evidence classes without upgrades.

### 2.1 What a specification guarantees

**Nothing directly usable.** No primary-source statement that an HMX MMA's
operand read completes before a later store to the same VTCM bytes was obtained.
The V81 HMX PRM (80-N2040-62 Rev. AA) could not be read by the audit's reader;
the V73 HVX sequential-consistency statement is a different generation and is
not a V81 guarantee. The reviewer-supplied V81 PRM §2.2 sentence ("HMX reads
operands from VTCM") describes the data path, not a completion ordering.

If a future primary-source statement is found, it must be recorded with the
instruction, conditions, document revision and section. Until then D8 stays
`DEVICE_ONLY`.

### 2.2 What existing device experiments observed

| Observation | Covered by |
| --- | --- |
| DSP store → HMX operand read (tile written before the MMA is issued) | every mode (`memset`/`memh` then `mxmem`), no explicit sync |
| MMA → `cvt.hf = acc` with no intervening instruction | Modes 41/42/45, bit-exact |
| HMX store → later scalar read of the same bytes | Mode 45 FNV loop |
| HMX store → later HVX vector read of the same bytes | Mode 45 `vsub.hf` loop, S1/S2 matched |
| **A store to a tile an already-issued MMA is reading** | **never exercised** |

The first four are the easy direction. The fifth row is D8. A device success in
the easy direction is not evidence for the hard direction.

### 2.3 What the disassembly established

- The shipped skel contains **zero** `syncht`, barrier, `dmwait`/`dmstart` or
  bare `sync` instructions; every recorded device result came from unsynchronized
  issue order.
- The B3 branch emits three MMA packets back to back with no intervening
  instruction, each reading its own resident tile.
- The `asm volatile("" ::: "memory")` barrier stops the compiler and emits
  nothing; it does nothing to the hardware.
- The assembler's MX surface (`mxclracc`, `mxmem`, `mxmem2`, `mxshl`,
  `mxswapacc`) contains no wait/sync mnemonic. `mxswapacc` semantics are unknown.

### 2.4 What is still a hypothesis

- in-order MX execution (A1);
- `syncht` completing MX operand consumption;
- a dummy `cvt.hf = acc` acting as a wait for the operand read;
- `mxswapacc` as a state transition that implies completion;
- any compiler-visible ordering implying hardware completion.

None of these is used as a premise of this experiment.

### 2.5 Correction to an earlier candidate

The static audit §9 sketched an experiment that overwrites `wgt` **between
MMA #1 and MMA #2**. That is the streaming design's D1 hazard, not Candidate B's
D8: Candidate B never writes a live operand between MMAs. This registration
replaces that sketch with the reuse Candidate B actually performs, after the
third MMA. The earlier sketch remains a valid *streaming* experiment and is not
withdrawn; it is simply a different question.

---

## 3. Fixed layout and instruction sequence

The experiment reuses the Candidate B layout and arithmetic. It drops only the
operations that cannot affect D8: Convert #2 (whose store is later than the
sentinel fill) and the HVX reconstruction (whose store is later than the fill).
The fill is the earliest reuse write, so it is the tightest test of the read.

```text
# prologue (not on the critical path)
pack Ahi -> VTCM +0      (16 vectors x 128 B)
pack Alo -> VTCM +2048
pack Whi -> VTCM +4096
pack Wlo -> VTCM +6144
build bias set 0 (+10240): scale 0x5400 (+64), shape 0 (x), bo 0
build bias set 1 (+10496): scale 0x5400 (+64), shape 4 (-x), bo 0
fill out_pos (+8192, 2048 B) = 0xA5A5

# critical path (fixed)
mxclracc
MMA #1: activation = mxmem(+0)   ; weight = mxmem(+4096)   # Ahi x Whi
MMA #2: activation = mxmem(+0)   ; weight = mxmem(+6144)   # Ahi x Wlo
MMA #3: activation = mxmem(+2048); weight = mxmem(+4096)   # Alo x Whi
# condition-dependent reuse fills (16 x 128 B HVX vector stores each):
#   A: none
#   B: fill Wlo (+6144) = 0xA5A5 ; fill Ahi (+0) = 0xA5A5
#   C: fill Wlo (+6144) = 0xA5A5
#   D: fill Ahi (+0) = 0xA5A5
Convert #1: cvt.hf = acc(Rs = 0x0001), bias set 0  # shape 0, x64, retain ACC
Store #1:   mxmem(+8192) = cvt

# readback
harvest out_pos (2048 B, VTCM) + metadata
```

Registration notes:

- **The fills are the reuse writes under test.** They must be 16 iterations of a
  128 B HVX vector store. A scalar or compiler-generated store form is a
  different experiment and must be registered separately; the disassembly audit
  (§6 of the evidence contract) pins the actual form. The CPU model records the
  fill as one 2,048 B range event; vector granularity is a device property and is
  checked in the disassembly, not in the CPU trace.
- The fills happen **after MMA #3**. A branch or predication that skips a fill
  for the wrong condition is a contract violation, not a D8 result.
- Convert #1 and Store #1 are after the fills. They do not write any operand
  region, so they cannot rescue a read that the fill already corrupted.
- The out_pos fill is not a reuse write: `+8192` is not an operand region. It
  only makes an unwritten store detectable.
- The full Candidate B also stores Convert #2 into `Wlo` and the HVX recon into
  `Ahi`. Those writes are later than the fills, so a safe fill does not
  automatically prove them safe. This is a stated scope limit, not a hidden
  assumption.
- No `syncht`, barrier, dummy `cvt` wait or host RPC may appear between MMA #3
  and the fills, or between the fills and Convert #1. Adding one changes the
  experiment.

### 3.1 What the minimal sequence is not

- It does not produce `out_neg` or `out_recon`, so it does not reproduce the
  full Candidate B evidence set.
- It does not test the later Convert #2 store or the HVX recon store; it tests
  the earliest write to each reused region.
- It does not test the Plain subcase, the B3 recon, or full-tile accuracy.
- It does not use a distance sweep: one condition run is one sample at the
  registered instruction distance. A single safe result does not bound other
  distances.

### 3.2 Static implementation feasibility (not implemented)

The existing RPC is `run(rout results, rout metadata, in long q, in long mode)`.
A condition selector can be carried by `q` without an IDL change, but the DSP
function must branch or predicate the fills on it. That is DSP work and is
**not designed, written, built or authorised here**. If the future
implementation cannot express four conditions in one build, the experiment must
be re-registered as four separate sequences; it must not silently run only one
condition and report it as D8.

The four conditions differ only in which of two fills execute. A static audit
can confirm from the object that each condition's fill set matches the table
above. No allocator behaviour, stack usage or timing is claimed.

---

## 4. Conditions

| Condition | Operand fills after MMA #3 | Purpose | What a match to the safe array means |
| --- | --- | --- | --- |
| **A** | none | numeric control | the minimal sequence is correct in this build |
| **B** | Wlo and Ahi | Candidate B's reuse | both reads completed before their fills |
| **C** | Wlo only | isolates D8a | the Wlo read completed before its fill |
| **D** | Ahi only | isolates D8b | the Ahi read completed before its fill |
| **E** | CPU only | detector proof | never a device condition |

Condition E is a CPU fault-injection control: it writes an operand before the
MMA read and requires the trace validator and the numeric classifier to detect
it. It must never be built into a device image.

Attribution rules:

- A wrong result under **C** is a D8a suspect; under **D** a D8b suspect;
  under **B** the observed array says Wlo, Ahi or both.
- A wrong result under **A** is a control anomaly (input, packing, convert,
  bias or output path), not a D8 result.
- An array matching none of the four references is inconclusive; partial
  corruption is not attributed.

---

## 5. Inputs and expected references

### 5.1 Operands

The registered deployed operand condition (`g_ops_deploy`), serialised as the
six-tile `operands_deploy` array:

| Array | Elements | SHA-256 |
| --- | ---: | --- |
| `operands_deploy` (ahi, alo, whi, wlo, a_plain, w_plain) | 6,144 u16 | `e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3` |

The four D8 tiles are the first four sections of that array, packed with the
registered PRM crouton mapping. Packed-tile FNV-1a 32 values (little-endian
bytes, the layout the MMA reads):

| Tile | Bytes | FNV-1a 32 |
| --- | ---: | --- |
| Ahi | 2,048 | `c6231266` |
| Alo | 2,048 | `3b2d22d2` |
| Whi | 2,048 | `8b88609f` |
| Wlo | 2,048 | `a66af695` |

The device must echo these four values; a mismatch is a contract failure and no
numeric interpretation is made.

Why this fixture: Ahi is non-zero in 934 of 1,024 cells and Wlo in 769; both
operands contribute to the same GEMM, so a lost read propagates to the output.
The sentinel `0xA5A5` is a normal negative fp16, so the corruption does not
disappear in the convert clamp and does not produce NaN/inf.

### 5.2 The four reference arrays

All references are `uint16` in decoded cell order `s*32+o`, little-endian.

| Reference | Meaning | SHA-256 | Non-zero cells |
| --- | --- | --- | ---: |
| **safe** | all reads completed | `c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d` | 548 |
| **wlo** | Wlo held the sentinel at MMA #2 | `f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb` | 617 |
| **ahi** | Ahi held the sentinel at MMA #1/#2 | `b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc` | 1,024 |
| **both** | both tiles held the sentinel | `5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d` | 1,024 |

Pairwise differences (1,024 cells):

| Pair | Differing cells |
| --- | ---: |
| safe vs wlo | 543 |
| safe vs ahi | 777 |
| safe vs both | 853 |
| wlo vs ahi | **726** |
| wlo vs both | 852 |
| ahi vs both | 640 |

The four arrays are pairwise distinct, so the experiment can attribute a
violation to D8a, D8b or both. The `wlo` vs `ahi` distance (726 cells) is the
discrimination that matters for separating the two sub-questions.

CPU cross-check: the safe array is **cell-for-cell identical** to the shipped
B3 subcase's positive output (`d8_safe_vs_shipped_b3_pos diff=0`). The minimal
sequence therefore uses the same arithmetic as the registered B3 model, and the
new digests do not alter any existing one.

### 5.3 How a reference is computed

For each output `(s,o)`, the three MMA passes accumulate in float32:

```text
for each MMA k:  dot_k = 0
                 for i: dot_k = f32(dot_k + f32(a_k[s][i] * b_k[o][i]))
                 acc    = f32(acc + dot_k)          # k = 1, 2, 3 in order
cvt  = RN16(64 * acc)
cell = 0 if cvt has the sign bit set, else cvt
```

`A_1 = Ahi, B_1 = Whi`; `A_2 = Ahi, B_2 = Wlo`; `A_3 = Alo, B_3 = Whi`.
A corrupt reference replaces the whole affected packed tile with `0xA5A5`
before the MMA executes. This is the "the read lost the race" outcome model; it
is not a proposed program order.

---

## 6. Classification

Evaluated in this order; the first match wins.

```text
contract gate (lengths / ids / build / counts / addresses / operand FNV /
               fill mask and word / guards) failure
      -> D8_CONTRACT_FAILED                       (stop, no numeric reading)
invalid access trace (CPU-side only)
      -> D8_INCONCLUSIVE_TRACE                    (fail closed)
condition A
      safe      -> D8_CONTROL_SAFE
      otherwise -> D8_CONTROL_ANOMALY             (not a D8 result)
condition B
      safe      -> D8_NO_VIOLATION_OBSERVED (both regions)
      wlo       -> D8_VIOLATION_WLO_SUSPECT
      ahi       -> D8_VIOLATION_AHI_SUSPECT
      both      -> D8_VIOLATION_BOTH_SUSPECT
      otherwise -> D8_INCONCLUSIVE_NUMERIC
condition C
      safe      -> D8_NO_VIOLATION_OBSERVED (Wlo)
      wlo       -> D8_VIOLATION_WLO_SUSPECT
      otherwise -> D8_INCONCLUSIVE_NUMERIC
condition D
      safe      -> D8_NO_VIOLATION_OBSERVED (Ahi)
      ahi       -> D8_VIOLATION_AHI_SUSPECT
      otherwise -> D8_INCONCLUSIVE_NUMERIC
```

Rules:

- **A trace anomaly overrides a matching numeric result.** A safe array with an
  invalid access trace is not "no violation".
- **An unmatched numeric result is never "no violation".** Partial corruption,
  a wrong input, a convert/bias error and an output-path error all land here.
- **Condition A is a control, not evidence.** A mismatch under A stops the run
  as a control anomaly; it says nothing about D8.
- **A safe match is exact bit equality on all 1,024 cells**, not a tolerance.
- Every terminal class except `D8_CONTRACT_FAILED` and `D8_INCONCLUSIVE_*`
  carries a `SUSPECT` or `OBSERVED` qualifier, never `PROVEN`.

---

## 7. What each outcome means (A / B / C)

### A. The device observes a wrong result (violation suspect)

This is consistent with a read that lost the race. It is **not** proof of a D8
violation until the other causes are excluded:

- input mix-up → the operand FNV gate catches it;
- wrong pack layout → the FNV gate over the packed tile catches it;
- wrong bias/scale/shape → the convert outputs would not match the corrupt
  reference either; but a bias error could coincidentally produce some other
  array, which is why the contract also pins the bias-set geometry and counts;
- output path / ABI (wrong offset, truncation, sign extension) → the store
  address, length and zero-extension checks;
- build mismatch → the Build ID echo;
- a fill that executed with the wrong mask or word → the fill mask/word echo and
  the disassembly.

If the contract gate passes and the array matches a corrupt reference exactly,
the registered interpretation is: **the fill beat the MMA read for every cell
that carries the difference**. The result is still recorded as `SUSPECT`
because a hardware-internal cause other than D8 cannot be excluded from one run.

### B. The device observes the safe result (no violation observed)

This means: in this build, at this instruction distance, in this run, every
element the MMA read was the pre-fill value. It does **not** mean:

- that D8 holds for other distances, other builds or other power states;
- that the later Convert #2 / recon stores are safe (they are later, not
  measured);
- that the full Candidate B run is numerically correct or device-ready;
- that `B3_R2_SYNC_UNPROVEN` is retired.

The result is recorded as `D8_NO_VIOLATION_OBSERVED`, never `D8_PROVEN`.

### C. Contract failure or insufficiency

No numerical interpretation is made. The run stops, the raw bytes and metadata
are preserved, and the cause is recorded as a contract failure. A contract
failure is not a synchronization result and must not be counted as one.

---

## 8. Stop conditions

The run stops immediately, with no retry, if any of the following holds:

- Build ID mismatch or missing; RPC or close status non-zero;
- results length ≠ 1,024 or metadata length ≠ 32;
- a slot is outside 0..65,535 (sign extension / truncation);
- any operand FNV mismatch;
- MMA count ≠ 3, Convert count ≠ 1, Store count ≠ 1, pack count ≠ 4, bias sets
  ≠ 2, readbacks ≠ 1;
- the store target ≠ `+8192`;
- the condition fill mask / fill word mismatch;
- any guard or untouched-region check fails;
- the condition selector echo differs from the requested condition;
- a disassembly audit condition of the evidence contract §6 fails;
- the observed array is inconclusive.

On a violation suspect: preserve the raw `out` array, the full metadata, the
digests of what was observed, the Build ID, the disassembly hash and the RPC
transcript, then stop. No "run it again to see".

---

## 9. Execution budget and authorisation gate

| Item | Value |
| --- | --- |
| Builds | 1 clean stamped DSP build, four condition paths |
| Device sessions | 1 |
| Runs | 4 (A, B, C, D) |
| Retries | 0 |
| Timing / performance | none |
| Profile | Tier 2 (`BACKGROUND_CORRECTNESS`) if ever executed |
| Authorisation | **NOT GRANTED** |

Forbidden now, by this document, regardless of approval: DSP source changes,
DSP builds, QAIRT invocation, ADB, device contact, performance measurement,
Mode 46 registration, `push`/`main` changes, PR merge.

Approving §9 would approve only: 1 build, 1 session, 4 condition runs, 0
retries, under the evidence contract. It would not approve a fifth run, a
distance sweep, a full-tile run or training integration.

---

## 10. CPU-side validation performed (2026-11-08)

Executable in `host_tests/hmx_acc_residual_rpc/hmx_d8_min_sim.c` and
`hmx_d8_independent_test.py`; run by `scripts/run_hmx_d8_sim.ps1` in the Fast /
Host gate.

| Check family | What it establishes | Result |
| --- | --- | --- |
| reference build | safe equals the shipped B3 positive output cell-for-cell; four digests match the independent Python derivation | PASS |
| distinctness | six pairwise array differences are non-zero; Wlo vs Ahi = 726 cells | PASS |
| access trace | D8-specific validator over the recorded pack / read / fill / store events; writes before the last read, one-byte overlaps, wrong counts and wrong fills fail | PASS |
| faults 1–5 | Wlo early, Ahi early, both early, one-byte boundary, sentinel-value early: trace invalid, numeric not safe; sentinel case equals the `wlo` reference | PASS |
| fault 6 | normal post-last-use reuse under A/B/C/D: trace valid and numeric safe | PASS |
| faults 7–10 | safe numeric + bad trace, corrupt numeric + good trace, missing evidence, length/ID mismatch | PASS |
| independent Python | re-derives arithmetic and references, 20 checks, 119 differential cases vs the C classifier | PASS |
| mutation regression | 3 classifier mutations (trace override removed, unknown read as safe, contract removed) all detected | PASS |
| toolchains | gcc `-O0`, gcc `-O2`, clang `-O2` | PASS |
| existing regressions | B3/R2 12,828 checks, 30/30 faults, C2 differential, six digests | unchanged |
| Fast / Host | `scripts/verify.ps1 -Profile Fast` / `-Profile Host` | Fast PASS (PASS=10 FAIL=0 SKIP=21); Host PASS (exit 0) |

`D8_CPU_PREREG_READY` is recorded on this basis. It is a CPU-preparation
verdict: the references, classifier, fault model and evidence contract are
executable and consistent. It is not a device result, not an authorisation and
not a retirement of `B3_R2_SYNC_UNPROVEN`.

---

## 11. Unresolved items carried forward

1. **D8 itself** — this document registers how to ask the question; the answer
   needs the device experiment.
2. **The later reuse writes** — Convert #2 into `Wlo` and the HVX recon into
   `Ahi` are later than the fills; a safe fill does not prove them safe.
3. **Distance sweep** — a single safe run does not bound the race window. A
   sweep would need its own registration and budget.
4. **Store-form scope** — the registered fill is 16 × 128 B HVX vector stores.
   If Candidate B's real fill lowers to scalar stores, that is a different
   experiment.
5. **Condition selector** — the `q`-based selector is a design proposal, not an
   implementation. Without it the experiment cannot run as registered.
6. **Allocator acceptance** of the `+0`/`+6144`/`+8192` map and 128 B accesses.
7. **Mode 46** — unregistered and `NOT_RUN`; a trigger mode is a prerequisite.
8. **C2** — unchanged and `NOT_RUN`; the order comparison is a separate
   document (`docs/hmx-v81-d8-vs-c2-experiment-order.md`).
9. **Full-tile accuracy and performance** — untouched.
10. **Training integration** — untouched.

---

## 12. Reproduction

```powershell
# CPU only; no SDK, no device, no ADB.
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
.\scripts\run_hmx_d8_sim.ps1
python host_tests\hmx_acc_residual_rpc\hmx_d8_independent_test.py
```

Key output lines: four `d8_ref` digests, `d8_operand_fnv`, `d8_diff`,
`d8_fault=01..10`, `d8_checks=20 fail=0`, `d8_verdict=D8_CPU_READY`,
`d8_differential=PASS`, `d8_mutation_check=PASS mutants=3`.

Artifacts under `build/` are gitignored and are not device evidence.
