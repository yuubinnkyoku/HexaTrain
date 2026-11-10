# Candidate B — D8 Minimal Device Experiment Pre-Registration (Revision 3 Proposal)

Status: **`D8_REV3_PROPOSED_CPU_ONLY`**, 2026-11-10. This document is a
**proposal**, not a registration. Revision 2
(`docs/hmx-v81-cand-b-d8-device-prereg.md`) remains the registered revision and
is not modified by this document. This proposal authorises nothing: no DSP
source, no DSP build, no QAIRT invocation, no ADB, no device contact, no
Mode 46, no release, no merge.

Labels unchanged by this document:

| Label | Value |
| --- | --- |
| `B3_R2_FULL_TILE_CPU_READY` | unchanged |
| `B3_R2_SYNC_UNPROVEN` | **unchanged (D8 is still unrun)** |
| `CPU_VALIDATED_PREFERRED_DESIGN` | unchanged (Candidate B) |
| Candidate C / C2 | CPU-only-consistent, `C2 NOT_RUN` |
| Mode 46 | **unregistered, `NOT_RUN`** |
| Device approval | **`NOT GRANTED`** |
| D8 | **`NOT_RUN`** |

The new label `D8_REV3_PROPOSED_CPU_ONLY` means: the CPU-side contract is
machine-checked and self-consistent; it is not an approval, not a device
result, and not a retirement of `B3_R2_SYNC_UNPROVEN`.

## Revision history

| Revision | Date | State | What changed |
| --- | --- | --- | --- |
| 1 | 2026-11-08 | superseded | first registration; out_pos fill in the prologue; fill mask inconsistent (C/D swapped). |
| 2 | 2026-11-09 | **registered** | out_pos fill moved after MMA #3; mask A=1/B=7/C=3/D=5; no registered number changed. |
| 3 | 2026-11-10 | **proposed, CPU only** | metadata renumbered into words 9..31 with bit fields; `q` selector fixed to 6..9; results-length and failure-preservation semantics corrected; VTCM guard initialisation and pre/post digests added; disassembly audit conditions extended. No registered number changed. |

Revision 3 is a response to
`docs/hmx-v81-d8-implementation-feasibility-audit.md`, which established
`D8_IMPLEMENTATION_BLOCKED`: revision 2 could not be implemented against the
existing `run()` metadata ownership, its `q=0..3` selector, its short/long
return rule, or its missing guard channel. This document fixes the design; it
does not fix the device.

### Revision 3 before / after

| Item | Revision 2 | Revision 3 proposal |
| --- | --- | --- |
| Metadata words | D8 used words 0..24, colliding with `run()` words 0..8 | D8 uses **9..31 only**; words 0..8 keep their `run()` meaning |
| Build ID | word 0 (collides with magic) + word 24 | **word 9**, one echo; identity-mode `metadata[2]` is the second, independent echo |
| Condition selector | `q=0..3` (rejected by the DSP and host gates) | **`q=6..9`**, `condition = q-6`; `q=5` and `q=10` rejected |
| Results length | "short or long return" is `D8_CONTRACT_FAILED` | **no return length exists**; host allocates exactly 1,024 and the DSP requires `resultsLen == 1024` |
| Failure evidence | preserve raw bytes on non-zero return | **return 0 once execution started**; the first-failure code is in metadata; pre-execution failures keep non-zero |
| Guard | "tail stays zero", no channel | **registered zero init + pre/post FNV + status word** |
| Cleanup | word 22 written by the handler | **word 30 written by `run()` for mode 46**, with a documented return-0 exception |
| Convert / Store order | inferred from one fused HMX store event | **issued-operation record** separates Convert and Store |
| Disassembly audit | counts + order + no-sync | adds addresses, 16 × 128 B fill form, guard init loop, and unresolved-address fail-closed |

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
| `+10752..+12288` | **guard (1,536 B)** | guard | — |

**D8a (Wlo):** did MMA #2 finish reading `Wlo` at `+6144` before the output
phase wrote it?

**D8b (Ahi):** did MMA #1/#2 finish reading `Ahi` at `+0` before the output
phase wrote it?

The experiment observes the numerical consequence of a lost read. It does not
observe the internal MX pipeline and does not assume that program order is a
completion order.

## 1. Three orders

1. **issue order** — the MMA packet is issued before the fill packet;
2. **execution order** — the MX unit consumes the operand tile at some time
   after issue;
3. **completion order** — the last byte the MMA reads is read.

Only (3) matters. The experiment is designed so that a violation of (3) changes
the 1,024-cell output in a way the CPU references identify. It cannot measure
when the read happened; it bounds whether the fill beat it. No specification
statement, no `syncht`, no dummy `cvt` wait and no `mxswapacc` semantics are
used as a premise.

## 2. What blocked revision 2, and what revision 3 changes

| Finding (audit §9) | Revision 3 resolution |
| --- | --- |
| `mode46_unregistered` | still **BLOCKED**; registration is a governance step, not a CPU step |
| `metadata_ownership_collision` | renumbered into 9..31; words 0..8 untouched; `d8-meta-layout` machine block |
| `q_selector_encoding` | `q=6..9`, condition `q-6`; identity probe uses its own `q=6` |
| `results_length_unobservable` | exact allocation + exact DSP capacity check; no return-length claim |
| `error_path_raw_loss` | execution-started failures return 0; pre-execution failures stay non-zero; cleanup word 30 |
| `guard_tail_no_channel` | zero init + pre/post FNV + status + first bad byte |
| `fixture_packing_new` | unchanged DSP work; the tile generator already reproduces the fixture |
| `fnv_timing` | FNV must be taken **before MMA #1**; recorded as an order rule |
| `bias_position_template` | follow mode 45's bias-before-`mxclracc` prologue, not mode 42 |
| `mode45_bias2_tail` | explicitly forbidden; the guard owns `+10752` |
| `fill_form_unproven` | disassembly-only; the runtime form field is a self-report |
| `disasm_audit_fill_gap` | new D8 audit rules, fail-closed on unresolved addresses |
| `dsp_build_env_absent` | still an environment constraint; not claimed as resolved |

The CPU-side findings are resolved by this proposal. Mode 46 registration, a
stamped build and the device questions remain outside CPU scope.

## 3. Fixed layout and instruction sequence

```text
# prologue (not on the critical path)
pack Ahi -> VTCM +0      (16 vectors x 128 B)
pack Alo -> VTCM +2048
pack Whi -> VTCM +4096
pack Wlo -> VTCM +6144
build bias set 0 (+10240): scale 0x5400 (+64), shape 0 (x), bo 0
build bias set 1 (+10496): scale 0x5400 (+64), shape 4 (-x), bo 0
guard init: +10752..+12288 <- 0x00 (12 x 128 B HVX stores)
guard pre:  FNV-1a 32 of +10752..+12288

# critical path (fixed)
mxclracc
MMA #1: activation = mxmem(+0)   ; weight = mxmem(+4096)   # Ahi x Whi
MMA #2: activation = mxmem(+0)   ; weight = mxmem(+6144)   # Ahi x Wlo
MMA #3: activation = mxmem(+2048); weight = mxmem(+4096)   # Alo x Whi
fill out_pos (+8192, 2048 B) = 0xA5A5        # 16 x 128 B
# condition fills (16 x 128 B each):
#   A: none
#   B: fill Wlo (+6144) = 0xA5A5 ; fill Ahi (+0) = 0xA5A5
#   C: fill Wlo (+6144) = 0xA5A5
#   D: fill Ahi (+0) = 0xA5A5
Convert #1: cvt.hf = acc(Rs = 0x0001), bias set 0
Store #1:   mxmem(+8192) = cvt

# epilogue (outside the critical interval)
guard post: FNV-1a 32 of +10752..+12288, compare with guard pre
readback:   results[0..1023], metadata[0..31]
```

Registration notes:

- **The guard init and both guard hashes are outside the MMA #3 → fill →
  Convert interval.** They cannot shorten the registered reuse distance. The
  guard init is 12 × 128 B stores, before `mxclracc`; the post hash is after
  Store #1.
- **Fill order is out_pos → Wlo → Ahi**, all after MMA #3 and before Convert
  #1. A branch that skips a required fill, or executes an operand fill for
  condition A, is a contract failure.
- **No `syncht`, barrier, DMA sync, dummy `cvt` wait, host RPC, checksum loop
  or scalar byte read of `+0`/`+6144`/`+8192`** may appear in the critical
  interval.
- **The guard is not a declared write region during the run.** After the
  registered init it must stay `0x00`; any later write is a contract failure.
- **Bias set 2 is forbidden.** Mode 45's third set lives at `+10752..+11008`,
  inside the D8 guard. Copying mode 45's prologue would violate the guard
  before the experiment starts.

### 3.1 Registered issued-operation order (machine-checked)

`hmx_d8_min_sim.c --d8-sequence-rev3` derives this list from its issued-operation
recorder. `hmx_d8_rev3_contract_test.py` parses the block below and requires all
four condition signatures to match the C list exactly. A different order is a
different experiment even when the 1,024 numbers agree.

<!-- d8-rev3-sequence:begin -->
PACK AHI
PACK ALO
PACK WHI
PACK WLO
BIAS 0
BIAS 1
GUARD INIT 12
GUARD PRE
MXCLRACC
MMA 1 AHI WHI
MMA 2 AHI WLO
MMA 3 ALO WHI
FILL OUT
FILL WLO B C
FILL AHI B D
CONVERT 1
STORE OUT
GUARD POST
<!-- d8-rev3-sequence:end -->

### 3.2 The `q` selector

The shared `run()` gate accepts `q` in `[6,10]`. Revision 2's `q=0..3` is
rejected before dispatch and cannot be implemented. Revision 3 maps:

<!-- d8-q-map:begin -->
6 A
7 B
8 C
9 D
<!-- d8-q-map:end -->

Rules:

- `condition = q - 6`; the handler rejects `q=10` (and anything outside 6..9)
  with `HMX_ERR_ARG` before execution.
- The metadata selector echoes both `condition` and `q`; the host requires the
  pair to match the requested `q`. A C/D swap in the table fails both echoes.
- The identity probe is a **separate** mode 30 call with its own `q=6`; it is
  not the experiment's selector. The host does not reuse the experiment `q`
  for the identity call.
- Existing modes' `q` meanings are unchanged.

### 3.3 Guard initialisation and verification

VTCM contents at entry are not guaranteed to be zero. Revision 3 registers an
explicit init instead of assuming a zeroed window:

1. write `0x00` to `+10752..+12288` with 12 × 128 B HVX stores, before
   `mxclracc`;
2. hash the guard with FNV-1a 32 (little-endian bytes) and echo `guard_pre`;
3. after Store #1, hash again and echo `guard_post`, the status, and the first
   differing byte offset;
4. the host requires `guard_pre == guard_post == FNV-1a 32 of 1,536 zero bytes`
   (`0xf5ebd5c5`), status `0`, and no differing byte.

If the init is missing, the pre digest is not the zero digest, the post digest
differs, or the status/self-report is inconsistent, the run is
`D8_CONTRACT_FAILED` with the guard reason. **The numeric result is not
interpreted**, even when it equals the safe reference: a dirty guard means the
memory window is not the registered one.

The post hash is computed after Store #1 and before the readback. It reads only
the guard, never an operand or `out_pos`, so it cannot serialise the reuse.

## 4. Conditions

| Condition | q | Operand fills after MMA #3 | Fill mask | Purpose |
| --- | ---: | --- | ---: | --- |
| **A** | 6 | none | `1` | numeric control |
| **B** | 7 | Wlo and Ahi | `7` | Candidate B's reuse |
| **C** | 8 | Wlo only | `3` | isolates D8a |
| **D** | 9 | Ahi only | `5` | isolates D8b |
| **E** | n/a | CPU only | n/a | fault injection; never a device condition |

The mask is the metadata echo checked by the contract gate, and the issued-order
validator derives the same mask from the observed fills. Condition C must echo
`3`, D `5`; a swap is a contract failure, not a D8 result.

## 5. Inputs and expected references (unchanged)

The registered operands, the four packed-tile FNVs, the four reference-array
SHA-256 values and the six pairwise differences are **unchanged from revision
2**. They are not re-derived here; revision 3 does not replace any registered
number.

| Item | Value |
| --- | --- |
| `operands_deploy` SHA-256 | `e5193d9980150245f7a0e0a6d9d7361db7be22c999c1a95594053dee1f0d47a3` |
| FNV Ahi / Alo / Whi / Wlo | `c6231266` / `3b2d22d2` / `8b88609f` / `a66af695` |
| `safe` / `wlo` / `ahi` / `both` SHA-256 | `c9abf30e…20203d` / `f0b3b4aa…81f6fb` / `b6328e88…760fcc` / `5285b463…2535d` |

Exact match means all 1,024 cells, no tolerance.

## 6. Metadata

The full 32-word layout, ownership table and bit fields are in
`docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md` §3 and
`host_tests/hmx_acc_residual_rpc/hmx_d8_contract.h`. Summary:

- words 0..8: written by `run()`, unchanged meaning on every mode;
- words 9..31: D8-only evidence;
- word 30: written by `run()` for mode 46 after cleanup;
- `metadata[7] = 1` and `metadata[8] = 1` on every return-0 D8 run.

D8 never writes words 0..8 and never relies on word 3, 7 or 8 being
handler-controlled; `run()` rewrites them after the handler.

## 7. Length, allocation and identity

### 7.1 Exact shape

| Parameter | Value |
| --- | ---: |
| results | 1,024 slots = 4,096 B |
| metadata | 32 slots = 128 B |
| total | 4,224 B |
| `resultsLen` | exactly 1,024 |
| `metadataLen` | exactly 32 |

The IDL signature is unchanged:
`run(rout sequence<long> results, rout sequence<long> metadata, in long q, in long mode)`.
`rout` length is the **client maximum**, not a returned count. There is no
short/long return to observe. The host allocates exactly 1,024; the DSP
requires `resultsLen == 1024 && metadataLen == 32` for mode 46 and returns
`HMX_ERR_ARG` otherwise, before execution. The host may not pass a larger
buffer and call it complete.

### 7.2 Unwritten slots

The host cannot distinguish "the DSP wrote 0" from "the DSP never wrote" after
`run()` zeroes the buffer. Revision 3 therefore combines three checks:

1. every slot must be a zero-extended u16 (`0..0xFFFF`);
2. the DSP echoes `slots_written = 1024` (a self-report; it is not independent);
3. all 1,024 cells are compared exactly with a reference. The safe reference
   has 548 non-zero cells spread across the array, so a truncated write cannot
   equal it.

The self-report limit is stated, not hidden: a dishonest or miscompiled store
loop is caught by the exact comparison and by the disassembly audit, not by the
count alone.

### 7.3 Identity gate

Before any D8 run:

1. mode 30 identity with `q=6`: `results[0..3] = {3, 6, 30, 81}`,
   `metadata[0] = 0x484d5831`, `metadata[1] = 81`, `metadata[7] = 0`,
   `metadata[8] = 0`, `metadata[2] = build id`;
2. the D8 run echoes the build id at word 9; the host requires it to equal the
   identity-mode value and the stamped value.

The identity-mode echo and the run echo are the two independent build-id
observations. The D8 table does not need a third copy.

## 8. Failure states and evidence preservation

`run()` returns non-zero for pre-execution failures and **0 for every failure
after execution started**, because a non-zero FastRPC return discards `rout`
data. "Execution started" means the D8 handler has acquired VTCM and HVX and
has written `metadata[9]`; at that point all evidence is in the `rout` buffers.

| State | Example | `run()` return | Metadata / results | Host class | Numeric reading |
| --- | --- | ---: | --- | --- | --- |
| **S0 pre-execution argument** | `resultsLen != 1024`, wrong `q`, unwritten mode 46 | `14001` | not propagated | `D8_CONTRACT_FAILED_PREEXEC` | none |
| **S1 pre-execution resource** | power, HMX lock, VTCM reserve, pointer alignment, HVX lock | `14006`/`14004`/`14003`/`14002`/`14010` | not propagated | `D8_CONTRACT_FAILED_PREEXEC` | none |
| **S2 contract failure after start** | count, mask, address, FNV, selector, order | `0` | full, `metadata[7]=1` | `D8_CONTRACT_FAILED` + word 26 reason | none |
| **S3 guard failure after numeric acquisition** | guard corrupt, post hash differs | `0` | full | `D8_CONTRACT_FAILED` + word 8 reason | none |
| **S4 cleanup failure** | unlock2/release/power-down | `0` for mode 46, `metadata[30] != 0` | full | `D8_CONTRACT_FAILED_CLEANUP` | none |
| **S5 RPC transport failure after invocation** | FastRPC error, handle failure | n/a | unavailable | `D8_EVIDENCE_LOST` (or `D8_TRANSPORT_FAILED` if provably pre-invocation) | none |
| **S6 host post-processing failure** | record/copy failure | n/a | raw buffer may survive | `D8_HOST_EVIDENCE_FAILED` | none |

Notes:

- **The return-0 rule is not "always return 0".** It applies only after
  execution started. Pre-execution failures keep the existing non-zero codes,
  so the harness still detects power/lock/VTCM/HVX problems.
- **Cleanup is the one evidence-loss risk.** The current `run()` returns a
  non-zero cleanup code, which would discard the arrays. Revision 3 requires a
  mode-46-only exception: after the handler, `run()` writes `metadata[30]` with
  the first cleanup error and returns 0 when `metadata[7] == 1`. The arrays
  survive; the host treats `metadata[30] != 0` as a contract failure. If the
  implementer refuses the exception, a cleanup failure after execution is
  `D8_EVIDENCE_LOST` and the run stops.
- **Transport failure before the RPC returns is unrecoverable.** The host
  cannot know whether the four MMAs ran. No retry, no interpretation.
- The handler returns `HMX_ERR_HVX` before writing `metadata[9]` if the HVX lock
  fails; that is S1, not S2.
- `run()` writes `metadata[7]=1` and `metadata[8]=1` for mode 46 after the
  handler returns 0. The D8 handler returns 0 for contract failures after start
  and carries the reason in word 26.

## 9. Classification

The order and the terminal classes are unchanged from revision 2 §6:

```text
contract gate (lengths / q / ids / build / counts / addresses / operand FNV /
               fill mask and word / guard digests) failure
      -> D8_CONTRACT_FAILED                       (stop, no numeric reading)
invalid issued order or access trace (CPU-side only)
      -> D8_INCONCLUSIVE_TRACE                    (fail closed)
condition A: safe -> D8_CONTROL_SAFE; otherwise D8_CONTROL_ANOMALY
condition B: safe -> D8_NO_VIOLATION_OBSERVED; wlo/ahi/both -> *_SUSPECT;
             otherwise D8_INCONCLUSIVE_NUMERIC
condition C: safe -> D8_NO_VIOLATION_OBSERVED (Wlo); wlo -> D8_VIOLATION_WLO_SUSPECT
condition D: safe -> D8_NO_VIOLATION_OBSERVED (Ahi); ahi -> D8_VIOLATION_AHI_SUSPECT
```

Every terminal class except `D8_CONTRACT_FAILED` and `D8_INCONCLUSIVE_*`
carries `SUSPECT` or `OBSERVED`, never `PROVEN`. A safe result is one build,
one instruction distance, one run.

## 10. Stop conditions

The run stops immediately, with no retry, if any of the following holds:

- any RPC status other than 0 except the documented cleanup case;
- `resultsLen != 1024`, `metadataLen != 32`, or any slot outside `0..65535`;
- build id mismatch (identity echo vs run echo vs stamped value), schema
  mismatch, selector mismatch, or q outside 6..9;
- any count mismatch: `mxclracc != 1`, MMA `!= 3`, Convert `!= 1`,
  Store `!= 1`, pack `!= 4`, bias `!= 2`;
- store target `!= +8192`, out fill `!= +8192`, or a Wlo/Ahi fill address that
  does not match the condition;
- any operand FNV mismatch;
- fill mask / fill word / fill form mismatch, including C/D swap or B missing a
  fill;
- any guard digest, guard status, guard first-bad-byte, or guard method
  mismatch, including a non-zero entry tail after the registered init;
- a `metadata[30]` cleanup error;
- the issued order differs from §3.1, or an access event is missing,
  duplicated, out of bounds or before the last MMA read;
- any disassembly audit condition of `docs/hmx-v81-d8-disasm-audit-spec.md`
  fails or cannot be resolved;
- the observed array matches none of the four references.

## 11. Execution budget and authorisation gate

| Item | Value |
| --- | --- |
| Builds | 1 clean stamped DSP build, four condition paths |
| Device sessions | 1 |
| Runs | 4 (A, B, C, D) |
| Retries | 0 |
| Timing / performance | none |
| Profile | Tier 2 (`BACKGROUND_CORRECTNESS`) if ever executed |
| Authorisation | **NOT GRANTED** |

Approving a future revision 3 §11 would approve only: 1 build, 1 session,
4 condition runs, 0 retries, under the evidence contract. It would not approve
a fifth run, a distance sweep, a full-tile run or training integration.

## 12. CPU-side validation performed (2026-11-10)

All CPU only; no SDK, no device, no Mode 46. The exact run record is in
`docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md` §8.

| Check family | Result |
| --- | --- |
| issued-order validator (mxclracc / 3 MMA / 4 packs / 2 bias / 1 Convert / 1 Store) | PASS |
| order faults 18–30 (bias after MMA, swap, missing; mxclracc missing/duplicate; MMA missing/swapped; Convert missing/duplicate; Store before Convert; dropped/duplicated records) | PASS |
| guard faults 31–38 (dirty entry, missing/broken init, head/middle/tail corruption, missing check, bias set 2, false self-report, numeric-safe corruption) | PASS |
| metadata / q / failure contract (104 C checks) | PASS |
| independent Python contract audit (140 checks, 244 replay cases) | PASS, 0 mismatches |
| existing D8 / C2 / B3 / CB-FIX-1 regressions | unchanged |

## 13. What only the device can establish

1. whether the MMA operand read completes before the fill write (D8 itself);
2. whether the allocator grants 12,288 B with 2 KiB alignment on this session;
3. whether 16 × 128 B HVX stores to `+0`/`+6144` are accepted;
4. the actual VTCM entry contents and the actual emitted fill form;
5. the real timing/ordering behaviour behind the numeric outcome.

## 14. Unresolved items carried forward

1. **Mode 46 registration** — unregistered and `NOT_RUN`; a trigger mode is a
   prerequisite.
2. **A stamped DSP build** — no Hexagon SDK or build environment in this
   checkout.
3. **Disassembly audit of the real object** — the rules and a synthetic parser
   test exist; no V81 binary was produced.
4. **The later reuse writes** — Convert #2 into `Wlo` and the HVX recon into
   `Ahi` are later than the fills and are not measured.
5. **Distance sweep** — a single safe run does not bound the race window.
6. **Store-form scope** — if the real fill lowers to scalar stores, the
   registered experiment does not describe it.
7. **Allocator acceptance** and the 128 B access pattern.
8. **C2** — unchanged and `NOT_RUN`; the order comparison is separate.
9. **Full-tile accuracy, performance and training integration** — untouched.

## 15. Reproduction

```powershell
# CPU only; no SDK, no device, no ADB.
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
.\scripts\run_hmx_d8_sim.ps1

# Issued-order signatures for all four conditions
.\build\hmx_d8_sim.exe --d8-sequence-rev3
```

Artifacts under `build/` are gitignored and are not device evidence.
