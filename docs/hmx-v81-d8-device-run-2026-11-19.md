# Candidate B — D8 Device Run Record (2026-11-19)

Status: **`D8_CONTRACT_FAILED`** — the session ran, the identity gate passed, and
condition A stopped at the handler's own operand-FNV gate **before the critical
interval**. No numeric D8 evidence was obtained. The registered budget (one
session, four runs, no retry) is **spent**.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md` (registered revision 3)
- `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md`
- `docs/hmx-v81-d8-fastrpc-failure-contract.md`
- `docs/hmx-v81-d8-disasm-audit-spec.md` (Appendix A is the real-object extension)
- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (implementation history)

## 1. What was run

| Item | Value |
| --- | --- |
| Date | 2026-11-19 (device clock 2026-11-18T21:15:11Z) |
| Session id | `d8-20261119-a` |
| Device | NX741J (`PQ85A01`, `canoe`, Android 16, build `MyOS16.0.28_NX741J_NEEA`) |
| DSP skel SHA-256 | `56bf5c03eac39fe83fdb7389a50ffafa4bdd7a56b72f3cadb7a4c16704e0c0d4` |
| DSP build id | `0x4D8B1121` |
| Host executable SHA-256 | `76dcf755e04e38b615ec20a3eb7a52281797ec14ff9d869a3b60945a2973d35c` |
| Hexagon SDK | 6.6.0.0, tools 19.0.07 |
| Budget | 1 session, **1 of 4 runs attempted**, 0 retries — **exhausted** |

The identity gate passed before the experiment:

```
build_id=0x4d8b1121 dsp_build_id=0x4d8b1121 build_id_status=MATCH
dsp_identity=MATCH
IDENTITY_OK=PASS
RESOURCE_CLEANUP_OK=PASS
exit_code=0
```

## 2. The four runs

Order is fixed: A(q=6) → B(q=7) → C(q=8) → D(q=9), each once, no retry.

| Condition | q | Fill mask | Label | Observed SHA-256 | Matched reference |
| --- | ---: | ---: | --- | --- | --- |
| A | 6 | 1 | **`D8_CONTRACT_FAILED`** | `e5a00aa9991ac8a5ee3109844d84a55583bd20572ad3ffcd42792f3c36b183ad` | none |
| B | 7 | 7 | **not run** | — | — |
| C | 8 | 3 | **not run** | — | — |
| D | 9 | 5 | **not run** | — | — |

The observed array for condition A is **1,024 zeros**. Its digest is exactly the
SHA-256 of 2,048 zero bytes, which was verified independently. The critical
interval never wrote `out_pos`, so there is no numeric reading to compare.

## 3. What the handler reported

Metadata words 9..31 from the run (the full 32-word block is in the evidence):

| Word | Field | Observed | Registered | |
| ---: | --- | --- | --- | --- |
| 9 | build id | `0x4D8B1121` | `0x4D8B1121` | ok |
| 10 | `mxclracc` | `0` | 1 | **fail** |
| 11 | MMA | `0` | 3 | **fail** |
| 12 | Convert | `0` | 1 | **fail** |
| 13 | Store | `0` | 1 | **fail** |
| 14 | pack | `4` | 4 | ok |
| 15 | bias | `2` | 2 | ok |
| 16 | fill | `0x00000000` | mask 1 | **fail** |
| 17 | address | `0x00000000` | `0x1000` | **fail** |
| 18 | selector | `0x00000600` | cond 0, q 6 | ok |
| 19 | FNV Ahi | `0x57A0EC09` | `0xC6231266` | **fail** |
| 20 | FNV Alo | `0xD0E0C444` | `0x3B2D22D2` | **fail** |
| 21 | FNV Whi | `0x9C9FE034` | `0x8B88609F` | **fail** |
| 22 | FNV Wlo | `0x29A90A5C` | `0xA66AF695` | **fail** |
| 23 | guard pre | `0xF5EBD5C5` | `0xF5EBD5C5` | ok |
| 24 | guard post | `0xF5EBD5C5` | `0xF5EBD5C5` | ok |
| 25 | guard status | `0x011FFF00` | status 0, first-bad none | ok |
| 26 | failure | `0x10` (16, `FNV`) | 0 | **the gate itself** |
| 27 | schema | `0x0D8A5003` | `0x0D8A5003` | ok |
| 28 | readback | `0x00200004` | — | — |
| 30 | cleanup | `0` | 0 | ok |

`first_bad = mxclracc` because that is the first field the contract checks in
order; the operative failure is `metadata[26] = 16` (`HMX_D8_FAIL_FNV`).

## 4. Reading

**The critical interval did not execute.** `mxclracc`, the three MMAs, the fills,
the Convert and the Store all report zero, `fill` and `address` are zero, and the
result array is untouched zeros. The handler built the four operand tiles
(pack = 4), loaded the two bias sets (bias = 2), initialised the 1,536 B guard
correctly (guard pre and post both equal the registered zero-guard FNV), then
computed the four operand FNVs, found them different from the registered values,
set `metadata[26] = 16`, and returned without issuing the sequence.

**This is the fail-closed design working.** The handler refused to run an
experiment whose fixture was not the registered one. Without that gate the run
would have produced numbers that looked like a D8 outcome but were an artefact of
a wrong operand set.

**Therefore this run answers nothing about MMA operand reuse.** It is a
`D8_CONTRACT_FAILED` (state S2 of the failure contract): no numeric reading, the
raw arrays preserved, no retry.

## 5. Cause of the FNV mismatch

The guard FNV is correct, so the FNV function and the VTCM read path work; the
difference is in the operand bytes themselves.

The gap is that **`build_d8_operands` was never executed on the CPU.** It is
defined in `hmx_mode46_d8.h`, referenced only from that header (its definition,
the pack helper and the handler), and the only file that includes the header is
`probe_dsp.c`, which is DSP-side. The CPU suites verify the contract checker, the
classifier and a separate host model of the arithmetic — they never produce the
DSP's tiles. So the four registered FNV constants were compared against whatever
the DSP generated only at run time, on the device, once.

A second, weaker possibility that the evidence cannot separate: the four
`vmem` stores in `d8_pack_tile` are followed by scalar byte reads for the FNV
with only a compiler `"memory"` clobber and no data-cache or `syncht` barrier.
The guard case cannot distinguish this, because `d8_guard_init` writes zeros and
a read that misses the store also sees zeros.

Both are fixable offline. Neither is a V81 HMX property, so neither bears on
`B3_R2_SYNC_UNPROVEN`.

## 6. What this run can and cannot establish

**Can:** the build `0x4D8B1121` reaches the DSP, passes identity, acquires VTCM
and HVX, generates its operand fixture, and **stops itself** when that fixture
does not match the registered one. The fail-closed path is exercised on real
hardware, which the CPU suites could not show.

**Cannot:** anything about the D8 question. Whether the MMA finished reading an
operand tile before the output phase overwrote it is **untested**.

## 7. Labels after the run

| Label | Before | After |
| --- | --- | --- |
| `B3_R2_SYNC_UNPROVEN` | unchanged | **unchanged** (no evidence either way) |
| `B3_R2_FULL_TILE_CPU_READY` | unchanged | unchanged |
| `CPU_VALIDATED_PREFERRED_DESIGN` | Candidate B | unchanged |
| Candidate C / C2 | `C2 NOT_RUN` | unchanged (separate management) |
| Mode 46 | registered, `NOT_RUN` | **registered, executed once, `D8_CONTRACT_FAILED`** |
| D8 | `NOT_RUN` | **`D8_CONTRACT_FAILED`** — no numeric evidence |

## 8. Before another attempt

The budget for revision 3 is spent, so a second session needs a **new
registration**. The work that has to happen first is offline:

1. make the DSP operand fixture executable and comparable on the CPU, so the four
   FNV constants are checked before any device contact;
2. decide whether the pack needs an explicit barrier before the scalar FNV read,
   and if so register that as part of the sequence rather than adding it quietly;
3. re-run the disassembly audit, the CPU suites and `verify.ps1 -Profile Fast`
   against the corrected build, and stamp a new build id.

None of that is authorised by revision 3 §11.
