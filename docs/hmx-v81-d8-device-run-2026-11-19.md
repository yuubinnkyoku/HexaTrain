# Candidate B — D8 Device Run Record (2026-11-19)

Status: **`D8_NOT_RUN`** — this document is the record template. The four
condition runs have **not** been executed. Every field marked `PENDING` is
filled from the session evidence after the single registered session.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md` (registered revision 3)
- `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md`
- `docs/hmx-v81-d8-fastrpc-failure-contract.md`
- `docs/hmx-v81-d8-disasm-audit-spec.md` (Appendix A is the real-object extension)
- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (implementation history)

## 1. What was run

| Item | Value |
| --- | --- |
| Date | PENDING |
| Session id | PENDING |
| Device | NX741J (`PQ85A01`, `canoe`, Android 16) |
| ADB endpoint | PENDING (the endpoint changes between sessions and is not committed) |
| DSP skel SHA-256 | `56bf5c03eac39fe83fdb7389a50ffafa4bdd7a56b72f3cadb7a4c16704e0c0d4` |
| DSP build id | `0x4D8B1121` |
| Host executable SHA-256 | PENDING |
| Hexagon SDK | 6.6.0.0, tools 19.0.07 |
| Budget spent | 1 build, 1 session, 4 runs, 0 retries |

The build and the audit are recorded in the worklog §16 and §17. The identity
gate passed before the experiment:

```
build_id=0x4d8b1121 dsp_build_id=0x4d8b1121 build_id_status=MATCH
dsp_identity=MATCH
IDENTITY_OK=PASS
exit_code=0
```

## 2. The four runs

Order is fixed: A(q=6) → B(q=7) → C(q=8) → D(q=9), each once, no retry.

| Condition | q | Fill mask | Label | Observed SHA-256 | Matched reference |
| --- | ---: | ---: | --- | --- | --- |
| A | 6 | 1 | PENDING | PENDING | PENDING |
| B | 7 | 7 | PENDING | PENDING | PENDING |
| C | 8 | 3 | PENDING | PENDING | PENDING |
| D | 9 | 5 | PENDING | PENDING | PENDING |

The four registered references are:

| Name | SHA-256 |
| --- | --- |
| `safe` | `c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d` |
| `wlo` | `f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb` |
| `ahi` | `b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc` |
| `both` | `5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d` |

## 3. Stop-condition checklist

Every row is checked against the session evidence. A single failure stops the
session and the remaining conditions are not run (prereg §10).

| # | Condition | Result |
| ---: | --- | --- |
| 1 | any RPC status other than 0, except the documented cleanup case | PENDING |
| 2 | `resultsLen != 1024`, `metadataLen != 32`, or a slot outside 0..65535 | PENDING |
| 3 | build id mismatch (identity echo vs run echo vs stamped) | PENDING |
| 4 | schema mismatch or selector mismatch, or q outside 6..9 | PENDING |
| 5 | count mismatch: mxclracc 1, MMA 3, Convert 1, Store 1, pack 4, bias 2 | PENDING |
| 6 | store target != `+8192`, out fill != `+8192`, or a Wlo/Ahi fill address that does not match the condition | PENDING |
| 7 | any operand FNV mismatch | PENDING |
| 8 | fill mask / word / form mismatch, including a C/D swap or B missing a fill | PENDING |
| 9 | any guard digest, status, first-bad-byte or method mismatch | PENDING |
| 10 | a `metadata[30]` cleanup error | PENDING |
| 11 | the issued order differs from §3.1, or an access event is missing, duplicated, out of bounds or before the last MMA read | PENDING |
| 12 | any disassembly audit condition fails or cannot be resolved | **PASS** (four conditions, worklog §16) |
| 13 | the observed array matches none of the four references | PENDING |

## 4. Evidence

Raw evidence lives beside the session, not in the repository:

| File | Content |
| --- | --- |
| `00-identity.txt` | identity call stdout |
| `NN-condition-X-qN.txt` | that run's full stdout |
| `NN-condition-X-slots.csv` | the 1,024 result slots, verbatim |
| `NN-condition-X-meta.csv` | the 32 metadata words, verbatim |

The session log records the skel and host digests, the UTC timestamps, the model
and the label of each run. PENDING: the archive location.

## 5. Reading

PENDING. The registered classification order is contract → numeric, and the
terminal classes are:

```text
condition A: safe -> D8_CONTROL_SAFE; otherwise D8_CONTROL_ANOMALY
condition B: safe -> D8_NO_VIOLATION_OBSERVED; wlo/ahi/both -> *_SUSPECT;
             otherwise D8_INCONCLUSIVE_NUMERIC
condition C: safe -> D8_NO_VIOLATION_OBSERVED (Wlo); wlo -> D8_VIOLATION_WLO_SUSPECT
condition D: safe -> D8_NO_VIOLATION_OBSERVED (Ahi); ahi -> D8_VIOLATION_AHI_SUSPECT
```

**No terminal class carries `PROVEN`.** A safe result is one build, one
instruction distance, one run. It does not retire `B3_R2_SYNC_UNPROVEN`, it does
not generalise to all tiles, and it does not bound the race window. A distance
sweep, a full-tile run and any performance measurement are outside the
revision 3 §11 budget.

## 6. What this run can and cannot establish

**Can:** whether the 1,024-cell outcome of this one build, at this one
instruction distance, matches one of the four registered references; and, if it
does not, which operand tile's corruption the outcome is consistent with.

**Cannot:** when the MMA read completed, whether the fill beat it in general,
how wide the race window is, or anything about timing or performance.

## 7. Labels after the run

| Label | Before | After |
| --- | --- | --- |
| `B3_R2_SYNC_UNPROVEN` | unchanged | PENDING (stays unproven either way) |
| `B3_R2_FULL_TILE_CPU_READY` | unchanged | unchanged |
| `CPU_VALIDATED_PREFERRED_DESIGN` | Candidate B | unchanged |
| Candidate C / C2 | `C2 NOT_RUN` | unchanged (separate management) |
| Mode 46 | registered, `NOT_RUN` | PENDING |
| D8 | `NOT_RUN` | PENDING |
