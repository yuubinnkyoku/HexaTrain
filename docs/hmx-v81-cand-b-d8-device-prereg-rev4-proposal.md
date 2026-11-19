# Candidate B — D8 Device Preregistration (Revision 4, Registered)

Status: **`D8_REV4_REGISTERED`**, 2026-11-19. The owner approved the budget in
§4 with one condition: settle explicitly whether the prologue needs a fence
between its vector stores and the scalar reads that hash them. That is settled
in §1.1 change 2, and the build was re-stamped as `0x4D8B1123`.

This document authorises exactly the budget in §4 and nothing else.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md` (registered revision 3)
- `docs/hmx-v81-cand-b-d8-evidence-contract-rev3-proposal.md`
- `docs/hmx-v81-d8-fastrpc-failure-contract.md`
- `docs/hmx-v81-d8-device-run-2026-11-19.md` (what revision 3 actually produced)
- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (implementation history)

## 0. Why a fourth revision exists

Revision 3 was registered and its budget was spent on one session. **The session
produced no numeric D8 evidence**: condition A stopped at the handler's own
operand-FNV gate before the critical interval, and the result array was 1,024
zeros.

The cause was found offline and is not a V81 HMX property:

1. `build_d8_operands` and `d8_guard_init` issued `vmem` stores **before**
   `qurt_hvx_lock`. `vmem` is an HVX instruction, so those stores did not land.
2. The CPU had never executed the DSP's operand generator, so the four
   registered FNV constants were first compared against the DSP's output at run
   time, on the device, once.

Both are fixed in the working tree. **Revision 4 asks for one more session to
observe the question revision 3 could not put to the hardware.**

## 1. What changed since revision 3

### 1.1 The two source changes

**Change 1 — the HVX lock moves above the vector prologue.**

```text
revision 3:  build_d8_operands -> d8_guard_init -> operand FNVs -> qurt_hvx_lock
revision 4:  qurt_hvx_lock -> build_d8_operands -> d8_guard_init -> operand FNVs
```

**Change 2 — a `qurt_mem_syncht()` fence between the vector stores and the
scalar reads that hash them.**

```text
revision 4:  qurt_hvx_lock
             -> build_d8_operands      (4 x 16 vmem stores)
             -> d8_guard_init          (12 vmem stores)
             -> qurt_mem_syncht()      <- the fence
             -> guard_pre = fnv(guard)
             -> fnv_ahi..wlo = fnv(operands)
             -> mxclracc ...           (the critical interval starts here)
```

`qurt_mem_syncht()` "does not return until all previous memory transactions
(cached and uncached load/store, mem_locked, and so on) that originated from the
current thread are complete and globally observable" (`qurt_memory.h`). That is
exactly the guarantee the read-back needs: the prologue writes VTCM with `vmem`
and then reads the same VTCM with ordinary scalar loads.

**Why both changes, and not one assumed to cover the other.** The revision 3
device run could not distinguish them. The guard hashed to the registered
zero-guard value, but `d8_guard_init` writes zeros, so a read that misses the
store also sees zeros. The operand FNVs hashed to their untouched contents,
which the lock bug alone explains. Rather than assume the lock fix covers the
visibility question, both are fixed and both are recorded.

**Neither change touches the registered issued order.** Section 3.1 lists
`PACK`, `BIAS`, `GUARD INIT`, `GUARD PRE`, `MXCLRACC`, `MMA`, `FILL`, `CONVERT`,
`STORE`, `GUARD POST`. `qurt_hvx_lock` is not an entry of that order; it is a
precondition for issuing any of them. The fence sits in the prologue, **outside**
the registered synchronisation ban, which covers `[mxclracc, Store #1]`. Taking
the lock first and fencing the prologue therefore change no registered target.

The emitted object confirms the placement rather than assuming it: `syncht` is at
segment 60 and `mxclracc` at segment 175, so the fence is 115 segments before the
interval opens.

### 1.2 What did NOT change

| Registered target | Status |
| --- | --- |
| issued-operation order (18 entries) | **unchanged** |
| VTCM offsets `+0/+2048/+4096/+6144/+8192/+10240/+10496/+10752` | **unchanged** |
| fill form `16 x 128 B`, mask A=1 B=7 C=3 D=5 | **unchanged** |
| Convert `Rs = 0x0001`, Store target `+8192` | **unchanged** |
| synchronisation ban in `[mxclracc, Store]` | **unchanged** |
| guard init `12 x 128 B`, no write after the init | **unchanged** |
| four reference SHA-256 values | **unchanged** |
| four operand FNV values | **unchanged** |
| metadata layout, schema `0x0D8A5003` | **unchanged** |
| classification order and terminal labels | **unchanged** |
| fail-closed policy | **unchanged** |

`hmx_d8_contract.h` is untouched.

### 1.3 The CPU gap is closed

`host_tests/hmx_acc_residual_rpc/probe_d8_fixture_cpu.c` now runs the DSP's
operand generator on the CPU and compares against the registered constants. It is
part of `run_hmx_d8_sim.ps1`, so **the fixture is checked before any device
contact**:

```text
d8_fixture ahi cpu=c6231266 registered=c6231266 match
d8_fixture alo cpu=3b2d22d2 registered=3b2d22d2 match
d8_fixture whi cpu=8b88609f registered=8b88609f match
d8_fixture wlo cpu=a66af695 registered=a66af695 match
d8_fixture_cpu=PASS
```

## 2. The one question (unchanged)

Candidate B keeps four operand tiles resident in VTCM and, after the last MMA,
writes output into two of them. **Did the MMA finish reading an operand tile
before the output phase overwrote it?** D8a = Wlo, D8b = Ahi. The numeric
consequence of a lost read is what is observed. Nothing about the MX pipeline
internals, and nothing about "program order equals completion order", is assumed.

## 3. Conditions (unchanged)

A(q=6) → B(q=7) → C(q=8) → D(q=9), each once, in that order, no retry.

## 4. Execution budget requested

| Item | Value |
| --- | --- |
| Builds | 1 clean stamped DSP build |
| Device sessions | **1** |
| Runs | 4 (A, B, C, D) |
| Retries | **0** |
| Timing / performance | none |
| Profile | Tier 2 (`BACKGROUND_CORRECTNESS`) |
| Authorisation | **REQUESTED** |

Approving revision 4 §4 would approve only: 1 build, 1 session, 4 condition runs,
0 retries, under the existing evidence contract. It would not approve a fifth
run, a distance sweep, a full-tile run, a second session, or training
integration.

## 5. What the build must satisfy before the session

All of these are already true for build `0x4D8B1122` in the working tree, and
must be re-established for whatever build is finally stamped:

| Gate | Requirement | State for the stamped build |
| --- | --- | --- |
| Disassembly audit | all four conditions PASS on the stamped object | **PASS** |
| CPU suite | `run_hmx_d8_sim.ps1` PASS, including `d8_fixture_cpu` | **PASS** |
| Identity | `build_id_status=MATCH`, `IDENTITY_OK=PASS` on the device | **PASS** |
| Digest | the skel SHA-256 recorded and the device copy verified | **PASS** |

The stamped build for revision 4 is **`0x4D8B1123`**, skel SHA-256
`a8461eb9188f83c5a3e66c0eb08c704eeebe4aa15cfb77e9cbe8df7d5f8cbd8c`.
`0x4D8B1122` is **not** reused: the source changed, so the build id must change.

The feasibility audit's synchronisation check was tightened at the same time. It
used to ban `syncht` anywhere in the handler, which would forbid the prologue
fence; it now bans it inside `[mxclracc, Store #1]` and separately asserts that
the prologue fence exists. The check count went from 186 to 187.

## 6. Stop conditions (unchanged from revision 3 §10)

The thirteen conditions of revision 3 §10 apply verbatim. In addition, revision 4
adds one:

| # | Condition |
| ---: | --- |
| 14 | **the operand FNVs do not match the registered constants** — the handler already stops itself with `metadata[26] = 16`; the session records `D8_CONTRACT_FAILED` and does not retry |

Condition 14 is what revision 3 hit. It is listed explicitly so the next session
does not treat it as an anomaly.

## 7. What revision 4 does not fix

If the operand FNVs now match and the critical interval runs, the run may still
produce a contract failure or an inconclusive numeric result. **Revision 4 does
not add a retry for that.** One session, four runs, no retries.

## 8. Labels if revision 4 is approved

| Label | Value |
| --- | --- |
| `B3_R2_SYNC_UNPROVEN` | **unchanged** — revision 4 does not retire it |
| `B3_R2_FULL_TILE_CPU_READY` | unchanged |
| `CPU_VALIDATED_PREFERRED_DESIGN` | unchanged (Candidate B) |
| Candidate C / C2 | unchanged, `C2 NOT_RUN` |
| Mode 46 | registered, executed twice |
| D8 | `D8_REV4_REGISTERED` if approved |

**No terminal class carries `PROVEN`.** A safe result is one build, one
instruction distance, one run. It does not generalise to all tiles and does not
bound the race window.

## 9. What the owner is being asked

Accept or reject one more session under the budget in §4, with the single source
change in §1.1 and the CPU gate in §1.3 already in the tree.

If rejected, the D8 question stays open with no device evidence, and the record
of the revision 3 attempt stands as `D8_CONTRACT_FAILED`.
