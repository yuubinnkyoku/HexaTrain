# Candidate B — D8 Device Preregistration (Revision 5 Proposal)

Status: **`D8_REV5_PROPOSED`**, 2026-11-21. This document authorises nothing.
It is a proposal for the owner to accept or reject.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev4-proposal.md` (registered revision 4)
- `docs/hmx-v81-cand-b-d8-device-prereg-rev3-proposal.md` (registered revision 3)
- `docs/hmx-v81-d8-device-run-2026-11-19.md` (what revisions 3 and 4 produced)
- `docs/hmx-v81-d8-disasm-audit-spec.md` (Appendix A, including A.12)
- `docs/hmx-d8-mode46-worklog-2026-11-18.md` (implementation history)

## 0. Why a fifth revision exists

Revisions 3 and 4 each spent a session and produced no numeric D8 evidence.
Both stopped at the handler's own operand-FNV gate, with **bit-identical**
results:

```text
d8_fnv ahi=57a0ec09 alo=d0e0c444 whi=9c9fe034 wlo=29a90a5c
```

The cause was found offline and is **not** a V81 HMX property:

`double_to_fp16` calls `frexp`, `ldexp`, `floor` and `fmod`. In the stamped skel
those resolved through `@plt` to the **device's** `libc.so`, while the simulator
linked its own copy. Same compiler, same flags, same instruction sequence,
different library behind it. The CPU and the simulator both reproduce the four
registered constants; only the device does not.

**Revision 5 changes the link, not the source.**

## 1. The change

### 1.1 Static bind with `-Bsymbolic`

```text
revision 4:  hexagon-clang -shared -fPIC -O2 -G0 -mv81 -mhmx -mhvx \
               -mhvx-length=128b -D__HEXAGON_ARCH__=81 \
               -DHMX_PROBE_BUILD_ID=0x4D8B1123 \
               probe_dsp.c hexatrain_hmx_acc_residual_skel.c libqurt.a

revision 5:  hexagon-clang -shared -fPIC -O2 -G0 -mv81 -mhmx -mhvx \
               -mhvx-length=128b -D__HEXAGON_ARCH__=81 \
               -DHMX_PROBE_BUILD_ID=0x4D8B1124 \
               -Wl,-Bsymbolic \
               probe_dsp.c hexatrain_hmx_acc_residual_skel.c \
               libc.a libgcc.a libqcc.a libqurt.a
```

**Link order matters**: `libqurt.a` must come last so the `sys_*` symbols that
`libc.a` needs resolve. Putting it earlier leaves twelve `sys_*` entries
undefined.

**`-Bsymbolic` is not optional.** Linking `libc.a` alone clears the UND entries
but leaves the calls going through `@plt`, so the device's dynamic linker can
still preempt them. `-Bsymbolic` binds the calls locally, which is what makes the
static definition the one that runs.

### 1.2 What did NOT change

| Registered target | Status |
| --- | --- |
| issued-operation order (18 entries) | **unchanged** |
| VTCM offsets `+0/+2048/+4096/+6144/+8192/+10240/+10496/+10752` | **unchanged** |
| fill form `16 x 128 B`, mask A=1 B=7 C=3 D=5 | **unchanged** |
| Convert `Rs = 0x0001`, Store target `+8192` | **unchanged** |
| synchronisation ban in `[mxclracc, Store #1]` | **unchanged** |
| guard init `12 x 128 B`, no write after the init | **unchanged** |
| four reference SHA-256 values | **unchanged** |
| four operand FNV values | **unchanged** |
| metadata layout, schema `0x0D8A5003` | **unchanged** |
| classification order and terminal labels | **unchanged** |
| fail-closed policy | **unchanged** |
| **source files** | **unchanged** |

`hmx_d8_contract.h` has no diff. The `<!-- d8-rev3-sequence:begin -->` block has
no diff. `git diff` against the base commit shows zero hits on the registered
numbers.

### 1.3 The handler is byte-identical

`run_mode46_d8_one_trial` has **280 segments in both builds**. Comparing the
instruction sequences after normalising addresses and symbol names gives
**zero differing lines**. The link changed addresses, not code.

### 1.4 `.dynsym` still exports the math symbols — this is expected

`frexp`, `ldexp`, `floor`, `fmod` and the `divdf3` family remain `DF .text`
entries in `.dynsym`. **This is not a defect and must not be "cleaned up".**
Because `-Bsymbolic` bound every call site locally, no call goes through the PLT,
so the dynamic linker never resolves them. Removing the exports would change
nothing and risks breaking the link.

## 2. The one question (unchanged)

Candidate B keeps four operand tiles resident in VTCM and, after the last MMA,
writes output into two of them. **Did the MMA finish reading an operand tile
before the output phase overwrote it?** D8a = Wlo, D8b = Ahi.

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

## 5. Pre-declared interpretation

**This is declared before the run and the results are read only against it.**

| Outcome | Reading |
| --- | --- |
| **A passes the FNV gate** | The libm hypothesis is confirmed. B/C/D run and the D8 question is put to the hardware for the first time. |
| **A stops at `metadata[26] = 16`** | The libm hypothesis is **refuted**. Preemption is impossible and the instruction sequence is identical, so the difference is elsewhere. The approved diagnostic dump (USR capture plus pre-conversion doubles) runs next, with no further approval needed. |

No other reading of the results is permitted.

## 6. Gates that must pass before the session

| Gate | Requirement | State |
| --- | --- | --- |
| Disassembly audit | all four conditions PASS on the stamped object | **PASS** |
| Link binding | `math_plt=0`, `math_und=0`, `und=17/17` | **PASS** |
| CPU suite | `run_hmx_d8_sim.ps1` PASS, including `d8_fixture_cpu` | **PASS** |
| `verify.ps1 -Profile Fast` | all PASS | **PASS** (174.0 s, PASS=10 FAIL=0) |
| Identity | `build_id_status=MATCH`, `IDENTITY_OK=PASS` | **pending** — runs at deploy time, before the session, and consumes no budget (§7) |

The stamped build is **`0x4D8B1124`**, skel SHA-256
`8b8e4378681043c4e323f41544653dcdf75d841c25282bdd51e1ce86b8d702aa`,
255,776 bytes. `0x4D8B1123` is **not** reused.

## 7. Deployment order

Per the established precedent: **deploy → digest check → identity run** (which
consumes no budget) → confirm `build_id_status=MATCH` → then
`sh d8_session.sh <session-id>` exactly once.

**If identity fails, stop and report. The session is not run.** A statically
linked libc could fail in the load-time init path, and identity is where that
would show.

## 8. Stop conditions

Conditions 1..14 of revision 3 §10 and revision 4 §6 apply verbatim, including
condition 14 (operand FNV mismatch is a normal failure, not an anomaly).

## 9. Labels

| Label | Value |
| --- | --- |
| `B3_R2_SYNC_UNPROVEN` | **unchanged** whatever the outcome |
| `B3_R2_FULL_TILE_CPU_READY` | unchanged |
| `CPU_VALIDATED_PREFERRED_DESIGN` | unchanged (Candidate B) |
| Candidate C / C2 | unchanged, `C2 NOT_RUN` |
| Mode 46 | registered, executed twice |
| D8 | `D8_CONTRACT_FAILED` until a run produces numeric evidence |

**No terminal class carries `PROVEN`.** A safe result is one build, one
instruction distance, one run.

## 10. What the owner is being asked

Accept or reject one more session under §4, with the link-only change in §1.1,
the pre-declared interpretation in §5, and the diagnostic dump held in reserve
per §5.
