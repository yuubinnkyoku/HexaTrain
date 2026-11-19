# D8 Second Session — Pre-flight Checklist

Status: **`D8_REV4_AWAITING_APPROVAL`**. Nothing here authorises a run. This is
the checklist that must be satisfied immediately before the single session
revision 4 asks for, so that the session is not spent on a preventable failure.

Companion documents:

- `docs/hmx-v81-cand-b-d8-device-prereg-rev4-proposal.md` (the budget request)
- `docs/hmx-v81-d8-device-run-2026-11-19.md` (what revision 3 produced)
- `docs/hmx-v81-d8-fastrpc-failure-contract.md` (S0..S6)
- `docs/agent/device-test-tiers.md` (Tier 2 conditions)

## 1. Build identity

| # | Check | How | State |
| ---: | --- | --- | --- |
| 1.1 | The stamped build is the audited one | skel SHA-256 equals the value the disassembly audit passed on | **`8f73eed5…7a0d`** |
| 1.2 | The device copy is that build | `sha256sum` on the device | **matches** |
| 1.3 | The host executable carries the same build id | `-DHMX_PROBE_BUILD_ID` at compile time | **`0x4D8B1122`** |
| 1.4 | Identity gate passes | `-strict-build-id`, mode 30, q=6 | **PASS**, `build_id_status=MATCH` |
| 1.5 | The disassembly audit passes on this object | four conditions | **PASS** |
| 1.6 | The CPU suite passes | `run_hmx_d8_sim.ps1` | **PASS**, including `d8_fixture_cpu` |

## 2. Device state (Tier 2)

| # | Check | Requirement | State |
| ---: | --- | --- | --- |
| 2.1 | One physical device | stable identifier proven, not an emulator | **proven** |
| 2.2 | One endpoint chosen deterministically | TCP endpoint selected | **done** |
| 2.3 | SELinux | `Enforcing` | **Enforcing** |
| 2.4 | Not rooted | `uid=2000(shell)` | **shell** |
| 2.5 | No active training or probe | no matching process | **none** |
| 2.6 | `focus_takeover_count` | 0 over the run | **to be measured** |
| 2.7 | No system-area change | push only to `/data/local/tmp` | **yes** |

## 3. The trap that cost revision 3 nothing but would cost revision 4

| # | Check | Why |
| ---: | --- | --- |
| 3.1 | **No `libcdsprpc.so` in the probe directory** | a copy there makes the DSP unreachable (`capability_status=-1`). The harness refuses to start if one is present. |
| 3.2 | `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` point at the probe directory | the vendor `libcdsprpc` must be the one that loads |

## 4. Harness

| # | Check | State |
| ---: | --- | --- |
| 4.1 | `d8_session.sh` deployed and executable | **yes** |
| 4.2 | `sh -n` syntax clean | **yes** |
| 4.3 | Stop logic tested | **14 checks PASS** |
| 4.4 | No build id hardcoded | **confirmed** |
| 4.5 | Evidence directory writable | **yes** |

## 5. What the session will do

Exactly once, with no retry path in the script:

1. identity (mode 30, q=6, `-strict-build-id`) — stops before any run if it fails;
2. A(q=6) — **stops the session if this is not `D8_CONTROL_SAFE`**, so B/C/D are
   not run;
3. B(q=7), C(q=8), D(q=9) — each once;
4. every run's stdout, the 1,024-slot array and the 32-word metadata are written
   to `evidence/<session-id>/` before anything else.

## 6. What revision 3 hit, and what to expect

Revision 3 stopped at `metadata[26] = 16` (`FNV`): the four operand tiles did not
hash to the registered constants, because the vector prologue issued `vmem`
before `qurt_hvx_lock`. That is fixed.

**If the same failure appears again, the fix did not take and the session stops
there.** It must not be retried: the budget has no retry, and a second attempt
would be an unregistered fifth run.

The CPU now reproduces the generator and matches all four constants, so a repeat
would mean the difference is in the VTCM path rather than the generator — a
different question that needs its own registration.

## 7. What a result would and would not mean

**Would:** whether the 1,024-cell outcome of this one build at this one
instruction distance matches one of the four registered references, and if not,
which operand tile's corruption it is consistent with.

**Would not:** when the MMA read completed, whether the fill beat it in general,
how wide the race window is, or anything about timing or performance.

**No terminal class carries `PROVEN`.** `B3_R2_SYNC_UNPROVEN` stays unproven
whatever the outcome.
