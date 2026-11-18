# PR #14 comment draft — D8 Mode 46 implementation and device session

Status: **final — the session has run, and it produced `D8_CONTRACT_FAILED`.**
This text was posted as a comment on PR #14 on 2026-11-19.

---

## D8 Mode 46 — registered, implemented, audited, executed once: `D8_CONTRACT_FAILED`

**Baseline.** Start `92c408757726d30efd03895c6c489ca14f268c7b` → end `81291824e506ccdd8cf3572029e05f63387cc927` (this commit, which carries the run record).
Work was done in an isolated worktree on `codex/hmx-d8-mode46-20261118`; the main
worktree's uncommitted changes were not touched. This PR stays Draft.

### What changed

| Item | Before | After |
| --- | --- | --- |
| Revision 3 | `D8_REV3_PROPOSED_CPU_ONLY` | **`D8_REV3_REGISTERED`** (owner authorisation 2026-11-18, budget = revision 3 §11) |
| Mode 46 | unregistered | **registered**, implemented, executed once |
| Device approval | `NOT GRANTED` | **GRANTED for the §11 budget only — spent** |
| D8 | `NOT_RUN` | **`D8_CONTRACT_FAILED`** (no numeric evidence) |
| `B3_R2_SYNC_UNPROVEN` | unchanged | **unchanged** |
| C2 | `NOT_RUN` | **unchanged** (separate management) |

**No registered number changed.** The four reference SHA-256 values, the four
operand FNVs, the fill mask table, the issued-order block and the VTCM offsets
are byte-identical to revision 2. `hmx_d8_contract.h` is untouched.

### Implementation

- **DSP** (`hmx_mode46.h`, `hmx_mode46_d8.h`, `probe_dsp.c`): the argument gate
  requires exactly 1,024 result slots and 32 metadata words and accepts q 6..9
  only; the handler reproduces the registered fixture, packs the four tiles,
  initialises the 1,536 B guard, takes the four operand FNVs before MMA #1, then
  runs the registered sequence. Nothing in the critical interval synchronises.
- **Host** (`hmx_mode46_host.h`, `probe_host.c`): the contract gate checks every
  field and reports the first failing one, the observed array is hashed as u16
  little endian and compared with the four references, and the classifier maps
  the result to the registered label. The word `PROVEN` never appears.
- **Cleanup exception**: for mode 46 only, a cleanup failure after execution
  started goes to `metadata[30]` and `run()` returns 0, so the arrays survive.
  Every other mode keeps the existing behaviour.

### Disassembly audit — PASS on the real object

```
d8_disasm_audit=PASS packs=64 guard=12 fills=48 mma=3 convert=1 store=1
```

All four conditions. The audit was extended for real objects (base-relative
resolution, loop expansion, per-segment derivation snapshots); Appendix A of
`docs/hmx-v81-d8-disasm-audit-spec.md` records that **no registered verification
target changed** — only the way the text is read.

**The audit caught two real implementation defects**, which is the reason it is
load-bearing rather than decorative:

1. `bias = mxmem2` was never issued: the payloads were written to memory but
   never loaded into the bias register, so the Convert had no registered bias
   state.
2. At `-O2` the compiler inlined the pack helper and folded the third pack onto
   the out_pos slot: `whi_local` was written to `+8192` and `+4096` was never
   written. Four different spellings of the destination produced the same fold,
   so the fix is `__attribute__((noinline))`, not a rewording.

### CPU regressions

| Suite | Result |
| --- | --- |
| `d8_sim` | PASS |
| rev3 contract self-test | 104 checks, 0 fail |
| independent Python audit | 143 checks, 0 fail |
| mutation check | 5/5 detected |
| disassembly rules (synthetic) | 25 cases, 0 fail |
| host gate and classifier | 121 checks, 0 fail |
| feasibility audit | 186 checks, 0 fail |
| `verify.ps1 -Profile Fast` | PASS (228 s) |

The four reference digests and the four operand FNVs match the registered values.

### Device session

| Item | Value |
| --- | --- |
| Device | NX741J (`PQ85A01`, `canoe`, Android 16) |
| DSP build id | `0x4D8B1121` |
| DSP skel SHA-256 | `56bf5c03eac39fe83fdb7389a50ffafa4bdd7a56b72f3cadb7a4c16704e0c0d4` |
| Identity gate | **PASS** (`build_id_status=MATCH`, `IDENTITY_OK=PASS`) |
| Runs | **1 of 4 attempted** — condition A — 0 retries, budget spent |

The session harness spends the budget exactly once: one identity call, then
A(q=6) → B(q=7) → C(q=8) → D(q=9), each once, with no retry path. It stops
before any experiment run if the identity gate fails, stops on a non-zero run,
stops before any numeric reading if the contract gate fails, and stops before
B/C/D if condition A is not `D8_CONTROL_SAFE`. Its stop logic is itself tested
(14 checks).

### Results

**`D8_CONTRACT_FAILED`. No numeric D8 evidence was obtained.**

Condition A(q=6) ran and stopped at the handler's own operand-FNV gate, before
the critical interval:

| Word | Field | Observed | Registered | |
| ---: | --- | --- | --- | --- |
| 10 | `mxclracc` | `0` | 1 | fail |
| 11 | MMA | `0` | 3 | fail |
| 12 | Convert | `0` | 1 | fail |
| 13 | Store | `0` | 1 | fail |
| 14 | pack | `4` | 4 | ok |
| 15 | bias | `2` | 2 | ok |
| 16 | fill | `0x00000000` | mask 1 | fail |
| 17 | address | `0x00000000` | `0x1000` | fail |
| 19 | FNV Ahi | `0x57A0EC09` | `0xC6231266` | fail |
| 20 | FNV Alo | `0xD0E0C444` | `0x3B2D22D2` | fail |
| 21 | FNV Whi | `0x9C9FE034` | `0x8B88609F` | fail |
| 22 | FNV Wlo | `0x29A90A5C` | `0xA66AF695` | fail |
| 23 | guard pre | `0xF5EBD5C5` | `0xF5EBD5C5` | **ok** |
| 24 | guard post | `0xF5EBD5C5` | `0xF5EBD5C5` | **ok** |
| 26 | failure | `16` (FNV) | 0 | the gate itself |

The result array is **1,024 zeros**; its observed SHA-256
`e5a00aa9991ac8a5ee3109844d84a55583bd20572ad3ffcd42792f3c36b183ad` is exactly the
SHA-256 of 2,048 zero bytes, verified independently. `first_bad=mxclracc` only
because that is the first field the contract checks in order.

**This is fail-closed working on real hardware.** The handler packed the four
tiles, loaded both bias sets, initialised the guard correctly (guard pre and post
both equal the registered zero-guard FNV), found the operand bytes did not match
the registered fixture, set `metadata[26] = 16`, and returned without issuing the
sequence. Without that gate the run would have produced numbers that looked like a
D8 outcome but were an artefact of a wrong operand set.

**So the run answers nothing about MMA operand reuse.** B/C/D were not run and
there was no retry, per the registered budget.

### Cause

The guard FNV is correct, so the FNV function and the VTCM read path work; the
difference is in the operand bytes.

**Primary: `build_d8_operands` has never run on the CPU.** It is defined in
`hmx_mode46_d8.h`, referenced only from that header, and the only file that
includes the header is `probe_dsp.c`, which is DSP-side. The CPU suites verify the
contract checker, the classifier and a separate host model of the arithmetic;
they never produce the DSP's tiles. The four registered FNV constants were first
compared against the DSP's output at run time, on the device, once.

**Alternative the evidence cannot separate:** the four `vmem` stores in
`d8_pack_tile` are followed by scalar byte reads for the FNV with only a compiler
`"memory"` clobber and no data-cache or `syncht` barrier. The guard case cannot
distinguish it, because `d8_guard_init` writes zeros and a read that misses the
store also sees zeros.

Neither is a V81 HMX property, so neither bears on `B3_R2_SYNC_UNPROVEN`.

### Before another attempt

The revision 3 budget is spent, so a second session needs a **new registration**.
The prerequisites are offline:

1. make the DSP operand fixture executable and comparable on the CPU, so the four
   FNV constants are checked before any device contact;
2. decide whether the pack needs an explicit barrier before the scalar FNV read,
   and if so register that as part of the sequence rather than adding it quietly;
3. re-run the disassembly audit, the CPU suites and `verify.ps1 -Profile Fast`
   against the corrected build, and stamp a new build id.

### `B3_R2_SYNC_UNPROVEN`

**Unchanged, and it stays unproven whatever the outcome.** D8 observes the
numeric consequence of a lost read for one build at one instruction distance in
one run. A safe result does not retire the label, does not generalise to all
tiles, and does not bound the race window. A distance sweep, a full-tile run and
any performance measurement are outside the revision 3 §11 budget and were not
performed.

### Scope

No merge, no Draft release, no release, no training integration. The old
`PhoneLM` worktree and its backups were not modified.
