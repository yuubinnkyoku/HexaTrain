# PR #14 comment draft — D8 Mode 46 implementation and device session

Status: **DRAFT — do not post until the four runs have been executed.**
Every field marked `PENDING` is filled from the session evidence.

---

## D8 Mode 46 — registered, implemented, audited; device session PENDING

**Baseline.** Start `92c408757726d30efd03895c6c489ca14f268c7b` → end PENDING.
Work was done in an isolated worktree on `codex/hmx-d8-mode46-20261118`; the main
worktree's uncommitted changes were not touched. This PR stays Draft.

### What changed

| Item | Before | After |
| --- | --- | --- |
| Revision 3 | `D8_REV3_PROPOSED_CPU_ONLY` | **`D8_REV3_REGISTERED`** (owner authorisation 2026-11-18, budget = revision 3 §11) |
| Mode 46 | unregistered | **registered**, DSP and host implemented |
| Device approval | `NOT GRANTED` | **GRANTED for the §11 budget only** |
| D8 | `NOT_RUN` | PENDING |
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
| Runs | PENDING |

The session harness spends the budget exactly once: one identity call, then
A(q=6) → B(q=7) → C(q=8) → D(q=9), each once, with no retry path. It stops
before any experiment run if the identity gate fails, stops on a non-zero run,
stops before any numeric reading if the contract gate fails, and stops before
B/C/D if condition A is not `D8_CONTROL_SAFE`. Its stop logic is itself tested
(14 checks).

### Results

PENDING

The four registered references the observed array is compared against, taken
verbatim from `hmx_mode46_host.h`:

```text
safe c9abf30e3b440ddaa2e21bf0b0d327195c7eb8d3f23c5960abaa7debea20203d
wlo  f0b3b4aa71b09377a826f2a6d98903d4aa56fdc45911c373057d3fe69881f6fb
ahi  b6328e88b4f9be50e6ace56b9a1db552a1ce3873f5d1f1a3016781cfe0760fcc
both 5285b463b94e16aac99635ae5a706c286665249be731292f8ad12449de32535d
```

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
