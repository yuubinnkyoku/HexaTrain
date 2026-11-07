# C2 CPU audit record — independent differential, design comparison, evidence contract

Status: **CPU-ONLY AUDIT RECORD**, 2026-11-07. Starting HEAD
`3d4e9f96c2f4fd3126acab2d1691e74c7821b642` (also the remote HEAD at audit start,
PR #14 still Draft).

Scope: everything that can be decided without a device — the C2 decision
procedure and its Python re-implementation, the three-design resource
comparison, and the C2 evidence contract. Prohibited and not performed: ADB,
FastRPC, real DSP, QAIRT, Mode 46, main-branch or training changes.

The audit was run in a separate git worktree at `origin/experiment/…` HEAD so
that the uncommitted changes present in the primary worktree were not touched,
reviewed as input, or committed.

This document records what was **found and fixed**. It is not a ratification of
the previous audit. Its purpose is to shrink the set of questions that still
require hardware, not to promote CPU results to device evidence.

---

## 1. Findings

| # | Finding | Kind | Fix |
| --: | --- | --- | --- |
| F1 | The C2 class enum put `STORE2_PATH_BROKEN` (1) before `COPY_NOT_EXECUTED` (2), while prereg §9 evaluates the per-cell `0xDEAD` residue **before** the X interpretation. The header comment and the evidence contract both claimed the numbering matched §9. Behaviour was unaffected (`c2_classify` uses explicit branches), but the comment told a future implementer to order by enum value — which would reintroduce a real misclassification. | code + doc | Renumbered `COPY=1`, `STORE2=2`; six runtime pins `c2_class_enum_*` added so the invariant cannot drift again |
| F2 | No machine comparison existed between the C classifier and the Python classifier. Two hand-written batteries can both match the doc and still disagree on an unlisted input. | test gap | Added `--c2-replay` to the C model and `hmx_c2_differential_test.py`: 1,818 identical cases through the shipped functions under gcc `-O0`, gcc `-O2`, clang `-O2`; mutation self-check proves the comparator fires |
| F3 | The evidence contract claimed Candidate B's "all six arrays 0/1024 diff", but the crosscheck compared only the two reconstructed arrays. Four of six arrays were unasserted. | test gap + doc | The crosscheck now compares `plain_pos/neg/recon` and `b3_pos/neg/recon`; result `0/1024` diff on every array |
| F4 | The three-design table collapsed two different L2 resources into one fixed-precision number: streaming's 6,144 B is mid-kernel **operand staging**, while B's 2,048 B and C's 10,240/6,144 B are **output-evidence retention**. The numbers are not comparable and must not be read as a memory ranking or a speed proxy. | doc + output | Split into `l2_midkernel_staging` / `l2_output_evidence` in the table and in the machine output |
| F5 | Two documents still described CB-FIX-1 as open after commit `3d4e9f9` closed the CPU contract. | doc | Updated to `CB-FIX-1_CPU_CONTRACT_CLOSED` (CPU model only), with D8 explicitly still device-only |
| F6 | Prereg §2 already lists "zero the untouched bytes … = 0x00", but `c2_model_run` never executed that step: the model's own `memset` supplied the zeros, so the §8 untouched-bytes row could not fail and the CPU check proved nothing about the sequence. A real VTCM allocation is also not guaranteed zero, so the registered attempt could have spent its single budget on allocator residue. | model + doc | The model executes the registered zero-fill over a dirty `0xA5` allocation; fault 28 omits the step and the contract fails |
| F7 | Guard checks were exercised only on the overrun side of the L2 arena, and no fault ever dirtied a VTCM guard. A checker shown only one side can be wrong about the other. | test gap | Faults 29 (L2 underrun) and 30 (VTCM guard content) added |
| F8 | `run_hmx_b3_r2_sim.ps1` did not create `build/`. `build/` is gitignored, so on a clean checkout the simulator's `fopen` failed silently and the Python audit exited 2 (`SKIP`) with no gate failure. | script | Runner creates `build/`; the simulator prints `b3_r2_dump=written|not_written` |
| F9 | The host ABI sizes (2,048 result slots, 32 metadata words) existed only in prose. | model | Constants `HMX_B3_R2_C2_RESULT_SLOTS/_METADATA_WORDS/_SLOT_BYTES/_P_SLOT_BASE/_X_SLOT_BASE` with seven `c2_abi_*` checks |
| F10 | The evidence contract did not require: exact `resultsLen`/`metadataLen`, a defined word-25 enumeration (or an explicit "echo only"), the copy to sit between the two stores in object order, or the prologue zero-fill in the disassembly audit. | doc | All four added to §5.2/§6 |

No registered expected value, digest, map offset or authorisation line was
changed. The check count moved 12,784 → 12,817 (+33) and the fault battery
27 → 30.

---

## 2. §9 correspondence table (prereg text / C / Python)

Run-level procedure, in the pinned order:

| # | Prereg §9 rule | C `c2_classify` | Python `classify` | Agreement |
| --: | --- | --- | --- | --- |
| 0 | contract gate first (`RPC`/Build-ID/counts/Rs/guards/untouched/map/weight tail) | `!contract_ok → CONTRACT_FAILED` | same | yes |
| 1 | ANY cell `0xDEAD` → `C2_COPY_NOT_EXECUTED` (partial included) | `deadic_cells > 0` | `dead > 0` | yes |
| 2 | `X` ≠ expected negative → `C2_STORE2_PATH_BROKEN` | `x_mismatch_cells > 0` | `x_mismatch > 0` | yes |
| 3 | C1 signature **and** non-zero C2 evidence → `C2_CAUSE_INCONCLUSIVE` | `c1>0 && c2_nonzero>0 → MIXED` (same label) | same | yes |
| 4 | C1 alone → `C1_STORE_VISIBILITY_SUSPECT`; with any zero/ambiguous/unexplained → `INCONCLUSIVE` | `c1>0` then `ambig‖zero_loss‖c2_zero‖unknown` | same | yes |
| 5 | non-zero C2 evidence → `C2_ORDERING_VIOLATED` | `c2_nonzero > 0` | same | yes |
| 6 | only ambiguous subnormal zeros → `C2_SUBNORMAL_OR_ZERO_LOSS` | `ambig>0` and the other three zero classes are 0 | same | yes |
| 7 | any zero-agreement / payload-loss / unexplained → `INCONCLUSIVE` | `c2_zero‖zero_loss‖unknown` | same | yes |
| 8 | all 1,024 cells the positive payload → `SUPPORTED` | `payload_match == 1024` | `match == 1024` | yes |
| 9 | otherwise → `INCONCLUSIVE` | final fallthrough | same | yes |

Cell-level §6 rows (first match wins):

| Order | Condition | C | Python | Agreement |
| --: | --- | --- | --- | --- |
| 1 | `0xDEAD` | dead | dead | yes |
| 2 | `0xA5A5` | C1 | C1 | yes |
| 3 | expected positive payload | match (checked before the zero rows, so a both-zero cell counts as a match) | same | yes |
| 4 | zero, positive payload subnormal, negative zero | ambiguous | same | yes |
| 5 | zero, positive payload normal, negative zero | zero-agreement (weak) | same | yes |
| 6 | zero, positive non-zero, negative non-zero | payload loss | same | yes |
| 7 | negative payload, and `pos != neg` | C2 signature (the `!=0` is implied by reaching this row with `p != 0`) | same | yes |
| 8 | anything else | unexplained | same | yes |

The one ordering **divergence** that the audit found was not in either
implementation but in the enum's numeric order (F1). It is now pinned.

---

## 3. Differential design and result

`hmx_c2_differential_test.py`:

1. derives the registered Plain payloads from the digest-pinned operands with
   its own arithmetic and refuses to continue unless both digests match;
2. computes the Python answer for every case **before** any C process runs;
3. writes the cases to `build/c2_differential_cases.txt`;
4. runs the shipped `c2_observe()`/`c2_classify()` through
   `hmx_b3_r2_full_tile_sim --c2-replay <file>`;
5. compares class streams; saves JSON repros on mismatch;
6. mutates one C answer and requires the comparator to detect it.

Case population:

| Kind | Count | What it covers |
| --- | --: | --- |
| Byte-level observations | 282 | the fixed battery, every §9 branch, boundary values, multi-fault combinations, 256 deterministic-seed mixtures across 5 profiles |
| Exhaustive counter combinations | 1,536 | all 2⁹ binary combinations at `match=0`; all 2⁸ combinations at `match=1,023` and at `match=1,024`; the same sweep with `contract_ok=0` |
| **Total** | **1,818** | |

Result (2026-11-07):

```text
c2_diff_impl=gcc-O2 cases=1818 mismatches=0 answers=1818
c2_diff_impl=gcc-O0 cases=1818 mismatches=0 answers=1818
c2_diff_impl=clang-O2 cases=1818 mismatches=0 answers=1818
c2_diff_mutation_detected=1 case=1 C2_ORDERING_VIOLATED->C2_CAUSE_INCONCLUSIVE
c2_diff_checks=5454 fail=0 implementations=3
c2_diff_verdict=C2_DIFFERENTIAL_PASS
```

Python class distribution over the 1,818 cases (every terminal class present):
`C2_PREREG_CONTRACT_FAILED` 514, `C2_COPY_NOT_EXECUTED` 568,
`C2_STORE2_PATH_BROKEN` 326, `C1_STORE_VISIBILITY_SUSPECT` 42,
`C2_ORDERING_VIOLATED` 100, `C2_SUBNORMAL_OR_ZERO_LOSS` 6,
`C2_CAUSE_INCONCLUSIVE` 260, `C2_MINIMAL_SEQUENCE_SUPPORTED` 2.

Limits of this result, stated so it is not over-read:

- It shows the two implementations read the **registered text** the same way.
  It cannot show the text models the device correctly.
- `C2_MINIMAL_SEQUENCE_SUPPORTED` in the histogram is the CPU nominal rendering
  (`C2_CPU_NOMINAL_MATCH` at the label layer); it is not device evidence.
- The differential is not wired into `verify.ps1` as a Python-only gate; it runs
  inside `scripts/run_hmx_b3_r2_sim.ps1`, which the Fast/Host HMX gate invokes.
  A machine without `python` reports `SKIP`, never `PASS`.

---

## 4. Three-design comparison (verified numbers)

Sources: `hmx_b3_r2.h` layout constants and the three models' own counters; the
`design_compare` line is printed by every run.

Common: all three do 4 MMA, 4 Convert, 4 Store, and 2 HVX reconstruction loops
for the Plain+B3 run, and all three produce the same six 1,024-cell arrays.

| Item | Streaming (registered) | Candidate B (phase reuse) | Candidate C (single OUT) |
| --- | --- | --- | --- |
| VTCM peak | 10,752 B (fresh act/wgt/pos/neg/recon + 2 bias) | 10,752 B peak; 8,704 B during the MMA phase | 10,752 B (4 operands + 1 shared OUT + 2 bias) |
| VTCM free margin | 1,536 B | 1,536 B | 1,536 B |
| L2 **operand staging** | 6,144 B (3 tiles; the run does not fit without it) | 0 | 0 |
| L2 **output evidence** | 0 in the model (outputs harvested from VTCM) | 2,048 B (1 tile: Plain recon, load-bearing across the B3 prologue) | 10,240 B raw (5 tiles) / 6,144 B folded (3 tiles) |
| Mid-kernel operand transfers | 3 tiles / 48 vectors / 6,144 B | 0 | 0 |
| HVX evidence copies | 0 | 16 vectors | 80 vectors |
| Operand rewrite between MMAs | yes, 2 regions | no | no |
| Output alias model | 3 fresh buffers | out_neg→Wlo, out_recon→Ahi after last MMA | one OUT rewritten in place |
| CPU numerical parity | registered arrays | **all six arrays asserted identical, 0/1024 diff each** | all six arrays asserted identical |
| Fault battery | faults 1–14 | 15–17 + CB-FIX-1 trace controls | 18–22 |
| Device evidence | none for the full tile; Mode 45 covers the C1 direction | none | none; C2 `NOT_RUN` |
| Design-specific unproven | `B3_R2_SYNC_UNPROVEN` + D1/D2/D3 | D8 (HMX read completion) | C2, C4, C5 |

What the numbers do **not** say:

- No cycle, byte/s, latency or speed-up figure is measured or implied. The CPU
  model counts events; it does not time a device.
- The L2 staging and L2 evidence columns are different resources and must not be
  combined into one "L2 usage" number.
- Candidate C's 10,240 B is a larger **evidence set**, not 8 KiB of extra cost
  for the same evidence: C must retain `plain_pos`/`plain_neg` raw because its
  `OUT` is rewritten three times per subcase, while B keeps those tiles in VTCM
  and only has to save the Plain recon.

**Ordering for a first device trial: B > streaming > C.** B keeps the smallest
L2 footprint, needs no mid-kernel operand transfers, has the smallest
design-specific unproven set after CB-FIX-1's CPU closure, and its six-array CPU
parity is now actually asserted. Streaming remains the registered baseline but
carries the operand-streaming hazards. C carries the C2 hardware question plus
the highest complexity and the largest evidence-retention requirement.
`CPU_VALIDATED_PREFERRED_DESIGN` for B is unchanged and is **not** a performance
verdict.

---

## 5. C2 evidence-contract static audit

| Item | Check performed | Result |
| --- | --- | --- |
| VTCM map | 7 declared regions, pairwise overlap 0; all offsets 128 B multiples; `X` at `+6144` (2 KiB multiple); `used=8960`, `margin=3328` | PASS |
| VTCM guards | 2 × 128 B adjacent to `X` (`+6016`, `+8192`), both clear of `wgt`; content check on `0x5A5A`; overrun/under-dirty exercised by faults 27/29/30 | PASS |
| Written-range ledger | 7 prologue fills = 6,912 B, disjoint, each equals its declared region | PASS |
| L2 arena | 2,560 B = 256 B slack + low guard (256..383) + `P` (384..2431) + high guard (2432..2559); all 128 B multiples; 16 copy vectors tile `P` exactly up to the high guard | PASS |
| Untouched bytes | 5,376 B = 1,920 B + 128 B in-extent gaps + 3,328 B tail; prereg §2's zero-fill is now actually executed by the model over a dirty allocation; fault 28 if omitted | PASS (CPU), device disassembly still required |
| FastRPC sizes | 2,048 result slots = 8,192 B, 32 metadata words = 128 B, total 8,320 B; exact-length assertion added to the host contract | PASS (constants), not implemented |
| Observation ↔ classifier | slots 0..1023 = `P`, 1024..2047 = `X`; decoded cell order `s*32+o`, zero-extended u16; classifier consumes the same two arrays; P-only decode swap measured at 0 coincidental C2 matches | PASS |
| Digests | 8 named / 7 distinct; all six printed values unchanged before and after the audit; independent re-derivation matches the two Candidate-C digests | PASS |
| Evidence order | `P` then `X` then guards/bias; raw bytes preserved before `rpcmem_free`; failure classes per §9 | PASS (design), not implemented |
| Disassembly plan | 8 original conditions + 2 added: the copy must lie between the two stores in object order, and the prologue zero-fill must be present | plan complete, nothing to execute |

Remaining **device** questions, deliberately not closed here:

1. whether an HVX read observes an immediately preceding MX store (C1/C3);
2. whether a later MX store can overtake that read (C2 — the experiment);
3. whether HMX operand reads complete before a later write to the same bytes
   (D8);
4. whether the allocator accepts the `+6016` guard offset and the 128 B gaps;
5. whether the sequence survives optimisation into the registered object order;
6. whether the 0x00 untouched-byte row holds with the DSP's own zero-fill.

---

## 6. Verification runs

| Run | Command | Result |
| --- | --- | --- |
| C model, gcc 15.2.0 `-O2` | `scripts/run_hmx_b3_r2_sim.ps1` | `b3_r2_checks=12817 fail=0`, `fault_injection=PASS detected=30/30` |
| C model, gcc 15.2.0 `-O0` | direct compile | identical count, digests and 30/30 |
| C model, clang 22.1.8 `-O2` | runner clang leg | exit 0, identical stdout for registered lines |
| Python independent audit | `hmx_c2_independent_classifier_test.py` | `checks=67 fail=0`, `C2_INDEPENDENT_AUDIT_PASS` |
| C/Python differential | `hmx_c2_differential_test.py` (gcc `-O0`/`-O2`, clang `-O2`) | `c2_diff_checks=5454 fail=0`, mutation detected |
| Digest immutability | all runs | six SHA-256 values equal the pre-registered constants |
| `verify.ps1 -Profile Fast` | `scripts/verify.ps1 -Profile Fast` | **PASS** (PASS=10 FAIL=0 SKIP=21; the first attempt failed only on `git-diff-check` for one trailing space in the new fault table, fixed before the passing run) |
| `verify.ps1 -Profile Host` | `scripts/verify.ps1 -Profile Host` | **PASS** (see the PR comment for the recorded totals) |

No expected value, decision criterion or registered hash was relaxed to obtain
any of these results.

---

## 7. CPU-closed vs device-open

**Closed on CPU (this audit):**

- the C and Python decision procedures agree on every branch and every
  boundary, mechanically and repeatedly;
- the enum numbering now encodes the §9 order and is pinned;
- Candidate B's six-array numerical parity is asserted, not claimed;
- the L2 guard is checked on both sides and the VTCM guard by content;
- the untouched-bytes contract is load-bearing rather than an artefact of the
  harness;
- the host ABI sizes and the slot/tile correspondence are machine-checked;
- the evidence contract has exact-length, enumeration and disassembly-order
  requirements for the future implementation.

**Still device-only / not run:**

- C1/C3 read-after-store observation in the shared-buffer context;
- C2 read-before-store ordering (the registered experiment);
- D8 HMX asynchronous operand-read completion;
- `B3_R2_SYNC_UNPROVEN`;
- Mode 46 (`NOT_RUN`, unregistered);
- the full-tile 32×32 numerical accuracy and performance on hardware;
- training integration;
- allocator acceptance of the revision 2 VTCM map.

---

## 8. Preconditions for the next device experiment

1. User approval of prereg revision 2 §11 (currently not granted).
2. A registered trigger mode (the contract reserves Mode 46; `NOT_RUN`) with the
   exact sequence — this is the missing implementation, not the design.
3. A stamped DSP build whose disassembly satisfies the ten conditions of the
   evidence contract §6, including copy-between-stores and the prologue
   zero-fill.
4. APK audit, pinned QAIRT root and Build ID, single endpoint, Tier 2 safety
   conditions, `focus_takeover_count=0`.
5. Host-side exact-length checks (`resultsLen == 2048`, `metadataLen == 32`) and
   preservation of raw bytes before `rpcmem_free`.
6. No retry budget: one attempt, and any contract failure stops without numeric
   interpretation.

Nothing in this document authorises any of that.

---

## 9. Reproduction

```powershell
# CPU only. Prepending the MSYS2 UCRT64 gcc/g++ is required on this machine.
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
.\scripts\run_hmx_b3_r2_sim.ps1
python host_tests\hmx_acc_residual_rpc\hmx_c2_independent_classifier_test.py
python host_tests\hmx_acc_residual_rpc\hmx_c2_differential_test.py
```

The runner creates `build/`, runs the C model under gcc and (when the MSVC
environment is available) clang, then both Python audits. `build/` contents are
gitignored and are not evidence.

Cross-references:

- `docs/hmx-v81-cand-c-c2-device-prereg.md` — decision procedure (§9), map
  (§3.1), registered facts (§8), reproduction (§13), unresolved (§14)
- `docs/hmx-v81-cand-c-c2-device-evidence-contract.md` — classifier audit (§3),
  DSP contract (§4), host contract (§5), disassembly plan (§6), comparison (§7),
  unproven (§8)
- `docs/hmx-cb-fix-1-cpu-access-audit.md` — Candidate B CPU access contract
