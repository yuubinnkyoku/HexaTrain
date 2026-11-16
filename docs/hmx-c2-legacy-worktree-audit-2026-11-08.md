# Legacy uncommitted-worktree audit and selective integration (2026-11-08)

Status: **AUDIT RECORD**, CPU-only. Base `e520c4d199a7b63199b1be7db3d5ce0873b1362d`
(PR #14 CPU-ready head). The long-lived uncommitted changes in the original
worktree were preserved, compared against the committed independent audit, and
selectively adopted. Nothing was discarded from the original worktree and no
committed expected value, digest or evidence contract was changed.

The original worktree was used read-only. Integration happened in a separate
worktree (`HexaTrain-integration-20261108`, branch
`codex/hmx-c2-integration-20261108`).

## 1. Preservation record

Original worktree: `D:\ghq\github.com\yuubinnkyoku\PhoneLM`
HEAD at capture: `3d4e9f96c2f4fd3126acab2d1691e74c7821b642`
Branch: `experiment/hmx-acc-residual-feedback` (upstream 3 commits behind at
capture; the new commits were not applied to this worktree)
Backup: `D:\ghq\github.com\yuubinnkyoku\PhoneLM-untracked-backup-20261108-0130`

Eight paths were modified or staged. Four are new (absent from HEAD):

| Path | Status | Index blob | Worktree SHA-256 |
| --- | --- | --- | --- |
| `docs/hmx-c2-cpu-differential-and-design-audit.md` | AM (new, staged; further worktree edits) | `c8bd8b1c…` | `836f1901…` |
| `docs/hmx-v81-cand-c-c2-device-evidence-contract.md` | MM (staged + further worktree edits) | `be2b224a…` | `71de9f9c…` |
| `docs/hmx-v81-cand-c-c2-device-prereg.md` | .M (unstaged only) | = HEAD `a72a0584…` | `0cdc0039…` |
| `host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c` | .M (unstaged only) | = HEAD `4d34dd2a…` | `d8a81d25…` |
| `host_tests/hmx_acc_residual_rpc/hmx_b3_three_design_bits_test.c` | A. (new, staged) | `8721addc…` | `8656a1e0…` |
| `host_tests/hmx_acc_residual_rpc/hmx_c2_diff_driver.c` | A. (new, staged) | `8bdcab4e…` | `2f292aaf…` |
| `host_tests/hmx_acc_residual_rpc/hmx_c2_differential_test.py` | A. (new, staged) | `1538b878…` | `0bd47fd9…` |
| `host_tests/hmx_acc_residual_rpc/hmx_c2_independent_classifier_test.py` | .M (unstaged only) | = HEAD `bacb9204…` | `d19301d4…` |

Saved in the backup directory:

- `status-porcelain-v2.txt`, `ls-files-staged.txt`, `file-hashes.json`;
- `patches/staged-vs-HEAD.patch`, `patches/unstaged-vs-index.patch`,
  `patches/worktree-vs-HEAD.patch`;
- byte-exact `worktree/`, `index/`, `head/` copies of every path;
- `MANIFEST.md` with the restore procedure.

**Restore proof.** A scratch copy of the HEAD files plus both patches
reproduced all eight files byte-for-byte (`RESTORE_ALL_MATCH=True`; every
worktree SHA-256 matched). The patches must be applied with
`core.autocrlf=false`, otherwise CRLF changes the bytes even though the text is
equivalent; this is recorded in the manifest. There were no untracked files.

## 2. Same-name file: old vs committed differential

The old `hmx_c2_differential_test.py` exists in one version only (index and
worktree identical, blob `1538b878…`). It ran against a separate adapter,
`hmx_c2_diff_driver.c`, which includes the simulator source with
`#define main hmx_full_tile_sim_original_main`.

It was executed from a scratch copy (not the original worktree):
`c2_differential=PASS fixed=16 deterministic_seeds=1024 comparisons=1040`.

| Criterion | Legacy (uncommitted) | Committed at `e520c4d` | Adopted change |
| --- | --- | --- | --- |
| Same-input mechanism | separate driver binary; line protocol, one case per line | `--c2-replay` mode of the shipped simulator; text case file | keep the committed version (no second binary, no `#define main` hack) |
| Expected-value independence | Python `classify()`; fixed cases also carry a hard-coded label | Python answers computed before C runs; no label pins | added optional `expect` pins and census `probe`s |
| §9 coverage | 16 fixed cases + 1,024 seeds | 284 observations + 1,536 exhaustive counter combinations | added the legacy mixed/single-C2 cases; kept the exhaustive sweep |
| Multi-fault detection | random combination of up to 11 cells, random contract/X | 256 profile-based mixtures + exhaustive counts | kept committed; legacy distribution not adopted |
| Deterministic randomness | 1,024 arbitrary-edit seeds | 256 profile seeds | kept 256 (diminishing returns; the counter space is exhaustive) |
| Mutation / miss detection | compile-time hook `HMX_C2_MUTANT_MIXED_AFTER_C1`, manual build; class-corruption self-check absent | in-memory class-corruption self-check | adopted the mutation *idea* as an automated two-mutant regression; did not adopt the hook |
| Exit codes | 1 mismatch, 2 missing dump/driver | 1 mismatch/pin, 2 environment | kept committed; label/coverage pins also exit 1 |
| Dump-generation failure | `py.read_dump` exits 2 | same, plus the runner now creates `build/` | kept committed (the silent-SKIP fix is in `e520c4d`) |
| Registered-value dependence | re-derives and checks both Candidate-C digests | same | unchanged |
| Failure minimisation | greedy single-cell shrink, writes `c2_diff_mismatch.json` | full-case JSON repro, no shrink | **not adopted** (diagnostic convenience; cannot be exercised without a real mismatch, so it would be untested code) |

## 3. Adopted (with evidence)

### 3.1 Mixed-signature priority

Section 9 is a first-match list and the enum numbering happens to match it, but
no committed case required a tile holding **both** a C1 cell and a non-zero
negative cell to take the mixed branch. Added:

- C: `c2_control_mixed_signal` (exact `HMX_C2_CLASS_MIXED_SIGNATURE`), distinct
  from C1 and C2, not a contract failure, plus
  `c2_control_mixed_counterfactual_reports_c1`.
- Python: `case_mixed_registered_label` and
  `case_mixed_counterfactual_after_c1_reports_c1`, using the rejected-order
  argument `classify(..., mixed_after_c1=True)`.

The counterfactual makes the case discriminating: under the rejected ordering
the same observation answers C1, so the regression cannot silently stop
testing the order.

### 3.2 Payload-loss row in the observer

The registered fixture is mutually exclusive, so no cell has a non-zero
expected positive *and* a non-zero expected negative. `c2_observe`'s
`zero_loss_other` row was therefore unreachable from the array battery; only
the classifier's counter sweep could see the count.

A synthetic expected pair reaches it. The legacy case, however, was **not
discriminating**: on a plain zero-loss state a missing count falls through to
`payload_match != 1024` and still returns `INCONCLUSIVE`, so a class-only
comparison could not fail. This was verified directly: the legacy differential
was run against a scratch mutant of the legacy observer that drops the count,
and it still reported `PASS fixed=16 comparisons=1040`. The adopted case
(`synthetic_c1_plus_zero_loss`) also places a C1 cell in the tile, so a missing
count returns `C1_STORE_VISIBILITY_SUSPECT` and the differential fails (one
mismatch, verified in §3.3 mutant 1).

The synthetic pair is used only as case input; it does not replace any
registered array or digest.

### 3.3 Automated mutation regression

`scripts/run_hmx_c2_mutation_check.ps1` applies two textual mutations to a
scratch copy of the simulator, compiles each, and requires the differential to
detect it:

| Mutation | Result | Detected by |
| --- | --- | --- |
| `c->zero_loss_other++` removed | **1 mismatch** | `synthetic_c1_plus_zero_loss` (`python=INCONCLUSIVE`, mutant=`C1`) |
| mixed test removed | **5 mismatches** | `mixed_signature` + four counter-sweep cases |

If a mutation pattern stops matching the source, the "mutant" equals the
original, the differential passes, and the script fails closed. The legacy
compile-time hook is not carried; the same property is now executed by the gate
instead of by a manual build.

The legacy hook claim was independently reproduced against a scratch build of
the legacy source: `FAIL c2_control_mixed_signal got=4 want=3`,
`b3_r2_checks=12798 fail=2`, exit 1.

### 3.4 Equal-evidence L2 accounting

The legacy audit pointed out that the L2 numbers were compared at unequal
evidence. Adopted into `docs/hmx-c2-cpu-audit-2026-11-07.md`:

| Scenario | Streaming | Candidate B | Candidate C |
| --- | ---: | ---: | ---: |
| 6,144 B staging + 6,144 B Plain evidence | **12,288 B peak** | 0 | 0 |
| Plain evidence only (3 tiles) | 6,144 B | **6,144 B** | 6,144 B |
| B3 output evidence | 0 (VTCM harvest) | 0 (VTCM harvest) | 4,096 B |
| Peak, full six-array evidence | 12,288 B | 6,144 B | 10,240 B |

This is an accounting scenario, not a measured peak. B's registered 2,048 B
save covers only the reconstructed tile; at equal evidence B needs 6,144 B. The
ranking B < C < streaming is unchanged.

### 3.5 Wire-order and sizing clarifications

Adopted into the evidence contract, with an internal inconsistency in the
legacy version fixed:

- `results` now specifies **raw pair-interleaved store-crouton order on the
  wire**, decoded exactly once by the host. The legacy change updated §5.2 but
  left §4 item 8 saying "decoded cell order"; both sections were updated
  together here. The registered expected bits and digests are unchanged.
- The "three orders of magnitude below" sizing claim is replaced by "proposed
  total 8,192 + 128 B; allocation and transport limits unverified until an
  approved implementation is audited". No limit was measured.

## 4. Not adopted / deferred

| Change | Decision | Reason |
| --- | --- | --- |
| `hmx_c2_diff_driver.c` | not adopted | Superseded by the integrated `--c2-replay` mode. A second compile of the whole simulator through `#define main` adds no independent oracle and doubles the build surface. |
| `hmx_b3_three_design_bits_test.c` | not adopted | It compares the same six arrays through the same model functions that the committed in-sim crosscheck now asserts (`cb_fix_1_six_array_parity`, 0/1024 diff). The separate executable was verified to pass, but it is redundant, and it does not test Candidate C's arrays (the comment defers them to the model). |
| Legacy 1,024-seed distribution | not adopted | The committed 256-profile set plus the exhaustive 2⁹/2⁸ counter sweep covers the decision table; more random draws add runtime without a new branch. |
| Greedy mismatch minimiser | deferred | Only runs on a real mismatch and would itself be untested; the full-case JSON repro is sufficient to reproduce. Revisit if a mismatch ever appears. |
| Legacy compile-time mutation hook | not adopted | Replaced by `run_hmx_c2_mutation_check.ps1`, which executes the property on every gate run. |
| Legacy standalone audit document | not merged | Its unique content (equal-evidence accounting, wire order, sizing honesty) is folded into the existing audit/contract documents so there is one source of truth. |
| Legacy count figures (12,790 checks, 81 Python checks, 1,040 comparisons) | not adopted as current numbers | They describe the legacy branch state. The integrated tree reports 12,828 checks, 81 Python checks over 18 fixed cases, and 5,460 comparisons over 1,820 cases. |

## 5. Integration verification

Committed at `c56d4bf` on top of `e520c4d`.

| Gate | Result |
| --- | --- |
| C model, gcc 15.2.0 `-O2` | `b3_r2_checks=12828 fail=0`, `fault_injection=PASS detected=30/30` |
| C model, gcc 15.2.0 `-O0` | identical count and digests |
| C model, clang 22.1.8 `-O2` | exit 0, identical registered lines |
| Python independent audit | `checks=81 fail=0`, `C2_INDEPENDENT_AUDIT_PASS` |
| C/Python differential | 1,820 cases × gcc `-O0`/`-O2` + clang `-O2` = `c2_diff_checks=5460 fail=0` |
| Mutation regression | `c2_mutation_check=PASS mutants=2` |
| Registered digests | six SHA-256 values unchanged |
| `verify.ps1 -Profile Fast` | **PASS** (PASS=10 FAIL=0 SKIP=21, 490.6s) |
| `verify.ps1 -Profile Host` | **PASS** (PASS=2 FAIL=0 SKIP=0, `host-tests PASS 898.2s`) |

## 6. Original worktree invariance

After all work, the original worktree was re-read and compared with the
preservation record. Its HEAD, index blobs, worktree SHA-256 values and
porcelain status are unchanged: HEAD still `3d4e9f9`, branch still `experiment/hmx-acc-residual-feedback`, `git status --porcelain=v2` byte-identical for every changed path, all eight worktree SHA-256 values identical, and all eight index blob ids identical (checked after the integration commit and the push). The integration worktree is a
separate directory; no patch was applied to the original tree, no `reset`,
`clean`, `stash`, `pull` or checkout was run there.

## 7. Residual issues

- The legacy worktree remains 3+ commits behind and still holds its changes in
  the index and working tree. Whoever owns it must decide whether to keep,
  rebase or discard them; this audit deliberately did not.
- The same path now exists in both places with different content. If the legacy
  index is committed as-is it will overwrite the integrated test. The legacy
  version has been executed and its useful parts are now committed, so the
  recommended action is to discard **that one path** from the legacy index
  after review — but that decision is the owner's, and nothing was done here.
- C2, D8, `B3_R2_SYNC_UNPROVEN`, Mode 46 and every device question remain
  exactly as before. This audit adds CPU verification only.
