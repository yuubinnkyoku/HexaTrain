# G1 Saved-P Provenance Recovery

Status: **Case 1 — full research chain recovered** (`RESEARCH_HISTORY_VERIFIED`,
`RESEARCH_SOURCE_RECOVERED`, `RESEARCH_EVIDENCE_RECOVERED`, `RESEARCH_FIXTURES_RECOVERED`,
`HOST_AUDIT_RECONCILED`, `HOST_BACKWARD_REFERENCE_SELFTEST_PASS`).

No device execution, no new DSP PASS, no fixture regeneration, no math/threshold/SDK change.
Protected assets untouched. **40k→60k / 200k training remains NO-GO.**
Absolute local paths are withheld from this committed report per repository policy and are
recorded in private evidence `build/reports/g1-saved-p-provenance-recovery/`.

## Report A — Environment map

Two separate clones of the same origin exist; the mismatch was a clone/workspace mismatch,
not lost history:

| | Session workspace ("audit clone") | Mission-expected workspace ("research clone") |
|---|---|---|
| Role | Host-audit session `e075d8c` | Forward-P certified research |
| HEAD | `e075d8c8df1bc09b3832a9225dae9e3df5bef964` | `e05f3539e2e0c0e461157a3abf1d9833d74b392b` |
| Branch | `investigation/g1-native-saved-p-backward-consumers` | `investigation/g1-s2-double-qk-native-p-closure` |
| Tree | clean before and after | clean before and after |
| Research-chain objects | absent (before this task) | present, complete |

- The agent environment (Windows / pwsh 7 / git 2.55.0.windows.5, single sandbox) operated in
  the audit clone because the session workspace configuration points there. The expected
  workspace exists and is a distinct `.git` object store.
- A third clone (`experiment/g1-expanded-data-100k` at `712a349`, clean) exists and was only
  identified, not used.
- **Verified divergence point:** both clones share `024aaef` (2026-10-10 long-linear preflight).
  The G1 S2 research continued exclusively in the research clone:
  `024aaef → … → 984bea4 (S2) → 32b0e27…934430e → b2ce127 (Mixed, pushed 2026-10-18 15:08)
  → 7f7fe9b…099baa5 → e05f353 (Double QK→P, unpushed, 17:57)`.
  The audit clone instead recreated `investigation/ordered-fp32-hvx-ffn2` from
  `experiment/g1-long-linear-200k` at `024aaef` (reflog 2026-10-16 02:54) and had no fetch
  containing the pushed `b2ce127` before this task. `e05f353` is unpushed by design, so it
  could never arrive in the audit clone by fetch.
- Additive actions by this task (disclosed, recorded in private evidence): one
  remote-tracking ref fetch in the audit clone; one isolated worktree + branch
  `investigation/g1-saved-p-provenance-reconciliation` (base `e05f353`) in the research clone.
  Both original worktrees remain clean at their original HEADs.
- Toolchain: session PATH lacked the MSYS2 ucrt64 bin directory; the compiler itself exists
  (`g++ (Rev8, MSYS2) 15.2.0`, sha256 recorded in private evidence). Root cause of the
  apparent "missing g++": child process `cc1plus.exe` failed with `0xC0000135`
  (DLL not found — ucrt64 bin absent from the child DLL search path). With a session-local
  PATH prefix the toolchain builds normally. LLVM `clang++ 22.1.8` reaches the linker but
  lacks MSVC CRT libraries, so it was not used.
- Fast 30 s stop was the agent harness caller timeout (30000 ms), not a profile assertion.
  Prior research record: Fast PASS in 54.7 s; this task in the fresh reconciliation worktree:
  `verify-local-fast PASS 95.8 s` (PASS=10 FAIL=0 SKIP=21).

## Report B — Commit search

| Commit | Meaning | Audit clone | Research clone | origin |
|---|---|---|---|---|
| `984bea42006c043111b3d5dec353a73b426b4192` | S2 LN1 DSP | missing | **found** (local branch `investigation/g1-c7-selective-ln1-dsp-feasibility`) | no remote head |
| `b2ce1272a9ab5575b06f5a422c7042063306560d` | Mixed QKV (pushed) | missing → **found** after read-only fetch (remote-tracking) | **found** (local branch tracks origin) | **found** (remote head) |
| `e05f3539e2e0c0e461157a3abf1d9833d74b392b` | Double QKᵀ→P closure (unpushed) | missing | **found** (branch HEAD) | missing (expected) |
| `024aaef44784355e8e278ef642e2752ae79846bb` | audit base | **found** (base of `e075d8c`) | **found** (ancestor of `e05f353`) | below `origin/experiment/g1-expanded-data-100k` |
| `e075d8c8df1bc09b3832a9225dae9e3df5bef964` | host audit | **found** (HEAD; 4 blobs verified) | absent (different clone, expected) | not pushed |

Ancestry (research clone, `merge-base --is-ancestor`, exit 0 for each):
`984bea4 → b2ce127`, `b2ce127 → e05f353`, `024aaef → e05f353`.
Research-clone reflog records the full 2026-10-18 sequence
(`b2ce127` checkout 16:44 → plan 16:45 → native-P implementation 16:56 → build fix 16:59 →
evidence 17:04/17:10 → `e05f353` 17:57). The mission "missing" states were properties of
the audit clone's object database only.

## Report C — Recovered source (arithmetic-level verification, at `e05f353`)

Verified by reading actual source, not file names:

- **S2 LN1**: `host_tests/fp32_custom/selective_ln1_arithmetic.h` (FP32-anchored reduction /
  double normalization / saved xhat+inverse) emitted as node `S2` / `S2NoReceipt`
  (`HexaSelectiveLn1` package) in the native diagnostic graph.
- **Mixed QKV**: `mixed_qkv_arithmetic.h` (Q/K: FP32→double, q=0..63 serial products,
  one final FP32 store; V: unchanged strict-IEEE HVX `hexatrain_ordered_hvx_qkv_nn64`) node
  `MixedPrecisionQKV` / `MixedPrecisionQKVNoReceipt` (`HexaMixedQkv`); lossless head split
  via node `QKVHeads` (`HexaFp32`, six head tensors).
- **Double QKᵀ**: `double_qk_arithmetic.h` (NT32 internal double, FP32 raw outputs,
  transpose-correct) node `ForwardRawQkDouble` / `ForwardRawQkDoubleNoReceipt`
  (`HexaDoubleQk`), rank-4 `[1,1,32,32]` unit leading axes.
- **Custom Scale / Mask / Softmax** (stage 2 of the same graph): `ForwardScoreScale`
  with `1.f/std::sqrt(32.f)` (documented bits `0x3e3504f3`), `ForwardMaskAdd` with causal
  `+0` / `-10000`, `ForwardSoftmax` (`HexaFp32` custom op) producing per-head `p0/p1` —
  the "Custom Softmax" of the mission chain.
- **Native S2→Mixed→Heads→DoubleQK→Scale→Mask→Softmax→P**: single graph in
  `double_qk_graph.h`; intermediates are `NATIVE` when `tap=false` and `APP_READ` only at
  declared tap boundaries; `observed=false` switches every producer to its `NoReceipt`
  variant → the four observer conditions (Observed/NoReceipt × Tap/NoTap) are structural.
- **Saved-P ownership for the two backward consumers**: `saved_p_native_ownership_test.cpp`
  (added by `e05f353`) constructs `ForwardSoftmax → savedP(NATIVE)`,
  `SoftmaxBackward(dp, p) → ds` and per-head `dvalue = MatMul(p, dc, transpose0=true)`,
  asserts identical P tensor IDs across producer + both consumers, P absent from graph
  inputs/outputs (no Host reinjection), and zero receipt tensors. Its own terminal line is
  `saved_p_ownership_host=PASS … DSP_BACKWARD_NOT_EXECUTED=1 device_calls=0` — construction
  audited, DSP execution explicitly not claimed.
- Production backward chain (generalized graph) matches the host audit: product →
  ReduceSum(last, keep_dims) → centered → `dScores` (= dMaskedLogits, scale applied by a
  separate node), dValue `MatMul` with `TRANSPOSE_IN0` only.

## Report D — Artifacts / evidence

- **Frozen fixtures recovered and hash-verified (direct SHA256 on recovered files):**

  | Artifact | SHA256 | Match |
  |---|---|---|
  | Double QK fixture (`double-inputs.bin`) | `f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2` | yes |
  | Mixed fixture (`mixed-inputs.bin`) | `8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb` | yes |
  | S2 fixture (`selective-ln1-inputs.bin`) | `492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02` | yes |
  | Original one-layer inputs | `388f4d8b30b86997d2f03d732e5a3210f7e2b3015af883579e3f0659df9d9f4f` | yes |
- **Frozen plan**: `double_qk_native_p_plan.json` records `forward_cases=44`,
  `holdout_cases=16`, `mixed_arithmetic_controls=8`, core coverage
  `44 forward × 2 heads × 25 gradients`, joint gates (p ≤1e-6, row ≤2e-6,
  dScores ≤2.5e-4, dValue ≤1e-3) unchanged from the mission table.
- **Binary identities** (from `final-closeout-private.json`): `libHexaDoubleQkDsp.so`
  `962a98c6…`, `libHexaDoubleQkHost.so` `e2df381e…`, `double_qk_probe` `911d66e8…`,
  `device_tested_source` `099baa5…`, `build_source` `d3298ec…` — all match the mission.
- **Numerical digests** present in private records (`cohort-aggregate-private.json`,
  `final-1-aggregate-private.json`, `final-closeout-private.json`):
  local `d8e507289b602c7cec0a5af898e23b0f5b98b3cbeb74d4bd97b07994dbf8edee`,
  P-assessment `456ad468cb9cb68303261535420999408d2f1fe60443a0e78886473c19c432ae`
  (identical strings recur across aggregate sections → cross-process digest consistency).
- **Certified metrics** present in the committed public summary of `e05f353`:
  P maxAbs `4.23648290781564e-7`, row-sum `2.27242708206177e-7`,
  P-induced dScores `2.45909971453266e-4`, P-induced dValue `2.78552437232082e-5`.
- **Private evidence root intact**: `build/reports/g1-s2-double-qk-native-p-closure/`
  with `final-1/2/3` device consoles, pilot records, cohort aggregate, regression consoles,
  headless results, Fast/Host/Qnn profile outputs, and the protected closeout
  (`all_preserved=true`, `prior_refs=46`, `push=false`, `training=NO-GO`,
  `new_P_DSP_backward=NOT_EXECUTED`). 40k checkpoint digest `89f197d3…` recorded in
  `initial-private.json`. **Missing required evidence: none.**
- Scope note: report / source / fixture / binary / historical-numerical-evidence were
  verified as **separate** states; historical DSP certification was recovered as records —
  not re-executed, and not attributed to any new binary.

## Report E — Host audit reconciliation (`e075d8c`)

- **Preservation**: `e075d8c` remains the audit-clone HEAD on
  `investigation/g1-native-saved-p-backward-consumers`, tree clean, blobs verified
  (`f385b0ea…` memo, `f0996b4b…` plan, `a3c31014…` reference lib, `ffe88641…` reference
  test). No reset/rebase/amend was performed.
- **Selective port** (blob-identical `git hash-object` check):
  `saved_p_backward_reference_lib.h` and `saved_p_backward_reference_test.cpp` were copied
  to the reconciliation worktree with exact blob identity.
  `saved_p_backward_native_plan.json` was ported and **deliberately revised**: `base_head`
  → `e05f353…`, `branch` → `investigation/g1-saved-p-provenance-reconciliation`,
  `fixtures.provenance` and `provenance_gaps` rewritten from "missing/UNRESOLVED" to the
  recovered facts. **Gates, math, observer list, critical-case list are byte-for-byte the
  certified values** (no threshold math or fixture change).
- **Not ported**: `docs/g1-native-saved-p-backward-consumers.md` — its provenance-gap
  statements described the audit-clone state only and are superseded by this report; it
  stays preserved inside `e075d8c`.
- **Compatibility review vs `e05f353`**: audit claims confirmed against the research
  source — per-head softmax P producer, `dValue` `TRANSPOSE_IN0`, softmax chain order
  (product → ReduceSum last keep_dims → centered → dScores), `dScores == dMaskedLogits`
  with separate scale stage, diagnostic `APP_READ` vs production `NATIVE` on the same tensor
  ID, no-Host-reinjection rule. The memo's "S2/Mixed/DoubleQK/Custom-Softmax absent" note is
  environment-specific (audit clone) and false for the research clone, as documented above.
- **Compile & self-test**: `BUILD_EXIT=0`, `RUN_EXIT=0` with `g++ 15.2.0` (ucrt64 PATH
  fix); results `softmax_fp32_vs_double=4.23e-09`, `dvalue_fp32_vs_double=5.14e-08`,
  `softmax_rowsum=1.23e-08`, terminal `saved_p_backward_reference_selftest=PASS`.
  Classification: **`HOST_BACKWARD_REFERENCE_SELFTEST_PASS`** — explicitly **not**
  `DSP_SAVED_P_BACKWARD_VERIFIED` (device_executed=false).
- **Exporter**: new allow-list exporter `scripts/export_public_provenance_recovery.py`
  with `--self-test` proving 6 rejection paths (extra key, missing key, misclassification,
  absolute path, device-execution claim, out-of-bound metric) plus digest stability and
  output identity: `provenance_recovery_exporter_selftest=PASS rejections=6 device_calls=0`.
  Public output `docs/results/g1-saved-p-provenance-recovery-summary.json` generated only
  through this exporter (`provenance_recovery_public_export=PASS`).
- **Remaining numerical gaps**: no DSP execution of the Saved-P consumers in this task
  (standalone Softmax Backward / dValue, single-consumer native, dual fan-out, observer
  cohort, Cartesian/holdout device coverage) — all belong to the next research.
- **Fast**: `verify-local-fast PASS 95.8 s` in the reconciliation worktree (with ucrt64 on
  PATH); Fast/Host/Qnn of the audit-clone commit remain documented there.

## Report F — Recovery action

- **Recovered**: full research history `024aaef → 984bea4 → b2ce127 → 099baa5 → e05f353`,
  Forward-P source (S2/Mixed/HVX V/Double QK/Custom Scale-Mask-Softmax/Native P),
  frozen fixtures (hash-verified), binary/hash identities, numerical digests,
  3-process private records, protected closeout — all in the research clone.
- **Missing**: nothing required; `e075d8c` exists only in the audit clone (expected), and
  its portable content was reconciled above.
- **Safe base for the next Saved-P Backward research**: research clone, branch from
  `e05f3539e2e0c0e461157a3abf1d9833d74b392b` (continue
  `investigation/g1-s2-double-qk-native-p-closure`, or branch
  `investigation/g1-native-saved-p-backward-consumers` there on top of `e05f353`).
  The isolated worktree `investigation/g1-saved-p-provenance-reconciliation` (this report,
  public summary, exporter, ported host reference) is the reconciliation branch.
- **Reproduction commands** (research clone; PATH prefix for the compiler):
  1. host self-test: `g++ -std=gnu++17 -O2 -Wall -Wextra host_tests/fp32_custom/saved_p_backward_reference_test.cpp -o <out>` then run (expects
     `saved_p_backward_reference_selftest=PASS`).
  2. exporter: `python scripts/export_public_provenance_recovery.py --self-test`;
     `python scripts/export_public_provenance_recovery.py --evidence <private.json> --output docs/results/g1-saved-p-provenance-recovery-summary.json`.
  3. gate: `.\scripts\verify.ps1 -Profile Fast` (observed PASS, ~55–96 s).
- **No destructive operations** were performed: no reset / rebase / amend / gc / prune /
  reflog expiry / worktree prune / push / PR / merge; checkpoint, optimizer, dataset,
  previous branches and private reports untouched; `6031` stays
  `UNRESOLVED / DORMANT / WATCH`, the 35/40-node Tap incident stays
  `REPRODUCED / CAUSE UNRESOLVED`.
- **Blockers before DSP consumer verification resumes**: (1) Tier-2 device preflight on the
  same physical device (identity, awake/top-resumed, no active run, pinned QAIRT 2.48.40.260702 /
  Build ID 2.48.40.260702151143, QNN build + APK audit); (2) choose the continuation branch in
  the research clone; (3) the recovered fixture cohort (44×2×25 + holdout) is ready — no
  fixture regeneration needed. **40k→60k / 200k training remains NO-GO.**


