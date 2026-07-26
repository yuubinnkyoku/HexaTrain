# QNN HTP fixed-state Runtime/context order study

This study separates three diagnostic graph variants into independent Runtime/context instances. Each fresh process chooses one three-slot plan, and every slot executes the same fixed language-model state for 100 repetitions. The measured environment was QAIRT 2.48.40.260702151143, QNN API 2.37, HTP V81, Android 16 with 2026-05-01 security patch, and SELinux Enforcing.

The model is FP32 with shape B1 × T8 × V32 × D16. Fixed canonical hashes were one-hot `d85d7d14ab07879ab62b29dc0be5eef0c51d29db0a1e6050b8d7ccb080bd00f1`, target `f1c1a960169be212ee9f4b5856b9add5b9f2dd5ff68b77ce962292d8e1c724cb`, and current parameters `5674c9ecf8bcb785a4db27a73afb11e33aa22c23670301fbe7865692fa83b93b`.

The public result contains 60 completed processes: 45 discovery processes in five balanced waves, each covering all six permutations plus the three homogeneous controls, and 15 confirmation processes covering the homogeneous controls. The three variants are full, stop-after-dinput, and stop-after-dembedding. Audit-only and smoke reports are deliberately outside the export allowlist.

The completed aggregate has the observational classification `FRESH_RUNTIME_CONTEXT_INSTANCE_ASSOCIATED_EXECUTION_VARIABILITY`. It contains 21 varying slots of 180 and 12 slots with nonfinite APP_READ elements. Every recorded QNN backend/device/context/graph create, graph finalize, and graph execute result was zero; the aggregate has 18,000 execute attempts and 18,000 successes. APP_READ poison residuals, caller-owned APP_WRITE integrity failures, activity failures, and focus failures were zero.

The homogeneous confirmation processes varied in 3/15 cases (exact 95% Clopper-Pearson interval 4.33–48.09%); all homogeneous processes varied in 8/30 (12.28–45.89%). Discovery any-process variation was 15/45 (20.00–48.95%). Across all processes, variation was 18/60 (18.85–43.21%), while nonfinite output occurred in 10/60 processes (8.29–28.52%).

The homogeneous process-discordance counts were 2/10 for full, 3/10 for stop-after-dinput, and 3/10 for stop-after-dembedding. Homogeneous-position varying-slot counts were 2/30, 4/30, and 3/30. Confirmation homogeneous patterns were 000=12, 010=2, and 001=1; combined homogeneous patterns were 000=22, 100=2, 010=3, 001=2, and 011=1. These are measured counts. The inference is narrower: the data do not support a graph-variant-specific explanation, a required fixed position, or a required preceding graph.

Discovery waves explicitly crossed variant generation order. Each plan below had five discovery processes; the table reports varying processes, not varying slots.

| Variant generation order | Varying processes / 5 |
| --- | ---: |
| FULL → DINPUT → DEMBEDDING | 3/5 |
| FULL → DEMBEDDING → DINPUT | 2/5 |
| DINPUT → FULL → DEMBEDDING | 0/5 |
| DINPUT → DEMBEDDING → FULL | 1/5 |
| DEMBEDDING → FULL → DINPUT | 3/5 |
| DEMBEDDING → DINPUT → FULL | 1/5 |
| FULL → FULL → FULL | 1/5 |
| DINPUT → DINPUT → DINPUT | 2/5 |
| DEMBEDDING → DEMBEDDING → DEMBEDDING | 2/5 |

The graph structures were fixed by variant: full had 123 source nodes, 153 source tensors, 15 inputs, and 31 audited outputs; stop-after-dinput had 98, 153, 14, and 18; stop-after-dembedding had 99, 153, 14, and 19. The full, dinput, and dembedding terminal boundaries were `lm_output_projection_next`, `lm_dinput`, and `lm_dembedding`, respectively.

For every one of the 21 varying slots, the first external audit tensor was `gradient_gamma1`; DINPUT also varied in all 21. Its candidate node is `UNMAPPED`: the audit sequence supplies a tie-breaking observation order, not node-level localization. Runtime and context/backend/device creation happen together in each fresh slot, so this study does not mutually separate those creation scopes.

The process-level exact Clopper-Pearson 95% interval is generated from the completed 60-process aggregate and recorded in the public summary. The experimental source milestone is `aa94e5a`.

## Cause classification and application audit

The observational classification is
`FRESH_RUNTIME_CONTEXT_INSTANCE_ASSOCIATED_EXECUTION_VARIABILITY`. Graph
variant is not a necessary condition: every homogeneous control reproduced a
discordant process in the independent confirmation cohort. A preceding graph
is also not necessary because variation occurred at position 1. No fixed
Runtime ordinal is sufficient because variation occurred at every position
and the homogeneous patterns include `100`, `010`, `001`, and `011`.

This classification does not assign an internal Qualcomm cause. Every slot
creates the Runtime, backend/device, context, and finalized graph together.
The homogeneous controls hold graph variant and fixed state constant while
that jointly-created instance bundle changes. They show discordance across
such instances, but do not identify which construction or execution scope
accounts for it.

All caller-owned APP_WRITE tensors retained their pre-execute raw hashes, and
the FULL-only learning-rate APP_WRITE buffer retained its exact bytes.
Successful source tensor creation, source node addition, actual QNN binding
counts, and all APP_READ output audits matched the declared variant structure.
Poison residual and unmodified APP_READ element counts were zero. No concrete
application initialization, binding, alias, or ownership error was found.
Nonfinite values were recorded as failures and were neither replaced nor
clamped.

## Regression

- Host reference tests and JVM unit tests passed.
- QAIRT-enabled `assembleDebug` and `assembleDebugAndroidTest` passed.
- The APK/QAIRT hash audit passed for build
  `2.48.40.260702151143`.
- Nineteen headless correctness suites passed. Activity and focus takeover
  counters were zero for every suite and every allow-listed study process.
- Headless app-private Skel corruption recovery reported `replaced` with the
  expected hash, and the next headless probe reported `reused`.
- The public exporter passed its strict 60-process/180-slot validation,
  PowerShell parse check, denylist negative test, invalid-output-path negative
  test, and exact-output-file-set check.

See [the aggregate data and export methodology](results/qnn-htp-graph-order-2026-07/README.md). In `process-results.csv`, `report_id` is an experiment identifier, not an operating-system process identifier. The public data contains only hashes and aggregate numeric audits; it excludes raw tensors, executable artifacts, endpoint information, local paths, operating-system process identifiers, timestamps, and raw diagnostics.
