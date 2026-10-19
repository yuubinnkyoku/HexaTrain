# G1 Backward Repair — Native Completion & Safety Monitor Diagnosis

## A. Starting evidence

Status: **IN_PROGRESS**. Native dValue single and dual are independently verified as full one-process pilots. Formal cohort, performance and remaining device regressions are not yet certified. Training remains **NO-GO**.

The starting commit is `c6c30c18713084e002ce5485b8f28d67d26faad2`, on the recovered research checkout, with a clean tree and no upstream. Work continues on `investigation/g1-backward-repair-native-completion`. The common Git directory and ancestry of `e05f3539e2e0c0e461157a3abf1d9833d74b392b`, `be4271efd1b8fb71f0c66f3ae7c35f04571aa78e`, and `49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2` were checked. No other clone was explored.

| Frozen artifact | SHA256 |
| --- | --- |
| Repair DSP | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` |
| Repair Host | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` |
| ARM probe | `23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05` |
| Existing Double QK DSP | `962a98c6f30e4e8f610abba22f69cd2f08fb216d5ed111586b45d7c73f8991e7` |
| Existing Forward Host | `e2df381e1682ab12e68784ee22ddc468fe553d5cf12c778a167c75990a83ea0a` |
| Double QK fixture | `f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2` |
| Frozen gradients | `c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b` |

All five package manifests, their source identities, binaries and fixtures passed the starting audit. Arithmetic source and DSP/Host/ARM compiled bytes are unchanged. The compiled source is `bb6aaceb42de505b29689a362ce8293687e7ab89`; the final bundle retains its original compilation manifest. Toolchain remains QAIRT `2.48.40.260702151143`, QNN API `2.37.0`, V81, Hexagon SDK `6.6.0.0` / Tools `19.0.07`, NDK `26.2.11394342`, UCRT64 g++ `15.2.0`.

Three previous complete pilots were rehashed and independently reassessed. Each has 2,200 Core + 256 Holdout pairs per observer, four observers, 9,824 pairs, zero arithmetic/observer word mismatches and exit0. They remain pilots; none is counted as a formal process.

| Previous role | Core maxAbs | Holdout maxAbs | Numerical digest |
| --- | ---: | ---: | --- |
| Standalone Softmax | 6.335392901668513e-5 | 2.2067718461471486e-6 | `fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267` |
| Standalone dValue | 1.68030018699028e-5 | 2.7630283376822717e-6 | `64248fcdfaf2534c37f40bc961987d0a627f6d4c457b3ba7cc553de2ac316e41` |
| Native Softmax | 1.0078856647055545e-4 | 4.047485298031006e-6 | `edcaa214b0833b00c77d8c410d9e32894eef7e6aaecbce0cfb9307933bfaa16d` |

## B. Top-resumed monitor diagnosis

The historical response has 153,912 characters, a valid ActivityManager activities header, and zero `topResumedActivity` / `mResumedActivity` fields. Its UTF-8 SHA is `fd55716fed2331971be59e377e92f586c096f033c2f3145ca1fb96aead9c98dd`. It is classified **VALID_RESPONSE_NO_FIELD**. The old ADB wrapper accepted the command, but the trace did not separately retain its numeric exit. Neither a lifecycle transition nor a parser spelling bug is established by this response. Missing information remains unknown and unsafe.

Plan `host_tests/fp32_custom/backward_native_monitor_plan.json` was committed before read-only diagnosis. Limits: three rounds, at most 35 ADB commands, 240 seconds total, 15 seconds per command. Actual: 25 commands / 7.391 seconds, no DSP arithmetic and no device mutation. Stable identities grouped aliases of the original physical device. Android API is 36; the exact build fingerprint, boot identity/uptime, app identities and raw responses remain private.

| Source | Observations | Interpretation |
| --- | --- | --- |
| Existing `dumpsys activity activities` | 3/3 known, same top-resumed; Awake | Existing authoritative control was usable at these times |
| Full `activity` service dump | 3/3 agreed with control | Sequential agreement; not simultaneous or independent equivalence proof |
| Listed `activity_task` service | 3/3 exit0, stdout/stderr empty | EMPTY_RESPONSE; cannot authorize arithmetic |

Command names came from actual service inventory and ActivityManager help. No focus-only substitute, retry-until-known loop or alternative monitor was adopted. The trace runner is unchanged (`ddaec3470adea03c287ef3138f64a0d04ac9a0ca408d096c24303385d0c9f811`). Its two-second top-resumed monitoring, strict per-process preflight, critical safety checks and owned-PID cancellation remain intact.

The Host parser distinguishes command failure, empty response, valid response without field, field present but unparseable, unknown, verified and unexpected Activity. Fixtures reject focus-only, Awake-only, truncated and multiple-Activity responses. The actual historical missing-field response is also independently rejected. Activity-switch and vendor dump internals were not experimentally established; no equivalent alternative monitor is certified.

| Hypothesis | Evidence / remaining uncertainty |
| --- | --- |
| Actual lifecycle change | Unresolved; no lifecycle transition was directly captured |
| Dump field omission | Observed; reason unresolved |
| Fixed-build output condition | Unresolved; current successful samples do not explain the historical absence |
| ActivityManager / ActivityTaskManager representation | Actual response difference established; `activity_task` empty is unusable |
| Monitor / ADB load effect | Unresolved; no causal load experiment |
| Command failure / truncation | Historical dump was nonempty and structurally valid; no unique command-failure attribution |

The first new Forward preflight stopped at **DEVICE_NOT_AWAKE_OR_UNKNOWN**, before any DSP execute. After explicit manual wake by the user, a new unused report folder passed fresh preflight. The earlier stop was retained, not overwritten or promoted.

## C. 6031 watch

Historical API6031 / DMA6033 / exit139 and all partial evidence remain intact. The prior same-boot default MatMul control completed 100 executes but retained its eight expected accuracy failures. The previous full custom pilots and new Forward/Native dValue processes provide additional non-reproduction evidence. State remains **UNRESOLVED / DORMANT / WATCH**; the root cause is not established. No reboot, automatic wake, force-stop or repeated arithmetic retry was used.

## D. Native dValue single

The new process uses the actual Forward-produced per-head P with frozen dContext, in the same graph. Inputs do not include Host P. NoTap uses `QNN_TENSOR_TYPE_NATIVE`; Tap exposes the produced P as APP_READ for diagnostics, without Host reinjection. The two diagnostic conditions must not be confused.

Native dValue completed all four observers: Core2,200 + Holdout256 per observer, 9,824 pairs, four graphs, 4,912 API-successful executes, exit0, finite / complete writes / canary / input immutability / teardown PASS. Independent FP32 arithmetic and observer final words both have zero mismatches.

| Metric | Actual | Gate |
| --- | ---: | ---: |
| Core / total dValue maxAbs | 1.3850631340517339e-5 | 0.01 |
| Holdout dValue maxAbs | 2.827814691741537e-6 | 0.01 |
| Consumer-local maxAbs | 1.637746467508805e-5 | Reported separately |
| P-induced actual maxAbs | 1.2432777690918329e-5 | Reported separately |
| P-induced conservative bound | 2.7855243723208197e-5 | 0.001 |
| P maxAbs | 4.236482907815642e-7 | 1e-6 |
| P row-sum error | 2.682209014892578e-7 | 2e-6 |

Digest: `e1238888e75695d08cbd5d267851c49025a3b3578fb394dd59b58bf15ed5ee1a`.

At the worst total coordinate (Observed+Tap, Forward18, Gradient9, Head1, output345, original source reference), actual DSP output is 45.94231414794922 versus original double45.94232799858056. Signed total `-1.3850631340517339e-5` = consumer-local `-1.2952018678902277e-5` + P-induced `-8.986126616150614e-7`; residual0. Separate stage maxima occur at different coordinates and are not summed.

Fresh Forward control independently passed all480 P outputs and old joint contracts: dScores perturbation bound `0.0002459099714532658`, dValue bound `2.7855243723208197e-5`, observer differences0. Raw P records were losslessly mapped to the unchanged Double QK assessor. Host-conditioned gradient estimates in this control are not DSP Backward measurements.

## E. Native dual fan-out

Dual independently completed **19,648 consumer pairs**, four observers, four graphs / 4,912 executes, exit0 and all runtime invariants. Softmax total maxAbs is `0.00010078856647055545` (Holdout `4.047485298031006e-6`); dValue total maxAbs is `1.3850631340517339e-5` (Holdout `2.827814691741537e-6`). Both old gates pass, with zero arithmetic / observer word mismatches. The P / perturbation metrics match the single-consumer controls. Dual digest is `58eed431aaccb1db918355314cbd0b0dbd5fd83955d41736a9d16ec0ace1eda0`.

Dual execution used measured permits from all four complete single/standalone roles plus the fresh Forward control. The original frozen runner's permit location is bridged by an exclusively created new singles permit; no prior permit or private evidence is overwritten. Permit routing only adds private report-root selection, with path containment and role/duplicate validation. Gates and arithmetic are unchanged.

Host construction tests verify 24 graphs, one upstream producer per-head P, both consumers using the same P ID, separate heads, no native Host input P, and NATIVE P in NoTap. Combined with actual DSP outputs, complete writes, repeated P identity and received-input words from both consumers, the full dual pilot verifies the tested lifetime / immutability contract. This does not expose invisible internal storage in NoReceipt/NoTap.

## F. Observer-free evidence

Native dValue NoReceipt/NoTap itself passed total accuracy and final output identity. Invisible P is not claimed as directly observed: its stored P words are the Observed+Tap anchor. Final output accuracy is checked against original ideal P independently. Observed receipts verify actual received P / gradient / output words; NoTap ownership comes from the unchanged graph construction.

Dual NoReceipt/NoTap itself passes both actual total-gradient gates, and final output words match the Observed+Tap anchor across all four conditions. **OBSERVER_FREE_ACCURACY_VERIFIED** and **OBSERVER_FINAL_WORD_IDENTITY_VERIFIED** are separate successful pilot checks. No claim about the cause of the historical 35/40-node Tap incident is made.

## G. Formal cohort

**NOT_EXECUTED**. The original plan requires five roles per OS process (`standalone_softmax`, `standalone_dvalue`, `native_softmax`, `native_dvalue`, `dual`), three independent OS processes, unchanged binary/fixtures, full coverage and matching per-role numerical digests. Pilots and incomplete attempts are excluded. No role or threshold is changed.

## H. Performance

**NOT_EXECUTED**. Formal numerical certification precedes performance. No previous Mixed QKV / Double QK / Forward timing is relabeled as dual Backward cost. The common local LN backward control must remain identical across Forward/single/dual comparisons. No layer/microbatch wall extrapolation is permitted.

## I. Fresh regression

| Check | Current status |
| --- | --- |
| Fast | PASS; final source gate still required at closeout |
| Host profile | PASS, 373.0s |
| Qnn profile | PASS, 530.7s; no install |
| Frozen custom Host arithmetic | PASS, 29,472 calls |
| Frozen graph / ABI preparation | PASS, 24 graphs / six rejection cases |
| Saved-P original / independent Host references | PASS |
| Fresh Forward P joint contract | PASS |
| Native dValue | Full one-process pilot PASS |
| Dual | Full one-process pilot PASS |
| Formal cohort | NOT_EXECUTED |
| Remaining old device variants / smoke / Muon W8 | NOT_EXECUTED |
| Project Formal profile | NOT_EXECUTED |

Old negative controls remain unchanged: C0 nine FAIL, cumulative P36 FAIL, Scale1 FAIL, Mask176 required predicate FAIL, Softmax224 local FAIL, standard Backward Softmax core0.3044056503066239 FAIL, standard dValue core0.035564650305797585 FAIL. These historical results are not described as this mission's fresh regressions. The prior 35/40-node Tap incident remains REPRODUCED / CAUSE UNRESOLVED.

## J. Architecture and protected state

No Production operator, Muon ABI, optimizer, LR, checkpoint codec, threshold, fixture, gradient scaling or training update is changed. The starting five protected artifact identities and all nonworking refs passed audit. Checkpoint material is hash-checked without decode/write. Prior private evidence is retained byte-for-byte. Normal battery/thermal/power observations remain monitoring-only; critical device protection and Awake/top-resumed remain hard gates.

Even a complete Saved-P closure would still leave actual PV/Gate/Output generation of dP/dContext, dQ/dK, Head Scatter, projection gradients and LN1 dY aggregation uncertified. Full Attention Backward, full Attention, full1-layer, full19-layer and TRAINING_READY are not granted. **40k→60k / 200k remains NO-GO**.

Private evidence is under `build/reports/g1-backward-repair-native-completion/`; raw tensor/activity/endpoint/app/device data stays private. The public projection rejects missing exit/coverage/gradient, wrong binary, observer downgrade, false native/formal claims, private fields and unmeasured performance. Push/PR/merge/history rewriting are prohibited.
