# G1 Backward Repair — Native Completion & Safety Monitor Diagnosis

## A. Starting evidence

Status: **BLOCKED_FORMAL_COHORT** at the Awake hard gate. Native dValue single and dual are independently verified as full one-process pilots. Formal cohort, performance and remaining device regressions are not certified. Training remains **NO-GO**. Manual wake is pending; no additional device arithmetic is launched automatically.

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

The first new Forward preflight stopped at **DEVICE_NOT_AWAKE_OR_UNKNOWN**, before any DSP execute. After explicit manual wake by the user, a new unused report folder passed fresh preflight. The earlier stop was retained, not overwritten or promoted. Both this stop and the later cohort preflight contain actual `mWakefulness=Dozing`; neither collected activity evidence or started a probe. These new Dozing observations do not establish the cause of the historical Awake / missing-top-resumed incident.

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

**NOT_EXECUTED**. The first cohort preflight stopped at DEVICE_NOT_AWAKE_OR_UNKNOWN before an owned probe process or stdout existed. It contributes zero OS processes and zero arithmetic executes. Its original directory is retained. A separately preregistered unused `cohort-1-awake` folder may occupy the first actual process slot only after manual wake and another fresh preflight; this is not an automatic arithmetic retry.

The original plan requires five roles per OS process (`standalone_softmax`, `standalone_dvalue`, `native_softmax`, `native_dvalue`, `dual`), three independent OS processes, unchanged binary/fixtures, full coverage and matching per-role numerical digests. Pilots and incomplete attempts are excluded. No role or threshold is changed. A supplemental role-contract digest directly includes actual output words, role, numerical ordering and measured contract outcomes; legacy numerical digests are preserved unchanged. Synthetic Host controls confirm that receipt/timing changes are excluded, while role, case, output or contract changes alter this supplemental digest.

## H. Performance

**NOT_EXECUTED**. Formal numerical certification precedes performance. No previous Mixed QKV / Double QK / Forward timing is relabeled as dual Backward cost. The common local LN backward control must remain identical across Forward/single/dual comparisons. No layer/microbatch wall extrapolation is permitted.

## I. Fresh regression

| Check | Current status |
| --- | --- |
| Fast | PASS, closeout94.2s |
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
| Fresh C0 Host negative | Expected nine FAIL reproduced, actual exit4 |
| Project Formal profile | BLOCKED_SCOPE / NOT_EXECUTED |

Old negative controls remain unchanged: C0 nine FAIL, cumulative P36 FAIL, Scale1 FAIL, Mask176 required predicate FAIL, Softmax224 local FAIL, standard Backward Softmax core0.3044056503066239 FAIL, standard dValue core0.035564650305797585 FAIL. These historical results are not described as this mission's fresh regressions. The prior 35/40-node Tap incident remains REPRODUCED / CAUSE UNRESOLVED.

The C0 negative was additionally rerun for this mission using hash-identical copies of the two required frozen LN fixture files in a new private working directory: nine expected failures, exit4, no checkpoint decode or fixture regeneration. The project Formal profile was inspected but not run: its Full implementation regenerates existing default private research reports and includes an unrelated CPU training probe. This mission forbids private evidence overwrite and training updates. No heavy step was skipped and relabeled Formal PASS; the separately preregistered DSP cohort remains a distinct, uncompleted requirement.

## J. Architecture and protected state

No Production operator, Muon ABI, optimizer, LR, checkpoint codec, threshold, fixture, gradient scaling or training update is changed. The five protected artifact identities and all nonworking refs passed starting and progress audits; all333 previous private evidence files remain byte-identical. Checkpoint material is hash-checked without decode/write. Normal battery/thermal/power observations remain monitoring-only; critical device protection and Awake/top-resumed remain hard gates.

Even a complete Saved-P closure would still leave actual PV/Gate/Output generation of dP/dContext, dQ/dK, Head Scatter, projection gradients and LN1 dY aggregation uncertified. Full Attention Backward, full Attention, full1-layer, full19-layer and TRAINING_READY are not granted. **40k→60k / 200k remains NO-GO**.

Private evidence is under `build/reports/g1-backward-repair-native-completion/`; raw tensor/activity/endpoint/app/device data stays private. The public projection rejects missing exit/coverage/gradient, wrong binary, observer downgrade, false native/formal claims, private fields and unmeasured performance. Push/PR/merge/history rewriting are prohibited.

Public aggregate: `docs/results/g1-backward-repair-native-completion-summary.json`. Exporter self-test rejects18 invalid evidence cases. Granted pilot classifications: **TOP_RESUMED_MONITOR_CHARACTERIZED**, **NATIVE_DVALUE_SINGLE_VERIFIED**, **SAVED_P_DUAL_NATIVE_VERIFIED**, **SAVED_P_LIFETIME_VERIFIED**, **SAVED_P_DUAL_OBSERVER_FREE_VERIFIED**. Formal status: **BLOCKED_FORMAL_COHORT**; root-cause / remaining-frontier status: **UNRESOLVED**. No alternative-monitor verification, formal-cohort verification or performance measurement classification is granted.

## Required answers (current evidence)

| # | Answer |
| --- | --- |
| 1 | Yes: recovered research checkout, exact c6c30c18713084e002ce5485b8f28d67d26faad2. |
| 2 | DSP/Host/probe and arithmetic source are unchanged; hashes are in A. |
| 3 | Both previous standalone pilots retained and independently reassessed PASS. |
| 4 | Previous complete Native Softmax retained and independently reassessed PASS. |
| 5 | The actual nonempty ActivityManager response omitted both authoritative resumed fields. |
| 6 | VALID_RESPONSE_NO_FIELD; actual lifecycle change and causal mechanism unresolved. |
| 7 | Full activity dump agreed3/3; listed activity_task returned exit0 with empty response3/3. |
| 8 | Existing control verified at observation times and throughout complete new pilots; no equivalent alternative adopted or certified. |
| 9 | Yes; unknown remains unsafe. Missing information was not inferred from focus or Awake. |
| 10 | No weakening: original trace runner, intervals, preflight and stop semantics unchanged. |
| 11 | No new API6031 in the completed Forward, Native dValue or dual pilots. |
| 12 | Root cause unresolved; UNRESOLVED / DORMANT / WATCH. |
| 13 | Yes: Native dValue full9,824 pairs, four observers, actual exit0. |
| 14 | Native dValue actual total maxAbs1.3850631340517339e-5. |
| 15 | P-induced actual1.2432777690918329e-5, bound2.7855243723208197e-5, consumer-local1.637746467508805e-5; signed coordinate decomposition in D. |
| 16 | All four Native dValue observer conditions PASS. |
| 17 | Yes: dual full19,648 consumer pairs, actual exit0 and complete teardown. |
| 18 | Same per-head P ID, one upstream producer, both consumers; Host construction plus actual received-input and output evidence. |
| 19 | No Host P reinjection. Tap is a diagnostic APP_READ exposure; NoTap uses NATIVE P. |
| 20 | Tested full dual pilot lifetime/immutability verified; no direct observation of invisible NoReceipt/NoTap P storage is claimed. |
| 21 | Dual dMaskedLogits total1.0078856647055545e-4; dValue total1.3850631340517339e-5. |
| 22 | Both pass original gates0.002 /0.01. |
| 23 | Yes, dual NoReceipt/NoTap itself passes both total-gradient accuracy gates. |
| 24 | Zero observer / arithmetic word mismatches; single-vs-dual output words also match for both consumers. |
| 25 | Full Core2,200/Holdout256 per observer and consumer, all four conditions. |
| 26 | Formal five-role cohort not executed yet; first preflight stopped before any probe OS process. |
| 27 | Three-process digest agreement not yet measured. |
| 28 | Yes: pilots and preflight-only stopped attempts are excluded from formal counts. |
| 29 | Fresh actual Forward P joint contract PASS, all480 P outputs. |
| 30 | Fresh HVX FFN2/QKV device regressions not executed yet. |
| 31 | Fresh Device smoke not executed yet. |
| 32 | Fresh Muon W8 114/114 not executed yet. |
| 33 | Backward additional execution cost not measured. |
| 34 | Formal numerical cohort is required before performance; current first-cohort preflight stopped at Awake. |
| 35 | Critical safety hard gates retained; ordinary battery/temperature/power monitored without new waivers. |
| 36 | Protected files/refs PASS; all333 prior private evidence files also rehashed unchanged. No checkpoint decode/write. |
| 37 | Actual PV/Gate/Output dP/dContext, then dQ/dK, Head Scatter, projection gradients and LN1 dY aggregation. |
| 38 | Neither full1-layer nor full19-layer is certified. |
| 39 | 40k→60k /200k training remains NO-GO. |

## 2026-10-20 — Formal certification closeout from dffbc50

### A. Starting state and preserved evidence

Final decision for this continuation: **BLOCKED_FORMAL_COHORT**. This is a safety/preflight block, not an arithmetic failure. The fresh read-only preflight again observed **Dozing**; manual wake is required before a separately authorized continuation and another fresh preflight. No DSP probe process was started. The earlier blocked observations and all pilot evidence above remain historical records.

The designated research checkout started clean at `dffbc50efaec04f59d16bf126f0631d355f99ae7`, on `investigation/g1-backward-repair-native-completion`, with no upstream. Git root/common-directory identity and ancestry of `c6c30c18713084e002ce5485b8f28d67d26faad2`, `49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2`, `be4271efd1b8fb71f0c66f3ae7c35f04571aa78e`, and `e05f3539e2e0c0e461157a3abf1d9833d74b392b` were verified. The other checkout and audit clone were not operated on. Existing branch history is retained; no push, PR, merge or history rewrite is performed.

All five frozen binary identities and both frozen fixture identities in A remain unchanged. All five bundle manifests were checked against current source/binary hashes, retaining frozen compiled source `bb6aaceb42de505b29689a362ce8293687e7ab89` and the fixed toolchain. No DSP/Host/ARM binary was rebuilt. The unchanged trace runner SHA256 is `ddaec3470adea03c287ef3138f64a0d04ac9a0ca408d096c24303385d0c9f811`; its ActivityManager check, two-second interval and stop behavior were not edited.

The existing five full pilots were independently reassessed from raw/compressed evidence, captured completion, original double references and source-ordered FP32 arithmetic. All five remain **PILOT_PASS**; their raw hashes and legacy digests match the cohort-entry ledger. Starting Forward P was independently reassessed PASS for all480 outputs, retaining P maxAbs `4.236482907815642e-7`, row-sum error `2.682209014892578e-7`, dScores bound `0.0002459099714532658`, and dValue bound `2.7855243723208197e-5`. This reassessment consumes historical device outputs; it is not fresh device execution.

### B. Fresh formal preflight

One read-only preflight ran in a new exclusively created private closeout directory. It used pinned QAIRT arguments, the existing approved APK path and the existing research monitoring policy. QAIRT inventory exit3 was advisory for optional samples; Build ID and required core files were present. Host storage, compiler, frozen custom package, protected identities, same physical device and active-run absence passed before the Awake gate.

Actual preflight exit was **2**; reason **DEVICE_NOT_AWAKE_OR_UNKNOWN**, with actual `mWakefulness=Dozing`. The authoritative top-resumed gate was not reached and is not reported PASS. Any foreground detail observed by the active-run check is not a substitute. Critical thermal, remaining competing-probe checks, installed-APK/device-storage checks and the immediate runner recheck were not completed after this early stop; full device readiness is therefore not certified.

Device executes were0 and device operations after stop were0. No trace-runner invocation, probe staging, automatic wake/unlock, foreground takeover, force-stop, installation or state polling followed. The historical153,912-character missing-field response and earlier Dozing folders were retained unchanged. No new missing-field episode was exercised; the observed new safety violation is Dozing.

Reproduction uses the existing `preflight_forward_p_resume.ps1` with explicit pinned QAIRT/Hexagon/NDK roots, Build ID `2.48.40.260702151143`, `-MixedQkvResearchMonitoring`, the approved existing APK and a new unique private `-ReportRoot`. The exact invocation, stdout/stderr, actual preflight exit2, raw power response and hashes are retained privately. Do not run the cohort from this blocked result; manual wake followed by a new complete PASS preflight is required.

### C. Three-process cohort

The unchanged preregistration requires five roles in each of three independent OS processes. The dual permit hash matched its registered entry. The supplemental role-contract specification and legacy digest remain unchanged; six Host checks confirmed output/role/case/contract sensitivity and timing/receipt-metadata exclusion.

Current continuation: **0 started processes, 0 completed processes, 0 graphExecute calls, 0 completed roles**. The unused `cohort-1-awake` slot was not consumed; the new directory contains preflight-only evidence. Actual probe exit and raw probe hash are **null**, and teardown is **NOT_APPLICABLE_NO_PROCESS**. The preflight exit2 is not a probe exit. All historical preflight-only stops and pilots are excluded from formal counts.

| Role | Retained pilot total maxAbs | Formal total maxAbs | Formal coverage / word contracts |
| --- | ---: | --- | --- |
| standalone_softmax | 6.335392901668513e-5 | null | NOT_EXECUTED |
| standalone_dvalue | 1.68030018699028e-5 | null | NOT_EXECUTED |
| native_softmax | 1.0078856647055545e-4 | null | NOT_EXECUTED |
| native_dvalue | 1.3850631340517339e-5 | null | NOT_EXECUTED |
| dual Softmax | 1.0078856647055545e-4 | null | NOT_EXECUTED |
| dual dValue | 1.3850631340517339e-5 | null | NOT_EXECUTED |

All pilots retain Core2,200/Holdout256 per observer, four observers, zero arithmetic/observer word mismatches, 4,912 graphExecute calls per role, actual exit0 and teardown success. None establishes three-process formal agreement. Formal legacy/supplemental digest agreement is **NOT_MEASURED**. No new6031 occurrence was observed because runtime arithmetic was not exercised; **UNRESOLVED / DORMANT / WATCH** remains unchanged.

### D. Native Saved-P scope

The retained dual pilot verifies one P producer per head, both consumers sharing the same per-head P tensor ID, NATIVE P in NoTap, no Host P reinjection, tested lifetime/immutability and both actual total gradients. NoReceipt/NoTap itself passes both gradient gates with final-word observer identity. These are pilot findings, with the invisible-storage observation limits stated in F; no formal certification is added by this continuation.

### E. Performance

**NOT_EXECUTED** because the three-process numerical entry gate is unmet. First execute,31 steady samples, median/p10/p90, execute wall, finalize, arithmetic/receipt windows and measured resource comparisons are unavailable for all Forward/single/dual configurations. They remain **null**, not zero. Existing pilot times are not substituted. The approximately17.8ms historical Forward measurement includes LN local Backward control; it is not a pure Forward cost. No diagnostic-time subtraction or19-layer/microbatch extrapolation is made.

### F. Verification and protected state

Fresh Fast and exporter checks for this documentation/exporter closeout are recorded below. Host/Qnn profile PASS records in I and the original public aggregate remain **historical**. They are not this continuation's fresh PASS. No candidate, graph, ABI, math or device runner changed; the only implementation change adds an allow-listed blocked-closeout projection to the public exporter, preserving its original18 rejection tests and adding14 unsafe/false-closeout rejection tests. This records the new preflight and explicitly separates historical profile results from fresh status; it does not authorize formal PASS.

Fresh device regressions are **NOT_EXECUTED**: S2 LN1, Mixed QKV double Q/K, strict IEEE HVX V, Double QK, Custom Scale/Mask/Softmax, Forward cumulative P, HVX FFN2/QKV, QK/Backward Scale, LN2/Softmax Backward controls, device smoke and Muon W8. Historical W8 114/114 is not reused as fresh evidence. Fresh Host/Qnn profiles are also NOT_EXECUTED after the safety stop. Overall regression closure remains BLOCKED. Project Formal is **BLOCKED_SCOPE / NOT_EXECUTED**, preserving the explicit scope restriction.

Closeout validation: **Fast PASS**, actual exit0,119.9s (outer profile PASS1/FAIL0/SKIP0; its intended inner Fast skips are not full-profile coverage). Original exporter18 reject paths and additional blocked-closeout14 reject paths **PASS**, device calls0. The exporter also refuses actual or partial cohort process evidence in every `cohort-*` directory, including `cohort-1-awake`, when projecting this zero-process blocked state. Public `formal_closeout.fresh_regression` is the freshness authority for this continuation; the older top-level regression entries remain explicitly historical.

The protected40k checkpoint retains SHA256 `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`. All five protected files, all333 ledger-listed earlier private evidence files, all pre-existing evidence files in this mission root and all nonworking refs were checked byte-for-byte. The existing100k baseline,200k UNSTABLE/step41,647 forensic history, dataset/tokenizer/cache/order, production code, Muon and checkpoint codec were not edited or executed. No checkpoint decode/write, training update or final test occurred. New evidence is confined to a new private closeout directory; no old evidence folder was reused.

Old negative controls remain historical and unchanged: C0 Host P9, old cumulative P36, Standard Scale1, Standard Mask176, Standard Softmax224 expected failures, Backward Softmax maxAbs0.30440565 and dValue0.03556465. None was rerun in this continuation. The35/40-node Tap incident stays **REPRODUCED / CAUSE UNRESOLVED**.

### G. Research decision and required answers

No formal Saved-P, lifetime, observer-free, total-gradient, performance or regression-closure certification is granted. Existing pilot classifications are preserved. Final classification is **BLOCKED_FORMAL_COHORT**, with unresolved runtime/frontier questions unchanged. Manual wake is the external condition for continuation; the original monitor and all hard gates still apply.

| # | Answer for this continuation |
| --- | --- |
| 1 | Yes; exact dffbc50efaec04f59d16bf126f0631d355f99ae7 in the designated research clone/branch, initially clean. |
| 2 | Yes; all five specified binaries, frozen source and fixtures are unchanged. |
| 3 | Yes; historical Forward P raw evidence independently reassessed PASS,480 outputs. |
| 4 | No; fresh Dozing. Authoritative top-resumed gate not reached. |
| 5 | Yes; earlier Dozing preflight contributes0 processes. |
| 6 | No; current preflight stopped before a probe process. |
| 7 | No formal role ran; all five retained pilots independently reassessed PASS. |
| 8 | No new coverage; historical pilots retain Core2,200/Holdout256 across four observers. |
| 9 | Formal null; standalone Softmax pilot6.335392901668513e-5. |
| 10 | Formal null; standalone dValue pilot1.68030018699028e-5. |
| 11 | Formal null; Native Softmax pilot1.0078856647055545e-4. |
| 12 | Formal null; Native dValue pilot1.3850631340517339e-5. |
| 13 | Formal null; dual Softmax pilot1.0078856647055545e-4. |
| 14 | Formal null; dual dValue pilot1.3850631340517339e-5. |
| 15 | Pilot arithmetic mismatches0 for every role; formal NOT_EXECUTED. |
| 16 | Pilot observer mismatches0 for every role; formal NOT_EXECUTED. |
| 17 | Retained dual pilot yes; formal NOT_EXECUTED. |
| 18 | Tested pilot lifetime/immutability yes; formal NOT_EXECUTED. |
| 19 | No Host P reinjection in the retained native pilots; no new process. |
| 20 | Retained dual NoReceipt/NoTap passes both total-gradient gates; no new measurement. |
| 21 | No;0/3 independent formal OS processes completed. |
| 22 | No formal exits; probe exit null, teardown not applicable. Preflight actual exit2 is separate. |
| 23 | Not measured; no three-process digest set exists. |
| 24 | Pilot legacy digests matched; supplemental Host sensitivity tests PASS. Formal cross-process comparison unavailable for both. |
| 25 | No new observed6031; runtime not exercised. Existing watch remains unresolved. |
| 26 | Dozing recurred; authoritative top-resumed query not reached, no new missing-field diagnosis. |
| 27 | Yes; all pilots, unsafe/partial/preflight-only attempts remain excluded. |
| 28 | No; formal numerical entry gate blocked. |
| 29 | Forward/single/dual median and p10/p90 all null. |
| 30 | No new timing comparison exists. Future comparisons must match LN control and graph conditions. |
| 31 | Fresh HVX FFN2/QKV NOT_EXECUTED. |
| 32 | Fresh device smoke NOT_EXECUTED. |
| 33 | Fresh W8 114/114 NOT_EXECUTED; historical PASS not promoted. |
| 34 | Yes; historical controls preserved, not rerun here. |
| 35 | Fresh Fast PASS, exit0/119.9s; Host/Qnn NOT_EXECUTED here, historical PASS retained. |
| 36 | Yes; Project Formal BLOCKED_SCOPE / NOT_EXECUTED. |
| 37 | Yes within the recorded hash/ref ledgers; protected5, prior333 and all existing mission evidence unchanged. |
| 38 | BLOCKED_FORMAL_COHORT; pilot verification retained, formal certification withheld. |
| 39 | Actual PV/Gate/Output dP/dContext, then dQ/dK, Head Scatter, projection gradients and LN1 dY aggregation remain uncertified. |
| 40 | Neither Full1-layer nor Full19-layer is certified; Full Attention Backward/Attention and TRAINING_READY also withheld. |
| 41 | 40k→60k and200k training remain NO-GO. |

## 2026-10-20 — User-directed fresh safety recheck after fbf9a78

After the user's instruction to continue, work resumed clean on the same branch at `fbf9a7833cc4627463455ff08f50cae2e170b6f3`. The fixed five bundle source/binary identities, five protected files,333 earlier ledger files, all existing mission evidence (including the preceding closeout) and all nonworking refs were rehashed unchanged before device access. No candidate or orchestration changes were made.

One new read-only preflight used a new exclusive private directory and the original pinned arguments/research monitoring policy. It again stopped with actual exit2, **DEVICE_NOT_AWAKE_OR_UNKNOWN**, and actual `mWakefulness=Dozing`. Same physical device and active-run absence passed. The authoritative Awake/top-resumed entry conditions did not both pass; top-resumed and subsequent device gates were not reached. This does not certify full device readiness or demonstrate a new missing-top-field incident.

No new probe process, staging or DSP arithmetic started; actual probe exit/raw hash are null and teardown is not applicable. Device operations after the stop were0. No automatic wake, unlock, foreground takeover, retry or additional state polling occurred. The unused `cohort-1-awake` slot and all previous stopped folders remain untouched. Formal count stays **0/3**; pilots remain pilots. Performance, fresh device regressions and fresh Host/Qnn profiles remain NOT_EXECUTED, and Project Formal remains BLOCKED_SCOPE / NOT_EXECUTED. No new6031 conclusion is available because runtime arithmetic was not exercised.

The public aggregate's existing **BLOCKED_FORMAL_COHORT** projection is retained unchanged: this recheck adds no new numerical result, process, performance value or certification. The41 answers in the preceding closeout retain their numerical/scope conclusions. Exact invocation, actual exit, raw power response and identity/hash audits for this recheck are preserved only in the new private evidence directory. Manual wake and another new complete PASS preflight are required for continuation. Training remains **NO-GO**.

The documentation-only recheck closeout passed fresh **Fast**, actual exit0/107.0s (outer PASS1/FAIL0/SKIP0). Host/Qnn/device/Formal results were not substituted by this Fast result.

## 2026-10-20 — Awake entry accepted, final top-resumed gate blocked

The next user-directed continuation started clean at `9d8de7c169efa28777b078fb3b36714f13402a46`, on the same research branch without upstream. Fixed five bundle source/binary identities, five protected files,333 earlier ledger files, previous mission/closeout evidence and nonworking refs passed another starting audit. No rebuild, numerical change, monitor edit or threshold change occurred.

This time the initial fresh read-only preflight **PASSed**, actual exit0, with **Awake and top-resumed verified**, same physical device, active-run absence, critical safety, storage and approved installed APK gates satisfied. The unchanged trace runner then exclusively created the preregistered `cohort-1-awake` directory. Its own separate fresh preflight also **PASSed**. APK audit and staging/hash checks completed before the runner's immediate prelaunch top-resumed check.

That final check stopped with **TOP_RESUMED_UNKNOWN**. Its saved response contains142,962 characters, a valid ActivityManager response header, and zero `topResumedActivity` or `mResumedActivity` fields: **VALID_RESPONSE_NO_FIELD**. The response is preserved privately in the new directory. The ADB wrapper accepted the command; its numeric command exit was not separately retained in the failure file, and no stronger exit-capture claim is made. The historical153,912-character missing-field response remains intact. Recurrence establishes another missing-field observation, not its cause or an alternative monitor's equivalence.

The failure happened **before probe launch**: no owned-PID announcement, probe stdout, probe completion or identity file exists. Actual runner exit was1; actual probe exit and partial probe-raw hash are **null**. Probe teardown is **NOT_APPLICABLE_NO_PROCESS**. This attempt contributes **0 OS processes,0 graphExecute calls and0 completed roles**. It is not an arithmetic failure or a partial numerical result. The power query after the final top check was never reached; Awake was verified at the two preflights, while wakefulness at the later failure instant is unmeasured. This attempt is therefore not labeled a new Dozing episode.

No retry, alternative query, wake/unlock, foreground takeover, force-stop, cleanup or further device operation followed the stop. Both new preflight records and the failed prelaunch response remain in new private evidence. `cohort-1-awake` is now a preserved preflight/prelaunch-only directory and must never be reused; a later authorized attempt needs a new unique directory and a complete fresh safety entry. The original five-role, three-process plan and both digest contracts are unchanged. No pilot is promoted, and formal coverage/digest agreement remains unmeasured.

The public allow-list export records the new missing-field classification/counts and adds **BLOCKED_TOP_RESUMED_STATE**, while retaining **BLOCKED_FORMAL_COHORT**, the historical pilot values and all other restrictions. Its `formal_closeout` object preserves the earlier dffbc50-based Dozing closeout; that historical object does not describe the newly accepted preflights. No formal numerical, observer-free, lifetime, performance or regression-closure certification is granted. Performance and fresh device/Host/Qnn regressions remain NOT_EXECUTED; Project Formal stays BLOCKED_SCOPE / NOT_EXECUTED. No new6031 observation is claimed because no DSP probe ran. Training remains **NO-GO**.

Updates to the preceding41 answers: **#4** is now two accepted fresh Awake/top-resumed preflights followed by an unsafe final top-resumed check, so overall cohort entry failed; **#6** remains no actual process launch; **#25** remains runtime unexercised; **#26** now includes a fresh VALID_RESPONSE_NO_FIELD recurrence, with failure-instant wakefulness unknown; **#27** excludes this new prelaunch-only attempt; **#38** remains BLOCKED_FORMAL_COHORT with additional BLOCKED_TOP_RESUMED_STATE. All formal maxAbs values, process exits, three-process digest agreement and performance values remain unavailable as stated in the preceding table. Preserved pilot results and the Full Attention/Full1-layer/Full19-layer/TRAINING_READY restrictions remain unchanged.

Closeout validation for this attempt: fresh **Fast PASS**, actual exit0/105.6s (outer PASS1/FAIL0/SKIP0); unchanged public exporter original18 plus blocked-closeout14 rejection cases **PASS**. The actual new missing-field response was independently rejected by the Host parser without any additional device command. The allow-list public export completed successfully; its success describes projection validity, not numerical/formal certification.

## 2026-10-20 — Exclusive cohort directory recovery preregistration

Following another explicit user retry instruction, the checkout started clean at `ec6428e300f03bbeee9b2aded3e553327b94d535`. Fixed five bundles, protected files/refs,333 earlier ledger files and all existing mission evidence passed the starting audit. The initial fresh read-only preflight passed Awake, top-resumed and all remaining entry gates. No new DSP process has yet been counted by this preregistration.

The earlier `cohort-1-awake` directory is retained as failed prelaunch evidence. To honor the requirement for a new exclusive directory, the Host certifier now accepts the existing cohort names with an optional eight-lowercase-hex suffix, directly within the same private mission root. Pilot names, other process slots, malformed suffixes, nested/outside paths, imported pilot mode and missing actual exits remain rejected. The unchanged trace runner still enforces exclusive directory creation. Seven valid directory cases and12 rejection cases passed the Host self-test with zero device calls. An AST comparison confirms that role-contract digest calculation and every other certification predicate are unchanged.

This is a Host directory-selection change only, outside every frozen compiled-source manifest. The exact source diff/hashes and the three intended unique directories are recorded privately before any arithmetic. DSP/Host/probe bytes, fixtures, old gates, five-role ordering, three independent process requirement, legacy/supplemental digest definitions and the trace runner remain unchanged. Fresh runner preflight is still required immediately before each process. This preregistration grants no numerical/formal, performance or regression certification and does not resolve the missing-top-field incident. Training remains NO-GO.

Directory-selection gate: fresh **Fast PASS**, actual exit0/119.3s (outer PASS1/FAIL0/SKIP0), before committing the Host change and attempting a fresh runner entry.
