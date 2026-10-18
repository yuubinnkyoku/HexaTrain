# G1 S2 Mixed QKV — diagnostic research

The fixed Mixed QKV candidate passed Standalone, S2-native, lossless Heads and
adjacent existing FP32 QK on DSP in three independent processes. Q/K match the
independent double-internal Host candidate words; V preserves strict IEEE HVX and
ordered FP32 words. Double raw QK and cumulative P remain unverified. The double
projection adds about8ms of DSP arithmetic per invocation.

## A. Source / contract and preregistration

Base `984bea42006c043111b3d5dec353a73b426b4192`, branch
`investigation/g1-s2-mixed-qkv-df-hvx-v`. Existing local research history and
protected refs are retained. A later explicit user instruction authorized ordinary
push of this dedicated branch; main/PR/merge/history rewrite remain excluded.

The generalized training source projects LN1 Y through distinct wq/wk/wv registry
entries: NN MatMul, both transpose flags false, T32/D64/H2, row-major Y[32,64],
weights[64,64], Q/K/V[32,64]. Diagnostic rank4 unit leading axes preserve physical
layout. Head0 channels0–31, Head1 channels32–63. Saved Q/K/V and heads remain live
through attention backward; weights are immutable external inputs here. Gate also
consumes Y and is outside this experiment. Production builder/optimizer/checkpoint
and training code are unchanged.

Separate `HexaMixedQkv` Host-preparation/DSP package adds only `MixedPrecisionQKV`
and `MixedPrecisionQKVNoReceipt`. Old HexaFp32 51 operations and HexaSelectiveLn1
four operations retain names/order/ABI/numerical sources/frozen binaries. All
three Host/DSP package pairs register before context creation. Host preparation
rejects arithmetic; the independent Host numerical harness is an oracle only.

Q/K exactly convert FP32 operands to double, use q=0..63 double products/serial
additions, then one final FP32 store. No FMA/fast-math/reassociation/tree reduction,
clipping, FP16 or BF16. V calls unchanged `hexatrain_ordered_hvx_qkv_nn64` once:
128B output-column tiling, q-ordered separate FP32 multiply/add rounding and strict
IEEE qfloat. It does not calculate old Q/K and discard them. Old full-HVX QKV is
a separate same-bundle control.

Before first device execution, `mixed_qkv_device_plan.json` froze Q/K Host words0,
V ordered words0, core/holdout independent-double Q/K/V maxAbs≤.002, LN saved
state≤.002, local LN backward≤.02, head words0 and adjacent FP32 NT QK maxAbs≤.01
with ordered words0. Arithmetic-only stress requires finite output/word identity,
separately from model-core absolute gates. Old P contract and source Scale/Mask/
Softmax mathematics are unchanged.

## B. Implementation / fixed codegen and actual runtime

Observed receipt20,512 INT32 words: header32, Y2048, three weights4096 each,
Q/K/V2048 each. Marker `0x48383301`, dedicated synchronous counter, geometry,
alignment, completion, one HVX-call count and separate Q/K/V timers. NoReceipt
removes receipt/counter traffic. Wrapper checks rank4 geometry, distinct weight
ranges, output/receipt disjointness, finite operands/results and complete writes.

Build source `32b0e2797151ca653225712e9b72871f51c03132`; clean device-tested source
`934430e3458da3e4f5c117ed2dcc574dcd0c5b68`. Manifest hashes bind the unchanged
arithmetic/probe sources. Fixed QAIRT2.48.40.260702151143, QNN API2.37.0, HTP V81,
Hexagon SDK6.6.0.0/tools19.0.07, NDK26.2.11394342.

Scalar flags: `-O2 -G0 -fPIC -ffp-contract=off -fno-fast-math -fno-vectorize
-fno-slp-vectorize -fno-unroll-loops`. Unchanged HVX translation unit additionally
uses128B HVX/qfloat, `-enable-xqf-gen=true`, `-hexagon-qfloat-mode=strict-ieee`.
Hidden linkage isolates the helper; provider/qhpi_init are explicit public exports.

Q/K assembly: `convert_sf2df`, partial products
`dfmpyfix/dfmpyll/dfmpylh/dfmpyhh`, serial `dfadd`, `convert_df2sf`. LLVM IR has no
fast flags/FMA intrinsic. Pipelining retains the increasing-q accumulator dependency;
no inner-loop accumulator spill/tree reduction. Static frames: wrapper Observed136
bytes/NoReceipt96, projection32, HVX helper0. These exclude callee/runtime/backend
workspace. No new double libm imports; remaining imports are QHPI, memory/string
operations and frame support. Registration/linkage/API/teardown all passed on DSP.

V helper's entire244-byte instruction sequence is identical in the actual frozen
old DSP library and new library (also old object), SHA256
`78c4d391f72efbdd39946cdf65a7d5212dcd7ed5269b26a40a124fbe499fe306`.
It contains vector multiply/add/conversion/load/store. Actual local wrapper path
calls double projection twice and that helper once; DSP completion and V words
confirm execution. Evidence combines immutable code, actual call path and outputs,
not just an HVX resource declaration. No PMU instruction count is claimed. Rounding/
subnormal controls pass within bounded coverage, not every possible IEEE input.

## C. Standalone / independent references

Frozen44 source-derived +16 independent holdout +8 Standalone arithmetic controls.
Original fixtures unchanged. Existing Wq/Wk reused; missing holdout Wv uses existing
`precision::next`, seed707003+case_index×104729, amplitude.0625. Controls include
asymmetric heads/weights, transpose sensitivity, cancellation, signed zero,
small/large finite and subnormal input. Independent double/ordered references use
`one::mm`, never the new body/HVX implementation.

New Host suite source `0522347f8d6719d3616231fa6646ab9f2621a806` passed three
processes,32 graph contracts and preparation tests. All278,528 Q/K words match
independent double NN;217,966 differ from old ordered FP32, intentionally. CPU
arithmetic rejection2/2, invalid signature rejection2/2. Supplemental fixture SHA256
`8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb`;
original S2 fixture remains
`492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02`.

Actual DSP Standalone passed all68 cases×four observer conditions in every final
process: Q/K Host words0, V ordered words0 and same-operand old-HVX V words0.
Signed-zero/subnormal controls also words0. API success, DSP marker, required
finite outputs, immutable inputs, complete writes, poison/canary and no arithmetic
CPU fallback are separately checked.

| Actual DSP, source44 + holdout16 | Q maxAbs | K maxAbs | V maxAbs |
|---|---:|---:|---:|
| Same received FP32 Y, independent double NN | 2.3707031538e-7 | 2.3790339831e-7 | 5.1214479058e-6 |
| Ideal double LN1→double NN, native stages | 6.9925630575e-7 | 6.2663098443e-7 | 5.5597568434e-6 |

Both retain .002 gates. First comparison isolates projection error on the received
operand; second includes S2 input error. Stress maxima Q.000449845,K.000810368,
V.008156732 are arithmetic-word/finite results; stress V is not claimed to meet
the model-core .002 gate. No gate/input distribution was changed.

## D. Native S2→Mixed QKV / observer boundaries

All32 configurations ran: four stages×baseline/Mixed×Observed/NoReceipt×Tap/NoTap.
Each final process has22,632 numerical predicates,1,984 case executions and separate
performance repetitions. Native stages use all60 core/holdout cases. Stages:
projection alone; S2+projection+old local LN Backward; above+Heads; above+existing
ordered FP32 raw QK. NoTap Y/xhat/inverse use NATIVE IDs feeding projection and
old Backward. Later Q/K/V are NATIVE into Heads, Q/K heads NATIVE into raw QK.
Shared graph builder removes INT32 receipts and selects all old NoReceipt ops,
not just the new Mixed specialization.

Observed receipts compare actual S2 Y to projection input, saved h/inverse to old
Backward operands, all three weights, projection outputs to HeadSplit inputs and
head outputs to QK inputs. All words match; no ideal Host Y substitution. Each
NoReceipt/NoTap graph independently passes its visible final outputs/references
and valid Observed anchors, variant mismatches0. Invisible intermediates are not
claimed directly measured. Observed correctness does not replace NoReceipt gates.

| Native S2 saved state/local Backward | Worst maxAbs | Gate |
|---|---:|---:|
| Y | 4.3431564878e-7 | .002 |
| xhat | 4.9209063580e-7 | .002 |
| inverse | 1.5488809083e-5 | .002 |
| dX | 6.6563704138e-5 | .02 |
| dGamma | 2.0294970653e-5 | .02 |
| dBeta | 7.8966841102e-6 | .02 |

S2 Host words and saved-state consumer words match. X/gamma/beta/dY/weights remain
correct frozen external inputs, not actual upstream G1 producers. Local dY is
not the complete Attention/V/Gate gradient aggregation.

## E. Lossless Heads / adjacent existing FP32 QK

Six Q/K/V heads copy the correct physical channel halves, words0/no head mixing.
Observed producer/consumer receipts pass; NoReceipt final heads pass independent
references. Existing raw QK receives Qh0,Qh1,Kh0,Kh1, NT32 with right operand
transposed, q=0..31 ordered FP32 multiply/add only. No Scale/Mask/Softmax.
Conditioned-double local error: H0 max1.1120751310e-5,H1 max2.3500583737e-5≤.01;
ordered output words0 across observers/processes. These are local QK gates,
not cumulative-P gates.

No double NT QK implemented. This reproduces S2+C6's S2 and double Q/K projection
portions on DSP, leaving its assumed double raw-QK boundary open. Previous Host
S2+C6 joint-P PASS cannot transfer to this FP32-QK graph. Optional DSP-Q/K→Host-P
counterfactual was not run; it would remain Host-mediated. No native P/P→Backward
certification is claimed.

## F. Same-bundle performance / cost

Performance follows accuracy PASS. Per process, fixture0, one discarded warm-up
then31 samples/graph. First is the graph's first execute, not a cold device boot.
Tables show ranges across three process statistics, in ms; aggregate retains exact
µs for all32 variants, including all p10/p90/first/finalize values.

| Standalone Observed+Tap | DSP arithmetic median | Receipt-inclusive median | Execute median | Execute p10 | Execute p90 |
|---|---:|---:|---:|---:|---:|
| Existing HVX QKV | .313–.314 | .804–.811 | 3.804–4.091 | 3.700–3.951 | 4.028–4.223 |
| Mixed QKV | 8.335 | 8.832–8.837 | 12.035–12.187 | 11.823–12.045 | 12.353–12.556 |

Mixed arithmetic separates Q/K8.229ms and HVX V.106ms. Total projection arithmetic
costs about26.6× the same-bundle three-HVX baseline, +8.02ms. Numerical feasibility
passes; production performance feasibility is not established. NoReceipt internal
math/receipt timers are unavailable and export as null. Wall includes QNN/API,
other nodes and diagnostic work, distinct from DSP math/receipt windows.

| NoReceipt/NoTap graph | Existing HVX execute median | Mixed execute median | Mixed first | Mixed finalize |
|---|---:|---:|---:|---:|
| Projection only | 3.971–3.999 | 11.847–12.092 | 12.668–12.808 | 41.652–51.320 |
| S2→projection + local LN Backward | 5.911–6.248 | 14.011–14.287 | 14.285–14.908 | 29.855–56.804 |
| Above→Heads | 6.280–6.669 | 14.646–14.776 | 14.697–14.880 | 30.854–33.283 |
| Above→existing FP32 QK | 8.419–8.716 | 16.522–16.739 | 17.143–17.402 | 30.138–48.907 |

All comparisons use the same final bundle/backend/fixtures/policy, not previous
research timings. Cohort remained31°C/status0/battery70%, external power absent;
no thermal throttling recorded.

| Instantiated NoReceipt/NoTap metadata, either projection | Nodes | Tensors | Native logical bytes | Host buffer lower bound bytes |
|---|---:|---:|---:|---:|
| Projection | 1 | 7 | 0 | 81,920 |
| S2→projection + local LN Backward | 3 | 17 | 16,512 | 99,328 |
| Above→Heads | 4 | 23 | 41,088 | 99,328 |
| Above→existing FP32 QK | 5 | 25 | 57,472 | 91,136 |

These are declaration totals, not measured physical DSP allocation; exclude guards,
fixtures, code, transport and QNN workspace. Mixed adds zero launches/explicit
scratch; existing S2 center scratch512 bytes remains. Mixed receipt82,048 bytes vs
old HVX82,016. Static frames in B do not predict dynamic allocation/scheduling.

Per layer/microbatch: Q/K262,144 double product/add terms, V131,072 FP32/HVX terms.
At19 layers×8 microbatches:39,845,888 double Q/K terms and19,922,944 V terms. Future
double NT raw QK would add9,961,472 terms in that logical scope and is not implemented.
Holding HVX resource during scalar double work may affect scheduling; wider
scheduling/memory costs are unmeasured. No multiplied wall-time throughput forecast.

## G. Safety / preserved stops and manual resume

Research-specific monitoring accepts ordinary battery decline, absent external
power and noncritical thermal status; project defaults unchanged. Pure11-case
policy tests retain OS critical status≥4, serious battery faults, shutdown-imminent
level0 and contradiction rejection. Unknown optional telemetry differs from unknown
critical protection. Identity, awake/top, active runs, storage, SDK/compiler/APK/
package/fixture and protected-state gates stay required.

Historical read-only Host ADB preflight hung before identity/awake returned (last
elapsed194s): no staging/probe. Own Host preflight terminated; OS denied direct
ADB-client cancellation. Other ADB processes/server not stopped. Evidence preserved.
Runner now bounds Host preflight180s without asserting unknown child cancellation
succeeded; this timeout path was not deliberately induced on device.

After user reconnect, full preflight passed and files staged/hash-matched, then
immediate top-resumed was unknown. Stop before execution. After explicit manual
screen-on/unlock confirmation, fresh checks and pilot-resume-2 passed. Pilot is
excluded from final-1/2/3. Same frozen candidate then passed three processes and
nine fresh regressions. Initial headless caller stopped on a stale native exit
code despite a PASS safety ledger; no test ran. Isolated corrected invocation
rechecked all conditions before fresh smoke/W8. These are preserved separate events.

Same physical device, APK, active-run absence, awake/top and fixed SDK checked
before every process and immediately before execution. Focus takeover0; runtime
critical/awake monitoring. Cohort+regressions:25 samples, max34°C/status0,
battery68–70%, no external power. Smoke/W8 each34°C/status0/battery68%. No critical
alarm. Approved installed APK/test APK snapshots were reaudited and explicitly
selected with SkipBuild/SkipInstall; no replacement. Profile-generated APK has a
different private identity and was not substituted.

No automatic wake/unlock, force-stop, app UI/input, foreground takeover, endpoint
guessing, root, SDK/clock/power policy change. Owned cancellation is exact-PID/
cmdline restricted; no runtime cancellation needed in successful processes.

## H. Validation / reproducibility / provenance

Fast PASS after evidence/exporter changes; Host profile PASS370.3s; Qnn profile
PASS853.7s (fresh Host378.9s, QNN build461.6s, APK audit4.3s). New Host, graph/
preparation, monitoring/preflight/headless/exporter SelfTests PASS. Frozen math/
probe/binaries were not rebuilt during resume; profile identities in aggregate.

Twenty-four unchanged legacy Host binaries freshly executed/hash-bound:23 exit0,
old P expected exit4/original nine failed predicates. R0–R5 attribution digest
`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7` unchanged.
Initial Host closure command missed two fixture arguments and returned1; corrected
command passed, empty first log retained. No math repair/gate change. Broad profile
PASS does not erase the old contract FAIL.

| Fresh device regression, all three packages registered | Result |
|---|---|
| S2 LN1/saved state/local Backward | PASS,60 cases×8 variants/2640 predicates |
| Scalar/HVX FFN2 K128, LN1/ordered words | PASS, failures/word mismatches0 |
| Scalar/HVX QKV K64, Heads | PASS, failures/word mismatches0 |
| Forward scalar QK / Backward raw+Scale | PASS/words0, includes LN2/Softmax backward controls |
| Forward Scale | Custom PASS; standard H0 max.00868434926008,0/44 FAIL; H1 max.0102382000946,1/44 FAIL |
| Mask | Custom PASS; standard176 required predicate FAIL reproduced |
| Softmax | Custom local PASS in old native chain; standard224 local FAIL/nonfinite0, expected exit4 |
| Original cumulative P chain | Local0/upstream0/words0; cumulative36 FAIL, expected exit4; native Backward not executed |
| Device smoke | Fresh HTP SDK/context PASS, API2.37.0/Build ID match, fallback false |
| Muon W8 | Fresh114/114; maxAbs1.490116119e-8, relative L2 4.40269854e-8, cosine1; RPC/finite PASS, fallback false |

Derivative regression probes change only initialization to register Mixed alongside
old packages, preserving old arithmetic/graphs/checks. Fresh smoke/W8 use the
approved existing APK/unchanged optimizer ABI. APK does not load standalone Mixed:
this is a fresh independent production-path regression associated with the frozen
bundle, not W8 execution of the new op. Mixed is exercised by its separate probe.

Each final process:22,632 predicates,1,984 case executions,32 variants; gates/API/
markers/immutable inputs/complete writes/finite/poison/canary/no arithmetic CPU
fallback/context-device-backend teardown PASS. All three share numerical digest
`568ec1bdff151db448021538681b38f49e54b5330e52b1b2718e3dccfb1a2549`.
Timing/receipt counters/device IDs excluded. Cross-process digest equality and
cross-observer words0 are separate checks. Pilot same digest, excluded from cohort.
Historical safety stops carry no numerical certification.

| Frozen artifact | SHA256 |
|---|---|
| Mixed DSP | `787dd5dcc97a7e5e2883d8671171288e023a23e40a64b7246572c5c5d30001c5` |
| Mixed Host preparation | `829fa8d78b6d61bc2c078c7ab32484fdfb3d37fb7f132ec89fc9f1c6a6823354` |
| Mixed probe | `39247f7d6cb3d4683452c818c07139ad0c5852e075776c80e9a67db08e2c2c86` |
| S2 regression derivative | `e7f9bf7c612e413be990da5d16771b9a6aab3660f98bd5afe2f4ae2a976a2a72` |
| Existing regression derivative | `b0bfef759832c8eb48a41ea0695ceceaa5251734833851b2d32ed950c39009de` |
| Supplemental fixture | `8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb` |

Compiler hashes, Host/static metrics, per-process errors/performance, thermal ranges
and regressions are allow-listed in summary. Private evidence remains under this
study's build reports. No raw tensors/device IDs/endpoints/APK hashes/SDK binaries/
private logs/absolute paths exported. Initial protected refs and five artifact
hashes rechecked at closeout, including40k
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
Final HEAD/clean tree/ordinary dedicated-branch push recorded privately after commit.
Evidence-only changes do not relabel tested source or change the numerical binary.

## I. Architecture decision / final questions

Next invest in **double NT raw QK precision closure**, reusing the ordered double
contract and actual native heads. Projection/runtime/ownership prerequisites now
pass; double raw QK is the missing mathematical boundary in Host S2+C6. Match
same-operand reference words first, then integrate existing Scale/Mask/Softmax and
reassess cumulative P/sensitivity with unchanged budgets. S2+C6 Host dScores bound
2.459100e-4 has only about1.64% allocation headroom: actual DSP P arithmetic and
observer-free native integration still need proof. No such NT/P work implemented.

The about8ms projection overhead merits later optimization/scheduling work after
precision closure. It does not justify changing rounding, V, fixtures or gates.
Numerical and performance feasibility are separate judgments.

| Question | Answer |
|---|---|
| 1. Single Mixed invocation? | Yes, double Q/K twice + unchanged HVX V once; two specializations. |
| 2. Double Q/K on actual DSP? | PASS,68 Standalone/60 native cases. |
| 3. Same-operand Q/K Host words? | All match, including bounded zero/subnormal/stress controls. |
| 4. Actual HVX V path? | Yes; identical244-byte helper, actual call path, DSP completion/output words. |
| 5. V ordered/old-HVX words? | All match for identical operands. |
| 6. V kernel changed? | No source/machine-code change. |
| 7. Weights distinguished? | Yes, distinct inputs/range checks/asymmetric fixtures/receipts. |
| 8. Existing51+4 ABI? | Preserved; all three package pairs coexist. |
| 9. True S2→Mixed NATIVE? | PASS, real DSP Y, no Host substitution. |
| 10. Producer/consumer Y words? | All observed receipts match. |
| 11. Saved-state→old Backward? | PASS, same NATIVE IDs, saved words/local gradient gates. |
| 12. Observer-free accuracy? | PASS for own visible final outputs; no direct claim on invisible intermediates. |
| 13. Lossless Heads? | PASS, six heads, all words. |
| 14. Adjacent FP32 QK? | PASS NT/head mapping/words0, local H0/H1 1.1121e-5/2.3501e-5≤.01. |
| 15. S2+C6 portion reproduced? | S2 + double Q/K projection only; raw QK still FP32, P unverified. |
| 16. Runtime/linkage/rounding issue? | None observed in frozen cohort/controls; bounded IEEE coverage. |
| 17. Same-bundle arithmetic cost? | Mixed8.335ms vs HVX.313–.314ms, about26.6× total projection/+8.02ms. |
| 18. Launch/stack/scratch/buffers? | Extra launch0; wrapper136/96, projection32-byte frames; scratch0 new; table F. |
| 19.19×8 cost? |39,845,888 double Q/K +19,922,944 V terms; wider wall/scheduling unmeasured. |
| 20. Ordinary power/thermal stopped research? | No; battery-only monitored, critical protections retained. |
| 21. Critical gates maintained? | Yes; unknown identity/top stopped pre-execution, manual resume fully rechecked. |
| 22. Three final processes? | Yes, unchanged source/binaries/fixtures; identical digest, pilot excluded. |
| 23. Fresh HVX FFN2/QKV regressions? | PASS alongside Mixed package. |
| 24. Fresh W8? |114/114 PASS; independent approved-APK regression, not Mixed-op execution. |
| 25. Old P FAIL? | Preserved, nine Host failed predicates/36 old device cumulative failures. |
| 26. Ready for double QK research? | Yes, native projection/Heads prerequisites verified; raw double QK not implemented. |
| 27. Remaining P→Backward? | Double raw QK, native Scale/Mask/Softmax cumulative budget, observer-free P, saved P lifetime/integrity and backward consumer gates. |
| 28.40k→60k/200k? | **NO-GO**. |

Classifications: `DSP_MIXED_QKV_STANDALONE_VERIFIED`,
`DSP_DOUBLE_QK_PROJECTION_VERIFIED`, `HVX_V_PATH_PRESERVED`,
`S2_MIXED_QKV_NATIVE_SEGMENT_VERIFIED`, `MIXED_QKV_HEADS_LOSSLESS_VERIFIED`,
`MIXED_QKV_ADJACENT_QK_VERIFIED`, `OBSERVER_FREE_SEGMENT_VERIFIED`,
`DSP_MIXED_QKV_PERFORMANCE_MEASURED`, bounded to these fixtures/graphs.

Full Attention/full1-layer/full19-layer/training correctness remain unverified.
Backward scatter, V/Gate branches, projection gradients, full LN1 dY aggregation,
materialization/accumulation and integration remain.35/40-node Tap incident stays
`REPRODUCED / CAUSE UNRESOLVED`;6031 stays `UNRESOLVED / DORMANT / WATCH`.
No training update/Final test/LR-optimizer-codec change/checkpoint decode or write.
40k→60k/200k remain **NO-GO**.
