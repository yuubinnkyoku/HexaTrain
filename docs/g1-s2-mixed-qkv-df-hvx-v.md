# G1 S2 Mixed QKV — diagnostic research

## A. Source / contract and preregistration

Base `984bea42006c043111b3d5dec353a73b426b4192`, branch
`investigation/g1-s2-mixed-qkv-df-hvx-v`. Local unpublished research is retained.

The generalized training source projects the same LN1 Y through distinct wq/wk/wv
parameter registry entries, with NN MatMul, neither operand transposed. T32/D64,
row-major Y[32,64], weights[64,64], outputs[32,64]. Head selectors map channels
0–31/32–63 independently. Rank4 diagnostic unit axes preserve physical layout.
Q/K/V and per-head saved values remain live through attention backward. The
source's Gate projection also consumes LN1 Y; it is outside this experiment.
Weights are fixed external graph inputs, not training updates.

The new independent `HexaMixedQkv` preparation/DSP package adds only
`MixedPrecisionQKV` and `MixedPrecisionQKVNoReceipt`. Old HexaFp32 51 operations
and HexaSelectiveLn1 four operations and their numerical sources stay unchanged.
All three packages are registered before context creation. Production builder,
optimizer, checkpoint and training code are unchanged.

Q/K use exact FP32-to-double conversion, increasing-q double product/addition,
one final float store. No FMA, fast-math, reassociation, tree reduction, clipping,
FP16 or BF16. V calls the unchanged `hexatrain_ordered_hvx_qkv_nn64` once, using
the existing 128B output-column tiling, q-ordered separate FP32 multiply/add and
strict IEEE qfloat toolchain. It does not calculate old Q/K and discard them.
The old full-HVX QKV graph is a separate same-bundle baseline control.

Before first device measurement, gates and receipt layout are frozen in
`host_tests/fp32_custom/mixed_qkv_device_plan.json`: Q/K Host candidate words0,
V ordered FP32 words0, core Q/K/V independent double maxAbs≤.002, saved LN state
≤.002, local LN backward≤.02, lossless head words0, adjacent existing FP32 NT
QK maxAbs≤.01 with ordered words0. Arithmetic-only stress requires word identity
and finite output, separately from the model-core absolute gates.

Observed receipt has 20,512 INT32 words: header32, Y2048, three distinct weights
4096 each, Q/K/V2048 each. Marker `0x48383301`, dedicated synchronous invocation
counter, geometry, alignment, completion and HVX-call count, separate Q/K and V
timers. NoReceipt specialization has no receipt output or counter traffic.
Host package rejects arithmetic; independent Host harness is an oracle only.

Frozen supplement preserves source44 and holdout16. Existing Wq/Wk are reused;
missing holdout Wv uses existing `precision::next` generation, separate seed
`707003+case_index*104729`, amplitude .0625. Eight standalone-only controls add
signed zero, small/large finite values, cancellation, asymmetric heads/weights,
transpose-sensitive input and subnormal input. Original fixtures are unchanged.
Independent double/ordered FP32 references use `one::mm`, never the new body.
The Host body/reference test checks 278,528 Q/K words; 217,966 differ from the
old ordered-FP32 model, as expected for the different mathematics.

## B. Implementation / fixed codegen

Classification is **BLOCKED_DEVICE_SAFETY / UNRESOLVED**, specifically missing
physical identity/awake evidence after an unresponsive Host ADB preflight.
No probe execution or package staging started. This is an infrastructure/safety
evidence gap, not an observed numerical or double-runtime failure.

Build source is `32b0e2797151ca653225712e9b72871f51c03132`. Fixed QAIRT build
`2.48.40.260702151143`, QNN API2.37.0, V81, Hexagon SDK6.6.0.0/tools19.0.07,
NDK26.2.11394342. Scalar flags include `-O2 -G0 -fPIC -ffp-contract=off
-fno-fast-math -fno-vectorize -fno-slp-vectorize -fno-unroll-loops`.
The unchanged HVX translation unit additionally uses 128B HVX/qfloat,
`-enable-xqf-gen=true` and `-hexagon-qfloat-mode=strict-ieee`.
Hidden visibility isolates the linked helper; provider and qhpi_init are explicit
public entry points. No new global counter or symbol overlaps old namespaces.

Q/K assembly contains `convert_sf2df`, double partial-product sequence
`dfmpyfix/dfmpyll/dfmpylh/dfmpyhh`, serial accumulator `dfadd`, and
`convert_df2sf` before float stores. LLVM IR has no fast flags or FMA intrinsic.
Software pipelining overlaps loads/product formation but retains the increasing-q
accumulator dependency. The 32-byte projection frame saves callee registers;
the audited inner loop has no accumulator spill/tree reduction. Observed wrapper
frame136 bytes, NoReceipt96 bytes, unchanged HVX helper0 bytes. These frames
exclude backend workspace, callee stack and runtime overhead.

The V helper's complete 244-byte instruction sequence is identical between the
frozen old DSP package and new linked library (also old object), SHA256
`78c4d391f72efbdd39946cdf65a7d5212dcd7ed5269b26a40a124fbe499fe306`.
It contains actual vector multiply/add/conversion/load/store instructions.
New wrapper call sites invoke double projection twice and this V helper once.
This is static execution-path evidence, not proof that a new DSP invocation ran.
No new double libm helper imports are required. Remaining runtime imports are
ordinary QHPI queries/registration, memcpy/memset/strcmp and runtime frame support.
Runtime resolution, DSP rounding/subnormal behavior remain **UNVERIFIED**.

## C. Standalone Host results / frozen inputs

New Host suite source `0522347f8d6719d3616231fa6646ab9f2621a806` builds independent
references, new arithmetic, graph contracts and preparation rejection tests.
Three independent Host processes reproduce identical full supplemental fixture:
`8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb`.
Original S2 fixture hash remains
`492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02`.

All278,528 Q/K words match independent double NN outputs across68 cases; the old
ordered-FP32 candidate differs in217,966 words. Signed zero/subnormal controls
are Host evidence only. V's Host ordered reference is not an executed HVX result.

| Host model, source44 + holdout16 | Q maxAbs | K maxAbs | V ordered-reference maxAbs |
|---|---:|---:|---:|
| Same received FP32 S2 Y vs double NN | 2.3707031538e-7 | 2.3790339831e-7 | 5.1214479058e-6 |
| Ideal double LN1→double NN | 6.9925630575e-7 | 6.2663098443e-7 | 5.5597568434e-6 |

Both comparisons retain the preregistered .002 core/holdout gate. Set2 stress is
judged by finite output/arithmetic words, not a model-quality absolute threshold.
Double reference calls `one::mm`, not `mixedarith::project` or the HVX body.
Frozen expected words come from the independent Host model, not Android CPU
arithmetic. Preparation tests reject CPU arithmetic2/2 and invalid signatures2/2.
DSP wrapper checks distinct input weight ranges, output/receipt disjointness,
rank4 shapes, finite inputs/outputs and complete fixed loops. Device canary,
poison, immutable input, DSP marker and API checks are implemented but **NOT RUN**.

## D. Native S2→Mixed QKV design / certification boundary

The32 Host graph contracts cover four stages × baseline/candidate ×Observed/
NoReceipt ×Tap/NoTap. All shape, declared output, receipt absence, node count and
ownership checks PASS. Stages are Standalone projection; S2+projection+old local
LN Backward; that graph+lossless QKVHeads; then existing ordered FP32 QKᵀ.
Baseline is existing full HVX QKV, using the same S2 Y and distinct weights.

In NoTap, S2 Y/xhat/inverse are NATIVE, same tensor IDs feed Mixed projection and
old LN Backward. In later stages Q/K/V are NATIVE to head selection; Q/K heads
are NATIVE to NT QK. APP_READ final outputs are evaluated for each graph itself.
Observed receipts compare actual LN producer Y to Mixed input, saved h/inverse
to old Backward operands, all three weights, projection outputs, head input and
output words, and QK inputs/outputs. No host ideal Y is injected in native stages.
X/gamma/beta/dY and weights remain correct frozen external inputs, not actual
upstream G1 producers.

NoReceipt/NoTap invisible tensors are not claimed directly observed. Its visible
Q/K/V or heads/raw scores and local LN gradients have independent references and
cross-variant anchors. This complete design has **not executed on device**;
neither native integrity nor observer-free accuracy is certified in this run.
Prior S2 standalone/native saved-state PASS remains prior-binary evidence only.

## E. Heads / adjacent consumer

Existing QKVHeads copies physical channels0–31/32–63 for each Q/K/V independently.
Observed validation checks every copy against actual producer receipt words.
Existing ForwardQKRaw receives Qh0,Qh1,Kh0,Kh1 and computes NT32 increasing-q
ordered FP32 dots; no Scale/Mask/Softmax is included. Host graph contracts confirm
the intended geometry and ownership; actual head losslessness, transpose behavior
and .01 local QK device gate remain **NOT RUN**.

No double NT QK was implemented. The new Q/K contract covers only the projection
part of S2+C6. Previous S2+C6 Host joint-P PASS assumes double QK too and cannot
be transferred to this ordered-FP32 adjacent graph. Optional DSP-Q/K→Host-P
counterfactual was not run because no DSP Q/K was obtained.

## F. Performance / logical cost

All execute, DSP math/receipt windows, first/steady/p10/p90/finalize and thermal
performance are **UNMEASURED**. Performance collection is gated behind all required
accuracy predicates. NoReceipt unavailable internal times export as null on a
future verified cohort. No historical-binary speedup or production wall estimate
is inferred from Host arithmetic, assembly or previous HVX timings.

| Source-derived NoReceipt/NoTap graph | Nodes | Tensors | Native logical bytes | Host buffer lower bound bytes |
|---|---:|---:|---:|---:|
| Projection only, either candidate | 1 | 7 | 0 | 81,920 |
| S2→projection + local LN Backward | 3 | 17 | 16,512 | 99,328 |
| Above→Heads | 4 | 23 | 41,088 | 99,328 |
| Above→ordered QKᵀ | 5 | 25 | 57,472 | 91,136 |

These are declaration-derived estimates, not measured DSP allocation. They
exclude canary/guard copies, fixture storage, transport, code, runtime and QNN
workspace. Mixed adds zero graph launches relative to full HVX QKV and zero
explicit scratch; existing S2 center scratch512 bytes remains. Observed Mixed
projection receipt82,048 bytes vs old HVX82,016 bytes, separately from timings.

Per layer/microbatch, Q/K262,144 double product/add terms and V131,072 unchanged
FP32/HVX terms. At19 layers×8 microbatches:39,845,888 double projection terms and
19,922,944 V terms. Future double raw QK would add9,961,472 terms across that
logical scope, but is not implemented. Holding the HVX resource across scalar
double work may affect scheduling; it is unmeasured. Numerical feasibility on
Host is established; DSP numerical and performance feasibility remain separate
unresolved questions.

## G. Safety / bounded stop

Research-specific monitoring accepts normal battery decline, absent external
power and noncritical thermal throttling. The default project policy is unchanged.
Pure policy self-test11 cases includes unknown partial sensor telemetry vs unknown
protection status, OS critical thermal rejection(status≥4), serious battery fault,
shutdown-imminent level0 and contradictions; device calls0. Physical identity,
awake/top-resumed, competing run, SDK/compiler/APK/package/protected identities
and storage gates are not waived. No automatic wake/unlock, force-stop, foreground
takeover, reconnect, endpoint guessing, root or clock/power policy change occurred.

The currently installed, previously authorized APK/test APK were snapshotted
privately before the Qnn profile build. Both use fixed SDK identities. New runner
accepts an explicitly named, audited existing APK for SkipBuild/SkipInstall only;
no candidate installation permission was inferred from an old approval, and no
APK replacement occurred. Profile-generated APK and installed HVX-capable APK
have distinct private identities and are not treated as the same artifact.

The single read-only preflight became unresponsive in Host ADB before stable
device identity/awake were returned. Last recorded elapsed194 seconds; the
preflight did not return its own final ledger. No package staging or probe started.
We identified only the direct Host ADB child of that preflight. The OS denied its
termination(Access denied); the Host preflight itself was terminated. Other ADB
processes/server were not stopped. There was no privilege escalation, automatic
reattach or second device attempt. This cancellation limitation remains private
evidence; it is not described as a successful safe device check. The research
runner now supervises preflight in a separate Host process with a180-second
deadline, terminating only that Host process on timeout and retaining unknown
child cancellation status. This guard is parser-checked and not device-tested.
Unknown external-power telemetry remains null rather than asserted absent.
Normal thermal/battery policy was not the cause, and no current device telemetry
is claimed.

Resume requires restoring responsive Host ADB/device access and re-running all
read-only gates, including known same physical identity and awake/top-resumed.
An outstanding Host ADB client may need manual investigation. No probe is known
to have started, and no live training/process was force-stopped. Future monitored
probe cancellation is restricted to its announced PID after exact cmdline check.

## H. Validation / provenance

Fast PASS (final log identity in aggregate); explicit Host profile PASS370.3s, including parity fault
battery; Qnn profile PASS853.7s, including fresh Host contract+diagnostic suite
378.9s, fixed-QNN APK build461.6s and APK audit4.3s. Separate new Host suite also
PASS:68 cases, three numerical processes,32 graph configurations and preparation
rejection. Exact profile log identities are recorded in the public aggregate.
The explicit existing-APK Host audit also PASS; installed identity could not be
checked because ADB did not respond. Host-only package/codegen build PASS; compilation is
not runtime certification. Pure monitoring/preflight/headless/exporter tests PASS.

Twenty-four unchanged legacy Custom Host binaries were freshly executed and
hash-bound:23 exit0 and original probability contract expected exit4. Original
9 failed predicates and R0–R5 attribution digest
`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7` reproduce.
This is reuse of unchanged Host executables for fresh executions, not rebuilding
them into the new QHPI library or transferring past device success. Initial
legacy closure invocation omitted two required fixture arguments and returned1;
the command was corrected from the existing runner, and its empty first log is
retained. No mathematical repair or gate change was involved.

All required new-bundle device regressions are **NOT RUN**: S2 saved state,
HVX FFN2/QKV, Scalar QK, backward/forward Scale, Mask, local Custom Softmax,
LN2/Softmax Backward, device smoke and fresh Muon W8. Historical114/114 is not a
new result. Standard Scale/Mask/Softmax and old cumulative-P device negative
controls were not remeasured. No pilot/final device cohort exists and no new
numerical digest is claimed. Overall validation is **BLOCKED**, even though the
Host/build gates pass.

| Frozen candidate artifact | SHA256 |
|---|---|
| DSP package | `787dd5dcc97a7e5e2883d8671171288e023a23e40a64b7246572c5c5d30001c5` |
| Host preparation package | `829fa8d78b6d61bc2c078c7ab32484fdfb3d37fb7f132ec89fc9f1c6a6823354` |
| Mixed probe | `39247f7d6cb3d4683452c818c07139ad0c5852e075776c80e9a67db08e2c2c86` |
| Supplemental fixture | `8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb` |

Remaining probe/compiler hashes and aggregate-only metrics are in the allow-list
summary. SDK binaries, APK hashes, device identifiers, absolute paths, raw tensors
and private logs are excluded. Initial refs and protected5 hashes are rechecked
at closeout, including40k SHA256
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
Final HEAD and clean working tree are recorded privately after the final commit.
No push/PR/merge/history rewrite or checkpoint decode/write occurred.

## I. Decision / final questions

First restore responsive read-only ADB and certify this frozen Mixed candidate
Standalone, then S2-native/Heads/adjacent ordered QK, observer-free and fresh
regressions/cohort. Static implementation readiness does not authorize skipping
that work. Only after new Q/K DSP words and unchanged HVX V are proven should the
next mathematics investment be double NT QK. Host budget headroom remains narrow;
optimizing old HVX speed cannot substitute for the missing precision boundary.

| Question | Answer |
|---|---|
| 1. Single Mixed invocation implemented? | Yes, separate diagnostic package, two specializations; runtime unmeasured. |
| 2. Double Q/K on actual DSP? | NOT RUN. |
| 3. DSP Q/K vs same-operand Host words? | NOT RUN; Host body matches278,528 independent-reference words. |
| 4. Actual HVX V execution? | Static path and identical vector machine code verified; execution NOT RUN. |
| 5. DSP V vs ordered words? | NOT RUN. |
| 6. V mathematics changed? | No; existing source and helper bytes identical. |
| 7. Weights correctly distinguished? | Three distinct signatures/range checks and asymmetric fixtures; runtime NOT RUN. |
| 8. Existing51+4 ABI changed? | No; original source and binaries preserved. |
| 9. True S2→Mixed NATIVE connection? | Graph ownership contract PASS; actual execution unverified. |
| 10. Producer Y vs received Y words? | Receipt check implemented; NOT RUN. |
| 11. Saved state→old Backward maintained? | Same tensor IDs/contracts; new-bundle regression NOT RUN. |
| 12. Observer-free final accuracy? | NOT RUN. |
| 13. Lossless head split? | Existing copy reused and full-word checks implemented; new execution NOT RUN. |
| 14. Adjacent FP32 QK local accuracy? | Correct NT graph/reference implemented; NOT RUN. |
| 15. S2+C6 reproduced on DSP? | No new DSP certification; only Q/K arithmetic is newly implemented. |
| 16. Runtime/linkage/rounding problems? | No runtime attempt; static codegen/link succeeds, actual behavior UNRESOLVED. |
| 17. Same-binary HVX vs Mixed arithmetic cost? | UNMEASURED. |
| 18. Launch/stack/scratch/buffers? | Same one projection launch; new wrapper136/96, project32 bytes; new scratch0; logical buffer table above. |
| 19. Main19×8 cost? |39,845,888 double Q/K terms plus19,922,944 HVX V terms; scheduling/wall unmeasured. |
| 20. Ordinary battery/thermal stopped work? | No; monitoring policy tested, actual stop was Host ADB nonresponse. |
| 21. Critical gates retained? | Yes; identity/awake not proven, so no device execution. |
| 22. Three final device processes? | No. Three Host processes only. |
| 23. Fresh bundle HVX regressions? | NOT RUN; Host regressions PASS. |
| 24. Fresh W8 114/114? | NOT RUN. |
| 25. Old P contract FAIL preserved? | Yes, nine Host failed predicates reproduced without threshold changes. |
| 26. Ready for double QK? | Source design is bounded; first complete Mixed DSP/native certification. |
| 27. Remaining P→Backward work? | Mixed DSP certification, double QK, native Scale/Mask/Softmax/cumulative budget, saved P lifetime/integrity and observer-free backward consumers. |
| 28.40k→60k/200k? | **NO-GO**. |

Full Attention, full1-layer/19-layer and training correctness remain unverified.
Backward scatter, V/Gate gradient branches, projection gradients and full LN1dY
aggregation also remain.35/40-node Tap incident stays `REPRODUCED / CAUSE
UNRESOLVED`;6031 stays `UNRESOLVED / DORMANT / WATCH`. No training update,
Final test, LR/optimizer change, codec change or checkpoint decode was performed.
