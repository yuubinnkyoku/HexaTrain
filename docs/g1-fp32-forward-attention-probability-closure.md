# G1 FP32 Forward Attention Probability Closure

Research base: `6a4ac12f454b39737918c12c7c93c978127b900f`.
Branch: `investigation/g1-fp32-forward-probability-closure`.

**BLOCKED / UNRESOLVED. No new DSP numerical certification.** The read-only
initial device checks passed, but the mandatory immediately-before-pilot
awake check failed with `DEVICE_NOT_AWAKE_OR_UNKNOWN`. No probe executed,
no wake/foreground operation occurred, and device operations stopped.
A subsequent Host-only source-model assessment also failed the fixed joint
P contract. Neither issue was bypassed. There is no final three-process cohort.

## A. Source audit

The generalized training builder computes per-head rank2 `[32,32]` QKᵀ
(transpose0 false, transpose1 true), multiplies by STATIC `[1,1]`
`1.0f / std::sqrt(float(dimension / numHeads))`, bits `0x3e3504f3`,
then adds the shared STATIC `[32,32]` causal mask. The mask is `+0.0f`
(`0x00000000`) for column ≤ row, `-10000.0f` (`0xc61c4000`) otherwise.
It is finite addition, including future positions; no replacement by infinity.
Graph-owned `maskData` and `attentionScale` back the constants through graph life.

Pinned QAIRT MasterOpDef Softmax defines exponential normalization along the
specified axis and an optional positive beta. Production rank2 axis 1 maps to
rank4 `[1,1,32,32]` axis 3; beta remains FLOAT32 1. QNN Add broadcasts matching
trailing dimensions; the diagnostic expands only unit leading axes, with the
same 1024 row-major words. Saved P has the same shape as scores. Production P
is NATIVE except requested diagnostic outputs, and feeds Forward PV, Backward
`Pᵀ dContext` and Softmax backward. Its graph registry lifetime spans both uses.
No production builder is changed in this study.


The builder uses FP32 by default for these tensor declarations. A FLOAT32
declaration is not evidence of internal standard HTP arithmetic precision.
Backward dValue uses transpose0=true, transpose1=false; dProbabilities uses
dContext Vᵀ. Softmax backward is P(dP−Σ(dP·P)), per row. Head identities remain
H0/H1, with no inter-head reduction.

The Host preparation package is registered on the HTP backend alongside the
DSP implementation. This does not execute a CPU tensor backend fallback.
There was no new device execution from which to establish an actual fallback,
marker, receipt, API, finite-output, or teardown result.

## B. Forward Scale

The new diagnostic-only `ForwardScoreScale / ForwardScoreScaleNoReceipt`
pair takes H0 raw, H1 raw, and a rank4 scalar. It produces two independent
1024-word matrices; the observed specialization additionally emits 4105
receipt words: header 8, inputs 2049, outputs 2048.
Scale bits are source/Host `0x3e3504f3`. **Device received bits are unmeasured.**
The new marker is `0x48383118` on DSP, `0x43505518` on Host tests.
Its counter is separate from BackwardScoreScale and is used only for
synchronous diagnostic execution.

The arithmetic calls existing `elementProduct`, without changing the old
Multiply/Add body. No fast-math/FMA/tree reduction or alternative MatMul
framework was introduced. Output/input, output/output and receipt overlaps
are rejected; all new inputs and outputs must be finite. The legacy 45 Op
names/order and numerical sources are preserved; registry total is 47.
The production builder never invokes these new Ops.

Host kernel checks passed 40 calls (20 deterministic controls × observed/free):
FP32 multiply words, scale bits, both-head mapping, immutable inputs, complete
writes, poison/canary, invalid shape, overlap and NaN/Inf rejection. Existing
Scalar MatMul/Multiply, LN, Softmax backward, RawDQDK and HVX reference contract
tests passed in the inherited 17-test suite.

The executable diagnostic contains 44 source cases plus 12 explicit arithmetic
controls, independent standard Multiply negative, and four LN1→HVX QKV→Heads→
QK→Scale observer variants. Large arithmetic controls do not grant a 0.01
model-core accuracy PASS. These graph variants were built, but **none ran on
the device**. Native Scale, standalone DSP words, observer-free accuracy and
new standard negative reproduction therefore remain UNRESOLVED.

## C. Causal Mask

Source semantics are fully audited: 528 allowed positions/head, including
the diagonal, and 496 future positions/head. The input/output shape is
rank2 [32,32], expanded to [1,1,32,32] without changing any physical word.
There is no row/column transpose, clipping, upper-triangle zeroing or
replacement by -infinity. Mask storage is STATIC and graph-owned.

The mask diagnostic independently supplies correct FP32 scaled scores to
standard Add. It records full output words, first word mismatch and the
stage-budget decision. This test was **not executed** because the earlier
device safety gate stopped the pilot before any graph ran.
Consequently standard Mask Add is neither PASS nor FAIL.

Existing Custom Add already supports two matching 1024-word operands and
has an observed 4104-word receipt specialization. Conditional graph wiring
is available for a future measured standard failure; no new Add arithmetic
or mask specialization was implemented. This is not evidence that Custom
Mask is needed or that either mask path is accurate on the new package.

## D. Probability gate preregistration (before device P measurements)

The gate is a conjunction, not a tolerance inferred from device results:

* Element maxAbs ≤ `1e-6` against independent stable double Softmax.
* Double row-sum error ≤ `2e-6`; every probability finite and nonnegative.
* For source core scores, future probabilities must be zero (the finite mask
  underflows), row 0 diagonal probability one, rowwise maxima respect logits
  up to the element tolerance. Stress cases use the finite-mask analytic
  reference; finite -10000 is not a universal strict-zero guarantee for arbitrary
  logits. Every row has an allowed diagonal and a positive normalizer.
* Additional componentwise P-error sensitivity bounds below must allocate at
  most `0.00025` to dScores (one eighth of 0.002), and `0.001` to dValue
  (one tenth of the prior 0.01 local gate). Remaining budgets cover consumer
  arithmetic. Actual downstream gradients must separately pass 0.002 / 0.01.

For reference P, actual P+e, frozen dP, and dContext, let
`a = Σ_j dP_j P_j`, `b = Σ_j |dP_j e_j|` in each row. The exact bound on
the Softmax backward change at i is
`|e_i| |dP_i-a| + |P_i+e_i| b`.
The dValue bound at column c/output k is `Σ_r |e_rc dContext_rk|`.
These bounds are checked on observed P errors against the fixed 25-case
frozen gradient distribution. They control the actual distribution, not
arbitrary future gradients. An element-only uniform box bound is too loose:
Host audit measures D=1170.1279296875, L=13043.1827430725,
V=344.296861231327, giving a 0.01538345165 dScores bound at 1e-6.
We therefore explicitly reject treating the element gate alone as sufficient.
`probability_contract.h` fixes the joint contract; Host perturbation tests
check the inequalities independently of device outputs.

Mask local gate: allowed values must match independent FP32 Add words (addition
of +0, including IEEE signed-zero behavior); future values must match FP32 Add words
and have maxAbs ≤ 0.001 against double addition. The blocked threshold exceeds
half an ULP near -10000, while allowing no lossy rounding of unmasked logits.
Native cumulative mask error additionally inherits the 0.01 scaled-score
budget at allowed positions, and ≤0.011 at future positions; downstream P
must still independently satisfy the stricter joint probability contract.
Arithmetic stress is separated from the 44 source-derived accuracy cases.


### Host assessment of the registered contract

The gate/header and mathematical budget were committed in
`198f3504deb9e779fd6bb030fd72e0635dc57d68`, before any device Softmax
measurement (there still are none). No gate was relaxed after Host results.

The componentwise bound is valid for actual probability error; the initial
small-perturbation test passed. The stronger subsequent assessment evaluates
44 source-model LN1/QKV cases × two heads against all 25 fixed frozen gradient
cases. This Cartesian coverage deliberately does not claim that all such
P/dP/dContext combinations occur in a real training trajectory.

| Host-only ordered Forward model | Maximum | Gate / allocated budget |
| --- | ---: | ---: |
| P element error from external LN1 X | 1.00750624248969e-6 | 1e-6 — FAIL |
| Double row-sum error | 2.53864868682285e-7 | 2e-6 — PASS |
| P-induced dScores componentwise bound | 4.64227603470169e-4 | 2.5e-4 — FAIL |
| P-induced dValue componentwise bound | 6.04851550175399e-5 | 1e-3 — PASS |
| Actual Host FP32 dScores error | 2.8771553135698e-4 | .002 — PASS |
| Actual Host FP32 dValue error | 4.50467220550199e-5 | .01 — PASS |

The existing Host Softmax oracle evaluates the shifted exponential in double,
rounds each exponential to FP32, and uses increasing-column FP32 sum and
FP32 division. It is a Host libm control, not a promised DSP exp word oracle.

The joint assessment returned exit 4 with 9 failed predicates across its
case/head/gradient combinations. This is a numerical failure, not an SDK,
transport, or QNN API failure. The broad Host profile and the three new
kernel/shape/bound-contract tests passing do not override it.

This result separates an element gate, a conservative propagation budget and
the actual downstream error. A propagation bound exceeding its allocation
does not mean dScores exceeded .002. Conversely, the downstream Host PASS
does not waive the joint P contract. The assessment includes accumulated
LN1/QKV/QK/Scale errors and Host libm; it does not isolate standard HTP exp
or predict the DSP candidate output.

Standard HTP Softmax was not measured. Therefore the condition to implement
Custom Forward Softmax was never met. No new exp approximation, Forward
Softmax arithmetic or HVX Softmax was added. The pinned SDK QHPI example
uses shifted `expf`, increasing-depth sum and normalization, which establishes
an API example, not an exp accuracy proof. DSP exp error envelope, subnormal
behavior, monotonicity, rounding and runtime linkage remain unverified.

## E. Forward P → Backward

Twenty Host graph constructions check the five boundaries Scale/Mask/P/
P→Softmax backward/P→dValue under four observer settings. The graph mock
validates 20 new Scale nodes and 24 standard Softmax nodes, including axis 3,
beta 1, and no INT receipt tensors in NoReceipt graphs.
For NoTap backward graphs P is NATIVE and the same tensor ID is passed to
the consumer. This establishes shape/ownership construction only.

There was no device P producer, saved-state lifetime execution, observed
producer/consumer word comparison, native P final dScores/dValue measurement,
or reference-P device control. P-final APP_READ and saved-P NATIVE graphs
must be evaluated separately when safely resumed. No Host extraction/
reinjection is presented as native connection.

Reference-fed P dependence has **not** been reduced. Correct frozen dP and
dContext remain the intended external controls for minimal P-consumer tests.
A failed correct-P standard dValue control must be recorded as a consumer
frontier; this study must not add Custom dValue to overcome it.

## F. Performance

No new device arithmetic, receipt window, first execute, steady median,
observer-free wall, finalize latency, physical allocation or thermal-correlated
performance result exists. No prior binary timing is copied into this study.
Graph description instrumentation records node/tensor/logical native memory,
Host buffer lower bounds and explicit scratch when the probe can execute.
The added Scale uses no explicit scratch; no performance improvement is claimed.

Source-derived work per layer/microbatch is 2048 Scale multiplies and 2048
Mask additions. A bounded rowwise Softmax would process 64 rows and 2048
elements; there is no implemented Custom Softmax launch to time.
Saved P logical payload is 2×32×32×4 = **8192 bytes**, 155648 bytes over
19 layers, 1245184 bytes over 19 layers×8 microbatches. These are logical
sums, not simultaneous physical peak allocations or throughput predictions.

Saving P avoids its recomputation but extends backward lifetime. Recomputation
would need correct LN1/QKV/QK/Scale/Mask/Softmax dependencies and consistent
external parameters. Neither scheduling nor recomputation/checkpoint policy
was changed. A future fusion can reduce launches/receipts only after independent
precision and ownership contracts hold; no fusion or HVX optimization was added.

## G. Architecture / precision coverage

All previous verified entries below refer to their original device-tested
binary and fixed external boundaries. **The new package has no device regression
PASS.** Local arithmetic, native connection, observer-free accuracy, actual
producer-fed input and full training are distinct evidence levels.

| Region | Existing evidence / remaining boundary in this study |
| --- | --- |
| LN1 F/B, HVX QKV, lossless Heads, QK raw | Previous local/native/observer-free fragment PASS; external LN1 X/parameters; new device regression blocked |
| Forward Scale | New Host arithmetic/shape PASS; DSP/native/observer-free UNRESOLVED |
| Mask | Source/Host semantics fixed; standard device unknown; Custom reuse only conditional |
| Softmax P | Host joint gate FAIL; standard device unknown; no Custom Forward Softmax |
| PV / Gate projection→Sigmoid / gated Context / Wo | Producer accuracy and connected Forward closure remain unverified |
| Residual / LN2 / FFN1→ReLU / FFN2 | Full upstream producer chain unverified; previous bounded LN/FFN2 fragments do not close it |
| LN2→dContext/dWo→Gate→dProbabilities | Previous reference-fed connected fragment PASS; saved Forward states external |
| Softmax backward→raw dQ/dK→Scale | Previous local/native/observer-free PASS; new device package not retested |
| dQ/dK Head Scatter | Previous standard .01 failure remains a separate unresolved frontier |
| dValue | Previous local standard .01 PASS on older fixtures; new P consumer unmeasured |
| Gate gradient branch / projection gradients | Producer-fed integration unverified |
| LN1 dY aggregation / LN1 backward | Q/K/V/Gate branches and source-order sums must all be correct; dQ/dK alone are insufficient |
| FFN backward / gradient registry-materialization / 8-MB accumulation | Full connected correctness remains unverified |

Q/K heads are used by dKraw/dQraw; V heads feed dProbabilities; P feeds
PV, Softmax backward and dValue. Gate and ungated saved context feed the
gate-gradient branch. Raw/scaled/masked scores can be recomputed from Q/K;
the source implementation still retains graph tensors. Q/K/V/Gate/context/P
ownership and lifetime must survive their actual consumers.

Full LN1 dY additionally requires dV scatter/aggregation, dWv, gate reduce/
scatter/Sigmoid derivative/dWg, dWq/dWk, the Q/K/V/Gate projection-input
gradients and their source-order Add chain. Full one-layer also needs the
FFN/residual branches, saved-state accuracy, gradient materialization and
eight-microbatch accumulation. None is certified by graph shape tests.

**Next priority: continue Forward Scale→Mask→Softmax P precision closure.**
First restore an awake, safely idle device; execute the Scale gate before
standard Mask and Softmax. The Host joint-budget failure must remain visible
while distinguishing local Softmax error from accumulated upstream error.
This has greater correctness value than further QK HVX speed work: saved P
still depends entirely on reference inputs. Any later budget revision requires
a new justified preregistration and cannot silently convert this FAIL to PASS.

The 35/40-node Tap incident remains **REPRODUCED / CAUSE UNRESOLVED**.
No new observer-free execution exists to change that status.
6031 remains **UNRESOLVED / DORMANT / WATCH**.

## H. Final decisions

| # | Question / answer |
| --- | --- |
| 1 | Forward Scale reuse PASS? Host multiply/word contract PASS; DSP standalone/native not run, UNRESOLVED. |
| 2 | Scale bits? Source/Host 0x3e3504f3; device received scalar not measured. |
| 3 | Exact mask? Add +0 at c≤r and finite -10000 at c>r, FP32 STATIC row-major matrix. |
| 4 | Standard Mask PASS/FAIL? UNMEASURED. |
| 5 | Minimal accurate Mask implementation? Standard Add if it passes; otherwise existing Custom Add; choice not established. |
| 6 | Standard Softmax P gate? UNMEASURED; Host cumulative model joint gate FAIL. |
| 7 | Custom Softmax needed? Unknown; the required measured standard-failure condition was not established. |
| 8 | DSP exp/normalization accurate? No new DSP math implementation or measurements. |
| 9 | P gate basis? Fixed element/row/invariant checks plus explicit componentwise gradient sensitivity budgets from frozen dP/dContext; Host broad test exposed 9 failed predicates. |
| 10 | LN1→P native PASS? No, graph shape only. |
| 11 | Observer-free P accuracy? UNRESOLVED. |
| 12 | Actual Forward P→Backward NATIVE? Constructed same-ID dependency only; no executed connection. |
| 13 | P→Softmax backward? No device result; Host actual dScores within .002 but joint P budget failed. |
| 14 | P→dValue? No new device result; Host actual dValue within .01. |
| 15 | Reduce reference-fed P? No. |
| 16 | Remaining external operands? LN1 X/gamma/beta/QKV weights; backward dP/dContext and all other saved-state/producer dependencies remain frozen/external in intended minimal controls. |
| 17 | Next Forward precision frontier? Scale is still the proven standard failure from prior binary; new Scale, then Mask/P require device proof. Beyond P: PV/Gate/Wo and residual producers. |
| 18 | Backward Head Scatter unresolved? Yes; not modified or reclassified. |
| 19 | HVX/W8 regression maintained? Legacy Host contracts pass; new-package device HVX and W8 114/114 BLOCKED, never substituted by historical results. |
| 20 | Highest-value next study? Forward P precision closure with safe device availability and explicit cumulative-error budget evidence. |
| 21 | Full one-layer missing? Producer-fed Forward states, Attention/FFN branches, scatter/projection/gate gradients, LN1 dY sums, saved lifetimes and complete materialization/accumulation. |
| 22 | 40k→60k / 200k? NO-GO; zero training updates, no final-test/LR/optimizer/backend changes. |

## I. Provenance / validation / reproduction

Initial HEAD was the complete research base SHA above, tree clean, no upstream
configured. There were 102 commits absent from known remote refs. Previous local
research refs and all five protected artifact hashes are preserved.
The 40k checkpoint remains
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
No checkpoint decoding or training was performed.

Initial compiled pilot source:
`198f3504deb9e779fd6bb030fd72e0635dc57d68`.
The pilot stopped at the awake gate before staging or probe execution.
A final Host-only candidate rebuild includes the producer/final word-audit
correction. Its source is `ac5be47c155725a0a442da540ab0359b34602a88`.
DSP SHA256: `45aebb0312f7d7f39cf01c61e9456cc3e87c70b73854db68c755ea2df6428d4c`.
Host package: `480014e5f0259a04b8065c2e911930a0f19b41568a5fc5b0182adb9499c2b210`.
Probe: `06fce37163459854f2c6c5f7e6a31b91a64b098a304a5dbf57afbb7019d0abea`.
These identify an untested device candidate; they do not imply DSP execution.
There is **no new device-tested source commit**, no numerical digest and zero
independent final device processes. Old device source b6794c4 and old binary
hashes are comparison identities only.

Pinned identities: QAIRT 2.48.40.260702151143; QNN API 2.37.0; HTP V81;
Hexagon SDK 6.6.0.0 / tools 19.0.07; NDK 26.2.11394342.
Compiler hashes and DSP/Host/probe hashes are in the aggregate.
MasterOpDef audit SHA256:
`9036c9c99df2554de7ddbd8afb86351c9070f2212f1dc99e0c9fe26d49039fe1`.

Frozen identities:

- one-layer inputs: `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`.
- LN1 parameters: `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`.
- source-derived LN1 producers: `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`.

Completed broad profiles: Fast 80.4s PASS; Host 424.1s PASS; Qnn 482.4s
PASS, including QNN APK build and SDK/APK audit. The optional SDK inventory
advisory was retained; core and pinned Build ID passed.

Validation classification is fail-closed: Fast/Host profile/Qnn and SDK/APK
audit results are checked from completed logs by the allow-list exporter.
The inherited 17 Host tests plus the three new kernel/graph/bound tests passed.
The added 44×2×25 probability numerical assessment **FAILS** its joint contract;
the full new targeted runner therefore exits nonzero. Device smoke, actual
new-package HVX FFN2/QKV, QK/BackwardScale/Softmax-backward regression and Muon
W8 114/114 are BLOCKED. Missing required gates prevent comprehensive PASS.

Reproduction: run `scripts/test_forward_probability_host.ps1` with the pinned
QAIRT root/Build ID (Host only; the probability assessment intentionally returns
a nonzero code while the fixed contract is unmet). `scripts/verify.ps1 -Profile
Host` checks the broad repository suite. For safe device resumption use
`scripts/run_fp32_custom_probe.ps1 -Variant oneProbabilityScale` with pinned
QAIRT/NDK arguments, only after an awake device satisfies all original guards.
Do not bypass them, wake the device automatically or advance past a numerical
failure.

Private records remain under
`build/reports/g1-fp32-forward-attention-probability-closure/`.
The [allow-list aggregate](results/g1-fp32-forward-attention-probability-closure-summary.json)
commits only public identities and classifications. Its exporter rejected ten
self-test corruptions. The Host result source/executable/log identities,
compiler, frozen inputs, five protected hashes, old Op prefix and unchanged
legacy numerical-source identities are validated before export.
Final repository HEAD and clean-tree/protected-ref checks are recorded in the
private final closeout and final response, after the closing report commit;
an immutable commit cannot embed its own final hash. No push/PR/merge or
history rewriting occurred.
