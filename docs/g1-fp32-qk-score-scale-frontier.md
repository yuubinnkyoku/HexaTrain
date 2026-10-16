# G1 FP32 QK raw score and backward score scale frontier

Status: **UNRESOLVED / validation in progress**. Diagnostic only, default off.
Production builders, optimizer, checkpoint codec and training updates are unchanged.
40k→60k / 200k remain **NO-GO**.

## A. Source & reuse audit

Research base `8e13b1ed1e2464b4aef790233682a11c43c838c5`, preserving the
unpublished local history of `investigation/ordered-fp32-hvx-qkv`.
Branch: `investigation/g1-fp32-qk-score-scale-frontier`.

The authority is `qnn_runtime_transformer_training_generalized.inc`, with shared
LN equations in `qnn_runtime_transformer_training.inc`.
T32/D64/H2 has per-head Q/K/V `[32,32]`, row stride 32 FP32 words.
Diagnostic shapes backfill two unit axes, retaining the row-major physical layout.
QK computes `S[r,j]=sum(q=0..31,Q[r,q]*K[j,q])`, MatMul transpose0=false,
transpose1=true. It contains no scale, mask, Softmax, clipping or Gate.

Backward uses `dQraw=dScores*K` (NN), `dKraw=dScores^T*Q` (TN), then two
ElementWiseMultiply nodes per head. Source constructs the shared STATIC `[1,1]`
scalar as `1.0f / std::sqrt(float(dimension / numHeads))`; the diagnostic backfills
it to `[1,1,1,1]`. This is a single rounded FP32 product per gradient element.
The source scalar object remains alive in the graph through execution.
Forward scaling acts on raw scores; backward scaling acts on raw gradients.
The common scalar does not make them the same tensor operation.

`ForwardQKRaw` specializes existing `attentioncustom::dprob` NT geometry with
independent marker `0x48383116` and counter. The default old marker remains
`0x4838310b`. Both call the unchanged `mmcustom::ordered` increasing-q dot,
separate rounded multiply/add, contraction off, no tree reduction.
`BackwardScoreScale` groups H0 dQ/dK, H1 dQ/dK and one scalar; it calls existing
`elementProduct`, shared with Scalar Multiply/SoftmaxProduct, without new math.
Its marker is `0x48383117`. Four new names append to the old 41-name registry;
old order/signatures/receipts and QHPI1.4 remain unchanged. Each has NoReceipt.
No same-kind concurrent diagnostic graph execution is supported by the counters.

External inputs and parameters are APP_WRITE, final outputs APP_READ, internal
edges NATIVE except Float Tap variants. The production builder is not connected
to either new Op. Existing HVX source/flags are unchanged.

### Gates registered before device evaluation

Existing raw QK .01, QKV .002, lossless head words, LN/saved state .002,
local LN backward .02, dContext/dWo .03, Softmax dScores .002, raw/scaled dQ/dK
.01 are retained. Forward standard Scale is assigned maxAbs **.01** from the
existing raw-score budget and the unchanged causal-mask input contract.
Backward standard Scatter inherits maxAbs **.01** from scaled dQ/dK: one-hot
placement has norm1 and performs no mathematical amplification. These gates
are fixed before collecting candidate device results. New large controls test
arithmetic identity separately from the 44/25-case stage accuracy certification.

## B. Forward QKᵀ

Pending standalone, native LN1→HVX QKV→Heads→QK, observed/free, negative
standard QK and next standard Scale measurements. No device PASS is claimed.

## C. Backward score scale

Scale expected FP32 word: `0x3e3504f3`, independently asserted on Host and
checked from the actual scale consumer receipt on device. Pending standalone,
native fused/separate Softmax, observed/free and next standard Scatter results.

## D. Coverage / saved state

Pending final source-derived coverage assessment. Local arithmetic,
native-connected, observer-free, full producer-fed and full training correctness
are separate claims. Correct frozen Forward P/Q/K/V/Gate remain external inputs
to the backward diagnostic. QK success alone does not establish Forward P.

## E. Performance

Measured only after required accuracy/word checks pass. Forward QK has 65,536
MAC/layer/MB, 1,245,184 MAC/19 layers, 9,961,472 MAC/19 layers×8 MB.
Backward Scale has 4,096 multiplies/layer/MB, 77,824/19 layers,
622,592/19 layers×8 MB. These are logical work, not throughput predictions.
No HVX QK implementation or Scalar/HVX QK speedup is claimed.

## F. Observer effect

The existing 35-node/40-node Float Tap incident remains
**REPRODUCED / CAUSE UNRESOLVED**. No large Tap bisection is performed.
NoReceipt hidden values are not directly observed; their own final outputs
must satisfy analytic gates and cross-variant word comparisons independently.

## G. Architecture decision

Pending candidate accuracy and next-consumer measurements. Precision closure
has priority over further speed work while required arithmetic is inaccurate.

## H. Final questions

Pending validated cohort. Full Attention / full1-layer / full19-layer are
UNVERIFIED; no local result upgrades them. Training remains NO-GO.

## I. Provenance

Previous DSP identity `9736778fc192fd71c7ab974341294ec4f4cd05618473c699848318e9839c9b11`
and probe `9e28d2b960f6dbc9c2dc4f4c17b58de4f4024c9c603967682c504836c1af143d`
are preserved privately for comparison, not new-candidate execution evidence.
Private records belong under `build/reports/g1-fp32-qk-score-scale-frontier/`.
Only allow-listed aggregates will be committed. QAIRT2.48.40.260702151143,
API2.37.0, V81, Hexagon SDK6.6.0.0/tools19.0.07, NDK26.2.11394342 remain pinned.
6031 remains **UNRESOLVED / DORMANT / WATCH**.
