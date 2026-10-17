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

## B–I. Execution status

Implementation and Host preparation/32-graph ownership tests are available.
Actual package codegen, runtime, native/observer-free results, same-binary costs,
fresh regressions and three-process evidence are pending. No device numerical
classification is issued from compilation or previous binary results.

Research-specific battery/thermal monitoring leaves default project policy
unchanged. Normal battery decline, absent external power and noncritical thermal
throttling are monitored; critical OS state, serious battery fault, contradictions,
unknown protection status, identity/awake/run conflict and protected-state gates
remain blocking. No wake/unlock/force-stop/foreground takeover or APK replacement
without current authorization. The already approved installed APK is retained
with a private hash snapshot; newly built Qnn profile APK is audited separately.

The adjacent QK arithmetic remains ordered FP32. This experiment does not
implement double NT QK, Scale/Mask/Softmax changes or native P→Backward. Old P
contract failures, 35/40-node incident (`REPRODUCED / CAUSE UNRESOLVED`) and6031
(`UNRESOLVED / DORMANT / WATCH`) remain. Full Attention/1-layer/19-layer/training
correctness is unverified. 40k→60k/200k remains **NO-GO**.
