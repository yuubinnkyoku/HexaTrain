# DSP FP32 Custom MatMul and LN2 backward segment

2026-10-13. Diagnostic-only research; training updates remain zero.

## Preregistration (before device numerical results)

Baseline HEAD is `277202f1b78fee36e376c5f59c499cbdf63c5469`, matching the
reported research HEAD. The starting tree was clean, with no upstream configured.
All existing local commits and experiment refs are preserved. Work branch:
`investigation/htp-fp32-custom-matmul-ln2`. Prior numerical source remains
`5e15d36d9c3c6db005e8cb2ac2be360e7569b4de`; its frozen binaries and evidence
are not replaced by this cohort.

The existing HexaFp32 QHPI registration, Copy/Add, LN Forward/Backward, flat4
FLOAT32/direct/DDR_ONLY tensors, QNN HTP runtime, guarded runner, and independent
LN references are reused. Production builders, Muon W8/IDL, optimizer, checkpoint
format, LR, clipping, and dataset/order are unchanged.

Two fixed diagnostic op names encode M/N/K/transpose; no generic dtype or shape
support is claimed. Physical inputs remain row-major; transpose changes indexing.

| Op | Physical inputs | M/N/K | Transpose A/B | Output |
| --- | --- | --- | --- | --- |
| MatMulDContext | `[32,64]`, `[64,64]` | 32/64/64 | false/true | `[32,64]` |
| MatMulDWo | `[32,64]`, `[32,64]` | 64/64/32 | true/false | `[64,64]` |

The kernel uses scalar FP32 product followed by sequential FP32 accumulation,
q=0..K-1, with a volatile product, `-ffp-contract=off`, `-O2`, no fast-math,
and disabled DSP vectorization. There is one candidate. Kahan/pairwise/HVX are
not explored in response to results. All inputs and final results must be finite;
nonfinite/overflow fails execution. This is bounded fixture coverage, not a proof
for every possible FP32 input. No CPU runtime fallback exists.

Before kernel implementation, the serialized frozen operands/LN inputs had SHA256
`daf61450486f6f8df40b7b10e631c268e9f5e2e5f2458e3a1eda4b0cb8c9978e`.
Each path retains the exact previous 20 D64 recipes × random/selector/cancellation
controls (60 cases), Host FP32 dX+residual, full upstream scale1, and seed97 operands.
An independent additional matrix has eight recipes, 32 distinct activation rows,
varying gradient magnitude/sign, low variance, large centered values, sparse and
repeated column patterns, and cancellation. Added fixtures are counted separately.
The standalone injected-gradient experiment does not establish DSP LN2 chaining.

Frozen gates: MatMul maxAbs versus double <=0.03, LN gradients <=0.02, forward
Y/xhat ordinary <=0.002 or registered low-variance <=0.06, inverse <=0.002.
Nonfinite, QNN error, missing execution marker, receipt mismatch, mutation, poison,
canary damage, stale counter, or build identity mismatch fails closed.
Record RMS, relative L2, maximum relative error (denominator floor0.001), FP32
word/ULP differences, input/output digests, and independent process identities.

Phase B: direct Custom MatMul, both paths, three independent processes, same exact
package/probe/fixture/runtime identity. Only after both paths pass proceed to C:
direct/native/tap Custom and direct/native standard controls in the same process
runtime. The standard control may retain its known numerical failures, but all
API/finite/ownership gates remain required. Compare direct/native/tap final words.
MatMul receipt contains every operand word actually read and every output word;
Copy receipt alone is insufficient. Three processes are required for C.

Only B/C success permits constructing D: real DSP LN Forward saved state →
DSP LN Backward → Custom Add → both Custom MatMuls. X/gamma/beta/dY/residual,
Wo/context are external inputs; Host dX/DRESIDUAL1 injection is forbidden.
Compare final outputs to independent double calculations starting at X.
Native/tap and optional fused LN backward+residual are distinct graph variants.
Retain producer/consumer full-word receipts at every native boundary.

Performance is measured only for accuracy-passing candidates, BACKGROUND_CORRECTNESS.
Separate arithmetic timer from full-word receipt/finite scans and graph wall;
setup/finalization/allocation and Host validation remain separate. No performance
ceiling is preregistered, so no BLOCKED_PERFORMANCE judgment is possible.
Do not multiply micrograph wall by layer or batch counts to predict throughput.

Pinned QAIRT Build ID `2.48.40.260702151143`, QNN API2.37.0, HTP V81,
Hexagon SDK6.6.0.0/tools19.0.07, NDK26.2.11394342. Source/compiler/package/probe,
SDK/Skel and fixture hashes must be tied to each cohort. Prior additional Copy
control remains DEVICE NOT RUN until independently executed. Binary hash changes
never inherit old accuracy results.

Host20GiB/Device12GiB, stable physical identity, Awake/top-resumed, active-run,
APK audit and 2-second foreground monitor gates remain required. Protected40k
SHA256 is `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
UNSTABLE resume guard, all protected evidence/cache/tokenizer/order, and old
experiment refs remain intact. 6031 remains UNRESOLVED / DORMANT / WATCH.

Prior DSP LN Forward/Backward/Residual verification is retained independently.
Standard HTP MatMul remains BLOCKED_MATMUL_PRECISION / MATMUL_PRECISION_STAGE_UNRESOLVED.
No production graph replacement, all-layer attention conversion, training update,
40k→60k/200k, final test, SDK/firmware update, push, PR, merge, or history rewrite.

## Results

Pending independent device validation. Host arithmetic is not Device verification.

Initial source `6ad33db` had three numerically passing direct-device pilots.
Sixteen of 136 generated operand sets differed in bits from the PC-frozen file:
the ARM LN oracle's math-library results were not assumed bitwise identical to PC.
These pilots are preserved separately and do not satisfy frozen-input identity.
The standalone probe now reads the exact preregistered binary, with SHA256 checked
locally and after device staging. This changes input transport, not the kernel,
gates, reduction order, or LayerNorm algorithm. The final cohort must rerun B/C.
The chain will use only external X/gamma/beta/dY/residual/context/Wo from that file;
it must never use the file's injected intermediate gradient.
