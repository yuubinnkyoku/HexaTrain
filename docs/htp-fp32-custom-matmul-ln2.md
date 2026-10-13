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

Final numerical cohort: **CUSTOM_DCONTEXT_VERIFIED / CUSTOM_DWO_VERIFIED /
CUSTOM_MATMUL_NATIVE_VERIFIED / DSP_LN2_MATMUL_SEGMENT_VERIFIED**.
**LN2_INTEGRATION_CANDIDATE**, bounded to a future one-layer diagnostic.
Production-ready and full-model training stability are not established.

Initial source `6ad33db` had three numerically passing direct-device pilots.
Sixteen of 136 generated operand sets differed in bits from the PC-frozen file:
the ARM LN oracle's math-library results were not assumed bitwise identical to PC.
These pilots are preserved separately and do not satisfy frozen-input identity.
The standalone probe now reads the exact preregistered binary, with SHA256 checked
locally and after device staging. This changes input transport, not the kernel,
gates, reduction order, or LayerNorm algorithm. The final cohort must rerun B/C.
The chain will use only external X/gamma/beta/dY/residual/context/Wo from that file;
it must never use the file's injected intermediate gradient.

## A. Baseline and reuse

The latest generalized builder (`qnn_runtime_transformer_training_generalized.inc`)
uses `DRESIDUAL1_LN` from LN2 backward, adds the FFN residual upstream to produce
`DRESIDUAL1`, then branches to `AttentionContextᵀ × DRESIDUAL1` and
`DRESIDUAL1 × Woᵀ`. Token row order is T32, feature order D64, contiguous row-major.
The diagnostic matches these identities, shapes and transpose directions.
Residual and dY are independent external inputs in the additional fixtures.

Prior DSP LN verification is retained. Prior standard HTP dWo/dContext failures
were 19/60 and 12/60, with maximum errors 0.983984801079 and 0.0623321533203.
The old five-path 166/600 failures and unresolved internal precision stage remain
historical evidence. No standard HTP precision configuration was re-explored.

## B. FP32 MatMul design and ABI

The two QHPI kernels each perform 131,072 MAC, row→column→increasing-K order.
For dContext, `A[i*64+k] * B[j*64+k]`; for dWo,
`A[k*64+i] * B[k*64+j]`. These index formulas implement mathematical transpose
without physically transposing the inputs. Explicit FLOAT32 rank4 shape checks
reject wrong geometry and FLOAT16 inputs/outputs. QHPI resources retain the
existing package's HVX worker contract, while arithmetic instructions are scalar.

`-std=c++17 -mv81 -O2 -G0 -fPIC -shared -ffp-contract=off`,
`-fno-vectorize -fno-slp-vectorize`, and `PREPARE_DISABLED` are retained on DSP.
NDK uses `-O2 -ffp-contract=off`. No fast-math, CPU arithmetic fallback, compensation,
clamp, or gradient shrink is introduced. The emitted dot loops contain scalar
`memw`, `sfmpy`, a volatile product store/load, and `sfadd`; there is no contracted
FMA or horizontal/vector reduction in these dot loops. This observation does not
describe the internals of standard HTP MatMul or LN math-library routines.

The integer MatMul receipt contains 16 metadata words, every physical A/B operand
word, and every output word. The consumer validates actual FLOAT32 loads, geometry,
execution marker/counter, finite results, and complete stores. Inputs are owned by
the Host until synchronous execute completes. Native tensors belong to the QNN
context; saved state is not retained across context free. APP_READ outputs use
NaN poison and canaries; every APP_WRITE vector is checked for mutation.

Observed Add and fused LN backward can optionally append their output words to
their existing receipt. Original receipt sizes remain accepted; LN formulas and
existing production builders are unchanged. Host tests cover original and expanded
receipt formats, geometry/dtype/nonfinite/overflow rejection, 136 exact FP32 kernel
comparisons, and 68 full scalar LN/residual reference checks.

## C. dContext accuracy

All numbers below are maxima in one cohort; all three independent processes have
identical typed numerical digests. Gates are applied to each case, without averaging.

| Experiment | Required 60 maxAbs double | Additional 8 maxAbs double | Failures |
| --- | ---: | ---: | ---: |
| Direct Custom, frozen injected gradient | 2.84283660932e-5 | 1.73684536094e-4 | 0/68 |
| Copy→native→Custom | 2.84283660932e-5 | 1.73684536094e-4 | 0/68 |
| Tap-enabled Custom | 2.84283660932e-5 | 1.73684536094e-4 | 0/68 |
| Actual DSP LN2→residual→Custom | 2.96326040541e-5 | 1.84878785603e-4 | 0/68 |
| Standard HTP control | 0.0623321533203 | 0.274147796668 | 12/60 + 1/8 |

Standalone results and each MatMul's local FP32 oracle are bitwise identical,
maximum ULP difference0. RMS, relative L2, maximum relative error, input/output
digests, nonfinite count and per-case gates are preserved privately. Standard
direct/native controls are bitwise equal; their failures do not cancel Custom PASS.
The final chain row is compared to a double result starting at X, not injected dX.

## D. dWo accuracy

| Experiment | Required 60 maxAbs double | Additional 8 maxAbs double | Failures |
| --- | ---: | ---: | ---: |
| Direct Custom, frozen injected gradient | 2.83033344658e-4 | 6.12229439412e-4 | 0/68 |
| Copy→native→Custom | 2.83033344658e-4 | 6.12229439412e-4 | 0/68 |
| Tap-enabled Custom | 2.83033344658e-4 | 6.12229439412e-4 | 0/68 |
| Actual DSP LN2→residual→Custom | 4.30033019711e-4 | 5.87338508012e-4 | 0/68 |
| Standard HTP control | 0.984083266045 | 1.45542014639 | 19/60 + 2/8 |

Custom local FP32 dot results have zero word/ULP difference. The slightly different
standard-control maximum versus the old Lane B report reflects the PC-frozen
operand cohort; the old recorded value is not rewritten. Both new standard variants
have 34/136 failures per process across the two paths; together 68/272 controls fail.
The original 60-case matrix and added 8-case matrix remain separately counted.

## E. Native FLOAT32 preservation

Copy input receipt, optional floating-point tap, and MatMul consumer receipt match
every gradient word. Both operands and the APP_READ result are FLOAT32 contracts,
independently checked from the native intermediate contract. Direct/native/tap
Custom final outputs match bitwise for all cases in each of three processes.

The final LN2 graph additionally compares Forward's saved xhat/inverse words to
Backward's input receipt, Backward dX to Add input, and Add output to both MatMul
input receipts. Fused backward/residual also captures its output for both consumers.
No tested boundary in this Custom segment loses observable input/output words.
Receipts are present in every graph; receipt-free production lowering is unverified.

The separate T32 Copy→native→Copy control passed 20 recipes in three fresh processes
under the final binary, with zero bit mismatches. The former source163a594 control
was never executed under its old binary and remains DEVICE NOT RUN historically.

## F. Actual DSP LN2 full segment

Only after the same exact final binary passed B/C was the guarded chain executed.
Nine outputs are observable, either as FLOAT32 outputs or producer integer words:
Y, xhat, inverse, dX, dGamma, dBeta, DRESIDUAL1, dContext and dWo.
Native and tap graphs have five Custom nodes; fused backward/residual has four.
The only external numerical inputs are X/gamma/beta/epsilon/dY/residual/Wo/context.
The serialized standalone gradient is not registered as a graph input in this mode.

| LN output | Required matrix maxAbs double | Additional matrix maxAbs double |
| --- | ---: | ---: |
| Y | 5.81823561774e-7 | 8.27691723160e-7 |
| xhat | 3.92462837340e-7 | 6.78915995422e-7 |
| inverse | 1.94379752543e-5 | 2.44125095037e-5 |
| dX | 1.53640447422e-5 | 1.03089135223e-4 |
| dGamma | 1.23042775968e-6 | 1.22078990561e-6 |
| dBeta | 0 | 5.66709786654e-7 |
| DRESIDUAL1 | 1.53640447422e-5 | 1.17751877838e-4 |

Native/tap/fused each pass all 68 cases; each variant has three independent-process
confirmations. Their final dContext/dWo words agree bitwise. Saved state comes from
the same graph's actual DSP Forward. The final MatMul gates use an independent
double analytic LN calculation and double dot products starting at the external X.
No hidden Host intermediate or training checkpoint is injected.

The final nine main processes executed 3,357 QNN graph calls. API success, finite
outputs, DSP execution receipts, no input mutation, full poison overwrite and
canary integrity are checked separately. Known standard-control numerical failures
remain present despite API/finite success. The additional Copy processes are counted
separately from the nine-process numerical cohort.

## G. Preliminary performance and implementation cost

Accuracy-passing candidates only; BACKGROUND_CORRECTNESS, 9 additional timed calls,
8 steady samples per graph/process. Values below are ranges of process medians.
All final main processes recorded thermal status0 and battery temperature39°C,
with wired power present. DSP frequency/worker scheduling and device power in watts
were not controlled or measured. These are diagnostic latency observations.

| Candidate | DSP MatMul arithmetic window | Graph execute wall |
| --- | ---: | ---: |
| Single dContext (mmDirect) | 3.905–3.936 ms | 6.831–7.132 ms |
| Single dWo (mmDirect) | 4.087–4.121 ms | 7.133–7.255 ms |
| Native-connected dContext | 3.8975–3.899 ms | 7.195–7.514 ms |
| Native-connected dWo | 4.090 ms | 7.325–7.557 ms |
| LN2 native segment | dContext4.765–4.7695 / dWo4.7255–4.728 ms | 9.674–9.757 ms |
| LN2 tap segment | dContext4.162–4.766 / dWo4.408–4.725 ms | 9.474–9.731 ms |
| LN2 fused segment | dContext4.7675–4.768 / dWo4.724–4.7265 ms | 9.134–9.181 ms |

A boundary-process direct dContext median was 0.745ms arithmetic/1.355ms wall,
unlike the other repetitions. Its first execute was 1.394s; other direct first
executes were about8–9ms. This outlier is retained, not selected as representative.
LN2 first executes were11.37–12.94ms. Background scheduling/frequency/transient
causes are unresolved; no throughput promotion follows these measurements.

The timer reads `c31:30` and converts the 19.2MHz qtimer to integer microseconds.
It measures elapsed entry windows, not exclusive DSP cycles. Parallel branches or
preemption can affect windows; kernel-window sums are not graph wall decompositions.
In the full segment, summed dContext+dWo windows can exceed graph wall. Concurrency
versus instrumentation/scheduling effects have not been resolved by a QNN timeline.

Standalone full-word receipt/finite scanning costs about172–177.5µs per MatMul,
recorded outside the arithmetic window. LN/Add old timers include diagnostic work;
their native segment windows are approximately0.866–0.869ms combined LN and
0.752–0.753ms Add, not receipt-free arithmetic measurements.
Host allocation/poison/copy median was85–138µs for standalone and492–495µs for chain.
MatMul reference/validation medians were1.53–2.25ms standalone and about1.53–1.55ms
in chain; LN references and receipt validation add further Host work.
Graph creation/finalization and first execute are exported separately.
Standalone finalization17.11–42.54ms; native LN2 finalization27.65–29.87ms,
tap68.51–74.76ms, fused69.07–75.40ms.

| Logical tensor payload | Bytes |
| --- | ---: |
| Either standalone MatMul, including full-word receipt | 65,600 |
| Copy/native-connected MatMul, including Copy receipt and native gradient | 98,400 |
| LN2 native graph, FLOAT32 tensors | 107,652 |
| LN2 native graph, integer receipts | 140,516 |
| LN2 native graph, total declared payload | 248,168 |

These exclude QNN internal allocations, Host input duplication, canary buffers and
alignment. MatMul allocates no dynamic scratch: scalar accumulator plus a volatile
product stack slot. Existing LN Forward has a bounded centered[128] stack array;
Backward has two Sum[128] arrays. Exact runtime scratch allocation, peak resident
memory and power draw remain UNRESOLVED; logical payload is not physical peak memory.
No performance ceiling was preregistered, so BLOCKED_PERFORMANCE is not assigned.

L19 × the two LN2 branches is4,980,736 FP32 MAC/microbatch, 39,845,888 MAC/Batch8
update. The branches require38 MatMul nodes, plus38 LN Forward/Backward and19 Add
nodes for95 unfused Custom nodes if this diagnostic pattern were replicated.
LN2 saved xhat/inverse logical payload is158,080 bytes/microbatch across19 layers;
all38 norms would be316,160 bytes. dWo materialization is311,296 bytes for19 layers,
and dContext outputs155,648 bytes. Lifetimes, reuse, graph splitting and the added
attention backward segment remain undetermined. No latency is multiplied by19,
38,304 or608 to claim production throughput.

HVX outlook: pinned compiler headers expose FP32 vector multiply/add intrinsics,
but no vector implementation was built. A bounded next candidate can vectorize
over output columns, preserving increasing-K order per lane and separate multiply
and add. dWo broadcasts one Context scalar and loads32 gradient columns; dContext
would benefit from a16KiB packed/transposed Wo tile for contiguous output-column
access. This avoids horizontal reductions, while FP32/subnormal behavior, alignment,
packing cost, scratch ownership, worker scheduling and full matrix parity still
need separate tests. The work is two vector kernels, an optional packing path and
their parity/lifetime tests; full-model migration is a separate implementation cost.
The scalar prototype establishes correctness, not a measured vector speedup.

## H. Next precision boundary

| Boundary | Input precision | Actual consumer | Accuracy status | Next requirement |
| --- | --- | --- | --- | --- |
| LN2 dX | Custom FLOAT32 | Residual Add | Verified here | Retain native ownership |
| DRESIDUAL1 | Native FLOAT32 | Custom dContext | Verified here | Attention consumer probe |
| DRESIDUAL1 | Native FLOAT32 | Custom dWo | Verified here | Gradient materialization |
| dContext APP_READ | Custom FLOAT32 | Future native head selector | UNRESOLVED | Same-word native output→selector accuracy |
| Selected head dContext | Standard HTP candidate | dProbabilities and dValue MatMuls | UNRESOLVED | MatMul and attention distribution fixtures |
| dWo APP_READ | Exact Custom store words | Host gradient registry/8-way accumulation | Materialization verified; accumulation UNRESOLVED | FP32 accumulation/scaling parity |

For actual G1 multiple heads, a selector MatMul first extracts each head's dContext.
Then `dProbabilities=dContext_head×V_headᵀ`, `dV=P_headᵀ×dContext_head`, softmax
product/reduce/center/multiply, dQ/dK MatMuls, attention scale, scatter/reduction and
projection gradients follow. The one-head branch skips head selection but still
has the two attention MatMuls. Gate/half variants add their own operations.
None are converted or accuracy-verified by this task.

Parameter registry uses row-major Wo[D,D], roleMUON. Production dWo is APP_READ,
copied to Host FLOAT32, schema identity checked, then each of8 microbatches contributes
`gradient[j] * 0.125f` to a FLOAT32 accumulator. The generalized loss gradient
already uses1/T; this probe keeps its supplied dY at full scale1 and does not insert
that loss normalization or microbatch scaling. No optimizer/training update is run.
Muon W8 and auxiliary Adam remain the existing consumers and are unchanged.

## I. Final decision

1. Custom dContext meets0.03: maximum standalone1.73685e-4 across both matrices.
2. Custom dWo meets0.03: maximum standalone6.12229e-4.
3. Native FLOAT32 preserves full consumer input words and bitwise final results.
4. Actual DSP LN2→residual→both MatMuls meets the X-origin double gates:
   dContext1.84879e-4, dWo5.87339e-4. Each variant has three process confirmations.
5. Next boundary is dContext→head selector/attention MatMuls and dWo→Host scaling
   and accumulation. Its final required FP32 extent remains unresolved.
6. Representative scalar arithmetic windows are about3.9–4.1ms standalone and
   4.7ms per branch in this diagnostic segment; scheduling variability is material.
7. HVX is worth a separately gated prototype if performance is needed. No speedup,
   changed reduction semantics or production readiness is assumed.
8. A real G1 one-layer diagnostic integration is justified as the next research
   candidate, including native-output consumer probes, registry parity and
   receipt-free observer control. This report does not implement or authorize it.
9. 40k→60k and200k remain **NO-GO**. Full backward precision, complete model
   integration and long-horizon stability require independent evidence.

## J. Provenance, reproducibility and protection

| Identity | Value |
| --- | --- |
| First scalar source/pilots | `6ad33db804bb1a62928ee407537d1b9700420743` |
| Frozen transport/prechain source | `3c35772a29be3a5d628cfa26acff902c00c74c89` |
| Final numerical source | `22d09e447be6902e802e37bc6418f0f132e060dc` |
| Final DSP package SHA256 | `a8bfcdd8d4feb20ecea9995f0bdf4937a0ca6deed414314f44bd8929be7e33e4` |
| Final Host package SHA256 | `5af836f0d2f95b0c7d7712ddd2fe8c3e6a447b6739ef01b506e7e24834c1e93d` |
| Final probe SHA256 | `81e5c8332d7d3daff0a60a2e705892e4783f23368b914103c151d59b93d2322b` |
| Hexagon compiler SHA256 | `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047` |
| NDK compiler SHA256 | `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3` |
| SDK | QAIRT2.48.40.260702151143 / QNN2.37.0 / HTP V81 |
| Tools | Hexagon6.6.0.0/tools19.0.07 / NDK26.2.11394342 |

Source-file and fixture SHA256, numerical process digests and sanitized timing are
in the allow-list exported aggregate. SDK/Skel/staged file hashes, device identity,
health/focus and raw per-case records remain in ignored private ledgers. Nine final
main processes use the exact same package/probe/source/runtime/fixture identities;
three additional Copy control processes use the same binaries. Earlier cohorts
and pilot evidence remain separate and are not overwritten or reused for promotion.

The bounded old-binary audit found identical inspected code/data/relocation sections
and different `.comment` linker command metadata with random temporary object names.
This explains a concrete source of hash variability, without a universal reproducible
build claim. During audit, objcopy appended its own command metadata to an input
binary. The old frozen numerical package was restored from an exact-SHA256 staged
copy, and all old frozen binaries were rechecked. Later disassembly uses read-only
objdump. Do not run objcopy against evidence originals, even for section dumping.
Exact raw binary identity remains mandatory; section similarity never substitutes
for a new binary's device cohort.

A read-only ADB audit stalled and was abandoned before any probe started. Fresh
read-only readiness and all guards were then rerun successfully; this is an
infrastructure event, not numerical failure or6031. Host20GiB/Device12GiB gates,
active-run protection, fixed APK/runtime audit and foreground monitoring were kept.
No target Activity was foregrounded. Protected corpus/tokenizer/order/40k and five
forensic checkpoint paths were hash-checked. Existing refs and UNSTABLE/resume guard
are preserved; 6031 stays UNRESOLVED / DORMANT / WATCH.

Validation: Fast source gates PASS; Qnn profile PASS (1109.0s, Host contracts and
diagnostics, shape validator, pinned APK build/audit); final custom package build
PASS; scalar/expanded receipt Host tests PASS; final device matrix and exporter
identity/independent-process checks PASS. Final documentation/export checks and
existing W8/device smoke results are recorded in the closeout below.
Private evidence is under ignored `build/reports/htp-fp32-custom-matmul-ln2/`, with
immutable process directories and frozen build copies. Public files contain no raw
tensors, checkpoint/corpus, device identifiers, private paths, APK hashes or SDK
binaries. Source commits and final documentation/export commit are separate.
Push/PR/merge/history rewrite and training updates are zero.

Closeout regression: Muon W8 headless114/114 PASS, maxAbs1.490116119e-8,
relativeL2 4.40269854e-8, minimum cosine1, RPC/finite success and fallbackfalse.
Device probe smoke PASS with pinned APK audit and foreground takeover0.
These synthetic diagnostics do not update the protected production training state.
