# Ordered FP32 HVX FFN2 feasibility

**Initial snapshot: BLOCKED / UNRESOLVED at the device safety gate.** The linked V81
package contains real HVX multiply/add code for NN `[32,128] × [128,64]`.
The device was Dozing, so no new DSP execution, numerical parity, native segment,
observer-free accuracy, or performance result is claimed. Code generation is a
separate positive result; it does not establish architectural feasibility.
40k→60k and 200k remain **NO-GO**.

After the user's continuation, a read-only refresh found Awake,79% battery,
36°C. Full preflight and a standalone pilot are being prepared; this initial
snapshot does not contain their results. The final report must supersede this
snapshot before any new device conclusion is made.

## A. Design and scope

Research branch: `investigation/ordered-fp32-hvx-ffn2`, starting at local
`5788d62d1632aa1b5650c1a2e23f2f71b170fc3e`. The starting tree was clean,
upstream unset, and 90 commits were unreachable from known remote refs. Those
local commits were retained. No fetch, reset, rebase, history rewrite, push, PR,
merge, production switch, optimizer/checkpoint change, or training update occurs.

`custom_matmul_ops.h::ordered` and `closure_ops.h` are byte-for-byte unchanged.
The existing 37 diagnostic operation names remain in order, with their existing
signatures, scalar counters, receipts, and QHPI interface 1.4. Two names are
appended: `FFN2Hvx` and `FFN2HvxNoReceipt`. Both are diagnostic-only/default-off;
the production backend never selects them. Their arithmetic exists only in the
DSP translation unit `hvx_ffn2.cpp`. Host preparation returns a fatal status if
asked to execute either HVX kernel; there is no scalar fallback.

The row-major layout is unchanged: A row = 512B, B row = 256B, Y row = 256B.
Each output row has two 128B tiles. `__HVX_LENGTH__ == 128` and
`sizeof(HVX_Vector) == 128` are compile-time requirements, giving 32 FP32 lanes.
The SDK's type definitions contain both 64B and 128B alternatives; this study
builds only the explicitly selected 128B mode. Runtime QHPI scheduling/vector
mode compatibility has not been tested on the device.

For each lane the mathematical contract is:

```text
acc = +0.0f
for q = 0 .. 127:
    product = round32(A[m,q] * B[q,n])
    acc = round32(acc + product)
Y[m,n] = acc
```

There is no horizontal/tree reduction. A scalar FP32 word is bit-copied to an
integer and broadcast with `Q6_V_vsplat_R`; B is loaded contiguously.
`Q6_Vsf_vmpy_VsfVsf` and `Q6_Vsf_vadd_VsfVsf` are separate operations.
FMA contraction, fast-math, and automatic vectorization are disabled. Only this
translation unit enables HVX and explicit strict IEEE mode. Scalar package flags
remain unchanged.

Exact-size `memcpy` loads/stores avoid a C++ assumption of 128B QNN pointer
alignment. The final machine code uses `vmemu`. Every source access stays within
the logical 128B tile; N64 needs no tail. No transpose, prepack, heap allocation,
staging scratch, or input-layout change is introduced. Backend physical
allocation/alignment and lifetime still require device verification. The new
Observed receipt records each pointer modulo 128 and the 128B vector width.

The new receipt has 16 header words, 12288 input words, and 2048 output words:
57408B. Its DSP marker is `0x48383114`, distinct from scalar `0x48383112`.
The header adds alignment, geometry, zero explicit scratch, and ABI version1.
Its counter is separate from scalar and NoReceipt. Same-kind concurrent graph
execution is outside this diagnostic contract. NoReceipt has no receipt or DSP
timer output; unobserved durations must remain null/unavailable.

## B. Compiler and disassembly

Pinned identities: QAIRT `2.48.40.260702151143`, QNN API `2.37.0`, HTP V81,
Hexagon SDK `6.6.0.0`, tools `19.0.07`, Android NDK `26.2.11394342`.

Local primary specifications audited:

- SDK `hvx_hexagon_protos.h`: the IEEE multiply/add intrinsics and broadcast.
- SDK `hexagon_types.h`: 128B vector and unaligned vector types.
- *Hexagon V81 HVX Programmer Reference Manual*, 80-N2040-63 AA,
  sections 2, 6.1–6.3: vector registers, extended qf32 state, rounding and IEEE
  intrinsics. FP32 holds 32 lanes in a 1024-bit register. Only round-to-even is
  supported by HVX floating point.
- *LLVM C and C++ Compiler User Guide*, “HVX and floating-point code generation”:
  legacy/lossy/compliant/strict modes differ. Compiler flags alone are not a
  device word-identity test.

The initial object audit exposed the default qfloat lowering. The final candidate
explicitly uses:

```text
-mv81 -mhvx -mhvx-length=128B -mhvx-qfloat
-mllvm -enable-xqf-gen=true -mllvm -hexagon-qfloat-mode=strict-ieee
-O2 -G0 -fPIC -ffp-contract=off -fno-fast-math
-fno-vectorize -fno-slp-vectorize -fno-unroll-loops
```

In the linked target symbol the compiler normalizes SF inputs to qf32, multiplies
qf32 operands, converts each product to SF, adds SF operands, and converts each
sum to SF. The qf32 intermediate must not be described as a single native SF
multiply or as retained extended-precision accumulation.

Static sites in the linked target: 4 vector multiplies, 4 vector adds, 8 SF
conversions, 8 normalization conversions, 4 broadcasts, and 5 unaligned memory
operations. These are static code sites, **not runtime instruction counts**.
No fused multiply-add, scalar multiply/add, call/library substitution, stack
frame, or stack spill appears inside that symbol. Scalar address arithmetic,
FP32 word loads for broadcast, loop predicates, and branch control remain.
QHPI wrapper finite checks/receipt copies are outside the arithmetic symbol.

The compiler software-pipelines q: a prologue supplies the first products, a
125-iteration hardware loop carries the accumulator, and an epilogue drains
the remaining products. Scalar A advances 4B and B advances 256B per q. The
recurrence consumes each rounded product in increasing q order, separately for
each tile; the outer hardware loop covers 32 rows. This is source/disassembly
evidence. Numerical confirmation of that order is **UNRESOLVED**.

The mode selects subnormal normalization and explicit rounding conversions.
Actual scalar DSP rounding/FTZ and HVX behavior, especially signed zero,
subnormal products/sums, and overflow rejection, remain a device test. No
alternative math or clipping was introduced to hide this uncertainty.

`scripts/audit_ordered_hvx_ffn2.py` inspects the linked symbol, ELF V81 tag,
vector operations, conversions, memory and absent calls/fusion/spills. Its
pattern checks complement the manual loop review; they are not a general
instruction interpreter or a proof of numerical parity.

## C. Standalone accuracy

The new `oneHvx` probe prepares four independent graph variants in the same final
package: Scalar/Observed, Scalar/NoReceipt, HVX/Observed, HVX/NoReceipt. Inputs
are identical FP32 words. Each variant owns its actual output; no Host-generated
FFN2 result is injected into a DSP graph.

The case set preserves checkpoint-derived frozen8 and existing stress8, and adds
12 controls: ordinary, ReLU-like, sparse, zero, signed zero, mixed weight signs,
large/small exact powers, cancellation, near-subnormal finite, fixed-seed random,
asymmetric row/column values, and safely scaled finite extrema. The added random
seed is `0x081ff123`, LCG1664525/1013904223. NaN, infinity, and unsafe overflow
are separate Host negative tests, not part of finite accuracy statistics.

Independent `one::mm` double and ordered-FP32 references do not call the Custom
dot. The probe compares all2048 output words with Host ordered FP32 and scalar
Observed, including signed-zero bit patterns, and reports mismatch count and
first mismatch words. Every Observed input/output receipt word is also checked.
The existing graph helper independently checks API return, output finite,
complete-write poison, canary, and input immutability.

Host results: the additional12 controls pass scalar ordered word equality and
the .002 double gate; maxAbs `1.32154673338e-6`, 24 Observed/NoReceipt calls.
Frozen8 Host scalar maxAbs is `7.18314831616e-7`. These are **Host scalar
results**, not a new DSP scalar baseline or HVX result. The existing closure
Host suite passes142 calls across its three operation kinds.

No new DSP outputs exist. Scalar DSP↔Host, HVX↔scalar, HVX↔Host, double
accuracy, and first mismatch are unmeasured. A future double-pass/word-fail must
be classified `HVX_NUMERICAL_GATE_PASS / ORDERED_WORD_IDENTITY_FAIL`, and must
stop native integration and performance promotion.

## D. Native segment and observer-free precision

HVX→Add and HVX→Add→LN1 F/B were **not executed or promoted**. This study stops
at standalone preparation because the required device gate precedes integration.
The new probe currently implements standalone only. It does not imply that a
successful standalone run completes the native/performance mission.

After standalone passes, reuse the existing DSP Add/LN implementations and
native ownership from `closure_probe.h`: FFN2 NATIVE → Add → X NATIVE → LN1,
with xhat/inverse retained through local backward. Check FFN2/X ≤.002, existing
case-specific Y/xhat/inverse gates, and dX/dGamma/dBeta ≤.02. Compare scalar/HVX
actual final words, producer→consumer receipt words, residual/parameter identity,
saved-state transfer, and observer variants independently. Backward dY remains
reference-fed. Do not inject a correct Host intermediate.

NoReceipt claims must use that variant's own actual final outputs and independent
double references. Native operands cannot be claimed directly observed in a
receipt-free graph. The existing 35-node Float Tap incident remains independent
and **UNRESOLVED**. Full Attention-derived dY, full one-layer Forward/Backward,
and all19 layers remain outside this result.

## E. Performance and safety

No new timing samples, paired speedups, process variability, finalize timing,
allocation/copy timing, or DSP timers are reported. Previously measured scalar
8.05–8.42ms and observer-free segment12.66–13.06ms remain historical context;
they are not denominators for this candidate. Missing NoReceipt timers are not0.

Predeclared acceptance remains unchanged: each of3 independent processes must
show ≥1.20× steady DSP arithmetic speedup for kernel promotion, and ≥1.10×
observer-free FFN2→Add→LN1 execute-wall speedup for segment promotion. Both
require double accuracy and bitwise identity first. Use warmup, at least16
steady samples per variant, ABBA and BAAB order, identical fixtures/package/
power/logging, medians/p10/p90/min/max and absolute differences. Stop on safety
or thermal guard failure. Keep pilot and final cohorts separate; exclude all
timing values from numerical digests. Device smoke and Muon W8 114/114 must be
rerun before final success. None has been rerun in this study.

At the read-only preflight, stable physical identity was resolved from two ADB
aliases of the same physical device, active training/headless runs were absent,
Host free space exceeded20GiB and device free space exceeded12GiB. Thermal
status0 and battery33°C were observed. Battery was15%, unpowered, and power state
was **Dozing**. Awake was a mandatory condition; device work stopped before
staging or executing the candidate. The cached top-resumed activity cannot
replace an Awake check. No wake, takeover, force-stop, permission/notification,
clock/charging/cooling, root/SELinux, firmware, SDK or training change occurred.

## F. Architecture implications

The ordered output-column design is a plausible shared arithmetic primitive,
with explicit qf32→SF rounding and layout-specific addressing kept separate.
It is not evidence that scalar Custom Op training is now fast enough.

| Candidate | Existing source contract | Reuse/design cost; no implementation here |
| --- | --- | --- |
| FFN2 | M32/K128/N64 NN | two32-lane tiles; contiguous B rows |
| Q/K/V | M32/K64/N64 NN | same tiling, three distinct parameter/output buffers |
| dContext | M32/K64/N64 NT | B column read; transpose/prepack and its lifetime/cost would be separate |
| dWo | M64/K32/N64 TN | strided A q read, contiguous B row; two tiles |
| dProbabilities | two M32/K32/N32 NT | right transpose/prepack; one tile/head |
| dQueryRaw | two M32/K32/N32 NN | contiguous right rows; one tile/head |
| dKeyRaw | two M32/K32/N32 TN | transpose-specific A addressing; one tile/head |

No other geometry receives HVX code. Future tails, packing ABI, counters,
instance ownership and concurrent scheduling need separate tests. Present static
register use cannot predict spills after row blocking or geometry expansion.

The generated code exposes costs worth measuring: normalization and two SF
rounding conversions per q, broadcast, contiguous B loads, loop dependence and
software-pipeline nops. The vector multiply is CVI_VX_DV and add CVI_VX, both
listed in slots2/3 by the header. This instruction mix and the dependency chain
are candidates for throughput limits; actual throughput, cache/DDR behavior,
DSP scheduling and QNN dispatch are unmeasured. No hardware counter was read.
CPU wall minus DSP math would not identify pure RPC/QNN overhead.

| Option | Assessment |
| --- | --- |
| A: scalar separate | prior fixed-fragment correctness evidence; current device regression pending |
| B: HVX FFN2 + existing Add/LN1 | arithmetic codegen demonstrated; numeric and wall benefit unresolved |
| C: HVX FFN2 + fused Add/LN1 | design only; potential8192B FFN2 intermediate and launch reductions |

Fusion must preserve native X lifetime for residual consumers and LN xhat/
inverse saved state. LN's row64 reduction is a separate ordered contract.
Direct row consumption might avoid a vector store/read, but vector/scalar
transition, scratch, saved-state writes and QNN complexity must be measured.
Arithmetic vectorization and boundary fusion savings are distinct. Without
device results, reprioritizing other MatMuls over FFN2 fusion is unsupported.
First finish this candidate's word/accuracy gates, then its paired timings.

FFN2 is262144 MAC/layer/microbatch, 4980736 MAC across19 layers, and39845888
MAC across19×8 microbatches. Its weights are32768B/layer, its unique logical
A+B+Y payload57344B, and its removable intermediate candidate8192B/layer.
These are geometry counts, not measured memory traffic, production scheduling,
training throughput, or full-model correctness.

## G. Decision

| Question | Evidence-based answer |
| --- | --- |
| 1. Real HVX multiply/add generated/executed? | Generated in linked V81 symbol: yes. DSP execution: unresolved. |
| 2. Increasing q confirmed? | Source/disassembly recurrence: yes. Numerical verification: unresolved. |
| 3. Scalar/HVX all output words equal? | Unmeasured. Host scalar↔independent ordered controls pass. |
| 4. Double accuracy maintained? | Host scalar controls pass; HVX device unresolved. |
| 5. Native FFN2→Add→LN1 pass? | Unmeasured; standalone device gate not reached. |
| 6. Observer-free accuracy pass? | Four standalone variants prepared; device unresolved. |
| 7. DSP math speedup? | No new samples; unavailable. |
| 8. Observer-free graph wall improved? | No new samples; unavailable. |
| 9. Memory/alignment cost? | zero explicit staging scratch/prepack; unaligned vector access; receipt57408B. Backend allocation/cost unresolved. |
| 10. Dominant factor if speedup insufficient? | No measured shortfall. Normalization/conversions, recurrence, broadcasts, memory and dispatch are hypotheses only. |
| 11. Generalization worth pursuing? | Plausible design reuse; practical value unresolved until FFN2 numeric/timing results. |
| 12. Next priority? | Resume standalone FFN2 under safe conditions, then native parity and paired performance. Defer other HVX geometries and fusion. |
| 13. Full one-layer gaps? | Prior Forward QKᵀ/scale/mask/P/PV/gate/Wo/FFN1/residual precision; Backward score scale/scatter/projection gradients/gate/FFN backward/DReLU/full LN1 dY; registry/branch/19-layer producer-fed scheduling. |
| 14. 40k→60k / 200k? | NO-GO. Training updates0;6031 remains UNRESOLVED / DORMANT / WATCH. |

| Independent classification | Status |
| --- | --- |
| HVX_FFN2_CODEGEN_VERIFIED | YES, linked target symbol only |
| HVX_FFN2_ORDERED_WORD_IDENTITY_VERIFIED | UNRESOLVED |
| HVX_FFN2_NUMERICAL_VERIFIED | UNRESOLVED |
| HVX_FFN2_NATIVE_SEGMENT_VERIFIED | UNRESOLVED |
| HVX_FFN2_OBSERVER_FREE_VERIFIED | UNRESOLVED |
| HVX_KERNEL_SPEEDUP_VERIFIED | UNRESOLVED |
| HVX_SEGMENT_SPEEDUP_VERIFIED | UNRESOLVED |
| BLOCKED_HVX_CODEGEN / NUMERICS / INTEGRATION | No such failure established |
| PERFORMANCE_NOT_PROMOTED | No performance candidate eligible; no measured below-gate result |
| Overall | BLOCKED by device safety; feasibility UNRESOLVED |

## H. Provenance and reproduction

Build/source hashes, exact HEAD, validation closeout and protected identities
are recorded in ignored `build/reports/ordered-fp32-hvx-ffn2/`. The source commit
is anchored after the implementation commit using ProbeOnly source/hash checks;
no device-tested source commit exists. Exact final report HEAD is obtained with
`git log -1 --format=%H -- docs/ordered-fp32-hvx-ffn2-feasibility.md`; its hash
and final clean status are saved in private closeout and the completion reply.

New DSP package SHA256:
`166b4974fabc64bc2cac03360173cce672c77d07149e84b5aa5e3add20fe854b`.
Host package:
`f27673572e38441b3a54ea1bd8c2838631485fe95bb764198e72249b691ece85`.
Probe:
`aefb7a364d3c704fd173f3f942d0c5fe597ddf4911cb9345c9966b6fd6eb553d`.
Linked target disassembly SHA256:
`7c88e9ea2d49667ac4c52d5eb6602f84e22863466d98fde6390e79427290a168`.

Hexagon compiler SHA256:
`039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`.
NDK compiler SHA256:
`ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3`.

Frozen fixture identities remain:

- one-layer inputs: `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`.
- LN1 parameters: `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`.
- LN1 producers: `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`.

The prior package/probe hashes from the mission match the preserved baseline.
They are not used as current numerical evidence. Frozen inputs were read without
decoding a checkpoint. Five protected hashes, including40k
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`,
are checked at entry/closeout. Previous branches,100k baseline,200k UNSTABLE,
Step41647/42k evidence, dataset/tokenizer/cache/order and private reports remain
preserved. Raw tensors, logs, identifiers, checkpoints, SDK binaries and absolute
local paths are not committed.

Reproduce Host preparation with the explicitly pinned SDK roots supplied as
variables (do not auto-discover/fallback):

```powershell
.\scripts\test_ordered_hvx_ffn2_host.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143
.\scripts\build_fp32_custom_probe.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -HexagonSdkRoot $HexagonSdkRoot -NdkRoot $NdkRoot
python scripts/audit_ordered_hvx_ffn2.py --hexagon-tools-bin $HexagonToolsBin --package build/fp32-custom/libHexaFp32Dsp.so --private-root build/reports/ordered-fp32-hvx-ffn2
.\scripts\verify.ps1 -Profile Fast
.\scripts\verify.ps1 -Profile Qnn -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143
```

When all device safety conditions are independently satisfied, a committed clean
source tree and matching manifest are required before this **standalone pilot**:

```powershell
.\scripts\run_fp32_custom_probe.ps1 -Variant oneHvx -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
```

This is not a benchmark/final-cohort command. Required remaining work is device
standalone gating, native segment implementation/verification, paired timing,
Device smoke/Muon regression, and3 final independent processes with identical
final binaries/fixtures and timing-free per-variant numerical digests. None may
be replaced by the build, static audit, previous study, or Host mock results.

Validation closeout is recorded below after all Host gates finish.
