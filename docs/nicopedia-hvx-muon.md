# HVX FP32 Muon: environment stop and host design

Status (2026-09-09 historical section): **HVX TOOLCHAIN BLOCKED**. The
continuation below supersedes that environment stop for the explicitly
verified probe only; it does not promote the production training backend.

## Baseline and scope

HEAD and freshly fetched origin/main are `eb90a42`. Existing uncommitted HTP
diagnostics, including modifications to the CPU optimizer's auxiliary Adam
helper, belong to the starting working tree. They are preserved. This task
does not edit the CPU oracle or production code, commit, or push.

Fast Muon means Original Muon executed on HVX FP32; it is not a new algorithm.
Turbo is a separate algorithm and must not inherit Original's quality evidence.
The historical QNN diagnostic remains in [nicopedia-htp-muon.md](nicopedia-htp-muon.md).
It reports a single direct 64x64 probe with exact X0 and amplified downstream
differences. Those observations motivate investigating HVX; they do not prove
HVX correctness or establish the HTP internal arithmetic format.

## Environment and authoritative references

Neither HEXAGON_SDK_ROOT nor HEXAGON_TOOLS_ROOT is set in the task process.
No hexagon-clang executable is on PATH. Searches of the Qualcomm installation,
Scoop installations and local repositories found no Hexagon compiler, QAIC,
rpcmem.h or HAP_vtcm.h. Standard Program Files/tools directory names were also
checked. This is a bounded local search, not proof that no arbitrary hidden
installation exists. QAIRT includes QNN runtime binaries and HTP headers;
these do not establish a custom DSP build and deployment toolchain. A Zig
installation contains hvx_hexagon_protos.h, which alone is insufficient.

References inspected, without copying third-party code:

- [Qualcomm FastRPC](https://github.com/qualcomm/fastrpc): CPU/DSP transport;
  its README points to Hexagon SDK documentation for integration details.
- [llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md):
  an existing custom Hexagon backend reference, not proof of this device's domain.
- [llama.cpp HVX arithmetic](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/hvx-arith.h):
  128-byte vector loops; architecture-dependent FP32 add/multiply intrinsics,
  with a distinct pre-v79 conversion path. Local compiler headers and emitted
  instructions still need validation before adopting any intrinsic.

Required to resume: an explicitly located compatible Hexagon SDK including
QAIC, Android FastRPC host support, DSP skeleton/QuRT/HAP libraries and examples;
a compiler/linker/runtime that supports target v81 and 128-byte HVX; and a
verified target-specific DSP domain and custom library loading mechanism.
No SDK was installed, moved or downloaded. No domain was guessed. CDSP/user-PD,
FastRPC mode, VTCM capacity and allocation restrictions remain unverified.
QAIRT remains pinned by the existing policy; no alternate version is selected.

## Host-side design (not wired into the runner)

Keep algorithm (`ORIGINAL`, `TURBO`) separate from backend (`CPU`, `HVX`).
Default remains ORIGINAL/CPU. A future HVX selection must fail closed if its
transport or kernel is unavailable, with no CPU arithmetic fallback.

Flow: unchanged QNN forward/backward -> host-visible gradients -> persistent
shared arena -> one RPC for all Muon matrices -> validate returned weights and
momentum -> CPU auxiliary Adam. Publish next state only after every matrix
passes status, layout, complete-write and finite checks. An RPC error must not
leave partially updated live model state. Separate transport failure from
nonfinite tensor failure and numerical parity failure.

Use ParameterRole::MUON from parameterRegistry(), never name substrings.
The existing host packing test provides the starting orientation contract:

| Group | Count | Original shape | Canonical shape | Scale |
| --- | ---: | --- | --- | --- |
| Attention matrices | 76 | 64x64 | 64x64 | 1 |
| W1 | 19 | 64x128 | 64x128 | sqrt(2) |
| W2 | 19 | 128x64 | 64x128 | 1 |

Each descriptor should retain registry identity, role, original dimensions,
fanIn/fanOut, transpose flag, element count and checked buffer offsets. Derive
scale from semantic fanOut/fanIn. Validate the whole descriptor table before
writing any output; reject duplicate, missing, overlapping or out-of-bounds
entries and nonfinite inputs. Existing packing code is a host reference, not
yet a hardened RPC wire ABI or an HVX implementation.

Total Muon elements: 622592. One FP32 plane occupies 2490368 bytes. Separate
input weight/gradient/momentum and output weight/momentum planes require
12451840 bytes before descriptors and diagnostic taps. Retain allocation and
mapping across steps; report packing and synchronization costs independently.
Offsets can be 128-byte aligned; every matrix plane size is a multiple of 128.
The SDK must establish actual allocation and cache-coherency requirements.

For one canonical rectangular matrix, X and BX need 32768 bytes each, and
A/A2/B need 16384 bytes each: 114688 bytes of arithmetic scratch, excluding
stack, transpose buffers, alignment and runtime overhead. Sequential matrices
can reuse scratch. This is a size calculation, not a VTCM allocation claim.
Request VTCM only after capacity/lifetime/concurrency constraints are verified;
failure must be explicit. An explicitly selected DDR_SHARED path may be tested
separately, never reported as VTCM or used as a silent fallback.

## Frozen algorithm and promotion protocol

Original identity stays `keller_original_64560829_fp32`: momentum .95,
Nesterov true, epsilon 1e-7, coefficients (3.4445, -4.7750, 2.0315), NS5.
Preserve m=.95*m_prev+.05*g; n=.05*g+.95*m; transpose tall matrices;
normalize by sqrt(sum(n*n))+epsilon; repeat A=XX^T, A2=AA, B=bA+cA2,
X=aX+BX five times; restore orientation and apply the semantic aspect scale.
The CPU double accumulation oracle remains immutable. FP32 accumulation and
FMA behavior on DSP must be measured, not hidden by changing that oracle.

First prove vector add/multiply actually executed on HVX, including DSP build
identity, domain, instruction evidence, vector bytes, RPC status, finite output
and fallback=false. Only then attempt frozen 64x64 GEMM, single Original update,
rectangular update, and full 114-matrix parity, in that order.

Reuse the actual frozen binary under build/htp-muon/numeric-diagnostic/.
Do not regenerate X0 with device sinf: historical diagnostics recorded a
host/device last-ulp input difference. Record byte length and SHA256 before
dispatch and compare identical bytes to CPU double and CPU float references.
Report maxAbs, meanAbs, RMS, relativeL2, cosine and finite status.

Keep maxAbs <= .002, relativeL2 <= .001, cosine >= .99999 and all-finite gates.
Compare momentum, normalized X, NS1, NS5 and updated weights. Original's
conceptual GEMM budget is 15 per matrix / 1710 per full update; this is not a
measured kernel counter. Instrument actual counts when implemented.

After correctness, compare same-device CPU and HVX wall time including pack,
RPC/sync, momentum, norm, NS, update and unpack. Separate one-time setup.
Require >=2x speedup before training integration. Classify <1000 ms GOOD,
<500 ms VERY GOOD, <200 ms EXCELLENT. Historical CPU timing is not a current
matched benchmark. Try at most simple and blocked/VTCM-aware kernels.

## Checkpoint, Turbo, training and limitations

NPRTCKPTV4 and Original algorithm identity remain unchanged. CPU/HVX resume
compatibility is a future device gate, not established by host codec tests.
Future Original smoke is 1/8/32 steps with identical initialization, data,
tokenizer, batch and schedule; check loss, parameter/state differences,
finite tensors, separate QNN/HVX statuses and fallback=false.

Turbo research and implementation are deferred until Original is correct and
measurably faster. No paper, official algorithm, license, preconditioner,
coefficients or epsilon has been selected here. Changing NS5 to NS4 alone is
not Turbo. Future Turbo must use the same HVX primitives and authoritative
preconditioning, exactly four NS steps, separate identity and no Original
checkpoint reuse. If V4 cannot express its identity safely, restrict research
smoke to non-resume mode; do not introduce V5 in this task.

Future comparisons need singular values and ||XX^T-I_64||_F for canonical
64x64 and 64x128 updates, plus update cosine/relativeL2/norm ratio. These are
algorithm comparisons, not Original/Turbo parity gates. All training and
performance results are NOT RUN. Final_test, seed2, long training, Full gate,
QNN Muon retries and HMX experiments are NOT RUN. Direct HMX is future work.

## Evidence and reproduction

Private environment/toolchain inventories, initial file hashes and targeted
test logs are under ignored build/hvx-muon/. The host tests exercise existing
code only; they cannot promote an HVX backend. Reproduce the selected host
compiles using commands in host-tests.json, then run the resulting executables.
Fast verification command: `pwsh -NoProfile -File scripts/verify_local.ps1 -Fast -SkipAndroidBuild`.
Observed validation: all four targeted host tests passed (Original optimizer,
existing 114-matrix pack/unpack sentinel, NS stage reference, V4 checkpoint),
compiled with `-Wall -Wextra -Wpedantic` without warnings. Fast verification
passed 6 checks with 0 failures and 22 intentional skips; JVM tasks were
up-to-date. This is not a Full gate. Initial 24 modified/untracked file hashes
remain identical, and git diff --check passes. The only new non-build file is
this document. No algorithm/backend dispatch or HVX host stub is implemented.
No Android APK or DSP binary is built while the toolchain stop holds.
Resume at environment verification and vector proof after the missing SDK and
target-specific loading prerequisites are explicitly available.

## 2026-09-10 continuation: matched performance and profile

The continuation used the same physical device and the frozen 114-matrix
fixture (76 canonical 64x64 matrices and 38 canonical 64x128 matrices).
Original remains `keller_original_64560829_fp32`, momentum `.95`, Nesterov,
and NS5. The CPU baseline ran as a persistent ARM64 process with one warm-up
update followed by five measured updates; no JNI or process-launch time was
included. The authoritative CPU number is the Muon-only timer exposed by the
existing CPU optimizer; the outer mixed-update wall time is retained only as
context because it also includes auxiliary Adam.

| measurement | value |
| --- | ---: |
| device CPU Muon-only best / median / mean | 744,049 / 746,281 / 746,137 us |
| combined HVX RPC-inclusive | 738,283 us |
| combined HVX DSP kernel | 454,322 us |
| same-device CPU / HVX RPC speedup | 1.008x (best CPU Muon-only) |

The two-group architecture was also measured separately before the combined
RPC experiment. Combining the groups into one RPC was safe for the fixed
workload and reduced RPC-inclusive time versus the separate-RPC sum, but it did
not approach the required 2x promotion threshold.

The DSP profile for the combined 114-matrix update recorded 1,710 GEMM calls
(15 per matrix). Stage totals were momentum+Nesterov 24,636 us,
normalization 13,369 us, NS1--NS5 65,852 / 65,762 / 65,626 / 65,602 /
65,608 us, and final update 12,584 us. GEMM calls totaled 56,224 us (14.4%
of profiled DSP work); vector-call time was 2,263 us (0.6%). FastRPC wall
overhead derived from the comparable host/DSP elapsed timers was 283,961 us
(38.5% of RPC wall). Host pack and unpack were 10,264 us and 19,987 us.
These measurements classify the implementation as **MIXED**: RPC overhead is
large, while the DSP portion is dominated by non-GEMM NS/data-movement work.

The exact scratch requirements are 131,072 bytes for one square matrix and
212,992 bytes for one rectangular matrix, including transpose and temporary
buffers. The device reported an 8 MiB VTCM page and successfully acquired the
requested VTCM scratch, but the identical workload became substantially slower
(about 1.02 ms square-group kernel and 0.82 ms rectangular-group kernel versus
about 0.28 ms and 0.24 ms with DDR scratch). VTCM outputs remained finite and
within the existing numerical gates, so VTCM is not selected. DSP/host buffers
were 128-byte aligned, all GEMM dimensions were multiples of 32, and the
combined contiguous layout check passed.

The combined output passed the existing square, rectangular, and full-114
finite/parity gates. No unchanged correctness suite was rerun. Training smoke,
Turbo-Muon research, long training, seed-2, and Full verification were not
started because the matched speedup remained below 2x.

## 2026-09-11 continuation: production mode, power vote, and multi-HVX

This continuation stopped after Phase C. Original Muon semantics and the CPU
oracle were unchanged. A separate production-shaped DSP path retained the one
FastRPC call per update but returned only next weights and momentum. It omitted
the six normalized/NS diagnostic planes, per-GEMM, per-NS, and per-matrix
timers, intermediate tap copies, and diagnostic scratch poisoning. The
diagnostic path remains available and unchanged in purpose.

With one worker and the default power state, one warm-up plus five measured
updates produced RPC best/median/mean of 489,303 / 493,228 / 494,742 us and
DSP-kernel best/median/mean of 394,561 / 396,723 / 396,506 us. The best paired
RPC/kernel-external delta was 94,742 us. Production I/O transfers 12,452,752
bytes per update: 7,471,104 input bytes, 912 hyperparameter bytes, and
4,980,736 output bytes. This is distinct from the old diagnostic result of
738,283 us RPC-inclusive and 454,322 us kernel time.

The benchmark-local performance vote follows the installed SDK profiling
example: `HAP_power_set_apptype(HAP_POWER_COMPUTE_CLIENT_CLASS)` followed by
`HAP_power_set_DCVS_v3` in `HAP_DCVS_V2_PERFORMANCE_MODE`, with core and bus
corners set to `HAP_DCVS_VCORNER_MAX` and sleep disabled. It is session-local
and is cleared during close. At W=1 its RPC best/median/mean was 462,109 /
464,091 / 468,638 us, improving on default by 5.88% / 6.28% / 5.57%.
The single documented high-performance configuration was effective; no
additional clock tuning was attempted within the bounded Phase B scope.

`qurt_hvx_get_units()` reported `0x800`, or eight 128-byte HVX units. The
installed source for `qhblas_hvx_matrix_matrix_mpy_af` implements serial loops
and its archive has no thread/worker dependency for that symbol, so QHL was
classified as not internally multithreaded for this primitive. Matrix-level
workers use a static cyclic partition, acquire their own 128-byte HVX context,
reuse 212,992 bytes of private scratch, and write only non-overlapping outputs.
All requested workers acquired HVX; no CPU fallback or silent serialization
path exists.

| workers | RPC best / median / mean (us) | kernel best / median / mean (us) | RPC speedup / efficiency | kernel speedup / efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 462,109 / 464,091 / 468,638 | 383,689 / 385,955 / 386,282 | 1.00x / 100.0% | 1.00x / 100.0% |
| 2 | 271,799 / 272,102 / 272,099 | 192,673 / 192,750 / 192,781 | 1.70x / 85.0% | 1.99x / 99.6% |
| 4 | 183,708 / 184,774 / 184,453 | 104,790 / 104,917 / 105,021 | 2.52x / 62.9% | 3.66x / 91.5% |
| 8 | 138,907 / 138,984 / 139,553 | 60,848 / 61,078 / 61,059 | 3.33x / 41.6% | 6.31x / 78.8% |

The W=8 output passed all 114 weight/momentum comparisons: worst maxAbs
`1.49e-8`, worst relativeL2 `4.39e-8`, and worst cosine
`0.9999999999999989`. All outputs were finite; ordering, W1/W2 orientation,
and aspect scales passed; FastRPC returned success and fallback was false.

The selected W=8 best RPC time is 138,907 us, or 5.36x versus the same-device
CPU anchor of 744,049 us, so Phase C is a performance pass and multi-HVX is
highly promising. Kernel scaling is substantially stronger than RPC scaling;
the best-RPC sample retained 77,824 us outside the whole-kernel timer. The
post-Phase-C bottleneck is therefore the RPC transfer/kernel-external portion,
not evidence of QHL serialization or an HVX resource limit. Evidence is under
ignored `build/hvx-muon/fast/`. No persistent VTCM, DMA/double buffering,
custom fused kernel, HMX, alternate Muon algorithm, QNN custom op, or training
work was started.

## 2026-09-11 continuation: FastRPC and kernel-external overhead

The W=8 production path was isolated from Muon/NS math and worker changes.
A no-compute RPC using the same cached `rpcmem` buffer directions measured
median wall times of 141 us for 128 bytes, 321 us for 1 MiB, and 787 us for
the current 12,452,752-byte payload. The generated QAIC stub/skel passes the
three large buffers directly, and production mode has no wrapper full-buffer
`memcpy` or output copy. The buffers and handle were already allocated/opened
once and reused across the warm-up and five measured calls.

A diagnostic-only wrapper profile identified the dominant external work as
scalar DSP finite scans: median 48,074 us for the 7,472,016 input and
hyperparameter bytes and 32,043 us for the 4,980,736 output bytes. HVX power
up and down were each 3 us median. Thus FastRPC fixed/payload cache
synchronization was measurable but not the approximately 78 ms bottleneck.

The single optimization removed those redundant DSP scans from production
mode. The immutable fixture remains fully checked while the host packs it;
the output buffer is still poisoned before every RPC and fully checked for
finite values immediately after every RPC. The latter host check took 2,335
us median in the final run. This preserves separate operation-status and
finite-output gates without changing the math, QHL kernel, worker count, or
buffer layout.

With the same W=8 performance vote, one warm-up and five measured calls, the
RPC best/median/mean became 62,930 / 64,906 / 64,474 us and kernel
best/median/mean became 61,139 / 62,078 / 61,848 us. The best paired
kernel-external delta fell from 77,824 us to 1,791 us. Full-114 parity passed
with worst maxAbs `1.49e-8`, relativeL2 `4.39e-8`, cosine
`0.9999999999999989`, finite output, successful RPC status, and no fallback.
No VTCM, DMA, fused kernel, HMX, alternate Muon, QNN custom op, or training
work was performed.

## 2026-09-11 continuation: actual optimizer-step integration

The validated production W8 backend is now connected to the same optimizer
call site used by Nicopedia hybrid training. The host path structurally packs
the 114 semantic `ParameterRole::MUON` matrices, performs one complete finite
scan of the exact RPC input, poisons the full output with NaNs, dispatches one
production-mode RPC, and scans the complete returned output for finiteness.
Weights, Muon momentum, and auxiliary-Adam state are built as candidates and
are published by the training loop only after all checks succeed. There is no
CPU arithmetic fallback and checkpoint format/identity are unchanged.

The same actual optimizer function was measured in a headless
`BACKGROUND_CORRECTNESS` run using the fixed synthetic model fixture (seed 1,
114 matrices, 622,592 Muon parameters), one warm-up, and five measured
repetitions. CPU and HVX include auxiliary Adam, result ownership, and all
validation/application work. The final run started and ended at 38.0 C with
Android thermal status 0. Host work was variable across the five samples, so
best, median, and mean are all retained rather than presenting a single
latency as stable.

| actual optimizer path | best / median / mean (us) |
| --- | ---: |
| CPU Original | 1,800,804 / 4,191,795 / 3,569,147 |
| HVX W8 Original | 531,316 / 855,625 / 729,664 |

| HVX W8 component | best / median / mean (us) |
| --- | ---: |
| gradient/state pack | 250,996 / 568,836 / 444,196 |
| exact RPC-input finite validation and output poison | 4,556 / 10,607 / 8,244 |
| FastRPC inclusive | 68,716 / 69,566 / 69,494 |
| DSP kernel (inside RPC) | 63,957 / 64,085 / 64,290 |
| host output finite validation | 7,637 / 7,759 / 7,740 |
| unpack, auxiliary Adam, and candidate application | 199,158 / 200,188 / 199,984 |

The aggregate end-to-end CPU/HVX ratios were 3.389x for best, 4.899x for
median, and 4.891x for mean. These ratios describe this five-repetition run;
the stable approximately 64 ms DSP kernel alongside variable host timings
means they are not a claim of steady-state throughput. The final full
comparison passed 114/114 matrices for both weight and momentum: maximum
absolute difference `1.490116119e-8`, worst per-matrix relative L2
`4.40269854e-8`, minimum cosine `1`, RPC status success, finite output, and
`fallback=false`. An earlier layout-negative run correctly failed parity
before publication; its input plane order was fixed to the DSP ABI's
per-matrix weight/gradient/momentum order.

The targeted pack/finite-validation host test, HVX-disabled host compile,
QNN+HVX Android/app-test build, APK QAIRT ABI/hash/path/2.47 audit, runner
self-test, and headless device gate passed. QNN return status and returned
tensor finiteness were checked independently; focus takeover was zero.
`verify_local.ps1 -Fast -SkipAndroidBuild` passed 6 checks with 0 failures and
22 intentional skips. Full verification and long training were not run.

## 2026-09-12 continuation: steady-state pack initialization removal

The persistent `PackedInputs` path was further profiled without changing the
RPC layout or Muon arithmetic. Across one warm-up plus five measured actual
full-114 optimizer updates, every measured update performed zero vector
reallocations. All 342 data-plane `resize` calls per update grew vectors within
existing capacity, and their value-initialization accounted for the entire
allocation/resize timer: best/median/mean 17,752 / 17,783 / 17,798 us.

The single optimization sizes the six weight/gradient/momentum pack planes once
during warm-up, retains their final sizes, and writes every matrix directly to
its checked canonical offset. Metadata, the flat RPC copy, unpack, auxiliary
Adam, input/output validation, and the DSP implementation remain unchanged.

| measurement | before best / median / mean (us) | after best / median / mean (us) |
| --- | ---: | ---: |
| actual HVX optimizer update | 124,444 / 126,333 / 126,486 | 103,327 / 104,137 / 105,553 |
| pack | 21,541 / 21,741 / 21,702 | 3,800 / 3,814 / 3,933 |
| allocation/resize | 17,752 / 17,783 / 17,798 | 0 / 0 / 0 |

Median actual-update time improved by 17.57%, and median pack time improved by
82.46%. The before and after runs both held Android thermal status 0 and 34.0 C.
The after run passed 114/114 weight and momentum comparisons with maximum
absolute difference `1.490116119e-8`, worst relative L2 `4.40269854e-8`, and
minimum cosine `1`. RPC status succeeded, all returned tensors were finite,
and fallback was false. The targeted host sentinel, QNN/HVX builds, pinned
QAIRT APK audits, and Fast verification passed. Full verification, direct RPC
packing, unpack changes, auxiliary-Adam changes, and kernel profiling were not
run.
