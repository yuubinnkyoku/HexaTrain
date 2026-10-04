# End-to-end training throughput investigation

## Status: BLOCKED — device acceptance is pending

This is an **interim investigation**, not a throughput promotion. The physical
ADB transport disconnected repeatedly during checkpoint collection and status
polling, and eventually had no online endpoint. The active-run/focus preflight
also failed closed with `RUN_STATE_UNCERTAIN`. No unknown run was stopped and no
device safety condition was relaxed.

The candidate code is prepared and host verified. **Matched performance,
candidate G1 device parity, device resume/eval/generation, final keep/revert, and
final baseline acceptance remain unverified.** Do not interpret the local
candidate commit or the accumulation micro-timing as an accepted speedup.

The current research baseline remains `headwise_g1_sigmoid`, 760,960 parameters,
`NPRTCKPTV5`. Legacy Control remains `none`, 758,528 parameters, `NPRTCKPTV4`.
Muon equations, NS5, momentum .95, Nesterov, auxiliary Adam, LR schedule,
architecture, checkpoint codec, incident tooling, and production defaults have
not been changed.

## Source and experiment identity

- Fresh branch parent: latest `origin/main` at start,
  `24827da9646ac143d114f28b1afbbf48ec2b21bf`.
- The preceding negative-result commit `c21ffb8` was inspected, not inherited.
  Its Split/Concat code is absent from this branch's parent and candidate.
- Physical device: NX741J / SM8850; stable identity was checked before each
  successful selection. USB/TCP aliases and transport evidence stay private.
- Pinned QAIRT Build ID: `2.48.40.260702151143`, V81, NDK r26c.
- Initial device profile: seed 2, 200 fresh updates per model, batch 8,
  V1024/T32/D64/FFN128/L19/H2, HVX W8 Original Muon + CPU auxiliary Adam.
  Aux LR .0033, Muon LR .0075, linear decay 4000–8000, schedule total 8000,
  target Aux LR .00015, experiment fork / parent LR .0033. This matches the
  promoted 1.5x research recipe; it does not change the training recipe.
- Checkpoints at 100 and 200. Native `training_step_ms` includes checkpoint
  writes and training progress callbacks, excludes graph setup and host pulling.

## Initial critical path

Source inspection found synchronous execution: eight sample preparations,
QNN executions and gradient reductions, then a blocking optimizer, state moves,
telemetry and any checkpoint write. No CPU/DSP overlap is implemented here.

Initial measurements are **one run per model**, in Control→G1 order. They are
profile observations, not matched before/after performance evidence. Battery
temperatures were approximately 31–33°C; Android thermal status was 0.

| metric | main Control | main G1 | accepted candidate Control | accepted candidate G1 |
|---|---:|---:|---:|---:|
| QNN fused forward+backward execute, ms/update | 23.960 | 26.627 | NOT_MEASURED | NOT_MEASURED |
| gradient accumulation, ms/update | 49.512 | 50.624 | NOT_MEASURED | NOT_MEASURED |
| parameter input/output bind, ms/update | 0.208 | 0.273 | NOT_MEASURED | NOT_MEASURED |
| optimizer wall, ms/update | 75.757 | 79.113 | NOT_MEASURED | NOT_MEASURED |
| complete training step, ms/update | 249.844 | 263.673 | NOT_MEASURED | NOT_MEASURED |
| updates/s | 4.002 | 3.793 | NOT_MEASURED | NOT_MEASURED |
| original target UTF-8 bytes/s | 2762.421 | 2617.539 | NOT_MEASURED | NOT_MEASURED |

The byte numerator is 138,035 original UTF-8 target bytes over 200 updates,
counting repeated training exposure, excluding context. It was reconstructed
using the same BPE model/cache and canonical selection generator; its selection
hash matches both native reports. It is not token/s or unique corpus bytes/s.
Initial G1-vs-Control complete-step overhead is +5.535%; with one sequential
pair it is descriptive, not an accepted final overhead estimate.

The additional exclusive phases explain why execute + accumulation + optimizer
alone is not a complete step decomposition:

| exclusive phase, ms/update | Control | G1 |
|---|---:|---:|
| batch preparation (includes input/target one-hot) | 3.320 | 3.355 |
| APP_WRITE schema validation | 4.316 | 4.463 |
| APP_READ allocation/resize | 0.174 | 0.232 |
| APP_READ poison fill | 22.250 | 24.217 |
| APP_READ materialization | 1.207 | 1.509 |
| combined poison + finite validation | 26.450 | 28.521 |
| gradient registry identity validation | 11.372 | 13.736 |
| optimizer result ownership move | 14.069 | 14.800 |
| checkpoint I/O | 4.156 | 2.491 |
| remaining host wall | 13.092 | 13.707 |

Including the small snapshot/bind/outer-validation terms, the existing exclusive
accounting plus residual closes to native training wall. The residual is
5.24% / 5.20%, and includes unmeasured initialization, destruction, host loss
calculation, gate diagnostics, telemetry and synchronization. It is not assigned
entirely to synchronization or physical transfer.

### Timing meanings and containment

- `fwd_backward_ms` and `*_us` are run totals; use per-update fields or divide
  by `run_completed_steps`. `training_step_ms` is already per update.
- QNN forward/backward are fused in one graph. Its host execute timer also
  contains driver work; separate forward, backward and driver transfer costs
  are **NOT_MEASURED**.
- `parameter_transfer` means **input/output bind**, not all parameter/gradient
  movement. It aliases APP_WRITE/APP_READ bind and must not be added twice.
  Gradient materialization and Muon packing are separately visible.
- `optimizer_update_wall` contains pack, RPC, validation, unpack and auxiliary
  Adam. RPC contains kernel. `aux_adam_us` is narrower than full Aux Adam wall.
- Poison and finite validation share one scan; poison-validation time is zero
  while the combined scan is charged to finite validation.
- New HVX phase fields reuse existing backend timers and are nested optimizer
  diagnostics. They add no new phase clocks. Byte exposure is counted after
  the timed loop. Dedicated timer-overhead measurement remains pending.
- New signed residual and `timing_accounting_ok` expose negative accounting
  residuals rather than hiding them behind the legacy nonnegative residual.

## Candidates and ranking

The largest measured single phase is optimizer wall, but unoptimized host
loops collectively exceed it. Candidates were ranked from that critical path,
not QNN node/tensor counts.

| candidate | expected gain / confidence | cost / semantic risk | current evidence |
|---|---|---|---|
| production optimization of training loop | remove much of ~50 ms accumulation; high local confidence | small; preserve multiply/add rounding with `-ffp-contract=off` | Control prototype measured; no accepted end-to-end gain |
| optimize QNN host wrapper as well | poison/finite/schema loops total ~53–58 ms; high attribution confidence | small; ordinary IEEE checks remain | built/audited; device result collection interrupted |
| exact integer FP32 classification in DSP | reduce scalar libc classification in ~60 ms kernel; assembly evidence, gain pending | small; no optimizer arithmetic changes | host classification parity and pinned build pass; device acceptance pending |
| cache/flatten registry, persistent buffers, fuse copies | secondary host traffic; re-profile after preceding candidates | larger ownership/API change | deferred until measured residual justifies it |
| rebalance W8 work / persistent DSP workers | ~9 ms profiled longest-shortest spread; upper bound is small versus initial whole step | more concurrency/lifetime complexity | inspected, not implemented |
| transfer overlap / asynchronous update | initial critical path is serial | ordering, failure atomicity, lifetime risk | deferred; cheaper semantics-preserving targets first |
| HTP node/layout changes | previous Split/Concat had no safe gain | graph correctness and measurement risk | not attempted; node count is not a runtime proxy |

### Prototype 1: training translation unit only

The debuggable APK already optimized Muon adapters at `-O2`, but compiled the
training loop and QNN host runtime at optimization level zero. The first small
prototype enabled `-O2;-ffp-contract=off` only for the training translation unit.

Control accumulation was **1.918 ms/update**, versus the initial main run's
49.512 ms. However, whole step was **289.012 ms**, versus initial 249.844 ms.
Unchanged wrapper poison/finite phases grew substantially across these runs,
so this unmatched comparison cannot isolate runtime effects from device state.
**The local accumulation improvement is not an end-to-end success.** This
standalone prototype is not adopted. Matched testing of the combined candidate
is required before retaining it.

Control checkpoints at 100 and 200 match main byte-for-byte, including optimizer
state, and first/last loss and final parameter hash match. No floating-point
reduction order or fused multiply/add was introduced. G1 candidate device
parity, long trajectory acceptance and time-to-quality remain pending.

### Optimizer reassessment and DSP candidate

The existing actual-optimizer benchmark ran one warmup plus five measured
updates, plus its W8 diagnostic profile. This is **attribution**, not final
training performance evidence.

| actual optimizer median | ms |
|---|---:|
| full HVX optimizer wall | 85.112 |
| pack | 3.789 |
| input validation | 1.071 |
| RPC | 62.616 |
| DSP kernel (inside RPC) | 60.491 |
| RPC minus kernel, paired | 1.390 |
| output validation | 1.327 |
| unpack/apply (includes Aux Adam) | 16.865 |
| full Aux Adam wall (inside unpack) | 11.443 |
| Aux registry construction / validation | 3.885 / 3.109 |
| Aux arithmetic | 0.240 |

Medians of nested phases need not sum to the median parent. W8 profiling
reported 15/15/14/14/14/14/14/14 matrices and all eight workers acquired HVX.
Longest/shortest worker medians were 58.766 / 49.543 ms; paired imbalance median
9.238 ms. Summed worker work was 381.714 ms, with GEMM 50.765 ms and non-GEMM
330.931 ms. **Summed worker time is parallel work, not step wall.**

The DSP compiler lowers array `isfinite` to a libc `_FDclass` call per element.
Across the current 114-matrix update, the input/intermediate/output validation
sites examine 17,588,224 floats. The candidate classifies their IEEE binary32
exponents with alias-safe `memcpy` plus integer masks. All existing scans remain;
NaN/Inf still fail, subnormals and signed zero remain finite. There is no change
to normalization, NS coefficients, QHL calls, matrix order or update equations.
Assembly inspection confirmed the array-loop classification calls disappear;
scalar configuration/norm checks retain their libc checks. This is not a
device throughput claim or an HVX-vectorization claim.

The portable host test compares 1,002,560 float bit patterns with `isfinite`,
including every exponent/sign, signaling and quiet NaNs, infinities, subnormals,
extreme mantissas, array tails and bad values at first/middle/last positions.
The new header is a Gradle DSP build input, so editing it cannot leave a stale
Skel. A final device numerical/finite gate is still mandatory.

## Failure handling and device health

- Main Control/G1 and prototype-1 Control each completed 200 updates with
  QNN failures 0, finite true, HVX RPC/fallback/nonfinite counts 0.
- The baseline optimizer benchmark had 114/114 finite parity comparisons,
  maxAbs `1.490116119e-8`, worst relative L2 `4.40269854e-8`, no fallback.
- Prototype-1 Control step200 host pull was truncated to 790,528 bytes from a
  6,622,141-byte remote checkpoint. The host decoder rejected it with
  `NPRT_CKPT_V4_STATE_BUDGET`. After verifying the same physical device, a
  complete re-pull matched main's checkpoint SHA-256. The truncated file and
  failed runner log were retained privately. This is a **transfer failure**,
  not a numerical regression or a 6031 classification.
- Prototype-2 polling stopped with `ADB_TRANSPORT_FAILURE`; its native terminal
  report has not been recovered. Its QNN/finite/6031 status is **UNKNOWN** and
  it is not counted as a successful run.
- **6031 remains UNRESOLVED / DORMANT / WATCH.** Incident tools and fail-closed
  gates remain. The successful collected runs did not report 6031; incomplete
  runs must be recovered before making any broader health claim.
- No SDK fallback/version change, app data deletion, firmware change,
  notification/permission change, forced unknown-run termination or UI takeover
  was used to recover transport.

## Verification and remaining acceptance work

Completed on candidate source:

- Fast PASS; Host PASS (contract + diagnostic + parity-policy battery).
- G1 / ungated host regressions, metadata staleness/consistency, checkpoint and
  deterministic host resume contracts PASS.
- FP32 finite classification test PASS.
- Pinned QAIRT/HVX app and androidTest builds PASS; APK ABI/hash/Build ID/path/
  2.47 audit PASS for saved baseline and candidate snapshots.

Still required, therefore overall **BLOCKED**:

1. Restore stable connectivity to the same physical device and pass active-run
   and foreground/focus preflight; recover the interrupted native report.
2. Profile the combined candidate on Control and G1; run optimizer parity and
   headless device smoke; measure clock instrumentation overhead.
3. Compare main and combined candidate in one balanced matched session:
   planned four repetitions × two models × two versions × 400 fresh updates,
   fixed recipe and checkpoints every 100 updates. Alternate version order
   and reverse model order. Record thermal/battery and analyze paired ratios.
   Strengthen measurement if a small gain cannot be separated from noise.
4. Compare checkpoint bytes, loss samples and metadata; load/eval/resume old
   V4/V5 checkpoints under the candidate, and verify generation on both paths.
5. Confirm identical byte exposure and report ms/update, updates/s, original
   bytes/s, final G1 overhead and time to the same measured quality. Do not
   substitute a projected time-to-quality for a measured quality milestone.
6. Keep/revert from whole-step evidence, re-profile any new bottleneck, update
   research/G1/Muon documentation with the accepted result, and commit the final
   acceptance decision. No push/PR/main merge is authorized.

Private raw evidence, immutable APK snapshots, recovery material, and prepared
matched-run/analysis commands are below `build/reports/training-throughput/`.
No raw checkpoint, device identifier, endpoint, logcat, private corpus or APK is
part of this document/commit.
