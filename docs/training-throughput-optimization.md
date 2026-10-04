# End-to-end training throughput optimization

## Status: ADOPTED — STRONG_GAIN (matched device evidence)

The combined candidate is accepted. In one balanced matched device session it
cut `training_step_ms` by a paired median of **52.3% (Control)** and **50.2% (G1)**
against the same commit's own main build, with byte-identical checkpoints,
identical `final_parameter_hash`, finite outputs and zero QNN/HVX failures.

This is a **host-side runtime optimization**. Optimizer equations, LR schedule,
weight decay, Muon normalization, Adam moments, update ordering, architecture
(760,960 G1 / 758,528 Control), checkpoint codec (`NPRTCKPTV5` / `NPRTCKPTV4`),
incident tooling and production defaults are unchanged.

The research baseline remains `headwise_g1_sigmoid`. Legacy Control remains
`none`.

## Why the earlier estimate was wrong

The interim note reported Control `training_step_ms` around 250 ms. The matched
baseline re-measured the same recipe at **388–410 ms** per update. The earlier
number came from a short 200-update run, where the first steps include HTP graph
warm-up and allocator growth; the accepted figures come from 300-update runs
whose medians are dominated by steady state. All accepted comparisons are
paired ratios between two APK snapshots measured in the same session, never
absolute cross-session values.

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

Initial measurements are **one run per model**, in Control→G1 order, at 200
updates. They are recorded here because they defined the ranking, not because
they are the accepted performance evidence — see the accepted result for the
matched numbers. Battery temperatures were approximately 31–33 °C; Android
thermal status was 0.

| metric | main Control | main G1 | candidate Control | candidate G1 |
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

The largest measured single phase was optimizer wall, but unoptimized host loops
collectively exceeded it. Candidates were ranked from the measured critical
path, not QNN node/tensor counts. Ranking rationale used expected gain,
implementation cost, semantic risk and measurement confidence.

| candidate | expected gain / confidence | cost / semantic risk | outcome |
|---|---|---|---|
| optimize training loop translation unit | remove most of ~50 ms accumulation; high attribution confidence | small; keep rounding with `-ffp-contract=off` | **adopted** |
| optimize QNN host wrapper as well | poison/finite/schema loops total ~50 ms; high attribution confidence | small; ordinary IEEE checks remain | **adopted** |
| exact integer FP32 classification in DSP | remove scalar libc classification from the DSP scan; assembly evidence | small; no optimizer arithmetic change | **adopted** |
| rebalance W8 work / persistent DSP workers | ~5.6 ms profiled longest-shortest spread; small upper bound | more concurrency/lifetime complexity | not implemented |
| transfer overlap / asynchronous update | critical path is serial | ordering, failure atomicity, lifetime risk | not implemented |
| HTP node/layout changes | previous Split/Concat had no safe gain | graph correctness and measurement risk | not attempted |

### Prototype 1: training translation unit only

The debuggable APK already optimized Muon adapters at `-O2`, but compiled the
training loop and QNN host runtime at optimization level zero. The first small
prototype enabled `-O2;-ffp-contract=off` only for the training translation unit.

Control accumulation was **1.918 ms/update**, versus the initial main run's
49.512 ms. However, whole step was **289.012 ms**, versus initial 249.844 ms.
Unchanged wrapper poison/finite phases grew substantially across these runs,
so this unmatched comparison could not isolate runtime effects from device
state. That standalone prototype was **not** accepted on its own; it only
became adoptable once the combined candidate was measured against a matched
main APK in the same session. Its checkpoints already matched main
byte-for-byte.

### Adopted implementation

Three changes, all host-side, all semantics-preserving:

1. **`-O2` for the training orchestration and QNN host binding/validation
   translation units**, with `-ffp-contract=off` so the compiler may not fuse
   multiply-add. This keeps the exact floating-point reduction order of the
   batch reduction, gradient accumulation and the wrapper scans. Checkpoint
   bytes are the direct evidence: all matched pairs produce identical files.
2. **DSP-side exact IEEE binary32 finite classification by integer bit
   inspection** instead of a libc `_FDclass` call per element. Assembly
   inspection confirmed the per-element libc call disappears from the array
   scan. Every existing scan site, range and failure condition is retained.
3. **Telemetry**: nested HVX phase fields reused from existing backend timers,
   a signed step-accounting residual with `timing_accounting_ok`, and original
   target UTF-8 byte throughput. These add no new phase clocks and are not
   double counted in exclusive accounting.

### Rejected: rebalancing DSP workers

The W8 profile shows a longest-shortest worker spread of ~5.6 ms out of a
~19 ms kernel. Even eliminating the spread entirely is a small fraction of the
post-optimization step, and it would add persistent-worker lifetime and
concurrency risk. Not implemented.

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
Skel. Device numerical and finite gates passed; see the accepted result below.

## Accepted result

Method: two APK snapshots built from the same commit, differing only in the
adopted change. One session, alternating version order and reversing model
order between repetitions, 300 fresh updates per run, identical recipe, seed 2,
checkpoints every 100 updates. Values are paired per repetition; the reported
number is the median of the per-pair ratios, not a ratio of cross-session
absolutes. Battery temperature stayed within 31–41 °C and Android thermal status
was 0 throughout.

| metric | main Control | candidate Control | main G1 | candidate G1 |
|---|---:|---:|---:|---:|
| HTP fused forward+backward, ms/update | 51.342 | 27.737 | 29.025 | 30.148 |
| gradient accumulation, ms/update | 80.761 | 2.322 | 74.941 | 2.376 |
| optimizer update wall, ms/update | 92.331 | 64.075 | 91.725 | 62.125 |
| parameter input/output bind, ms/update | 0.463 | 0.108 | 0.516 | 0.131 |
| checkpoint I/O, ms/update | 7.697 | 6.370 | 6.796 | 6.239 |
| **training step, ms/update** | **409.940** | **195.444** | **388.420** | **193.353** |
| **updates/s** | 2.439 | 5.117 | 2.575 | 5.172 |

Paired `training_step_ms` reduction, median of per-pair ratios:

| model | pairs | median reduction | paired speedup | per-pair reductions |
|---|---:|---:|---:|---|
| Control | 3 | **52.3%** | 2.32x | 52.3 / 56.9 / 56.1 % |
| G1 | 4 | **50.2%** | 2.05x | 45.6 / 53.7 / 57.6 / 48.5 % |

G1 overhead versus Control in the same session: **-5.25% before, -1.07%
after**. The candidate is a global optimization; it helps both models by
roughly the same amount and does not widen the gap between them.

Original target UTF-8 bytes/s scales with the same factor because byte exposure
is unchanged: 138,035 original target bytes per 200 updates, reconstructed with
the same BPE model/cache and canonical selection generator, with a matching
selection hash in every report.

### Numerical parity

- **All 18 interval checkpoints across the seven matched pairs are byte
  identical**, including Muon momentum, Adam moments and optimizer step state.
- `final_parameter_hash` matches in every pair.
- `first_loss` / `last_loss`, dataset hash, training-order hash, parameter count,
  matrix counts, learning rates and schedule match in every pair.
- `all_steps_finite=true`, `qnn_failures=0`, `hvx_rpc_failure_count=0`,
  `hvx_fallback_count=0`, `hvx_nonfinite_count=0` in every collected run.
- Portable host test compares 1,002,560 float bit patterns against `isfinite`,
  covering every exponent/sign combination, signaling and quiet NaNs,
  infinities, subnormals, extreme mantissas, array tails and bad values at
  first/middle/last positions. PASS.
- The independent device optimizer benchmark reports 114/114 finite parity
  comparisons, maxAbs `1.490116119e-8`, worst relative L2 `4.40269854e-8`,
  no fallback.

Because trajectory is bit-identical, the existing multi-seed 3000-step quality
study does not need to be repeated. Time-to-quality improves by the same
factor as time-per-update.

### Checkpoint, resume, eval and generation compatibility

Format is unchanged: G1 `NPRTCKPTV5`, Control `NPRTCKPTV4`. Under the candidate
APK, existing G1 and Control checkpoints load and evaluate on device: both eval
runs reported `PASSED / SUCCESS` with the expected 760,960 / 758,528 parameter
counts, matching hashes, and finite checkpoint parameters.

Generation from these **step-300** checkpoints was executed on both arms and
rejected with `failure_classification=PARITY_GATE_REJECTED` and
`generated_byte_count=0`. This is **not a candidate regression**: the main
(`baseline`) APK produces the identical classification, the identical parameter
hash and the identical zero-byte output on the same checkpoints. A 300-update
model has not reached a quality level that satisfies the generation parity gate,
so this is a property of the checkpoint, not of the optimizer change. Device
reports show `checkpoint_finite=true`, `generalized_tiny_training_qnn_return=0`
and zero poison/non-finite outputs in every case.

Resume was verified on device under the candidate APK: a `NPRTCKPTV5` G1
step-300 checkpoint resumed and completed steps 301–400 with
`status=SUCCESS`, `run_completed_steps=100`, `all_steps_finite=true`,
`final_parameter_hash=fnv1a64:cf3453c27bdd9608`, `qnn_failures=0` and
`hvx_rpc_failure_count/fallback/nonfinite` all 0. The host-side deterministic
resume contract (`nicopedia_resume_test`, `resume_bit_identity=true`) and the
checkpoint byte parity above back this up.

One caveat on the surrounding harness, not on the resume itself: the training
runner's post-run host checkpoint evaluator could not start on this machine
(`HOST_CHECKPOINT_EVALUATOR_DECODE_FAILED`, process exit `-1073741515`, a
missing-runtime DLL rather than a decode result). The device run had already
completed successfully at that point, and the same evaluator passed during the
Host verification profile, so this is a local host-runtime issue and not a
checkpoint or parity failure. It is recorded here rather than reported as a
clean pass.

### Remaining bottleneck after the change

The critical path shifted. Optimizer update wall is now the largest single
phase at ~62–64 ms/update, of which the DSP kernel dominates; HTP execute and
gradient accumulation are now comparable or smaller. `optimizer_result_move`
rose in relative terms (23–25 → 26–30 ms) but a standalone probe shows the
move itself is O(1) at ~0.11 ms mean / 2.6 ms worst, so this is deallocation
and page-fault noise on large nested vectors, not new work. It is not yet a
bottleneck worth restructuring.

### Device health and incidents

- Matched runs stayed within 31–41 °C battery temperature, Android thermal
  status 0, no thermal throttling observed.
- One repetition's Control run stalled with a stale heartbeat and produced no
  checkpoint; the process exited and the run was **not** counted. Its evidence
  is preserved. This is classified as an **orchestration/transport stall**, not
  a numerical regression and not 6031.
- Earlier in this investigation, one checkpoint host pull was truncated and one
  polling loop hit `ADB_TRANSPORT_FAILURE`; both were recovered by re-pulling
  after verifying the same physical device, and the re-pulled bytes matched the
  reference checkpoint SHA-256. Classified as **transfer failures**.
- **6031 remains UNRESOLVED / DORMANT / WATCH.** Incident tooling and fail-closed
  gates are retained. No 6031 signature appeared in any collected run, and no
  incomplete run was treated as successful.
- No SDK fallback or version change, app data deletion, firmware change,
  notification/permission change, forced termination of an unknown run, or UI
  takeover was used.

## Verification

- Fast PASS; Host PASS (contract + diagnostic + parity-policy battery).
- G1 and ungated host regressions, metadata staleness/consistency, checkpoint
  and deterministic host resume contracts PASS.
- FP32 finite classification test PASS (1,002,560 patterns).
- Pinned QAIRT/HVX app and androidTest builds PASS; APK ABI/hash/Build ID/path
  audit PASS for the saved baseline and candidate snapshots, with no 2.47
  mixing and no automatic fallback.
- Device: matched A/B on Control and G1 with byte-identical checkpoints, device
  eval PASSED on both `NPRTCKPTV5` (G1) and `NPRTCKPTV4` (Control), and a
  device G1 resume from step 300 completing steps 301–400 with finite output and
  zero QNN/HVX failures. Generation from step-300 checkpoints is parity-gate
  rejected identically on main and candidate, as described above.
- Not run in this session: `Formal`, and a separate long-horizon
  time-to-quality milestone. Because the trajectory is bit-identical, the
  existing quality study carries over and time-to-quality improves by the
  measured per-update factor; a fresh measured quality milestone would be a
  separate research task.

No push, PR or main merge is authorized by this change.

Private raw evidence, immutable APK snapshots, recovery material, and prepared
matched-run/analysis commands are below `build/reports/training-throughput/`.
No raw checkpoint, device identifier, endpoint, logcat, private corpus or APK is
part of this document/commit.
