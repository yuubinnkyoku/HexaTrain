# Optimizer and training critical-path follow-up

This follow-up starts from main `98d7c01f0904345a21d87ae6726dafda066d0b6a`
on `codex/training-critical-path-v2`. It uses the already optimized main as its
before arm. The earlier [throughput result](training-throughput-optimization.md)
remains historical evidence; its 390–410 ms baseline is not reused here.

The research baseline remains `headwise_g1_sigmoid` (760,960 parameters), with
`none` Control (758,528 parameters). Original Muon, five Newton–Schulz iterations,
normalization, Aux Adam, moments, LR schedule, update order, weight decay,
checkpoint V4/V5 codecs and fail-closed policy remain unchanged.

## Initial profile and timer interpretation

Latest-main physical-device profiling used seed 2, L19/H2/V1024/T32/D64/F128,
batch 8, 300 updates, eight HVX workers, Muon LR 0.0075, Aux Adam LR 0.0033,
linear decay at steps 4000–8000, target Aux LR 0.00015. Checkpoints were written
every 100 updates. QAIRT Build ID was `2.48.40.260702151143` throughout.

| Exclusive wall phase, ms/update | Control | G1 |
| --- | ---: | ---: |
| Whole training update | 182.973 | 177.362 |
| Optimizer update | 58.670 | 55.510 |
| Optimizer result/state move, outside optimizer | 26.267 | 24.988 |
| Gradient registry validation | 26.462 | 25.952 |
| Fused HTP forward/backward execute | 27.644 | 29.343 |
| Gradient accumulation | 2.250 | 2.214 |
| Checkpoint I/O, amortized | 6.472 | 6.567 |

Initial Control/G1 rates were respectively 5.465/5.638 updates per second and
3,780.238/3,899.821 original UTF-8 bytes per second.

Control's additional exclusive host phases were batch preparation 6.983 ms,
APP_WRITE schema validation 1.174, APP_READ allocation 0.135, poison fill 0.888,
materialization 2.184, finite validation 11.559, parameter bind 0.095, and
unclassified host work 12.188. These phases close the wall accounting; an alias
or nested timer must not be added again.

| Inside Control optimizer, ms/update | Time | Includes / relationship |
| --- | ---: | --- |
| Muon pack | 7.566 | Registry construction 5.716; flat copy 0.572 |
| Host RPC input validation | 2.766 | Separate from pack and RPC |
| RPC wall | 18.365 | Includes DSP execution span 17.248 |
| Host RPC output validation | 2.364 | Separate from RPC and unpack |
| Unpack and Aux Adam | 27.566 | Candidate generation 5.465, decode 2.669, full Aux wall 18.242 |
| Inside full Aux wall | — | Registry construction 6.368, validation 4.745, arithmetic 0.381 |

G1's corresponding optimizer breakdown was pack 7.088, input validation 2.462,
RPC 17.874 (DSP span 17.268), output validation 1.981, and unpack/Aux 26.072 ms;
full Aux wall was 17.711 ms. There is one matrix-batched RPC per update, with
114 matrices, unchanged matrix grouping and worker assignment. Mutex/session
setup and ownership changes are not inferred from a timer name.

The smaller `aux_adam_us` interval is not the full Aux Adam wall. It omits much
of registry and candidate work. `parameter_transfer` aliases parameter binding;
it is not a measurement of physical host-to-HTP transfer. Forward and backward
execute inside one QNN graph and cannot be separately timed by the existing
driver call. Its execute-call wall is the critical-path measurement, including
any driver-side transfer and synchronization; it is not pure HTP arithmetic.

DSP `muon_kernel` is the span around `run_release_parallel`: scratch/worker
allocation, thread creation, HVX lock acquisition, matrix kernels and finite
checks, joins and frees. It is not pure Newton–Schulz arithmetic. Mode 5 uses
direct RPC input/output buffers. Pre-lock server validation and power requests
are outside this span. RPC minus DSP span combines transport, wrapper,
validation and power overhead; it is not a standalone fixed FastRPC cost.

The separate physical optimizer profiler measured a 19.804 ms DSP span and
110.696 ms summed worker work: GEMM 53.376, non-GEMM 57.405, transpose 5.487,
vector work 3.242, normalization 1.648. Longest/shortest worker were
18.660/13.041 ms. These are overlapping worker totals, not additional wall
phases. Dispatch, lock, synchronization, allocation and result transfer do not
have independent exclusive measurements. No timing claim invents that split.

The largest exclusive phase was optimizer wall, but the main actionable cause
was shared host registry and parameter lifetime code compiled without
optimization. The same translation unit is used by pack/unpack, Aux Adam,
gradient validation, one-hot preparation and parameter state destruction.

## Candidate decisions

| Candidate | Expected gain / complexity / semantic risk | Decision and evidence |
| --- | --- | --- |
| Optimize shared CPU-reference/registry translation unit | Large across several host phases; very small implementation; CPU oracle contraction policy must be preserved | Selected with `-O2` and its original contraction default |
| Also disable contraction in that translation unit | No needed runtime advantage; changes CPU oracle rounding | Rejected: training checkpoints were exact, but generation oracle fields drifted, including logit extrema delta up to 1.431e-6 |
| Vectorize integer finite scans on locked DSP kernel | Several ms optimizer opportunity; modest build change; no FP arithmetic | Reverted with the validation family: a microbenchmark gain did not establish an accepted end-to-end gain |
| Healthy APP_READ integer finite/poison scan | About 11–13 ms initial scan; small reusable helper; scalar failure diagnostic retained | Reverted with the validation family |
| Shared host optimizer integer finite scan | Several ms input/output/Aux health work; small helper; no FP reduction | Reverted with the validation family |
| Persistent workers, rebalancing, async overlap, direct transactional writeback | Smaller opportunity after host fixes; greater ownership/order/failure risk | Deferred; not represented as measured negative results |
| Further QNN graph rewrite | HTP becomes important, but node count alone previously failed to predict speed | Deferred; existing Split/Concat negative evidence remains applicable |

Disassembly confirmed direct parameter-vector destruction in the optimized
translation unit, replacing calls through eleven unoptimized vector destructor
wrappers per layer. The measured state-move phase fell from about 25–26 ms to
about 0.2–0.3 ms. There is no ownership redesign or early writeback.

The rejected contraction-off prototype changed the CPU oracle's multiply/add
rounding contract: separate multiply and add rounding instead of permitted
contraction. That change was unnecessary. Its 300-update training checkpoints
and recorded loss curve were exact, but generation oracle fields were not;
the original CPU contraction policy was restored rather than accepting that
error or starting a quality study for a discarded prototype.

The discarded validation prototype used IEEE binary32 exponent bits and
integer OR, retaining NaN,
Inf, subnormal and signed-zero semantics without floating reassociation.
`memcpy` avoided aliasing violations; loops retained bounded tails. APP_READ used
a healthy fast scan and ran the original detailed scalar diagnostic on any
failure, preserving poison, nonfinite counts and first location. No safety scan
was removed. Only `original_qhl.c` enabled the DSP vectorizer, whose callers hold
the 128-byte HVX lock; the pre-lock RPC wrapper did not enable it. QHL libraries
and all numerical kernel equations were unchanged. These runtime/build
changes, their helper and their extra finite-pattern tests were reverted.

Twelve clean balanced pairs compared this family against the CPU build fix
alone, with 300 updates per arm. Control/G1 paired median step reductions were
17.48%/19.21%, but descriptive bootstrap 95% intervals were respectively
[-3.04%, 21.34%] and [-2.68%, 28.26%]. Only 9/12 and 8/12 pairs improved.
The adoption gate required at least 5% median gain and a positive lower median
confidence bound for both models; the family failed and is **NO_SAFE_GAIN**.
Its positive optimizer microbenchmark is not a reason to retain it. Background
phone activity and DVFS were uncontrolled; a causal explanation for changing
non-optimizer phases is not established.

No new training hot-path timers were added. Training keeps its existing mode-5
RPC path; the extra worker/GEMM profiling is confined to the separate optimizer
benchmark's mode 7. The APK auditor also keeps byte arrays intact instead of
boxing each byte through PowerShell's pipeline. Its report matched the old
auditor on the same main APK; this reduces orchestration time, not the measured
native training-step time.

## Measurement and compatibility

The CPU build fix meets **STRONG_GAIN** in the complete balanced physical
comparison and is kept on this branch. It has not been pushed, opened as a PR,
or merged into main. Single prototype timings are diagnostic only.

| Matched arm medians | Control before | Control after | G1 before | G1 after |
| --- | ---: | ---: | ---: | ---: |
| Training step, ms/update | 606.550 | 260.228 | 630.283 | 272.856 |
| Optimizer wall, ms/update | 155.822 | 72.573 | 162.680 | 75.155 |
| Updates/sec | 1.649 | 3.843 | 1.587 | 3.665 |
| Original UTF-8 bytes/sec | 1,140.599 | 2,658.604 | 1,097.662 | 2,535.512 |

Control/G1 paired median training-step reductions are **57.08% / 56.71%**,
with paired speedups **2.330x / 2.310x**. All four pairs improve for both models.
Descriptive bootstrap 95% median-gain intervals are **[56.81%, 59.48%]** and
**[54.39%, 56.84%]**. The effect is much larger than paired variance.
The G1-versus-Control paired overhead is **3.82% before / 4.76% after**;
this background session does not establish a universal architecture overhead.
Each 400-update run sees 276,732 original target UTF-8 bytes.

The absolute times differ substantially from the initial profile above.
The accepted claim compares the same session's main and candidate arms, not
the initial diagnostic run or the preceding optimization's historical values.
Phone activity and DVFS were uncontrolled; CPU frequency was not measured.
The measured builds are pinned QAIRT/HVX-enabled debug APKs.

### Re-profile after the CPU build fix

| Phase median, ms/update | Control before → after | G1 before → after |
| --- | ---: | ---: |
| State/result ownership move, outside optimizer | 115.187 → 0.622 | 114.603 → 0.648 |
| Gradient registry validation | 104.035 → 27.021 | 114.566 → 30.464 |
| Batch preparation | 28.256 → 0.183 | 27.986 → 0.182 |
| QNN fused execute-call wall | 46.792 → 46.870 | 49.743 → 50.067 |
| APP_READ finite/poison validation | 46.481 → 46.748 | 46.377 → 47.327 |
| Gradient accumulation | 7.711 → 7.816 | 7.793 → 7.967 |
| Native checkpoint I/O, amortized | 31.989 → 28.050 | 33.105 → 30.661 |
| Unclassified host | 54.496 → 15.746 | 54.861 → 16.414 |
| Muon pack, inside optimizer | 25.879 → 10.716 | 27.846 → 11.407 |
| Input validation, inside optimizer | 10.533 → 10.591 | 10.411 → 10.635 |
| RPC wall, inside optimizer | 19.065 → 18.684 | 19.400 → 18.689 |
| DSP span, inside RPC | 17.222 → 17.186 | 17.158 → 17.111 |
| Output validation, inside optimizer | 7.569 → 7.556 | 7.505 → 7.579 |
| Unpack/Aux, inside optimizer | 92.664 → 24.949 | 98.070 → 26.823 |
| Full Aux wall, inside unpack | 62.935 → 17.859 | 67.446 → 19.385 |
| Aux arithmetic, inside full Aux | 1.385 → 1.415 | 1.397 → 1.443 |

Every run passes its own exclusive wall-accounting closure. Component medians
need not add exactly to the whole-step median. Nested rows must not be added
again. DSP and QNN execute-call times barely change while shared host phases
shrink, supporting the host diagnosis rather than a DSP-kernel speed claim.
The largest remaining phase is optimizer wall, about **73–75 ms**, including
host validation/pack/unpack and an unchanged **17 ms** DSP span. QNN execute
and APP_READ validation are each about **47–50 ms**. The secondary validation
family was tested in twelve balanced pairs and reverted; asynchronous workers,
ownership redesign and numerical changes are not justified by this evidence.

The primary comparison uses four balanced main/candidate pairs, 400 updates
per arm, the same seed-2 recipe, and checkpoints every 100 updates. Two pairs
run main first and two candidate first; model order also alternates Control/G1
and G1/Control. Reported before/after phase and rate values are arm medians;
the improvement is the median of matched step-time ratios. Build, APK audit,
installation, graph setup and host checkpoint downloads are outside the
native update timer. Native checkpoint writing remains inside it.
One-update runs additionally compare complete optimizer state, and separate
V4/V5 resumes from the same existing step-300 checkpoints compare the
300→400 segment with uninterrupted main training.

Public numeric tables use the explicit allow-list exporter. Raw checkpoints,
run identities, device identities, endpoints and APK hashes stay private.
Bootstrap intervals resample paired ratios 50,000 times with a fixed seed;
they describe this background-device session rather than independent quality
seeds or a guarantee under every thermal/DVFS state.

The device was a physical NX741J / SM8850. Runs stayed headless in
`BACKGROUND_CORRECTNESS`; the phone remained usable. Foreground apps and DVFS
varied, so paired ratios, balanced order and repetitions are required. Thermal
status, battery temperature and voltage were captured for every run. No
exclusive mode, UI foregrounding, reboot, data clearing or firmware change was
used.

The runner verifies physical identity, installed APK identity, run ID, owner,
PID/heartbeat/lock and terminal status. A zero-exit truncated ADB pull was
detected by checkpoint decoding and recovered against device SHA; the binary
receiver now validates source size/SHA before transfer, local bytes, and source
identity after transfer. A later host disk-full pull was recovered from the
already terminal owned run without rerunning training. Neither infrastructure
failure is included as a training regression. Private evidence and exception
logs are retained. Identical checkpoint copies were independently byte-verified
before ReFS copy-on-write storage; they remain distinct files, not hard links.
The comparison interrupted by the disk incident is retained for parity only;
a fresh comparison in the same arm order replaces it in the accepted performance
cohort because recovery allowed cooling between arms.

Two other validation-family pairs were interrupted by an ADB identity-query
timeout and a large streaming APK-install timeout. Their completed native
arms retained healthy evidence, but the incomplete pairs were excluded in
full, rather than scored as numerical or performance failures. No completed
clean pair was excluded for its speed. The APK installer now stages audited
bytes in a content-addressed device cache, checks size/SHA before publication,
rechecks active-run gates immediately before local installation, then verifies
the installed APK. Unknown or corrupt cache state fails closed. Both training
and headless runners use it. Unconditional force-stop is removed; headless
timeout/health failures preserve device ownership for reconciliation rather
than killing an unverified device process. Earlier tests used the old runner;
this does not claim that the whole investigation never invoked force-stop.

All existing-checkpoint eval fields matched before/after. Generation compared
1,228 semantic fields per model, excluding timing. Both legacy baseline and
candidate retained `PARITY_GATE_REJECTED` with zero emitted bytes; this is
equivalent existing failure behavior, not a claim of successful generation.
Restoring the CPU oracle's contraction policy made those fields exact.

The existing quality study is inherited because equations/order are unchanged
and the observed checkpoint and recorded-loss trajectory is exact. Loss CSV
samples every 25 updates; this is not a measurement of every update's full
parameter hash. Checkpoints also include optimizer state and moments.

The primary comparison has **32 byte-identical checkpoint pairs** (four
intervals, four repetitions, two models), plus **two one-update checkpoint
pairs** covering optimizer output/state and moments. Initial/final parameter
hashes, first/last loss and every recorded loss-curve sample match. Actual
V4 and V5 resumes on both main and candidate ran the same existing step-300
checkpoint through step 400: final checkpoint bytes/hash and the recorded
300→400 loss segment match uninterrupted main, with healthy QNN/HVX/finite
fields. Load/eval, resume and generation compatibility are separate checks.
The primary runs recorded before/after battery temperatures of **30–33°C**
and thermal status **0**; this is not a claim that DVFS was fixed.
Long-run time-to-bpb/time-to-quality was not separately measured; proportional
time-to-quality gain is an inference conditional on unchanged trajectory and
sustained throughput, not a new quality result.

QNN success and finite tensors are checked independently. No fallback is
allowed. Incident 6031 tooling is preserved; status remains
`UNRESOLVED / DORMANT / WATCH`, with no observed recurrence in these runs.

## Verification scope

- Fast: PASS; parser, difference/binary audit, metadata, runner/policy SelfTests,
  short JVM and CPU-reference contracts. Heavy suites intentionally skip in
  this profile and are covered separately below.
  Final-source Fast took 76.4 seconds (10 checks passed, 21 intentional skips).
  The final documentation/package review repeated Fast: PASS in 78.7 seconds.
- Host: PASS (contract, diagnostic and parity-policy battery), including G1
  identity/counts/gradients, ungated deterministic trajectory, optimizer,
  pack/unpack, NS stages, finite classification, checkpoint V4/V5, resume,
  generation policy and first-nonfinite diagnostics. The discarded finite
  prototype separately passed an extended 1,002,560-pattern oracle with
  unaligned vector tails and bad/poison locations; that is prototype evidence,
  not a test added to the final implementation.
  Final-source Host took 400.1 seconds, with no skipped Host suites.
- Pinned Android native/debug/test APK build and APK audits: PASS. QAIRT core
  and Build ID matched; optional inventory omissions were advisory. No 2.47
  strings or host SDK paths were packaged; SDK library hashes matched. NDK
  26.2.11394342, Hexagon SDK 6.6.0.0 / tools 19.0.07 / v81 / HVX128 and pinned
  QHL libraries were retained.
  Fresh compilation reproduced every ARM library byte-for-byte. The Hexagon
  linker reordered PLT slots when its compiler temporary object names changed;
  only call-address relocations and those names differed. Canonical disassembly
  resolved every call to the same symbol, with identical arithmetic and order.
  The unchanged-source, audited measured pinned Skel and matching properties
  were reused for the final package. All **11 packaged native binaries** then
  matched the measured candidate byte-for-byte. This is explicit artifact
  reuse, not an SDK fallback or a DSP optimization. The fresh pinned compile
  and JVM tests passed before reuse; temporary paths/hashes remain private.
- Physical headless device smoke and full optimizer oracle benchmark: PASS.
  All 114 matrices retained the baseline oracle error (max absolute
  1.490116119e-8, worst relative L2 4.40269854e-8, cosine 1). This is CPU-versus-
  DSP numerical agreement, distinct from before/after bitwise trajectory parity.
- Runner binary-transfer SelfTest and allow-list exporter SelfTest: PASS;
  truncation/oversize/hash mismatch, nonfinite/rate/recipe/parity rejection and
  private-field exclusion are covered. The exporter writes under ignored
  `build/` and prevents source/output collision.
- Mocked APK-cache fault tests: PASS for hit, miss, test-APK flags, failed
  install, corruption, truncation and unknown remote state. Failed cases cannot
  reach installed-identity success; active-run checks precede installation.

Formal, external publication, main merge, long quality runs and SDK changes
are outside this runtime follow-up. The measured device evidence supports a
short-run throughput claim; it does not convert skipped Formal work into PASS.
