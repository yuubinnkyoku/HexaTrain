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

The balanced physical comparison is strong matched-session evidence for this
candidate. Status remains **STRONG_GAIN / LANDING_CANDIDATE; DEVICE_AUDIT_PENDING**.
The 57% result is not promoted to a final or general speedup: the matched
before arm was much slower than the earlier profile, and a fresh device audit
must establish whether the candidate gain reproduces outside that slow state.
The candidate is not pushed, opened as a PR, or merged into main. Single
prototype timings remain diagnostic only.

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
The matched-session ratios compare its own before and candidate arms; they do
not establish that the gain generalizes from the initial profile. Phone
activity and DVFS were uncontrolled; CPU frequency was not measured.
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

## Landing-candidate provenance audit (2026-10-07)

This audit keeps the runtime candidate fixed. Exact base `origin/main` is
`98d7c01f0904345a21d87ae6726dafda066d0b6a`; candidate HEAD before audit-only
commits is `485651bfd64dd173af70654ed635e7a15a519085` on
`codex/training-critical-path-v2`. The four candidate commits are
`51c0047`, `35cf2c7`, `4016ca4`, and `485651b`.

Candidate path inventory:

- Build/runtime: `app/src/main/cpp/CMakeLists.txt` adds per-source `-O2` for
  `tiny_language_model_cpu.cpp`. No candidate C++ source file changed.
- Documentation: `docs/optimizer-throughput-optimization.md`,
  `docs/research-priorities.md`, and `docs/training-throughput-optimization.md`.
- Test-only: `host_tests/nicopedia_apk_cache_self_test.ps1`.
- Build/audit/runner tooling: `scripts/audit_qnn_apk.ps1`,
  `scripts/export_public_training_throughput_summary.ps1`,
  `scripts/nicopedia_runner_common.ps1`,
  `scripts/run_nicopedia_htp_training.ps1`, and
  `scripts/run_qnn_headless_tests.ps1`.

The only change compiled into the native library is the per-source compiler
option. Runner changes affect audit orchestration, APK identity checking, and
reporting, not the model's training math. The CMake build-system change affects
every function in that translation unit, including CPU reference numerical
routines, rather than only registry/lifetime routines. There are no generated
candidate files in Git.

### Effective compile command

Clean `origin/main` and candidate configurations used the same host, Android
NDK 26.2.11394342, Android arm64-v8a/API 26 target, QAIRT Build ID
`2.48.40.260702151143`, Hexagon SDK 6.6.0.0, HVX Muon enabled, and QNN enabled.
The complete generated commands are retained under ignored
`build/provenance-audit-20261007-r1/` as `compile_commands.json` for both
configurations. After normalizing only source/build directory tokens, the
target command diff is exactly one argument:

```diff
  ... -std=gnu++17 -fPIC ... -Wall -Wextra -Wpedantic
+ -O2
  -o CMakeFiles/phonelm_native.dir/tiny_language_model_cpu.cpp.o
```

Before had `-g` and no explicit optimization flag (therefore Clang's default
unoptimized code); candidate adds source-level `-O2`. Inherited and target
flags, `-D` definitions, include paths, architecture target, `-ffp-contract`,
`NDEBUG`, sanitizer, debug-info, and LTO settings are otherwise identical.
Neither command has `-ffast-math`, `-flto`, an explicit `-ffp-contract`, or
`-march`/`-mcpu`/`-mfpu`. Both retain `-g`, warnings, `-D_FORTIFY_SOURCE=2`,
and the same sysroot. There is no `-DNDEBUG`, so this change does not disable
assertions. O2 enables normal compiler loop/vectorization and inlining passes;
no explicit vectorization flag changed. `qnn_transformer_training.cpp` keeps
its existing `-O2;-ffp-contract=off` policy in both revisions. The CPU TU's
original contraction default is unchanged.

The exact command result is the source/build graph comparison, not the short
source diff. `scripts/compare_native_compile_commands.py` captures both
normalized full commands and the token-level diff in ignored
`build/provenance-audit-20261007-r1/compile-command-comparison-private.json`;
it reports `ONLY_ADDED_-O2`. The generated compile databases contain host
paths and are not committed.

### Host native-object and source review

The target translation unit compiled and the full `phonelm_native` shared
library linked successfully from both clean configs. Its unstripped ARM
object went from 166,668 to 85,164 bytes of `.text` across its text sections
(48.9% smaller); the object file itself grew from 2,077,800 to 2,127,392 bytes
because debug data is retained. In the complete library, `.text` went from
1,996,768 to 1,985,344 bytes (11,424 bytes, 0.57% smaller). Dynamic dependency
names and the set of 233 undefined imports were identical. `R_AARCH64_JUMP_SLOT`
relocations fell from 1,500 to 1,434; the other reported relocation types were
unchanged. The function-symbol table had 8,077 before and 7,754 after; among
the changed symbols, `parameterRegistry` shrank 508→292 bytes,
`validateParameterRegistry(vector<ParameterInfo>)` 1,636→956 bytes,
`validateParameterRegistry(TinyTransformerParameters)` 496→468 bytes, and
`TinyTransformerLayerParameters::~TinyTransformerLayerParameters()` grew
160→212 bytes. `splitParameterRegistry` grew 860→1,016 bytes, consistent with
inlining/code layout not uniformly shrinking every function. The private
section/symbol/relocation report is generated by
`scripts/compare_native_artifacts.py` and kept under ignored `build/`.
These are compiler/linker outputs, not a device-time estimate. QNN/HTP Skel
and unrelated DSP binaries were not changed by the candidate source diff.

Static review found no new arithmetic, aliasing, uninitialized-read,
object-lifetime, iterator-invalidation, or synchronization code: the candidate
adds only a compiler option. Registry dimension products use checked
`size_t` multiplication; existing vector moves use their normal ownership
semantics. This review cannot prove absence of all undefined behavior. No
sanitizer is present in the Android production command. A targeted host
ASan/UBSan link attempt was blocked because the installed host G++ lacks
`libasan` and `libubsan`; it produced no sanitizer findings. Existing exact
training checkpoint/loss-curve parity and generation-oracle evidence remain
valid for the adopted candidate. The rejected FP-contraction-off prototype
and reverted DSP vectorization, APP_READ scan, and CPU finite-scan family were
not re-experimented.

### Why the two reported CPU phases can shrink

`optimizer_result_move_ms` brackets four assignments of `std::move` for the
current parameters, Muon momentum, and two Adam states. With the same default
allocator, `std::vector` move assignment transfers vector ownership; it does
not copy each float. The optimizer has already produced its result state
outside this timer. Move assignment still releases the previous destination
state, including nested layer vectors. O2 can inline the repeated vector and
layer destructor/cleanup path and simplify short loops; earlier candidate
disassembly showed the before `TinyTransformerLayerParameters` destructor
calling the `std::vector<float>` destructor wrapper eleven times, while O2
inlined those checks and release calls into direct pointer tests and
`operator delete` calls. The same linked destructor grew 160→212 bytes because
the calls were expanded inline. That gives a
plausible host-code mechanism for a large timer reduction without changing
state ownership or arithmetic. It does not establish why the measured
absolute 115 ms became 0.6 ms on that device session, nor imply that all
allocations were removed.

For gradient validation, an update has eight microbatches. Per microbatch the
code builds `accum` and `source` registries outside the timed interval, then
times `nprtValidateRegistryIdentity`, which rebuilds two registries, validates
both, and compares semantic entries, shapes, axes, and element counts. With
19 layers and 13 metadata definitions, gated Control has 192 applicable
entries and G1 has 211. Per Control update this means 32 registry
constructions (6,144 `ParameterInfo` entries traversed), 16 validation passes
(3,072 entry visits, including uniqueness/hash, shape/role/axis checks), and
8 identity passes (1,536 entry comparisons). These are source-derived counts;
`unordered_set` node/bucket allocations and string/shape-vector allocations
depend on the standard library. Work is O(P) per registry/validation/identity
pass, plus string hashing/comparison and shape-axis checks. No validation was
removed. O2 can inline metadata assembly and iterator/loop machinery. These
counts explain an opportunity, not a conversion from host work to device ms.

### Initial profile versus matched comparison

The initial profile used 300 updates; matched before and candidate used 400.
Both used seed 2, L19/H2, V1024/T32/D64/F128, batch 8, Muon with eight HVX
workers, the same LR schedule (`0.0033` Aux Adam peak, `0.0075` Muon rate,
decay 4000–8000 of total 8000, target `0.00015`), and checkpoint interval 100.
Both report `production_fast` host validation and the production minimal
APP_READ ABI. Initial battery temperatures were 36–37°C; matched runs were
31–33°C for candidate and 32–33°C for before; thermal status was 0. CPU
frequency, governor, cpuset, battery saver, and screen/foreground state were
not captured in the old runs.

| Identity/setting | Initial 300-update profile | Matched 400-update before/candidate |
| --- | --- | --- |
| APK filename | `app-debug.apk`; exact run-to-artifact binding UNKNOWN | `app-debug.apk`; before and candidate app hashes differ |
| Android test APK | Exact initial run hash UNKNOWN | Same filename and byte-identical APK for before/candidate |
| Embedded build fingerprint | Not recorded; APK has no provenance asset | Not recorded; APK has no provenance asset |
| Runner / instrumentation | `instrumentation-v2` family | Same family; exact per-run script hash UNKNOWN |
| Seed / batch / model / optimizer / QAIRT / HVX | Same recipe listed above | Same recipe listed above |
| Steps | 300 | 400 |
| Checkpoints | 100, 200, 300 | 100, 200, 300, 400 |
| Validation / ABI mode | `production_fast` / `production_minimal` | `production_fast` / `production_minimal` |
| Battery / thermal | 36–37°C / status 0 | 31–33°C candidate, 32–33°C before / status 0 |
| CPU/DVFS telemetry | UNKNOWN | UNKNOWN |
| Exact app versionCode/versionName | UNKNOWN | UNKNOWN |

The saved artifact manifest proves that matched before/candidate app APK bytes
differ and their androidTest APK bytes match. Initial result files do not bind
each run to that manifest by hash, so exact initial-to-matched APK identity is
UNKNOWN. No raw APK hash is published. CLI args differ in steps, run ID,
report path, and arm gate; complete per-run command lines were not preserved.
Telemetry options were absent in old data. The 300/400 difference is not the
only measurement difference; the large absolute drift in matched-before is
unresolved.

| Phase | Initial main | Matched before | Ratio |
| --- | ---: | ---: | ---: |
| QNN execute-call wall | 27.644 ms | 46.792 ms | 1.692 |
| DSP span | 17.248 ms | 17.222 ms | 0.999 |
| State move | 26.267 ms | 115.187 ms | 4.386 |
| Registry validation | 26.462 ms | 104.035 ms | 3.931 |
| Optimizer wall | 58.670 ms | 155.822 ms | 2.656 |
| Training step | 182.973 ms | 606.550 ms | 3.315 |

CPU-side measured phases and QNN execute-call wall were slower in matched
before while DSP span was effectively unchanged. This supports a non-DSP
source of drift but does not identify CPU DVFS, host activity, instrumentation
overhead, or another cause. Candidate median 260.228 ms is also 1.42x slower
than initial main 182.973 ms. Existing temperature, run order, arm, and elapsed
samples are too few and lack CPU frequency to support a cause claim. Correlation
output is descriptive only. In the saved legacy reports, run order and device
CPU-frequency/governor telemetry are absent; run IDs reveal the experiment arm
but cannot bind every measurement to an APK hash. Battery temperature is
available for 16 matched runs and has a descriptive Pearson `r≈0.062` with
`training_step_ms`; Android thermal status is 0 in those samples.
`training_total_seconds` and `training_step_ms` come from the same timer/step
count (`r≈1`), so that is tautological rather than an independent signal. Arm
comparisons are available from paired results, but APK identity and arm are
confounded for per-run correlation because old manifests do not bind each row
to the artifact hash. Battery saver and screen/focus state are unknown.

The 300/400 paths share the same per-update loop and eight microbatches. Both
are below decay start 4000, so use the same schedule plateau and have no
warmup branch. At 400, one more checkpoint is written; loss-curve points add
325/350/375 and the final point, and the report/CSV has 100 additional update
rows. Progress/heartbeat cadence is unchanged, and each update rebuilds the
same registry. There is no periodic generation or validation hook. Final CPU
replay follows the last checkpoint and is outside the per-update timer.

### APK identity and future build provenance

The runner installs content-addressed app and androidTest APK bytes, verifies
device-cache and installed-package SHA256, and checks again after install. The
new runner accepts explicit APK paths, rechecks both installed packages
immediately before instrumentation and after the run, and pins runs to one
stable physical-device serial. The prepared audit runner writes a private
per-run manifest with app and androidTest hashes, app native library hash,
QAIRT HTP Skel hash, HVX probe Skel hash, QAIRT Build ID, installed
versionCode/versionName, declared source commit, host branch/commit/tree/dirty
state, host CMake/compile-command fingerprints when available, arguments, and
telemetry paths. If the APK has no embedded fingerprint it records
`DECLARED_UNVERIFIED`; host checkout provenance is not proof of APK source.

For future builds, embed a generated `assets/phonelm-build-provenance.json`
with commit SHA, tree SHA, dirty flag, branch, build timestamp, CMake config
hash, normalized target compile-flag fingerprint, QAIRT Build ID, HVX Skel
SHA256, and `libphonelm_native.so` SHA256. Generate it from Gradle/CMake inputs
and verify packaged values after assemble. Existing APKs do not contain this
asset, so the sidecar helps the next audit but cannot prove source provenance
retrospectively.

### Prepared next-device audit

`scripts/run_training_critical_path_device_audit.ps1` provides two modes:

- `BeforeOnly300vs400`: Control-only 300, 400, 400, 300 on one before APK.
- `MatchedAB`: 400 updates, four pairs per model by default, before/candidate
  × Control/G1. Odd/even repetitions reverse arm and model order; three pairs
  can be explicitly selected.

It requires explicit pinned QAIRT/Hexagon SDK roots and APK paths, refuses to
build, checks branch/candidate ancestry and runtime-source identity, pins one
physical device, uses the active-run fail-closed and ownership-safe runner,
and does not retry a failed run. Install occurs only when APK identity changes.
All output remains below ignored `build/reports/`. `MatchedAB` order is
before-Control, before-G1, candidate-G1, candidate-Control on odd repetitions;
even repetitions reverse both arm and model order. Each run manifest stores its
arm, model, pair, order, explicit APK/app-test hashes, and expected report path.

Runner review: the exact app and androidTest APK bytes are staged through the
content-addressed APK cache, then their installed package hashes are checked
after install, before instrumentation, and after the run. The audit wrapper
requires different app hashes and the same androidTest hash for the two A/B
arms. The analyzer also rejects an A/B report if either arm changes app APK
identity across runs or if the test APK differs between arms. Existing APKs
have no embedded build fingerprint, so the manifest can identify which APK bytes ran
but labels source provenance `DECLARED_UNVERIFIED`; it does not pretend the
host checkout SHA proves APK origin. The runner checks stale/active heartbeat,
test process, service/activity/task, and focus state before and after install;
unknown live state fails closed. It polls status every two seconds, emits
progress/heartbeat at the configured 30-second interval, enforces the declared
checkpoint-stall and outer 30-minute bounds, and saves instrumentation stdout,
stderr, status, result report, checkpoints, loss curve, and learning-rate CSV.
There is no package-data/cache clear and no automatic measured-run retry. On a
timeout it can stop only the instrumentation run carrying that exact run ID;
after success it reclaims only the completed run's owned app/test process.
No unrelated app process is force-stopped. Pre-run CPU/thermal sampling is one
ADB shell call; the mid-run sample occurs at most once in the existing
progress callback, and post-run sampling follows report collection.

All output remains below ignored `build/reports/`. Example:

```powershell
./scripts/run_training_critical_path_device_audit.ps1 `
  -Mode MatchedAB -QairtSdkRoot $PinnedQairtRoot `
  -ExpectedBuildId '2.48.40.260702151143' -HexagonSdkRoot $PinnedHexagonRoot `
  -BeforeApkPath 'build/audit-input/before/app-debug.apk' `
  -BeforeAndroidTestApkPath 'build/audit-input/before/app-debug-androidTest.apk' `
  -CandidateApkPath 'build/audit-input/candidate/app-debug.apk' `
  -CandidateAndroidTestApkPath 'build/audit-input/candidate/app-debug-androidTest.apk'
```

`scripts/capture_android_cpu_telemetry.ps1` takes one combined ADB shell
snapshot at pre, approximate mid-run, and post. It records online cores,
per-core current/min/max frequency and governor, cpuset, process status/sched/
cgroup, thermal status, battery temperature/saver, screen/foreground, uptime,
and load average. Missing or permission-denied values become `NOT_AVAILABLE`.
The mid-run snapshot is attempted once at a declared elapsed threshold, not by
polling the training hot path. Old reports cannot be backfilled with CPU
frequency data.

`scripts/analyze_training_critical_path_audit.py` normalizes initial,
matched-before/candidate, negative-result summary, and new reports. It emits
per-run CSV, per-pair before/after, paired speedup, median and fixed-seed
50,000-resample bootstrap CI, Control/G1 summaries, G1-vs-Control overhead,
phase distributions, thermal ranges, descriptive correlations, the overall
acceptance classification, and incomplete/excluded runs. The 300/400 script pairs adjacent runs as
300→400 and 400→300; the analyzer reports the medians for each step count and
the per-phase paired 400/300 ratios and per-update changes. This is descriptive
with two pairs, not an acceptance verdict. Exclusions are fixed before runs:
transport or
heartbeat/process failure, missing/incomplete report, QNN/HVX failure,
fallback, nonfinite tensors, and APK identity mismatch. Slow performance is
never an exclusion reason. CPU-frequency correlation remains
`NOT_AVAILABLE`. The FP-contraction-off prototype and reverted optimization
family are not rerun.

This audit does not choose `LAND`, `CONDITION_DEPENDENT_GAIN`, or
`NO_REPRODUCIBLE_GAIN`. The headline stays **STRONG_GAIN / LANDING_CANDIDATE;
DEVICE_AUDIT_PENDING** until the balanced physical-device audit is complete.

### 2026-10-07 physical-device audit (partial)

Current evidence classification: **INSUFFICIENT_VALID_PAIRS**. The landing
candidate remains fixed and its headline remains **STRONG_GAIN /
LANDING_CANDIDATE; DEVICE_AUDIT_PENDING**. No final performance adoption is
made.

The single physical device was identified as NX741J / SM8850. The prepared
before and candidate APKs both passed the pinned QAIRT 2.48.40.260702 / HTP
V81 audit. The runner verified the installed app and androidTest APK bytes
against the selected files after installation and before/after each run.
Both arms used the same androidTest APK and versionCode/versionName (1 / 0.1.0);
the app APK and native library hashes differed as expected. The APKs do not
embed a build fingerprint, so the per-run source attribution remains
`DECLARED_UNVERIFIED`. A private ignored build-provenance sidecar records the
source commit/tree, build procedure, APK/native/QNN/HVX hashes, and the
candidate CMake/compile fingerprints. It is not an embedded or signed
attestation; the before compile-command fingerprint was not retained.

The first Phase A attempt was stopped after two runs when the mid-telemetry
callback was observed rewriting its snapshot at each progress callback. Those
two runs and the unstarted run manifest remain preserved as protocol-deviation
evidence and are not used below. The callback's state was moved to script scope
in the isolated tooling commit `7f5d927`; a focused scope check produced
`CAPTURE, SKIP, SKIP`. The subsequent Phase A recorded exactly one mid snapshot
per run.

#### Before-only 300/400

The fixed-before sequence completed in balanced order 300, 400, 400, 300.
All four runs reported QNN success, finite tensors, HVX backend, no fallback,
and complete interval checkpoints. Thermal status remained 0 and battery
temperature ranged from 31–34°C.

| Phase (ms/update) | 300 median (n=2) | 400 median (n=2) | Pair 1: 400 vs 300 | Pair 2: 400 vs 300 |
| --- | ---: | ---: | ---: | ---: |
| QNN execute | 28.767 | 35.146 | +39.9% | +3.9% |
| DSP span | 17.334 | 17.198 | -0.5% | -1.1% |
| State move | 28.774 | 62.792 | +194.9% | +32.9% |
| Registry validation | 27.808 | 56.620 | +183.3% | +23.0% |
| Optimizer wall | 61.477 | 97.566 | +110.5% | +9.7% |
| Training step | 191.635 | 351.327 | +148.3% | +17.5% |

Both adjacent pairs were slower at 400 updates, but the size of the increase
varied substantially. DSP span stayed nearly constant while state move and
registry validation grew. Run length therefore explains part of the slow
before behavior, but this two-pair comparison does not explain the full
183-to-606 ms historical difference. Across these four observations, the
descriptive Pearson correlations with `training_step_ms` were CPU frequency
`r=-0.853`, elapsed time `r=0.955`, run order `r=-0.244`, and battery
temperature `r=-0.215`. These small-sample associations do not establish a
cause.

#### Matched A/B (400 updates)

The first complete balanced pair produced:

| Model | Before | Candidate | Gain | Paired speedup | Fixed bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 561.644 ms | 189.026 ms | 66.34% | 2.971x | [66.34%, 66.34%] |
| G1 | 533.317 ms | 262.015 ms | 50.87% | 2.035x | [50.87%, 50.87%] |

Each model has only one valid pair in this session; its bootstrap interval is
degenerate and provides no useful uncertainty estimate. These numbers are
interim observations, not an acceptance result or a general-device claim.
Their roughly 58.6% mean gain is numerically close to the earlier 57% result,
but this single pair per model cannot establish that the historical slow-state
gain is a clean-state general speedup; the before arm was still around
533–562 ms/update and the valid-pair count is below threshold.
The matched pair's main phase changes were CPU-side: Control state move
106.056→0.395 ms/update and registry validation 94.408→18.110; G1 state move
97.486→0.608 and registry validation 96.770→29.730. DSP span stayed near
17 ms/update. For this one pair, candidate G1 was 38.6% slower than candidate
Control; the analyzer's all-completed-run G1-vs-Control summaries are also
recorded in the private output, but include unmatched repetition-2 candidate
runs and are not acceptance-grade.

One before/Control repetition-2 training report completed 400/400 updates and
reported QNN success, HVX_W8, finite outputs, and no fallback, but the runner
failed while collecting post-run checkpoint artifacts. The manifest records
`RUNNER_FAILURE`; the exact exception was not persisted, so its subcause is
`UNKNOWN`. It is not excluded for being slow. The analyzer reports one
excluded run and two incomplete model pairs, leaving one valid pair per model.
The `single_flight_result=ALREADY_RUNNING` diagnostic also appears in a
successful run and is not evidence of a new 6031 event. Issue 6031 remains
**UNRESOLVED / DORMANT / WATCH**.

Across six completed matched reports, thermal status remained 0 and battery
temperature ranged from 33–34°C. Descriptive correlations with
`training_step_ms` were CPU frequency `r=0.473`, battery temperature `r=0.489`,
run order `r=-0.824`, and wall elapsed time `r=0.767`; the sample is small and
arm/run order are confounded, so no DVFS causal claim is made.

After the runner failure, three read-only preflight snapshots showed no active
PhoneLM run, battery saver off, screen dozing, thermal status 0, and battery
temperature 32–33°C. System load average rose from 9.27 to 14.42 to 17.77 for
the one-minute value (the last five-minute value was 7.91), while little-core
frequency remained at 883.2 MHz. This continuing device background activity
failed the clean-session preflight, so no further matched runs were started.
The audit analyzer's fixed result is **INSUFFICIENT_VALID_PAIRS**; the
performance question remains pending until the device returns to a clean
background-load state and enough new, complete pairs can be collected.
