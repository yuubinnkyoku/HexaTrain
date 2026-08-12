# Nicopedia HTP D/FFN width search

This is a research-only record for the dedicated `codex/nicopedia-dffn-search`
worktree.  The production Nicopedia L19 preset and UI remain T32/D16/FFN32.
No final-test split was opened.  Device runs used the pinned QAIRT 2.48.40.260702
Build ID 2.48.40.260702151143 and the HTP backend; the research harness requires
a fresh build/install and read-only APK audit before Tier 3 work.  CPU replay
is a diagnostic tail, not a substitute for HTP execution.

## Design

The primary comparison fixes L19/H2/T32/V256, batch 8, learning rate 0.003,
NPRTCKPTV2 resume semantics, and seed 1 for a 320-step screen.  The candidate
grid was deliberately small:

| candidate | D | FFN | trainable parameters | screen outcome |
| --- | ---: | ---: | ---: | --- |
| anchor | 16 | 32 | 48,320 | completed |
| D-only | 32 | 32 | 135,552 | dropped: terminal/ADB responsiveness degraded before a report |
| FFN-only | 16 | 64 | 67,776 | native training finite, but the headless activity/focus invariant failed |
| D+FFN | 32 | 64 | 174,464 | not started after the reproducible D32 responsiveness failure |

Parameter count uses
`2*V*D + L*(4*D^2 + 4*D + 2*D*FFN)`.  The research runner also records a
12-byte-per-parameter checkpoint payload estimate (parameters plus Adam m/v),
with the NPRTCKPTV2 header and registry overhead measured separately.

The screening cutoffs are fail-closed: nonzero QNN return, non-finite
application tensors, CPU fallback, or candidate-local failed status, more than 2x the
same-run anchor step time, or more than 4x the same-run anchor checkpoint size
drops a candidate.  Activity/focus, transport, thermal, battery, build, and
identity/preflight failures stop the phase rather than becoming candidate rows.
Generation must also be healthy, with valid/invalid UTF-8
byte counts no worse than the anchor; short-period-loop is retained as a
diagnostic rather than a sole promotion gate.  Extended (1,000 steps, seeds 1/2/4) and final (4,000-step
resume plus full-cap held-out evaluation and headless generation) phases are
implemented in `scripts/run_nicopedia_dffn_search.ps1`, but were not promoted
because no non-anchor screen candidate passed the safety/quality gates.
If a candidate reaches Final, the harness runs a matched anchor control at the
same final step count and seed set before interpreting the candidate result.

## Screen measurements

The completed anchor screen was one seed and 320 HTP steps, followed by 512
validation and 512 development chunks and a 64-byte greedy generation check.

| candidate | val NLL | dev NLL | HTP execute ms/step | steady-state training-loop wall ms/step | checkpoint bytes | QNN return | tensors finite | CPU fallback | generation |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| T32/D16/FFN32 | 2.805991517 | 2.923729841 | 168.675704 | 501.536799 | 596,137 | true | true | false | health true; 63/64 valid UTF-8 bytes; short-period-loop fraction 1.0 |
| T32/D32/FFN32 | n/a | n/a | n/a | n/a | not terminal | not established | not established | not run |
| T32/D16/FFN64 | n/a (held out) | n/a (held out) | 220.008459 | 666.426870 | 829,609 | true | true | false | not accepted |
| T32/D32/FFN64 | n/a | n/a | n/a | n/a | not run | not run | not run | not run |

HTP execute time is `(fused_forward_backward_qnn_us + adam_qnn_us) /
completed_steps`; the reported wall time is the steady-state training loop
average (host accumulation and checkpoint I/O included, initialization and
diagnostic replay excluded).
The FFN-only native report had 3,609 graph executes, zero graph-execute
failures, last QNN result 0, all training outputs finite, and no fallback.  The
overall device runner nevertheless failed its non-invasive headless gate:
`activity_create_count=1`, `focus_takeover_count=1`, and an `AssertionError`
was raised by the activity/headless invariant after the native work (the CPU
replay itself completed).  Its held-out NLL and generation were therefore
intentionally not promoted or used as a model-quality result.

The D32-only run produced no terminal report.  Its heartbeat stopped while the
native phase was still active and ADB shell calls became unresponsive; the
device was force-stopped only after that explicit safety condition.  Thermal
status remained normal and battery health/level remained good; thermal was
recorded, not used as a quality cutoff.

## Screening-basis normalization (2026-08-12)

The host `CHECKPOINT_MAGIC` failure was a fixture/reader version mismatch, not
a corrupt device checkpoint.  `nicopedia_cpu_generate` accepted only the
`NPRTCKPTV1` magic while the canonical device writer now emits
`NPRTCKPTV2`.  The host reader now accepts both versions and, for V2, consumes
and validates the Adam first- and second-moment registries (count, name,
shape, and finiteness).  A temporary V2 self-test fixture is deleted by the
test and never becomes research evidence.  The complete `run_host_tests.ps1`
sequence passes with this reader.

A dedicated `nicopedia-dffn-probe` path was added for width diagnosis.  It
does not call the normal long-training function, so it omits its step-0
comparison, fixed eight-step trajectory, CPU replay, and checkpoint tail.  It
prepares the same L19/H2/T32 training and Adam graphs, executes one B8 update
(eight fused forward/backward executes plus the required Adam chunks), and
reports graph preparation, QNN returns, all application-visible tensor
finiteness, updated parameter/moment finiteness, CPU fallback, and phase
progress independently.  The normal training and resume paths retain their
existing behavior.

The attempted D32/F32 probe did not reach native graph preparation.  The first
attempt found an AndroidX lifecycle binary mismatch because the installed
main APK did not match the freshly audited research APK.  Read-only inspection
then found a different PhoneLM UI build actively running
`StandaloneTrainingActivity` and its foreground service; the device-side main
APK was replaced again after a direct verified install.  The probe therefore
stopped at the concurrent-run/provenance preflight and D32 was not retried.
This is infrastructure evidence, not a D32 QNN or numeric failure.

The same observation leaves external UI/task interference as a live
hypothesis for the earlier D16/FFN64 activity/focus failure, but does not
distinguish it from other post-run lifecycle causes.  What is independently
established is narrower: its native work had already completed all 3,609 QNN
executes successfully and finitely before the lifecycle gate failed.  The
runner now records activity counters at reset, environment
prepare, before native, and after native.  It also force-stops only the target
package after an install (without clearing data), preserves the zero-activity
gate, and verifies SHA-256 equality for both installed main and test APKs
before instrumentation.  A concurrent GUI run remains a hard stop.

Because that independently owned UI training was still active, no fresh
anchor/candidate re-screen was run in this normalization pass.  In particular,
there are no new NLL, execute-time, wall-time, checkpoint-size, or generation
measurements to combine with the earlier table.  The next device session must
start with an exclusive, provenance-clean one-update D32/F32 probe.  If it
fails in graph preparation or QNN execute, the next bounded probe is
D24/FFN48; if D32/F32 is healthy, use short matched screens for D16/FFN32,
D16/FFN64, D24/FFN48, and D32/FFN32 before any multiple-seed extension.

## Clean D32 probe and attempted 32-step matched screen

On the next clean-device window, D32/FFN32 completed the dedicated one-update
probe.  The fixed-QAIRT build and APK audit passed, the installed main and test
APK hashes matched their local artifacts, and all activity/focus snapshots
were zero.  Both the L19 training graph and Adam graph prepared and finalized.
All eight fused B8 forward/backward executes and all five Adam chunks returned
successfully: QNN attempts/successes were 13/13, failures were zero, the last
and effective QNN results were zero, every checked output/gradient/updated
parameter/moment was finite, and CPU fallback was false.  Aggregate QNN time
was 72.077 ms for the eight fused executes and 167.014 ms for the five Adam
chunks.  This one update refutes an unconditional D32/L19 graph-construction,
first-execute, B8-update, or application-visible finite failure under this
exact state; it does not exclude a later long-run hang, watchdog interaction,
or unobserved DSP-internal memory pressure.

The previous AndroidX `Lifecycle.Event.Companion` crash was not reproduced
after a controlled build/install of a mutually matching main/test APK pair.
That old failure occurred before the test body/native entry while the device
held artifacts from another build.  No production dependency was upgraded;
the non-reproduction with a provenance-matched pair is consistent with a
stale/mismatched instrumentation artifact, but does not identify the exact
Android class-loading cause.  In either case it is not D32 native evidence.

A new private 32-step experiment was then preregistered with the fixed
candidate order D16/FFN32, D16/FFN64, D24/FFN48, D32/FFN32 and a unique
experiment-bound device run ID.  The anchor completed training, 128+128-chunk
held-out evaluation, and the short generation diagnostic.  D16/FFN64 completed
training and wrote a verified checkpoint.  Its evaluation preflight then
failed closed with `RUN_ALREADY_ACTIVE`; a read-only post-failure inspection
found a PhoneLM process and a non-visible MainActivity task while another
launcher remained top.  This host observation was not retained as a separate
raw device artifact, so it identifies the invalid run boundary but does not by
itself prove who launched the Activity.  The D16/FFN64 training report's
reset, environment, before-native, and after-native snapshots were all zero,
which confines the observed process/task mismatch to after that recorded
native/run boundary.  The eval preflight rejected the live process and stopped
the entire phase without force-stopping it.  D24/FFN48 and D32/FFN32 were
therefore not started, and the attempt is not a complete matched set.

The valid-but-incomplete native measurements are retained only to diagnose the
interrupted set; they are not mixed with the earlier 320-step evidence and do
not select a winner:

| candidate | valid scope | initial/final training NLL | delta / relative | HTP ms/step | wall ms/step | delta/wall-s | delta/HTP-s | checkpoint | QNN | finite/fallback |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| D16/FFN32 | train + eval + generation | 5.545779 / 3.507030 | 2.038749 / 36.76% | 205.19 | 1108.58 | 0.0575 | 0.3105 | 596,137 | 401/401, failures 0 | true / false |
| D16/FFN64 | training only; eval not started | 5.583708 / 3.477986 | 2.105722 / 37.71% | 260.71 | 1483.08 | 0.0444 | 0.2524 | 829,609 | 441/441, failures 0 | true / false |
| D24/FFN48 | not run | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| D32/FFN32 | matched screen not run; one-update probe healthy | n/a | n/a | n/a | n/a | n/a | n/a | no probe checkpoint | 13/13, failures 0 | true / false |

For the anchor, held-out validation/development NLL was 3.471694/4.326000.
The D16/FFN64 held-out values are intentionally absent because its evaluation
never started.  No multiple-seed or longer training is justified until one
clean four-candidate matched set completes.

For context, the previously recorded long anchor (8,000 steps, a different
run length) is val NLL 2.168420875 and dev NLL 2.146852226 at about 490 ms per
step.  It is not mixed with the interrupted 32-step screen above.

## Real-training follow-up (2026-08-12)

After the interrupted short-screen work, the clean gate was corrected so that
an inactive terminal status, stopped heartbeat, cached PhoneLM process, and
non-visible task history are recorded as inactive evidence rather than an
automatic `RUN_ALREADY_ACTIVE`.  A live heartbeat, test-process run lock,
foreground service, resumed/focused PhoneLM activity, or unknown/contradictory
state still fails closed.  The gate records reasons such as
`STALE_HEARTBEAT_ONLY`, `CACHED_PROCESS_ONLY`, `INACTIVE_TASK_ONLY`,
`ACTIVE_HEARTBEAT`, and `ACTIVE_ACTIVITY`; true/false/unknown self-test cases
pass.  The headless wait also ignores only a stale foreign status marker before
the new run identity appears; a live foreign marker remains an identity error.

The first fresh D24/FFN48 run reached and retained NPRTCKPTV2 checkpoints at
steps 250, 500, 750, and 1000.  Its host wrapper stopped during the diagnostic
CPU replay tail after step 1000 because no new checkpoint arrived for the
600-second watchdog window; this is not a native/QNN failure.  The checkpoints
were recovered before any rerun and all headers matched T32/D24/FFN48/L19/H2/
V256/seed 1.  Host checkpoint evaluation was finite at every point; the training-run diagnostic tail evaluated one chunk per split (CPU evaluator), so validation NLL moved 3.408663 (step 250) to 3.160144 (step 1000) and development NLL moved 2.870006 to 2.468762 under that 1+1-chunk condition.  On the dedicated held-out evaluation at step 1000 with identical inputs, host and HTP agree: 64+64 chunks give host validation 2.224771 vs HTP 2.224831 (delta 6.0e-05) and host development 2.639204 vs HTP 2.639192 (delta 1.2e-05), checkpoint_parameter_hash fnv1a64:f31295e03606f4ce, cache hashes fnv1a64:00fa2577fe03d1e5 / fnv1a64:f3914ae3cfd17464, 128/128 QNN success, finite true, fallback false.  The large apparent gap between the 1-chunk tail (3.16) and the 64-chunk HTP (2.22) is therefore an intended evaluation-condition difference (chunk count and first-chunk variance), not a CPU/HTP implementation or numerical divergence.  Full-cap host evaluation at 8192+16384 chunks gives validation 2.355019 and development 2.335255 for the same checkpoint, confirming the same condition dependence.

A controlled resume from step 1000 to 1250 then reached `PASSED/complete` and
produced a new 1,236,265-byte V2 checkpoint.  The resume report recorded
`resume_from_step=1000`, 3,000/3,000 QNN executes, zero failures, finite=true,
QNN return success=true, and CPU fallback=false.  Its short segment began at
NLL 2.418257 and ended at 2.536023, so that segment alone is not a monotonic
quality claim.  The 64+64-chunk HTP evaluation at step 1000 was independently
healthy (validation NLL 2.224831, development NLL 2.639192, 128/128 executes,
zero failures, finite=true, CPU fallback=false).  The resumed training report
measured 255.9 HTP execute ms/step (fused plus Adam cumulative time divided by
250) and 821.7 steady-state wall ms/step.  Greedy HTP generation produced 64
bytes with 63 valid UTF-8 bytes and one invalid byte; the short-period loop
fraction was 1.0, so generation was technically healthy but visibly repetitive.

The first D32/FFN32 real-training attempt did not produce a checkpoint or native
report.  The instrumentation ended with `Process crashed` while the headless
status remained RUNNING; read-only recovery then observed a PhoneLM process and
`StandaloneTrainingActivity` resumed/focused.  No QNN return, finite, graph,
OOM, or checkpoint evidence was available, and the run is therefore not
classified as a D32 model failure.  A post-recovery retry was first blocked by
installed/local APK hash mismatch, and a subsequent clean-gate read observed
the PhoneLM activity active again.  No install, force-stop, or automatic D32
retry was performed while that activity was active.  The remaining hypotheses
are runner/transport/lifecycle/provenance interference and, separately,
unobserved native latency or graph/DSP behavior; none is established here.

This follow-up is not a matched winner study and uses one seed only.  D24 is a
valid intermediate training/eval/resume path, but no wider configuration is
promoted to production.  D32 requires a new controlled device window with
matching APKs and no active activity before another real-training attempt.

## Checkpoint and identity safety

The canonical anchor keeps its historical filename.  Every non-anchor research filename
contains `-t<T>-d<D>-f<FFN>-step<N>.ckpt`; header validation, generation,
evaluation, resume, and host decoding all compare T/D/FFN/L/H/V/seed/step and
reject mismatches.  QNN return-code success and tensor finiteness are emitted
as independent fields.  Raw checkpoints, prompts, device identifiers, and
logcat remain under ignored `build/` paths and are not part of this document.
The legacy divergence-localization debug intent remains anchor-only; the width
search uses the generalized training/eval/generation headless paths instead.
Continuation phases require a fresh fixed-QAIRT build/install and bind the
screen, extended, and final CSV to the experiment id, source/dirty-tree
fingerprint, main and androidTest APK hashes, private train/validation/
development cache hashes, and phase step/seed/chunk identity.  A reused report
directory or changed artifact is rejected fail-closed.

## Decision

No wider configuration is recommended for production from this screen.  The
anchor remains the next production candidate: it is the only configuration
with a complete non-invasive screen, held-out NLL, and generation evidence in
this run.  Generation promotion also requires the explicit UTF-8 guard above;
the anchor's short-period-loop fraction of 1.0 is recorded as a quality
shortfall, not treated as proof of useful text.  The FFN-only result suggests
a roughly 1.34x steady-state training-loop wall step-time, 1.27x HTP
execute-time, and 1.39x checkpoint-size cost, but its held-out evaluation was
invalidated by the later live-process preflight.  The D32 one-update probe
establishes graph construction and one B8 optimizer update only; it does not
establish 32-step throughput or long-run stability.  A future search should
start only after the device is clean, then repeat the four-way 32-step set
from a fresh experiment root.  Multiple seeds and any 4,000-step promotion
remain unjustified until that matched set completes.

For the normalization changes, fixed-QAIRT `assembleDebug` and
`assembleDebugAndroidTest`, APK ABI/hash/path audit, the runner/probe
self-tests, and the complete `run_host_tests.ps1` sequence passed.  The final
formal `verify_local.ps1 -WithQairt` was attempted after these changes but did
not complete within the 904-second execution ceiling, so formal verification
is incomplete and must not be reported as PASS.  This does not replace the
successful targeted QNN Android build/APK audit or complete host-test result.
