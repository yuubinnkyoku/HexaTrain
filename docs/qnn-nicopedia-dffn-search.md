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

The same observation substantially raises confidence that the earlier
D16/FFN64 activity/focus failure was external UI/task interference rather
than width-dependent native behavior: its native work had already completed
all 3,609 QNN executes successfully and finitely before the lifecycle gate
failed.  The runner now records activity counters at reset, environment
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

For context, the previously recorded long anchor (8,000 steps, a different
run length) is val NLL 2.168420875 and dev NLL 2.146852226 at about 490 ms per
step.  It is not mixed with the 320-step screen comparison above.

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
shortfall, not treated as proof of useful text.  The FFN-only result suggests a roughly 1.33x steady-state training-loop wall
step-time, 1.30x HTP execute-time, and 1.39x checkpoint-size cost before its
headless safety failure; the D32 graph needs a
separate memory/graph investigation before another Tier 3 attempt.  A future
search should first diagnose and resolve the activity/headless invariant
failure, then repeat
the four-way screen with fresh anchor and multiple seeds before any 4,000-step
promotion.

For the normalization changes, fixed-QAIRT `assembleDebug` and
`assembleDebugAndroidTest`, APK ABI/hash/path audit, the runner/probe
self-tests, and the complete `run_host_tests.ps1` sequence passed.  The final
formal `verify_local.ps1 -WithQairt` was attempted after these changes but did
not complete within the 904-second execution ceiling, so formal verification
is incomplete and must not be reported as PASS.  This does not replace the
successful targeted QNN Android build/APK audit or complete host-test result.
