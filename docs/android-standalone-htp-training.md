# Android standalone HTP training

## Scope and architecture

This document defines the Android application path for a user-initiated,
standalone HTP training run. It is a design and integration contract; it does
not claim that the path has been run on a physical device.

The current screen exposes the fixed canonical preset as read-only text. On
Start, the repository validates the SAF selection and creates an immutable
`TrainingRequest`. `TrainingSession` owns the request, a worker, structured
state, and running timing aggregates. In a QNN-enabled build the registry
injects `NativeHtpTrainingBackend`; it stages and validates the selected
NPRTBYTEV1 document, invokes the existing mode-100 JNI entrypoint from the
session worker, and maps the native report/callback into typed progress and
terminal state. A QNN-disabled build injects `UnavailableTrainingBackend` and
fails closed; it never silently selects CPU training.

```
UI -> TrainingRequest -> TrainingSession -> JNI adapter -> native HTP runtime
                         |                    |
                         v                    v
                 process-scoped repository  native checkpoint V2
                         |
                         v
              LiveUpdateForegroundService / notification
```

Only one process-wide training session may execute at once. A second start is
rejected with a structured `Failed` terminal summary; it must not attach to or
cancel the active native run.

## Compose training dashboard

The standalone screen is a PhoneLM-specific Jetpack Compose dashboard built
with stable Material 3 1.4 APIs. It uses a black, high-contrast surface
hierarchy, an expanded shape scale, large progress typography, a fixed action
dock, dynamic Android 12+ accent colors, and measured-value animation. The
dedicated Material 3 Expressive alpha theme/motion APIs are intentionally not
used, so experimental APIs do not escape into the application theme.

The screen is split into a training hero, observed training state, compute
activity history, fused/Adam/host performance, loss history, model/dataset
identity, a human event timeline, expandable structured diagnostics, and a
terminal summary. The external patch-manager screen in the design brief was
used only as an information-hierarchy reference. PhoneLM's Compose components,
state model, layout, colors, charts, and action semantics are original and do
not reuse its code or component structure.

`TrainingDashboardRecorder` belongs to the application-scoped session rather
than the Activity. Consequently loss/activity history, event ordering, and
terminal summary survive Activity recreation without creating a second worker.
It stores at most 512 observed history points and 256 human events. It never
interpolates loss, step, activity, timing, or memory. Missing observations stay
unavailable in the UI.

Native progress telemetry is emitted at the first step, every eight completed
steps, checkpoint completion, final step, and interruption. The eight-step UI
cadence is independent from the production checkpoint interval (250 steps) and
does not change training math, graph execution, optimizer state, checkpoint
format, or the production preset. Interval timing remains weighted by
`timing_sample_steps`. The Activity coalesces ordinary callbacks to the latest
state over a 125 ms window; phase changes, checkpoints, and terminal states
bypass that window. This bounds Compose/main-thread work without predicting
intermediate values or delaying lifecycle-critical events.

Process memory is the Android process PSS observation. Its relatively expensive
read is cached for one second while process CPU time is sampled at each accepted
progress boundary. HTP activity remains the ratio of measured QNN execute wall
time to the corresponding observation-window wall time. It is not NPU, DSP,
device, or hardware utilization.

The Android benchmark adapter and standalone adapter share a small Kotlin
`NativeRunArbiter` before entering the legacy process-global JNI bridge. This
prevents a standalone Stop from cancelling a benchmark-owned native run; the
native `gRunning` check remains a second defensive gate.

## ModelConfig is the configuration source

`TrainingModelConfig` is the single application-level source for model identity:
vocabulary/tokenizer identity, sequence length, dimensions, layers, heads,
optimizer parameters, seed, dataset identity, and checkpoint compatibility
fields. `TrainingPlan.NICOPEDIA_L19` adds the authoritative target (8,000
steps) and the NPRTCKPTV2 format/version. The UI only renders these values.

The JNI adapter maps that immutable snapshot to the native `TrainingConfig`.
Neither defaults in the UI nor values reconstructed after process death may
silently replace fields in the snapshot. Native validation remains authoritative
for resource bounds and graph-supported configurations. A disagreement between
the Kotlin snapshot, JNI mapping, and native acceptance is a terminal `Failed`
result with a configuration summary, not a fallback to another model or backend.

## Dataset URI and SAF persistence

The dataset is selected through the Storage Access Framework. Persist only the
selected `content://` URI, its granted read permission, and non-sensitive
identity metadata such as byte count and digest. Take persistable read
permission at selection time and verify it again before Start and Resume.

The current Android adapter persists the URI/display name and validated
content identity in app-private `SharedPreferences`, takes/releases
persistable grants, and verifies a persisted read grant. Selection runs on the
Activity IO executor. `NativeHtpTrainingBackend` repeats the validation,
copies the stream to an app-private run directory as `train_pilot.bin`,
computes SHA-256 plus the native cache FNV identity, and atomically replaces
the staged file before JNI. No URI or guessed filesystem path is passed to
native; only the backend's private staging directory is.

Dataset and checkpoint metadata use synchronous preference commits so a
process restart does not intentionally race an asynchronous write. The native
V2 writer provides atomic replacement; Android filesystem power-loss
durability is not claimed beyond the platform's flush/rename behavior.

## TrainingSession state machine

The Kotlin state uses these phases today:

- `IDLE`, `PREPARING`, and `INITIALIZING_HTP` cover request and native
  preflight setup.
- `TRAINING` and `SAVING_CHECKPOINT` carry structured progress.
- `PAUSED`: a compatible, finite checkpoint is available for explicit resume.
- `COMPLETED`, `INTERRUPTED`, and `ERROR` are terminal states.

The future JNI adapter may expose finer `VALIDATING`, `STARTING`,
`CHECKPOINTING`, `PAUSING`, and `RECONNECTING` events without changing the UI
contract.

Legal transitions are explicit and monotonic except an explicit resume from
`PAUSED`. A terminal native result is accepted only after its QNN return-code
status, required output tensor finite-status, and no-fallback evidence have
all been checked. A successful QNN return code alone is insufficient; missing
evidence turns completion into `ERROR`.
The Stop action is exposed only after a native training progress boundary is
observable (or while checkpointing). A close during preparation is retained as
a pending native stop and is never reported as an accepted UI stop.

## Structured progress and timing semantics

The JNI ABI is the existing synchronous String callback/report entrypoint,
but the backend treats its KEY=VALUE records as a strict typed boundary;
display text is never used as the state source. The Android adapter associates
one local run ID with the request and accepts only the native phase/step/loss
records emitted during that call; the current native callback does not provide
a native sequence number or live-query status. Monotonic elapsed time and
terminal ordering are therefore owned by `TrainingSession`, not fabricated
from log lines. A future handle-based JNI API must add an explicit sequence and
status query before claiming live process-death reattachment.

Timing fields have these meanings:

- `startedElapsedMs` and `eventElapsedMs` use a monotonic clock and measure
  active process time only.
- `wallClockMs` is display/audit metadata and is not used to calculate duration
  or timeout after clock changes.
- `nativeExecuteMs` measures the native call interval when supplied; it is not
  an HTP utilization measurement.
- `checkpointElapsedMs` includes serialization and durable replacement, and is
  reported separately from a training step.

Progress can be throttled for UI and notifications, but terminal state,
checkpoint completion, failure, and cancellation events are structured and are
not inferred from raw log text. `TimingAccumulator` keeps sum/count values
independent of UI refresh frequency. A phase is labeled HTP only when its
sample carries authoritative QNN/tensor-finite/no-fallback evidence; mixed
HTP/CPU samples are rendered unavailable rather than relabeled as HTP. The
current native Nicopedia graph has one fused forward/backward QNN execute, so
its measured value is exposed as `Fwd+Backward (fused)`; separate Forward and
Backward rows remain unavailable until native graph boundaries can distinguish
them.
Progress timing is provisional until the terminal report also validates the
pinned QAIRT runtime identity; a mismatch becomes an error and cannot produce
a successful HTP result.

## HTP activity and non-utilization wording

An HTP-active run means that the selected QNN HTP backend was created under the
pinned QAIRT policy, every required graph execute returned success, required
output tensors were finite, and the native report records no CPU fallback for
the claimed operation. It does not mean that every application operation ran
on HTP: URI I/O, tokenization, orchestration, checkpoint serialization, and
Android/JNI work can use the CPU.

Use the wording “the learning-step numerical operations were executed by the
QNN HTP graph” only when the run record supports it. Do not claim NPU-only
training, complete CPU non-utilization, or QNN automatic differentiation.
HTP creation failure, unavailable required symbols, a nonzero QNN return code,
or a non-finite required tensor ends the run as a failed terminal result; there is no
automatic QAIRT-version, backend, or CPU fallback.

The CPU process percentage is a non-privileged process CPU-time delta divided
by monotonic wall-time delta. `100%` means one logical CPU fully busy; a
multi-threaded process may exceed 100%, and the value is unavailable when
Android cannot provide either sample. This is process activity, not a device
CPU-utilization or HTP-occupancy claim.

## Checkpoint and resume

Standalone training reuses the existing native `NPRTCKPTV2` checkpoint codec
and its atomic write path. The Android layer adds a small durable session record
that references the checkpoint by opaque app-owned identifier and stores the
`ModelConfig`/dataset compatibility digests, completed step, and terminal
checkpoint status. It does not duplicate or reinterpret parameter or Adam
state.

Resume is fail-closed. Before native load, verify all of: V2 format, finite
checkpoint health, model and tokenizer compatibility, optimizer identity,
dataset digest/order identity, seed, requested resume step, and existence of a
persisted checkpoint record with a resolvable native payload. Native codec
validation then remains a second required gate. Any mismatch, partial write,
missing record, or failed validation returns a typed terminal `Failed` result
with a resume-incompatibility summary; starting from scratch requires an
explicit new request. The Android writer currently guarantees atomic
replacement and flush/rename behavior, not power-loss durability beyond the
platform filesystem contract.

The integration point is the existing native V2 save/load contract, not a new
Kotlin checkpoint serializer. This branch has an app-managed checkpoint
directory and metadata-only index plus `TrainingResumeRequest`; it does not
write or parse native parameter/Adam payloads. JNI must expose only opaque
save/load requests, compatibility metadata, and structured health results.
Raw checkpoints and tensor payloads are never surfaced to notifications, logs,
or public results.

## Foreground service, process death, and reconnect

Reuse `LiveUpdateForegroundService` and `LiveUpdateNotificationController` for
user-visible lifetime and progress. The service keeps a user-started run out of
the cached-empty process state; it is not evidence that native execution is
still alive and must not be made sticky merely to simulate resume.

On Activity recreation, the process-scoped repository is re-subscribed and the
selected URI is restored from app-private preferences. Notification callbacks
are serialized by the Android lifecycle adapter, and the foreground run is
started before the worker is queued. After a complete process death, only the
URI and metadata-only checkpoint index are restored; a native
run is not guessed to be alive. Explicit resume is offered only when a
compatible finite V2 metadata record (including dataset identity) is present.
Native live-run reattachment remains pending the JNI status contract.

## JNI contract and test backend boundary

The production JNI contract accepts an immutable request and cancellation
signal, returns a typed terminal result, and emits typed progress. It must
return enough evidence to distinguish native/QNN failure from tensor
non-finiteness and to report the pinned runtime identity. It must never select
another QAIRT SDK, reuse an incompatible cached runtime, or silently run a CPU
implementation.

A fake backend may implement the same Kotlin interface only in unit-test or
debug builds. It must be visibly identified as fake in state and UI, cannot
write production-compatible checkpoint claims, and cannot be selected by a
release build. A JNI stub is a development limitation, not HTP evidence; the
UI must show the run as unavailable or failed rather than successful.

## Physical-device validation not yet performed

Implemented in this branch: canonical config/plan, SAF selector and URI
preference restore, NPRTBYTEV1 staging/identity validation, production
QNN-enabled `NativeHtpTrainingBackend` wiring, structured `TrainingSession`
state, evidence-gated terminal handling, measured fused/Adam QNN timing and
checkpoint-I/O/host aggregation, native stop at a safe step boundary,
NPRTCKPTV2 opaque-path registration, metadata catalog, start-over
confirmation, and foreground-service notification adapter. The native V2
codec remains the single source of truth for parameters and Adam m/v; Kotlin
stores only compatibility metadata and an app-private opaque-path index.
Separate Forward versus Backward timing, in-run pause, and live native
process-death reattachment remain blocked by the existing native graph/JNI
surface. Resume after process death is metadata/path based and fail-closed;
it is not a claim that a killed native run is still alive.

The following are required before claiming the standalone path works on a
physical device. They are intentionally not executed by this document.

- Confirm the fixed QAIRT root and build ID using the repository policy before
  QNN build, APK audit, or device operation.
- Run the required local verification gates, QNN build, and APK audit.
- Validate a real SAF selection, persisted-grant loss, provider disappearance,
  and changed-content rejection.
- Validate start, progress, cancellation, interval checkpoint, V2 resume,
  incompatible-resume rejection, and process-death reconnect.
- Confirm the HTP activity definition with QNN return-code and required-tensor
  finite evidence separately, and confirm zero fallback for the claimed run.
- Apply the applicable device Tier gate. Any UI foregrounding, permissions, or
  formal UI validation requires explicit Tier 3 authorization.

## Integration audit snapshot

This integration was based on the committed training baseline
`7c9aa9947c71ca17dd6eeeac399944e92f5a135f` and incorporated the standalone UI
commit `d2de9e8f2d81d02c823578bef230311aedb99b6a`. The final integration SHA
is reported by the handoff rather than embedded here, so this document does
not become stale after the integration commit.
The shared production surfaces are the existing `NativeBridge`,
`TrainingEngine`, `nicopediaHtpTraining`, `NPRTBYTEV1` cache reader, and
`NPRTCKPTV2` writer/loader. This integration branch changes only the
standalone session/backend adapters plus the minimal native stop/evidence/
telemetry bridge. Existing headless PowerShell runners remain reference
tools and are not started by the Android app.

## PRE-DEVICE CHECKLIST

- Run `git diff --check` and the required host/JVM/build gates from
  `docs/agent/verification.md` in this integration worktree.
- For a QNN-enabled build, verify the explicit QAIRT root and Build ID and run
  the fixed-argument QNN build/APK audit. Do not use an auto-discovered SDK.
- Confirm the APK contains the production backend wiring and does not contain
  a test/fake backend selection in the release path.
- Review the native report contract: `qnn_return_code_success`,
  `output_tensors_finite`, and `cpu_fallback` are independent fields.
- Leave all device, emulator, ADB, installation, instrumentation, and
  long-training checks for the device-validation agent.

## Integration order

1. Define `ModelConfig`, `TrainingRequest`, compatibility digest, and the
   `TrainingSession` state machine with unit tests.
2. Add the SAF selector, persisted grant checks, and deterministic dataset
   identity validation.
3. Add the JNI typed request/progress/terminal contract and a debug-only fake
   backend for state-machine tests.
4. Connect the adapter to the existing native `NPRTCKPTV2` save/load and HTP
   execution contracts, preserving their fail-closed validation.
5. Reuse the foreground service and notification controller, then implement
   durable process-death reconnect.
6. Complete host/build/APK gates, followed by only the explicitly authorized
   physical-device Tier validation.
