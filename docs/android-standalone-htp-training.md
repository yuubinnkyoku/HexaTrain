# Android standalone HTP training

## Scope and architecture

This document defines the Android application path for a user-initiated,
standalone HTP training run. It is a design and integration contract; it does
not claim that the path has been run on a physical device.

The current screen exposes the fixed canonical preset as read-only text. On
Start, the repository validates the SAF selection and creates an immutable
`TrainingRequest`. `TrainingSession` owns the request, a worker, structured
state, running timing aggregates, and the future native handle. The Android
adapter reuses `LiveUpdateForegroundService`/`LiveUpdateNotificationController`.
The production default is deliberately `UnavailableTrainingBackend` until the
typed JNI contract is implemented; it transitions to `ERROR` and never claims
an HTP run.

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
rejected with `ALREADY_RUNNING`; it must not attach to or cancel the active
native run.

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
the Kotlin snapshot, JNI mapping, and native acceptance is a terminal
`FAILED_CONFIG`, not a fallback to another model or backend.

## Dataset URI and SAF persistence

The dataset is selected through the Storage Access Framework. Persist only the
selected `content://` URI, its granted read permission, and non-sensitive
identity metadata such as byte count and digest. Take persistable read
permission at selection time and verify it again before Start and Resume.

The current Android adapter persists the URI/display name in app-private
`SharedPreferences`, takes/releases persistable grants, verifies a persisted
read grant and a lightweight readable descriptor, and does not read the file
on the UI thread. A bounded
stream or app-private normalized `train_pilot.bin` copy is still a native
integration task; no guessed filesystem path or fallback dataset is used.

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

## Structured progress and timing semantics

JNI progress is structured data, not parsed display text. Each event includes
the session ID, sequence number, monotonic elapsed time, phase, completed and
total steps when known, and optional loss/checkpoint metadata. Sequence numbers
must increase; duplicate or late events are ignored after durable recording.

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
HTP/CPU samples are rendered unavailable rather than relabeled as HTP.

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
or a non-finite required tensor ends the run as `FAILED_RUNTIME`; there is no
automatic QAIRT-version, backend, or CPU fallback.

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
durably completed checkpoint record. Native codec validation then remains a
second required gate. Any mismatch, partial write, missing record, or failed
validation is `FAILED_RESUME_INCOMPATIBLE`; starting from scratch requires an
explicit new request.

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
preference restore, structured `TrainingSession`/repository state, timing
accumulation and formatting, evidence-gated HTP labels, fail-closed terminal
payload handling, metadata-only checkpoint catalog, start-over confirmation,
and foreground-service notification adapter. Stub/scaffold only:
the JNI/native backend, native cache import/tokenization, native V2 save/load,
real phase timing evidence, pause/resume, and live process-death reattachment.

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
