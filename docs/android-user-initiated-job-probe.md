# Android user-initiated job compute-only probe

Date: 2026-08-17

## Verdict

**UNSUPPORTED.** User-Initiated Jobs are User-Initiated Data Transfer (UIDT)
jobs in the current API, AOSP implementation, CTS expectation, and the tested
nubia Android 16 framework. A pure on-device compute job with no network
requirement is rejected by `JobInfo.Builder.build()`. HexaTrain training must
not add a dummy network constraint or otherwise present compute as a transfer,
and production training remains off UIJ.

## Why this was investigated

HexaTrain GUI training is explicitly started by the user and may perform
several hours of on-device HTP compute. A previous overnight experiment slowed
substantially in deep Doze, so UIJ appeared superficially similar to the desired
user-visible, long-running execution model. This probe tested API eligibility;
it did not migrate training or test training performance.

## Official behavior

- `android.permission.RUN_USER_INITIATED_JOBS` is a normal permission added in
  API 34. The API reference points callers to `setUserInitiated(true)`.
- `JobInfo.Builder.setUserInitiated(true)` represents an explicit user request.
  Scheduling must occur while the app is foreground/otherwise activity-launch
  eligible, requires `RUN_USER_INITIATED_JOBS`, and requires a job notification.
  The reference text says the network-only restriction applies in Android 14,
  but it does not say that later releases accept compute-only UIJs.
- The current UIDT guide describes the API specifically as a long-duration data
  transfer mechanism and constructs jobs with a required network.
- `JobService.setNotification(...)` is required within 10 seconds for a job
  whose `JobParameters.isUserInitiatedJob()` is true. It supplies the status-bar
  and Task Manager notification. `isUserInitiatedJob()` is the runtime guarantee
  indicator and can be false when UIJ requirements were not met.
- Current AOSP `JobInfo.enforceValidity()` unconditionally checks UIJs for
  `networkRequest == null` and throws `IllegalArgumentException` with
  `A user-initiated data transfer job must specify a valid network type`.
- Current CTS `JobInfoTest.testUserInitiatedJob()` requires the no-network
  `.setUserInitiated(true).build()` case to fail, while accepted cases explicitly
  use `NETWORK_TYPE_ANY`. CTS `UserInitiatedJobTest` likewise states that all
  user-initiated jobs require a network.

Primary sources:

- [RUN_USER_INITIATED_JOBS API](https://developer.android.com/reference/android/Manifest.permission#RUN_USER_INITIATED_JOBS)
- [JobInfo.Builder.setUserInitiated](https://developer.android.com/reference/android/app/job/JobInfo.Builder#setUserInitiated(boolean))
- [JobService.setNotification](https://developer.android.com/reference/android/app/job/JobService#setNotification(android.app.job.JobParameters,int,android.app.Notification,int))
- [JobParameters.isUserInitiatedJob](https://developer.android.com/reference/android/app/job/JobParameters#isUserInitiatedJob())
- [User-initiated data transfer guide](https://developer.android.com/develop/background-work/background-tasks/uidt)
- [Current AOSP JobInfo.java](https://android.googlesource.com/platform/frameworks/base/+/346fd1aeb8326a1e494ae65d0ed7b67029dbce21/apex/jobscheduler/framework/java/android/app/job/JobInfo.java#2333)
- [Current CTS JobInfoTest.java](https://android.googlesource.com/platform/cts/+/refs/heads/main/tests/JobScheduler/src/android/jobscheduler/cts/JobInfoTest.java#877)
- [Current CTS UserInitiatedJobTest.java](https://android.googlesource.com/platform/cts/+/refs/heads/main/tests/JobScheduler/src/android/jobscheduler/cts/UserInitiatedJobTest.java)

The API-reference wording and current enforcement must therefore be reported
separately: the reference calls out Android 14 by name, while current AOSP and
CTS still require a network and define/test the facility as UIDT.

## Isolated device probe

The temporary debug-only probe declared `RUN_USER_INITIATED_JOBS` and a private
`JobService` protected by `android.permission.BIND_JOB_SERVICE`. Its only
possible work was a 1.5-second CPU arithmetic loop. It referenced no training,
QNN/HTP, network, socket, or transfer API. The probe constructed exactly:

```kotlin
JobInfo.Builder(jobId, probeServiceComponent)
    .setUserInitiated(true)
    .build()
```

The failure result retained the exception class/message and asserted that no
pending job existed. Had construction succeeded, a visible debug Activity
button was the only path to `JobScheduler.schedule()`, and the service would
have called `setNotification()` immediately. Construction did not succeed, so
that path was never entered. The temporary implementation and test were removed
after recording this aggregate result.

Device and build facts:

- Device: nubia NX741J physical device; stable identity checked, identifier and
  ADB endpoint omitted
- OS: Android 16, API 36
- Public fingerprint subset:
  `nubia/PQ85A01-UN/*:16/BQ2A.250705.001-BP2A.250605.031.A3/*:release-keys`
- HexaTrain target SDK: 36
- HexaTrain compile SDK: 36.1
- Repetitions: 3 independent instrumentation invocations
- `JobInfo.build()`: rejected 3/3
- Exception: `java.lang.IllegalArgumentException`
- Message: `A user-initiated data transfer job must specify a valid network type`
- Pending probe job after rejection: absent
- `schedule()`: skipped because build failed
- `onStartJob()` / `isUserInitiatedJob()`: skipped because schedule was forbidden
- Notification / Task Manager / CPU loop: skipped because the service did not start
- Doze probe: skipped because compute-only UIJ did not exist

The nubia firmware result matches current AOSP and CTS; no OEM-specific
exception or target-SDK compatibility behavior was observed.

## Current HexaTrain lifecycle

`StandaloneTrainingActivity` obtains an application-scoped
`StandaloneTrainingRepository`, reconnects subscriptions across Activity
recreation, and dispatches Start/Stop/Pause/Resume away from the UI thread.
Start is a direct GUI action and requests notification permission when needed.

The repository owns one `TrainingSession`, persisted dataset access and
checkpoint selection. Start rejects an already-active session, stop requests
native cancellation, pause/resume are accepted only when the backend supports
them, and a terminal state closes the run notification lifecycle. Native
training, generation, and benchmark entry share `NativeRunArbiter`, preventing
a second process-local JNI owner.

`AndroidTrainingRunLifecycle` starts `LiveUpdateForegroundService`, posts an
ongoing progress notification, and stops the foreground service on completion,
failure, or cancellation. The service is `exported=false`, currently declares
the `dataSync` type, and returns `START_NOT_STICKY`. The GUI training path does
not itself acquire a partial wake lock; `WAKE_LOCK` is declared in the main
manifest, but the explicit four-hour partial wake lock belongs to the separate
instrumentation runner.

`HeadlessDeviceTestRunner` starts no Activity. It uses an application-private
kernel-backed file lock, an atomic status/heartbeat record, a four-hour partial
wake lock released in `finally`, and activity/focus counters. Host runners also
deduplicate ADB aliases by stable device identity and fail closed on active or
uncertain GUI/headless run evidence.

## Execution-model comparison

| Aspect | Foreground Service (current GUI) | User-Initiated Job | Headless ADB runner |
| --- | --- | --- | --- |
| Intended API use | User-noticeable work that must continue while the app is not visible; the declared FGS type must match the work | User-initiated, long-running **network data transfer** | Development/research instrumentation, not a production background API |
| Explicit user initiation | Yes, GUI Start | Required for scheduling | Host/operator command, not GUI initiation |
| Notification | Ongoing run notification | Required via `setNotification()` when runtime UIJ is true | None; private host/status evidence |
| Task Manager | Android 13+ lists apps running FGS and lets the user stop the whole app | Running UIJ is shown; Stop kills the process and blocks app rescheduling of that stopped job | Not represented as GUI training/UIJ; instrumentation is externally controlled |
| Doze behavior | No official guarantee here that FGS prevents OEM deep-Doze HTP slowdown; prior HexaTrain slowdown remains separate evidence | Not tested because compute-only construction is illegal | Partial wake lock is present, but this probe makes no claim that it guarantees HTP throughput in deep Doze |
| Execution quota/limit | Current `dataSync` FGS is limited to a shared 6 hours per 24 hours while backgrounded for target SDK 35+ | Not subject to ordinary quota, but still constrained/stoppable for system health and transfer duration | Controlled by host/runner timeouts and the four-hour wake-lock timeout, not JobScheduler quota |
| User stop semantics | App Stop requests cancellation; Android Task Manager Stop removes the entire app with no callback | Task Manager Stop terminates the process; the stopped job cannot be rescheduled by the app | Host owns cancellation/transport recovery; GUI semantics do not apply |
| Long compute suitability | Plausible user-visible carrier, but current `dataSync` classification and 6-hour limit need a separate architecture review | Unsuitable: network contract rejects pure compute | Suitable only for controlled research, not portable end-user lifecycle |
| Device portability | Android FGS rules plus OEM power policy; deep-Doze behavior needs separate evidence | Standards path rejects compute before OEM differences matter | Depends on ADB, test APK, host scripts, and device preflight |
| HexaTrain GUI fit | Already integrated with Start/Stop/Pause/Resume and checkpoint recovery | No | No; deliberately headless and separated |

Relevant current guidance:

- [Background task API selection](https://developer.android.com/develop/background-work/background-tasks)
- [Long-running WorkManager workers (includes local ML model crunching)](https://developer.android.com/develop/background-work/background-tasks/persistent/how-to/long-running)
- [Foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types)
- [Foreground service timeouts](https://developer.android.com/develop/background-work/services/fgs/timeout)
- [User stopping an FGS app from Task Manager](https://developer.android.com/develop/background-work/services/fgs/handle-user-stopping)
- [Doze and App Standby](https://developer.android.com/training/monitoring-device-state/doze-standby)

## Recommendation

Keep production training off UIJ. Do not add `RUN_USER_INITIATED_JOBS`, a UIJ
`JobService`, `setUserInitiated(true)`, a fake network constraint, estimated
network bytes, or dummy traffic to the production lifecycle.

Review the existing GUI execution carrier separately. In particular, pure HTP
training is not described by the current `dataSync` examples, and that type now
has a six-hour/24-hour background limit. A separate design task should compare
the current FGS, a long-running WorkManager worker (whose official examples
include local ML model crunching), and a properly declared `specialUse` FGS,
including Play review implications, timeout handling, checkpoint recovery, and
nubia power behavior. This note does not select or implement that migration.

The research-only ADB Doze-control proposal also remains separate: capture the
original device-idle/stayon state, apply only a development control, run the
experiment, and restore the original state in `finally`. No such control or
Doze mutation was performed in this UIJ probe.
