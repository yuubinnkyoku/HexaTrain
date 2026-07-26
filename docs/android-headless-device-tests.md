# Android headless device tests

PhoneLM device tests can run through `AndroidJUnitRunner` without starting
`MainActivity`. The runner initializes the application QNN environment directly,
acquires a process-wide file lock, holds a partial wake lock, updates a fixed
4,096-byte status record atomically, and writes the detailed report only after a
suite finishes. The existing Activity UI remains available for manual use but is
not part of the automated path.

Run a suite from PowerShell:

```powershell
.\scripts\run_qnn_headless_tests.ps1 `
  -QairtSdkRoot <QAIRT-2.48-root> `
  -Suite device-probe `
  -TestMode BACKGROUND_CORRECTNESS
```

The script requires exactly one online ADB device. It builds and audits both
APKs, installs them, invokes `adb shell am instrument`, samples the top resumed
package every two seconds, and pulls only the private status and report into the
ignored build report directory. Device endpoints are never written to public
results. `EXCLUSIVE_BENCHMARK` is reserved for controlled performance runs;
normal use should select `BACKGROUND_CORRECTNESS`.

## Safety and recovery

The application lifecycle counters must remain zero and host sampling must show
zero focus takeovers. A second lock attempt returns `ALREADY_RUNNING`. The lock
is a kernel-backed `FileChannel` lock, so process death releases it; a later run
atomically replaces stale status. A 30-second heartbeat and `finally` cleanup
cover long tests. The partial wake lock has a four-hour timeout and is always
released.

Failure recovery was verified after an invalid suite and after force-stopping a
running instrumentation process. Skel recovery was verified by corrupting only
the app-private deployed copy; the next headless probe restored it from the
audited APK asset and reported `qnn_skel_action=replaced`. No vendor, system,
SDK, or APK asset file is modified.

## Suites

The runner maps the device probe; QNN forward; linear; MLP split, fused, and
full-step; Transformer forward and backward components; tiny-LM CE; SGD,
Momentum, and Adam; API trace; callback bound; fixed-state reproducibility; and
the longer stability/generation diagnostics. Every invocation is one suite and
one report. Multi-phase research work is split across separate invocations so
the foreground remains available and recovery is unambiguous.

The status schema includes the schema version, run ID, suite, state, PID,
start/heartbeat timestamps, phase, current/completed/total tests, result,
failure code, and a relative report identifier. Status writes use a temporary
file, `fsync`, and atomic replacement where the filesystem supports it.
