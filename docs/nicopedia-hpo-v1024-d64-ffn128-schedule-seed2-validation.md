# Nicopedia V1024 / D64 / FFN128 schedule seed2 validation

## Verdict

**SEED2 VALIDATION COMPLETE — Strong reproduction.**

The frozen S4000 linear-decay recipe improved both held-out splits at seed 2.
Its balanced improvement over C1500 was `-0.069102772` bits per original UTF-8
byte (bpb), compared with `-0.068351102` bpb at seed 1. This is a heuristic
reproduction classification, not a significance test; only two seeds are
available.

The untouched final/test split was not opened or evaluated. No additional HPO,
checkpoint selection, late-checkpoint selection, commit, or push was performed.

## Purpose

This run checks whether the schedule gain selected at seed 1 reproduces under a
paired seed-2 initialization and data order. It compares only the two frozen
recipes and evaluates both exact step-8000 checkpoints once on the same 256
validation plus 256 development chunks.

## Frozen recipe

- Model: V1024, T32, D64, FFN128, L19, H2; 758,528 parameters.
- Batch: 8 samples per update.
- Seed: 2.
- Training target: 8,000 steps per arm, from the seed-2 initialization.
- C1500 baseline: constant learning rate `0.0015`.
- S4000 schedule: learning rate `0.0022` through step 4,000, then linear decay
  to `0.0001` at step 8,000.
- Runtime: QNN/HTP with QAIRT `2.48.40.260702`, build ID
  `2.48.40.260702151143`; no CPU fallback.
- Execution order: C1500 followed by S4000 on one physical device, with no
  overlapping training.

## Why paired seed2

Changing only the recipe while holding initialization, batch order, tokenizer,
cache, model identity, and total exposure fixed makes the within-seed delta the
relevant comparison. Seed 2 is an independent repeat of the seed-1 recipe pair,
not a new tuning round.

## Seed2 data and initialization identity

The C1500 and S4000 reports agree on every paired identity field:

| Field | C1500-seed2 | S4000-seed2 |
| --- | --- | --- |
| Initial parameter hash | `fnv1a64:9e14cc657a220637` | `fnv1a64:9e14cc657a220637` |
| Full training-order hash | `fnv1a64:37fe7bac20c91642` | `fnv1a64:37fe7bac20c91642` |
| Cache-content hash | `fnv1a64:0c7b2826f5f26fea` | `fnv1a64:0c7b2826f5f26fea` |
| Tokenizer hash | `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798` | same |
| Target tokens seen | 2,048,000 | 2,048,000 |
| Target UTF-8 bytes seen | 5,491,256 | 5,491,256 |
| Unique chunks seen | 46,616 | 46,616 |
| Unique articles seen | 1,949 | 1,949 |

The independent four-update smoke runs also matched initialization, the
four-update order hash `fnv1a64:0e2e15196d851431`, cache, and tokenizer. Both
smokes completed all four updates with QNN success, finite tensors, zero graph
failures, and no fallback. Their short-run quality status was not used as a
loss-improvement gate.

## C1500 result

The fresh 0-to-8,000 run completed with 32 interval checkpoints. Every pulled
NPRTCKPTV3 checkpoint passed architecture/tokenizer identity checks and a host
decode finiteness check. The exact step-8,000 checkpoint SHA-256 was
`7444019ca986d328ad28e1b4eae0567accd1ddfb7ae1d1fb0bab28862983642e` and its
parameter hash was `fnv1a64:cda26ed8680db2b0`.

| Metric | Value |
| --- | ---: |
| Validation bpb | 2.337929461 |
| Development bpb | 2.612091100 |
| Balanced bpb | 2.475010281 |
| Validation/development chunks | 256 / 256 |
| QNN return-code success | true |
| Output tensors finite | true |
| Graph-execute failures | 0 |
| CPU fallback | false |

## S4000 result

The original fresh 0-to-8,000 HTP phase wrote all 32 interval checkpoints,
including exact step 8,000. Its host wrapper was lost during the subsequent
fresh-run CPU replay. The device process later disappeared with a stale
`RUNNING/cpu_replay` status, so this was classified as a harness interruption,
not as a QNN, HTP, OOM, or numerical failure.

Recovery followed the frozen overnight-recovery rule:

1. All 32 original checkpoints were pulled and independently verified for
   model/tokenizer identity and finite decoded parameters.
2. The original telemetry was complete through step 7,940; it reached the
   device file's exact 256-KiB cap because scheduled-LR rows are longer than
   constant-LR rows.
3. The highest resumable verified checkpoint, step 7,750, was resumed with the
   same seed, schedule, optimizer, cache, tokenizer, and data-order identity.
4. The recovery completed steps 7,751 through 8,000 with QNN success, finite
   tensors, zero graph failures, and no fallback.
5. The recovered step-8,000 checkpoint was bitwise identical to the original:
   SHA-256
   `fd517e4340415c7b804eddbb787e944c76d30921bb8864a84b9d8497a27e1bd1`.
6. Original telemetry through step 7,940 plus recovery telemetry through step
   8,000 formed a gap-free 1-to-8,000 sequence and matched the frozen LR formula
   at every step.

The terminal recovery report retained the cumulative full-run exposure and
full-order identities shown above. Its final parameter hash was
`fnv1a64:c145c8728713d36a`.

| Metric | Value |
| --- | ---: |
| Validation bpb | 2.280231091 |
| Development bpb | 2.531583926 |
| Balanced bpb | 2.405907509 |
| Validation/development chunks | 256 / 256 |
| QNN return-code success | true |
| Output tensors finite | true |
| Graph-execute failures | 0 |
| CPU fallback | false |

## Seed1 reference

The seed-1 values were re-read from the existing exact-step-8,000, 256+256
source reports rather than copied from a summary:

| Arm | Validation bpb | Development bpb | Balanced bpb |
| --- | ---: | ---: | ---: |
| C1500-seed1 | 2.338413313 | 2.624521011 | 2.481467162 |
| S4000-seed1 | 2.284370268 | 2.541861852 | 2.413116060 |

Seed-1 deltas (`S4000 - C1500`) are `-0.054043045` validation,
`-0.082659159` development, and `-0.068351102` balanced bpb.

## Cross-seed deltas

| Seed | Constant Val | Constant Dev | Constant Bal | Schedule Val | Schedule Dev | Schedule Bal | Delta Bal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.338413313 | 2.624521011 | 2.481467162 | 2.284370268 | 2.541861852 | 2.413116060 | -0.068351102 |
| 2 | 2.337929461 | 2.612091100 | 2.475010281 | 2.280231091 | 2.531583926 | 2.405907509 | -0.069102772 |

For seed 2, `Delta Val = -0.057698370`, `Delta Dev = -0.080507174`, and
`Delta Balanced = -0.069102772` bpb. Both splits improved. The mean balanced
delta across the two seeds is `-0.068726937` bpb. With only `n=2`, this mean is
descriptive and is not a variance or significance estimate.

## Reproduction classification

Seed 2 meets the predefined **Strong reproduction** rule: balanced delta is at
most `-0.03` bpb and both validation and development improve. The gain-retention
ratio is `abs(-0.069102772) / 0.068351102 = 1.010997`, or about **101.10%** of
the seed-1 gain. Gain retention is a descriptive heuristic, not a formal
statistic.

## Compute

| Work | HTP training updates |
| --- | ---: |
| C1500 primary | 8,000 |
| S4000 primary | 8,000 |
| Two successful four-update smokes | 8 |
| S4000 recovery replay (7,750 to 8,000) | 250 |
| Failed C1500 argument preflight | 0 |
| Total executed HTP updates | 16,258 |

The requested primary budget remained 16,000 new updates. Retry/recovery
overhead was 258 HTP updates (8 smoke plus 250 recovery); the rejected C1500
argument attempt reached no native update.

- C1500 HTP training: 33,813.868 s, 4,226.734 ms/update, 162.397 target
  UTF-8 bytes/s. The complete fresh run including CPU replay took 81,286 s
  (22 h 34 m 46 s).
- S4000 original HTP phase: approximately 45,885.327 s from device run start to
  the step-8,000 checkpoint mtime, 5,735.666 ms/update and 119.673 target
  UTF-8 bytes/s. The last live heartbeat was 78,856.471 s after start, during
  the interrupted CPU replay.
- S4000 recovery: 852.444 s for 250 updates, 3,409.778 ms/update and 200.752
  target UTF-8 bytes/s.
- The observed end-to-end span from the first successful C1500 smoke through
  the final S4000 evaluation was 47 h 11 m 12 s, including monitoring,
  reconnect, artifact recovery, smoke, and evaluation time.

These throughput values are systems evidence only. Quality metrics were not
adjusted for runtime or temperature.

## Systems caveats

- Android thermal status remained `0` throughout observed training, replay,
  recovery, and evaluation samples. Battery temperature ranged from 36 to
  44 degrees C; no emergency/shutdown condition occurred.
- The C1500 host PTY disappeared while the device run remained live. Device
  heartbeat/status was used until terminal `PASSED`, then all artifacts were
  recovered and verified.
- The S4000 host PTY disappeared after exact step 8,000 was written and while
  CPU replay was active. The later stale status had no live process. This is a
  harness interruption; no ADB transport interruption or runner terminal alone
  was counted as a model failure.
- The S4000 original and recovery step-8,000 checkpoint hashes match exactly,
  providing a stronger trajectory-continuity check than filename or step
  metadata alone.
- Fresh C1500 performed the runner's CPU replay. S4000's original CPU replay did
  not finish, while its resumed recovery correctly reported
  `cpu_replay_performed=false`. The paired quality evaluation uses the verified
  HTP step-8,000 checkpoints and does not claim CPU/HTP numerical parity.
- QNN success and tensor finiteness were checked as separate conditions. No
  claim is made that training was NPU-only, that the CPU was unused, or that
  QNN performed automatic differentiation.

## Verification

- `verify_local.ps1 -Fast -SkipAndroidBuild`: PASS (diff/secret audits,
  PowerShell parser, QAIRT-selection self-test, and JVM unit tests; 0 failures).
- Training runner, evaluation runner, and schedule-v2b wrapper self-tests: PASS.
- Existing Nicopedia learning-rate schedule host-test executable: PASS.
- Fixed-QAIRT QNN build: PASS.
- APK audit: PASS for ABI, hashes, fixed path/build identity, and absence of
  QAIRT 2.47 mixing.
- C1500/S4000 four-update HTP smokes: PASS for QNN health and identity.
- C1500 32/32 checkpoint identity and host finiteness checks: PASS.
- S4000 original 32/32 checkpoint identity and host finiteness checks: PASS.
- S4000 recovery checkpoint continuity: PASS; original and recovered step-8,000
  SHA-256 values match.
- C1500 constant telemetry: 8,000/8,000 rows and required anchors PASS.
- S4000 merged telemetry: continuous steps 1 through 8,000 and per-step frozen
  LR formula PASS.
- Exact step-8,000 256+256 HTP evaluations: PASS for both arms, with QNN
  success, finite tensors, zero graph failures, and no fallback.
- Full verification gate: intentionally not run, as required for this task.

## Files changed

This task adds only this aggregate documentation file. Raw checkpoints,
private caches, instrumentation output, device identity, and other private
evidence remain ignored under `build/` and are not committed.

## Final split status

**UNTOUCHED.** The unused final/test split was not evaluated, selected on, or
included in training.

## Next step

Freeze the complete S4000 recipe exactly as validated. After review in a
separate task, evaluate the untouched final split once for final generalization.
Do not start another HPO round as part of this validation.
