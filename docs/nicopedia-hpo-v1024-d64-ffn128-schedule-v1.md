# Nicopedia HPO: V1024 / T32 / D64 / FFN128 / L19 / H2, Schedule-v1

Status: COMPLETE for the declared Schedule-v1 sweep (seed 1). This is a
single-seed frontier observation, not a statistical significance claim.

## Motivation and hypothesis

LR-v1/v1b found that LR `0.0022` was the strongest observed early (step 4000)
point, while LR `0.0015` was the strongest exact step-8000 constant-LR point.
The constant-LR final frontier from `0.0015` through `0.0042` was relatively
flat, and `0.00105` or lower was clearly worse. Schedule-v1 tests whether
keeping the early LR at `0.0022` and decaying only the late segment to `0.0015`
improves the exact-budget constant baselines.

Only decay start is varied. No architecture, optimizer, peak LR, target LR,
seed, or evaluation budget is changed.

## Fixed configuration

* vocabulary `V=1024`; byte-BPE tokenizer SHA-256
  `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; 758,528 parameters
* batch 8; seed 1; Adam (`beta1=.9`, `beta2=.999`, `eps=1e-8`)
* gradient clipping disabled; weight decay 0
* dataset cache `fnv1a64:0c7b2826f5f26fea`; canonical order identity
  `fnv1a64:0e2e15196d851431`
* QAIRT build `2.48.40.260702151143`; QNN HTP backend

At step 8000 every candidate must account for 2,048,000 target tokens,
5,491,256 original UTF-8 bytes, 46,616 chunks, and 1,949 articles. The primary
evaluation is the exact step-8000 checkpoint with 256 validation and 256
development chunks, measured in original UTF-8 bytes.

## Schedule definitions

The optimizer step is global and 1-indexed. Both endpoints are inclusive:

```text
if step <= decay_start:
    lr = 0.0022
elif step >= 8000:
    lr = 0.0015
else:
    progress = (step - decay_start) / (8000 - decay_start)
    lr = 0.0022 + progress * (0.0015 - 0.0022)
```

* `C1500`: constant `0.0015` (reused baseline)
* `C2200`: constant `0.0022` (reused baseline)
* `S4000`: decay start 4000; linear segment 4000–8000
* `S6000`: decay start 6000; linear segment 6000–8000

The child run receives the actual per-step LR. A runtime telemetry CSV is
validated against this formula at every step and at the declared anchor steps
4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, and 8000. Parent-only anchors
are marked as inherited from the validated constant-LR prefix.

## Explicit fork and parent reuse

Schedule-v1 is an explicit experiment fork, not a relaxation of normal resume.
Normal resume still requires the same trial and hyperparameters. Each child
manifest records the parent trial, checkpoint hash, parent/fork step, parent LR,
schedule, decay bounds, peak/target LR, model/tokenizer/data/order identity,
and Adam identity. Adam first and second moments and the global step counter
are inherited; they are never reset.

* `S4000` reuses the validated constant-`0.0022` exact step-4000 checkpoint.
* `S6000` reuses the validated constant-`0.0022` exact step-6000 checkpoint.

Checkpoint paths, device identity, raw telemetry, and checkpoint hashes remain
private under `build/` and are not exported here.

## Training and systems evidence

Both candidates run serially on one physically identified device. The first
four updates after each fork are an HTP smoke: QNN return-code success,
finite tensors, zero graph failures, no CPU fallback, restored optimizer
state, and identity/telemetry checks are required. Healthy candidates continue
to step 8000 without early pruning.

The runner records wall time/update, bytes/s, QNN health, thermal state, and
fallback status. An ADB transport interruption is infrastructure evidence, not
evidence of QNN or numerical failure; recovery requires reconnect, state check,
and checkpoint/header/hash validation without force-stop or data deletion.

Both training runs reported Android thermal status `0` before and after, healthy
battery state, and no thermal abort. The private ledger records battery
temperature changes of 35→38 °C for S4000 and 37→41 °C for S6000; these are
device-system observations, not schedule-quality evidence. Run-segment
throughput was approximately 198.740 original-UTF-8 bytes/s for S4000 and
224.479 bytes/s for S6000; this is a device-performance metric, not a quality
ranking signal.

## Primary comparison

The authoritative values are the private Schedule-v1 `summary.csv` and
validated manifests. Values are single-seed (`seed=1`) observations, not
statistically significant claims.

| Schedule | decay start | Val256 | Dev256 | Balanced |
| --- | ---: | ---: | ---: | ---: |
| constant 0.0015 | — | 2.338413313 | 2.624521011 | 2.481467162 |
| constant 0.0022 | — | 2.349580766 | 2.620467215 | 2.485023991 |
| linear S4000 | 4000 | 2.330467873 | 2.595317375 | **2.462892624** |
| linear S6000 | 6000 | **2.330305757** | 2.600519830 | 2.465412794 |

The schedule deltas versus C1500 are:

| Schedule | ΔVal | ΔDev | ΔBalanced |
| --- | ---: | ---: | ---: |
| S4000 | -0.007945440 | -0.029203636 | -0.018574538 |
| S6000 | -0.008107556 | -0.024001181 | -0.016054369 |

Versus C2200, S4000 is `-0.019112893`, `-0.025149840`, and `-0.022131367`
for Val, Dev, and Balanced; S6000 is `-0.019275009`, `-0.019947385`, and
`-0.019611197`. Both schedules improve both splits against both constants;
S4000 is best by Balanced and Dev, while S6000 has the lowest Val by
`0.000162116` bpb. Balanced is therefore not hiding a split reversal here.

## Compute accounting

The naive two-fresh-run cost is 16,000 steps. Actual new training was 4,000
steps for S4000 plus 2,000 for S6000 (6,000 total), reusing 4,000 and 6,000
prefix steps respectively. Thus 10,000 steps were reused, 10,000 steps were
saved, and the saving was 62.5%. Native training time was 13,803.861 s for
S4000 and 6,101.516 s for S6000; exact 256+256 HTP evaluation added 73.313 s
and 64.505 s respectively. Host collection/recovery overhead is excluded
from this native-device sum and is recorded separately in the private ledger.

## LR telemetry and intermediate diagnostics

The runtime telemetry files contain 4,000 rows for S4000 (steps 4001–8000)
and 2,000 rows for S6000 (steps 6001–8000). Aggregate anchor validation,
including the inherited parent endpoint(s), was:

| Step | S4000 LR | S6000 LR |
| ---: | ---: | ---: |
| 4000 | 0.002200000 | 0.002200000 |
| 4500 | 0.002112500 | 0.002200000 |
| 5000 | 0.002025000 | 0.002200000 |
| 5500 | 0.001937500 | 0.002200000 |
| 6000 | 0.001850000 | 0.002200000 |
| 6500 | 0.001762500 | 0.002025000 |
| 7000 | 0.001675000 | 0.001850000 |
| 7500 | 0.001587500 | 0.001675000 |
| 8000 | 0.001500000 | 0.001500000 |

Actual floating-point telemetry differed from these decimal expectations by
less than the runner's `2e-8` tolerance at every checked anchor. The parent
checkpoint endpoint was recorded as inherited constant-LR telemetry, not
fabricated child telemetry.

No early-pruning diagnostic was used for ranking. Both candidates reached the
exact step-8000 primary budget and passed QNN return-code, finite-tensor, and
`cpu_fallback=false` gates.

## Systems caveat and recovery

The first S4000 final host collection encountered an ADB transport/collection
target collision because the smoke and final invocations used the same
telemetry filename. The device had already reported `SUCCESS` at step 8000;
all checkpoint headers, hashes, finite state, QNN health, exposure totals,
and the terminal report were revalidated, and the final telemetry was
recovered without retraining. The runner now preserves prior segment telemetry
before pulling a new segment and reuses a completed trial instead of starting
it again. This incident is infrastructure evidence, not a QNN or numerical
failure. No force-stop, data deletion, parallel trial, or SDK fallback was
used.

## Infrastructure and verification

The implementation is limited to an explicit schedule configuration, exact
per-step LR application, fork-only manifest semantics, runtime LR telemetry,
and the telemetry target-collision recovery fix. Schedule formula/boundary
host tests, runner self-tests, the host test battery, Kotlin compilation, the
fixed-QAIRT QNN build, and APK ABI/hash/path/2.47 audits passed. The requested
Full gate was not run; no commit or push was made.

## Verdict and next recommendation

**Schedule-v1 COMPLETE.** Decay is useful on this seed-1 exact-budget test:
both schedules improve C1500 by `0.016054–0.018575` Balanced bpb and improve
both Val and Dev. The best decay start by Balanced/Dev is **4000**; S6000 is
slightly better on Val alone. This is strong directional evidence, but not a
significance claim.

Recommended next steps (maximum two):

1. Confirm S4000 against seed 2 before expanding schedule shape or target.
2. If confirmed, run Schedule-v2 with exactly one additional axis (shape or
   target), keeping the fixed model and optimizer.

Architecture changes remain out of scope.
