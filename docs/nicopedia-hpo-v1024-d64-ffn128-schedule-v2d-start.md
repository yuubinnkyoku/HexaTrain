# Nicopedia HPO: Schedule-v2d decay start

Status: **Schedule-v2d COMPLETE** (seed 1, one direct full-tail candidate,
exact step-8000 256+256 held-out evaluation).

This task changed only the linear decay start. Peak LR, target LR, decay end,
model, data order, optimizer, tokenizer, and seed were held fixed.

## Motivation

Schedule-v2b/v2c established the current target `.0001` linear recipe around
`start=4000`. The existing S4000 and S6000 results differed by only
`0.002039181` Balanced bpb, so the next question was whether starting earlier
at step 3000 is useful.

## Why no cheap proxy

Schedule-v2c showed that a proxy/full-tail ranking can invert for small shape
differences. S3000 was therefore evaluated directly through step 8000. No
64+64 or 128+128 selection, tail proxy, intermediate ranking, or pruning was
used.

## Fixed configuration

* `V=1024`, `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; 758,528 parameters
* batch `8`, seed `1`, byte-BPE tokenizer
* tokenizer SHA-256
  `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* dataset cache `fnv1a64:0c7b2826f5f26fea`; fixed canonical order hash
  `fnv1a64:0e2e15196d851431`
* Adam `beta1=.9`, `beta2=.999`, `eps=1e-8`; gradient clip disabled; weight
  decay `0`
* peak LR `.0022`, target LR `.0001`, linear decay, end `8000`
* QAIRT build `2.48.40.260702151143`; QNN HTP backend
* step-8000 exposure: 2,048,000 target tokens, 5,491,256 original UTF-8
  bytes, 46,616 chunks, 1,949 articles

## Parent and explicit fork

The exact validated C2200 constant-LR parent was reused:

* `parent_trial_id=hpo-lr-v1-lr0p0022-seed1`
* step `3000`, LR `.0022`
* checkpoint SHA-256
  `8fb46059e785b371480f1e15c8ab078b055982506916217972355cb5e05d2769`
* parameter hash `fnv1a64:78ccaf543c969f6e`
* `NPRTCKPTV3`, model/tokenizer/data identity, Adam state (93 chunks), and
  finite state validated
* parent QNN return code succeeded, output tensors were finite, graph failures
  were `0`, and fallback was `false`

The S3000 manifest records the trial ID, parent trial ID, parent checkpoint
hash, `parent_step=3000`, peak/target LR, `schedule_type=linear`, decay
start/end, fixed model/tokenizer/data/order/optimizer identity, and
`experiment_fork=true`. No cross-schedule resume was used. Parent creation
overhead was zero.

## Schedule

The existing linear schedule infrastructure was used with 1-indexed global
steps:

```text
step <= 3000:
  lr = .0022

3000 < step <= 8000:
  p = (step - 3000) / 5000
  lr = .0022 + p * (.0001 - .0022)
```

Thus step 3000 is `.0022` and step 8000 is `.0001`.

The 4-update smoke from the step-3000 parent passed with finite tensors,
QNN success, zero graph failures, no CPU fallback, and the expected fork and
schedule identity. It was a health check only; it was not used for ranking.

## Runtime LR telemetry

The step-3000 value is the validated parent boundary; the remaining values
are runtime telemetry. All absolute formula differences were below `6.7e-11`,
well inside the fail-closed `2e-8` tolerance.

| Step | Expected LR | Actual LR |
| ---: | ---: | ---: |
| 3000 | 0.002200000000 | 0.002199999988 |
| 3500 | 0.001990000000 | 0.001990000019 |
| 4000 | 0.001780000000 | 0.001779999933 |
| 4500 | 0.001570000000 | 0.001569999964 |
| 5000 | 0.001360000000 | 0.001359999995 |
| 5500 | 0.001150000000 | 0.001150000026 |
| 6000 | 0.000940000000 | 0.000939999998 |
| 6500 | 0.000730000000 | 0.000729999971 |
| 7000 | 0.000520000000 | 0.000520000001 |
| 7500 | 0.000310000000 | 0.000310000003 |
| 8000 | 0.000100000000 | 0.000099999997 |

## Step-8000 primary result

The final checkpoint was step 8000, SHA-256
`52ef4c460ddaefd21020c98974b154e9eb81d51ae23652f3ea7c373b86632a5f`, with
parameter hash `fnv1a64:1360e10593972b3d`. The HTP evaluation used exactly
256 validation and 256 development chunks and the original UTF-8 byte metric.

* Val256: `2.283554460` bpb
* Dev256: `2.539764759` bpb
* Balanced: `2.4116596095` bpb

The evaluation report independently records QNN return-code success, finite
output tensors, zero graph execute failures, and `cpu_fallback=false`.

## Primary comparison

All rows are exact step-8000 / 256+256 evaluations. Negative deltas are
improvements relative to S4000.

| Start | Val256 | Dev256 | Balanced | Delta vs S4000 |
| ---: | ---: | ---: | ---: | ---: |
| 3000 | 2.283554460 | 2.539764759 | 2.411659610 | -0.001456451 |
| 4000 | 2.284370268 | 2.541861852 | 2.413116060 | 0 |
| 6000 | 2.286752948 | 2.543557534 | 2.415155241 | +0.002039181 |

For S3000 versus S4000, `Delta Val=-0.000815808`,
`Delta Dev=-0.002097093`, and `Delta Balanced=-0.001456451`.

## Start sensitivity verdict

S3000 improves both splits, but the Balanced improvement is only
`0.001456451` bpb, below the required `0.005` threshold. With one seed this
is not a statistical-significance claim. The observed numeric best is S3000,
but the stopping rule keeps the S4000 recipe as the operational anchor:
start sensitivity is **flat/noise-sensitive**, and the start axis is stopped.

Do not run start 2000, 3500, 4500, 5000, or a finer grid in this task.

## Compute accounting

| Quantity | Value |
| --- | ---: |
| naive fresh steps | 8,000 |
| actual new steps | 5,000 |
| reused parent-prefix steps | 3,000 |
| parent-creation overhead | 0 |
| saved steps | 3,000 |
| saving | 37.5% |

The smoke contributed 4 updates and the full tail resumed at step 3004 for
4,996 updates; these are counted once as 5,000 new updates. The full-tail
native training time was `18,584.369 s` (`3,719.849583 ms/update` and
`184.424399` run bytes/s). The 4-update smoke took `49.554 s` native time;
it is recorded separately from the tail wall time. The exact 256+256 HTP
evaluation took `76.702 s`.

## Systems and safety caveats

Training and evaluation independently reported QNN return-code success and
finite tensors. Training had graph execute failures `0`,
`cpu_fallback=false`, and no fallback attempt; evaluation had the same health
result. Thermal status stayed `0` (`0 -> 0`) and the full-tail battery
temperature was `37 -> 37` degrees C. Runtime and thermal observations are
system measurements, not schedule-quality evidence.

The run was single-flight on one physically identified device in background
correctness mode; no focus takeover occurred. The fixed-QAIRT QNN APK audit
passed. “HTP” here means the training-step numerical operations were executed
through the QNN HTP backend; this does not claim NPU-only execution, no CPU
use, or QNN automatic differentiation.

## Verification

* initial workspace safety checks (`git status`, recent log, `git diff --check`):
  PASS; pre-existing dirty changes were preserved
* schedule host tests and training runner self-test: PASS
* fixed-QAIRT QNN build/install and APK audit: PASS
* v2d PowerShell parser, self-test, Plan, and artifact-reuse Run: PASS
* parent/checkpoint identity and finite host probes: PASS
* 4-update smoke, direct full tail to step 8000, telemetry, and exact 256+256
  HTP evaluation: PASS
* `verify_local.ps1 -SkipAndroidBuild`: see final task report

The Full gate, UI validation, generation, commit, and push were intentionally
not run for this task.

## Next step

Because S3000 did not meet the `0.005` improvement threshold, keep
`start=4000`, freeze this schedule-start axis, and use a later task to
reproduce the best schedule versus the constant baseline with seed 2. Do not
add cosine, sqrt, fine start grids, or architecture changes yet.
