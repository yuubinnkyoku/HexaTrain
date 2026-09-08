# Nicopedia HPO: Schedule-v2c cooldown shape

Status: **Schedule-v2c COMPLETE** (seed 1, one selected shape, exact step-8000
256+256 held-out evaluations).

This task changed only the cooldown shape. Peak LR, target LR, decay bounds,
model, data order, optimizer, and seed were held fixed. The candidate was
`Sqrt6000`; the existing `Linear6000` artifact remained the proxy anchor.

## Motivation

Schedule-v2b identified linear decay with `target=.0001` as the current best.
The next low-cost question was whether a more front-loaded cooldown improves
the same fixed tail. The single new shape was tested before considering cosine;
this keeps the shape axis one-dimensional and avoids adding another candidate
after a single-seed result.

## Why sqrt before cosine

The tested sqrt shape is explicit and easy to audit:

```text
p = (step - decay_start) / (decay_end - decay_start)
shape = 1 - sqrt(p)
lr = target + (peak - target) * shape
```

This gives a faster initial drop than linear while preserving the exact
non-zero target endpoint. Cosine was intentionally not added: if this shape
axis is flat or loses on the full tail, changing decay start has higher
information value than widening the shape grid.

## Fixed configuration

* `V=1024`, `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; 758,528 parameters
* batch `8`, seed `1`, byte-BPE tokenizer
* tokenizer SHA-256 `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* dataset cache `fnv1a64:0c7b2826f5f26fea`; canonical order hash
  `fnv1a64:0e2e15196d851431`
* Adam `beta1=.9`, `beta2=.999`, `eps=1e-8`; gradient clip disabled; weight
  decay `0`
* QAIRT build `2.48.40.260702151143`; QNN HTP backend
* step-8000 exposure: 2,048,000 target tokens, 5,491,256 original UTF-8
  bytes, 46,616 chunks, 1,949 articles

## Common parent and explicit fork

The proxy used the validated C2200 constant-LR checkpoint at step 6000:

* step `6000`, prefix LR `.0022`
* checkpoint SHA-256
  `ee8c122612b832c89e7c6a3c3d360c76f38a28c06f260f1b0a40d969dc309063`
* parameter hash `fnv1a64:bde59040980ad8ce`
* `NPRTCKPTV3`, model/data/order/tokenizer identity, Adam state, and finite
  state revalidated; QNN success, graph failures `0`, fallback `false`

`Sqrt6000` was an explicit fork with `parent_trial_id`, parent checkpoint hash,
`parent_step=6000`, `schedule_type=sqrt`, `runner_schedule=sqrt_decay`, and
the complete fixed configuration in its manifest. Cross-schedule resume was
rejected by identity validation. The full-tail confirmation used the separately
validated C2200 step-4000 parent and the same sqrt definition over 4000 steps.

## Schedule definitions

The existing proxy anchor was:

```text
Linear6000: peak=.0022, start=6000, end=8000, target=.0001
shape = 1 - p
```

The new proxy was:

```text
Sqrt6000: peak=.0022, start=6000, end=8000, target=.0001
shape = 1 - sqrt(p)
```

Both schedules are held at the peak through the start boundary and meet
`lr=.0001` exactly at step 8000. The host formula test also covers step 6001,
step 7999, midpoint behavior, and non-zero-target scaling.

## Runtime LR telemetry

The actual runtime values matched the host formula within the fail-closed
`2e-8` tolerance:

| Step | Linear expected | Sqrt expected | Sqrt actual | Actual - expected |
| ---: | ---: | ---: | ---: | ---: |
| 6000 | 0.002200000000 | 0.002200000000 | 0.002200000000 | 0 |
| 6250 | 0.001937500000 | 0.001457537880 | 0.001457537874 | -5.75e-12 |
| 6500 | 0.001675000000 | 0.001150000000 | 0.001150000026 | 2.60e-11 |
| 6750 | 0.001412500000 | 0.000914017885 | 0.000914017903 | 1.77e-11 |
| 7000 | 0.001150000000 | 0.000715075760 | 0.000715075759 | -3.08e-13 |
| 7250 | 0.000887500000 | 0.000539804228 | 0.000539804227 | -1.71e-12 |
| 7500 | 0.000625000000 | 0.000381346652 | 0.000381346647 | -5.55e-12 |
| 7750 | 0.000362500000 | 0.000235629872 | 0.000235629865 | -7.34e-12 |
| 8000 | 0.000100000000 | 0.000100000000 | 0.000099999997 | -2.53e-12 |

The 4-update Sqrt6000 smoke and 4-update Sqrt4000 smoke both passed before
their respective tails.

## Proxy result

Negative deltas are improvements relative to the existing Linear6000 artifact.

| Shape | Val256 | Dev256 | Balanced | Delta vs linear |
| --- | ---: | ---: | ---: | ---: |
| linear | 2.286752948 | 2.543557534 | 2.415155241 | 0 |
| sqrt | 2.285014287 | 2.542925482 | 2.413969885 | -0.001185356 |

The proxy improved both Val and Dev, so the requested threshold-A rule allowed
one full-tail confirmation even though the Balanced improvement was below
0.005 bpb.

## Full-tail confirmation

The Sqrt4000 tail was run from the validated C2200 step-4000 parent:

| Shape | Val256 | Dev256 | Balanced |
| --- | ---: | ---: | ---: |
| linear | 2.284370268 | 2.541861852 | 2.413116060 |
| sqrt | 2.288342631 | 2.547527249 | 2.417934940 |

Sqrt4000 minus Linear4000 was `+0.003972363` Val, `+0.005665397` Dev, and
`+0.004818880` Balanced. The full-tail result is therefore worse. This is one
seed and is not a statistical-significance claim.

## Shape verdict

The proxy signal was positive, but it did not survive the full-tail check.
The final primary verdict is **linear維持; sqrt shape worse**. Shape exploration
ends here; cosine is not automatically added.

## Compute accounting

| Quantity | Proxy | Including full-tail |
| --- | ---: | ---: |
| naive fresh steps | 8,000 | 16,000 |
| actual new steps | 2,000 | 6,000 |
| reused parent-prefix steps | 6,000 | 10,000 |
| saved steps | 6,000 | 10,000 |
| saving | 75% | 62.5% |

The two measured training runner wall times sum to approximately 26,883.7 s
(about 7 h 28 min), including checkpoint transfer and runner overhead. The
recorded native training totals sum to about 26,659.9 s. Smoke probes are
tracked separately and are not counted as tail steps.

## Systems and safety caveats

Both terminal tails independently report QNN return-code success, finite
outputs at every step, graph execute failures `0`, and `cpu_fallback=false`.
Fallback was neither attempted nor succeeded. Thermal status stayed `0`; the
observed battery temperature ranges were 42→39°C for Sqrt6000 and 39→37°C for
Sqrt4000. Runtime/thermal differences are recorded as system observations,
not attributed to cooldown shape. The fixed QAIRT build and QNN APK audit
passed; an optional SDK inventory advisory did not affect required artifacts.

Execution was single-flight on one physically identified device. No parallel
trial, broad force-stop, app-data deletion, endpoint/serial disclosure, or
cross-schedule resume was used. Private checkpoints and raw device evidence
remain under ignored `build/` paths.

## Verification

* schedule host formula test and `run_host_tests.ps1`: PASS
* training runner self-test and v2c orchestrator self-test: PASS
* PowerShell parser and `git diff --check`: PASS
* `verify_local.ps1 -SkipAndroidBuild`: PASS (25 PASS; Android build stages
  intentionally skipped)
* fixed-QAIRT QNN-enabled debug/androidTest build and APK audit: PASS
* Sqrt6000/Sqrt4000 smoke, terminal training, checkpoint validation, runtime
  telemetry, and 256+256 HTP evaluation: PASS

Full gate, Compose, generation, commit, and push were intentionally not run.

## Next step

Keep the existing **Linear target `.0001`** recipe. Per the requested stopping
rule, the next single experiment should hold `target=.0001` fixed and test
`decay-start=3000`; do not add cosine or architecture changes in this task.
