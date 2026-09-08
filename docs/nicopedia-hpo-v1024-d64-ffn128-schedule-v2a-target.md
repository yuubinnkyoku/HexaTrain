# Nicopedia HPO: Schedule-v2a target-LR sweep

Status: **COMPLETE**. The 2,000-step proxy sweep completed for all three new
targets, and the conditional T0400 full-tail confirmation completed on the
same device after reconnecting and recovering the terminal checkpoint/report.
This is a seed-1 observation, not a statistical significance claim.

## Motivation and Schedule-v1 evidence

Schedule-v1 held peak LR at `0.0022` and showed that a linear late decay was
better than constant LR at the same exact step budget:

| Schedule-v1 run | Val256 | Dev256 | Balanced |
| --- | ---: | ---: | ---: |
| C1500, constant `.0015` | 2.338413313 | 2.624521011 | 2.481467162 |
| C2200, constant `.0022` | 2.349580766 | 2.620467215 | 2.485023991 |
| S4000, linear target `.0015` | 2.330467873 | 2.595317375 | **2.462892624** |
| S6000, linear target `.0015` | **2.330305757** | 2.600519830 | 2.465412794 |

Schedule-v2a therefore changes only the decay target. Decay start, shape,
peak LR, optimizer, model, data order, seed, and evaluation budget remain
fixed.

## Why a tail proxy

Every new target forks the validated C2200 step-6000 checkpoint and runs only
steps 6001–8000. This avoids repeating the common 0–6000 prefix: three fresh
8,000-step trials would cost 24,000 steps, while the proxy costs 6,000 new
steps. The proxy is justified directionally by the Schedule-v1 S4000/S6000
Balanced gap of about `0.00252` bpb, but it does **not** assume that S6000
ranking always preserves S4000 ranking. A proxy winner meeting the prescribed
threshold is confirmed with one S4000-start full tail.

## Fixed configuration

* `V=1024`, `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; 758,528 parameters
* batch `8`, seed `1`, byte-BPE tokenizer
* tokenizer SHA-256 `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* dataset cache `fnv1a64:0c7b2826f5f26fea`; canonical order hash
  `fnv1a64:0e2e15196d851431`
* Adam `beta1=.9`, `beta2=.999`, `eps=1e-8`; gradient clip disabled; weight
  decay `0`
* QAIRT build `2.48.40.260702151143`; QNN HTP backend
* expected exact step-8000 exposure: 2,048,000 target tokens, 5,491,256
  original UTF-8 bytes, 46,616 chunks, 1,949 articles

The global step is 1-indexed and both endpoints are inclusive:

```text
step <= 6000: lr = 0.0022
6000 < step < 8000:
    p = (step - 6000) / 2000
    lr = 0.0022 + p * (target - 0.0022)
step >= 8000: lr = target
```

## Common parent and explicit fork

All proxy children use the validated C2200 constant-LR step-6000 checkpoint,
with `parent_step=6000` and parent checkpoint SHA-256
`ee8c122612b832c89e7c6a3c3d360c76f38a28c06f260f1b0a40d969dc309063` recorded
in each manifest. The parent identity is V/T/D/FFN/L/H `1024/32/64/128/19/2`,
batch 8, seed 1, parameter hash `fnv1a64:bde59040980ad8ce`, tokenizer/data/order
identity above, Adam state intact, finite state, QNN health valid, and
`fallback=false`. Each child manifest records the fork flag, parent trial/hash,
parent step, peak/target LR, decay bounds/type, and all model/data/optimizer
identity fields. Normal cross-LR resume semantics were not broadened.

## Target search space and proxy results

The existing S6000 target `.0015` artifact is reused without retraining. New
children are T1000, T0700, and T0400. Every new child passed the first 2–4
update smoke and completed step 8000 with runtime telemetry checked at steps
6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, and 8000.

| Target LR | Val256 | Dev256 | Balanced | Δ vs `.0015` target |
| ---: | ---: | ---: | ---: | ---: |
| `.0015` (reused S6000) | 2.330305757 | 2.600519830 | 2.465412794 | 0 |
| `.0010` | 2.313034397 | 2.587311777 | 2.450173087 | -0.015239707 |
| `.0007` | 2.303790595 | 2.568418868 | 2.436104732 | -0.029308062 |
| `.0004` | 2.294507954 | 2.559579893 | **2.427043924** | -0.038368870 |

The proxy best is **T0400 (`.0004`)**. It improves both Val and Dev versus
T1500 by more than the prescribed `≈0.005` Balanced threshold, so the one
allowed full-tail confirmation was executed.

## Proxy validity caveat

The proxy result is a strong directional seed-1 observation, not proof that the
target ranking is invariant to decay start. The S6000/S4000 Schedule-v1
difference is only about `0.00252` Balanced bpb, and ranking inversion remains
possible. The full-tail check is the safeguard against treating the proxy
ranking as final.

## Full-tail confirmation

Executed: **yes**. The selected run was exactly the required C2200 step-4000
parent with linear decay from `.0022 @ 4000` to T0400 `.0004 @ 8000`; no other
target was run from step 4000. Smoke step 4004, terminal step 8000, checkpoint
header/hash, runtime telemetry, and exact 256+256 HTP evaluation all passed.
An intermediate ADB transport interruption was classified as infrastructure
failure; after reconnect, the terminal device report and final checkpoint were
recovered without force-stop, data deletion, or parallel training.

| S4000 target LR | Val256 | Dev256 | Balanced |
| ---: | ---: | ---: | ---: |
| `.0015` (existing S4000 artifact) | 2.330467873 | 2.595317375 | 2.462892624 |
| `.0004` (confirmed T0400) | **2.291483826** | **2.560856598** | **2.426170212** |

The confirmed T0400 improves Balanced by `0.036722412` bpb versus the existing
S4000 `.0015` anchor, with both Val and Dev lower.

## Compute accounting

| Quantity | Value |
| --- | ---: |
| naive fresh proxy sweep | 24,000 steps |
| successful new proxy training | 6,000 steps |
| reused common prefix | 18,000 steps |
| proxy steps saved | 18,000 (75%) |
| full-tail confirmation | +4,000 steps (completed) |
| actual new work | 10,000 steps |
| total prefix reused | 22,000 steps |
| total steps saved | 14,000 (58.33%) |

The proxy sweep saved `18,000` steps (75%) versus three fresh 8,000-step
trials; including the one required full-tail confirmation, the final new-work
accounting is `10,000` versus `24,000` naive steps.

## Systems

The three proxy runs and the one full-tail confirmation were serial on one
physically identified device. Native proxy training times were approximately
6,111 s (T1000), 6,206 s (T0700), and 6,223 s (T0400), with segment throughput
about 224.13, 220.69, and 220.11 original-UTF-8 bytes/s respectively. The
confirmed full-tail report recorded approximately 13,333.6 s and 205.75
original-UTF-8 bytes/s. These are device scheduling/thermal observations, not
quality evidence. All smoke/final runs reported QNN success, finite tensors,
zero graph failures, and `cpu_fallback=false`. Thermal status remained `0`;
observed battery temperatures were approximately 34–42 °C for proxy runs and
34–43 °C during the full-tail run.

## Verification

PowerShell parser checks, runner self-tests, the targeted learning-rate host
test, `verify_local.ps1 -SkipAndroidBuild`, fixed-QAIRT Android build, and APK
audit passed. Runtime telemetry matched the linear formula at every required
anchor. The full-tail training report independently recorded QNN return-code
success, finite tensors, zero graph-execution failures, and
`cpu_fallback=false`; the 256+256 evaluator reported the same health fields.
The full gate, Compose, generation, commit, and push were intentionally not
run for this experiment.

## Verdict and next recommendation

**Schedule-v2a COMPLETE.** The target sensitivity is **strong**: lower targets
improved monotonically in the S6000 proxy, and the T0400 winner also improved
both splits in the required S4000 full-tail confirmation. Proxy/full-tail
winner-vs-anchor ranking agreement is **yes**. Do not call the result
statistically significant with seed 1.

1. Validate the `.0004` target with seed 2 (or one additional shape proxy).
2. If seed-2 gain is not reproduced, keep `.0015` and compare linear versus
   cosine proxy.

Architecture changes remain out of scope.
