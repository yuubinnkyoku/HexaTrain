# Nicopedia HPO: V1024 / T32 / D64 / FFN128 / L19 / H2, LR-v1

Status: COMPLETE for the declared LR-v1 sweep (seed 1). This document does not
claim an optimal learning rate; it reports the best observed LR in the declared
LR-v1 search space.

## Motivation

The 758,528-parameter FFN128 candidate is small enough that learning-rate
sensitivity can dominate a comparison. This is consistent with the caution in
Lourie et al., “Small-Scale Experiments: Are We There Yet?” (arXiv:2608.11859):
small-model conclusions should be evaluated on a tuned frontier rather than a
single arbitrary setting. See the [paper](https://arxiv.org/abs/2608.11859).
That reference motivates controlled measurement here; it does not establish a
result for this model, dataset, or device.

## Fixed configuration

Every trial keeps the current FFN128 experiment identity unchanged:

* vocabulary 1024; tokenizer `byte_bpe`; tokenizer SHA-256
  `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* context `T=32`; `D=64`; `FFN=128`; `L=19`; `H=2`
* parameter count 758,528; batch size 8; seed 1
* Adam, beta1 0.9, beta2 0.999, epsilon 1e-8
* gradient clip disabled; weight decay 0; constant LR schedule; warmup and
  decay steps 0
* canonical cache identity `fnv1a64:0c7b2826f5f26fea` and training-order
  identity/hash `fnv1a64:0e2e15196d851431`
* QNN HTP backend with the pinned QAIRT build
  `2.48.40.260702151143`

The only independent variable is the scalar learning rate.

## Search space and trial identity

The five predeclared points are `0.0015`, `0.0022`, `0.0030`, `0.0042`, and
`0.0060`. Deterministic IDs are of the form
`hpo-lr-v1-lr0pXXXX-seed1`; an ID never represents another LR. The device
order is recorded in the ledger and is deliberately not monotonically sorted.

The incumbent `LR=0.0030` is reused only after validating the exact model,
tokenizer, data/order, optimizer/stability identity, checkpoint header,
parameter hash, finite state, and QNN-health fields. It is never resumed with a
different LR.

## Multi-fidelity policy

Each new LR first performs a 4-step HTP smoke. A smoke may not be pruned merely
because its loss rises; QNN success, finite tensors, no CPU fallback, and exact
identity are the health gate. Healthy trials continue through rungs 1000 and
4000. Rung 1000 records a 64+64 diagnostic and is not an aggressive ranking
oracle. Rung 4000 records 128+128 and uses balanced bpb
`(Val bpb + Dev bpb) / 2` for conservative promotion.

Promotion keeps the best candidate, any candidate within 0.03 bpb of that best
up to three candidates, and at least two candidates when available. The 0.03
margin is a conservative noise-aware engineering rule, not a significance
threshold. A candidate is pruned only for the declared health/corruption/
identity/failure conditions or for missing the promotion set.

Promoted trials continue to step 8000. The primary comparison is exact-budget
256+256 at step 8000. A secondary late selector evaluates steps 6000, 6500,
7000, 7500, and 8000 at 128+128, then evaluates the selected late checkpoint at
256+256. The primary ranking is always step-8000 256+256 balanced bpb.

This conservative policy is intentional: prior V1024 work observed early
plateaus followed by later improvement, and the FFN128 experiment showed a
ranking inversion between 64+64 and 256+256. Cheap early metrics are therefore
diagnostic evidence, not an oracle.

## Trial ledger

The reusable runner is `scripts/run_nicopedia_hpo_lr.ps1`. It writes only
research artifacts below `build/hpo/nicopedia-v1024-d64-f128/lr-v1/`:

* `plan.json`: fixed identity, LR space, trial order, rungs, and policy
* `trials/<trial-id>/manifest.json`: machine-readable identity, status,
  checkpoint, rung, health, and timing fields
* `trials.jsonl`: append-only phase events
* `summary.csv`: bpb rows suitable for Python/R analysis
* `compute.json`: executed/reused steps and savings

Resume is fail-closed. The runner validates checkpoint magic/header, model and
tokenizer identity, checkpoint step, manifest LR, and actual parameter hash;
filename matching alone is insufficient. A complete, valid exact-match trial
is not rerun.

## Results

The authoritative numeric tables are stored in `summary.csv` and the validated
trial manifests. Interpret differences as follows: `>0.03` bpb is a meaningful
candidate difference, `0.01–0.03` is small/moderate, and `<0.01` is
noise-sensitive/inconclusive. This is a single-seed sweep, not a statistical
significance claim.

### Rung 1000

All four new LR candidates passed the QNN/numerical health gate. The incumbent
has no exact step-1000 artifact and was not retrained just to fill that row.

| LR | Val bpb | Dev bpb | balanced bpb | health | status |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.0015 | 3.273010388 | 3.583689022 | 3.428349705 | PASS | COMPLETE |
| 0.0022 | 3.241813540 | 3.544772203 | 3.393292872 | PASS | COMPLETE |
| 0.0030 | — | — | — | anchor | exact rung unavailable |
| 0.0042 | 3.230328103 | 3.518341961 | 3.374335032 | PASS | COMPLETE |
| 0.0060 | 3.262786251 | 3.551971189 | 3.407378720 | PASS | COMPLETE |

All four have QNN return-code success, finite tensors, zero graph failures, and
`cpu_fallback=false`. No candidate was pruned for being temporarily last at
this rung. `.0030` is recorded as `existing anchor / exact rung1000 unavailable`.

### Rung 4000

The 128+128 primary rung is complete for all four new candidates:

| LR | Val bpb | Dev bpb | balanced bpb | health | promotion |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.0015 | 2.744185321 | 3.267589547 | 3.005887434 | PASS | PROMOTED |
| 0.0022 | 2.717130022 | 3.245737228 | 2.981433625 | PASS | PROMOTED (best) |
| 0.0042 | 2.729113727 | 3.267731790 | 2.998422759 | PASS | PROMOTED |
| 0.0060 | 2.765934096 | 3.247646502 | 3.006790299 | PASS | PRUNED by max-3 cap |

`.0030` has no validated exact step-4000 artifact and remains an 8000-step
comparison anchor. `.0022` was the best 4000-step candidate; `.0015` and
`.0042` were within the conservative 0.03 bpb margin. `.0060` was also within
that margin but was excluded by the explicit maximum-three rule, not by health
failure. This is conservative successive-halving, not aggressive early pruning.

### Rung 8000 and final comparison

Every promoted candidate reached the same exposure at step 8000
(2,048,000 target tokens, 5,491,256 original UTF-8 bytes, 46,616 chunks,
1,949 articles). The exact-budget 256+256 primary comparison is:

| LR | Val bpb | Dev bpb | balanced bpb | checkpoint hash |
| ---: | ---: | ---: | ---: | --- |
| 0.0015 | 2.338413313 | 2.624521011 | **2.481467162** | `fnv1a64:3ee6512fa4097587` |
| 0.0022 | 2.349580766 | 2.620467215 | 2.485023991 | `fnv1a64:428545ff03cf6bf2` |
| 0.0030 (incumbent) | 2.351117109 | 2.621106690 | 2.486111900 | `fnv1a64:d6eae31f0ff6d7b2` |
| 0.0042 | 2.350257904 | 2.622528721 | 2.486393313 | `fnv1a64:94317688643e42a8` |

The best observed LR in LR-v1 by the predeclared primary metric is `0.0015`.
Relative to the `.0030` incumbent, its primary deltas are Val `-0.012703796`,
Dev `+0.003414321`, and balanced `-0.004644738` bpb.

For the three new promoted finalists, the late selector evaluated
6000/6500/7000/7500/8000 at 128+128 and then evaluated only the selected
checkpoint at 256+256. The incumbent `.0030` has existing compatible 6500 and
7500 256+256 references; the best available incumbent late reference (7500) is
kept as the comparison anchor rather than retraining or fabricating missing
late-selector points:

| LR | selected step | Val bpb | Dev bpb | balanced bpb |
| ---: | ---: | ---: | ---: | ---: |
| 0.0015 | 6500 | 2.360608407 | 2.614626320 | **2.487617364** |
| 0.0022 | 6500 | 2.371976054 | 2.622524428 | 2.497250241 |
| 0.0030 (incumbent anchor) | 7500 | 2.357021533 | 2.627340584 | 2.492181058 |
| 0.0042 | 7500 | 2.355469711 | 2.628498818 | 2.491984265 |

The secondary late-best delta for `.0015` versus the incumbent's best available
late checkpoint is Val `+0.003586874`, Dev `-0.012714264`, balanced
`-0.004563694` bpb. Primary and late-selector rankings are intentionally kept
separate; their inversion is part of the result.

### LR sensitivity

The completed-primary best-to-worst stable gap is `0.004926151` bpb, below the
0.01 noise-sensitive guide. The curve is therefore relatively flat over the
observed interior, with late-selector ranking inversion (`.0022` is second in
the primary metric but last among the late-best rows). The lower search boundary
was hit: `.0015` is the best observed point and no lower LR was tested. The
upper `.0060` boundary was not the best point. This does not establish an
optimum, significance, or full tuning for a seed-1 sweep.

## Compute and systems evidence

The ledger records trial count, new steps, reused steps, promoted/pruned counts,
wall time, ms/update, bytes/s, thermal state, QNN return-code health, finite
tensors, and fallback flags. ADB transport interruptions are infrastructure
events, not numerical evidence. Performance comparisons are annotated when
thermal throttling is extreme.

The validated 1000-step training-time measurements are:

| LR | training seconds | ms/update | original bytes/s |
| ---: | ---: | ---: | ---: |
| 0.0015 | 2877.16 | 2888.71 | 238.13 |
| 0.0022 | 3426.83 | 3440.60 | 199.93 |
| 0.0042 | 3217.04 | 3229.96 | 212.97 |
| 0.0060 | 3098.17 | 3110.61 | 221.14 |

These are systems observations at different times, not quality evidence. The
naive full sweep is 40,000 steps. Actual new training was 24,000 steps,
including a 1,500-step `.0060` partial segment observed without a final report;
22,500 new steps have complete SUCCESS reports. Reused artifacts account for
12,000 steps (four exact rung-1000 checkpoints plus the exact `.0030` 8000-step
incumbent), and the `.0060` promotion exclusion avoided a further 4,000 steps.
Thus the validated accounting is 16,000 steps saved, or 40.0% versus naive.
Known SUCCESS training-report wall time is 79,182.97 seconds (~22.0 hours),
excluding the unreported partial segment, held-out evaluation time, and ADB
transport waits.

### Recovery incident

The first HPO parent had host-ledger bugs in stdout return-value handling and
`OrderedDictionary.ContainsKey`. They produced stale `FAILED` wrapper states,
not QNN or numerical failures. The runner was minimally fixed to use a safe
property-presence check for `late_selector`; valid artifacts were adopted into
the manifests rather than retrained. ADB transport loss occurred during the
run, but no force-stop or parallel trial was issued. The overnight host watcher
was not a reliable monitor while ADB was offline; completion was established
after reconnect from checkpoint headers, hashes, reports, finite/QNN health,
and exposure evidence.

## Caveats and next phase

This is an LR-v1, seed-1 frontier only. It does not explore architecture,
schedule, Adam betas, epsilon, clipping, weight decay, or a second seed. The
output should be described as an “LR-tuned frontier v1”, not “fully tuned”.

Recommended next steps (maximum two):

1. Extend the lower LR boundary around `0.0015` (the best observed point is on
   the lower edge); keep the same fixed architecture and seed first.
2. Run a separate LR-schedule phase only after deciding whether the lower-boundary
   extension changes the frontier.

Architecture changes remain out of scope. This result is an “LR-tuned frontier
v1”, not “fully tuned”.
