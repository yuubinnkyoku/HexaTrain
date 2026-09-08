# Nicopedia HPO: V1024 / T32 / D64 / FFN128 / L19 / H2, LR-v1b

Status: COMPLETE for the declared lower-boundary extension (seed 1). This
report describes the best observed LR in the combined LR-v1/LR-v1b search; it
does not claim an optimal LR, statistical significance, or full tuning.

## Motivation

LR-v1 selected `0.0015`, the lower edge of its declared search space. LR-v1b
extends only that edge with `0.00105` and `0.00075`; architecture, data order,
optimizer, schedule, and seed remain unchanged. The motivation is consistent
with the small-model sensitivity caution in Lourie et al., “Small-Scale
Experiments: Are We There Yet?” ([arXiv:2608.11859](https://arxiv.org/abs/2608.11859)):
small-model conclusions should be measured on a tuned frontier rather than
inferred from one arbitrary hyperparameter setting. That paper motivates the
controlled measurement here; it does not establish a result for this model,
dataset, or device.

## Fixed configuration

Every LR-v1b trial uses the exact FFN128 identity:

* vocabulary `V=1024`; tokenizer `byte_bpe`; tokenizer SHA-256
  `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
* `T=32`, `D=64`, `FFN=128`, `L=19`, `H=2`; parameter count 758,528
* batch size 8; seed 1
* Adam beta1 0.9, beta2 0.999, epsilon `1e-8`
* gradient clip disabled; weight decay 0
* constant LR schedule; warmup 0; decay 0
* dataset cache content hash `fnv1a64:0c7b2826f5f26fea`
* canonical training-order identity/hash
  `fnv1a64:0e2e15196d851431`
* QAIRT `2.48.40.260702151143`; QNN backend HTP

The independent variable is the scalar learning rate only. The expected
step-8000 exposure is 2,048,000 target tokens, 5,491,256 original UTF-8 bytes,
46,616 chunks, and 1,949 articles.

## Search points and reuse

New deterministic trial IDs are:

* `hpo-lr-v1b-lr0p00105-seed1`
* `hpo-lr-v1b-lr0p00075-seed1`

The LR-v1 anchors `0.0015`, `0.0022`, `0.0030`, and `0.0042` were imported only
after manifest, checkpoint header, parameter hash, finite-state, tokenizer,
data/order, and QNN-health checks. No LR-v1 anchor was retrained, and no
checkpoint from another LR was used for resume.

## Multi-fidelity policy

Each new point passed a 4-step HTP smoke and the 1000-step 64+64 diagnostic.
Healthy candidates were retained through the 4000-step 128+128 rung; both
LR-v1b candidates were promoted by the declared no-prune policy. The primary
endpoint is exact step-8000 256+256 balanced bpb. The late selector is
diagnostic only for LR-v1b and evaluates 6500, 7500, and 8000 at 128+128.
Early metrics are not treated as an oracle because LR-v1 showed 4000-to-8000
ranking changes and small-eval noise.

## Rung 1000

| LR | Val bpb | Dev bpb | balanced bpb | health | status |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.00075 | 3.309631390 | 3.597768561 | 3.453699976 | PASS | COMPLETE |
| 0.00105 | 3.294614011 | 3.580042380 | 3.437328196 | PASS | COMPLETE |
| 0.00150 | 3.273010388 | 3.583689022 | 3.428349705 | PASS | REUSED |
| 0.00220 | 3.241813540 | 3.544772203 | 3.393292872 | PASS | REUSED |
| 0.00300 | — | — | — | anchor | exact rung unavailable |
| 0.00420 | 3.230328103 | 3.518341961 | 3.374335032 | PASS | REUSED |

The 1000-step values were diagnostic only. No candidate was pruned for being
temporarily last at this rung.

## Rung 4000

| LR | Val bpb | Dev bpb | balanced bpb | health | promoted |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.00075 | 2.792252247 | 3.331598842 | 3.061925545 | PASS | yes |
| 0.00105 | 2.760819265 | 3.319877233 | 3.040348249 | PASS | yes |
| 0.00150 | 2.744185321 | 3.267589547 | 3.005887434 | PASS | anchor |
| 0.00220 | 2.717130022 | 3.245737228 | 2.981433625 | PASS | anchor |

Both new points were retained to the fixed-budget endpoint. The anchor
`0.0030` has no exact validated step-4000 artifact and remains an 8000-step
comparison anchor.

## Step-8000 primary: combined LR frontier

The fixed-step 256+256 table is the primary comparison. The checkpoint hash is
the validated parameter hash reported by the evaluator.

| LR | Val bpb @8000 | Dev bpb @8000 | balanced bpb @8000 | source | checkpoint hash |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.00075 | 2.383858843 | 2.667133756 | 2.525496300 | LR-v1b | `fnv1a64:05c1075db4ab28f9` |
| 0.00105 | 2.354378605 | 2.635076844 | 2.494727724 | LR-v1b | `fnv1a64:529a111ff766eeebb` |
| 0.00150 | 2.338413313 | 2.624521011 | **2.481467162** | LR-v1 anchor | `fnv1a64:3ee6512fa4097587` |
| 0.00220 | 2.349580766 | 2.620467215 | 2.485023991 | LR-v1 anchor | `fnv1a64:428545ff03cf6bf2` |
| 0.00300 | 2.351117109 | 2.621106690 | 2.486111900 | LR-v1 anchor | `fnv1a64:d6eae31f0ff6d7b2` |
| 0.00420 | 2.350257904 | 2.622528721 | 2.486393313 | LR-v1 anchor | `fnv1a64:94317688643e42a8` |

The lower-boundary extension did not improve on `0.0015`. Relative to the
incumbent `0.0030`, the best observed `0.0015` has Val `-0.012703796`, Dev
`+0.003414321`, and balanced `-0.004644738` bpb.

## Late selector (secondary diagnostic)

LR-v1b intentionally does not run a second 256+256 evaluation for the selected
late checkpoint. The following 128+128 values are only for checkpoint-shape
diagnostics and do not change the primary ranking.

| LR | step | Val bpb | Dev bpb | balanced bpb |
| ---: | ---: | ---: | ---: | ---: |
| 0.00075 | 6500 | 2.729314804 | 3.245889843 | **2.987602324** |
| 0.00075 | 7500 | 2.727313643 | 3.274801044 | 3.001057344 |
| 0.00075 | 8000 | 2.727355409 | 3.284437364 | 3.005896387 |
| 0.00105 | 6500 | 2.713238515 | 3.222251661 | 2.967745088 |
| 0.00105 | 7500 | 2.697785294 | 3.265889204 | 2.981837249 |
| 0.00105 | 8000 | 2.690158572 | 3.237098333 | **2.963628453** |

The late diagnostic best is step 6500 for `0.00075` and step 8000 for
`0.00105`. These are not substitutes for the fixed-step primary endpoint.

## Boundary and sensitivity interpretation

* Best observed LR in the combined LR-v1/LR-v1b frontier: `0.0015`.
* Lower-boundary verdict: **closed for this extension**. Both new points below
  `0.0015` were worse at the primary endpoint; this does not prove a global
  optimum.
* Best-to-worst primary gap across the six stable 8000-step rows:
  `2.525496300 - 2.481467162 = 0.044029138` bpb.
* The LR-v1 interior range alone was much flatter (`0.004926151` bpb). The
  combined curve is a shallow interior plateau from roughly `0.0015` to
  `0.0042`, followed by a clear quality shortfall at `0.00105` and especially
  `0.00075` under this single-seed measurement.

These are seed-1 observations. Differences are not called statistically
significant; a difference below roughly 0.01 bpb remains noise-sensitive in
this protocol.

## Compute and systems accounting

LR-v1b standalone planned new compute was `2 * 8000 = 16,000` steps. The
device execution history was:

* `0.00105`: `[4, 996, 3000, 4000] = 8,000` steps.
* `0.00075`: `[4, 996, 3000, 2000, 500, 1250, 750] = 8,500` steps. The
  additional 500 steps are duplicate work from an interrupted recovery
  attempt; all final checkpoints were identity- and finite-validated.

Thus actual new training execution was 16,500 steps, with 500 retry-overhead
steps and no standalone compute saving (`-3.13%` versus 16,000 planned). The
four LR-v1 anchors represent 32,000 reused steps. Against a hypothetical fresh
six-point 48,000-step run, anchor reuse avoided 31,500 fresh steps in this
phase (`65.63%`), after counting the retry overhead.

Known successful instrumentation wall time was approximately 47,736.172 s
(~13.26 h), excluding interrupted/unreported training attempts, held-out eval
time, and ADB transport waits. Representative final-segment observations:

| LR | segment | ms/update | original bytes/s |
| ---: | --- | ---: | ---: |
| 0.00075 | 7250→8000 | 3220.219 | 212.836 |
| 0.00105 | 4000→8000 | 3453.401 | 198.599 |

The timing spread is treated as device scheduling/thermal noise, not LR
quality evidence. Thermal status remained 0 with observed temperatures about
37–43°C. All completed training and eval artifacts report QNN return-code
success, finite tensors/checkpoints, zero graph failures where reported, and
`cpu_fallback=false`.

## Recovery and ledger notes

The reusable runner is `scripts/run_nicopedia_hpo_lr.ps1`; artifacts are under
`build/hpo/nicopedia-v1024-d64-f128/lr-v1b/`. Resume remained same-trial and
fail-closed. During recovery, ADB aliases changed between `.16` and `.17`; the
runner selected one endpoint only after stable serial/product identity
verification. Two harness interruptions occurred while preserving valid
checkpoints; neither was classified as a QNN or numerical failure.

The runner now:

* records final checkpoint path, format, and parameter hash in the manifest;
* deduplicates late-selector rows on retry;
* retains the finite heartbeat bound and exact-run ownership checks.

`summary.csv`, `manifests.json`, `trials.jsonl`, and `compute.json` are
machine-readable research artifacts. Raw checkpoints and device evidence stay
under ignored `build/` paths and are not committed.

## Verification

* PowerShell parser check: PASS.
* HPO runner self-test (`v1b`): PASS.
* HTP training runner self-test: PASS.
* Final training: `PASS NICOPEDIA_HTP`, QAIRT build ID pinned.
* Final 256+256 and late 128+128 evals: HTP PASS, finite, no CPU fallback.
* No APK/native rebuild was needed after the runner-only fixes; Full gate,
  Compose, generation, seed-2, architecture, optimizer, and schedule searches
  were intentionally not run.

## Next phase

1. Start a dedicated LR-schedule phase with peak-LR candidate `0.0022` (the
   LR-v1 4000-step leader) and a decay target around `0.0015` (the fixed-step
   8000-step leader). This is a hypothesis, not a result of schedule HPO.
2. If schedule results warrant it, repeat only the selected schedule/peak
   region with seed 2.

Architecture changes remain out of scope. The result is an “LR-tuned frontier
v1”, not “fully tuned”.
