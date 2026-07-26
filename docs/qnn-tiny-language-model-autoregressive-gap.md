# Tiny language model autoregressive-gap study

## Outcome

This study is `GOAL_PARTIAL_SUCCESS`. For every finite same-prefix result, the
evaluation and generation host paths produce identical inputs and identical
logits inside each backend. Expanding training from phase 0 to balanced phase
coverage makes the CPU reference exact on all 20 rollouts. The final HTP run,
however, has nonfinite training/evaluation output in two seeds and nonfinite
generation output in three seeds. It reaches only 4/20 exact free-running
rollouts and 4/20 exact oracle-prefix rollouts.

The result does not meet the required 16/20 HTP exact threshold. No performance
claim is made, and no closed-loop augmentation was attempted after the three-candidate
HTP exploration limit was reached.

## Forward-path audit

The audited prefix is `0,1,2,3,0,1,2,3`. Evaluation constructs it through
`languageBatch`; generation constructs the token IDs and one-hot rows independently.
Both routes use valid-token count 8, positions 0 through 7, the same fixed sinusoidal
position embedding, the same upper-triangular causal mask, the same parameters, and
logit row 7.

| comparison | maximum absolute error | mean absolute error | maximum relative error |
| --- | ---: | ---: | ---: |
| CPU evaluation vs CPU generation | 0 | 0 | 0 |
| HTP evaluation vs HTP generation | 0 | 0 | 0 |
| CPU evaluation vs HTP evaluation, seed 1 | 0.0122664 | 0.00247691 | 0.399705 |
| CPU generation vs HTP generation, seed 1 | 0.0122664 | 0.00247691 | 0.399705 |

The final representative seed has CPU/HTP maximum logit error 0.0122664 and
agrees on ArgMax token 0 and top-3 `0,1,3`. Seeds 1, 2, and 4 have finite
same-prefix outputs and exact evaluation/generation parity inside each backend;
seeds 3 and 5 are nonfinite, so no ArgMax, top-3, or finite error claim is made
for them.

The CPU/HTP difference is within the established one-step forward tolerance and
does not change the audited decision. Evaluation and generation share the same
QNN graph by design; the parity test independently constructs their host inputs
and checks all returned logits.

## Inference implementation audit

- Input-target shift is `target[t] = input[t+1]` within the generated rule.
- Loss covers all eight non-padding rows. This fixed-T8 graph has no padding rows
  or padding-loss mask.
- Generation reads `[T-1,V]`, not a padding row or flattened adjacent row.
- Softmax and ArgMax use the vocabulary axis.
- CPU and HTP use positions 0 through 7 on every sliding-window forward.
- The causal attention mask excludes all columns `j > row`.
- Free-running generation erases exactly one oldest token and appends exactly one
  predicted token. Oracle rollout appends the expected token instead.
- Each QNN execute binds the final current parameter buffer; current/next and Adam
  moment buffers do not alias.

No inference implementation mismatch was found.

## Dataset and sampling diagnosis

The baseline has four phase-0 batches, one per rule. Each pattern receives the
same number of updates, but only one window phase is observed. Sliding-window
generation changes the content phase while resetting position indices to 0 through
7, so later generation prefixes are outside baseline prefix coverage.

The adopted diagnostic candidate alternates phase 0 and phase 1 for each pattern:

```text
0:0, 1:0, 2:0, 3:0, 0:1, 1:1, 2:1, 3:1
```

At 1000 steps every pattern-phase pair appears 125 times. Pattern update counts
remain equal at 250. Target counts are 500 for tokens 0 through 7, 1000 for tokens
8 and 9, 625 for tokens 10 and 11, and 750 for token 12. Unused vocabulary IDs
13 through 31 remain softmax negatives only. Every position receives the same
number of examples within a pattern.

Variable prefix-length coverage is unsupported by the current fixed-T8 API because
it has neither a padding attention mask nor a loss mask. Faking padding was rejected.

## CPU candidate results

All CPU candidates use Adam, learning rate 0.0003, global gradient clipping at 10,
1000 steps, true sliding-window generation, and deterministic seeds.

| candidate | screening seeds | teacher-forced accuracy | mean margin | five-seed oracle exact | five-seed free exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| phase 0 only | 3 | 0.932692 | 2.16175 | 11/12 | 11/12 |
| balanced phase 0/1 | 3 then 5 | 0.982692 (5 seed) | 2.39114 | 20/20 | 20/20 |
| all rule phases | 3 then 5 | 0.996154 (5 seed) | 2.96585 | 20/20 | 20/20 |

For balanced phase 0/1, the five-seed pattern-by-position accuracy matrix is:

```text
Pattern A: 0.750, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
Pattern B: 0.950, 0.950, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
Pattern C: 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
Pattern D: 0.867, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000
```

The corresponding mean-margin matrix is:

```text
Pattern A: 0.619, 2.078, 2.042, 2.043, 1.953, 2.096, 2.050, 2.003
Pattern B: 1.920, 2.072, 2.050, 2.183, 2.278, 2.173, 2.196, 2.132
Pattern C: 3.112, 2.993, 3.010, 2.987, 2.995, 2.984, 3.001, 2.974
Pattern D: 2.486, 3.075, 3.121, 2.921, 3.203, 3.223, 3.095, 3.213
```

The aggregate confusion matrix has 511 correct rows out of 520. The nine errors
are `0→4`, `0→6`, `3→0` twice, `3→8`, `5→4`, `7→12`, `10→2`, and `11→12`.

## HTP candidate results

Three candidates were evaluated, which is the exploration limit.

| candidate | sampling | clip | finite | exact result |
| --- | --- | ---: | --- | --- |
| 1 | all phases | 10 | no; seed 2 non-finite | seed 1 reached 4/4 |
| 2 | all phases | 5 | no; seed 2 non-finite | 3/20 observed |
| 3 | balanced phase 0/1 | 10 | no; final run seeds 3 and 5 non-finite | 4/20 oracle and 4/20 free |

The final HTP candidate keeps the published model shape, Adam, learning rate
0.0003, clip 10, and 1000 steps.

| seed | initial loss | final loss | initial accuracy | final accuracy | oracle exact | free exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.44444 | 0.438405 | 0 | 0.96875 | 3/4 | 3/4 |
| 2 | 3.47524 | 1.41715 | 0 | 0.5625 | 0/4 | 0/4 |
| 3 | 3.46281 | NOT_AVAILABLE | 0.03125 | 0.375 | 0/4 | 0/4 |
| 4 | 3.46225 | 0.986265 | 0 | 0.8125 | 1/4 | 1/4 |
| 5 | 3.46466 | NOT_AVAILABLE | 0 | 0.21875 | 0/4 | 0/4 |

Seeds 1, 2, and 4 complete 1000 finite steps. Seed 3 stops after 635 finite steps
and seed 5 after 93; their final evaluation loss is nonfinite. Across free and
oracle modes, 9 of 40 bounded rollouts encounter nonfinite output, containing
2182 NaNs and no infinities. No run uses CPU fallback. Repeated HTP runs do not
reproduce the same failing seeds or exact count, so deterministic HTP replay is
explicitly false. The first finite detailed error is seed 1, Pattern B, generation
step 3. Its context is `7,4,5,6,7,4,5,6`; expected token 7 is assigned probability
0.0000346899, while predicted token 4 is assigned 0.425293. The top-3 is `4,5,9`,
margin is -9.41406, and entropy is 1.47104. The raw bounded report also records
these fields for every representative-seed generation step. Public
rollout aggregates are in the result directory without raw callbacks or device data.

Oracle and free-running fail at the same prefix positions for this candidate, so
the residual HTP failure is not primarily an exposure-gap cascade.

## Cause classification

The cause is `MULTIPLE_CAUSES`:

1. `PREFIX_COVERAGE_DEFICIT`: phase-0-only CPU training is weaker, while phase
   coverage makes CPU oracle and free-running rollout exact.
2. HTP phase01 numerical instability after accumulated CPU/HTP training-trajectory
   divergence: the same phase-covered data is finite and exact on CPU, while HTP
   can terminate early and emit NaN logits for particular prefixes. This is
   consistent with the previously documented divergence beginning in the token
   embedding gradient.

`INFERENCE_IMPLEMENTATION_MISMATCH` is rejected. `EXPOSURE_GAP` is not the
dominant remaining cause because oracle and free results coincide.

## Execution boundary and regressions

CPU supplies token IDs, constructs one-hot inputs, computes Adam bias correction
and clip scale, performs ArgMax, and updates context. HTP executes embedding,
causal Transformer, logits, backward gradients, gradient scaling, and Adam math.
Training uses two graphs and two executes per update; inference uses the training
forward graph with one execute per generated token.

Verified regressions include host references, Android unit tests, QNN APK audit,
device probe/forward, force-stop cycles, Skel reuse and app-private corruption
recovery, linear training, MLP split/fused/full-step, Transformer forward/backward,
tiny Transformer training, cross entropy, tiny LM SGD, Adam one-step/convergence,
and the bounded Adam inference diagnostic. The diagnostic itself correctly fails
on nonfinite phase01 HTP output; it is not recorded as a passing exact-generation
regression. Momentum one-step passes; the separate
historical momentum convergence candidate remains non-finite and is not a new
regression introduced here.

Performance and first-error perturbation robustness are `NOT_REACHED`, because
they are gated on a GOAL_SUCCESS exact-generation candidate.

## Public evidence

Allow-listed aggregate files are under
`docs/results/qnn-htp-tiny-language-model-generation-2026-07/`. Raw reports stay
under ignored `build/reports/`. The exporter stages output privately, rejects
paths outside repository `docs/results`, rejects reparse-point destinations,
scans for paths, endpoints, credentials, callbacks, and forbidden binary
extensions, then publishes the sanitized files.
