# L19 critical-margin stabilization objective

## Scope and classification

This investigation is the follow-up to the first-error/margin decomposition
([`qnn-htp-l19-first-error-margin-2026-08`](results/qnn-htp-l19-first-error-margin-2026-08/README.md)),
whose conclusion was `CRITICAL_TOKEN_MARGIN_LOSS`: the tokens the final
checkpoint corrects relative to the validation-selected checkpoint carry the
hard negative margin. The hypothesis prior carried into this goal is that a
checkpoint-selection objective or a training loss that maximizes the
target-vs-competitor logit margin of critical tokens can stabilize
free-running generation.

The goal is a host-only deterministic CPU replay over the pinned margin
partitions. It does not claim an HTP execution result, an overflow, or a
device-side effect. Both candidate families (checkpoint-selection objectives
and margin-aware training losses) failed their preregistered development
gates, so no candidate change was adopted, AR_FINAL_HOLDOUT_V3 stays unopened,
and no HTP smoke or formal run was scheduled. The public evidence is in
[`qnn-l19-critical-margin-stabilization-2026-08`](results/qnn-l19-critical-margin-stabilization-2026-08/README.md).

## Partition contract

`MARGIN_CALIBRATION_V1` (`fnv1a64:71806d5bf19c090a`) and
`MARGIN_DEVELOPMENT_V1` (`fnv1a64:f06fcc3e2d12ca99`) fix 24 cases each
(MIXED_PREFIX_SUFFIX_2_6_7_DISTRACTOR_OFFSET_1/2; 8-token prefix and a 4- or
8-token rollout per case, 144 evaluated tokens over 24 cases) before any
candidate is evaluated. Across the two partitions, case IDs, initial prefixes,
and complete sequences have zero overlap; the 13 shared successor transitions
at 144 total occurrences are reported rather than hidden in
`dataset-overlap.csv`. Both partitions are also disjoint from TRAIN,
AR_VALIDATION_V3, AR_DEVELOPMENT_V3, and AR_FINAL_HOLDOUT_V3 by case, prefix,
and full sequence (zero in all nine `dataset-overlap.csv` rows), with the same
structural transition sharing reported there.

## Regeneration anchors

The four reference trajectories (L19 seeds 1/2/4 and the L18 seed-2 control)
were regenerated from the CPU reference in this goal and reproduce the
canonical margin bundle exactly: pinned selected steps 16/4/12/4 re-selected,
selected and final exact counts identical to the canonical configuration.csv
for all four runs, and bitwise loss and gradient parity across every one of
the 320 training steps (`consistency.csv`, re-checked by the exporter against
the published margin bundle before exporting).

## Checkpoint-selection objectives: 12 variants, development REJECT

Twelve margin-aware checkpoint-selection objectives are scored on the 23-step
cadence 0..320 per configuration (`objective-scores.csv`), correlated with
development quality (`objective-correlations.csv`), and cross-validated by
leave-one-seed-out (`leave-one-seed-out.csv`):

- MARGIN_DEFICIT_MEAN_D0 / D025 / D05 — mean target-vs-top1 deficit at floors
  0.0 / 0.25 / 0.5.
- LOWER_TAIL_MARGIN_Q10 / Q20 — 10th / 20th percentile of the target margin.
- SOFT_WORST_MARGIN_T025 / T05 / T1 — tau-scaled logsumexp of negative margins.
- SEQUENCE_SURVIVAL_NLL_T025 / T05 / T1 — survival-NLL style sequence costs.
- FIRST_ERROR_HAZARD_INV_POSITION — inverse first-error-position hazard.

The strongest single-correlation variant is LOWER_TAIL_MARGIN_Q20 (Spearman
0.95 against development token exact on seed 4), and the best-ranked variant
by the preregistered pooled-evidence comparator is MARGIN_DEFICIT_MEAN_D0
(Spearman 0.83 on seed 1), but the development
gate (`development-gate.csv`, `decision.csv`) rejects every variant: no
variant reaches even one supported seed; pooled token/sequence exact are
non-worse in no variant; the L18 control is worse in all variants; first-error
median survival does not improve; and every LOSO fold collapses
(collapse-free false, mean token delta -59 across all three folds). The
best-variant selection is an early low-exact checkpoint (51 tokens below the
trajectory-best checkpoint and 43 below the final checkpoint on seed 1):
maximizing margin quality and maximizing final-checkpoint quality do not
co-select on this 320-step trajectory.

Gradient attribution (`gradient-attribution.csv`, 4x320 steps, per-step loss
and gradient parity true) classifies the decoupling driver as
`OUTPUT_HEAD_RANKING_DRIFT`: the output-projection gradient share grows while
the critical-token share of the loss falls below the critical-underweight
threshold late in training. Conclusion:
`CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT`.

## Training families: 2 preregistered families, 12 runs, both REJECT

The two margin-aware loss families were implemented host-only as a verbatim
copy of the CPU training reference, with bitwise parity of the copied backward
pass against `tiny::forwardBackward` on the CE dLogits enforced at cadence 32
in every run (last parity step 320 in all eight full runs):

- PAIRWISE_MARGIN_CE_V1 (delta=0.5, lambda=1.0): CE plus a pairwise hinge on
  deficient target-vs-competitor margins.
- SEQUENCE_WORST_MARGIN_CE_V1 (tau=0.5, lambda=1.0): CE plus
  lambda*tau*logsumexp(-margin/tau) over the sequence's worst margin.

Micro smoke (4 runs, 64 steps, cadence 4) confirmed mechanics: runs finite,
margins improve, parity holds. The full gate (8 runs, 320 steps;
`training-family-gate.csv`, `training-family-decision.csv`):

| Family | Finite | Improved seeds (of 3) | Control non-worse | Margin improved | Stability | Pooled token delta | Pass |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| PAIRWISE_MARGIN_CE_V1 | true | 2 | false | true | true | +67 | false |
| SEQUENCE_WORST_MARGIN_CE_V1 | false | 1 | true | false | false | -61 | false |

PAIRWISE more than doubles final token exact on two of three L19 seeds (30 to
69 and 46 to 79) with dev margins and NLL improving, but regresses seed 2
(63 to 58) and the L18 control (65 to 62), failing the preregistered
control-non-worse and 2-of-3-seeds conditions. SEQUENCE_WORST at tau=0.5 is
unstable: the worst-margin term keeps pushing easy train margins and diverges
development NLL (NOT_FINITE in all four full runs, visible in
`training-trajectory.csv`).

Conclusion: `NO_TRAINING_FAMILY_ACCEPTED`. The canonical AR-selected
checkpoints remain the final candidate; the final holdout stays unopened and
no HTP smoke or formal run is scheduled (host-only rejection, no candidate
change).

## Reproduction

- Objective probe + benchmark:
  `scripts/run_critical_margin_objective_benchmark.ps1` (optionally `-SelfTest`),
  reports under `build/reports/qnn-critical-margin-objective`.
- Training families (micro then full): same script with
  `-Train -Micro -BaselineDir <objective report>` and
  `-Train -BaselineDir <objective report>`, reports under
  `build/reports/qnn-critical-margin-training-micro` /
  `build/reports/qnn-critical-margin-training`.
- Public export:
  `scripts/export_public_qnn_l19_critical_margin_results.ps1` (optionally
  `-SelfTest`).
- Host tests: `scripts/run_host_tests.ps1` (unit test plus probe self-test).
