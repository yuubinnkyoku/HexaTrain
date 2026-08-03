# Critical-margin stabilization objective, August 2026

This bundle is the follow-up to the first-error/margin decomposition
(docs/results/qnn-htp-l19-first-error-margin-2026-08), whose conclusion was
CRITICAL_TOKEN_MARGIN_LOSS. The hypothesis prior carried into this
investigation is that a checkpoint-selection or training objective that
maximizes the target-vs-competitor logit margin of critical (hard-negative)
tokens can stabilize generation quality. Everything here is a host-only
deterministic CPU replay (fixed QAIRT-free C++ reference) over the pinned
MARGIN_CALIBRATION_V1 ($(fnv1a64:71806d5bf19c090a), 24 cases) and
MARGIN_DEVELOPMENT_V1 ($(fnv1a64:f06fcc3e2d12ca99), 24 cases) partitions; no
device or HTP run contributed data.

## Checkpoint-selection objectives (12 variants, all REJECT)

All four reference runs were regenerated from the CPU reference in this goal
and reproduce the canonical margin bundle exactly (consistency.csv): pinned
selected steps 16/4/12/4, selected and final exact counts identical to the
canonical configuration.csv, and bitwise loss and gradient parity across all
320 steps of every run.

| Configuration | Regenerated trajectory |
| --- | --- |
| L19 seed 1 | 14/144 (0/24 seq) at step 16 -> 30/144 (2/24 seq) at step 320 |
| L19 seed 2 | 20/144 (0/24 seq) at step 4 -> 63/144 (6/24 seq) at step 320 |
| L19 seed 4 | 22/144 (0/24 seq) at step 12 -> 46/144 (6/24 seq) at step 320 |
| L18 control | 18/144 (0/24 seq) at step 4 -> 65/144 (8/24 seq) at step 320 |

Twelve checkpoint-selection objectives were scored on the 23-step cadence
(0..320) per configuration (objective-scores.csv) and cross-validated by
leave-one-seed-out (leave-one-seed-out.csv). Objective/quality correlations
reach 0.95 (LOWER_TAIL_MARGIN_Q20 vs development token exact on seed 4); the
best-ranked variant by the preregistered pooled-evidence comparator is
MARGIN_DEFICIT_MEAN_D0 (Spearman 0.83 vs development token exact on seed 1).
Every objective fails the preregistered development gate in
development-gate.csv: no variant reaches a
single supported seed, pooled token/sequence exact are non-worse in no
variant, the L18 control is worse, first-error median survival does not
improve, and every LOSO fold collapses (mean token delta -59, collapse-free
false in all three folds). The best variant MARGIN_DEFICIT_MEAN_D0 selects an
early low-exact checkpoint (51 tokens below the trajectory-best checkpoint and
43 below the final checkpoint on seed 1); margin quality and final-checkpoint
quality do not co-select.

Gradient attribution (gradient-attribution.csv, 4x320 steps, per-step loss and
gradient parity true) classifies the driver of the margin/quality decoupling
as OUTPUT_HEAD_RANKING_DRIFT: the output-projection gradient share rises
while the critical-token share of the loss falls below the
critical-underweight threshold. Decision: CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT.

## Training families (2 preregistered families, 12 runs, both REJECT)

Two margin-aware loss families were implemented host-only as a verbatim copy
of the CPU training reference (bitwise parity of the backward pass with the
CE dLogits is enforced at cadence 32 in every run; last parity step 320 in
all runs):

- PAIRWISE_MARGIN_CE_V1 (delta=0.5, lambda=1.0): CE plus a pairwise hinge on
  deficient target-vs-competitor margins.
- SEQUENCE_WORST_MARGIN_CE_V1 (tau=0.5, lambda=1.0): CE plus
  lambda*tau*logsumexp(-margin/tau) over the sequence's worst margin.

Micro smoke (4 runs, 64 steps) confirmed mechanics (finite, margin improving,
parity holds). The full gate (8 runs, 320 steps) rejects both families
(training-family-gate.csv, training-family-decision.csv):

| Family | Finite | Improved seeds (of 3) | Control non-worse | Margin improved | Stability | Pooled token delta | Pass |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| PAIRWISE_MARGIN_CE_V1 | true | 2 | false | true | true | 67 | false |
| SEQUENCE_WORST_MARGIN_CE_V1 | false | 1 | true | false | false | -61 | false |

PAIRWISE improves two of three L19 seeds (30->69, more than doubling, and
46->79) with margins and NLL improving, but regresses seed 2 (63->58) and
the L18 control (65->62), failing the preregistered control-non-worse and
2-of-3-seeds conditions. SEQUENCE_WORST at tau=0.5 is unstable: the worst-margin
term keeps pushing easy train margins and diverges the dev NLL (NOT_FINITE in
all four runs). Conclusion: NO_TRAINING_FAMILY_ACCEPTED; the canonical
AR-selected checkpoints remain the final candidate and the final holdout
partition stays unopened (no candidate change, so no HTP smoke or formal run
was scheduled in this goal).

All CSVs here are derived metrics only (exact counts, NLL, margins, gradient
norms, gate decisions); raw checkpoints, parameter payloads, device
identifiers, endpoints, paths, and log streams are excluded. Report roots and
commands for reproduction: scripts/run_critical_margin_objective_benchmark.ps1
(objective probe, -Train, -Micro) and scripts/export_public_qnn_l19_critical_margin_results.ps1.