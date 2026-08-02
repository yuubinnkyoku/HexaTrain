# Autoregressive validation quality, August 2026

This bundle records a CPU-gated investigation of the finite, seed-dependent
autoregressive quality shortfall at T8/D16/FFN32/L19/H2. It does not claim an
HTP numerical failure: the prior evidence classifies the issue as finite
autoregressive generalization quality.

AR_ROLLOUT_NLL_V1 starts from each initial prefix, feeds each argmax token
back into the next context, and scores the known target token at every rollout
position. The accompanying metrics record rollout NLL, token and sequence
exact counts, first-error position, recovery after an error, teacher-forced
NLL, and their gap. Checkpoint ranking is fixed before evaluation: lower
rollout NLL (tolerance 1e-7), then higher token exact, then higher sequence
exact, then earlier step.

The fixed partitions are deterministic. TRAIN has four homogeneous phase-zero
patterns. Each fresh partition has 24 mixed-prefix cases spanning four pattern
families, suffix lengths 3/4/5, and rollouts 4/8. Case IDs, initial prefixes,
and complete sequences have zero overlap across all four partitions. The
learned successor-transition overlap is structural and reported precisely in
dataset-overlap.csv; it is not represented as zero.

AR_VALIDATION_V3 selects a checkpoint, AR_DEVELOPMENT_V3 decides whether that
selection is predictive, and AR_FINAL_HOLDOUT_V3 remains unopened unless the
development gate passes. CPU reference regeneration ran exactly once each for
L19 seeds 1, 2, and 4 and L18 seed 2 control. Existing legacy checkpoint
replay supplied 56 of the requested 92 replay entries; 36 requested entries
were unavailable from the stored cadence. The stored state includes Adam
moments, but it does not unambiguously reconstruct historical training loss,
gradient, or update records. Those replay trajectory fields are therefore
marked NOT_AVAILABLE rather than inferred; parameter norm is available.

Selected steps were L19 seed 1: 16, seed 2: 4, seed 4: 12, and L18 control:
4. Although all development evaluations were finite, the selected checkpoints
did not meet the predeclared development gate against FINAL_STEP:

| Development comparison | Selected rollout NLL | Final rollout NLL | Selected token / sequence exact | Final token / sequence exact |
| --- | ---: | ---: | ---: | ---: |
| L19 seed 2 | 3.197133 | 4.183425 | 20/144; 0/24 | 63/144; 6/24 |
| L19 pooled seeds 1, 2, 4 | 3.185107 | 6.631546 | 56/432; 0/72 | 139/432; 14/72 |
| L18 seed 2 control | 3.21603 | 5.302605 | 18/144; 0/24 | 65/144; 8/24 |

The lower selected NLL did not translate into the required token or sequence
quality: seed 2 did not strictly improve, pooled L19 and control non-worsening
requirements did not hold, and the required multi-seed support was absent.
The decision is therefore AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE.

Consequently AR_FINAL_HOLDOUT_V3 was not opened, checkpoint selection was not
adopted, no HTP smoke or five-seed formal run started, and no thermal device
measurement was taken. All such rows use NOT_RUN_GATE_REJECTED; they are not
passes. No stabilizer was selected because the preregistered trajectory review
did not establish a concrete, general causal candidate. The legacy FINAL_STEP
baseline remains Oracle/Free 13/20; this bundle does not use it to select a
candidate.

The public files deliberately exclude model-state payloads, package material,
device identifiers, endpoint data, paths, and log streams. FNV-1a partition
identifiers are determinism checks, not cryptographic authenticity claims.