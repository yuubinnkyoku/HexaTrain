# QNN HTP autoregressive validation

## Scope and classification

This investigation addresses `FINITE_AUTOREGRESSIVE_GENERALIZATION_SHORTFALL`
for T8/D16/FFN32/L19/H2: learning-step numerical operations can remain finite
while held-out free-running generation varies materially by seed. It is not an
HTP overflow claim, QNN execution failure, or hardware depth-limit claim.

The first evaluation classified the checkpoint-selection candidate as
`AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE`. The CPU development gate rejected
it, so AR_FINAL_HOLDOUT_V3 was not opened and no HTP smoke, formal five-seed,
or stabilizer run was started. The public evidence is in
[`qnn-htp-autoregressive-validation-2026-08`](results/qnn-htp-autoregressive-validation-2026-08/README.md).

## Partition contract

`AR_ROLLOUT_NLL_V1` fixes four deterministic collections before a candidate is
evaluated:

| Collection | Role | Cases |
| --- | --- | ---: |
| TRAIN | Learning patterns | 4 |
| AR_VALIDATION_V3 | Checkpoint ranking | 24 |
| AR_DEVELOPMENT_V3 | CPU adoption decision | 24 |
| AR_FINAL_HOLDOUT_V3 | One-time final evaluation after development acceptance | 24 |

Across collections, case IDs, initial prefixes, and complete sequences have
zero overlap. Successor transitions are structurally shared by these compact
cyclic patterns; their unique and occurrence-level overlap is reported rather
than hidden. The partition hashes and full deterministic case definitions are
published in the results bundle.

## Objective and ranking

For each fresh case, evaluation starts at the specified prefix. At every
rollout position it scores the expected token's negative log probability, then
uses the model argmax as the next input token. It records rollout NLL, token
exactness, sequence exactness, first-error position, recovery after an error,
teacher-forced NLL, and the free-running/teacher-forced gap.

Ranking uses this fixed order with a 1e-7 NLL tolerance:

1. Lowest autoregressive rollout NLL.
2. Highest token exact count.
3. Highest sequence exact count.
4. Earlier checkpoint step.

All 320 training steps still run. The candidate is eligible only when its
development checkpoint is finite, improves the L19 seed-2 development result,
does not worsen the L19 aggregate or L18 control, introduces no per-case
collapse, and has multi-seed support. Final holdout data are never used to
choose or break a candidate tie.

## CPU result and next action

The single CPU regeneration for each intended case selected steps 16, 4, 12,
and 4 for L19 seeds 1, 2, and 4 and L18 seed 2 respectively. Development did
not meet the fixed acceptance conditions. The final collection therefore
remains unopened; all device/formal fields are explicitly recorded as
`NOT_RUN_GATE_REJECTED` rather than passed.

No general stabilizer was selected. The preregistered trajectory evidence did
not identify a causal metric and matching intervention that could justify the
limited CPU stabilizer search. Any later candidate must establish that evidence
before touching the unopened final collection.

Legacy FINAL_STEP evidence remains the separate baseline (Oracle 13/20, Free
13/20). It was not reused as an objective-selection test. The public bundle is
allow-listed and excludes checkpoints, model-state payloads, package artifacts,
device identity, endpoint data, local paths, and log streams.
