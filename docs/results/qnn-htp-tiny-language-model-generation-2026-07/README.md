# QNN HTP tiny language-model autoregressive-gap results

This directory publishes allow-listed aggregate evidence for same-prefix parity,
pattern-balanced phase sampling, five-seed oracle-prefix rollout, and five-seed
free-running rollout. The result is GOAL_PARTIAL_SUCCESS. For the finite representative
same-prefix evaluation, evaluation/generation forward parity is exact within each
backend. All five phase01 HTP training states were attempted, but only finite
same-prefix results are comparable; some training/evaluation and generation outputs
were nonfinite. The bounded rollout audit detected this in
9 independently bounded rollouts.
HTP produced 4 of 20 exact free-running rollouts and 4 of 20
exact oracle-prefix rollouts; interrupted rollouts are counted as non-exact.

Raw callback output, logcat, device endpoints, binaries, APKs, weights, optimizer
state, host paths, and app-private paths are intentionally excluded.
