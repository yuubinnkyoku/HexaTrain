# L19 context-supervision stability study, August 2026

This host-only bundle tests whether homogeneous-only training is the upstream
cause of the seed-dependent mixed-context failure while keeping ordinary
learned causal Attention enabled.

The intervention replaces 80 of 320 training batches (25%) with deterministic
mixed prefixes. Every row keeps the original target contract:
`target = successor(current token)`. The intervention therefore teaches
invariance to irrelevant prefix tokens; it does not add a target that requires
useful past context.

For L19 seeds 1/2/4, canonical AR_DEVELOPMENT_V3 free-running token exact is
30/63/46 of 144. A matched homogeneous control with the same 320 steps, 2,560
supervised rows, special-step cadence, family exposure, and aggregate
input/target histogram reaches 65/105/45. The 25% interleaved mixed condition
reaches 144/136/144. Reordering the exact same 80 mixed and 240 homogeneous
batches gives 144/141/144 with mixed-first and 144/144/144 with mixed-last;
both satisfy the preregistered all-seed stability gate. L18 seed 2 reaches
144/144 under both curricula.

Smaller fixed doses fail: mixed-last at 12.5% reaches 112/111/114, and 6.25%
reaches 51/38/61. Thus 25% (80 batches) is the minimum tested effective dose,
not a claim of a global mathematical minimum.

Q/K/V/O all update from initialization in the passing runs. Aggregate
Attention output norm remains nonzero and non-self mass remains broad, so the
result is not caused by disabling Attention or replacing it with a fixed
pattern. No V/O freeze or other parameter intervention is needed.

The official objective has a unique current-token successor in every audited
partition. Consequently, useful-context learning is not independently
identifiable here. The supported causal conclusion is narrower: homogeneous
training leaves behavior on mixed prefixes unconstrained, while target-invariant
mixed supervision removes the observed seed instability. Learning
irrelevant-prefix invariance is the leading mechanism, but no paired-prefix
invariance metric was collected, so it is not uniquely established.

The histogram-matched control does not preserve row-level token position,
co-occurrence, context arrangement, or the resulting gradient geometry. A
regularization or optimization effect of mixed-family co-occurrence therefore
remains an alternative interpretation of why the intervention works. The
study identifies the upstream training-data condition and an effective causal
intervention, but does not uniquely identify invariance as the only internal
explanation.

No device, HTP, QNN, QAIRT, ADB, Android/JNI, UI, or final-holdout evaluation
was used. The CSV files contain allow-listed aggregates only. Raw checkpoints,
parameters, optimizer state, gradients, hidden state, logits, Attention
matrices, local paths, and private run identifiers are not published.

```powershell
.\scripts\run_l19_context_supervision_stability.ps1 -SelfTest
.\scripts\export_public_qnn_l19_context_supervision_results.ps1 -SelfTest
```
