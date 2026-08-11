# Nicopedia L19 HTP-native T64-context training, August 2026

This public bundle contains aggregate-only evidence for canonical L19
training with a 64-token context (T64), optimizer step 0 through 8,000. The
numerical operations of each training step ran in explicit QNN HTP graphs;
host control and input preparation remained on CPU. This is not an NPU-only
claim and QNN automatic differentiation was not used.

The best and final checkpoint is step 8,000. The 250-step CPU evaluator
screening shows held-out validation NLL improving from 3.0133 at step 250 to
2.3487 at step 8,000; development NLL improves from 2.6632 to 2.0544. The
full-cap HTP-native validation at step 8,000 is 2.150418944 (CPU
2.150434198) and development is 2.131013825 (CPU 2.131041532), measured over
12,288 graph executes with zero failures, zero nonfinite chunks, and no CPU
fallback.

Training stopped at the predeclared hard ceiling of 8,000 steps, not because
a plateau was proven. Every 1,000-step segment resumed from the previous
segment's NPRTCKPTV2 checkpoint; the checkpoint parameter hashes, steps, and
config identity were verified on-device and on-host at each boundary. The
T64 run is a context-extension milestone and does not by itself change the
legacy CPU-equivalence parity thresholds.

Files:

- `t64-full-cap-evaluation.csv`: HTP-native and CPU held-out metrics at the
  final checkpoint.
- `t64-trajectory.csv`: 250-step CPU evaluator screening results.
- `t64-segment-health.csv`: resume segment health and runtime aggregates.
- `manifest.json`: scope, policy, and final-test status.

Private dataset text, prompts, generated content, raw bytes/tokens, logits,
checkpoints, device identifiers, endpoints, and local paths are excluded.
Nicopedia final test and the synthetic final holdout were not opened.
