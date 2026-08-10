# Nicopedia L19 HTP-native long training, August 2026

This public bundle contains aggregate-only evidence for canonical L19
training from optimizer step 2,000 through 8,000. The numerical operations
of each training step ran in explicit QNN HTP graphs; host control and input
preparation remained on CPU. This is not an NPU-only claim and QNN automatic
differentiation was not used.

The best and final checkpoint is step 8,000. Full-cap HTP-native validation
NLL improved from 2.419448562 at step 1,000 to 2.168420875; development NLL
improved from 2.403955266 to 2.146852226. All full-cap evaluations used
24,576 graph executes, zero failures, zero nonfinite chunks, and no CPU
fallback. The corresponding step-8,000 CPU values were 2.168439351 and
2.146876079.

Training stopped at the predeclared hard ceiling, not because a plateau was
proven. Improvement per 1,000 steps was diminishing but remained positive.
The legacy CPU-equivalence parity thresholds were unchanged and still reject
generation. Experimental HTP-native generation used a separate health gate;
only aggregate byte-quality metrics are included here.

Files:

- `full-cap-evaluation.csv`: HTP-native and CPU held-out metrics.
- `medium-trajectory.csv`: 250-step CPU evaluator screening results.
- `segment-health.csv`: resume segment health and runtime aggregates.
- `generation-aggregates.csv`: no prompts or generated content.
- `regression-health.csv`: synthetic scale-formal and FFN372 finite checks.
- `manifest.json`: scope, policy, and final-test status.

Private dataset text, prompts, generated content, raw bytes/tokens, logits,
checkpoints, device identifiers, endpoints, and local paths are excluded.
Nicopedia final test and the synthetic final holdout were not opened.

The synthetic scale-formal headless regression passed for five seeds with
3,573/3,573 QNN executes. The FFN372/L3/H4 COUNT_FROM_ONE seed-5 regression
also reached `SUCCESS`, 14,088 executes, zero nonzero returns, and finite
training/evaluation. Its legacy host command timed out while the device run
continued, so the terminal report was recovered without relaunching. The
EXACT_SEED comparison half was not rerun in this milestone; no new direct-seed
equivalence claim is made.
