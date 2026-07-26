# QNN HTP tiny language model numerical stability

> July 2026 reproducibility update: the earlier convergence result below is a
> historical run, not a stable cross-process guarantee. A new headless
> fixed-state study found intermittent full-graph variation beginning at
> `embedding_input_gradient` / `lm_dinput` in one of five fresh processes, while
> the standalone `lm_dembedding` micrograph and fixed Adam graph remained
> deterministic. Formal phase01 repeats at learning rate 0.0003 and clip 10 did
> not keep all 25 process/seed trajectories finite. See
> [QNN HTP fixed-state reproducibility](qnn-htp-fixed-state-reproducibility.md).

## Result

This follow-up reached `GOAL_PARTIAL_SUCCESS`. Five independent seeds completed
1,000 Adam updates without NaN, Inf, or QNN CPU fallback, and every seed improved
evaluation loss and accuracy. The median evaluation-loss reduction was 54.7896%;
one seed reached evaluation accuracy 75% or higher.

The remaining success gate was not met: autoregressive continuation reproduced
0/4 complete rule sequences rather than at least 3/4. The training stability and
convergence result is therefore retained, but the result is not promoted to full
success.

The model is unchanged:
`B=1, T=8, V=32, D=16, heads=1, layers=1, FFN=32`, pre-norm ReLU, causal FP32.
The training and evaluation phases remain disjoint.

## Synchronized diagnosis

The CPU and HTP trajectories were compared at steps
0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, and 320. Checkpoint
serialization and restoration passed. At each checkpoint, four paths isolated
the gradient and optimizer:

| Path | Gradient | Adam update |
|---|---|---|
| A | CPU | CPU |
| B | HTP | CPU |
| C | CPU | HTP |
| D | HTP | HTP |

The first major divergence was at step 10 in `token_embedding_gradient`, produced
by `lm_dembedding`. The CPU-gradient/HTP-Adam path C remained bounded, while the
HTP-gradient paths separated from the CPU reference. At step 100, the extreme
HTP gradient also caused optimizer-path divergence. The free HTP trajectory's
last finite step was 121; step 122 first exposed a non-finite
`adam_denominator`.

The initiating defect is therefore repeated-update HTP backward drift, with a
secondary Adam denominator interaction after gradients have already become
extreme. Adam's formula, step number, bias-correction binding, and state-buffer
handoff were not the initiating cause.

The audit also found and fixed a LayerNorm-backward graph-construction defect:
two element-wise products had shared one output tensor, creating duplicate
producers. Separate tensors now represent `dxhat*xhat` and `dy*xhat`. That repair
was necessary for a valid graph but did not alone prevent late divergence.

## Stabilization

The adopted configuration is Adam with learning rate 0.0003, beta1 0.9, beta2
0.999, epsilon `1e-8`, 1,000 steps, and global gradient-norm clipping at 10.
For gradient vector `g`, CPU computes only:

```text
scale = min(1, 10 / (sqrt(sum(g²)) + 1e-6))
```

HTP applies `g * scale` and performs all Adam moment and parameter arithmetic.
Across five seeds, clipping was active for 1,792 updates; the minimum scale was
0.0418975 and the maximum pre-clip norm was 238.678.

The two finalists were:

| Configuration | Finite seeds | Median evaluation-loss reduction | Seeds at accuracy >= 75% |
|---|---:|---:|---:|
| `lr=0.0003`, clip 5, 1,000 steps | 5/5 | 59.4520% | 0/5 |
| `lr=0.0003`, clip 10, 1,000 steps | 5/5 | 54.7896% | 1/5 |

Clip 10 was selected because it passed every finite and directional convergence
gate and produced the strongest accuracy result. A clip of 1 at the same
learning rate also stayed finite but had lower median reduction (49.9699%).
Learning rate 0.003 with clips 5 or 1 remained non-finite; learning rate 0.001
with clip 1 left one seed non-finite. Experimental update clipping did not stop
the failure and was rejected. Epsilon changes and learning rate 0.0001 were not
needed once the synchronized paths isolated the cause.

Per-seed final evaluation:

| Seed | Loss reduction | Accuracy | Correct-token probability | Entropy |
|---:|---:|---:|---:|---:|
| 1 | 71.1097% | 0.78125 | 0.413651 | 1.58902 |
| 2 | 54.7896% | 0.53125 | 0.264449 | 1.72493 |
| 3 | 53.1491% | 0.46875 | 0.260236 | 1.78078 |
| 4 | 60.0071% | 0.46875 | 0.289395 | 1.75056 |
| 5 | 43.4097% | 0.37500 | 0.266151 | 1.59262 |

## Autoregressive inference

HTP computes logits; CPU performs argmax, context shifting, and next-token
one-hot construction. All logits stayed finite. Exact continuation nevertheless
failed all four patterns:

| Rule | Token accuracy | Mean expected-token probability | Minimum margin |
|---|---:|---:|---:|
| 0–3 repeating | 0.375 | 0.226238 | -2.18701 |
| 4–7 repeating | 0.625 | 0.329697 | -9.23047 |
| 8/9 alternating | 0.000 | 0.318237 | -0.466797 |
| 10–12 repeating | 0.250 | 0.194562 | -3.28735 |

This is a capability failure, not a numerical failure: no inference NaN or Inf
was observed.

## Runtime boundary and performance

The implementation creates two QNN graphs and executes two graphs per update.
CPU responsibilities are fixed-buffer handoff, graph control, stable
cross-entropy reporting, Adam bias-correction scalars, and the single global
clip scalar. Forward, backward, gradient scaling, and Adam tensor arithmetic
execute on HTP. The QNN CPU backend is not initialized.

Measured estimates were 102,879 us graph initialization, 129,856 us graph
creation, 103,674 us graph finalization, 1,349 us first execute, and 1,743 us
steady-state mean execute. The two-execute update estimate is 286.927
updates/second or 2,295.42 tokens/second.

## Regression verification

The final APK build and QAIRT 2.48 APK audit passed. Device probe, forward,
force-stop cycles, Skel reuse, app-private Skel corruption recovery, linear
training, MLP split/fused/full-step, Transformer forward, Softmax backward,
attention backward, LayerNorm backward, Transformer MSE training, tiny-LM
cross-entropy/SGD, Momentum one-step, Adam diagnostics, and the pre-existing
tiny-LM inference all passed. Host reference and Gradle unit tests also passed.

The API-trace A/B test completed ten runs per condition for all three shapes,
including injected execute/finalize failures, recovery, and fixed-size trace
bounds. It reported success with no CPU fallback. Device checks stayed at 34°C
and thermal status 0 during the final regression runs.

Public, allowlisted aggregates are under
`docs/results/qnn-htp-tiny-language-model-stability-2026-07`. They are generated
by `scripts/export_public_tiny_lm_stability_results.ps1`. Raw reports remain
ignored under `build/reports`; device endpoints, private paths, binaries, APKs,
callback output, and raw tensor dumps are excluded.
