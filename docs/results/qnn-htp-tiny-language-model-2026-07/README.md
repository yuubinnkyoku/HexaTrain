# QNN HTP tiny language model results — July 2026

Result: `GOAL_PARTIAL_SUCCESS`. The explicit next-token training path, 5-seed loss/accuracy direction, deterministic replay, and autoregressive inference succeeded. The additional convergence threshold did not: median evaluation loss reduction was `6.060655%`, and `1/5` seeds reached at least 75% evaluation token accuracy.

## Configuration

- QAIRT build `2.48.40.260702151143`, QNN API `2.37.0`, HTP V81, FP32.
- `B=1`, `T=8`, `V=32`, `D=16`, one causal head, one pre-norm layer, `FFN=32`, ReLU.
- Separate trainable token embedding and output projection; fixed sinusoidal position embedding.
- Four deterministic non-constant patterns, tokens `0..12`, separate phase-offset evaluation sequences.
- Five seeds, 320 updates, learning rate `0.01`.

## CPU and HTP responsibilities

CPU builds token and target one-hot tensors, supplies batches, controls execution, copies separate next-parameter buffers to fixed current buffers, computes stable log-sum-exp scalar loss and an independent reference, and performs ArgMax/context updates during generation.

HTP performs token-embedding MatMul, position addition, causal Transformer forward, output projection, Softmax, `(P-Y)/N`, explicit Transformer/embedding/projection backward operations, and SGD next-parameter calculations. The QNN CPU backend was not initialized; no fallback, NaN, Inf, or execute failure was observed.

One graph was created and finalized. Each update used one execute; the 5-seed run recorded 1,605 executes including one final evaluation per seed. Current and next buffers did not alias.

## Correctness highlights

- CE microtest `dLogits` maximum absolute error: `0.0001964271`.
- One-step aggregate gradient maximum absolute error: `0.0001563841`.
- One-step next-parameter maximum absolute error: `0.00005939603`.
- 5/5 seeds: final evaluation loss below initial loss and final token accuracy above initial accuracy.
- Same-seed replay: exact published trajectory match.
- Generation: expected `0,1,2,3,0,1,2,3` reproduced; HTP logits and CPU ArgMax.

`summary.json`, `seeds.csv`, `trajectory.csv`, and `gradient-check.csv` are allow-listed aggregate exports. Raw QNN callback/logcat output, endpoints, private paths, binaries, APKs, and tensor/weight dumps are excluded.

## Follow-up convergence study

`convergence-summary.json`, `convergence-sweeps.csv`, and
`htp-convergence-candidates.csv` are allow-listed aggregates from the bounded
SGD, Momentum SGD, and Adam follow-up. The CPU finalists satisfy the additional
convergence condition, but every permitted HTP finalist eventually becomes
non-finite during repeated updates. The follow-up status is therefore
`PARTIAL_SUCCESS`; performance and four-pattern final-optimizer inference are
`NOT_REACHED`.
