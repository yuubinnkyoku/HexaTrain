# QNN HTP tiny language model training

## Scope and model

PhoneLM extends the existing tiny causal Transformer into a next-token model with `B=1`, `T=8`, `V=32`, `D=16`, one head, one pre-norm layer, `FFN=32`, ReLU, and FP32. Token and output-projection weights are separate trainable matrices. Position embeddings are fixed sinusoidal values, so they have no gradient or optimizer state.

The CPU converts token IDs to row-major one-hot `[B*T,V]` tensors. HTP evaluates `[B*T,V] @ [V,D]`, adds `[B*T,D]` position values, executes the causal Transformer, and evaluates `[B*T,D] @ [D,V]` logits. Targets are the same sequences shifted by one token; no future token is included in an input row.

## Cross entropy boundary

QAIRT 2.48 HTP executes Softmax and the mathematically exact cross-entropy logit gradient `(P-Y)/(B*T)`. The published scalar loss is calculated on CPU with max-shifted log-sum-exp because a verified HTP Log/LogSoftmax path was not available in the local headers and device-finalized graph. This is explicit backward code, not QNN automatic differentiation.

The CPU also constructs token/target one-hot tensors, selects batches, controls graph execution, copies each separate next-parameter buffer into the fixed current-parameter buffer, computes the independent reference, and performs inference ArgMax/context updates. HTP performs embedding arithmetic, Transformer forward and backward arithmetic, logits, Softmax, `dLogits`, all listed parameter gradients, and SGD next-parameter arithmetic. The QNN CPU backend is not initialized and no backend fallback occurs.

## CPU reference and finite differences

The independent CPU implementation uses stable cross entropy and explicit backward for token embedding, Wq/Wk/Wv/Wo, both LayerNorm gamma/beta pairs, FFN W1/W2, and output projection. The micro configuration is `V=8, B=1, T=3, D=4, FFN=8`; it checks three deterministic elements per parameter with central differences at `epsilon=1e-3`, excluding ReLU discontinuities. Acceptance uses absolute error for very small gradients; the complete per-element evidence remains in ignored `build/reports` output.

| Parameter | Maximum absolute error |
|---|---:|
| token embedding | 4.597194e-5 |
| output projection | 6.736093e-5 |
| Wq | 6.450236e-6 |
| Wk | 7.398373e-6 |
| Wv | 5.038735e-5 |
| Wo | 6.813882e-5 |
| Norm1 gamma | 8.812931e-5 |
| Norm1 beta | 6.369225e-5 |
| Norm2 gamma | 1.121727e-4 |
| Norm2 beta | 9.832258e-5 |
| FFN W1 | 2.330879e-5 |
| FFN W2 | 7.293542e-5 |

## Device results

The CE microtest covers normal, large-positive, large-negative, target-maximum, target-minimum, and equal-logit rows. Its maximum probability error is `0.001181424`, maximum `dLogits` error is `0.0001964271`, probability row-sum error is `0.001517773`, and gradient row-sum error is `0.0002674403`.

One-step comparison reports loss error `2.145767e-6`, embedded-input error `9.828806e-5`, logits error `2.664514e-5`, probability error `7.482618e-5`, `dLogits` error `3.192574e-5`, embedding-gradient error `9.597372e-5`, output-projection-gradient error `2.519973e-5`, aggregate major-gradient error `0.0001563841`, and next-parameter error `5.939603e-5`. Major weights change and every checked output is finite.

Five seeds use 320 updates at learning rate `0.01`. The train set cycles four non-constant deterministic rules over tokens 0 through 12; evaluation uses held-out phase offsets. All five seeds reduce evaluation loss and increase evaluation accuracy, with exact same-seed trajectory replay. One graph is created and finalized, and one execute is issued per update. Current and next parameters never alias.

The additional convergence threshold is not reached: median evaluation loss reduction is `6.060655%` rather than 20%, and only one of five seeds reaches 75% evaluation accuracy. This remains a partial-success result after three bounded tuning campaigns within the requested limits. No performance benchmark was run after this result because performance measurement was gated on full convergence success.

Autoregressive verification trains the known `0,1,2,3` rule and reproduces `0,1,2,3,0,1,2,3`. HTP generates logits; CPU takes ArgMax and updates context.

## Reproduction and publication

Run `scripts/run_host_tests.ps1` for CPU reference checks and `scripts/run_qnn_htp_tiny_language_model_tests.ps1` for the four device modes. Raw device reports stay under ignored `build/reports`. `scripts/export_public_tiny_language_model_results.ps1` reads only a fixed key allow-list and emits aggregate JSON/CSV; it never publishes callback/logcat text, device endpoints, private paths, function addresses, APKs, Qualcomm binaries, or full tensors.

The follow-up autoregressive-gap investigation is documented in
[`qnn-tiny-language-model-autoregressive-gap.md`](qnn-tiny-language-model-autoregressive-gap.md).
It adds true sliding-window CPU diagnostics, five-seed CPU/HTP same-prefix parity,
oracle/free-running separation, and pattern-balanced phase sampling without changing
the model shape. CPU reaches 20/20 exact rollouts, while the final HTP candidate
does not remain finite across five seeds; the bounded run reaches 4/20 exact
rollouts, so the published outcome remains `GOAL_PARTIAL_SUCCESS`.
