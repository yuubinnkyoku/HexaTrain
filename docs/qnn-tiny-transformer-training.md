# Tiny Transformer status

## Forward milestone

The implemented block is B=1, T=4, D=16, one head, FFN=32, one layer, and ReLU:

`pre-LayerNorm -> QKV projections -> causal attention -> output projection -> residual -> pre-LayerNorm -> FFN -> residual`

PhoneLM creates one fused QNN graph. HTP performs every block operation and one `graphExecute`; CPU code only creates deterministic inputs/weights and computes an independent reference. There is no CPU fallback.

Device result: maximum absolute error 0.001065, mean absolute error 0.000252, graph finalize/execute success, one graph boundary, one execute, and no NaN/Inf. Acceptance limits are max absolute error < 0.02 and mean absolute error < 0.005, leaving margin for accumulated HTP floating-point rounding across 18 nodes.

## Training milestone

PhoneLM now trains the same block with an explicit QNN graph for forward, MSE loss, backward, and SGD. It does not use automatic differentiation. CPU code creates deterministic inputs and teacher targets, computes an independent scalar reference, binds current/next parameter buffers, and controls each step.

The single graph contains both LayerNorm backward paths, causal attention backward, Q/K/V and output-projection gradients, FFN/ReLU backward, residual gradients, and distinct SGD outputs for Norm1 gamma/beta, Wq, Wk, Wv, Wo, Norm2 gamma/beta, W1, and W2. Current parameters are `APP_WRITE`; next parameters are separate `APP_READ` tensors. CPU ping-pong binding changes buffers between executes without in-place aliasing.

One-step device result on 2026-07-25:

| Metric | Result |
| --- | ---: |
| Forward max abs, HTP vs CPU | `9.940862656e-04` |
| Loss abs difference | `2.016779035e-06` |
| dOutput max abs difference | `3.072619438e-05` |
| All gradients max abs difference | `2.213238622e-05` |
| All next parameters max abs difference | `6.799399853e-05` |
| Runtime initialization | `237503.386 us` |
| Graph create / finalize | `77862.395 / 47268.437 us` |
| First execute | `2499.063 us` |

Every gamma, beta, and weight gradient is reported separately; their maximum errors range from `1.800948667e-06` to `2.213238622e-05`. All next-parameter comparisons are also emitted. The predefined `2e-2` update threshold was not changed. Major weights changed, graph create/finalize/execute returned zero, and no NaN, Inf, execute failure, or CPU fallback occurred.

Five deterministic seeds each complete 100 updates with learning rate `0.01` against nonzero teacher-Transformer targets. All five final losses are below their initial values. A first exploratory teacher offset made seed 1's loss (`~6.3e-5`) smaller than the observed HTP forward quantization scale; the teacher offset was increased threefold to strengthen the learning signal, without changing a tolerance. A final evaluation execute follows each seed, producing `505` graph executes total.

QNN callback capture is disabled and log level is WARN. Public evidence is restricted to aggregate errors, loss checkpoints, seeds, timing/counters, fixed-size API trace allow-list fields, and fallback/finite status. Raw callback text, device endpoints, host paths, Qualcomm binaries, APKs, and raw tensors remain untracked.

Reproduce with `scripts/run_qnn_htp_transformer_training_tests.ps1`. Reports remain untracked below `build/reports/qnn-tiny-transformer-training-tests/`. Cross entropy and the tiny next-token language model remain optional follow-up work.
