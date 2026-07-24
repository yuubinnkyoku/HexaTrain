# Tiny Transformer status

## Forward milestone

The implemented block is B=1, T=4, D=16, one head, FFN=32, one layer, and ReLU:

`pre-LayerNorm -> QKV projections -> causal attention -> output projection -> residual -> pre-LayerNorm -> FFN -> residual`

PhoneLM creates one fused QNN graph. HTP performs every block operation and one `graphExecute`; CPU code only creates deterministic inputs/weights and computes an independent reference. There is no CPU fallback.

Device result: maximum absolute error 0.001065, mean absolute error 0.000252, graph finalize/execute success, one graph boundary, one execute, and no NaN/Inf. Acceptance limits are max absolute error < 0.02 and mean absolute error < 0.005, leaving margin for accumulated HTP floating-point rounding across 18 nodes.

## Training status

Backward microtests, embedding, cross entropy, optimizer integration, a tiny language model, and multi-step loss reduction are `NOT_REACHED`. QNN automatic differentiation is not assumed. Any future backward path must be implemented explicitly and checked against analytic and finite-difference CPU references.