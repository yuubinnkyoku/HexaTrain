# QNN HTP scaled dot-product attention

The tested graph is single-head B=1, H=1, T=4, D=8:

`QK^T -> scale by 1/sqrt(D) -> add causal mask -> Softmax -> probability*V`

All operations are in one finalized HTP graph and one `graphExecute`; CPU fallback is false. CPU code supplies deterministic Q/K/V and an independent reference only.

Device result: output maximum absolute error 0.000782, probability maximum absolute error 0.001465, maximum row-sum error 0.001465, future-token probability maximum 0, and no NaN/Inf. Acceptance limits are 0.005 for output, 0.003 for probabilities/row sums, and 0.002 for future probabilities.