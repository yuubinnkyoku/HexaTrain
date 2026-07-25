# QNN HTP scaled dot-product attention backward

PhoneLM implements single-head causal scaled dot-product attention backward as an explicit QAIRT 2.48 QNN graph. It does not use automatic differentiation. CPU code supplies tensors and execution control and independently evaluates analytic and central finite-difference gradients.

## Correctness graph

The shape is `B=1, H=1, T=4, D=8`, with `scale=1/sqrt(8)`. The graph performs the existing forward definition and then:

```text
dP = dO V^T
dV = P^T dO
dScores = P * (dP - ReduceSum(dP * P, last axis, keep_dims=true))
dQ = (dScores K) * scale
dK = (dScores^T Q) * scale
```

Q, K, V, dO, and the causal mask are `APP_WRITE`. P, dScores, dQ, dK, and dV are `APP_READ` correctness outputs. Other tensors are `NATIVE`; the scalar scale and reduction axis are `STATIC`. The graph is created and finalized once and all forward/backward operations execute in one graph call. The mask is a constant input to the mathematical function under test and no mask gradient is requested.

## Independent checks

The CPU analytic implementation uses the same causal definition but independent loops. Central finite differences use `epsilon=1e-3`, double perturbations, double Softmax evaluation, and the scalar loss `sum(O * dO)`. Every element of Q, K, and V is checked (96 derivatives total).

Device result on 2026-07-25:

| Metric | Result |
| --- | ---: |
| CPU analytic vs numeric max abs, all Q/K/V | `5.960464478e-08` |
| CPU analytic vs numeric mean abs | `2.097901112e-09` |
| CPU dQ / dK / dV numeric max abs | `3.725290298e-09` / `3.725290298e-09` / `5.960464478e-08` |
| HTP vs CPU max abs, dQ / dK / dV | `5.085021257e-05` / `1.049870625e-04` / `6.293058395e-04` |
| HTP vs CPU mean abs, combined gradients | `7.824714339e-05` |
| HTP vs CPU max relative, denominator floor `1e-3` | `4.998082295e-02` |
| HTP vs CPU dScores max abs | `2.437829971e-04` |
| HTP vs CPU probability max abs | `1.464843750e-03` |
| Future probability / future dScores max | `0` / `0` |

The QNN graph create, finalize, and execute results were zero, with one successful execute and no execute failure. No NaN, Inf, or CPU fallback was observed. Each dQ/dK/dV maximum absolute error is below the predefined `1e-2` threshold; no tolerance was widened.

QNN callback capture is disabled and log level is WARN. Public evidence is limited to aggregate errors, shape, graph counters, fixed-size API trace allow-list fields, and fallback/finite status. Raw callback text, device endpoints, host absolute paths, Qualcomm binaries, APKs, and raw tensors remain unpublished.

Reproduce with `scripts/run_qnn_htp_transformer_backward_tests.ps1`. Individual reports remain untracked under `build/reports/qnn-htp-transformer-backward-tests/`.
