# QNN HTP Softmax backward

PhoneLM implements Softmax backward as an explicit QAIRT 2.48 QNN graph. It does not use automatic differentiation. The CPU supplies inputs, invokes the graph, and independently computes the analytic and finite-difference references.

## Graph and shape

The logical shape is `B=1, H=1, T=4, D=4`. The QNN graph flattens `B/H/T` to four rows and keeps `D=4` as the last axis, so the registered tensor shape is `4x4` and the reduction axis is `1`.

```text
product = dS * S
dot = ReduceSum(product, axis=1, keep_dims=true)
centered = dS - dot
dX = S * centered
```

`S` and `dS` are `APP_WRITE`; `dX` is `APP_READ`; intermediates are `NATIVE`. The correctness graph is created and finalized once, then executed once for each of five deterministic cases.

## Gradient check

The scalar loss for the central finite difference is `sum(softmax(X) * dS)`. Perturbations and loss accumulation use double precision so that `epsilon=1e-3` remains exact enough for the cases near `+/-1000` and the causal-mask sentinel. The analytic path uses the FP32 tensors supplied to QNN.

Device result on 2026-07-25:

| Metric | Result |
| --- | ---: |
| CPU analytic vs central difference max abs | `2.980232239e-08` |
| CPU analytic vs central difference mean abs | `5.784024427e-09` |
| HTP vs CPU max abs | `2.492368221e-04` |
| HTP vs CPU mean abs | `2.604658577e-05` |
| HTP vs CPU max relative, denominator floor `1e-3` | `7.265186287e-02` |
| Maximum absolute row sum of `dX` | `1.831054688e-04` |

The cases are normal small values, large positive values, large negative values, equal-valued rows, and a causal-mask distribution. All five graph executions returned success. No NaN, Inf, CPU fallback, or QNN graph-execute failure was observed. The maximum absolute HTP error is below the predefined `5e-3` threshold; the threshold was not relaxed.

QNN callback capture is disabled and QNN log level is WARN. The normal fixed-size API trace records real graph execute return codes and `dladdr` basenames. Raw callback text, device endpoints, absolute host paths, Qualcomm binaries, APKs, and raw tensor dumps are not published.

Reproduce with `scripts/run_qnn_htp_transformer_backward_tests.ps1`. Individual device reports remain untracked under `build/reports/qnn-htp-transformer-backward-tests/`.
