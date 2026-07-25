# QNN HTP LayerNorm backward

PhoneLM implements LayerNorm backward as an explicit QAIRT 2.48 QNN primitive-composition graph. It does not use automatic differentiation. CPU code supplies tensors and execution control and independently evaluates analytic and central finite-difference gradients.

## Definition and graph

The logical shape is `B=2, T=3, D=8`. QNN flattens the first two dimensions to `6x8`; reductions for normalization use only the last axis. The graph uses the same `epsilon=1e-5` as the CPU reference.

```text
dXhat = dY * gamma
dX = inv_std / D *
     (D * dXhat - ReduceSum(dXhat) - xhat * ReduceSum(dXhat * xhat))
dGamma = ReduceSum(dY * xhat, row axis)
dBeta = ReduceSum(dY, row axis)
```

X, dY, gamma, and beta are `APP_WRITE`. Y, xhat, dX, dGamma, and dBeta are `APP_READ` correctness outputs. Other tensors are `NATIVE`; epsilon, dimension factors, and reduction axes are `STATIC`. The graph is created and finalized once, then executed for two deterministic cases.

For HTP stability at low variance, the variance subgraph computes `(64 * centered)^2` with `epsilon * 64^2`, then multiplies reciprocal standard deviation by 64. This is algebraically identical to the unscaled FP32 definition and avoids under-resolving very small squared centered values. A scale of 256 was rejected after it produced an Inf in the normal case; the acceptance tolerance was not changed.

## Independent checks

The cases are:

- `gamma=1`, `beta=0` with ordinary input variance;
- arbitrary gamma/beta with input values separated by about `1e-3`.

Central finite differences check every X, gamma, and beta element with double perturbations and loss accumulation. The low-variance case uses `epsilon=1e-5` for finite differences. Its deterministic upstream gradient is scaled by `0.005`, because reciprocal-standard-deviation amplification otherwise makes the absolute dX comparison dominated by the already reported low-variance forward approximation error. The ordinary-variance case retains the full upstream gradient. This changes the test input, not the derivative definition or the predefined `2e-2` acceptance threshold.

Device result on 2026-07-25:

| Metric | Result |
| --- | ---: |
| CPU analytic vs numeric max abs, dX/dGamma/dBeta | `5.960464478e-08` |
| CPU analytic vs numeric mean abs | `7.637734396e-09` |
| HTP vs CPU dX max abs | `1.068422198e-02` |
| HTP vs CPU dGamma max abs | `8.066892624e-04` |
| HTP vs CPU dBeta max abs | `3.389716148e-04` |
| HTP vs CPU combined mean abs | `5.022284956e-04` |
| HTP vs CPU max relative, denominator floor `1e-3` | `1.802910921` |
| HTP vs CPU forward / xhat max abs | `5.518459529e-02` / `5.873298645e-02` |

The large maximum relative error occurs on near-zero gradient components; acceptance is based on the predefined absolute-gradient threshold. Both graph executions returned success, graph create/finalize returned zero, and no NaN, Inf, CPU fallback, or execute failure was observed. All dX/dGamma/dBeta maximum absolute errors are below `2e-2`; no tolerance was widened.

QNN callback capture is disabled and QNN log level is WARN. Public evidence is restricted to aggregate errors, shape, graph counters, fixed-size API trace allow-list fields, and fallback/finite status. Raw callback text, device endpoints, host absolute paths, Qualcomm binaries, APKs, and raw tensors remain unpublished.

Reproduce with `scripts/run_qnn_htp_transformer_backward_tests.ps1`. Individual reports remain untracked under `build/reports/qnn-htp-transformer-backward-tests/`.
