# QNN HTP next-token cross entropy

## Result

QAIRT 2.48.40 HTP finalize and execute succeeded for a `B=2, T=3, V=8` cross-entropy-gradient microtest. The six rows cover normal logits, large positive logits, large negative logits, correct class at the maximum, correct class at the minimum, and equal logits.

| Metric | Result | Limit |
|---|---:|---:|
| CPU stable cross-entropy scalar | 4.669507095 | reference |
| probability max absolute error | 0.001181424 | 0.005 |
| dLogits max absolute error | 0.000196427 | 0.005 |
| probability row-sum max error | 0.001517773 | 0.005 |
| dLogits row-sum max error | 0.000267440 | 0.005 |
| NaN / Inf | none | none |
| CPU fallback | false | false |

## Boundary

- HTP computes Softmax and the exact cross-entropy gradient `(P - Y) / N` using explicit QNN graph nodes.
- CPU computes only the published scalar loss with stable logsumexp.
- CPU also prepares target one-hot values and controls execution.

The scalar loss is therefore not claimed as HTP output. API trace recorded one successful HTP graph execute, zero execute failures, and no CPU backend initialization or fallback attempt.

Raw local evidence is kept under `build/reports/` and is not tracked.
