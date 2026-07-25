# QNN HTP Adam for the tiny language model

Adam uses beta1 0.9, beta2 0.999, epsilon `1e-8`, zero initial first/second moments, and separate current/next buffers. CPU computes only the step-dependent bias-correction scalars. HTP updates `m` and `v`, applies bias correction, square root/division, and updates weights.

The generic flattened optimizer graph contains 3,136 parameters and 6,272 FP32 optimizer-state elements (25,088 bytes). Together with the training/backward graph, the configuration has two graphs and two executes per update. CPU performs fixed-buffer handoff and graph control; this is not a CPU optimizer fallback.

HTP cannot safely evaluate zero divided by an effectively vanished tiny denominator. The finalized implementation uses an algebraically equivalent common scale for numerator and denominator and selects exact zero for entries whose corrected second moment is zero. This preserves the Adam zero-gradient invariant without changing nonzero updates.

One-step device comparison passes:

| Quantity | Maximum absolute error |
|---|---:|
| gradient | 0.00240893 |
| first moment next | 0.000240797 |
| second moment next | 1.38159e-7 |
| first moment hat | 0.00240893 |
| second moment hat | 0.000138159 |
| next weight | 0.00305894 |

The bounded 12-configuration CPU search selected `lr=0.003`/320 steps and `lr=0.001`/640 steps. CPU medians were 94.5603% and 94.3134%, with 5/5 seeds at evaluation accuracy 75% or higher. Both permitted HTP finalists improved before eventually becoming non-finite, so Adam did not meet the additional HTP convergence condition.

Callback capture is disabled, QNN log level is WARN, API trace is enabled, graph calls succeed, and the QNN CPU backend is not initialized. Raw evidence remains under ignored `build/reports`.
