# Tiny Transformer HTP training results — 2026-07

Shape: `B=1, T=4, D=16, heads=1, ffn_dim=32`. Loss: MSE. Optimizer: SGD with learning rate `0.01`. Each seed uses a deterministic nonzero teacher Transformer target.

| Seed | Initial | Step 1 | Step 2 | Step 5 | Step 10 | Step 20 | Step 50 | Step 100 | Final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0005372860 | 0.0005393028 | 0.0005393028 | 0.0005378723 | 0.0005373955 | 0.0005359650 | 0.0005316734 | 0.0005254745 | 0.0005249977 |
| 2 | 0.0021292255 | 0.0021286011 | 0.0021286011 | 0.0021228790 | 0.0021133423 | 0.0020942688 | 0.0020484924 | 0.0019836426 | 0.0019817352 |
| 3 | 0.0008112396 | 0.0008115768 | 0.0008120537 | 0.0008111000 | 0.0008072853 | 0.0008044243 | 0.0007953644 | 0.0007824898 | 0.0007824898 |
| 4 | 0.0005956893 | 0.0005974770 | 0.0005970001 | 0.0005960464 | 0.0005941391 | 0.0005893707 | 0.0005817413 | 0.0005702972 | 0.0005702972 |
| 5 | 0.0022284316 | 0.0022335052 | 0.0022239685 | 0.0021972656 | 0.0021648407 | 0.0021038055 | 0.0019588470 | 0.0017566681 | 0.0017518997 |

All five final losses are below their initial losses. Final parameter norms are `6.808044`, `6.806161`, `6.807445`, `6.807463`, and `6.805520`. Per-seed CPU/HTP maximum parameter differences are `0.0011164`, `0.0020607`, `0.0013257`, `0.0010173`, and `0.0034074`.

The HTP graph is created and finalized once. It executes once per training update plus one final evaluation per seed (`505` total). Initialization was `236835.520 us`, graph create/finalize were `97106.302 / 48121.198 us`, first execute was `2457.761 us`, and subsequent QNN execute calls averaged `935.967 us`. The same device-side unoptimized scalar CPU reference averaged `565.262 us`; at this tiny shape the correctness-oriented HTP graph is slower, and this is not an optimized CPU benchmark. Thermal status remained `0`; the surrounding formal device regressions reported `36–38 C`.

No NaN, Inf, graph execute failure, or CPU fallback was observed. CPU responsibility is deterministic input/target generation, independent reference computation, ping-pong binding, and execution control; HTP responsibility is Transformer forward, MSE, explicit backward operations, and SGD.

## Regression status

The final APK build and audit, host tests, JVM unit tests, QNN probe/forward, Skel reuse, force-stop cycles, linear training, HTP dW/backward, split MLP, fused backward, MLP full-step, Transformer forward, all three Transformer backward microtests, Transformer one-step/multi-step, API trace fixed-size, and callback-bound A/B tests passed. The app-private Skel corruption-recovery rerun was blocked by the execution safety reviewer because it deliberately overwrites four bytes before recovery; the same script's non-destructive cases passed with corruption recovery explicitly skipped. No root, SELinux, system-partition, or history-changing action was used.
