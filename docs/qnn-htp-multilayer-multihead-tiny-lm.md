# QNN HTP multi-layer / multi-head Tiny LM

## Result

PhoneLM validated explicit Transformer forward, backward, and Adam-update graphs on QNN HTP through B=1, T=32, V=32, D=32, FFN=32, two layers, and two attention heads. The maximum configuration completed 320 Adam steps at learning rate 0.003 without clipping for five finite seeds. Oracle and free-running generation were exact for 20/20 cases each, and three formal reruns were bitwise reproducible.

The application constructs the forward, backward, and optimizer graphs explicitly. It maintains deterministic parameter and Adam-state registries, supplies Adam bias-correction scalars, and orchestrates the training step. QNN autograd was not used. The claim is that the training-step numerical operations were executed on HTP; it is not a claim that the entire application ran only on an accelerator or that HTP internal arithmetic is FP32.

## Fixed environment and graph contract

QNN-enabled builds, APK audit, and device execution used QAIRT build `2.48.40.260702151143`. Runtime, V81 Stub, and V81 Skel assets in the APK matched that distribution, and the APK audit found no QAIRT 2.47 strings.

The generalized schema accepts one or two layers and one or two heads and rejects unsupported or inconsistent shapes. Q, K, and V parameters remain D-by-D. Multi-head attention splits the projected dimension in deterministic layer/head order, applies Softmax over the key/token axis, and concatenates heads back to D. The graph validator checks layer chaining, head shapes, Softmax and Reduce axes, `keep_dims`, declared output shapes, APP_READ/APP_WRITE direction, and generalized gradient bindings. Host tests also verify that parameter buffers, head temporaries, and Adam first/second moments do not alias.

## Single-layer regression

The generalized path reproduced the T=32, D=32, V=32, L=1, H=1 baseline for seed 1:

| Metric | Result |
| --- | --- |
| Initial loss | 3.480286956 |
| Final loss | 0.1905737069 |
| Final accuracy | 0.984375 |
| Parameter hash | `7de7fb...` |
| Logits hash | `d4bb775...` |
| Oracle / free generation | 4/4 / 4/4 |
| Non-finite / QNN execute failures | 0 / 0 |

The pre- and post-schema reports had identical loss, accuracy, parameter hash, and logits hash, so no divergence step or tensor was found.

## Staged scale-up

| Stage | Configuration | Seeds | Oracle | Free | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | T8 D16 L2 H1 | 5/5 finite | 19/20 | 19/20 | End-to-end finite; one quality miss |
| 2 | T16 D16 L2 H1 | seed 1 | 20/20 | 20/20 | Validated |
| 2 | T32 D32 L2 H1 | seed 1 | 20/20 | 20/20 | Validated |
| 3 | T8 D16 L1 H2 | 5/5 finite | 20/20 | 20/20 | Validated |
| 4 | T16 D16 L1 H2 | seed 1 | 20/20 | 20/20 | Validated |
| 4 | T32 D32 L1 H2 | seed 1 | 20/20 | 20/20 | Validated |
| 5 | T16 D16 L2 H2 | seed 1 | 20/20 | 20/20 | Validated |
| 5 | T32 D32 L2 H2 | 5/5 finite | 20/20 | 20/20 | Maximum validated |

Stage 1 seed 5 phase 3 produced 19/20 exact tokens in both generation modes. QNN execution still succeeded and all five training trajectories were finite. This is recorded as a generation-quality boundary, not an application, runtime, or hardware failure.

## CPU/HTP and numerical safety

The maximum diagnostic compared 16 finite scopes: logits and dlogits; both layer-input gradients; attention probabilities for every layer/head pair; representative layer gradients; and next-parameter, first-moment, and second-moment state for both CPU-gradient/HTP-Adam and HTP-gradient/HTP-Adam paths. The largest published head-probability absolute error was 0.00146484375. All compared scopes had zero non-finite elements.

Per-layer checks covered the T-by-1 Softmax dot shape, key-axis Softmax, Reduce `keep_dims`, centered LayerNorm scale 8, CENTERED and SQUARE finiteness, attention-score and probability ranges, gradients, Adam moments, and denominator outputs. No NaN/Inf replacement, unexplained clamp, epsilon increase, learning-rate reduction, or tolerance-only workaround was used.

## Reproducibility and performance

Three independent maximum-configuration formal runs produced identical canonical parameter and representative-logit hashes. The median observed steady training-step time was 60.5534 ms, with 16.5144 updates/s, 528.459 tokens/s, and 6.75974 ms median generation-token latency. These measurements came from correctness runs and are not presented as an isolated peak benchmark.

## Device regression and Live Update

The audited APK passed the 19 headless correctness suites, nine focused graph/replay/optimizer suites, and Skel replace-then-reuse recovery. Activity/focus takeover remained zero. One first attempt at the stability suite lost its host-side ADB monitoring connection after the device test had already passed; a complete retry supplied valid host evidence and is the counted result.

An actual UI run confirmed maximum-mode configuration display, start, background continuation, completion, and notification-tap return. It also exposed that Transformer/LM progress had only been returned at completion, leaving the notification at step zero. The callback was then wired through the Transformer training dispatcher. A post-fix maximum device run reported phase, seed, step, and loss updates and completed at seed 5/5, step 320/320 with an ongoing-to-complete notification transition.

## Publication and limits

The detailed allow-list-exported tables are in [qnn-htp-multilayer-multihead-2026-07](results/qnn-htp-multilayer-multihead-2026-07/README.md). The bundle excludes raw tensors/logits, identifiers, endpoints, absolute paths, APKs, QAIRT binaries and headers, Stub/Skel assets, checkpoints, and device logs.

The maximum result is the largest evaluated configuration and not a hardware-limit claim. The current implementation intentionally rejects more than two layers or more than two heads. Extending those bounds, adding broader seeds at intermediate sizes, and running isolated performance benchmarks are separate milestones.
