# QNN HTP Tiny LM post-fix end-to-end training and generation

## Result

Classification: `END_TO_END_TRAINING_AND_GENERATION_REPRODUCIBLE`

Evaluation source commit: `e6661d8cd5b806580f5b927a961fe223cd89f165`

This classification follows the requested rule: all five seeds completed the
fixed training/evaluation protocol, while repeat-training evidence is scoped to
the designated representative seed. It does not claim that seeds 2–5 were each
retrained three times.

The Softmax output-shape correction and LayerNorm centered-value scale correction
were evaluated from training through autoregressive generation. All five HTP
training seeds remained finite, reduced loss, and completed with no nonzero QNN
execute return. Oracle and free-running generation were both 20/20 exact. Three
independent repeats of the representative seed were bitwise reproducible.

The result does not mean that the entire application ran only on the NPU. CPU
code owns orchestration, data preparation, optimizer bookkeeping, result
checking, and the CPU reference. The training-step numerical graphs described
here execute on HTP. Backward computation is expressed as explicit QNN graphs;
QNN autograd is not used. A successful QNN execute return and a finite output
tensor are checked and reported separately.

## Protocol

The fixed configuration was Adam, learning rate 0.003, 320 updates, no gradient
clipping, and the established five seeds. The model was B=1, T=8, V=32, D=16,
one attention head, one layer, FFN=32, ReLU, pre-norm, and causal attention. No
dataset, case, seed, model-size, or hyperparameter search was performed.

The CPU reference used the same initial parameters, batches, targets, and Adam
formula. Small differences in initial loss reflect the separately executed CPU
and HTP numerical paths, not different starting data.

QAIRT build ID was `2.48.40.260702151143`. Runtime, Stub, and Skel hashes matched
the selected QAIRT 2.48.40 SDK before execution.

## Five-seed training

HTP results:

| Seed | Initial loss | Final loss | Reduction (%) | Accuracy | Nonfinite | QNN execute/nonzero | Final gradient L2 | Final parameter L2 | Canonical parameter hash |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.444437623 | 0.01613830021 | 99.53146778 | 1.00000 | 0 | 648 / 0 | 0.06001740239 | 11.98706814 | `0ba41957248b776a18b5db000afed50f787c15f860194497cb0ff616cee8134b` |
| 2 | 3.475236416 | 0.07947546753 | 97.71309177 | 0.96875 | 0 | 648 / 0 | 0.05335780943 | 12.27207468 | `74f839d88e1eab1c2475f46cce2696d4c8367a5e5ed8f5863d216c7dbc297750` |
| 3 | 3.462806106 | 0.1735408767 | 94.98843217 | 0.96875 | 0 | 648 / 0 | 0.07297854849 | 12.01660371 | `62b1714f6a7d2ec421a2cf8939935526acf303ef087b61730802da388af58f7a` |
| 4 | 3.462253809 | 0.01882231917 | 99.45635646 | 1.00000 | 0 | 648 / 0 | 0.06761019982 | 12.19000261 | `f3e67178492feb2a39f89f944ef8fdca37399c1bf8e037a4bf9590c32c50ad32` |
| 5 | 3.464655459 | 0.2934597878 | 91.52989983 | 0.93750 | 0 | 648 / 0 | 0.07749291241 | 11.86129419 | `ce7b1bbb3989c4e5f2a2984d7ad19c8497aa4dbdcac61318206deeafb4e9f9c0` |

CPU reference results:

| Seed | Initial loss | Final loss | Reduction (%) | Accuracy | Nonfinite | Final gradient L2 | Final parameter L2 | Canonical parameter hash |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.444429696 | 0.1674873093 | 95.13744439 | 0.96875 | 0 | 0.03329896056 | 11.93722469 | `94e571c0d29fc41592f20dd8d1b56ced303a84c20f6ee4d3fe2c5c607914efd4` |
| 2 | 3.475229263 | 0.1890411384 | 94.56032612 | 0.96875 | 0 | 0.06466700422 | 12.37995392 | `331170c2416eb83b234af7bca737fa0f67dddd436ca3277cb7f2e491db90c536` |
| 3 | 3.462803364 | 0.1699576006 | 95.09190726 | 0.93750 | 0 | 0.04393938333 | 12.12167020 | `08bad432df0cf86ee3629e4e97a435dcd55a96d2469198e694d561ee2d2a5ff5` |
| 4 | 3.462248445 | 0.2479991278 | 92.83704992 | 0.90625 | 0 | 0.06702244633 | 12.32248834 | `3f47fc180cda9b3dfb0339757c20a1991a9ad42039940bccbd02aa074354e7e7` |
| 5 | 3.464650035 | 1.678407505 | 51.55621814 | 0.84375 | 0 | 0.2192289761 | 12.31412993 | `ac95b90069fa55423e97a2702e4f45d222a2f0759e93e08dede8efdb8d011918` |

The measured HTP final accuracy was not rewritten to match the earlier 1.0
expectation: seeds 2, 3, and 5 ended at 0.96875, 0.96875, and 0.93750.

## Generation

Twenty fixed cases were evaluated across the five seeds. Each case contains an
ID, prefix, expected continuation, generated continuation, exact-match flag,
first mismatch step and tokens, top-3, and expected-token probability.

Oracle generation appends the expected token to the context after every step.
Free-running generation appends the model's own previous prediction. A
dedicated context-evolution assertion verifies that the free-running path does
not read the expected next token. Both modes start from the same published case
prefixes.

| Mode | Exact |
|---|---:|
| Oracle | 20/20 |
| Free-running | 20/20 |

There was no generation mismatch to analyze. Complete per-step records are in
the public CSV files linked below.

## CPU/HTP logits

The public comparison has 320 rows: oracle and free-running modes, five seeds,
four cases, and eight generation steps. Its primary scope is
`CPU_TRAINED_CPU_VS_HTP_TRAINED_HTP`; consequently it includes the effect of
different CPU and HTP training trajectories as well as execution-path numeric
differences.

| Measure | Result |
|---|---:|
| Maximum absolute difference | 18.59372616 |
| Mean of per-row mean absolute differences | 2.129667332 |
| Maximum relative L2 difference | 1.661078141 |
| ArgMax agreement | 312/320 |
| Ordered top-3 agreement | 94/320 |

For the stricter path-consistency check, the evaluation and generation paths
were given the identical prefix. CPU evaluation versus CPU generation and HTP
evaluation versus HTP generation both had zero max, mean, and relative-L2
difference. The representative identical-prefix CPU-versus-HTP comparison had
maximum absolute difference 0.0362739563 with matching ArgMax and top-3.

The first CPU/HTP ArgMax disagreement in the broader 320-row comparison was
oracle case `s3_p1`, step 2. The prior context was
`6,7,4,5,6,7,4,5`; the expected token was 6. CPU ranked
`5,6,12`, while HTP ranked `6,5,12`. CPU expected-token probability was
0.1416620463 and HTP expected-token probability was 0.7680664062. The row had
max absolute error 3.758351326, mean absolute error 1.210941926, and relative
L2 error 0.1850626159. HTP's token-6 versus token-5 margin was 1.6875; CPU's
expected-token versus top-token margin was -1.784258844. Because the two sides
use separately trained parameters in this comparison, this is a trajectory
comparison, not evidence of a same-parameter HTP execution error.

## Reproducibility

The representative seed (seed 1) was trained three independent times.
Checkpoint loss, final accuracy, final canonical parameter hash, final logits
hash, all oracle and free-running sequences, and both exact aggregates were
identical. This representative-seed repeat is classified
`BITWISE_REPRODUCIBLE`; seeds 2–5 have the five-seed single-run evidence above,
not independent repeat-training evidence.

## Performance

Three `EXCLUSIVE_BENCHMARK` runs used the same build, device, and model. Training
used 16 warm-up and 1,584 measured steps; generation used 16 warm-up and 304
measured tokens. Reported values summarize runs rather than selecting the
single fastest observation.

| Metric | Median | P95 |
|---|---:|---:|
| Initialization (ms) | 82.7315 | 99.2023 |
| Graph creation (ms) | 79.2383 | 142.520 |
| Finalize (ms) | 99.6031 | 153.972 |
| Steady training step median (ms) | 7.69271 | 10.0727 |
| Steady training step p95 (ms) | 8.11130 | 10.6176 |
| Updates/s | 129.993 | 139.697 |
| Tokens/s | 1,039.95 | 1,117.58 |
| Generation token latency median (ms) | 1.05047 | 1.57422 |
| Generation token latency p95 (ms) | 1.16156 | 1.82141 |

The three device-temperature observations were 36→36 °C, 36→36 °C, and
38→38 °C. The device was awake, unplugged, thermally normal, and the app did
not take foreground focus during headless measurement.

## Cause-fix regression

The final evaluated APK passed the 19 headless correctness suites and eight
focused tap/isolation suites. Coverage included shape-validator negative and
positive cases, Softmax/attention/LayerNorm backward, Adam one-step,
`DPROBABILITIES→DSCORES`, LayerNorm2
`CENTERED_S2→SQUARE2`, `STOP_AFTER_DINPUT`, full fixed state, checkpoint
replay, and the CPU/HTP gradient × CPU/HTP Adam matrix. QNN execute nonzero,
nonfinite, poison residual, and unexpected APP_WRITE change counts were zero.
Focus takeover count was zero.

The pre-fix same-checkpoint `CENTERED_S2` ranges were:

| Seed | Minimum | Maximum |
|---:|---:|---:|
| 1 | -256.75 | 200.5 |
| 2 | -256.5 | 242.0 |
| 3 | -257.25 | 148.75 |
| 4 | -146.125 | 256.5 |
| 5 | -256.25 | 235.375 |

Before the scale fix, `SQUARE2` was the first bad tensor and contained positive
infinity. With centered scale 8, the post-fix same-checkpoint replay was finite
and deterministic for all 500 checks (100 per seed), including `SQUARE2`.
The observed overflow threshold is consistent with FP16-range intermediate
behavior; this does not assert that HTP internal arithmetic is FP16.

Skel audit, reuse, and recovery completed with the selected SDK hash. Recovery
was exercised without clearing app data; the final build subsequently reused
the matching Skel.

## Live Update smoke

One final run was started from the real UI on Android 16.0. Progress began and
phase, step, and loss changed. The app was moved out of the foreground and the
run completed without a process crash. The completion notification reported
step 9526/10000 with `ongoing=false`; tapping it returned
`PhoneLM/.MainActivity` to the top-resumed activity. Notification permission
was already granted and was not changed.

Android 16 used the expected progress-style fallback; promoted ongoing is not
supported on this device. The final completion state was directly observed.
The ongoing notification had been observed during the earlier smoke pass, while
the final post-fix run specifically reconfirmed completion, non-ongoing state,
tap return, and absence of the former UI output-memory failure.

## Public artifacts

The allow-list exporter publishes only:

- [bundle README](results/qnn-htp-post-fix-generation-2026-07/README.md)
- [training seeds](results/qnn-htp-post-fix-generation-2026-07/training-seeds.csv)
- [oracle generation](results/qnn-htp-post-fix-generation-2026-07/oracle-generation.csv)
- [free-running generation](results/qnn-htp-post-fix-generation-2026-07/free-generation.csv)
- [logits comparison](results/qnn-htp-post-fix-generation-2026-07/logits-comparison.csv)
- [reproducibility](results/qnn-htp-post-fix-generation-2026-07/reproducibility.csv)
- [performance](results/qnn-htp-post-fix-generation-2026-07/performance.csv)

APK files, QAIRT binaries and headers, checkpoints, tensor dumps, logcat, device
identifiers, endpoints, secrets, and private or absolute paths are excluded.
