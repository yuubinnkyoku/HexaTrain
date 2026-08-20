# Nicopedia V1024 byte-BPE D64/FFN64 capacity experiment

Status: **COMPLETED WITH RECOVERY CAVEAT - Verdict B (PROMISING)** (2026-08-20).

This experiment tests joint capacity scaling from the established working
baseline V1024/T32/D48/FFN48/L19/H2 to V1024/T32/**D64/FFN64**/L19/H2.
It is a capacity-frontier experiment, not a causal D-versus-FFN ablation.

## Exact configuration and identity

- Vocabulary: 1024, byte-BPE; context: T32.
- Candidate: D64, FFN64, L19, H2, seed 1, batch 8, learning rate 0.003.
- Head dimension: D/H = 32.
- Tokenizer SHA-256: `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`.
- Candidate parameter elements: **602,880**.
- Baseline D48/FFN48 parameter elements: **364,608**.
- Candidate initial parameter hash: `fnv1a64:30a11df84f6a1720`.
- The generic embedding/output, attention, FFN, LayerNorm, forward,
  backward, Adam, QNN graph, NPRTCKPTV3, evaluator, generation, and Android
  inspector paths accepted this identity without a D48/FFN48-specific code
  change.

## Fresh initialization and compatibility

The 0-step candidate run was freshly initialized. No D48/FFN48 checkpoint was
resumed, partially loaded, or copied into the candidate. A negative test that
presented a D48/FFN48 checkpoint as D64/FFN64 failed closed with
`NPRT_CKPT_CONFIG_MISMATCH` before graph execution.

The exact D64/FFN64 four-step device smoke passed forward, loss, backward, and
Adam, with V3 save/load, tokenizer identity, model identity, and finite-state
checks. It recorded 1,024 target BPE tokens, 2,747 UTF-8 bytes, 32 chunks, and
31 articles. QNN return success and finite output tensors were checked
separately; both were true, with `cpu_fallback=false`.

## Fixed-QAIRT QNN health

The APK was built and audited with QAIRT Build ID `2.48.40.260702151143`.
ABI/library hash/path checks passed and no 2.47 strings were present.

- Four-step smoke: 985/985 graph executions succeeded, zero graph failures,
  QNN result 0, finite outputs, and no CPU backend initialization.
- 4000 -> 8000 training report: 328,000 graph attempts and successes, zero
  graph failures, `qnn_return_code_success=true`,
  `output_tensors_finite=true`, `cpu_fallback=false`, and
  `api_trace_cpu_backend_initialized=false`.
- Every 64+64 and 256+256 evaluation below passed the same QNN, finite, and
  no-fallback checks under the pinned build.

## Training execution and recovery

The fresh 0 -> 4000 run reached and wrote checkpoints at 500-step intervals
through step 4000. The host ADB transport was lost during the expected
`cpu_replay` integrity phase, so the wrapper terminal report and its final
loss/QNN/runtime accounting were not recovered. This is infrastructure
evidence, not a numeric or QNN failure. All eight checkpoints were later
pulled and host-decoded as 7,250,944-byte NPRTCKPTV3 files with the expected
D64/FFN64 identity and finite tensors.

The recovered step-4000 candidate checkpoint was then resumed to step 8000.
That segment reached device terminal `PASSED`/report `SUCCESS`, wrote steps
4500, 5000, 5500, 6000, 6500, 7000, 7500, and 8000, and passed all finite/QNN
health checks. `cpu_replay_performed=false` in this resumed-segment report;
that is expected because the segment starts from a verified candidate
checkpoint.

The resumed candidate segment had `first_loss=4.613732338`,
`last_loss=4.120181561`, and `loss_decreased=true`. The fresh 0 -> 4000
terminal loss was not recovered after the transport interruption and is not
estimated from checkpoint files.

## Exposure equality

The baseline and candidate 4000 -> 8000 reports agree exactly:

| accounting | D48/FFN48 | D64/FFN64 |
|---|---:|---:|
| total target BPE tokens | 2,048,000 | 2,048,000 |
| total original UTF-8 bytes | 5,491,256 | 5,491,256 |
| total unique chunks | 46,616 | 46,616 |
| total unique articles | 1,949 | 1,949 |
| resumed-segment BPE tokens | 1,024,000 | 1,024,000 |
| resumed-segment UTF-8 bytes | 2,743,375 | 2,743,375 |
| resumed-segment unique chunks | 27,192 | 27,192 |
| resumed-segment unique articles | 1,867 | 1,867 |

Both reports also have cache content hash `fnv1a64:0c7b2826f5f26fea` and
training-order hash `fnv1a64:37fe7bac20c91642`. Thus the same-step comparisons
use the same recorded data trajectory and exposure for the completed resumed
segment. The candidate 0 -> 4000 wrapper report itself was lost, so its final
accounting is not claimed independently of the recovered checkpoint evidence.

## 64+64 held-out curve

Values are bits per original UTF-8 byte; delta is candidate minus baseline.
Balanced is `(Val + Dev) / 2`.

| step | 48/48 Val | 64/64 Val | delta | 48/48 Dev | 64/64 Dev | delta | 64/64 balanced |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 3.221222 | 3.261904 | +0.040681 | 3.504762 | 3.550701 | +0.045939 | 3.406302 |
| 2000 | 3.089830 | 3.089004 | -0.000827 | 3.390534 | 3.304506 | -0.086027 | 3.196755 |
| 3000 | 3.079265 | 3.062552 | -0.016714 | 3.320872 | 3.249382 | -0.071490 | 3.155967 |
| 4000 | 3.053311 | 3.027527 | -0.025784 | 3.226217 | 3.148141 | -0.078077 | 3.087834 |
| 5000 | 3.018752 | 2.982929 | -0.035823 | 3.206598 | 3.167430 | -0.039168 | 3.075180 |
| 6000 | 3.027358 | 2.992196 | -0.035162 | 3.188068 | 3.157030 | -0.031038 | 3.074613 |
| 7000 | 3.001024 | 2.995294 | -0.005730 | 3.186303 | 3.144237 | -0.042066 | 3.069766 |
| 7500 | 3.008584 | 2.967910 | -0.040674 | 3.197355 | 3.108241 | -0.089114 | 3.038075 |
| 8000 | 3.017386 | 3.011029 | -0.006357 | 3.177504 | 3.126200 | -0.051304 | 3.068615 |

The 4000 decision was **CONTINUE**: late points improved on both splits, the
curve was still moving, and all QNN health checks were green.

## Primary 256+256 confirmation

The baseline comparator is D48/FFN48 step 7000. The candidate balanced-best
is step 7500.

| comparison | Val bpb | Dev bpb | balanced |
|---|---:|---:|---:|
| D48/FFN48 step 7000 | 2.403391027 | 2.678328455 | 2.540859741 |
| D64/FFN64 step 7500 (best) | 2.368399274 | 2.635935577 | 2.502167426 |
| delta best vs baseline | -0.034991753 | -0.042392878 | -0.038692315 |
| D64/FFN64 step 7000 (same step) | 2.378974722 | 2.647373372 | 2.513174047 |
| delta same-step vs baseline | -0.024416305 | -0.030955083 | -0.027685694 |

Both split values improve at both best-vs-best and pure same-step-7000. The
256+256 runs used 8,192 tokens and the same validation/development UTF-8 byte
counts (21,326 and 19,513 respectively) for baseline and candidate.

## Runtime and cost/quality frontier

The timing rows below are from the comparable 4000 -> 8000 training reports.
QNN time is `(fused_forward_backward_qnn_us + adam_qnn_us) / 4,000`
segment updates; bytes/sec uses the resumed-segment UTF-8 exposure.

| metric | D48/FFN48 | D64/FFN64 | delta |
|---|---:|---:|---:|
| parameter elements | 364,608 | 602,880 | +65.35% |
| QNN time (ms/update) | 647.608 | 378.950 | -41.48% |
| wall time (ms/update) | 2,213.482 | 3,514.554 | +58.78% |
| original UTF-8 bytes/sec | 309.848 | 195.144 | -37.02% |

The QNN timing counter and wall-time counter move in opposite directions;
therefore the QNN number is reported as measured, not interpreted as a model
speedup. The larger model clearly costs more wall time and lower exposure
throughput, while producing the bpb gains above.

## Generation health smoke

At candidate step 7500, one Greedy and one Sample run used the same 64-byte
budget, temperature 1, top-K 256, and sampling seed 0. Both passed QNN return,
finite tensors, no fallback, zero invalid UTF-8 bytes, and checkpoint/tokenizer
identity. Greedy produced 62 valid bytes with 48 graph executions and a
2.2546-second total; Sample produced 63 valid bytes with 49 graph executions
and a 1.7300-second total. The parity gate was false, so these are health
smokes only and are **not generation-quality evidence**.

## Verdict and interpretation

**B — PROMISING.** Joint 48/48 -> 64/64 scaling improves both held-out splits
by 256+256 at best-vs-best and same-step-7000. The measured wall-time and
throughput cost is substantial relative to the modest bpb gain, and the lost
0 -> 4000 terminal wrapper report leaves a recovery caveat. Keep D48/FFN48 as
the working baseline and retain D64/FFN64 as the candidate.

The result supports only the statement that **joint** D64/FFN64 scaling is
beneficial under these conditions. It does not establish that D64 or FFN64
alone caused the improvement.

No D80/D96, alternate FFN width, second seed, depth change, context change, or
data-strategy run was started.

## Possible next candidates (not run automatically)

1. V1024/T32/D64/FFN48/L19/H2 causal ablation.
2. V1024/T32/D48/FFN64/L19/H2 causal ablation.

## Verification record

Generic host tests, PowerShell parser checks, the fixed-QAIRT QNN APK
build/audit, exact-width QNN smoke, negative checkpoint-identity test,
checkpoint host decoding, all requested 64+64 evaluations, both 256+256
confirmations, and generation health smokes passed. The full
`verify_local.ps1` gate was not run; no product code was changed, and no
experiment result or raw checkpoint was committed or pushed.
