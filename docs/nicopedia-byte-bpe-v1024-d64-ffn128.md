# Nicopedia V1024 byte-BPE D64/FFN128 capacity experiment

Status: **COMPLETED - Verdict B (SMALL CONSISTENT IMPROVEMENT)** (2026-08-23).

This experiment isolates the FFN width as a single independent variable on top
of the established D64/FFN64 candidate:
V1024/T32/D64/**FFN64**/L19/H2 →
V1024/T32/**D64/FFN128**/L19/H2.
Everything else (tokenizer, dataset, training order, optimizer, LR, batch,
seed) is held fixed; the run is fresh from step 0.

## Config

- Vocabulary: 1024 byte-BPE; context T32; head dim D/H = 32.
- Candidate: D64, FFN128, L19, H2, seed 1, batch 8, LR 0.003.
- Tokenizer SHA-256: `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`.
- Parameter elements: **758,528**, verified by both the host registry helper
  (`parameterElementCount` via `cpu_reference_training`) and the device-side
  `resource_estimator_parameter_elements` field. No hard-coded constant was
  used anywhere in the path.
- Baseline D64/FFN64: 602,880 elements. Delta: +155,648 (+25.82%).
- Cache content hash `fnv1a64:0c7b2826f5f26fea`; training order hash
  `fnv1a64:0e2e15196d851431` for segment 0→500 and `fnv1a64:37fe7bac20c91642`
  recorded at the resumed segments' reports — identical to the D48/FFN48 and
  D64/FFN64 baselines.

## Fresh initialization and identity

The candidate was freshly initialized at step 0. No D64/FFN64 checkpoint was
resumed or copied. Checkpoint filenames follow the non-anchor convention
(`htp-seed1-l19-t32-d64-f128-step<N>.ckpt`), and every pulled checkpoint was
decoded through the host evaluator with seed/layers/D/FFN/step identity plus
NPRTCKPTV3 tokenizer-kind/hash checks. A negative test was not re-run here;
the generic resume-identity rejection was already exercised in the D64/FFN64
experiment and the code path is unchanged.

No product code change was required: the research/headless training path,
evaluator, generation path, native validation (`FeedForwardDimension ≤ 1024`),
and graph shape validator all accepted FFN128 generically.

## Smoke

Two-segment device smoke under pinned QAIRT Build ID `2.48.40.260702151143`
(APK audit passed: arm64-v8a, V81 stub/skel hash match, no 2.47 strings, no
host paths):

- Fresh 4-step segment: QNN return success, 1,213/1,213 graph executions OK,
  zero failures, finite outputs, `cpu_fallback=false`. Identity
  `T32_D64_FFN128`, 758,528 elements verified on-device.
- The fresh smoke's terminal status was FAILED solely on the built-in
  loss-decrease health gate (`first_loss=7.489538193`,
  `last_loss=7.79915905`). All QNN/finiteness fields were green and the CPU
  reference showed the same early bump (step-4 CPU loss 8.59), so this is an
  initial-training transient, not an HTP numerical fault. Per the runner's own
  classification this is a quality-gate trip, not a code regression.
- Resume 4→8 segment: terminal SUCCESS, five checkpoints written and
  host-decoded with matching identity and `finite=true`.

## Training execution

Long training ran as three wrapper segments because the TCP ADB link to the
device dropped twice during single-wrapper runs (infrastructure evidence, not
numeric failure):

| segment | steps | result | first/last loss |
|---|---|---|---|
| S1 fresh | 0→500 | SUCCESS (`loss_decreased=true`) | 7.4895 → 4.9444 |
| S2 resume | 500→4000 | SUCCESS | 5.2422 → 4.2289 |
| S3 resume | 4000→8000 | SUCCESS | 4.5799 → 4.0870 |

All segments: `all_steps_finite=true`, `final_finite=true`, zero graph
failures, `cpu_fallback=false`, HTP backend build ID match. Segment S1
performed the full CPU replay integrity pass; resumed segments skip it by
design.

## Exposure equality

The final report matches the baseline exactly:

| accounting | D64/FFN64 | D64/FFN128 |
|---|---:|---:|
| total target BPE tokens | 2,048,000 | 2,048,000 |
| total original UTF-8 bytes | 5,491,256 | 5,491,256 |
| unique chunks | 46,616 | 46,616 |
| unique articles | 1,949 | 1,949 |

Same cache content hash and same training order hash family as the baselines.

## 64+64 held-out curve

Bits per original UTF-8 byte; balanced = (Val + Dev) / 2.

| step | Val bpb | Dev bpb | balanced |
|---:|---:|---:|---:|
| 4000 | 2.97471828 | 3.115238211 | 3.044978246 |
| 4500 | 2.950346274 | 3.141603655 | 3.045974964 |
| 5000 | 2.968115907 | 3.155713599 | 3.061914753 |
| 5500 | 3.023419357 | 3.14509149 | 3.084255424 |
| 6000 | 2.962472845 | 3.09886549 | 3.030669168 |
| 6500 | 2.971115736 | 3.08304484 | 3.027080288 |
| 7000 | 2.97431337 | 3.118463783 | 3.046388576 |
| 7250 | 2.95863448 | 3.12403871 | 3.041336595 |
| 7500 | 2.963348895 | 3.107300886 | 3.03532489 |
| 7750 | 2.970209737 | 3.128225612 | 3.049217674 |
| 8000 | 2.978292647 | 3.118336582 | 3.048314614 |

Candidate best by balanced: **step 6500** (3.027080). For reference, the
D64/FFN64 curve at the same 64+64 capacity had its best at step 7500.

## Primary 256+256 confirmation

Same validation/development UTF-8 byte counts as the baseline runs
(21,326 / 19,513 bytes over 8,192 tokens).

| comparison | Val bpb | Dev bpb | balanced |
|---|---:|---:|---:|
| D64/FFN64 step 7500 (baseline best) | 2.368399274 | 2.635935577 | 2.502167426 |
| D64/FFN128 step 6500 (candidate 64+64-best) | 2.372527708 | 2.620970851 | 2.496749280 |
| D64/FFN128 step 7500 (same-step) | 2.357021533 | 2.627340584 | 2.492181059 |

Deltas vs baseline:

- best-vs-best (FFN128 s6500): ΔVal +0.004128, ΔDev −0.014965, ΔBalanced −0.005418
- same-step (FFN128 s7500): ΔVal −0.011378, ΔDev −0.008595, ΔBalanced −0.009986

Both comparisons improve balanced; the split-level signs differ between the
two comparisons, so the improvement is small and partly split-dependent.

## Runtime and cost

Comparable 4000→8000 resumed-segment reports:

| metric | D64/FFN64 | D64/FFN128 | delta |
|---|---:|---:|---:|
| parameter elements | 602,880 | 758,528 | +25.82% |
| QNN time (ms/update) | 378.950 | 503.928 | +32.97% |
| wall time (ms/update) | 3,514.554 | 3,389.864 | −3.55% |
| original UTF-8 bytes/sec | 195.144 | 202.322 | +3.68% |

As in prior experiments the QNN counter moves opposite to wall time; the QNN
timing number is reported as measured and not interpreted as a slowdown.
Wall-time and throughput are essentially flat between FFN64 and FFN128 in
these runs (the FFN128 segment benefited from cooler ambient conditions;
thermal state 0 throughout, battery 35–36 °C). No claim of a speedup is made.

## Generation health smoke

One Greedy run at candidate step 7500, MaxNewBytes = 8, htp-native gate:
status SUCCESS, QNN return success, 31/31 graph executions OK, finite
checkpoint/tensors, `cpu_fallback=false`, checkpoint NPRTCKPTV3 with matching
tokenizer kind/hash, `feed_forward_dimension=128`. Generated content quality
was not evaluated (parity gate not applicable); this is a health smoke only.

## Verdict and interpretation

**B — SMALL CONSISTENT IMPROVEMENT.**

- Evidence for "FFN64 width was a bottleneck" is **weak-to-moderate**: the
  256+256 balanced metric improves in both comparisons, but the magnitude
  (≈0.005–0.010 bpb for +25.8% parameters) is smaller than the D48/48→D64/64
  joint scaling gain (≈0.039) and the Val/Dev split signs are inconsistent
  across comparison pairs.
- The 64+64 curve is noisy (±0.03 swings between adjacent checkpoints), so
  single-point rankings within one run should not be over-read.
- Cost: wall-time throughput is essentially unchanged versus FFN64 in these
  paired runs; memory grows with the parameter count (+25.8%).

No D80, FFN96/160/256, L/T/H change, SwiGLU, GQA, MTP, tokenizer retrain, or
production UI change was made. The production Model Settings catalog remains
unchanged.

## Verification record

Runner self-tests, pinned-QAIRT check, QNN-enabled APK build + audit,
parameter-count host check, two-segment smoke, negative-free identity checks
on every pulled checkpoint, three training segments with full health gates,
eleven 64+64 evals, three 256+256 evals, and one generation health smoke all
passed under the pinned Build ID. The full `verify_local.ps1` gate was not run
(no product code changed); targeted/Fast verification only. Nothing was
committed or pushed.

## Caveats

- Single seed, single run; no repetition-based error bars.
- The fresh 0→4000 single-wrapper attempt was lost to ADB transport drops
  twice; training completed via segmented resumes instead (protocol-compliant,
  COMPLETE-checkpoint-only).
- The QNN timing counter contradicts wall time as before; cost conclusions use
  wall clock and throughput only.
- 5500 shows a local spike in the 64+64 curve; adjacent checkpoints do not
  confirm it.
