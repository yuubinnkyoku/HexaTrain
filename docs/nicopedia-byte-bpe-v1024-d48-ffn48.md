# Nicopedia V1024 byte-BPE D48/FFN48 capacity experiment

Status: **COMPLETED — targeted experiment, no full gate or push** (2026-08-19).

This experiment changed only the FFN width against the D48/FFN32 reference:
V1024 / T32 / D48 / **FFN48** / L19 / H2, seed 1, batch 8, learning rate
0.003, fresh initialization. The byte-BPE tokenizer and data trajectory were
unchanged:

sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798

## Identity and host validation

- Candidate parameter count: **364,608**.
- D48/FFN32 reference parameter count: 335,424; FFN48 adds 29,184
  parameters (+8.7%).
- The generic FFN path, QNN graph, checkpoint parser, evaluation, and
  generation paths accept FFN48 without an FFN32 hardcode or an FFN >= D
  assumption. No product-code change was needed for this capacity point.
- The V1024/T32/D48/FFN48 host layout/forward/backward/Adam smoke passed;
  parameters, gradients, optimizer state, and updated parameters were finite.
- A D48/FFN32 NPRTCKPTV3 checkpoint presented as FFN48 was rejected with
  CHECKPOINT_IDENTITY_MISMATCH before device execution.

## Fixed QAIRT, APK, and device smoke

The APK used the pinned QAIRT Build ID 2.48.40.260702151143. The QNN APK
audit passed for arm64-v8a: library hashes matched the fixed SDK and no 2.47
strings were present.

The 2-step FFN48 device smoke passed:

- HTP initialization and the D48/FFN48 graph preparation succeeded.
- qnn_return_code_success=true and output_tensors_finite=true were
  independently true.
- cpu_fallback=false, nan_detected=false, and inf_detected=false.
- All graph executes returned QNN result 0; the CPU backend was not
  initialized.
- Checkpoints were tokenizer-bound NPRTCKPTV3 files with verified identity
  and finite host decode.
- Accounting matched the FFN32 smoke: 512 target BPE tokens, 1,358 original
  UTF-8 bytes, 16 unique chunks, and 15 unique articles.

## Training execution and recovery

The first fresh 0→4000 attempt hit the runner's 300-second checkpoint-stall
guard before a checkpoint. ApplicationExitInfo classified the stop as a
runner-requested force-stop; this is not evidence of a QNN/native crash.

A second fresh 0→4000 attempt used a 500-step cadence and a 3,600-second stall
guard. It wrote and verified steps 500 through 3000, then stopped at the
runner guard before step 3500. The highest checkpoint was recovered and
verified as V1024/T32/D48/FFN48/seed1 with finite host decode.

After the user reconnected the device, a resume from step 3000 to 4000 passed,
and a second resume from 4000 to 8000 passed. Both runs used the fixed QAIRT
arguments, a 7,200-second checkpoint-stall guard, and the same 500-step
checkpoint cadence. The wrapper restored the pre-run power/idle state after
each successful run. The earlier ADB transport interruption is recorded as
infrastructure recovery evidence, not as a model or QNN failure.

## Training health and exposure

| Segment | Resume step | Target BPE tokens | Target UTF-8 bytes | Unique chunks | Unique articles | Training-order hash | Wall time | ms/update |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 3000→4000 | 3000 | 1,024,000 | 2,747,881 | 27,207 | 1,871 | fnv1a64:da7de1636de5356b | 2,401.295 s | 2,401.294 |
| 4000→8000 | 4000 | 2,048,000 | 5,491,256 | 46,616 | 1,949 | fnv1a64:37fe7bac20c91642 | 8,853.930 s | 2,213.482 |

The cumulative exposure and order hashes match the corresponding D48/FFN32
reference reports at steps 4000 and 8000. Both terminal training reports
had status=SUCCESS, all_steps_finite=true, final_finite=true,
qnn_return_code_success=true, output_tensors_finite=true,
cpu_fallback=false, nan_detected=false, and inf_detected=false.
The 8000-step report recorded 212,000 graph executes with 0 failures and last
QNN result 0; the CPU backend was not initialized.

The 8000-step candidate used 2,213.482 ms/update versus 2,174.279 ms/update
for the fresh D48/FFN32 0→4000 reference, a rough +1.8% per-update cost. The
segments have different resume histories and are not a controlled performance
benchmark; the number is an architecture-cost indication only.

## HTP 64+64 curve

Every row below passed the same HTP health checks: 128/128 graph executes,
finite output, zero non-finite chunks, QNN return success, and no CPU
fallback. Delta is FFN48 minus the D48/FFN32 reference; negative is better.

| Step | FFN48 Val | FFN48 Dev | FFN32 Val | FFN32 Dev | Δ Val | Δ Dev | FFN48 balanced mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 3.221222 | 3.504762 | 3.264491 | 3.547975 | -0.043269 | -0.043213 | 3.362992 |
| 2000 | 3.089830 | 3.390534 | 3.130852 | 3.348641 | -0.041022 | +0.041892 | 3.240182 |
| 3000 | 3.079265 | 3.320872 | 3.106552 | 3.344518 | -0.027286 | -0.023646 | 3.200069 |
| 4000 | 3.053311 | 3.226217 | 3.079106 | 3.211610 | -0.025795 | +0.014608 | 3.139764 |
| 5000 | 3.018752 | 3.206598 | 3.007022 | 3.250510 | +0.011730 | -0.043911 | 3.112675 |
| 6000 | 3.027358 | 3.188068 | 3.035100 | 3.220747 | -0.007742 | -0.032679 | 3.107713 |
| 7000 | **3.001024** | 3.186303 | 3.027405 | 3.198287 | -0.026381 | -0.011985 | **3.093663** |
| 7500 | 3.008584 | 3.197355 | 3.028240 | 3.190456 | -0.019656 | +0.006899 | 3.102969 |
| 8000 | 3.017386 | **3.177504** | 3.030828 | 3.214271 | -0.013441 | -0.036767 | 3.097445 |

The balanced-best point by the simple mean is step 7000. FFN48 improves
validation at 8/9 points and development at 6/9 points; the late band
(5000–8000) improves both splits at 4/5 points. The curve is therefore a
consistent but noisy quality improvement, not a uniform early win.

## 256+256 confirmation

The larger held-out run also passed 512/512 HTP graph executes with finite
outputs, QNN return success, and no fallback.

| Checkpoint | FFN48 Val | FFN48 Dev | D48/FFN32 step7500 reference | Δ Val vs reference | Δ Dev vs reference |
| ---: | ---: | ---: | --- | ---: | ---: |
| 7000 (balanced best) | **2.403391** | **2.678328** | 2.440660 / 2.705480 | -0.037269 | -0.027152 |
| 7500 | **2.413165** | **2.696504** | 2.440660 / 2.705480 | -0.027495 | -0.008977 |

The FFN48 candidate beats the established D48/FFN32 step-7500 anchor on both
splits at both measured checkpoints. This is the strongest primary quality
evidence in the experiment, although it remains one seed and a fixed held-out
subset rather than final-test evidence.

## Generation smoke

An optional greedy HTP-native generation smoke at the balanced-best step 7000
passed checkpoint identity, QNN return, finite prefix/AR/generation tensors,
and no fallback. It produced 63 valid UTF-8 bytes under the 64-byte cap.
The aggregate report recorded generation_gate=true, but parity_gate=false
and ar_gate=false; generated text is private and this run is health evidence,
not a generation-quality claim.

## Verdict

**B — PROMISING ENOUGH TO CONTINUE, but do not promote FFN48 as the default yet.**

- Quality: the HTP 64+64 curve is mostly better, and 256+256 improves both
  splits by 0.027–0.037 bits/byte at the balanced and 7500 checkpoints.
- Cost: parameter count rises 8.7%; the observed training update cost rises
  roughly 1.8% in the available comparison, subject to resume/device-state
  caveats.
- Correctness: training and evaluation remained finite, QNN return codes were
  successful, and no CPU fallback occurred in the terminal reports.
- Interpretation: FFN width is a credible bottleneck candidate for D48, but
  one seed and the noisy late curve are insufficient for a default-model
  decision or a claim that it is the only bottleneck.

## Remaining concerns (maximum three)

1. Only seed 1 was run; the quality lift needs variance confirmation.
2. Runtime comparisons are targeted indications, not a controlled benchmark;
   the device was reused across long resumes and evaluations.
3. Generation health passed, but parity and AR quality gates were false, so
   generation is not evidence of free-running quality.

## Next candidates (do not run automatically)

1. Repeat V1024/T32/D48/FFN48 through step 8000 with seed 2 and the same
   64+64/256+256 gates; combine seeds before making a default decision.
2. If seed-2 confirms the lift, test D48/FFN64 with the same early 0→4000
   gate to measure whether the FFN curve is still capacity-limited.
3. If runtime is the deciding constraint, repeat FFN48 and FFN32 under a
   controlled thermal/power benchmark before selecting the production width.

No further capacity experiment was started after the requested report.

## Verification record

Fast verification passed (git diff --check, PowerShell parser/preflight,
host tests, QNN shape/self-tests, JVM unit tests, and QAIRT self-test).
Targeted verification also passed the fixed-QAIRT APK build/audit, V1024 FFN48
host and device smoke, resumed 0→8000 HTP training, all listed HTP evaluation
points, 256+256 evaluation, and the optional generation smoke. The full
verify_local.ps1 gate was intentionally not run, and no commit or push was
performed.
