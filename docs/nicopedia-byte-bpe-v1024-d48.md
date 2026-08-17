# Nicopedia V1024 byte-BPE D48 comparison

This experiment changes exactly one architectural variable against the
V1024 byte-BPE baseline: the hidden dimension D goes from 32 to 48
(head dimension 24 with H=2). Tokenizer, dataset, data order, T32, FFN32,
L19, H2, batch 8, learning rate 0.003, and seed 1 all stay fixed.

Hypothesis: the V1024 BPE token space (1024 ids) is under-represented by
D=32, making hidden-state capacity the next bottleneck after the D32 run
plateaued in the low-3 bits/byte range after step ~7000.

## Configuration and identity

- tokenizer: byte BPE V=1024,
  `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
- baseline: V1024 / T32 / D32 / FFN32 / L19 / H2 (184,704 parameters)
- candidate: V1024 / T32 / D48 / FFN32 / L19 / H2
- parameter count: **335,424** (layout-implementation derived, host test
  cross-checked against the D32 184,704 reference; FFN stays 32 because no
  `FFN >= D` invariant exists — w1 is D×FFN, w2 is FFN×D)
- checkpoint format: NPRTCKPTV3 with tokenizer binding; D32 checkpoints are
  rejected on D mismatch by design; training started from fresh
  initialization (`initial_parameter_hash=fnv1a64:cbb15769830df32c` at
  step 0 and re-confirmed by both run reports)

## Implementation changes

None were required. The generic model, QNN graph, checkpoint, resume,
evaluator, generation, and Android paths are dimension-parameterized; D48
needed only `-Dimension 48` on the existing runners (the JNI gate accepts
even dimensions 2..256; `dimension % heads == 0` holds for 48/2).

Host validation before any device run:

- parameter count = layout formula = 335,424
- forward / loss / backward / Adam smoke on V1024/T32/D48/FFN32/L19/H2:
  shapes correct, all tensors/gradients/Adam state finite
- `nicopedia_resume_test` PASS (D mismatch rejection, V3 identity)
- on-device 2-step smoke: 491/491 graph executes successful, finite,
  `cpu_fallback=false`, Build ID `2.48.40.260702151143`, APK audit clean
  (all QNN .so hashes match the pinned SDK, no 2.47 contamination)

## Training trajectory (fresh 0 -> 8000)

| segment | runner | notes |
| --- | --- | --- |
| 0 -> 4000 | fresh | PASS; 196,393/196,393 executes, finite, no fallback |
| 4000 -> 6750 | resume | runner watchdog killed it during device Doze; all 11 checkpoints pulled and fully verified on host (identity + registry + finite) |
| 6750 -> 8000 | recovery resume | PASS; 61,250/61,250 executes, finite, no fallback |

Cumulative exposure at step 8000: 2,048,000 target BPE tokens,
5,491,256 target UTF-8 bytes, 46,616 unique chunks, 1,949 unique articles —
**identical to the D32 8000-step run in all four fields**, confirming the
data-order RNG (fixed splitmix64 seed 20260806) does not depend on D.

Endpoint losses: 7.163345 (step 1) -> 4.375782 (step 4000) -> 4.280010
(step 8000). All steps finite; final parameter hash
`fnv1a64:84be183b3a5e6c01`; step-8000 checkpoint SHA-256
`f6e41b274989f78fb80fe04fa06d4f95a670d303474ddba8476c2db0051e6170`.

Device environment incidents (classified, not model failures):

- after ~01:00 local the phone (screen off, effectively on battery due to
  its charge limit) entered deep Doze; the held partial wake lock does not
  protect against deep Doze, so HTP progress degraded to maintenance-window
  cadence (~19 min per 250 steps) and then the host stall watchdog
  (initially 300 s default, then 1,200 s) terminated two runs
- recovery followed the ADB/checkpoint protocol: no immediate relaunch,
  remote heartbeat checked, device checkpoints pulled, NPRTCKPTV3 fully
  verified on host, highest verified checkpoint (step 6750) used as the
  resume point
- the final 6750 -> 8000 segment ran with `dumpsys deviceidle disable` +
  `svc power stayon true` (both restored afterwards); its wall timings are
  therefore power-limited environment values, not architecture cost
- one ADB transport failure during evaluation was recovered the same way;
  the affected eval was re-run to completion

## Learning curve (64+64 held-out, same split/selection/evaluator)

| step | D32 val bpb | D48 val bpb | Δval | D32 dev bpb | D48 dev bpb | Δdev |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 3.631696 | 3.661321 | +0.029625 | 4.151628 | 4.180040 | +0.028412 |
| 500 | 3.351819 | 3.391310 | +0.039491 | 3.707824 | 3.681405 | -0.026419 |
| 750 | 3.260099 | 3.258582 | -0.001517 | 3.566046 | 3.562250 | -0.003796 |
| 1000 | 3.275601 | 3.264491 | -0.011110 | 3.526902 | 3.547975 | +0.021073 |
| 1500 | 3.188587 | 3.169102 | -0.019485 | 3.476483 | 3.467220 | -0.009263 |
| 2000 | 3.130733 | 3.130852 | +0.000119 | 3.406925 | 3.348641 | -0.058284 |
| 2500 | 3.104291 | 3.130337 | +0.026046 | 3.398181 | 3.392855 | -0.005326 |
| 3000 | 3.090014 | 3.106552 | +0.016538 | 3.414745 | 3.344518 | -0.070227 |
| 3500 | 3.105639 | 3.127822 | +0.022183 | 3.343086 | 3.277243 | -0.065843 |
| 4000 | 3.085050 | 3.079106 | -0.005944 | 3.363386 | 3.211610 | -0.151776 |
| 4500 | 3.027949 | 3.042253 | +0.014304 | 3.293105 | 3.245541 | -0.047564 |
| 5000 | 3.039987 | 3.007022 | -0.032965 | 3.267308 | 3.250510 | -0.016798 |
| 5500 | 3.079968 | 3.084081 | +0.004113 | 3.289753 | 3.243648 | -0.046105 |
| 6000 | 3.037882 | 3.035100 | -0.002782 | 3.261467 | 3.220747 | -0.040720 |
| 6500 | 3.042824 | 3.039755 | -0.003069 | 3.248774 | 3.230129 | -0.018646 |
| 7000 | 3.018001 | 3.027405 | +0.009404 | 3.258361 | 3.198287 | -0.060074 |
| 7250 | 3.012299 | 3.032930 | +0.020631 | 3.272545 | 3.215103 | -0.057442 |
| 7500 | 3.016735 | 3.028240 | +0.011505 | 3.240731 | 3.190456 | -0.050276 |
| 7750 | 3.042431 | 3.062844 | +0.020412 | 3.298263 | 3.238917 | -0.059346 |
| 8000 | 3.051031 | 3.030828 | -0.020203 | 3.261416 | 3.214271 | -0.047145 |

D48 best validation: step 5000 (3.007022). D48 best development:
step 7500 (3.190456). D48 balanced best ((val+dev)/2): step 7500
(3.109348 vs D32 step-7500 balanced 3.128733). Both models pick step 7500
as balanced best, so the comparison below is same-step.

## 256+256 confirmation (balanced best, step 7500)

| model | validation bpb | development bpb |
| --- | ---: | ---: |
| D32 (reference) | 2.471451 | 2.732622 |
| D48 | **2.440660** | **2.705480** |
| difference | -0.030791 | -0.027142 |

Both splits improve; 0 non-finite chunks, QNN success, finite, no
fallback. Cross-tokenizer caveats do not apply (identical tokenizer).

## Runtime

Healthy-phase values (0 -> 4000 fresh segment; per update):

| metric | D32 reference | D48 |
| --- | ---: | ---: |
| QNN execute time (fused fwd/bwd + Adam) | 443.808 ms | 601.62 ms (+35.6%) |
| — fused forward/backward | — | 96.32 ms |
| — Adam | — | 505.30 ms |
| compute-active training wall | — | 2,174.28 ms |
| parameters | 184,704 | 335,424 (+81.5%) |

D48 HTP training throughput: 117.75 target BPE tokens/s, 315.93 UTF-8
bytes/s, 3,318.9 s per MiB of original target text (compute-active phase
only). Mean QNN execute time 12.28 ms/call over 196,000 training-phase
calls (49 calls per update). The overall runner wall for 0 -> 4000 was
~21,600 s including the ~12,400 s CPU reference replay diagnostic and
checkpoint pulls; the replay is evidence, not training compute. The
6750 -> 8000 recovery segment ran power-limited (5,276 ms/step wall) and
is excluded from architecture-cost comparisons.

## Generation smoke (balanced best step 7500, max 64 new bytes)

Fixed development prompt (repo-standard, 21 bytes / 12 tokens; the D32
step-7500 runs used a different private 15-byte prompt, so this is a
mode-level aggregate comparison, not a same-prompt comparison). Raw
outputs remain private.

| mode | valid UTF-8 bytes | invalid bytes | tokens | max scalar repeat | short-loop fraction | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 64 | 0 | 18 | 2 | 0.344828 | 5.84 |
| Sample (temp 1, top-K 256, seed 12345) | 63 | 0 | 23 | 0 | 0.275862 | 7.31 |

D32 step-7500 reference aggregates (private prompt): Greedy 64/0 valid
bytes with repeat run 3; Sample 64/0 with repeat run 0. All runs passed
QNN return-code, finite-output, checkpoint-anchor, and no-fallback gates.
Generation is not the primary metric; held-out is.

## Verdict

**B — D48 consistently improves or matches D32, but the cost/quality
tradeoff still needs discussion.**

- Q1: yes — at 256+256 the balanced-best checkpoint improves both splits
  (-0.031 val, -0.027 dev bpb); development improves at 10/10 late
  checkpoints at 64+64.
- Q2: mainly the late curve/plateau, not early speed (early deltas are
  ~0 or slightly worse; the advantage accrues from mid-training).
- Q3: partially — the development-split plateau was likely hidden-capacity
  limited; the validation split stays roughly tied, so a different
  bottleneck dominates there.
- Q4: quality gain is a few hundredths bpb for +35.6% QNN time per update
  and +81.5% parameters; that is a real but not decisive trade.
- Q5: D48's validation still plateaus around 3.01-3.03 bpb; with FFN32
  now narrower than D48, the FFN (and depth/data strategy) are the next
  bottleneck candidates.

No final-test access, no full-cap evaluation, no tokenizer change, no
D64/FFN/L/H/T changes, no public export of raw text, and no training
beyond step 8000 were performed. Existing D32 checkpoints were neither
overwritten nor renamed.
