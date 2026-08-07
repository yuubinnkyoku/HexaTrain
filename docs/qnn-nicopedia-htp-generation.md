# Nicopedia byte-level HTP generation

Milestone: load the HTP-trained Nicopedia real-text checkpoint
(`build/reports/nicopedia-htp-training/htp-seed<S>-l<L>-step320.ckpt`,
NPRTCKPTV1 private format) into the QNN HTP graph and run byte-level
generation: arbitrary UTF-8 prompt → bytes → HTP forward → next byte →
append → repeat.  The generation forward's numerical operations run on HTP;
CPU performs prompt encoding, the 32-byte rolling context window, sampling,
and host control.  This is **not** an NPU-only inference claim.

Design: `docs/agent/` rules (verification, QAIRT policy, device tiers,
numerical evidence) apply unchanged.

## Entry points

| Layer | Entry |
|---|---|
| C++ core (shared, host-testable) | `app/src/main/cpp/nicopedia_generation.h` |
| Device generation flow | `qnn::nicopediaHtpGeneration` in `app/src/main/cpp/qnn/qnn_transformer_training.cpp` |
| Mode | `ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA_GENERATE` (101) |
| JNI | `NativeBridge.nativeRunNicopediaGenerate` |
| Kotlin debug intent | `phonelm.mode=QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA_GENERATE` |
| CLI | `scripts/run_nicopedia_htp_generate.ps1` |
| Public exporter | `scripts/export_public_qnn_nicopedia_htp_generation_results.ps1` |
| Host test | `host_tests/nicopedia_htp_generation_test.cpp` (CI-safe, in `run_host_tests.ps1`) |

## Protocol (fixed by the HTP training milestone)

- Byte vocabulary V=256 (every byte value is a token; **no EOS**).  Context is
  always a rolling 32-byte window (`config.tokens`).  A shorter history is
  NUL-padded at the front; a longer history uses the **last 32 bytes**
  (training-window semantics).  `nicopedia_gen::buildGenerationContext`.
- Stop condition is `MaxNewBytes` (1..1024) only.
- Display is lossless: valid UTF-8 passes through; every invalid or truncated
  byte is emitted as a literal `\xNN` escape (`safeUtf8Display`); bytes are
  never dropped or replaced.
- Greedy: first argmax on ties (`greedyArgmax`).  Sampling:
  `sampleTopK` — deterministic per platform for (seed, step): splitmix64
  state → top-53-bit uniform, fp64 softmax over the temperature-scaled top-K
  logits (unique total order: logit desc, index asc), inverse-CDF draw.
  Cross-platform last-ulp `exp` differences are expected and documented, not
  claimed bit-identical.  Sampling runs on device CPU; it is not required to
  run on HTP.

## Device flow (`nicopediaHtpGeneration`)

1. Validate seed/layers/heads, `GenerateConfig` (maxNewBytes, temperature,
   topK, samplingSeed).
2. Parse the checkpoint filename `htp-seed<seed>-l<layers>-step<step>.ckpt`
   (fail closed on any deviation) and load `NPRTCKPTV1` with full
   validation: magic, header config vs `tiny_lm::Config` equality, seed/step,
   canonical registry order/names/element counts (via
   `tiny_lm::parameterRegistry` of the empty `Params` shape), trailing-byte
   check, per-value finiteness.  Registry hash `nprtParameterHash`
   (`fnv1a64:` over name || native-LE u64 count || float bytes) is returned
   and compared by the host against the training milestone's
   `final_parameter_hash` anchor (fail closed).
3. Read the prompt file (app-private `prompt.bin`), build the 32-byte
   context; report `prompt_byte_count`, `prompt_truncated`,
   `context_used_bytes`, `context_padding_bytes` (never the prompt text).
4. HTP runtime initialize + `prepareTinyTransformerTraining` (FULL,
   no Adam — the same forward+backward graph already parity-proven in
   training; lr=0 makes the step a pure forward).
5. **Fixed-prefix parity gate**: 4 deterministic prefixes
   (`nicopedia_gen::parityPrefixes`: 21-byte Japanese, 26-byte ASCII,
   32-byte byte edges incl. 0x00/0x01/0x7f/0x80/0xff/0xfe, 32-byte truncated
   UTF-8 lead 0xE3).  CPU reference (`forwardBackwardGeneralized`) vs HTP
   (`executeTinyTransformerTraining` with zero target); tolerances:
   logits maxAbs < 2e-2, probabilities maxAbs < 5e-3, all finite.
6. **AR parity gate**: 8 greedy steps from the first prefix context, both
   sides appending their own argmax; while contexts are aligned compare with
   the same tolerances.  Divergence is allowed only as numerical noise:
   blocked iff the CPU top-1/top-2 margin at the diverging step > 1e-2
   (`ar_divergence_blocked`), which fails the gate.
7. `generation_gate = parity_gate && ar_gate`.  Only when the gate passes,
   run the user-prompt autoregressive loop (greedy or `sampleTopK`) for
   maxNewBytes; every step's last-row logits must be finite or the run fails
   as `EXECUTION_NONFINITE`.
8. Report (private KEY=VALUE, `NICOPEDIA_HTP_GENERATION`): headers, seed,
   checkpoint identity (hash, elements, file bytes, finiteness), prompt
   aggregates, gates, per-prefix/per-step parity rows, generated hex +
   UTF-8 stats, timing (initialize/graph create/finalize, ms per byte,
   execute count), `cpu_fallback=false`, nan/inf flags, API trace.

## Host CLI (`scripts/run_nicopedia_htp_generate.ps1`)

```
.\scripts\run_nicopedia_htp_generate.ps1 -Model L19 -Seed 1 -Prompt 'こんにちは' `
    -MaxNewBytes 64 -Mode Greedy
```

- `-Model L6|L19`, `-Seed` (L6: 1/2/4, L19: 1), `-Prompt <string>` or
  `-PromptFile <file>` (raw UTF-8 bytes), `-MaxNewBytes 1..1024`,
  `-Mode Greedy|Sample`, `-Temperature`, `-TopK 1..256`, `-SamplingSeed`.
- Resolves the checkpoint and its `final_parameter_hash` anchor; fails closed
  before any device work if either is missing.
- Stages checkpoint + prompt under `files/checkpoints/nicopedia-generation/`
  (app-private), drives the debug intent, polls `device-test-result.txt`,
  verifies `status=SUCCESS`, `cpu_fallback=false`, gates all true,
  `checkpoint_parameter_hash` == anchor, and `generated_byte_count` ==
  maxNewBytes.
- Computes `prompt_sha256`, `generated_sha256`, `checkpoint_sha256` on the
  host; writes the annotated private report under
  `build/reports/nicopedia-htp-generation/` and aggregate events under
  `build/private-diagnostics/nicopedia-htp-generation-goal/` (events.jsonl +
  state.json).  **Prompt text and generated bytes never enter these files**;
  terminal output may show the lossless display.
- `-SelfTest` validates the lossless display, SHA-256 plumbing, and QAIRT
  pinning without a device.

## Public/private split

Public bundle `docs/results/qnn-nicopedia-htp-generation-2026-08/` is
written only by the allow-list exporter.  Published: config aggregates,
byte counts, UTF-8 validity stats, parity/AR error magnitudes and agreement
booleans, gate results, timing, thermal/battery, device model/SOC, QAIRT
build ID.  Never published: generated bytes/hex, prompt text, article or
token content, argmax byte values, checkpoints, serials/endpoints, absolute
paths, and every content fingerprint (`generated_sha256`, `prompt_sha256`,
`checkpoint_parameter_hash`).

## Verification

- CI-safe host test `host_tests/nicopedia_htp_generation_test.cpp`: context
  window semantics, greedy tie rules, sampler determinism/top-K confinement/
  temperature collapse, lossless display incl. truncation/surrogates, hex,
  window slide, AR replay determinism with a synthetic provider.
- `verify_local.ps1 -SkipAndroidBuild` then full `verify_local.ps1 -WithQairt`
  (QAIRT pinning, QNN build, `audit_qnn_apk.ps1`).
- Device runs (Tier 2/3-adjacent, explicit user approval): parity (4
  prefixes), AR (8 steps), greedy and sampling demos for L6 seed1/2/4 and
  L19 seed1; checkpoint identity vs anchors; thermal status recorded only
  (abort only on explicit EMERGENCY/SHUTDOWN).

## Results

Device: `NX741J` (SM8850, HTP V81), QAIRT 2.48.40 (build
`2.48.40.260702151143`), 2026-08-07.  Prompt `こんにちは世界` (21 bytes)
unless noted; checkpoint step 320, anchors match on every run.

| model | seed | mode | temp/topK | bytes | parity | AR(8) | gen | ms/byte |
|---|---|---|---|---|---|---|---|---|
| L19 | 1 | greedy | – | 64 | 4/4 | 8/8 | true | 39.2 |
| L19 | 1 | greedy (141 B prompt, right-aligned 32) | – | 32 | 4/4 | 8/8 | true | 40.2 |
| L19 | 1 | sample | 0.85/64 | 64 | 4/4 | 8/8 | true | 67.8 |
| L6 | 1 | greedy | – | 64 | 4/4 | 8/8 | true | 27.5 |
| L6 | 2 | greedy | – | 64 | 4/4 | 8/8 | true | 32.7 |
| L6 | 4 | greedy | – | 64 | 4/4 | 8/8 | true | 21.0 |

All runs: `cpu_fallback=false`, `ar_divergence_blocked=false`, nan/inf=false,
thermal 0/0, battery Δ=0, checkpoint parameter hash matches the per-seed
anchor recorded in the training milestone (fingerprints stay in the private
report and are not published).  Parity caps:
logits maxAbs ≤ 0.014 < 2e-2, probs maxAbs ≤ 0.0022 < 5e-3 across every
prefix/step.  AR argmax matched CPU 8/8 steps in all runs (post-divergence
margins 0.24–1.26 ≫ 1e-2 gate).

Observation: greedy byte streams for the three L6 seeds and L19 agree
byte-for-byte on the demo prompt — the 512-parameter step-320 byte model has
collapsed onto a single attractor (repeated `の` sequence); sampling
(different seed/temperature) produces diverse, often invalid-UTF-8 byte
streams shown losslessly with `\xNN` escapes.  This is an underfit-model
behavior, not a parity or numerical failure (all gates pass).

Public allow-listed bundle: `docs/results/qnn-nicopedia-htp-generation-2026-08/`
(aggregates only; excludes generated bytes, prompts, argmax byte values, and
every fingerprint per the exporter's allow list).
