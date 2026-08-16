# Nicopedia raw-byte BPE V1024 pilot

This experiment keeps the Nicopedia transformer context at 32 tokens while
replacing the legacy one-byte-per-token vocabulary with a deterministic
1024-token raw-byte BPE vocabulary. IDs 0 through 255 remain literal bytes;
IDs 256 through 1023 are 768 train-split-only binary merges. There is no UNK,
BOS, EOS, PAD, or Android SentencePiece dependency.

The canonical tokenizer model is private. Checkpoints, token caches, merge
lists, expanded token bytes, and generated raw text also remain private. Only
aggregate metrics and the tokenizer SHA-256 identity may be reported.

## Identity and formats

- tokenizer kind: `byte_bpe`
- vocabulary: 1024
- merge count: 768
- tokenizer identity:
  `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`
- token cache: `NPRTBPEV1` with uint16 token IDs and tokenizer hash binding
- checkpoint: `NPRTCKPTV3` with tokenizer kind and SHA-256 binding
- legacy V256 caches and `NPRTCKPTV2` checkpoints retain their existing
  interpretation

All V1024 cache, checkpoint, evaluation, resume, and generation paths reject
missing or mismatched tokenizer identity. Final-test caches are not created.

## Tokenizer aggregate

The tokenizer was learned from the deterministic 1,995-article train-pilot
subset only. Merge-pair counting does not cross article boundaries.

| Split | Articles | UTF-8 bytes | BPE tokens | Tokens / Unicode character | Bytes / token | Token ratio vs V256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,995 | 8,418,135 | 3,133,267 | 0.941924 | 2.686696 | 0.372204 |
| validation | 500 | 2,287,095 | 861,057 | 0.956568 | 2.656148 | 0.376485 |
| development | 508 | 2,102,683 | 838,654 | 0.979516 | 2.507212 | 0.398849 |

Train article-length token ratios have p50 0.354582 and p95 0.451385.
Unknown rate is zero. Exact byte round-trip passed for the aggregate corpus
checks and the Japanese, ASCII, newline, symbol, emoji, mixed-text, and
arbitrary-byte fixtures.

## Model and 1000-step pilot

Configuration: V1024 / T32 / D32 / FFN32 / L19 / H2, seed 1, batch 8,
learning rate 0.003, fresh initialization. Parameter count is 184,704.

- loss: 6.953825 to 5.181724
- wall time: 3,802.903 ms/update
- QNN execution time: 443.808 ms/update (fused forward/backward plus Adam)
- successful graph executions: 31,249 / 31,249
- target exposure: 256,000 tokens and 687,872 decoded bytes
- unique exposure: 7,684 chunks and 1,474 articles
- QNN return codes: success
- tensors and checkpoint state: finite
- CPU fallback: false
- checkpoints: steps 250, 500, 750, and 1000; all V3 identities verified

The device runner also performs an explicit CPU reference replay for parity
evidence. That diagnostic is separate from fallback and is not included in
the QNN execution time above.

## Short held-out evaluation

This is a diagnostic 64-validation-chunk plus 64-development-chunk evaluation,
not a full-cap or final-test result.

| Split | Token NLL | Target tokens | Target bytes | Nats / byte | Bits / byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 5.476631 | 2,048 | 4,940 | 2.270474 | 3.275601 |
| development | 4.889324 | 2,048 | 4,096 | 2.444662 | 3.526902 |

The matching HTP evaluation completed 128 / 128 graph executions with finite
outputs and no fallback. Token NLL is recorded but is not used as a
cross-tokenizer comparison metric.

For context only, the available V256/T32/D32 checkpoint is at step 12,000,
not step 1,000. On the same 64+64 chunk caps it measured 2.917272 validation
bits/byte and 3.433995 development bits/byte. Because optimizer updates and
byte exposure are not matched, this is not a controlled primary comparison.
The pilot classification is therefore **C (unclear)**.

## Generation smoke

One fixed private prompt was used with a 32-byte hard cap. Generated raw text
remains private.

| Mode | Bytes | Tokens | Invalid UTF-8 bytes | Short-loop fraction | Max scalar repeat | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 31 | 17 | 8 | 0.178571 | 2 | 0.798188 |
| Sample (temperature 0.8, top-K 64, seed 1) | 32 | 18 | 5 | 0.103448 | 0 | 1.068209 |

Both runs passed QNN return-code, finite-output, V3/tokenizer identity, and
no-fallback gates. For BPE, a decoded token that would exceed MaxNewBytes is
discarded without resampling; therefore output may be shorter than the cap.
V256 continues to require its legacy exact byte-count behavior.

No final-test access, full-cap evaluation, public export, or training beyond
1,000 V1024 steps was performed.

## Step 250–1000 learning-curve follow-up

The existing V1024 checkpoints were evaluated without further training. All
four evaluations used the same private validation/development caches, 64
chunks per split, the same evaluation order, and the tokenizer identity above.

| Step | Validation bits/byte | Delta | Development bits/byte | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 3.631696 | — | 4.151628 | — |
| 500 | 3.351819 | -0.279876 | 3.707824 | -0.443804 |
| 750 | 3.260099 | -0.091720 | 3.566046 | -0.141778 |
| 1000 | 3.275601 | +0.015503 | 3.526902 | -0.039144 |

Every evaluation completed 128 / 128 QNN graph executions with finite output,
zero QNN failures, and no CPU fallback. The validation metric regressed from
750 to 1000 while the development improvement slowed to 0.039144 bits/byte.
The requirement that both splits retain meaningful improvement was therefore
not met. No resume to step 1250 or beyond was started, and no additional
generation smoke was run.

The target equal-exposure point for V256/T32/D32 is approximately step 2687.
No checkpoint near that point exists. The closest exact-architecture V256
checkpoint is step 8000, with 2,048,000 target bytes versus 687,872 for the
V1024 run (2.98 times as much exposure). Its diagnostic 64+64 result was
2.956404 validation bits/byte and 3.439248 development bits/byte. This cannot
establish equal-exposure sample efficiency; that comparison remains skipped
rather than treating step 8000 as exposure matched.

At step 1000, V1024 measured 3,802.903 ms/update, 443.808 QNN ms/update,
14.316 QNN ms/call, 67.317 target tokens/second, and 180.881 decoded target
bytes/second. Using the mean wall time as a rough denominator, the final
750-to-1000 interval improved development by about 0.148 bits/byte per hour
while validation changed by -0.059 bits/byte per hour (negative means worse).

Follow-up classification: **D (plateau/mixed regression; stop V1024 at step
1000)**. The early curve shows that step 1000 was not simply uniformly
under-trained: improvement decayed sharply and the validation split had
already turned upward.

## User-authorized step 1000–4000 extension

The earlier stop decision above was subsequently superseded by an explicit
request to run one bounded continuation from the healthy step-1000 V3
checkpoint through step 4000. The run retained the same tokenizer and model
identity, seed, batch size, learning rate, and complete Adam state. It wrote
and verified V3 checkpoints every 250 steps and did not proceed beyond step
4000.

The diagnostic learning curve below uses the same private caches, order, and
64 validation plus 64 development chunk caps at every point. Delta is relative
to the preceding evaluated checkpoint.

| Step | Validation bits/byte | Delta | Development bits/byte | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 3.631696 | — | 4.151628 | — |
| 500 | 3.351819 | -0.279876 | 3.707824 | -0.443804 |
| 750 | 3.260099 | -0.091720 | 3.566046 | -0.141778 |
| 1000 | 3.275601 | +0.015503 | 3.526902 | -0.039144 |
| 1500 | 3.188587 | -0.087014 | 3.476483 | -0.050419 |
| 2000 | 3.130733 | -0.057854 | 3.406925 | -0.069558 |
| 2500 | 3.104291 | -0.026442 | 3.398181 | -0.008743 |
| 3000 | 3.090014 | -0.014278 | 3.414745 | +0.016564 |
| 3500 | 3.105639 | +0.015626 | 3.343086 | -0.071659 |
| 4000 | 3.085050 | -0.020590 | 3.363386 | +0.020301 |

All six new 64+64 evaluations completed 128 / 128 QNN graph executions with
finite output, zero QNN failures, and no fallback. Validation was best at step
4000; development was best at step 3500. Step 4000 is the balanced selection:
it has the best validation result and its development result is only 0.020301
bits/byte above the development-only optimum. From step 1000 to 4000 it
improved validation by 0.190552 and development by 0.163515 bits/byte, showing
that step 1000 was under-trained despite the noisy local reversal at that
checkpoint.

### Exposure and runtime

The continuation performed 3,000 optimizer updates and saw 768,000 target
tokens representing 2,060,009 decoded target bytes. It covered 21,225 unique
chunks and 1,817 unique articles during that resume interval. Cumulative
exposure through step 4000 was 1,024,000 target tokens and 2,747,881 decoded
target bytes.

- loss over the resume interval: 5.047637 to 4.514064
- wall time: 1,248.836 ms/update
- QNN execution time: 355.544 ms/update and 11.469 ms/call
- throughput: 204.991 target tokens/s and 549.848 decoded target bytes/s
- QNN graph executions: 93,000 / 93,000 successful
- parameters, Adam state, and visible outputs: finite
- CPU fallback: false

For the available V256 step-4000-to-8000 interval, the corresponding normalized
throughput was 414.531 target bytes/s. The V1024 continuation therefore
processed 32.6% more original-text bytes per wall second, while requiring
substantially more QNN time per optimizer update. This is a run-interval
efficiency comparison, not a claim about complete end-to-end wall time from
fresh initialization.

The exposure fields in a resumed training report previously counted the first
`completedSteps` entries of the deterministic order. Training itself used the
correct resumed entries, so checkpoints and model results were unaffected.
Reporting now emits both cumulative-through-checkpoint and resume-interval
token, byte, unique-chunk, and unique-article exposure.

### Equal-text-exposure comparison

V1024 step 3000 had 2,061,214 cumulative decoded target bytes, only 13,214
bytes (0.645%) above the 2,048,000 bytes of V256 step 8000. On the same 64+64
diagnostic caps:

| Model | Step | Updates | Target byte exposure | Validation bits/byte | Development bits/byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| V256 | 8000 | 8000 | 2,048,000 | 2.956404 | 3.439248 |
| V1024 | 3000 | 3000 | 2,061,214 | 3.090014 | 3.414745 |

V256 was better on validation by 0.133610 bits/byte, while V1024 was better on
development by 0.024503 bits/byte. Equal-text-exposure sample efficiency is
therefore mixed and not established as a V1024 win. No matching V256 step-4000
checkpoint exists, so equal-update comparison was skipped rather than creating
a new V256 run.

### Larger held-out confirmation and generation smoke

The selected step-4000 checkpoint was confirmed with 256 chunks per split:

| Model | Split | Target tokens | Target bytes | Nats/byte | Bits/byte |
| --- | --- | ---: | ---: | ---: | ---: |
| V1024 step 4000 | validation | 8,192 | 21,326 | 1.768060 | 2.550771 |
| V1024 step 4000 | development | 8,192 | 19,513 | 1.946601 | 2.808352 |
| V256 step 8000 | validation | 8,192 | 8,192 | 2.303163 | 3.322762 |
| V256 step 8000 | development | 8,192 | 8,192 | 2.579629 | 3.721619 |

Both evaluations completed 512 / 512 successful QNN executions, remained
finite, and used no fallback. V1024 is substantially better in this larger
diagnostic, but it also has 34.2% more cumulative byte exposure than V256, so
this is not the exposure-matched comparison.

One private prompt was generated in each mode at the selected checkpoint.
Greedy produced 31 bytes from 8 tokens (one invalid UTF-8 byte, short-loop
fraction 0.392857, scalar repeat zero, 0.496689 seconds). Sample produced 31
bytes from 11 tokens (two invalid UTF-8 bytes, short-loop fraction 0.357143,
scalar repeat two, 0.719323 seconds). Both passed QNN return-code, finite,
identity, and no-fallback gates; generated text remains private.

Extension classification: **B (promising enough to continue)**. Token
compression translated into continued held-out improvement and higher original
byte throughput, but the exposure-matched split results do not yet support an
unambiguous sample-efficiency advantage over V256.

## Overnight continuation attempt (transport-blocked)

An explicit continuation from the verified step-4000 V3 checkpoint toward the
step-8000 hard ceiling was started with the same identity and checkpoint
interval. Before training, the equal-exposure 256+256 check completed
successfully: V1024 step 3000 measured 2.585977 validation and 2.851475
development bits/byte, versus V256 step 8000 at 3.322762 and 3.721619 under the
same 256+256 condition. During the continuation, the device-side progress
reached the checkpoint-count corresponding to step 6250; those interval files
were not pulled and host-verified before the ADB transport disappeared.

The headless status poll then reported `ADB_TRANSPORT_FAILURE` while the last
observed phase was training. The physical endpoint remained unreachable after
safe ADB reconnect and server restart attempts. This is an infrastructure
transport interruption, not evidence of QNN failure, non-finite state, or
training divergence. No step-8000 report, post-4000 learning curve, equal-
exposure step-12000 comparison, or generation result is claimed from this
attempt. The last locally verified resume source remains step 4000.

## Overnight continuation completed after reconnect

The device was reconnected without restarting or replacing the active process.
The same `bpe-v1024-resume4000-step8000-overnight` trajectory subsequently
reached the declared hard ceiling at step 8000. No training beyond step 8000,
final-test access, full-cap evaluation, public export, commit, or push was
performed. The earlier transport-blocked note above is retained as an
experiment-history record; it was not a model or QNN failure.

### Step-4000 to step-8000 training

Configuration remained V1024 / T32 / D32 / FFN32 / L19 / H2, seed 1, batch 8,
learning rate 0.003, with the step-4000 NPRTCKPTV3 Adam state resumed
fail-closed. Parameter count was 184,704. The resume report was SUCCESS with
NPRTCKPTV3, tokenizer kind `byte_bpe`, and the exact tokenizer identity above.

- resume: step 4000 -> 8000, 4,000 optimizer updates
- loss: 4.788613 -> 4.342779
- interval exposure: 1,024,000 BPE target tokens and 2,743,375 UTF-8 bytes
- cumulative exposure: 2,048,000 BPE target tokens and 5,491,256 UTF-8 bytes
- cumulative unique exposure: 46,616 chunks and 1,949 articles
- interval unique exposure: 27,192 chunks and 1,867 articles
- QNN graph executions: 124,000 / 124,000 successful, failure 0
- all steps finite, final finite, CPU fallback false
- final parameter hash: `fnv1a64:1c011dc39ec1343e`
- step-8000 checkpoint SHA-256:
  `e4e87dd13658f1999cbc378ae1023958670e1f79bcd753e7270687d29a6153d7`

Observed resume-interval runtime was 6,318.389 ms/update, QNN compute was
387.848 ms/update (12.511 ms/call), target throughput was 40.517 BPE
tokens/s and 108.547 original UTF-8 bytes/s (9,660.088 s/MiB). The measured
wall interval includes the overnight device/transport conditions and is not
treated as an intrinsic architecture-only benchmark.

| Model / interval | Wall ms/update | QNN ms/update | Original UTF-8 bytes/s | Seconds/MiB |
| --- | ---: | ---: | ---: | ---: |
| V256 step 4000->8000 reference | 617.566 | 117.576 | 414.531 | 2,529.550 |
| V1024 step 4000->8000 | 6,318.389 | 387.848 | 108.547 | 9,660.088 |

### 64+64 held-out learning curve

Every checkpoint below used the same private validation/development caches,
deterministic order, 64 chunks per split, HTP execution, and tokenizer hash.
Delta is relative to the preceding listed checkpoint. These are diagnostic
caps, not final-test or full-cap results.

| Step | Validation bits/byte | Val Δ | Development bits/byte | Dev Δ | Balanced mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 3.631696 | — | 4.151628 | — | 3.891662 |
| 500 | 3.351819 | -0.279876 | 3.707824 | -0.443804 | 3.529822 |
| 750 | 3.260099 | -0.091720 | 3.566046 | -0.141778 | 3.413072 |
| 1000 | 3.275601 | +0.015503 | 3.526902 | -0.039144 | 3.401252 |
| 1500 | 3.188587 | -0.087014 | 3.476483 | -0.050419 | 3.332535 |
| 2000 | 3.130733 | -0.057854 | 3.406925 | -0.069558 | 3.268829 |
| 2500 | 3.104291 | -0.026442 | 3.398181 | -0.008743 | 3.251236 |
| 3000 | 3.090014 | -0.014278 | 3.414745 | +0.016564 | 3.252379 |
| 3500 | 3.105639 | +0.015626 | 3.343086 | -0.071659 | 3.224363 |
| 4000 | 3.085050 | -0.020590 | 3.363386 | +0.020301 | 3.224218 |
| 4250 | 3.089454 | +0.004404 | 3.283544 | -0.079842 | 3.186499 |
| 4500 | 3.027949 | -0.061505 | 3.293105 | +0.009561 | 3.160527 |
| 4750 | 3.068291 | +0.040342 | 3.298785 | +0.005680 | 3.183538 |
| 5000 | 3.039987 | -0.028304 | 3.267308 | -0.031477 | 3.153648 |
| 5250 | 3.062798 | +0.022811 | 3.277326 | +0.010017 | 3.170062 |
| 5500 | 3.079968 | +0.017170 | 3.289753 | +0.012427 | 3.184860 |
| 5750 | 3.061076 | -0.018892 | 3.276390 | -0.013363 | 3.168733 |
| 6000 | 3.037882 | -0.023194 | 3.261467 | -0.014923 | 3.149674 |
| 6250 | 3.065089 | +0.027208 | 3.257503 | -0.003964 | 3.161296 |
| 6500 | 3.042824 | -0.022266 | 3.248774 | -0.008729 | 3.145799 |
| 6750 | 3.029991 | -0.012833 | 3.290158 | +0.041384 | 3.160074 |
| 7000 | 3.018001 | -0.011990 | 3.258361 | -0.031796 | 3.138181 |
| 7250 | **3.012299** | -0.005702 | 3.272545 | +0.014184 | 3.142422 |
| 7500 | 3.016735 | +0.004436 | **3.240731** | -0.031814 | **3.128733** |
| 7750 | 3.042431 | +0.025696 | 3.298263 | +0.057532 | 3.170347 |
| 8000 | 3.051031 | +0.008600 | 3.261416 | -0.036846 | 3.156224 |

The validation best is step 7250. The development and balanced best are both
step 7500. Relative to step 4000, step 8000 is better by 0.034018 validation
bits/byte and 0.101970 development bits/byte, although the final 7500->8000
interval itself is a local regression on both splits. Thus the trajectory is
still noisy but not an early step-4000 plateau.

All 16 new evaluations (4250 through 8000) completed 128 / 128 successful
QNN graph executions, with finite outputs, zero failures, and no CPU fallback.
The existing 250 through 4000 reports have the same health result.

### Equal-original-text-exposure comparisons

The primary metric is bits per decoded UTF-8 byte; token NLL is not compared
across tokenizers.

First, V1024 step 3000 had 2,061,214 training bytes versus V256 step 8000's
2,048,000 bytes (0.645% higher for V1024). On identical 256+256 held-out
caps:

| Model | Step | Updates | Training byte exposure | Val bits/byte | Dev bits/byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| V256 | 8000 | 8000 | 2,048,000 | 3.322762 | 3.721619 |
| V1024 BPE | 3000 | 3000 | 2,061,214 | **2.585977** | **2.851475** |

This is a strong positive signal for V1024 at nearly equal original-text
exposure. It is not an equal-update comparison.

Second, the nearest V1024 checkpoint to V256 step 12000's 3,072,000 training
bytes was step 4500 at 3,090,597 bytes (0.605% higher). The 256+256 results
were:

| Model | Step | Updates | Training byte exposure | Val bits/byte | Dev bits/byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| V256 | 12000 | 12000 | 3,072,000 | 3.320735 | 3.692554 |
| V1024 BPE | 4500 | 4500 | 3,090,597 | **2.497973** | **2.752471** |

The balanced-best V1024 step 7500 was independently confirmed on 256+256:
validation 2.471451 and development 2.732622 bits/byte. All three 256+256
evaluations had 512 / 512 QNN successes, failure 0, finite outputs, and no
CPU fallback.

### Generation smoke

Using the balanced-best step-7500 checkpoint and one private prompt per mode,
with `MaxNewBytes=64`, the V1024 runs passed the HTP-native generation gate;
the same smoke was also run on the existing V256 step-12000 checkpoint. The
prompt and generated bytes remain private.

| Model | Mode | Runs | Generated bytes | Generated tokens | Invalid UTF-8 bytes | Short-loop fraction | Max scalar repeat | Elapsed (s) | QNN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1024 step7500 | Greedy | 1 | 64 | 41 | 0 | 0.200000 | 3 | 34.294 | 70/70 |
| V1024 step7500 | Sample | 1 | 64 | 21 | 0 | 0.413793 | 0 | 35.329 | 50/50 |
| V256 step12000 | Greedy | 1 | 64 | 64 | 1 | 0.620690 | 2 | 33.250 | 92/92 |
| V256 step12000 | Sample | 1 | 64 | 64 | 2 | 0.310345 | 2 | 30.685 | 92/92 |

All four runs had finite outputs, CPU fallback false, focus takeover 0, and
the exact checkpoint/model identity for their respective model (including the
V1024 tokenizer hash). These one-run generation observations are smoke
evidence only and do not override the held-out quality metrics.

### Overnight continuation verdict

The step-1000 D classification was an early local-curve decision; the
step-1000->4000 continuation already recovered the validation reversal, and
step-4000->8000 retained a lower balanced held-out score with strong
equal-exposure wins over V256. The measured continuation wall throughput was
not faster than the available V256 reference (108.547 versus 414.531 original
bytes/s), so the runtime advantage is not established for this overnight
interval. The appropriate classification is **B (promising enough to
continue)**: V1024 is a strong quality/sample-efficiency candidate, but the
runtime tradeoff and noisy late curve require a separate decision before any
longer run.
