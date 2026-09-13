# HexaTrain V1024/D64/FFN128 untouched final evaluation

## Purpose

This is a read-once generalization check of the frozen best-observed recipe
from HPO. It is not HPO. The final result must not be used to change the
learning rate, target, decay start, schedule shape, architecture, seed list,
checkpoint step, or candidate selection.

## Frozen protocol

The protocol was frozen at `2026-09-01T12:23:24.5886056Z`, before any of the
four checkpoints was evaluated on the final split. The evaluation order is:

1. A1: C1500 seed 1, exact step 8000
2. B1: S4000 seed 1, exact step 8000
3. A2: C1500 seed 2, exact step 8000
4. B2: S4000 seed 2, exact step 8000

The primary metric is bits per original UTF-8 byte. It is computed as total
token NLL in nats divided by the original UTF-8 byte lengths represented by
the target BPE tokens, then divided by `ln(2)`. The two final-cache shards are
combined by total NLL and byte count, not by averaging shard bpb values.

The model is V1024/T32/D64/FFN128/L19/H2 with 758,528 parameters. The tokenizer
is `byte_bpe`, identity
`sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`.
Evaluation uses the existing HTP-native teacher-forced runner with QAIRT Build
ID `2.48.40.260702151143`. QNN success, finite output, zero graph failures, and
no CPU fallback are required. The host CPU evaluator is identity-only
diagnostic evidence and is not a fallback or primary result.

Full final evaluation was rejected before opening the split as extremely
high-cost for the current one-window-per-QNN-graph evaluator. The final split
contains 2,931 articles and 12,765,573 cleaned UTF-8 bytes, corresponding to
roughly 160,000 T32 BPE windows per checkpoint and roughly 640,000 serial graph
executions across the four frozen checkpoints.

Instead, all four checkpoints use the same pre-frozen deterministic 512-window
sample. Eligible final articles are sorted by the existing stable SHA-256
subset key; non-overlapping T32 byte-BPE windows are enumerated within each
article; incomplete tails are discarded; the first 512 windows are selected.
No random seed is involved. Windows 0--255 and 256--511 form two runner shards.
The exact private article hashes/window indices and selection hash are saved in
the ignored machine-readable final-evaluation artifacts before model
evaluation.

Materialization reverified every source-file SHA-256 and reproduced the corpus
aggregate exactly: 2,931 final articles, 12,765,573 cleaned UTF-8 bytes, and
428 global exact-text deduplications. The frozen sample contains 512 windows
from 20 articles, 16,384 target tokens, and 45,103 original UTF-8 target bytes.
Its selection hash is
`sha256:cdc1fb57bbfefdcf23856643dfa4495a0de03f07daf326899b86a3c328787236`.

## Why final was untouched

Before this task, repository reports, HPO artifacts, and private cache
inventory consistently recorded final-test as unopened. No final cache was
present. Prior HPO used train, validation, and development data only. The
sample algorithm and checkpoint list were frozen before final sample
materialization, and exact selected items are frozen before the first model
evaluation.

## Final split identity

Dataset: Nicopedia data, version `2024-11-25`.

- Source aggregate: `sha256:b3185ea689ffa64c71f5ea8fe25f3779dbf1f0cd0d103b6e2fe05792e635bf86`
- Corpus aggregate: `sha256:891d2cf3f9fd36acf2f9ccbd8865c4a07293d2d1aba454d9e501a29c606779b7`
- Assignment: stable article-level SHA-256 split; final bucket 9900--9999
- Dedupe: exact cleaned-text SHA-256 before split assignment
- Expected final articles: 2,931
- Expected final cleaned UTF-8 bytes: 12,765,573

## Frozen recipes

C1500 uses constant LR 0.0015 through exact step 8000. S4000 uses LR 0.0022
through step 4000 and linear decay to LR 0.0001 at exact step 8000. S4000 is
the frozen best-observed recipe from HPO.

## Checkpoint identities

| ID | Seed | Recipe | Step | Checkpoint SHA-256 | Parameter hash |
| --- | ---: | --- | ---: | --- | --- |
| A1 | 1 | C1500 | 8000 | `e71c8399c61ca10daa32c71a9fc362a42ecc52bd5ba8869e0724f85231b6ce4d` | `fnv1a64:3ee6512fa4097587` |
| B1 | 1 | S4000 | 8000 | `31337cd922dfdf71888bb74ac752dee70f3043c7b8b11232a70fe909861a9f12` | `fnv1a64:ab1004ac6712b758` |
| A2 | 2 | C1500 | 8000 | `7444019ca986d328ad28e1b4eae0567accd1ddfb7ae1d1fb0bab28862983642e` | `fnv1a64:cda26ed8680db2b0` |
| B2 | 2 | S4000 | 8000 | `fd517e4340415c7b804eddbb787e944c76d30921bb8864a84b9d8497a27e1bd1` | `fnv1a64:c145c8728713d36a` |

All four checkpoint headers match the frozen seed, step, architecture, and
tokenizer. Independent host decoding confirms finite parameters and the listed
parameter hashes. Their existing exact-step HTP reports confirm QNN success,
finite output, no fallback, and zero graph failures.

## Validation reference frozen before final

| Seed | C1500 Balanced bpb | S4000 Balanced bpb | Delta S4000 - C1500 |
| ---: | ---: | ---: | ---: |
| 1 | 2.481467162 | 2.413116060 | -0.068351102 |
| 2 | 2.475010281 | 2.405907509 | -0.069102772 |

Mean Val/Dev delta: **-0.068726937 bpb**.

## Results

All four frozen checkpoints completed on the same sample in the frozen order.
Final bpb combines the two shards by NLL and original UTF-8 target bytes.

| Seed | C1500 Final bpb | S4000 Final bpb | Delta S4000 - C1500 |
| ---: | ---: | ---: | ---: |
| 1 | 2.273236727 | 2.206494070 | -0.066742657 |
| 2 | 2.262120899 | 2.203432612 | -0.058688287 |

Mean Final delta: **-0.062715472 bpb**.

## Paired deltas and cross-seed mean

The schedule improves both frozen seed pairs. The predeclared heuristic
classification is **strong final confirmation** because both deltas are
negative and their mean is at most -0.03 bpb. This is not a statistical
significance claim: only two seeds were evaluated.

## Validation-vs-final gain

The frozen mean Val/Dev schedule gain is -0.068726937 bpb. The mean Final gain
is -0.062715472 bpb. The signs agree and descriptive gain retention is
`abs(-0.062715472) / 0.068726937 = 0.9125`, or about **91.3%**. Gain retention
is descriptive and is not a formal statistic.

## Seed 1 result

C1500 final bpb is 2.273236727 and S4000 final bpb is 2.206494070. The paired
delta is -0.066742657 bpb.

## Seed 2 result

C1500 final bpb is 2.262120899 and S4000 final bpb is 2.203432612. The paired
delta is -0.058688287 bpb.

## Systems

All four evaluations returned QNN success, finite checkpoint/output tensors,
zero graph failures, and no CPU fallback. Each executed 512 HTP graphs, for
2,048 total. QAIRT Build ID was `2.48.40.260702151143`. Android thermal status
was 0 before and after every evaluation; observed battery temperature was
34--36 degrees C. Summed device evaluation time was 300.696 seconds; the
serial span from the first instrumentation start through the fourth HTP report
was 492.975 seconds. Training performed by this task: **0 steps**.

## Caveats

This is a pre-frozen deterministic 512-window sample, not the full 2,931-
article final split. The sample contains 45,103 original UTF-8 target bytes
from 20 articles. The current evaluator's one-window-per-graph cost made full
final evaluation impractical. The n=2 classification is heuristic; it does not
establish statistical significance, global optimality, or proven optimality.

## Conclusion

The untouched final split sample reproduces the schedule gain for both seeds.
S4000 remains the frozen best-observed recipe from HPO. No final-driven HPO,
checkpoint fishing, seed selection, architecture change, or new training was
performed. The final split must not be reused to tune this recipe. Further
architecture research should start with a new Val/Dev workflow.

Later, HVX Muon + Aux Adam was promoted as the primary quality baseline
beside Adam S4000 (control). That freeze did not open this final sample.
See `docs/nicopedia-hvx-muon.md` ("Formal baseline freeze (HVX Muon)").
