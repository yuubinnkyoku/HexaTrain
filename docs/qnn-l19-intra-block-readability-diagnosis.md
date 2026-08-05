# L19 intra-block readability diagnosis

## Scope and classification

> **現在の状態: learned-probe局在判定はsuperseded。** TRAIN row契約不具合により、
> 本文のprobe絶対値とtap/block低下数を現在の原因証拠として使用しない。
> observer、head clone、直接forward介入は独立である。現在の原因判定は
> [`qnn-l19-seed-instability-root-cause.md`](qnn-l19-seed-instability-root-cause.md)
> を正本とする。

The previous milestone
([`qnn-l19-readout-representation-diagnosis.md`](qnn-l19-readout-representation-diagnosis.md))
established `DEEP_DEGRADATION`: the next-token identity that a linear probe
reads trivially from the embedding (144/144 development tokens) is destroyed
by the deep stack, and retraining the output head does not recover it. This
milestone asks the follow-up question: **inside a block, where does the
linear readability disappear** — at the attention block, at the FFN, at the
layer norms, or as a smooth coordinate drift that a linear map could undo?

The answer is decided by preregistered fixed thresholds (protocol
`INTRA_BLOCK_READABILITY_V1`, `thresholds_fixed_before_results=true`) over
per-tap linear softmax probes, cross-tap transfers, and alignment fits, on
the final checkpoint (step 320) of the pinned configurations only.

The diagnosis is host-only: deterministic CPU replay of the pinned
trajectories, no device, HTP, or QNN execution. `AR_FINAL_HOLDOUT_V3` stays
unopened (hash `fnv1a64:aa5081e6df658b4a` verified only). Dataset roles are
pinned: TRAIN = probe fitting (32 rows), MARGIN_CALIBRATION_V1 = probe step
selection (144), MARGIN_DEVELOPMENT_V1 = evaluation (144). The public
evidence is in
[`qnn-l19-intra-block-readability-2026-08`](results/qnn-l19-intra-block-readability-2026-08/README.md).

**Classification: `ATTENTION`.** In the deep band (blocks 11-18), the
attention block output loses ≥5 tokens of linear readability relative to its
input in 6 of 8 blocks in 2 of 3 L19 seeds (0/6/6), the FFN in 4 of 8 blocks
in 2 seeds (0/4/4), and the norms in one seed (5/8 each); alignment fits
classify 9-13 of 16 deep-band pairs per seed as genuine information loss
(not a coordinate transform) with the L18 control at 7/16 (guard OK); and no
block shows residual-stream overwrite in any configuration.

## Method

For each pinned configuration (L19 seeds 1/2/4 and the L18 seed-2 control)
the canonical 320-step trajectory is regenerated from the CPU reference
exactly as in the margin goal and the readout milestone (anchors re-verified
against the pinned trajectory table). On the final checkpoint, features are
extracted at 7 taps per block — NORM1, ATT_UPDATE, AFTER_ATTN, NORM2,
FFN_UPDATE, AFTER_FFN, plus CTX/RELU at the five representative blocks
{1,4,8,12,16} — and at EMBEDDING (126 taps on L19, 120 on L18).

Per tap, a 32-way linear softmax probe is trained on the 32 TRAIN rows only
(Adam lr=0.01, 2000 full-batch steps, features z-scored with TRAIN-row
statistics), the step is selected on MARGIN_CALIBRATION_V1 by calibration CE
with token-exact tie-break, and the selected probe is scored once on
MARGIN_DEVELOPMENT_V1. Independent probes: 498 (3×126 + 120; aliases share
probes). Cross-tap transfers evaluate a source probe on destination
features (coarse: block-input→AFTER_ATTN and AFTER_ATTN→AFTER_FFN, raw +
renormalized; fine: 6 intra-block pairs at representative blocks and the
last block): 444 evals. Alignments fit LS and Procrustes maps between the
TRAIN feature Gramians of the coarse pairs and re-score the mapped probe on
dev: 150 fits. Free-running rollouts on 3 taps per config (max-drop block
input, max-drop block output, POST_LN_FINAL): 12. Token-only baselines: 3
(seen/unseen-pinned). Trajectory regenerations: 4. All counts equal the
preregistered budget exactly.

Fixed thresholds (not changed after results): per-op dev-TF drop ≥5/144,
transfer gap ≥5, attn ratio >1.0 with cos <−0.5 (residual overwrite),
pair verdicts CT when aligned residual ≤1 and recovery ≥0.75, IL when
recovery ≤0.25 and residual loss ≥3; deep-band majority = 6/8 blocks per
seed, ≥2/3 seeds; pair-level IL/CT majority = ≥10/16 pairs, ≥2/3 seeds;
verdict priority RESIDUAL_OVERWRITE → ATTENTION → FFN → LAYERNORM →
COORDINATE_DRIFT → LINEAR_INFO_LOSS → MIXED → CUMULATIVE. Head-clone parity
(probe vs head on identical features) must pass |Δlogit| ≤1e-4 with 0
argmax/rank/exact mismatches on every configuration.

## Evidence

Per-op dev-TF drop counts in the deep band (blocks 11-18, 8 blocks per
seed; drop = probe exact on the op input minus probe exact on the op
output, ≥5 tokens):

| operation | L19_S1 | L19_S2 | L19_S4 |
|---|---|---|---|
| attention (NORM1 → AFTER_ATTN) | 0/8 | 6/8 | 6/8 |
| FFN (NORM2 → AFTER_FFN) | 0/8 | 4/8 | 4/8 |
| norm1 (block input → NORM1) | 5/8 | 0/8 | 0/8 |
| norm2 (AFTER_ATTN → NORM2) | 5/8 | 0/8 | 0/8 |
| residual overwrite (attn / ffn) | 0/0 | 0/0 | 0/0 |

In seeds 2 and 4 the loss concentrates at the attention block output; in
seed 1 it concentrates at the norms. No block in any configuration satisfies
the residual-overwrite criterion (update norm > input norm with cosine
< −0.5): the residual stream is never overwritten.

Alignment verdicts on the 16 deep-band pairs per seed (fit on TRAIN,
re-scored on dev):

| verdict | L19_S1 | L19_S2 | L19_S4 | L18 control (10..17) |
|---|---|---|---|---|
| COORDINATE_TRANSFORM | 1 | 1 | 0 | 0 |
| INFORMATION_LOSS | 9 | 11 | 13 | 7 |
| MIXED | 6 | 4 | 3 | 9 |

The deep band is not a smooth coordinate drift: only 1 pair per seed
(maximum) is a pure linear map of its input. Information loss is at
majority (≥10/16) in seeds 2 and 4. The L18 control, evaluated over its
blocks 10-17, reaches IL 7/16 — below the guard (<9) — so the deep-band IL
concentration is depth-specific, not an artifact of the probe/alignment
machinery.

Free-running rollouts on the three protocol-selected taps (max-drop block
input, max-drop block output, POST_LN_FINAL) versus the head:

| tap | L19_S1 | L19_S2 | L19_S4 | L18 control |
|---|---|---|---|---|
| shallow tap (b00/b01 AFTER_FFN or EMBEDDING) | 126/144 | 127/144 | 144/144 | 115/144 |
| max-drop block output | 105/144 | 87/144 | 112/144 | 75/144 |
| POST_LN_FINAL (deepest tap) | 49/144 | 40/144 | 45/144 | 47/144 |
| current head (free-running) | 50/144 | 65/144 | 86/144 | 60/144 |

The deepest tap reads at head level or below; the shallow taps read at
87-144. The largest single-block drop is at a shallow block in every
configuration (blocks 0-2, 9-19 tokens), while the deep band accumulates the
systematic per-op pattern above.

Head-clone parity passes on every configuration: the head-equivalent probe
reproduces the head's logits within 9.1e-7 max |Δlogit| with zero argmax,
rank, and exact-token mismatches — the probe machinery is not the source of
any measured drop. Trajectory anchors match the pinned table (token/seq/NLL
within 1e-6) and dataset anchors match the pinned hashes on all 4
configurations; every probe and transfer ran to a finite outcome
(no PROBE_NONFINITE).

## Interpretation

The deep readout degradation is decomposed: it is not a head-side readout
failure (clone parity), not a residual overwrite (0 blocks), and not a
coordinate drift (1/1/0 CT pairs). It is op-level information loss that
destroys linear readability at the attention block output in 2 of 3 seeds
(6/8 blocks each, ≥5 tokens) and at the norms in the remaining seed, with
the FFN losing a smaller share (4/8) in the attention-dominant seeds. The
alignment machinery confirms the loss is not recoverable by any linear map
of the source features (IL 9/11/13 of 16 pairs), and the L18 control (IL
7/16) shows the concentration is depth-specific. By the fixed priority
rules this is `ATTENTION`: the attention write path is the primary
in-block destroyer of the linear next-token identity in the L19 deep band.

As with the readout milestone, this diagnosis does not select an HTP/device
run; the fix direction is the attention path (context mixing that writes
non-linearly-identifiable features into the residual stream), to be
investigated in a training-side experiment. The final holdout remains
unopened.

> **修正注記（2026-08-05、probe-optimization audit による再検証）**
> 本診断の probe は z-score 特徴 + legacy Adam 学習で得られた。その後、
> 同じ tap を用いた canonical 学習（PCA whitening 特徴・whitened 空間のみへの
> L2 λ=1e-4・gauge 固定・L-BFGS 収束判定）では、CTX と ATT の probe は同一の
> 収束点に到達する（dev token exact CTX==ATT 16/16, 32/32, 52/52, 30/30）。
> ブロック内の attention 書き込みによる readability 低下（NORM1→AFTER_ATTN で
> dev exact ≥5 低下）は canonical でも再現し、SEED_2 4/9、SEED_4 5/8、
> L18 6/9 のブロックで維持される。したがって「attention 書き込み経路が
> 線形 next-token 同一性を破壊する」という本診断の結論は不変である。
> ただし深度方向の曲線は canonical では非単調であり（EMBEDDING 144 から深層で
> 26〜65 の帯域、spearman ρ −0.49〜+0.13）、legacy で見えた単調な深層低下は
> 一部 legacy-Adam の artifact を含む。検証内容は
> [`qnn-l19-probe-optimization-audit.md`](qnn-l19-probe-optimization-audit.md)
> （PROBE_OPTIMIZATION_AUDIT_V1 version 6、
> hash `fnv1a64:b36b4745b9b4807f`）を参照。

## Reproduction

> **追加修正（2026-08-05、seed-instability root-cause再監査）**
> TRAIN probe row生成に4行の契約不整合が判明したため、学習済みprobeに基づく
> block/tap別の絶対scoreと低下数は再生成まで原因証拠から除外する。observer、
> head clone parity、直接forward介入はこの不具合に依存しない。Attention経路の
> 因果的重要性は、別のAttention-zero対FFN-zero学習介入で再構築した。

- Probe run: `scripts/run_l19_intra_block_readability.ps1` (optionally
  `-SelfTest`; `--run --report-root`), reports under
  `build/reports/qnn-intra-block-readability` (private tap cache under
  `build/reports/qnn-intra-block-readability/private-taps`).
- Public export:
  `scripts/export_public_qnn_l19_intra_block_results.ps1` (optionally
  `-SelfTest`).
- Host tests: `scripts/run_host_tests.ps1` (intra-block self-test incl. a
  solver/alignment regression guard).
