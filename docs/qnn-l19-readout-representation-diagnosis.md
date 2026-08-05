# L19 readout / representation diagnosis

## Scope and classification

This investigation answers one question: why does the L19 model fail its
generation quality gate — is the failure a readout problem (the final
representation still holds the answer but the output head cannot read it), a
generalization problem (the internal representation itself did not
generalize), or a deep-representation problem (the representation degrades
with depth)? The answer is decided by preregistered fixed thresholds
(protocol `READOUT_PROBE_V1`) over linear softmax probes trained per layer.

The diagnosis is host-only: deterministic CPU replay of the pinned
trajectories, no device, HTP, or QNN execution. `AR_FINAL_HOLDOUT_V3` stays
unopened (hash `fnv1a64:aa5081e6df658b4a` verified only). The public evidence
is in
[`qnn-l19-readout-representation-diagnosis-2026-08`](results/qnn-l19-readout-representation-diagnosis-2026-08/README.md).

**Classification: `DEEP_DEGRADATION`.** The next-token information that is
trivially readable at the embedding layer is progressively destroyed by the
deeper layers: a linear probe on the embedded input achieves perfect
free-running prediction on the development partition (144/144 tokens, all
configurations and checkpoints), while the same probe on the final
representation reaches only 40-49/144. The current output head (itself a
linear map on the final representation) reaches 50-86/144, and retraining it
in three frozen-backbone variants does not close the gap.

## Method

For each pinned configuration (L19 seeds 1/2/4 and the L18 seed-2 control)
the canonical 320-step trajectory is regenerated from the CPU reference
exactly as in the margin goal (anchors re-verified). At three checkpoints per
config — the AR-selected step (16/4/12/4), the best token-exact step
(32/128/80/160), and the final step 320 — hidden states are extracted for
every representation (22 on L19: embedded input, 19 block outputs, pre/post
final-layer-norm; 21 on L18).

Per layer and checkpoint a 32-way linear softmax probe is trained on the 32
TRAIN rows only (Adam lr=0.01, 2000 full-batch steps, features z-scored with
TRAIN-row statistics), the step is selected on MARGIN_CALIBRATION_V1 by
calibration CE with token-exact tie-break, and the selected probe is scored
once on MARGIN_DEVELOPMENT_V1 (teacher-forced, free-running, and on head
contexts). Dataset roles are pinned: TRAIN = learning, MARGIN_CALIBRATION_V1
= selection, MARGIN_DEVELOPMENT_V1 = evaluation, AR_FINAL_HOLDOUT_V3 =
unopened. 151 probes are trained in total (12 checkpoints; final checkpoints
get all layers, non-final checkpoints get 8 representative layers), plus 12
frozen-backbone head-retraining runs (warm-start / re-init / bias-only at
step 320, Adam lr=0.003, 320 steps, selection on calibration).

## Evidence

Per-layer probe free-running token exact on MARGIN_DEVELOPMENT_V1 at step 320
(monotone decay with depth in every configuration):

| representation | L19_S1 | L19_S2 | L19_S4 | L18 control |
|---|---|---|---|---|
| L0 embedded input | 144/144 | 144/144 | 144/144 | 144/144 |
| L1 block out | 126/144 | 127/144 | 112/144 | 109/144 |
| L4 block out | 98/144 | 77/144 | 98/144 | 77/144 |
| L8 block out | 90/144 | 54/144 | 77/144 | 32/144 |
| L12 block out | 50/144 | 53/144 | 46/144 | 38/144 |
| L16 block out | 61/144 | 43/144 | 48/144 | 30/144 |
| L19 block out | 52/144 | 29/144 | 32/144 | 28/144 |
| POST_LN_FINAL (final) | 49/144 | 40/144 | 45/144 | 47/144 |
| current head (linear, final) | 50/144 | 65/144 | 86/144 | 60/144 |

The probe on the embedding predicts every development rollout token exactly
(from the token identity alone; the development partition is disjoint from
TRAIN by case, prefix, and sequence), and the linear readout quality decays
monotonically with depth at every checkpoint of every configuration
(`probe-layer-curve.csv`, `probe-training-grid.csv`). The head is a linear
map applied to the final representation, so the readout-failure hypothesis
(information intact at the output, head cannot read it) is directly tested by
the final-layer probe; the final-layer probe does not beat the head
(pooled 113 vs 201), and the best intermediate probe beats the final probe by
319 pooled tokens with 3 of 3 seeds above the 5-token threshold, confirmed at
multiple checkpoints — `DEEP_DEGRADATION` by the fixed rules.

## Head retraining does not fix it

The 12 frozen-backbone runs show no consistent L19 improvement
(`head-retraining.csv`): the warm-start variant re-selects step 0 for all
configs (no improvement at all), re-init improves the L18 control 60→68 but
moves L19 seeds by -6/+1/-5, and bias-only reaches 69 (seed 2) and 85 (seed
4) while leaving seeds 1 and the control at baseline... the control gains the
most under bias-only (60→72, sequence exact 9→11). Retrained heads lower dev
NLL in most configurations (better calibration) but do not change the
free-running exact counts enough to matter; the bottleneck is not the head.

## Interpretation

The trained model's deep stack progressively loses the input-token identity
that a linear readout needs; the information present at the embedding is no
longer linearly accessible at the output of the network, and the head
retraining variants cannot recover it. Both L19 and the L18 control exhibit
the same decay, so the phenomenon is not specific to the L19 depth; the
diagnosis therefore does not select a head-side or checkpoint-selection
change, and no HTP/device run is scheduled. The final holdout remains
unopened.

> 後続監査による修正（2026-08, PROBE_OPTIMIZATION_AUDIT_V1）: 「深度とともに
> 単調減少」は legacy Adam probe の artifact として一部修正された。同特徴・
> 同 partition で PCA-whitened + L-BFGS の canonical solver を用いると曲線は
> 非単調（spearman ρ≈−0.05）となり、legacy の単調減衰は最適化不足に帰属
> する。深層ほど線形読み出し情報が根本的に無い、という主張は撤回し、
> 標準化・最適化が不十分な probe での観測として言い換える。
> また「legacy Adam は両 tap で収束する」も一部修正され、legacy の最終勾配
> は最適性から遠く、信頼できる probe には canonical solver が必要とされた。

## Reproduction

- Probe run: `scripts/run_l19_readout_probe.ps1` (optionally `-SelfTest`),
  reports under `build/reports/qnn-readout-representation-diagnosis` (private
  hidden-state cache under `build/reports/qnn-readout-probe/private-hidden`).
- Public export:
  `scripts/export_public_qnn_l19_readout_results.ps1` (optionally
  `-SelfTest`).
- Host tests: `scripts/run_host_tests.ps1` (probe self-test incl. a
  probe-learning regression guard).
