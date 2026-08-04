# L19 readout / representation diagnosis, August 2026

This bundle is a host-only CPU diagnosis of why the L19 model fails its
generation quality gate. It does not open the AR_FINAL_HOLDOUT_V3 dataset
(hash verified only: fnv1a64:aa5081e6df658b4a) and performs no device, HTP, or QNN work.
All numbers come from the checked-in CPU reference implementation
(tiny_language_model_cpu.cpp), regenerated deterministically.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control), the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe. At three checkpoints per config (AR-selected step, best
token-exact step, final step 320) a hidden-state observer extracts teacher
forced features for every layer (22 representations on L19: embedded input,
19 block outputs, pre/post final-layer-norm; 21 on L18). A 32-way linear
softmax probe (Adam lr=0.01, 2000 steps, calibration step selection) is
trained per layer on TRAIN rows only. Free-running rollouts are scored with
the current head, with the probe, and with the probe evaluated on head
contexts (drift analysis). Additionally the final output head is retrained
three ways on the step-320 checkpoint: warm-start (A), re-init (B), and
bias-only (C), always freezing everything except the trained parameter.

Dataset roles follow the pinned protocol: TRAIN = probe/head learning,
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened.

## Anchor integrity

All trajectory anchors (AR_DEVELOPMENT_V3 token/sequence exact and NLL at the
selected and final steps) match the pinned bundle values; integers match
exactly and NLL matches within 1e-6 (float32-limited). See
trajectory-anchors.csv.

## Verdict

| configuration | AR_DEV selected -> final |
|---|---|
| L19_SEED_1: AR_DEV 14/32 tok at selected step -> 30/32 tok (2/8 seq) at step 320 |
| L19_SEED_2: AR_DEV 20/32 tok at selected step -> 63/32 tok (6/8 seq) at step 320 |
| L19_SEED_4: AR_DEV 22/32 tok at selected step -> 46/32 tok (6/8 seq) at step 320 |
| L18_SEED_2_CONTROL: AR_DEV 18/32 tok at selected step -> 65/32 tok (8/8 seq) at step 320 |

Cause classification (fixed thresholds, never tuned):
**DEEP_DEGRADATION**

pooled_head_fr=201 pooled_final_probe_fr=113 seed_wins_ge5=0 control_delta=-32 best_intermediate_pooled=432 intermediate_wins_ge5=3 deep_multi_checkpoint=true pooled_train_tf=87 pooled_dev_tf=264

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (READOUT_PROBE_V1) before any
results were produced.

## Files

- dataset-anchors.csv - dataset roles and hash pins
- trajectory-anchors.csv - regenerated trajectory vs pinned anchors
- baseline-current-head.csv - current head TF/FR metrics per checkpoint
- probe-selection.csv - per-layer probe training results and selected steps
- probe-layer-curve.csv - probe free-running curve vs head per layer
- probe-training-grid.csv - full 81-point calibration grid per probe
- head-retraining.csv - output-head retraining A/B/C results
- representation-metrics.csv - eta2/effective-rank/norm/agreement per layer
- head-geometry.csv - output head row norms, singular values, effective rank
- decision.csv - cause classification
- summary.csv - head TF/FR summary rows per checkpoint
- manifest.json - SHA-256 allow-list manifest