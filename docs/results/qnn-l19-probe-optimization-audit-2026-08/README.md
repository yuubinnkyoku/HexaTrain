# L19 probe-optimization audit, August 2026

This bundle is a host-only CPU audit of the probe-optimization hypothesis for
the previously reported CTX_CONCAT vs ATT_UPDATE dev-token-exact drop. The
published READOUT_PROBE_V1 legacy anchors (24/6, 37/24, 57/47, 68/64) are
reproduced bitwise by the runner (cond-1, calibration-selected legacy Adam
probe); this bundle records what a coordinate-stable canonical solver
(PCA-whitened features, L2 on whitened weights only, gauge-fixed, L-BFGS
certified with GD as reference) finds at the same taps.

Headline: at every max-drop layer the canonical CTX and ATT probes reach the
same convergence point (identical whitened-space objective and dev token
exact), so the projection drop is an artifact of the legacy Adam pipeline
(C1_OPTIMIZATION_INSUFFICIENCY, 4/4 layers), not of the representation.

L19_SEED_1:L9 CTX_CONCAT legacy=24 canonical=16; L19_SEED_1:L9 ATT_UPDATE legacy=6 canonical=16; L19_SEED_2:L7 CTX_CONCAT legacy=37 canonical=32; L19_SEED_2:L7 ATT_UPDATE legacy=24 canonical=32; L19_SEED_4:L12 CTX_CONCAT legacy=57 canonical=52; L19_SEED_4:L12 ATT_UPDATE legacy=47 canonical=52; L18_SEED_2_CONTROL:L6 CTX_CONCAT legacy=68 canonical=30; L18_SEED_2_CONTROL:L6 ATT_UPDATE legacy=64 canonical=30

Protocol PROBE_OPTIMIZATION_AUDIT_V1 version 6 (AMENDMENT_1..5, fixed before results),
hash fnv1a64:b36b4745b9b4807f. Dataset partitions pinned: TRAIN
fnv1a64:5a64ca2d1aa7f29f, MARGIN_CALIBRATION_V1 fnv1a64:71806d5bf19c090a,
MARGIN_DEVELOPMENT_V1 fnv1a64:f06fcc3e2d12ca99; AR_FINAL_HOLDOUT_V3
fnv1a64:aa5081e6df658b4a remains unopened. All evidence is CPU host-side; no device,
QAIRT, or QNN involvement.

## Current status

Full-rank classifier-coordinate transport and equivalent whitened-space
objectives remain mathematical evidence. Learned-probe absolute scores and
z-statistics below are superseded by the TRAIN row-contract correction and
are not current root-cause evidence.

## Superseding measurement correction (2026-08-05)

Both legacy and canonical probes in this historical bundle were trained on
rows containing four TRAIN-contract conflicts. Their absolute scores and
z-statistics are excluded from later causal claims until corrected-row
regeneration. The algebraic conclusion that a full-rank output projection
admits classifier-coordinate transport, and the equivalent whitened-space
objectives, remain valid; the historical learned scores are not used as
root-cause evidence.