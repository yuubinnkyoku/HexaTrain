# L19 output-projection information audit, August 2026

This bundle is a host-only CPU audit that tests whether the Attention output
projection actually discards linear next-token information. The null
hypothesis is that the projection is invertible and the information is still
present; the alternative is rank deficiency or extreme ill-conditioning that
makes the class direction unreachable.

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. At every target
layer the 16x16 Attention output-projection matrix W is decomposed in double
precision. At the previously reported max-drop layer, a 32-way linear softmax
probe is trained on the concatenated head context (CTX_CONCAT), transported
through W using the pseudoinverse with the protocol's fixed tolerance, and
evaluated on the Attention update (ATT_UPDATE). The same ATT_UPDATE probe is
also trained from scratch and warm-started from the transported weights.

Dataset roles follow the pinned protocol: TRAIN = probe learning (32 rows),
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
OUTPUT_PROJECTION_AUDIT_V1): CPU trajectory regenerations <= 4, matrix
decompositions <= 60, full probe transports <= 24, warm-start trainings <= 24.

## Verdict

Diagnosis (fixed thresholds, never tuned):
**OUTPUT_PROJECTION_PRESERVES_INFORMATION**

All audited matrices are full rank and the transported context probe matches the projection probe at step 0 with no argmax flips; the observed previous drop is a learning/standardization/optimization artifact.

Projection-transport parity (DEVELOPMENT, max-drop layer):
L19_SEED_1: ctx=24 trans=24 scratch=6 maxdiff=1.36966153146e-06; L19_SEED_2: ctx=37 trans=37 scratch=24 maxdiff=1.91771038027e-06; L19_SEED_4: ctx=57 trans=57 scratch=47 maxdiff=3.0246865963e-06; L18_SEED_2_CONTROL: ctx=68 trans=68 scratch=64 maxdiff=8.22284617064e-07

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (OUTPUT_PROJECTION_AUDIT_V1) before
any results were produced.

## Files

- configuration.csv / dataset-usage.csv - configs and dataset role hashes
- projection-matrix-summary.csv / singular-value-summary.csv - matrix decomposition
- rank-and-conditioning.csv - rank and condition number summary
- probe-transport-summary.csv / probe-transport-by-seed.csv - transport parity
- float-double-comparison.csv - float vs double transport accuracy
- nullspace-summary.csv - null-space fraction of the context probe
- from-scratch-vs-transport.csv - optimization comparison
- depth-control.csv - L18 depth control comparison
- diagnosis.csv - formal conclusion
- previous-result-correction.csv - how to rephrase the previous report
- next-step-candidates.csv - candidate follow-ups
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest