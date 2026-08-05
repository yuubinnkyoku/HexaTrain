# L19 attention-internal diagnosis, August 2026

This bundle is a host-only CPU diagnosis that decomposes the deep-layer
attention-path linear readability loss of the L19 model into
normalized-input, Q/K/V projections, attention scores/weights, per-head
context, concat, output projection, and residual add. It does not open the
AR_FINAL_HOLDOUT_V3 dataset (hash verified only: fnv1a64:aa5081e6df658b4a) and performs no
device, HTP, or QNN work. All numbers come from the checked-in CPU reference
implementation (tiny_language_model_cpu.cpp), regenerated deterministically.

## Current status: historical verdict superseded

The OUTPUT_PROJECTION verdict, legacy learned-probe scores, cross-seed swap,
and projection-contribution norm/cosine below are retained as historical
artifacts and are not current cause evidence. The seed-instability root-cause
investigation is the current decision source.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. An attention
observer extracts, per target layer, the LN1 output, Q/K/V projections,
per-head contexts, concatenated context, attention update and post-attention
residual. A 32-way linear softmax probe (Adam lr=0.01, 2000 steps,
calibration step selection) is trained per tap on TRAIN rows only. Head-level
counterfactual interventions (head zero, head only, cross-seed context swap,
attention-weight/value separation, head pair) are evaluated on
MARGIN_DEVELOPMENT_V1 rows. Cross-seed context swaps are explicit
counterfactual interventions between models, not natural inferences.

Dataset roles follow the pinned protocol: TRAIN = probe learning (32 rows),
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
ATTENTION_INTERNAL_V1): attention taps <= 500, head probe trainings <= 400,
head-zero <= 300, head-only <= 150, context swaps <= 60, attention/value
separation <= 32, head pairs <= 24, free-running <= 40.

## Historical verdict (superseded)

Diagnosis (fixed thresholds, never tuned):
**OUTPUT_PROJECTION**

S1 proj_drop=18 S2 proj_drop=13 S4 proj_drop=10 control_proj_drop=4; S2 head-zero deltas at max-drop: 1 1; S4 head-zero deltas at max-drop: -1 0; S2 AV recovery (D): 8 1; S4 AV recovery (D): 0 -1; S2 swap recovery: 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0; S4 swap recovery: 0 -1 -1 0 -6 -1 0 0 1 0 0 0 0 0 0 0

Context vs output-projection (dev TF exact, max-drop layer):
L19_SEED_1: ctx=24 upd=6 drop=18; L19_SEED_2: ctx=37 upd=24 drop=13; L19_SEED_4: ctx=57 upd=47 drop=10; L18_SEED_2_CONTROL: ctx=68 upd=64 drop=4

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (ATTENTION_INTERNAL_V1) before any
results were produced.

## Superseding measurement correction (2026-08-05)

The historical TRAIN probe rows contain four contract conflicts, so learned
probe absolute scores are excluded from later causal claims. In addition,
the published cross-seed context swap reused a malformed donor-row-0 vector
for every development row, and projection-contribution norm/cosine retained
only one row. Those swap and aggregate claims are invalid and were not used
by the root-cause decision. Full-rank, pseudoinverse/transport evidence and
direct logit interventions are independent. The implementation now has
row-wise replacement and self-swap identity tests.

## Files

- dataset-anchors.csv / trajectory-anchors.csv - dataset and trajectory anchors
- head-probe-by-seed.csv - per-tap probe results per seed
- attention-statistics.csv - entropy/max-weight/self/prev/cosine statistics
- head-ablation.csv - head-zero ablation results
- head-only.csv - head-only (single head) results
- context-vs-projection.csv - context vs output-projection readability
- projection-contributions.csv - per-head output-projection contributions
- attention-vs-value-swap.csv - attention-weight vs value separation
- head-pair-interactions.csv - head-pair interventions
- teacher-forced-free-running.csv - teacher-forced vs free-running
- depth-control.csv - L18 depth control comparison
- diagnosis.csv - attention-internal cause classification
- next-step-candidates.csv - candidate next steps
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest