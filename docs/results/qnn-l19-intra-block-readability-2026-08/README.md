# L19 intra-block readability diagnosis, August 2026

This bundle is a host-only CPU diagnosis that decomposes the deep readout
degradation of the L19 model into intra-block causes. It does not open the
AR_FINAL_HOLDOUT_V3 dataset (hash verified only: fnv1a64:aa5081e6df658b4a) and performs no
device, HTP, or QNN work. All numbers come from the checked-in CPU reference
implementation (tiny_language_model_cpu.cpp), regenerated deterministically.

## Current status: learned-probe localization superseded

The TRAIN row-contract correction excludes learned-probe absolute tap scores
and block/drop counts below from current cause evidence. Observer output,
head-clone parity, and direct forward interventions are independent. The
seed-instability root-cause investigation is the current decision source.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. An intra-block
observer extracts 126 taps on L19 (120 on L18): block input (residual), LN1
input/output, attention context/update, post-attention, LN2 input/output, FFN
update, ReLU, post-FFN. A 32-way linear softmax probe (Adam lr=0.01, 2000
steps, calibration step selection) is trained per tap on TRAIN rows only.
Per-block geometry (update/residual ratios and cosines) and three analyses of
readability transfer between adjacent taps are computed on
MARGIN_DEVELOPMENT_V1 rows only: (1) direct probe transfer, (2) function-level
least-squares alignment and Procrustes alignment with recovery R =
clamp((a-n)/(i-n), 0, 1), and (3) per-op dev TF exact drops. Free-running
rollouts (12) separate gold-prefix from closed-loop degradation. The head is
cloned as a linear probe on the final block output (head-clone parity,
|logit|<=1e-4 tolerance) so probe and head readouts are directly comparable.

Dataset roles follow the pinned protocol: TRAIN = probe/alignment learning
(32 rows), MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1
= final evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
INTRA_BLOCK_READABILITY_V1): 498 independent probe trainings, 444 cross-tap
transfer evals, 150 alignment fits, 12 free-running rollouts, 4 trajectory
regenerations, 3 token baselines; actual structural counts are within all
limits (see budget.csv).

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

Diagnosis (fixed thresholds, never tuned):
**ATTENTION**

L19 deep-band CT/16 per seed: 1/1/0; IL/16: 9/11/13; attn_drop_blocks/8: 0/6/6; ffn_drop_blocks/8: 0/4/4; norm1_drop_blocks/8: 5/0/0; norm2_drop_blocks/8: 5/0/0; attn_overwrite_blocks/8: 0/0/0; ffn_overwrite_blocks/8: 0/0/0; control_il=7 (L18 deep-band IL below majority (guard OK))

Head-clone parity on the final block output (all configs): L19_SEED_1: max_logit_delta 4.75475705031e-07, flips 0, exact_mismatch 0, pass true; L19_SEED_2: max_logit_delta 9.07812527373e-07, flips 0, exact_mismatch 0, pass true; L19_SEED_4: max_logit_delta 5.16857255661e-07, flips 0, exact_mismatch 0, pass true; L18_SEED_2_CONTROL: max_logit_delta 7.46395851792e-07, flips 0, exact_mismatch 0, pass true

Max per-block dev-TF drop (block_input probe vs after_ffn probe): L19_SEED_1=block 2, L19_SEED_2=block 1, L19_SEED_4=block 0, L18_SEED_2_CONTROL=block 2

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (INTRA_BLOCK_READABILITY_V1) before
any results were produced.

## Superseding measurement correction (2026-08-05)

The TRAIN probe row builder used by this historical bundle created four
contract-conflicting synthetic rows (28/32 current-token exact rather than
the formal batch's 32/32). Learned-probe absolute tap scores and drop counts
are excluded from subsequent causal claims until regenerated with the
corrected contract. Observer outputs, head-clone parity, and direct forward
interventions are independent of that row builder. Attention-path causality
is instead established by the later Attention-zero versus FFN-zero training
intervention.

## Files

- dataset-anchors.csv - dataset roles and hash pins
- trajectory-anchors.csv - regenerated trajectory vs pinned anchors
- tap-probes.csv - per-tap probe training results and selected steps
- tap-transfers.csv - cross-tap probe transfer (raw/norm, coarse/fine)
- tap-alignments.csv - LS/Procrustes alignment recovery and pair verdicts
- tap-geometry.csv - residual/update norms, ratios, cosines, overwrite flags
- tap-aux.csv - per-tap eta2, effective rank, between/within, cosine
- clone-parity.csv - head-as-probe clone parity on the final block output
- token-baselines.csv - token-only baselines A/B/C (no model)
- tap-free-running.csv - free-running rollouts on selected taps
- diagnosis.csv - intra-block cause classification
- summary.csv - head/probe/max-drop summary rows per configuration
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest