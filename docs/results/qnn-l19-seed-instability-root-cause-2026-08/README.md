# L19 seed-instability root-cause investigation, August 2026

> **Follow-up (2026-08):** Ordinary learned Attention becomes stable across
> L19 seeds 1/2/4 when 80 of 320 batches use target-invariant mixed prefixes;
> a histogram-matched homogeneous control does not. See
> `../qnn-l19-context-supervision-stability-2026-08/README.md`.

> **Correction / follow-up (2026-08):** A later minimal-mechanism audit found
> that broad distractor-token mixing plus V/O/rest co-adaptation, not learned Q/K
> selection by itself, is the smaller causal mechanism. See
> `../qnn-l19-attention-minimal-cause-2026-08/README.md`. Historical
> `param_content_hash` fields were produced by `fnv1aParams`, which hashes only
> zero/nonzero support rather than float contents. Quality aggregates are
> unchanged, but those fields are not exact checkpoint-content identities.

This host-only investigation re-audits the finite seed-dependent generation
quality of T8/D16/FFN32/L19/H2 without opening AR_FINAL_HOLDOUT_V3. The causal
result is a compound root cause: the target rule is a deterministic
current-token successor, training repeats only homogeneous phase-0 contexts,
and, under that canonical homogeneous-only training distribution, the deep
Attention path causally mediates the seed-dependent failures on mixed distractor
prefixes. This is not an unconditional necessity or sufficiency claim about
Attention across training distributions. Argmax feedback then amplifies local
ranking differences into sequence-exact differences. A learned context
shortcut is the leading mechanism candidate, but branch removal also changes
gradient competition and normalization, so that finer mechanism is not proven.

The key intervention is diagnostic, not a production recommendation. With all
Attention residual branches structurally zero throughout the same 320-step CPU
training recipe, L19 seeds 1/2/4 and the L18 seed-2 scope control all reach
144/144 mixed-development free-running tokens and 24/24 exact sequences. The
same-size FFN-zero negative control worsens every configuration (23-39/144).
Late output-head freeze/moment reset does not rescue the L19 seeds consistently,
so output-head ranking drift is a seed-specific downstream compensator rather
than the common root cause.

All baseline and intervention scores use AR_DEVELOPMENT_V3. The MARGIN
calibration/development partitions are audited separately for context
uniqueness and are not mixed with these AR scores. Branch-run finite flags
cover training loss and scored logits/probabilities; they are not an exhaustive
hidden-state, gradient, or optimizer-moment tensor audit.

The evaluator audit found no target ambiguity: current-token and previous-two-
token mappings are unique in TRAIN, AR_VALIDATION_V3, AR_DEVELOPMENT_V3,
MARGIN_CALIBRATION_V1, and MARGIN_DEVELOPMENT_V1.
It also found and corrected diagnostic foundations for future use: historical
probe TRAIN rows included four contract-conflicting synthetic rows (28/32
instead of 32/32), historical cross-seed context swaps were malformed and
lacked a true row-wise identity, and contribution norm/cosine aggregates used
one row. Those historical swap/aggregate claims are excluded here. Direct head
AR metrics, canonical trajectory anchors, output-projection rank/transport,
and the new branch intervention do not depend on those defective measurements.

No device, HTP, QAIRT, ADB, UI, or final-holdout execution was performed. Raw
checkpoints, parameters, optimizer states, hidden states, logits, attention
matrices, gradients, local paths, and private identifiers are not published.
