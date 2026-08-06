# L19 Attention minimal-cause investigation, August 2026

> **Follow-up (2026-08):** A later data intervention preserves ordinary
> learned Attention and stabilizes all target L19 seeds with 25% target-invariant
> mixed-prefix supervision. The matched homogeneous control fails, and V/O
> freezing is unnecessary. See
> `../qnn-l19-context-supervision-stability-2026-08/README.md`.

This host-only bundle decomposes the earlier whole-Attention-zero result.
The strongest supported mechanism is a combination: broad mixing of irrelevant
distractor tokens is harmful during training, and learned V/O plus the rest of
the model co-adapt to that path in a seed-dependent way. Learned content-dependent
Q/K selection is not necessary.

Training with fixed self-only Attention reaches 144/144 tokens and 24/24 exact
sequences for L19 seeds 1/2/4 and the L18 seed-2 control. Fixed previous-token
Attention is nearly stable, whereas fixed causal-uniform Attention fails strongly.
Freezing Q/K at initialization also fails; freezing V/O improves every L19 seed
and completely recovers L18, but does not completely recover L19 seed 2. The
claim strength is therefore `major factor`, not a fully factorized initialization
root cause.

Evaluation-only removal does not recover the trained models, so the final
Attention forward value alone is insufficient. The harmful state is formed in
the training trajectory. All interventions are diagnostic counterfactuals, not
production recommendations.

Historical `fnv1aParams` values hash only the zero/nonzero support mask, not float
contents. The quality aggregates are unchanged, but those historical values must
not be used as exact checkpoint identities. New private identities hash registry
names, shapes, and float bytes.

No device, HTP, QNN, QAIRT, ADB, Android/JNI, UI, or final-holdout evaluation was
performed. Raw parameters, optimizer states, checkpoints, gradients, hidden
states, logits, Attention matrices, local paths, and private identifiers are not
published.

The CSV files contain configuration, evidence, hypotheses, preregistered decisions,
intervention protocols, evaluation-only results, training results, controls,
seed/depth comparisons, causal claims, diagnosis, and remaining uncertainties.
`manifest.json` binds the allow-list, tracked sources, and private aggregate hashes.

```powershell
.\scripts\run_l19_attention_minimal_cause.ps1 -SelfTest
.\scripts\export_public_qnn_l19_attention_minimal_cause.ps1 -SelfTest
```
