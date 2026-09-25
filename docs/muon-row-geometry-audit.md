# Muon Row-Geometry Audit (Phase 2)

**Status:** diagnostic complete on existing HVX Muon quality checkpoints  
**Date:** 2026-09-22  
**Tool:** [`scripts/muon_row_geometry_audit.py`](../scripts/muon_row_geometry_audit.py)  
**Evidence:** `build/muon-geometry-audit/seed1-250-8000/`, `build/muon-geometry-audit/seed2-key-steps/`

## Question

Does the current Original-Muon 8,000-step trajectory show the row-geometry drift
that Muown is designed to control (`max row norm ↑` synchronized with
`spectral norm ↑`)?

This is the Phase-2 gate from [`research-priorities.md`](research-priorities.md).
If drift is weak, Muown priority drops. If drift is strong, keep or raise it.

## Method

Existing quality checkpoints only. No new training, no device work.

- Checkpoints: seed1 steps 250/500/750/1000/1500/2000/3000/4000/5000/6000/7000/8000,
  seed2 steps 250/500/1000/2000/4000/6000/8000
- Format: NPRTCKPTV4, 114 Muon matrices, 622,592 Muon parameters
- Semantic row direction follows SSOT `fan_out_axis`
  (`metadata/transformer_parameter_metadata.json`): storage is `[input, output]`,
  so each output-neuron vector is a **column** of the stored matrix. W1/W2 use
  output-neuron direction, not raw storage rows.
- Metrics per matrix / step: max / median / RMS semantic row norm, spectral norm
  (largest singular value), effective rank (participation ratio), sampled row
  coherence, mean/max angular update vs previous step.

Per [`docs/agent/numerical-evidence.md`](agent/numerical-evidence.md), this is
diagnostic geometry on trained artifacts. Two seeds agree, but this is not a
reproduction of the full quality baseline.

## Results

### Drift is strong and synchronized

| Metric | Seed1 (250→8000) | Seed2 (250→8000) |
| --- | ---: | ---: |
| `corr(max row norm, spectral norm)` mean | 0.9927 | 0.9923 |
| same correlation median | 0.9949 | 0.9955 |
| max-row-norm ratio (last/first) median | 2.462 | 2.428 |
| spectral-norm ratio (last/first) median | 2.694 | 2.658 |
| verdict hint | KEEP_OR_RAISE | KEEP_OR_RAISE |

Mean trajectory (seed1, all 114 matrices):

| step | max row norm (mean) | spectral norm (mean) | angular update mean (rad) |
| ---: | ---: | ---: | ---: |
| 250 | 0.770 | 1.261 | — |
| 1000 | 1.027 | 1.804 | 0.374 |
| 2000 | 1.296 | 2.272 | 0.442 |
| 4000 | 1.733 | 2.995 | 0.455 |
| 6000 | 1.966 | 3.382 | 0.252 |
| 8000 | 2.020 | 3.456 | 0.074 |

### Interpretation

1. **`max row norm ↑` and `spectral norm ↑` track each other** (corr ≈ 0.99
   in both seeds). This is the phenomenon Muown's row-norm control targets.
2. Growth is substantial, not a rounding artifact: roughly **2.4–2.7×** from
   step 250 to 8000 on both seeds.
3. **Mean per-row angular update collapses late** (≈0.45 rad at 2k–4k →
   ≈0.07 rad at 8k). That is consistent with the implicit angular step-size
   decay analysis behind Muown/AngularMuown: larger row norms make the same
   additive update rotate rows less.
4. Effective rank stays near 29–35 and coherence is moderate (≈0.38–0.45 mean),
   so the drift is magnitude geometry, not an obvious rank collapse.

## Verdict

Phase-2 gate is **positive for Muown**.

- Do **not** lower Muown priority.
- Proceed to Phase 4 (Muown CPU reference + short A/B) when capacity allows.
- Keep Original Muon as the frozen quality baseline; Muown is a new optimizer
  identity / checkpoint schema (fail-closed), not an in-place rewrite.
- AngularMuown (P2) remains a follow-up; the late angular-update collapse is
  supporting evidence, not a replacement for the row-norm control question.

## Artifacts

- `muon-row-geometry-detail.csv` — per matrix / step metrics
- `muon-row-geometry-summary.csv` — per seed / step aggregates
- `muon-row-geometry-drift.csv` — per matrix ratio / correlation
- `muon-row-geometry-verdict.json` — machine-readable gate result

## Limitations

- Step 0 (initialization) was not present in the quality checkpoint set; the
  trajectory starts at step 250. Ratios are therefore 250→8000, not 0→8000.
- Coherence uses a deterministic sample of at most 32 semantic rows per matrix.
- No CPU-double oracle comparison in this audit; it describes trained HVX
  checkpoints only.
