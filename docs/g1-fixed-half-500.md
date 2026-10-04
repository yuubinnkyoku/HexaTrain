# Current G1 vs Fixed 0.5 branch scale (2026-09)

## Status

**COMPLETE.** Decision: **KEEP_CURRENT_G1_LEARNED_GATE**.

## Background

Identity-init G1 (`2*sigmoid`, `Wg=0`, step-0 gate = 1.0) lost the early
sample-efficiency gain versus Current G1 (ΔBalanced +0.10 at step100). That
suggests the gain depends on starting near a 0.5 attention-branch scale.

This A/B asks whether a **static** `Yh = 0.5 * Ah` (no Wg, no learnability)
reproduces Current G1.

## Hypotheses

| | claim | prediction |
|---|---|---|
| H1 | G1 value is mainly initial suppression ≈ 0.5 | Fixed ≈ Current |
| H2 | 0.5 prior **plus** learned adaptation | Current > Fixed |
| H3 | fixed 0.5 is a better regularizer | Fixed > Current |

## Arms (factor isolation)

| | Arm A = Current G1 | Arm B = Fixed 0.5 |
|---|---|---|
| math | `Yh = Ah * sigmoid(N @ Wg)` | `Yh = 0.5 * Ah` |
| Wg | `[64,2] × 19`, AUX_ADAM | **none** |
| params | 760,960 | **758,528** |
| Aux Adam params | 138,368 | 135,936 |
| Muon | 114 matrices / 622,592 | same |
| checkpoint | NPRTCKPTV5 | NPRTCKPTV5 |
| attention_gate id | `headwise_g1_sigmoid` (=1) | `fixed_half` (=3) |

Unchanged: LR, optimizer split, residual, RMSNorm, heads, Wv, batch, dataset.

## Resource delta (Fixed vs Current)

```text
parameters:        -2432
optimizer state:   -4864  (Wg Adam m+v)
Wg matmul/sigmoid/backward: removed
```

## Correctness (host)

```text
fixed_half_parameter_count=758528
fixed_half_forward_scale=0.5          # layer-0 context = 0.5 * ungated
fixed_half_no_wg=true
fixed_half_cross_resume_rejected=true
```

Fixed-half checkpoints reject resume as `none` and as Current G1 (and
reciprocally). Forward-only generation path uses the same architecture id.

## Run

```text
run order: Current → FixedHalf
seed 1, step-0 fresh
LR 1.5x: Muon 0.00750 / Aux Adam 0.00330 / target 0.000150
V1024 / T32 / D64 / FFN128 / L19 / H2, batch 8
eval: Val/Dev first 256 chunks, canonical original UTF-8 byte bpb
steps 500, checkpoints 100/200/300/400/500
device NX741J, QAIRT 2.48.40.260702151143
```

Current G1 parameter hashes match the identity-init A/B / stress-grid 1.5x
runs **exactly** (historical sanity OK).

## Quality (Δ = Fixed − Current; negative favors Fixed)

| step | Current Val | Fixed Val | ΔVal | Current Dev | Fixed Dev | ΔDev | ΔBalanced |
|-----:|------------:|----------:|-----:|------------:|----------:|-----:|----------:|
| 100 | 3.241628 | 3.260542 | +0.018914 | 3.464354 | 3.492914 | +0.028561 | **+0.023737** |
| 200 | 3.047356 | 3.049900 | +0.002544 | 3.253588 | 3.262446 | +0.008858 | **+0.005701** |
| 300 | 2.904970 | 2.904398 | -0.000572 | 3.136970 | 3.141246 | +0.004276 | **+0.001852** |
| 400 | 2.774821 | 2.807532 | +0.032711 | 3.010131 | 3.033251 | +0.023120 | **+0.027916** |
| 500 | 2.735525 | 2.736026 | +0.000502 | 2.988763 | 3.000769 | +0.012006 | **+0.006254** |

Fixed never wins. Gap is modest compared with identity-init (+0.10 early),
so the 0.5 suppression prior explains **most** of the early gain, but learned
adaptation still adds a consistent edge (H2).

## Time-to-bpb (step-based; wall NOT_MEASURED)

| target | Current | Fixed | Δstep |
|-------:|--------:|------:|------:|
| 3.10 | 300 | 300 | 0 |
| 3.00 | 400 | 400 | 0 |
| 2.95 | 400 | 400 | 0 |
| 2.90 | **400** | **500** | **+100** |
| 2.85 | >500 | >500 | — |

## Gate behavior

Current G1 trajectory means ~0.08–0.35 (soft shrink, per-head/layer learned).
Fixed 0.5 is constant: mean = min = max = 0.5, std = 0 (theoretical).

## Runtime

| arm | training wall | ms/update | fwd/bwd ms/update |
|---|---:|---:|---:|
| Current | 185.5 s | 371.1 | 29.1 |
| Fixed | 179.7 s | 359.4 | 27.1 |

Fixed is slightly cheaper (~3%). No performance claim from one run.

## Health

Both arms: QNN success, HVX 0/0/0, fallback 0, finite, thermal 0,
NPRTCKPTV5, checkpoint decode verified.

## Interpretation

> At this horizon/shape/LR, a fixed 0.5 attention-branch scale recovers most
> of Current G1's early advantage over ungated, but **learned per-head/per-layer
> adaptation still adds value** (step500 ΔBalanced +0.006, target 2.90 100
> steps slower without it).

H1 is largely true as a *mechanism* (0.5 prior dominates), but H2 is the
better *decision* model: Current ≥ Fixed at every checkpoint.

This is **not** a claim that learnable gates are permanently unnecessary.

## Decision

**KEEP_CURRENT_G1_LEARNED_GATE**

Close the fixed-0.5 lane as a replacement. Do not run a 0.4/0.5/0.6 sweep.

## Artifacts

- `docs/results/g1-fixed-half-500-2026-09/`
- `scripts/run_g1_fixed_half_ab.ps1`
- `scripts/g1_fixed_half_analyze.py`

## Commits

```text
feat(research): add fixed-half attention branch variant
docs(research): record fixed-half G1 ab result
```

push: NOT PERFORMED.
