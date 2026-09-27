# G1 identity-init 500-step A/B (2026-09)

## Status

**COMPLETE.** Device A/B executed 2026-09-27 on NX741J (HTP / HVX Muon).
Decision: **KEEP_CURRENT_G1**.

## Hypothesis

Current G1 early optimization gain may come from either

1. initializing the attention branch near a 0.5 gate (soft shrink), or
2. the learnable head-wise gate itself.

This A/B isolates those factors.

## Arms (factor isolation)

Only two factors differ. Everything else is identical.

| | Arm A = Current G1 | Arm B = Identity-init G1 |
|---|---|---|
| gate math | `G = sigmoid(N @ Wg)` | `G = 2 * sigmoid(N @ Wg)` |
| Wg init | current deterministic linear | **zero** |
| step-0 G | ~0.5 neighborhood | **exactly 1** (`Yh = Ah`) |
| Wg shape | `[64,2] × 19` | same |
| params | 760,960 | 760,960 |
| optimizer | Wg → AUX_ADAM, 114 matrices → Original Muon | same |
| checkpoint | NPRTCKPTV5 | NPRTCKPTV5 |
| attention_gate id | `headwise_g1_sigmoid` (=1) | `headwise_g1_scale2_identity` (=2) |

## Math

```text
G = scale * sigmoid(z),  scale = 1 (current) or 2 (identity)
dG/dz = G * (1 - G/scale)
```

Wg LR was **not** halved. Same optimizer/LR for both arms.

## Step-0 identity proof (host)

`host_tests/headwise_g1_gate_test.cpp` PASS:

```text
identity_init_gate_mean/min/max = 1
identity_init_wg_zero = true
identity_init_dwg_nonzero = true
identity_checkpoint_cross_resume_rejected = true
ungated context parity at Wg=0 / scale=2
```

## Run recipe

```text
seed 1, step-0 fresh
LR 1.5x: Muon 0.00750 / Aux Adam 0.00330 / target 0.000150
linear_decay 4000→8000 (500-step region is pre-decay, constant peak)
batch 8, V1024 / T32 / D64 / FFN128 / L19 / H2
eval: Val/Dev first 256 chunks, canonical original UTF-8 byte bpb
steps 500, checkpoints 100/200/300/400/500
QAIRT 2.48.40.260702151143
device NX741J, HTP forward/backward, HVX Muon, CPU Aux Adam
```

## Quality (Balanced bpb; Δ = Identity − Current)

Historical stress-grid Current G1 1.5x is reproduced **exactly**.

| step | Current Val | Identity Val | ΔVal | Current Dev | Identity Dev | ΔDev | ΔBalanced |
|-----:|------------:|-------------:|-----:|------------:|-------------:|-----:|----------:|
| 100 | 3.241628 | 3.340035 | +0.098407 | 3.464354 | 3.566327 | +0.101974 | **+0.100190** |
| 200 | 3.047356 | 3.084760 | +0.037405 | 3.253588 | 3.285475 | +0.031887 | **+0.034646** |
| 300 | 2.904970 | 2.921797 | +0.016827 | 3.136970 | 3.158149 | +0.021179 | **+0.019003** |
| 400 | 2.774821 | 2.820489 | +0.045668 | 3.010131 | 3.039870 | +0.029739 | **+0.037704** |
| 500 | 2.735525 | 2.749066 | +0.013542 | 2.988763 | 2.985007 | -0.003756 | **+0.004893** |

Negative Δ would favor Identity. All ΔBalanced are **positive**.

## Time-to-bpb (step-based; wall NOT_MEASURED)

| target | Current first step | Identity first step | Δstep |
|-------:|-------------------:|--------------------:|------:|
| 3.10 | 300 | 300 | 0 |
| 3.00 | 400 | 400 | 0 |
| 2.95 | 400 | 400 | 0 |
| 2.90 | **400** | **500** | **+100** |
| 2.85 | >500 | >500 | — |

Identity is **slower** to 2.90 by 100 steps.

## Gate trajectory (training aggregate)

Current G1 stays in `(0,1)` with means ~0.08–0.35 (soft shrink throughout).
Identity-init starts at 1.0 and moves: layer-0 means ~0.54–0.57, max observed
~1.81, so it does **not** stay at identity. Deep-layer means settle ~0.14–0.44
(below 1). Scale semantics: Identity gates live in `(0,2)`; legacy
`g>0.9` fractions are not comparable to Current G1.

## Runtime / health

| arm | training wall | ms/update |
|---|---:|---:|
| Current | 213.6 s | 427.2 |
| Identity | 208.3 s | 416.6 |

Difference is small; no performance claim. Both arms: QNN success, HVX
0/0/0, no fallback, all finite, thermal ≤0 after, NPRTCKPTV5.

Note: checkpoint-static gate diagnostics failed for Identity (host diag tool
emitted `headwise_g1_gate_diagnostics=FAIL`); trajectory aggregates above come
from the device training report and are sufficient for interpretation.

## Interpretation

Identity-init **loses the early sample-efficiency gain** that Current G1
showed against ungated control (ΔBalanced at step100 is +0.10 worse). By
step500 the two are nearly tied (ΔBalanced +0.005), with Identity slightly
worse and slower to target 2.90.

This implies the Current G1 early gain depends on **starting near a 0.5 gate**
(attention-branch suppression prior), not merely on having a learnable
head-wise gate. The `2*sigmoid` / `Wg=0` identity lane does not reproduce the
early benefit.

## Decision

**KEEP_CURRENT_G1**

Close the `2*sigmoid` / `Wg=0` identity-init lane. Do **not** promote
Identity-init to a broader LR grid. reduced-channel gate remains unimplemented
and is out of scope for this task.

## Artifacts

- `docs/results/g1-identity-init-500-2026-09/`
- `scripts/run_g1_identity_ab.ps1`
- `scripts/g1_identity_ab_analyze.py`

## Commits

```text
feat(research): add identity-init headwise G1 variant
docs(research): record identity-init G1 500-step A/B
```

push: NOT PERFORMED.
