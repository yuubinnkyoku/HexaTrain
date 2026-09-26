# G1 high-LR stress grid (2026-09)

## Purpose

Determine whether current `headwise_g1_sigmoid` (G1) can use a higher learning
rate than the ungated control more stably, and whether that reduces steps /
original UTF-8 bytes / training wall time to fixed canonical bpb targets.

This is **not** a final-quality claim at the baseline LR. Prior seed-1 8000-step
evidence already showed early sample-efficiency gains that faded to a tie by
step 8000.

## Prior G1 evidence

| step | ΔBalanced |
|-----:|----------:|
| 500 | -0.032712 |
| 1000 | -0.036686 |
| 1500 | -0.021864 |
| 2000 | -0.007113 |
| 2500 | -0.016147 |
| 3000 | -0.017483 |
| 3500 | -0.007219 |
| 4000 | -0.007490 |
| 5000 | +0.004985 |
| 6000 | +0.008981 |
| 7000 | +0.009498 |
| 8000 | -0.000367 |

Interpretation: strong early sample-efficiency; 8000-step final quality is a
tie with control at the baseline LR.

## Experiment identity

- parent: `origin/main` = `b251a5a6a07239a5fe429b69404a66fd7ad83c0e`
- branch: `experiment/g1-lr-stress`
- model: V1024 / T32 / D64 / FFN128 / L19 / H2
- seed: 1 (fresh step-0 starts; no checkpoint continuation)
- batch: 8
- optimizer: Original Muon + Aux Adam, momentum 0.95, Nesterov, NS5
- Muon backend: HVX_W8
- Control: `attention_gate=none`, 758,528 params, NPRTCKPTV4
- G1: `attention_gate=headwise_g1_sigmoid`, 760,960 params, NPRTCKPTV5
- Wg: shape [64,2], no bias, layer-specific, AUX_ADAM
- dataset: `fnv1a64:0c7b2826f5f26fea`, order seed 20260806
- tokenizer: Byte-BPE V=1024 (canonical)
- eval: Val first 256 + Dev first 256 chunks, `bits_per_utf8_byte`
- QAIRT: `2.48.40.260702` / Build ID `2.48.40.260702151143`
- device: NX741J (physical), HTP forward/backward, HVX Muon, CPU Aux Adam

## LR scaling policy

Muon and Aux Adam scaled by the same multiplier; ratio preserved. Schedule is
still `linear_decay` (4000→8000) but 500-step stress runs are entirely before
decay start, so effective LR is constant at peak.

| multiplier | Muon LR | Aux Adam LR | target LR |
|-----------:|--------:|------------:|----------:|
| 1.00x | 0.00500 | 0.00220 | 0.000100 |
| 1.25x | 0.00625 | 0.00275 | 0.000125 |
| 1.50x | 0.00750 | 0.00330 | 0.000150 |
| 2.00x | 0.01000 | 0.00440 | 0.000200 |

Runtime float32 values are recorded in `resolved-grid.csv` /
`shared-parameter-hash.csv`.

## Run order

```
1.00x: Control → G1
1.25x: G1 → Control
1.50x: Control → G1
2.00x: G1 → Control
```

Matched pairs held the shared device lock. Device was kept awake
(`svc power stayon true`, `deviceidle disable`) after an early Doze hang.

## Canonical quality (Balanced bpb)

Historical 1.0x sanity (step500): Control `2.896905`, G1 `2.864193`. Fresh
1.0x rerun matches exactly — identity confirmed.

### Quality table (Val / Dev / ΔBalanced)

| LR | step | Ctrl Val | G1 Val | ΔVal | Ctrl Dev | G1 Dev | ΔDev | ΔBalanced |
|---:|-----:|---------:|-------:|-----:|---------:|-------:|-----:|----------:|
| 1.00 | 100 | 3.405158 | 3.305306 | -0.099852 | 3.695693 | 3.548288 | -0.147405 | **-0.123629** |
| 1.00 | 200 | 3.130764 | 3.075969 | -0.054795 | 3.364080 | 3.278300 | -0.085780 | **-0.070287** |
| 1.00 | 300 | 2.946549 | 2.920335 | -0.026214 | 3.193773 | 3.146552 | -0.047221 | **-0.036717** |
| 1.00 | 400 | 2.845914 | 2.805898 | -0.040015 | 3.089022 | 3.024020 | -0.065002 | **-0.052509** |
| 1.00 | 500 | 2.765700 | 2.738926 | -0.026774 | 3.028110 | 2.989461 | -0.038650 | **-0.032712** |
| 1.25 | 100 | 3.388840 | 3.264141 | -0.124699 | 3.662327 | 3.488639 | -0.173687 | **-0.149193** |
| 1.25 | 200 | 3.109603 | 3.044380 | -0.065224 | 3.320994 | 3.265895 | -0.055099 | **-0.060161** |
| 1.25 | 300 | 2.935867 | 2.910031 | -0.025836 | 3.195922 | 3.141590 | -0.054333 | **-0.040084** |
| 1.25 | 400 | 2.832648 | 2.803685 | -0.028963 | 3.077194 | 3.049824 | -0.027370 | **-0.028167** |
| 1.25 | 500 | 2.768474 | 2.740344 | -0.028130 | 3.007113 | 3.004141 | -0.002972 | **-0.015551** |
| 1.50 | 100 | 3.369602 | 3.241628 | -0.127974 | 3.617385 | 3.464354 | -0.153031 | **-0.140503** |
| 1.50 | 200 | 3.104290 | 3.047356 | -0.056934 | 3.312446 | 3.253588 | -0.058858 | **-0.057896** |
| 1.50 | 300 | 2.936161 | 2.904970 | -0.031191 | 3.180286 | 3.136970 | -0.043316 | **-0.037254** |
| 1.50 | 400 | 2.840513 | 2.774821 | -0.065693 | 3.070719 | 3.010131 | -0.060589 | **-0.063141** |
| 1.50 | 500 | 2.771933 | 2.735525 | -0.036408 | 3.012327 | 2.988763 | -0.023564 | **-0.029986** |
| 2.00 | 100 | 3.339596 | 3.230172 | -0.109424 | 3.572595 | 3.439795 | -0.132800 | **-0.121112** |
| 2.00 | 200 | 3.094954 | 3.057651 | -0.037303 | 3.293997 | 3.266443 | -0.027555 | **-0.032429** |
| 2.00 | 300 | 2.951633 | 2.896169 | -0.055464 | 3.212540 | 3.131205 | -0.081335 | **-0.068400** |
| 2.00 | 400 | 2.820983 | 2.800765 | -0.020218 | 3.063052 | 3.035132 | -0.027920 | **-0.024069** |
| 2.00 | 500 | 2.770158 | 2.746389 | -0.023769 | 3.021261 | 2.990902 | -0.030359 | **-0.027064** |

G1 wins every matched cell. Early-step advantage is largest; 500-step
advantage remains ~0.016–0.033 ΔBalanced.

## Stability

| LR | Control | G1 |
|---:|:-------:|:--:|
| 1.00 | STABLE | STABLE |
| 1.25 | STABLE | STABLE |
| 1.50 | STABLE | STABLE |
| 2.00 | STABLE | STABLE |

No arm hit a hard stop (non-finite / fallback / QNN-HVX fatal / loss explosion).
Both max stable multipliers = **2.0x** (500-step stress region only).

Soft flags: G1 gate mean falls with LR (see Gate behavior). Control 2.0x wall
time was elevated (~904 s vs ~770 s); thermal status stayed ≤2.

## Time-to-bpb (primary = first observed eval checkpoint at or below target)

Targets fixed a priori: 3.10 / 3.00 / 2.95 / 2.90 / 2.85.

| target | Ctrl best LR | Ctrl step | Ctrl wall s | G1 best LR | G1 step | G1 wall s | Δstep | Δwall s |
|-------:|-------------:|----------:|------------:|-----------:|--------:|----------:|------:|--------:|
| 3.10 | 1.50 | 300 | 770.2 | 2.00 | 300 | 791.5 | 0 | +21.3 |
| 3.00 | 2.00 | 400 | 903.9 | 1.50 | 400 | 792.5 | 0 | **-111.5** |
| 2.95 | 2.00 | 400 | 903.9 | 1.50 | 400 | 792.5 | 0 | **-111.5** |
| 2.90 | 1.25 | 500 | 776.2 | 1.50 | 400 | 792.5 | **-100** | +16.2 |
| 2.85 | — | >500 | — | — | >500 | — | — | — |

At matched 1.5x, G1 hits 2.90 at step 400 (Balanced 2.892) while control needs
step 500 (2.892) — **20% fewer steps**.

Original UTF-8 bytes: Muon-hybrid reports in this harness do not emit
`target_utf8_bytes_seen`; bytes are identical across arms at equal step under
the shared DataCursor. Estimated 500-step exposure ≈ 343k original bytes
(proportional to the 8000-step canonical 5,491,256). Marked estimated in CSV.

## Gate behavior (checkpoint-static, Val 256 windows)

| LR | mean of head means | min head mean | max head mean |
|---:|-------------------:|--------------:|--------------:|
| 1.00 | 0.154 | 0.029 | 0.529 |
| 1.25 | (see gate-static.csv) | | |
| 1.50 | (see gate-static.csv) | | |
| 2.00 | 0.078 | 0.016 | 0.300 |

At 2.0x several deep-layer heads show `g<0.1` fractions near 0.8–1.0 (gate
collapse toward closed). No arm was hard-stopped on gate saturation alone.
Wg weight norm rises slightly with LR (mean 1.03→1.12, max 1.51→1.83).

## Muon geometry

`muon_row_geometry_audit.py` on 1.5x G1 step 100→500:
`corr(max row norm, spectral norm) ≈ 0.98`, max-row-norm ratio ≈ 1.42,
spectral-norm ratio ≈ 1.44 over 500 steps. Consistent with known early drift;
no new pathology unique to high LR.

## Runtime caveat

Training wall is the primary time metric (excludes eval). G1 overhead is ~3%
ms/update vs control (extra Wg projection). Control 2.0x wall was inflated
(~1808 ms/update vs ~1540); do not treat that single arm as a G1 speedup.

ReLU²-adjacent cross-phase runtime anomalies were not observed; E0 backlog
item unchanged.

## Health

- QNN return success: 8/8
- HVX failures / fallback / non-finite: 0 / 0 / 0
- thermal status after: ≤2 all arms
- focus takeover: 0
- checkpoint decode: all interval checkpoints verified
- historical 1.0x sanity: exact match

## Decision

**PROMOTE_G1_GATE_VARIANTS**

Reason:
1. Stability region was **not** expanded (both max stable = 2.0x).
2. But G1 is consistently better on canonical Balanced bpb at every matched
   LR and every eval step.
3. Time-to-bpb improves at fixed targets (notably 2.90: 400 vs 500 steps at
   1.5x; 2.95/3.00: ~12% training-wall reduction on best-LR comparison).
4. 500-step quality is better, not worse.

Correct combined claim with historical 8000 evidence:

> G1 does not establish a final-quality gain at the baseline LR on seed1 at
> step 8000, but it improves short-horizon canonical bpb at matched LR and
> reduces time-to-bpb under the tested LR grid. Observed 500-step stability
> region is the same as control (2.0x). This promotes gate-variant exploration
> (`2*sigmoid` / `Wg=0` identity-init), which are **not** implemented tonight.

## Next gate (proposed only — not executed)

- A/B #1: current sigmoid G1 vs `2*sigmoid` + `Wg=0` identity-init
- A/B #2: identity-init positive only → reduced-channel gate

## Artifacts

- `docs/results/g1-lr-stress-2026-09/` — CSVs / manifests (this tree)
- `build/g1-lr-stress/` — raw checkpoints, evals, telemetry (not committed)
- runner: `scripts/run_headwise_g1_lr_stress.ps1`
- analyzer: `scripts/g1_lr_stress_analyze.py`

## Limitations

- seed 1 only (candidate selection / stability diagnostic)
- 500-step stress, not long-horizon stability
- original-byte time-to-target uses step-proportional estimate (no per-step
  byte counter in Muon-hybrid report)
- single device night; Control 2.0x wall outlier
