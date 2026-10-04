# G1 high-LR stress grid (2026-09)

## Purpose

Determine whether current `headwise_g1_sigmoid` (G1) can use a higher learning
rate than the ungated control more stably, and whether that reduces optimizer
steps (and, when measurable, training wall / estimated original UTF-8 bytes)
to fixed canonical bpb targets.

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

## Split-level audit of the same 20 cells (recomputed from primary reports)

`quality-paired.csv` only carries bpb. Recomputing NLL, top-k, and mean rank
from the 40 primary eval reports (full table in
[results/g1-lr-stress-2026-09/README.md](results/g1-lr-stress-2026-09/README.md))
gives a more precise claim than "G1 wins every cell":

| metric | Val | Dev |
|---|---|---|
| bpb better (G1) | 20 / 20 | 20 / 20 |
| NLL better (G1) | 20 / 20 | 20 / 20 |
| mean rank better | 20 / 20 | 19 / 20 |
| top-5 better | 19 / 20 | 20 / 20 |
| top-1 better | 17 / 20 | 20 / 20 |

- **No cell shows the Val/Dev sign inversion** seen at steps 1750 / 3000 of the
  1.5x full-8000 run: bpb and NLL move down on both splits in all 20 cells.
  That is a property of this 500-step horizon, not a resolution of the longer
  run's split disagreement.
- The late-step advantage is **Val-dominant**. At 1.25x step 500 the Dev gain
  shrinks to Δbpb `−0.002972` / ΔNLL `−0.004908`, Dev mean rank gets
  `+0.904` worse, and Val top-1 drops 29 tokens. Val top-1 also regresses at
  1.5x step 500 (−22 tokens) and 2.0x step 200 (−3 tokens).
- Consequence for wording: report "canonical bpb and NLL improve on both
  splits", not "every metric on every split improves", and do not read any of
  this as a calibration or robustness result (seed 1, 256/256 windows, no ECE).

## Stability

| LR | Control | G1 |
|---:|:-------:|:--:|
| 1.00 | STABLE | STABLE |
| 1.25 | STABLE | STABLE |
| 1.50 | STABLE | STABLE |
| 2.00 | STABLE | STABLE |

No arm hit a hard stop (non-finite / fallback / QNN-HVX fatal / loss explosion).
Both max stable multipliers = **2.0x** (500-step stress region only).

**Stability claim:** No observed 500-step stability-region expansion up to 2.0x.
G1 does **not** expand the stability region in this grid.

Soft flags: G1 gate mean falls with LR (see Gate behavior). Control 2.0x wall
time was elevated (~904 s vs ~770 s); thermal status stayed `0` in every arm.

## Time-to-bpb (primary = first observed eval checkpoint at or below target)

Targets fixed a priori: 3.10 / 3.00 / 2.95 / 2.90 / 2.85.

**Wall semantics (corrected):** Muon-hybrid reports emit only full-run
`training_total_seconds`. Checkpoint mtimes are clustered at end-of-run and do
not recover cumulative training wall at intermediate steps. Therefore
`*_checkpoint_training_wall_s` is the **measured cumulative training wall**
only when the first-hit step equals `completed_steps` (here 500). Otherwise it
is `NOT_MEASURED`. A 500-step full-run wall is never reused as a step-300/400
target wall. Runtime full-run totals remain in `runtime.csv`.

| target | Ctrl best LR | Ctrl step | Ctrl ckpt wall | G1 best LR | G1 step | G1 ckpt wall | Δstep | Δwall |
|-------:|-------------:|----------:|---------------:|-----------:|--------:|-------------:|------:|------:|
| 3.10 | 1.50 | 300 | NOT_MEASURED | 2.00 | 300 | NOT_MEASURED | 0 | CENSORED |
| 3.00 | 2.00 | 400 | NOT_MEASURED | 1.50 | 400 | NOT_MEASURED | 0 | CENSORED |
| 2.95 | 2.00 | 400 | NOT_MEASURED | 1.50 | 400 | NOT_MEASURED | 0 | CENSORED |
| 2.90 | 1.25 | 500 | 776.2 | 1.50 | 400 | NOT_MEASURED | **-100** | CENSORED |
| 2.85 | — | >500 | — | — | >500 | — | — | — |

### Primary time-to-target conclusion (step-based)

At matched 1.5x, G1 first observes Balanced ≤ 2.90 at **step 400**
(Balanced 2.892) while best control first observes it at **step 500**
(Control 1.25x Balanced 2.888).

```text
observed step-to-target 2.90:
Control best = 500 steps
G1 best      = 400 steps
= 20% fewer optimizer steps
```

Wall-time improvement is **not claimed**: checkpoint cumulative training wall
was not measured at step 400. Only the Control 1.25x step-500 wall (776.2 s)
is a measured cumulative figure.

### Original UTF-8 bytes (estimated only)

Muon-hybrid reports in this harness do not emit a per-step
`target_utf8_bytes_seen`. Columns are therefore
`*_estimated_original_bytes_to_target`, computed step-proportional to the
8000-step canonical exposure 5,491,256 (e.g. 500-step ≈ 343,203.5). These are
**estimates**, not measured counters. Equal-step exposure is identical across
Control/G1 under the shared DataCursor / batch / order, so equal-step
comparisons remain valid.

## Gate behavior (checkpoint-static, Val 256 windows)

| LR | step100 mean | step500 mean | step500 min head | step500 max head | step500 mean `g<0.1` | heads with `g<0.1` ≥ 0.999 |
|---:|-------------:|-------------:|-----------------:|-----------------:|----------------------:|---------------------------:|
| 1.00 | 0.2276 | 0.1541 | 0.0291 | 0.5290 | 0.4655 | 0 |
| 1.25 | 0.1955 | 0.1311 | 0.0322 | 0.4759 | 0.5626 | 1 (L8h0) |
| 1.50 | 0.1623 | 0.1064 | 0.0201 | 0.3256 | 0.6576 | 12 (L5–L9, L12) |
| 2.00 | 0.1221 | 0.0776 | 0.0162 | 0.2997 | 0.7532 | 18 (L4–L12 partial, L15, L16) |

At 2.0x several deep-layer heads show `g<0.1` fractions near 1.0
(e.g. L4–L12 multiple heads). This is **soft saturation / strong branch
suppression**, not a hard failure. It is **not** evidence that the gate
collapsed globally or that training diverged: all 2.0x arms completed 500
finite steps with QNN/HVX clean health. No arm was hard-stopped on gate
saturation alone.

Wg weight norm rises slightly with LR (mean 1.03→1.12, max 1.51→1.83).

## Muon geometry

`muon_row_geometry_audit.py` on 1.5x G1 step 100→500:
`corr(max row norm, spectral norm) ≈ 0.98`, max-row-norm ratio ≈ 1.42,
spectral-norm ratio ≈ 1.44 over 500 steps. Consistent with known early drift;
no new pathology unique to high LR.

## Runtime caveat

Full-run training wall is recorded in `runtime.csv` (excludes eval). G1
overhead is ~3% ms/update vs control (extra Wg projection). Control 2.0x wall
was inflated (~1808 ms/update vs ~1540); do not treat that single arm as a G1
speedup. Full-run walls must not be read as intermediate checkpoint
time-to-target (see Time-to-bpb).

ReLU²-adjacent cross-phase runtime anomalies were not observed; E0 backlog
item unchanged.

## Health

- QNN return success: 8/8
- HVX failures / fallback / non-finite: 0 / 0 / 0
- eval reports: 40 / 40 `status=SUCCESS` with
  `validation_nonfinite_chunks=0` and `development_nonfinite_chunks=0`
- thermal status `0` before and after all 8 arms (battery temperature 29–31 °C;
  battery level 83 % → 66 % over the single-device night)
- focus takeover: 0
- checkpoint decode: all interval checkpoints verified
- historical 1.0x sanity: exact match
- Val NLL decreases monotonically across the 5 checkpoints in all 8 arms
  (endpoint-level evidence only; see Limitations)

## Runner audit (committed tree)

`scripts/run_headwise_g1_lr_stress.ps1` and
`scripts/run_nicopedia_htp_training.ps1` were audited at HEAD.

- `Get-StressHardStopFlags` requires `status=SUCCESS`, `all_steps_finite`,
  `final_finite`, `output_tensors_finite`, `qnn_return_code_success`, and
  forbids `cpu_fallback` / `fallback` / `nan_detected` / `inf_detected`, plus
  zero `api_trace_graph_execute_failure_count` / `hvx_rpc_failure_count` /
  `hvx_fallback_count` / `hvx_nonfinite_count`. This contract correctly
  classifies the 8-arm grid as SUCCESS/STABLE.
- `linear_decay` allow-list accepts scaled peaks `0.0022 / 0.00275 / 0.0033 /
  0.0044` and targets including `0.000125 / 0.00015`, with parent LR equal to
  peak LR. Matches the resolved stress grid.
- A prior session edit failure (`String to replace not found` /
  `ChildProcess.kill`) did **not** leave gaps in the committed tree; the
  required hard-stop and schedule logic are present. No runner code change was
  required for this closure.

## Decision

**PROMOTE_G1_GATE_VARIANTS**

Reason:
1. 20/20 matched LR/checkpoint points: G1 canonical Balanced bpb is better.
2. Historical 1.0x step500 result reproduced exactly on the fresh rerun.
3. All arms finite / no fallback / no QNN-HVX error (8/8 SUCCESS, STABLE).
4. Target 2.90: G1 1.5x first observes ≤2.90 at step 400; best Control first
   observes it at step 500 → **20% fewer optimizer steps**.
5. Observed 500-step stability region is the same as control (both 2.0x). No
   stability-region expansion is claimed.

Wall-time improvement is **not** a required PROMOTE condition and is **not**
claimed (checkpoint cumulative training wall was not measured at intermediate
steps).

Correct combined claim with historical 8000 evidence:

> G1 does not establish a final-quality gain at the baseline LR on seed1 at
> step 8000, but it improves short-horizon canonical bpb at matched LR and
> reduces observed step-to-bpb under the tested LR grid. Observed 500-step
> stability region is the same as control (2.0x). This promoted gate-variant
> exploration.

Split-level qualifier for that claim: within the grid Val and Dev bpb / NLL
move down together in all 20 cells (no split sign inversion), but the late-step
advantage is Val-dominant and Val top-1 regresses in 3 cells. See
[Split-level audit](#split-level-audit-of-the-same-20-cells-recomputed-from-primary-reports).

## Follow-up status (updated after this stress grid)

- A/B #1 (identity-init `2*sigmoid` + `Wg=0`): **done**, decision
  `KEEP_CURRENT_G1` — see [g1-identity-init-500.md](g1-identity-init-500.md).
  Identity lane closed.
- A/B #2 (Fixed 0.5 branch scale, no `Wg`): **done**, decision
  `KEEP_CURRENT_G1_LEARNED_GATE` —
  see [g1-fixed-half-500.md](g1-fixed-half-500.md). Fixed-scale lane closed as a
  replacement; the 0.5 suppression prior explains most of the early gain.
- 1.5x long-horizon continuation (2000, then 8000): **done**. G1 keeps a small
  Balanced-bpb edge at 2000 but the split-level disagreement at steps
  1750 / 3000 is unresolved —
  see [g1-1p5x-long-2000.md](g1-1p5x-long-2000.md) and
  [g1-1p5x-full-8000-2026-09](results/g1-1p5x-full-8000-2026-09/README.md).
- **Next lane = multi-seed replication of the 1.5x arm (design only, not run)**:
  seed 2 / 4, fresh step-0, 3000 steps, split-level trajectory as the primary
  judgment. 1.5x was selected from primary results (best mean and late-step
  ΔBalanced, and the only arm with a step-based time-to-bpb gain); 2.0x is
  excluded because gate saturation reaches 18/38 heads. Long training is Tier 3,
  so nothing has been started — see
  [g1-1p5x-multiseed-3000.md](g1-1p5x-multiseed-3000.md).
- reduced-channel gate: still unimplemented / out of scope.

## Artifacts

- `docs/results/g1-lr-stress-2026-09/` — committed compact evidence tree
  (primary `eval256-step*-htp.txt` / `*-result.txt` per arm + analyzer CSVs).
  Its [README](results/g1-lr-stress-2026-09/README.md) holds the split-level
  audit tables recomputed from the primary reports.
- `build/g1-lr-stress/` — raw checkpoints, gate-diagnostic inputs, telemetry
  (not committed)
- runner: `scripts/run_headwise_g1_lr_stress.ps1`
- analyzer: `scripts/g1_lr_stress_analyze.py`

## Limitations

- seed 1 only (candidate selection / stability diagnostic)
- 500-step stress, not long-horizon stability
- original-byte time-to-target uses step-proportional estimate
  (`estimated_original_bytes_to_target`); no per-step byte counter in
  Muon-hybrid report
- checkpoint cumulative training wall is only measured at the full-run
  endpoint (step 500); intermediate target walls are `NOT_MEASURED`
- single device night; Control 2.0x wall outlier
- **no per-update loss, gradient-norm, or update-norm series is emitted by the
  Muon-hybrid report**, so a transient loss spike between eval checkpoints is
  unobservable here. Stability is asserted only at endpoint level
  (finiteness + hard-stop flags + monotone 5-point Val NLL), and the absence of
  a spike is **not** claimed
- the 20-cell split-level audit is single-seed: 3 / 20 cells regress on Val
  top-1 and 1 / 20 on Dev mean rank, so "G1 improves every metric on every
  split" is not a supported statement
- lr2.0 Grid mode was interrupted (`HOST_KILLED` / multi-device adb); Control
  2.0x was completed via a separate `Run`. Aggregate CSVs cover all 8 arms;
  compact lr2.0 raw artifacts were copied from `build/g1-lr-stress` into this
  tree for the evidence record.
