# G1 high-LR stress grid — evidence tree (2026-09)

Split-level audit record for the LR stress grid documented in
[`docs/g1-lr-stress.md`](../../g1-lr-stress.md). Every number below was
recomputed from the primary reports in this tree; the derived CSVs in the
directory root are analyzer output, not primary evidence.

## Experiment identity

- arms: Control (`attention_gate=none`, 758,528 params, NPRTCKPTV4) vs
  G1 (`attention_gate=headwise_g1_sigmoid`, 760,960 params, NPRTCKPTV5)
- multipliers: 1.0x / 1.25x / 1.5x / 2.0x applied to Muon **and** Aux Adam
- model V1024 / T32 / D64 / FFN128 / L19 / H2, seed 1, batch 8, fresh step-0
  starts, `linear_decay` 4000→8000 with a 500-step horizon (constant peak LR)
- eval: Val first 256 + Dev first 256 chunks, HTP native
  (`evaluation_math_responsibility=HTP`), 8,192 scored tokens per split
- device: NX741J, HTP forward/backward, HVX_W8 Muon, CPU Aux Adam
- QAIRT build ID `2.48.40.260702151143` for all 8 arms
- shared training order hash `fnv1a64:ce1000529cb0eff4`, dataset
  `fnv1a64:0c7b2826f5f26fea`

## Layout

- `lr{1.0,1.25,1.5,2.0}/{control,g1}/eval256-step{100,200,300,400,500}-htp.txt`
  — 40 primary eval reports
- `.../seed1-l19-v1024-t32-d64-f128-steps500-result.txt` — primary training
  report per arm; `lr1.0` additionally holds the matched 8-update smoke reports
- `.../gate-static-step{100..500}.txt` — G1 only, host-side gate diagnostics
- root CSV/JSON (`quality*.csv`, `stability.csv`, `runtime.csv`,
  `device-health.csv`, `gate-static.csv`, `gate-trajectory.csv`,
  `training-telemetry.csv`, `wgate-geometry.csv`, `resolved-grid.csv`,
  `shared-parameter-hash.csv`, `time-to-bpb.csv`, `grid-outcomes.json`,
  `stress-summary.json`, `run-order.json`) — analyzer aggregates

## Run health (primary reports)

- 8/8 arms: `status=SUCCESS`, `completed_steps=500`
- 40/40 eval reports: `status=SUCCESS`, `validation_nonfinite_chunks=0`,
  `development_nonfinite_chunks=0`, `api_trace_graph_execute_failure_count=0`
- 8/8 arms: `qnn_return_code_success=true`, `cpu_fallback=false`,
  `nan_detected=false`, `inf_detected=false`, `hvx_rpc_failure_count=0`,
  `hvx_fallback_count=0`, `hvx_nonfinite_count=0`, `focus_takeover_count=0`
- thermal status `0` before and after every arm; battery temperature 29–31 °C;
  battery level fell 83 % → 66 % across the single-device night
- no arm carries a hard-stop flag; `stability.csv` is `STABLE` for all 8 arms

| LR | Control first/last train loss | G1 first/last train loss |
|---:|------------------------------:|-------------------------:|
| 1.00 | 7.489538 / 4.770673 | 7.334945 / 4.677936 |
| 1.25 | 7.489538 / 4.747361 | 7.334945 / 4.643236 |
| 1.50 | 7.489538 / 4.774783 | 7.334945 / 4.699153 |
| 2.00 | 7.489538 / 4.759222 | 7.334945 / 4.670839 |

Val NLL at the five checkpoints decreases monotonically for every arm, so
nothing in this tree indicates an unrecovered divergence at the 100-step
sampling resolution. These reports contain **no** per-update loss, gradient, or
update-norm series, so a transient spike between checkpoints is not observable
here and is **not** claimed to be absent.

## Split-level paired deltas (G1 − Control, recomputed from primary reports)

Negative bpb / NLL = G1 better. Top-1 is reported as scored-token counts out of
8,192 per split.

| LR | step | ΔVal bpb | ΔDev bpb | ΔVal NLL | ΔDev NLL | ΔVal top1 | ΔDev top1 |
|---:|-----:|---------:|---------:|---------:|---------:|----------:|----------:|
| 1.00 | 100 | −0.099852 | −0.147405 | −0.180177 | −0.243373 | +40 | +86 |
| 1.00 | 200 | −0.054795 | −0.085780 | −0.098875 | −0.141627 | +105 | +165 |
| 1.00 | 300 | −0.026214 | −0.047221 | −0.047301 | −0.077964 | +70 | +64 |
| 1.00 | 400 | −0.040015 | −0.065002 | −0.072206 | −0.107322 | +9 | +75 |
| 1.00 | 500 | −0.026774 | −0.038650 | −0.048312 | −0.063812 | +54 | +72 |
| 1.25 | 100 | −0.124699 | −0.173687 | −0.225013 | −0.286766 | +68 | +137 |
| 1.25 | 200 | −0.065224 | −0.055099 | −0.117693 | −0.090970 | +37 | +88 |
| 1.25 | 300 | −0.025836 | −0.054333 | −0.046620 | −0.089706 | +127 | +90 |
| 1.25 | 400 | −0.028963 | −0.027370 | −0.052262 | −0.045190 | +42 | +90 |
| 1.25 | 500 | −0.028130 | −0.002972 | −0.050758 | −0.004908 | **−29** | +74 |
| 1.50 | 100 | −0.127974 | −0.153031 | −0.230923 | −0.252662 | +50 | +112 |
| 1.50 | 200 | −0.056934 | −0.058858 | −0.102734 | −0.097178 | +93 | +101 |
| 1.50 | 300 | −0.031191 | −0.043316 | −0.056283 | −0.071517 | +53 | +140 |
| 1.50 | 400 | −0.065693 | −0.060589 | −0.118539 | −0.100035 | +37 | +230 |
| 1.50 | 500 | −0.036408 | −0.023564 | −0.065696 | −0.038905 | **−22** | +10 |
| 2.00 | 100 | −0.109424 | −0.132800 | −0.197451 | −0.219260 | +55 | +71 |
| 2.00 | 200 | −0.037303 | −0.027555 | −0.067311 | −0.045495 | **−3** | +35 |
| 2.00 | 300 | −0.055464 | −0.081335 | −0.100083 | −0.134288 | +66 | +138 |
| 2.00 | 400 | −0.020218 | −0.027920 | −0.036482 | −0.046097 | +44 | +76 |
| 2.00 | 500 | −0.023769 | −0.030359 | −0.042890 | −0.050124 | +39 | +126 |

Sign counts over the 20 matched cells:

- bpb: better on Val 20/20 and on Dev 20/20 — **no** cell where Val improved
  while Dev worsened
- NLL: better on Val 20/20 and on Dev 20/20 (same sign on both splits in every
  cell)
- mean rank: better on Val 20/20, on Dev 19/20 (exception: 1.25x step 500 at
  `+0.904`)
- top-5: better on Val 19/20, on Dev 20/20
- top-1: better on Dev 20/20, but **regresses on Val in 3 cells** (1.25x
  step 500 −29 tokens, 1.50x step 500 −22 tokens, 2.00x step 200 −3 tokens)

Consistency check against the derived aggregate: `quality-paired.csv` matches
this recomputation to ≤ 1e-9 on all 20 cells and all 9 fields.

### Gate behavior at the checkpoint endpoints

Step 500, over the 38 heads of `gate-static.csv`:

| LR | mean of head means | min head mean | max head mean | mean `g<0.1` | heads with `g<0.1` ≥ 0.999 |
|---:|-------------------:|--------------:|--------------:|-------------:|---------------------------:|
| 1.00 | 0.1541 | 0.0291 | 0.5290 | 0.4655 | 0 |
| 1.25 | 0.1311 | 0.0322 | 0.4759 | 0.5626 | 1 |
| 1.50 | 0.1064 | 0.0201 | 0.3256 | 0.6576 | 12 |
| 2.00 | 0.0776 | 0.0162 | 0.2997 | 0.7532 | 18 |

`g>0.9` never exceeds `0.0011` for any head, and `Wg` weight norm grows only
mildly with LR (mean 1.028 → 1.118, max 1.514 → 1.826 from 1.0x to 2.0x;
`wgate-geometry.csv`). This is low-side branch suppression that scales with LR,
not a hard failure — and with no gradient-norm or update-norm record in this
tree it is also **not** evidence that a suppressed branch is dead or that the
suppression is what produces the gain.

## What this licenses and what it does not

Licensed: at matched LR inside this grid, G1 lowers canonical Balanced bpb and
lowers **both** Val and Dev NLL at every one of the 20 checkpoints, and it does
so **without** expanding the observed 500-step stability region (max stable
multiplier 2.0x for both arms).

Not licensed:

- **A uniform quality improvement across metrics and splits.** The late 1.25x
  and 1.5x cells are Val-dominant: at 1.25x step 500 the Dev gain collapses to
  `−0.002972` bpb / `−0.004908` NLL while Val top-1 falls by 29 tokens and Dev
  mean rank rises by `+0.904`. Val top-1 regresses in 3 of 20 cells. Report the
  gain as bpb/NLL-consistent but metric- and split-specific at late steps, not
  as a monotone "better on everything".
- **Robustness, calibration, or generalization.** Seed 1 only, 256/256 eval
  windows, no ECE and no repeated seeds. The absence of a Val/Dev sign flip
  here is *not* a resolution of the sign flips observed at steps 1750 / 3000 of
  the 1.5x full-8000 run
  ([g1-1p5x-full-8000-2026-09](../g1-1p5x-full-8000-2026-09/README.md));
  this grid only shows the shorter 500-step horizon does not exhibit them.
- **Wall-time speedup.** Full-run training walls are in the table below; only
  the Control 2.0x arm is anomalous (1807.9 ms/update against ~1540 ms/update
  for its own 1.5x pair), so the 0.876 G1/Control ratio at 2.0x is a
  Control-side outlier and must not be read as a G1 speedup. `run-order.json`
  puts that arm last in the night (order 8 of 8, starting at 67 % battery
  against 69 % for its paired G1 arm), which is the plainest explanation, but
  the tree holds no continuous clock or thermal trace that could confirm it.

| LR | Ctrl wall s | G1 wall s | G1/Ctrl | Ctrl ms/update | G1 ms/update | Δ fwd/bwd ms | Δ Muon ms |
|---:|------------:|----------:|--------:|---------------:|-------------:|-------------:|----------:|
| 1.00 | 742.56 | 766.70 | 1.033 | 1485.11 | 1533.40 | +3.45 | +10.77 |
| 1.25 | 776.25 | 777.96 | 1.002 | 1552.50 | 1555.92 | +2.96 | +5.88 |
| 1.50 | 770.20 | 792.47 | 1.029 | 1540.40 | 1584.95 | +3.00 | +8.94 |
| 2.00 | 903.93 | 791.45 | 0.876 | 1807.86 | 1582.90 | +3.49 | +6.44 |

Aux Adam differs by less than 0.1 ms/update in every pair. G1's own component
deltas are stable across LR; the wall ratios are not, which is why
`time-to-bpb.csv` keeps intermediate walls as `NOT_MEASURED` / `CENSORED`.
Estimated original-byte throughput (step-proportional; equal-step exposure is
identical across arms under the shared DataCursor and order) is
462.2 / 447.6, 442.1 / 441.2, 445.6 / 433.1, 379.7 / 433.6 bytes/s for
Control / G1 at 1.0 / 1.25 / 1.5 / 2.0x.

## Reproduction

- runner: `scripts/run_headwise_g1_lr_stress.ps1`
  (`Plan` / `Smoke` / `Run` / `Grid` / `Finish` / `Analyze`)
- analyzer: `scripts/g1_lr_stress_analyze.py`
- decision and follow-up lanes: [`docs/g1-lr-stress.md`](../../g1-lr-stress.md)
- raw checkpoints, gate-diagnostic inputs, and per-run logs stay in ignored
  `build/g1-lr-stress/`; only the compact reports in this tree are committed
