# Headwise G1 gated-attention experiment

This research variant changes one architecture operation only. The baseline
keeps `attention_gate=none`; the candidate uses
`attention_gate=headwise_g1_sigmoid` with one bias-free `Wg[D,H]` per layer.

For normalized block input `N=LN1(X)` and head context `Ah=SDPAh(Q,K,V)`:

```text
Z = N Wg
G = sigmoid(Z)
Yh = Ah * G[:,h:h+1]
Y = concat(Y0..YH-1) Wo
```

The gate is therefore after SDPA and immediately before concatenation/`Wo`.
It does not modify attention probabilities, V, the post-`Wo` value, the
residual, or the FFN. The manually constructed backward graph is:

```text
dAh      = dYh * Gh
dGh      = reduce_sum(dYh * Ah, axis=head_dimension)
dZ       = dGh * G * (1-G)
dWg      = transpose(N) dZ
dN_gate  = dZ transpose(Wg)
dLN1     = dLN1_qkv + dN_gate
```

## Parameter and optimizer identity

The candidate adds `64*2*19=2,432` parameters, producing 760,960 total.
`attention_gate_weight` follows the parameter registry's per-layer naming and
has shape `[64,2]`.

It is classified as `AUX_ADAM`, not Muon. The production HVX Original-Muon
contract has fixed 64x64, 64x128, and transposed 128x64 packing groups; 64x2
does not fit that contract. Adding a new packing group and W8 schedule would be
a disproportionate optimizer refactor for this architecture experiment and
could perturb the 114 existing matrices. The resulting split is:

```text
Muon:       114 matrices, 622,592 parameters
Aux Adam:    97 entries,  138,368 parameters
Total:                    760,960 parameters
```

The ungated registry and V4 checkpoint byte identity remain unchanged. Gated
mixed-optimizer checkpoints use `NPRTCKPTV5`, schema 5, registry version 2,
and serialize the architecture enum. Resume, evaluation, cold generation,
and prepared FORWARD_ONLY generation compare the requested architecture with
the checkpoint identity and fail closed on a mismatch.

## Resource delta

For V1024/T32/D64/FFN128/L19/H2, the application resource estimator reports
the following candidate-only deltas for the FULL training graph:

```text
parameters/checkpoint payload: +2,432 floats = +9,728 bytes
Adam m/v state:                +4,864 floats = +19,456 bytes
QNN nodes:                     +399
QNN tensors:                   +476
```

The baseline estimate is unchanged. The counts include forward gate
projection/sigmoid/head broadcast-multiply and the explicit backward chain,
including the extra LN1-gradient accumulation.

## Initialization and diagnostics

`Wg` uses the existing Q/K/V/O linear initialization scale and deterministic
seed policy; it has no bias. A host diagnostic evaluates the formal L19/H2
initial model and reports overall mean/min/max and saturation fraction. Device
training reports per-layer/per-head mean, standard deviation, min/max,
`g<0.1` fraction, and `g>0.9` fraction as aggregate values only.

## A/B protocol

[`scripts/run_headwise_g1_ab.ps1`](../scripts/run_headwise_g1_ab.ps1) freezes
the V1024/T32/D64/FFN128/L19/H2 seed-1 HVX-Muon S4000-style recipe. Its default
`Plan` mode is read-only, `Smoke1` runs one candidate update, and the explicit
`Candidate2000` mode creates checkpoints/evaluations at steps 500, 1000, 1500,
and 2000. A control artifact may be reused only when its tokenizer, dataset,
order, seed, optimizer, schedule, and `attention_gate=none` identity match.
The script does not start 4000- or 8000-step training.

## Verification status (uncommitted)

Host suite (`scripts/run_host_tests.ps1`) passes, including the new
`headwise_g1_gate_test` and ungated regressions. Pinned QAIRT APK audit
succeeds for the QNN/HVX-enabled debug APK.

`Smoke1` on the NX741J device completed one HTP forward/backward update with:

```text
attention_gate=headwise_g1_sigmoid
parameter_count=760960
checkpoint_format=NPRTCKPTV5
forward_backward_backend=HTP
optimizer_muon_backend=HVX_W8
qnn_return_code_success=true
output_tensors_finite=true
cpu_fallback=false
focus_takeover_count=0
```

Per-layer/per-head gate aggregates were finite and unsaturated
(`g<0.1` and `g>0.9` fractions all 0; layer means roughly 0.30–0.66).
The stale pre-V5 `htp_checkpoint_eval.exe` initially rejected the new
magic; the evaluator now rebuilds when its source is newer than the binary,
and V5 decode of the smoke checkpoint succeeds.

Remaining formal gates closed in the follow-up pass:

- `verify_local.ps1 -WithQairt`: PASS (30/0)
- Matched 8-update ungated/gated device smokes: both PASS under the same
  APK/QAIRT/seed/recipe; parameter counts 758,528 / 760,960 and formats
  NPRTCKPTV4 / NPRTCKPTV5
- Gated V5 resume (fresh 4 → resume → 8) is bitwise identical to fresh 8
- Gated FORWARD_ONLY generation (`htp-smoke`, Greedy, 4 bytes) PASS
- Ungated FORWARD_ONLY generation regression PASS
- `Candidate2000` runner remains Plan-only until explicitly selected

## Candidate2000 quality result (seed1, 256/256 eval protocol)

Control is the formal HVX Muon baseline
(`quality-hvx-seed1-step8000` + `quality-hvx-step1000` step500),
`attention_gate=none`, `muon_lr=0.005`, NPRTCKPTV4.

| step | control Bal | candidate Bal | ΔBalanced |
|-----:|------------:|--------------:|----------:|
| 500 | 2.896905 | 2.864193 | **-0.032712** |
| 1000 | 2.759711 | 2.723026 | **-0.036686** |
| 1500 | 2.678232 | 2.656368 | **-0.021864** |
| 2000 | 2.613837 | 2.606724 | **-0.007113** |

Triage: **B PROMISING** (2000 ΔBalanced in -0.005..-0.02, curve stays
improving; early points were stronger). Candidate training health was clean
(NPRTCKPTV5, 760,960 params, HTP+HVX_W8, no fallback/non-finite).

Gate fields in training reports are named
`gate_training_trajectory_l{L}_h{H}_*` with
`gate_diagnostics_kind=training_trajectory_aggregate`. They accumulate gates
from every training batch of a *changing* model and are **not**
checkpoint-static distributions. The earlier note that step2000
`mean_of_head_means ≈ 0.156` refers to that trajectory aggregate only.
Checkpoint-static diagnostics (fixed Val windows, frozen weights) are produced
by `headwise_g1_gate_diagnostics` and stored as
`candidate-eval256-step{N}/gate-static-diagnostics.txt`.

Val + Dev are architecture / continuation **model-selection** splits. Dev is
not an unseen holdout or final confirmation split. The final split remains
unopened.

Runner note: A/B script `muon_learning_rate` was corrected from the draft
`0.010` to the formal-control-matched `0.005`. Device eval must propagate
`attentionGate` (fixed in `nicopediaHtpEvaluate` dispatch). Host CPU evaluator
now reuses production `forwardTraceGeneralized` so G1 is applied.

2000-step run STOPs here; 4000/8000 are not started.
