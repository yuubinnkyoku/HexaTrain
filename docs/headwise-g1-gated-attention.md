# Headwise G1 gated-attention experiment

`headwise_g1_sigmoid` is the **current research baseline** since 2026-10-04
(`PROMOTE_WITH_RUNTIME_FOLLOWUP`); `attention_gate=none` is retained as the
**legacy / ungated control**. See [Baseline promotion](#baseline-promotion).

This research variant changes one architecture operation only. The legacy
control keeps `attention_gate=none`; the candidate uses
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

These are **candidate-only resource-estimator / structural counts** for the FULL
training graph, computed by the application resource estimator
(`transformer_resource_estimator.h`), not measurements. They are **not**:

- RSS / peak-RAM measurement
- ORT (resident set) measurement
- a physical checkpoint-file-size delta measurement

Per-run QNN node/tensor output and physical checkpoint payload delta were
`NOT_MEASURED` in the multi-seed lane. The control estimate is unchanged. The
counts include forward gate projection/sigmoid/head broadcast-multiply and the
explicit backward chain, including the extra LN1-gradient accumulation; see
[Baseline promotion](#baseline-promotion) for the per-node and per-tensor
classification.

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

## Baseline promotion

Promotion decision: **`PROMOTE_WITH_RUNTIME_FOLLOWUP`** (2026-10-04).

Scope of this promotion: `headwise_g1_sigmoid` becomes the **current research
baseline** for subsequent architecture research — the parent arm that a new
candidate is compared against. It does **not** change production/default
behavior, does not remove the ungated control, and is not a quality
re-verification of G1.

Terminology used from here on:

| name | `attention_gate` | role |
|---|---|---|
| **legacy control / ungated control** | `none` | historical anchor and regression reference. Implementation, `NPRTCKPTV4`, ungated regression tests, and ungated generation/eval compatibility are retained unchanged. |
| **current research baseline** | `headwise_g1_sigmoid` | parent arm for new architecture candidates |

Existing script/API identifiers (`attention_gate=none`, `ARM_IDENTITY["control"]`,
runner `-Arm Control`, artifact directory `control/`) are **not** renamed: those
names are part of existing evidence provenance and renaming them would break
traceability of already-published results.

### Quality evidence

Quality evidence is the preregistered multi-seed lane and is independent of the
systems promotion judgement below. Primary sources:
[g1-1p5x-multiseed-3000.md](g1-1p5x-multiseed-3000.md) (protocol and combined
interpretation) and
[results/g1-1p5x-multiseed-3000-2026-09/](results/g1-1p5x-multiseed-3000-2026-09/).

- Preregistered decision (seed 2 / 4, `analyzer` exit 0, `problems=0`):
  `decision: ambiguous_tie_breaker`. **This decision is not modified by the
  promotion.** The exploratory seed 3 is excluded from R1–R5
  (`EXPLORATORY_SEED_EXCLUDED seeds=[3]`).
- Early simultaneous Val/Dev improvement (R1) reproduces in **4/4 seeds**
  (seed 1 reference + seeds 2 / 4 preregistered + seed 3 exploratory).
- The late Dev reversal is observed in only **2/4 seeds** (seed 1: steps 1750 +
  3000; seed 4: step 2500; seeds 2 / 3: none), and the positions do not align
  across seeds. It is read as seed- and trajectory-dependent, **not** as a
  consistent structural event. It is not claimed that the reversal does not
  exist, nor that it is a confirmed G1 mechanism.
- Stability: all arms `status=SUCCESS`, `graphExecute 24000/24000`, QNN
  failure 0, tensor finite on all splits, HVX failure / fallback / non-finite
  `0 / 0 / 0`, no CPU fallback, no 6031 recurrence.

What this evidence does **not** establish: that G1 improves on every seed,
every step, and every metric; that late Dev reversal is absent; or that G1 is
faster at runtime.

### Systems cost

Primary source:
[g1-1p5x-baseline-promotion-cost.md](g1-1p5x-baseline-promotion-cost.md)
(paired Control vs G1 on seeds 2 / 3 / 4, extracted from existing artifacts
only — **no additional training was run**).

Architecture / resource cost (identical across all seeds):

| metric | Control | G1 | delta |
|---|---:|---:|---:|
| parameter_count | 758,528 | 760,960 | **+2,432 (+0.321%)** |
| Muon matrix count | 114 | 114 | 0 |
| Muon parameter count | 622,592 | 622,592 | 0 |
| Aux Adam parameter count | 135,936 | 138,368 | +2,432 |

The added parameters are gate weights on the **Aux Adam** side only; the Muon
matrix set is unchanged.

Runtime cost (device-side per-update, paired ratio, Δ = G1 − Control, positive
= G1 slower). Median over 3 seeds:

| metric | median Δ |
|---|---:|
| HTP forward+backward | **+9.95%** |
| mean QNN execute / call | **+9.95%** |
| parameter transfer | +20.07% |
| HVX Muon | +2.10% |
| wall/update (`training_step_ms`) | −4.76% |
| `ESTIMATED_TIME_TO_BPB` | median advantage +4.76%, G1 wins 17/21 targets |

Seed 3's absolute runtime values are outliers (its Control arm alone runs at
~3.6x seed 2's per-update cost under a run-wide condition difference). The
arm-pair ratios remain usable, so only the paired Control/G1 ratio is used.
Cross-seed absolute runtime comparison is
**`CROSS_SEED_ABSOLUTE_RUNTIME_NOT_COMPARABLE`** and the cause of the seed 3
condition difference is **not** attributed to any specific mechanism such as
DVFS. With 3 seeds no statistical significance is claimed — only sign
consistency.

The estimator deltas quoted in [Resource delta](#resource-delta) above are
**candidate-only structural counts from the application resource estimator**
(`transformer_resource_estimator.h`), not measurements. Specifically they are
**not**: RSS / peak-RAM measurement, ORT measurement, or a physical
checkpoint-file-size delta measurement. Per-run QNN node/tensor output,
checkpoint payload bytes, and checkpoint timestamps were `NOT_MEASURED` in this
lane.

### Promotion decision

`PROMOTE_WITH_RUNTIME_FOLLOWUP`.

Criteria as defined in advance:

| verdict | condition |
|---|---|
| PROMOTE | quality gain holds / parameter ≲1% / no stability regression / wall ≤5% / time-to-bpb improves |
| PROMOTE_WITH_RUNTIME_FOLLOWUP | quality and time-to-bpb win, but wall is +5–10% or QNN execute overhead is clear |
| HOLD | runtime cost roughly cancels the quality advantage |
| DO_NOT_PROMOTE | loses on time-to-bpb, or a G1-specific stability problem |

Basis for the verdict:

- Parameter overhead +0.321% satisfies the ≤1% criterion.
- Quality gain holds under multi-seed (R1 4/4).
- No stability regression, no 6031 recurrence.
- `ESTIMATED_TIME_TO_BPB` improves (median +4.76%, 17/21 targets).
- However HTP execute / mean QNN execute is clearly **+9.95%** (median), and the
  wall/update sign is inconsistent across seeds (seeds 2 / 4 negative, seed 3
  positive). This does not reach the strongest PROMOTE condition
  ("wall +0–3% and time-to-bpb improves").

Correct statement of the promotion:

> G1 has a very small parameter overhead, and the HTP `graphExecute` itself is
> about 10% heavier. On the other hand, in the observed quality trajectory it
> reduces the number of steps needed, and on the existing target set the
> estimated time-to-bpb is overall better than Control. It is therefore promoted
> to baseline, with the HTP runtime overhead retained as a follow-up
> optimization item.

Statements that must **not** be made: that G1 is faster at runtime; that the
HTP cost is free; that the cause of seed 3 being slower is determined; that G1
improves on all seeds / steps / metrics; that the late Dev reversal does not
exist.

### Remaining runtime follow-up

This is a separate optimization lane, **not** a G1 re-verification lane. Its
primary goal is to reduce the HTP `graphExecute` overhead of ≈ +10% **without
changing G1 quality**.

Where the `+399` nodes and `+476` tensors originate (classification from
`transformer_resource_estimator.h` and the graph builder in
`app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc`; L19/H2):

Forward, per layer `2 + 2*H = 6` nodes (`+114` total):

| nodes | emitter | note |
|---:|---|---|
| 1 | gate projection MatMul `LN1 × Wg` | the gate projection itself |
| 1 | `QNN_OP_SIGMOID` on gate logits | |
| `2*H` = 4 | per-head gate select MatMul + broadcast multiply | `H2` makes this small; a larger `H` scales this term linearly |

Backward, per layer `5*H + 5 = 15` nodes (`+285` total):

| nodes | emitter | note |
|---:|---|---|
| `H` = 2 | per-head `dAh = dYh * Gh` | |
| `H` = 2 | per-head `dGh = dYh * Ah` product | |
| `H` = 2 | per-head reduce of `dGh` | |
| `H` = 2 | per-head scatter of `dGh` back to `[T,H]` | |
| 1 | `1 - G` | |
| 1 | `G * (1 - G)` derivative | |
| 1 | `dZ = dGh * G(1-G)` | |
| 1 | `dWg = LN1ᵀ * dZ` | |
| 1 | `dN_gate = dZ * Wgᵀ` | |
| 1 | `dLN1 = dLN1_qkv + dN_gate` accumulation | the extra LN1 gradient accumulation |

Tensors, per layer `9 + 8*H = 25` (`+475`) plus 1 extra global `APP_READ` slot
for the `attention_gates` output (`+1`), i.e. `25 × 19 + 1 = +476`.

Candidate directions (not implemented in this pass):

- fuse the gate projection / sigmoid / broadcast multiply chain
- remove unnecessary intermediate tensors (`1-G`, `G(1-G)` are elementwise and
  may be foldable into consumers)
- reduce per-head select/scatter MatMuls by keeping the gate broadcast in
  `[T,H]` form instead of the `[T,1]` per-head form
- fuse the backward chain and fold the LN1 gradient accumulation
- reduce parameter transfer overhead (measured +20.07% median)
- reconsider the treatment of the static `[64,2]` small-shape gate
- graph construction overhead

Success conditions for a future optimization pass (research targets, **not**
gates on this promotion):

- G1 mathematical definition unchanged
- checkpoint compatibility maintained
- parameter hash / deterministic parity as expected
- existing quality evidence not invalidated
- reduced HTP execute overhead, no fallback, all tensors finite, no new 6031
  signal
- target: reduce the current ≈ +10% HTP execute overhead to **≤ +5%**, and if
  possible **≤ +3%**
