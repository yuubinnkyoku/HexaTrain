# Nicopedia CPU/HTP parity gate: scientific re-audit protocol

Status: **MEASURED-2026-08-08 (protocol fixed before measurement; see appendix for the recorded outcome).**

This document fixes the parity re-audit protocol **before** the new device
metrics are measured. It defines the metric set, the candidate policies, the
adoption rule, and the acceptance conditions. Nothing in this file may be
tuned after device results are collected; any change invalidates the audit.

Scope: the CPU/HTP parity gate of the Nicopedia L19 seed-1 byte-level
language model (V=256, T=32, D=16, FFN=32, H=2), checkpoints
`htp-seed1-l19-step320.ckpt` and `htp-seed1-l19-step1000.ckpt` (same QAIRT
2.48.40.260702 build `2.48.40.260702151143`, same device class NX741J
SM8850 HTP V81).

## 1. Problem

The legacy gate (`qnn_transformer_training.cpp`, `nicopediaHtpGeneration`)

```
per-row PASS  <=> finite && raw_logits_max_abs < 2e-2 && probability_max_abs < 5e-3
AR gate       <=> per-step same tolerance && !(argmax diverged && cpu_margin > 1e-2)
```

rejects the step-1000 checkpoint on one prefix (`raw max abs ≈ 0.0308`) while
`argmax`, top-5 set overlap, probability error (<= 3e-3), and cosine
(> 0.999998) all agree at every AR step. The scientific hypothesis to test is
the **gauge freedom** of raw logits: softmax probabilities are invariant to a
per-row common offset `z -> z + c`, so a raw `max abs` bound conflates

1. common-mode offset (`c`) — decision- and distribution-irrelevant; and
2. shape noise (`delta - c`) — the only quantity that can change predictions.

We also test whether the observed growth 0.0111 (step320) -> 0.0308 (step1000)
is explained by logit-scale growth (CPU logit RMS 3.21 -> 3.57) or is a
genuine numerical degradation in relative terms.

## 2. Measured metrics (per prefix row = last-token logits row, V elements)

For CPU `z`, HTP `w`, `d_i = w_i - z_i`, `p = softmax(z)`, `q = softmax(w)`:

| metric | definition |
|---|---|
| delta_mean | mean(d) |
| delta_median | median(d) |
| delta_std | population std of d |
| raw_max_abs | max_i abs(d_i) |
| raw_rms | sqrt(mean d_i^2) |
| centered_max_abs | max abs of (d - mean(d)) |
| centered_rms | sqrt(mean (d - mean(d))^2) |
| logsoftmax_max_abs | max abs of the per-token log-softmax diff |
| logsoftmax_rms | sqrt(mean of squared log-softmax diffs) |
| prob_max_abs | max abs(p - q) |
| prob_mean_abs | mean abs(p - q) |
| prob_l1 | sum abs(p - q) (= 2 * TV distance) |
| js_divergence | symmetric Jensen-Shannon over (p,q), 1e-30 clamp |
| cosine_raw | cos(z, w) |
| cosine_centered | cos(z - mean(z), w - mean(w)) |
| cpu_rms / cpu_std / htp_rms / htp_std | per-side RMS / std |
| scale_ratio | htp_std / cpu_std (mean-centered std ratio) |
| rel_max | raw_max_abs / cpu_rms (scale-aware shape bound, reporting only) |
| argmax_cpu / argmax_htp | first-max argmax |
| margin_cpu / margin_htp | top1 - top2 gap on each side |
| topk_set_overlap / topk_order_match | top-5 index set overlap; exact order match |
| finite | all elements finite on both sides |
| row_degenerate | cpu_std <= 1e-6 (constant row, e.g. all-zero logits) |

Decision-theoretic identity: for the L-inf perturbation bound
`delta = raw_max_abs` of `w` against `z`, `argmax(w) == argmax(z)` is
*guaranteed* when `margin_cpu > 2*delta`. If argmax differs while
`margin_cpu > 2*delta`, the difference cannot be explained by a bounded
perturbation -> **anomaly (FAIL)**. If argmax differs and
`margin_cpu <= 2*delta`, the difference is a possible sampling-floor effect,
**recorded (`decision_ambiguous`), not failed**. This replaces the legacy
magic AR margin 1e-2 with an analytically derived bound.

## 3. Prefix set (20 total; 4 existing + 16 new; fixed now)

All synthetic/deterministic, no licensed corpus text, no article content.
Byte vectors defined in `nicopedia_generation.h::parityPrefixes()`.
Existing: `japanese_utf8_21`, `ascii_26`, `byte_edges_32`, `truncated_utf8_32`.
New: `ascii_short_5`, `japanese_13`, `japanese_32_exact`, `japanese_long_48`,
`mixed_13_punct`, `punctuation_24`, `control_whitespace_12`,
`digits_punct_32`, `utf8_emoji_16`, `katakana_12`, `halfwidth_16`,
`pseudo_random_32`, `hiragana_9`, `lowercase_31p`, `invalid_utf8_34`,
`leading_ff_8`. Evaluation covers all 20 prefixes for both gates.

## 4. Candidate policies (parallel shadow evaluation; fixed pre-result)

Per-row conjunction, evaluated on the same rows. `L` is the legacy gate
(unchanged, still the blocking gate until adoption).

| id | policy (per-row conjunction) |
|---|---|
| L (legacy) | finite && raw_max_abs < 2e-2 && prob_max_abs < 5e-3 |
| P (prob dim) | finite && !degenerate && prob_max_abs < 5e-3 && prob_l1 < 2e-2 && js_divergence < 5e-3 |
| C (shape dim) | P && centered_max_abs < 2e-2 && centered_rms < 5e-3 && logsoftmax_max_abs < 1e-1 && logsoftmax_rms < 5e-2 && scale_ratio in [0.995, 1.005] && raw_max_abs < 5e-1 |
| D (decision dim) | P && (argmax matches OR (margin_cpu <= 2*raw_max_abs, decision_ambiguous)) |
| F (full) | C && D |

Threshold rationale (all absolute, none derived from step-1000 values):

- prob_max_abs < 5e-3 — unchanged legacy acceptance (any token prob may move
  by at most 0.005).
- prob_l1 < 2e-2 — total-variation distance <= 1e-2 (probability mass moved
  by at most 1%).
- js_divergence < 5e-3 — distribution shape parity at well below the TV
  bound; any known-good run measures orders below this.
- centered_max_abs < 2e-2 — same absolute strength as the legacy raw bound,
  applied to the gauge-free residual. This is the central change; no raw
  loosen beyond this is attempted.
- centered_rms < 5e-3 — residual RMS at 0.5% of a 1.0-logit-scale unit.
- logsoftmax_max_abs < 1e-1 / logsoftmax_rms < 5e-2 — per-token log-prob
  distortion at most 0.1 nats.
- scale_ratio in [0.995, 1.005] — HTP numeric path must not rescale beyond
  FP16-rounding-scale (~1e-3 per element).
- raw_max_abs < 5e-1 — catastrophic guard: a *shape* error (or common
  offset) above 0.5 logits on a V=256 row cannot be survival noise and
  would dominate any probability; it is an overflow-margin guard, not the
  main per-element bound.
- decision: guarantee condition is `margin > 2*raw_max_abs` (theorem, not
  tuned). Rows with CPU margin <= 2*raw bound are ambiguous by construction.

- Legacy magic `cpu_margin > 1e-2` in AR divergence is replaced by the
  analytic `margin > 2*delta` rule, kept identical in spirit (see Section 9).

## 5. Adoption criteria (pre-fixed; ALL must hold)

1. Step-320 known-good run passes every clause of F (parity 20 prefixes + 8
   AR steps).
2. Step-1000 passes F for the full 20-prefix set and the 8 AR steps.
3. Same-step repeat (step-1000 rerun) shows the same pass/fail verdict
   (stability).
4. Synthetic fault battery (host) verdicts match Section 6 expectations.
5. NaN / Inf / -Inf always FAIL every policy.
6. Argmax-breaking perturbation at a decisive margin is FAILED.
7. Probability distortion (scale-down or mass redistribution) is FAILED.
8. Checkpoint-mismatch and CPU-fallback gates are untouched (independent).
9. No existing regression gates weakened (FFN372 square +Inf / nonfinite
   gates are in different code paths and untouched).
10. Independent Reviewer PASS (task agent, read-only).

Only if 1..10 hold does any candidate become the ACTIVE parity policy for
step-1000 generation. The adoption rule picks **F**; if F fails, the
strongest passing candidate is recorded for reporting only and generation
stays BLOCKED (allowed outcome — the audit then precisely identifies the
blocking criterion).

## 6. Synthetic fault battery (host test, deterministic, no device)

Reference row shape `z0`: synthetic spectrum, top-1 margin 0.3. Each fault
builds `w` and asserts the expected verdicts for L / F (and the DM gating
flag). Wrong verdict = host-test failure (CI).

|#|fault|expected|
|---|---|---|
|1|common offset +1e-3|L OK, F OK|
|2|common offset +3e-2|L FAIL, F OK (gauge irrelevance demonstrated)|
|3|common offset +5e-1|L FAIL, F FAIL (catastrophic raw bound)|
|4|single logit +5e-2|L FAIL, F FAIL (centered/shape + prob)|
|5|top1/top2 swap, margin 0.4, delta small|L FAIL, F FAIL (decision guarantee)|
|6|top1/top2 swap, margin 5e-3 (ambiguous)|F records `decision_ambiguous`, no FAIL (documented semantics)|
|7|logits x1.02|L OK, F FAIL (scale_ratio)|
|8|logits x1.004|L OK, F OK (in-band)|
|9|Gaussian noise sigma 1e-3|L OK, F OK|
|10|Gaussian noise sigma 1e-1|L FAIL, F FAIL|
|11|mass redistribution (softmax-shift: scale down interior logits x0.05)|L FAIL (prob_max), F FAIL|
|12|NaN element|L FAIL, F FAIL|
|13|+Inf element|L FAIL, F FAIL|
|14|-Inf element|L FAIL, F FAIL|
|15|all-zero row (degenerate) | both FAIL (row_degenerate rule)|

The battery also asserts the decision-margin theorem algebra on constructed
rows and prints a `PARITY_POLICY_HOST_RESULTS` block; the exporter
consumes its CSV column.

## 7. Device evidence protocol (fixed)

- Runs (each = one device/emulator parity invocation):
  - `audit_step320` — full parity (20 prefixes) + AR, greedy 64B
  - `audit_step1000` — full parity + AR (report allowed even if gate fails)
  - `step1000_greedy` / `step1000_sample` — only if F passes the audit
  - `step320_greedy` / `step320_sample` — same-APK rerun for apples-to-apples
  - optional repeat `step1000` audit (determinism)
- budgets: device parity runs <= 12 (8 planned), HTP generation <= 6,
  CPU generation <= 4 (existing CPU step1000 reused, no rerun), APK
  build/install <= 5, device restart <= 1.
- Every run must be a real device; host identity must be the fixed serial
  `324753221196` (verified before use).

## 8. Public artifacts (allow-list exporter only)

`docs/results/qnn-nicopedia-htp-parity-policy-2026-08/`:
README.md, manifest.json, policy-candidates.csv, step-parity-summary.csv,
scale-analysis.csv, decision-margin-summary.csv, probability-parity.csv,
synthetic-fault-results.csv, policy-decision.csv, generation-comparison.csv,
limitations.csv.

Never published: prompt bytes, generated bytes, raw logits, fingerprints
(sha256 / fnv1a64), checkpoint hashes, device serials/endpoints, absolute
paths.

## 9. Reviewer (independent, read-only; mandatory for adoption)

Checks (with a written finding per item):
1. No after-the-fact tuning (git history of this file + decision log).
2. Gauge argument and metric formulas are correct.
3. Decision-margin theorem is correct and the AR margin change is sound.
4. Fault battery covers the legacy and new failure modes, verdicts correct.
5. Thresholds are fixed and rationales match the text.
6. Regression gates untouched (continues to protect FFN372 +Inf etc.).
7. Evidence complete; step1000 claims match exact values.
8. Generation claims are not overstated (valid-UTF8 fractions etc. as
   measured, no NPU-only claims).
9. At least one residual risk is stated.
If no such review, the result is "Reviewer未実施" (incomplete).

## 10. Final state decision

- Criteria 1..10 all hold: adopt F as ACTIVE parity gate (legacy L remains
  computed for reporting). Run step1000 HTP generation (greedy/sample) and
  compare vs step320 + CPU.
- Otherwise: keep L as blocking gate; step1000 generation stays BLOCKED with
  a quantified, exact reason (this is the allowed outcome per the task).

## Appendix A. Measured outcome (2026-08-08, source commit 0dc9116)

Runs: device serial 324753221196 (NX741J SM8850, QAIRT
2.48.40.260702 / build 2.48.40.260702151143), greedy fixed-seed generation
at step320 and step1000 (3 runs total: step320, step1000, step1000-run2),
budgets 4 parity runs / 2 QNN builds / 2 installs as fixed. Host fault
battery regenerated from `host_tests/nicopedia_parity_policy_test.cpp`
(15 synthetic rows), stage-group parity:
`G2 stage0 none G3 stage0 half` (`fnv1a64` fingerprints 0x71314261 /
0x60004e5c), model on subsystem root.

Results (per-prefix fields in `build/reports/nicopedia-htp-generation/`):

- step320 greedy 1/20 candidate failures: `parity_11` centered
  RMS = 6.19e-3 > 5e-3 → candidate shape FAIL (centered max 1.43e-2 ok);
  legacy additionally fails `parity_18` raw = 2.297e-2 >= 2e-2, which
  candidate F passes (centered 3.24e-3). All other prefixes pass
  (candidate full = true at 19/20; `parity_gate_candidate=false`).
- step1000 greedy 3/20 candidate failures: `parity_8` (centered RMS
  5.67e-3 > 5e-3), `parity_13` (prob L1 2.98e-2 > 2e-2, centered max
  4.34e-2, legacy raw 4.28e-2 — broken row), `parity_14` (centered RMS
  5.40e-3 > 5e-3). Legacy additionally fails `parity_0` raw = 3.080e-2
  >= 2e-2 (gauge-like shift: centered 9.76e-3, cosine 0.999999 — exactly
  the documented gauge case, but the 2e-2 raw bound rejects it).
- AR: 8/8 steps argmax match at both steps. Legacy AR gate fails at
  step1000 because AR step-0 raw 3.080e-2 >= 2e-2
  (`ar_gate=false`, no divergence: `ar_divergence_blocked=false`);
  candidate AR gate passes all 8 (`ar_gate_candidate=true`) because each
  AR row passes candidate F (prob, shape, decision). step320: both
  `ar_gate=true` and `ar_gate_candidate=true`.
- Determinism: step1000-run2 gate fields byte-identical to step1000
  (criterion 3 PASS).
- Legacy gate semantics unchanged: parity and AR checks still use the
  fixed 2e-2 raw bound and 5e-3 prob bound per row; the 20-prefix list
  (L19 parity + byte-edge/truncation/short/ASCII coverage) is the only
  scope change. `gate_policy=legacy` in every device run; candidate
  fields are shadow (`candidate_full`, `ar_gate_candidate`).

Adoption evaluation (fixed rule, sections 9-10): criteria 1-2 FAIL
(step320: candidate rejected at parity_11 centered RMS 6.19e-3;
step1000: candidate rejected at parity_8 / 13 / 14 with the exact
metrics above), 3 PASS, 4-8 PASS, 9 reviewer finding 2026-08-08
(build/reviewer-findings.md) => **final decision: `KEEP_LEGACY_BLOCKED`**
— step1000 HTP generation stays BLOCKED (generation_gate=false in all
device runs) with the exact reason above; legacy stays the active gate
with its documented raw-bound shortfall (parity_0 legacy-fail at
3.080e-2 despite gauge-like agreement); no threshold or this document
changed after measurement (git history: appendix added as one block,
protocol text untouched).

Adoption evaluation (fixed rule, section 9/10): criteria 1-2 FAIL
(candidate F does not pass the step320/step1000 audits), 3 PASS,
4-8 PASS, 9 reviewer finding 2026-08-08 (build/reviewer-findings.md) =>
no threshold or this document
changed after measurement (git history: appendix added as one block,
protocol text untouched).