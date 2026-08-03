# First-error and margin decomposition, August 2026

This bundle decomposes why the L19 (T8/D16/FFN32/H2) final checkpoint trades a
higher autoregressive rollout NLL for higher token and sequence exact counts
relative to the validation-selected checkpoint. It is a host-only CPU replay
over the fixed AR_DEVELOPMENT_V3 partition; no device or HTP run contributed
data.

Each development case is replayed under the selected checkpoint and under the
320-step final checkpoint. For every token we record the midrank of the
target, its expected minus top-1 logit margin, top-1 minus top-2 margin,
softmax entropy, and token NLL, together with the first autoregressive error
position and its case class. Tokens are grouped into four buckets by whether
each checkpoint is exact: BOTH_CORRECT, SELECTED_CORRECT_FINAL_WRONG,
SELECTED_WRONG_FINAL_CORRECT (SWFC), and BOTH_WRONG. Cross-prefix conditions D
and E (one checkpoint run over the other checkpoint's free-running prefix)
corroborate whether a diverged prefix, rather than local ranking, explains an
exact gain.

Pooled over L19 seeds 1, 2, and 4 (432 development tokens per checkpoint
pair), the hard negative margin concentrates on the tokens the final
checkpoint corrects:

| Seed (selected step) | Selected exact | Final exact | SWFC tokens | SWFC median margin |
| --- | ---: | ---: | ---: | ---: |
| 1 (16) | 14/144 (0/24 seq) | 30/144 (2/24 seq) | 25 | -1.493876 |
| 2 (4) | 20/144 (0/24 seq) | 63/144 (6/24 seq) | 59 | -0.339281 |
| 4 (12) | 22/144 (0/24 seq) | 46/144 (6/24 seq) | 40 | -1.067867 |
| L18 control (4) | 18/144 (0/24 seq) | 65/144 (8/24 seq) | 57 | -0.235616 |

The final checkpoint's distribution is sharper and less calibrated: mean
entropy drops while the expected margin turns strongly negative (see
margin-rank-summary.csv and 	oken-buckets.csv). The L18 depth control
retains at-least-L19 final exact
(CONTROL_FINAL_EXACT_AT_LEAST_L19), so the pattern is not depth-specific
beyond the L19 ceiling.

The precommitted hypothesis decision is in hypothesis-decision.csv:

- H1 (easy-token NLL dominance) is not supported: no easy token at the 0.5
  expected-probability threshold, and the both-correct bucket does not carry
  the pooled NLL movement.
- H2 (critical-token margin loss) is supported: 124 pooled SWFC tokens
  (threshold 3) with a negative pooled median margin (below the 0 threshold).
- H3 (prefix-drift amplification) is not supported: no corroborated
  prefix-drift-amplified case (0 vs threshold 2); common-prefix-attribution.csv
  attributes each case individually.

The conclusion is CRITICAL_TOKEN_MARGIN_LOSS. The four auxiliary-objective
candidates in 
ext-objective-candidates.csv are NOT_RUN_CANDIDATE; they
are recorded for a separate, later investigation.

All exact counts, steps, and rollout NLL values reproduce the canonical
autoregressive bundle (docs/results/qnn-htp-autoregressive-validation-2026-08)
and are re-checked by the exporter against r-development-metrics.csv and
r-validation-metrics.csv before publishing. The public files deliberately
exclude model-state payloads, package material, device identifiers, endpoint
data, paths, and log streams.