# Nicopedia CPU/HTP divergence localization (L19 seed1)

Status: **MEASURED-2026-08-08** (protocol fixed before measurement; private
protocol/state under `build/private-diagnostics/nicopedia-htp-divergence-localization/`).

Scope: Nicopedia L19 seed1 (V=256, T=32, D=16, FFN=32, H=2), checkpoints
`htp-seed1-l19-step320.ckpt` and `htp-seed1-l19-step1000.ckpt`, QAIRT
`2.48.40.260702` build `2.48.40.260702151143`, NX741J SM8850 HTP V81.
The parity gate is unchanged; this work localizes where the CPU/HTP tensor
difference first grows.

## 1. Instrumentation

A private diagnostic entry point (`QNN_HTP_NICOPEDIA_DIVERGENCE_LOCALIZATION`)
prepares the generalized training graph once per tap scope and compares every
fixed parity prefix at identical boundaries:

- `NONE`: untapped baseline (logits fingerprint control).
- `COARSE`: every layer output (`layer_<NNN>_output`, 19 boundaries) plus logits.
- `FINE`: one selected layer's forward intermediates (LN1 stages, Q/K/V,
  per-head attention probabilities, attention context, residual1, LN2 stages,
  FF1/ReLU/FF2, output), head probabilities from the established H>1 ABI taps.

The per-head score tensors are not tappable as layer-shared tensors in the H=2
graph (they are not connected to any node); the fine scope therefore uses the
per-head probability tensors, which are always APP_READ.

## 2. Instrumentation control

The untapped `NONE` run and every instrumented scope produce the identical
HTP logits fingerprint (canonical SHA-256, recorded privately) and the same
`parity_13` FAIL status; two COARSE runs are bit-exact on every boundary
metric. Instrumentation is therefore not an artifact and the tapped tensors
are usable as evidence.

## 3. Coarse scan (step1000, parity_13)

| boundary | diff RMS |
|---|---|
| layer 0 output | 7.1e-4 |
| layer 9 output | 3.3e-3 |
| layer 18 output | 7.1e-3 |
| logits | 5.4e-3 |

The error grows monotonically from layer 0 to layer 18 with no layer boundary
whose amplification `rel_b / rel_{b-1}` reaches 3 (protocol first-divergence
rule). Per the pre-fixed protocol fallback this is
**GRADUAL_ACCUMULATION** (max relative output error >= 2x relative input
error). The same profile shape appears on every prefix (parity_0/8/13/14);
parity_13 is the worst case only because its late-layer accumulation is
largest, not because of a prefix-specific mechanism.

## 4. Step320 vs step1000

The same boundary profile at step320 is 1.6–2.9x smaller; the layer-0 fine
profile shows the same tensor ordering. Weight-independent tensors (per-head
probabilities, LN inverse) keep a fixed small error across steps while
matmul-path tensors scale with the checkpoint weight magnitude. This is
magnitude-dependent rounding, not a step1000-specific new divergence.

## 5. Fine scan

At layer 0 the largest single amplification is LN1 (input ~4e-5 -> output
~4.4e-4), which is the expected LayerNorm inverse-standard-deviation scaling
(inv ~6.45; LN intermediates show no anomalous amplification and the inverse
error is small, rel ~3e-4). Per-head attention probabilities are the most
accurate tensors measured (diff RMS ~8.5e-5 at every layer and both steps).
Errors are element-wise (common-mode fraction ~0–0.1), matching the parity
policy observation that the parity_13 residual cannot be explained by a
common logit offset.

## 6. Conclusion

There is no single divergent op with an anomalous amplification: the CPU/HTP
difference is a ~1e-3 relative, element-wise rounding level spread across all
matmul/LayerNorm arithmetic, accumulating through the residual path over 19
blocks (GRADUAL_ACCUMULATION). It is consistent with HTP fp16-equivalent
kernel precision scaled by operand magnitude. The parity gate is unchanged;
precision mitigation is a candidate for a future milestone.

## 7. Reproducibility and controls

- Two COARSE step1000 runs: identical fingerprint, bit-identical boundary
  metrics.
- step320 control: same profile at 1.6–2.9x smaller magnitude.
- parity_8 / parity_14 (borderline) and parity_0 (good): same profile shape.
- All runs: fixed serial device identity, thermal recorded (0), no CPU
  fallback, no training.

## 8. Private data

Raw prompts, token windows, activations, hidden states, logits, checkpoints,
parameters, device serials and ADB endpoints stay under
`build/private-diagnostics/` and never enter public artifacts.
