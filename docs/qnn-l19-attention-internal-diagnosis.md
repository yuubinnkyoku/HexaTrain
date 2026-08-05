# L19 Attention Internal Diagnosis

## Overview

> **現在の状態: 本文中の旧因果判定はsuperseded。** 以下の
> `OUTPUT_PROJECTION` verdict、legacy probe score、cross-seed swap、
> projection contribution norm/cosineは現在の原因証拠として使用しない。
> 履歴と当時の判定規則を保持するため残している。現在の判定は
> [`qnn-l19-seed-instability-root-cause.md`](qnn-l19-seed-instability-root-cause.md)
> を正本とする。

This document describes the ATTENTION_INTERNAL_V1 host-only diagnosis that
decomposes the deep-layer attention-path linear readability loss of the L19
transformer into its internal stages: normalized input, Q/K/V projections,
attention scores, softmax weights, per-head context, head concatenation,
output projection, and residual addition.

The diagnosis is CPU-only and deterministic. No device, HTP, QNN graph, or
Android code is involved. The forward pass is the verbatim host copy
(`critical_margin_training_lib.h`), and an intervention forward
(`attention_internal_diagnosis_lib.h`) reproduces the same arithmetic with
per-head context buffers so head-level interventions can be applied before
the concat and output projection.

## Protocol

The protocol (ATTENTION_INTERNAL_V1) was fixed before any results were read.
See `build/private-diagnostics/attention-internal-goal/protocol.json` for the
full protocol, including target layers, probe conditions, intervention scopes,
budgets, and verdict rules.

### Target configurations

| Configuration | Seed | Layers | Max-drop block (0-based) |
|---|---|---|---|
| L19_SEED_1 | 1 | 19 | 9 |
| L19_SEED_2 | 2 | 19 | 7 |
| L19_SEED_4 | 4 | 19 | 12 |
| L18_SEED_2_CONTROL | 2 | 18 | 6 |

All configurations use FINAL_STEP (step 320) checkpoints.

### Tap kinds

Per target layer, the observer extracts:
- NORM1 (LN1 output, dim 16)
- Q, K, V projections (dim 16 each)
- CTX_H0, CTX_H1 (per-head context, dim 8 each)
- CTX_CONCAT (concatenated context, dim 16)
- ATT_UPDATE (ctx @ Wo, dim 16)
- AFTER_ATTN (residual after attention, dim 16)

### Interventions

- **Head zero**: zero one head's context, recompute forward from that layer
- **Head only**: zero all but one head
- **Cross-seed context swap**: replace a head's context with another model's
  (counterfactual, not natural inference)
- **Attention-weight / value separation**: swap Q/K (attention pattern) and/or
  V between models at the max-drop layer
- **Head pair**: zero both heads or keep both at 6 fixed layer slots
- **Free-running**: teacher-forced vs free-running on representative taps

## Results

**Historical verdict (superseded): OUTPUT_PROJECTION**

The evidence supports the hypothesis that the attention-path readability loss
is primarily caused by the output projection, not by a specific head, the
attention weights, or the value content.

### Key evidence

| Configuration | CTX_CONCAT dev TF | ATT_UPDATE dev TF | Projection drop |
|---|---|---|---|
| L19_SEED_1 | 24 | 6 | 18 |
| L19_SEED_2 | 37 | 24 | 13 |
| L19_SEED_4 | 57 | 47 | 10 |
| L18_SEED_2_CONTROL | 68 | 64 | 4 |

The projection drop (CTX_CONCAT probe minus ATT_UPDATE probe) is >= 10 tokens
for all three L19 seeds and only 4 for the L18 control. This meets the fixed
threshold (>= 5 tokens, >= 2 seeds consistent).

Head-zero ablation at the max-drop layer shows no single head removal improves
dev token exact by >= 5 tokens (S2: 1, 1; S4: -1, 0), ruling out a specific
harmful head.

Attention-weight/value separation at the max-drop layer shows limited recovery
(S2 combo D: 8, 1; S4 combo D: 0, -1), ruling out a pure weight-side or
value-side cause.

Cross-seed context swaps show near-zero recovery, consistent with model
coordinate mismatch rather than a single transferable fix.

> 後続監査による修正（2026-08, PROBE_OPTIMIZATION_AUDIT_V1）: 「出力射影が
> 原因」という解釈は、probe 学習の artifact として修正された。座標安定な
> canonical solver（PCA-whitened, L2 λ=1e-4, L-BFGS）では全最大低下層で
> CTX == ATT（dev token exact 16/16, 32/32, 52/52, 30/30）となり、投影差
> は legacy Adam の標準化不足によるもの（C1_OPTIMIZATION_INSUFFICIENCY）
> と結論された。低次元 head 介入の干渉実測そのものは維持される。

## Public bundle

> **追加修正（2026-08-05、seed-instability root-cause再監査）**
> TRAIN probe row生成に4行の契約不整合があり、旧probe絶対値は再生成まで除外する。
> また旧cross-seed context swapはdonor row 0の不正なreplacementを全DEV rowへ
> 再利用しており、row-wise identityを満たさなかった。旧swap recoveryは無効である。
> projection contributionのnorm/cosineも1 rowだけを保持していたため無効である。
> 実装とself-testは修正した。full-rank、疑似逆、probe transport、直接logit介入は
> 独立であり、Attention経路の因果判定は新しいbranch介入で行った。

The public results are in
`docs/results/qnn-l19-attention-internal-diagnosis-2026-08/`.

## Files

- `host_tests/attention_internal_diagnosis_lib.h` - library
- `host_tests/attention_internal_diagnosis.cpp` - runner + self-test
- `scripts/run_l19_attention_internal_diagnosis.ps1` - build/run script
- `scripts/export_public_qnn_l19_attention_internal_results.ps1` - exporter
