# V4.1 exact optimizer reference

**Status:** exact-gradient CPU reference gate complete  
**Branch:** `research/v41-optimizer-split-diagnostic`  
**Canonical parent:** `7a4640a977135debd9f9da5047c9a2e6ecec41cc`  
**Date:** 2026-09  
**Scope:** DataCursor-exact gradient replay + Head-wise Muon 1/8/32-step + embedding Adam/momentum/Sinkhorn 1/8/32-step. No 500-step training, no production optimizer change, no device use.

## stale `build/hvx-promotion` path

**cause:** 前回 diagnostic の探索中に、agent が存在しない `build/hvx-promotion` を glob した。  
正しくは `build/reports/hvx-promotion/`。

**impact:** 前回の数値への影響 **なし**。  
`scripts/v41_optimizer_split_diagnostic.py` は最初から `build/reports/hvx-promotion` を参照している。  
script / cleanup path の問題ではない。空 directory は作っていない。

## method

Formal HVX Muon NPRTCKPTV4 checkpoint から DataCursor を復元し、
`train_pilot.bin` の canonical order（seed `20260806`, batchSize `8`）で
**同一 batch sequence** の CPU `forwardBackwardGeneralized` により exact gradient を再生成した。

```text
checkpoint DataCursor.recordIndex = step * 8
next-step selection = order[recordIndex .. recordIndex+7]
gradient = mean over those 8 records
control / candidate にはまったく同じ gradient を入力
```

Muon math は現行 formal Original Muon と同一（partition のみ変更）:

```text
momentum' = 0.95 * momentum + 0.05 * grad
nesterov  = 0.05 * grad + 0.95 * momentum'
orthogonal = zeropowerNewtonSchulzFp32(nesterov, rows, cols, NS5)
scale = sqrt(max(1, fanOut / fanIn))
param -= muon_lr * scale * orthogonal
coefficients (3.4445, -4.7750, 2.0315), epsilon 1e-7
```

### exact gradient reproducibility

| step | record_index | mean_loss | replay cosine | replay relL2 | finite |
| --- | --- | --- | --- | --- | --- |
| 500 | 4000 | 5.24824 | 1.0 | 0 | true |
| 2000 | 16000 | 4.66161 | 1.0 | 0 | true |
| 8000 | 64000 | 4.01778 | 1.0 | 0 | true |

dataset identity: `fnv1a64:0c7b2826f5f26fea`  
order seed: `20260806`  
tokenizer: `byte_bpe` / `sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`

## Head-wise Muon exact 1-step

同一 actual gradient に対する full-matrix `[64,64]` vs semantic head-wise `[64,32] × 2`。

| checkpoint | arm | median cosine(full,split) | min cosine | median relL2 | max relL2 |
| --- | --- | --- | --- | --- | --- |
| step500 | Wq/Wk | 0.751 | 0.739 | 0.765 | 0.831 |
| step2000 | Wq/Wk | 0.777 | 0.751 | 0.696 | 0.745 |
| step8000 | Wq/Wk | 0.800 | 0.762 | 0.639 | 0.712 |

- 前回 momentum-proxy（cosine 0.741–0.805 / relL2 0.632–0.787）と**同オーダーで一致**
- head0/head1 update norm ratio ≈ 1.00–1.04
- head0/head1 angular step ratio ≈ 0.99–1.03
- update/weight ratio は full/split で近い（step size ではなく方向の差）

## Head-wise Muon 1/8/32-step（step2000 origin）

同一 DataCursor sequence。full と split を独立に stateful に進める。

| traj_step | arm | loss | param_div (relL2 vs start) | update cosine (to full traj) | non-finite |
| --- | --- | --- | --- | --- | --- |
| 1 | full | 4.66161 | 0.00452 | 1.0 | false |
| 8 | full | 4.50276 | 0.02899 | 1.0 | false |
| 32 | full | 4.39171 | 0.09269 | 1.0 | false |
| 1 | wqwk_headwise | 4.66161 | 0.00461 | 0.825 | false |
| 8 | wqwk_headwise | 4.50341 | 0.03094 | 0.100 | false |
| 32 | wqwk_headwise | 4.39434 | 0.10094 | -0.076 | false |

- loss は full / split でほぼ同水準（短期 loss では優劣不明）
- parameter は 32 step で relL2 0.09–0.10 まで乖離
- update geometry は 8 step 以降 trajectory が分岐（cosine ≈ 0）
- 両 arm とも finite

## embedding / output optimizer exact

同一 actual gradient から B0 Aux Adam / B1 momentum-only / B2 momentum+Sinkhorn を分岐。

semantic vocab axis（SSOT）:

```text
token_embedding    [1024, 64]  vocab = axis 0
output_projection  [64, 1024]  vocab = axis 1
```

### 1-step（step2000）

| suffix | cos(Adam,mom) | cos(mom,Sink) | cos(Adam,Sink) | relL2(Adam,mom) | vocab CV mom | vocab CV Sink |
| --- | --- | --- | --- | --- | --- | --- |
| token_embedding | 0.395 | 0.389 | 0.164 | 0.999 | 2.37 | ~0 |
| output_projection | 0.363 | 0.504 | 0.227 | 0.999 | 1.56 | ~0 |

- Adam と momentum の update 方向は大きく異なる（cosine ≈ 0.36–0.40）
- Sinkhorn は vocab-axis CV をほぼ 0 へ下げる（balancing 自体は機能）
- ただし cold-start momentum の update RMS は Adam の約 1/700 で、
  naive Sinkhorn（scale 非保存）がその微小 update を大きく引き伸ばす

### 32-step（step2000 origin）

| traj_step | arm | loss | weight_norm (embed) | param_div | non-finite |
| --- | --- | --- | --- | --- | --- |
| 1 | B0_adam | 4.66161 | 39.71 | 0.0010 | false |
| 8 | B0_adam | 4.50362 | 39.71 | 0.0044 | false |
| 32 | B0_adam | 4.39558 | 39.72 | 0.0115 | false |
| 1 | B1_momentum | 4.66161 | 39.70 | ~0 | false |
| 8 | B1_momentum | 4.50282 | 39.70 | ~0 | false |
| 32 | B1_momentum | 4.39165 | 39.70 | 0.0001 | false |
| 1 | B2_sinkhorn | 4.66161 | 40.98 | 0.201 | false |
| 8 | B2_sinkhorn | **19.69** | 68.16 | 1.40 | false |
| 32 | B2_sinkhorn | **22.03** | 185.4 | 4.63 | false |

output_projection も同様に B2 は weight norm 33.6 → 555.8 へ爆発。

**解釈:**

- B1 momentum-only は finite で loss は B0 と同等だが、cold-start + Adam 同一 LR のため
  update が極端に小さく、Adam 代替としての比較になっていない
- B2 naive Sinkhorn は scale を保存せず、loss / weight norm が不安定
- 短期 loss だけで品質優劣は断定しない

## state bytes

| component | parameters | persistent | temporary |
| --- | --- | --- | --- |
| token_embedding Adam m+v | 65,536 | 524,288 (512 KiB) | 0 |
| output_projection Adam m+v | 65,536 | 524,288 (512 KiB) | 0 |
| pair Adam m+v total | 131,072 | 1,048,576 (1 MiB) | 0 |
| pair momentum-only | 131,072 | 524,288 (512 KiB) | 0 |
| pair momentum+Sinkhorn | 131,072 | 524,288 (512 KiB) | 524,288 (512 KiB) |

前回の「約 512 KiB 削減余地」を reference 設計でも確認。
Sinkhorn persistent state は momentum-only と同一。working copy が一時 512 KiB。

## 結論

### Head-wise Muon

exact gradient でも momentum-proxy と同程度の update 方向変化
（cosine 0.75–0.80）が確認でき、32-step は finite で parameter が乖離する。

→ **PROMOTE_WQWK_HEADWISE_MUON_500STEP**

次: Wq/Wk head-wise Original Muon の 500-step CPU/reference A/B。
Wq/Wk/Wv は exploratory のまま。

### embedding / output

- semantic vocab axis は明確、state 削減は実在
- しかし momentum-only は LR / state init 未調整で Adam と不公平
- Sinkhorn は scale 非保存の実装で 32-step 不安定

→ **HOLD_EMBEDDING_HEAD_OPTIMIZER_SPLIT**

次: scale-preserving Sinkhorn と LR-matched momentum-only を作り直してから
500-step を再判定する。production には入れない。

## artifacts

```text
docs/results/v41-optimizer-reference-2026-09/
  exact-gradient-replay.csv
  headwise-muon-1step.csv
  headwise-muon-trajectory.csv
  sinkhorn-1step.csv
  sinkhorn-trajectory.csv
  optimizer-state-bytes.csv
  run-manifest.json
  step500/ step2000/ step8000/   (per-checkpoint raw)
host_tests/v41_optimizer_exact_reference.cpp
scripts/run_v41_optimizer_exact_reference.ps1
```

## device

不使用。device lock 未取得・未解放。

## やらないこと

500-step training、production Head-wise Muon、HVX kernel、
Sinkhorn production optimizer、NS係数変更、Muown、LR sweep。
