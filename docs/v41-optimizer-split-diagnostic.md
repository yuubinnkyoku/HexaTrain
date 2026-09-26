# V4.1 optimizer split diagnostic

**Status:** diagnostic complete on existing HVX Muon quality checkpoints  
**Branch:** `research/v41-optimizer-split-diagnostic`  
**Canonical parent:** `7a4640a977135debd9f9da5047c9a2e6ecec41cc` (origin/main)  
**Date:** 2026-09  
**Scope:** checkpoint-only head diagnostic + offline 1-step full-vs-split Muon replay + embedding/output-projection optimizer audit. No production optimizer change, no device training, no QNN graph change.

## 目的

DeepSeek-V4.1 系の optimizer split（Wq/Wk head-wise Muon / Sinkhorn-balanced embedding update）を
**実装する前に**、現行 HexaTrain の checkpoint / gradient / optimizer state から
「どの parameter group を分離する価値があるか」を定量化する。

本タスクは診断のみである。Head-wise Muon 本体、Sinkhorn optimizer 本体、
production training path への導入は行わない。

## 使用 checkpoint

Formal HVX Original-Muon quality checkpoints（NPRTCKPTV4）を使用した。
新規 training は起動していない。final split は未使用。

| seed | step | source |
| --- | --- | --- |
| 1 | 500 | `build/reports/hvx-promotion/quality-hvx-step1000/` |
| 1 | 2000 | `build/reports/hvx-promotion/quality-hvx-seed1-step8000/` |
| 1 | 8000 | `build/reports/hvx-promotion/quality-hvx-seed1-step8000/` |
| 2 | 500 / 2000 / 8000 | `build/reports/hvx-promotion/quality-hvx-seed2-step8000/` |

- format: NPRTCKPTV4, optimizer identity `muon_aux_adam`
- model: V1024 / T32 / D64 / FFN128 / L19 / H2
- Muon: `lr=0.005`, momentum `0.95`, Nesterov `true`, NS5
- registry: 192 entries = 114 Muon matrices + 78 Aux Adam tensors
- Aux Adam parameter count: 135,936（`token_embedding` 65,536 + `output_projection` 65,536 + norms 4,864）
  — current main でも `65,536 + 65,536 = 131,072` は成立

checkpoint inventory: `docs/results/v41-optimizer-split-diagnostic-2026-09/checkpoint-inventory.csv`

## SSOT orientation

parameter shape / orientation / optimizer role は
`metadata/transformer_parameter_metadata.json`
（生成元 `app/src/main/cpp/transformer_parameter_metadata.h`）に従う。

| suffix | role | shape | fan_out | fan_in | head split |
| --- | --- | --- | --- | --- | --- |
| `wq` / `wk` / `wv` | MUON | `[MODEL, MODEL]` = `[64, 64]` | axis 1 | axis 0 | fan_out を 2 分割 → `[64, 32] × 2` |
| `token_embedding` | AUX_ADAM | `[VOCABULARY, MODEL]` = `[1024, 64]` | −1 | −1 | — |
| `output_projection` | AUX_ADAM | `[MODEL, VOCABULARY]` = `[64, 1024]` | −1 | −1 | — |

- storage は `[input, output]`。fan_out_axis=1 なので semantic output neuron は **column**。
- head 分割は **fan_out 軸（column）** を行う。name substring や storage row/column の決め打ちはしていない。
- `output_projection` は保存 shape `[64, 1024]` であり semantic vocabulary axis は **axis 1**。
  `token_embedding` は `[1024, 64]` で vocabulary axis は **axis 0**。両者は保存上の軸が異なる。
- production optimizer semantics は変更していない。

Muon 数式は現行 Original Muon と一致させる（factor は matrix partition のみ）:

```text
momentum' = 0.95 * momentum + 0.05 * grad
nesterov  = 0.05 * grad + 0.95 * momentum'     (Nesterov true)
orthogonal = zeropowerNewtonSchulzFp32(nesterov, rows, cols, NS5)
scale = sqrt(max(1, fanOut / fanIn))
param -= muon_lr * scale * orthogonal
```

NS coefficients `(3.4445, -4.7750, 2.0315)`, epsilon `1e-7` は変更しない。
NS8 / Polar Express / Muown は混ぜない。

## checkpoint-only 結果

raw table: `headwise-muon-checkpoint.csv`（layer × parameter × head）  
aggregate: `headwise-muon-checkpoint-aggregate.csv`

### weight / momentum head imbalance

H2 の Wq/Wk/Wv を fan_out で 2 分割したときの head0/head1 RMS ratio
（`h1/h0`、1.0 が完全均衡）:

| seed | step | suffix | median | p90 | max |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | wq | 1.006 | 1.047 | 1.073 |
| 1 | 2000 | wq | 0.994 | 1.067 | 1.117 |
| 1 | 8000 | wq | 0.972 | 1.024 | 1.163 |
| 1 | 500 | wk | 1.007 | 1.034 | 1.060 |
| 1 | 2000 | wk | 0.984 | 1.062 | 1.104 |
| 1 | 8000 | wk | 0.977 | 1.029 | 1.142 |
| 2 | 500–8000 | wq/wk/wv | 0.978–1.020 | 1.029–1.061 | 1.043–1.139 |

- median ratio はほぼ 1.0（head 間 scale 差は小さい）
- p90 で 1.02–1.07、max で 1.04–1.16 まで乖離する layer はある
- layer consistency は seed 両方で同様。trajectory 上も極端な非対称成長はない
- momentum RMS ratio も同程度に均衡

**head 間 scale 差が小さいこと自体は Head-wise Muon の棄却条件ではない。**
Muon は preconditioner（NS orthogonalization）を分けるだけで更新方向が変わるため、
1-step replay まで実施した。

### checkpoint displacement

consecutive checkpoint 間の weight displacement は
`headwise-muon-displacement.csv` に保存。250–6000 step 間隔の累積であり、
1-step の substitute ではない。head 間で angular displacement に大きな系統差はない。

## 1-step replay 結果

`headwise-muon-replay.csv` / `headwise-muon-replay-aggregate.csv`

### 方法

Forward/Backward は分岐させず、**同一の入力行列**を full-matrix Muon と head-wise Muon の
両方へ入れ、optimizer update 計算だけを offline で分岐させた。

- Control: full-matrix Muon `Wq/Wk [64,64]`
- Candidate A（V4.1-faithful）: head-wise `Wq/Wk [64,32] × 2`
- Candidate B（exploratory）: `Wq/Wk/Wv [64,32] × 2`

入力行列は checkpoint 保存済み Muon **momentum**（momentum-proxy replay）。
DataCursor 同一 batch の真の gradient replay は device / private dataset が必要なため
本ラウンドでは未実施。momentum は EMA 済み gradient 幾何であり、
preconditioner 分割が更新方向へ与える影響の測定には十分である。
production training path は変更していない。

### full vs split update geometry（median across 19 layers × 2 heads）

| seed | step | arm | cosine(full, split) | rel. L2(full, split) | update norm ratio split/full |
| --- | --- | --- | --- | --- | --- |
| 1 | 500 | Wq/Wk | 0.748–0.750 | 0.759–0.763 | 1.119–1.131 |
| 1 | 2000 | Wq/Wk | 0.773–0.779 | 0.693–0.694 | 1.053–1.056 |
| 1 | 8000 | Wq/Wk | 0.797–0.803 | 0.633–0.637 | 1.016–1.022 |
| 2 | 500 | Wq/Wk | 0.741–0.745 | 0.769–0.787 | 1.119–1.142 |
| 2 | 2000 | Wq/Wk | 0.774–0.776 | 0.694–0.696 | 1.051–1.052 |
| 2 | 8000 | Wq/Wk | 0.804–0.805 | 0.632 | 1.018–1.035 |

- **cosine(full, split) は 0.74–0.80**。update 方向が実質的に変わる。
- **relative L2 は 0.63–0.79**。maxAbs 差も layer 全体で同程度のオーダー。
- 両 seed、early/mid/late で一貫して同様の差。
- update norm 自体の split/full 比は 1.02–1.14 程度で、大きさは近い。
  効いているのは方向（preconditioner）であり、単純な step size ではない。

### head 間 update geometry（Candidate A）

| metric | 典型値 |
| --- | --- |
| head0/head1 update norm ratio (h1/h0) | 0.98–1.01 |
| head0/head1 angular step ratio | 0.99–1.01 |

split 後も head 間 update norm / angular step は均衡を保つ。
「片方の head だけが暴走する」型の悪化は観測されない。

### Wq/Wk/Wv exploratory（Candidate B）

| seed | step | cosine(full, split) | rel. L2 |
| --- | --- | --- | --- |
| 1/2 | 500 | 0.769–0.775 | 0.719–0.733 |
| 1/2 | 2000 | 0.776–0.780 | 0.672–0.681 |
| 1/2 | 8000 | 0.788–0.789 | 0.665–0.676 |

Wq/Wk と同程度（やや小さい）の update 方向変化。
**V4.1-faithful = Wq/Wk**、**exploratory = Wq/Wk/Wv** として分離する。
第一候補は Wq/Wk のみ。

## embedding / output projection audit

`embedding-head-optimizer.csv`

| item | token_embedding | output_projection |
| --- | --- | --- |
| shape（保存） | `[1024, 64]` | `[64, 1024]` |
| semantic shape | VOCABULARY × MODEL | MODEL × VOCABULARY |
| semantic vocab axis | **0**（行 = token） | **1**（列 = token） |
| parameter count | 65,536 | 65,536 |
| Adam m bytes | 262,144 | 262,144 |
| Adam v bytes | 262,144 | 262,144 |
| total optimizer-state bytes | **524,288 (512 KiB)** | **524,288 (512 KiB)** |

- `65,536 + 65,536 = 131,072` は current main の checkpoint でも確認済み。
  Aux Adam 総数 135,936 の約 96.42%。
- m+v を momentum 1 本へ置き換えるだけで **約 512 KiB** の state 削減余地
  （2 行列合計 1 MiB の半分）。
- vocab 方向 norm 分布の CV は step 8000 で
  embedding 0.175 / output 0.261。疎・不安定な update ではない。
- update/weight ratio（approx `m/(sqrt(v)+eps)`）は late でも O(1)。

## Sinkhorn feasibility

`sinkhorn-feasibility.csv`（**解析用のみ。training parameter へは適用していない**）

| item | token_embedding | output_projection |
| --- | --- | --- |
| balance 対象 axis | vocab = rows (axis 0) | vocab = columns (axis 1) |
| update matrix size | 1024 × 64 | 64 × 1024 |
| update matrix bytes | 262,144 | 262,144 |
| 1 回の balancing working bytes（2×） | 524,288 | 524,288 |
| before row-norm CV | 0.71 | 0.038 |
| after row-norm CV | 0.26 | ~0 |
| before col-norm CV | 0.031 | 0.62 |
| after col-norm CV | ~0 | ~0 |
| cosine to original update | 0.843 | 0.849 |
| relative L2 | 0.883 | 0.592 |

- semantic vocabulary axis は SSOT から一意に定義できる。
- offline balancing は row/col norm CV を明確に下げ、update を O(1) に変える
  （training 未適用の feasibility のみ）。
- balancing 前提が不自然な形状ではない。working bytes も 512 KiB 級で実用的。

## 32-step reference

**not performed.**

1-step replay で full-vs-split の update 方向差が
cosine 0.74–0.80 / rel.L2 0.63–0.79 と明確であり、
promotion 条件の「split で update direction が実質的に変わる」をすでに満たす。
32-step は loss trajectory を伴うため DataCursor 同一 batch の host CPU reference
training loop が必要で、次の gate（500-step CPU/reference A/B）へ含める方が効率的。

## 結論

### Head-wise Muon

1. head 間 weight / momentum **scale 差は小さい**（median ratio ≈ 1.0）。
   これは棄却条件ではない。
2. それでも head 単位へ matrix を分けて NS を回すと、
   **update 方向が cosine 0.74–0.80 まで変わる**。preconditioner 分割の効果は実質的である。
3. head 間 update norm / angular step は split 後も均衡（ratio ≈ 1.0）。
   悪化方向の非対称は出ていない。
4. 両 seed・early/mid/late で一貫。

→ **PROMOTE_WQWK_HEADWISE_MUON_CPU_REFERENCE**

次タスクで Wq/Wk head-wise Muon の 500-step CPU/reference A/B へ進む。
Wq/Wk/Wv は exploratory のまま据え置き（Wq/Wk の結果を見てから判断）。

### Sinkhorn

1. `token_embedding` / `output_projection` で Aux Adam の 96.42% を占め、
   state 削減余地は約 512 KiB。
2. semantic vocabulary axis は SSOT から明確。
3. vocab 方向 norm 分布は偏るが疎・不安定ではない。
4. offline balancing は update geometry を明確に変える余地がある。

→ **PROMOTE_SINKHORN_CPU_REFERENCE**

次タスクで Nesterov momentum-only / momentum+Sinkhorn の CPU reference を
Aux Adam baseline と分離比較する。本体実装はまだ行わない。

## 次の gate

1. Wq/Wk head-wise Muon 500-step CPU/reference A/B
   （同一 DataCursor batch / 同一 seed / NS5 / 現行 coefficients）
2. （任意）DataCursor 同一 batch での真の gradient 1-step replay で momentum-proxy を置換
3. Sinkhorn CPU reference: Aux Adam baseline vs momentum-only vs momentum+Sinkhorn
4. Wq/Wk/Wv exploratory は上記 Wq/Wk の結果を見てから

## やらないこと（本タスク）

production Head-wise Muon、HVX kernel、500/2000-step training、
Muown、NS係数変更、Polar Express、effective batch / LR sweep、
Sinkhorn production optimizer、QNN graph 変更、packed QKV、CSA2、MQA。

## Artifacts

```text
docs/results/v41-optimizer-split-diagnostic-2026-09/
  checkpoint-inventory.csv
  headwise-muon-checkpoint.csv
  headwise-muon-checkpoint-summary.csv
  headwise-muon-checkpoint-aggregate.csv
  headwise-muon-displacement.csv
  headwise-muon-replay.csv
  headwise-muon-replay-aggregate.csv
  embedding-head-optimizer.csv
  sinkhorn-feasibility.csv
  diagnostic-run.json
scripts/v41_optimizer_split_diagnostic.py
```

## Device

不使用（checkpoint-only + offline replay）。device lock 未取得・未解放。
