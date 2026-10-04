# G1 1.5x full-8000（seed1、step 500–8000）

## 実験 identity

- parent: step 2000 までの継続実験（`g1-1p5x-long-2000-2026-09`、Control / G1）
- 本実験: 親の step2000 checkpoint から 8000 まで継続、LR 1.5x、`linear_decay` 4000→8000
- seed 1 のみ、Val/Dev 各先頭 256 chunk、byte-BPE V1024
- QAIRT `2.48.40.260702151143`、評価は HTP native（`evaluation_math_responsibility=HTP`）
- runner: `scripts/run_g1_1p5x_full8000.ps1`（modes: Plan / Run / Finish / Analyze）
- analyzer: `scripts/g1_1p5x_full8000_analyze.py`（balanced = (val bpb + dev bpb) / 2）
- Control: `attention_gate=none` / 758,528 params / `NPRTCKPTV4`
- G1: `attention_gate=headwise_g1_sigmoid` / 760,960 params / `NPRTCKPTV5`

## ファイル構成

- `control/`, `g1/`: 各 arm の host 側 evidence
  - `eval256-step<step>-htp.txt`: 一次評価レポート（本実験の新規 step は 2500 以降）
  - `gate-static-step<step>.txt`: G1 のみ、host 側 gate diagnostic（500, 750–8000 の各 step）
  - `seed1-l19-v1024-t32-d64-f128-steps8000-result.txt`: device 側 training report
  - `source-seed1-*-steps2000-result.txt`: 親実験の step2000 report の写し
  - `training-curve*.csv`, `learning-rate-telemetry.csv`, `run-manifest.json`
- 集計 CSV（`quality.csv`, `quality-paired.csv`, `phase-summary.csv`, `time-to-bpb.csv`,
  `training-telemetry.csv`, `runtime.csv`, `health.csv`, `quality-1p0-vs-1p5.csv`,
  `trajectory-summary.json`）は analyzer の生成物であり、一次結果ではない。
  一次データが必要な場合は `control/`, `g1/` 配下の `-htp.txt` を参照する。

## run health（一次レポートより）

- 両 arm とも `status=SUCCESS`、`completed_steps=8000`（追加 6000 step）
- `all_steps_finite=true`、`final_finite=true`、`output_tensors_finite=true`
- `qnn_return_code_success=true`（QNN return code の成功と tensor 有限性は別々に確認）
  - `api_trace_graph_execute_failure_count=0`（両 arm）
- `cpu_fallback=false`、`nan_detected=false`、`inf_detected=false`
- HVX: `hvx_rpc_failure_count=0`、`hvx_fallback_count=0`、`hvx_nonfinite_count=0`（両 arm）
- `focus_takeover_count=0`（両 arm）
- wall: Control `training_total_seconds=2405.799915`、
  G1 `training_total_seconds=2731.705738`

## 両 arm 共通点（一次レポートより）

- 全 32 eval ファイル（両 arm × 16 step: 500–2000 の親履歴 + 本実験 2500–8000 が
  `status=SUCCESS`、かつ `validation_nonfinite_chunks=0`、
  `development_nonfinite_chunks=0`
- 新規 eval 18 ファイル（両 arm × 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000）
  も同一条件を満たす。step 8000 を含め、`development_nonfinite_chunks=0` の
  例外ファイルは存在しない

## 主結果（`quality-paired.csv`）

- Δbalanced bpb（G1 − Control）は 16 評価点すべてで負（G1 が小さい＝良い）。
  ただし「全指標・全 split で一貫して改善」とは表現しない。理由は下記。
- phase summary（一次データからの再計算と一致）:
  - 500–2000: mean `-0.030318`
  - 2000–4000: mean `-0.019064`
  - 4000–8000: mean `-0.023363`
  - 500–8000: mean `-0.025469`
- step 8000: Control val bpb 2.210185 / dev bpb 2.515175 に対し
  G1 val bpb 2.201097 / dev bpb 2.490210（Δval −0.009087、Δdev −0.024965）
- `time-to-bpb.csv`: target 2.50 に G1 は step 4500、Control は step 5000 で到達。
  target 2.35 は G1 が step 8000 で到達、Control は未達（`>8000`、CENSORED）

## 解釈上の留保（中心）

balanced bpb は `(val bpb + dev bpb) / 2` の単純平均である。step 1750 と step 3000 では、
この平均の改善は validation 側の改善だけで成立しており、development NLL は
Control より悪化している。同じ 2 checkpoint では val / dev 両方の top-1 も
G1 が Control を下回っている。したがって、以下を事実として記録する。

- G1 が balanced bpb で全評価点において Control を上回った（Δ < 0 が 16/16）
- しかし「全指標・全 split で一貫して品質を改善した」とは表現しない
- NLL と top-1 の乖離は事実として記録するが、「calibration が改善した」とは
  断定しない。calibration の主張には ECE / Brier score 等の追加測定が必要
- dev NLL の差は絶対値で約 +0.019（step 1750）/ +0.026（step 3000）であり、
  dev NLL 約 4.6 に対して約 0.4–0.6%
- seed 1 単独の実験であり、seed noise / split-specific interaction /
  一時的な trajectory 差のいずれによるものかは未同定

### step 1750（一次ファイルから直接読んだ値）

- Control val NLL = `4.604696165`
  （`docs/results/g1-1p5x-long-2000-2026-09/control/eval256-step1750-htp.txt:22`）
- G1 val NLL = `4.573807483`
  （`docs/results/g1-1p5x-long-2000-2026-09/g1/eval256-step1750-htp.txt:22`）
- Δval NLL = −0.030889
- Control dev NLL = `4.645030287`（同 control ファイル `:34`）
- G1 dev NLL = `4.663810426`（同 g1 ファイル `:34`）
- Δdev NLL = +0.018780
- Control val top-1 = `0.1340332031` → 表記上 0.1340（同 control ファイル `:24`）
- G1 val top-1 = `0.1247558594` → 表記上 0.1248（同 g1 ファイル `:24`）
- Control dev top-1 = `0.1052246094` → 表記上 0.1052（同 control ファイル `:36`）
- G1 dev top-1 = `0.09875488281` → 表記上 0.0988（同 g1 ファイル `:36`）
- 両ファイルとも `status=SUCCESS`、validation / development の
  `nonfinite_chunks=0`、checkpoint finite、`qnn_return_code_success=true`、
  `output_tensors_finite=true`、`cpu_fallback=false`、graph execute 512/512 成功。

### step 3000（一次ファイルから直接読んだ値）

- Control val NLL = `4.412555994`（`control/eval256-step3000-htp.txt:22`）
- G1 val NLL = `4.358896517`（`g1/eval256-step3000-htp.txt:22`）
- Δval NLL = −0.053659
- Control dev NLL = `4.528141588`（同 control ファイル `:34`）
- G1 dev NLL = `4.553646845`（同 g1 ファイル `:34`）
- Δdev NLL = +0.025505
- Control val top-1 = `0.1597900391` → 表記上 0.1598（同 control ファイル `:24`）
- G1 val top-1 = `0.154296875` → 表記上 0.1543（同 g1 ファイル `:24`）
- Control dev top-1 = `0.1274414062` → 表記上 0.1274（同 control ファイル `:36`）
- G1 dev top-1 = `0.1254882812` → 表記上 0.1255（同 g1 ファイル `:36`）
- 両ファイルとも step 1750 と同じ run health を満たす
  （`status=SUCCESS`、両 split の `nonfinite_chunks=0`、他 health 項目は上記参照）。

## 追跡可能性

- Δval NLL / Δdev NLL は一次ファイルの NLL 値を引き算して再計算した
  （1750: Δval −0.030889 / Δdev +0.018780、3000: Δval −0.053659 / Δdev +0.025505）
- top-1 の 4 桁表記は一次ファイルの実測値を小数第 4 位で丸めたもの
  （0.1340332031 → 0.1340、0.1247558594 → 0.1248、
  0.1052246094 → 0.1052、0.09875488281 → 0.0988、
  0.1597900391 → 0.1598、0.154296875 → 0.1543、
  0.1274414062 → 0.1274、0.1254882812 → 0.1255）
- cached hash: val `fnv1a64:cd9b85f1256b6621`、dev `fnv1a64:f7b72398d954193a`。
  4 ファイルすべて同一（異なる split・cache の取り違えなし）
- 本メモは公開 exporter の対象に追加していない。公開集計が必要な場合は
  対応する exporter を通す（本 README は内部 evidence の解釈メモである）
