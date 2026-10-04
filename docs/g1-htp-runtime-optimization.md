# G1 HTP runtime optimization — negative result（NO_SAFE_GAIN）

**結論: `NO_SAFE_GAIN`。** G1 の HTP runtime overhead を安全に削減できる方法を、
実機測定の上では見つけられなかった。**コード変更は revert 済みで、旧 G1 baseline が
そのまま現在の research baseline である。**

本書はその測定と放棄した、および見送った案の記録である。

---

## 1. 目的と判定基準

G1 (`headwise_g1_sigmoid`) は research baseline として昇格済みである。品質・意味論・
checkpoint 互換性は確定済みなので、本書のスコープは **runtime overhead だけ** である。

- Control (`attention_gate=none`) に対する G1 の HTP execute overhead を削減する
- 既存記録値: median **+9.95%**（[g1-1p5x-baseline-promotion-cost.md](g1-1p5x-baseline-promotion-cost.md)）
- 目標: +5% 以下を強く成功、+3% 以下を非常に良いとする

ただし **数値だけを追って品質や安全性を犠牲にしない**、という条件が最優先である。

## 2. Root bottleneck の特定

G1 が Control に対して graph に追加する node / tensor を、層ごとに分解した
（`numHeads=2`, `numLayers=19`, `t=32`, `d=64` — host contract の実測設定）。

### Node delta: **+399 nodes**（19 層 × 21 node/層）

| 構成 | node/層 | 計 | 数学的に必須か |
|---|---:|---:|---|
| gate projection MatMul（`LN1 × Wg`） | 1 | 19 | **必須** |
| gate Sigmoid | 1 | 19 | **必須** |
| per-head `gate_select` MatMul（`[T,H] × one-hot`） | 2 | 38 | layout のみ |
| per-head `gate_multiply` | 2 | 38 | **必須** |
| per-head `dcontext_gate` | 2 | 38 | **必須** |
| per-head `dgate_product` | 2 | 38 | **必須** |
| per-head `dgate_reduce` | 2 | 38 | **必須** |
| per-head `dgate_scatter` MatMul（`[T,1] × one-hotᵀ`） | 2 | 38 | layout のみ |
| `dG` 累加 add chain | 1 | 19 | layout のみ |
| gate derivative（`G*(1-G)`） | 1 | 19 | **必須** |
| gate `dlogits` | 1 | 19 | **必須** |
| gate `dweight` | 1 | 19 | **必須** |
| gate `dln1` | 1 | 19 | **必須** |
| `dln1_gate_add` | 1 | 19 | **必須** |
| **合計** | **21** | **399** | うち layout のみ = **95（24%）** |

forward: `2 + 2h = 6`、backward: `5h + 5 = 15`（estimator の式と一致）。

### Tensor delta: **+476 tensors**（19 層 × 25 + 1）

per-head に 8 個（gate selector、gate、gated context、dGate scatter、dGate product、
dGate、dG accumulator、および 2）、layer shared に 9 個、global に 1 個。

### 判定

**G1 overhead の約 76% は数学的に必須**であり、**削除可能な 24%（layout 処理）だけが
最適化の標的**になる。これが最初に実施した構造分解であり、以下の判断基準

> 「+399 nodes を機械的に減らす」ことが目的ではない

を裏づける。問題は、**この 24% が HTP 上で実際にコストになっているか**である。

## 3. 試行した案: per-head MatMul selector/scatter → Split / Concat

### 設計

`[T,H]` の gate 行列から head ごとの `[T,1]` 列を取り出す処理を、
one-hot `[H,H]` との MatMul（H 個）ではなく **QNN `Split`（axis=1）1 node** に置き換えた。
backward の `dG` 再配置は、`one-hot` MatMul + ゼロ埋め add chain（H + H−1 node）に
代えて **QNN `Concat`（axis=1）1 node** にした。

- 数学的演算（sigmoid、gate multiply、dgate product、reduce、derivative、dW、dln1）は**一切変更していない**
- forward/backward の計算順序は不変
- QAIRT 2.48.40.260702 の `Split` は `axis` + `split_index`（`[1, H-1]` の開始 index 列）で
  `H` 個の slice に分割する。実装はこれに一致
- fail-closed として、shape validator に Split / Concat の宣言 shape 検証を追加

### 結果（node / tensor）

| metric | before | after | delta |
|---|---:|---:|---:|
| G1 node delta vs Control | +399 | **+342** | −57 |
| G1 tensor delta vs Control | +476 | **+382** | −94 |
| forward node/層 | 6 | 5 | −1 |
| backward node/層 | 15 | 13 | −2 |
| tensor slot/層 | 25 | 20 | −5 |

Host contract（`headwise_g1_gate_test`）の実測値であり、推測ではない。
実 device でも estimator と実 graph の node count が一致することを
fail-closed check（`emitted node count mismatch`）が保証している。

### 実測 runtime（実機、同一条件）

device: 物理端末 1 台、QAIRT **2.48.40.260702**（Build ID `2.48.40.260702151143`）、
Hexagon SDK 6.6.0.0、Muon backend HVX。
条件: seed 2 / 19 layers / 400 steps / tokens 32 / vocab 1024 / d 64 / ffn 128 /
`linear_decay(4000→8000)` / Muon lr 0.0075 / aux Adam lr 0.0033。
各 arm を 5 repetition、Control と G1 を交互実行。絶対値は thermal・battery 条件に
強く依存するため、**同じ repetition 内の Control に対する paired ratio** を比較した。

**HTP forward+backward execute / update の G1 vs Control（正 = G1 が遅い）:**

| repetition | baseline（旧 G1） | Split/Concat 版 |
|---|---:|---:|
| 1 | +5.32% | +9.20% |
| 2 | +5.48% | +10.11% |
| 3 | +9.23% | +10.48% |
| 4 | +9.06% | +9.48% |
| 5 | +10.64% | +38.80%（外れ値） |
| **median** | **+9.06%** | **+10.11%** |

5 repetition 中央値（run total / steps）:

| metric | baseline Control | baseline G1 | optimized Control | optimized G1 |
|---|---:|---:|---:|---:|
| graphExecute ms/update | 27.31 | 29.49（**+7.98%**） | 26.31 | 28.97（**+10.11%**） |
| training step ms | 404.90 | 412.96（+1.99%） | 375.43 | 392.14（+4.45%） |
| parameter transfer ms/update | 0.457 | 0.569（+24.64%） | 0.420 | 0.508（+20.95%） |
| optimizer update wall ms/update | 92.05 | 93.96（+2.08%） | 88.65 | 92.10（+3.89%） |

baseline 側の +7.98% / +9.06% は既存記録の +9.95% と整合しており、測定系は妥当である。

### 判定

**改善しない。** 目標の +5% 以下には到達せず、baseline に対して中立〜やや悪化である。

## 4. なぜ node を 57 減らしても時間にならないのか

**QNN node count は HTP runtime cost の proxy にならない。** これが本書の中心的な知見である。

QNN `Split`（axis=1）は 1 つの入力から H 個の出力 tensor を生成するため、
QNN/HTP 内部で H 個の slice を materialize する。元の per-head `MatMul`
（`[T,H] × one-hot [H,H]` → `[T,1]`）も結局は同じ量のデータを読む。
つまり **layout 処理の総データ移動量は変わらず、node 数だけが減る**。
node を潰しても時間が短縮できず、HTP の実行コストは
node 数ではなく **実際に DSP 上で read/write される tensor データ量とグラフ構造** に支配される。

逆に言えば、**残りの 76%（数学的に必須）は、短縮するには数学的な意味を変えるしかない**。
runtime optimization としてそれは許されないため、この方向には進めない。

## 5. rejected / 未実施の案

| 案 | 判定 | 理由 |
|---|---|---|
| per-head MatMul selector/scatter → `Split`/`Concat` | **revert 済み** | bitwise parity は保ったが実測で中立〜悪化（§3） |
| `gate_multiply` と `dcontext_gate` の融合 | 不採用 | `[T,1] × [T,dh]` は QNN element-wise broadcast で既に 1 node。融合余地なし |
| `gradient_accumulation` の host 側高速化 | スコープ外 | 69–83 ms/update と支配的だが、Control にも共通する G1 非固有コスト。G1 baseline の判断を左右しない（§6） |
| gate projection の低ランク化 / 共有 | **不可** | G1 の数学的意味論の変更和规范に抵触。別 research candidate であり、本 optimization のスコープ外 |
| 構造を変えない graph 順序の並べ替え | 未試行 | QAIRT の compile optimizer に任せている領域。効果予測が困難で、+9.95% には届かない |

## 6. Remaining bottleneck（残存する支配的コスト）

`fwd+bwd` は 27–29 ms/update だが、以下がこれを数倍上回る。
**いずれも G1 固有ではなく、Control でも同程度に発生するため、G1 baseline の
quality 判断には影響しない。**

| component | per-update | 備考 |
|---|---:|---|
| gradient_accumulation | 69–83 ms | Muon gradient の CPU 側 reduction。全 host cost の支配項 |
| optimizer_update_wall | 88–94 ms | Muon HVX kernel + pack/unpack + aux Adam |
| fwd+bwd（HTP） | 26–29 ms | G1/Control 差は +8〜+10% が全て |
| parameter_transfer | 0.42–0.57 ms | G1 が Control 比 +21〜+25%（gate 分の追加 parameter があるため） |

つまり **total device wall の大半は optimizer / gradient 処理側にあり、
fwd+bwd の G1 overhead は全体。研究 throughput を上げるなら optimizer 側の
最適化が次の iteration の proper な対象であり、G1 固有ではない。**

## 7. Parity / compatibility（本試行で確認した項目）

Split/Concat 版は、**以下を完全に満たしていた**。つまり安全性そのものは問題なかった。

- **bitwise parity**: 400 steps 完走後の `final_parameter_hash` が baseline と完全一致。
  G1 `fnv1a64:cf3453c27bdd9608`、Control `fnv1a64:098cc1d9a73fe890`（baseline / Split-Concat 版の両方で同一）。
  200 steps でも G1 `fnv1a64:fb7c2b402ffa71c4`、Control `fnv1a64:e65ecfe26c2441b7` で一致
- **determinism**: 5 repetition すべてで同一 hash（同一 device・同一 seed）
- **first/last loss**: baseline と完全一致（400 steps: G1 `7.324726` → `5.425243`）
- **parameter identity**: `parameter_count=760960`、Muon matrix 114、Muon parameter 622,592 すべて不変
- **checkpoint compatibility**: NPRTCKPTV5 の size（6,652,678 bytes）と identity は不変。
  `step100` / `step200` の checkpoint SHA256 も baseline と一致
- **health**: 全 10 run（5 rep × 2 arm）で `qnn_failures=0`、`all_steps_finite=true`。
  6031 incident の再発・fallback・nonfinite は 0 件
- **fail-closed**: 実 device で `emitted node count mismatch` 及其它 fail-closed は
  一度も trigger していない

**これを踏まえても採用しない。** parity が完璧であっても runtime 改善がなければ、
コードを変える理由にならない（`NO_SAFE_GAIN` の定義）。

## 8. Baseline status

### `NO_SAFE_GAIN`

- 旧 G1 baseline（`headwise_g1_sigmoid`）を**そのまま維持**する
- Split/Concat 版は revert 済みで、repository にコード変更は残っていない
- G1 の quality evidence・parameter identity・checkpoint compatibility は一切影響を受けていない
- 既存の multiseed quality study を再実施する必要はない（意味論が無傷）

## 9. 検証ログ

本試行の evidence は build 配下（private）にあり、commit しない。

- baseline 測定: `build/reports/g1-runtime/b400-*`（5 rep × 2 arm、400 steps）
- Split/Concat 版測定: `build/reports/g1-runtime/a400-*`（5 rep × 2 arm、400 steps）
- 短い sanity: `build/reports/g1-runtime/{before,after}-*`（200 steps）
- Host / Fast: `build/reports/g1-runtime/{host,fast}-reverted.log`（revert 後 PASS）