# L19 probe最適化監査（PROBE_OPTIMIZATION_AUDIT_V1）

## 結論

**CTX_CONCAT と ATT_UPDATE の dev-token-exact 差（24/6, 37/24, 57/47, 68/64）は、表現の情報損失ではなく、legacy Adam 線形 probe 学習の座標標準化不足による最適化 artifact である。**

座標安定な canonical solver（PCA whitening 特徴・whitened 空間のみへの L2 λ=1e-4・gauge固定・L-BFGS 判定）では、全最大低下層で CTX と ATT の probe が同一の収束点に到達する（dev token exact が CTX==ATT で 16/16, 32/32, 52/52, 30/30）。verdict は `C1_OPTIMIZATION_INSUFFICIENCY`（C1=4 層、C2=1、C3=3、C4=0、C5=0）。

## 背景

- `ATTENTION_INTERNAL_V1` は「出力射影が原因」と結論した。
- `OUTPUT_PROJECTION_AUDIT_V1` は出力射影が線形情報を保持していることを数学的に示し、「射影後の probe 学習・標準化の問題」を次候補とした。
- 本監査（PROBE_OPTIMIZATION_AUDIT_V1）は probe 学習側のうち、特に Adam の座標依存性（標準化不足がどの程度最適化を妨げるか）を事前登録 protocol で切り分ける。

前回の from-scratch probe（READOUT_PROBE_V1）は z-score 標準化のみで Adam により学習され、raw 座標の悪条件（TRAIN z-cov 条件数 κ≈4.2e7〜2.5e8）が学習率・勾配スケールに強く影響する。一方 PCA whitening は共分散を等方的にし、L-BFGS は座標に対して不変な二次近似を持つ。したがって「同一点で CTX と ATT が異なる性能」が真の表現差か、Adam の最適化 artifact かを切り分けられる。

## 方法（host-only CPU）

対象は前回と同一の4構成（FINAL step 320）。

| 構成 | 最大低下層 | 公開 anchor（CTX / ATT dev exact） |
|---|---|---|
| L19_SEED_1 | layer 9 | 24 / 6 |
| L19_SEED_2 | layer 7 | 37 / 24 |
| L19_SEED_4 | layer 12 | 57 / 47 |
| L18_SEED_2_CONTROL | layer 6 | 68 / 64 |

### 特徴・目的関数

- 特徴は各 tap の teacher-forced hidden state。z-score 標準化（floor 1e-6）後、TRAIN 共分散の PCA whitening（全 rank 保持、eigenvalue floor は Gram SVD の数値 rank 判定）を適用する。
- 目的は少数クラス softmax 線形 probe の交差エントロピー。CE は token ごとの素朴平均（per-token mean CE）で、raw logit 空間に bias 補正を伴う等価写像（`mapProbeZToWhitened`、AMENDMENT_3 の `b_u[c] = b_z[c] + μᵀw_z[c]`）を持つ。
- L2 正則化 λ=1e-4 は **whitened 空間の重みのみ** に適用する（raw 座標には適用しない）。
- gauge固定: 各特徴ごとのクラス平均ゼロ化 + bias 総和ゼロ（softmax 出力を変えない不変性を固定）。solver は初期化時と各 step で適用する。

### solver の役割（固定）

- **L-BFGS が学習の収束を保証し、C1 以外の判定基準の数値の正本である**。
  収束は勾配ゲート付き flat stop（‖grad‖≈2e-8、更新幅の相対変動で判定）。
- **GD は参照**。C1 の規則は規則文の通り `trainCE(CANONICAL_GD zero)` と
  CANONICAL_GD の最終勾配を用いる（GD の参照役割）。その他の判定基準は
  CANONICAL_LBFGS 解を正本とし、L-BFGS が収束しない層のみ GD を使いフラグする。
  観測データでは両参照で同一の判定になる（写像済み legacy trainCE≈1.4 vs
  canonical≈0.001、勾配比は両参照で ≥100倍）。
- GD は L2 下限 λ_min=1e-4 により線形収束率が制約され、10000 iters では
  gradTol 1e-8 に到達しない（到達可能精度は勾配 5.5e-5 程度）。そのため GD の
  収束 assertion は到達可能精度に基づく。
- 全 solver の CE は whitened 座標で比較する（1_mapped 行: legacy z-score 解を whitened に写像した値）。

### 検証項目（固定）

- `AMENDMENT_2` の cond5: transport 等価初期化（tap 自身の legacy z-score 解を `mapProbeZToWhitened` で写像）で、cross-solver・cross-init の一意性を確認する。
- `C1（最適化不足）`: 規則通り、写像済み legacy probe の trainCE と CANONICAL_GD の trainCE の差 ≥1e-3、かつ legacy 最終勾配 ≥ 100× canonical 最終勾配（canonical 全目的関数、L2 を含む）。
- `C2（標準化）`: TRAIN z-cov 条件数 ≥1e3、whitened 共分散の単位行列からの乖離 maxCovDeviation ≤1e-8、および legacy-Adam-on-whitened の CTX/ATT gap ≤2 token。
- `C3（Adam の座標依存）`: 合成変換テスト（32 語彙クラス、legacy が変化し canonical が不変、閾値は solver の flat-stop 精度 1e-3、測定値 1.4e-4（直交）/ 3.6e-14（一般））+ 実データの legacy gap ≥5・canonical gap ≤2。
- `C4（学習不定性）`: legacy の from-scratch と transport 初期化の trainCE 差 ≤1e-6・dev exact 差 ≥5・z-score TRAIN 設計行列の row nullspace 割合 ≥0.9・null 成分の dev 最大 |dlogit| ≥1e-2。
- `C5（calibration 選択）`: 選択 step と最終 step の trainCE 差 ≥1e-3、選択 step 割合の CTX/ATT 差 ≥0.1。

## 結果

### legacy anchors の bitwise 再現

canonical 実行は前回公開の READOUT_PROBE_V1 anchor を cond-1（calibration 選択済み legacy Adam、raw z-score 座標）で完全再現した。

| 構成 | CTX dev exact | ATT dev exact |
|---|---|---|
| L19_SEED_1 | 24 | 6 |
| L19_SEED_2 | 37 | 24 |
| L19_SEED_4 | 57 | 47 |
| L18_SEED_2_CONTROL | 68 | 64 |

### canonical solver では CTX == ATT

全最大低下層で、whitened + L-BFGS の canonical probe は CTX と ATT で同一の dev token exact に収束した。

| 構成 | CTX canonical dev exact | ATT canonical dev exact |
|---|---|---|
| L19_SEED_1 | 16 | 16 |
| L19_SEED_2 | 32 | 32 |
| L19_SEED_4 | 52 | 52 |
| L18_SEED_2_CONTROL | 30 | 30 |

同一収束点は dev exact だけでなく whitened 空間の目的関数でも CTX≈ATT を 8 桁一致する。L-BFGS は flat stop（勾配 ≈2e-8）、cross-init の最大差は 7.6e-5 であり、解の一意性が確認された。

### 判定と補助観測

- **C1（最適化不足）: 4/4 層**。legacy 最終勾配は canonical 目的の最適値から大きく外れる（flat stop の許容勾配 2e-8 に対し legacy は 1e-2 オーダー）。
- **C2（標準化）: 1/4**。
- **C3（座標依存）: 3/4**。
- C4（学習不定性）: 0、C5（calibration 選択）: 0。
- 曲線診断: legacy の dev exact は深度とともに単調に減少するが、canonical 曲線は
  非単調（spearman ρ は構成ごとに −0.49 〜 +0.13）。これは「深層ほど読み出しにくい」
  という前回の観測が Adam の artifact であることを支持する。
- 特徴形状: TRAIN z-cov の条件数は ATT で 4.2e7〜2.5e8、CTX で 5.3e5〜9.3e5。近 null 次元は eigenvalue floor に張り付くが、dead coordinate はない（全 rank 維持）。
- attention の deep band 低下は canonical でも維持（表現側の真の傾向）であり、射影の full-rank transport parity は前回のまま維持される。

## 前回診断の修正

`previous-result-corrections.csv` の5件（Allow-list 公開済み）。

1. 「CTX dev exact が ATT を上回る（投影 drop）」→ **撤回・言い換え**。canonical では CTX==ATT であり、差は probe 学習 artifact。
2. 「dev-TF 精度は深度とともに単調減少」→ **一部修正**。legacy 単調減少は Adam artifact。canonical は非単調。
3. 「Attention blocks 11-18 が readout を劣化させる」→ **維持**（canonical でも deep band 低下）。
4. 「出力射影は線形クラス情報を保持」→ **維持**。
5. 「legacy Adam probe は両 tap で収束する」→ **一部修正**。legacy の最終勾配は最適性から遠く、信頼できる probe には canonical solver が必要。

## 次の候補

- canonical L-BFGS probe（whitened, L2=1e-4）を層診断の既定手法にする。
- legacy Adam の production diagnostics における置換（最終勾配が最適性から遠いため）。
- PCA-whitened 特徴の既定化: 今回は inactive（条件性の evidence は記録済み）。
- 学習データ不定性監査: 該当なし（null 方向に隠れた差はなし）。

## 実行予算（1実行あたり）

- CPU trajectory 再生成: 4 / 4
- detailed_optimization_comparisons: 20 / 24
- representative_non_drop_comparisons: 32 / 32
- canonical_probe_full_reevaluation: 461 / 700
- convex_solver_runs: 76 / 100
- legacy_adam_runs: 40 / 60
- final holdout 開封: 0 / 0

AMENDMENT_4（結果確定前）により report-writer の列ずれ修正のため、
AMENDMENT_5（独立レビュー対応）により corrected-layer-curve の uint64 wrap・
合成不変性ミラーのクラス数不一致・C2/C3 規則文言の整合のため、同一 protocol を
再実行した。予算 cap は1実行あたりの cap として不変で、各再実行も同一予算内に
収まる。累積実行量は private-diagnostics の `run-budget.csv` に記録される
（情報用、gate ではない）。

## 公開成果物

- [docs/results/qnn-l19-probe-optimization-audit-2026-08/](results/qnn-l19-probe-optimization-audit-2026-08/)
  - README.md、manifest.json、各集計 CSV（configuration / dataset-usage / legacy-vs-canonical-probe / corrected-layer-curve / corrected-attention-taps / feature-geometry / row-nullspace / calibration-selection / optimization-summary / diagnosis / previous-result-corrections / next-step-candidates / budget）
  - 生 tap、生 probe 重み、生 logit、raw checkpoint は含まない
- [docs/qnn-l19-probe-optimization-audit.md](qnn-l19-probe-optimization-audit.md)（本稿）

## 修正した文書

- [docs/qnn-l19-attention-internal-diagnosis.md](qnn-l19-attention-internal-diagnosis.md) — 「出力射影が原因」を probe 学習 artifact として修正注記。
- [docs/qnn-l19-readout-representation-diagnosis.md](qnn-l19-readout-representation-diagnosis.md) — 単調減少・legacy 収束の主張を一部修正注記。
- [docs/qnn-l19-intra-block-readability-diagnosis.md](qnn-l19-intra-block-readability-diagnosis.md) — deep band 低下が canonical solver でも維持されることを注記（結論は不変）。
- [docs/qnn-l19-output-projection-information-audit.md](qnn-l19-output-projection-information-audit.md) — transport parity 維持の注記追記（結論は不変）。

## 制約

- host-only CPU のみ。QNN graph、Android/JNI、実機、HTP、QAIRT、final holdout は変更・使用していない。
- production コード（`app/src/main/cpp`）は変更していない。
- 結論は結果を見る前に固定した protocol（PROBE_OPTIMIZATION_AUDIT_V1 version 6、hash `fnv1a64:b36b4745b9b4807f`）に従った。AMENDMENT_1（solver 役割・収束）、AMENDMENT_2（row/nullspace・transport 等価初期化・gauge 自己テスト・C1 の意味）、AMENDMENT_3（写像 bias 補正）、AMENDMENT_4（writer 修正に伴う再実行、予算は1実行 cap のまま）、AMENDMENT_5（独立レビューによる corrected-layer-curve wrap・合成ミラー修正・C2/C3 規則文言整合）はいずれも結果公開前に適用した。