# L19 Attention 出力射影の情報損失監査（OUTPUT_PROJECTION_AUDIT_V1）

## 結論

**出力射影は線形な次token情報を保持している。前回観測された probe 性能の低下は、射影そのものによる情報損失ではなく、射影後の表現に対する probe の初期化・標準化・最適化の問題である。**

## 背景

前回の `ATTENTION_INTERNAL_V1` 診断では、次が観測された。

- 各 head の context には比較的多くの情報が残る。
- head を結合した context から Attention 出力射影を通すと probe 性能が大きく低下する。
- 単一 head 除去では一貫して改善しない。
- Attention 重みだけ、または V だけを置換しても一貫して回復しない。
- teacher-forced でも低下する。

しかし Attention 出力射影は $16 \times 16$ の線形変換である。行列が可逆なら、context から線形に読み取れる情報は理論上、射影後からも線形に読み取れる。したがって前回の結果だけでは次の2つを区別できなかった。

1. 出力射影がランクを落とし、正解token情報を実際に失わせている。
2. 出力射影は情報を保持しているが、座標系の悪条件化・標準化・probeの初期化・最適化によって射影後の probe が十分に学習できていない。

本監査はこの二者を厳密に切り分ける。

## 方法（host-only CPU）

対象は前回と同一の4構成。

- `L19_SEED_1` FINAL step 320, max-drop layer 9
- `L19_SEED_2` FINAL step 320, max-drop layer 7
- `L19_SEED_4` FINAL step 320, max-drop layer 12
- `L18_SEED_2_CONTROL` FINAL step 320, max-drop layer 6

前回対象層すべて（L19: 1,4,7,8,9,11,12,13,14,15,16,17,18；L18: 1,4,6,8,10,11,12,13,14,15,16,17）で行列統計を取得し、最大低下層4件で完全な probe transport を実施した。

### 行列向き

実コードの `mm(z.ctx, p.wo, tokens, dim, dim)`（row-major）から、context を行ベクトル $x$、出力を $y$、出力射影を $W$ として

$$
y = x W
$$

とした。$W$ は `p.wo` を $16 \times 16$ row-major として解釈する。

### z-score 標準化を含む transport

context probe は z-score 標準化の上で

$$
\text{logits} = z_x A + b, \qquad z_x = (x - \mu_x) D_x^{-1}
$$

を学習する。これを生座標に戻すと

$$
C = D_x^{-1} A, \qquad b_{\text{raw}} = b - \mu_x C
$$

射影後の生座標で $W B_{\text{raw}}^T = C^T$、つまり

$$
B_{\text{raw}} = C (W^+)^T
$$

を満たす係数を求める。可逆なら $W^+ = W^{-1}$、一般には Moore-Penrose 疑似逆行列を用いる。射影後の z-score 座標に戻すと

$$
B = D_y B_{\text{raw}}, \qquad b_y = b_{\text{raw}} + \mu_y B_{\text{raw}}
$$

となる。すべての転置・broadcast は実コードの `probeForward` と同一の row-vector 規約に従う。

### 閾値（結果前固定）

- 数学的 rank 判定：$\text{tol}_{\text{double}} = \max(m,n) \cdot \epsilon_{\text{double}} \cdot \sigma_{\max}$
- float 実効 rank 判定：$\text{tol}_{\text{float}} = \max(m,n) \cdot \epsilon_{\text{float}} \cdot \sigma_{\max}$
- 疑似逆行列には $\text{tol}_{\text{double}}$ を使用
- zero-step logit 一致：最大 logit 差 $\le 10^{-5}$、argmax flip $=0$、token exact 差 $=0$

## 結果

### 特異値と rank

全対象層、全seedで数学的 rank は 16（full rank）。最小特異値は約 $2.6\times10^{-3}$ 〜 $3.5\times10^{-2}$、条件数は約 20 〜 460。いずれも float 実効 rank も 16 であり、rank 不足は観測されなかった。

### null-space 成分

最大低下層4件の context probe 生座標係数 $C$ における null-space 成分は、$\|C_{\text{lost}}\|_F / \|C\|_F$ で $10^{-14}$ 〜 $10^{-13}$ オーダー。これは数値丸めであり、射影の到達可能部分空間外に分類方向が出ていない。

### zero-step logit 一致（最重要）

context probe と transport 済み射影後 probe の zero-step logit を同一 teacher-forced 行で比較した結果（DEVELOPMENT partition）：

| 構成 | layer | 最大 logit 差 | argmax flip | token exact 差 |
|---|---|---|---|---|
| L19_SEED_1 | 9 | $1.37\times10^{-6}$ | 0 | 0 |
| L19_SEED_2 | 7 | $1.92\times10^{-6}$ | 0 | 0 |
| L19_SEED_4 | 12 | $3.02\times10^{-6}$ | 0 | 0 |
| L18_SEED_2_CONTROL | 6 | $8.22\times10^{-7}$ | 0 | 0 |

double transport で context probe と完全に一致する。

### float との比較

float 精度で計算した疑似逆行列による transport でも、最大 logit 差は $2\times10^{-5}$ 〜 $3\times10^{-5}$、argmax flip は 0、token exact 差は 0。条件数が float に対して大きい層はなかったため、「float では読み出せない」ほどの悪条件化は観測されなかった。

### from-scratch probe との比較

射影後表現でゼロ初期化から学習した probe は、context probe より性能が低い（例：L19_SEED_1 layer 9 DEVELOPMENT で context 24、from-scratch 6）。しかし transport 済み probe は step 0 で context probe と一致し、同じ学習 protocol で追加学習してもその一致は維持される。これは出力射影が情報を失っているのではなく、from-scratch 学習が同じ解を見つけられていないことを示す。

## 判定基準との照らし合わせ

- 数学的 rank が 16：✅
- transport 済み probe が context probe と logit 一致：✅
- argmax flip 0、token exact 差 0：✅
- double transport 一致、float transport も分類に影響せず一致：✅

したがって「出力射影が情報を保持」の条件を満たす。

## 前回診断の修正

前回の表現「出力射影で情報が失われる」は、今回の数学監査により正しくない。正しい表現は以下のいずれかとなる。

- 出力射影は線形情報を保持している。
- 観測された性能低下は、射影後の probe 学習・標準化・数値条件によるものである。

## 次の候補

- 射影後の probe 学習を改善する（より長い学習、warm-start、異なる初期化・標準化）。
- 複数層にわたる residual/FFN 経路での情報回復を調べる。
- head 間の interference や multi-head accumulation が実際に情報を損なうかを独立に監査する。

## 実行予算

- CPU trajectory 再生成：4 / 4
- 行列分解：51 / 60
- 完全 probe transport：4 / 24
- transport warm-start 学習：4 / 24
- 実機試験：0
- HTP 試験：0
- final holdout 開封：0

## 公開成果物

- [docs/results/qnn-l19-output-projection-information-audit-2026-08/](results/qnn-l19-output-projection-information-audit-2026-08/)
  - README.md、manifest.json、各集計 CSV
  - 生行列・生 probe 重み・生 logit・端末情報は含まない

## 制約

- host-only CPU のみ。QNN graph、Android/JNI、実機、HTP、final holdout は変更・使用していない。
- 出力射影の値そのものは変更していない。
- 結論は結果を見る前に固定した protocol（OUTPUT_PROJECTION_AUDIT_V1、hash `fnv1a64:c35a2e6ae3102772`）に従った。
