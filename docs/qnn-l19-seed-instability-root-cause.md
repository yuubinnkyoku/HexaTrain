# L19 seed依存生成品質の根本原因再監査

> **2026-08 Attention最小原因の追補:** 後続の固定pattern・subgroup freeze監査により、
> Attention全体より小さい機構まで分解した。自己位置限定学習は全seedを完全回復させる一方、
> 一様因果混合はQ/K学習なしでも大きな失敗を再現した。Q/K更新は必要条件ではなく、広域の
> distractor混合とV/O・残りのモデルの共同適応が主要因である。詳細は
> [Attention最小原因監査](qnn-l19-attention-minimal-cause.md)を参照。
>
> また、本書と旧bundleで`parameter content hash`と呼んだ`fnv1aParams`はfloat内容ではなく
> zero/nonzero support maskのhashだった。品質集計は変わらないが、exact checkpoint identityの
> 証拠としては使用しない。

## 結論

L19 Transformerのseed差は、一つの数値不具合や出力headの後半driftではなく、
データ、モデル経路、学習軌道、生成評価が連鎖した複合原因である。

1. 正解規則は現在tokenだけで一意に決まる。
2. 320 stepの学習はhomogeneous phase-0 contextだけを反復する。
3. Attention経路の存在はmixed prefixでのseed依存失敗に因果的に必要である。
   文脈shortcutが最有力の詳細機構だが、branch間のgradient競合や正規化状態まで
   分離した証拠ではない。
4. seedが変える入力は初期parameterだけであり、その差は決定的な学習軌道を経て
   異なる文脈依存挙動として残る。ただし初期化と個別更新の寄与は未分離である。
5. mixed distractor prefixで局所的な順位差が現れ、argmax free-runningが
   その誤りを後続tokenと系列完全一致へ増幅する。

Attention branchを全学習期間で構造的に0とする診断介入では、L19 seed
1/2/4とL18 seed 2 controlの全てがmixed developmentでteacher-forced、
free-runningとも144/144 token、24/24 sequenceへ到達した。同じ幅のFFN
branchを0とする負の対照は全構成を23--39/144へ悪化させた。このため、
Attention経路の存在がmixed-context不安定性の主要因であることは、複数seedと
深度scope controlで因果支持される。文脈shortcutそのものは機構候補に留める。

これはAttention実装の誤り、Attentionを除くproduction変更、L19だけの深度境界、
あるいはAdam moment単独の原因を意味しない。L18 controlも同じ介入で回復したため、
少なくとも検証したL18 seed 2にも同じ効果があり、L19だけに排他的な機構ではない。
L18の他seedや深度一般への外挿はしない。

## 安全範囲

全調査はhost-only CPUで行った。実機、HTP、QNN、QAIRT、ADB、UIは使用して
いない。`AR_FINAL_HOLDOUT_V3`は既知hash
`fnv1a64:aa5081e6df658b4a`との同一性だけを確認し、評価には使用していない。
Attention-zero介入は原因確認用の反事実実験であり、改善案として採用していない。

## 競合仮説

開始時に9個の独立仮説を登録した。表のH9/H10は、初期H9の「数値・HTP要因」を
測定範囲の監査後に分割したscope refinementであり、結果前の新規仮説を装って
いない。判定条件は各実験前にignored領域の
`build/private-diagnostics/l19-root-cause-goal/decisions/`へ記録した。

| 仮説 | 最も識別的な実験 | 結果 |
|---|---|---|
| 評価器・probeの人工物 | row/target契約、swap identity、集計監査 | 一部確認。旧probe/swap集計に不具合 |
| prefixの曖昧性 | splitごとの直接列挙 | 棄却。曖昧context 0 |
| 学習/評価context差 | 同checkpointのhomogeneous/mixed比較 | 寄与を支持。ただし単独ではseed差を説明せず |
| teacher forcingとfree-runningのずれ | 同prefix集合のTF/FR比較 | 増幅要因として支持 |
| 学習後半の共通output-head drift | step 24からfreeze/moment reset | 共通原因を棄却 |
| Attention Adam moment履歴 | step 24のAttention moment reset | 単独原因を支持せず |
| Attention経路がmixed-context不安定性を媒介 | Attention-zero training | 経路レベルの主要因として因果支持 |
| L19だけに排他的な深度・conditioning | L18 seed 2同一介入 | 検証範囲では非排他的。L18 seed 2も完全回復 |
| HTP・実機固有 | CPUで同じseed差を再現 | 棄却 |
| hidden/gradient/momentを含む非有限・float病理 | training lossと評価対象logit/prob監査 | 評価範囲では支持せず。全tensor監査ではない |

## 測定器の再監査

dataset identity、checkpoint identity、parameter hash、canonical trajectory、
teacher-forced/free-running分離、exact/NLL/first-error/marginの定義、partition
分離を既存manifestとsourceから再確認した。その過程で三つの診断上の不具合を
発見した。

### TRAIN probe row契約

`teacherForcedRows`は、正式training batchのposition別targetをcontinuation列と
同じように扱い、契約に存在しない4行を作っていた。このlegacy TRAIN行では
current-token exactが28/32だったが、正式batchのrow/targetを使うと32/32になる。
このため旧readout、intra-block、attention-internal、probe-optimizationの
学習済みprobe絶対値、深度/tap差、z-statは再生成まで原因証拠から除外する。

### cross-seed context swap

旧実装はdonorのdevelopment row 0だけを使い、token-row indexを欠いた位置へ
replacementを書き、同じ不正vectorを全development rowへ再利用していた。
row-wise identityが成立せず、対照方向もなかったため、公開済みswap recoveryは
無効であり、今回の因果判定には使用していない。実装は同一development rowの
全token rowを交換するよう修正し、self-swap identityをself-testへ追加した。

### projection contribution集計

旧実装は全development rowについてlogitは平均していたが、norm/cosineは最初の
1 rowだけを保持していた。per-row分解とcaller側集計へ修正した。旧norm/cosine
aggregateは無効である。直接logit介入、出力射影のfull-rank、疑似逆、probe
transportはこの不具合に依存しない。

## データ構造

| split | context | occurrence | unique | ambiguous |
|---|---:|---:|---:|---:|
| TRAIN | current token | 32 | 13 | 0 |
| TRAIN | previous 2 tokens | 28 | 13 | 0 |
| AR validation | current token | 144 | 13 | 0 |
| AR validation | previous 2 tokens | 144 | 13 | 0 |
| AR validation | 8-token model window | 144 | 100 | 0 |
| AR validation | full causal prefix | 144 | 144 | 0 |
| AR development | current token | 144 | 13 | 0 |
| AR development | previous 2 tokens | 144 | 13 | 0 |
| AR development | 8-token model window | 144 | 100 | 0 |
| AR development | full causal prefix | 144 | 144 | 0 |
| Margin calibration | current token | 144 | 13 | 0 |
| Margin calibration | previous 2 tokens | 144 | 13 | 0 |
| Margin calibration | 8-token model window | 144 | 73 | 0 |
| Margin calibration | full causal prefix | 144 | 144 | 0 |
| Margin development | current token | 144 | 13 | 0 |
| Margin development | previous 2 tokens | 144 | 13 | 0 |
| Margin development | 8-token model window | 144 | 74 | 0 |
| Margin development | full causal prefix | 144 | 144 | 0 |

targetは全splitで現在tokenから一意に決まる。従ってprefix ambiguityや識別不能性は
原因ではない。一方、trainingはhomogeneous phase-0だけで、mixed distractor
prefixを含まない。入力embeddingのtoken-only可読性は単純な規則を示すだけで、
深層表現が汎化することの証明には使っていない。

## 実験サイクル1: context shiftと評価増幅

この表のmixedは`AR_DEVELOPMENT_V3`である。旧probe文書の
`MARGIN_DEVELOPMENT_V1`（例えばL19 seed 1 final 50/144）とは別partitionであり、
両者を混在させていない。

| 構成 | homogeneous FR | mixed TF | mixed FR |
|---|---:|---:|---:|
| L19 seed 1 | 32/32 | 96/144 | 30/144 |
| L19 seed 2 | 24/32 | 103/144 | 63/144 |
| L19 seed 4 | 32/32 | 101/144 | 46/144 |
| L18 seed 2 | 26/32 | 114/144 | 65/144 |

mixedで全構成が低下し、TFからFRへさらに低下する。しかしhomogeneousにもseed差が
残るため、context shiftだけでは不安定性を完結に説明しない。context shiftを上流の
寄与、free-running exactを増幅器と判定した。

## 実験サイクル2: late state介入

canonical step 24から320まで、output freeze、output Adam moment reset、
Attention Adam moment resetを行った。mixed FR tokenのbaseline差は次だった。

| 構成 | output freeze | output moment reset | Attention moment reset |
|---|---:|---:|---:|
| L19 seed 1 | +1 | +2 | 0 |
| L19 seed 2 | -30 | -7 | -4 |
| L19 seed 4 | +2 | +2 | -5 |
| L18 seed 2 | -4 | -14 | -22 |

方向はseed間で一致しない。特にseed 2ではlate output headが劣化原因ではなく、
上流表現を補償していた。従って共通のoutput-head ranking driftとAttentionの
moment履歴単独原因を棄却した。初期化や学習軌道が異なる解を選ぶという説明は残るが、
この実験だけでAdam momentを根本原因とはしない。

## 実験サイクル3: 経路の因果介入

同一seed、同一data order、同一320-step recipeで、Attention residual branchを
全層・全stepで0にした。負の対照では同じhidden幅へ書くFFN branchを0にした。

| 構成 | baseline mixed FR | Attention-zero | FFN-zero |
|---|---:|---:|---:|
| L19 seed 1 | 30/144 | 144/144 | 26/144 |
| L19 seed 2 | 63/144 | 144/144 | 39/144 |
| L19 seed 4 | 46/144 | 144/144 | 27/144 |
| L18 seed 2 | 65/144 | 144/144 | 23/144 |

Attention-zeroは全構成でteacher-forcedも144/144、free-running sequenceも
24/24、homogeneousも32/32であり、training lossと評価対象logit/probabilityは
finiteだった。hidden state、gradient、optimizer momentの全tensor finite監査を
意味しない。FFN-zeroは全構成で悪化したため、同じparameter数のgenericな
branch-zero/capacity controlでは効果を再現しない。ただし両branchの機能とgradient
は対称ではない。データ規則に不要なcontext mixingを可能にする経路を除いた方向だけが、
予測通りmixed汎化とseed差を同時に消した。

## 原因の強さ

Attention経路の存在は、複数seed、L18 seed 2 scope control、予測方向の
因果介入、FFN-zero負の対照を持つため「主要因」以上である。さらに、上流の
homogeneous-only training分布、Attentionが媒介するmixed-context感度、
free-running増幅が、局所margin/ranking差と系列exact差を一つの連鎖として説明
するため、この適用範囲では複合的な根本原因と判定する。この強い因果判定は
data/context差、Attention経路、free-running増幅に限定し、初期化と個別updateの
内訳は未解決の機構候補とする。

ただし、どのAttention head、layer、Q/K/V要素が失敗を担うか、文脈shortcutと
gradient競合・正規化変化のどれがbranch介入効果を生むか、初期化とupdateの
どちらが最初の分岐を決めるかは未特定である。Attention-zeroは構造的に強い
介入なので、自然なtrained model内の最小十分経路までは同定していない。

## 過去の説明との関係

- critical token margin不足は、系列exactを増幅する下流現象として維持する。
- 学習後半のoutput-head ranking driftはseed固有の補償を含み、共通原因ではない。
- 深層でのlinear readability低下とAttention block前後差は、旧TRAIN row bugの
  影響を受けるため、旧probe絶対値による原因主張を保留する。
- 出力射影のfull-rank、疑似逆、座標transportは維持する。
- 射影前後のlegacy probe差を最適化artifactとする数学的説明は維持するが、
  旧TRAIN行から学習した絶対scoreは再利用しない。
- 旧cross-seed swapとprojection contribution norm/cosineは無効とする。

## 再現

```powershell
.\scripts\run_l19_seed_instability_diagnostics.ps1 -SelfTest
.\scripts\run_l19_seed_instability_diagnostics.ps1 -DataAuditOnly
.\scripts\run_l19_seed_instability_diagnostics.ps1
.\scripts\run_l19_seed_instability_diagnostics.ps1 -OptimizationInterventions
.\scripts\run_l19_seed_instability_diagnostics.ps1 -BranchAblations
.\scripts\export_public_qnn_l19_seed_instability_results.ps1 -SelfTest
```

公開aggregateは
`docs/results/qnn-l19-seed-instability-root-cause-2026-08/`に置く。raw checkpoint、
parameter、optimizer state、hidden state、logit、Attention行列、gradient、絶対path、
private run identityは公開しない。
