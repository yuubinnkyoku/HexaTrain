# L19 context supervisionによるseed安定化

## 結論

> **実テキストでの境界:** 後続の
> [ニコニコ大百科実テキストpilot](qnn-nicopedia-real-text-pilot.md)では、held-out NLLの
> 同程度のseed不安定性は再現しなかった。以下の因果結論は、current tokenだけでtargetが
> 決まる均質な人工TRAINに対するものであり、自然な日本語へ無条件に一般化しない。

均質なphase-0 batchだけを反復するTRAINは、L19のmixed-context seed不安定性を
生む上流のデータ条件だった。通常のlearned causal Attentionを変えず、320 stepの
うち80 batch（25%）だけをtarget不変のmixed prefixへ置換すると、L19 seed 1/2/4は
事前登録した安定性gateをすべて通過した。

介入batchの各rowもtargetは常に`successor(current token)`である。過去tokenを
参照しなければ解けない課題は追加していない。従って有用なcontext利用信号は回復の
必要条件ではなく、target不変なmixed-context supervisionだけで十分だった。ただし
同じactive suffixを複数prefixで対にした不変性指標は測定していないため、「不要な
prefix tokenへの不変性を学習した」という内部機構は一部支持に留める。

同じ320 step、2,560 supervised row、family exposure、special-step cadence、aggregate
input/target histogramを持つphase-diverse homogeneous controlでは全seedを安定化
できなかった。この対照により、単なるdata量・step数・token頻度だけでは説明できない。

25%の同一multisetをmixed-firstとmixed-lastへ並べ替えると両方がgateを通ったため、
curriculumの向きは必要条件ではない。ただしseed 2のtoken exactはinterleaved 136、
mixed-first 141、mixed-last 144であり、順序が残差へ影響しないという意味ではない。

12.5%と6.25%は失敗したので、最小の有効介入は「検証した三段階の中では25%」である。
これは全ての割合・scheduleに対する数学的最小値の主張ではない。

## 範囲と安全条件

- host-only CPU、T8/D16/FFN32/H2、L19 seed 1/2/4、L18 seed 2 control。
- Adam、learning rate 0.003、320 step、initial m/v=0を維持した。
- AR_DEVELOPMENT_V3だけをprimary評価に使用した。
- AR_FINAL_HOLDOUT_V3は既知hashとの一致だけを確認し、評価していない。
- device、HTP、QNN、QAIRT、ADB、Android/JNI、UIは使用・変更していない。
- data interventionは診断結果であり、production既定値へ採用していない。

## 測定器監査

新runnerは、pinned TRAIN/AR validation/development/final hash、case数、teacher row数、
current-token target契約、training example数、sampling順序、seed、checkpoint内容hash、
parameter/m/vのcanonical bitwise parity、teacher-forcedとfree-runningの分離、token/
sequence exact、NLL、margin、first error、finiteをfail closedで検査した。

既知のlegacy TRAIN probe row不具合、旧cross-seed swap、先頭rowだけのprojection
contribution集計、zero-support-maskだけをhashした旧`fnv1aParams`は、新しい因果判定に
使用していない。最初のpre-result実行では旧support-mask hashをcontent hash anchorへ
誤って転記したためgateが停止した。exact値は一致しており、既存private正本のfloat-byte
content hashへ修正してから全実験を実行した。判定条件は変更していない。

## 競合仮説

開始時に次を競合させた。

1. homogeneous-only TRAINによる不要context不変性の未学習。
2. 有用な過去tokenを利用する学習信号の不足。
3. data量またはtraining step不足。
4. curriculum順序または学習後半の忘却。
5. dataでは解消しないV/O初期値と他branchの共同適応。
6. 広域混合による信号希釈・平均化。
7. 最適化、正規化、容量、評価器の人工物。

公式の全partitionではcurrent-tokenからtargetが一意に決まる。このため「有用contextが
性能を上げるか」は情報理論的に独立同定できない。今回のprimary介入は、この制約を
隠さず、targetを変えないdistractor-prefix多様化だけに限定した。

## data intervention

25%条件は16 stepごとに4 batchを置換する。各置換groupでは全4 familyを一度ずつ
activeにし、distractor offsetを+1/+2で交互にする。各mixed batchは6 distractor token
と2 active tokenから成る。各rowのtargetは入力rowのcurrent tokenの固定successorで、
family境界の次input tokenをtargetへ流用しない。

matched homogeneousは同じ位置でphase-diverseな4 homogeneous batchを使う。4 batch
単位のinput/target histogramはmixed条件と完全一致する。通常の非置換stepはcanonical
phase-0 round robinを維持する。

## 実験サイクル1: context多様性とmatched control

AR_DEVELOPMENT_V3 free-running token exact / 144:

| 構成 | canonical | matched homogeneous | 25% interleaved mixed |
|---|---:|---:|---:|
| L19 seed 1 | 30 | 65 | 144 |
| L19 seed 2 | 63 | 105 | 136 |
| L19 seed 4 | 46 | 45 | 144 |
| L18 seed 2 | 65 | 119 | 144 |

sequence exact / 24はL19でcanonical 2/6/6、matched 11/16/7、mixed 24/23/24だった。
mixedは全seedで大きく改善したが、seed 2が事前登録の140-token gateを4下回ったため、
cycle 1単独では「部分改善」と判定した。

matched controlもseed 1/2では改善しており、phase多様性や最適化軌道への寄与はある。
しかしseed 4は46から45で、全seed安定化を再現しない。従ってdata量・phase多様性は
補助要因になり得るが、十分原因ではない。

このmatched controlはaggregate histogramを一致させるが、row内のtoken位置、共起、
context arrangement、そこから生じるgradient geometryまでは一致させない。従って
「不要prefix不変性」は最も直接的な説明だが、mixed family共起がもたらすregularizationや
位置依存の最適化効果を完全には分離していない。これらは残る別解釈として保持する。

## 実験サイクル2: curriculum

interleaved条件と全く同じ80 mixed + 240 homogeneous batchを、mixed-firstまたは
mixed-lastへ並べ替えた。

| 構成 | mixed-first token / sequence | mixed-last token / sequence |
|---|---:|---:|
| L19 seed 1 | 144 / 24 | 144 / 24 |
| L19 seed 2 | 141 / 23 | 144 / 24 |
| L19 seed 4 | 144 / 24 | 144 / 24 |
| L18 seed 2 | 144 / 24 | 144 / 24 |

両順序が全L19 seedのgateを通り、homogeneous continuationも全構成32/32だった。
従って25%では「先にmixedが必要」「後半mixedが必要」のどちらも必要条件ではない。
一方、interleavedはseed 2のfull-stability gateを満たさないため、例の順序全般が無関係、
または全scheduleが同等とは判定しない。

## 実験サイクル3: 最小dose

新しい割合を結果後に追加せず、事前固定した12.5%と6.25%だけをmixed-lastで比較した。

| dose | L19 seed 1 | L19 seed 2 | L19 seed 4 | L18 seed 2 |
|---|---:|---:|---:|---:|
| 12.5% (40/320) | 112 | 111 | 114 | 124 |
| 6.25% (20/320) | 51 | 38 | 61 | 61 |

両doseはfull-stability gateを通らず、homogeneous性能も一部で悪化した。探索上限に従い
別の割合を追加していない。従って25%が最小の検証済み有効doseである。

## 通常AttentionとV/O共同適応

合格runではQ/K/V/Oの全groupが初期値から更新され、固定pattern、freeze、alpha scaling、
branch zeroを使用していない。development aggregateのAttention non-self massは約
0.64--0.68、Attention output normも約1.0--2.2であり、Attentionを実質的に無効化した
回復ではない。

canonicalで有害だったV/O・他branchの共同適応は、V/O固定なしでmixed supervisionに
より解消できた。この結果はV/O初期値が常に無関係だと証明するものではないが、検証した
seed不安定性に対して独立の構造介入を必要とする要因ではなかった。そのためV-only/O-only
freezeやAttention/rest初期値factorialは実施しなかった。

## 最終原因判定

均質TRAINは上流原因として支持される。より正確には、均質な現在token-successor課題だけ
ではmixed prefix上の挙動が拘束されず、通常Attentionの広域混合を前提としたV/O・残りの
モデルのseed依存共同適応が許される。target不変のmixed prefixを十分量学習させると、
通常Attentionと全parameter更新を保ったまま、このseed差が消える。不変性学習はこの結果と
整合する最有力機構だが、paired-prefix測定がないため一意には確定しない。

証拠の強さは、この適用範囲で「主要因」とする。性能改善だけでなく、matched histogram
control、same-multiset curriculum、dose negative、L18 scope control、Attention activity guardを
持つ因果支持である。final holdout未開封のためproduction採用や最終性能の結論ではない。
row内の共起・位置と不変性を独立に操作していないため、data介入の内部説明を唯一の
根本原因とまでは断定しない。

公開aggregateは
[結果bundle](results/qnn-l19-context-supervision-stability-2026-08/README.md)にある。

```powershell
.\scripts\run_l19_context_supervision_stability.ps1 -SelfTest
.\scripts\export_public_qnn_l19_context_supervision_results.ps1 -SelfTest
```
