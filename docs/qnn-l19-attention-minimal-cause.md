# L19 Attention経路の最小原因監査

> **2026-08 context supervision追補:** 本書の「広域distractor混合とV/O・残りの
> モデルの共同適応」は、canonical homogeneous-only TRAINでの失敗機構として維持される。
> 後続のdata介入では、通常Attentionのまま
> target不変mixed prefixを80/320 batch学習すると全L19 seedが安定し、matched
> homogeneous controlは失敗した。V/O固定は不要だったが、この介入だけでV/O機構を
> 一意に再同定したわけではない。詳細は
> [context supervision監査](qnn-l19-context-supervision-stability.md)を参照。

## 結論

L19のseed不安定性を生むAttention側の主要因は、内容依存のQ/K学習そのものではない。
均質なphase-0 contextだけで学習すると、Attentionがactive suffix外の無関係なtokenまで
広く混合できることと、その混合にV/Oおよび残りのモデルが共同適応することが組み合わさり、
seedごとに異なるmixed-context解を作る。自由生成では、その小さな順位差が次のprefixへ
フィードバックされる。

この判定はAttention全体のzero介入より小さい。自己位置だけを見る固定Attentionで最初から
学習すると、L19 seed 1/2/4とL18 seed 2はすべて144/144 token、24/24 sequenceになる。
直前tokenだけを見る固定Attentionは140/144、144/144、134/144であり、一様因果混合は
31/144、24/144、78/144まで悪化した。直前tokenはmixed prefixのactive suffix内だが、
一様混合は前半のdistractorも必ず含む。この比較は、単なるAttention容量ではなく、
広い無関係contextを混ぜる能力を識別する。

Q/Kを初期値へ固定してもL19は40/144、75/144、84/144に留まるため、学習されたQ/K選択は
必要条件ではない。逆にV/Oを初期値へ固定すると132/144、110/144、134/144まで改善し、
L18 controlは144/144へ回復した。したがってV/Oの更新履歴は主要な増幅要因である。ただし
L19ではV/O固定だけで完全回復しないため、V/O単独を根本原因とはしない。

証拠の強さは「主要因」とする。初期V/Oとその他parameterの初期値を完全因子化する
320-step実験は、事前登録した24本の上限へ達したため行っていない。

## 範囲と安全条件

- host-only CPUだけを使用した。
- 対象はT8/D16/FFN32/H2、L19 seed 1/2/4とL18 seed 2 controlである。
- TRAIN順序は全seedで同じ`(step-1)%4, phase=0`、Adam m/vはすべて0から開始する。
- AR_DEVELOPMENT_V3だけを原因探索へ使用した。
- AR_FINAL_HOLDOUT_V3はhash確認だけで、評価していない。
- device、HTP、QNN、QAIRT、ADB、Android/JNI、UIは使用・変更していない。
- 介入は診断用の反事実であり、production改善として採用していない。

## 測定器の再監査

新runnerは、AR developmentのpinned hashとordered row hash、既知exact anchor、canonical
`runFormalCpu`と独立no-op loopのparameter/m/v bitwise一致、独立評価器のcase別exact・
sequence・first-error一致、alpha=1 no-op、固定patternの因果mask、freeze範囲、finiteを
実験より先にfail closedで確認した。

### 新たに見つかったhash不具合

`readout_probe::fnv1aParams`は、名前から想定されるfloat内容ではなく、各floatが0か非0かを
1 byteへ落としてhashしていた。従来bundleの`param_content_hash_*`はcheckpoint内容hashではなく、
実質的なzero-support-mask hashである。今回のrunnerはregistry名、shape、float bytesを含む
content hashを使用し、Adam m/vも別々にhashした。

この不具合は過去のexact品質値、Attention-zero、FFN-zeroの集計を変えない。ただし過去hashを
exact checkpoint identityの証拠として使う説明は訂正する。

## 競合仮説

開始時は、測定人工物、最終Attention forward、内容依存Q/K、V/O、training gradient経路、
residual量とLayerNorm、初期値と共同適応、単なる容量削減を競合させた。測定人工物は独立評価器と
bitwise no-op、容量説は既存FFN-zeroで棄却した。最終forwardだけ、単純な残差量、学習Q/Kの
必要性も今回の介入で反証した。

## 実験サイクル1: forwardとtraining履歴

同じcanonical step-320 checkpointを変えず、評価時だけ全層W_Oをscaleした。またpatternを
自己位置、直前token、一様因果へ置換し、FFN W2 zeroを負の対照とした。

| 介入 | L19 s1 | L19 s2 | L19 s4 | L18 s2 |
|---|---:|---:|---:|---:|
| canonical | 30 | 63 | 46 | 65 |
| Attention alpha=0 | 42 | 38 | 78 | 16 |
| 固定self | 45 | 56 | 48 | 70 |
| 固定previous | 41 | 58 | 30 | 69 |
| 固定uniform | 25 | 41 | 60 | 33 |
| FFN W2 zero | 20 | 13 | 9 | 19 |

値はfree-running token exact / 144である。alpha=0でも完全回復せず、seed 2とL18では悪化した。
alphaを1から0へ下げる単調改善も全seedでは成立しない。したがって最終Attention forwardや
residual量だけでは説明できず、学習中に形成されたtrajectoryが必要である。

## 実験サイクル2: token間混合

Q/Kを0へ固定し、softmax Jacobianを学習へ流さず、V/OとLN1以下は学習させた。selfとpreviousは
ともに各rowでone-hotなので、重みのentropy、L1/L2 norm、最大値が等しい。

| 学習pattern | L19 s1 | L19 s2 | L19 s4 | L18 s2 |
|---|---:|---:|---:|---:|
| self | 144 / 24 | 144 / 24 | 144 / 24 | 144 / 24 |
| previous | 140 / 21 | 144 / 24 | 134 / 22 | 144 / 24 |

各値はtoken exact / 144とsequence exact / 24である。自己位置限定は全構成で完全回復した。
別token参照はseed 1/4に小さい残差を生んだが、canonical失敗よりかなり軽い。

## 実験サイクル3: 広域混合とQ/K・V/O更新

| 学習介入 | L19 s1 | L19 s2 | L19 s4 | L18 s2 |
|---|---:|---:|---:|---:|
| 一様因果混合 | 31 / 2 | 24 / 0 | 78 / 12 | 21 / 1 |
| Q/K初期値固定 | 40 / 2 | 75 / 8 | 84 / 13 | 55 / 6 |
| V/O初期値固定 | 132 / 22 | 110 / 17 | 134 / 22 | 144 / 24 |

一様混合はQ/Kを学習せず、すべての過去tokenを混ぜる。それでも全seedで大きく失敗するため、
内容依存Q/K学習は必要ではない。previousとの差はL19全seedで事前登録条件を満たし、active suffix外の
distractorを含む広域混合を支持した。

Q/K固定でも失敗する一方、V/O固定は大幅に改善する。この非対称性から、V/O更新は有害な広域混合へ
モデルを適応させる主要な増幅経路と判断する。L19 seed 2で残る110/144は、固定V/Oだけでは
非Attention側や学習Q/Kとの共同適応を除けないことを示す。

## 初期値と更新履歴

seedが直接変えるのは初期parameterだけである。Adamの初期m/v、batch順序、学習率scheduleは
全seedで同一なので、「seed固有の初期optimizer state」は実装上存在しない。

評価時だけAttentionを除いても回復せず、学習開始からpatternを制限すると回復するため、更新履歴は
必要である。V/O freezeの改善はV/O更新履歴の因果寄与を示す。一様混合では全seedのQ/Kが同じ0でも
31/24/78へ分かれるため、Q/K初期値は共通のseed源ではない。

一方、V/O初期値とembedding/FFN/head初期値のどちらがseed差を主に決めるかは未分離である。

## 最終判定

因果鎖は、均質TRAINがmixed distractorへのAttention利用を拘束しないこと、自己位置限定なら全seedが
完全回復すること、active suffix内の直前tokenなら小さい誤りに留まること、内容非依存でも広域一様
混合が失敗を再現すること、Q/K学習を止めても失敗するがV/O学習を止めると大幅改善すること、
最終forwardだけ除いても回復しないこと、free-running argmaxが残差を増幅することから成る。

従ってAttention全体より小さい主要因は「無関係tokenを含む広域context混合」と
「その混合に対するV/O・残りのモデルのseed依存共同適応」の組合せである。

## 残る不確実性

- 初期V/Oと非Attention初期値の寄与率は分離していない。
- V/O固定でもL19は完全回復しないため、LayerNormや非Attention parameterのどの更新が残差を
  担うかは未特定である。
- fixed patternは診断用であり、通常学習の改善方式として採用していない。
- AR final holdoutを開いていないため、性能候補の最終評価ではない。

公開aggregateは
[結果bundle](results/qnn-l19-attention-minimal-cause-2026-08/README.md)にある。
