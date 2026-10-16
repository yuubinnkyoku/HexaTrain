# G1 FP32 QK raw score / backward score scale frontier

**Forward QKᵀ rawとBackward score scaleのprecision frontierを閉じた。**
同一最終package/probe、各Lane3独立processでStandalone・native接続・
各graph自身のobserver-free accuracyを独立に確認した。既存Scalar ordered FP32を再利用し、
新packageのHVX FFN2/QKV、Muon W8 114/114回帰もPASSした。

次の標準consumerはForward Score Scale、Backward Head Scatterで精度FAILを再現した。
**次の主投資はForward Scale→Causal Mask→Softmax Pの精度修復。**
全Attention/full1-layer/full19-layerはUNVERIFIED。
Production backendとtraining updatesは変更せず、40k→60k / 200kは引き続きNO-GO。
[公開aggregate](results/g1-fp32-qk-score-scale-frontier-summary.json)に全6 processの集計を保存した。

## A. Source & reuse audit

開始時は指定base `8e13b1ed1e2464b4aef790233682a11c43c838c5`、
`investigation/ordered-fp32-hvx-qkv`、clean tree、upstreamなし。
既知remote refsへ未到達のlocal98 commitsを正として、新規
`investigation/g1-fp32-qk-score-scale-frontier`を作成した。
前研究refsと未push履歴を保持し、fetch/main再構築/history rewrite/push/PR/mergeは0。

数学の正本は `qnn_runtime_transformer_training_generalized.inc`、
共通LNは `qnn_runtime_transformer_training.inc`。production sourceは変更していない。
T32/D64/H2の実graph契約を以下に再抽出した。

| Edge / equation | QNN / geometry | Ownership and lifetime |
| --- | --- | --- |
| LN1 → Q/K/V | three MatMul NN、[32,64]×[64,64] | distinct parameter-registry weights、shared native Y |
| Q/K/V → head h | MatMul、STATIC selector [64,32] | physical channels h×32..h×32+31、head mixingなし |
| Qhead × Kheadᵀ | MatMul transpose0=false / transpose1=true、K32 | raw S [32,32]、row stride32 FP32 words |
| S → scaled → masked → P | Multiply scalar → Add causal mask → Softmax axis1/beta1 | raw/scaled/maskedはNATIVE。PもproductionはNATIVE、diagnostic設定ではAPP_READ |
| LN2 F/B → residual → dContext / dWo | LN saved state、Add、NT [32,64]×[64,64]ᵀ / TN [32,64]ᵀ×[32,64] | LN2 X/dY、upstream residual、Wo/contextはfixed external inputs |
| dContext → head → Gate → dProbabilities | selector、row Multiply、NT K32 | saved V/Gateを使用 |
| dScores | P×(dP − ReduceSum(dP×P)) | source ReduceSum axis1 / keep_dims=true。診断rank4ではaxis3 |
| dQraw / dKraw | dScores×K (NN) / dScoresᵀ×Q (TN) | saved K/Qはhead別native values |
| raw dQ/dK → scaled dQ/dK | ElementWiseMultiply scalar、各[32,32] | 4 outputs順はH0 dQ、H0 dK、H1 dQ、H1 dK |
| scaled → head scatter | MatMul input1 transpose=true、[32,32]×[64,32]ᵀ | [32,64]へone-hot placement、次にhead別Add |

Sourceは `g.attentionScale = 1.0f / std::sqrt(float(dimension / numHeads))` として
STATIC [1,1] tensorを生成し、clientBufはgraph-owned scalarを参照する。
T32/D64/H2のFP32 wordは **0x3e3504f3**。
double式の任意の丸め位置へ置き換えていない。
Forwardはraw scoreへ、Backwardはraw gradientsへ乗算する。
同じscalarを使うことと、同じtensor演算であることは区別した。

診断rank4はunit leading axesを追加するだけで、row-major layoutを変えない。
parametersとfrozen operandsはAPP_WRITE、final outputsはAPP_READ、
internal tensorsはNATIVE、Float Tap版だけ内部をAPP_READにする。
source parameter registryとlayer-local cacheは別のtensor identityを持ち、
parameter buffersはexecute中に不変。LN saved xhat/inverseもlocal backwardまで保持する。

`ForwardQKRaw`は既存 `attentioncustom::dprob` のNT geometryをtemplate specializationで再利用する。
旧markerのdefaultは0x0bを保持し、新marker **0x48383116** と独立counterを与えた。
input identityはQ0/Q1/K0/K1へ明示し、dProbabilitiesのgradient/saved-V semanticsを流用しない。
dot本体は変更しない。

```text
acc = +0.0f
for q = 0 .. 31:
    product = round32(Q[r,q] * K[j,q])
    acc = round32(acc + product)
S[r,j] = acc
```

`BackwardScoreScale`は既存 `elementProduct`（Scalar Multiply/SoftmaxProductと共通）を
4 gradientsへ適用する。新数学kernelや第二MatMul frameworkを作っていない。
markerは **0x48383117**。4 inputsの危険なoutput/receipt overlapを拒否する。
paired/grouped invocationごとに一つのcounterを持つ。同種graphのconcurrent executionは診断契約外。

旧41 Opの順序・Op名・ABI・receipts、QHPI1.4を維持し、Observed/NoReceiptの4名を末尾へ追加した。
QK receiptは6152 words、Scale receiptは8201 words。
既存Scalar ordered dot、HVX FFN2/QKV helper、strict IEEE HVX flagsは変更していない。
FMA contraction off、no fast-math、増加q、独立multiply/add rounding、
no clipping/FP16/BF16/tree reductionを維持する。
QHPI_RESOURCE_HVX宣言だけをvector arithmeticの実行証拠とはしない。

### Device評価前のgate登録

初回実装commitで以下を登録し、その後にdevice pilotを採取した。
既存QKV .002、lossless HeadSplit全word一致、LN1 Y/xhat/inverse .002、
local LN backward .02、dContext/dWo .03、Softmax dScores .002、
raw/scaled dQ/dK .01、raw QK .01を維持した。

次Forward standard Scaleは既存raw-score budgetとmask input contractから **.01**。
次Backward standard Scatterはone-hot placementが数学上増幅を起こさないため、
scaled-gradient budgetを引き継いで **.01**。観測後のgate変更は0。

## B. Forward QKᵀ

最終source **b6794c4572d10c163acb5201e811546ac564243e** / 同一package/probeの
**3独立processで認定した**。以下の数値は全processで一致した。

Standaloneは44 LN1/QKV casesを中心に、既存10 arithmetic controlsとsigned-zeroを加えた55 cases。
ordinary、low variance、mixed signs、cancellation、large/small、
head-asymmetric、transpose-sensitive、fixed randomを含む。
Host referenceはCustom helperを呼ばない独立double / ordered FP32。
Standalone Q/Kは正しいFP32 LN outputからのordered projection/head sliceである。

| Target / 44-case core | Head0 maxAbs | Head1 maxAbs | Gate / result in completed processes |
| --- | ---: | ---: | --- |
| Custom Standalone | 1.15976369557e-5 | 2.63832183123e-5 | .01、全case PASS |
| Native LN1→HVX QKV→Heads→Custom raw | 2.27499500625e-5 | 3.79479784556e-5 | .01、全case PASS |
| Standard QK Standalone negative | .0446647357344 | .0478782524430 | 各8/44 FAIL |
| Native Standard QK negative | .0446639307669 | .0478892556466 | 各8/44 FAIL、前研究の負例を再現 |

Standaloneとnative negativeはinput rounding/reference boundaryが異なる。
両者の最大値を同一条件の再現値とは扱わない。
Customは55 cases全wordでordered Host FP32と一致。全controlsのdouble最大は
Head0 3.13973856123e-5 / Head1 5.53950023345e-5。
transpose、head独立性、有限性、input mutation0、全面書き込み、poison/canary、
API成功、Observed DSP markerを確認した。

Native graphは実DSP LN1 Y→実HVX Q/K/V→実lossless HeadSplit→実Custom QKを接続する。
Host理想Q/Kを途中注入していない。
同じgraphでQKV最大誤差Q 2.4316564895e-6 / K 2.77017446315e-6 /
V 5.7436790506e-6、全headsはprojection sliceと全word一致。
LN1 Y 5.81823561774e-7、xhat 6.66749372158e-7、inverse 1.94379752543e-5、
local dX 1.53640447422e-5 / dGamma 1.96494390394e-5 / dBeta 7.89668411016e-6。
既存stage gatesの新しいFAILは0。

Observed × NoReceipt、Tapあり × なしの4 native variantsを各44 cases実行した。
NoTapはY、saved state、QKV、全headsをNATIVEにする。
ObservedではHeadSplitの実producer receiptとQK consumer operand receiptを全word照合した。
LN YはTapで直接照合し、NoTap Observedではsaved xhat/gamma/betaの独立affine再構成と
QKV consumer Yを照合する。LN receiptがYを直接記録したとは主張しない。
NoReceipt/noTapはそのgraph自身のfinal raw scores/dX/dGamma/dBetaをdoubleへ比較し、
Observed final scoresと全word一致した。不可視中間値の直接観測ではない。
Standaloneは内部edgeを持たないため、Float Tapで変更する中間tensorはない。

次consumerはCustom raw→標準HTP Multiply。4 observer variantsを診断した。

| Native raw → standard Scale | Head0 | Head1 |
| --- | ---: | ---: |
| LN1起点double reference maxAbs | .00868263396559 | **.0102389901429** |
| 実受信rawに対するlocal double maxAbs | .00868569795921 | **.0102382000946** |
| Gate .01 failure /44 | 0 | **1** |

Observed/Tapではこのconsumer graph内のraw自体にも.01とordered-word監査を行った。
標準ScaleのFAILを受け、Mask/Forward Softmax/Pへは進めていない。
**次Forward frontierはstandard Score Scaleという観測結果**を保存する。
Standalone、native接続、各graph自身のobserver-free accuracyを独立に確認し、
**DSP_FORWARD_QK_RAW_VERIFIED / FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIED**を認定する。

Reference boundaryは正しい固定LN1 X、gamma/beta、QKV weights、local dY。
全G1 producer起点や全19層Forwardの正しさを認定しない。

## C. Backward score scale

最終source **b6794c4572d10c163acb5201e811546ac564243e** / 同一最終binaryの
**3独立processで認定した**。最終cohort/途中停止cohortは含めていない。

Standaloneは既存25 frozen/native cases＋10 controls。
4出力のpositive/negative、head非対称、zero/signed zero、small/large finite、
cancellation由来、固定seed randomを維持した。

| Gradient | Standalone core maxAbs | Native fused chain maxAbs | Gate |
| --- | ---: | ---: | ---: |
| H0 dQ | 1.85788940144e-6 | 7.56094509313e-6 | .01 |
| H0 dK | 1.74719298229e-6 | 8.91115141854e-6 | .01 |
| H1 dQ | 1.39315102388e-6 | 7.44824347976e-6 | .01 |
| H1 dK | 8.73450971994e-7 | 9.84360316991e-6 | .01 |

coreは全PASS、受信rawに対するFP32 multiply全word一致。
scale **0x3e3504f3**はHost/sourceのconstructionと一致し、
全processでreceipt内のscalar wordを検査し、各235件の実受信bits logでも確認した。

追加large controlではFP32 outputのULPが大きく、double maxAbsは最大511.810546875。
これは既存25-case .01認定へ含めないarithmetic-identity stress。
全word一致・finiteは成立し、非finiteの0置換やclippingは行っていない。
double accuracyとFP32 identityを混同しない。

Native chainは
DSP LN2 F/B→Residual→dContext/dWo→HeadSplit→Gate→dProbabilities→
FP32 Softmax→RawDQDK→Score Scale。
fused Softmaxとseparate Softmaxを各4 observer variantsで実行し、
final scaled gradientsは同じobserver条件およびObserved anchorと全word一致した。
NoTapの途中dScores/rawはNATIVEで、Host理想値の注入は0。

LN2 Y/xhat最大7.30727450549e-7、inverse 1.31226658873e-6、
dX 3.37811693463e-6、dGamma 3.24221166323e-5、dBeta 7.89668411016e-6、
Residual 3.7610555772e-6、dContext 2.47924264372e-5、dWo 3.11343890047e-5。
Gate後gradient最大1.41316580304e-5、dProbabilities最大.000330738517277、
dScores最大2.26174586722e-5、raw dQ/dK最大5.21589968514e-5。
すべて既存gate内だった。Observedはraw producer/scale consumer/scale scalar/final gradientsの
receipt wordsを照合。NoReceipt/noTapは自身のactual final gradientsをdoubleへ比較し、
不可視rawを直接測定したとは主張しない。

標準Scale negativeはnative Custom rawを受信させ、以前と同一の最大誤差を再現した。

| Standard Scale negative /25 | H0 dQ | H0 dK | H1 dQ | H1 dK |
| --- | ---: | ---: | ---: | ---: |
| maxAbs | .0127901180026 | .0322076710836 | .0175945349651 | .0213635282663 |
| .01 failures | 2 | 2 | 2 | 2 |

次consumerは正しいCustom scaled gradients→標準one-hot Head Scatter。

| Standard Scatter 最終cohort /25 | H0 dQ | H0 dK | H1 dQ | H1 dK |
| --- | ---: | ---: | ---: | ---: |
| LN2起点double maxAbs | **.0118618792579** | **.0143292945277** | .00979156600529 | .00988647173366 |
| 実scaled inputに対するlocal maxAbs | **.0118598937988** | **.0143356323242** | .00979614257812 | .00988388061523 |
| .01 failures | 1 | 1 | 0 | 0 |
| local referenceとの最大word差 /2048 output words | 992 | 1024 | 992 | 1024 |

head selector transposeで正しいhalfへ配置する数学と、
標準HTPが受信FP32 wordsをlosslessに保存するかは別の問題だった。
sourceではその後head0/head1のdisjoint scattersをAddしてDQ/DKを構成する。
今回の診断はscatterまでで、後続Add/Projection gradientsを実行・認定しない。
**次Backward frontierはstandard Head Scatterという最終cohort観測**を保存する。
**DSP_BACKWARD_SCORE_SCALE_VERIFIED / FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIED**を認定する。

Reference-fedなのはLN2 X/dY、upstream residual、Wo/context、
Forward saved P/Q/K/V/Gate。実G1 Forwardがこれらを正確に生成した証明ではない。

## D. Coverage / saved state

| Correctness level | 今回の認定範囲 | Remaining boundary |
| --- | --- | --- |
| Local mathematical | QK / Scaleの同一operand ordered words、独立double core gateを各3 processで確認 | 未試験geometry/全入力一般化 |
| Native connected | 実LN1→HVX QKV→Heads→QK、実LN2→Softmax→Raw→Scale | frozen external boundaryより前 |
| Observer-free | 各graph自身のNoReceipt/noTap final accuracy、cross-variant words | production全graphへは未昇格 |
| Producer-fed | fragment内部のCustom producer→consumerを認定 | LN1 X / LN2 inputs / Forward saved statesはreference-fed |
| Full training | 今回は認定範囲外 | 全Forward/Backward/gradient/materialization/optimizer整合 |

新packageで既存HVX FFN2/QKVのdevice回帰も実行し、数学/word/observer-free gatesを確認した。
LN1/LN2、Scalar MatMul/Multiply、Softmax、HeadSplit、RawDQDKの既存契約も維持した。

| Forward state | Actual producer / backward consumer | Lifetime and remaining closure |
| --- | --- | --- |
| Qhead | LN1→Wq→selector / dKraw | ForwardからAttention backwardまでnative保存。今回はQKへのproducer接続を観測 |
| Khead | LN1→Wk→selector / dQraw | 同上。動的Kはstatic weightではない |
| Vhead | LN1→Wv→selector / dProbabilities | 今回もHVX QKV→Headsで生成するが、正しいP/PV連結は未認定 |
| Gate | LN1→Wg→Sigmoid→head selector / dContext Gate、gate derivative | producer accuracy未認定、Backwardではfrozen external |
| raw scores | Q×Kᵀ / 次Scale | Backwardの直接consumerはない。Q/Kから再計算可能 |
| scaled / masked | Multiply / causal Add | P生成の一時状態。保存省略/再計算の実装は今回変更なし |
| P | masked scores→Softmax(axis1,beta1) / PᵀdContext、Softmax product/final | production NATIVE、diagnostic APP_READ。Backward consumerまで保持が必要 |

Pをreference-fedからproducer-fedへ移すには、まずScaleのFAILを修復し、
causal mask→Softmax Pの独立数学/row normalization/causal/finite/accuracyと、
Tap/Receiptなしの実producer連結を検証する必要がある。
Pの新しいgateは今回は未設定・未評価。
将来は既存dScores .002、raw/scaled dQ/dK .01やdV精度予算を満たせるP誤差を、
Host sensitivity/referenceで事前登録する必要がある。
QK成功だけでP、Gate、PVが正しくなったとは認定しない。

Full LN1 dYには、scaled dQ/dK scatterとhead集約に加えて以下が残る。

- dV=Pᵀ×dContext、V scatter/集約、dWv。
- gate gradient: selected gradient×saved head contextのreduce、
  gate scatter/集約、Sigmoid derivative、dGateLogits、dWg、dLn1Gate。
- dWq/dWk=LN1Yᵀ×DQ/DK、DQ/DK/DV×Wq/Wk/Wvᵀ。
- source順のdLn1Q+dLn1K、続いてV、最後にGate branchのAdd。
- その全dYでのLN1 backwardとresidual gradient合流。
- FFN2/FFN1 backward、ReLU mask、LN2 dYの実producer、
  全gradient materialization/registry mapping、8-microbatch .125f accumulation。

今回のlocal LN1 backward dYは外部入力である。
従来dWo shadow accumulationの検証を、全gradientの8MB accumulationへ一般化しない。
dQ/dKだけでLN1 dYは再構築できない。
embedding/logits/loss、全19層のlifetime/scheduling、optimizer接続も未認定である。

## E. Performance

正当性のgate成立後に採取した限定的な費用測定。すべて同じ最終package/probe、
BACKGROUND_CORRECTNESS、各Lane3独立process。以下の範囲はprocess別medianのmin/maxである。
各graphのfirst executeを別記し、anchor＋2 warmups後9 steady samples/processを採取した。
exclusive benchmark、clock/power設定変更は0。過去binaryとのspeedup計算は行わない。

| Graph | DSP arithmetic ms | receipt-inclusive window ms | execute wall ms |
| --- | ---: | ---: | ---: |
| QK Standalone Observed | 1.981–1.984 | 2.139–2.142 | 4.787–5.128 |
| QK Standalone NoReceipt | 未観測 | 未観測 | 4.736–4.852 |
| LN1→HVX QKV→Heads→QK Observed | 1.980–1.981 | 2.131–2.133 | 6.580–7.070 |
| 同 NoReceipt/noTap | 未観測 | 未観測 | 6.295–6.682 |
| Scale Standalone Observed | .054–.055 | .236–.239 | 2.874–3.079 |
| Scale Standalone NoReceipt | 未観測 | 未観測 | 2.839–3.131 |
| LN2→Softmax→Raw→Scale Observed | .053–.055 | .228–.232 | 16.746–17.477 |
| 同 NoReceipt/noTap | 未観測 | 未観測 | 15.828–16.575 |

QKは2 heads合計の算術window、Scaleは4 outputs合計。
native Aのwallは**LN1 saved-state local backwardも含む**ためForward-only latencyではない。
native BはdWo branchも含む。
windowはfinite scans/copies等を含むがQHPI全invocationやRPC全体の時間とは一致しない。
NoReceiptの未観測timerを0へ置換していない。
wall−mathを純粋なbackend/RPC overheadとは呼ばない。

| NoReceipt/noTap graph | nodes/tensors | native logical B | total logical B | Host buffer lower bound B |
| --- | ---: | ---: | ---: | ---: |
| QK Standalone | 1/6 | 0 | 24576 | 41024 |
| LN1→HVX QKV→Heads→QK＋local B | 5/25 | 65664 | 148612 | 149152 |
| Scale Standalone | 1/9 | 0 | 32772 | 49288 |
| LN2 fused→Scale＋dWo | 11/47 | 90240 | 222604 | 198880 |

新QK/Scaleにexplicit tensor/heap scratch、packing、transpose bufferは0。
stack scalars/pointer arrays、既存LN stackやbackend workspaceをphysical peak0とはしない。
logical tensor payloadと2×input＋APP_READ outputs/guardsのHost下限であり、
VTCM/DDR traffic/allocator peak/RSS増分は分離未測定。

| Free graph | first execute ms | graph finalize ms |
| --- | ---: | ---: |
| QK Standalone | 4.891–5.344 | 27.316–45.828 |
| LN1→QK＋local B | 7.174–9.602 | 35.040–62.950 |
| Scale Standalone | 3.129–3.382 | 27.520–33.439 |
| LN2 fused→Scale＋dWo | 17.056–17.420 | 36.594–62.316 |

先行graphsでcontextがwarmになった後の各graph初回であり、cold boot測定ではない。
全6 processでthermal0→0、battery36°C→36°C、AC=true / USB=false / Wireless=false。
他app/scheduler/clockを固定したproduction throughputではない。

QK workは2×32×32×32=**65,536 MAC/layer/MB**。
19層1,245,184 MAC、19層×8MB **9,961,472 MAC**。
Scale workは2 heads×2 gradients×32×32=**4,096 multiplies/layer/MB**、
19層77,824、19層×8MB **622,592 multiplies**。
これらは算術量であり、wallの単純倍算をproduction速度へ転用しない。
今回HVX QKは実装していない。Scalar/HVX QK speedupは認定しない。

## F. Observer effect

新segmentではObserved/NoReceiptとFloat Tap有無を分け、
実際のfinal outputsのdouble accuracyとcross-variant全word一致を確認した
（各Laneの最終3 process）。
Observedのreceipt equality、NoReceiptのactual final accuracy、
process間digest一致は別々の検査である。

既存35-node/40-node incidentは **REPRODUCED / CAUSE UNRESOLVED** のまま。
新segmentの成功は旧graph内部原因の解決ではない。
大規模Tap bisectionは行わず、標準HTPの内部precision/packing/fusion機構を断定しない。
NATIVE FLOAT32宣言だけでは、今回の標準Scale/Scatterの精度は保証されなかった。

## G. Architecture decision

**次の主投資は1: Forward Score Scale→Causal Mask→Softmax P precision closure。**
今回の両Laneの最終cohortと新package回帰は完了した。

Forward Scaleは正しいCustom rawを受けても.01を超え、P生成を直接阻む。
PはdVとSoftmax backwardの双方が必要とするsaved stateで、
現Backwardのreference-fed境界を縮めるにはそのproducer closureが重要である。
Mask/Softmaxには新しい数学・row reduction・range/finite検証が必要であり、
今の小さい契約を保った段階的検証が適切。
Backward Scatterも実測FAILで次候補だが、その先にはdV/gate/projectionの未検証branchが多い。
wider integrationだけでは標準演算の誤差は修復されない。

| QK option | Assessment |
| --- | --- |
| A Scalar維持 | 1.981–1.984ms/paired QKという限定費用を観測。現在は正確性を優先し維持 |
| B Ordered HVX NT | 右Kは動的row-major。output-column lanesにはstrided/gather loadsかpretransposeが必要。packingなら2×4096=8192B/layer/MBの追加payload候補、保存Kとpacked Kのlifetimeを別監査。q順/独立rounding/N32 tilingの新検証が必要 |
| C Forward precision先行 | 実測Scale frontierを解消してP依存を閉じる。今回の主推薦 |

HvX NTの設計可能性を実装・性能成功へ昇格させない。
数値精度が未成立のまま、既存MatMul高速化だけを優先しない。
19層×8MBのpacking/storage/scheduling、registry/ABIの運用費用には
FP32_ARCHITECTURE_REASSESSMENT_REQUIREDを維持する。

## H. Final questions

| # | Answer |
| --- | --- |
| 1 Forward Scalar QK PASS? | YES、44-case core .01と3独立process |
| 2 正しいQ/Kに対するordered words? | YES、Standalone/Observed/Tapで全word一致。不可視値の直接観測とはしない |
| 3 LN1→HVX QKV→Heads→QK native? | PASS、実DSP/HVX producerのnative tensor接続 |
| 4 Observer-free raw accuracy? | PASS、各NoReceipt/noTap graph自身のfinal outputで独立判定 |
| 5 次Forward consumer? | 標準Scale Head1が1/44 FAIL、maxAbs.0102389901429 |
| 6 正しいBackward scale bits? | 0x3e3504f3、production construction、Host assertion、実receipt/受信bits logが一致 |
| 7 Custom Scale Standalone? | PASS、25-case core .01、全controlsのordered FP32 words一致 |
| 8 LN2→Softmax→Raw→Scale? | PASS、fused/separate両方、native chain全stageで既存gate維持 |
| 9 Observer-free scaled gradients? | PASS、そのvariant自身のdouble accuracyとcross-variant words |
| 10 次Backward Scatter? | H0 dQ/dKが各1/25 FAIL。H1は.01以内でもword lossあり |
| 11 Forward Pに残るもの? | Scale修復、Mask/Softmax producer accuracy、P保存/lifetimeとnative consumer連結。Gate/PVも別途 |
| 12 Full LN1 dYに残るもの? | scatter/集約、Q/K/V weight/input gradients、dV、Gate branch、source順Add、LN1 backward、residual合流 |
| 13 HVX FFN2/QKV回帰維持? | YES、新package/probeで実行し、Scalar/HVX数学・words・native/observer-freeとABIを確認 |
| 14 Scalar QK費用? | paired math1.981–1.984ms、free native wall6.295–6.682ms、local B込み |
| 15 精度修復かHVX高速化か? | 精度修復を優先。Forward Scale→Mask→Pを主推薦 |
| 16 Full1-layer不足? | P/Gate/PV/Wo/FFN/residual producers、Scatter/Projection/Gate/FFN gradients、全dY、materialization/8MB accumulation、lifetime/registry統合 |
| 17 40k→60k / 200k? | NO-GO、training updates0 |

| Required classification | Final disposition |
| --- | --- |
| DSP_FORWARD_QK_RAW_VERIFIED | VERIFIED、Standalone、44-case core .01 |
| FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIED | VERIFIED、reference-fed LN1 Xからのnative接続 |
| DSP_BACKWARD_SCORE_SCALE_VERIFIED | VERIFIED、正しいraw inputsとscale bits |
| FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIED | VERIFIED、正しいfrozen外部LN2/Forward state条件 |
| OBSERVER_FREE_PRECISION_VERIFIED | VERIFIED、両対象graph自身のNoReceipt/noTap final outputs |
| BLOCKED_FORWARD_CONSUMER | Standard Score Scale、Head1 .01 FAIL |
| BLOCKED_BACKWARD_CONSUMER | Standard Head Scatter、H0 dQ/dK .01 FAIL |
| FP32_ARCHITECTURE_REASSESSMENT_REQUIRED | 保存状態/FP32 ownership、registry/ABI/性能費用の拡大を検討 |
| UNRESOLVED | 旧observer機構、Full1-layer/19-layer correctness、未測定production scheduling |

全Attention Forward / Full1-layer / Full19-layer VERIFIEDは使用しない。

## I. Provenance

| Identity | Value |
| --- | --- |
| Branch | investigation/g1-fp32-qk-score-scale-frontier |
| Research base | 8e13b1ed1e2464b4aef790233682a11c43c838c5 |
| Final device-tested source | b6794c4572d10c163acb5201e811546ac564243e |
| DSP package SHA256 | 0b7251a67a69140c8688548b79d8e2f7d280ac74df6050b181841c63726c7c32 |
| Host package SHA256 | a252605cd7fea45190f65602b07563946f062683cd6fad763f5bdfffcdb3df7c |
| Final probe SHA256 | 002c90c918a621a9935496d01780473d39d75f376341923a172b34e9713fd600 |
| Hexagon compiler SHA256 | 039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047 |
| NDK compiler SHA256 | ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3 |
| Frozen inputs SHA256 | 388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f |
| LN1 parameters SHA256 | 7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f |

QAIRT2.48.40.260702151143、QNN API2.37.0、HTP V81、
Hexagon SDK6.6.0.0/tools19.0.07、NDK26.2.11394342を固定した。
compiler、SDK manifest、HVX headers/types、V81 PRM、compiler guideのhashを再照合した。
QNN graph configsは既存DiagnosticGraphのdefault config、explicit FLOAT32、
flat rank4、DDR-only QHPI signatures、HTP登録。CPU backend fallbackは0。
Host referenceは検証用であり、CPU完全不使用/NPU-only/QNN自動微分とは主張しない。

最終Aは各754 executes、8756 metrics、4686 word comparisons。
3 process共通timing-free numerical digest:
`1fab9e1327056516d9b1e4239656950029740787cd24693b6ec794770287a4d3`。
最終Bは各715 executes、12180 metrics、3120 word comparisons。
3 process共通digest:
`baa67163cd2a5ebf541d063132ffd185e4063339aed682155ca98835320f408b`。
合計4407 executes、62808 metrics、23418 word comparisons。
digestはordered metric/word/contract行のhashで、raw tensor SHAではない。
timing/counterを除外し、全word検査はprobe内で別実施する。
検査数はvariants/repetitionsを含み、その数の独立fixtureを意味しない。

全最終processでAPI errors0、finite、input mutation0、APP-visible poison/canary/complete writes、
Observed DSP markers、fallback0、context/device/backend teardown成功、focus takeover0。
3-process数値一致と、graph variant間のbitwise同値性を別々に検査した。
source/SDK/compiler/fixtures/QNN config/package/probeは全6 processで一致。
cohort開始後のcandidate再buildは0。

途中でawake check、続行指示後のADB_NO_ONLINE_DEVICEによる安全停止があった。
wake、endpoint推測、別端末接続、foreground takeoverは0。
ユーザーの再接続後、元の物理identity・awake・active run不在・容量・thermal等を確認して再開した。
停止前のpilot/部分2-processと停止時stagingはprivateに保存し、
**最終cohortは両Lane3 processを新規採取**した。旧recordsを最終3 processへ混在させていない。
再開sourceでのProbeOnly再buildは停止前candidateと全binary bytesが一致した。

| Validation | Result |
| --- | --- |
| Fast | PASS、candidate/closeoutおよび最終report確定後gateのlogsを保存 |
| Host profile | PASS、273.3s、contract/diagnostic＋parity battery |
| Qnn profile | PASS、399.6s、Host349.9s＋pinned core/API＋QNN build＋APK audit |
| Targeted Host | 既存15 suite＋新kernel140 calls＋新28 graphs/24 nodes PASS、最終graph再compile/retest PASS |
| Linked HVX FFN2/QKV codegen | 同じpackageで両audit PASS |
| Device smoke | 再開sourceでPASS、77.0s、BACKGROUND_CORRECTNESS |
| Muon W8 | 再開sourceで実際に114/114 PASS、maxAbs1.490116119e-8、relativeL2 4.40269854e-8、cosine1、RPC success/finite/fallback=false |
| Final independent cohort | PASS、A3/3、B3/3、各Laneのdigest一致 |
| New package HVX FFN2 regression | PASS、596 executes /1266 metrics /2158 word comparisons |
| New package HVX QKV regression | PASS、1388 executes /10086 metrics /17457 word comparisons |

Host/Qnn対象の数値source、SDK/APKの契約は最終candidateでも不変。
package/probe再build後のbytesを照合し、新packageのdevice回帰を新規実行した。
Qnn/APK auditはpinned core/API、arm64 ABI、V81 Stub/Skel hashes、host path不在、
2.47混入不在を検査。optional inventory advisoryだけを許容し、SDK fallbackは0。

旧device-tested source `a50da5c3a569d34fb6896c8fcffb3754dc7f2312`、
旧DSP `9736778fc192fd71c7ab974341294ec4f4cd05618473c699848318e9839c9b11`、
旧probe `9e28d2b960f6dbc9c2dc4f4c17b58de4f4024c9c603967682c504836c1af143d`
を照合し、旧bytesをprivateに保管した。新binaryの回帰は今回の実行結果で認定した。
40k checkpoint `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`
を含む既存保護identity計5件、前研究branch refsを最終実行後にも再確認した。
100k baseline、200k UNSTABLE、Step41647/42k evidence、dataset/tokenizer/cache/data order、
private numerical reportsを保持した。checkpoint decode、optimizer/codec変更、
training updates、Final test、LR探索は0。6031は **UNRESOLVED / DORMANT / WATCH**。

Private evidenceはignored `build/reports/g1-fp32-qk-score-scale-frontier/`。
allow-list exporterは正確なmetric/word coverage、source/binary/fixture、6独立process、
SDK/compiler、physical identity、同じpackageの旧HVX回帰、W8 raw結果と検証logsを検査する。
最終exportの48破損条件を通常Python/optimized modeの双方で拒否した。
停止時には別のBLOCKED aggregateを保存し、通常final exporterが未完cohortを拒否することも確認した。
raw tensors/checkpoints/corpus、device identifiers、APK hash、SDK binaries、
private logs、local absolute pathsは公開aggregate/commitへ含めない。

```powershell
.\scripts\test_qk_score_scale_host.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143
.\scripts\run_fp32_custom_probe.ps1 -Variant oneQkScaleA -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
.\scripts\run_fp32_custom_probe.ps1 -Variant oneQkScaleB -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
python scripts/export_public_qk_score_scale.py --private-root build/reports/g1-fp32-qk-score-scale-frontier --output docs/results/g1-fp32-qk-score-scale-frontier-summary.json --self-test
```

再現時はclean sourceと固定SDK/fixturesを用意し、物理同一性・active run保護・awake・容量・
thermal/battery・APK auditを満たしてからheadless実行する。安全条件は緩和しない。
Final report HEADは `git log -1 --format=%H -- docs/g1-fp32-qk-score-scale-frontier.md`
で解決し、exact HEADと最終clean statusをprivate closeoutと完了回答へ保存する。
push/PR/merge/main変更/history rewriteは0。
