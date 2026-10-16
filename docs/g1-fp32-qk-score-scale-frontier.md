# G1 FP32 QK raw score / backward score scale frontier

**BLOCKED / UNRESOLVED — 必須の最終認定は未完了。**
診断専用/default-offの2演算とHost検証を実装した。Forwardは最終candidateの
2独立process、Backwardはそれ以前のprobeによる1 pilotで、Standalone・native接続・
observer-freeの数値条件を満たした。しかしForward最終3回目の実行前に
`DEVICE_NOT_AWAKE_OR_UNKNOWN`で安全停止した。wake操作は行っていない。
その後のユーザー明示の続行指示に対する再開確認でも、host-side ADB inventoryが
`ADB_NO_ONLINE_DEVICE`で停止した。endpoint推測・別端末接続・device操作は行っていない。

**3-process × 2 Laneと新packageのHVX device回帰が未完了のため、
今回のVERIFIED分類は一つも確定しない。** 全Attention/full1-layer/full19-layerはUNVERIFIED。
Production backendとtraining updatesは変更せず、40k→60k / 200kは引き続きNO-GO。
[公開aggregate](results/g1-fp32-qk-score-scale-frontier-summary.json)もBLOCKEDを保持する。

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

最終source **e30b66c3980f3c4537282f227dd809b3a7a6299f** / 同一package/probeで
**2独立processのみ完了**。以下はその両processで一致した数値であり、3-process認定ではない。

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
局所・native・observer-freeの成功事実はあるが、必須cohortが欠けるため
DSP_FORWARD_QK_RAW_VERIFIED / FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIEDは確定しない。

Reference boundaryは正しい固定LN1 X、gamma/beta、QKV weights、local dY。
全G1 producer起点や全19層Forwardの正しさを認定しない。

## C. Backward score scale

以下はsource **d640e86064cff113aa963640c2a7468fe43d1463**、
旧probeによる**1 pilot**の結果である。
DSP/Host package bytesは最終candidateと同一だが、probe bytesは異なる。
最終probeは受信scale bits logとnext-raw監査/finalize計時を追加しており、
**最終candidateのLane B processは0件**。pilotをその代用にしない。

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
pilotでもreceipt内のscalar wordをprobeが検査した。
pilotには独立のreceived-bits log行がなく、最終probe用の直接log証拠と区別する。

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

| Standard Scatter pilot /25 | H0 dQ | H0 dK | H1 dQ | H1 dK |
| --- | ---: | ---: | ---: | ---: |
| LN2起点double maxAbs | **.0118618792579** | **.0143292945277** | .00979156600529 | .00988647173366 |
| 実scaled inputに対するlocal maxAbs | **.0118598937988** | **.0143356323242** | .00979614257812 | .00988388061523 |
| .01 failures | 1 | 1 | 0 | 0 |
| local referenceとの最大word差 /2048 output words | 992 | 1024 | 992 | 1024 |

head selector transposeで正しいhalfへ配置する数学と、
標準HTPが受信FP32 wordsをlosslessに保存するかは別の問題だった。
sourceではその後head0/head1のdisjoint scattersをAddしてDQ/DKを構成する。
今回の診断はscatterまでで、後続Add/Projection gradientsを実行・認定しない。
**次Backward frontierはstandard Head Scatterというpilot観測**を保存する。
DSP_BACKWARD_SCORE_SCALE_VERIFIED / FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIEDも未確定。

Reference-fedなのはLN2 X/dY、upstream residual、Wo/context、
Forward saved P/Q/K/V/Gate。実G1 Forwardがこれらを正確に生成した証明ではない。

## D. Coverage / saved state

| Correctness level | 今回確認できた範囲 | 未完了 |
| --- | --- | --- |
| Local mathematical | QK最終2 process、Scale pilotで同一operand ordered wordsとdouble core gate | 最終3-process両Lane |
| Native connected | LN1→HVX QKV→Heads→QKの実native接続、LN2→Scaleはpilot | Lane B最終probe |
| Observer-free | 各実行graph自身のNoReceipt/noTap final accuracyとcross-variant words | 必須最終cohort/新HVX device回帰 |
| Producer-fed | fragment内のCustom producer→consumerのみ | frozen external boundaryより前の実G1 producer |
| Full training | 今回は認定範囲外 | 全Forward/Backward/gradient/materialization/optimizer整合 |

以前のLN1、LN2、HVX QKV/FFN2、Softmax、RawDQDK等の研究成果は保持した。
今回の新packageに対するdevice回帰は安全停止で未実行であり、
過去binaryのPASSを新binaryのPASSへ転用しない。

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

以下はaccuracy/word gate成立後だけ採取した**未確定の費用観測**。
正式な速度改善認定はしない。
Aは最終2 processのmedian範囲、Bは異なるprobeによる1 pilotであり、
両者や過去研究binaryとのspeedup計算は行わない。
各graphは別途first executeを記録し、anchor＋2 warmups後9 steady samples。
exclusive benchmark、clock/power変更は0。

| Graph / evidence | DSP arithmetic ms | receipt-inclusive window ms | execute wall ms |
| --- | ---: | ---: | ---: |
| QK Standalone Observed / A2 | 1.975–1.979 | 2.132–2.136 | 5.135–5.151 |
| QK Standalone NoReceipt / A2 | 未観測 | 未観測 | 4.950–5.068 |
| LN1→HVX QKV→Heads→QK Observed / A2 | 1.980 | 2.134–2.135 | 7.061–7.099 |
| 同 NoReceipt/noTap / A2 | 未観測 | 未観測 | 6.641–6.752 |
| Scale Standalone Observed / pilot | .053 | .240 | 3.055 |
| Scale Standalone NoReceipt / pilot | 未観測 | 未観測 | 2.943 |
| LN2→Softmax→Raw→Scale Observed / pilot | .054 | .230 | 17.284 |
| 同 NoReceipt/noTap / pilot | 未観測 | 未観測 | 16.180 |

QKは2 heads合計の算術window。Scaleは4 outputs合計。
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

A free Standalone first wall 5.594–5.668ms、finalize47.379–48.224ms。
A free native first8.710–10.279ms、finalize56.616–76.547ms。
B free Standalone first3.525ms、finalize42.136ms。
B free native first20.097ms、**pilotではnative finalize未計時**。
最終probeには計時を追加したがBを実行できていない。cold boot測定ではない。

A2はthermal0→0、battery33°C→33°C、
B pilotはthermal0→0、battery34°C→34°C。
いずれもAC/USB/Wireless powered=false。自然なdevice条件を記録したもので、
他app/scheduler/clockを制御したproduction throughputではない。

QK workは2×32×32×32=**65,536 MAC/layer/MB**。
19層1,245,184 MAC、19層×8MB **9,961,472 MAC**。
Scale workは2 heads×2 gradients×32×32=**4,096 multiplies/layer/MB**、
19層77,824、19層×8MB **622,592 multiplies**。
これらは算術量であり、wallの単純倍算をproduction速度へ転用しない。
今回HVX QKは実装していない。

## F. Observer effect

新segmentではObserved/NoReceiptとFloat Tap有無を分け、
実際のfinal outputsのdouble accuracyとcross-variant全word一致を確認した
（A最終2 process、Bはpilot限定）。
Observedのreceipt equality、NoReceiptのactual final accuracy、
process間digest一致は別々の検査である。

既存35-node/40-node incidentは **REPRODUCED / CAUSE UNRESOLVED** のまま。
新segmentの成功は旧graph内部原因の解決ではない。
大規模Tap bisectionは行わず、標準HTPの内部precision/packing/fusion機構を断定しない。
NATIVE FLOAT32宣言だけでは、今回の標準Scale/Scatterの精度は保証されなかった。

## G. Architecture decision

**次の主投資は1: Forward Score Scale→Causal Mask→Softmax P precision closure。**
ただし最初に今回の安全停止を解消できる実行条件で、両Laneの最終cohortと
新package回帰を完了する必要がある。

Forward Scaleは正しいCustom rawを受けても.01を超え、P生成を直接阻む。
PはdVとSoftmax backwardの双方が必要とするsaved stateで、
現Backwardのreference-fed境界を縮めるにはそのproducer closureが重要である。
Mask/Softmaxには新しい数学・row reduction・range/finite検証が必要であり、
今の小さい契約を保った段階的検証が適切。
Backward Scatterも実測FAILで次候補だが、その先にはdV/gate/projectionの未検証branchが多い。
wider integrationだけでは標準演算の誤差は修復されない。

| QK option | Assessment |
| --- | --- |
| A Scalar維持 | 約1.98ms/paired QKという限定費用を観測。現在は正確性を優先し維持 |
| B Ordered HVX NT | 右Kは動的row-major。output-column lanesにはstrided/gather loadsかpretransposeが必要。packingなら2×4096=8192B/layer/MBの追加payload候補、保存Kとpacked Kのlifetimeを別監査。q順/独立rounding/N32 tilingの新検証が必要 |
| C Forward precision先行 | 実測Scale frontierを解消してP依存を閉じる。今回の主推薦 |

HvX NTの設計可能性を実装・性能成功へ昇格させない。
数値精度が未成立のまま、既存MatMul高速化だけを優先しない。
19層×8MBのpacking/storage/scheduling、registry/ABIの運用費用には
FP32_ARCHITECTURE_REASSESSMENT_REQUIREDを維持する。

## H. Final questions

| # | Answer |
| --- | --- |
| 1 Forward Scalar QK PASS? | 完了した最終2 processでは44-case gate PASS。3-process最終認定はBLOCKED |
| 2 正しいQ/Kに対するordered words? | YES、Standalone/ObservedおよびTapで全word一致。不可視値は直接観測としない |
| 3 LN1→HVX QKV→Heads→QK native? | 完了2 processでPASS、最終分類UNRESOLVED |
| 4 Observer-free raw accuracy? | 各NoReceipt/noTap graph自身でPASS、最終cohort未完 |
| 5 次Forward consumer? | 標準Scale Head1が1/44 FAIL、maxAbs.0102389901429 |
| 6 正しいBackward scale bits? | 0x3e3504f3、production float constructionとHost assertion。pilot receipt wordsも一致 |
| 7 Custom Scale Standalone? | 25-case coreと全wordはpilot PASS。最終probe未実行 |
| 8 LN2→Softmax→Raw→Scale? | fused/separate両方pilot PASS。最終probe未実行 |
| 9 Observer-free scaled gradients? | pilotの自身final outputでPASS、最終3 processは未完 |
| 10 次Backward Scatter? | pilotでH0 dQ/dKが各1/25 FAIL。H1は.01以内でもword lossあり |
| 11 Forward Pに残るもの? | Scale修復、Mask/Softmax producer accuracy、P保存/lifetimeとnative consumer連結。Gate/PVも別途 |
| 12 Full LN1 dYに残るもの? | scatter/集約、Q/K/V weight/input gradients、dV、Gate branch、source順Add、LN1 backward、residual合流 |
| 13 HVX FFN2/QKV回帰維持? | Hostとlinked codegen PASS。新packageのdevice回帰はBLOCKED、過去PASSで代用しない |
| 14 Scalar QK費用? | A2でpaired math1.975–1.979ms、free native wall6.641–6.752ms、local B込み。正式性能認定ではない |
| 15 精度修復かHVX高速化か? | 精度修復を優先 |
| 16 Full1-layer不足? | 両Lane認定完了、P/Gate/PV/Wo/FFN/residual producers、Scatter/Projection/Gate/FFN gradients、全dY、materialization/8MB accumulation、lifetime/registry統合 |
| 17 40k→60k / 200k? | NO-GO、training updates0 |

| Required classification | Current disposition |
| --- | --- |
| DSP_FORWARD_QK_RAW_VERIFIED | UNRESOLVED、最終2/3 process |
| FP32_FORWARD_QK_NATIVE_SEGMENT_VERIFIED | UNRESOLVED、同上 |
| DSP_BACKWARD_SCORE_SCALE_VERIFIED | UNRESOLVED、pilotのみ |
| FP32_BACKWARD_SCORE_SCALE_SEGMENT_VERIFIED | UNRESOLVED、pilotのみ |
| OBSERVER_FREE_PRECISION_VERIFIED | UNRESOLVED、必要な最終cohort未完 |
| BLOCKED_FORWARD_CONSUMER | Standard Score Scaleの実測FAIL、最終2 processで再現 |
| BLOCKED_BACKWARD_CONSUMER | Standard Head Scatterのpilot FAIL、最終確認未完 |
| FP32_ARCHITECTURE_REASSESSMENT_REQUIRED | 追加FP32 ownership、保存状態、registry/性能費用の検討が必要 |
| UNRESOLVED | safety停止、最終cohort、新package HVX回帰、旧observer内部機構 |

## I. Provenance

| Identity | Value |
| --- | --- |
| Branch | investigation/g1-fp32-qk-score-scale-frontier |
| Research base | 8e13b1ed1e2464b4aef790233682a11c43c838c5 |
| Candidate device-tested numerical source | e30b66c3980f3c4537282f227dd809b3a7a6299f（Lane A 2 process） |
| Pilot source | d640e86064cff113aa963640c2a7468fe43d1463 |
| DSP package SHA256 | 0b7251a67a69140c8688548b79d8e2f7d280ac74df6050b181841c63726c7c32 |
| Host package SHA256 | a252605cd7fea45190f65602b07563946f062683cd6fad763f5bdfffcdb3df7c |
| Candidate probe SHA256 | 002c90c918a621a9935496d01780473d39d75f376341923a172b34e9713fd600 |
| Pilot probe SHA256 | 67ceaedbabd1fcdf66211ae426edfbeb8175204247cf59dda327fbd9770d6959 |
| Hexagon compiler SHA256 | 039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047 |
| NDK compiler SHA256 | ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3 |
| Frozen inputs SHA256 | 388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f |
| LN1 parameters SHA256 | 7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f |

QAIRT2.48.40.260702151143、QNN API2.37.0、HTP V81、
Hexagon SDK6.6.0.0/tools19.0.07、NDK26.2.11394342を固定した。
compiler、SDK manifest、HVX headers/types、V81 PRM、compiler guideのhashを再照合した。
QNN graph configsは既存DiagnosticGraphのdefault config、explicit FLOAT32、
flat rank4、DDR-only QHPI signatures、HTP登録。CPU backend fallbackは0。
Host reference計算は検証用であり、CPU完全不使用/NPU-only/QNN自動微分とは主張しない。

最終A2は各754 executes、8756 metrics、4686 word comparisons。
共通timing-free numerical digest:
`1fab9e1327056516d9b1e4239656950029740787cd24693b6ec794770287a4d3`。
B pilotは715 executes、12180 metrics、3120 word comparisons。
pilot digest:
`ea9559a5df8db1960f605739aa0264a70d450e336e9601048628635c90eb78be`。
digestはordered metric/word/contract行のhashで、raw tensor SHAではない。
timing/counterを除外し、全word検査はprobe内で別実施する。
検査数はvariants/repetitionsを含み、その数の独立fixtureを意味しない。

完了processはAPI errors0、finite、input mutation0、APP-visible poison/canary/complete writes、
Observed DSP marker、fallback0、context/device/backend teardown成功、focus takeover0。
A最終3回目は二段目のawake checkで拒否され、graph execute/teardownの成功記録は存在しない。
staging先の掃除やwakeも含め、停止後のdevice操作は0。
明示再開後もhost-side inventoryのADB_NO_ONLINE_DEVICEで止まり、端末への操作は0。
pilot、完了最終2 process、実行前停止を別保存した。freeze後candidate再buildは0。

| Validation | Result |
| --- | --- |
| Fast | PASS、89.7s / candidate59.2s / closeout61.2s |
| Host profile | PASS、273.3s、contract/diagnostic＋parity battery |
| Qnn profile | PASS、399.6s、Host349.9s＋pinned core/API＋QNN build＋APK audit |
| Targeted Host | 既存15 suite＋新kernel140 calls＋新28 graphs/24 nodes PASS、最終graph再compile/retest PASS |
| Linked HVX FFN2/QKV codegen | 新packageで両audit PASS、device executionとは別 |
| Device smoke | 今回sourceでPASS、93.5s、BACKGROUND_CORRECTNESS |
| Muon W8 | 今回sourceで実際に114/114 PASS、maxAbs1.490116119e-8、relativeL2 4.40269854e-8、cosine1、RPC success/finite/fallback=false |
| Final independent cohort | **BLOCKED**、A2/3、B0/3 |
| New package HVX FFN2/QKV device regression | **BLOCKED**、安全停止後のため未実行 |

旧device-tested source `a50da5c3a569d34fb6896c8fcffb3754dc7f2312`、
旧DSP `9736778fc192fd71c7ab974341294ec4f4cd05618473c699848318e9839c9b11`、
旧probe `9e28d2b960f6dbc9c2dc4f4c17b58de4f4024c9c603967682c504836c1af143d`
を照合し、旧bytesをprivateに保管した。新candidate PASSの代用にしない。
40k checkpoint `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`
とtokenizer/cache/manifest/run-planを含む既存保護identity計5件、前研究branch refsを再確認した。
100k baseline、200k UNSTABLE、Step41647/42k evidence、private numerical reportsを保持した。
checkpoint decode、corpus/data order/optimizer/codec変更、training updates、Final testは0。
6031は **UNRESOLVED / DORMANT / WATCH**。

Private evidenceはignored `build/reports/g1-fp32-qk-score-scale-frontier/`。
allow-list blocked exporterは正確なmetric/word coverage、source/binary/fixture、
safety停止、未完cohort、今回のW8/検証logsを検査する。
8 numerical/coverage/teardown破損条件を通常Pythonとoptimized modeで拒否する。
通常のfinal exporterも未完cohortを拒否し、VERIFIEDへ昇格しない。
raw tensors/checkpoints/corpus、device identifiers、APK hash、SDK binaries、
private logs、local absolute pathsは公開aggregate/commitへ含めない。

再開には、元の物理端末へのonline接続、awake、他の安全条件が成立する必要がある。
新しいclean source commitと同一candidate bytesのidentityを固定して
**両Laneの3独立processを新規cohortとして採取し、新packageでoneHvx/oneQkvHvx回帰を実行する**。
現A2/pilotを新しい成功cohortへ混在させない。
gate緩和、wake/foreground takeover、training再開は再開手順に含まない。

```powershell
# Host-only reproduction:
.\scripts\test_qk_score_scale_host.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143

# Device reproduction: only after safety conditions and explicit resumption.
.\scripts\run_fp32_custom_probe.ps1 -Variant oneQkScaleA -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
.\scripts\run_fp32_custom_probe.ps1 -Variant oneQkScaleB -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot

# Current incomplete evidence must remain BLOCKED:
python scripts/export_public_qk_score_scale_blocked.py --private-root build/reports/g1-fp32-qk-score-scale-frontier --output docs/results/g1-fp32-qk-score-scale-frontier-summary.json --self-test
```

Final report HEADは `git log -1 --format=%H -- docs/g1-fp32-qk-score-scale-frontier.md`
で解決する。exact HEADと最終clean statusはprivate closeoutと完了回答へ保存する。
