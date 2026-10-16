# Ordered FP32 HVX Q/K/V projection feasibility

**K64/N64 NNへの一般化は成立した。** 同一package/probeのScalar/HVXを3独立processで
比較し、Q/K/Vのordered-FP32全word一致、double accuracy、native LN1→QKV→HeadSplit、
observer-free final accuracyを確認した。Standaloneの3 projection合計DSP mathは
**35.11–35.51倍**、observer-free LN1→QKV→HeadSplit＋local Backwardのexecute wallは
**3.31–3.40倍**。各processが事前の1.20倍／1.10倍gateを満たした。

新packageでのFFN2回帰、Host、Qnn、Device smoke、Muon W8 **114/114**もPASS。
次の主投資は**D: Precision Frontier Closure**を推奨する。未成立のForward QKᵀ等が
full1-layer correctnessを阻んでおり、速度だけで追加NN移植を優先しない。
Production backendは変更せず、40k→60k／200k trainingは引き続き**NO-GO**。

## A. Source and ABI audit

Branch `investigation/ordered-fp32-hvx-qkv`、base
`2a59e9b7c208597cbdb597b5ac7c290ff2624eac`。指定HEAD・clean tree・upstreamなしから開始。
既知remote refsに未到達のlocal95 commitsと、以前の15 branch refsを保持した。
fetch、main再構築、reset/rebase/amend、history rewrite、push/PR/mergeは0。

`qnn_runtime_transformer_training_generalized.inc`のLN1 Forward outputをQ/K/Vが共有する。
各weightは`LayerParameters::wq/wk/wv`、parameter metadataではPER_LAYER・MUON・MODEL×MODEL。
三つのMatMulはtranspose引数なしで、`addMatMulNode`へNNとして渡る。
logical `[32,64] × [64,64]`を診断ABIでは`[1,1,32,64] × [1,1,64,64]`へbackfillする。
row-major indexingは`Y[m*64+q]`と`W[q*64+n]`。weight identityとQ/K/V順序を維持した。

```text
acc = +0.0f
for q = 0 .. 63:
    product = round32(Y[m,q] * W[q,n])
    acc = round32(acc + product)
O[m,n] = acc
```

productionでは次に各head selector `[64,32]`とのMatMul、その後QKᵀ、scale、mask、Softmax、
PVへ進む。今回の診断は既存lossless `QKVHeads`を使い、rowごとのchannels0–31／32–63を
各`[32,32]`へcopyする。production selectorやAttention数学は変更していない。
LN1 Yにはgate projectionも依存するため、将来のfusionでYを消せるとは限らない。

既存`ProjectionQKV`／`ProjectionQKVNoReceipt`、scalar数学、旧receipt/counterを維持した。
旧39 Op名と順序、QHPI interface1.4を保持し、diagnostic-only/default-offの
`ProjectionQKVHvx`／`ProjectionQKVHvxNoReceipt`を末尾へ追加、計41名。
4 inputs・3 FP32 outputs、ObservedだけINT receiptを一つ追加する。
三つのprojectionを一つのCustom invocation内で順に実行し、別launch方式は実装していない。

HVX markerは`0x48383115`、Scalar QKVは`0x4838310d`。HVXに独立counterを設けた。
新receiptは24 header＋14336 input＋6144 output words、82016B。
旧Scalar receipt81952Bより64B大きい。NoReceiptにはINT tensorがない。
HostからHVX kernelを呼ぶとfatalとなり、scalar emulationへfallbackしない。

## B. HVX implementation and linked ELF

`hvx_ordered_nn.h::ordered_hvx_nn<K>`を共有し、K64/K128、M32/N64に明示的に制限した。
既存FFN2翻訳単位のstrict IEEE設定を再利用する。3 QKVは同じK64 helperを3回呼ぶ。
Yのregister reuseを3 projectionへ広げる追加fusionは行っていない。

128-byte vector modeはcompile-timeの`__HVX_LENGTH__`／`sizeof(HVX_Vector)`で検査した。
FP32 32 lanes、N64は2 tiles、各laneが一つのoutput columnを担当する。
増加qのaccumulator依存鎖を維持し、tree reduction・FMA・拡張accumulator持ち越しはない。

```text
-mv81 -mhvx -mhvx-length=128B -mhvx-qfloat
-mllvm -enable-xqf-gen=true -mllvm -hexagon-qfloat-mode=strict-ieee
-O2 -G0 -fPIC -ffp-contract=off -fno-fast-math
-fno-vectorize -fno-slp-vectorize -fno-unroll-loops
```

固定Hexagon SDK6.6.0.0、tools19.0.07のcompiler guideはstrict-ieeeをIEEE-754準拠modeとして
記載する。guide/header/types/manifest/V81 HVX PRMの5 hashesを前研究と再照合した。
SDK headerの`Q6_Vsf_vmpy_VsfVsf`はCVI_VX_DV、`Q6_Vsf_vadd_VsfVsf`はCVI_VX、slots2/3。
flagsはHVX翻訳単位だけに適用し、既存Scalar翻訳単位のbuild flagsを維持した。

K64 linked symbol `hexatrain_ordered_hvx_qkv_nn64`を独立に監査した。

| Static instruction evidence | Result |
| --- | ---: |
| linked ELF target | V81 |
| vector multiply sites | 4 |
| vector add sites | 4 |
| FP32 rounding conversions | 8 |
| SF→qf normalization sites | 8 |
| broadcasts | 4 |
| unaligned vector memory sites | 5 |
| FMA / call / stack spill in target | 0 / 0 / 0 |

これは動的命令数ではない。qf32 product→SF conversionと、SF operandsのadd→SF conversionが
別々にあり、積とaccumulationを各qでFP32へ丸める。prologue、61-trip hardware loop、
epilogueで64 productsを処理する。Aは4B、weightは256Bずつ進み、2 tilesと32 rowsを走査する。
29 vector register名v0–v28を参照するが、peak live数やoccupancyを実測した値ではない。
Scalar address/word-load/loop制御とsoftware-pipeline nopsは残る。

`audit_ordered_hvx_qkv.py`は既存FFN2 ELF auditを呼び、重複したdisassembler実装を作らない。
V81、multiply/add、conversion、broadcast/vmemu、trip count/strides、FMA/call/spill不在を検査する。
QHPI_RESOURCE_HVX宣言だけで成功とせず、同じpackageのDSP markerと実出力も確認した。
compiler-only auditのexecution欄は未判定のまま保存し、cohortとの統合判定をaggregateで行う。
subnormal/roundingはcompiler説明だけで認定せず、実機のfinite controlsでword比較した。

Y、weights、outputsのrow strideは256B。exact-size128B `memcpy`を`vmemu`へlowerし、
未確認の128B pointer alignmentをC++上で仮定しない。全read/writeはtile内、column tailなし。
ObservedのY/Wq/Wk/Wv/Q/K/V mod128は全件0だった。NoReceiptの不可視pointerのalignmentを
直接測定したとは主張しない。outputとinput/output/receiptの危険なoverlapはfatalにする。
explicit scratch、heap allocation、transpose、prepack、alignment stagingは0。
QHPI wrapperのallocframe operandはObserved HVX152B／Scalar120B、NoReceiptは双方96B。
core本体はframeless。wrapperにもvector stack accessはなく、意図したqhpi/memcpyと
3 arithmetic helper calls以外のcallはない。これはframe operandの監査で、callee/QuRTを含む
total stack usageや既存LN stack/backend workspaceの物理増分は分離未測定。

共通helper化後のFFN2は、returnまでの65 instruction wordsが旧binaryと全byte一致した。
symbol抽出末尾に追加された3 wordsはreturn後のalignment nopsである。
加えて新package/probeで既存`oneHvx`を実行し、596 executes、1266 metrics、2158 word比較、
standalone/native/observer-freeの全gate PASSを確認した。旧binaryの成功を新binaryへ流用していない。

## C. Numerical accuracy and three-process reproducibility

既存の24 source-derived/frozen cases＋20 LN pathological cases、計44を保持した。
同じfrozen input fileとLN1 parameter fileを使い、checkpoint decodeは行っていない。
さらに13独立controlsをstandaloneへ追加、計57 casesとした。
ordinary、constant、low variance、large offset、mixed signs、cancellation、sparse、
fixed random、QKV asymmetric、signed zero、subnormal finite、safe large/small、
transpose-sensitive indexingを含む。seed `0x08164123`、LCG1664525/1013904223。
NaN/infinity/unsafe overflowはHost negative testsへ分離した。

`one::mm`の独立double analyticとvolatile product/addによるHost ordered FP32を使用する。
Custom dotやHVX helperをreferenceから呼ばない。各caseでQ/K/V各2048、計6144 output wordsを
signed zeroも区別して比較した。Scalar/HVX、同じobserver条件、Observed/NoReceipt、Tap有無の
必須word比較はすべて0 mismatch。実operandが見える条件のHost ordered MatMul比較も全word一致。

| QKV output | Standalone double maxAbs | Native chain double maxAbs | Gate |
| --- | ---: | ---: | ---: |
| Q | 2.47147319188e-6 | 2.4316564895e-6 | .002 |
| K | 2.3994641607e-6 | 2.77017446315e-6 | .002 |
| V | 5.09577405872e-6 | 5.7436790506e-6 | .002 |

Standaloneは供給したFP32 Y wordsに対するdouble MatMul、chainはdouble LN1 Yからの解析式。
reference境界が異なるため、Qの二つの最大値の差を回帰と扱わない。
13 Host controlsは26 Scalar callsでmaxAbs `4.07919287682e-7`、word/guard/immutability PASS。
HVX Host実行拒否もPASSした。Host mockによるHVX演算成功は主張しない。

| Across final cohort | Worst case RMS | Worst relative L2 |
| --- | ---: | ---: |
| Q | 3.36618342726e-7 | 1.95284792319e-7 |
| K | 3.92429089394e-7 | 1.94454865349e-7 |
| V | 7.89937860959e-7 | 2.03847628456e-7 |

これはcase別metricの最大であり、全要素をpoolしたRMSではない。max-relativeの分母floorは.001。
全case/observer条件の集計と分布は[公開aggregate](results/ordered-fp32-hvx-qkv-summary.json)に保存した。

最終cohortは同じsource/package/probeの3独立process。各1388 executes、10086 metric checks、
17457全word比較、合計4164 executes／30258 metrics／52371比較。
反復とobserver variantsを含む検査数であり、その数の独立fixtureを意味しない。
API errors0、finite PASS、input mutation0、APP-visible complete-write/poison/canary PASS、
CPU fallback0、teardown成功、foreground takeover0。

3 process共通のtiming-free canonical numerical digest:
`b307c1d1c193ba6edd0bcb8557ca81b460ff5c7410220cd075cc30e96f5221b2`。
対象はonemetric、hvxwords、qkvgate、summary行。timing、counter、alignment、IDs/pathsを除く。
raw tensor自体のSHA256ではなく、全word検査はprobe内で別に行う。

初期pilotでは不可視NoReceipt YをHost LN oracleとbitwise比較する検査上の誤りがあり、
Scalar版を含む計31294 word差を出してintegration/performanceを停止した。
Host LN oracle自体はDSP LNと全word一致する契約ではない。修正は不可視oracleのword gateを
除くことだけで、double gateと実際のScalar/HVX/observer final word gatesを維持した。
不可視oracleとの差は診断metricとして残し、MatMulのsame-operand word failureと混同しない。
修正後probeを再buildし、pilot2件を最終3 processから明示的に除外した。
最終candidateの数学・精度gateを緩めたものではない。

## D. Native LN1, lossless heads and observer-free execution

Stage0はstandalone、stage1はDSP LN1→QKV＋saved-state local Backward、stage2は
DSP LN1→QKV→既存QKVHeads＋local Backward。各stageでScalar/HVX × Observed/NoReceipt ×
Tap有無の8 variantsを用意し、native stagesは各44 casesで独立に判定した。
X、gamma/beta、weights、dYは正しい固定外部入力である。
stage1/2へHost Yや理想QKVを途中注入していない。

No-TapではY/xhat/inverseをNATIVEにし、stage2のQ/K/VもNATIVE。
APP_WRITE inputsとAPP_READ final heads/dX/dGamma/dBetaを維持する。
saved xhat/inverseをlocal Backwardまで保持する。**dYはreference-fed**であり、
全Attention backward由来dYやfull1-layer producer-fed correctnessは認定しない。

| LN / local Backward | Double maxAbs | Gate |
| --- | ---: | ---: |
| Y | 5.81823561774e-7 | .002 |
| xhat | 6.66749372158e-7 | .002 |
| inverse | 1.94379752543e-5 | .002 |
| dX | 1.53640447422e-5 | .02 |
| dGamma | 1.96494390394e-5 | .02 |
| dBeta | 7.89668411016e-6 | .02 |

Q/K/Vの各headは対応projection sliceとbitwise一致。Head0/1最大誤差は、Q
`1.67515014926e-6 / 2.4316564895e-6`、K `2.34935266796e-6 / 2.77017446315e-6`、
V `5.7436790506e-6 / 4.65698120244e-6`、すべて.002以内。

ObservedではQKV input/output receiptを全words照合した。TapありではLN1 Y APP_READと
QKV consumer receiptを直接照合する。既存LN receiptにはYそのものは含まれず、No-Tap
Observedでは保存xhat・gamma/betaの独立FP32 affine再構成とconsumer Y全wordsを照合する。
これを直接のLN Y receipt観測と呼ばない。xhat/inverseのForward/Backward receipt一致、
dY/gamma、HeadSplit consumer inputsとoutputsも全words検査した。

NoReceipt/no-Tapはそのvariant自身の最終QKVまたはheads/dX/dGamma/dBetaをdoubleと比較し、
同じobserver条件のScalar版およびObserved anchorと全words比較した。
不可視Y/QKV/saved stateを直接観測したとは主張しない。LN Host oracleとのFP word差が
残る診断欄と、実Scalar/HVX必須word mismatch0を分けて保存した。
既存35-node Float Tap incidentは**REPRODUCED / CAUSE UNRESOLVED**のまま。

Optional frozen FFN2→Add→LN1→QKV接続は設計評価だけとした。必須Laneの固定X契約と
weight identityを維持し、別fragmentのfixture結合や大きなownership変更を追加していない。
前層ReLU/Residual1や全19層のproducer correctnessを今回の成功から推定しない。

## E. Paired performance and memory

同じ最終binary、fixture0、BACKGROUND_CORRECTNESS。各graph初回を記録し、各variant2 warmups後、
ABBA/BAAB交互8 blocks、**各variant16 steady samples/process**。
全accuracy/word gates成立後だけ性能へ進み、事前kernel1.20倍／segment1.10倍基準は不変。

| Process | Scalar QKV math ms | HVX QKV math ms | kernel ratio | Scalar free segment ms | HVX free segment ms | segment ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.238 | .3165 | 35.507× | 17.263 | 5.222 | 3.306× |
| 2 | 11.236 | .3200 | 35.113× | 17.195 | 5.175 | 3.323× |
| 3 | 11.234 | .3170 | 35.438× | 17.235 | 5.074 | 3.397× |

mathはstandalone Observedの3 projection合計で、load/broadcast/arithmetic/storeと3 helper callsを
含む。finite scansやreceipt copiesは除く。絶対短縮は**10.916–10.922ms**。
segmentはNoReceipt/no-Tap stage2で、LN1 Forward/HeadSplit/**local Backwardも含む**。
Forward-only latencyではなく、絶対短縮は**12.020–12.161ms**。

| Stage / instrumentation | Scalar steady wall ms | HVX steady wall ms |
| --- | ---: | ---: |
| Standalone / Observed | 15.187–15.257 | 4.172–4.271 |
| Standalone / NoReceipt | 15.515–15.904 | 3.919–4.181 |
| LN1→QKV＋local B / Observed | 15.988–16.139 | 4.995–5.146 |
| LN1→QKV＋local B / NoReceipt | 15.430–15.752 | 4.426–4.703 |
| LN1→QKV→Heads＋local B / Observed | 17.608–17.754 | 5.518–5.653 |
| LN1→QKV→Heads＋local B / NoReceipt | 17.195–17.263 | 5.074–5.222 |

範囲はprocess別medianのmin/maxであり、poolしたmedianではない。
HVX receipt-inclusive QKV windowはstandalone .849–.868ms、stage1 .960–.986ms、
stage2 .960–.969ms。windowはtimerで囲んだfinite/copy等を含むが、事前shape/alias検査や
一部header書き込みを含むQHPI全invocation時間ではない。Scalarのstage2 mathは
12.365–12.374msで、standaloneと同じ値ではなかった。異なるstage間の単純差をop costとしない。

| Free stage2 process | Scalar p10/p90 ms | HVX p10/p90 ms | Scalar min/max ms | HVX min/max ms |
| --- | --- | --- | --- | --- |
| 1 | 17.172 / 17.547 | 4.927 / 5.880 | 16.821 / 17.696 | 4.869 / 6.557 |
| 2 | 16.906 / 18.056 | 4.851 / 5.340 | 16.215 / 19.219 | 4.675 / 5.482 |
| 3 | 16.999 / 17.975 | 4.887 / 5.497 | 16.790 / 18.743 | 4.790 / 7.183 |

stage2 block paired ratio最小は2.838×。ABBA中央値は3.152/3.380/3.477×、
BAABは3.327/3.322/3.314×で逆転なし。kernel block最小も35.052×。
全3 processでthermal0→0、battery36°C→36°C、AC=true／USB=false／wireless=false。
clock/scheduler/他appは制御しておらず、exclusive benchmark、foreground takeover、強制冷却は0。
tail variationをproduction schedulingの保証へ転用しない。

free stage2の当該graph初回wallはScalar17.463–20.952ms、HVX5.018–5.905ms。
先行accuracy graphsでcontextがwarmになった後の初回であり、cold boot latencyではない。
finalize wrapperはScalar53.854–73.711ms、HVX45.337–77.790ms。finalize高速化は認定しない。
Host run内allocation/snapshot/copy等のsteady medianはScalar115.1–116.5µs、HVX116.7–117.2µs。
inputs() map構築、Host references、個別input validation/finite scanの時間は分離未測定。
QNN API wallとlogging wrapper込みwallを別保存した。wall−mathを純粋なRPC/QNN overheadとしない。
NoReceiptのDSP timerは未観測で、aggregateはnullとし0µsへ置き換えない。

| NoReceipt/no-Tap graph | nodes/tensors | native FP tensors/bytes | logical bytes | Host buffer lower bound B |
| --- | ---: | ---: | ---: | ---: |
| QKV | 1 / 7 | 0 / 0 | 81920 | 139360 |
| LN1→QKV＋local B | 3 / 17 | 3 / 16512 | 115844 | 165568 |
| LN1→QKV→Heads＋local B | 4 / 23 | 6 / 41088 | 140420 | 165664 |

Scalar/HVXの上表ownership/layoutは同一。ObservedのHVX receiptだけ追加64B。
logical bytesはDDR traffic/VTCM/physical allocator使用量ではない。peak RSSは
119512/119496/118656KiBで、全36 graphs、fixtures、Host oracles等を含む。
単一QKV kernelのphysical memory増分やscratch実測値ではない。

## F. Architecture and next investment

新packageのFFN2回帰でHVX math median210.5µs、今回QKV3本は316.5–320.0µsだった。
同じbinaryの限定観測で、QKVのMAC数はFFN2の1.5倍。QKV合計のeffective throughputは
約2.46–2.48GFLOP/s、FFN2回帰は約2.49GFLOP/s。hardware peakやcycle-counter値ではない。
各projectionを別timerで測っていないため、合計の3分の1を個別callの実測と呼ばない。
K64の短いpipelineが大きなthroughput悪化を起こした証拠は、この範囲では見られなかった。
FFN2旧cohortの性能値は今回のspeedup計算に使っていない。

contiguous weight loads、output-column SIMD、q-loop/index制御の共有が設計上の利点。
normalize/conversion、broadcast、依存鎖、prologue/epilogue、3 callとY再読、load/storeは残る。
32 lanesから倍率を予測していない。legacy Scalarのvolatile product/loop/address命令mixの差も
35倍の比率に含む。cache/DDR、DSP scheduling、dispatch、register occupancyの寄与は未測定。
残余wallにはLN、HeadSplit、finite checks、copies、backend、Host logging等が含まれ、
hardware counterなしで一つを支配因子と断定できない。

| Geometry / source contract | HVX status and design consequence |
| --- | --- |
| FFN2 M32/K128/N64 NN | 旧HVX実証＋今回新packageの数学/ABI回帰PASS |
| QKV M32/K64/N64 NN、3 projections | 今回3-process精度/性能PASS、共有helper成立 |
| dQueryRaw 2 heads M32/K32/N32 NN | Scalar verified、HVX未実装。将来の速度研究候補 |
| dKeyRaw 2 heads M32/K32/N32 TN | Scalar verified、左transpose addressingを別監査 |
| dContext M32/K64/N64 NT | Scalar verified、右transpose/prepackの費用を別検証 |
| dWo M64/K32/N64 TN | Scalar verified、Mと左strideが異なる |
| dProbabilities 2 heads M32/K32/N32 NT | Scalar verified、dynamic Vのpacking lifetimeが問題 |
| Forward QKᵀ 2 heads M32/K32/N32 NT | 標準HTP accuracy BLOCKED、Custom未実装 |

今回QKV以外の新geometryをHVX化していない。共有できたのはstrict SF multiply/add、
ordered q、M32/N64 tilingと安全memory access。N32/tail、TN/NT、per-head packing、異なるMは
未検証。特にdynamic K/Vの右transposeはstatic weight prepackと異なり、毎MBの費用/lifetimeを
測る必要がある。各geometryのsource contractを優先し、今回のword/性能結果を一般保証にしない。

registry/receiptの増加には独立marker、shape、source/binary provenance、regressionの維持が必要。
次の拡張ではnamed geometry/receipt metadataの小さなSSOTを検討する価値がある。
今回は大規模registry refactorや第二package frameworkを作っていない。
static diagnostic countersには同種graphをconcurrent実行しない制約があり、production schedulingや
並列MBのresource配置は未測定である。

**主投資はD: Precision Frontier Closure。** 正しい既存NN算術を高速化できる証拠は2 geometryへ
広がったが、Forward QKᵀ、Backward score scale等の未成立領域がfull1-layer correctnessを止める。
次の独立研究でこれらの契約/accuracyを先に確定する。今回それらの実装は開始していない。
速度だけの候補ではAのdQueryRaw K32/N32 NNが次に単純だが、現時点の主優先にはしない。
BのNT/TNは動的transpose費用、CのFFN2→Add→LN1 fusionは8192B中間/launch削減と
X/saved-state lifetimeを追加検証する設計候補。fusion自体は実装していない。

QKVは393216 MAC/layer/MB、19層7471104、19層×8MBで59768832 MAC。
FFN2との合計は655360 MAC/layer/MB、19層×8MBで99614720 MAC。
QKV weightsは49152B/layer、19層933888B。19層のQKV outputs466944B、Y155648B、
LN saved state158080B。これは算術/論理payload集計で、同時保持量やtraining throughput予測ではない。

## G. Final decisions

| Question | Answer |
| --- | --- |
| 1. K64/N64実HVX命令生成・実行? | YES。linked V81 vector arithmetic＋DSP marker/outputs |
| 2. q順維持? | YES。64 productsのrounded recurrenceをdisassemblyと数値で確認 |
| 3. Q/K/V全words Scalar一致? | YES。各2048 words、signed zeroを区別 |
| 4. 各.002 double gate? | PASS。Q/K/V個別、閾値不変 |
| 5. DSP LN1→HVX QKV? | PASS。実LN producerのnative Y、固定外部X/parameters |
| 6. HeadSplit lossless? | PASS。既存copy kernel、全slice words一致 |
| 7. Observer-free accuracy? | PASS。そのvariantのactual final outputsで判定 |
| 8. DSP kernel speedup? | 35.11–35.51×、11.234–11.238ms→.3165–.3200ms |
| 9. Integrated free wall? | 3.31–3.40×、17.195–17.263ms→5.074–5.222ms、local B込み |
| 10. 3 process再現? | YES。timingを除く数値digest一致 |
| 11. FFN2 accuracy/ABI維持? | PASS。65 arithmetic instruction words一致＋新package596 executes回帰 |
| 12. 新scratch/alignment/packing費用? | 明示0、vmemu、Observed receipt追加64B、Observed wrapper frame operand追加32B。NoReceipt frame96Bは同じ |
| 13. NN共通設計は有効? | K64/K128・M32/N64でYES。M/N/transpose拡張は別gate |
| 14. 次の高速化geometry? | 将来候補はdQueryRaw K32/N32 NN。主投資は精度障害を先に解消 |
| 15. 次のprecision frontier? | Forward QKᵀ、Backward score scale。今回未実装 |
| 16. Full1-layer不足? | Forward scores/scale/mask/P/PV/gate/Wo/FFN1/ReLU/residual、Backward scale/scatter/projection/gate/FFN/full LN1 dY、producer-fed native連結、production registry/ownership/scheduling |
| 17. 40k→60k／200k? | NO-GO。training updates0、6031はUNRESOLVED / DORMANT / WATCH |

以下を別々に認定する。

- HVX_QKV_CODEGEN_VERIFIED
- HVX_QKV_ORDERED_WORD_IDENTITY_VERIFIED
- HVX_QKV_NUMERICAL_VERIFIED
- HVX_QKV_NATIVE_SEGMENT_VERIFIED
- HVX_QKV_OBSERVER_FREE_VERIFIED
- HVX_QKV_KERNEL_SPEEDUP_VERIFIED
- HVX_QKV_SEGMENT_SPEEDUP_VERIFIED

BLOCKED_HVX_QKV_CODEGEN/NUMERICS/INTEGRATION、PERFORMANCE_NOT_PROMOTEDは最終候補に該当しない。
35-node機構、full1-layer／19-layer correctness、production速度は未解決のままである。

## H. Provenance, regression and reproduction

Device-tested source **`a50da5c3a569d34fb6896c8fcffb3754dc7f2312`**。
最終3 processで同じpackage/probe/SDK/fixturesを照合し、freeze後のcandidate再buildは0。
final HEADは`git log -1 --format=%H -- docs/ordered-fp32-hvx-qkv-feasibility.md`で取得し、
exact hashとclean treeをprivate closeoutおよび完了回答に記録する。

| Identity | SHA256 |
| --- | --- |
| DSP package | `9736778fc192fd71c7ab974341294ec4f4cd05618473c699848318e9839c9b11` |
| Host package | `08ee5a005b012136a9c0050f1d39e30251ec32421873af5a7a8efc57df3e6423` |
| Probe | `9e28d2b960f6dbc9c2dc4f4c17b58de4f4024c9c603967682c504836c1af143d` |
| Hexagon compiler | `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047` |
| NDK compiler | `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3` |
| K64 target disassembly | `d4eaa44b2e0bab5121d55b74c88708f28ab150f5767663e86c76e6f034b2b8c2` |
| Frozen inputs | `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f` |
| LN1 parameters | `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f` |

QAIRT2.48.40.260702151143、QNN API2.37.0、HTP V81、Hexagon SDK6.6.0.0／tools19.0.07、
Android NDK26.2.11394342を固定した。SDK manifests/headers/manual hashesは公開aggregateに保存する。
raw reports、pilot/final cohort manifests、frozen binaries、build identities、各process ledger、
paired distributionsのsourceはignored `build/reports/ordered-fp32-hvx-qkv/`に保存した。

初期Fast99.8s／closeout Fast100.7s PASS、Host contract/diagnostic suiteはQnn profile内で575.5s PASS。
追加targeted Host suite15/15、既存shape3920、FFN2 graph20、QKV graph24 ownership PASS。
Qnn profile695.4s PASS、Device smoke118.3s PASS、Muon W8の新規実行114/114 PASS。
W8はmaxAbs `1.490116119e-8`、worst relative L2 `4.40269854e-8`、minimum cosine1、
RPC success、output finite、fallback=false、focus takeover0、thermal0→0／battery37°C→37°C。
保護training stateを更新する試験ではない。

Qnn/APK auditでpinned SDK core/API、arm64 ABI、V81 Stub/SkelとSDK hashes、host path不在、
2.47混入不在を検査した。optional inventory不足だけadvisoryで、SDK fallbackは0。
各device run前に同一physical identity、awake/top-resumed、active run不在、Host≥20GiB／
device≥12GiB、thermal/power、APK/QNN/Skel、package/probe/compiler/fixtures、protected hashesを確認した。
無断wake、foreground takeover、clock/充電設定変更、force-stop、root/SELinux/firmware/SDK変更は0。

保護40k checkpoint SHA256
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`と、
tokenizer/cache/manifest/run-plan計5 hashesを照合した。100k baseline、200k UNSTABLE、
Step41647/42k evidence、以前のlocal branches/private reportsを保持した。
checkpoint decode、training updates、Final test、LR/optimizer/codec/data order変更は0。

公開はallow-list exporterだけを使用する。source/binary/fixture/SDK、全metric/word coverage、
実DSP markers、paired order、未観測timer、teardown、digest、FFN2/W8、protected/branch identitiesを
fail closedに検査した。14破損条件の拒否は通常Pythonと`python -O`の両方でPASS。
raw tensor/checkpoint/corpus、device identifiers、SDK binaries、private logs、absolute local pathsはcommitしない。

再現時はclean source、pinned roots、manifest/fixtures、device safetyを満たすこと。

```powershell
.\scripts\test_ordered_hvx_qkv_host.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143
.\scripts\build_fp32_custom_probe.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -HexagonSdkRoot $HexagonSdkRoot -NdkRoot $NdkRoot
python scripts/audit_ordered_hvx_qkv.py --hexagon-tools-bin $HexagonToolsBin --package build/fp32-custom/libHexaFp32Dsp.so --private-root build/reports/ordered-fp32-hvx-qkv
.\scripts\run_fp32_custom_probe.ps1 -Variant oneQkvHvx -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
.\scripts\run_fp32_custom_probe.ps1 -Variant oneHvx -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
.\scripts\run_qnn_headless_tests.ps1 -Suite hvx-muon-optimizer-benchmark -TestMode BACKGROUND_CORRECTNESS -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -HexagonSdkRoot $HexagonSdkRoot
.\scripts\export_public_ordered_hvx_qkv.ps1 -SelfTest -OutputPath docs/results/ordered-fp32-hvx-qkv-summary.json
```

pilotは別保存し、同じ最終bytesで3独立processを新規採取してcohort manifestを作る。
任意の過去recordsを混ぜてexportしない。今回の実装範囲はQKV K64/N64 NNだけである。
