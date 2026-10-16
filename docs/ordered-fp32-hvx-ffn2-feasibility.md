# Ordered FP32 HVX FFN2 feasibility

**FFN2のordered-FP32互換と実機高速化は成立した。** 同一DSP packageのscalar/HVXを
3独立processで比較し、全必須word差0、double accuracy PASS、native Add/LN1および
observer-free精度PASSを確認した。Standalone DSP mathは**37.90–39.56倍**、
observer-free FFN2→Add→LN1＋local Backwardのexecute wallは**2.69–2.75倍**。
各processが事前基準1.20倍／1.10倍を満たした。

**必須回帰の全工程完了はBLOCKED。** Device smokeはPASSしたが、直後にADB接続が
消失し、Muon W8 114/114は今回未実行である。以前のW8結果を流用しない。
このtransport停止と、成立したFFN2の数値／性能結果を分ける。
Production backendは変更せず、40k→60k／200kは引き続き**NO-GO**。

## A. Design and scope

Branch `investigation/ordered-fp32-hvx-ffn2`、local base
`5788d62d1632aa1b5650c1a2e23f2f71b170fc3e`。開始時clean、upstreamなし、known remote refsに
未到達のlocal90 commitsを保持した。mainからの再構築、fetch、reset、rebase、履歴変更、
push、PR、mergeは行っていない。

対象は固定NN `[32,128] × [128,64] → [32,64]`だけ。`custom_matmul_ops.h::ordered`と
`closure_ops.h`の既存scalar数学はbaseから不変。既存37 Opの名前・順序・ABI・receipt・
counterを維持し、diagnostic-only/default-offの`FFN2Hvx`／`FFN2HvxNoReceipt`を追加した。
QHPI interface1.4は不変。Host preparationでHVX演算を要求するとfatalとなり、scalarへ
fallbackしない。Production graph／optimizer／checkpoint codec／data orderは変更しない。

128B vectorは`__HVX_LENGTH__ == 128`、`sizeof(HVX_Vector) == 128`でbuild時に検査し、
FP32 32 lanesを確認した。SDKには64B/128B両型定義があるが、今回は128Bだけをbuildし、
V81 DSPで実行した。各rowのN64を2 tilesに分け、各laneが一つのoutput columnを担当する。

```text
acc = +0.0f
for q = 0 .. 127:
    product = round32(A[m,q] * B[q,n])
    acc = round32(acc + product)
Y[m,n] = acc
```

q方向のtree reduction／reorderはない。FP32 productとaddを分離し、FMA contractionと
fast-mathを無効化した。HVX専用翻訳単位だけでvector arithmeticを有効化し、scalar側の
従来flagsは維持した。

row-major layoutはA row512B、B row256B、Y row256Bのまま。Aのwordをbit-copyして
`Q6_V_vsplat_R`でbroadcastし、Bの連続32 wordsを読む。128B exact-size `memcpy`が
`vmemu` load/storeへ展開され、QNN pointerの128B alignmentをC++上で仮定しない。
全read/writeはlogical tile内、N64にtailなし。transpose／prepack／staging／heap allocationは0。
Observedで実測したA/B/Yのmod128は全件0だったが、NoReceiptの非可視pointerや将来の
allocationにも同じalignmentが保証されるとは主張しない。

新Observed receiptは16 header＋12288 input＋2048 output words、57408B。
scalar receiptより32B大きく、timing比較にもこの差が含まれる。markerはHVX
`0x48383114`、scalar `0x48383112`、counterは別。新headerでpointer alignment、vector幅、
geometry、明示scratch0、ABI1を記録する。NoReceiptはINT receiptを持たない。
同種graphのconcurrent executionはこのstatic-counter診断の対象外。

## B. SDK, compiler and disassembly

QAIRT `2.48.40.260702151143`／QNN API `2.37.0`／HTP V81／Hexagon SDK `6.6.0.0`／
tools `19.0.07`／NDK `26.2.11394342`を固定した。SDK header、型定義、compiler guide、
V81 HVX Programmer Reference Manual（80-N2040-63 AA、sections2/6.1–6.3）を監査した。

初期objectの既定qfloat loweringはIEEE互換の保証に使えないため、最終candidateは
compiler guide記載のstrict modeを明示した。

```text
-mv81 -mhvx -mhvx-length=128B -mhvx-qfloat
-mllvm -enable-xqf-gen=true -mllvm -hexagon-qfloat-mode=strict-ieee
-O2 -G0 -fPIC -ffp-contract=off -fno-fast-math
-fno-vectorize -fno-slp-vectorize -fno-unroll-loops
```

`Q6_Vsf_vmpy_VsfVsf`／`Q6_Vsf_vadd_VsfVsf`は、SF入力をqf32へnormalizeし、
qf32 multiply→SF conversion、SF add→SF conversionへlowerされる。積と各accumulationが
それぞれFP32へ丸められ、extended accumulatorを次qまで保持する方式ではない。
HVXはround-to-evenを扱う。subnormal／signed zeroの実挙動はflagsだけで認定せず、
追加finite controlsを含む実機word比較で検査した。

リンク済み`hexatrain_ordered_hvx_ffn2`にはstatic sitesとしてvector multiply4、add4、
SF conversion8、normalization8、broadcast4、unaligned memory5がある。これは動的命令数ではない。
FMA／scalar multiply-add／call・library置換／stack frame・spillは対象symbol内にない。
scalar address演算・broadcast用word load・loop制御は残る。29 vector register名v0–v28を
参照するが、peak live register数やruntime occupancyを測った値ではない。

q loopはcompilerによりsoftware pipeline化された。prologue、125回のhardware loop、
epilogueで128 productsを処理し、rounded productを増加q順でaccumulateする。
Aは4B、Bは256Bずつ進み、2 tileと32 rowsを走査する。disassemblyの依存鎖と、
実機のHost ordered／scalar DSP／HVX全words一致を別々に確認した。

`scripts/audit_ordered_hvx_ffn2.py`のlinked ELF V81／vector op／conversion／memory／
fusion・call・spill不在検査と、HVX marker付きDSP実行の両方を保存した。
QHPI_RESOURCE_HVXの宣言だけを実HVX arithmeticの証拠にしていない。

## C. Standalone accuracy

checkpoint-derived frozen8＋既存stress8＋追加12 controls、計28 cases。
追加controlsはordinary、ReLU-like、sparse、zero、signed zero、mixed weight signs、
large/small powers、cancellation、near-subnormal finite、fixed random、asymmetric、
safely scaled finite extrema。seed `0x081ff123`、LCG1664525/1013904223。
NaN／infinity／unsafe overflowはHost negative testsへ分け、finite accuracy統計に混ぜない。
checkpoint decode／training updateは行っていない。

独立`one::mm` doubleとordered-FP32 referenceはCustom dotを呼ばない。4 variants
（scalar/HVX × Observed/NoReceipt）すべて、同一input wordsを使った。

| Required comparison | Final3 processes |
| --- | --- |
| scalar DSP vs independent Host ordered FP32 | all2048 output words/case一致 |
| HVX vs scalar DSP／Host ordered | signed zeroを区別して全words一致 |
| each variant's double maxAbs | `2.96602956951e-6`、gate .002 PASS |
| Observed operand/output receipts | 全words一致、DSP markers確認 |
| NoReceipt final output | そのvariant自身のactual outputでaccuracy／scalar parity PASS |

追加Host controlsは24 calls、maxAbs `1.32154673338e-6`でPASS。
Host scalar frozen8のmaxAbs `7.18314831616e-7`と、上表のDSP値は別の結果である。
HVXのHost emulationを成功扱いしないnegative controlもPASSした。

## D. Native Add / LN1 and observer-free precision

standalone通過後、stage1はFFN2→DSP Add、stage2はFFN2→DSP Add→DSP LN1 Forward＋
saved-state local Backwardを接続した。既存Add/LN arithmeticをそのまま再利用した。
Host正解FFN2やXを途中注入していない。A/W2/Residual1/gamma/betaとlocal dYは固定外部入力。
**Backward dYはreference-fed**であり、全Attention由来dYの正当性は認定しない。

各stageはscalar/HVX × Observed/NoReceipt × Float Tap有無、8 variants、各16 cases。
No-Tap中間FFN2/X/xhat/inverseはNATIVE、parametersは個別APP_WRITE、最終出力はAPP_READ。
xhat/inverseはBackward完了まで保持する。Observedでproducer/consumerの全receipt words、
残差、gamma/beta、dY、saved state、dX receiptを照合した。TapではAPP_READとも照合する。

| Output | Double maxAbs across final cohort | Gate |
| --- | ---: | ---: |
| FFN2 | 2.96602956951e-6 | .002 |
| LN1 X | 3.20444814861e-6 | .002 |
| LN1 Y | 4.61443698363e-7 | .002 |
| xhat | 1.41215464877e-6 | .002 |
| inverse | 2.66506694713e-7 | .002 |
| dX | 3.42286708843e-6 | .02 |
| dGamma | 2.7886468132e-5 | .02 |
| dBeta | 7.89668411016e-6 | .02 |

scalar/HVXの全必須final words、Observed/NoReceipt間、Tap有無間はすべて一致した。
同じobserver variantのscalar/HVXを直接比較し、scalar Observed anchorとの比較も別記録した。
NoReceipt/no-Tapはそのvariantのactual final XまたはY/dX/dGamma/dBetaをdoubleと比較する。
内部operandを直接観測したとは主張しない。非可視中間のHost計算はoracleであり測定値ではない。
既存35-node Float Tap incidentは独立にUNRESOLVEDのまま。

QNN return0とfiniteは別に確認した。input mutation、poison残り、canary failureは0。
Host20-graph shape/ownership testもPASSし、NoReceiptにINT tensorsがないことを確認した。

## E. Performance

同一最終DSP package／probe、同じfixture0、BACKGROUND_CORRECTNESS条件。
各pairにfirst executeと各variant2 warmups、その後ABBA/BAAB交互8 blocks。
**各variant16 steady samples/process**。数値gate前の性能昇格は行わない。
事前のkernel1.20倍／observer-free segment1.10倍基準は変更していない。

| Process | scalar DSP math ms | HVX DSP math ms | kernel speedup | scalar free segment ms | HVX free segment ms | segment speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.072 | .213 | 37.897× | 13.279 | 4.926 | 2.696× |
| 2 | 8.427 | .213 | 39.563× | 13.261 | 4.824 | 2.749× |
| 3 | 8.072 | .209 | 38.622× | 13.113 | 4.868 | 2.694× |

kernelはstandalone Observedのmath window、segmentはNoReceipt/no-Tap stage2全体。
stage2 wallにはlocal Backwardも含む。Fwdだけのlatencyと呼ばない。
絶対差はkernel7.859–8.214ms、segment8.245–8.437ms。

以下はprocessごとのsteady medianの範囲であり、全sampleを混ぜたmedianではない。

| Stage / instrumentation | scalar wall ms | HVX wall ms | scalar FFN2 math ms | HVX FFN2 math ms |
| --- | ---: | ---: | ---: | ---: |
| Standalone / Observed | 11.473–11.946 | 3.307–3.811 | 8.072–8.427 | .209–.213 |
| Standalone / NoReceipt | 11.493–11.787 | 3.473–3.658 | unobserved | unobserved |
| FFN2→Add / Observed | 12.606–12.963 | 4.329–4.678 | 8.461–8.466 | .210–.212 |
| FFN2→Add / NoReceipt | 12.099–12.407 | 3.822–4.065 | unobserved | unobserved |
| FFN2→Add→LN1 F/B / Observed | 13.967–14.134 | 5.627–5.759 | 8.470–8.472 | .212–.212 |
| FFN2→Add→LN1 F/B / NoReceipt | 13.113–13.279 | 4.824–4.926 | unobserved | unobserved |

HVX receipt-inclusive FFN2 windowはstandalone .517–.549ms、Add .508–.522ms、LN .612–.643ms。
math windowはload/broadcast/ordered arithmetic/storeとfunction callを含み、finite検査・
receiptコピーを除く。receipt windowはその周辺検査・コピーを含む。NoReceipt timerは0ではなく未観測。
whole wallは同期QNN call＋既存Host logging wrapperであり、Add/LNのmath timeを分離していない。
API単体のwallも別保存した。wall−mathを純粋なRPC/QNN overheadとは解釈しない。

| Free stage2 process | scalar p10/p90 ms | HVX p10/p90 ms | scalar min/max ms | HVX min/max ms |
| --- | --- | --- | --- | --- |
| 1 | 12.950 / 13.496 | 4.777 / 4.989 | 12.819 / 14.130 | 4.682 / 4.997 |
| 2 | 12.813 / 13.517 | 4.612 / 5.167 | 12.084 / 13.773 | 4.476 / 5.324 |
| 3 | 12.986 / 13.248 | 4.696 / 5.127 | 12.975 / 13.371 | 4.652 / 5.503 |

stage2 freeのblock paired speedup最小は2.553×。ABBA中央値は2.675/2.711/2.699×、
BAABは2.748/2.694/2.696×で、run orderによる逆転はなかった。battery37°C→37°C、
thermal0→0、AC powered=true／USB=false／wireless=falseが3 processで一致した。
clock／scheduler／他appは固定しておらず、exclusive benchmark／foreground takeoverはない。

free stage2のgraph初回wallはscalar13.481–13.898ms、HVX5.209–5.545ms。
これは先行graphでcontextがwarmになった後の当該graph初回である。
finalize wrapperはscalar44.647–66.261ms、HVX43.963–52.240ms。backend変動も含むため、
finalize speedupを認定しない。Host allocation/copy medianはscalar62.4–64.0µs、HVX62.1–65.1µs。
input size/immutability検査を含むHost helperの範囲と、DSP finite scanの個別時間は分離未測定。
詳細なmedian/p10/p90/min/max、API wall、first/finalize、paired分布は公開aggregateに保存した。

| NoReceipt representative graph | nodes/tensors | logical bytes | native FP bytes | Host buffer lower bound bytes |
| --- | ---: | ---: | ---: | ---: |
| FFN2 | 1/3 | 57344 | 0 | 106528 |
| FFN2→Add | 2/5 | 73728 | 8192 | 122912 |
| FFN2→Add→LN1＋local B | 4/15 | 107652 | 24704 | 149120 |

scalar/HVXの上記ownership/layoutは同一。明示kernel scratch0、prepack0、alignment staging0。
logical bytesはbackend物理allocation／VTCM／DDR trafficではない。
process peak RSS KiBは110608/110532/111100。複数graph・fixtures・CPU referencesも含み、
単一HVX kernelのphysical memory増分ではない。

## F. Bottlenecks and architecture

実測でFFN2の主要scalar arithmetic費用を取り除けた。出力列SIMD化によるcontiguous B load、
scalar q-loop制御／indexingの共有、vector register上のaccumulationが設計上の利点である。
ただし個別の寄与はcounterで分離していない。HVXでもAのbroadcast、qf32 normalization、
積と和のSF conversion、依存鎖、software-pipeline nops、load/storeが残る。
headerではmultiplyはCVI_VX_DV、addはCVI_VX、いずれもslots2/3。DSP scheduling、
cache/DDR、dispatchの内訳は未測定である。

geometry FLOPをmath windowで割ったeffective kernel throughputは約2.46–2.51GFLOP/s。
これはhardware FP32命令のpeak throughputではない。32 lanesから32倍を予測した結果でもない。
scalarのvolatile product roundingや異なるinstruction/address/loop mixも速度比に含まれる。

| MatMul candidate | Existing source geometry | Generalization assessment; implementationなし |
| --- | --- | --- |
| Q/K/V | M32/K64/N64 NN | 同じ2 output tiles。次のbounded研究の第一候補 |
| dContext | M32/K64/N64 NT | right transpose/addressingとprepack費用・lifetimeを別検証 |
| dWo | M64/K32/N64 TN | strided A、contiguous B。q順とownershipを再検証 |
| dProbabilities | 2 heads M32/K32/N32 NT | one tile/head、right transpose依存 |
| dQueryRaw | 2 heads M32/K32/N32 NN | contiguous output tiling候補 |
| dKeyRaw | 2 heads M32/K32/N32 TN | transpose-specific A addressing候補 |

共通化候補はSF multiply/add契約、増加q、output tiling、vector幅と安全load/store。
transpose／packing／tail／ABI／native lifetimeは各source contractを正とし、今回他geometryを
HVX実装していない。異なるgeometryや同時実行で同じ精度／register圧／speedupを保証しない。

**FP32 Custom MatMulをHVXへ展開する価値は条件付きで肯定**する。FFN2では算術と実segmentの
両方で改善し、scalar Custom Op中心の基盤の主要制約に対処できる証拠を得た。
全trainingが現実的な速度になるかは、残るMatMul／他演算の精度とproduction scheduling次第で未確定。

| Option | Assessment |
| --- | --- |
| A: scalar separate | 同一最終binaryでの比較baseline |
| B: HVX FFN2＋既存Add/LN1 | 今回の数値／速度成立対象 |
| C: HVX FFN2＋fused Add/LN1 | 設計のみ。8192B FFN2中間とlaunch削減候補 |

fusionはnative Xのresidual consumer lifetime、LN xhat/inverse saved state、row64 reductionを
維持する必要がある。vector storeとrow consumption、vector/scalar移行、scratch、QNN graph
複雑性の費用は未測定。今回の算術高速化をfusion効果と混同しない。
**次はQKVなど別NN MatMulの独立HVX feasibilityを優先**し、FFN2 fusionは境界費用の追加証拠を
得て再評価する。FFN2 math .21msとsegment4.8–4.9msの差からdispatchの純費用は算出しない。

FFN2は262144 MAC/layer/MB、19層で4980736、19層×8MBで39845888 MAC。
weights32768B/layer、unique A+B+Y57344B、fusion削減候補8192B/layer。
これらはgeometry上の量であり、19層の実traffic／schedule／training速度予測ではない。

## G. Decisions

| Question | Answer |
| --- | --- |
| 1. 実HVX FP32 multiply/add生成・実行? | YES。linked V81命令列＋DSP marker/outputs |
| 2. q順をdisassemblyと数値で確認? | YES。rounded recurrence＋全word一致 |
| 3. Scalar全output words一致? | YES。Host ordered、scalar DSP、HVX、observer variants |
| 4. Double gate維持? | YES。閾値変更なし、上表参照 |
| 5. Native FFN2→Add→LN1? | PASS。実producerから接続、local BもPASS |
| 6. Observer-free accuracy? | PASS。そのvariant自身のactual final outputsで判定 |
| 7. DSP math speedup? | 37.90–39.56×、絶対差7.86–8.21ms |
| 8. Free graph wall改善? | 2.69–2.75×、約13.1–13.3ms→4.82–4.93ms |
| 9. Memory/alignment費用? | 明示scratch/prepack/staging0、vmemu、Observed receipt追加32B。backend物理費用未測定 |
| 10. 速度不足時の支配要因? | 今回は基準超過。残余wallの要因分解は未確定 |
| 11. 他MatMulへ一般化価値? | 条件付きYES。各geometryの精度・layout・速度は別gate |
| 12. 次の優先? | 他NN MatMulのbounded HVX研究、次に根拠を追加してfusionを再評価 |
| 13. Full1-layer未解決? | Forward QKᵀ/scale/mask/P/PV/gate/Wo/FFN1/residual精度、Backward score scale/scatter/projection gradients/gate/FFN backward/DReLU/full LN1 dY、branch/registry integration、19層producer-fedとschedule |
| 14. 40k→60k／200k? | NO-GO。training updates0、6031はUNRESOLVED / DORMANT / WATCH |

HVX_FFN2_CODEGEN_VERIFIED、HVX_FFN2_ORDERED_WORD_IDENTITY_VERIFIED、
HVX_FFN2_NUMERICAL_VERIFIED、HVX_FFN2_NATIVE_SEGMENT_VERIFIED、
HVX_FFN2_OBSERVER_FREE_VERIFIED、HVX_KERNEL_SPEEDUP_VERIFIED、
HVX_SEGMENT_SPEEDUP_VERIFIEDを、それぞれの上記scopeで認定する。
BLOCKED_HVX_CODEGEN／NUMERICS／INTEGRATION、PERFORMANCE_NOT_PROMOTEDは該当しない。
W8回帰のtransport停止、35-node機構、full1-layer／19-layer correctnessとproduction速度は未解決。

## H. Provenance, validation and reproduction

Device-tested source **`515a32a7346251d908c9f1037adf30015bfcd6b9`**。
同一DSP packageのscalar/HVXと同一probeでpilotを分離し、最終3独立processを固定した。
各process596 executes、1266 metrics、2158全word比較、16 steady samples/variant。
合計1788 executes、3798 metrics、6474比較。API failures0、finite PASS、mutation0、
complete-write/poison/canary PASS、teardown成功、focus takeover0。

3 process共通のtiming-free canonical numerical SHA256:
`8a1294d616d4e139652c1735bec171542ef4e284d3ff1477b79b01ef55aa89f9`。
対象はonemetric／hvxwords／gate／summary行で、timing/counter/alignment/IDs/pathsを除外する。
raw tensorそのもののSHA256ではなく、全word比較はprobe内で別途実施した。

| Binary / compiler | SHA256 |
| --- | --- |
| DSP package | `166b4974fabc64bc2cac03360173cce672c77d07149e84b5aa5e3add20fe854b` |
| Host package | `f27673572e38441b3a54ea1bd8c2838631485fe95bb764198e72249b691ece85` |
| one-layer probe | `e56d4fb0d5460b5f2174b88e0c6c9505d62ec6f1df717dd34da50a8574e6d501` |
| Hexagon compiler | `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047` |
| NDK compiler | `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3` |
| linked target disassembly | `7c88e9ea2d49667ac4c52d5eb6602f84e22863466d98fde6390e79427290a168` |

Frozen SHA256: inputs `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`、
LN1 parameters `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`、
producers `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`。
SDK header/manual/manifest hashesと全分布は[公開aggregate](results/ordered-fp32-hvx-ffn2-summary.json)。
raw reports、各process ledger、compiler/SDK identity、frozen binaries、pilot、final cohort manifestは
ignored `build/reports/ordered-fp32-hvx-ffn2/`に保存する。

検証は最終Fast71.3s PASS、Host contract/diagnostic suite PASS、追加Host13/13 PASS、既存shape3920＋
新20 graph ownership PASS、Qnn profile1125.8s PASS、Device smoke80.8s PASS。
Qnnはpinned SDK、core/API、QNN-enabled build、APK ABI/SDK hash/path/2.47混入を検査した。
optional inventory不足だけadvisory、fallbackなし。新probeはProbeOnlyでpackage/compiler/SDK/
source hashesを照合してbuildし、freeze後のpackage/probe再buildは0。

Muon W8 114/114はADB_NO_ONLINE_DEVICEで開始前停止。Device smoke後のtransport障害であり、
数値FAILやoptimizer回帰とは判定しない。オンライン端末もmDNS connect serviceもなかった。
無断wake／USB設定変更／endpoint推測／別端末reattachは行わない。接続回復後に同一physical
identityと全安全条件を再検査して、この必須回帰を完了する必要がある。

初期のDozing／Host容量停止はprivate履歴に保持し、容量回復後に全preflightをやり直した。
各final runでHost≥20GiB、device≥12GiB、awake、top-resumed、active run不在、thermal/power、
APK/QNN/V81 Skel/package/probe/fixtures/protected hashesを確認した。foreground takeover、
clock／充電設定／強制冷却／root／SELinux／firmware／SDK変更は0。

保護40k checkpoint `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`と
tokenizer/cache/manifest/run-plan計5 hashesを入口・各run・closeoutで照合する。
100k baseline、200k UNSTABLE、Step41647/42k evidence、以前のlocal branches/private reportsを保持した。
checkpoint decode、training updates、Final test、LR/optimizer変更は0。

公開はallow-list exporterだけを使い、raw tensor/checkpoint/corpus/logcat/identifiers/absolute pathsを
commitしない。exporterは3 processのsource/binary/fixture/SDK/coverage/accuracy/word identity、
paired order、未観測timer、teardown、digestをfail closedに検査する。8破損条件の拒否を確認した。
最終HEADは`git log -1 --format=%H -- docs/ordered-fp32-hvx-ffn2-feasibility.md`で取得し、
exact hashとclean treeをprivate closeoutおよび完了回答に記録する。push/PR/mergeは0。

再現時は明示的にpinned SDK rootsを渡し、source clean／manifest／safety gatesを満たすこと。

```powershell
.\scripts\test_ordered_hvx_ffn2_host.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143
.\scripts\build_fp32_custom_probe.ps1 -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -HexagonSdkRoot $HexagonSdkRoot -NdkRoot $NdkRoot
python scripts/audit_ordered_hvx_ffn2.py --hexagon-tools-bin $HexagonToolsBin --package build/fp32-custom/libHexaFp32Dsp.so --private-root build/reports/ordered-fp32-hvx-ffn2
.\scripts\run_fp32_custom_probe.ps1 -Variant oneHvx -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId 2.48.40.260702151143 -NdkRoot $NdkRoot
.\scripts\export_public_ordered_hvx_ffn2.ps1 -FinalCohort -SelfTest -OutputPath docs/results/ordered-fp32-hvx-ffn2-summary.json
```

pilotと最終3 processは別記録へ保存し、同じ最終bytesを各process前に照合する。cohort manifestを
作らずに過去recordsを混ぜてexportしない。GraphはFFN2だけをHVX化しており、他MatMul／fusion／
production backend／全19層／training updatesは今回の実装対象外である。
