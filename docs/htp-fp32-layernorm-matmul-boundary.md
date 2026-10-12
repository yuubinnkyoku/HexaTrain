# DSP FP32 LayerNorm and downstream MatMul boundary

2026-10-13。base `bb0b3a8aff22b5d5beb1a7083fa3ff6dac541d3f`。
branch `investigation/htp-fp32-layernorm-matmul-boundary`。
この文書は診断の事前登録と最終結果を分けて記録する。

最終分類:

- **DSP_LAYERNORM_FORWARD_VERIFIED / DSP_LAYERNORM_BACKWARD_VERIFIED / DSP_LAYERNORM_RESIDUAL_VERIFIED**。
  D64の2×64/32×64と、D8/16/32/64/128の100-case matrixを両row shapeで検証した。
- **BLOCKED_MATMUL_PRECISION / MATMUL_PRECISION_STAGE_UNRESOLVED**。
  標準HTP MatMulは5種類のshapeでaccuracy gateを満たさない。Copy/native境界は精度保持する。
- full G1 integrationは**UNRESOLVED**。両lane PASSを条件とするone-layer統合へは進まなかった。
  BLOCKED_GRAPH_INTEGRATIONとは分類しない。API/graph接続不能ではなく、MatMul accuracyの問題である。

200k再開は不可。production backend、全38 LN、optimizer/checkpoint/data orderを変更していない。

## 事前登録

固定QAIRT2.48.40.260702151143/QNN2.37.0/HTP V81、Hexagon6.6.0.0/tools19.0.07、NDK26.2.11394342。
既存HexaFp32 QHPI package、registration、build/source identity、runnerを拡張する。
production graph、Muon W8、IDL、optimizer、checkpoint、double referenceを変更しない。

Lane A:

- Xはrow-major `[T,D]`。gamma/beta `[D]`、epsilon1e-5、saved xhat `[T,D]`、inverse `[T,1]`。
  QHPI flat4へbackfillし、row strideはD。rowwise inverseとgamma/betaはkernel内でbroadcastする。
- anchor-relative centering、Kahan FP32、scaled RMS、`hypot(rms,sqrt(epsilon))`。
  不要なvarianceはmaterializeしない。FMA off、fast-mathなし、scalar FP32、vectorization off。
- 最初はD64の2/32 rows×20 recipesのForward。Y/xhat ordinary0.002、登録済み低分散0.06。
  inverseは既存boundary diagnosticのabsolute0.002。Forward PASS後のみBackwardを実装する。
- dX/dGamma/dBetaはdouble maxAbs0.02、full upstream1、同一DSP Forward由来のsaved state。
  Host FP32との差、RMS/relative、finite count、output/fixture bit digestを別々に記録する。
- D64 Backward PASS後、D8/16/32/64/128の各20 recipesへ拡張する。
  32-row casesは既存2-row X/dYを繰返し、非trivialなrow signを付ける。勾配の縮小はしない。
- Native residualはDSP Forward→DSP Backward→Custom Add。native/tap/fusedを分ける。
  Host saved stateを注入した試験はDSP LNの証拠にしない。
- 既存double finite-difference matrixの5e-4 gateを維持する。

Lane B:

- T32/D64/F128/V1024のdWo、dContext、dW2、dReLU、dEmbeddingを診断対象とする。
  parameterとactivation gradientの両方を含む。標準MatMul単独とCustom Copy→native→MatMulを比較する。
- 20 D64 recipesの正しいHost FP32 dX+residual(dY)を入力する。勾配scale1。
  matrix coefficientsは固定integer seed、weights scale0.125、activation scale1、oneHotは0/1。
  transpose/reductionを実G1と一致させる。入力/出力のFP32/double referenceを独立計算する。
- MatMul出力gateは既存generalized gradient診断のmaxAbs0.03に合わせる。
  `qnn_transformer_training.cpp`のinitialOneStepCorrect/gradient diagnosticを根拠とする。
  LN dXの0.02 gateを別tensorへ流用しない。出力0.02も追加記録するが昇格gateではない。
- Copy receiptで全FP32 gradient wordsを検査する。MatMul内のoperand conversionと
  accumulation/storeの区別は観測可能な範囲に限る。selector/cancellation controlsを使用する。
- 非有限/API error/buffer mutation/poison/canary/identity failureではfail closed。
  Lane Bのnumerical FAILはLane Aの独立diagnosticを止めない。

成功候補は3独立process。同じbinary/fixtures/runtime/SDK hashを維持する。
accuracy PASS後のみBACKGROUND_CORRECTNESS timingを行う。
38/304/608倍のdiagnostic wallをproduction throughput予測にしない。
Host20GiB/Device12GiB、active run protection、top-resumed監視を維持する。
60k/200k、全38 LN、production置換、final_test、SDK/firmware/root/SELinux変更、push/PR/mergeは実施しない。

## Reuse・ABI・数学

既存HexaFp32のCopy/Add/Multiply/Affine登録とreceipt ABIを維持し、LnForward/LnBackward/LnResidualを追加。
API initializationと可変shape/multiple-output helperを共有化した。第二のpackage frameworkは作っていない。
Muonのproduction modes5/7、QAIC IDL、production LayerNorm builderとdouble referenceは不変。

| Op | 入力 | 出力 |
| --- | --- | --- |
| LnForward | X `[T,D]`, gamma/beta `[D]`, epsilon scalar | Y/xhat `[T,D]`, inverse `[T,1]`, integer receipt |
| LnBackward | dY `[T,D]`, gamma `[D]`, 同一Forwardのxhat/inverse | dX `[T,D]`, dGamma/dBeta `[D]`, receipt |
| LnResidual | LnBackward入力＋residual `[T,D]` | dX+residual, dGamma/dBeta, receipt内に加算前dX |

QHPI signatureはFLOAT32/flat4/direct/DDR_ONLY。`[T,D]`を`[1,1,T,D]`へbackfillする。
gamma/betaをlast dimensionへbroadcastし、inverseはrowごとに読む。係数のHost展開は不要。
sampled dimensionsは8/16/32/64/128、Tは2/32。未試験dimensionへ一般化しない。
Scalarアクセスを使用し、HVX vector alignment/vector reductionは未検証。

数学は標準LayerNormとanalytic backwardである。
anchor-relative `X-X[0]`のKahan meanでcenteredを作り、amaxでscaled RMSを計算する。
`a==0`はrms0、denominatorは`hypot(rms,sqrt(epsilon))`、inverseはその逆数。
variance自体は出力・scratchへ保存しない。dXhatとdXhat*xhatのrow reduction、
dGamma/dBetaのrow間reductionにも独立Kahan sumsを使う。
Backwardは`inverse*(dY*gamma-mean(dY*gamma)-xhat*mean(dY*gamma*xhat))`。
gamma/beta方向、epsilon1e-5、full upstream1、モデルの定義を変更していない。

初回D64 Forwardの40 casesがPASSしてからBackwardを追加し、D64 Backward PASS後にmatrix/residualを実行した。
同時に複数の数値アルゴリズムを探索していない。
DSP disassemblyにscalar FP arithmetic、sqrtf/hypotf callsとdivisionのcompiler/library sequenceを確認。
user expressionはFMA contraction off、fast-mathなし、DSP vectorization off。
divisionやmath libraryの内部には`sfmpy:lib`等があり、library内部までFMAなしとは主張しない。
Hostとの差も実測で評価し、FP32宣言だけでPASSとしない。
FP32全有限入力の包括的なoverflow/underflow/subnormal証明ではない。
特にanchor差分やcompensated mean自体がoverflowする極端な入力は未試験で、今回の最大centered4096を超えて一般化しない。

## Lane A — Forward/Backward結果

最終実機sourceは`5e15d36d9c3c6db005e8cb2ac2be360e7569b4de`。
lnMatrixを3独立process、各100 recipes×2 row shapes、計600 casesで再確認。
Y/xhat/inverse/dX/dGamma/dBetaの全必須gateがPASSした。
すべて同一DSP Forwardのsaved stateを使い、Host stateを注入していない。

| 出力 | 全dimension/両row shape max double error | max Host FP32との差 |
| --- | ---: | ---: |
| Y | 5.81823561774e-7 | 2.38418579102e-7 |
| xhat | 3.9246283734e-7 | 2.38418579102e-7 |
| inverse | 1.94379752543e-5 | 3.0517578125e-5 |
| dX | 1.65460269841e-5 | 1.52587890625e-5 |
| dGamma | 1.23042775968e-6 | 7.15255737305e-7 |
| dBeta | 0 | 0 |

HostとDSPのsqrt/hypot/division semanticsの差はゼロと仮定しない。
各caseのmaxAbs/RMS、`max(abs(error)/max(0.001,abs(reference)))`、finite count、bit digestをprivateで保存した。
3 processのtyped numerical digestは一致する。平均errorでの昇格は行っていない。
既存double analytic-vs-finite-difference100-case suite（h=1e-6/2e-6、5e-4 gate）はQnn profileでPASS。
DSP backwardは独立double analytic referenceと比較した。Forwardのfinite性からBackward成功を推定していない。

| D64 fixture | DSP dX / residual max double error（各2/32 rows） |
| --- | ---: |
| low_variance | 1.09583253902e-5 |
| constant | 1.0685343284e-5 |
| large_mean_low_variance | 8.15331415538e-6 |
| random_seed97 | 1.53640447422e-5 |

ordinary、epsilon_dominated、random seeds1/7/31/97、centered256/512/4096、
scale boundary、sparse ±269.75/400を含め全20 recipesがPASS。
旧HTP low-variance dX0.27230037、dynamic scalingのaccuracy FAILは過去の保存済み実機証拠として比較する。
同じ旧失敗primitiveを再探索していない。今回の32-row/DSP計算を過去HTP試験へ読み替えない。
Lane Aでfirst failing operationは観測されなかった。

## Native residual・ownership

Forward→native xhat/inverse→Backward→native dX→Custom Addを構築した。
Forward receiptはproducerが保存したxhat/inverse wordsを返し、Backward receiptの受信wordsと照合する。
Backward receiptは加算前dX wordsを返し、Custom Addの受信wordsと全要素比較する。
浮動小数点tapなしでもnative精度を確認した。

native/tap/fusedの各variantを3独立process、D64の2/32 rows×20 recipesで実行。
全variantで最大dX+residual double error `1.53640447422e-5`、dGamma/dBetaも0.02 gate PASS。
3方式のcase別residual数値・出力word digestは完全一致した。
fused版はBackward＋residual Addを1 kernelにまとめ、Forwardは別kernelのまま。
入力Xから実DSP LNを通した結果であり、前回の外部bracket/inverse fixtureとは異なる。

Host APP_WRITE vectorsはsynchronous execute終了まで所有し、全input bit mutationを検査する。
APP_READ bufferはNaN poisonと前後canaryを持ち、全面書込み・境界外書込み・finiteを検査した。
Native tensorはQNN contextが所有する。context free後にsaved stateを持ち越さない。
source_destructive=false、self-slicingなし、kernel内scratchはstackのcentered/Sum arrays。
全customはinteger receiptをAPP_READへ返す診断版。receiptを完全に除いたproduction loweringは未検証。

## Lane B — 実G1依存とMatMul

最新generalized builderとshared LayerNorm backwardを確認した。

- LN2: dX=`DRESIDUAL1_LN`→residual Add→`DRESIDUAL1`。
  `dWo=AttentionContext^T*DRESIDUAL1`、`dContext=DRESIDUAL1*Wo^T`へ分岐。
- LN1: dX=`DINPUT_NORM`→residual Add→`DINPUT`。
  下層upstreamとなり`dW2=ReLU^T*upstream`、`dReLU=upstream*W2^T`へ分岐。
- 最下層: `dEmbedding=OneHot^T*DINPUT`。
- parameter gradientsはAPP_READでmaterializeされ、Hostで8 microbatchesを0.125f倍してaccumulateし、
  既存HVX Muon/Host Aux Adamへ渡す。今回このproduction経路は変更していない。

| Path | 実行するmatrix shape | K / transpose | output |
| --- | --- | --- | --- |
| dWo | `[32,64]^T × [32,64]` | K32, transpose input0 | `[64,64]` |
| dContext | `[32,64] × [64,64]^T` | K64, transpose input1 | `[32,64]` |
| dW2 | `[32,128]^T × [32,64]` | K32, transpose input0 | `[128,64]` |
| dReLU | `[32,64] × [128,64]^T` | K64, transpose input1 | `[32,128]` |
| dEmbedding | `[32,1024]^T × [32,64]` | K32, transpose input0 | `[1024,64]` |

全tensor FLOAT32。各pathでdirectとCustom Copy→nativeを別graphとした。
動的非trivial operandsによるMatMulのoutput、configured node、API executeを確認する。
CPU double dot、逐次FP32 dot（product round32、FMAなし）、HTP outputを比較する。
加えて理想double LN dX+residualからのMatMulを別referenceとして計算した。

各pathは20 recipes×random/selector/cancellation controls=60 checks。
各3 process、direct/native計1,800 checksで166/600 FAIL/processが一致した。
Copyは全2,048 words bit-exact。direct/nativeの最終output hashは全300組で一致した。

| Path | maxAbs vs double（供給したFP32入力） | 理想LN依存からのmaxAbs | FAIL / 60（各direct/native） |
| --- | ---: | ---: | ---: |
| dWo | 0.983984801079 | 0.983972020344 | 19 |
| dContext | 0.0623321533203 | 0.0623274891022 | 12 |
| dW2 | 0.883949152723 | 0.883950493337 | 19 |
| dReLU | 0.0623321533203 | 0.0623274891022 | 12 |
| dEmbedding | 0.124664306641 | 0.124654978204 | 21 |

parameterとactivation gradientの両方に反例がある。
控えめなrelative errorやfinite/API成功をmaxAbs gateの代用にしない。
0.02 supplemental gateも記録したが、MatMul acceptanceは事前の0.03である。
全19層/実checkpoint/実corpusのgradient distributionが同じと仮定しない。
dW2のsigned stress行列は実ReLUの非負分布そのものではないため、その一般化は制限する。
主代表のdWo/dContext/dEmbeddingに加えdReLUもFAILし、3種類以上の実shape要件を満たす。
OneHotは0/1、反復token位置を持つ。checkpoint読込みは不要と判断し、hash検査だけ実施した。

### Precision-loss位置

Custom output storeとnative Copy境界は精度保持。最初の観測損失はstandard MatMulの後。
内部operand conversion/accumulation/result storeを直接tapできていない。

FP16 latticeへ丸めたoperandでdouble dotし、FP16 outputへ丸めるcounterfactual modelは
580/600 checksでexact、残りを含む最大差0.03125。
input roundingだけのmodel最大差0.496337890625、output roundingだけは1。
例: dWo low_variance/randomの実error0.609795943868に対し、input-only modelとの残差0.41583520174、
output-only0.5、combined0。selector caseではinput/outputのどちらでも説明できる。
したがってoperand/output双方の低精度化は有力な説明だが、accumulationの型や順序を確定しない。
unexplained casesにはreduction order、FMA、rounding tie等の検討が残る。
Muon Newton–Schulzの反復誤差とこの単一MatMulの誤差を混同しない。
厳密な内部位置の分類はMATMUL_PRECISION_STAGE_UNRESOLVEDとする。

## Performance・resource予備測定

accuracy PASSのLane Aだけを追加測定した。1 warm＋8 steady samples/shape/process。
QNN INFO logging、integer receipts、Host status出力付きのBACKGROUND_CORRECTNESS。
全15 final processのthermal0、38–40°C、foreground takeover0。
power/charging snapshotsをprivateで保存。clock/power policyを変更していない。

| D64 graph | 2 rowsのsteady median µs（3 process） | 32 rows |
| --- | --- | --- |
| Forward+Backward | 869.0885 / 1033.750 / 912.0315 | 1053.9845 / 1132.604 / 1031.3285 |
| Native residual | 949.2965 / 1089.661 / 1144.5575 | 1724.036 / 1651.0415 / 1445.156 |
| Float tap residual | 882.526 / 1009.1665 / 934.844 | 1364.6095 / 1490.052 / 1481.016 |
| Fused Backward+residual | 901.120 / 868.984 / 894.349 | 1297.8905 / 1267.6045 / 1285.729 |

1 native-residual processのDSP timer範囲は、2 rowsでForward8–29µs/Backward8–12µs/Add13–16µs、
32 rowsでForward111–194µs/Backward103–110µs/Add178–227µs。
Forward/Backward timerはinput receipt copyを除きoutput receipt writeを含む。
既存Add timerはinput receipt copyも含むため、純粋なAdd arithmetic costと比較しない。
tapがnativeより速いケースがあり、単一graphの速度順位をproductionへ一般化しない。

registration、device/context/graph creation、finalize、first executeは個別API wallを保存した。
Host buffer allocation/input snapshot copyは別区間。QNN内部transfer/queue/synchronizationの分離、
package dlopen単独時間、純粋なkernel arithmetic時間は未測定。
Host/QNN overheadをexecute wall minus kernelと単純に同一視しない。

LN graph13 tensors/2 nodes、native residual16 tensors/3 nodes、fused14 tensors/2 nodes。
native xhat+inverseはT32/D64で8,320 bytes、dX8,192 bytes。
Forward receipt17,060 bytes、Backward receipt24,992 bytes、Add receipt24,608 bytes。
診断metadata量はproductionの必須payloadではない。
custom HAP allocation/VTCM直接使用0。centered512 bytes、Backward compensated arrays2,048 bytesのstack領域。
runtime自体にはVTCM8MiB予約とDDR yield bufferがある。process全体VTCM0とは主張しない。
Lane A peak Host RSSは約76,196–81,644KiB、backend/prepare/複数graphsを含む。
DSP全体peakは未測定。MatMul候補はaccuracy FAILのためperformance promotionしなかった。

## 必要な最小FP32拡張と費用

標準MatMulへの直接再投入案は不採用。
次の限定的研究は、LN2 residualの2分岐dContext/dWoをFP32 consumer prototypeで評価すること。
native FLOAT32は利用可能で、producer/consumer間にHost stagingを入れる必要は今回観測しなかった。
FP32 MatMul、residual+MatMul融合、gradient segment延長が候補。
別layoutだけで標準MatMulのprecisionを回復できる証拠はない。

| 候補 | D64/T32算術量 | 保存/出力の代表payload | 未解決リスク |
| --- | ---: | --- | --- |
| dContext FP32 MatMul | 131,072 MAC | Wo16KiB、gradient8KiB、result8KiB | FP32 reduction/accumulator、SDK scheduling |
| dWo FP32 MatMul | 131,072 MAC | context8KiB、gradient8KiB、result16KiB | transposed reduction、gradient materialization |
| dW2/dReLU | 各262,144 MAC | ReLU/W2、result各32/16KiB | FFN分岐と実ReLU distribution |
| embedding scatter/reduce | sparseなら2,048 contributions | result256KiB | repeated-token reductionとownership |

scalar prototypeは小さく反証可能だが、production FP32 MatMulにはblocking、layout、parallel reduction、
buffer reuse、数値許容性の検証が必要。今回DSP MatMulは実装していない。
既存LN38 normsのsaved xhat/inverseは約309KiB/microbatchという論理payload見積もり。
microbatch8のLN2 2分岐とLN1下層2分岐を全部FP32化する仮定では約115M MAC/updateが追加範囲となる。
これは必要範囲の仮定に基づく算術量であり、実時間や最終移植範囲ではない。
attention backward、より上流forwardの情報損失、後続gradient操作はまだ検証が必要。

38箇所を登録するだけでは解決しない。integrationにはpackageのapp loading、rank/broadcast、
forward saved state lifetime、backward分岐、gradient registry、observer-free lowering、regression/evidenceが必要。
scalar correctnessを維持してからHVX vectorizationを検討する。
304/608倍のmicrograph wallをend-to-end training予測にしない。
今回のpackage内接続には外部RPCをnormごとに増やさない設計上の可能性があるが、実統合性能は未測定。

## Decisionへの回答

1. DSP Forwardは対象matrixで正しい。2/32 rowsと5 dimensionsを実測した。
2. Backwardも同じDSP Forwardのsaved stateで0.02 gateを満たす。
3. native residualまで精度保持する。tap/fused最終値も一致する。
4. standard HTP MatMulへの戻しは対象fixturesで0.03 gate FAIL。
5. 最低でもresidualに続く必要MatMulをFP32領域へ含める候補が必要。最後に必要な境界は未確定。
6. standard MatMulを使うone-layer統合へのgoは出さない。限定FP32 dContext/dWo prototypeへの投資価値はある。
7. 全19層はpackage/math/graph ownership/gradient分岐の大きな移植になる。上記payload/MAC見積もり以上の性能費用は未測定。
8. 200k再開不可。元UNSTABLEとresume guardを維持する。

## Provenance・verification・保護

数値source exact HEAD: `5e15d36d9c3c6db005e8cb2ac2be360e7569b4de`。
Forward-only初回source e405510、Backward初回58e83f4を別pilotとして保持する。
最終15 processは同一package/probe/SDK libraries、同一fixtures、各variant3 process。
QNN executes3,192、API failure0、fallback0、全outputs finite、input mutation/poison/canary failure0。
case別dataとRMS/relative/bit digestはprivate。allow-list exporterは15 reportsのcount、gate、
source/binary/runtime/device identity、independent-process digestを確認し、sanitized summaryをprivateに作る。

| 検証 | 結果 |
| --- | --- |
| Fast（Forward source） | PASS、96.7s |
| Fast（final source/exporter/report） | PASS、92.6s |
| Qnn profile | PASS、577.3s。Host contracts/shape、100-case double finite differences/FP32 oracle、QNN build/APK audit |
| QNN+HVX APK build | PASS、66s。W8 headless runnerでfixed APK audit |
| Muon W8 regression | PASS、114/114、maxAbs1.490116119e-8、relativeL2 4.40269854e-8、cosine1、RPC/finite成功、fallbackfalse |
| final device numerical matrix | Lane A PASS、Lane B accuracy FAILを別保存。全process API/ownership/finite gate成功 |
| exporter SelfTest / process consistency | PASS |
| device-probe smoke | PASS、BACKGROUND_CORRECTNESS、fixed APK audit、foreground takeover0 |

一度top-resumed不明で最終確認を実行前に停止した。端末同一性/awake/top-resumed/残存processなしをread-onlyで確認し、
全guardを再実行して新しいprocessで再開した。失敗artifactは保存し、数値失敗や6031と混同していない。
6031はUNRESOLVED / DORMANT / WATCHのまま。Formal training/release/full-model promotionはしていない。

build manifestにはsource files SHA256、compiler SHA256、SDK build/architecture/flags、package/probe binary identity。
各run ledgerにはstage後remote SHA256、APK/QNN/Skel、fixture digest、mode、power/thermal、foreground監視を保存。
最終documentation HEADは完了メッセージとprivate final identitiesへ記録し、実機sourceと区別する。

private evidence: ignored `build/reports/htp-fp32-layernorm-matmul/`、`build/fp32-custom/`、同prefix qnn-headless reports。
保護5主要artifactsと3 forensic identities（複製を含む5 forensic paths）、全既存branch refs、過去private indexを終了時にも照合した。
40k checkpointはhashだけ読み、deserializeしていない。100k baseline/41,647/42k evidenceとUNSTABLEは保存。
source以外のraw tensor、checkpoint、corpus、device identifiers、SDK binaries、private logsはcommitしない。
Push/PR/merge/main変更、history rewrite、学習updatesは0。
最終working treeはclean。Host20GiB/Device12GiBとactive runなしをprivate closeoutで再確認する。
