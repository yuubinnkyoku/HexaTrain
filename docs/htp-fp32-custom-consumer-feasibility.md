# HTP FP32 custom consumer feasibility

2026-10-12。研究base `5984154646e3154fe0aabf5e2033515cfd782282`。
専用branch `investigation/htp-fp32-custom-consumer`。
結論は **FP32_CUSTOM_CONSUMER_VERIFIED / FP32_SEGMENT_VERIFIED**。
固定SDKと実機で、Custom Multiply→native FLOAT32→Custom Addの経路は
FP32算術と必要gradient精度を維持した。LN末尾のscale→residual Addの最大double誤差は
`1.25273591038e-5`、既存基準`0.02`を満たす。
G1への **INTEGRATION_CANDIDATE** は特定したが、後続standard arithmeticには
**BLOCKED_CONSUMER** が残る。実G1のupstream/MatMul/32-token統合とproduction性能は **UNRESOLVED**。
LayerNorm kernel実装、production置換、学習再開は行っていない。

## 事前登録した比較

固定QAIRT `2.48.40.260702151143`、QNN core API2.37.0、V81、
Hexagon SDK6.6.0.0/tools19.0.07、NDK26.2.11394342。
production graph、LayerNorm数学、optimizer、thresholdは変更しない。

- Copy: finite raw words、signed zero、subnormal、FP16格子外をbitwise比較。
- Add/Multiply: 動的2入力。独立Host FP32とDSP scalar FP32にbitwise一致を要求。
  `-ffp-contract=off`、vectorization disabled、fast-mathなし。
  raw arithmeticはAddで最小subnormalを加算、Multiplyで0.5を乗算する。
  full finite input rangeを含むが数学的overflowを意図するfixtureは使わない。
- Affineと連結: `round32(round32(x*b)+c)`、FMA contractionなし。
  同じ式のdouble referenceとの差も別指標で評価。
- D64全20 recipes、full upstream1。anchor-relative Host oracleのdX/dGamma/dBetaを
  consumer入力とする。理想double LN gradientへ同じconsumer式を適用し、各maxAbs≤0.02。
  単独FP32算術一致、LN gradient error、transport一致を分離する。
- 実依存tail: 既存double stagesのbracketとinverse/DをFP32へroundして供給。
  `bracket*(inverse/D)+residual`を2ノードおよびfused opで比較。
  residual fixtureは既存full-upstream dY。実checkpoint activationを使用しない。
  このtailはLayerNorm全体の実装でも実G1 replayでもない。
- APP_WRITE→Custom Multiply→NATIVE FLOAT32→Custom Add→APP_READ。
  consumerの全入力raw wordsをinteger receiptに返す。tapあり/なしを別processで比較。
- Custom→Reshape/standard Add/Multiplyと同入力・同式のstandard単独を比較。
  standard arithmeticの不一致はcandidate成功へ読み替えない。
- API成功、DSP専用marker、実行counter、全operand receipt、全面poison上書き、
  input mutation0、全出力finite、CPU fallbackなしを独立確認。
- 最終成功候補は同一binaryで3独立process。CopyだけでFP32算術を証明しない。
- accuracy合格候補だけ33回のbackground timing（firstと32 steady samples）。
  kernel timerにはreceipt overheadが含まれ、演算単体の時間と断定しない。
  性能のproduction上限は未定義。今回BLOCKED_PERFORMANCEの断定は行わない。

## 実装と能力監査

SDK実ファイルのQHPI header、QHPI sample、FP16 sample、HTP Makefile、
local migration guideを参照。SDK/third-party sourceをrepositoryへコピーしていない。
自作のdiagnostic packageはflat FLOAT32 DDR input/outputとINT32 receiptを登録する。
ARM host libraryはprepare用、Hexagon libraryはexecute用。CPU backendは選択しない。
HVX worker上でscalar FP32命令を実行する構成であり、HVX vector FP32実装とは主張しない。

package buildは成功。初回CPU registrationは`qhpi_register_ops_v1`の未解決symbolで4005。
固定`libQnnHtpPrepare`をhost libraryのDT_NEEDEDへ追加して解消。
`DSP` targetは4003（invalid target）；固定backendは`CPU`と`HTP`で登録成功。
初期MAIN resourceにはruntime warningがあり、HVX resourceへ変更して再検証した。
これら初期試行は昇格証拠に含めない。

| 監査項目 | 固定SDK / 実機で確認した範囲 |
| --- | --- |
| package生成/build | QHPIで自作packageを記述し、ARM/Hexagon shared libraryのbuild成功。SDK generatorはWindows/Linux scriptが存在するがXML生成は使っていない |
| V81 target | SDK HTP Makefileに`htp_v81`。sample defaultはSDK6.4.0/tools19.0.04で今回のpinと異なる。sampleのmulti-SDK Makefileをそのまま使用せず、固定6.6.0.0/19.0.07で`-mv81` buildした |
| Android host registration | ARM libraryを`backendRegisterOpPackage(...,"CPU")`でprepare側へ登録。これはQNN CPU backendの選択やCPU tensor算術ではない |
| HTP library / dependencies | Hexagon libraryを同APIの`"HTP"` targetへ登録。ARM側はlibQnnHtpPrepare/libc++、DSP側は固定V81 SkelのQHPI symbolsとHAP timerに依存 |
| unsigned実行 | 固定SDKのunsigned V81 Skelをhash照合してstageし、非rootのAndroid shell processでcustom kernelを実行。app UID内のpackage packaging/lifecycleは未統合。PD security設定を変更していない |
| FLOAT32 I/O | QNN FLOAT32 / QHPI FLOAT32、flat4/direct/DDR_ONLY signaturesで登録・finalize・execute成功 |
| FLOAT32 arithmetic | DSP固有marker、counter、全operand receipt、scalar `sfadd`/`sfmpy` disassembly、独立FP32 reference一致で確認。FLOAT32宣言だけを根拠にしない |
| Scratch / VTCM / DDR | custom HAP allocationなし、scratch0、custom tensorはDDR_ONLY。SDK scratch資料は実行時HAP_malloc/freeを規定し、signatureはDDR/TCM制約を持つ。runtime全体は8MiB VTCMとDDR yield bufferを確保するため、process全体VTCM0とは主張しない |
| primitiveとの接続 | Custom→Reshapeは精度保持。Custom→standard Add/MultiplyはAPI/finite成功だがgradient accuracy FAIL |
| lowering / dtype | custom inputのinteger receiptは供給した全FP32 operand wordsと一致。連結consumer receiptもHost FP32中間と一致。builtin内部のoperand/accumulator/store型は未分離 |
| registration/finalize/execute | 最終33 processでCPU/HTP registration、device/context/graph create、全tensor/node登録、finalize、execute、freeのreturn0 |
| 出力保存精度 | Copy raw128 wordsとgradient全要素でbitwise一致。Add/Multiply/Affineの算術後FP32値も一致。native consumer受信とAPP_READ最終出力を分けて確認 |

QHPIはv1 operator/kernel descriptorsとpackage interface1.4を使用する。
prepare logの`libnative v85 (requested v81)`はARM側prepare simulatorの表示であり、
Hexagon packageの`-mv81`と固定V81 Stub/SkelをV85へ変更したものではない。
最終候補に`invalid resource flag`はない。core registryがopを見つける前のwarningの後に
QHPI registryが各custom kernelをmatchし、DSP専用receiptで実executeを確認した。

公式オンラインの[Op Package Migration Guide](https://docs.qualcomm.com/nav/home/writing_opPackage_migration_guide.html?product=924033590759186372)
はversionごとの再buildを要求する。現在の固定header/libraryを正とし、最新の記述を実行可否の保証にしない。
固定SDKのQHPI文書はversioned C ABIを提供すると説明する。legacy C++ migrationの一般論と区別し、
今回確認したbuild/runtimeの組合せ以外のABI互換性は未検証。

## 数値結果

最終実機build sourceは`44dc205ae91c4884dfae7056f9d08e4d9e55fa39`。
11 graph variants×3独立process、同じprobe/package/Skelと同じfixture generator。
QNN successful executes2,895、custom kernel receipt3,213、typed numerical checks2,160。
API failure0、input mutation0、poison残存0、nonfinite0、CPU fallback0。
standard graphのprocess exit4は意図したaccuracy不合格であり、API failureではない。

Host FP32 referenceはAndroid NDKで独立compileしたscalar式、DSPはHexagon toolsでcompileした
scalar式。double LN referenceとanchor-relative Host oracleは既存診断を再利用した。
DSPでLN数学そのものを実行した結果ではない。

| 経路 | dX max double error | dGamma | dBeta | 20 recipesのdouble gate |
| --- | ---: | ---: | ---: | --- |
| Copy | 1.25273591038e-5 | 2.05071293724e-7 | 0 | 20/20 PASS |
| Custom Add | 1.64777568727e-5 | 2.05071293724e-7 | 0 | 20/20 PASS |
| Custom Multiply | 1.74432031201e-5 | 3.00283783172e-7 | 0 | 20/20 PASS |
| Custom Multiply→Custom Add | 1.74432031201e-5 | 3.00283783172e-7 | 5.96046447754e-8 | 20/20 PASS |
| Fused Affine | 1.74432031201e-5 | 3.00283783172e-7 | 5.96046447754e-8 | 20/20 PASS |
| Custom Multiply→Reshape | 1.74432031201e-5 | 3.00283783172e-7 | 0 | 20/20 PASS |
| standard Add | 0.0624905674103 | 0.000741185548158 | 0.000163078308105 | dX14/20 PASS |
| standard Multiply | 0.0799960718734 | 0.000612052153109 | 0.000101923942566 | dX13/20 PASS |
| Custom Multiply→standard Add | 0.0623850237797 | 0.000612052153109 | 0.000101923942566 | dX14/20 PASS |
| Custom Multiply→standard Multiply | 0.0390193839174 | 0.000193901466128 | 3.82214784622e-5 | dX14/20 PASS |

上表のgradientは理想LN gradientに対応するconsumer式を適用した後の比較。
standard単独とcustom単独は同じ動的input/同じ式、hybrid Addとcustom chainも同じ式である。
各表値は3 processで一致した。全customの独立FP32演算referenceとの差とbit mismatchは0。
standard/hybridのtyped `gate`はFP32一致も含むため、そのFAIL数をdouble 0.02 gateのFAIL数と混ぜない。
dGamma/dBetaのdouble accuracyはstandardでも全20 recipes PASSだが、必要dXがFAILした。
元のAdd(x,0)/Multiply(x,1)の単純再投入を再試行していない。

Copyのraw128-word fixtureはsigned zero、最小normal/subnormal、最大finite、FP16格子外、
1024±0.001とseed97 finite patternsを含む。3 processで全bit一致。
Addのraw arithmeticは最小subnormalを動的tensorとして加算し、Multiplyは0.5を乗算。
各3 processでHost FP32とbit一致。identity最適化で算術PASSと判定していない。
選んだraw fixturesのPASSは全IEEE corner case/overflow semanticsの包括的証明ではない。

## 最小実依存segmentとnative精度

実LN backward builderの末尾は`bracket * (inverse / D)`。
その出力をresidual Addが消費する2演算を、独立graphで確認した。
bracket/inverseは既存double stagesからround32、residualは既存full-upstream dY。
出力は**理想double LN dX + residual**と比較し、inverse丸めも誤差へ含めた。

| 必須fixture | tail→residual max double error |
| --- | ---: |
| ordinary | 4.81955565457e-8 |
| low_variance | 1.09583253902e-5 |
| constant | 9.83198819426e-6 |
| large_mean_low_variance | 8.15331415538e-6 |
| epsilon_dominated | 1.02730125633e-5 |
| random_seed7 | 1.25273591038e-5 |
| random_seed97 | 1.08817127682e-5 |
| centered512 / centered4096（既存recipe名half_scale_limit / very_large_signed） | 各4.81955565457e-8 |

全20 recipes×3 process、native連結、float tap連結、fused版の各方式がPASS。
最終出力とinputの全execute digestは3方式とも
`36b18a4fce14a0f0f3d8cfd0d864b694f9b8c5088e638d8996a96d141d1c6949`。
native中間のconsumer receiptはHost FP32 multiply結果と全word一致。
APP_READでしか精度を保存できないという制約は今回観測しなかった。

精度保持点はproducer store→native tensor→consumer load→custom算術→APP_READまで。
standard算術へ出る位置では上表のFAILが生じる。
float tap追加による最終値のobserver effectはこのfixture集合では0。
ただし全customはinteger receiptをAPP_READで返す診断版である。
receiptを完全に除去したproduction graphのloweringは未検証。

## 性能予備測定

128要素、33繰返しのうちperf index1–32をsteady samplesとした。
accuracy合格のcustom経路だけ追加timingを実施。専用benchmarkやclock固定はしていない。
全33 processのthermal status0、battery温度38–41°C、focus takeover0。
battery/charging stateは各private ledgerに保存。power voteは変更していない。
QNN INFO logging有効で、receiptとfinite validationがある。

| 経路 | 3 processのsteady median execute µs | first graph execute µs | finalize µs |
| --- | --- | --- | --- |
| Add | 838.568 / 981.146 / 881.4065 | 3050.938 / 2982.917 / 3009.636 | 27453.854 / 26233.854 / 25267.031 |
| Multiply | 1037.8385 / 1036.7445 / 865.3645 | 3127.760 / 3068.646 / 3005.885 | 23760.313 / 23779.167 / 18952.708 |
| Fused Affine | 834.1145 / 840.7555 / 842.3695 | 3016.718 / 2880.885 / 3037.031 | 22969.375 / 17516.823 / 25436.198 |
| Native 2-node chain | 968.984 / 1081.2765 / 1077.344 | 2945.677 / 3007.604 / 3299.583 | 24044.219 / 25135.000 / 23442.605 |
| Float tap chain | 853.203 / 857.630 / 864.5055 | 3073.697 / 2966.406 / 3034.323 | 27649.531 / 20577.604 / 24599.115 |

first列はAPI内部call wall。steady列はAPIとそのstatus出力を含む診断wall。
tapが速いケースがあり、fused/native/tapの速度順位を一般化しない。
HAP timerのkernel区間はAdd11–16µs、Multiply11–17µs、Affine17–26µs。
input receipt copy/check用metadata writeを含むため、純粋な算術compute時間ではない。
chainの各kernelは11–16µsで、Host→DSP同期だけを独立測定した値でもない。

CPU/HTP registration、device/context/graph creationは個別API timingをprivateに記録。
例としてnative chainの1 processはCPU registration56,832.864µs、HTP registration18.594µs、
device create90,215.469µs、context create7,166.875µs、graph create65,234.062µs。
backend dlopen単独時間とpackage loading内部の細分化は未測定。
Host buffer allocation/poison/input snapshot copyとreceipt/finite postcheckは別区間で記録したが、
allocationとcopyそれぞれの純時間、QNN内部transfer/queue/syncの分離は未測定。

単独5 tensors/1 node、native chain7 tensors/2 nodes、fused5 tensors/1 node。
Copyにも共通callerの未使用b/c APP_WRITEがあり、tensor countは最適なproduction構成ではない。
receiptは各kernel `(3*128+8)*4=1,568 bytes`。
custom scratch0、custom DDR_ONLY、直接VTCM使用0。
native chain runtime allocation estimate63,488 bytes、DDR mempool6,144 bytesというprepare logがある。
process RSSはbackend/prepare等を含み、segmentだけのpeakではない。
測定したchain/fused/tapのHost peak RSSは約75,920–76,984KiB。
DSP全体peak memoryは未測定。runtime VTCM予約8MiBとyield bufferは別に存在する。

### Full-model budgetの限界

G1 L19は38 LN、microbatch8。今回のbackward tailだけなら304 segment invocations/updateが設計上対応する。
既存forward/backward独立FastRPC案の608 calls/updateとは数え方が違う。
fused128要素micrograph medianを304回機械的に掛けると約254–256ms、nativeなら約295–329ms。
これは分割graphの診断wallの参考値で、32-token tensor、LN reductions、saved state、
実MatMul、tensor transfer、graph splitを含むfull-model予測ではない。
実G1 tensorは32×64=2,048要素で今回の16倍。係数はprobeでは要素へ展開した。

packageを既存graph内へ置けば、custom nodeごとに外部Host RPCを要求する設計を避けられる可能性がある。
しかし現時点のmicrograph execute wallを304 nodeの追加costとして掛けることもできない。
融合、broadcast、共有buffer、scheduler/tiling、MatMulを跨ぐ範囲の実測が必要。
kernelが速くてもgraphを多数分割すれば学習が遅くなり得る。
production上限は事前定義されていないのでBLOCKED_PERFORMANCEを断定しない。

独立FastRPCの同等算術segmentは今回は実装しなかった。
Custom Op不成立時の代替条件が成立しなかったためである。
既存no-compute transport timingやMuon timingをLN segmentのcompute性能へ読み替えない。
Custom Opが独立FastRPCより速いという実測比較はUNRESOLVED。

## G1 dependencyの静的確認

`qnn_runtime_transformer_training_generalized.inc`の実ノード定義による。

- LN2 dX=`DRESIDUAL1_LN` → residual Add(upstream,dX)=`DRESIDUAL1`。
  そこから`dwo=ATTENTION_CONTEXT^T*DRESIDUAL1`と
  `dcontext=DRESIDUAL1*wo^T`へ分岐し、attention backwardへ進む。
- LN1 dX=`DINPUT_NORM` → residual Add(DRESIDUAL1,dX)=`DINPUT`。
  次の下層のupstreamとなり`dw2=RELU^T*upstream`と`drelu=upstream*w2^T`へ分岐。
  最下層では`dembedding=oneHot^T*DINPUT`。
- dGamma/dBetaはparameter-gradient APP_READへmaterializeされる。
  実行層のHost gradient取得とMuon/Aux Adamへの後続経路は既存のまま。
- backwardの末尾Multiplyとresidual Addは連続する実依存の2演算。
  その後のMatMulが合格するかはAdd(x,0)の既知FAILから推定せず、未測定境界として扱う。
- producer側ではLNのcentered Subtractとsaved inverse/xhatが必要。
  1024±0.001の差分消失は既存synthetic証拠。実G1 producerの初損失点は未確定。
  後段custom opで失った差分を復元できるとは主張しない。

具体的な配置候補はFP32 LN backwardの末尾とresidual Addを同じcustom execution domainに置くこと。
native FLOAT32連結が可能なので、Host stagingを挟むことは今回のprecision proofに必須ではない。
ただし今回のequal-shape係数はG1のinverse/D row broadcastをそのまま受け取るABIではない。
32-token shapeとbroadcast対応をdiagnosticで先に検証する必要がある。

dGamma/dBetaを含むparameter gradientsはAPP_READへ取得した後、
training runnerがHost FP32で各microbatch gradientを`0.125f`倍してaccumulateし、
既存HVX Muon / Host Aux Adamへ渡す。gradient propagationのHTP MatMul経路とは分ける。
今回これらproduction処理、Muon W8 ABI、IDL、mode5/7、checkpoint codecを変更していない。

| 精度境界 | 現在の判断 |
| --- | --- |
| 最初に必要情報を失うproducer | synthetic large-offset Subtractは確定済み。実G1のfirst lossはUNRESOLVED |
| 最初の必要consumer | LN1/LN2直後のresidual Add。Custom Addで精度保持できることをfixtureで実証 |
| 次の分岐境界 | dcontext/dwo、下層drelu/dw2、最下層dembeddingのMatMul。未測定 |
| FP32が最後に必要な境界 | この最小試験だけでは確定できない。residualで止めてstandard MatMulへ戻せるとは保証しない |

### 次の設計投資と200k判断

1. QNN custom arithmeticは固定環境で実行できる。scalar FP32で必要gradient gateを満たす。
2. Custom Op間のnative FLOAT32は、この2-node/fixture集合で精度を維持する。
3. 実G1 residual consumerの数式を含む最小segmentはPASS。実G1 state/32-token graphへの接続は未実施。
4. 少なくともLN backward末尾とresidual Addを含める。saved xhat/inverse/centered情報を失うなら
   producer側へ拡張が必要。MatMulまでの必要範囲は未確定で、全Transformer移植を提案しない。
5. Custom graph内接続にはHost往復削減の設計上の利点がある。FastRPCとの実測優劣は未確定。
6. **小さなD64 DSP LN diagnostic kernelへ投資する価値はある。**
   anchor-relative FP32/Kahan oracleを実DSPで検証し、native residual consumerまで含める次段階を提案する。
   実装難度にはreductionの演算順序、subnormal/FMA、row broadcast、保存state、tiling、scratch ownershipがある。
   その後のMatMul acceptanceを測定する前に38 LN/19層へ拡大しない。
7. **200k再開は不可。** 元UNSTABLEを維持し、40k→60k/200k、final_test、LR探索は実施しない。

この成功はLayerNorm forward/backward全体のDSP correctness、全19層統合、production-readyの証明ではない。

## 検証・provenance

| 工程 | 結果 |
| --- | --- |
| Fast（初期probe/runner） | PASS、52.6s |
| Fast（最終source/文書/exporter） | PASS、54.5s |
| Qnn profile | PASS、303.5s。pinned core/check、Host contracts/shape/numerical diagnostic、QNN APK build/audit |
| final custom package/probe build | PASS、固定V81/ARM compilers、source/binary hash manifest |
| final custom device probes | 11 variants×3 process。API/finite/ownership PASS、custom accuracy PASS、standard consumer accuracy FAILを区別 |
| QNN+HVX APK build/audit | 初回shell引用不足でFAIL、修正後42s build成功・audit PASS |
| Muon W8 headless regression | PASS、114/114、maxAbs1.490116119e-8、worst relativeL2 4.40269854e-8、RPC成功、finite、fallbackfalse |
| device-probe headless smoke | PASS、BACKGROUND_CORRECTNESS、fixed APK identity、focus takeover0、thermal0、40°C |
| typed exporter | SelfTest PASS、3-process count/同一binary/同一device/input-output digest、既存0.02 gateを確認 |

今回のprofileはfeasibility diagnostic検証であり、Formal training/release promotionは未実施。
既存Host anchored oracle100-caseのgateもQnn profileのHost診断で維持した。
新しいforward/xhat device qualityを測定したとは主張しない。

branchは冒頭の専用branch。実機numerical sourceのexact HEADは
`44dc205ae91c4884dfae7056f9d08e4d9e55fa39`。
runner closeoutと文書commitは別で、最終repository HEADは完了メッセージとprivate final identitiesに記録する。
private build-identityはpackage/probe/reference/fixtureのsource SHA256、compiler SHA256、
build source HEAD、FP contraction/vectorization flags、package/probe binary SHA256を持つ。
各private process ledgerはstage後のremote hash、runtime/Skel、APK audit、端末、thermal/power、modeを持つ。
public summaryはtyped allow-list exporterから生成し、privateに保持する。
異なる3 process間とnative/tap/fusedのinput-output digest一致を確認した。

再現入口は`scripts/build_fp32_custom_probe.ps1`、`scripts/run_fp32_custom_probe.ps1`、
`scripts/export_public_fp32_custom_consumer.ps1`。
buildは明示QairtSdkRoot/ExpectedBuildId/HexagonSdkRoot/NdkRootを要求し、
runはcommitted source、build/source/binary identity、保護identity、storage、active-run guardを要求する。
fixture生成は既存`layernorm_*_reference.h`とinteger boundary fixture。
source/archive hashes、SDK actual file audit、disassembly、33 process reports、sanitized aggregateは
ignored `build/reports/htp-fp32-custom-consumer/`と`build/fp32-custom/`に保持する。
W8 evidenceは同prefixのprivate qnn-headless report。

既存8 protected artifacts、base研究branch、100k baseline/UNSTABLE/forensic refsを開始時と終了時に照合した。
今回の専用作業branchの通常commit進行を除き、既存refsは不変。
最終closeoutは保護artifact PASS、UNSTABLE維持、storage gateとclean working treeをprivateで記録する。

## 保護とscope

保護8 artifactsのSHA256、既存UNSTABLE状態、branch refsをprivateで照合。
checkpointをdeserializeせず、hash検査だけ実施する。
Host20GiB/Device12GiB、active runなし、端末同一性、pinned APK audit、
BACKGROUND_CORRECTNESS、2秒top-resumed監視を維持する。
学習updates0、60k/200kなし、final_testなし、全LN移植なし。
6031は別incident `UNRESOLVED / DORMANT / WATCH`のまま。
Push/PR/main merge、SDK/firmware/root/SELinux変更なし。
private evidenceはignored `build/reports/htp-fp32-custom-consumer/`。
raw tensor、device identity、local絶対path、APK/SDK binariesはcommitしない。
