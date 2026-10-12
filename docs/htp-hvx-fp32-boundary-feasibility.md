# HTP/HVX FP32 boundary feasibility

開始source: `ed5f372f3182ba0302c17122987e4b510dd83d23`。
branch: `investigation/htp-hvx-fp32-boundary-feasibility`。
2026-10-12。結論は **`BLOCKED_REINJECTION`**。
Host↔DSPとReshapeのcopy経路はbit-exactだが、正しいFP32 dXをHTP consumerへ戻すと
Add(x,0)またはMultiply(x,1)だけでも既存の0.02 gateを超える。
DSP LayerNorm kernel、full-model graph分割、60k/200kを含む学習updatesは実行していない。

## 事前登録した判断基準

- Host/DSP: 256 finite FP32 raw patterns、3独立process、各6往復。
  全要素bit mismatch0、DSP独立fixture照合0、input mutation0、complete write、return0を別々に検査。
  poisonは各期待wordのbit complement。DSPはinteger memcpyのみで全要素を返す。
  DSP受信全wordを独立した整数metadata channelでも返し、Host送信bitと全要素照合する。
- QNN: FLOAT32 APP_WRITE→単一Reshape→APP_READと、Add(x,0)、Multiply(x,1)、Subtract(x,offset)を分ける。
  Reshapeもbackend loweringを経由するため、単純なメモリ共有または純粋なtransportとは仮定しない。
- D64既存20 recipes、full upstream1、double referenceとanchored FP32 Host oracleを再利用。
  input/centered/inverse/xhat/dY/dXhat/dX/dGamma/dBetaを検査する。
  Host oracleはDSP演算の代用ではなく、再投入する高精度fixtureの生成にのみ使う。
- dX/dGamma/dBetaはdoubleとの差≤0.02。xhatは従来ordinary0.002、登録済み低分散0.06を維持。
  input/centered/inverse/dY/dXhatの再投入は独立boundary diagnosticとしてFP32値との差0.002も報告。
  transfer bit-exactとconsumer accuracyを混同しない。
- HTPから取得済みの値をDSPへ渡すB3と、Host oracle fixtureをDSPで無演算echoしてHTPへ戻すB4を区別。
  B4はDSP LayerNormのPASSではない。
- 単独micrographの必要精度がFAILならfull model分割、DSP LayerNorm kernel、性能promotionへ進まない。
  別のcustom opや拡張FP32演算方式まで不可能とは断定しない。

## Reuse audit

既存`hexatrain_hvx_probe.run`のfloat sequence ABI、QAIC Stub/Skel生成、rpcmem、cdsp unsigned PD、
V81 buildを再利用。未使用mode8をdiagnostic専用で追加し、IDLは変更しない。
production Muonのmode5/7、W8 workers、optimizer publication、checkpoint/resumeは変更しない。
新しいRPC framework、CPU fallback、SDK変更はない。

production `nicopedia_hvx_muon.cpp`はpersistent Session/mutex、RPC全期間のbuffer ownership、
finite validationとpoisonを持つ。DSPはscratch allocationとworker completion後に結果をpublishする。
診断は独立Sessionで128-byte alignmentをHost/DSP両側で確認する。
raw copy試験にはFP32演算、cast、FMA、fast-mathの算術効果を含めない。
既存buildは`-mv81 -mhvx -mhvx-length=128B -O2`で、明示fast-mathは追加しない。

過去HTP MuonのNS5 relativeL2≈0.052は同じAPI成功/finiteとaccuracyの分離を要求するが、
MatMulとLNの誤差起源を同一とは断定しない。
FLOAT32/compensation、weights packing、activation fusionの否定的結果、context splittingの5010を再利用し、再探索しない。
旧LN backwardの0.005 upstreamとscale64→8→1はfull-upstream D64の証明ではない。
fixed-state reproducibility/bisectionではgraph構成・順序に依存した変動があったため、
今回のdynamic LN tap/native最終五出力もbitwise比較する。

## Transport accuracy — TRANSPORT_VERIFIED

実機sourceを`575d94661beafe5b53e9e050f0e04b5f500dfc18`へ固定。
3独立instrumentation process、同一APK/Skel、各6固定fixture往復と360 stage往復を実行した。
各process内の固定6回はQNN Runtime初期化・graph実行より先に行う。

| 検査 | 最終3 processの結果 |
| --- | --- |
| 固定fixture | 256有限FP32要素 × 6回 × 3 process = 4,608要素 |
| 固定fixture identity | FNV1a64 `0b2b772440d58a12`。source内integer generator、seed97 |
| 全transport | 366 calls/process、計1,098 calls。37,456要素/process、計112,368要素 |
| DSP独立fixture照合 | 全要素mismatch0 |
| DSP受信integer metadata vs Host raw bits | 全要素mismatch0 |
| DSP→Host float output vs Host raw bits | 全要素mismatch0 |
| input immutability / complete write | mutation0、期待wordのcomplement poisonから全面書込み |
| input/output finite / return / close | 全finite、全return0、close0。別々に判定 |
| alignment / domain | HostとDSP側のinput/output各128-byte gate、cdsp unsigned PD |

signed zero、FP16非表現値、0.25±0.001、1024±0.001、256/512/4096付近、
最小normal、subnormal、固定seedの幅広い有限raw patternsを含む。
今回はNaN/Inf輸送fixtureを実行せず、normal health gateへ混ぜない。
DSPでHVX浮動小数点計算を実行したという主張ではなく、既存HVX用FastRPC基盤上の
DSP integer-copy/receipt probeである。演算のない輸送にFP32情報損失は観測されなかった。

## QNN boundary accuracy — copyとconsumerの分離

APP_WRITE/APP_READは全てFLOAT32、callback capture無効、graph precision設定を再探索していない。
1 node、shape `[2,64]`のReshape/Add/Multiply/Subtractをそれぞれ独立graphで実行。
元のFLOAT32宣言を内部FP32演算の保証として扱わない。

| 経路 | 観測 |
| --- | --- |
| B1 Host→Reshape→APP_READ | raw boundary fixture128要素でbit mismatch0。D64 20-case×9 fieldsでもmismatch0 |
| B2 Host→Subtract→APP_READ | 1024±0.001の最初の64要素は全て0。referenceとの差0.0009765625 |
| B3 HTP APP_READ→Host staging→rpcmem→DSP | 7 HTP-derived fieldsの取得済みbit patternsを全て保持 |
| B4 oracle fixture→DSP無演算echo→Host staging→Reshape | 全要素bit-exact。dX reference gate20/20 PASS |
| B4同fixture→Add(x,0) / Multiply(x,1) | 各dX gate14/20 PASS、6/20 FAIL。消失はconsumer後 |

B3の7 fieldsはcentered/inverse/xhat/dXhat/dX/dGamma/dBeta。
inputとdYは追加のHost APP_WRITE controlであり、そのstage差0をHTP内部入力精度の証明にしない。
Host stagingを実際に使い、QNN tensorとDSPが同じnative memoryを直接共有するとは仮定しない。
DSPに渡せた値の精度と、HTPで既に失った精度は分離した。

raw boundary fixtureではReshapeのmismatch0に対し、Add0は105/128、Multiply1は97/128が異なる。
両consumerの最大absolute差は0.0009765625、全finite。
Add0によるnegative zeroの符号変化はIEEEの演算semanticsでも起こり得るため、
bit mismatch件数だけを精度不良の根拠にせず、offsetの微小差分とdX reference errorを根拠とした。

Reshapeがbackend内でどう最適化されたかは未観測であり、FP32 HTP arithmeticを証明しない。
一方、同じAPI境界でcopyがbit-exact、Subtractが差分を失うという比較は、
APP_WRITEへの提供そのものと算術後の観測を区別する直接証拠になる。
内部のoperand conversion、accumulator、store conversionのどれが単独原因かはさらに未分離。

## LayerNorm-specific accuracy

既存20 D64 recipes、2 rows、epsilon1e-5、full upstreamを変更せず使用。
double referenceとanchor-relative FP32 Host oracleを独立に評価。
常時0.5、precision compensation等の棄却候補は再探索しない。
既存100-case/D8–128 Host検証は再実行したが、新しいdevice boundaryはD64に限定した。

| fixture | 正しいHost FP32 dX→DSP→Reshapeのdouble差 | 同じ値→Add0のdouble差 | Multiply1 | 0.02 gate |
| --- | ---: | ---: | ---: | --- |
| low_variance | 1.09583254e-5 | 0.06067470365 | 同値 | FAIL |
| constant | 1.06853433e-5 | 0.04788665352 | 同値 | FAIL |
| large_mean_low_variance | 8.15331416e-6 | 0.04936928809 | 同値 | FAIL |
| epsilon_dominated | ≤1.25273592e-5 | 0.04710671548 | 同値 | FAIL |
| random_seed7 | ≤1.25273592e-5 | 0.05843990337 | 同値 | FAIL |
| random_seed97 | ≤1.25273592e-5 | 0.06232748910 | 同値 | FAIL |
| ordinary | ≤1.25273592e-5 | 2.49509778e-6 | 同値 | PASS |
| centered256/512/4096 | 各1.93885498e-8 | 各2.49509778e-6 | 同値 | PASS |

| 20-case maximum double error | DSP echo→Reshape | DSP echo→Add0 / Multiply1 |
| --- | ---: | ---: |
| xhat | 3.92462837e-7 | 6.04243836e-4 |
| dX | 1.25273591e-5 | 0.06232748910 |
| dGamma | 2.05071294e-7 | 2.25341448e-4 |
| dBeta | 0 | 0 |

forward xhat/dGamma/dBetaは20/20 PASS。dXだけでも採用条件は満たさない。
新しいinverse/input/centered等のboundary diagnostic criteriaも含め、
Add/Multiplyの計360 checks中32 FAIL/process。うち必須dXは12 FAIL（6 fixtures×2 consumers）。
すべてfiniteであり、nonfiniteを隠す処理もtolerance変更もない。
affine yや全モデルresidual/backwardはこの境界probeでは測定していない。

実数上identityのconsumerで、正確なFP32 fixtureとの差がconsumer後に増えるため、
DSP LayerNorm kernel自体が正しかったとしてもこのscalar FP32再投入方式の0.02 gateを救えない。
低分散0.06067470365とconstant0.04788665352は、既存研究のnearest finite FP16 lattice boundとも一致する。
これは測定したconsumer出力経路の制約であり、全QNN op/packageが同一内部型だという主張ではない。

今回のHTP-derived比較は旧primitive candidateのうちDYNAMIC_RANGEを使った。
512/4096の既知overflowで輸送probe自体が停止するのを避けるためで、恒久採用ではない。
low_varianceでHTP vs Host oracle dX差0.27229309082、inverse差0.488494873047、
constant dX差0.165649414062、large-offset dX差6.87336730957を観測。
これらはdoubleとの差ではない。旧source`ed5f372`のoriginal dX double error0.272300373917は
既存3-process aggregateを保全して参照し、新測定として再ラベルしない。

## 情報損失の分類とFull-model stop

| 分類 | 今回の証拠範囲 |
| --- | --- |
| A 上流HTPでの損失 | synthetic Subtractで微小差分を消失。LN centered/inverse誤差も取得。実際のG1上流tensorの精度は今回未測定 |
| B HTP→Host→DSP輸送 | 取得済みbit patternsは全要素保持。BLOCKED_TRANSPORTではない |
| C DSP FP32算術 | kernel未実装・未測定。Host oracle PASSをDSP PASSへ読み替えない |
| D DSP→HTP consumer | Add0/Multiply1で必須dX error FAIL。BLOCKED_REINJECTIONを直接支持 |
| E full-model tensor boundary | 条件未成立でgraph分割未実施。BLOCKED_GRAPH_INTEGRATIONと断定しない |

dynamic LN tap/native五出力は各20-case×3 processでbit mismatch0。
763 typed numerical records/processが全て一致し、SHA256は
`df6eb5b3687adb43846971e6bad3741b8b3e79e0246546eed357e7d06e74e849`。
これは入力またはcheckpoint hashではなく、allow-listed numerical metricsのdigest。
過去のgraph-order variabilityを棄却する包括的証明でもない。

userのconditional go gateに従い、micrographの必須consumer精度がFAILした段階で
G1 L19の新しいsplit/tap replayとDSP LayerNorm kernelを停止した。
保護40kをhash照合でread-only使用し、optimizer/data位置を伴う新しいfull-model replayは実施していない。
41,647/42kの過去finite evidenceは保存し、今回のkernel correctnessやintegration PASSへ転用しない。

## DSP kernel・数学・compile scope

**DSP_LAYERNORM_VERIFIEDは未成立。kernel実装0、training integration0。**
CPU oracleのanchor-relative centering、Kahan FP32 accumulation、scaled RMS、hypot denominator、
standard analytic backwardは既存100-case Host suiteでPASS。
今回のD64 oracle20-caseもdoubleとの差は上表の通りだが、DSP演算の代用ではない。
DSPのFMA contraction、fast-math、subnormal arithmetic、gradient reductionsは未評価。
integer copyのsubnormal保持を、subnormal浮動小数点演算の正しさと取り違えない。
LayerNormの定義、epsilon、LR、Muon/Aux Adamの数式は変更していない。

## Performanceと実装コスト

BACKGROUND_CORRECTNESSでのdiagnostic timing。EXCLUSIVE_BENCHMARK、clock固定、thermal設定変更はない。
RPC wallは同期呼び出し全体、DSP timerはcopy/全word照合/receipt writeのみ。
host packはrpcmem copyとpoison、postcheckはreturned vector/receipt照合を含む。
snapshotは整数receiptを返すため、純粋なproduction transportよりmetadataが多い。

| 128要素stage往復、各240 samples | process1 median µs | process2 | process3 |
| --- | ---: | ---: | ---: |
| RPC-inclusive wall | 496.172 | 495.365 | 279.505 |
| DSP copy/check/receipt | 8 | 8 | 7 |
| Host copy/poison | 20.833 | 20.7295 | 10.104 |
| Host validation | 62.969 | 62.864 | 31.563 |

このlogical array payloadは2,080 bytes（input/expected/output各512 bytes、receipt metadata544 bytes）。
wire framingやremote call引数の固定overheadは含まない。
固定256-word fixtureは各5 samples（最初の1回除外）、RPC medians298.386/443.750/296.145µs。
DSP medians87/85/87µsには独立fixtureのinteger generator検証が含まれ、
LayerNorm compute timeではない。全payload別metricsは[aggregate](results/htp-hvx-fp32-boundary/aggregates.json)に保存。
process間のlatency差が大きく、steady-state throughputやkernel速度改善を主張しない。
RPC−DSPにはmarshal/dispatch/queue/synchronization等が混在し、同期だけを測定した値ではない。

Host rpcmemは最大4096 wordsのinput/expected/outputと4104整数metadata、計65,568 bytesを所有。
APP_READ vectors・QNN memory・DSP scratchを含むRSSではない。
QNN6 probe graphs、最終3 processで18 create/finalize、1,809 successful executes、API failure0。
paired graph split overhead、graph setup/finalization wall、RSS/peak、実LN compute/full-step throughputは未測定。
LayerNorm correctnessが成立していないため、performance promotionは行わない。

既存Muonの141µs/128 bytes、321µs/約1MiB、787µs/約12.45MBは過去no-compute transport。
今回とはpayload、receipt、power/context履歴が違い、比較値を新しいLN latencyとして流用しない。
G1の38 LayerNorm×forward/backward×batch8=608 dependent calls/update、
x/y/dY/dXだけの9,961,472 bytes≈9.5MiB/updateという設計budgetは依然として推定。
今回の128-word median×608は約170–302msになるが、これは**異なるpayloadのreceipt probeを機械的に掛けた参考値**。
実LNの32 rows、saved state、graph boundaries、kernel computeを含む実測見積もりではない。
従ってBLOCKED_PERFORMANCEとは分類しない。

## 統合方式の判断

| 方式 | 数値・integration | 同期/transfer | 実装・maintenanceと次の投資 |
| --- | --- | --- | --- |
| A HTP custom op package | FP32保存/算術/consumerも新probeが必要。保証なし | package内接続ならHost往復を減らせる可能性 | package生成、FP32 tensor contract、V81 build/lifecycle、精度を維持するconsumerまで監査。小さなbit-copy/FP32 consumerのfeasibilityから着手する価値あり |
| B independent HVX FastRPC | 輸送PASS。既存HTP arithmeticへscalar値を戻す方式はBLOCKED_REINJECTION | naive608同期/update、Host stagingあり | Muon基盤は再利用可能だが、LN kernelだけ追加しても再投入誤差は残る。全面移植へ進まない |
| C 少分割hybrid | precision-critical backward segmentをDSP内に保持し、適切な境界までFP32で接続する設計は未検証 | 精度必要区間の拡張で同期を減らせる可能性 | LN間はattention/FFN/residual依存がある。38個の単純batch化では解決しない。範囲・payload・checkpoint semanticsを先に具体化する必要 |
| D QNN primitive composition | 既存dynamic scalingはrangeを改善、accuracyはBLOCKED。今回consumerでも再確認 | 外部RPCは不要 | 既に否定されたprecision設定を再探索しない。scalar output精度を超える多plane等は大規模consumer変更になるため本scope外 |

FP32を壊さないcopy boundaryは存在する。しかし現在のtraining graphに戻して
必要なconsumer算術を実行する経路まで、0.02 dX accuracyを維持できる方式は確認できなかった。
次の投資は、LN kernel単体の作成より先に、**FP32を保つconsumerを含む最小custom-op/segment境界**の確認。
native producer→custom op→consumer→APP_READのraw accuracy、配置/lowering、同期・memoryを測る。
この方式まで不可能と断定せず、precision-preserving routeがPASSした場合にのみ最小DSP LNを作る。

| Go classification | 判定 |
| --- | --- |
| TRANSPORT_VERIFIED | YES、上記finite fixture/HTP-derived outputsの実測範囲 |
| HTP_BOUNDARY_VERIFIED | copy-only ReshapeはYES。必要consumer精度を含むLayerNorm routeとしてはNO |
| DSP_LAYERNORM_VERIFIED | NO、conditional gate FAILで未実装 |
| INTEGRATION_FEASIBLE | NO、現行scalar consumerへの再投入はBLOCKED_REINJECTION。代替package/segmentはUNRESOLVED |
| PRODUCTION_READY | NO。production置換、training、checkpoint/resume/qualityの新validationなし |

## Regression、safety、provenance

既存Muon W8 ABIを同一APKでBACKGROUND_CORRECTNESS検証：114/114 matrices、
maxAbs1.490116119e-8、worst relativeL2 4.40269854e-8、cosine1、RPC成功、finite、fallbackfalse。
これはstandalone optimizer fixtureの回帰であり、保護G1 stateのtraining updatesではない。
IDL、production modes5/7、Muon/Aux Adam、shared G1 training LayerNorm builderの数式は変更していない。

| 検証 | 結果 |
| --- | --- |
| Fast initial / source / receipt / fail-closed | PASS、108.7 / 73.7 / 66.6 / 60.9s |
| 最初のQnn profile | Host runner include引数の誤りでFAIL。pinned check25.3s、QNN APK build461.5s、audit4.1sはPASS |
| 修正後Host | PASS、607.3s、全contract/diagnostic、reference、shape、optimizer/checkpoint/parity |
| 最終QNN+HVX build/APK audit | PASS。final primaryはsource575d946をbuild、repeat2/3は同じinstalled APK identityを確認 |
| 最終3 device probes | 診断完了・transport/API/finite gateはPASS。**consumer accuracyはBLOCKED_REINJECTION** |
| Muon W8 device ABI regression | PASS、114/114、foreground takeover0 |
| typed exporter | SelfTest、3 process/provenance/input identity照合、数値digest一致、allow-list export |

headless `status=SUCCESS`/wrapper PASSはauditが完了しtransport/QNNが健全だったことを表す。
consumerのgate0とdecisionBLOCKED_REINJECTIONを同時に保存し、candidateのaccuracy成功にはしない。
元のUNSTABLE runのstatus、resume拒否guard、40k/41k/41,646/42k evidenceを変更していない。
Formal training/promotion certification、final test、LR探索、production変更は行っていない。

実機QAIRT `2.48.40.260702151143`、QNN API2.37.0、HTP V81、Hexagon SDK6.6.0.0、
Hexagon tools19.0.07、Android NDK26.2.11394342。SDK/library hashとAPK ABI/path/2.47監査を保持。
3 process thermal0、32°C、foreground takeover0。
Host20GiB/Device12GiB gatesを全boundary起動で維持（final launch host≈44.2GB、device≈44.2–44.4GB）。
6031は再発せず、`UNRESOLVED / DORMANT / WATCH`を別incidentとして保持。

| commit / evidence | identityと意味 |
| --- | --- |
| base | ed5f372f3182ba0302c17122987e4b510dd83d23 |
| 初期probe/source | 72e69a6de25d3b1df9e37b9f53e67ca34f605381。1 pilot、6 QNN graphs/603 calls/366 RPC |
| integer receipt probe | e200f34bede8b835d9e331fe9a3f4b9110751be0。1 pilot、同数calls。全word受信監査追加 |
| 最終実機source | 575d94661beafe5b53e9e050f0e04b5f500dfc18。consumer nonfinite即停止、3独立process |
| 保護40k SHA256 | 89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d |
| dataset/cache/tokenizer/order | 保護identityを起点/終了時に照合。新しいtraining/evaluation sequenceを生成していない |

全5 boundary process（2 pilots+3 final）でQNN3,015 successful calls、RPC1,830 calls、API/RPC failure0。
public aggregateは最終3 processを主証拠として重複省略し、pilotsはprivateで保持する。
private evidenceはignored `build/reports/htp-hvx-boundary/` と同じprivate IDのheadless report。
private indexにsource、run IDs、device/runtime、APK/Skel/fixture/build hashes、thermal、runner/verificationを対応付ける。
final identitiesに保護8 hashes、branch refs、UNSTABLE、HEAD、clean working treeを記録する。
docs/exporter closeout commitは実機sourceとは別であり、最終HEADは完了メッセージとprivate final identityへ記録。

raw tensors、device identifiers、checkpoint、run IDs、APK/SDK binaries、corpus、local absolute pathはcommitしない。
既存100k成功experiment、元200k branch、42k forensic evidenceを保存。
Push/PR/main merge、履歴書換え、optimizer reset、CPU fallback、SDK変更、app data削除、UI foregroundを行っていない。
今回の結果から60k/200kの再開は認可できない。
