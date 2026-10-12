# HTP LayerNorm numerical redesign

開始source `bc6b977482fdf3e355a6b86b50aab53eb46e9c0d`。
専用branch `investigation/htp-layernorm-numerical-redesign`。既存experimentとUNSTABLEを保護する。

## 結論

**BLOCKED_ACCURACY。恒久LayerNormを採用していない。40k→60k / 200kは未開始、今回のtraining updatesは0。**
dynamic range scalingは今回の512/4096を含む全100組の最終outputsをfiniteにしたが、低分散backward accuracyを満たさない。
mean形式のbackward、scale8、明示FLOAT32、precision compensationも不合格。
direct LayerNormは大振幅のforwardを改善するが、大offset＋低分散で入力差を失う。
現在試したprimitive経路では、最終dXの出力精度自体が0.02 gateの障害であり、単なる式の並べ替えでは解消しない。
これは全ての可能なQNN graph/custom opが不可能という証明ではない。

D64の0.27230037は、forward由来のinverse/xhat誤差とbackwardでの追加丸めの複合である。
独立processで全3,574数値recordが完全一致。計2,652 QNN executesは全return0、fallback0。
CPU reference/Host/buildのPASSを、device accuracyのPASSへ読み替えない。

## 事前登録した条件とgate

この節は新しいdevice結果を見る前に固定する。D8/16/32/64/128、2 rows。
旧16 recipesにfixed-seed random 1/7/31/97（通常、低分散、大振幅、大offset低分散）を追加。
512/4096を今回はrequiredへ昇格。全20 recipes × 5 dimensions =100組/候補。

- Double analytic / central difference: max absolute error <0.0005、h=1e-6と2e-6。
- y/xhat: ordinary・大振幅・境界は既存0.002、low varianceは前回の事前bound0.06を維持。
- dX/dGamma/dBeta: 各max absolute error <=0.02。upstream_scale=1。
- 全outputsとtaps finite。QNN return0とfiniteを別判定。tap有無bitwise一致を検査。
- 正常trajectoryのbitwise parityを無言で変更しない。今回の候補はdiagnostic graphのみ。
- 合格leaf候補があっても、full-model known failureとhealth gate未実施なら60kを開始しない。

候補はPREVIOUS、DYNAMIC_RANGEのみ変更、MEAN_BACKWARDのみ変更、SCALE8のみ変更、
FLOAT32_CONFIGのみ変更、COMPENSATION_CONFIGのみ変更を独立して比較する。
dynamicはs=128/max(128,max(abs(centered)))、epsilon_s=epsilon*s²。squareはscale後の一経路のみ。
mean backwardはinv*(dxhat-mean(dxhat)-xhat*mean(dxhat*xhat))。
scale8は過去のprecision aidを現input/上流gradientで再評価する診断で、production採用ではない。

まず旧graph各stageとdouble/FP32を比較し、forward値だけをHTP値へ置換するdouble backward replayで
inverseのみ、xhatのみ、両方の寄与を測る。tap bracketとinv/Dのdouble積と最終dXの差を
最後のmultiplyの局所誤差として別記する。寄与は非線形なので単純加算しない。

primitive graphはAdd/Subtract/Multiply/Divide/Sqrt/Rsqrt、Reciprocal（1/xのDivide）、
ReduceMean/ReduceSum。正値のみ、zero denominator/negative sqrtを除外する。
FP32宣言と内部精度を区別し、IEEE binary16 latticeへの一致は観測の比較対象に限定する。
native/tap、独立process、defaultと明示FLOAT32を比較する。

上の条件はdevice測定前のcommitに保存した。以下は測定結果である。

補助的に、FP32 DSP kernelの設計oracleをHostのみで検査する。
anchor-relative mean/center、Kahan accumulation、scaled RMSとhypot(RMS,sqrt(epsilon))を使う。
全100ケースにforward0.002・各gradient0.02を適用。CPU実行はreference/設計検査だけであり、
runtimeに接続しない。DSP/HVXでの精度・性能・graph integrationを保証するものではない。

## A. D64 backward errorの起点と増幅

標準recipeはrow0=`0.25±0.001`、row1=`±1`、gamma/beta/dYは非自明なbinary fractions。
比較対象は実際のFP32入力をdoubleへ変換したLayerNormで、入力をHTP精度へ丸め直してreferenceにしていない。
全stageのmax/RMS/relative error（relative denominator floor0.001）、finiteな範囲、nonfinite数を取得した。
集計は2 rows全体であり、row0のscalarと混同しない。

| stage | 観測 / max absolute error |
| --- | ---: |
| mean | 7.45058e−9 |
| centered | 2.34320760e−5、row0 magnitude0.0009765625 |
| square / variance | 4.63148356e−8、row0 variance9.53674316406e−7 |
| variance+epsilon | row0 1.09672546387e−5 |
| inverse | row0 302、ideal301.51149325、error0.48850675 |
| xhat | 0.00658798285 |
| dXhat / sum(dXhat) / D*dXhat / first difference | いずれも0 |
| dXhat*xhat | 0.00314915212 |
| sum(dXhat*xhat) | 0.00356090104 |
| projected bracket | 0.00215240014 |
| dX | **0.272300373917**、RMS0.08924089999 |
| dGamma / dBeta | 0.00247236855 / 0 |

小さいmean errorから、重大な差が最初に観測されるのはcentering。
この入力でdXhatやその単独reductionが初発原因という証拠はなく、上のbinary upstreamではexactだった。
centeringで変化したvarianceとepsilon roundingがinverseへ影響し、xhatとprojectionを経てdXへ増幅する。
stageの物理単位は異なり、dX用の0.02を全intermediateへ適用して「最初の違反」とは呼ばない。

観測variance+epsilonにdouble rsqrtを適用すると301.961126205。HTP rsqrtの302との差は0.0388737952。
したがってinverse全誤差0.48850675の全てがrsqrt演算だけの誤差ではない。
quantized variance/epsilonを入力とする段階の寄与と、rsqrt自体の有限誤差を分離できる。

同じgamma/dYで、forwardの一部だけをHTP値へ置換してdouble backwardを再計算した:

| 介入（backward演算はdouble） | ideal dXとの差 |
| --- | ---: |
| inverseだけHTP値 | 0.232196837470 |
| xhatだけHTP値 | 0.00983413219732 |
| inverseとxhat両方HTP値 | **0.242046902858** |
| 両方HTP値のdouble backwardと実HTP dX | 0.0563721992075 |
| tap bracket * tap(inv/D)のdouble積と実HTP dX | **0.05322265625** |

最後の行は最終multiplyで生まれる局所丸めで、forward errorとは別に0.02を超える。
各行はmaxを取るelementが同じとは限らず、非線形介入なので数値を単純加算しない。
forward誤差だけでもFAIL、backward自身にも追加誤差があるという結論である。

### 出力精度の障害

有限な観測dXは比較対象のIEEE binary16格子上に一致した（PREVIOUSの全100組でfinite elementの格子距離0）。
これは内部の全演算がIEEE binary16という断定ではない。
ideal dX各elementを最も近い同格子へ置いたときの最大誤差は、低分散recipeで**0.06067470365**、
constantで**0.04788665352**。いずれも0.02より大きい。
この格子制約を保つ最終出力では、途中計算をexactにしてもgateに到達しない。
SCALE8のconstant dX errorは0.04788665352まで改善したが、なおFAILである。
より高精度なdXの生成・保存経路が必要であり、許容誤差の拡大やupstream縮小を解決策にしない。
最終reviewで、診断用格子距離helperのrange外処理を「IEEE変換のInf」から「最も近い有限endpoint」へ修正した。
±65505/1e8のHost edge testsを追加。測定matrixのideal/finite HTP dXは全て±65504内で、
今回の格子下限やdevice recordsに影響しない。これはmodel値をclampする処理ではない。

### 大offsetの情報消失

`1024±0.001`はFP32 buffer内で差を保持するが、HTP subtractionではcenteredが全0になる。
independent Subtract graphにもa=`1024±0.001`, b=`1024`を直接渡し、全0を観測した。
情報はsubtractまでの入力変換または演算中で失われており、後段squareやbackward起因ではない。
APP_WRITE import時とkernel operand変換時のどちらかまでは未分離である。
direct LayerNormもxhat error0.295066653643となる。center後のscalingは失われた差を復元できない。
このrecipeのforward-only double backward errorは6.81282260844、実HTP dX error6.87336948344。

## B. Primitive演算の実測とrange制約

全tensorはFLOAT32宣言。2x64の独立graphで9演算、13入力、default/明示FLOAT32を比較した。
createと、利用可能なgraphSetConfigの両方へ精度指定を渡し、設定成功を記録。
default/FLOAT32の117 primitive recordsは完全一致し、低精度interface挙動を精度指定では改善できなかった。

| 入力 / 演算 | HTP観測 | 意味 |
| --- | --- | --- |
| 255.875² | 65472 finite | 最後に試したfinite側 |
| 256² | +Inf | overflow側。正確な閾値はこの間のbracketまで |
| ReduceSum(64 x 512) | 32768 | この範囲はfinite |
| ReduceSum(64 x 1024) | +Inf | reduction出力のrangeも有限性を失う |
| ReduceMean(64 x 4096) | 4096 | SumとMeanのoverflow挙動を同一と仮定しない |
| Add(1e−8,1e−8), Sqrt(1e−8), Mean(1e−8) | 0 | 小さい正値の情報消失 |
| 5.96046447754e−8のMean | 同じ値 | 試した範囲でsubnormal相当の値は残る |
| 1e−7のMean | 1.19209289551e−7 | 丸めを観測 |
| (1e−5)² | 0 | square underflow |
| Rsqrt(1e−5) | 316 | 理想約316.22777と異なる |
| Reciprocal via Divide(1,1e−5) | 65504 | backend側の有限飽和を観測 |
| Divide(1e−5,1e−5) | 0.65576171875 | 有限でも数学的には1と大きく異なる |
| Rsqrt(1e−8) | 12848 | Sqrt/Add等と同じ単純丸めモデルでは説明し切れない |

ReciprocalはSDKに独立op macroがなく1/xのDivideで試した。アプリ側でclamp/NaN置換した結果ではない。
backend自身の飽和/近似を確認したため、Inf検知だけでは正しさを保証できない。
underflowのexact cutoff、内部accumulator format、全演算共通のprecisionモデルは未確定。
RsqrtやDivideを理想演算で置き換えてHTP結果を推測せず、各opの観測を根拠とする。
これらはoptimizerを持たない診断graphで、各executeで固定入力を供給し全出力bufferをpoisonしている。
range反例の非有限値を次の入力・parameter・checkpointへ継承せず、QNN nonzeroではsuite自体を停止する。

## C. 数学と候補比較

正のsについて、`v_s=mean((s*c)^2)`, `epsilon_s=epsilon*s²`,
`inv=s/sqrt(v_s+epsilon_s)`, `xhat=c*inv`, `y=gamma*xhat+beta`。
実数では標準LayerNormと同じ。sが入力依存でも同じ関数として恒等的に相殺し、
sの導関数を独立の学習量としてbackwardへ持ち込む必要はない。
max/切替境界でも等価なLayerNorm自体はepsilon>0で滑らかだが、rounded device関数の連続性/accuracyは別検証である。
gamma/betaとdGamma/dBetaの定義は変更していない。

dynamicでは`c_s`のabsolute maxを実数上128以下とする。測定したsquare overflow境界256に対し
2倍のmagnitude余裕、4倍のsquare余裕を持つ。ReduceMeanを使用し、未選択の危険squareを別経路で実行しない。
constant rowではdenominator=max(128,0)でs=1、divide-by-zeroを作らない。
この設計でもcentering前の情報消失、巨大input差分overflow、scaled epsilon underflow、最終gradient精度は別問題である。
任意FP32入力を安全にする証明として採用しない。
今回の境界fixturesは以前の255.875 selectorの前後である。dynamic自身の128付近だけを
狙った新fixture、さらに大きいactivation/epsilon domainは未測定で、採用時には追加gateが必要。

| 候補 | PASS組 /100 | 最終5 outputs finite組 | nodes / tensors | 決定 |
| --- | ---: | ---: | ---: | --- |
| PREVIOUS selective0.5 | 46 | 85 | 29 /42 | accuracy/range FAIL |
| DYNAMIC_RANGE | 58 | 100 | 30 /45 | **BLOCKED_ACCURACY** |
| MEAN_BACKWARD | 46 | 85 | 27 /42 | accuracy/range FAIL |
| SCALE8 | 36 | 70 | 29 /42 | accuracy/range FAIL |
| FLOAT32_CONFIG | 46 | 85 | 29 /42 | PREVIOUSと全最終outputs bits同じ、FAIL |
| COMPENSATION_CONFIG | 46 | 85 | 29 /42 | 同上、FAIL |

finite組数は最終outputsのerror fieldsが全finiteであることから算出。全tapのfiniteも合格判定に含める。
不合格caseで、最終finiteだけを全intermediateのaccuracy certificationと呼ばない。
SCALE8はselectorを保持し、通常branch s=8/epsilon*64、危険branch s=0.5/epsilon*0.25とした診断。
当時のControl固定8 recipeの長期結果とは異なる条件である。

D64 dynamicの512/4096は全outputs/taps finite、y error6.25e−6、dX error2.4951e−6、dGamma error約1.8750e−6。
全5 dimensionsでこの2 recipesはPASS。random large amplitude seed31もD64でy error0.00111689、dX error6.55868e−7。
一方、低分散0.25±0.001のdX errorは0.27230037のまま。constant0.16564399、大offset6.87336948も改善しない。
MEAN_BACKWARDも同じ低分散errorだった。低分散forward/dXを悪化させるscale8のdX errorは0.36397936。

direct opはforward-onlyで独立比較した。512/4096のD64 xhat error約5e−6、低分散0.00634384。
大offsetはFAIL。backward/inverse出力のないこのprobeをforward/backward候補のPASSとはしない。

## D. Independent reference / normal-range parity

double two-pass reference、FP32版、全X/gamma/betaのcentral differencesを別に確認。
100ケース、2 perturbation sizes、既存selectiveとdynamicについて数学的同等性を確認。
CPU analytic/numeric worst3.02508124577e−5で0.0005 gateを通過。
dynamic同等性誤差はroundoff範囲で1e−10より小さい。
CPU FP32のnaive absolute-mean版も大offset randomでは精度が不十分な例があるため、「FP32なら演算順序不問」とはしない。

tap有無は600 candidate/dimension/recipe組全てで、y/xhat/dX/dGamma/dBetaのfloat bits一致。
独立process再実行も同じAPK・同じsourceで3,574数値record完全一致。
同一process内のtap/nativeの2 executesを2独立training runsとは数えない。

PREVIOUSに対する最終outputs bitsが変わった組数は、dynamic33、mean-backward3、scale8 45、
FLOAT32/compensation0（各100組）。この差を隠して通常trajectoryへ導入していない。
dynamicは今回target128を使うため、以前の255.875 thresholdまでの全domainについてbitwise保証をしない。
正常prefix41k/41,646の既存checkpoint parity証拠は保存したが、**新候補のfull-model parityは未実施**。

旧legacy backwardは低分散upstream0.005という別fixtureで、過去にPASSしたがfull-upstreamを保証しない。
過去のscale64→8はcentered-scale後squareの直接overflowを解消したbounded range修正であり、
今回の未発動低分散gradient accuracyや大offsetのloss of significanceを保証する修正ではなかった。
今回のAは同種のsquare overflowがscale1・新しい大activation分布で発生したもの。
Bの低分散/出力格子問題は、旧coverageが露呈させなかった別のaccuracy制約である。

## E. 既知failureと長期試験の停止

元41,647/microbatch1/layer17 LN1の有限−269.75→平方Inf→LN2/FFN→logits NaNのorigin判定は維持。
高LRがそのactivationを生んだ原因までは未確定であり、今回のleaf testでMuon/Aux Adam起因とはしない。
新dynamicの±269.75 leafはPASSしたが、全accuracy gateはFAILしているので
full-model41,647 replay・40k→42k・40k→60kを新条件で開始していない。
synthetic leafの突破を、実際の失敗microbatchの全gradient correctnessと呼ばない。

既存selectiveの40k→42k、2,000 updates/16,000 calls finiteというevidenceは前回のsourceのまま保持。
正常prefixのwhole checkpoint identityも変更しない。これらは新dynamic trajectoryの成果ではない。
42k/45k/50k/55k/60kの新しいoptimizer、gate、quality、throughputはない。
QNN API successとnumerical failureを区別し、元UNSTABLEのresume guardを解除しない。
100k baseline・LR schedule・parameter grouping・checkpoint serializationは変更なし。final_testは未使用。
6031は別のUNRESOLVED / DORMANT / WATCH、今回発生0。

## F. FP32 DSP/HVX経路への設計判断とコスト

次の有望な設計は、FP32を入力からdX保存まで維持する専用forward/backward kernelである。
Host設計oracleは以下を使用し、全100組で厳しいforward0.002とgradient0.02を通過。
最大dX error **1.65460269841e−5**。DSPでの合格ではなく、数式/FP32演算設計の実現可能性を示す。

- anchor=x[0]、delta meanをKahan FP32で蓄積、center=(x-anchor)-mean(delta)。大offsetを足し戻してから差し引かない。
- a=max(abs(center))、scaled sum-of-squaresからRMSを計算。a=0を分岐し、zero除算を避ける。
- denominator=hypot(RMS,sqrt(epsilon))、inv=1/denominator。大center自体をsquareしない。
- xhat=center*inv、標準mean形式backward、dGamma/dBetaも補償付きFP32蓄積。

実数ではvariance+epsilonを変えない。tiny epsilon・極端なFP32 inputs・巨大upstream・shape全域は
この100組で証明されていない。oracleのdiagnostic varianceはrange外でoverflowし得る。
DSP compilerのfast-math/FMA、sqrt/hypot、FP32 reduction/storeとbuffer lifetimeを別途検証する。
このoracleはRuntimeへ接続しておらず、CPU fallbackにも使用していない。

| 統合案 | 必要な変更 / 定量的な規模 |
| --- | --- |
| QNN HTP custom op package | forward/backwardの2kernel、dtype/shape登録、y/xhat/invおよびdX/dGamma/dBetaのcontract、package build/audit。kernel入出力がFP32を保つことをbit-patternで確認。周辺primitiveで再量子化されるなら全gradientのgateも必要 |
| 独立HVX FastRPC | 現行fused training graphをLN境界で分割、同期/lifetime/state管理、FP32 tensor転送とregistry接続が必要。G1には38 norms、forward+backward76依存kernel/microbatch、batch8では608 calls/updateとなる単純分割案 |

R32/D64でxhat+inv保存は1norm8,320 bytes、38norm316,160 bytes/microbatch。
単純FastRPC分割のx/y/dY/dXだけのlogical payloadは9,961,472 bytes/update（約9.5MiB）。
gamma/beta、saved-state転送、HTP境界との二重転送、allocator/copy費用はこれに含まない。
これはcall/payload設計の下限予算であり、測ったlatencyやRSSではない。
custom opならhost往復を避けられる可能性があるが、SDK/packageが実精度を保てる保証は未取得。
既存Muon HVX RPCがあることは、Normalizationのkernel/graph契約が完成している証拠ではない。

diagnostic dynamicはpreviousより1node/3tensors増、mean-backwardは2nodes減。
production trainingのgraphは変更しておらず、採用済みoverheadは0。
accuracy成立後に行うべきpaired finalization/time、RSS/peak、full-step throughput、checkpoint/resume performanceは
今回は未実施。過去trajectoryの速度差を新LayerNormのcausal overheadとして流用しない。

## G. 検証・再現・未解決事項

| 検証 | 結果 |
| --- | --- |
| pre-device Fast | PASS、53.5s（最初のFast77.8sもPASS） |
| report/export後のFast | PASS、61.0s。privacy/diff/parser、通常host/JVM gate |
| 最終review Fast | PASS、54.1s。格子helper edge補強後。100-case oracle metricsはbyte一致 |
| Qnn gate | PASS、633.9s。pinned check、全Host258.8s、build366.1s、audit3.5s。optional inventoryはadvisory、core complete |
| 最終Host | PASS、277.6s。100-case independent/dynamic gradientsとFP32 oracle、shape、optimizer/registry/checkpoint/resume、既存diagnostic/parity battery |
| 新suite PRIMARY / INDEPENDENT_REPEAT | **accuracy FAIL**、各1,326/1,326 QNN成功、failure0、fallback0 |
| 最終Device legacy backward | PASS、63.9s、BACKGROUND_CORRECTNESS。旧0.005低分散upstream fixtureの合格であり、新accuracy FAILを置き換えない |
| headless build / installed APK / foreground | pinned build/audit、同一APK identity確認、両process focus takeover0 |
| thermal / storage | 両process thermal0、battery temperature34°C、Host20GiB/Device12GiB launch gateを維持 |
| exporter | typed allow-list SelfTest、2 private reports検証、独立再exportのbyte hash一致 |
| 新candidate full-model / stress60k | **BLOCKED_ACCURACY / 未実施** |

```powershell
.\scripts\run_layernorm_stability_validation.ps1 -Suite layernorm-redesign `
  -QairtSdkRoot $qairtRoot -ExpectedBuildId '2.48.40.260702151143'
# 独立repeat: 同一installed APKをauditしたうえで -SkipBuild -SkipInstall。
# 現候補のFAILEDはscientific accuracy failureで、transport successと区別する。
.\scripts\export_public_layernorm_redesign.ps1 -SelfTest
```

将来DSP candidateが上の必須matrixを通過した後、41k/41,646と既知failure stateで全必要gradientsを確認。
correctnessのためbitwise parityを変更する場合は新implementation version/新trajectoryとして明示する。
その後だけ独立40k forkで42k→45k→50k→55k→60kを事前固定Long Linear条件で観測し、
optimizer/data cursor/storage/thermal/QNN/HVXをfail-closedで確認する。
途中修正は同じrunを継続せず別runとする。60k finiteから200kを保証しない。
**現時点の200k判断はNO-GO。** limitsを広げたりLRを下げてleaf accuracyをPASS扱いにしない。

## H. Provenanceと保全

両新device runsのexact sourceは`29da9599f84ae171cc7b897714c08e21a50b3e46`。
そのcommitに事前criteriaと候補実装を保存。final docs/exportのcommitはGit履歴とprivate最終identityで解決。
public report/export確定commitは`4f20bc313c626a43dfbeb923a831aa237b421e14`。
原experiment `6bca2a7a24f04f8b8b2d4084feb337a12c21e00a`、旧investigation
`1a57e3ae214fd320004b8f9f32715ab86f50c986`、旧validation `bc6b977482fdf3e355a6b86b50aab53eb46e9c0d`は保護。

| artifact | 保護SHA256 |
| --- | --- |
| 40k checkpoint | 89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d |
| expanded cache | bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf |
| tokenizer | 9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798 |
| manifest | e8a77e06abab3da5ff7fe26eb9744c3bffad8cf5fbf006bbf4c7f582393961e4 |

original run-plan hash/statusUNSTABLEも照合。sourceのshared training/generalized builderに今回の数式変更なし。
test input identityは同commitのprocedural recipes、100のdimension/name組、fixed seeds1/7/31/97、
固定gamma/beta/dY/epsilonと同一APKで解決する。両processのtyped numerical records 3,340件のSHA256は
`53c7e51bf1ed584249bc7a94851fe00214a2a2598131d2560f72695bf4af120d`。
これは入力bufferやcheckpointのhashではない。

private base=`build/reports/htp-layernorm-redesign/`。
`initial-identities.json`、`public-index.json`、各`validation-plan.json`/`device-report.txt`、Host/reference/oracle/verify logs、
headless同一private IDのAPK audit/device identity/foreground/thermalを保存。
公開[aggregates](results/htp-layernorm-redesign/aggregates.json)は全dimensionのcase metricsとD64 stage/primitive aggregates、
repeatの完全一致hashのみ。raw tensors、corpus、checkpoint、device/APK identifiers、run IDs、ローカル絶対pathは公開しない。

最終identity記録でprotected hashes・branches・original statusとclean working treeを確認する。
push/PR/main merge、reset/rebase/amend、削除、SDK/firmware変更、UI foreground、optimizer reset、failureの成功化を行わない。
