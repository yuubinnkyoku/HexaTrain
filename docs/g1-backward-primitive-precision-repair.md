# G1 Backward primitive precision attribution / selective repair

**新Customの両Standaloneは実DSP pilot PASS。Nativeと3-process正式認定は未完了。**
現在のblockerはtop-resumed情報の欠落であり、Native dValueを安全停止している。
標準primitiveの受信tensor境界に対する局所誤差を分離し、独立Host参照で旧gateを満たす
ordered FP32候補を実装した。標準FLOAT32 MatMul診断でDMA abort / API6031が発生し、
続くdefault標準MatMulと新Custom最初のexecuteも同じAPI6031 / process139で終了した。
初回の新Custom数値出力は得られなかったが、その後のdefault対照は100 executesを完走した。
同じfrozen binaryでSoftmaxとdValueを再開し、それぞれ全9,824 pairsを独立にPASSした。
Native、正式cohort、性能の成功を主張しない。
前回停止後の再開確認では同一端末とAwakeを再確認したが、障害後の再起動は未実施だった。
その後ユーザーから追加の再現確認指示があり、同一bootでの1プロセスに限定した別planを固定した。
**40k→60k / 200k trainingはNO-GO。**

## A. Baseline / provenance

指定研究cloneのclean tree、root/common dir、branch/upstream、完全HEADを確認した。
baseは `49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2`、作業branchは
`investigation/g1-backward-primitive-precision-repair`。
`e05f3539e2e0c0e461157a3abf1d9833d74b392b` と
`be4271efd1b8fb71f0c66f3ae7c35f04571aa78e` のancestryを確認した。
復旧済みの正本・既存manifestを再利用し、研究chainの再探索はしていない。

[precision plan](../host_tests/fp32_custom/backward_primitive_precision_plan.json) は
最初の新DSP診断前に `ce0091e` でcommitした。
[selection](../host_tests/fp32_custom/backward_repair_selection.json) と
[cohort transport supplement](../host_tests/fp32_custom/backward_repair_cohort_plan.json) も
対象の実機演算前にcommitした。gate、既存fixture、Forward package、Production builderを変更していない。

| Immutable input / existing package | SHA256 |
|---|---|
| Double QK fixture | `f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2` |
| Frozen gradient derivative | `c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b` |
| Existing Forward DSP | `962a98c6f30e4e8f610abba22f69cd2f08fb216d5ed111586b45d7c73f8991e7` |
| Existing Forward Host | `e2df381e1682ab12e68784ee22ddc468fe553d5cf12c778a167c75990a83ea0a` |
| Existing Forward probe | `911d66e8b0fc3ba56afd544b5c588f3ef4f0027ee08f92a96b809ee4d44b56f6` |

freshな標準Standaloneを両consumer・全4条件で実行した。
各条件Core 44×2×25=2,200、Holdout 16×2×8=256、各consumer合計9,824 pairs。
全API成功、finite、全面書き込み、poison/canary、入力不変、teardownと、数値FAILを分けて記録した。

| Existing standard consumer | Core maxAbs | Holdout maxAbs | Gate | Fresh outcome |
|---|---:|---:|---:|---|
| Softmax / dMaskedLogits | 0.3044056503066239 | 0.02663227731332496 | 0.002 | NUMERICAL_FAIL |
| dValue | 0.035564650305797585 | 0.00388396910312494 | 0.01 | NUMERICAL_FAIL |

Softmaxのaccuracy FAILは3,022 pairs、accuracy/observerのunion FAILは4,390。
observer差は2,036 pairs / 10,412 words、最大0.15625、signed-zero差4 words。
dValueはaccuracy FAIL 328 pairs、observer word差0。
旧FAILを新candidateの結果で上書きしていない。

## B. Received-input precision attribution

数学は `dot[r]=Σ P[r,j]*dP[r,j]`、`dMaskedLogits=P*(dP-dot)`、`dValue=P^T*dContext`。
Backward内の追加scaleはない。Forward scale bitsは既存の `0x3e3504f3`。
標準opのFLOAT32宣言と内部precisionを同一視しない。
標準opには独立したDSP operand receiptがないため、既知のAPP_WRITE words、入力不変検査、
Tapの実中間出力をtensor境界の根拠とする。不可視内部registerのwordsを直接観測したとは主張しない。

独立NumPy実装でFP32 / FP16 / BF16候補を同じ受信境界へ適用し、全caseのword一致数、
maxAbs、RMS、ULP分布、signed zero、subnormal、overflow/underflowを集計した。
modelは事前定義した候補であり、実出力から各要素に別の丸めを当てはめていない。

| Tap boundary | Received-input double maxAbs | FP16 operand / FP16 result model |
|---|---:|---|
| First product | 0.4091988129730453 | 2,514,944 / 2,514,944 words一致、各Tap observerで差0 |
| ReduceSum on actual product | 0.21875 | half-output double-sumでも1,834 / 78,592 words不一致 |
| Subtract on actual dP/dot | 0.74847412109375 | 2,514,944 / 2,514,944 words一致、各Tap observerで差0 |
| Final product on actual P/centered | 0.15047404170036316 | 2,514,944 / 2,514,944 words一致、各Tap observerで差0 |

最悪Subtract F14/H1/G9/index374では、FP16モデルのoperand変換差0.24847412109375と
出力丸め差0.5の和が実局所差0.74847412109375になる。これは**実受信dotで条件付けた**比較であり、
上流dot errorを再度Subtractの局所誤差へ加えていない。
Product最悪のsigned差はoperand変換−0.3015327973480453、出力丸め−0.107666015625。
Final最悪はそれぞれ0.09139201045036316、0.05908203125で、model residualは0。
異なるcaseのstage最大値を足してtotal gradient誤差とはしていない。

rank2/rank4、6種類の標準primitive graph、24 controls×2 headsを独立に評価し、576 executesを完走した。
correct FP32 productを外部入力したReduceSumだけでもmaxAbs=0.33431053313006487。
Product→NATIVE ReduceSumは0.3467024564743042、既知dotを渡したSubtractは0.84783935546875。
これらは変更されたcontrol graph自身の値であり、元の全chainの局所最大値とは別に扱う。
全control入力wordsをPCで独立再構築し、一致を確認した。

dValueのone-hot / identity / permutation / head-asymmetric controlsは、実HTP上のrank2/rank4で
期待wordsと一致した。transpose0=true / transpose1=falseの数学layoutを確認した。
全frozen casesでは「FP16 operands、ordered FP32 accumulation、FP16 output」のmodelが
2,514,532 / 2,514,944 wordsを説明するが、412 words不一致、最大model差0.0078125が残る。
product、accumulation順序、output丸めの内部実装は一意に特定できない。

独立large-magnitude controlでは、FP16-exact operandsとexact productsでも、
FP32なら有限なMatMul結果にrankごと512 words、計1,024 wordsのInfが出た。
入力変換だけでは説明できず、accumulationまたはoutput境界のrange制限を示す。
このcontrolのAPIは成功したが、finiteはFAIL。数値FAILを隠さず全件保存した。

分類は `STANDARD_BACKWARD_PRECISION_ATTRIBUTED` / `STANDARD_BACKWARD_PRECISION_MODEL_SUPPORTED`。
内部実装そのものは `INTERNAL_PRECISION_UNRESOLVED` を維持する。

## C. Fixed SDK / documented settings

QAIRT `2.48.40.260702151143`、QNN core API `2.37.0`、HTP V81、Hexagon SDK `6.6.0.0`、
Tools `19.0.07`、NDK `26.2.11394342` を固定した。
SDK public headers、graph configuration、backend generic configuration、既存Production source、
SDK内のrelaxed-precision rewrite helpersを監査した。
公開HTP headerで存在を確認したprecision / compensationを診断graphだけに指定した。
内部の `relaxed_precision_flag` を公開APIとみなして指定していない。
監査した公開headerには、この標準op群の内部FP32を保証するper-op switchを特定できなかった。

| Diagnostic setting | Softmax pilot | dValue diagnostic |
|---|---|---|
| Backend default | 全件accuracy FAIL / observer差あり | 全件accuracy FAIL / observer差0 |
| Explicit FLOAT32 | 600 pairs、246 union FAIL、observer差696 words | 部分実行後API6031 / exit139、accuracy集計未完了 |
| FLOAT32 + precision compensation | 同じ600 pairs、246 union FAIL、observer差696 words | 6031後に追加実行を止めた |

FLOAT32指定のAPI受理を内部FP32の証明としない。Production graph/backend設定は変更していない。

## D. Observer analysis

F2/H0/G9/index32のTap対NoTap最大差0.15625をfresh baselineで保持した。
standalone標準graphではObserved/NoReceiptラベル間にreceipt専用nodeは存在せず、
対応する同じTap条件のwordsは一致した。差はTapで中間をAPP_READへ出す条件とNoTapの間にある。
graph出力・中間materializationの違いは確認できるが、proprietary fusion / reduction order / precisionの
どれが最初の内部divergenceを生んだかは未確定。
**最初に直接比較できる差はfinal dMaskedLogits。NoTap内部の最初の差を見たとは主張しない。**
旧35/40-node Tap incidentの原因解明とも区別する。

## E. Host candidate matrix / selection

すべて同じfrozen FP32 Pとgradientを使用した。gate、G9/G12、mask、scaleを変えていない。
既存holdoutは過去に観測済みであり、新しい未見datasetとは呼ばない。

| Host arithmetic | Core maxAbs | Holdout maxAbs | Old gate |
|---|---:|---:|---|
| Softmax separate ordered FP32 | 6.3353929017e-5 | 2.2067718461e-6 | PASS ≤0.002 |
| Softmax full internal double → FP32 | 7.6129856268e-6 | 9.4710632936e-7 | PASS |
| dValue ordered FP32 TN32 | 1.6803001870e-5 | 2.7630283377e-6 | PASS ≤0.01 |
| dValue full internal double → FP32 | 3.6969525468e-6 | 4.0787588240e-7 | PASS |

FP32だけでgateを満たしたので、修復算術へ内部doubleを導入していない。
Softmaxの4 primitiveすべてに局所誤差があるため、個別置換なら4 nodes/headになる。
今回は同じFP32丸め境界を保つfused path 1 node/headを選んだ。速度を理由に選んだのではない。
新しい数学系統はSoftmaxとdValueの2系統、算術candidateはordered-FP32の1件に限定した。

## F. Custom implementation / compiler audit

[arithmetic](../host_tests/fp32_custom/backward_repair_arithmetic.h) と
[package](../host_tests/fp32_custom/backward_repair_package.cpp) は専用namespace `HexaBackwardRepair`。
Softmax / SoftmaxTapのObserved・NoReceipt variantsとDValueTN32の2 variantsを登録する。
旧HexaFp32 / SelectiveLn1 / MixedQkv / DoubleQkの名前、ABI、receipt、数学bodyを変更していない。

Softmaxは別々のFP32 product、increasing-column FP32 sum、FP32 subtract、FP32 final multiply。
sumの開始値は+0、FMAなし。融合後も丸め境界を維持する。
dValueは既存Scalar ordered MatMulの演算順・index契約を専用TN32 bodyへ適用した。
`P[t,i]*dContext[t,j]` をt=0..31で別々に丸めて加算する。
既存のcoupled kernel/registryを持つheaderや旧MatMul bodyを変更していない。

Host ABI試験は29,472 calls、全2,200/256 pairs、rank2/rank4、6 variantsでPASS。
独立oracleとのwords、double accuracy、alignment、shape、overlap、nonfinite拒否、
input immutability、poison/canary、complete writesを検査した。
Host preparationは6 kernelsすべてでarithmeticを拒否する。
Host graph構築24件で、旧Forward topology、同一P ID、単一producer、head分離、NoReceiptを確認した。
これらはHost証拠であり、実DSP arithmetic / lifetime認定ではない。

compilerはcontraction / fast-math / auto-vectorization / unrollを無効にした。
実assemblyのscalar `sfmpy` / `sfadd` / `sfsub` を確認し、LLVM FMA / fast / vector-reduceを拒否した。
新pathはscalar SF演算なので、HVX qfloat arithmeticのstrict-IEEE設定を使用したと誇張しない。
resource指定はQHPI HVX、算術命令はscalar。volatileによるproduct/addのstack storeを含む。

| Kernel variant | Largest single static frame bytes |
|---|---:|
| Softmax NoTap Observed / NoReceipt | 8,440 / 8,368 |
| Softmax Tap Observed / NoReceipt | 184 / 128 |
| dValue Observed / NoReceipt | 160 / 80 |
| Arithmetic helpers | 8 |

Softmax NoTapのexplicit scratchは8,320 bytes/head。上表は静的frameでありruntime peakではない。
未解決importはQHPI tensor/registration、memcpy/memset/strcmp、runtime unwind等。
NoReceipt specializationはreceiptコピー・timerを実行しない。

| Frozen new artifact | SHA256 |
|---|---|
| DSP | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` |
| Host preparation | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` |
| ARM probe | `23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05` |

compile sourceは `bb6aaceb42de505b29689a362ce8293687e7ab89`。
後続runnerのversioned permit / exclusive evidence処理は別source identityとして保存した。
最終bundleはv4のcompiled bytesをそのまま複製し、旧manifestも保存した。
v3/v4間のobject・assembly・IR・Host .soは一致したが、linked DSP ELF bytesは異なった。
そのlink差を算術変更やruntime結果とは扱わず、最終v4 bytesを固定した。

## G. Standalone DSP / runtime episode and recovery pilots

初回の新Custom Observed+Tap graphはregister / create / finalizeまで成功したが、
最初のexecuteで約10秒後にAPI6031、stderrにDMA abort 6033、process exit139を記録した。
数値pair出力はなく、full writes / finite / teardownの成功は確認できない。
同じ現象が直前の標準FLOAT32 MatMulとdefault標準MatMulにもあった。
共通runtime状態と新kernel固有の問題の因果は未確定。
NoTap variantは未実行であり、NoTapの大きいstack frameがこの最初のTap失敗の原因とはしない。

その後、同一binaryで実機Standaloneを再実行し、rawを独立PC oracleで照合した。

| Repaired DSP pilot | Core maxAbs | Holdout maxAbs | Gate | Arithmetic / observer word mismatches |
|---|---:|---:|---:|---:|
| Softmax / dMaskedLogits | 0.00006335392901668513 | 0.0000022067718461471486 | 0.002 | 0 / 0 |
| dValue TN32 | 0.0000168030018699028 | 0.0000027630283376822717 | 0.01 | 0 / 0 |

各consumerでCore 2,200 / Holdout 256を全4 observer条件で検証し、合計9,824 pairs、
4 graphs / 4,912 executesを完走した。API成功、finite、入力不変、全面書き込み、
poison/canary、DSP receipt、teardownを確認した。これは各1 OS processのpilotであり、
3独立processの正式認定へ昇格させない。
Softmax digestは `fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267`、
dValueは `64248fcdfaf2534c37f40bc961987d0a627f6d4c457b3ba7cc553de2ac316e41`。
旧6031の `UNRESOLVED / DORMANT / WATCH` 記録を残した上で、本研究初回episodeの新規再現を追加した。
初回episodeを消さず、後続試験での非再現と区別する。根本原因は未確定のままWATCHを維持する。

停止位置を保存stdoutと独立照合した。FLOAT32 MatMul診断は90 executesがAPI成功し、
最後の完了pairはObserved+Tap / F3/G14の両headだった。loop順から次のF3/G15での停止が
推定されるが、失敗executeのDSP受信wordsは得られておらず、特定operandが原因とはしない。
続くdefault MatMulと新Customは成功executeが0、rawはともに0 bytesだった。
共通のDMA abort / 6031というsignatureだけでは、同じ根本原因や新kernel本体の実行を証明できない。

再開時の読み取り専用preflightは同一端末、Awake、active run不在等を再確認してPASSした。
一方、uptimeと前回失敗evidenceの保存時刻から、障害後の再起動は未実施と確認した。
この確認で数値probeは実行しておらず、runtimeの復帰や新DSP精度の認定には用いない。
新しい確認とlog比較は別のprivate folderへ保存し、元の失敗evidenceを変更していない。

ユーザーの明示指示による同一boot再現確認は
[recheck plan](../host_tests/fp32_custom/backward_6031_same_boot_recheck_plan.json) に事前登録した。
対象は前回失敗した既存default MatMul、同一probe、ForwardLimit=1、1 OS processのみ。
fresh safety preflightを省略せず、API失敗の再発時は追加試行を止める。
このbounded再確認は新Custom修復の認定や全件精度検査とは別である。

このdefault対照は100 executes / 200 pairsを完走し、6031は再現しなかった。
actual OS exit4は旧数学gateの8 accuracy FAILによるもので、API失敗ではない。
finite / guards / teardown、observer word差0を独立に確認した。
前後のuptimeから推定したboot時刻差は約0.54秒で、再起動なしと整合した。
さらに旧Forward Pを14,356 numerical recordsで再確認し、独立合同gateはPASSした。
これらの回復対照は新Standaloneの直接精度証拠と混同しない。

dValueの最初の再開はsleep、続く2回は実行直前のtop-resumed不明で停止した。
いずれも数値probe開始前であり、別folderに保存した。手動wakeと通常アプリの維持後に
fresh preflightから再開し、4回目の再開folderで上記の全件pilotを得た。

## H. Conditional Native / actual gradient

両Standaloneのpilot gateは成立した。Native Softmax singleは開始したが、Observed+NoTap / F18/G13付近で
top-resumed不明となり、所有を確認したprobeへSIGINTを送り停止した。6031は観測していない。
所有processの終了を読み取りで確認し、partial raw 140,369,920 bytesを端末/Host SHA一致で保存した。
actual OS exitは未捕捉でありnull、coverage/teardownは未完了。API成功済みの一部pairを
Native PASSへ数えない。この時点ではNative dValue singleとsame-P dualは未実行であり、後続試験を以下に分ける。
この中断processではNATIVE Pのlifetime、入力不変、observer-free accuracy、actual total gradientを認定しない。
Host再注入でNative認定を代用していない。

後続の読み取り専用8 samplesはすべてAwake・top-resumed既知だったが、失敗瞬間のresponseは未保存だった。
[限定diagnostic plan](../host_tests/fp32_custom/backward_top_response_evidence_plan.json) と
[trace runner](../scripts/run_backward_repair_trace_probe.ps1) を追加し、失敗responseの記録だけを増やす。
元runnerはfrozenのまま残し、source比較で差分をfailure evidenceとorchestration identityに限定した。
Host self-testでは既知1 case、unknown2 casesを確認し、unknownは追加device callやretryなしで拒否した。
parser、監視周期、安全判定、所有process停止、DSP/Host/probe bytesは変えない。
trace実行時のorchestration SHAをprivate completionとidentityに保存し、独立assessorで照合する。

最初のtrace Native Softmaxは全9,824 pairs / 全4条件を完走し、独立照合でもPASSした。
total maxAbs=0.00010078856647055545、consumer-local=0.0000637084093888518、
P-induced actual=0.00009955926756788358。旧P-only bound=0.0002459099714532658と別に判定した。
observer / arithmetic word差はいずれも0、API / finite / 全面書き込み / teardownもPASS。
digestは `edcaa214b0833b00c77d8c410d9e32894eef7e6aaecbce0cfb9307933bfaa16d`。
これは1-process pilotであり、正式3-processへ数えない。

続く元runnerのNative dValueはObserved+NoTap / F8/G19付近で同じ安全monitor停止となった。
数値FAILとはせず、この中断も別evidenceとして保持する。trace runnerは元runnerと同じ判定を保ち、
初回full実行の独立PASS後、残りの既登録roleにも同じsource hashで利用する。

Native dValueのtrace実行もObserved+Tap / F40/G1付近で同じ安全停止となり、追加の算術再試行は止めた。
失敗瞬間の応答はSystem.String / 153,912 charactersの有効なActivityManager dumpだったが、
`topResumedActivity` と `mResumedActivity` がともに0行だった。単なる空応答やregex表記不一致とはしない。
別のfocus情報で既知のtop-resumedを代用せず、unknown判定と所有process停止を維持した。
所有processの終了を確認し、partial raw 49,704,960 bytesを端末/Host SHA一致で保存した。
actual OS exitはnull、全件Native dValue精度・teardownは未認定。原因は端末の前面状態報告として未解決である。

準備済みgraphはheadごとに旧Forward Softmax producerが作る1つのP IDを新consumerへ渡す。
Native graphのinput mapにはPを追加せず、外部dP/dContextだけをAPP_WRITEとする。
NoReceipt/NoTapではPを直接観測したとせず、observed anchorを評価上の条件付けに使うことを明示する。
そのgraph自身のfinal output accuracyはideal double Pに対して別途評価する。
測定器にはP-induced、consumer-local、total errorと旧P-only boundを分けた独立PC検査を実装した。
その実装準備を実機認定へ昇格させない。

## I. Performance

dual total-gradient accuracy gateが未成立のため、accuracy-first規則で実機costは未測定。
31 samples、first execute、median/p10/p90、finalize、
consumer arithmetic / receipt-inclusive / execute wall、node/tensor/memory inventoryの保存経路を準備した。
NoReceiptの内部timerはnull。std QNN内部timerも取得したと主張しない。
異なるobserver graphのwallを純粋な算術速度と呼ばず、Forward共通LN local backward controlを揃える。
旧約17.8msを新Backward costへ流用せず、19層×8 microbatchへ外挿していない。

## J. Fresh regressions / safety / evidence

| Check | Current evidence |
|---|---|
| Fast | closeout PASS、89.5秒。public artifact追加後にもFastを確認し、各logは別名で保存 |
| Host | PASS、511.0秒 |
| Qnn | PASS、444.9秒、fixed SDK build / APK audit |
| New arithmetic / preparation / graph Host suite | PASS、29,472 calls / 6 rejected preparation kernels / 24 graphs |
| Recovered independent references / shape / gradient export | PASS、3 Host processesで同じfrozen gradient SHA |
| New standard primitive controls | 576 executes、API成功、layout controls PASS、Inf 1,024 wordsの数値FAILを保存 |
| Fresh Double QK + custom Scale/Mask/Softmax + P | PASS、14,356 numerical records、joint failed predicates 0 |
| Fresh old standard Backward | 両consumerの既知accuracy FAILを再現 |
| Fresh C0 Host negative | expected exit4、9 FAILを再現 |
| Other required fresh HVX / device smoke / Muon W8 | top-resumed安全停止のため未完了。過去114/114をfresh成功へ転用しない |
| Formal profile / three-process cohort | 未実行。この保存は進捗とBLOCKEDの記録であり、正式結果の確定ではない |

fresh Forward合同値はP maxAbs=4.236482907815642e-7、row error core=2.2724270820617676e-7 /
holdout=2.682209014892578e-7、P-induced dScores bound=0.0002459099714532658、
dValue bound=0.000027855243723208197。source/legacy references、Core/Holdout、全4条件で旧gateを維持した。
これは新DSP Backwardの成功ではない。

旧standard Scale 1 FAIL、Mask 176 required FAIL、Softmax 224 local FAIL、cumulative P 36 FAILは
既存negativeとして保持した。未実行のfresh回帰をPASSとしない。
35/40 Tap incidentは `REPRODUCED / CAUSE UNRESOLVED` のまま。

各processで同一物理端末、SDK/compiler、approved existing APK、package/fixture/protected hashes、
storage、active training/competing QNNなし、awake/top-resumedをfresh確認した。
BACKGROUND_CORRECTNESS、foreground takeover=0。通常のbattery低下、35〜41℃の記録温度、
外部給電条件を研究停止理由にしていない。sleep時は停止し、ユーザーの手動wake後に再確認した。
6031の前後でもthermal status0 / health2を記録した。critical保護条件は維持した。
自動wake/unlock、force-stop、root、clock/power変更、APK交換、firmware/SDK変更はしていない。

Host preflightのhang、compiler引数のscalar splat、test main衝突、旧manifest互換の問題もprivateに保存し、
Host/runtime/mathの失敗を混同しないよう修正した。shared ADB serverは停止していない。
最初のnonfinite standard controlは途中停止したため、失敗を消さず、同じcontrolを全件記録する別probeを作った。
そのprobeはfinite FAILを明記し、新Customのfinite gateは弱めていない。

private evidenceは `build/reports/g1-backward-primitive-precision-repair/`、
public aggregateは [summary](results/g1-backward-primitive-precision-repair-summary.json)。
[allow-list exporter](../scripts/export_public_backward_precision.py) はraw/identity/coverageと独立数値を再検査する。
false PASS、欠落process、wrong binary、old FAIL上書き、observer downgrade、wrong gate、数値欠落、
raw field、protected mismatch、wrong role、未測定値の偽装、unsafe/duplicate pilot選択等の16件拒否self-testをPASSした。
raw tensors、APK hash、device IDs、endpoint、private logs、絶対local path、SDK binariesはcommitしない。

## K. Decision / resumption

最小の有望候補はfused ordered FP32 Softmax + ordered Scalar FP32 TN32 dValue。
Host数学、codegen、両Standaloneの実機pilotは成立した。
初回のAPI6031 / exit139を保存した上で、同一bootのdefault対照と後続pilotでは非再現だった。
根本原因は未確定であり、runtime恒久修復とはしない。現在はNative dValueがtop-resumed情報欠落でBLOCKED、
dual / 正式cohort / performanceは未実行である。継続にはAndroidがtop-resumedを継続報告する状態を確認し、
Awake、同一端末接続等のfresh安全条件を維持する。
再開時もfresh preflightから行い、既存evidenceは上書きしない。

Host側の最終照合ではprotected 5 files、開始時の50 refs、frozen repairのbinaryと23 source hashesを
再確認した。現在branchの追加commitだけを許容し、他のrefsは開始時と一致した。

再現入口は `build_backward_repair_probe.ps1`、`run_backward_repair_probe.ps1`、
`assess_backward_repair.py`。runnerにはfixed QAIRT root/Build ID、Hexagon SDK、NDK、
完了したQnn validation log、新しいprivate ReportRootを明示する。
frozen bundleのStandaloneを先に独立評価し、両方の全件gateが成立した場合だけ
`permit_backward_repair_native.py` が段階別permitを生成する。gateのwaiverはない。
Native single成功前にdualへ進まず、dual accuracy成功前に性能・5-role正式cohortへ進まない。
cohortは同じbinaryで3つの独立OS process、各process内で全required rolesを検査する計画。

手動でruntimeが回復した後の再開例を示す。`HEXAGON_SDK_ROOT` と `ANDROID_NDK_ROOT` は
事前照合済みの固定Hexagon SDK 6.6.0.0 / NDK 26.2.11394342 rootを明示する。
既存失敗folderを再利用せず、下記の新folderも存在したら別の未使用suffixを選ぶ。
最初に既存Forward Pのbounded recovery controlを実施し、fresh preflightと旧合同gateを確認する。

```powershell
. ./scripts/qairt_version.ps1
$repairArgs = @{
    QairtSdkRoot = $PhoneLmQairtSdkRoot
    ExpectedBuildId = $PhoneLmQairtBuildId
    HexagonSdkRoot = $env:HEXAGON_SDK_ROOT
    NdkRoot = $env:ANDROID_NDK_ROOT
    QnnValidationLog = 'build/reports/g1-backward-primitive-precision-repair/qnn-profile.txt'
    Candidate = 'repair-final'
    Mode = 'standalone_softmax'
    ReportRoot = 'build/reports/g1-backward-primitive-precision-repair/pilot-softmax-recovery'
}
./scripts/run_backward_repair_probe.ps1 @repairArgs
python scripts/assess_backward_repair.py `
    --report-root $repairArgs.ReportRoot `
    --output build/reports/g1-backward-primitive-precision-repair/softmax-recovery-assessment-private.json
```

API6031 / process139が再発したら同じruntimeで繰り返さず、完了記録とpartial rawを保存する。
上記のindependent assessorは不完全なAPI/coverageをPASSにせず拒否する。

PV/Gate/Output、実dContext/dP producers、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregationは
引き続き未認定。FULL_ATTENTION_BACKWARD / FULL_ATTENTION / FULL1_LAYER / FULL19_LAYER /
TRAINING_READYを認定しない。protected checkpoint decode/write、training、final_test、optimizer/LR/ABI/codec変更はない。
作業は専用branchの追加commitのみで、push/PR/merge/history rewriteなし。

## Final questions 1–35

| # | Answer |
|---:|---|
| 1 | 指定clone、完全base 49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2を使用。 |
| 2 | 旧Forward Pのsource/binary/fixtureを保持し、fresh合同gate PASS。 |
| 3 | 最初のProductから受信入力に対する局所誤差がある。FP16 operand/result modelが全Tap wordsを説明。 |
| 4 | Product/Subtract/Finalは各observer 2,514,944 wordsすべてFP16 model一致。Reduceは1,834不一致、MatMul closest modelは412不一致。BF16候補は全一致しない。 |
| 5 | 広範囲model支持であり、内部実装の確定証拠ではない。 |
| 6 | 分離できた。correct FP32 product→標準Reduce単体でも0.3343105。 |
| 7 | 実受信dP/dotに対するoperand変換差0.2484741＋出力丸め0.5。上流dot errorとは別。 |
| 8 | 0.1504740417。FP16 modelの変換差0.0913920＋出力丸め0.0590820。 |
| 9 | 実HTP構造controlsでtranspose/layoutは正しい。precision/orderは未確定、標準accuracy FAILとrange FAILを保持。 |
| 10 | freshに再現。2,036 pairs / 10,412 words、最大0.15625。 |
| 11 | 直接比較可能なのはfinal dMaskedLogits。NoTap内部の最初のdivergenceは未確定。 |
| 12 | FLOAT32 graph precisionとprecision compensationの公開APIがあった。 |
| 13 | Softmaxは設定だけで旧gateを満たさない。dValue設定診断はAPI6031で未完了。 |
| 14 | Product、Reduce、Subtract、FinalとdValue MatMulに修復が必要な局所誤差を確認。 |
| 15 | 全4 primitiveの置換になるため、丸め境界を保つfused pathを選択。性能優位は未主張。 |
| 16 | 別々のFP32 product / increasing-index sum / subtract / final product。TN32も別々のproduct/add、FMAなし。 |
| 17 | 既存Scalar orderedアルゴリズムとindex契約を新TN32 bodyへ適用。旧body/ABIは変更していない。 |
| 18 | 新DSP pilot core 6.335392901668513e-5、holdout 2.2067718461471486e-6。 |
| 19 | 新DSP pilot core 1.68030018699028e-5、holdout 2.7630283376822717e-6。 |
| 20 | 両consumerの実DSP pilotで全2,200/256、全4条件をPASS。 |
| 21 | 両consumerのpilotで4 observer条件のaccuracyをPASS。 |
| 22 | 両consumerのpilotでfinal observer差0 words、算術契約差0 words。 |
| 23 | 正式cohort 0 process。各Standaloneの1-process pilotを正式3-processへ数えない。 |
| 24 | 旧標準Softmax 0.30440565 / dValue 0.03556465のFAILをfreshに保持。 |
| 25 | codegen PASS。初回DMA abort / 6031 / exit139は保存。後続対照・両pilotは実行成功、初回原因は未確定。 |
| 26 | dual total-gradient accuracy未認定のため実DSP cost未測定。静的stack/scratchのみ監査。 |
| 27 | Softmax singleはtrace pilotで全件PASS。dValue singleは安全monitor中断で未認定。 |
| 28 | dual native未実行。 |
| 29 | Softmax single pilotのactual total maxAbs 0.00010078856647055545を旧gateでPASS。dValue/dual/正式cohortは未完了。 |
| 30 | Fast/Host/Qnnとfresh Forward PASS。他のrequired fresh HVX/smoke/W8は未完了、以前の114/114は歴史的証拠。 |
| 31 | 通常battery/temp/給電を停止条件にせず監視。critical保護とawake条件は維持。 |
| 32 | 固定SDK、同一端末、protected状態、非破壊操作を維持。checkpoint/optimizer/trainingを変更しない。 |
| 33 | fused ordered FP32 Softmax + Scalar ordered FP32 TN32が実DSP Standalone pilotをPASS。Native/正式認定は未完了。 |
| 34 | 次はStandalone→Nativeと正式cohort。その後もPV/Gate/Output、dQ/dK、scatter、projection、LN1 aggregationが残る。 |
| 35 | 40k→60k / 200kはNO-GOを維持。 |
