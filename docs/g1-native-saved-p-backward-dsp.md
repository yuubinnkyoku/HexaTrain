# G1 Native Saved-P Backward DSP — standalone accuracy stop

**Saved-P Backward dual-consumer native closureは未認定。** Production相当の標準QNN
consumerをHTP backendで実行したところ、Standalone段階で両gradient gateを超えた。
NATIVE single / dual、NATIVE lifetime、NATIVE observer-free、3-process正式cohort、
Backward追加cost測定へは進んでいない。**40k→60k / 200k trainingはNO-GO。**

分類は `BLOCKED_SOFTMAX_BACKWARD_ACCURACY`、`BLOCKED_DVALUE_ACCURACY`、
`BLOCKED_OBSERVER_INDEPENDENCE`、`UNRESOLVED`。前二者はStandaloneでの停止理由であり、
未実行のNATIVE graphが数値FAILしたという意味ではない。

## A. Provenance / unchanged Forward

- 指定された研究cloneで実施。session cloneは実装先にしていない。絶対pathはprivate evidenceのみ。
- 開始時の研究clone HEADは `e05f3539e2e0c0e461157a3abf1d9833d74b392b`。
  cleanを確認し、指定base `be4271efd1b8fb71f0c66f3ae7c35f04571aa78e` から
  `investigation/g1-native-saved-p-backward-dsp` を作成した。
- `e05f353` が `be4271e` のancestorであることを確認。差分は6件のHost監査・plan・provenance追加のみ。
  復旧時にForward DSP数学、package、fixture、production builderは変更されていない。
- 新device planは最初のDevice arithmetic前に `2d14efc59dfb15c50faf260c79d8efe83a431e5c`
  でcommit。raw evidenceをbinary保存へ変更する補足planも、該当binaryの実機演算前にcommitした。
  gate / fixture / graph / precision設定は変更していない。
- 新probeのbuild sourceは `b81f3c9`。後続はHost監査・exporter・documentationの変更。
  同じ下記probe bytesを両Standaloneで使用した。既存DSP/Host packageは再利用し、新DSP kernelは追加していない。

| Identity | SHA256 |
|---|---|
| Protected Double QK DSP | `962a98c6f30e4e8f610abba22f69cd2f08fb216d5ed111586b45d7c73f8991e7` |
| Protected Host preparation | `e2df381e1682ab12e68784ee22ddc468fe553d5cf12c778a167c75990a83ea0a` |
| Protected Forward probe | `911d66e8b0fc3ba56afd544b5c588f3ef4f0027ee08f92a96b809ee4d44b56f6` |
| New standalone probe | `d53e5dbd3bb919f1c3f5be33a27ae4ab3721c660db7392e76c1b4edc04885085` |
| Original Double QK fixture | `f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2` |
| Frozen gradient derivative file | `c742bfa2ed410a5f1aa42cb51b9d4b0e9ae5e35b3813d52661283c8027ab969b` |

QAIRT `2.48.40.260702151143`、QNN API `2.37.0`、Hexagon SDK `6.6.0.0`、Tools `19.0.07`、
HTP V81、NDK `26.2.11394342` を固定。HostはMSYS2 UCRT64 g++ 15.2.0、compiler hash
`5fa84c1460fd741371f730788ba3f994cf1d682be866fe7d3ed799e81844670a`。
compiler絶対pathはprivateに保存し、必要なPATHはprocess内だけで設定した。
QAIRT fallback、SDK変更、APK置換は行っていない。

旧Forwardのnumerical digests `d8e507289b602c7cec0a5af898e23b0f5b98b3cbeb74d4bd97b07994dbf8edee` /
`456ad468cb9cb68303261535420999408d2f1fe60443a0e78886473c19c432ae` は保存した。
新Backwardの認定証拠には転用していない。

## B. Exact Production mathematics / ownership

正本は [generalized builder](../app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc)。
head prefixは `prefix + "head_" + zeroPadded(head) + '_'`。
各headの実node列は次の通りで、packageは `qti.aisw`、宣言dtypeはFLOAT_32。

| Source suffix | Operator / inputs / output |
|---|---|
| `dvalue` | MatMul(probabilities, dContext), transpose0=true / transpose1=false → dValue |
| `softmax_product` | ElementWiseMultiply(dProbabilities, probabilities) → product |
| `softmax_dot` | ReduceSum(product), last axis, keep_dims=true → row dot |
| `softmax_centered` | ElementWiseSubtract(dProbabilities, row dot) → centered |
| `dscores` | ElementWiseMultiply(probabilities, centered) → dScores |

ProductionはP/dP `[32,32]`、dot `[32,1]`、axis=1。診断graphは既存Forward ABIに合わせ、
単位leading axesを付けた `[1,1,32,32]` / `[1,1,32,1]`、axis=3で同じ行方向演算を行う。
broadcast、row-major、transpose、shape推論と宣言outputをHost validatorで確認した。
Softmax Backwardを単一Custom opへ融合していない。

`dot[r]=Σ P[r,j]*dP[r,j]`、`dScores=P*(dP-dot)`。
この `dScores` は **dMaskedLogits** で、旧Host sensitivityのdScoresと同じ境界。
`dRawScores=scale*dMaskedLogits` とは区別する。scale bits `0x3e3504f3` はForwardに一度適用し、
今回のBackward consumerには追加適用していない。dValueは `P^T*dContext`。

標準QNN ReduceSum/MatMulの内部precision、積の丸め、accumulation orderは、このsourceからは
規定できない。backend-defaultのgraph precision設定を維持した。FLOAT_32宣言から
HTP内部FP32を断定しない。serial FP32 oracleとの差をbitwise合否に使用していない。

[新graph](../host_tests/fp32_custom/saved_p_backward_graph.h) のHost構築監査は24 graphでPASS。
新NATIVE構成では単一の既存ForwardSoftmax nodeがhead別Pを一度生成し、その同じQNN IDを
product、final product、dValueが参照する。head0/1 ID分離、NoTapのNATIVE P、
Pがgraph inputでないこと、全upstream NoReceipt化も検査した。
**これはmock APIでの構築証明であり、DSP上のfan-out/lifetime/immutability認定ではない。**

## C. Host references / fixtures

復旧済み `saved_p_backward_reference_*` を維持し、self-testを再実行してPASS。
非対称permutation PのdValue転置、one-hot Pのゼロ微分、一様Pの解析解も独立にPASSした。
最終Host suiteでは3独立processによるgradient exportがすべて同一hashとなった。
元のdP/dContextが既にFP32 wordであることも検査し、rescaleや新しい丸めによる置換を拒否する。

- A: ideal double Pによる独立oracle。新しいNATIVE total gradient評価は未実施。
- B: 分離product → serial FP32 reduction → subtraction → product、およびserial FP32 MatMul。
  node境界を表す診断参照で、標準HTP内部の厳密word oracleではない。
- C: actual Device Pで条件付けたHost参照。freshな既存Forward P回帰で評価したが、新DSP Backwardの証明ではない。
- D: 今回実測したHTP consumer出力。**Standaloneのfrozen external Pに限定**される。

新Standalone local oracleは、実入力とword一致するfrozen P/dP/dContextをdoubleで計算したもの。
PC側Pythonで全出力を再計算し、C++参照、raw output FNV、fixture/head/gradient identities、
全API success、execute件数、teardown、全coverageを独立照合した。
raw binaryはdevice SHA256とpull後SHA256を照合し、lossless展開後のSHA256も検査した。

各consumer・各observer条件でCore **44×2×25=2,200 pairs**、Holdout **16×2×8=256 pairs**。
4条件合計9,824 pairs/consumer、両consumer合計19,648。Coreとholdoutを分けて集計した。
Gradient9/12縮小、fixture専用補正、閾値変更、case除外はない。
同じPの反復を独立Forward試行として数えていない。実loss-derived trajectoryとの対応は未認定。

## D. Standalone DSP results

| Consumer | Core maxAbs | Holdout maxAbs | Gate | Accuracy FAIL pairs (4条件合計) |
|---|---:|---:|---:|---:|
| Softmax Backward / dMaskedLogits | 0.3044056503066239 | 0.02663227731332496 | 0.002 | Core 2,606 / Holdout 416 |
| dValue | 0.035564650305797585 | 0.00388396910312494 | 0.01 | Core 328 / Holdout 0 |

新worstはSoftmaxが **Forward19/H1/G9、要素495 (row15,col15)**、
dValueが **Forward23/H1/G9、要素345 (row10,col25)**。
最大serial FP32 oracle差はそれぞれ0.30438536405563354、0.03556251525878906。

各consumerで4 graph / 4,912 graphExecute。最終processのexit=4は実測した新consumerのFAILであり、
旧negative-control用expected exit4には分類していない。
全QNN API成功とは別に、finite、全面書き込み、poison消去、canary、入力不変、teardownをPASSした。
選択backendはHTPで、CPU arithmetic fallbackを使用していない。Host/ARM上の参照計算・
graph制御・evidence圧縮は別処理であり、CPUを使わない学習とは主張しない。

## E. Failure attribution

SoftmaxのForward4/H1/G9では、最初の標準 `P*dP` の時点で受信入力double参照との差が
0.09624734689077741。全入力P/gradient wordsは凍結fixtureと一致していた。
この例のproductは `half(half(P)*half(dP))` の診断モデルとword一致した。
単一例のモデル一致だけで内部FP16を断定しない。

Tap条件のreceived-input local error最大値は次の通り。

| Boundary | maxAbs | Worst |
|---|---:|---|
| product vs received P,dP | 0.4091988129730453 | F16/H0/G9, index98 |
| reduction vs received product | 0.21875 | F10/H1/G9, row6 |
| subtraction vs received dP,dot | 0.74847412109375 | F14/H1/G9, index374 |
| final product vs received P,centered | 0.15047404170036316 | F4/H1/G9, index198 |

各最大値は異なる座標・caseの値であり、足し算して実total errorとは扱わない。
dValueは正しいtranspose/layoutとfrozen received operandsでの標準MatMul出力がgate未達。
内部product precisionとaccumulation orderの切り分けはUNRESOLVEDで、新Customへ置き換えていない。

## F. Native single / dual / lifetime

両Standaloneが未達なので、新device probeはNATIVE実行を許可していない。
single P→Softmax、single P→dValue、同一Pからのdual fan-outはすべてNOT_EXECUTED。
Host構築の同一ID証明をDSP認定へ転用していない。

Standaloneで確認したP入力不変はAPP_WRITE外部入力についてのもの。
Forward-generated NATIVE Pの生存期間、consumer後の内容不変、Host再注入なしの実機証明は未認定。
Host ideal Pを再注入したgraphをNATIVE成功と呼んでいない。

## G. Cumulative precision / old Forward contract

今回の数値は `D(P_frozen)-DoubleBackward(P_frozen)` というconsumer-local error。
新graph内でのactual Forward P→DSP backwardが未実行のため、要求された
P-only propagation、consumer-added error、actual NATIVE total errorの同時分解は未完了。
別graphの最大値を足してNATIVE total errorを推定していない。

freshな既存Forward probeでは44/16全fixtureと4 observer条件を評価し、旧合同契約を維持した。

| Existing Forward contract | Fresh value | Gate |
|---|---:|---:|
| P maxAbs | 4.236482907815642e-7 | 1e-6 |
| P row-sum error (Core / Holdout) | 2.2724270820617676e-7 / 2.682209014892578e-7 | 2e-6 |
| P-induced dScores bound | 2.459099714532658e-4 | 2.5e-4 |
| P-induced dValue bound | 2.7855243723208197e-5 | 0.001 |
| Host-conditioned dScores | 1.0078856647055545e-4 | 0.002 |
| Host-conditioned dValue | 1.3850631340517339e-5 | 0.01 |

後二者はHost-conditioned Cであり、新DSP consumer出力ではない。
P/row predicatesはForward/head単位、gradient predicatesはForward/head/gradient単位で計数した。
旧P worst F16/H0、旧bound worst F21/H1/G9、指定された旧C0 critical casesも全Cartesianに含む。
source/legacy referenceを分けたfresh評価でrequired failed predicates=0。

## H. Observer independence / process evidence

StandaloneのObserved+Tap / Observed+NoTap / NoReceipt+Tap / NoReceipt+NoTapを全件評価した。
標準primitiveには受信word receiptがなく、Tapのintermediateと同一graph入力ID・実出力から評価した。
不可視NATIVE Pを観測したとは主張しない。

Softmaxはobserver差2,036 pairs、10,412 words（うち非ゼロ数値差10,408、signed-zero差4）。
最大observer差は0.15625、F2/H0/G9/index32のTap対NoTap。
dValueはobserver word差0だが、独立accuracy自体がFAIL。
new NATIVE observer-free認定は行っていない。

最終Standaloneのcanonical numerical digestは公開summaryに保存する。
timing、counter、timestamp、process/device IDsを含めず、raw output hashを独立照合した。
Softmaxの2完全processは同じnumerical digestとなった。この再実行は同一binaryの失敗再現確認であり、
要求された全NATIVE構成の3独立process cohortではない。
正式candidateが成立しなかったため、そのcohortは0。

## I. Performance / device safety

accuracy成立後に測定する事前契約に従い、Forward-only / single / dualの同一final bundleによる
31-sample Backward追加cost測定はNOT_EXECUTED、公開値はnull。
旧Forward診断wall約17.8msにはLN local Backward controlが含まれ、Backward overheadへ流用しない。
19層×8microbatchの推定も行っていない。

各device processのfresh preflightで同一物理端末、awake、known top-resumed、active trainingなし、
競合QNNなし、storage、固定SDK/compiler、installed APK、package/fixture/protected hashesを確認した。
BACKGROUND_CORRECTNESSを使用し、foreground takeover=0。
通常のbattery/temperature/外部給電を停止条件にせず、研究monitoring policyを使用。
最終Standaloneではthermal status0、battery health2。40℃の通常温度も記録して継続した。
critical/emergency保護、状態不明時のfail-closedを維持した。

pilotではstaging条件式の誤りを修正後、ASCII出力processをTOP_RESUMED_UNKNOWNで
owned PIDだけ中断した。別の完全processでraw readbackの引数解釈と低速ADB転送を修正した。
不完全/終了状態不明の試行はprivateに保存し、最終公開集計には含めていない。
修正済みrunnerの最終processは実exit、全coverage、raw SHAを取得した。
C0 Host試験の初回fixture CWD不足もinfrastructureとして保存し、正しいprivate CWDで再実行した。
caller timeoutを数値FAILへ分類せず、無関係なADB server/processは停止していない。
自動wake/unlock、force-stop、root、APK置換、power/clock policy変更は実施していない。

## J. Fresh verification / negative controls

| Verification | Result |
|---|---|
| Fast | PASS（最終code/documentation closeoutも検証） |
| Host | PASS、634.7秒 |
| Qnn | PASS、861.8秒。固定SDK check / QNN build / APK audit完了 |
| Saved-P Host references / independent analytic tests / 24 graph shapes | PASS |
| Frozen gradient word export, 3 Host processes | PASS |
| S2 LN1 saved state / local backward | Fresh PASS、2,640 numerical records |
| Mixed double Q/K + existing HVX V | Fresh PASS、22,632 records、68 cases / 32 variants |
| Double QK + Custom Scale/Mask/Softmax + Forward P | Fresh PASS、14,356 records、joint predicates=0 FAIL |
| Existing HVX FFN2 / HVX QKV | Fresh PASS、accuracy/word/guard failures=0 |
| Existing QK/Backward Scale controls | Fresh PASS（oneQkScaleA/B） |
| Existing LN2 / Custom Softmax backward paths | Existing regression graph内でFresh PASS。今回の標準Softmax分解とは別の演算path |
| Device smoke | Fresh PASS、HTP / API2.37 / Build ID一致 / fallback=false |
| Muon W8 | Fresh PASS、114/114、maxAbs=1.490116119e-8、relativeL2=4.40269854e-8、cosine=1 |

旧negative controlsを新consumer結果と分離した。

- C0 Host P: **9 FAIL**をfresh再現（expected exit4）。
- Old Device cumulative P: **36 FAIL**をfresh再現、local/upstream FAIL=0（expected exit4）。
- Standard Scale: 88 records中 **1 FAIL**、maxAbs=0.0102382000946、旧値と一致。
- Standard Mask: **176 required predicate FAIL**、Custom Maskの契約はPASS。
- Standard Softmax: **224 local FAIL**、nonfinite=0（expected exit4）。
- 35/40-node Tap: `REPRODUCED / CAUSE UNRESOLVED` を保持。今回の再現試験対象には拡張していない。
- 6031: `UNRESOLVED / DORMANT / WATCH` を保持。

旧negativeのexpected exit4判定を新Standalone consumerに適用していない。
Fast/Host/Qnn/既存回帰PASSは、新Saved-P accuracy FAILや未実施NATIVE gateを上書きしない。

## K. Decision / artifacts / reproduction

次のfrontierは、標準multiply / reduction / subtraction / MatMulのreceived-input precisionと
backend-default実装の限定調査。既存Production precision optionsを比較するなら、旧FAILを保存し、
別candidateの契約を先に固定してから行う。新Custom Backward、Forward数学変更、gate緩和は実施していない。
PV/Gate/Outputによる実dContext/dP producer、dQ/dK、Head Scatter、projection gradients、
LN1 dY aggregationは引き続き独立した未認定領域。FULL_ATTENTION_BACKWARD / FULL1 / FULL19 /
TRAINING_READYは認定しない。
既存local controlのPASSを、新P・実upstreamと接続したcumulative認定へ昇格させていない。

公開aggregateは [allow-list summary](results/g1-native-saved-p-backward-dsp-summary.json)。
private rootは `build/reports/g1-native-saved-p-backward-dsp/`。
[exporter](../scripts/export_public_saved_p_backward.py) はraw/fixture/output hash、全coverage/API、
fresh回帰、保護source/binary/stateを直接検査し、10件の拒否self-testをPASSした。
missing evidence、false DSP PASS、wrong binary、incomplete coverage、数値FAIL隠蔽、raw field、
observer downgrade、欠落数値、絶対path、実終了状態の欠落を拒否する。
raw tensors、checkpoint、corpus、device/endpoint/run IDs、APK hash、private logs、絶対local path、
SDK binariesは公開集計・commitへ含めていない。

再検証の入口は `test_saved_p_backward_host.ps1`、固定引数付き `build_saved_p_backward_probe.ps1`、
`run_saved_p_backward_probe.ps1 -Candidate final -Mode standalone_softmax|standalone_dvalue`。
毎回新しいprivate ReportRoot、固定QAIRT/Hexagon/NDK引数、完了したQnn logが必要。
既存fixture/evidenceを上書きしない。SDK/compilerの実pathはprivate ledgerを使う。
公開再生成は `python scripts/export_public_saved_p_backward.py --report-root <private-root> --output <new-output>`。

40k checkpoint、100k baseline、200k UNSTABLE、Step41,647/42k evidence、dataset/tokenizer/cache/order、
previous refs / provenance recovery / audit cloneを保存した。protected checkpoint decode/write、
LR/optimizer/ABI/codec変更、training、final testは行っていない。local commitsのみで、push/PR/mergeなし。

## Final questions 1–35

| # | Answer |
|---:|---|
| 1 | 正しい研究cloneを使用。実path照合はprivateに保存。 |
| 2 | be4271efd1b8fb71f0c66f3ae7c35f04571aa78e起点。 |
| 3 | e05f353のForward source/binaryをhash照合して保持。 |
| 4 | Host reference self-test PASS。 |
| 5 | dValue MatMul、softmax_product → ReduceSum → centered subtraction → dscores product。 |
| 6 | 旧dScores = dMaskedLogitsと確定。 |
| 7 | Backward consumerへのscale二重適用なし。 |
| 8 | dValue transpose0=true / transpose1=false、row-major、head分離を確認。 |
| 9 | Standalone SoftmaxはDSP accuracy FAIL。 |
| 10 | Standalone dValueはDSP accuracy FAIL。 |
| 11 | Standalone dMaskedLogits maxAbs=0.3044056503066239。NATIVE totalは未測定。 |
| 12 | Standalone dValue maxAbs=0.035564650305797585。NATIVE totalは未測定。 |
| 13 | Standalone consumer-local errorは独立測定。新NATIVE P-only / consumer / totalの同時分解は未完了。 |
| 14 | Single P→Softmax NATIVEは未実行。 |
| 15 | Single P→dValue NATIVEは未実行。 |
| 16 | 同一P ID共有はHost構築のみ確認。DSP dualは未実行。 |
| 17 | DSP NATIVE P lifetimeは未認定。 |
| 18 | Standalone入力Pは不変。Forward NATIVE Pのconsumer後不変は未認定。 |
| 19 | Host構築は再注入なし。実DSP NATIVE接続としての証明は未実施。 |
| 20 | NATIVE observer-free両gradientは未認定。Standaloneもaccuracy未達。 |
| 21 | Core2,200を各consumer・4条件で全件評価。NATIVE Cartesianは0。 |
| 22 | Holdout256も各consumer・4条件で全件評価。NATIVEは0。 |
| 23 | 新Standalone worst dMaskedLogits=F19/H1/G9/index495。 |
| 24 | 新Standalone worst dValue=F23/H1/G9/index345。 |
| 25 | 正式な全構成3-process cohortは未実施。digest一致をaccuracy代用にしていない。 |
| 26 | 既存Forward Pの旧合同契約はfresh PASS。 |
| 27 | 旧C0の9 FAILをfresh保持。 |
| 28 | Fresh HVX FFN2/QKV PASS。 |
| 29 | Fresh Muon W8 114/114 PASS。 |
| 30 | Single/dual Backward追加costは未測定。accuracy gateで停止。 |
| 31 | 研究用battery/thermal/power monitoringを遵守。通常40℃で継続。 |
| 32 | critical保護、unknown状態のfail-closed、owned-process停止を維持。 |
| 33 | 標準Backward primitiveのlocal precision / reduction / MatMul frontier。 |
| 34 | 実PV/Gate/Output upstream、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregationが残る。 |
| 35 | 40k→60k / 200kはNO-GOを維持。 |
