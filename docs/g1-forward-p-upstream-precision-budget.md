# G1 Forward P upstream precision budget

## A. Initial evidence and decision

**HostではFP32 tensor境界を維持して旧P合同契約を満たせる。今回の候補集合ではC7（LN1・Q/K Projection・QKᵀの内部double演算）が、追加の高精度保存tensorを必要としない最小のstage集合だった。QKᵀ accumulatorだけの改善では不十分。DSP prototypeは実装していない。**

これは全アルゴリズムについての最小性の証明でも、DSP上のP precision closureでもない。既存ordered FP32、前研究の累積P、full1-layer/full19-layerの認定状態は変えない。40k→60k/200kはNO-GO。

起点は`3c8190a79803dbd73a5ced370710fbc5ee0c6218`、開始時clean、未push local history 117 commits。専用branchは`investigation/g1-forward-p-upstream-precision-budget`。以前のbranch refs・protected artifacts・既存packageを照合して保持した。

旧9 FAILの再生成log SHA256は前研究と同一：`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7`。旧合同契約testの実exitは4、attribution testは0。既存R0–R5とsigned perturbation／componentwise boundの独立検算を再実行した。

|旧path|P maxAbs|
|---|---:|
|R0 ideal double|0|
|R1 correct FP32 masked input→Host Softmax|2.507290394e-7|
|R2 FP32 through QKᵀ→ideal downstream|9.215255858e-7|
|R3 FP32 through Scale→ideal Mask/Softmax|9.791111950e-7|
|R4 FP32 masked input→double Softmax|9.791111950e-7|
|R5 full legacy Host model|1.007506242e-6|

最大値の差は独立したstage寄与の加法分解ではない。R2のsigned P-only dScores自体が予算を超える既存結果も保持する。上界だけの精緻化、局所Softmax改善、閾値緩和で今回の問題を消してはいない。

## B. Precision boundaries and independent references

数学の正本は`app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc`。T32/D64/H2、Q/K ProjectionはNN `[32,64]×[64,64]`、headはrow-major `[32,32]`、rawはNT `[32,32]×[32,32]ᵀ`。HeadSplitはcopyで新しい算術誤差を作らない。

Scaleはsourceの`1.0f / std::sqrt(float(dimension / numHeads))`、`0x3e3504f3`をそのままdoubleへ昇格する。Maskはallowed `+0`、future `-10000`、`column<=row`、528/496 positions per head。Softmaxはproduction rank2 axis1／diagnostic rank4 axis3、beta1。定数・axis・epsilon・variance定義・parameter・下流gateを変更していない。

|境界|現行F|C7|高精度保持案|
|---|---|---|---|
|LN1 reduction/normalization/affine|anchor-relative FP32/Kahan/hypot|独立two-pass double mean/variance、double normalization/affine|B1/B3/B4はYをdoubleで保持|
|LN1 Y|FP32|FP32へround|保持案のみdouble|
|Q/K K64 product/sum|productごとFP32、q昇順FP32 sum|double product、q昇順double sum|B2/B3/B4はQ/K出力をdoubleで保持|
|Projection→HeadSplit|FP32、lossless copy|同じFP32境界|保持案はHost double copy|
|QK K32 product/sum|productごとFP32、q昇順FP32 sum|double product、q昇順double sum|B4のみrawをdoubleでScaleへ渡す|
|raw output|FP32|FP32|B4のみ保持|
|Scale/Mask output|FP32|FP32|全候補ともScale以後はFP32|
|P output|FP32|FP32|全候補ともFP32|

Dは独立double、Fは旧ordered FP32、Mは明示したstageだけを変更、Sは実際に受信するoperandに対するdouble local reference。Custom arithmetic bodyはD/M/Sに含めない。`one::mm`、独立LN reference、独立Softmax referenceを使用し、新MatMulも別のloopで記述した。Host flagsは`-O2 -ffp-contract=off -fno-fast-math`。新候補は旧Scalar/HVX word contractと異なる数学として扱う。

**epsilonの参照境界も分離した。** 旧LN double referenceはdouble literal `1e-5`、実FP32 kernelのepsilonは`1e-5f`。旧referenceを変更せず再現した上で、sourceのFP32 epsilonをdoubleに昇格するDも追加し、全候補を両方に対して評価した。両ideal Pの最大差は`6.675058839e-10`。この差を旧9 FAILの修正として利用していない。表のlegacy値とsource値は別recordで、C7は両方PASSする。

独立して書いたlong double ForwardとのDのP差は120 headsで最大`2.275957200e-15`。新Fモデルは独立旧Fに対してP 122,880 words一致（mismatch0、signed-zeroを含むword比較）。同じFP32 Q/K operandへのdouble QKと`round32(one::mm(...))`も一致した。これらはHost referenceの検査でありDSP精度認定ではない。

### 表現floor

|source Dに対する孤立境界実験、44×2×25|P maxAbs|dScores bound|
|---|---:|---:|
|ideal Pを最近接FP32へ丸めるだけ|2.978684122e-8|2.114026160e-5|
|ideal LN YだけFP32保存→ideal downstream|1.190391255e-7|7.476333469e-5|
|ideal Q/KだけFP32保存→ideal downstream|6.782797246e-8|5.249559330e-5|
|ideal rawだけFP32保存→ideal downstream|2.329622589e-7|1.636649203e-4|

最近接FP32 Pのみが、対象referenceに対するelementwise出力表現の下限を与える。他の行は孤立した丸めのcounterfactualであり、累積誤差の不可避な下限ではない。実際にC7がPASSするため、現行Fの`1.0075e-6`を「FP32 dtypeそのものの限界」とは呼べない。

## C. Controlled candidate matrix

事前計画`host_tests/fp32_custom/upstream_precision_plan.json`をcommit `8ed572f`で評価前に固定した。C0–C7は段出力をすべてFP32へroundする。C1/C2/C3は順にLNのみ／Q/Kのみ／QKのみ、C4/C5/C6はLN+Q/K／LN+QK／Q/K+QK、C7は3段。Vの演算は候補から除外し、weightの値は元のFP32のまま。

以下は旧D・旧R5 Softmaxを使うimmutable-contract比較。全候補について44 Forward×2 heads×25 gradients、P/row/causal/finite/nonnegative、signed perturbation、保守的上界、実Host dScores/dValueを評価した。詳細な全metricsは公開summaryにある。

|候補|P maxAbs|dScores bound|failed predicates|failed Forward/head pairs|
|---|---:|---:|---:|---:|
|C0|1.00750624e-6|4.64227603e-4|9|6|
|C1|1.38146330e-6|6.07756453e-4|6|4|
|C2|1.21612250e-6|3.31945939e-4|5|4|
|C3|8.57075495e-7|3.64826770e-4|2|2|
|C4|7.39285341e-7|3.27630822e-4|5|5|
|C5|5.30669084e-7|3.94295929e-4|2|2|
|C6|5.59052272e-7|2.82402922e-4|2|2|
|**C7**|**2.88646297e-7**|**1.44921899e-4**|**0**|**0**|

単独改善で悪化する最大値もそのまま保存した。誤差の符号・logit競合・丸め境界の位置が変わるため、stageの局所精度向上から累積Pの単調改善は導けない。C3/C5/C6のElement PASSをJoint PASSと誤認しない。

|追加の一般的算術候補|定義|P maxAbs|dScores bound|FAIL数|
|---|---|---:|---:|---:|
|LN_reduce|anchor-relative reductionをdouble、center/rmsをFP32へ戻し既存norm/affine|1.29205633e-6|5.84576180e-4|9|
|LN_norm|既存FP32 reduction、sqrt/hypot/reciprocalとh積をdouble、h出力FP32|9.64230783e-7|4.59474368e-4|6|
|LN_affine|既存FP32 h、gamma*h+betaだけdouble、YはFP32|8.32778427e-7|4.22627609e-4|6|
|QK_sum64|FP32-rounded productをdouble sum|8.57075495e-7|3.64826749e-4|2|
|QK_Kahan|FP32 product＋FP32 Kahan sum|8.83111601e-7|4.30619972e-4|3|
|QK_Neumaier|FP32 product＋FP32 Neumaier sum|8.57075495e-7|3.64826749e-4|2|
|LN_QK_Kahan|full double LN＋QK Kahan、Q/Kは旧F|6.66207559e-7|3.94295929e-4|2|
|Q_sum64|Q/KのFP32-rounded productをdouble sum|1.21612250e-6|3.01701690e-4|3|
|Q_Kahan|Q/KのFP32 product＋Kahan|1.00750624e-6|3.89361765e-4|5|
|Q_Neumaier|Q/KのFP32 product＋Neumaier|1.21612250e-6|3.01701690e-4|3|
|Q_QK_Kahan|両MatMulをKahan、LNは旧F|6.08431922e-7|2.81675492e-4|2|

全候補のrow/causal/finite/nonnegativeと実Host gradient gatesはPASS。表のFAILはP要素またはdScores配分予算。補償和はproduct roundingを回復せず、Kahanという名称だけでdouble相当とはならない。

|cross-boundary案|C7からの変更|追加保存bytes/layer/MB|P maxAbs|dScores bound|合同|
|---|---|---:|---:|---:|---|
|B1|LN Y保持|8,192|3.58530451e-7|2.44689439e-4|PASS|
|B2|Q/K保持|16,384|2.88646297e-7|1.12035460e-4|PASS|
|B3|YとQ/K保持|24,576|3.69057951e-7|2.62258390e-4|FAIL 1|
|B4|Y/Q/K/rawをScaleまで保持|32,768|2.50729039e-7|1.11978552e-4|PASS|
|**B5として選ぶC7**|**FP32境界維持**|**0**|**2.88646297e-7**|**1.44921899e-4**|**PASS**|

保持bytesは既存FP32状態との差。QNN FLOAT32 tensorにdoubleが入るという意味ではない。B群はHostのみであり、実装にはFP64 dtypeのbackend契約、high/low pair、sidecar、または内部保持するfused kernelの新設計が必要。

C7はC0よりP maxAbsを約71%、dScores boundを約69%減らす。全候補を網羅した最小性ではなく、今回のstage subsetと列挙した補償和候補の中での最小領域である。例えばfull double LNと複数の安価なMatMul補償方式を組み合わせる全組合せまでは探索していない。より安い新候補が存在しないとは主張しない。

## D. Old failures, signed sensitivity and new residuals

旧9 predicatesは5 Forward cases、6 Forward/head pairs、head indices0/1、gradient9/12に集中する。P要素FAILは21/H1で、8つのdScores bound FAILと重複して数える。

|Forward/head/gradient|旧失敗predicate|旧value|
|---|---|---:|
|16/1/9|dScores bound|4.6422760347e-4|
|17/1/9|dScores bound|2.5089241020e-4|
|20/1/9|dScores bound|2.6736259971e-4|
|21/1/9|dScores bound|4.0506914619e-4|
|21/1/12|dScores bound|3.9143508628e-4|
|21/1/—|P maxAbs|1.0075062425e-6|
|22/0/9|dScores bound|2.5959560410e-4|
|22/0/12|dScores bound|3.0420190833e-4|
|22/1/9|dScores bound|3.2094733713e-4|

|C7 focus、source D|P maxAbs|g9 dScores bound|g12 dScores bound|
|---|---:|---:|---:|
|16/H1|1.27612991e-7|6.22739562e-5|2.46322291e-5|
|17/H1|1.94262486e-7|8.05148929e-5|3.76638432e-5|
|20/H1|1.89757694e-7|1.09429161e-4|6.25698590e-5|
|21/H1|2.14388683e-7|1.44922060e-4|7.81472055e-5|
|22/H0|2.65291808e-7|8.84499787e-5|5.20392610e-5|
|22/H1|1.48424939e-7|7.05811327e-5|3.63694831e-5|

旧21/H1のcritical coordinateはrow1/col1（mask allowed）。C7のそのcaseの最大誤差もcoordinate33で、未来maskの変更による改善ではない。他のcaseの最大座標が移動することも保存している。全88 head casesを測り、旧FAILだけの補正はない。

旧式を不変に用いる：`a=ΣdP·P, t=ΣdP·e`、`Δds_i=e_i(dP_i-a)-(P_i+e_i)t`。absolute componentwise boundと実signed differenceを別々に独立再計算し、dValueについても`eᵀdContext`を検算する。全candidateでboundがsigned差を下回らないことを検査した。

|C7、legacy D、full Cartesian|実測|旧gate|
|---|---:|---:|
|P maxAbs|2.886462966e-7|1e-6|
|row-sum error|2.682581908e-7|2e-6|
|dScores bound|1.449218994e-4|2.5e-4|
|dValue bound|1.859756632e-5|1e-3|
|signed P-only dScores|1.067250688e-4|診断値、上界とは別|
|signed P-only dValue|1.202859622e-5|診断値|
|actual Host FP32 dScores|1.376270728e-4|.002|
|actual Host FP32 dValue|1.564496581e-5|.01|

source Dでの最大boundは`1.449220597e-4`で同じ判定。actual gradientはHostで候補Pを条件として計算したもの。実DSP backward、loss-derived trajectory、roundingを含む離散写像の厳密な自動微分を証明したものではない。

## E. Holdout, reference models and validation

独立holdoutは事前固定16 seeds×8 pattern（ordinary、low variance、large mean、alternating cancellation、small/competing logits、small/large centered X）で、gamma/beta/weights/dYも新規生成する。8つのgradient seedsからdP/dContextを生成し、16×2×8=256 Cartesian pairsを全候補で評価した。個別の失敗値を見て値・seed・閾値を調整していない。

holdout binary SHA256：`12be9d57a41bc60dfebc1dab3ea9f796cb63485cf1335cfc847c7fbe5f941c5e`。これは新しいcheckpoint-derived実モデルfixtureではない。

|source D、holdout|C0|C7|
|---|---:|---:|
|P maxAbs|2.00188483e-7|6.91324934e-8|
|dScores bound|1.58312180e-5|5.50033164e-6|
|actual dScores|8.07600060e-6|3.48333052e-6|
|合同FAIL|0|0|

全holdout候補はgate内。C7はこの独立集合の最大値も改善したが、C0もPASSする穏やかな集合であり、独立実モデルに対する強いrobustness認定には不足する。境界保持B3が既存集合でFAILする事実も消さない。

metadataで対応するsame-frozen-fixtureは24×2=48 pairs。source DでC0のboundは`2.534208349e-5`だがP要素FAILが1件残る。C7のboundは`1.566664066e-5`、合同PASS。Cartesianとsame-frozenの結論を別記し、actual training trajectory対応は`ACTUAL_TRAINING_MATCHED_UNAVAILABLE`。

**下流のFP32 shift境界も別モデルで確認した。** 旧R5はshiftをdoubleで計算してexp出力からFP32へroundする。現在のCustom Softmax用独立Host referenceはshiftもFP32へroundしてHost libmでexpを求める。両モデルの全23候補を両D・両集合で評価した。後者のC0は8 FAIL、旧R5のC0は9 FAILであり、旧結果を上書きしない。C7は両モデルでPASS。DSP exp wordsとの一致や、新DSP candidateの実行結果をこのHostモデルから推定しない。

### 検証結果

|工程|今回|
|---|---|
|旧targeted Host 13 legacy＋11 current|23 exit0、旧合同契約のみexpected exit4|
|R0–R5／旧9 FAIL|再現、元log SHA一致|
|23候補×2 reference×2集合|完走、各Cartesian pairでactual/bound検算|
|matrix独立Host process|同一exeで3回、全数値record SHA一致|
|FP32-shift別モデル|1 process、同じexe、結果は別log|
|独立long double／旧F words／saved-state test|PASS|
|representation floor test|8 paths/sets、PASS、別exe/1 process|
|公開exporter self-test|17 summary破損を拒否、allocationとactual gateの分離を検査|
|Fast|初回Gradle daemon接続失敗、同条件再試行65.5s PASS|
|Host profile|759.0s PASS|
|QAIRT read-only inventory|exit3、optional不足advisory、core/build ID一致|
|既存APKのread-only audit|SUCCESS、ABI/hash/path/2.47検査|
|新Qnn build/Device/HVX/W8|NOT RUN、package/graph/kernelを変更していない|

Fast初回失敗はprivate evidenceに保持し、数値FAILとして数えない。既存Scalar MatMul/Multiply、LN、HeadSplit、Softmax、QK/ScaleのHost regressionsを再実行した。既存HVX/ABI sourceと最終実機binaryもhash不変。過去W8 114/114は過去packageの結果としてのみ保持し、今回の新候補のPASSには転用しない。

## F. Saved state and Hexagon feasibility

### LN1

C7のFP32 saved xhat/inverseを使う独立double analytical replayでは、44 casesのdX誤差最大`6.527037897e-6`、dGamma `4.709736771e-6`、dBeta0。holdoutはdX `1.985506060e-5`、dGamma `2.937153498e-7`、いずれも既存.02内。

これは保存表現の誤差検査であり、新LN DSP backward実行ではない。C7のYは未丸めdouble hからaffine後にroundするため、FP32 saved hから旧FP32 affineを再生したYとは最大`2.384185791e-7`異なる。mean/variance/epsilonの数学は同じでも算術契約は異なる。将来のLN OpにはY/xhat/inverse/backwardを一体として事前登録した新契約が必要。

### Q/K

C7はForward QKのoperandもsaved Q/Kも同じFP32 words。frozen 2,200 pairsの独立QK analytical gradientとの局所比較で、保存表現由来のdQ/dK誤差0、既存FP32 MatMul/Scaleの算術差最大`3.697200195e-5≤.01`。

B2/B4はForwardの高精度Q/KをBackwardでFP32保存値に戻すと、表現だけでそれぞれ`9.001189845e-6`／`9.686232481e-6`の局所gradient差が生じる。この範囲ではgate内でも「同じ値を使ったForward/Backward」とは呼べない。C7の保存表現が小さい利点になる。

Q/K/Pは現sourceのBackwardまで生存させる。production H>1のP APP_READ ABI、Q/K native lifetime、Backward dQueryRaw/dKeyRawとscale、dValueのP使用を監査した。今回は実Forward→Backwardのtensor連結やlifetimeを変更していない。新LN Yを接続すればV/Gate branchにも影響するため、V算術を変更しない場合でも共通Yからの回帰が必要。現在のP-only Host結果はそのbranch全体を認定しない。

### 固定toolchainでのcompile-only監査

QAIRT `2.48.40.260702151143`、QNN API `2.37.0`、V81、Hexagon SDK `6.6.0.0`、tools `19.0.07`、NDK `26.2.11394342`を維持。`upstream_precision_codegen.cpp`をV81にcompileし、assembly・object disassembly・LLVM IR・undefined symbolsをprivateに保存した。

* double addは`dfadd`、double productは`dfmpyfix/dfmpyll/dfmpylh/dfmpyhh`、入力は`convert_sf2df`へloweringされた。
* 単独double productは2個のfixと4個の部分積命令、5 packet groupsで構成された。単一のFP64 vector MACへ置き換わるという証拠ではない。
* double divisionは`__hexagon_divdf3`、LN inverse probeは外部`sqrt`も参照する。scalar add/mulまで全てsoftware library呼出しになるという仮説はこのcompiler出力では支持されない。
* IRにFMA/fast arithmeticはない。ただしhelper/libmの実際のruntime linkage、subnormal/例外/丸め、完全kernelのstack/spill、DSP実測latencyは未確認。
* 小さなdot harnessでは8-byte frameが出る。完全Opのstack/scratch費用として流用しない。
* generic QNNにはFLOAT64 enumが存在するが、固定QHPI element-type enumにはFLOAT64がない。enumの有無からHTP native FLOAT64 tensorサポートを認定しない。
* strict IEEE HVXの既存FP32 output-column tiling/indexingは設計知識として使えるが、そのFP32 accumulatorsを保ったままC7と同じ数学にはできない。今回FP64 HVX datapathを証明・実装していない。

compiler guideのfast-math/strict IEEE節も固定SDK内で確認した。SDK PDFやdisassemblyをcommitしていない。監査初回はnm executable名の誤りで停止し、SDKに実在する`hexagon-nm`へ修正してcompile-only監査を再実行した。SDKの変更はない。

## G. Device prototype decision

開始時の単発read-only preflightはPASS：同一物理端末、awake、known top-resumed、active runなし、storage、thermal/battery/power、既存APK/SDK/package/compiler/fixture/protected identityを確認。probe実行0、端末への変更0。その時点の安全確認を後日の実行許可として持ち越さない。

**新DSP arithmetic／registration／receipt／probe／native graphは未実装・未測定。** QK-onlyのC3/sum64/Kahan/Neumaierが旧配分予算を満たさないため、QKだけを実装しても目的を達成できない。C7には新LN saved-state算術、新Q/K projector、新QKの3領域が必要であり、一つの新geometryというbounded prototypeのentry条件は確立していない。

依頼の§19–20/§33が認めるHost feasibility/architecture評価までで停止した。これはdevice safety failureや「DSPでdoubleが不可能」という結論ではない。新候補のruntime数学・linkage・完全Backward互換性・latencyは`UNRESOLVED`。新candidateに対するstandalone/native/observer-free/P cumulativeのDSP認定は全てNOT RUN。

既存binaryのHVX/W8 PASSは保持するが、新候補へ転用しない。今回のHost CPU reference計算は研究用であり、DSP graphのCPU fallbackではない。

## H. Accuracy–cost and architecture

### Geometry上の費用

|C7の仕事、layer/MB|product terms|accumulator additions|
|---|---:|---:|
|Q/K NN、2×32×64×64|262,144|262,144|
|raw QK NT、2×32×32×32|65,536|65,536|
|MatMul合計|327,680|327,680|

double LNの素直なtwo-passモデルは追加で6,144 multiplies、10,272 add/subtract、96 divides、32 sqrt（gamma/betaを含む、保存・conversion/copyを除く）。19 layers×8 MBの論理量はMatMul 49,807,360 product terms、LN込み50,741,248 multiplies／51,368,704 add/subtract。これらは実行命令数やproduction throughputではない。

Vの既存K64 projectionには131,072 FP32 product termsが残る。現在のpaired QKV launchは3出力を持つため、新Q/Kだけを別Opにすると、Vを得るために旧QKVも実行し不要Q/Kを計算する暫定設計もあり得る。その場合の余分な262,144 FP32 termsは上表に含めていない。旧ABIを変えずVを分離・再利用する設計を先に確定すべきで、launch/node数の増加を0と断定しない。

C7の追加native/saved stateは0 bytesという**表現上の設計値**。scalar accumulatorはfloat4→double8で最低+4 bytes/output同時処理分、LN rowをdouble centered配列で持つなら512 bytes（FP32 rowより+256）。完全kernelのregister/spill/stackは未測定。LN FP32 xhat+inverseは既存8,320 bytesで、生存期間を短縮しない。

B1/B2/B3/B4の追加storageはそれぞれ8/16/24/32 KiB/layer/MB。保存Q/Kのdouble化はbackwardまで寿命が伸び得る。19×8の単純総和を同時peak memoryとは扱わない。fused islandは外部FP64 dtypeを避けても内部scratchとbranch/saved-state契約を必要とする。

|候補群|精度|費用・risk|判断|
|---|---|---|---|
|QK-only accumulator/補償和|Element PASS、bound FAIL|少ない変更でも予算未達|prototypeを先行しない|
|C7|全合同PASS|高精度3段、extra saved0、旧HVX Q/K算術とは別|優先する設計anchor|
|B1|PASS、boundの余裕約2%|Y sidecar/fusion必要|C7より優先しない|
|B2|boundは改善|Q/K storage/Backward値の不一致|現時点で不要|
|B3|FAIL|保持量が増えても予算超過|不採用|
|B4|PASS|最大storageと新scale入力契約|現時点で不要|

以前のHVX QKV約.32ms／Scalar QK約1.98msは別binary/instrumentationの参考値。C7の時間へ換算しない。新算術のlatency、receipt込み、observer-free wall、追加launch、完全stack、scheduleは**UNMEASURED**。speedupやproduction wallの単純19×8倍算はない。

### 推薦する投資先

**上流精度実装の段階設計を優先する。** C7を数学的anchorに、LN1のY/xhat/inverse/backward契約とQ/K/V launch分離を事前登録し、より安価な補償方式との組合せを新しい独立検証で比較してから、一領域ずつDSP standaloneへ進める。P gateの再設計やBackward Scatter、既存MatMulのHVX速度改善を今回の未解決精度の代替にしない。

現行FP32境界でHost PASSが存在するので、P contract緩和や高精度saved tensorのABI拡張を今すぐ選ぶ根拠はない。Host PASSは実機PASSを予告しない。新DSP候補ではruntime数学、最終binaryの3独立process、observer variants、全上流stage、HVX/W8を改めて測る必要がある。

Full1-layerにはこの移植に加え、producer-fed P、PV/Gate/output、Forward/Backwardの全saved state、scatter/projection/Gate branches、LN1 dY aggregation、FFN/residual backward、gradient materialization、8-MB accumulationが残る。

## I. Final questions

1. **旧9 FAIL再現**：同一log SHA・旧exit4で再現。
2. **floor**：現行Fの観測P errorは1.007506e-6。FP32最終出力表現floorは2.978684e-8で、前者は不可避なdtype限界ではない。
3. **LNのみ**：C1 P=1.381463e-6、bound=6.077565e-4、6 FAIL。単独変更は悪化もある。
4. **Q/Kのみ**：C2 P=1.216122e-6、bound=3.319459e-4、5 FAIL。
5. **QKのみ**：C3 P=8.570755e-7、bound=3.648268e-4、2 FAIL。
6. **LN＋Q/K**：C4 P=7.392853e-7、bound=3.276308e-4、5 FAIL。
7. **Q/K＋QK**：C6 P=5.590523e-7、bound=2.824029e-4、2 FAIL。
8. **LN＋QK**：C5 P=5.306691e-7、bound=3.942959e-4、2 FAIL。
9. **3段**：C7 P=2.886463e-7、bound=1.449219e-4、0 FAIL。
10. **FP32 stage outputsでPASS可能か**：C7が旧D/source D、両下流Hostモデル、両集合でPASS。
11. **高精度tensor保持が必要か**：今回の集合では不要。普遍的不要性の証明ではない。
12. **P element最小案**：C0–C7のstage集合ならC3が単独stage案。部分演算まで含めた仕事量ではLN_affine（2048 affine products/adds）が小さいElement-only候補。どちらも合同FAIL。
13. **dScores bound最小案**：評価したFP32-boundary候補ではC7。
14. **全合同最小案**：同じC7。より安い未評価の組合せを排除してはいない。
15. **旧FAIL外でも改善するか**：全44 casesを評価、独立holdoutの最大P/boundも改善。ただし単一case全ての単調改善は主張しない。
16. **matchedとの差**：same-frozenで感度は小さく、C0はP要素FAILが残る。C7は両集合PASS。実training matchedは不明。
17. **saved Q/K**：C7は同じFP32値を使う。局所Host gradient gate内。完全Backward/training整合性は未認定。
18. **DSP FP64**：固定compilerはscalar DF命令を生成する。helper/linkage/実runtime精度/時間を含む実用性は未確定。
19. **補償FP32代替**：試したKahan/Neumaier単独および列挙組合せは合同FAIL。将来の別組合せは未評価。
20. **storage/latency**：C7 extra saved0、最小accumulator+4 bytes/output、LN row案+256 bytes。latency未測定。
21. **HVX再利用**：layout、tiling設計、V/旧回帰を再利用できる。旧FP32 accumulatorのままC7にはならない。
22. **prototype価値**：上流3段の段階設計には価値がある。QK-only prototypeは今回の目的を達成しない。
23. **Device accuracy**：新候補は未実装・NOT RUN。初期safety PASSだけで昇格しない。
24. **次の優先**：上流精度実装の契約・段階設計。Backward Scatterやcontract緩和を先行しない。
25. **Full1-layer障害**：C7 DSP移植、saved-state/branch整合性、P→Backward native、残るForward/gradient branches/materialization/accumulation。
26. **40k→60k/200k**：NO-GO、training updates0。

## J. Provenance and reproducibility

|identity|値|
|---|---|
|branch|`investigation/g1-forward-p-upstream-precision-budget`|
|base|`3c8190a79803dbd73a5ced370710fbc5ee0c6218`|
|candidate/reference source|`084c3520053d08e0190d36f113d3da003ed6f5ba`|
|prospective plan commit|`8ed572f`|
|supplementary floor source|`120700448a7fe9b9d8b146d07daa6d7a9cdc0a55`|
|plan SHA256|`94cc55e84194a06c7023dcbf04c5289577fcd9c4efa78b53463e5a7a3bd96aae`|
|matrix Host exe SHA256|`9144787b137ef970f3286b6d4e5c39a68e9cc69ddb662f2c8538e17dcaf2c479`|
|consistency Host exe SHA256|`7f712363ecd0b49f2fbca6262697cedff929e6773eaedbf2f7162e0204b42518`|
|floor Host exe SHA256|`a5f670c51064f80003d6eb4a3dac17e91230f7e7a7f5fc07249d97e1dbcbab78`|
|Host compiler SHA256|`5fa84c1460fd741371f730788ba3f994cf1d682be866fe7d3ed799e81844670a`|
|Hexagon compiler SHA256|`039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`|
|previous device-tested source|`65148e19237b162da14042b500892ef5716e9b8d`、新候補のdevice-tested sourceなし|
|unchanged DSP SHA256|`10770ea36f8c994462377e5bce68e4d8619befc7576ddf21cb61d41fd2ad2857`|
|unchanged Host package SHA256|`4322bded8951959f4ca37840a638fc6332e7cc03b45b8dd8ab44fe3646df81e9`|
|unchanged probe SHA256|`9726ea13adce1ce4cb3684ad7ae28d8eff3b9011e8a75909a1093978097603f2`|
|frozen one-layer inputs SHA256|`388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`|
|frozen LN1 parameters SHA256|`7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`|
|frozen producer inputs SHA256|`ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`|
|Host process1/2/3 numerical-record SHA256|全て`daa9f90c6506f808c516110f64c1887ea099dae657d16a89051065ed144dc126`|
|separate FP32-shift model log SHA256|`ab965f2f5437362fb42735f7c449c322ef47e426acfab82c8d88e34a5299df6e`|

数値record digestにtime/device IDは入れない。3回は同一exeの別Host processであり、DSPの3-process cohortではない。reference/source/fixture/SDK、全process logs、API非該当、新device実行0、保護identityはprivate evidenceへ記録する。

Host reproductionは新しいprivate出力directoryを使い、`scripts/test_forward_p_attribution.ps1 -Legacy`、`scripts/test_forward_p_upstream_precision.ps1`へ固定QAIRT root/Build IDとReportRootを明示する。floor testも同じ`g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math`、QNN include、`tiny_language_model_cpu.cpp`、`qnn_graph_shape_validator.cpp`でcompileし、LN1 parameterファイルを置いたprivate directoryでone-layer fixtureを引数に実行する。compile-only監査は`scripts/audit_forward_p_precision_codegen.ps1`に固定Hexagon rootも明示する。

公開summaryは`scripts/export_public_forward_p_upstream_precision.py`のallow-listで生成する。全case/pairからsummary/gate/failure数を再計算し、旧9 FAIL digest、coverage、3-process、exe/log/holdout identity、Fast/Host completion、HostとDeviceの分類分離を確認する。raw tensor、SDK binary、checkpoint、corpus、device identity、private log、absolute local pathを公開しない。

private rootは`build/reports/g1-forward-p-upstream-precision-budget/`。最終HEAD・clean tree・report/aggregate hashes・5 protected artifact hashes・既存branch refsの不変性はcommit後の`final-closeout-private.json`で固定する（自分自身のcommit hashを文書へ埋め込む循環を避ける）。既存DSP arithmetic source/SDK/packageに変更なし、push/PR/merge/history rewriteなし。

40k checkpoint identityは`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`。100k baseline、200k UNSTABLE、forensic records、dataset/tokenizer/cache/order、optimizer/codec、private evidenceを保持。checkpoint decodeなし。35/40-node Tap incidentは`REPRODUCED / CAUSE UNRESOLVED`、6031は`UNRESOLVED / DORMANT / WATCH`。

今回の分類：`UPSTREAM_PRECISION_ATTRIBUTION_VERIFIED`、`FP32_BOUNDARY_FLOOR_CHARACTERIZED`、`MINIMAL_PRECISION_ISLAND_IDENTIFIED`、**C7 Host候補に限る**`HOST_P_JOINT_CONTRACT_PASS`。現行Fは`HOST_P_JOINT_CONTRACT_FAIL`を維持。新DSP runtime/cumulative/observer-free/trainingは`UNRESOLVED / NOT RUN`。
