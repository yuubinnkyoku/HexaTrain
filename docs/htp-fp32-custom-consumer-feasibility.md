# HTP FP32 custom consumer feasibility

2026-10-12。研究base `5984154646e3154fe0aabf5e2033515cfd782282`。
専用branch `investigation/htp-fp32-custom-consumer`。
最終判定は実機repetitionの集計後に記載する。

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

## 実装と能力監査（進行中）

SDK実ファイルのQHPI header、QHPI sample、FP16 sample、HTP Makefile、
local migration guideを参照。SDK/third-party sourceをrepositoryへコピーしていない。
自作のdiagnostic packageはflat FLOAT32 DDR input/outputとINT32 receiptを登録する。
ARM host libraryはprepare用、Hexagon libraryはexecute用。CPU backendは選択しない。
HVX worker上でscalar FP32命令を実行する構成であり、HVX vector FP32実装とは主張しない。

package buildは成功。初回CPU registrationは`qhpi_register_ops_v1`の未解決symbolで4005。
固定`libQnnHtpPrepare`をhost libraryのDT_NEEDEDへ追加して解消。
`DSP` targetは4003（invalid target）；固定backendは`CPU`と`HTP`で登録成功。
初期MAIN resourceにはruntime warningがあり、HVX resourceへ変更して再検証する。
これら初期試行は昇格証拠に含めない。

公式オンラインの[Op Package Migration Guide](https://docs.qualcomm.com/nav/home/writing_opPackage_migration_guide.html?product=924033590759186372)
はversionごとの再buildを要求する。現在の固定header/libraryを正とし、最新の記述を実行可否の保証にしない。

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
