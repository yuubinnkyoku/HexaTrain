# DSP FP32 LayerNorm and downstream MatMul boundary

2026-10-13。base `bb0b3a8aff22b5d5beb1a7083fa3ff6dac541d3f`。
branch `investigation/htp-fp32-layernorm-matmul-boundary`。
この文書は診断の事前登録と最終結果を分けて記録する。

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
