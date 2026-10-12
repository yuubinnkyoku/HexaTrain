# HTP LayerNorm numerical redesign

開始source `bc6b977482fdf3e355a6b86b50aab53eb46e9c0d`。
専用branch `investigation/htp-layernorm-numerical-redesign`。既存experimentとUNSTABLEを保護する。

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

結果・採用判断・provenanceは測定後に追記する。40k→60kと200kは未開始。

補助的に、FP32 DSP kernelの設計oracleをHostのみで検査する。
anchor-relative mean/center、Kahan accumulation、scaled RMSとhypot(RMS,sqrt(epsilon))を使う。
全100ケースにforward0.002・各gradient0.02を適用。CPU実行はreference/設計検査だけであり、
runtimeに接続しない。DSP/HVXでの精度・性能・graph integrationを保証するものではない。
