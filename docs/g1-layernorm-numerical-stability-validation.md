# G1 LayerNorm numerical stability validation

2026-10-12。検証branch: `investigation/g1-layernorm-stability-validation`。開始source: `1a57e3ae214fd320004b8f9f32715ab86f50c986`。

## 結論と停止理由

**検証結果はBLOCKED。60k stress testは開始していない。** selective row rescalingは実数上正しい変形で、既知の41,647の平方overflowを回避する。しかし固定QAIRT/HTP上の独立reference比較では、finiteな低分散入力にもbackward誤差があり、既存のgradient absolute error limit `0.02`を超えた。0.5という単一scaleではcentered値512以上の平方overflowも防げない。これを恒久的に正しく学習できるLayerNormとして採用する証拠は得られなかった。

| 分類 | 判定と範囲 |
| --- | --- |
| NUMERICAL_ORIGIN_CONFIRMED | 既存41,647のLN1平方overflowの証拠をidentity照合して再利用。LN2/FFN伝播も既存判定を維持 |
| MATHEMATICAL_CORRECTNESS_VERIFIED | **実数上のforward/backward等価性とCPU analytic/numericはPASS。HTP実装のreference accuracy gateはFAIL**。包括的なcorrectness verifiedとは分類しない |
| KNOWN_FAILURE_FIXED | 既存40k→42k・固定失敗stateのfinite回避は確認済み。失敗stateの全モデルgradientがCPU referenceと正しく一致するという追加証明はない |
| EXTENDED_STABILITY_VERIFIED | **未成立**。42k→60kは0 updates、45k/50k/55k/60k checkpointなし |
| LONG_HORIZON_STABILITY_UNPROVEN | 200k以上は未検証。大きなactivationを生むdynamicsと高LRの因果関係も未解決 |

元runのUNSTABLE、resume拒否guard、保護40k、旧100k baselineを変更していない。今回production LayerNorm/optimizer/LRの数式変更は行わず、独立診断・reference・テスト・公開aggregate exporterを追加した。新しい数値実装を複数同時に試すことも、finite性をaccuracy PASSへ読み替えることもしていない。

## 1. 数式と実装監査

正本はlocalの共通 `TransformerTrainingLayerNormBuilder`。generalized G1は各layerのLN1/LN2に別tensor IDを渡す。forwardの実際の順序:

```text
m = ReduceMean(x, last axis, keep_dims=true)
c = x - m
a = ReduceMax(Abs(c), last axis, keep_dims=true)
s = Select(a > 255.875, 0.5, 1)
epsilon_s = Select(a > 255.875, epsilon * 0.25, epsilon)
c_s = c * s
q = c_s * c_s
v_s = ReduceMean(q, last axis, keep_dims=true)
inv_s = Rsqrt(v_s + epsilon_s)
inv = inv_s * s
xhat = c * inv
y = xhat * gamma + beta
```

meanを取る前のinput scalingではない。危険なsquareを二通り計算してSelectする方式でもない。Selectはscale/epsilonにのみ適用し、squareは選択後の`c_s`に対して1回構築する。通常branchのscale1とepsilon、gamma/betaのbroadcast/orderを保持する。診断graphも別の近似実装ではなく、この共通builderのforward/backwardを呼ぶ。

任意の正のsについて `v_s=s²v` なので `s / sqrt(s²(v+epsilon)) = 1 / sqrt(v+epsilon)`。epsilonの0.25倍は元epsilonを増減する意味ではなく、scaled varianceの単位へ変換する。gamma/betaは再スケールしない。

backwardは次式で、**unscaled** inverseとxhatを使用する:

```text
dxhat = dy * gamma
dx = inv / D * (D*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
dgamma = sum_rows(dy*xhat)
dbeta = sum_rows(dy)
```

scale選択はinput-dependentだが、実数上はsへの依存性が恒等的に消える。branch内のds/dx項も相殺され、切替点の両branchが同じLayerNormと同じ導関数になる。epsilon>0の元LayerNormは滑らかであり、scale selectorの非微分性を学習gradientへ追加する必要はない。

ただし丸め後の両branchは同じ関数とは限らない。HTP量子化のplateauや閾値の数値的なjumpの導関数を、実数LayerNormのanalytic gradientと混同しない。epsilon、variance、rsqrt、reduction、backwardに有限精度の誤差が残る。

## 2. 過去のincidentとの関係

[過去のlate-nonfinite report](qnn-htp-late-nonfinite-root-cause.md)では、小型TransformerのLN2で64倍したcentered値が約256を超え、squareが+Infになった。64→8は同じscaled variance/epsilon/inverseの実数等価変形を保ったまま、低分散のprecision補助とoverflow余裕のtrade-offを変更した。5 seeds・固定state replay等のfinite regressionは、そのモデルと入力分布の有限性を確認した。

今回はG1/T32/D64/L19の長期高LR条件で、**scale1でも**centered値−269.75に達した。同じsquare range問題の再出現だが、過去の8倍scaleが今回のG1へ誤って残っていたという事象ではない。G1のforensic optionはscale1、必要時だけ0.5。低分散を拡大する旧scale64の別micrographは現在も別実装条件である。

[旧backward test](qnn-htp-layernorm-backward.md)はD8/6 rows/2ケース。低分散upstreamを0.005倍し、dX/dGamma/dBetaの最大絶対誤差0.02をgateにした。forward誤差約0.0587は記録したが、広範囲のforward accuracyのgateではなかった。固定state・finite trainingがPASSしても、未試験の大きなmeanやfull upstreamの低分散backwardを保証しない。今回各processで旧microtestはPASSしたまま、新しいfull-upstream検証はFAILした。

同SDKの[precision-control調査](qnn-nicopedia-htp-precision-mitigation.md)も、graphのFLOAT32指定から内部FP32演算を保証できないことを示している。今回はその既存試験を再実施していない。全HTP演算を一律FP16と断定せず、観測したintermediate/error/rangeを報告する。

## 3. Independent double / FP32 / numerical gradient

`layernorm_stability_reference.h`は2-pass mean/variance、unscaled LayerNorm定義、独立analytic backwardとscalar loss `sum(y*dy)`を実装。QNN builderやoptimizerをreferenceとして使わない。double referenceとfloat演算版を分離する。

HostではD=8/16/64/128、2 rows、16 recipes、計64ケース。全X/gamma/betaをdoubleでcentral perturbationし、h=1e−6と2e−6を両方確認した。閾値255.875の前・同値・後も含む。scaleあり/なしのdouble forward/dX/dGamma/dBeta差は全ケース0、analytic/numeric最大絶対誤差は **2.47964530331e−5**。既存CPU gate5e−4より小さく、両hでPASS。

入力はfloatのrecipeをdoubleへ正確に変換して比較する。FP32入力へ丸められる前の理想的なdecimal値をreferenceにしない。mean/variance/inverse/xhat/yと全gradientのreferenceを作り、float referenceとの差もprivate reportに保存する。HTPの比較対象は元のFP32入力に対するLayerNormであり、内部で失われた入力情報に合わせてreferenceを丸め直して合格させない。

HTP量子化のstaircaseをh=1e−6でfinite differenceしてもanalytic LayerNormの検証にならないため、numeric gradientはdoubleで検証した。HTPには同じupstream/gamma/betaでanalytic referenceを直接比較する。残るbackend誤差をCPU finite-difference PASSで打ち消さない。

## 4. 事前固定したacceptance criteria

| 判定 | 基準 |
| --- | --- |
| CPU analytic / central difference | max absolute error <5e−4、2つのhを両方確認 |
| 未発動の透明性 | 元scale1 graphとy/xhat/dX/dGamma/dBetaの全float bits一致 |
| ordinary / boundary / rescaled high-variance forward | y/xhat max absolute error ≤0.002。既存direct LayerNormの基準を借用した保守的な診断gate |
| low-variance forward | 新しい診断bound0.06をdevice測定前に固定。歴史的0.0587から設定した値で、旧テストのacceptance thresholdとは呼ばない |
| 全required caseのbackward | dX/dGamma/dBeta各max absolute error ≤0.02。旧backward gateを維持、upstream縮小なし |
| finite | 最終5 outputsおよびtap付きgraphの全21 intermediate tapsを別検査 |
| QNN | create/finalize/execute成功、execute failure0、CPU fallbackなし |

required casesと、512/4096という既知のsingle-half range外を探るexploratory casesを事前に分けた。exploratory nonfiniteをtraining SUCCESSには数えない。requiredな有限入力でaccuracyがFAILしたため、exploratoryを除いても60kの開始条件は成立しない。thresholdを事後に広げず、絶対誤差を都合のよいrelative metricへ置き換えていない。

## 5. Deviceケースと独立再現

固定QAIRT2.48.40.260702、Build ID `2.48.40.260702151143`、同じ保護実験の物理端末。独立processの試験を3回行った:

| 公開label | scope | execute数 | accuracy |
| --- | --- | ---: | --- |
| INITIAL_REFERENCE | D8/64/128、14 recipes、各3 repeats、tap付き | 126/126成功 | FAIL |
| NATIVE_CONTROL | 同じ14＋2 sparse recipes、tap有無・元scale1比較 | 219/219成功 | FAIL |
| NATIVE_REPEATED | 同じNATIVE_CONTROL、独立process、同一APK | 219/219成功 | FAIL、数値record完全一致 |

上のcountsは独立LayerNorm graphのみ。各processで最初に実行した旧backward microtestの2 callsは別で、legacy gate=PASS。3 repeatsは同一process内であり、3独立training runsとは数えない。新しい学習updatesは0。

各recipeのrow0は指定のsymmetric ±amplitude（または2要素だけのsparse ±amplitude）、row1は通常±1。非自明なgamma/beta、dyを使い、dyを0.005倍しない。ordinary、小centered、低分散、epsilon支配、constant、大mean＋低分散、255.75/255.875/256、269.75、400、512、4096、大mean8192±100、sparse outliersを含む。rowsの混在、全要素が大きいreductionも検査する。

tap有無の全5 final outputsは、48のdimension/recipe組でbitwise一致した。未発動9 recipes×3 dimensions=27組は元scale1 graphともbitwise一致。小型graphの全最終outputsについての観測であり、任意graph compiler configurationへの保証ではない。

## 6. Forward / backward反例

D64、各3 repetitionsと独立process再実行で同じ値:

| row0 recipe | finite | y max abs error | dX max abs error | dGamma max abs error | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| ordinary ±1 | true | 6.25e−6 | 2.4951e−6 | 3.1250e−6 | PASS |
| small centered ±0.01 | true | 0.00042198 | **0.02722150** | 0.00012847 | backward FAIL |
| 0.25±0.001 | true | 0.00823498 | **0.27230037** | 0.00247237 | backward FAIL |
| 0.25±0.00001 | true | 0.00395525 | **0.16493980** | 0.00118845 | backward FAIL |
| constant0.25 | true | 6.25e−6 | **0.16564399** | 1.8750e−6 | backward FAIL |
| 1024±0.001 | true | **0.36883332** | **6.87336948** | **0.11065187** | forward/backward FAIL |
| ±269.75、select0.5 | true | 6.25e−6 | 2.4951e−6 | 1.8750e−6 | このrecipeはPASS |
| sparse ±269.75 | true | 0.00052871 | 5.5645e−6 | 0.00022534 | このrecipeはPASS |
| mean8192±100 | true | **0.02539063** | 9.4452e−5 | 0.00781437 | forward FAIL |
| ±512、select0.5 | false | nonfinite | nonfinite | nonfinite | range反例 |

dBetaはこれらのdyに対してmax error0。これはdX/dGammaが正しい証明ではない。D8/128にも同種の反例があり、最終suiteではrequired 126 executions中51がaccuracy FAIL（42 dimension/recipe組中17組、各3 repetitions）。fail判定はpublic aggregateから再計算できる。

## 7. Intermediate precision / rangeの測定

- `0.25±0.001`のcentered maxは0.0009765625、variance9.53674316406e−7、variance+epsilon1.09672546387e−5、inverse302。double inverseとの差0.48850675。center/subtraction・square・epsilon・rsqrtの有限誤差がbackwardで増幅する。
- `1024±0.001`ではFP32入力に残っている差に対し、HTPのcentered row0はすべて0、variance0。double varianceは9.53674316406e−7。centerの後だけをscaleしても、すでに失われた差は復元できない。
- constant rowではvariance0、epsilon出力1.00135803223e−5、inverse316、double inverse316.227766016838。finiteであってもinverse精度とbackward絶対誤差のgateは別問題。
- 255.75: square65408、255.875: square65472、どちらもscale1でfinite。256: scale0.5、square16384でfinite。269.75: scale0.5、square18192でfinite。閾値直下のnormalized値には丸め差があり、実数上の完全なbranch連続性をbitwiseには保証しない。
- 全要素が255.875のrowでもReduceMean(square)はfiniteだった。内部がFP32 accumulatorか、別の等価計算法かはこれだけでは断定できない。単純なFP16 ReduceSumのrange仮定だけで安全性を判定しない。
- 512: selector0.5は正しく発動し、centered_s=256、square64要素が非有限。4096も非有限。QNN return0で、squareからvariance/inverse/backwardへ非有限が進む。観測したrsqrt outputはInfで、理想rsqrt(Inf)=0の代用推論をしない。

安全なrescaled recipesでは、row0 scale0.5 / row1 scale1とfinite outputsを確認した。選択されないscale1のunsafe squareが別branchで計算されるgraph構造はない。この範囲ではunsafeな非選択経路の汚染も観測しなかった。

固定255.875/0.5を任意のactivationへ一般化する安全余裕は確立していない。512の反例で上限不足を直接確認した。さらに大きいmeanのreduction、epsilon underflow、巨大upstreamのbackward range、他backend/SDK/shapeのroundingは別途検証が必要。

## 8. 既知failure regionと正常prefixの再利用

既存evidenceを再hashし、独立read-only checkpoint decoderで42kを検査した。source `8a6350203a6cb214d6d5ca9a288e5a344a0b46be`の旧診断であり、今回の新sourceで再学習した結果に書き換えていない。

| anchor | SHA256 |
| --- | --- |
| protected40k | 89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d |
| original/adaptive41k | afaf63968fdaa15ac90c93fc411c1e646ab4c45cbb41a45b34b87c72f9b10bbe |
| last finite / first-scale前41,646 | def39c33ba33aa66978588b0ffa61d05d08f29574a3442e522ff43fc5d0db654 |
| adaptive42k | 3fd11199db8ecfeb988fad07d01491ab81a20b62dd1646b99a7c93b066b89a51 |

41kと41,646のwhole checkpoint hash一致により、parameter/Muon momentum/Adam m/v、cursorを含むserialized stateのparityを再確認した。正常40k固定callの32,768 logitsと760,960 gradientsのraw parityも既存hashで確認済み。修正未発動時のcheckpoint/update/optimizer透明性のanchorを維持する。

既知41,647/microbatch1ではlayer17 LN1/token26が初回scale0.5、LN1 square18192、LN2 square18016、全regular outputsと114 tapsがfinite。40k→42kの2,000 updates / 16,000 successful calls、24/19,456,000 norm-row rescalings、QNN/HVX failure/fallback/nonfinite0は元evidenceのまま。

42k decoderはseed1/Muon+Aux Adam/cursor42001 micro0/dataset `fnv1a64:245228bcbe304f51`を確認。parameter L2=462.5975845、Muon momentum L2=0.07028990、Adam m L2=0.37138712、Adam v L2=1.05369847、すべてfinite。order `fnv1a64:83f9c96cf10261a0`、tokenizer/cache identityも一致。resume可能なidentityと、今回要求されたaccuracyの認証は別であり、これを理由にstress launch gateを迂回しない。

## 9. G1 / optimizer / residual gradient

共通backwardはLN1/LN2に同じ定義を適用し、gamma/betaと入力gradientを返す。残りのresidual加算、G1 attention backward、gradient accumulation、Muon/Aux Adamへのparameter groupingは今回変更していない。通常固定callの全parameter gradient bitwise parityは旧anchorの範囲で維持される。

今回の独立graphにはattention/optimizerがなく、低分散のgradient accuracy FAILと512 square failureを再現した。これらの直接原因をMuonやAux Adamに帰属させる証拠はない。一方、発動したfull G1 graphの全gradientがreferenceに正しく一致するという検証は未完了。LN局所gateがFAILしているため、finite gradientをoptimizerへ渡せたことだけで「正しく学習した」と判断しない。

42k時点のgate/norm/LR/QNN/HVX telemetryは既存レポート・aggregateを参照し、新しいtrajectoryの結果として再解釈しない。45k/50k/55k/60kのgate saturation、optimizer/update norms、Val/Devは取得していない。final_testは使用していない。

## 10. Performance / resources / generalization

小型共通builder graphはunscaled24 nodes/37 tensors、adaptive29/42。全19layerの既存G1 adaptive診断では190 nodes/193 tensors増分、追加APP_READ byte予算4864を旧application policy内で検証済み。Select/maxはdevice graph内であり、scaleを選ぶためのhost↔device round tripは追加しない。ただし全normのhealth tapsをmicrobatchごとに転送/検査する診断コストはある。

旧adaptive40k→42kのtraining_step_ms=136.1879269、fwd_backward_ms_per_update=36.07842553、step_wall_p95=176.050468ms、optimizer wall=37.09837113ms。棄却済み常時0.5条件ではそれぞれ124.7498688ms /33.20011577msで、単純差は約9.17% /8.67%。**異なるtrajectory・監視scope・run時刻なのでselective方式だけのcausal overheadとして採用しない。** exclusive benchmarkは行っていない。

新しい独立graphのcreate/finalize/executeは全成功。3 runsのforeground takeover0、thermal status0。repeatのbattery温度は開始/終了34°C。storage gate20GiB/12GiBを毎launch維持。厳密なgraph allocation、finalization wall time、device RSS/peak、paired long-training throughputは未測定で、恒久実装としてのperformance認証は未成立。correctness FAIL後に長期trainingを回してperformanceのみを埋めない。

builderはLN1/LN2・全19layer・rowwiseでありlayer17のhardcodeではない。独立試験ではD8/64/128と混在rowsを確認したが、precision反例も同じように一般化した。production adaptive optionは現在G1 FULL/scale1/全norm監視に制限され、Controlの既存scale8 recipeへの無条件適用は未検証。既存100k recipeのcheckpoint/品質は変更せず、今回の修正の品質結果へ読み替えない。

## 11. 検証と再現入口

Fast、Host reference、固定QAIRT Qnn build/audit、headless独立試験を実施。通常build/contractのPASSと**候補accuracyのFAIL**を分けて記録する。正式quality milestone/200k acceptance/成功のFormal認証を行ったとは主張しない。

| gate | 結果 |
| --- | --- |
| 最終Fast | PASS、53.1s。公開exportの決定性修正後も通常gateを通過 |
| Fast（native control追加時） | PASS、91.9s。parser/privacy/diff、JVM unit、CPU contract |
| 独立CPU reference | PASS、64ケース、2 finite-difference h、最大誤差2.47964530331e−5 |
| 最終Qnn内の全Host contract + diagnostic | PASS、250.3s。optimizer/NS、G1、shape、registry、checkpoint/resume、disabled runtime、parity policyも含む |
| 最終Qnn | PASS、286.4s。固定QAIRT確認、28.2s QNN build、APK audit。optional inventory不足はadvisory、coreはcomplete |
| 最終Device | PASS、82.9s、BACKGROUND_CORRECTNESSのdevice-probe。新LayerNorm候補のaccuracy FAILをこのsmokeで置き換えない |
| public exporter | SelfTest/実private report照合を実施。hash gateでJSON key順の非決定性を検出しordered fieldsへ修正。数値内容・classificationは変更せず、独立process再exportのbyte hash一致を確認 |
| headless validation用debug/androidTest APK | build/audit/installed identityを確認。新suiteは既存backward suiteと別mode103、既存enum値は維持 |
| LayerNorm required accuracy | **FAIL**、3独立process。QNN API成功と区別 |
| 60k stress / full-model rescaled backward認証 | **BLOCKED / 未実施** |

```powershell
# SDK rootはprivate環境の固定値を明示。別SDKへfallbackしない。
.\scripts\run_layernorm_stability_validation.ps1 `
  -QairtSdkRoot $qairtRoot -ExpectedBuildId '2.48.40.260702151143'
# 現候補はcorrectness FAILを返し、stressをBLOCKEDにすることが期待される。
.\scripts\export_public_layernorm_stability_validation.ps1 -SelfTest
```

公開aggregateは `docs/results/g1-layernorm-stability-validation/aggregates.json`。typed allow-list exporterがsource/SDK/count、required FAILとclassification、nonfinite→null+explicit flagを検査する。raw tensors、corpus、prompt、run IDs、device/APK identities、logcat、ローカル絶対pathはcommitしない。

## 12. 次の実装と60k/200k計画への条件

1. selective0.5は既知failureのbounded workaroundとして保持し、広範囲のcorrectnessがverifiedな恒久LayerNormとは呼ばない。LR/clipping/optimizer resetでreference誤差を隠さない。
2. range-only危険判定とprecision危険判定を分けて設計する。低分散rowの旧gradientをbitwise維持することと、今回の0.02 accuracy gateを満たすことは、反例の入力では同時に成立しない。「通常row」の適用domainと新条件を先に定義し、parity gateを無言で弱めない。
3. centered値512以上には多段/指数scale等の候補が必要。ただしepsilon underflow、mean/centerでの情報消失、rsqrtの精度、backward rangeも独立に解決・検証する。center後のscalingだけでは大meanのcancellationを復元できない。高精度centering、別kernel、graph split/host処理などは別の実装条件として性能とparityも比較する。今回は未検証候補を追加採用しない。
4. 新候補で同じdouble/FP32/finite-differenceとHTP反例suiteを通し、source/hashを固定して41,647固定stateの全必要gradient/残差経路を検証する。
5. そのPASS後のみ独立40k fork（または数値条件互換性を確認した42k）から、45k/50k/55k/60k境界のfinite/QNN/HVX/rescale/norm/gate/LR/thermal/storage/performanceを記録する。上限60k、optimizer/data/resume semanticsを維持。Val/Devは計画時に固定、qualityを見てLRを調整しない。
6. 60kをPASSしてから、追加seed、より広いactivation/upstream domain、長期品質を分けた**新しい200k計画**を審査する。60k有限だけで200k以上を保証しない。現時点で200k正式実験へ進める判断は **NO-GO**。

6031はUNRESOLVED / DORMANT / WATCHを維持。今回のprobeで6031は発生していない。高LRと大activationの因果関係、quality比較、長期schedule優劣は別の未解決問題。

## 13. Provenance / 保存

| source | exact commit |
| --- | --- |
| protected original experiment | 6bca2a7a24f04f8b8b2d4084feb337a12c21e00a |
| previous investigation final | 1a57e3ae214fd320004b8f9f32715ab86f50c986 |
| initial shared reference/runtime | 445a7fceab2bb98f0afcc0bb6e8c5bc29f51e19c |
| INITIAL_REFERENCE launch | 3403874d58646d1b121c9b0672761aecec661901 |
| NATIVE_CONTROL launch | 0ffa6351a9b53f4fd7dba659761627babaf28653 |
| NATIVE_REPEATED launch / exporter | 79673b3dcb86bbf65d2548e706e6568e9955c0cf |
| reused adaptive training source | 8a6350203a6cb214d6d5ca9a288e5a344a0b46be |

NATIVE_REPEATEDはNATIVE_CONTROLと同じinstalled APK。間のnative diffはtest thresholdについてのcommentのみで数値code変更なし。exact build/APK/physical-device identityは各headless audit/installed identityとprivate ledgerで解決する。initial script追加もnative sourceを変更していない。

cache SHA256 `bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf`、tokenizer `9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`、manifest `e8a77e06abab3da5ff7fe26eb9744c3bffad8cf5fbf006bbf4c7f582393961e4`を開始時に照合。旧HVX SDK/compiler identityは元planのまま、新しいisolated LayerNorm試験ではHVX optimizerを実行しない。

private baseは `build/reports/g1-layernorm-validation/`。`initial-identities.json`、`public-index.json`、各`validation-plan.json`、`device-report.txt`、reference/verification logs、reused42k decoderを保存。headlessのaudit/installed identity/foreground/thermal evidenceは `build/reports/qnn-headless/`の同じprivate IDで照合する。既存forensic evidenceは別namespaceのまま、削除/上書きしない。

このreportと診断tooling/allow-list aggregateのみを専用branchへcommitする。最終report commitのexact SHAはGit履歴とprivate最終identity recordで解決し、working treeをcleanにする。push/PR/main merge、元UNSTABLE resume、production200k再開、app data clear、force-stop、SDK/firmware変更、既存artifact削除を行わない。
