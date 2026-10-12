# G1 Long Linear numerical instability investigation

2026-10-12。対象は `experiment/g1-long-linear-200k` の `6bca2a7a24f04f8b8b2d4084feb337a12c21e00a`。調査branchは `investigation/g1-long-linear-nonfinite`。元の実験のclassificationは **UNSTABLEのまま**であり、200kは再開していない。

## 結論と判定の範囲

step 41,647 / microbatch 1 のlogits NaNについて、有限なLayerNorm centered値の二乗がHTP経路で+Infになる発生演算を同定した。layer番号・microbatch・tensor indexは以下すべてzero-based、updateは学習のglobal stepである。

Root classification: **CONFIRMED_ROOT_CAUSE / APP_NUMERIC_RANGE_DEFECT**。これはこの状態・入力に対するnumerical originの判定である。高LR、Muon、Aux Adam、G1 gate collapseのいずれが大きなresidualを生む学習dynamicsを支配したかは別の問題で、確定していない。HTP内部精度をFP32とも一律FP16とも断定しない。

確認済みの回避策は、LayerNormの各rowで危険な範囲のcentered値だけを0.5倍し、epsilonも0.25倍してinverseを戻す条件付き再スケールである。LR・optimizer・data orderを維持し、同じ失敗stateを有限にした。40k→42kの独立診断も有限で、41k checkpointと最初の介入直前41,646 stateが元条件とfile bytesまで一致した。初回介入は元のfirst bad update/microbatch/layer/rowに一致した。

常時scale0.5という先行候補は、正常prefixの1,000 updates後にparameter relative L2差0.230391が生じ、**互換修正としてのparity gateはFAIL**だった。この候補は採用しない。条件付き候補は確認した正常anchorのbitwise parityと元failure regionのfinite gateを通過したが、確認範囲はseed1・40k→42kである。production defaultは変更せず、200kの安定性・品質・100k recipe全体との互換性を主張しない。

## A. Reproduction / localization

G1 V1024/T32/D64/FFN128/L19/H2、760,960 parameters、batch8、seed1。Muon HVX_W8、NS5、momentum0.95、Nesterov、Aux Adam、weight decay0、clipping無効。4k→200kの元のLinear LR、同じexpanded cache/tokenizer、同じtraining orderを用いた。

学習stepのforward/backward数値演算をHTPで実行し、MuonはHVX_W8を使用した。host側ではdata/parameter orchestration、loss処理、gradient accumulation、Aux Adam等を行う。

| 条件 | 実施数 | 結果 |
| --- | ---: | --- |
| 元APK、40kから独立fork | 1 | 13,170 calls成功、243,059 nonfinite、logits NaN7,428 |
| captureのみ、40kから独立fork、tap NONE | 1 | 同じcall数・非有限値数。first detected update41,647、microbatch1を直接記録 |
| 41,646の有限serialized state、cold process、NONE | 1 | optimizer未適用の1 graph callで同じ異常 |
| 同じ固定状態、COARSE | 1 | 同じraw logits。layer17 outputで最初のlayer-boundary NaN |
| 同じ固定状態、FINE layer17 | 1 | 同じNaN位置。有限logitsに小さいobserver effectあり |
| 同じ固定状態、LN1_SQUARE layer17 | 1 | NONEとraw logits一致。有限inputからsquareの+Infを同定 |
| 同じ固定状態、scale0.5、LN1_SQUARE | 1 accepted run | 全regular APP_READとtapが有限。optimizer未適用 |
| 同じ40kからscale0.5、40,001–42,000 | 1 | 2,000 updates有限、16,000 QNN training calls成功、HVX RPC failure/fallback/nonfinite各0 |
| 元condition、全19layerのsquare監視、40kから独立fork | 1 | 同じ41,647 / microbatch1で破綻。直前stateと失敗logits raw hashがNONEと一致 |
| 条件付きscale、同じ固定失敗state | 1 | 最初のscaleはlayer17 LN1/token26、全regular APP_READ・114 tapsがfinite |
| 条件付きscale、同じ40kから40,001–42,000 | 1 | 2,000 updates・16,000 QNN callsが有限。41k/41,646 stateは元条件とbitwise一致 |

元条件のlast verified finite updateは41,646。ただしここでのfiniteは通常APP_READ出力とoptimizer stateの検査範囲であり、全hidden tensorの過去の健全性を意味しない。元条件のfirst detectedは41,647 / microbatch1、40kからのtraining graph execute index13,169（one-based13,170）。original APP_READ registryではlogitsが先に検出される。これはcausal operator順ではない。

独立trajectoryの元条件3回でfirst bad updateはすべて41,647だった。cold fixed-state probesも同じ入力で再現した。seed間の発生率や、未試験のdevice/runtimeにおける非決定性は測定していない。

全19layerのLN1/LN2 centered_s・squareを毎microbatch監視する独立再実行（76 taps）でも、41,647 / microbatch1でlayer17 LN1 squareが最初に非有限となった。41,646までの監視対象はfiniteで、失敗直前のserialized state SHAと失敗logits raw SHAはcapture NONEと完全一致した。これにより、保護40k以降のLayerNorm squareに限定したhistorical onsetとprefixの無変化を確認した。すべてのhidden演算や40k以前のhistoryまで証明したとはしない。追加tapを含む非有限値数は244,341であり、元APP_READの243,059とは検査対象数が違う。

同じinput/targetのidentityはhostのread-only cache/order再構成と一致した。data selection index333,169、recordの対応とラベル範囲をprivate evidenceで照合した。公開本文・raw token・raw promptは保存しない。

## B. Numerical origin / propagation

41,647 / microbatch1、layer17 LN1:

| tensor | 観測 |
| --- | --- |
| layer16 output | 2,048/2,048 finite、min−238.75、max279.25、RMS23.28938 |
| ln1_centered_s | 2,048/2,048 finite、min−269.75、max248.25、RMS22.63473 |
| ln1_square | 2,047 finite、+Inf1、first bad1715 = token26/channel51 |
| ln1_variance_eps | +Inf1、token26 |
| ln1_inv | +Inf1、token26。RSQRT内部単体の挙動をこのtapだけから断定しない |
| ln1 affine output | token26の64 valuesが非有限（+Inf29/−Inf35） |
| layer17 output | NaN62、first1664 = token26/channel0 |
| layer18 output | NaN291 |
| logits | NaN7,428/32,768 |

実測したsquare input element1715は−269.75。CPU FP32式の結果は **72,765.0625** で有限。HTPのAPP_READ squareは+Infで、QNN returnは0だった。宣言・転送bufferがfloat32であることは演算内部のFP32 rangeを保証していなかった。

Layer17のQ/K/VにもNaNが現れた。一方、この失敗入力ではhead probabilities/contextがfinite zero、gate出力もfiniteで、residual1はlayer16出力と一致していた。したがって「LN1 NaNがattention contextを経てそのままlogitsへ伝播した」という単純な説明は誤りである。LN2も同じcentered inputでsquare overflowを起こし、LN2の非有限値からFFN、layer output、次layer、logitsへ伝播した。

```text
finite large residual / centered value
  → LN1 ElementWiseMultiply(centered_s, centered_s): +Inf
  → LN1 normalization / Q,K,Vの異常
  → この入力ではattention側の異常がfinite zero contextに隠れる
finite large residualが残る
  → LN2 square: +Inf
  → LN2 / FFN / layer17 output: NaN
  → layer18 / logits / backward: 非有限値
```

checkpoint/data/optimizerを使わないone-node shape[1] procedural graphでも、−269.75の二乗は3/3でQNN return0・+Inf、−134.875では3/3でreturn0・finite18,192だった。CPU FP32の後者は18,191.265625。この結果はFP16相当のrange/roundingと整合するが、全HTP演算の内部精度の断定ではない。同一process内の3 repetitionsであり、3独立training runsとは数えない。

## C. Hypotheses / causal assessment

| 仮説 | 支持 / 反証 / 判定 |
| --- | --- |
| HTP経路のsquare range overflow | 有限input、実測+Inf、CPU-finite、one-node再現、scale介入での同一状態救済が一致。CONFIRMED |
| 高LRがactivation driftを起こす | 40kでpeakの約82.5%、旧recipeより大きいLR、有限parameter norm増加とlocal residual tailが観測された。LRだけの介入を行っていないためPLAUSIBLE_ONLY |
| Muon / NS / HVXがfirst origin | 現updateのMuonを適用せずcold forwardで再現。直前stateと既存health stageはfinite。first originとして反証。過去のdynamicsへの関与や実装全体の無欠陥は証明していない |
| Aux Adam / G1 weightsがfirst origin | Adam moments/gate weightsはfinite、gateそのものはfinite、optimizer未適用で再現。first originとして反証。gate collapseの品質上の意味は未解決 |
| 無効input / data-order mismatch / checkpoint corruption | checkpoint codec/cursor、cache/tokenizer identity、hostとdeviceのinput/target hashが一致。one-nodeはこれらを使用しない。今回のsquare異常の必要原因として反証 |
| stale buffer / lifetime / registry bug | fresh processとone-nodeでも再現し、poison0、正常prefixの全stateが元APKとbitwise一致。今回のrange異常の必要原因として強く否定。全コードのmemory安全性証明ではない |
| 6031と共通原因 | 今回のQNN returnは0。共通因果の直接証拠なし。別incident、UNRESOLVED / DORMANT / WATCHを維持 |

高LRを原因と確定するためだけに多数のLR条件を走らせることはしなかった。最も情報量の大きい同一stateの演算range介入を優先した。qualityによるLR探索・final_test使用は行っていない。

## D. Optimizer / gate / state telemetry

| serialized state L2 | 40k | 41,646 |
| --- | ---: | ---: |
| parameters | 454.603094 | 460.816997 |
| Muon momentum | 0.06181053 | 0.07423747 |
| Adam m | 0.34850012 | 0.41901807 |
| Adam v | 0.15613546 | 0.15450903 |

全stateがfinite。parameter max absは両時点6.0766935。global normの小さい増加からlocal activation tailが安全とは推定できない。40k gate mean0.009597、gate<0.1が99.19%という既存観測はfinite gate collapseであり、NaN生成の証明ではない。

41kの元条件ではgradient L2=2.083238、parameter L2=458.351987、Muon last-update delta L2=0.351129、Aux Adam=0.210198。update/parameter ratioはそれぞれ約7.66e−4、4.59e−4。gradient、momentum、normalized、NS output、update、parameters、HVX outputの各health gateはfiniteだった。これはcheckpoint boundary/window telemetryであり、全stepのnorm時系列を取得したとは主張しない。

42kのscale0.5条件はgradient L2=1.686050、parameter L2=464.212857、Muon delta=0.318254、Adam delta=0.211855。これを元条件42kと比較することはできない。元条件は到達していない。

条件付きscaleの42kではgradient L2=1.717437、parameter L2=462.597585、Muon delta=0.346338、Adam delta=0.215222。Aux LR=0.002689285669、Muon LR=0.006112012547で、元のLinear scheduleのまま。41kでは両state/telemetryが元条件と一致し、Aux LR=0.002705357037、Muon LR=0.006148538552だった。clippingは無効のままで、optimizerの全stage healthはfinite。

## E. Implementation / parity / verification

実装は独立forensic runner、scoped forward taps、正常pre-update stateの保存、input/state/gradient/tap health summary、固定state・microbatch probe、raw gradient/logitsのprivate保存、read-only host checkpoint/input/CPU reference inspector、allow-list public exporter。scope NONEの通常training graph・optimizer・LR・checkpoint schemaは変更していない。新しいenumは末尾追加で既存値を維持した。

scope NONE/COARSE/LN1_SQUAREの失敗logits raw hashは完全一致。FINE layer17ではNaN mask一致、finite25,340 values中446が異なり、max abs差0.0078125、RMS差0.000309656。fine tapをNO_OBSERVER_EFFECTとは扱わない。

元APKとcapture-only NONEの41k checkpointは全file bytesで一致し、parameter/Muon momentum/Adam m/vの差は0。計測だけによるprefix変更はこの比較で否定した。

scale0.5は `u=s(x−mean); v_s=mean(u²); epsilon_s=s²epsilon; inv=s*rsqrt(v_s+epsilon_s)` を使用し、既存backwardはunscaled xhat/invを使う。実数上のforward/backward数式は同じ。NaN置換、clamp、epsilon増加、optimizer reset、loss skip、CPU fallbackはない。

同じ40k state・40001/microbatch0では、scale1/0.5の全32,768 logitsと760,960 gradientsがraw bitwise一致した。しかし1,000 updates累積後の41k checkpointは一致しなかった:

| state | max abs delta | relative L2 delta |
| --- | ---: | ---: |
| parameters | 1.016953385 | 0.2303910185 |
| Muon momentum | 0.002177124261 | 4.091049109 |
| Adam m | 0.05021008477 | 1.358420301 |
| Adam v | 0.00216034241 | 0.9999847728 |

parameter cosineは0.973130。短いbitwise一致を長期parityへ外挿しない。toleranceの緩和やparity PASSへの読み替えはしない。scale0.5は **new numerical implementation condition** としてのbounded mitigationであり、既存100k/200k recipeへ自動採用する修正ではない。

条件付きrange候補は、各rowのmax abs centered値が255.875を超えた場合だけs=0.5、それ以外はs=1とする。epsilonもrowごとにs²倍し、既存backwardにはunscaled inverse/xhatを渡す。閾値は観測されたrange挙動とbinary16の256直下の値に基づく保守的な診断条件であり、HTP内部精度の公式保証ではない。任意に大きいactivationへの保証はない。全layerの平方と38個の選択scaleをAPP_READで検査し、最初のscale適用前のparameter/optimizer stateを独立namespaceに保存する。通常defaultへの自動導入はない。

条件付き候補の結果:

- 40k / update40,001 / microbatch0の全32,768 logitsと760,960 gradientsは元NONEとraw bitwise一致。全選択scaleは1。
- 元の固定失敗stateでは最初のscaleが41,647 / microbatch1、`layer_017_ln1_safe_scale`、row26だった。計4 norm-rowsを0.5倍し、layer17 LN1 squareのmaxはfinite18,192、LN2はfinite18,016になった。32,768 logits、全parameter gradients、全114 tapsがfinite。input/target hashesと保存state SHAは元条件と一致した。
- この固定状態のlogitsは常時0.5候補ともraw bitwise一致した。ただし常時0.5候補の長期parity FAILはそのまま維持する。
- 40k→42kの1独立trajectoryでは16,000/16,000 QNN executeがreturn0、regular APP_READ・114 tapsとoptimizer stateの各finite gateを通過。HVX RPC failure/fallback/nonfiniteは各0、CPU fallbackなし、focus takeover等の各counterは0。
- 41k checkpoint SHAは元条件と同じ`afaf63968fdaa15ac90c93fc411c1e646ab4c45cbb41a45b34b87c72f9b10bbe`。初回scale直前41,646 snapshotは同じ`def39c33ba33aa66978588b0ffa61d05d08f29574a3442e522ff43fc5d0db654`。両anchorでparameters/Muon momentum/Adam m/vのmax abs差・relative L2差はすべて0。
- 初回介入は41,647 / microbatch1 / layer17 LN1 / token26で元の平方overflowに対応した。2,000 updatesの19,456,000 norm-row observations中、0.5の選択は24回だった。42k checkpointのfinite decodeも通過した。

この比較は保存した正常anchorと健康な固定graph callのbitwise parityであり、すべての過去microbatchの全tensorをraw比較したとはしない。破綻後の元trajectoryはfiniteなparity anchorとして使用しない。数式は実数上等価だが、介入したrowの有限精度表現は変わる。toleranceの緩和、NaN/Inf置換、clamp、loss skip、optimizer resetは行っていない。

最初のadaptive prepareは追加190 nodesの期待数を登録していないためresource estimator照合で停止した。graph executeは0回でBLOCKEDとして保持した。照合を省略せず、追加node/tensorとAPP_READ bufferの予算を明示し修正した。prepare前の失敗に存在しないcapture fileを取得しようとしてADBエラーを重ねる処理も修正し、元errorを保存する。これは数値成功runに数えない。
検証は以下の範囲。正式quality結果・releaseのためのFormal、200k stabilityの保証は含まない。

| gate | 結果 |
| --- | --- |
| 最終Fast | PASS、53.3s。差分/secret-path監査、PowerShell parser、JVM unit、CPU reference等 |
| Host contract + diagnostic suites（最終Qnn内） | PASS、239.0s。finite/edge、G1、registry、optimizer/NS、checkpoint/resume、shape/BOOL/Select、数式等価性、parity policyを含む |
| 最終Qnn | PASS、299.9s。fixed QAIRT、実QNN build、APK audit。optional SDK sample inventoryの不足はadvisory、core/build/auditは通過 |
| HVX付きdebug + androidTest APK | PASS。実験native source8a63502、同一SDK/compiler、各runでAPK audit/installed identityを確認 |
| 最終Device | PASS、83.6s、BACKGROUND_CORRECTNESSのdevice-probe。実験のHTP/HVX finite検証は上記の固定probeと2k trajectoryで別途実施 |
| 常時0.5の正常prefix parity | FAIL。採用しない |
| 条件付きscaleの正常anchor parity / failure region | PASS（上記の限定範囲）。旧100k全trajectoryの新条件での再実行は未実施 |
| public exporter SelfTest + 実source aggregate照合 | PASS。private field排除、hash/count、SDK/order/condition identity、QNN success/finite/完了stepを確認 |

通常runnerが実施した1-record/splitのcheckpoint finite smokeはquality探索には使わず、final_testは使用していない。

最初のfinite probeはreport prefix / explicit QNN success field不足で既存headless契約に拒否された。数値はfiniteだったがrun successには数えなかった。必須fieldを満たす診断reportへ修正し再実行した。別のpreflight/artifact取得失敗もprivate ledgerに残し、numerical reproductionから分離した。安全・parity・UNSTABLE guardは弱めていない。

## F. Recommendations / unresolved work

1. 元のUNSTABLE 200kを再開しない。元conditionの再実行は同じ破綻を再現するため推奨しない。
2. 条件付きscaleは元failure regionで検証した回避策として保持する。導入する新計画には全normのrange/finite監視、同一anchor・data order、独立run namespaceを維持する。42kを越える検証、追加seed、独立品質評価を今回済ませたとはしない。常時0.5候補はparity FAILのため互換修正として採用しない。
3. LR低下・shorter decayは安定性の別仮説として計画できるが、原因修正や品質改善が確認されたとは扱わない。現在の結果をqualityに基づくschedule最適化へ転用しない。
4. Muon/G1に局所的な修正を入れる直接証拠は得ていない。gate dynamicsの研究はnumerical range incidentと区別する。
5. 将来の100k→200k fixed-terminal controlは、既存100kのfinite結果を踏まえた候補として合理的。ただし独立plan・数値安全gate・同一budget比較を先に定義する。100k checkpointや既存品質値を書き換えない。
6. 大きなresidualが生じるdynamicsの支配因子、常時scalingの累積差の最初の発生箇所、全hidden tensorのhistorical onset、200kでのrange余裕、複数seed/qualityは未解決。条件付き0.5も任意に大きいactivationへの保証ではなく、例えばcentered値512ならhalf-squareは65,536となる。監視対象の不変条件が破れたら停止し、さらに多段のscaling等を無検証で追加しない。

## G. Provenance / evidence protection

公開aggregateは `scripts/export_public_g1_numerical_forensics.ps1` のallow-list exportから作る。`docs/results/g1-long-linear-numerical-forensics/aggregates.json` に各conditionのexact source SHA、checkpoint/hash、scope、数値aggregateを保存する。raw checkpoints/tensors、cache本文、token IDs、device/run identities、APK hashes、ログはcommitしない。

private evidence baseは `build/reports/g1-long-linear-forensics/`。`public-index.json` が公開labelから独立run/report directoryへの対応を持つ。各 `forensic-plan.json`、runner.log、result、health CSV、raw floats、decode/compare reportにsource/build/device/runtime identity、LR、optimizer、diagnostics、cursor、health/failureを保存した。公開docにはrun IDsとローカル絶対pathを記載しない。

| identity | SHA / value |
| --- | --- |
| protected40k checkpoint | 89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d |
| diagnostic41646 finite serialized state | def39c33ba33aa66978588b0ffa61d05d08f29574a3442e522ff43fc5d0db654 |
| unscaled41000 checkpoint, original/capture一致 | afaf63968fdaa15ac90c93fc411c1e646ab4c45cbb41a45b34b87c72f9b10bbe |
| scale0.5 41000 checkpoint | 5b0cd59306d2dd2c05909c568c2759df7ad3af983f74a9d28011b41936608a69 |
| scale0.5 42000 checkpoint | ff0f6e2cd1c106006374e73d213a251715ffc01169713df775b9fa6f4468e525 |
| conditional scale 42000 checkpoint | 3fd11199db8ecfeb988fad07d01491ab81a20b62dd1646b99a7c93b066b89a51 |
| expanded cache SHA256 | bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf |
| tokenizer SHA256 | 9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798 |
| cache manifest SHA256 | e8a77e06abab3da5ff7fe26eb9744c3bffad8cf5fbf006bbf4c7f582393961e4 |
| canonical dataset content | fnv1a64:245228bcbe304f51 |
| extended training order, seed20260806 | fnv1a64:83f9c96cf10261a0 |
| QAIRT / runtime Build ID | 2.48.40.260702 / 2.48.40.260702151143 |
| DSP tools | Hexagon SDK6.6.0.0 / tools19.0.07、original planのmanifest/compiler hashes一致 |
| preserved100k source | a3e3ce3339914d00a59673af29357f5ce180f209 |

条件付き候補のnative実験sourceは`8a6350203a6cb214d6d5ca9a288e5a344a0b46be`。公開exportとexplicit SDK argument化は後続tooling commitであり、実験を別sourceで実施したことに書き換えていない。conditionごとのsource SHAはaggregateに記載する。

再現入口は`run_g1_numerical_forensics.ps1`。SDK rootsはprivate環境で確認した値を明示して渡し、自動fallbackしない。例（`$protected40k`とSDK variablesはprivate provenanceで解決する）:

```powershell
$pinned = @{ QairtSdkRoot=$qairtRoot; ExpectedBuildId='2.48.40.260702151143'; HexagonSdkRoot=$hexagonRoot }
.\scripts\run_g1_numerical_forensics.ps1 @pinned -CheckpointPath $protected40k `
  -CheckpointSha256 '89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d' `
  -FromStep 40000 -ToStep 42000 -TapScope ALL_LAYER_NORM_SQUARES -CenteredScale adaptive -Install
```

元conditionの再現は`-CenteredScale 1`、固定失敗probeはprivate indexの41,646 snapshot/hash、`-FromStep 41646 -ToStep 41647 -ProbeBatch 1`を使用する。UNSTABLE run自体をresumeしない。

最終stateは調査branchに実装・このレポート・allow-list aggregateのみをcommitしたclean working tree。commit自体のSHAはGit履歴、commit後のstate/保護identity確認はprivateの`final-protected-identities.json`で解決する。元branch/plan/checkpoint/cacheを保護し、host20GiB/device12GiB gateを維持した。既存evidence削除、data clear、reboot、force stop、SDK/firmware変更、push、PR、main merge、production200k再開は行っていない。
