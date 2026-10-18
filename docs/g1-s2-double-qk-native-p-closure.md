# G1 S2 Double QKᵀ — Native Forward P Precision Closure

S2 LN1→Mixed QKV→Lossless Heads→Double QKᵀ→既存Custom Scale→Mask→Softmax Pは、
固定fixtureについて旧P合同契約を全4 observer条件・3独立processでPASSした。
Production builderには接続していない。新Forward Pを使うDSP Attention backward、
full1-layer、19-layer trainingは未認定。40k→60k／200kは引き続き**NO-GO**。

## A. Source / baseline / 不変の契約

研究baseは`b2ce1272a9ab5575b06f5a422c7042063306560d`、専用branchは
`investigation/g1-s2-double-qk-native-p-closure`。新branchはlocal commitまでで、push、
PR、merge、main変更、history rewriteは行っていない。以前のMixed branchへのpush許可は
今回へ転用していない。

正本のgeneralized training builderでは、T32/D64/H2のQ/K/Vは別parameter registryを
持つNN64 projection。Head0はchannels0–31、Head1は32–63。各headのQ/K `[32,32]`から
`S[r,c]=Σ(q=0..31) Q[r,q] K[c,q]`を作る。transpose0=false、transpose1=true。
診断rank4 `[1,1,32,32]`はunit leading axesとrow-major物理layoutを維持する。

新Raw QKの契約は次のとおり。入力と保存outputはFP32であり、内部だけbinary64。

```text
acc = +0.0 (binary64)
for q = 0..31:
    product = round64(exact_double(Q[r,q]) * exact_double(K[c,q]))
    acc = round64(acc + product)
S[r,c] = round32(acc)
```

Scaleはこのkernelに融合しない。Sourceの`1.0f / sqrt(float(64/2))`のbitsは
`0x3e3504f3`。Maskはcolumn≤rowで`+0`、futureで`-10000` (`0xc61c4000`)をAddする。
各head528 allowed／496 future。Softmaxはproduction rank2 axis1、診断rank4 axis3、
beta1。Maskの-inf化、clipping、追加normalization、epsilon変更はない。

実機演算前の事前登録は`double_qk_native_p_plan.json`、commit `7f7fe9b`。
Raw local gate .01、同一FP32 operandに対する独立double NT32最終FP32 words一致を固定。
上流LN saved state .002、LN local backward .02、QKV .002、head words0、Scale .01、
Mask .001、Softmax local 1e-6、旧P合同契約は一切変更していない。

Referenceを区別した。

| Reference | 用途 |
|---|---|
| A: independent `one::mm` double NT32→FP32 | 新QKのwords／local double accuracy |
| B: ordered FP32 NT32 | 既存FP32 QK control。新doubleとのword差は想定内 |
| C: Host S2+C6、FP32 boundaries | 元のSoftmax modelと明示FP32-shift modelを別評価 |
| D: ideal LN1-origin double Forward | 累積Pと感度評価。Source float epsilonの正確な昇格とlegacy double epsilonを別保持 |

参照計算は新DSP arithmetic helperを呼ばない。元C0のlegacy合同契約9 FAILも新規再現した。
Source-exact oracleとlegacy oracleの両方で新candidateがPASSしたため、oracle変更で旧FAILを
消した結果ではない。

モデルcaseは元44＋既存holdout16。Standaloneには既存Mixed arithmetic controls8＋
新固定head controls8を加え、計76 cases。追加seedは`943401+control_index*1009`
（control_index=0..7、global ordinal68..75）。asymmetry、cancellation、signed zero、
subnormal、large/small、transpose sensitivity、one-hot、near-equalを含む。
Stress16はword／finite試験であり、P合同契約のfixtureには混入させていない。

## B. 新Double QK / ABI / compiler / runtime

独立`HexaDoubleQk` packageに`ForwardRawQkDouble`と`ForwardRawQkDoubleNoReceipt`を追加。
入力順Q0,Q1,K0,K1、出力順raw0,raw1。旧HexaFp32 51、SelectiveLn1 4、MixedQkv 2の
name/order/ABI/receipt/numerical source/binaryは不変。4 packageのHost/DSP計8登録を
context作成前に行った。Production builder/default backendには接続していない。

Observed markerは`0x48383401`、独立counter。receiptは6176 INT32 words:
header32＋実input4096＋実output2048。geometry、alignment、completion、math／receipt
windowを含む。NoReceiptは自身と全上流のINT receipt tensorを除去する。Host preparation
は算術をfatalで拒否し、Host oracleと分離。正しいshape、4-byte alignment、有限入力、
input/output・output同士・receipt overlap拒否、complete writesを検査する。

Build sourceは`d3298ec9e79e5948a17a755450573938cd542512`、実機sourceは
`099baa5bc9941674312be86c488d0039b864c326`。
最終cohort中にsource／binary／fixtureを変更していない。
初回build helperはPowerShell予約変数`Host`との衝突でDSP compile前に停止した。
`hostSuite`へ修正し、Host-tested source hashesを照合して再build。数学の修正ではない。

固定QAIRT2.48.40.260702151143、QNN API2.37.0、V81、Hexagon SDK6.6.0.0／tools19.0.07、
NDK26.2.11394342。flagsは`-O2 -G0 -fPIC -ffp-contract=off -fno-fast-math
-fno-vectorize -fno-slp-vectorize -fno-unroll-loops`。hidden linkageでprovider/qhpi_initを
明示exportし、旧namespace/counterと分離した。

ASM/IR/objectを監査し、`convert_sf2df`、`dfmpyfix/dfmpyll/dfmpylh/dfmpyhh`、serial
`dfadd`、最後の`convert_df2sf`を確認。software pipelineのprologue/30-iteration loop/
epilogueは32個の積をq昇順で加算し、複数accumulator/tree/FMA/FP32 intermediateはない。
inner accumulatorのstack spillはなく、48-byte NT frameはcallee-save用。
Observed wrapper120／NoReceipt96 bytes、shape helper56 bytes。callee/runtime/workspaceを
含む実stack peakとは区別する。new explicit scratch0。新double libm importはなく、
QHPI、memory/string、frame supportのみ。実機registration/linkage/execute/teardownは成功。
HVX resource予約を新QKのvector演算証拠とは扱わない。今回のNTはScalar doubleである。

## C. Standalone結果

独立Host fixture生成3 processのbinary hashは一致。155,648 QK output wordsが独立
double参照と一致した。旧ordered FP32とのword差は103,969／155,648で、異なる数学契約に
対応する。Host kernel test152 callsではguards、poison、complete writes、入力不変、
invalid shape／overlap／alignment／nonfinite拒否を確認。CPU arithmetic拒否2/2、
invalid preparation signature拒否2/2。Host-only testをDSP実行の証拠には転用していない。

実DSPでは76 cases×4条件を各final processで実行。両headとも同一operandの各oracleと
全word一致し、signed-zero/subnormal controlsもPASS。Stress absolute errorはULP依存で
あるため、モデルcore .01 gateと同一視しない。

| Core44、同一FP32 Q/Kに対するlocal double maxAbs | H0 | H1 | Gate |
|---|---:|---:|---:|
| 既存ordered FP32 QK | 1.1120751310e-5 | 2.3500583737e-5 | .01 |
| 新double NT32→FP32 | 3.2811681763e-6 | 3.8046493529e-6 | .01 |

Holdout16の新double local maxはH0 9.1287525805e-7、H1 6.6328221493e-7。
旧FP32／新doubleはいずれもそれぞれのword契約を満たした。

## D. Native QK / Local Forward stages

同一graphで実S2 Y→Mixed Q/K/V→lossless Heads→新double QKを接続。Scale、Mask、
Softmaxまでの途中にHost正解tensorを再注入していない。X/gamma/beta/dY/weightsは正しい
frozen external inputsであり、実19-layer上流producerが接続済みという意味ではない。

ObservedはS2 Y→Mixed input、Q/K/V→Heads、Heads→QK、raw→Scale、scaled→Mask、
masked→Softmaxを実receiptで全word照合した。S2 xhat/inverse→旧LN backward controlも
同じNATIVE IDsで保持。NoTapは内部Y/xhat/inverse/QKV/QK heads/raw/scaled/maskedをNATIVE
にし、NoReceiptは全上流INT観測出力を除去。各graph自身の最終raw/Pとlocal LN gradients
を評価し、valid final boundariesでobserver間words一致。不可視中間値を直接測ったとは
主張しない。

20 graph configurations（Standalone FP32/double、native FP32/double raw、native double
P、各4 observer条件）。final processごとに14,356局所数値判定、1328 case executions。
Standalone76、native60 cases。API、DSP marker、finite、input mutation0、complete writes、
poison/canary、no arithmetic CPU fallback、context/device/backend teardownがPASS。

ObservedでDSP値とのword identityを確認したfrozen stageを、独立Host double R0と比較した。
追加Host attributionは120 headsで理想P reference mismatch0。以下はcore44の最大値。

| Stage | 受信operandに対するlocal error | LN1-origin cumulative error |
|---|---:|---:|
| S2 Y | 既存S2 gate内・Host words0 | 4.0199117857e-7 |
| Q | local gate内・Host words0 | 6.9925630575e-7 |
| K | local gate内・Host words0 | 6.2663098443e-7 |
| V | ordered words0 | 5.5597568434e-6 |
| Raw double QK | 3.8046493529e-6 | 1.7507003676e-5 |
| Scale | 4.7676826398e-7、FP32 words0 | 3.2983863942e-6 |
| Mask | .00048828125、FP32 words0 | .0004883462552 |
| 正しい受信FP32 masked scoresへのdouble Softmax | — | 4.2982627069e-7 |
| 実DSP Softmax P | local max2.1165226771e-7 | 4.2364829078e-7 |

Maskの大きなabsolute errorはfutureの-10000 Add丸めであり、allowed位置のcumulative maxは
2.9981631933e-6。future Pはunderflowで0となり、このMask maxがそのままP errorへ伝播する
わけではない。これらはcounterfactual／局所比較であり、maximaの単純加算で独立寄与へ
一意分解してはいない。NoTap内部値に対する直接観測の主張でもない。

Softmaxの受信exponentからmax/shift/sum/divideを独立replayしたwordsは0 mismatch。
Observed 120 records（60 cases×Tap/NoTap）のexp vs double maxAbsは4.9265108881e-8、
probe内CPU exp oracleとのword差は6982。PCの元Host P modelとは11,379 words、明示
FP32-shift modelとは4749 wordsが異なる（1 variant、60×2×1024 outputs）。
Host libmとのP bit identityを合格条件にはしていない。局所double accuracyと累積契約を
別々に満たした。NoReceiptの内部expを観測したという主張はない。

## E. 累積P合同契約 / 感度

CoreのP/row判定は44×2=88単位、gradient判定は44×2×25=2200 Cartesian pairs。
同じP errorを25重複FAILとして集計していない。Holdoutは16×2=32と256 pairsを別集計。
4 observer条件×source/legacy oracle×core/holdoutの全16集計でfailed predicates0。

| Core source oracle、全observer共通 | 実測 | 旧requirement |
|---|---:|---:|
| P element maxAbs | 4.23648290781564e-7 | ≤1e-6 |
| 最大P RMS | 3.96702158106254e-8 | 保存metric |
| Double row-sum error | 2.27242708206177e-7 | ≤2e-6 |
| P-induced dScores componentwise bound | 2.45909971453266e-4 | ≤2.5e-4 |
| P-induced dValue bound | 2.78552437232082e-5 | ≤.001 |
| Signed P-only dScores perturbationのmaxAbs | 9.95591098131854e-5 | boundと別metric |
| Signed P-only dValue perturbationのmaxAbs | 1.24327776909183e-5 | boundと別metric |
| Host-conditioned actual dScores maxAbs | 1.00788566470555e-4 | ≤.002 |
| Host-conditioned actual dValue maxAbs | 1.38506313405173e-5 | ≤.01 |
| Finite/nonnegative/causal invariants | PASS、future P=0 | 必須 |

勾配は**HOST_CONDITIONED_ON_DEVICE_P**。新Forward Pを消費するAttention DSP backwardは
**DSP_BACKWARD_NOT_EXECUTED**。独立した旧backward回帰とS2 local LN backward controlの
実行とは区別する。正しいfrozen dP/dContextを縮小・再scaleしていない。

`a=ΣdP·P, t=ΣdP·e`として全componentで
`ΔdS_i=e_i(dP_i-a)-(P_i+e_i)t`と実差が一致し、
`|e_i||dP_i-a|+|P_i+e_i|Σ|dP·e|`が実signed差を下回らないことを確認。
dValueも`eᵀdContext`のsigned差と絶対値和の上界を独立確認した。

新worst PはForward16/H0、coordinate884（row27/column20）。P≈.3818076、top-2 logit
gap .4292736053。誤差の符号付き分離は、上流FP32 masked input由来+4.1415898527e-7、
Softmax local +9.4893055080e-9、合計+4.2364829078e-7。この点では元Host modelと同じP word。

新worst dScores boundはForward21/H1/Gradient9、coordinate99（row3/column3、P≈.7096522）。
term1=6.9877772987e-6、term2=2.3892219415e-4。
signed t=-4.7036882808e-5に対しΣ|dP·e|=3.3667506288e-4で相殺があり、同coordinateの
実signed perturbationは+4.0367603980e-5。保守上界と実誤差は異なる。
残budgetは4.0900285467e-6、約1.636%。worst Host actual dScoresは23/H0/G9、dValueは
18/H1/G9。旧critical6 forward/head pairs×G9/G12もprivateで詳細保存した。

## F. Host予測とDSP差 / 失敗の扱い

元Host S2+C6はP max4.2364829078e-7、dScores bound2.4590997145e-4で再現PASS。
明示FP32-shift Host modelもPASS。DSPでは上記二つのmaxが同じで、dValue boundは
Host2.7840368516e-5から2.7855243723e-5へ変化したが契約内。

Hostとの最初のword差はSoftmax側であり、raw／Scale／Maskは同一operandのFP32 oracleと
全word一致した。FP32 shift boundaryとexpf/libm差を、上流誤差と混同しない。
正しいFP32 masked inputへの理想double SoftmaxでもR0比4.2982627069e-7の誤差があり、
実DSP local roundingは位置によって増幅も相殺もする。新P max点は上流誤差が支配する。
「Softmaxだけが悪い」「局所PASSなら累積PASS」とは結論していない。

新candidateのmath／native／P budget FAILは発生しなかった。C7 fallbackやP contract V2は
不要であり、実装していない。旧C0はP max1.0075062425e-6、dScores bound4.6422760347e-4、
元9 failed predicatesのままFAIL。新candidate PASSで旧結果を上書きしていない。

## G. Performance / memory / logical work

正確性PASS後、fixture0・1 warm-up除外＋31 samples。表は3 processの統計値の範囲、ms。
Firstはそのgraphの初回executeでありdevice cold startではない。全variantのp10/p90、
first、finalize、math／receipt windowsはpublic aggregateに保存した。

| Standalone Observed+Tap、同一bundle | QK math median | Receipt window median | Execute median |
|---|---:|---:|---:|
| 既存ordered FP32 | 1.977–1.979 | 2.136–2.139 | 5.239–5.317 |
| 新double NT32 | 2.038–2.041 | 2.196–2.200 | 5.276–5.369 |

新QKの演算窓は約3%増（約59–64µs）であり、速度改善成果ではない。receipt headerは
旧8／新32 wordsで、windowは検査・payload copyを含む各ABIのkernel内計測。API転送／
全instrumentation費用とは区別する。

| NoReceipt+NoTap graph | Execute median | First | Finalize |
|---|---:|---:|---:|
| FP32 QK Standalone | 5.136–5.196 | 5.425–5.716 | 46.471–52.606 |
| Double QK Standalone | 5.177–5.337 | 5.422–5.713 | 46.152–52.203 |
| S2→Mixed→Heads→FP32 QK | 16.567–16.679 | 17.008–17.306 | 45.097–51.645 |
| S2→Mixed→Heads→Double QK | 16.705–16.741 | 17.192–17.833 | 57.130–68.954 |
| Full S2→P | 17.784–17.825 | 18.439–18.660 | 43.459–53.070 |

Full S2→Pのp10は17.586–17.658、p90は18.204–18.396ms。native graphのwallにはlocal
LN backward controlを含む。純Forwardのみの7-node graphは測っていない。NoReceipt内部
math timersはnullであり0msではない。wallにはQNN/API・共通logging等の費用が含まれ、
BACKGROUND_CORRECTNESSの測定である。exclusive schedulingやproduction throughputを
保証しない。以前の別binaryの時刻からspeedupを算出していない。

fresh別graph controlではS2 arithmetic394µs／receipt874µs（31 samples）、Mixed math8335µs
（Q/K8229、HVX V106）、Scale control math中央値25µs／window126µs（既存9-rep policy）、
正しいmasked scoresへのCustom Softmax case0 receiptでmath685µs／window705µsを確認。
これらは異なるgraph/input/観測条件でありfull native Pのwallへ足し合わせない。
Mask単体arith timerとNoReceipt graph内の厳密なnode別時間分解は未測定。

| NoReceipt+NoTap metadata | Nodes | Tensors | Native logical bytes | Host buffer lower bound |
|---|---:|---:|---:|---:|
| QK Standalone | 1 | 6 | 0 | 24,576 |
| Native raw QK（LN backward control含む） | 5 | 25 | 57,472 | 91,136 |
| Native P（同control含む） | 8 | 33 | 82,048 | 91,136 |

宣言量であり実DSP physical allocationではない。fixture/oracle/guards、code、transport、
QNN workspace、複数graph保持を除く。new scratch0、既存S2 center512 bytes、Softmax row
exponent128 bytesは別。STATIC mask4096／scale4 bytes等もlogical native量と区別する。

Q/K projection262,144＋raw QK65,536=327,680 double product terms/layer/microbatch。
19×8で49,807,360 terms、別途HVX V19,922,944 FP32 terms。積数はinstruction数／wallでは
ない。Mixed scalar double Q/Kの約8.2msが引き続き大きな費用で、HVX resource予約中の
scalar work、wider scheduling、saved-state allocationは将来評価が必要。wall単純倍算による
production予測はしていない。

## H. Validation / safety / 再現性

Fast PASS（最終log identityはaggregate）、Host PASS308.9s、Qnn PASS336.0s。
QnnはHost278.1s、pinned QAIRT/core確認25.8s、APK build29.2s、APK audit2.9sを含む。
optional SDK inventory advisoryをcore欠落と混同していない。新Host suite、20 graph契約、
prep/kernel guards、独立stage attribution、saved-P Host construction audit、exporter検査が
PASS。追加Host-only sourceは実行済みprivate source snapshotとbyte一致し、DSP candidateを
変更していない。

追加Host分析は次で再実行できる（既存private evidenceを上書きしない出力先を使う）。
通常のHost suite入口は`scripts/test_double_qk_host.ps1`である。

```powershell
$sdk = $env:PHONELM_QAIRT_SDK_ROOT # 監査済み2.48.40.260702151143のrootを事前設定
if (!$sdk) { throw 'pinned QAIRT root required' }
$out = 'build/reports/double-qk-analysis-reproduction'
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
$support = @('app/src/main/cpp/tiny_language_model_cpu.cpp',
             'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp')
foreach ($name in @('saved_p_native_ownership_test', 'double_qk_stage_attribution_test')) {
    g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$sdk/include/QNN" `
        "host_tests/fp32_custom/$name.cpp" @support -o "$out/$name.exe"
    if ($LASTEXITCODE) { throw "compile failed: $name" }
}
& "$out/saved_p_native_ownership_test.exe"
& "$out/double_qk_stage_attribution_test.exe" `
    build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin `
    build/fp32-double-qk/double-inputs.bin build/fp32-mixed-qkv/mixed-inputs.bin
```

旧Custom Host24 binariesをhash照合してfresh実行し、23 exit0とC0 expected exit4を確認。
元9 FAILを説明するattribution log SHA256も
`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7`のまま。

| 新4-package bundleでのfresh実機回帰 | Result |
|---|---|
| S2 saved state／local backward | PASS、2640 predicates |
| Mixed QKV／HVX V／Heads | PASS、22,632 predicates／words0 |
| Scalar/HVX FFN2 K128／QKV K64 | PASS、word契約維持 |
| Scalar forward QK／LN2→Softmax backward→RawDQDK→Scale | PASS、既存gate／words維持 |
| Forward Scale | Custom PASS、standard H1 1/44 FAIL（max.0102382000946）再現 |
| Mask | Custom PASS、standard176 required predicate FAIL再現 |
| Softmax | Custom local PASS、standard224 local FAIL／nonfinite0、expected exit4 |
| Old C0 P chain | local0/upstream0/words0、cumulative36 FAIL、expected exit4 |
| Device smoke | Fresh HTP/context/API2.37.0、SDK build ID一致、fallback false |
| Muon W8 | Fresh114/114、maxAbs1.490116119e-8、relative L2 4.40269854e-8、cosine1、RPC/finite PASS、fallback false |

旧回帰のderivative probeはinitializerだけを変え、4 packageを登録。数学・graph・判定は
旧sourceのまま。smoke/W8は監査済み既存APKを明示選択し、SkipBuild/SkipInstallで新規実行。
APKは新standalone Double QKをloadしない。W8は独立production-path回帰であり、新Opは
別probeで実行した。過去114/114を今回の結果へ転用していない。

Raw pilot、P pilotはfinal cohortへ数えていない。同一最終binaryでfinal-1/2/3を実行し、
各14,356局所判定／1328 case executions／全P合同契約がPASS。numerical digestsは全3で一致。

| Digest | SHA256 |
|---|---|
| Ordered local numerical records | `d8e507289b602c7cec0a5af898e23b0f5b98b3cbeb74d4bd97b07994dbf8edee` |
| P合同／感度のordered Host assessment | `456ad468cb9cb68303261535420999408d2f1fe60443a0e78886473c19c432ae` |

Timing/counters/device IDsはdigestに含まない。process間再現性とobserver間word一致は別に検査。
SDK/API成功と有限性、poison/canary、入力不変、書込み完了、DSP marker、teardownも別確認。

2 pilots＋3 cohort＋10 regressions＋2 headlessの各投入前にread-only safetyを確認し、投入直前
にもawake/top／競合を再確認。全gate PASS。cohort+回帰28 monitoring samplesは33–34°C、
thermal status0、battery90%、external power true。headlessも34°C/status0/90%。重大保護は
維持し、通常のbattery/temperature/powerは明示research monitoring policyで監視。純11-case
policy self-testもPASS。default project policyは変更していない。

No wake/unlock、force-stop、foreground takeover、root、SDK/firmware/clock/power変更、
無許可APK入替え、endpoint推測。Host/device容量gateとprotected hashesも各preflightでPASS。
記録済みprotected5 artifactsと以前のrefsをcloseoutで再確認し、40k SHA256
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`を維持した。

## I. Architecture / 次のfrontier / Final questions

次は**実Forward saved P→既存Softmax backward／dValue consumerのnative接続**を研究する。
Pの数学的前提とobserver-free累積精度は今回の固定条件で成立した。SourceのPは通常NATIVE
（diagnostic APP_READ条件あり）で、PV、Softmax backward、dValueに同じhead probability IDを
使う。最後のconsumerまで保持し、Q/K/V headsと必要Gate stateも各backward consumerまで
保持する。raw/scaled/maskedは再計算可能なforward中間値であり、Pの保存を代替しない。

Host-only graph auditはNATIVE Pの同じIDをSoftmax backwardとtranspose0=trueのdValueへ
配線し、APP_READ→Host再注入やreceipt無しでshape/ownershipを確認した。これは新PのDSP
Backward実行でもaccuracy certificationでもない。次研究では正しいfrozen dP/dContextから
実consumerを実行し、その後既存DSP upstreamと結合した累積誤差を別評価する。

| Correctness level | 今回の範囲 |
|---|---|
| Local double NT32 | VERIFIED、同一FP32 heads／独立oracle words |
| Native connected S2→P | VERIFIED、reference-fed external X/parameters |
| Observer-free final P | VERIFIED、各NoReceipt/NoTap自身の旧合同契約 |
| Actual full-G1 upstream-fed X／training state | UNVERIFIED |
| New saved P→DSP Attention backward | NOT EXECUTED、Host構築auditのみ |
| Full Attention／1-layer／19-layer training | UNVERIFIED |

PV、Gate projection/Sigmoid、context/gating/scatter/Wo、全residual/FFN producer、Backward
head scatter/projection gradients、V/Gate gradient branches、LN1 dY全aggregation、gradient
materialization／8-microbatch accumulation／optimizer整合性は今回のP成功から認定しない。
狭いS2 budgetのため将来の独立fixtureでも事前登録した回帰を維持し、FAIL時にthresholdや
scaleを変えない。C7等の追加修復はその場合の別研究である。現時点では精度closureを
HVX高速化へ置き換えるより、saved P ownershipと実consumerの精度接続を優先する。

| # | Final answer |
|---|---|
| 1 | Double NT32をDSP実装・Standalone認定した。 |
| 2 | Nativeでは実Mixed Q/K producerから受信した。Host head再注入なし。 |
| 3 | Head0/1、channel mapping、全入力wordが一致。 |
| 4 | 同一operandの独立Host double NT32→FP32 wordsと完全一致。 |
| 5 | 旧FP32 oracleとは103,969/155,648 words差。H1 local errorは2.3501e-5→3.8046e-6。 |
| 6 | runtime/linkage/rounding問題は固定cohort/controlsで観測なし。普遍IEEE証明ではない。 |
| 7 | Standalone local .01 gate／words PASS。 |
| 8 | S2→Mixed→Heads→Double QK native PASS。 |
| 9 | raw→Scaleの実receipt words／scalar bits一致。 |
| 10 | scaled→Maskとsource mask bits／indexing一致。 |
| 11 | Softmaxは実Mask producer outputを受信し、ordered replay一致。 |
| 12 | Cumulative P maxAbs4.23648290781564e-7。 |
| 13 | Row-sum error2.27242708206177e-7。 |
| 14 | Finite/nonnegative/causal、future P=0が成立。 |
| 15 | dScores bound2.45909971453266e-4≤.00025。 |
| 16 | dValue bound2.78552437232082e-5≤.001。 |
| 17 | Host-conditioned actual dS1.0078857e-4／dV1.3850631e-5で既存gate内。DSP backwardではない。 |
| 18 | 全44×2×25を旧合同契約でPASS。全4 observer・両oracle、holdoutもPASS。 |
| 19 | P:16/H0、bound:21/H1/G9、actual dS:23/H0/G9、actual dV:18/H1/G9。 |
| 20 | 元C0の9 FAILを不変で新規再現した。 |
| 21 | NoReceipt+NoTap自身のP合同契約PASS、final words一致。 |
| 22 | 全3独立processのlocal/P assessment digestが一致。 |
| 23 | Double QK math median2.038–2.041ms。 |
| 24 | 旧FP32 1.977–1.979ms比約3%増。 |
| 25 | S2→P wall17.784–17.825ms、local LN backward controlを含む8-node診断graph。 |
| 26 | Fresh HVX FFN2/QKV回帰PASS。 |
| 27 | S2/Mixed QK、HVX V、saved-state word契約を新bundleで維持。 |
| 28 | Fresh Muon W8 114/114 PASS。新Op実行とは別の回帰。 |
| 29 | research battery/thermal/power monitoringを遵守。実測はcharging／status0。 |
| 30 | identity/awake/top/active-run/SDK/APK/protected/critical guardsを維持。 |
| 31 | 次研究としてsaved P→実Backward native接続へ進める。今回は未実行。 |
| 32 | 新P FAILは無し。今後FAILならsource-word差と局所／累積を分けてから修復先を決める。 |
| 33 | 上記の未接続Forward/BW/gradient/materialization/accumulation領域と実上流入力の証拠。 |
| 34 | 40k→60k／200kは**NO-GO**。 |

## J. Provenance / publication

| Frozen candidate | SHA256 |
|---|---|
| Double DSP | `962a98c6f30e4e8f610abba22f69cd2f08fb216d5ed111586b45d7c73f8991e7` |
| Host preparation | `e2df381e1682ab12e68784ee22ddc468fe553d5cf12c778a167c75990a83ea0a` |
| Probe | `911d66e8b0fc3ba56afd544b5c588f3ef4f0027ee08f92a96b809ee4d44b56f6` |
| New frozen fixture | `f577f12b24a7ec21c3ffd6935f2683385725daa0c87ea039f71207e2791d1ed2` |
| Existing Mixed fixture | `8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb` |
| Existing S2 fixture | `492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02` |

Original one-layer input `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`、
LN parameters `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`も不変。
新fixtureの39 role hashes、独立generator source hashes、derivative probes、compiler hashes、
Host／process／regression identitiesは[allow-list summary](results/g1-s2-double-qk-native-p-closure-summary.json)
に保存した。Raw tensor、checkpoint/corpus、APK hash、端末ID/endpoint、SDK binaries、
private logs、absolute local pathsは公開していない。

Private evidenceは`build/reports/g1-s2-double-qk-native-p-closure/`。最終HEADとclean treeは
最後のcommit後にprivate closeoutへ記録する。cohort後の追加はHost-only分析／evidenceで、
numerical candidate binaryは変更していない。新branchはpush未許可のためlocalのみ。

分類は`DSP_DOUBLE_QK_STANDALONE_VERIFIED`、`S2_MIXED_DOUBLE_QK_NATIVE_VERIFIED`、
`FORWARD_P_LOCAL_STAGES_VERIFIED`、`FORWARD_P_CUMULATIVE_VERIFIED`、
`FORWARD_P_OBSERVER_FREE_VERIFIED`、`FORWARD_P_SENSITIVITY_VERIFIED`、
`DOUBLE_QK_PERFORMANCE_MEASURED`。固定geometry／fixtures／reference-fed入力境界に限定する。

35/40-node Tap incidentは`REPRODUCED / CAUSE UNRESOLVED`、6031は
`UNRESOLVED / DORMANT / WATCH`のまま。Training update、Final test、LR/optimizer/codec変更、
checkpoint decode/writeは行っていない。
