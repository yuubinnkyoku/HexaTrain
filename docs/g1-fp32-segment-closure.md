# G1 FP32 Segment Closure

3領域とも、指定geometryと正しい固定外部入力の範囲でStandalone / native接続 /
Tap・Receiptなしのaccuracyが成立した。同一最終binaryで各Lane3独立process、
計6489 QNN executes、75396 metricsを検証した。required metrics 61092件は全PASS。
次の標準consumerはBackward score scaleとForward QKᵀ score MatMulでFAILした。
全1-layer／19層のcorrectnessへは昇格しない。40k→60k／200kは引き続き**NO-GO**。

## A. Reuse & Contract Audit


Diagnostic-only/default-off study on `investigation/g1-fp32-segment-closure`,
starting from `2f104551406fab4e511175d7287d0b3b7429de82`. The starting tree
was clean, upstream unset, with 85 local commits unreachable from known remote
refs. No fetch, remote reconstruction, history rewriting, production graph change,
training update, or optimizer/checkpoint ABI change is part of this study.

The source of mathematical truth remains
`app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc`.
All diagnostic matrices are flat row-major with unit leading axes `[1,1,R,C]`.
Row stride is C FP32 words. Rank4 backfill does not introduce a batch/head axis.

| Tensor/producer → consumer | Shape; transpose; reduction | Ownership/lifetime; gate |
| --- | --- | --- |
| saved Q/K → raw gradients | each head 32×32 | external immutable Forward state through Attention backward |
| Softmax dScores → dQraw | 32×32 × K; NN; K32 | native producer to raw consumer; .01 |
| Softmax dScores → dKraw | 32×32 transposed × Q; TN; K32 | native producer to raw consumer; .01 |
| raw gradients → score scale | 32×32 × scalar `float(1/sqrt(32))` | native; next-consumer .01 inherited dQ/dK budget |
| scaled gradient → scatter → projection gradients | head 32×32 → 32×64, selector transpose_in1 | source dependency only; no new arithmetic beyond raw |
| LN1 Y → Q/K/V | 32×64 × three distinct 64×64 weights; NN; K64 | native Y has shared Q/K/V and production gate-projection consumers; .002 |
| Q/K/V → HeadSplit | each 32×64 → two 32×32 | copy `head[r,c]=projection[r,h*32+c]`; all words identical, .002 |
| Qhead/Khead → scores | 32×32 × 32×32; NT; K32 | next standard consumer only; preregistered .01 raw-score diagnostic gate |
| layer17 ReLU A → FFN2 | 32×128 × W2 128×64; NN; K128 | frozen external A/W2; .002 |
| FFN2 + Residual1 → layer18 LN1 X | 32×64 + 32×64 | native intermediate, immutable external residual; .002 |
| X → LN1 Y/xhat/inverse | 32×64; reductions64 | native xhat 32×64/inverse 32×1 retained to local backward; existing .002 fixture gates |
| saved xhat/inverse + gamma + dY → dX/dGamma/dBeta | dX32×64, parameters1×64 | external reference-fed dY; .02 |

External operands are APP_WRITE, final observable outputs APP_READ, internal
edges NATIVE (APP_READ only in Float Tap variants). Parameters remain distinct
APP_WRITE buffers; native activation storage never owns parameter values.
The synchronous graphExecute contract retains all input buffers and native state
until consumers complete, then context teardown releases graphs. Same-kind
concurrent graphs are outside this diagnostic's counter contract.

`closure_contract.h` defines diagnostic ABI v1 names, input/output word counts,
markers and gates. `closure_ops.h` reuses `mmcustom::ordered` for NN/TN/K128;
`headcustom::copyHeads` is the existing HeadSplit copy body, shared without a
second selector kernel. Paired QKVHeads has one invocation/counter, so its three
independent inputs cannot race on the old HeadSplit counter. Existing registered
Op signatures/receipts and QHPI interface1.4 remain unchanged; six new observed/
NoReceipt names extend the diagnostic registry from31 to37. The scalar baseline
keeps separate FP32 product rounding, increasing-q accumulation, FMA off, no
fast-math and disabled DSP compiler vectorization. No vector arithmetic execution
is inferred from the QHPI HVX resource declaration.

Host references use independent `one::mm` double and ordered FP32 equations;
they do not call the Custom dot. Independent LN references and saved-state
oracle remain unchanged. Accuracy, bitwise identity, QNN return status and tensor
finiteness are separate checks. No gate, input scale or mathematical definition
has been relaxed. Native receipt comparisons include every input/output word;
observer-free graphs are judged from their own final outputs.


Sourceではparameter registry由来weights/gamma/betaはAPP_WRITE、one-hot selectorは
STATIC。activationsはminimal/diagnostic設定に応じNATIVE/APP_READとなる。
今回のprobeはその依存をunit-leading-axis rank4へ写し、物理row-major順を維持する。
Q/K/V全projectionの数学はNN、head scoreはNT、dQrawはNN、dKrawはTN。
selector scatterはselectorのtranspose_in1であり、rawとは別の演算である。

## B. dQ/dK Raw

25 frozen/full-upstream＋10 stress cases、2 headsを別々に判定する。
Q/K/dScoresを非対称にし、ordinary、mixed signs、cancellation、large/small、
low variance、fixed random、transpose-sensitive/head-asymmetric値を含めた。
以下は各process共通のmaxAbs。Standaloneのreferenceは実際に供給するFP32
operandから計算する独立double/ordered FP32。native chainはLN2起点の独立double
式との差と、receiptで受信した実operandに対する局所誤差を分ける。

| Target | Standalone /35 | Native fused chain /25 | Received-operand local /25 | gate |
| --- | ---: | ---: | ---: | ---: |
| H0 dQueryRaw | 4.07267048104e-05 | 4.31917280963e-05 | 1.71148026027e-05 | .01 |
| H0 dKeyRaw | 0.000260925255589 | 4.52113789606e-05 | 3.6012176821e-05 | .01 |
| H1 dQueryRaw | 6.15365696603e-05 | 3.97556808309e-05 | 2.08200245737e-05 | .01 |
| H1 dKeyRaw | 0.000239181536926 | 5.21589968514e-05 | 3.39361496628e-05 | .01 |

Receipt付きの全raw outputsは受信operandのordered FP32と全word一致。input head
mapping、Q/K ownership、全出力write、input immutability、finite、poison/canaryを検査。
標準Standalone raw MatMulは最大0.943904876709、
native標準dQraw/dKrawは従来の最大.107461168893/.126280396953を再現した。
標準negative controlsのFAILをCustomのFAILやtransport failureへ混ぜない。

`DSP LN2 F/B → Residual → dContext (+dWo) → HeadSplit → Gate → dProbabilities
→ Softmax → RawDQDK`をnative接続した。separate/fused Softmax ×4 observer variants
の最終raw wordsも全比較で一致し、NoReceipt/no-Tap variant自身の最終raw outputsが
.01を満たす。dScores gate .002、前段LN .02、dContext/dWo .03、Gate/DP .01も維持。
実DSP LN2から生成したgradientであり、途中のdScoresをHost理想値へ置換していない。
LN2 X/dY、residual upstream、Wo/context、Forward saved P/Q/K/V/Gateは正しい
frozen external inputsであり、実HTP Forward Pの精度を認定するものではない。

次のsource consumerはraw × `float(1/sqrt(32))`。既存dQ/dKの.01 budgetを
Device評価前に登録した。標準scaleのmaxAbsはH0 dQ/dK=
0.0127901180026/0.0322076710836、H1=
0.0175945349651/0.0213635282663。
各head/gradientで2/25 FAIL。receiptで見た実rawに対する局所scale誤差も最大約
.032201289564であり、rawの小さな上流誤差だけでは説明できない。
**BLOCKED_ATTENTION_CONSUMER = standard score scale**として停止した。
head scatterとprojection gradientsへは接続せず、新規Custom arithmeticも追加しない。

**DSP_DQDK_RAW_MATMUL_VERIFIED / FP32_ATTENTION_DQDK_RAW_SEGMENT_VERIFIED**。

## C. Q/K/V Head Selection

標準selectorはone-hot `[64,32]`を右から掛ける操作であり、数学は連続32 channelsのcopy。
既存HeadSplitの`memcpy`本体を共有した。QKVHeadsは3 inputs /6 outputsを一つの
launchで処理する。Q/K/V別tensor/parameter identityを維持し、旧HeadSplit counterを
三つの独立nodeから共有する方式は採らない。旧HeadSplit/NoReceipt ABIは不変。

Standalone head selection44 cases ×6 headsはobserved/freeとも全word一致、maxAbs0。
Hostではsigned zeroも区別した。native graphは実DSP LN1 Y→実DSP Q/K/V→HeadSplit。
途中の理想projection注入はない。LN1 X、parameters、local backward dYはreference-fed。

| Native target /44 cases | maxAbs | gate |
| --- | ---: | ---: |
| y | 5.81823561774e-07 | .002 |
| hat | 6.66749372158e-07 | .002 |
| inverse | 1.94379752543e-05 | .002 |
| qkv0 | 2.4316564895e-06 | .002 |
| qkv1 | 2.77017446315e-06 | .002 |
| qkv2 | 5.7436790506e-06 | .002 |
| dx | 1.53640447422e-05 | .02 |
| dg | 1.96494390394e-05 | .02 |
| db | 7.89668411016e-06 | .02 |

各headは対応する実projectionとの全word一致。projection receiptの実YはLNのsaved
xhat/gamma/betaからのFP32式とも照合し、TapありではY/xhat/inverseの直接wordsも
照合する。saved state→Backwardのreceipt一致も保持した。
NoReceipt/no-Tapでは内部Q/K/Vを直接観測したとは主張せず、6 final headsの独立
accuracyと観測版final headsとの全word比較を行い、全44 cases PASS。

標準selector controlはQ/K最大約.00195で既存.002 budget内だがbitwise不一致。
Vは.00580941960462/.00562922005712、各8/44 FAILを再現した。
copy kernelのlossless identityと、この独立accuracy budgetは別の指標である。

次のForward consumerは標準`Qhead × Khead^T`、scale前raw score。
既存generic stage budgetとconsumer診断の単位を使い.01をDevice前に登録した。
H0 maxAbs 0.0446639307669、8/44 FAIL。
H1 maxAbs 0.0478892556466、8/44 FAIL。
receipt付きの実head入力に対する局所double誤差も約.0446649/.0478887でFAIL。
NoReceipt consumer graphも自身のfinal scoreで独立判定する。その内部headに対する
Host予測との差は`NextExpected`と明示し、不可視の実operandを観測した扱いにしない。
**BLOCKED_FORWARD_CONSUMER = standard QKᵀ score MatMul**で停止した。
score scale、mask、Forward Softmax/P、gate projection/Sigmoid、PVは未認定。

**FP32_LN1_QKV_HEAD_SELECTION_VERIFIED**。

## D. FFN2→Residual→LN1

固定8 checkpoint-derived fragmentはlayer17 ReLU A `[32,128]`、W2 `[128,64]`、
Residual1 `[32,64]`からlayer18 LN1へ渡す。既存frozen bytesを再利用した。
追加8 casesはReLU-like/nonnegative、sparse、mixed weights、cancellation、magnitude、
low variance、fixed randomを含む。ordered NN K128 dotは既存templateそのもの。
Standaloneを全caseで合格した後だけnative Addへ、Add合格後だけLN1へ進んだ。

| DSP target | Frozen8 maxAbs | All16 maxAbs | gate |
| --- | ---: | ---: | ---: |
| ffn2 | 7.18314831616e-07 | 2.96602956951e-06 | .002 |
| x | 7.18314831616e-07 | 3.20444814861e-06 | .002 |
| y | 3.00357770433e-07 | 4.61443698363e-07 | .002 |
| hat | 1.28344704198e-06 | 1.41215464877e-06 | .002 |
| inverse | 2.66506694713e-07 | 2.66506694713e-07 | .002 |
| dx | 3.42286708843e-06 | 3.42286708843e-06 | .02 |
| dg | 2.7886468132e-05 | 2.7886468132e-05 | .02 |
| db | 7.89668411016e-06 | 7.89668411016e-06 | .02 |

Standalone FFN2も同じ最大誤差を満たす。DSP FFN2 receiptはA/W2全words、ordered
FP32 output全wordsを検査し、Addの受信FFN2 wordsと一致。residual operandも全words
一致、LNの受信XはAdd output receiptと一致、saved hat/inverseはBackwardまで一致。
input mutation0、finite、API0、complete writes、poison/canary、DSP markersを別々に検査。

| Frozen8 control | FFN2 maxAbs | X maxAbs | ideal LN1 Y | ideal dX |
| --- | ---: | ---: | ---: | ---: |
| Standard FFN2 + standard Add | .00379717476751 | .0051742804804 | .00209740259207 | .0240631979769 |
| Standard FFN2 + DSP Add | .00379717476751 | .00379717476751 | .00138289763701 | .0200067829808 |
| Correct Host FP32 FFN2 + DSP Add | 7.18314831616e-7 | 7.18314831616e-7 | 3.00357770433e-7 | 3.42286708843e-6 |
| Actual DSP FFN2 + DSP Add + DSP LN1 | 7.18314831616e-7 | 7.18314831616e-7 | 3.00357770433e-7 | 3.42286708843e-6 |

最後の行だけが今回のnative DSP FFN2成功であり、Host-correct controlの成功から
読み替えたものではない。旧standard failuresとHost controlを最終3 processでも再実行。
standard FFN2はfrozen8全て、追加stress込み12/16で.002 FAILした。

Observer-free X accuracyはFFN2→Addの2-node graphのfinal Xで測る。
FFN2→Add→LN1＋local Backwardの4-node graphでは、そのvariant自身のfinal Y/dX/
dGamma/dBetaを判定し、不可視Xを直接観測したとは主張しない。4 observer variantsの
final outputsは全words一致だった。Forwardのhat/inverseもobserved/Tapで既存gate内。

X gateは独立double FFN2+residualとの比較。LN reference APIへの理想Xはそのdouble
結果を一度FP32へ丸めて渡す（既存controlと同じboundary）。LN計算自体は独立double。
local dYは正しい外部FP32入力であり、全Attention由来dYの認定ではない。
A/W2/residualもCPU trace由来の正しい外部入力であり、全19層HTP Forwardのproducer-fed
correctnessではない。LN kernelのアルゴリズムやLR/epsilonを変更して誤差を隠していない。

**DSP_FFN2_MATMUL_VERIFIED / FP32_FFN2_RESIDUAL_LN1_SEGMENT_VERIFIED**。

## E. Observer Effect

各新segmentはReceipt/no-receipt × Float Tap/no-Tapを比較した。
Custom領域のfinal raw gradients、6 heads、FFN/Add/LN final outputsは各variant自身の
accuracyを満たす。必要な全word比較は同一process内で直接実施し、process間は
configuration別metric fingerprintsを含むcanonical digestを比較する。
NoReceiptはreceipt tensor/copy/counter writesを除く。共通sourceのtimer readsやHost
loggingまでゼロとは主張しない。NoReceipt timer0は未観測sentinelで、math時間0ではない。
signature/tensor数も変わるため、backend内部loweringまで固定した実験ではない。

35-node/40-node Tap差は前研究の**REPRODUCED / CAUSE UNRESOLVED**を維持する。
今回そのreproducerを新たにbisectionしていない。新Custom領域のobserver-free成功から
旧差の原因解決を推定しない。backend fusion、kernel selection、memory placement、
schedulingのどれが原因かは直接確認できずUNRESOLVED。SDK逆解析・firmware変更なし。

Pilotは最終cohortから除外した。最初のAPK audit呼出しはPowerShell5のLatin1 API不在で
probe投入前に停止し、installed PowerShell7で全auditを実行し直した。B pilotは
NoReceipt final headをHost LN oracle由来の予測wordsと誤ってbitwise比較し停止した。
これはHost↔DSP LNの既知の微小な丸め差を拒否する不適切なassertだった。
修正後はobserved版で実受信projectionとのword比較、free版で自身のaccuracyと観測版
final headsとのword比較を行う。gateや数学は不変。停止pilotにはteardown成功を主張せず、
修正版同一binaryで全9 processをやり直した。W8 buildの一度の失敗もPowerShell引数分割
による呼出し失敗であり、修正したbuild＋実機114/114だけをregression結果とした。

## F. Segment Architecture

Diagnostic segment ABI v1はname/version、数学、shape/layout、parameter identity、
native ownership、saved lifetime、gate、rounding、execution domain、observer variants、
independent reference、failure classと再現cohortを一単位として扱う。QHPI1.4/旧ABIは維持。
各Laneは外部inputの正しさを条件として独立判定し、別LaneのFAILを転用しない。
全てsynchronous graphExecute、context teardownまでgraph所有物を保持する診断である。

`closure_contract.h`はname/count/marker/gateの小さなSSOT。既存ordered dotとHeadSplit
copy本体は一つ。shape/reference indexingは独立検査のため完全にkernelから生成しない。
手書きのregistry names、probe tensor geometry、gate literalsにはまだ重複があり、
kernel/receipt/shape testsとmanifest hashでdriftを検出する。production parameter SSOTや
registryの大改修は行っていない。長いpositional boolean constructorとcompactなprobe行は
将来のAI agentにとって誤設定/reviewのリスクであり、LOCの少なさを保守性の証明にしない。
次の拡張ではこの診断範囲のnamed configurationを検討すべきだが、今回の研究gateを
縮小するための新frameworkや大規模refactorは追加していない。

| Candidate | Separate ownership / potential fusion | Risk / judgment |
| --- | --- | --- |
| Softmax→dQ/dK | dScores8192B、2 heads×2 consumers、paired raw1 launch。fusion候補は中間と1 launch削減 | dK TNがdScoresの転置方向を読む。row streaming/partial accumulatorsでもq順とshared Q/K lifetimeを保つ必要。今回はfusion未実装 |
| LN1→QKV→HeadSplit | Y8192B、QKV24576B、heads24576B、saved hat/inverse8320B。projection出力をhead layoutへ直接storeすればQKV中間と1 launch削減候補 | Yはgate projectionに加え後のdWq/dWk/dWv/dWgでも必要。Y/saved stateを無条件に削除できない |
| FFN2→Residual→LN1 | FFN28192B、X8192B、residual8192B、saved state8320B。FFN2+Addなら中間8192Bと1 launch削減候補、LNまで融合すれば更に1 launch候補 | Xは次層Attention residual branchにも必要で、全G1ではstoreを残す必要。row64 LN順序とsaved stateを維持する必要 |

Fusion候補の優先順位は**FFN2→Residual→LN1**。正確な固定fragmentがあり、単一路径と
row-local LNのownershipが明確。ただしscalar MatMul時間をfusionで解消できるとは
示されていない。今回の数学的native連結PASSと、未実装のfused kernelを区別する。
同種graphのconcurrent executionへ広げる場合はper-instance counter/receipt ownershipの
再設計が必要。現在のpaired launch証拠を19層並列scheduleへ転用しない。

## G. Performance

非exclusive BACKGROUND_CORRECTNESS、fixture0の9反復の先頭を除いた8回のmedianを
processごとに求め、その3 processの範囲を示す。NoReceiptのkernel timeは未観測。
math windowは新kernel内のordered dots/copyのみ、receipt-inclusive windowは入力検査/
copyと出力finite/copyを含む。wallは同期QNN API/Host logging overhead込みで別の量。

| Graph / observed native vs free native | Scalar math ms | Receipt window ms | Execute wall ms |
| --- | ---: | ---: | ---: |
| Standalone paired dQ/dK | 4.074–4.075 | 4.331–4.336 | observed 7.327–7.405 / free 7.229–7.344 |
| LN2→fused Softmax→raw | 4.097–4.098 | 4.346–4.351 | observed 16.918–17.047 / free 19.465–19.579 |
| LN2→separate Softmax→raw | 4.078–4.079 | 4.332–4.335 | observed 17.037–17.623 / free 16.089–16.495 |
| LN1→QKV→heads + saved-state B control | 0.067–0.068 | 0.365–0.368 | observed 17.120–17.252 / free 16.486–16.969 |
| Standalone FFN2 | 8.052–8.418 | 8.407–8.766 | observed 11.453–11.750 / free 11.597–11.670 |
| FFN2→Add | 8.456–8.463 | 8.793–8.807 | observed 12.577–12.723 / free 11.888–12.229 |
| FFN2→Add→LN1 + saved-state B control | 8.422–8.479 | 8.764–8.828 | observed 13.493–13.800 / free 12.658–13.055 |

LN2 chainのmath欄は最後のRawDQDKだけ、FFN2 chainのmath欄はFFN2だけである。
LN1→headsのmath欄は最後のQKVHeads copy windowであり、QKV dot3本を含まない。
過去のQKV約11.2ms、dProb約1.99msは別instrumentationのdiagnostic参考値である。
各段のscalar mathとgraph wallを足し引きして未観測kernel時間を作らない。
fused/separateのwall差にはprocess/graph条件の変動があり、普遍的なfusion speedupを
認定しない。失敗した標準consumer graphの時間を成功方式の性能結果へ含めない。

| Observer-free representative graph | nodes/tensors | logical B | native FP B | Host buffer lower bound B | finalize ms /3 processes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw standalone | 1/10 | 40960 | 0 | 65664 | 35.131–44.679 |
| LN2/fused/raw | 10/43 | 206220 | 73856 | 198880 | 53.112–81.359 |
| LN2/separate/raw | 13/49 | 222860 | 90496 | 198880 | 45.758–92.581 |
| LN1/QKV/heads + local B | 4/23 | 140420 | 41088 | 165664 | 46.705–60.860 |
| FFN2 standalone | 1/3 | 57344 | 0 | 106528 | 51.424–57.114 |
| FFN2/Add | 2/5 | 73728 | 8192 | 122912 | 48.453–67.208 |
| FFN2/Add/LN1 + local B | 4/15 | 107652 | 24704 | 149120 | 26.626–52.864 |

Host buffer lower boundはinput＋immutable snapshot＋APP_READ output/guardsの論理和。
metadata、fixtures、double references、allocator reuse、runtime workspaceは含まない。
No dynamic scratch allocationを新kernel sourceで確認したが、compiler stack frameや
backend scratch/DDR/VTCM allocation/trafficは未測定。graphはprocess内でcontext teardown
まで複数保持されるため、process RSSも単一segmentのphysical allocationではない。

oneClosureA: process peak RSS KiB [111108, 111420, 111424]、thermal status [0]、battery [34.0, 35.0]℃、USB powered ['false']。

oneClosureB: process peak RSS KiB [105136, 105276, 106428]、thermal status [0]、battery [34.0, 35.0]℃、USB powered ['false']。

oneClosureC: process peak RSS KiB [106120, 106140, 105212]、thermal status [0]、battery [34.0, 35.0]℃、USB powered ['false']。

端末clock、scheduler、他app activityは固定していない。受信receiptのDSP V81 markerと
HTP backendのみを実行するprobe codeを確認。CPU上のreference計算は比較用であり、
学習stepの数値演算をHTPで実行する領域を検証した。NPU-only/CPU不使用/自動微分とは
主張しない。NoReceiptにはmarkerがなく、観測版identityとHTP-only登録/実行経路を併用する。

### 19-layer arithmetic/lifetime accounting

T32/D64/F128/H2、microbatch8。MAC=1 multiply＋1 addの2 FLOP。

| Work | layer/MB MAC | L19/MB MAC | L19/8MB MAC |
| --- | ---: | ---: | ---: |
| dQ/dK raw4 dots | 131072 | 2490368 | 19922944 |
| FFN2 K128 | 262144 | 4980736 | 39845888 |
| Previously verified dContext/dWo/dProb/QKV | 720896 | 13697024 | 109576192 |
| Sum of these verified-region candidate shapes | 1114112 | 21168128 | 169345024 |

HeadSplitは6144 FP32 wordsのcopy、数学的MAC0。sourceのone-hot selectorをdense
MatMulとして数えると6×65536 MACだが、backendが実際にどう計算したかは未測定。
Softmax/LN/Add等の非MatMul演算は上表のMACへ混ぜない。上表はgeometry上の仕事量で
あり、19層実producer-fed実行や全training FLOPを実測した表ではない。

Raw unique compulsory input/output40960B、FFN257344B、QKVHeads49152B。
FFN2 weightsは32768B/layer、L19で622592BをMB間共有する。LN1 saved stateは
8320B/layer、L19で158080B、8MB同時保持なら1264640B。FFN2中間削減候補8192Bは
L19で155648B、8MB同時なら1245184B。逐次MBで8倍の同時保持は必須ではない。
QKV materialization24576Bの削減候補はL19で466944B、8同時で3735552Bだが、
Y/saved state/headsの必要lifetimeは別に残る。liveness aliasとbranch/scheduleを含む
実physical memory、同期/DDR/VTCM費用は未測定。未検証Forward/Backwardとoptimizer
storageもこのpayloadに含めない。wallを19/38/304/608倍したproduction throughput予測は
作らない。mathematically required work、verified work、scalar math、diagnostic wallを分ける。

### HVX feasibility / next investment

固定toolsの`hvx_hexagon_protos.h`に`Q6_Vsf_vmpy_VsfVsf`と
`Q6_Vsf_vadd_VsfVsf`があることをread-only確認した。FP32 vector APIの存在はscalarと
同じ丸め/FTZやword identityの実証ではない。output-column parallelizationなら各laneの
qを0→K−1に保ち、独立FP32 multiply→add、FMA contraction offを維持する候補になる。
FFN2 NNはBの各q-rowがcontiguous、N64は32-lane tile2個でtail不要。pointer alignment、
vector stores、register pressure、A broadcast/B reuse、row blocking、instruction resource
contentionを測る必要がある。TN dKもAのstrided q readとB row reuseを設計する。
NT geometryはtranspose/prepack費用を別計測する。q方向のtree reductionやFMA化は
現契約を変えるため採らない。scalarとの差全wordと独立double accuracyを再試験し、
差があれば高速化成功へ昇格しない。32 lanesから32倍speedupを推定しない。

**次の投資先は一つ: C segmentをfixtureにした、q順保持HVX FFN2 MatMulの独立feasibility研究。**
fusion候補自体はCが優先だが、今回8ms級のscalar dotを占める費用への対処を先に
検証する。full fusion/production統合/全1-layer移植を同時に開始しない。

## H. Decision

1. **dQueryRaw/dKeyRaw standaloneは両方PASS**。.01と受信operand ordered FP32 wordsを満たす。
2. **Softmaxからnative接続してもPASS**。fused/separate、4 observer variants、3 processで確認。
3. **次のBackward frontierはstandard score scale**。head scatter/projection gradientsは未検証。
4. **Q/K/V selectorをlossless化できた**。実projectionの全wordsとhead indexingが一致。
5. **LN1→QKV→HeadSplitのobserver-free accuracy成立**。X/parametersはreference-fed。
6. **次のForward frontierはstandard QKᵀ raw score MatMul**。後続mask/scale/P/gateは未認定。
7. **DSP FFN2は.002 gate PASS**。固定8＋追加8、NN K128 ordered dot。
8. **実DSP FFN2→DSP Add→DSP LN1のnative連結成立**。Host正解FFN2注入ではない。
9. **LN1 Y/dXの理想固定fragmentに対する誤差が回復**。saved xhat/inverse/dGamma/dBetaも既存gate内。
10. **reference-fedは何か**: AのLN2 X/dY/残差/Wo/context/saved Forward P/Q/K/V/Gate、BのLN1 X/parameters/local dY、Cの前層ReLU A/W2/Residual1/LN parameters/local dY。全19層HTP producer起点ではない。
11. **次のfusion候補はFFN2→Residual→LN1**。FFN2中間とlaunchを削減する設計候補、実装/速度成功は未認定。
12. **scalar MatMul性能問題は重く扱う**。今回raw約4ms、FFN2約8msで、elementwise/copy fusionだけでは主要math費用を除けない。
13. **HVX/blocking独立研究へ進める価値あり**。CのNN geometryでordered words/double gate維持を先に検証する。
14. **Full1-layerに欠ける演算**: Forward score/scale/mask/Softmax P/PV/gate/Sigmoid/Wo/FFN1と残りresidual精度、Backward score scale/scatter/projection gradients、gate branch、FFN1/FFN2 backward/DReLU、全branch→LN1 dY、全gradient materialization/registry integration。embedding/final logits/lower-layer path、19層producer-fedとscheduleも未認定。
15. **40k→60k/200kはNO-GO**。training更新、Final test、LR/optimizer変更は0。6031はUNRESOLVED / DORMANT / WATCH。

| Independent classification | Status / scope |
| --- | --- |
| DSP_DQDK_RAW_MATMUL_VERIFIED | VERIFIED; standalone NN/TN, T32/H2/Dhead32 |
| FP32_ATTENTION_DQDK_RAW_SEGMENT_VERIFIED | VERIFIED; correct external LN2/Forward saved inputs |
| FP32_LN1_QKV_HEAD_SELECTION_VERIFIED | VERIFIED; reference-fed LN1 X |
| DSP_FFN2_MATMUL_VERIFIED | VERIFIED; fixed8＋stress8 K128 |
| FP32_FFN2_RESIDUAL_LN1_SEGMENT_VERIFIED | VERIFIED; correct fixed external producer fragment and local dY |
| OBSERVER_FREE_PRECISION_VERIFIED | VERIFIED independently for these3 segments' own final outputs |
| BLOCKED_ATTENTION_CONSUMER | Standard score scale, .01 FAIL |
| BLOCKED_FORWARD_CONSUMER | Standard QKᵀ score MatMul, .01 FAIL |
| BLOCKED_UPSTREAM_PRECISION | Full G1 upstream limitations from prior study remain; not a FAIL of C's correct fixed fragment |
| FP32_ARCHITECTURE_REASSESSMENT_REQUIRED | Full-model scalar arithmetic/scheduling/ownership costs remain material |
| UNRESOLVED | 35-node backend mechanism, full1-layer/19-layer correctness, physical memory and production performance |

ONE_LAYER_BACKWARD_VERIFIED/FULL_G1_CORRECTNESS_VERIFIEDは使用しない。

## I. Provenance

Branch `investigation/g1-fp32-segment-closure`。Device-tested source commit **`dbcb9d8875a462e86d064eca1a25f264a5bddba4`**。
Final HEADは本reportを確定したcommit。self-containing hashは本文へ埋め込めないため、
`git log -1 --format=%H -- docs/g1-fp32-segment-closure.md`で取得し、exact hashをprivate
closeoutと完了回答に記録する。最終working tree cleanをcloseoutで確認する。

- libQnnHtp.so: `6e6da5284060ca4369bb6fd3c6f2b661a9cebabb2cf4cf695cd77f558082265f`。
- libHexaFp32Dsp.so: `83632016f46d93ff91fffcf9e3e28f53de9acede6cc46735b987a28188a30b51`。
- libHexaFp32Host.so: `32de6943e7c98d70ecbf0a0569a00889a622609342809e51729bfb449e54f94b`。
- libQnnHtpPrepare.so: `cd31f94ae79ad402312e75262297d1e1073f85927d95f7905357a10b1d32a263`。
- libQnnHtpV81Skel.so: `a0b9750d900a0afbba240e3506ba63ba91ec0e7f1b3a44de9afbb6818a665a32`。
- one_layer_probe: `dc711176ad914847f49c45a8597294d46246cc81a854147a4881e46d3a759e55`。
- libQnnHtpV81Stub.so: `16188365d86ef09a9067e8e04b7fa1c604729d1870219b1bbf04ee904e6c3ffa`。

Compiler SHA256:

- hexagon_compiler_sha256: `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`。
- ndk_compiler_sha256: `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3`。

QAIRT2.48.40.260702151143 / API2.37.0 / HTP V81 / Hex SDK6.6.0.0 /
tools19.0.07 / NDK26.2.11394342。DSP packageは初回build後の同一bytesを再利用し、
ProbeOnlyがpackage source/compiler/SDK/binary hashesを照合して最終probeをbuildした。
最終freeze後の再buildは0。過去package/probe/SDK manifest/private cohortは保持する。
前研究source `5eb5b6bf73e56c33732590e086bcd7b6be72dbb6`、過去DSP package
`fcee5c0052274e7ae5cfaef093542099b84d5d62c57aed41ac18368cc1f16fb1`とも事前照合した。

Frozen fixture SHA256:

- one-layer-inputs.bin: `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`。
- ln1-parameters.bin: `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`。
- ln1-producers.bin: `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`。

| Final lane /3 independent processes | executes/process | metrics/process | required/process | canonical numerical SHA256 |
| --- | ---: | ---: | ---: | --- |
| oneClosureA | 805 | 15220 | 12680 | `1d2afbcde577cf8b29e7321f693e87f9668cecd51fe9aeda35b09172b247aacb` |
| oneClosureB | 1018 | 8888 | 6996 | `6a97e3e0cc65e42a8325534adf5b95212a450eea6cb913a5ae6a4e6ae11dc72c` |
| oneClosureC | 340 | 1024 | 688 | `5e2205e3e1011c882e2a3f4d1e588bc26bdf527a1da641a4c0aec92dd6a9092f` |

Digestはordered onemetric rows（各configurationのoutput fingerprint/error/gate/word差を含む）
のSHA256。timing/counter/paths/device identifiersを除外する。raw tensors自体のSHA256では
なく、異なるconfiguration間の完全同値性の証明でもない。同一configurationの3 process
recordsが全groups/counts/digest一致し、configuration間の必要な全word比較はprobe内で別実施。
API return0、context/device/backend teardown成功、input mutation0、complete writes、
poison/canary、finiteとreceipt identityを検査した。NoReceiptにはreceiptを捏造しない。

Validation:

- Fast PASS、最終numerical source120.0s（最初96.4s）、report/aggregate確定後118.0s。
- Host contract＋diagnostic suite PASS556.3s（Qnn profile内で同じHost suiteを実行）。
- 追加Host: new kernel142 calls、shape3920 checks、既存HeadSplit24、MatMul136＋chain68、QKV/Product136、Softmax210、dProb70 PASS。旧QKV stressのdouble最大.0322703は任意入力のarithmetic identity試験であり、LN-derived .002合格へ転用しない。
- Qnn profile PASS648.2s、pinned core/API/build/APK audit。optional inventory不足のみadvisory、fallbackなし。
- Device smoke PASS125.4s、BACKGROUND_CORRECTNESS、identity/finite/fallback gate。
- Muon W8 **114/114 PASS**、maxAbs1.490116119e-8、worst relative L2 4.40269854e-8、cosine1、RPC success、finite、fallback=false。
- 各Lane3 processのrequired gates、source/compiler/SDK/package/probe/frozen fixture identityをallowlist exporterで照合。
- 公開exporterは最終9 processのpositive検査と66破損条件の拒否を確認。Python optimized modeでも66条件を拒否し、private fieldを出力しないことを確認した。

Safetyは各process前にstable physical device identity、awake、top-resumed、active run不在、
Host20GiB/device12GiB、thermal/battery、APK audit、QNN/Skel/package/probe hashesを確認。
全runはheadless/background、focus takeover0。無断wake、force-stop、settings/root/SELinux/
firmware/SDK変更なし。保護40k checkpoint
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`、
tokenizer/cache/manifest/run-planの5 hashesを開始時・各run・終了時に照合し、200k statusは
UNSTABLEのまま。100k baseline、Step41647/42k forensic/private reportsはwrite対象にせず、
前local branchesは不変。raw evidence/endpoint/device identity/local absolute pathsはignored
build内だけに保持する。新たなtraining/Final test/checkpoint decode/optimizer変更は0。

公開可能aggregateは`scripts/export_public_segment_closure.ps1`→Python allowlist exporterで
作成し、[`g1-fp32-segment-closure-summary.json`](results/g1-fp32-segment-closure-summary.json)
へ保存する。raw tensor/checkpoint/corpus/private logs/SDK binaries/APK hashはcommitしない。
push/PR/merge/main変更/history rewriteは0。
