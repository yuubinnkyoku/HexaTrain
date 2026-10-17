# G1 C7 Selective LN1 / DSP feasibility

> A–Jは当初Host研究終了時の証拠。後続の実装・実機再開については末尾Kを参照する。旧Host数値・FAIL・未実測状態を遡って書き換えない。

## A. Initial evidence

**C6 + LN1 normalizationのみのS2が、旧P合同契約をHostでPASSした。** Full C7をそのまま移植する前に、S2のLN1 standaloneを実装候補とする。これは実DSPの精度・性能認定ではない。

起点は`dc8ffd3cb963eff953429230689ed6ff624a1091`、開始時clean、local未push commits 125。専用branchは`investigation/g1-c7-selective-ln1-dsp-feasibility`。GitHub mainから再構築せず、旧branch refs・protected artifacts・既存DSP source/binariesを保持した。

固定環境はQAIRT `2.48.40.260702151143`、QNN API `2.37.0`、V81、Hexagon SDK `6.6.0.0`、tools `19.0.07`、NDK `26.2.11394342`。新QNN graph、QHPI登録、DSP packageは作成していない。

不変の旧合同契約:

| Predicate | Gate |
|---|---:|
| P element maxAbs | 1e-6 |
| Double row-sum error | 2e-6 |
| P-induced dScores componentwise bound | 0.00025 |
| P-induced dValue bound | 0.001 |
| Actual Host dScores / dValue | 0.002 / 0.01 |
| Causal / finite / nonnegative | 全て成立 |

旧baseline runnerはexpected exit **4**、attribution runnerは9 FAILを説明・再現してexit **0**。R0–R5を含む元attribution log SHA256は今回も`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7`。

| Forward/head/gradient | Original failed predicate | Value |
|---|---|---:|
| 16/H1/9 | dScores bound | 0.000464227603470169 |
| 17/H1/9 | dScores bound | 0.000250892410201521 |
| 20/H1/9 | dScores bound | 0.000267362599711403 |
| 21/H1/9 | dScores bound | 0.000405069146187632 |
| 21/H1/12 | dScores bound | 0.000391435086282405 |
| 21/H1 | P element | 1.0075062424896863e-6 |
| 22/H0/9 | dScores bound | 0.000259595604102629 |
| 22/H0/12 | dScores bound | 0.000304201908331672 |
| 22/H1/9 | dScores bound | 0.000320947337127792 |

9 predicatesは5 Forward cases / 6 head-case pairs / 2 gradient casesに集中する。新候補がPASSしても、この旧FAILは変更しない。

前研究C0–C7・23候補・既存holdout・同一frozen state・saved Q/K検査も新Host processで再実行した。3 processのmatrix digestは全て旧`daa9f90c6506f808c516110f64c1887ea099dae657d16a89051065ed144dc126`に一致。C7はP `2.886462966422698e-7`、dScores bound `0.0001449218994345951`、failed predicates 0を再現した。

## B. Selective LN1 arithmetic and matrix

### Fixed boundaries

Prospective planは`df7210d`でcommit。S0–S7以外の追加算術探索は行っていない。

Rはmean/centering/variance、Nはepsilon・denominator/reciprocal/normalized multiply、Aはgamma product/beta addition。実sourceの数学`(X-mean)/sqrt(var+epsilon)*gamma+beta`を保ち、既存FP32のanchor-relative/Kahan/amax/hypot実装を非選択部分の基準とした。

| Boundary | R selected | R unselected |
|---|---|---|
| Mean | increasing-j double sum / 64、内部double | FP32 anchor-relative Kahan delta / 64 |
| Center | X − double mean | FP32 `(X-anchor)-delta` |
| Variance | double center² increasing-j sum / 64 | FP32 amax-scaled square Kahan、FP32 rms |
| R→N | 両方選択時はdouble center/varianceを維持 | FP32 center/rmsをdoubleへ正確に昇格 |
| R only | centerとsqrt(variance) rmsをFP32へ丸めて既存Nへ | — |

Mean/varianceは内部一時値であり、新saved tensorを作らない。Rの直接二passは安定な既存FP32 reductionと異なる算術contractだが、同じLayerNorm数学である。

N選択時、Rありは`1/sqrt(double variance + double(1e-5f))`、Rなしは`1/hypot(double(fp32 rms), sqrt(double(1e-5f)))`。N非選択時は従来のFP32 hypot/reciprocal。Nのnormalized multiplyはdouble。N+A連続選択時のみ未丸めhをdouble affineへ渡す。それ以外はhをFP32に丸める。

Saved inverseは常に`float(internal inverse)`、saved xhatは常に`float(internal normalized value)`。A選択時はdouble gamma product/addを最後にFP32へ丸める。A非選択時はsaved FP32 hを使い、独立FP32 multiply、独立FP32 addでYを生成する。FMA off、fast-math off。

**S0はC6、S7はC7とY/xhat/inverse/Q/K/raw/Pの全word一致**。独立reference自身は新算術helperを呼ばない。比較には元のindependent two-pass double、独立double MatMul/Softmaxを使用し、前研究の独立long-double対照も再実行した。

Source epsilon `1e-5f`とlegacy referenceのdouble literal `1e-5`を分離し、両方に対して評価した。Scale `0x3e3504f3`、mask +0 / `-10000`、lower causal、Softmax axis/betaは不変。Legacy R5のdouble shift→exp丸めと、FP32 shift→Host expモデルも別々に評価した。Host libmはDSP exp word oracleではない。

### Original Cartesian, legacy reference / R5

各行44×2×25=2,200 pairs。P/row predicateはForward/headごと一度数え、gradientとの重複を避けた。

| Candidate | R/N/A | P maxAbs | dScores bound | Failed predicates | Failed head cases |
|---|---|---:|---:|---:|---:|
| S0 = C6 | 000 | 5.59052272e-7 | 2.82402922e-4 | 2 | 2 |
| S1 | 100 | 4.39842982e-7 | 2.44689439e-4 | 0 | 0 |
| S2 | 010 | 4.23648023e-7 | 2.45909496e-4 | 0 | 0 |
| S3 | 001 | 4.74321471e-7 | 2.82402922e-4 | 2 | 2 |
| S4 | 110 | 3.58530451e-7 | 2.44689439e-4 | 0 | 0 |
| S5 | 101 | 3.80238337e-7 | 2.54416835e-4 | 1 | 1 |
| S6 | 011 | 5.35539221e-7 | 3.86069583e-4 | 1 | 1 |
| S7 = C7 | 111 | 2.88646297e-7 | 1.44921899e-4 | 0 | 0 |

Source reference・FP32-shiftモデルでも、joint PASS集合は**S1/S2/S4/S7**。S5のFP32-shift P/boundは`3.41665499e-7 / 2.54358297e-4`で、引き続きFAIL。全候補の実Host gradient・row/causal/finite/nonnegative条件はPASSしたが、これでP allocation FAILを取り消さない。

S0/S3の残FAILは16/H1/9と21/H1/9、S5は16/H1/9、S6は21/H1/9。旧9 predicatesの各Forward/head/gradient、worst P coordinate、worst bound coordinate、signed/absolute t、boundの二項を公開aggregateの`critical_original_predicates`と`failures`へ保存した。P RMSと全合同metricも各候補/reference/model/set別に保存した。

S2のsource-reference結果:

| Metric | Original Cartesian | Same frozen 48 | Prior holdout 256 | New confirmation 256 |
|---|---:|---:|---:|---:|
| P maxAbs | 4.23648291e-7 | 4.23648291e-7 | 9.96834343e-8 | 1.35363189e-7 |
| Row error | 2.86392151e-7 | 2.86392151e-7 | 2.68220901e-7 | 1.94646418e-7 |
| dScores bound | 2.45909971e-4 | 1.10999965e-5 | 6.34814495e-6 | 7.72542101e-5 |
| dValue bound | 2.78403685e-5 | 3.13344422e-6 | 8.13556261e-7 | 7.07832824e-6 |
| Signed P-only dScores | 1.09208559e-4 | 9.75763012e-6 | 2.77551009e-6 | 4.11218631e-5 |
| Signed P-only dValue | 1.24202602e-5 | 1.51276090e-6 | 5.38393139e-7 | 4.65200005e-6 |
| Actual Host dScores | 1.37627100e-4 | 1.36230495e-5 | 3.61526943e-6 | 5.00618136e-5 |
| Actual Host dValue | 2.19405846e-5 | 6.04416399e-6 | 2.82781469e-6 | 1.50629324e-5 |

Signed actual perturbation、triangle-inequality componentwise bound、allocated budget、実gradient gateを別判定した。Boundがsigned actualを下回らないことと独立式との一致を全pairで検査した。

## C. Independent confirmation

Set Aは元44×2×25、Set Bはmetadataで対応する24×2=48 pairs、Set Cは前研究の16×2×8。実training trajectoryの対応を証明した集合ではなく、`ACTUAL_TRAINING_MATCHED_UNAVAILABLE`を維持する。

新Set Dはplanの16固定seeds、8パターン、新しいQ/K weights、8 gradient seeds、16×2×8=256 pairs。低分散、4096 offset、符号相殺、small Q/Kによるnear-tied狙い、競合logits、head非対称を含む構成。dP amplitude 4/32/256/512、dContext 2/16。Frozen gradient9/12の複製ではない。パターン名は生成目的を表し、全行が同じlogit gapを持つ保証ではない。

開発集合を確認した後、**S2を`a8f5105`で選択・commitしてから**Set Dを初評価した。対象はS2とS0/S7 anchorsのみ。S2/S7は両epsilon reference・両Softmaxモデルで全gate PASS。失敗に応じた再選択、seed/threshold変更、fixture専用補正はない。

Set DではS0=C6もPASS。既存Set Cでも旧C0がPASSしていた。Synthetic確認集合は独立な回帰証拠だが、元frozen Cartesianより必ず厳しい集合、実checkpoint全分布、実学習generalizationの証明とはしない。

同一Host binaryで開発・confirmation各3独立process。数値digestにtiming・端末IDを含めず、各cohort内で全記録一致。別Softmaxモデルのdigest一致とは区別する。

## D. LN1 saved state / Backward

S2の数学を「従来FP32 anchored reduction → double denominator/inverse/normalized multiply → FP32 saved h/inverse → 従来FP32 affine」と定義する。FP32 native outputsはY `[32,64]`、h `[32,64]`、inverse `[32,1]`。Gamma/betaは元の同一FP32 parameter。YはQ/K/V/Gateの全consumerに共有し、h/inverseはlocal backwardまで保持する。

以下は旧`lncustom::backward<false,false>`実bodyをQHPI Host mockで実行した結果。新Forwardと旧BackwardのDSP接続ではない。double analytical referenceはsource epsilon。旧local gate `maxAbs<=0.02`は不変。

| S2 metric | Frozen 44 | Prior holdout 16 |
|---|---:|---:|
| Y maxAbs | 4.01991179e-7 | 4.34315649e-7 |
| xhat maxAbs | 4.92090636e-7 | 3.58143431e-7 |
| inverse maxAbs | 1.37200952e-5 | 1.54888091e-5 |
| Actual old FP32 backward dX | 1.31047490e-5 | 6.65637041e-5 |
| dGamma | 2.02949707e-5 | 1.12327927e-6 |
| dBeta | 7.89668411e-6 | 3.96743417e-7 |
| Y from saved xhat, old affine | **0 words / 0 error** | **0 words / 0 error** |

全S0–S7について同じForward output/saved状態と旧Backwardを比較し、上記Backward gateを維持した。Guard、poison/complete writes、finite、input immutabilityもHostで検査した。

S2のh生成には未丸めdouble inverseを使う。Inverseをsaved FP32に置換して同じcenterからhを再計算すると、frozenで16,816 words、max `4.76837158203125e-7`、prior holdoutで8,443 words、max `2.384185791015625e-7`異なる。従って**saved hは直接保持し、inverseだけからhを再生成して同一と扱わない**。Yはsaved hから再構成できる。

S7はYの旧affine replayがfrozen 11,884 words / max `2.384185791015625e-7`異なる。S2はこのAffine再構成差を解消するが、Forwardの内部double値全体をsaved floatで完全再現する契約ではない。

Backwardは同じ連続LayerNorm式の解析的gradientをFP32 saved近似で評価する。丸め写像そのものの厳密な微分、partial-precisionアルゴリズムの厳密autodiff、一層全体のgradient正当性とは区別する。旧Backwardを変更していない。

Q/Kは高精度MatMulのFP32 outputをそのまま保存・使用し、別近似のQ/KをBackwardへ渡さない。全candidateのsaved representation mismatchは0。S2のreceived-Pから独立double analytical QK gradientと旧FP32 raw/scaleを比較したlocal errorは、FP32-shift frozenで`3.101743523359346e-5 <=0.01`。dScores/dValueは表の実Host gateを満たす。Head Scatter以降、LN1 dY全branch集約は未解決である。

## E. Q/K/V mixed-precision ABI design

Source上のLN1 YはQ/K/VとGate projectionの共通入力。Weightsは同じWq/Wk/Wv、Gamma/Betaを使い、Forward saved Q/K/V/Gate/Pのlifetimeを短縮しない。入力Yを改善すればV/Gateも変わるので、Q/K/P-only Host PASSからV/Gate正当性を認定しない。

| Option | Projection launches | Duplicate FP32 Q/K work | Design / risk |
|---|---:|---:|---|
| A new mixed invocation | 1 | 0 | 高精度Q/K + 既存HVX helperをVに一度呼ぶ |
| B new Q/K + V-only | 2 | 0 | 独立resource/receipt、追加launch・Y fanout |
| C old QKV + new Q/K | 2 | 262,144 terms | 診断control限定、標準方式にしない |

第一候補は**A**。既存の`hexatrain_ordered_hvx_qkv_nn64(a,b,y)`は1つのNN行列を計算する関数なので、Wv/Vだけへ一回呼べる設計。旧ProjectionQKVHvx invocationを3本分再実行する必要はない。ただし新wrapperとDSP実行は未実装。

新診断名のOp、Observed/NoReceipt別specialization、独立marker/counterを定義し、旧51-op登録・names・receipts・ABIを変更しない。提案signatureはFP32 Y `[1,1,32,64]`と3 weights `[1,1,64,64]`、3 FP32 outputs `[1,1,32,64]`、Observedのみ新receipt。Counterはspecialization/graph ownerを分離し、global旧counter共有やconcurrent execution順序を前提にしない。

Q/KはFP32 operandをdoubleへ昇格し、q増加、独立double product/add、最後のみfloat store。Vは別TUの既存strict IEEE HVX、128B vector / FP32 product/add / q順を保持する。Q/Kの旧HVXとのword差は新contract上のFAILではない。新mixed OpでHVX resourceを保持する時間、scalar/HVX scheduling・新receipt schema/sizeはruntime設計・実機検証が残る。

Tensor output方向は診断Standalone APP_READ、native chain NATIVE、external inputs APP_WRITE。Unit leading axes、row-major、head0/head1独立、QKᵀ右transposeを維持する。Production builderへ接続していない。

## F. Hexagon feasibility

Actual template算術とQ/K NN K64、QKᵀ NT K32をV81でcompileした。Flagsは`-std=c++17 -mv81 -O2 -G0 -ffp-contract=off -fno-fast-math -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fstack-usage`。リンク用probeのみ追加`-fPIC`。No QHPI package / registration / deployment。

確認したlowering:

- double addは`dfadd`、multiplyは`dfmpyfix`2、`dfmpyll`1、`dfmpylh`2、`dfmpyhh`1のpartial-product sequence。
- FP32→double `convert_sf2df`、double→FP32 `convert_df2sf`。変換の実際の丸め/subnormal/exception状態は実行していない。
- NN/NTのLLVM IRはsingle double accumulatorのq順fadd。機械語ではsoftware-pipelineのprologue/epilogueがあるが、tree reduction/FMAへ変更した証拠はない。IRにfast/fma無し。
- S2は`hypot`、`__hexagon_divdf3`、旧reductionの`sqrtf`を呼ぶ。S7はdouble`sqrt`とdivision helper。S1はdouble`sqrt`とFP32`hypotf`。

| Function | Static frame bytes, excludes callees | Folded spill/reload sites | Static sf2df / df2sf sites |
|---|---:|---:|---:|
| S0 | 560 | 4 / 3 | 3 / 5 |
| S1 | 560 | 4 / 3 | 10 / 6 |
| S2 | 560 | 4 / 3 | 4 / 7 |
| S3 | 560 | 4 / 3 | 6 / 5 |
| S4 | 568 | 5 / 4 | 7 / 4 |
| S5 | 560 | 4 / 4 | 13 / 6 |
| S6 | 560 | 4 / 3 | 8 / 8 |
| S7 | 568 | 5 / 4 | 11 / 5 |
| Q/K NN64, one matrix | 32 | 2 / 2 | 6 / 1 |
| QK NT32, one head | 48 | 3 / 3 | 6 / 1 |

Sitesはstatic assembly出現数で、動的回数・cycles・maximum live register数ではない。`selective.su`とassembly/disassemblyをprivate保存。Shared QHPI wrapper・math callee・HTP workspaceを含むstack/physical memoryは未測定。

固定SDKのV81 archiveではdivision helperは`libgcc.a`、`hypot/sqrt`は`libc.a`に定義される。`libm.a`は小さいempty objectのarchiveだった。最初のlibm/libgccだけのsymbol auditはhypot未発見で停止し、libcも含む実際の提供元監査へ修正した。SDK自体は変更していない。

**Linkageは未成立・未解決**。Plain shared ELFで`--no-undefined`を指定するとmath/initialization依存が未解決（exit1）。次に固定SDKのPIC libc/libgccを明示してstartfiles/defaultlibs無しで限定検査すると、math symbolだけでなく`sys_Tlsalloc/get/set`、`sys_sbrk`、`sys_Mtxdst`などOS依存が残った（exit1）。失敗log・二条件を保持し、fake OS stubや数値helperへの置換、unbounded flag探索は行わなかった。

これはbare linkの結果であり、QURT/QHPI loaderの実環境で解決不能という証明ではない。既存packageがsqrtfを使用できることも、新hypot/DF helperのloader解決を保証しない。将来は正規DSP runtimeのlink/import設計、rounding/subnormal/exception、math accuracy、callee stackを検証する必要がある。

Generic QNN datatypeにFLOAT64があっても、固定QHPI enumにFLOAT64はない。新候補は**内部double + FP32 native tensor**で実現する設計で、FLOAT64 tensorやFP64 HVX datapathを仮定しない。Scalar DF loweringからFP64 throughputを断定しない。

## G. Conditional DSP prototype / safety

**NOT IMPLEMENTED / NOT MEASURED**。新packageのStandalone accuracy、DSP arithmetic/receipt時間、QNN execute wall、first/steady/p10/p90、3-device-process cohortはいずれもNOT RUN。

開始時read-only preflightは`HOST_STORAGE_HEADROOM_LOW`で停止。Host free≥20GiB条件を満たさず、physical identity/awake/top-resumed等の端末側確認前に停止した。今回端末がawakeかどうかを確認済みとは主張しない。device executes 0、停止後device operations 0。空き容量確保のための既存artifact削除もしていない。

Host選定・saved-state/FP32 ABI・codegenは成立したが、安全な実機試験条件が未成立であり、runtime link/importも未解決。Entry条件を満たさないのでmicroprototypeへ進めなかった。ユーザー指示の許容するHost/design/compile-only停止として記録する。端末状態を無期限pollせず、wake/unlock/force-stop/foreground/APK replacementを行っていない。

再開時は20GiB以上のHost free確保後、全安全preflightを新たに実行し、同一物理端末・awake・known top-resumed・active run absence・device free≥12GiB・thermal/power・APK/SDK/Skel/package/fixture/protected identityを確認する必要がある。

## H. Accuracy–cost decision / regression

### Logical arithmetic per layer / microbatch

全S候補共通でQ/K 262,144、QKᵀ 65,536、合計327,680 double productと327,680 double add。各MatMul outputはfloat。これはinstruction数やcycleではない。

以下は**新sourceに明示したLNの演算**。Math helper内部のwork、FP32の非選択stage、compilerによる定数除算→multiply、conversionsは別扱い。

| Candidate | Double mul | Double add/sub | /64 | Double reciprocal | Double sqrt | Double hypot | Frozen joint |
|---|---:|---:|---:|---:|---:|---:|---|
| S0 | 0 | 0 | 0 | 0 | 0 | 0 | FAIL |
| S1 | 2048 | 6144 | 64 | 0 | 32 | 0 | PASS |
| S2 | 2048 | 0 | 0 | 32 | 0 | 32 | PASS |
| S3 | 2048 | 2048 | 0 | 0 | 0 | 0 | FAIL |
| S4 | 4096 | 6176 | 64 | 32 | 32 | 0 | PASS |
| S5 | 4096 | 8192 | 64 | 0 | 32 | 0 | FAIL |
| S6 | 4096 | 2048 | 0 | 32 | 0 | 32 | FAIL |
| S7 | 6144 | 8224 | 64 | 32 | 32 | 0 | PASS |

sqrt(epsilon)は固定定数で、上表のruntime sqrtには数えない。S7はcenterの計算結果をrow temporaryに再利用するため、前研究referenceの繰返しcenter計算を数えた費用と一致する必要はない。算術word一致は別途確認済み。ConversionsはFのstatic sitesを示し、最適化で消えるsource castsを動的instructionsへ換算しない。

19×8=152の論理scopeではMatMul products/adds各49,807,360。S2 LN normalized multiplies311,296、reciprocal4,864、hypot4,864。S7 LN multiplies933,888、add/sub1,250,048、/64 9,728、reciprocal/sqrt各4,864。Production wallへ単純倍算しない。

LN output storesはY2048+h2048+inverse32=4,128 floats、Q/K+raw stores6,144 floats。追加saved payloadは0、既存h/inverse8,320 bytes、saved Q/K16,384 bytesを維持する。内部center temporaryはdouble64=512 bytes。旧実kernelは最大D128用float128=512 bytesを確保しており、新scratchが必ず256 bytes増えるとは断言しない。固定D64の最小float rowとの比較なら+256 bytes。Frame560/568はcalleesを含まない。

提案AではLN→mixed QKV→paired QKの3 launches、B/Cなら4。下流Scale/Mask/Softmax launchesは共通で別計上。Cはさらに262,144 FP32 product terms/layer/MBを重複し、19×8で39,845,888 terms。Launch/resource scheduling、physical memory traffic、HTP workspace、QNN finalize/node/tensor数は新graphをbuildしていないため未測定。

### Pareto / next implementation

S1/S2はsingle-componentで開発PASS、S4/S7は追加精度によりP要素誤差をさらに改善する。S1のdScores余裕は約2.12%、S2は約1.64%、S7は約42.0%。S2はS1より明示的double add/subが少ない一方、hypot/division helperがあるので**DSP上で最速と判定していない**。

候補選定はS2。旧合同契約、両reference/models、独立confirmation、saved-stateの検査を満たす、評価済み集合内の小さいprecision subsetである。S1は別のPareto候補だが新confirmationは未実施。全アルゴリズムの最小性やS2の最速性を主張しない。

次は**S2 LN1 standalone**を一領域prototypeとして実装する。FP32 outputs/h/inverseと旧local Backwardを同時に検査し、hypot/DF helperのloader/runtime/費用という未知数を先に解く。Q/K doubleとQK doubleの実装は依然必要で、LN1だけの成功をP closureと呼ばない。Full C7は精度余裕のあるfallback/reference anchorとして保持する。いま全C7を一括移植、旧HVXの速度改善、P gate緩和へ進む根拠はない。

### Regression

| Validation | Current task result |
|---|---|
| Original nine / R0–R5 | REPRODUCED、元log digest一致 |
| Old C0–C7 / Cartesian / same frozen / holdout / QK saved | REPRODUCED、3 process一致 |
| New S0–S7 / two references / two Host Softmax models | analysis PASS、候補別FAIL保持 |
| New S2 confirmation | joint PASS、同一binary 3 process一致 |
| Actual old LN backward Host / new state / poison/guard | PASS |
| Existing Scalar MatMul/Multiply / QK transpose / Scale / Mask / Softmax | Host suite PASS |
| Existing HVX FFN2/QKV geometry / registration/ABI contracts | Host PASS、HostでHVX実行を成功扱いしない |
| Fast | 最終PASS、59.9s（先行93.8sも別log保持） |
| Host profile | PASS、450.1s |
| Fixed SDK / existing APK read-only audit | core/build ID PASS、optional inventory advisory exit3 / APK SUCCESS |
| New Qnn build / Device smoke | NOT RUN、新QNN node/package無し、Device safety停止 |
| New DSP HVX/FFN2/QKV/P/Scale/Backward regression | NOT RUN |
| New-package Muon W8 114/114 | NOT RUN、過去PASSを転用しない |

Host suiteには未配置checkpointを使うoptional self-test SKIP表示があるが、実際のHost profile出口はPASS=2/FAIL=0/SKIP=0。新数値analysisや旧FAIL判定の代用にはしていない。

保存状態Host harnessの初compileはQHPI header include欠落で失敗し、includeを明示して再実行PASS。補助検査追加前のsaved harnessはpilot、追加後のfinal saved binary/logを別保存。初runtime symbol監査の提供元見落とし、bare link失敗もprivateに保持する。新P算術の選定・threshold・fixtureを変更した修正ではない。

公開exporterはcase/pairのCartesian一意coverage、summary最大値・predicate数・matched値、bound、旧log identity、3-process/binary/source hashes、protected状態を再検査するallow-list。SelfTestは11種類の破損・未許可fieldをrejectし、allocationとactual-gradient gateを分離する。Raw tensors/row配列、IDs、絶対pathは公開しない。

## I. Final answers

1. S0–S7の開発合同PASSはS1/S2/S4/S7（両reference・両Softmaxモデル）。
2. ReductionのみS1はP4.39843e-7、bound2.44689e-4でPASS。
3. NormalizationのみS2はP4.23648e-7、bound2.45909e-4でPASS。
4. AffineのみS3はbound2.82403e-4でFAIL。
5. R+NのS4はPASS、R+AのS5とN+AのS6はFAIL。
6. Full C7/S7の精度を再現し、全stageのanchor word差0。
7. 今回独立確認まで満たした小さい候補はS2。最小性は評価済み集合内、latency最小は未認定。
8. S2はC6に対しP maxAbsとdScores boundを同時に改善した。
9. 既存holdout・新confirmationもPASS。ただしS0=C6も新confirmation PASSで、generalizationの限界がある。
10. S2 Yはsaved xhatから旧affineで全word再構成。内部inverseからhへの全演算再現はsaved FP32 inverseだけでは不可能なので、hを直接保存する。
11. 旧LN local backward実bodyのHost gateを維持。DSP新Forward接続は未検証。
12. 同一mixed invocationの共存設計は可能。既存HVX helperをVに一度呼ぶ設計、実装・実機は未検証。
13. Q/KとVを分離するとprojection launchは1→2、重複演算不要。旧QKVを併用すると追加262,144 FP32 terms。
14. 新Q/K float outputsそのものをForward/Backwardへ保存するのでrepresentation identityは0 mismatch。実native接続は未検証。
15. dfadd、partial-product DF multiply、sf2df/df2sf、double division helper、sqrt/hypot calls。F参照。
16. Scalar doubleの実DSP accuracy/latencyは未測定。
17. FP64 tensor不要。内部doubleを用い全native/saved境界はFP32。
18. 新saved payload0、center512 bytes、静的frame LN560/568、NN32、NT48。Callees/physical workspace/schedulingは未測定。
19. Bounded DSP prototypeは実装していない。
20. Host storage safety gate不成立。加えてplain ELF runtime link/import設計は未解決。
21. C7は数学anchorとして価値があるが、先にS2を試す根拠が得られた。一括移植は推奨しない。
22. 次はLN1 S2 standalone。Helper accuracy/linkageとsaved-stateを同時に検証する。
23. 旧HVX/W8 source/ABI/packageを保持しHost contract PASS。新packageの実機HVX/W8回帰はNOT RUN。
24. Forward P累積DSP精度は認定不可。
25. 高精度LN/QK projection/QK rawのDSP実装・runtime・native/observer-free P合同精度、V/Gate/PV/output、saved P→Backward、Head Scatter、projection/Gate gradients、LN1 dY集約、FFN/residual gradients/materialization、8MB accumulationが残る。
26. 40k→60k / 200kは引き続き**NO-GO**。

## J. Provenance / reproduction

| Item | Identity |
|---|---|
| Prospective plan commit | `df7210d` |
| Plan SHA256 | `15393639a8463e9b2528796fbe0ae296dfd448aa897c57277f41e5a44a987c38` |
| Host matrix source | `41fc929dc6ba82f55805924de61642a8f1b524c1` |
| Host matrix exe SHA256 | `d4ca5b9fd3184b01186ff2cb4d8bfd98b7d6bb409b8b0edff30de108f1e98738` |
| Development, 3 identical process digests | `cdbfb7a304ba93ef7927966882362e5a3563c411d2fbc0ba002ebb6a6683b3ab` |
| Development FP32-shift digest | `8ce49b4de1c7ab413ce5be9a24697896530ee7f3d602256beda6c865b0e0a326` |
| Selection / confirmation source | `a8f510522f3b8e93a13c414a0bdddcc7c97fa59f` |
| Confirmation, 3 identical process digests | `ac2074c883fad647fa05ba9cd579530f149b479b118489b1e85c32eaf4481e3c` |
| Confirmation FP32-shift digest | `85b151ac4223380cd06c528a63fa0b5b2618b84c226c8402d351846bd128b89a` |
| Final saved harness source | `28a8dfdc05759fe439239dabf5eaf38e0665be9d` |
| Final saved harness exe SHA256 | `f89799daec7a8518d1d9f5b9fa001a3e7a79ea4089bf8ec12373276fa7409fe4` |
| Codegen / bounded link audit source | `9b736087139edb0a2d155314a36473d85c97c6f5` |
| Last historical device-tested source | `65148e19237b162da14042b500892ef5716e9b8d` |
| Historical DSP package | `10770ea36f8c994462377e5bce68e4d8619befc7576ddf21cb61d41fd2ad2857` |
| Historical Host package | `4322bded8951959f4ca37840a638fc6332e7cc03b45b8dd8ab44fe3646df81e9` |
| Historical probe | `9726ea13adce1ce4cb3684ad7ae28d8eff3b9011e8a75909a1093978097603f2` |
| Host compiler | `5fa84c1460fd741371f730788ba3f994cf1d682be866fe7d3ed799e81844670a` |
| Hexagon compiler | `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047` |
| NDK compiler, historical package anchor | `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3` |

Matrix binaryは同一のまま、補助saved-state検査とcodegen/link監査を別binary/source identityで区別する。新DSP source/device cohortは存在しない。

Frozen inputs SHA256はone-layer `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`、LN parameters `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`、producer fixture `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`。旧synthetic input digestを再現。新confirmation入力はcommit済みplanのLCG/seed/FP32生成式とHost source/hashで固定し、raw tensorをcommitしない。

再現手順は新private directoryに固定QAIRT root/Build IDを明示し、`test_forward_p_attribution.ps1 -Legacy`、`test_forward_p_upstream_precision.ps1`、`test_selective_ln1_precision.ps1 -Phase Development`。選定済みselectionを保存してから同じexeで`-Phase Confirmation -Selected 2`。Saved harnessは同じg++ flags/QNN includeとCPU/shape-validator sourcesで別compile/run。Codegenは`audit_selective_ln1_codegen.ps1`へ固定Hexagon rootも明示。最後に`export_public_selective_ln1.py --selftest`、private-root/output指定で再集計する。

Private rootは`build/reports/g1-c7-selective-ln1-dsp-feasibility/`。Public aggregateは`docs/results/g1-c7-selective-ln1-dsp-feasibility-summary.json`。最終HEAD・clean tree・旧refs・protected hashes・report/aggregate identityはcommit後のprivate `final-closeout-private.json`で固定する（自己commit hash埋込みの循環を避ける）。

40k checkpoint SHA256 `89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`、他baseline/forensic/dataset/tokenizer/cache/orderを保持。Checkpoint decodeなし、production/optimizer/codec変更なし、training更新なし、Final dataset testなし、push/PR/merge/history rewriteなし。

分類は`SELECTIVE_LN1_MATRIX_VERIFIED`、`SELECTIVE_LN1_JOINT_CONTRACT_PASS`（S2 Host）、`C7_HOST_REFERENCE_REPRODUCED`、`MINIMAL_HOST_PRECISION_SUBSET_IDENTIFIED`（評価範囲内）、`LN1_SAVED_STATE_CONTRACT_CHARACTERIZED`、`QKV_MIXED_PRECISION_ABI_DESIGNED`、`DSP_DOUBLE_CODEGEN_AUDITED`、`BLOCKED_DEVICE_SAFETY`。Runtime linkage/rounding/latencyは`UNRESOLVED`。DSP standalone/performance/C7 native PのVERIFIED分類は使用しない。

35/40-node Tap incidentは`REPRODUCED / CAUSE UNRESOLVED`、6031は`UNRESOLVED / DORMANT / WATCH`。Full Attention / Full1-layer / Full19-layer correctnessは未認定。

## K. S2 standalone implementation / safe resume

### Authorization and safety

ユーザーの続行指示により再開。最初のread-only再計測はHost free 18.88GiBで既定20GiB gate未達。ユーザーが**今回に限りHost容量gateを免除、他の安全条件は維持**と明示したため、default-offの`UserAuthorizedHostStorageWaiver`引数を設けた。既定20GiB、SDK/端末の他gateは不変。SelfTestは通常容量PASS、容量不足BLOCKED、明示免除時のみ許可を検査する。

免除指定の初回read-only preflightは同一物理端末、awake、top-resumed、active run absence、external power/thermal、device storage、APK、SDK/compiler、旧package、protected identityをPASS。実行許可は後のprocessへ持ち越さず、投入直前に再検査する。

Qnn profile終了後のpilot-1 preflightでは、Host容量は約78.3GiBで通常gateもPASSした。一方、`EXTERNAL_POWER_UNKNOWN_OR_ABSENT`で停止。probe投入前、device execute 0、停止後device操作0。初回preflightのpower PASSを後の状態へ転用しない。手動の給電確認を依頼し、wake/unlock/foreground/force-stop/root/APK replacementを実施していない。

### Implemented bounded geometry

**新packageはbuild済み、当該power停止時点ではDSP未実行**。別名`HexaSelectiveLn1`にBaseline/S2 × Observed/NoReceiptの4 operationsを追加。既存`HexaFp32` 51 operations・ABI・receipts・HVX binaryは変更しない。

S2は前研究と同じ`selective::ln<2>` body、T32/D64のLN1のみ。Q/KやQKᵀのdouble DSP実装へ拡張していない。Baselineは旧LN1 kernel bodyを再利用し、新library内でnamespaceを分離して旧counter/global symbol ownershipとの衝突を避ける。Host準備用packageは算術呼出しを全4入口で拒否する。Host単体検査用macroをDevice/Android packageのbuildに指定しない。

同一binaryに旧BaselineとS2を含み、同じinput/backend/外側timer条件で比較する。4 inputsはFP32 X/gamma/beta/source epsilon、3 outputsはFP32 Y/xhat/inverse。Observedのみ独立marker/receiptを追加。Source epsilonは1e-5fのまま。Native xhat/inverseは同一graphの旧`LnBackward`または`LnBackwardNoReceipt`が直接受け取る。Host正解を中間へ再注入しない。

8 variantsはBaseline/S2 × Observed/NoReceipt × Tap/NoTap。NoTapではh/inverseをNATIVE、final Y/dX/dGamma/dBetaのみAPP_READとする。Observedではproducer保存値・Backward consumer受信値を全word照合。NoReceipt/NoTapの不可視stateを直接観測したとは主張しない。

### Fixed fixtures and gates

Device planは`7a1deb2`で実測前にcommit。元44ケース＋既存独立holdout16ケースの60ケースをFP32で凍結。CandidateごとのHost wordsと、Custom bodyを呼ばないsource-epsilon independent double Y/xhat/inverse/dX/dGamma/dBetaを別に保存する。

Host fixture exporterは新registration 4 operations、signature拒否、入力不変、全面書込み・finiteを検査し、60ケースでPASS。初compileのInfo構造体field位置誤りを修正してから凍結・実機buildへ進んだ。旧数学やthreshold変更ではない。

事前登録gateはS2 Host words mismatch 0、Y/xhat/inverse maxAbs≤0.002、local backward dX/dGamma/dBeta≤0.02、finite、poison/canary、input mutation0、API success、Observed DSP marker、CPU fallback無し。BaselineのHost word差も別metricに残す。全variantがPASSした場合のみtimingを収集し、正式候補は同一final binaryを3独立processで確認する。

Timingは31 steady samplesのmedian/p10/p90、first execute、finalize、Observedのkernel-body/receipt-inclusive window、execute wallを記録する設計。NoReceiptの内部arithmetic時間は未測定扱い。WallはForward＋local Backwardの2-node graphで、probeのAPI loggingも含む。旧Baseline body内のvalidation/timer費用もbody windowに含まれるため、純粋なDF instruction throughputやproduction speedupと解釈しない。

### Build / runtime evidence

固定SDK・compilerで新DSP/Host package、Standalone probe、regression probeのbuildがPASS。通常QHPI link条件ではdouble division helperがbinary内で解決し、`hypot/hypotf/sqrtf`等はruntime importとして残る。固定SDK dynamic libc/libgccには該当exportがある。先のbare `--no-undefined` linkとは条件が異なる。**新packageの実loader解決と算術実行は、power停止時点ではUNVERIFIED**。

| Identity | SHA256 / commit |
|---|---|
| New build source | `b7729eecc1c3fc91992c61edb787c0a17bfcf771` |
| New DSP package | `7f4db8c5f7ab6b25903e090cb9024d54c19c8b0d962d8e1452cb0ed8d337df7c` |
| New Host package | `8f87dee9eedff60d6c28b74f3b9a9f5cd81f4f272e5b64d988c12c1e55204370` |
| Standalone probe | `70fe2a42282c23b792e78f654b40de5a4a3f00c4f76925f5fbb9cb9f86b22d2a` |
| Regression probe | `73213a492ca408e86e4adeb45f76e41dffa0177b269152d6c29186e0f7aef485` |
| Frozen device inputs / Host words / double targets | `492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02` |

既存回帰probeを変更せず再利用するwrapperは、新旧両packageをcontext生成前に同時登録する。これにより将来のfresh HVX FFN2/QKV、QK/Scale、Mask、Softmax回帰を新bundleのidentityへ紐付ける。W8/Device smokeも新規実行した結果のみ記録し、過去の114/114を転用しない。

### Resume validation / current limits

Fast profile PASS 69.3s。Qnn profile PASS 777.5s（Host suite359.3s、QNN APK407.3s、SDK/APK auditを含む）。新Host packageのCPU算術拒否4/4、invalid signature拒否4/4はPASS。旧Scalar MatMul/Multiply、LN、HeadSplit、Softmax、QK/Scale、HVX FFN2/QKV arithmetic/ABIのHost回帰も新規実行PASS。旧P合同契約runnerはexpected exit4、元9 FAILを説明するattribution/R0–R5 logは元のSHA256と一致した。

現時点の分類は`BLOCKED_DEVICE_SAFETY`（外部給電）。S2は`BUILT_NOT_DEVICE_VERIFIED`、Standalone精度・runtime loading・性能・Device smoke・新bundleのHVX/W8実機回帰・3-process cohortはNOT RUN。最新APKのinstalled一致も、power gateより後の確認になるため未完了である。未実行工程を過去の実機PASSで補わない。

追記の公開aggregateは`docs/results/g1-c7-selective-ln1-dsp-resume-summary.json`。旧Host aggregateは初回研究のsnapshotとして保持する。追加allow-list exporterは安全停止、binary/source identity、観測variant coverage、タイミングを除く数値digestを扱い、SelfTestで欠損/未許可raw field/unsafe ledgerを拒否、word gateとprivacyを検査した。

Private evidenceは従来root内の`resume-1/`。powerで停止したpilot-1を成功cohortへ混ぜない。再開追記のfinal HEAD、clean tree、旧refs、protected identities、report/aggregate hashesは`resume-1/final-closeout-private.json`にcommit後記録し、旧Host研究のcloseoutを保持する。新private build rootを使用し、従来DSP package/probeやprivate evidenceを上書きしていない。新packageに高精度Q/K/QKは無く、DSP native LN1→P、Full1-layer/19-layer、training readinessは認定しない。40k→60k/200kは引き続きNO-GO。
