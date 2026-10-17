# G1 Forward P numerical attribution and safe resume

## A. Initial state

Base: `52f5f04edf70a69fdc48251baddb2348472f5c1d`.
Branch: `investigation/g1-forward-p-attribution-safe-resume`.
The initial tree was clean, with no upstream and 105 local commits absent from
known remote refs. Local research history was used without reconstruction,
remote operations or history rewriting. Previous branch refs were frozen.

Two independent blockers were retained: the original Host P joint contract had
nine failed predicates; the previous device attempt stopped before execution at
`DEVICE_NOT_AWAKE_OR_UNKNOWN`. Fast/Host/Qnn success did not cancel either.

The initial, previously untested device candidate had compiled source
`ac5be47c155725a0a442da540ab0359b34602a88`:

| Artifact | SHA256 |
| --- | --- |
| DSP package | `45aebb0312f7d7f39cf01c61e9456cc3e87c70b73854db68c755ea2df6428d4c` |
| Host package | `480014e5f0259a04b8065c2e911930a0f19b41568a5fc5b0182adb9499c2b210` |
| one-layer probe | `06fce37163459854f2c6c5f7e6a31b91a64b098a304a5dbf57afbb7019d0abea` |

The initial last device-tested source was
`b6794c4572d10c163acb5201e811546ac564243e`; its PASS is not transferred.
Initial Host attribution changed no DSP arithmetic. Subsequent measured standard
Mask/Softmax failures justified bounded diagnostic additions described below.
Production builders, existing arithmetic bodies and HVX kernels remain unchanged.

Final51-op candidate source: `65148e19237b162da14042b500892ef5716e9b8d`.
Three independent processes per variant now verify Scale, reused Custom Mask and
Custom local Softmax on this binary. Native LN1-origin P still fails the unchanged
cumulative contract. The intervening ADB transport stop was preserved and resolved
only after the user's manual reconnection confirmation. Current numerical blocker:
**BLOCKED_HOST_P_BUDGET**; current device safety is not blocked.

| Current artifact | SHA256 |
| --- | --- |
| DSP package | `10770ea36f8c994462377e5bce68e4d8619befc7576ddf21cb61d41fd2ad2857` |
| Host package | `4322bded8951959f4ca37840a638fc6332e7cc03b45b8dd8ab44fe3646df81e9` |
| one-layer probe | `9726ea13adce1ce4cb3684ad7ae28d8eff3b9011e8a75909a1093978097603f2` |

Source audit reconfirmed rank2 Q/K `[32,32]`, QK transpose flags false/true;
FP32 scale `1.0f/std::sqrt(float(64/2))`, bits `0x3e3504f3`; and a row-major
finite Add mask, +0 (`0x00000000`) for c≤r, -10000 (`0xc61c4000`) for c>r.
There are 528 allowed and 496 future elements per head. Production Softmax is
axis1/beta1; rank4 diagnostic `[1,1,32,32]` uses axis3/beta1. Heads are independent.
Graph-owned `attentionScale`/`maskData` outlive the STATIC tensor buffers.
No constant, transpose, saved lifetime or Forward expression was changed.

## B. All nine failed predicates

The unchanged original assessment returned **exit 4 / nine failures**, with
44 Forward fixtures × 2 heads × 25 gradient fixtures = 2,200 combinations.
The new analysis returns exit0 only when it independently reproduces this
failure and verifies its decomposition/bounds. It does not turn the old test green.

| Forward | Head | Gradient | Predicate | Actual | Threshold | Excess |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 16 | 1 | 9 | dScores P-bound | 4.64227603470169e-4 | 2.5e-4 | 2.14227603470169e-4 |
| 17 | 1 | 9 | dScores P-bound | 2.50892410201521e-4 | 2.5e-4 | 8.92410201520542e-7 |
| 20 | 1 | 9 | dScores P-bound | 2.67362599711403e-4 | 2.5e-4 | 1.73625997114029e-5 |
| 21 | 1 | 9 | dScores P-bound | 4.05069146187632e-4 | 2.5e-4 | 1.55069146187632e-4 |
| 21 | 1 | 12 | dScores P-bound | 3.91435086282405e-4 | 2.5e-4 | 1.41435086282405e-4 |
| 21 | 1 | N/A | P element | 1.00750624248969e-6 | 1e-6 | 7.50624248968639e-9 |
| 22 | 0 | 9 | dScores P-bound | 2.59595604102629e-4 | 2.5e-4 | 9.59560410262908e-6 |
| 22 | 0 | 12 | dScores P-bound | 3.04201908331672e-4 | 2.5e-4 | 5.42019083316718e-5 |
| 22 | 1 | 9 | dScores P-bound | 3.20947337127792e-4 | 2.5e-4 | 7.09473371277923e-5 |

These are **five unique Forward fixtures, six Forward/head pairs, two head
indices and two gradient fixtures**, not nine independent Forward failures.
Six bound failures use gradient9, two use gradient12. The first failure in
Forward/head/gradient traversal and the worst bound are (16,1,9).
The only P element failure is (21,1); its P is reused across all 25 gradients
but its element failure is counted once. No row or dValue-bound predicate failed.

| Forward/head | P maximum-error row,col | P reference → Host | Bound maximum row,col for gradients |
| --- | --- | --- | --- |
| 16/1 | 29,20 | .594409085551458 → .594409883022308 | g9:6,2 |
| 17/1 | 7,5 | .763832327441894 → .763831913471222 | g9:7,5 |
| 20/1 | 27,22 | .286563206695415 → .286562711000443 | g9:27,22 |
| 21/1 | 1,1 | .465815758517393 → .465816766023636 | g9:7,3; g12:5,3 |
| 22/0 | 23,13 | .503519357937363 → .503520190715790 | g9:3,2; g12:23,13 |
| 22/1 | 3,3 | .383099251152481 → .383098661899567 | g9:3,3 |

All critical P and bound positions are allowed causal positions. None receives
the -10000 mask value; no transpose or causal-boundary indexing defect was found.
Private records preserve each failure's full critical row distribution, logit
range/top2 gap, dP/dContext magnitudes, signed perturbation, actual FP32 consumer
error, bound coordinate, excess and reference identities. The allow-list summary
exports scalar diagnostics only, without rows or raw tensors.

Forward16–23 use the frozen 40k parameter-derived fixture family. The failed
Forward indices are all in this family. Gradient9/12 belong to the synthetic
8× Wv/Wo family; gradient9 has head1 max|dP|=1170.1279296875 and max|dContext|
=42.1717758178711. Gradient12 head1 values are 504.587585449219 and
23.1130428314209. These operands were not reduced or rescaled.

Reproduction recipes remain the original ones: synthetic parameter seed1+i;
token IDs `(r*73+i*31)%1024`, except i%8=5 uses r%4; residual LCG seed
`97+i*7919`, update `1664525*seed+1013904223`, with the original i%8 amplitude
recipe. The 20 added LN stress fixtures preserve the deterministic
`redesignCases(64)` recipe and inherit fixture0 weights. No fixture was replaced.

## C. Error attribution

`probability_attribution_test.cpp` uses independent analytic `one::mm`,
LayerNorm double/anchored FP32 references and stable Softmax. It does not call
Custom kernel arithmetic. All inputs/weights are the same frozen FP32 values.
The double model uses the source FP32 scale scalar promoted to double, rather
than substituting a higher-precision scale. R0 retains double intermediates;
the FP32 model rounds at the same previously registered tensor/math boundaries.

| Path | Definition | P maxAbs vs R0 | Maximum P RMS | Row error |
| --- | --- | ---: | ---: | ---: |
| R0 | Ideal double LN1→QKV→QK→Scale→Mask→Softmax | 0 | 0 | 8.88e-16 |
| R1 | Correct double masked scores rounded at FP32 input→Host Softmax | 2.50729039419362e-7 | 1.74789025423579e-8 | 2.62392859440297e-7 |
| R2 | Full upstream FP32 through ordered QK; ideal double downstream | 9.21525585839689e-7 | 9.48624409796836e-8 | 1.11e-15 |
| R3 | R2 plus FP32 Scale; ideal double Mask/Softmax | 9.79111195009708e-7 | 1.00515917603270e-7 | 8.88e-16 |
| R4 | Full FP32 masked scores→double stable Softmax | 9.79111195009708e-7 | 1.00515917603270e-7 | 8.88e-16 |
| R5 | Original full Host FP32 model | **1.00750624248969e-6** | 1.00303759021279e-7 | 2.53864868682285e-7 |

R0 is the reference, not a proposed implementation. The worst relative errors
with denominator `max(|P_ref|,1e-12)` are respectively 0, 8.93042e-7,
8.31716e-6, 8.25090e-6, 8.25090e-6, 8.26355e-6. Each path's row distributions,
max logit/top2 gap, near-zero (<1e-8), near-one (>1-1e-6), zero counts and
all 25-gradient metrics are stored privately. There was no allowed-position
underflow. Future entries underflow to zero under the unchanged finite mask;
no clipping or explicit replacement with zero is performed.

R4 vs R5 isolates the Host Softmax arithmetic on the **same received FP32
masked scores**: maximum 2.46768375955497e-7 (fixture23/head0). Correct-score
input quantization alone gives max P difference 1.18269157467488e-7. On the
same FP32 inputs, normalized exp-output rounding contributes a counterfactual
max1.27051681397816e-8; increasing-column FP32 denominator accumulation adds
max2.29773871707195e-7; final division rounding adds max2.98011099131301e-8.
These maxima occur at possibly different positions and are not additive.

**Host model limitation:** the inherited `one::softmax(fp=true)` computes the
shift and exp in double, then rounds exp to float, accumulates the denominator
in float and divides in float. Its decomposition was checked word-for-word
against that model. It is not a strict FP32 exp oracle, a DSP exp accuracy
measurement or evidence about standard HTP Softmax internals.

At the sole P failure (21,1), the row1 top2 gap changes from
.136950612998282 (R0) to .136946909585731 (R2), then .136946678161621 (R3/R4).
At row1,col1, P is .465815758517393 (R0), .465816680042979 (R2),
.465816737628588 (R3/R4), and .465816766023636 (R5). The final local change
2.83950e-8 is enough to cross the 1e-6 cumulative gate, though it is small.
Row1 allowed scores are about12.7 and12.56, with a competing probability near
.5; the future scores are near -10000 and do not drive this failure.

The first nonzero divergence is already LN1 Y. At (21,1), LN1 Y maxAbs is
1.78862e-7; Q/K errors are 1.66915e-6/2.44840e-6; cumulative raw QK error is
3.79479784555770e-5. Local ordered QK error against the **received** FP32 Q/K
is 3.01880984636682e-5. The LN-only counterfactual P error is 4.16921e-7;
FP32 QKV on rounded correct LN gives 5.43940e-7; actual FP32 Q/K with double
QK gives 4.74671e-7; ordered QK gives R2's 9.21526e-7. Thus upstream rounding,
including ordered QK accumulation, dominates this cumulative failure; blaming
Softmax alone would be incorrect. These are counterfactual paths, not a unique
additive attribution to independent stages.

Scale's local max score error across the set is 4.76768e-7. Mask's local max
score error is 4.88229e-4 at finite future entries, consistent with FP32 spacing
near -10000. R3 and R4 probabilities agree on this set: that future-score
rounding does not change the critical distribution. A large masked-score
maxAbs does not establish a large probability error.

## D. Sensitivity bounds and provenance

With reference P, actual P+e, fixed dP, `a=ΣdP_j P_j`, `t=ΣdP_j e_j`,

`ΔdScores_i = e_i(dP_i-a) - (P_i+e_i)t`.

Taking the triangle inequality gives the existing componentwise bound
`|e_i||dP_i-a| + |P_i+e_i|Σ|dP_j e_j|`.
The old `impact()` is mathematically consistent with this expression.
All sums run in increasing column order, separately per row/head, in double.
Independent loops checked the signed identity, the bound, and
`ΔdValue=eᵀdContext` with its sum-of-absolute-products bound for **13,200
path/head/gradient combinations**. Double identity tolerance1e-10 was used only
for the algebra check, not as a relaxed P/gradient gate. No bound/indexing bug
was found.

| Evaluation | Signed P-only dScores error | P dScores upper bound | Actual Host FP32 dScores error | P dValue upper bound | Actual Host FP32 dValue error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Cartesian R5 | 2.90578493277849e-4 | **4.64227603470169e-4** | 2.87715531356980e-4 | 6.04851550175399e-5 | 4.50467220550199e-5 |
| Same frozen fixture R5, 24×2 | 1.22118430283535e-5 | 2.53419586634975e-5 | 1.38264352784745e-5 | 3.39481663783728e-6 | 7.03007068381112e-6 |

Signed P-only perturbation holds all other arithmetic in double. Actual Host
errors include the FP32 consumer's multiply/reduce/subtract/divide rounding;
the P-only bound is not claimed to bound that additional arithmetic. Even R0
has actual FP32 dScores error5.89401e-5 because its consumer rounds.

At worst-bound (16,1,9), position6,2 has e=-4.81707928767783e-7,
P=.366726671559690 and dP=-599.425720214844. The two absolute bound terms are
2.53941022110489e-4 and2.10286581359680e-4. Signed t=1.70027873508722e-4,
while Σ|dP e|=5.73415772254178e-4: their ratio is .296518. The bound is
2.42306× the actual signed error at that element (also the global signed
maximum for this combination). Actual Host FP32 error for that pair is
1.93222103376911e-4. Other cases also cancel between the two signed terms;
for (21,1,9) the bound maximum is not the actual-error maximum.
The e/dP normalized dot and signed terms are retained per combination.

The maximum bound and maximum actual error are **different combinations**.
The bound4.642e-4 exceeds its **allocated .00025 budget**; actual error2.877e-4
remains below the **.002 downstream gate**. Both statements are required.
R2's bound already reaches4.20771e-4; R3/R4 reach4.70722e-4. Ideal Softmax
alone therefore cannot repair the existing propagation-budget failure.

Provenance matching uses the original24 fixture indices only. The fixture
generator saves layer input, LN1, Q/K/V, parameters and deterministic gradient
recipe from one Forward trace per index. The LN1 parameter companion validates
a hash anchor over the same layer input/LN1 before attaching gamma/beta.
Thus Forward i / gradient i for i<24 is a justified **same frozen state**
diagnostic, not nearest-neighbour pairing. Synthetic LN cases24–43 and the
extra gradient24 have no such match and are excluded from this subset.
This is not an actual loss-derived training trajectory: the upstream gradients
are fixed diagnostic recipes. **Actual training-trajectory matched evaluation
is unavailable.** No checkpoint was decoded to extend the set.
Matched sensitivity is much smaller, but the same P element failure persists,
and the Cartesian FAIL remains part of the original robustness contract.

The maximum signed **P-only** dScores perturbation over all original R5 Cartesian
pairs is2.90578493277849e-4 atForward21/H1/gradient9, already above the2.5e-4
allocation. This is a different combination from the largest conservative bound
(Forward16/H1/gradient9). Thus bound conservatism alone cannot explain away every
allocation failure. R4, with independently evaluated ideal double Softmax on
the actual FP32 masked input, still has signed P-only error2.91163815433038e-4;
R2 already has2.85016061980059e-4. Merely tightening the bound or improving exp/
normalization cannot certify the existing upstream path under the old allocation.
These are original Host counterfactual results, not new DSP backward executions.

## E. Contract judgment

The original gates are unchanged: P1e-6; double row sum2e-6; P-induced dScores
bound .00025; P-induced dValue bound .001; actual dScores .002; actual dValue
.01; finite/nonnegative/causal invariants. **HOST_P_JOINT_CONTRACT_FAIL**.

The original1/8 dScores and1/10 dValue allocations reserve consumer/other-input
rounding headroom. They guarantee a conservative P-error contribution bound
for the frozen Cartesian inputs, not a measured training-quality threshold.
The uniform P box is insufficient; the componentwise check is necessary under
that registered contract. No defect in the joint-contract formulas/reference indexing was found that
would justify changing the old FAIL. A separate receipt qualification fix is
documented in J; it does not explain or remove the original Host nine. The Host exp model limitation is
explicitly retained, not silently replaced by another reference.

Three distinct answers: Host local Softmax on correct FP32 masked input is
within element/row/invariant requirements; full cumulative Host P fails the
joint contract; actual tested downstream Host gradients remain within their
existing gates. No corresponding DSP claims follow from Host results.

No V2 contract is adopted or numerically selected here. A future
`P_CONTRACT_V2_PROPOSAL` would need a justified error allocation, a bound for
the joint P/consumer arithmetic and other saved-state errors, independent
fixtures and prospective registration. Same-fixture sensitivity may motivate
that investigation but cannot retroactively replace Cartesian robustness or
choose a threshold fitted to these observations. First prioritize cumulative
upstream accuracy research under the existing contract.

## F. Device resume and actual DSP results

The user manually unlocked/reconnected the original physical device and explicitly
authorized data-preserving audited APK replacement. The existing install helper
verified installed bytes without app launch, input, data clear, permission/power
change or focus takeover. Historical power/keyguard/APK stops remain in the
chronological evidence. The old47-op combined pilot's exit139 during a standard
Mask nonfinite stress remains a failed pilot, not a Scale certification.

The49-op pilots established standard Mask/Softmax failures and justified bounded
Custom diagnostics. The51-op Scale/Mask pilots then completed, followed by a
read-only `ADB_TRANSPORT_FAILURE` before the next process. No probe or device
operation/retry followed that stop until the user explicitly confirmed manual
reconnection/readiness. This was transport failure, not a numerical FAIL or proof
of sleep. A new complete preflight passed on the same physical identity.

The final51-op binary was never rebuilt during the final cohort. Scale, Mask and
P diagnostics each ran in **three independent processes**, all with freshly
passing complete preflights and immediate pre-execute awake checks. Same-device,
active-run absence, Host/device free-space minima, SDK/compiler/package/fixture/
protected identities, awake/non-target top and power/thermal gates passed.
All nine completed QNN API/teardown successfully, with required Custom outputs
finite, input mutation0, complete writes, poison/canary0, DSP receipts, fallback0
and focus takeovers0. Maximum battery temperature34°C and thermal status0 were
recorded before/after these processes; external power was present.

| Target | Current51-op measured result |
| --- | --- |
| Custom Scale standalone Observed/NoReceipt | PASS, core maxAbs4.76768263979e-7; all56 cases ordered FP32 words match |
| Native LN1→HVX QKV→Heads→QK→Scale, four observer modes | PASS, maxAbs6.50802195956e-6≤.01; all prior stage gates and cross-mode final words pass |
| Standard Scale negative | H0 max.00868434926008,0/44 FAIL; H1 max.0102382000946,1/44 FAIL |
| Standard Mask Add |88/88 core heads FAIL,176 absolute/word predicates; whole-matrix maxAbs4.0018029213 |
| Reused Custom Mask standalone | PASS, core maxAbs.00048828125≤.001; all56 cases finite/ordered words/complete writes |
| Native LN1→Mask, four modes | PASS, allowed maxAbs6.50802195956e-6≤.01; future .000490612061185≤.011 |
| Standard Softmax local negative, separate current-binary pilot |88/88 core element FAIL, maxAbs.00189736825901; core row error max.002197265625; all112 head/control outputs finite |
| Standard Softmax stress | all-control maxAbs1 and row error1;224 total local failed predicates; normal teardown |
| Custom Softmax standalone, both observer modes | PASS on all56 cases, local maxAbs2.77010371863e-7≤1e-6 |
| Native received-input local Softmax, three visible-input modes | PASS, maxAbs2.28490675491e-7; hidden NoReceipt/NoTap input is not directly measured |
| Native LN1-origin cumulative P, all four modes | **FAIL**, maxAbs1.00750624249e-6>1e-6; nine predicates per mode |
| Cumulative row/invariants | row error2.38418579101562e-7≤2e-6; finite/nonnegative/causal invariants pass |
| P-induced dScores / dValue bounds | .000461683381131073>.00025 **FAIL** / .0000604586857555339≤.001 |
| Host-conditioned actual dScores / dValue | .00028771553135698≤.002 / .0000450314963892917≤.01 |
| Native P→Backward execution | NOT RUN: required cumulative P contract fails |

Standard Mask's first word mismatch is allowed position(0,0) in all88 core heads.
Core outputs are finite, so the whole-matrix maxAbs is also a finite upper bound
for future-position error. A separate standard future-only maximum was not
logged; no exact partition maximum is invented. Custom future/allowed metrics
and all received-operand FP32 Add words were independently checked. The standard
large-magnitude control's nonfinite values remain failed negatives; they were
not changed to zero or included in required Custom finite PASS.

Standard Softmax's worst core case is16/H0, first effectively one-hot row. This
supports a normalization/exp-approximation hypothesis, not identification of
internal backend arithmetic or dtype. The source mask, scale bits, axis and beta
are unchanged. Current standard Softmax is an explicitly failed pilot negative,
not a claimed three-process standard certification.

Custom receipts independently replay max/shift/ordered sum/divide from received
DSP exponent words: arithmetic mismatches0. Maximum exp-vs-double error was
4.81254868178738e-8 on measured shifted inputs. There were8547 Host-exp word
differences across144 receipts per P process, recorded separately rather than
treated as a Host-libm word oracle. Final DSP `expf` is runtime resolved; its exact
implementation identity remains unknown. Measured fixture accuracy is established
without claiming universal correctly rounded transcendental behavior.

Each P process reports36 cumulative failed predicates: four observer modes ×
(one P element failure plus eight dScores allocation failures). Forward/head/
gradient identities are the same as the original Host nine; original values in
B–E remain unchanged. Device bounds are a separate assessment, not replacement
of the old contract. All8800 P/gradient combinations per process have passing
Host-conditioned actual gradient gates. These are Host arithmetic evaluated on
device-produced P, **not executed DSP backward consumer results**.

NoTap/NoReceipt final P words equal the observed variant's full words, but its own
ideal-LN1-origin comparison still fails. Thus observer independence is verified
as an equality property; **FORWARD_P_OBSERVER_FREE_VERIFIED is not granted**.
The35/40-node Tap incident remains REPRODUCED / CAUSE UNRESOLVED.

The NoReceipt/NoTap P graph has8 nodes,33 tensors,90240 native logical bytes and
149152 Host-buffer lower-bound bytes. These are logical counts, not allocator
peak/RSS or production scheduling. Softmax uses128 bytes of logical row-exponent
stack storage; generic graph scratch0 excludes implicit stack/frames/spills.
Observed Softmax math timestamps include internal receipt stores. No isolated
Softmax kernel speedup or cumulative-P performance improvement is claimed.

## G. Native P, ownership and saved state

The real native chain executed without Host intermediate injection:
reference external X/gamma/beta/weights → DSP LN1 → HVX QKV → lossless HeadSplit
→ ordered DSP QK → DSP Scale → reused FP32 Add Mask → Custom local Softmax.
Observed receipts match producer/consumer words, including Scale→Mask and
Mask→Softmax. Each variant's own final P was evaluated against ideal double
Forward. All preceding gates, saved LN1 xhat/inverse and local backward controls
passed; the cumulative P joint contract did not.

In NoTap graphs the internal Q/K/heads/raw/scaled/masked tensors are NATIVE.
Final P is the required diagnostic APP_READ output. Invisible NoReceipt/NoTap
intermediates are not represented as directly observed. This proves execution
and local/native integrity only within reference-fed external inputs; it does
not prove those inputs came from the actual19-layer Forward producers.

Source production per-head P is graph-owned [32,32] and APP_READ in the established
H>1 ABI. It feeds PV, Softmax backward and dValue=PᵀdContext (transpose0=true,
transpose1=false). Q/K feed raw backward; V feeds dProbabilities. Gate and saved
ungated context feed the separate gate gradient. Raw/scaled/masked scores can be
recomputed from Q/K but current graph ownership/lifetimes still apply.

Experimental same-ID NATIVE saved-P backward scaffolding passed the80-graph Host
construction audit. Its backward consumer execution/lifetime is **not tested**:
the required P contract failed, so no APP_READ extraction/reinjection or native
P→Backward experiment was performed. PV, Gate/Sigmoid, Wo/residual/LN2 and FFN
producer accuracy remain external dependencies.

Full LN1 dY still requires accurate dQ/dK/dV scatter, dWq/dWk/dWv, Gate reduce/
Sigmoid/dWg, all projection-input gradients and source-order Add aggregation.
The prior standard backward Head Scatter failure remains unresolved. Full
one-layer additionally needs accurate saved states, FFN/residual backward,
gradient materialization/registry and eight-microbatch accumulation.
No full1-layer or19-layer training certification follows.

## H. Regression and reproducibility

The current targeted Host suite rebuilt13 legacy FFN2/arithmetic/reference/shape
tests plus11 current tests: QKV2, QK/BackwardScale2, Forward Scale/Mask/Softmax3,
graph/bounds2, original P1 and attribution1. There were23 zero exits and the
unchanged expected original P exit4. The graph audit covered80 constructions.
Existing Scalar/HVX word, LN, HeadSplit, Multiply/Softmax-backward Host contracts
remain intact. Broad Host PASS does not override the targeted P failure.

Final-source broad Host passed353.6s; Qnn passed339.0s including required core/build-ID,
shape correctness, QNN-enabled APK and ABI/hash/path/2.47 audit. The existing
optional W8 library packaging was retained explicitly for this Qnn build, with
the pinned Hexagon root; Host APK bytes remained identical to the already
verified install. This did not select a production training backend. Optional SDK
inventory advisory did not replace required core checks. Final Fast completion
is independently validated in the public aggregate.

Fresh current-package device regressions all passed: oneHvx (Scalar/HVX FFN2),
oneQkvHvx (Scalar/HVX QKV), oneQkScaleA, oneQkScaleB and oneSoftmax (LN2/Softmax
backward, including inherited separate/fused controls). They used the unchanged
final package/probe, not historical results or Forward-produced P for backward.

Final-source Device profile passed87.8s. Fresh Muon W8 achieved114/114, maxAbs1.490116119e-8,
worst relative L2 4.40269854e-8, minimum cosine1, RPC success, finite outputs and
fallback=false. This is the separate existing synthetic W8 component's fresh
regression alongside this candidate, not a claim that the QHPI package implements
Muon or that a real training state was updated. Historical114/114 was not reused.

The Headless runner's opt-in `-RequireAwakeSafety` checks the study's strict awake,
known non-target top, power, battery and thermal conditions after build/install
immediately before execution, during long execution and after completion.
Device/W8 explicitly used it, with bounded180-second instrumentation timeout.
Seven synthetic unsafe conditions were rejected without device calls; Fast
passed. No UI/wake/policy/permission manipulation was introduced. Device/W8
recorded before/after guard PASS, thermal status0, battery temperature34°C for
both suites, and zero focus takeovers.

Each final variant has three complete independent processes and one ordered
numerical digest; timings, IDs, receipt counters and standard-negative values
are excluded. Tensor word hashes in these numerical records are FNV64; SHA256
below hashes the canonical ordered records, not exported raw tensor bytes.
Full per-element observer comparisons are independently checked inside each run.

| Variant | Processes | Ordered numerical record SHA256 |
| --- | ---: | --- |
| Scale |3| `995556389700a88f309c8f04b57f5b9390646f14e1d16b37e77a2094ece0d7d1` |
| Mask |3| `542349d60441d3bf81485512608794e7c10044d3fe0e42304c551d01530bfb46` |
| P diagnosis (cumulative FAIL) |3| `e08ed25c124bf1a10c629a8b18d3b5129de53395759bbfaaaf2d0f2f42ac3c82` |

Three original Host-only independent attribution executions have digest
`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7`.
The current candidate attribution log matches it exactly. Host reproduction,
device process reproduction and within-process observer word equality are three
distinct checks.

All five protected hashes and previous branch refs are checked at start/export/
closeout. The40k checkpoint remains
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
No protected checkpoint decode, corpus/order/cache, optimizer/codec/production ABI
change, training update or final-test execution occurred.
35/40-node incident: REPRODUCED / CAUSE UNRESOLVED.
6031: UNRESOLVED / DORMANT / WATCH.

## I. Final decision

**Prioritize upstream cumulative FP32 accuracy research** at LN1/QKV/ordered-QK
rounding boundaries. Correct local Softmax now executes on DSP, yet cumulative
P and the reserved Cartesian propagation budget still fail. Speeding up QK or
fitting a different Softmax cannot restore information already lost upstream.
Use higher-precision diagnostic counterfactuals to identify the necessary
boundary, then prospectively specify any new arithmetic/ownership contract.
Do not silently replace ordered FP32 with FMA/tree reduction or relax P gates.
A V2 allocation would require a separate justified, prospectively validated
proposal; none is adopted or fitted here.

| # | Question / answer |
| --- | --- |
|1| Original nine reproduced? Yes: original assessment exit4, attribution exit0; unchanged values and thresholds. |
|2| Concentration? Forward16/17/20/21/22, six Forward/head pairs; gradients9/12 for eight bounds, one independent P predicate. |
|3| Largest P growth? Upstream through ordered QK; R2≈9.215e-7 before downstream rounding. Counterfactual differences are not uniquely additive stage contributions. |
|4| Local Softmax error? Original Host model2.46768e-7; DSP Custom standalone2.77010e-7, native visible-input local2.28491e-7. Standard DSP fails. |
|5| Upstream/local separated? Yes: R0–R5, R4vsR5 and independent received-input DSP local references. |
|6| Bound conservatism? Worst original bound2.42306× same-combination signed maximum; cancellation ratio.296518 for ΣdP e. |
|7| Actual gradients? Original Host and current device-P-conditioned Host gradients pass .002/.01. Device backward on produced P is not executed. |
|8| Matched vs Cartesian? Same-frozen-fixture sensitivity is lower; actual training-state matching unavailable. Native same-fixture ds-bound2.48055e-5; Cartesian FAIL and P element FAIL remain. |
|9| Old contract? FAIL, nine original predicates retained. Native four-mode diagnosis also fails nine per mode. |
|10| Joint-contract formula/indexing bug? None found. A separate volatile receipt-copy issue was fixed and fully retested; it does not remove the nine Host failures. Old Host exp model remains documented. |
|11| V2? No adopted/proposed numeric threshold. Future theory, independent fixtures, prospective registration and safeguards are required. |
|12| Awake safety? PASS for all executed final/regression processes; earlier transport stop preserved and manually resolved. |
|13| New Scale DSP? PASS standalone/native/all observer modes, three processes. |
|14| Mask? Standard FAIL; source-exact reused Custom Add PASS standalone/native, three processes. |
|15| Standard Softmax? Local FAIL on current binary:88/88 core heads. |
|16| Custom needed? Measured failure justified it; bounded Custom strategy now locally PASS in three processes. |
|17| LN1→P cumulative? FAIL1.007506e-6 and dScores allocation; all preceding stage gates pass. |
|18| Observer-free cumulative P? FAIL under its own reference; final words equal observed. No precision VERIFIED promotion. |
|19| P→Backward native? Held at shape/ownership audit because required P contract fails. No reinjection shortcut. |
|20| HVX/W8? Fresh current-package FFN2/QKV and other Custom regressions PASS; Device smoke PASS; W8 new114/114. |
|21| Largest blocker? Upstream cumulative rounding and its Cartesian P-error allocation, not local Custom Softmax math. |
|22| Full1-layer missing? Accurate saved Forward producers, PV/Gate/output branches, scatter/projection/Gate gradients, LN1 dY sums, FFN/residual backward, materialization and accumulation. |
|23|40k→60k/200k? **NO-GO**, zero training updates; full1-layer/full19-layer correctness unverified. |

Classifications: **HOST_P_ERROR_ATTRIBUTION_VERIFIED**, **HOST_P_JOINT_CONTRACT_FAIL**,
**FORWARD_SCALE_DSP_VERIFIED**, **CAUSAL_MASK_DEVICE_VERIFIED** (Custom Add),
**SOFTMAX_P_LOCAL_VERIFIED** (Custom), **BLOCKED_HOST_P_BUDGET** and
**BLOCKED_STANDARD_SOFTMAX** (standard backend path). Current device safety is
not blocked. No cumulative/observer-free/backward-connected P certification is
granted; overall research status remains BLOCKED by the original numerical budget.

## J. Provenance and reproduction

Original committed Host analysis snapshot:
`748c67ce0cbd88739a0ff174284a75b2d7bccb5e`. Its initial execution HEAD and later
source-hash binding are both preserved. Original logs and three-process digest
are unchanged. Current targeted Host suite and final QHPI package/probes were
built from`65148e19237b162da14042b500892ef5716e9b8d`. All final numerical cohort
and five Custom regression execution HEADs are that same commit; their compiled
source/binary/fixture hashes match the same final build. The earlier
`ecbe619e860d4a897815d1d5268fe66b680df57b` candidate's binaries and completed
cohort are frozen separately; none of its device results substitutes for the
final-source executions.

The strict Headless guard originated in
`edc8f41a8eec6355a42dbff858d4dc85621648ec`; final-source Device/W8 execution
HEAD is`65148e19237b162da14042b500892ef5716e9b8d`. Its guard code is unchanged.
The receipt-only fix changed the DSP/Host packages; the probe remained identical. Report/exporter
edits after execution are not represented as device-tested arithmetic changes.
Final HEAD and clean status are recorded after the final report commit in private
closeout and the final response; a commit cannot contain its own hash.

QAIRT2.48.40.260702151143, QNN API2.37.0, HTPV81, Hexagon SDK6.6.0.0,
tools19.0.07, NDK26.2.11394342 are pinned. Compiler SHA256:

- Host g++: `5fa84c1460fd741371f730788ba3f994cf1d682be866fe7d3ed799e81844670a`.
- Hexagon: `039c2fc1840b06a3601fa63972d86d1b575d7a1afeae2f3504a1843df147c047`.
- NDK: `ce4566f8c00c9d36d2724bd4af6a2714d8daed1c09f2290d7c4751a766fbb7b3`.

Frozen input SHA256:

- one-layer inputs: `388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`.
- LN1 parameters: `7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f`.
- LN1 producer fixture: `ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580`.

Host reproduction: `scripts/test_forward_p_attribution.ps1 -Legacy` with explicit
pinned QAIRT root/Build ID. The runner preserves expected original exit4 separately
from analysis exit0 and writes to the new study directory. Host flags:
`-std=c++17 -O2 -ffp-contract=off -fno-fast-math`.
The new read-only preflight also requires explicit pinned Hexagon SDK/NDK roots;
it performs one attempt, writes partial results, and exits2 on safety stop.
No automatic retry, wake, device installation or bypass is included.

Private evidence: `build/reports/g1-forward-p-numerical-attribution-safe-resume/`.
Source/executable/log identities, all row distributions, repeat digests, partial
device checks, protected/ref snapshots and complete profile logs remain private.
The [public aggregate](results/g1-forward-p-numerical-attribution-safe-resume-summary.json)
is produced by`scripts/export_public_forward_p_resume_device.py` in complete-process
mode. It verifies original FAIL, bounds/gates/coverage, current Host identities,
all three same-binary independent processes per variant, full observer word
comparisons, API/teardown/finite/poison/input guards, actual current regressions,
passed process-specific preflights, Headless before/after guards, physical
identity and protected refs/hashes. Local PASS cannot overwrite cumulative FAIL.
The earlier explicit transport-stop export is frozen privately and remains a
historical checkpoint; it is not the current classification. The original
Host/safety-stop exporter remains separate and unchanged.
Raw tensors/checkpoints/corpus/device IDs/endpoints/logs/APK hashes/SDK binaries
and private absolute paths are excluded. No push, PR, merge or history rewrite.


### Resume after the initial closeout

A later user-authorized continuation started from clean HEAD
`387191b861ce98c5d88abad93b5f200402fddd55` on the same branch. It made one new
read-only preflight attempt in a separate private child directory. The original
preflight, Host attribution, candidate binaries and original nine failures were
preserved. No DSP/probe arithmetic, numerical contract or production code changed.

Same physical identity, active-run absence, package/probe/compiler/protected
identities, Host storage and the awake snapshot passed. The next gate stopped at
**TOP_RESUMED_UNKNOWN**, before checking external power, device free space or
installed APK. Offline inspection of the already-captured activity dump found
keyguard showing, while the required `topResumedActivity` record was absent.
There were cached ResumedActivity/focused-app records, but these do not satisfy
the current top-resumed contract while keyguard is showing. No fallback to those
records, unlock, input event, wake or further device query was performed.

Thus this continuation's stop is separate from both the original awake failure
and the preceding external-power stop. Awake alone does not establish a fully
ready device. A user was asked to unlock the device manually; a new device
attempt requires a subsequent reply and a fresh complete preflight. Device
executions and operations after this stop remain zero. Scale/Mask/Softmax,
HVX/W8 device regressions and the device cohort remain NOT RUN.

The preflight now accepts an explicit private `ReportRoot`, refuses to overwrite
an existing preflight record, and rejects report destinations outside ignored
`build/reports`. Two Host guard tests verify those early exits without device
calls. Four new fixture-contained exporter tests reject contradictory top state,
unknown awake state or false device-execution claims. The public aggregate keeps
the original stop and adds a distinct latest-attempt summary with no raw dump,
app names, device identifiers or private directory names. Existing Host/Qnn
results retain their original provenance; no new numerical certification follows
from this scripts/evidence-only change. Fast passed in63.1s for this continuation;
both Host guard tests and the four new exporter corruption tests passed.

### Resume after manual unlock

The user subsequently confirmed a manual unlock and authorized a fresh check.
This attempt started from clean `c5fe6cbea8523ece7acf2bb8703c598839482f91`.
Same physical identity, no active run, awake, top-resumed, external power/thermal,
device free space, candidate/compiler and protected identities passed. The next
gate stopped before execution at **INSTALLED_APK_IDENTITY_MISMATCH_OR_UNKNOWN**.
This resolves the observed lock-screen blocker for that snapshot, but not the
complete device preflight. No device operation followed this stop.

The local APK was re-audited on Host: pinned QAIRT/V81 library hashes, ABI,
Build ID and forbidden path/version checks passed. The original APK check did
not retain its hash readback, so this record cannot distinguish a genuine
installed-package mismatch from an unrecognized readback. It is not evidence
of a corrupt APK or an incorrect DSP numerical result.

The preflight now retains that readback privately and distinguishes parsed hash
mismatch from unavailable/invalid hash output. It accepts whitespace and hex case
as the shared repository hash checks do, without relaxing byte identity. Five
Host-only self-tests passed, covering exact/whitespace/case matches, a genuine
mismatch and invalid output. Fast passed in64.9s. No kernel/probe math changed.

The user was asked whether to use the existing data-preserving verified APK
replacement helper (`pm install -r`, no app launch/UI input) followed by a new
complete preflight, or permit read-only diagnosis only. That device action was
not performed while awaiting the reply, in accordance with the requested
safety-failure stop. The candidate still has zero device executions; Scale,
Mask, Softmax, HVX/W8 regression and device cohort remain NOT RUN. The Host
nine-predicate failure, original thresholds and training NO-GO remain unchanged.

### Authorized APK replacement and first numerical pilot

The user explicitly authorized data-preserving APK replacement and continuation.
A fresh preflight distinguished a genuine installed-hash mismatch. The existing
verified cached APK helper installed the audited APK with `pm install -r`;
installed hash then matched. No app was launched, no data clear/permission/UI
change occurred, and the installation monitor recorded zero focus takeovers.
A further complete read-only preflight passed all gates before the probe.

The original47-op candidate then executed `oneProbabilityScale`. Its partial
records contain passing standalone and native Scale checks, but the combined
pilot continued into the standard Mask negative control. Standard Mask failed
on core inputs. At the large-magnitude control it returned nonfinite output;
the strict diagnostic exit was followed by process exit139 without normal
context/device/backend teardown. **This is a failed pilot, not a completed
Scale certification or a final cohort.** Its binary and logs are frozen privately.

The next bounded candidate separates Scale, Mask and standard local Softmax
processes. Existing `allowNonfiniteControl` is used only for explicitly labelled
standard negative graphs, which retain FAIL and their original output words;
Custom graphs retain strict finite/poison/canary/input/API checks. The Mask
negative control is no longer able to abort Scale's otherwise separate process.

Because standard Mask failure was measured, the conditional mask path reuses
the existing **AddNoReceipt arithmetic body verbatim**. A paired diagnostic
wrapper calls it for H0/H1, shares one immutable source mask and owns a separate
receipt/counter. It never updates the old generic Add global counter. It adds
`ForwardMaskAdd`/`ForwardMaskAddNoReceipt` after the unchanged47-name prefix;
existing operation contracts and arithmetic bodies are preserved. Receipt5128
words contain3072 input and2048 output words plus8 header words; DSP marker is
`0x48383119`. Shapes are three `[1,1,32,32]` inputs and two such outputs.
The wrapper verifies the exact source-mask bits and rejects overlap/nonfinite
input. It is diagnostic-only and supports synchronous single graph execution;
no concurrent-execute or production integration claim is made.

Gates remain fixed: all44 core cases require local Mask maxAbs≤.001 plus full
FP32 Add word identity; native allowed/future budgets are .01/.011. All56
standalone cases require arithmetic identity, finite output and complete writes.
The existing large-magnitude stress is retained with separately reported double
error; unavoidable FP32 Add rounding there does not redefine the core gate.
Standard local Softmax uses all44 source cases plus the existing12 controls,
stable double reference on the same received FP32 masked input, element1e-6,
row2e-6, finite/nonnegative and source-conditioned causal invariants. At arbitrary
very large scores a finite -10000 mask need not suppress every future position;
the reference uses that exact expression, without substituting -inf or clipping.
Local Softmax success cannot cancel the original cumulative nine failures or
authorize P→Backward accuracy certification.

### Conditional local Softmax strategy

The49-op pilot completed normal teardown for Scale and Mask with no required
accuracy or word failures. The independent standard Softmax pilot then measured
**88/88 core head element failures**, maximum1.89736825901e-3 against1e-6.
All outputs were finite; row-normalization errors were around1e-3. The largest
element error was at the first, effectively one-hot row in several cases. This
is consistent with a normalization/exp approximation error, but final output
alone cannot identify the backend's internal operation or dtype. No native P
closure or standard local Softmax PASS is inferred.

This measured failure activates one bounded Custom strategy: FP32 max shift,
pinned toolchain `expf`, increasing-column rounded FP32 sum, then FP32 division.
There is no epsilon, clipping, extra normalization, mask replacement or fitted
coefficient. Library audit found `float expf(float)`, implemented by
`libc.a:expf.o` with alias `h_expf` in the V81/G0/PIC toolchain archive, SHA256
`9927efdb27bc776939f7f3228f96fca5e7c60ba311ba892fd92b318861391a85`.
The SDK library is prebuilt and its polynomial uses scalar floating multiply/
accumulate instructions. Our compiler's contraction-off flag does not redefine
that prebuilt transcendental implementation. No no-FMA/bit-exact exp claim is
made; existing ordered MatMul/Add/Multiply bodies and strict HVX are unchanged.
Final DSP symbol inspection found`expf` undefined (runtime resolved), not a linked
definition of the inspected libc archive member. The archive audit therefore
does not establish which implementation executes on the device. Actual DSP exp
accuracy must be independently measured from receipts; no library identity or
accuracy is inferred from the example/archive alone. Disassembly remains private.

`ForwardSoftmax`/`ForwardSoftmaxNoReceipt` append after the49-op prefix. Paired
receipt8328 words contain2048 inputs,64 row maxima,2048 shifted inputs,2048
exponents,64 denominators,2048 final probabilities and8 header words. Marker
is`0x4838311a`. Arithmetic replay checks max, subtraction, ordered sum and
division using the **received exponent words**. Host exp word differences are
reported separately from exp-vs-double error; Host libm is not assumed to be a
DSP bit oracle. NoReceipt has128 bytes of logical row-exponent stack scratch;
compiler frame/spills are separate and not predicted from this number.

An independent new Host reference models this strategy's FP32 shift/exp/sum/
divide boundaries. It does not replace the original R5 model or its nine FAIL.
Forty Host kernel calls passed, max probability error3.29406492416418e-7 against
double, with complete writes, immutable input, guard/poison checks and invalid
shape/overlap/nonfinite refusal. These are Host results, not DSP certification.

The next P diagnostic first requires Custom standalone local accuracy, then
diagnoses the native LN1-origin P in four observer conditions. Local received-
masked-input error and ideal-LN1-origin cumulative error are distinct records.
Every cumulative output is evaluated with all25 frozen gradients under the
unchanged allocations, with separately labelled **Host-conditioned** consumer
errors. No native backward graph is executed in that diagnostic. A cumulative
failure remains a nonzero exit and cannot be promoted to P precision closure.


### Historical first51-op cohort after manual reconnection

The user explicitly confirmed connection and unlock after the new transport
failure. Full preflight passed, then the current51-op standard Softmax negative
and Custom P pilot completed normally. Local failures0/upstream failures0/
arithmetic word mismatches0 permitted native P diagnosis, whose cumulative
failure36 prevented backward execution.

Final Scale/Mask/P cohorts each completed three independent processes on the
unchanged package/probe and reproduced their ordered numerical digests.
Five existing Custom device regressions, guarded Device smoke and fresh W8
114/114 then passed. That stage ended device work for postprocessing; the later
receipt source audit below triggered a distinct new-binary validation phase.
These outcomes superseded the
earlier NOT RUN statuses **for the executed targets only**, without erasing
historical failed pilots or safety stops. Full cumulative P, native backward on
produced P and full training remain unverified/blocked as detailed in F–I.

Reproduce with explicit pinned QAIRT/Hexagon/NDK roots and fresh private output
directories. Targeted Host runner preserves original test exit4. Device variants
are`oneProbabilityScale`,`oneProbabilityMask`,`oneProbabilitySoftmax` (negative
pilot), and`oneProbabilityP` (actual exit4 for cumulative budget FAIL). The generic
runner's own zero exit for an expected negative is not an accuracy PASS; the
allow-list validator checks the underlying process/summary.

Device smoke uses`verify.ps1 -Profile Device -RequireAwakeSafety`; W8 uses
`run_qnn_headless_tests.ps1 -Suite hvx-muon-optimizer-benchmark -TestMode BACKGROUND_CORRECTNESS -RequireAwakeSafety -TimeoutSeconds 180`,
both with pinned SDK arguments. The strict option adds read-only safety checks;
it never wakes, reconnects, changes notifications/permissions or takes foreground.


### Receipt qualification audit and final-source recertification

After the first complete51-op cohort, source review found a byte-copy path that
cast away volatile qualification from the shifted-score object. It now performs
a volatile value read into ordinary float`observedShift`, then copies that
ordinary object's bytes. This avoids relying on an access path that strips
volatile qualification; the [C++ draft cv access rule](https://eel.is/c++draft/dcl.type.cv#5)
explains the distinction. No observed arithmetic/receipt word mismatch motivated
this change, and it is not a correction of the original P-contract formula.

An exact source comparison proves the only package source change is that copy
in`forward_softmax_op.h`: arithmetic expressions, receipt layout, operation ABI
and all existing Scalar/HVX sources are unchanged. Nevertheless, new package
identities were built and all targeted Host tests, broad Host/Qnn profiles,
pilots, final independent-process cohorts and required device regressions were
executed again. The earlier cohort is archived separately and not reused.

Final source is`65148e19237b162da14042b500892ef5716e9b8d`; package hashes in A
identify this corrected source. Its independently recorded numerical digests
equal the prior candidate's digests, which is an observed equality rather than
transferred certification. The old nine failures remain exactly reproducible.
