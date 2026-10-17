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

The unchanged, previously untested device candidate has compiled source
`ac5be47c155725a0a442da540ab0359b34602a88`:

| Artifact | SHA256 |
| --- | --- |
| DSP package | `45aebb0312f7d7f39cf01c61e9456cc3e87c70b73854db68c755ea2df6428d4c` |
| Host package | `480014e5f0259a04b8065c2e911930a0f19b41568a5fc5b0182adb9499c2b210` |
| one-layer probe | `06fce37163459854f2c6c5f7e6a31b91a64b098a304a5dbf57afbb7019d0abea` |

The last historical device-tested source remains
`b6794c4572d10c163acb5201e811546ac564243e`; its PASS is not a PASS for this candidate.
No production or DSP arithmetic was changed in this study.

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

## E. Contract judgment

The original gates are unchanged: P1e-6; double row sum2e-6; P-induced dScores
bound .00025; P-induced dValue bound .001; actual dScores .002; actual dValue
.01; finite/nonnegative/causal invariants. **HOST_P_JOINT_CONTRACT_FAIL**.

The original1/8 dScores and1/10 dValue allocations reserve consumer/other-input
rounding headroom. They guarantee a conservative P-error contribution bound
for the frozen Cartesian inputs, not a measured training-quality threshold.
The uniform P box is insufficient; the componentwise check is necessary under
that registered contract. No mathematical or implementation defect was found
that would justify changing the old FAIL. The Host exp model limitation is
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

## F. Device resume

One read-only preflight was attempted after Host attribution and Qnn completion.
Pinned package/probe/source/compiler checks, protected artifacts and Host
storage passed. Same physical device, absence of active training/headless/custom
probe, awake and non-target top-resumed all passed. **The old awake blocker
was not reproduced.** Thermal status0, healthy/present battery88%,35°C were
captured privately.

The new preflight additionally required confirmed AC/USB/wireless external
power. All reported false, so it stopped with
`EXTERNAL_POWER_UNKNOWN_OR_ABSENT`. This is an **additional conservative
condition introduced by this study's preflight**, not a previously mandated
repository requirement or proof that battery operation was unsafe. The healthy
battery/thermal readings must not be described as an unsafe-device finding.
The stop condition was retained, and there were **zero device operations after
the stop**, zero staging operations, zero probe executions and no automatic
wake/reconnect, force-stop, foreground takeover or power-policy change.

Device free space and installed-APK identity checks occur after the power gate
and were **not reached**. SDK/local APK/V81 Stub/Skel audit passed in Qnn;
installed APK correctness and complete device readiness are not claimed.
There was no immediate pre-probe awake recheck because no probe was submitted.

| Device experiment | Result |
| --- | --- |
| Custom Forward Scale standalone/native, four observer conditions | NOT RUN |
| Standard Scale negative control, new candidate | NOT RUN |
| Standard finite Mask Add | NOT RUN |
| Standard Softmax, correct masked inputs | NOT RUN |
| Conditional Custom Mask/Softmax | Not implemented; required standard failure unmeasured |
| LN1→P / observer-free cumulative P | UNVERIFIED |
| Final same-binary three-process cohort | NOT RUN, zero processes |

The existing `oneProbabilityScale` remains the first device step after a future
fully passing preflight and a clean committed source. Scale must pass before
Mask, then local Softmax. The current Scale probe includes a standalone Mask
diagnostic only after its own Scale gates; it is not a P executor. No unexecuted
Mask/Softmax graph scaffold is represented as a complete device experiment.

## G. Native P and saved state

The unchanged diagnostic graph construction uses same-ID P tensors for existing
Softmax-backward/dValue consumers; NoTap makes intermediate P NATIVE. Shape
checks cover axis, head mapping and declaration ownership. This is a static
construction audit only; runtime lifetime and numerical integrity are untested.

Production P is graph-owned per-head `[32,32]`, consumed by PV, Softmax backward
and dValue=PᵀdContext (transpose0=true/transpose1=false). Q/K feed raw backward,
V feeds dProbabilities; Gate and saved ungated context feed gate gradients.
Raw/scaled/masked scores are recomputable from Q/K but the current graph still
owns those tensors. Saved buffers must survive all consumers.

Because the required cumulative contract remains unresolved, **no P→Backward
numerical experiment was performed**. No P extraction/reinjection, hidden
NoReceipt measurement or native-connected certification is claimed. Reference
external LN1 inputs/parameters and frozen dP/dContext remain external boundaries.
PV, Gate/Sigmoid, Wo/residual/LN2 producers and FFN branches remain separate
unverified producer dependencies.

Full LN1 dY requires accurate dQ/dK/dV scatter, dWq/dWk/dWv, Gate reduce/
Sigmoid/dWg, Q/K/V/Gate projection-input gradients and source-order Add sums.
The existing backward Head Scatter standard failure remains unresolved. Full
one-layer additionally needs saved-state accuracy, FFN/residual backward,
materialization/registry and eight-microbatch accumulation. No full1-layer or
19-layer training claim follows from this Host assessment.

## H. Regression

The new targeted suite rebuilt/reran13 legacy FFN2/arithmetic/reference/shape
tests, two QKV tests, two QK/BackwardScale tests, three Forward Scale/graph/bound
tests, the original P assessment and the new attribution test. The20 ordinary
tests passed; original P test returned expected exit4; attribution analysis
returned exit0. Existing Scalar/HVX word, LN, HeadSplit, Multiply/Softmax-backward
Host contracts remain intact. These are Host contracts, not DSP execution.

Qnn profile **PASS382.8s**, including Host shape/contracts, pinned SDK check,
QNN-enabled APK and ABI/hash/path/2.47 audit. Optional inventory advisory remains;
required core and Build ID passed. Fast **PASS72.8s**. A separate broad Host
profile **PASS306.1s** was also run; its completed result is in the allow-list aggregate.
Broad Host PASS cannot override the targeted P joint-contract FAIL.

Device smoke, actual candidate HVX FFN2/QKV, Scalar QK, BackwardScale,
Softmax-backward and Muon W8 **DEVICE_REGRESSION_NOT_RUN**. No historical
114/114 or previous binary numerical results are substituted. There are no
device numerical digests or final device processes.

Three **Host-only** independent executions of the identical attribution binary
produced the same ordered numerical/log digest
`c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7`.
The log excludes timing/process/device IDs and includes deterministic fixture
IDs. This does not satisfy the required three-process device cohort.

All five protected artifact hashes and previous branch refs are checked at
start/export/closeout. The40k checkpoint remains
`89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d`.
No checkpoint decode, corpus/order/cache changes, optimizer/codec/production ABI
change, training update or final-test execution occurred.
35/40-node Tap incident: **REPRODUCED / CAUSE UNRESOLVED**.
6031: **UNRESOLVED / DORMANT / WATCH**.

## I. Final decision

**Prioritize upstream cumulative FP32 accuracy research**, especially the LN/QKV
and ordered QK rounding boundaries, before Softmax optimization or native saved-P
backward certification. A higher-precision diagnostic counterfactual can establish
what would meet the unchanged budget, but changing production mathematics or
ordered-FP32 contracts is outside this study. Local DSP Scale/Mask/Softmax
measurements remain necessary once a fully passing device preflight is available.
Host evidence does not establish any HTP Softmax failure or justify Custom exp.

| # | Question / answer |
| --- | --- |
| 1 | Nine old FAIL reproduced? Yes, unchanged original exit4 and independent attribution. |
| 2 | Concentration? Forward16,17,20,21,22; six Forward/head pairs; gradients9/12 for eight bounds, one P predicate. |
| 3 | Largest P growth? Upstream through ordered QK dominates the worst cumulative case; R2≈9.215e-7 before Softmax/Scale rounding. Counterfactual attribution is not uniquely additive. |
| 4 | Host local Softmax error? Max2.46768e-7 on received FP32 scores; DSP unknown. |
| 5 | Upstream vs local separated? Yes, R4 vs R5 and received-operand local checks. |
| 6 | Bound conservatism? Worst bound2.42306× same-combination signed maximum; cancellation ratio .296518 for ΣdP e. |
| 7 | Actual gradient gates? Tested Host dScores2.87716e-4<.002, dValue4.50467e-5<.01; DSP unmeasured. |
| 8 | Matched vs Cartesian? Same-frozen-fixture sensitivity is smaller and within allocations; P element FAIL persists. Actual training-matched evidence unavailable. Cartesian FAIL retained. |
| 9 | Old P contract? FAIL, nine predicates. |
| 10 | Formula/indexing bug? None found; bound identity checked independently. Host exp model limitation documented. |
| 11 | V2 needed? Potential future investigation, no adoption or fitted threshold. Requires theory, independent fixtures, prospective preregistration and regression safeguards. |
| 12 | Awake gate? PASS this attempt. Complete safety preflight stopped at added external-power condition. |
| 13 | Candidate DSP Scale? NOT RUN. |
| 14 | Standard Mask? NOT RUN. |
| 15 | Standard Softmax local? NOT RUN. Host local model is within local gate. |
| 16 | Custom Softmax needed? Unknown; no measured standard failure, none implemented. |
| 17 | LN1→P cumulative precision? Host original joint contract FAIL; device unverified. |
| 18 | Observer-free P? UNVERIFIED, no executed graph. |
| 19 | Native Forward P→Backward? Shape/ownership audit only; numerical certification withheld. |
| 20 | Candidate HVX/W8 device regressions? NOT RUN; historical results not reused. |
| 21 | Largest precision blocker? Cumulative upstream P error and its frozen Cartesian propagation budget; device local accuracy remains unknown. |
| 22 | Full1-layer missing? Accurate saved Forward producers, Attention/FFN branches, scatter/projection/Gate gradients, LN1 dY sums, complete materialization/accumulation and executed ownership tests. |
| 23 | 40k→60k/200k? **NO-GO**, zero training updates. |

Classifications: **HOST_P_ERROR_ATTRIBUTION_VERIFIED**, **HOST_P_JOINT_CONTRACT_FAIL**,
**BLOCKED_HOST_P_BUDGET**, **BLOCKED_DEVICE_SAFETY**. No DSP/local/native/cumulative/
observer-free/backward-connected P classification is granted.

## J. Provenance and reproduction

Committed analysis source snapshot:
`748c67ce0cbd88739a0ff174284a75b2d7bccb5e`. The first run occurred
with base HEAD and new uncommitted analysis files; the later committed snapshot
was bound against all nine test source hashes. Private identity records retain
both execution HEAD and the post-run source binding, rather than claiming those
files existed in the base commit. All inherited compiled source hashes remain
unchanged. Final HEAD/clean status are recorded after the report commit in private
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
is produced by `scripts/export_public_forward_p_attribution.py`, which validates
original FAIL, coverage, gates, bounds, source/binary/fixture/protected identities,
profiles and safety stop before selecting allowed scalar fields. It rejects nine
test corruptions and cannot emit a device VERIFIED classification in this mode.
Raw tensors/checkpoints/corpus/device IDs/endpoints/logs/APK hashes/SDK binaries
and private absolute paths are excluded. No push, PR, merge or history rewrite.
