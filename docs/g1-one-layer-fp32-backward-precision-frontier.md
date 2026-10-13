# G1 one-layer FP32 backward precision frontier

## Preregistration

Base `665a88256d8afb806bc1d222802691b2c3a0e904`, clean tree, no upstream.
Branch `investigation/g1-one-layer-fp32-backward`. Local unpushed research is
authoritative. Previous numerical source45b4cee and exact frozen package/probe,
all old refs/evidence/checkpoints remain preserved.

Production source audit: generalized training builder, shared LN builder,
generalized execute materialization, training runner accumulation, parameter
metadata, CPU generalized trace. Actual G1 is T32/D64/F128/H2/head32 with
HEADWISE_G1_SIGMOID, scale1, causal additive mask -10000 and epsilon1e-5.
Both head selector/scatter MatMuls and sigmoid-gate forward/backward are included.
No one-head simplification and no production builder replacement.

Fixtures are frozen before device results:16 coherent synthetic single-layer
forward states (eight gradient recipes, ordinary and stressed Wo/Wv ranges),
8 read-only40k checkpoint parameter layer18 states. Artificial token rows and
external upstream gradients are used; no private corpus, training batch replay,
final test or checkpoint update. Recipes: normal, large8, small1e-4, cancellation,
mixed magnitude, repeated tokens/patterns, seeded random, zero. Each batch group
has eight distinct inputs. Full external upstream is retained without shrink.
CPU generalized forward supplies saved inputs; analytic double and independent
sequential FP32 references provide per-stage comparisons. Checkpoint-derived
saved state is CPU-produced, not evidence for a full19-layer HTP forward.
Frozen fixture SHA256:
`388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f`.

| Tensor / contract | Frozen absolute gate | Basis |
| --- | ---: | --- |
| LN Y/xhat/inverse | 0.002 | existing ordinary forward gate |
| LN dX/dGamma/dBeta, residual | 0.02 | existing LN gradient gate |
| dContext/dWo, projection dW, aggregate dWo | 0.03 | existing generalized parameter/MatMul diagnostic |
| Head selector | bit-exact local input words; abs0.002 global | selector must copy one feature without arithmetic; ordinary precision budget |
| Head gated dContext, dProbabilities, dV, dQ/dK | 0.01 | tighter than general gradient0.03; reserves error budget for projection/merge |
| Softmax product/dot/centered | 0.005 | bounded probability-weighted reduction; centered error passes through probability |
| Softmax score gradient | 0.002 | row Softmax Jacobian infinity norm <=0.5; reserves downstream dot budget |
| Gate product/dot/dlogits | 0.01 | same activation-gradient budget |
| dLN1 contribution | 0.02 | consumer LN input-gradient budget; LN1 itself remains gated by frontier |
| HTP upstream X/dY/context | 0.002/0.01/0.002 | ordinary activation / gradient producer controls |

These are bounded diagnostic absolute budgets, not guarantees for arbitrary
coefficient norms. Host references must meet them before device trials.
Record maxAbs, RMS, relativeL2, maxrelative(floor0.001), nonfinite, output words,
local(actual-input) versus global propagation error, and FP16 counterfactuals.
FP16 model agreement does not reveal actual internal dtype or accumulation.

Native/tap/fused LN2 and standalone/integrated outputs are compared on the same
fixtures/runtime. Only after a blocking operation is observed may one new Custom
operation type be added to refute it. Further blockers stop DSP expansion.
Accuracy-failing full Attention is not performance-promoted.

Host materialization uses the existing registry (layer18.wo, MUON,64x64) in shadow
storage. The exact production expression is copied: destination[j] += source[j]
*0.125f, microbatch0..7. Compare with independent FP32 ordered accumulation and
double analytic average, check every registry entry and immutable model state.
No optimizer function is called; training updates0.

Three independent final processes, identical binary/fixture/SDK; QNN API, finite,
kernel receipts, full writes, canaries, mutation, and observer effect are separate
gates. Pinned QAIRT2.48.40.260702151143/QNN2.37.0/V81, Hexagon6.6.0/tools19.0.07,
NDK26.2.11394342. No fast-math/FMA change/vectorization.

Initial readiness was BLOCKED_DEVICE:Dozing/top unknown. No device numerical run
was started. User reported manual readiness; guards must be rechecked immediately
before runs. Host20GiB/Device12GiB and protected40k SHA256
89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d remain required.
UNSTABLE guard remains set.6031 remains UNRESOLVED / DORMANT / WATCH.

No practical performance ceiling is defined: no BLOCKED_PERFORMANCE by impression.
Record setup, first/steady, kernel elapsed windows, copies, validation, logical
traffic/RSS and device conditions; runtime VTCM/DDR/power may remain unresolved.
Do not extrapolate micrograph wall to full-model throughput.
40k→60k /200k NO-GO. No all19-layer integration, production optimizer/backend,
checkpoint/data-order/SDK change, push, PR, merge or history rewrite.

## Bounded change after exploratory Device results

Source4e78f62 exploratory graph: all24 standalone and integrated-native LN2
segment gates passed, but integrated Attention had19 final-gradient gate failures.
At tap case0, dWq words differed from native (observer effect); further full
Attention execution stopped. No full Attention success is claimed.
The selector changed values: case0 dContext differs from its independent FP32
reference by <=2.53319740295e-7, while h0-selected differs by6.56604766846e-4.
By the triangle inequality this cannot be a bit-exact selection of actual dContext.
The predeclared selector copy contract is therefore blocked before gate multiply.

One new type, HeadSplit, copies the two T32/head32 columns with FP32 memcpy,
FLOAT32/flat4/direct/DDR_ONLY, receipt of2048 input plus both1024 output words.
No reductions, transpose arithmetic, broadcast expansion or HVX implementation.
The final independent control uses actual DSP dContext, standard selector
direct/Copy-native/tap/native-output-Copy variants. Only a reproduced selector
blocker permits HeadSplit → standard broadcast gate Multiply. Any next failure
stops Custom expansion. The old full Attention observer mismatch remains unresolved.

An independent HTP-produced LN2 segment includes CPU-fed LN1 output → Q/K/V
projection → head selectors → QKᵀ scale/mask/Softmax → contexts → sigmoid gates,
scatter/sum → Wo projection + input residual → Custom LN2. Its dY is produced
from external upstream, W2ᵀ, FF1 ReLU mask, W1ᵀ. LN1/embedding and language-loss
producers are omitted. Upstream projection q/k/v have ordinary forward0.002 gates
fixed before that lane's first device result. Compare X/dY/context to CPU generalized
saved state separately, then compare Custom LN2/MatMul to double using the actual
received X/dY/context. A failing upstream is not a LayerNorm kernel failure.
