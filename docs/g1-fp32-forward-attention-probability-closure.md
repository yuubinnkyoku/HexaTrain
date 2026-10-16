# G1 FP32 Forward Attention Probability Closure

Research base: `6a4ac12f454b39737918c12c7c93c978127b900f`.
Branch: `investigation/g1-fp32-forward-probability-closure`.
Status: work in progress; no new device certification yet.

## A. Source audit

The generalized training builder computes per-head rank2 `[32,32]` QKᵀ
(transpose0 false, transpose1 true), multiplies by STATIC `[1,1]`
`1.0f / std::sqrt(float(dimension / numHeads))`, bits `0x3e3504f3`,
then adds the shared STATIC `[32,32]` causal mask. The mask is `+0.0f`
(`0x00000000`) for column ≤ row, `-10000.0f` (`0xc61c4000`) otherwise.
It is finite addition, including future positions; no replacement by infinity.
Graph-owned `maskData` and `attentionScale` back the constants through graph life.

Pinned QAIRT MasterOpDef Softmax defines exponential normalization along the
specified axis and an optional positive beta. Production rank2 axis 1 maps to
rank4 `[1,1,32,32]` axis 3; beta remains FLOAT32 1. QNN Add broadcasts matching
trailing dimensions; the diagnostic expands only unit leading axes, with the
same 1024 row-major words. Saved P has the same shape as scores. Production P
is NATIVE except requested diagnostic outputs, and feeds Forward PV, Backward
`Pᵀ dContext` and Softmax backward. Its graph registry lifetime spans both uses.
No production builder is changed in this study.

## D. Probability gate preregistration (before device P measurements)

The gate is a conjunction, not a tolerance inferred from device results:

* Element maxAbs ≤ `1e-6` against independent stable double Softmax.
* Double row-sum error ≤ `2e-6`; every probability finite and nonnegative.
* For source core scores, future probabilities must be zero (the finite mask
  underflows), row 0 diagonal probability one, rowwise maxima respect logits
  up to the element tolerance. Stress cases use the finite-mask analytic
  reference; finite -10000 is not a universal strict-zero guarantee for arbitrary
  logits. Every row has an allowed diagonal and a positive normalizer.
* Additional componentwise P-error sensitivity bounds below must allocate at
  most `0.00025` to dScores (one eighth of 0.002), and `0.001` to dValue
  (one tenth of the prior 0.01 local gate). Remaining budgets cover consumer
  arithmetic. Actual downstream gradients must separately pass 0.002 / 0.01.

For reference P, actual P+e, frozen dP, and dContext, let
`a = Σ_j dP_j P_j`, `b = Σ_j |dP_j e_j|` in each row. The exact bound on
the Softmax backward change at i is
`|e_i| |dP_i-a| + |P_i+e_i| b`.
The dValue bound at column c/output k is `Σ_r |e_rc dContext_rk|`.
These bounds are checked on observed P errors against the fixed 25-case
frozen gradient distribution. They control the actual distribution, not
arbitrary future gradients. An element-only uniform box bound is too loose:
Host audit measures D=1170.1279296875, L=13043.1827430725,
V=344.296861231327, giving a 0.01538345165 dScores bound at 1e-6.
We therefore explicitly reject treating the element gate alone as sufficient.
`probability_contract.h` fixes the joint contract; Host perturbation tests
check the inequalities independently of device outputs.

Mask local gate: allowed values must preserve received input words (addition
of +0, with source +0 convention); future values must match FP32 Add words
and have maxAbs ≤ 0.001 against double addition. The blocked threshold exceeds
half an ULP near -10000, while allowing no lossy rounding of unmasked logits.
Native cumulative mask error additionally inherits the 0.01 scaled-score
budget at allowed positions, and ≤0.011 at future positions; downstream P
must still independently satisfy the stricter joint probability contract.
Arithmetic stress is separated from the 44 source-derived accuracy cases.

## I. Initial provenance

The starting tree was clean; no upstream was configured; 102 local commits
were absent from known remote refs. Previous research refs are preserved.
Initial read-only physical-device, active-run, awake, top-resumed, thermal and
protected-artifact checks passed. Private records stay under the ignored
study report directory. These checks are repeated before every device run.

Training remains NO-GO for 40k→60k and 200k. Backward Head Scatter remains
an independent unresolved precision frontier. The old 35/40-node Tap incident
remains REPRODUCED / CAUSE UNRESOLVED; 6031 remains UNRESOLVED / DORMANT / WATCH.
