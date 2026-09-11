# HTP-native Muon v1 investigation

## Verdict

`HTP MUON BLOCKED`

The QAIRT 2.48 HTP V81 graph executes the required batched operations with
finite outputs and no CPU fallback, but its five-step Newton–Schulz result does
not meet the frozen CPU-oracle parity gate. Training integration, performance
promotion, and trajectory/resume device tests were therefore not run.

## Baseline and oracle

- Baseline: `eb90a42` on `main`, equal to `origin/main` at task start.
- Immutable CPU oracle: `4a0f343`, specifically
  `nicopedia_muon::zeropowerNewtonSchulzFp32()` and
  `nicopedia_muon::update()`.
- QAIRT root/version: fixed 2.48.40 SDK; Build ID
  `2.48.40.260702151143`.
- Backend: `libQnnHtp.so`, V81 Stub/Skel, QNN API 2.37.0.

## Implemented validation design

The registry-driven host packer validates all 114 `ParameterRole::MUON`
matrices and preserves registry identity:

- square branch: `[76, 64, 64]` for Wq/Wk/Wv/Wo;
- rectangular branch: `[38, 64, 128]`, with W1 followed by transposed W2;
- per-matrix aspect scale: `sqrt(2)` for the first 19 W1 matrices and `1` for
  the 19 transposed W2 matrices.

Sentinel tests cover layer 0 and layer 18 W1/W2 packing and unpacking. A small
CPU refactor exposes `updateAuxiliaryAdamOnly()` while sharing the original Adam
formula; the existing combined CPU Muon result remains covered by its host
regression test.

The diagnostic HTP optimizer is one QNN graph with independent square and
rectangular branches and one `graphExecute` per update. Momentum, Nesterov,
per-matrix normalization, five Newton–Schulz iterations, aspect scaling,
learning-rate scaling, and weight subtraction are all graph operations.
Inputs/outputs and native tensors are declared FLOAT32. State and parameter
transfer are intentionally host ping-pong; Aux Adam remains on CPU.

To avoid loss when squaring approximately `1e-4` Nesterov values, normalization
uses the algebraically equivalent expression

```text
(1024 * X) / (sqrt(sum((1024 * X)^2)) + 1024 * 1e-7)
```

with axes `{1,2}` and `keep_dims=true`. This preserves the frozen mathematical
normalization while moving the squared input into a safer dynamic range.

APP_READ outputs are poisoned before execution. QNN success, poison removal,
finite outputs, and absence of fallback are checked independently.

## Capability result

The single diagnostic graph exercised rank-3 MatMul, transpose-input MatMul,
ReduceSum axes `{1,2}`, keep-dims broadcast, and rectangular batched MatMul.
Graph creation/finalization/execution returned success, all inspected outputs
were finite, and `cpu_fallback=false`. Focus takeover was zero in every run.

## Numeric result

The frozen final gate is maxAbs `<= 2e-3`, relativeL2 `<= 1e-3`, and cosine
`>= 0.99999`.

| Stage | maxAbs | relativeL2 | cosine | Result |
| --- | ---: | ---: | ---: | --- |
| normalized 64x64 | 0.0000696480 | 0.000831682 | 0.999999800 | diagnostic only |
| NS iteration 1 | 0.000230964 | 0.00215444 | 0.999997898 | fail |
| NS iteration 5, 64x64 | 0.0167918 | 0.0522179 | 0.998637126 | fail |
| NS iteration 5, 64x128 | 0.0121972 | 0.0526525 | 0.998615861 | fail |

The original unscaled normalization had relativeL2 `0.140127`; the power-of-two
pre-scale reduced this to `0.000831682`. Error then grows across Newton–Schulz
MatMul iterations. Explicit graph FLOAT32 and FLOAT32 plus precision
compensation produced identical metrics. This localizes the remaining failure
to the HTP numerical path used by the iterative polynomial; it is not a QNN
return-code failure, nonfinite result, fallback, transpose, or per-batch norm
axis failure. The threshold was not relaxed.

## Numeric root-cause investigation

Host-only Newton–Schulz diagnostic
(`host_tests/nicopedia_muon_numeric_diagnostic.cpp`, wired into
`scripts/run_host_tests.ps1`):

- Frozen square input: generator
  `makeValues(0.02/0.001/0.0002, phases 1/2/3)` plus validation-style
  Nesterov accumulation and direct normalization. SHA256
  `62749421f6dbcba569c4f226e4561bab22e2c58eba3a990e05025fb29a859e4f`,
  FNV1a64 `70a2b57657bea1a0`, min `-0.03859590366`, max `0.03901480883`,
  RMS `0.01562482438`, Frobenius `0.9999887601`. Rectangular `[64,128]`
  probe is secondary only.
- CPU-double chain reproduces the frozen oracle bit-exactly at iter1/iter5.
- One NS stage decomposition (CPU float vs CPU double): `A` maxAbs
  `1.49e-08`/relL2 `1.30e-07`, `A2` `2.61e-08`/`1.62e-07`, `B`
  `7.45e-08`/`2.19e-07`, `BX` `4.47e-08`/`1.57e-07`, `X1` `4.47e-08`/
  `2.44e-07`. No stage exceeds the frozen gate. The float-vs-double X1
  gap is about 8841x smaller than the observed HTP iter1 gap
  (`2.15e-03`).
- Controlled MatMul probes (`K=16/32/64/128`, positive/cancellation/wide
  range/Muon range/identity-like): CPU float-vs-double relL2 is
  `~1e-07`; FP16-quantized-input-plus-double-accumulate relL2 is
  `~2e-05`–`5e-05`. Elementwise multiply is bit-exact, so the normal FP32
  multiply path is not suspect.
- FP16-accumulation emulation on the frozen input gives per-element A
  errors `~1e-05`–`3e-05`, the same order as the observed HTP behavior.
  An FP16-input-only simulation stays near `4.5e-05` relL2, about 47x
  below HTP iter1.
- Chunked float accumulation (`2x32`, `4x16`, `8x8`) changes CPU X1 only
  at relL2 `~1e-07`; all chunked iter1 references still pass the frozen
  gate. CPU-side chunking therefore cannot discriminate the HTP gap, and
  no broader chunking grid was run.
- CPU float-vs-double trajectory stays tiny through NS5: iter1 relL2
  `2.44e-07`, iter5 `9.50e-06`, cosine `1.0`. Observed HTP NS5 relL2
  `5.22e-02` is about 5500x larger. Intermediate magnitudes stay finite
  (`B`/`BX` Frobenius up to about `11.7` at iter4), so this is rounding
  amplification, not overflow.
- No algebraic rewrite was adopted: the diagnostic uses exact oracle
  ordering `A`, `A2`, `B`, `BX`, `Xnext`.

Classification: `LOWER_INTERNAL_PRECISION`. The HTP backend behaves as if
MatMul accumulation is below FP32 precision (FP16-like, error order
`1e-05`, not FP32 `1e-08`), and the NS polynomial amplifies that backend
rounding. This is not an `FP32_ACCUMULATION_DIFFERENCE` finding: CPU float
accumulation is thousands of times closer to the oracle than HTP is.

Evidence: ignored `build/htp-muon/numeric-diagnostic/` (`input.json`,
`stage-parity.json`, `matmul-precision.json`, `chunked-matmul.json`,
`ns-iteration-error.json`, plus frozen `.bin` inputs).

Recommended implementation path: investigate an alternate QNN/HVX/HMX
implementation path rather than chunked FP32 accumulation.

## Direct HTP NS iteration-1 stage localization

Diagnostic-only single-matrix stage probe
(`runHtpMuonNsStageProbe`, suite `htp-muon-ns-stage-probe`): the frozen
64x64 normalized input is bound directly as graph APP_WRITE and
iteration-1 `X0`/`A`/`A2`/`B`/`BX`/`X1` are read back as APP_READ, plus a
standalone `A=XX^T` tap with the identical MatMul configuration. Exact
oracle order `A`, `A2`, `B`, `BX`, `Xnext`; no algebraic rewrite. CPU
`DOUBLE` accumulates each dot product in double (oracle semantics);
CPU `FLOAT` uses float multiply/add/accumulation in graph node order.
Production graph ABI is unchanged; the taps live in the diagnostic probe
graph only (the pre-existing optimizer-graph diagnostic outputs are
untouched).

Device run `htp-muon-ns-stage-default2` (single 64x64 matrix, default
precision config, fixed QAIRT `2.48.40.260702151143`, V81, one
`graphExecute`, QNN success, all tapped tensors finite, no fallback,
`focus_takeover_count=0`):

| Stage | HTP-vs-double relL2 | HTP-vs-float relL2 |
| --- | ---: | ---: |
| X0 | 0 | 0 |
| A | 0.0002079345704 | 0.0002079393907 |
| A2 | 0.0002179122034 | 0.0002179119150 |
| B | 0.0007035717033 | 0.0007035738525 |
| BX | 0.0004769197584 | 0.0004769171168 |
| X1 | 0.0013624770970 | 0.0013624764300 |

For reference, on-device CPU float-vs-double gaps are `A`
`1.24e-07`, `A2` `1.57e-07`, `B` `2.14e-07`, `BX` `1.56e-07`, `X1`
`2.40e-07`: HTP is about 1675x (A) to 5685x (X1) above the entire
float-vs-double reference gap. HTP-vs-double and HTP-vs-float are
identical to 6+ digits at every stage, so the HTP output is not closer
to the float reference.

Standalone `A` agrees bit-exactly with chained `A`
(`chained_vs_standalone_relL2=0`), so the first MatMul backend path is
deterministic within the graph; the gap is precision, not path
divergence. Frozen-gate reading per stage (diagnostic only, gate
unchanged): `X0`/`A`/`A2`/`B`/`BX` pass, `X1` fails
(`maxAbs=1.68e-04`, `relL2=1.36e-03`, `cosine=0.999999374`).

The observed X1 gap here (`1.36e-03`) is smaller than the earlier
optimizer-graph iter1 gap (`2.15e-03`) because the stage probe feeds the
normalized tensor directly and skips the optimizer graph's
momentum/Nesterov/prescaled-normalization prefix; the comparison is
internally consistent (HTP vs on-device CPU refs from the same X0) but
not a repetition of the full-graph number.

Input identity note: the device-regenerated X0 reports
`input_fnv1a64=8acce9e8c5728db4` vs the frozen host FNV
`70a2b57657bea1a0` (`input_identity_match=false`). The generator source
is identical; the difference is last-ulp `sinf` variance between the
x86 host and the ARM device libm, and HTP `X0` matches the device CPU
`X0` bit-exactly, so the stage deltas above are not an input-mismatch
artifact. The frozen host SHA256 `62749421...59e4f` input was reused as
specified; no new random input was introduced.

Classification: `HTP MUON MATMUL PRECISION CONFIRMED`, precision class
`FP16-like` (A-stage error order `1e-05`, matching FP16-accumulation
emulation; FP32 would be `1e-08`). This is a MatMul backend precision
finding at the first MatMul (`A` already ~1675x the float reference),
not an elementwise scaling/add/broadcast bug (`B` adds no new
discontinuity beyond the inherited `A`/`A2` error) and not a
normalization bug (`X0` is exact).

Evidence: ignored `build/reports/qnn-headless/htp-muon-ns-stage-default2/`
(`device-report.txt`, `status.json`, `activity-sampling.json`,
`apk-audit.txt`) plus `build/htp-muon/apk-audit-ns-stage-probe.txt`.
Training integration, 114-matrix work, seed sweeps, final_test, and
precision-flag grids remain not run.

## Performance, training, and resume

- Full `[76,64,64]` + `[38,64,128]` correctness: not run; single-matrix gate
  failed.
- Full optimizer performance and comparison with the historical 5685.84
  ms/update CPU result: not run.
- Training integration and 1/8/32/100-step trajectories: not run.
- Device cross-backend resume tests: not run.
- Checkpoint format/code: unchanged (`NPRTCKPTV4`); existing host checkpoint
  and resume tests pass.
- `final_test`, seed2 quality work, 8000-step training, and Full verification:
  not run.

## Evidence and next candidate

Machine-readable summaries and raw local runner reports are under the ignored
`build/htp-muon/` and `build/reports/qnn-headless/` trees. APK audits confirm
the fixed SDK hashes, arm64 ABI, expected Build ID, no 2.47 strings, and no host
SDK path in the APK.

The next useful experiment is a focused HTP MatMul/accumulation diagnostic for
the Newton–Schulz polynomial (for example, tap each iteration and compare an
equivalent decomposition supported by QAIRT), without changing the CPU oracle
or promotion threshold. HTP-resident state, graph fusion, and training should
remain deferred until this parity gate passes.
