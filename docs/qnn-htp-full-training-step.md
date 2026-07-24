# QNN HTP full training step

## Scope

This mode explicitly builds the numerical work of one bias-free two-layer ReLU MLP training step as a QNN graph for the QAIRT 2.48 HTP backend. QNN automatic differentiation is not used. CPU code still supplies mini-batches, calls `graphExecute`, hands the output weight buffers to the next step, reads results, and performs correctness/reporting work.

Validated stack: QAIRT `2.48.40.260702151143`, QNN Core API `2.37.0`, HTP Backend API `5.48.0`, V81 Stub/Skel, Android NDK `26.2.11394342`.

## Graph

Inputs are `X [B,I]`, `Y [B,O]`, `W1_current [I,H]`, `W2_current [H,O]`, and dynamic `learning_rate [1,1]`.

```text
Z1 = MatMul(X, W1_current)
H  = ReLU(Z1)
P  = MatMul(H, W2_current)
E  = ElementWiseSubtract(P, Y)
E2 = ElementWiseMultiply(E, E)
loss = ReduceMean(E2, axes={0,1}, keep_dims=true)
dP = ElementWiseMultiply(E, static_grad_scale)
dW2 = MatMul(H, dP, transpose_in0=true)
dH  = MatMul(dP, W2_current, transpose_in1=true)
mask = ElementWiseGreater(H, static_zero)
dZ1 = ElementWiseSelect(mask, dH, static_zero)
dW1 = MatMul(X, dZ1, transpose_in0=true)
W1_next = ElementWiseSubtract(W1_current,
                              ElementWiseMultiply(dW1, learning_rate))
W2_next = ElementWiseSubtract(W2_current,
                              ElementWiseMultiply(dW2, learning_rate))
```

`dH` depends on `W2_current`; it does not depend on `W2_next`. Loss has shape `[1,1]`, represents the mean over all `B*O` elements, and uses FP32 accumulation supported by the HTP ReduceMean kernel. `static_grad_scale [1,1]` contains `2/(B*O)`. The ReduceMean axes tensor is STATIC `UINT32 [2] = {0,1}`. The ReLU-backward zero tensor is full shape `[B,H]`; the mask is `BOOL_8`. The derivative rule is `H > 0`, otherwise zero.

## Tensor exposure

Correctness mode exposes `H`, `P`, `E`, `dP`, `dW2`, `dH`, `mask`, `dZ1`, and `dW1` as APP_READ outputs in addition to `loss`, `W1_next`, and `W2_next`. Benchmark mode exposes only the three required outputs; diagnostic values remain NATIVE intermediates. Inputs are APP_WRITE. The axes, gradient scale, and zero constants are STATIC.

## Weight handoff

The graph has no cycle and retains no automatic weight state. Two app-owned buffers per weight are used. Within an execute, current input and next output pointers are always distinct. Because `graphExecute` is synchronous, the two vector owners are swapped only after success; the former input buffer becomes the next output buffer. This is a true ping-pong binding with zero copied weight bytes per step. No unverified in-place alias is used.

## Lifecycle and execute count

The training graph is created and finalized once per run and reused for every mini-batch. One `graphExecute` performs one numerical training step. The previous retained baseline uses one forward execute and one fused-backward execute per step, while CPU computes loss, dP, and SGD. The new mode reduces this from two executes to one. Initialization, first execute, steady execute, handoff, and full-step timings are reported separately; performance claims use benchmark mode and not the CPU analytic reference work.

## Correctness details

Micro tests first validate MSE/dP and SGD using `ElementWiseSubtract`, `ElementWiseMultiply`, and `ReduceMean`. Full correctness compares every exposed tensor and next weight with the CPU analytic implementation. A BOOL mask mismatch is also checked against the HTP-produced `H`; this must be zero. CPU-versus-HTP mask differences caused by a MatMul value crossing zero are separately counted and accepted only when the corresponding CPU `|Z1| < 1e-3`; the same strict `> 0` rule is used on both paths.

Trajectory reports save deterministic weighted checksums, L2 norms, CPU/HTP maximum differences, and loss at steps 0, 1, 2, 5, 10, 20, 50, 100, and the final step. Teacher/student generation, initialization, batch order, and learning rate are shared with the existing MLP tests. Teacher-weight identity is not a success condition because hidden-unit permutation and ReLU scaling symmetries exist.

## Fallback and CPU boundary

Full-step mode fails immediately if any graph create/finalize/execute, loss, dP, gradient, or optimizer result is unavailable. It never substitutes CPU loss, dP, ReLU backward, linear backward, or SGD. `cpu_fallback=false`, `htp_loss_used=true`, `htp_dp_used=true`, `htp_optimizer_used=true`, and `htp_full_step_used=true` are emitted on success.

It is accurate to say that the numerical training step—forward, MSE loss, dP, linear backward, ReLU backward, and SGD weight update—runs in an explicit HTP graph. It is not accurate to say that training uses no CPU, that HTP automatically retains weights between steps, or that QNN performs automatic differentiation.

## Reproduction

```powershell
.\scripts\run_qnn_htp_full_step_tests.ps1 `
  -QairtSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702' `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Repetitions 3 `
  -Seeds 20260710,20260711,20260712,20260713,20260714 `
  -RunPerformance
```

The runner requires exactly one online ADB target, redacts the endpoint from persisted data, enforces its thermal guard, resumes successful run files, audits the APK against QAIRT 2.48, and does not add APKs, Qualcomm binaries, logs, CSV, JSON, or generated reports to Git.
## API trace evidence

Each local full-step run contains a fixed-line `api_trace_version=1` summary. The API trace records the actual return codes and resulting handle states observed by PhoneLM for provider selection, backend/device/context creation, full-step graph creation and finalization, and aggregated `graphExecute` calls. It is distinct from `htp_full_step_used`, which is PhoneLM's aggregate success decision.

The `*_symbol_library` fields are the basenames returned by Android `dladdr` for function pointers copied from the selected provider function table. They contain neither pointer addresses nor absolute paths. A basename identifies the shared object that owns the function pointer; it is not a counter of HTP hardware activity or utilization.

The bounded `qnn_callback_begin` block is different evidence: its message text is supplied by the Qualcomm QNN runtime to the registered callback. PhoneLM saves at most 64 messages, 32 KiB total, and 1024 bytes per message. This raw callback block remains only in local individual run files under `build/reports/.../runs/`; public CSV, JSON, `environment.json`, and `docs/results/` exports do not contain it.

Neither evidence source supports a claim that NPU utilization was a particular percentage. It establishes which QNN library/provider functions were selected, their observed API results, and what the runtime reported through its callback.
## Runtime callback modes and measured impact

The same debug APK exposes three full-step test modes. `QNN_HTP_MLP_FULL_STEP_BENCHMARK` keeps API counters enabled, disables callback capture before the callback hot path, and requests QNN `WARN`. `QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE` uses the same workload with bounded capture and QNN `VERBOSE`. Debug-only failure modes inject an effective failure after a successful QNN `graphExecute` call 37 or `graphFinalize`; actual QNN and effective results are separate trace fields.

A 2026-07-25 A/B test used alternating `A B B A` / `B A A B` order, 10 runs per condition and shape, 100 steps per run, 32–34 °C battery temperature, and thermal status 0. B increased median `graphExecute` by 248.4–370.7% and median full-step time by 56.8–192.6%. The callback/logging configuration is therefore the primary cause of the earlier traced-build performance regression. Counter-only A produced full-step medians of 1744.167 us (`8/128/128/64`), 6071.302 us (`8/256/256/128`), and 6382.708 us (`32/256/256/128`). Public aggregate results are in `docs/results/qnn-htp-full-training-step-2026-07/api-trace-ab-summary.csv`; raw runs remain untracked under `build/reports/qnn-htp-api-trace-ab/`.

The debug failure tests observed `attempt=38`, `success=37`, `failure=1`, and `first_failure_call=37` while `last_qnn_result=0` and `effective_result=-9001`. Finalize injection likewise retained the actual QNN result 0 separately from effective result -9002. A subsequent normal run had no injection state, no fallback, and succeeded. Counter-only 640- and 1280-step traces were both 46 lines and 1859/1862 bytes.
