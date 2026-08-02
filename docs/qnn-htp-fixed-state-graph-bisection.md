# QNN HTP fixed-state graph-prefix bisection

## Result

This investigation reached success path C. In two independent fresh headless
processes, the fixed E checkpoint full graph produced two canonical
`embedding_input_gradient` outputs during 100 executions of one finalized
graph. In the same full-first protocol, the subsequently created graph ending
at `lm_dembedding` was deterministic in both processes. The graph ending at
`lm_dinput` was deterministic in one and varying in the other.

This is a reproducible graph-variant/runtime-order switch, not an application
fix or a causal proof about the SGD tail. Each variant uses a newly initialized
Runtime, context, and graph in the same app process. Graph structure is
therefore confounded with context creation and execution order. The result
narrows the dependency to full-graph composition or process/backend history,
but does not establish a Qualcomm backend defect.

## Environment

- Device: nubia NX741J, Android 16 user/release build
- Security patch: 2026-05-01
- SELinux: Enforcing
- QAIRT: 2.48.40.260702151143
- QNN API: 2.37.0
- HTP architecture: V81
- ABI: arm64-v8a
- NDK: r26c
- Model declaration: `B1_T8_V32_D16`, one head/layer, FFN 32, FP32 tensors
- Source baseline: `1aa4e40eabf4a55d8c41b39f3737f58406e4df84`

The test used the existing Instrumentation headless runner. `MainActivity` was
never created or resumed and never became the top activity. No root, SELinux,
system/vendor, QAIRT SDK, Stub, or Skel source was changed.

## Minimal procedure

Build, audit, install, and execute the paired graph-prefix suite:

```powershell
.\scripts\run_qnn_headless_tests.ps1 `
  -QairtSdkRoot <QAIRT-2.48.40-root> `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Suite qnn-graph-bisection `
  -TestMode BACKGROUND_CORRECTNESS `
  -RunId bisection-01
```

Repeat with a new `RunId`; the script force-stops the package before each
Instrumentation invocation, making the app process fresh. The final audit
completed 14 paired processes and stopped when two finite full-graph positives
had been observed. One additional process was rejected by the numerical audit.

## Fixed manifest

The E checkpoint is reconstructed by the CPU reference from seed 1 after two
Adam steps. Every graph execution uses the same binary32 bytes:

| Buffer | Canonical SHA-256 |
|---|---|
| One-hot input | `d85d7d14ab07879ab62b29dc0be5eef0c51d29db0a1e6050b8d7ccb080bd00f1` |
| Target | `f1c1a960169be212ee9f4b5856b9add5b9f2dd5ff68b77ce962292d8e1c724cb` |
| Current parameters | `5674c9ecf8bcb785a4db27a73afb11e33aa22c23670301fbe7865692fa83b93b` |

The runner hashes APP_WRITE input and state before and after every execute.
Every primary run reported `app_write_hashes_unchanged=true`.

## Graph variants

All variants register tensors in the existing enum order and retain the same
node prefix. The shortened graphs necessarily change unused update tensors from
APP_READ/APP_WRITE to NATIVE and omit their output bindings. Each row is built
in a separate Runtime/context, in the table order.

| Variant | Terminal boundary | `lm_dembedding` | SGD update nodes | Next-parameter outputs |
|---|---|---:|---:|---:|
| `full` | `lm_output_projection_next` | yes | 24 | 12 |
| `stop_after_dinput` | `lm_dinput` | no | 0 | 0 |
| `stop_after_dembedding` | `lm_dembedding` | yes | 0 | 0 |

The tail consists of 12 update pairs: gamma1, beta1, Wq, Wk, Wv, Wo, gamma2,
beta2, W1, W2, token embedding, and output projection. Each pair multiplies the
gradient by the learning rate and subtracts the scaled value from the current
parameter.

## Primary observation

Each process executed the variants in the order `full`,
`stop_after_dinput`, `stop_after_dembedding`, 100 times each.

| Process | Variant | Unique DINPUT hashes | Hash frequencies | Max absolute difference | First different run |
|---:|---|---:|---|---:|---:|
| 8 | full | 2 | stable `98086e…`: 53; alternate `07d40a…`: 47 | 406.733734 | 54 |
| 8 | stop after DINPUT | 2 | stable `98086e…`: 97; alternate `6ff2e6…`: 3 | 566.478485 | 98 |
| 8 | stop after DEMBEDDING | 1 | `98086e…`: 100 | 0 | none |
| 16 | full | 2 | stable `98086e…`: 48; alternate `bede9f…`: 52 | 950.488281 | 49 |
| 16 | stop after DINPUT | 1 | `98086e…`: 100 | 0 | none |
| 16 | stop after DEMBEDDING | 1 | `98086e…`: 100 | 0 | none |

The complete 64-character hashes and counts are published in
[`process-results.csv`](results/qnn-htp-graph-bisection-2026-07/process-results.csv).
The two alternate full-graph hashes differ between processes; they are not one
shared alternate value. `stop_after_dinput` also varied late in process 8, so
removing the SGD tail is not a generally sufficient determinism condition.

In both finite full-positive processes, the first changing exposed tensor was
`embedding_input_gradient` at element 0. `token_embedding_gradient` and
`next_token_embedding` changed downstream beginning at element 128. Earlier
`transformer_output` remained deterministic.

All 4,200 executions in the 14 completed paired processes returned QNN code 0.
Every bound APP_READ tensor was audited: 31 for full, 18 for
`stop_after_dinput`, and 19 for `stop_after_dembedding`. APP_READ poison
residuals and nonfinite counts were zero in those completed processes. The
logical APP_READ poison alternated between finite `+/-1.1415926`; the vector
binding API still cannot provide a physical guard region.

An additional fresh process returned QNN code 0 for all 100 full-graph
executions but failed the numerical audit with 584 nonfinite APP_READ elements
(438 in `embedding_input_gradient`). The shortened variants were not executed
after that failure. It is published separately rather than counted as a
completed paired process.

## Incidence and reliability

Among 14 completed paired processes, within-process variability occurred in
2/14 full graphs (14.3%, exact 95% interval 1.8%-42.8%), 3/14
`stop_after_dinput` graphs (21.4%, 4.7%-50.8%), and 1/14
`stop_after_dembedding` graphs (7.1%, 0.2%-33.9%). In both full-positive
processes the later-created `stop_after_dembedding` graph was stable, which is
the repeated path-C switch. Including the separately published numerical-audit
failure, the full graph varied in 3/15 processes (20.0%, 4.3%-48.1%).

These counts do not separate graph structure from Runtime/context order. The
prior source baseline found 1/5 varying processes; an additional unchanged-HEAD
baseline found 0/5.

The 100 executions inside a process are repeated measurements, not 100
independent process samples.

## Measured facts and interpretation

Measured:

- fixed input, target, and current-parameter hashes matched in every execute;
- QNN returned success for all 4,300 final-audit executes, including the
  process rejected for nonfinite values;
- poison and stale-output audits passed throughout; nonfinite audits passed in
  14 completed processes and rejected one additional process;
- among 14 completed finite paired processes, the full graph varied in two; one
  additional full-varying process was rejected for nonfinite values;
- the subsequently initialized `stop_after_dembedding` graph was deterministic
  in both full-positive processes;
- graph variability also occurred with a stable full graph in other processes,
  showing that graph shape alone is not sufficient.

Interpretation:

- `lm_dembedding` was not sufficient to reproduce the variation in the
  later-created graph in either full-positive process, but it varied in another
  process;
- it supports
  `GRAPH_VARIANT_OR_RUNTIME_ORDER_DEPENDENT_EXECUTION_VARIABILITY`;
- no app-owned alias, size, reset, binding, or synchronous-lifetime defect was
  found;
- the remaining layer includes graph lowering, scheduling, scratch allocation,
  context creation order, or process/backend history.

Unresolved:

- `DRESIDUAL1` and `DINPUT_NORM` remain NATIVE, so `lm_dinput` is the first
  exposed candidate, not a proven causal node;
- the causal update node, output promotion, tensor type, or generation-order
  component inside the removed group is not isolated;
- variant structure is confounded with separate Runtime/context creation and
  execution order;
- detailed profiling was not enabled. It exposes timing rather than tensor
  values and can itself change lowering. The success condition was reached
  before profiling was needed.

## Public artifacts

The allow-listed aggregates are in
[`results/qnn-htp-graph-bisection-2026-07`](results/qnn-htp-graph-bisection-2026-07/).
They are regenerated by
`scripts/export_public_qnn_graph_bisection_results.ps1`. The exporter validates
the fixed hashes, QNN codes, poison counts, and exact input reports, and rejects
private paths, endpoints, log captures, APK names, private keys, and binary
artifact names.

No raw tensor, raw checkpoint, APK, raw logcat, device endpoint, Qualcomm
binary, or private SDK path is published.
