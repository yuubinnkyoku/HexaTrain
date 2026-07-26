# QNN HTP fixed-state reproducibility

## Result

This study now reaches success path C. The initial headless test isolated an
intermittent full-graph HTP execution variation, and the follow-up graph-prefix
bisection identified a repeatable operational switch: in two fresh processes
the full graph varied, while the subsequently initialized graph ending at
`lm_dembedding` was deterministic. The `lm_dinput` boundary was mixed, and
graph structure remains confounded with Runtime/context creation order.
The full technical report is
[`qnn-htp-fixed-state-graph-bisection.md`](qnn-htp-fixed-state-graph-bisection.md).

The original study found that, with identical checkpoint, input, parameters,
and optimizer state, one of five fresh processes produced two canonical hashes
during 100 executions of checkpoint E. The first changing exposed tensor was
`embedding_input_gradient`, produced by `lm_dinput`; its maximum absolute
difference was 5431.98079. The downstream `token_embedding_gradient`
(`lm_dembedding`) and next token embedding changed afterward. Earlier exposed
logits, probabilities, dlogits, output-projection gradient, and Transformer
output remained stable in that run.

All QNN execute calls returned success, APP_READ poison residuals were zero, and
APP_WRITE input/state hashes were unchanged. The source full-graph runs did not
emit a separate nonfinite-element counter, so that count is `NOT_MEASURED`
rather than inferred as zero. Later runs emit it explicitly. This
evidence does not support a host vector lifetime, byte-count, alias, or reset
defect. It supports the narrower classification
`BACKEND_EXECUTION_VARIABILITY_FULL_GRAPH`. The backend's internal precision is
not inferred from FP32 tensor declarations.

## Fixed manifest and hashing

The version-1 manifest fixes seed 1, the four-batch round-robin schedule, and
`B1_T8_V32_D16`, one head/layer, FFN 32, FP32 tensor declarations. E, D, and L
are deterministic CPU-reference Adam trajectory checkpoints. Each records input,
target, current parameters, first and second moments, step and scalar settings.
The test reconstructs them deterministically in memory; persistent raw
checkpoint save/load is not implemented and no raw state is published.

Raw hashes are SHA-256 over every binary32 byte. Canonical hashes first encode
IEEE-754 binary32 little-endian, normalize negative zero to positive zero, and
normalize NaNs to `0x7fc00000`. APP_WRITE and APP_READ buffers use alternating
finite `+/-1.1415926` poison. Logical poison overwrite is checked; physical
guard regions are unsupported by the runtime vector binding API.

## Scope matrix

For every E/D/L snapshot, the standalone `OneHot^T @ dX` graph ran 100 times on
the same graph, 30 graph recreations in one context, 20 context recreations with
backend/device retained, and 10 full runtime/backend recreations. Every scope
had one raw and canonical output hash, zero repeat difference, zero nonfinite
values, zero poison residuals, and QNN success. CPU/HTP differences were small
and deterministic.

The full graph then ran 100 times per checkpoint in each of five fresh
processes. Four processes were stable. One process varied at E beginning with
`embedding_input_gradient`; its downstream embedding gradient and SGD result
also varied. A diagnostic graph change that kept `DEMBEDDING` native and exposed
a zero-add copy did not remove variability and was rejected. It changed graph
lowering and moved the observed variation, so it is not an application fix.

Fixed-state Adam outputs (`m_next`, `v_next`, corrected moments, denominator,
normalized update, update, and next weight) were stable for E/D/L. The
zero-gradient control kept inputs and moments unchanged, but HTP `x - 0`
differed deterministically from the host input by at most 0.000473618507; this
is reported rather than hidden.

The public aggregates are in
[`results/qnn-htp-fixed-state-reproducibility-2026-07`](results/qnn-htp-fixed-state-reproducibility-2026-07/).
They contain no raw checkpoint, endpoint, private path, callback, logcat, APK,
or Qualcomm binary.

## API evidence and limits

The local QAIRT 2.48 headers document `QnnGraph_execute` as synchronous. APP_READ
and APP_WRITE client buffers therefore remain live through its return, which
the implementation does. Detailed profiling and optrace can expose execution
and node timing, not intermediate tensor values. Value bisection consequently
requires graph recomposition with selected tensors promoted to APP_READ.
Profiling was not needed for the value result and was not enabled.

The follow-up does not identify a single causal node or prove a backend defect.
`DRESIDUAL1` and `DINPUT_NORM` remain native, and removing the SGD tail changes
output types/bindings and lowering as well as node count. The established result
is therefore graph-composition-dependent execution variability, not a
single-operation minimal reproducer.
