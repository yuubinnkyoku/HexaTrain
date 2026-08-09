# Nicopedia HTP execution-path audit (L19 seed1 step1000)

Status: **MEASURED-2026-08-09**. The protocol and raw device evidence remain
under `build/private-diagnostics/` and `build/reports/`; only aggregate results
are published here.

Scope: Nicopedia L19 seed1, step1000, QAIRT `2.48.40.260702` build
`2.48.40.260702151143`, HTP V81. The checkpoint, model graph mathematics,
CPU reference, tokenizer, fixed parity inputs, and legacy parity gate were not
changed. No additional training was performed.

## Why this path was selected

The prior localization found gradual error accumulation from layer 0 through
layer 18, with the same shape but 1.6--2.9x smaller error at step320. Prior
precision, precision-compensation, weight-packing, and activation-fusion
configs were bit-identical over all 20 prefixes and eight autoregressive
steps. A single defective block, Softmax, ReLU, and instrumentation artifacts
had already been weakened.

The pinned SDK header exposes
`QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED`. Its documented
behavior is to split a graph by structure and compile each part independently.
It was therefore the highest-value remaining documented route that could
change compiler partition or kernel boundaries without changing model
mathematics. The interpretation was fixed before measurement:

- accepted and improved: repeat, add checkpoint/prefix controls, then run the
  unchanged formal 20-prefix gate;
- accepted and bit-identical: no numerical mitigation through graph splitting;
- rejected: record the exact API stage and error, and treat the route as
  unavailable in this environment.

## API delivery control

The established default still passes a null context-config pointer to
`QnnContext_create`. Explicit `false` and explicit `true` each pass exactly one
official HTP custom config wrapped by `QNN_CONTEXT_CONFIG_OPTION_CUSTOM`.
Runtime diagnostics independently record the requested boolean, non-null
pointer, config count, API owner, return code, context-handle state, graph
execution count, and CPU fallback state.

Explicit `false` is an important delivery control: if only `true` failed, the
failure could be feature activation rather than rejection of this context
configuration route.

## Measured result

| context graph-splitting request | context-create result | graph executes | CPU fallback | numerical result |
|---|---:|---:|---|---|
| unset, null config | success | 28/28 successful | false | unchanged baseline |
| explicit false | 5010 (`QNN_CONTEXT_ERROR_INVALID_CONFIG`) | 0 | false | unavailable before graph creation |
| explicit true | 5010 (`QNN_CONTEXT_ERROR_INVALID_CONFIG`) | 0 | false | unavailable before graph creation |

The default baseline was repeated with an identical full-execution
fingerprint. It retained the known `parity_13` errors: raw logit max absolute
error `4.281e-2`, centered logit RMS `2.235e-2`, and probability L1 error
`2.977e-2`. All 20 parity-prefix outputs and eight autoregressive-step outputs
were finite; all 28 HTP graph executions returned success; the CPU backend was
not initialized and no fallback was attempted.

Both non-default values were rejected by `QnnContext_create` before any graph
was created or executed. They therefore cannot be called numerically improved,
worse, or bit-identical. The result is instead strong negative evidence that
this documented partitioning control is not a viable online context-creation
route on the measured QAIRT 2.48.40 + HTP V81 environment.
Because no candidate graph executed, a tensor-finiteness result and execution
fingerprint do not exist for the rejected runs; their absence is not treated as
a numerical result. The rejection evidence is the exact API code, null context
handle, zero graph executions, and absence of CPU fallback.

## Interpretation and stopping decision

This result weakens an actionable monolithic-versus-split compilation
explanation: the exposed split control cannot produce an alternative executable
here. It does not prove that graph splitting is unsupported on every backend,
device, context-binary path, or SDK version.

An additional standalone MatMul/LayerNorm scale sweep was considered but not
run. PhoneLM already has standalone operator tests, and the stronger existing
evidence uses the real L19 graph, real step320/step1000 checkpoints, all fixed
prefixes, and fine taps through LayerNorm and MatMul-path intermediates. A new
standalone graph could select a different lowering and would be confirmatory,
not a better discriminator for this checkpoint.

Context-binary restore was also not run. The audited SDK API and samples make
it a real cache/restore route, but provide no documented precision or compiler
partition change for an already-finalized graph. After the only documented
partitioning option was rejected, a plain serialization round trip had lower
expected information than the evidence already collected.

The most supported explanation remains many small, deterministic,
magnitude-dependent backend rounding differences accumulating through 19
residual blocks. The HTP internal datatype or accumulator format is not
directly observed. The most informative next research stage is a controlled
QAIRT/backend-version comparison using the same checkpoint and unchanged gate,
rather than further enumeration of 2.48.40 memory, scheduling, or cache knobs.

The legacy parity gate remains FAIL, and step1000 HTP generation remains
BLOCKED.

## Privacy

Raw prompts, tokens, tensors, activations, logits, checkpoints, parameters,
device identifiers, ADB endpoints, execution fingerprints, and private run
identifiers are not included in this document.
