# QNN HTP generic Transformer depth/head resource envelope

## Outcome

PhoneLM now generates explicit Transformer forward, backward, and Adam update
graphs for generic positive layer and head counts. The CPU reference, parameter,
gradient and optimizer-state registries, QNN graph builder, diagnostic scopes,
shape validator, graph-map exporter, and application-visible resource estimator
use the same deterministic layer/head indexing. The embedding dimension must be
divisible by the head count.

PhoneLM executed the numerical operations of these explicit graphs on QNN HTP.
The application still owns graph construction, fixed-buffer handoff, control
flow, reporting, and the independent CPU reference. QNN autograd was not used.
This is not a claim that CPU was unused or that HTP internal arithmetic has a
particular precision.

QAIRT was fixed to SDK `2.48.40.260702`, Build ID
`2.48.40.260702151143`. The inventory command returned the advisory
`QAIRT_SDK_FOUND_INVENTORY_INCOMPLETE` because optional CLI tools and samples
were absent. The strict PhoneLM gate passed: the QNN Android build and APK
audit matched the fixed runtime, V81 Stub, and V81 Skel; no QAIRT 2.47 string
or host SDK path was present.

## Generic implementation

The canonical registry is token embedding first, followed by zero-padded layer
indices and the ten parameter groups within each layer, then output projection.
Every parameter has exactly one matching gradient, Adam first moment, and Adam
second moment. Backward construction iterates layers in reverse. Attention
temporaries are independently named by layer and head, while Wq/Wk/Wv remain
full `[D,D]` matrices.

Checked arithmetic covers parameter, gradient, optimizer, tensor, byte, node,
attention, and activation counts. Overflow, invalid divisibility, impossible
shapes, duplicate names/IDs, producer errors, registry mismatch, and unexpected
storage aliasing are rejected before graph execution. The conservative
estimator reports application-visible allocations only; it does not estimate
DSP/runtime-internal memory.

The established `ReduceSum keep_dims=true` Softmax-dot shape and LayerNorm
centered scale 8 apply to every layer and head. Adam is executed in checked
chunks of at most 32,768 elements. An observed Divide-input nonfinite boundary
from the former fixed algebraic scale was corrected by selecting a bounded
common numerator/denominator scale and by preselecting a safe denominator for
the zero-moment branch. No NaN/Inf replacement, learning-rate reduction,
epsilon increase, unexplained clamp, or CPU numerical fallback was added.

## Host and baseline regression

Host forward/backward/Adam tests cover L1/H1, L2/H2, L3/H2, L4/H2, L2/H4,
L3/H4, L4/H4, and L3/H8, including representative finite-difference gradients.
Negative tests cover invalid counts/divisibility, shape and binding errors,
aliasing, duplicate names, and resource arithmetic overflow.

The T32/D32/FFN32/L2/H2 baseline completed 5/5 finite seeds and retained the
established seed-1 legacy parameter hash
`c61d3f2b796773d2dba9eea219d9b3f12403ca00824f9f7b80c626b1bfcbd5cc`
and logits hash
`35da152a62fcb535ecf2d47e7860409c6f5fefbb29f1d30388685dd8b0c7599b`.
Seed-1 final loss was `0.1628002109`, accuracy was `0.984375`, and Oracle/free
generation were each 20/20.

## Device envelope

The largest formal combined configuration was
B1/T16/V32/D32/FFN64/L6/H8 (head dimension 4). It completed 5/5 finite
320-step seeds with Oracle and free-running generation each 20/20.

The other formal categories were:

| Category | Configuration | Finite seeds | Oracle | Free |
|---|---|---:|---:|---:|
| Baseline | T32/D32/FFN32/L2/H2 | 5/5 | 20/20 | 20/20 |
| Depth | T8/D16/FFN32/L18/H2 | 5/5 | 13/20 | 13/20 |
| Heads | T8/D128/FFN32/L2/H128 | 5/5 | 20/20 | 20/20 |
| Combined | T16/D32/FFN64/L6/H8 | 5/5 | 20/20 | 20/20 |
| Width | T16/D128/FFN256/L3/H4 | 5/5 | 20/20 | 20/20 |

Across all formal categories, 25/25 training seeds were finite and Oracle/free
generation were each exact for 93/100 cases. The L18 result is a
generation-quality boundary, not a QNN execution or numerical failure.

One-step graph creation, finalization, forward, backward, and Adam succeeded
through L256/H2, L2/H256, T512/D32/FFN64/L3/H4, and
D256/FFN512/L3/H4. Exploration stopped at clear experiment-time saturation;
these points are an evaluated envelope, not a hardware limit.

## First failures

The adjacent formal depth boundary was L18 success followed by L19 failure at
the final evaluation in `layer_018_output`. A deeper diagnostic reproduced a
first nonfinite `dembedding` at L64, seed 1, step 301.

At T16/D256/L3/H4, FFN371 completed seed 1 for 320 steps while FFN372 first
produced nonfinite logits at step 32. FFN371/372 therefore separates the
observed D×FFN trajectory boundary by one FFN element.

For both boundaries the schema, CPU reference, shape/alias/resource audits,
graph create/finalize, and QNN calls succeeded. QNN returned success while an
application-visible tensor became nonfinite, so the result is classified as a
reproducible numerical boundary, not a graph-capacity or hardware-maximum
claim.

## Reproducibility, performance, and thermal state

Three independent L6/H8 runs had identical hashes for all 320 loss values, all
320 accuracy values, final parameters, and step-320 logits:
`BITWISE_REPRODUCIBLE`.

Five `EXCLUSIVE_BENCHMARK` runs of L6/H8 produced these medians and ranges:

| Metric | Median | Range |
|---|---:|---:|
| Initialization | 75.0984 ms | 70.6561–79.0588 ms |
| Graph creation | 83.7153 ms | 76.5259–96.1498 ms |
| Finalize | 2543.56 ms | 2505.49–2653.62 ms |
| Steady training step | 88.9739 ms | 88.3895–92.2583 ms |
| Updates/s | 11.2393 | 10.8391–11.3136 |
| Tokens/s | 179.828 | 173.426–181.017 |
| Generation token latency | 9.29719 ms | 8.75464–9.46667 ms |
| Process peak RSS | 327592 KiB | 325508–328076 KiB |

Correctness and exclusive performance results are separate in the public
bundle. Formal-run battery temperature stayed between 38 and 41 °C and Android
thermal status remained 0. Battery temperature is not reported as CPU
temperature.

## UI and regressions

The L6/H8 320-step formal configuration completed through the real UI. Its
configuration, phase, seed, step, total, and loss were visible; foreground and
background notification updates worked; `ongoing` changed from true to false;
notification tap resumed MainActivity; and the completed notification
auto-cancelled.

The final audited APK passed the existing 19 headless correctness suites plus
the callback-bound suite, nine focused fixed-state/tap/optimizer suites, and
Skel replace-then-reuse recovery. Headless Activity create/resume and focus
takeover counts were zero.

## Public evidence

The allow-list-exported data are in
[qnn-htp-generic-depth-head-2026-07](results/qnn-htp-generic-depth-head-2026-07/README.md).
The bundle contains configuration, resource, per-seed, CPU/HTP comparison,
generation, reproducibility, performance, thermal, failure-boundary, and UI
tables. It excludes raw tensors/logits, checkpoints, device/connection
identifiers, private paths, APKs, QAIRT binaries/headers, Stub/Skel assets, and
device logs.
