# QNN HTP generic Transformer depth/head results

PhoneLM executed the numerical operations of explicit Transformer forward,
backward, and Adam update graphs on QNN HTP. Parameter/state registries, CPU
reference code, shape validation, graph-map generation, diagnostics, and
application-visible resource accounting accept generic positive layer and head
counts where the embedding dimension is divisible by the head count.

The maximum formal combined configuration was B1/T16/V32/D32/FFN64/L6/H8.
Five of five 320-step seeds remained finite and Oracle and free-running
generation were each exact for 20/20 cases. H128 at L2 and D128/FFN256 at
L3/H4 also completed five formal seeds. The published baseline
T32/D32/FFN32/L2/H2 retained its established legacy canonical parameter hash,
representative logits hash, losses, accuracy, and 20/20 generation results.

The first adjacent formal depth failure was L19 after L18 completed five finite
training seeds. At D256/L3/H4, FFN371 completed seed 1 for 320 steps and FFN372
first failed at step 32. These are reproducible numerical trajectory boundaries:
QNN graph execution returned success while an application-visible tensor became
nonfinite. They are not claims of an HTP hardware maximum. One-step graph
creation, finalization, forward, backward, and Adam covered the larger envelope
listed in `manifest.json`; exploration stopped at clear runtime saturation.

`resource-estimates.csv` is conservative application-visible accounting and
must not be interpreted as DSP/runtime-internal memory use. Correctness and
exclusive benchmark measurements are separated. Battery temperature and
Android thermal status are reported without labeling battery temperature as CPU
temperature.

This directory is produced by an allow-list exporter. Raw tensors/logits,
checkpoints, device and connection identifiers, private paths, APKs, QAIRT
libraries/headers, Stub/Skel assets, and device logs are excluded.