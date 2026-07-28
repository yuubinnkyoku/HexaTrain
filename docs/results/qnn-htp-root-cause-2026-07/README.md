# QNN HTP fixed-state root-cause results

This directory is a public, aggregate-only record of the fixed-state investigation.
It contains no tensor payloads, process identifiers, timestamps, device endpoints,
private filesystem paths, APKs, QNN binaries, or logcat captures.

`summary.json` records the fixed input/state hashes, exact two-sided 95% Clopper-Pearson
intervals for process-level variability, and the verified QNN and buffer invariants.

The following statements intentionally have different evidence strength:

- Confirmed: `SOFTMAX_DOT` was declared as `[8,8]`, while its
  `ReduceSum(axis=1, keep_dims=true)` producer requires `[8,1]`.
- Confirmed: pre-fix `STOP_AFTER_DINPUT` varied in 2/24 fresh processes; after
  correcting the declaration it varied in 0/50, while corrected FULL varied in
  0/10. The corrected runs completed 6,000/6,000 executes with no nonfinite
  element, poison residual, APP_WRITE change, Activity launch, or focus takeover.
- Inference: the shape mismatch caused the earlier fresh-instance-associated
  variability and nonfinite outputs.
- Not determined: which HTP lowering, scratch allocation, or internal execution
  mechanism was affected by the invalid declaration.

`experiment-results.csv` has one row per fresh-process graph experiment.  It exposes
only status, structural counts, aggregate execution/buffer checks, canonical-hash
cardinality, canonical hash-ID run-length encoding, and aggregate numerical fields.
It intentionally does not expose hash values. `NOT_REPORTED_LEGACY_REPORT` marks the
older baseline reports, which had hash frequencies but did not contain a run sequence.

`first-change.csv` contains the corresponding tap observations. In the paired
`DPROBABILITIES` plus `DSCORES` taps, DPROBABILITIES is stable while DSCORES varies;
the first-changing interval is:

`DPROBABILITIES → Multiply(tt_smp) → SOFTMAX_PRODUCT → ReduceSum(tt_smd) → SOFTMAX_DOT → Subtract(tt_smc) → SOFTMAX_CENTERED → Multiply(tt_ds) → DSCORES`

The tap is APP_READ promotion only. It preserves the source node set and creation
order, but changes output exposure; equivalence of backend lowering is not asserted.

`node-map.csv`, `tensor-map.csv`, and `graph-map-README.md` are generated independently
from the graph construction map and are not modified by the public result exporter.

Regenerate the three aggregate artifacts from private headless reports with:

```powershell
.\scripts\export_public_qnn_root_cause_results.ps1
```

The exporter accepts only this directory as a public output root, validates the exact
required run allow-list and fixed hashes, validates headless/QNN/poison/APP_WRITE
invariants, requires the exact seven-file public allow-list, scans all seven staged
files for prohibited material, and promotes generated aggregates only after
validation succeeds. Re-running it produces byte-identical artifacts.
