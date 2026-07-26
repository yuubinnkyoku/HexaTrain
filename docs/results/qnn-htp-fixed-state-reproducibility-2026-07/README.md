# QNN HTP fixed-state reproducibility results

This directory contains allow-listed aggregate output from five independent
headless fixed-state processes and five independent phase01 processes:

- `summary.json`: study configuration, first varying tensor/node/scope, and
  publication assertions.
- `scope-matrix.csv`: E/D/L standalone scope counts and aggregate outcomes.
- `full-graph-tensors.csv`: per-checkpoint exposed-tensor repeat and
  fresh-process aggregates.
- `phase01-repeats.csv`: completed-step counts for each process and seed.

Generate the files from ignored private device reports with:

```powershell
.\scripts\export_public_qnn_reproducibility_results.ps1
```

The exporter accepts only this destination and emits only aggregate scalar
values. It rejects absolute paths, device identifiers, raw callbacks/logcat,
and binary artifact names. Raw checkpoints, weights, optimizer states, device
reports, and Qualcomm artifacts are not included.
