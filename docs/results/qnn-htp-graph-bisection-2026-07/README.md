# QNN HTP fixed-state graph-prefix bisection

This directory contains allow-listed aggregates from headless fixed-state
fresh-process experiments:

- `summary.json`: the fixed manifest, primary paired result, classification,
  and remaining unknowns.
- `process-results.csv`: per-process, per-variant hashes, repeat differences,
  poison/nonfinite counts, and QNN return-code evidence.
- `graph-variants.csv`: the exact terminal boundary and removed node group for
  each graph variant.
- `numerical-audit-failures.csv`: a QNN-success process rejected because the
  full graph produced nonfinite APP_READ elements.

The final comparison completed 14 paired fresh processes. Full,
`stop_after_dinput`, and `stop_after_dembedding` varied in 2, 3, and 1
processes, respectively. In both full-positive processes, the later-created
`stop_after_dembedding` graph was stable. All 4,200 completed-pair executions
returned QNN code 0; fixed input/state hashes were unchanged, all bound
APP_READ tensors were audited, APP_READ poison residuals were zero, and
nonfinite counts were zero. One additional full-graph process returned QNN code
0 but was rejected with 584 nonfinite elements. Runtime/context creation order
remains confounded with graph structure.

Generate the aggregates from ignored private headless reports with:

```powershell
.\scripts\export_public_qnn_graph_bisection_results.ps1
```

No raw tensor, device endpoint, private path, APK, raw logcat, or Qualcomm
binary is included.
