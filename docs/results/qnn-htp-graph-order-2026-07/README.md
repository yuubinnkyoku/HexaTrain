# QNN HTP fixed-state runtime/context order study

This public aggregate contains 60 allow-listed fresh-process reports: 45 discovery reports and 15 confirmation reports. Each process selected one three-slot plan; every slot created a fresh Runtime/context and executed 100 fixed-state repetitions.

The observational classification is FRESH_RUNTIME_CONTEXT_INSTANCE_ASSOCIATED_EXECUTION_VARIABILITY. The completed data show 21 varying slots of 180 and 12 slots with nonfinite APP_READ elements. All recorded QNN create/finalize/execute results were zero; APP_READ poison residuals and APP_WRITE integrity failures were zero.

Confirmation homogeneous-plan process discordance was 3/15; across all homogeneous plans it was 8/30. Homogeneous process-discordance counts were full 2/10, stop-after-dinput 3/10, and stop-after-dembedding 3/10. Homogeneous-position varying-slot counts were 2/30, 4/30, and 3/30. Exact process-level Clopper-Pearson intervals are recorded in summary.json.

The first external audit tensor was gradient_gamma1 in all 21 varying slots; DINPUT also varied in all 21. The candidate node is UNMAPPED because audit order is a tie-breaking limitation, not a node-level localization result.

Measured result: variation is present across variants and positions, including discordant instances inside homogeneous plans. Inference: this design does not support a graph-variant-specific cause, a fixed required position, or a required preceding graph. Runtime/backend/device/context/graph construction is simultaneous in every slot, so those instance scopes remain confounded.

process-results.csv report_id is an experiment identifier, not an operating-system process identifier. Files contain aggregates only. No raw tensor values, device endpoints, paths, executable artifacts, operating-system process identifiers, timestamps, or raw diagnostics are published. Source milestone: aa94e5a.
