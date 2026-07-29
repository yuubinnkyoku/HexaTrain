# QNN HTP Tiny LM staged scaling results

Baseline HEAD: 3ca6b08e6d62f49369b54cc4e197970089489f52

Implementation HEAD: 10fccac1335765ad04efe703d9543b2104fe1ce7

Seed 2-5 reproducibility: BITWISE_REPRODUCIBLE. Together with the previously
confirmed seed 1 result, all five seeds are bitwise reproducible.

Maximum stable configuration:
tokens=32,dimension=32,layers=1,heads=1.

Result classification:
END_TO_END_TRAINING_AND_GENERATION_REPRODUCIBLE_SCALED_SINGLE_LAYER.

Stop reason: LAYERS_2_UNSUPPORTED. The layers=2 stage failed explicit
configuration validation because the current parameter and graph schema
represents one Transformer layer. heads=2 was not run.

The maximum stable configuration completed HTP training-step numerical
execution with 5/5 finite seeds, loss reduction for 5/5 seeds, Oracle
20/20, Free-running 20/20,
zero nonzero QNN execute returns, no unexpected APP_WRITE changes, zero poison
residual, and a passing teacher-forcing exclusion assertion.

The same-prefix CPU/HTP maximum absolute logits difference was
0.04186820984.

Across three correctness-preserving benchmark runs, the run-level median
initialization was 92.093 ms, graph creation
was 146.134 ms, finalize was
221.989 ms, steady training step was
19.8302 ms,
50.4281 updates/s,
1613.7 tokens/s, generation token latency
was 1.42745 ms, and process
peak RSS was 229460 KiB.

The reproducibility, full-graph isolation, and focused backward-tap regressions
passed. Reproduce the device runs with scripts/run_qnn_headless_tests.ps1,
the explicit QAIRT 2.48.40 SDK root and expected build ID, the stage suites in
stages.csv, BACKGROUND_CORRECTNESS for correctness, and
EXCLUSIVE_BENCHMARK for the three performance repetitions. Run
scripts/verify_local.ps1 and its -WithQairt form before publication.

Raw checkpoints, tensor dumps, device identifiers, local paths, distributed
SDK files, APKs, and private device reports are excluded.

The baseline report did not expose APP_WRITE mutation and poison-residual
counters per seed. Those two fields are therefore marked as not reported in
seed-reproducibility.csv rather than inferred. The separate reproducibility
regression passed, and the scaled formal evaluation directly reported
APP_WRITE hashes unchanged and zero APP_READ poison residual.