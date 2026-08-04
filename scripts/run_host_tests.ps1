$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$OutputDirectory = Join-Path $Root "build\host-tests"
$CpuExecutable = Join-Path $OutputDirectory "cpu_reference_training_test.exe"
$QnnSdkIndependentExecutable = Join-Path $OutputDirectory "qnn_sdk_independent_test.exe"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    (Join-Path $Root "app\src\main\cpp\cpu_reference_training.cpp") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "host_tests\cpu_reference_training_test.cpp") `
    -o $CpuExecutable
if ($LASTEXITCODE -ne 0) {
    throw "CPU host test compilation failed"
}

& $CpuExecutable
if ($LASTEXITCODE -ne 0) {
    throw "CPU host tests failed"
}

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    (Join-Path $Root "app\src\main\cpp\cpu_reference_training.cpp") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_backend_info.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_host_quantization.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_hybrid_training.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_graph_shape_validator.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_runtime_stub.cpp") `
    (Join-Path $Root "host_tests\qnn_sdk_independent_test.cpp") `
    -o $QnnSdkIndependentExecutable
if ($LASTEXITCODE -ne 0) {
    throw "QNN SDK-independent host test compilation failed"
}

& $QnnSdkIndependentExecutable
if ($LASTEXITCODE -ne 0) {
    throw "QNN SDK-independent host tests failed"
}

$DepthQualityExecutable = Join-Path $OutputDirectory "depth_quality_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp") `
    (Join-Path $Root "host_tests\depth_quality_test.cpp") `
    -o $DepthQualityExecutable
if ($LASTEXITCODE -ne 0) {
    throw "depth quality host test compilation failed"
}

& $DepthQualityExecutable
if ($LASTEXITCODE -ne 0) {
    throw "depth quality host tests failed"
}

$MarginAnalysisExecutable = Join-Path $OutputDirectory "margin_analysis_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp") `
    (Join-Path $Root "host_tests\margin_analysis_test.cpp") `
    -o $MarginAnalysisExecutable
if ($LASTEXITCODE -ne 0) {
    throw "margin analysis host test compilation failed"
}

& $MarginAnalysisExecutable
if ($LASTEXITCODE -ne 0) {
    throw "margin analysis host tests failed"
}

$CriticalMarginObjectiveExecutable = Join-Path $OutputDirectory "critical_margin_objective_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp") `
    (Join-Path $Root "host_tests\critical_margin_objective_test.cpp") `
    -o $CriticalMarginObjectiveExecutable
if ($LASTEXITCODE -ne 0) {
    throw "critical margin objective host test compilation failed"
}

& $CriticalMarginObjectiveExecutable
if ($LASTEXITCODE -ne 0) {
    throw "critical margin objective host tests failed"
}

$CriticalMarginProbeExecutable = Join-Path $OutputDirectory "critical_margin_objective_probe.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\critical_margin_objective_probe.cpp") `
    -o $CriticalMarginProbeExecutable
if ($LASTEXITCODE -ne 0) {
    throw "critical margin objective probe compilation failed"
}

& $CriticalMarginProbeExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "critical margin objective probe self-test failed"
}

$ReadoutProbeExecutable = Join-Path $OutputDirectory "readout_probe_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\readout_probe.cpp") `
    -o $ReadoutProbeExecutable
if ($LASTEXITCODE -ne 0) {
    throw "readout probe compilation failed"
}

& $ReadoutProbeExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "readout probe self-test failed"
}

$IntraBlockExecutable = Join-Path $OutputDirectory "intra_block_readability_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\intra_block_readability.cpp") `
    -o $IntraBlockExecutable
if ($LASTEXITCODE -ne 0) {
    throw "intra-block readability compilation failed"
}

& $IntraBlockExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "intra-block readability self-test failed"
}

$AttentionInternalExecutable = Join-Path $OutputDirectory "attention_internal_diagnosis_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\attention_internal_diagnosis.cpp") `
    -o $AttentionInternalExecutable
if ($LASTEXITCODE -ne 0) {
    throw "attention-internal diagnosis compilation failed"
}

& $AttentionInternalExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "attention-internal diagnosis self-test failed"
}
