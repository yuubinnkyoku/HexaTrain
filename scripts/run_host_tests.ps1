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
