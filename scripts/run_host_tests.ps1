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

$OutputProjectionExecutable = Join-Path $OutputDirectory "output_projection_audit_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\output_projection_audit.cpp") `
    -o $OutputProjectionExecutable
if ($LASTEXITCODE -ne 0) {
    throw "output-projection audit compilation failed"
}

& $OutputProjectionExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "output-projection audit self-test failed"
}

$ProbeOptimizationExecutable = Join-Path $OutputDirectory "probe_optimization_audit_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp") `
    (Join-Path $Root "host_tests\probe_optimization_audit.cpp") `
    -o $ProbeOptimizationExecutable
if ($LASTEXITCODE -ne 0) {
    throw "probe-optimization audit compilation failed"
}

& $ProbeOptimizationExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "probe-optimization audit self-test failed"
}

$SeedInstabilityExecutable = Join-Path $OutputDirectory "seed_instability_diagnostics_test.exe"
& g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "host_tests\seed_instability_diagnostics.cpp") `
    -o $SeedInstabilityExecutable
if ($LASTEXITCODE -ne 0) {
    throw "seed-instability diagnostics compilation failed"
}

& $SeedInstabilityExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "seed-instability diagnostics self-test failed"
}

$AttentionMinimalCauseExecutable = Join-Path $OutputDirectory "attention_minimal_cause_test.exe"
& g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "host_tests\attention_minimal_cause.cpp") `
    -o $AttentionMinimalCauseExecutable
if ($LASTEXITCODE -ne 0) {
    throw "attention-minimal-cause diagnostics compilation failed"
}

& $AttentionMinimalCauseExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "attention-minimal-cause diagnostics self-test failed"
}

$ContextSupervisionExecutable = Join-Path $OutputDirectory "context_supervision_stability_test.exe"
& g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    -I (Join-Path $Root "host_tests") `
    (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp") `
    (Join-Path $Root "host_tests\context_supervision_stability.cpp") `
    -o $ContextSupervisionExecutable
if ($LASTEXITCODE -ne 0) {
    throw "context-supervision stability compilation failed"
}

& $ContextSupervisionExecutable --self-test
if ($LASTEXITCODE -ne 0) {
    throw "context-supervision stability self-test failed"
}

# CI uses a deterministic synthetic Japanese fixture; the licensed corpus and
# all private token/checkpoint artifacts remain outside the repository.
# Resolve pwsh explicitly: $PSHOME may point at Windows PowerShell when this
# script runs under a different shell.
$pwshCandidates = @(
    (Join-Path $PSHOME "pwsh.exe"),
    (Join-Path (Split-Path -Parent $PSHOME) "PowerShell\7\pwsh.exe"),
    (Get-Command pwsh -ErrorAction SilentlyContinue).Source
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $pwshCandidates) { throw "PWSH_NOT_FOUND" }
$pwshExe = $pwshCandidates[0]
& $pwshExe -NoProfile -File (Join-Path $Root "scripts\run_nicopedia_real_text_pilot.ps1") -SelfTest
if ($LASTEXITCODE -ne 0) {
    throw "Nicopedia real-text pipeline self-tests failed"
}

$NicopediaGenerationExecutable = Join-Path $OutputDirectory "nicopedia_htp_generation_test.exe"
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root "app\src\main\cpp") `
    (Join-Path $Root "host_tests\nicopedia_htp_generation_test.cpp") `
    -o $NicopediaGenerationExecutable
if ($LASTEXITCODE -ne 0) {
    throw "Nicopedia HTP generation host test compilation failed"
}

& $NicopediaGenerationExecutable
if ($LASTEXITCODE -ne 0) {
    throw "Nicopedia HTP generation host tests failed"
}
