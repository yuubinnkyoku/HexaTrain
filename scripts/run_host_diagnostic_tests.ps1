# Diagnostic / research host suite.
# Ownership: diagnostic and probe tooling self-tests, research-only analysis
# helpers, and qnn_first_nonfinite diagnostics contracts.
#
# Production/runtime product contracts live in run_host_contract_tests.ps1.
# scripts/run_host_tests.ps1 remains the compatibility orchestrator and runs
# this suite after the contract suite. Coverage is not reduced.
param(
    # Optional shared session created by run_host_tests.ps1. A matching token is
    # required to join it; empty means this invocation owns a fresh standalone session.
    [string]$ObjectSessionDir = "",
    [string]$ObjectSessionToken = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$OutputDirectory = Join-Path $Root "build\host-tests"
. (Join-Path $PSScriptRoot "host_test_common.ps1")

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$CppInclude = Join-Path $Root "app\src\main\cpp"
$HostInclude = Join-Path $Root "host_tests"

if ($ObjectSessionDir) {
    Initialize-PhoneLmHostObjectSession -SessionDirectory $ObjectSessionDir -ExpectedSessionToken $ObjectSessionToken
} else {
    Initialize-PhoneLmHostObjectSession `
        -SessionDirectory (Join-Path $Root "build\host-test-objects\standalone-diagnostic") `
        -Fresh
}

Write-Host "===== host diagnostic suite ====="

# Dedicated qnn_first_nonfinite diagnostics contract (codec / summaries / replay).
$FirstNonfiniteExecutable = Join-Path $OutputDirectory "qnn_first_nonfinite_diagnostics_test.exe"
Invoke-PhoneLmHostCppCompile -Label "qnn_first_nonfinite diagnostics host test" `
    -Output $FirstNonfiniteExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\qnn_first_nonfinite_diagnostics_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "qnn_first_nonfinite diagnostics host test" `
    -Executable $FirstNonfiniteExecutable
Write-Host "qnn_first_nonfinite_diagnostics_host_test=PASS"

# Research / diagnostic tooling self-tests (tooling correctness, not product ABI).
$ReadoutProbeExecutable = Join-Path $OutputDirectory "readout_probe_test.exe"
Invoke-PhoneLmHostCppCompile -Label "readout probe" `
    -Output $ReadoutProbeExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\readout_probe.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "readout probe self-test" `
    -Executable $ReadoutProbeExecutable -Arguments @("--self-test")

$IntraBlockExecutable = Join-Path $OutputDirectory "intra_block_readability_test.exe"
Invoke-PhoneLmHostCppCompile -Label "intra-block readability" `
    -Output $IntraBlockExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\intra_block_readability.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "intra-block readability self-test" `
    -Executable $IntraBlockExecutable -Arguments @("--self-test")

$AttentionInternalExecutable = Join-Path $OutputDirectory "attention_internal_diagnosis_test.exe"
Invoke-PhoneLmHostCppCompile -Label "attention-internal diagnosis" `
    -Output $AttentionInternalExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\attention_internal_diagnosis.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "attention-internal diagnosis self-test" `
    -Executable $AttentionInternalExecutable -Arguments @("--self-test")

$OutputProjectionExecutable = Join-Path $OutputDirectory "output_projection_audit_test.exe"
Invoke-PhoneLmHostCppCompile -Label "output-projection audit" `
    -Output $OutputProjectionExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\output_projection_audit.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "output-projection audit self-test" `
    -Executable $OutputProjectionExecutable -Arguments @("--self-test")

$ProbeOptimizationExecutable = Join-Path $OutputDirectory "probe_optimization_audit_test.exe"
Invoke-PhoneLmHostCppCompile -Label "probe-optimization audit" `
    -Output $ProbeOptimizationExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\probe_optimization_audit.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "probe-optimization audit self-test" `
    -Executable $ProbeOptimizationExecutable -Arguments @("--self-test")

$SeedInstabilityExecutable = Join-Path $OutputDirectory "seed_instability_diagnostics_test.exe"
Invoke-PhoneLmHostCppCompile -Label "seed-instability diagnostics" `
    -Output $SeedInstabilityExecutable `
    -Std "-std=c++20" `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\seed_instability_diagnostics.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "seed-instability diagnostics self-test" `
    -Executable $SeedInstabilityExecutable -Arguments @("--self-test")

# Muon numeric diagnostic tooling (host-only analysis helper).
$NicopediaMuonNumericDiagnosticExecutable = Join-Path $OutputDirectory "nicopedia_muon_numeric_diagnostic.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia Muon numeric diagnostic" `
    -Output $NicopediaMuonNumericDiagnosticExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_optimizer.cpp"),
        (Join-Path $Root "host_tests\nicopedia_muon_numeric_diagnostic.cpp")) `
    -IncludeDirs @($CppInclude)
$NicopediaMuonNumericDiagnosticDir = Join-Path $Root "build\htp-muon\numeric-diagnostic"
New-Item -ItemType Directory -Force -Path $NicopediaMuonNumericDiagnosticDir | Out-Null
Invoke-PhoneLmHostCppRun -Label "Nicopedia Muon numeric diagnostic" `
    -Executable $NicopediaMuonNumericDiagnosticExecutable `
    -Arguments @($NicopediaMuonNumericDiagnosticDir)
Write-Host "nicopedia_muon_numeric_diagnostic=PASS"

Complete-PhoneLmHostObjectSession
Test-PhoneLmHostRunnerSelfCheck
Write-Host "run_host_diagnostic_tests=PASS"
