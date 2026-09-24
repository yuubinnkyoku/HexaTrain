# Product / contract host suite.
# Ownership: production/runtime invariants, stable schema/identity contracts,
# optimizer/runtime correctness, metadata contract, formal experiment contract
# self-tests that protect current evidence protocols.
#
# Diagnostic/probe tooling self-tests live in run_host_diagnostic_tests.ps1.
# scripts/run_host_tests.ps1 remains the compatibility orchestrator.
param(
    # Optional shared session created by run_host_tests.ps1. A matching token is
    # required to join it; empty means this invocation owns a fresh standalone session.
    [string]$ObjectSessionDir = "",
    [string]$ObjectSessionToken = "",
    # Fast: metadata staleness + CPU reference only (verify.ps1 -Profile Fast).
    # All: full product/contract suite (default; coverage is not reduced).
    [ValidateSet("All", "Fast")][string]$Suite = "All"
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
        -SessionDirectory (Join-Path $Root "build\host-test-objects\standalone-contract") `
        -Fresh
}

Write-Host "===== host contract suite ====="

# Parameter metadata exporter contract (ABI / generated artifact staleness).
$MetadataExporterExecutable = Join-Path $OutputDirectory "export_transformer_parameter_metadata.exe"
Invoke-PhoneLmHostCppCompile -Label "Parameter metadata exporter" `
    -Output $MetadataExporterExecutable `
    -Sources @(Join-Path $Root "host_tests\export_transformer_parameter_metadata.cpp") `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Parameter metadata exporter self-test" `
    -Executable $MetadataExporterExecutable -Arguments @("--self-test")
Invoke-PhoneLmHostCppRun -Label "Parameter metadata exporter staleness check" `
    -Executable $MetadataExporterExecutable -Arguments @(
        "--check",
        (Join-Path $Root "metadata\transformer_parameter_metadata.json"),
        (Join-Path $Root "app\src\main\java\com\yuubinnkyoku\phonelm\GeneratedTransformerParameterMetadata.kt"))
Write-Host "transformer_parameter_metadata_contract=PASS"

# CPU reference training / checkpoint selection contract.
$CpuExecutable = Join-Path $OutputDirectory "cpu_reference_training_test.exe"
Invoke-PhoneLmHostCppCompile -Label "CPU host test" `
    -Output $CpuExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\cpu_reference_training.cpp"),
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\cpu_reference_training_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "CPU host tests" -Executable $CpuExecutable

if ($Suite -eq "Fast") {
    # Fast profile: generated-artifact staleness + CPU reference only.
    # Remaining product contracts stay in Suite All (Host / Formal / PrGate).
    Complete-PhoneLmHostObjectSession
    Test-PhoneLmHostRunnerSelfCheck
    Write-Host "run_host_contract_tests=PASS (Fast: metadata+cpu-reference)"
    return
}

# Headwise G1 gate capacity / identity contract.
$HeadwiseGateExecutable = Join-Path $OutputDirectory "headwise_g1_gate_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Headwise G1 gate host test" `
    -Output $HeadwiseGateExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_checkpoint.cpp"),
        (Join-Path $Root "host_tests\headwise_g1_gate_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Headwise G1 gate host test" -Executable $HeadwiseGateExecutable

# QNN-independent product contracts: quantization, shape validator,
# disabled-runtime fail-closed, prepared-generation identity.
# qnn_first_nonfinite diagnostics coverage is owned by the diagnostic suite.
$QnnSdkIndependentExecutable = Join-Path $OutputDirectory "qnn_sdk_independent_test.exe"
Invoke-PhoneLmHostCppCompile -Label "QNN SDK-independent host test" `
    -Output $QnnSdkIndependentExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\cpu_reference_training.cpp"),
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_backend_info.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_host_quantization.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_hybrid_training.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_graph_shape_validator.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_runtime_stub.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_checkpoint.cpp"),
        (Join-Path $Root "host_tests\qnn_sdk_independent_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "QNN SDK-independent host tests" -Executable $QnnSdkIndependentExecutable

# Depth/seed/validation contracts (seed selection, AR partitions, stability).
# first_nonfinite codec v2 coverage moved to diagnostic suite dedicated test.
$DepthQualityExecutable = Join-Path $OutputDirectory "depth_quality_test.exe"
Invoke-PhoneLmHostCppCompile -Label "depth quality host test" `
    -Output $DepthQualityExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp"),
        (Join-Path $Root "host_tests\depth_quality_test.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "depth quality host tests" -Executable $DepthQualityExecutable

# Margin analysis host contract (kept: formal evidence tooling + CPU invariants).
$MarginAnalysisExecutable = Join-Path $OutputDirectory "margin_analysis_test.exe"
Invoke-PhoneLmHostCppCompile -Label "margin analysis host test" `
    -Output $MarginAnalysisExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp"),
        (Join-Path $Root "host_tests\margin_analysis_test.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "margin analysis host tests" -Executable $MarginAnalysisExecutable

# Critical-margin objective host contract (kept: current experiment formal gate meaning).
$CriticalMarginObjectiveExecutable = Join-Path $OutputDirectory "critical_margin_objective_test.exe"
Invoke-PhoneLmHostCppCompile -Label "critical margin objective host test" `
    -Output $CriticalMarginObjectiveExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "app\src\main\cpp\validation_checkpoint.cpp"),
        (Join-Path $Root "host_tests\critical_margin_objective_test.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "critical margin objective host tests" -Executable $CriticalMarginObjectiveExecutable

# Formal experiment contract self-tests (not moved by name alone):
# these protect current public/experiment protocol invariants.
$CriticalMarginProbeExecutable = Join-Path $OutputDirectory "critical_margin_objective_probe.exe"
Invoke-PhoneLmHostCppCompile -Label "critical margin objective probe" `
    -Output $CriticalMarginProbeExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp"),
        (Join-Path $Root "host_tests\critical_margin_objective_probe.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "critical margin objective probe self-test" `
    -Executable $CriticalMarginProbeExecutable -Arguments @("--self-test")

$AttentionMinimalCauseExecutable = Join-Path $OutputDirectory "attention_minimal_cause_test.exe"
Invoke-PhoneLmHostCppCompile -Label "attention-minimal-cause diagnostics" `
    -Output $AttentionMinimalCauseExecutable `
    -Std "-std=c++20" `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\attention_minimal_cause.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "attention-minimal-cause diagnostics self-test" `
    -Executable $AttentionMinimalCauseExecutable -Arguments @("--self-test")

$ContextSupervisionExecutable = Join-Path $OutputDirectory "context_supervision_stability_test.exe"
Invoke-PhoneLmHostCppCompile -Label "context-supervision stability" `
    -Output $ContextSupervisionExecutable `
    -Std "-std=c++20" `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\context_supervision_stability.cpp")) `
    -IncludeDirs @($CppInclude, $HostInclude)
Invoke-PhoneLmHostCppRun -Label "context-supervision stability self-test" `
    -Executable $ContextSupervisionExecutable -Arguments @("--self-test")

# Nicopedia pipeline / tokenizer / generation / parity / resume / optimizer contracts.
Invoke-PhoneLmHostPwshScript -Label "Nicopedia real-text pipeline self-tests" `
    -ScriptPath (Join-Path $Root "scripts\run_nicopedia_real_text_pilot.ps1") `
    -Arguments @("-SelfTest")

$NicopediaByteBpeExecutable = Join-Path $OutputDirectory "nicopedia_byte_bpe_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia byte-BPE host test" `
    -Output $NicopediaByteBpeExecutable `
    -Sources @(Join-Path $Root "host_tests\nicopedia_byte_bpe_test.cpp") `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia byte-BPE host tests" -Executable $NicopediaByteBpeExecutable

$NicopediaGenerationExecutable = Join-Path $OutputDirectory "nicopedia_htp_generation_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia HTP generation host test" `
    -Output $NicopediaGenerationExecutable `
    -Sources @(Join-Path $Root "host_tests\nicopedia_htp_generation_test.cpp") `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia HTP generation host tests" -Executable $NicopediaGenerationExecutable

$ParityPolicyExecutable = Join-Path $OutputDirectory "nicopedia_parity_policy_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia parity policy host test" `
    -Output $ParityPolicyExecutable `
    -Sources @(Join-Path $Root "host_tests\nicopedia_parity_policy_test.cpp") `
    -IncludeDirs @($CppInclude)
$parityReportDir = Join-Path $Root "build\reports\nicopedia-parity-policy"
[IO.Directory]::CreateDirectory($parityReportDir) | Out-Null
Invoke-PhoneLmHostCppRun -Label "Nicopedia parity policy host tests" `
    -Executable $ParityPolicyExecutable `
    -Arguments @(Join-Path $parityReportDir 'synthetic-fault-results.csv')

$NicopediaResumeExecutable = Join-Path $OutputDirectory "nicopedia_resume_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia resume host test" `
    -Output $NicopediaResumeExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\nicopedia_resume_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia resume host test" `
    -Executable $NicopediaResumeExecutable -WorkingDirectory $OutputDirectory
Write-Host "nicopedia_resume_host_test=PASS"

$NicopediaScheduleExecutable = Join-Path $OutputDirectory "nicopedia_learning_rate_schedule_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia learning-rate schedule host test" `
    -Output $NicopediaScheduleExecutable `
    -Sources @(Join-Path $Root "host_tests\nicopedia_learning_rate_schedule_test.cpp") `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia learning-rate schedule host test" -Executable $NicopediaScheduleExecutable
Write-Host "nicopedia_learning_rate_schedule_host_test=PASS"

$NicopediaMuonExecutable = Join-Path $OutputDirectory "nicopedia_muon_optimizer_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia Muon optimizer host test" `
    -Output $NicopediaMuonExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_optimizer.cpp"),
        (Join-Path $Root "host_tests\nicopedia_muon_optimizer_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia Muon optimizer host test" -Executable $NicopediaMuonExecutable
Write-Host "nicopedia_muon_optimizer_host_test=PASS"

$NicopediaHtpMuonPackExecutable = Join-Path $OutputDirectory "nicopedia_htp_muon_pack_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia HTP Muon pack host test" `
    -Output $NicopediaHtpMuonPackExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_htp_muon.cpp"),
        (Join-Path $Root "host_tests\nicopedia_htp_muon_pack_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia HTP Muon pack host test" -Executable $NicopediaHtpMuonPackExecutable
Write-Host "nicopedia_htp_muon_pack_host_test=PASS"

# Muon optimizer correctness retained in contract suite (not diagnostic by name).
$NicopediaMuonNsStageExecutable = Join-Path $OutputDirectory "nicopedia_muon_ns_stage_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia Muon NS stage host test" `
    -Output $NicopediaMuonNsStageExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_optimizer.cpp"),
        (Join-Path $Root "host_tests\nicopedia_muon_ns_stage_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia Muon NS stage host test" -Executable $NicopediaMuonNsStageExecutable
Write-Host "nicopedia_muon_ns_stage_host_test=PASS"

$NicopediaMuonCheckpointExecutable = Join-Path $OutputDirectory "nicopedia_muon_checkpoint_test.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia Muon checkpoint host test" `
    -Output $NicopediaMuonCheckpointExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_optimizer.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_checkpoint.cpp"),
        (Join-Path $Root "host_tests\nicopedia_muon_checkpoint_test.cpp")) `
    -IncludeDirs @($CppInclude)
Invoke-PhoneLmHostCppRun -Label "Nicopedia Muon checkpoint host test" -Executable $NicopediaMuonCheckpointExecutable
Write-Host "nicopedia_muon_checkpoint_host_test=PASS"

# Compile-check remains in contract coverage (historical compare tool buildability).
$NicopediaMuonCheckpointCompareExecutable = Join-Path $OutputDirectory "nicopedia_muon_checkpoint_compare.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia Muon checkpoint compare" `
    -Output $NicopediaMuonCheckpointCompareExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_optimizer.cpp"),
        (Join-Path $Root "app\src\main\cpp\nicopedia_muon_checkpoint.cpp"),
        (Join-Path $Root "host_tests\nicopedia_muon_checkpoint_compare.cpp")) `
    -IncludeDirs @($CppInclude)
Write-Host "nicopedia_muon_checkpoint_compare_build=PASS"

# CPU generation structural/checkpoint contracts.
$NicopediaCpuGenerateExecutable = Join-Path $OutputDirectory "nicopedia_cpu_generate.exe"
Invoke-PhoneLmHostCppCompile -Label "Nicopedia CPU generation" `
    -Output $NicopediaCpuGenerateExecutable `
    -Sources @(
        (Join-Path $Root "app\src\main\cpp\tiny_language_model_cpu.cpp"),
        (Join-Path $Root "host_tests\nicopedia_cpu_generate.cpp")) `
    -IncludeDirs @($CppInclude)
$traceSelfTest = & $NicopediaCpuGenerateExecutable --trace-self-test
if ($LASTEXITCODE -ne 0) { throw "Nicopedia CPU trace self-test failed" }
$checkpointV2SelfTest = & $NicopediaCpuGenerateExecutable --checkpoint-v2-self-test
if ($LASTEXITCODE -ne 0) { throw "Nicopedia CPU generation NPRTCKPTV2 self-test failed" }
$selfTestCkpt = Join-Path $Root "build\reports\nicopedia-htp-training\htp-seed1-l19-step320.ckpt"
if (Test-Path -LiteralPath $selfTestCkpt) {
    $selfTestGreedy = & $NicopediaCpuGenerateExecutable $selfTestCkpt "e4babae5b7a5e79fa5e883bde381a8e381af" "greedy" "64"
    if ($LASTEXITCODE -ne 0) { throw "Nicopedia CPU generation greedy self-test failed" }
    $selfTestSample = & $NicopediaCpuGenerateExecutable $selfTestCkpt "e3838be382b3e3838be382b3e381a8e381af" "sample" "128" "0.6" "16" "42"
    if ($LASTEXITCODE -ne 0) { throw "Nicopedia CPU generation sample self-test failed" }
    $mapG = @{}
    $selfTestGreedy | Where-Object { $_ -match '^([A-Za-z0-9_]+)=(.*)$' } | ForEach-Object { $mapG[$Matches[1]] = $Matches[2] }
    $mapS = @{}
    $selfTestSample | Where-Object { $_ -match '^([A-Za-z0-9_]+)=(.*)$' } | ForEach-Object { $mapS[$Matches[1]] = $Matches[2] }
    if ($mapG['status'] -ne 'SUCCESS' -or $mapG['generate_mode'] -ne 'greedy' -or $mapG['generated_byte_count'] -ne '64') {
        throw "Nicopedia CPU generation greedy self-test output mismatch"
    }
    if ($mapS['status'] -ne 'SUCCESS' -or $mapS['generate_mode'] -ne 'sample' -or $mapS['generated_byte_count'] -ne '128') {
        throw "Nicopedia CPU generation sample self-test output mismatch"
    }
    Write-Host "nicopedia_cpu_generate_self_test=PASS"
} else {
    Write-Host "nicopedia_cpu_generate_self_test=SKIP (checkpoint not present)"
}

Complete-PhoneLmHostObjectSession
Test-PhoneLmHostRunnerSelfCheck
Write-Host "run_host_contract_tests=PASS"
