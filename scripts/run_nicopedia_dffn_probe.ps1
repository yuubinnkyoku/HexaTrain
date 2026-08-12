# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Minimal, research-only diagnostic probe for a larger Nicopedia D/FFN graph.
#
# The probe requires a native one-update mode.  It must never invoke the
# ordinary long-training suite: that path has a fixed trajectory contract and
# is unsafe at Steps=1.  It records only aggregate diagnostics below build/
# and never retries a failed D32 candidate automatically.
param(
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [switch]$AllowTier3Research,
    [ValidateRange(2, 256)][int]$Dimension = 32,
    [ValidateRange(2, 1024)][int]$FeedForwardDimension = 32,
    [ValidateRange(1, 240)][int]$PollLimit = 60,
    [ValidateRange(1, 10)][int]$PollSeconds = 2,
    [ValidateRange(10, 600)][int]$CheckpointStallSeconds = 120,
    [string]$RunId = ("dffn-probe-" + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$ReportRoot = '',
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
if (-not $ReportRoot) { $ReportRoot = Join-Path $root 'build\reports\nicopedia-dffn-probe' }
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
$ReportRoot = [IO.Path]::GetFullPath($ReportRoot)
if (-not $ReportRoot.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'DFFN_PROBE_REPORT_ROOT_MUST_BE_UNDER_BUILD'
}

function Get-ModelEstimate([int]$D, [int]$F) {
    # Exact flattened parameter count used by the D/FFN search: 2VD +
    # L(4D^2 + 4D + 2D*FFN), with V=256 and L=19.
    $parameters = [int64](2 * 256 * $D + 19 * (4 * $D * $D + 4 * $D + 2 * $D * $F))
    return [ordered]@{
        parameter_count = $parameters
        parameter_bytes_fp32 = 4 * $parameters
        checkpoint_payload_estimate_bytes = 12 * $parameters
        optimizer_working_set_estimate_bytes = 16 * $parameters
        activation_estimate_bytes_per_batch = [int64](4 * 8 * 32 * ($D * 14 + $F * 4 + 256 * 3))
    }
}
function Get-Fields([string]$Path) {
    $map = @{}
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $map }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([^=\r\n]+)=(.*)$') { $map[$Matches[1]] = $Matches[2] }
    }
    return $map
}
function Get-FieldValue([hashtable]$Map, [string]$Name) {
    if ($null -ne $Map -and $Map.ContainsKey($Name)) { return [string]$Map[$Name] }
    return $null
}
function Get-JsonField([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json).$Name } catch { return $null }
}
function Assert-NativeOneUpdateProbeContract([string]$RunnerPath) {
    if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) { throw 'DFFN_PROBE_RUNNER_MISSING' }
    $source = Get-Content -LiteralPath $RunnerPath -Raw
    # Both tokens are mandatory. The runner must expose a distinct host flag
    # and dispatch it to a distinct native/headless suite; matching merely a
    # generic training mode is deliberately insufficient.
    if ($source -notmatch '(?i)\bOneUpdateProbe\b' -or $source -notmatch '(?i)nicopedia-dffn-probe') {
        throw 'DFFN_PROBE_NATIVE_ONE_UPDATE_MODE_UNAVAILABLE'
    }
}
function Classify-Probe([hashtable]$Fields, [string]$RunnerError, [string]$PartialPath) {
    # Status/heartbeat evidence is kept separate from graph and execute
    # evidence. A missing report never becomes a numeric failure.
    $phase = Get-JsonField $PartialPath 'current_phase'
    $heartbeat = Get-JsonField $PartialPath 'last_heartbeat'
    if ($RunnerError -match '(?i)ADB_(TRANSPORT|TIMEOUT|COMMAND)|INSTRUMENTATION_EXIT|DEVICE_UNRESPONSIVE|ADB_NO_ONLINE|ADB_STABLE_IDENTITY') {
        return [ordered]@{ classification = 'ADB_TRANSPORT_OR_DEVICE'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'none'; reason = 'Transport/device state is not a QNN numeric result.' }
    }
    if ($RunnerError -match '(?i)HEADLESS_HEARTBEAT_STALE|HEADLESS_TIMEOUT|CHECKPOINT_PROGRESS_STALLED') {
        return [ordered]@{ classification = 'RUNNER_HEARTBEAT_OR_TIMEOUT'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'none'; reason = 'Heartbeat/timeout needs runner or device diagnosis before another graph run.' }
    }
    if ($RunnerError -match '(?i)ACTIVITY_LAUNCHED|FOCUS_TAKEOVER|HEADLESS_ACTIVITY') {
        return [ordered]@{ classification = 'HEADLESS_ACTIVITY_INVARIANT'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'none'; reason = 'Native evidence is retained, but the headless lifecycle contract was violated.' }
    }
    if ($RunnerError -match '(?i)RUN_ALREADY_ACTIVE|APK_PROVENANCE|RUN_ID_REUSE|PHYSICAL_DEVICE_REQUIRED|THERMAL_STATUS_UNAVAILABLE|BATTERY_|QAIRT_') {
        return [ordered]@{ classification = 'RUNNER_PREFLIGHT_OR_PROVENANCE'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'none'; reason = 'A safety, provenance, or concurrent-run preflight rejected execution before native graph work.' }
    }
    if ($null -eq $Fields -or $Fields.Count -eq 0) {
        return [ordered]@{ classification = 'NO_NATIVE_REPORT'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'none'; reason = 'No terminal native report was retained; do not infer graph or tensor health.' }
    }
    $finalize = if ($Fields.ContainsKey('api_trace_full_step_graph_finalize_result')) { $Fields.api_trace_full_step_graph_finalize_result } elseif ($Fields.ContainsKey('graph_finalize_result')) { $Fields.graph_finalize_result } else { $null }
    $attempts = if ($Fields.ContainsKey('api_trace_graph_execute_attempt_count')) { [int]$Fields.api_trace_graph_execute_attempt_count } else { 0 }
    $qnn = if ($Fields.ContainsKey('api_trace_last_qnn_result')) { $Fields.api_trace_last_qnn_result } elseif ($Fields.ContainsKey('qnn_execute_result')) { $Fields.qnn_execute_result } else { $null }
    if ($null -eq $finalize -or $finalize -ne '0') {
        return [ordered]@{ classification = 'GRAPH_PREPARE_OR_FINALIZE'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'd24-f48'; reason = 'No verified finalized graph; reduce both D and FFN before another device run.' }
    }
    if ($attempts -lt 1 -or $null -eq $qnn -or $qnn -ne '0') {
        return [ordered]@{ classification = 'FIRST_QNN_EXECUTE'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'd24-f48'; reason = 'Graph finalized but the first execute was absent or nonzero.' }
    }
    if ((Get-FieldValue $Fields 'qnn_return_code_success') -ne 'true' -or
        (Get-FieldValue $Fields 'output_tensors_finite') -ne 'true' -or
        (Get-FieldValue $Fields 'cpu_fallback') -ne 'false') {
        return [ordered]@{ classification = 'HTP_HEALTH_REJECTED'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'd24-f48'; reason = 'QNN return, finite tensor, and fallback evidence must all pass independently.' }
    }
    return [ordered]@{ classification = 'ONE_STEP_HEALTHY'; phase = $phase; heartbeat = $heartbeat; retry_d32 = $false; next_candidate = 'd32-f64'; reason = 'D32/F32 completed one verified HTP update; proceed to the joint candidate screening once.' }
}

if ($SelfTest) {
    $anchor = Get-ModelEstimate 16 32
    if ($anchor.parameter_count -ne 48320) { throw 'DFFN_PROBE_SELFTEST_PARAMETER_COUNT' }
    $temp = Join-Path $env:TEMP ('phonelm-dffn-probe-' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($temp) | Out-Null
    try {
        $partial = Join-Path $temp 'partial.json'
        '{"current_phase":"execute","last_heartbeat":123}' | Set-Content -LiteralPath $partial -Encoding utf8
        $healthy = @{ api_trace_full_step_graph_finalize_result='0'; api_trace_graph_execute_attempt_count='1'; api_trace_last_qnn_result='0'; qnn_return_code_success='true'; output_tensors_finite='true'; cpu_fallback='false' }
        if ((Classify-Probe $healthy '' $partial).classification -ne 'ONE_STEP_HEALTHY') { throw 'DFFN_PROBE_SELFTEST_HEALTHY' }
        $prepare = @{ api_trace_full_step_graph_finalize_result='-1' }
        if ((Classify-Probe $prepare '' $partial).classification -ne 'GRAPH_PREPARE_OR_FINALIZE') { throw 'DFFN_PROBE_SELFTEST_PREPARE' }
        if ((Classify-Probe @{} '' $partial).classification -ne 'NO_NATIVE_REPORT') { throw 'DFFN_PROBE_SELFTEST_NO_REPORT' }
        if ((Classify-Probe @{} 'ADB_TRANSPORT_FAILURE' $partial).classification -ne 'ADB_TRANSPORT_OR_DEVICE') { throw 'DFFN_PROBE_SELFTEST_TRANSPORT' }
        if ((Classify-Probe @{} 'RUN_ALREADY_ACTIVE' $partial).classification -ne 'RUNNER_PREFLIGHT_OR_PROVENANCE') { throw 'DFFN_PROBE_SELFTEST_PREFLIGHT' }
        $runner = Join-Path $temp 'runner.ps1'
        'param([switch]$OneUpdateProbe) # nicopedia-dffn-probe' | Set-Content -LiteralPath $runner -Encoding utf8
        Assert-NativeOneUpdateProbeContract $runner
        'param()' | Set-Content -LiteralPath $runner -Encoding utf8
        $rejected = $false
        try { Assert-NativeOneUpdateProbeContract $runner } catch { $rejected = $_.Exception.Message -eq 'DFFN_PROBE_NATIVE_ONE_UPDATE_MODE_UNAVAILABLE' }
        if (-not $rejected) { throw 'DFFN_PROBE_SELFTEST_CONTRACT_FAIL_CLOSED' }
    } finally { Remove-Item -LiteralPath $temp -Recurse -Force }
    Write-Host 'run_nicopedia_dffn_probe_self_test=PASS'
    exit 0
}

if (-not $AllowTier3Research) { throw 'TIER3_RESEARCH_PERMISSION_REQUIRED: use -AllowTier3Research for device probe' }
if (($Dimension % 2) -ne 0) { throw 'DFFN_PROBE_DIMENSION_MUST_BE_EVEN' }
if ($RunId -notmatch '^[A-Za-z0-9._-]{1,64}$') { throw 'DFFN_PROBE_RUN_ID_INVALID' }
[IO.Directory]::CreateDirectory($ReportRoot) | Out-Null

$modelTag = "-t32-d$Dimension-f$FeedForwardDimension"
$trainingRoot = Join-Path $root 'build\reports\nicopedia-htp-training'
$resultPath = Join-Path $trainingRoot "seed1-l19$modelTag-steps1-result.txt"
$partialPath = Join-Path $trainingRoot "seed1-l19$modelTag-steps1-partial-status.json"
$runner = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
Assert-NativeOneUpdateProbeContract $runner
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
& (Join-Path $root 'gradlew.bat') :app:assembleDebug :app:assembleDebugAndroidTest `
    '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" `
    "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
if ($LASTEXITCODE -ne 0) { throw 'DFFN_PROBE_BUILD_FAILED' }
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk `
    -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
    -ReportPath (Join-Path $ReportRoot "$RunId-apk-audit-private.txt")
if ($LASTEXITCODE -ne 0) { throw 'DFFN_PROBE_APK_AUDIT_FAILED' }
$runnerError = ''
$startedUtc = [DateTime]::UtcNow
try {
    & $runner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -OneUpdateProbe -Seed 1 -Layers 19 -Tokens 32 -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -BatchSize 8 -Steps 1 -CheckpointInterval 1 -CheckpointStallSeconds $CheckpointStallSeconds -PollLimit $PollLimit -PollSeconds $PollSeconds -ProgressEverySeconds 10 -RunId $RunId -SkipBuild
    if ($LASTEXITCODE -ne 0) { $runnerError = "RUNNER_EXIT_$LASTEXITCODE" }
} catch { $runnerError = $_.Exception.Message }
$fields = Get-Fields $resultPath
$diagnosis = [ordered]@{
    schema = 'PHONELM_NICOPEDIA_DFFN_PROBE_V1'
    qairt_build_id = $ExpectedBuildId
    dimension = $Dimension; feed_forward_dimension = $FeedForwardDimension
    fixed = [ordered]@{ layers = 19; heads = 2; tokens = 32; vocabulary = 256; batch_size = 8; learning_rate = 0.003; steps = 1; checkpoint_format = 'NPRTCKPTV2' }
    estimate = Get-ModelEstimate $Dimension $FeedForwardDimension
    elapsed_seconds = [Math]::Round(([DateTime]::UtcNow - $startedUtc).TotalSeconds, 3)
    runner_error_class = if ($runnerError) { $runnerError } else { 'none' }
    graph_finalize_result = Get-FieldValue $fields 'api_trace_full_step_graph_finalize_result'
    graph_execute_attempt_count = Get-FieldValue $fields 'api_trace_graph_execute_attempt_count'
    qnn_last_result = Get-FieldValue $fields 'api_trace_last_qnn_result'
    qnn_return_code_success = Get-FieldValue $fields 'qnn_return_code_success'
    output_tensors_finite = Get-FieldValue $fields 'output_tensors_finite'
    cpu_fallback = Get-FieldValue $fields 'cpu_fallback'
    first_execute_us = Get-FieldValue $fields 'first_execute_us'
    training_step_ms = Get-FieldValue $fields 'training_step_ms'
    activity_create_count = Get-FieldValue $fields 'activity_create_count'
    activity_resume_count = Get-FieldValue $fields 'activity_resume_count'
    focus_takeover_count = Get-FieldValue $fields 'focus_takeover_count'
    classification = Classify-Probe $fields $runnerError $partialPath
}
$diagnosis | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $ReportRoot "$RunId-diagnosis.json") -Encoding utf8
if ($diagnosis.classification.classification -ne 'ONE_STEP_HEALTHY') { throw "DFFN_PROBE_FAILED_CLOSED: $($diagnosis.classification.classification)" }
Write-Host "DFFN_PROBE_PASS classification=ONE_STEP_HEALTHY next_candidate=d32-f64"
