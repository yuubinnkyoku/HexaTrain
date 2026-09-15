# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Focused prepared-generation benchmark for inference performance comparison.
# Runs nicopedia-generate-prepared suite and extracts timing/phase metrics.
param(
    [Parameter(Mandatory = $true)][string]$ApkPath,
    [Parameter(Mandatory = $true)][string]$TestApkPath,
    [Parameter(Mandatory = $true)][string]$CheckpointPath,
    [string]$TokenizerPath = '',
    [string]$PromptText = '日本の首都は',
    [int]$Seed = 1,
    [int]$Layers = 19,
    [int]$Heads = 2,
    [int]$Tokens = 32,
    [int]$Dimension = 64,
    [int]$FeedForwardDimension = 64,
    [int]$Vocabulary = 1024,
    [int]$CheckpointStep = 7500,
    [int]$MaxNewBytes = 32,
    [string]$AttentionGate = 'headwise_g1_sigmoid',
    [string]$Mode = 'Greedy',
    [string]$Label = 'bench',
    [string]$QairtSdkRoot = 'C:\Qualcomm\AIStack\QAIRT\2.48.40.260702',
    [string]$ExpectedBuildId = '2.48.40.260702151143',
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = 'com.yuubinnkyoku.phonelm'
$reportDir = Join-Path $root "build\reports\prepared-bench-$Label"
[IO.Directory]::CreateDirectory($reportDir) | Out-Null

$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Write-Host "Device: $($deviceInfo.Model) serial=$($deviceInfo.Serial)"

function Adb([string[]]$Arguments) {
    return (Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments $Arguments).Output
}

if (-not $SkipInstall) {
    Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
    Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
    Write-Host "Installing APKs..."
    Adb @('install', '-r', $ApkPath) | Out-Null
    Adb @('install', '-r', '-t', $TestApkPath) | Out-Null
    Adb @('shell', 'am', 'force-stop', $package) | Out-Null
}

$thermalBefore = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'before'
Write-Host "Thermal before: status=$($thermalBefore.thermal_status) temp=$($thermalBefore.battery_temperature_c)C"

# Stage inputs
$runId = "bench-$Label-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$remoteDir = "files/headless-input/$runId"
Assert-PhoneLmHeadlessInputFresh -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir
Adb @('shell', 'run-as', $package, 'mkdir', '-p', $remoteDir) | Out-Null

$checkpointName = "htp-seed${Seed}-l${Layers}-t${Tokens}-d${Dimension}-f${FeedForwardDimension}-step${CheckpointStep}.ckpt"
$tmpCkpt = "/data/local/tmp/bench-ckpt"
Adb @('push', $CheckpointPath, $tmpCkpt) | Out-Null
Adb @('shell', 'run-as', $package, 'cp', $tmpCkpt, "$remoteDir/$checkpointName") | Out-Null
Adb @('shell', 'rm', '-f', $tmpCkpt) | Out-Null

if ($Vocabulary -eq 1024 -and $TokenizerPath -ne '') {
    $tmpTok = "/data/local/tmp/bench-tok"
    Adb @('push', $TokenizerPath, $tmpTok) | Out-Null
    Adb @('shell', 'run-as', $package, 'cp', $tmpTok, "$remoteDir/byte-bpe-v1024.model") | Out-Null
    Adb @('shell', 'rm', '-f', $tmpTok) | Out-Null
}

$promptBytes = [Text.Encoding]::UTF8.GetBytes($PromptText)
$localPrompt = Join-Path $env:TEMP "bench-prompt-$runId.bin"
[IO.File]::WriteAllBytes($localPrompt, $promptBytes)
$tmpPrompt = "/data/local/tmp/bench-prompt"
Adb @('push', $localPrompt, $tmpPrompt) | Out-Null
Adb @('shell', 'run-as', $package, 'cp', $tmpPrompt, "$remoteDir/prompt.bin") | Out-Null
Adb @('shell', 'rm', '-f', $tmpPrompt) | Out-Null
Remove-Item -LiteralPath $localPrompt -Force -ErrorAction SilentlyContinue

# Run instrumentation
Clear-PhoneLmResultMarker -Adb $adb -Device $device -Package $package
$instrumentDir = Join-Path $reportDir "instrumentation"
[IO.Directory]::CreateDirectory($instrumentDir) | Out-Null

Write-Host "Running nicopedia-generate-prepared suite..."
$instrument = Start-PhoneLmHeadlessInstrumentation -Adb $adb -Device $device -Package $package `
    -Class "$package.HeadlessDeviceTestRunner" -Suite 'nicopedia-generate-prepared' -RunId $runId `
    -Arguments @{
        seed = $Seed; vocabulary = $Vocabulary; layers = $Layers; heads = $Heads
        tokens = $Tokens; dimension = $Dimension; feedForwardDimension = $FeedForwardDimension
        checkpointStep = $CheckpointStep; generateMode = $Mode.ToLowerInvariant()
        maxNewBytes = $MaxNewBytes; temperature = '1.0'; topK = 256; samplingSeed = 0
        attentionGate = $AttentionGate
    } `
    -StdoutPath (Join-Path $instrumentDir 'stdout.txt') -StderrPath (Join-Path $instrumentDir 'stderr.txt')

$waited = Wait-PhoneLmHeadlessStatus -Process $instrument -Adb $adb -Device $device -Package $package `
    -PollLimit 300 -PollSeconds 2 -ProgressEverySeconds 15 -Label "bench-$Label" `
    -ExpectedRunId $runId `
    -ConditionAction {
        param($elapsed)
        $state = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase "bench-$elapsed-sec"
        Write-Host "progress elapsed=$elapsed thermal=$($state.thermal_status) temp=$($state.battery_temperature_c)C"
    }

$thermalAfter = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'after'
Write-Host "Thermal after: status=$($thermalAfter.thermal_status) temp=$($thermalAfter.battery_temperature_c)C"

$result = Get-PhoneLmHeadlessReport -StatusJson $waited.StatusJson -Adb $adb -Device $device -Package $package
$resultPath = Join-Path $reportDir "result.txt"
Set-Content -LiteralPath $resultPath -Value $result -Encoding utf8
Write-Host "Report saved: $resultPath"

# Extract key metrics
$metrics = @{}
foreach ($line in $result -split "`n") {
    if ($line -match '^([a-z_]+)=(.+)$') { $metrics[$Matches[1]] = $Matches[2].Trim() }
}

Write-Host "`n=== Benchmark Results ($Label) ==="
foreach ($key in @(
    'status', 'prepared_graph_reused', 'prepared_engine_run_count',
    'generation_total_seconds', 'generation_ms_per_byte',
    'generation_measured_steps',
    'phase_one_hot_build_us', 'phase_schema_validation_us',
    'phase_app_write_bind_us', 'phase_app_write_snapshot_us',
    'phase_app_read_allocate_us', 'phase_app_read_poison_fill_us',
    'phase_qnn_graph_execute_us', 'phase_immutability_check_us',
    'phase_output_materialize_us', 'phase_poison_scan_us', 'phase_finite_scan_us',
    'phase_last_token_extract_us', 'phase_argmax_or_sampling_us',
    'phase_context_shift_us', 'phase_execute_total_us', 'phase_host_unclassified_us',
    'phase_app_write_bytes_per_token', 'phase_app_read_bytes_per_token',
    'phase_app_write_tensor_count', 'phase_app_read_tensor_count',
    'graph_execute_count', 'parity_us', 'generation_loop_us', 'total_wall_us',
    'generated_token_count', 'generated_byte_count',
    'cpu_fallback', 'qnn_return_code_success', 'output_tensors_finite'
)) {
    if ($metrics.Contains($key)) { Write-Host "  $key=$($metrics[$key])" }
}

$metrics | ConvertTo-Json | Set-Content (Join-Path $reportDir "metrics.json") -Encoding utf8
Write-Host "`nDone. Report: $reportDir"
