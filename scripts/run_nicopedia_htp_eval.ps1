# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# HTP-native held-out evaluation runner for the Nicopedia L19 model.
#
# Pushes the private validation/development caches and a private checkpoint
# (NPRTCKPTV1 or NPRTCKPTV2) into the app files directory, drives
# QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA_EVAL (teacher-forced forward runs on
# the HTP graph), and pulls the aggregate NLL/perplexity/top-1/top-5/metrics
# report.  The host CPU evaluator runs on the same checkpoint/caches for the
# comparison column.
#
# This is the HTP-native model-health metric: QNN failures or non-finite
# tensors make the device report FAILED (EVALUATION_NONFINITE).  There is no
# parity gate and no CPU fallback.
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [int]$Seed = 1,
  [int]$Layers = 19,
  [int]$Heads = 2,
  [int]$Tokens = 32,  # context window length (8..256; 32 = legacy T32 behavior)
  [int]$Dimension = 32,
  [int]$FeedForwardDimension = 32,
  [int]$CheckpointStep = 1000,
  [int]$ValidationChunks = 8192,
  [int]$DevelopmentChunks = 16384,
  [string]$CheckpointPath = "",
  [string]$CacheRoot = "",
  [int]$PollLimit = 7200,
  [int]$PollSeconds = 2,
  [int]$ProgressEverySeconds = 30,
  [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff'),
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

if ($SelfTest) {
  if ($PollLimit -lt 1 -or $PollSeconds -lt 1 -or $ProgressEverySeconds -lt 1) { throw 'SELFTEST_POLL_CONFIGURATION' }
  if ($Tokens -ne 32) { throw "SELFTEST_TOKENS_DEFAULT: expected=32 actual=$Tokens" }
  if ($Dimension -ne 16 -or $FeedForwardDimension -ne 32) { throw 'SELFTEST_MODEL_DIMENSIONS_DEFAULT' }
  $sample = "status=SUCCESS`ncheckpoint_step=1000`n"
  if (-not ($sample -match '(?m)^status=SUCCESS\s*$')) { throw 'SELFTEST_TERMINAL_STATUS' }
  Write-Host 'run_nicopedia_htp_eval_self_test=PASS'
  exit 0
}
if ($Tokens -lt 8 -or $Tokens -gt 256) { throw 'NICOPEDIA_TOKENS_INVALID: Tokens must be in 8..256' }
if ($Dimension -lt 2 -or $Dimension -gt 256 -or ($Dimension % 2) -ne 0) { throw 'NICOPEDIA_DIMENSION_INVALID: Dimension must be even and in 2..256' }
if ($FeedForwardDimension -lt 2 -or $FeedForwardDimension -gt 1024) { throw 'NICOPEDIA_FFN_INVALID: FeedForwardDimension must be in 2..1024' }
if ($RunId -notmatch '^[A-Za-z0-9._-]{1,64}$') { throw 'RUN_ID_INVALID' }
$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = 'com.yuubinnkyoku.phonelm'
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$testApk = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
$reportRoot = Join-Path $root "build\reports\nicopedia-htp-eval"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

$modelTag = if ($Tokens -eq 32 -and $Dimension -eq 32 -and $FeedForwardDimension -eq 32) { '' } else { "-t$Tokens-d$Dimension-f$FeedForwardDimension" }
$evalDataRoot = if ($Tokens -eq 32) { 'build\private-data\nicopedia-real-text' } else { 'build\private-data\nicopedia-real-text-t64' }
if (-not $CacheRoot) { $CacheRoot = Join-Path $root (Join-Path $evalDataRoot 'caches') }
if (-not $CheckpointPath) {
  $checkpointName = Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $CheckpointStep
  $CheckpointPath = Join-Path $root (Join-Path 'build\reports\nicopedia-htp-training' $checkpointName)
}
if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) { throw "CHECKPOINT_MISSING: $CheckpointPath" }
$checkpointHeader = Get-PhoneLmCheckpointHeaders -Path $CheckpointPath
if ($checkpointHeader.Magic -ne 'NPRTCKPTV2' -or $checkpointHeader.Seed -ne $Seed -or
    $checkpointHeader.Layers -ne $Layers -or $checkpointHeader.Heads -ne $Heads -or
    $checkpointHeader.Vocabulary -ne 256 -or $checkpointHeader.Tokens -ne $Tokens -or
    $checkpointHeader.Dimension -ne $Dimension -or
    $checkpointHeader.FeedForward -ne $FeedForwardDimension -or
    $checkpointHeader.Step -ne $CheckpointStep) {
  throw 'CHECKPOINT_IDENTITY_MISMATCH'
}
$validationCache = Join-Path $CacheRoot 'validation.bin'
$developmentCache = Join-Path $CacheRoot 'development.bin'
if (-not (Test-Path -LiteralPath $validationCache -PathType Leaf)) { throw "VALIDATION_CACHE_MISSING: $validationCache" }
if (-not (Test-Path -LiteralPath $developmentCache -PathType Leaf)) { throw "DEVELOPMENT_CACHE_MISSING: $developmentCache" }
$checkpointResolved = [IO.Path]::GetFullPath($CheckpointPath)
$cacheResolved = [IO.Path]::GetFullPath($CacheRoot)
$allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $checkpointResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CheckpointPath must resolve below the repository build directory"
}
if (-not $cacheResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CacheRoot must resolve below the repository build directory"
}

if (-not $SkipBuild) {
  & (Join-Path $root 'gradlew.bat') :app:assembleDebug :app:assembleDebugAndroidTest '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
  if ($LASTEXITCODE -ne 0) { throw 'APK build failed' }
}
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
$stateBefore = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'before'
$serial = $deviceInfo.Serial
$model = $deviceInfo.Model
$soc = $deviceInfo.Soc

function Adb([string[]]$Arguments) {
  return (Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments $Arguments).Output
}

if (-not $SkipInstall) {
  if (-not (Test-Path -LiteralPath $apk -PathType Leaf) -or -not (Test-Path -LiteralPath $testApk -PathType Leaf)) { throw 'APK_OR_TEST_APK_MISSING' }
  Adb @('install', '-r', $apk) | Out-Null
  Adb @('install', '-r', '-t', $testApk) | Out-Null
  Adb @('shell', 'am', 'force-stop', $package) | Out-Null
  Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
}
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package $package -LocalApk $apk
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package "$package.test" -LocalApk $testApk
# Stage the checkpoint + held-out caches under the run-scoped headless input
# directory. HeadlessDeviceTestRunner owns the native lifecycle.
$remoteDir = "files/headless-input/$RunId"
Assert-PhoneLmHeadlessInputFresh -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir
Adb @('shell', 'run-as', $package, 'mkdir', '-p', $remoteDir) | Out-Null
function PushPrivate([string]$LocalPath, [string]$RemoteName) {
  $tmp = "/data/local/tmp/$RemoteName"
  Adb @('push', $LocalPath, $tmp) | Out-Null
  Adb @('shell', 'run-as', $package, 'cp', $tmp, "$remoteDir/$RemoteName") | Out-Null
  Adb @('shell', 'rm', '-f', $tmp) | Out-Null
}
PushPrivate $CheckpointPath (Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $CheckpointStep)
PushPrivate $validationCache 'validation.bin'
PushPrivate $developmentCache 'development.bin'

Clear-PhoneLmResultMarker -Adb $adb -Device $device -Package $package
$instrumentDir = Join-Path $reportRoot "instrumentation-$RunId"
if (Test-Path -LiteralPath $instrumentDir) { throw 'RUN_ID_REUSE: host instrumentation directory exists' }
[IO.Directory]::CreateDirectory($instrumentDir) | Out-Null
$instrument = $null
try {
  $instrument = Start-PhoneLmHeadlessInstrumentation -Adb $adb -Device $device -Package $package `
  -Class "$package.HeadlessDeviceTestRunner" -Suite 'nicopedia-eval' -RunId $RunId `
  -Arguments @{ seed = $Seed; layers = $Layers; heads = $Heads; tokens = $Tokens; dimension = $Dimension; feedForwardDimension = $FeedForwardDimension; checkpointStep = $CheckpointStep; validationChunks = $ValidationChunks; developmentChunks = $DevelopmentChunks } `
  -StdoutPath (Join-Path $instrumentDir 'stdout.txt') -StderrPath (Join-Path $instrumentDir 'stderr.txt')
$waited = Wait-PhoneLmHeadlessStatus -Process $instrument -Adb $adb -Device $device -Package $package `
  -PollLimit $PollLimit -PollSeconds $PollSeconds -ProgressEverySeconds $ProgressEverySeconds -Label "eval-step-$CheckpointStep" `
  -ExpectedRunId $RunId `
  -PartialPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-step$CheckpointStep-partial-status.json") `
  -StatusProgressAction {
    param($elapsed, $status)
    $phase = [regex]::Match($status, '"current_phase"\s*:\s*"([^"]*)"').Groups[1].Value
    $done = [regex]::Match($status, '"completed_tests"\s*:\s*(\d+)').Groups[1].Value
    $total = [regex]::Match($status, '"total_tests"\s*:\s*(\d+)').Groups[1].Value
    Write-Host "progress phase=eval elapsed_seconds=$elapsed status_phase=$phase completed=$done/$total checkpoint_step=$CheckpointStep"
  } `
  -ConditionAction {
    param($elapsed)
    $state = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase "eval-$elapsed-sec"
    Write-Host "health elapsed_seconds=$elapsed thermal=$($state.thermal_status) battery_temp_c=$($state.battery_temperature_c) battery_voltage_mv=$($state.battery_voltage_mv)"
  } `
  -FocusAction {
    param($elapsed)
    if ((Get-PhoneLmTopPackage -Adb $adb -Device $device) -eq $package) { throw 'FOCUS_TAKEOVER_DETECTED' }
  }
$result = Get-PhoneLmHeadlessReport -StatusJson $waited.StatusJson -Adb $adb -Device $device -Package $package
if ($waited.ProcessExitCode -ne 0) { throw "INSTRUMENTATION_EXIT_FAILURE: code=$($waited.ProcessExitCode)" }
Assert-PhoneLmHeadlessNoActivity -Text $result
if ($waited.StatusJson -notmatch '"status"\s*:\s*"PASSED"' -or $result -notmatch '(?m)^status=SUCCESS\s*$') {
  $result | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-step$CheckpointStep-result.txt") -Encoding utf8
  throw "NICOPEDIA_EVAL_FAILED: seed=$Seed layers=$Layers step=$CheckpointStep"
}
$reportMap = Assert-PhoneLmHealthReport -Text $result -ExpectedBuildId $ExpectedBuildId -ExpectedStep $CheckpointStep -Kind eval
if (-not $reportMap.Contains('model_dimension') -or -not $reportMap.Contains('feed_forward_dimension') -or
    [int]$reportMap.model_dimension -ne $Dimension -or [int]$reportMap.feed_forward_dimension -ne $FeedForwardDimension) {
  throw 'EVAL_REPORT_MODEL_IDENTITY_MISMATCH'
}
foreach ($field in @('validation_chunks', 'development_chunks', 'validation_tokens', 'development_tokens')) {
  if (-not $reportMap.Contains($field)) { throw "HTP_EVAL_CAP_FIELD_MISSING: $field" }
}
if ([int64]$reportMap.validation_chunks -ne $ValidationChunks -or
    [int64]$reportMap.development_chunks -ne $DevelopmentChunks -or
    [int64]$reportMap.validation_tokens -ne ([int64]$ValidationChunks * $Tokens) -or
    [int64]$reportMap.development_tokens -ne ([int64]$DevelopmentChunks * $Tokens)) {
  throw 'HTP_EVAL_CAPACITY_MISMATCH'
}
$stateAfter = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'after'
$annotated = $result.TrimEnd() + "`n" +
  "device_model=$model`n" +
  "device_soc=$soc`n" +
  "android_thermal_status_before=$($stateBefore.thermal_status)`n" +
  "android_thermal_status_after=$($stateAfter.thermal_status)`n" +
  "battery_health_before=$($stateBefore.battery_health)`n" +
  "battery_health_after=$($stateAfter.battery_health)`n" +
  "battery_present_before=$($stateBefore.battery_present.ToString().ToLowerInvariant())`n" +
  "battery_present_after=$($stateAfter.battery_present.ToString().ToLowerInvariant())`n" +
  "battery_level_before=$($stateBefore.battery_level)`n" +
  "battery_level_after=$($stateAfter.battery_level)`n" +
  "battery_voltage_mv_before=$($stateBefore.battery_voltage_mv)`n" +
  "battery_voltage_mv_after=$($stateAfter.battery_voltage_mv)`n" +
  "battery_temperature_c_before=$($stateBefore.battery_temperature_c)`n" +
  "battery_temperature_c_after=$($stateAfter.battery_temperature_c)`n" +
  "compile_time_qairt_build_id=$ExpectedBuildId`n" +
  "private_serial_recorded_for_identity_only=true`n"
$annotated | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-step$CheckpointStep-htp.txt") -Encoding utf8
# Host-side CPU evaluation of the same checkpoint + caches for comparison.
$hostEvalExe = Join-Path $root 'build\host-tests\htp_checkpoint_eval.exe'
if (-not (Test-Path -LiteralPath $hostEvalExe -PathType Leaf)) {
  & g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $root 'app\src\main\cpp') `
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
    (Join-Path $root 'host_tests\htp_checkpoint_eval.cpp') `
    -o $hostEvalExe
  if ($LASTEXITCODE -ne 0) { throw "htp_checkpoint_eval build failed" }
}
$hostEval = & $hostEvalExe $CheckpointPath $validationCache $developmentCache $ValidationChunks $DevelopmentChunks
if ($LASTEXITCODE -ne 0) { throw "htp_checkpoint_eval failed: $hostEval" }
$hostEval | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-step$CheckpointStep-cpu.txt") -Encoding utf8
$hostMap = Get-PhoneLmKeyValueMap -Text ($hostEval -join "`n")
foreach ($key in @('seed', 'layers', 'dimension', 'feed_forward_dimension', 'step', 'parameter_hash', 'finite', 'validation_nll', 'development_nll', 'validation_chunks', 'development_chunks', 'validation_tokens', 'development_tokens')) {
  if (-not $hostMap.Contains($key)) { throw "HOST_EVALUATOR_FIELD_MISSING: $key" }
}
if ([int]$hostMap.seed -ne $Seed -or [int]$hostMap.layers -ne $Layers -or [int]$hostMap.dimension -ne $Dimension -or [int]$hostMap.feed_forward_dimension -ne $FeedForwardDimension -or [int]$hostMap.step -ne $CheckpointStep -or $hostMap.finite -ne 'true') {
  throw 'HOST_EVALUATOR_CHECKPOINT_IDENTITY_MISMATCH'
}
if ([int64]$hostMap.validation_chunks -ne $ValidationChunks -or
    [int64]$hostMap.development_chunks -ne $DevelopmentChunks -or
    [int64]$hostMap.validation_tokens -ne ([int64]$ValidationChunks * $Tokens) -or
    [int64]$hostMap.development_tokens -ne ([int64]$DevelopmentChunks * $Tokens)) {
  throw 'HOST_EVALUATOR_CAPACITY_MISMATCH'
}
if ($hostMap.parameter_hash -ne $reportMap.checkpoint_parameter_hash) {
  throw "HOST_EVALUATOR_PARAMETER_HASH_MISMATCH: device and CPU checkpoint identities differ"
}
Write-Host "CPU evaluator decoded checkpoint identity=verified parameter_hash=$($hostMap.parameter_hash)"
Write-Host "HTP-native eval PASS: seed=$Seed layers=$Layers step=$CheckpointStep"
Write-Host "Reports: $reportRoot"
} finally {
  [void](Stop-PhoneLmCompletedHeadlessProcesses -Adb $adb -Device $device -Package $package -ExpectedRunId $RunId)
  Stop-PhoneLmOwnedInstrumentation -Process $instrument
}
