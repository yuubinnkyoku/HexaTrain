# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Nicopedia real-text HTP training runner.
#
# Pushes the minimal private tokenized pilot input (train_pilot.bin,
# NPRTBYTEV1) into the app-private files directory and drives
# QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA, which runs the same real-text
# training batches through the CPU reference and the QNN HTP graph with
# identical input/initial-parameter identity.  Results remain under
# build/reports; only aggregate fields are published by the allow-list
# exporter.
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [int]$Seed = 1,
  [int]$Layers = 19,
  [int]$Steps = 32,
  [int]$Tokens = 32,  # context window length (8..256; 32 = legacy T32 behavior)
  [int]$Dimension = 16,
  [int]$FeedForwardDimension = 32,
  [int]$BatchSize = 8,   # canonical pilot config (protocol.json): 8 samples/step
  [string]$CachePath = "",
  [int]$PollLimit = 7200,
  [int]$PollSeconds = 2,
  [int]$ProgressEverySeconds = 30,
  [int]$CheckpointStallSeconds = 300,
  [int]$ResumeStep = 0,
  [int]$CheckpointInterval = 250,
  [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff'),
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

if ($SelfTest) {
  if ($BatchSize -ne 8) { throw "SELFTEST_BATCH_SIZE_DEFAULT: expected=8 actual=$BatchSize" }
  if ($Layers -ne 19) { throw "SELFTEST_LAYERS_DEFAULT: expected=19 actual=$Layers" }
  if ($Tokens -ne 32) { throw "SELFTEST_TOKENS_DEFAULT: expected=32 actual=$Tokens" }
  if ($Dimension -ne 16 -or $FeedForwardDimension -ne 32) { throw "SELFTEST_MODEL_DIMENSIONS_DEFAULT" }
  $canonicalName = Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 16 -FeedForwardDimension 32 -Step 320
  $candidateName = Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 32 -FeedForwardDimension 64 -Step 320
  if ($canonicalName -ne 'htp-seed1-l19-step320.ckpt' -or
      $candidateName -ne 'htp-seed1-l19-t32-d32-f64-step320.ckpt' -or
      $canonicalName -eq $candidateName) { throw 'SELFTEST_CHECKPOINT_MODEL_IDENTITY' }
  if ($CheckpointInterval -lt 1 -or $PollSeconds -lt 1 -or $PollLimit -lt 1 -or $ProgressEverySeconds -lt 1 -or $CheckpointStallSeconds -lt 1) { throw 'SELFTEST_POLL_CONFIGURATION' }
  if (-not ("status=SUCCESS`n" -match '(?m)^status=(SUCCESS|FAILED)\s*$')) { throw 'SELFTEST_TERMINAL_STATUS' }
  $progressState = [ordered]@{ Count = 0; LastProgressUtc = [DateTime]::UtcNow }
  Update-PhoneLmCheckpointProgress -State $progressState -CheckpointCount 1 -NowUtc ([DateTime]::UtcNow) -StallSeconds 1
  if ($progressState.Count -ne 1) { throw 'SELFTEST_CHECKPOINT_PROGRESS_MUTATION' }
  $progressState.LastProgressUtc = [DateTime]::UtcNow.AddSeconds(-2)
  $stalled = $false
  try { Update-PhoneLmCheckpointProgress -State $progressState -CheckpointCount 1 -NowUtc ([DateTime]::UtcNow) -StallSeconds 1 } catch { $stalled = $_.Exception.Message -match '^CHECKPOINT_PROGRESS_STALLED:' }
  if (-not $stalled) { throw 'SELFTEST_CHECKPOINT_STALL_FAIL_CLOSED' }
  Write-Host 'run_nicopedia_htp_training_self_test=PASS'
  exit 0
}
if ($RunId -notmatch '^[A-Za-z0-9._-]{1,64}$') { throw 'RUN_ID_INVALID' }
if ($Steps -lt 1 -or $Steps -gt 8000) { throw 'NICOPEDIA_L19_HARD_CEILING: Steps must be in 1..8000' }
if ($Tokens -lt 8 -or $Tokens -gt 256) { throw 'NICOPEDIA_TOKENS_INVALID: Tokens must be in 8..256' }
if ($Dimension -lt 2 -or $Dimension -gt 256 -or ($Dimension % 2) -ne 0) { throw 'NICOPEDIA_DIMENSION_INVALID: Dimension must be even and in 2..256' }
if ($FeedForwardDimension -lt 2 -or $FeedForwardDimension -gt 1024) { throw 'NICOPEDIA_FFN_INVALID: FeedForwardDimension must be in 2..1024' }
$root = Split-Path -Parent $PSScriptRoot
$modelTag = if ($Tokens -eq 32 -and $Dimension -eq 16 -and $FeedForwardDimension -eq 32) { '' } else { "-t$Tokens-d$Dimension-f$FeedForwardDimension" }
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = 'com.yuubinnkyoku.phonelm'
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$testApk = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
$reportRoot = Join-Path $root "build\reports\nicopedia-htp-training"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

# The private token cache lives under build/private-data and is never
# committed.  The host pushes only the minimal pilot input the device needs.
$trainingDataRoot = if ($Tokens -eq 32) { 'build\private-data\nicopedia-real-text' } else { 'build\private-data\nicopedia-real-text-t64' }
if (-not $CachePath) { $CachePath = Join-Path $root (Join-Path $trainingDataRoot 'caches\train_pilot.bin') }
if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { throw "PRIVATE_CACHE_MISSING: $CachePath" }
$cacheResolved = [IO.Path]::GetFullPath($CachePath)
$allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $cacheResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CachePath must resolve below the repository build directory"
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
}
# Stage the private tokenized pilot input under the app files directory.
$remoteDir = "files/headless-input/$RunId"
Assert-PhoneLmHeadlessInputFresh -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir
Adb @('shell', 'run-as', $package, 'mkdir', '-p', $remoteDir) | Out-Null
$tmpOnDevice = "/data/local/tmp/phonelm-headless-$RunId-train"
Adb @('push', $cacheResolved, $tmpOnDevice) | Out-Null
Adb @('shell', 'run-as', $package, 'cp', $tmpOnDevice, "$remoteDir/train_pilot.bin") | Out-Null
Adb @('shell', 'rm', '-f', $tmpOnDevice) | Out-Null

# Canonical resume: the previous segment's NPRTCKPTV2 checkpoint is pulled by
# that run and kept under build/reports; it is staged on-device and the mode
# verifies step/seed/config identity before continuing. NPRTCKPTV1 checkpoints
# are rejected by the device (RESUME_ADAM_STATE_MISSING).
if ($ResumeStep -gt 0) {
  if ($ResumeStep -ge $Steps) { throw "RESUME_MUST_BE_BELOW_STEPS: $ResumeStep >= $Steps" }
  $resumeCheckpointName = Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $ResumeStep
  $resumeCheckpoint = Join-Path $reportRoot $resumeCheckpointName
  if (-not (Test-Path -LiteralPath $resumeCheckpoint -PathType Leaf)) {
    throw "RESUME_CHECKPOINT_MISSING: $resumeCheckpoint"
  }
  $tmpCkpt = "/data/local/tmp/phonelm-headless-$RunId-resume"
  Adb @('push', $resumeCheckpoint, $tmpCkpt) | Out-Null
  Adb @('shell', 'run-as', $package, 'cp', $tmpCkpt,
    "$remoteDir/$resumeCheckpointName") | Out-Null
  Adb @('shell', 'rm', '-f', $tmpCkpt) | Out-Null
  Write-Host "RESUME_STAGE checkpoint=$resumeCheckpointName"
}

# Run the NICOPEDIA mode through the debug intent path. Existing processes and
# result markers were checked above; lifecycle remains headless.
Clear-PhoneLmResultMarker -Adb $adb -Device $device -Package $package
$instrumentDir = Join-Path $reportRoot "instrumentation-$RunId"
if (Test-Path -LiteralPath $instrumentDir) { throw 'RUN_ID_REUSE: host instrumentation directory exists' }
[IO.Directory]::CreateDirectory($instrumentDir) | Out-Null
$instrument = $null
$checkpointProgress = [ordered]@{
  Count = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir).Count
  LastProgressUtc = [DateTime]::UtcNow
}
try {
  $instrument = Start-PhoneLmHeadlessInstrumentation -Adb $adb -Device $device -Package $package `
  -Class "$package.HeadlessDeviceTestRunner" -Suite 'nicopedia-long-training' -RunId $RunId `
  -Arguments @{ seed = $Seed; layers = $Layers; heads = 2; tokens = $Tokens; dimension = $Dimension; feedForwardDimension = $FeedForwardDimension; steps = $Steps; batchSize = $BatchSize; resumeStep = $ResumeStep; checkpointInterval = $CheckpointInterval } `
  -StdoutPath (Join-Path $instrumentDir 'stdout.txt') -StderrPath (Join-Path $instrumentDir 'stderr.txt')
$waited = Wait-PhoneLmHeadlessStatus -Process $instrument -Adb $adb -Device $device -Package $package `
  -PollLimit $PollLimit -PollSeconds $PollSeconds -ProgressEverySeconds $ProgressEverySeconds -Label "training-step-$Steps" `
  -ExpectedRunId $RunId `
  -PartialPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-steps$Steps-partial-status.json") `
  -StatusProgressAction {
    param($elapsed, $status)
    $phase = [regex]::Match($status, '"current_phase"\s*:\s*"([^"]*)"').Groups[1].Value
    $done = [regex]::Match($status, '"completed_tests"\s*:\s*(\d+)').Groups[1].Value
    $total = [regex]::Match($status, '"total_tests"\s*:\s*(\d+)').Groups[1].Value
    $ckpts = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir)
    Update-PhoneLmCheckpointProgress -State $checkpointProgress -CheckpointCount $ckpts.Count `
      -NowUtc ([DateTime]::UtcNow) -StallSeconds $CheckpointStallSeconds
    Write-Host "progress phase=training elapsed_seconds=$elapsed status_phase=$phase completed=$done/$total checkpoint_count=$($ckpts.Count)"
  } `
  -ConditionAction {
    param($elapsed)
    $state = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase "training-$elapsed-sec"
    Write-Host "health elapsed_seconds=$elapsed thermal=$($state.thermal_status) battery_temp_c=$($state.battery_temperature_c) battery_voltage_mv=$($state.battery_voltage_mv)"
  } `
  -FocusAction {
    param($elapsed)
    if ((Get-PhoneLmTopPackage -Adb $adb -Device $device) -eq $package) { throw 'FOCUS_TAKEOVER_DETECTED' }
  }
$result = Get-PhoneLmHeadlessReport -StatusJson $waited.StatusJson -Adb $adb -Device $device -Package $package
Assert-PhoneLmHeadlessNoActivity -Text $result
$deviceCompleted = $waited.StatusJson -match '"status"\s*:\s*"PASSED"' -and $result -match '(?m)^status=SUCCESS\s*$'
if (-not $deviceCompleted) {
  if ($waited.ProcessExitCode -ne 0) { throw "INSTRUMENTATION_EXIT_FAILURE: code=$($waited.ProcessExitCode)" }
  $result | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-steps$Steps-result.txt") -Encoding utf8
  throw "NICOPEDIA_HTP_FAILED: seed=$Seed layers=$Layers steps=$Steps"
}
# The device terminal state (status PASSED + report status=SUCCESS for the
# expected run) is the authoritative completion signal. The host `am
# instrument` wrapper can exit nonzero on an adb transport hiccup after the
# device has already finished and written its report; that is a host artifact,
# not a failed segment. Genuine device failures still abort above.
if ($waited.ProcessExitCode -ne 0) {
  Write-Host "WARN INSTRUMENTATION_EXIT_NONZERO_AFTER_DEVICE_COMPLETION code=$($waited.ProcessExitCode) (device run PASSED; wrapper exit treated as host transport artifact)"
}
$reportMap = Assert-PhoneLmHealthReport -Text $result -ExpectedBuildId $ExpectedBuildId -ExpectedStep $Steps -Kind training
if (-not $reportMap.Contains('model_dimension') -or -not $reportMap.Contains('feed_forward_dimension') -or
    [int]$reportMap.model_dimension -ne $Dimension -or [int]$reportMap.feed_forward_dimension -ne $FeedForwardDimension) {
  throw 'TRAINING_REPORT_MODEL_IDENTITY_MISMATCH'
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
# The serial is recorded in the private report for the same-device
# reattach check; it is stripped by the public exporter.
$annotated | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-steps$Steps-result.txt") -Encoding utf8
$annotated | Add-Content -LiteralPath (Join-Path $reportRoot "device-identity-private.txt") -Encoding utf8
# Pull every interval NPRTCKPTV2 checkpoint and the loss curve back to
# build/reports. Both stay out of the public bundle and out of git.
$checkpointNames = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir)
$expectedSteps = @()
$firstExpected = if ($ResumeStep -gt 0) { $ResumeStep + $CheckpointInterval } else { $CheckpointInterval }
for ($s = $firstExpected; $s -le $Steps; $s += $CheckpointInterval) { $expectedSteps += $s }
if ($expectedSteps -notcontains $Steps) { $expectedSteps += $Steps }
foreach ($expected in $expectedSteps) {
  $name = Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $expected
  if ($checkpointNames -notcontains $name) { throw "CHECKPOINT_INTERVAL_MISSING: $name" }
}
foreach ($name in $checkpointNames) {
  if ($name -notmatch "^htp-seed$Seed-l$Layers(-t$Tokens-d$Dimension-f$FeedForwardDimension)?-step(\d+)\.ckpt$") { continue }
  $stepName = [int]$Matches[2]
  $local = Join-Path $reportRoot $name
  $pulled = Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
    -RemotePath "$remoteDir/$name" -LocalPath $local -MinimumBytes 1024
  $header = Get-PhoneLmCheckpointHeaders -Path $local
  if ($header.Step -ne $stepName -or $header.Seed -ne $Seed -or $header.Layers -ne $Layers -or $header.Heads -ne 2 -or $header.Vocabulary -ne 256 -or $header.Tokens -ne $Tokens -or $header.Dimension -ne $Dimension -or $header.FeedForward -ne $FeedForwardDimension) {
    throw "CHECKPOINT_IDENTITY_MISMATCH: $name"
  }
  if ($header.Magic -ne 'NPRTCKPTV2') { throw "CHECKPOINT_RESUME_FORMAT_INVALID: $name" }
  # When the production evaluator and held-out caches are available, decode
  # every pulled checkpoint through the same host path used by the eval runner
  # (one chunk is sufficient for identity/finiteness; full-cap eval is a
  # separate milestone).  Header validation above remains fail-closed.
  $hostEvalExe = Join-Path $root 'build\host-tests\htp_checkpoint_eval.exe'
  $validationHost = Join-Path $root (Join-Path $trainingDataRoot 'caches\validation.bin')
  $developmentHost = Join-Path $root (Join-Path $trainingDataRoot 'caches\development.bin')
  if (-not (Test-Path -LiteralPath $hostEvalExe -PathType Leaf) -or -not (Test-Path -LiteralPath $validationHost -PathType Leaf) -or -not (Test-Path -LiteralPath $developmentHost -PathType Leaf)) {
    throw 'HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE'
  }
  $hostDecoded = & $hostEvalExe $local $validationHost $developmentHost 1 1
  if ($LASTEXITCODE -ne 0) { throw "HOST_CHECKPOINT_EVALUATOR_DECODE_FAILED: $name" }
  $hostIdentity = Get-PhoneLmKeyValueMap -Text ($hostDecoded -join "`n")
  foreach ($field in @('seed', 'layers', 'dimension', 'feed_forward_dimension', 'step', 'parameter_hash', 'finite')) { if (-not $hostIdentity.Contains($field)) { throw "HOST_CHECKPOINT_EVALUATOR_FIELD_MISSING: $field" } }
  if ([int]$hostIdentity.seed -ne $Seed -or [int]$hostIdentity.layers -ne $Layers -or [int]$hostIdentity.dimension -ne $Dimension -or [int]$hostIdentity.feed_forward_dimension -ne $FeedForwardDimension -or [int]$hostIdentity.step -ne $stepName -or $hostIdentity.finite -ne 'true') { throw "HOST_CHECKPOINT_EVALUATOR_IDENTITY_MISMATCH: $name" }
  Write-Host "checkpoint step=$stepName size=$($pulled.Size) sha256=$($pulled.Sha256) identity=verified"
}
$finalCkptName = Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $Steps
# The device writes the curve with an untagged name; the host keeps the
# legacy name for the anchor and a model-tagged name for every other context
# so width experiments cannot collide with the production curve files.
$curveRemote = "training-curve-$Steps.csv"
$curveLocal = if ($modelTag) { "training-curve$modelTag-$Steps.csv" } else { $curveRemote }
if ($checkpointNames.Count -eq 0) { throw 'CHECKPOINT_PULL_VERIFY_FAILED: no checkpoints' }
Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
  -RemotePath "$remoteDir/$curveRemote" -LocalPath (Join-Path $reportRoot $curveLocal) -MinimumBytes 1 | Out-Null
Write-Host "Pulled $($checkpointNames.Count) interval checkpoints + $curveLocal"
Write-Host "PASS NICOPEDIA_HTP seed=$Seed layers=$Layers steps=$Steps"
Write-Host "Reports: $reportRoot"
} finally {
  [void](Stop-PhoneLmCompletedHeadlessProcesses -Adb $adb -Device $device -Package $package -ExpectedRunId $RunId)
  Stop-PhoneLmOwnedInstrumentation -Process $instrument
}
