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
  [ValidateSet(256, 1024)][int]$Vocabulary = 256,
  [int]$Dimension = 32,
  [int]$FeedForwardDimension = 32,
  [int]$BatchSize = 8,   # canonical pilot config (protocol.json): 8 samples/step
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$LearningRate = '0.003',
  [ValidateSet('constant','linear_decay','sqrt_decay')][string]$LearningRateSchedule = 'constant',
  [ValidateRange(0,100000)][int]$DecayStartStep = 0,
  [ValidateRange(0,100000)][int]$DecayEndStep = 0,
  [ValidateRange(0,100000)][int]$ScheduleTotalSteps = 0,
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$TargetLearningRate = '',
  [switch]$ExperimentFork,
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$ParentLearningRate = '0',
  [ValidateSet('Adam','Muon')][string]$Optimizer = 'Adam',
  [ValidateSet('none','headwise_g1_sigmoid','headwise_g1_scale2_identity','fixed_half')][string]$AttentionGate = 'none',
  [ValidateSet('CPU','HVX')][string]$MuonBackend = 'CPU',
  [string]$HexagonSdkRoot = '',
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$MuonLearningRate = '0.010',
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$MuonMomentum = '0.95',
  [ValidateRange(1,99)][int]$MuonNsSteps = 5,
  [string]$CachePath = "",
  [string]$EvalCacheRoot = "",
  [string]$TokenizerModelPath = "",
  [string]$ReportRoot = "",
  [string]$AppApkPath = "",
  [string]$AndroidTestApkPath = "",
  [string]$AuditTelemetryDirectory = "",
  [string]$ExpectedDeviceSerial = "",
  [ValidateRange(0, 86400)][int]$MidTelemetryAfterSeconds = 0,
  [int]$PollLimit = 7200,
  [int]$PollSeconds = 2,
  [int]$ProgressEverySeconds = 30,
  [int]$CheckpointStallSeconds = 300,
  [int]$ResumeStep = 0,
  [int]$CheckpointInterval = 250,
  [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff'),
  [string]$CacheSourceRunId = "",
  [switch]$BuildInstallOnly,
  [switch]$OneUpdateProbe,
  [switch]$AllowQualityFailure,
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

function Get-PhoneLmExpectedLearningRate {
  param(
    [Parameter(Mandatory = $true)][ValidateSet('constant','linear_decay','sqrt_decay')][string]$Schedule,
    [Parameter(Mandatory = $true)][double]$PeakLearningRate,
    [Parameter(Mandatory = $true)][double]$TargetLearningRate,
    [Parameter(Mandatory = $true)][int]$DecayStartStep,
    [Parameter(Mandatory = $true)][int]$DecayEndStep,
    [Parameter(Mandatory = $true)][int]$Step
  )
  if ($Schedule -eq 'constant' -or $Step -le $DecayStartStep) { return $PeakLearningRate }
  if ($Step -ge $DecayEndStep) { return $TargetLearningRate }
  $progress = ($Step - $DecayStartStep) / [double]($DecayEndStep - $DecayStartStep)
  if ($Schedule -eq 'sqrt_decay') {
    $shape = 1.0 - [math]::Sqrt($progress)
    return $TargetLearningRate + (($PeakLearningRate - $TargetLearningRate) * $shape)
  }
  return $PeakLearningRate + ($progress * ($TargetLearningRate - $PeakLearningRate))
}

function Get-PhoneLmMemAvailableKilobytes {
  param([AllowEmptyString()][string]$MemInfoText)
  $memoryMatch = [regex]::Match($MemInfoText, '(?im)^MemAvailable:\s*(\d+)\s*kB\s*$')
  if (-not $memoryMatch.Success) { throw 'DEVICE_MEMAVAILABLE_UNAVAILABLE' }
  return [long]$memoryMatch.Groups[1].Value
}

function Get-PhoneLmAppPrivateRoot {
  param([AllowEmptyString()][string]$WorkingDirectory)
  $root = $WorkingDirectory.Trim().TrimEnd('/')
  if ($root -notmatch '^/data/(?:user/\d+|data)/com\.yuubinnkyoku\.phonelm$') {
    throw 'APP_PRIVATE_ROOT_REJECTED'
  }
  return $root
}

if ($SelfTest) {
  $transferHash = 'a' * 64
  Assert-PhoneLmBinaryTransferIdentity -ExpectedSize 100 -ExpectedSha256 $transferHash -ActualSize 100 -ActualSha256 $transferHash
  foreach ($badTransfer in @(@{ Size = 99; Hash = $transferHash }, @{ Size = 101; Hash = $transferHash }, @{ Size = 100; Hash = ('b' * 64) })) {
    $transferRejected = $false
    try { Assert-PhoneLmBinaryTransferIdentity -ExpectedSize 100 -ExpectedSha256 $transferHash -ActualSize $badTransfer.Size -ActualSha256 $badTransfer.Hash } catch { $transferRejected = $_.Exception.Message -eq 'ADB_BINARY_TRANSFER_IDENTITY_MISMATCH' }
    if (-not $transferRejected) { throw 'SELFTEST_BINARY_TRANSFER_FAIL_CLOSED' }
  }
  if ($BatchSize -ne 8) { throw "SELFTEST_BATCH_SIZE_DEFAULT: expected=8 actual=$BatchSize" }
  if ($Layers -ne 19) { throw "SELFTEST_LAYERS_DEFAULT: expected=19 actual=$Layers" }
  if ($Tokens -ne 32) { throw "SELFTEST_TOKENS_DEFAULT: expected=32 actual=$Tokens" }
  if ($Vocabulary -ne 256) { throw "SELFTEST_VOCABULARY_DEFAULT: expected=256 actual=$Vocabulary" }
  if ($Dimension -ne 32 -or $FeedForwardDimension -ne 32) {
      throw "SELFTEST_MODEL_DIMENSIONS_DEFAULT: expected=D32/FFN32 actual=D$Dimension/FFN$FeedForwardDimension"
  }
  if ($LearningRate -ne '0.003') { throw "SELFTEST_LEARNING_RATE_DEFAULT: expected=0.003 actual=$LearningRate" }
  if ($LearningRateSchedule -ne 'constant' -or $DecayStartStep -ne 0 -or $DecayEndStep -ne 0 -or $ScheduleTotalSteps -ne 0 -or $TargetLearningRate -ne '' -or $ExperimentFork -or $ParentLearningRate -ne '0') { throw 'SELFTEST_SCHEDULE_DEFAULT' }
  if ($Optimizer -ne 'Adam' -or $MuonBackend -ne 'CPU' -or $MuonLearningRate -ne '0.010' -or $MuonMomentum -ne '0.95' -or $MuonNsSteps -ne 5) { throw 'SELFTEST_OPTIMIZER_DEFAULT' }
  $sqrtCases = @(
    [pscustomobject]@{ Step = 6000; Expected = 0.0022 },
    [pscustomobject]@{ Step = 6001; Expected = 0.002153042572472504 },
    [pscustomobject]@{ Step = 7000; Expected = 0.00071507575950825 },
    [pscustomobject]@{ Step = 7999; Expected = 0.000100525065641411 },
    [pscustomobject]@{ Step = 8000; Expected = 0.0001 }
  )
  foreach ($case in $sqrtCases) {
    $actual = Get-PhoneLmExpectedLearningRate -Schedule 'sqrt_decay' -PeakLearningRate 0.0022 -TargetLearningRate 0.0001 -DecayStartStep 6000 -DecayEndStep 8000 -Step $case.Step
    if ([math]::Abs($actual - $case.Expected) -gt 1.0e-12) { throw "SELFTEST_SQRT_FORMULA: step=$($case.Step) actual=$actual expected=$($case.Expected)" }
  }
  $scaledMidpoint = Get-PhoneLmExpectedLearningRate -Schedule 'sqrt_decay' -PeakLearningRate 0.0022 -TargetLearningRate 0.0005 -DecayStartStep 6000 -DecayEndStep 8000 -Step 7000
  if ([math]::Abs($scaledMidpoint - 0.000997918471982869) -gt 1.0e-12) { throw "SELFTEST_SQRT_TARGET_SCALING: actual=$scaledMidpoint" }

  # Production anchor: T32/D32/FFN32 uses the canonical untagged filename.
  $canonicalName = Get-PhoneLmCheckpointName `
      -Seed 1 `
      -Layers 19 `
      -Tokens 32 `
      -Dimension 32 `
      -FeedForwardDimension 32 `
      -Step 320

  # Historical/research D16 is no longer canonical and must be explicitly tagged.
  $d16Name = Get-PhoneLmCheckpointName `
      -Seed 1 `
      -Layers 19 `
      -Tokens 32 `
      -Dimension 16 `
      -FeedForwardDimension 32 `
      -Step 320

  # Other non-anchor configurations remain explicitly tagged.
  $candidateName = Get-PhoneLmCheckpointName `
      -Seed 1 `
      -Layers 19 `
      -Tokens 32 `
      -Dimension 32 `
      -FeedForwardDimension 64 `
      -Step 320

  if ($canonicalName -ne 'htp-seed1-l19-step320.ckpt' -or
      $d16Name -ne 'htp-seed1-l19-t32-d16-f32-step320.ckpt' -or
      $candidateName -ne 'htp-seed1-l19-t32-d32-f64-step320.ckpt' -or
      $canonicalName -eq $d16Name -or
      $canonicalName -eq $candidateName -or
      $d16Name -eq $candidateName) {
      throw 'SELFTEST_CHECKPOINT_MODEL_IDENTITY'
  }
  if ($CheckpointInterval -lt 1 -or $PollSeconds -lt 1 -or $PollLimit -lt 1 -or $ProgressEverySeconds -lt 1 -or $CheckpointStallSeconds -lt 1) { throw 'SELFTEST_POLL_CONFIGURATION' }
  if (-not (Test-PhoneLmHeadlessOwnerPid -StatusJson '{"status":"RUNNING","pid":1234}' -PidOutput '1234 9876') -or
      (Test-PhoneLmHeadlessOwnerPid -StatusJson '{"status":"RUNNING","pid":1234}' -PidOutput '91234')) {
    throw 'SELFTEST_HEADLESS_PROCESS_EXIT_DETECTION'
  }
  if ((Get-PhoneLmMemAvailableKilobytes -MemInfoText "MemTotal: 100000 kB`nMemAvailable:    5302244 kB`nBuffers: 10 kB") -ne 5302244) {
    throw 'SELFTEST_MEMAVAILABLE_PARSE'
  }
  $memInfoRejected = $false
  try { [void](Get-PhoneLmMemAvailableKilobytes -MemInfoText 'MemTotal: 100000 kB') } catch { $memInfoRejected = $_.Exception.Message -eq 'DEVICE_MEMAVAILABLE_UNAVAILABLE' }
  if (-not $memInfoRejected) { throw 'SELFTEST_MEMAVAILABLE_FAIL_CLOSED' }
  if ((Get-PhoneLmAppPrivateRoot -WorkingDirectory '/data/user/0/com.yuubinnkyoku.phonelm/') -ne '/data/user/0/com.yuubinnkyoku.phonelm' -or
      (Get-PhoneLmAppPrivateRoot -WorkingDirectory '/data/data/com.yuubinnkyoku.phonelm/') -ne '/data/data/com.yuubinnkyoku.phonelm') {
    throw 'SELFTEST_APP_PRIVATE_ROOT_ACCEPT'
  }
  $privateRootRejected = $false
  try { [void](Get-PhoneLmAppPrivateRoot -WorkingDirectory '/data/local/tmp') } catch { $privateRootRejected = $_.Exception.Message -eq 'APP_PRIVATE_ROOT_REJECTED' }
  if (-not $privateRootRejected) { throw 'SELFTEST_APP_PRIVATE_ROOT_REJECT' }
  $inactiveEvidence = @{ status_state = 'terminal'; status_uncertain = $false; process_present = $true; test_process_present = $false; fgs_present = $false; service_present = $false; service_uncertain = $false; activity_known = $true; activity_active = $false; task_present = $true }
  $inactiveDecision = Resolve-PhoneLmRunConflict $inactiveEvidence
  if ($inactiveDecision.active -or @($inactiveDecision.reasons) -notcontains 'CACHED_PROCESS_ONLY' -or @($inactiveDecision.reasons) -notcontains 'INACTIVE_TASK_ONLY') { throw 'SELFTEST_CACHED_PROCESS_FALSE_POSITIVE' }
  $heartbeatEvidence = $inactiveEvidence.Clone(); $heartbeatEvidence.status_state = 'active'; $heartbeatDecision = Resolve-PhoneLmRunConflict $heartbeatEvidence
  if (-not $heartbeatDecision.active -or @($heartbeatDecision.reasons) -notcontains 'ACTIVE_HEARTBEAT') { throw 'SELFTEST_ACTIVE_HEARTBEAT_NOT_BLOCKED' }
  $staleEvidence = $inactiveEvidence.Clone(); $staleEvidence.status_state = 'stale'; $staleDecision = Resolve-PhoneLmRunConflict $staleEvidence
  if ($staleDecision.active -or @($staleDecision.reasons) -notcontains 'STALE_HEARTBEAT_ONLY') { throw 'SELFTEST_STALE_HEARTBEAT_FALSE_POSITIVE' }
  $fgsEvidence = $inactiveEvidence.Clone(); $fgsEvidence.fgs_present = $true; $fgsDecision = Resolve-PhoneLmRunConflict $fgsEvidence
  if (-not $fgsDecision.active -or @($fgsDecision.reasons) -notcontains 'ACTIVE_FGS') { throw 'SELFTEST_ACTIVE_FGS_NOT_BLOCKED' }
  $activityEvidence = $inactiveEvidence.Clone(); $activityEvidence.activity_active = $true; $activityDecision = Resolve-PhoneLmRunConflict $activityEvidence
  if (-not $activityDecision.active -or @($activityDecision.reasons) -notcontains 'ACTIVE_ACTIVITY') { throw 'SELFTEST_ACTIVE_ACTIVITY_NOT_BLOCKED' }
  $unknownEvidence = $inactiveEvidence.Clone(); $unknownEvidence.activity_known = $false; $unknownDecision = Resolve-PhoneLmRunConflict $unknownEvidence
  if (-not $unknownDecision.active -or @($unknownDecision.reasons) -notcontains 'RUN_STATE_UNCERTAIN') { throw 'SELFTEST_UNKNOWN_STATE_NOT_BLOCKED' }
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
if ($CacheSourceRunId -and $CacheSourceRunId -notmatch '^[A-Za-z0-9._-]{1,64}$') { throw 'CACHE_SOURCE_RUN_ID_INVALID' }
$deviceCacheSourceRunId = if ($CacheSourceRunId) { $CacheSourceRunId } else { $RunId }
if ($Steps -lt 1 -or $Steps -gt 100000) { throw 'NICOPEDIA_L19_HARD_CEILING: Steps must be in 1..100000' }
try { $muonLearningRateValue = [double]::Parse($MuonLearningRate, [Globalization.CultureInfo]::InvariantCulture) } catch { throw 'NICOPEDIA_MUON_LEARNING_RATE_INVALID' }
try { $muonMomentumValue = [double]::Parse($MuonMomentum, [Globalization.CultureInfo]::InvariantCulture) } catch { throw 'NICOPEDIA_MUON_MOMENTUM_INVALID' }
if (-not [double]::IsFinite($muonLearningRateValue) -or $muonLearningRateValue -le 0 -or $muonLearningRateValue -gt 1) { throw 'NICOPEDIA_MUON_LEARNING_RATE_INVALID' }
if (-not [double]::IsFinite($muonMomentumValue) -or $muonMomentumValue -lt 0 -or $muonMomentumValue -ge 1) { throw 'NICOPEDIA_MUON_MOMENTUM_INVALID' }
if ($Optimizer -eq 'Muon') {
  if ($Vocabulary -ne 1024 -or $Tokens -ne 32 -or $Dimension -ne 64 -or $FeedForwardDimension -ne 128 -or $Layers -ne 19 -or $BatchSize -ne 8) { throw 'NICOPEDIA_MUON_ARCHITECTURE_MISMATCH' }
  if ($MuonMomentum -ne '0.95' -or $MuonNsSteps -ne 5) { throw 'NICOPEDIA_MUON_V1_HYPERPARAMETER_MISMATCH' }
}
try { $learningRateValue = [double]::Parse($LearningRate, [Globalization.CultureInfo]::InvariantCulture) } catch { throw 'NICOPEDIA_LEARNING_RATE_INVALID' }
if (-not [double]::IsFinite($learningRateValue) -or $learningRateValue -le 0 -or $learningRateValue -gt 1) { throw 'NICOPEDIA_LEARNING_RATE_INVALID' }
if (-not $TargetLearningRate) { $TargetLearningRate = $LearningRate }
try { $targetLearningRateValue = [double]::Parse($TargetLearningRate, [Globalization.CultureInfo]::InvariantCulture) } catch { throw 'NICOPEDIA_TARGET_LEARNING_RATE_INVALID' }
try { $parentLearningRateValue = [double]::Parse($ParentLearningRate, [Globalization.CultureInfo]::InvariantCulture) } catch { throw 'NICOPEDIA_PARENT_LEARNING_RATE_INVALID' }
if (-not [double]::IsFinite($targetLearningRateValue) -or $targetLearningRateValue -lt 0 -or $targetLearningRateValue -gt 1) { throw 'NICOPEDIA_TARGET_LEARNING_RATE_INVALID' }
if (-not [double]::IsFinite($parentLearningRateValue) -or $parentLearningRateValue -lt 0 -or $parentLearningRateValue -gt 1) { throw 'NICOPEDIA_PARENT_LEARNING_RATE_INVALID' }
if ($ScheduleTotalSteps -eq 0) { $ScheduleTotalSteps = $Steps }
if ($LearningRateSchedule -eq 'constant') {
  if ($DecayStartStep -ne 0 -or $DecayEndStep -ne 0 -or $ExperimentFork) { throw 'NICOPEDIA_CONSTANT_SCHEDULE_INVALID' }
} elseif ($LearningRateSchedule -eq 'linear_decay') {
  # Linear HPO forks keep the validated peak/parent LR and may vary only the
  # declared target endpoint.  Restrict the endpoint to the Schedule-v2a
  # allow-list so an accidental cross-experiment resume cannot silently run.
  # G1 high-LR stress grid scales Aux Adam/Muon by 1.0/1.25/1.5/2.0 while
  # keeping the 0.0022 : 0.0001 ratio; accept those scaled peaks/targets.
  $allowedLinearPeakLearningRates = @('0.0022', '0.00275', '0.0033', '0.0044')
  $allowedLinearTargetLearningRates = @('0.0015', '0.0010', '0.0007', '0.0004', '0.0002', '0.0001', '0.0000', '0.000125', '0.00015')
  if (-not ($allowedLinearPeakLearningRates -contains $LearningRate) -or
      -not ($allowedLinearTargetLearningRates -contains $TargetLearningRate) -or
      $DecayStartStep -le 0 -or $DecayStartStep -ge $DecayEndStep -or
      $DecayEndStep -gt $ScheduleTotalSteps -or -not $ExperimentFork -or
      $parentLearningRateValue -ne $learningRateValue) { throw 'NICOPEDIA_LINEAR_SCHEDULE_INVALID' }
} else {
  # Schedule-v2c is deliberately a single fixed-target shape comparison.  Do
  # not allow the runner to silently turn this into another LR/parent sweep.
  if ($learningRateValue -ne 0.0022 -or $TargetLearningRate -ne '0.0001' -or
      $DecayStartStep -le 0 -or $DecayStartStep -ge $DecayEndStep -or
      $DecayEndStep -gt $ScheduleTotalSteps -or -not $ExperimentFork -or
      $parentLearningRateValue -ne 0.0022) { throw 'NICOPEDIA_SQRT_SCHEDULE_INVALID' }
}
if ($OneUpdateProbe -and $Steps -ne 1) { throw 'NICOPEDIA_DFFN_PROBE_REQUIRES_STEPS_1' }
if ($OneUpdateProbe -and $BatchSize -ne 8) { throw 'NICOPEDIA_DFFN_PROBE_REQUIRES_BATCH_8' }
if ($OneUpdateProbe -and $ResumeStep -ne 0) { throw 'NICOPEDIA_DFFN_PROBE_DOES_NOT_SUPPORT_RESUME' }
if ($OneUpdateProbe -and $Vocabulary -ne 256) { throw 'NICOPEDIA_DFFN_PROBE_IS_LEGACY_V256_ONLY' }
if ($Tokens -lt 8 -or $Tokens -gt 256) { throw 'NICOPEDIA_TOKENS_INVALID: Tokens must be in 8..256' }
if ($Dimension -lt 2 -or $Dimension -gt 256 -or ($Dimension % 2) -ne 0) { throw 'NICOPEDIA_DIMENSION_INVALID: Dimension must be even and in 2..256' }
if ($FeedForwardDimension -lt 2 -or $FeedForwardDimension -gt 1024) { throw 'NICOPEDIA_FFN_INVALID: FeedForwardDimension must be in 2..1024' }
$root = Split-Path -Parent $PSScriptRoot
$vocabularyTag = if ($Vocabulary -eq 256) { '' } else { "-v$Vocabulary" }
$shapeTag = if ($Tokens -eq 32 -and $Dimension -eq 32 -and $FeedForwardDimension -eq 32) { '' } else { "-t$Tokens-d$Dimension-f$FeedForwardDimension" }
$modelTag = "$vocabularyTag$shapeTag"
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = 'com.yuubinnkyoku.phonelm'
$apk = if ($AppApkPath) { if ([IO.Path]::IsPathRooted($AppApkPath)) { [IO.Path]::GetFullPath($AppApkPath) } else { [IO.Path]::GetFullPath((Join-Path $root $AppApkPath)) } } else { Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk' }
$testApk = if ($AndroidTestApkPath) { if ([IO.Path]::IsPathRooted($AndroidTestApkPath)) { [IO.Path]::GetFullPath($AndroidTestApkPath) } else { [IO.Path]::GetFullPath((Join-Path $root $AndroidTestApkPath)) } } else { Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk' }
$reportDirectory = if ($Vocabulary -eq 1024) {
  "build\reports\nicopedia-htp-training-v1024"
} else { "build\reports\nicopedia-htp-training" }
$reportCandidate = if ($ReportRoot) {
  if ([IO.Path]::IsPathRooted($ReportRoot)) { $ReportRoot } else { Join-Path $root $ReportRoot }
} else { Join-Path $root $reportDirectory }
$reportRoot = [IO.Path]::GetFullPath($reportCandidate)
$allowedReportRoot = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $reportRoot.StartsWith($allowedReportRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'ReportRoot must resolve below the repository build directory' }
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null
$auditTelemetryRoot = ''
if ($AuditTelemetryDirectory) {
  $auditTelemetryRoot = if ([IO.Path]::IsPathRooted($AuditTelemetryDirectory)) { [IO.Path]::GetFullPath($AuditTelemetryDirectory) } else { [IO.Path]::GetFullPath((Join-Path $root $AuditTelemetryDirectory)) }
  $allowedTelemetryRoot = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
  if (-not $auditTelemetryRoot.StartsWith($allowedTelemetryRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'AuditTelemetryDirectory must resolve below the repository build directory' }
}

# The private token cache lives under build/private-data and is never
# committed.  The host pushes only the minimal pilot input the device needs.
$trainingDataRoot = if ($Vocabulary -eq 1024) { 'build\private-data\nicopedia-real-text-bpe-v1024' } elseif ($Tokens -eq 32) { 'build\private-data\nicopedia-real-text' } else { 'build\private-data\nicopedia-real-text-t64' }
if (-not $CachePath) { $CachePath = Join-Path $root (Join-Path $trainingDataRoot 'caches\train_pilot.bin') }
if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { throw "PRIVATE_CACHE_MISSING: $CachePath" }
$cacheResolved = [IO.Path]::GetFullPath($CachePath)
$allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
$expandedTrainManifest = $null
if ($Optimizer -eq 'Muon' -and [IO.Path]::GetFileName($cacheResolved) -eq 'train-expanded.bin') {
  $cacheDirectory = Split-Path -Parent $cacheResolved
  $privateDataDirectory = Split-Path -Parent $cacheDirectory
  $expandedManifestPath = Join-Path $privateDataDirectory 'expanded-train-manifest.json'
  if (-not (Test-Path -LiteralPath $expandedManifestPath -PathType Leaf)) { throw 'EXPANDED_TRAIN_MANIFEST_MISSING' }
  $expandedTrainManifest = Get-Content -LiteralPath $expandedManifestPath -Raw | ConvertFrom-Json
  if ($expandedTrainManifest.schema -ne 'NICOPEDIA_G1_EXPANDED_TRAIN_V1' -or
      $expandedTrainManifest.final_test_opened -ne $false -or
      $expandedTrainManifest.final_test_tokenized_for_training_or_evaluation -ne $false -or
      $expandedTrainManifest.final_test_used_for_training_or_quality -ne $false -or
      $expandedTrainManifest.final_test_evaluated -ne $false -or
      $expandedTrainManifest.final_test_body_scan_scope -ne 'cleaning_and_exact_text_deduplication_only' -or
      $expandedTrainManifest.final_test_dedupe_only_scan.performed -ne $true -or
      $expandedTrainManifest.final_test_dedupe_only_scan.model_facing_or_quality_facing_access -ne $false -or
      $expandedTrainManifest.cache.format -ne 'NPRTBPEV1' -or
      [int]$expandedTrainManifest.cache.context -ne $Tokens -or
      [long]$expandedTrainManifest.cache.records -lt 800000 -or
      [long]$expandedTrainManifest.cache.records -gt 15000000 -or
      $expandedTrainManifest.tokenizer.kind -ne 'byte_bpe' -or
      [int]$expandedTrainManifest.tokenizer.vocabulary -ne $Vocabulary -or
      $expandedTrainManifest.tokenizer.sha256 -ne 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798' -or
      [int]$expandedTrainManifest.training_order.hard_ceiling_steps -ne 100000 -or
      [long]$expandedTrainManifest.training_order.selection_count -ne 800000 -or
      $expandedTrainManifest.split_leakage.selected_train_intersects_validation -ne 0 -or
      $expandedTrainManifest.split_leakage.selected_train_intersects_development -ne 0 -or
      $expandedTrainManifest.split_leakage.selected_train_intersects_final_test -ne 0) {
    throw 'EXPANDED_TRAIN_MANIFEST_IDENTITY_MISMATCH'
  }
  if ($Seed -ne 1 -or $Optimizer -ne 'Muon' -or $AttentionGate -ne 'headwise_g1_sigmoid' -or
      $Vocabulary -ne 1024 -or $Tokens -ne 32 -or $Dimension -ne 64 -or
      $FeedForwardDimension -ne 128 -or $Layers -ne 19 -or $BatchSize -ne 8 -or
      $ScheduleTotalSteps -ne 100000 -or $LearningRate -ne '0.0033' -or
      $LearningRateSchedule -ne 'linear_decay' -or $DecayStartStep -ne 4000 -or
      $DecayEndStep -ne 8000 -or $TargetLearningRate -ne '0.00015' -or
      -not $ExperimentFork -or $ParentLearningRate -ne '0.0033' -or
      $MuonLearningRate -ne '0.0075' -or $MuonMomentum -ne '0.95' -or
      $MuonNsSteps -ne 5 -or $CheckpointInterval -ne 1000) {
    throw 'EXPANDED_G1_EXPERIMENT_IDENTITY_MISMATCH'
  }
  $expandedCacheSha256 = (Get-FileHash -LiteralPath $cacheResolved -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($expandedTrainManifest.cache.sha256 -ne "sha256:$expandedCacheSha256") { throw 'EXPANDED_TRAIN_CACHE_SHA256_MISMATCH' }
}
$evalCacheRootCandidate = if ($EvalCacheRoot) {
  if ([IO.Path]::IsPathRooted($EvalCacheRoot)) { $EvalCacheRoot } else { Join-Path $root $EvalCacheRoot }
} else { Join-Path $root (Join-Path $trainingDataRoot 'caches') }
$evalCacheRoot = [IO.Path]::GetFullPath($evalCacheRootCandidate)
if (-not $evalCacheRoot.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'EvalCacheRoot must resolve below the repository build directory' }
foreach ($heldoutName in @('validation.bin', 'development.bin')) {
  if (-not (Test-Path -LiteralPath (Join-Path $evalCacheRoot $heldoutName) -PathType Leaf)) { throw "HELDOUT_CACHE_MISSING: $heldoutName" }
}
if ($null -ne $expandedTrainManifest) {
  foreach ($splitName in @('validation', 'development')) {
    $heldoutPath = Join-Path $evalCacheRoot "$splitName.bin"
    $heldoutSha256 = (Get-FileHash -LiteralPath $heldoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expandedTrainManifest.heldout_cache_identity.$splitName.sha256 -ne "sha256:$heldoutSha256") {
      throw "EXPANDED_TRAIN_HELDOUT_CACHE_SHA256_MISMATCH: $splitName"
    }
  }
}
if (-not $cacheResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CachePath must resolve below the repository build directory"
}
$tokenizerResolved = ''
if ($Vocabulary -eq 1024) {
  if (-not $TokenizerModelPath) { $TokenizerModelPath = Join-Path $root (Join-Path $trainingDataRoot 'tokenizer\byte-bpe-v1024.model') }
  if (-not (Test-Path -LiteralPath $TokenizerModelPath -PathType Leaf)) { throw "PRIVATE_TOKENIZER_MODEL_MISSING: $TokenizerModelPath" }
  $tokenizerResolved = [IO.Path]::GetFullPath($TokenizerModelPath)
  if (-not $tokenizerResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'TokenizerModelPath must resolve below the repository build directory' }
  if ($null -ne $expandedTrainManifest) {
    $actualTokenizerSha256 = (Get-FileHash -LiteralPath $tokenizerResolved -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expandedTrainManifest.tokenizer.sha256 -ne "sha256:$actualTokenizerSha256") { throw 'EXPANDED_TRAIN_TOKENIZER_SHA256_MISMATCH' }
  }
}

if (-not $SkipBuild -and $Optimizer -eq 'Muon' -and $MuonBackend -eq 'HVX') {
  if ([string]::IsNullOrWhiteSpace($HexagonSdkRoot) -or -not (Test-Path -LiteralPath $HexagonSdkRoot -PathType Container)) {
    throw 'MUON_HVX_REQUIRES_HEXAGON_SDK_ROOT'
  }
}
if (-not $SkipBuild) {
  $gradleArgs = @(
    ':app:assembleDebug', ':app:assembleDebugAndroidTest',
    '-Pphonelm.enableQnn=true',
    "-Pqairt.sdkRoot=$QairtSdkRoot",
    "-Pqairt.expectedBuildId=$ExpectedBuildId",
    '--no-daemon'
  )
  if ($Optimizer -eq 'Muon' -and $MuonBackend -eq 'HVX') {
    $gradleArgs += @('-Pphonelm.enableHvxMuon=true', "-Phexagon.sdkRoot=$HexagonSdkRoot")
  }
  & (Join-Path $root 'gradlew.bat') $gradleArgs
  if ($LASTEXITCODE -ne 0) { throw 'APK build failed' }
}
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
if ($ExpectedDeviceSerial -and $deviceInfo.Serial -ne $ExpectedDeviceSerial) { throw 'ADB_EXPECTED_DEVICE_IDENTITY_MISMATCH' }
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
$stateBefore = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'before'
$serial = $deviceInfo.Serial
$model = $deviceInfo.Model
$soc = $deviceInfo.Soc

function Adb {
  param(
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [ValidateRange(1, 600)][int]$TimeoutSeconds = 60
  )
  return (Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments $Arguments `
    -TimeoutSeconds $TimeoutSeconds).Output
}

if (-not $SkipInstall) {
  if (-not (Test-Path -LiteralPath $apk -PathType Leaf) -or -not (Test-Path -LiteralPath $testApk -PathType Leaf)) { throw 'APK_OR_TEST_APK_MISSING' }
  # Repeated large streaming installs can stall the transport. Stage the exact
  # audited bytes once, verify size/SHA, and install locally with bounded calls.
  Install-PhoneLmVerifiedCachedApk -Adb $adb -Device $device -ActivePackage $package `
    -TargetPackage $package -LocalApk $apk
  Install-PhoneLmVerifiedCachedApk -Adb $adb -Device $device -ActivePackage $package `
    -TargetPackage "$package.test" -LocalApk $testApk -TestApk
  # Installation can restore a retained process/task. Recheck live ownership
  # before proceeding; never force-stop an app that may have become active.
  # Instrumentation owns its subsequent process and the same focus gate.
  Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
  Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $package
}
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package $package -LocalApk $apk
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package "$package.test" -LocalApk $testApk
if ($BuildInstallOnly) {
  Write-Host "build_install_only=SUCCESS apk=$apk test_apk=$testApk"
  exit 0
}
# Stage the private tokenized pilot input under the app files directory.
$remoteDir = "files/headless-input/$RunId"
Assert-PhoneLmHeadlessInputFresh -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir
Adb @('shell', 'run-as', $package, 'mkdir', '-p', $remoteDir) | Out-Null
$cacheTransferTimeoutSeconds = 600
$cacheSizeBytes = [IO.FileInfo]::new($cacheResolved).Length
$cacheSha = (Get-FileHash -LiteralPath $cacheResolved -Algorithm SHA256).Hash.ToLowerInvariant()
if ($CacheSourceRunId -and $CacheSourceRunId -ne $RunId) {
  $cacheSourcePath = "files/headless-input/$CacheSourceRunId/train_pilot.bin"
  $sourceIdentity = (Adb -Arguments @('shell', 'run-as', $package, 'sha256sum', $cacheSourcePath) `
    -TimeoutSeconds $cacheTransferTimeoutSeconds).Trim()
  if ($sourceIdentity -notmatch ('^' + [regex]::Escape($cacheSha) + '\s')) { throw 'CACHE_SOURCE_DEVICE_IDENTITY_MISMATCH' }
  $appPrivateRootOutput = Adb @('shell', 'run-as', $package, 'pwd')
  $appPrivateRoot = Get-PhoneLmAppPrivateRoot -WorkingDirectory ([string]::Join("`n", [string[]]$appPrivateRootOutput))
  $cacheSourceAbsolutePath = "$($appPrivateRoot.TrimEnd('/'))/$cacheSourcePath"
  $cacheLink = Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @(
    'shell', 'run-as', $package, 'ln', '-s', $cacheSourceAbsolutePath, "$remoteDir/train_pilot.bin") -AllowFailure
  if ($cacheLink.ExitCode -ne 0) { throw "CACHE_SOURCE_SYMLINK_FAILED: $($cacheLink.Classification)" }
  $linkedIdentity = (Adb -Arguments @('shell', 'run-as', $package, 'sha256sum', "$remoteDir/train_pilot.bin") `
    -TimeoutSeconds $cacheTransferTimeoutSeconds).Trim()
  if ($linkedIdentity -notmatch ('^' + [regex]::Escape($cacheSha) + '\s')) { throw 'CACHE_SYMLINK_IDENTITY_MISMATCH' }
  Write-Host "TRAIN_CACHE_STAGE mode=app-private-symlink source_run_id=$CacheSourceRunId size_bytes=$cacheSizeBytes sha256=$cacheSha device_sha256=verified timeout_seconds=$cacheTransferTimeoutSeconds"
} else {
  $tmpOnDevice = "/data/local/tmp/phonelm-headless-$RunId-train"
  Adb -Arguments @('push', $cacheResolved, $tmpOnDevice) `
    -TimeoutSeconds $cacheTransferTimeoutSeconds | Out-Null
  Adb -Arguments @('shell', 'run-as', $package, 'cp', $tmpOnDevice, "$remoteDir/train_pilot.bin") `
    -TimeoutSeconds $cacheTransferTimeoutSeconds | Out-Null
  Adb @('shell', 'rm', '-f', $tmpOnDevice) | Out-Null
  $deviceCacheIdentity = (Adb -Arguments @('shell', 'run-as', $package, 'sha256sum', "$remoteDir/train_pilot.bin") `
    -TimeoutSeconds $cacheTransferTimeoutSeconds).Trim()
  if ($deviceCacheIdentity -notmatch ('^' + [regex]::Escape($cacheSha) + '\s')) {
    throw 'CACHE_DEVICE_IDENTITY_MISMATCH'
  }
  Write-Host "TRAIN_CACHE_STAGE mode=copy size_bytes=$cacheSizeBytes sha256=$cacheSha device_sha256=verified timeout_seconds=$cacheTransferTimeoutSeconds"
}
if ($Vocabulary -eq 1024) {
  $tmpTokenizer = "/data/local/tmp/phonelm-headless-$RunId-tokenizer"
  Write-Host 'TRAIN_STAGE_START name=tokenizer_push'
  Adb @('push', $tokenizerResolved, $tmpTokenizer) | Out-Null
  Write-Host 'TRAIN_STAGE_COMPLETE name=tokenizer_push'
  Write-Host 'TRAIN_STAGE_START name=tokenizer_copy'
  Adb @('shell', 'run-as', $package, 'cp', $tmpTokenizer, "$remoteDir/byte-bpe-v1024.model") | Out-Null
  Write-Host 'TRAIN_STAGE_COMPLETE name=tokenizer_copy'
  Write-Host 'TRAIN_STAGE_START name=tokenizer_temp_cleanup'
  Adb @('shell', 'rm', '-f', $tmpTokenizer) | Out-Null
  Write-Host 'TRAIN_STAGE_COMPLETE name=tokenizer_temp_cleanup'
  Copy-Item -LiteralPath $tokenizerResolved -Destination (Join-Path $reportRoot 'byte-bpe-v1024.model') -Force
}

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
# Revalidate both installed package identities immediately before launch. The
# same identity was checked after installation; this closes the staging window.
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package $package -LocalApk $apk
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package "$package.test" -LocalApk $testApk
$instrumentDir = Join-Path $reportRoot "instrumentation-$RunId"
if (Test-Path -LiteralPath $instrumentDir) { throw 'RUN_ID_REUSE: host instrumentation directory exists' }
[IO.Directory]::CreateDirectory($instrumentDir) | Out-Null
$instrument = $null
$checkpointProgress = [ordered]@{
  Count = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir).Count
  LastProgressUtc = [DateTime]::UtcNow
}
$script:midTelemetryCaptured = $false
try {
  $suite = if ($OneUpdateProbe) { 'nicopedia-dffn-probe' } else { 'nicopedia-long-training' }
  $instrumentSteps = if ($OneUpdateProbe) { 1 } else { $Steps }
  $instrument = Start-PhoneLmHeadlessInstrumentation -Adb $adb -Device $device -Package $package `
  -Class "$package.HeadlessDeviceTestRunner" -Suite $suite -RunId $RunId `
  -Arguments @{ seed = $Seed; vocabulary = $Vocabulary; layers = $Layers; heads = 2; tokens = $Tokens; dimension = $Dimension; feedForwardDimension = $FeedForwardDimension; attentionGate = $AttentionGate; learningRate = $LearningRate; learningRateSchedule = $LearningRateSchedule; decayStartStep = $DecayStartStep; decayEndStep = $DecayEndStep; scheduleTotalSteps = $ScheduleTotalSteps; targetLearningRate = $TargetLearningRate; experimentFork = $ExperimentFork.ToString().ToLowerInvariant(); parentLearningRate = $ParentLearningRate; optimizer = $Optimizer; muonBackend = $MuonBackend; muonLearningRate = $MuonLearningRate; muonMomentum = $MuonMomentum; muonNsSteps = $MuonNsSteps; muonNesterov = 'true'; steps = $instrumentSteps; batchSize = $BatchSize; resumeStep = $(if ($OneUpdateProbe) { 0 } else { $ResumeStep }); checkpointInterval = $CheckpointInterval; allowQualityFailure = $AllowQualityFailure.ToString().ToLowerInvariant(); cacheSourceRunId = $deviceCacheSourceRunId } `
  -StdoutPath (Join-Path $instrumentDir 'stdout.txt') -StderrPath (Join-Path $instrumentDir 'stderr.txt')
$waited = Wait-PhoneLmHeadlessStatus -Process $instrument -Adb $adb -Device $device -Package $package `
  -PollLimit $PollLimit -PollSeconds $PollSeconds -ProgressEverySeconds $ProgressEverySeconds -Label "training-step-$Steps" `
  -ExpectedRunId $RunId `
  -PartialPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-steps$Steps-partial-status.json") `
  -StatusProgressAction {
    param($elapsed, $status)
    if (-not $script:midTelemetryCaptured -and $auditTelemetryRoot -and $MidTelemetryAfterSeconds -gt 0 -and $elapsed -ge $MidTelemetryAfterSeconds) {
      $script:midTelemetryCaptured = $true
      $midPath = Join-Path $auditTelemetryRoot 'mid.json'
      try {
        & (Join-Path $PSScriptRoot 'capture_android_cpu_telemetry.ps1') -AdbPath $adb -Device $device -Package $package -Phase mid -OutputPath $midPath -RunId $RunId | Out-Null
        Write-Host 'audit_telemetry_mid=CAPTURED'
      } catch {
        [IO.Directory]::CreateDirectory($auditTelemetryRoot) | Out-Null
        [IO.File]::WriteAllText((Join-Path $auditTelemetryRoot 'mid-error.txt'), 'NOT_AVAILABLE', [Text.UTF8Encoding]::new($false))
        Write-Host 'audit_telemetry_mid=NOT_AVAILABLE'
      }
    }
    $phase = [regex]::Match($status, '"current_phase"\s*:\s*"([^"]*)"').Groups[1].Value
    $done = [regex]::Match($status, '"completed_tests"\s*:\s*(\d+)').Groups[1].Value
    $total = [regex]::Match($status, '"total_tests"\s*:\s*(\d+)').Groups[1].Value
    if ($OneUpdateProbe) {
      Write-Host "progress phase=probe elapsed_seconds=$elapsed status_phase=$phase completed=$done/$total"
    } else {
      # Fresh runs perform a potentially long CPU replay after the last
      # canonical checkpoint.  No checkpoint is expected during that phase,
      # so checkpoint-stall protection applies only while native training is
      # still active.  The outer heartbeat/poll timeout remains fail-closed.
      if ($phase -eq 'cpu_replay') {
        Write-Host "progress phase=training elapsed_seconds=$elapsed status_phase=$phase completed=$done/$total checkpoint_stall_ignored=true"
      } else {
        $ckpts = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir)
        Update-PhoneLmCheckpointProgress -State $checkpointProgress -CheckpointCount $ckpts.Count `
          -NowUtc ([DateTime]::UtcNow) -StallSeconds $CheckpointStallSeconds
        Write-Host "progress phase=training elapsed_seconds=$elapsed status_phase=$phase completed=$done/$total checkpoint_count=$($ckpts.Count)"
      }
    }
  } `
  -ConditionAction {
    param($elapsed)
    $state = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase "training-$elapsed-sec"
    $memoryLines = Adb @('shell', 'cat', '/proc/meminfo')
    $memoryText = [string]::Join("`n", [string[]]$memoryLines)
    $memoryAvailableKb = Get-PhoneLmMemAvailableKilobytes -MemInfoText $memoryText
    Write-Host "health elapsed_seconds=$elapsed thermal=$($state.thermal_status) battery_temp_c=$($state.battery_temperature_c) battery_voltage_mv=$($state.battery_voltage_mv) mem_available_kb=$memoryAvailableKb"
  } `
  -FocusAction {
    param($elapsed)
    if ((Get-PhoneLmTopPackage -Adb $adb -Device $device) -eq $package) { throw 'FOCUS_TAKEOVER_DETECTED' }
  }
$result = Get-PhoneLmHeadlessReport -StatusJson $waited.StatusJson -Adb $adb -Device $device -Package $package
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package $package -LocalApk $apk
Assert-PhoneLmInstalledApkMatches -Adb $adb -Device $device -Package "$package.test" -LocalApk $testApk
# Preserve the private probe report before lifecycle assertions fail the host
# wrapper.  This is needed to classify an activity/focus invariant violation
# while retaining the native graph/QNN evidence; it is never committed.
if ($OneUpdateProbe) {
  $result | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers$modelTag-steps$Steps-result.txt") -Encoding utf8
}
Assert-PhoneLmHeadlessNoActivity -Text $result
$deviceCompleted = $waited.StatusJson -match '"status"\s*:\s*"PASSED"' -and
  (($result -match '(?m)^status=SUCCESS\s*$') -or ($AllowQualityFailure -and $result -match '(?m)^status=FAILED\s*$'))
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
$reportMap = if ($OneUpdateProbe) {
  $probeMap = Get-PhoneLmKeyValueMap -Text $result
  foreach ($key in @('status', 'qnn_return_code_success', 'output_tensors_finite', 'cpu_fallback', 'nan_detected', 'inf_detected', 'graph_execute_count', 'api_trace_graph_execute_attempt_count', 'api_trace_graph_execute_success_count', 'api_trace_graph_execute_failure_count', 'api_trace_last_qnn_result', 'api_trace_effective_result', 'api_trace_cpu_backend_initialized', 'api_trace_fallback_attempted', 'api_trace_fallback_succeeded')) {
    if (-not $probeMap.Contains($key)) { throw "PROBE_REPORT_FIELD_MISSING: $key" }
  }
  if ($probeMap.status -ne 'SUCCESS' -or $probeMap.qnn_return_code_success -ne 'true' -or $probeMap.output_tensors_finite -ne 'true' -or $probeMap.cpu_fallback -ne 'false' -or $probeMap.nan_detected -ne 'false' -or $probeMap.inf_detected -ne 'false') { throw 'PROBE_REPORT_HEALTH_REJECTED' }
  if ($probeMap.api_trace_last_qnn_result -ne '0' -or $probeMap.api_trace_effective_result -ne '0' -or $probeMap.api_trace_cpu_backend_initialized -ne 'false' -or $probeMap.api_trace_fallback_attempted -ne 'false' -or $probeMap.api_trace_fallback_succeeded -ne 'false') { throw 'PROBE_REPORT_QNN_HEALTH_REJECTED' }
  foreach ($key in @('training_graph_prepared', 'adam_graph_prepared', 'graph_finalize_training_count', 'graph_finalize_adam_count', 'graph_execute_count', 'expected_fused_forward_backward_execute_count', 'fused_forward_backward_execute_count', 'expected_adam_execute_count', 'adam_execute_count')) {
    if (-not $probeMap.Contains($key)) { throw "PROBE_REPORT_FIELD_MISSING: $key" }
  }
  $fusedCount = [int]$probeMap.fused_forward_backward_execute_count
  $adamCount = [int]$probeMap.adam_execute_count
  if ($probeMap.training_graph_prepared -ne 'true' -or $probeMap.adam_graph_prepared -ne 'true' -or
      [int]$probeMap.graph_finalize_training_count -ne 1 -or [int]$probeMap.graph_finalize_adam_count -ne 1) { throw 'PROBE_REPORT_GRAPH_PREPARE_REJECTED' }
  if ($fusedCount -ne [int]$probeMap.expected_fused_forward_backward_execute_count -or
      $adamCount -ne [int]$probeMap.expected_adam_execute_count -or
      [int]$probeMap.graph_execute_count -ne ($fusedCount + $adamCount) -or
      [int]$probeMap.api_trace_graph_execute_attempt_count -ne [int]$probeMap.graph_execute_count -or
      [int]$probeMap.api_trace_graph_execute_success_count -ne [int]$probeMap.graph_execute_count -or
      [int]$probeMap.api_trace_graph_execute_failure_count -ne 0) { throw 'PROBE_REPORT_EXECUTE_COUNT_MISMATCH' }
  $probeMap
} elseif ($Optimizer -eq 'Muon') {
  # The Muon health branch is bound to the Nicopedia Muon pilot architecture
  # (V1024/D64/FFN128/L19/H2).  Its parameter counts are derived from the
  # generated SSOT metadata artifact so adding a parameter upstream only
  # requires regenerating the artifact.
  $muonDerived = Get-PhoneLmParameterMetadataDerivation `
      -Vocabulary 1024 -Dimension 64 -FeedForwardDimension 128 -Layers 19 -Heads 2 `
      -HeadwiseG1 ($AttentionGate -like 'headwise_g1_*')
  $hasGateIdentity = ($AttentionGate -ne 'none')
  $expectedCheckpointFormat = if ($hasGateIdentity) { 'NPRTCKPTV5' } else { 'NPRTCKPTV4' }
  $expectedAuxAdamParameters = $muonDerived.aux_adam_parameter_count
  $expectedMuonBackend = if ($MuonBackend -eq 'HVX') { 'HVX_W8' } else { 'CPU' }
  $muonMap = Get-PhoneLmKeyValueMap -Text $result
  foreach ($key in @('optimizer','qnn_return_code_success','output_tensors_finite','final_finite','all_steps_finite','cpu_fallback','fallback','checkpoint_format','completed_steps','muon_matrix_count','muon_parameter_count','aux_adam_parameter_count','forward_backward_backend','optimizer_muon_backend','optimizer_aux_adam_backend','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) {
    if (-not $muonMap.Contains($key)) { throw "MUON_REPORT_FIELD_MISSING: $key" }
  }
  if ($muonMap.optimizer -ne 'muon_aux_adam' -or $muonMap.qnn_return_code_success -ne 'true' -or
      $muonMap.output_tensors_finite -ne 'true' -or $muonMap.final_finite -ne 'true' -or $muonMap.all_steps_finite -ne 'true' -or
      $muonMap.cpu_fallback -ne 'false' -or $muonMap.fallback -ne 'false' -or $muonMap.checkpoint_format -ne $expectedCheckpointFormat -or
      [int]$muonMap.completed_steps -ne $Steps -or [int]$muonMap.muon_matrix_count -ne $muonDerived.muon_matrix_count -or
      [long]$muonMap.muon_parameter_count -ne $muonDerived.muon_parameter_count -or [long]$muonMap.aux_adam_parameter_count -ne $expectedAuxAdamParameters -or
      $muonMap.forward_backward_backend -ne 'HTP' -or $muonMap.optimizer_muon_backend -ne $expectedMuonBackend -or
      $muonMap.optimizer_aux_adam_backend -ne 'CPU' -or $muonMap.api_trace_graph_execute_failure_count -ne '0' -or
      $muonMap.api_trace_fallback_attempted -ne 'false' -or $muonMap.api_trace_fallback_succeeded -ne 'false') { throw 'MUON_REPORT_HEALTH_REJECTED' }

  if ($MuonBackend -eq 'HVX') {
    if (-not $muonMap.Contains('hvx_rpc_failure_count') -or
        -not $muonMap.Contains('hvx_fallback_count') -or
        -not $muonMap.Contains('hvx_nonfinite_count')) { throw 'MUON_HVX_REPORT_FIELDS_MISSING' }
    if ([int]$muonMap.hvx_rpc_failure_count -ne 0 -or
        [int]$muonMap.hvx_fallback_count -ne 0 -or
        [int]$muonMap.hvx_nonfinite_count -ne 0) { throw 'MUON_HVX_HEALTH_REJECTED' }
  }
  $muonMap
} elseif ($AllowQualityFailure) {
  $smokeMap = Get-PhoneLmKeyValueMap -Text $result
  foreach ($key in @('qnn_return_code_success', 'output_tensors_finite', 'cpu_fallback', 'nan_detected', 'inf_detected', 'completed_steps', 'api_trace_graph_execute_failure_count', 'api_trace_last_qnn_result', 'api_trace_effective_result', 'api_trace_cpu_backend_initialized', 'api_trace_fallback_attempted', 'api_trace_fallback_succeeded')) {
    if (-not $smokeMap.Contains($key)) { throw "SMOKE_REPORT_FIELD_MISSING: $key" }
  }
  if ($smokeMap.qnn_return_code_success -ne 'true' -or $smokeMap.output_tensors_finite -ne 'true' -or
      $smokeMap.cpu_fallback -ne 'false' -or $smokeMap.nan_detected -ne 'false' -or $smokeMap.inf_detected -ne 'false' -or
      $smokeMap.api_trace_graph_execute_failure_count -ne '0' -or $smokeMap.api_trace_last_qnn_result -ne '0' -or
      $smokeMap.api_trace_effective_result -ne '0' -or $smokeMap.api_trace_cpu_backend_initialized -ne 'false' -or
      $smokeMap.api_trace_fallback_attempted -ne 'false' -or $smokeMap.api_trace_fallback_succeeded -ne 'false' -or
      [int]$smokeMap.completed_steps -ne $Steps) { throw 'SMOKE_REPORT_HEALTH_REJECTED' }
  $smokeMap
} else {
  Assert-PhoneLmHealthReport -Text $result -ExpectedBuildId $ExpectedBuildId -ExpectedStep $Steps -Kind training
}
if (-not $reportMap.Contains('model_dimension') -or -not $reportMap.Contains('feed_forward_dimension') -or
    [int]$reportMap.model_dimension -ne $Dimension -or [int]$reportMap.feed_forward_dimension -ne $FeedForwardDimension) {
  throw 'TRAINING_REPORT_MODEL_IDENTITY_MISMATCH'
}
if ($null -ne $expandedTrainManifest) {
  $expectedRecordsSeen = [long]$Steps * $BatchSize
  $expectedTokensSeen = $expectedRecordsSeen * $Tokens
  $expectedOrderHash = [string]$expandedTrainManifest.training_order.hash
  foreach ($field in @('target_tokens_seen','records_seen','unique_records_seen','unique_articles_seen',
      'cache_traversal_fraction','completed_epochs_equivalent','target_original_utf8_bytes_seen',
      'training_order_hash','exposure_scope')) {
    if (-not $reportMap.Contains($field)) { throw "EXPOSURE_REPORT_FIELD_MISSING: $field" }
  }
  $expectedTraversal = $expectedRecordsSeen / [double]$expandedTrainManifest.cache.records
  if ([long]$reportMap.records_seen -ne $expectedRecordsSeen -or
      [long]$reportMap.target_tokens_seen -ne $expectedTokensSeen -or
      [long]$reportMap.unique_records_seen -le 0 -or
      [long]$reportMap.unique_records_seen -gt [long]$expandedTrainManifest.cache.records -or
      [long]$reportMap.unique_records_seen -gt $expectedRecordsSeen -or
      [long]$reportMap.unique_articles_seen -le 0 -or
      [long]$reportMap.unique_articles_seen -gt [long]$reportMap.unique_records_seen -or
      [math]::Abs(([double]$reportMap.cache_traversal_fraction) - $expectedTraversal) -gt 1.0e-9 -or
      [math]::Abs(([double]$reportMap.completed_epochs_equivalent) - $expectedTraversal) -gt 1.0e-9 -or
      [long]$reportMap.target_original_utf8_bytes_seen -le 0 -or
      $reportMap.training_order_hash -ne $expectedOrderHash -or
      $reportMap.exposure_scope -ne 'cumulative_from_fresh_initialization') {
    throw 'EXPOSURE_REPORT_ACCOUNTING_MISMATCH'
  }
}
if (-not $reportMap.Contains('learning_rate') -or [single]$reportMap.learning_rate -ne [single]$LearningRate) {
  throw 'TRAINING_REPORT_LEARNING_RATE_MISMATCH'
}
if (-not $reportMap.Contains('learning_rate_schedule') -or $reportMap.learning_rate_schedule -ne $LearningRateSchedule -or
    -not $reportMap.Contains('learning_rate_decay_start_step') -or [int]$reportMap.learning_rate_decay_start_step -ne $DecayStartStep -or
    -not $reportMap.Contains('learning_rate_decay_end_step') -or [int]$reportMap.learning_rate_decay_end_step -ne $DecayEndStep -or
    -not $reportMap.Contains('learning_rate_schedule_total_steps') -or [int]$reportMap.learning_rate_schedule_total_steps -ne $ScheduleTotalSteps -or
    -not $reportMap.Contains('learning_rate_target') -or [single]$reportMap.learning_rate_target -ne [single]$TargetLearningRate -or
    -not $reportMap.Contains('experiment_fork') -or ($reportMap.experiment_fork -ne $ExperimentFork.ToString().ToLowerInvariant()) -or
    -not $reportMap.Contains('parent_learning_rate') -or [single]$reportMap.parent_learning_rate -ne [single]$ParentLearningRate) {
  throw 'TRAINING_REPORT_SCHEDULE_MISMATCH'
}
$stateAfter = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'after'
$annotated = $result.TrimEnd() + "`n" +
  "cache_source_run_id=$deviceCacheSourceRunId`n" +
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
if ($OneUpdateProbe) {
  Write-Host "PASS NICOPEDIA_DFFN_PROBE seed=$Seed layers=$Layers dimension=$Dimension ffn=$FeedForwardDimension"
  Write-Host "Reports: $reportRoot"
  return
}
# Pull every interval NPRTCKPTV2 checkpoint and the loss curve back to
# build/reports. Both stay out of the public bundle and out of git.
$checkpointNames = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir)
$expectedSteps = @()
# Checkpoints are emitted on absolute step multiples.  A resumed segment may
# start between two multiples (for example 1000 -> 4000 with interval 320),
# so begin at the first multiple strictly above ResumeStep rather than adding
# an interval to the resume point.
$firstExpected = if ($ResumeStep -gt 0) {
  ([Math]::Floor($ResumeStep / [double]$CheckpointInterval) + 1) * $CheckpointInterval
} else { $CheckpointInterval }
for ($s = $firstExpected; $s -le $Steps; $s += $CheckpointInterval) { $expectedSteps += $s }
if ($expectedSteps -notcontains $Steps) { $expectedSteps += $Steps }
foreach ($expected in $expectedSteps) {
  $name = Get-PhoneLmCheckpointName -Seed $Seed -Layers $Layers -Tokens $Tokens -Dimension $Dimension -FeedForwardDimension $FeedForwardDimension -Step $expected
  if ($checkpointNames -notcontains $name) { throw "CHECKPOINT_INTERVAL_MISSING: $name" }
}
$lightEvalRows = @()
foreach ($name in $checkpointNames) {
  if ($name -notmatch "^htp-seed$Seed-l$Layers(-t$Tokens-d$Dimension-f$FeedForwardDimension)?-step(\d+)\.ckpt$") { continue }
  $stepName = [int]$Matches[2]
  if ($stepName -le $ResumeStep) { continue }
  $local = Join-Path $reportRoot $name
  $pulled = Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
    -RemotePath "$remoteDir/$name" -LocalPath $local -MinimumBytes 1024
  if ($Optimizer -eq 'Muon') {
    $magic = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($local), 0, 11)
    $expectedMagic = if ($AttentionGate -ne 'none') { "NPRTCKPTV5`n" } else { "NPRTCKPTV4`n" }
    if ($magic -ne $expectedMagic) { throw "CHECKPOINT_RESUME_FORMAT_INVALID: $name" }
  } else {
    $header = Get-PhoneLmCheckpointHeaders -Path $local
    if ($header.Step -ne $stepName -or $header.Seed -ne $Seed -or $header.Layers -ne $Layers -or $header.Heads -ne 2 -or $header.Vocabulary -ne $Vocabulary -or $header.Tokens -ne $Tokens -or $header.Dimension -ne $Dimension -or $header.FeedForward -ne $FeedForwardDimension) { throw "CHECKPOINT_IDENTITY_MISMATCH: $name" }
    $expectedCheckpointFormat = if ($Vocabulary -eq 1024) { 'NPRTCKPTV3' } else { 'NPRTCKPTV2' }
    if ($header.Magic -ne $expectedCheckpointFormat) { throw "CHECKPOINT_RESUME_FORMAT_INVALID: $name" }
    if ($Vocabulary -eq 1024) {
      $modelHash = 'sha256:' + (Get-FileHash -LiteralPath $tokenizerResolved -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($header.TokenizerKind -ne 'byte_bpe' -or $header.TokenizerHash -ne $modelHash) { throw "CHECKPOINT_TOKENIZER_IDENTITY_MISMATCH: $name" }
    }
  }
  # Run the same fixed one-record Val/Dev sample through the host evaluator at
  # every 1k checkpoint. Full-cap HTP evaluations use separately predeclared
  # milestone checkpoints. Header validation above remains fail-closed.
  $hostEvalExe = Join-Path $root 'build\host-tests\htp_checkpoint_eval.exe'
  $validationHost = Join-Path $evalCacheRoot 'validation.bin'
  $developmentHost = Join-Path $evalCacheRoot 'development.bin'
  Ensure-PhoneLmHostCheckpointEvaluator -Root $root -ExePath $hostEvalExe | Out-Null
  if (-not (Test-Path -LiteralPath $hostEvalExe -PathType Leaf) -or -not (Test-Path -LiteralPath $validationHost -PathType Leaf) -or -not (Test-Path -LiteralPath $developmentHost -PathType Leaf)) {
    throw 'HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE'
  }
  $hostDecoded = & $hostEvalExe $local $validationHost $developmentHost 1 1
  if ($LASTEXITCODE -ne 0) { throw "HOST_CHECKPOINT_EVALUATOR_DECODE_FAILED: $name" }
  $hostIdentity = Get-PhoneLmKeyValueMap -Text ($hostDecoded -join "`n")
  foreach ($field in @('seed', 'layers', 'step', 'parameter_hash', 'finite',
      'validation_nll', 'validation_top1', 'validation_chunks', 'validation_tokens',
      'validation_target_utf8_bytes', 'validation_bits_per_utf8_byte',
      'development_nll', 'development_top1', 'development_chunks',
      'development_tokens', 'development_target_utf8_bytes',
      'development_bits_per_utf8_byte')) {
    if (-not $hostIdentity.Contains($field)) { throw "HOST_CHECKPOINT_EVALUATOR_FIELD_MISSING: $field" }
  }
  if ([int]$hostIdentity.seed -ne $Seed -or [int]$hostIdentity.layers -ne $Layers -or [int]$hostIdentity.step -ne $stepName -or $hostIdentity.finite -ne 'true') { throw "HOST_CHECKPOINT_EVALUATOR_IDENTITY_MISMATCH: $name" }
  if ([int]$hostIdentity.validation_chunks -ne 1 -or [int]$hostIdentity.development_chunks -ne 1 -or
      [int]$hostIdentity.validation_tokens -ne $Tokens -or [int]$hostIdentity.development_tokens -ne $Tokens -or
      [long]$hostIdentity.validation_target_utf8_bytes -le 0 -or
      [long]$hostIdentity.development_target_utf8_bytes -le 0) {
    throw "HOST_CHECKPOINT_EVALUATOR_SAMPLE_MISMATCH: $name"
  }
  $lightEvalRows += [pscustomobject]@{
    step = $stepName
    evaluator_backend = 'host_cpu'
    validation_chunks = [int]$hostIdentity.validation_chunks
    validation_tokens = [int]$hostIdentity.validation_tokens
    validation_target_utf8_bytes = [long]$hostIdentity.validation_target_utf8_bytes
    validation_nll = [string]$hostIdentity.validation_nll
    validation_bpb = [string]$hostIdentity.validation_bits_per_utf8_byte
    validation_top1 = [string]$hostIdentity.validation_top1
    development_chunks = [int]$hostIdentity.development_chunks
    development_tokens = [int]$hostIdentity.development_tokens
    development_target_utf8_bytes = [long]$hostIdentity.development_target_utf8_bytes
    development_nll = [string]$hostIdentity.development_nll
    development_bpb = [string]$hostIdentity.development_bits_per_utf8_byte
    development_top1 = [string]$hostIdentity.development_top1
    parameter_hash = [string]$hostIdentity.parameter_hash
    finite = [string]$hostIdentity.finite
  }
  Write-Host "checkpoint step=$stepName size=$($pulled.Size) sha256=$($pulled.Sha256) identity=verified"
}
if ($Optimizer -eq 'Muon') {
  if ($lightEvalRows.Count -ne $expectedSteps.Count) { throw 'LIGHTWEIGHT_HELDOUT_EVAL_BOUNDARY_COUNT_MISMATCH' }
  $lightEvalPath = Join-Path $reportRoot "lightweight-heldout-evaluation-$RunId.csv"
  $lightEvalRows | Sort-Object -Property step | Export-Csv -LiteralPath $lightEvalPath -NoTypeInformation -Encoding utf8
  Write-Host "lightweight_heldout_evaluation=$lightEvalPath records=$($lightEvalRows.Count) sample=first_cache_record_per_split"
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
$telemetryLocal = Join-Path $reportRoot 'learning-rate-telemetry.csv'
# A smoke and its later final continuation share a report directory, while
# each native invocation emits a segment-local telemetry file. Preserve a
# prior segment instead of letting the generic binary receiver reject the
# legitimate new target as a stale mismatch.
if (Test-Path -LiteralPath $telemetryLocal -PathType Leaf) {
  $priorTelemetryRows = @(Import-Csv -LiteralPath $telemetryLocal -ErrorAction SilentlyContinue)
  $expectedTelemetryRows = if ($reportMap.Contains('run_completed_steps')) { [int]$reportMap.run_completed_steps } else { -1 }
  $firstTelemetryStep = if ($priorTelemetryRows.Count -gt 0) { [int]$priorTelemetryRows[0].step } else { -1 }
  if ($priorTelemetryRows.Count -ne $expectedTelemetryRows -or $firstTelemetryStep -ne ($ResumeStep + 1)) {
    $priorPath = Join-Path $reportRoot ("learning-rate-telemetry-prior-$ResumeStep.csv")
    if (Test-Path -LiteralPath $priorPath -PathType Leaf) {
      $priorPath = Join-Path $reportRoot ("learning-rate-telemetry-prior-$ResumeStep-" + [guid]::NewGuid().ToString('N') + '.csv')
    }
    Move-Item -LiteralPath $telemetryLocal -Destination $priorPath
  }
}
Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
  -RemotePath "$remoteDir/learning-rate-telemetry.csv" -LocalPath $telemetryLocal -MinimumBytes 1 | Out-Null
$telemetryRows = @(Import-Csv -LiteralPath $telemetryLocal)
if (-not $reportMap.Contains('run_completed_steps') -or $telemetryRows.Count -ne [int]$reportMap.run_completed_steps) { throw 'LEARNING_RATE_TELEMETRY_COUNT_MISMATCH' }
foreach ($row in $telemetryRows) {
  $telemetryStep = [int]$row.step
  if ($telemetryStep -lt ($ResumeStep + 1) -or $telemetryStep -gt $Steps) { throw 'LEARNING_RATE_TELEMETRY_STEP_RANGE_MISMATCH' }
  if (-not $row.PSObject.Properties['learning_rate_schedule'] -or $row.learning_rate_schedule -ne $LearningRateSchedule) { throw "LEARNING_RATE_TELEMETRY_SCHEDULE_MISMATCH: step=$telemetryStep" }
  $expectedLr = Get-PhoneLmExpectedLearningRate -Schedule $LearningRateSchedule -PeakLearningRate $learningRateValue -TargetLearningRate $targetLearningRateValue -DecayStartStep $DecayStartStep -DecayEndStep $DecayEndStep -Step $telemetryStep
  if ([math]::Abs(([double]$row.scheduled_lr) - $expectedLr) -gt 2.0e-8) { throw "LEARNING_RATE_TELEMETRY_MISMATCH: step=$telemetryStep" }
}
$requiredTelemetrySteps = if ($LearningRateSchedule -eq 'sqrt_decay') {
  @(6000,6250,6500,6750,7000,7250,7500,7750,8000)
} else {
  @(4000,4500,5000,5500,6000,6500,7000,7500,8000)
}
$requiredTelemetrySteps = @($requiredTelemetrySteps | Where-Object { $_ -gt $ResumeStep -and $_ -le $Steps })
foreach ($requiredStep in $requiredTelemetrySteps) {
  if (@($telemetryRows | Where-Object { [int]$_.step -eq $requiredStep }).Count -ne 1) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISSING: $requiredStep" }
}
$expectedTelemetryBoundaries = @()
$nextTelemetryBoundary = if ($ResumeStep -gt 0) {
  ([Math]::Floor($ResumeStep / [double]$CheckpointInterval) + 1) * $CheckpointInterval
} else { $CheckpointInterval }
for ($s = $nextTelemetryBoundary; $s -le $Steps; $s += $CheckpointInterval) { $expectedTelemetryBoundaries += $s }
if ($expectedTelemetryBoundaries -notcontains $Steps) { $expectedTelemetryBoundaries += $Steps }

$optimizerHealthRows = @()
if ($Optimizer -eq 'Muon') {
  $optimizerHealthLocal = Join-Path $reportRoot "optimizer-health-telemetry-$RunId.csv"
  Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
    -RemotePath "$remoteDir/optimizer-health-telemetry.csv" -LocalPath $optimizerHealthLocal -MinimumBytes 1 | Out-Null
  $optimizerHealthRows = @(Import-Csv -LiteralPath $optimizerHealthLocal)
  if ($optimizerHealthRows.Count -ne $expectedTelemetryBoundaries.Count) { throw 'OPTIMIZER_HEALTH_TELEMETRY_BOUNDARY_COUNT_MISMATCH' }
  $previousHealthBoundary = $ResumeStep
  for ($i = 0; $i -lt $optimizerHealthRows.Count; $i++) {
    $row = $optimizerHealthRows[$i]
    $expectedBoundary = [int]$expectedTelemetryBoundaries[$i]
    if ([int]$row.step -ne $expectedBoundary -or
        [int]$row.window_start_step -ne ($previousHealthBoundary + 1) -or
        [int]$row.window_updates -ne ($expectedBoundary - $previousHealthBoundary)) {
      throw "OPTIMIZER_HEALTH_TELEMETRY_POSITION_MISMATCH: step=$($row.step)"
    }
    foreach ($field in @('gradient_finite','momentum_finite','normalized_finite','ns_output_finite','update_finite','parameters_finite','qnn_return_code_success','hvx_output_finite')) {
      if ($row.$field -ne 'true') { throw "OPTIMIZER_HEALTH_TELEMETRY_FAILURE: step=$($row.step) field=$field" }
    }
    if ($row.nonfinite_detected -ne 'false') { throw "OPTIMIZER_HEALTH_TELEMETRY_NONFINITE_DETECTED: step=$($row.step)" }
    if ($row.gradient_clipping_enabled -ne 'false' -or [int]$row.clipped_steps_window -ne 0 -or
        [int]$row.hvx_rpc_status -ne 0 -or $row.hvx_fallback -ne 'false' -or
        $row.cpu_fallback -ne 'false') { throw "OPTIMIZER_HEALTH_TELEMETRY_POLICY_MISMATCH: step=$($row.step)" }
    foreach ($field in @('gradient_l2_norm_at_boundary','parameter_l2_norm_at_boundary','muon_parameter_delta_l2_last_update','aux_adam_parameter_delta_l2_last_update','muon_update_ms_window','aux_adam_update_ms_window','optimizer_update_ms_window')) {
      $number = [double]$row.$field
      if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt 0) {
        throw "OPTIMIZER_HEALTH_TELEMETRY_NONFINITE: step=$($row.step) field=$field"
      }
    }
    $previousHealthBoundary = $expectedBoundary
  }
}

$gateTelemetryRows = @()
if ($AttentionGate -in @('headwise_g1_sigmoid','headwise_g1_scale2_identity')) {
  $gateTelemetryLocal = Join-Path $reportRoot "gate-telemetry-$RunId.csv"
  Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
    -RemotePath "$remoteDir/gate-telemetry.csv" -LocalPath $gateTelemetryLocal -MinimumBytes 1 | Out-Null
  $gateTelemetryRows = @(Import-Csv -LiteralPath $gateTelemetryLocal)
  $expectedGateRows = $expectedTelemetryBoundaries.Count * $Layers * 2
  if ($gateTelemetryRows.Count -ne $expectedGateRows) { throw 'GATE_TELEMETRY_ROW_COUNT_MISMATCH' }
  $previousGateBoundary = $ResumeStep
  foreach ($boundary in $expectedTelemetryBoundaries) {
    $windowRows = @($gateTelemetryRows | Where-Object { [int]$_.step -eq [int]$boundary })
    if ($windowRows.Count -ne ($Layers * 2)) { throw "GATE_TELEMETRY_LAYER_HEAD_COUNT_MISMATCH: step=$boundary" }
    $windowUpdates = [int]$boundary - $previousGateBoundary
    $seenLayerHeads = @{}
    foreach ($row in $windowRows) {
      $layer = [int]$row.layer
      $head = [int]$row.head
      $key = "$layer`:$head"
      if ($layer -lt 0 -or $layer -ge $Layers -or $head -notin @(0,1) -or $seenLayerHeads.ContainsKey($key)) {
        throw "GATE_TELEMETRY_DUPLICATE_OR_INVALID_LAYER_HEAD: step=$boundary key=$key"
      }
      $seenLayerHeads[$key] = $true
      if ([int]$row.window_start_step -ne ($previousGateBoundary + 1) -or
          [int]$row.window_end_step -ne [int]$boundary -or
          [int]$row.count -ne ($windowUpdates * 8 * $Tokens)) {
        throw "GATE_TELEMETRY_POSITION_OR_COUNT_MISMATCH: step=$boundary key=$key"
      }
      foreach ($field in @('mean','stddev','min','max','below_0_1_fraction','above_0_9_fraction')) {
        $number = [double]$row.$field
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt 0) {
          throw "GATE_TELEMETRY_NONFINITE: step=$boundary key=$key field=$field"
        }
      }
      if ([double]$row.min -gt [double]$row.max -or [double]$row.min -lt 0 -or
          [double]$row.max -gt 1 -or [double]$row.mean -gt 1 -or
          [double]$row.below_0_1_fraction -gt 1 -or
          [double]$row.above_0_9_fraction -gt 1) {
        throw "GATE_TELEMETRY_RANGE_MISMATCH: step=$boundary key=$key"
      }
    }
    $previousGateBoundary = [int]$boundary
  }
}
Write-Host "Pulled $($checkpointNames.Count) interval checkpoints + $curveLocal + learning-rate-telemetry.csv + $($optimizerHealthRows.Count) optimizer windows + $($gateTelemetryRows.Count) gate summaries"
Write-Host "PASS NICOPEDIA_HTP seed=$Seed layers=$Layers steps=$Steps"
Write-Host "Reports: $reportRoot"
} finally {
  [void](Stop-PhoneLmCompletedHeadlessProcesses -Adb $adb -Device $device -Package $package -ExpectedRunId $RunId)
  Stop-PhoneLmOwnedInstrumentation -Process $instrument
}
