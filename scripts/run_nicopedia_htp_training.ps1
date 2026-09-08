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
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$MuonLearningRate = '0.010',
  [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$MuonMomentum = '0.95',
  [ValidateRange(1,99)][int]$MuonNsSteps = 5,
  [string]$CachePath = "",
  [string]$TokenizerModelPath = "",
  [string]$ReportRoot = "",
  [int]$PollLimit = 7200,
  [int]$PollSeconds = 2,
  [int]$ProgressEverySeconds = 30,
  [int]$CheckpointStallSeconds = 300,
  [int]$ResumeStep = 0,
  [int]$CheckpointInterval = 250,
  [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff'),
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

if ($SelfTest) {
  if ($BatchSize -ne 8) { throw "SELFTEST_BATCH_SIZE_DEFAULT: expected=8 actual=$BatchSize" }
  if ($Layers -ne 19) { throw "SELFTEST_LAYERS_DEFAULT: expected=19 actual=$Layers" }
  if ($Tokens -ne 32) { throw "SELFTEST_TOKENS_DEFAULT: expected=32 actual=$Tokens" }
  if ($Vocabulary -ne 256) { throw "SELFTEST_VOCABULARY_DEFAULT: expected=256 actual=$Vocabulary" }
  if ($Dimension -ne 32 -or $FeedForwardDimension -ne 32) {
      throw "SELFTEST_MODEL_DIMENSIONS_DEFAULT: expected=D32/FFN32 actual=D$Dimension/FFN$FeedForwardDimension"
  }
  if ($LearningRate -ne '0.003') { throw "SELFTEST_LEARNING_RATE_DEFAULT: expected=0.003 actual=$LearningRate" }
  if ($LearningRateSchedule -ne 'constant' -or $DecayStartStep -ne 0 -or $DecayEndStep -ne 0 -or $ScheduleTotalSteps -ne 0 -or $TargetLearningRate -ne '' -or $ExperimentFork -or $ParentLearningRate -ne '0') { throw 'SELFTEST_SCHEDULE_DEFAULT' }
  if ($Optimizer -ne 'Adam' -or $MuonLearningRate -ne '0.010' -or $MuonMomentum -ne '0.95' -or $MuonNsSteps -ne 5) { throw 'SELFTEST_OPTIMIZER_DEFAULT' }
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
if ($Steps -lt 1 -or $Steps -gt 12000) { throw 'NICOPEDIA_L19_HARD_CEILING: Steps must be in 1..12000' }
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
  $allowedLinearTargetLearningRates = @('0.0015', '0.0010', '0.0007', '0.0004', '0.0002', '0.0001', '0.0000')
  if ($learningRateValue -ne 0.0022 -or
      -not ($allowedLinearTargetLearningRates -contains $TargetLearningRate) -or
      $DecayStartStep -le 0 -or $DecayStartStep -ge $DecayEndStep -or
      $DecayEndStep -gt $ScheduleTotalSteps -or -not $ExperimentFork -or
      $parentLearningRateValue -ne 0.0022) { throw 'NICOPEDIA_LINEAR_SCHEDULE_INVALID' }
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
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$testApk = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
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

# The private token cache lives under build/private-data and is never
# committed.  The host pushes only the minimal pilot input the device needs.
$trainingDataRoot = if ($Vocabulary -eq 1024) { 'build\private-data\nicopedia-real-text-bpe-v1024' } elseif ($Tokens -eq 32) { 'build\private-data\nicopedia-real-text' } else { 'build\private-data\nicopedia-real-text-t64' }
if (-not $CachePath) { $CachePath = Join-Path $root (Join-Path $trainingDataRoot 'caches\train_pilot.bin') }
if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { throw "PRIVATE_CACHE_MISSING: $CachePath" }
$cacheResolved = [IO.Path]::GetFullPath($CachePath)
$allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $cacheResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CachePath must resolve below the repository build directory"
}
$tokenizerResolved = ''
if ($Vocabulary -eq 1024) {
  if (-not $TokenizerModelPath) { $TokenizerModelPath = Join-Path $root (Join-Path $trainingDataRoot 'tokenizer\byte-bpe-v1024.model') }
  if (-not (Test-Path -LiteralPath $TokenizerModelPath -PathType Leaf)) { throw "PRIVATE_TOKENIZER_MODEL_MISSING: $TokenizerModelPath" }
  $tokenizerResolved = [IO.Path]::GetFullPath($TokenizerModelPath)
  if (-not $tokenizerResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'TokenizerModelPath must resolve below the repository build directory' }
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
  # The verified QNN-enabled app APK is large (~190 MB) and may legitimately
  # exceed the ordinary command timeout over a TCP ADB transport.  Keep the
  # normal short timeout for health/control operations, but give installation
  # a bounded one-shot window so a transport timeout is not mistaken for a
  # trial numerical failure.
  Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @('install', '-r', $apk) -TimeoutSeconds 300 | Out-Null
  Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments @('install', '-r', '-t', $testApk) -TimeoutSeconds 300 | Out-Null
  # `adb install -r` can restore a retained task and start the app process.
  # Re-establish the headless baseline before instrumentation; this does not
  # clear app data or weaken the later activity/focus invariant.
  Adb @('shell', 'am', 'force-stop', $package) | Out-Null
  Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $package
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
$tmpOnDevice = "/data/local/tmp/phonelm-headless-$RunId-train"
Adb @('push', $cacheResolved, $tmpOnDevice) | Out-Null
Adb @('shell', 'run-as', $package, 'cp', $tmpOnDevice, "$remoteDir/train_pilot.bin") | Out-Null
Adb @('shell', 'rm', '-f', $tmpOnDevice) | Out-Null
if ($Vocabulary -eq 1024) {
  $tmpTokenizer = "/data/local/tmp/phonelm-headless-$RunId-tokenizer"
  Adb @('push', $tokenizerResolved, $tmpTokenizer) | Out-Null
  Adb @('shell', 'run-as', $package, 'cp', $tmpTokenizer, "$remoteDir/byte-bpe-v1024.model") | Out-Null
  Adb @('shell', 'rm', '-f', $tmpTokenizer) | Out-Null
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
$instrumentDir = Join-Path $reportRoot "instrumentation-$RunId"
if (Test-Path -LiteralPath $instrumentDir) { throw 'RUN_ID_REUSE: host instrumentation directory exists' }
[IO.Directory]::CreateDirectory($instrumentDir) | Out-Null
$instrument = $null
$checkpointProgress = [ordered]@{
  Count = @(Get-PhoneLmCheckpointNames -Adb $adb -Device $device -Package $package -RemoteDir $remoteDir).Count
  LastProgressUtc = [DateTime]::UtcNow
}
try {
  $suite = if ($OneUpdateProbe) { 'nicopedia-dffn-probe' } else { 'nicopedia-long-training' }
  $instrumentSteps = if ($OneUpdateProbe) { 1 } else { $Steps }
  $instrument = Start-PhoneLmHeadlessInstrumentation -Adb $adb -Device $device -Package $package `
  -Class "$package.HeadlessDeviceTestRunner" -Suite $suite -RunId $RunId `
  -Arguments @{ seed = $Seed; vocabulary = $Vocabulary; layers = $Layers; heads = 2; tokens = $Tokens; dimension = $Dimension; feedForwardDimension = $FeedForwardDimension; learningRate = $LearningRate; learningRateSchedule = $LearningRateSchedule; decayStartStep = $DecayStartStep; decayEndStep = $DecayEndStep; scheduleTotalSteps = $ScheduleTotalSteps; targetLearningRate = $TargetLearningRate; experimentFork = $ExperimentFork.ToString().ToLowerInvariant(); parentLearningRate = $ParentLearningRate; optimizer = $Optimizer; muonLearningRate = $MuonLearningRate; muonMomentum = $MuonMomentum; muonNsSteps = $MuonNsSteps; muonNesterov = 'true'; steps = $instrumentSteps; batchSize = $BatchSize; resumeStep = $(if ($OneUpdateProbe) { 0 } else { $ResumeStep }); checkpointInterval = $CheckpointInterval; allowQualityFailure = $AllowQualityFailure.ToString().ToLowerInvariant() } `
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
    Write-Host "health elapsed_seconds=$elapsed thermal=$($state.thermal_status) battery_temp_c=$($state.battery_temperature_c) battery_voltage_mv=$($state.battery_voltage_mv)"
  } `
  -FocusAction {
    param($elapsed)
    if ((Get-PhoneLmTopPackage -Adb $adb -Device $device) -eq $package) { throw 'FOCUS_TAKEOVER_DETECTED' }
  }
$result = Get-PhoneLmHeadlessReport -StatusJson $waited.StatusJson -Adb $adb -Device $device -Package $package
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
  $muonMap = Get-PhoneLmKeyValueMap -Text $result
  foreach ($key in @('optimizer','qnn_return_code_success','output_tensors_finite','final_finite','all_steps_finite','cpu_fallback','fallback','checkpoint_format','completed_steps','muon_matrix_count','muon_parameter_count','aux_adam_parameter_count','forward_backward_backend','optimizer_muon_backend','optimizer_aux_adam_backend','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) {
    if (-not $muonMap.Contains($key)) { throw "MUON_REPORT_FIELD_MISSING: $key" }
  }
  if ($muonMap.optimizer -ne 'muon_aux_adam' -or $muonMap.qnn_return_code_success -ne 'true' -or
      $muonMap.output_tensors_finite -ne 'true' -or $muonMap.final_finite -ne 'true' -or $muonMap.all_steps_finite -ne 'true' -or
      $muonMap.cpu_fallback -ne 'false' -or $muonMap.fallback -ne 'false' -or $muonMap.checkpoint_format -ne 'NPRTCKPTV4' -or
      [int]$muonMap.completed_steps -ne $Steps -or [int]$muonMap.muon_matrix_count -ne 114 -or
      [long]$muonMap.muon_parameter_count -ne 622592 -or [long]$muonMap.aux_adam_parameter_count -ne 135936 -or
      $muonMap.forward_backward_backend -ne 'HTP' -or $muonMap.optimizer_muon_backend -ne 'CPU' -or
      $muonMap.optimizer_aux_adam_backend -ne 'CPU' -or $muonMap.api_trace_graph_execute_failure_count -ne '0' -or
      $muonMap.api_trace_fallback_attempted -ne 'false' -or $muonMap.api_trace_fallback_succeeded -ne 'false') { throw 'MUON_REPORT_HEALTH_REJECTED' }
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
foreach ($name in $checkpointNames) {
  if ($name -notmatch "^htp-seed$Seed-l$Layers(-t$Tokens-d$Dimension-f$FeedForwardDimension)?-step(\d+)\.ckpt$") { continue }
  $stepName = [int]$Matches[2]
  $local = Join-Path $reportRoot $name
  $pulled = Receive-PhoneLmBinary -Adb $adb -Device $device -Package $package `
    -RemotePath "$remoteDir/$name" -LocalPath $local -MinimumBytes 1024
  if ($Optimizer -eq 'Muon') {
    $magic = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($local), 0, 11)
    if ($magic -ne "NPRTCKPTV4`n") { throw "CHECKPOINT_RESUME_FORMAT_INVALID: $name" }
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
  # The host evaluator intentionally reports the compact identity fields
  # (seed/layers/step); the full V2 architecture identity was already
  # fail-closed against the checkpoint header above.  Do not require fields
  # that older evaluator binaries do not emit here.
  foreach ($field in @('seed', 'layers', 'step', 'parameter_hash', 'finite')) { if (-not $hostIdentity.Contains($field)) { throw "HOST_CHECKPOINT_EVALUATOR_FIELD_MISSING: $field" } }
  if ([int]$hostIdentity.seed -ne $Seed -or [int]$hostIdentity.layers -ne $Layers -or [int]$hostIdentity.step -ne $stepName -or $hostIdentity.finite -ne 'true') { throw "HOST_CHECKPOINT_EVALUATOR_IDENTITY_MISMATCH: $name" }
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
Write-Host "Pulled $($checkpointNames.Count) interval checkpoints + $curveLocal + learning-rate-telemetry.csv"
Write-Host "PASS NICOPEDIA_HTP seed=$Seed layers=$Layers steps=$Steps"
Write-Host "Reports: $reportRoot"
} finally {
  [void](Stop-PhoneLmCompletedHeadlessProcesses -Adb $adb -Device $device -Package $package -ExpectedRunId $RunId)
  Stop-PhoneLmOwnedInstrumentation -Process $instrument
}
