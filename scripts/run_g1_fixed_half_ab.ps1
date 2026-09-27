# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Factor-isolation A/B: Current G1 (learned sigmoid gate) vs
# Fixed 0.5 branch scale (Yh = 0.5 * Ah, no Wg).
# Fixed 1.5x LR, seed1, 500 steps.
# Modes: Plan | Smoke | Run | Ab | Analyze
[CmdletBinding()]
param(
  [ValidateSet('Current','FixedHalf')][string]$Arm = 'Current',
  [ValidateSet('Plan','Smoke','Run','Ab','Analyze')][string]$Mode = 'Plan',
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$ReportRoot = 'build/g1-fixed-half-500-2026-09',
  [string]$ResultsRoot = 'docs/results/g1-fixed-half-500-2026-09',
  [string]$DeviceLockRoot = 'D:\ghq\github.com\yuubinnkyoku\.hexatrain-device-lock',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [int]$Steps = 500,
  [int]$CheckpointInterval = 100,
  [int]$SmokeSteps = 8,
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
$training = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$evaluation = Join-Path $PSScriptRoot 'run_nicopedia_htp_eval.ps1'
$analysisPy = Join-Path $PSScriptRoot 'g1_fixed_half_analyze.py'

$evalSteps = @(100, 200, 300, 400, 500)
$lr = [ordered]@{
  aux_adam_s = '0.0033'
  muon_s = '0.0075'
  target_s = '0.00015'
}

$arms = [ordered]@{
  Current = [ordered]@{
    attention_gate = 'headwise_g1_sigmoid'
    parameter_count = 760960
    checkpoint_format = 'NPRTCKPTV5'
  }
  FixedHalf = [ordered]@{
    attention_gate = 'fixed_half'
    parameter_count = 758528
    checkpoint_format = 'NPRTCKPTV5'
  }
}

function Get-FixedArmDirectory([string]$ArmName) {
  return Join-Path $ReportRoot $ArmName.ToLowerInvariant()
}

function Get-FixedResultsArmDirectory([string]$ArmName) {
  return Join-Path $ResultsRoot $ArmName.ToLowerInvariant()
}

function Write-FixedJson([string]$Path, $Payload) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  $Payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Test-FixedArmComplete([string]$ArmName) {
  $dir = Get-FixedArmDirectory $ArmName
  $result = Join-Path $dir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $Steps)
  if (-not (Test-Path -LiteralPath $result -PathType Leaf)) { return $false }
  $map = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $result -Raw)
  if (-not $map.Contains('status') -or $map.status -ne 'SUCCESS') { return $false }
  if (-not $map.Contains('completed_steps') -or [int]$map.completed_steps -ne $Steps) { return $false }
  foreach ($step in $evalSteps) {
    $ck = Join-Path $dir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
    if (-not (Test-Path -LiteralPath $ck -PathType Leaf)) { return $false }
  }
  return $true
}

function Get-FixedHardStopFlags([string]$ResultPath) {
  $map = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $ResultPath -Raw)
  $flags = @()
  if ($map.Contains('status') -and $map.status -ne 'SUCCESS') { $flags += 'status=' + $map.status }
  foreach ($pair in @(
      @('all_steps_finite','true'), @('final_finite','true'),
      @('output_tensors_finite','true'), @('qnn_return_code_success','true'),
      @('cpu_fallback','false'), @('fallback','false'),
      @('nan_detected','false'), @('inf_detected','false'))) {
    if (-not $map.Contains($pair[0]) -or $map[$pair[0]] -ne $pair[1]) {
      $flags += ('{0}={1}' -f $pair[0], $(if ($map.Contains($pair[0])) { $map[$pair[0]] } else { '<missing>' }))
    }
  }
  foreach ($pair in @(
      @('api_trace_graph_execute_failure_count','0'),
      @('hvx_rpc_failure_count','0'),
      @('hvx_fallback_count','0'),
      @('hvx_nonfinite_count','0'))) {
    if ($map.Contains($pair[0]) -and [int]$map[$pair[0]] -ne 0) {
      $flags += ('{0}={1}' -f $pair[0], $map[$pair[0]])
    }
  }
  return ,$flags
}

function Get-FixedLockOwnerPath { return Join-Path $DeviceLockRoot 'owner.txt' }
function Test-FixedDeviceLock { return (Test-Path -LiteralPath $DeviceLockRoot -PathType Container) }

function Acquire-FixedDeviceLock {
  if (Test-FixedDeviceLock) {
    $owner = if (Test-Path -LiteralPath (Get-FixedLockOwnerPath)) {
      (Get-Content -LiteralPath (Get-FixedLockOwnerPath) -Raw) } else { '' }
    throw ("DEVICE_LOCK_HELD path=$DeviceLockRoot owner=$($owner -replace '\n',' | ')")
  }
  New-Item -ItemType Directory -Force -Path $DeviceLockRoot | Out-Null
  Set-Content -LiteralPath (Get-FixedLockOwnerPath) -Value @(
    "owner=g1-fixed-half-agent", "pid=$PID",
    "timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))",
    "repo=$root", "experiment=g1-fixed-half-500-2026-09",
    "branch=$(git -C $root branch --show-current)"
  ) -Encoding utf8
  Write-Host "device_lock_acquired=$DeviceLockRoot"
}

function Release-FixedDeviceLock {
  if (-not (Test-FixedDeviceLock)) { return }
  $ownerPath = Get-FixedLockOwnerPath
  if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    $text = Get-Content -LiteralPath $ownerPath -Raw
    if ($text -notmatch 'g1-fixed-half-agent' -and $text -notmatch "pid=$PID") {
      throw "DEVICE_LOCK_NOT_OURS: refusing to release $DeviceLockRoot"
    }
  }
  Remove-Item -LiteralPath $DeviceLockRoot -Recurse -Force
  Write-Host "device_lock_released=$DeviceLockRoot"
}

function Enable-FixedDeviceAwake {
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  $dev = (Resolve-PhoneLmDevice -Adb $adb).Endpoint
  & $adb -s $dev shell input keyevent KEYCODE_WAKEUP | Out-Null
  & $adb -s $dev shell wm dismiss-keyguard | Out-Null
  & $adb -s $dev shell svc power stayon true | Out-Null
  & $adb -s $dev shell dumpsys deviceidle disable | Out-Null
  Write-Host "device_awake_and_idle_disabled=true endpoint=$dev"
}

function Invoke-FixedArm {
  param(
    [Parameter(Mandatory=$true)][ValidateSet('Current','FixedHalf')][string]$ArmName,
    [ValidateSet('Smoke','Run')][string]$ArmMode = 'Run'
  )
  $spec = $arms[$ArmName]
  $gate = $spec.attention_gate
  $armDir = Get-FixedArmDirectory $ArmName
  $resultDir = Get-FixedResultsArmDirectory $ArmName
  New-Item -ItemType Directory -Force -Path $armDir | Out-Null
  New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

  $armSteps = if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $Steps }
  $armInterval = if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $CheckpointInterval }
  $armEvalSteps = if ($ArmMode -eq 'Smoke') { @() } else { $evalSteps }

  $existingResult = Join-Path $armDir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $armSteps)
  if ($ArmMode -eq 'Run' -and (Test-FixedArmComplete $ArmName)) {
    Write-Host "arm_already_complete arm=$ArmName dir=$armDir"
    return [pscustomobject]@{ arm = $ArmName; status = 'REUSED'; dir = $armDir }
  }

  Write-Host "=== FIXED HALF AB ARM arm=$ArmName gate=$gate mode=$ArmMode steps=$armSteps ==="
  Write-Host ("resolved muon_lr={0} aux_adam_lr={1} target_lr={2}" -f $lr.muon_s, $lr.aux_adam_s, $lr.target_s)

  $trainParams = @{
    QairtSdkRoot = $QairtSdkRoot
    ExpectedBuildId = $ExpectedBuildId
    Seed = 1
    Layers = 19
    Steps = $armSteps
    Tokens = 32
    Vocabulary = 1024
    Dimension = 64
    FeedForwardDimension = 128
    BatchSize = 8
    LearningRate = $lr.aux_adam_s
    LearningRateSchedule = 'linear_decay'
    DecayStartStep = 4000
    DecayEndStep = 8000
    ScheduleTotalSteps = 8000
    TargetLearningRate = $lr.target_s
    ExperimentFork = $true
    ParentLearningRate = $lr.aux_adam_s
    Optimizer = 'Muon'
    MuonBackend = 'HVX'
    AttentionGate = $gate
    MuonLearningRate = $lr.muon_s
    MuonMomentum = '0.95'
    MuonNsSteps = 5
    HexagonSdkRoot = $HexagonSdkRoot
    CachePath = $CachePath
    TokenizerModelPath = $TokenizerModelPath
    ReportRoot = $armDir
    CheckpointInterval = $armInterval
    CheckpointStallSeconds = 1800
    SkipBuild = [bool]$SkipBuild
    SkipInstall = [bool]$SkipInstall
  }
  & $training @trainParams
  if ($LASTEXITCODE -ne 0) {
    $flags = @()
    if (Test-Path -LiteralPath $existingResult -PathType Leaf) {
      $flags = Get-FixedHardStopFlags $existingResult
    }
    Write-Host "FIXED_HALF_TRAINING_FAILED arm=$ArmName exit=$LASTEXITCODE flags=$($flags -join ';')"
    return [pscustomobject]@{ arm = $ArmName; status = 'FAILED'; dir = $armDir; hard_flags = ($flags -join '|') }
  }

  $hardFlags = @(Get-FixedHardStopFlags $existingResult | Where-Object { $_ })
  if ($hardFlags.Count -gt 0) {
    Write-Host "FIXED_HALF_HARD_STOP arm=$ArmName flags=$($hardFlags -join ';')"
    return [pscustomobject]@{ arm = $ArmName; status = 'UNSTABLE'; dir = $armDir; hard_flags = ($hardFlags -join '|') }
  }

  foreach ($step in $armEvalSteps) {
    $checkpoint = Join-Path $armDir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
    $evalDir = Join-Path $armDir ('eval256-step{0}' -f $step)
    $existingEval = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingEval) { Write-Host "eval_reuse arm=$ArmName step=$step"; continue }
    $evalParams = @{
      QairtSdkRoot = $QairtSdkRoot
      ExpectedBuildId = $ExpectedBuildId
      Seed = 1
      Layers = 19
      Heads = 2
      Tokens = 32
      Vocabulary = 1024
      Dimension = 64
      FeedForwardDimension = 128
      AttentionGate = $gate
      CheckpointStep = $step
      CheckpointPath = $checkpoint
      TokenizerModelPath = $TokenizerModelPath
      ValidationChunks = 256
      DevelopmentChunks = 256
      ReportRoot = $evalDir
      SkipBuild = $true
      SkipInstall = $true
    }
    & $evaluation @evalParams
    if ($LASTEXITCODE -ne 0) { throw "FIXED_HALF_EVAL_FAILED arm=$ArmName step=$step" }
  }

  # Current G1 only: checkpoint-static gate diagnostics.
  if ($ArmName -eq 'Current') {
    $diagExe = Join-Path $root 'build\host-tests\headwise_g1_gate_diagnostics.exe'
    if (-not (Test-Path -LiteralPath $diagExe -PathType Leaf)) {
      & g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
        -I (Join-Path $root 'app\src\main\cpp') `
        (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
        (Join-Path $root 'app\src\main\cpp\nicopedia_muon_checkpoint.cpp') `
        (Join-Path $root 'host_tests\headwise_g1_gate_diagnostics.cpp') `
        -o $diagExe
      if ($LASTEXITCODE -ne 0) { throw 'gate diagnostics build failed' }
    }
    foreach ($step in $armEvalSteps) {
      $ck = Join-Path $armDir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
      $diagPath = Join-Path $armDir ('gate-static-step{0}.txt' -f $step)
      if (Test-Path -LiteralPath $diagPath -PathType Leaf) { continue }
      & $diagExe $ck $TokenizerModelPath $ValidationCache 32 256 |
        Set-Content -LiteralPath $diagPath -Encoding utf8
      if ($LASTEXITCODE -ne 0) { Write-Host "GATE_DIAGNOSTICS_WARN arm=$ArmName step=$step" }
    }
  }

  Copy-Item -LiteralPath $existingResult -Destination (Join-Path $resultDir (Split-Path -Leaf $existingResult)) -Force
  $lrTelemetry = Join-Path $armDir 'learning-rate-telemetry.csv'
  if (Test-Path -LiteralPath $lrTelemetry -PathType Leaf) {
    Copy-Item -LiteralPath $lrTelemetry -Destination (Join-Path $resultDir 'learning-rate-telemetry.csv') -Force
  }
  foreach ($step in $armEvalSteps) {
    $evalDir = Join-Path $armDir ('eval256-step{0}' -f $step)
    $htp = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($htp) {
      Copy-Item -LiteralPath $htp.FullName -Destination (Join-Path $resultDir ("eval256-step$step-htp.txt")) -Force
    }
    $diagPath = Join-Path $armDir ('gate-static-step{0}.txt' -f $step)
    if (Test-Path -LiteralPath $diagPath -PathType Leaf) {
      Copy-Item -LiteralPath $diagPath -Destination (Join-Path $resultDir (Split-Path -Leaf $diagPath)) -Force
    }
  }

  Write-Host "arm_complete arm=$ArmName"
  return [pscustomobject]@{ arm = $ArmName; status = 'COMPLETED'; dir = $armDir; result_dir = $resultDir }
}

function Invoke-FixedAnalyze {
  if (-not (Test-Path -LiteralPath $analysisPy -PathType Leaf)) {
    Write-Host 'ANALYZE_PY_MISSING'
    return
  }
  $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
  & $python $analysisPy --report-root $ReportRoot --results-root $ResultsRoot --steps $Steps `
    --eval-steps ($evalSteps -join ',')
  if ($LASTEXITCODE -ne 0) { throw 'FIXED_HALF_ANALYZE_FAILED' }
}

if ($SelfTest -or $Mode -eq 'Plan') {
  if ($arms.Current.parameter_count -ne 760960) { throw 'SELFTEST_CURRENT_PARAMS' }
  if ($arms.FixedHalf.parameter_count -ne 758528) { throw 'SELFTEST_FIXED_PARAMS' }
  if ($arms.Current.attention_gate -ne 'headwise_g1_sigmoid') { throw 'SELFTEST_CURRENT_GATE' }
  if ($arms.FixedHalf.attention_gate -ne 'fixed_half') { throw 'SELFTEST_FIXED_GATE' }
  if ($arms.FixedHalf.checkpoint_format -ne 'NPRTCKPTV5') { throw 'SELFTEST_FIXED_FMT' }
  if ([double]$lr.muon_s -ne 0.0075) { throw 'SELFTEST_LR' }
  Write-Host 'run_g1_fixed_half_ab_self_test=PASS'
  Write-Host "arms=Current($($arms.Current.attention_gate),760960),FixedHalf($($arms.FixedHalf.attention_gate),758528)"
  Write-Host "lr=muon:$($lr.muon_s) aux:$($lr.aux_adam_s) target:$($lr.target_s)"
  if ($Mode -eq 'Plan') {
    Write-Host 'plan_only_no_device_work'
    Write-Host "run_order=Current->FixedHalf"
    Write-Host "report_root=$ReportRoot"
    Write-Host "results_root=$ResultsRoot"
  }
  exit 0
}

if ($Mode -eq 'Smoke') {
  if (-not (Test-FixedDeviceLock)) { Acquire-FixedDeviceLock }
  Enable-FixedDeviceAwake
  $outcome = Invoke-FixedArm -ArmName $Arm -ArmMode Smoke
  $armOutcome = @($outcome) | Where-Object { $_ -is [pscustomobject] -and $_.PSObject.Properties['arm'] } | Select-Object -Last 1
  Write-FixedJson (Join-Path $ReportRoot ('smoke-{0}.json' -f $Arm.ToLowerInvariant())) $armOutcome
  if (-not $armOutcome -or $armOutcome.status -notin @('COMPLETED', 'REUSED')) { throw 'SMOKE_FAILED' }
  Write-Host "PASS FIXED_SMOKE arm=$Arm"
  exit 0
}

if ($Mode -eq 'Run') {
  if (-not (Test-FixedDeviceLock)) { Acquire-FixedDeviceLock }
  $outcome = Invoke-FixedArm -ArmName $Arm -ArmMode Run
  $armOutcome = @($outcome) | Where-Object { $_ -is [pscustomobject] -and $_.PSObject.Properties['arm'] } | Select-Object -Last 1
  Write-FixedJson (Join-Path $ReportRoot ('arm-{0}.json' -f $Arm.ToLowerInvariant())) $armOutcome
  Write-Host "PASS FIXED_RUN arm=$Arm status=$($armOutcome.status)"
  exit 0
}

if ($Mode -eq 'Ab') {
  if (-not (Test-FixedDeviceLock)) { Acquire-FixedDeviceLock }
  Enable-FixedDeviceAwake
  $outcomes = @()
  try {
    foreach ($armName in @('Current','FixedHalf')) {
      $outcome = Invoke-FixedArm -ArmName $armName -ArmMode Run
      $armOutcome = @($outcome) | Where-Object { $_ -is [pscustomobject] -and $_.PSObject.Properties['arm'] } | Select-Object -Last 1
      $outcomes += $armOutcome
      Write-FixedJson (Join-Path $ReportRoot 'ab-outcomes.json') $outcomes
      Write-FixedJson (Join-Path $ResultsRoot 'ab-outcomes.json') $outcomes
    }
    Invoke-FixedAnalyze
    Write-Host 'PASS FIXED_AB'
    Write-Host ("outcomes={0}" -f ($outcomes | ForEach-Object { "$($_.arm)=$($_.status)" }) -join ' ')
  } finally {
    Release-FixedDeviceLock
  }
  exit 0
}

if ($Mode -eq 'Analyze') {
  Invoke-FixedAnalyze
  Write-Host 'PASS FIXED_ANALYZE'
  exit 0
}

throw "MODE_INVALID: $Mode"
