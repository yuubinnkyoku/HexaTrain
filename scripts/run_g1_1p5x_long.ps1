# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# G1 1.5x long-horizon validation: continue stress-grid step500 → 2000.
# Arms: Control (ungated) and Current G1 only. No architecture variants.
# Modes: Plan | Run | Analyze
[CmdletBinding()]
param(
  [ValidateSet('Control','G1')][string]$Arm = 'Control',
  [ValidateSet('Plan','Run','Analyze')][string]$Mode = 'Plan',
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$SourceRoot = 'build/g1-lr-stress/lr1.5',
  [string]$ReportRoot = 'build/g1-1p5x-long-2000-2026-09',
  [string]$ResultsRoot = 'docs/results/g1-1p5x-long-2000-2026-09',
  [string]$DeviceLockRoot = 'D:\ghq\github.com\yuubinnkyoku\.hexatrain-device-lock',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [int]$ResumeStep = 500,
  [int]$Steps = 2000,
  [int]$CheckpointInterval = 250,
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
$analysisPy = Join-Path $PSScriptRoot 'g1_1p5x_long_analyze.py'

$evalSteps = @(500, 750, 1000, 1250, 1500, 1750, 2000)
$newEvalSteps = @($evalSteps | Where-Object { $_ -gt $ResumeStep })
$lr = [ordered]@{ aux_adam_s = '0.0033'; muon_s = '0.0075'; target_s = '0.00015' }

$arms = [ordered]@{
  Control = [ordered]@{ attention_gate = 'none'; source = 'control'; parameter_count = 758528 }
  G1 = [ordered]@{ attention_gate = 'headwise_g1_sigmoid'; source = 'g1'; parameter_count = 760960 }
}

function Get-LongArmDirectory([string]$ArmName) { Join-Path $ReportRoot $ArmName.ToLowerInvariant() }
function Get-LongResultsArmDirectory([string]$ArmName) { Join-Path $ResultsRoot $ArmName.ToLowerInvariant() }

function Write-LongJson([string]$Path, $Payload) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  $Payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Get-LongHardStopFlags([string]$ResultPath) {
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

function Get-LongLockOwnerPath { return Join-Path $DeviceLockRoot 'owner.txt' }
function Test-LongDeviceLock { return (Test-Path -LiteralPath $DeviceLockRoot -PathType Container) }

function Acquire-LongDeviceLock {
  if (Test-LongDeviceLock) {
    $owner = if (Test-Path -LiteralPath (Get-LongLockOwnerPath)) {
      (Get-Content -LiteralPath (Get-LongLockOwnerPath) -Raw) } else { '' }
    throw ("DEVICE_LOCK_HELD path=$DeviceLockRoot owner=$($owner -replace '\n',' | ')")
  }
  New-Item -ItemType Directory -Force -Path $DeviceLockRoot | Out-Null
  Set-Content -LiteralPath (Get-LongLockOwnerPath) -Value @(
    "owner=g1-1p5x-long-agent", "pid=$PID",
    "timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))",
    "repo=$root", "experiment=g1-1p5x-long-2000-2026-09",
    "branch=$(git -C $root branch --show-current)"
  ) -Encoding utf8
  Write-Host "device_lock_acquired=$DeviceLockRoot"
}

function Release-LongDeviceLock {
  if (-not (Test-LongDeviceLock)) { return }
  $ownerPath = Get-LongLockOwnerPath
  if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    $text = Get-Content -LiteralPath $ownerPath -Raw
    if ($text -notmatch 'g1-1p5x-long-agent' -and $text -notmatch "pid=$PID") {
      throw "DEVICE_LOCK_NOT_OURS: refusing to release $DeviceLockRoot"
    }
  }
  Remove-Item -LiteralPath $DeviceLockRoot -Recurse -Force
  Write-Host "device_lock_released=$DeviceLockRoot"
}

function Enable-LongDeviceAwake {
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  $dev = (Resolve-PhoneLmDevice -Adb $adb).Endpoint
  & $adb -s $dev shell input keyevent KEYCODE_WAKEUP | Out-Null
  & $adb -s $dev shell wm dismiss-keyguard | Out-Null
  & $adb -s $dev shell svc power stayon true | Out-Null
  & $adb -s $dev shell dumpsys deviceidle disable | Out-Null
  Write-Host "device_awake_and_idle_disabled=true endpoint=$dev"
}

function Invoke-LongArm([string]$ArmName) {
  $spec = $arms[$ArmName]
  $gate = $spec.attention_gate
  $srcDir = Join-Path $SourceRoot $spec.source
  $armDir = Get-LongArmDirectory $ArmName
  $resultDir = Get-LongResultsArmDirectory $ArmName
  New-Item -ItemType Directory -Force -Path $armDir | Out-Null
  New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

  $srcCkpt = Join-Path $srcDir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $ResumeStep)
  $srcResult = Get-ChildItem $srcDir -Filter ('seed1-*-steps{0}-result.txt' -f $ResumeStep) | Select-Object -First 1
  if (-not (Test-Path -LiteralPath $srcCkpt -PathType Leaf)) { throw "SOURCE_CKPT_MISSING: $srcCkpt" }
  if (-not $srcResult) { throw "SOURCE_RESULT_MISSING: $srcDir" }

  # Stage step500 checkpoint and source identity for resume.
  Copy-Item -LiteralPath $srcCkpt -Destination (Join-Path $armDir (Split-Path -Leaf $srcCkpt)) -Force
  Copy-Item -LiteralPath $srcResult.FullName -Destination (Join-Path $resultDir ('source-' + $srcResult.Name)) -Force

  $existingResult = Join-Path $armDir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $Steps)
  if (Test-Path -LiteralPath $existingResult -PathType Leaf) {
    $map = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $existingResult -Raw)
    if ($map.Contains('status') -and $map.status -eq 'SUCCESS' -and
        $map.Contains('completed_steps') -and [int]$map.completed_steps -eq $Steps) {
      Write-Host "arm_already_complete arm=$ArmName"
      return [pscustomobject]@{ arm = $ArmName; status = 'REUSED'; dir = $armDir }
    }
  }

  Write-Host "=== LONG ARM arm=$ArmName gate=$gate resume=$ResumeStep -> $Steps ==="
  $trainParams = @{
    QairtSdkRoot = $QairtSdkRoot
    ExpectedBuildId = $ExpectedBuildId
    Seed = 1
    Layers = 19
    Steps = $Steps
    ResumeStep = $ResumeStep
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
    CheckpointInterval = $CheckpointInterval
    CheckpointStallSeconds = 1800
    SkipBuild = [bool]$SkipBuild
    SkipInstall = [bool]$SkipInstall
  }
  & $training @trainParams
  if ($LASTEXITCODE -ne 0) {
    $flags = @()
    if (Test-Path -LiteralPath $existingResult -PathType Leaf) {
      $flags = Get-LongHardStopFlags $existingResult
    }
    Write-Host "LONG_TRAINING_FAILED arm=$ArmName flags=$($flags -join ';')"
    return [pscustomobject]@{ arm = $ArmName; status = 'FAILED'; dir = $armDir; hard_flags = ($flags -join '|') }
  }

  $hardFlags = @(Get-LongHardStopFlags $existingResult | Where-Object { $_ })
  if ($hardFlags.Count -gt 0) {
    Write-Host "LONG_HARD_STOP arm=$ArmName flags=$($hardFlags -join ';')"
    return [pscustomobject]@{ arm = $ArmName; status = 'UNSTABLE'; dir = $armDir; hard_flags = ($hardFlags -join '|') }
  }

  foreach ($step in $newEvalSteps) {
    $checkpoint = Join-Path $armDir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
    $evalDir = Join-Path $armDir ('eval256-step{0}' -f $step)
    $existingEval = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingEval) { Write-Host "eval_reuse arm=$ArmName step=$step"; continue }
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "CKPT_MISSING arm=$ArmName step=$step" }
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
    if ($LASTEXITCODE -ne 0) { throw "LONG_EVAL_FAILED arm=$ArmName step=$step" }
  }

  # G1 only: static gate diagnostics at each new checkpoint.
  if ($ArmName -eq 'G1') {
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
    foreach ($step in $evalSteps) {
      $ck = Join-Path $armDir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
      $diagPath = Join-Path $armDir ('gate-static-step{0}.txt' -f $step)
      if (Test-Path -LiteralPath $diagPath -PathType Leaf) { continue }
      if (-not (Test-Path -LiteralPath $ck -PathType Leaf)) { continue }
      & $diagExe $ck $TokenizerModelPath $ValidationCache 32 256 |
        Set-Content -LiteralPath $diagPath -Encoding utf8
      if ($LASTEXITCODE -ne 0) { Write-Host "GATE_DIAGNOSTICS_WARN arm=$ArmName step=$step" }
    }
  }

  # Copy compact artifacts.
  Copy-Item -LiteralPath $existingResult -Destination (Join-Path $resultDir (Split-Path -Leaf $existingResult)) -Force
  foreach ($telemetry in @('learning-rate-telemetry.csv', 'training-curve-v1024-t32-d64-f128-2000.csv', 'training-curve-2000.csv')) {
    $src = Join-Path $armDir $telemetry
    if (Test-Path -LiteralPath $src -PathType Leaf) {
      Copy-Item -LiteralPath $src -Destination (Join-Path $resultDir $telemetry) -Force
    }
  }
  foreach ($step in $evalSteps) {
    $evalDir = Join-Path $armDir ('eval256-step{0}' -f $step)
    $htp = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($htp) {
      Copy-Item -LiteralPath $htp.FullName -Destination (Join-Path $resultDir ("eval256-step$step-htp.txt")) -Force
    }
    $diagPath = Join-Path $armDir ('gate-static-step{0}.txt' -f $step)
    if (Test-Path -LiteralPath $diagPath -PathType Leaf) {
      Copy-Item -LiteralPath $diagPath -Destination (Join-Path $resultDir (Split-Path -Leaf $diagPath)) -Force
    }
    # Also copy source 500-step eval from stress tree for complete curve.
  }
  $src500Eval = Join-Path $srcDir 'eval256-step500'
  $src500Htp = Get-ChildItem -Path $src500Eval -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($src500Htp -and -not (Test-Path (Join-Path $resultDir 'eval256-step500-htp.txt'))) {
    Copy-Item -LiteralPath $src500Htp.FullName -Destination (Join-Path $resultDir 'eval256-step500-htp.txt') -Force
  }
  if ($ArmName -eq 'G1') {
    $src500Gate = Join-Path $srcDir 'gate-static-step500.txt'
    if ((Test-Path $src500Gate) -and -not (Test-Path (Join-Path $resultDir 'gate-static-step500.txt'))) {
      Copy-Item -LiteralPath $src500Gate -Destination (Join-Path $resultDir 'gate-static-step500.txt') -Force
    }
  }

  Write-Host "arm_complete arm=$ArmName"
  return [pscustomobject]@{ arm = $ArmName; status = 'COMPLETED'; dir = $armDir; result_dir = $resultDir }
}

function Invoke-LongAnalyze {
  if (-not (Test-Path -LiteralPath $analysisPy -PathType Leaf)) {
    Write-Host 'ANALYZE_PY_MISSING'
    return
  }
  $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
  & $python $analysisPy --report-root $ReportRoot --results-root $ResultsRoot `
    --steps $Steps --eval-steps ($evalSteps -join ',')
  if ($LASTEXITCODE -ne 0) { throw 'LONG_ANALYZE_FAILED' }
}

if ($SelfTest -or $Mode -eq 'Plan') {
  if ($arms.Control.parameter_count -ne 758528) { throw 'SELFTEST_CONTROL_PARAMS' }
  if ($arms.G1.parameter_count -ne 760960) { throw 'SELFTEST_G1_PARAMS' }
  if ($arms.G1.attention_gate -ne 'headwise_g1_sigmoid') { throw 'SELFTEST_G1_GATE' }
  if ($ResumeStep -ne 500 -or $Steps -ne 2000) { throw 'SELFTEST_RANGE' }
  Write-Host 'run_g1_1p5x_long_self_test=PASS'
  Write-Host "resume=$ResumeStep steps=$Steps interval=$CheckpointInterval"
  Write-Host "eval_steps=$($evalSteps -join ',')"
  Write-Host "run_order=Control->G1"
  if ($Mode -eq 'Plan') {
    Write-Host 'plan_only_no_device_work'
    Write-Host "source_root=$SourceRoot"
    Write-Host "report_root=$ReportRoot"
    Write-Host "results_root=$ResultsRoot"
  }
  exit 0
}

if ($Mode -eq 'Run') {
  if (-not (Test-LongDeviceLock)) { Acquire-LongDeviceLock }
  Enable-LongDeviceAwake
  $outcome = Invoke-LongArm -ArmName $Arm
  $armOutcome = @($outcome) | Where-Object { $_ -is [pscustomobject] -and $_.PSObject.Properties['arm'] } | Select-Object -Last 1
  Write-LongJson (Join-Path $ReportRoot ('arm-{0}.json' -f $Arm.ToLowerInvariant())) $armOutcome
  Write-Host "PASS LONG_RUN arm=$Arm status=$($armOutcome.status)"
  exit 0
}

if ($Mode -eq 'Analyze') {
  Invoke-LongAnalyze
  Write-Host 'PASS LONG_ANALYZE'
  exit 0
}

throw "MODE_INVALID: $Mode"
