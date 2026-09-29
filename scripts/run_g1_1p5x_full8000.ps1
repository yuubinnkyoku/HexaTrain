# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# G1 1.5x full-horizon validation: continue step2000 -> 8000 (formal horizon).
# Arms: Control (ungated) and Current G1 only. No architecture variants.
# Modes: Plan | Run | Finish | Analyze
# Semantics: ResumeStep=2000 (absolute), Steps=8000 (absolute final step, NOT additional).
# Finish: idempotent post-training completion (resilient checkpoint pull, missing
# evals, gate diagnostics, artifact copy) for an arm whose device-side training
# already reported SUCCESS. Used after an ADB transport interruption.
[CmdletBinding()]
param(
  [ValidateSet('Control','G1')][string]$Arm = 'Control',
  [ValidateSet('Plan','Run','Finish','Analyze')][string]$Mode = 'Plan',
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$SourceRoot = 'build/g1-1p5x-long-2000-2026-09',
  [string]$ReportRoot = 'build/g1-1p5x-full-8000-2026-09',
  [string]$ResultsRoot = 'docs/results/g1-1p5x-full-8000-2026-09',
  [string]$DeviceLockRoot = 'D:\ghq\github.com\yuubinnkyoku\.hexatrain-device-lock',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [int]$ResumeStep = 2000,
  [int]$Steps = 8000,
  [int]$CheckpointInterval = 500,
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
$analysisPy = Join-Path $PSScriptRoot 'g1_1p5x_full8000_analyze.py'
$longPackage = 'com.yuubinnkyoku.phonelm'

# Contract: final absolute step 8000. ResumeStep is absolute (2000), Steps is
# the absolute final step (8000), NOT additional steps. 2000+8000=10000 is forbidden.
$evalSteps = @(2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000)
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
    "owner=g1-1p5x-full8000-agent", "pid=$PID",
    "timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))",
    "repo=$root", "experiment=g1-1p5x-full-8000-2026-09",
    "branch=$(git -C $root branch --show-current)"
  ) -Encoding utf8
  Write-Host "device_lock_acquired=$DeviceLockRoot"
}

function Release-LongDeviceLock {
  if (-not (Test-LongDeviceLock)) { return }
  $ownerPath = Get-LongLockOwnerPath
  if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    $text = Get-Content -LiteralPath $ownerPath -Raw
    if ($text -notmatch 'g1-1p5x-full8000-agent' -and $text -notmatch "pid=$PID") {
      throw "DEVICE_LOCK_NOT_OURS: refusing to release $DeviceLockRoot"
    }
  }
  Remove-Item -LiteralPath $DeviceLockRoot -Recurse -Force
  Write-Host "device_lock_released=$DeviceLockRoot"
}

function Enable-LongDeviceAwake {
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  $device = (Resolve-PhoneLmDevice -Adb $adb).Endpoint
  # Bounded, fail-open wake sequence: a stuck keyguard/WM call must not hang
  # the arm before training starts (observed as an unbounded adb client hang).
  foreach ($command in @(
      @('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP'),
      @('shell', 'wm', 'dismiss-keyguard'),
      @('shell', 'svc', 'power', 'stayon', 'true'),
      @('shell', 'dumpsys', 'deviceidle', 'disable'))) {
    $result = Invoke-PhoneLmAdb -Adb $adb -Device $device -Arguments $command -TimeoutSeconds 60 -AllowFailure
    if ($result.ExitCode -ne 0) {
      Write-Host "device_wake_warning command=$($command -join ' ') classification=$($result.Classification)"
    }
  }
  Write-Host "device_awake_and_idle_disabled=true endpoint=$device"
}

function Invoke-LongEvals([string]$ArmName) {
  $armDir = Get-LongArmDirectory $ArmName
  $gate = $arms[$ArmName].attention_gate
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
}

function Invoke-LongGateStatic([string]$ArmName) {
  if ($ArmName -ne 'G1') { return }
  $armDir = Get-LongArmDirectory $ArmName
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

function Copy-LongArtifacts([string]$ArmName) {
  # Publish the compact per-arm result bundle. Idempotent: every copy is
  # overwrite-safe, so Finish can republish after recovering new artifacts.
  $spec = $arms[$ArmName]
  $armDir = Get-LongArmDirectory $ArmName
  $resultDir = Get-LongResultsArmDirectory $ArmName
  New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
  $existingResult = Join-Path $armDir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $Steps)
  if (-not (Test-Path -LiteralPath $existingResult -PathType Leaf)) { throw "ARM_RESULT_MISSING arm=$ArmName" }
  Copy-Item -LiteralPath $existingResult -Destination (Join-Path $resultDir (Split-Path -Leaf $existingResult)) -Force
  foreach ($telemetry in @('learning-rate-telemetry.csv', 'training-curve-v1024-t32-d64-f128-8000.csv', 'training-curve-8000.csv')) {
    $src = Join-Path $armDir $telemetry
    if (Test-Path -LiteralPath $src -PathType Leaf) {
      Copy-Item -LiteralPath $src -Destination (Join-Path $resultDir $telemetry) -Force
    }
  }
  # Resolve LR schedule + manifest: same 1.5x trajectory for both arms.
  $manifest = [ordered]@{
    experiment = 'g1-1p5x-full-8000-2026-09'
    parent_2000_experiment = 'g1-1p5x-long-2000-2026-09'
    branch = (git -C $root branch --show-current)
    arm = $ArmName
    attention_gate = $spec.attention_gate
    parameter_count = $spec.parameter_count
    seed = 1
    dataset_hash = 'fnv1a64:0c7b2826f5f26fea'
    order_seed = 20260806
    resume_step_absolute = $ResumeStep
    steps_absolute_final = $Steps
    additional_steps = ($Steps - $ResumeStep)
    checkpoint_interval = $CheckpointInterval
    eval_steps = $evalSteps
    muon_peak_lr = $lr.muon_s
    aux_adam_peak_lr = $lr.aux_adam_s
    target_lr = $lr.target_s
    muon_target_lr = '0.0003409090859'
    schedule = 'linear_decay'
    decay_start_step = 4000
    decay_end_step = 8000
    schedule_total_steps = 8000
    experiment_fork = $true
  }
  Write-LongJson (Join-Path $resultDir 'run-manifest.json') $manifest
  Write-LongJson (Join-Path $armDir 'run-manifest.json') $manifest
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
  }
  # Historical 500-2000 evals live in the parent 2000-step results tree and are
  # reused as trajectory context (not re-evaluated). Only new steps are evaluated here.
  $parentResults = Join-Path $root 'docs/results/g1-1p5x-long-2000-2026-09'
  $parentArm = Join-Path $parentResults $ArmName.ToLowerInvariant()
  foreach ($step in @(500, 750, 1000, 1250, 1500, 1750, 2000)) {
    $parentHtp = Join-Path $parentArm ("eval256-step$step-htp.txt")
    if ((Test-Path -LiteralPath $parentHtp -PathType Leaf) -and -not (Test-Path -LiteralPath (Join-Path $resultDir ("eval256-step$step-htp.txt")))) {
      Copy-Item -LiteralPath $parentHtp -Destination (Join-Path $resultDir ("eval256-step$step-htp.txt")) -Force
    }
    if ($ArmName -eq 'G1') {
      $parentGate = Join-Path $parentArm ("gate-static-step$step.txt")
      if ((Test-Path -LiteralPath $parentGate -PathType Leaf) -and -not (Test-Path -LiteralPath (Join-Path $resultDir ("gate-static-step$step.txt")))) {
        Copy-Item -LiteralPath $parentGate -Destination (Join-Path $resultDir ("gate-static-step$step.txt")) -Force
      }
    }
  }
  Write-Host "artifacts_published arm=$ArmName result_dir=$resultDir"
  return [pscustomobject]@{ arm = $ArmName; result_dir = $resultDir }
}

function Get-LongArmRunId([string]$ArmName) {
  # The training runner stages inputs at files/headless-input/<RunId> and keeps
  # its host instrumentation stream in <reportRoot>/instrumentation-<RunId>.
  # Finish must read the directory the completed run actually wrote, so the run
  # id is resolved from the arm report tree instead of being re-derived.
  $armDir = Get-LongArmDirectory $ArmName
  $candidates = @(Get-ChildItem -Path $armDir -Directory -Filter 'instrumentation-*' -ErrorAction SilentlyContinue |
    Sort-Object -Property Name)
  if ($candidates.Count -ne 1) { throw "ARM_RUN_ID_AMBIGUOUS arm=$ArmName count=$($candidates.Count)" }
  return $candidates[0].Name.Substring('instrumentation-'.Length)
}

function Get-LongExpectedCheckpointSteps {
  # Absolute multiples strictly above ResumeStep, plus the final step. Mirrors
  # the training runner so Finish requires exactly the same checkpoint set.
  $expected = @()
  $first = if ($ResumeStep -gt 0) {
    ([Math]::Floor($ResumeStep / [double]$CheckpointInterval) + 1) * $CheckpointInterval
  } else { $CheckpointInterval }
  for ($s = $first; $s -le $Steps; $s += $CheckpointInterval) { $expected += $s }
  if ($expected -notcontains $Steps) { $expected += $Steps }
  return ,$expected
}

function Assert-LongCheckpointIdentity([string]$Path, [string]$Gate, [int]$Step) {
  # Same fail-closed format/identity/finiteness contract the training runner
  # applies to every pulled checkpoint.
  $magic = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($Path), 0, 11)
  $expectedMagic = if ($Gate -ne 'none') { "NPRTCKPTV5`n" } else { "NPRTCKPTV4`n" }
  if ($magic -ne $expectedMagic) { throw "CHECKPOINT_RESUME_FORMAT_INVALID: $(Split-Path -Leaf $Path)" }
  $hostEvalExe = Join-Path $root 'build\host-tests\htp_checkpoint_eval.exe'
  $validationHost = [IO.Path]::GetFullPath($ValidationCache)
  $developmentHost = Join-Path (Split-Path -Parent $validationHost) 'development.bin'
  Ensure-PhoneLmHostCheckpointEvaluator -Root $root -ExePath $hostEvalExe | Out-Null
  if (-not (Test-Path -LiteralPath $hostEvalExe -PathType Leaf) -or
      -not (Test-Path -LiteralPath $validationHost -PathType Leaf) -or
      -not (Test-Path -LiteralPath $developmentHost -PathType Leaf)) {
    throw 'HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE'
  }
  $hostDecoded = & $hostEvalExe $Path $validationHost $developmentHost 1 1
  if ($LASTEXITCODE -ne 0) { throw "HOST_CHECKPOINT_EVALUATOR_DECODE_FAILED: $(Split-Path -Leaf $Path)" }
  $hostIdentity = Get-PhoneLmKeyValueMap -Text ($hostDecoded -join "`n")
  foreach ($field in @('seed', 'layers', 'step', 'parameter_hash', 'finite')) {
    if (-not $hostIdentity.Contains($field)) { throw "HOST_CHECKPOINT_EVALUATOR_FIELD_MISSING: $field" }
  }
  if ([int]$hostIdentity.seed -ne 1 -or [int]$hostIdentity.layers -ne 19 -or
      [int]$hostIdentity.step -ne $Step -or $hostIdentity.finite -ne 'true') {
    throw "HOST_CHECKPOINT_EVALUATOR_IDENTITY_MISMATCH: $(Split-Path -Leaf $Path)"
  }
  return $hostIdentity
}

function Pull-LongMissingCheckpoints([string]$ArmName) {
  # Finish never re-trains. It only recovers the artifacts the interrupted adb
  # transport did not deliver, then re-verifies them with the same gates.
  $gate = $arms[$ArmName].attention_gate
  $armDir = Get-LongArmDirectory $ArmName
  $runId = Get-LongArmRunId $ArmName
  $remoteDir = "files/headless-input/$runId"
  $expectedSteps = Get-LongExpectedCheckpointSteps
  $missing = @()
  foreach ($step in $expectedSteps) {
    $name = Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 64 -FeedForwardDimension 128 -Step $step
    $local = Join-Path $armDir $name
    if (Test-Path -LiteralPath $local -PathType Leaf) { continue }
    $missing += [pscustomobject]@{ step = $step; name = $name; local = $local }
  }
  $artifactNames = @("training-curve-$Steps.csv", 'learning-rate-telemetry.csv')
  $missingArtifacts = @()
  foreach ($artifactName in $artifactNames) {
    $artifactLocal = Join-Path $armDir $artifactName
    if (Test-Path -LiteralPath $artifactLocal -PathType Leaf) { continue }
    $missingArtifacts += [pscustomobject]@{ name = $artifactName; local = $artifactLocal }
  }
  if ($missing.Count -eq 0 -and $missingArtifacts.Count -eq 0) {
    Write-Host "checkpoint_pull_noop arm=$ArmName run_id=$runId expected=$($expectedSteps.Count)"
    return [pscustomobject]@{ arm = $ArmName; run_id = $runId; pulled_checkpoints = 0; pulled_artifacts = @() }
  }
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  $device = (Resolve-PhoneLmDevice -Adb $adb).Endpoint
  Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
  # Active-run protection: Finish only reads. A live session makes the device
  # directory contents ambiguous, so fail closed instead of pulling from it.
  Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package $longPackage
  Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package $longPackage
  # Raw listing, not Get-PhoneLmCheckpointNames: the curve and the LR
  # telemetry live in the same directory but are not checkpoint files.
  $remoteNames = @((Invoke-PhoneLmAdb -Adb $adb -Device $device `
    -Arguments @('shell', 'run-as', $longPackage, 'ls', '-1', $remoteDir)).Text -split "`r?`n" |
    Where-Object { $_ } | Sort-Object -Unique)
  if ($remoteNames.Count -eq 0) { throw "REMOTE_RUN_DIR_EMPTY: $remoteDir" }
  $pulledCheckpoints = 0
  foreach ($item in $missing) {
    if ($remoteNames -notcontains $item.name) { throw "REMOTE_CKPT_MISSING: $($item.name) dir=$remoteDir" }
    $pulled = Receive-PhoneLmBinary -Adb $adb -Device $device -Package $longPackage `
      -RemotePath "$remoteDir/$($item.name)" -LocalPath $item.local -MinimumBytes 1024
    $identity = Assert-LongCheckpointIdentity -Path $item.local -Gate $gate -Step $item.step
    $pulledCheckpoints++
    Write-Host "checkpoint_pulled arm=$ArmName step=$($item.step) size=$($pulled.Size) sha256=$($pulled.Sha256) parameter_hash=$($identity.parameter_hash)"
  }
  $pulledArtifacts = @()
  foreach ($artifact in $missingArtifacts) {
    if ($remoteNames -notcontains $artifact.name) { throw "REMOTE_ARTIFACT_MISSING: $($artifact.name) dir=$remoteDir" }
    Receive-PhoneLmBinary -Adb $adb -Device $device -Package $longPackage `
      -RemotePath "$remoteDir/$($artifact.name)" -LocalPath $artifact.local -MinimumBytes 1 | Out-Null
    $pulledArtifacts += $artifact.name
    Write-Host "artifact_pulled arm=$ArmName name=$($artifact.name)"
  }
  Write-Host "checkpoint_pull_complete arm=$ArmName run_id=$runId pulled=$pulledCheckpoints expected=$($expectedSteps.Count) artifacts=$($pulledArtifacts -join ',')"
  return [pscustomobject]@{ arm = $ArmName; run_id = $runId; pulled_checkpoints = $pulledCheckpoints; pulled_artifacts = $pulledArtifacts }
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

  # Stage step2000 checkpoint and source identity for resume.
  # Resume identity contract: checkpoint SHA256, parameter hash, optimizer-state
  # identity, step=2000, DataCursor/orderSeed=20260806, dataset, tokenizer,
  # LR/schedule (linear_decay 4000->8000), attention_gate are verified by the
  # device resume path (step/seed/config identity) plus the manifest checks below.
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

  # Evaluations and static gate diagnostics are idempotent helpers so Run and
  # Finish share one implementation and one gate policy.
  Invoke-LongEvals -ArmName $ArmName
  Invoke-LongGateStatic -ArmName $ArmName
  Copy-LongArtifacts -ArmName $ArmName

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
  if ($ResumeStep -ne 2000 -or $Steps -ne 8000) { throw 'SELFTEST_RANGE' }
  if ($Steps -ne 8000) { throw 'SELFTEST_FINAL_STEP_MUST_BE_8000' }
  if ($lr.aux_adam_s -ne '0.0033' -or $lr.muon_s -ne '0.0075' -or $lr.target_s -ne '0.00015') { throw 'SELFTEST_LR_1P5X' }
  Write-Host 'run_g1_1p5x_full8000_self_test=PASS'
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
  $lockAcquiredHere = $false
  try {
    if (-not (Test-LongDeviceLock)) { Acquire-LongDeviceLock; $lockAcquiredHere = $true }
    Enable-LongDeviceAwake
    $outcome = Invoke-LongArm -ArmName $Arm
    $armOutcome = @($outcome) | Where-Object { $_ -is [pscustomobject] -and $_.PSObject.Properties['arm'] } | Select-Object -Last 1
    Write-LongJson (Join-Path $ReportRoot ('arm-{0}.json' -f $Arm.ToLowerInvariant())) $armOutcome
    Write-Host "PASS LONG_RUN arm=$Arm status=$($armOutcome.status)"
    exit 0
  } finally {
    # Always release the lock this process acquired (finally equivalent).
    # Release-LongDeviceLock refuses to touch locks owned by other agents.
    if ($lockAcquiredHere) { Release-LongDeviceLock }
  }
}

if ($Mode -eq 'Finish') {
  # Finish is the post-training recovery path for an arm whose device-side run
  # already reported SUCCESS. It never re-trains: it re-publishes the training
  # verdict, pulls only the artifacts the interrupted transport did not
  # deliver, evaluates every missing step, and republishes the result bundle.
  $lockAcquiredHere = $false
  try {
    if (-not (Test-LongDeviceLock)) { Acquire-LongDeviceLock; $lockAcquiredHere = $true }
    $armDir = Get-LongArmDirectory $Arm
    if (-not (Test-Path -LiteralPath $armDir -PathType Container)) { throw "ARM_DIR_MISSING: $armDir" }
    $armResult = Join-Path $armDir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $Steps)
    if (-not (Test-Path -LiteralPath $armResult -PathType Leaf)) { throw "TRAINING_RESULT_MISSING: $armResult" }
    $map = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $armResult -Raw)
    if (-not ($map.Contains('status') -and $map.status -eq 'SUCCESS')) {
      throw "TRAINING_RESULT_NOT_SUCCESS: status=$($map.status)"
    }
    if (-not ($map.Contains('completed_steps') -and [int]$map.completed_steps -eq $Steps)) {
      throw "TRAINING_INCOMPLETE: completed_steps=$($map.completed_steps) expected=$Steps"
    }
    $finishFlags = @(Get-LongHardStopFlags $armResult | Where-Object { $_ })
    if ($finishFlags.Count -gt 0) { throw "LONG_HARD_STOP arm=$Arm flags=$($finishFlags -join ';')" }
    Write-Host "training_already_complete arm=$Arm completed_steps=$($map.completed_steps) final_parameter_hash=$($map.final_parameter_hash)"
    Enable-LongDeviceAwake
    $pullOutcome = Pull-LongMissingCheckpoints -ArmName $Arm
    Invoke-LongEvals -ArmName $Arm
    Invoke-LongGateStatic -ArmName $Arm
    $publishOutcome = Copy-LongArtifacts -ArmName $Arm
    $finish = [pscustomobject][ordered]@{
      arm = $Arm
      status = 'FINISHED'
      mode = 'Finish'
      dir = $armDir
      result_dir = (Get-LongResultsArmDirectory $Arm)
      run_id = $pullOutcome.run_id
      pulled_checkpoints = $pullOutcome.pulled_checkpoints
      pulled_artifacts = @($pullOutcome.pulled_artifacts)
      final_parameter_hash = $map.final_parameter_hash
      published = [bool]$publishOutcome
    }
    Write-LongJson (Join-Path $ReportRoot ('arm-{0}-finish.json' -f $Arm.ToLowerInvariant())) $finish
    Write-Host "PASS LONG_FINISH arm=$Arm status=$($finish.status) pulled=$($finish.pulled_checkpoints)"
    exit 0
  } finally {
    # Always release the lock this process acquired (finally equivalent).
    # Release-LongDeviceLock refuses to touch locks owned by other agents.
    if ($lockAcquiredHere) { Release-LongDeviceLock }
  }
}

if ($Mode -eq 'Analyze') {
  Invoke-LongAnalyze
  Write-Host 'PASS LONG_ANALYZE'
  exit 0
}

throw "MODE_INVALID: $Mode"
