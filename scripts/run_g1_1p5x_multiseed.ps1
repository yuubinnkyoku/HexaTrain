# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# G1 1.5x multi-seed replication: fresh step-0 starts on seeds outside seed 1.
# Protocol: docs/g1-1p5x-multiseed-3000.md (criteria R1-R5 are pre-registered).
# Control (ungated) vs headwise_g1_sigmoid only; no architecture or LR variants.
# Seeds 2 and 4 are pre-registered; another seed needs -AllowExploratorySeed and
# is never mixed into the pre-registered R1-R5 decision by the analyzer.
# Modes: Plan | Smoke | Seed | All | Analyze
[CmdletBinding()]
param(
  [ValidateSet('Plan','Smoke','Seed','All','Analyze')][string]$Mode = 'Plan',
  [object[]]$Seeds = @(2,4),
  [ValidateSet('All','Control','G1')][string]$Arm = 'All',
  [switch]$AllowExploratorySeed,
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$ReportRoot = 'build/g1-1p5x-multiseed',
  [string]$ResultsRoot = 'docs/results/g1-1p5x-multiseed-3000-2026-09',
  [string]$DeviceLockRoot = 'D:\ghq\github.com\yuubinnkyoku\.hexatrain-device-lock',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [int]$Steps = 3000,
  [int]$CheckpointInterval = 250,
  [object[]]$EvalSteps = @(500,1000,1500,1750,2000,2500,3000),
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
function ConvertTo-MultiseedIntList([object]$Value) {
  # A `-File` invocation cannot carry array literals, so "-Seeds 2,4" arrives as a
  # single string. Both an int array and a comma / space separated string are valid.
  if ($null -eq $Value) { return @() }
  if ($Value -is [string]) {
    return @($Value -split '[,\s]+' | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
  }
  if ($Value -is [int] -or $Value -is [long]) { return @([int]$Value) }
  $values = @()
  foreach ($item in $Value) { $values += ConvertTo-MultiseedIntList $item }
  return $values
}
$Seeds = ConvertTo-MultiseedIntList $Seeds
$EvalSteps = ConvertTo-MultiseedIntList $EvalSteps
if ($Seeds.Count -eq 0) { throw 'SEEDS_EMPTY: pass at least one seed greater than 1' }
if ($EvalSteps.Count -eq 0) { throw 'EVAL_STEPS_EMPTY: pass at least one evaluation step' }


$root = Split-Path -Parent $PSScriptRoot
$training = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$evaluation = Join-Path $PSScriptRoot 'run_nicopedia_htp_eval.ps1'
$analysisPy = Join-Path $PSScriptRoot 'g1_multiseed_analyze.py'
$epochTag = 'g1-1p5x-multiseed-3000-2026-09'
$lockOwner = 'g1-multiseed-agent'

# Pre-registered in docs/g1-1p5x-multiseed-3000.md.  Adding a seed after seeing
# results is a protocol event: it needs -AllowExploratorySeed, it is recorded in
# seed-registry.json, and the analyzer keeps it out of the pre-registered
# decision.  If this list changes, the document has to change first.
$preregisteredSeeds = @(2, 4)

# Fixed 1.5x identity. Exact decimal strings keep the LR fields free of binary
# float drift; they are the same values committed in g1-lr-stress and the
# 2000 / 8000 step trees.
$lr15 = [ordered]@{
  muon = '0.0075'
  aux_adam = '0.0033'
  target = '0.00015'
}

# Schedule stays anchored to the 8000-step horizon so steps 1-3000 run at the
# constant peak LR, exactly as seed 1 did before its decay started at 4000.
$base = [ordered]@{
  seed_reference = 1
  batch = 8
  vocabulary = 1024
  tokens = 32
  dimension = 64
  feed_forward = 128
  layers = 19
  heads = 2
  schedule = 'linear_decay'
  decay_start = 4000
  decay_end = 8000
  schedule_total = 8000
  muon_momentum = '0.95'
  muon_ns_steps = 5
  optimizer = 'Muon'
  muon_backend = 'HVX'
  reported_optimizer = 'muon_aux_adam'
  reported_muon_backend = 'HVX_W8'
  validation_chunks = 256
  development_chunks = 256
  dataset_hash = 'fnv1a64:0c7b2826f5f26fea'
  training_order_seed = 20260806
  tokenizer_identity = 'byte-bpe-v1024'
  evaluation_protocol = 'val-first-256-dev-first-256-v1024-bpe'
  control_parameter_count = 758528
  g1_parameter_count = 760960
}

function Get-MultiseedSeedRole([int]$Seed) {
  if ($preregisteredSeeds -contains $Seed) { return 'preregistered' }
  if ($Seed -eq $base.seed_reference) { return 'reference' }
  return 'exploratory'
}

function Get-MultiseedArmIdentity([string]$ArmName) {
  $isG1 = $ArmName -eq 'G1'
  return [ordered]@{
    arm = $ArmName
    attention_gate = $(if ($isG1) { 'headwise_g1_sigmoid' } else { 'none' })
    parameter_count = $(if ($isG1) { $base.g1_parameter_count } else { $base.control_parameter_count })
    checkpoint_format = $(if ($isG1) { 'NPRTCKPTV5' } else { 'NPRTCKPTV4' })
    muon_lr = $lr15.muon
    aux_adam_lr = $lr15.aux_adam
    target_lr = $lr15.target
  }
}

function Test-MultiseedProtocol {
  # The protocol only means something while steps 1-3000 sit on the constant
  # peak of the 8000-step schedule and both verdict bands are observed.
  if ($Steps -gt $base.decay_start) {
    throw "STEPS_EXCEED_DECAY_START steps=$Steps decay_start=$($base.decay_start)"
  }
  if ($Steps -lt $EvalSteps[-1]) {
    throw "EVAL_STEP_EXCEEDS_STEPS last_eval=$($EvalSteps[-1]) steps=$Steps"
  }
  if ($Mode -ne 'Smoke' -and ($EvalSteps -notcontains 1000)) {
    throw "EVAL_STEPS_MISSING_R1_BAND eval_steps=$($EvalSteps -join ',') requires 1000"
  }
  $lateBand = @($EvalSteps | Where-Object { $_ -ge 1750 -and $_ -le 3000 })
  if ($Mode -ne 'Smoke' -and $lateBand.Count -eq 0) {
    throw "EVAL_STEPS_MISSING_R2_BAND eval_steps=$($EvalSteps -join ',') requires a step in 1750-3000"
  }
  foreach ($seed in $Seeds) {
    if ($seed -le $base.seed_reference) {
      throw "SEED_NOT_FRESH: seed=$seed, fresh replication needs seed > $($base.seed_reference)"
    }
    if ((Get-MultiseedSeedRole $seed) -ne 'preregistered' -and -not $AllowExploratorySeed) {
      throw ("SEED_NOT_PREREGISTERED: seed={0} preregistered={1}; pass -AllowExploratorySeed to " +
        "run it as an exploratory sample (excluded from the R1-R5 decision)") -f `
        $seed, ($preregisteredSeeds -join ',')
    }
  }
  if ($Mode -eq 'Seed' -and $Arm -eq 'All') {
    throw 'ARM_REQUIRED: -Mode Seed runs one arm per session, pass -Arm Control or -Arm G1'
  }
}

function Get-MultiseedSeedRegistry {
  # Cumulative on purpose: seed 2 and seed 4 are separate sessions, and the
  # registry of the second session must not drop the first one's provenance.
  $registry = [ordered]@{}
  foreach ($path in @((Join-Path (Resolve-MultiseedPath $ReportRoot) 'seed-registry.json'),
                      (Join-Path (Resolve-MultiseedPath $ResultsRoot) 'seed-registry.json'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    try {
      $data = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
      foreach ($entry in @($data.seeds)) {
        if ($null -eq $entry) { continue }
        $registry[[int]$entry.seed] = [string]$entry.role
      }
    } catch {
      Write-Host "seed_registry_read_soft_fail=$path"
    }
  }
  foreach ($seed in $preregisteredSeeds) { $registry[$seed] = 'preregistered' }
  foreach ($seed in $Seeds) {
    if (-not $registry.Contains([int]$seed)) { $registry[[int]$seed] = 'exploratory' }
  }
  $exploratory = @($registry.Keys | Where-Object { $registry[$_] -eq 'exploratory' } | Sort-Object)
  return [ordered]@{
    version = 1
    protocol = 'docs/g1-1p5x-multiseed-3000.md'
    preregistered_seeds = @($preregisteredSeeds)
    allow_exploratory_seed = [bool]$AllowExploratorySeed
    exploratory_seeds = @($exploratory)
    seeds = @($registry.Keys | Sort-Object | ForEach-Object {
        [ordered]@{ seed = [int]$_; role = $registry[$_] }
      })
  }
}

function Get-MultiseedArmDirectory([int]$Seed, [string]$ArmName,
                                   [string]$RelativeRoot = 'Report') {
  $rootDir = $(if ($RelativeRoot -eq 'Results') { $ResultsRoot } else { $ReportRoot })
  return Join-Path (Join-Path $rootDir ('seed{0}' -f $Seed)) $ArmName.ToLowerInvariant()
}

function Resolve-MultiseedPath([string]$Path) {
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $root $Path
}

function New-MultiseedDirectory([string]$Path) {
  [IO.Directory]::CreateDirectory($Path) | Out-Null
}

function Write-MultiseedText([string]$Path, [string]$Text) {
  # BOM-less UTF-8: the analyzer (python) and git both read these files directly,
  # and a UTF-8 BOM is not valid JSON for strict readers.
  New-MultiseedDirectory (Split-Path -Parent $Path)
  [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}

function Write-MultiseedJson([string]$Path, $Payload) {
  Write-MultiseedText $Path ($Payload | ConvertTo-Json -Depth 12)
}

function Get-MultiseedKeyValue([string]$Path) {
  return Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $Path -Raw)
}

function Get-MultiseedHardStopFlags([string]$ResultPath) {
  $map = Get-MultiseedKeyValue $ResultPath
  $flags = @()
  if (-not $map.Contains('status') -or $map.status -ne 'SUCCESS') {
    $flags += 'status=' + $(if ($map.Contains('status')) { $map.status } else { '<missing>' })
  }
  foreach ($pair in @(
      @('all_steps_finite','true'),
      @('final_finite','true'),
      @('output_tensors_finite','true'),
      @('qnn_return_code_success','true'),
      @('cpu_fallback','false'),
      @('nan_detected','false'),
      @('inf_detected','false'))) {
    if (-not $map.Contains($pair[0]) -or $map[$pair[0]] -ne $pair[1]) {
      $flags += ('{0}={1}' -f $pair[0], $(if ($map.Contains($pair[0])) { $map[$pair[0]] } else { '<missing>' }))
    }
  }
  foreach ($pair in @(
      @('api_trace_graph_execute_failure_count','0'),
      @('hvx_rpc_failure_count','0'),
      @('hvx_fallback_count','0'),
      @('hvx_nonfinite_count','0'),
      @('focus_takeover_count','0'))) {
    if ($map.Contains($pair[0]) -and $map[$pair[0]] -ne $pair[1]) {
      $flags += ('{0}={1}' -f $pair[0], $map[$pair[0]])
    }
  }
  return $flags
}

function Test-MultiseedDeviceLock {
  return [IO.Directory]::Exists((Resolve-MultiseedPath $DeviceLockRoot))
}

function Get-MultiseedLockOwnerPath {
  return (Join-Path (Resolve-MultiseedPath $DeviceLockRoot) 'owner.txt')
}

function Read-MultiseedDeviceLock {
  if (-not (Test-MultiseedDeviceLock)) { return $null }
  $ownerPath = Get-MultiseedLockOwnerPath
  $owner = if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    (Get-Content -LiteralPath $ownerPath -Raw)
  } else { '' }
  return [pscustomobject]@{ path = $DeviceLockRoot; owner = $owner }
}

function Enable-MultiseedDeviceAwake {
  $adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  & $adb shell input keyevent KEYCODE_WAKEUP | Out-Null
  & $adb shell wm dismiss-keyguard | Out-Null
  & $adb shell svc power stayon true | Out-Null
  & $adb shell dumpsys deviceidle disable | Out-Null
  Write-Host 'device_awake_and_idle_disabled=true'
}

function Acquire-MultiseedDeviceLock {
  if (Test-MultiseedDeviceLock) {
    $existing = Read-MultiseedDeviceLock
    throw ("DEVICE_LOCK_HELD path=$DeviceLockRoot owner=$($existing.owner -replace '\n',' | ')")
  }
  New-Item -ItemType Directory -Force -Path (Resolve-MultiseedPath $DeviceLockRoot) | Out-Null
  $ownerLines = @(
    ("owner=$lockOwner"),
    "pid=$PID",
    "timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))",
    "repo=$root",
    "experiment=$epochTag",
    "branch=$(git -C $root branch --show-current)"
  )
  Set-Content -LiteralPath (Get-MultiseedLockOwnerPath) -Value $ownerLines -Encoding utf8
  Write-Host "device_lock_acquired=$DeviceLockRoot"
}

function Release-MultiseedDeviceLock {
  if (-not (Test-MultiseedDeviceLock)) { return }
  $ownerPath = Get-MultiseedLockOwnerPath
  if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    $text = Get-Content -LiteralPath $ownerPath -Raw
    if ($text -notmatch $lockOwner -and $text -notmatch "pid=$PID") {
      throw "DEVICE_LOCK_NOT_OURS: refusing to release $DeviceLockRoot"
    }
  }
  Remove-Item -LiteralPath (Resolve-MultiseedPath $DeviceLockRoot) -Recurse -Force
  Write-Host "device_lock_released=$DeviceLockRoot"
}

function Test-MultiseedWeOwnDeviceLock {
  if (-not (Test-MultiseedDeviceLock)) { return $false }
  $lock = Read-MultiseedDeviceLock
  return ($lock.owner -match $lockOwner -or $lock.owner -match "pid=$PID")
}

function Wait-MultiseedDeviceLock {
  param([int]$PollSeconds = 45, [int]$MaxWaitSeconds = 14400)
  $deadline = [DateTime]::UtcNow.AddSeconds($MaxWaitSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    if (-not (Test-MultiseedDeviceLock)) {
      try { Acquire-MultiseedDeviceLock; return } catch { Write-Host $_.Exception.Message }
    } else {
      $lock = Read-MultiseedDeviceLock
      Write-Host "device_lock_busy now=$([DateTimeOffset]::UtcNow.ToString('o')) owner=$($lock.owner -replace '\n',' | ')"
    }
    Start-Sleep -Seconds $PollSeconds
  }
  throw 'DEVICE_LOCK_WAIT_TIMEOUT'
}

function Start-MultiseedDeviceGuard {
  if (Test-MultiseedWeOwnDeviceLock) {
    Write-Host 'device_lock_already_owned=continuing'
  } elseif (Test-MultiseedDeviceLock) {
    $lock = Read-MultiseedDeviceLock
    Write-Host "DEVICE_LOCK_PRESENT waiting owner=$($lock.owner -replace '\n',' | ')"
    Wait-MultiseedDeviceLock
  } else {
    Acquire-MultiseedDeviceLock
  }
  Enable-MultiseedDeviceAwake
}

function Invoke-MultiseedArm {
  param([int]$Seed, [string]$ArmName, [string]$ArmMode)
  $id = Get-MultiseedArmIdentity $ArmName
  $gate = $id.attention_gate
  $isG1 = $ArmName -eq 'G1'
  $armDir = Get-MultiseedArmDirectory $Seed $ArmName 'Report'
  $resultDir = Get-MultiseedArmDirectory $Seed $ArmName 'Results'
  New-MultiseedDirectory $armDir
  New-MultiseedDirectory $resultDir

  $armSteps = $(if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $Steps })
  $armInterval = $(if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $CheckpointInterval })
  $armEvalSteps = $(if ($ArmMode -eq 'Smoke') { @() } else { $EvalSteps })

  $resultFile = ('seed{0}-l19-v1024-t32-d64-f128-steps{1}-result.txt' -f $Seed, $armSteps)
  $existingResult = Join-Path $armDir $resultFile
  if ($ArmMode -eq 'All' -and (Test-MultiseedArmComplete $Seed $ArmName)) {
    Write-Host "arm_already_complete seed=$Seed arm=$ArmName dir=$armDir"
    return [pscustomobject]@{ seed = $Seed; arm = $ArmName; status = 'REUSED'; dir = $armDir }
  }

  Write-Host "=== MULTISEED ARM seed=$Seed arm=$ArmName mode=$ArmMode steps=$armSteps ==="
  Write-Host ("resolved muon_lr={0} aux_adam_lr={1} target_lr={2}" -f $lr15.muon, $lr15.aux_adam, $lr15.target)
  Write-MultiseedJson (Join-Path $resultDir 'arm-identity.json') ([ordered]@{
    seed = $Seed
    seed_role = (Get-MultiseedSeedRole $Seed)
    epoch_tag = $epochTag
    protocol = 'docs/g1-1p5x-multiseed-3000.md'
    base = $base
    arm = $id
    steps = $armSteps
    eval_steps = @($armEvalSteps)
    record_created_utc = [DateTimeOffset]::UtcNow.ToString('o')
  })

  $trainStart = [DateTimeOffset]::UtcNow
  $trainParams = @{
    QairtSdkRoot = $QairtSdkRoot
    ExpectedBuildId = $ExpectedBuildId
    Seed = $Seed
    Layers = $base.layers
    Steps = $armSteps
    Tokens = $base.tokens
    Vocabulary = $base.vocabulary
    Dimension = $base.dimension
    FeedForwardDimension = $base.feed_forward
    BatchSize = $base.batch
    LearningRate = $lr15.aux_adam
    LearningRateSchedule = $base.schedule
    DecayStartStep = $base.decay_start
    DecayEndStep = $base.decay_end
    ScheduleTotalSteps = $base.schedule_total
    TargetLearningRate = $lr15.target
    ExperimentFork = $true
    ParentLearningRate = $lr15.aux_adam
    Optimizer = $base.optimizer
    MuonBackend = $base.muon_backend
    AttentionGate = $gate
    MuonLearningRate = $lr15.muon
    MuonMomentum = $base.muon_momentum
    MuonNsSteps = $base.muon_ns_steps
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
  $trainExit = $LASTEXITCODE
  $trainEnd = [DateTimeOffset]::UtcNow
  if ($trainExit -ne 0) {
    $flags = @()
    if (Test-Path -LiteralPath $existingResult -PathType Leaf) {
      $flags = Get-MultiseedHardStopFlags $existingResult
    }
    Write-Host "MULTISEED_TRAINING_FAILED seed=$Seed arm=$ArmName exit=$trainExit flags=$($flags -join ';')"
    return [pscustomobject]@{
      seed = $Seed; arm = $ArmName; status = 'FAILED'; dir = $armDir
      hard_flags = ($flags -join '|'); train_start = $trainStart.ToString('o')
      train_end = $trainEnd.ToString('o')
    }
  }

  $hardFlags = @(Get-MultiseedHardStopFlags $existingResult | Where-Object { $_ })
  if ($hardFlags.Count -gt 0) {
    Write-Host "MULTISEED_HARD_STOP seed=$Seed arm=$ArmName flags=$($hardFlags -join ';')"
    return [pscustomobject]@{
      seed = $Seed; arm = $ArmName; status = 'UNSTABLE'; dir = $armDir
      hard_flags = ($hardFlags -join '|'); train_start = $trainStart.ToString('o')
      train_end = $trainEnd.ToString('o')
    }
  }
  return (Complete-MultiseedArm -Seed $Seed -ArmName $ArmName -Id $id -IsG1 $isG1 `
          -ArmDir $armDir -ResultDir $resultDir -ResultFile $resultFile `
          -ArmEvalSteps $armEvalSteps -TrainStart $trainStart -TrainEnd $trainEnd `
          -HardFlags $hardFlags)
}

function Complete-MultiseedArm {
  param([int]$Seed, [string]$ArmName, $Id, [bool]$IsG1, [string]$ArmDir,
        [string]$ResultDir, [string]$ResultFile, [int[]]$ArmEvalSteps,
        $TrainStart, $TrainEnd, [string[]]$HardFlags)
  $gate = $Id.attention_gate
  $resultPath = Join-Path $ArmDir $ResultFile

  foreach ($step in $ArmEvalSteps) {
    $checkpoint = Join-Path $ArmDir ('htp-seed{0}-l19-t32-d64-f128-step{1}.ckpt' -f $Seed, $step)
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
      throw "CHECKPOINT_MISSING seed=$Seed arm=$ArmName step=$step path=$checkpoint"
    }
    $evalDir = Join-Path $ArmDir ('eval256-step{0}' -f $step)
    $existingEval = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue |
      Select-Object -First 1
    if ($existingEval) { Write-Host "eval_reuse seed=$Seed arm=$ArmName step=$step"; continue }
    $evalParams = @{
      QairtSdkRoot = $QairtSdkRoot
      ExpectedBuildId = $ExpectedBuildId
      Seed = $Seed
      Layers = $base.layers
      Heads = $base.heads
      Tokens = $base.tokens
      Vocabulary = $base.vocabulary
      Dimension = $base.dimension
      FeedForwardDimension = $base.feed_forward
      AttentionGate = $gate
      CheckpointStep = $step
      CheckpointPath = $checkpoint
      TokenizerModelPath = $TokenizerModelPath
      ValidationChunks = $base.validation_chunks
      DevelopmentChunks = $base.development_chunks
      ReportRoot = $evalDir
      SkipBuild = $true
      SkipInstall = $true
    }
    & $evaluation @evalParams
    if ($LASTEXITCODE -ne 0) {
      throw "MULTISEED_EVAL_FAILED seed=$Seed arm=$ArmName step=$step"
    }
  }

  if ($IsG1) {
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
    foreach ($step in $ArmEvalSteps) {
      $ck = Join-Path $ArmDir ('htp-seed{0}-l19-t32-d64-f128-step{1}.ckpt' -f $Seed, $step)
      $diagPath = Join-Path $ArmDir ('gate-static-step{0}.txt' -f $step)
      if (Test-Path -LiteralPath $diagPath -PathType Leaf) { continue }
      & $diagExe $ck $TokenizerModelPath $ValidationCache $base.tokens $base.validation_chunks |
        Set-Content -LiteralPath $diagPath -Encoding utf8
      if ($LASTEXITCODE -ne 0) { throw "GATE_DIAGNOSTICS_FAILED seed=$Seed step=$step" }
    }
  }

  # Compact artifacts only: no checkpoints, no libraries, no device paths.
  Copy-Item -LiteralPath $resultPath -Destination (Join-Path $ResultDir $ResultFile) -Force
  $curve = Join-Path $ArmDir ('training-curve-t32-d64-f128-{0}.csv' -f $Steps)
  if (-not (Test-Path -LiteralPath $curve -PathType Leaf)) {
    $curve = Join-Path $ArmDir ('training-curve-{0}.csv' -f $Steps)
  }
  if (Test-Path -LiteralPath $curve -PathType Leaf) {
    Copy-Item -LiteralPath $curve -Destination (Join-Path $ResultDir (Split-Path -Leaf $curve)) -Force
  }
  $lrTelemetry = Join-Path $ArmDir 'learning-rate-telemetry.csv'
  if (Test-Path -LiteralPath $lrTelemetry -PathType Leaf) {
    Copy-Item -LiteralPath $lrTelemetry -Destination (Join-Path $ResultDir 'learning-rate-telemetry.csv') -Force
  }
  foreach ($step in $ArmEvalSteps) {
    $evalDir = Join-Path $ArmDir ('eval256-step{0}' -f $step)
    $htp = Get-ChildItem -Path $evalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue |
      Select-Object -First 1
    if (-not $htp) { throw "EVAL_REPORT_MISSING seed=$Seed arm=$ArmName step=$step" }
    Copy-Item -LiteralPath $htp.FullName `
      -Destination (Join-Path $ResultDir ("eval256-step$step-htp.txt")) -Force
    if ($IsG1) {
      $diagPath = Join-Path $ArmDir ('gate-static-step{0}.txt' -f $step)
      if (-not (Test-Path -LiteralPath $diagPath -PathType Leaf)) {
        throw "GATE_REPORT_MISSING seed=$Seed step=$step"
      }
      Copy-Item -LiteralPath $diagPath -Destination (Join-Path $ResultDir (Split-Path -Leaf $diagPath)) -Force
    }
  }

  $map = Get-MultiseedKeyValue $resultPath
  Write-Host "arm_complete seed=$Seed arm=$ArmName flags=$($HardFlags -join ';')"
  return [pscustomobject]@{
    seed = $Seed
    arm = $ArmName
    status = 'COMPLETED'
    dir = $ArmDir
    result_dir = $ResultDir
    hard_flags = ($HardFlags -join '|')
    train_start = $TrainStart.ToString('o')
    train_end = $TrainEnd.ToString('o')
    training_total_seconds = $(if ($map.Contains('training_total_seconds')) { [double]$map.training_total_seconds } else { $null })
    training_step_ms = $(if ($map.Contains('training_step_ms')) { [double]$map.training_step_ms } else { $null })
    initial_parameter_hash = $(if ($map.Contains('initial_parameter_hash')) { $map.initial_parameter_hash } else { $null })
    final_parameter_hash = $(if ($map.Contains('final_parameter_hash')) { $map.final_parameter_hash } else { $null })
  }
}

function Test-MultiseedArmComplete([int]$Seed, [string]$ArmName) {
  $dir = Get-MultiseedArmDirectory $Seed $ArmName 'Results'
  $result = Join-Path $dir ('seed{0}-l19-v1024-t32-d64-f128-steps{1}-result.txt' -f $Seed, $Steps)
  if (-not (Test-Path -LiteralPath $result -PathType Leaf)) { return $false }
  $map = Get-MultiseedKeyValue $result
  if (-not $map.Contains('status') -or $map.status -ne 'SUCCESS') { return $false }
  if (-not $map.Contains('completed_steps') -or [int]$map.completed_steps -ne $Steps) { return $false }
  foreach ($step in $EvalSteps) {
    $report = Join-Path $dir ('eval256-step{0}-htp.txt' -f $step)
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) { return $false }
  }
  return $true
}

function Invoke-MultiseedAnalysis {
  $tree = Resolve-MultiseedPath $ResultsRoot
  New-MultiseedDirectory $tree
  $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
  # The markdown tables are the commit-ready evidence summary; keep them next to
  # the CSVs so a result tree is self-describing without re-running the analysis.
  $output = & $python $analysisPy --tree $tree --out $tree 2>&1 | Out-String
  $analysisExit = $LASTEXITCODE
  $logPath = Join-Path $tree 'analysis.md'
  Write-MultiseedText $logPath $output
  Write-Host $output
  if ($analysisExit -ne 0) {
    throw "MULTISEED_ANALYSIS_FAILED exit=$analysisExit tree=$ResultsRoot log=$logPath"
  }
  Write-Host "analysis_written root=$ResultsRoot files=quality-split-level,gate-trajectory,run-health,run-identity,verdicts,analysis.md"
}

function Invoke-MultiseedSequence([string]$ArmMode) {
  Test-MultiseedProtocol
  $runOrder = @()
  $index = 0
  foreach ($seed in $Seeds) {
    # Alternate which arm starts first so no arm is always the first HTP session
    # of a seed; this is the same ordering control the committed grid used.
    $arms = @('Control','G1')
    if (($index % 2) -eq 1) { $arms = @('G1','Control') }
    foreach ($armName in $arms) {
      if ($Arm -ne 'All' -and $armName -ne $Arm) { continue }
      $runOrder += [pscustomobject]@{
        order = ($runOrder.Count + 1); seed = $seed; arm = $armName; mode = $ArmMode
        role = (Get-MultiseedSeedRole $seed)
      }
    }
    $index += 1
  }
  if ($runOrder.Count -eq 0) { throw 'EMPTY_RUN_PLAN' }
  Write-Host ("plan entries={0} seeds={1} steps={2} eval_steps={3}" -f `
    $runOrder.Count, ($Seeds -join '+'), $Steps, ($EvalSteps -join ','))
  foreach ($seed in $Seeds) {
    Write-Host ("seed_role seed={0} role={1}" -f $seed, (Get-MultiseedSeedRole $seed))
  }
  foreach ($entry in $runOrder) {
    Write-Host ("plan order={0} seed={1} arm={2} mode={3} role={4}" -f `
      $entry.order, $entry.seed, $entry.arm, $entry.mode, $entry.role)
    Write-Host ("plan dir seed={0} arm={1} report={2}\seed{0}\{3} results={4}\seed{0}\{3}" -f `
      $entry.seed, $entry.arm, $ReportRoot, $entry.arm.ToLowerInvariant(), $ResultsRoot)
  }
  if ($Mode -eq 'Plan') {
    Write-Host "plan_only=true device_touched=false files_written=false"
    Write-Host ("identity {0}" -f (($base.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' '))
    Write-Host ("lr_identity muon={0} aux_adam={1} target={2} fork=true" -f $lr15.muon, $lr15.aux_adam, $lr15.target)
    Write-Host "report_root=$ReportRoot results_root=$ResultsRoot"
    return @()
  }
  Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ReportRoot) 'run-order.json') $runOrder
  Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ResultsRoot) 'run-order.json') $runOrder
  $seedRegistry = Get-MultiseedSeedRegistry
  Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ReportRoot) 'seed-registry.json') $seedRegistry
  Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ResultsRoot) 'seed-registry.json') $seedRegistry
  Write-Host ("seed_registry preregistered={0} exploratory={1} allow_exploratory={2}" -f `
    ($preregisteredSeeds -join ','), (@($seedRegistry.exploratory_seeds) -join ','), $seedRegistry.allow_exploratory_seed)

  $outcomes = @()
  $adbPath = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
  try {
    # The lock is taken before any device query so a failed device resolution
    # cannot leave the guard half-held.
    Start-MultiseedDeviceGuard
    $devEndpoint = (Resolve-PhoneLmDevice -Adb $adbPath).Endpoint
    foreach ($entry in $runOrder) {
      try {
        $state = Get-PhoneLmThermalBatteryState -Adb $adbPath -Device $devEndpoint -Phase 'multiseed-pre'
        if ([int]$state.thermal_status -ge 4) {
          Write-Host "thermal_wait_before_arm status=$($state.thermal_status) seed=$($entry.seed) arm=$($entry.arm)"
          $waited = 0
          while ($waited -lt 1800) {
            Start-Sleep -Seconds 60
            $waited += 60
            $state = Get-PhoneLmThermalBatteryState -Adb $adbPath -Device $devEndpoint -Phase 'multiseed-pre-wait'
            if ([int]$state.thermal_status -le 2) { break }
          }
        }
      } catch {
        Write-Host "thermal_probe_soft_fail=$($_.Exception.Message)"
      }
      Enable-MultiseedDeviceAwake

      $outcome = Invoke-MultiseedArm -Seed $entry.seed -ArmName $entry.arm -ArmMode $entry.mode
      $outcomes += $outcome
      Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ReportRoot) 'multiseed-outcomes.json') $outcomes
      Write-MultiseedJson (Join-Path (Resolve-MultiseedPath $ResultsRoot) 'multiseed-outcomes.json') $outcomes
      if ($outcome.status -in @('FAILED','UNSTABLE')) {
        Write-Host "sequence_stopped seed=$($entry.seed) arm=$($entry.arm) status=$($outcome.status) flags=$($outcome.hard_flags)"
        break
      }
    }
  } finally {
    Release-MultiseedDeviceLock
  }
  return $outcomes
}

if ($SelfTest) {
  $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
  & $python $analysisPy --selftest
  exit $LASTEXITCODE
}

if ($Mode -eq 'Analyze') {
  Invoke-MultiseedAnalysis
  exit 0
}

$armMode = switch ($Mode) {
  'Smoke' { 'Smoke' }
  'Seed' { 'Run' }
  'All' { 'All' }
  default { 'Plan' }
}
$outcomes = Invoke-MultiseedSequence $armMode
if ($Mode -eq 'All') {
  $completed = @($outcomes | Where-Object { $_.status -in @('COMPLETED','REUSED') })
  if ($completed.Count -ne $outcomes.Count -or $outcomes.Count -eq 0) {
    Write-Host "analysis_skipped outcomes=$($outcomes.Count) completed=$($completed.Count)"
    exit 1
  }
  Invoke-MultiseedAnalysis
}
exit 0

