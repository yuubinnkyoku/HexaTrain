# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# G1 high-LR stability / time-to-bpb stress grid.
# Current headwise_g1_sigmoid vs ungated control only. No architecture variants.
# Modes: Plan | Smoke | Run | Grid | Analyze
[CmdletBinding()]
param(
  [ValidateSet('Control','G1')][string]$Arm = 'Control',
  [ValidateSet('1.0','1.25','1.5','2.0')][string]$Multiplier = '1.0',
  [ValidateSet('Plan','Smoke','Run','Grid','Analyze')][string]$Mode = 'Plan',
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$DevelopmentCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/development.bin',
  [string]$ReportRoot = 'build/g1-lr-stress',
  [string]$ResultsRoot = 'docs/results/g1-lr-stress-2026-09',
  [string]$DeviceLockRoot = 'D:\ghq\github.com\yuubinnkyoku\.hexatrain-device-lock',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  [int]$Steps = 500,
  [int]$CheckpointInterval = 100,
  [int]$SmokeSteps = 8,
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest,
  [switch]$ForceGrid
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
$training = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$evaluation = Join-Path $PSScriptRoot 'run_nicopedia_htp_eval.ps1'
$muonAudit = Join-Path $PSScriptRoot 'muon_row_geometry_audit.py'
$analysisPy = Join-Path $PSScriptRoot 'g1_lr_stress_analyze.py'

$epochTag = 'g1-lr-stress-2026-09'
$evalSteps = @(100, 200, 300, 400, 500)
if ($Steps -ne 500) {
  # Keep all arms on one shared eval grid. Reduced step counts only for smoke.
  if ($Mode -ne 'Smoke' -and $Mode -ne 'Plan') {
    if ($Steps -ne 100 -and $Steps -ne 250) { throw "STRESS_STEPS_UNSUPPORTED: $Steps" }
  }
  $evalSteps = @(100, 250, 500) | Where-Object { $_ -le $Steps }
  if ($Steps -eq 100) { $evalSteps = @(100) }
}

# --- formal LR identity (1.0x anchors) ---
$base = [ordered]@{
  aux_adam = 0.0022
  muon = 0.005
  target = 0.0001
  schedule = 'linear_decay'
  decay_start = 4000
  decay_end = 8000
  schedule_total = 8000
  seed = 1
  batch = 8
  vocabulary = 1024
  tokens = 32
  dimension = 64
  feed_forward = 128
  layers = 19
  heads = 2
  muon_momentum = 0.95
  muon_ns_steps = 5
  muon_nesterov = $true
  optimizer = 'Muon'
  muon_backend = 'HVX'
  validation_chunks = 256
  development_chunks = 256
  dataset_hash = 'fnv1a64:0c7b2826f5f26fea'
  dataset_order_identity = 'training_order_seed=20260806'
  validation_window_identity = 'val-first-256-chunks-v1024-bpe'
  development_window_identity = 'dev-first-256-chunks-v1024-bpe'
}

$capacity = [ordered]@{
  control_parameter_count = 758528
  g1_parameter_count = 760960
  parameter_delta = 2432
  muon_matrix_count = 114
  muon_parameter_count = 622592
}

function Get-StressMultiplierValue([string]$Value) {
  switch ($Value) {
    '1.0' { return 1.0 }
    '1.25' { return 1.25 }
    '1.5' { return 1.5 }
    '2.0' { return 2.0 }
    default { throw "MULTIPLIER_INVALID: $Value" }
  }
}

function Resolve-StressLr([double]$Mult) {
  # Exact decimal strings from the formal LR grid. Avoid binary float drift
  # in the identity fields handed to the device runner.
  switch ('{0:R}' -f $Mult) {
    '1' { return [ordered]@{ multiplier = 1.0; aux_adam_lr = 0.0022; muon_lr = 0.005; target_lr = 0.0001
            aux_adam_lr_s = '0.0022'; muon_lr_s = '0.005'; target_lr_s = '0.0001' } }
    '1.25' { return [ordered]@{ multiplier = 1.25; aux_adam_lr = 0.00275; muon_lr = 0.00625; target_lr = 0.000125
            aux_adam_lr_s = '0.00275'; muon_lr_s = '0.00625'; target_lr_s = '0.000125' } }
    '1.5' { return [ordered]@{ multiplier = 1.5; aux_adam_lr = 0.0033; muon_lr = 0.0075; target_lr = 0.00015
            aux_adam_lr_s = '0.0033'; muon_lr_s = '0.0075'; target_lr_s = '0.00015' } }
    '2' { return [ordered]@{ multiplier = 2.0; aux_adam_lr = 0.0044; muon_lr = 0.01; target_lr = 0.0002
            aux_adam_lr_s = '0.0044'; muon_lr_s = '0.01'; target_lr_s = '0.0002' } }
    default { throw "MULTIPLIER_RESOLVE_FAILED: $Mult" }
  }
}

function Get-StressArmIdentity([string]$ArmName, [double]$Mult) {
  $lr = Resolve-StressLr $Mult
  $isG1 = $ArmName -eq 'G1'
  return [ordered]@{
    arm = $ArmName
    multiplier = $Mult
    attention_gate = if ($isG1) { 'headwise_g1_sigmoid' } else { 'none' }
    parameter_count = if ($isG1) { $capacity.g1_parameter_count } else { $capacity.control_parameter_count }
    checkpoint_format = if ($isG1) { 'NPRTCKPTV5' } else { 'NPRTCKPTV4' }
    muon_lr = $lr.muon_lr
    aux_adam_lr = $lr.aux_adam_lr
    target_lr = $lr.target_lr
    muon_lr_s = $lr.muon_lr_s
    aux_adam_lr_s = $lr.aux_adam_lr_s
    target_lr_s = $lr.target_lr_s
    muon_lr_runtime = $lr.muon_lr_s
    aux_adam_lr_runtime = $lr.aux_adam_lr_s
    target_lr_runtime = $lr.target_lr_s
    seed = $base.seed
    batch = $base.batch
    schedule = $base.schedule
    decay_start_step = $base.decay_start
    decay_end_step = $base.decay_end
    schedule_total_steps = $base.schedule_total
    vocabulary = $base.vocabulary
    tokens = $base.tokens
    dimension = $base.dimension
    feed_forward_dimension = $base.feed_forward
    layers = $base.layers
    heads = $base.heads
    optimizer = $base.optimizer
    muon_backend = $base.muon_backend
    muon_momentum = $base.muon_momentum
    muon_ns_steps = $base.muon_ns_steps
    validation_chunks = $base.validation_chunks
    development_chunks = $base.development_chunks
    dataset_hash = $base.dataset_hash
    dataset_order_identity = $base.dataset_order_identity
    tokenizer_identity = 'byte-bpe-v1024'
    evaluation_protocol = 'val-first-256-dev-first-256-v1024-bpe'
    checkpoint_steps = $evalSteps
  }
}

function Get-StressArmKey([string]$ArmName, [string]$MultValue) {
  $tag = $MultValue.Replace('.', 'p')
  return ('lr{0}-{1}' -f $MultValue, $ArmName.ToLowerInvariant())
}

function Get-StressArmDirectory([string]$ArmName, [string]$MultValue) {
  $lrDir = 'lr' + $MultValue
  return Join-Path (Join-Path $ReportRoot $lrDir) $ArmName.ToLowerInvariant()
}

function Get-StressResultsArmDirectory([string]$ArmName, [string]$MultValue) {
  $lrDir = 'lr' + $MultValue
  return Join-Path (Join-Path $ResultsRoot $lrDir) $ArmName.ToLowerInvariant()
}

function Write-StressCsv([string]$Path, [object[]]$Rows, [string[]]$Columns) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  if (-not $Rows -or $Rows.Count -eq 0) {
    ($Columns -join ',') | Set-Content -LiteralPath $Path -Encoding utf8
    return
  }
  $Rows | Select-Object -Property $Columns | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

function Write-StressJson([string]$Path, $Payload) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  $Payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

# --- device lock ---
function Get-StressLockOwnerPath { return Join-Path $DeviceLockRoot 'owner.txt' }

function Test-StressDeviceLock {
  return (Test-Path -LiteralPath $DeviceLockRoot -PathType Container)
}

function Read-StressDeviceLock {
  if (-not (Test-StressDeviceLock)) { return $null }
  $ownerPath = Get-StressLockOwnerPath
  $owner = if (Test-Path -LiteralPath $ownerPath -PathType Leaf) { (Get-Content -LiteralPath $ownerPath -Raw) } else { '' }
  return [pscustomobject]@{ path = $DeviceLockRoot; owner = $owner }
}

function Enable-StressDeviceAwake {
  param([string]$Adb = '', [string]$Device = '')
  if (-not $Adb) { $Adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe' }
  # Endpoint-scoped on purpose: a raw `adb shell ...` resolves by transport
  # count, so a second (even offline) transport makes every command fail with
  # "more than one device/emulator" while the failure stays invisible.  The
  # commands stay best-effort, but each exit code is reported because sleep /
  # idle state is part of the thermal comparability of a long run.
  # The endpoint is resolved here (not at the call sites) because this helper
  # runs before the arm loop resolves it.  Redaction is the shared helper's job.
  if (-not $Device) {
    $Device = ''
    try { $Device = (Resolve-PhoneLmDevice -Adb $Adb).Endpoint } catch { $Device = '' }
  }
  $failures = @()
  foreach ($command in @(
      @('awake', @('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP')),
      @('dismiss_keyguard', @('shell', 'wm', 'dismiss-keyguard')),
      @('stayon', @('shell', 'svc', 'power', 'stayon', 'true')),
      @('deviceidle', @('shell', 'dumpsys', 'deviceidle', 'disable')))) {
    $adbArgs = if ($Device) { @('-s', $Device) + $command[1] } else { $command[1] }
    & $Adb @adbArgs *> $null
    if ($LASTEXITCODE -ne 0) { $failures += ('{0}=exit{1}' -f $command[0], $LASTEXITCODE) }
  }
  if ($failures.Count -eq 0) {
    Write-Host 'device_awake_and_idle_disabled=true'
  } else {
    Write-Host ('device_awake_and_idle_disabled=false failures={0} endpoint_resolved={1}' `
        -f ($failures -join ','), ([bool]$Device))
  }
}

function Acquire-StressDeviceLock {
  if (Test-StressDeviceLock) {
    $existing = Read-StressDeviceLock
    throw ("DEVICE_LOCK_HELD path=$DeviceLockRoot owner=$($existing.owner -replace '\n',' | ')")
  }
  New-Item -ItemType Directory -Force -Path $DeviceLockRoot | Out-Null
  $ownerLines = @(
    "owner=g1-lr-stress-agent",
    "pid=$PID",
    "timestamp=$([DateTimeOffset]::UtcNow.ToString('o'))",
    "repo=$root",
    "experiment=$epochTag",
    "branch=$(git -C $root branch --show-current)"
  )
  Set-Content -LiteralPath (Get-StressLockOwnerPath) -Value $ownerLines -Encoding utf8
  Write-Host "device_lock_acquired=$DeviceLockRoot"
}

function Release-StressDeviceLock {
  if (-not (Test-StressDeviceLock)) { return }
  $ownerPath = Get-StressLockOwnerPath
  if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
    $text = Get-Content -LiteralPath $ownerPath -Raw
    if ($text -notmatch 'g1-lr-stress-agent' -and $text -notmatch "pid=$PID") {
      throw "DEVICE_LOCK_NOT_OURS: refusing to release $DeviceLockRoot"
    }
  }
  Remove-Item -LiteralPath $DeviceLockRoot -Recurse -Force
  Write-Host "device_lock_released=$DeviceLockRoot"
}

function Wait-StressDeviceLock {
  param([int]$PollSeconds = 45, [int]$MaxWaitSeconds = 14400)
  $deadline = [DateTime]::UtcNow.AddSeconds($MaxWaitSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    if (-not (Test-StressDeviceLock)) {
      try { Acquire-StressDeviceLock; return } catch { Write-Host $_.Exception.Message }
    } else {
      $lock = Read-StressDeviceLock
      Write-Host "device_lock_busy now=$([DateTimeOffset]::UtcNow.ToString('o')) owner=$($lock.owner -replace '\n',' | ')"
    }
    Start-Sleep -Seconds $PollSeconds
  }
  throw 'DEVICE_LOCK_WAIT_TIMEOUT'
}

# --- identity / health helpers ---
function Get-PhoneLmFileSha256([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "ARTIFACT_MISSING: $Path" }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-StressKeyValue([string]$Path) {
  return Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $Path -Raw)
}

function Test-StressArmComplete([string]$ArmName, [string]$MultValue) {
  $dir = Get-StressArmDirectory $ArmName $MultValue
  $result = Join-Path $dir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $Steps)
  if (-not (Test-Path -LiteralPath $result -PathType Leaf)) { return $false }
  $map = Read-StressKeyValue $result
  if (-not $map.Contains('status') -or $map.status -ne 'SUCCESS') { return $false }
  if (-not $map.Contains('completed_steps') -or [int]$map.completed_steps -ne $Steps) { return $false }
  foreach ($step in $evalSteps) {
    $ck = Join-Path $dir ('htp-seed1-l19-t32-d64-f128-step{0}.ckpt' -f $step)
    if (-not (Test-Path -LiteralPath $ck -PathType Leaf)) { return $false }
  }
  return $true
}

function Get-StressHardStopFlags([string]$ResultPath) {
  $map = Read-StressKeyValue $ResultPath
  $flags = @()
  if ($map.Contains('status') -and $map.status -ne 'SUCCESS') { $flags += 'status=' + $map.status }
  foreach ($pair in @(
      @('all_steps_finite','true'),
      @('final_finite','true'),
      @('output_tensors_finite','true'),
      @('qnn_return_code_success','true'),
      @('cpu_fallback','false'),
      @('fallback','false'),
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
      @('hvx_nonfinite_count','0'))) {
    if ($map.Contains($pair[0]) -and [int]$map[$pair[0]] -ne 0) {
      $flags += ('{0}={1}' -f $pair[0], $map[$pair[0]])
    }
  }
  return ,$flags
}

function Test-StressLossExplosion([string]$CurvePath) {
  # Conservative: only treat multi-point catastrophic growth with no recovery
  # as explosion. A single spike is a soft flag, not a hard stop.
  if (-not (Test-Path -LiteralPath $CurvePath -PathType Leaf)) { return $false }
  $rows = @(Import-Csv -LiteralPath $CurvePath -ErrorAction SilentlyContinue)
  if ($rows.Count -lt 4) { return $false }
  $losses = @($rows | ForEach-Object { [double]$_.loss })
  $window = [Math]::Min(10, [Math]::Max(3, [int][Math]::Floor($losses.Count / 5)))
  $median = ($losses[0..($window-1)] | Sort-Object)[[int][Math]::Floor($window/2)]
  if ($median -le 0) { return $false }
  $blow = 0
  $maxRatio = 0.0
  foreach ($loss in $losses) {
    $ratio = $loss / $median
    if ($ratio -gt $maxRatio) { $maxRatio = $ratio }
    if ($ratio -gt 20.0) { $blow++ } else { $blow = 0 }
    if ($blow -ge 5) { return $true }
  }
  return $false
}

function Get-StressSoftFlags([string]$ArmName, [string]$MultValue, [string]$ResultPath) {
  $flags = @()
  if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
    $flags += 'result_missing'
    return $flags
  }
  $map = Read-StressKeyValue $ResultPath
  if ($map.Contains('nan_detected') -and $map.nan_detected -eq 'true') { $flags += 'nan_detected' }
  if ($map.Contains('inf_detected') -and $map.inf_detected -eq 'true') { $flags += 'inf_detected' }
  if ($map.Contains('android_thermal_status_after')) {
    $thermal = [int]$map.android_thermal_status_after
    if ($thermal -ge 3) { $flags += "thermal_status_after=$thermal" }
  }
  if ($map.Contains('android_thermal_status_before')) {
    $thermal = [int]$map.android_thermal_status_before
    if ($thermal -ge 3) { $flags += "thermal_status_before=$thermal" }
  }
  if ($map.Contains('focus_takeover_count') -and [int]$map.focus_takeover_count -ne 0) {
    $flags += 'focus_takeover_count=' + $map.focus_takeover_count
  }
  $curve = Join-Path (Get-StressArmDirectory $ArmName $MultValue) ('training-curve-t32-d64-f128-{0}.csv' -f $Steps)
  if (-not (Test-Path -LiteralPath $curve -PathType Leaf)) {
    $curve = Join-Path (Get-StressArmDirectory $ArmName $MultValue) ('training-curve-{0}.csv' -f $Steps)
  }
  if (Test-StressLossExplosion $curve) { $flags += 'loss_explosion_detected' }
  if ($ArmName -eq 'G1' -and $map.Contains('gate_training_trajectory_aggregate')) {
    foreach ($key in $map.Keys) {
      if ($key -like 'gate_training_trajectory_*_mean') {
        $mean = [double]$map[$key]
        if ($mean -lt 0.05 -or $mean -gt 0.95) { $flags += "gate_traj_$key=$mean"; break }
      }
    }
  }
  return ,$flags
}

function Classify-StressStability($HardFlags, $SoftFlags) {
  $hard = @($HardFlags | Where-Object { $_ })
  $soft = @($SoftFlags | Where-Object { $_ })
  if ($hard.Count -gt 0) { return 'UNSTABLE' }
  $strong = @($soft | Where-Object { $_ -match 'loss_explosion|nan_detected|inf_detected|thermal_status_after=[45]' })
  if ($strong.Count -gt 0) { return 'MARGINAL' }
  if ($soft.Count -gt 0) { return 'MARGINAL' }
  return 'STABLE'
}

# --- Plan / SelfTest ---
function Write-StressResolvedGrid {
  $rows = @()
  foreach ($multValue in @('1.0','1.25','1.5','2.0')) {
    $mult = Get-StressMultiplierValue $multValue
    foreach ($armName in @('Control','G1')) {
      $id = Get-StressArmIdentity $armName $mult
      $rows += [pscustomobject][ordered]@{
        arm = $id.arm
        multiplier = $id.multiplier
        attention_gate = $id.attention_gate
        parameter_count = $id.parameter_count
        checkpoint_format = $id.checkpoint_format
        muon_lr = ('{0:R}' -f $id.muon_lr)
        aux_adam_lr = ('{0:R}' -f $id.aux_adam_lr)
        target_lr = ('{0:R}' -f $id.target_lr)
        muon_lr_runtime = $id.muon_lr_runtime
        aux_adam_lr_runtime = $id.aux_adam_lr_runtime
        target_lr_runtime = $id.target_lr_runtime
        seed = $id.seed
        batch = $id.batch
        schedule = $id.schedule
        decay_start_step = $id.decay_start_step
        decay_end_step = $id.decay_end_step
        schedule_total_steps = $id.schedule_total_steps
        vocabulary = $id.vocabulary
        tokens = $id.tokens
        dimension = $id.dimension
        feed_forward_dimension = $id.feed_forward_dimension
        layers = $id.layers
        heads = $id.heads
        optimizer = $id.optimizer
        muon_backend = $id.muon_backend
        muon_momentum = $id.muon_momentum
        muon_ns_steps = $id.muon_ns_steps
        validation_chunks = $id.validation_chunks
        development_chunks = $id.development_chunks
        dataset_hash = $id.dataset_hash
        dataset_order_identity = $id.dataset_order_identity
        tokenizer_identity = $id.tokenizer_identity
        evaluation_protocol = $id.evaluation_protocol
        checkpoint_steps = ($id.checkpoint_steps -join '|')
      }
    }
  }
  $path = Join-Path $ResultsRoot 'resolved-grid.csv'
  Write-StressCsv $path $rows @($rows[0].PSObject.Properties.Name)
  Write-StressCsv (Join-Path $ReportRoot 'resolved-grid.csv') $rows @($rows[0].PSObject.Properties.Name)
  return $rows
}

function Test-StressSelfTest {
  $grid = Write-StressResolvedGrid
  if ($grid.Count -ne 8) { throw 'SELFTEST_GRID_COUNT' }
  foreach ($row in $grid) {
    $expectedParams = if ($row.arm -eq 'G1') { 760960 } else { 758528 }
    $expectedFormat = if ($row.arm -eq 'G1') { 'NPRTCKPTV5' } else { 'NPRTCKPTV4' }
    $expectedGate = if ($row.arm -eq 'G1') { 'headwise_g1_sigmoid' } else { 'none' }
    if ([int]$row.parameter_count -ne $expectedParams -or $row.checkpoint_format -ne $expectedFormat -or $row.attention_gate -ne $expectedGate) {
      throw "SELFTEST_IDENTITY arm=$($row.arm)"
    }
  }
  $c10 = $grid | Where-Object { $_.arm -eq 'Control' -and $_.multiplier -eq 1.0 } | Select-Object -First 1
  $g10 = $grid | Where-Object { $_.arm -eq 'G1' -and $_.multiplier -eq 1.0 } | Select-Object -First 1
  if ([double]$c10.muon_lr -ne 0.005 -or [double]$c10.aux_adam_lr -ne 0.0022 -or [double]$c10.target_lr -ne 0.0001) { throw 'SELFTEST_LR_1X' }
  if ([double]$g10.muon_lr -ne 0.005) { throw 'SELFTEST_G1_LR_1X' }
  $c20 = $grid | Where-Object { $_.arm -eq 'Control' -and $_.multiplier -eq 2.0 } | Select-Object -First 1
  if ([math]::Abs(([double]$c20.muon_lr) - 0.01) -gt 1e-12 -or [math]::Abs(([double]$c20.aux_adam_lr) - 0.0044) -gt 1e-12) { throw 'SELFTEST_LR_2X' }
  if ((Get-StressMultiplierValue '1.25') -ne 1.25) { throw 'SELFTEST_MULT' }
  Write-Host 'run_headwise_g1_lr_stress_self_test=PASS'
  Write-Host "resolved_grid=$($grid.Count)"
  foreach ($row in $grid) {
    Write-Host ("arm={0} mult={1} gate={2} params={3} muon_lr={4} aux_lr={5} target_lr={6} fmt={7}" -f `
      $row.arm, $row.multiplier, $row.attention_gate, $row.parameter_count, $row.muon_lr_runtime, $row.aux_adam_lr_runtime, $row.target_lr_runtime, $row.checkpoint_format)
  }
}

# --- single-arm execution ---
function Invoke-StressArm {
  param(
    [Parameter(Mandatory=$true)][string]$ArmName,
    [Parameter(Mandatory=$true)][string]$MultValue,
    [ValidateSet('Smoke','Run')][string]$ArmMode = 'Run'
  )
  $mult = Get-StressMultiplierValue $MultValue
  $id = Get-StressArmIdentity $ArmName $mult
  $isG1 = $ArmName -eq 'G1'
  $gate = $id.attention_gate
  $armDir = Get-StressArmDirectory $ArmName $MultValue
  $resultDir = Get-StressResultsArmDirectory $ArmName $MultValue
  New-Item -ItemType Directory -Force -Path $armDir | Out-Null
  New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

  $armSteps = if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $Steps }
  $armInterval = if ($ArmMode -eq 'Smoke') { $SmokeSteps } else { $CheckpointInterval }
  $armEvalSteps = if ($ArmMode -eq 'Smoke') { @() } else { $evalSteps }

  $existingResult = Join-Path $armDir ('seed1-l19-v1024-t32-d64-f128-steps{0}-result.txt' -f $armSteps)
  if ($ArmMode -eq 'Run' -and (Test-StressArmComplete $ArmName $MultValue)) {
    Write-Host "arm_already_complete arm=$ArmName multiplier=$MultValue dir=$armDir"
    return [pscustomobject]@{ arm = $ArmName; multiplier = $MultValue; status = 'REUSED'; dir = $armDir }
  }

  Write-Host "=== STRESS ARM arm=$ArmName multiplier=$MultValue mode=$ArmMode steps=$armSteps ==="
  Write-Host ("resolved muon_lr={0} aux_adam_lr={1} target_lr={2}" -f $id.muon_lr_runtime, $id.aux_adam_lr_runtime, $id.target_lr_runtime)

  $trainStart = [DateTimeOffset]::UtcNow
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
    LearningRate = $id.aux_adam_lr_s
    LearningRateSchedule = 'linear_decay'
    DecayStartStep = 4000
    DecayEndStep = 8000
    ScheduleTotalSteps = 8000
    TargetLearningRate = $id.target_lr_s
    ExperimentFork = $true
    ParentLearningRate = $id.aux_adam_lr_s
    Optimizer = 'Muon'
    MuonBackend = 'HVX'
    AttentionGate = $gate
    MuonLearningRate = $id.muon_lr_s
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
  $trainExit = $LASTEXITCODE
  $trainEnd = [DateTimeOffset]::UtcNow
  if ($trainExit -ne 0) {
    $flags = @()
    if (Test-Path -LiteralPath $existingResult -PathType Leaf) {
      $flags = Get-StressHardStopFlags $existingResult
    }
    Write-Host "STRESS_TRAINING_FAILED arm=$ArmName multiplier=$MultValue exit=$trainExit flags=$($flags -join ';')"
    return [pscustomobject]@{
      arm = $ArmName; multiplier = $MultValue; status = 'FAILED'
      dir = $armDir; hard_flags = ($flags -join '|'); train_start = $trainStart.ToString('o')
      train_end = $trainEnd.ToString('o')
    }
  }

  $resultMap = Read-StressKeyValue $existingResult
  $hardFlags = @(Get-StressHardStopFlags $existingResult | Where-Object { $_ })
  if ($hardFlags.Count -gt 0) {
    Write-Host "STRESS_HARD_STOP arm=$ArmName multiplier=$MultValue flags=$($hardFlags -join ';')"
    return [pscustomobject]@{
      arm = $ArmName; multiplier = $MultValue; status = 'UNSTABLE'
      dir = $armDir; hard_flags = ($hardFlags -join '|'); train_start = $trainStart.ToString('o')
      train_end = $trainEnd.ToString('o')
    }
  }

  # Canonical evaluations at shared points.
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
    if ($LASTEXITCODE -ne 0) { throw "STRESS_EVAL_FAILED arm=$ArmName multiplier=$MultValue step=$step" }
  }

  # Checkpoint-static gate diagnostics for G1.
  if ($isG1) {
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
      if ($LASTEXITCODE -ne 0) { throw "GATE_DIAGNOSTICS_FAILED arm=$ArmName step=$step" }
    }
  }

  # Copy compact artifacts into results tree (no checkpoints/binaries).
  $resultFile = Split-Path -Leaf $existingResult
  Copy-Item -LiteralPath $existingResult -Destination (Join-Path $resultDir $resultFile) -Force
  $curveSrc = Join-Path $armDir ('training-curve-t32-d64-f128-{0}.csv' -f $armSteps)
  if (-not (Test-Path -LiteralPath $curveSrc -PathType Leaf)) {
    $curveSrc = Join-Path $armDir ('training-curve-{0}.csv' -f $armSteps)
  }
  if (Test-Path -LiteralPath $curveSrc -PathType Leaf) {
    Copy-Item -LiteralPath $curveSrc -Destination (Join-Path $resultDir (Split-Path -Leaf $curveSrc)) -Force
  }
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
    if ($isG1) {
      $diagPath = Join-Path $armDir ('gate-static-step{0}.txt' -f $step)
      if (Test-Path -LiteralPath $diagPath -PathType Leaf) {
        Copy-Item -LiteralPath $diagPath -Destination (Join-Path $resultDir (Split-Path -Leaf $diagPath)) -Force
      }
    }
  }

  $softFlags = Get-StressSoftFlags $ArmName $MultValue $existingResult
  $stability = Classify-StressStability $hardFlags $softFlags
  Write-Host "arm_complete arm=$ArmName multiplier=$MultValue stability=$stability soft=$($softFlags -join ';')"
  return [pscustomobject]@{
    arm = $ArmName
    multiplier = $MultValue
    status = 'COMPLETED'
    stability = $stability
    dir = $armDir
    result_dir = $resultDir
    hard_flags = ($hardFlags -join '|')
    soft_flags = ($softFlags -join '|')
    train_start = $trainStart.ToString('o')
    train_end = $trainEnd.ToString('o')
    training_total_seconds = $(if ($resultMap.Contains('training_total_seconds')) { [double]$resultMap.training_total_seconds } else { $null })
    training_step_ms = $(if ($resultMap.Contains('training_step_ms')) { [double]$resultMap.training_step_ms } else { $null })
    run_target_utf8_bytes_seen = $(if ($resultMap.Contains('run_target_utf8_bytes_seen')) { [int64]$resultMap.run_target_utf8_bytes_seen } else { $null })
    initial_parameter_hash = $(if ($resultMap.Contains('initial_parameter_hash')) { $resultMap.initial_parameter_hash } else { $null })
    final_parameter_hash = $(if ($resultMap.Contains('final_parameter_hash')) { $resultMap.final_parameter_hash } else { $null })
  }
}

# --- grid ---
function Test-StressWeOwnDeviceLock {
  if (-not (Test-StressDeviceLock)) { return $false }
  $lock = Read-StressDeviceLock
  return ($lock.owner -match 'g1-lr-stress-agent' -or $lock.owner -match "pid=$PID")
}

function Invoke-StressGrid {
  $plan = @(
    @{ mult = '1.0'; arms = @('Control','G1') },
    @{ mult = '1.25'; arms = @('G1','Control') },
    @{ mult = '1.5'; arms = @('Control','G1') },
    @{ mult = '2.0'; arms = @('G1','Control') }
  )
  $runOrder = @()
  foreach ($pair in $plan) {
    foreach ($armName in $pair.arms) {
      $runOrder += [pscustomobject]@{ order = ($runOrder.Count + 1); multiplier = $pair.mult; arm = $armName }
    }
  }
  Write-StressJson (Join-Path $ReportRoot 'run-order.json') $runOrder
  Write-StressJson (Join-Path $ResultsRoot 'run-order.json') $runOrder

  if (Test-StressWeOwnDeviceLock) {
    Write-Host 'device_lock_already_owned=continuing'
  } elseif (Test-StressDeviceLock) {
    if (-not $ForceGrid) {
      $lock = Read-StressDeviceLock
      Write-Host "DEVICE_LOCK_PRESENT waiting owner=$($lock.owner -replace '\n',' | ')"
      Wait-StressDeviceLock
    } else {
      throw "DEVICE_LOCK_PRESENT_FORCE_REFUSED owner=$((Read-StressDeviceLock).owner -replace '\n',' | ')"
    }
  } else {
    Acquire-StressDeviceLock
  }
  Enable-StressDeviceAwake

  $outcomes = @()
  $gridFailed = $false
  try {
    $adbPath = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
    $devInfo = Resolve-PhoneLmDevice -Adb $adbPath
    $devEndpoint = $devInfo.Endpoint
    foreach ($item in $runOrder) {
      # Thermal comparability: wait if previous arm left the device hot.
      try {
        $state = Get-PhoneLmThermalBatteryState -Adb $adbPath -Device $devEndpoint -Phase 'grid-pre'
        if ([int]$state.thermal_status -ge 4) {
          Write-Host "thermal_wait_before_arm status=$($state.thermal_status) arm=$($item.arm) multiplier=$($item.multiplier)"
          $waited = 0
          while ($waited -lt 1800) {
            Start-Sleep -Seconds 60
            $waited += 60
            $state = Get-PhoneLmThermalBatteryState -Adb $adbPath -Device $devEndpoint -Phase 'grid-pre-wait'
            if ([int]$state.thermal_status -le 2) { break }
          }
        }
      } catch {
        Write-Host "thermal_probe_soft_fail=$($_.Exception.Message)"
      }
      Enable-StressDeviceAwake

      $outcome = Invoke-StressArm -ArmName $item.arm -MultValue $item.multiplier -ArmMode Run
      $outcomes += $outcome
      Write-StressJson (Join-Path $ReportRoot 'grid-outcomes.json') $outcomes
      Write-StressJson (Join-Path $ResultsRoot 'grid-outcomes.json') $outcomes
    }
  } catch {
    $gridFailed = $true
    Write-Host "GRID_EXCEPTION: $($_.Exception.Message)"
    throw
  } finally {
    # Keep the lock if the grid aborted so a retry can continue without
    # re-acquire races. Release only after a complete pass.
    if (-not $gridFailed) {
      Release-StressDeviceLock
    } else {
      Write-Host 'device_lock_retained_after_grid_failure=true'
    }
  }
  return $outcomes
}

# --- analyze ---
function Invoke-StressAnalyze {
  if (Test-Path -LiteralPath $analysisPy -PathType Leaf) {
    $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
    & $python $analysisPy `
      --report-root $ReportRoot `
      --results-root $ResultsRoot `
      --steps $Steps `
      --eval-steps ($evalSteps -join ',')
    if ($LASTEXITCODE -ne 0) { throw 'STRESS_ANALYZE_FAILED' }
    return
  }
  Write-Host 'ANALYZE_PY_MISSING'
}

# --- entry ---
if ($SelfTest -or $Mode -eq 'Plan') {
  Test-StressSelfTest
  if ($Mode -eq 'Plan') {
    $grid = Write-StressResolvedGrid
    Write-Host 'plan_only_no_device_work'
    Write-Host "results_root=$ResultsRoot"
    Write-Host "report_root=$ReportRoot"
    Write-Host "eval_steps=$($evalSteps -join ',')"
    Write-Host "run_order=1.00x:Control->G1;1.25x:G1->Control;1.50x:Control->G1;2.00x:G1->Control"
  }
  exit 0
}

if ($Mode -eq 'Smoke') {
  # Identity smoke at 1.0x for the requested arm. Keep the lock so an
  # immediately following Grid can continue without re-acquire races.
  if (-not (Test-StressWeOwnDeviceLock)) {
    if (Test-StressDeviceLock) {
      $lock = Read-StressDeviceLock
      throw "DEVICE_LOCK_HELD_FOR_SMOKE owner=$($lock.owner -replace '\n',' | ')"
    }
    Acquire-StressDeviceLock
  }
  Enable-StressDeviceAwake
  $outcome = Invoke-StressArm -ArmName $Arm -MultValue '1.0' -ArmMode Smoke
  Write-StressJson (Join-Path $ReportRoot ('smoke-{0}.json' -f $Arm.ToLowerInvariant())) $outcome
  if ($outcome.status -ne 'COMPLETED' -and $outcome.status -ne 'REUSED') { throw 'SMOKE_FAILED' }
  Write-Host "PASS STRESS_SMOKE arm=$Arm"
  exit 0
}

if ($Mode -eq 'Run') {
  if (-not (Test-StressDeviceLock)) { Acquire-StressDeviceLock }
  $outcome = Invoke-StressArm -ArmName $Arm -MultValue $Multiplier -ArmMode Run
  Write-StressJson (Join-Path $ReportRoot ('arm-{0}.json' -f (Get-StressArmKey $Arm $Multiplier))) $outcome
  Write-Host "PASS STRESS_RUN arm=$Arm multiplier=$Multiplier status=$($outcome.status)"
  exit 0
}

if ($Mode -eq 'Grid') {
  $outcomes = Invoke-StressGrid
  Invoke-StressAnalyze
  Write-Host 'PASS STRESS_GRID'
  Write-Host ("outcomes={0}" -f ($outcomes | ForEach-Object { "$($_.arm)@$($_.multiplier)=$($_.status)" }) -join ' ')
  exit 0
}

if ($Mode -eq 'Analyze') {
  Invoke-StressAnalyze
  Write-Host 'PASS STRESS_ANALYZE'
  exit 0
}

throw "MODE_INVALID: $Mode"
