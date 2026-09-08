# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Controlled LR-only HPO runner for the V1024/T32/D64/FFN128/L19/H2 seed-1
# Nicopedia experiment.  All ledger/checkpoint/evaluation material remains
# private under build/; this script never exports raw evidence.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Run',
  [ValidateSet('v1','v1b')][string]$Variant = 'v1',
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/lr-v1',
  [string]$AnchorLedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/lr-v1',
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$SchemaVersion = 1
$TrialPrefix = "hpo-lr-$Variant"
$LrSpace = if ($Variant -eq 'v1b') { @('0.00105','0.00075') } else { @('0.0015','0.0022','0.0030','0.0042','0.0060') }
$TrialOrder = if ($Variant -eq 'v1b') { @('0.00105','0.00075') } else { @('0.0042','0.0022','0.0060','0.0015','0.0030') }
$AnchorLrs = if ($Variant -eq 'v1b') { @('0.0015','0.0022','0.0030','0.0042') } else { @() }
$Root = Split-Path -Parent $PSScriptRoot
$TrainingRunner = Join-Path $Root 'scripts/run_nicopedia_htp_training.ps1'
$EvalRunner = Join-Path $Root 'scripts/run_nicopedia_htp_eval.ps1'
$TrainingDataRoot = Join-Path $Root 'build/private-data/nicopedia-real-text-bpe-v1024'
$TokenizerPath = Join-Path $TrainingDataRoot 'tokenizer/byte-bpe-v1024.model'
$TrainCache = Join-Path $TrainingDataRoot 'caches/train_pilot.bin'
$EvalCacheRoot = Join-Path $TrainingDataRoot 'caches'
$DefaultSourceReportRoot = Join-Path $Root 'build/reports/nicopedia-htp-training-v1024'

$Fixed = [ordered]@{
  vocabulary = 1024; tokens = 32; dimension = 64; feed_forward_dimension = 128
  layers = 19; heads = 2; parameter_count = 758528
  tokenizer_kind = 'byte_bpe'
  tokenizer_hash = 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798'
  batch_size = 8; seed = 1
  optimizer = 'ADAM'; beta1 = 0.9; beta2 = 0.999; epsilon = '1e-8'
  gradient_clip = 'disabled'; weight_decay = 0; lr_schedule = 'constant'
  warmup_steps = 0; decay_steps = 0
  dataset_identity = 'nicopedia-real-text-bpe-v1024/train_pilot.bin'
  dataset_cache_content_hash = 'fnv1a64:0c7b2826f5f26fea'
  training_order_identity = 'fixed canonical train_pilot round-robin order'
  training_order_hash = 'fnv1a64:0e2e15196d851431'
  qnn_backend = 'HTP'; planned_max_steps = 8000
  planned_target_tokens = 2048000; expected_original_utf8_bytes = 5491256
  expected_chunks = 46616; expected_articles = 1949
}

function Resolve-UnderBuild([string]$Path,[string]$Label) {
  $resolved = if ([IO.Path]::IsPathRooted($Path)) { [IO.Path]::GetFullPath($Path) } else { [IO.Path]::GetFullPath((Join-Path $Root $Path)) }
  $allowed = [IO.Path]::GetFullPath((Join-Path $Root 'build')) + [IO.Path]::DirectorySeparatorChar
  if (-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { throw "$Label must resolve below build" }
  return $resolved
}

$Ledger = Resolve-UnderBuild $LedgerRoot 'LedgerRoot'
$AnchorLedger = Resolve-UnderBuild $AnchorLedgerRoot 'AnchorLedgerRoot'
$TrialsRoot = Join-Path $Ledger 'trials'
$EventsPath = Join-Path $Ledger 'trials.jsonl'
$SummaryPath = Join-Path $Ledger 'summary.csv'

function Get-TrialId([string]$Lr) { return "$TrialPrefix-lr$($Lr.Replace('.','p'))-seed1" }
function Get-AnchorTrialId([string]$Lr) { return "hpo-lr-v1-lr$($Lr.Replace('.','p'))-seed1" }
function Get-AnchorTrialDir([string]$Lr) { return Join-Path (Join-Path $AnchorLedger 'trials') (Get-AnchorTrialId $Lr) }
function Get-TrialDir([string]$Lr) { return Join-Path $TrialsRoot (Get-TrialId $Lr) }
function Get-NowUtc { return [DateTime]::UtcNow.ToString('o') }
function Write-Event([hashtable]$Fields) {
  $row = [ordered]@{ utc = Get-NowUtc }
  foreach ($key in $Fields.Keys) { $row[$key] = $Fields[$key] }
  $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsPath -Encoding utf8
}
function Read-KeyValue([string]$Path) {
  # Use a real Hashtable because downstream identity checks intentionally use
  # ContainsKey; an OrderedDictionary exposes Contains but not ContainsKey.
  $map = @{}
  foreach ($line in (Get-Content -LiteralPath $Path)) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line -notmatch '=') { continue }
    $pair = $line -split '=',2
    if ($map.Contains($pair[0]) -and $map[$pair[0]] -ne $pair[1].Trim()) { throw "duplicate report key conflict: $($pair[0])" }
    $map[$pair[0]] = $pair[1].Trim()
  }
  return $map
}
function Get-Balanced([hashtable]$Map) {
  if (-not $Map.ContainsKey('validation_bits_per_utf8_byte') -or -not $Map.ContainsKey('development_bits_per_utf8_byte')) { return $null }
  return ([double]$Map.validation_bits_per_utf8_byte + [double]$Map.development_bits_per_utf8_byte) / 2.0
}
function Get-CheckpointName([int]$Step) { return "htp-seed1-l19-t32-d64-f128-step$Step.ckpt" }
function Get-EvalStem([int]$Step,[int]$V,[int]$D) { return "seed1-l19-t32-d64-f128-step$Step-v$V-d$D" }

function New-Manifest([string]$Lr,[string]$Status='PENDING') {
  $git = (& git -C $Root rev-parse HEAD).Trim()
  $m = [ordered]@{ trial_id = Get-TrialId $Lr; hpo_schema_version = $SchemaVersion; git_revision = $git; status = $Status }
  foreach ($key in $Fixed.Keys) { $m[$key] = $Fixed[$key] }
  $m.learning_rate = $Lr
  $m.checkpoint_format = 'NPRTCKPTV3'
  $m.completed_steps = 0
  $m.checkpoint_path = ''
  $m.checkpoint_parameter_hash = ''
  $m.reused = $false
  $m.reuse_source = ''
  $m.smoke_health = 'PENDING'
  $m.created_utc = Get-NowUtc
  return $m
}
function Save-Manifest([string]$Lr,[hashtable]$Manifest) {
  $dir = Get-TrialDir $Lr; [IO.Directory]::CreateDirectory($dir) | Out-Null
  $Manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $dir 'manifest.json') -Encoding utf8
}
function Load-Manifest([string]$Lr) {
  $path = Join-Path (Get-TrialDir $Lr) 'manifest.json'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}
function Assert-Identity([hashtable]$Map,[string]$Lr,[int]$Step,[string]$Kind) {
  foreach ($key in @('model_dimension','feed_forward_dimension','layers','heads','vocabulary_size','context_tokens','checkpoint_step','checkpoint_parameter_elements','checkpoint_parameter_hash','checkpoint_finite','tokenizer_kind','tokenizer_hash','status','qnn_return_code_success','output_tensors_finite','cpu_fallback','validation_nonfinite_chunks','development_nonfinite_chunks')) {
    if (-not $Map.ContainsKey($key)) { throw "$Kind report missing $key" }
  }
  if ($Map.status -ne 'SUCCESS' -or $Map.qnn_return_code_success -ne 'true' -or $Map.output_tensors_finite -ne 'true' -or $Map.cpu_fallback -ne 'false' -or $Map.checkpoint_finite -ne 'true') { throw "$Kind report health rejected" }
  if ([int]$Map.model_dimension -ne 64 -or [int]$Map.feed_forward_dimension -ne 128 -or [int]$Map.layers -ne 19 -or [int]$Map.heads -ne 2 -or [int]$Map.vocabulary_size -ne 1024 -or [int]$Map.context_tokens -ne 32 -or [int]$Map.checkpoint_step -ne $Step -or [int]$Map.checkpoint_parameter_elements -ne 758528) { throw "$Kind report model identity mismatch" }
  if ($Map.tokenizer_kind -ne 'byte_bpe' -or $Map.tokenizer_hash -ne $Fixed.tokenizer_hash -or $Map.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "$Kind tokenizer/checkpoint identity mismatch" }
  if ($Map.ContainsKey('learning_rate') -and [single]$Map.learning_rate -ne [single]$Lr) { throw "$Kind learning-rate identity mismatch" }
  if ($Map.validation_nonfinite_chunks -ne '0' -or $Map.development_nonfinite_chunks -ne '0') { throw "$Kind nonfinite held-out chunks" }
}
function Test-Checkpoint([string]$Path,[int]$Step,[string]$Lr) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
  . (Join-Path $PSScriptRoot 'qairt_version.ps1')
  . (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
  try {
    $h = Get-PhoneLmCheckpointHeaders -Path $Path
    return $h.Magic -eq 'NPRTCKPTV3' -and $h.Vocabulary -eq 1024 -and $h.Tokens -eq 32 -and $h.Dimension -eq 64 -and $h.FeedForward -eq 128 -and $h.Layers -eq 19 -and $h.Heads -eq 2 -and $h.Seed -eq 1 -and $h.Step -eq $Step -and $h.TokenizerKind -eq 'byte_bpe' -and $h.TokenizerHash -eq $Fixed.tokenizer_hash
  } catch { return $false }
}
function Find-Report([string]$Dir,[int]$Step) { return Get-ChildItem -LiteralPath $Dir -File -Filter "*steps$Step-result.txt" -ErrorAction SilentlyContinue | Select-Object -First 1 }
function Find-EvalReport([string]$Dir,[int]$Step,[int]$V,[int]$D) { return Join-Path $Dir ((Get-EvalStem $Step $V $D) + '-htp.txt') }
function Test-ExistingTrainingArtifact([string]$Lr,[int]$Step) {
  $dir = Join-Path (Get-TrialDir $Lr) 'training'
  $report = Find-Report $dir $Step
  $checkpoint = Join-Path $dir (Get-CheckpointName $Step)
  if (-not $report -or -not (Test-Checkpoint $checkpoint $Step $Lr)) { return $false }
  try {
    $map = Read-KeyValue $report.FullName
    return $map.status -eq 'SUCCESS' -and
      $map.qnn_return_code_success -eq 'true' -and
      $map.output_tensors_finite -eq 'true' -and
      $map.cpu_fallback -eq 'false' -and
      $map.final_finite -eq 'true' -and
      $map.learning_rate -and [single]$map.learning_rate -eq [single]$Lr
  } catch { return $false }
}

function Invoke-Training([string]$Lr,[int]$Step,[int]$Resume,[int]$Interval,[string]$Phase,[switch]$AllowQuality) {
  $dir = Join-Path (Get-TrialDir $Lr) 'training'; [IO.Directory]::CreateDirectory($dir) | Out-Null
  $existing = Find-Report $dir $Step
  $existingCheckpoint = Join-Path $dir (Get-CheckpointName $Step)
  if ($existing -and (Test-Checkpoint $existingCheckpoint $Step $Lr)) {
    $existingMap = Read-KeyValue $existing.FullName
    $healthy = $existingMap.qnn_return_code_success -eq 'true' -and
      $existingMap.output_tensors_finite -eq 'true' -and
      $existingMap.cpu_fallback -eq 'false' -and
      $existingMap.final_finite -eq 'true' -and
      $existingMap.learning_rate -and [single]$existingMap.learning_rate -eq [single]$Lr
    if ($healthy) {
      Write-Event @{ trial_id = Get-TrialId $Lr; phase = "$Phase-reuse"; lr = $Lr; steps = $Step; resume_step = $Resume; wall_time_ms = 0; status = 'REUSED'; qnn_health = 'PASS'; checkpoint = $existingCheckpoint }
      return @{ report = $existing.FullName; map = $existingMap; wall_ms = 0; reused = $true }
    }
  }
  $runIdRaw = (Get-TrialId $Lr) + '-' + $Phase + '-' + (Get-Date -Format 'yyyyMMddHHmmssfff')
  $runId = $runIdRaw.Substring(0,[Math]::Min(63,$runIdRaw.Length))
  # One native graph phase can exceed 30 minutes on this fixed V1024 model.
  # Keep a bounded two-hour checkpoint-stall guard; no identity, finite,
  # QNN-return, thermal, or focus checks are relaxed.
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-Seed',1,'-Layers',19,'-Steps',$Step,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-BatchSize',8,'-LearningRate',$Lr,'-ResumeStep',$Resume,'-CheckpointInterval',$Interval,'-CheckpointStallSeconds',7200,'-CachePath',$TrainCache,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$dir,'-RunId',$runId)
  if ($SkipBuild -or $script:Prepared) { $args += '-SkipBuild' }
  if ($SkipInstall -or $script:Prepared) { $args += '-SkipInstall' }
  if ($AllowQuality) { $args += '-AllowQualityFailure' }
  $manifest = Load-Manifest $Lr
  $manifest.status = 'RUNNING'; $manifest.current_phase = $Phase; $manifest.current_steps = $Step; Save-Manifest $Lr $manifest
  $sw = [Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $TrainingRunner @args | Out-Host; $code = $LASTEXITCODE; $sw.Stop()
  if ($code -ne 0) { throw "TRAINING_PHASE_FAILED:${Phase}:${Lr}:exit=$code" }
  $report = Find-Report $dir $Step
  if (-not $report) { throw "TRAINING_REPORT_MISSING:${Phase}:${Lr}:${Step}" }
  $map = Read-KeyValue $report.FullName
  $manifest = Load-Manifest $Lr
  $manifest.completed_steps = $Step
  $manifest.smoke_health = 'QNN_FINITE_HEALTH_PASS'
  # Keep the manifest's root checkpoint fields synchronized with the
  # validated terminal training report.  These fields are the ledger's
  # source-of-truth pointer for resume/reuse; leaving them blank would make a
  # completed artifact impossible to audit without guessing from filenames.
  $finalCheckpoint = Join-Path $dir (Get-CheckpointName $Step)
  if (-not (Test-Checkpoint $finalCheckpoint $Step $Lr)) { throw "TRAINING_CHECKPOINT_IDENTITY_MISMATCH:${Phase}:${Lr}:${Step}" }
  $manifest.checkpoint_path = $finalCheckpoint
  if ($map.ContainsKey('final_parameter_hash') -and [string]$map.final_parameter_hash -match '^fnv1a64:[0-9a-f]{16}$') {
    $manifest.checkpoint_parameter_hash = [string]$map.final_parameter_hash
  }
  if ($map.ContainsKey('checkpoint_format') -and [string]$map.checkpoint_format -in @('NPRTCKPTV1','NPRTCKPTV2','NPRTCKPTV3')) {
    $manifest.checkpoint_format = [string]$map.checkpoint_format
  }
  $manifest.last_wall_time_ms = [math]::Round($sw.Elapsed.TotalMilliseconds,1)
  if ($map.ContainsKey('training_step_ms')) { $manifest.training_step_ms = [double]$map.training_step_ms }
  if ($map.ContainsKey('run_target_utf8_bytes_seen') -and $map.ContainsKey('training_total_seconds')) {
    $manifest.bytes_per_second = [math]::Round(([double]$map.run_target_utf8_bytes_seen / [double]$map.training_total_seconds),3)
  }
  Save-Manifest $Lr $manifest
  Write-Event @{ trial_id = Get-TrialId $Lr; phase = $Phase; lr = $Lr; steps = $Step; resume_step = $Resume; wall_time_ms = [math]::Round($sw.Elapsed.TotalMilliseconds,1); status = 'COMPLETED'; qnn_health = 'PASS'; checkpoint = (Join-Path $dir (Get-CheckpointName $Step)) }
  $script:Prepared = $true
  return @{ report = $report.FullName; map = $map; wall_ms = $sw.Elapsed.TotalMilliseconds }
}
function Invoke-Eval([string]$Lr,[int]$Step,[int]$V,[int]$D) {
  $trainDir = Join-Path (Get-TrialDir $Lr) 'training'; $evalDir = Join-Path (Get-TrialDir $Lr) "eval/rung-$Step-v$V-d$D"; [IO.Directory]::CreateDirectory($evalDir) | Out-Null
  $checkpoint = Join-Path $trainDir (Get-CheckpointName $Step)
  if (-not (Test-Checkpoint $checkpoint $Step $Lr)) { throw "EVAL_CHECKPOINT_IDENTITY_MISMATCH:${Lr}:${Step}" }
  $existing = Find-EvalReport $evalDir $Step $V $D
  if (Test-Path -LiteralPath $existing -PathType Leaf) {
    try {
      $existingMap = Read-KeyValue $existing
      Assert-Identity $existingMap $Lr $Step 'eval-reuse'
      $existingBalanced = Get-Balanced $existingMap
      Write-Event @{ trial_id = Get-TrialId $Lr; phase = "eval-$Step-reuse"; lr = $Lr; steps = $Step; val_bpb = [double]$existingMap.validation_bits_per_utf8_byte; dev_bpb = [double]$existingMap.development_bits_per_utf8_byte; balanced_bpb = $existingBalanced; validation_chunks = $V; development_chunks = $D; status = 'REUSED'; qnn_health = 'PASS'; checkpoint_hash = $existingMap.checkpoint_parameter_hash; wall_time_ms = 0 }
      return @{ path = $existing; map = $existingMap; balanced = $existingBalanced; wall_ms = 0; reused = $true }
    } catch {
      # An incomplete/corrupt report is not evidence of a valid eval; the
      # normal fresh run below will replace it after identity checks.
    }
  }
  $runIdRaw = (Get-TrialId $Lr) + "-eval-$Step-$V-$D-" + (Get-Date -Format 'yyyyMMddHHmmssfff')
  $runId = $runIdRaw.Substring(0,[Math]::Min(63,$runIdRaw.Length))
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-SkipBuild','-SkipInstall','-Seed',1,'-Layers',19,'-Heads',2,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-CheckpointStep',$Step,'-ValidationChunks',$V,'-DevelopmentChunks',$D,'-CheckpointPath',$checkpoint,'-CacheRoot',$EvalCacheRoot,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$evalDir,'-RunId',$runId)
  $sw = [Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $EvalRunner @args | Out-Host; $code = $LASTEXITCODE; $sw.Stop(); if ($code -ne 0) { throw "EVAL_PHASE_FAILED:${Lr}:${Step}:${V}:${D}:exit=$code" }
  $path = Find-EvalReport $evalDir $Step $V $D; if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "EVAL_REPORT_MISSING:${Lr}:${Step}:${V}:${D}" }
  $map = Read-KeyValue $path; Assert-Identity $map $Lr $Step 'eval'
  $balanced = Get-Balanced $map
  Write-Event @{ trial_id = Get-TrialId $Lr; phase = "eval-$Step"; lr = $Lr; steps = $Step; val_bpb = [double]$map.validation_bits_per_utf8_byte; dev_bpb = [double]$map.development_bits_per_utf8_byte; balanced_bpb = $balanced; validation_chunks = $V; development_chunks = $D; status = 'COMPLETED'; qnn_health = 'PASS'; checkpoint_hash = $map.checkpoint_parameter_hash; wall_time_ms = [math]::Round($sw.Elapsed.TotalMilliseconds,1) }
  return @{ path = $path; map = $map; balanced = $balanced; wall_ms = $sw.Elapsed.TotalMilliseconds }
}

function Invoke-IncumbentEval([string]$Lr,[int]$Step,[int]$V,[int]$D) {
  # The incumbent checkpoint is intentionally kept in the existing experiment
  # root.  If an exact-budget report is absent, run only the held-out eval;
  # never retrain LR=.003 or copy a different-LR checkpoint into this trial.
  $checkpoint = Join-Path $DefaultSourceReportRoot (Get-CheckpointName $Step)
  if (-not (Test-Checkpoint $checkpoint $Step $Lr)) { throw "INCUMBENT_CHECKPOINT_IDENTITY_MISMATCH:${Lr}:${Step}" }
  $evalDir = Join-Path (Get-TrialDir $Lr) "eval/rung-$Step-v$V-d$D"
  [IO.Directory]::CreateDirectory($evalDir) | Out-Null
  $existing = Find-EvalReport $evalDir $Step $V $D
  if (Test-Path -LiteralPath $existing -PathType Leaf) {
    $map = Read-KeyValue $existing; Assert-Identity $map $Lr $Step 'incumbent-eval'
    return @{ path = $existing; map = $map; balanced = Get-Balanced $map; wall_ms = 0; reused = $true }
  }
  $runIdRaw = (Get-TrialId $Lr) + "-incumbent-eval-$Step-$V-$D-" + (Get-Date -Format 'yyyyMMddHHmmssfff')
  $runId = $runIdRaw.Substring(0,[Math]::Min(63,$runIdRaw.Length))
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-SkipBuild','-SkipInstall','-Seed',1,'-Layers',19,'-Heads',2,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-CheckpointStep',$Step,'-ValidationChunks',$V,'-DevelopmentChunks',$D,'-CheckpointPath',$checkpoint,'-CacheRoot',$EvalCacheRoot,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$evalDir,'-RunId',$runId)
  $sw = [Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $EvalRunner @args | Out-Host; $code = $LASTEXITCODE; $sw.Stop()
  if ($code -ne 0) { throw "INCUMBENT_EVAL_PHASE_FAILED:${Lr}:${Step}:${V}:${D}:exit=$code" }
  $path = Find-EvalReport $evalDir $Step $V $D
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "INCUMBENT_EVAL_REPORT_MISSING:${Lr}:${Step}:${V}:${D}" }
  $map = Read-KeyValue $path; Assert-Identity $map $Lr $Step 'incumbent-eval'
  $balanced = Get-Balanced $map
  Write-Event @{ trial_id = Get-TrialId $Lr; phase = "incumbent-eval-$Step"; lr = $Lr; steps = $Step; val_bpb = [double]$map.validation_bits_per_utf8_byte; dev_bpb = [double]$map.development_bits_per_utf8_byte; balanced_bpb = $balanced; validation_chunks = $V; development_chunks = $D; status = 'COMPLETED'; qnn_health = 'PASS'; checkpoint_hash = $map.checkpoint_parameter_hash; wall_time_ms = [math]::Round($sw.Elapsed.TotalMilliseconds,1) }
  return @{ path = $path; map = $map; balanced = $balanced; wall_ms = $sw.Elapsed.TotalMilliseconds; reused = $false }
}

function Import-Incumbent([string]$Lr) {
  $src = $DefaultSourceReportRoot; $manifest = Load-Manifest $Lr; if ($manifest -and $manifest.status -eq 'COMPLETED') { return $manifest }
  $m = New-Manifest $Lr 'PENDING'; $m.reused = $false
  $report = Join-Path $src 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
  $checkpoint = Join-Path $src (Get-CheckpointName 8000)
  if (-not (Test-Path $report) -or -not (Test-Checkpoint $checkpoint 8000 $Lr)) { return $m }
  $rm = Read-KeyValue $report
  if ($rm.status -ne 'SUCCESS' -or $rm.learning_rate -notmatch '^0\.003') { return $m }
  if ([int]$rm.parameter_element_count -ne 758528 -or $rm.tokenizer_hash -ne $Fixed.tokenizer_hash -or $rm.all_steps_finite -ne 'true' -or $rm.final_finite -ne 'true' -or $rm.cpu_fallback -ne 'false' -or $rm.qnn_return_code_success -ne 'true' -or $rm.output_tensors_finite -ne 'true') { return $m }
  $m.status = 'COMPLETED'; $m.completed_steps = 8000; $m.reused = $true; $m.reuse_source = 'existing FFN128 LR=0.003 report/checkpoint with header, finite, parameter-count and QNN-health validation'; $m.checkpoint_path = $checkpoint; $m.checkpoint_parameter_hash = $rm.final_parameter_hash; $m.reuse_validation = 'PASS'; Save-Manifest $Lr $m
  Write-Event @{ trial_id = Get-TrialId $Lr; phase = 'incumbent-reuse'; lr = $Lr; steps = 8000; status = 'REUSED'; qnn_health = 'PASS'; checkpoint_hash = $rm.final_parameter_hash; wall_time_ms = 0 }
  return $m
}

function Get-AnchorSummaryRow([string]$Lr,[int]$Step) {
  $path = Join-Path $AnchorLedger 'summary.csv'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "ANCHOR_SUMMARY_MISSING:$Lr" }
  $rows = @(Import-Csv -LiteralPath $path | Where-Object { $_.lr -eq $Lr -and $_.rung -eq [string]$Step -and [int]$_.steps -eq $Step })
  if ($rows.Count -ne 1) { throw "ANCHOR_SUMMARY_ROW_MISSING:${Lr}:$Step" }
  $row = $rows[0]
  if ($row.status -eq 'FAILED' -or $row.qnn_health -ne 'PASS') { throw "ANCHOR_SUMMARY_HEALTH_REJECTED:${Lr}:$Step" }
  return $row
}

function Import-Anchor([string]$Lr) {
  if ($Variant -ne 'v1b') { throw 'ANCHOR_IMPORT_ONLY_V1B' }
  $sourceTrialId = Get-AnchorTrialId $Lr
  $sourceDir = Get-AnchorTrialDir $Lr
  $sourceManifestPath = Join-Path $sourceDir 'manifest.json'
  if (-not (Test-Path -LiteralPath $sourceManifestPath -PathType Leaf)) { throw "ANCHOR_MANIFEST_MISSING:$Lr" }
  $source = Get-Content -LiteralPath $sourceManifestPath -Raw | ConvertFrom-Json -AsHashtable
  if ($source.status -ne 'COMPLETED' -or [int]$source.completed_steps -lt 8000) { throw "ANCHOR_STATUS_INVALID:$Lr" }
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','tokenizer_kind','tokenizer_hash','batch_size','seed','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','lr_schedule','warmup_steps','decay_steps','dataset_identity','dataset_cache_content_hash','training_order_identity','training_order_hash')) {
    if (-not $source.ContainsKey($key) -or [string]$source[$key] -ne [string]$Fixed[$key]) { throw "ANCHOR_IDENTITY_MISMATCH:${Lr}:$key" }
  }
  if ([single]$source.learning_rate -ne [single]$Lr) { throw "ANCHOR_LR_MISMATCH:$Lr" }
  $sourceCheckpoint = [string]$source.checkpoint_path
  if (-not (Test-Checkpoint $sourceCheckpoint 8000 $Lr)) { throw "ANCHOR_CHECKPOINT_INVALID:$Lr" }
  if ([string]::IsNullOrWhiteSpace([string]$source.checkpoint_parameter_hash) -or [string]$source.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "ANCHOR_CHECKPOINT_HASH_MISSING:$Lr" }
  $sourceFinal = Get-AnchorSummaryRow $Lr 8000
  if ([string]$sourceFinal.checkpoint_hash -ne [string]$source.checkpoint_parameter_hash) { throw "ANCHOR_HASH_MISMATCH:$Lr" }
  $sourceEval = Join-Path $sourceDir "eval/rung-8000-v256-d256/$(Get-EvalStem 8000 256 256)-htp.txt"
  if (-not (Test-Path -LiteralPath $sourceEval -PathType Leaf)) { throw "ANCHOR_EVAL_MISSING:$Lr" }
  $sourceEvalMap = Read-KeyValue $sourceEval
  Assert-Identity $sourceEvalMap $Lr 8000 'anchor-eval'
  $m = New-Manifest $Lr 'COMPLETED'
  $m.reused = $true
  $m.reuse_source = "validated LR-v1 anchor $sourceTrialId"
  $m.reuse_validation = 'PASS'
  $m.anchor_source_trial_id = $sourceTrialId
  $m.anchor_source_manifest = $sourceManifestPath
  $m.completed_steps = 8000
  $m.current_steps = 8000
  $m.current_phase = 'anchor-reuse'
  $m.checkpoint_path = $sourceCheckpoint
  $m.checkpoint_parameter_hash = [string]$source.checkpoint_parameter_hash
  $m.smoke_health = 'QNN_FINITE_HEALTH_PASS'
  $m.rung8000 = [ordered]@{ val_bpb=[double]$sourceEvalMap.validation_bits_per_utf8_byte; dev_bpb=[double]$sourceEvalMap.development_bits_per_utf8_byte; balanced_bpb=(Get-Balanced $sourceEvalMap); health='PASS'; source='LR-v1 anchor' }
  Save-Manifest $Lr $m
  Write-Event @{ trial_id=Get-TrialId $Lr; phase='anchor-reuse'; lr=$Lr; steps=8000; status='REUSED'; qnn_health='PASS'; checkpoint_hash=$m.checkpoint_parameter_hash; reuse_source=$sourceTrialId; wall_time_ms=0 }
  return $m
}

function Import-AnchorEvalRow([string]$Lr,[int]$Step,[int]$V,[int]$D) {
  $sourceDir = Get-AnchorTrialDir $Lr
  $path = Join-Path $sourceDir "eval/rung-$Step-v$V-d$D/$(Get-EvalStem $Step $V $D)-htp.txt"
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  $map = Read-KeyValue $path; Assert-Identity $map $Lr $Step 'anchor-eval'
  return [pscustomobject]@{ lr=$Lr; rung=$Step; steps=$Step; val_bpb=[double]$map.validation_bits_per_utf8_byte; dev_bpb=[double]$map.development_bits_per_utf8_byte; balanced_bpb=(Get-Balanced $map); status='REUSED'; checkpoint_hash=$map.checkpoint_parameter_hash; wall_time_ms=0; qnn_health='PASS'; source='LR-v1 anchor' }
}

function Import-IncumbentEval([string]$Lr,[int]$Step,[int]$V,[int]$D) {
  $path = Join-Path (Join-Path $Root 'build/reports/nicopedia-htp-eval-v1024') ((Get-EvalStem $Step $V $D) + '-htp.txt')
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  $map = Read-KeyValue $path
  Assert-Identity $map $Lr $Step 'incumbent-eval'
  return @{ path = $path; map = $map; balanced = Get-Balanced $map }
}

function Write-Summary([array]$Rows) {
  $all = @()
  if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) { $all += @(Import-Csv -LiteralPath $SummaryPath) }
  $all += @($Rows)
  if ($all.Count -eq 0) { return }
  $byKey = @{}
  foreach ($row in $all) {
    if ($null -eq $row) { continue }
    $key = "{0}|{1}|{2}" -f $row.lr,$row.rung,$row.steps
    $byKey[$key] = $row
  }
  @($byKey.Values | Sort-Object @{Expression={[double]$_.lr}}, @{Expression={[int]$_.steps}}, rung) | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8
}

function Invoke-Plan {
  [IO.Directory]::CreateDirectory($TrialsRoot) | Out-Null
  foreach ($lr in $LrSpace) { if (-not (Load-Manifest $lr)) { Save-Manifest $lr (New-Manifest $lr 'PENDING') } }
  $latePolicy = if ($Variant -eq 'v1b') { 'diagnostic only: 6500/7500/8000 at 128+128; no late 256 primary' } else { 'step8000 256+256; late selector 6000/6500/7000/7500/8000 at 128+128' }
  $plan = [ordered]@{ hpo_schema_version = $SchemaVersion; variant=$Variant; architecture = $Fixed; search_space = $LrSpace; anchor_learning_rates=$AnchorLrs; anchor_ledger=$AnchorLedger; trial_order = $TrialOrder; rungs = @(1000,4000,8000); promotion_margin_bpb = 0.03; max_promoted = if($Variant -eq 'v1b'){2}else{3}; min_promoted = if($Variant -eq 'v1b'){2}else{2}; promotion_policy=if($Variant -eq 'v1b'){'retain both new candidates unless health/collapse failure'}else{'conservative 0.03 bpb margin / minimum-two rule'}; final_eval = $latePolicy; created_utc = Get-NowUtc }
  $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Ledger 'plan.json') -Encoding utf8
}

function Invoke-Run {
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'Run requires explicit QairtSdkRoot and ExpectedBuildId' }
  . (Join-Path $PSScriptRoot 'qairt_version.ps1')
  Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  if (-not (Test-Path -LiteralPath $TrainCache) -or -not (Test-Path -LiteralPath $TokenizerPath)) { throw 'HPO_PRIVATE_INPUT_MISSING' }
  Invoke-Plan
  $script:Prepared = $false
  $rows = [Collections.Generic.List[object]]::new()
  $incumbent = $null
  if ($Variant -eq 'v1b') {
    # LR-v1 anchors are imported only after manifest, checkpoint header/hash,
    # exact held-out report, and fixed-config/QNN-health checks pass.  Their
    # checkpoint is referenced in place; no cross-LR resume or retraining is
    # possible from this ledger.
    foreach ($anchorLr in $AnchorLrs) {
      $anchor = Import-Anchor $anchorLr
      $rows.Add([pscustomobject]@{ lr=$anchorLr; rung=8000; steps=8000; val_bpb=[double]$anchor.rung8000.val_bpb; dev_bpb=[double]$anchor.rung8000.dev_bpb; balanced_bpb=[double]$anchor.rung8000.balanced_bpb; status='REUSED'; checkpoint_hash=$anchor.checkpoint_parameter_hash; wall_time_ms=0; qnn_health='PASS'; source='LR-v1 anchor' })
      foreach ($anchorSpec in @(@{step=1000;v=64;d=64},@{step=4000;v=128;d=128})) {
        $anchorRow = Import-AnchorEvalRow $anchorLr $anchorSpec.step $anchorSpec.v $anchorSpec.d
        if ($anchorRow) { $rows.Add($anchorRow) }
      }
    }
  } else {
    # Import the incumbent before device work; it is never retrained.
    $incumbent = Import-Incumbent '0.0030'
    if ($incumbent.status -eq 'COMPLETED') {
      $incumbentFinal = Import-IncumbentEval '0.0030' 8000 256 256
      if (-not $incumbentFinal) { $incumbentFinal = Invoke-IncumbentEval '0.0030' 8000 256 256 }
      if ($incumbentFinal) {
        $rows.Add([pscustomobject]@{ lr='0.0030'; rung=8000; steps=8000; val_bpb=[double]$incumbentFinal.map.validation_bits_per_utf8_byte; dev_bpb=[double]$incumbentFinal.map.development_bits_per_utf8_byte; balanced_bpb=$incumbentFinal.balanced; status='REUSED'; checkpoint_hash=$incumbentFinal.map.checkpoint_parameter_hash; wall_time_ms=0; qnn_health='PASS' })
      }
    }
  }
  foreach ($lr in $TrialOrder) {
    if ($lr -eq '0.0030' -and $incumbent.status -eq 'COMPLETED') { continue }
    $m = Load-Manifest $lr; if ($m.status -eq 'COMPLETED' -and $m.completed_steps -ge 4000) { continue }
    try {
      # A valid rung-1000 artifact already proves that this LR has passed the
      # device smoke/health gate.  Reuse it and do not spend a duplicate
      # four-step training run when recovering a partial ledger.
      if (-not (Test-ExistingTrainingArtifact $lr 1000)) {
        $smoke = Invoke-Training $lr 4 0 4 'smoke' -AllowQuality
      } else {
        Write-Event @{ trial_id = Get-TrialId $lr; phase = 'smoke-reuse-from-rung1000'; lr = $lr; steps = 4; status = 'REUSED'; qnn_health = 'PASS'; wall_time_ms = 0 }
      }
      $r1 = Invoke-Training $lr 1000 4 250 'rung1000' -AllowQuality
      $e1 = Invoke-Eval $lr 1000 64 64
      $m = Load-Manifest $lr; $m.status = 'COMPLETED'; $m.rung1000 = [ordered]@{ val_bpb = [double]$e1.map.validation_bits_per_utf8_byte; dev_bpb = [double]$e1.map.development_bits_per_utf8_byte; balanced_bpb = $e1.balanced; health = 'PASS' }; Save-Manifest $lr $m
      $r4 = Invoke-Training $lr 4000 1000 250 'rung4000' -AllowQuality
      $e4 = Invoke-Eval $lr 4000 128 128
      $m = Load-Manifest $lr; $m.completed_steps = 4000; $m.rung4000 = [ordered]@{ val_bpb = [double]$e4.map.validation_bits_per_utf8_byte; dev_bpb = [double]$e4.map.development_bits_per_utf8_byte; balanced_bpb = $e4.balanced; health = 'PASS' }; Save-Manifest $lr $m
      $rows.Add([pscustomobject]@{ lr=$lr; rung=1000; steps=1000; val_bpb=[double]$e1.map.validation_bits_per_utf8_byte; dev_bpb=[double]$e1.map.development_bits_per_utf8_byte; balanced_bpb=$e1.balanced; status='COMPLETED'; checkpoint_hash=$r1.map.final_parameter_hash; wall_time_ms=$r1.wall_ms; qnn_health='PASS' })
      $rows.Add([pscustomobject]@{ lr=$lr; rung=4000; steps=4000; val_bpb=[double]$e4.map.validation_bits_per_utf8_byte; dev_bpb=[double]$e4.map.development_bits_per_utf8_byte; balanced_bpb=$e4.balanced; status='COMPLETED'; checkpoint_hash=$r4.map.final_parameter_hash; wall_time_ms=$r4.wall_ms; qnn_health='PASS' })
    } catch {
      $m = Load-Manifest $lr; if ($null -eq $m) { $m = New-Manifest $lr }
      $m.status = 'FAILED'; $m.failure = $_.Exception.Message; Save-Manifest $lr $m
      Write-Event @{ trial_id = Get-TrialId $lr; phase = 'failed'; lr = $lr; status = 'FAILED'; qnn_health = 'UNKNOWN'; detail = $_.Exception.Message }
    }
  }
  # Select conservatively from the 4000 rung.  The incumbent remains an
  # anchor when its metric is within the declared margin and costs no compute.
  $candidates = foreach ($lr in $LrSpace) { $m = Load-Manifest $lr; if ($m -and $m.ContainsKey('rung4000') -and $null -ne $m.rung4000) { [pscustomobject]@{ lr=$lr; balanced=[double]$m.rung4000.balanced_bpb } } }
  $ordered = @($candidates | Sort-Object balanced)
  if ($ordered.Count -gt 0) {
    $best = $ordered[0].balanced
    if ($Variant -eq 'v1b') {
      # Only two new points are in LR-v1b.  Keep both through the fixed-budget
      # endpoint; a small rung-4000 difference is not an oracle here.
      $promote = @($ordered)
    } else {
      $promote = @($ordered | Where-Object { $_.balanced -le $best + 0.03 } | Select-Object -First 3)
      if ($promote.Count -lt 2) { $promote = @($ordered | Select-Object -First ([Math]::Min(2,$ordered.Count))) }
    }
    foreach ($candidate in $promote) {
      $lr = $candidate.lr; $m = Load-Manifest $lr; $m.status = 'PROMOTED'; $m.promotion_reason = if ($Variant -eq 'v1b') { 'both LR-v1b candidates retained by no-prune policy' } elseif ($lr -eq $ordered[0].lr) { 'best balanced bpb at rung 4000' } else { 'within conservative 0.03 bpb margin / minimum-two rule' }; Save-Manifest $lr $m
    }
    if ($Variant -ne 'v1b') {
      foreach ($lr in $LrSpace) {
        $m = Load-Manifest $lr; if ($m -and $m.status -eq 'COMPLETED' -and $m.completed_steps -ge 4000 -and @($promote.lr) -notcontains $lr -and $lr -ne '0.0030') { $m.status='PRUNED'; $m.prune_reason='not in conservative top-three promotion set'; Save-Manifest $lr $m }
      }
    }
    foreach ($candidate in $promote) {
      $lr = $candidate.lr; $m = Load-Manifest $lr
      if ($m.reused) { continue }
      try {
        $r8 = Invoke-Training $lr 8000 4000 250 'rung8000' -AllowQuality
        $e8 = Invoke-Eval $lr 8000 256 256
        $m = Load-Manifest $lr; $m.status='COMPLETED'; $m.completed_steps=8000; $m.rung8000=[ordered]@{ val_bpb=[double]$e8.map.validation_bits_per_utf8_byte; dev_bpb=[double]$e8.map.development_bits_per_utf8_byte; balanced_bpb=$e8.balanced; health='PASS' }; Save-Manifest $lr $m
        $rows.Add([pscustomobject]@{ lr=$lr; rung=8000; steps=8000; val_bpb=[double]$e8.map.validation_bits_per_utf8_byte; dev_bpb=[double]$e8.map.development_bits_per_utf8_byte; balanced_bpb=$e8.balanced; status='COMPLETED'; checkpoint_hash=$r8.map.final_parameter_hash; wall_time_ms=$r8.wall_ms; qnn_health='PASS' })
        $lateSteps = if ($Variant -eq 'v1b') { @(6500,7500,8000) } else { @(6000,6500,7000,7500,8000) }
        foreach ($late in $lateSteps) {
          $lateEval = Invoke-Eval $lr $late 128 128
          $m = Load-Manifest $lr
          if (-not $m.ContainsKey('late_selector') -or $null -eq $m.late_selector) { $m.late_selector = @() }
          # Make late-selector recovery idempotent.  A host/eval retry may
          # revisit an already valid step; retain one authoritative row per
          # checkpoint instead of growing duplicate manifest entries.
          $priorLate = @($m.late_selector | Where-Object { [int]$_.step -ne $late })
          $m.late_selector = @($priorLate + [ordered]@{ step=$late; balanced_bpb=$lateEval.balanced; val_bpb=[double]$lateEval.map.validation_bits_per_utf8_byte; dev_bpb=[double]$lateEval.map.development_bits_per_utf8_byte })
          Save-Manifest $lr $m
        }
        $m = Load-Manifest $lr
        $lateBest = @($m.late_selector | Sort-Object {[double]$_.balanced_bpb} | Select-Object -First 1)
        if ($lateBest.Count -eq 1 -and $Variant -eq 'v1b') {
          $m = Load-Manifest $lr; $m.late_selector_best_128 = [ordered]@{ step=[int]$lateBest[0].step; val_bpb=[double]$lateBest[0].val_bpb; dev_bpb=[double]$lateBest[0].dev_bpb; balanced_bpb=[double]$lateBest[0].balanced_bpb; health='PASS'; diagnostic_only=$true }; Save-Manifest $lr $m
        } elseif ($lateBest.Count -eq 1) {
          $lateStep = [int]$lateBest[0].step
          $lateFinal = Invoke-Eval $lr $lateStep 256 256
          $m = Load-Manifest $lr; $m.late_selector_best = [ordered]@{ step=$lateStep; val_bpb=[double]$lateFinal.map.validation_bits_per_utf8_byte; dev_bpb=[double]$lateFinal.map.development_bits_per_utf8_byte; balanced_bpb=$lateFinal.balanced; checkpoint_hash=$lateFinal.map.checkpoint_parameter_hash; health='PASS' }; Save-Manifest $lr $m
          $rows.Add([pscustomobject]@{ lr=$lr; rung='late-best'; steps=$lateStep; val_bpb=[double]$lateFinal.map.validation_bits_per_utf8_byte; dev_bpb=[double]$lateFinal.map.development_bits_per_utf8_byte; balanced_bpb=$lateFinal.balanced; status='COMPLETED'; checkpoint_hash=$lateFinal.map.checkpoint_parameter_hash; wall_time_ms=$lateFinal.wall_ms; qnn_health='PASS' })
        }
      } catch {
        $m = Load-Manifest $lr; $m.status='FAILED'; $m.failure=$_.Exception.Message; Save-Manifest $lr $m; Write-Event @{ trial_id=Get-TrialId $lr; phase='final-failed'; lr=$lr; status='FAILED'; qnn_health='UNKNOWN'; detail=$_.Exception.Message }
      }
    }
  }
  # The incumbent's existing final and late reports are imported for the
  # exact-budget comparison; no new LR=.003 training is performed.
  Write-Summary @($rows)
  $completed = @($LrSpace | ForEach-Object { $m=Load-Manifest $_; if ($m -and $m.status -eq 'COMPLETED') { $_ } })
  $totalNew = [int](@($rows | Where-Object { $_.lr -ne '0.0030' -and $_.rung -in @(1000,4000,8000) } | ForEach-Object { $_.steps } | Measure-Object -Sum).Sum)
  $naiveSteps = $LrSpace.Count * 8000
  $reusedSteps = if ($Variant -eq 'v1b') { $AnchorLrs.Count * 8000 } else { 8000 }
  $promotedCount = if ($ordered.Count -gt 0) { @($promote).Count } else { 0 }
  $eff = [ordered]@{ variant=$Variant; trial_count=$LrSpace.Count; anchor_trial_count=$AnchorLrs.Count; total_training_steps_executed=$totalNew; actual_new_training_steps=$totalNew; reused_steps=$reusedSteps; naive_steps=$naiveSteps; compute_saving_percent=if($naiveSteps -gt 0){[math]::Round((1-(($totalNew)/$naiveSteps))*100,2)}else{0}; promoted_count=$promotedCount; completed_trials=$completed; boundary_hit=($Variant -eq 'v1' -and $completed -contains '0.0060' -and $ordered.Count -gt 0 -and $ordered[0].lr -eq '0.0060'); generated_utc=Get-NowUtc }
  $eff | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Ledger 'compute.json') -Encoding utf8
}

function Invoke-Summarize {
  $manifests = @($LrSpace | ForEach-Object { Load-Manifest $_ } | Where-Object { $_ })
  $manifests | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Ledger 'manifests.json') -Encoding utf8
  if (Test-Path -LiteralPath $SummaryPath) { Get-Content -LiteralPath $SummaryPath }
  foreach ($m in $manifests) { Write-Output ("{0} status={1} steps={2}" -f $m.trial_id,$m.status,$m.completed_steps) }
}

if ($SelfTest) {
  if ($Variant -eq 'v1b') {
    if (@($LrSpace).Count -ne 2 -or (Get-TrialId '0.00105') -ne 'hpo-lr-v1b-lr0p00105-seed1' -or (Get-TrialId '0.00075') -ne 'hpo-lr-v1b-lr0p00075-seed1') { throw 'HPO_SELFTEST_TRIAL_ID_V1B' }
    if (@($AnchorLrs).Count -ne 4 -or (Get-AnchorTrialId '0.0015') -ne 'hpo-lr-v1-lr0p0015-seed1') { throw 'HPO_SELFTEST_ANCHOR_ID_V1B' }
  } elseif (@($LrSpace).Count -ne 5 -or (Get-TrialId '0.0015') -ne 'hpo-lr-v1-lr0p0015-seed1') { throw 'HPO_SELFTEST_TRIAL_ID' }
  if ($Fixed.dimension -ne 64 -or $Fixed.feed_forward_dimension -ne 128 -or $Fixed.parameter_count -ne 758528 -or $Fixed.lr_schedule -ne 'constant') { throw 'HPO_SELFTEST_FIXED_CONFIG' }
  $fake = @([pscustomobject]@{lr='0.0030';balanced=2.50},[pscustomobject]@{lr='0.0042';balanced=2.49},[pscustomobject]@{lr='0.0060';balanced=2.515}) | Sort-Object balanced
  if ($fake[0].lr -ne '0.0042' -or @($fake | Where-Object {$_.balanced -le $fake[0].balanced + .03}).Count -ne 3) { throw 'HPO_SELFTEST_PROMOTION' }
  Write-Host 'run_nicopedia_hpo_lr_self_test=PASS'; exit 0
}
[IO.Directory]::CreateDirectory($Ledger) | Out-Null
switch ($Mode) { 'Plan' { Invoke-Plan }; 'Run' { Invoke-Run }; 'Summarize' { Invoke-Summarize } }
