# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# HexaTrain HPO Schedule-v2d: direct full-tail check of decay start=3000.
#
# This runner is deliberately a one-candidate experiment.  It reuses the
# validated C2200 constant-LR step-3000 parent, runs the S3000 linear tail
# directly to step 8000, and compares it with the validated S4000/S6000
# anchors.  No cheap tail proxy or intermediate ranking is implemented.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Run',
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2d',
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$Ledger = [IO.Path]::GetFullPath((Join-Path $Root $LedgerRoot))
$BuildRoot = [IO.Path]::GetFullPath((Join-Path $Root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $Ledger.StartsWith($BuildRoot, [StringComparison]::OrdinalIgnoreCase)) {
  throw 'Ledger root must resolve below build'
}
$TrialsRoot = Join-Path $Ledger 'trials'
$EventsPath = Join-Path $Ledger 'trials.jsonl'
$SummaryPath = Join-Path $Ledger 'summary.csv'
$PlanPath = Join-Path $Ledger 'plan.json'
$ComputePath = Join-Path $Ledger 'compute.json'
$TelemetryComparisonPath = Join-Path $Ledger 'lr-telemetry-comparison.csv'
$FullTailPath = Join-Path $Ledger 'full-tail-comparison.csv'

$TrainingRunner = Join-Path $Root 'scripts/run_nicopedia_htp_training.ps1'
$EvalRunner = Join-Path $Root 'scripts/run_nicopedia_htp_eval.ps1'
$DataRoot = Join-Path $Root 'build/private-data/nicopedia-real-text-bpe-v1024'
$TokenizerPath = Join-Path $DataRoot 'tokenizer/byte-bpe-v1024.model'
$TrainCache = Join-Path $DataRoot 'caches/train_pilot.bin'
$EvalCacheRoot = Join-Path $DataRoot 'caches'

# Immutable source artifacts.
$AnchorLedger = Join-Path $Root 'build/hpo/nicopedia-v1024-d64-f128/lr-v1'
$AnchorTrial = Join-Path $AnchorLedger 'trials/hpo-lr-v1-lr0p0022-seed1'
$V2bLedger = Join-Path $Root 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2b'
$Linear6000Trial = Join-Path $V2bLedger 'trials/hpo-schedule-v2b-t0100-seed1'
$Linear4000Trial = Join-Path $V2bLedger 'trials/hpo-schedule-v2b-full-s4000-t0100-seed1'

$PeakLearningRate = 0.0022
$TargetLearningRate = 0.0001
$DecayStartStep = 3000
$DecayEndStep = 8000
$ScheduleTotalSteps = 8000

$RunnerSchedule = 'linear_decay'
$TelemetrySteps = @(3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000)
$SmokeUpdates = 4
$TelemetryTolerance = 2.0e-8

$Fixed = [ordered]@{
  vocabulary = 1024; tokens = 32; dimension = 64; feed_forward_dimension = 128
  layers = 19; heads = 2; parameter_count = 758528; batch_size = 8; seed = 1
  tokenizer_kind = 'byte_bpe'; tokenizer_hash = 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798'
  optimizer = 'ADAM'; beta1 = 0.9; beta2 = 0.999; epsilon = '1e-8'; gradient_clip = 'disabled'; weight_decay = 0
  dataset_identity = 'nicopedia-real-text-bpe-v1024/train_pilot.bin'; dataset_cache_content_hash = 'fnv1a64:0c7b2826f5f26fea'
  training_order_identity = 'fixed canonical train_pilot round-robin order'; training_order_hash = 'fnv1a64:0e2e15196d851431'
  qnn_backend = 'HTP'; planned_max_steps = 8000; planned_target_tokens = 2048000
  expected_original_utf8_bytes = 5491256; expected_chunks = 46616; expected_articles = 1949
}

$Linear4000Anchor = [ordered]@{
  name = 'Linear4000'; shape = 'linear'; runner_schedule = 'linear_decay'
  trial_id = 'hpo-schedule-v2b-full-s4000-t0100-seed1'; target = '0.0001'; peak = '0.0022'
  start = 4000; end = 8000; val = 2.284370268; dev = 2.541861852; balanced = 2.413116060
}
$Linear6000Anchor = [ordered]@{
  name = 'Linear6000'; shape = 'linear'; runner_schedule = 'linear_decay'
  trial_id = 'hpo-schedule-v2b-t0100-seed1'; target = '0.0001'; peak = '0.0022'
  start = 6000; end = 8000; val = 2.286752948; dev = 2.543557534; balanced = 2.415155241
}
$LinearAnchor = $Linear4000Anchor
$Specs = @(
  [ordered]@{
    name = 'S3000'; shape = 'linear'; schedule_type = 'linear'; runner_schedule = $RunnerSchedule
    trial_id = 'hpo-schedule-v2d-s3000-seed1'; target = '0.0001'; peak = '0.0022'
    start = 3000; end = 8000; reused = $false; parent_step = 3000
  }
)

function NowUtc { [DateTime]::UtcNow.ToString('o') }

function WriteEvent([hashtable]$Fields) {
  [IO.Directory]::CreateDirectory($Ledger) | Out-Null
  $row = [ordered]@{ utc = NowUtc }
  foreach ($key in $Fields.Keys) { $row[$key] = $Fields[$key] }
  $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsPath -Encoding utf8
}

function TrialDir([string]$TrialId) { Join-Path $TrialsRoot $TrialId }
function ManifestPath([string]$TrialId) { Join-Path (TrialDir $TrialId) 'manifest.json' }

function ReadKeyValue([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "REPORT_MISSING:$Path" }
  $map = [ordered]@{}
  foreach ($line in Get-Content -LiteralPath $Path) {
    if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $map[$Matches[1]] = $Matches[2].Trim() }
  }
  return $map
}

function LoadManifest([string]$TrialId) {
  $path = ManifestPath $TrialId
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}

function SaveManifest([string]$TrialId, [System.Collections.IDictionary]$Manifest) {
  $dir = TrialDir $TrialId
  [IO.Directory]::CreateDirectory($dir) | Out-Null
  $Manifest | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath (ManifestPath $TrialId) -Encoding utf8
}

function Balanced([System.Collections.IDictionary]$Map) {
  return ([double]$Map.validation_bits_per_utf8_byte + [double]$Map.development_bits_per_utf8_byte) / 2.0
}

function ExpectedLinearLearningRate([double]$Target, [int]$Step, [int]$Start = 3000, [int]$End = 8000) {
  if ($Step -le $Start) { return $PeakLearningRate }
  if ($Step -ge $End) { return $Target }
  $progress = ($Step - $Start) / [double]($End - $Start)
  return $PeakLearningRate + $progress * ($Target - $PeakLearningRate)
}

function ExpectedLearningRate([System.Collections.IDictionary]$Spec, [int]$Step) {
  if ($Spec.shape -ne 'linear') { throw "UNKNOWN_SCHEDULE_SHAPE:$($Spec.shape)" }
  return ExpectedLinearLearningRate ([double]$Spec.target) $Step ([int]$Spec.start) ([int]$Spec.end)
}

function CheckpointName([int]$Step) { "htp-seed1-l19-t32-d64-f128-step$Step.ckpt" }
function EvalName([int]$Step) { "seed1-l19-t32-d64-f128-step$Step-v256-d256-htp.txt" }

function AssertParentHash([string]$Expected, [string]$Actual) {
  if ($Expected -ne $Actual) { throw 'PARENT_CHECKPOINT_HASH_MISMATCH' }
}

function AssertMapFields([System.Collections.IDictionary]$Map, [string[]]$Fields, [string]$Prefix) {
  foreach ($key in $Fields) {
    if (-not $Map.Contains($key)) { throw "${Prefix}_FIELD_MISSING:$key" }
  }
}

function AssertQnnFiniteHealth([System.Collections.IDictionary]$Map, [string]$Prefix) {
  AssertMapFields $Map @(
    'qnn_return_code_success','output_tensors_finite','cpu_fallback','nan_detected','inf_detected',
    'api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded'
  ) $Prefix
  # Return-code success and tensor finiteness are deliberately independent
  # checks; either one failing rejects the artifact.
  if ($Map.qnn_return_code_success -ne 'true' -or $Map.output_tensors_finite -ne 'true' -or
      $Map.cpu_fallback -ne 'false' -or $Map.nan_detected -ne 'false' -or $Map.inf_detected -ne 'false' -or
      $Map.api_trace_graph_execute_failure_count -ne '0' -or
      $Map.api_trace_fallback_attempted -ne 'false' -or $Map.api_trace_fallback_succeeded -ne 'false') {
    throw "${Prefix}_QNN_HEALTH_REJECTED"
  }
}

function ValidateEval([string]$Path, [int]$Step) {
  $m = ReadKeyValue $Path
  AssertMapFields $m @(
    'status','seed','layers','heads','model_dimension','feed_forward_dimension','checkpoint_step',
    'checkpoint_format','checkpoint_finite','checkpoint_parameter_elements','checkpoint_parameter_hash',
    'context_tokens','vocabulary_size','tokenizer_kind','tokenizer_hash','validation_chunks','development_chunks',
    'validation_nonfinite_chunks','development_nonfinite_chunks','validation_bits_per_utf8_byte',
    'development_bits_per_utf8_byte','api_trace_runtime_backend_build_id'
  ) 'EVAL'
  if ($m.status -ne 'SUCCESS' -or [int]$m.seed -ne 1 -or [int]$m.layers -ne 19 -or [int]$m.heads -ne 2 -or
      [int]$m.model_dimension -ne 64 -or [int]$m.feed_forward_dimension -ne 128 -or
      [int]$m.checkpoint_step -ne $Step -or $m.checkpoint_format -ne 'NPRTCKPTV3' -or
      $m.checkpoint_finite -ne 'true' -or [int]$m.checkpoint_parameter_elements -ne $Fixed.parameter_count -or
      [int]$m.context_tokens -ne 32 -or [int]$m.vocabulary_size -ne 1024 -or
      $m.tokenizer_kind -ne $Fixed.tokenizer_kind -or $m.tokenizer_hash -ne $Fixed.tokenizer_hash -or
      [int]$m.validation_chunks -ne 256 -or [int]$m.development_chunks -ne 256 -or
      $m.validation_nonfinite_chunks -ne '0' -or $m.development_nonfinite_chunks -ne '0') {
    throw "EVAL_IDENTITY_OR_HEALTH_REJECTED:$Path"
  }
  AssertQnnFiniteHealth $m 'EVAL'
  if ($m.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "EVAL_HASH_INVALID:$Path" }
  if ($m.api_trace_runtime_backend_build_id -ne $ExpectedBuildId -and
      $m.api_trace_runtime_backend_build_id -ne "v$ExpectedBuildId") { throw "EVAL_QAIRT_BUILD_MISMATCH:$Path" }
  return $m
}

function ValidateTrainingMap(
  [System.Collections.IDictionary]$Map,
  [System.Collections.IDictionary]$Spec,
  [int]$Steps,
  [int]$Resume
) {
  AssertMapFields $Map @(
    'status','completed_steps','run_completed_steps','model_dimension','feed_forward_dimension','parameter_element_count',
    'optimizer_chunk_count','qnn_return_code_success','output_tensors_finite','cpu_fallback','final_finite','all_steps_finite',
    'final_parameter_hash','training_total_seconds','training_step_ms','run_target_utf8_bytes_seen',
    'android_thermal_status_before','android_thermal_status_after','battery_temperature_c_before','battery_temperature_c_after',
    'api_trace_runtime_backend_build_id','api_trace_backend_requested',
    'learning_rate_schedule','learning_rate_schedule_identity','learning_rate_decay_start_step','learning_rate_decay_end_step','learning_rate_schedule_total_steps',
    'learning_rate_peak','learning_rate_target','experiment_fork','parent_learning_rate','resume_from_step',
    'resume_checkpoint_format','run_target_tokens_seen','target_tokens_seen','target_utf8_bytes_seen','unique_chunks_seen',
    'unique_articles_seen'
  ) 'TRAINING'
  if ($Map.status -ne 'SUCCESS' -or [int]$Map.completed_steps -ne $Steps -or
      [int]$Map.run_completed_steps -ne ($Steps - $Resume) -or [int]$Map.model_dimension -ne 64 -or
      [int]$Map.feed_forward_dimension -ne 128 -or [int]$Map.parameter_element_count -ne $Fixed.parameter_count -or
      [int]$Map.optimizer_chunk_count -ne 93 -or $Map.qnn_return_code_success -ne 'true' -or
      $Map.output_tensors_finite -ne 'true' -or $Map.cpu_fallback -ne 'false' -or
      $Map.final_finite -ne 'true' -or $Map.all_steps_finite -ne 'true' -or
      $Map.final_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$' -or
      $Map.learning_rate_schedule -ne $Spec.runner_schedule -or
      $Map.learning_rate_schedule_identity -ne $Spec.runner_schedule -or
      [int]$Map.learning_rate_decay_start_step -ne [int]$Spec.start -or
      [int]$Map.learning_rate_decay_end_step -ne [int]$Spec.end -or
      [int]$Map.learning_rate_schedule_total_steps -ne $ScheduleTotalSteps -or
      [single]$Map.learning_rate_peak -ne [single]$Spec.peak -or
      [single]$Map.learning_rate_target -ne [single]$Spec.target -or
      $Map.experiment_fork -ne 'true' -or [single]$Map.parent_learning_rate -ne [single]'0.0022' -or
      [int]$Map.resume_from_step -ne $Resume -or $Map.resume_checkpoint_format -ne 'NPRTCKPTV3') {
    throw "TRAINING_HEALTH_OR_SCHEDULE_REJECTED:$($Spec.name):$Steps"
  }
  AssertQnnFiniteHealth $Map 'TRAINING'
  if ($Map.api_trace_backend_requested -ne 'HTP' -or
      ($Map.api_trace_runtime_backend_build_id -ne $ExpectedBuildId -and
       $Map.api_trace_runtime_backend_build_id -ne "v$ExpectedBuildId")) {
    throw "TRAINING_QAIRT_IDENTITY_REJECTED:$($Spec.name):$Steps"
  }
  $expectedRunTokens = [int64](($Steps - $Resume) * $Fixed.batch_size * $Fixed.tokens)
  if ([int64]$Map.run_target_tokens_seen -ne $expectedRunTokens) {
    throw "EXPOSURE_RUN_TOKENS_MISMATCH:$($Spec.name):$Steps"
  }
  if ($Steps -eq $ScheduleTotalSteps) {
    foreach ($pair in @(
      @('target_tokens_seen', $Fixed.planned_target_tokens),
      @('target_utf8_bytes_seen', $Fixed.expected_original_utf8_bytes),
      @('unique_chunks_seen', $Fixed.expected_chunks),
      @('unique_articles_seen', $Fixed.expected_articles)
    )) {
      if ([int64]$Map[$pair[0]] -ne [int64]$pair[1]) { throw "EXPOSURE_TOTAL_MISMATCH:$($pair[0])" }
    }
  }
  return $Map
}

function AssertCheckpointIdentity([string]$Path, [int]$Step) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "PARENT_CHECKPOINT_MISSING:$Step" }
  $header = Get-PhoneLmCheckpointHeaders -Path $Path
  if ($header.Magic -ne 'NPRTCKPTV3' -or $header.Vocabulary -ne $Fixed.vocabulary -or
      $header.Tokens -ne $Fixed.tokens -or $header.Dimension -ne $Fixed.dimension -or
      $header.FeedForward -ne $Fixed.feed_forward_dimension -or $header.Layers -ne $Fixed.layers -or
      $header.Heads -ne $Fixed.heads -or $header.Seed -ne $Fixed.seed -or $header.Step -ne $Step -or
      $header.TokenizerKind -ne $Fixed.tokenizer_kind -or $header.TokenizerHash -ne $Fixed.tokenizer_hash) {
    throw "CHECKPOINT_IDENTITY_MISMATCH:$Step"
  }
  if ((Get-Item -LiteralPath $Path).Length -le 0) { throw "CHECKPOINT_EMPTY:$Step" }
  return $header
}

function AssertParentCheckpoint([int]$Step) {
  $manifestPath = Join-Path $AnchorTrial 'manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'PARENT_MANIFEST_MISSING' }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
  if ($manifest.status -ne 'COMPLETED' -or [int]$manifest.completed_steps -ne 8000 -or
      [int]$manifest.current_steps -ne 8000) { throw 'PARENT_COMPLETION_INVALID' }
  foreach ($key in @(
    'vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed',
    'tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay',
    'dataset_cache_content_hash','training_order_hash','lr_schedule','learning_rate'
  )) {
    $expected = if ($key -eq 'lr_schedule') { 'constant' } elseif ($key -eq 'learning_rate') { '0.0022' } else { [string]$Fixed[$key] }
    if (-not $manifest.Contains($key) -or [string]$manifest[$key] -ne $expected) {
      throw "PARENT_MANIFEST_IDENTITY_MISMATCH:$key"
    }
  }
  $reportPath = Join-Path (Join-Path $AnchorTrial 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
  $report = ReadKeyValue $reportPath
  AssertMapFields $report @(
    'status','completed_steps','learning_rate','parameter_element_count','optimizer_chunk_count',
    'all_steps_finite','final_finite','checkpoint_written','final_parameter_hash','resume_checkpoint_format'
  ) 'PARENT'
  if ($report.status -ne 'SUCCESS' -or [int]$report.completed_steps -ne 8000 -or
      [single]$report.learning_rate -ne [single]'0.0022' -or
      [int]$report.parameter_element_count -ne $Fixed.parameter_count -or [int]$report.optimizer_chunk_count -ne 93 -or
      $report.all_steps_finite -ne 'true' -or $report.final_finite -ne 'true' -or
      $report.checkpoint_written -ne 'true' -or $report.final_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') {
    throw 'PARENT_TRAINING_IDENTITY_OR_FINITE_REJECTED'
  }
  AssertQnnFiniteHealth $report 'PARENT'
  $path = Join-Path (Join-Path $AnchorTrial 'training') (CheckpointName $Step)
  AssertCheckpointIdentity $path $Step | Out-Null
  $evalExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
  $validation = Join-Path $DataRoot 'caches/validation.bin'
  $development = Join-Path $DataRoot 'caches/development.bin'
  if (-not (Test-Path -LiteralPath $evalExe -PathType Leaf) -or
      -not (Test-Path -LiteralPath $validation -PathType Leaf) -or
      -not (Test-Path -LiteralPath $development -PathType Leaf)) {
    throw 'PARENT_HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE'
  }
  $probe = & $evalExe $path $validation $development 1 1
  if ($LASTEXITCODE -ne 0) { throw "PARENT_HOST_CHECKPOINT_EVALUATOR_FAILED:$Step" }
  $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join "`n")
  if ($probeMap.step -ne [string]$Step -or $probeMap.seed -ne '1' -or $probeMap.layers -ne '19' -or
      $probeMap.dimension -ne '64' -or $probeMap.feed_forward_dimension -ne '128' -or
      $probeMap.finite -ne 'true' -or $probeMap.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') {
    throw "PARENT_HOST_IDENTITY_MISMATCH:$Step"
  }
  $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  return [ordered]@{
    path = $path; step = $Step; sha256 = "sha256:$sha"; parameter_hash = $probeMap.parameter_hash
    source_trial_id = 'hpo-lr-v1-lr0p0022-seed1'; source_manifest = $manifestPath
  }
}

function ValidateCheckpointFile([string]$Path, [int]$Step) {
  AssertCheckpointIdentity $Path $Step | Out-Null
  $evalExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
  $validation = Join-Path $DataRoot 'caches/validation.bin'
  $development = Join-Path $DataRoot 'caches/development.bin'
  if (-not (Test-Path -LiteralPath $evalExe -PathType Leaf) -or
      -not (Test-Path -LiteralPath $validation -PathType Leaf) -or
      -not (Test-Path -LiteralPath $development -PathType Leaf)) {
    throw 'PARENT_HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE'
  }
  $probe = & $evalExe $Path $validation $development 1 1
  if ($LASTEXITCODE -ne 0) { throw "PARENT_HOST_CHECKPOINT_EVALUATOR_FAILED:$Step" }
  $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join [Environment]::NewLine)
  foreach ($field in @('step','seed','layers','dimension','feed_forward_dimension','finite','parameter_hash')) {
    if (-not $probeMap.Contains($field)) { throw "PARENT_HOST_IDENTITY_FIELD_MISSING:$field" }
  }
  if ([int]$probeMap.step -ne $Step -or [int]$probeMap.seed -ne 1 -or
      [int]$probeMap.layers -ne 19 -or [int]$probeMap.dimension -ne 64 -or
      [int]$probeMap.feed_forward_dimension -ne 128 -or $probeMap.finite -ne 'true' -or
      $probeMap.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') {
    throw "PARENT_HOST_IDENTITY_MISMATCH:$Step"
  }
  $sha = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  return [ordered]@{
    path = $Path; step = $Step; sha256 = "sha256:$sha"; parameter_hash = $probeMap.parameter_hash
  }
}

function ResolveC2200Parent([int]$RequiredStep) {
  if ($RequiredStep -ne 3000) { throw "UNSUPPORTED_PARENT_STEP:$RequiredStep" }
  $exactPath = Join-Path (Join-Path $AnchorTrial 'training') (CheckpointName $RequiredStep)
  if (Test-Path -LiteralPath $exactPath -PathType Leaf) {
    return (AssertParentCheckpoint $RequiredStep)
  }

  # Validate the immutable C2200 source before considering a lower checkpoint.
  [void](AssertParentCheckpoint 8000)
  $anchorTraining = Join-Path $AnchorTrial 'training'
  $candidates = @(Get-ChildItem -LiteralPath $anchorTraining -Filter 'htp-seed1-l19-t32-d64-f128-step*.ckpt' -File |
    ForEach-Object {
      if ($_.Name -match 'step(\d+)\.ckpt$') { [pscustomobject]@{ step = [int]$Matches[1]; path = $_.FullName } }
    } |
    Where-Object { $_.step -gt 0 -and $_.step -lt $RequiredStep } |
    Sort-Object step -Descending)
  foreach ($candidate in $candidates) {
    try {
      $validated = ValidateCheckpointFile $candidate.path $candidate.step
      return [ordered]@{
        path = $validated.path; step = $validated.step; sha256 = $validated.sha256
        parameter_hash = $validated.parameter_hash; source_trial_id = 'hpo-lr-v1-lr0p0022-seed1'
        source_manifest = (Join-Path $AnchorTrial 'manifest.json')
        source_checkpoint_step = $validated.step
        parent_creation_overhead_steps = $RequiredStep - $validated.step
      }
    } catch {
      continue
    }
  }
  throw "C2200_PARENT_BEFORE_REQUIRED_STEP_MISSING:$RequiredStep"
}

function ValidateManifestIdentity(
  [System.Collections.IDictionary]$Manifest,
  [System.Collections.IDictionary]$Spec,
  [System.Collections.IDictionary]$Parent
) {
  if ($null -eq $Manifest) { return }
  foreach ($pair in @(
    @('trial_id', $Spec.trial_id), @('target_lr', $Spec.target), @('peak_lr', $Spec.peak),
    @('schedule_type', $Spec.schedule_type), @('parent_trial_id', $Parent.source_trial_id),
    @('parent_checkpoint_hash', $Parent.sha256), @('parent_parameter_hash', $Parent.parameter_hash)
  )) {
    if (-not $Manifest.Contains($pair[0]) -or [string]$Manifest[$pair[0]] -ne [string]$pair[1]) {
      throw "EXISTING_MANIFEST_IDENTITY_MISMATCH:$($Spec.trial_id):$($pair[0])"
    }
  }
  foreach ($pair in @(
    @('parent_step', $Parent.step), @('fork_step', $Parent.step),
    @('decay_start_step', $Spec.start), @('decay_end_step', $Spec.end),
    @('schedule_total_steps', $ScheduleTotalSteps), @('reused_prefix_steps', $Parent.step)
  )) {
    if (-not $Manifest.Contains($pair[0]) -or [int]$Manifest[$pair[0]] -ne [int]$pair[1]) {
      throw "EXISTING_MANIFEST_IDENTITY_MISMATCH:$($Spec.trial_id):$($pair[0])"
    }
  }
  if (-not $Manifest.Contains('runner_schedule') -or
      [string]$Manifest.runner_schedule -ne [string]$Spec.runner_schedule -or
      [string]$Manifest.experiment_fork -ne 'True' -and [string]$Manifest.experiment_fork -ne 'true') {
    throw "EXISTING_MANIFEST_SCHEDULE_OR_FORK_MISMATCH:$($Spec.trial_id)"
  }
  if (-not $Manifest.Contains('fixed_config') -or $null -eq $Manifest.fixed_config) {
    throw "EXISTING_MANIFEST_FIXED_CONFIG_MISSING:$($Spec.trial_id)"
  }
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','dataset_cache_content_hash','training_order_hash')) {
    if (-not $Manifest.fixed_config.Contains($key) -or [string]$Manifest.fixed_config[$key] -ne [string]$Fixed[$key]) {
      throw "EXISTING_MANIFEST_FIXED_CONFIG_MISMATCH:$($Spec.trial_id):$key"
    }
  }
}

function NewManifest([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  return [ordered]@{
    schema_version = 1; experiment = 'HexaTrain HPO Schedule-v2d'; trial_id = $Spec.trial_id; status = 'PENDING'; created_utc = (NowUtc)
    fixed_config = $Fixed; target_lr = $Spec.target; target_name = $Spec.name; peak_lr = $Spec.peak
    decay_start_step = $Spec.start; decay_end_step = $Spec.end; schedule_total_steps = $ScheduleTotalSteps
    schedule_type = $Spec.schedule_type; runner_schedule = $Spec.runner_schedule; shape = $Spec.shape
    schedule_expression = 'step <= decay_start: peak_lr; otherwise peak_lr + ((step-decay_start)/(decay_end-decay_start)) * (target_lr-peak_lr); step >= decay_end: target_lr'
    warmup_steps = 0; experiment_fork = $true; parent_trial_id = $Parent.source_trial_id
    parent_checkpoint_path = $Parent.path; parent_checkpoint_hash = $Parent.sha256; parent_parameter_hash = $Parent.parameter_hash
    parent_step = $Parent.step; fork_step = $Parent.step; parent_learning_rate = '0.0022'; reused_prefix_steps = $Parent.step
    parent_creation_overhead_steps = if ($Parent.Contains('parent_creation_overhead_steps')) { [int]$Parent.parent_creation_overhead_steps } else { 0 }
    actual_new_steps = 0; completed_steps = 0; checkpoint_path = ''; checkpoint_parameter_hash = ''
    smoke_health = 'PENDING'; final_health = 'PENDING'; git_revision = (& git -C $Root rev-parse HEAD).Trim()
  }
}

function EnsureManifestIdentity([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  $m = LoadManifest $Spec.trial_id
  if ($null -eq $m) { return (NewManifest $Spec $Parent) }
  ValidateManifestIdentity $m $Spec $Parent
  return $m
}

function CopyParentCheckpoint([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'
  [IO.Directory]::CreateDirectory($training) | Out-Null
  $destination = Join-Path $training (CheckpointName $Parent.step)
  if (Test-Path -LiteralPath $destination -PathType Leaf) {
    $actual = 'sha256:' + (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    AssertParentHash $Parent.sha256 $actual
    AssertCheckpointIdentity $destination $Parent.step | Out-Null
  } else {
    Copy-Item -LiteralPath $Parent.path -Destination $destination
  }
  return $training
}

function InvokeTraining(
  [System.Collections.IDictionary]$Spec,
  [int]$Steps,
  [int]$Resume,
  [string]$Phase,
  [switch]$Smoke,
  [int]$CheckpointStallSeconds = 7200,
  [int]$PollLimit = 7200,
  [string]$TrainingRootOverride = ''
) {
  $training = if ([string]::IsNullOrWhiteSpace($TrainingRootOverride)) {
    Join-Path (TrialDir $Spec.trial_id) 'training'
  } else {
    [IO.Path]::GetFullPath($TrainingRootOverride)
  }
  [IO.Directory]::CreateDirectory($training) | Out-Null
  if ($CheckpointStallSeconds -lt 1 -or $PollLimit -lt 1) { throw 'TRAINING_POLL_CONFIGURATION_INVALID' }
  $runId = "$($Spec.trial_id)-$Phase-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"
  if ($runId.Length -gt 63) { $runId = $runId.Substring(0, 63) }
  $args = @(
    '-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-Seed',1,'-Layers',19,'-Steps',$Steps,
    '-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-BatchSize',8,
    '-LearningRate','0.0022','-LearningRateSchedule',$Spec.runner_schedule,'-DecayStartStep',$Spec.start,
    '-DecayEndStep',$Spec.end,'-ScheduleTotalSteps',$ScheduleTotalSteps,'-TargetLearningRate',$Spec.target,
    '-ParentLearningRate','0.0022','-ExperimentFork','-ResumeStep',$Resume,'-CheckpointInterval',250,
    '-CheckpointStallSeconds',$CheckpointStallSeconds,'-PollLimit',$PollLimit,'-CachePath',$TrainCache,
    '-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$training,'-RunId',$runId
  )
  if ($SkipBuild -or $script:Prepared) { $args += '-SkipBuild' }
  if ($SkipInstall -or $script:Prepared) { $args += '-SkipInstall' }
  if ($Smoke) { $args += '-AllowQualityFailure' }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & pwsh -NoProfile -File $TrainingRunner @args | Out-Host
  $code = $LASTEXITCODE
  $sw.Stop()
  if ($code -ne 0) { throw "TRAINING_FAILED:$($Spec.name):$Phase:exit=$code" }
  $reportPath = Join-Path $training "seed1-l19-v1024-t32-d64-f128-steps$Steps-result.txt"
  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "TRAINING_REPORT_MISSING:$Phase" }
  $map = ReadKeyValue $reportPath
  ValidateTrainingMap $map $Spec $Steps $Resume | Out-Null
  return [ordered]@{ report = $reportPath; map = $map; wall_ms = $sw.Elapsed.TotalMilliseconds }
}

function FindPartialTailResumeStep([System.Collections.IDictionary]$Spec) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'
  $finalReportPath = Join-Path $training 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
  if (Test-Path -LiteralPath $finalReportPath -PathType Leaf) {
    $finalMap = ReadKeyValue $finalReportPath
    if ($finalMap.Contains('resume_from_step') -and [int]$finalMap.resume_from_step -ge [int]$Spec.start -and
        [int]$finalMap.resume_from_step -lt 8000) { return [int]$finalMap.resume_from_step }
  }
  $partialPath = Join-Path $training 'seed1-l19-v1024-t32-d64-f128-steps8000-partial-status.json'
  if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf)) { return [int]$Spec.start }
  $partial = Get-Content -LiteralPath $partialPath -Raw | ConvertFrom-Json -AsHashtable
  if ([string]$partial.status -ne 'RUNNING' -or [string]$partial.run_id -notlike "$($Spec.trial_id)-final-*") {
    return [int]$Spec.start
  }
  $candidates = @(Get-ChildItem -LiteralPath $training -Filter 'htp-seed1-l19-t32-d64-f128-step*.ckpt' -File |
    ForEach-Object { if ($_.Name -match 'step(\d+)\.ckpt$') { [int]$Matches[1] } } |
    Where-Object { $_ -gt [int]$Spec.start -and $_ -lt 8000 } | Sort-Object -Descending)
  foreach ($step in $candidates) {
    $path = Join-Path $training (CheckpointName $step)
    try { AssertCheckpointIdentity $path $step | Out-Null } catch { continue }
    $evalExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
    $validation = Join-Path $DataRoot 'caches/validation.bin'
    $development = Join-Path $DataRoot 'caches/development.bin'
    if (-not (Test-Path -LiteralPath $evalExe -PathType Leaf) -or
        -not (Test-Path -LiteralPath $validation -PathType Leaf) -or
        -not (Test-Path -LiteralPath $development -PathType Leaf)) { return [int]$Spec.start }
    $probe = & $evalExe $path $validation $development 1 1
    if ($LASTEXITCODE -ne 0) { continue }
    $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join "`n")
    if ($probeMap.step -eq [string]$step -and $probeMap.seed -eq '1' -and $probeMap.finite -eq 'true') {
      return $step
    }
  }
  return [int]$Spec.start
}

function ReadTelemetryRows([string]$TrainingDir, [string]$ExpectedSchedule) {
  $actual = @{}
  foreach ($prior in @(Get-ChildItem -LiteralPath $TrainingDir -Filter 'learning-rate-telemetry-prior-*.csv' -File -ErrorAction SilentlyContinue)) {
    foreach ($row in @(Import-Csv -LiteralPath $prior.FullName)) {
      if (-not $row.PSObject.Properties['learning_rate_schedule'] -or $row.learning_rate_schedule -ne $ExpectedSchedule) { throw "LEARNING_RATE_TELEMETRY_SCHEDULE_MISMATCH:$($prior.Name)" }
      $actual[[int]$row.step] = [double]$row.scheduled_lr
    }
  }
  $path = Join-Path $TrainingDir 'learning-rate-telemetry.csv'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'LEARNING_RATE_TELEMETRY_MISSING' }
  foreach ($row in @(Import-Csv -LiteralPath $path)) {
    if (-not $row.PSObject.Properties['learning_rate_schedule'] -or $row.learning_rate_schedule -ne $ExpectedSchedule) { throw "LEARNING_RATE_TELEMETRY_SCHEDULE_MISMATCH:$($path)" }
    $actual[[int]$row.step] = [double]$row.scheduled_lr
  }
  return $actual
}

function WriteTelemetryAnchors([System.Collections.IDictionary]$Spec, [int]$Resume) {
  $trainingDir = Join-Path (TrialDir $Spec.trial_id) 'training'
  $actual = @{}
  foreach ($prior in @(Get-ChildItem -LiteralPath $trainingDir -Filter 'learning-rate-telemetry-prior-*.csv' -File -ErrorAction SilentlyContinue)) {
    foreach ($row in @(Import-Csv -LiteralPath $prior.FullName)) {
      if (-not $row.PSObject.Properties['step'] -or
          -not $row.PSObject.Properties['scheduled_lr'] -or
          -not $row.PSObject.Properties['learning_rate_schedule'] -or
          $row.learning_rate_schedule -ne $Spec.runner_schedule) {
        throw "LEARNING_RATE_TELEMETRY_SCHEDULE_MISMATCH:$($prior.Name)"
      }
      $actual[[int]$row.step] = [double]$row.scheduled_lr
    }
  }
  $path = Join-Path $trainingDir 'learning-rate-telemetry.csv'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'LEARNING_RATE_TELEMETRY_MISSING' }
  foreach ($row in @(Import-Csv -LiteralPath $path)) {
    if (-not $row.PSObject.Properties['step'] -or
        -not $row.PSObject.Properties['scheduled_lr'] -or
        -not $row.PSObject.Properties['learning_rate_schedule'] -or
        $row.learning_rate_schedule -ne $Spec.runner_schedule) {
      throw "LEARNING_RATE_TELEMETRY_SCHEDULE_MISMATCH:$path"
    }
    $actual[[int]$row.step] = [double]$row.scheduled_lr
  }
  $rows = foreach ($step in $TelemetrySteps) {
    $expected = ExpectedLearningRate $Spec $step
    $source = 'runtime_telemetry'
    if ($actual.ContainsKey($step)) {
      $value = $actual[$step]
    } elseif ($step -le $Resume -and $step -le [int]$Spec.start) {
      $value = $PeakLearningRate
      $source = 'validated_parent_constant_lr'
    } elseif ($step -eq $Resume) {
      $value = $expected
      $source = 'validated_resume_checkpoint_boundary_formula'
    } else {
      throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISSING:$step"
    }
    if (-not [double]::IsFinite([double]$value)) { throw "LEARNING_RATE_TELEMETRY_NONFINITE:$step" }
    $delta = [double]$value - [double]$expected
    if ([math]::Abs($delta) -gt $TelemetryTolerance) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISMATCH:$step" }
    [pscustomobject]@{ step = $step; expected_lr = $expected; actual_lr = $value; delta = $delta; source = $source }
  }
  $out = Join-Path $trainingDir 'schedule-telemetry-anchors.csv'
  @($rows) | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8
  return [ordered]@{ path = $out; rows = @($rows) }
}

function InvokeEval([System.Collections.IDictionary]$Spec, [int]$Step) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'
  $checkpoint = Join-Path $training (CheckpointName $Step)
  $evalDir = Join-Path (TrialDir $Spec.trial_id) "eval/step-$Step-v256-d256"
  [IO.Directory]::CreateDirectory($evalDir) | Out-Null
  $path = Join-Path $evalDir (EvalName $Step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $runId = "$($Spec.trial_id)-eval-$Step-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"
    if ($runId.Length -gt 63) { $runId = $runId.Substring(0, 63) }
    & pwsh -NoProfile -File $EvalRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -SkipBuild -SkipInstall `
      -Seed 1 -Layers 19 -Heads 2 -Tokens 32 -Vocabulary 1024 -Dimension 64 -FeedForwardDimension 128 `
      -CheckpointStep $Step -ValidationChunks 256 -DevelopmentChunks 256 -CheckpointPath $checkpoint `
      -CacheRoot $EvalCacheRoot -TokenizerModelPath $TokenizerPath -ReportRoot $evalDir -RunId $runId | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "EVAL_FAILED:$($Spec.name):$Step" }
  }
  $m = ValidateEval $path $Step
  return [ordered]@{
    path = $path; map = $m; val = [double]$m.validation_bits_per_utf8_byte
    dev = [double]$m.development_bits_per_utf8_byte; balanced = (Balanced $m); health = 'PASS'
  }
}

function NewManifestShape([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  $m = EnsureManifestIdentity $Spec $Parent
  if ($null -eq $m) { throw "MANIFEST_CREATE_FAILED:$($Spec.trial_id)" }
  return $m
}

function ValidateExistingLinearArtifact(
  [System.Collections.IDictionary]$Anchor,
  [string]$TrialPath,
  [int]$ParentStep,
  [int]$ExpectedStart,
  [System.Collections.IDictionary]$Parent,
  [double]$ExpectedVal,
  [double]$ExpectedDev,
  [double]$ExpectedBalanced
) {
  $manifestPath = Join-Path $TrialPath 'manifest.json'
  $evalPath = Join-Path $TrialPath 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
  foreach ($path in @($manifestPath,$evalPath)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "LINEAR_ANCHOR_MISSING:$path" } }
  $m = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
  foreach ($pair in @(
    @('trial_id',$Anchor.trial_id), @('status','COMPLETED'), @('target_lr','0.0001'), @('peak_lr','0.0022'), @('schedule_type','linear'),
    @('schedule_total_steps',8000), @('warmup_steps',0),
    @('parent_trial_id','hpo-lr-v1-lr0p0022-seed1'), @('parent_checkpoint_hash',$Parent.sha256),
    @('parent_parameter_hash',$Parent.parameter_hash)
  )) {
    if (-not $m.Contains($pair[0]) -or [string]$m[$pair[0]] -ne [string]$pair[1]) { throw "LINEAR_ANCHOR_MANIFEST_MISMATCH:$($pair[0])" }
  }
  if ($m.Contains('runner_schedule') -and [string]$m.runner_schedule -ne 'linear_decay') {
    throw 'LINEAR_ANCHOR_MANIFEST_MISMATCH:runner_schedule'
  }
  foreach ($pair in @(@('completed_steps',8000),@('decay_start_step',$ExpectedStart),@('decay_end_step',8000),@('parent_step',$ParentStep),@('fork_step',$ParentStep),@('reused_prefix_steps',$ParentStep))) {
    if (-not $m.Contains($pair[0]) -or [int]$m[$pair[0]] -ne [int]$pair[1]) { throw "LINEAR_ANCHOR_MANIFEST_MISMATCH:$($pair[0])" }
  }
  if (-not $m.Contains('fixed_config') -or $null -eq $m.fixed_config) { throw 'LINEAR_ANCHOR_FIXED_CONFIG_MISSING' }
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','dataset_cache_content_hash','training_order_hash')) {
    if (-not $m.fixed_config.Contains($key) -or [string]$m.fixed_config[$key] -ne [string]$Fixed[$key]) { throw "LINEAR_ANCHOR_FIXED_CONFIG_MISMATCH:$key" }
  }
  $e = ValidateEval $evalPath 8000
  if ([math]::Abs(([double]$e.validation_bits_per_utf8_byte) - $ExpectedVal) -gt 1e-9 -or
      [math]::Abs(([double]$e.development_bits_per_utf8_byte) - $ExpectedDev) -gt 1e-9 -or
      [math]::Abs((Balanced $e) - $ExpectedBalanced) -gt 1e-9) { throw "LINEAR_ANCHOR_METRIC_MISMATCH:$TrialPath" }
  return [ordered]@{ manifest = $manifestPath; eval = $evalPath; map = $e; val = [double]$e.validation_bits_per_utf8_byte; dev = [double]$e.development_bits_per_utf8_byte; balanced = (Balanced $e) }
}

function WriteTelemetryComparison([array]$TelemetryRows) {
  $rows = foreach ($row in $TelemetryRows) {
    [pscustomobject]@{
      step = [int]$row.step
      expected_lr = [double]$row.expected_lr
      actual_lr = [double]$row.actual_lr
      delta = [double]$row.delta
      source = [string]$row.source
    }
  }
  @($rows) | Export-Csv -LiteralPath $TelemetryComparisonPath -NoTypeInformation -Encoding utf8
  return @($rows)
}

function NewComparisonRow(
  [int]$Start,
  [string]$TrialId,
  [System.Collections.IDictionary]$Eval,
  [string]$Status,
  [string]$Source,
  [double]$Start4000Balanced
) {
  return [pscustomobject]@{
    start = $Start
    trial_id = $TrialId
    val_bpb = [double]$Eval.val
    dev_bpb = [double]$Eval.dev
    balanced_bpb = [double]$Eval.balanced
    delta_vs_start4000 = [math]::Round(([double]$Eval.balanced - $Start4000Balanced), 9)
    status = $Status
    source = $Source
    checkpoint_hash = $Eval.map.checkpoint_parameter_hash
    qnn_health = 'PASS'
  }
}

function GetStartVerdict([double]$DeltaVal, [double]$DeltaDev, [double]$DeltaBalanced) {
  $improvement = -1.0 * $DeltaBalanced
  if ($improvement -ge 0.005 -and $DeltaVal -le 0.0 -and $DeltaDev -le 0.0) {
    return 'strong improvement'
  }
  if ($improvement -le -0.005) { return 'worse' }
  return 'flat'
}

function InvokePlan {
  [IO.Directory]::CreateDirectory($TrialsRoot) | Out-Null
  $availableParent = ResolveC2200Parent 3000
  $parent4000 = AssertParentCheckpoint 4000
  $parent6000 = AssertParentCheckpoint 6000
  $linear4000 = ValidateExistingLinearArtifact $Linear4000Anchor $Linear4000Trial 4000 4000 $parent4000 $Linear4000Anchor.val $Linear4000Anchor.dev $Linear4000Anchor.balanced
  $linear6000 = ValidateExistingLinearArtifact $Linear6000Anchor $Linear6000Trial 6000 6000 $parent6000 $Linear6000Anchor.val $Linear6000Anchor.dev $Linear6000Anchor.balanced
  $plan = [ordered]@{
    schema_version = 1
    experiment = 'HexaTrain HPO Schedule-v2d'
    motivation = 'Directly test decay start=3000 after the S4000/S6000 linear target=.0001 anchors'
    fixed_config = $Fixed
    required_parent_step = 3000
    available_c2200_parent = $availableParent
    parent_creation_required = ([int]$availableParent.step -ne 3000)
    existing_linear4000 = $linear4000
    existing_linear6000 = $linear6000
    candidates = @(
      [ordered]@{
        name = 'S3000'; shape = 'linear'; schedule_type = 'linear'; runner_schedule = 'linear_decay'
        target_lr = '0.0001'; peak_lr = '0.0022'; decay_start_step = 3000; decay_end_step = 8000
        status = 'NEW'; full_tail_steps = 5000
      }
    )
    no_cheap_tail_proxy = $true
    no_intermediate_ranking = $true
    smoke_updates = $SmokeUpdates
    telemetry_steps = $TelemetrySteps
    primary_eval = 'step8000 exact 256+256'
    created_utc = (NowUtc)
  }
  $plan | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $PlanPath -Encoding utf8
  if ([int]$availableParent.step -eq 3000) {
    $manifest = EnsureManifestIdentity $Specs[0] $availableParent
    SaveManifest $Specs[0].trial_id $manifest
  }
  return [ordered]@{
    parent = $availableParent
    parent4000 = $parent4000
    parent6000 = $parent6000
    linear4000 = $linear4000
    linear6000 = $linear6000
  }
}

function CompleteManifest(
  [System.Collections.IDictionary]$Spec,
  [System.Collections.IDictionary]$Parent,
  [System.Collections.IDictionary]$Run,
  [System.Collections.IDictionary]$Telemetry,
  [int]$ActualNewSteps
) {
  $m = LoadManifest $Spec.trial_id
  ValidateManifestIdentity $m $Spec $Parent
  $m.status = 'COMPLETED'; $m.failure = $null; $m.completed_steps = 8000; $m.actual_new_steps = $ActualNewSteps
  $m.checkpoint_path = Join-Path (TrialDir $Spec.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'
  $m.checkpoint_parameter_hash = $Run.map.final_parameter_hash; $m.final_health = 'PASS'; $m.telemetry_anchors_path = $Telemetry.path
  $m.wall_time_ms = if ([double]$Run.wall_ms -gt 0.0) { [double]$Run.wall_ms } else { [double]$Run.map.training_total_seconds * 1000.0 }
  $m.training_total_seconds = [double]$Run.map.training_total_seconds
  $m.training_step_ms = [double]$Run.map.training_step_ms
  if ([double]$Run.map.training_total_seconds -gt 0) {
    $m.run_bytes_per_second = [math]::Round(([double]$Run.map.run_target_utf8_bytes_seen / [double]$Run.map.training_total_seconds), 6)
  }
  $m.run_target_tokens = [int64]$Run.map.run_target_tokens_seen; $m.run_target_utf8_bytes = [int64]$Run.map.run_target_utf8_bytes_seen
  $m.total_target_tokens = [int64]$Run.map.target_tokens_seen; $m.total_target_utf8_bytes = [int64]$Run.map.target_utf8_bytes_seen
  $m.unique_chunks = [int64]$Run.map.unique_chunks_seen; $m.unique_articles = [int64]$Run.map.unique_articles_seen
  $m.android_thermal_status_before = [int]$Run.map.android_thermal_status_before; $m.android_thermal_status_after = [int]$Run.map.android_thermal_status_after
  $m.battery_temperature_c_before = [double]$Run.map.battery_temperature_c_before; $m.battery_temperature_c_after = [double]$Run.map.battery_temperature_c_after
  SaveManifest $Spec.trial_id $m
  return $m
}

function AddPrimaryToManifest([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Eval) {
  $m = LoadManifest $Spec.trial_id
  if ($null -eq $m) { throw "MANIFEST_MISSING_AFTER_RUN:$($Spec.trial_id)" }
  $m.primary = [ordered]@{ val_bpb = $Eval.val; dev_bpb = $Eval.dev; balanced_bpb = $Eval.balanced; health = 'PASS'; checkpoint_hash = $Eval.map.checkpoint_parameter_hash }
  SaveManifest $Spec.trial_id $m
  return $m
}

function EnsureExactParent([System.Collections.IDictionary]$AvailableParent) {
  if ([int]$AvailableParent.step -eq 3000) {
    $AvailableParent.parent_creation_overhead_steps = 0
    $AvailableParent.parent_creation_wall_time_ms = 0.0
    return $AvailableParent
  }

  $parentRoot = Join-Path (Join-Path (TrialDir $Specs[0].trial_id) 'training') 'parent-create'
  [IO.Directory]::CreateDirectory($parentRoot) | Out-Null
  $sourceName = CheckpointName ([int]$AvailableParent.step)
  $stagedSource = Join-Path $parentRoot $sourceName
  if (Test-Path -LiteralPath $stagedSource -PathType Leaf) {
    AssertParentHash $AvailableParent.sha256 ('sha256:' + (Get-FileHash -LiteralPath $stagedSource -Algorithm SHA256).Hash.ToLowerInvariant())
    AssertCheckpointIdentity $stagedSource ([int]$AvailableParent.step) | Out-Null
  } else {
    Copy-Item -LiteralPath $AvailableParent.path -Destination $stagedSource
  }

  $createdPath = Join-Path $parentRoot (CheckpointName 3000)
  $createdReport = Join-Path $parentRoot 'seed1-l19-v1024-t32-d64-f128-steps3000-result.txt'
  $creationRun = $null
  if (-not (Test-Path -LiteralPath $createdPath -PathType Leaf)) {
    $creationRun = InvokeTraining $Specs[0] 3000 ([int]$AvailableParent.step) 'parent-create' -TrainingRootOverride $parentRoot
  } elseif (Test-Path -LiteralPath $createdReport -PathType Leaf) {
    $creationMap = ReadKeyValue $createdReport
    ValidateTrainingMap $creationMap $Specs[0] 3000 ([int]$AvailableParent.step) | Out-Null
    $creationRun = [ordered]@{ report = $createdReport; map = $creationMap; wall_ms = 0.0 }
  }
  $validated = ValidateCheckpointFile $createdPath 3000
  $script:Prepared = $true
  return [ordered]@{
    path = $validated.path; step = 3000; sha256 = $validated.sha256
    parameter_hash = $validated.parameter_hash
    source_trial_id = $AvailableParent.source_trial_id
    source_manifest = $AvailableParent.source_manifest
    source_checkpoint_step = [int]$AvailableParent.step
    parent_creation_overhead_steps = 3000 - [int]$AvailableParent.step
    parent_creation_wall_time_ms = if ($null -eq $creationRun) { 0.0 } else { [double]$creationRun.wall_ms }
  }
}

function InvokeRun {
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'Run requires explicit QairtSdkRoot and ExpectedBuildId' }
  Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  foreach ($path in @($TrainCache,$TokenizerPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "HPO_PRIVATE_INPUT_MISSING:$path" }
  }

  $script:Prepared = $false
  $wall = [Diagnostics.Stopwatch]::StartNew()
  $setup = InvokePlan
  $parent = EnsureExactParent $setup.parent
  if ([int]$setup.parent.step -ne [int]$parent.step) {
    $planned = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json -AsHashtable
    $planned.exact_parent = $parent
    $planned | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $PlanPath -Encoding utf8
  }
  $manifest = EnsureManifestIdentity $Specs[0] $parent
  $manifest.parent_creation_overhead_steps = [int]$parent.parent_creation_overhead_steps
  $manifest.parent_creation_wall_time_ms = [double]$parent.parent_creation_wall_time_ms
  SaveManifest $Specs[0].trial_id $manifest
  CopyParentCheckpoint $Specs[0] $parent | Out-Null

  $linear4000Eval = [ordered]@{
    path = $setup.linear4000.eval; map = $setup.linear4000.map
    val = $setup.linear4000.val; dev = $setup.linear4000.dev
    balanced = $setup.linear4000.balanced; health = 'PASS'
  }
  $linear6000Eval = [ordered]@{
    path = $setup.linear6000.eval; map = $setup.linear6000.map
    val = $setup.linear6000.val; dev = $setup.linear6000.dev
    balanced = $setup.linear6000.balanced; health = 'PASS'
  }

  $candidate = $Specs[0]
  $candidateEval = $null
  $candidateTelemetry = $null
  $candidateRun = $null
  $candidateStatus = 'COMPLETED'
  try {
    $manifest = LoadManifest $candidate.trial_id
    $evalPath = Join-Path (TrialDir $candidate.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
    if ($manifest.status -eq 'COMPLETED' -and [int]$manifest.completed_steps -eq 8000 -and
        $manifest.Contains('primary') -and (Test-Path -LiteralPath $evalPath -PathType Leaf)) {
      $candidateEval = InvokeEval $candidate 8000
      $candidateTelemetry = WriteTelemetryAnchors $candidate 3000
      WriteTelemetryComparison $candidateTelemetry.rows | Out-Null
      if (-not $manifest.Contains('wall_time_ms') -or [double]$manifest.wall_time_ms -le 0.0) {
        if ($manifest.Contains('training_total_seconds') -and [double]$manifest.training_total_seconds -gt 0.0) {
          $manifest.wall_time_ms = [double]$manifest.training_total_seconds * 1000.0
          SaveManifest $candidate.trial_id $manifest
        }
      }
      $candidateStatus = 'REUSED'
    } else {
      $smokeReportPath = Join-Path (Join-Path (TrialDir $candidate.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps3004-result.txt'
      if (Test-Path -LiteralPath $smokeReportPath -PathType Leaf) {
        $smokeMap = ReadKeyValue $smokeReportPath
        ValidateTrainingMap $smokeMap $candidate 3004 3000 | Out-Null
        $manifest = LoadManifest $candidate.trial_id
        $manifest.smoke_health = 'PASS'
        $manifest.smoke_report = $smokeReportPath
        $manifest.smoke_updates = $SmokeUpdates
        SaveManifest $candidate.trial_id $manifest
        $script:Prepared = $true
      } else {
        $smoke = InvokeTraining $candidate 3004 3000 'smoke' -Smoke
        $manifest = LoadManifest $candidate.trial_id
        $manifest.smoke_health = 'PASS'
        $manifest.smoke_report = $smoke.report
        $manifest.smoke_updates = $SmokeUpdates
        SaveManifest $candidate.trial_id $manifest
        $script:Prepared = $true
      }

      $resume = FindPartialTailResumeStep $candidate
      $finalReportPath = Join-Path (Join-Path (TrialDir $candidate.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
      if (Test-Path -LiteralPath $finalReportPath -PathType Leaf) {
        $existingMap = ReadKeyValue $finalReportPath
        ValidateTrainingMap $existingMap $candidate 8000 $resume | Out-Null
        $candidateRun = [ordered]@{ report = $finalReportPath; map = $existingMap; wall_ms = 0.0 }
      } else {
        $candidateRun = InvokeTraining $candidate 8000 $resume 'final' -CheckpointStallSeconds 21600 -PollLimit 43200
      }
      $candidateTelemetry = WriteTelemetryAnchors $candidate $resume
      [void](CompleteManifest $candidate $parent $candidateRun $candidateTelemetry 5000)
      $candidateEval = InvokeEval $candidate 8000
      [void](AddPrimaryToManifest $candidate $candidateEval)
      $candidateStatus = 'COMPLETED'
    }
  } catch {
    $failedManifest = LoadManifest $candidate.trial_id
    if ($null -ne $failedManifest) {
      $failedManifest.status = 'FAILED'
      $failedManifest.failure = $_.Exception.Message
      SaveManifest $candidate.trial_id $failedManifest
    }
    WriteEvent @{ trial_id = $candidate.trial_id; phase = 'full-tail'; status = 'FAILED'; detail = $_.Exception.Message }
    throw
  }

  $candidateRow = NewComparisonRow 3000 $candidate.trial_id $candidateEval $candidateStatus $candidate.trial_id $linear4000Eval.balanced
  $s4000Row = NewComparisonRow 4000 $Linear4000Anchor.trial_id $linear4000Eval 'EXISTING' 'existing Linear4000' $linear4000Eval.balanced
  $s6000Row = NewComparisonRow 6000 $Linear6000Anchor.trial_id $linear6000Eval 'EXISTING' 'existing Linear6000' $linear4000Eval.balanced
  $rows = @($candidateRow,$s4000Row,$s6000Row)
  @($rows) | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8
  @($rows) | Export-Csv -LiteralPath $FullTailPath -NoTypeInformation -Encoding utf8

  $deltaVal = [double]$candidateEval.val - [double]$linear4000Eval.val
  $deltaDev = [double]$candidateEval.dev - [double]$linear4000Eval.dev
  $deltaBalanced = [double]$candidateEval.balanced - [double]$linear4000Eval.balanced
  $improvement = -1.0 * $deltaBalanced
  $bothImprove = $deltaVal -le 0.0 -and $deltaDev -le 0.0
  $startVerdict = GetStartVerdict $deltaVal $deltaDev $deltaBalanced
  $wall.Stop()

  $candidateManifest = LoadManifest $candidate.trial_id
  $parentOverhead = [int]$parent.parent_creation_overhead_steps
  $tailSteps = 5000
  $actualNew = $tailSteps + $parentOverhead
  $recordedWallMs = if ($candidateManifest.Contains('wall_time_ms')) {
    [double]$candidateManifest.wall_time_ms
  } else { [double]$candidateRun.wall_ms }
  if ($recordedWallMs -le 0.0 -and $candidateManifest.Contains('training_total_seconds')) {
    $recordedWallMs = [double]$candidateManifest.training_total_seconds * 1000.0
  }
  $recordedWallMs += [double]$parent.parent_creation_wall_time_ms
  $compute = [ordered]@{
    schema_version = 1
    experiment = 'HexaTrain HPO Schedule-v2d'
    naive_fresh_steps = 8000
    actual_new_steps = $actualNew
    tail_new_steps = $tailSteps
    reused_prefix_steps = 3000
    parent_reused_steps = 3000
    parent_creation_overhead_steps = $parentOverhead
    saved_steps = 8000 - $actualNew
    saving_percent = [math]::Round((1.0 - ($actualNew / 8000.0)) * 100.0, 2)
    smoke_probe_steps = $SmokeUpdates
    wall_time_ms = [math]::Round($recordedWallMs, 1)
    training_wall_time_ms = [math]::Round($recordedWallMs, 1)
    training_wall_time_seconds = [math]::Round($recordedWallMs / 1000.0, 3)
    training_step_ms = if ($candidateManifest.Contains('training_step_ms')) { [double]$candidateManifest.training_step_ms } else { $null }
    bytes_per_second = if ($candidateManifest.Contains('run_bytes_per_second')) { [double]$candidateManifest.run_bytes_per_second } else { $null }
    full_tail_executed = $true
    cheap_tail_proxy_used = $false
    candidate = 'S3000'
    primary_eval = 'step8000 exact 256+256'
    delta_val_vs_start4000 = $deltaVal
    delta_dev_vs_start4000 = $deltaDev
    delta_balanced_vs_start4000 = $deltaBalanced
    improvement_balanced_bpb = $improvement
    both_splits_not_worse = $bothImprove
    start_verdict = $startVerdict
    next_start_axis = if ($startVerdict -eq 'strong improvement') { 'continue to 2000' } else { 'stop' }
    qnn_health = 'PASS'
    finite = 'true'
    cpu_fallback = 'false'
    fallback_attempted = 'false'
    thermal_status = if ($candidateManifest.Contains('android_thermal_status_before')) {
      "$($candidateManifest.android_thermal_status_before)->$($candidateManifest.android_thermal_status_after)"
    } else { 'NOT_RECORDED' }
    battery_temperature_c = if ($candidateManifest.Contains('battery_temperature_c_before')) {
      "$($candidateManifest.battery_temperature_c_before)->$($candidateManifest.battery_temperature_c_after)"
    } else { 'NOT_RECORDED' }
    generated_utc = (NowUtc)
  }
  $compute | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $ComputePath -Encoding utf8

  $candidateManifest = LoadManifest $candidate.trial_id
  $candidateManifest.primary_comparison = [ordered]@{
    baseline_start = 4000
    delta_val = $deltaVal; delta_dev = $deltaDev; delta_balanced = $deltaBalanced
    balanced_improvement = $improvement; both_splits_not_worse = $bothImprove
    verdict = $startVerdict
  }
  SaveManifest $candidate.trial_id $candidateManifest
  WriteEvent @{
    trial_id = $candidate.trial_id; phase = 'complete'; status = 'COMPLETED'
    candidate = 'S3000'; full_tail = $true; no_cheap_tail_proxy = $true
    start_verdict = $startVerdict; delta_balanced = $deltaBalanced
  }
}

function InvokeSummarize {
  foreach ($path in @($SummaryPath,$FullTailPath,$TelemetryComparisonPath,$ComputePath,$PlanPath)) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      Write-Output "--- $path"
      Get-Content -LiteralPath $path
    }
  }
  foreach ($spec in $Specs) {
    $m = LoadManifest $spec.trial_id
    if ($null -ne $m) {
      Write-Output "$($spec.trial_id) status=$($m.status) steps=$($m.completed_steps)"
    }
  }
}

if ($SelfTest) {
  if ($Fixed.vocabulary -ne 1024 -or $Fixed.tokens -ne 32 -or $Fixed.dimension -ne 64 -or
      $Fixed.feed_forward_dimension -ne 128 -or $Fixed.layers -ne 19 -or $Fixed.heads -ne 2 -or
      $Fixed.parameter_count -ne 758528 -or $Fixed.batch_size -ne 8 -or $Fixed.seed -ne 1 -or
      $Fixed.tokenizer_kind -ne 'byte_bpe' -or
      $Fixed.tokenizer_hash -ne 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798' -or
      $Fixed.dataset_cache_content_hash -ne 'fnv1a64:0c7b2826f5f26fea' -or
      $Fixed.planned_target_tokens -ne 2048000 -or
      $Fixed.expected_original_utf8_bytes -ne 5491256 -or
      $Fixed.expected_chunks -ne 46616 -or $Fixed.expected_articles -ne 1949) {
    throw 'SCHEDULE_V2D_SELFTEST_FIXED'
  }
  if ($Specs.Count -ne 1 -or $Specs[0].name -ne 'S3000' -or
      $Specs[0].shape -ne 'linear' -or $Specs[0].schedule_type -ne 'linear' -or
      $Specs[0].runner_schedule -ne 'linear_decay' -or $Specs[0].target -ne '0.0001' -or
      $Specs[0].peak -ne '0.0022' -or $Specs[0].start -ne 3000 -or
      $Specs[0].end -ne 8000 -or $Specs[0].parent_step -ne 3000) {
    throw 'SCHEDULE_V2D_SELFTEST_SPEC'
  }
  if ($TelemetrySteps.Count -ne 11 -or
      ($TelemetrySteps -join ',') -ne '3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000') {
    throw 'SCHEDULE_V2D_SELFTEST_TELEMETRY_STEPS'
  }
  $expected = @{
    3000 = 0.0022; 3500 = 0.00199; 4000 = 0.00178; 4500 = 0.00157
    5000 = 0.00136; 5500 = 0.00115; 6000 = 0.00094; 6500 = 0.00073
    7000 = 0.00052; 7500 = 0.00031; 8000 = 0.0001
  }
  foreach ($step in $TelemetrySteps) {
    $value = ExpectedLearningRate $Specs[0] ([int]$step)
    if (-not [double]::IsFinite($value) -or
        [math]::Abs($value - [double]$expected[[int]$step]) -gt 1.0e-12) {
      throw "SCHEDULE_V2D_SELFTEST_FORMULA:$step"
    }
  }
  for ($i = 1; $i -lt $TelemetrySteps.Count; $i++) {
    $previous = ExpectedLearningRate $Specs[0] ([int]$TelemetrySteps[$i - 1])
    $current = ExpectedLearningRate $Specs[0] ([int]$TelemetrySteps[$i])
    if (-not ($current -lt $previous)) { throw "SCHEDULE_V2D_SELFTEST_MONOTONIC:$($TelemetrySteps[$i])" }
  }
  if ([math]::Abs((ExpectedLearningRate $Specs[0] 3000) - 0.0022) -gt 1.0e-12 -or
      [math]::Abs((ExpectedLearningRate $Specs[0] 8000) - 0.0001) -gt 1.0e-12) {
    throw 'SCHEDULE_V2D_SELFTEST_ENDPOINT'
  }
  if ((GetStartVerdict 0.0 -0.01 -0.006) -ne 'strong improvement') {
    throw 'SCHEDULE_V2D_SELFTEST_VERDICT_STRONG'
  }
  if ((GetStartVerdict 0.01 -0.02 -0.006) -ne 'flat') {
    throw 'SCHEDULE_V2D_SELFTEST_VERDICT_MIXED'
  }
  if ((GetStartVerdict 0.006 0.006 0.006) -ne 'worse') {
    throw 'SCHEDULE_V2D_SELFTEST_VERDICT_WORSE'
  }
  $parentProbe = [ordered]@{
    source_trial_id = 'parent'; sha256 = 'sha256:parent'
    parameter_hash = 'fnv1a64:parent'; step = 3000
  }
  $wrongManifest = [ordered]@{
    trial_id = $Specs[0].trial_id; target_lr = '0.0001'; peak_lr = '0.0022'
    schedule_type = 'wrong'; runner_schedule = 'wrong_decay'
    parent_trial_id = 'parent'; parent_checkpoint_hash = 'sha256:parent'
    parent_parameter_hash = 'fnv1a64:parent'; parent_step = 3000; fork_step = 3000
    decay_start_step = 3000; decay_end_step = 8000; schedule_total_steps = 8000
    reused_prefix_steps = 3000; experiment_fork = $true
    fixed_config = $Fixed
  }
  $rejected = $false
  try { ValidateManifestIdentity $wrongManifest $Specs[0] $parentProbe } catch {
    $rejected = $_.Exception.Message -match 'schedule_type'
  }
  if (-not $rejected) { throw 'SCHEDULE_V2D_SELFTEST_WRONG_SCHEDULE_RESUME' }
  $badHash = $false
  try { AssertParentHash 'sha256:expected' 'sha256:wrong' } catch {
    $badHash = $_.Exception.Message -eq 'PARENT_CHECKPOINT_HASH_MISMATCH'
  }
  if (-not $badHash) { throw 'SCHEDULE_V2D_SELFTEST_PARENT_HASH' }
  Write-Host 'run_nicopedia_hpo_schedule_v2d_self_test=PASS'
  exit 0
}

switch ($Mode) {
  'Plan' { InvokePlan }
  'Run' { InvokeRun }
  'Summarize' { InvokeSummarize }
}
