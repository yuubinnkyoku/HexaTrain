# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# HexaTrain HPO Schedule-v2b: close the lower target-LR boundary.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Run',
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2b',
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
if (-not $Ledger.StartsWith($BuildRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Ledger root must resolve below build' }
$TrialsRoot = Join-Path $Ledger 'trials'
$SummaryPath = Join-Path $Ledger 'summary.csv'
$PlanPath = Join-Path $Ledger 'plan.json'
$ComputePath = Join-Path $Ledger 'compute.json'
$FullTailPath = Join-Path $Ledger 'full-tail-comparison.csv'
$TrainingRunner = Join-Path $Root 'scripts/run_nicopedia_htp_training.ps1'
$EvalRunner = Join-Path $Root 'scripts/run_nicopedia_htp_eval.ps1'
$DataRoot = Join-Path $Root 'build/private-data/nicopedia-real-text-bpe-v1024'
$TokenizerPath = Join-Path $DataRoot 'tokenizer/byte-bpe-v1024.model'
$TrainCache = Join-Path $DataRoot 'caches/train_pilot.bin'
$EvalCacheRoot = Join-Path $DataRoot 'caches'
$AnchorLedger = Join-Path $Root 'build/hpo/nicopedia-v1024-d64-f128/lr-v1'
$AnchorTrial = Join-Path $AnchorLedger 'trials/hpo-lr-v1-lr0p0022-seed1'
$V2aLedger = Join-Path $Root 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2a'
$V2aProxyTrial = Join-Path $V2aLedger 'trials/hpo-schedule-v2a-t0400-seed1'
$V2aFullTrial = Join-Path $V2aLedger 'trials/hpo-schedule-v2a-full-s4000-t0400-seed1'

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
$AllowedTargets = @('0.0004', '0.0002', '0.0001', '0.0000')
$Specs = @(
  [ordered]@{ name = 'T0400'; target = '0.0004'; trial_id = 'hpo-schedule-v2b-t0400-seed1'; start = 6000; reused = $true }
  [ordered]@{ name = 'T0200'; target = '0.0002'; trial_id = 'hpo-schedule-v2b-t0200-seed1'; start = 6000; reused = $false }
  [ordered]@{ name = 'T0100'; target = '0.0001'; trial_id = 'hpo-schedule-v2b-t0100-seed1'; start = 6000; reused = $false }
  [ordered]@{ name = 'T0000'; target = '0.0000'; trial_id = 'hpo-schedule-v2b-t0000-seed1'; start = 6000; reused = $false }
)

function NowUtc { [DateTime]::UtcNow.ToString('o') }
function TrialDir([string]$id) { Join-Path $TrialsRoot $id }
function ManifestPath([string]$id) { Join-Path (TrialDir $id) 'manifest.json' }
function ReadKeyValue([string]$path) {
  $map = [ordered]@{}
  foreach ($line in Get-Content -LiteralPath $path) { if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $map[$Matches[1]] = $Matches[2].Trim() } }
  return $map
}
function LoadManifest([string]$id) {
  $path = ManifestPath $id
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}
function SaveManifest([string]$id, [System.Collections.IDictionary]$manifest) {
  $dir = TrialDir $id; [IO.Directory]::CreateDirectory($dir) | Out-Null
  $manifest | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath (ManifestPath $id) -Encoding utf8
}
function Balanced([System.Collections.IDictionary]$map) { return ([double]$map.validation_bits_per_utf8_byte + [double]$map.development_bits_per_utf8_byte) / 2.0 }
function ExpectedLearningRate([double]$target, [int]$step, [int]$start = 6000) {
  if ($step -le $start) { return 0.0022 }
  if ($step -ge 8000) { return $target }
  return 0.0022 + (($step - $start) / [double](8000 - $start)) * ($target - 0.0022)
}
function CheckpointName([int]$step) { "htp-seed1-l19-t32-d64-f128-step$step.ckpt" }
function EvalName([int]$step) { "seed1-l19-t32-d64-f128-step$step-v256-d256-htp.txt" }
function AssertTarget([string]$target) { if ($AllowedTargets -notcontains $target) { throw "TARGET_LR_NOT_ALLOWED:$target" } }
function AssertParentHash([string]$expected, [string]$actual) { if ($expected -ne $actual) { throw 'PARENT_CHECKPOINT_HASH_MISMATCH' } }

function ValidateEval([string]$path, [int]$step) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "EVAL_MISSING:$path" }
  $m = ReadKeyValue $path
  foreach ($key in @('status','checkpoint_step','checkpoint_parameter_hash','checkpoint_finite','qnn_return_code_success','output_tensors_finite','cpu_fallback','validation_nonfinite_chunks','development_nonfinite_chunks','validation_bits_per_utf8_byte','development_bits_per_utf8_byte','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) { if (-not $m.Contains($key)) { throw "EVAL_FIELD_MISSING:$key" } }
  if ($m.status -ne 'SUCCESS' -or [int]$m.checkpoint_step -ne $step -or $m.checkpoint_finite -ne 'true' -or $m.qnn_return_code_success -ne 'true' -or $m.output_tensors_finite -ne 'true' -or $m.cpu_fallback -ne 'false' -or $m.validation_nonfinite_chunks -ne '0' -or $m.development_nonfinite_chunks -ne '0' -or $m.api_trace_graph_execute_failure_count -ne '0' -or $m.api_trace_fallback_attempted -ne 'false' -or $m.api_trace_fallback_succeeded -ne 'false') { throw "EVAL_HEALTH_REJECTED:$path" }
  if ($m.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "EVAL_HASH_INVALID:$path" }
  return $m
}
function ValidateTrainingMap([System.Collections.IDictionary]$map, [System.Collections.IDictionary]$spec, [int]$steps, [int]$resume) {
  foreach ($key in @('status','completed_steps','run_completed_steps','qnn_return_code_success','output_tensors_finite','cpu_fallback','final_finite','all_steps_finite','learning_rate_schedule','learning_rate_decay_start_step','learning_rate_decay_end_step','learning_rate_schedule_total_steps','learning_rate_peak','learning_rate_target','experiment_fork','parent_learning_rate','resume_from_step','run_target_tokens_seen')) { if (-not $map.Contains($key)) { throw "TRAINING_FIELD_MISSING:$key" } }
  if ($map.status -ne 'SUCCESS' -or [int]$map.completed_steps -ne $steps -or [int]$map.run_completed_steps -ne ($steps - $resume) -or $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or $map.cpu_fallback -ne 'false' -or $map.final_finite -ne 'true' -or $map.all_steps_finite -ne 'true' -or $map.learning_rate_schedule -ne 'linear_decay' -or [int]$map.learning_rate_decay_start_step -ne $spec.start -or [int]$map.learning_rate_decay_end_step -ne 8000 -or [int]$map.learning_rate_schedule_total_steps -ne 8000 -or [single]$map.learning_rate_peak -ne [single]'0.0022' -or [single]$map.learning_rate_target -ne [single]$spec.target -or $map.experiment_fork -ne 'true' -or [single]$map.parent_learning_rate -ne [single]'0.0022' -or [int]$map.resume_from_step -ne $resume) { throw "TRAINING_HEALTH_OR_SCHEDULE_REJECTED:$($spec.name):$steps" }
  $expectedTokens = [int64](($steps - $resume) * $Fixed.batch_size * $Fixed.tokens)
  if ([int64]$map.run_target_tokens_seen -ne $expectedTokens) { throw "EXPOSURE_RUN_TOKENS_MISMATCH:$($spec.name):$steps" }
  if ($steps -eq 8000) { foreach ($pair in @(@('target_tokens_seen', $Fixed.planned_target_tokens), @('target_utf8_bytes_seen', $Fixed.expected_original_utf8_bytes), @('unique_chunks_seen', $Fixed.expected_chunks), @('unique_articles_seen', $Fixed.expected_articles))) { if (-not $map.Contains($pair[0]) -or [int64]$map[$pair[0]] -ne [int64]$pair[1]) { throw "EXPOSURE_TOTAL_MISMATCH:$($pair[0])" } } }
}
function AssertParentCheckpoint([int]$step) {
  $manifestPath = Join-Path $AnchorTrial 'manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'PARENT_MANIFEST_MISSING' }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
  if ([int]$manifest.completed_steps -ne 8000 -or [int]$manifest.current_steps -ne 8000 -or $manifest.status -ne 'COMPLETED') { throw 'PARENT_COMPLETION_INVALID' }
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','dataset_cache_content_hash','training_order_hash','lr_schedule','learning_rate')) {
    $expected = if ($key -eq 'lr_schedule') { 'constant' } elseif ($key -eq 'learning_rate') { '0.0022' } else { [string]$Fixed[$key] }
    if (-not $manifest.Contains($key) -or [string]$manifest[$key] -ne $expected) { throw "PARENT_MANIFEST_IDENTITY_MISMATCH:$key" }
  }
  $reportMap = ReadKeyValue (Join-Path (Join-Path $AnchorTrial 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt')
  foreach ($key in @('qnn_return_code_success','output_tensors_finite','cpu_fallback','final_finite','all_steps_finite','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) { if (-not $reportMap.Contains($key)) { throw "PARENT_HEALTH_FIELD_MISSING:$key" } }
  if ($reportMap.qnn_return_code_success -ne 'true' -or $reportMap.output_tensors_finite -ne 'true' -or $reportMap.cpu_fallback -ne 'false' -or $reportMap.final_finite -ne 'true' -or $reportMap.all_steps_finite -ne 'true' -or $reportMap.api_trace_graph_execute_failure_count -ne '0' -or $reportMap.api_trace_fallback_attempted -ne 'false' -or $reportMap.api_trace_fallback_succeeded -ne 'false') { throw 'PARENT_QNN_HEALTH_REJECTED' }
  $path = Join-Path (Join-Path $AnchorTrial 'training') (CheckpointName $step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "PARENT_CHECKPOINT_MISSING:$step" }
  $header = Get-PhoneLmCheckpointHeaders -Path $path
  if ($header.Magic -ne 'NPRTCKPTV3' -or $header.Vocabulary -ne 1024 -or $header.Tokens -ne 32 -or $header.Dimension -ne 64 -or $header.FeedForward -ne 128 -or $header.Layers -ne 19 -or $header.Heads -ne 2 -or $header.Seed -ne 1 -or $header.Step -ne $step -or $header.TokenizerKind -ne $Fixed.tokenizer_kind -or $header.TokenizerHash -ne $Fixed.tokenizer_hash) { throw "PARENT_CHECKPOINT_IDENTITY_MISMATCH:$step" }
  $evalExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'; $validation = Join-Path $DataRoot 'caches/validation.bin'; $development = Join-Path $DataRoot 'caches/development.bin'
  if (-not (Test-Path -LiteralPath $evalExe -PathType Leaf) -or -not (Test-Path -LiteralPath $validation -PathType Leaf) -or -not (Test-Path -LiteralPath $development -PathType Leaf)) { throw 'PARENT_HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE' }
  $probe = & $evalExe $path $validation $development 1 1
  if ($LASTEXITCODE -ne 0) { throw "PARENT_HOST_CHECKPOINT_EVALUATOR_FAILED:$step" }
  $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join "`n")
  if ($probeMap.step -ne [string]$step -or $probeMap.seed -ne '1' -or $probeMap.layers -ne '19' -or $probeMap.dimension -ne '64' -or $probeMap.feed_forward_dimension -ne '128' -or $probeMap.finite -ne 'true' -or $probeMap.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "PARENT_HOST_IDENTITY_MISMATCH:$step" }
  $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  return [ordered]@{ path = $path; step = $step; sha256 = "sha256:$sha"; parameter_hash = $probeMap.parameter_hash; source_trial_id = 'hpo-lr-v1-lr0p0022-seed1'; source_manifest = $manifestPath }
}
function AssertV2aSource {
  $proxyManifestPath = Join-Path $V2aProxyTrial 'manifest.json'; $proxyEvalPath = Join-Path $V2aProxyTrial 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
  $fullManifestPath = Join-Path $V2aFullTrial 'manifest.json'; $fullEvalPath = Join-Path $V2aFullTrial 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
  foreach ($p in @($proxyManifestPath,$proxyEvalPath,$fullManifestPath,$fullEvalPath)) { if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "V2A_SOURCE_MISSING:$p" } }
  $proxy = Get-Content $proxyManifestPath -Raw | ConvertFrom-Json -AsHashtable; $full = Get-Content $fullManifestPath -Raw | ConvertFrom-Json -AsHashtable
  if ($proxy.status -ne 'COMPLETED' -or [int]$proxy.completed_steps -ne 8000 -or [string]$proxy.target_lr -ne '0.0004' -or [int]$proxy.parent_step -ne 6000) { throw 'V2A_PROXY_ANCHOR_INVALID' }
  if ($full.status -ne 'COMPLETED' -or [int]$full.completed_steps -ne 8000 -or [string]$full.target_lr -ne '0.0004' -or [int]$full.parent_step -ne 4000 -or [int]$full.decay_start_step -ne 4000) { throw 'V2A_FULL_ANCHOR_INVALID' }
  $proxyEval = ValidateEval $proxyEvalPath 8000; $fullEval = ValidateEval $fullEvalPath 8000
  if ([math]::Abs(([double]$proxyEval.validation_bits_per_utf8_byte) - 2.294507954) -gt 1e-9 -or [math]::Abs(([double]$proxyEval.development_bits_per_utf8_byte) - 2.559579893) -gt 1e-9 -or [math]::Abs((Balanced $proxyEval) - 2.427043924) -gt 1e-9) { throw 'V2A_PROXY_ANCHOR_METRIC_MISMATCH' }
  if ([math]::Abs(([double]$fullEval.validation_bits_per_utf8_byte) - 2.291483826) -gt 1e-9 -or [math]::Abs(([double]$fullEval.development_bits_per_utf8_byte) - 2.560856598) -gt 1e-9 -or [math]::Abs((Balanced $fullEval) - 2.426170212) -gt 1e-9) { throw 'V2A_FULL_ANCHOR_METRIC_MISMATCH' }
  return [ordered]@{ proxy_manifest = $proxyManifestPath; proxy_eval = $proxyEvalPath; proxy = $proxyEval; full_manifest = $fullManifestPath; full_eval = $fullEvalPath; full = $fullEval }
}
function NewManifest([System.Collections.IDictionary]$spec, [System.Collections.IDictionary]$parent) {
  return [ordered]@{ schema_version = 1; experiment = 'HexaTrain HPO Schedule-v2b'; trial_id = $spec.trial_id; status = 'PENDING'; created_utc = (NowUtc); fixed_config = $Fixed; target_lr = $spec.target; target_name = $spec.name; peak_lr = '0.0022'; decay_start_step = $spec.start; decay_end_step = 8000; schedule_type = 'linear'; schedule_total_steps = 8000; warmup_steps = 0; experiment_fork = $true; parent_trial_id = $parent.source_trial_id; parent_checkpoint_path = $parent.path; parent_checkpoint_hash = $parent.sha256; parent_parameter_hash = $parent.parameter_hash; parent_step = $parent.step; fork_step = $parent.step; parent_learning_rate = '0.0022'; reused_prefix_steps = $parent.step; actual_new_steps = 0; completed_steps = 0; checkpoint_path = ''; checkpoint_parameter_hash = ''; smoke_health = 'PENDING'; final_health = 'PENDING'; git_revision = (& git -C $Root rev-parse HEAD).Trim() }
}
function EnsureManifestIdentity([System.Collections.IDictionary]$spec, [System.Collections.IDictionary]$parent) {
  AssertTarget $spec.target
  $m = LoadManifest $spec.trial_id
  if ($null -eq $m) { return (NewManifest $spec $parent) }
  if ([string]$m.target_lr -ne $spec.target -or [int]$m.parent_step -ne $parent.step -or [string]$m.parent_checkpoint_hash -ne $parent.sha256 -or [string]$m.parent_trial_id -ne $parent.source_trial_id -or [int]$m.decay_start_step -ne $spec.start) { throw "EXISTING_MANIFEST_IDENTITY_MISMATCH:$($spec.trial_id)" }
  return $m
}
function CopyParentCheckpoint([System.Collections.IDictionary]$spec, [System.Collections.IDictionary]$parent) {
  $training = Join-Path (TrialDir $spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null; $destination = Join-Path $training (CheckpointName $parent.step)
  if (Test-Path -LiteralPath $destination -PathType Leaf) { AssertParentHash $parent.sha256 ('sha256:' + (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()) } else { Copy-Item -LiteralPath $parent.path -Destination $destination -Force }
}
function InvokeTraining([System.Collections.IDictionary]$spec, [int]$steps, [int]$resume, [string]$phase, [switch]$smoke, [int]$CheckpointStallSeconds = 7200, [int]$PollLimit = 7200) {
  $training = Join-Path (TrialDir $spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null; if ($CheckpointStallSeconds -lt 1 -or $PollLimit -lt 1) { throw 'TRAINING_POLL_CONFIGURATION_INVALID' }; $runId = "$($spec.trial_id)-$phase-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"; if ($runId.Length -gt 63) { $runId = $runId.Substring(0,63) }
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-Seed',1,'-Layers',19,'-Steps',$steps,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-BatchSize',8,'-LearningRate','0.0022','-LearningRateSchedule','linear_decay','-DecayStartStep',$spec.start,'-DecayEndStep',8000,'-ScheduleTotalSteps',8000,'-TargetLearningRate',$spec.target,'-ParentLearningRate','0.0022','-ExperimentFork','-ResumeStep',$resume,'-CheckpointInterval',250,'-CheckpointStallSeconds',$CheckpointStallSeconds,'-PollLimit',$PollLimit,'-CachePath',$TrainCache,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$training,'-RunId',$runId)
  if ($SkipBuild -or $script:Prepared) { $args += '-SkipBuild' }; if ($SkipInstall -or $script:Prepared) { $args += '-SkipInstall' }; if ($smoke) { $args += '-AllowQualityFailure' }
  $sw = [Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $TrainingRunner @args | Out-Host; $code = $LASTEXITCODE; $sw.Stop(); if ($code -ne 0) { throw "TRAINING_FAILED:$($spec.name):$phase:exit=$code" }
  $reportPath = Join-Path $training "seed1-l19-v1024-t32-d64-f128-steps$steps-result.txt"; if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "TRAINING_REPORT_MISSING:$phase" }; $map = ReadKeyValue $reportPath; ValidateTrainingMap $map $spec $steps $resume
  return [ordered]@{ report = $reportPath; map = $map; wall_ms = $sw.Elapsed.TotalMilliseconds }
}
function WriteTelemetryAnchors([System.Collections.IDictionary]$spec, [int]$resume) {
  $trainingDir = Join-Path (TrialDir $spec.trial_id) 'training'; $path = Join-Path $trainingDir 'learning-rate-telemetry.csv'; if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'LEARNING_RATE_TELEMETRY_MISSING' }
  $actual = @{}
  # A recovered child may have a prefix telemetry fragment from the
  # interrupted run.  Merge it for anchors before the resume boundary while
  # preferring the current segment's runtime rows on duplicate steps.
  foreach ($prior in @(Get-ChildItem -LiteralPath $trainingDir -Filter 'learning-rate-telemetry-prior-*.csv' -File -ErrorAction SilentlyContinue)) { foreach ($row in @(Import-Csv -LiteralPath $prior.FullName)) { $actual[[int]$row.step] = [double]$row.scheduled_lr } }
  foreach ($row in @(Import-Csv -LiteralPath $path)) { $actual[[int]$row.step] = [double]$row.scheduled_lr }
  $rows = foreach ($step in @(6000,6250,6500,6750,7000,7250,7500,7750,8000)) {
    $expected = ExpectedLearningRate ([double]$spec.target) $step $spec.start
    $source = 'runtime_telemetry'
    if ($step -le $resume -and $step -le $spec.start) { $value = 0.0022; $source = 'validated_parent_constant_lr' }
    elseif ($actual.ContainsKey($step)) { $value = $actual[$step] }
    elseif ($step -eq $resume) { $value = $expected; $source = 'validated_resume_checkpoint_boundary_formula' }
    else { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISSING:$step" }
    if ([math]::Abs($value - $expected) -gt 2.0e-8) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISMATCH:$step" }
    [pscustomobject]@{ step = $step; expected_lr = $expected; actual_lr = $value; source = $source }
  }
  $out = Join-Path (Join-Path (TrialDir $spec.trial_id) 'training') 'schedule-telemetry-anchors.csv'; @($rows) | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8; return $out
}
function InvokeEval([System.Collections.IDictionary]$spec, [int]$step) {
  $training = Join-Path (TrialDir $spec.trial_id) 'training'; $checkpoint = Join-Path $training (CheckpointName $step); $evalDir = Join-Path (TrialDir $spec.trial_id) "eval/step-$step-v256-d256"; [IO.Directory]::CreateDirectory($evalDir) | Out-Null; $path = Join-Path $evalDir (EvalName $step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $evalRunId = "$($spec.trial_id)-eval-$step-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"; if ($evalRunId.Length -gt 63) { $evalRunId = $evalRunId.Substring(0,63) }; & pwsh -NoProfile -File $EvalRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -SkipBuild -SkipInstall -Seed 1 -Layers 19 -Heads 2 -Tokens 32 -Vocabulary 1024 -Dimension 64 -FeedForwardDimension 128 -CheckpointStep $step -ValidationChunks 256 -DevelopmentChunks 256 -CheckpointPath $checkpoint -CacheRoot $EvalCacheRoot -TokenizerModelPath $TokenizerPath -ReportRoot $evalDir -RunId $evalRunId | Out-Host; if ($LASTEXITCODE -ne 0) { throw "EVAL_FAILED:$($spec.name):$step" } }
  $m = ValidateEval $path $step; return [ordered]@{ path = $path; map = $m; val = [double]$m.validation_bits_per_utf8_byte; dev = [double]$m.development_bits_per_utf8_byte; balanced = (Balanced $m); health = 'PASS' }
}
function FindPartialTailResumeStep([System.Collections.IDictionary]$spec) {
  # A host instrumentation timeout may leave a verified interval checkpoint
  # on-device while the child manifest is still FAILED.  Resume only from a
  # checkpoint in this exact child directory, and only when the private
  # partial marker identifies an interrupted RUNNING status.  Never infer a
  # cross-target/cross-parent resume from a foreign artifact.
  $training = Join-Path (TrialDir $spec.trial_id) 'training'
  $partialPath = Join-Path $training 'seed1-l19-v1024-t32-d64-f128-steps8000-partial-status.json'
  $finalReportPath = Join-Path $training 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
  if (Test-Path -LiteralPath $finalReportPath -PathType Leaf) {
    $finalMap = ReadKeyValue $finalReportPath
    if ($finalMap.Contains('resume_from_step') -and [int]$finalMap.resume_from_step -ge $spec.start -and [int]$finalMap.resume_from_step -lt 8000) { return [int]$finalMap.resume_from_step }
  }
  if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf)) { return 6000 }
  $partial = Get-Content -LiteralPath $partialPath -Raw | ConvertFrom-Json -AsHashtable
  if ([string]$partial.status -ne 'RUNNING' -or [string]$partial.run_id -notlike "$($spec.trial_id)-final-*") { return 6000 }
  $candidates = @(Get-ChildItem -LiteralPath $training -Filter 'htp-seed1-l19-t32-d64-f128-step*.ckpt' -File |
    ForEach-Object { if ($_.Name -match 'step(\d+)\.ckpt$') { [int]$Matches[1] } } |
    Where-Object { $_ -gt $spec.start -and $_ -lt 8000 } | Sort-Object -Descending)
  foreach ($step in $candidates) {
    $path = Join-Path $training (CheckpointName $step)
    $header = Get-PhoneLmCheckpointHeaders -Path $path
    if ($header.Magic -eq 'NPRTCKPTV3' -and $header.Vocabulary -eq 1024 -and $header.Tokens -eq 32 -and $header.Dimension -eq 64 -and $header.FeedForward -eq 128 -and $header.Layers -eq 19 -and $header.Heads -eq 2 -and $header.Seed -eq 1 -and $header.Step -eq $step -and $header.TokenizerKind -eq $Fixed.tokenizer_kind -and $header.TokenizerHash -eq $Fixed.tokenizer_hash) {
      $probe = & (Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe') $path (Join-Path $DataRoot 'caches/validation.bin') (Join-Path $DataRoot 'caches/development.bin') 1 1
      if ($LASTEXITCODE -eq 0) {
        $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join "`n")
        if ($probeMap.step -eq [string]$step -and $probeMap.seed -eq '1' -and $probeMap.layers -eq '19' -and $probeMap.finite -eq 'true') { return $step }
      }
    }
  }
  return 6000
}
function NewProxyRow([System.Collections.IDictionary]$spec, [System.Collections.IDictionary]$eval, [string]$status) { return [pscustomobject]@{ phase = 'proxy'; target_name = $spec.name; target_lr = $spec.target; trial_id = $spec.trial_id; val_bpb = $eval.val; dev_bpb = $eval.dev; balanced_bpb = $eval.balanced; delta_vs_0004 = $null; status = $status; checkpoint_hash = $eval.map.checkpoint_parameter_hash; qnn_health = 'PASS' } }
function InvokePlan {
  [IO.Directory]::CreateDirectory($TrialsRoot) | Out-Null; $parent = AssertParentCheckpoint 6000; $source = AssertV2aSource; $plan = [ordered]@{ schema_version = 1; experiment = 'HexaTrain HPO Schedule-v2b'; motivation = 'Close the lower target-LR boundary after the monotonic Schedule-v2a result'; fixed_config = $Fixed; common_parent = $parent; v2a_source_of_truth = $source; targets = $Specs; peak_lr = '0.0022'; decay_start_step = 6000; decay_end_step = 8000; schedule_type = 'linear'; warmup_steps = 0; proxy_steps = 2000; primary_eval = 'step8000 exact 256+256'; created_utc = (NowUtc) }; $plan | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $PlanPath -Encoding utf8
  foreach ($spec in $Specs) { $m = EnsureManifestIdentity $spec $parent; if ($spec.reused) { $m.status = 'REUSED'; $m.artifact_trial_id = 'hpo-schedule-v2a-t0400-seed1'; $m.artifact_eval_path = (Join-Path $V2aProxyTrial 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt') }; SaveManifest $spec.trial_id $m }; return [ordered]@{ parent = $parent; source = $source }
}
function InvokeRun {
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'Run requires explicit QairtSdkRoot and ExpectedBuildId' }; Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  foreach ($path in @($TrainCache,$TokenizerPath)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "HPO_PRIVATE_INPUT_MISSING:$path" } }
  $setup = InvokePlan; $parent = $setup.parent; $source = $setup.source; $script:Prepared = $false; $rows = [Collections.Generic.List[object]]::new(); $wall = [Diagnostics.Stopwatch]::StartNew(); $proxyNewSteps = 0
  $anchorSpec = $Specs[0]; $anchorEval = [ordered]@{ path = $source.proxy_eval; map = $source.proxy; val = [double]$source.proxy.validation_bits_per_utf8_byte; dev = [double]$source.proxy.development_bits_per_utf8_byte; balanced = (Balanced $source.proxy); health = 'PASS' }; $anchorManifest = LoadManifest $anchorSpec.trial_id; $anchorManifest.primary = [ordered]@{ val_bpb = $anchorEval.val; dev_bpb = $anchorEval.dev; balanced_bpb = $anchorEval.balanced; health = 'PASS'; checkpoint_hash = $anchorEval.map.checkpoint_parameter_hash }; SaveManifest $anchorSpec.trial_id $anchorManifest; $rows.Add((NewProxyRow $anchorSpec $anchorEval 'REUSED'))
  foreach ($spec in @($Specs | Where-Object { -not $_.reused })) {
    $manifest = EnsureManifestIdentity $spec $parent; SaveManifest $spec.trial_id $manifest; CopyParentCheckpoint $spec $parent | Out-Null
    try {
      $evalPath = Join-Path (TrialDir $spec.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
      if ($manifest.status -eq 'COMPLETED' -and [int]$manifest.completed_steps -eq 8000 -and $manifest.Contains('primary') -and (Test-Path -LiteralPath $evalPath -PathType Leaf)) { $manifest.failure = $null; SaveManifest $spec.trial_id $manifest; $eval = InvokeEval $spec 8000; $rows.Add((NewProxyRow $spec $eval 'REUSED')); $proxyNewSteps += 2000; $script:Prepared = $true; continue }
      $smokeReport = Join-Path (Join-Path (TrialDir $spec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps6004-result.txt'
      if (-not (Test-Path -LiteralPath $smokeReport -PathType Leaf)) { $smoke = InvokeTraining $spec 6004 6000 'smoke' -Smoke; $manifest = LoadManifest $spec.trial_id; $manifest.smoke_health = 'PASS'; $manifest.smoke_report = $smoke.report; SaveManifest $spec.trial_id $manifest; $script:Prepared = $true }
      $resumeForFinal = FindPartialTailResumeStep $spec; if ($resumeForFinal -gt 6000) { Write-Host "RESUME_PARTIAL_TAIL target=$($spec.target) resume_step=$resumeForFinal" }
      $finalReportPath = Join-Path (Join-Path (TrialDir $spec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
      $run = if (Test-Path -LiteralPath $finalReportPath -PathType Leaf) {
        $existingMap = ReadKeyValue $finalReportPath; ValidateTrainingMap $existingMap $spec 8000 $resumeForFinal; [ordered]@{ report = $finalReportPath; map = $existingMap; wall_ms = 0 }
      } else { InvokeTraining $spec 8000 $resumeForFinal 'final' }
      $telemetry = WriteTelemetryAnchors $spec $resumeForFinal; $manifest = LoadManifest $spec.trial_id; $manifest.status = 'COMPLETED'; $manifest.failure = $null; $manifest.completed_steps = 8000; $manifest.actual_new_steps = 2000; $manifest.resume_step_used = $resumeForFinal; $manifest.resumed_tail_steps = 8000 - $resumeForFinal; $manifest.checkpoint_path = Join-Path (TrialDir $spec.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; $manifest.checkpoint_parameter_hash = $run.map.final_parameter_hash; $manifest.final_health = 'PASS'; $manifest.wall_time_ms = $run.wall_ms; $manifest.training_total_seconds = [double]$run.map.training_total_seconds; $manifest.training_step_ms = [double]$run.map.training_step_ms; $manifest.run_bytes_per_second = [math]::Round(([double]$run.map.run_target_utf8_bytes_seen / [double]$run.map.training_total_seconds), 6); $manifest.run_target_tokens = [int64]$run.map.run_target_tokens_seen; $manifest.run_target_utf8_bytes = [int64]$run.map.run_target_utf8_bytes_seen; $manifest.total_target_tokens = [int64]$run.map.target_tokens_seen; $manifest.total_target_utf8_bytes = [int64]$run.map.target_utf8_bytes_seen; $manifest.unique_chunks = [int64]$run.map.unique_chunks_seen; $manifest.unique_articles = [int64]$run.map.unique_articles_seen; $manifest.telemetry_anchors_path = $telemetry; SaveManifest $spec.trial_id $manifest
      $eval = InvokeEval $spec 8000; $manifest = LoadManifest $spec.trial_id; $manifest.primary = [ordered]@{ val_bpb = $eval.val; dev_bpb = $eval.dev; balanced_bpb = $eval.balanced; health = 'PASS'; checkpoint_hash = $eval.map.checkpoint_parameter_hash }; SaveManifest $spec.trial_id $manifest; $rows.Add((NewProxyRow $spec $eval 'COMPLETED')); $proxyNewSteps += 2000; $script:Prepared = $true
    } catch { $manifest = LoadManifest $spec.trial_id; $manifest.status = 'FAILED'; $manifest.failure = $_.Exception.Message; SaveManifest $spec.trial_id $manifest; throw }
  }
  $anchorRow = @($rows | Where-Object { $_.target_lr -eq '0.0004' })[0]; $newRows = @($rows | Where-Object { $_.target_lr -ne '0.0004' }); $proxyBest = @($newRows | Sort-Object balanced_bpb)[0]; $improvement = [double]$anchorRow.balanced_bpb - [double]$proxyBest.balanced_bpb; $bothImprove = ([double]$proxyBest.val_bpb -lt [double]$anchorRow.val_bpb) -and ([double]$proxyBest.dev_bpb -lt [double]$anchorRow.dev_bpb)
  $fullExecuted = $false; $fullBest = $null; $fullAnchor = $null
  if ($improvement -ge 0.005 -or $bothImprove) {
    $parent4000 = AssertParentCheckpoint 4000; $fullSpec = [ordered]@{ name = "S4000_$($proxyBest.target_name)"; target = $proxyBest.target_lr; trial_id = "hpo-schedule-v2b-full-s4000-$($proxyBest.target_name.ToLowerInvariant())-seed1"; start = 4000; reused = $false }; $fullManifest = EnsureManifestIdentity $fullSpec $parent4000; SaveManifest $fullSpec.trial_id $fullManifest; CopyParentCheckpoint $fullSpec $parent4000 | Out-Null
    $fullEvalPath = Join-Path (TrialDir $fullSpec.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
    if (-not ($fullManifest.status -eq 'COMPLETED' -and $fullManifest.Contains('primary') -and (Test-Path -LiteralPath $fullEvalPath -PathType Leaf))) {
      $fullSmoke = Join-Path (Join-Path (TrialDir $fullSpec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps4004-result.txt'
      if (-not (Test-Path -LiteralPath $fullSmoke -PathType Leaf)) { $null = InvokeTraining $fullSpec 4004 4000 'smoke' -Smoke; $script:Prepared = $true }
      $fullResume = FindPartialTailResumeStep $fullSpec
      $fullReportPath = Join-Path (Join-Path (TrialDir $fullSpec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
      $fullRun = if (Test-Path -LiteralPath $fullReportPath -PathType Leaf) {
        $existingFullMap = ReadKeyValue $fullReportPath; ValidateTrainingMap $existingFullMap $fullSpec 8000 $fullResume; [ordered]@{ report = $fullReportPath; map = $existingFullMap; wall_ms = 0 }
      } else { InvokeTraining $fullSpec 8000 $fullResume 'final' -CheckpointStallSeconds 21600 -PollLimit 43200 }
      $fullTelemetry = WriteTelemetryAnchors $fullSpec $fullResume; $fullManifest = LoadManifest $fullSpec.trial_id; $fullManifest.status = 'COMPLETED'; $fullManifest.failure = $null; $fullManifest.completed_steps = 8000; $fullManifest.actual_new_steps = 4000; $fullManifest.resume_step_used = $fullResume; $fullManifest.resumed_tail_steps = 8000 - $fullResume; $fullManifest.telemetry_anchors_path = $fullTelemetry; $fullManifest.checkpoint_path = Join-Path (TrialDir $fullSpec.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; $fullManifest.checkpoint_parameter_hash = $fullRun.map.final_parameter_hash; $fullManifest.final_health = 'PASS'; $fullManifest.training_total_seconds = [double]$fullRun.map.training_total_seconds; $fullManifest.training_step_ms = [double]$fullRun.map.training_step_ms; $fullManifest.run_bytes_per_second = [math]::Round(([double]$fullRun.map.run_target_utf8_bytes_seen / [double]$fullRun.map.training_total_seconds), 6); SaveManifest $fullSpec.trial_id $fullManifest
    }
    $fullEval = InvokeEval $fullSpec 8000; $fullManifest = LoadManifest $fullSpec.trial_id; $fullManifest.primary = [ordered]@{ val_bpb = $fullEval.val; dev_bpb = $fullEval.dev; balanced_bpb = $fullEval.balanced; health = 'PASS'; checkpoint_hash = $fullEval.map.checkpoint_parameter_hash }; SaveManifest $fullSpec.trial_id $fullManifest; $fullAnchor = $source.full; $fullRows = @([pscustomobject]@{ target_lr = '0.0004'; val_bpb = [double]$fullAnchor.validation_bits_per_utf8_byte; dev_bpb = [double]$fullAnchor.development_bits_per_utf8_byte; balanced_bpb = (Balanced $fullAnchor); source = 'S4000 existing v2a T0400 artifact' }, [pscustomobject]@{ target_lr = $fullSpec.target; val_bpb = $fullEval.val; dev_bpb = $fullEval.dev; balanced_bpb = $fullEval.balanced; source = $fullSpec.trial_id }); @($fullRows) | Export-Csv -LiteralPath $FullTailPath -NoTypeInformation -Encoding utf8; $fullExecuted = $true; $fullBest = $fullEval; $proxyNewSteps += 4000
  }
  foreach ($row in $rows) { $row.delta_vs_0004 = [math]::Round(([double]$row.balanced_bpb - [double]$anchorRow.balanced_bpb), 9) }; @($rows | Sort-Object phase, target_lr) | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8; $wall.Stop()
  $finalTarget = '0.0004'; $finalBalanced = [double]$anchorRow.balanced_bpb; if ($fullExecuted -and $fullBest.balanced -lt (Balanced $fullAnchor)) { $finalTarget = $proxyBest.target_lr; $finalBalanced = $fullBest.balanced }
  $boundary = if ($finalTarget -eq '0.0000') { 'zero-hit' } elseif ($finalTarget -eq '0.0004') { 'closed' } else { 'inconclusive' }; $actualNew = [int]$proxyNewSteps; $fullSteps = if ($fullExecuted) { 4000 } else { 0 }; $reusedPrefix = 18000 + $fullSteps; $compute = [ordered]@{ naive_fresh_steps = 24000; actual_new_steps = $actualNew; proxy_new_steps = 6000; full_tail_confirmation_steps = $fullSteps; smoke_probe_steps = if ($fullExecuted) { 16 } else { 12 }; reused_prefix_steps = $reusedPrefix; saved_steps = 24000 - $actualNew; saving_percent = [math]::Round((1 - ($actualNew / 24000.0)) * 100, 2); wall_time_ms = [math]::Round($wall.Elapsed.TotalMilliseconds, 1); proxy_best_target = $proxyBest.target_lr; proxy_best_balanced_bpb = [double]$proxyBest.balanced_bpb; proxy_improvement_bpb = $improvement; full_tail_executed = $fullExecuted; final_best_target = $finalTarget; final_best_balanced_bpb = $finalBalanced; boundary_verdict = $boundary; generated_utc = (NowUtc) }; $compute | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $ComputePath -Encoding utf8
}
function InvokeSummarize { if (Test-Path -LiteralPath $SummaryPath) { Get-Content -LiteralPath $SummaryPath }; if (Test-Path -LiteralPath $FullTailPath) { Get-Content -LiteralPath $FullTailPath }; if (Test-Path -LiteralPath $ComputePath) { Get-Content -LiteralPath $ComputePath } }

if ($SelfTest) {
  if ($Fixed.parameter_count -ne 758528 -or $AllowedTargets.Count -ne 4) { throw 'SCHEDULE_V2B_SELFTEST_FIXED' }
  $seen = @{}; foreach ($spec in $Specs) { AssertTarget $spec.target; if ($seen.ContainsKey($spec.trial_id)) { throw 'SCHEDULE_V2B_SELFTEST_DUPLICATE_ID' }; $seen[$spec.trial_id] = $true }
  foreach ($target in $AllowedTargets) { if ([math]::Abs((ExpectedLearningRate ([double]$target) 6000) - 0.0022) -gt 1e-12 -or [math]::Abs((ExpectedLearningRate ([double]$target) 8000) - [double]$target) -gt 1e-12) { throw "SCHEDULE_V2B_SELFTEST_ENDPOINT:$target" } }
  $zero = [double](ExpectedLearningRate 0.0 8000); if ($zero -ne 0.0) { throw 'SCHEDULE_V2B_SELFTEST_ZERO_ENDPOINT' }
  $failed = $false; try { AssertParentHash 'sha256:expected' 'sha256:wrong' } catch { $failed = $_.Exception.Message -eq 'PARENT_CHECKPOINT_HASH_MISMATCH' }; if (-not $failed) { throw 'SCHEDULE_V2B_SELFTEST_WRONG_PARENT' }
  Write-Host 'run_nicopedia_hpo_schedule_v2b_self_test=PASS'; exit 0
}
switch ($Mode) { 'Plan' { InvokePlan }; 'Run' { InvokeRun }; 'Summarize' { InvokeSummarize } }
