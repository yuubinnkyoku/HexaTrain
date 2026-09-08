# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# HexaTrain HPO Schedule-v2a: lower only the linear-decay target LR.
#
# All research artifacts remain private below build/.  The runner reuses the
# validated C2200 step-6000 checkpoint as the common fork parent and runs the
# three new targets one at a time for the 6000->8000 proxy tail.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Run',
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2a',
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
$EventsPath = Join-Path $Ledger 'trials.jsonl'
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
$ScheduleV1Ledger = Join-Path $Root 'build/hpo/nicopedia-v1024-d64-f128/schedule-v1'

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
$AllowedTargets = @('0.0015', '0.0010', '0.0007', '0.0004')
$Specs = @(
  [ordered]@{ name = 'T1500'; target = '0.0015'; trial_id = 'hpo-schedule-v2a-t1500-seed1'; start = 6000; reused = $true }
  [ordered]@{ name = 'T1000'; target = '0.0010'; trial_id = 'hpo-schedule-v2a-t1000-seed1'; start = 6000; reused = $false }
  [ordered]@{ name = 'T0700'; target = '0.0007'; trial_id = 'hpo-schedule-v2a-t0700-seed1'; start = 6000; reused = $false }
  [ordered]@{ name = 'T0400'; target = '0.0004'; trial_id = 'hpo-schedule-v2a-t0400-seed1'; start = 6000; reused = $false }
)

function NowUtc { [DateTime]::UtcNow.ToString('o') }
function TrialDir([string]$TrialId) { Join-Path $TrialsRoot $TrialId }
function ManifestPath([string]$TrialId) { Join-Path (TrialDir $TrialId) 'manifest.json' }
function ReadKeyValue([string]$Path) {
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
  $dir = TrialDir $TrialId; [IO.Directory]::CreateDirectory($dir) | Out-Null
  $Manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (ManifestPath $TrialId) -Encoding utf8
}
function WriteEvent([hashtable]$Fields) {
  [IO.Directory]::CreateDirectory($Ledger) | Out-Null
  $row = [ordered]@{ utc = NowUtc }
  foreach ($key in $Fields.Keys) { $row[$key] = $Fields[$key] }
  $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsPath -Encoding utf8
}
function Balanced([System.Collections.IDictionary]$Map) {
  return ([double]$Map.validation_bits_per_utf8_byte + [double]$Map.development_bits_per_utf8_byte) / 2.0
}
function ExpectedLearningRate([double]$Target, [int]$Step, [int]$Start = 6000) {
  if ($Step -le $Start) { return 0.0022 }
  if ($Step -ge 8000) { return $Target }
  return 0.0022 + (($Step - $Start) / [double](8000 - $Start)) * ($Target - 0.0022)
}
function CheckpointName([int]$Step) { "htp-seed1-l19-t32-d64-f128-step$Step.ckpt" }
function EvalName([int]$Step) { "seed1-l19-t32-d64-f128-step$Step-v256-d256-htp.txt" }
function AssertTarget([string]$Target) {
  if ($AllowedTargets -notcontains $Target) { throw "TARGET_LR_NOT_ALLOWED:$Target" }
}
function AssertParentHash([string]$Expected, [string]$Actual) {
  if ($Expected -ne $Actual) { throw 'PARENT_CHECKPOINT_HASH_MISMATCH' }
}
function ValidateEval([string]$Path, [int]$Step) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "EVAL_MISSING:$Path" }
  $m = ReadKeyValue $Path
  foreach ($key in @('status','checkpoint_step','checkpoint_parameter_hash','checkpoint_finite','qnn_return_code_success','output_tensors_finite','cpu_fallback','validation_nonfinite_chunks','development_nonfinite_chunks','validation_bits_per_utf8_byte','development_bits_per_utf8_byte','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) {
    if (-not $m.Contains($key)) { throw "EVAL_FIELD_MISSING:$key" }
  }
  if ($m.status -ne 'SUCCESS' -or [int]$m.checkpoint_step -ne $Step -or $m.checkpoint_finite -ne 'true' -or $m.qnn_return_code_success -ne 'true' -or $m.output_tensors_finite -ne 'true' -or $m.cpu_fallback -ne 'false' -or $m.validation_nonfinite_chunks -ne '0' -or $m.development_nonfinite_chunks -ne '0' -or $m.api_trace_graph_execute_failure_count -ne '0' -or $m.api_trace_fallback_attempted -ne 'false' -or $m.api_trace_fallback_succeeded -ne 'false') { throw "EVAL_HEALTH_REJECTED:$Path" }
  if ($m.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "EVAL_HASH_INVALID:$Path" }
  return $m
}
function ValidateTrainingMap([System.Collections.IDictionary]$Map, [System.Collections.IDictionary]$Spec, [int]$Steps, [int]$Resume) {
  foreach ($key in @('status','completed_steps','run_completed_steps','qnn_return_code_success','output_tensors_finite','cpu_fallback','final_finite','all_steps_finite','learning_rate_schedule','learning_rate_decay_start_step','learning_rate_decay_end_step','learning_rate_schedule_total_steps','learning_rate_peak','learning_rate_target','experiment_fork','parent_learning_rate','resume_from_step','run_target_tokens_seen')) {
    if (-not $Map.Contains($key)) { throw "TRAINING_FIELD_MISSING:$key" }
  }
  if ($Map.status -notin @('SUCCESS','FAILED') -or [int]$Map.completed_steps -ne $Steps -or [int]$Map.run_completed_steps -ne ($Steps - $Resume) -or $Map.qnn_return_code_success -ne 'true' -or $Map.output_tensors_finite -ne 'true' -or $Map.cpu_fallback -ne 'false' -or $Map.final_finite -ne 'true' -or $Map.all_steps_finite -ne 'true' -or $Map.learning_rate_schedule -ne 'linear_decay' -or [int]$Map.learning_rate_decay_start_step -ne $Spec.start -or [int]$Map.learning_rate_decay_end_step -ne 8000 -or [int]$Map.learning_rate_schedule_total_steps -ne 8000 -or [single]$Map.learning_rate_peak -ne [single]'0.0022' -or [single]$Map.learning_rate_target -ne [single]$Spec.target -or $Map.experiment_fork -ne 'true' -or [single]$Map.parent_learning_rate -ne [single]'0.0022' -or [int]$Map.resume_from_step -ne $Resume) { throw "TRAINING_HEALTH_OR_SCHEDULE_REJECTED:$($Spec.name):$Steps" }
  $expectedTokens = [int64](($Steps - $Resume) * $Fixed.batch_size * $Fixed.tokens)
  if ([int64]$Map.run_target_tokens_seen -ne $expectedTokens) { throw "EXPOSURE_RUN_TOKENS_MISMATCH:$($Spec.name):$Steps" }
  if ($Steps -eq 8000) {
    foreach ($pair in @(@('target_tokens_seen', $Fixed.planned_target_tokens), @('target_utf8_bytes_seen', $Fixed.expected_original_utf8_bytes), @('unique_chunks_seen', $Fixed.expected_chunks), @('unique_articles_seen', $Fixed.expected_articles))) {
      if (-not $Map.Contains($pair[0]) -or [int64]$Map[$pair[0]] -ne [int64]$pair[1]) { throw "EXPOSURE_TOTAL_MISMATCH:$($pair[0])" }
    }
  }
}
function AssertParentCheckpoint([int]$Step) {
  $manifestPath = Join-Path $AnchorTrial 'manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'PARENT_MANIFEST_MISSING' }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
  if ([int]$manifest.completed_steps -ne 8000 -or [int]$manifest.current_steps -ne 8000 -or $manifest.status -ne 'COMPLETED') { throw 'PARENT_COMPLETION_INVALID' }
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','dataset_cache_content_hash','training_order_hash','lr_schedule','learning_rate')) {
    $expected = if ($key -eq 'lr_schedule') { 'constant' } elseif ($key -eq 'learning_rate') { '0.0022' } else { [string]$Fixed[$key] }
    if (-not $manifest.Contains($key) -or [string]$manifest[$key] -ne $expected) { throw "PARENT_MANIFEST_IDENTITY_MISMATCH:$key" }
  }
  $parentReport = Join-Path (Join-Path $AnchorTrial 'training') 'seed1-l19-v1024-t32-d64-f128-steps8000-result.txt'
  $reportMap = ReadKeyValue $parentReport
  foreach ($key in @('qnn_return_code_success','output_tensors_finite','cpu_fallback','final_finite','all_steps_finite','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded')) { if (-not $reportMap.Contains($key)) { throw "PARENT_HEALTH_FIELD_MISSING:$key" } }
  if ($reportMap.qnn_return_code_success -ne 'true' -or $reportMap.output_tensors_finite -ne 'true' -or $reportMap.cpu_fallback -ne 'false' -or $reportMap.final_finite -ne 'true' -or $reportMap.all_steps_finite -ne 'true' -or $reportMap.api_trace_graph_execute_failure_count -ne '0' -or $reportMap.api_trace_fallback_attempted -ne 'false' -or $reportMap.api_trace_fallback_succeeded -ne 'false') { throw 'PARENT_QNN_HEALTH_REJECTED' }
  $path = Join-Path (Join-Path $AnchorTrial 'training') (CheckpointName $Step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "PARENT_CHECKPOINT_MISSING:$Step" }
  $header = Get-PhoneLmCheckpointHeaders -Path $path
  if ($header.Magic -ne 'NPRTCKPTV3' -or $header.Vocabulary -ne 1024 -or $header.Tokens -ne 32 -or $header.Dimension -ne 64 -or $header.FeedForward -ne 128 -or $header.Layers -ne 19 -or $header.Heads -ne 2 -or $header.Seed -ne 1 -or $header.Step -ne $Step -or $header.TokenizerKind -ne $Fixed.tokenizer_kind -or $header.TokenizerHash -ne $Fixed.tokenizer_hash) { throw "PARENT_CHECKPOINT_IDENTITY_MISMATCH:$Step" }
  $evalExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
  $validation = Join-Path $DataRoot 'caches/validation.bin'; $development = Join-Path $DataRoot 'caches/development.bin'
  if (-not (Test-Path -LiteralPath $evalExe -PathType Leaf) -or -not (Test-Path -LiteralPath $validation -PathType Leaf) -or -not (Test-Path -LiteralPath $development -PathType Leaf)) { throw 'PARENT_HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE' }
  $probe = & $evalExe $path $validation $development 1 1
  if ($LASTEXITCODE -ne 0) { throw "PARENT_HOST_CHECKPOINT_EVALUATOR_FAILED:$Step" }
  $probeMap = Get-PhoneLmKeyValueMap -Text ($probe -join "`n")
  if ($probeMap.step -ne [string]$Step -or $probeMap.seed -ne '1' -or $probeMap.layers -ne '19' -or $probeMap.dimension -ne '64' -or $probeMap.feed_forward_dimension -ne '128' -or $probeMap.finite -ne 'true' -or $probeMap.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "PARENT_HOST_IDENTITY_MISMATCH:$Step" }
  $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  return [ordered]@{ path = $path; step = $Step; sha256 = "sha256:$sha"; parameter_hash = $probeMap.parameter_hash; source_trial_id = 'hpo-lr-v1-lr0p0022-seed1'; source_manifest = $manifestPath }
}
function NewManifest([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  return [ordered]@{
    schema_version = 1; experiment = 'HexaTrain HPO Schedule-v2a'; trial_id = $Spec.trial_id; status = 'PENDING'; created_utc = (NowUtc)
    fixed_config = $Fixed; target_lr = $Spec.target; target_name = $Spec.name
    peak_lr = '0.0022'; decay_start_step = $Spec.start; decay_end_step = 8000; schedule_type = 'linear'; schedule_total_steps = 8000; warmup_steps = 0
    experiment_fork = $true; parent_trial_id = $Parent.source_trial_id; parent_checkpoint_path = $Parent.path; parent_checkpoint_hash = $Parent.sha256; parent_parameter_hash = $Parent.parameter_hash; parent_step = $Parent.step; fork_step = $Parent.step; parent_learning_rate = '0.0022'
    reused_prefix_steps = 6000; actual_new_steps = 0; completed_steps = 0; checkpoint_path = ''; checkpoint_parameter_hash = ''; smoke_health = 'PENDING'; final_health = 'PENDING'
    git_revision = (& git -C $Root rev-parse HEAD).Trim()
  }
}
function EnsureManifestIdentity([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  $m = LoadManifest $Spec.trial_id
  if ($null -eq $m) { return (NewManifest $Spec $Parent) }
  if ([string]$m.target_lr -ne $Spec.target -or [int]$m.parent_step -ne $Parent.step -or [string]$m.parent_checkpoint_hash -ne $Parent.sha256 -or [string]$m.parent_trial_id -ne $Parent.source_trial_id) { throw "EXISTING_MANIFEST_IDENTITY_MISMATCH:$($Spec.trial_id)" }
  return $m
}
function CopyParentCheckpoint([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Parent) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null
  $destination = Join-Path $training (CheckpointName $Parent.step)
  if (Test-Path -LiteralPath $destination -PathType Leaf) {
    $actual = 'sha256:' + (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    AssertParentHash $Parent.sha256 $actual
  } else { Copy-Item -LiteralPath $Parent.path -Destination $destination -Force }
  return $training
}
function InvokeTraining([System.Collections.IDictionary]$Spec, [int]$Steps, [int]$Resume, [string]$Phase, [switch]$Smoke) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null
  $runId = "$($Spec.trial_id)-$Phase-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"; if ($runId.Length -gt 63) { $runId = $runId.Substring(0, 63) }
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-Seed',1,'-Layers',19,'-Steps',$Steps,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-BatchSize',8,'-LearningRate','0.0022','-LearningRateSchedule','linear_decay','-DecayStartStep',$Spec.start,'-DecayEndStep',8000,'-ScheduleTotalSteps',8000,'-TargetLearningRate',$Spec.target,'-ParentLearningRate','0.0022','-ExperimentFork','-ResumeStep',$Resume,'-CheckpointInterval',250,'-CheckpointStallSeconds',7200,'-CachePath',$TrainCache,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$training,'-RunId',$runId)
  if ($SkipBuild -or $script:Prepared) { $args += '-SkipBuild' }; if ($SkipInstall -or $script:Prepared) { $args += '-SkipInstall' }; if ($Smoke) { $args += '-AllowQualityFailure' }
  $sw = [Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $TrainingRunner @args | Out-Host; $code = $LASTEXITCODE; $sw.Stop()
  if ($code -ne 0) { throw "TRAINING_FAILED:$($Spec.name):$Phase:exit=$code" }
  $reportPath = Join-Path $training "seed1-l19-v1024-t32-d64-f128-steps$Steps-result.txt"
  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "TRAINING_REPORT_MISSING:$Phase" }
  $map = ReadKeyValue $reportPath; ValidateTrainingMap $map $Spec $Steps $Resume
  return [ordered]@{ report = $reportPath; map = $map; wall_ms = $sw.Elapsed.TotalMilliseconds }
}
function WriteTelemetryAnchors([System.Collections.IDictionary]$Spec, [int]$Resume) {
  $path = Join-Path (Join-Path (TrialDir $Spec.trial_id) 'training') 'learning-rate-telemetry.csv'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'LEARNING_RATE_TELEMETRY_MISSING' }
  $actual = @{}; foreach ($row in @(Import-Csv -LiteralPath $path)) { $actual[[int]$row.step] = [double]$row.scheduled_lr }
  $rows = foreach ($step in @(6000,6250,6500,6750,7000,7250,7500,7750,8000)) {
    $expected = ExpectedLearningRate ([double]$Spec.target) $step $Spec.start
    if ($step -le $Resume) { $value = 0.0022; $source = 'validated_parent_constant_lr' } elseif (-not $actual.ContainsKey($step)) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISSING:$step" } else { $value = $actual[$step]; $source = 'runtime_telemetry' }
    if ([math]::Abs($value - $expected) -gt 2.0e-8) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISMATCH:$step" }
    [pscustomobject]@{ step = $step; expected_lr = $expected; actual_lr = $value; source = $source }
  }
  $out = Join-Path (Join-Path (TrialDir $Spec.trial_id) 'training') 'schedule-telemetry-anchors.csv'; @($rows) | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8; return $out
}
function InvokeEval([System.Collections.IDictionary]$Spec, [int]$Step) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'; $checkpoint = Join-Path $training (CheckpointName $Step); $evalDir = Join-Path (TrialDir $Spec.trial_id) "eval/step-$Step-v256-d256"; [IO.Directory]::CreateDirectory($evalDir) | Out-Null
  $path = Join-Path $evalDir (EvalName $Step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    & pwsh -NoProfile -File $EvalRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -SkipBuild -SkipInstall -Seed 1 -Layers 19 -Heads 2 -Tokens 32 -Vocabulary 1024 -Dimension 64 -FeedForwardDimension 128 -CheckpointStep $Step -ValidationChunks 256 -DevelopmentChunks 256 -CheckpointPath $checkpoint -CacheRoot $EvalCacheRoot -TokenizerModelPath $TokenizerPath -ReportRoot $evalDir -RunId "$($Spec.trial_id)-eval-$Step-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "EVAL_FAILED:$($Spec.name):$Step" }
  }
  $m = ValidateEval $path $Step
  return [ordered]@{ path = $path; map = $m; val = [double]$m.validation_bits_per_utf8_byte; dev = [double]$m.development_bits_per_utf8_byte; balanced = (Balanced $m); health = 'PASS' }
}
function NewProxyRow([System.Collections.IDictionary]$Spec, [System.Collections.IDictionary]$Eval, [string]$Status) {
  return [pscustomobject]@{ phase = 'proxy'; target_name = $Spec.name; target_lr = $Spec.target; trial_id = $Spec.trial_id; val_bpb = $Eval.val; dev_bpb = $Eval.dev; balanced_bpb = $Eval.balanced; delta_vs_0015 = $null; status = $Status; checkpoint_hash = $Eval.map.checkpoint_parameter_hash; qnn_health = 'PASS' }
}
function WriteSummary([array]$Rows) {
  @($Rows | Sort-Object phase, target_lr | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8)
}
function InvokePlan {
  [IO.Directory]::CreateDirectory($TrialsRoot) | Out-Null
  $parent = AssertParentCheckpoint 6000
  $plan = [ordered]@{ schema_version = 1; experiment = 'HexaTrain HPO Schedule-v2a'; motivation = 'Search only the linear-decay target LR using a common C2200@6000 fork parent'; fixed_config = $Fixed; common_parent = $parent; targets = $Specs; peak_lr = '0.0022'; decay_start_step = 6000; decay_end_step = 8000; schedule_type = 'linear'; warmup_steps = 0; proxy_steps = 2000; primary_eval = 'step8000 exact 256+256'; created_utc = (NowUtc) }
  $plan | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $PlanPath -Encoding utf8
  foreach ($spec in $Specs) { $m = EnsureManifestIdentity $spec $parent; if ($spec.reused) { $m.status = 'REUSED'; $m.artifact_trial_id = 'hpo-schedule-v1-s6000-seed1'; $m.artifact_eval_path = Join-Path $ScheduleV1Ledger 'trials/hpo-schedule-v1-s6000-seed1/eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt' }; SaveManifest $spec.trial_id $m }
  return $parent
}
function InvokeRun {
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'Run requires explicit QairtSdkRoot and ExpectedBuildId' }
  Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  foreach ($path in @($TrainCache, $TokenizerPath)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "HPO_PRIVATE_INPUT_MISSING:$path" } }
  $parent = InvokePlan
  $script:Prepared = $false; $rows = [Collections.Generic.List[object]]::new(); $wall = [Diagnostics.Stopwatch]::StartNew(); $proxyNewSteps = 0
  $anchorSpec = $Specs[0]
  $anchorEvalPath = Join-Path $ScheduleV1Ledger 'trials/hpo-schedule-v1-s6000-seed1/eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
  $anchorMap = ValidateEval $anchorEvalPath 8000
  $anchorEval = [ordered]@{ path = $anchorEvalPath; map = $anchorMap; val = [double]$anchorMap.validation_bits_per_utf8_byte; dev = [double]$anchorMap.development_bits_per_utf8_byte; balanced = (Balanced $anchorMap); health = 'PASS' }
  $anchorManifest = LoadManifest $anchorSpec.trial_id; $anchorManifest.primary = [ordered]@{ val_bpb = $anchorEval.val; dev_bpb = $anchorEval.dev; balanced_bpb = $anchorEval.balanced; health = 'PASS'; checkpoint_hash = $anchorMap.checkpoint_parameter_hash }; SaveManifest $anchorSpec.trial_id $anchorManifest
  $rows.Add((NewProxyRow $anchorSpec $anchorEval 'REUSED'))
  foreach ($spec in @($Specs | Where-Object { -not $_.reused })) {
    $manifest = EnsureManifestIdentity $spec $parent; SaveManifest $spec.trial_id $manifest; CopyParentCheckpoint $spec $parent | Out-Null
    try {
      $evalPath = Join-Path (TrialDir $spec.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
      if ($manifest.status -eq 'COMPLETED' -and [int]$manifest.completed_steps -eq 8000 -and $manifest.Contains('primary') -and (Test-Path -LiteralPath $evalPath -PathType Leaf)) {
        $eval = InvokeEval $spec 8000; $rows.Add((NewProxyRow $spec $eval 'REUSED')); $proxyNewSteps += 2000; $script:Prepared = $true; continue
      }
      $smokeReport = Join-Path (Join-Path (TrialDir $spec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps6004-result.txt'
      if (-not (Test-Path -LiteralPath $smokeReport -PathType Leaf)) {
        $smoke = InvokeTraining $spec 6004 6000 'smoke' -Smoke; $manifest = LoadManifest $spec.trial_id; $manifest.smoke_health = 'PASS'; $manifest.smoke_report = $smoke.report; SaveManifest $spec.trial_id $manifest; $script:Prepared = $true
      }
      $run = InvokeTraining $spec 8000 6000 'final'; $telemetry = WriteTelemetryAnchors $spec 6000; $manifest = LoadManifest $spec.trial_id; $manifest.status = 'COMPLETED'; $manifest.completed_steps = 8000; $manifest.actual_new_steps = 2000; $manifest.checkpoint_path = Join-Path (TrialDir $spec.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; $manifest.checkpoint_parameter_hash = $run.map.final_parameter_hash; $manifest.final_health = 'PASS'; $manifest.wall_time_ms = $run.wall_ms; $manifest.training_total_seconds = [double]$run.map.training_total_seconds; $manifest.training_step_ms = [double]$run.map.training_step_ms; $manifest.run_bytes_per_second = [math]::Round(([double]$run.map.run_target_utf8_bytes_seen / [double]$run.map.training_total_seconds), 6); $manifest.run_target_tokens = [int64]$run.map.run_target_tokens_seen; $manifest.run_target_utf8_bytes = [int64]$run.map.run_target_utf8_bytes_seen; $manifest.total_target_tokens = [int64]$run.map.target_tokens_seen; $manifest.total_target_utf8_bytes = [int64]$run.map.target_utf8_bytes_seen; $manifest.unique_chunks = [int64]$run.map.unique_chunks_seen; $manifest.unique_articles = [int64]$run.map.unique_articles_seen; $manifest.telemetry_anchors_path = $telemetry; SaveManifest $spec.trial_id $manifest
      $eval = InvokeEval $spec 8000; $manifest = LoadManifest $spec.trial_id; $manifest.primary = [ordered]@{ val_bpb = $eval.val; dev_bpb = $eval.dev; balanced_bpb = $eval.balanced; health = 'PASS'; checkpoint_hash = $eval.map.checkpoint_parameter_hash }; SaveManifest $spec.trial_id $manifest; $rows.Add((NewProxyRow $spec $eval 'COMPLETED')); $proxyNewSteps += 2000; $script:Prepared = $true
    } catch { $manifest = LoadManifest $spec.trial_id; $manifest.status = 'FAILED'; $manifest.failure = $_.Exception.Message; SaveManifest $spec.trial_id $manifest; WriteEvent @{ trial_id = $spec.trial_id; phase = 'proxy'; status = 'FAILED'; detail = $_.Exception.Message }; throw }
  }
  $anchorRow = @($rows | Where-Object { $_.target_lr -eq '0.0015' })[0]
  $newRows = @($rows | Where-Object { $_.target_lr -ne '0.0015' })
  $proxyBest = @($newRows | Sort-Object balanced_bpb)[0]
  $improvement = [double]$anchorRow.balanced_bpb - [double]$proxyBest.balanced_bpb
  $bothImprove = ([double]$proxyBest.val_bpb -lt [double]$anchorRow.val_bpb) -and ([double]$proxyBest.dev_bpb -lt [double]$anchorRow.dev_bpb)
  $fullExecuted = $false; $fullBest = $null
  if ($improvement -gt 0.005 -or $bothImprove) {
    $parent4000 = AssertParentCheckpoint 4000
    $fullSpec = [ordered]@{ name = "S4000_$($proxyBest.target_name)"; target = $proxyBest.target_lr; trial_id = "hpo-schedule-v2a-full-s4000-$($proxyBest.target_name.ToLowerInvariant())-seed1"; start = 4000; reused = $false }
    $fullManifest = EnsureManifestIdentity $fullSpec $parent4000; SaveManifest $fullSpec.trial_id $fullManifest; CopyParentCheckpoint $fullSpec $parent4000 | Out-Null
    $fullEvalPath = Join-Path (TrialDir $fullSpec.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'
    if (-not ($fullManifest.status -eq 'COMPLETED' -and $fullManifest.Contains('primary') -and (Test-Path -LiteralPath $fullEvalPath -PathType Leaf))) {
      $fullSmoke = Join-Path (Join-Path (TrialDir $fullSpec.trial_id) 'training') 'seed1-l19-v1024-t32-d64-f128-steps4004-result.txt'
      if (-not (Test-Path -LiteralPath $fullSmoke -PathType Leaf)) { $null = InvokeTraining $fullSpec 4004 4000 'smoke' -Smoke; $script:Prepared = $true }
      $fullRun = InvokeTraining $fullSpec 8000 4000 'final'; $fullTelemetry = WriteTelemetryAnchors $fullSpec 4000; $fullManifest = LoadManifest $fullSpec.trial_id; $fullManifest.status = 'COMPLETED'; $fullManifest.completed_steps = 8000; $fullManifest.actual_new_steps = 4000; $fullManifest.telemetry_anchors_path = $fullTelemetry; $fullManifest.checkpoint_path = Join-Path (TrialDir $fullSpec.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; $fullManifest.checkpoint_parameter_hash = $fullRun.map.final_parameter_hash; SaveManifest $fullSpec.trial_id $fullManifest
    }
    $fullEval = InvokeEval $fullSpec 8000; $s4000AnchorPath = Join-Path $ScheduleV1Ledger 'trials/hpo-schedule-v1-s4000-seed1/eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'; $s4000Anchor = ValidateEval $s4000AnchorPath 8000
    $fullRows = @([pscustomobject]@{ target_lr = '0.0015'; val_bpb = [double]$s4000Anchor.validation_bits_per_utf8_byte; dev_bpb = [double]$s4000Anchor.development_bits_per_utf8_byte; balanced_bpb = (Balanced $s4000Anchor); source = 'S4000 existing artifact' }, [pscustomobject]@{ target_lr = $fullSpec.target; val_bpb = $fullEval.val; dev_bpb = $fullEval.dev; balanced_bpb = $fullEval.balanced; source = $fullSpec.trial_id })
    @($fullRows) | Export-Csv -LiteralPath $FullTailPath -NoTypeInformation -Encoding utf8; $fullExecuted = $true; $fullBest = $fullEval; $proxyNewSteps += 4000
  }
  foreach ($row in $rows) { $row.delta_vs_0015 = [math]::Round(([double]$row.balanced_bpb - [double]$anchorRow.balanced_bpb), 9) }
  WriteSummary @($rows); $wall.Stop()
  $actualNew = [int]$proxyNewSteps; $naive = 24000; $fullSteps = if ($fullExecuted) { 4000 } else { 0 }; $smokeSteps = if ($fullExecuted) { 16 } else { 12 }; $reusedPrefix = 18000 + $fullSteps
  $compute = [ordered]@{ naive_fresh_steps = $naive; actual_new_steps = $actualNew; proxy_new_steps = 6000; full_tail_confirmation_steps = $fullSteps; smoke_probe_steps = $smokeSteps; reused_prefix_steps = $reusedPrefix; saved_steps = $naive - $actualNew; saving_percent = [math]::Round((1 - ($actualNew / [double]$naive)) * 100, 2); wall_time_ms = [math]::Round($wall.Elapsed.TotalMilliseconds, 1); proxy_best_target = $proxyBest.target_lr; proxy_best_balanced_bpb = [double]$proxyBest.balanced_bpb; proxy_improvement_bpb = $improvement; full_tail_executed = $fullExecuted; generated_utc = (NowUtc) }
  $compute | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ComputePath -Encoding utf8
}
function InvokeSummarize { if (Test-Path -LiteralPath $SummaryPath) { Get-Content -LiteralPath $SummaryPath }; if (Test-Path -LiteralPath $ComputePath) { Get-Content -LiteralPath $ComputePath } }

if ($SelfTest) {
  if ($Fixed.parameter_count -ne 758528 -or $AllowedTargets.Count -ne 4) { throw 'SCHEDULE_V2A_SELFTEST_FIXED' }
  $seen = @{}; foreach ($spec in $Specs) { AssertTarget $spec.target; if ($seen.ContainsKey($spec.trial_id)) { throw 'SCHEDULE_V2A_SELFTEST_DUPLICATE_ID' }; $seen[$spec.trial_id] = $true }
  foreach ($target in $AllowedTargets) {
    if ([math]::Abs((ExpectedLearningRate ([double]$target) 6000) - 0.0022) -gt 1e-12 -or [math]::Abs((ExpectedLearningRate ([double]$target) 8000) - [double]$target) -gt 1e-12) { throw "SCHEDULE_V2A_SELFTEST_ENDPOINT:$target" }
  }
  $failed = $false; try { AssertParentHash 'sha256:expected' 'sha256:wrong' } catch { $failed = $_.Exception.Message -eq 'PARENT_CHECKPOINT_HASH_MISMATCH' }; if (-not $failed) { throw 'SCHEDULE_V2A_SELFTEST_WRONG_PARENT' }
  Write-Host 'run_nicopedia_hpo_schedule_v2a_self_test=PASS'; exit 0
}

switch ($Mode) { 'Plan' { InvokePlan }; 'Run' { InvokeRun }; 'Summarize' { InvokeSummarize } }
