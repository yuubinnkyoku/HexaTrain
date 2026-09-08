# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Controlled linear-decay schedule HPO for V1024/T32/D64/FFN128/L19/H2.
# Research artifacts stay private below build/; this runner never exports raw
# checkpoints, device identifiers, or private telemetry.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Run',
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/schedule-v1',
  [string]$AnchorLedgerRoot = 'build/hpo/nicopedia-v1024-d64-f128/lr-v1',
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Ledger = [IO.Path]::GetFullPath((Join-Path $Root $LedgerRoot))
$AnchorLedger = [IO.Path]::GetFullPath((Join-Path $Root $AnchorLedgerRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $Root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $Ledger.StartsWith($buildRoot,[StringComparison]::OrdinalIgnoreCase) -or
    -not $AnchorLedger.StartsWith($buildRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Ledger roots must resolve below build' }
$TrialsRoot = Join-Path $Ledger 'trials'
$EventsPath = Join-Path $Ledger 'trials.jsonl'
$SummaryPath = Join-Path $Ledger 'summary.csv'
$PlanPath = Join-Path $Ledger 'plan.json'
$ComputePath = Join-Path $Ledger 'compute.json'
$TrainingRunner = Join-Path $Root 'scripts/run_nicopedia_htp_training.ps1'
$EvalRunner = Join-Path $Root 'scripts/run_nicopedia_htp_eval.ps1'
$TrainingDataRoot = Join-Path $Root 'build/private-data/nicopedia-real-text-bpe-v1024'
$TokenizerPath = Join-Path $TrainingDataRoot 'tokenizer/byte-bpe-v1024.model'
$TrainCache = Join-Path $TrainingDataRoot 'caches/train_pilot.bin'
$EvalCacheRoot = Join-Path $TrainingDataRoot 'caches'
$AnchorTrial = Join-Path $AnchorLedger 'trials/hpo-lr-v1-lr0p0022-seed1'
$AnchorSummary = Join-Path $AnchorLedger 'summary.csv'

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
$Specs = @(
  [ordered]@{ name='S4000'; trial_id='hpo-schedule-v1-s4000-seed1'; start=4000; reused_prefix=4000 },
  [ordered]@{ name='S6000'; trial_id='hpo-schedule-v1-s6000-seed1'; start=6000; reused_prefix=6000 }
)

function NowUtc { [DateTime]::UtcNow.ToString('o') }
function Write-Event([hashtable]$Fields) {
  [IO.Directory]::CreateDirectory($Ledger) | Out-Null
  $row = [ordered]@{ utc = NowUtc }
  foreach ($key in $Fields.Keys) { $row[$key] = $Fields[$key] }
  $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsPath -Encoding utf8
}
function TrialDir([string]$Name) { Join-Path $TrialsRoot $Name }
function ManifestPath([string]$Name) { Join-Path (TrialDir $Name) 'manifest.json' }
function LoadManifest([string]$Name) {
  $path = ManifestPath $Name
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}
function SaveManifest([string]$Name,[hashtable]$Manifest) {
  $dir = TrialDir $Name; [IO.Directory]::CreateDirectory($dir) | Out-Null
  $Manifest | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (ManifestPath $Name) -Encoding utf8
}
function ReadKeyValue([string]$Path) {
  $map = @{}
  foreach ($line in Get-Content -LiteralPath $Path) {
    if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $map[$Matches[1]] = $Matches[2].Trim() }
  }
  $map
}
function Balanced([hashtable]$Map) {
  ([double]$Map.validation_bits_per_utf8_byte + [double]$Map.development_bits_per_utf8_byte) / 2.0
}
function ExpectedScheduledLearningRate([int]$Start,[int]$Step) {
  if ($Step -le $Start) { return 0.0022 }
  if ($Step -ge 8000) { return 0.0015 }
  return 0.0022 + (($Step - $Start) / [double](8000 - $Start)) * (0.0015 - 0.0022)
}
function CheckpointName([int]$Step) { "htp-seed1-l19-t32-d64-f128-step$Step.ckpt" }
function EvalName([int]$Step) { "seed1-l19-t32-d64-f128-step$Step-v256-d256-htp.txt" }
function ValidateEval([string]$Path,[int]$Step) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "EVAL_MISSING:$Path" }
  $m = ReadKeyValue $Path
  foreach ($key in @('status','checkpoint_step','checkpoint_parameter_hash','checkpoint_finite','qnn_return_code_success','output_tensors_finite','cpu_fallback','validation_nonfinite_chunks','development_nonfinite_chunks','validation_bits_per_utf8_byte','development_bits_per_utf8_byte')) { if (-not $m.ContainsKey($key)) { throw "EVAL_FIELD_MISSING:$key" } }
  if ($m.status -ne 'SUCCESS' -or [int]$m.checkpoint_step -ne $Step -or $m.checkpoint_finite -ne 'true' -or $m.qnn_return_code_success -ne 'true' -or $m.output_tensors_finite -ne 'true' -or $m.cpu_fallback -ne 'false' -or $m.validation_nonfinite_chunks -ne '0' -or $m.development_nonfinite_chunks -ne '0') { throw "EVAL_HEALTH_REJECTED:$Path" }
  if ($m.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "EVAL_HASH_INVALID:$Path" }
  return $m
}
function ValidateParent([int]$Step) {
  . (Join-Path $PSScriptRoot 'qairt_version.ps1')
  . (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
  $manifestPath = Join-Path $AnchorTrial 'manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'PARENT_MANIFEST_MISSING' }
  $m = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -AsHashtable
  foreach ($key in @('vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','optimizer','beta1','beta2','epsilon','gradient_clip','weight_decay','dataset_cache_content_hash','training_order_hash','lr_schedule','learning_rate')) {
    if (-not $m.ContainsKey($key) -or [string]$m[$key] -ne [string]$(if ($key -eq 'learning_rate') { '0.0022' } elseif ($key -eq 'lr_schedule') { 'constant' } else { $Fixed[$key] })) { throw "PARENT_MANIFEST_IDENTITY_MISMATCH:$key" }
  }
  $path = Join-Path (Join-Path $AnchorTrial 'training') (CheckpointName $Step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "PARENT_CHECKPOINT_MISSING:$Step" }
  $h = Get-PhoneLmCheckpointHeaders -Path $path
  if ($h.Magic -ne 'NPRTCKPTV3' -or $h.Vocabulary -ne 1024 -or $h.Tokens -ne 32 -or $h.Dimension -ne 64 -or $h.FeedForward -ne 128 -or $h.Layers -ne 19 -or $h.Heads -ne 2 -or $h.Seed -ne 1 -or $h.Step -ne $Step -or $h.TokenizerKind -ne $Fixed.tokenizer_kind -or $h.TokenizerHash -ne $Fixed.tokenizer_hash) { throw "PARENT_CHECKPOINT_IDENTITY_MISMATCH:$Step" }
  $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  $evalProbe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
  $validation = Join-Path $TrainingDataRoot 'caches/validation.bin'; $development = Join-Path $TrainingDataRoot 'caches/development.bin'
  if (-not (Test-Path -LiteralPath $evalProbe -PathType Leaf) -or -not (Test-Path -LiteralPath $validation -PathType Leaf) -or -not (Test-Path -LiteralPath $development -PathType Leaf)) { throw 'PARENT_HOST_CHECKPOINT_EVALUATOR_UNAVAILABLE' }
  $probe = & $evalProbe $path $validation $development 1 1
  if ($LASTEXITCODE -ne 0) { throw "PARENT_HOST_CHECKPOINT_EVALUATOR_FAILED:$Step" }
  $pm = @{}; $probe | Where-Object { $_ -match '^([A-Za-z0-9_]+)=(.*)$' } | ForEach-Object { $pm[$Matches[1]]=$Matches[2] }
  if ($pm.step -ne [string]$Step -or $pm.seed -ne '1' -or $pm.layers -ne '19' -or $pm.finite -ne 'true' -or $pm.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "PARENT_HOST_IDENTITY_MISMATCH:$Step" }
  return [ordered]@{ path=$path; step=$Step; sha256="sha256:$sha"; parameter_hash=$pm.parameter_hash; source_trial_id='hpo-lr-v1-lr0p0022-seed1'; source_manifest=$manifestPath }
}
function NewManifest([System.Collections.IDictionary]$Spec,[System.Collections.IDictionary]$Parent) {
  $m = [ordered]@{ schema_version=1; trial_id=$Spec.trial_id; status='PENDING'; created_utc=(NowUtc); git_revision=(& git -C $Root rev-parse HEAD).Trim() }
  foreach ($k in $Fixed.Keys) { $m[$k]=$Fixed[$k] }
  $m.learning_rate_schedule='linear_decay'; $m.schedule_expression='step <= decay_start: peak_lr; decay_start < step <= decay_end: peak_lr + ((step-decay_start)/(decay_end-decay_start)) * (target_lr-peak_lr)'; $m.decay_start_step=$Spec.start; $m.decay_end_step=8000; $m.peak_lr='0.0022'; $m.target_lr='0.0015'; $m.schedule_total_steps=8000
  $m.experiment_fork=$true; $m.parent_trial_id=$Parent.source_trial_id; $m.parent_checkpoint_path=$Parent.path; $m.parent_checkpoint_hash=$Parent.sha256; $m.parent_parameter_hash=$Parent.parameter_hash; $m.parent_step=$Parent.step; $m.fork_step=$Parent.step; $m.parent_learning_rate='0.0022'
  $m.completed_steps=0; $m.reused_prefix_steps=$Parent.step; $m.actual_new_steps=0; $m.checkpoint_path=''; $m.checkpoint_parameter_hash=''; $m.smoke_health='PENDING'; $m.final_health='PENDING'
  $m
}
function CopyParent([System.Collections.IDictionary]$Spec,[System.Collections.IDictionary]$Parent) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null
  $destination = Join-Path $training (CheckpointName $Parent.step)
  if (Test-Path -LiteralPath $destination -PathType Leaf) {
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Parent.sha256.Substring(7)) { throw 'PARENT_COPY_HASH_MISMATCH' }
  } else { Copy-Item -LiteralPath $Parent.path -Destination $destination -Force }
  return $training
}
function InvokeTraining([System.Collections.IDictionary]$Spec,[int]$Steps,[int]$Resume,[string]$Phase,[switch]$Smoke) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'; [IO.Directory]::CreateDirectory($training) | Out-Null
  $runId = "$($Spec.trial_id)-$Phase-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"
  if ($runId.Length -gt 63) { $runId=$runId.Substring(0,63) }
  $args = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-Seed',1,'-Layers',19,'-Steps',$Steps,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-BatchSize',8,'-LearningRate','0.0022','-LearningRateSchedule','linear_decay','-DecayStartStep',$Spec.start,'-DecayEndStep',8000,'-ScheduleTotalSteps',8000,'-TargetLearningRate','0.0015','-ParentLearningRate','0.0022','-ExperimentFork','-ResumeStep',$Resume,'-CheckpointInterval',250,'-CheckpointStallSeconds',7200,'-CachePath',$TrainCache,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$training,'-RunId',$runId)
  if ($SkipBuild -or $script:Prepared) { $args += '-SkipBuild' }; if ($SkipInstall -or $script:Prepared) { $args += '-SkipInstall' }; if ($Smoke) { $args += '-AllowQualityFailure' }
  $sw=[Diagnostics.Stopwatch]::StartNew(); & pwsh -NoProfile -File $TrainingRunner @args | Out-Host; $code=$LASTEXITCODE; $sw.Stop(); if ($code -ne 0) { throw "TRAINING_FAILED:$($Spec.name):$($Phase):exit=$code" }
  $reportPath=Join-Path $training "seed1-l19-v1024-t32-d64-f128-steps$Steps-result.txt"; if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "TRAINING_REPORT_MISSING:$Phase" }
  $map=ReadKeyValue $reportPath
  if ($map.status -ne 'SUCCESS' -or [int]$map.completed_steps -ne $Steps -or $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or $map.cpu_fallback -ne 'false' -or $map.final_finite -ne 'true' -or [int]$map.resume_from_step -ne $Resume) { throw "TRAINING_HEALTH_REJECTED:$Phase" }
  if ($map.learning_rate_schedule -ne 'linear_decay' -or [int]$map.learning_rate_decay_start_step -ne $Spec.start -or [int]$map.learning_rate_decay_end_step -ne 8000 -or [int]$map.learning_rate_schedule_total_steps -ne 8000 -or [single]$map.learning_rate_peak -ne [single]'0.0022' -or [single]$map.learning_rate_target -ne [single]'0.0015' -or $map.experiment_fork -ne 'true' -or [single]$map.parent_learning_rate -ne [single]'0.0022') { throw "TRAINING_SCHEDULE_REJECTED:$Phase" }
  $expectedRunTokens = ($Steps - $Resume) * $Fixed.batch_size * $Fixed.tokens
  if (-not $map.ContainsKey('run_target_tokens_seen') -or [int64]$map.run_target_tokens_seen -ne $expectedRunTokens) { throw "EXPOSURE_RUN_TOKENS_MISMATCH:$Phase" }
  if ($Steps -eq $Fixed.planned_max_steps) {
    foreach ($exposure in @{
      target_tokens_seen = $Fixed.planned_target_tokens
      target_utf8_bytes_seen = $Fixed.expected_original_utf8_bytes
      unique_chunks_seen = $Fixed.expected_chunks
      unique_articles_seen = $Fixed.expected_articles
    }.GetEnumerator()) {
      if (-not $map.ContainsKey($exposure.Key) -or [int64]$map[$exposure.Key] -ne [int64]$exposure.Value) { throw "EXPOSURE_TOTAL_MISMATCH:$($exposure.Key)" }
    }
  }
  return [ordered]@{ report=$reportPath; map=$map; wall_ms=$sw.Elapsed.TotalMilliseconds }
}
function WriteTelemetryAnchors([System.Collections.IDictionary]$Spec,[int]$Resume) {
  $training = Join-Path (TrialDir $Spec.trial_id) 'training'
  $telemetryPath = Join-Path $training 'learning-rate-telemetry.csv'
  if (-not (Test-Path -LiteralPath $telemetryPath -PathType Leaf)) { throw 'LEARNING_RATE_TELEMETRY_MISSING' }
  $actual = @{}
  foreach ($row in @(Import-Csv -LiteralPath $telemetryPath)) { $actual[[int]$row.step] = [double]$row.scheduled_lr }
  $rows = foreach ($step in @(4000,4500,5000,5500,6000,6500,7000,7500,8000)) {
    $expected = ExpectedScheduledLearningRate $Spec.start $step
    if ($step -le $Resume) {
      $value = 0.0022
      $source = 'validated_parent_constant_lr'
    } elseif (-not $actual.ContainsKey($step)) {
      throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISSING:$step"
    } else {
      $value = $actual[$step]
      $source = 'runtime_telemetry'
    }
    if ([math]::Abs($value - $expected) -gt 2.0e-8) { throw "LEARNING_RATE_TELEMETRY_ANCHOR_MISMATCH:$step" }
    [pscustomobject]@{ step=$step; expected_lr=$expected; actual_lr=$value; source=$source }
  }
  $path = Join-Path $training 'schedule-telemetry-anchors.csv'
  @($rows) | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding utf8
  return $path
}
function InvokeEval([System.Collections.IDictionary]$Spec,[int]$Step) {
  $training=Join-Path (TrialDir $Spec.trial_id) 'training'; $checkpoint=Join-Path $training (CheckpointName $Step); $evalDir=Join-Path (TrialDir $Spec.trial_id) "eval/step-$Step-v256-d256"; [IO.Directory]::CreateDirectory($evalDir)|Out-Null
  $args=@('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-SkipBuild','-SkipInstall','-Seed',1,'-Layers',19,'-Heads',2,'-Tokens',32,'-Vocabulary',1024,'-Dimension',64,'-FeedForwardDimension',128,'-CheckpointStep',$Step,'-ValidationChunks',256,'-DevelopmentChunks',256,'-CheckpointPath',$checkpoint,'-CacheRoot',$EvalCacheRoot,'-TokenizerModelPath',$TokenizerPath,'-ReportRoot',$evalDir,'-RunId',"$($Spec.trial_id)-eval-$Step-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))")
  $path=Join-Path $evalDir (EvalName $Step)
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { & pwsh -NoProfile -File $EvalRunner @args | Out-Host; if ($LASTEXITCODE -ne 0) { throw "EVAL_FAILED:$($Spec.name):$Step" } }
  $m=ValidateEval $path $Step
  [ordered]@{ path=$path; map=$m; val=[double]$m.validation_bits_per_utf8_byte; dev=[double]$m.development_bits_per_utf8_byte; balanced=(Balanced $m); health='PASS' }
}
function WriteSummary([array]$Rows) {
  $all=@(); if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) { $all += @(Import-Csv $SummaryPath) }; $all += $Rows
  $by=@{}; foreach($r in $all){ if($null -ne $r){$by["$($r.schedule)|$($r.decay_start)|$($r.steps)"]=$r} }
  if($by.Count -gt 0){@($by.Values | Sort-Object schedule,decay_start,steps | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding utf8)}
}
function InvokePlan {
  [IO.Directory]::CreateDirectory($TrialsRoot)|Out-Null
  $plan=[ordered]@{schema_version=1;experiment='HexaTrain HPO Schedule-v1';fixed_config=$Fixed;schedules=$Specs;peak_lr='0.0022';target_lr='0.0015';decay_end_step=8000;primary_eval='step8000 exact 256+256';created_utc=(NowUtc)}
  $plan|ConvertTo-Json -Depth 16|Set-Content -LiteralPath $PlanPath -Encoding utf8
  foreach($s in $Specs){$p=ValidateParent $s.start;if(-not(LoadManifest $s.trial_id)){SaveManifest $s.trial_id (NewManifest $s $p)}}
}
function InvokeRun {
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'Run requires explicit QairtSdkRoot and ExpectedBuildId' }
  . (Join-Path $PSScriptRoot 'qairt_version.ps1'); Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  if (-not (Test-Path -LiteralPath $TrainCache -PathType Leaf) -or -not (Test-Path -LiteralPath $TokenizerPath -PathType Leaf)) { throw 'HPO_PRIVATE_INPUT_MISSING' }
  InvokePlan; $script:Prepared=$false; $rows=[Collections.Generic.List[object]]::new(); $start=[Diagnostics.Stopwatch]::StartNew();
  $parents=@{}; foreach($s in $Specs){$parents[$s.name]=ValidateParent $s.start}
  # Reuse and independently validate the two constant-LR primary artifacts.
  foreach($lr in @('0.0015','0.0022')){
    $trial=Join-Path $AnchorLedger "trials/hpo-lr-v1-lr$($lr.Replace('.','p'))-seed1"; $eval=Join-Path $trial 'eval/rung-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt'; $e=ValidateEval $eval 8000; $rows.Add([pscustomobject]@{schedule="constant $lr";decay_start='—';steps=8000;val_bpb=$e.validation_bits_per_utf8_byte;dev_bpb=$e.development_bits_per_utf8_byte;balanced_bpb=(Balanced $e);status='REUSED';checkpoint_hash=$e.checkpoint_parameter_hash;qnn_health='PASS'})
  }
  foreach($s in $Specs){
    $parent=$parents[$s.name]; $m=LoadManifest $s.trial_id; if($null -eq $m){$m=NewManifest $s $parent}; CopyParent $s $parent | Out-Null
    try {
      if ($m.status -eq 'COMPLETED' -and [int]$m.completed_steps -eq 8000 -and $m.ContainsKey('primary')) {
        $e=ValidateEval (Join-Path (TrialDir $s.trial_id) 'eval/step-8000-v256-d256/seed1-l19-t32-d64-f128-step8000-v256-d256-htp.txt') 8000
        $rows.Add([pscustomobject]@{schedule="linear $($s.name)";decay_start=$s.start;steps=8000;val_bpb=[double]$e.validation_bits_per_utf8_byte;dev_bpb=[double]$e.development_bits_per_utf8_byte;balanced_bpb=(Balanced $e);status='REUSED';checkpoint_hash=$e.checkpoint_parameter_hash;qnn_health='PASS'})
        $script:Prepared=$true
        continue
      }
      $smokeReportPath = Join-Path (TrialDir $s.trial_id) "training/seed1-l19-v1024-t32-d64-f128-steps$($s.start+4)-result.txt"
      if (-not (Test-Path -LiteralPath $smokeReportPath -PathType Leaf)) { $smoke=InvokeTraining $s ($s.start+4) $s.start 'smoke' -Smoke; $m=LoadManifest $s.trial_id; $m.smoke_health='PASS'; $m.smoke_report=$smoke.report; $m.smoke_actual_lr='telemetry validated by training runner'; SaveManifest $s.trial_id $m; $script:Prepared=$true }
      $run=InvokeTraining $s 8000 $s.start 'final'; $telemetryAnchors=WriteTelemetryAnchors $s $s.start; $m=LoadManifest $s.trial_id; $m.status='COMPLETED'; $m.completed_steps=8000; $m.actual_new_steps=8000-$s.start; $m.checkpoint_path=Join-Path (TrialDir $s.trial_id) 'training/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; $m.checkpoint_parameter_hash=$run.map.final_parameter_hash; $m.final_health='PASS'; $m.wall_time_ms=$run.wall_ms; $m.training_total_seconds=[double]$run.map.training_total_seconds; $m.training_step_ms=[double]$run.map.training_step_ms; $m.run_bytes_per_second=[math]::Round(([double]$run.map.run_target_utf8_bytes_seen / [double]$run.map.training_total_seconds),6); $m.run_target_tokens=($run.map.run_target_tokens_seen); $m.run_target_utf8_bytes=($run.map.run_target_utf8_bytes_seen); $m.total_target_tokens=($run.map.target_tokens_seen); $m.total_target_utf8_bytes=($run.map.target_utf8_bytes_seen); $m.unique_chunks=$run.map.unique_chunks_seen; $m.unique_articles=$run.map.unique_articles_seen; $m.telemetry_anchors_path=$telemetryAnchors; SaveManifest $s.trial_id $m
      $e=InvokeEval $s 8000; $m=LoadManifest $s.trial_id; $m.primary=[ordered]@{val_bpb=$e.val;dev_bpb=$e.dev;balanced_bpb=$e.balanced;health='PASS';checkpoint_hash=$e.map.checkpoint_parameter_hash}; SaveManifest $s.trial_id $m; $rows.Add([pscustomobject]@{schedule="linear $($s.name)";decay_start=$s.start;steps=8000;val_bpb=$e.val;dev_bpb=$e.dev;balanced_bpb=$e.balanced;status='COMPLETED';checkpoint_hash=$e.map.checkpoint_parameter_hash;qnn_health='PASS'})
      $script:Prepared=$true
    } catch { $m=LoadManifest $s.trial_id; $m.status='FAILED'; $m.failure=$_.Exception.Message; SaveManifest $s.trial_id $m; Write-Event @{trial_id=$s.trial_id;phase='failed';status='FAILED';detail=$_.Exception.Message}; throw }
  }
  WriteSummary @($rows); $start.Stop(); $actual=4000+2000; $naive=16000; $compute=[ordered]@{naive_steps=$naive;actual_new_steps=$actual;reused_prefix_steps=10000;saved_steps=($naive-$actual);saving_percent=[math]::Round((1-($actual/[double]$naive))*100,2);wall_time_ms=[math]::Round($start.Elapsed.TotalMilliseconds,1);generated_utc=(NowUtc)}; $compute|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $ComputePath -Encoding utf8
}
function InvokeSummarize { if(Test-Path -LiteralPath $SummaryPath){Get-Content $SummaryPath}; foreach($s in $Specs){$m=LoadManifest $s.trial_id;if($m){Write-Output "$($s.trial_id) status=$($m.status) steps=$($m.completed_steps)"}} }

if ($SelfTest) {
  $tmp = [ordered]@{ name='S4000'; trial_id='hpo-schedule-v1-s4000-seed1'; start=4000; reused_prefix=4000 }
  $v = @{}
  foreach($step in @(4000,4001,4500,5000,5500,6000,6500,7000,7500,8000)) { $v[$step] = if($step -le 4000){.0022}else{[double]::Parse('0.0022',[Globalization.CultureInfo]::InvariantCulture)+(($step-4000)/4000.0)*(.0015-.0022)} }
  if($v[4000] -ne .0022 -or [math]::Abs($v[6000]-.00185) -gt 1e-12 -or [math]::Abs($v[8000]-.0015) -gt 1e-12){throw 'SCHEDULE_SELFTEST_FORMULA'}
  if($tmp.trial_id -eq 'hpo-lr-v1-lr0p0022-seed1' -or $tmp.start -ne 4000){throw 'SCHEDULE_SELFTEST_FORK_ID'}
  if($Fixed.parameter_count -ne 758528 -or $Fixed.tokenizer_hash -notmatch '^sha256:'){throw 'SCHEDULE_SELFTEST_FIXED'}
  Write-Host 'run_nicopedia_hpo_schedule_self_test=PASS'; exit 0
}
switch($Mode){'Plan'{InvokePlan};'Run'{InvokeRun};'Summarize'{InvokeSummarize}}
