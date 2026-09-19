# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Fixed Headwise-G1 quality experiment protocol.
# Candidate2000 / Candidate4000 are explicit; default Plan never starts device work.
[CmdletBinding()]
param(
  [ValidateSet('Plan','Smoke1','Candidate2000','Candidate4000')][string]$Mode = 'Plan',
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$ValidationCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/validation.bin',
  [string]$DevelopmentCache = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/development.bin',
  [string]$ControlRoot = 'build/reports/hvx-promotion',
  [string]$ReportRoot = 'build/headwise-g1-ab/seed1',
  [string]$ExperimentId = '',
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SkipIdentityPreflight,
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
$candidateRoot = Join-Path $ReportRoot 'candidate-headwise-g1'
if (-not $ExperimentId) {
  $ExperimentId = 'headwise-g1-seed1-' + $Mode.ToLowerInvariant()
}

function Get-PhoneLmFileSha256([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "ARTIFACT_MISSING: $Path"
  }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-PhoneLmKeyValueFile([string]$Path) {
  return Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $Path -Raw)
}

$fixed = [ordered]@{
  vocabulary = 1024; tokens = 32; dimension = 64; feed_forward_dimension = 128
  layers = 19; heads = 2; batch_size = 8; seed = 1
  optimizer = 'Muon'; muon_backend = 'HVX'; attention_gate = 'headwise_g1_sigmoid'
  learning_rate = '0.0022'; learning_rate_schedule = 'linear_decay'
  decay_start_step = 4000; decay_end_step = 8000; schedule_total_steps = 8000
  target_learning_rate = '0.0001'; muon_learning_rate = '0.005'
  muon_momentum = '0.95'; muon_nesterov = $true; muon_ns_steps = 5
  checkpoint_format = 'NPRTCKPTV5'
  validation_chunks = 256; development_chunks = 256
}

# Parameter counts are derived from the generated SSOT artifact; no hand-written
# count registry.  This experiment is headwise-G1 gated.
$MetadataDerived = Get-PhoneLmParameterMetadataDerivation `
  -Vocabulary ([uint64]$fixed.vocabulary) `
  -Dimension ([uint64]$fixed.dimension) `
  -FeedForwardDimension ([uint64]$fixed.feed_forward_dimension) `
  -Layers ([uint64]$fixed.layers) `
  -Heads ([uint64]$fixed.heads) `
  -HeadwiseG1 $true
$fixed.parameter_count = $MetadataDerived.parameter_count
$fixed.parameter_delta = $MetadataDerived.parameter_delta
$fixed.muon_matrix_count = $MetadataDerived.muon_matrix_count
$fixed.muon_parameter_count = $MetadataDerived.muon_parameter_count
$fixed.aux_adam_parameter_count = $MetadataDerived.aux_adam_parameter_count

$controlCheckpoints = [ordered]@{
  '500'  = Join-Path $ControlRoot 'quality-hvx-step1000/htp-seed1-l19-t32-d64-f128-step500.ckpt'
  '1000' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step1000.ckpt'
  '1500' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step1500.ckpt'
  '2000' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step2000.ckpt'
  '2500' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step2500.ckpt'
  '3000' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step3000.ckpt'
  '3500' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step3500.ckpt'
  '4000' = Join-Path $ControlRoot 'quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step4000.ckpt'
}

function Assert-PhoneLmHeadwiseControlIdentity {
  param(
    [Parameter(Mandatory=$true)][string]$CheckpointPath,
    [Parameter(Mandatory=$true)][int]$Step
  )
  $header = Get-PhoneLmCheckpointHeaders -Path $CheckpointPath
  if ($header.Magic -ne 'NPRTCKPTV4' -or $header.Vocabulary -ne 1024 -or
      $header.Tokens -ne 32 -or $header.Dimension -ne 64 -or
      $header.FeedForward -ne 128 -or $header.Layers -ne 19 -or
      $header.Heads -ne 2 -or $header.Seed -ne 1 -or $header.Step -ne $Step) {
    throw "CONTROL_CHECKPOINT_IDENTITY_MISMATCH step=$Step path=$CheckpointPath"
  }
  if ($header.PSObject.Properties.Name.Contains('attentionGate') -and
      [int]$header.attentionGate -ne 0) {
    throw "CONTROL_ATTENTION_GATE_NOT_NONE step=$Step"
  }
  $bytes = [IO.File]::ReadAllBytes($CheckpointPath)
  # Mixed-optimizer V4 header: attentionGate is always 0 (ungated).
  return [pscustomobject]@{
    path = $CheckpointPath
    sha256 = Get-PhoneLmFileSha256 $CheckpointPath
    magic = $header.Magic
    step = [int]$header.Step
    seed = [int]$header.Seed
  }
}

function Assert-PhoneLmHeadwiseCandidateIdentity {
  param(
    [Parameter(Mandatory=$true)][string]$CheckpointPath,
    [Parameter(Mandatory=$true)][int]$Step,
    [Parameter(Mandatory=$true)][string]$ResultPath
  )
  if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
    throw "CANDIDATE_RESULT_MISSING: $ResultPath"
  }
  $map = Read-PhoneLmKeyValueFile $ResultPath
  foreach ($pair in @(
      @('attention_gate','headwise_g1_sigmoid'),
      @('checkpoint_format','NPRTCKPTV5'),
      @('muon_lr','0.004999999888'),
      @('aux_adam_lr','0.002199999988'))) {
    if (-not $map.Contains($pair[0]) -or $map[$pair[0]] -ne $pair[1]) {
      throw "CANDIDATE_IDENTITY_FIELD_MISMATCH: $($pair[0]) expected=$($pair[1]) actual=$(if($map.Contains($pair[0])){$map[$pair[0]]}else{'<missing>'})"
    }
  }
  if ([int]$map.parameter_count -ne $MetadataDerived.parameter_count -or
      [int]$map.muon_matrix_count -ne $MetadataDerived.muon_matrix_count -or
      [int]$map.muon_parameter_count -ne $MetadataDerived.muon_parameter_count -or
      [int]$map.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count) {
    throw 'CANDIDATE_PARAMETER_PARTITION_MISMATCH'
  }
  if ($map.Contains('completed_steps') -and [int]$map.completed_steps -lt $Step) {
    throw "CANDIDATE_STEP_SHORT completed=$($map.completed_steps) required=$Step"
  }
  $header = Get-PhoneLmCheckpointHeaders -Path $CheckpointPath
  if ($header.Magic -ne 'NPRTCKPTV5' -or $header.Step -ne $Step -or
      $header.Vocabulary -ne 1024 -or $header.Tokens -ne 32 -or
      $header.Dimension -ne 64 -or $header.FeedForward -ne 128 -or
      $header.Layers -ne 19 -or $header.Heads -ne 2 -or $header.Seed -ne 1) {
    throw "CANDIDATE_CHECKPOINT_HEADER_MISMATCH step=$Step"
  }
  if ($header.PSObject.Properties.Name.Contains('attentionGate') -and
      [int]$header.attentionGate -ne 1) {
    throw "CANDIDATE_ATTENTION_GATE_NOT_GATED step=$Step"
  }
  return [pscustomobject]@{
    path = $CheckpointPath
    sha256 = Get-PhoneLmFileSha256 $CheckpointPath
    magic = $header.Magic
    step = [int]$header.Step
    parameter_hash = [string]$map.final_parameter_hash
  }
}

function Write-PhoneLmHeadwiseManifest {
  param(
    [Parameter(Mandatory=$true)][hashtable]$Payload,
    [Parameter(Mandatory=$true)][string]$Path
  )
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding utf8
  Write-Host "manifest_written=$Path"
}

if ($SelfTest) {
  if ($Mode -ne 'Plan' -or $fixed.parameter_count -ne $MetadataDerived.parameter_count -or
      $fixed.muon_learning_rate -ne '0.005' -or
      $fixed.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count -or
      $MetadataDerived.parameter_delta -eq 0 -or
      $MetadataDerived.muon_parameter_count + $MetadataDerived.aux_adam_parameter_count -ne $MetadataDerived.parameter_count) {
    throw 'HEADWISE_G1_AB_SELFTEST_IDENTITY'
  }
  Write-Host 'run_headwise_g1_ab_self_test=PASS'
  exit 0
}

if ($Mode -eq 'Plan') {
  [pscustomobject]$fixed | ConvertTo-Json -Depth 4
  Write-Host "experiment_id=$ExperimentId"
  Write-Host 'control_reuse_requires_exact_recipe_and_attention_gate=none'
  Write-Host 'next=Smoke1 | Candidate2000 | Candidate4000'
  exit 0
}

# --- identity preflight ---
$tokenizerHash = 'sha256:' + (Get-FileHash -LiteralPath $TokenizerModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestPath = Join-Path $ReportRoot 'experiment-manifest.json'
$evalPoints = @()
$controlIdentity = @{}
$candidateIdentity = @{}

if ($Mode -eq 'Candidate4000') {
  $resumeStep = 2000
  $steps = 4000
  $checkpointInterval = 500
  $evalSteps = @(2500,3000,3500,4000)
  $allEvalSteps = @(500,1000,1500,2000,2500,3000,3500,4000)
  $inputCkpt = Join-Path $candidateRoot 'htp-seed1-l19-t32-d64-f128-step2000.ckpt'
  $inputResult = Join-Path $candidateRoot 'seed1-l19-v1024-t32-d64-f128-steps2000-result.txt'
  if (-not $SkipIdentityPreflight) {
    $candidateIdentity['2000'] = Assert-PhoneLmHeadwiseCandidateIdentity `
      -CheckpointPath $inputCkpt -Step 2000 -ResultPath $inputResult
  } else {
    $candidateIdentity['2000'] = [pscustomobject]@{
      path = $inputCkpt
      sha256 = Get-PhoneLmFileSha256 $inputCkpt
      magic = 'NPRTCKPTV5'
      step = 2000
    }
  }
} else {
  $resumeStep = 0
  $steps = if ($Mode -eq 'Smoke1') { 1 } else { 2000 }
  $checkpointInterval = if ($Mode -eq 'Smoke1') { 1 } else { 500 }
  $evalSteps = if ($Mode -eq 'Candidate2000') { @(500,1000,1500,2000) } else { @() }
  $allEvalSteps = $evalSteps
}

foreach ($step in $allEvalSteps) {
  $ck = $controlCheckpoints[[string]$step]
  if ($ck -and (Test-Path -LiteralPath $ck -PathType Leaf)) {
    if (-not $SkipIdentityPreflight) {
      $controlIdentity[[string]$step] = Assert-PhoneLmHeadwiseControlIdentity -CheckpointPath $ck -Step $step
    } else {
      $controlIdentity[[string]$step] = [pscustomobject]@{
        path = $ck
        sha256 = Get-PhoneLmFileSha256 $ck
        magic = 'NPRTCKPTV4'
        step = $step
      }
    }
  }
}

# --- training ---
$resumeArgs = @()
if ($resumeStep -gt 0) { $resumeArgs += @('-ResumeStep', $resumeStep) }
& $training -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
  -Seed 1 -Layers 19 -Steps $steps -Tokens 32 -Vocabulary 1024 `
  -Dimension 64 -FeedForwardDimension 128 -BatchSize 8 `
  -LearningRate '0.0022' -LearningRateSchedule linear_decay `
  -DecayStartStep 4000 -DecayEndStep 8000 -ScheduleTotalSteps 8000 `
  -TargetLearningRate '0.0001' -ExperimentFork -ParentLearningRate '0.0022' `
  -Optimizer Muon -MuonBackend HVX -AttentionGate headwise_g1_sigmoid `
  -MuonLearningRate '0.005' -MuonMomentum '0.95' -MuonNsSteps 5 `
  -CachePath $CachePath -TokenizerModelPath $TokenizerModelPath `
  -ReportRoot $candidateRoot -CheckpointInterval $checkpointInterval `
  -CheckpointStallSeconds 1800 @resumeArgs `
  -SkipBuild:$SkipBuild -SkipInstall:$SkipInstall
if ($LASTEXITCODE -ne 0) { throw 'HEADWISE_G1_TRAINING_FAILED' }

# --- eval + gate diagnostics + manifest rows ---
foreach ($step in $evalSteps) {
  $checkpoint = Join-Path $candidateRoot "htp-seed1-l19-t32-d64-f128-step$step.ckpt"
  & $evaluation -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
    -Seed 1 -Layers 19 -Heads 2 -Tokens 32 -Vocabulary 1024 `
    -Dimension 64 -FeedForwardDimension 128 -AttentionGate headwise_g1_sigmoid `
    -CheckpointStep $step -CheckpointPath $checkpoint `
    -TokenizerModelPath $TokenizerModelPath `
    -ValidationChunks 256 -DevelopmentChunks 256 `
    -ReportRoot (Join-Path $ReportRoot "candidate-eval256-step$step") `
    -SkipBuild -SkipInstall
  if ($LASTEXITCODE -ne 0) { throw "HEADWISE_G1_EVAL_FAILED: step=$step" }
}

# Control evals for steps without an existing 256/256 report.
foreach ($step in $allEvalSteps) {
  if ($step -eq 2000) { continue }
  $ctrlDir = Join-Path $ReportRoot "control-eval256-step$step"
  $existing = Get-ChildItem -Path $ctrlDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
  $ctrlCk = $controlCheckpoints[[string]$step]
  if ($existing -and (Test-Path -LiteralPath $ctrlCk -PathType Leaf)) { continue }
  if (-not (Test-Path -LiteralPath $ctrlCk -PathType Leaf)) {
    throw "CONTROL_CHECKPOINT_MISSING step=$step path=$ctrlCk"
  }
  & $evaluation -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
    -Seed 1 -Layers 19 -Heads 2 -Tokens 32 -Vocabulary 1024 `
    -Dimension 64 -FeedForwardDimension 128 -AttentionGate none `
    -CheckpointStep $step -CheckpointPath $ctrlCk `
    -TokenizerModelPath $TokenizerModelPath `
    -ValidationChunks 256 -DevelopmentChunks 256 `
    -ReportRoot $ctrlDir `
    -SkipBuild -SkipInstall
  if ($LASTEXITCODE -ne 0) { throw "HEADWISE_G1_CONTROL_EVAL_FAILED: step=$step" }
}

# Recompute machine-readable A/B table from artifacts.
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

$comparisonRows = @()
foreach ($step in $allEvalSteps) {
  $candEvalDir = Join-Path $ReportRoot "candidate-eval256-step$step"
  $candHtp = Get-ChildItem -Path $candEvalDir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
  $candCpu = Get-ChildItem -Path $candEvalDir -Filter '*-cpu.txt' -ErrorAction SilentlyContinue | Select-Object -First 1
  $ctrlEvalPath = if ($step -eq 2000) {
    Join-Path $ControlRoot 'eval-hvx-seed1-step2000/seed1-l19-t32-d64-f128-step2000-v256-d256-htp.txt'
  } else {
    $dir = Join-Path $ReportRoot "control-eval256-step$step"
    if (Test-Path -LiteralPath $dir) {
      (Get-ChildItem -Path $dir -Filter '*-htp.txt' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
    } else { $null }
  }
  if (-not $candHtp -or -not (Test-Path -LiteralPath $ctrlEvalPath -PathType Leaf)) {
    Write-Host "skip_comparison_step=$step (missing eval artifacts)"
    continue
  }
  $candMap = Read-PhoneLmKeyValueFile $candHtp.FullName
  $ctrlMap = Read-PhoneLmKeyValueFile $ctrlEvalPath
  $cVal = [double]$candMap.validation_bits_per_utf8_byte
  $cDev = [double]$candMap.development_bits_per_utf8_byte
  $uVal = [double]$ctrlMap.validation_bits_per_utf8_byte
  $uDev = [double]$ctrlMap.development_bits_per_utf8_byte
  $row = [ordered]@{
    step = $step
    control_val_bpb = $uVal
    candidate_val_bpb = $cVal
    delta_val_bpb = $cVal - $uVal
    control_dev_bpb = $uDev
    candidate_dev_bpb = $cDev
    delta_dev_bpb = $cDev - $uDev
    control_balanced_bpb = ($uVal + $uDev) / 2.0
    candidate_balanced_bpb = ($cVal + $cDev) / 2.0
    delta_balanced_bpb = (($cVal + $cDev) / 2.0) - (($uVal + $uDev) / 2.0)
    candidate_checkpoint_sha256 = if ($candidateIdentity.Contains([string]$step)) { $candidateIdentity[[string]$step].sha256 } else { Get-PhoneLmFileSha256 (Join-Path $candidateRoot "htp-seed1-l19-t32-d64-f128-step$step.ckpt") }
    control_checkpoint_sha256 = if ($controlIdentity.Contains([string]$step)) { $controlIdentity[[string]$step].sha256 } else { $null }
    validation_chunks = 256
    development_chunks = 256
    final_split_used = $false
  }
  $comparisonRows += [pscustomobject]$row

  # Checkpoint-static gate diagnostics on identical Val windows.
  $diagPath = Join-Path $candEvalDir 'gate-static-diagnostics.txt'
  & $diagExe `
    (Join-Path $candidateRoot "htp-seed1-l19-t32-d64-f128-step$step.ckpt") `
    $TokenizerModelPath $ValidationCache 32 256 |
    Set-Content -LiteralPath $diagPath -Encoding utf8
  if ($LASTEXITCODE -ne 0) { throw "GATE_DIAGNOSTICS_FAILED step=$step" }
}

$comparisonPath = Join-Path $ReportRoot 'ab-comparison.json'
$comparisonRows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $comparisonPath -Encoding utf8
Write-Host "comparison_written=$comparisonPath"

$finalStep = if ($Mode -eq 'Candidate4000') { 4000 } else { 2000 }
$finalCkpt = Join-Path $candidateRoot "htp-seed1-l19-t32-d64-f128-step$finalStep.ckpt"
$finalResult = Join-Path $candidateRoot ("seed1-l19-v1024-t32-d64-f128-steps$finalStep-result.txt")
if (Test-Path -LiteralPath $finalResult -PathType Leaf) {
  $finalMap = Read-PhoneLmKeyValueFile $finalResult
  $candidateIdentity[[string]$finalStep] = [pscustomobject]@{
    path = $finalCkpt
    sha256 = Get-PhoneLmFileSha256 $finalCkpt
    magic = $finalMap.checkpoint_format
    step = $finalStep
    parameter_hash = $finalMap.final_parameter_hash
  }
}

$manifest = [ordered]@{
  schema = 'HEADWISE_G1_AB_MANIFEST_V1'
  experiment_id = $ExperimentId
  mode = $Mode
  generated_utc = [DateTimeOffset]::UtcNow.ToString('o')
  attention_gate = $fixed.attention_gate
  model_config = [ordered]@{
    vocabulary = 1024; tokens = 32; dimension = 64
    feed_forward_dimension = 128; layers = 19; heads = 2
  }
  seed = 1
  batch_size = 8
  tokenizer_hash = $tokenizerHash
  dataset_hash = 'fnv1a64:0c7b2826f5f26fea'
  dataset_order_identity = 'training_order_seed=20260806'
  train_cache_path = $CachePath
  validation_cache_path = $ValidationCache
  development_cache_path = $DevelopmentCache
  validation_window_identity = 'val-first-256-chunks-v1024-bpe'
  development_window_identity = 'dev-first-256-chunks-v1024-bpe'
  final_split_used = $false
  optimizer = [ordered]@{
    muon_algorithm = 'keller_original_64560829_fp32'
    muon_backend = 'HVX_W8'
    muon_learning_rate = '0.005'
    muon_momentum = 0.95
    muon_nesterov = $true
    muon_ns_steps = 5
    aux_adam_learning_rate = '0.0022'
    aux_adam_beta1 = 0.9
    aux_adam_beta2 = 0.999
    aux_adam_epsilon = 1e-8
    aux_adam_weight_decay = 0.0
    wg_role = 'AUX_ADAM'
  }
  schedule = [ordered]@{
    kind = 'linear_decay'
    decay_start_step = 4000
    decay_end_step = 8000
    schedule_total_steps = 8000
  }
  checkpoint_format = 'NPRTCKPTV5'
  qairt_build_id = $ExpectedBuildId
  resume_from_step = $resumeStep
  candidate_input_checkpoint = if ($candidateIdentity.Contains('2000')) { $candidateIdentity['2000'] } else { $null }
  candidate_checkpoints = $candidateIdentity
  control_checkpoints = $controlIdentity
  evaluations = $comparisonRows
}
Write-PhoneLmHeadwiseManifest -Payload $manifest -Path $manifestPath
Write-Host "PASS HEADWISE_G1_AB mode=$Mode"
