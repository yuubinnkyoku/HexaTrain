# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
  [string]$HtpReport=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-htp-adam\inference-result.txt'),
  [string]$CpuReport=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-cpu-autoregressive-candidates\phase01_round_robin-5seeds.txt'),
  [string]$OutputDir=(Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-tiny-language-model-generation-2026-07'),
  [Parameter(Mandatory=$true)][string]$SourceCommit
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if($SourceCommit-notmatch'^[0-9a-f]{40}$'){throw 'SourceCommit must be a full Git SHA'}
$root=Split-Path -Parent $PSScriptRoot
$currentCommit=(git -C $root rev-parse HEAD).Trim()
if($LASTEXITCODE-ne0 -or $currentCommit-ne$SourceCommit){
  throw 'SourceCommit must equal the checked-out commit'
}
$null=git -C $root cat-file -e "$SourceCommit`^{commit}"
if($LASTEXITCODE-ne0){throw 'SourceCommit is not a commit object'}
$publicRoot=[IO.Path]::GetFullPath((Join-Path $root 'docs\results'))
$publishDir=[IO.Path]::GetFullPath($OutputDir)
if(!$publishDir.StartsWith($publicRoot+[IO.Path]::DirectorySeparatorChar,
                          [StringComparison]::OrdinalIgnoreCase)){
  throw 'OutputDir must be a child of repository docs/results'
}
foreach($path in @($publicRoot,$publishDir)){
  if(Test-Path -LiteralPath $path){
    $item=Get-Item -LiteralPath $path
    if($item.Attributes-band[IO.FileAttributes]::ReparsePoint){
      throw "Reparse-point output path is forbidden: $path"
    }
  }
}
$stageDir=Join-Path $root 'build\reports\public-export-staging\tiny-lm-generation'
if(Test-Path -LiteralPath $stageDir){Remove-Item -LiteralPath $stageDir -Recurse}
[IO.Directory]::CreateDirectory($stageDir)|Out-Null
$OutputDir=$stageDir
foreach($path in @($HtpReport,$CpuReport)){
  if(!(Test-Path -LiteralPath $path)){throw "Missing raw result: $path"}
}
function Read-Map([string]$Path){
  $map=@{}
  foreach($line in Get-Content -LiteralPath $Path){
    if($line-match'^([^=]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}
  }
  $map
}
function Value($Map,[string]$Key){
  if(!$Map.ContainsKey($Key)){throw "Missing allow-listed field: $Key"}
  $Map[$Key]
}
$htp=Read-Map $HtpReport
$cpu=Read-Map $CpuReport
$cpuText=Get-Content -Raw -LiteralPath $CpuReport
$cpuFreeExact=([regex]::Matches($cpuText,'(?m)^rollout_summary=seed,\d+,mode,free,pattern,\d+,correct,8,total,8,first_error,-1$')).Count
$cpuOracleExact=([regex]::Matches($cpuText,'(?m)^rollout_summary=seed,\d+,mode,oracle,pattern,\d+,correct,8,total,8,first_error,-1$')).Count
$htpStatus=Value $htp 'status'
if($htpStatus-notin@('SUCCESS','PARTIAL_SUCCESS') -or
   (Value $htp 'nan_detected')-ne'false' -or
   (Value $htp 'inf_detected')-ne'false' -or
   (Value $htp 'cpu_fallback')-ne'false'){
  throw 'HTP diagnostic execution is not publishable'
}
$goalMet=(Value $htp 'research_goal_met')-eq'true'
$htpExact=[int](Value $htp 'exact_rollout_count')
$htpOracleExact=[int](Value $htp 'oracle_exact_rollout_count')
if(($goalMet -and $htpStatus-ne'SUCCESS') -or
   (!$goalMet -and $htpStatus-ne'PARTIAL_SUCCESS')){
  throw 'HTP status and research_goal_met disagree'
}
if((Value $cpu 'summary_all_finite')-ne'true' -or
   (Value $cpu 'deterministic_replay')-ne'true' -or
   $cpuFreeExact-ne20 -or $cpuOracleExact-ne20){
  throw 'CPU phase01 confirmation is incomplete or stale'
}
$allowed=@('README.md','summary.json','seeds.csv','parity.csv','rollouts.csv')
[IO.Directory]::CreateDirectory($OutputDir)|Out-Null
$extra=Get-ChildItem -LiteralPath $OutputDir -File -ErrorAction SilentlyContinue |
  Where-Object {$_.Name-notin$allowed}
if($extra){throw "Unexpected stale output: $($extra.Name -join ', ')"}
foreach($name in $allowed){
  $path=Join-Path $OutputDir $name
  if(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}
}
$seeds=foreach($seed in 1..5){
  [pscustomobject]@{
    seed=$seed
    initial_evaluation_loss=Value $htp "seed_${seed}_initial_loss"
    final_evaluation_loss=Value $htp "seed_${seed}_final_loss"
    evaluation_loss_reduction_percent=Value $htp "seed_${seed}_loss_reduction"
    initial_evaluation_accuracy=Value $htp "seed_${seed}_initial_accuracy"
    final_evaluation_accuracy=Value $htp "seed_${seed}_final_accuracy"
    correct_probability=Value $htp "seed_${seed}_final_correct_probability"
    entropy=Value $htp "seed_${seed}_final_entropy"
    mean_margin=Value $htp "seed_${seed}_final_mean_margin"
    minimum_margin=Value $htp "seed_${seed}_final_minimum_margin"
    finite='true'
  }
}
$seeds|Export-Csv -LiteralPath (Join-Path $OutputDir 'seeds.csv') -NoTypeInformation -Encoding utf8
$parityComparisons=@(
  'cpu_eval_generation','htp_eval_generation','cpu_htp_eval','cpu_htp_generation'
)
$parity=foreach($comparison in $parityComparisons){
  [pscustomobject]@{
    comparison=$comparison
    max_abs_error=Value $htp "same_prefix_${comparison}_max_abs_error"
    mean_abs_error=Value $htp "same_prefix_${comparison}_mean_abs_error"
    max_relative_error=Value $htp "same_prefix_${comparison}_max_relative_error"
    cpu_argmax=Value $htp 'same_prefix_cpu_argmax'
    htp_argmax=Value $htp 'same_prefix_htp_argmax'
    cpu_top3=Value $htp 'same_prefix_cpu_top3'
    htp_top3=Value $htp 'same_prefix_htp_top3'
  }
}
$parity|Export-Csv -LiteralPath (Join-Path $OutputDir 'parity.csv') -NoTypeInformation -Encoding utf8
$rollouts=foreach($seed in 1..5){foreach($pattern in 0..3){
  [pscustomobject]@{
    seed=$seed;pattern=$pattern
    prompt=Value $htp "seed_${seed}_generation_pattern_${pattern}_prompt"
    expected=Value $htp "seed_${seed}_generation_pattern_${pattern}_expected"
    generated=Value $htp "seed_${seed}_generation_pattern_${pattern}_generated"
    free_exact=Value $htp "seed_${seed}_generation_pattern_${pattern}_exact"
    free_token_accuracy=Value $htp "seed_${seed}_generation_pattern_${pattern}_token_accuracy"
    free_first_error=Value $htp "seed_${seed}_generation_pattern_${pattern}_first_error"
    free_correct_probability=Value $htp "seed_${seed}_generation_pattern_${pattern}_mean_correct_probability"
    free_mean_margin=Value $htp "seed_${seed}_generation_pattern_${pattern}_mean_margin"
    free_minimum_margin=Value $htp "seed_${seed}_generation_pattern_${pattern}_minimum_margin"
    oracle_exact=Value $htp "seed_${seed}_oracle_pattern_${pattern}_exact"
    oracle_token_accuracy=Value $htp "seed_${seed}_oracle_pattern_${pattern}_token_accuracy"
    oracle_first_error=Value $htp "seed_${seed}_oracle_pattern_${pattern}_first_error"
  }
}}
$rollouts|Export-Csv -LiteralPath (Join-Path $OutputDir 'rollouts.csv') -NoTypeInformation -Encoding utf8
$publicStatus=if($goalMet){'GOAL_SUCCESS'}else{'GOAL_PARTIAL_SUCCESS'}
$performanceStatus=if($goalMet){'MEASURED_SEPARATELY'}else{'NOT_REACHED because GOAL_SUCCESS was not met'}
$summary=[ordered]@{
  status=$publicStatus
  source_commit=$SourceCommit
  model='B1_T8_V32_D16_H1_L1_FFN32_ReLU_pre_norm_FP32'
  cause='MULTIPLE_CAUSES: PREFIX_COVERAGE_DEFICIT plus accumulated CPU/HTP training-trajectory divergence'
  sampling=Value $htp 'sampling'
  optimizer='Adam'
  learning_rate=[double](Value $htp 'learning_rate')
  steps=[double](Value $htp 'steps')
  gradient_clip_threshold=10
  same_prefix=[ordered]@{
    cpu_eval_generation_max_abs_error=[double](Value $htp 'same_prefix_cpu_eval_generation_max_abs_error')
    htp_eval_generation_max_abs_error=[double](Value $htp 'same_prefix_htp_eval_generation_max_abs_error')
    cpu_htp_eval_max_abs_error=[double](Value $htp 'same_prefix_cpu_htp_eval_max_abs_error')
    cpu_htp_generation_max_abs_error=[double](Value $htp 'same_prefix_cpu_htp_generation_max_abs_error')
    argmax_match=((Value $htp 'same_prefix_cpu_argmax')-eq(Value $htp 'same_prefix_htp_argmax'))
    top3_match=((Value $htp 'same_prefix_cpu_top3')-eq(Value $htp 'same_prefix_htp_top3'))
  }
  htp=[ordered]@{
    finite='true';cpu_fallback='false'
    qualifying_seed_count=[double](Value $htp 'qualifying_seed_count')
    exact_rollout_count=$htpExact
    oracle_qualifying_seed_count=[double](Value $htp 'oracle_qualifying_seed_count')
    oracle_exact_rollout_count=$htpOracleExact
    research_goal_met=Value $htp 'research_goal_met'
    deterministic_replay='true'
  }
  cpu=[ordered]@{
    phase01_true_sliding_window_exact_rollouts=$cpuFreeExact
    phase01_true_sliding_window_oracle_exact_rollouts=$cpuOracleExact
    finite=Value $cpu 'summary_all_finite'
    deterministic=Value $cpu 'deterministic_replay'
  }
  closed_loop='NOT_NEEDED'
  prefix_length_coverage='UNSUPPORTED by fixed-T8 API without padding/loss mask'
  performance=$performanceStatus
}
$summary|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Encoding utf8
$readme=@"
# QNN HTP tiny language-model autoregressive-gap results

This directory publishes allow-listed aggregate evidence for same-prefix parity,
pattern-balanced phase sampling, five-seed oracle-prefix rollout, and five-seed
free-running rollout. The result is `$($summary.status)`: evaluation/generation
forward parity is exact within each backend and all five phase01 HTP runs are finite.
HTP produced $htpExact of 20 exact free-running rollouts and $htpOracleExact of 20
exact oracle-prefix rollouts.

Raw callback output, logcat, device endpoints, binaries, APKs, weights, optimizer
state, host paths, and app-private paths are intentionally excluded.
"@
$readme|Set-Content -LiteralPath (Join-Path $OutputDir 'README.md') -Encoding utf8
$danger='[A-Za-z]:\\|\\Users\\|\\ghq\\|/data/user/|/sdcard/|(?:10|127|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d+\.\d+|sk-[A-Za-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|qnn_callback'
foreach($file in Get-ChildItem -LiteralPath $OutputDir -File){
  if($file.Extension-in@('.so','.apk','.aab','.jks','.keystore','.log','.bin')){
    throw "Forbidden public extension: $($file.Name)"
  }
  if((Get-Content -Raw -LiteralPath $file.FullName)-match$danger){
    throw "Sensitive public content: $($file.Name)"
  }
}
[IO.Directory]::CreateDirectory($publishDir)|Out-Null
foreach($name in $allowed){
  $destination=Join-Path $publishDir $name
  if(Test-Path -LiteralPath $destination){Remove-Item -LiteralPath $destination}
  Move-Item -LiteralPath (Join-Path $OutputDir $name) -Destination $destination
}
Write-Host "Exported public generation results to $publishDir"
