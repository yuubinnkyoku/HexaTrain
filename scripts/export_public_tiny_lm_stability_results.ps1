# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
  [string]$InputDir=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-htp-adam'),
  [string]$OutputDir=(Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-tiny-language-model-stability-2026-07'),
  [string]$SourceCommit=''
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(!$SourceCommit){$SourceCommit=(git -C (Split-Path -Parent $PSScriptRoot) rev-parse HEAD).Trim()}
if($SourceCommit-notmatch'^[0-9a-f]{40}$'){throw 'SourceCommit must be a full Git SHA'}
function Read-Fields([string]$Name){
  $path=Join-Path $InputDir $Name
  if(!(Test-Path -LiteralPath $path)){throw "Missing raw result: $Name"}
  $map=@{}
  foreach($line in Get-Content -LiteralPath $path){
    if($line-match'^([^=]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}
  }
  $map
}
function Value($Map,[string]$Key){
  if(!$Map.ContainsKey($Key)){throw "Missing field: $Key"}
  $Map[$Key]
}
function Number($Map,[string]$Key){[double](Value $Map $Key)}
$diagnostic=Read-Fields 'one-step-result.txt'
$clip5=Read-Fields 'candidate-1-result.txt'
$final=Read-Fields 'candidate-2-result.txt'
$inference=Read-Fields 'inference-result.txt'
if((Value $diagnostic 'status')-ne'SUCCESS'){throw 'Diagnostic result is not SUCCESS'}
if((Value $final 'status')-ne'SUCCESS' -or
   (Value $final 'nan_detected')-ne'false' -or
   (Value $final 'inf_detected')-ne'false' -or
   (Value $final 'cpu_fallback')-ne'false'){throw 'Final convergence result is not publishable'}
if((Value $final 'all_seeds_loss_decreased')-ne'true' -or
   (Value $final 'all_seeds_accuracy_increased')-ne'true' -or
   (Value $final 'additional_convergence_condition')-ne'true'){
  throw 'Final convergence gates were not met'
}
if((Value $inference 'status')-ne'FAILED' -or (Value $inference 'nan_detected')-ne'false'){
  throw 'Expected a finite, threshold-missing inference result'
}
$allowed=@('summary.json','seeds.csv','trajectory.csv','synchronized-checkpoints.csv','inference.csv','README.md')
[IO.Directory]::CreateDirectory($OutputDir)|Out-Null
$extras=Get-ChildItem -LiteralPath $OutputDir -File -ErrorAction SilentlyContinue |
  Where-Object {$_.Name-notin$allowed}
if($extras){throw "Unexpected stale output: $($extras.Name -join ', ')"}
foreach($name in $allowed){
  $path=Join-Path $OutputDir $name
  if(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}
}
$seeds=foreach($seed in 1..5){
  [pscustomobject]@{
    seed=$seed
    initial_evaluation_loss=Value $final "seed_${seed}_initial_loss"
    final_evaluation_loss=Value $final "seed_${seed}_final_loss"
    evaluation_loss_reduction_percent=Value $final "seed_${seed}_loss_reduction"
    initial_evaluation_accuracy=Value $final "seed_${seed}_initial_accuracy"
    final_evaluation_accuracy=Value $final "seed_${seed}_final_accuracy"
    final_correct_token_probability=Value $final "seed_${seed}_final_correct_probability"
    final_entropy=Value $final "seed_${seed}_final_entropy"
    final_mean_logit_margin=Value $final "seed_${seed}_final_mean_margin"
    final_minimum_logit_margin=Value $final "seed_${seed}_final_minimum_margin"
    cpu_htp_parameter_max_abs_difference=Value $final "seed_${seed}_cpu_htp_parameter_max_abs_difference"
    finite='true'
  }
}
$seeds|Export-Csv -LiteralPath (Join-Path $OutputDir 'seeds.csv') -NoTypeInformation -Encoding utf8
$steps=@(1,2,5,10,20,50,100,200,320,640,1000)
$trajectory=foreach($seed in 1..5){foreach($step in $steps){
  [pscustomobject]@{
    seed=$seed;step=$step
    train_loss=Value $final "seed_${seed}_step_${step}_loss"
    train_accuracy=Value $final "seed_${seed}_step_${step}_accuracy"
    gradient_l2_norm=Value $final "seed_${seed}_step_${step}_global_gradient_l2_norm"
    update_l2_norm=Value $final "seed_${seed}_step_${step}_global_update_l2_norm"
    parameter_l2_norm=Value $final "seed_${seed}_step_${step}_global_parameter_l2_norm"
    first_moment_l2_norm=Value $final "seed_${seed}_step_${step}_first_moment_l2_norm"
    second_moment_l2_norm=Value $final "seed_${seed}_step_${step}_second_moment_l2_norm"
  }
}}
$trajectory|Export-Csv -LiteralPath (Join-Path $OutputDir 'trajectory.csv') -NoTypeInformation -Encoding utf8
$checkpoints=foreach($step in @(0,1,2,5,10,20,50,100,150,200,250,300,320)){
  [pscustomobject]@{
    checkpoint=$step
    logits_max_abs_error=Value $diagnostic "checkpoint_${step}_logits_max_abs_error"
    probabilities_max_abs_error=Value $diagnostic "checkpoint_${step}_probabilities_max_abs_error"
    dlogits_max_abs_error=Value $diagnostic "checkpoint_${step}_dlogits_max_abs_error"
    token_embedding_gradient_max_abs_error=Value $diagnostic "checkpoint_${step}_gradient_token_embedding_max_abs_error"
    path_c_optimizer_max_abs_error=Value $diagnostic "checkpoint_${step}_path_c_cpu_gradient_htp_optimizer_max_abs_error"
    path_d_optimizer_max_abs_error=Value $diagnostic "checkpoint_${step}_path_d_htp_gradient_htp_optimizer_max_abs_error"
    minimum_v_hat=Value $diagnostic "checkpoint_${step}_optimizer_v_hat_min"
    minimum_sqrt_v_hat=Value $diagnostic "checkpoint_${step}_optimizer_sqrt_v_hat_min"
    minimum_denominator=Value $diagnostic "checkpoint_${step}_optimizer_denominator_min"
  }
}
$checkpoints|Export-Csv -LiteralPath (Join-Path $OutputDir 'synchronized-checkpoints.csv') -NoTypeInformation -Encoding utf8
$patterns=foreach($pattern in 0..3){
  [pscustomobject]@{
    pattern=$pattern
    prompt=Value $inference "generation_pattern_${pattern}_prompt"
    expected=Value $inference "generation_pattern_${pattern}_expected"
    generated=Value $inference "generation_pattern_${pattern}_generated"
    exact=Value $inference "generation_pattern_${pattern}_exact"
    token_accuracy=Value $inference "generation_pattern_${pattern}_token_accuracy"
    mean_correct_probability=Value $inference "generation_pattern_${pattern}_mean_correct_probability"
    mean_logit_margin=Value $inference "generation_pattern_${pattern}_mean_margin"
    minimum_logit_margin=Value $inference "generation_pattern_${pattern}_minimum_margin"
    fallback='false';nan_inf='false'
  }
}
$patterns|Export-Csv -LiteralPath (Join-Path $OutputDir 'inference.csv') -NoTypeInformation -Encoding utf8
$summary=[ordered]@{
  status='GOAL_PARTIAL_SUCCESS'
  stop_reason='Five-seed HTP convergence passed, but autoregressive exact continuation was 0/4 rather than at least 3/4.'
  source_commit=$SourceCommit
  model='B1_T8_V32_D16_H1_L1_FFN32_ReLU_pre_norm_FP32'
  diagnostic=[ordered]@{
    learning_rate=Number $diagnostic 'learning_rate'
    checkpoints=Value $diagnostic 'checkpoint_steps'
    checkpoint_roundtrip=Value $diagnostic 'checkpoint_save_load_deterministic'
    one_step_correct=Value $diagnostic 'one_step_correct'
    first_major_divergence_checkpoint=Number $diagnostic 'first_major_divergence_checkpoint'
    first_major_divergence_tensor=Value $diagnostic 'first_major_divergence_tensor'
    first_major_divergence_node=Value $diagnostic 'first_major_divergence_node'
    first_optimizer_divergence_checkpoint=Number $diagnostic 'first_optimizer_divergence_checkpoint'
    classification=Value $diagnostic 'first_divergence_classification'
    last_finite_step=Number $diagnostic 'free_trajectory_last_finite_step'
    first_nonfinite_step=Number $diagnostic 'free_trajectory_first_nonfinite_step'
    first_nonfinite_tensor=Value $diagnostic 'free_trajectory_first_nonfinite_tensor'
  }
  final=[ordered]@{
    configuration=Value $final 'configuration_id'
    learning_rate=Number $final 'learning_rate'
    steps=Number $final 'steps'
    seeds=5
    gradient_clip_threshold=Number $final 'global_gradient_clip_threshold'
    median_evaluation_loss_reduction_percent=Number $final 'median_loss_reduction'
    accuracy_75_seed_count=Number $final 'accuracy_75_seed_count'
    all_seeds_loss_decreased=Value $final 'all_seeds_loss_decreased'
    all_seeds_accuracy_increased=Value $final 'all_seeds_accuracy_increased'
    all_steps_finite='true'
    cpu_fallback='false'
    graph_count=Number $final 'graph_count'
    executes_per_update=Number $final 'execute_count_per_training_step'
    cpu_scalar_boundary='Adam bias correction and global gradient clip scale'
    htp_responsibility='Forward, backward, gradient scaling, and Adam arithmetic'
  }
  performance=[ordered]@{
    graph_initialization_us=Number $final 'graph_initialization_us'
    graph_create_us=Number $final 'graph_create_us'
    graph_finalize_us=Number $final 'graph_finalize_us'
    first_execute_us=Number $final 'first_execute_us'
    steady_execute_mean_us=Number $final 'steady_execute_mean_us'
    updates_per_second_estimate=Number $final 'updates_per_second_estimate'
    tokens_per_second_estimate=Number $final 'tokens_per_second_estimate'
  }
  inference=[ordered]@{
    exact_pattern_count=Number $inference 'exact_pattern_count'
    required_exact_pattern_count=3
    finite='true'
  }
  rejected=[ordered]@{
    epsilon='NOT_NEEDED'
    learning_rates='0.003 and 0.001 remained unstable under repeated updates'
    gradient_clip_thresholds='1, 5, and 10 evaluated'
    update_clipping='REJECTED: did not prevent non-finite state in bounded trials'
  }
}
$summary|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Encoding utf8
$readme=@"
# QNN HTP tiny language-model numerical-stability results

This directory contains public, aggregate-only evidence for the synchronized checkpoint,
2x2 gradient/optimizer split, five-seed convergence, and four-pattern inference study.
The result is `GOAL_PARTIAL_SUCCESS`: all five HTP training seeds remained finite and met
the evaluation-loss convergence gate, while autoregressive continuation did not meet 3/4.

Raw weights, optimizer states, callback output, logs, device endpoints, binaries, APKs,
host paths, and app-private paths are intentionally excluded.
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
Write-Host "Exported public stability results to $OutputDir"
