# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$SourceReportDir,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [Parameter(Mandatory=$true)][string]$ExperimentSourceCommit
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Read-KeyValue([string]$Path){
  if(!(Test-Path -LiteralPath $Path -PathType Leaf)){throw "Required report missing: $Path"}
  $map=@{}
  foreach($line in Get-Content -LiteralPath $Path){
    if($line-match '^([A-Za-z0-9_]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}
  }
  $map
}
function Need([hashtable]$Map,[string]$Key){
  if(!$Map.ContainsKey($Key)){throw "Missing allow-listed key: $Key"}
  $Map[$Key]
}
function Number([string]$Value){[double]::Parse($Value,[Globalization.CultureInfo]::InvariantCulture)}
if($ExperimentSourceCommit-notmatch '^[0-9a-fA-F]{40}$'){throw 'ExperimentSourceCommit must be a full commit hash'}
$ce=Read-KeyValue (Join-Path $SourceReportDir 'cross-entropy-final-result.txt')
$one=Read-KeyValue (Join-Path $SourceReportDir 'one-step-final-result.txt')
$multi=Read-KeyValue (Join-Path $SourceReportDir 'multi-step-fixed-buffer-result.txt')
$inference=Read-KeyValue (Join-Path $SourceReportDir 'inference-result.txt')
foreach($report in @($ce,$one,$multi,$inference)){
  if((Need $report 'status')-ne'SUCCESS'){throw 'Source report is not successful'}
  if((Need $report 'cpu_fallback')-ne'false'){throw 'Source report used fallback'}
  if((Need $report 'nan_detected')-ne'false'-or(Need $report 'inf_detected')-ne'false'){throw 'Source report is non-finite'}
}
[IO.Directory]::CreateDirectory($OutputDir)|Out-Null
$reductions=@()
$seedRows=for($seed=1;$seed-le5;$seed++){
  $initial=Number (Need $multi "seed_${seed}_initial_loss")
  $final=Number (Need $multi "seed_${seed}_final_loss")
  $reduction=100.0*($initial-$final)/$initial
  $reductions+=$reduction
  [pscustomobject][ordered]@{
    seed=$seed;initial_loss=$initial;final_loss=$final;loss_reduction_percent=$reduction
    initial_accuracy=Number (Need $multi "seed_${seed}_initial_accuracy")
    final_accuracy=Number (Need $multi "seed_${seed}_final_accuracy")
    parameter_norm=Number (Need $multi "seed_${seed}_parameter_norm")
    cpu_htp_parameter_max_abs_difference=Number (Need $multi "seed_${seed}_cpu_htp_parameter_max_abs_difference")
  }
}
$seedRows|Export-Csv -LiteralPath (Join-Path $OutputDir 'seeds.csv') -NoTypeInformation -Encoding utf8
$trajectoryRows=foreach($seed in 1..5){
  foreach($step in @(0,1,2,5,10,20,50,100,320)){
    $lossKey=if($step-eq0){"seed_${seed}_initial_loss"}else{"seed_${seed}_step_${step}_loss"}
    $accuracyKey=if($step-eq0){"seed_${seed}_initial_accuracy"}else{"seed_${seed}_step_${step}_accuracy"}
    [pscustomobject][ordered]@{seed=$seed;step=$step;loss=Number (Need $multi $lossKey);token_accuracy=Number (Need $multi $accuracyKey);split='train_batch'}
  }
  [pscustomobject][ordered]@{seed=$seed;step=320;loss=Number (Need $multi "seed_${seed}_final_loss");token_accuracy=Number (Need $multi "seed_${seed}_final_accuracy");split='evaluation'}
}
$trajectoryRows|Export-Csv -LiteralPath (Join-Path $OutputDir 'trajectory.csv') -NoTypeInformation -Encoding utf8
$sorted=@($reductions|Sort-Object)
$summary=[ordered]@{
  status='PARTIAL_SUCCESS';experiment_source_commit=$ExperimentSourceCommit.ToLowerInvariant()
  shape=Need $multi 'shape';steps=[int](Need $multi 'steps');seeds=[int](Need $multi 'seeds')
  learning_rate=Number (Need $multi 'learning_rate');dataset_seed=Need $multi 'dataset_seed'
  all_seeds_loss_decreased=$true;all_seeds_accuracy_increased=$true
  median_loss_reduction_percent=$sorted[2]
  seeds_at_or_above_75_percent_evaluation_accuracy=@($seedRows|Where-Object{$_.final_accuracy-ge0.75}).Count
  deterministic_replay=(Need $multi 'deterministic_replay')-eq'true'
  graph_count=[int](Need $multi 'graph_count');execute_count_per_training_step=[int](Need $multi 'execute_count_per_training_step')
  graph_execute_count=[int](Need $multi 'graph_execute_count')
  cpu_htp_parameter_max_abs_difference=Number (Need $multi 'cpu_htp_parameter_max_abs_difference')
  cross_entropy=[ordered]@{loss_scalar='CPU_STABLE_LOGSUMEXP';gradient='HTP';dlogits_max_abs_error=Number (Need $ce 'dlogits_max_abs_error')}
  one_step=[ordered]@{loss_abs_error=Number (Need $one 'loss_abs_error');gradient_max_abs_error=Number (Need $one 'gradient_max_abs_error');next_parameter_max_abs_error=Number (Need $one 'next_parameter_max_abs_error')}
  inference=[ordered]@{expected_continuation=Need $inference 'expected_continuation';argmax='CPU';logits='HTP'}
  qairt_build_id=Need $multi 'compile_time_sdk_build_id';qnn_api=Need $multi 'compile_time_qnn_api_version'
  cpu_fallback=$false;nan_inf=$false
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'summary.json'),($summary|ConvertTo-Json -Depth 5)+"`n",[Text.UTF8Encoding]::new($false))
Write-Host "Exported allow-listed tiny LM results to $OutputDir"
