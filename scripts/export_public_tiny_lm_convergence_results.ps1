# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
  [string]$ReportRoot=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports'),
  [string]$OutputDir=(Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-tiny-language-model-2026-07')
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
function Read-Map([string]$Path){
  if(!(Test-Path -LiteralPath $Path)){throw "Missing source report: $Path"}
  $map=@{}
  foreach($line in Get-Content -LiteralPath $Path){
    if($line-match'^([A-Za-z0-9_]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}
  }
  $map
}
function Value([hashtable]$Map,[string]$Key,[string]$Default='NOT_RECORDED'){
  if($Map.ContainsKey($Key)){$Map[$Key]}else{$Default}
}
[IO.Directory]::CreateDirectory($OutputDir)|Out-Null
$sweepRows=@()
foreach($spec in @(
  @{optimizer='SGD';path='tiny-lm-cpu-sweep\configurations.csv'},
  @{optimizer='MOMENTUM_SGD';path='tiny-lm-cpu-momentum-sweep\configurations.csv'},
  @{optimizer='ADAM';path='tiny-lm-cpu-adam-sweep\configurations.csv'})){
  foreach($row in Import-Csv (Join-Path $ReportRoot $spec.path)){
    $sweepRows+=[pscustomobject][ordered]@{
      optimizer=$spec.optimizer;configuration_id=$row.id;learning_rate=$row.lr
      steps=$row.steps;momentum=if($row.PSObject.Properties.Name-contains'momentum'){$row.momentum}else{'NOT_APPLICABLE'}
      median_loss_reduction_percent=$row.median;minimum_loss_reduction_percent=$row.minimum
      seeds_at_or_above_75_percent=$row.accuracy75
      all_seeds_loss_decreased=$row.allLoss;all_seeds_accuracy_increased=$row.allAccuracy
      nan_inf_count=$row.nanInf;deterministic_replay=$row.deterministic
    }
  }
}
$sweepRows|Export-Csv (Join-Path $OutputDir 'convergence-sweeps.csv') -NoTypeInformation -Encoding utf8
$candidateRows=@()
foreach($spec in @(
  @{optimizer='SGD';name='sgd-1';path='tiny-lm-htp-convergence\candidate-1-result.txt'},
  @{optimizer='SGD';name='sgd-2';path='tiny-lm-htp-convergence\candidate-2-result.txt'},
  @{optimizer='SGD';name='sgd-3';path='tiny-lm-htp-convergence\candidate-3-result.txt'},
  @{optimizer='MOMENTUM_SGD';name='momentum-1';path='tiny-lm-htp-momentum\candidate-1-result.txt'},
  @{optimizer='MOMENTUM_SGD';name='momentum-2';path='tiny-lm-htp-momentum\candidate-2-result.txt'},
  @{optimizer='ADAM';name='adam-1';path='tiny-lm-htp-adam\candidate-1-result.txt'},
  @{optimizer='ADAM';name='adam-2';path='tiny-lm-htp-adam\candidate-2-result.txt'})){
  $map=Read-Map (Join-Path $ReportRoot $spec.path)
  $candidateRows+=[pscustomobject][ordered]@{
    optimizer=$spec.optimizer;candidate=$spec.name;configuration_id=Value $map 'configuration_id'
    status=Value $map 'status';learning_rate=Value $map 'learning_rate';steps=Value $map 'steps'
    median_loss_reduction_percent=Value $map 'median_loss_reduction'
    seeds_at_or_above_75_percent=Value $map 'accuracy_75_seed_count'
    all_seeds_loss_decreased=Value $map 'all_seeds_loss_decreased'
    all_seeds_accuracy_increased=Value $map 'all_seeds_accuracy_increased'
    additional_convergence_condition=Value $map 'additional_convergence_condition'
    nan_detected=Value $map 'nan_detected';cpu_fallback=Value $map 'cpu_fallback'
    graph_count=Value $map 'graph_count';execute_count_per_training_step=Value $map 'execute_count_per_training_step'
  }
}
$candidateRows|Export-Csv (Join-Path $OutputDir 'htp-convergence-candidates.csv') -NoTypeInformation -Encoding utf8
$momentumOne=Read-Map (Join-Path $ReportRoot 'tiny-lm-htp-momentum\one-step-result.txt')
$adamOne=Read-Map (Join-Path $ReportRoot 'tiny-lm-htp-adam\one-step-result.txt')
$summary=[ordered]@{
  status='PARTIAL_SUCCESS'
  stop_reason='Bounded SGD, Momentum SGD, and Adam HTP candidates did not meet the additional convergence condition without NaN/Inf.'
  shape='B1_T8_V32_D16_H1_L1_F32'
  dataset=[ordered]@{
    patterns=4;train_sequences=4;evaluation_sequences=4;train_phase=0;evaluation_phase=1
    input_token_frequency='0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:4,9:4,10:3,11:3,12:2'
    target_token_frequency='0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:4,9:4,10:2,11:3,12:3'
    train_evaluation_leakage=$false;future_token_leakage=$false;causal_mask=$true
  }
  search_counts=[ordered]@{sgd_cpu=48;sgd_htp=3;momentum_cpu=24;momentum_htp=2;adam_cpu=12;adam_htp=2}
  cpu_best=[ordered]@{
    sgd='lr=0.1,steps=1000,init=1,shuffle,median=99.90563384%,accuracy75=5'
    momentum='lr=0.01,momentum=0.95,steps=1000,fixed,median=98.80187239%,accuracy75=5'
    adam='lr=0.003,steps=320,fixed,median=94.56031737%,accuracy75=5'
  }
  htp_classification='NUMERICALLY_UNSTABLE_REPEATED_UPDATES'
  momentum_one_step=[ordered]@{
    status=Value $momentumOne 'status';gradient_max_abs_error=Value $momentumOne 'gradient_max_abs_error'
    next_velocity_max_abs_error=Value $momentumOne 'next_velocity_max_abs_error'
    next_parameter_max_abs_error=Value $momentumOne 'next_parameter_max_abs_error'
  }
  adam_one_step=[ordered]@{
    status=Value $adamOne 'status';gradient_max_abs_error=Value $adamOne 'gradient_max_abs_error'
    first_moment_next_max_abs_error=Value $adamOne 'first_moment_next_max_abs_error'
    second_moment_next_max_abs_error=Value $adamOne 'second_moment_next_max_abs_error'
    first_moment_hat_max_abs_error=Value $adamOne 'first_moment_hat_max_abs_error'
    second_moment_hat_max_abs_error=Value $adamOne 'second_moment_hat_max_abs_error'
    next_parameter_max_abs_error=Value $adamOne 'next_parameter_max_abs_error'
  }
  boundary=[ordered]@{
    cpu='token/one-hot construction, batch and graph control, stable CE scalar, bias-correction scalars, independent reference, buffer handoff'
    htp='embedding, Transformer forward/backward, logits, Softmax, CE dLogits, parameter gradients, Momentum or Adam optimizer arithmetic'
    cpu_backend_fallback=$false
  }
  final_optimizer='NOT_ESTABLISHED'
  four_pattern_inference='NOT_REACHED'
  performance='NOT_REACHED'
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'convergence-summary.json'),($summary|ConvertTo-Json -Depth 6)+"`n",[Text.UTF8Encoding]::new($false))
Write-Host "Exported allow-listed convergence results to $OutputDir"
