# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
    [switch]$SkipInstall,
    [ValidateSet('step','candidate1','candidate2','inference','all')]
    [string]$Scope='all'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot
$adb=Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$package='com.yuubinnkyoku.phonelm'
$activity="$package/.MainActivity"
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot=Join-Path $root 'build\reports\tiny-lm-htp-adam'
[IO.Directory]::CreateDirectory($reportRoot)|Out-Null
$online=@()
foreach($line in (& $adb devices)){if($line-match'^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne1){throw "Expected one online device; found $($online.Count)"}
$device=$online[0]
function Adb([string[]]$Arguments){
  $output=& $adb -s $device @Arguments 2>&1
  if($LASTEXITCODE-ne0){throw "ADB command failed (endpoint redacted): $($Arguments-join ' ')`n$output"}
  $output
}
if(!$SkipInstall){Adb @('install','-r',$apk)|Out-Null}
function Run-Test([string]$Mode,[string]$Name,[string[]]$Required,[int]$PollLimit=14400,[switch]$AllowPartial){
  Adb @('shell','am','force-stop',$package)|Out-Null
  & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
  Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$Mode)|Out-Null
  $result=''
  for($poll=0;$poll-lt$PollLimit;$poll++){
    Start-Sleep -Milliseconds 500
    $result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join"`n"
    if($result-match'(?m)^status=(SUCCESS|PARTIAL_SUCCESS|FAILED)$'){break}
  }
  [IO.File]::WriteAllText((Join-Path $reportRoot "$Name-result.txt"),$result+"`n",[Text.UTF8Encoding]::new($false))
  $accepted=$result-match'(?m)^status=SUCCESS$' -or
    ($AllowPartial -and $result-match'(?m)^status=PARTIAL_SUCCESS$')
  if(!$accepted){throw "$Mode failed"}
  foreach($pattern in $Required){if($result-notmatch$pattern){throw "$Mode missing $pattern"}}
  $common=@('(?m)^cpu_fallback=false$','(?m)^api_trace_graph_execute_failure_count=0$')
  if($Mode-ne'QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP'){
    $common+=@('(?m)^nan_detected=false$','(?m)^inf_detected=false$')
  }
  foreach($pattern in $common){
    if($result-notmatch$pattern){throw "$Mode missing $pattern"}
  }
  Write-Host "PASS $Mode"
}
if($Scope-in@('step','all')){
  Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_STEP' 'one-step' @(
    '(?m)^optimizer=ADAM$','(?m)^graph_count=2$','(?m)^checkpoint_count=13$',
    '(?m)^checkpoint_save_load_deterministic=true$',
    '(?m)^one_step_correct=true$',
    '(?m)^first_divergence_classification=',
    '(?m)^path_a=CPU_GRADIENT_CPU_OPTIMIZER$',
    '(?m)^path_b=HTP_GRADIENT_CPU_OPTIMIZER$',
    '(?m)^path_c=CPU_GRADIENT_HTP_OPTIMIZER$',
    '(?m)^path_d=HTP_GRADIENT_HTP_OPTIMIZER$',
    '(?m)^major_weight_changed=true$','(?m)^bias_correction_scalar_responsibility=CPU$',
    '(?m)^optimizer_math_responsibility=HTP$')
}
if($Scope-in@('candidate1','all')){
  Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_1' 'candidate-1' @(
    '(?m)^additional_convergence_condition=true$','(?m)^all_seeds_loss_decreased=true$',
    '(?m)^all_seeds_accuracy_increased=true$','(?m)^execute_count_per_training_step=2$')
}
if($Scope-in@('candidate2','all')){
  Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_CANDIDATE_2' 'candidate-2' @(
    '(?m)^additional_convergence_condition=true$','(?m)^all_seeds_loss_decreased=true$',
    '(?m)^all_seeds_accuracy_increased=true$','(?m)^execute_count_per_training_step=2$')
}
if($Scope-in@('inference','all')){
  Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_ADAM_INFERENCE' 'inference' @(
    '(?m)^research_goal_met=(true|false)$',
    '(?m)^sampling=pattern_balanced_phase01_round_robin$',
    '(?m)^seed_count=5$','(?m)^qualifying_seed_count=\d+$',
    '(?m)^exact_rollout_count=\d+$',
    '(?m)^oracle_qualifying_seed_count=\d+$',
    '(?m)^oracle_exact_rollout_count=\d+$',
    '(?m)^same_prefix_host_inputs_identical=true$',
    '(?m)^same_prefix_cpu_eval_generation_max_abs_error=0$',
    '(?m)^same_prefix_htp_eval_generation_max_abs_error=0$',
    '(?m)^same_prefix_seed_count=5$',
    '(?m)^same_prefix_all_cpu_eval_generation_argmax_match=true$',
    '(?m)^same_prefix_all_htp_eval_generation_argmax_match=true$',
    '(?m)^same_prefix_all_cpu_eval_generation_logits_match=true$',
    '(?m)^same_prefix_all_htp_eval_generation_logits_match=true$',
    '(?m)^same_prefix_all_cpu_htp_argmax_match=true$',
    '(?m)^same_prefix_all_cpu_htp_top3_match=true$',
    '(?m)^logits_responsibility=HTP$','(?m)^argmax_responsibility=CPU$') -AllowPartial
}
Write-Host 'HTP_ADAM=PASS'
