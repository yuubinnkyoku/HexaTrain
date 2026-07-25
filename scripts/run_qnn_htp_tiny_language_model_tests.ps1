# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$adb=Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$package='com.yuubinnkyoku.phonelm'
$activity="$package/.MainActivity"
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot=Join-Path $root 'build\reports\qnn-tiny-language-model'
[IO.Directory]::CreateDirectory($reportRoot)|Out-Null
if(!$SkipBuild){
  & (Join-Path $root 'gradlew.bat') :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
  if($LASTEXITCODE-ne0){throw 'APK build failed'}
}
$online=@()
foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne1){throw "Expected one online ADB device; found $($online.Count)"}
$device=$online[0]
function Adb([string[]]$Arguments){
  $output=& $adb -s $device @Arguments 2>&1
  if($LASTEXITCODE-ne0){throw "ADB command failed (endpoint redacted): $($Arguments-join ' ')`n$output"}
  $output
}
function Require([string]$Text,[string]$Pattern,[string]$Description){
  if($Text-notmatch $Pattern){throw "Missing $Description"}
}
function Run-Test([string]$Mode,[string]$Name,[string[]]$Required,[int]$PollLimit=2400){
  Adb @('shell','am','force-stop',$package)|Out-Null
  & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
  Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$Mode)|Out-Null
  $result=''
  for($poll=0;$poll-lt$PollLimit;$poll++){
    Start-Sleep -Milliseconds 500
    $result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n"
    if($result-match '(?m)^status=(SUCCESS|FAILED)$'){break}
  }
  Require $result '(?m)^status=SUCCESS$' "$Mode success"
  Require $result '(?m)^cpu_fallback=false$' 'no CPU backend fallback'
  Require $result '(?m)^nan_detected=false$' 'no NaN'
  Require $result '(?m)^inf_detected=false$' 'no Inf'
  Require $result '(?m)^api_trace_graph_execute_failure_count=0$' 'zero graph execute failures'
  foreach($pattern in $Required){Require $result $pattern "$Mode evidence $pattern"}
  [IO.File]::WriteAllText((Join-Path $reportRoot "$Name-result.txt"),$result+"`n",[Text.UTF8Encoding]::new($false))
  Write-Host "PASS $Mode"
}
Run-Test 'QNN_HTP_CROSS_ENTROPY_CHECK' 'cross-entropy' @(
  '(?m)^shape=B2_T3_V8$','(?m)^dlogits_max_abs_error=','(?m)^cross_entropy_gradient=HTP$'
)
Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_STEP' 'one-step' @(
  '(?m)^shape=B1_T8_V32_D16_H1_L1_F32$','(?m)^major_weight_changed=true$','(?m)^execute_count_per_step=1$'
)
Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_MULTI_STEP' 'multi-step' @(
  '(?m)^steps=320$','(?m)^seeds=5$','(?m)^all_seeds_loss_decreased=true$','(?m)^all_seeds_accuracy_increased=true$','(?m)^deterministic_replay=true$','(?m)^graph_execute_count=1605$'
) 4000
Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_INFERENCE' 'inference' @(
  '(?m)^expected_continuation=0,1,2,3,0,1,2,3$','(?m)^argmax_responsibility=CPU$','(?m)^logits_responsibility=HTP$'
) 3000
Write-Host 'QNN HTP tiny language model tests passed. Raw reports remain under build/reports.'
