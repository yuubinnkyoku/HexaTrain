# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
  [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
  [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
  [switch]$SkipInstall,
  [ValidateRange(1,3)][int]$StartCandidate=1
)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
  -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$adb=Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$package='com.yuubinnkyoku.phonelm';$activity="$package/.MainActivity"
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot=Join-Path $root 'build\reports\tiny-lm-htp-convergence'
[IO.Directory]::CreateDirectory($reportRoot)|Out-Null
$online=@();foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne1){throw "Expected one online device; found $($online.Count)"};$device=$online[0]
function Adb([string[]]$Arguments){$output=& $adb -s $device @Arguments 2>&1;if($LASTEXITCODE-ne0){throw "ADB command failed (endpoint redacted): $($Arguments-join ' ')`n$output"};$output}
if(!$SkipInstall){Adb @('install','-r',$apk)|Out-Null}
function Run-Candidate([string]$Mode,[string]$Name,[int]$PollLimit=7200){
  Adb @('shell','am','force-stop',$package)|Out-Null;& $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
  Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$Mode)|Out-Null
  $result='';for($poll=0;$poll-lt$PollLimit;$poll++){Start-Sleep -Milliseconds 500;$result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join"`n";if($result-match '(?m)^status=(SUCCESS|FAILED)$'){break}}
  [IO.File]::WriteAllText((Join-Path $reportRoot "$Name-result.txt"),$result+"`n",[Text.UTF8Encoding]::new($false))
  if($result-notmatch '(?m)^status=SUCCESS$'){throw "$Mode failed"}
  foreach($pattern in @('(?m)^additional_convergence_condition=true$','(?m)^all_seeds_loss_decreased=true$','(?m)^all_seeds_accuracy_increased=true$','(?m)^cpu_fallback=false$','(?m)^nan_detected=false$','(?m)^inf_detected=false$','(?m)^api_trace_graph_execute_failure_count=0$')){if($result-notmatch$pattern){throw "$Mode missing $pattern"}}
  Write-Host "PASS $Mode"
}
$tests=@(
  @{index=1;mode='QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_1';name='candidate-1'},
  @{index=2;mode='QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_2';name='candidate-2'},
  @{index=3;mode='QNN_HTP_TINY_LANGUAGE_MODEL_SGD_CANDIDATE_3';name='candidate-3'}
)
$failures=@();foreach($test in $tests){if($test.index-lt$StartCandidate){continue};try{Run-Candidate $test.mode $test.name}catch{$failures+=$test.name;Write-Warning $_}}
if($failures.Count){throw "HTP candidate failures: $($failures-join ',')"}
Write-Host 'HTP_CANDIDATES=PASS'
