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
$reportRoot=Join-Path $root 'build\reports\qnn-htp-transformer-backward-tests'
[IO.Directory]::CreateDirectory($reportRoot)|Out-Null
if(!$SkipBuild){
  & (Join-Path $root 'gradlew.bat') :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
  if($LASTEXITCODE-ne 0){throw 'APK build failed'}
}
$online=@()
foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne 1){throw "Expected one online ADB device; found $($online.Count)"}
$device=$online[0]
function Adb([string[]]$Arguments){$out=& $adb -s $device @Arguments 2>&1;if($LASTEXITCODE-ne 0){throw "ADB command failed (endpoint redacted): $($Arguments-join ' ')`n$out"};$out}
function Require([string]$Text,[string]$Pattern,[string]$Description){if($Text-notmatch $Pattern){throw "Missing $Description"}}
Adb @('install','-r',$apk)|Out-Null
$tests=@(
  @{mode='QNN_HTP_SOFTMAX_BACKWARD_CHECK';name='softmax-backward';required=@(
    '(?m)^shape=B1_H1_T4_D4$','(?m)^epsilon=0\.001','(?m)^cpu_analytic_vs_numeric_max_abs_error=',
    '(?m)^htp_vs_cpu_max_abs_error=','(?m)^max_row_gradient_sum_abs=',
    '(?m)^graph_create_result=0$','(?m)^graph_finalize_result=0$','(?m)^graph_execute_result=0$')},
  @{mode='QNN_HTP_ATTENTION_BACKWARD_CHECK';name='attention-backward';required=@(
    '(?m)^shape=B1_H1_T4_D8$','(?m)^causal_mask=true$','(?m)^mask_gradient=false$',
    '(?m)^cpu_analytic_vs_numeric_dq_max_abs_error=','(?m)^cpu_analytic_vs_numeric_dk_max_abs_error=',
    '(?m)^cpu_analytic_vs_numeric_dv_max_abs_error=','(?m)^htp_vs_cpu_dq_max_abs_error=',
    '(?m)^htp_vs_cpu_dk_max_abs_error=','(?m)^htp_vs_cpu_dv_max_abs_error=',
    '(?m)^future_probability_max=0(?:\.0+)?$','(?m)^future_dscores_max=0(?:\.0+)?$',
    '(?m)^graph_create_result=0$','(?m)^graph_finalize_result=0$','(?m)^graph_execute_result=0$')}
)
foreach($test in $tests){
  Adb @('shell','am','force-stop',$package)|Out-Null
  & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
  Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$test.mode)|Out-Null
  $result='';for($poll=0;$poll-lt 240;$poll++){Start-Sleep -Milliseconds 500;$result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n";if($result-match '(?m)^status=(SUCCESS|FAILED)$'){break}}
  Require $result '(?m)^status=SUCCESS$' "$($test.mode) success";Require $result '(?m)^cpu_fallback=false$' 'no fallback';Require $result '(?m)^nan_detected=false$' 'no NaN';Require $result '(?m)^inf_detected=false$' 'no Inf';Require $result '(?m)^api_trace_graph_execute_failure_count=0$' 'zero QNN execute failures'
  foreach($pattern in $test.required){Require $result $pattern "$($test.mode) evidence $pattern"}
  $result|Set-Content -LiteralPath (Join-Path $reportRoot "$($test.name)-result.txt") -Encoding utf8
  Write-Host "PASS $($test.mode)"
}
Write-Host "QNN HTP Transformer backward tests passed. Reports: $reportRoot"
