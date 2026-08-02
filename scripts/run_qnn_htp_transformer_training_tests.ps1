# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$adb=Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk';$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$package='com.yuubinnkyoku.phonelm';$activity="$package/.MainActivity"
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot=Join-Path $root 'build\reports\qnn-tiny-transformer-training-tests';[IO.Directory]::CreateDirectory($reportRoot)|Out-Null
if(!$SkipBuild){& (Join-Path $root 'gradlew.bat') :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon;if($LASTEXITCODE-ne0){throw 'APK build failed'}}
$online=@();foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}};if($online.Count-ne1){throw "Expected one online ADB device; found $($online.Count)"};$device=$online[0]
function Adb([string[]]$Arguments){$out=& $adb -s $device @Arguments 2>&1;if($LASTEXITCODE-ne0){throw "ADB command failed (endpoint redacted): $($Arguments-join ' ')`n$out"};$out}
function Require([string]$Text,[string]$Pattern,[string]$Description){if($Text-notmatch $Pattern){throw "Missing $Description"}}
Adb @('install','-r',$apk)|Out-Null
$tests=@(
 @{mode='QNN_HTP_TINY_TRANSFORMER_TRAINING_STEP';name='one-step';required=@('(?m)^shape=B1_T4_D16_H1_F32$','(?m)^gradient_max_abs_error=','(?m)^next_weight_max_abs_error=','(?m)^major_weight_changed=true$','(?m)^execute_count_per_step=1$')},
 @{mode='QNN_HTP_TINY_TRANSFORMER_TRAINING_MULTI_STEP';name='multi-step';required=@('(?m)^steps=100$','(?m)^seeds=5$','(?m)^all_seeds_loss_decreased=true$','(?m)^deterministic_replay=true$','(?m)^graph_execute_count=505$')}
)
foreach($test in $tests){
 Adb @('shell','am','force-stop',$package)|Out-Null;& $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
 Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$test.mode)|Out-Null
 $result='';for($poll=0;$poll-lt1200;$poll++){Start-Sleep -Milliseconds 500;$result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n";if($result-match '(?m)^status=(SUCCESS|FAILED)$'){break}}
 Require $result '(?m)^status=SUCCESS$' "$($test.mode) success";Require $result '(?m)^cpu_fallback=false$' 'no fallback';Require $result '(?m)^nan_detected=false$' 'no NaN';Require $result '(?m)^inf_detected=false$' 'no Inf';Require $result '(?m)^api_trace_graph_execute_failure_count=0$' 'zero execute failures'
 foreach($pattern in $test.required){Require $result $pattern "$($test.mode) evidence $pattern"}
 $result|Set-Content -LiteralPath (Join-Path $reportRoot "$($test.name)-result.txt") -Encoding utf8
 Write-Host "PASS $($test.mode)"
}
Write-Host "QNN HTP tiny Transformer training tests passed. Reports: $reportRoot"
