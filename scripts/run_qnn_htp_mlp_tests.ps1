param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [ValidateRange(1,20)][int]$Repetitions=3,
 [long[]]$Seeds=@(20260710,20260711,20260712,20260713,20260714),
 [string]$ReportRoot,
 [switch]$SkipBuild,[switch]$SkipRegression,[switch]$RunPerformance
)
$ErrorActionPreference='Stop';$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
if(!$ReportRoot){$ReportRoot=Join-Path $root ('build\reports\qnn-htp-mlp-'+(Get-Date -Format yyyyMMdd-HHmmss))}
[IO.Directory]::CreateDirectory((Join-Path $ReportRoot 'runs'))|Out-Null
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk';$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$adb=Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe';$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk';$package='com.yuubinnkyoku.phonelm';$activity="$package/.MainActivity"
$online=@();foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}};if($online.Count-ne 1){throw "Expected exactly one online ADB device; found $($online.Count)"};$device=$online[0]
function Adb([string[]]$a){$o=& $adb -s $device @a 2>&1;if($LASTEXITCODE-ne 0){throw "ADB failed (endpoint redacted): $($a-join ' ')`n$o"};$o}
function Fields([string]$x){$h=[ordered]@{};foreach($l in ($x-split "`r?`n")){if($l-match '^([A-Za-z0-9_]+)=(.*)$'){$h[$Matches[1]]=$Matches[2]}};$h}
function Thermal{$b=(Adb @('shell','dumpsys','battery'))-join "`n";$t=(Adb @('shell','dumpsys','thermalservice'))-join "`n";$c=if($b-match '(?m)^\s*temperature:\s*(\d+)'){[double]$Matches[1]/10}else{$null};$s=if($t-match 'mStatus=(\d+)'){$Matches[1]}else{'unknown'};[pscustomobject]@{temperature=$c;status=$s}}
function Guard([string]$phase){$v=Thermal;Write-Host "$phase temperature=$($v.temperature)C thermal_status=$($v.status)";if($null-ne$v.temperature-and$v.temperature-ge45){throw "Thermal guard: $($v.temperature)C"}}
function RunMode([string]$mode,[string]$name,[int]$b,[int]$i,[int]$h,[int]$o,[int]$steps,[int]$samples,[int]$epochs,[double]$lr,[long]$seed,[bool]$benchmark){
 $path=Join-Path $ReportRoot "runs\$name.txt";$x='';if(Test-Path $path){$x=[IO.File]::ReadAllText($path);if($x-notmatch '(?m)^status=SUCCESS$'){$x=''}}
 if(!$x){Guard $name;Adb @('shell','am','force-stop',$package)|Out-Null;& $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null;$rate=$lr.ToString('R',[Globalization.CultureInfo]::InvariantCulture);$bm=if($benchmark){'true'}else{'false'};Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$mode,'--ei','phonelm.batch_size',"$b",'--ei','phonelm.dimension',"$i",'--ei','phonelm.hidden_dimension',"$h",'--ei','phonelm.output_dimension',"$o",'--ei','phonelm.steps',"$steps",'--ei','phonelm.sample_count',"$samples",'--ei','phonelm.epochs',"$epochs",'--ez','phonelm.benchmark_mode',$bm,'--es','phonelm.learning_rate',$rate,'--es','phonelm.seed',"$seed")|Out-Null
  for($n=0;$n -lt 1200;$n++){Start-Sleep -Milliseconds 500;$x=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n";if($x-match '(?m)^status=(SUCCESS|FAILED)$'){break}};[IO.File]::WriteAllText($path,$x,[Text.UTF8Encoding]::new($false))}
 $f=Fields $x;if($f.status-ne'SUCCESS'){throw "$name status=$($f.status)"};if($f.cpu_fallback-ne'false'){throw "$name CPU fallback"};if($f.nan_inf-ne'false'){throw "$name NaN/Inf"};return $f
}
Push-Location $root
try{
 Write-Host 'Estimated duration: correctness and 5 seeds 5-10 minutes; performance adds 10-20 minutes.'
 if(!$SkipBuild){.\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon;if($LASTEXITCODE -ne 0){throw 'build failed'};& "$PSScriptRoot\audit_qnn_apk.ps1" -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $ReportRoot 'apk-audit.txt')}
 Adb @('install','-r',$apk)|Out-Null
 $dx=RunMode QNN_HTP_DX_CHECK dx-correctness 2 4 5 3 1 2 0 .1 $Seeds[0] $false;[pscustomobject]$dx|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'dx-correctness.csv')
 $gc=RunMode QNN_MLP_GRADIENT_CHECK gradient-check 2 4 5 3 1 2 0 .1 $Seeds[0] $false;[pscustomobject]$gc|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'gradient-check.csv')
 foreach($m in @('QNN_CPU_MLP_TRAINING','QNN_HTP_MLP_CPU_BACKWARD','QNN_HTP_MLP_HTP_LINEAR_BACKWARD')){RunMode $m "small-$m" 2 4 5 3 100 20 10 .1 $Seeds[0] $false|Out-Null}
 $seedRows=@();foreach($seed in $Seeds){foreach($m in @('QNN_CPU_MLP_TRAINING','QNN_HTP_MLP_CPU_BACKWARD','QNN_HTP_MLP_HTP_LINEAR_BACKWARD')){$f=RunMode $m "seed-$seed-$m" 8 128 128 64 640 512 10 .5 $seed $false;$seedRows+=[pscustomobject]@{seed=$seed;mode=$m;initial_loss=$f.initial_loss;final_loss=$f.final_loss;prediction_mse=$f.prediction_mse;prediction_max_abs_error=$f.prediction_max_abs_error;forward_backend=$f.forward_backend;dw2_backend=$f.dw2_backend;dh_backend=$f.dh_backend;relu_backward_backend=$f.relu_backward_backend;dw1_backend=$f.dw1_backend;optimizer_backend=$f.optimizer_backend;cpu_fallback=$f.cpu_fallback;nan_inf=$f.nan_inf;status=$f.status}}};$seedRows|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'seeds.csv')
 if($RunPerformance){$perf=@();$shapes=@(@(2,4,5,3),@(8,128,128,64),@(8,256,256,128));for($r=1;$r -le $Repetitions;$r++){foreach($sh in $shapes){$order=if($r%2){@('QNN_CPU_MLP_TRAINING','QNN_HTP_MLP_BENCHMARK')}else{@('QNN_HTP_MLP_BENCHMARK','QNN_CPU_MLP_TRAINING')};foreach($m in $order){$f=RunMode $m "perf-b$($sh[0])-i$($sh[1])-r$r-$m" $sh[0] $sh[1] $sh[2] $sh[3] 100 512 0 .5 $Seeds[0] $true;$perf+=[pscustomobject]@{mode=$m;batch_size=$sh[0];input_dim=$sh[1];hidden_dim=$sh[2];output_dim=$sh[3];run_index=$r;forward_median_us=$f.forward_median_us;backward2_median_us=$f.second_layer_backward_median_us;relu_backward_median_us=$f.relu_backward_median_us;dw1_median_us=$f.first_layer_dw_median_us;optimizer_median_us=$f.optimizer_median_us;weight_update_median_us=$f.weight_update_median_us;full_step_median_us=$f.full_step_median_us;cpu_fallback=$f.cpu_fallback;status=$f.status}}}};$perf|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'performance.csv')}
 if(!$SkipRegression){& "$PSScriptRoot\run_qnn_device_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId;& "$PSScriptRoot\run_qnn_training_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -BatchSize 8 -InputDim 128 -Steps 100 -LearningRate 5;& "$PSScriptRoot\run_qnn_htp_backward_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Repetitions 1 -Seeds 20260710 -SkipBuild -SkipRegression}
 [ordered]@{qairt_build_id=$ExpectedBuildId;seeds=$Seeds;modes=@('CPU all','HTP forward + CPU backward','HTP forward + HTP linear backward');cpu_fallback=$false;status='SUCCESS'}|ConvertTo-Json -Depth 4|Set-Content -Encoding utf8 (Join-Path $ReportRoot 'summary.json')
 "# QNN HTP 2-layer MLP`n`nstatus: SUCCESS`n`nQAIRT Build ID: $ExpectedBuildId`n`nReLU backward and optimizer are CPU operations. HTP runs forward, dW2, dH, and dW1 in the linear-backward mode."|Set-Content -Encoding utf8 (Join-Path $ReportRoot 'summary.md')
 Write-Host "report_root=$ReportRoot`nstatus=SUCCESS"
}finally{Pop-Location}
