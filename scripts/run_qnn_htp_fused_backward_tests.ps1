param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [ValidateRange(1,20)][int]$Repetitions=3,
  [long[]]$Seeds=@(20260710,20260711,20260712,20260713,20260714),
  [string]$ReportRoot,
  [switch]$SkipBuild,
  [switch]$SkipRegression,
  [switch]$RunPerformance
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(!$ReportRoot){$ReportRoot=Join-Path $root ('build\reports\qnn-htp-fused-backward-'+(Get-Date -Format yyyyMMdd-HHmmss))}
[IO.Directory]::CreateDirectory((Join-Path $ReportRoot 'runs'))|Out-Null
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk';$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$adb=Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe';$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$package='com.yuubinnkyoku.phonelm';$activity="$package/.MainActivity"
$online=@();foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne 1){throw "Expected exactly one online ADB device; found $($online.Count)"};$device=$online[0]
function Adb([string[]]$a){$o=& $adb -s $device @a 2>&1;if($LASTEXITCODE-ne 0){throw "ADB failed (endpoint redacted): $($a-join ' ')`n$o"};$o}
function Fields([string]$x){$h=[ordered]@{};foreach($l in ($x-split "`r?`n")){if($l-match '^([A-Za-z0-9_]+)=(.*)$'){$h[$Matches[1]]=$Matches[2]}};$h}
function DeviceState{$b=(Adb @('shell','dumpsys','battery'))-join "`n";$t=(Adb @('shell','dumpsys','thermalservice'))-join "`n";[pscustomobject]@{temperature=if($b-match '(?m)^\s*temperature:\s*(\d+)'){[double]$Matches[1]/10}else{$null};level=if($b-match '(?m)^\s*level:\s*(\d+)'){$Matches[1]}else{'unknown'};charging=if($b-match '(?m)^\s*status:\s*(\d+)'){$Matches[1]}else{'unknown'};thermal=if($t-match 'mStatus=(\d+)'){$Matches[1]}else{'unknown'}}}
function Guard([string]$phase){$v=DeviceState;Write-Host "$phase temperature=$($v.temperature)C thermal_status=$($v.thermal)";if($null-ne $v.temperature-and$v.temperature-ge45){throw "Thermal guard: $($v.temperature)C"};$v}
function RunMode([string]$mode,[string]$name,[int]$b,[int]$i,[int]$h,[int]$o,[int]$steps,[int]$samples,[int]$epochs,[double]$lr,[long]$seed,[bool]$benchmark){
  $path=Join-Path $ReportRoot "runs\$name.txt";$x=''
  if(Test-Path $path){$x=[IO.File]::ReadAllText($path);if($x-notmatch '(?m)^status=SUCCESS$'){$x=''}}
  $before=Guard $name
  if(!$x){
    Adb @('shell','am','force-stop',$package)|Out-Null;Adb @('shell','run-as',$package,'rm','-f','files/device-test-result.txt')|Out-Null
    $rate=$lr.ToString('R',[Globalization.CultureInfo]::InvariantCulture);$bm=if($benchmark){'true'}else{'false'}
    Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$mode,'--ei','phonelm.batch_size',"$b",'--ei','phonelm.dimension',"$i",'--ei','phonelm.hidden_dimension',"$h",'--ei','phonelm.output_dimension',"$o",'--ei','phonelm.steps',"$steps",'--ei','phonelm.sample_count',"$samples",'--ei','phonelm.epochs',"$epochs",'--ez','phonelm.benchmark_mode',$bm,'--es','phonelm.learning_rate',$rate,'--es','phonelm.seed',"$seed")|Out-Null
    for($n=0;$n-lt 1200;$n++){Start-Sleep -Milliseconds 500;$x=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n";if($x-match '(?m)^status=(SUCCESS|FAILED)$'){break}}
    [IO.File]::WriteAllText($path,$x,[Text.UTF8Encoding]::new($false))
  }
  $after=DeviceState;$f=Fields $x
  if($f.status-ne'SUCCESS'){throw "$name status=$($f.status) error=$($f.error)"};if($f.cpu_fallback-ne'false'){throw "$name CPU fallback"};if($f.nan_inf-ne'false'){throw "$name NaN/Inf"}
  [pscustomobject]@{fields=$f;before=$before;after=$after}
}
Push-Location $root
try{
  Write-Host 'Estimated duration: correctness + 5 seeds 5-10 minutes; performance and regressions add 15-30 minutes.'
  if(!$SkipBuild){.\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon;if($LASTEXITCODE-ne 0){throw 'build failed'};& "$PSScriptRoot\audit_qnn_apk.ps1" -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $ReportRoot 'apk-audit.txt')}
  Adb @('install','-r',$apk)|Out-Null
  $relu=RunMode QNN_HTP_RELU_BACKWARD_CHECK relu-backward-correctness 2 4 5 3 1 2 0 .1 $Seeds[0] $false
  [pscustomobject]$relu.fields|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'relu-backward-correctness.csv')
  $small=RunMode QNN_HTP_MLP_FUSED_BACKWARD fused-small-correctness 2 4 5 3 100 20 10 .1 $Seeds[0] $false
  [pscustomobject]$small.fields|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'fused-correctness.csv')
  $seedRows=@();foreach($seed in $Seeds){foreach($mode in @('QNN_CPU_MLP_TRAINING','QNN_HTP_MLP_HTP_LINEAR_BACKWARD','QNN_HTP_MLP_FUSED_BACKWARD')){$r=RunMode $mode "seed-$seed-$mode" 8 128 128 64 640 512 10 .5 $seed $false;$f=$r.fields;$seedRows+=[pscustomobject]@{seed=$seed;mode=$mode;initial_loss=$f.initial_loss;final_loss=$f.final_loss;prediction_mse=$f.prediction_mse;prediction_max_abs_error=$f.prediction_max_abs_error;dw2_max_abs_error=$f.dw2_max_abs_error;dh_max_abs_error=$f.dh_max_abs_error;dz1_max_abs_error=$f.dz1_max_abs_error;dw1_max_abs_error=$f.dw1_max_abs_error;forward_backend=$f.forward_backend;dw2_backend=$f.dw2_backend;dh_backend=$f.dh_backend;relu_backward_backend=$f.relu_backward_backend;dw1_backend=$f.dw1_backend;optimizer_backend=$f.optimizer_backend;cpu_fallback=$f.cpu_fallback;nan_inf=$f.nan_inf;status=$f.status}}};$seedRows|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'seeds.csv')
  $perf=@();if($RunPerformance){$shapes=@(@(2,4,5,3),@(8,128,128,64),@(8,256,256,128),@(32,256,256,128));for($n=1;$n-le$Repetitions;$n++){foreach($sh in $shapes){$order=if($n%2){@('QNN_HTP_MLP_BENCHMARK','QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK')}else{@('QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK','QNN_HTP_MLP_BENCHMARK')};foreach($mode in $order){$r=RunMode $mode "perf-b$($sh[0])-i$($sh[1])-r$n-$mode" $sh[0] $sh[1] $sh[2] $sh[3] 100 512 0 .5 $Seeds[0] $true;$f=$r.fields;$perf+=[pscustomobject]@{mode=$mode;batch_size=$sh[0];input_dim=$sh[1];hidden_dim=$sh[2];output_dim=$sh[3];run_index=$n;graph_count=if($mode-like'*FUSED*'){2}else{4};backward_execute_count_per_step=$f.backward_execute_count_per_step;forward_median_us=$f.forward_median_us;loss_dp_median_us=$f.loss_dp_median_us;backward_median_us=$f.second_layer_backward_median_us;relu_backward_median_us=$f.relu_backward_median_us;optimizer_median_us=$f.optimizer_median_us;weight_update_median_us=$f.weight_update_median_us;full_step_median_us=$f.full_step_median_us;runtime_initialization_us=$f.runtime_initialization_us;total_time_us=([double]$f.runtime_initialization_us+100*[double]$f.full_step_median_us);temperature_before=$r.before.temperature;temperature_after=$r.after.temperature;thermal_status_before=$r.before.thermal;thermal_status_after=$r.after.thermal;cpu_fallback=$f.cpu_fallback;status=$f.status}}}};$perf|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'performance.csv')}
  $breakEven=@();foreach($g in ($perf|Group-Object batch_size,input_dim,hidden_dim,output_dim)){$split=[double](($g.Group|Where-Object mode -eq QNN_HTP_MLP_BENCHMARK|Measure-Object full_step_median_us -Average).Average);$fused=[double](($g.Group|Where-Object mode -eq QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK|Measure-Object full_step_median_us -Average).Average);$splitInit=[double](($g.Group|Where-Object mode -eq QNN_HTP_MLP_BENCHMARK|Measure-Object runtime_initialization_us -Average).Average);$fusedInit=[double](($g.Group|Where-Object mode -eq QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK|Measure-Object runtime_initialization_us -Average).Average);$save=$split-$fused;$initPenalty=$fusedInit-$splitInit;$breakEven+=[pscustomobject]@{shape=$g.Name;split_full_step_us=$split;fused_full_step_us=$fused;split_initialization_us=$splitInit;fused_initialization_us=$fusedInit;initialization_delta_us=$initPenalty;per_step_saving_us=$save;break_even_steps=if($save-gt0){[math]::Max(1,[math]::Ceiling([math]::Max(0,$initPenalty)/$save))}else{'none'}}};$breakEven|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'break-even.csv')
  if(!$SkipRegression){& "$PSScriptRoot\run_qnn_device_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId;& "$PSScriptRoot\run_qnn_training_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -BatchSize 8 -InputDim 128 -Steps 100 -LearningRate 5;& "$PSScriptRoot\run_qnn_htp_backward_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Repetitions 1 -Seeds 20260710 -SkipBuild -SkipRegression;& "$PSScriptRoot\run_qnn_htp_mlp_tests.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Repetitions 1 -Seeds 20260710 -SkipBuild -SkipRegression}
  $summary=[ordered]@{qairt_build_id=$ExpectedBuildId;relu_backward_method='ElementWiseGreater+ElementWiseSelect';zero_rule='GT_ZERO';fused_nodes=@('dW2','dH','Greater','Select','dW1');backward_execute_reduction='3 to 1 per step';cpu_operations=@('loss','dP','SGD','weight update');cpu_fallback=$false;status='SUCCESS'}
  [IO.File]::WriteAllText((Join-Path $ReportRoot 'summary.json'),($summary|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false));[IO.File]::WriteAllText((Join-Path $ReportRoot 'summary.md'),"# QNN HTP fused backward`n`nstatus: SUCCESS`n`nQAIRT Build ID: $ExpectedBuildId`n`nHTP: forward, dW2, dH, ReLU backward, dW1. CPU: loss, dP, SGD, weight updates.`n",[Text.UTF8Encoding]::new($false));Write-Host "report_root=$ReportRoot`nstatus=SUCCESS"
}finally{Pop-Location}
