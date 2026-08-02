param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [ValidateRange(3,5)][int]$Repetitions=3,
 [long[]]$Seeds=@(20260710,20260711,20260712,20260713,20260714),
 [ValidateSet("Seeds","Stage1","Refine","Extended","All")][string]$Phase="Seeds",
 [int]$WarmupSteps=5,[int]$MeasuredSteps=30,[int]$TimeoutSeconds=900,
 [double]$LearningRate=5,[string]$ReportRoot="",
 [int[]]$Batches=@(),[int[]]$Dimensions=@(),
 [switch]$SkipBuild,[switch]$SkipRegression,[switch]$RerunSuccessful,[switch]$WhatIf
)
$ErrorActionPreference="Stop";$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "qairt_version.ps1")
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$adb=Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA "Android\Sdk";$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$package="com.yuubinnkyoku.phonelm";$activity="$package/.MainActivity";$apk=Join-Path $root "app\build\outputs\apk\debug\app-debug.apk"
if (!$ReportRoot){$ReportRoot=Join-Path $root ("build\reports\qnn-training-benchmark-"+(Get-Date -Format "yyyyMMdd-HHmmss"))}
$runs=Join-Path $ReportRoot "runs";[IO.Directory]::CreateDirectory($runs)|Out-Null
$devices=@(& $adb devices|Where-Object{$_ -match '^(\S+)\s+device$'}|ForEach-Object{$Matches[1]})
if ($devices.Count -ne 1){throw "Expected one online ADB device; found $($devices.Count)"};$device=$devices[0]
function A([string[]]$x,[switch]$Ignore){$o=& $adb -s $device @x 2>&1;if (!$Ignore -and $LASTEXITCODE -ne 0){throw "ADB failed (endpoint redacted): $($x -join ' ') $o"};return $o}
function Fields([string]$s){$h=[ordered]@{};foreach ($l in($s-split"\r?\n")){if ($l -match '^([A-Za-z0-9_]+)=(.*)$'){$h[$Matches[1]]=$Matches[2]}};return $h}
function N($h,[string]$k){$v=if ($h -is [Collections.IDictionary]){if (!$h.Contains($k)){throw "Missing $k"};$h[$k]}else{$h.$k};if ($null -eq $v){throw "Missing $k"};return [double]::Parse([string]$v,[Globalization.CultureInfo]::InvariantCulture)}
function State{
 $b=(A @("shell","dumpsys","battery") -Ignore)-join" ";$t=(A @("shell","dumpsys","thermalservice") -Ignore)-join" ";$u=(A @("shell","cat","/proc/uptime") -Ignore)-join" "
 $temp=$null;if ($b -match 'temperature:\s*(\d+)'){$temp=[double]$Matches[1]/10};$level=$null;if ($b -match 'level:\s*(\d+)'){$level=[int]$Matches[1]};$charge=$null;if ($b -match 'status:\s*(\d+)'){$charge=[int]$Matches[1]}
 $thermal=$null;foreach ($p in @('Thermal Status:\s*(\d+)','mStatusOverride:\s*(\d+)','mStatus:\s*(\d+)')){if ($t -match $p){$thermal=[int]$Matches[1];break}};$elapsed=$null;if ($u -match '^\s*([0-9.]+)'){$elapsed=[double]$Matches[1]}
 return [ordered]@{temperature=$temp;thermal=$thermal;level=$level;charging=$charge;elapsed=$elapsed}
}
function Safe{$limit=(Get-Date).AddMinutes(15);while ($true){$s=State;if ($null -eq $s.thermal -or $s.thermal -lt 3){return $s};if ((Get-Date) -gt $limit){throw "Thermal status stayed SEVERE"};Write-Host "Cooling: thermal=$($s.thermal)";Start-Sleep 30}}
function Block([string]$s,[string]$a,[string]$b,[string]$p){$m=[regex]::Match($s,"(?ms)^$a\r?\n(.*?)^$b\r?$");if (!$m.Success){throw "Missing $a"};$m.Groups[1].Value.TrimEnd()|Set-Content $p -Encoding utf8}
function Check($h,[string]$mode,[int]$steps){
 foreach ($k in @("status","cpu_fallback","nan_detected","inf_detected","htp_execute_failures","runtime_weight_update_failures")){if (!$h.Contains($k)){throw "Missing $k"}}
 if ($h.status -ne "SUCCESS" -or $h.cpu_fallback -ne "false" -or $h.nan_detected -ne "false" -or $h.inf_detected -ne "false" -or $h.htp_execute_failures -ne "0" -or $h.runtime_weight_update_failures -ne "0"){throw "$mode failed: $($h.error)"}
 if ((N $h "final_loss") -ge (N $h "initial_loss") -or (N $h "final_weight_error") -ge (N $h "initial_weight_error")){throw "$mode did not converge"}
 if ($mode -like "QNN_HTP*"){
  $e=@{compile_time_sdk_build_id=$ExpectedBuildId;backend_build_id_match="true";provider_core_api_version="2.37.0";provider_backend_api_version="5.48.0";graph_create_count="1";graph_finalize_count="1";runtime_weight_update_count="$steps";backward_on_htp="false";optimizer_on_htp="false"}
  foreach ($x in $e.GetEnumerator()){if ($h[$x.Key] -ne $x.Value){throw "Expected $($x.Key)=$($x.Value), got $($h[$x.Key])"}};if ((N $h "max_abs_error") -gt 1e-3){throw "HTP reference mismatch"}
 }
}
function One([string]$backend,$c,[int]$index){
 $bench=$c.kind -eq "bench";$mode=if ($bench){"QNN_"+$backend+"_TRAINING_BENCHMARK"}else{"QNN_"+$backend+"_MULTIBATCH_TRAINING"}
 $dir=Join-Path $runs ("b"+$c.b+"-d"+$c.d+"-s"+$c.seed+"-"+$c.tag);[IO.Directory]::CreateDirectory($dir)|Out-Null;$base=$backend.ToLower()+"-run-"+$index;$json=Join-Path $dir ($base+".json")
 if ((Test-Path $json) -and !$RerunSuccessful){$old=Get-Content $json -Raw|ConvertFrom-Json;if ($old.status -eq "SUCCESS"){return $old}};if ($WhatIf){Write-Host "WHATIF $mode b=$($c.b) d=$($c.d)";return}
 A @("shell","am","force-stop",$package)|Out-Null;Start-Sleep 3;$before=Safe;A @("logcat","-c")|Out-Null;A @("shell","run-as",$package,"rm","-f","files/device-test-result.txt")|Out-Null
 $rate=$LearningRate.ToString("R",[Globalization.CultureInfo]::InvariantCulture);$interval=if ($bench){[Math]::Max(1,[int]($c.measured/5))}else{1};$flag=if ($bench){"true"}else{"false"}
 $x=@("shell","am","start","-W","-n",$activity,"--es","phonelm.mode",$mode,"--ei","phonelm.batch_size","$($c.b)","--ei","phonelm.dimension","$($c.d)","--ei","phonelm.steps","$($c.steps)","--ei","phonelm.warmup_steps","$($c.warm)","--ei","phonelm.sample_count","$($c.samples)","--ei","phonelm.epochs","$($c.epochs)","--ei","phonelm.measured_steps","$($c.measured)","--ei","phonelm.correctness_interval","$interval","--ez","phonelm.benchmark_mode",$flag,"--es","phonelm.learning_rate",$rate,"--es","phonelm.seed","$($c.seed)");A $x|Out-Null
 $w=[Diagnostics.Stopwatch]::StartNew();$result="";while ($w.Elapsed.TotalSeconds -lt $TimeoutSeconds){Start-Sleep -Milliseconds 500;$result=(A @("shell","run-as",$package,"cat","files/device-test-result.txt") -Ignore)-join[Environment]::NewLine;if ($result -match "(?m)^execution_mode=$mode\s*$" -and $result -match '(?m)^status=(SUCCESS|FAILED)\s*$'){break}}
 if ($result -notmatch '(?m)^status=(SUCCESS|FAILED)\s*$'){A @("shell","am","force-stop",$package)|Out-Null;throw "$mode timeout"}
 $after=State;$log=(A @("logcat","-d","-b","all","-v","threadtime"))-join[Environment]::NewLine;if ($log -match 'qnn_open.*(0x80000600|error)|getHandle.*(0xf|error)|transport error.*1002'){throw "FastRPC error"}
 $h=Fields $result;$steps=if ($c.epochs -gt 0){$c.epochs*[int][Math]::Ceiling($c.samples/[double]$c.b)}else{$c.warm+$c.measured};Check $h $mode $steps
 $result|Set-Content (Join-Path $dir ($base+".txt")) -Encoding utf8;$log|Set-Content (Join-Path $dir ($base+"-logcat.txt")) -Encoding utf8;Block $result "timings_csv_begin" "timings_csv_end" (Join-Path $dir ($base+"-timings.csv"));Block $result "epoch_csv_begin" "epoch_csv_end" (Join-Path $dir ($base+"-loss.csv"))
 $r=[ordered]@{};foreach ($z in $h.GetEnumerator()){$r[$z.Key]=$z.Value};$r.run_index=$index;$r.temperature_before=$before.temperature;$r.temperature_after=$after.temperature;$r.thermal_status_before=$before.thermal;$r.thermal_status_after=$after.thermal;$r.battery_level_before=$before.level;$r.battery_level_after=$after.level;$r.charging_status_before=$before.charging;$r.charging_status_after=$after.charging;$r.elapsed_realtime_before_s=$before.elapsed;$r.elapsed_realtime_after_s=$after.elapsed;$r|ConvertTo-Json -Depth 4|Set-Content $json -Encoding utf8;return [pscustomobject]$r
}
$config=[Collections.Generic.List[object]]::new()
if ($Phase -in @("Seeds","All")){foreach ($s in $Seeds){$config.Add([pscustomobject]@{kind="seed";b=8;d=128;samples=512;seed=$s;epochs=10;warm=0;measured=0;steps=640;reps=1;tag="seed"})}}
if ($Phase -in @("Stage1","All")){foreach ($b in @(1,8,32)){foreach ($d in @(64,128,256,512)){$config.Add([pscustomobject]@{kind="bench";b=$b;d=$d;samples=[Math]::Max(512,$d);seed=20260710;epochs=0;warm=$WarmupSteps;measured=$MeasuredSteps;steps=$MeasuredSteps;reps=$Repetitions;tag="stage1"})}}}
if ($Phase -in @("Refine","All")){foreach ($b in @(1,8,16,32,64)){foreach ($d in @(192,256,320,384,448,512)){$config.Add([pscustomobject]@{kind="bench";b=$b;d=$d;samples=512;seed=20260710;epochs=0;warm=$WarmupSteps;measured=$MeasuredSteps;steps=$MeasuredSteps;reps=$Repetitions;tag="refine"})}}}
if ($Phase -in @("Extended","All")){foreach ($b in @(8,32,64)){foreach ($d in @(768,1024)){$m=[Math]::Min(10,$MeasuredSteps);$config.Add([pscustomobject]@{kind="bench";b=$b;d=$d;samples=1024;seed=20260710;epochs=0;warm=$WarmupSteps;measured=$m;steps=$m;reps=$Repetitions;tag="extended"})}}}
if ($Batches.Count -gt 0){$config=@($config|Where-Object{$_.b -in $Batches})}
if ($Dimensions.Count -gt 0){$config=@($config|Where-Object{$_.d -in $Dimensions})}
$count=($config|Measure-Object reps -Sum).Sum*2;Write-Host "Planned runs=$count; minimum wait=$([Math]::Ceiling($count*3/60)) minutes"
Push-Location $root
try{
 if (!$SkipBuild -and !$WhatIf){& .\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon;if ($LASTEXITCODE -ne 0){throw "Build failed"};& (Join-Path $PSScriptRoot "audit_qnn_apk.ps1") -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $ReportRoot "qnn-apk-audit.txt")}
 if (!$WhatIf){A @("install","-r",$apk)|Out-Null};if (!$SkipRegression -and !$WhatIf){& (Join-Path $PSScriptRoot "run_qnn_training_tests.ps1") -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -BatchSize 8 -InputDim 128 -Steps 100 -LearningRate 5 -SkipBuild -SkipRegression;if ($LASTEXITCODE -ne 0){throw "Regression failed"}}
 $all=[Collections.Generic.List[object]]::new();foreach ($c in $config){for ($i=1;$i -le $c.reps;$i++){$order=if ($i%2 -eq 1){@("CPU","HTP")}else{@("HTP","CPU")};foreach ($b in $order){$r=One $b $c $i;if ($null -ne $r){$all.Add($r)}}}}
 if (!$WhatIf){
  $sweep=foreach ($r in $all){[pscustomobject]@{seed=$r.seed;backend=$r.backend;batch_size=$r.batch_size;input_dim=$r.input_dim;output_dim=$r.output_dim;steps=$r.steps;run_index=$r.run_index;initial_loss=$r.initial_loss;final_loss=$r.final_loss;initial_weight_error=$r.initial_weight_error;final_weight_error=$r.final_weight_error;initialization_us=$r.initialization_us;first_execute_us=$r.first_execute_time_us;steady_execute_median_us=$r.steady_state_execute_median_us;output_copy_median_us=$r.output_copy_median_us;gradient_median_us=$r.cpu_gradient_median_us;weight_copy_median_us=$r.weight_buffer_copy_median_us;runtime_update_median_us=$r.runtime_weight_update_median_us;full_step_median_us=$r.full_step_median_us;total_time_us=$r.total_training_time_us;temperature_before=$r.temperature_before;temperature_after=$r.temperature_after;thermal_status_before=$r.thermal_status_before;thermal_status_after=$r.thermal_status_after;cpu_fallback=$r.cpu_fallback;status=$r.status}};$sweep|Export-Csv (Join-Path $ReportRoot "sweep.csv") -NoTypeInformation -Encoding utf8
  $seedComparison=foreach ($g in @($all|Where-Object{$_.measurement_mode -eq "correctness"}|Group-Object seed)){$cpu=@($g.Group|Where-Object backend -eq "CPU")[0];$htp=@($g.Group|Where-Object backend -eq "HTP")[0];if ($null -eq $cpu -or $null -eq $htp){continue};[pscustomobject]@{seed=$g.Name;cpu_initial_loss=$cpu.initial_loss;htp_initial_loss=$htp.initial_loss;cpu_final_loss=$cpu.final_loss;htp_final_loss=$htp.final_loss;cpu_htp_final_loss_difference=[Math]::Abs((N $cpu "final_loss")-(N $htp "final_loss"));cpu_final_weight_error=$cpu.final_weight_error;htp_final_weight_error=$htp.final_weight_error;cpu_htp_final_weight_max_abs_difference=$htp.final_weight_difference;cpu_loss_reduction_ratio=$cpu.loss_reduction_ratio;htp_loss_reduction_ratio=$htp.loss_reduction_ratio;cpu_weight_error_reduction_ratio=$cpu.weight_error_reduction_ratio;htp_weight_error_reduction_ratio=$htp.weight_error_reduction_ratio}}
  $seedComparison|Export-Csv (Join-Path $ReportRoot "seed-comparison.csv") -NoTypeInformation -Encoding utf8
  $seedSummary=foreach ($backend in @("CPU","HTP")){$rows=@($all|Where-Object{$_.measurement_mode -eq "correctness" -and $_.backend -eq $backend});if (!$rows){continue};$loss=@($rows|ForEach-Object{N $_ "final_loss"});$weight=@($rows|ForEach-Object{N $_ "final_weight_error"});$lm=($loss|Measure-Object -Average).Average;$wm=($weight|Measure-Object -Average).Average;[pscustomobject]@{backend=$backend;seed_count=$rows.Count;final_loss_mean=$lm;final_loss_stddev=[Math]::Sqrt((($loss|ForEach-Object{($_-$lm)*($_-$lm)}|Measure-Object -Sum).Sum)/$loss.Count);final_weight_error_mean=$wm;final_weight_error_stddev=[Math]::Sqrt((($weight|ForEach-Object{($_-$wm)*($_-$wm)}|Measure-Object -Sum).Sum)/$weight.Count)}}
  $seedSummary|Export-Csv (Join-Path $ReportRoot "seed-summary.csv") -NoTypeInformation -Encoding utf8
  $be=foreach ($g in($sweep|Group-Object batch_size,input_dim)){$cpu=@($g.Group|Where-Object backend -eq "CPU");$htp=@($g.Group|Where-Object backend -eq "HTP");if (!$cpu -or !$htp){continue};$cs=($cpu|Measure-Object full_step_median_us -Average).Average;$hs=($htp|Measure-Object full_step_median_us -Average).Average;$cf=($cpu|Measure-Object steady_execute_median_us -Average).Average;$hf=($htp|Measure-Object steady_execute_median_us -Average).Average;$init=($htp|Measure-Object initialization_us -Average).Average;[pscustomobject]@{batch_size=$cpu[0].batch_size;dimension=$cpu[0].input_dim;cpu_forward_median_us=$cf;htp_forward_median_us=$hf;cpu_step_median_us=$cs;htp_step_median_us=$hs;htp_initialization_us=$init;forward_only_winner=if ($hf -lt $cf){"HTP"}else{"CPU"};full_step_winner=if ($hs -lt $cs){"HTP"}else{"CPU"};break_even_steps=if ($cs -gt $hs){[Math]::Ceiling($init/($cs-$hs))}else{$null}}};$be|Export-Csv (Join-Path $ReportRoot "break-even.csv") -NoTypeInformation -Encoding utf8
  [ordered]@{qairt_build_id=$ExpectedBuildId;phase=$Phase;run_count=$all.Count;cpu_fallback=$false;htp_backward=$false;htp_optimizer=$false;status="SUCCESS"}|ConvertTo-Json|Set-Content (Join-Path $ReportRoot "summary.json") -Encoding utf8;@("# PhoneLM QNN training benchmark","","QAIRT Build ID: $ExpectedBuildId","Phase: $Phase","Runs: $($all.Count)","HTP performs forward only; CPU performs loss, gradient, optimizer, and APP_WRITE buffer update.")|Set-Content (Join-Path $ReportRoot "summary.md") -Encoding utf8;Write-Host "Report: $ReportRoot"
 }
}catch{[ordered]@{status="FAILED";error=$_.Exception.Message}|ConvertTo-Json|Set-Content (Join-Path $ReportRoot "failure.json") -Encoding utf8;throw}finally{Pop-Location}
