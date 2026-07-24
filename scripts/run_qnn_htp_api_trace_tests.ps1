param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [ValidateRange(10,20)][int]$RunsPerCondition=10,
  [string]$ReportRoot,
  [switch]$SkipBuild
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if(!$ReportRoot){$ReportRoot=Join-Path $root 'build\reports\qnn-htp-api-trace-ab'}
[IO.Directory]::CreateDirectory((Join-Path $ReportRoot 'runs'))|Out-Null
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$adb=Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe'
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$package='com.yuubinnkyoku.phonelm';$activity="$package/.MainActivity"
$online=@();foreach($line in (& $adb devices)){if($line-match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count-ne1){throw "Expected exactly one online ADB device; found $($online.Count)"};$device=$online[0]
function Adb([string[]]$a){$o=& $adb -s $device @a 2>&1;if($LASTEXITCODE-ne0){throw "ADB failed (endpoint redacted): $($a-join ' ')`n$o"};$o}
function Fields([string]$x){$h=[ordered]@{};foreach($l in($x-split"`r?`n")){if($l-match'^([A-Za-z0-9_]+)=(.*)$'){$h[$Matches[1]]=$Matches[2]}};$h}
function DeviceState{$b=(Adb @('shell','dumpsys','battery'))-join"`n";$t=(Adb @('shell','dumpsys','thermalservice'))-join"`n";[pscustomobject]@{temperature=if($b-match'(?m)^\s*temperature:\s*(\d+)'){[double]$Matches[1]/10}else{$null};thermal=if($t-match'mStatus=(\d+)'){$Matches[1]}else{'unknown'}}}
function Guard([string]$phase){$s=DeviceState;Write-Host "$phase temperature=$($s.temperature)C thermal_status=$($s.thermal)";if($null-ne$s.temperature-and$s.temperature-ge45){throw "Thermal guard: $($s.temperature)C"};$s}
function RunMode([string]$mode,[string]$name,[int]$b,[int]$i,[int]$h,[int]$o,[int]$steps,[bool]$expectSuccess){
  $before=Guard $name
  Adb @('shell','am','force-stop',$package)|Out-Null
  Adb @('shell','run-as',$package,'rm','-f','files/device-test-result.txt')|Out-Null
  Adb @('shell','am','start','-W','-n',$activity,'--es','phonelm.mode',$mode,'--ei','phonelm.batch_size',"$b",'--ei','phonelm.dimension',"$i",'--ei','phonelm.hidden_dimension',"$h",'--ei','phonelm.output_dimension',"$o",'--ei','phonelm.steps',"$steps",'--ei','phonelm.sample_count','512','--ei','phonelm.epochs','0','--ez','phonelm.benchmark_mode','true','--es','phonelm.learning_rate','0.5','--es','phonelm.seed','20260710')|Out-Null
  $x='';for($n=0;$n-lt2400;$n++){Start-Sleep -Milliseconds 250;$x=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join"`n";if($x-match'(?m)^status=(SUCCESS|FAILED)$'){break}}
  [IO.File]::WriteAllText((Join-Path $ReportRoot "runs\$name.txt"),$x,[Text.UTF8Encoding]::new($false))
  $after=DeviceState;$f=Fields $x;$expected=if($expectSuccess){'SUCCESS'}else{'FAILED'}
  if($f.status-ne$expected){throw "$name status=$($f.status), expected $expected, error=$($f.error)"}
  if($f.api_trace_fallback_attempted-ne'false'-or$f.api_trace_fallback_succeeded-ne'false'){throw "$name fallback detected"}
  if([int]$f.qnn_callback_saved_message_count-gt64-or[int]$f.qnn_callback_saved_bytes-gt32768){throw "$name callback bound exceeded"}
  [pscustomobject]@{raw=$x;fields=$f;before=$before;after=$after}
}
function Percentile([double[]]$values,[double]$p){$v=@($values|Sort-Object);if(!$v.Count){return $null};$idx=[math]::Ceiling($p*$v.Count)-1;$v[[math]::Max(0,$idx)]}
Push-Location $root
try{
  if(!$SkipBuild){& .\gradlew.bat :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon;if($LASTEXITCODE-ne0){throw'build failed'};& "$PSScriptRoot\audit_qnn_apk.ps1" -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $ReportRoot 'apk-audit.txt')}
  Adb @('install','-r',$apk)|Out-Null
  $rows=@();$globalOrder=0
  $shapes=@(@(8,128,128,64),@(8,256,256,128),@(32,256,256,128))
  foreach($shape in $shapes){
    $counts=@{A=0;B=0};$cycle=0
    while($counts.A-lt$RunsPerCondition-or$counts.B-lt$RunsPerCondition){
      $order=if($cycle%2-eq0){@('A','B','B','A')}else{@('B','A','A','B')}
      foreach($condition in $order){if($counts[$condition]-ge$RunsPerCondition){continue};$counts[$condition]++;$globalOrder++
        $mode=if($condition-eq'A'){'QNN_HTP_MLP_FULL_STEP_BENCHMARK'}else{'QNN_HTP_MLP_FULL_STEP_BENCHMARK_CAPTURE'}
        $name="ab-b$($shape[0])-i$($shape[1])-h$($shape[2])-o$($shape[3])-$condition-$($counts[$condition])"
        $r=RunMode $mode $name $shape[0] $shape[1] $shape[2] $shape[3] 100 $true;$f=$r.fields
        if($condition-eq'A' -and $f.qnn_callback_capture_enabled-ne'false'){throw "$name capture should be disabled"}
        if($condition-eq'B' -and $f.qnn_callback_capture_enabled-ne'true'){throw "$name capture should be enabled"}
        $rows+=[pscustomobject]@{run_order=$globalOrder;shape="$($shape[0])/$($shape[1])/$($shape[2])/$($shape[3])";condition=$condition;callback_capture=$f.qnn_callback_capture_enabled;qnn_log_level=$f.qnn_callback_log_level;temperature_before_c=$r.before.temperature;temperature_after_c=$r.after.temperature;thermal_before=$r.before.thermal;thermal_after=$r.after.thermal;batch_preparation_median_us=$f.batch_preparation_median_us;input_bind_median_us=$f.input_bind_median_us;output_bind_median_us=$f.output_access_bind_median_us;weight_handoff_median_us=$f.weight_buffer_handoff_median_us;graph_execute_median_us=$f.training_graph_execute_median_us;steady_graph_execute_median_us=$f.steady_execute_median_us;full_step_median_us=$f.full_step_median_us;full_step_p90_us=$f.full_step_p90_us;full_step_p95_us=$f.full_step_p95_us;callback_saved=$f.qnn_callback_saved_message_count;callback_dropped=$f.qnn_callback_dropped_message_count;status=$f.status}
      };$cycle++
    }
  }
  $rows|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'runs.csv')
  $summary=@();foreach($g in($rows|Group-Object shape,condition)){$x=@($g.Group);$summary+=[pscustomobject]@{shape=$x[0].shape;condition=$x[0].condition;runs=$x.Count;callback_capture=$x[0].callback_capture;qnn_log_level=$x[0].qnn_log_level;graph_execute_median_of_medians_us=Percentile ([double[]]$x.graph_execute_median_us) .5;steady_execute_median_of_medians_us=Percentile ([double[]]$x.steady_graph_execute_median_us) .5;full_step_median_of_medians_us=Percentile ([double[]]$x.full_step_median_us) .5;full_step_p90_across_runs_us=Percentile ([double[]]$x.full_step_median_us) .9;full_step_p95_across_runs_us=Percentile ([double[]]$x.full_step_median_us) .95;temperature_min_c=($x.temperature_before_c+$x.temperature_after_c|Measure-Object -Minimum).Minimum;temperature_max_c=($x.temperature_before_c+$x.temperature_after_c|Measure-Object -Maximum).Maximum;thermal_max=($x.thermal_before+$x.thermal_after|Measure-Object -Maximum).Maximum}}
  $summary|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'summary.csv')
  $fx=RunMode QNN_HTP_MLP_FULL_STEP_FAIL_EXECUTE fail-execute-37 2 4 5 3 100 $false;$f=$fx.fields
  foreach($entry in([ordered]@{api_trace_graph_execute_attempt_count='38';api_trace_graph_execute_success_count='37';api_trace_graph_execute_failure_count='1';api_trace_graph_execute_first_failure_call='37';api_trace_failure_injection_enabled='true';api_trace_failure_injection_point='graphExecute';api_trace_failure_injection_call='37';api_trace_last_qnn_result='0';api_trace_effective_result='-9001'}).GetEnumerator()){if($f[$entry.Key]-ne$entry.Value){throw "execute failure $($entry.Key)=$($f[$entry.Key])"}}
  $ff=RunMode QNN_HTP_MLP_FULL_STEP_FAIL_FINALIZE fail-finalize 2 4 5 3 1 $false;$f=$ff.fields
  foreach($entry in([ordered]@{api_trace_full_step_graph_finalize_result='0';api_trace_failure_injection_enabled='true';api_trace_failure_injection_point='graphFinalize';api_trace_last_qnn_result='0';api_trace_effective_result='-9002'}).GetEnumerator()){if($f[$entry.Key]-ne$entry.Value){throw "finalize failure $($entry.Key)=$($f[$entry.Key])"}}
  $recovery=RunMode QNN_HTP_MLP_FULL_STEP_BENCHMARK failure-recovery 2 4 5 3 10 $true
  if($recovery.fields.api_trace_graph_execute_attempt_count-ne'10'-or$recovery.fields.api_trace_failure_injection_enabled-ne'false'){throw'failure state polluted recovery run'}
  $fixed=@();foreach($steps in @(640,1280)){$r=RunMode QNN_HTP_MLP_FULL_STEP_BENCHMARK "fixed-size-$steps" 8 128 128 64 $steps $true;$m=[regex]::Match($r.raw,'(?ms)^api_trace_version=1\r?\n.*?^api_trace_fallback_succeeded=(?:true|false)\r?$');if(!$m.Success){throw"fixed-size-$steps trace missing"};$fixed+=[pscustomobject]@{steps=$steps;trace_lines=($m.Value-split"`r?`n").Count;trace_bytes=[Text.Encoding]::UTF8.GetByteCount($m.Value)}}
  if($fixed[0].trace_lines-ne$fixed[1].trace_lines-or[math]::Abs($fixed[0].trace_bytes-$fixed[1].trace_bytes)-gt100-or$fixed[0].trace_bytes-ge8192-or$fixed[1].trace_bytes-ge8192){throw'fixed-size trace assertion failed'}
  $fixed|Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $ReportRoot 'fixed-size.csv')
  [IO.File]::WriteAllText((Join-Path $ReportRoot 'status.txt'),"status=SUCCESS`nruns_per_condition=$RunsPerCondition`nfailure_execute=SUCCESS`nfailure_finalize=SUCCESS`nfailure_recovery=SUCCESS`nfixed_size=SUCCESS`n",[Text.UTF8Encoding]::new($false))
  Write-Host "report_root=$ReportRoot`nstatus=SUCCESS"
}finally{Pop-Location}