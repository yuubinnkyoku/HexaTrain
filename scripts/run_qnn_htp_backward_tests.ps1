param(
    [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
    [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
    [ValidateRange(1,20)][int]$Repetitions=3,
    [long[]]$Seeds=@(20260710,20260711,20260712,20260713,20260714),
    [string]$ReportRoot,
    [switch]$SkipBuild,
    [switch]$SkipRegression,
    [switch]$RunPerformance,
    [switch]$RunBreakEven,
    [int[]]$BreakEvenBatches=@(8),
    [int[]]$BreakEvenDimensions=@(64,128,192,256,320,512)
)

$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(!$ReportRoot){$ReportRoot=Join-Path $root ('build\reports\qnn-htp-backward-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))}
[IO.Directory]::CreateDirectory($ReportRoot)|Out-Null
[IO.Directory]::CreateDirectory((Join-Path $ReportRoot 'runs'))|Out-Null
$adb=Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME=Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$package='com.yuubinnkyoku.phonelm'
$activity="$package/.MainActivity"
$apk=Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'

$online=@()
foreach($line in (& $adb devices)){if($line -match '^(\S+)\s+device$'){$online+=$Matches[1]}}
if($online.Count -ne 1){throw "Expected exactly one online ADB device; found $($online.Count)."}
$device=$online[0]

function Invoke-Adb([string[]]$Arguments){
    $output=& $adb -s $device @Arguments 2>&1
    if($LASTEXITCODE -ne 0){throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output"}
    return $output
}
function Read-Fields([string]$Text){
    $fields=[ordered]@{}
    foreach($line in ($Text -split "`r?`n")){if($line -match '^([A-Za-z0-9_]+)=(.*)$'){$fields[$Matches[1]]=$Matches[2]}}
    return $fields
}
function Number($Fields,[string]$Name,[double]$Default=0){
    if(!$Fields.Contains($Name)){return $Default}
    return [double]::Parse($Fields[$Name],[Globalization.CultureInfo]::InvariantCulture)
}
function Assert-Field($Fields,[string]$Name,[string]$Expected){
    if(!$Fields.Contains($Name)-or $Fields[$Name] -ne $Expected){throw "Expected $Name=$Expected; actual=$($Fields[$Name])"}
}
function Read-Thermal{
    $battery=(Invoke-Adb @('shell','dumpsys','battery'))-join "`n"
    $temperature=$null
    if($battery -match '(?m)^\s*temperature:\s*(\d+)'){$temperature=[double]$Matches[1]/10.0}
    $thermal=(Invoke-Adb @('shell','dumpsys','thermalservice'))-join "`n"
    $status='unknown'
    if($thermal -match '(?m)mStatus=(\d+)'){$status=$Matches[1]}
    [pscustomobject]@{temperature_c=$temperature;status=$status}
}
function Require-Cool([string]$Phase){
    $value=Read-Thermal
    Write-Host "$Phase temperature=$($value.temperature_c)C thermal_status=$($value.status)"
    if($null -ne $value.temperature_c -and $value.temperature_c -ge 45.0){throw "Device too warm for long phase: $($value.temperature_c)C"}
    return $value
}

$script:rows=[Collections.Generic.List[object]]::new()
$existingSweep=Join-Path $ReportRoot 'sweep.csv'
if(Test-Path -LiteralPath $existingSweep){foreach($row in (Import-Csv -LiteralPath $existingSweep)){$script:rows.Add($row)}}
function Run-Mode{
    param([string]$Mode,[string]$Name,[int]$Batch,[int]$Dimension,[int]$Steps,
          [int]$SampleCount,[int]$Epochs,[double]$LearningRate,[long]$Seed,
          [int]$RunIndex,[bool]$Benchmark)
    $phase=if($Name -like 'correctness-*'){'correctness'}elseif($Name -like 'seed-*'){'seed'}elseif($Name -like 'sweep-*'){'performance'}elseif($Name -like 'break-*'){'break-even'}else{'other'}
    $dir=Join-Path $ReportRoot ('runs\'+$Name)
    [IO.Directory]::CreateDirectory($dir)|Out-Null
    $resultPath=Join-Path $dir 'result.txt'
    $logPath=Join-Path $dir 'logcat.txt'
    $result=''
    if(Test-Path -LiteralPath $resultPath){
        $candidate=Get-Content -Raw -LiteralPath $resultPath
        if($candidate -match '(?m)^status=SUCCESS$'){$result=$candidate; Write-Host "resume=$Name"}
    }
    $before=Read-Thermal
    if(!$result){
        Invoke-Adb @('shell','am','force-stop',$package)|Out-Null
        Invoke-Adb @('logcat','-c')|Out-Null
        & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null|Out-Null
        $rate=$LearningRate.ToString('R',[Globalization.CultureInfo]::InvariantCulture)
        $benchmarkText=if($Benchmark){'true'}else{'false'}
        $warmupText=if($Benchmark){'10'}else{'0'}
        $measuredText=if($Benchmark){"$Steps"}else{'0'}
        $intervalText=if($Benchmark){"$([Math]::Max(1,[int]($Steps/5)))"}else{'1'}
        Invoke-Adb @('shell','am','start','-W','-n',$activity,
            '--es','phonelm.mode',$Mode,'--ei','phonelm.batch_size',"$Batch",
            '--ei','phonelm.dimension',"$Dimension",'--ei','phonelm.steps',"$Steps",
            '--ei','phonelm.sample_count',"$SampleCount",'--ei','phonelm.epochs',"$Epochs",
            '--ei','phonelm.warmup_steps',$warmupText,
            '--ei','phonelm.measured_steps',$measuredText,
            '--ei','phonelm.correctness_interval',$intervalText,
            '--ez','phonelm.benchmark_mode',$benchmarkText,'--es','phonelm.learning_rate',$rate,
            '--es','phonelm.seed',"$Seed")|Out-Null
        for($poll=0;$poll -lt 1200;$poll++){
            Start-Sleep -Milliseconds 500
            $result=(& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null)-join "`n"
            if($result -match "(?m)^execution_mode=$([regex]::Escape($Mode))$" -and $result -match '(?m)^status=(SUCCESS|FAILED)$'){break}
        }
        $log=(Invoke-Adb @('logcat','-d','-b','all','-v','threadtime'))-join "`n"
        [IO.File]::WriteAllText($resultPath,$result,[Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($logPath,$log,[Text.UTF8Encoding]::new($false))
    }
    $after=Read-Thermal
    [ordered]@{run_name=$Name;phase=$phase;temperature_before_c=$before.temperature_c;temperature_after_c=$after.temperature_c;thermal_status_before=$before.status;thermal_status_after=$after.status}|ConvertTo-Json|Set-Content -Encoding utf8 -LiteralPath (Join-Path $dir 'run-metadata.json')
    $f=Read-Fields $result
    Assert-Field $f 'execution_mode' $Mode
    Assert-Field $f 'status' 'SUCCESS'
    Assert-Field $f 'cpu_fallback' 'false'
    Assert-Field $f 'nan_detected' 'false'
    Assert-Field $f 'inf_detected' 'false'
    Assert-Field $f 'compile_time_sdk_build_id' $ExpectedBuildId
    if($Mode -like '*HTP_DW*'){
        Assert-Field $f 'htp_dw_used' 'true'; Assert-Field $f 'cpu_dw_fallback' 'false'
        Assert-Field $f 'dw_graph_execute_result' 'success'; Assert-Field $f 'dw_graph_create_count' '1'
        Assert-Field $f 'dw_graph_finalize_count' '1'; Assert-Field $f 'runtime_weight_update_failures' '0'
    }
    $script:rows.Add([pscustomobject]@{
        run_name=$Name;phase=$phase
        seed=$Seed;mode=$Mode;batch_size=$Batch;input_dim=$Dimension;output_dim=$Dimension
        steps=(Number $f 'steps' $Steps);run_index=$RunIndex;forward_backend=$f['forward_backend'];dw_backend=$f['dw_backend']
        transpose_backend=$f['transpose_backend'];initial_loss=(Number $f 'initial_loss');final_loss=(Number $f 'final_loss')
        initial_weight_error=(Number $f 'initial_weight_error');final_weight_error=(Number $f 'final_weight_error')
        forward_median_us=(Number $f 'steady_state_execute_median_us');dp_generation_median_us=(Number $f 'cpu_dp_generation_median_us')
        transpose_median_us=(Number $f 'x_transpose_median_us');dw_execute_median_us=(Number $f 'dw_execute_median_us')
        dw_copy_median_us=((Number $f 'x_or_xt_copy_median_us')+(Number $f 'dp_copy_median_us')+(Number $f 'dw_output_copy_median_us'))
        cpu_dw_median_us=(Number $f 'cpu_gradient_median_us');optimizer_median_us=(Number $f 'cpu_optimizer_median_us')
        runtime_update_median_us=(Number $f 'runtime_weight_update_median_us');full_step_median_us=(Number $f 'full_step_median_us')
        total_time_us=(Number $f 'total_training_time_us');initialization_us=(Number $f 'initialization_us')
        dw_max_abs_error=(Number $f 'dw_max_abs_error');cpu_fallback=$f['cpu_fallback'];status=$f['status']
        temperature_before=$before.temperature_c;temperature_after=$after.temperature_c
        thermal_status_before=$before.status;thermal_status_after=$after.status
    })
    if($null -ne $after.temperature_c -and $after.temperature_c -ge 45.0){throw "Device reached $($after.temperature_c)C; stopping."}
}

Push-Location $root
try{
    Write-Host 'Estimated duration: correctness/seeds 5-10 minutes; performance 10-20 minutes; default batch-8 break-even 10-20 minutes.'
    if(!$SkipBuild){
        .\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
        if($LASTEXITCODE -ne 0){throw 'QAIRT clean build failed'}
        & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $ReportRoot 'apk-audit.txt')
    }
    Invoke-Adb @('install','-r',$apk)|Out-Null
    Require-Cool 'correctness'|Out-Null
    Run-Mode 'QNN_HTP_DW_CHECK' 'correctness-small' 2 4 1 4 0 0.1 $Seeds[0] 1 $false
    Require-Cool 'five-seed-training'|Out-Null
    for($i=0;$i -lt $Seeds.Count;$i++){
        $order=if($i%2 -eq 0){@('QNN_HTP_TRAINING_BENCHMARK','QNN_HTP_FORWARD_HTP_DW_TRAINING')}else{@('QNN_HTP_FORWARD_HTP_DW_TRAINING','QNN_HTP_TRAINING_BENCHMARK')}
        foreach($mode in $order){Run-Mode $mode ("seed-$($Seeds[$i])-$mode") 8 128 640 512 10 5 $Seeds[$i] 1 $false}
    }
    if($RunPerformance){
        Require-Cool 'performance'|Out-Null
        $shapes=@(@(2,4),@(8,128),@(8,192),@(8,256),@(32,256),@(32,512))
        foreach($shape in $shapes){for($run=1;$run -le $Repetitions;$run++){
            $modes=if($run%2 -eq 1){@('QNN_HTP_TRAINING_BENCHMARK','QNN_HTP_FORWARD_HTP_DW_BENCHMARK')}else{@('QNN_HTP_FORWARD_HTP_DW_BENCHMARK','QNN_HTP_TRAINING_BENCHMARK')}
            foreach($mode in $modes){Run-Mode $mode ("sweep-b$($shape[0])-d$($shape[1])-r$run-$mode") $shape[0] $shape[1] 100 512 0 5 $Seeds[0] $run $true}
        }}
    }
    if($RunBreakEven){
        Require-Cool 'break-even'|Out-Null
        foreach($b in $BreakEvenBatches){foreach($d in $BreakEvenDimensions){for($run=1;$run -le $Repetitions;$run++){
            foreach($mode in @('QNN_HTP_TRAINING_BENCHMARK','QNN_HTP_FORWARD_HTP_DW_BENCHMARK')){Run-Mode $mode ("break-b$b-d$d-r$run-$mode") $b $d 100 512 0 5 $Seeds[0] $run $true}
        }}}
    }
    $script:rows=@($script:rows|Group-Object phase,seed,mode,batch_size,input_dim,steps,run_index|ForEach-Object{$_.Group[-1]})
    $script:rows|Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'sweep.csv')
    $seedRows=$script:rows|Where-Object{$_.mode -in @('QNN_HTP_TRAINING_BENCHMARK','QNN_HTP_FORWARD_HTP_DW_TRAINING') -and $_.steps -eq 640}
    $seedRows|Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'seeds.csv')
    $correctness=$script:rows|Where-Object{$_.mode -eq 'QNN_HTP_DW_CHECK'}
    $correctness|Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'correctness.csv')
    $breakRows=[Collections.Generic.List[object]]::new()
    foreach($group in ($script:rows|Where-Object{$_.mode -like '*BENCHMARK'}|Group-Object batch_size,input_dim)){
        $cpu=@($group.Group|Where-Object{$_.mode -eq 'QNN_HTP_TRAINING_BENCHMARK'})
        $htp=@($group.Group|Where-Object{$_.mode -eq 'QNN_HTP_FORWARD_HTP_DW_BENCHMARK'})
        if($cpu.Count -and $htp.Count){
            $cpuDw=($cpu|Measure-Object cpu_dw_median_us -Average).Average
            $htpDw=($htp|ForEach-Object{$_.dw_execute_median_us+$_.dw_copy_median_us}|Measure-Object -Average).Average
            $cpuStep=($cpu|Measure-Object full_step_median_us -Average).Average; $htpStep=($htp|Measure-Object full_step_median_us -Average).Average
            $initDelta=($htp|Measure-Object initialization_us -Average).Average-($cpu|Measure-Object initialization_us -Average).Average
            $saving=$cpuStep-$htpStep
            $breakRows.Add([pscustomobject]@{batch_size=$cpu[0].batch_size;dimension=$cpu[0].input_dim;cpu_dw_us=$cpuDw;htp_dw_path_us=$htpDw;dw_faster=($htpDw-lt$cpuDw);cpu_full_step_us=$cpuStep;htp_full_step_us=$htpStep;full_step_faster=($htpStep-lt$cpuStep);initialization_delta_us=$initDelta;break_even_steps=if($saving-gt 0){[Math]::Ceiling([Math]::Max(0,$initDelta)/$saving)}else{-1}})
        }
    }
    $breakRows|Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'break-even.csv')
    $summary=[ordered]@{test='PHONELM_QNN_HTP_BACKWARD_DW';qairt_build_id=$ExpectedBuildId;status='SUCCESS';cpu_fallback=$false;rows=$script:rows;break_even=$breakRows}
    $summary|ConvertTo-Json -Depth 8|Set-Content -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'summary.json')
    @("# QNN HTP dW summary","","- QAIRT Build ID: ``$ExpectedBuildId``","- Status: SUCCESS","- CPU fallback: false","- Runs: $($script:rows.Count)","- See ``correctness.csv``, ``seeds.csv``, ``sweep.csv``, and ``break-even.csv``.")|Set-Content -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'summary.md')
    if(!$SkipRegression){
        & (Join-Path $PSScriptRoot 'run_qnn_device_tests.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -RepeatCount 1
        & (Join-Path $PSScriptRoot 'run_qnn_training_tests.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -BatchSize 8 -InputDim 128 -Steps 100 -LearningRate 5 -SkipBuild -SkipRegression
    }
    Write-Host "report_root=$ReportRoot"
}finally{Pop-Location}
