param(
    [Parameter(Mandatory = $true)][Alias("SdkRoot")][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [ValidateRange(1, 4096)][int]$BatchSize = 8,
    [ValidateRange(1, 4096)][int]$InputDim = 128,
    [ValidateRange(1, 100000)][int]$Steps = 100,
    [ValidateRange(0.000001, 10.0)][double]$LearningRate = 5.0,
    [long]$Seed = 20260710,
    [switch]$RunLarge,
    [switch]$SkipBuild,
    [switch]$SkipRegression
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "qairt_version.ps1")
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA "Android\Sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = "com.yuubinnkyoku.phonelm"
$activity = "$package/.MainActivity"
$apk = Join-Path $root "app\build\outputs\apk\debug\app-debug.apk"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reportRoot = Join-Path $root "build\reports\qnn-training-$timestamp"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

$online = @()
foreach ($line in (& $adb devices)) {
    if ($line -match '^(\S+)\s+device$') { $online += $Matches[1] }
}
if ($online.Count -ne 1) { throw "Expected exactly one online ADB device; found $($online.Count)." }
$device = $online[0]

function Invoke-Adb([string[]]$Arguments) {
    $output = & $adb -s $device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output"
    }
    return $output
}

function Read-Fields([string]$Text) {
    $fields = [ordered]@{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $fields[$Matches[1]] = $Matches[2] }
    }
    return $fields
}

function Assert-Field($Fields, [string]$Name, [string]$Expected) {
    if (-not $Fields.Contains($Name) -or $Fields[$Name] -ne $Expected) {
        throw "Expected $Name=$Expected, actual=$($Fields[$Name])"
    }
}

function To-Number($Fields, [string]$Name) {
    if (-not $Fields.Contains($Name)) { throw "Missing numeric result field: $Name" }
    return [double]::Parse($Fields[$Name], [Globalization.CultureInfo]::InvariantCulture)
}

function Save-Artifacts([string]$Name, [string]$Result, [string]$Log) {
    $directory = Join-Path $reportRoot $Name
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $Result | Set-Content -LiteralPath (Join-Path $directory "device-test-result.txt") -Encoding utf8
    $Log | Set-Content -LiteralPath (Join-Path $directory "logcat.txt") -Encoding utf8
    $fields = Read-Fields $Result
    $fields | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $directory "summary.json") -Encoding utf8
    $match = [regex]::Match($Result, '(?ms)^loss_csv_begin\r?\n(.*?)^loss_csv_end$')
    if ($match.Success) { $match.Groups[1].Value.TrimEnd() | Set-Content -LiteralPath (Join-Path $directory "loss.csv") -Encoding utf8 }
    $timings = @("metric,value_us")
    foreach ($entry in $fields.GetEnumerator()) {
        if ($entry.Key -match '(_time_us|_us)$') { $timings += "$($entry.Key),$($entry.Value)" }
    }
    $timings | Set-Content -LiteralPath (Join-Path $directory "timings.csv") -Encoding utf8
    return $fields
}

function Run-Mode(
    [string]$Mode, [string]$Name, [int]$Batch, [int]$Dimension,
    [int]$StepCount, [double]$Rate
) {
    Invoke-Adb @("shell", "am", "force-stop", $package) | Out-Null
    Invoke-Adb @("logcat", "-c") | Out-Null
    & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null | Out-Null
    $rateText = $Rate.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
    Invoke-Adb @(
        "shell", "am", "start", "-W", "-n", $activity,
        "--es", "phonelm.mode", $Mode,
        "--ei", "phonelm.batch_size", "$Batch",
        "--ei", "phonelm.dimension", "$Dimension",
        "--ei", "phonelm.steps", "$StepCount",
        "--ei", "phonelm.warmup_steps", "0",
        "--es", "phonelm.learning_rate", $rateText,
        "--es", "phonelm.seed", "$Seed"
    ) | Out-Null
    $result = ""
    for ($poll = 0; $poll -lt 1200; $poll++) {
        Start-Sleep -Milliseconds 500
        $result = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
        if ($result -match "(?m)^execution_mode=$([regex]::Escape($Mode))$" -and
            $result -match '(?m)^status=(SUCCESS|FAILED)$') { break }
    }
    $log = (& $adb -s $device logcat -d -b all -v threadtime) -join "`n"
    $fields = Save-Artifacts $Name $result $log
    Assert-Field $fields "execution_mode" $Mode
    Assert-Field $fields "status" "SUCCESS"
    Assert-Field $fields "cpu_fallback" "false"
    Assert-Field $fields "nan_detected" "false"
    Assert-Field $fields "inf_detected" "false"
    if ($Mode -eq "QNN_HTP_LINEAR_TRAINING") {
        Assert-Field $fields "backend_library" "libQnnHtp.so"
        Assert-Field $fields "compile_time_sdk_build_id" $ExpectedBuildId
        if ($fields["runtime_backend_build_id"] -notmatch "^v?$([regex]::Escape($ExpectedBuildId))$") { throw "Runtime QAIRT build ID mismatch" }
        Assert-Field $fields "backend_build_id_match" "true"
        Assert-Field $fields "provider_core_api_version" "2.37.0"
        Assert-Field $fields "provider_backend_api_version" "5.48.0"
        Assert-Field $fields "npu_forward_used" "true"
        Assert-Field $fields "backward_on_htp" "false"
        Assert-Field $fields "optimizer_on_htp" "false"
        Assert-Field $fields "graph_create_count" "1"
        Assert-Field $fields "graph_finalize_count" "1"
        Assert-Field $fields "runtime_weight_update_count" "$StepCount"
        Assert-Field $fields "graph_execute_count" "$($StepCount + 1)"
        Assert-Field $fields "runtime_weight_update_worked" "true"
        Assert-Field $fields "output_matches_updated_cpu_reference" "true"
        if ((To-Number $fields "max_abs_error") -gt 1.0e-3) { throw "HTP training forward error exceeds 1e-3" }
        if ($log -match 'qnn_open.*(0x80000600|error)|getHandle.*(0xf|error)|transport error.*1002') {
            throw "FastRPC/Skel error signature found in $Name logcat"
        }
    }
    if ($Mode -ne "QNN_LINEAR_GRADIENT_CHECK") {
        if ((To-Number $fields "final_loss") -ge (To-Number $fields "initial_loss")) { throw "$Mode loss did not decrease" }
        if ((To-Number $fields "final_weight_error") -ge (To-Number $fields "initial_weight_error")) { throw "$Mode weight error did not decrease" }
    }
    return $fields
}

function Compare-Runs($Cpu, $Htp, [string]$Name) {
    $comparison = [ordered]@{
        name = $Name
        same_seed = $Cpu.seed -eq $Htp.seed
        same_shape = $Cpu.batch_size -eq $Htp.batch_size -and $Cpu.input_dim -eq $Htp.input_dim -and $Cpu.output_dim -eq $Htp.output_dim
        same_steps = $Cpu.steps -eq $Htp.steps
        same_learning_rate = $Cpu.learning_rate -eq $Htp.learning_rate
        cpu_initial_loss = To-Number $Cpu "initial_loss"
        htp_initial_loss = To-Number $Htp "initial_loss"
        cpu_final_loss = To-Number $Cpu "final_loss"
        htp_final_loss = To-Number $Htp "final_loss"
        final_weight_max_difference = To-Number $Htp "final_weight_difference"
        final_prediction_max_difference = To-Number $Htp "final_prediction_difference"
        htp_max_forward_error = To-Number $Htp "max_abs_error"
    }
    if (-not $comparison.same_seed -or -not $comparison.same_shape -or -not $comparison.same_steps -or -not $comparison.same_learning_rate) {
        throw "$Name CPU/HTP initial conditions differ"
    }
    if ($comparison.htp_max_forward_error -gt 1.0e-3) { throw "$Name HTP forward comparison failed" }
    $comparison | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot "$Name-comparison.json") -Encoding utf8
    return $comparison
}

Push-Location $root
try {
    if (-not $SkipBuild) {
        .\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
        if ($LASTEXITCODE -ne 0) { throw "Clean QAIRT/QNN APK build failed" }
        & (Join-Path $PSScriptRoot "audit_qnn_apk.ps1") -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $reportRoot "qnn-apk-audit.txt")
        if ($LASTEXITCODE -ne 0) { throw "APK audit failed" }
    }
    Invoke-Adb @("install", "-r", $apk) | Out-Null

    $gradient = Run-Mode "QNN_LINEAR_GRADIENT_CHECK" "gradient-check" 2 4 10 $LearningRate
    Assert-Field $gradient "gradient_check_passed" "true"

    $smallCpu = Run-Mode "QNN_CPU_LINEAR_TRAINING" "small-cpu" 2 4 10 $LearningRate
    $smallHtp = Run-Mode "QNN_HTP_LINEAR_TRAINING" "small-htp" 2 4 10 $LearningRate
    $smallComparison = Compare-Runs $smallCpu $smallHtp "small"

    $basicCpu = Run-Mode "QNN_CPU_LINEAR_TRAINING" "basic-cpu" $BatchSize $InputDim $Steps $LearningRate
    $basicHtp = Run-Mode "QNN_HTP_LINEAR_TRAINING" "basic-htp" $BatchSize $InputDim $Steps $LearningRate
    $basicComparison = Compare-Runs $basicCpu $basicHtp "basic"

    $largeStatus = "SKIPPED"
    if ($RunLarge) {
        $largeCpu = Run-Mode "QNN_CPU_LINEAR_TRAINING" "large-cpu" 32 512 200 $LearningRate
        $largeHtp = Run-Mode "QNN_HTP_LINEAR_TRAINING" "large-htp" 32 512 200 $LearningRate
        $largeComparison = Compare-Runs $largeCpu $largeHtp "large"
        $largeStatus = "SUCCESS"
    }

    $regressionStatus = "SKIPPED"
    if (-not $SkipRegression) {
        & (Join-Path $PSScriptRoot "run_qnn_device_tests.ps1") -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
        if ($LASTEXITCODE -ne 0) { throw "Existing QAIRT 2.48 regression failed" }
        $regressionStatus = "SUCCESS"
    }

    $summary = [ordered]@{
        test = "PHONELM_QNN_LINEAR_TRAINING"
        qairt_build_id = $ExpectedBuildId
        gradient_check = "SUCCESS"
        small = "SUCCESS"
        basic = "SUCCESS"
        large = $largeStatus
        existing_qnn_regression = $regressionStatus
        cpu_fallback = $false
        htp_backward = $false
        htp_optimizer = $false
        status = "SUCCESS"
    }
    $summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot "summary.json") -Encoding utf8
    $summary.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } | Set-Content -LiteralPath (Join-Path $reportRoot "summary.txt") -Encoding utf8
    Get-Content -LiteralPath (Join-Path $reportRoot "summary.txt")
} catch {
    $failureLog = (& $adb -s $device logcat -d -b all -v threadtime) -join "`n"
    $failureLog | Set-Content -LiteralPath (Join-Path $reportRoot "failure-logcat.txt") -Encoding utf8
    @{ status = "FAILED"; error = $_.Exception.Message } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot "summary.json") -Encoding utf8
    throw
} finally {
    Pop-Location
}
