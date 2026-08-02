param(
    [Parameter(Mandatory = $true)][Alias("SdkRoot")][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [ValidateRange(1, 100)][int]$RepeatCount = 10,
    [switch]$SkipCorruptionRecovery
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
$reportRoot = Join-Path $root "build\reports\qnn-device-tests-2.48"
$apk = Join-Path $root "app\build\outputs\apk\debug\app-debug.apk"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

$online = @()
foreach ($line in (& $adb devices)) {
    if ($line -match '^(\S+)\s+device$') { $online += $Matches[1] }
}
if ($online.Count -ne 1) { throw "Expected exactly one online ADB device; found $($online.Count)." }
$device = $online[0]

function Invoke-Adb([string[]]$Arguments) {
    $output = & $adb -s $device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output" }
    return $output
}

function Assert-Match([string]$Text, [string]$Pattern, [string]$Description) {
    if ($Text -notmatch $Pattern) { throw "Missing required evidence: $Description" }
}

function Run-Mode([string]$Mode, [string]$Name) {
    Invoke-Adb @("shell", "am", "force-stop", $package) | Out-Null
    Invoke-Adb @("logcat", "-c") | Out-Null
    & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null | Out-Null
    Invoke-Adb @("shell", "am", "start", "-W", "-n", $activity, "--es", "phonelm.mode", $Mode) | Out-Null
    $result = ""
    for ($poll = 0; $poll -lt 240; $poll++) {
        Start-Sleep -Milliseconds 500
        $result = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
        if ($result -match "execution_mode=$Mode" -and $result -match '(?m)^status=(SUCCESS|FAILED)$') { break }
    }
    $log = (& $adb -s $device logcat -d -b all -v threadtime) -join "`n"
    $resultPath = Join-Path $reportRoot "$Name-result.txt"
    $logPath = Join-Path $reportRoot "$Name-logcat.txt"
    $result | Set-Content -LiteralPath $resultPath -Encoding utf8
    $log | Set-Content -LiteralPath $logPath -Encoding utf8
    Assert-Match $result '(?m)^status=SUCCESS$' "$Mode status"
    Assert-Match $result '(?m)^cpu_fallback=false$' "$Mode CPU fallback guard"
    Assert-Match $result "(?m)^compile_time_sdk_build_id=$([regex]::Escape($ExpectedBuildId))$" "compile-time QAIRT build ID"
    Assert-Match $result "(?m)^runtime_backend_build_id=v?$([regex]::Escape($ExpectedBuildId))$" "runtime backend build ID"
    Assert-Match $result '(?m)^backend_build_id_match=true$' "compile/runtime backend match"
    Assert-Match $result '(?m)^provider_core_api_version=2\.37\.0$' "provider core API"
    Assert-Match $result '(?m)^provider_backend_api_version=5\.48\.0$' "provider backend API"
    Assert-Match $result "(?m)^qnn_skel_dir=.*files/qnn-dsp/$([regex]::Escape($ExpectedBuildId))$" "versioned app-private Skel path"
    Assert-Match $result '(?ms)^qnn_skel_expected_sha256=([0-9a-f]{64})$.*^qnn_skel_actual_sha256=\1$' "deployed Skel SHA-256"
    if ($log -match 'qnn_open.*(0x80000600|error)|getHandle.*(0xf|error)|transport error.*1002') {
        throw "FastRPC/Skel error signature found in $logPath"
    }
    if ($Mode -eq 'QNN_HTP_DEVICE_PROBE') {
        foreach ($required in @(
            '(?m)^QnnBackend_create=0$', '(?m)^QnnDevice_create=0$', '(?m)^QnnContext_create=0$',
            '(?m)^device_handle_null=false$', '(?m)^context_handle_null=false$'
        )) { Assert-Match $result $required "device probe API result $required" }
    } elseif ($Mode -eq 'QNN_HTP_FORWARD') {
        foreach ($required in @(
            '(?m)^backend_library=libQnnHtp\.so$', '(?m)^graph_create=success$',
            '(?m)^graph_finalize=success$', '(?m)^graph_execute=success$',
            '(?m)^runtime_weight_update=success$', '(?m)^second_execute=success$',
            '(?m)^runtime_weight_update_worked=true$', '(?m)^npu_forward_used=true$'
        )) { Assert-Match $result $required "HTP forward result $required" }
        $errorMatch = [regex]::Match($result, '(?m)^forward_max_absolute_error=([0-9.eE+-]+)$')
        if (-not $errorMatch.Success -or [double]::Parse($errorMatch.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) -gt 1.0e-5) {
            throw "Forward maximum absolute error is missing or exceeds 1e-5"
        }
    }
    return [pscustomobject]@{ Result = $result; Log = $log }
}

Push-Location $root
try {
    .\gradlew.bat :app:clean :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
    if ($LASTEXITCODE -ne 0) { throw "Clean QAIRT/QNN APK build failed" }
    & (Join-Path $PSScriptRoot "audit_qnn_apk.ps1") -ApkPath $apk -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath "build/reports/qnn-apk-audit-2.48.txt"
    if ($LASTEXITCODE -ne 0) { throw "APK audit failed" }
    Invoke-Adb @("install", "-r", $apk) | Out-Null

    $allLogs = [Text.StringBuilder]::new()
    for ($i = 1; $i -le $RepeatCount; $i++) {
        $probe = Run-Mode "QNN_HTP_DEVICE_PROBE" ("probe-{0:d2}" -f $i)
        [void]$allLogs.AppendLine($probe.Log); [void]$allLogs.AppendLine($probe.Result)
    }
    for ($i = 1; $i -le $RepeatCount; $i++) {
        $forward = Run-Mode "QNN_HTP_FORWARD" ("forward-{0:d2}" -f $i)
        [void]$allLogs.AppendLine($forward.Log); [void]$allLogs.AppendLine($forward.Result)
    }
    for ($i = 1; $i -le 3; $i++) {
        $probe = Run-Mode "QNN_HTP_DEVICE_PROBE" ("cycle-{0:d2}-probe" -f $i)
        $forward = Run-Mode "QNN_HTP_FORWARD" ("cycle-{0:d2}-forward" -f $i)
        Invoke-Adb @("shell", "am", "force-stop", $package) | Out-Null
        [void]$allLogs.AppendLine($probe.Log); [void]$allLogs.AppendLine($probe.Result); [void]$allLogs.AppendLine($forward.Log); [void]$allLogs.AppendLine($forward.Result)
    }
    Assert-Match $allLogs.ToString() 'qnn_skel_action=reused' "Skel reuse without recopy"

    $recovery = "SKIPPED"
    if (-not $SkipCorruptionRecovery) {
        $relativeSkel = "files/qnn-dsp/$ExpectedBuildId/libQnnHtpV81Skel.so"
        Invoke-Adb @("shell", "run-as", $package, "dd", "if=/dev/zero", "of=$relativeSkel", "bs=4", "count=1") | Out-Null
        $recovered = Run-Mode "QNN_HTP_DEVICE_PROBE" "corruption-recovery-probe"
        Assert-Match $recovered.Result '(?m)^qnn_skel_action=replaced$' "corrupt app-private Skel recovery"
        $recovery = "SUCCESS"
        [void]$allLogs.AppendLine($recovered.Log); [void]$allLogs.AppendLine($recovered.Result)
    }

    if ($allLogs.ToString() -match '(Fatal signal|F DEBUG|tombstone|handle.*(leak|failed to free)|FastRPC.*leak)') {
        throw "Crash, handle-release failure, or FastRPC leak signature detected"
    }
    @(
        "test=PHONELM_QNN_HTP_REGRESSION",
        "qairt_build_id=$ExpectedBuildId",
        "probe_repetitions=$RepeatCount",
        "forward_repetitions=$RepeatCount",
        "probe_forward_force_stop_cycles=3",
        "skel_reuse=SUCCESS",
        "skel_corruption_recovery=$recovery",
        "cpu_fallback=false",
        "status=SUCCESS"
    ) | Set-Content -LiteralPath (Join-Path $reportRoot "summary.txt") -Encoding utf8
    Get-Content -LiteralPath (Join-Path $reportRoot "summary.txt")
} catch {
    $failureLog = (& $adb -s $device logcat -d -b all -v threadtime) -join "`n"
    $failureLog | Set-Content -LiteralPath (Join-Path $reportRoot "failure-logcat.txt") -Encoding utf8
    @("status=FAILED", "error=$($_.Exception.Message)") | Set-Content -LiteralPath (Join-Path $reportRoot "summary.txt") -Encoding utf8
    throw
} finally {
    Pop-Location
}
