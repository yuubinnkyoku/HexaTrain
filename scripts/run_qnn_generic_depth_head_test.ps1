[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9][a-z0-9._-]*$')][string]$ConfigurationId,
    [ValidateRange(1, 65536)][int]$SequenceLength = 8,
    [ValidateRange(13, 65536)][int]$VocabularySize = 32,
    [ValidateRange(1, 65536)][int]$EmbeddingDimension = 16,
    [ValidateRange(1, 65536)][int]$FeedForwardDimension = 32,
    [ValidateRange(1, 65536)][int]$NumLayers = 1,
    [ValidateRange(1, 65536)][int]$NumHeads = 1,
    [ValidateRange(1, 100000)][int]$Steps = 1,
    [ValidateRange(1, 100000)][int]$Seeds = 1,
    [ValidateRange(10, 86400)][int]$TimeoutSeconds = 1800,
    [ValidateSet('CORRECTNESS', 'EXCLUSIVE_BENCHMARK', 'UI_VALIDATION')]
    [string]$TestMode = 'CORRECTNESS',
    [switch]$InstallAuditedApk,
    [switch]$AllowFailure
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Fail([string]$Message) { throw "generic depth/head device test: $Message" }
function Invoke-Adb([string[]]$Arguments) {
    $output = & $script:Adb -s $script:Device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output"
    }
    return $output
}
function Read-DeviceState {
    $battery = (Invoke-Adb @('shell', 'dumpsys', 'battery')) -join "`n"
    $thermal = (Invoke-Adb @('shell', 'dumpsys', 'thermalservice')) -join "`n"
    $level = [regex]::Match($battery, '(?m)^\s*level:\s*(\d+)').Groups[1].Value
    $temperatureTenths = [regex]::Match(
        $battery, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
    $thermalStatus = [regex]::Match(
        $thermal, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
    if (-not $level -or -not $temperatureTenths -or -not $thermalStatus) {
        Fail 'could not read battery level, battery temperature, or Android thermal status'
    }
    return [pscustomobject][ordered]@{
        battery_level = [int]$level
        battery_temperature_c = [int]$temperatureTenths / 10.0
        android_thermal_status = [int]$thermalStatus
    }
}
function Result-Value([string]$Report, [string]$Key) {
    $match = [regex]::Match(
        $Report, "(?m)^$([regex]::Escape($Key))=(.*)$")
    if ($match.Success) { return $match.Groups[1].Value.Trim() }
    return 'NOT_REPORTED'
}
function Read-PhoneNotification {
    $keys = (Invoke-Adb @('shell', 'cmd', 'notification', 'list')) -join "`n"
    $line = ($keys -split "`n" | Where-Object {
        $_ -match 'com\.yuubinnkyoku\.phonelm'
    } | Select-Object -First 1)
    if (-not $line) { return '' }
    $key = $line.Trim()
    # Notification keys contain pipe characters. Quote the complete key so the
    # remote Android shell does not interpret those characters as pipelines.
    $quotedKey = "'$key'"
    return (Invoke-Adb @(
        'shell', 'cmd', 'notification', 'get', $quotedKey)) -join "`n"
}
function Read-UiHierarchy {
    $remote = '/sdcard/phonelm-generic-ui-validation.xml'
    Invoke-Adb @('shell', 'uiautomator', 'dump', $remote) | Out-Null
    try {
        return (Invoke-Adb @('shell', 'cat', $remote)) -join "`n"
    } finally {
        Invoke-Adb @('shell', 'rm', '-f', $remote) | Out-Null
    }
}

if (($EmbeddingDimension % $NumHeads) -ne 0) {
    Fail 'EmbeddingDimension must be divisible by NumHeads'
}

$root = Split-Path -Parent $PSScriptRoot
$script:Adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $script:Adb -PathType Leaf)) {
    Fail 'adb executable is unavailable'
}
$online = @((& $script:Adb devices) | Where-Object {
    $_ -match '^(\S+)\s+device$'
} | ForEach-Object { ($_ -split '\s+')[0] })
if ($online.Count -ne 1) {
    Fail "expected exactly one online ADB device; found $($online.Count)"
}
$script:Device = $online[0]

Push-Location $root
try {
    & .\scripts\check_qairt.ps1 -SdkRoot $QairtSdkRoot `
        -ExpectedBuildId $ExpectedBuildId | Out-Null
    $inventoryExit = $LASTEXITCODE
    if ($inventoryExit -notin @(0, 3)) {
        Fail "fixed QAIRT inventory check failed with exit $inventoryExit"
    }
    $apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
    $auditPath = Join-Path $root 'build\reports\qnn-apk-audit-generic.txt'
    $apkHash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
    $cachedAudit = if (Test-Path -LiteralPath $auditPath -PathType Leaf) {
        Get-Content -LiteralPath $auditPath -Raw
    } else { '' }
    $auditMatches = $cachedAudit -match "(?m)^qairt_build_id=$([regex]::Escape($ExpectedBuildId))\r?$" -and
        $cachedAudit -match "(?m)^apk_sha256=$([regex]::Escape($apkHash))\r?$" -and
        $cachedAudit -match '(?m)^forbidden_2_47_strings=false\r?$' -and
        $cachedAudit -match '(?m)^host_sdk_path_present=false\r?$' -and
        $cachedAudit -match '(?m)^status=SUCCESS\r?$'
    if (-not $auditMatches) {
        & .\scripts\audit_qnn_apk.ps1 -ApkPath $apk `
            -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
            -ReportPath $auditPath | Out-Null
        if (-not $?) { Fail 'fixed QAIRT APK audit failed' }
    }
    if ($InstallAuditedApk) {
        Invoke-Adb @('install', '-r', $apk) | Out-Null
    }

    $before = Read-DeviceState
    if ($before.battery_temperature_c -ge 45.0) {
        Fail "THERMAL_ABORT: battery temperature is $($before.battery_temperature_c) C"
    }
    if ($before.android_thermal_status -ge 3) {
        Fail "THERMAL_ABORT: Android thermal status is $($before.android_thermal_status)"
    }

    $package = 'com.yuubinnkyoku.phonelm'
    $activity = "$package/.MainActivity"
    Invoke-Adb @('shell', 'am', 'force-stop', $package) | Out-Null
    & $script:Adb -s $script:Device shell run-as $package rm -f `
        files/device-test-result.txt 2>$null | Out-Null
    $arguments = @(
        'shell', 'am', 'start', '-W', '-n', $activity,
        '--es', 'phonelm.mode', 'QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC',
        '--ei', 'phonelm.batch_size', '1',
        '--ei', 'phonelm.dimension', [string]$EmbeddingDimension,
        '--ei', 'phonelm.hidden_dimension', [string]$FeedForwardDimension,
        '--ei', 'phonelm.output_dimension', [string]$VocabularySize,
        '--ei', 'phonelm.steps', [string]$Steps,
        '--ei', 'phonelm.warmup_steps', '0',
        '--es', 'phonelm.learning_rate', '0.003',
        '--es', 'phonelm.seed', '1',
        '--ei', 'phonelm.sample_count', [string]$SequenceLength,
        '--ei', 'phonelm.epochs', [string]$NumLayers,
        '--ei', 'phonelm.measured_steps', [string]$NumHeads,
        '--ei', 'phonelm.correctness_interval', [string]$Seeds,
        '--ez', 'phonelm.benchmark_mode',
        $(if ($TestMode -eq 'EXCLUSIVE_BENCHMARK') { 'true' } else { 'false' })
    )
    if ($TestMode -eq 'UI_VALIDATION') {
        $arguments += @('--ez', 'phonelm.live_update', 'true')
    }
    $wall = [Diagnostics.Stopwatch]::StartNew()
    Invoke-Adb $arguments | Out-Null
    $uiConfigVisible = $false
    $uiProgressVisible = $false
    $uiForegroundUpdate = $false
    $uiBackgroundUpdate = $false
    $uiOngoingDuringRun = $false
    $uiOngoingAfterCompletion = $true
    $uiNotificationTapReturnedActivity = $false
    $uiAutoCancel = $false
    if ($TestMode -eq 'UI_VALIDATION') {
        Start-Sleep -Seconds 3
        $hierarchy = Read-UiHierarchy
        $uiConfigVisible =
            $hierarchy -match 'generic_configuration=.*L\d+ H\d+'
        $foregroundNotification = Read-PhoneNotification
        $uiForegroundUpdate = $foregroundNotification -match 'PhoneLM'
        $uiProgressVisible =
            $foregroundNotification -match '(step|training|seed|loss|初期化|学習)'
        $uiOngoingDuringRun =
            $foregroundNotification -match '(?i)(ongoing|FLAG_ONGOING_EVENT|flags=0x[0-9a-f]*2)'
        Invoke-Adb @('shell', 'input', 'keyevent', 'KEYCODE_HOME') | Out-Null
        Start-Sleep -Seconds 4
        $backgroundNotification = Read-PhoneNotification
        $uiBackgroundUpdate =
            $backgroundNotification -match 'PhoneLM' -and
            $backgroundNotification -ne $foregroundNotification
    }
    $report = ''
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $candidate = (& $script:Adb -s $script:Device shell run-as $package `
            cat files/device-test-result.txt 2>$null) -join "`n"
        if ($candidate -and $candidate -notmatch '^cat:') { $report = $candidate }
        if ($report -match '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)$') {
            break
        }
    }
    $wall.Stop()
    if (-not $report) { Fail 'TIMEOUT: no device result was produced' }
    if ($report -notmatch '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)$') {
        Fail 'TIMEOUT: device result did not reach a terminal status'
    }
    if ($TestMode -eq 'UI_VALIDATION') {
        $completedNotification = ''
        foreach ($attempt in 1..20) {
            Start-Sleep -Milliseconds 500
            $completedNotification = Read-PhoneNotification
            if ($completedNotification -match '(完了|失敗|キャンセル)') { break }
        }
        $uiOngoingAfterCompletion =
            $completedNotification -match '(?i)(ongoing|FLAG_ONGOING_EVENT|flags=0x[0-9a-f]*2)'
        $completedAutoCancel =
            $completedNotification -match '(?i)(auto.?cancel|FLAG_AUTO_CANCEL|flags=0x[0-9a-f]*1[0-9a-f]*)'
        Invoke-Adb @('shell', 'cmd', 'statusbar', 'expand-notifications') | Out-Null
        Start-Sleep -Seconds 2
        $phoneNode = [System.Text.RegularExpressions.Match]::Empty
        foreach ($attempt in 1..6) {
            $shade = Read-UiHierarchy
            $phoneNode = [regex]::Match(
                $shade,
                '<node[^>]*(?:text|content-desc)="[^"]*PhoneLM[^"]*(?:完了|失敗|キャンセル)[^"]*"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
            if (-not $phoneNode.Success) {
                $phoneNode = [regex]::Match(
                    $shade,
                    '<node[^>]*(?:text|content-desc)="[^"]*PhoneLM[^"]*"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
            }
            if ($phoneNode.Success) { break }
            Invoke-Adb @('shell', 'input', 'swipe', '1000', '800', '1000', '300', '400') | Out-Null
            Start-Sleep -Milliseconds 500
        }
        if ($phoneNode.Success) {
            $tapX = ([int]$phoneNode.Groups[1].Value +
                [int]$phoneNode.Groups[3].Value) / 2
            $tapY = ([int]$phoneNode.Groups[2].Value +
                [int]$phoneNode.Groups[4].Value) / 2
            Invoke-Adb @('shell', 'input', 'tap', [string][int]$tapX,
                [string][int]$tapY) | Out-Null
            foreach ($attempt in 1..10) {
                Start-Sleep -Milliseconds 500
                $activities = (Invoke-Adb @(
                    'shell', 'dumpsys', 'activity', 'activities')) -join "`n"
                if ($activities -match
                    '(?:mResumedActivity:|topResumedActivity=).*com\.yuubinnkyoku\.phonelm') {
                    $uiNotificationTapReturnedActivity = $true
                    break
                }
            }
            foreach ($attempt in 1..10) {
                if (-not (Read-PhoneNotification)) {
                    $uiAutoCancel = $completedAutoCancel
                    break
                }
                Start-Sleep -Milliseconds 500
            }
        }
    }
    $after = Read-DeviceState

    $reportRoot = Join-Path $root 'build\reports\qnn-generic-depth-head'
    [IO.Directory]::CreateDirectory($reportRoot) | Out-Null
    $privatePath = Join-Path $reportRoot "$ConfigurationId.txt"
    @(
        "configuration_id=$ConfigurationId"
        "test_mode=$TestMode"
        'foreground_intentional=true'
        'focus_takeover_count=NOT_APPLICABLE_FOREGROUND_RUN'
        "inventory_exit=$inventoryExit"
        "battery_level_before=$($before.battery_level)"
        "battery_temperature_before_c=$($before.battery_temperature_c)"
        "thermal_status_before=$($before.android_thermal_status)"
        "battery_level_after=$($after.battery_level)"
        "battery_temperature_after_c=$($after.battery_temperature_c)"
        "thermal_status_after=$($after.android_thermal_status)"
        "wall_time_ms=$($wall.Elapsed.TotalMilliseconds)"
        $report
    ) | Set-Content -LiteralPath $privatePath -Encoding utf8
    $status = Result-Value $report 'status'
    if ($TestMode -eq 'UI_VALIDATION') {
        $uiPassed = $status -eq 'SUCCESS' -and $uiConfigVisible -and
            $uiProgressVisible -and $uiForegroundUpdate -and
            $uiBackgroundUpdate -and $uiOngoingDuringRun -and
            -not $uiOngoingAfterCompletion -and
            $uiNotificationTapReturnedActivity -and $uiAutoCancel
        @(
            "status=$(if ($uiPassed) { 'PASS' } else { 'FAIL' })"
            "config_visible=$($uiConfigVisible.ToString().ToLowerInvariant())"
            "progress_visible=$($uiProgressVisible.ToString().ToLowerInvariant())"
            "foreground_update=$($uiForegroundUpdate.ToString().ToLowerInvariant())"
            "background_update=$($uiBackgroundUpdate.ToString().ToLowerInvariant())"
            "ongoing_during_run=$($uiOngoingDuringRun.ToString().ToLowerInvariant())"
            "ongoing_after_completion=$($uiOngoingAfterCompletion.ToString().ToLowerInvariant())"
            "notification_tap_returned_activity=$($uiNotificationTapReturnedActivity.ToString().ToLowerInvariant())"
            "auto_cancel=$($uiAutoCancel.ToString().ToLowerInvariant())"
        ) | Set-Content -LiteralPath (
            Join-Path $reportRoot 'ui_validation_summary.txt') -Encoding utf8
    }

    [pscustomobject][ordered]@{
        configuration_id = $ConfigurationId
        status = $status
        failure_classification = Result-Value $report 'failure_classification'
        sequence_length = Result-Value $report 'sequence_length'
        embedding_dimension = Result-Value $report 'embedding_dimension'
        feed_forward_dimension = Result-Value $report 'feed_forward_dimension'
        transformer_layers = Result-Value $report 'transformer_layers'
        attention_heads = Result-Value $report 'attention_heads'
        steps = Result-Value $report 'steps'
        seed_count = Result-Value $report 'seed_count'
        nonfinite_count = Result-Value $report 'seed_1_nonfinite_count'
        qnn_nonzero_return_count = Result-Value $report 'formal_qnn_nonzero_return_count'
        graph_creation_ms = Result-Value $report 'performance_graph_creation_ms'
        finalize_ms = Result-Value $report 'performance_finalize_ms'
        peak_rss_kib = Result-Value $report 'process_peak_rss_kib'
        battery_temperature_before_c = $before.battery_temperature_c
        battery_temperature_after_c = $after.battery_temperature_c
        thermal_status_before = $before.android_thermal_status
        thermal_status_after = $after.android_thermal_status
        private_report = $privatePath
    } | Format-List
    if ($status -ne 'SUCCESS' -and -not $AllowFailure) {
        Fail "device report status is $status"
    }
} finally {
    Pop-Location
}
