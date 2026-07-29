# SPDX-License-Identifier: Apache-2.0
# Runs the formal post-fix Tiny LM evaluation through the existing headless gate.
param(
    [Parameter(Mandatory = $true)][Alias('SdkRoot')][string]$QairtSdkRoot,
    [string]$ExpectedBuildId = '2.48.40.260702151143',
    [ValidateRange(3, 5)][int]$Repetitions = 3,
    [switch]$RunPerformance,
    [ValidateRange(60, 86400)][int]$TimeoutSeconds = 14400,
    [string]$RunId = ('postfix-e2e-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildReportsRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot 'build\reports'))
$headlessReportsRoot = [IO.Path]::GetFullPath((Join-Path $buildReportsRoot 'qnn-headless'))
$reportRoot = [IO.Path]::GetFullPath((Join-Path $buildReportsRoot (Join-Path 'qnn-htp-post-fix-e2e' $RunId)))
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$headlessRunner = Join-Path $PSScriptRoot 'run_qnn_headless_tests.ps1'
$qairtChecker = Join-Path $PSScriptRoot 'check_qairt.ps1'

if (@(& git -C $repositoryRoot status --porcelain --untracked-files=normal).Count -ne 0) {
    throw 'Formal evaluation requires a clean worktree so source_commit identifies the built source exactly.'
}

function Assert-ChildPath([string]$Path, [string]$Parent, [string]$Description) {
    $resolvedPath = [IO.Path]::GetFullPath($Path)
    $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($resolvedParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description escaped its allowed directory."
    }
    return $resolvedPath
}

if ($RunId.Length -notin 1..64 -or $RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw 'RunId must match [A-Za-z0-9._-]+ and be at most 64 characters.'
}
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw "ADB was not found: $adb" }
if (-not (Test-Path -LiteralPath $headlessRunner -PathType Leaf)) { throw 'Headless runner was not found.' }
if (Test-Path -LiteralPath $reportRoot) { throw "Report run already exists: $RunId" }
$reportRoot = Assert-ChildPath $reportRoot $buildReportsRoot 'Report root'
$null = [IO.Directory]::CreateDirectory($reportRoot)

function Invoke-Adb([string[]]$Arguments) {
    $output = & $adb -s $script:device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output" }
    return ($output -join "`n")
}

function Get-TopPackageState {
    $activities = Invoke-Adb @('shell', 'dumpsys', 'activity', 'activities')
    $match = [regex]::Match($activities, '(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
    [ordered]@{
        top_resumed_known = $match.Success
        phonelm_is_top = $match.Success -and $match.Groups[1].Value -eq 'com.yuubinnkyoku.phonelm'
    }
}

function Get-DeviceState {
    $battery = Invoke-Adb @('shell', 'dumpsys', 'battery')
    $thermal = Invoke-Adb @('shell', 'dumpsys', 'thermalservice')
    $power = Invoke-Adb @('shell', 'dumpsys', 'power')
    $temperature = $null
    if ($battery -match '(?m)^\s*temperature:\s*(\d+)') { $temperature = [double]$Matches[1] / 10.0 }
    $batteryStatus = if ($battery -match '(?m)^\s*status:\s*(\d+)') { [int]$Matches[1] } else { $null }
    $pluggedValues = @()
    foreach ($source in @('AC', 'USB', 'Wireless')) {
        if ($battery -match "(?m)^\s*$([regex]::Escape($source)) powered:\s*(true|false)") {
            $pluggedValues += ($Matches[1] -eq 'true')
        }
    }
    $plugged = if ($pluggedValues.Count) { [bool]($pluggedValues -contains $true) } else { $null }
    $thermalStatus = if ($thermal -match '(?m)mStatus(?:Override)?=(\d+)') { [int]$Matches[1] } else { $null }
    $wakefulness = if ($power -match '(?m)^\s*mWakefulness=(\S+)') { $Matches[1] } else { 'unknown' }
    $top = Get-TopPackageState
    [ordered]@{
        captured_at_utc = [DateTime]::UtcNow.ToString('o')
        battery_temperature_c = $temperature
        battery_status = $batteryStatus
        plugged = $plugged
        thermal_status = $thermalStatus
        screen_wakefulness = $wakefulness
        top_resumed_known = $top.top_resumed_known
        phonelm_is_top = $top.phonelm_is_top
    }
}

function Assert-Cool([object]$State, [string]$Phase) {
    if ($null -ne $State.battery_temperature_c -and $State.battery_temperature_c -ge 45.0) {
        throw "$Phase refused: battery temperature is $($State.battery_temperature_c)C (limit 45C)."
    }
}

function Get-ReportValue([string]$Report, [string]$Key) {
    $match = [regex]::Match($Report, "(?m)^$([regex]::Escape($Key))=(.*)$")
    if (-not $match.Success) { throw "Device report is missing required field: $Key" }
    return $match.Groups[1].Value.Trim()
}

function Assert-FormalProtocol([string]$Report, [string]$Mode) {
    if ((Get-ReportValue $Report 'status') -ne 'SUCCESS') { throw "$Mode did not report status=SUCCESS." }
    $expected = [ordered]@{
        test = 'post_fix_end_to_end_generation'
        optimizer = 'ADAM'
        learning_rate = '0.003'
        steps = '320'
        sampling = 'pattern_round_robin_phase0'
        global_gradient_clipping = 'disabled'
        seed_count = '5'
        cpu_fallback = 'false'
        nan_detected = 'false'
        inf_detected = 'false'
        generation_nonfinite_detected = 'false'
        same_prefix_nonfinite_count = '0'
        generation_nonfinite_count = '0'
        generation_nan_count = '0'
        generation_inf_count = '0'
        logits_responsibility = 'HTP'
        argmax_responsibility = 'CPU'
        formal_oracle_case_count = '20'
        formal_free_case_count = '20'
        formal_cpu_all_finite = 'true'
        formal_prefix_comparisons_finite = 'true'
        free_running_context_update = 'PREVIOUS_PREDICTION'
        oracle_context_update = 'EXPECTED_TOKEN'
        free_running_teacher_forcing = 'false'
        logits_comparison_position = 'LAST_POSITION_V32'
        logits_comparison_parameter_scope = 'CPU_TRAINED_CPU_VS_HTP_TRAINED_HTP'
        raw_logits_visibility = 'PRIVATE_DEVICE_REPORT_ONLY'
        compile_time_qairt_build_id = $ExpectedBuildId
        backend_requested = 'HTP'
        headless_test_mode = $Mode
        focus_takeover_count = '0'
        phonelm_became_top_activity_count = '0'
        activity_create_count = '0'
        activity_resume_count = '0'
        single_flight_result = 'ALREADY_RUNNING'
    }
    foreach ($entry in $expected.GetEnumerator()) {
        if ((Get-ReportValue $Report $entry.Key) -ne $entry.Value) {
            throw "Formal protocol mismatch: $($entry.Key) must be $($entry.Value)."
        }
    }
    foreach ($numericKey in @('exact_rollout_count', 'oracle_exact_rollout_count', 'qualifying_seed_count', 'oracle_qualifying_seed_count')) {
        if ((Get-ReportValue $Report $numericKey) -notmatch '^\d+$') { throw "Device report has invalid $numericKey." }
    }
    foreach ($seed in 1..5) {
        foreach ($entry in ([ordered]@{
            "seed_${seed}_completed_steps" = '320'
            "seed_${seed}_all_steps_finite" = 'true'
            "seed_${seed}_final_evaluation_finite" = 'true'
            "seed_${seed}_nonfinite_count" = '0'
            "seed_${seed}_qnn_nonzero_return_count" = '0'
            "seed_${seed}_cpu_all_steps_finite" = 'true'
            "seed_${seed}_cpu_nonfinite_count" = '0'
        }).GetEnumerator()) {
            if ((Get-ReportValue $Report $entry.Key) -ne $entry.Value) {
                throw "Formal protocol mismatch: $($entry.Key) must be $($entry.Value)."
            }
        }
        if ((Get-ReportValue $Report "seed_${seed}_qnn_execute_count") -notmatch '^\d+$') {
            throw "Device report has invalid seed_${seed}_qnn_execute_count."
        }
    }
    if ((Get-ReportValue $Report 'api_trace_graph_execute_failure_count') -ne '0') {
        throw 'QNN graph execute failures were reported.'
    }
}

function Invoke-FormalRun([string]$Phase, [int]$Index, [string]$TestMode, [bool]$SkipPreparation) {
    $childRunId = "$RunId-$Phase-$('{0:d2}' -f $Index)"
    $childReportRoot = Assert-ChildPath (Join-Path $headlessReportsRoot $childRunId) $headlessReportsRoot 'Headless report root'
    $runDirectory = Assert-ChildPath (Join-Path $reportRoot "$Phase-$('{0:d2}' -f $Index)") $reportRoot 'Run directory'
    $null = [IO.Directory]::CreateDirectory($runDirectory)
    $before = Get-DeviceState
    Assert-Cool $before "$Phase run $Index"
    $metadata = [ordered]@{
        schema_version = 1
        source_commit = (& git -C $repositoryRoot rev-parse HEAD).Trim()
        phase = $Phase
        repetition = $Index
        headless_suite = 'post-fix-end-to-end'
        headless_test_mode = $TestMode
        headless_run_id = $childRunId
        headless_report_relative_path = ('qnn-headless/' + $childRunId + '/device-report.txt')
        device_before = $before
    }
    try {
        $arguments = @{
            QairtSdkRoot = $QairtSdkRoot
            ExpectedBuildId = $ExpectedBuildId
            Suite = 'post-fix-end-to-end'
            TestMode = $TestMode
            TimeoutSeconds = $TimeoutSeconds
            RunId = $childRunId
        }
        if ($SkipPreparation) {
            $arguments.SkipBuild = $true
            $arguments.SkipInstall = $true
            $arguments.SkipAudit = $true
        }
        & $headlessRunner @arguments
        if ($LASTEXITCODE -ne 0) { throw "Headless runner failed for $childRunId." }
        $deviceReportPath = Assert-ChildPath (Join-Path $childReportRoot 'device-report.txt') $headlessReportsRoot 'Device report'
        if (-not (Test-Path -LiteralPath $deviceReportPath -PathType Leaf)) { throw "Device report was not produced for $childRunId." }
        $report = Get-Content -Raw -LiteralPath $deviceReportPath
        Assert-FormalProtocol $report $TestMode
        foreach ($artifactName in @('device-report.txt', 'status.json', 'activity-sampling.json')) {
            $artifactPath = Assert-ChildPath (Join-Path $childReportRoot $artifactName) $headlessReportsRoot 'Headless artifact'
            if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
                throw "Headless artifact was not produced: $artifactName"
            }
            Copy-Item -LiteralPath $artifactPath -Destination (Join-Path $runDirectory $artifactName)
        }
        $metadata.device_report_status = Get-ReportValue $report 'status'
        $metadata.exact_rollout_count = [int](Get-ReportValue $report 'exact_rollout_count')
        $metadata.oracle_exact_rollout_count = [int](Get-ReportValue $report 'oracle_exact_rollout_count')
        $metadata.qnn_graph_execute_failure_count = [int](Get-ReportValue $report 'api_trace_graph_execute_failure_count')
        $metadata.status = 'SUCCESS'
    } catch {
        $metadata.status = 'FAILED'
        $metadata.error = $_.Exception.Message
        throw
    } finally {
        $metadata.device_after = Get-DeviceState
        $metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDirectory 'host-metadata.json') -Encoding utf8
        Assert-Cool $metadata.device_after "$Phase run $Index"
    }
    return [pscustomobject]$metadata
}

$summary = [ordered]@{
    schema_version = 1
    run_id = $RunId
    source_commit = (& git -C $repositoryRoot rev-parse HEAD).Trim()
    expected_build_id = $ExpectedBuildId
    repetitions = $Repetitions
    performance_requested = [bool]$RunPerformance
    device_serial_recorded = $false
    status = 'FAILED'
    runs = @()
}

try {
    $online = @((& $adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } | ForEach-Object { $Matches[1] })
    if ($online.Count -ne 1) { throw "Expected exactly one online ADB device; found $($online.Count)." }
    $script:device = $online[0]
    $emulatorProperties = @(
        Invoke-Adb @('shell', 'getprop', 'ro.kernel.qemu')
        Invoke-Adb @('shell', 'getprop', 'ro.boot.qemu')
        Invoke-Adb @('shell', 'getprop', 'ro.hardware')
    )
    if (($emulatorProperties -join "`n") -match '(?im)^(1|.*(goldfish|ranchu|emulator).*)$') {
        throw 'Emulator device rejected; a physical device is required.'
    }
    $qairtCheck = (& $qairtChecker -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId 2>&1) -join "`n"
    $qairtCheckExit = $LASTEXITCODE
    $qairtCheck | Write-Host
    $qairtCheckStatus = Get-ReportValue $qairtCheck 'status'
    if ($qairtCheckExit -eq 2 -or $qairtCheckStatus -eq 'BLOCKED_BY_QAIRT_SDK_NOT_INSTALLED') {
        throw 'QAIRT SDK was not found at the explicit root.'
    }
    if ($qairtCheckExit -eq 4 -or $qairtCheckStatus -eq 'QAIRT_BUILD_ID_MISMATCH') {
        throw "QAIRT build ID did not match $ExpectedBuildId."
    }
    if ($qairtCheckExit -notin 0, 3) {
        throw "QAIRT SDK check failed with exit code $qairtCheckExit."
    }
    if ((Get-ReportValue $qairtCheck 'expected_build_id_match') -ne 'true') {
        throw 'QAIRT SDK check did not confirm expected_build_id_match=true.'
    }
    $resolvedQairtRoot = [IO.Path]::GetFullPath((Get-ReportValue $qairtCheck 'sdk_root'))
    if ($resolvedQairtRoot -ne [IO.Path]::GetFullPath($QairtSdkRoot)) {
        throw 'QAIRT SDK check did not honor the explicit root.'
    }

    $runs = [Collections.Generic.List[object]]::new()
    for ($index = 1; $index -le $Repetitions; $index++) {
        $runs.Add((Invoke-FormalRun 'correctness' $index 'BACKGROUND_CORRECTNESS' ($index -ne 1)))
    }
    if ($RunPerformance) {
        for ($index = 1; $index -le $Repetitions; $index++) {
            $runs.Add((Invoke-FormalRun 'performance' $index 'EXCLUSIVE_BENCHMARK' $true))
        }
    }
    $summary.runs = @($runs)
    $summary.status = 'SUCCESS'
} catch {
    $summary.error = $_.Exception.Message
    throw
} finally {
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportRoot 'summary.json') -Encoding utf8
}

Write-Host "report_root=$reportRoot"
Write-Host 'status=SUCCESS'
