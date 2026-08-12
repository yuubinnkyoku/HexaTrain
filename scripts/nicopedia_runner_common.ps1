# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Shared, device-facing safety helpers for the Nicopedia runners.
#
# This file intentionally contains no public-export logic.  Device identity,
# raw result text and binary checkpoint material remain private under build/.

Set-StrictMode -Version Latest

function Stop-PhoneLmProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) { Stop-PhoneLmProcessTree -RootProcessId ([int]$child.ProcessId) }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

function Update-PhoneLmCheckpointProgress {
    param(
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$State,
        [ValidateRange(0, 1000000)][int]$CheckpointCount,
        [Parameter(Mandatory = $true)][DateTime]$NowUtc,
        [ValidateRange(1, 86400)][int]$StallSeconds
    )
    if (-not $State.Contains('Count') -or -not $State.Contains('LastProgressUtc')) { throw 'CHECKPOINT_PROGRESS_STATE_INVALID' }
    if ($CheckpointCount -gt [int]$State.Count) {
        $State.Count = $CheckpointCount
        $State.LastProgressUtc = $NowUtc
        return
    }
    if (($NowUtc - [DateTime]$State.LastProgressUtc).TotalSeconds -ge $StallSeconds) {
        throw "CHECKPOINT_PROGRESS_STALLED: no new canonical checkpoint for $StallSeconds seconds"
    }
}

function Get-PhoneLmAdbResult {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [AllowEmptyString()][string]$Device = '',
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60
    )
    $stdoutPath = Join-Path ([IO.Path]::GetTempPath()) ("phonelm-adb-" + [guid]::NewGuid().ToString('N') + '.out')
    $stderrPath = Join-Path ([IO.Path]::GetTempPath()) ("phonelm-adb-" + [guid]::NewGuid().ToString('N') + '.err')
    $process = $null
    $timedOut = $false
    try {
        $adbArguments = if ([string]::IsNullOrWhiteSpace($Device)) { $Arguments } else { @('-s', $Device) + $Arguments }
        $process = Start-Process -FilePath $Adb -ArgumentList $adbArguments `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
            -PassThru -WindowStyle Hidden -ErrorAction Stop
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $timedOut = $true
            Stop-PhoneLmProcessTree -RootProcessId $process.Id
            [void]$process.WaitForExit(5000)
        }
        # WaitForExit() without a timeout flushes async redirected streams once
        # the process is known to be terminal.  Never call it on a live client.
        if ($process.HasExited) { $process.WaitForExit() }
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { @(Get-Content -LiteralPath $stdoutPath) } else { @() }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { @(Get-Content -LiteralPath $stderrPath) } else { @() }
        $output = @($stdout) + @($stderr)
        $exitCode = if ($timedOut -or -not $process.HasExited) { 124 } else { [int]$process.ExitCode }
    } finally {
        if ($null -ne $process) { $process.Dispose() }
        Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
    $text = ($output -join "`n")
    $classification = if ($timedOut) { 'ADB_TRANSPORT_TIMEOUT' } elseif ($exitCode -eq 0) { 'SUCCESS' } elseif ($text -match '(?i)(device offline|device not found|no devices/emulators found|unauthorized|transport|cannot connect|connection (?:reset|closed)|closed by peer|adb server is out of date)') { 'ADB_TRANSPORT_FAILURE' } else { 'ADB_COMMAND_FAILURE' }
    [pscustomobject][ordered]@{
        Output = $output
        Text = $text
        ExitCode = $exitCode
        Classification = $classification
    }
}

function Invoke-PhoneLmAdb {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60,
        [switch]$AllowFailure
    )
    $result = Get-PhoneLmAdbResult -Adb $Adb -Device $Device -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
    if ($result.ExitCode -ne 0 -and -not $AllowFailure) {
        # Do not include the endpoint or command output in the exception.  The
        # caller can classify transport failures without leaking device data.
        throw "$($result.Classification): adb operation failed"
    }
    return $result
}

function Resolve-PhoneLmDevice {
    param([Parameter(Mandatory = $true)][string]$Adb)
    if (-not (Test-Path -LiteralPath $Adb -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }
    $listing = Get-PhoneLmAdbResult -Adb $Adb -Arguments @('devices', '-l')
    if ($listing.ExitCode -ne 0) { throw "$($listing.Classification): devices listing failed" }
    $listed = $listing.Output
    $endpoints = @()
    foreach ($line in $listed) {
        if ($line -match '^([^\s]+)\s+device(?:\s+(.*))?$') {
            $endpoints += [pscustomobject]@{ Endpoint = $Matches[1]; Descriptor = $Matches[2] }
        }
    }
    if ($endpoints.Count -eq 0) { throw 'ADB_NO_ONLINE_DEVICE' }
    $physical = @()
    foreach ($item in $endpoints) {
        $serialResult = Invoke-PhoneLmAdb -Adb $Adb -Device $item.Endpoint -Arguments @('shell', 'getprop', 'ro.serialno')
        $serial = ($serialResult.Text -replace '[\r\n]', '').Trim()
        if ([string]::IsNullOrWhiteSpace($serial) -or $serial -match '(?i)unknown|\bnull\b') { throw 'ADB_STABLE_IDENTITY_UNAVAILABLE' }
        $bootResult = Invoke-PhoneLmAdb -Adb $Adb -Device $item.Endpoint -Arguments @('shell', 'getprop', 'ro.boot.serialno') -AllowFailure
        $bootSerial = ($bootResult.Text -replace '[\r\n]', '').Trim()
        if ([string]::IsNullOrWhiteSpace($bootSerial)) { $bootSerial = $serial }
        if ($bootSerial -ne $serial) { throw 'ADB_STABLE_IDENTITY_MISMATCH' }
        $modelResult = Invoke-PhoneLmAdb -Adb $Adb -Device $item.Endpoint -Arguments @('shell', 'getprop', 'ro.product.model')
        $socResult = Invoke-PhoneLmAdb -Adb $Adb -Device $item.Endpoint -Arguments @('shell', 'getprop', 'ro.soc.model') -AllowFailure
        $physical += [pscustomobject][ordered]@{
            Endpoint = $item.Endpoint
            Serial = $serial
            BootSerial = $bootSerial
            Identity = $serial
            Model = ($modelResult.Text -replace '[\r\n]', '').Trim()
            Soc = ($socResult.Text -replace '[\r\n]', '').Trim()
        }
    }
    $groups = @($physical | Group-Object Identity)
    if ($groups.Count -ne 1) { throw "ADB_PHYSICAL_DEVICE_AMBIGUOUS: distinct physical devices=$($groups.Count)" }
    # Multiple endpoint aliases for one phone are accepted only after the
    # stable device-side identity above proves they are the same physical unit.
    $selected = @($groups[0].Group | Sort-Object @{ Expression = { if ($_.Endpoint -match ':') { 1 } else { 0 } } }, @{ Expression = { $_.Endpoint } })[0]
    Write-Host "ADB physical device selected (stable identity verified; endpoint redacted; aliases=$($groups[0].Count))"
    return $selected
}

function Assert-PhoneLmPhysicalDevice {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device)
    $qemu = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'getprop', 'ro.kernel.qemu')
    $hardware = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'getprop', 'ro.hardware')
    if (($qemu.Text -match '(?im)^\s*1\s*$') -or ($hardware.Text -match '(?i)goldfish|ranchu')) {
        throw 'PHYSICAL_DEVICE_REQUIRED: emulator rejected'
    }
}

function Get-PhoneLmThermalBatteryState {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [string]$Phase = 'unknown'
    )
    $thermalResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'dumpsys', 'thermalservice')
    $thermalMatch = [regex]::Match($thermalResult.Text, '(?im)^\s*(?:Current\s+)?Thermal\s+Status\s*:\s*(\d+)\s*$')
    if (-not $thermalMatch.Success) { throw "THERMAL_STATUS_UNAVAILABLE: phase=$Phase" }
    $status = [int]$thermalMatch.Groups[1].Value
    if ($status -ge 5) { throw "THERMAL_ABORT: phase=$Phase status=$status (EMERGENCY/SHUTDOWN)" }
    $batteryResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'dumpsys', 'battery')
    $batteryText = $batteryResult.Text
    $healthMatch = [regex]::Match($batteryText, '(?im)^\s*health\s*:\s*(\d+)\s*$')
    $presentMatch = [regex]::Match($batteryText, '(?im)^\s*present\s*:\s*(true|false)\s*$')
    $tempMatch = [regex]::Match($batteryText, '(?im)^\s*temperature\s*:\s*(-?\d+)\s*$')
    $voltageMatch = [regex]::Match($batteryText, '(?im)^\s*voltage\s*:\s*(-?\d+)\s*$')
    $levelMatch = [regex]::Match($batteryText, '(?im)^\s*level\s*:\s*(\d+)\s*$')
    if (-not $healthMatch.Success -or -not $presentMatch.Success -or -not $tempMatch.Success -or -not $voltageMatch.Success) {
        throw "BATTERY_STATE_UNAVAILABLE: phase=$Phase"
    }
    $health = [int]$healthMatch.Groups[1].Value
    $present = $presentMatch.Groups[1].Value.ToLowerInvariant() -eq 'true'
    $temperatureDeciC = [int]$tempMatch.Groups[1].Value
    $voltageMv = [int]$voltageMatch.Groups[1].Value
    if ($health -ne 2) { throw "BATTERY_HEALTH_FAULT: phase=$Phase health=$health" }
    if (-not $present) { throw "BATTERY_NOT_PRESENT: phase=$Phase" }
    if ($temperatureDeciC -le 0 -or $temperatureDeciC -gt 600) { throw "BATTERY_TEMPERATURE_FAULT: phase=$Phase" }
    if ($voltageMv -le 0 -or $voltageMv -gt 6000) { throw "BATTERY_VOLTAGE_FAULT: phase=$Phase" }
    [pscustomobject][ordered]@{
        thermal_status = $status
        battery_health = $health
        battery_present = $present
        battery_temperature_c = $temperatureDeciC / 10.0
        battery_temperature_deci_c = $temperatureDeciC
        battery_voltage_mv = $voltageMv
        battery_level = if ($levelMatch.Success) { [int]$levelMatch.Groups[1].Value } else { -1 }
    }
}

function Get-PhoneLmActivityEvidence {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'dumpsys', 'activity', 'activities') -AllowFailure
    if ($result.ExitCode -ne 0) {
        if ($result.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($result.Classification): activity preflight interrupted" }
        return [pscustomobject][ordered]@{ known = $false; active = $false; task_present = $false; top_package = 'UNKNOWN'; resumed_package = 'UNKNOWN'; focused_package = 'UNKNOWN' }
    }
    $text = $result.Text
    $escaped = [regex]::Escape($Package)
    $top = [regex]::Match($text, '(?im)^\s*topResumedActivity=.*?\bu\d+\s+([^/\s]+)/')
    $resumed = [regex]::Match($text, '(?im)^\s*(?:mResumedActivity|ResumedActivity):.*?\bu\d+\s+([^/\s]+)/')
    $focused = [regex]::Match($text, '(?im)^\s*m(?:CurrentFocus|FocusedWindow)=Window\{.*?\s([^/\s]+)/')
    $topPackage = if ($top.Success) { $top.Groups[1].Value } elseif ($resumed.Success) { $resumed.Groups[1].Value } elseif ($focused.Success) { $focused.Groups[1].Value } else { 'UNKNOWN' }
    $resumedPackage = if ($resumed.Success) { $resumed.Groups[1].Value } else { 'UNKNOWN' }
    $focusedPackage = if ($focused.Success) { $focused.Groups[1].Value } else { 'UNKNOWN' }
    $activityKnown = $top.Success -or $resumed.Success -or $focused.Success -or $text -match '(?im)^\s*m(?:CurrentFocus|FocusedWindow)='
    $taskPresent = $text -match "(?im)^\s*\*?\s*Task\{[^\r\n]*\bA=[^:\s]+:$escaped\b" -or $text -match "(?im)^\s*packageName=$escaped(?:$|\s)"
    $active = @($topPackage, $resumedPackage, $focusedPackage) -contains $Package
    [pscustomobject][ordered]@{
        known = [bool]$activityKnown
        active = [bool]$active
        task_present = [bool]$taskPresent
        top_package = $topPackage
        resumed_package = $resumedPackage
        focused_package = $focusedPackage
    }
}

function Get-PhoneLmRunEvidence {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $statusText = Get-PhoneLmHeadlessStatus -Adb $Adb -Device $Device -Package $Package
    $statusState = 'none'
    $statusUncertain = $false
    $statusHeartbeatAgeMs = $null
    if (-not [string]::IsNullOrWhiteSpace($statusText)) {
        try {
            $status = $statusText | ConvertFrom-Json
            if ($null -eq $status.status) { $statusUncertain = $true }
            elseif ([string]$status.status -in @('STARTING', 'RUNNING')) { $statusState = 'active' }
            elseif ([string]$status.status -in @('PASSED', 'FAILED')) { $statusState = 'terminal' }
            else { $statusUncertain = $true }
            if ($statusState -eq 'active') {
                if ($null -eq $status.last_heartbeat -or [int64]$status.last_heartbeat -le 0) { $statusUncertain = $true }
                else {
                    $statusHeartbeatAgeMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - [int64]$status.last_heartbeat
                    # A stopped heartbeat is not proof of an active run.  The
                    # process/task/service/activity evidence below still gates
                    # the decision; stale status alone is retained as an
                    # auditable inactive reason and never treated as success.
                    if ($statusHeartbeatAgeMs -gt 120000) { $statusState = 'stale' }
                }
            }
        } catch { $statusUncertain = $true }
    }
    $processPresent = $false
    $testProcessPresent = $false
    foreach ($process in @([pscustomobject]@{ name = $Package; field = 'main' }, [pscustomobject]@{ name = "$Package.test"; field = 'test' })) {
        $pidResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'pidof', $process.name) -AllowFailure
        if ($pidResult.ExitCode -ne 0 -and $pidResult.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($pidResult.Classification): process preflight interrupted" }
        if ($pidResult.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($pidResult.Text)) {
            if ($process.field -eq 'main') { $processPresent = $true } else { $testProcessPresent = $true }
        }
    }
    $serviceResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'dumpsys', 'activity', 'services', $Package) -AllowFailure
    if ($serviceResult.ExitCode -ne 0 -and $serviceResult.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($serviceResult.Classification): service preflight interrupted" }
    $serviceUncertain = $serviceResult.ExitCode -ne 0
    $serviceText = $serviceResult.Text
    $fgsPresent = $serviceResult.ExitCode -eq 0 -and $serviceText -match '(?i)isForeground\s*=\s*true|foreground\s+service|foreground=true'
    $servicePresent = $serviceResult.ExitCode -eq 0 -and $serviceText -match '(?i)ServiceRecord|serviceRecord'
    $activity = Get-PhoneLmActivityEvidence -Adb $Adb -Device $Device -Package $Package
    [pscustomobject][ordered]@{
        status_state = $statusState
        status_uncertain = [bool]$statusUncertain
        status_heartbeat_age_ms = $statusHeartbeatAgeMs
        process_present = [bool]$processPresent
        test_process_present = [bool]$testProcessPresent
        fgs_present = [bool]$fgsPresent
        service_present = [bool]$servicePresent
        service_uncertain = [bool]$serviceUncertain
        activity_known = [bool]$activity.known
        activity_active = [bool]$activity.active
        task_present = [bool]$activity.task_present
        top_package = [string]$activity.top_package
        resumed_package = [string]$activity.resumed_package
        focused_package = [string]$activity.focused_package
    }
}

function Resolve-PhoneLmRunConflict {
    param([Parameter(Mandatory = $true)]$Evidence)
    $activeReasons = [System.Collections.Generic.List[string]]::new()
    if ($Evidence.status_state -eq 'active') { [void]$activeReasons.Add('ACTIVE_HEARTBEAT') }
    if ($Evidence.test_process_present) { [void]$activeReasons.Add('ACTIVE_RUN_LOCK') }
    if ($Evidence.fgs_present) { [void]$activeReasons.Add('ACTIVE_FGS') }
    if ($Evidence.service_present -and -not $Evidence.fgs_present) { [void]$activeReasons.Add('ACTIVE_SERVICE') }
    if ($Evidence.activity_active) { [void]$activeReasons.Add('ACTIVE_ACTIVITY') }
    if ($Evidence.status_uncertain -or $Evidence.service_uncertain -or -not $Evidence.activity_known) { [void]$activeReasons.Add('RUN_STATE_UNCERTAIN') }
    if ($Evidence.process_present -and $Evidence.status_state -eq 'none') { [void]$activeReasons.Add('RUN_STATE_UNCERTAIN') }
    if ($activeReasons.Count -gt 0) {
        return [pscustomobject][ordered]@{ active = $true; reasons = @($activeReasons | Select-Object -Unique) }
    }
    $inactiveReasons = [System.Collections.Generic.List[string]]::new()
    if ($Evidence.status_state -eq 'stale') { [void]$inactiveReasons.Add('STALE_HEARTBEAT_ONLY') }
    if ($Evidence.process_present) { [void]$inactiveReasons.Add('CACHED_PROCESS_ONLY') }
    if ($Evidence.task_present) { [void]$inactiveReasons.Add('INACTIVE_TASK_ONLY') }
    if ($inactiveReasons.Count -eq 0) { [void]$inactiveReasons.Add('CLEAN_NO_ACTIVE_EVIDENCE') }
    [pscustomobject][ordered]@{ active = $false; reasons = @($inactiveReasons | Select-Object -Unique) }
}

function Assert-PhoneLmNoExistingRun {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $evidence = Get-PhoneLmRunEvidence -Adb $Adb -Device $Device -Package $Package
    $decision = Resolve-PhoneLmRunConflict -Evidence $evidence
    $reason = ($decision.reasons -join ',')
    Write-Host "clean_gate=$(-not $decision.active) reasons=$reason status=$($evidence.status_state) process=$($evidence.process_present) test_process=$($evidence.test_process_present) fgs=$($evidence.fgs_present) activity=$($evidence.activity_active) task=$($evidence.task_present) top=$($evidence.top_package)"
    if ($decision.active) { throw "RUN_ALREADY_ACTIVE: $reason" }
    # device-test-result.txt is an ephemeral legacy marker.  A terminal or
    # partial marker is safe to clear only after the live process/service and
    # headless single-flight checks above have passed; callers clear it next.
}

function Assert-PhoneLmNoExistingHeadlessRun {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    # Use the same evidence-based gate as the process/task check.  A stale
    # STARTING/RUNNING marker after an owned timeout is not itself an active
    # session; live heartbeat, lock, service, focused activity, or unknown
    # state still fails closed.
    $evidence = Get-PhoneLmRunEvidence -Adb $Adb -Device $Device -Package $Package
    $decision = Resolve-PhoneLmRunConflict -Evidence $evidence
    $reason = ($decision.reasons -join ',')
    Write-Host "headless_single_flight_gate=$(-not $decision.active) reasons=$reason status=$($evidence.status_state) heartbeat_age_ms=$($evidence.status_heartbeat_age_ms)"
    if ($decision.active) { throw "RUN_ALREADY_ACTIVE: $reason" }
}

function Assert-PhoneLmHeadlessInputFresh {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package, [Parameter(Mandatory = $true)][string]$RemoteDir)
    $listing = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'ls', '-1', $RemoteDir) -AllowFailure
    if ($listing.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($listing.Text)) { throw 'RUN_ID_REUSE: headless input directory is not empty' }
    if ($listing.ExitCode -ne 0 -and $listing.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($listing.Classification): input freshness check interrupted" }
}

function Clear-PhoneLmResultMarker {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'rm', '-f', 'files/device-test-result.txt')
    if ($result.ExitCode -ne 0) { throw 'RESULT_MARKER_CLEAR_FAILED' }
}

function Stop-PhoneLmOwnedInstrumentation {
    param([AllowNull()][System.Diagnostics.Process]$Process)
    if ($null -eq $Process) { return }
    try {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            $Process.Kill()
            $Process.WaitForExit(5000)
        }
    } catch { }
    try { $Process.Dispose() } catch { }
}

function Stop-PhoneLmOwnedHeadlessRun {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$ExpectedRunId
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedRunId)) { return $false }
    try {
        # Never force-stop on a stale/unknown status.  The atomic headless
        # status must identify this exact run and still be live first.
        $status = Get-PhoneLmHeadlessStatus -Adb $Adb -Device $Device -Package $Package
        $idPattern = '"run_id"\s*:\s*"' + [regex]::Escape($ExpectedRunId) + '"'
        if ($status -notmatch $idPattern -or $status -notmatch '"status"\s*:\s*"(STARTING|RUNNING)"') { return $false }
        foreach ($ownedPackage in @($Package, "$Package.test")) {
            $stopped = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'am', 'force-stop', $ownedPackage) -AllowFailure
            if ($stopped.ExitCode -ne 0) { return $false }
        }
        return $true
    } catch {
        # A transport failure means ownership cannot be re-established; do
        # not issue a best-effort kill against an unknown device/run.
        return $false
    }
}

function Stop-PhoneLmCompletedHeadlessProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$ExpectedRunId
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedRunId)) { return $false }
    try {
        # Android may retain the target process after am instrument reports a
        # terminal result.  Reclaim only the process belonging to this exact
        # completed run so the next preflight does not confuse an idle cache
        # with an active job.
        $status = Get-PhoneLmHeadlessStatus -Adb $Adb -Device $Device -Package $Package
        $idPattern = '"run_id"\s*:\s*"' + [regex]::Escape($ExpectedRunId) + '"'
        if ($status -notmatch $idPattern -or
            $status -notmatch '"status"\s*:\s*"(PASSED|FAILED)"') {
            return $false
        }
        foreach ($ownedPackage in @($Package, "$Package.test")) {
            $stopped = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'am', 'force-stop', $ownedPackage) -AllowFailure
            if ($stopped.ExitCode -ne 0) { return $false }
        }
        return $true
    } catch {
        return $false
    }
}

function Get-PhoneLmKeyValueMap {
    param([Parameter(Mandatory = $true)][string]$Text)
    $map = [ordered]@{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $pair = $line -split '=', 2
        if ($pair.Count -ne 2) { continue }
        if ($map.Contains($pair[0])) {
            if ($map[$pair[0]] -ne $pair[1].Trim()) { throw "REPORT_CONFLICTING_DUPLICATE_KEY: $($pair[0])" }
            continue
        }
        $map[$pair[0]] = $pair[1].Trim()
    }
    return $map
}

function Assert-PhoneLmHealthReport {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
        [int]$ExpectedStep = 0,
        [ValidateSet('training', 'eval', 'generation')][string]$Kind = 'eval'
    )
    $map = Get-PhoneLmKeyValueMap $Text
    foreach ($key in @('status', 'cpu_fallback', 'nan_detected', 'inf_detected', 'qnn_return_code_success', 'output_tensors_finite', 'graph_execute_count', 'api_trace_graph_execute_attempt_count', 'api_trace_graph_execute_success_count', 'api_trace_graph_execute_failure_count', 'api_trace_last_qnn_result', 'api_trace_effective_result', 'api_trace_cpu_backend_initialized', 'api_trace_fallback_attempted', 'api_trace_fallback_succeeded')) {
        if (-not $map.Contains($key)) { throw "REPORT_FIELD_MISSING: $Kind $key" }
    }
    if ($map.status -ne 'SUCCESS' -or $map.cpu_fallback -ne 'false' -or $map.nan_detected -ne 'false' -or $map.inf_detected -ne 'false' -or $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true') { throw "REPORT_HEALTH_REJECTED: $Kind" }
    $attempts = 0L; $successes = 0L; $failures = 0L; $graphs = 0L
    if (-not [long]::TryParse($map.api_trace_graph_execute_attempt_count, [ref]$attempts) -or $attempts -le 0) { throw "REPORT_QNN_ATTEMPT_INVALID: $Kind" }
    if (-not [long]::TryParse($map.api_trace_graph_execute_success_count, [ref]$successes) -or $successes -ne $attempts) { throw "REPORT_QNN_SUCCESS_INVALID: $Kind" }
    if (-not [long]::TryParse($map.api_trace_graph_execute_failure_count, [ref]$failures) -or $failures -ne 0) { throw "REPORT_QNN_FAILURE: $Kind" }
    if ($map.api_trace_last_qnn_result -ne '0' -or $map.api_trace_effective_result -ne '0' -or $map.api_trace_cpu_backend_initialized -ne 'false' -or $map.api_trace_fallback_attempted -ne 'false' -or $map.api_trace_fallback_succeeded -ne 'false') { throw "REPORT_QNN_HEALTH_REJECTED: $Kind" }
    if (-not [long]::TryParse($map.graph_execute_count, [ref]$graphs) -or $graphs -le 0) { throw "REPORT_GRAPH_COUNT_INVALID: $Kind" }
    if (-not $map.Contains('api_trace_backend_requested') -or $map.api_trace_backend_requested -ne 'HTP') { throw "REPORT_BACKEND_INVALID: $Kind" }
    if (-not $map.Contains('backend_build_id_match') -or $map.backend_build_id_match -ne 'true') { throw "REPORT_BUILD_ID_UNVERIFIED: $Kind" }
    $runtimeBuild = if ($map.Contains('api_trace_runtime_backend_build_id')) { $map.api_trace_runtime_backend_build_id } elseif ($map.Contains('runtime_backend_build_id')) { $map.runtime_backend_build_id } else { '' }
    if ($runtimeBuild -ne $ExpectedBuildId -and $runtimeBuild -ne "v$ExpectedBuildId") { throw "REPORT_RUNTIME_BUILD_ID_MISMATCH: $Kind" }
    if ($ExpectedStep -gt 0) {
        $stepKey = if ($Kind -eq 'training') { 'completed_steps' } else { 'checkpoint_step' }
        if (-not $map.Contains($stepKey) -or [int]$map[$stepKey] -ne $ExpectedStep) { throw "REPORT_STEP_MISMATCH: $Kind" }
    }
    if ($Kind -eq 'training') {
        foreach ($key in @('all_steps_finite', 'final_finite', 'checkpoint_written', 'final_parameter_hash')) { if (-not $map.Contains($key)) { throw "REPORT_FIELD_MISSING: training $key" } }
        if ($map.all_steps_finite -ne 'true' -or $map.final_finite -ne 'true' -or $map.checkpoint_written -ne 'true' -or $map.final_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw 'REPORT_TRAINING_HEALTH_REJECTED' }
    } elseif ($Kind -eq 'eval') {
        foreach ($key in @('validation_nll', 'development_nll', 'validation_nonfinite_chunks', 'development_nonfinite_chunks', 'checkpoint_format', 'checkpoint_finite', 'checkpoint_parameter_hash')) { if (-not $map.Contains($key)) { throw "REPORT_FIELD_MISSING: eval $key" } }
        if ($map.checkpoint_format -notin @('NPRTCKPTV1', 'NPRTCKPTV2') -or $map.checkpoint_finite -ne 'true' -or $map.validation_nonfinite_chunks -ne '0' -or $map.development_nonfinite_chunks -ne '0' -or $map.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw 'REPORT_EVAL_HEALTH_REJECTED' }
    }
    return $map
}

function Get-PhoneLmCheckpointHeaders {
    param([Parameter(Mandatory = $true)][string]$Path)
    $bytes = [IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 43) { throw 'CHECKPOINT_TOO_SMALL' }
    $magic = [Text.Encoding]::ASCII.GetString($bytes, 0, 11)
    if ($magic -notin @("NPRTCKPTV1`n", "NPRTCKPTV2`n")) { throw 'CHECKPOINT_MAGIC_MISMATCH' }
    function U32([byte[]]$b, [int]$o) { return [uint32](([uint64]$b[$o] * 16777216) + ([uint64]$b[$o + 1] * 65536) + ([uint64]$b[$o + 2] * 256) + [uint64]$b[$o + 3]) }
    [pscustomobject][ordered]@{ Magic = $magic.Trim(); Vocabulary = U32 $bytes 11; Tokens = U32 $bytes 15; Dimension = U32 $bytes 19; FeedForward = U32 $bytes 23; Layers = U32 $bytes 27; Heads = U32 $bytes 31; Seed = U32 $bytes 35; Step = U32 $bytes 39 }
}

function Receive-PhoneLmBinary {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [int64]$MinimumBytes = 1024,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 120
    )
    $directory = Split-Path -Parent $LocalPath
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $incoming = "$LocalPath.incoming.$([guid]::NewGuid().ToString('N'))"
    # adb exec-out emits raw bytes, but routing it through cmd.exe /c mangles
    # the quoted redirect target (cmd strips the outer quote pair, so the
    # destination path ends up unterminated). Spawn adb directly and copy its
    # stdout byte stream to disk; never route binary data through PowerShell
    # string pipelines, which would corrupt the checkpoint.
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Adb
    $psi.Arguments = '-s "{0}" exec-out run-as "{1}" cat "{2}"' -f $Device, $Package, $RemotePath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $pull = [System.Diagnostics.Process]::new()
    $pull.StartInfo = $psi
    $stream = $null
    $stderr = ''
    try {
        [void]$pull.Start()
        $stream = [IO.File]::Create($incoming)
        # Drain stdout and stderr concurrently so neither pipe can fill and
        # deadlock the child; stdout is copied byte-for-byte to the file.
        $copy = $pull.StandardOutput.BaseStream.CopyToAsync($stream)
        $stderrTask = $pull.StandardError.ReadToEndAsync()
        $pullTimedOut = -not $pull.WaitForExit($TimeoutSeconds * 1000)
        if ($pullTimedOut) {
            Stop-PhoneLmProcessTree -RootProcessId $pull.Id
            [void]$pull.WaitForExit(5000)
        }
        $copy.GetAwaiter().GetResult() | Out-Null
        $stream.Dispose(); $stream = $null
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($pull.HasExited) { $pull.WaitForExit() }
        $pullExitCode = if ($pullTimedOut -or -not $pull.HasExited) { 124 } else { [int]$pull.ExitCode }
        if ($pullTimedOut) { throw 'ADB_TRANSPORT_TIMEOUT: binary pull exceeded its command deadline' }
        if ($pullExitCode -ne 0) {
            $transport = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('get-state') -AllowFailure
            if ($transport.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($transport.Classification): binary pull interrupted" }
            throw "CHECKPOINT_PULL_FAILED: adb exit=$pullExitCode stderr=$($stderr.Trim())"
        }
        if (-not (Test-Path -LiteralPath $incoming -PathType Leaf)) { throw 'CHECKPOINT_PULL_MISSING' }
        $size = (Get-Item -LiteralPath $incoming).Length
        if ($size -lt $MinimumBytes) { throw "CHECKPOINT_PULL_TOO_SMALL: $size" }
        $hash = (Get-FileHash -LiteralPath $incoming -Algorithm SHA256).Hash.ToLowerInvariant()
        if (Test-Path -LiteralPath $LocalPath -PathType Leaf) {
            $old = Get-Item -LiteralPath $LocalPath
            $oldHash = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($old.Length -ne $size -or $oldHash -ne $hash) { throw "CHECKPOINT_STALE_TARGET_MISMATCH: $([IO.Path]::GetFileName($LocalPath))" }
            Remove-Item -LiteralPath $incoming -Force
        } else {
            Move-Item -LiteralPath $incoming -Destination $LocalPath
        }
        [pscustomobject][ordered]@{ Path = $LocalPath; Size = $size; Sha256 = $hash }
    } finally {
        if ($stream) { $stream.Dispose() }
        $pull.Dispose()
        if (Test-Path -LiteralPath $incoming -PathType Leaf) { Remove-Item -LiteralPath $incoming -Force -ErrorAction SilentlyContinue }
    }
}

function Get-PhoneLmCheckpointName {
    param(
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Layers,
        [Parameter(Mandatory = $true)][int]$Tokens,
        [Parameter(Mandatory = $false)][int]$Dimension = 16,
        [Parameter(Mandatory = $false)][int]$FeedForwardDimension = 32,
        [Parameter(Mandatory = $true)][int]$Step
    )
    # The canonical anchor retains its legacy name. Every non-anchor model
    # identity carries T/D/FFN tags, so resume and evaluation cannot silently
    # select an incompatible checkpoint.
    if ($Tokens -eq 32 -and $Dimension -eq 16 -and $FeedForwardDimension -eq 32) {
        return "htp-seed$Seed-l$Layers-step$Step.ckpt"
    }
    return "htp-seed$Seed-l$Layers-t$Tokens-d$Dimension-f$FeedForwardDimension-step$Step.ckpt"
}

function Get-PhoneLmCheckpointNames {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package, [Parameter(Mandatory = $true)][string]$RemoteDir)
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'ls', '-1', $RemoteDir)
    return @($result.Text -split "`r?`n" | Where-Object { $_ -match '^htp-seed\d+-l\d+(-t\d+-d\d+-f\d+)?-step\d+\.ckpt$' } | Sort-Object -Unique)
}

function Assert-PhoneLmInstalledApkMatches {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$LocalApk
    )
    if (-not (Test-Path -LiteralPath $LocalApk -PathType Leaf)) { throw 'APK_PROVENANCE_LOCAL_MISSING' }
    $pathResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'pm', 'path', $Package)
    $paths = @($pathResult.Output | Where-Object { $_ -match '^package:(/data/app/[^\s]+/base\.apk)$' })
    if ($paths.Count -ne 1) { throw 'APK_PROVENANCE_INSTALLED_PATH_INVALID' }
    [void]($paths[0] -match '^package:(/data/app/[^\s]+/base\.apk)$')
    $remotePath = $Matches[1]
    $hashResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'sha256sum', $remotePath)
    if ($hashResult.Text -notmatch '(?i)^([0-9a-f]{64})\s+') { throw 'APK_PROVENANCE_INSTALLED_HASH_INVALID' }
    $installedHash = $Matches[1].ToLowerInvariant()
    $localHash = (Get-FileHash -LiteralPath $LocalApk -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($installedHash -ne $localHash) { throw 'APK_PROVENANCE_INSTALLED_HASH_MISMATCH' }
}

function Wait-PhoneLmResult {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [int]$PollLimit = 7200,
        [int]$PollSeconds = 5,
        [string]$RemoteResult = 'files/device-test-result.txt',
        [string]$ProgressLabel = 'run',
        [scriptblock]$StatusProgressAction,
        [string]$PartialPath = ''
    )
    if ($PollLimit -lt 1 -or $PollSeconds -lt 1) { throw 'POLL_CONFIGURATION_INVALID' }
    $started = [DateTimeOffset]::UtcNow
    $last = ''
    for ($poll = 0; $poll -lt $PollLimit; $poll++) {
        if ($poll -gt 0) { Start-Sleep -Seconds $PollSeconds }
        $read = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'cat', $RemoteResult) -AllowFailure
        if ($read.ExitCode -ne 0 -and $read.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($read.Classification): result polling interrupted" }
        if ($read.ExitCode -eq 0) { $last = $read.Text }
        $elapsed = [int]([DateTimeOffset]::UtcNow - $started).TotalSeconds
        if ($StatusProgressAction) { & $StatusProgressAction $elapsed $last }
        # A result is terminal only with an explicit status line.  Prefixes,
        # partial reports and progress fragments never terminate the wait.
        if ($last -match '(?m)^status=(SUCCESS|FAILED)\s*$') { return [pscustomobject]@{ Text = $last; ElapsedSeconds = $elapsed; Terminal = $true } }
    }
    if ($PartialPath -ne '') { $last | Set-Content -LiteralPath $PartialPath -Encoding utf8 }
    throw "RUN_TIMEOUT_PARTIAL_RESULT: label=$ProgressLabel elapsed_seconds=$([int]([DateTimeOffset]::UtcNow - $started).TotalSeconds)"
}

function Start-PhoneLmHeadlessInstrumentation {
    param(
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][string]$Class,
        [Parameter(Mandatory = $true)][string]$Suite,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][hashtable]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )
    $runner = "$Package.test/androidx.test.runner.AndroidJUnitRunner"
    $argv = @('-s', $Device, 'shell', 'am', 'instrument', '-w', '-r', '-e', 'class', $Class,
        '-e', 'suite', $Suite, '-e', 'testMode', 'BACKGROUND_CORRECTNESS',
        '-e', 'liveUpdateNotification', 'false', '-e', 'runId', $RunId)
    foreach ($entry in $Arguments.GetEnumerator()) {
        $argv += @('-e', [string]$entry.Key, [string]$entry.Value)
    }
    $argv += $runner
    # Start asynchronously.  The instrumentation owns Android lifecycle and
    # writes an atomic status/heartbeat record; no Activity intent is used.
    return Start-Process -FilePath $Adb -ArgumentList $argv -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru -WindowStyle Hidden
}

function Get-PhoneLmHeadlessStatus {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'cat', 'files/headless/status.json') -AllowFailure
    if ($result.ExitCode -ne 0) {
        if ($result.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($result.Classification): headless status polling interrupted" }
        return ''
    }
    return $result.Text.Trim()
}

function Get-PhoneLmTopPackage {
    param([Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device)
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'dumpsys', 'activity', 'activities') -AllowFailure
    if ($result.ExitCode -ne 0) { if ($result.Classification -in @('ADB_TRANSPORT_FAILURE', 'ADB_TRANSPORT_TIMEOUT')) { throw "$($result.Classification): focus sampling interrupted" }; return 'UNKNOWN' }
    foreach ($pattern in @(
        '(?im)^\s*topResumedActivity=.*?\bu\d+\s+([^/\s]+)/',
        '(?im)^\s*(?:mResumedActivity|ResumedActivity):.*?\bu\d+\s+([^/\s]+)/',
        '(?im)^\s*m(?:CurrentFocus|FocusedWindow)=Window\{.*?\s([^/\s]+)/'
    )) {
        $match = [regex]::Match($result.Text, $pattern)
        if ($match.Success) { return $match.Groups[1].Value }
    }
    return 'UNKNOWN'
}

function Wait-PhoneLmHeadlessStatus {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$Adb,
        [Parameter(Mandatory = $true)][string]$Device,
        [Parameter(Mandatory = $true)][string]$Package,
        [int]$PollLimit = 7200,
        [int]$PollSeconds = 2,
        [int]$ProgressEverySeconds = 30,
        [scriptblock]$StatusProgressAction,
        [scriptblock]$ConditionAction,
        [scriptblock]$FocusAction,
        [string]$Label = 'headless',
        [string]$PartialPath = '',
        [string]$ExpectedRunId = ''
    )
    $started = [DateTimeOffset]::UtcNow
    $lastStatus = ''
    $lastProgress = -1
    $lastCondition = -30
    $seenExpectedStatus = $false
    $transportFaults = 0
    try {
        for ($poll = 0; $poll -lt $PollLimit; $poll++) {
            $Process.Refresh()
            $elapsed = [int]([DateTimeOffset]::UtcNow - $started).TotalSeconds
            try {
                $lastStatus = Get-PhoneLmHeadlessStatus -Adb $Adb -Device $Device -Package $Package
                if ($lastStatus -ne '' -and $ExpectedRunId -ne '') {
                    $expectedPattern = '"run_id"\s*:\s*"' + [regex]::Escape($ExpectedRunId) + '"'
                    if ($lastStatus -match $expectedPattern) {
                        $seenExpectedStatus = $true
                    } else {
                        # Immediately after launch the atomic status file may
                        # still contain a terminal record from the previous run.
                        # Ignore that record only until this run is first seen;
                        # a foreign live status, or any identity change after the
                        # expected run appeared, is always fatal.
                        $foreignLive = $lastStatus -match '"status"\s*:\s*"(STARTING|RUNNING)"'
                        $foreignHeartbeatStale = $false
                        if ($foreignLive -and $lastStatus -match '"last_heartbeat"\s*:\s*(\d+)') {
                            $foreignHeartbeatStale = ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - [int64]$Matches[1]) -gt 120000
                        }
                        if ($seenExpectedStatus -or ($foreignLive -and -not $foreignHeartbeatStale)) {
                            throw 'HEADLESS_STATUS_IDENTITY_MISMATCH'
                        }
                        $lastStatus = ''
                    }
                }
                if ($FocusAction) { & $FocusAction $elapsed }
                if ($ConditionAction -and ($elapsed -ge $lastCondition + 30)) { $lastCondition = $elapsed; & $ConditionAction $elapsed }
                if ($StatusProgressAction -and ($elapsed -eq 0 -or $elapsed -ge $lastProgress + $ProgressEverySeconds)) { $lastProgress = $elapsed; & $StatusProgressAction $elapsed $lastStatus }
                # A fully successful poll round clears the fault streak.
                $transportFaults = 0
            } catch {
                # A single adb call may exceed its 60 s deadline while the
                # device itself is healthy (the segment keeps training); the
                # previous run aborted mid-segment on exactly that transient.
                # Tolerate a bounded streak of transport faults, then fail
                # closed exactly as before so persistent transport loss still
                # stops the segment. Non-transport errors (focus takeover,
                # thermal abort, identity/heartbeat mismatch, checkpoint
                # stall) are never retried.
                if ($_.Exception.Message -notmatch 'ADB_TRANSPORT_(FAILURE|TIMEOUT)') { throw }
                $transportFaults++
                if ($transportFaults -ge 5) { throw }
                $lastStatus = ''
                Start-Sleep -Seconds 10
                continue
            }
            $terminal = $lastStatus -match '"status"\s*:\s*"(PASSED|FAILED)"'
            if (-not $terminal -and $lastStatus -match '"last_heartbeat"\s*:\s*(\d+)') {
                $heartbeatMs = [int64]$Matches[1]
                $nowMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
                if ($heartbeatMs -le 0 -or $nowMs - $heartbeatMs -gt 120000) { throw 'HEADLESS_HEARTBEAT_STALE' }
            }
            if ($terminal -and $Process.HasExited) { break }
            if ($Process.HasExited -and -not $terminal) { break }
            if ($elapsed -ge ($PollLimit * $PollSeconds)) { break }
            Start-Sleep -Seconds $PollSeconds
        }
    } catch {
        [void](Stop-PhoneLmOwnedHeadlessRun -Adb $Adb -Device $Device -Package $Package -ExpectedRunId $ExpectedRunId)
        $Process.Refresh()
        if (-not $Process.HasExited) { $Process.Kill() }
        throw
    }
    $Process.Refresh()
    $elapsedFinal = [int]([DateTimeOffset]::UtcNow - $started).TotalSeconds
    $terminalFinal = $lastStatus -match '"status"\s*:\s*"(PASSED|FAILED)"'
    if ($Process.HasExited -and -not $terminalFinal) {
        # `am instrument` can close its host stdout immediately after the
        # test writes the atomic terminal status.  A poll that raced just
        # before that rename still holds RUNNING here, so perform a bounded
        # final re-read before classifying a completed test as timeout.
        for ($grace = 0; $grace -lt 10 -and -not $terminalFinal; $grace++) {
            Start-Sleep -Seconds 1
            $freshStatus = Get-PhoneLmHeadlessStatus -Adb $Adb -Device $Device -Package $Package
            if ($freshStatus -ne '' -and $ExpectedRunId -ne '' -and
                $freshStatus -notmatch ('"run_id"\s*:\s*"' + [regex]::Escape($ExpectedRunId) + '"')) {
                throw 'HEADLESS_STATUS_IDENTITY_MISMATCH'
            }
            if ($freshStatus -ne '') { $lastStatus = $freshStatus }
            $terminalFinal = $lastStatus -match '"status"\s*:\s*"(PASSED|FAILED)"'
        }
    }
    if (-not $terminalFinal -or -not $Process.HasExited) {
        [void](Stop-PhoneLmOwnedHeadlessRun -Adb $Adb -Device $Device -Package $Package -ExpectedRunId $ExpectedRunId)
        if (-not $Process.HasExited) { $Process.Kill() }
        if ($PartialPath -ne '') { $lastStatus | Set-Content -LiteralPath $PartialPath -Encoding utf8 }
        throw "HEADLESS_TIMEOUT_PARTIAL_STATUS: label=$Label elapsed_seconds=$elapsedFinal"
    }
    [pscustomobject][ordered]@{ StatusJson = $lastStatus; ElapsedSeconds = $elapsedFinal; ProcessExitCode = $Process.ExitCode }
}

function Get-PhoneLmHeadlessReport {
    param([Parameter(Mandatory = $true)][string]$StatusJson, [Parameter(Mandatory = $true)][string]$Adb, [Parameter(Mandatory = $true)][string]$Device, [Parameter(Mandatory = $true)][string]$Package)
    $match = [regex]::Match($StatusJson, '"report_relative_path"\s*:\s*"([^"]+)"')
    if (-not $match.Success -or [string]::IsNullOrWhiteSpace($match.Groups[1].Value)) { throw 'HEADLESS_REPORT_PATH_MISSING' }
    $relative = $match.Groups[1].Value
    if ($relative -notmatch '^headless/reports/[A-Za-z0-9._-]+\.txt$') { throw 'HEADLESS_REPORT_PATH_INVALID' }
    $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'run-as', $Package, 'cat', "files/$relative")
    return $result.Text
}

function Assert-PhoneLmHeadlessNoActivity {
    param([Parameter(Mandatory = $true)][string]$Text)
    foreach ($key in @('activity_create_count', 'activity_resume_count', 'phonelm_became_top_activity_count', 'focus_takeover_count')) {
        $match = [regex]::Match($Text, "(?m)^$([regex]::Escape($key))=(\d+)$")
        if (-not $match.Success -or [int]$match.Groups[1].Value -ne 0) { throw "HEADLESS_ACTIVITY_INVARIANT_FAILED: $key" }
    }
}
