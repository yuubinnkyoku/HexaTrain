# SPDX-License-Identifier: Apache-2.0
# Depth-quality diagnostics runner (Phase B).
#
# Launches the generic trainer in EXACT_SEED mode per (configuration, seed)
# with per-step trajectory recording and private checkpoint dumps enabled.
# The training itself stays on the established protocols; diagnostics are
# read-only observers of already copied-out outputs. Checkpoint binaries are
# pulled as base64 and stored under build/reports (never committed).
#
# Configurations cover the requested comparison set:
#   L18/H2 seed 2, L19/H2 seed 2 (known bad), L19/H2 seed 1 (best CSV loss),
#   L19/H2 seed 4 (generation-best CSV), L18/H2 seed 1 (best CSV loss).
[CmdletBinding()]
param(
    [string]$QairtSdkRoot = '',
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [string]$Runs = 'l18s2,l19s2,l19s1,l19s4,l18s1',
    [ValidateRange(0, 6)][int]$TrainingStabilityMode = 0,
    [switch]$NoTrajectory,
    [switch]$NoCheckpoints,
    [switch]$InstallAuditedApk,
    [switch]$NoLiveUpdate
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $QairtSdkRoot) { throw '-QairtSdkRoot is required' }
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
    -ExpectedBuildId $ExpectedBuildId
if ($TrainingStabilityMode -eq 5) { throw 'RESIDUAL_BRANCH_SCALING is unsupported on device' }

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$package = 'com.yuubinnkyoku.phonelm'
$activity = "$package/.MainActivity"
$workRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build\reports\qnn-depth-quality'))
[IO.Directory]::CreateDirectory($workRoot) | Out-Null

$checkOutput = (& (Join-Path $PSScriptRoot 'check_qairt.ps1') -SdkRoot $QairtSdkRoot `
        -ExpectedBuildId $ExpectedBuildId 2>&1) -join "`n"
$checkExit = $LASTEXITCODE
if ($checkExit -notin @(0, 3)) { throw "fixed QAIRT inventory check failed with exit $checkExit" }
if ($checkOutput -notmatch '(?m)^expected_build_id_match=true$') { throw 'fixed QAIRT build id was not confirmed' }
$apk = Join-Path $repoRoot 'app\build\outputs\apk\debug\app-debug.apk'
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk `
    -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
    -ReportPath (Join-Path $workRoot 'apk-audit.txt') | Out-Null
if (-not $?) { throw 'fixed QAIRT APK audit failed' }

function Resolve-Device {
    $online = @((& $adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } | ForEach-Object { ($_ -split '\s+')[0] })
    if ($online.Count -ne 1) { return $null }
    return $online[0]
}
$device = Resolve-Device
if (-not $device) { throw 'expected exactly one online ADB device' }
$packageList = (& $adb -s $device shell pm list packages $package) -join "`n"
if ($packageList -notmatch [regex]::Escape($package)) {
    if (-not $InstallAuditedApk) { throw 'package not installed; rerun with -InstallAuditedApk' }
    & $adb -s $device install -r $apk | Out-Null
}

function Read-DeviceState([string]$Device) {
    $battery = (& $adb -s $Device shell dumpsys battery) -join "`n"
    $thermal = (& $adb -s $Device shell dumpsys thermalservice) -join "`n"
    $temperature = [regex]::Match($battery, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
    $status = [regex]::Match($thermal, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
    if (-not $temperature -or -not $status) { return $null }
    return [pscustomobject][ordered]@{
        battery_temperature_c = [int]$temperature / 10.0
        android_thermal_status = [int]$status
    }
}

$definitions = @{
    l18s2 = [ordered]@{ T = 8; V = 32; D = 16; FFN = 32; L = 18; H = 2; Seed = 2 }
    l19s2 = [ordered]@{ T = 8; V = 32; D = 16; FFN = 32; L = 19; H = 2; Seed = 2 }
    l19s1 = [ordered]@{ T = 8; V = 32; D = 16; FFN = 32; L = 19; H = 2; Seed = 1 }
    l19s4 = [ordered]@{ T = 8; V = 32; D = 16; FFN = 32; L = 19; H = 2; Seed = 4 }
    l18s1 = [ordered]@{ T = 8; V = 32; D = 16; FFN = 32; L = 18; H = 2; Seed = 1 }
}
foreach ($runId in ($Runs -split ',')) {
    $runId = $runId.Trim()
    if (-not $definitions.Contains($runId)) { throw "unknown run: $runId" }
    $c = $definitions[$runId]
    $label = "$runId"
    if ($TrainingStabilityMode -ne 0) { $label = "$runId-stab$TrainingStabilityMode" }
    $runDir = Join-Path $workRoot $label
    [IO.Directory]::CreateDirectory($runDir) | Out-Null
    $reportPath = Join-Path $runDir 'device-report.txt'
    if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
        $cached = [IO.File]::ReadAllText($reportPath)
        if ($cached -match '(?m)^status=') { Write-Host "$label : cached report reused"; continue }
    }
    $before = Read-DeviceState $device
    if ($before -and $before.android_thermal_status -ge 5) {
        throw "Android thermal status $($before.android_thermal_status) (EMERGENCY/SHUTDOWN)"
    }
    & $adb -s $device shell am force-stop $package | Out-Null
    & $adb -s $device shell run-as $package rm -f files/device-test-result.txt | Out-Null
    & $adb -s $device shell run-as $package rm -rf "files/checkpoints/$label" | Out-Null
    $arguments = @(
        'shell', 'am', 'start', '-W', '-n', $activity,
        '--es', 'phonelm.mode', 'QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC',
        '--ei', 'phonelm.batch_size', '1',
        '--ei', 'phonelm.dimension', [string]$c['D'],
        '--ei', 'phonelm.hidden_dimension', [string]$c['FFN'],
        '--ei', 'phonelm.output_dimension', [string]$c['V'],
        '--ei', 'phonelm.steps', '320',
        '--ei', 'phonelm.warmup_steps', '0',
        '--es', 'phonelm.learning_rate', '0.003',
        '--es', 'phonelm.seed', [string]$c['Seed'],
        '--ei', 'phonelm.sample_count', [string]$c['T'],
        '--ei', 'phonelm.epochs', [string]$c['L'],
        '--ei', 'phonelm.measured_steps', [string]$c['H'],
        '--ei', 'phonelm.correctness_interval', [string]$c['Seed'],
        '--ez', 'phonelm.benchmark_mode', 'false',
        '--es', 'phonelm.seed_selection_mode', 'EXACT_SEED'
    )
    if ($TrainingStabilityMode -ne 0) {
        $stabilityNames = @{ 1 = 'WARMUP64'; 2 = 'DECAY_LINEAR'; 3 = 'ZERO_OUTPUT_PROJ_BRANCH_INIT'; 4 = 'DEPTH_SCALED_BRANCH_INIT'; 6 = 'GRADIENT_CLIP_1' }
        $arguments += @('--es', 'phonelm.training_stability_mode', $stabilityNames[$TrainingStabilityMode])
    }
    if (-not $NoTrajectory) { $arguments += @('--ez', 'phonelm.diagnostic_trajectory', 'true') }
    if (-not $NoCheckpoints) { $arguments += @('--es', 'phonelm.checkpoint_dump_dir', $label) }
    if (-not $NoLiveUpdate) { $arguments += @('--ez', 'phonelm.live_update', 'true') }
    $started = [DateTime]::UtcNow
    $launchOutput = & $adb -s $device @arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "activity launch failed: $launchOutput" }
    $report = ''
    $deadline = [DateTime]::UtcNow.AddHours(2)
    while ([DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 15
        $resolved = Resolve-Device
        if (-not $resolved) { continue }
        $device = $resolved
        $candidate = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
        if ($candidate -and $candidate -match '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)$') {
            $report = $candidate
            break
        }
    }
    if (-not $report) { throw "$label : no terminal device result" }
    $completed = [DateTime]::UtcNow
    $after = Read-DeviceState $device
    [IO.File]::WriteAllText($reportPath, $report)
    @(
        "wall_time_ms=$([math]::Round(($completed - $started).TotalMilliseconds, 1))"
        "started_utc=$($started.ToString('o'))"
        "completed_utc=$($completed.ToString('o'))"
        "battery_temperature_before_c=$(if ($before) { $before.battery_temperature_c } else { 'NOT_READ' })"
        "battery_temperature_after_c=$(if ($after) { $after.battery_temperature_c } else { 'NOT_READ' })"
        "thermal_status_before=$(if ($before) { $before.android_thermal_status } else { 'NOT_READ' })"
        "thermal_status_after=$(if ($after) { $after.android_thermal_status } else { 'NOT_READ' })"
        "qairt_build_id=$ExpectedBuildId"
    ) -join "`n" | Set-Content -LiteralPath (Join-Path $runDir 'run.txt') -Encoding utf8
    # Pull private checkpoints as base64.
    if (-not $NoCheckpoints) {
        $ckptDir = Join-Path $runDir 'checkpoints'
        [IO.Directory]::CreateDirectory($ckptDir) | Out-Null
        $listing = (& $adb -s $device shell run-as $package ls "files/checkpoints/$label" 2>$null) -join "`n"
        $pulled = 0
        foreach ($name in ($listing -split '[\r\n]+')) {
            $name = $name.Trim()
            if ($name -notmatch '^ckpt_seed\d+_step\d+\.bin$') { continue }
            $encoded = (& $adb -s $device shell run-as $package base64 "files/checkpoints/$label/$name" 2>$null) -join ''
            if (-not $encoded) { continue }
            try {
                $bytes = [Convert]::FromBase64String($encoded)
                [IO.File]::WriteAllBytes((Join-Path $ckptDir $name), $bytes)
                $pulled++
            } catch {
                $pulled += 0
            }
        }
        Write-Host "$label : checkpoints pulled=$pulled"
    }
    $status = ([regex]::Match($report, '(?m)^status=.*$').Value)
    $hashLine = ([regex]::Match($report, "(?m)^seed_$($c['Seed'])_final_parameter_canonical_hash=.*$").Value)
    Write-Host "$label : $status wall=$([math]::Round(($completed - $started).TotalSeconds, 0))s $hashLine"
}
Write-Host 'depth quality diagnostics complete'
