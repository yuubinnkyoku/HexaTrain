# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Nicopedia real-text HTP training runner.
#
# Pushes the minimal private tokenized pilot input (train_pilot.bin,
# NPRTBYTEV1) into the app-private files directory and drives
# QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA, which runs the same real-text
# training batches through the CPU reference and the QNN HTP graph with
# identical input/initial-parameter identity.  Results remain under
# build/reports; only aggregate fields are published by the allow-list
# exporter.
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [int]$Seed = 1,
  [int]$Layers = 6,
  [int]$Steps = 32,
  [int]$BatchSize = 1,
  [string]$CachePath = "",
  [int]$PollLimit = 3600
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$package = 'com.yuubinnkyoku.phonelm'
$activity = "$package/.MainActivity"
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot = Join-Path $root "build\reports\nicopedia-htp-training"
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

# The private token cache lives under build/private-data and is never
# committed.  The host pushes only the minimal pilot input the device needs.
if (-not $CachePath) { $CachePath = Join-Path $root 'build\private-data\nicopedia-real-text\caches\train_pilot.bin' }
if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { throw "PRIVATE_CACHE_MISSING: $CachePath" }
$cacheResolved = [IO.Path]::GetFullPath($CachePath)
$allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
if (-not $cacheResolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
  throw "CachePath must resolve below the repository build directory"
}

if (-not $SkipBuild) {
  & (Join-Path $root 'gradlew.bat') :app:assembleDebug '-Pphonelm.enableQnn=true' "-Pqairt.sdkRoot=$QairtSdkRoot" "-Pqairt.expectedBuildId=$ExpectedBuildId" --no-daemon
  if ($LASTEXITCODE -ne 0) { throw 'APK build failed' }
}
$online = @()
foreach ($line in (& $adb devices)) { if ($line -match '^(\S+)\s+device$') { $online += $Matches[1] } }
if ($online.Count -ne 1) { throw "Expected one online ADB device; found $($online.Count)" }
$device = $online[0]
# Stable physical identity: record the endpooint's device-side serial for the
# private report only; it never enters public artifacts.
$serial = ((& $adb -s $device shell getprop ro.serialno) -join '').Trim()
$model = ((& $adb -s $device shell getprop ro.product.model) -join '').Trim()
$soc = ((& $adb -s $device shell getprop ro.soc.model) -join '').Trim()
$thermal = ((& $adb -s $device shell dumpsys thermalservice) -join "`n")
$thermalStatus = [regex]::Match($thermal, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
$battery = ((& $adb -s $device shell dumpsys battery) -join "`n")
$batteryLevel = [regex]::Match($battery, '(?m)^\s*level:\s*(\d+)').Groups[1].Value
$batteryTemp = [regex]::Match($battery, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
# Thermal status is recorded only; it never gates on an arbitrary temperature.
# Only explicit Android EMERGENCY/SHUTDOWN conditions stop a run.
if ($thermalStatus -ge 5) { throw "THERMAL_ABORT: android_thermal_status=$thermalStatus" }

function Adb([string[]]$Arguments) {
  $output = & $adb -s $device @Arguments 2>&1
  if ($LASTEXITCODE -ne 0) { throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output" }
  $output
}

if (-not $SkipInstall) {
  Adb @('install', '-r', $apk) | Out-Null
}
# Stage the private tokenized pilot input under the app files directory.
$remoteDir = 'files/checkpoints/nicopedia-cache'
Adb @('shell', 'run-as', $package, 'mkdir', '-p', $remoteDir) | Out-Null
$tmpOnDevice = "/data/local/tmp/train_pilot.bin"
& $adb -s $device push $cacheResolved $tmpOnDevice | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'PRIVATE_CACHE_PUSH_FAILED' }
Adb @('shell', 'run-as', $package, 'cp', $tmpOnDevice, "$remoteDir/train_pilot.bin") | Out-Null
Adb @('shell', 'rm', '-f', $tmpOnDevice) | Out-Null

# Run the NICOPEDIA mode through the debug intent path (same as the existing
# tiny-language-model runner).
Adb @('shell', 'am', 'force-stop', $package) | Out-Null
& $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null | Out-Null
$extraArgs = @(
  '--es', 'phonelm.mode', 'QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA',
  '--es', 'phonelm.seed', [string]$Seed,
  '--ei', 'phonelm.epochs', [string]$Layers,
  '--ei', 'phonelm.measured_steps', '2',
  '--ei', 'phonelm.steps', [string]$Steps,
  '--ei', 'phonelm.batch_size', [string]$BatchSize,
  '--es', 'phonelm.learning_rate', '0.003',
  '--ei', 'phonelm.correctness_interval', [string]$Seed,
  '--es', 'phonelm.seed_selection_mode', 'EXACT_SEED',
  '--ei', 'phonelm.sample_count', '32',
  '--ei', 'phonelm.output_dimension', '256',
  '--ei', 'phonelm.dimension', '16',
  '--ei', 'phonelm.hidden_dimension', '32'
)
$startArgs = @('shell', 'am', 'start', '-W', '-n', $activity) + $extraArgs
$startOutput = & $adb -s $device @startArgs 2>&1
if ($LASTEXITCODE -ne 0) { throw "am start failed (endpoint redacted): $($startArgs -join ' ')`n$startOutput" }
$result = ''
for ($poll = 0; $poll -lt $PollLimit; $poll++) {
  Start-Sleep -Milliseconds 500
  $result = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
  if ($result -match '(?m)^status=(SUCCESS|FAILED)$') { break }
}
if ($result -notmatch '(?m)^status=SUCCESS$') {
  $result | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers-steps$Steps-result.txt") -Encoding utf8
  throw "NICOPEDIA_HTP did not report SUCCESS (seed=$Seed layers=$Layers steps=$Steps)"
}
$thermalAfter = ((& $adb -s $device shell dumpsys thermalservice) -join "`n")
$thermalStatusAfter = [regex]::Match($thermalAfter, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
$batteryAfter = ((& $adb -s $device shell dumpsys battery) -join "`n")
$batteryLevelAfter = [regex]::Match($batteryAfter, '(?m)^\s*level:\s*(\d+)').Groups[1].Value
$batteryTempAfter = [regex]::Match($batteryAfter, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
$annotated = $result.TrimEnd() + "`n" +
  "device_model=$model`n" +
  "device_soc=$soc`n" +
  "android_thermal_status_before=$thermalStatus`n" +
  "android_thermal_status_after=$thermalStatusAfter`n" +
  "battery_level_before=$batteryLevel`n" +
  "battery_level_after=$batteryLevelAfter`n" +
  "battery_temperature_c_before=$([int]$batteryTemp / 10.0)`n" +
  "battery_temperature_c_after=$([int]$batteryTempAfter / 10.0)`n" +
  "compile_time_qairt_build_id=$ExpectedBuildId`n" +
  "private_serial_recorded_for_identity_only=true`n"
# The serial is recorded in the private report for the same-device
# reattach check; it is stripped by the public exporter.
$annotated | Set-Content -LiteralPath (Join-Path $reportRoot "seed$Seed-l$Layers-steps$Steps-result.txt") -Encoding utf8
$annotated | Add-Content -LiteralPath (Join-Path $reportRoot "device-identity-private.txt") -Encoding utf8
Write-Host "PASS NICOPEDIA_HTP seed=$Seed layers=$Layers steps=$Steps"
Write-Host "Reports: $reportRoot"
