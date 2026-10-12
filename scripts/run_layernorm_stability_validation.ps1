# SPDX-License-Identifier: Apache-2.0
param(
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [switch]$SkipBuild,
  [switch]$SkipInstall
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if (& git -C $root status --porcelain) { throw 'VALIDATION_SOURCE_MUST_BE_COMMITTED' }
$source=(& git -C $root rev-parse HEAD).Trim()
$plan=Get-Content (Join-Path $root 'build/reports/g1-long-linear-200k-primary/run-plan.json') -Raw | ConvertFrom-Json
if ($plan.status -ne 'UNSTABLE') { throw 'PROTECTED_ORIGINAL_STATUS_MISMATCH' }
$protected=Get-Content (Join-Path $root 'build/reports/g1-long-linear-forensics/final-protected-identities.json') -Raw | ConvertFrom-Json
foreach($p in $protected.protected_sha256.psobject.Properties) {
  if ((Get-FileHash -LiteralPath (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value) { throw 'PROTECTED_IDENTITY_MISMATCH' }
}
$hostFree=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($root)).AvailableFreeSpace
if ($hostFree -lt 20GB) { throw 'HOST_STORAGE_HEADROOM_LOW' }
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
$device=Resolve-PhoneLmDevice -Adb $adb
if ($device.Serial -ne $plan.device_serial_private) { throw 'PHYSICAL_DEVICE_IDENTITY_MISMATCH' }
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device.Endpoint
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
$df=(Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','df','-k','/data/user/0')).Text
$line=@($df -split "`n" | Where-Object {$_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%'})
if ($line.Count -ne 1) { throw 'DEVICE_STORAGE_STATUS_UNKNOWN' }
$deviceFree=[long](($line[0] -split '\s+')[3])*1024L
if ($deviceFree -lt 12GB) { throw 'DEVICE_STORAGE_HEADROOM_LOW' }
$id='lnvalidation-'+[guid]::NewGuid().ToString('N')
$directory=Join-Path $root "build/reports/g1-layernorm-validation/$id"
[IO.Directory]::CreateDirectory($directory)|Out-Null
$ledger=[ordered]@{schema='LAYER_NORM_VALIDATION_V1';source_sha=$source;run_id_private=$id;
 qairt_build_id=$ExpectedBuildId;device_identity_private=$device.Serial;host_free_bytes=$hostFree;device_free_bytes=$deviceFree;
 protected_identity_verified=$true;started_utc=[DateTime]::UtcNow.ToString('o');status='RUNNING';failure=$null}
function Save-Ledger { [IO.File]::WriteAllText((Join-Path $directory 'validation-plan.json'),($ledger|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false)) }
Save-Ledger
try {
 & (Join-Path $PSScriptRoot 'run_qnn_headless_tests.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
   -Suite layernorm-stability -RunId $id -SkipBuild:$SkipBuild -SkipInstall:$SkipInstall `
   -TestMode BACKGROUND_CORRECTNESS *> (Join-Path $directory 'runner.txt')
 $ledger.status='PASS'
} catch { $ledger.status='FAIL';$ledger.failure=$_.Exception.Message
} finally {
 $ledger['completed_utc']=[DateTime]::UtcNow.ToString('o')
 $report=Join-Path $root "build/reports/qnn-headless/$id/device-report.txt"
 if(Test-Path $report){Copy-Item -LiteralPath $report -Destination (Join-Path $directory 'device-report.txt')}
 Save-Ledger
}
Write-Host "layernorm_validation_status=$($ledger.status) private_report=$directory"
if($ledger.status -ne 'PASS'){throw 'LAYERNORM_CORRECTNESS_NOT_ESTABLISHED_STRESS_TEST_BLOCKED'}
