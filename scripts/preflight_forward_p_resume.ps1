# Read-only, single-attempt safety preflight. Never wakes, reconnects, installs,
# stops a process or changes foreground/power policy. Failure ends device work.
param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/g1-forward-p-numerical-attribution-safe-resume'
$record=[ordered]@{status='BLOCKED';reason=$null;read_only=$true;device_executes=0;device_operations_after_stop=0;gates=[ordered]@{}}
try {
 Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
 $qnn=Get-Content (Join-Path $out 'qnn.txt') -Raw
 if($qnn -notmatch '===== verify.ps1 -Profile Qnn =====' -or $qnn -notmatch 'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$'){throw 'QNN_PROFILE_NOT_COMPLETE'}
 $analysis=Get-Content (Join-Path $out 'analysis-identity-private.json') -Raw|ConvertFrom-Json
 if($analysis.tests.probability_attribution_test.exit_code -ne 0 -or $analysis.tests.probability_reference_test.exit_code -ne 4){throw 'HOST_ATTRIBUTION_NOT_COMPLETE'}
 $record.gates.host_analysis='PASS';$record.gates.sdk_apk_host='PASS'
 if([IO.DriveInfo]::new([IO.Path]::GetPathRoot($root)).AvailableFreeSpace -lt 20GB){throw 'HOST_STORAGE_HEADROOM_LOW'}
 $record.gates.host_storage='PASS'
 $build=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw|ConvertFrom-Json
 $manifest=Get-Content (Join-Path $HexagonSdkRoot 'hexagon_sdk.json') -Raw|ConvertFrom-Json
 $tools=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
 if($manifest.root.id.version -ne '6.6.0.0' -or $tools.Count -ne 1 -or $tools[0].build -ne '19.0.07'){throw 'HEXAGON_IDENTITY_MISMATCH'}
 if((Get-Content (Join-Path $NdkRoot 'source.properties') -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'NDK_IDENTITY_MISMATCH'}
 if((Get-FileHash (Join-Path $HexagonSdkRoot 'tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++.exe')).Hash.ToLowerInvariant() -ne $build.hexagon_compiler_sha256 -or (Get-FileHash (Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe')).Hash.ToLowerInvariant() -ne $build.ndk_compiler_sha256){throw 'COMPILER_IDENTITY_MISMATCH'}
 $record.gates.compiler='PASS'
 foreach($p in $build.source_hashes.psobject.Properties){if((Get-FileHash (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'BUILD_SOURCE_MISMATCH'}}
 foreach($p in $build.binary_hashes.psobject.Properties){if((Get-FileHash (Join-Path $root "build/fp32-custom/$($p.Name)")).Hash.ToLowerInvariant() -ne $p.Value){throw 'BUILD_BINARY_MISMATCH'}}
 $record.gates.package_probe='PASS';$record.build=$build
 $protected=Get-Content (Join-Path $root 'build/reports/g1-long-linear-forensics/final-protected-identities.json') -Raw|ConvertFrom-Json
 foreach($p in $protected.protected_sha256.psobject.Properties){if((Get-FileHash -LiteralPath (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'PROTECTED_IDENTITY_MISMATCH'}}
 $record.gates.protected='PASS'
 $plan=Get-Content (Join-Path $root 'build/reports/g1-long-linear-200k-primary/run-plan.json') -Raw|ConvertFrom-Json
 if($plan.status -ne 'UNSTABLE'){throw 'PROTECTED_STATUS_MISMATCH'}
 $adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
 $device=Resolve-PhoneLmDevice -Adb $adb
 $record.device_private=$device
 if($device.Serial -ne $plan.device_serial_private){throw 'PHYSICAL_IDENTITY_MISMATCH'}
 Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device.Endpoint
 $record.gates.same_physical_device='PASS'
 Assert-PhoneLmNoExistingRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
 Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
 if((Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','ps','-A')).Text -match '(?m)\b(?:fp32_probe|layernorm_probe|one_layer_probe)\s*$'){throw 'EXISTING_CUSTOM_PROBE'}
 $record.gates.active_run_absence='PASS'
 $power=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','dumpsys','power')
 $record.power_private=$power.Text
 if($power.Text -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
 $record.gates.awake='PASS'
 $activity=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','dumpsys','activity','activities')
 $record.activity_private=$activity.Text
 if($activity.Text -notmatch '(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/'){throw 'TOP_RESUMED_UNKNOWN'}
 if($Matches[1] -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_ALREADY_FOREGROUND'}
 $record.gates.top_resumed='PASS'
 $thermal=Get-PhoneLmThermalBatteryState -Adb $adb -Device $device.Endpoint -Phase 'forward-p-resume'
 if($thermal.thermal_status -gt 1 -or $thermal.battery_temperature_c -gt 40 -or $thermal.battery_level -lt 30){throw 'BOUNDED_DIAGNOSTIC_POWER_THERMAL_UNSAFE'}
 $record.thermal_private=$thermal
 $battery=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','dumpsys','battery')
 $record.battery_private=$battery.Text
 if($battery.Text -notmatch '(?m)^\s*(?:AC|USB|Wireless) powered:\s*true\s*$'){throw 'EXTERNAL_POWER_UNKNOWN_OR_ABSENT'}
 $record.gates.power_thermal='PASS'
 $df=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','df','-k','/data/user/0')
 $line=@($df.Text -split "`n"|Where-Object {$_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%'})
 if($line.Count -ne 1 -or [long](($line[0] -split '\s+')[3])*1024L -lt 12GB){throw 'DEVICE_STORAGE_HEADROOM_LOW_OR_UNKNOWN'}
 $record.gates.device_storage='PASS'
 $package=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','pm','path','com.yuubinnkyoku.phonelm')
 if($package.Text.Trim() -notmatch '^package:(/[^\r\n]+/base\.apk)$'){throw 'INSTALLED_APK_UNKNOWN'}
 $installed=Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','sha256sum',$Matches[1])
 $apkHash=(Get-FileHash (Join-Path $root 'app/build/outputs/apk/debug/app-debug.apk')).Hash.ToLowerInvariant()
 if($installed.Text -notmatch ('^'+$apkHash+'\s')){throw 'INSTALLED_APK_IDENTITY_MISMATCH_OR_UNKNOWN'}
 $record.gates.installed_apk='PASS'
 $record.status='PASS'
} catch {$record.reason=$_.Exception.Message;Write-Host "safe_resume=BLOCKED reason=$($record.reason)"} finally {
 $record|ConvertTo-Json -Depth 30|Out-File (Join-Path $out 'device-preflight-private.json') -Encoding utf8
}
if($record.status -ne 'PASS'){exit 2}
Write-Host 'safe_resume=PASS read_only=1 immediate_runner_recheck_required=1'
