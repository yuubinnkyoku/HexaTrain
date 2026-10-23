# Owned thermal/battery auxiliary probe for the repaired formal runner.
#
# Runs as a short-lived child process started by the asynchronous auxiliary lane so
# that thermal/battery sampling can never delay the 2 s foreground sample. The child
# reuses the unchanged monitoring helpers; it never stops, wakes or mutates anything.
param(
 [Parameter(Mandatory=$true)][string]$Adb,
 [Parameter(Mandatory=$true)][string]$Device,
 [Parameter(Mandatory=$true)][string]$OutPath
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
. (Join-Path $PSScriptRoot 'mixed_qkv_monitoring_policy.ps1')
$state=Get-MixedQkvMonitoringState -Adb $Adb -Device $Device
$classification=Resolve-MixedQkvMonitoringState $state
$record=[ordered]@{
 classification=$classification
 thermal_status=$state.thermal_status
 battery_health=$state.battery_health
 battery_present=$state.battery_present
 battery_temperature_c=$state.battery_temperature_c
 battery_voltage_mv=$state.battery_voltage_mv
 battery_level=$state.battery_level
 utc=[DateTime]::UtcNow.ToString('o')
}
if([IO.File]::Exists($OutPath)){throw 'THERMAL_PROBE_OUTPUT_EXISTS'}
$stream=[IO.File]::Open($OutPath,[IO.FileMode]::CreateNew)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($record|ConvertTo-Json));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
