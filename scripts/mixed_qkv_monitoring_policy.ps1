# Pure, diagnostic-only policy. No device calls and no project-wide default change.
function Resolve-MixedQkvMonitoringState([object]$State) {
 if($null -eq $State){return 'BLOCKED_PROTECTION_TELEMETRY_UNKNOWN'}
 foreach($key in @('thermal_status','battery_health','battery_present')){if($null -eq $State.$key){return 'BLOCKED_PROTECTION_TELEMETRY_UNKNOWN'}}
 if($State.thermal_status -lt 0 -or $State.thermal_status -gt 6){return 'BLOCKED_CONTRADICTORY_TELEMETRY'}
 if($State.thermal_status -ge 4){return 'BLOCKED_OS_CRITICAL_THERMAL'}
 if($State.battery_health -ne 2 -or $State.battery_present -ne $true){return 'BLOCKED_BATTERY_FAULT'}
 if($null -ne $State.battery_level -and ($State.battery_level -lt -1 -or $State.battery_level -gt 100)){return 'BLOCKED_CONTRADICTORY_TELEMETRY'}
 if($null -ne $State.battery_level -and $State.battery_level -eq 0){return 'BLOCKED_SHUTDOWN_IMMINENT'}
 if($null -ne $State.battery_temperature_c -and ($State.battery_temperature_c -le 0 -or $State.battery_temperature_c -gt 60)){return 'BLOCKED_BATTERY_TEMPERATURE_FAULT'}
 if($null -ne $State.battery_voltage_mv -and ($State.battery_voltage_mv -le 0 -or $State.battery_voltage_mv -gt 6000)){return 'BLOCKED_CONTRADICTORY_TELEMETRY'}
 if($null -eq $State.battery_temperature_c -or $null -eq $State.battery_level -or $State.battery_level -eq -1){return 'MONITOR_PARTIAL_TELEMETRY_PROTECTION_KNOWN'}
 return 'MONITOR'
}
function Test-MixedQkvMonitoringPolicy {
 $safe=@{thermal_status=0;battery_temperature_c=39;battery_level=80;battery_health=2;battery_present=$true;battery_voltage_mv=4000}
 $cases=@(@{change=@{};expected='MONITOR'},@{change=@{battery_level=12};expected='MONITOR'},@{change=@{external_power=$false};expected='MONITOR'},@{change=@{thermal_status=2;battery_temperature_c=43};expected='MONITOR'},@{change=@{thermal_status=3};expected='MONITOR'},@{change=@{thermal_status=4};expected='BLOCKED_OS_CRITICAL_THERMAL'},@{change=@{battery_health=3};expected='BLOCKED_BATTERY_FAULT'},@{change=@{battery_level=101};expected='BLOCKED_CONTRADICTORY_TELEMETRY'},@{change=@{battery_temperature_c=$null};expected='MONITOR_PARTIAL_TELEMETRY_PROTECTION_KNOWN'},@{change=@{thermal_status=$null};expected='BLOCKED_PROTECTION_TELEMETRY_UNKNOWN'},@{change=@{battery_level=0};expected='BLOCKED_SHUTDOWN_IMMINENT'})
 foreach($case in $cases){$s=$safe.Clone();foreach($k in $case.change.Keys){$s[$k]=$case.change[$k]};if((Resolve-MixedQkvMonitoringState ([pscustomobject]$s)) -ne $case.expected){throw 'MONITORING_SELFTEST_FAILED'}}
 Write-Host 'mixed_monitoring_selftest=PASS cases=11 device_calls=0 default_policy_changed=0'
}
function Get-MixedQkvMonitoringState([string]$Adb,[string]$Device) {
 $thermal=Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell','dumpsys','thermalservice')
 $battery=Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell','dumpsys','battery')
 $state=[ordered]@{thermal_status=$null;battery_health=$null;battery_present=$null;battery_temperature_c=$null;battery_voltage_mv=$null;battery_level=$null}
 $m=[regex]::Match($thermal.Text,'(?im)^\s*(?:Current\s+)?Thermal\s+Status\s*:\s*(\d+)\s*$');if($m.Success){$state.thermal_status=[int]$m.Groups[1].Value}
 foreach($spec in @(@('health','battery_health',1),@('temperature','battery_temperature_c',10),@('voltage','battery_voltage_mv',1),@('level','battery_level',1))){$m=[regex]::Match($battery.Text,"(?im)^\s*$($spec[0])\s*:\s*(-?\d+)\s*$");if($m.Success){$state[$spec[1]]=[double]$m.Groups[1].Value/$spec[2]}}
 $m=[regex]::Match($battery.Text,'(?im)^\s*present\s*:\s*(true|false)\s*$');if($m.Success){$state.battery_present=$m.Groups[1].Value -eq 'true'}
 $state.external_power=($battery.Text -match '(?m)^\s*(?:AC|USB|Wireless) powered:\s*true\s*$')
 return [pscustomobject]$state
}
