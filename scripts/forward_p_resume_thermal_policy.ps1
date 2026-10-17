# Pure policy for the bounded research resume. No device operations.
# Human waiver affects the study's 40C / status 1 / battery 30% thresholds only.
function Resolve-ForwardPResumeThermalState([object]$State,[bool]$ExplicitWaiver) {
 if($null -eq $State){return 'BLOCKED'}
 foreach($key in @('thermal_status','battery_temperature_c','battery_level','battery_health','battery_present')){
  if($null -eq $State.$key){return 'BLOCKED'}
 }
 if($State.thermal_status -lt 0 -or $State.thermal_status -ge 5 -or
    $State.battery_temperature_c -le 0 -or $State.battery_temperature_c -gt 60 -or
    $State.battery_level -lt 0 -or $State.battery_level -gt 100 -or
    $State.battery_health -ne 2 -or $State.battery_present -ne $true){return 'BLOCKED'}
 if($State.thermal_status -le 1 -and $State.battery_temperature_c -le 40 -and $State.battery_level -ge 30){return 'PASS'}
 if($ExplicitWaiver){return 'WAIVED_BY_EXPLICIT_USER_FOR_CURRENT_RESUME'}
 return 'BLOCKED'
}
