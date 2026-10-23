# Host-only dry run of the repaired G1 backward formal foreground monitor.
#
# Purpose: prove, without any device contact, that the repaired scheduler holds the
# registered "two-second foreground monitoring" contract at the *measured historical*
# per-command cost profile, and that every failure mode still fails closed.
#
# Evidence basis for the emulated latencies (no new device measurement is invented):
#   build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/monitor-cadence-attribution-private.json
#   competition (ps) 379 ms = post_top_to_ps_completion median 2379.454 ms - 2000 ms additive wait
#   awake (dumpsys power) 994 ms, top (dumpsys activity activities) 361 ms
# Those three commands are the historical serial block; the original runner added a
# 2000 ms WaitForExit *in series*, which produced the observed 4.03 s median interval.
#
# This script never calls adb. It owns one host child process as the "probe" so that the
# owned-exit path is exercised for real, and it owns one short-lived host child per
# auxiliary (thermal) claim so that the auxiliary lane is exercised for real.
param(
 [string]$EvidencePath='build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/monitor-dryrun-private.json',
 [int]$Samples=10,
 [switch]$SelfTest,
 [switch]$Force
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1')

$PeriodMilliseconds=2000
# The registered requirement is start-to-start <= 2000 ms. Lead only shortens the
# scheduled gap; it can never extend it. 100 ms absorbs this host's coalesced
# Start-Sleep granularity (measured 15-31 ms per 5 ms step) with margin.
$LeadMilliseconds=100
$CommandTimeoutSeconds=8
$AuxiliaryPeriodMilliseconds=6000
$AuxiliaryCollectBoundMilliseconds=4000

$competitionLatencyMilliseconds=379
$awakeLatencyMilliseconds=994
$topLatencyMilliseconds=361

$competitionText=" 1000  1000 system         surfaceflinger /system/bin/surfaceflinger`n 1200  1000 u0_a10         com.example.home com.example.home`n"
$awakeText=" mWakefulness=Awake`n mWakefulnessChanging=false`n"
$topText="  topResumedActivity=ActivityRecord{2f1a3c4 u0 com.example.home/.MainActivity t42}`n"

function New-DryRunScripts {
 param([int]$CompetitionMs,[int]$AwakeMs,[int]$TopMs,[switch]$Timeout,[string]$Awake,[string]$Top)
 $state=@{competition_ms=$CompetitionMs;awake_ms=$AwakeMs;top_ms=$TopMs;timeout=[bool]$Timeout;awake_text=$Awake;top_text=$Top}
 $invoker={
  param([string[]]$Arguments,[int]$TimeoutSeconds)
  $key='competition'
  if($Arguments -contains 'power'){$key='awake'}elseif($Arguments -contains 'activities'){$key='top'}
  $delay=0
  if($key -eq 'competition'){$delay=$state.competition_ms}elseif($key -eq 'awake'){$delay=$state.awake_ms}else{$delay=$state.top_ms}
  if($delay -gt 0){Start-Sleep -Milliseconds $delay}
  if($state.timeout){return [pscustomobject]@{ExitCode=124;Text='';Classification='ADB_TRANSPORT_TIMEOUT'}}
  $text=if($key -eq 'awake'){$state.awake_text}elseif($key -eq 'top'){$state.top_text}else{$competitionText}
  return [pscustomobject]@{ExitCode=0;Text=$text;Classification='ADB_OK'}
 }.GetNewClosure()
 $competitionProbe={
  param($text)
  if([string]::IsNullOrEmpty($text)){return [pscustomobject]@{Ok=$false;Known=$false;Reason='COMPETITION_UNKNOWN';Detail=$null}}
  return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}
 }
 $awakeProbe={
  param($text)
  if($text -match '(?m)^\s*mWakefulness=Awake\s*$'){return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}}
  if($text -match '(?m)^\s*mWakefulness='){return [pscustomobject]@{Ok=$false;Known=$true;Reason='AWAKE_LOST';Detail=$null}}
  return [pscustomobject]@{Ok=$false;Known=$false;Reason='AWAKE_UNKNOWN';Detail=$null}
 }
 $topProbe={
  param($text)
  $m=[regex]::Match($text,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
  if(!$m.Success){return [pscustomobject]@{Ok=$false;Known=$false;Reason='TOP_RESUMED_UNKNOWN';Detail=$null}}
  if($m.Groups[1].Value -eq 'com.example.target'){return [pscustomobject]@{Ok=$false;Known=$true;Reason='FOREGROUND_TAKEOVER';Detail=$m.Groups[1].Value}}
  return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$m.Groups[1].Value}
 }
 return [pscustomobject]@{Invoker=$invoker;CompetitionProbe=$competitionProbe;AwakeProbe=$awakeProbe;TopProbe=$topProbe}
}

function Invoke-DryRunSample {
 param([object]$Monitor,[object]$Scripts,[string]$EvidencePath,[string]$Phase,[object]$AuxStarter,[object]$AuxCollector)
 $parameters=@{Monitor=$Monitor;AdbInvoker=$Scripts.Invoker;TopProbe=$Scripts.TopProbe;AwakeProbe=$Scripts.AwakeProbe;CompetitionProbe=$Scripts.CompetitionProbe;Phase=$Phase}
 if($null -ne $AuxStarter){$parameters['AuxiliaryStarter']=$AuxStarter}
 if($null -ne $AuxCollector){$parameters['AuxiliaryCollector']=$AuxCollector}
 if(-not [string]::IsNullOrEmpty($EvidencePath)){$parameters['EvidencePath']=$EvidencePath}
 return (Invoke-PhoneLmMonitorSample @parameters)
}

function Test-DryRunFailClosed {
 # Every failure mode must produce contract_met = false with the specific ledger change.
 $cases=[System.Collections.Generic.List[object]]::new()
 $healthy=New-DryRunScripts -CompetitionMs 0 -AwakeMs 0 -TopMs 0 -Awake $awakeText -Top $topText
 $timeout=New-DryRunScripts -CompetitionMs 0 -AwakeMs 0 -TopMs 0 -Timeout -Awake $awakeText -Top $topText
 $asleep=New-DryRunScripts -CompetitionMs 0 -AwakeMs 0 -TopMs 0 -Awake " mWakefulness=Asleep`n" -Top $topText
 $unknownTop=New-DryRunScripts -CompetitionMs 0 -AwakeMs 0 -TopMs 0 -Awake $awakeText -Top "no top line`n"
 $takeover=New-DryRunScripts -CompetitionMs 0 -AwakeMs 0 -TopMs 0 -Awake $awakeText -Top "  topResumedActivity=ActivityRecord{9 u0 com.example.target/.A t1}`n"

 function Assert-Case([string]$Name,[bool]$Condition,[string]$Detail){
  $cases.Add([pscustomobject]@{name=$Name;passed=[bool]$Condition;detail=$Detail})
 }

 $m1=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $r1=Invoke-DryRunSample -Monitor $m1 -Scripts $timeout -Phase 'timeout'
 Assert-Case 'timeout_counted' ([int]$m1.CommandTimeouts -ge 1) ("command_timeouts=$($m1.CommandTimeouts)")
 Assert-Case 'timeout_contract_failed' ((Get-PhoneLmMonitorSummary -Monitor $m1).contract_met -eq $false) 'contract_met'

 $m2=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $r2=Invoke-DryRunSample -Monitor $m2 -Scripts $asleep -Phase 'asleep'
 Assert-Case 'asleep_invalid' ([int]$m2.InvalidSamples -ge 1) ("invalid_samples=$($m2.InvalidSamples)")
 Assert-Case 'asleep_contract_failed' ((Get-PhoneLmMonitorSummary -Monitor $m2).contract_met -eq $false) 'contract_met'

 $m3=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $r3=Invoke-DryRunSample -Monitor $m3 -Scripts $unknownTop -Phase 'unknown-top'
 Assert-Case 'unknown_top_unknown' ([int]$m3.UnknownStates -ge 1) ("unknown_states=$($m3.UnknownStates)")
 Assert-Case 'unknown_top_contract_failed' ((Get-PhoneLmMonitorSummary -Monitor $m3).contract_met -eq $false) 'contract_met'

 $m4=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $r4=Invoke-DryRunSample -Monitor $m4 -Scripts $takeover -Phase 'takeover'
 Assert-Case 'takeover_invalid' ([int]$m4.InvalidSamples -ge 1) ("invalid_samples=$($m4.InvalidSamples)")
 Assert-Case 'takeover_reason' (@($r4.violations|Where-Object {$_.reason -eq 'SAMPLE_STATE_INVALID'}).Count -ge 1) 'violation SAMPLE_STATE_INVALID'

 # A sample that outlives the period must be recorded as a missed deadline.
 $slow=New-DryRunScripts -CompetitionMs 2400 -AwakeMs 0 -TopMs 0 -Awake $awakeText -Top $topText
 $m5=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $r5=Invoke-DryRunSample -Monitor $m5 -Scripts $slow -Phase 'slow'
 Assert-Case 'slow_sample_missed' ([int]$m5.MissedDeadlines -ge 1 -and @($r5.violations|Where-Object {$_.reason -eq 'SAMPLE_OVERLAPS_NEXT_DEADLINE'}).Count -ge 1) ("missed_deadlines=$($m5.MissedDeadlines)")

 # A start-to-start above the period must be recorded even when the sample itself is fast.
 $m6=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds
 $m6.PreviousSampleStartMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m6.Clock)-3000.0
 $m6.NextSampleIndex=1
 $r6=Invoke-DryRunSample -Monitor $m6 -Scripts $healthy -Phase 'start-to-start'
 Assert-Case 'start_to_start_missed' (@($r6.violations|Where-Object {$_.reason -eq 'START_TO_START_EXCEEDED'}).Count -ge 1) 'violation START_TO_START_EXCEEDED'
 Assert-Case 'start_to_start_contract_failed' ((Get-PhoneLmMonitorSummary -Monitor $m6).contract_met -eq $false) 'contract_met'

 $failed=@($cases|Where-Object {-not $_.passed})
 return [pscustomobject][ordered]@{Cases=$cases;Failures=@($failed|ForEach-Object {$_.name});Passed=($failed.Count -eq 0)}
}

$failClosed=Test-DryRunFailClosed
foreach($case in $failClosed.Cases){
 Write-Host ('dryrun_case={0} result={1} {2}' -f $case.name,$(if($case.passed){'PASS'}else{'FAIL'}),$case.detail)
}
if($SelfTest){$requiredSamples=2}else{$requiredSamples=8}
if($Samples -lt $requiredSamples){$Samples=$requiredSamples}

$scratch=Join-Path $root ('build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/host-tests/dryrun-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch)|Out-Null
$script:dryRunChildren=[System.Collections.Generic.List[object]]::new()

function New-AuxiliaryLane {
 param([object]$Children)
 $starter={
  param($monitor)
  $path=Join-Path $scratch ('aux-'+[guid]::NewGuid().ToString('N')+'.txt')
  $command="Start-Sleep -Milliseconds 250; Set-Content -LiteralPath '$path' -Value MONITOR"
  $child=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-Command',$command) -WindowStyle Hidden -PassThru
  $Children.Add($child)
  [pscustomobject]@{StartedMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $monitor.Clock);Process=$child;Path=$path}
 }.GetNewClosure()
 $collector={
  param($claim)
  $process=$claim.Process
  $process.Refresh()
  if(-not $process.HasExited){return [pscustomobject]@{Complete=$false;Classification=$null}}
  if([IO.File]::Exists($claim.Path)){
   $token=([IO.File]::ReadAllText($claim.Path)).Trim()
   return [pscustomobject]@{Complete=$true;Classification=$token}
  }
  return [pscustomobject]@{Complete=$false;Classification='AUXILIARY_SAMPLE_FAILED'}
 }.GetNewClosure()
 return [pscustomobject]@{Starter=$starter;Collector=$collector}
}

$ownedProcess=$null
$monitor=$null
$summary=$null
$ownedExitDetected=$false
$startedUtc=[DateTime]::UtcNow
try {
 $ownedProcess=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-Command','Start-Sleep -Seconds 90') -WindowStyle Hidden -PassThru
 $monitor=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds $CommandTimeoutSeconds -AuxiliaryPeriodMilliseconds $AuxiliaryPeriodMilliseconds -AuxiliaryCollectBoundMilliseconds $AuxiliaryCollectBoundMilliseconds
 $scripts=New-DryRunScripts -CompetitionMs $competitionLatencyMilliseconds -AwakeMs $awakeLatencyMilliseconds -TopMs $topLatencyMilliseconds -Awake $awakeText -Top $topText
 $aux=New-AuxiliaryLane -Children $script:dryRunChildren
 $sampleCount=0
 for($index=0;$index -lt $Samples;$index++){
  $wait=Wait-PhoneLmMonitorDeadline -Monitor $monitor -OwnedExitProbe { $ownedProcess.Refresh(); $ownedProcess.HasExited }
  if($wait -eq 'OWNED_EXIT'){Write-Host ('dryrun_iteration={0} wait=OWNED_EXIT' -f $index);break}
  $null=Invoke-DryRunSample -Monitor $monitor -Scripts $scripts -Phase 'dryrun' -EvidencePath "$scratch/monitor-samples.jsonl" -AuxStarter $aux.Starter -AuxCollector $aux.Collector
  $sampleCount++
  $ownedProcess.Refresh()
  if($ownedProcess.HasExited){break}
 }
 $summary=Get-PhoneLmMonitorSummary -Monitor $monitor
 # Owned exit must be detected and must not force one more sample.
 $ownedProcess.Kill();$null=$ownedProcess.WaitForExit(5000)
 $exitProbe=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds
 $exitWait=Wait-PhoneLmMonitorDeadline -Monitor $exitProbe -OwnedExitProbe { $true }
 $ownedExitDetected=($exitWait -eq 'OWNED_EXIT')
} finally {
 foreach($child in $script:dryRunChildren){try{$child.Refresh();if(-not $child.HasExited){$child.Kill()}}catch{}}
 if($null -ne $ownedProcess){try{$ownedProcess.Refresh();if(-not $ownedProcess.HasExited){$ownedProcess.Kill()}}catch{}}
}

$dryRunPassed=(
 $failClosed.Passed -and
 $null -ne $summary -and
 $summary.contract_met -eq $true -and
 $summary.monitoring_status -eq 'PASS' -and
 [int]$summary.sample_count -ge $requiredSamples -and
 [double]$summary.maximum_start_to_start_milliseconds -le $PeriodMilliseconds -and
 $ownedExitDetected
)

if(-not $SelfTest){
 $target=[IO.Path]::GetFullPath((Join-Path $root $EvidencePath))
 $record=[ordered]@{
  schema=1
  kind='G1_SHARED_LIFETIME_MONITOR_DRYRUN'
  host_only=$true
  device_calls=0
  adb_invocations=0
  emulated_latencies=$true
  emulation_basis='monitor-cadence-attribution-private.json medians: competition(ps)=379 ms, awake=994 ms, top=361 ms'
  period_milliseconds=[double]$PeriodMilliseconds
  lead_milliseconds=[double]$LeadMilliseconds
  command_timeout_seconds=[int]$CommandTimeoutSeconds
  auxiliary_period_milliseconds=[double]$AuxiliaryPeriodMilliseconds
  auxiliary_collect_bound_milliseconds=[double]$AuxiliaryCollectBoundMilliseconds
  sample_count=[int]$summary.sample_count
  minimum_start_to_start_milliseconds=$summary.minimum_start_to_start_milliseconds
  median_start_to_start_milliseconds=$summary.median_start_to_start_milliseconds
  maximum_start_to_start_milliseconds=[double]$summary.maximum_start_to_start_milliseconds
  maximum_sample_duration_milliseconds=[double]$summary.maximum_sample_duration_milliseconds
  missed_deadlines=[int]$summary.missed_deadlines
  command_timeouts=[int]$summary.command_timeouts
  invalid_samples=[int]$summary.invalid_samples
  unknown_states=[int]$summary.unknown_states
  auxiliary_samples=[int]$summary.auxiliary_samples
  auxiliary_failures=[int]$summary.auxiliary_failures
  violation_count=$summary.violation_count
  violation_reasons=@($summary.violation_reasons)
  contract_met=[bool]$summary.contract_met
  monitoring_status=[string]$summary.monitoring_status
  owned_exit_detected=[bool]$ownedExitDetected
  host_timeout_tests_pass=[bool]$failClosed.Passed
  host_timeout_cases=@($failClosed.Cases|ForEach-Object {[ordered]@{name=$_.name;passed=$_.passed;detail=$_.detail}})
  dry_run_passed=[bool]$dryRunPassed
  monitor_module_sha256=(Get-FileHash (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1')).Hash.ToLowerInvariant()
  dryrun_script_sha256=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant()
  evidence_samples_relative=$null
  started_utc=$startedUtc.ToString('o')
  utc=[DateTime]::UtcNow.ToString('o')
 }
 $directory=[IO.Path]::GetDirectoryName($target)
 if(-not [IO.Directory]::Exists($directory)){[void][IO.Directory]::CreateDirectory($directory)}
 if(([IO.File]::Exists($target)) -and -not $Force){throw 'DRYRUN_EVIDENCE_EXISTS'}
 $temporary="$target.tmp"
 $stream=[IO.File]::Open($temporary,[IO.FileMode]::Create)
 try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($record|ConvertTo-Json -Depth 8));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
 [IO.File]::Move($temporary,$target,$true)
 Write-Host ('monitor_dryrun_evidence={0}' -f $EvidencePath)
}
$maxText=if($null -ne $summary){[double]$summary.maximum_start_to_start_milliseconds}else{0}
Write-Host ('monitor_dryrun_samples={0} max_start_to_start_ms={1} contract_met={2} fail_closed_cases={3}' -f $(if($null -ne $summary){$summary.sample_count}else{0}),$maxText,$(if($null -ne $summary){$summary.contract_met}else{$false}),$failClosed.Cases.Count)
if($null -ne $summary){Write-Host ('monitor_dryrun_counters missed={0} timeouts={1} invalid={2} unknown={3} aux_fail={4} violations={5} reasons={6}' -f $summary.missed_deadlines,$summary.command_timeouts,$summary.invalid_samples,$summary.unknown_states,$summary.auxiliary_failures,$summary.violation_count,($summary.violation_reasons -join '|'))}
if($dryRunPassed){Write-Host 'backward_monitor_dryrun=PASS device_calls=0';exit 0}
Write-Host ('backward_monitor_dryrun=FAIL failures={0}' -f (($failClosed.Failures -join ',')))
exit 1
