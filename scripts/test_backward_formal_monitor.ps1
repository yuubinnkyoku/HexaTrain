# Host-only fault-injection test for scripts/backward_formal_monitor.ps1.
# No adb, no device, no network: every ADB interaction is a synthetic scriptblock that only
# sleeps and returns fabricated text. The only writes are inside a GUID-suffixed directory
# under build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/host-tests/.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1')

$root=Split-Path -Parent $PSScriptRoot
$reportBase=Join-Path $root 'build/reports/g1-backward-shared-lifetime-formal/monitor-transport-recovery/host-tests'
$out=Join-Path $reportBase ('host-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($out)|Out-Null

# Assertion (d) guard: the module must not contain any process-termination primitive,
# and nothing in this harness ever attempts one.
$moduleText=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1') -Raw
if($moduleText -match '(?i)Stop-Process|\.Kill\(|taskkill'){throw 'MODULE_CONTAINS_PROCESS_TERMINATION'}
$script:processTerminationAttempts=0

function Assert-Case {
 param([string]$Check,[bool]$Condition)
 if(!$Condition){throw "CASE_ASSERTION_FAILED $Check"}
}

function Assert-DeadlineAccounting {
 param([object]$Monitor,[string]$CaseName,[switch]$TextPreserved)
 Assert-Case "$CaseName lateness_count" ($Monitor.LatenessMilliseconds.Count -eq $Monitor.Samples.Count)
 Assert-Case "$CaseName duration_count" ($Monitor.SampleDurations.Count -eq $Monitor.Samples.Count)
 for($i=0;$i -lt $Monitor.Samples.Count;$i++){
  $sample=$Monitor.Samples[$i]
  Assert-Case "$CaseName sample[$i].sample_start" ($null -ne $sample.sample_start)
  Assert-Case "$CaseName sample[$i].sample_complete" ($null -ne $sample.sample_complete)
  Assert-Case "$CaseName sample[$i].deadline" ($null -ne $sample.scheduled_deadline_milliseconds)
  Assert-Case "$CaseName sample[$i].duration" ($null -ne $sample.duration_milliseconds -and $sample.duration_milliseconds -ge 0)
  Assert-Case "$CaseName sample[$i].lateness" ($null -ne $sample.lateness_milliseconds)
  Assert-Case "$CaseName sample[$i].commands" ($sample.commands.Count -eq 3)
  foreach($name in 'competition','awake','top'){
   $command=$sample.commands[$name]
   Assert-Case "$CaseName sample[$i].$name entry" ($null -ne $command)
   Assert-Case "$CaseName sample[$i].$name start_end" ($null -ne $command.command_start -and $null -ne $command.command_end -and $command.command_end -ge $command.command_start)
   Assert-Case "$CaseName sample[$i].$name classification" ($null -ne $command.classification)
   Assert-Case "$CaseName sample[$i].$name probe" ($null -ne $command.probe)
   if($TextPreserved){
    Assert-Case "$CaseName sample[$i].$name raw_text_preserved" ($null -ne $command.text)
   } else {
    Assert-Case "$CaseName sample[$i].$name raw_text_hidden" ($null -eq $command.text)
   }
   Assert-Case "$CaseName sample[$i].$name sha256" (![string]::IsNullOrEmpty($command.text_sha256))
   Assert-Case "$CaseName sample[$i].$name text_length" ($null -ne $command.text_length)
  }
 }
}

function Complete-Case {
 param([int]$Number,[string]$Name,[object]$Monitor,[switch]$TextPreserved)
 if($script:processTerminationAttempts -ne 0){throw "case=$Number $Name result=FAIL reason=process_termination_attempted"}
 if($null -ne $Monitor){Assert-DeadlineAccounting -Monitor $Monitor -CaseName "case$Number" -TextPreserved:$TextPreserved}
 Write-Host "case=$Number $Name result=PASS"
}

# ---------------------------------------------------------------- fake device layer
$script:fake=[pscustomobject]@{
 LatencyMs=@{competition=0;awake=0;top=0}
 Results=@{}
 ExtraLatencyWhilePendingMs=0
 PendingMonitor=$null
 Calls=0
 CallKeys=[System.Collections.Generic.List[string]]::new()
}
function Reset-FakeDevice {
 param([hashtable]$Latency=@{},[hashtable]$Results=@{},[int]$ExtraLatencyWhilePendingMs=0)
 $script:fake.LatencyMs=@{competition=0;awake=0;top=0}
 foreach($key in $Latency.Keys){$script:fake.LatencyMs[[string]$key]=[int]$Latency[$key]}
 $script:fake.Results=@{}
 foreach($key in $Results.Keys){$script:fake.Results[[string]$key]=$Results[$key]}
 $script:fake.ExtraLatencyWhilePendingMs=$ExtraLatencyWhilePendingMs
 $script:fake.PendingMonitor=$null
 $script:fake.Calls=0
 $script:fake.CallKeys.Clear()
}

function Get-FakeCommandKey {
 param([string[]]$Arguments)
 if($Arguments -contains 'ps'){return 'competition'}
 if($Arguments -contains 'power'){return 'awake'}
 if($Arguments -contains 'activities'){return 'top'}
 throw 'FAKE_UNEXPECTED_ARGUMENTS'
}

function Get-FakeHealthyText {
 param([string]$Key)
 switch($Key){
  'competition' {
   return ("    1000    1000 system     surfaceflinger /system/bin/surfaceflinger --arg`n" +
           "    2000    2000 system     logd /system/bin/logd`n")
  }
  'awake' { return "POWER MANAGER (dumpsys power):`n  mWakefulness=Awake`n" }
  'top' { return ("ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities):`n" +
                  "  topResumedActivity=ActivityRecord{aa11bb22 u5 com.example.home/.Launcher t42}`n") }
  default { return "synthetic unknown command`n" }
 }
}

# Synthetic latency via Start-Sleep only; returns fabricated ExitCode/Text/Classification.
function Invoke-FakeAdb {
 param([string[]]$Arguments,[int]$TimeoutSeconds)
 $script:fake.Calls++
 $key=Get-FakeCommandKey -Arguments $Arguments
 $script:fake.CallKeys.Add($key)
 $latency=[int]$script:fake.LatencyMs[$key]
 # Extra latency lands on a single transport operation (awake) while a simulated transfer
 # claim is pending: one delayed command, not three, keeps the accounting attributable.
 if($key -eq 'awake' -and $script:fake.ExtraLatencyWhilePendingMs -gt 0 -and $null -ne $script:fake.PendingMonitor -and $null -ne $script:fake.PendingMonitor.AuxiliaryPending){
  $latency+=$script:fake.ExtraLatencyWhilePendingMs
 }
 if($latency -gt 0){Start-Sleep -Milliseconds $latency}
 if($script:fake.Results.ContainsKey($key)){return $script:fake.Results[$key]}
 return [pscustomobject]@{ExitCode=0;Text=(Get-FakeHealthyText -Key $key);Classification='ADB_OK'}
}

# ---------------------------------------------------------------- fake probes
$script:probeCfg=@{Top='auto';Awake='auto';Competition='auto'}
function Reset-Probes {
 $script:probeCfg=@{Top='auto';Awake='auto';Competition='auto'}
}

function Invoke-FakeTopProbe {
 param($text)
 if($script:probeCfg.Top -eq 'throw'){throw 'fake top probe exploded'}
 if($script:probeCfg.Top -eq 'unknown'){return [pscustomobject]@{Ok=$true;Known=$false;Reason='TOP_RESUMED_UNKNOWN';Detail=$null}}
 if($script:probeCfg.Top -eq 'invalid'){return [pscustomobject]@{Ok=$false;Known=$true;Reason='UNEXPECTED_FOREGROUND_PACKAGE';Detail=$null}}
 $matched=[regex]::Match([string]$text,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
 if(!$matched.Success){return [pscustomobject]@{Ok=$true;Known=$false;Reason='TOP_RESUMED_UNKNOWN';Detail=$null}}
 $package=$matched.Groups[1].Value
 if($package -eq 'com.yuubinnkyoku.phonelm'){
  return [pscustomobject]@{Ok=$false;Known=$true;Reason='UNEXPECTED_FOREGROUND_PACKAGE';Detail=@{package=$package}}
 }
 return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=@{package=$package}}
}

function Invoke-FakeAwakeProbe {
 param($text)
 if($script:probeCfg.Awake -eq 'throw'){throw 'fake awake probe exploded'}
 $value=[string]$text
 if($value -match '(?m)^\s*mWakefulness=Awake\s*$'){
  return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}
 }
 $stateMatch=[regex]::Match($value,'(?m)^\s*mWakefulness=(\w+)\s*$')
 if($stateMatch.Success){
  return [pscustomobject]@{Ok=$false;Known=$true;Reason='DEVICE_NOT_AWAKE';Detail=@{wakefulness=$stateMatch.Groups[1].Value}}
 }
 return [pscustomobject]@{Ok=$true;Known=$false;Reason='POWER_STATE_UNKNOWN';Detail=$null}
}

function Invoke-FakeCompetitionProbe {
 param($text)
 if($script:probeCfg.Competition -eq 'throw'){throw 'fake competition probe exploded'}
 foreach($line in ([string]$text -split "`n")){
  if($line -match '^\s*(\d+)\s+\d+\s+\S+\s+(\S+)'){
   $name=$Matches[2]
   if($name -match '(?i)(probe|qnn-net-run|simple_app|hmx|hexatrain|training)'){
    return [pscustomobject]@{Ok=$false;Known=$true;Reason='COMPETING_NPU_WORKLOAD';Detail=@{line=$line}}
   }
  }
 }
 return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}
}

# ---------------------------------------------------------------- shared helpers
function New-ShortMonitor {
 # Short period for speed. Lead 40 ms keeps the measured start-to-start inside the 60 ms
 # period even when the host sleep granularity is one ~15.5 ms timer tick.
 return New-PhoneLmForegroundMonitor -PeriodMilliseconds 60 -LeadMilliseconds 40 -CommandTimeoutSeconds 2 -AuxiliaryPeriodMilliseconds 600000 -AuxiliaryCollectBoundMilliseconds 600000
}

function Invoke-CaseSample {
 param(
  [object]$Monitor,
  [scriptblock]$AuxiliaryStarter,
  [scriptblock]$AuxiliaryCollector,
  [string]$EvidencePath,
  [string]$Phase,
  [switch]$PreserveText
 )
 $invokeArguments=@{
  Monitor=$Monitor
  AdbInvoker=${function:Invoke-FakeAdb}
  TopProbe=${function:Invoke-FakeTopProbe}
  AwakeProbe=${function:Invoke-FakeAwakeProbe}
  CompetitionProbe=${function:Invoke-FakeCompetitionProbe}
  AuxiliaryStarter=$AuxiliaryStarter
  AuxiliaryCollector=$AuxiliaryCollector
 }
 if($PSBoundParameters.ContainsKey('EvidencePath')){$invokeArguments.EvidencePath=$EvidencePath}
 if($PSBoundParameters.ContainsKey('Phase')){$invokeArguments.Phase=$Phase}
 if($PreserveText){$invokeArguments.PreserveText=$true}
 return Invoke-PhoneLmMonitorSample @invokeArguments
}

function Get-Sha256Hex {
 param([string]$Text)
 $sha=[Security.Cryptography.SHA256]::Create()
 try {return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))) -replace '-','').ToLowerInvariant()}
 finally {$sha.Dispose()}
}

# ---------------------------------------------------------------- cold-start warm-up
# The first Invoke in a fresh process pays ~150 ms of one-time JIT/initialisation cost;
# warm it up so fault-injection measurements reflect steady-state behaviour.
Reset-FakeDevice
$null=Start-Sleep -Milliseconds 5
$warmup=New-PhoneLmForegroundMonitor -PeriodMilliseconds 500 -LeadMilliseconds 5 -CommandTimeoutSeconds 2
$null=Invoke-CaseSample -Monitor $warmup -Phase 'warmup'
$null=Invoke-CaseSample -Monitor $warmup -Phase 'warmup'
$warmWait=New-ShortMonitor
$warmWait.StartMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $warmWait.Clock)+10.0
Assert-Case 'warmup_wait' ((Wait-PhoneLmMonitorDeadline -Monitor $warmWait) -eq 'DEADLINE_REACHED')

# ---------------------------------------------------------------- module self-test
$selfTest=Test-PhoneLmMonitorSelfTest
if(!$selfTest.Passed){throw ('MONITOR_SELFTEST_FAILED failures='+($selfTest.Failures -join ';'))}

# ---------------------------------------------------------------- case 1: normal fast response
# No synthetic latency: a normal fast response completes in a few milliseconds.
Reset-FakeDevice
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r1=Invoke-CaseSample -Monitor $m -Phase 'case1'
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r2=Invoke-CaseSample -Monitor $m -Phase 'case1'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
Assert-Case 'c1 healthy_state' ($r1.state -eq 'SAMPLE_STATE_HEALTHY' -and $r2.state -eq 'SAMPLE_STATE_HEALTHY')
Assert-Case 'c1 exit_zero' ($r1.commands.competition.exit_code -eq 0 -and $r1.commands.awake.exit_code -eq 0 -and $r1.commands.top.exit_code -eq 0)
Assert-Case 'c1 classification' ($r1.commands.top.classification -eq 'ADB_OK')
Assert-Case 'c1 no_timeout' ($r1.commands.awake.timeout -eq $false -and $m.CommandTimeouts -eq 0)
Assert-Case 'c1 command_order' ((@($script:fake.CallKeys) -join ',') -eq 'competition,awake,top,competition,awake,top')
Assert-Case 'c1 no_violations' ($m.Violations.Count -eq 0 -and $r1.violations.Count -eq 0)
Assert-Case 'c1 counters_zero' ($m.MissedDeadlines -eq 0 -and $m.UnknownStates -eq 0 -and $m.InvalidSamples -eq 0 -and $m.CommandTimeouts -eq 0)
Assert-Case 'c1 s2s_within_period' ($m.MaximumStartToStartMilliseconds -le 60.0)
Assert-Case 'c1 duration_within_period' ($m.MaximumSampleDurationMilliseconds -le 60.0)
Assert-Case 'c1 contract' ($summary.contract_met -eq $true -and $summary.monitoring_status -eq 'PASS')
Assert-Case 'c1 raw_text_hidden' ($r1.commands.top.text -eq $null -and $r1.commands.top.text_length -gt 0 -and $r1.commands.top.text_sha256 -eq (Get-Sha256Hex (Get-FakeHealthyText -Key 'top')))
Complete-Case -Number 1 -Name 'normal_fast_response' -Monitor $m

# ---------------------------------------------------------------- case 2: response taking ~60% of the period
# Injected command time is 36 ms = 60% of the 60 ms period. Windows timer coalescing on this
# host makes a 36 ms Start-Sleep regularly land at a later tick boundary (observed 47-76 ms),
# so the deterministic assertions are: the full injected latency is consumed, and the
# overlap accounting is consistent in both directions (violation recorded if and only if
# the measured duration exceeds the period; fail-closed contract when it does).
Reset-FakeDevice -Latency @{top=36}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case2'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
Assert-Case 'c2 duration_consumes_injected_latency' ($r.duration_milliseconds -ge 36.0)
$overlap2=(@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_OVERLAPS_NEXT_DEADLINE')
if($r.duration_milliseconds -gt 60.0){
 Assert-Case 'c2 overlap_accounted_when_exceeded' ($overlap2 -eq $true -and $m.MissedDeadlines -ge 1 -and $summary.missed_deadlines -ge 1)
 Assert-Case 'c2 fail_closed_when_exceeded' ($summary.contract_met -eq $false -and $summary.monitoring_status -eq 'CONTRACT_VIOLATION')
} else {
 Assert-Case 'c2 clean_when_within_period' ($overlap2 -eq $false -and $m.MissedDeadlines -eq 0 -and $summary.contract_met -eq $true -and $summary.monitoring_status -eq 'PASS')
}
Assert-Case 'c2 s2s_within_period' ($m.MaximumStartToStartMilliseconds -le 60.0)
Complete-Case -Number 2 -Name 'response_sixty_percent_of_period' -Monitor $m

# ---------------------------------------------------------------- case 3: response exceeding the period
Reset-FakeDevice -Latency @{competition=25;awake=25;top=25}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case3'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$reasons=@($r.violations|ForEach-Object {$_.reason})
Assert-Case 'c3 duration_exceeds_period' ($r.duration_milliseconds -gt 60.0)
Assert-Case 'c3 overlap_violation' ($reasons -contains 'SAMPLE_OVERLAPS_NEXT_DEADLINE')
Assert-Case 'c3 missed_deadline_accounted' ($m.MissedDeadlines -ge 1 -and $summary.missed_deadlines -ge 1)
Assert-Case 'c3 fail_closed' ($summary.contract_met -eq $false -and $summary.monitoring_status -eq 'CONTRACT_VIOLATION')
Complete-Case -Number 3 -Name 'response_exceeds_period' -Monitor $m

# ---------------------------------------------------------------- case 4: ExitCode 124 / ADB_TRANSPORT_TIMEOUT
Reset-FakeDevice -Results @{
 awake=[pscustomobject]@{ExitCode=124;Text='';Classification='ADB_TRANSPORT_TIMEOUT'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case4'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$awake=$r.commands.awake
Assert-Case 'c4 exit_124_recorded' ($awake.exit_code -eq 124)
Assert-Case 'c4 timeout_flag' ($awake.timeout -eq $true)
Assert-Case 'c4 classification_preserved' ($awake.classification -eq 'ADB_TRANSPORT_TIMEOUT')
Assert-Case 'c4 timeout_counter' ($m.CommandTimeouts -ge 1 -and $summary.command_timeouts -ge 1)
Assert-Case 'c4 fail_closed_unknown' ($m.UnknownStates -ge 1 -and (@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_UNKNOWN'))
Assert-Case 'c4 probe_reason' ($awake.probe.known -eq $false -and $awake.probe.reason -eq 'POWER_STATE_UNKNOWN')
Assert-Case 'c4 empty_text_hashed' ($awake.text_length -eq 0 -and $awake.text_sha256 -eq (Get-Sha256Hex ''))
Assert-Case 'c4 state_unknown' ($r.state -eq 'SAMPLE_STATE_UNKNOWN')
Assert-Case 'c4 contract_false' ($summary.contract_met -eq $false)
Complete-Case -Number 4 -Name 'exit_124_transport_timeout' -Monitor $m

# ---------------------------------------------------------------- case 5: empty response text
Reset-FakeDevice -Results @{
 competition=[pscustomobject]@{ExitCode=0;Text='';Classification='ADB_OK'}
 awake=[pscustomobject]@{ExitCode=0;Text='';Classification='ADB_OK'}
 top=[pscustomobject]@{ExitCode=0;Text='';Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case5'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$reasons=@($r.violations|ForEach-Object {$_.reason})
Assert-Case 'c5 top_unknown' ($r.commands.top.probe.known -eq $false -and $r.commands.top.probe.reason -eq 'TOP_RESUMED_UNKNOWN')
Assert-Case 'c5 awake_unknown' ($r.commands.awake.probe.known -eq $false -and $r.commands.awake.probe.reason -eq 'POWER_STATE_UNKNOWN')
Assert-Case 'c5 unknown_counter' ($m.UnknownStates -ge 2)
Assert-Case 'c5 violation_recorded' ($reasons -contains 'SAMPLE_STATE_UNKNOWN')
Assert-Case 'c5 fail_closed' ($r.state -eq 'SAMPLE_STATE_UNKNOWN' -and $summary.contract_met -eq $false)
Assert-Case 'c5 lengths_zero' ($r.commands.top.text_length -eq 0 -and $r.commands.competition.text_length -eq 0)
Complete-Case -Number 5 -Name 'empty_response_text' -Monitor $m

# ---------------------------------------------------------------- case 6: valid response without topResumedActivity
Reset-FakeDevice -Results @{
 top=[pscustomobject]@{ExitCode=0;Text=("ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities):`n  ResumedActivity: null`n  mFocusedActivity: null`n");Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case6'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
Assert-Case 'c6 known_false' ($r.commands.top.probe.known -eq $false)
Assert-Case 'c6 reason' ($r.commands.top.probe.reason -eq 'TOP_RESUMED_UNKNOWN')
Assert-Case 'c6 violation' ((@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_UNKNOWN'))
Assert-Case 'c6 counter' ($m.UnknownStates -ge 1 -and $summary.unknown_states -ge 1)
Assert-Case 'c6 exit_zero_classification' ($r.commands.top.exit_code -eq 0 -and $r.commands.top.classification -eq 'ADB_OK')
Assert-Case 'c6 fail_closed' ($summary.contract_met -eq $false)
Complete-Case -Number 6 -Name 'missing_top_resumed_activity' -Monitor $m

# ---------------------------------------------------------------- case 7: unexpected top-resumed package
Reset-FakeDevice -Results @{
 top=[pscustomobject]@{ExitCode=0;Text=("  topResumedActivity=ActivityRecord{beef u0 com.yuubinnkyoku.phonelm/.MainActivity t3}`n");Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case7'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$topProbe=$r.commands.top.probe
Assert-Case 'c7 ok_false_known_true' ($topProbe.known -eq $true -and $topProbe.ok -eq $false)
Assert-Case 'c7 reason' ($topProbe.reason -eq 'UNEXPECTED_FOREGROUND_PACKAGE')
Assert-Case 'c7 detail_package_preserved' ($topProbe.detail.package -eq 'com.yuubinnkyoku.phonelm')
Assert-Case 'c7 invalid_counter' ($m.InvalidSamples -ge 1 -and $summary.invalid_samples -ge 1)
Assert-Case 'c7 violation' ((@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_INVALID'))
Assert-Case 'c7 fail_closed' ($r.state -eq 'SAMPLE_STATE_INVALID' -and $summary.contract_met -eq $false)
Complete-Case -Number 7 -Name 'unexpected_top_package' -Monitor $m

# ---------------------------------------------------------------- case 8: Dozing wakefulness
Reset-FakeDevice -Results @{
 awake=[pscustomobject]@{ExitCode=0;Text="POWER MANAGER (dumpsys power):`n  mWakefulness=Dozing`n";Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case8'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$awakeProbe=$r.commands.awake.probe
Assert-Case 'c8 dozing_detected' ($awakeProbe.ok -eq $false -and $awakeProbe.known -eq $true)
Assert-Case 'c8 reason' ($awakeProbe.reason -eq 'DEVICE_NOT_AWAKE')
Assert-Case 'c8 detail_wakefulness' ($awakeProbe.detail.wakefulness -eq 'Dozing')
Assert-Case 'c8 invalid_counter' ($m.InvalidSamples -ge 1)
Assert-Case 'c8 violation' ((@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_INVALID'))
Assert-Case 'c8 fail_closed' ($summary.contract_met -eq $false -and $summary.monitoring_status -eq 'CONTRACT_VIOLATION')
Complete-Case -Number 8 -Name 'dozing_wakefulness' -Monitor $m

# ---------------------------------------------------------------- case 9: ADB disconnect ADB_TRANSPORT_FAILURE
Reset-FakeDevice -Results @{
 awake=[pscustomobject]@{ExitCode=1;Text='';Classification='ADB_TRANSPORT_FAILURE'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case9'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$awake=$r.commands.awake
Assert-Case 'c9 classification_preserved' ($awake.classification -eq 'ADB_TRANSPORT_FAILURE')
Assert-Case 'c9 exit_preserved' ($awake.exit_code -eq 1)
Assert-Case 'c9 not_a_timeout' ($awake.timeout -eq $false -and $m.CommandTimeouts -eq 0 -and $summary.command_timeouts -eq 0)
Assert-Case 'c9 fail_closed_unknown' ($m.UnknownStates -ge 1 -and $r.state -eq 'SAMPLE_STATE_UNKNOWN')
Assert-Case 'c9 violation' ((@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_UNKNOWN'))
Assert-Case 'c9 contract_false' ($summary.contract_met -eq $false)
Complete-Case -Number 9 -Name 'adb_transport_failure' -Monitor $m

# ---------------------------------------------------------------- case 10: competing QNN process line
Reset-FakeDevice -Results @{
 competition=[pscustomobject]@{ExitCode=0;Text=("    4321       1 root       qnn-net-run --model /data/local/tmp/g1.bin`n" +
  "    1000    1000 system     surfaceflinger /system/bin/surfaceflinger --arg`n");Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case10'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$competitionProbe=$r.commands.competition.probe
Assert-Case 'c10 detected' ($competitionProbe.ok -eq $false -and $competitionProbe.known -eq $true)
Assert-Case 'c10 reason' ($competitionProbe.reason -eq 'COMPETING_NPU_WORKLOAD')
Assert-Case 'c10 detail_line_preserved' ($null -ne $competitionProbe.detail.line -and $competitionProbe.detail.line -match 'qnn-net-run')
Assert-Case 'c10 invalid_counter' ($m.InvalidSamples -ge 1)
Assert-Case 'c10 violation' ((@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_STATE_INVALID'))
Assert-Case 'c10 fail_closed' ($r.state -eq 'SAMPLE_STATE_INVALID' -and $summary.contract_met -eq $false)
Complete-Case -Number 10 -Name 'competing_qnn_process' -Monitor $m

# ---------------------------------------------------------------- case 11: critical thermal via auxiliary collector
Reset-FakeDevice
Reset-Probes
$script:auxStarterCalls=0
$script:auxCollectorCalls=0
$starter={
 param($monitorObject)
 $script:auxStarterCalls++
 return [pscustomobject]@{Kind='TRANSFER';StartedMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $monitorObject.Clock)}
}
$thermalCollector={
 param($claim)
 $script:auxCollectorCalls++
 return [pscustomobject]@{Complete=$true;Classification='BLOCKED_OS_CRITICAL_THERMAL'}
}
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$null=Invoke-CaseSample -Monitor $m -AuxiliaryStarter $starter -AuxiliaryCollector $thermalCollector -Phase 'case11'
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r2=Invoke-CaseSample -Monitor $m -AuxiliaryStarter $starter -AuxiliaryCollector $thermalCollector -Phase 'case11'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$reasons=@($m.Violations|ForEach-Object {$_.reason})
Assert-Case 'c11 starter_once' ($script:auxStarterCalls -eq 1)
Assert-Case 'c11 collector_once' ($script:auxCollectorCalls -eq 1)
Assert-Case 'c11 aux_sample_recorded' ($m.AuxiliarySamples.Count -eq 1 -and $m.AuxiliarySamples[0].classification -eq 'BLOCKED_OS_CRITICAL_THERMAL')
Assert-Case 'c11 aux_failure_accounted' ($m.AuxiliaryFailures -ge 1 -and $summary.auxiliary_failures -ge 1)
Assert-Case 'c11 violation_recorded' ($reasons -contains 'AUXILIARY_SAMPLE_FAILED')
Assert-Case 'c11 sample_violation_present' ((@($r2.violations|ForEach-Object {$_.reason}) -contains 'AUXILIARY_SAMPLE_FAILED'))
Assert-Case 'c11 pending_cleared' ($null -eq $m.AuxiliaryPending)
Assert-Case 'c11 fail_closed' ($summary.contract_met -eq $false -and $summary.monitoring_status -eq 'CONTRACT_VIOLATION')
Complete-Case -Number 11 -Name 'critical_thermal_auxiliary' -Monitor $m

# ---------------------------------------------------------------- case 12: a probe that throws
Reset-FakeDevice
Reset-Probes
$script:probeCfg.Awake='throw'
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$threw=$false
$r=$null
try {$r=Invoke-CaseSample -Monitor $m -Phase 'case12'}
catch {$threw=$true}
Reset-Probes
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$reasons=@($m.Violations|ForEach-Object {$_.reason})
$evaluationFailure=$m.Violations|Where-Object {$_.reason -eq 'SAMPLE_EVALUATION_FAILURE'}|Select-Object -First 1
Assert-Case 'c12 invoke_returns_record' ($threw -eq $false -and $null -ne $r)
Assert-Case 'c12 violation_recorded' ($reasons -contains 'SAMPLE_EVALUATION_FAILURE')
Assert-Case 'c12 message_preserved' ($null -ne $evaluationFailure -and $evaluationFailure.message -match 'fake awake probe exploded')
Assert-Case 'c12 probe_entry_error' ($r.commands.awake.probe.reason -eq 'PROBE_EVALUATION_EXCEPTION' -and $r.commands.awake.probe.detail -match 'fake awake probe exploded')
Assert-Case 'c12 fail_closed_unknown' ($m.UnknownStates -ge 1 -and $r.state -eq 'SAMPLE_STATE_UNKNOWN')
Assert-Case 'c12 contract_false' ($summary.contract_met -eq $false)
Assert-Case 'c12 no_termination' ($script:processTerminationAttempts -eq 0)
Complete-Case -Number 12 -Name 'probe_throws' -Monitor $m

# ---------------------------------------------------------------- case 13: monitor worker early termination
Reset-FakeDevice
Reset-Probes
$m=New-ShortMonitor
$null=Complete-PhoneLmMonitor -Monitor $m -Status 'ABORTED'
Assert-Case 'c13 status_aborted' ($m.Status -eq 'ABORTED')
Assert-Case 'c13 completion_timestamped' ($null -ne $m.CompletedMilliseconds -and $m.CompletedMilliseconds -gt 0)
Assert-Case 'c13 no_sample_consumed' ($m.NextSampleIndex -eq 0 -and $m.Samples.Count -eq 0)
$invalidThrew=$false
try {Complete-PhoneLmMonitor -Monitor $m -Status 'BOGUS'|Out-Null}
catch {$invalidThrew=$true}
Assert-Case 'c13 invalid_status_rejected' ($invalidThrew -eq $true)
$summary=Get-PhoneLmMonitorSummary -Monitor $m
Assert-Case 'c13 summary_fail_closed' ($summary.sample_count -eq 0 -and $summary.contract_met -eq $false -and $summary.monitoring_status -eq 'CONTRACT_VIOLATION')
Complete-Case -Number 13 -Name 'aborted_before_sample' -Monitor $m

# ---------------------------------------------------------------- case 14: owned process exiting during wait
Reset-FakeDevice
Reset-Probes
$m=New-ShortMonitor
$m.StartMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m.Clock)+3000.0
$waitStart=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
$ownedResult=Wait-PhoneLmMonitorDeadline -Monitor $m -OwnedExitProbe {$true}
$ownedEnd=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
Assert-Case 'c14 owned_exit' ($ownedResult -eq 'OWNED_EXIT')
Assert-Case 'c14 returns_immediately' (($ownedEnd-$waitStart) -lt 500.0)
Assert-Case 'c14 no_sample_consumed' ($m.NextSampleIndex -eq 0 -and $m.Samples.Count -eq 0 -and $m.Status -eq 'READY')
# deadline itself is still honoured when the probe reports no exit:
$m2=New-ShortMonitor
$m2.StartMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m2.Clock)+20.0
$deadlineResult=Wait-PhoneLmMonitorDeadline -Monitor $m2 -OwnedExitProbe {$false}
$deadlineEnd=Get-PhoneLmMonitorMilliseconds -Clock $m2.Clock
Assert-Case 'c14 deadline_reached' ($deadlineResult -eq 'DEADLINE_REACHED')
# The final sleep step is requested at <=5 ms, but the OS may deliver the wake at the next
# coalesced timer boundary; the module never requests sleep past the deadline, so the
# observed overshoot stays inside one coalesced wake quantum (observed max ~31 ms).
Assert-Case 'c14 bounded_overshoot' (($deadlineEnd-$m2.StartMilliseconds) -lt 80.0)
Complete-Case -Number 14 -Name 'owned_exit_wait' -Monitor $m

# ---------------------------------------------------------------- case 15: delayed stdout equivalent
# One late Text return: the top command's response takes 30 ms while the other two are fast.
# The payload is a valid (delayed) top response so the healthy path is what is measured.
$payload=("ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities):`n" +
          "  topResumedActivity=ActivityRecord{cafe u7 com.example.home/.Launcher t9}`n" +
          'delayed-stdout-payload-'+('x'*400)+"`n"+"second line of late output`n")
Reset-FakeDevice -Latency @{top=30} -Results @{
 top=[pscustomobject]@{ExitCode=0;Text=$payload;Classification='ADB_OK'}
}
Reset-Probes
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r=Invoke-CaseSample -Monitor $m -Phase 'case15' -PreserveText
$summary=Get-PhoneLmMonitorSummary -Monitor $m
$top=$r.commands.top
$payloadBytes=[Text.Encoding]::UTF8.GetBytes($payload)
Assert-Case 'c15 latency_consumed' ($r.duration_milliseconds -ge 30.0)
Assert-Case 'c15 text_preserved' ($top.text -eq $payload)
Assert-Case 'c15 length_exact' ($top.text_length -eq $payloadBytes.Length)
Assert-Case 'c15 sha256_exact' ($top.text_sha256 -eq (Get-Sha256Hex $payload))
Assert-Case 'c15 no_timeout' ($top.timeout -eq $false -and $top.exit_code -eq 0 -and $m.CommandTimeouts -eq 0)
Assert-Case 'c15 classification' ($top.classification -eq 'ADB_OK')
Assert-Case 'c15 state_healthy' ($r.state -eq 'SAMPLE_STATE_HEALTHY')
# Timer coalescing may push the delayed response past the period; the accounting must match
# the measurement in either direction.
$overlap15=(@($r.violations|ForEach-Object {$_.reason}) -contains 'SAMPLE_OVERLAPS_NEXT_DEADLINE')
Assert-Case 'c15 overlap_consistency' ($overlap15 -eq ($r.duration_milliseconds -gt 60.0))
Complete-Case -Number 15 -Name 'delayed_stdout_return' -Monitor $m -TextPreserved

# ---------------------------------------------------------------- case 16: evidence collection competing for ADB
# Base transport latency is zero; when the simulated transfer claim is pending the invoker
# adds 25 ms to the awake command of that sample only (tick rounding keeps it under 60 ms).
Reset-FakeDevice -ExtraLatencyWhilePendingMs 25
Reset-Probes
$script:auxCollectorCalls=0
$slowCollector={
 param($claim)
 $script:auxCollectorCalls++
 return [pscustomobject]@{Complete=$false;Classification=$null}
}
$evidencePath=Join-Path $out 'case16-evidence.jsonl'
$m=New-ShortMonitor
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r1=Invoke-CaseSample -Monitor $m -AuxiliaryCollector $slowCollector -EvidencePath $evidencePath -Phase 'case16'
# A simulated transfer claim becomes pending before the next sample; the fake invoker then
# adds latency on exactly the sample where that claim competes for the transport.
$m.AuxiliaryPending=[pscustomobject]@{Kind='TRANSFER';StartedMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m.Clock)}
$m.AuxiliaryPendingStartedMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m.Clock)
$script:fake.PendingMonitor=$m
$null=Wait-PhoneLmMonitorDeadline -Monitor $m
$r2=Invoke-CaseSample -Monitor $m -AuxiliaryCollector $slowCollector -EvidencePath $evidencePath -Phase 'case16'
$summary=Get-PhoneLmMonitorSummary -Monitor $m
Assert-Case 'c16 competition_latency_consumed' ($r2.duration_milliseconds -ge ($r1.duration_milliseconds+15.0))
Assert-Case 'c16 collector_called_once' ($script:auxCollectorCalls -eq 1)
Assert-Case 'c16 claim_still_pending' ($null -ne $m.AuxiliaryPending)
Assert-Case 'c16 no_collection_timeout' ((@($m.Violations|ForEach-Object {$_.reason}) -contains 'AUXILIARY_COLLECTION_TIMEOUT') -eq $false)
Assert-Case 'c16 aux_failures_zero' ($m.AuxiliaryFailures -eq 0)
Assert-Case 'c16 s2s_within_period' ($m.MaximumStartToStartMilliseconds -le 60.0)
# Timer coalescing may push the pending-claim sample past the period; aside from a possible
# overlap violation (accounted in either direction, as in case 2) nothing else may fail.
Assert-Case 'c16 violations_only_overlap' (@($m.Violations|Where-Object {$_.reason -ne 'SAMPLE_OVERLAPS_NEXT_DEADLINE'}).Count -eq 0)
Assert-Case 'c16 evidence_exists' (Test-Path -LiteralPath $evidencePath)
$evidenceLines=@([IO.File]::ReadAllLines($evidencePath))
Assert-Case 'c16 evidence_line_count' ($evidenceLines.Count -eq 2 -and $evidenceLines.Count -eq $m.Samples.Count)
$parsedFirst=$evidenceLines[0]|ConvertFrom-Json
$parsedSecond=$evidenceLines[1]|ConvertFrom-Json
Assert-Case 'c16 evidence_parses' ($parsedFirst.sample_index -eq 0 -and $parsedSecond.sample_index -eq 1)
Assert-Case 'c16 evidence_phase' ($parsedFirst.phase -eq 'case16' -and $parsedSecond.phase -eq 'case16')
$evidenceBytes=[IO.File]::ReadAllBytes($evidencePath)
$bomPresent=($evidenceBytes.Length -ge 3 -and $evidenceBytes[0] -eq 0xEF -and $evidenceBytes[1] -eq 0xBB -and $evidenceBytes[2] -eq 0xBF)
Assert-Case 'c16 evidence_no_bom' ($bomPresent -eq $false)
Complete-Case -Number 16 -Name 'evidence_competing_for_adb' -Monitor $m

# ---------------------------------------------------------------- one realistic 2000 ms period run
# Lead 50 ms (module default is 25 ms): this host's Start-Sleep wakes are coalesced to timer
# boundaries (observed p50 ~15.5 ms, max ~31 ms for a 5 ms step), and the lead is exactly the
# knob that compensates that granularity so the MEASURED start-to-start stays <= the period.
Reset-FakeDevice -Latency @{competition=2;awake=2;top=2}
Reset-Probes
$realistic=New-PhoneLmForegroundMonitor -PeriodMilliseconds 2000 -LeadMilliseconds 50 -CommandTimeoutSeconds 8 -AuxiliaryPeriodMilliseconds 6000 -AuxiliaryCollectBoundMilliseconds 4000
for($i=0;$i -lt 4;$i++){
 $waitResult=Wait-PhoneLmMonitorDeadline -Monitor $realistic
 Assert-Case "realistic.wait[$i]" ($waitResult -eq 'DEADLINE_REACHED')
 $null=Invoke-CaseSample -Monitor $realistic -Phase 'realistic'
}
$realisticSummary=Get-PhoneLmMonitorSummary -Monitor $realistic
Assert-DeadlineAccounting -Monitor $realistic -CaseName 'realistic'
Assert-Case 'realistic.sample_count' ($realisticSummary.sample_count -eq 4)
foreach($startToStart in @($realistic.StartToStartMilliseconds)){
 Assert-Case 'realistic.s2s_within_period' ($startToStart -le 2000.0)
}
Assert-Case 'realistic.duration_within_period' ($realistic.MaximumSampleDurationMilliseconds -le 2000.0)
Assert-Case 'realistic.contract' ($realisticSummary.contract_met -eq $true -and $realisticSummary.monitoring_status -eq 'PASS')
Assert-Case 'realistic.counters_zero' ($realisticSummary.missed_deadlines -eq 0 -and $realisticSummary.command_timeouts -eq 0 -and $realisticSummary.invalid_samples -eq 0 -and $realisticSummary.unknown_states -eq 0 -and $realisticSummary.violation_count -eq 0)
$realisticMaxMilliseconds=[double]$realistic.MaximumStartToStartMilliseconds

# ---------------------------------------------------------------- report
$identity=[ordered]@{
 kind='G1_BACKWARD_FORMAL_MONITOR_HOST_TESTS'
 cases=16
 selftest_cases=[int]$selfTest.Cases
 realistic_period_milliseconds=2000
 realistic_maximum_start_to_start_milliseconds=$realisticMaxMilliseconds
 device_calls=0
 no_process_termination=$true
 report_directory=$out
}
$identity|ConvertTo-Json -Depth 4|Out-File -LiteralPath (Join-Path $out 'host-tests-identity.json') -Encoding utf8

$maxText=$realisticMaxMilliseconds.ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)
$moduleSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host ('monitor_module_sha256={0}' -f $moduleSha256)
Write-Host ('backward_formal_monitor_host=PASS cases=16 selftest_cases={0} realistic_period=2000 max_start_to_start_ms={1} device_calls=0' -f $selfTest.Cases,$maxText)
