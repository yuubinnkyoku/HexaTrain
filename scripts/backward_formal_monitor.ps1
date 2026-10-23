# Dot-sourceable G1 backward formal foreground monitor scheduler module.
# Host-only: no device calls, no process termination, no exit, no top-level side effects.
# Scheduling decisions use only the monotonic Stopwatch clock (GetTimestamp/Frequency).
# Wall-clock UTC may appear only as an auxiliary annotation and never influences a decision.
# Safe to dot-source under 'Set-StrictMode -Version Latest': every property read here is
# pre-initialised on objects this module creates, and caller objects are read defensively.

function New-PhoneLmMonitorClock {
 return [pscustomobject][ordered]@{
  Frequency=[long][Diagnostics.Stopwatch]::Frequency
  Origin=[long][Diagnostics.Stopwatch]::GetTimestamp()
 }
}

function Get-PhoneLmMonitorMilliseconds {
 param([Parameter(Mandatory=$true)][object]$Clock)
 $ticks=[long]([Diagnostics.Stopwatch]::GetTimestamp() - [long]$Clock.Origin)
 return [double]$ticks * 1000.0 / [double]$Clock.Frequency
}

# Defensive member read: works for PSCustomObject, hashtables and ordered dictionaries
# without tripping StrictMode on missing properties.
function Get-PhoneLmMonitorMember {
 param([object]$Object,[string]$Name,[object]$Default=$null)
 if($null -eq $Object){return $Default}
 if($Object -is [System.Collections.IDictionary]){
  if($Object.Contains($Name)){return $Object[$Name]}
  return $Default
 }
 $property=$Object.PSObject.Properties[$Name]
 if($null -eq $property){return $Default}
 return $property.Value
}

function New-PhoneLmForegroundMonitor {
 param(
  [int]$PeriodMilliseconds=2000,
  [int]$LeadMilliseconds=25,
  [int]$CommandTimeoutSeconds=8,
  [int]$AuxiliaryPeriodMilliseconds=6000,
  [int]$AuxiliaryCollectBoundMilliseconds=4000
 )
 if($PeriodMilliseconds -le 0){throw 'MONITOR_PERIOD_MUST_BE_POSITIVE'}
 if($LeadMilliseconds -lt 0){throw 'MONITOR_LEAD_MUST_BE_NON_NEGATIVE'}
 if($CommandTimeoutSeconds -le 0){throw 'MONITOR_COMMAND_TIMEOUT_MUST_BE_POSITIVE'}
 if($AuxiliaryPeriodMilliseconds -le 0){throw 'MONITOR_AUXILIARY_PERIOD_MUST_BE_POSITIVE'}
 if($AuxiliaryCollectBoundMilliseconds -le 0){throw 'MONITOR_AUXILIARY_BOUND_MUST_BE_POSITIVE'}
 return [pscustomobject][ordered]@{
  PeriodMilliseconds=[double]$PeriodMilliseconds
  LeadMilliseconds=[double]$LeadMilliseconds
  CommandTimeoutSeconds=[int]$CommandTimeoutSeconds
  AuxiliaryPeriodMilliseconds=[double]$AuxiliaryPeriodMilliseconds
  AuxiliaryCollectBoundMilliseconds=[double]$AuxiliaryCollectBoundMilliseconds
  Clock=(New-PhoneLmMonitorClock)
  StartMilliseconds=$null
  NextSampleIndex=0
  ScheduledDeadlineMilliseconds=$null
  PreviousSampleStartMilliseconds=$null
  PreviousSampleCompleteMilliseconds=$null
  MinimumStartToStartMilliseconds=$null
  MaximumStartToStartMilliseconds=[double]0
  MaximumSampleDurationMilliseconds=[double]0
  StartToStartMilliseconds=[System.Collections.Generic.List[double]]::new()
  SampleDurations=[System.Collections.Generic.List[double]]::new()
  LatenessMilliseconds=[System.Collections.Generic.List[double]]::new()
  Samples=[System.Collections.Generic.List[object]]::new()
  Violations=[System.Collections.Generic.List[object]]::new()
  MissedDeadlines=0
  CommandTimeouts=0
  InvalidSamples=0
  UnknownStates=0
  AuxiliarySamples=[System.Collections.Generic.List[object]]::new()
  AuxiliaryFailures=0
  AuxiliaryPending=$null
  AuxiliaryPendingStartedMilliseconds=$null
  AuxiliaryLastStartMilliseconds=$null
  CompletedMilliseconds=$null
  Status='READY'
 }
}

function Get-PhoneLmMonitorDeadline {
 # Pure deadline arithmetic (no sleeping). Sample index 0 launches at StartMilliseconds,
 # bootstrapped from the first monotonic reading on first use; for index k>=1 the deadline is
 # PreviousSampleStart + Period - Lead. The lead compensates host timer granularity and probe
 # drift so that the MEASURED start-to-start interval never exceeds PeriodMilliseconds; the
 # registered 2 s requirement itself is not relaxed (Lead is subtracted, never added).
 param([Parameter(Mandatory=$true)][object]$Monitor)
 if([int]$Monitor.NextSampleIndex -le 0){
  if($null -eq $Monitor.StartMilliseconds){
   $Monitor.StartMilliseconds=Get-PhoneLmMonitorMilliseconds -Clock $Monitor.Clock
  }
  return [double]$Monitor.StartMilliseconds
 }
 return [double]($Monitor.PreviousSampleStartMilliseconds + $Monitor.PeriodMilliseconds - $Monitor.LeadMilliseconds)
}

function Wait-PhoneLmMonitorDeadline {
 param(
  [Parameter(Mandatory=$true)][object]$Monitor,
  [object]$OwnedExitProbe=$null
 )
 $deadline=Get-PhoneLmMonitorDeadline -Monitor $Monitor
 while($true){
  if($null -ne $OwnedExitProbe){
   if((& $OwnedExitProbe) -eq $true){return 'OWNED_EXIT'}
  }
  $now=Get-PhoneLmMonitorMilliseconds -Clock $Monitor.Clock
  if($now -ge $deadline){return 'DEADLINE_REACHED'}
  $remaining=$deadline-$now
  $stepMilliseconds=[Math]::Ceiling([Math]::Min(5.0,$remaining))
  if($stepMilliseconds -lt 1){$stepMilliseconds=1}
  Start-Sleep -Milliseconds ([int]$stepMilliseconds)
 }
}

function Test-PhoneLmMonitorFailedClassification {
 param([object]$Classification)
 if($null -eq $Classification){return $false}
 $text=[string]$Classification
 return [bool]($text -match '(?i)timeout|fail|error|unavailable|critical|blocked|abort|disconnect')
}

function Invoke-PhoneLmMonitorSample {
 [CmdletBinding()]
 param(
  [Parameter(Mandatory=$true)][object]$Monitor,
  [Parameter(Mandatory=$true)][scriptblock]$AdbInvoker,
  [Parameter(Mandatory=$true)][scriptblock]$TopProbe,
  [Parameter(Mandatory=$true)][scriptblock]$AwakeProbe,
  [Parameter(Mandatory=$true)][scriptblock]$CompetitionProbe,
  [scriptblock]$AuxiliaryStarter,
  [scriptblock]$AuxiliaryCollector,
  [string]$EvidencePath,
  [string]$Phase,
  [switch]$PreserveText
 )
 $m=$Monitor
 $record=[ordered]@{}
 # First statement of the sample: monotonic sample start.
 $sampleStartMilliseconds=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
 $scheduledDeadline=$m.ScheduledDeadlineMilliseconds
 if($null -eq $scheduledDeadline){
  $scheduledDeadline=Get-PhoneLmMonitorDeadline -Monitor $m
 }
 $record.sample_index=[int]$m.NextSampleIndex
 $record.phase=$Phase
 $record.scheduled_deadline_milliseconds=[double]$scheduledDeadline
 $record.sample_start=$sampleStartMilliseconds
 $record.sample_start_utc=[DateTime]::UtcNow.ToString('o') # auxiliary annotation only; never a decision input
 $record.sample_complete=$null
 $record.duration_milliseconds=$null
 $record.lateness_milliseconds=($sampleStartMilliseconds-[double]$scheduledDeadline)
 $record.start_to_start_milliseconds=$null
 $record.state='SAMPLE_STATE_EVALUATED'
 $record.command_exception=$false
 $record.commands=[ordered]@{}
 $record.violations=@()
 $record.auxiliary=$null

 $violations=[System.Collections.Generic.List[object]]::new()
 $unknownObserved=$false
 $invalidObserved=$false

 # Fixed command order, identical to the historical runner.
 $specs=@(
  [pscustomobject]@{name='competition';arguments=[string[]]@('shell','ps','-A','-o','PID,PPID,USER,NAME,ARGS');probe=$CompetitionProbe},
  [pscustomobject]@{name='awake';arguments=[string[]]@('shell','dumpsys','power');probe=$AwakeProbe},
  [pscustomobject]@{name='top';arguments=[string[]]@('shell','dumpsys','activity','activities');probe=$TopProbe}
 )
 $executions=[System.Collections.Generic.List[object]]::new()

 foreach($spec in $specs){
  $entry=[ordered]@{
   arguments=$spec.arguments
   command_start=$null
   command_end=$null
   exit_code=$null
   classification=$null
   timeout=$false
   error_message=$null
   text_length=0
   text_sha256=$null
   text=$null
   probe=$null
  }
  $execution=[pscustomobject]@{name=$spec.name;entry=$entry;commandFailed=$false;probeResult=$null;probeError=$null}
  $entry.command_start=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
  $text=''
  try {
   $result=& $AdbInvoker $spec.arguments $m.CommandTimeoutSeconds
   $exitCode=Get-PhoneLmMonitorMember -Object $result -Name 'ExitCode' -Default $null
   $rawText=Get-PhoneLmMonitorMember -Object $result -Name 'Text' -Default ''
   $classification=Get-PhoneLmMonitorMember -Object $result -Name 'Classification' -Default 'ADB_CLASSIFICATION_UNKNOWN'
   if($null -eq $rawText){$rawText=''}
   $text=[string]$rawText
   $entry.exit_code=$exitCode
   $entry.classification=[string]$classification
   $isTimeout=(($null -ne $exitCode) -and ("$exitCode" -eq '124')) -or ([string]$classification -eq 'ADB_TRANSPORT_TIMEOUT')
   $entry.timeout=[bool]$isTimeout
   if($entry.timeout){$m.CommandTimeouts=[int]$m.CommandTimeouts+1}
   $utf8=[Text.Encoding]::UTF8.GetBytes($text)
   $entry.text_length=[int]$utf8.Length
   $sha=[Security.Cryptography.SHA256]::Create()
   try {$entry.text_sha256=([BitConverter]::ToString($sha.ComputeHash($utf8)) -replace '-','').ToLowerInvariant()}
   finally {$sha.Dispose()}
   if($PreserveText){$entry.text=$text}
  } catch {
   $execution.commandFailed=$true
   $entry.exit_code=$null
   $entry.classification='MONITOR_COMMAND_EXCEPTION'
   $entry.timeout=$false
   $entry.error_message=$_.Exception.Message
   $record.command_exception=$true
   $violations.Add([pscustomobject]@{reason='MONITOR_COMMAND_EXCEPTION';command=$spec.name;message=$_.Exception.Message})
  }
  $entry.command_end=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock

  if(!$execution.commandFailed){
   try {
    $execution.probeResult=& $spec.probe $text
   } catch {
    $execution.probeError=$_.Exception.Message
    $entry.probe=[ordered]@{ok=$false;known=$false;reason='PROBE_EVALUATION_EXCEPTION';detail=$_.Exception.Message}
   }
  } else {
   $entry.probe=[ordered]@{ok=$false;known=$false;reason='COMMAND_FAILED_SEPARATELY_ACCOUNTED';detail=$entry.error_message}
  }
  $record.commands[$spec.name]=$entry
  $executions.Add($execution)
 }

 # State evaluation happens AFTER the commands. Fail closed: unknown is never healthy.
 foreach($execution in $executions){
  $entry=$execution.entry
  if($execution.commandFailed){
   $invalidObserved=$true
   $m.InvalidSamples=[int]$m.InvalidSamples+1
   continue
  }
  if($null -ne $execution.probeError){
   $unknownObserved=$true
   $m.UnknownStates=[int]$m.UnknownStates+1
   $violations.Add([pscustomobject]@{reason='SAMPLE_EVALUATION_FAILURE';command=$execution.name;message=$execution.probeError})
   continue
  }
  $probeResult=$execution.probeResult
  $known=[bool](Get-PhoneLmMonitorMember -Object $probeResult -Name 'Known' -Default $false)
  $ok=[bool](Get-PhoneLmMonitorMember -Object $probeResult -Name 'Ok' -Default $false)
  $reason=[string](Get-PhoneLmMonitorMember -Object $probeResult -Name 'Reason' -Default 'PROBE_RESULT_INCOMPLETE')
  $detail=Get-PhoneLmMonitorMember -Object $probeResult -Name 'Detail' -Default $null
  if($null -eq $entry.probe -or [string]$entry.probe.reason -ne 'PROBE_EVALUATION_EXCEPTION'){
   $entry.probe=[ordered]@{ok=$ok;known=$known;reason=$reason;detail=$detail}
  }
  if(!$known){
   $unknownObserved=$true
   $m.UnknownStates=[int]$m.UnknownStates+1
   $violations.Add([pscustomobject]@{reason='SAMPLE_STATE_UNKNOWN';command=$execution.name;probe_reason=$reason})
  }
  if(!$ok){
   $invalidObserved=$true
   $m.InvalidSamples=[int]$m.InvalidSamples+1
   $violations.Add([pscustomobject]@{reason='SAMPLE_STATE_INVALID';command=$execution.name;probe_reason=$reason})
  }
 }

 # Auxiliary handling, in the specified order: collect a pending claim first, then maybe start one.
 $auxiliaryRecord=$null
 if($null -ne $m.AuxiliaryPending -and $null -ne $AuxiliaryCollector){
  $pending=$m.AuxiliaryPending
  $claimStart=Get-PhoneLmMonitorMember -Object $pending -Name 'StartedMilliseconds' -Default $null
  if($null -eq $claimStart){$claimStart=$m.AuxiliaryPendingStartedMilliseconds}
  if($null -eq $claimStart){$claimStart=$m.AuxiliaryLastStartMilliseconds}
  $completed=$false
  $classification=$null
  $collectorError=$null
  try {
   $collectorResult=& $AuxiliaryCollector $pending
   $completed=[bool](Get-PhoneLmMonitorMember -Object $collectorResult -Name 'Complete' -Default $false)
   $classification=Get-PhoneLmMonitorMember -Object $collectorResult -Name 'Classification' -Default $null
  } catch {
   $collectorError=$_.Exception.Message
  }
  $collectedAt=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
  $auxiliaryRecord=[ordered]@{
   phase='collect'
   started_milliseconds=[double]$claimStart
   collected_milliseconds=$collectedAt
   complete=$completed
   classification=$classification
   result=$null
   error_message=$collectorError
  }
  if($null -ne $collectorError){
   $auxiliaryRecord.result='AUXILIARY_SAMPLE_FAILED'
   $m.AuxiliaryFailures=[int]$m.AuxiliaryFailures+1
   $violations.Add([pscustomobject]@{reason='AUXILIARY_SAMPLE_FAILED';classification='MONITOR_AUXILIARY_COLLECTOR_EXCEPTION';message=$collectorError})
   $m.AuxiliaryPending=$null
   $m.AuxiliaryPendingStartedMilliseconds=$null
  } elseif(!$completed){
   $elapsed=$collectedAt-[double]$claimStart
   if($elapsed -gt $m.AuxiliaryCollectBoundMilliseconds){
    $auxiliaryRecord.result='AUXILIARY_COLLECTION_TIMEOUT'
    $m.AuxiliaryFailures=[int]$m.AuxiliaryFailures+1
    $violations.Add([pscustomobject]@{reason='AUXILIARY_COLLECTION_TIMEOUT';elapsed_milliseconds=$elapsed;bound_milliseconds=$m.AuxiliaryCollectBoundMilliseconds})
    $m.AuxiliaryPending=$null
    $m.AuxiliaryPendingStartedMilliseconds=$null
   } else {
    $auxiliaryRecord.result='AUXILIARY_PENDING'
   }
  } else {
   $auxiliaryRecord.result='AUXILIARY_SAMPLE_RECORDED'
   $m.AuxiliarySamples.Add([pscustomobject][ordered]@{
    classification=$classification
    complete=$true
    started_milliseconds=[double]$claimStart
    collected_milliseconds=$collectedAt
   })
   if(Test-PhoneLmMonitorFailedClassification -Classification $classification){
    $m.AuxiliaryFailures=[int]$m.AuxiliaryFailures+1
    $violations.Add([pscustomobject]@{reason='AUXILIARY_SAMPLE_FAILED';classification=[string]$classification})
   }
   $m.AuxiliaryPending=$null
   $m.AuxiliaryPendingStartedMilliseconds=$null
  }
 }
 if($null -eq $m.AuxiliaryPending -and $null -ne $AuxiliaryStarter){
  $nowAux=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
  if($null -eq $m.AuxiliaryLastStartMilliseconds -or ($nowAux-$m.AuxiliaryLastStartMilliseconds) -ge $m.AuxiliaryPeriodMilliseconds){
   try {
    $claim=& $AuxiliaryStarter $m
    $claimStart=Get-PhoneLmMonitorMember -Object $claim -Name 'StartedMilliseconds' -Default $nowAux
    $m.AuxiliaryPending=$claim
    $m.AuxiliaryPendingStartedMilliseconds=[double]$claimStart
    $m.AuxiliaryLastStartMilliseconds=$nowAux
    $auxiliaryRecord=[ordered]@{
     phase='start'
     started_milliseconds=[double]$claimStart
     collected_milliseconds=$null
     complete=$false
     classification=$null
     result='AUXILIARY_CLAIM_STARTED'
     error_message=$null
    }
   } catch {
    $m.AuxiliaryFailures=[int]$m.AuxiliaryFailures+1
    $violations.Add([pscustomobject]@{reason='AUXILIARY_SAMPLE_FAILED';classification='MONITOR_AUXILIARY_START_FAILED';message=$_.Exception.Message})
   }
  }
 }
 $record.auxiliary=$auxiliaryRecord

 # sample_complete is recorded after evaluation (state + auxiliary) completes.
 $sampleCompleteMilliseconds=Get-PhoneLmMonitorMilliseconds -Clock $m.Clock
 $record.sample_complete=$sampleCompleteMilliseconds

 $duration=$sampleCompleteMilliseconds-$sampleStartMilliseconds
 $record.duration_milliseconds=$duration
 $m.SampleDurations.Add([double]$duration)
 if($duration -gt $m.PeriodMilliseconds){
  $m.MissedDeadlines=[int]$m.MissedDeadlines+1
  $violations.Add([pscustomobject]@{reason='SAMPLE_OVERLAPS_NEXT_DEADLINE';duration_milliseconds=$duration;period_milliseconds=$m.PeriodMilliseconds})
 }
 if($duration -gt $m.MaximumSampleDurationMilliseconds){$m.MaximumSampleDurationMilliseconds=[double]$duration}

 if($null -ne $m.PreviousSampleStartMilliseconds){
  $startToStart=$sampleStartMilliseconds-$m.PreviousSampleStartMilliseconds
  $record.start_to_start_milliseconds=$startToStart
  $m.StartToStartMilliseconds.Add([double]$startToStart)
  if($null -eq $m.MinimumStartToStartMilliseconds -or $startToStart -lt $m.MinimumStartToStartMilliseconds){
   $m.MinimumStartToStartMilliseconds=[double]$startToStart
  }
  if($startToStart -gt $m.MaximumStartToStartMilliseconds){$m.MaximumStartToStartMilliseconds=[double]$startToStart}
  if($startToStart -gt $m.PeriodMilliseconds){
   $m.MissedDeadlines=[int]$m.MissedDeadlines+1
   $violations.Add([pscustomobject]@{reason='START_TO_START_EXCEEDED';start_to_start_milliseconds=$startToStart;period_milliseconds=$m.PeriodMilliseconds})
  }
 }

 $m.LatenessMilliseconds.Add([double]$record.lateness_milliseconds)
 if($unknownObserved -and !$invalidObserved){$record.state='SAMPLE_STATE_UNKNOWN'}
 elseif($invalidObserved){$record.state='SAMPLE_STATE_INVALID'}
 else {$record.state='SAMPLE_STATE_HEALTHY'}
 $record.violations=@($violations)

 foreach($violation in $violations){$m.Violations.Add($violation)}
 $m.Samples.Add([pscustomobject]$record)
 $m.PreviousSampleStartMilliseconds=$sampleStartMilliseconds
 $m.PreviousSampleCompleteMilliseconds=$sampleCompleteMilliseconds
 $m.NextSampleIndex=[int]$m.NextSampleIndex+1
 $m.ScheduledDeadlineMilliseconds=Get-PhoneLmMonitorDeadline -Monitor $m

 if(![string]::IsNullOrEmpty($EvidencePath)){
  try {
   $line=ConvertTo-Json -InputObject ([pscustomobject]$record) -Depth 8 -Compress
   [IO.File]::AppendAllText($EvidencePath, $line+"`n", [Text.UTF8Encoding]::new($false))
  } catch {
   throw ('MONITOR_EVIDENCE_WRITE_FAILED: '+$_.Exception.Message)
  }
 }
 return [pscustomobject]$record
}

function Get-PhoneLmMonitorMedian {
 # Existing convention: sorted values, index = count/2 (integer division).
 param([System.Collections.Generic.List[double]]$Values)
 if($null -eq $Values -or $Values.Count -eq 0){return $null}
 $sorted=@($Values | Sort-Object)
 return [double]$sorted[[int]($sorted.Count/2)]
}

function Get-PhoneLmMonitorSummary {
 param([Parameter(Mandatory=$true)][object]$Monitor)
 $m=$Monitor
 $sampleCount=[int]$m.Samples.Count
 $firstStart=$null
 $lastComplete=$null
 if($sampleCount -gt 0){
  $firstStart=$m.Samples[0].sample_start
  $lastComplete=$m.Samples[$sampleCount-1].sample_complete
 }
 $latenessMinimum=$null
 $latenessMedian=$null
 $latenessMaximum=$null
 if($m.LatenessMilliseconds.Count -gt 0){
  $sorted=@($m.LatenessMilliseconds | Sort-Object)
  $latenessMinimum=[double]$sorted[0]
  $latenessMaximum=[double]$sorted[$sorted.Count-1]
  $latenessMedian=[double]$sorted[[int]($sorted.Count/2)]
 }
 $violationReasons=@($m.Violations | ForEach-Object {[string](Get-PhoneLmMonitorMember -Object $_ -Name 'reason' -Default 'UNKNOWN')} | Sort-Object -Unique)
 # contract_met: sample_count>=1, all failure counters 0 (missed_deadlines, command_timeouts,
 # invalid_samples, unknown_states, auxiliary_failures plus violation_count checked separately),
 # maximum start-to-start <= period, maximum sample duration <= period.
 $contractMet=(
  $sampleCount -ge 1 -and
  [int]$m.MissedDeadlines -eq 0 -and
  [int]$m.CommandTimeouts -eq 0 -and
  [int]$m.InvalidSamples -eq 0 -and
  [int]$m.UnknownStates -eq 0 -and
  [int]$m.AuxiliaryFailures -eq 0 -and
  $m.Violations.Count -eq 0 -and
  $m.MaximumStartToStartMilliseconds -le $m.PeriodMilliseconds -and
  $m.MaximumSampleDurationMilliseconds -le $m.PeriodMilliseconds
 )
 return [ordered]@{
  schema=1
  kind='G1_BACKWARD_FORMAL_MONITOR_SUMMARY'
  period_milliseconds=[double]$m.PeriodMilliseconds
  lead_milliseconds=[double]$m.LeadMilliseconds
  command_timeout_seconds=[int]$m.CommandTimeoutSeconds
  auxiliary_period_milliseconds=[double]$m.AuxiliaryPeriodMilliseconds
  sample_count=$sampleCount
  first_sample_start_milliseconds=$firstStart
  last_sample_complete_milliseconds=$lastComplete
  minimum_start_to_start_milliseconds=$m.MinimumStartToStartMilliseconds
  median_start_to_start_milliseconds=(Get-PhoneLmMonitorMedian -Values $m.StartToStartMilliseconds)
  maximum_start_to_start_milliseconds=[double]$m.MaximumStartToStartMilliseconds
  maximum_sample_duration_milliseconds=[double]$m.MaximumSampleDurationMilliseconds
  lateness_milliseconds=@{minimum=$latenessMinimum;median=$latenessMedian;maximum=$latenessMaximum}
  missed_deadlines=[int]$m.MissedDeadlines
  command_timeouts=[int]$m.CommandTimeouts
  invalid_samples=[int]$m.InvalidSamples
  unknown_states=[int]$m.UnknownStates
  auxiliary_samples=[int]$m.AuxiliarySamples.Count
  auxiliary_failures=[int]$m.AuxiliaryFailures
  violation_count=$m.Violations.Count
  violation_reasons=$violationReasons
  monitoring_status=$(if($contractMet){'PASS'}else{'CONTRACT_VIOLATION'})
  contract_met=[bool]$contractMet
 }
}

function Complete-PhoneLmMonitor {
 param(
  [Parameter(Mandatory=$true)][object]$Monitor,
  [Parameter(Mandatory=$true)][ValidateSet('STOPPED','ABORTED')][string]$Status
 )
 $Monitor.Status=$Status
 $Monitor.CompletedMilliseconds=Get-PhoneLmMonitorMilliseconds -Clock $Monitor.Clock
 return $Monitor
}

function Write-PhoneLmMonitorSummary {
 param(
  [Parameter(Mandatory=$true)][object]$Monitor,
  [Parameter(Mandatory=$true)][string]$Path
 )
 if(Test-Path -LiteralPath $Path){throw 'MONITOR_SUMMARY_ALREADY_EXISTS'}
 $summary=Get-PhoneLmMonitorSummary -Monitor $Monitor
 $json=ConvertTo-Json -InputObject ([pscustomobject]$summary) -Depth 8
 $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
 try {
  $bytes=[Text.UTF8Encoding]::new($false).GetBytes($json)
  $stream.Write($bytes,0,$bytes.Length)
 } finally {$stream.Dispose()}
 return $Path
}

function Test-PhoneLmMonitorSelfTest {
 # Pure / deadline-arithmetic assertions only. No device calls. Target runtime well under 5 s.
 $state=[pscustomobject]@{Cases=0;Failures=[System.Collections.Generic.List[string]]::new()}
 $assert={
  param([bool]$Condition,[string]$Name)
  $state.Cases=[int]$state.Cases+1
  if(!$Condition){$state.Failures.Add($Name)}
 }
 $instantText={
  param([string[]]$Arguments,$TimeoutSeconds)
  $key='competition'
  if($Arguments -contains 'power'){$key='awake'}
  elseif($Arguments -contains 'activities'){$key='top'}
  $text=" 1000 1000 system surfaceflinger /system/bin/surfaceflinger`n"
  if($key -eq 'awake'){$text=" mWakefulness=Awake`n"}
  if($key -eq 'top'){$text="  topResumedActivity=ActivityRecord{a u0 com.example.home/.Launcher t1}`n"}
  return [pscustomobject]@{ExitCode=0;Text=$text;Classification='ADB_OK'}
 }
 $healthyProbe={
  param($text)
  return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}
 }

 # Warm-up: the first invocation of the crypto/JSON/probe paths in a fresh process costs
 # ~150 ms of one-time JIT/initialisation. Without it, cold-start latency would be
 # misattributed as a real violation by these short-period arithmetic assertions.
 $warmupMonitor=New-PhoneLmForegroundMonitor -PeriodMilliseconds 400 -LeadMilliseconds 5
 $null=Invoke-PhoneLmMonitorSample -Monitor $warmupMonitor -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 1..3 | ForEach-Object { Start-Sleep -Milliseconds 5 }

 # Case 1: deadline arithmetic for index 0 bootstraps StartMilliseconds.
 $m1=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $deadline0=Get-PhoneLmMonitorDeadline -Monitor $m1
 & $assert (($null -ne $m1.StartMilliseconds) -and ($deadline0 -eq [double]$m1.StartMilliseconds)) 'deadline_index0'

 # Case 2: deadline arithmetic for index k>=1 is PreviousSampleStart + Period - Lead.
 $m2=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $m2.StartMilliseconds=1000.0
 $m2.PreviousSampleStartMilliseconds=1000.0
 $m2.NextSampleIndex=3
 & $assert ((Get-PhoneLmMonitorDeadline -Monitor $m2) -eq 1025.0) 'deadline_index_k'

 # Case 3: launch at index 0 reaches its deadline without sleeping past it.
 $m3=New-PhoneLmForegroundMonitor -PeriodMilliseconds 20 -LeadMilliseconds 5
 $m3.StartMilliseconds=Get-PhoneLmMonitorMilliseconds -Clock $m3.Clock
 $waitStart=Get-PhoneLmMonitorMilliseconds -Clock $m3.Clock
 $waitResult=Wait-PhoneLmMonitorDeadline -Monitor $m3
 $waitEnd=Get-PhoneLmMonitorMilliseconds -Clock $m3.Clock
 & $assert (($waitResult -eq 'DEADLINE_REACHED') -and (($waitEnd-$waitStart) -lt 200)) 'launch_index0_wait'

 # Case 4: lateness sign, early start (deadline in the future) is negative.
 $m4=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $m4.ScheduledDeadlineMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m4.Clock)+60.0
 $r4=Invoke-PhoneLmMonitorSample -Monitor $m4 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 & $assert (($r4.lateness_milliseconds -lt 0) -and ($r4.violations.Count -eq 0)) 'lateness_early_negative'

 # Case 5: lateness sign, late start (deadline already passed) is positive.
 $m5=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $m5.ScheduledDeadlineMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $m5.Clock)-25.0
 $r5=Invoke-PhoneLmMonitorSample -Monitor $m5 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 & $assert (($r5.lateness_milliseconds -gt 0) -and ($m5.LatenessMilliseconds.Count -eq 1)) 'lateness_late_positive'

 # Case 6: start-to-start violation detection.
 $m6=New-PhoneLmForegroundMonitor -PeriodMilliseconds 20 -LeadMilliseconds 5
 $null=Invoke-PhoneLmMonitorSample -Monitor $m6 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 Start-Sleep -Milliseconds 45
 $null=Invoke-PhoneLmMonitorSample -Monitor $m6 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $s6Reasons=@($m6.Violations | ForEach-Object {$_.reason})
 & $assert (($m6.MissedDeadlines -ge 1) -and ($s6Reasons -contains 'START_TO_START_EXCEEDED') -and ($m6.MaximumStartToStartMilliseconds -gt 20)) 'start_to_start_violation'

 # Case 7: duration-overlap detection (sample duration exceeds the period).
 $slowInvoker={
  param($Arguments,$TimeoutSeconds)
  Start-Sleep -Milliseconds 12
  return (& $instantText $Arguments $TimeoutSeconds)
 }
 $m7=New-PhoneLmForegroundMonitor -PeriodMilliseconds 20 -LeadMilliseconds 5
 $null=Invoke-PhoneLmMonitorSample -Monitor $m7 -AdbInvoker $slowInvoker -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $s7Reasons=@($m7.Violations | ForEach-Object {$_.reason})
 & $assert (($s7Reasons -contains 'SAMPLE_OVERLAPS_NEXT_DEADLINE') -and ($m7.MissedDeadlines -ge 1)) 'duration_overlap_detection'

 # Case 8: timeout accounting (ExitCode 124 / ADB_TRANSPORT_TIMEOUT).
 $timeoutInvoker={
  param($Arguments,$TimeoutSeconds)
  if($Arguments -contains 'power'){
   return [pscustomobject]@{ExitCode=124;Text='';Classification='ADB_TRANSPORT_TIMEOUT'}
  }
  return (& $instantText $Arguments $TimeoutSeconds)
 }
 $m8=New-PhoneLmForegroundMonitor -PeriodMilliseconds 40 -LeadMilliseconds 5
 $r8=Invoke-PhoneLmMonitorSample -Monitor $m8 -AdbInvoker $timeoutInvoker -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 & $assert (($m8.CommandTimeouts -ge 1) -and ($r8.commands.awake.timeout -eq $true) -and ($r8.commands.awake.exit_code -eq 124)) 'timeout_accounting'

 # Case 9: unknown state fail-closed accounting.
 $unknownProbe={
  param($text)
  return [pscustomobject]@{Ok=$true;Known=$false;Reason='TOP_RESUMED_UNKNOWN';Detail=$null}
 }
 $m9=New-PhoneLmForegroundMonitor -PeriodMilliseconds 40 -LeadMilliseconds 5
 $r9=Invoke-PhoneLmMonitorSample -Monitor $m9 -AdbInvoker $instantText -TopProbe $unknownProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $s9Reasons=@($r9.violations | ForEach-Object {$_.reason})
 & $assert (($m9.UnknownStates -ge 1) -and ($s9Reasons -contains 'SAMPLE_STATE_UNKNOWN') -and ($r9.state -eq 'SAMPLE_STATE_UNKNOWN')) 'unknown_state_fail_closed'

 # Case 10: invalid state fail-closed accounting.
 $invalidProbe={
  param($text)
  return [pscustomobject]@{Ok=$false;Known=$true;Reason='UNEXPECTED_FOREGROUND_PACKAGE';Detail=$null}
 }
 $m10=New-PhoneLmForegroundMonitor -PeriodMilliseconds 40 -LeadMilliseconds 5
 $r10=Invoke-PhoneLmMonitorSample -Monitor $m10 -AdbInvoker $instantText -TopProbe $invalidProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $s10Reasons=@($r10.violations | ForEach-Object {$_.reason})
 & $assert (($m10.InvalidSamples -ge 1) -and ($s10Reasons -contains 'SAMPLE_STATE_INVALID') -and ($r10.state -eq 'SAMPLE_STATE_INVALID')) 'invalid_state_fail_closed'

 # Case 11: evidence append failure must rethrow, never be silent.
 $m11=New-PhoneLmForegroundMonitor -PeriodMilliseconds 40 -LeadMilliseconds 5
 $badEvidencePath=Join-Path ([IO.Path]::GetTempPath()) ('phoneLM-missing-'+[guid]::NewGuid().ToString('N')+'\no\such\dir\monitor.jsonl')
 $evidenceThrew=$false
 $evidenceMessage=''
 try {
  $null=Invoke-PhoneLmMonitorSample -Monitor $m11 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe -EvidencePath $badEvidencePath
 } catch {
  $evidenceThrew=$true
  $evidenceMessage=$_.Exception.Message
 }
 & $assert ($evidenceThrew -and $evidenceMessage -match 'MONITOR_EVIDENCE_WRITE_FAILED') 'evidence_append_failure_rethrown'

 # Case 12: summary median uses the sorted index count/2 convention.
 $m12=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 foreach($value in 40.0,10.0,30.0,20.0){$m12.StartToStartMilliseconds.Add($value)}
 $summary12=Get-PhoneLmMonitorSummary -Monitor $m12
 & $assert (($summary12.median_start_to_start_milliseconds -eq 30.0) -and ($summary12.schema -eq 1) -and ($summary12.kind -eq 'G1_BACKWARD_FORMAL_MONITOR_SUMMARY')) 'summary_median_convention'

 # Case 13: contract_met true on a clean run.
 $m13=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $null=Invoke-PhoneLmMonitorSample -Monitor $m13 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $null=Invoke-PhoneLmMonitorSample -Monitor $m13 -AdbInvoker $instantText -TopProbe $healthyProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $summary13=Get-PhoneLmMonitorSummary -Monitor $m13
 & $assert (($summary13.contract_met -eq $true) -and ($summary13.monitoring_status -eq 'PASS') -and ($summary13.sample_count -eq 2)) 'contract_met_true'

 # Case 14: contract_met false on a violation run.
 $m14=New-PhoneLmForegroundMonitor -PeriodMilliseconds 30 -LeadMilliseconds 5
 $null=Invoke-PhoneLmMonitorSample -Monitor $m14 -AdbInvoker $instantText -TopProbe $invalidProbe -AwakeProbe $healthyProbe -CompetitionProbe $healthyProbe
 $summary14=Get-PhoneLmMonitorSummary -Monitor $m14
 & $assert (($summary14.contract_met -eq $false) -and ($summary14.monitoring_status -eq 'CONTRACT_VIOLATION') -and ($summary14.violation_count -ge 1)) 'contract_met_false'

 $failures=@($state.Failures)
 return [pscustomobject]@{
  Passed=($failures.Count -eq 0)
  Cases=[int]$state.Cases
  Failures=$failures
 }
}
