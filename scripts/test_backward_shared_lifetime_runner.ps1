# Host-only execution of the actual capture function with a fake ADB result.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$runner=Join-Path $PSScriptRoot 'run_backward_shared_lifetime_device.ps1'
$tokens=$null;$parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($runner,[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw 'RUNNER_PARSE_FAILURE'}
$capture=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Capture-Readonly'},$true)
if(!$capture){throw 'CAPTURE_FUNCTION_MISSING'}
Invoke-Expression $capture.Extent.Text
$out=Join-Path $root ('build/reports/g1-backward-shared-lifetime-diagnostic/runner-host-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($out)|Out-Null
$adb='fixture-adb';$d=@{Endpoint='fixture-endpoint'};$script:recorded=@();$script:mockExit=0
function Invoke-PhoneLmAdb {
 param([string]$Adb,[string]$Device,[string[]]$Arguments,[switch]$AllowFailure)
 if($Adb -ne 'fixture-adb' -or $Device -ne 'fixture-endpoint' -or !$AllowFailure -or !$Arguments.Count){throw 'CAPTURE_ARGUMENTS_LOST'}
 $script:recorded+=,@($Arguments)
 if($script:infraMode){return [pscustomobject]@{ExitCode=$script:infraExit;Text=$(if($Arguments[-1].EndsWith('/status')){$script:infraStatus}else{$script:infraCmdline})}}
 return [pscustomobject]@{ExitCode=$script:mockExit;Text='synthetic readonly response'}
}
$cases=@(
 @('logcat','-b','crash','-b','main','-b','system','-d','-t','4000'),
 @('shell','ls','-l','/data/tombstones'),
 @('shell','dumpsys','dropbox'),
 @('shell','dumpsys','dropbox','--print','SYSTEM_TOMBSTONE'),
 @('shell','cat','/data/tombstones/tombstone_00')
)
for($i=0;$i -lt $cases.Count;$i++){
 $r=Capture-Readonly "case-$i" $cases[$i]
 if($r.ExitCode -ne 0 -or ($script:recorded[$i] -join '|') -cne ($cases[$i] -join '|')){throw 'CAPTURE_ARGUMENT_ORDER'}
 $saved=Get-Content "$out/case-$i-private.json" -Raw|ConvertFrom-Json
 if($saved.exit -ne 0 -or $saved.text_private -ne 'synthetic readonly response'){throw 'CAPTURE_EVIDENCE_NOT_SAVED'}
}
$script:mockExit=1;$r=Capture-Readonly 'unavailable' @('shell','ls','/inaccessible')
if($r.ExitCode -ne 1 -or (Get-Content "$out/unavailable-private.json" -Raw|ConvertFrom-Json).exit -ne 1){throw 'FAILED_COLLECTION_MASKED'}
$infrastructure=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-VerifiedInfrastructure'},$true)
if(!$infrastructure){throw 'INFRASTRUCTURE_CLASSIFIER_MISSING'}
Invoke-Expression $infrastructure.Extent.Text
$script:infraMode=$true;$script:infraExit=0
$kernelLine='100 2 root [kworker/R-policy_probe_wq] [kworker/R-policy_probe_wq]'
$traceLine='200 1 nobody traced_probes traced_probes'
$kernelStatus="Name:`tkworker/R-policy_probe_wq`nPPid:`t2`nUid:`t0`t0`t0`t0`nKthread:`t1`n"
$traceStatus="Name:`ttraced_probes`nPPid:`t1`nUid:`t9999`t9999`t9999`t9999`nKthread:`t0`n"
foreach($case in @('kernel','trace','bad-cmdline','bad-kthread','bad-user','bad-parent','unreadable','unknown-probe','unknown-kworker')){
 $script:verifiedInfrastructure=@{};$script:infraExit=0;$line=$traceLine;$expected=$false
 $script:infraStatus=$traceStatus;$script:infraCmdline='/system/bin/traced_probes'+[char]0
 switch($case){
  'kernel' {$line=$kernelLine;$script:infraStatus=$kernelStatus;$script:infraCmdline='';$expected=$true}
  'trace' {$expected=$true}
  'bad-cmdline' {$script:infraCmdline='/data/local/tmp/npu-probe'+[char]0}
  'bad-kthread' {$line=$kernelLine;$script:infraStatus=$kernelStatus.Replace("Kthread:`t1","Kthread:`t0");$script:infraCmdline=''}
  'bad-user' {$line=$traceLine.Replace('nobody','root')}
  'bad-parent' {$line=$traceLine.Replace('200 1','200 2')}
  'unreadable' {$script:infraExit=1}
  'unknown-probe' {$line='300 1 shell training_probe training_probe'}
  'unknown-kworker' {$line='300 2 root [kworker/hmx_probe] [kworker/hmx_probe]'}
 }
 if((Test-VerifiedInfrastructure $line) -ne $expected){throw "INFRASTRUCTURE_CLASSIFICATION $case"}
 if($expected){$before=$script:recorded.Count;if(!(Test-VerifiedInfrastructure $line) -or $script:recorded.Count -ne $before){throw 'INFRASTRUCTURE_PIN_NOT_REUSED'}}
}
$text=Get-Content $runner -Raw
if($text -notmatch 'CRASH_LOG_COLLECTION_UNAVAILABLE' -or $text -notmatch 'CRASH_STREAM_NOT_RUNNING_NO_PROBE' -or
 $text -notmatch 'HOST_INVENTORY_REVIEW_TIMEOUT_NO_PROBE' -or $text -notmatch 'HOST_INVENTORY_REVIEW_NOT_BOUND' -or
 $text -notmatch 'probe-attempt-private.json",\[IO.FileMode\]::CreateNew' -or
 ([regex]::Matches($text,'\./saved_p_backward_probe double-inputs').Count -ne 1)) {throw 'ONE_SHOT_CAPTURE_GUARD_MISSING'}
Write-Host 'shared_lifetime_runner_host=PASS capture_cases=6 infrastructure_cases=9 device_calls=0 argument_binding=PASS failed_capture_preserved=PASS unknown_workloads_rejected=PASS process_cap=1'
