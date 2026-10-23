# Independent five-role runner, revision 3.
#
# v3 differs from v1/v2 in exactly two places, both host-side orchestration:
#  1. the in-run foreground monitor is the fixed absolute 2 s deadline scheduler in
#     scripts/backward_formal_monitor.ps1 instead of a 2000 ms WaitForExit added in
#     series with the query block;
#  2. post-exit evidence readback is the bounded, resumable, per-chunk verified
#     transport in scripts/backward_formal_transport.ps1 instead of one `adb pull`
#     bounded by a single 180 s wrapper timeout.
# No arithmetic, candidate, graph, fixture, observer or Backend/Device lifetime change.
param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot,
 [Parameter(Mandatory=$true)][string]$ReportRoot,[Parameter(Mandatory=$true)][string]$QnnValidationLog,
 [ValidateSet('shared-five-role','shared-performance')][string]$Mode='shared-five-role',
 [Parameter(Mandatory=$true)][ValidateSet('pilot','formal-1','formal-2','formal-3','performance')][string]$Stage,
 [ValidateSet(60)][int]$ForwardLimit=60,
 [ValidateSet('shared-lifetime-five-role-final')][string]$Candidate='shared-lifetime-five-role-final',[switch]$SelfTest)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
. (Join-Path $PSScriptRoot 'mixed_qkv_monitoring_policy.ps1')
. (Join-Path $PSScriptRoot 'backward_formal_monitor.ps1')
. (Join-Path $PSScriptRoot 'backward_formal_transport.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
function Test-SavedPStagingCollision([hashtable]$Hashes,[string]$Name,[string]$Hash) {
 return ($Hashes.ContainsKey($Name) -and $Hashes[$Name] -ne $Hash)
}
if($SelfTest) {
 $hashesTest=@{same='a'}
 if((Test-SavedPStagingCollision $hashesTest 'new' 'b') -or
    (Test-SavedPStagingCollision $hashesTest 'same' 'a') -or
    !(Test-SavedPStagingCollision $hashesTest 'same' 'b')){throw 'STAGING_COLLISION_SELFTEST'}
 $monitorSelfTest=Test-PhoneLmMonitorSelfTest
 if($monitorSelfTest.Failures.Count){throw ('MONITOR_SELFTEST_FAILED: '+($monitorSelfTest.Failures -join ','))}
 $transferSelfTest=Test-PhoneLmTransferSelfTest
 $transferFailures=@($transferSelfTest.Cases|Where-Object {-not $_.Passed})
 if($transferFailures.Count){throw ('TRANSPORT_SELFTEST_FAILED: '+(($transferFailures|ForEach-Object {$_.Name}) -join ','))}
 Write-Host ('saved_p_runner_v3_selftest=PASS staging_cases=3 monitor_cases={0} transport_cases={1} device_calls=0' -f $monitorSelfTest.Cases,$transferSelfTest.Cases.Count)
 exit 0
}
$root=Split-Path -Parent $PSScriptRoot;$out=[IO.Path]::GetFullPath($ReportRoot)
$mission=Join-Path $root 'build/reports/g1-backward-shared-lifetime-formal'
trap {
 $record=@{classification='MISSION_STOP';stage=$Stage;reason=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o');no_automatic_retry=$true;runner_revision=3}
 $stopPath=Join-Path $mission 'mission-stop-private.json'
 if(!(Test-Path $stopPath)){$stream=[IO.File]::Open($stopPath,[IO.FileMode]::CreateNew);try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($record|ConvertTo-Json));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}}
 throw $_
}
if($out -ne [IO.Path]::GetFullPath((Join-Path $mission $Stage))){throw 'EXACT_STAGE_DIRECTORY_REQUIRED'}
if(($Stage -eq 'performance') -ne ($Mode -eq 'shared-performance')){throw 'STAGE_MODE_SCOPE'}
# v3 is the repaired formal-1 runner. Pilot/formal-2/formal-3/performance remain bound
# to their already frozen runners; running them here would silently change their identity.
if($Stage -ne 'formal-1'){throw 'RUNNER_V3_SCOPE_FORMAL1_ONLY'}
& python -B "$PSScriptRoot/certify_backward_shared_lifetime_formal_v2.py" --entry $Stage
if($LASTEXITCODE){throw 'INDEPENDENT_PREVIOUS_STAGE_NOT_PASS'}
# Hard host entry gate: host repairs, monitor dry run, pilot numerics, device gate,
# prior attempt, committed source, runner and plan identity.
& python -B "$PSScriptRoot/assess_monitor_transport_recovery.py" --entry formal-1
if($LASTEXITCODE){throw 'MONITOR_TRANSPORT_ENTRY_NOT_PASS'}
if(!$out.StartsWith([IO.Path]::GetFullPath("$root/build/reports")+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_REPORT_REQUIRED'}
if(Test-Path $out){throw 'EXCLUSIVE_PROCESS_REPORT_REQUIRED'}
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$qnn=Get-Content $QnnValidationLog -Raw
if($qnn -notmatch '===== verify.ps1 -Profile Qnn =====' -or $qnn -notmatch 'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$'){throw 'CURRENT_QNN_GATE_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$approved="$root/build/reports/g1-s2-mixed-qkv-df-hvx-v/approved-apk/app-debug.apk"
# Fresh per invocation. No waiver for storage, no wake/unlock/install.
$preflightArgs=@('-NoProfile','-File',"$PSScriptRoot/preflight_forward_p_resume.ps1",'-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-HexagonSdkRoot',$HexagonSdkRoot,'-NdkRoot',$NdkRoot,'-ReportRoot',"$out/preflight",'-MixedQkvResearchMonitoring','-ExpectedApkPath',$approved)
if(@($preflightArgs|Where-Object {$_ -match '"'}).Count){throw 'PREFLIGHT_ARGUMENT_QUOTE_REJECTED'}
$quoted=@($preflightArgs|ForEach-Object {'"'+$_+'"'})
$ownedPreflight=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList $quoted -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/preflight-console.txt" -RedirectStandardError "$out/preflight-stderr.txt"
$preflightStarted=[DateTime]::UtcNow
while(!$ownedPreflight.WaitForExit(1000)){
 if(([DateTime]::UtcNow-$preflightStarted).TotalSeconds -ge 180){
  $ownedPreflight.Kill()
  @{classification='HOST_PREFLIGHT_TIMEOUT';device_probe_executes=0;only_owned_host_preflight_terminated=$true;shared_adb_server_untouched=$true}|ConvertTo-Json|Out-File "$out/preflight-timeout-private.json" -Encoding utf8
  throw 'HOST_PREFLIGHT_TIMEOUT_NO_DEVICE_EXECUTION'
 }
}
$ownedPreflight.Refresh()
if($ownedPreflight.ExitCode){throw 'BLOCKED_DEVICE_SAFETY'}
$preflight=Get-Content "$out/preflight/device-preflight-private.json" -Raw|ConvertFrom-Json
if($preflight.status -ne 'PASS'){throw 'PREFLIGHT_NOT_PASS'}
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe';$d=$preflight.device_private
function Adb {param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments);$r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments;if($r.ExitCode){throw 'ADB_OPERATION_FAILED'};return $r.Text}
# Explicit per-command bound: a 1.4 GB sha256sum or gzip on the device is legitimately
# slower than the 60 s default, and a silent default bound is what made the original
# readback unpublishable.
function AdbTimed {
 param([Parameter(Mandatory=$true)][string[]]$Arguments,[int]$TimeoutSeconds=120,[switch]$AllowFailure)
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds -AllowFailure:$AllowFailure
 return $r
}
$candidateDir="$root/build/fp32-saved-p-backward-$Candidate"
$build=Get-Content "$candidateDir/build-identity.json" -Raw|ConvertFrom-Json
foreach($p in $build.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'CANDIDATE_SOURCE_CHANGED'}}
$probe="$candidateDir/saved_p_backward_probe"
if((Get-FileHash $probe).Hash.ToLowerInvariant() -ne $build.probe_sha256){throw 'CANDIDATE_BINARY_CHANGED'}
if($build.candidate -ne 'shared-lifetime-five-role-final' -or $build.classification -ne 'FIVE_ROLE_REQUIRES_FRESH_QUALIFICATION'){throw 'FIVE_ROLE_CANDIDATE_REQUIRED'}
$files=@($probe,"$root/build/reports/g1-native-saved-p-backward-dsp/gradients.bin")
if($build.PSObject.Properties.Name -contains 'binary_hashes'){
 foreach($p in $build.binary_hashes.PSObject.Properties){if((Get-FileHash "$candidateDir/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'REPAIR_BINARY_CHANGED'};if($p.Name -like 'lib*.so'){$files+="$candidateDir/$($p.Name)"}}
}
# Research entry authorization, not a fabricated native accuracy certificate.
$permitFile=Join-Path $out 'native-permit-private.json'
@{purpose='USER_AUTHORIZED_FIVE_ROLE_RESEARCH_ENTRY_ONLY';stage=$Stage;probe_sha256=$build.probe_sha256;plan_sha256=(Get-FileHash "$root/host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json").Hash.ToLowerInvariant();not_formal_certification=$true;runner_revision=3}|ConvertTo-Json|Out-File $permitFile -Encoding utf8
$files+=$permitFile
if((Get-FileHash $files[1]).Hash.ToLowerInvariant() -ne $build.gradient_sha256){throw 'FROZEN_GRADIENT_CHANGED'}
foreach($name in @('fp32-custom','fp32-selective-ln1','fp32-mixed-qkv','fp32-double-qk')) {
 $bundle=Get-Content "$root/build/$name/build-identity.json" -Raw|ConvertFrom-Json
 foreach($p in $bundle.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FORWARD_SOURCE_CHANGED'}}
 foreach($p in $bundle.binary_hashes.PSObject.Properties){if((Get-FileHash "$root/build/$name/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FORWARD_BINARY_CHANGED'}}
 $files+=Get-ChildItem "$root/build/$name" -File|Where-Object {$_.Name -match '^lib.*\.so$|^(?:double|mixed|selective-ln1)-inputs\.bin$'}|Select-Object -ExpandProperty FullName
}
foreach($name in @('libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so')){$files+="$QairtSdkRoot/lib/aarch64-android/$name"}
$files+="$QairtSdkRoot/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
$files+="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
& "$PSScriptRoot/audit_qnn_apk.ps1" -ApkPath $approved -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath "$out/apk-audit.txt" *> "$out/apk-console.txt"
if((Get-Content "$out/apk-audit.txt" -Raw) -notmatch 'status=SUCCESS'){throw 'APK_AUDIT_FAILED'}
$leaf=Split-Path $out -Leaf
if($leaf -notmatch '^[a-zA-Z0-9-]{1,64}$'){throw 'UNSAFE_REMOTE_EVIDENCE_NAME'}
$remote='/data/local/tmp/hexasbp-'+$build.probe_sha256.Substring(0,16)+'-'+$leaf
Adb @('shell','mkdir','-p',$remote)|Out-Null;$hashes=@{}
foreach($file in $files) {
 $name=Split-Path $file -Leaf;$hash=(Get-FileHash $file).Hash.ToLowerInvariant()
 if(Test-SavedPStagingCollision $hashes $name $hash){throw 'STAGING_NAME_COLLISION'}
 Adb @('push',$file,$remote)|Out-Null
 if(((Adb @('shell','sha256sum',"$remote/$name")) -split '\s+')[0] -ne $hash){throw 'STAGED_HASH_MISMATCH'};$hashes[$name]=$hash
}
Adb @('shell','chmod','700',"$remote/saved_p_backward_probe")|Out-Null
# Pre-launch one-shot gates keep the exact v1 semantics and evidence shape.
function Awake {
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','dumpsys','power') -AllowFailure
 @{exit=$r.ExitCode;text_private=$r.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/awake-private.jsonl" -Encoding utf8
 return ($r.ExitCode -eq 0 -and $r.Text -match '(?m)^\s*mWakefulness=Awake\s*$')
}
function Top {
 $a=Adb @('shell','dumpsys','activity','activities')
 $m=[regex]::Match($a,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
 if(!$m.Success){
  try {
   $failure=[ordered]@{reason='TOP_RESUMED_UNKNOWN';observed_utc=[DateTime]::UtcNow.ToString('o');response_type=$(if($null -eq $a){'null'}else{$a.GetType().FullName});actual_response_private=$a;retry_performed=$false}
   $failurePath=Join-Path $out ('top-unknown-'+[guid]::NewGuid().ToString('N')+'-private.json')
   $failureStream=[IO.File]::Open($failurePath,[IO.FileMode]::CreateNew)
   try{$failureBytes=[Text.UTF8Encoding]::new($false).GetBytes(($failure|ConvertTo-Json -Depth 4));$failureStream.Write($failureBytes,0,$failureBytes.Length)}finally{$failureStream.Dispose()}
  }catch{}
  throw 'TOP_RESUMED_UNKNOWN'
 }
 @{known=$true;package_private=$m.Groups[1].Value;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/top-private.jsonl" -Encoding utf8
 return $m.Groups[1].Value
}
if((Top) -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_FOREGROUND'}
if(!(Awake)){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
Assert-PhoneLmNoExistingRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
if((Adb @('shell','ps','-A')) -match '(?m)\b(?:\S*probe|qnn-net-run|simple_app)\s*$'){throw 'COMPETING_QNN_PROBE'}
$before=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
if((Resolve-MixedQkvMonitoringState $before) -like 'BLOCKED*'){throw 'CRITICAL_DEVICE_PROTECTION'}
function Capture-Readonly([string]$Name,[string[]]$CaptureArguments) {
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $CaptureArguments -AllowFailure
 @{exit=$r.ExitCode;text_private=$r.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Depth 5|Out-File "$out/$Name-private.json" -Encoding utf8
 return $r
}
$baseline=Capture-Readonly 'logcat-before' @('logcat','-b','crash','-b','main','-b','system','-d','-t','4000')
if($baseline.ExitCode){throw 'CRASH_LOG_COLLECTION_UNAVAILABLE'}
Capture-Readonly 'tombstones-before' @('shell','ls','-l','/data/tombstones')|Out-Null
Capture-Readonly 'dropbox-before' @('shell','dumpsys','dropbox')|Out-Null
$script:verifiedInfrastructure=@{}
function Test-VerifiedInfrastructure([string]$Line) {
 if($Line -notmatch '^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+?)\s*$'){return $false}
 $processNumber=$Matches[1];$parentNumber=$Matches[2];$userName=$Matches[3];$processName=$Matches[4];$command=$Matches[5]
 $kernel=($parentNumber -eq '2' -and $userName -eq 'root' -and $processName -eq '[kworker/R-policy_probe_wq]' -and $command -eq $processName)
 $trace=($parentNumber -eq '1' -and $userName -eq 'nobody' -and $processName -eq 'traced_probes' -and $command -eq 'traced_probes')
 if(!$kernel -and !$trace){return $false}
 $identity="$processNumber|$parentNumber|$userName|$processName|$command"
 if($script:verifiedInfrastructure.ContainsKey($identity)){return $true}
 $status=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','cat',"/proc/$processNumber/status") -AllowFailure
 $cmdline=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','cat',"/proc/$processNumber/cmdline") -AllowFailure
 if($status.ExitCode -ne 0 -or $cmdline.ExitCode -ne 0){return $false}
 $verified=$false
 if($kernel){$verified=($status.Text -match '(?m)^Kthread:\s*1\s*$' -and $status.Text -match '(?m)^PPid:\s*2\s*$' -and $status.Text -match '(?m)^Uid:\s*0\s+0\s+0\s+0\s*$' -and [string]::IsNullOrEmpty($cmdline.Text))}
 if($trace){$verified=($status.Text -match '(?m)^Name:\s*traced_probes\s*$' -and $status.Text -match '(?m)^Kthread:\s*0\s*$' -and $status.Text -match '(?m)^PPid:\s*1\s*$' -and $status.Text -match '(?m)^Uid:\s*9999\s+9999\s+9999\s+9999\s*$' -and $cmdline.Text -ceq ('/system/bin/traced_probes'+[char]0))}
 @{verified=$verified;identity_private=$identity;status_private=$status.Text;cmdline_private=$cmdline.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/infrastructure-private.jsonl" -Encoding utf8
 if($verified){$script:verifiedInfrastructure[$identity]=$true}
 return $verified
}
function Competition {
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','ps','-A','-o','PID,PPID,USER,NAME,ARGS') -AllowFailure
 @{exit=$r.ExitCode;text_private=$r.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/competition-private.jsonl" -Encoding utf8
 if($r.ExitCode){throw 'COMPETITION_UNKNOWN'}
 foreach($line in ($r.Text -split "`n")){
  if($line -match '^\s*(\d+)\s+\d+\s+\S+\s+(\S+)'){
   $observedPid=$Matches[1];$name=$Matches[2]
   if($name -match '(?i)(probe|saved_p_backwa|qnn-net-run|simple_app|hmx|hexatrain|training)'){
    if(Test-VerifiedInfrastructure $line){continue}
    $ownedMatch=$null
    if(Test-Path "$out/stdout.txt"){
     $f=[IO.File]::Open("$out/stdout.txt",[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
     $reader=[IO.StreamReader]::new($f);try{$ownedMatch=[regex]::Match($reader.ReadToEnd(),'(?m)^owned_probe_pid=(\d+)\s*$')}finally{$reader.Dispose()}
    }
    if(!$ownedMatch -or !$ownedMatch.Success -or $ownedMatch.Groups[1].Value -ne $observedPid){throw 'COMPETING_NPU_WORKLOAD'}
   }
  }
 }
}
Competition
$hostInventory=Get-CimInstance Win32_Process|Select-Object ProcessId,ParentProcessId,Name,CommandLine
$hostInventory|ConvertTo-Json -Depth 4|Out-File "$out/host-processes-private.json" -Encoding utf8
# A fresh human-independent review of this saved inventory is required by the
# caller before launch; the exclusive permit binds this runner and candidate.
$reviewStarted=[DateTime]::UtcNow
while(!(Test-Path "$out/launch-permit-private.json")){
 if(([DateTime]::UtcNow-$reviewStarted).TotalSeconds -ge 120){throw 'HOST_INVENTORY_REVIEW_TIMEOUT_NO_PROBE'}
 Start-Sleep -Milliseconds 200
}
$permit=Get-Content "$out/launch-permit-private.json" -Raw|ConvertFrom-Json
if($permit.host_inventory_sha256 -ne (Get-FileHash "$out/host-processes-private.json").Hash.ToLowerInvariant()){throw 'HOST_INVENTORY_REVIEW_NOT_BOUND'}
if($permit.candidate_sha256 -ne $build.probe_sha256 -or $permit.runner_sha256 -ne (Get-FileHash $PSCommandPath).Hash.ToLowerInvariant() -or $permit.plan_sha256 -ne (Get-FileHash "$root/host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json").Hash.ToLowerInvariant() -or !$permit.host_workload_review_pass -or !$permit.host_gates_pass -or !$permit.protected_pass -or !$permit.source_delta_pass){throw 'DIAGNOSTIC_ENTRY_NOT_VALIDATED'}
Competition
if((Top) -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_FOREGROUND'}
if(!(Awake)){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
Assert-PhoneLmNoExistingRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
$before=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
if((Resolve-MixedQkvMonitoringState $before) -like 'BLOCKED*'){throw 'CRITICAL_DEVICE_PROTECTION'}
$claim=[IO.File]::Open("$mission/$Stage-attempt-private.json",[IO.FileMode]::CreateNew)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes((@{stage=$Stage;maximum_os_processes=1;retry=0;candidate_sha256=$build.probe_sha256;utc=[DateTime]::UtcNow.ToString('o');runner_revision=3}|ConvertTo-Json));$claim.Write($bytes,0,$bytes.Length)}finally{$claim.Dispose()}
# ---------------------------------------------------------------- post-exit transport
function Get-TransferFileSha256([string]$Path) { return (Get-PhoneLmTransferFileSha256 -Path $Path) }
$identityInvoker={
 param($Arguments,$TimeoutSeconds)
 $bound=[int][Math]::Min(600,[Math]::Max(1,[Math]::Ceiling([double]$TimeoutSeconds)))
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments -TimeoutSeconds $bound -AllowFailure
 return [pscustomobject]@{ExitCode=$r.ExitCode;Text=$r.Text;Classification=$r.Classification}
}
$chunkBytes=8388608
# Binary-safe chunk reader. adb exec-out writes raw bytes to the receiving file; no
# PowerShell text pipeline, no CRLF translation, no shell redirection.
$chunkInvoker={
 param([long]$SkipBytes,[long]$CountBytes,[string]$OutputPath,[double]$TimeoutSeconds)
 $remotePath=$script:currentRemotePath
 $blocks=[long]($SkipBytes/[long]$script:currentChunkBytes)
 $countBlocks=[long][Math]::Ceiling([double]$CountBytes/[double]$script:currentChunkBytes)
 $psi=[Diagnostics.ProcessStartInfo]::new()
 $psi.FileName=$adb
 $psi.Arguments='-s "{0}" exec-out dd if="{1}" bs={2} skip={3} count={4}' -f $d.Endpoint,$remotePath,$script:currentChunkBytes,$blocks,$countBlocks
 $psi.UseShellExecute=$false
 $psi.RedirectStandardOutput=$true
 $psi.RedirectStandardError=$true
 $psi.CreateNoWindow=$true
 $pull=[Diagnostics.Process]::new();$pull.StartInfo=$psi
 $stream=$null;$stderrTask=$null;$timedOut=$false
 try {
  [void]$pull.Start()
  $stream=[IO.File]::Create($OutputPath)
  $copy=$pull.StandardOutput.BaseStream.CopyToAsync($stream)
  $stderrTask=$pull.StandardError.ReadToEndAsync()
  $timedOut=-not $pull.WaitForExit([int]([Math]::Max(1000,[double]$TimeoutSeconds*1000)))
  if($timedOut){Stop-PhoneLmProcessTree -RootProcessId $pull.Id;[void]$pull.WaitForExit(5000)}
  $copy.GetAwaiter().GetResult()|Out-Null
  $stream.Dispose();$stream=$null
  $stderr=$stderrTask.GetAwaiter().GetResult()
  if($pull.HasExited){$pull.WaitForExit()}
  $exitCode=if($timedOut -or -not $pull.HasExited){124}else{[int]$pull.ExitCode}
  $written=[long]-1
  if(Test-Path -LiteralPath $OutputPath){$written=([IO.FileInfo]::new($OutputPath)).Length}
  $classification=if($timedOut){'ADB_WRAPPER_TIMEOUT'}elseif($exitCode -eq 0){'OK'}elseif($stderr -match '(?i)(device offline|device not found|no devices/emulators found|unauthorized|closed by peer|connection reset)'){'ADB_TRANSPORT_DISCONNECTED'}else{'ADB_COMMAND_FAILURE'}
  return [pscustomobject]@{ExitCode=$exitCode;Classification=$classification;BytesWritten=$written}
 } finally {
  if($stream){$stream.Dispose()}
  $pull.Dispose()
 }
}
$roleOrder=@('standalone_softmax','standalone_dvalue','native_softmax','native_dvalue','dual')
$roleFiles=@{standalone_softmax='cohort-standalone_softmax.bin';standalone_dvalue='cohort-standalone_dvalue.bin';native_softmax='cohort-native_softmax.bin';native_dvalue='cohort-native_dvalue.bin';dual='cohort-dual.bin'}
$transportRoles=[ordered]@{}
function Readback-Evidence {
 $readback=@{}
 foreach($name in @('cohort-standalone_softmax.bin','cohort-standalone_dvalue.bin','cohort-native_softmax.bin','cohort-native_dvalue.bin','cohort-dual.bin','saved-p-words.bin','performance-words.bin')){
  $remoteFile="$remote/$name"
  $found=AdbTimed -Arguments @('shell','test','-f',$remoteFile) -TimeoutSeconds 120 -AllowFailure
  if($found.ExitCode){continue}
  $hashResult=AdbTimed -Arguments @('shell','sha256sum',$remoteFile) -TimeoutSeconds 600
  $hash=($hashResult.Text -split '\s+')[0]
  if($hash -notmatch '^[0-9a-f]{64}$'){throw 'ROLE_RAW_HASH_UNREADABLE'}
  $gzipPath="$remoteFile.gz"
  $gzipResult=AdbTimed -Arguments @('shell',("toybox gzip -c $remoteFile > $gzipPath")) -TimeoutSeconds 900
  if($gzipResult.ExitCode){throw 'REMOTE_GZIP_FAILED'}
  $gzipHash=((AdbTimed -Arguments @('shell','sha256sum',$gzipPath) -TimeoutSeconds 900).Text -split '\s+')[0]
  if($gzipHash -notmatch '^[0-9a-f]{64}$'){throw 'ROLE_GZIP_HASH_UNREADABLE'}
  $beforeIdentity=Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $identityInvoker -RemotePath $gzipPath -TimeoutSeconds 300
  if(-not $beforeIdentity.Exists){throw 'REMOTE_GZIP_MISSING'}
  if($beforeIdentity.Sha256 -ne $gzipHash){throw 'REMOTE_GZIP_IDENTITY_MISMATCH'}
  $script:currentRemotePath=$gzipPath
  $script:currentChunkBytes=$chunkBytes
  $target="$out/$name.gz"
  $received=Receive-PhoneLmEvidenceFile -ChunkInvoker $chunkInvoker -RemotePath $gzipPath -DestinationPath $target -ExpectedSizeBytes ([long]$beforeIdentity.SizeBytes) -ExpectedSha256 $beforeIdentity.Sha256 -ChunkBytes $chunkBytes -ChunkTimeoutSeconds 120 -ChunkAttempts 2 -FileTimeoutSeconds 2700 -EvidencePath "$out/transfer-$name.jsonl"
  $hostSha=Get-TransferFileSha256 -Path $target
  $afterIdentity=Get-PhoneLmRemoteEvidenceIdentity -AdbInvoker $identityInvoker -RemotePath $gzipPath -TimeoutSeconds 300
  $integrity=Test-PhoneLmTransferIntegrity -BeforeIdentity $beforeIdentity -AfterIdentity $afterIdentity -Received $received -ReceivedSha256 $hostSha
  if(!$integrity.Verified){
   @{file=$name;status=$received.Status;reasons=@($integrity.Reasons);before=$beforeIdentity;after=$afterIdentity}|ConvertTo-Json -Depth 8|Out-File "$out/post-exit-transport-failure-private.json" -Encoding utf8
   throw ('POST_EXIT_EVIDENCE_INCOMPLETE: '+($integrity.Reasons -join '; '))
  }
  & python -c "import gzip,sys,shutil; src=gzip.open(sys.argv[1],'rb'); dst=open(sys.argv[2],'xb'); shutil.copyfileobj(src,dst); dst.close(); src.close()" "$target" "$out/$name"
  if($LASTEXITCODE -or (Get-FileHash "$out/$name").Hash.ToLowerInvariant() -ne $hash){throw 'ROLE_EVIDENCE_HASH'}
  $readback[$name]=@{sha256=$hash;gzip_sha256=$hostSha;gzip_bytes=[long]$received.Bytes;chunks=[int]$received.Chunks;transfer_status=$received.Status;remote_before_sha256=$beforeIdentity.Sha256;remote_after_sha256=$afterIdentity.Sha256}
  foreach($role in $roleOrder){if($roleFiles[$role] -eq $name){
   $transportRoles[$role]=[ordered]@{file=$name;status=$received.Status;host_sha256=$hostSha;remote_before_sha256=$beforeIdentity.Sha256;remote_after_sha256=$afterIdentity.Sha256;bytes=[long]$received.Bytes;chunks=[int]$received.Chunks;reconstructed_raw_sha256=$hash}
  }}
 }
 return $readback
}
# ---------------------------------------------------------------- monitor plan
$PeriodMilliseconds=2000
$LeadMilliseconds=100
$ThermalChildren=[System.Collections.Generic.List[object]]::new()
$monitorInvoker={
 param([string[]]$Arguments,[int]$TimeoutSeconds)
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds -AllowFailure
 $kind='competition'
 if($Arguments -contains 'power'){$kind='awake'}elseif($Arguments -contains 'activities'){$kind='top'}
 if($kind -ne 'top'){
  @{exit=$r.ExitCode;text_private=$r.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/$kind-private.jsonl" -Encoding utf8
 }
 return [pscustomobject]@{ExitCode=$r.ExitCode;Text=$r.Text;Classification=$r.Classification}
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
 if(!$m.Success){
  try {
   $failure=[ordered]@{reason='TOP_RESUMED_UNKNOWN';observed_utc=[DateTime]::UtcNow.ToString('o');response_type=$(if($null -eq $text){'null'}else{'System.String'});actual_response_private=$text;retry_performed=$false}
   $failurePath=Join-Path $out ('top-unknown-'+[guid]::NewGuid().ToString('N')+'-private.json')
   $failureStream=[IO.File]::Open($failurePath,[IO.FileMode]::CreateNew)
   try{$failureBytes=[Text.UTF8Encoding]::new($false).GetBytes(($failure|ConvertTo-Json -Depth 4));$failureStream.Write($failureBytes,0,$failureBytes.Length)}finally{$failureStream.Dispose()}
  }catch{}
  return [pscustomobject]@{Ok=$false;Known=$false;Reason='TOP_RESUMED_UNKNOWN';Detail=$null}
 }
 $package=$m.Groups[1].Value
 @{known=$true;package_private=$package;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/top-private.jsonl" -Encoding utf8
 if($package -eq 'com.yuubinnkyoku.phonelm'){return [pscustomobject]@{Ok=$false;Known=$true;Reason='FOREGROUND_TAKEOVER';Detail=$package}}
 return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$package}
}
$competitionProbe={
 param($text)
 if([string]::IsNullOrEmpty($text)){return [pscustomobject]@{Ok=$false;Known=$false;Reason='COMPETITION_UNKNOWN';Detail=$null}}
 foreach($line in ($text -split "`n")){
  if($line -match '^\s*(\d+)\s+\d+\s+\S+\s+(\S+)'){
   $observedPid=$Matches[1];$name=$Matches[2]
   if($name -match '(?i)(probe|saved_p_backwa|qnn-net-run|simple_app|hmx|hexatrain|training)'){
    if(Test-VerifiedInfrastructure $line){continue}
    $ownedMatch=$null
    if(Test-Path "$out/stdout.txt"){
     $f=[IO.File]::Open("$out/stdout.txt",[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
     $reader=[IO.StreamReader]::new($f);try{$ownedMatch=[regex]::Match($reader.ReadToEnd(),'(?m)^owned_probe_pid=(\d+)\s*$')}finally{$reader.Dispose()}
    }
    if(!$ownedMatch -or !$ownedMatch.Success -or $ownedMatch.Groups[1].Value -ne $observedPid){return [pscustomobject]@{Ok=$false;Known=$true;Reason='COMPETING_NPU_WORKLOAD';Detail=$null}}
   }
  }
 }
 return [pscustomobject]@{Ok=$true;Known=$true;Reason='OK';Detail=$null}
}
$thermalProbeScript=Join-Path $PSScriptRoot 'backward_formal_thermal_probe.ps1'
$auxiliaryStarter={
 param($monitor)
 $path=Join-Path $out ('thermal-aux-'+[guid]::NewGuid().ToString('N')+'.json')
 $child=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-File',$thermalProbeScript,'-Adb',$adb,'-Device',$d.Endpoint,'-OutPath',$path) -WindowStyle Hidden -PassThru
 $ThermalChildren.Add($child)
 return [pscustomobject]@{StartedMilliseconds=(Get-PhoneLmMonitorMilliseconds -Clock $monitor.Clock);Process=$child;Path=$path}
}
$auxiliaryCollector={
 param($claim)
 $child=$claim.Process
 $child.Refresh()
 if(-not $child.HasExited){return [pscustomobject]@{Complete=$false;Classification=$null}}
 if(-not [IO.File]::Exists($claim.Path)){return [pscustomobject]@{Complete=$false;Classification='AUXILIARY_SAMPLE_FAILED'}}
 try{$state=Get-Content -LiteralPath $claim.Path -Raw|ConvertFrom-Json}catch{return [pscustomobject]@{Complete=$false;Classification='AUXILIARY_SAMPLE_FAILED'}}
 return [pscustomobject]@{Complete=$true;Classification=[string]$state.classification}
}
$crashCollector=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'logcat','-b','crash','-b','main','-b','system','-v','threadtime','-T','1') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/logcat-stream-private.txt" -RedirectStandardError "$out/logcat-stream-stderr-private.txt"
Start-Sleep -Milliseconds 200
$crashCollector.Refresh();if($crashCollector.HasExited){throw 'CRASH_STREAM_NOT_RUNNING_NO_PROBE'}
$process=$null
$monitor=$null
$monitorContractFailed=$false
try {
$cmd="cd $remote && export LD_LIBRARY_PATH=$remote && export ADSP_LIBRARY_PATH='$remote;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp' && ( ./saved_p_backward_probe double-inputs.bin mixed-inputs.bin selective-ln1-inputs.bin gradients.bin $Mode $ForwardLimit & child=`$!; echo owned_probe_pid=`$child; wait `$child )"
$process=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'shell',('"'+$cmd+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/stdout.txt" -RedirectStandardError "$out/stderr.txt"
$started=[DateTime]::UtcNow;$samples=@($before);$takeovers=0
function Stop-Owned([string]$Reason) {
 $stream=[IO.File]::Open("$out/stdout.txt",[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
 $reader=[IO.StreamReader]::new($stream)
 try{$text=$reader.ReadToEnd()}finally{$reader.Dispose()}
 $m=[regex]::Match($text,'(?m)^owned_probe_pid=(\d+)\s*$');$cancelled=$false
 if($m.Success){$owned=[int]$m.Groups[1].Value;$r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','cat',"/proc/$owned/cmdline") -AllowFailure
  if(!$r.ExitCode -and $r.Text.Split([char]0)[0] -eq './saved_p_backward_probe'){Adb @('shell','kill','-INT',"$owned")|Out-Null;$cancelled=$true}}
 @{reason=$Reason;owned_probe_cancelled=$cancelled}|ConvertTo-Json|Out-File "$out/stop-private.json" -Encoding utf8
 throw $Reason
}
try {
 $monitor=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds 8 -AuxiliaryPeriodMilliseconds 6000 -AuxiliaryCollectBoundMilliseconds 4000
 while($true) {
  $wait=Wait-PhoneLmMonitorDeadline -Monitor $monitor -OwnedExitProbe { $process.Refresh(); $process.HasExited }
  if($wait -eq 'OWNED_EXIT'){break}
  $monitorElapsed=(Get-PhoneLmMonitorMilliseconds -Clock $monitor.Clock)-[double]$monitor.StartMilliseconds
  if($monitorElapsed -ge 1200000){Stop-Owned 'OWNED_PROBE_TIMEOUT'}
  $record=Invoke-PhoneLmMonitorSample -Monitor $monitor -AdbInvoker $monitorInvoker -TopProbe $topProbe -AwakeProbe $awakeProbe -CompetitionProbe $competitionProbe -AuxiliaryStarter $auxiliaryStarter -AuxiliaryCollector $auxiliaryCollector -EvidencePath "$out/monitor-samples-private.jsonl" -Phase 'in-run'
  if(@($record.violations|Where-Object {$_.reason -eq 'MONITOR_COMMAND_EXCEPTION'}).Count){Stop-Owned 'MONITOR_INFRASTRUCTURE_FAILURE'}
  $awakeState=$record.commands['awake'].probe
  if($null -ne $awakeState -and !$awakeState.known){Stop-Owned 'AWAKE_LOST_OR_UNKNOWN'}
  if($null -ne $awakeState -and !$awakeState.ok){Stop-Owned 'AWAKE_LOST'}
  $topState=$record.commands['top'].probe
  if($null -ne $topState -and !$topState.known){Stop-Owned 'TOP_RESUMED_UNKNOWN'}
  if($null -ne $topState -and !$topState.ok){$takeovers++;Stop-Owned 'FOREGROUND_TAKEOVER'}
  $competitionState=$record.commands['competition'].probe
  if($null -ne $competitionState -and !$competitionState.ok){Stop-Owned 'COMPETING_NPU_WORKLOAD'}
  if($null -ne $record.auxiliary){
   if($record.auxiliary.result -in @('AUXILIARY_SAMPLE_FAILED','AUXILIARY_COLLECTION_TIMEOUT')){Stop-Owned 'MONITOR_AUXILIARY_FAILURE'}
   if($record.auxiliary.result -eq 'AUXILIARY_SAMPLE_RECORDED' -and [string]$record.auxiliary.classification -like 'BLOCKED*'){Stop-Owned 'CRITICAL_DEVICE_PROTECTION'}
  }
  $process.Refresh()
  if($process.HasExited){break}
 }
 # Post-exit checks on a fresh monitor: they must not contaminate the in-run cadence ledger.
 $postMonitor=New-PhoneLmForegroundMonitor -PeriodMilliseconds $PeriodMilliseconds -LeadMilliseconds $LeadMilliseconds -CommandTimeoutSeconds 8
 $postRecord=Invoke-PhoneLmMonitorSample -Monitor $postMonitor -AdbInvoker $monitorInvoker -TopProbe $topProbe -AwakeProbe $awakeProbe -CompetitionProbe $competitionProbe -EvidencePath "$out/monitor-post-exit-samples-private.jsonl" -Phase 'post-exit'
 Write-PhoneLmMonitorSummary -Monitor $postMonitor -Path "$out/monitor-post-exit-summary-private.json"|Out-Null
 $postTop=$postRecord.commands['top'].probe
 $postAwake=$postRecord.commands['awake'].probe
 $postCompetition=$postRecord.commands['competition'].probe
 if($null -ne $postTop -and !$postTop.known){Stop-Owned 'TOP_RESUMED_UNKNOWN_AFTER_EXECUTE'}
 if($null -ne $postTop -and !$postTop.ok){$takeovers++;Stop-Owned 'FOREGROUND_TAKEOVER_AFTER_EXECUTE'}
 if($null -ne $postAwake -and !$postAwake.ok){Stop-Owned 'AWAKE_LOST_AFTER_EXECUTE'}
 if($null -ne $postCompetition -and !$postCompetition.ok){Stop-Owned 'COMPETING_NPU_WORKLOAD_AFTER_EXECUTE'}
 $monitorSummary=Get-PhoneLmMonitorSummary -Monitor $monitor
 Write-PhoneLmMonitorSummary -Monitor $monitor -Path "$out/monitor-summary-private.json"|Out-Null
 if(![bool]$monitorSummary.contract_met){
  $monitorContractFailed=$true
  @{classification='MONITOR_CONTRACT_FAILED';period_milliseconds=[double]$PeriodMilliseconds;lead_milliseconds=[double]$LeadMilliseconds;missed_deadlines=$monitorSummary.missed_deadlines;violation_reasons=@($monitorSummary.violation_reasons);maximum_start_to_start_milliseconds=$monitorSummary.maximum_start_to_start_milliseconds;not_a_numerical_failure=$true}|ConvertTo-Json|Out-File "$out/monitor-contract-failure-private.json" -Encoding utf8
 }
 $after=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
 $samples+=,$after;$samples|ConvertTo-Json -Depth 8|Out-File "$out/monitor-private.json" -Encoding utf8
 @{source_sha=(git -C $root rev-parse HEAD);stage=$Stage;orchestration_file='scripts/run_backward_shared_lifetime_formal_v3.ps1';orchestration_sha256=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant();build=$build;mode=$Mode;forward_limit=$ForwardLimit;
  exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$before;thermal_after=$after;
  monitor_summary_sha256=(Get-FileHash "$out/monitor-summary-private.json").Hash.ToLowerInvariant();
  duration_seconds=([DateTime]::UtcNow-$started).TotalSeconds}|ConvertTo-Json -Depth 18|Out-File "$out/execution-before-readback-private.json" -Encoding utf8
 $readback=Readback-Evidence
 $transportRecord=[ordered]@{schema=1;kind='G1_POST_EXIT_TRANSPORT_RECORD';runner_revision=3;chunk_bytes=$chunkBytes;chunk_timeout_seconds=120;chunk_attempts=2;file_timeout_seconds=2700;transfer_verified=([bool]($transportRoles.Count -eq $roleOrder.Count));roles=$transportRoles;utc=[DateTime]::UtcNow.ToString('o')}
 $transportRecord|ConvertTo-Json -Depth 8|Out-File "$out/transport-record-private.json" -Encoding utf8
 @{source_sha=(git -C $root rev-parse HEAD);stage=$Stage;orchestration_file='scripts/run_backward_shared_lifetime_formal_v3.ps1';orchestration_sha256=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant();build=$build;hashes=$hashes;device_private=$d;mode=$Mode;forward_limit=$ForwardLimit;
  role_raw=$readback;
  exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$before;thermal_after=$after;
  monitoring='USER_DIRECTED_RESEARCH';run_mode='BACKGROUND_CORRECTNESS';protected_identity='PASS';safety_complete=(!$monitorContractFailed);
  monitor_contract_met=[bool]$monitorSummary.contract_met;monitor_period_milliseconds=[double]$PeriodMilliseconds;monitor_lead_milliseconds=[double]$LeadMilliseconds;monitor_missed_deadlines=$monitorSummary.missed_deadlines;
  post_exit_transport_verified=($transportRoles.Count -eq $roleOrder.Count);runner_revision=3;
  duration_seconds=([DateTime]::UtcNow-$started).TotalSeconds}|ConvertTo-Json -Depth 18|Out-File "$out/identity-private.json" -Encoding utf8
 if((Resolve-MixedQkvMonitoringState $after) -like 'BLOCKED*'){throw 'CRITICAL_PROTECTION_AFTER_EXECUTE'}
 if($process.ExitCode -ne 0){throw 'ACTUAL_PROBE_NONZERO_EXIT'}
 Select-String -Path "$out/stdout.txt" -Pattern '^summary,'|ForEach-Object {$_.Line}
}catch {
 if(!(Test-Path "$out/stop-private.json")){Stop-Owned ('MONITOR_INFRASTRUCTURE_FAILURE: '+$_.Exception.Message)}
 throw
}
 # A timing-contract violation is not an infrastructure failure: readback and identity above
 # are written first, and the probe is not killed for the scheduler's own lateness.
 if($monitorContractFailed){throw 'MONITOR_CONTRACT_VIOLATION'}
} finally {
 foreach($child in $ThermalChildren){try{$child.Refresh();if(-not $child.HasExited){$child.Kill()}}catch{}}
 # Preserve actual completion and partial raw after a monitor/API/process failure.
 if(!(Test-Path "$out/identity-private.json") -and $null -ne $process){
  try{
   $process.WaitForExit(5000)|Out-Null;$process.Refresh()
   $observedExit=if($process.HasExited){$process.ExitCode}else{$null}
   @{stage=$Stage;actual_exit=$observedExit;process_still_running=(!$process.HasExited);no_new_arithmetic=$true;runner_revision=3}|ConvertTo-Json|Out-File "$out/partial-completion-private.json" -Encoding utf8
   if($process.HasExited -and !(Test-Path "$out/saved-p-words.bin")){
    $partial=Readback-Evidence
    @{stage=$Stage;actual_exit=$observedExit;role_raw=$partial}|ConvertTo-Json -Depth 8|Out-File "$out/partial-readback-private.json" -Encoding utf8
   }
  }catch{@{reason=$_.Exception.Message;not_numerical_pass=$true}|ConvertTo-Json|Out-File "$out/partial-readback-failure-private.json" -Encoding utf8}
 }
 try{Capture-Readonly 'logcat-after' @('logcat','-b','crash','-b','main','-b','system','-d','-t','12000')|Out-Null}catch{}
 try{Capture-Readonly 'dropbox-after' @('shell','dumpsys','dropbox','--print','SYSTEM_TOMBSTONE')|Out-Null}catch{}
 try{
  $listing=Capture-Readonly 'tombstones-after' @('shell','ls','/data/tombstones')
  if(!$listing.ExitCode){foreach($line in ($listing.Text -split "`n")){if($line.Trim() -match '^tombstone_\d+$'){Capture-Readonly ($line.Trim()) @('shell','cat',('/data/tombstones/'+$line.Trim()))|Out-Null}}}
 }catch{}
 if(!$crashCollector.HasExited){$crashCollector.Kill();$crashCollector.WaitForExit()}
}
exit $process.ExitCode
