# Independent five-role runner. No old runner, plan, package or evidence is modified.
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
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
function Test-SavedPStagingCollision([hashtable]$Hashes,[string]$Name,[string]$Hash) {
 return ($Hashes.ContainsKey($Name) -and $Hashes[$Name] -ne $Hash)
}
if($SelfTest) {
 $hashesTest=@{same='a'}
 if((Test-SavedPStagingCollision $hashesTest 'new' 'b') -or
    (Test-SavedPStagingCollision $hashesTest 'same' 'a') -or
    !(Test-SavedPStagingCollision $hashesTest 'same' 'b')){throw 'STAGING_COLLISION_SELFTEST'}
 Write-Host 'saved_p_runner_selftest=PASS staging_cases=3 device_calls=0';exit 0
}
$root=Split-Path -Parent $PSScriptRoot;$out=[IO.Path]::GetFullPath($ReportRoot)
$mission=Join-Path $root 'build/reports/g1-backward-shared-lifetime-formal'
trap {
 $record=@{classification='MISSION_STOP';stage=$Stage;reason=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o');no_automatic_retry=$true}
 $stopPath=Join-Path $mission 'mission-stop-private.json'
 if(!(Test-Path $stopPath)){$stream=[IO.File]::Open($stopPath,[IO.FileMode]::CreateNew);try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($record|ConvertTo-Json));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}}
 throw $_
}
if($out -ne [IO.Path]::GetFullPath((Join-Path $mission $Stage))){throw 'EXACT_STAGE_DIRECTORY_REQUIRED'}
if(($Stage -eq 'performance') -ne ($Mode -eq 'shared-performance')){throw 'STAGE_MODE_SCOPE'}
& python -B "$PSScriptRoot/certify_backward_shared_lifetime_formal.py" --entry $Stage
if($LASTEXITCODE){throw 'INDEPENDENT_PREVIOUS_STAGE_NOT_PASS'}
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
@{purpose='USER_AUTHORIZED_FIVE_ROLE_RESEARCH_ENTRY_ONLY';stage=$Stage;probe_sha256=$build.probe_sha256;plan_sha256=(Get-FileHash "$root/host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json").Hash.ToLowerInvariant();not_formal_certification=$true}|ConvertTo-Json|Out-File $permitFile -Encoding utf8
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
function Awake {
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','dumpsys','power') -AllowFailure
 @{exit=$r.ExitCode;text_private=$r.Text;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json -Compress|Add-Content "$out/awake-private.jsonl" -Encoding utf8
 return ($r.ExitCode -eq 0 -and $r.Text -match '(?m)^\s*mWakefulness=Awake\s*$')
}
function Top {
 $a=Adb @('shell','dumpsys','activity','activities')
 $m=[regex]::Match($a,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/')
 if(!$m.Success){
  # Preserve the exact failed response, then keep the original fail-closed stop.
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
 # Inspect process executable names, not incidental substrings in owned shell commands.
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
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes((@{stage=$Stage;maximum_os_processes=1;retry=0;candidate_sha256=$build.probe_sha256;utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json));$claim.Write($bytes,0,$bytes.Length)}finally{$claim.Dispose()}
function Readback-Evidence {
$readback=@{}
foreach($name in @('cohort-standalone_softmax.bin','cohort-standalone_dvalue.bin','cohort-native_softmax.bin','cohort-native_dvalue.bin','cohort-dual.bin','saved-p-words.bin','performance-words.bin')){
 $found=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','test','-f',"$remote/$name") -AllowFailure
 if($found.ExitCode){continue}
 $hash=((Adb @('shell','sha256sum',"$remote/$name")) -split '\s+')[0]
 Adb @('shell',"toybox gzip -c $remote/$name > $remote/$name.gz")|Out-Null
 $gzipHash=((Adb @('shell','sha256sum',"$remote/$name.gz")) -split '\s+')[0]
 $pull=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('pull',"$remote/$name.gz","$out/$name.gz") -TimeoutSeconds 180
 if($pull.ExitCode -or (Get-FileHash "$out/$name.gz").Hash.ToLowerInvariant() -ne $gzipHash){throw 'COMPRESSED_ROLE_EVIDENCE_TRANSFER'}
 & python -c "import gzip,sys,shutil; src=gzip.open(sys.argv[1],'rb'); dst=open(sys.argv[2],'xb'); shutil.copyfileobj(src,dst); dst.close(); src.close()" "$out/$name.gz" "$out/$name"
 if($LASTEXITCODE -or (Get-FileHash "$out/$name").Hash.ToLowerInvariant() -ne $hash){throw 'ROLE_EVIDENCE_HASH'}
 if(((Adb @('shell','sha256sum',"$remote/$name")) -split '\s+')[0] -ne $hash){throw 'REMOTE_ROLE_CHANGED'}
 $readback[$name]=@{sha256=$hash;gzip_sha256=$gzipHash}
}
return $readback
}
$crashCollector=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'logcat','-b','crash','-b','main','-b','system','-v','threadtime','-T','1') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/logcat-stream-private.txt" -RedirectStandardError "$out/logcat-stream-stderr-private.txt"
Start-Sleep -Milliseconds 200
$crashCollector.Refresh();if($crashCollector.HasExited){throw 'CRASH_STREAM_NOT_RUNNING_NO_PROBE'}
try {
$cmd="cd $remote && export LD_LIBRARY_PATH=$remote && export ADSP_LIBRARY_PATH='$remote;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp' && ( ./saved_p_backward_probe double-inputs.bin mixed-inputs.bin selective-ln1-inputs.bin gradients.bin $Mode $ForwardLimit & child=`$!; echo owned_probe_pid=`$child; wait `$child )"
$process=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'shell',('"'+$cmd+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/stdout.txt" -RedirectStandardError "$out/stderr.txt"
$started=[DateTime]::UtcNow;$lastMonitor=$started;$samples=@($before);$takeovers=0
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
 do {
  Competition
  if(!(Awake)){Stop-Owned 'AWAKE_LOST'}
  if((Top) -eq 'com.yuubinnkyoku.phonelm'){$takeovers++;Stop-Owned 'FOREGROUND_TAKEOVER'}
  if(([DateTime]::UtcNow-$lastMonitor).TotalSeconds -ge 10) {
   $s=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint;$samples+=$s;$lastMonitor=[DateTime]::UtcNow
   if((Resolve-MixedQkvMonitoringState $s) -like 'BLOCKED*'){Stop-Owned 'CRITICAL_DEVICE_PROTECTION'}
   if(!(Awake)){Stop-Owned 'AWAKE_LOST'}
  }
  if(([DateTime]::UtcNow-$started).TotalSeconds -ge 1200){Stop-Owned 'OWNED_PROBE_TIMEOUT'}
  $process.WaitForExit(2000)|Out-Null;$process.Refresh()
 }while(!$process.HasExited)
}catch {
 if(!(Test-Path "$out/stop-private.json")){Stop-Owned ('MONITOR_INFRASTRUCTURE_FAILURE: '+$_.Exception.Message)}
 throw
}
if((Top) -eq 'com.yuubinnkyoku.phonelm'){Stop-Owned 'FOREGROUND_TAKEOVER_AFTER_EXECUTE'}
if(!(Awake)){Stop-Owned 'AWAKE_LOST_AFTER_EXECUTE'}
Competition
$after=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
$samples+=,$after;$samples|ConvertTo-Json -Depth 8|Out-File "$out/monitor-private.json" -Encoding utf8
@{source_sha=(git -C $root rev-parse HEAD);stage=$Stage;orchestration_file='scripts/run_backward_shared_lifetime_formal.ps1';orchestration_sha256=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant();build=$build;mode=$Mode;forward_limit=$ForwardLimit;
 exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$before;thermal_after=$after;
 duration_seconds=([DateTime]::UtcNow-$started).TotalSeconds}|ConvertTo-Json -Depth 18|Out-File "$out/execution-before-readback-private.json" -Encoding utf8
$readback=Readback-Evidence

@{source_sha=(git -C $root rev-parse HEAD);stage=$Stage;orchestration_file='scripts/run_backward_shared_lifetime_formal.ps1';orchestration_sha256=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant();build=$build;hashes=$hashes;device_private=$d;mode=$Mode;forward_limit=$ForwardLimit;
 role_raw=$readback;
 exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$before;thermal_after=$after;
 monitoring='USER_DIRECTED_RESEARCH';run_mode='BACKGROUND_CORRECTNESS';protected_identity='PASS';safety_complete=$true;
 duration_seconds=([DateTime]::UtcNow-$started).TotalSeconds}|ConvertTo-Json -Depth 18|Out-File "$out/identity-private.json" -Encoding utf8
if((Resolve-MixedQkvMonitoringState $after) -like 'BLOCKED*'){throw 'CRITICAL_PROTECTION_AFTER_EXECUTE'}
if($process.ExitCode -ne 0){throw 'ACTUAL_PROBE_NONZERO_EXIT'}
Select-String -Path "$out/stdout.txt" -Pattern '^summary,'|ForEach-Object {$_.Line}
}finally{
 # Preserve actual completion and partial raw after a monitor/API/process failure.
 if(!(Test-Path "$out/identity-private.json") -and $null -ne $process){
  try{
   $process.WaitForExit(5000)|Out-Null;$process.Refresh()
   $observedExit=if($process.HasExited){$process.ExitCode}else{$null}
   @{stage=$Stage;actual_exit=$observedExit;process_still_running=(!$process.HasExited);no_new_arithmetic=$true}|ConvertTo-Json|Out-File "$out/partial-completion-private.json" -Encoding utf8
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
