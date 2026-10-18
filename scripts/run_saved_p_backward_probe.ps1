param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot,
 [Parameter(Mandatory=$true)][string]$ReportRoot,[Parameter(Mandatory=$true)][string]$QnnValidationLog,
 [ValidateSet('standalone_softmax','standalone_dvalue')][string]$Mode,
 [ValidateRange(1,60)][int]$ForwardLimit=60,
 [ValidatePattern('^[a-z0-9-]{1,32}$')][string]$Candidate='candidate',[switch]$SelfTest)
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
if(!$out.StartsWith([IO.Path]::GetFullPath("$root/build/reports")+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_REPORT_REQUIRED'}
if(Test-Path $out){throw 'EXCLUSIVE_PROCESS_REPORT_REQUIRED'}
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$qnn=Get-Content $QnnValidationLog -Raw
if($qnn -notmatch '===== verify.ps1 -Profile Qnn =====' -or $qnn -notmatch 'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$'){throw 'CURRENT_QNN_GATE_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$approved="$root/build/reports/g1-s2-mixed-qkv-df-hvx-v/approved-apk/app-debug.apk"
# Fresh per invocation. No waiver for storage, no wake/unlock/install.
& pwsh -NoProfile -File "$PSScriptRoot/preflight_forward_p_resume.ps1" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -HexagonSdkRoot $HexagonSdkRoot -NdkRoot $NdkRoot -ReportRoot "$out/preflight" -MixedQkvResearchMonitoring -ExpectedApkPath $approved *> "$out/preflight-console.txt"
if($LASTEXITCODE){throw 'BLOCKED_DEVICE_SAFETY'}
$preflight=Get-Content "$out/preflight/device-preflight-private.json" -Raw|ConvertFrom-Json
if($preflight.status -ne 'PASS'){throw 'PREFLIGHT_NOT_PASS'}
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe';$d=$preflight.device_private
function Adb {param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments);$r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments;if($r.ExitCode){throw 'ADB_OPERATION_FAILED'};return $r.Text}
$candidateDir="$root/build/fp32-saved-p-backward-$Candidate"
$build=Get-Content "$candidateDir/build-identity.json" -Raw|ConvertFrom-Json
foreach($p in $build.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'CANDIDATE_SOURCE_CHANGED'}}
$probe="$candidateDir/saved_p_backward_probe"
if((Get-FileHash $probe).Hash.ToLowerInvariant() -ne $build.probe_sha256){throw 'CANDIDATE_BINARY_CHANGED'}
$files=@($probe,"$root/build/reports/g1-native-saved-p-backward-dsp/gradients.bin")
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
function Top {$a=Adb @('shell','dumpsys','activity','activities');$m=[regex]::Match($a,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/');if(!$m.Success){throw 'TOP_RESUMED_UNKNOWN'};return $m.Groups[1].Value}
if((Top) -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_FOREGROUND'}
if((Adb @('shell','dumpsys','power')) -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
Assert-PhoneLmNoExistingRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
if((Adb @('shell','ps','-A')) -match '(?m)\b(?:\S*probe|qnn-net-run|simple_app)\s*$'){throw 'COMPETING_QNN_PROBE'}
$before=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
if((Resolve-MixedQkvMonitoringState $before) -like 'BLOCKED*'){throw 'CRITICAL_DEVICE_PROTECTION'}
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
  if((Top) -eq 'com.yuubinnkyoku.phonelm'){$takeovers++;Stop-Owned 'FOREGROUND_TAKEOVER'}
  if(([DateTime]::UtcNow-$lastMonitor).TotalSeconds -ge 10) {
   $s=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint;$samples+=$s;$lastMonitor=[DateTime]::UtcNow
   if((Resolve-MixedQkvMonitoringState $s) -like 'BLOCKED*'){Stop-Owned 'CRITICAL_DEVICE_PROTECTION'}
   if((Adb @('shell','dumpsys','power')) -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){Stop-Owned 'AWAKE_LOST'}
  }
  if(([DateTime]::UtcNow-$started).TotalSeconds -ge 600){Stop-Owned 'OWNED_PROBE_TIMEOUT'}
  $process.WaitForExit(2000)|Out-Null;$process.Refresh()
 }while(!$process.HasExited)
}catch {
 if(!(Test-Path "$out/stop-private.json")){Stop-Owned ('MONITOR_INFRASTRUCTURE_FAILURE: '+$_.Exception.Message)}
 throw
}
$after=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
$samples+=,$after;$samples|ConvertTo-Json -Depth 8|Out-File "$out/monitor-private.json" -Encoding utf8
$binaryHash=(Adb @('shell','sha256sum',"$remote/saved-p-words.bin") -split '\s+')[0]
Adb @('pull',"$remote/saved-p-words.bin","$out/saved-p-words.bin")|Out-Null
if((Get-FileHash "$out/saved-p-words.bin").Hash.ToLowerInvariant() -ne $binaryHash){throw 'OUTPUT_EVIDENCE_HASH_MISMATCH'}
@{source_sha=(git -C $root rev-parse HEAD);build=$build;hashes=$hashes;device_private=$d;mode=$Mode;forward_limit=$ForwardLimit;
 raw_evidence_sha256=$binaryHash;
 exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$before;thermal_after=$after;
 monitoring='USER_DIRECTED_RESEARCH';run_mode='BACKGROUND_CORRECTNESS';protected_identity='PASS';
 duration_seconds=([DateTime]::UtcNow-$started).TotalSeconds}|ConvertTo-Json -Depth 18|Out-File "$out/identity-private.json" -Encoding utf8
if((Resolve-MixedQkvMonitoringState $after) -like 'BLOCKED*'){throw 'CRITICAL_PROTECTION_AFTER_EXECUTE'}
Select-String -Path "$out/stdout.txt" -Pattern '^summary,'|ForEach-Object {$_.Line}
exit $process.ExitCode
