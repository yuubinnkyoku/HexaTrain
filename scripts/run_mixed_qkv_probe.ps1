param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot,[Parameter(Mandatory=$true)][string]$ReportRoot,[Parameter(Mandatory=$true)][string]$QnnValidationLog,[switch]$UserAuthorizedHostStorageWaiver,[switch]$UserAuthorizedExternalPowerWaiver,[switch]$UserAuthorizedPowerThermalWaiver,[ValidateSet('mixed_qkv','selective_ln1','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB','oneProbabilityScale','oneProbabilityMask','oneProbabilitySoftmax','oneProbabilityP')][string]$Variant='mixed_qkv')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
. (Join-Path $PSScriptRoot 'mixed_qkv_monitoring_policy.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath((Join-Path $root 'build/reports'))+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_EVIDENCE_REQUIRED'}
if(Test-Path "$out/identity-private.json"){throw 'EXISTING_PROCESS_EVIDENCE'}
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$qnn=Get-Content -LiteralPath $QnnValidationLog -Raw
if($qnn -notmatch '===== verify.ps1 -Profile Qnn =====' -or $qnn -notmatch 'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$'){throw 'CURRENT_QNN_PROFILE_NOT_COMPLETE'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$waiver=@{};if($UserAuthorizedHostStorageWaiver){$waiver.UserAuthorizedHostStorageWaiver=$true};if($UserAuthorizedExternalPowerWaiver){$waiver.UserAuthorizedExternalPowerWaiver=$true};if($UserAuthorizedPowerThermalWaiver){$waiver.UserAuthorizedPowerThermalWaiver=$true}
$preflightArgs=@('-NoProfile','-File',(Join-Path $PSScriptRoot 'preflight_forward_p_resume.ps1'),'-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-HexagonSdkRoot',$HexagonSdkRoot,'-NdkRoot',$NdkRoot,'-ReportRoot',"$out/preflight",'-MixedQkvResearchMonitoring','-ExpectedApkPath',"$root/build/reports/g1-s2-mixed-qkv-df-hvx-v/approved-apk/app-debug.apk")
foreach($key in $waiver.Keys){$preflightArgs+="-$key"}
# Bound the Host preflight itself: an unresponsive ADB launch must never turn
# this diagnostic into indefinite polling. No server or unknown child is killed.
if(@($preflightArgs|Where-Object {$_ -match '"'}).Count){throw 'PREFLIGHT_ARGUMENT_QUOTE_REJECTED'}
$quoted=@($preflightArgs|ForEach-Object {'"'+$_+'"'})
$preflightProcess=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList $quoted -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/preflight.txt" -RedirectStandardError "$out/preflight-stderr.txt"
$preflightStart=[DateTime]::UtcNow
while(!$preflightProcess.WaitForExit(1000)){
 if(([DateTime]::UtcNow-$preflightStart).TotalSeconds -ge 180){
  $hostStopped=$false
  try{$preflightProcess.Kill();$hostStopped=$true}catch{}
  @{reason='HOST_PREFLIGHT_TIMEOUT';device_probe_executes=0;only_owned_host_preflight_terminated=$hostStopped;child_client_cancellation_unverified=$true}|ConvertTo-Json|Out-File "$out/preflight-timeout-private.json" -Encoding utf8
  throw 'HOST_PREFLIGHT_TIMEOUT_NO_DEVICE_EXECUTION'
 }
}
$preflightProcess.Refresh()
if($preflightProcess.ExitCode){throw 'DEVICE_SAFETY_GATE_FAILED'}
$preflight=Get-Content "$out/preflight/device-preflight-private.json" -Raw|ConvertFrom-Json
if($preflight.status -ne 'PASS'){throw 'PREFLIGHT_NOT_PASS'}
$build=Get-Content (Join-Path $root 'build/fp32-mixed-qkv/build-identity.json') -Raw|ConvertFrom-Json
foreach($p in $build.source_hashes.PSObject.Properties){if((Get-FileHash (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'NEW_SOURCE_CHANGED'}}
foreach($p in $build.binary_hashes.PSObject.Properties){if((Get-FileHash (Join-Path $root "build/fp32-mixed-qkv/$($p.Name)")).Hash.ToLowerInvariant() -ne $p.Value){throw 'NEW_BINARY_CHANGED'}}
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
$d=$preflight.device_private
function Adb {param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments);$r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments;if($r.ExitCode){throw 'ADB_OPERATION_FAILED'};return $r.Text}
if((Adb @('shell','ps','-A')) -match '(?m)\b(?:fp32_probe|layernorm_probe|one_layer_probe|selective_ln1_probe|selective_ln1_regression_probe|mixed_qkv_probe|mixed_s2_regression_probe|mixed_existing_regression_probe)\s*$'){throw 'ACTIVE_CUSTOM_PROBE'}
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath "$root/build/reports/g1-s2-mixed-qkv-df-hvx-v/approved-apk/app-debug.apk" -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath "$out/apk-audit.txt" *> "$out/apk-audit-console.txt"
if((Get-Content "$out/apk-audit.txt" -Raw) -notmatch 'status=SUCCESS'){throw 'APK_AUDIT_FAILED'}
$remote='/data/local/tmp/hexas2-'+$build.binary_hashes.mixed_qkv_probe.Substring(0,16)
$files=@('libHexaMixedQkvDsp.so','libHexaMixedQkvHost.so','mixed_qkv_probe','mixed_s2_regression_probe','mixed_existing_regression_probe','mixed-inputs.bin')|ForEach-Object {Join-Path $root "build/fp32-mixed-qkv/$_"}
$s2=Get-Content "$root/build/fp32-selective-ln1/build-identity.json" -Raw|ConvertFrom-Json
foreach($p in $s2.binary_hashes.PSObject.Properties){if((Get-FileHash "$root/build/fp32-selective-ln1/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'S2_BINARY_CHANGED'}}
foreach($p in $s2.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'S2_SOURCE_CHANGED'}}
$files+=@('libHexaSelectiveLn1Dsp.so','libHexaSelectiveLn1Host.so','selective-ln1-inputs.bin')|ForEach-Object {Join-Path $root "build/fp32-selective-ln1/$_"}
$fixturePaths=@('build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin','build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin','build/reports/g1-fp32-attention-ln1-frontier/ln1-producers.bin')
$fixtureHashes=@('388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f','7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f','ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580')
for($i=0;$i -lt 3;$i++){$file=Join-Path $root $fixturePaths[$i];if((Get-FileHash $file).Hash.ToLowerInvariant() -ne $fixtureHashes[$i]){throw 'OLD_FIXTURE_CHANGED'};$files+=$file}
$files+=@('libHexaFp32Dsp.so','libHexaFp32Host.so')|ForEach-Object {Join-Path $root "build/fp32-custom/$_"}
foreach($name in @('libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so')){$files+=Join-Path $QairtSdkRoot "lib/aarch64-android/$name"}
$files+=Join-Path $QairtSdkRoot 'lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so'
$files+=Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'
Adb @('shell','mkdir','-p',$remote)|Out-Null
$hashes=@{}
foreach($file in $files){$name=Split-Path $file -Leaf;$hash=(Get-FileHash $file).Hash.ToLowerInvariant();$old=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','sha256sum',"$remote/$name") -AllowFailure;if($old.ExitCode -ne 0 -and $old.Text -notmatch 'No such file or directory'){throw 'STAGING_STATE_UNKNOWN'};if($old.ExitCode -ne 0 -or ($old.Text -split '\s+')[0] -ne $hash){Adb @('push',$file,$remote)|Out-Null};if(((Adb @('shell','sha256sum',"$remote/$name")) -split '\s+')[0] -ne $hash){throw 'STAGED_HASH_MISMATCH'};$hashes[$name]=$hash}
$executable=if($Variant -eq 'mixed_qkv'){'mixed_qkv_probe'}elseif($Variant -eq 'selective_ln1'){'mixed_s2_regression_probe'}else{'mixed_existing_regression_probe'}
Adb @('shell','chmod','700',"$remote/$executable")|Out-Null
function Top {$a=Adb @('shell','dumpsys','activity','activities');$m=[regex]::Match($a,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/');if(!$m.Success){throw 'TOP_RESUMED_UNKNOWN'};return $m.Groups[1].Value}
if((Top) -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_FOREGROUND'}
if((Adb @('shell','dumpsys','power')) -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
Assert-PhoneLmNoExistingRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
if((Adb @('shell','ps','-A')) -match '(?m)\b(?:fp32_probe|layernorm_probe|one_layer_probe|selective_ln1_probe|selective_ln1_regression_probe|mixed_qkv_probe|mixed_s2_regression_probe|mixed_existing_regression_probe)\s*$'){throw 'ACTIVE_CUSTOM_PROBE_BEFORE_EXECUTE'}
$thermalBefore=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
$beforeState=Resolve-MixedQkvMonitoringState $thermalBefore
if($beforeState -like 'BLOCKED*'){throw 'POWER_THERMAL_UNSAFE'}
$argument=if($Variant -eq 'mixed_qkv'){'mixed-inputs.bin selective-ln1-inputs.bin'}elseif($Variant -eq 'selective_ln1'){'selective-ln1-inputs.bin'}else{$Variant}
$cmd="cd $remote && export LD_LIBRARY_PATH=$remote && export ADSP_LIBRARY_PATH='$remote;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp' && ( ./$executable $argument & child=`$!; echo owned_probe_pid=`$child; wait `$child )"
$process=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'shell',('"'+$cmd+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out/stdout.txt" -RedirectStandardError "$out/stderr.txt"
$takeovers=0;$started=[DateTime]::UtcNow;$lastMonitor=$started;$monitorSamples=@($thermalBefore)
function Stop-OwnedDiagnostic([string]$Reason,[object]$State){
 # Only this invocation's announced PID, verified by cmdline. Never app force-stop.
 $pidMatch=$null
 try{
  $stream=[IO.File]::Open("$out/stdout.txt",[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
  $reader=[IO.StreamReader]::new($stream)
  try{$pidMatch=[regex]::Match($reader.ReadToEnd(),'(?m)^owned_probe_pid=(\d+)\s*$')}finally{$reader.Dispose()}
 }catch{}
 $cancelled=$false
 if($null -ne $pidMatch -and $pidMatch.Success){$owned=[int]$pidMatch.Groups[1].Value;$cmdline=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','cat',"/proc/$owned/cmdline") -AllowFailure;if($cmdline.ExitCode -eq 0 -and $cmdline.Text.Split([char]0)[0] -eq "./$executable"){Adb @('shell','kill','-INT',"$owned")|Out-Null;$cancelled=$true}}
 @{reason=$Reason;state=$State;owned_probe_cancelled=$cancelled;no_app_force_stop=$true}|ConvertTo-Json -Depth 8|Out-File "$out/runtime-monitor-stop-private.json" -Encoding utf8
 throw $Reason
}
do {
 if((Top) -eq 'com.yuubinnkyoku.phonelm'){$takeovers++;Stop-OwnedDiagnostic 'TARGET_FOREGROUND_DURING_PROBE' $null}
 if(([DateTime]::UtcNow-$lastMonitor).TotalSeconds -ge 10){$sample=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint;$monitorSamples+=$sample;$lastMonitor=[DateTime]::UtcNow;if((Resolve-MixedQkvMonitoringState $sample) -like 'BLOCKED*'){Stop-OwnedDiagnostic 'CRITICAL_PROTECTION_DURING_PROBE' $sample};if((Adb @('shell','dumpsys','power')) -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){Stop-OwnedDiagnostic 'AWAKE_LOST_DURING_PROBE' $sample}}
 if(([DateTime]::UtcNow-$started).TotalSeconds -gt 240){Stop-OwnedDiagnostic 'OWNED_PROBE_TIMEOUT' $null}
 $process.WaitForExit(2000)|Out-Null;$process.Refresh()
}while(!$process.HasExited)
$monitorSamples|ConvertTo-Json -Depth 8|Out-File "$out/runtime-monitor-samples-private.json" -Encoding utf8
$after=Get-MixedQkvMonitoringState -Adb $adb -Device $d.Endpoint
$afterState=Resolve-MixedQkvMonitoringState $after
@{source_sha=(git -C $root rev-parse HEAD);build=$build;device_private=$d;hashes=$hashes;variant=$Variant;mode='BACKGROUND_CORRECTNESS';research_monitoring_policy='MIXED_QKV_USER_DIRECTED';host_storage_waiver=[bool]$UserAuthorizedHostStorageWaiver;external_power_waiver=[bool]$UserAuthorizedExternalPowerWaiver;power_thermal_waiver=[bool]$UserAuthorizedPowerThermalWaiver;thermal_before_gate=$beforeState;thermal_after_gate=$afterState;exit_code=$process.ExitCode;focus_takeover_count=$takeovers;thermal_before=$thermalBefore;thermal_after=$after;protected_identity='PASS';status='COMPLETE';started_utc=$started.ToString('o')}|ConvertTo-Json -Depth 22|Out-File "$out/identity-private.json" -Encoding utf8
if($takeovers){throw 'FOREGROUND_TAKEOVER'}
if($afterState -like 'BLOCKED*'){throw 'POWER_THERMAL_UNSAFE_AFTER_EXECUTE'}
$text=Get-Content "$out/stdout.txt" -Raw
if($text -notmatch '(?m)^summary,'){throw 'PROBE_INCOMPLETE'}
Select-String -Path "$out/stdout.txt" -Pattern '^summary,'|ForEach-Object {$_.Line}
exit $process.ExitCode
