param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$NdkRoot,
 [ValidateSet('Copy','Add','Multiply','Affine','chain','chainTap','reshape','htpAdd','htpMultiply','standardAdd','standardMultiply','lnForward','lnBackward','lnMatrix','lnResidual','lnResidualTap','lnResidualFused','matmul','copyNative','mmDirect','mmBoundary','mmLn2','oneStandard','oneSegment','oneBoundary','oneFinal','oneBroadcast','oneUpstream','oneAttention','oneLn1','oneObserver','oneSoftmax','oneProjection','oneProducer','oneClosureA','oneClosureB','oneClosureC','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB')][string]$Variant='Copy'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(& git -C $root status --porcelain){throw 'SOURCE_MUST_BE_COMMITTED'}
$build=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw | ConvertFrom-Json
foreach($p in $build.source_hashes.psobject.Properties){
 if((Get-FileHash -LiteralPath (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'BUILD_SOURCE_MISMATCH'}
}
foreach($p in $build.binary_hashes.psobject.Properties){
 if((Get-FileHash -LiteralPath (Join-Path $root "build/fp32-custom/$($p.Name)")).Hash.ToLowerInvariant() -ne $p.Value){throw 'BUILD_BINARY_MISMATCH'}
}
if($Variant -eq 'mmLn2'){
 $gate=Get-Content (Join-Path $root 'build/reports/htp-fp32-custom-matmul-ln2/independent-gate.json') -Raw|ConvertFrom-Json
 if($gate.phase -ne 'Independent' -or $gate.processes -ne 6 -or $gate.source_sha -ne $build.source_sha -or $gate.package_sha256 -ne $build.binary_hashes.'libHexaFp32Dsp.so' -or $gate.probe_sha256 -ne $build.binary_hashes.layernorm_probe){throw 'LN2_REQUIRES_SAME_BINARY_INDEPENDENT_MATMUL_GATE'}
 if(@($gate.results|Where-Object {$_.variant -notin @('standard','standardNative') -and $_.failures -ne 0}).Count){throw 'LN2_INDEPENDENT_ACCURACY_NOT_ESTABLISHED'}
}
$plan=Get-Content (Join-Path $root 'build/reports/g1-long-linear-200k-primary/run-plan.json') -Raw | ConvertFrom-Json
if($plan.status -ne 'UNSTABLE'){throw 'PROTECTED_STATUS_MISMATCH'}
$protected=Get-Content (Join-Path $root 'build/reports/g1-long-linear-forensics/final-protected-identities.json') -Raw | ConvertFrom-Json
foreach($p in $protected.protected_sha256.psobject.Properties){
 if((Get-FileHash -LiteralPath (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'PROTECTED_IDENTITY_MISMATCH'}
}
if([IO.DriveInfo]::new([IO.Path]::GetPathRoot($root)).AvailableFreeSpace -lt 20GB){throw 'HOST_STORAGE_HEADROOM_LOW'}
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
$d=Resolve-PhoneLmDevice -Adb $adb
if($d.Serial -ne $plan.device_serial_private){throw 'PHYSICAL_IDENTITY_MISMATCH'}
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $d.Endpoint
Assert-PhoneLmNoExistingRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $d.Endpoint -Package com.yuubinnkyoku.phonelm
if($Variant -in @('oneBroadcast','oneUpstream','oneAttention','oneLn1','oneObserver','oneSoftmax','oneProjection','oneProducer','oneClosureA','oneClosureB','oneClosureC','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB')){
 $awake=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','dumpsys','power')
 if($awake.Text -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
 $activity=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','dumpsys','activity','activities')
 if($activity.Text -notmatch '(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/'){throw 'TOP_RESUMED_UNKNOWN'}
 if($Matches[1] -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_ALREADY_FOREGROUND'}
 Get-PhoneLmThermalBatteryState -Adb $adb -Device $d.Endpoint -Phase 'frontier-preflight'|Out-Null
}
if((Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','ps','-A')).Text -match '(?m)\b(?:fp32_probe|layernorm_probe|one_layer_probe)\s*$'){throw 'EXISTING_CUSTOM_PROBE'}
function Adb {param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
 $r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments
 if($r.ExitCode -ne 0){throw 'ADB_OPERATION_FAILED'}
 return $r.Text
}
$df=Adb @('shell','df','-k','/data/user/0')
$line=@($df -split "`n" | Where-Object {$_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%'})
if($line.Count -ne 1 -or [long](($line[0] -split '\s+')[3])*1024L -lt 12GB){throw 'DEVICE_STORAGE_HEADROOM_LOW_OR_UNKNOWN'}
$research=$Variant -match '^ln|^matmul|^copyNative|^mm'
$executable=if($Variant -match '^one'){'one_layer_probe'}elseif($research){'layernorm_probe'}else{'fp32_probe'}
$id='customfp32-'+[guid]::NewGuid().ToString('N')
$reportBase=if($Variant -in @('oneQkScaleA','oneQkScaleB')){'g1-fp32-qk-score-scale-frontier'}elseif($Variant -eq 'oneQkvHvx'){'ordered-fp32-hvx-qkv'}elseif($Variant -eq 'oneHvx'){'ordered-fp32-hvx-ffn2'}elseif($Variant -match '^oneClosure'){'g1-fp32-segment-closure'}elseif($Variant -in @('oneSoftmax','oneProjection','oneProducer')){'g1-fp32-softmax-ln1-projection'}elseif($Variant -in @('oneAttention','oneLn1','oneObserver')){'g1-fp32-attention-ln1-frontier'}elseif($Variant -in @('oneBroadcast','oneUpstream')){'g1-fp32-broadcast-upstream-frontier'}elseif($Variant -match '^one'){'g1-one-layer-fp32-backward'}elseif($Variant -match '^mm'){'htp-fp32-custom-matmul-ln2'}elseif($research){'htp-fp32-layernorm-matmul'}else{'htp-fp32-custom-consumer'}
$report=Join-Path $root "build/reports/$reportBase/$id"
[IO.Directory]::CreateDirectory($report)|Out-Null
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath (Join-Path $root 'app/build/outputs/apk/debug/app-debug.apk') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath "$report/apk-audit.txt" *> "$report/apk-audit-console.txt"
$remote='/data/local/tmp/hexafp32-'+$build.binary_hashes.$executable.Substring(0,16)
Adb @('shell','mkdir','-p',$remote)|Out-Null
$files=@('libHexaFp32Host.so','libHexaFp32Dsp.so',$executable) | ForEach-Object {Join-Path $root "build/fp32-custom/$_"}
if($Variant -match '^mm'){
 $fixture=Join-Path $root 'build/reports/htp-fp32-custom-matmul-ln2/frozen-inputs.bin'
 if((Get-FileHash -LiteralPath $fixture).Hash.ToLowerInvariant() -ne 'daf61450486f6f8df40b7b10e631c268e9f5e2e5f2458e3a1eda4b0cb8c9978e'){throw 'FROZEN_FIXTURE_IDENTITY_MISMATCH'}
 $files+=$fixture
}
if($Variant -match '^one'){
 $fixture=Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'
 $fixtureIdentity=Get-Content (Join-Path $root 'build/reports/g1-one-layer-fp32-backward/fixture-identity-private.json') -Raw|ConvertFrom-Json
 if((Get-FileHash -LiteralPath $fixture).Hash.ToLowerInvariant() -ne $fixtureIdentity.sha256){throw 'ONE_LAYER_FIXTURE_MISMATCH'}
 $files+=$fixture
}
if($Variant -in @('oneUpstream','oneLn1','oneProjection','oneProducer','oneClosureB','oneClosureC','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB')){
 $ln1Fixture=Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin'
 $ln1Identity=Get-Content (Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-fixture-identity.json') -Raw|ConvertFrom-Json
 if((Get-FileHash -LiteralPath $ln1Fixture).Hash.ToLowerInvariant() -ne $ln1Identity.sha256){throw 'LN1_FIXTURE_IDENTITY_MISMATCH'}
 $files+=$ln1Fixture
}
if($Variant -in @('oneLn1','oneProducer','oneClosureC','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB')){
 $producer=Join-Path $root 'build/reports/g1-fp32-attention-ln1-frontier/ln1-producers.bin'
 $producerIdentity=Get-Content (Join-Path $root 'build/reports/g1-fp32-attention-ln1-frontier/producer-fixture-identity.json') -Raw|ConvertFrom-Json
 if((Get-FileHash -LiteralPath $producer).Hash.ToLowerInvariant() -ne $producerIdentity.sha256){throw 'PRODUCER_FIXTURE_IDENTITY_MISMATCH'}
 $files+=$producer
}
foreach($name in @('libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so')){$files+=Join-Path $QairtSdkRoot "lib/aarch64-android/$name"}
$files+=Join-Path $QairtSdkRoot 'lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so'
$files+=Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'
$hashes=@{}
if($Variant -match '^one'){
 function RemoteHashes {
  $arguments=@('shell','sha256sum')+@($files|ForEach-Object {"$remote/$(Split-Path $_ -Leaf)"})
  $result=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $arguments -AllowFailure
  if($result.ExitCode -ne 0 -and $result.Text -notmatch 'No such file or directory'){throw 'STAGING_STATE_UNKNOWN'}
  $values=@{};foreach($line in $result.Text -split "`n"){
   $match=[regex]::Match($line.Trim(),'^([0-9a-f]{64})\s+(.+)$')
   if($match.Success){$values[$match.Groups[2].Value]=$match.Groups[1].Value}
  };return $values
 }
 $old=RemoteHashes
 foreach($file in $files){$name=Split-Path $file -Leaf;$hash=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant();if($old["$remote/$name"] -ne $hash){Adb @('push',$file,$remote)|Out-Null};$hashes[$name]=$hash}
 $verified=RemoteHashes
 foreach($name in $hashes.Keys){if($verified["$remote/$name"] -ne $hashes[$name]){throw 'STAGED_ARTIFACT_MISMATCH'}}
}else{foreach($file in $files){
 $name=Split-Path $file -Leaf;$hash=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()
 $old=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments @('shell','sha256sum',"$remote/$name") -AllowFailure
 if($old.ExitCode -ne 0 -and $old.Text -notmatch 'No such file or directory'){throw 'STAGING_STATE_UNKNOWN'}
 if($old.ExitCode -ne 0 -or ($old.Text -split '\s+')[0] -ne $hash){Adb @('push',$file,$remote)|Out-Null}
 $remoteHash=((Adb @('shell','sha256sum',"$remote/$name")) -split '\s+')[0]
 if($remoteHash -ne $hash){throw 'STAGED_ARTIFACT_MISMATCH'};$hashes[$name]=$hash
}}
Adb @('shell','chmod','700',"$remote/$executable")|Out-Null
function Top { $s=Adb @('shell','dumpsys','activity','activities');$m=[regex]::Match($s,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/');if(!$m.Success){throw 'TOP_RESUMED_UNKNOWN'};return $m.Groups[1].Value }
$before=Top
if($before -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_ALREADY_FOREGROUND'}
$power=Adb @('shell','dumpsys','power')
if($power -notmatch '(?m)^\s*mWakefulness=Awake\s*$'){throw 'DEVICE_NOT_AWAKE_OR_UNKNOWN'}
$conditionBefore=Adb @('shell','dumpsys','battery')
$thermalBefore=Adb @('shell','dumpsys','thermalservice')
$cmd="cd $remote && export LD_LIBRARY_PATH=$remote && export ADSP_LIBRARY_PATH='$remote;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp' && ./$executable $Variant"
$process=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'shell',('"'+$cmd+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$report/stdout.txt" -RedirectStandardError "$report/stderr.txt"
$takeovers=0;$started=[DateTime]::UtcNow
do {
 if((Top) -eq 'com.yuubinnkyoku.phonelm'){$takeovers++}
 if(([DateTime]::UtcNow-$started).TotalSeconds -gt 120){throw 'CUSTOM_PROBE_TIMEOUT'}
 $process.WaitForExit(2000)|Out-Null;$process.Refresh()
} while(!$process.HasExited)
$conditionAfter=Adb @('shell','dumpsys','battery');$thermalAfter=Adb @('shell','dumpsys','thermalservice')
$ledger=[ordered]@{source_sha=(& git -C $root rev-parse HEAD).Trim();build=$build;branch=(& git -C $root branch --show-current).Trim();variant=$Variant;mode='BACKGROUND_CORRECTNESS';device_private=$d;hashes=$hashes;exit_code=$process.ExitCode;focus_takeover_count=$takeovers;battery_before=$conditionBefore;battery_after=$conditionAfter;thermal_before=$thermalBefore;thermal_after=$thermalAfter;protected_identity='PASS';status='COMPLETE';started_utc=$started.ToString('o')}
[IO.File]::WriteAllText("$report/identity-private.json",($ledger|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
if($takeovers){throw 'FOCUS_TAKEOVER'}
$stdout=Get-Content "$report/stdout.txt" -Raw
if($stdout -match '(?m)^control_nonfinite,' -and $Variant -ne 'oneLn1'){throw 'NONFINITE_CONTROL_OUTSIDE_LN1_DIAGNOSTIC'}
# Standard LN1 negative controls may be nonfinite and remain accuracy FAIL.
# Required Custom outputs still fail closed inside DiagnosticGraph::run.
if($stdout -notmatch '(?m)^summary,' -or $stdout -match '(?m)^(?!control_nonfinite,)(?:\w*RECEIPT_FAIL|\w*NONFINITE\w*|INPUT_MUTATION)'){throw 'CUSTOM_PROBE_INCOMPLETE_OR_UNSAFE'}
Write-Host "variant=$Variant exit=$($process.ExitCode) private_report=$report"
Select-String -Path "$report/stdout.txt" -Pattern '^summary,|^raw_copy,|^result,' | ForEach-Object {$_.Line}
if($Variant -notin @('standardAdd','standardMultiply','htpAdd','htpMultiply','matmul') -and $process.ExitCode -ne 0){throw 'CUSTOM_ACCURACY_NOT_ESTABLISHED'}
