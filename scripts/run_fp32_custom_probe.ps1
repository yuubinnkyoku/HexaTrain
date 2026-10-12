param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$NdkRoot,
 [ValidateSet('Copy','Add','Multiply','Affine','chain','chainTap','reshape','htpAdd','htpMultiply','standardAdd','standardMultiply')][string]$Variant='Copy'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(& git -C $root status --porcelain){throw 'SOURCE_MUST_BE_COMMITTED'}
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
function Adb([string[]]$Arguments){$r=Invoke-PhoneLmAdb -Adb $adb -Device $d.Endpoint -Arguments $Arguments;return $r.Text}
$df=Adb @('shell','df','-k','/data/user/0')
$line=@($df -split "`n" | Where-Object {$_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%'})
if($line.Count -ne 1 -or [long](($line[0] -split '\s+')[3])*1024L -lt 12GB){throw 'DEVICE_STORAGE_HEADROOM_LOW_OR_UNKNOWN'}
$id='customfp32-'+[guid]::NewGuid().ToString('N')
$report=Join-Path $root "build/reports/htp-fp32-custom-consumer/$id"
[IO.Directory]::CreateDirectory($report)|Out-Null
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath (Join-Path $root 'app/build/outputs/apk/debug/app-debug.apk') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath "$report/apk-audit.txt" *> "$report/apk-audit-console.txt"
$remote="/data/local/tmp/$id"
Adb @('shell','mkdir','-p',$remote)|Out-Null
$files=@('libHexaFp32Host.so','libHexaFp32Dsp.so','fp32_probe') | ForEach-Object {Join-Path $root "build/fp32-custom/$_"}
foreach($name in @('libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so')){$files+=Join-Path $QairtSdkRoot "lib/aarch64-android/$name"}
$files+=Join-Path $QairtSdkRoot 'lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so'
$files+=Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'
$hashes=@{}
foreach($file in $files){
 $name=Split-Path $file -Leaf;$hash=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()
 Adb @('push',$file,$remote)|Out-Null
 $remoteHash=(Adb @('shell','sha256sum',"$remote/$name") -split '\s+')[0]
 if($remoteHash -ne $hash){throw 'STAGED_ARTIFACT_MISMATCH'};$hashes[$name]=$hash
}
Adb @('shell','chmod','700',"$remote/fp32_probe")|Out-Null
function Top { $s=Adb @('shell','dumpsys','activity','activities');$m=[regex]::Match($s,'(?m)^\s*topResumedActivity=.*\su\d+\s+([^/\s]+)/');if(!$m.Success){throw 'TOP_RESUMED_UNKNOWN'};return $m.Groups[1].Value }
$before=Top
if($before -eq 'com.yuubinnkyoku.phonelm'){throw 'TARGET_ALREADY_FOREGROUND'}
$conditionBefore=Adb @('shell','dumpsys','battery')
$thermalBefore=Adb @('shell','dumpsys','thermalservice')
$cmd="cd $remote && export LD_LIBRARY_PATH=$remote && export ADSP_LIBRARY_PATH='$remote;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp' && ./fp32_probe $Variant"
$process=Start-Process -FilePath $adb -ArgumentList @('-s',$d.Endpoint,'shell',('"'+$cmd+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$report/stdout.txt" -RedirectStandardError "$report/stderr.txt"
$takeovers=0;$started=[DateTime]::UtcNow
do {
 if((Top) -eq 'com.yuubinnkyoku.phonelm'){$takeovers++}
 if(([DateTime]::UtcNow-$started).TotalSeconds -gt 120){throw 'CUSTOM_PROBE_TIMEOUT'}
 $process.WaitForExit(2000)|Out-Null;$process.Refresh()
} while(!$process.HasExited)
$conditionAfter=Adb @('shell','dumpsys','battery');$thermalAfter=Adb @('shell','dumpsys','thermalservice')
$ledger=[ordered]@{source_sha=(& git -C $root rev-parse HEAD).Trim();branch=(& git -C $root branch --show-current).Trim();variant=$Variant;mode='BACKGROUND_CORRECTNESS';device_private=$d;hashes=$hashes;exit_code=$process.ExitCode;focus_takeover_count=$takeovers;battery_before=$conditionBefore;battery_after=$conditionAfter;thermal_before=$thermalBefore;thermal_after=$thermalAfter;protected_identity='PASS';status='COMPLETE';started_utc=$started.ToString('o')}
[IO.File]::WriteAllText("$report/identity-private.json",($ledger|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
if($takeovers){throw 'FOCUS_TAKEOVER'}
$stdout=Get-Content "$report/stdout.txt" -Raw
if($stdout -notmatch 'summary,' -or $stdout -match 'RECEIPT_FAIL|NONFINITE|INPUT_MUTATION'){throw 'CUSTOM_PROBE_INCOMPLETE_OR_UNSAFE'}
Write-Host "variant=$Variant exit=$($process.ExitCode) private_report=$report"
Select-String -Path "$report/stdout.txt" -Pattern '^summary,|^raw_copy,|^result,' | ForEach-Object {$_.Line}
if($Variant -notin @('standardAdd','standardMultiply','htpAdd','htpMultiply') -and $process.ExitCode -ne 0){throw 'CUSTOM_ACCURACY_NOT_ESTABLISHED'}
