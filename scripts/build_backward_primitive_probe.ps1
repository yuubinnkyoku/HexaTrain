param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$NdkRoot,[ValidatePattern('^[a-z0-9-]{1,32}$')][string]$Candidate='primitive-v2')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/qairt_version.ps1"
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out="$root/build/fp32-saved-p-backward-$Candidate"
if(Test-Path $out){throw 'EXCLUSIVE_BUNDLE_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
if((Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256){throw 'COMPILER_CHANGED'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -fno-fast-math -static-libstdc++ "-I$QairtSdkRoot/include/QNN" "$root/host_tests/fp32_custom/backward_primitive_probe.cpp" -ldl -o "$out/saved_p_backward_probe"
if($LASTEXITCODE){throw 'PRIMITIVE_BUILD'}
$sources=@{}
foreach($p in @('host_tests/fp32_custom/backward_primitive_probe.cpp','host_tests/fp32_custom/backward_primitive_precision_plan.json','scripts/build_backward_primitive_probe.ps1')){$sources[$p]=(Get-FileHash "$root/$p").Hash.ToLowerInvariant()}
$prior=Get-Content "$root/build/fp32-saved-p-backward-final/build-identity.json" -Raw|ConvertFrom-Json
foreach($p in $prior.source_hashes.PSObject.Properties){$sources[$p.Name]=$p.Value}
@{source_sha=(git -C $root rev-parse HEAD);source_hashes=$sources;probe_sha256=(Get-FileHash "$out/saved_p_backward_probe").Hash.ToLowerInvariant();
 gradient_sha256=$prior.gradient_sha256;qairt_build_id=$ExpectedBuildId;new_DSP_kernel=$false;controls=576}|ConvertTo-Json -Depth 9|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'backward_primitive_probe_build=PASS controls=576 production_changed=0'
