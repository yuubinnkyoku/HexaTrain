param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/qairt_version.ps1"
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$hex="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++.exe"
if((Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256 -or
 (Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256){throw 'COMPILER_CHANGED'}
foreach($mode in 0..2){
 $out="$root/build/fp32-saved-p-backward-precision$mode"
 if(Test-Path $out){throw 'EXCLUSIVE_BUNDLE_REQUIRED'}
 [IO.Directory]::CreateDirectory($out)|Out-Null
 & $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -fno-fast-math -static-libstdc++ "-DBACKWARD_PRECISION_MODE=$mode" "-I$QairtSdkRoot/include/QNN" "$root/host_tests/fp32_custom/backward_precision_probe.cpp" -ldl -o "$out/saved_p_backward_probe"
 if($LASTEXITCODE){throw 'PRECISION_PROBE_BUILD'}
 $sources=@{}
 foreach($p in @('host_tests/fp32_custom/backward_precision_probe.cpp','host_tests/fp32_custom/backward_primitive_precision_plan.json','scripts/build_backward_precision_probe.ps1')){$sources[$p]=(Get-FileHash "$root/$p").Hash.ToLowerInvariant()}
 $prior=Get-Content "$root/build/fp32-saved-p-backward-final/build-identity.json" -Raw|ConvertFrom-Json
 foreach($p in $prior.source_hashes.PSObject.Properties){$sources[$p.Name]=$p.Value}
 @{source_sha=(git -C $root rev-parse HEAD);source_hashes=$sources;probe_sha256=(Get-FileHash "$out/saved_p_backward_probe").Hash.ToLowerInvariant();
 gradient_sha256=$prior.gradient_sha256;precision_mode=$mode;qairt_build_id=$ExpectedBuildId;new_DSP_kernel=$false}|ConvertTo-Json -Depth 9|Out-File "$out/build-identity.json" -Encoding utf8
}
Write-Host 'backward_precision_probe_build=PASS configurations=3 production_changed=0'
