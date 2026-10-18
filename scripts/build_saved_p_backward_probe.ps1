param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out=Join-Path $root 'build/fp32-saved-p-backward'
if(Test-Path "$out/build-identity.json"){throw 'CANDIDATE_ALREADY_BUILT'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$plan=Get-Content "$root/host_tests/fp32_custom/saved_p_backward_device_plan.json" -Raw|ConvertFrom-Json
$manifest=Get-Content "$HexagonSdkRoot/hexagon_sdk.json" -Raw|ConvertFrom-Json
$tools=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tools.Count -ne 1 -or $tools[0].build -ne '19.0.07' -or
 (Get-Content "$NdkRoot/source.properties" -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'PINNED_TOOLCHAIN_MISMATCH'}
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$hex="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++.exe"
if((Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256 -or
 (Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256){throw 'COMPILER_IDENTITY_CHANGED'}
$evidence="$root/build/reports/g1-native-saved-p-backward-dsp"
if((Get-Content "$evidence/reference-selftest.txt" -Raw) -notmatch 'saved_p_backward_reference_selftest=PASS' -or
 (Get-Content "$evidence/graph-test.txt" -Raw) -notmatch 'saved_p_backward_graph_host=PASS'){throw 'HOST_GATE_NOT_PASS'}
if((Get-FileHash "$evidence/gradients.bin").Hash.ToLowerInvariant() -ne $plan.gradient_sha256){throw 'GRADIENT_IDENTITY_CHANGED'}
foreach($name in @('fp32-custom','fp32-selective-ln1','fp32-mixed-qkv','fp32-double-qk')) {
 $bundle=Get-Content "$root/build/$name/build-identity.json" -Raw|ConvertFrom-Json
 foreach($p in $bundle.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FORWARD_SOURCE_CHANGED'}}
 foreach($p in $bundle.binary_hashes.PSObject.Properties){if((Get-FileHash "$root/build/$name/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FORWARD_BINARY_CHANGED'}}
}
$src="$root/host_tests/fp32_custom"
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -fno-fast-math -static-libstdc++ "-I$QairtSdkRoot/include/QNN" "-I$src" "$src/saved_p_backward_probe.cpp" -ldl -o "$out/saved_p_backward_probe"
if($LASTEXITCODE){throw 'PROBE_BUILD_FAILED'}
$sources=@{}
foreach($p in @(Get-ChildItem "$src/saved_p_backward*" -File)){$sources['host_tests/fp32_custom/'+$p.Name]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
$sources['scripts/build_saved_p_backward_probe.ps1']=(Get-FileHash $PSCommandPath).Hash.ToLowerInvariant()
$sources['scripts/run_saved_p_backward_probe.ps1']=(Get-FileHash "$PSScriptRoot/run_saved_p_backward_probe.ps1").Hash.ToLowerInvariant()
@{source_sha=(git -C $root rev-parse HEAD);source_hashes=$sources;probe_sha256=(Get-FileHash "$out/saved_p_backward_probe").Hash.ToLowerInvariant();
 gradient_sha256=$plan.gradient_sha256;plan_sha256=(Get-FileHash "$src/saved_p_backward_device_plan.json").Hash.ToLowerInvariant();
 qairt_build_id=$ExpectedBuildId;ndk_compiler_sha256=$old.ndk_compiler_sha256;hexagon_compiler_sha256=$old.hexagon_compiler_sha256;
 dsp_arithmetic='unchanged protected packages plus standard qti.aisw operators';new_DSP_kernel=$false}|ConvertTo-Json -Depth 8|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'saved_p_backward_probe_build=PASS new_DSP_kernel=0'
