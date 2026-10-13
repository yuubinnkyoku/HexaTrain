param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,
 [Parameter(Mandatory=$true)][string]$NdkRoot
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/fp32-custom'
[IO.Directory]::CreateDirectory($out)|Out-Null
$manifest=Get-Content (Join-Path $HexagonSdkRoot 'hexagon_sdk.json') -Raw | ConvertFrom-Json
$tools=@($manifest.root.tools.info | Where-Object {$_.name -eq 'Hexagon Tools'})
if($tools.Count -ne 1 -or $tools[0].build -ne '19.0.07'){throw 'PINNED_HEXAGON_TOOLS_MISMATCH'}
if((Get-Content (Join-Path $NdkRoot 'source.properties') -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'PINNED_NDK_MISMATCH'}
$hex=Join-Path $HexagonSdkRoot 'tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++.exe'
$arm=Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe'
$src=Join-Path $root 'host_tests/fp32_custom/package.cpp'
$inc=Join-Path $QairtSdkRoot 'include/QNN'
& $hex -std=c++17 -mv81 -O2 -G0 -fPIC -shared -ffp-contract=off -fno-vectorize -fno-slp-vectorize -DPREPARE_DISABLED "-I$inc" "-I$HexagonSdkRoot/incs" "-I$HexagonSdkRoot/incs/stddef" "-I$HexagonSdkRoot/rtos/qurt/computev81/include/qurt" $src -o "$out/libHexaFp32Dsp.so"
if($LASTEXITCODE){throw 'CUSTOM_DSP_BUILD_FAILED'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -fPIC -shared -ffp-contract=off "-I$inc" $src "-L$QairtSdkRoot/lib/aarch64-android" -lQnnHtpPrepare -o "$out/libHexaFp32Host.so"
if($LASTEXITCODE){throw 'CUSTOM_HOST_BUILD_FAILED'}
if(Test-Path (Join-Path $root 'host_tests/fp32_custom/probe.cpp')) {
 & $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -static-libstdc++ "-I$inc" (Join-Path $root 'host_tests/fp32_custom/probe.cpp') -ldl -o "$out/fp32_probe"
 if($LASTEXITCODE){throw 'CUSTOM_PROBE_BUILD_FAILED'}
}
Write-Host 'custom_package_build=PASS'
if(Test-Path (Join-Path $root 'host_tests/fp32_custom/layernorm_probe.cpp')){
 & $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -static-libstdc++ "-I$inc" (Join-Path $root 'host_tests/fp32_custom/layernorm_probe.cpp') -ldl -o "$out/layernorm_probe"
 if($LASTEXITCODE){throw 'LAYERNORM_PROBE_BUILD_FAILED'}
}
$sourceHashes=@{}
foreach($relative in @('host_tests/fp32_custom/package.cpp','host_tests/fp32_custom/probe.cpp',
 'app/src/main/cpp/qnn/layernorm_stability_reference.h','app/src/main/cpp/qnn/layernorm_redesign_reference.h','host_tests/hvx_rpc/fp32_boundary_fixture.h')){
 $sourceHashes[$relative]=(Get-FileHash -LiteralPath (Join-Path $root $relative)).Hash.ToLowerInvariant()
}
$binaryHashes=@{}
foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','fp32_probe')){
 $binaryHashes[$name]=(Get-FileHash -LiteralPath (Join-Path $out $name)).Hash.ToLowerInvariant()
}
foreach($relative in @('host_tests/fp32_custom/layernorm_ops.h','host_tests/fp32_custom/probe_support.h','host_tests/fp32_custom/layernorm_probe.cpp','host_tests/fp32_custom/matmul_probe.h','host_tests/fp32_custom/custom_matmul_ops.h','host_tests/fp32_custom/custom_matmul_probe.h','host_tests/fp32_custom/custom_matmul_fixture.h')){
 if(Test-Path (Join-Path $root $relative)){$sourceHashes[$relative]=(Get-FileHash (Join-Path $root $relative)).Hash.ToLowerInvariant()}
}
if(Test-Path "$out/layernorm_probe"){$binaryHashes['layernorm_probe']=(Get-FileHash "$out/layernorm_probe").Hash.ToLowerInvariant()}
$identity=@{source_sha=(& git -C $root rev-parse HEAD).Trim();source_hashes=$sourceHashes;binary_hashes=$binaryHashes;
 qairt_build_id=$ExpectedBuildId;hexagon_compiler_sha256=(Get-FileHash $hex).Hash.ToLowerInvariant();
 ndk_compiler_sha256=(Get-FileHash $arm).Hash.ToLowerInvariant();fp_contract='off';vectorization='disabled';architecture='v81'}
[IO.File]::WriteAllText("$out/build-identity.json",($identity|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
