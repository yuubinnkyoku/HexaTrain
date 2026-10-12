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
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -fPIC -shared -ffp-contract=off "-I$inc" $src -o "$out/libHexaFp32Host.so"
if($LASTEXITCODE){throw 'CUSTOM_HOST_BUILD_FAILED'}
if(Test-Path (Join-Path $root 'host_tests/fp32_custom/probe.cpp')) {
 & $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -static-libstdc++ "-I$inc" (Join-Path $root 'host_tests/fp32_custom/probe.cpp') -ldl -o "$out/fp32_probe"
 if($LASTEXITCODE){throw 'CUSTOM_PROBE_BUILD_FAILED'}
}
Write-Host 'custom_package_build=PASS'
