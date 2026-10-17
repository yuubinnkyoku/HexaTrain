param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot,[Parameter(Mandatory=$true)][string]$FixturePath)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$manifest=Get-Content (Join-Path $HexagonSdkRoot 'hexagon_sdk.json') -Raw|ConvertFrom-Json
$tool=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tool.Count -ne 1 -or $tool[0].build -ne '19.0.07' -or (Get-Content (Join-Path $NdkRoot 'source.properties') -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'PINNED_COMPILER_SDK_MISMATCH'}
$out=Join-Path $root 'build/fp32-selective-ln1'
if(Test-Path (Join-Path $out 'build-identity.json')){throw 'CANDIDATE_ALREADY_BUILT'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$hexbin=Join-Path $HexagonSdkRoot 'tools/HEXAGON_Tools/19.0.07/Tools/bin'
$hex=Join-Path $hexbin 'hexagon-clang++.exe'
$arm=Join-Path $NdkRoot 'toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe'
$old=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw|ConvertFrom-Json
if((Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256 -or (Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256){throw 'COMPILER_HASH_CHANGED'}
$inc=Join-Path $QairtSdkRoot 'include/QNN'
$src=Join-Path $root 'host_tests/fp32_custom/selective_ln1_package.cpp'
$flags=@('-std=c++17','-mv81','-O2','-G0','-fPIC','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-DPREPARE_DISABLED',"-I$inc","-I$HexagonSdkRoot/incs","-I$HexagonSdkRoot/incs/stddef","-I$HexagonSdkRoot/rtos/qurt/computev81/include/qurt")
& $hex @flags -fstack-usage -c $src -o "$out/selective.o";if($LASTEXITCODE){throw 'DSP_OBJECT_FAILED'}
& $hex -mv81 -G0 -shared "$out/selective.o" -o "$out/libHexaSelectiveLn1Dsp.so";if($LASTEXITCODE){throw 'DSP_LINK_FAILED'}
& (Join-Path $hexbin 'hexagon-nm.exe') --undefined-only "$out/libHexaSelectiveLn1Dsp.so" > "$out/dsp-undefined-private.txt";if($LASTEXITCODE){throw 'NM_FAILED'}
& (Join-Path $hexbin 'hexagon-llvm-objdump.exe') -d "$out/selective.o" > "$out/disassembly-private.txt";if($LASTEXITCODE){throw 'DISASSEMBLY_FAILED'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -fPIC -shared -ffp-contract=off -fno-fast-math "-I$inc" $src "-L$QairtSdkRoot/lib/aarch64-android" -lQnnHtpPrepare -o "$out/libHexaSelectiveLn1Host.so";if($LASTEXITCODE){throw 'HOST_PACKAGE_FAILED'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -ffp-contract=off -fno-fast-math -static-libstdc++ "-I$inc" (Join-Path $root 'host_tests/fp32_custom/selective_ln1_probe.cpp') -ldl -o "$out/selective_ln1_probe";if($LASTEXITCODE){throw 'PROBE_FAILED'}
Copy-Item -LiteralPath $FixturePath -Destination "$out/selective-ln1-inputs.bin"
$sources=@{};foreach($p in @('host_tests/fp32_custom/selective_ln1_package.cpp','host_tests/fp32_custom/selective_ln1_probe.cpp','host_tests/fp32_custom/selective_ln1_fixture.h','host_tests/fp32_custom/selective_ln1_arithmetic.h','host_tests/fp32_custom/selective_ln1_device_plan.json','host_tests/fp32_custom/layernorm_ops.h','host_tests/fp32_custom/probe_support.h')){$sources[$p]=(Get-FileHash (Join-Path $root $p)).Hash.ToLowerInvariant()}
$binaries=@{};foreach($p in @('libHexaSelectiveLn1Dsp.so','libHexaSelectiveLn1Host.so','selective_ln1_probe','selective-ln1-inputs.bin')){$binaries[$p]=(Get-FileHash "$out/$p").Hash.ToLowerInvariant()}
@{source_sha=(git -C $root rev-parse HEAD);qairt_build_id=$ExpectedBuildId;source_hashes=$sources;binary_hashes=$binaries;old_package_bundle=$old;hexagon_compiler_sha256=(Get-FileHash $hex).Hash.ToLowerInvariant();ndk_compiler_sha256=(Get-FileHash $arm).Hash.ToLowerInvariant();flags=$flags;geometry='LN1 T32 D64 S2 only';cpu_execution_in_host_package='REJECTED';production_graph_changed=$false}|ConvertTo-Json -Depth 15|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'selective_ln1_build=PASS runtime_loading=UNVERIFIED old_package_unchanged=1'
