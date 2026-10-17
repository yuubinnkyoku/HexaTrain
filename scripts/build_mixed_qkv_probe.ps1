param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out=Join-Path $root 'build/fp32-mixed-qkv'
if(Test-Path "$out/build-identity.json"){throw 'CANDIDATE_ALREADY_BUILT'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$manifest=Get-Content "$HexagonSdkRoot/hexagon_sdk.json" -Raw|ConvertFrom-Json
$tool=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tool.Count -ne 1 -or $tool[0].build -ne '19.0.07' -or (Get-Content "$NdkRoot/source.properties" -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'PINNED_SDK_MISMATCH'}
$hexbin="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin";$hex="$hexbin/hexagon-clang++.exe";$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
if((Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256 -or (Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256){throw 'COMPILER_HASH_CHANGED'}
$inc="$QairtSdkRoot/include/QNN";$src="$root/host_tests/fp32_custom";$flags=@('-std=c++17','-mv81','-O2','-G0','-fPIC','-fvisibility=hidden','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-DPREPARE_DISABLED',"-I$inc","-I$HexagonSdkRoot/incs","-I$HexagonSdkRoot/incs/stddef","-I$HexagonSdkRoot/rtos/qurt/computev81/include/qurt")
& $hex @flags -fstack-usage -c "$src/mixed_qkv_package.cpp" -o "$out/mixed.o";if($LASTEXITCODE){throw 'DSP_COMPILE'}
& $hex @flags -S "$src/mixed_qkv_package.cpp" -o "$out/mixed.s";if($LASTEXITCODE){throw 'DSP_ASM'}
& $hex @flags -S -emit-llvm "$src/mixed_qkv_package.cpp" -o "$out/mixed.ll";if($LASTEXITCODE){throw 'DSP_IR'}
$hvxFlags=@('-mhvx','-mhvx-length=128B','-mhvx-qfloat','-mllvm','-enable-xqf-gen=true','-mllvm','-hexagon-qfloat-mode=strict-ieee')
& $hex @flags @hvxFlags -fstack-usage -c "$src/hvx_ffn2.cpp" -o "$out/hvx.o";if($LASTEXITCODE){throw 'HVX_COMPILE'}
& $hex @flags @hvxFlags -S "$src/hvx_ffn2.cpp" -o "$out/hvx.s";if($LASTEXITCODE){throw 'HVX_ASM'}
& $hex -mv81 -G0 -shared "$out/mixed.o" "$out/hvx.o" -o "$out/libHexaMixedQkvDsp.so";if($LASTEXITCODE){throw 'DSP_LINK'}
& "$hexbin/hexagon-nm.exe" --undefined-only "$out/libHexaMixedQkvDsp.so" > "$out/undefined-private.txt";if($LASTEXITCODE){throw 'NM'}
& "$hexbin/hexagon-llvm-objdump.exe" -d "$out/libHexaMixedQkvDsp.so" > "$out/disassembly-private.txt";if($LASTEXITCODE){throw 'DISASSEMBLY'}
$asm=Get-Content "$out/mixed.s" -Raw;$ir=Get-Content "$out/mixed.ll" -Raw;$hvx=Get-Content "$out/hvx.s" -Raw
if($asm -notmatch 'dfadd\(' -or $asm -notmatch 'dfmpyfix\(' -or $asm -notmatch 'convert_sf2df' -or $asm -notmatch 'convert_df2sf' -or $ir -match 'llvm\.fma|fadd fast|fmul fast' -or $hvx -notmatch 'vmpy' -or $hvx -notmatch 'vadd'){throw 'CODEGEN_CONTRACT'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -fPIC -shared -ffp-contract=off -fno-fast-math "-I$inc" "$src/mixed_qkv_package.cpp" "-L$QairtSdkRoot/lib/aarch64-android" -lQnnHtpPrepare -o "$out/libHexaMixedQkvHost.so";if($LASTEXITCODE){throw 'HOST_PACKAGE'}
$armflags=@('--target=aarch64-linux-android26','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-static-libstdc++',"-I$inc","-I$src")
& $arm @armflags "$src/mixed_qkv_probe.cpp" -ldl -o "$out/mixed_qkv_probe";if($LASTEXITCODE){throw 'PROBE'}
# Generated wrappers reuse the old graph/validation code verbatim. Only initializer
# is replaced, so every old regression loads all three packages before context.
$text=[IO.File]::ReadAllText("$src/selective_ln1_probe.cpp")
$begin=$text.IndexOf('void initializeSelective(){');$end=$text.IndexOf('struct Graph {',$begin)
if($begin -lt 0 -or $end -le $begin){throw 'INITIALIZER_ANCHOR'}
$text=$text.Substring(0,$begin)+"#include `"mixed_qkv_initialize.h`"`n#define initializeSelective initializeMixed`n"+$text.Substring($end)
[IO.File]::WriteAllText("$out/mixed-s2-generated.cpp",$text,[Text.UTF8Encoding]::new($false))
& $arm @armflags "$out/mixed-s2-generated.cpp" -ldl -o "$out/mixed_s2_regression_probe";if($LASTEXITCODE){throw 'S2_REGRESSION_PROBE'}
$reg="#define main unused_s2_main`n#include `"mixed-s2-generated.cpp`"`n#undef main`n#define initializeProbe initializeMixed`n#include `"one_layer_probe.cpp`"`n"
[IO.File]::WriteAllText("$out/mixed-existing-generated.cpp",$reg,[Text.UTF8Encoding]::new($false))
& $arm @armflags "$out/mixed-existing-generated.cpp" "$root/app/src/main/cpp/tiny_language_model_cpu.cpp" -ldl -o "$out/mixed_existing_regression_probe";if($LASTEXITCODE){throw 'OLD_REGRESSION_PROBE'}
Copy-Item -LiteralPath "$root/build/reports/g1-s2-mixed-qkv-df-hvx-v/host/mixed-inputs.bin" -Destination "$out/mixed-inputs.bin"
$sources=@{};foreach($p in @(Get-ChildItem "$src/mixed_qkv*" -File)){$rel='host_tests/fp32_custom/'+$p.Name;$sources[$rel]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
foreach($p in @('host_tests/fp32_custom/hvx_ffn2.cpp','host_tests/fp32_custom/hvx_ordered_nn.h','host_tests/fp32_custom/probe_support.h')){$sources[$p]=(Get-FileHash "$root/$p").Hash.ToLowerInvariant()}
$hashes=@{};foreach($name in @('libHexaMixedQkvDsp.so','libHexaMixedQkvHost.so','mixed_qkv_probe','mixed_s2_regression_probe','mixed_existing_regression_probe','mixed-inputs.bin')){$hashes[$name]=(Get-FileHash "$out/$name").Hash.ToLowerInvariant()}
$evidence=@{};foreach($name in @('mixed.s','mixed.ll','mixed.o','mixed.su','hvx.s','hvx.o','hvx.su','undefined-private.txt','disassembly-private.txt','mixed-s2-generated.cpp','mixed-existing-generated.cpp')){$evidence[$name]=(Get-FileHash "$out/$name").Hash.ToLowerInvariant()}
@{source_sha=(git -C $root rev-parse HEAD);qairt_build_id=$ExpectedBuildId;source_hashes=$sources;binary_hashes=$hashes;codegen_hashes=$evidence;hexagon_compiler_sha256=(Get-FileHash $hex).Hash.ToLowerInvariant();ndk_compiler_sha256=(Get-FileHash $arm).Hash.ToLowerInvariant();flags=$flags;hvx_flags=$hvxFlags;old_package_bundle=$old;cpu_arithmetic='REJECTED';production_binding=$false}|ConvertTo-Json -Depth 20|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'mixed_build=PASS double_codegen=PASS hvx_codegen=PASS runtime=UNMEASURED'
