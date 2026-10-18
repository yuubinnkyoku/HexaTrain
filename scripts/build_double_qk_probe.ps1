param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out=Join-Path $root 'build/fp32-double-qk'
if(Test-Path "$out/build-identity.json"){throw 'CANDIDATE_ALREADY_BUILT'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$manifest=Get-Content "$HexagonSdkRoot/hexagon_sdk.json" -Raw|ConvertFrom-Json
$tool=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tool.Count -ne 1 -or $tool[0].build -ne '19.0.07' -or (Get-Content "$NdkRoot/source.properties" -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'PINNED_SDK_MISMATCH'}
$hexbin="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin";$hex="$hexbin/hexagon-clang++.exe";$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
if((Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256 -or (Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256){throw 'COMPILER_HASH_CHANGED'}
$host=Get-Content "$root/build/reports/g1-s2-double-qk-native-p-closure/host-final/identity-private.json" -Raw|ConvertFrom-Json
if($host.tests.host_joint_exit -ne 0 -or $host.source_sha -ne (git -C $root rev-parse HEAD)){throw 'HOST_SOURCE_OR_JOINT_GATE'}
$inc="$QairtSdkRoot/include/QNN";$src="$root/host_tests/fp32_custom";$flags=@('-std=c++17','-mv81','-O2','-G0','-fPIC','-fvisibility=hidden','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-DPREPARE_DISABLED',"-I$inc","-I$HexagonSdkRoot/incs","-I$HexagonSdkRoot/incs/stddef","-I$HexagonSdkRoot/rtos/qurt/computev81/include/qurt")
& $hex @flags -fstack-usage -c "$src/double_qk_package.cpp" -o "$out/double.o";if($LASTEXITCODE){throw 'DSP_COMPILE'}
& $hex @flags -S "$src/double_qk_package.cpp" -o "$out/double.s";if($LASTEXITCODE){throw 'DSP_ASM'}
& $hex @flags -S -emit-llvm "$src/double_qk_package.cpp" -o "$out/double.ll";if($LASTEXITCODE){throw 'DSP_IR'}
& $hex -mv81 -G0 -shared "$out/double.o" -o "$out/libHexaDoubleQkDsp.so";if($LASTEXITCODE){throw 'DSP_LINK'}
& "$hexbin/hexagon-nm.exe" --undefined-only "$out/libHexaDoubleQkDsp.so" > "$out/undefined-private.txt";if($LASTEXITCODE){throw 'NM'}
& "$hexbin/hexagon-llvm-objdump.exe" -d "$out/libHexaDoubleQkDsp.so" > "$out/disassembly-private.txt";if($LASTEXITCODE){throw 'DISASSEMBLY'}
$asm=Get-Content "$out/double.s" -Raw;$ir=Get-Content "$out/double.ll" -Raw
if($asm -notmatch 'dfadd\(' -or $asm -notmatch 'dfmpyfix\(' -or $asm -notmatch 'convert_sf2df' -or $asm -notmatch 'convert_df2sf' -or $ir -match 'llvm\.fma|fadd fast|fmul fast'){throw 'CODEGEN_CONTRACT'}
& $arm --target=aarch64-linux-android26 -std=c++17 -O2 -fPIC -shared -ffp-contract=off -fno-fast-math "-I$inc" "$src/double_qk_package.cpp" "-L$QairtSdkRoot/lib/aarch64-android" -lQnnHtpPrepare -o "$out/libHexaDoubleQkHost.so";if($LASTEXITCODE){throw 'HOST_PACKAGE'}
$armflags=@('--target=aarch64-linux-android26','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-static-libstdc++',"-I$inc","-I$src")
& $arm @armflags "$src/double_qk_probe.cpp" -ldl -o "$out/double_qk_probe";if($LASTEXITCODE){throw 'PROBE'}
# Old graph/math/checks retained verbatim; initialization alone registers all four packages.
$text=[IO.File]::ReadAllText("$src/mixed_qkv_probe.cpp");$anchor='#include "mixed_qkv_initialize.h"'
if(([regex]::Matches($text,[regex]::Escape($anchor))).Count -ne 1){throw 'MIXED_INITIALIZER_ANCHOR'}
$text=$text.Replace($anchor,"#include `"double_qk_initialize.h`"`n#define initializeMixed initializeDoubleQk")
[IO.File]::WriteAllText("$out/double-mixed-generated.cpp",$text,[Text.UTF8Encoding]::new($false))
& $arm @armflags "$out/double-mixed-generated.cpp" -ldl -o "$out/double_mixed_regression_probe";if($LASTEXITCODE){throw 'MIXED_REGRESSION'}
$text=[IO.File]::ReadAllText("$src/selective_ln1_probe.cpp");$begin=$text.IndexOf('void initializeSelective(){');$end=$text.IndexOf('struct Graph {',$begin)
if($begin -lt 0 -or $end -le $begin){throw 'S2_INITIALIZER_ANCHOR'}
$text=$text.Substring(0,$begin)+"#include `"double_qk_initialize.h`"`n#define initializeSelective initializeDoubleQk`n"+$text.Substring($end)
[IO.File]::WriteAllText("$out/double-s2-generated.cpp",$text,[Text.UTF8Encoding]::new($false))
& $arm @armflags "$out/double-s2-generated.cpp" -ldl -o "$out/double_s2_regression_probe";if($LASTEXITCODE){throw 'S2_REGRESSION'}
$reg="#define main unused_s2_main`n#include `"double-s2-generated.cpp`"`n#undef main`n#define initializeProbe initializeDoubleQk`n#include `"one_layer_probe.cpp`"`n"
[IO.File]::WriteAllText("$out/double-existing-generated.cpp",$reg,[Text.UTF8Encoding]::new($false))
& $arm @armflags "$out/double-existing-generated.cpp" "$root/app/src/main/cpp/tiny_language_model_cpu.cpp" -ldl -o "$out/double_existing_regression_probe";if($LASTEXITCODE){throw 'EXISTING_REGRESSION'}
Copy-Item -LiteralPath "$root/build/reports/g1-s2-double-qk-native-p-closure/host-final/double-1.bin" -Destination "$out/double-inputs.bin"
$sources=@{};foreach($p in @(Get-ChildItem "$src/double_qk*" -File)){$rel='host_tests/fp32_custom/'+$p.Name;$sources[$rel]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
$hashes=@{};foreach($name in @('libHexaDoubleQkDsp.so','libHexaDoubleQkHost.so','double_qk_probe','double_mixed_regression_probe','double_s2_regression_probe','double_existing_regression_probe','double-inputs.bin')){$hashes[$name]=(Get-FileHash "$out/$name").Hash.ToLowerInvariant()}
$evidence=@{};foreach($name in @('double.s','double.ll','double.o','double.su','undefined-private.txt','disassembly-private.txt','double-mixed-generated.cpp','double-s2-generated.cpp','double-existing-generated.cpp')){$evidence[$name]=(Get-FileHash "$out/$name").Hash.ToLowerInvariant()}
$deps=@{};foreach($name in @('fp32-custom','fp32-selective-ln1','fp32-mixed-qkv')){$deps[$name]=Get-Content "$root/build/$name/build-identity.json" -Raw|ConvertFrom-Json}
@{source_sha=(git -C $root rev-parse HEAD);qairt_build_id=$ExpectedBuildId;source_hashes=$sources;binary_hashes=$hashes;codegen_hashes=$evidence;hexagon_compiler_sha256=(Get-FileHash $hex).Hash.ToLowerInvariant();ndk_compiler_sha256=(Get-FileHash $arm).Hash.ToLowerInvariant();flags=$flags;dependency_bundles=$deps;cpu_arithmetic='REJECTED';production_binding=$false}|ConvertTo-Json -Depth 30|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'double_qk_build=PASS double_codegen=PASS old_math_unchanged=1 runtime=UNMEASURED'
