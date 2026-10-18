param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot,
 [ValidatePattern('^[a-z0-9-]{1,32}$')][string]$Candidate='repair-v4')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/qairt_version.ps1"
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out="$root/build/fp32-saved-p-backward-$Candidate"
if(Test-Path $out){throw 'EXCLUSIVE_REPAIR_BUNDLE_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$old=Get-Content "$root/build/fp32-custom/build-identity.json" -Raw|ConvertFrom-Json
$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$hexbin="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin";$hex="$hexbin/hexagon-clang++.exe"
$hostcc=(Get-Command g++).Source
$plan=Get-Content "$root/host_tests/fp32_custom/saved_p_backward_device_plan.json" -Raw|ConvertFrom-Json
if((Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256 -or (Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256 -or (Get-FileHash $hostcc).Hash.ToLowerInvariant() -ne $plan.toolchain.host_compiler_sha256){throw 'COMPILER_CHANGED'}
$manifest=Get-Content "$HexagonSdkRoot/hexagon_sdk.json" -Raw|ConvertFrom-Json
$tools=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tools.Count -ne 1 -or $tools[0].build -ne '19.0.07' -or (Get-Content "$NdkRoot/source.properties" -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'FIXED_TOOLCHAIN_CHANGED'}
$src="$root/host_tests/fp32_custom";$inc="$QairtSdkRoot/include/QNN"
$common=@('-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',"-I$inc")
foreach($name in @('contract','prepare','graph')){
 $source=if($name -eq 'graph'){"$src/backward_repair_graph_test.cpp"}else{"$src/backward_repair_contract_test.cpp"}
 $extra=@(if($name -eq 'contract'){'-DBACKWARD_REPAIR_HOST_CONTRACT'})
 & $hostcc @common @extra $source "$root/app/src/main/cpp/tiny_language_model_cpu.cpp" "$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp" -o "$out/$name.exe" *> "$out/$name-build.txt"
 if($LASTEXITCODE){throw "HOST_BUILD $name"}
 if($name -eq 'contract'){& "$out/$name.exe" "$root/build/fp32-double-qk/double-inputs.bin" "$root/build/reports/g1-native-saved-p-backward-dsp/gradients.bin" *> "$out/$name.txt"}else{& "$out/$name.exe" *> "$out/$name.txt"}
 if($LASTEXITCODE){throw "HOST_GATE $name"}
}
$flags=@('-std=c++17','-mv81','-O2','-G0','-fPIC','-fvisibility=hidden','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-DPREPARE_DISABLED',"-I$inc","-I$HexagonSdkRoot/incs","-I$HexagonSdkRoot/incs/stddef","-I$HexagonSdkRoot/rtos/qurt/computev81/include/qurt")
& $hex @flags -fstack-usage -c "$src/backward_repair_package.cpp" -o "$out/repair.o" *> "$out/dsp-build.txt";if($LASTEXITCODE){throw 'DSP_COMPILE'}
& $hex @flags -S "$src/backward_repair_package.cpp" -o "$out/repair.s";if($LASTEXITCODE){throw 'DSP_ASM'}
& $hex @flags -S -emit-llvm "$src/backward_repair_package.cpp" -o "$out/repair.ll";if($LASTEXITCODE){throw 'DSP_IR'}
& $hex -mv81 -G0 -shared "$out/repair.o" -o "$out/libHexaBackwardRepairDsp.so";if($LASTEXITCODE){throw 'DSP_LINK'}
& "$hexbin/hexagon-nm.exe" --undefined-only "$out/libHexaBackwardRepairDsp.so" > "$out/undefined-private.txt";if($LASTEXITCODE){throw 'NM'}
& "$hexbin/hexagon-llvm-objdump.exe" -d "$out/libHexaBackwardRepairDsp.so" > "$out/disassembly-private.txt";if($LASTEXITCODE){throw 'DISASSEMBLY'}
$asm=Get-Content "$out/repair.s" -Raw;$ir=Get-Content "$out/repair.ll" -Raw
if($asm -notmatch 'sfmpy\(' -or $asm -notmatch 'sfadd\(' -or $asm -notmatch 'sfsub\(' -or $ir -match 'llvm\.fma|fadd fast|fmul fast|fsub fast|llvm\.vector\.reduce'){throw 'FP32_CODEGEN_CONTRACT'}
& $arm @common --target=aarch64-linux-android26 -fPIC -shared "$src/backward_repair_package.cpp" "-L$QairtSdkRoot/lib/aarch64-android" -lQnnHtpPrepare -o "$out/libHexaBackwardRepairHost.so";if($LASTEXITCODE){throw 'HOST_PACKAGE'}
& $arm @common --target=aarch64-linux-android26 -static-libstdc++ "$src/backward_repair_probe.cpp" -ldl -o "$out/saved_p_backward_probe";if($LASTEXITCODE){throw 'REPAIR_PROBE'}
$sources=@{}
foreach($p in @(Get-ChildItem "$src/backward_repair*" -File)+@(Get-Item "$src/backward_primitive_precision_plan.json")){$sources['host_tests/fp32_custom/'+$p.Name]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
foreach($p in @('scripts/build_backward_repair_probe.ps1','scripts/run_backward_repair_probe.ps1')){$sources[$p]=(Get-FileHash "$root/$p").Hash.ToLowerInvariant()}
$prior=Get-Content "$root/build/fp32-saved-p-backward-final/build-identity.json" -Raw|ConvertFrom-Json
foreach($p in $prior.source_hashes.PSObject.Properties){$sources[$p.Name]=$p.Value}
$binaries=@{};foreach($p in @('libHexaBackwardRepairHost.so','libHexaBackwardRepairDsp.so','saved_p_backward_probe')){$binaries[$p]=(Get-FileHash "$out/$p").Hash.ToLowerInvariant()}
$evidence=@{};foreach($p in @('repair.s','repair.ll','repair.su','undefined-private.txt','disassembly-private.txt','contract.txt','prepare.txt','graph.txt')){$evidence[$p]=(Get-FileHash "$out/$p").Hash.ToLowerInvariant()}
@{source_sha=(git -C $root rev-parse HEAD);bundle_leaf=(Split-Path $out -Leaf);source_hashes=$sources;binary_hashes=$binaries;codegen_host_hashes=$evidence;probe_sha256=$binaries.saved_p_backward_probe;gradient_sha256=$prior.gradient_sha256;
 qairt_build_id=$ExpectedBuildId;host_compiler_sha256=$plan.toolchain.host_compiler_sha256;ndk_compiler_sha256=$old.ndk_compiler_sha256;hexagon_compiler_sha256=$old.hexagon_compiler_sha256;flags=$flags;new_DSP_kernel=$true;production_binding=$false;cpu_arithmetic='REJECTED';candidate='ordered-fp32-v1'}|ConvertTo-Json -Depth 10|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host 'backward_repair_build=PASS host_contract=PASS host_prepare_reject=PASS graph=PASS scalar_FP32_codegen=PASS runtime=UNMEASURED'
