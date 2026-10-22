param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$NdkRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/qairt_version.ps1"
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
if(git -C $root status --porcelain){throw 'COMMITTED_SOURCE_REQUIRED'}
$out="$root/build/fp32-saved-p-backward-shared-lifetime-v1"
if(Test-Path $out){throw 'EXCLUSIVE_DIAGNOSTIC_BUNDLE_REQUIRED'}
$original="$root/build/fp32-saved-p-backward-repair-final"
$old=Get-Content "$original/build-identity.json" -Raw|ConvertFrom-Json
if($old.source_sha -ne 'bb6aaceb42de505b29689a362ce8293687e7ab89'){throw 'ORIGINAL_SOURCE_IDENTITY'}
foreach($p in $old.source_hashes.PSObject.Properties){if((Get-FileHash "$root/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FROZEN_SOURCE_CHANGED'}}
foreach($p in $old.binary_hashes.PSObject.Properties){if((Get-FileHash "$original/$($p.Name)").Hash.ToLowerInvariant() -ne $p.Value){throw 'FROZEN_BINARY_CHANGED'}}
$arm="$NdkRoot/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe"
$hex="$HexagonSdkRoot/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++.exe"
$cc=(Get-Command g++).Source
if((Get-FileHash $arm).Hash.ToLowerInvariant() -ne $old.ndk_compiler_sha256 -or
 (Get-FileHash $hex).Hash.ToLowerInvariant() -ne $old.hexagon_compiler_sha256 -or
 (Get-FileHash $cc).Hash.ToLowerInvariant() -ne $old.host_compiler_sha256){throw 'FIXED_COMPILER_IDENTITY'}
$sdk=Get-Content "$HexagonSdkRoot/hexagon_sdk.json" -Raw|ConvertFrom-Json
$tools=@($sdk.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($sdk.root.id.version -ne '6.6.0.0' -or $tools.Count -ne 1 -or $tools[0].build -ne '19.0.07' -or
 (Get-Content "$NdkRoot/source.properties" -Raw) -notmatch 'Pkg.Revision\s*=\s*26.2.11394342'){throw 'FIXED_TOOLCHAIN_IDENTITY'}
& pwsh -NoProfile -File "$PSScriptRoot/check_qairt.ps1" -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
if($LASTEXITCODE -notin @(0,3)){throw 'FIXED_QAIRT_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$src="$root/host_tests/fp32_custom";$flags=@('-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',"-I$QairtSdkRoot/include/QNN")
$support=@("$root/app/src/main/cpp/tiny_language_model_cpu.cpp","$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp")
foreach($test in @('backward_shared_lifetime_test','backward_repair_contract_test','backward_repair_graph_test','saved_p_backward_reference_test','saved_p_oracle_independent_test','saved_p_backward_graph_test')){
 $extra=@();if($test -eq 'backward_repair_contract_test'){$extra=@('-DBACKWARD_REPAIR_HOST_CONTRACT')}
 & $cc @flags @extra "$src/$test.cpp" @support -o "$out/$test.exe" *> "$out/$test-build.txt"
 if($LASTEXITCODE){throw "HOST_BUILD $test"}
 if($test -eq 'backward_repair_contract_test'){
  & "$out/$test.exe" "$root/build/fp32-double-qk/double-inputs.bin" "$root/build/reports/g1-native-saved-p-backward-dsp/gradients.bin" *> "$out/$test.txt"
 }else{& "$out/$test.exe" *> "$out/$test.txt"}
 if($LASTEXITCODE){throw "HOST_TEST $test"}
}
& $cc @flags "$src/backward_repair_contract_test.cpp" @support -o "$out/prepare.exe" *> "$out/prepare-build.txt"
if($LASTEXITCODE){throw 'HOST_PREPARE_BUILD'}
& "$out/prepare.exe" *> "$out/prepare.txt";if($LASTEXITCODE){throw 'HOST_PREPARE_GATE'}
& $arm @flags --target=aarch64-linux-android26 -static-libstdc++ "$src/backward_shared_lifetime_probe.cpp" -ldl -o "$out/saved_p_backward_probe" *> "$out/arm-build.txt"
if($LASTEXITCODE){throw 'ARM_BUILD'}
# Copy, never rebuild, the frozen DSP/Host packages.
foreach($name in @('libHexaBackwardRepairDsp.so','libHexaBackwardRepairHost.so')){
 [IO.File]::Copy("$original/$name","$out/$name",$false)
 if((Get-FileHash "$out/$name").Hash -ne (Get-FileHash "$original/$name").Hash){throw 'DSP_HOST_NOT_BIT_IDENTICAL'}
}
$sources=@{};foreach($p in $old.source_hashes.PSObject.Properties){$sources[$p.Name]=$p.Value}
foreach($p in @('host_tests/fp32_custom/backward_shared_lifetime.h','host_tests/fp32_custom/backward_shared_lifetime_probe.cpp','host_tests/fp32_custom/backward_shared_lifetime_test.cpp','host_tests/fp32_custom/backward_shared_lifetime_diagnostic_plan.json','scripts/build_backward_shared_lifetime_probe.ps1','scripts/run_backward_shared_lifetime_probe.ps1')){$sources[$p]=(Get-FileHash "$root/$p").Hash.ToLowerInvariant()}
$binaries=@{};foreach($name in @('saved_p_backward_probe','libHexaBackwardRepairDsp.so','libHexaBackwardRepairHost.so')){$binaries[$name]=(Get-FileHash "$out/$name").Hash.ToLowerInvariant()}
if($binaries.saved_p_backward_probe -eq $old.probe_sha256){throw 'ORIGINAL_CANDIDATE_IMPERSONATION'}
$diff=git -C $root diff --no-index -- host_tests/fp32_custom/backward_repair_probe.cpp host_tests/fp32_custom/backward_shared_lifetime_probe.cpp
[IO.File]::WriteAllText("$out/exact-source-diff.txt",($diff -join "`n"),[Text.UTF8Encoding]::new($false))
$headers=@{};foreach($h in @('QnnContext.h','QnnBackend.h','QnnDevice.h','QnnLog.h','QnnInterface.h','QnnSdkBuildId.h')){$headers[$h]=(Get-FileHash "$QairtSdkRoot/include/QNN/$h").Hash.ToLowerInvariant()}
@{source_sha=(git -C $root rev-parse HEAD);source_hashes=$sources;binary_hashes=$binaries;probe_sha256=$binaries.saved_p_backward_probe;
 bundle_leaf=(Split-Path $out -Leaf);candidate='shared-lifetime-boundary-v1';classification='DIAGNOSTIC_ONLY_NOT_FORMAL';
 original_source_sha=$old.source_sha;original_binary_hashes=$old.binary_hashes;gradient_sha256=$old.gradient_sha256;
 qairt_build_id=$ExpectedBuildId;qnn_api='2.37.0';hexagon_sdk='6.6.0.0';hexagon_tools='19.0.07';htp='V81';ndk='26.2.11394342';host_gcc='15.2.0';
 ndk_compiler_sha256=$old.ndk_compiler_sha256;hexagon_compiler_sha256=$old.hexagon_compiler_sha256;host_compiler_sha256=$old.host_compiler_sha256;
 sdk_header_sha256=$headers;flags=$flags;new_DSP_kernel=$false;production_binding=$false;logger_repair=$false;
 lifetime='process backend/device, sequential role contexts';exact_source_diff_sha256=(Get-FileHash "$out/exact-source-diff.txt").Hash.ToLowerInvariant();host_gates='PASS'}|ConvertTo-Json -Depth 14|Out-File "$out/build-identity.json" -Encoding utf8
Write-Host ('shared_lifetime_build=PASS arm_sha256='+$binaries.saved_p_backward_probe+' DSP_HOST_BIT_IDENTICAL=1 DEVICE_EXECUTES=0')
