# Allow-list export for the initial blocked snapshot or explicit final cohort.
# Only -FinalCohort can export device numerical/performance evidence.
param([string]$OutputPath='', [switch]$FinalCohort, [switch]$SelfTest)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$private=Join-Path $root 'build/reports/ordered-fp32-hvx-ffn2'
if($FinalCohort){
 if(!$OutputPath){$OutputPath=Join-Path $private 'final-public-summary.json'}
 $arguments=@((Join-Path $root 'scripts/export_public_ordered_hvx_ffn2.py'),'--root',$root,'--output',$OutputPath)
 if($SelfTest){$arguments+='--self-test'}
 & python @arguments
 if($LASTEXITCODE){throw 'FINAL_COHORT_EXPORT_FAILED'}
 return
}
$build=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw|ConvertFrom-Json
$audit=Get-Content (Join-Path $private 'codegen-audit.json') -Raw|ConvertFrom-Json
$stop=Get-Content (Join-Path $private 'safety-stop-private.json') -Raw|ConvertFrom-Json
if($build.qairt_build_id -ne '2.48.40.260702151143' -or $build.architecture -ne 'v81' -or $build.hvx_vector_bytes -ne 128 -or $build.hvx_fp_mode -ne 'strict-ieee' -or $build.fp_contract -ne 'off'){throw 'BUILD_CONTRACT_MISMATCH'}
if($build.source_sha -notmatch '^[a-f0-9]{40}$'){throw 'SOURCE_COMMIT_MISSING'}
foreach($p in $build.source_hashes.psobject.Properties){if((Get-FileHash (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'SOURCE_HASH_MISMATCH'}}
$binary=[ordered]@{}
foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe')){
 $hash=(Get-FileHash (Join-Path $root "build/fp32-custom/$name")).Hash.ToLowerInvariant()
 if($hash -ne $build.binary_hashes.$name){throw 'BINARY_HASH_MISMATCH'}
 $binary[$name]=$hash
}
if($audit.package_sha256 -ne $binary['libHexaFp32Dsp.so'] -or $audit.codegen -ne 'HVX_FFN2_CODEGEN_VERIFIED' -or $audit.dsp_execution -ne 'UNRESOLVED' -or $audit.fma_sites -ne 0 -or $audit.calls_in_target -ne 0 -or $audit.stack_spill_sites -ne 0){throw 'CODEGEN_AUDIT_MISMATCH'}
$target=Join-Path $private 'target-disassembly-private.txt'
if((Get-FileHash $target).Hash.ToLowerInvariant() -ne $audit.target_disassembly_sha256){throw 'DISASSEMBLY_HASH_MISMATCH'}
if($stop.blocking_gate -ne 'HOST_STORAGE_HEADROOM_LOW' -or $stop.host_free_bytes -ge 20GB -or $stop.required_host_free_bytes -ne 20GB -or $stop.candidate_device_executes -ne 0 -or $stop.protected_hashes_verified -ne 5){throw 'SAFETY_STOP_MISMATCH'}
if(@(Get-ChildItem -LiteralPath $private -Directory -Filter 'customfp32-*').Count){throw 'DEVICE_PROCESS_EVIDENCE_REQUIRES_DIFFERENT_EXPORTER'}
$suite=Get-Content (Join-Path $private 'host-suite-complete.txt') -Raw
$tests=@('ordered_ffn2_host_test','closure_kernel_test','segment_kernel_test','custom_matmul_kernel_test','head_split_kernel_test','observer_kernel_test','softmax_kernel_test','attention_kernel_test','broadcast_kernel_test','one_layer_reference_test','upstream_reference_test','one_layer_graph_test')
foreach($test in $tests){if($suite -notmatch ('(?m)^'+[regex]::Escape($test)+'=PASS\r?$')){throw 'HOST_SUITE_INCOMPLETE'}}
$control=Get-Content (Join-Path $private 'ordered_ffn2_host_test.txt') -Raw
if($control -notmatch '^ordered_ffn2_host=PASS controls=12 calls=24 maxAbs=([\deE.+-]+) scalar_words=1 hvx_host_execution_rejected=1 negative_separate=1') {throw 'HOST_CONTROL_MISSING'}
$maxAbs=[double]$Matches[1]
if([double]::IsNaN($maxAbs) -or [double]::IsInfinity($maxAbs) -or $maxAbs -gt .002){throw 'HOST_CONTROL_ACCURACY'}
$fast=Get-Content (Join-Path $private 'verify-fast.txt') -Raw
$qnn=Get-Content (Join-Path $private 'verify-qnn.txt') -Raw
if($fast -notmatch 'PASS=1 FAIL=0 SKIP=0' -or $qnn -notmatch 'PASS=4 FAIL=0 SKIP=0'){throw 'VERIFICATION_INCOMPLETE'}
$baseline=Get-Content (Join-Path $private 'baseline-private.json') -Raw|ConvertFrom-Json
foreach($p in $baseline.protected.protected_sha256.psobject.Properties){if((Get-FileHash (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'PROTECTED_HASH_MISMATCH'}}
$fixtures=[ordered]@{}
foreach($p in $baseline.fixture_hashes.psobject.Properties){if((Get-FileHash (Join-Path $root $p.Name)).Hash.ToLowerInvariant() -ne $p.Value){throw 'FROZEN_FIXTURE_MISMATCH'};$fixtures[(Split-Path $p.Name -Leaf)]=$p.Value}
$instruction=[ordered]@{}
foreach($key in @('multiply_sites','add_sites','fp32_conversion_sites','normalization_sites','broadcast_sites','unaligned_memory_sites','fma_sites','calls_in_target','stack_spill_sites')){$instruction[$key]=[int]$audit.$key}
$sdkRecord=Get-Content (Join-Path $private 'sdk-hashes.json') -Raw|ConvertFrom-Json
$sdkHashes=[ordered]@{}
foreach($key in @('sdk_manifest','hvx_protos','hexagon_types','v81_hvx_prm','compiler_guide')){if($sdkRecord.$key -notmatch '^[a-f0-9]{64}$'){throw 'SDK_HASH_MISSING'};$sdkHashes[$key]=$sdkRecord.$key}
$result=[ordered]@{
 schema='ORDERED_HVX_FFN2_BLOCKED_V1';branch='investigation/ordered-fp32-hvx-ffn2';source_commit=$build.source_sha;device_tested_source_commit=$null
 geometry=@{m=32;k=128;n=64;transpose_a=$false;transpose_b=$false};qairt_build_id=$build.qairt_build_id;qnn_api='2.37.0';architecture='V81'
 hexagon_sdk='6.6.0.0';hexagon_tools='19.0.07';ndk='26.2.11394342';fp_contract='off';hvx_fp_mode='strict-ieee';vector_bytes=128;fp32_lanes=32
 binary_sha256=$binary;hexagon_compiler_sha256=$build.hexagon_compiler_sha256;ndk_compiler_sha256=$build.ndk_compiler_sha256;sdk_sha256=$sdkHashes;fixture_sha256=$fixtures
 target_disassembly_sha256=$audit.target_disassembly_sha256;static_instruction_sites=$instruction;dsp_execution='UNRESOLVED'
 host=@{tests_passed=12;controls=12;control_calls=24;scalar_control_max_abs=$maxAbs;scalar_ordered_words='PASS';hvx_host_emulation=$false;shape_checks=3920;fast='PASS';qnn='PASS'}
 device=@{candidate_executes=0;independent_processes=0;word_mismatches=$null;double_max_abs=$null;numerical_digest=$null;native_segment='UNRESOLVED';observer_free='UNRESOLVED';kernel_speedup=$null;segment_speedup=$null;smoke='BLOCKED';muon_w8='BLOCKED'}
 explicit_kernel_scratch_bytes=0;prepack_bytes=0;observed_receipt_bytes=57408;protected_hashes_verified=5
 classification=@('HVX_FFN2_CODEGEN_VERIFIED','UNRESOLVED');blocking_gate='HOST_STORAGE_HEADROOM_LOW';training_updates=0;training='NO-GO'
}
if(!$OutputPath){$OutputPath=Join-Path $private 'public-summary.json'}
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),($result|ConvertTo-Json -Depth 9)+"`n",[Text.UTF8Encoding]::new($false))
Write-Host 'ordered_hvx_ffn2_public_export=PASS (Host/codegen only; device unresolved)'
