param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot;$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath("$root/build/reports")+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_REPORT_REQUIRED'}
if(Test-Path $out){throw 'EXCLUSIVE_HOST_REPORT_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$plan=Get-Content "$root/host_tests/fp32_custom/saved_p_backward_device_plan.json" -Raw|ConvertFrom-Json
$compiler=(Get-Command g++).Source
if((Get-FileHash $compiler).Hash.ToLowerInvariant() -ne $plan.toolchain.host_compiler_sha256){throw 'HOST_COMPILER_CHANGED'}
$common=@('-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',"-I$QairtSdkRoot/include/QNN")
$support=@("$root/app/src/main/cpp/tiny_language_model_cpu.cpp","$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp")
foreach($test in @('saved_p_backward_reference_test','saved_p_oracle_independent_test','saved_p_backward_graph_test','saved_p_backward_fixture_export')) {
 $extra=if($test -in @('saved_p_backward_graph_test','saved_p_backward_fixture_export')){$support}else{@()}
 & $compiler @common "$root/host_tests/fp32_custom/$test.cpp" @extra -o "$out/$test.exe" *> "$out/$test-build.txt"
 if($LASTEXITCODE){throw "HOST_BUILD_FAILED $test"}
 if($test -ne 'saved_p_backward_fixture_export') {
  & "$out/$test.exe" *> "$out/$test.txt";if($LASTEXITCODE){throw "HOST_TEST_FAILED $test"}
 }
}
$original="$root/build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin"
if((Get-FileHash $original).Hash.ToLowerInvariant() -ne $plan.protected_forward_identities.original_fixture){throw 'ORIGINAL_FIXTURE_CHANGED'}
foreach($i in 1..3) {
 & "$out/saved_p_backward_fixture_export.exe" $original "$out/gradients-$i.bin" *> "$out/freeze-$i.txt"
 if($LASTEXITCODE -or (Get-FileHash "$out/gradients-$i.bin").Hash.ToLowerInvariant() -ne $plan.gradient_sha256){throw 'FROZEN_GRADIENT_WORD_MISMATCH'}
}
. "$PSScriptRoot/mixed_qkv_monitoring_policy.ps1";Test-MixedQkvMonitoringPolicy
@{source_sha=(git -C $root rev-parse HEAD);compiler_sha256=$plan.toolchain.host_compiler_sha256;
 plan_sha256=(Get-FileHash "$root/host_tests/fp32_custom/saved_p_backward_device_plan.json").Hash.ToLowerInvariant();
 gradient_sha256=$plan.gradient_sha256;reference_selftest='PASS';independent_oracle='PASS';graph_shape_ownership='PASS';
 gradient_processes=3;device_calls=0}|ConvertTo-Json|Out-File "$out/identity-private.json" -Encoding utf8
Write-Host 'saved_p_backward_host=PASS reference=PASS independent_oracle=PASS graph=24 gradient_processes=3 device_calls=0'
