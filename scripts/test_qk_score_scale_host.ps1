param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/g1-fp32-qk-score-scale-frontier/host'
[IO.Directory]::CreateDirectory($out)|Out-Null
& (Join-Path $PSScriptRoot 'test_ordered_hvx_qkv_host.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
foreach($test in @('qk_scale_host_test','qk_scale_graph_test')){
 $sources=@((Join-Path $root "host_tests/fp32_custom/$test.cpp"))
 if($test -eq 'qk_scale_graph_test'){$sources+=Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp';$sources+=Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp'}
 $exe=Join-Path $out "$test.exe"
 & g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o $exe
 if($LASTEXITCODE){throw "HOST_COMPILE_FAILED: $test"}
 & $exe (Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin') > (Join-Path $out "$test.txt")
 if($LASTEXITCODE){throw "HOST_TEST_FAILED: $test"}
 Write-Host "$test=PASS"
}
