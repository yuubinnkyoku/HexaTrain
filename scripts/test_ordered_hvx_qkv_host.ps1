param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/ordered-fp32-hvx-qkv'
[IO.Directory]::CreateDirectory($out)|Out-Null
& (Join-Path $PSScriptRoot 'test_ordered_hvx_ffn2_host.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportRoot (Join-Path $out 'ffn2-host-regression')
$compiler=(Get-Command g++ -ErrorAction Stop).Source
foreach($test in @('ordered_qkv_host_test','ordered_qkv_graph_test')){
 $sources=@((Join-Path $root ('host_tests/fp32_custom/'+$test+'.cpp')))
 if($test -eq 'ordered_qkv_graph_test'){$sources+=Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp';$sources+=Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp'}
 $exe=Join-Path $out ($test+'.exe')
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o $exe
 if($LASTEXITCODE){throw "QKV_HOST_COMPILE_FAILED: $test"}
 & $exe > (Join-Path $out ($test+'.txt'))
 if($LASTEXITCODE){throw "QKV_HOST_TEST_FAILED: $test exit=$LASTEXITCODE"}
 Write-Host "$test=PASS"
}
Write-Host 'ordered_hvx_qkv_host_suite=PASS tests=15 (Host only)'
