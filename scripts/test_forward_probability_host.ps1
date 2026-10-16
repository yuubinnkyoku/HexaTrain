param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[switch]$NewOnly)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/g1-fp32-forward-attention-probability-closure/host'
[IO.Directory]::CreateDirectory($out)|Out-Null
if(!$NewOnly){& (Join-Path $PSScriptRoot 'test_qk_score_scale_host.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId}
foreach($test in @('forward_probability_host_test','probability_graph_test','probability_gate_test','probability_reference_test')){
 $sources=@((Join-Path $root "host_tests/fp32_custom/$test.cpp"))
 if($test -ne 'forward_probability_host_test'){$sources+=Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp';$sources+=Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp'}
 $exe=Join-Path $out "$test.exe"
 & g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o $exe
 if($LASTEXITCODE){throw "PROBABILITY_HOST_COMPILE_FAILED: $test"}
 if($test -eq 'probability_reference_test'){Copy-Item -LiteralPath (Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin') -Destination (Join-Path $out 'ln1-parameters.bin');Push-Location $out}
 try {& $exe (Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin') > (Join-Path $out "$test.txt")} finally {if($test -eq 'probability_reference_test'){Pop-Location}}
 if($LASTEXITCODE){throw "PROBABILITY_HOST_TEST_FAILED: $test exit=$LASTEXITCODE"}
 Write-Host "$test=PASS"
}
