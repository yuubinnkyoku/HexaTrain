param(
 [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
 [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
 [string]$ReportRoot=''
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/ordered-fp32-hvx-ffn2'
if($ReportRoot){$out=[IO.Path]::GetFullPath($ReportRoot)}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture=Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'
$ln=Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin'
$producer=Join-Path $root 'build/reports/g1-fp32-attention-ln1-frontier/ln1-producers.bin'
$expected=@('388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f','7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f','ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580')
$fixtures=@($fixture,$ln,$producer)
for($i=0;$i -lt 3;$i++){if((Get-FileHash -LiteralPath $fixtures[$i]).Hash.ToLowerInvariant() -ne $expected[$i]){throw 'FROZEN_FIXTURE_MISMATCH'}}
$compiler=(Get-Command g++ -ErrorAction Stop).Source
$tests=@('ordered_ffn2_host_test','closure_kernel_test','segment_kernel_test','custom_matmul_kernel_test','head_split_kernel_test','observer_kernel_test','softmax_kernel_test','attention_kernel_test','broadcast_kernel_test','one_layer_reference_test','upstream_reference_test','one_layer_graph_test','ordered_hvx_graph_test')
foreach($test in $tests){
 $exe=Join-Path $out ($test+'.exe')
 $sources=@((Join-Path $root ('host_tests/fp32_custom/'+$test+'.cpp')))
 if($test -in @('one_layer_graph_test','ordered_hvx_graph_test')){$sources+=Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp';$sources+=Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp'}
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o $exe
 if($LASTEXITCODE){throw "HOST_COMPILE_FAILED: $test"}
 [string[]]$arguments=@()
 if($test -eq 'closure_kernel_test'){$arguments=@($fixture,$ln,$producer)}
 elseif($test -notin @('ordered_ffn2_host_test','custom_matmul_kernel_test','ordered_hvx_graph_test')){$arguments=@($fixture)}
 Write-Host "running=$test argc=$($arguments.Count)"
 & $exe @arguments > (Join-Path $out ($test+'.txt'))
 if($LASTEXITCODE){throw "HOST_TEST_FAILED: $test exit=$LASTEXITCODE argc=$($arguments.Count)"}
 Write-Host "$test=PASS"
}
Write-Host 'ordered_hvx_ffn2_host_suite=PASS (Host arithmetic/shape only; device evidence is separate)'
