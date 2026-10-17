param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[switch]$Legacy,[string]$ReportRoot='')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'build/reports/g1-forward-p-numerical-attribution-safe-resume'
if($ReportRoot){$out=[IO.Path]::GetFullPath($ReportRoot)}
if(Test-Path -LiteralPath (Join-Path $out 'analysis-identity-private.json')){throw 'ATTRIBUTION_EVIDENCE_ALREADY_EXISTS'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture=Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'
$ln=Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin'
$producer=Join-Path $root 'build/reports/g1-fp32-attention-ln1-frontier/ln1-producers.bin'
$expected=@('388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f','7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f','ee9e3185b9a5857d874b5791a1e4b34c153401367dc208a1b74fa3bafa1a9580')
$fixtures=@($fixture,$ln,$producer)
for($i=0;$i -lt 3;$i++){if((Get-FileHash -LiteralPath $fixtures[$i]).Hash.ToLowerInvariant() -ne $expected[$i]){throw 'FROZEN_FIXTURE_MISMATCH'}}
Copy-Item -LiteralPath $ln -Destination (Join-Path $out 'ln1-parameters.bin')
if($Legacy){& (Join-Path $PSScriptRoot 'test_ordered_hvx_ffn2_host.ps1') -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportRoot (Join-Path $out 'legacy')}
$tests=@('forward_mask_host_test','forward_probability_host_test','probability_graph_test','probability_gate_test','probability_reference_test','probability_attribution_test')
if($Legacy){$tests=@('ordered_qkv_host_test','ordered_qkv_graph_test','qk_scale_host_test','qk_scale_graph_test')+$tests}
$compiler=(Get-Command g++).Source
$identities=@{}
foreach($test in $tests){
 $sources=@((Join-Path $root "host_tests/fp32_custom/$test.cpp"))
 if($test -match 'graph|probability_reference|probability_attribution|probability_gate'){$sources+=Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp';$sources+=Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp'}
 $exe=Join-Path $out "$test.exe"
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o $exe
 if($LASTEXITCODE){throw "ATTRIBUTION_COMPILE_FAILED: $test"}
 $log=Join-Path $out $(if($test -eq 'probability_attribution_test'){'attribution.jsonl'}else{"$test.txt"})
 Push-Location $out
 try {& $exe $fixture > $log;$code=$LASTEXITCODE} finally {Pop-Location}
 $identities[$test]=@{exit_code=$code;source_sha256=(Get-FileHash $sources[0]).Hash.ToLowerInvariant();executable_sha256=(Get-FileHash $exe).Hash.ToLowerInvariant();log_sha256=(Get-FileHash $log).Hash.ToLowerInvariant()}
 if($code -ne $(if($test -eq 'probability_reference_test'){4}else{0})){throw "ATTRIBUTION_TEST_UNEXPECTED_EXIT: $test exit=$code"}
 Write-Host "$test exit=$code $(if($code){'ORIGINAL_CONTRACT_FAIL_REPRODUCED'}else{'PASS'})"
}
@{source_commit=(& git -C $root rev-parse HEAD);compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();flags='-std=c++17 -O2 -ffp-contract=off -fno-fast-math';tests=$identities}|ConvertTo-Json -Depth 8|Out-File (Join-Path $out 'analysis-identity-private.json') -Encoding utf8
Write-Host 'attribution_analysis_suite=PASS old_joint_contract=FAIL expected_exit=4 gates_unchanged=1'
