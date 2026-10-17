param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot;$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath("$root/build/reports")+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_OUTPUT_REQUIRED'}
if(Test-Path "$out/identity-private.json"){throw 'EXISTING_EVIDENCE'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture="$root/build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin";$ln="$root/build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin";$s2="$root/build/fp32-selective-ln1/selective-ln1-inputs.bin"
$expected=@('388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f','7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f','492d82c6e6b28080c078f2fb3d1ab84abb4946dc2224af62702fa3a2773d6c02')
$inputs=@($fixture,$ln,$s2);for($i=0;$i -lt 3;$i++){if((Get-FileHash $inputs[$i]).Hash.ToLowerInvariant() -ne $expected[$i]){throw 'FROZEN_INPUT_CHANGED'}}
Copy-Item -LiteralPath $ln -Destination "$out/ln1-parameters.bin"
$compiler=(Get-Command g++).Source;$src="$root/host_tests/fp32_custom";$tests=@{}
foreach($test in @('mixed_qkv_fixture_export','mixed_qkv_prepare_test','mixed_qkv_graph_test')){
 $sources=@("$src/$test.cpp");if($test -ne 'mixed_qkv_prepare_test'){$sources+=@("$root/app/src/main/cpp/tiny_language_model_cpu.cpp","$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp")}
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" @sources -o "$out/$test.exe";if($LASTEXITCODE){throw 'HOST_COMPILE'}
 Push-Location $out
 try{if($test -eq 'mixed_qkv_fixture_export'){foreach($i in 1..3){& "./$test.exe" $fixture $s2 "mixed-$i.bin" > "$test-$i.txt";if($LASTEXITCODE){throw 'HOST_ARITHMETIC'}};if(@(Get-FileHash mixed-1.bin,mixed-2.bin,mixed-3.bin|Select-Object -ExpandProperty Hash -Unique).Count -ne 1){throw 'INDEPENDENT_HOST_PROCESS_MISMATCH'}}else{& "./$test.exe" > "$test.txt";if($LASTEXITCODE){throw 'HOST_CONTRACT'}}}finally{Pop-Location}
 $tests[$test]=@{exit_code=0;exe_sha256=(Get-FileHash "$out/$test.exe").Hash.ToLowerInvariant()}
}
. "$PSScriptRoot/mixed_qkv_monitoring_policy.ps1";Test-MixedQkvMonitoringPolicy
@{source_sha=(git -C $root rev-parse HEAD);compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();inputs=$expected;tests=$tests;fixture_sha256=(Get-FileHash "$out/mixed-1.bin").Hash.ToLowerInvariant();device_executes=0}|ConvertTo-Json -Depth 8|Out-File "$out/identity-private.json" -Encoding utf8
Write-Host 'mixed_host_suite=PASS numerical_processes=3 graphs=32 preparation_only=1'
