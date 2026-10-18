param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot;$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath("$root/build/reports")+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_OUTPUT_REQUIRED'}
if(Test-Path "$out/identity-private.json"){throw 'EXISTING_EVIDENCE'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture="$root/build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin";$mixed="$root/build/fp32-mixed-qkv/mixed-inputs.bin";$ln="$root/build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin"
if((Get-FileHash $fixture).Hash.ToLowerInvariant() -ne '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f' -or (Get-FileHash $mixed).Hash.ToLowerInvariant() -ne '8ffbe05a7bc95900a07453588926a6368df0b496909d7a9cf314a603e5935eeb' -or (Get-FileHash $ln).Hash.ToLowerInvariant() -ne '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f'){throw 'FROZEN_INPUT_CHANGED'}
Copy-Item -LiteralPath $ln -Destination "$out/ln1-parameters.bin"
$compiler=(Get-Command g++).Source;$src="$root/host_tests/fp32_custom";$common=@('-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',"-I$QairtSdkRoot/include/QNN")
$support=@("$root/app/src/main/cpp/tiny_language_model_cpu.cpp","$root/app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp")
$tests=@{}
foreach($test in @('double_qk_fixture_export','double_qk_contract_test','double_qk_graph_test','double_qk_assessment')){
 & $compiler @common "$src/$test.cpp" @support -o "$out/$test.exe";if($LASTEXITCODE){throw "HOST_COMPILE $test"}
 $tests[$test]=@{executable_sha256=(Get-FileHash "$out/$test.exe").Hash.ToLowerInvariant()}
}
& $compiler @common -DDOUBLE_QK_HOST_CONTRACT_TEST "$src/double_qk_contract_test.cpp" @support -o "$out/double_qk_kernel_test.exe";if($LASTEXITCODE){throw 'KERNEL_HOST_COMPILE'}
Push-Location $out
try{
 foreach($i in 1..3){& ./double_qk_fixture_export.exe $fixture $mixed "double-$i.bin" > "fixture-$i.txt";if($LASTEXITCODE){throw 'HOST_WORD_CONTRACT'}}
 if(@(Get-FileHash double-1.bin,double-2.bin,double-3.bin|Select-Object -ExpandProperty Hash -Unique).Count -ne 1){throw 'HOST_PROCESS_MISMATCH'}
 & ./double_qk_contract_test.exe > prepare.txt;if($LASTEXITCODE){throw 'PREPARATION_CONTRACT'}
 & ./double_qk_kernel_test.exe double-1.bin > kernel.txt;if($LASTEXITCODE){throw 'KERNEL_CONTRACT'}
 & ./double_qk_graph_test.exe > graph.txt;if($LASTEXITCODE){throw 'GRAPH_CONTRACT'}
 & ./double_qk_assessment.exe $fixture double-1.bin --host > host-joint.jsonl;if($LASTEXITCODE){throw 'HOST_S2_C6_JOINT_FAILED'}
 & ./double_qk_assessment.exe $fixture double-1.bin --host-shift > host-shift-joint.jsonl;$shift=$LASTEXITCODE;if($shift -notin 0,4){throw 'SHIFT_ASSESSMENT_ERROR'}
 . "$PSScriptRoot/mixed_qkv_monitoring_policy.ps1";Test-MixedQkvMonitoringPolicy
 $tests.host_joint_exit=0;$tests.host_shift_joint_exit=$shift
}finally{Pop-Location}
$tests.double_qk_kernel_test=@{executable_sha256=(Get-FileHash "$out/double_qk_kernel_test.exe").Hash.ToLowerInvariant()}
$sources=@{};foreach($p in @(Get-ChildItem "$src/double_qk*" -File)){$sources['host_tests/fp32_custom/'+$p.Name]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
$logs=@{};foreach($p in @(Get-ChildItem $out -File|Where-Object {$_.Extension -in '.txt','.jsonl'})){$logs[$p.Name]=(Get-FileHash $p.FullName).Hash.ToLowerInvariant()}
@{source_sha=(git -C $root rev-parse HEAD);source_hashes=$sources;compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();tests=$tests;log_hashes=$logs;fixture_sha256=(Get-FileHash "$out/double-1.bin").Hash.ToLowerInvariant();device_executes=0}|ConvertTo-Json -Depth 8|Out-File "$out/identity-private.json" -Encoding utf8
Write-Host 'double_qk_host_suite=PASS host_joint=PASS device_executes=0'
