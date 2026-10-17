param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath((Join-Path $root 'build/reports'))+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'EVIDENCE_MUST_BE_PRIVATE'}
if(Test-Path (Join-Path $out 'matrix-1.jsonl')){throw 'EVIDENCE_ALREADY_EXISTS'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture=Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'
$ln=Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin'
if((Get-FileHash $fixture).Hash.ToLowerInvariant() -ne '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f' -or (Get-FileHash $ln).Hash.ToLowerInvariant() -ne '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f'){throw 'FROZEN_INPUT_MISMATCH'}
Copy-Item -LiteralPath $ln -Destination (Join-Path $out 'ln1-parameters.bin')
$compiler=(Get-Command g++ -ErrorAction Stop).Source
$tests=@('upstream_precision_test','upstream_precision_consistency_test')
$ids=@{}
foreach($test in $tests){
 $src=Join-Path $root "host_tests/fp32_custom/$test.cpp"
 $exe=Join-Path $out "$test.exe"
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" $src (Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp') (Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp') -o $exe
 if($LASTEXITCODE){throw "COMPILE_FAILED $test"}
 $logs=@{}
 Push-Location $out
 try {
  if($test -eq 'upstream_precision_test'){
   for($i=1;$i -le 3;$i++){& $exe $fixture > "matrix-$i.jsonl";if($LASTEXITCODE){throw 'MATRIX_FAILED'};$logs["matrix-$i.jsonl"]=(Get-FileHash "matrix-$i.jsonl").Hash.ToLowerInvariant();Write-Host "precision_matrix_process=$i completed=1"}
   & $exe $fixture --fp32-shift > 'matrix-fp32-shift.jsonl';if($LASTEXITCODE){throw 'SHIFT_MODEL_FAILED'};$logs['matrix-fp32-shift.jsonl']=(Get-FileHash 'matrix-fp32-shift.jsonl').Hash.ToLowerInvariant()
  }else{& $exe $fixture > 'consistency.jsonl';if($LASTEXITCODE){throw 'CONSISTENCY_FAILED'};$logs['consistency.jsonl']=(Get-FileHash 'consistency.jsonl').Hash.ToLowerInvariant()}
 }finally{Pop-Location}
 $ids[$test]=@{source_sha256=(Get-FileHash $src).Hash.ToLowerInvariant();executable_sha256=(Get-FileHash $exe).Hash.ToLowerInvariant();logs=$logs;exit_code=0}
}
$hashes=@($ids.upstream_precision_test.logs['matrix-1.jsonl'],$ids.upstream_precision_test.logs['matrix-2.jsonl'],$ids.upstream_precision_test.logs['matrix-3.jsonl'])
if(@($hashes|Select-Object -Unique).Count -ne 1){throw 'INDEPENDENT_HOST_PROCESS_MISMATCH'}
@{source_commit=(git -C $root rev-parse HEAD);compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();flags='-std=c++17 -O2 -ffp-contract=off -fno-fast-math';plan_sha256=(Get-FileHash (Join-Path $root 'host_tests/fp32_custom/upstream_precision_plan.json')).Hash.ToLowerInvariant();fixture_sha256=(Get-FileHash $fixture).Hash.ToLowerInvariant();synthetic_inputs_sha256=(Get-FileHash (Join-Path $out 'synthetic-inputs.bin')).Hash.ToLowerInvariant();tests=$ids;device_executes=0}|ConvertTo-Json -Depth 12|Out-File (Join-Path $out 'host-identity-private.json') -Encoding utf8
Write-Host 'upstream_precision_host_analysis=PASS independent_processes=3 device_certification=NOT_RUN'
