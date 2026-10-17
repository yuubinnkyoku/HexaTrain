param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$ReportRoot,[ValidateSet('Development','Confirmation')][string]$Phase='Development',[int]$Selected=-1)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath((Join-Path $root 'build/reports'))+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_OUTPUT_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$fixture=Join-Path $root 'build/reports/g1-one-layer-fp32-backward/one-layer-inputs.bin'
$ln=Join-Path $root 'build/reports/g1-fp32-broadcast-upstream-frontier/ln1-parameters.bin'
if((Get-FileHash $fixture).Hash.ToLowerInvariant() -ne '388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f' -or (Get-FileHash $ln).Hash.ToLowerInvariant() -ne '7102ec13544df457922942e3282b15a8612da91c51156692000fc0dd0a21f97f'){throw 'FIXTURE_MISMATCH'}
$exe=Join-Path $out 'selective_ln1_test.exe'
$compiler=(Get-Command g++).Source
$plan=Join-Path $root 'host_tests/fp32_custom/selective_ln1_plan.json'
if($Phase -eq 'Development'){
 if(Test-Path (Join-Path $out 'development-1.jsonl')){throw 'EXISTING_EVIDENCE'}
 Copy-Item -LiteralPath $ln -Destination (Join-Path $out 'ln1-parameters.bin')
 & $compiler -std=c++17 -O2 -ffp-contract=off -fno-fast-math "-I$QairtSdkRoot/include/QNN" (Join-Path $root 'host_tests/fp32_custom/selective_ln1_test.cpp') (Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp') (Join-Path $root 'app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp') -o $exe
 if($LASTEXITCODE){throw 'COMPILE_FAILED'}
 $files=@{};foreach($p in @('host_tests/fp32_custom/selective_ln1_test.cpp','host_tests/fp32_custom/selective_ln1_arithmetic.h','host_tests/fp32_custom/selective_ln1_plan.json')){$files[$p]=(Get-FileHash (Join-Path $root $p)).Hash.ToLowerInvariant()}
 @{source_commit=(git -C $root rev-parse HEAD);compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();exe_sha256=(Get-FileHash $exe).Hash.ToLowerInvariant();source_hashes=$files;fixture_sha256=(Get-FileHash $fixture).Hash.ToLowerInvariant();ln_sha256=(Get-FileHash $ln).Hash.ToLowerInvariant();plan_sha256=(Get-FileHash $plan).Hash.ToLowerInvariant();flags='-std=c++17 -O2 -ffp-contract=off -fno-fast-math';device_executes=0}|ConvertTo-Json -Depth 8|Out-File (Join-Path $out 'identity-private.json') -Encoding utf8
 $mode='development';$extra=@()
}else{
 if($Selected -lt 0 -or $Selected -gt 7){throw 'SELECTION_REQUIRED'}
 $identity=Get-Content (Join-Path $out 'identity-private.json') -Raw|ConvertFrom-Json
 $selection=Get-Content (Join-Path $out 'selection-private.json') -Raw|ConvertFrom-Json
 if($selection.selected -ne "S$Selected" -or (Get-FileHash $exe).Hash.ToLowerInvariant() -ne $identity.exe_sha256 -or $selection.confirmation_already_evaluated -ne $false){throw 'SELECTION_OR_BINARY_MISMATCH'}
 if(Test-Path (Join-Path $out 'confirmation-1.jsonl')){throw 'CONFIRMATION_ALREADY_EXISTS'}
 $mode='confirm';$extra=@("$Selected")
}
$prefix=if($Phase -eq 'Development'){'development'}else{'confirmation'}
$hashes=@()
Push-Location $out
try{
 for($i=1;$i -le 3;$i++){& $exe $fixture $mode @extra > "$prefix-$i.jsonl";if($LASTEXITCODE){throw 'NUMERICAL_ANALYSIS_FAILED'};$hashes+=(Get-FileHash "$prefix-$i.jsonl").Hash.ToLowerInvariant();Write-Host "$prefix process=$i PASS"}
 & $exe $fixture $mode @extra --fp32-shift > "$prefix-fp32-shift.jsonl";if($LASTEXITCODE){throw 'SHIFT_ANALYSIS_FAILED'}
 if(@($hashes|Select-Object -Unique).Count -ne 1){throw 'PROCESS_DIGEST_MISMATCH'}
 @{source_commit=(git -C $root rev-parse HEAD);exe_sha256=(Get-FileHash $exe).Hash.ToLowerInvariant();numerical_digests=$hashes;shift_digest=(Get-FileHash "$prefix-fp32-shift.jsonl").Hash.ToLowerInvariant();device_executes=0;status='HOST_ANALYSIS_COMPLETED'}|ConvertTo-Json|Out-File "$prefix-processes-private.json" -Encoding utf8
}finally{Pop-Location}
