param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath((Join-Path $root 'build/reports'))+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_OUTPUT_REQUIRED'}
if(Test-Path (Join-Path $out 'identity-private.json')){throw 'EXISTING_CODEGEN_EVIDENCE'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$manifest=Get-Content (Join-Path $HexagonSdkRoot 'hexagon_sdk.json') -Raw|ConvertFrom-Json
$tool=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tool.Count -ne 1 -or $tool[0].build -ne '19.0.07'){throw 'PINNED_SDK_MISMATCH'}
$bin=Join-Path $HexagonSdkRoot 'tools/HEXAGON_Tools/19.0.07/Tools/bin'
$cc=Join-Path $bin 'hexagon-clang++.exe'
$build=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw|ConvertFrom-Json
if((Get-FileHash $cc).Hash.ToLowerInvariant() -ne $build.hexagon_compiler_sha256){throw 'COMPILER_MISMATCH'}
$src=Join-Path $root 'host_tests/fp32_custom/selective_ln1_codegen.cpp'
$flags=@('-std=c++17','-mv81','-O2','-G0','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-fstack-usage')
& $cc @flags -S $src -o "$out/selective.s";if($LASTEXITCODE){throw 'ASM_FAILED'}
& $cc @flags -S -emit-llvm $src -o "$out/selective.ll";if($LASTEXITCODE){throw 'IR_FAILED'}
& $cc @flags -c $src -o "$out/selective.o";if($LASTEXITCODE){throw 'OBJECT_FAILED'}
& (Join-Path $bin 'hexagon-llvm-objdump.exe') -d "$out/selective.o" > "$out/disassembly-private.txt";if($LASTEXITCODE){throw 'DISASSEMBLY_FAILED'}
& (Join-Path $bin 'hexagon-nm.exe') --undefined-only "$out/selective.o" > "$out/undefined-private.txt";if($LASTEXITCODE){throw 'NM_FAILED'}
$asm=Get-Content "$out/selective.s" -Raw
$ir=Get-Content "$out/selective.ll" -Raw
if($asm -notmatch 'dfadd\(' -or $asm -notmatch 'dfmpyfix\(' -or $asm -notmatch 'convert_sf2df' -or $asm -notmatch 'convert_df2sf' -or $ir -match 'llvm\.fma|fadd fast|fmul fast'){throw 'CODEGEN_CONTRACT_FAILED'}
$hashes=@{};foreach($file in @('selective.s','selective.ll','selective.o','disassembly-private.txt','undefined-private.txt')){$hashes[$file]=(Get-FileHash "$out/$file").Hash.ToLowerInvariant()}
@{source_commit=(git -C $root rev-parse HEAD);source_sha256=(Get-FileHash $src).Hash.ToLowerInvariant();arithmetic_sha256=(Get-FileHash (Join-Path $root 'host_tests/fp32_custom/selective_ln1_arithmetic.h')).Hash.ToLowerInvariant();compiler_sha256=(Get-FileHash $cc).Hash.ToLowerInvariant();sdk_manifest_sha256=(Get-FileHash (Join-Path $HexagonSdkRoot 'hexagon_sdk.json')).Hash.ToLowerInvariant();flags=$flags;outputs=$hashes;runtime_linkage_verified=$false;runtime_rounding_verified=$false;device_executes=0;status='COMPILE_ONLY'}|ConvertTo-Json -Depth 8|Out-File "$out/identity-private.json" -Encoding utf8
Write-Host 'selective_codegen=PASS runtime=UNMEASURED'
