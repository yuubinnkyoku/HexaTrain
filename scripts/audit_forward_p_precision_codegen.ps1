param([Parameter(Mandatory=$true)][string]$QairtSdkRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuildId,[Parameter(Mandatory=$true)][string]$HexagonSdkRoot,[Parameter(Mandatory=$true)][string]$ReportRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath($ReportRoot)
if(!$out.StartsWith([IO.Path]::GetFullPath((Join-Path $root 'build/reports'))+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'PRIVATE_OUTPUT_REQUIRED'}
[IO.Directory]::CreateDirectory($out)|Out-Null
$manifest=Get-Content (Join-Path $HexagonSdkRoot 'hexagon_sdk.json') -Raw|ConvertFrom-Json
$tools=@($manifest.root.tools.info|Where-Object {$_.name -eq 'Hexagon Tools'})
if($manifest.root.id.version -ne '6.6.0.0' -or $tools.Count -ne 1 -or $tools[0].build -ne '19.0.07'){throw 'PINNED_TOOLS_MISMATCH'}
$bin=Join-Path $HexagonSdkRoot 'tools/HEXAGON_Tools/19.0.07/Tools/bin'
$compiler=Join-Path $bin 'hexagon-clang++.exe'
$build=Get-Content (Join-Path $root 'build/fp32-custom/build-identity.json') -Raw|ConvertFrom-Json
if((Get-FileHash $compiler).Hash.ToLowerInvariant() -ne $build.hexagon_compiler_sha256){throw 'COMPILER_ANCHOR_MISMATCH'}
$source=Join-Path $root 'host_tests/fp32_custom/upstream_precision_codegen.cpp'
$flags=@('-std=c++17','-mv81','-O2','-G0','-ffp-contract=off','-fno-fast-math','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops')
& $compiler @flags -S $source -o "$out/precision-v81.s"
if($LASTEXITCODE){throw 'ASM_FAILED'}
& $compiler @flags -S -emit-llvm $source -o "$out/precision-v81.ll"
if($LASTEXITCODE){throw 'IR_FAILED'}
& $compiler @flags -c $source -o "$out/precision-v81.o"
if($LASTEXITCODE){throw 'OBJECT_FAILED'}
& (Join-Path $bin 'hexagon-llvm-objdump.exe') -d "$out/precision-v81.o" > "$out/precision-disassembly-private.txt"
if($LASTEXITCODE){throw 'DISASSEMBLY_FAILED'}
& (Join-Path $bin 'hexagon-nm.exe') --undefined-only "$out/precision-v81.o" > "$out/precision-undefined-private.txt"
if($LASTEXITCODE){throw 'NM_FAILED'}
$asm=Get-Content "$out/precision-v81.s" -Raw
$ir=Get-Content "$out/precision-v81.ll" -Raw
$undef=Get-Content "$out/precision-undefined-private.txt" -Raw
if($asm -notmatch 'dfadd\(' -or $asm -notmatch 'dfmpyfix\(' -or $asm -notmatch 'dfmpyll\(' -or $asm -notmatch 'dfmpyhh\(' -or $asm -notmatch 'convert_sf2df' -or $undef -notmatch '__hexagon_divdf3' -or $undef -notmatch '\bsqrt\b' -or $ir -match 'llvm\.fma|fadd fast|fmul fast'){throw 'UNEXPECTED_FP64_LOWERING'}
$qhpi=Get-Content (Join-Path $QairtSdkRoot 'include/QNN/HTP/core/qhpi.h') -Raw
$qnn=Get-Content (Join-Path $QairtSdkRoot 'include/QNN/QnnTypes.h') -Raw
if($qhpi -match 'QHPI_FLOAT64' -or $qhpi -notmatch 'QHPI_FLOAT32 = 4' -or $qnn -notmatch 'QNN_DATATYPE_FLOAT_64'){throw 'DTYPE_AUDIT_CHANGED'}
@{source_commit=(git -C $root rev-parse HEAD);source_sha256=(Get-FileHash $source).Hash.ToLowerInvariant();compiler_sha256=(Get-FileHash $compiler).Hash.ToLowerInvariant();sdk_manifest_sha256=(Get-FileHash (Join-Path $HexagonSdkRoot 'hexagon_sdk.json')).Hash.ToLowerInvariant();flags=$flags;assembly_sha256=(Get-FileHash "$out/precision-v81.s").Hash.ToLowerInvariant();object_sha256=(Get-FileHash "$out/precision-v81.o").Hash.ToLowerInvariant();llvm_ir_sha256=(Get-FileHash "$out/precision-v81.ll").Hash.ToLowerInvariant();scalar_dfadd_emitted=$true;scalar_partial_product_df_multiply_emitted=$true;division_helper='__hexagon_divdf3';ln_sqrt_external=$true;qhpi_float64_enum=$false;generic_qnn_float64_enum=$true;runtime_linkage_verified=$false;runtime_rounding_verified=$false;hardware_latency_measured=$false;device_executes=0;status='COMPILE_ONLY_FEASIBILITY'}|ConvertTo-Json -Depth 8|Out-File "$out/codegen-identity-private.json" -Encoding utf8
Write-Host 'precision_codegen=PASS compile_only=1 device_executes=0 runtime_linkage_unverified=1 latency_unmeasured=1'
