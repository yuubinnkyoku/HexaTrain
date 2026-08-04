# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only CPU intra-block readability diagnosis for the L19 quality-gate
# investigation (deep readout degradation decomposition). The runner is built
# from the checked-in CPU reference implementation; no Android, QAIRT, ADB, or
# device state is used. Output is private evidence under build/.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot,
    [string]$TapRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $root 'build\reports\qnn-intra-block-readability'
}
if ([string]::IsNullOrWhiteSpace($TapRoot)) {
    $TapRoot = Join-Path $root 'build\reports\qnn-intra-block-readability\private-taps'
}

function Assert-BuildOutput([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $buildRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "ReportRoot/TapRoot must remain under build/: $Path"
    }
    return $full
}

$ReportRoot = Assert-BuildOutput $ReportRoot
$TapRoot = Assert-BuildOutput $TapRoot
[IO.Directory]::CreateDirectory($ReportRoot) | Out-Null
[IO.Directory]::CreateDirectory($TapRoot) | Out-Null

$exe = Join-Path $ReportRoot 'intra-block-readability.exe'
$includeMain = Join-Path $root 'app\src\main\cpp'
$includeHost = Join-Path $root 'host_tests'
$sources = @(
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp'),
    (Join-Path $root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp'),
    (Join-Path $root 'host_tests\intra_block_readability.cpp')
)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I $includeMain -I $includeHost $sources -o $exe
if ($LASTEXITCODE -ne 0) { throw 'intra-block readability compilation failed' }

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) { throw 'intra-block readability self-test failed' }
    Write-Host 'intra-block readability self-test PASS'
    exit 0
}

& $exe --run --report-root $ReportRoot --tap-root $TapRoot
if ($LASTEXITCODE -ne 0) { throw 'intra-block readability diagnosis run failed' }
Write-Host "intra-block readability diagnosis COMPLETE: $ReportRoot"
