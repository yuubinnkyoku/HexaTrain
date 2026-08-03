# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only CPU first-error/margin decomposition runner for the L19
# autoregressive quality shortfall.  The probe is built from the checked-in
# CPU reference implementation; no Android, QAIRT, ADB, or device state is
# used by this script.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $root 'build\reports\qnn-l19-first-error-margin-2026-08'
}

function Resolve-PathText([string]$Path) {
    return [IO.Path]::GetFullPath($Path)
}

function Assert-BuildOutput([string]$Path) {
    $full = Resolve-PathText $Path
    $prefix = $buildRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "ReportRoot must remain under build/: $Path"
    }
    return $full
}

$ReportRoot = Assert-BuildOutput $ReportRoot
[IO.Directory]::CreateDirectory($ReportRoot) | Out-Null

$exe = Join-Path $ReportRoot 'margin-decomposition-probe.exe'
$includeMain = Join-Path $root 'app\src\main\cpp'
$includeHost = Join-Path $root 'host_tests'
$sources = @(
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp'),
    (Join-Path $root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp'),
    (Join-Path $root 'host_tests\margin_decomposition_probe.cpp')
)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I $includeMain -I $includeHost $sources -o $exe
if ($LASTEXITCODE -ne 0) { throw 'margin decomposition probe compilation failed' }

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) { throw 'margin decomposition probe self-test failed' }
    Write-Host 'margin decomposition probe self-test PASS'
    exit 0
}

& $exe --run --output $ReportRoot
if ($LASTEXITCODE -ne 0) { throw 'margin decomposition CPU run failed' }
Write-Host "margin decomposition CPU run COMPLETE: $ReportRoot"
