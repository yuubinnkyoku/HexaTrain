# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only CPU readout/representation diagnosis for the L19 quality-gate
# investigation. The runner is built from the checked-in CPU reference
# implementation; no Android, QAIRT, ADB, or device state is used. Output is
# private evidence under build/.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot,
    [string]$HiddenRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $root 'build\reports\qnn-readout-representation-diagnosis'
}
if ([string]::IsNullOrWhiteSpace($HiddenRoot)) {
    $HiddenRoot = Join-Path $root 'build\reports\qnn-readout-probe\private-hidden'
}

function Assert-BuildOutput([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $buildRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "ReportRoot/HiddenRoot must remain under build/: $Path"
    }
    return $full
}

$ReportRoot = Assert-BuildOutput $ReportRoot
$HiddenRoot = Assert-BuildOutput $HiddenRoot
[IO.Directory]::CreateDirectory($ReportRoot) | Out-Null
[IO.Directory]::CreateDirectory($HiddenRoot) | Out-Null

$exe = Join-Path $ReportRoot 'readout-probe.exe'
$includeMain = Join-Path $root 'app\src\main\cpp'
$includeHost = Join-Path $root 'host_tests'
$sources = @(
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp'),
    (Join-Path $root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp'),
    (Join-Path $root 'host_tests\readout_probe.cpp')
)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I $includeMain -I $includeHost $sources -o $exe
if ($LASTEXITCODE -ne 0) { throw 'readout probe compilation failed' }

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) { throw 'readout probe self-test failed' }
    Write-Host 'readout probe self-test PASS'
    exit 0
}

& $exe --run --report-root $ReportRoot --hidden-root $HiddenRoot
if ($LASTEXITCODE -ne 0) { throw 'readout probe diagnosis run failed' }
Write-Host "readout probe diagnosis COMPLETE: $ReportRoot"
