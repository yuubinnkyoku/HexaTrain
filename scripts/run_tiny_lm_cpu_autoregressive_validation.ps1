# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only CPU replay/trajectory runner for AR_VALIDATION_V3.  The probe is
# deliberately built from the checked-in CPU reference implementation; no
# Android, QAIRT, ADB, or device state is used by this script.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot,
    [string]$CheckpointRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $root 'build\reports\qnn-autoregressive-validation-cpu'
}
if ([string]::IsNullOrWhiteSpace($CheckpointRoot)) {
    $CheckpointRoot = Join-Path $root 'build\reports\qnn-depth-quality'
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
$CheckpointRoot = Resolve-PathText $CheckpointRoot
[IO.Directory]::CreateDirectory($ReportRoot) | Out-Null

$exe = Join-Path $ReportRoot 'autoregressive-validation-probe.exe'
$includeMain = Join-Path $root 'app\src\main\cpp'
$includeHost = Join-Path $root 'host_tests'
$sources = @(
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp'),
    (Join-Path $root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp'),
    (Join-Path $root 'host_tests\autoregressive_validation_probe.cpp')
)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I $includeMain -I $includeHost $sources -o $exe
if ($LASTEXITCODE -ne 0) { throw 'autoregressive validation probe compilation failed' }

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) { throw 'autoregressive validation probe self-test failed' }
    Write-Host 'autoregressive validation probe self-test PASS'
    exit 0
}

& $exe --run --output $ReportRoot --checkpoint-root $CheckpointRoot
if ($LASTEXITCODE -ne 0) { throw 'autoregressive validation CPU run failed' }
Write-Host "autoregressive validation CPU run COMPLETE: $ReportRoot"
