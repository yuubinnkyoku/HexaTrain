# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only CPU checkpoint-objective benchmark runner for the L19
# critical-margin stabilization investigation. The probe is built from the
# checked-in CPU reference implementation; no Android, QAIRT, ADB, or device
# state is used by this script. Output is private evidence under build/.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [switch]$Train,
    [switch]$Micro,
    [string]$BaselineDir,
    [string]$ReportRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
if ([string]::IsNullOrWhiteSpace($ReportRoot)) {
    $ReportRoot = Join-Path $root 'build\reports\qnn-critical-margin-objective'
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

$exe = Join-Path $ReportRoot 'critical-margin-objective-probe.exe'
$includeMain = Join-Path $root 'app\src\main\cpp'
$includeHost = Join-Path $root 'host_tests'
$sources = @(
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp'),
    (Join-Path $root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp'),
    (Join-Path $root 'host_tests\critical_margin_objective_probe.cpp')
)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I $includeMain -I $includeHost $sources -o $exe
if ($LASTEXITCODE -ne 0) { throw 'critical margin objective probe compilation failed' }

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) { throw 'critical margin objective probe self-test failed' }
    Write-Host 'critical margin objective probe self-test PASS'
    exit 0
}

if ($Train) {
    if ([string]::IsNullOrWhiteSpace($BaselineDir)) {
        throw '-BaselineDir is required with -Train'
    }
    $baseline = [IO.Path]::GetFullPath($BaselineDir)
    if ($Micro) {
        & $exe --train --micro --baseline $baseline --output $ReportRoot
    } else {
        & $exe --train --baseline $baseline --output $ReportRoot
    }
    if ($LASTEXITCODE -ne 0) { throw 'critical margin objective training run failed' }
    Write-Host "critical margin objective training COMPLETE: $ReportRoot"
    exit 0
}

& $exe --run --output $ReportRoot
if ($LASTEXITCODE -ne 0) { throw 'critical margin objective CPU benchmark failed' }
Write-Host "critical margin objective CPU benchmark COMPLETE: $ReportRoot"
