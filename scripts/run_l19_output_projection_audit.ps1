# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only OUTPUT_PROJECTION_AUDIT_V1 runner: determine whether the L19
# attention output projection actually loses linear next-token information.
# No device, QAIRT, ADB, QNN graph, or Android involvement.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-output-projection-audit'),
    [string]$TapRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-attention-internal-diagnosis\private-taps')
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Assert-BuildOutput([string]$Path, [string]$Purpose) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath((Join-Path $Root 'build')) + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Purpose must remain under build/: $Path"
    }
}

Assert-BuildOutput $ReportRoot 'ReportRoot'
Assert-BuildOutput $TapRoot 'TapRoot'

$exe = Join-Path $ReportRoot 'output-projection-audit.exe'
[void](New-Item -ItemType Directory -Force -Path $ReportRoot)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root 'app\src\main\cpp') `
    -I (Join-Path $Root 'host_tests') `
    (Join-Path $Root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
    (Join-Path $Root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp') `
    (Join-Path $Root 'host_tests\output_projection_audit.cpp') `
    -o $exe
if ($LASTEXITCODE -ne 0) {
    throw 'output-projection audit compilation failed'
}

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) {
        throw 'output-projection audit self-test failed'
    }
    Write-Host 'output-projection audit self-test PASS'
    exit 0
}

& $exe --run --report-root $ReportRoot --tap-root $TapRoot
if ($LASTEXITCODE -ne 0) {
    throw 'output-projection audit run failed'
}
