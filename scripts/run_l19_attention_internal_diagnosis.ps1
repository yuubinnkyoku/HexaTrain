# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only ATTENTION_INTERNAL_V1 runner: decompose the deep-layer
# attention-path linear readability loss of the L19 transformer into
# Q/K/V, scores, weights, per-head context, concat, output projection, and
# residual add. No device, QAIRT, ADB, QNN graph, or Android involvement.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-attention-internal-diagnosis'),
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

$exe = Join-Path $ReportRoot 'attention-internal-diagnosis.exe'
[void](New-Item -ItemType Directory -Force -Path $ReportRoot)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root 'app\src\main\cpp') `
    -I (Join-Path $Root 'host_tests') `
    (Join-Path $Root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
    (Join-Path $Root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp') `
    (Join-Path $Root 'host_tests\attention_internal_diagnosis.cpp') `
    -o $exe
if ($LASTEXITCODE -ne 0) {
    throw 'attention-internal diagnosis compilation failed'
}

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) {
        throw 'attention-internal diagnosis self-test failed'
    }
    Write-Host 'attention-internal diagnosis self-test PASS'
    exit 0
}

& $exe --run --report-root $ReportRoot --tap-root $TapRoot
if ($LASTEXITCODE -ne 0) {
    throw 'attention-internal diagnosis run failed'
}