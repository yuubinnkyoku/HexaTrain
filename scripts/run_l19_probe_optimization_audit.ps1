# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only PROBE_OPTIMIZATION_AUDIT_V1 runner: determine why the legacy Adam
# linear softmax probe on ATT_UPDATE dev-token-exact trails CTX_CONCAT despite
# full-rank transport parity. No device, QAIRT, ADB, QNN graph, or Android
# involvement; production code is untouched.
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-probe-optimization-audit'),
    [string]$AttnTapRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-attention-internal-diagnosis\private-taps'),
    [string]$IntraTapRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-intra-block-readability\private-taps')
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
Assert-BuildOutput $AttnTapRoot 'AttnTapRoot'
Assert-BuildOutput $IntraTapRoot 'IntraTapRoot'

$exe = Join-Path $ReportRoot 'probe-optimization-audit.exe'
[void](New-Item -ItemType Directory -Force -Path $ReportRoot)

& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $Root 'app\src\main\cpp') `
    -I (Join-Path $Root 'host_tests') `
    (Join-Path $Root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
    (Join-Path $Root 'app\src\main\cpp\qnn\qnn_first_nonfinite_diagnostics.cpp') `
    (Join-Path $Root 'host_tests\probe_optimization_audit.cpp') `
    -o $exe
if ($LASTEXITCODE -ne 0) {
    throw 'probe-optimization audit compilation failed'
}

if ($SelfTest) {
    & $exe --self-test
    if ($LASTEXITCODE -ne 0) {
        throw 'probe-optimization audit self-test failed'
    }
    Write-Host 'probe-optimization audit self-test PASS'
    exit 0
}

& $exe --run --report-root $ReportRoot --attn-tap-root $AttnTapRoot --intra-tap-root $IntraTapRoot
if ($LASTEXITCODE -ne 0) {
    throw 'probe-optimization audit run failed'
}
