param(
    [switch]$SelfTest,
    [switch]$DataAuditOnly,
    [switch]$OptimizationInterventions,
    [switch]$BranchAblations,
    [string]$ReportRoot = "build/reports/qnn-l19-seed-instability-root-cause"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot "build/host-tests"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
$exe = Join-Path $buildRoot "seed_instability_diagnostics.exe"

$compiler = Get-Command clang++ -ErrorAction SilentlyContinue
if (-not $compiler) {
    $compiler = Get-Command g++ -ErrorAction SilentlyContinue
}
if (-not $compiler) {
    throw "CXX_COMPILER_NOT_FOUND"
}

& $compiler.Source -std=c++20 -O2 `
    -I (Join-Path $repoRoot "app/src/main/cpp") `
    -I (Join-Path $repoRoot "host_tests") `
    (Join-Path $repoRoot "host_tests/seed_instability_diagnostics.cpp") `
    (Join-Path $repoRoot "app/src/main/cpp/tiny_language_model_cpu.cpp") `
    -o $exe
if ($LASTEXITCODE -ne 0) {
    throw "SEED_INSTABILITY_COMPILE_FAILED"
}

if ($SelfTest) {
    & $exe --self-test
} elseif ($DataAuditOnly) {
    $resolvedReportRoot = Join-Path $repoRoot $ReportRoot
    & $exe --data-audit-root $resolvedReportRoot
} elseif ($OptimizationInterventions) {
    $resolvedReportRoot = Join-Path $repoRoot $ReportRoot
    & $exe --optimization-root $resolvedReportRoot
} elseif ($BranchAblations) {
    $resolvedReportRoot = Join-Path $repoRoot $ReportRoot
    & $exe --branch-root $resolvedReportRoot
} else {
    $resolvedReportRoot = Join-Path $repoRoot $ReportRoot
    & $exe --report-root $resolvedReportRoot
}
if ($LASTEXITCODE -ne 0) {
    throw "SEED_INSTABILITY_RUN_FAILED"
}
