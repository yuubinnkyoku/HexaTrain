param(
    [switch]$SelfTest,
    [switch]$Cycle1,
    [switch]$Cycle2,
    [switch]$Cycle3,
    [string]$ReportRoot = "build/private-diagnostics/context-supervision-goal/cycle-001"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot "build/host-tests"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
$exe = Join-Path $buildRoot "context_supervision_stability.exe"

$compiler = Get-Command clang++ -ErrorAction SilentlyContinue
if (-not $compiler) { $compiler = Get-Command g++ -ErrorAction SilentlyContinue }
if (-not $compiler) { throw "CXX_COMPILER_NOT_FOUND" }

& $compiler.Source -std=c++20 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $repoRoot "app/src/main/cpp") `
    -I (Join-Path $repoRoot "host_tests") `
    (Join-Path $repoRoot "host_tests/context_supervision_stability.cpp") `
    (Join-Path $repoRoot "app/src/main/cpp/tiny_language_model_cpu.cpp") `
    -o $exe
if ($LASTEXITCODE -ne 0) { throw "CONTEXT_SUPERVISION_STABILITY_COMPILE_FAILED" }

if ($SelfTest) {
    & $exe --self-test
} elseif ($Cycle1) {
    $resolvedRoot = Join-Path $repoRoot $ReportRoot
    & $exe --cycle1 $resolvedRoot
} elseif ($Cycle2) {
    $resolvedRoot = Join-Path $repoRoot $ReportRoot
    & $exe --cycle2 $resolvedRoot
} elseif ($Cycle3) {
    $resolvedRoot = Join-Path $repoRoot $ReportRoot
    & $exe --cycle3 $resolvedRoot
} else {
    throw "Select -SelfTest, -Cycle1, -Cycle2, or -Cycle3"
}
if ($LASTEXITCODE -ne 0) { throw "CONTEXT_SUPERVISION_STABILITY_RUN_FAILED" }
