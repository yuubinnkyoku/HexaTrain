param(
    [switch]$SelfTest,
    [switch]$Cycle1,
    [string]$TrainModes = "",
    [string]$OutputFile = "training-results.csv",
    [string]$Hybrids = "",
    [string]$ReportRoot = "build/private-diagnostics/attention-minimal-cause-goal/manual"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot "build/host-tests"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
$exe = Join-Path $buildRoot "attention_minimal_cause.exe"

$compiler = Get-Command clang++ -ErrorAction SilentlyContinue
if (-not $compiler) { $compiler = Get-Command g++ -ErrorAction SilentlyContinue }
if (-not $compiler) { throw "CXX_COMPILER_NOT_FOUND" }

& $compiler.Source -std=c++20 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $repoRoot "app/src/main/cpp") `
    -I (Join-Path $repoRoot "host_tests") `
    (Join-Path $repoRoot "host_tests/attention_minimal_cause.cpp") `
    (Join-Path $repoRoot "app/src/main/cpp/tiny_language_model_cpu.cpp") `
    -o $exe
if ($LASTEXITCODE -ne 0) { throw "ATTENTION_MINIMAL_CAUSE_COMPILE_FAILED" }

$resolvedRoot = Join-Path $repoRoot $ReportRoot
if ($SelfTest) {
    & $exe --self-test
} elseif ($Cycle1) {
    & $exe --cycle1 $resolvedRoot
} elseif ($TrainModes) {
    & $exe --train-modes $TrainModes $OutputFile $resolvedRoot
} elseif ($Hybrids) {
    & $exe --hybrids $Hybrids $resolvedRoot
} else {
    throw "Select -SelfTest, -Cycle1, -TrainModes, or -Hybrids"
}
if ($LASTEXITCODE -ne 0) { throw "ATTENTION_MINIMAL_CAUSE_RUN_FAILED" }
