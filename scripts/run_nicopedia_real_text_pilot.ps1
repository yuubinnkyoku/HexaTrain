param(
    [switch]$SelfTest,
    [switch]$Inventory,
    [switch]$Prepare,
    [switch]$Benchmark,
    [switch]$Train,
    [switch]$Compare,
    [string]$SourceRoot = "",
    [string]$PrivateRoot = "build/private-data/nicopedia-real-text",
    [int]$Context = 32,
    [int]$Layers = 6,
    [int]$MeasuredSteps = 5,
    [int]$Seed = 1,
    [int]$Steps = 320,
    [int]$BatchSize = 1,
    [int]$CheckpointInterval = 40,
    [string]$Checkpoint1 = "",
    [string]$Checkpoint2 = "",
    [string]$Checkpoint4 = "",
    [string]$ReportRoot = "build/private-diagnostics/nicopedia-real-text-goal"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $repoRoot "build/host-tests"
$pythonScript = Join-Path $repoRoot "scripts/nicopedia_real_text_pipeline.py"
$sourceFile = Join-Path $repoRoot "host_tests/nicopedia_real_text_pilot.cpp"
$cpuSource = Join-Path $repoRoot "app/src/main/cpp/tiny_language_model_cpu.cpp"
$executable = Join-Path $buildRoot "nicopedia_real_text_pilot.exe"

function Resolve-UnderBuild([string]$RelativeOrAbsolute, [string]$Label) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $repoRoot $RelativeOrAbsolute))
    $allowed = [IO.Path]::GetFullPath((Join-Path $repoRoot "build")) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must resolve below the repository build directory"
    }
    return $resolved
}

function Build-Runner {
    New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
    if ((Test-Path -LiteralPath $executable -PathType Leaf) -and
        (Get-Item -LiteralPath $executable).LastWriteTimeUtc -ge (Get-Item -LiteralPath $sourceFile).LastWriteTimeUtc -and
        (Get-Item -LiteralPath $executable).LastWriteTimeUtc -ge (Get-Item -LiteralPath $cpuSource).LastWriteTimeUtc) {
        return
    }
    $compiler = Get-Command clang++ -ErrorAction SilentlyContinue
    if (-not $compiler) { $compiler = Get-Command g++ -ErrorAction SilentlyContinue }
    if (-not $compiler) { throw "CXX_COMPILER_NOT_FOUND" }
    $arguments = @(
        "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
        "-I", (Join-Path $repoRoot "app/src/main/cpp"),
        "-I", (Join-Path $repoRoot "host_tests"),
        $sourceFile, $cpuSource
    )
    if ($IsWindows -or $env:OS -eq "Windows_NT") { $arguments += "-lpsapi" }
    $arguments += @("-o", $executable)
    & $compiler.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_REAL_TEXT_COMPILE_FAILED" }
}

$selectedModes = @($SelfTest, $Inventory, $Prepare, $Benchmark, $Train, $Compare) |
    Where-Object { $_ }
if ($selectedModes.Count -ne 1) {
    throw "Select exactly one mode: -SelfTest, -Inventory, -Prepare, -Benchmark, -Train, or -Compare"
}

$private = Resolve-UnderBuild $PrivateRoot "PrivateRoot"
$reports = Resolve-UnderBuild $ReportRoot "ReportRoot"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "PYTHON_NOT_FOUND" }

if ($SelfTest) {
    & $python.Source $pythonScript --self-test
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_PIPELINE_SELF_TEST_FAILED" }
    Build-Runner
    & $executable --self-test
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_TRAINER_SELF_TEST_FAILED" }
    exit 0
}

if ($Inventory -or $Prepare) {
    if (-not $SourceRoot) { throw "SourceRoot is required" }
    $source = [IO.Path]::GetFullPath($SourceRoot)
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "SOURCE_ROOT_NOT_FOUND" }
    $mode = if ($Inventory) { "--inventory" } else { "--prepare" }
    & $python.Source $pythonScript $mode --source-root $source --private-root $private --context $Context
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_CORPUS_$($mode.TrimStart('-').ToUpperInvariant())_FAILED" }
    exit 0
}

Build-Runner
$trainCache = Join-Path $private "caches/train_pilot.bin"
$validationCache = Join-Path $private "caches/validation.bin"
$developmentCache = Join-Path $private "caches/development.bin"
foreach ($required in @($trainCache, $validationCache, $developmentCache)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "PRIVATE_CACHE_MISSING" }
}

if ($Benchmark) {
    $output = Join-Path $reports "smoke/benchmark-l$Layers.csv"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
    & $executable --benchmark $trainCache $output $Layers $MeasuredSteps
} elseif ($Train) {
    if ($Seed -notin @(1, 2, 4)) { throw "FORMAL_SEED_MUST_BE_1_2_OR_4" }
    if ($Layers -notin @(6, 19)) { throw "FORMAL_LAYERS_MUST_BE_6_OR_19" }
    $output = Join-Path $reports "formal/l$Layers/seed-$Seed"
    & $executable --train $trainCache $validationCache $output $Seed $Layers $Steps $BatchSize $CheckpointInterval
} elseif ($Compare) {
    foreach ($checkpoint in @($Checkpoint1, $Checkpoint2, $Checkpoint4)) {
        if (-not $checkpoint) { throw "Compare requires Checkpoint1, Checkpoint2, and Checkpoint4" }
        $resolved = Resolve-UnderBuild $checkpoint "Checkpoint"
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { throw "CHECKPOINT_NOT_FOUND" }
    }
    $output = Join-Path $reports "formal/l$Layers/comparison"
    & $executable --compare $trainCache $developmentCache $output `
        (Resolve-UnderBuild $Checkpoint1 "Checkpoint1") `
        (Resolve-UnderBuild $Checkpoint2 "Checkpoint2") `
        (Resolve-UnderBuild $Checkpoint4 "Checkpoint4")
}
if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_REAL_TEXT_RUN_FAILED" }
