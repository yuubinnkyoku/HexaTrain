param(
    [switch]$SelfTest,
    [switch]$Inventory,
    [switch]$Prepare,
    [switch]$AuditEvidence,
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
    $resolved = if ([IO.Path]::IsPathRooted($RelativeOrAbsolute)) {
        [IO.Path]::GetFullPath($RelativeOrAbsolute)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $RelativeOrAbsolute))
    }
    $allowed = [IO.Path]::GetFullPath((Join-Path $repoRoot "build")) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must resolve below the repository build directory"
    }
    $cursor = Get-Item -LiteralPath $resolved -ErrorAction SilentlyContinue
    if (-not $cursor) { $cursor = Get-Item -LiteralPath (Split-Path -Parent $resolved) -ErrorAction SilentlyContinue }
    while ($cursor -and $cursor.FullName.StartsWith($allowed.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "$Label contains a reparse point" }
        $cursor = $cursor.Parent
    }
    return $resolved
}

function Build-Runner {
    New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
    $compileInputs = @(
        $sourceFile, $cpuSource,
        (Join-Path $repoRoot "app/src/main/cpp/tiny_language_model_cpu.h"),
        (Join-Path $repoRoot "app/src/main/cpp/qnn/qnn_runtime.h"),
        (Join-Path $repoRoot "app/src/main/cpp/transformer_resource_estimator.h")
    )
    $executableTime = if (Test-Path -LiteralPath $executable -PathType Leaf) { (Get-Item -LiteralPath $executable).LastWriteTimeUtc } else { $null }
    if ($executableTime -and -not @($compileInputs | Where-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc -gt $executableTime }).Count) {
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

$selectedModes = @($SelfTest, $Inventory, $Prepare, $AuditEvidence, $Benchmark, $Train, $Compare) |
    Where-Object { $_ }
if ($selectedModes.Count -ne 1) {
    throw "Select exactly one mode: -SelfTest, -Inventory, -Prepare, -AuditEvidence, -Benchmark, -Train, or -Compare"
}

$private = Resolve-UnderBuild $PrivateRoot "PrivateRoot"
$reports = Resolve-UnderBuild $ReportRoot "ReportRoot"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "PYTHON_NOT_FOUND" }

if ($SelfTest) {
    $absoluteProbe = [IO.Path]::GetFullPath((Join-Path $repoRoot "build/path-resolution-selftest/probe.bin"))
    if ((Resolve-UnderBuild $absoluteProbe "AbsolutePathProbe") -ne $absoluteProbe) {
        throw "ABSOLUTE_BUILD_PATH_RESOLUTION_FAILED"
    }
    & $python.Source $pythonScript --self-test
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_PIPELINE_SELF_TEST_FAILED" }
    Build-Runner
    & $executable --self-test
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_TRAINER_SELF_TEST_FAILED" }
    exit 0
}

if ($AuditEvidence) {
    & $python.Source $pythonScript --audit-evidence --private-root $private
    if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_PRIVATE_EVIDENCE_AUDIT_FAILED" }
    Build-Runner
    foreach ($auditLayers in @(6, 19)) {
        $auditOutput = Join-Path $reports "formal/l$auditLayers/comparison/checkpoint-provenance.csv"
        $auditCheckpoints = @(1, 2, 4) | ForEach-Object {
            Join-Path $reports "formal/l$auditLayers/seed-$_/selected-private.ckpt"
        }
        foreach ($checkpoint in $auditCheckpoints) {
            if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "CANONICAL_CHECKPOINT_MISSING" }
        }
        & $executable --checkpoint-audit $auditOutput @auditCheckpoints
        if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_CHECKPOINT_EVIDENCE_AUDIT_FAILED" }
    }
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
    if ($Layers -notin @(6, 19) -or $MeasuredSteps -lt 1 -or $MeasuredSteps -gt 100) {
        throw "BENCHMARK_REQUIRES_L6_OR_L19_AND_STEPS_1_TO_100"
    }
    $output = Join-Path $reports "smoke/benchmark-l$Layers.csv"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
    & $executable --benchmark $trainCache $output $Layers $MeasuredSteps
} elseif ($Train) {
    if ($Seed -notin @(1, 2, 4)) { throw "FORMAL_SEED_MUST_BE_1_2_OR_4" }
    if ($Layers -notin @(6, 19)) { throw "FORMAL_LAYERS_MUST_BE_6_OR_19" }
    if ($Context -ne 32 -or $Steps -ne 1000 -or $BatchSize -ne 8 -or $CheckpointInterval -ne 100) {
        throw "FORMAL_PROTOCOL_MUST_BE_T32_STEPS1000_BATCH8_INTERVAL100"
    }
    $output = Join-Path $reports "formal/l$Layers/seed-$Seed"
    & $executable --train $trainCache $validationCache $output $Seed $Layers $Steps $BatchSize $CheckpointInterval
} elseif ($Compare) {
    $suppliedCheckpoints = @($Checkpoint1, $Checkpoint2, $Checkpoint4)
    $canonicalCheckpoints = @(1, 2, 4) | ForEach-Object {
        [IO.Path]::GetFullPath((Join-Path $reports "formal/l$Layers/seed-$_/selected-private.ckpt"))
    }
    for ($checkpointIndex = 0; $checkpointIndex -lt $suppliedCheckpoints.Count; ++$checkpointIndex) {
        $checkpoint = $suppliedCheckpoints[$checkpointIndex]
        if (-not $checkpoint) { throw "Compare requires Checkpoint1, Checkpoint2, and Checkpoint4" }
        $resolved = Resolve-UnderBuild $checkpoint "Checkpoint"
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { throw "CHECKPOINT_NOT_FOUND" }
        if ($resolved -ne $canonicalCheckpoints[$checkpointIndex]) { throw "COMPARE_REQUIRES_CANONICAL_CHECKPOINTS" }
    }
    $output = Join-Path $reports "formal/l$Layers/comparison"
    & $executable --compare $trainCache $developmentCache $output `
        (Resolve-UnderBuild $Checkpoint1 "Checkpoint1") `
        (Resolve-UnderBuild $Checkpoint2 "Checkpoint2") `
        (Resolve-UnderBuild $Checkpoint4 "Checkpoint4")
}
if ($LASTEXITCODE -ne 0) { throw "NICOPEDIA_REAL_TEXT_RUN_FAILED" }
