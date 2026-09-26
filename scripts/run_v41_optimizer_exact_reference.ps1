# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Host-only V4.1 exact optimizer reference gate.
# Loads formal HVX Muon NPRTCKPTV4 checkpoints, reconstructs DataCursor
# batches from train_pilot.bin, and writes exact-gradient / Muon / Sinkhorn
# diagnostics. Does not train and does not touch the device.

param(
    [string]$OutputRoot = 'docs/results/v41-optimizer-reference-2026-09',
    [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
    [string]$BpePath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
    [string]$DatasetHash = 'fnv1a64:0c7b2826f5f26fea',
    [switch]$SkipBuild,
    [switch]$NoTrajectory
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-UnderRoot([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $root $Path
}

$OutputRoot = Resolve-UnderRoot $OutputRoot
$CachePath = Resolve-UnderRoot $CachePath
$BpePath = Resolve-UnderRoot $BpePath
$exe = Join-Path $root 'build/host-tests/v41_optimizer_exact_reference.exe'

if (-not $SkipBuild) {
    $compiler = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $compiler) { throw 'g++ not found on PATH' }
    New-Item -ItemType Directory -Force -Path (Join-Path $root 'build/host-tests') | Out-Null
    & g++ -std=c++17 -O2 `
        -I (Join-Path $root 'app/src/main/cpp') `
        -I (Join-Path $root 'app/src/main/cpp/qnn') `
        -o $exe `
        (Join-Path $root 'app/src/main/cpp/tiny_language_model_cpu.cpp') `
        (Join-Path $root 'app/src/main/cpp/nicopedia_muon_optimizer.cpp') `
        (Join-Path $root 'app/src/main/cpp/nicopedia_muon_checkpoint.cpp') `
        (Join-Path $root 'host_tests/v41_optimizer_exact_reference.cpp')
    if ($LASTEXITCODE -ne 0) { throw 'COMPILE_FAILED' }
}

$checkpoints = @(
    @{ step = 500;  path = 'build/reports/hvx-promotion/quality-hvx-step1000/htp-seed1-l19-t32-d64-f128-step500.ckpt'; trajectory = $false },
    @{ step = 2000; path = 'build/reports/hvx-promotion/quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step2000.ckpt'; trajectory = (-not $NoTrajectory) },
    @{ step = 8000; path = 'build/reports/hvx-promotion/quality-hvx-seed1-step8000/htp-seed1-l19-t32-d64-f128-step8000.ckpt'; trajectory = $false }
)

foreach ($item in $checkpoints) {
    $ckpt = Resolve-UnderRoot $item.path
    if (-not (Test-Path -LiteralPath $ckpt)) {
        throw "CHECKPOINT_MISSING:$ckpt"
    }
    $outDir = Join-Path $OutputRoot ("step" + $item.step)
    $args = @(
        '--checkpoint', $ckpt,
        '--cache', $CachePath,
        '--bpe', $BpePath,
        '--output', $outDir,
        '--dataset-hash', $DatasetHash
    )
    if (-not $item.trajectory) { $args += '--no-trajectory' }
    Write-Host "v41_optimizer_exact_reference step=$($item.step)"
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "REFERENCE_FAILED:step=$($item.step)"
    }
}

Write-Host "run_v41_optimizer_exact_reference=PASS output=$OutputRoot"
