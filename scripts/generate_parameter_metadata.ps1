# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Generates or validates metadata/transformer_parameter_metadata.json
# directly from the C++ SSOT (app/src/main/cpp/transformer_parameter_metadata.h).
#
# Usage:
#   .\scripts\generate_parameter_metadata.ps1          # generate/update checked-in artifact
#   .\scripts\generate_parameter_metadata.ps1 -Check   # check for staleness (fail if differs)
#   .\scripts\generate_parameter_metadata.ps1 -SelfTest# run contract self-tests
param(
    [switch]$Check,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\host-tests"
$Executable = Join-Path $BuildDir "export_transformer_parameter_metadata.exe"
$Source = Join-Path $Root "host_tests\export_transformer_parameter_metadata.cpp"
$Header = Join-Path $Root "app\src\main\cpp\transformer_parameter_metadata.h"
$Artifact = Join-Path $Root "metadata\transformer_parameter_metadata.json"

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

$needBuild = $true
if (Test-Path -LiteralPath $Executable -PathType Leaf) {
    $exeTime = (Get-Item -LiteralPath $Executable).LastWriteTimeUtc
    $srcTime = (Get-Item -LiteralPath $Source).LastWriteTimeUtc
    $hdrTime = (Get-Item -LiteralPath $Header).LastWriteTimeUtc
    if ($exeTime -gt $srcTime -and $exeTime -gt $hdrTime) {
        $needBuild = $false
    }
}

if ($needBuild) {
    if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
        throw "g++ compiler not found on PATH (required to build export_transformer_parameter_metadata)"
    }
    & g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
        -I (Join-Path $Root "app\src\main\cpp") `
        $Source `
        -o $Executable
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to compile export_transformer_parameter_metadata"
    }
}

if ($SelfTest) {
    & $Executable --self-test
    if ($LASTEXITCODE -ne 0) {
        throw "export_transformer_parameter_metadata self-test failed"
    }
    exit 0
}

if ($Check) {
    if (-not (Test-Path -LiteralPath $Artifact -PathType Leaf)) {
        throw "STALENESS_CHECK_FAILED: $Artifact does not exist. Run scripts\generate_parameter_metadata.ps1 to generate it."
    }
    & $Executable --check $Artifact
    if ($LASTEXITCODE -ne 0) {
        throw "STALENESS_CHECK_FAILED: $Artifact is stale compared to transformer_parameter_metadata.h."
    }
    exit 0
}

# Default: generate / write
$metadataDir = Split-Path -Parent $Artifact
if (-not (Test-Path -LiteralPath $metadataDir)) {
    New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
}

& $Executable --write $Artifact
if ($LASTEXITCODE -ne 0) {
    throw "Failed to write parameter metadata to $Artifact"
}

# Verify generated artifact passes check
& $Executable --check $Artifact
if ($LASTEXITCODE -ne 0) {
    throw "Generated parameter metadata verification failed"
}
Write-Host "Generated parameter metadata: $Artifact"
