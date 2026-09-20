# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Generates or validates the machine-readable artifacts derived from the C++
# SSOT (app/src/main/cpp/transformer_parameter_metadata.h):
#   - metadata/transformer_parameter_metadata.json
#   - app/src/main/java/com/yuubinnkyoku/phonelm/GeneratedTransformerParameterMetadata.kt
#
# Usage:
#   .\scripts\generate_parameter_metadata.ps1          # generate/update checked-in artifacts
#   .\scripts\generate_parameter_metadata.ps1 -Check   # check for staleness (fail if differs)
#   .\scripts\generate_parameter_metadata.ps1 -SelfTest# run generic contract self-tests
param(
    [switch]$Check,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\host-tests"
$Executable = Join-Path $BuildDir "export_transformer_parameter_metadata.exe"
$Source = Join-Path $Root "host_tests\export_transformer_parameter_metadata.cpp"
$JsonArtifact = Join-Path $Root "metadata\transformer_parameter_metadata.json"
$KotlinArtifact = Join-Path $Root "app\src\main\java\com\yuubinnkyoku\phonelm\GeneratedTransformerParameterMetadata.kt"

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Always recompile: the exporter includes the SSOT header transitively, so an
# mtime check on the direct sources alone can leave a stale executable.
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

if ($SelfTest) {
    & $Executable --self-test
    if ($LASTEXITCODE -ne 0) {
        throw "export_transformer_parameter_metadata self-test failed"
    }
    exit 0
}

if ($Check) {
    if (-not (Test-Path -LiteralPath $JsonArtifact -PathType Leaf) -or
        -not (Test-Path -LiteralPath $KotlinArtifact -PathType Leaf)) {
        throw "STALENESS_CHECK_FAILED: generated artifacts missing. Run scripts\generate_parameter_metadata.ps1 to generate them."
    }
    & $Executable --check $JsonArtifact $KotlinArtifact
    if ($LASTEXITCODE -ne 0) {
        throw "STALENESS_CHECK_FAILED: generated artifacts are stale compared to transformer_parameter_metadata.h."
    }
    exit 0
}

# Default: generate / write both artifacts
$metadataDir = Split-Path -Parent $JsonArtifact
if (-not (Test-Path -LiteralPath $metadataDir)) {
    New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
}

& $Executable --write $JsonArtifact $KotlinArtifact
if ($LASTEXITCODE -ne 0) {
    throw "Failed to write generated metadata artifacts"
}

# Verify generated artifacts pass check
& $Executable --check $JsonArtifact $KotlinArtifact
if ($LASTEXITCODE -ne 0) {
    throw "Generated parameter metadata verification failed"
}
Write-Host "Generated parameter metadata: $JsonArtifact"
Write-Host "Generated Kotlin metadata: $KotlinArtifact"