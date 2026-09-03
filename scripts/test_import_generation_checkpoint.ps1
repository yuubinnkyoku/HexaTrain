# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'import_generation_checkpoint.ps1') -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'GENERATION_CHECKPOINT_IMPORT_SELFTEST_FAILED' }
