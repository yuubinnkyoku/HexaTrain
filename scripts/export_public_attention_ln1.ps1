param([Parameter(Mandatory=$true)][string]$CohortPath,[Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference='Stop'
# Public export policy entry. Python implementation selects only aggregate
# metrics and immutable compiler/runtime/fixture hashes; no private ledger copy.
& python (Join-Path $PSScriptRoot 'export_public_attention_ln1.py') $CohortPath $OutputPath
if($LASTEXITCODE -ne 0){throw 'ATTENTION_LN1_PUBLIC_EXPORT_REJECTED'}
Write-Output 'attention_ln1_public_export=PASS'
