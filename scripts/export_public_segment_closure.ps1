param([Parameter(Mandatory=$true)][string]$Cohort,[Parameter(Mandatory=$true)][string]$Source,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
& python (Join-Path $PSScriptRoot 'export_public_segment_closure.py') --cohort $Cohort --source $Source --output $Output
if($LASTEXITCODE){throw 'SEGMENT_CLOSURE_PUBLIC_EXPORT_FAILED'}
