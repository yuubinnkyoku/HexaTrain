# Explicit final-cohort allow-list export. Defaults to ignored private output.
param([string]$OutputPath='', [switch]$SelfTest)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if(!$OutputPath){$OutputPath=Join-Path $root 'build/reports/ordered-fp32-hvx-qkv/final-public-summary.json'}
$arguments=@((Join-Path $PSScriptRoot 'export_public_ordered_hvx_qkv.py'),'--root',$root,'--output',$OutputPath)
if($SelfTest){$arguments+='--self-test'}
& python @arguments
if($LASTEXITCODE){throw 'QKV_FINAL_COHORT_EXPORT_FAILED'}
