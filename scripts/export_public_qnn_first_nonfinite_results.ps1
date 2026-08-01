[CmdletBinding()]
param([string]$Destination, [switch]$SelfTest)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'docs\results\qnn-htp-first-nonfinite-2026-07'
$allow = @('README.md','manifest.json','failure-reproduction.csv','checkpoint-replay.csv','operation-localization.csv','minimal-reproducer.csv','fix-validation.csv','training-seeds.csv','generation-oracle.csv','generation-free.csv','ui-validation.csv','post-fix-formal.csv','formal-run-resume.csv')
foreach ($name in $allow) { if (-not (Test-Path (Join-Path $source $name))) { throw "missing allow-listed file: $name" } }
$text = ($allow | ForEach-Object { Get-Content -Raw (Join-Path $source $_) }) -join "`n"
if ($text -match '(?i)(raw checkpoint|adb endpoint[:=]\S|device serial|app\\build\\private|[A-Z]:\\Users\\|\.apk\b|libQnn.*\.so)') { throw 'private or binary material detected' }
if ($text -match '\b\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}\b') { throw 'ADB endpoint material detected' }
$manifest = Get-Content -Raw (Join-Path $source 'manifest.json') | ConvertFrom-Json
if ($manifest.classification -ne 'APP_NUMERIC_RANGE_DEFECT' -or $manifest.files.Count -ne 11) { throw 'manifest consistency failure' }
foreach ($file in $manifest.files) { if ($file -notin $allow) { throw "manifest lists non-allow-listed file: $file" } }
if ($manifest.result -ne 'FIRST_NONFINITE_OPERATION_IDENTIFIED_AND_FIXED_FORMALLY_VALIDATED') { throw 'manifest result mismatch' }
if ($SelfTest) { 'SELF_TEST=PASS'; exit 0 }
if (-not $Destination) { throw '-Destination is required' }
New-Item -ItemType Directory -Force $Destination | Out-Null
foreach ($name in $allow) { Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $Destination $name) }
"exported=$Destination"
