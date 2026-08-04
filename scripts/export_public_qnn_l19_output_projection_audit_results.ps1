# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 output-projection information audit
# (host-only CPU evidence). Copies only the allow-listed files from the
# private report root, schema-checks each, verifies dataset hashes and budget,
# and scans for private identifiers.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-output-projection-audit'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-output-projection-information-audit-2026-08'),
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'configuration.csv', 'dataset-usage.csv',
    'projection-matrix-summary.csv', 'singular-value-summary.csv',
    'rank-and-conditioning.csv', 'probe-transport-summary.csv',
    'probe-transport-by-seed.csv', 'float-double-comparison.csv',
    'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
    'depth-control.csv', 'diagnosis.csv',
    'previous-result-correction.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceFiles = @(
    'configuration.csv', 'dataset-usage.csv',
    'projection-matrix-summary.csv', 'singular-value-summary.csv',
    'rank-and-conditioning.csv', 'probe-transport-summary.csv',
    'probe-transport-by-seed.csv', 'float-double-comparison.csv',
    'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
    'depth-control.csv', 'diagnosis.csv',
    'previous-result-correction.csv', 'next-step-candidates.csv', 'budget.csv'
)
$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
$kProtocolHash = 'fnv1a64:c35a2e6ae3102772'
$kVerdicts = @('OUTPUT_PROJECTION_PRESERVES_INFORMATION',
    'OUTPUT_PROJECTION_ILL_CONDITIONED',
    'OUTPUT_PROJECTION_LOSES_INFORMATION',
    'SEED_LAYER_DEPENDENT',
    'UNDETERMINED')
$script:FixtureInput = $null
$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function Fail([string]$Message) { throw "output-projection public export: $Message" }
function Safe([string]$Text) {
    return $Text -notmatch '(?im)([a-z]:[\/]|\\\\[^\/\s]+[\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll|bin|exe)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|hardware[_ -]?identifier|android_id|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|\badb\s+-s\s+|\b(?:raw[_ -]?(?:checkpoint|parameters?)|raw[_ -]?(?:adam|optimizer)(?:[_ -]?state)?|raw[_ -]?tensor(?:[_ -]?(?:dump|data))?|raw[_ -]?(?:projection|probe|weight|logit|hidden))\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}
function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}
function RequireUnderRepository([string]$Path, [string]$Purpose) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs')) + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { Fail "$Purpose must remain under docs/: $Path" }
    return $full
}
function RequireOutputRoot([string]$Path) {
    $full = RequireUnderRepository $Path 'OutputRoot'
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { [void](New-Item -ItemType Directory -Path $Path -Force) }
    return $full
}
function SourcePath([string]$Name) {
    if ($script:FixtureInput) { return (Join-Path $script:FixtureInput $Name) }
    return (Join-Path $ReportRoot $Name)
}
function ParseFinite([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float, $script:Invariant, [ref]$parsed)) { Fail "non-numeric value: $RowName.$Field" }
    if (-not [double]::IsFinite($parsed)) { Fail "non-finite value: $RowName.$Field" }
    return $parsed
}
function ParseInt([string]$Value, [string]$Field, [string]$RowName) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, $script:Invariant, [ref]$parsed)) { Fail "non-integer value: $RowName.$Field" }
    return $parsed
}
function AssertConfig([string]$Value, [string]$Field, [string]$RowName) {
    if ($allConfigs -notcontains $Value) { Fail "unknown configuration_id: $RowName.$Field = $Value" }
}

function AssertSourceEvidence() {
    $ds = @(Import-Csv -LiteralPath (SourcePath 'dataset-usage.csv'))
    if ($ds.Count -ne 4) { Fail 'dataset-usage row count mismatch (expected 4)' }
    foreach ($row in $ds) {
        if ($row.dataset -eq 'TRAIN' -and $row.hash -ne $kTrainHash) { Fail 'TRAIN hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_CALIBRATION_V1' -and $row.hash -ne $kCalibrationHash) { Fail 'CAL hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_DEVELOPMENT_V1' -and $row.hash -ne $kDevelopmentHash) { Fail 'DEV hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.hash -ne $kFinalHash) { Fail 'FINAL hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.rows -ne '0') { Fail 'FINAL holdout must remain unopened' }
    }
    $cfg = @(Import-Csv -LiteralPath (SourcePath 'configuration.csv'))
    if ($cfg.Count -ne 4) { Fail 'configuration row count mismatch (expected 4)' }
    foreach ($row in $cfg) { AssertConfig $row.configuration_id 'configuration_id' 'configuration.csv' }
    $missing = @($allConfigs | Where-Object { $_ -notin $cfg.configuration_id })
    if ($missing.Count -gt 0) { Fail "configuration.csv missing: $($missing -join ', ')" }

    foreach ($name in @('projection-matrix-summary.csv', 'singular-value-summary.csv',
        'rank-and-conditioning.csv', 'probe-transport-summary.csv',
        'probe-transport-by-seed.csv', 'float-double-comparison.csv',
        'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
        'depth-control.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
        foreach ($row in $rows) {
            AssertConfig $row.configuration_id 'configuration_id' $name
        }
    }
    foreach ($name in @('diagnosis.csv', 'previous-result-correction.csv',
        'next-step-candidates.csv', 'budget.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
    }
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    if ($kVerdicts -notcontains $diag.verdict) { Fail "diagnosis verdict outside fixed set: $($diag.verdict)" }
    if ($diag.criteria_fixed_before_results -ne 'true') { Fail 'diagnosis criteria_fixed_before_results must be true' }
    $budget = @(Import-Csv -LiteralPath (SourcePath 'budget.csv'))
    foreach ($row in $budget) {
        if ($row.ok -ne 'true') { Fail "budget limit exceeded: $($row.item)" }
        if ([int]$row.count -gt [int]$row.limit) { Fail "budget count above limit: $($row.item)" }
    }
    $nullspace = @(Import-Csv -LiteralPath (SourcePath 'nullspace-summary.csv'))
    foreach ($row in $nullspace) {
        [void](ParseFinite $row.overall_lost_fraction 'overall_lost_fraction' "nullspace-summary:$($row.configuration_id):L$($row.layer)")
    }
    $transport = @(Import-Csv -LiteralPath (SourcePath 'probe-transport-summary.csv'))
    foreach ($row in $transport) {
        [void](ParseFinite $row.max_logit_diff 'max_logit_diff' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
        [void](ParseInt $row.argmax_flips 'argmax_flips' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
        [void](ParseInt $row.token_exact_diff 'token_exact_diff' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
    }
}

function CopySafe([string]$Name) {
    $source = SourcePath $Name
    $text = [IO.File]::ReadAllText($source)
    if (-not (Safe $text)) { Fail "unsafe source content: $Name" }
    WriteUtf8 $Name $text
}
function GetSha256([string]$Name) {
    return (Get-FileHash -LiteralPath (Join-Path $OutputRoot $Name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
function AssertBundle() {
    $entries = @(Get-ChildItem -LiteralPath $OutputRoot -Force)
    if (@($entries | Where-Object { $_.PSIsContainer }).Count -ne 0) { Fail 'public bundle must not contain subdirectories' }
    $actual = @($entries.Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) { Fail 'public bundle allow-list mismatch' }
    foreach ($entry in $entries) {
        if (-not (Safe ([IO.File]::ReadAllText($entry.FullName)))) { Fail "unsafe generated file: $($entry.Name)" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.schema -ne 'OUTPUT_PROJECTION_AUDIT_V1' -or $manifest.schema_version -ne 1) { Fail 'manifest schema mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) { Fail 'manifest run accounting mismatch' }
    if ($manifest.protocol_hash -ne $kProtocolHash) { Fail 'manifest protocol_hash mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}
function NewReadme() {
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    $summary = @(Import-Csv -LiteralPath (SourcePath 'probe-transport-summary.csv') | Where-Object { $_.partition -eq 'DEVELOPMENT' } | ForEach-Object { "$($_.configuration_id): ctx=$($_.context_token_exact) trans=$($_.transport_token_exact) scratch=$($_.from_scratch_token_exact) maxdiff=$($_.max_logit_diff)" }) -join '; '
    @"
# L19 output-projection information audit, August 2026

This bundle is a host-only CPU audit that tests whether the Attention output
projection actually discards linear next-token information. The null
hypothesis is that the projection is invertible and the information is still
present; the alternative is rank deficiency or extreme ill-conditioning that
makes the class direction unreachable.

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. At every target
layer the 16x16 Attention output-projection matrix W is decomposed in double
precision. At the previously reported max-drop layer, a 32-way linear softmax
probe is trained on the concatenated head context (CTX_CONCAT), transported
through W using the pseudoinverse with the protocol's fixed tolerance, and
evaluated on the Attention update (ATT_UPDATE). The same ATT_UPDATE probe is
also trained from scratch and warm-started from the transported weights.

Dataset roles follow the pinned protocol: TRAIN = probe learning (32 rows),
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
OUTPUT_PROJECTION_AUDIT_V1): CPU trajectory regenerations <= 4, matrix
decompositions <= 60, full probe transports <= 24, warm-start trainings <= 24.

## Verdict

Diagnosis (fixed thresholds, never tuned):
**$($diag.verdict)**

$($diag.reason)

Projection-transport parity (DEVELOPMENT, max-drop layer):
$summary

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (OUTPUT_PROJECTION_AUDIT_V1) before
any results were produced.

## Files

- configuration.csv / dataset-usage.csv - configs and dataset role hashes
- projection-matrix-summary.csv / singular-value-summary.csv - matrix decomposition
- rank-and-conditioning.csv - rank and condition number summary
- probe-transport-summary.csv / probe-transport-by-seed.csv - transport parity
- float-double-comparison.csv - float vs double transport accuracy
- nullspace-summary.csv - null-space fraction of the context probe
- from-scratch-vs-transport.csv - optimization comparison
- depth-control.csv - L18 depth control comparison
- diagnosis.csv - formal conclusion
- previous-result-correction.csv - how to rephrase the previous report
- next-step-candidates.csv - candidate follow-ups
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest
"@
}

if ($SelfTest) {
    $script:FixtureInput = $null
    AssertSourceEvidence
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\output-projection-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $fixtureRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $fixtureInput = Join-Path $fixtureRoot 'input'
    $fixtureOutput = Join-Path $fixtureRoot 'output'
    [void](New-Item -ItemType Directory -Path $fixtureInput -Force)
    [void](New-Item -ItemType Directory -Path $fixtureOutput -Force)
    foreach ($name in $sourceFiles) {
        $source = Join-Path $ReportRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture source missing: $name" }
        [IO.File]::Copy($source, (Join-Path $fixtureInput $name), $true)
    }
    $script:FixtureInput = $fixtureInput
    $savedOutput = $OutputRoot
    $OutputRoot = $fixtureOutput
    $failed = $false
    try {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'diagnosis.csv'))
        $tampered = $text -replace 'criteria_fixed_before_results', 'criteria_fixed_after_results'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'diagnosis.csv'), $tampered)
        AssertSourceEvidence
    } catch { $failed = $true }
    if (-not $failed) { Fail 'self-test negative case did not fail: tampered diagnosis header' }
    $script:FixtureInput = $fixtureInput
    $failed = $false
    try {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'budget.csv'))
        $tampered = $text -replace '(?m)^([^,]+,[^,]+,[^,]+,)true$', '$1false'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'budget.csv'), $tampered)
        AssertSourceEvidence
    } catch { $failed = $true }
    if (-not $failed) { Fail 'self-test negative case did not fail: tampered budget ok flag' }
    $OutputRoot = $savedOutput
    $script:FixtureInput = $null
    [IO.Directory]::Delete($fixtureRoot, $true)
    Write-Host "output-projection public export: self-test PASS"
    exit 0
}

$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $ReportRoot -PathType Container)) { Fail 'ReportRoot does not exist' }
AssertSourceEvidence
foreach ($name in $sourceFiles) { CopySafe $name }
WriteUtf8 'README.md' (NewReadme)
$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{ name = $name; sha256 = (GetSha256 $name) }
}
$manifest = [ordered]@{
    schema = 'OUTPUT_PROJECTION_AUDIT_V1'
    schema_version = 1
    protocol = 'OUTPUT_PROJECTION_AUDIT_V1'
    protocol_hash = $kProtocolHash
    verdict = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0].verdict
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = @($manifestFiles)
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle
Write-Host "output-projection public export: PASS ($OutputRoot)"
