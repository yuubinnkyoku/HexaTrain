# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 probe-optimization audit
# (host-only CPU evidence). Copies only the allow-listed files from the
# private report root, schema-checks each, verifies dataset hashes and budget,
# and scans for private identifiers.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-probe-optimization-audit'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-probe-optimization-audit-2026-08'),
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'configuration.csv', 'dataset-usage.csv',
    'legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
    'corrected-attention-taps.csv', 'feature-geometry.csv',
    'row-nullspace.csv', 'calibration-selection.csv',
    'optimization-summary.csv', 'diagnosis.csv',
    'previous-result-corrections.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceFiles = @(
    'configuration.csv', 'dataset-usage.csv',
    'legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
    'corrected-attention-taps.csv', 'feature-geometry.csv',
    'row-nullspace.csv', 'calibration-selection.csv',
    'optimization-summary.csv', 'diagnosis.csv',
    'previous-result-corrections.csv', 'next-step-candidates.csv', 'budget.csv'
)
$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
$kProtocolId = 'PROBE_OPTIMIZATION_AUDIT_V1'
$kProtocolHash = 'fnv1a64:b36b4745b9b4807f'
$kVerdicts = @('C1_OPTIMIZATION_INSUFFICIENCY',
    'C2_STANDARDIZATION',
    'C3_ADAM_COORDINATE_DEPENDENCE',
    'C4_TRAINING_INDETERMINACY',
    'C5_CALIBRATION_SELECTION',
    'UNDETERMINED')
$script:FixtureInput = $null
$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function Fail([string]$Message) { throw "probe-optimization public export: $Message" }
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
    if ([double]::IsNaN($parsed) -or $parsed -eq [double]::PositiveInfinity -or $parsed -eq [double]::NegativeInfinity) { Fail "non-finite value: $RowName.$Field" }
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

    foreach ($name in @('legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
        'corrected-attention-taps.csv', 'feature-geometry.csv',
        'row-nullspace.csv', 'calibration-selection.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
        foreach ($row in $rows) {
            AssertConfig $row.configuration_id 'configuration_id' $name
        }
    }
    $opt = @(Import-Csv -LiteralPath (SourcePath 'optimization-summary.csv'))
    if ($opt.Count -eq 0) { Fail 'optimization-summary.csv empty' }
    foreach ($row in $opt) {
        AssertConfig $row.configuration_id 'configuration_id' 'optimization-summary.csv'
        if ($row.solver -like 'CANONICAL_*') {
            [void](ParseFinite $row.grad_norm 'grad_norm' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
            [void](ParseFinite $row.objective 'objective' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
        }
        [void](ParseFinite $row.train_ce 'train_ce' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
        [void](ParseInt $row.dev_exact 'dev_exact' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
    }
    foreach ($name in @('diagnosis.csv', 'previous-result-corrections.csv',
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
    $lvc = @(Import-Csv -LiteralPath (SourcePath 'legacy-vs-canonical-probe.csv'))
    foreach ($row in $lvc) {
        [void](ParseInt $row.legacy_dev_exact 'legacy_dev_exact' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
        [void](ParseInt $row.canonical_lbfgs_dev_exact 'canonical_lbfgs_dev_exact' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
        [void](ParseFinite $row.canonical_lbfgs_train_ce 'canonical_lbfgs_train_ce' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
    }
}

function CopySource([string]$Name) {
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
    if ($manifest.schema -ne $kProtocolId -or $manifest.schema_version -ne 1) { Fail 'manifest schema mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) { Fail 'manifest run accounting mismatch' }
    if ($manifest.protocol_hash -ne $kProtocolHash) { Fail 'manifest protocol_hash mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}
function NewReadme() {
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    $lvc = @(Import-Csv -LiteralPath (SourcePath 'legacy-vs-canonical-probe.csv'))
    $maxDrop = @($lvc | Where-Object { $_.tap -eq 'ATT_UPDATE' -or $_.tap -eq 'CTX_CONCAT' } | ForEach-Object {
            "$($_.configuration_id):L$($_.layer) $($_.tap) legacy=$($_.legacy_dev_exact) canonical=$($_.canonical_lbfgs_dev_exact)" }) -join '; '
    @"
# L19 probe-optimization audit, August 2026

This bundle is a host-only CPU audit of the probe-optimization hypothesis for
the previously reported CTX_CONCAT vs ATT_UPDATE dev-token-exact drop. The
published READOUT_PROBE_V1 legacy anchors (24/6, 37/24, 57/47, 68/64) are
reproduced bitwise by the runner (cond-1, calibration-selected legacy Adam
probe); this bundle records what a coordinate-stable canonical solver
(PCA-whitened features, L2 on whitened weights only, gauge-fixed, L-BFGS
certified with GD as reference) finds at the same taps.

Headline: at every max-drop layer the canonical CTX and ATT probes reach the
same convergence point (identical whitened-space objective and dev token
exact), so the projection drop is an artifact of the legacy Adam pipeline
(C1_OPTIMIZATION_INSUFFICIENCY, 4/4 layers), not of the representation.

$maxDrop

Protocol $kProtocolId version 6 (AMENDMENT_1..5, fixed before results),
hash $kProtocolHash. Dataset partitions pinned: TRAIN
$kTrainHash, MARGIN_CALIBRATION_V1 $kCalibrationHash,
MARGIN_DEVELOPMENT_V1 $kDevelopmentHash; AR_FINAL_HOLDOUT_V3
$kFinalHash remains unopened. All evidence is CPU host-side; no device,
QAIRT, or QNN involvement.
"@
}

if ($SelfTest) {
    $script:ReportRootBackup = $ReportRoot
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('qnn-probe-opt-export-selftest-' + [Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Path $tempRoot -Force)
    try {
        foreach ($name in $sourceFiles) {
            Copy-Item -LiteralPath (Join-Path $ReportRoot $name) -Destination $tempRoot
        }
        $script:FixtureInput = $tempRoot
        AssertSourceEvidence
        Write-Host 'probe-optimization public export: SELF-TEST PASS (schema checks)'
    } finally {
        $script:FixtureInput = $null
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    exit 0
}

$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $ReportRoot -PathType Container)) { Fail 'ReportRoot does not exist' }
AssertSourceEvidence
foreach ($name in $sourceFiles) { CopySource $name }
WriteUtf8 'README.md' (NewReadme)
$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{ name = $name; sha256 = (GetSha256 $name) }
}
$manifest = [ordered]@{
    schema = $kProtocolId
    schema_version = 1
    protocol_hash = $kProtocolHash
    generated_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ')
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = $manifestFiles
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle
Write-Host "probe-optimization public export: PASS ($OutputRoot)"