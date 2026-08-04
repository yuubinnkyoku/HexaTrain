# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 attention-internal diagnosis bundle
# (host-only CPU evidence). Copies only the allow-listed files from the
# private report root, schema-checks each, verifies the trajectory anchors
# against the pinned values, and scans for private identifiers.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-attention-internal-diagnosis'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-attention-internal-diagnosis-2026-08'),
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'dataset-anchors.csv', 'trajectory-anchors.csv',
    'head-probe-by-seed.csv', 'attention-statistics.csv',
    'head-ablation.csv', 'head-only.csv',
    'context-vs-projection.csv', 'projection-contributions.csv',
    'attention-vs-value-swap.csv', 'head-pair-interactions.csv',
    'teacher-forced-free-running.csv', 'depth-control.csv',
    'diagnosis.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceFiles = @(
    'dataset-anchors.csv', 'trajectory-anchors.csv',
    'head-probe-by-seed.csv', 'attention-statistics.csv',
    'head-ablation.csv', 'head-only.csv',
    'context-vs-projection.csv', 'projection-contributions.csv',
    'attention-vs-value-swap.csv', 'head-pair-interactions.csv',
    'teacher-forced-free-running.csv', 'depth-control.csv',
    'diagnosis.csv', 'next-step-candidates.csv', 'budget.csv'
)
$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
$kPinned = @{
    'L19_SEED_1' = @{ arSelected = 16; arSelTok = 14; arSelSeq = 0; arFinalTok = 30; arFinalSeq = 2; arFinalNll = 8.1239203249880703 }
    'L19_SEED_2' = @{ arSelected = 4; arSelTok = 20; arSelSeq = 0; arFinalTok = 63; arFinalSeq = 6; arFinalNll = 4.1834252619661516 }
    'L19_SEED_4' = @{ arSelected = 12; arSelTok = 22; arSelSeq = 0; arFinalTok = 46; arFinalSeq = 6; arFinalNll = 7.5872917441801651 }
    'L18_SEED_2_CONTROL' = @{ arSelected = 4; arSelTok = 18; arSelSeq = 0; arFinalTok = 65; arFinalSeq = 8; arFinalNll = 5.3026052051209884 }
}
$kVerdicts = @('SPECIFIC_HEAD', 'WEIGHT_SIDE', 'VALUE_SIDE', 'OUTPUT_PROJECTION',
    'HEAD_INTERFERENCE', 'MULTI_HEAD_ACCUMULATION', 'SEED_DEPENDENT', 'UNDETERMINED')
$script:FixtureInput = $null
$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function Fail([string]$Message) { throw "attention-internal public export: $Message" }
function Safe([string]$Text) {
    return $Text -notmatch '(?im)([a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll|bin|exe)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|hardware[_ -]?identifier|android_id|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|\badb\s+-s\s+|\b(?:raw[_ -]?(?:checkpoint|parameters?)|raw[_ -]?(?:adam|optimizer)(?:[_ -]?state)?|raw[_ -]?tensor(?:[_ -]?(?:dump|data))?)\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
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
    $ds = @(Import-Csv -LiteralPath (SourcePath 'dataset-anchors.csv'))
    if ($ds.Count -ne 4) { Fail 'dataset-anchors row count mismatch (expected 4)' }
    foreach ($row in $ds) {
        if ($row.dataset -eq 'TRAIN' -and $row.hash -ne $kTrainHash) { Fail 'TRAIN hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_CALIBRATION_V1' -and $row.hash -ne $kCalibrationHash) { Fail 'CAL hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_DEVELOPMENT_V1' -and $row.hash -ne $kDevelopmentHash) { Fail 'DEV hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.hash -ne $kFinalHash) { Fail 'FINAL hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.rows -ne '0') { Fail 'FINAL holdout must remain unopened' }
    }
    $tr = @(Import-Csv -LiteralPath (SourcePath 'trajectory-anchors.csv'))
    if ($tr.Count -ne 20) { Fail 'trajectory-anchors row count mismatch (expected 4x5)' }
    foreach ($row in $tr) {
        AssertConfig $row.configuration_id 'configuration_id' 'trajectory-anchors'
        if ($row.match -ne 'true') { Fail "trajectory-anchors pinned anchor mismatch: $($row.configuration_id)" }
        $pinned = $kPinned[$row.configuration_id]
        switch ($row.checkpoint) {
            'AR_DEV_SELECTED' {
                if ([int]$row.step -ne $pinned.arSelected) { Fail "trajectory-anchors selected step mismatch: $($row.configuration_id)" }
                if ($row.metric -eq 'token_exact' -and [int]$row.value -ne $pinned.arSelTok) { Fail "trajectory-anchors AR selected token mismatch: $($row.configuration_id)" }
                if ($row.metric -eq 'sequence_exact' -and [int]$row.value -ne $pinned.arSelSeq) { Fail "trajectory-anchors AR selected seq mismatch: $($row.configuration_id)" }
            }
            'AR_DEV_FINAL' {
                if ([int]$row.step -ne 320) { Fail "trajectory-anchors final step mismatch: $($row.configuration_id)" }
                if ($row.metric -eq 'token_exact' -and [int]$row.value -ne $pinned.arFinalTok) { Fail "trajectory-anchors AR final token mismatch: $($row.configuration_id)" }
                if ($row.metric -eq 'sequence_exact' -and [int]$row.value -ne $pinned.arFinalSeq) { Fail "trajectory-anchors AR final seq mismatch: $($row.configuration_id)" }
                if ($row.metric -eq 'autoregressive_nll') {
                    [void](ParseFinite $row.value 'autoregressive_nll' 'trajectory-anchors')
                    if ([math]::Abs([double]$row.value - $pinned.arFinalNll) -gt 1e-6) { Fail "trajectory-anchors AR final NLL mismatch: $($row.configuration_id)" }
                }
            }
            default { Fail "trajectory-anchors unknown checkpoint: $($row.checkpoint)" }
        }
    }
    foreach ($name in @('attention-statistics.csv', 'head-ablation.csv', 'head-only.csv',
        'context-vs-projection.csv', 'projection-contributions.csv',
        'attention-vs-value-swap.csv', 'head-pair-interactions.csv',
        'teacher-forced-free-running.csv', 'depth-control.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
        foreach ($row in $rows) {
            AssertConfig $row.configuration_id 'configuration_id' $name
        }
    }
    foreach ($name in @('next-step-candidates.csv', 'budget.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
    }
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    if ($kVerdicts -notcontains $diag.verdict) { Fail "diagnosis verdict outside fixed set: $($diag.verdict)" }
    if ($diag.thresholds_fixed_before_results -ne 'true') { Fail 'diagnosis thresholds_fixed_before_results must be true' }
    if ([string]::IsNullOrWhiteSpace($diag.reasons)) { Fail 'diagnosis reasons must not be empty' }
    $budget = @(Import-Csv -LiteralPath (SourcePath 'budget.csv'))
    foreach ($row in $budget) {
        if ($row.ok -ne 'true') { Fail "budget limit exceeded: $($row.item)" }
        if ([int]$row.count -gt [int]$row.limit) { Fail "budget count above limit: $($row.item)" }
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
    if ($manifest.schema -ne 'ATTENTION_INTERNAL_DIAGNOSIS_V1' -or $manifest.schema_version -ne 1) { Fail 'manifest schema mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) { Fail 'manifest run accounting mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}
function NewReadme() {
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    $proj = @(Import-Csv -LiteralPath (SourcePath 'context-vs-projection.csv') | ForEach-Object { "$($_.configuration_id): ctx=$($_.ctx_concat_dev_tf_exact) upd=$($_.attn_update_dev_tf_exact) drop=$($_.proj_drop)" }) -join '; '
    @"
# L19 attention-internal diagnosis, August 2026

This bundle is a host-only CPU diagnosis that decomposes the deep-layer
attention-path linear readability loss of the L19 model into
normalized-input, Q/K/V projections, attention scores/weights, per-head
context, concat, output projection, and residual add. It does not open the
AR_FINAL_HOLDOUT_V3 dataset (hash verified only: $kFinalHash) and performs no
device, HTP, or QNN work. All numbers come from the checked-in CPU reference
implementation (tiny_language_model_cpu.cpp), regenerated deterministically.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. An attention
observer extracts, per target layer, the LN1 output, Q/K/V projections,
per-head contexts, concatenated context, attention update and post-attention
residual. A 32-way linear softmax probe (Adam lr=0.01, 2000 steps,
calibration step selection) is trained per tap on TRAIN rows only. Head-level
counterfactual interventions (head zero, head only, cross-seed context swap,
attention-weight/value separation, head pair) are evaluated on
MARGIN_DEVELOPMENT_V1 rows. Cross-seed context swaps are explicit
counterfactual interventions between models, not natural inferences.

Dataset roles follow the pinned protocol: TRAIN = probe learning (32 rows),
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
ATTENTION_INTERNAL_V1): attention taps <= 500, head probe trainings <= 400,
head-zero <= 300, head-only <= 150, context swaps <= 60, attention/value
separation <= 32, head pairs <= 24, free-running <= 40.

## Verdict

Diagnosis (fixed thresholds, never tuned):
**$($diag.verdict)**

$($diag.reasons)

Context vs output-projection (dev TF exact, max-drop layer):
$proj

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (ATTENTION_INTERNAL_V1) before any
results were produced.

## Files

- dataset-anchors.csv / trajectory-anchors.csv - dataset and trajectory anchors
- head-probe-by-seed.csv - per-tap probe results per seed
- attention-statistics.csv - entropy/max-weight/self/prev/cosine statistics
- head-ablation.csv - head-zero ablation results
- head-only.csv - head-only (single head) results
- context-vs-projection.csv - context vs output-projection readability
- projection-contributions.csv - per-head output-projection contributions
- attention-vs-value-swap.csv - attention-weight vs value separation
- head-pair-interactions.csv - head-pair interventions
- teacher-forced-free-running.csv - teacher-forced vs free-running
- depth-control.csv - L18 depth control comparison
- diagnosis.csv - attention-internal cause classification
- next-step-candidates.csv - candidate next steps
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest
"@
}

if ($SelfTest) {
    $script:FixtureInput = $null
    AssertSourceEvidence
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\attention-internal-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
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
        $tampered = $text -replace 'thresholds_fixed_before_results', 'thresholds_fixed_after_results'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'diagnosis.csv'), $tampered)
        AssertSourceEvidence
    } catch { $failed = $true }
    if (-not $failed) { Fail 'self-test negative case did not fail: tampered diagnosis header' }
    $script:FixtureInput = $fixtureInput
    $failed = $false
    try {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'))
        $tampered = $text -replace '(?m),true\r?$', ',false'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'), $tampered)
        AssertSourceEvidence
    } catch { $failed = $true }
    if (-not $failed) { Fail 'self-test negative case did not fail: tampered trajectory match' }
    $OutputRoot = $savedOutput
    $script:FixtureInput = $null
    [IO.Directory]::Delete($fixtureRoot, $true)
    Write-Host "attention-internal public exporter self-test PASS"
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
    schema = 'ATTENTION_INTERNAL_DIAGNOSIS_V1'
    schema_version = 1
    protocol = 'ATTENTION_INTERNAL_V1'
    verdict = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0].verdict
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = @($manifestFiles)
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle
Write-Host "attention-internal public export: PASS ($OutputRoot)"