# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 readout/representation diagnosis
# bundle (host-only CPU evidence). Only the files listed in $allowed are
# copied from the private report root; every file is schema-checked, the
# trajectory anchors are cross-checked against the pinned bundle values, and
# generated content is scanned for private identifiers before publication.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-readout-representation-diagnosis'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-readout-representation-diagnosis-2026-08'),
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'dataset-anchors.csv',
    'trajectory-anchors.csv',
    'baseline-current-head.csv',
    'probe-selection.csv',
    'probe-layer-curve.csv',
    'probe-training-grid.csv',
    'head-retraining.csv',
    'representation-metrics.csv',
    'head-geometry.csv',
    'decision.csv',
    'summary.csv'
)
$sourceFiles = @(
    'dataset-anchors.csv',
    'trajectory-anchors.csv',
    'baseline-current-head.csv',
    'probe-selection.csv',
    'probe-layer-curve.csv',
    'probe-training-grid.csv',
    'head-retraining.csv',
    'representation-metrics.csv',
    'head-geometry.csv',
    'decision.csv',
    'summary.csv'
)
$l19Configs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4')
$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
# Pinned anchors (from the AR/margin bundles; the runner asserts these at
# runtime with NLL tolerance 1e-6; integers must match exactly).
$kPinned = @{
    'L19_SEED_1' = @{ arSelected = 16; arSelTok = 14; arSelSeq = 0; arFinalTok = 30; arFinalSeq = 2; arFinalNll = 8.1239203249880703 }
    'L19_SEED_2' = @{ arSelected = 4; arSelTok = 20; arSelSeq = 0; arFinalTok = 63; arFinalSeq = 6; arFinalNll = 4.1834252619661516 }
    'L19_SEED_4' = @{ arSelected = 12; arSelTok = 22; arSelSeq = 0; arFinalTok = 46; arFinalSeq = 6; arFinalNll = 7.5872917441801651 }
    'L18_SEED_2_CONTROL' = @{ arSelected = 4; arSelTok = 18; arSelSeq = 0; arFinalTok = 65; arFinalSeq = 8; arFinalNll = 5.3026052051209884 }
}
$kVeridicts = @('READOUT_FAILURE', 'DEEP_DEGRADATION', 'GENERALIZATION_GAP',
    'MIXED_READOUT_FAILURE_AND_DEEP_DEGRADATION', 'UNDETERMINED')
$kCadence = @(0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000)

function Fail([string]$Message) { throw "readout diagnosis public export: $Message" }

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
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        Fail "$Purpose must remain under docs/: $Path"
    }
    return $full
}

function RequireOutputRoot([string]$Path) {
    $full = RequireUnderRepository $Path 'OutputRoot'
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        [void](New-Item -ItemType Directory -Path $Path -Force)
    }
    return $full
}

function SourcePath([string]$Name) {
    if ($script:FixtureInput) { return (Join-Path $script:FixtureInput $Name) }
    return (Join-Path $ReportRoot $Name)
}

function RequireHeader([string]$Name, [string[]]$Expected) {
    $path = SourcePath $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "source file missing: $Name" }
    $actual = @((Get-Content -LiteralPath $path -TotalCount 1).Split(','))
    if (($actual -join ',') -ne ($Expected -join ',')) { Fail "source schema mismatch: $Name" }
}

$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function ParseFinite([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float, $script:Invariant, [ref]$parsed)) {
        Fail "non-numeric value: $RowName.$Field"
    }
    if (-not [double]::IsFinite($parsed)) { Fail "non-finite value: $RowName.$Field" }
    return $parsed
}

function ParseInt([string]$Value, [string]$Field, [string]$RowName) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, $script:Invariant, [ref]$parsed)) {
        Fail "non-integer value: $RowName.$Field"
    }
    return $parsed
}

$script:DatasetAnchorRows = $null
$script:TrajectoryRows = $null
$script:BaselineRows = $null
$script:ProbeSelectionRows = $null
$script:LayerCurveRows = $null
$script:GridRows = $null
$script:HeadRetrainingRows = $null
$script:RepMetricRows = $null
$script:HeadGeometryRows = $null
$script:DecisionRow = $null
$script:SummaryRows = $null
$script:FixtureInput = $null

function AssertSourceEvidence() {
    $script:DatasetAnchorRows = @(Import-Csv -LiteralPath (SourcePath 'dataset-anchors.csv'))
    if ($script:DatasetAnchorRows.Count -ne 4) { Fail 'dataset-anchors row count mismatch (expected 4)' }
    foreach ($row in $script:DatasetAnchorRows) {
        if ($row.dataset -eq 'TRAIN' -and $row.hash -ne $kTrainHash) { Fail 'TRAIN hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_CALIBRATION_V1' -and $row.hash -ne $kCalibrationHash) { Fail 'CAL hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_DEVELOPMENT_V1' -and $row.hash -ne $kDevelopmentHash) { Fail 'DEV hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.hash -ne $kFinalHash) { Fail 'FINAL hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.rows -ne '0') { Fail 'FINAL holdout must remain unopened' }
    }

    $script:TrajectoryRows = @(Import-Csv -LiteralPath (SourcePath 'trajectory-anchors.csv'))
    if ($script:TrajectoryRows.Count -ne 20) { Fail 'trajectory-anchors row count mismatch (expected 4x5)' }
    foreach ($row in $script:TrajectoryRows) {
        if ($allConfigs -notcontains $row.configuration_id) { Fail "trajectory-anchors unknown config: $($row.configuration_id)" }
        if ($row.match -ne 'true') { Fail "trajectory-anchors pinned anchor mismatch: $($row.configuration_id) $($row.checkpoint) $($row.metric)" }
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

    $script:BaselineRows = @(Import-Csv -LiteralPath (SourcePath 'baseline-current-head.csv'))
    if ($script:BaselineRows.Count -ne 12) { Fail 'baseline-current-head row count mismatch (expected 4x3 checkpoints)' }
    foreach ($row in $script:BaselineRows) {
        if ($allConfigs -notcontains $row.configuration_id) { Fail "baseline unknown config: $($row.configuration_id)" }
        foreach ($field in @('head_train_tf_token_exact', 'head_cal_tf_token_exact', 'head_dev_tf_token_exact',
            'head_cal_fr_token_exact', 'head_cal_fr_sequence_exact', 'head_dev_fr_token_exact',
            'head_dev_fr_sequence_exact', 'head_dev_fr_nll', 'head_dev_fr_median_survival',
            'head_dev_fr_margin_q10', 'head_dev_tf_mean_rank', 'head_dev_tf_mean_nll')) {
            [void](ParseFinite $row.$field $field 'baseline-current-head')
        }
    }

    $script:ProbeSelectionRows = @(Import-Csv -LiteralPath (SourcePath 'probe-selection.csv'))
    $expectedProbes = (22 * 3) + 21 + (8 * 8)
    if ($script:ProbeSelectionRows.Count -ne $expectedProbes) { Fail "probe-selection row count mismatch (expected $expectedProbes)" }
    foreach ($row in $script:ProbeSelectionRows) {
        if ($allConfigs -notcontains $row.configuration_id) { Fail "probe-selection unknown config: $($row.configuration_id)" }
        if ($row.finite -ne 'true') { Fail "probe-selection non-finite probe: $($row.configuration_id) step $($row.checkpoint_step) rep $($row.rep)" }
        if ([int]$row.selected_step -lt 0 -or [int]$row.selected_step -gt 2000 -or
            ($kCadence -notcontains [int]$row.selected_step)) { Fail "probe-selection off-cadence selected step: $($row.configuration_id) rep $($row.rep)" }
        foreach ($field in @('train_ce', 'cal_ce', 'dev_tf_mean_rank', 'dev_tf_mean_nll', 'dev_fr_nll')) {
            if (-not [string]::IsNullOrWhiteSpace($row.$field)) { [void](ParseFinite $row.$field $field 'probe-selection') }
        }
    }

    $script:LayerCurveRows = @(Import-Csv -LiteralPath (SourcePath 'probe-layer-curve.csv'))
    if ($script:LayerCurveRows.Count -ne $expectedProbes) { Fail "probe-layer-curve row count mismatch (expected $expectedProbes)" }
    foreach ($row in $script:LayerCurveRows) {
        foreach ($field in @('probe_dev_fr_token_exact', 'head_dev_fr_token_exact', 'probe_minus_head_fr',
            'probe_dev_tf_token_exact', 'probe_train_tf_token_exact', 'probe_cal_fr_token_exact')) {
            [void](ParseFinite $row.$field $field 'probe-layer-curve')
        }
    }

    $script:GridRows = @(Import-Csv -LiteralPath (SourcePath 'probe-training-grid.csv'))
    if ($script:GridRows.Count -ne ($expectedProbes * 81)) { Fail "probe-training-grid row count mismatch (expected $($expectedProbes * 81))" }
    foreach ($row in $script:GridRows) {
        if ($kCadence -notcontains [int]$row.grid_step) { Fail "probe-training-grid off-cadence step: $($row.grid_step)" }
        foreach ($field in @('train_ce', 'cal_ce')) { [void](ParseFinite $row.$field $field 'probe-training-grid') }
    }
    $selectedPerProbe = @($script:GridRows | Group-Object configuration_id, checkpoint_step, rep | ForEach-Object {
        @($_.Group | Group-Object is_selected | Where-Object Name -eq 'true').Count
    } | Sort-Object -Unique)
    if (($selectedPerProbe -join ',') -ne '1') { Fail 'probe-training-grid is_selected must appear exactly once per probe' }

    $script:HeadRetrainingRows = @(Import-Csv -LiteralPath (SourcePath 'head-retraining.csv'))
    if ($script:HeadRetrainingRows.Count -ne 12) { Fail 'head-retraining row count mismatch (expected 4x3 candidates)' }
    foreach ($row in $script:HeadRetrainingRows) {
        if ($allConfigs -notcontains $row.configuration_id) { Fail "head-retraining unknown config: $($row.configuration_id)" }
        if ($row.candidate -notin @('A_WARM_START', 'B_REINIT', 'C_BIAS_ONLY')) { Fail "head-retraining unknown candidate: $($row.candidate)" }
        if ($row.finite -ne 'true') { Fail "head-retraining non-finite run: $($row.configuration_id) $($row.candidate)" }
        if ($row.frozen_unchanged -ne 'true') { Fail "head-retraining freeze violated: $($row.configuration_id) $($row.candidate)" }
        foreach ($field in @('train_ce', 'cal_ce', 'dev_tf_mean_nll', 'dev_fr_nll',
            'dev_fr_median_survival', 'dev_fr_margin_q10')) {
            [void](ParseFinite $row.$field $field 'head-retraining')
        }
    }

    $script:RepMetricRows = @(Import-Csv -LiteralPath (SourcePath 'representation-metrics.csv'))
    if ($script:RepMetricRows.Count -ne ($expectedProbes * 2)) { Fail "representation-metrics row count mismatch (expected $($expectedProbes * 2))" }
    foreach ($row in $script:RepMetricRows) {
        if ($row.dataset -notin @('TRAIN', 'MARGIN_DEVELOPMENT_V1')) { Fail "representation-metrics unknown dataset: $($row.dataset)" }
        foreach ($field in @('eta2', 'effective_rank', 'norm_ratio', 'hidden_margin_midmedian',
            'sign_agreement', 'alignment_cosine')) {
            [void](ParseFinite $row.$field $field 'representation-metrics')
        }
    }

    $script:HeadGeometryRows = @(Import-Csv -LiteralPath (SourcePath 'head-geometry.csv'))
    if ($script:HeadGeometryRows.Count -ne 196) { Fail 'head-geometry row count mismatch (expected 4x(32+16+1))' }
    foreach ($row in $script:HeadGeometryRows) {
        if ($row.item -notin @('class_row_norm', 'singular_value', 'effective_rank')) { Fail "head-geometry unknown item: $($row.item)" }
        [void](ParseFinite $row.value 'value' 'head-geometry')
    }

    $script:DecisionRow = @(Import-Csv -LiteralPath (SourcePath 'decision.csv'))[0]
    if ($kVeridicts -notcontains $script:DecisionRow.verdict) { Fail "decision verdict outside fixed set: $($script:DecisionRow.verdict)" }
    if ($script:DecisionRow.thresholds_fixed_before_results -ne 'true') { Fail 'decision thresholds_fixed_before_results must be true' }
    if ([string]::IsNullOrWhiteSpace($script:DecisionRow.reasons)) { Fail 'decision reasons must not be empty' }

    $script:SummaryRows = @(Import-Csv -LiteralPath (SourcePath 'summary.csv'))
    if ($script:SummaryRows.Count -ne 84) { Fail 'summary row count mismatch (expected 12x7)' }
    foreach ($row in $script:SummaryRows) {
        if ($row.scope -notin @('head_tf', 'head_fr')) { Fail "summary unknown scope: $($row.scope)" }
        [void](ParseFinite $row.value 'value' 'summary')
    }
}

function NewSelfTestFixture() {
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\readout-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
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
    return [pscustomobject]@{ Root = $fixtureRoot; Input = $fixtureInput; Output = $fixtureOutput }
}

function ExpectSelfTestFailure([string]$Name, [scriptblock]$Action) {
    $failed = $false
    try { & $Action } catch { $failed = $true }
    if (-not $failed) { Fail "self-test negative case did not fail: $Name" }
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
    if ($manifest.schema -ne 'READOUT_REPRESENTATION_DIAGNOSIS_V1' -or $manifest.schema_version -ne 1) {
        Fail 'manifest schema mismatch'
    }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) {
        Fail 'manifest run accounting mismatch'
    }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}

function NewReadme() {
    $trajectory = $script:TrajectoryRows
    $line = {
        param($config)
        $rows = @($trajectory | Where-Object configuration_id -eq $config)
        $sel = @($rows | Where-Object checkpoint -eq 'AR_DEV_SELECTED')
        $fin = @($rows | Where-Object checkpoint -eq 'AR_DEV_FINAL')
        $selTok = @($sel | Where-Object metric -eq 'token_exact')[0].value
        $finTok = @($fin | Where-Object metric -eq 'token_exact')[0].value
        $finSeq = @($fin | Where-Object metric -eq 'sequence_exact')[0].value
        "${config}: AR_DEV $selTok/32 tok at selected step -> $finTok/32 tok ($finSeq/8 seq) at step 320"
    }
    $head = @($script:HeadRetrainingRows | Where-Object configuration_id -eq 'L19_SEED_1' | Where-Object candidate -eq 'A_WARM_START')[0]
    $decision = $script:DecisionRow
@"
# L19 readout / representation diagnosis, August 2026

This bundle is a host-only CPU diagnosis of why the L19 model fails its
generation quality gate. It does not open the AR_FINAL_HOLDOUT_V3 dataset
(hash verified only: $kFinalHash) and performs no device, HTP, or QNN work.
All numbers come from the checked-in CPU reference implementation
(tiny_language_model_cpu.cpp), regenerated deterministically.

## Current status: learned-probe measurements superseded

The TRAIN row-contract correction excludes learned-probe absolute scores and
depth curves below from current cause evidence. AR trajectories, head-clone
parity, and direct head interventions are independent. The seed-instability
root-cause investigation is the current decision source.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control), the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe. At three checkpoints per config (AR-selected step, best
token-exact step, final step 320) a hidden-state observer extracts teacher
forced features for every layer (22 representations on L19: embedded input,
19 block outputs, pre/post final-layer-norm; 21 on L18). A 32-way linear
softmax probe (Adam lr=0.01, 2000 steps, calibration step selection) is
trained per layer on TRAIN rows only. Free-running rollouts are scored with
the current head, with the probe, and with the probe evaluated on head
contexts (drift analysis). Additionally the final output head is retrained
three ways on the step-320 checkpoint: warm-start (A), re-init (B), and
bias-only (C), always freezing everything except the trained parameter.

Dataset roles follow the pinned protocol: TRAIN = probe/head learning,
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened.

## Anchor integrity

All trajectory anchors (AR_DEVELOPMENT_V3 token/sequence exact and NLL at the
selected and final steps) match the pinned bundle values; integers match
exactly and NLL matches within 1e-6 (float32-limited). See
trajectory-anchors.csv.

## Verdict

| configuration | AR_DEV selected -> final |
|---|---|
$((@($l19Configs + @('L18_SEED_2_CONTROL')) | ForEach-Object { "| $(& $line $_) |" }) -join "`n")

Cause classification (fixed thresholds, never tuned):
**$($decision.verdict)**

$($decision.reasons)

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (READOUT_PROBE_V1) before any
results were produced.

## Superseding measurement correction (2026-08-05)

The seed-instability root-cause re-audit found that the legacy TRAIN probe
row builder interpreted position-wise training targets as a continuation and
created four contract-conflicting rows (current-token exact 28/32 rather than
the formal batch's 32/32). Learned-probe absolute scores and depth curves in
this historical bundle are excluded from subsequent causal claims until
regenerated with the corrected row contract. AR trajectories, head-clone
parity, and head interventions are independent of that row builder.

## Files

- dataset-anchors.csv - dataset roles and hash pins
- trajectory-anchors.csv - regenerated trajectory vs pinned anchors
- baseline-current-head.csv - current head TF/FR metrics per checkpoint
- probe-selection.csv - per-layer probe training results and selected steps
- probe-layer-curve.csv - probe free-running curve vs head per layer
- probe-training-grid.csv - full 81-point calibration grid per probe
- head-retraining.csv - output-head retraining A/B/C results
- representation-metrics.csv - eta2/effective-rank/norm/agreement per layer
- head-geometry.csv - output head row norms, singular values, effective rank
- decision.csv - cause classification
- summary.csv - head TF/FR summary rows per checkpoint
- manifest.json - SHA-256 allow-list manifest
"@
}

$selfTestContext = $null
if ($SelfTest) {
    $script:FixtureInput = $null
    # Pre-fly over the live private root, then run negative cases on a
    # disposable fixture so the live reports are never modified.
    AssertSourceEvidence
    $fixture = NewSelfTestFixture
    $script:FixtureInput = $fixture.Input
    $savedOutput = $OutputRoot
    $OutputRoot = $fixture.Output
    $negative = {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'decision.csv'))
        $tampered = $text -replace 'thresholds_fixed_before_results', 'thresholds_fixed_after_results'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'decision.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered decision header' $negative
    $script:FixtureInput = $fixture.Input
    $negative2 = {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'))
        $tampered = $text -replace '"true"', '"false"'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered trajectory match' $negative2
    $script:FixtureInput = $fixture.Input
    $negative3 = {
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'dataset-anchors.csv'))
        $tampered = $text.Replace($kFinalHash, 'fnv1a64:0000000000000000')
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'dataset-anchors.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered final hash' $negative3
    $OutputRoot = $savedOutput
    $script:FixtureInput = $null
    [IO.Directory]::Delete($fixture.Root, $true)
    Write-Host "readout diagnosis public exporter self-test PASS"
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
    schema = 'READOUT_REPRESENTATION_DIAGNOSIS_V1'
    schema_version = 1
    protocol = 'READOUT_PROBE_V1'
    verdict = $script:DecisionRow.verdict
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = @($manifestFiles)
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle

Write-Host "readout diagnosis public export: PASS ($OutputRoot)"
