# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 intra-block readability diagnosis
# bundle (host-only CPU evidence). Only the files listed in $allowed are
# copied from the private report root; every file is schema-checked, the
# trajectory anchors are cross-checked against the pinned bundle values, and
# generated content is scanned for private identifiers before publication.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-intra-block-readability'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-intra-block-readability-2026-08'),
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
    'tap-probes.csv',
    'tap-transfers.csv',
    'tap-alignments.csv',
    'tap-geometry.csv',
    'tap-aux.csv',
    'clone-parity.csv',
    'token-baselines.csv',
    'tap-free-running.csv',
    'diagnosis.csv',
    'summary.csv',
    'budget.csv'
)
$sourceFiles = @(
    'dataset-anchors.csv',
    'trajectory-anchors.csv',
    'tap-probes.csv',
    'tap-transfers.csv',
    'tap-alignments.csv',
    'tap-geometry.csv',
    'tap-aux.csv',
    'clone-parity.csv',
    'token-baselines.csv',
    'tap-free-running.csv',
    'diagnosis.csv',
    'summary.csv',
    'budget.csv'
)
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
$kVerdicts = @('RESIDUAL_OVERWRITE', 'ATTENTION', 'FFN', 'LAYERNORM',
    'COORDINATE_DRIFT', 'LINEAR_INFO_LOSS', 'MIXED', 'CUMULATIVE')
$kPairVerdicts = @('COORDINATE_TRANSFORM', 'INFORMATION_LOSS', 'MIXED')
$kCadence = @(0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000)

function Fail([string]$Message) { throw "intra-block readability public export: $Message" }

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

function ParseCond([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    if ($Value -eq 'inf') { return [double]::PositiveInfinity }
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

function AssertConfig([string]$Value, [string]$Field, [string]$RowName) {
    if ($allConfigs -notcontains $Value) { Fail "unknown configuration_id: $RowName.$Field = $Value" }
}

$script:DatasetAnchorRows = $null
$script:TrajectoryRows = $null
$script:TapProbeRows = $null
$script:TransferRows = $null
$script:AlignmentRows = $null
$script:GeometryRows = $null
$script:AuxRows = $null
$script:CloneParityRows = $null
$script:BaselineRows = $null
$script:FreeRunningRows = $null
$script:DiagnosisRow = $null
$script:SummaryRows = $null
$script:BudgetRows = $null
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
        AssertConfig $row.configuration_id 'configuration_id' 'trajectory-anchors'
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

    $script:TapProbeRows = @(Import-Csv -LiteralPath (SourcePath 'tap-probes.csv'))
    if ($script:TapProbeRows.Count -ne 498) { Fail 'tap-probes row count mismatch (expected 3x126+120)' }
    foreach ($row in $script:TapProbeRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-probes'
        if ($row.finite -ne 'true') { Fail "tap-probes non-finite probe: $($row.configuration_id) $($row.tap_name)" }
        if ([int]$row.selected_step -lt 0 -or [int]$row.selected_step -gt 2000 -or
            ($kCadence -notcontains [int]$row.selected_step)) { Fail "tap-probes off-cadence selected step: $($row.configuration_id) $($row.tap_name)" }
        foreach ($field in @('train_tf_token_exact', 'cal_tf_token_exact', 'dev_tf_token_exact')) {
            [void](ParseInt $row.$field $field 'tap-probes')
        }
        foreach ($field in @('dev_tf_mean_rank', 'dev_tf_mean_nll', 'dev_tf_mean_margin',
            'dev_tf_margin_q10', 'dev_tf_top2', 'dev_tf_top3')) {
            [void](ParseFinite $row.$field $field 'tap-probes')
        }
    }

    $script:TransferRows = @(Import-Csv -LiteralPath (SourcePath 'tap-transfers.csv'))
    if ($script:TransferRows.Count -ne 444) { Fail 'tap-transfers row count mismatch (expected 444)' }
    foreach ($row in $script:TransferRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-transfers'
        if ($row.pair_kind -notin @('coarse', 'fine')) { Fail "tap-transfers unknown pair_kind: $($row.pair_kind)" }
        if ($row.pair_kind -eq 'coarse' -and $row.variant -notin @('raw', 'norm')) { Fail "tap-transfers unknown coarse variant: $($row.variant)" }
        if ($row.pair_kind -eq 'fine' -and $row.variant -ne 'raw') { Fail "tap-transfers fine variant must be raw: $($row.variant)" }
        foreach ($field in @('src_probe_dev_tf_exact', 'transfer_dev_tf_exact', 'delta_vs_src',
            'dst_probe_dev_tf_exact', 'delta_vs_dst')) {
            [void](ParseInt $row.$field $field 'tap-transfers')
        }
    }

    $script:AlignmentRows = @(Import-Csv -LiteralPath (SourcePath 'tap-alignments.csv'))
    if ($script:AlignmentRows.Count -ne 150) { Fail 'tap-alignments row count mismatch (expected 150)' }
    foreach ($row in $script:AlignmentRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-alignments'
        if ($row.finite -ne 'true') { Fail "tap-alignments non-finite fit: $($row.configuration_id) $($row.src_tap) -> $($row.dst_tap)" }
        if ($kPairVerdicts -notcontains $row.verdict) { Fail "tap-alignments unknown verdict: $($row.verdict)" }
        foreach ($field in @('fit_rank', 'native_dev_tf_exact', 'transfer_dev_tf_exact',
            'aligned_ls_dev_tf_exact', 'aligned_orth_dev_tf_exact', 'residual_loss_ls',
            'residual_loss_orth')) {
            [void](ParseInt $row.$field $field 'tap-alignments')
        }
        foreach ($field in @('cond', 'rel_residual_ls', 'rel_residual_orth', 'recovery_ls', 'recovery_orth')) {
            if ($field -eq 'cond') { [void](ParseCond $row.$field $field 'tap-alignments') }
            else { [void](ParseFinite $row.$field $field 'tap-alignments') }
        }
        $r = [double]$row.recovery_ls
        if ($r -lt 0.0 -or $r -gt 1.0) { Fail "tap-alignments recovery outside [0,1]: $($row.configuration_id) $($row.src_tap)" }
        if ($row.verdict -eq 'COORDINATE_TRANSFORM' -and ($r -lt 0.75 -or [int]$row.residual_loss_ls -gt 1)) {
            Fail "tap-alignments verdict rule violation (CT): $($row.configuration_id) $($row.src_tap)"
        }
        if ($row.verdict -eq 'INFORMATION_LOSS' -and ($r -gt 0.25 -or [int]$row.residual_loss_ls -lt 3)) {
            Fail "tap-alignments verdict rule violation (IL): $($row.configuration_id) $($row.src_tap)"
        }
    }

    $script:GeometryRows = @(Import-Csv -LiteralPath (SourcePath 'tap-geometry.csv'))
    if ($script:GeometryRows.Count -ne 75) { Fail 'tap-geometry row count mismatch (expected 75)' }
    foreach ($row in $script:GeometryRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-geometry'
        if ($row.dataset -ne 'MARGIN_DEVELOPMENT_V1') { Fail "tap-geometry dataset must be MARGIN_DEVELOPMENT_V1: $($row.dataset)" }
        foreach ($field in @('residual_norm', 'attn_update_norm', 'attn_ratio', 'cos_attn',
            'after_attn_norm', 'ffn_update_norm', 'ffn_ratio', 'cos_ffn', 'after_ffn_norm')) {
            [void](ParseFinite $row.$field $field 'tap-geometry')
        }
        if ($row.attn_overwrite -notin @('true', 'false') -or $row.ffn_overwrite -notin @('true', 'false')) {
            Fail "tap-geometry overwrite flags must be boolean: $($row.configuration_id) block $($row.block)"
        }
        $attnExpect = ([double]$row.attn_ratio -gt 1.0) -and ([double]$row.cos_attn -lt -0.5)
        $ffnExpect = ([double]$row.ffn_ratio -gt 1.0) -and ([double]$row.cos_ffn -lt -0.5)
        if (($row.attn_overwrite -eq 'true') -ne $attnExpect) { Fail "tap-geometry attn_overwrite inconsistent: $($row.configuration_id) block $($row.block)" }
        if (($row.ffn_overwrite -eq 'true') -ne $ffnExpect) { Fail "tap-geometry ffn_overwrite inconsistent: $($row.configuration_id) block $($row.block)" }
    }

    $script:AuxRows = @(Import-Csv -LiteralPath (SourcePath 'tap-aux.csv'))
    if ($script:AuxRows.Count -ne 498) { Fail 'tap-aux row count mismatch (expected 3x126+120)' }
    foreach ($row in $script:AuxRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-aux'
        foreach ($field in @('eta2_dev', 'effective_rank_train', 'between_within_dev',
            'mean_pairwise_cosine_dev', 'cond_train')) {
            if ($field -eq 'cond_train') { [void](ParseCond $row.$field $field 'tap-aux') }
            else { [void](ParseFinite $row.$field $field 'tap-aux') }
        }
    }

    $script:CloneParityRows = @(Import-Csv -LiteralPath (SourcePath 'clone-parity.csv'))
    if ($script:CloneParityRows.Count -ne 4) { Fail 'clone-parity row count mismatch (expected 4)' }
    foreach ($row in $script:CloneParityRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'clone-parity'
        if ($row.pass -ne 'true') { Fail "clone-parity must pass: $($row.configuration_id)" }
        if ([double]$row.max_logit_delta -ge 1e-4) { Fail "clone-parity logit delta exceeds 1e-4: $($row.configuration_id)" }
        if ([int]$row.argmax_flips -ne 0 -or [int]$row.rank_flips -ne 0 -or [int]$row.exact_mismatch -ne 0) {
            Fail "clone-parity flips/mismatch must be zero: $($row.configuration_id)"
        }
    }

    $script:BaselineRows = @(Import-Csv -LiteralPath (SourcePath 'token-baselines.csv'))
    if ($script:BaselineRows.Count -ne 9) { Fail 'token-baselines row count mismatch (expected 3x3)' }
    foreach ($row in $script:BaselineRows) {
        if ($row.baseline -notin @('BASELINE_A', 'BASELINE_B', 'BASELINE_C')) { Fail "token-baselines unknown baseline: $($row.baseline)" }
        if ($row.dataset -notin @('TRAIN', 'CAL', 'DEV')) { Fail "token-baselines unknown dataset: $($row.dataset)" }
        if ($row.baseline -eq 'BASELINE_A' -or $row.baseline -eq 'BASELINE_C') {
            if ([int]$row.seen -ne [int]$row.total -or [int]$row.unseen -ne 0) {
                Fail "token-baselines A/C must see every row: $($row.baseline) $($row.dataset)"
            }
        }
    }
    $bCal = @($script:BaselineRows | Where-Object { $_.baseline -eq 'BASELINE_B' -and $_.dataset -eq 'CAL' })[0]
    $bDev = @($script:BaselineRows | Where-Object { $_.baseline -eq 'BASELINE_B' -and $_.dataset -eq 'DEV' })[0]
    $bTrain = @($script:BaselineRows | Where-Object { $_.baseline -eq 'BASELINE_B' -and $_.dataset -eq 'TRAIN' })[0]
    if ([int]$bTrain.seen -ne 32 -or [int]$bTrain.unseen -ne 0) { Fail 'token-baselines B TRAIN must be all seen (32)' }
    if ([int]$bCal.unseen -ne 83 -or [int]$bDev.unseen -ne 83) { Fail 'token-baselines B CAL/DEV unseen must be 83 (deterministic pin)' }

    $script:FreeRunningRows = @(Import-Csv -LiteralPath (SourcePath 'tap-free-running.csv'))
    if ($script:FreeRunningRows.Count -ne 12) { Fail 'tap-free-running row count mismatch (expected 4x3)' }
    foreach ($row in $script:FreeRunningRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'tap-free-running'
        if ($row.all_finite -ne 'true') { Fail "tap-free-running non-finite rollout: $($row.configuration_id) $($row.tap_name)" }
        foreach ($field in @('token_exact', 'token_total', 'sequence_exact', 'sequence_total', 'nll',
            'median_first_error_survival', 'lower_tail_margin_q10')) {
            [void](ParseFinite $row.$field $field 'tap-free-running')
        }
    }

    $script:DiagnosisRow = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    if ($kVerdicts -notcontains $script:DiagnosisRow.verdict) { Fail "diagnosis verdict outside fixed set: $($script:DiagnosisRow.verdict)" }
    if ($script:DiagnosisRow.thresholds_fixed_before_results -ne 'true') { Fail 'diagnosis thresholds_fixed_before_results must be true' }
    if ([string]::IsNullOrWhiteSpace($script:DiagnosisRow.reasons)) { Fail 'diagnosis reasons must not be empty' }

    $script:SummaryRows = @(Import-Csv -LiteralPath (SourcePath 'summary.csv'))
    if ($script:SummaryRows.Count -ne 48) { Fail 'summary row count mismatch (expected 12x4)' }
    foreach ($row in $script:SummaryRows) {
        AssertConfig $row.configuration_id 'configuration_id' 'summary'
        if ($row.scope -notin @('head', 'head_input_tap', 'embedding_tap', 'max_drop_block', 'clone_parity')) {
            Fail "summary unknown scope: $($row.scope)"
        }
        if ($row.metric -eq 'tap_name' -or $row.metric -eq 'pass') { continue }
        if ($row.metric -eq 'block') {
            [void](ParseInt $row.value 'value' 'summary')
            continue
        }
        [void](ParseFinite $row.value 'value' 'summary')
    }

    $script:BudgetRows = @(Import-Csv -LiteralPath (SourcePath 'budget.csv'))
    if ($script:BudgetRows.Count -ne 6) { Fail 'budget row count mismatch (expected 6)' }
    foreach ($row in $script:BudgetRows) {
        if ($row.ok -ne 'true') { Fail "budget limit exceeded: $($row.item)" }
        if ([int]$row.count -gt [int]$row.limit) { Fail "budget count above limit: $($row.item)" }
    }
}

function NewSelfTestFixture() {
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\intra-block-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
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
    if ($manifest.schema -ne 'INTRA_BLOCK_READABILITY_DIAGNOSIS_V1' -or $manifest.schema_version -ne 1) {
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
    $decision = $script:DiagnosisRow
    $clones = @($script:CloneParityRows | ForEach-Object { "$($_.configuration_id): max_logit_delta $($_.max_logit_delta), flips $($_.argmax_flips), exact_mismatch $($_.exact_mismatch), pass $($_.pass)" }) -join '; '
    $maxDrop = @($script:SummaryRows | Where-Object scope -eq 'max_drop_block' | Where-Object metric -eq 'block' | ForEach-Object { "$($_.configuration_id)=block $($_.value)" }) -join ', '
@"
# L19 intra-block readability diagnosis, August 2026

This bundle is a host-only CPU diagnosis that decomposes the deep readout
degradation of the L19 model into intra-block causes. It does not open the
AR_FINAL_HOLDOUT_V3 dataset (hash verified only: $kFinalHash) and performs no
device, HTP, or QNN work. All numbers come from the checked-in CPU reference
implementation (tiny_language_model_cpu.cpp), regenerated deterministically.

## Current status: learned-probe localization superseded

The TRAIN row-contract correction excludes learned-probe absolute tap scores
and block/drop counts below from current cause evidence. Observer output,
head-clone parity, and direct forward interventions are independent. The
seed-instability root-cause investigation is the current decision source.

## Method

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. An intra-block
observer extracts 126 taps on L19 (120 on L18): block input (residual), LN1
input/output, attention context/update, post-attention, LN2 input/output, FFN
update, ReLU, post-FFN. A 32-way linear softmax probe (Adam lr=0.01, 2000
steps, calibration step selection) is trained per tap on TRAIN rows only.
Per-block geometry (update/residual ratios and cosines) and three analyses of
readability transfer between adjacent taps are computed on
MARGIN_DEVELOPMENT_V1 rows only: (1) direct probe transfer, (2) function-level
least-squares alignment and Procrustes alignment with recovery R =
clamp((a-n)/(i-n), 0, 1), and (3) per-op dev TF exact drops. Free-running
rollouts (12) separate gold-prefix from closed-loop degradation. The head is
cloned as a linear probe on the final block output (head-clone parity,
|logit|<=1e-4 tolerance) so probe and head readouts are directly comparable.

Dataset roles follow the pinned protocol: TRAIN = probe/alignment learning
(32 rows), MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1
= final evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
INTRA_BLOCK_READABILITY_V1): 498 independent probe trainings, 444 cross-tap
transfer evals, 150 alignment fits, 12 free-running rollouts, 4 trajectory
regenerations, 3 token baselines; actual structural counts are within all
limits (see budget.csv).

## Anchor integrity

All trajectory anchors (AR_DEVELOPMENT_V3 token/sequence exact and NLL at the
selected and final steps) match the pinned bundle values; integers match
exactly and NLL matches within 1e-6 (float32-limited). See
trajectory-anchors.csv.

## Verdict

| configuration | AR_DEV selected -> final |
|---|---|
$((@($allConfigs | ForEach-Object { "| $(& $line $_) |" }) -join "`n"))

Diagnosis (fixed thresholds, never tuned):
**$($decision.verdict)**

$($decision.reasons)

Head-clone parity on the final block output (all configs): $clones

Max per-block dev-TF drop (block_input probe vs after_ffn probe): $maxDrop

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (INTRA_BLOCK_READABILITY_V1) before
any results were produced.

## Superseding measurement correction (2026-08-05)

The TRAIN probe row builder used by this historical bundle created four
contract-conflicting synthetic rows (28/32 current-token exact rather than
the formal batch's 32/32). Learned-probe absolute tap scores and drop counts
are excluded from subsequent causal claims until regenerated with the
corrected contract. Observer outputs, head-clone parity, and direct forward
interventions are independent of that row builder. Attention-path causality
is instead established by the later Attention-zero versus FFN-zero training
intervention.

## Files

- dataset-anchors.csv - dataset roles and hash pins
- trajectory-anchors.csv - regenerated trajectory vs pinned anchors
- tap-probes.csv - per-tap probe training results and selected steps
- tap-transfers.csv - cross-tap probe transfer (raw/norm, coarse/fine)
- tap-alignments.csv - LS/Procrustes alignment recovery and pair verdicts
- tap-geometry.csv - residual/update norms, ratios, cosines, overwrite flags
- tap-aux.csv - per-tap eta2, effective rank, between/within, cosine
- clone-parity.csv - head-as-probe clone parity on the final block output
- token-baselines.csv - token-only baselines A/B/C (no model)
- tap-free-running.csv - free-running rollouts on selected taps
- diagnosis.csv - intra-block cause classification
- summary.csv - head/probe/max-drop summary rows per configuration
- budget.csv - pre-registered execution budget accounting
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
        Copy-Item -LiteralPath (Join-Path $script:ReportRoot 'diagnosis.csv') -Destination (Join-Path $script:FixtureInput 'diagnosis.csv') -Force
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'diagnosis.csv'))
        $tampered = $text -replace 'thresholds_fixed_before_results', 'thresholds_fixed_after_results'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'diagnosis.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered diagnosis header' $negative
    $script:FixtureInput = $fixture.Input
    $negative2 = {
        Copy-Item -LiteralPath (Join-Path $script:ReportRoot 'trajectory-anchors.csv') -Destination (Join-Path $script:FixtureInput 'trajectory-anchors.csv') -Force
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'))
        # The match column is the last, unquoted field (CsvWriter quotes only
        # fields containing commas/newlines/quotes): tamper ',true' at EOL.
        $tampered = $text -replace '(?m),true\r?$', ',false'
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'trajectory-anchors.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered trajectory match' $negative2
    $script:FixtureInput = $fixture.Input
    $negative3 = {
        Copy-Item -LiteralPath (Join-Path $script:ReportRoot 'dataset-anchors.csv') -Destination (Join-Path $script:FixtureInput 'dataset-anchors.csv') -Force
        $text = [IO.File]::ReadAllText((Join-Path $script:FixtureInput 'dataset-anchors.csv'))
        $tampered = $text.Replace($kFinalHash, 'fnv1a64:0000000000000000')
        [IO.File]::WriteAllText((Join-Path $script:FixtureInput 'dataset-anchors.csv'), $tampered)
        AssertSourceEvidence
    }
    ExpectSelfTestFailure 'tampered final hash' $negative3
    $OutputRoot = $savedOutput
    $script:FixtureInput = $null
    [IO.Directory]::Delete($fixture.Root, $true)
    Write-Host "intra-block readability public exporter self-test PASS"
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
    schema = 'INTRA_BLOCK_READABILITY_DIAGNOSIS_V1'
    schema_version = 1
    protocol = 'INTRA_BLOCK_READABILITY_V1'
    verdict = $script:DiagnosisRow.verdict
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = @($manifestFiles)
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle

Write-Host "intra-block readability public export: PASS ($OutputRoot)"
