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
$sourceSchemas = [ordered]@{
    'dataset-anchors.csv' = 'dataset,role,hash,rows'
    'trajectory-anchors.csv' = 'configuration_id,checkpoint,step,metric,value,pinned_anchor,match'
    'baseline-current-head.csv' = 'configuration_id,checkpoint_step,is_final,head_train_tf_token_exact,head_cal_tf_token_exact,head_dev_tf_token_exact,head_cal_fr_token_exact,head_cal_fr_sequence_exact,head_dev_fr_token_exact,head_dev_fr_sequence_exact,head_dev_fr_nll,head_dev_fr_median_survival,head_dev_fr_margin_q10,head_dev_tf_mean_rank,head_dev_tf_mean_nll'
    'probe-selection.csv' = 'configuration_id,checkpoint_step,rep,rep_name,selected_step,finite,nonfinite_step,nonfinite_what,train_ce,train_token_exact,cal_ce,cal_token_exact,max_logit_abs,dev_tf_token_exact,dev_tf_mean_rank,dev_tf_mean_nll,dev_fr_token_exact,dev_fr_sequence_exact,dev_fr_nll,dev_head_ctx_token_exact,dev_head_ctx_seq_exact,step2000_train_ce,step2000_train_token_exact'
    'probe-layer-curve.csv' = 'configuration_id,checkpoint_step,rep,rep_name,probe_dev_fr_token_exact,head_dev_fr_token_exact,probe_minus_head_fr,probe_dev_tf_token_exact,probe_train_tf_token_exact,probe_cal_fr_token_exact'
    'probe-training-grid.csv' = 'configuration_id,checkpoint_step,rep,grid_step,train_ce,cal_ce,train_token_exact,cal_token_exact,is_selected'
    'head-retraining.csv' = 'configuration_id,candidate,selected_step,finite,frozen_unchanged,train_ce,train_token_exact,cal_ce,cal_token_exact,dev_tf_token_exact,dev_tf_mean_nll,dev_fr_token_exact,dev_fr_sequence_exact,dev_fr_nll,dev_fr_median_survival,dev_fr_margin_q10,dev_tf320_token_exact,dev_fr320_token_exact,dev_fr320_sequence_exact'
    'representation-metrics.csv' = 'configuration_id,checkpoint_step,rep,rep_name,dataset,eta2,effective_rank,norm_ratio,hidden_margin_midmedian,sign_agreement,alignment_cosine'
    'head-geometry.csv' = 'configuration_id,item,index,value'
    'decision.csv' = 'verdict,reasons,thresholds_fixed_before_results'
    'summary.csv' = 'configuration_id,checkpoint_step,is_final,scope,metric,value'
}

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
$kBestExactSteps = @{ L19_SEED_1 = 32; L19_SEED_2 = 128; L19_SEED_4 = 80; L18_SEED_2_CONTROL = 160 }
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

function RequireHeader([string]$Name, [string]$Expected) {
    $path = SourcePath $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "source file missing: $Name" }
    $line = Get-Content -LiteralPath $path -TotalCount 1
    if ([string]::IsNullOrWhiteSpace($line)) { Fail "source schema mismatch: $Name" }
    $actual = @($line.Split(',') | ForEach-Object { $_.Trim('"') }) -join ','
    if ($actual -cne $Expected) { Fail "source schema mismatch: $Name" }
}
function Write-CanonicalFixtureCsv([string]$Dir, [string]$Name, [string]$PartialHeader, [string[]]$Rows) {
    $partialColumns = @($PartialHeader.Split(','))
    $columns = @($sourceSchemas[$Name].Split(','))
    $canonicalRows = foreach ($row in $Rows) {
        $values = @($row.Split(','))
        if ($values.Count -ne $partialColumns.Count) { Fail "self-test fixture row shape mismatch: $Name" }
        $fields = @{}
        for ($i = 0; $i -lt $partialColumns.Count; $i++) { $fields[$partialColumns[$i]] = $values[$i] }
        ($columns | ForEach-Object { if ($fields.ContainsKey($_)) { $fields[$_] } else { '0' } }) -join ','
    }
    Write-FixtureCsv (Join-Path $Dir $Name) $sourceSchemas[$Name] @($canonicalRows)
}
function Set-FixtureField([string]$Dir, [string]$Name, [int]$Index, [string]$Field, [string]$Value) {
    $path = Join-Path $Dir $Name
    $rows = @(Import-Csv -LiteralPath $path)
    $rows[$Index].$Field = $Value
    $rows | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding utf8
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

function Get-RepName([int]$Depth, [int]$Rep) {
    if ($Rep -eq 0) { return 'L0_EMBEDDED' }
    if ($Rep -le $Depth) { return "L${Rep}_BLOCK_OUT" }
    if ($Rep -eq ($Depth + 1)) { return 'PRE_LN_FINAL' }
    return 'POST_LN_FINAL'
}

function New-ExpectedProbeCases() {
    # host_tests/readout_probe.cpp kSpecs/repsFor and readout_probe_lib.h kRepresentativeReps.
    foreach ($cfg in $allConfigs) {
        $depth = if ($cfg -eq 'L18_SEED_2_CONTROL') { 18 } else { 19 }
        foreach ($step in @($kPinned[$cfg].arSelected, $kBestExactSteps[$cfg], 320)) {
            $reps = if ($step -eq 320) { @(0..($depth + 2)) } else { @(0,4,8,12,16,$depth,($depth + 1),($depth + 2)) }
            foreach ($rep in $reps) {
                [pscustomobject]@{ configuration_id = $cfg; checkpoint_step = [int]$step; rep = [int]$rep; rep_name = (Get-RepName $depth $rep) }
            }
        }
    }
}

function Get-ProbeKey([object]$Row) {
    return "$($Row.configuration_id)|$([int]$Row.checkpoint_step)|$([int]$Row.rep)"
}

function Assert-ExactKeySet([string[]]$Actual, [string[]]$Expected, [string]$Label) {
    if ($Actual.Count -ne $Expected.Count) { Fail "$Label expected key set mismatch (row count)" }
    if (@($Actual | Sort-Object -Unique).Count -ne $Actual.Count) { Fail "$Label expected key set mismatch (duplicate key)" }
    if (($Actual | Sort-Object) -join "`n" -cne (($Expected | Sort-Object) -join "`n")) {
        Fail "$Label expected key set mismatch (missing or unexpected key)"
    }
    # Keep cardinalities explicit: 38 probes per L19 config, 37 for L18;
    # non-final checkpoints have eight reps, final has depth+3 reps.
    foreach ($cfg in $allConfigs) {
        if (@($Actual | Where-Object { $_.StartsWith("$cfg|") }).Count -ne
            @($Expected | Where-Object { $_.StartsWith("$cfg|") }).Count) {
            Fail "$Label configuration cardinality mismatch: $cfg"
        }
        foreach ($step in @($kPinned[$cfg].arSelected, $kBestExactSteps[$cfg], 320)) {
            if (@($Actual | Where-Object { $_.StartsWith("$cfg|$step|") }).Count -ne
                @($Expected | Where-Object { $_.StartsWith("$cfg|$step|") }).Count) {
                Fail "$Label checkpoint cardinality mismatch: $cfg step $step"
            }
        }
    }
}

function AssertSourceEvidence() {
    foreach ($name in $sourceFiles) { RequireHeader $name $sourceSchemas[$name] }
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
    $expectedCases = @(New-ExpectedProbeCases)
    if ($expectedCases.Count -ne 151) { Fail 'readout protocol probe count mismatch (expected 151)' }
    $expectedKeys = @($expectedCases | ForEach-Object { Get-ProbeKey $_ })
    $expectedProbes = $expectedCases.Count
    if ($script:ProbeSelectionRows.Count -ne $expectedProbes) { Fail "probe-selection row count mismatch (expected $expectedProbes)" }
    Assert-ExactKeySet @($script:ProbeSelectionRows | ForEach-Object { Get-ProbeKey $_ }) $expectedKeys 'probe-selection'
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
    Assert-ExactKeySet @($script:LayerCurveRows | ForEach-Object { Get-ProbeKey $_ }) $expectedKeys 'probe-layer-curve'
    foreach ($row in $script:LayerCurveRows) {
        foreach ($field in @('probe_dev_fr_token_exact', 'head_dev_fr_token_exact', 'probe_minus_head_fr',
            'probe_dev_tf_token_exact', 'probe_train_tf_token_exact', 'probe_cal_fr_token_exact')) {
            [void](ParseFinite $row.$field $field 'probe-layer-curve')
        }
    }

    $script:GridRows = @(Import-Csv -LiteralPath (SourcePath 'probe-training-grid.csv'))
    if ($script:GridRows.Count -ne ($expectedProbes * 81)) { Fail "probe-training-grid row count mismatch (expected $($expectedProbes * 81))" }
    $expectedGridKeys = @($expectedKeys | ForEach-Object { $key = $_; $kCadence | ForEach-Object { "$key|$_" } })
    Assert-ExactKeySet @($script:GridRows | ForEach-Object { "$(Get-ProbeKey $_)|$([int]$_.grid_step)" }) $expectedGridKeys 'probe-training-grid'
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
    $expectedMetricKeys = @($expectedKeys | ForEach-Object { "$($_)|TRAIN"; "$($_)|MARGIN_DEVELOPMENT_V1" })
    Assert-ExactKeySet @($script:RepMetricRows | ForEach-Object { "$(Get-ProbeKey $_)|$($_.dataset)" }) $expectedMetricKeys 'representation-metrics'
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

function Write-FixtureCsv([string]$Path, [string]$Header, [string[]]$Rows) {
    [IO.File]::WriteAllLines($Path, (@($Header) + $Rows), $utf8)
}
function ExpectSelfTestRejects([string]$Label, [string]$ExpectedMarker, [scriptblock]$Action) {
    # Every case starts with the exact positive fixture, including files removed by earlier cases.
    foreach ($file in @(Get-ChildItem -LiteralPath $script:FixtureInput -File)) {
        [IO.File]::Delete($file.FullName)
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $script:PristineSelfTestInput -File)) {
        [IO.File]::Copy($file.FullName, (Join-Path $script:FixtureInput $file.Name))
    }
    try { & $Action } catch {
        if ($_.Exception.Message.Contains($ExpectedMarker, [StringComparison]::Ordinal)) {
            Write-Host "self-test expected rejection: $Label [$ExpectedMarker]"
            return
        }
        Fail "self-test wrong rejection for ${Label}: $($_.Exception.Message) (expected $ExpectedMarker)"
    }
    Fail "self-test negative case did not fail: $Label"
}
function New-SyntheticSelfTestRoot() {
    $root = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\exporter-selftest-fixture-" + [Guid]::NewGuid().ToString('N'))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $root.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $input = Join-Path $root 'input'
    $output = Join-Path $root 'output'
    [void](New-Item -ItemType Directory -Path $input -Force)
    [void](New-Item -ItemType Directory -Path $output -Force)
    return [pscustomobject]@{ Root = $root; Input = $input; Output = $output }
}
function Write-DatasetUsageFixture([string]$Dir) {
    Write-FixtureCsv (Join-Path $Dir 'dataset-usage.csv') 'dataset,role,hash,rows' @(
        "TRAIN,probe,$kTrainHash,32",
        "MARGIN_CALIBRATION_V1,step_select,$kCalibrationHash,144",
        "MARGIN_DEVELOPMENT_V1,eval,$kDevelopmentHash,144",
        "AR_FINAL_HOLDOUT_V3,unopened,$kFinalHash,0")
}
function Write-ConfigurationFixture([string]$Dir, [string]$Header) {
    Write-FixtureCsv (Join-Path $Dir 'configuration.csv') $Header @(
        'L19_SEED_1,1,19,320,2,16',
        'L19_SEED_2,2,19,320,1,4',
        'L19_SEED_4,4,19,320,0,12',
        'L18_SEED_2_CONTROL,2,18,320,2,4')
}
function Write-DatasetAnchorFixture([string]$Dir) {
    Write-FixtureCsv (Join-Path $Dir 'dataset-anchors.csv') 'dataset,role,hash,rows' @(
        "TRAIN,probe,$kTrainHash,32",
        "MARGIN_CALIBRATION_V1,step_select,$kCalibrationHash,144",
        "MARGIN_DEVELOPMENT_V1,eval,$kDevelopmentHash,144",
        "AR_FINAL_HOLDOUT_V3,unopened,$kFinalHash,0")
}
function Write-TrajectoryAnchorFixture([string]$Dir) {
    $rows = @()
    foreach ($cfg in $allConfigs) {
        $p = $kPinned[$cfg]
        $rows += "$($cfg),AR_DEV_SELECTED,$($p.arSelected),token_exact,$($p.arSelTok),true"
        $rows += "$($cfg),AR_DEV_SELECTED,$($p.arSelected),sequence_exact,$($p.arSelSeq),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,token_exact,$($p.arFinalTok),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,sequence_exact,$($p.arFinalSeq),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,autoregressive_nll,$($p.arFinalNll),true"
    }
    Write-CanonicalFixtureCsv $Dir 'trajectory-anchors.csv' 'configuration_id,checkpoint,step,metric,value,match' $rows
}
if ($SelfTest) {
    $fixture = New-SyntheticSelfTestRoot
    try {
        $in = $fixture.Input
        Write-DatasetAnchorFixture $in
        Write-TrajectoryAnchorFixture $in
        $ckptSteps = @{ AR_DEV_SELECTED = 0; AR_DEV_FINAL = 320 }
        $baseline = @()
        foreach ($cfg in $allConfigs) {
            foreach ($cp in @('AR_DEV_SELECTED','AR_DEV_FINAL','BEST_TOKEN_EXACT')) {
                $baseline += "$($cfg),$cp,1,1,1,1,1,1,0.5,0.5,1.0,2.0,0.5,2.0"
            }
        }
        Write-CanonicalFixtureCsv $in 'baseline-current-head.csv' 'configuration_id,checkpoint,head_train_tf_token_exact,head_cal_tf_token_exact,head_dev_tf_token_exact,head_cal_fr_token_exact,head_cal_fr_sequence_exact,head_dev_fr_token_exact,head_dev_fr_sequence_exact,head_dev_fr_nll,head_dev_fr_median_survival,head_dev_fr_margin_q10,head_dev_tf_mean_rank,head_dev_tf_mean_nll' $baseline
        $expectedCases = @(New-ExpectedProbeCases)
        $probeRows = @()
        $curveRows = @()
        $gridRows = @()
        $repRows = @()
        foreach ($case in $expectedCases) {
            $key = "$($case.configuration_id),$($case.checkpoint_step),$($case.rep)"
            $probeRows += "$key,$($case.rep_name),true,0,1.0,0.5,2.0,0.5,0.4"
            $curveRows += "$key,$($case.rep_name),10,20,-10,12,12,8"
            foreach ($gridStep in $kCadence) {
                $sel = if ($gridStep -eq 0) { 'true' } else { 'false' }
                $gridRows += "$key,$gridStep,$sel,1.0,0.5"
            }
            foreach ($ds in @('TRAIN','MARGIN_DEVELOPMENT_V1')) {
                $repRows += "$key,$($case.rep_name),$ds,0.1,2.0,1.0,0.5,0.8,0.1"
            }
        }
        Write-CanonicalFixtureCsv $in 'probe-selection.csv' 'configuration_id,checkpoint_step,rep,rep_name,finite,selected_step,train_ce,cal_ce,dev_tf_mean_rank,dev_tf_mean_nll,dev_fr_nll' $probeRows
        Write-CanonicalFixtureCsv $in 'probe-layer-curve.csv' 'configuration_id,checkpoint_step,rep,rep_name,probe_dev_fr_token_exact,head_dev_fr_token_exact,probe_minus_head_fr,probe_dev_tf_token_exact,probe_train_tf_token_exact,probe_cal_fr_token_exact' $curveRows
        Write-CanonicalFixtureCsv $in 'probe-training-grid.csv' 'configuration_id,checkpoint_step,rep,grid_step,is_selected,train_ce,cal_ce' $gridRows
        $head = @()
        foreach ($cfg in $allConfigs) {
            foreach ($cand in @('A_WARM_START','B_REINIT','C_BIAS_ONLY')) {
                $head += "$($cfg),$($cand),true,true,1.0,0.5,2.0,0.5,0.5,0.1"
            }
        }
        Write-CanonicalFixtureCsv $in 'head-retraining.csv' 'configuration_id,candidate,finite,frozen_unchanged,train_ce,cal_ce,dev_tf_mean_nll,dev_fr_nll,dev_fr_median_survival,dev_fr_margin_q10' $head
        Write-CanonicalFixtureCsv $in 'representation-metrics.csv' 'configuration_id,checkpoint_step,rep,rep_name,dataset,eta2,effective_rank,norm_ratio,hidden_margin_midmedian,sign_agreement,alignment_cosine' $repRows
        $geom = @()
        foreach ($cfg in $allConfigs) {
            for ($j = 0; $j -lt 32; $j++) { $geom += "$($cfg),class_row_norm,$($j),1.0" }
            for ($j = 0; $j -lt 16; $j++) { $geom += "$($cfg),singular_value,$($j),1.0" }
            $geom += "$($cfg),effective_rank,0,8.0"
        }
        Write-CanonicalFixtureCsv $in 'head-geometry.csv' 'configuration_id,item,index,value' $geom
        Write-CanonicalFixtureCsv $in 'decision.csv' 'verdict,thresholds_fixed_before_results,reasons' @(
            'UNDETERMINED,true,synthetic-fixture-contract-check')
        $summary = @()
        foreach ($cfg in $allConfigs) {
            foreach ($cp in @('AR_DEV_SELECTED','AR_DEV_FINAL','BEST_TOKEN_EXACT')) {
                foreach ($sc in @('head_tf','head_fr')) {
                    foreach ($m in @('token_exact','sequence_exact','nll','median_survival','margin_q10','mean_rank','mean_nll')) {
                        $summary += "$($cfg),$($cp),$($sc),$($m),0.5"
                    }
                }
            }
        }
        # 4*3*2*7 = 168; need 84 -> use 2 metrics * 3 ckpt * 2 scope * 7? 4 configs * 3 * 7 = 84 with one scope? Use 4*3*2*3.5 no.
        # 84 = 12 * 7: 4 configs * 3 checkpoints * 7 metrics, scope only head_tf or head_fr mixed
        $summary = @()
        $scopes = @('head_tf','head_fr')
        $metrics = @('token_exact','sequence_exact','nll','median_survival','margin_q10','mean_rank','mean_nll')
        $si = 0
        foreach ($cfg in $allConfigs) {
            foreach ($cp in @('AR_DEV_SELECTED','AR_DEV_FINAL','BEST_TOKEN_EXACT')) {
                foreach ($m in $metrics) {
                    $summary += "$($cfg),$($cp),$($scopes[$si % 2]),$($m),0.5"
                    $si++
                }
            }
        }
        Write-CanonicalFixtureCsv $in 'summary.csv' 'configuration_id,checkpoint,scope,metric,value' $summary
        $script:FixtureInput = $in
        AssertSourceEvidence
        $script:PristineSelfTestInput = Join-Path $fixture.Root 'pristine'
        [void](New-Item -ItemType Directory -Path $script:PristineSelfTestInput)
        foreach ($file in @(Get-ChildItem -LiteralPath $in -File)) {
            [IO.File]::Copy($file.FullName, (Join-Path $script:PristineSelfTestInput $file.Name))
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'probe schema mismatch' 'source schema mismatch: probe-selection.csv' {
            $path = Join-Path $in 'probe-selection.csv'
            $lines = [IO.File]::ReadAllLines($path)
            $lines[0] = $lines[0].Replace('rep_name', 'wrong_rep_name')
            [IO.File]::WriteAllLines($path, $lines, $utf8)
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'duplicate probe key with unchanged row count' 'probe-selection expected key set mismatch (duplicate key)' {
            $rows = @(Import-Csv -LiteralPath (Join-Path $in 'probe-selection.csv'))
            Set-FixtureField $in 'probe-selection.csv' 0 'rep' $rows[1].rep
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'unexpected probe key with unchanged row count' 'probe-selection expected key set mismatch (missing or unexpected key)' {
            Set-FixtureField $in 'probe-selection.csv' 0 'rep' '999'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'layer-curve key differs from selection' 'probe-layer-curve expected key set mismatch (duplicate key)' {
            $rows = @(Import-Csv -LiteralPath (Join-Path $in 'probe-layer-curve.csv'))
            Set-FixtureField $in 'probe-layer-curve.csv' 0 'rep' $rows[1].rep
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'grid step duplicated for probe' 'probe-training-grid expected key set mismatch (duplicate key)' {
            Set-FixtureField $in 'probe-training-grid.csv' 0 'grid_step' '25'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'representation dataset duplicated for probe' 'representation-metrics expected key set mismatch (duplicate key)' {
            Set-FixtureField $in 'representation-metrics.csv' 0 'dataset' 'MARGIN_DEVELOPMENT_V1'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'tampered decision thresholds' 'decision thresholds_fixed_before_results must be true' {
            Set-FixtureField $in 'decision.csv' 0 'thresholds_fixed_before_results' 'false'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'trajectory match false' 'trajectory-anchors pinned anchor mismatch:' {
            $anchors = @(Import-Csv -LiteralPath (Join-Path $in 'trajectory-anchors.csv'))
            $anchors[0].match = 'false'
            $anchors | Export-Csv -LiteralPath (Join-Path $in 'trajectory-anchors.csv') -NoTypeInformation -Encoding utf8
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'tampered final hash' 'FINAL hash pin mismatch' {
            Write-FixtureCsv (Join-Path $in 'dataset-anchors.csv') 'dataset,role,hash,rows' @(
                "TRAIN,probe,$kTrainHash,32",
                "MARGIN_CALIBRATION_V1,step_select,$kCalibrationHash,144",
                "MARGIN_DEVELOPMENT_V1,eval,$kDevelopmentHash,144",
                "AR_FINAL_HOLDOUT_V3,unopened,fnv1a64:0000000000000000,0")
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'missing required file' 'dataset-anchors.csv' {
            Remove-Item -LiteralPath (Join-Path $in 'dataset-anchors.csv') -Force
            AssertSourceEvidence
        }
        Write-Host 'readout diagnosis public exporter self-test PASS (fixture-contained)'
    } finally {
        $script:FixtureInput = $null
        $script:PristineSelfTestInput = $null
        if (Test-Path -LiteralPath $fixture.Root) { [IO.Directory]::Delete($fixture.Root, $true) }
    }
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
