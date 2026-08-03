# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$ObjectiveRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-critical-margin-objective'),
    [string]$TrainingRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-critical-margin-training'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-critical-margin-stabilization-2026-08'),
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'configuration.csv', 'consistency.csv',
    'dataset-partitions.csv', 'dataset-overlap.csv', 'dataset-hashes.csv',
    'checkpoint-metrics.csv', 'checkpoint-selection.csv',
    'objective-scores.csv', 'objective-correlations.csv',
    'leave-one-seed-out.csv', 'development-gate.csv',
    'gradient-attribution.csv', 'decision.csv',
    'training-family-gate.csv', 'training-family-metrics.csv',
    'training-family-decision.csv', 'training-trajectory.csv'
)
$objectiveFiles = @(
    'configuration.csv', 'consistency.csv',
    'dataset-partitions.csv', 'dataset-overlap.csv', 'dataset-hashes.csv',
    'checkpoint-metrics.csv', 'checkpoint-selection.csv',
    'objective-scores.csv', 'objective-correlations.csv',
    'leave-one-seed-out.csv', 'development-gate.csv',
    'gradient-attribution.csv', 'decision.csv'
)
$trainingFiles = @(
    'training-family-gate.csv', 'training-family-metrics.csv',
    'training-family-decision.csv', 'training-trajectory.csv'
)
$l19Configs = @('L19_SEED_1','L19_SEED_2','L19_SEED_4')
$kCadence = @(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320)
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'

function Fail([string]$Message) { throw "critical margin stabilization public export: $Message" }

function Safe([string]$Text) {
    return $Text -notmatch '(?im)([a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll|bin|exe)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|hardware[_ -]?identifier|android_id|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|\badb\s+-s\s+|\b(?:raw[_ -]?(?:checkpoint|parameters?)|raw[_ -]?(?:adam|optimizer)(?:[_ -]?state)?|raw[_ -]?tensor(?:[_ -]?(?:dump|data))?)\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}

function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}

function CsvText($Rows) { return (($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n" }

function RequireUnderRepository([string]$Path, [string]$Purpose) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $repoRoot.TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { Fail "$Purpose must be inside the repository" }
    return $full
}

function RequireOutputRoot([string]$Path) {
    $full = RequireUnderRepository $Path 'OutputRoot'
    $docsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results\qnn-l19-critical-margin-stabilization-2026-08'))
    $buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if ($full -ne $docsRoot -and -not $full.StartsWith($buildRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Fail 'OutputRoot must be the designated public directory or a build-contained validation directory'
    }
    return $full
}

function SourcePath([string]$Name) {
    # In self-test mode all reads and negative-test tampering operate on the
    # disposable fixture copies; the live private reports are never touched.
    if ($script:FixtureInput) { return (Join-Path $script:FixtureInput $Name) }
    if ($trainingFiles -contains $Name) { return (Join-Path $TrainingRoot $Name) }
    return (Join-Path $ObjectiveRoot $Name)
}

function RequireHeader([string]$Name, [string[]]$Expected) {
    $path = SourcePath $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing source file: $Name" }
    $first = ([IO.File]::ReadAllLines($path))[0]
    $actual = @($first.Trim('"').Split('","'))
    if (($actual -join ',') -ne ($Expected -join ',')) { Fail "source schema mismatch: $Name" }
}

function RequireHeaderIn([string]$Root, [string]$Name, [string[]]$Expected) {
    $path = Join-Path $Root $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing source file: $Name" }
    $first = ([IO.File]::ReadAllLines($path))[0]
    $actual = @($first.Trim('"').Split('","'))
    if (($actual -join ',') -ne ($Expected -join ',')) { Fail "source schema mismatch: $Name" }
}

$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function ParseFinite([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float,
            $script:Invariant, [ref]$parsed)) { Fail "non-numeric value: $RowName.$Field" }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
        Fail "non-finite value: $RowName.$Field"
    }
    return $parsed
}

function ParseInt([string]$Value, [string]$Field, [string]$RowName) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer,
            $script:Invariant, [ref]$parsed)) { Fail "non-integer value: $RowName.$Field" }
    return $parsed
}

function ScalarConsistency([double]$A, [double]$B, [string]$RowName) {
    if ([math]::Abs($A - $B) -gt 1.0e-9 * [math]::Max(1.0, [math]::Abs($A))) {
        Fail "scalar mismatch: $RowName (recomputed=$B reported=$A)"
    }
}

function GetConfigurationIdentity([string]$Configuration) {
    switch ($Configuration) {
        'L19_SEED_1' { return [pscustomobject]@{ depth=19; seed=1; pinned=16 } }
        'L19_SEED_2' { return [pscustomobject]@{ depth=19; seed=2; pinned=4 } }
        'L19_SEED_4' { return [pscustomobject]@{ depth=19; seed=4; pinned=12 } }
        'L18_SEED_2_CONTROL' { return [pscustomobject]@{ depth=18; seed=2; pinned=4 } }
        default { Fail "unknown configuration identity: $Configuration" }
    }
}

$script:ConfigRows = $null
$script:ConsistencyRows = $null
$script:PartitionRows = $null
$script:OverlapRows = $null
$script:HashRows = $null
$script:CheckpointMetricRows = $null
$script:SelectionRows = $null
$script:ObjectiveScoreRows = $null
$script:CorrelationRows = $null
$script:LosoRows = $null
$script:GateRows = $null
$script:AttributionRows = $null
$script:FixtureInput = $null
$script:DecisionRow = $null
$script:FamilyGateRows = $null
$script:FamilyMetricRows = $null
$script:FamilyDecisionRow = $null

function GetMarginBundleConfigRows() {
    $anchorRoot = Join-Path $repoRoot 'docs\results\qnn-htp-l19-first-error-margin-2026-08'
    if (-not (Test-Path -LiteralPath $anchorRoot -PathType Container)) {
        Fail 'canonical margin bundle is missing; cannot anchor the regeneration evidence'
    }
    $path = Join-Path $anchorRoot 'configuration.csv'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail 'canonical margin configuration.csv is missing' }
    $rows = @(Import-Csv -LiteralPath $path)
    $expected = @('source','configuration_id','depth','seed','selected_step','pinned_selected_step','selected_step_matches_pinned','best_token_exact_step','best_token_exact_count','best_sequence_exact_step','best_sequence_exact_count','validation_selected_ar_nll','selected_nll','final_nll','nll_delta','selected_token_exact','final_token_exact','token_total','selected_sequence_exact','final_sequence_exact','sequence_total','easy_token_count','critical_token_count','easy_nll_gain','critical_nll_loss','swfc_count','swfc_median_margin','swfc_median_rank','swfc_median_entropy_selected','swfc_median_entropy_final','attribution_dominant')
    if (@(($rows | Select-Object -First 1).PSObject.Properties.Name) -join ',' -ne ($expected -join ',')) {
        Fail 'canonical margin configuration.csv schema mismatch'
    }
    return $rows
}

function AssertConfigurationEvidence() {
    if ($script:ConfigRows.Count -ne 4) { Fail 'configuration row count mismatch' }
    foreach ($row in $script:ConfigRows) {
        $identity = GetConfigurationIdentity $row.configuration_id
        if ($row.source -ne 'CPU_REFERENCE_REGENERATION') { Fail "configuration source mismatch: $($row.configuration_id)" }
        if ([int]$row.depth -ne $identity.depth -or [int]$row.seed -ne $identity.seed) { Fail "configuration identity mismatch: $($row.configuration_id)" }
        if ([int]$row.steps -ne 320 -or [int]$row.evaluation_step_count -ne 23) { Fail "configuration step mismatch: $($row.configuration_id)" }
        if ($row.calibration_partition -ne 'MARGIN_CALIBRATION_V1' -or $row.development_partition -ne 'MARGIN_DEVELOPMENT_V1') { Fail "configuration partition mismatch: $($row.configuration_id)" }
        if ($row.calibration_hash -ne $kCalibrationHash -or $row.development_hash -ne $kDevelopmentHash) { Fail "configuration hash mismatch: $($row.configuration_id)" }
        if ([int]$row.pinned_selected_step -ne $identity.pinned) { Fail "configuration pinned step mismatch: $($row.configuration_id)" }
        if ($row.selected_step_matches_pinned -ne 'true') { Fail "configuration pinned selection mismatch: $($row.configuration_id)" }
    }
}

function AssertConsistencyEvidence() {
    if ($script:ConsistencyRows.Count -ne 4) { Fail 'consistency row count mismatch' }
    $marginConfigs = GetMarginBundleConfigRows
    foreach ($row in $script:ConsistencyRows) {
        $identity = GetConfigurationIdentity $row.configuration_id
        if ([int]$row.pinned_selected_step -ne $identity.pinned) { Fail "consistency pinned step mismatch: $($row.configuration_id)" }
        if ([int]$row.regenerated_selected_step -ne $identity.pinned) { Fail "consistency regenerated step mismatch: $($row.configuration_id)" }
        if ($row.selected_step_matches_pinned -ne 'true' -or $row.anchors_match -ne 'true') { Fail "consistency anchor mismatch: $($row.configuration_id)" }
        if ($row.loss_parity_all_steps -ne 'true' -or $row.gradient_parity_all_steps -ne 'true') { Fail "consistency parity mismatch: $($row.configuration_id)" }
        $anchor = @($marginConfigs | Where-Object configuration_id -eq $row.configuration_id)[0]
        if ($null -eq $anchor) { Fail "missing margin anchor: $($row.configuration_id)" }
        if ([int]$row.selected_token_exact -ne [int]$anchor.selected_token_exact -or
            [int]$row.final_token_exact -ne [int]$anchor.final_token_exact -or
            [int]$row.selected_sequence_exact -ne [int]$anchor.selected_sequence_exact -or
            [int]$row.final_sequence_exact -ne [int]$anchor.final_sequence_exact) {
            Fail "consistency exact-count anchor mismatch: $($row.configuration_id)"
        }
    }
}

function AssertDatasetEvidence() {
    if ($script:PartitionRows.Count -ne 6) { Fail 'dataset-partitions row count mismatch' }
    $cal = @($script:PartitionRows | Where-Object partition -eq 'MARGIN_CALIBRATION_V1')[0]
    $dev = @($script:PartitionRows | Where-Object partition -eq 'MARGIN_DEVELOPMENT_V1')[0]
    if ($null -eq $cal -or $null -eq $dev) { Fail 'margin partitions missing' }
    if ($cal.hash -ne $kCalibrationHash -or $dev.hash -ne $kDevelopmentHash) { Fail 'margin partition hash mismatch' }
    if ([int]$cal.case_count -ne 24 -or [int]$cal.token_count -ne 144 -or
        [int]$dev.case_count -ne 24 -or [int]$dev.token_count -ne 144) { Fail 'margin partition case/token count mismatch' }
    if ($script:OverlapRows.Count -ne 9) { Fail 'dataset-overlap row count mismatch' }
    $pair = @($script:OverlapRows | Where-Object { $_.left_partition -eq 'MARGIN_CALIBRATION_V1' -and $_.right_partition -eq 'MARGIN_DEVELOPMENT_V1' })[0]
    if ($null -eq $pair) { Fail 'calibration/development overlap row missing' }
    foreach ($field in @('case_id_overlap','initial_prefix_overlap','full_sequence_overlap')) {
        if ([int]$pair.$field -ne 0) { Fail "overlap violation: $field" }
    }
    if ([int]$pair.unique_transition_overlap -ne 13 -or [int]$pair.transition_occurrence_multiset_overlap -ne 144) {
        Fail 'calibration/development transition overlap mismatch'
    }
    if ($script:HashRows.Count -ne 6) { Fail 'dataset-hashes row count mismatch' }
    $calHash = @($script:HashRows | Where-Object partition -eq 'MARGIN_CALIBRATION_V1')[0]
    $devHash = @($script:HashRows | Where-Object partition -eq 'MARGIN_DEVELOPMENT_V1')[0]
    if ($null -eq $calHash -or $calHash.hash -ne $kCalibrationHash) { Fail 'calibration hash pin mismatch' }
    if ($null -eq $devHash -or $devHash.hash -ne $kDevelopmentHash) { Fail 'development hash pin mismatch' }
}

function AssertObjectiveEvidence() {
    if ($script:ObjectiveScoreRows.Count -ne 1104) { Fail 'objective-scores row count mismatch (expected 4x12x23)' }
    foreach ($row in $script:ObjectiveScoreRows) {
        if ($row.finite -ne 'true') { Fail "non-finite objective score: $($row.configuration_id) $($row.objective) step $($row.step)" }
        [void](ParseFinite $row.score 'score' 'objective-scores')
        if ($kCadence -notcontains [int]$row.step) { Fail "objective score off-cadence step: $($row.step)" }
    }
    $objectiveCount = @($script:ObjectiveScoreRows | Select-Object -ExpandProperty objective -Unique).Count
    if ($objectiveCount -ne 12) { Fail "objective count mismatch: $objectiveCount" }
    foreach ($config in @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')) {
        $steps = @($script:ObjectiveScoreRows | Where-Object configuration_id -eq $config | Select-Object -ExpandProperty step -Unique)
        if ($steps.Count -ne 23) { Fail "objective step count mismatch: $config" }
    }
    if ($script:CorrelationRows.Count -ne 48) { Fail 'objective-correlations row count mismatch (expected 4x12)' }
    foreach ($row in $script:CorrelationRows) {
        [void](ParseFinite $row.spearman_objective_vs_development_token_exact 'spearman' 'objective-correlations')
    }
    if ($script:SelectionRows.Count -ne 48) { Fail 'checkpoint-selection row count mismatch (expected 4x12)' }
    foreach ($row in $script:SelectionRows) {
        if ($kCadence -notcontains [int]$row.selected_step) { Fail "checkpoint selection off-cadence step: $($row.selected_step)" }
    }
}

function AssertLosoAndGateEvidence() {
    if ($script:LosoRows.Count -ne 3) { Fail 'leave-one-seed-out row count mismatch' }
    foreach ($row in $script:LosoRows) {
        if ($row.chosen_objective -ne 'MARGIN_DEFICIT_MEAN_D0') { Fail "loso chosen objective mismatch: seed $($row.held_out_seed)" }
        if ($row.collapse_free -ne 'false' -or $row.finite -ne 'true') { Fail "loso outcome mismatch: seed $($row.held_out_seed)" }
    }
    if ($script:GateRows.Count -ne 12) { Fail 'development-gate row count mismatch' }
    foreach ($row in $script:GateRows) {
        if ($row.finite -ne 'true' -or $row.gate_pass -ne 'false' -or $row.pass -ne 'false') { Fail "development gate row mismatch: $($row.objective)" }
        if ([int]$row.supported_seeds -ne 0 -or [int]$row.stable_supported_seeds -ne 0) { Fail "development gate seed support mismatch: $($row.objective)" }
        if ($row.loso_collapse_free -ne 'false') { Fail "development gate loso mismatch: $($row.objective)" }
    }
}

function AssertAttributionEvidence() {
    if ($script:AttributionRows.Count -ne 1280) { Fail 'gradient-attribution row count mismatch (expected 4x320)' }
    foreach ($row in $script:AttributionRows) {
        [void](ParseFinite $row.loss 'loss' 'gradient-attribution')
        if ($row.loss_parity -ne 'true' -or $row.gradient_parity -ne 'true') {
            Fail "attribution parity mismatch: $($row.configuration_id) step $($row.step)"
        }
        if ([int]$row.step -lt 1 -or [int]$row.step -gt 320) { Fail "attribution step out of range: $($row.step)" }
    }
    foreach ($config in @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')) {
        $count = @($script:AttributionRows | Where-Object configuration_id -eq $config).Count
        if ($count -ne 320) { Fail "attribution row count mismatch: $config" }
    }
}

function AssertDecisionEvidence() {
    if ($null -eq $script:DecisionRow) { Fail 'decision row missing' }
    if ([int]$script:DecisionRow.passing_variant_count -ne 0) { Fail 'decision passing variant count mismatch' }
    if ($script:DecisionRow.best_variant -ne 'MARGIN_DEFICIT_MEAN_D0') { Fail 'decision best variant mismatch' }
    if ($script:DecisionRow.development_gate -ne 'REJECT' -or $script:DecisionRow.loso_collapse_free -ne 'REJECT') { Fail 'decision gate mismatch' }
    if ($script:DecisionRow.gradient_attribution -ne 'OUTPUT_HEAD_RANKING_DRIFT') { Fail 'decision attribution mismatch' }
    if ($script:DecisionRow.recommended_training_family -ne 'PAIRWISE_MARGIN_CE_V1;SEQUENCE_WORST_MARGIN_CE_V1') { Fail 'decision recommended family mismatch' }
    if ($script:DecisionRow.recommended_training_family_evidence -ne 'GRADIENT_ATTRIBUTION') { Fail 'decision family evidence mismatch' }
    if ($script:DecisionRow.checkpoint_objective_conclusion -ne 'CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT') { Fail 'decision conclusion mismatch' }
}

function AssertTrainingEvidence() {
    if ($script:FamilyGateRows.Count -ne 2) { Fail 'training-family-gate row count mismatch' }
    if ($script:FamilyMetricRows.Count -ne 8) { Fail 'training-family-metrics row count mismatch' }
    $pairwise = @($script:FamilyGateRows | Where-Object family_id -eq 'PAIRWISE_MARGIN_CE_V1')[0]
    $sequence = @($script:FamilyGateRows | Where-Object family_id -eq 'SEQUENCE_WORST_MARGIN_CE_V1')[0]
    if ($null -eq $pairwise -or $null -eq $sequence) { Fail 'training family gate rows incomplete' }
    if ($pairwise.finite -ne 'true' -or $pairwise.improved_seeds_count -ne '2' -or
        $pairwise.control_nonworse -ne 'false' -or $pairwise.margin_improved -ne 'true' -or
        $pairwise.stability_pass -ne 'true' -or $pairwise.pass -ne 'false') {
        Fail 'pairwise family gate mismatch'
    }
    if ($sequence.finite -ne 'false' -or $sequence.margin_improved -ne 'false' -or
        $sequence.stability_pass -ne 'false' -or $sequence.pass -ne 'false' -or
        $sequence.pooled_ltm_delta -ne 'NOT_FINITE') {
        Fail 'sequence-worst family gate mismatch'
    }
    # Cross-check pooled token/sequence deltas against the per-run metrics.
    foreach ($gate in $script:FamilyGateRows) {
        $runs = @($script:FamilyMetricRows | Where-Object { $_.family_id -eq $gate.family_id -and $_.configuration_id -in $l19Configs })
        $sumToken = 0; $sumSeq = 0; $sumLtm = 0.0; $ltmCount = 0
        foreach ($run in $runs) {
            # Token/sequence deltas are integers and are pooled over the L19
            # runs regardless of the run's finite flag (mirrors the probe).
            $sumToken += [int]$run.delta_token_exact_vs_baseline
            $sumSeq += [int]$run.delta_sequence_exact_vs_baseline
            if ($run.delta_ltm_vs_baseline -ne 'NOT_FINITE') {
                $sumLtm += ParseFinite $run.delta_ltm_vs_baseline 'delta_ltm_vs_baseline' "training-family-metrics/$($run.family_id)"
                $ltmCount++
            }
        }
        if ([int]$gate.pooled_token_delta -ne $sumToken) { Fail "pooled token delta mismatch: $($gate.family_id)" }
        if ([int]$gate.pooled_sequence_delta -ne $sumSeq) { Fail "pooled sequence delta mismatch: $($gate.family_id)" }
        if ($gate.pooled_ltm_delta -ne 'NOT_FINITE') {
            ScalarConsistency (ParseFinite $gate.pooled_ltm_delta 'pooled_ltm_delta' "gate/$($gate.family_id)") ($sumLtm / [double]$ltmCount) "gate/$($gate.family_id).pooled_ltm_delta"
        }
        $expectedPass = ($gate.finite -eq 'true' -and [int]$gate.improved_seeds_count -ge 2 -and
            $gate.control_nonworse -eq 'true' -and $gate.margin_improved -eq 'true' -and
            $gate.stability_pass -eq 'true')
        if (($gate.pass -eq 'true') -ne $expectedPass) { Fail "family gate pass mismatch: $($gate.family_id)" }
    }
    foreach ($run in $script:FamilyMetricRows) {
        if ([int]$run.last_parity_step -ne 320) { Fail "parity cadence mismatch: $($run.family_id) $($run.configuration_id)" }
        if ($run.family_parameter -ne 'delta=0.5,lambda=1.0' -and $run.family_parameter -ne 'tau=0.5,lambda=1.0') { Fail "family parameter mismatch: $($run.family_id)" }
    }
    if ($null -eq $script:FamilyDecisionRow) { Fail 'training-family-decision row missing' }
    if ([int]$script:FamilyDecisionRow.steps -ne 320 -or [int]$script:FamilyDecisionRow.passing_family_count -ne 0) { Fail 'training decision counts mismatch' }
    if ($script:FamilyDecisionRow.best_family -ne 'NONE' -or
        $script:FamilyDecisionRow.training_development_gate -ne 'REJECT' -or
        $script:FamilyDecisionRow.decision -ne 'NO_TRAINING_FAMILY_ACCEPTED') { Fail 'training decision mismatch' }
}

function AssertSourceEvidence() {
    foreach ($name in $objectiveFiles + $trainingFiles) {
        if (-not (Test-Path -LiteralPath (SourcePath $name) -PathType Leaf)) { Fail "missing source file: $name" }
    }
    $script:ConfigRows = @(Import-Csv -LiteralPath (SourcePath 'configuration.csv'))
    $script:ConsistencyRows = @(Import-Csv -LiteralPath (SourcePath 'consistency.csv'))
    $script:PartitionRows = @(Import-Csv -LiteralPath (SourcePath 'dataset-partitions.csv'))
    $script:OverlapRows = @(Import-Csv -LiteralPath (SourcePath 'dataset-overlap.csv'))
    $script:HashRows = @(Import-Csv -LiteralPath (SourcePath 'dataset-hashes.csv'))
    $script:CheckpointMetricRows = @(Import-Csv -LiteralPath (SourcePath 'checkpoint-metrics.csv'))
    $script:SelectionRows = @(Import-Csv -LiteralPath (SourcePath 'checkpoint-selection.csv'))
    $script:ObjectiveScoreRows = @(Import-Csv -LiteralPath (SourcePath 'objective-scores.csv'))
    $script:CorrelationRows = @(Import-Csv -LiteralPath (SourcePath 'objective-correlations.csv'))
    $script:LosoRows = @(Import-Csv -LiteralPath (SourcePath 'leave-one-seed-out.csv'))
    $script:GateRows = @(Import-Csv -LiteralPath (SourcePath 'development-gate.csv'))
    $script:AttributionRows = @(Import-Csv -LiteralPath (SourcePath 'gradient-attribution.csv'))
    $script:DecisionRow = @(Import-Csv -LiteralPath (SourcePath 'decision.csv'))[0]
    $script:FamilyGateRows = @(Import-Csv -LiteralPath (SourcePath 'training-family-gate.csv'))
    $script:FamilyMetricRows = @(Import-Csv -LiteralPath (SourcePath 'training-family-metrics.csv'))
    $script:FamilyDecisionRow = @(Import-Csv -LiteralPath (SourcePath 'training-family-decision.csv'))[0]
    AssertConfigurationEvidence
    AssertConsistencyEvidence
    AssertDatasetEvidence
    AssertObjectiveEvidence
    AssertLosoAndGateEvidence
    AssertAttributionEvidence
    AssertDecisionEvidence
    AssertTrainingEvidence
    # Checkpoint metrics: 23 cadence steps x 2 partitions x 4 configs.
    if ($script:CheckpointMetricRows.Count -ne 184) { Fail 'checkpoint-metrics row count mismatch (expected 4x2x23)' }
    foreach ($row in $script:CheckpointMetricRows) {
        if ($kCadence -notcontains [int]$row.step) { Fail "checkpoint-metrics off-cadence step: $($row.step)" }
        if ($row.partition -notin @('MARGIN_CALIBRATION_V1','MARGIN_DEVELOPMENT_V1')) { Fail "checkpoint-metrics partition mismatch: $($row.partition)" }
        if ($row.all_finite -ne 'true') { Fail "checkpoint-metrics non-finite row: $($row.configuration_id) $($row.partition) $($row.step)" }
    }
    $trajectoryRows = @(Import-Csv -LiteralPath (SourcePath 'training-trajectory.csv'))
    if ($trajectoryRows.Count -ne 384) { Fail 'training-trajectory row count mismatch (expected 8 runs x 2 partitions x 24 cadence rows)' }
    $pairwiseTrajectory = @($trajectoryRows | Where-Object family_id -eq 'PAIRWISE_MARGIN_CE_V1')
    $sequenceTrajectory = @($trajectoryRows | Where-Object family_id -eq 'SEQUENCE_WORST_MARGIN_CE_V1')
    if ($pairwiseTrajectory.Count -ne 192 -or $sequenceTrajectory.Count -ne 192) { Fail 'training-trajectory family row count mismatch' }
    foreach ($row in $pairwiseTrajectory) {
        if ($row.all_finite -ne 'true') { Fail "training-trajectory pairwise non-finite row: $($row.family_id) $($row.configuration_id) $($row.partition) $($row.step)" }
    }
    if (@($sequenceTrajectory | Where-Object all_finite -ne 'true').Count -lt 24) {
        Fail 'training-trajectory sequence divergence rows missing (expected non-finite rows for the diverged family)'
    }
    foreach ($row in $trajectoryRows) {
        if ($row.partition -notin @('MARGIN_CALIBRATION_V1','MARGIN_DEVELOPMENT_V1')) { Fail "training-trajectory partition mismatch: $($row.partition)" }
        if ($row.all_finite -eq 'true') {
            [void](ParseFinite $row.autoregressive_nll 'autoregressive_nll' 'training-trajectory')
            [void](ParseFinite $row.lower_tail_margin_q10 'lower_tail_margin_q10' 'training-trajectory')
        }
    }
}

function NewSelfTestFixture() {
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\critical-margin-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $fixtureRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $fixtureInput = Join-Path $fixtureRoot 'input'
    $fixtureOutput = Join-Path $fixtureRoot 'output'
    [void](New-Item -ItemType Directory -Path $fixtureInput -Force)
    [void](New-Item -ItemType Directory -Path $fixtureOutput -Force)
    foreach ($name in $objectiveFiles) {
        $source = Join-Path $ObjectiveRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture objective source missing: $name" }
        [IO.File]::Copy($source, (Join-Path $fixtureInput $name), $true)
    }
    foreach ($name in $trainingFiles) {
        $source = Join-Path $TrainingRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture training source missing: $name" }
        [IO.File]::Copy($source, (Join-Path $fixtureInput $name), $true)
    }
    return [pscustomobject]@{ Root=$fixtureRoot; Input=$fixtureInput; Output=$fixtureOutput }
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
    if ($manifest.schema -ne 'CRITICAL_MARGIN_STABILIZATION_V1' -or $manifest.schema_version -ne 1 -or
        $manifest.result_classification -ne 'CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT') { Fail 'manifest decision mismatch' }
    if ($manifest.passing_variant_count -ne 0 -or $manifest.accepted_training_family -ne $null -or
        $manifest.training_family_decision -ne 'NO_TRAINING_FAMILY_ACCEPTED') { Fail 'manifest training decision mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) {
        Fail 'manifest run accounting mismatch'
    }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}

function NewReadme() {
    $format = { param($value) ([double]$value).ToString('0.######', $script:Invariant) }
    $consensus = @{}
    foreach ($config in @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')) {
        $consensus[$config] = @($script:ConsistencyRows | Where-Object configuration_id -eq $config)[0]
    }
    $pairwise = @($script:FamilyGateRows | Where-Object family_id -eq 'PAIRWISE_MARGIN_CE_V1')[0]
    $sequence = @($script:FamilyGateRows | Where-Object family_id -eq 'SEQUENCE_WORST_MARGIN_CE_V1')[0]
    $rowText = {
        param($c)
        "$($c.selected_token_exact)/144 ($($c.selected_sequence_exact)/24 seq) at step $($c.pinned_selected_step) -> $($c.final_token_exact)/144 ($($c.final_sequence_exact)/24 seq) at step 320"
    }
@"
# Critical-margin stabilization objective, August 2026

This bundle is the follow-up to the first-error/margin decomposition
(docs/results/qnn-htp-l19-first-error-margin-2026-08), whose conclusion was
`CRITICAL_TOKEN_MARGIN_LOSS`. The hypothesis prior carried into this
investigation is that a checkpoint-selection or training objective that
maximizes the target-vs-competitor logit margin of critical (hard-negative)
tokens can stabilize generation quality. Everything here is a host-only
deterministic CPU replay (fixed QAIRT-free C++ reference) over the pinned
MARGIN_CALIBRATION_V1 (`$($kCalibrationHash)`, 24 cases) and
MARGIN_DEVELOPMENT_V1 (`$($kDevelopmentHash)`, 24 cases) partitions; no
device or HTP run contributed data.

## Checkpoint-selection objectives (12 variants, all REJECT)

All four reference runs were regenerated from the CPU reference in this goal
and reproduce the canonical margin bundle exactly (consistency.csv): pinned
selected steps 16/4/12/4, selected and final exact counts identical to the
canonical configuration.csv, and bitwise loss and gradient parity across all
320 steps of every run.

| Configuration | Regenerated trajectory |
| --- | --- |
| L19 seed 1 | $(& $rowText $consensus['L19_SEED_1']) |
| L19 seed 2 | $(& $rowText $consensus['L19_SEED_2']) |
| L19 seed 4 | $(& $rowText $consensus['L19_SEED_4']) |
| L18 control | $(& $rowText $consensus['L18_SEED_2_CONTROL']) |

Twelve checkpoint-selection objectives were scored on the 23-step cadence
(0..320) per configuration (objective-scores.csv) and cross-validated by
leave-one-seed-out (leave-one-seed-out.csv). Objective/quality correlations
reach 0.95 (LOWER_TAIL_MARGIN_Q20 vs development token exact on seed 4); the
best-ranked variant by the preregistered pooled-evidence comparator is
MARGIN_DEFICIT_MEAN_D0 (Spearman 0.83 vs development token exact on seed 1).
Every objective fails the preregistered development gate in
development-gate.csv: no variant reaches a
single supported seed, pooled token/sequence exact are non-worse in no
variant, the L18 control is worse, first-error median survival does not
improve, and every LOSO fold collapses (mean token delta -59, collapse-free
false in all three folds). The best variant MARGIN_DEFICIT_MEAN_D0 selects an
early low-exact checkpoint (51 tokens below the trajectory-best checkpoint and
43 below the final checkpoint on seed 1); margin quality and final-checkpoint
quality do not co-select.

Gradient attribution (gradient-attribution.csv, 4x320 steps, per-step loss and
gradient parity true) classifies the driver of the margin/quality decoupling
as `OUTPUT_HEAD_RANKING_DRIFT`: the output-projection gradient share rises
while the critical-token share of the loss falls below the
critical-underweight threshold. Decision: `CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT`.

## Training families (2 preregistered families, 12 runs, both REJECT)

Two margin-aware loss families were implemented host-only as a verbatim copy
of the CPU training reference (bitwise parity of the backward pass with the
CE dLogits is enforced at cadence 32 in every run; last parity step 320 in
all runs):

- PAIRWISE_MARGIN_CE_V1 (delta=0.5, lambda=1.0): CE plus a pairwise hinge on
  deficient target-vs-competitor margins.
- SEQUENCE_WORST_MARGIN_CE_V1 (tau=0.5, lambda=1.0): CE plus
  lambda*tau*logsumexp(-margin/tau) over the sequence's worst margin.

Micro smoke (4 runs, 64 steps) confirmed mechanics (finite, margin improving,
parity holds). The full gate (8 runs, 320 steps) rejects both families
(training-family-gate.csv, training-family-decision.csv):

| Family | Finite | Improved seeds (of 3) | Control non-worse | Margin improved | Stability | Pooled token delta | Pass |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| PAIRWISE_MARGIN_CE_V1 | true | 2 | false | true | true | $($pairwise.pooled_token_delta) | false |
| SEQUENCE_WORST_MARGIN_CE_V1 | false | 1 | true | false | false | $($sequence.pooled_token_delta) | false |

PAIRWISE improves two of three L19 seeds (30->69, more than doubling, and
46->79) with margins and NLL improving, but regresses seed 2 (63->58) and
the L18 control (65->62), failing the preregistered control-non-worse and
2-of-3-seeds conditions. SEQUENCE_WORST at tau=0.5 is unstable: the worst-margin
term keeps pushing easy train margins and diverges the dev NLL (NOT_FINITE in
all four runs). Conclusion: `NO_TRAINING_FAMILY_ACCEPTED`; the canonical
AR-selected checkpoints remain the final candidate and the final holdout
partition stays unopened (no candidate change, so no HTP smoke or formal run
was scheduled in this goal).

All CSVs here are derived metrics only (exact counts, NLL, margins, gradient
norms, gate decisions); raw checkpoints, parameter payloads, device
identifiers, endpoints, paths, and log streams are excluded. Report roots and
commands for reproduction: scripts/run_critical_margin_objective_benchmark.ps1
(objective probe, -Train, -Micro) and scripts/export_public_qnn_l19_critical_margin_results.ps1.
"@
}

$selfTestContext = $null
$selfTestPublicSnapshot = @{}
if ($SelfTest) {
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-l19-critical-margin-stabilization-2026-08'
    foreach ($name in $allowed) {
        $snapshotPath = Join-Path $publicSnapshotRoot $name
        if (Test-Path -LiteralPath $snapshotPath -PathType Leaf) {
            $selfTestPublicSnapshot[$name] = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash
        }
    }
    $selfTestContext = NewSelfTestFixture
    $script:FixtureInput = $selfTestContext.Input
    $OutputRoot = $selfTestContext.Output
}
$ObjectiveRoot = RequireUnderRepository $ObjectiveRoot 'ObjectiveRoot'
$TrainingRoot = RequireUnderRepository $TrainingRoot 'TrainingRoot'
$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $ObjectiveRoot -PathType Container)) { Fail 'ObjectiveRoot does not exist' }
if (-not (Test-Path -LiteralPath $TrainingRoot -PathType Container)) { Fail 'TrainingRoot does not exist' }
if (-not (Test-Path -LiteralPath $OutputRoot)) { [void](New-Item -ItemType Directory -Path $OutputRoot -Force) }

RequireHeaderIn $ObjectiveRoot 'configuration.csv' @('source','configuration_id','depth','seed','steps','evaluation_step_count','calibration_partition','development_partition','calibration_hash','development_hash','pinned_selected_step','selected_step_matches_pinned')
RequireHeaderIn $ObjectiveRoot 'consistency.csv' @('configuration_id','pinned_selected_step','regenerated_selected_step','selected_token_exact','final_token_exact','selected_sequence_exact','final_sequence_exact','selected_step_matches_pinned','anchors_match','loss_parity_all_steps','gradient_parity_all_steps')
RequireHeaderIn $ObjectiveRoot 'dataset-partitions.csv' @('partition','domain','hash','case_count','token_count')
RequireHeaderIn $ObjectiveRoot 'dataset-overlap.csv' @('left_partition','right_partition','case_id_overlap','initial_prefix_overlap','full_sequence_overlap','unique_transition_overlap','transition_occurrence_multiset_overlap')
RequireHeaderIn $ObjectiveRoot 'dataset-hashes.csv' @('partition','hash')
RequireHeaderIn $ObjectiveRoot 'checkpoint-metrics.csv' @('configuration_id','partition','step','token_exact','token_total','sequence_exact','sequence_total','autoregressive_nll','median_first_error_survival','lower_tail_margin_q10','all_finite')
RequireHeaderIn $ObjectiveRoot 'checkpoint-selection.csv' @('configuration_id','objective','priority','selected_step','score','calibration_token_exact','calibration_sequence_exact','development_token_exact','development_sequence_exact','development_nll','development_median_survival','development_lower_tail_margin_q10','delta_token_exact_vs_final','delta_sequence_exact_vs_final')
RequireHeaderIn $ObjectiveRoot 'objective-scores.csv' @('configuration_id','objective','priority','step','score','finite')
RequireHeaderIn $ObjectiveRoot 'objective-correlations.csv' @('configuration_id','objective','spearman_objective_vs_development_token_exact','spearman_objective_vs_development_sequence_exact','spearman_objective_vs_development_first_error_survival','spearman_objective_vs_calibration_token_exact','selection_regret_token_exact','selection_regret_sequence_exact','near_tie_step_count','delta_token_exact_vs_final','delta_sequence_exact_vs_final')
RequireHeaderIn $ObjectiveRoot 'leave-one-seed-out.csv' @('held_out_seed','chosen_objective','chosen_parameter','chosen_priority','selected_step','token_exact','token_total','sequence_exact','sequence_total','delta_token_exact_vs_final','delta_sequence_exact_vs_final','collapse_free','finite')
RequireHeaderIn $ObjectiveRoot 'development-gate.csv' @('objective','priority','finite','seed2_strict','pooled_token_nonworse','pooled_sequence_nonworse','control_nonworse','first_error_median_nonworse','supported_seeds','stable_supported_seeds','no_case_collapse','gate_pass','loso_collapse_free','loso_mean_token_delta','pass')
RequireHeaderIn $ObjectiveRoot 'gradient-attribution.csv' @('configuration_id','step','loss','accuracy','mean_target_margin','mean_target_rank','mean_target_nll','critical_token_share','critical_loss_share','easy_token_share','grad_norm_total','grad_norm_embedding','grad_norm_output_projection','grad_norm_layer_mean','grad_norm_first_layer','grad_norm_last_layer','grad_norm_attn_share','grad_norm_ffn_share','grad_norm_norm_share','grad_norm_depth_ratio','grad_norm_output_share','loss_parity','gradient_parity')
RequireHeaderIn $ObjectiveRoot 'decision.csv' @('hypothesis_prior','passing_variant_count','best_variant','development_gate','loso_collapse_free','gradient_attribution','recommended_training_family','recommended_training_family_evidence','checkpoint_objective_conclusion')
RequireHeaderIn $TrainingRoot 'training-family-gate.csv' @('family_id','finite','improved_seeds_count','control_nonworse','margin_improved','stability_pass','pooled_token_delta','pooled_sequence_delta','pooled_ltm_delta','pass')
RequireHeaderIn $TrainingRoot 'training-family-metrics.csv' @('family_id','configuration_id','seed','layers','steps','family_parameter','final_ar_dev_token_exact','final_ar_dev_sequence_exact','final_ar_dev_nll','stability_ar_dev_token_exact','stability_ar_dev_sequence_exact','final_margin_dev_lower_tail_margin_q10','final_margin_dev_median_survival','final_margin_calib_lower_tail_margin_q10','final_nll','final_margin_term','final_total_loss','final_mean_margin','final_critical_share','final_gradient_norm','last_parity_step','finite','delta_token_exact_vs_baseline','delta_sequence_exact_vs_baseline','delta_ltm_vs_baseline')
RequireHeaderIn $TrainingRoot 'training-family-decision.csv' @('steps','passing_family_count','best_family','training_development_gate','decision')
RequireHeaderIn $TrainingRoot 'training-trajectory.csv' @('family_id','configuration_id','partition','step','token_exact','token_total','sequence_exact','sequence_total','autoregressive_nll','median_first_error_survival','lower_tail_margin_q10','all_finite')

if ($SelfTest) {
    # Validate the fixture roots carry the required files.
    foreach ($name in $objectiveFiles + $trainingFiles) {
        if (-not (Test-Path -LiteralPath (SourcePath $name) -PathType Leaf)) { Fail "fixture missing source file: $name" }
    }
}

AssertSourceEvidence

foreach ($name in $objectiveFiles + $trainingFiles) { CopySafe $name }
WriteUtf8 'README.md' (NewReadme)

$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{name=$name;sha256=(GetSha256 $name)}
}
$manifest = [ordered]@{
    schema='CRITICAL_MARGIN_STABILIZATION_V1'; schema_version=1
    result_classification='CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT'
    hypothesis_prior='CRITICAL_TOKEN_MARGIN_LOSS'
    quality_configuration='T8/D16/FFN32/L19/H2'; control_configuration='T8/D16/FFN32/L18/H2'
    calibration_partition='MARGIN_CALIBRATION_V1'; development_partition='MARGIN_DEVELOPMENT_V1'
    calibration_hash=$kCalibrationHash; development_hash=$kDevelopmentHash
    calibration_cases=24; development_cases=24; prefix_tokens=8; rollout_tokens='4/8'
    transition_overlap_unique=13; transition_overlap_occurrence=144
    cpu_replay_runs=4; device_runs=0; htp_runs=0; final_holdout_opened=$false
    passing_variant_count=0; best_variant='MARGIN_DEFICIT_MEAN_D0'
    development_gate='REJECT'; loso_collapse_free='REJECT'
    gradient_attribution='OUTPUT_HEAD_RANKING_DRIFT'
    training_families=@('PAIRWISE_MARGIN_CE_V1','SEQUENCE_WORST_MARGIN_CE_V1')
    training_micro_runs=4; training_full_runs=8; training_steps=320
    accepted_training_family=$null; training_family_decision='NO_TRAINING_FAMILY_ACCEPTED'
    regression_anchor='docs/results/qnn-htp-l19-first-error-margin-2026-08'
    prohibited_payloads_published=$false; files=$manifestFiles
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle

if ($SelfTest) {
    if (-not (Safe 'calibration_hash="fnv1a64:71806d5bf19c090a"')) { Fail 'safe-text false rejection' }
    foreach ($unsafe in @('C:\\local\\report.txt','/tmp/report.txt','adb -s serial shell','android_id=1','raw_checkpoint=payload','raw tensor dump','app_private_path=files/x','model.apk')) {
        if (Safe $unsafe) { Fail "unsafe self-test rejected incorrectly: $unsafe" }
    }
    try { RequireOutputRoot ([IO.Path]::GetTempPath()); Fail 'outside output-root rejection failed' } catch { if ($_.Exception.Message -match 'outside output-root rejection failed') { throw } }
    $decisionPath = SourcePath 'decision.csv'
    $decisionOriginal = [IO.File]::ReadAllText($decisionPath)
    try {
        [IO.File]::WriteAllText($decisionPath, $decisionOriginal.Replace('CHECKPOINT_OBJECTIVE_DEVELOPMENT_REJECT', 'C:\local\mismatch'), $utf8)
        ExpectSelfTestFailure 'unsafe source content' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($decisionPath, $decisionOriginal, $utf8) }
    $hashPath = SourcePath 'dataset-hashes.csv'
    $hashOriginal = [IO.File]::ReadAllText($hashPath)
    try {
        [IO.File]::WriteAllText($hashPath, $hashOriginal.Replace($kCalibrationHash, 'fnv1a64:0000000000000000'), $utf8)
        ExpectSelfTestFailure 'hash pin rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($hashPath, $hashOriginal, $utf8) }
    $consistencyPath = SourcePath 'consistency.csv'
    $consistencyOriginal = [IO.File]::ReadAllText($consistencyPath)
    try {
        [IO.File]::WriteAllText($consistencyPath, $consistencyOriginal.Replace('"true","true","true"', '"true","false","true"', 1), $utf8)
        ExpectSelfTestFailure 'anchor rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($consistencyPath, $consistencyOriginal, $utf8) }
    $gatePath = SourcePath 'training-family-gate.csv'
    $gateOriginal = [IO.File]::ReadAllText($gatePath)
    try {
        [IO.File]::WriteAllText($gatePath, $gateOriginal.Replace('"PAIRWISE_MARGIN_CE_V1","true","2"', '"PAIRWISE_MARGIN_CE_V1","true","3"', 1), $utf8)
        ExpectSelfTestFailure 'training gate rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($gatePath, $gateOriginal, $utf8) }
    $scoresPath = SourcePath 'objective-scores.csv'
    $scoresOriginal = [IO.File]::ReadAllText($scoresPath)
    try {
        [IO.File]::WriteAllText($scoresPath, [regex]::Replace($scoresOriginal, '0\.31014478275978441', 'NaN', 1), $utf8)
        ExpectSelfTestFailure 'non-finite rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($scoresPath, $scoresOriginal, $utf8) }
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-l19-critical-margin-stabilization-2026-08'
    foreach ($name in $selfTestPublicSnapshot.Keys) {
        $after = (Get-FileHash -LiteralPath (Join-Path $publicSnapshotRoot $name) -Algorithm SHA256).Hash
        if ($after -ne $selfTestPublicSnapshot[$name]) { Fail "self-test modified public docs: $name" }
    }
    if ($selfTestContext -and (Test-Path -LiteralPath $selfTestContext.Root)) {
        Remove-Item -LiteralPath $selfTestContext.Root -Recurse -Force
    }
}

Write-Host "critical margin stabilization public export: PASS ($OutputRoot)"
