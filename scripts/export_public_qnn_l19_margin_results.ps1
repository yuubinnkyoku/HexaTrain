# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-l19-first-error-margin-2026-08'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-l19-first-error-margin-2026-08'),
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json', 'configuration.csv',
    'checkpoint-comparison.csv', 'token-buckets.csv',
    'first-error-summary.csv', 'margin-rank-summary.csv',
    'common-prefix-attribution.csv', 'seed-comparison.csv',
    'depth-control.csv', 'hypothesis-decision.csv',
    'next-objective-candidates.csv'
)
$copied = @(
    'configuration.csv', 'checkpoint-comparison.csv', 'token-buckets.csv',
    'first-error-summary.csv', 'margin-rank-summary.csv',
    'common-prefix-attribution.csv', 'seed-comparison.csv',
    'depth-control.csv', 'hypothesis-decision.csv',
    'next-objective-candidates.csv'
)
$privateTokenFiles = @(
    'margin-tokens-L19_SEED_1.csv', 'margin-tokens-L19_SEED_2.csv',
    'margin-tokens-L19_SEED_4.csv', 'margin-tokens-L18_SEED_2_CONTROL.csv'
)
$l19Configs = @('L19_SEED_1','L19_SEED_2','L19_SEED_4')
$cadence = @(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320)

function Fail([string]$Message) { throw "l19 margin public export: $Message" }

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
    $docsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results\qnn-htp-l19-first-error-margin-2026-08'))
    $buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if ($full -ne $docsRoot -and -not $full.StartsWith($buildRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Fail 'OutputRoot must be the designated public directory or a build-contained validation directory'
    }
    return $full
}

function RequireHeader([string]$Name, [string[]]$Expected) {
    $path = Join-Path $InputRoot $Name
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
$script:ComparisonRows = $null
$script:BucketRows = $null
$script:ErrorRows = $null
$script:RankRows = $null
$script:AttributionRows = $null
$script:SeedRows = $null
$script:DepthRows = $null
$script:HypothesisRows = $null
$script:ObjectiveRows = $null
$script:DevelopmentRows = $null
$script:ValidationRows = $null
$script:PartitionRows = $null
$script:DriftCaseCount = 0
$script:PooledSwfcCount = 0
$script:PooledSwfcMedian = 0.0
$script:PooledEasyFraction = 0.0
$script:PooledPercentGain = 0.0

function GetDevelopmentAnchorRows() {
    $anchorRoot = Join-Path $repoRoot 'docs\results\qnn-htp-autoregressive-validation-2026-08'
    if (-not (Test-Path -LiteralPath $anchorRoot -PathType Container)) {
        Fail 'canonical autoregressive bundle is missing; cannot anchor the regression evidence'
    }
    $developmentPath = Join-Path $anchorRoot 'ar-development-metrics.csv'
    $validationPath = Join-Path $anchorRoot 'ar-validation-metrics.csv'
    $partitionPath = Join-Path $anchorRoot 'dataset-partitions.csv'
    foreach ($name in @($developmentPath, $validationPath, $partitionPath)) {
        if (-not (Test-Path -LiteralPath $name -PathType Leaf)) { Fail "canonical anchor missing: $name" }
    }
    return [pscustomobject]@{
        Development=@(Import-Csv -LiteralPath $developmentPath)
        Validation=@(Import-Csv -LiteralPath $validationPath)
        Partitions=@(Import-Csv -LiteralPath $partitionPath)
    }
}

function AssertConfigurationEvidence() {
    if ($script:ConfigRows.Count -ne 4) { Fail 'configuration row count mismatch' }
    $expectedIds = @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')
    if ((($script:ConfigRows.configuration_id | Sort-Object) -join ',') -ne (($expectedIds | Sort-Object) -join ',')) {
        Fail 'configuration identity set mismatch'
    }
    foreach ($row in $script:ConfigRows) {
        $rowName = "configuration/$($row.configuration_id)"
        $identity = GetConfigurationIdentity $row.configuration_id
        if ($row.source -ne 'CPU_REFERENCE_REGENERATION' -or
            (ParseInt $row.depth 'depth' $rowName) -ne $identity.depth -or
            (ParseInt $row.seed 'seed' $rowName) -ne $identity.seed) { Fail "configuration identity mismatch: $rowName" }
        if ((ParseInt $row.selected_step 'selected_step' $rowName) -ne $identity.pinned -or
            $row.selected_step_matches_pinned -ne 'true' -or
            (ParseInt $row.pinned_selected_step 'pinned_selected_step' $rowName) -ne $identity.pinned) {
            Fail "configuration pinned step mismatch: $rowName"
        }
        $tokenTotal = ParseInt $row.token_total 'token_total' $rowName
        $sequenceTotal = ParseInt $row.sequence_total 'sequence_total' $rowName
        if ($tokenTotal -ne 144 -or $sequenceTotal -ne 24) { Fail "configuration denominator mismatch: $rowName" }
        $selectedNll = ParseFinite $row.selected_nll 'selected_nll' $rowName
        $finalNll = ParseFinite $row.final_nll 'final_nll' $rowName
        ScalarConsistency (ParseFinite $row.nll_delta 'nll_delta' $rowName) (($finalNll - $selectedNll) * $tokenTotal) "$rowName.nll_delta"
        [void](ParseFinite $row.validation_selected_ar_nll 'validation_selected_ar_nll' $rowName)
        foreach ($field in @('best_token_exact_step','best_sequence_exact_step')) {
            if ((ParseInt $row.$field $field $rowName) -notin $cadence) { Fail "configuration best-step cadence mismatch: $rowName.$field" }
        }
        $bestTokens = ParseInt $row.best_token_exact_count 'best_token_exact_count' $rowName
        if ($bestTokens -lt 0 -or $bestTokens -gt 144 -or
            (ParseInt $row.best_sequence_exact_count 'best_sequence_exact_count' $rowName) -gt 24) {
            Fail "configuration best exact counts out of range: $rowName"
        }
        if ((ParseInt $row.easy_token_count 'easy_token_count' $rowName) -ne 0 -or
            (ParseFinite $row.easy_nll_gain 'easy_nll_gain' $rowName) -ne 0.0) {
            Fail "configuration easy-token invariant mismatch: $rowName"
        }
        $critical = ParseInt $row.critical_token_count 'critical_token_count' $rowName
        $swfc = ParseInt $row.swfc_count 'swfc_count' $rowName
        if ($critical -lt $swfc -or $critical -gt $tokenTotal -or
            (ParseFinite $row.swfc_median_margin 'swfc_median_margin' $rowName) -ge 0.0) {
            Fail "configuration critical-token invariant mismatch: $rowName"
        }
        if ($row.attribution_dominant -ne 'MIXED') { Fail "configuration dominant attribution mismatch: $rowName" }
        $validationAnchorRow = @($script:ValidationRows | Where-Object {
            $_.configuration_id -eq $row.configuration_id -and $_.role -eq 'SELECTED' })
        $selectedDevRow = @($script:DevelopmentRows | Where-Object {
            $_.configuration_id -eq $row.configuration_id -and $_.role -eq 'SELECTED' })[0]
        $finalDevRow = @($script:DevelopmentRows | Where-Object {
            $_.configuration_id -eq $row.configuration_id -and $_.role -eq 'FINAL_STEP' })[0]
        if ($validationAnchorRow.Count -ne 1 -or $null -eq $selectedDevRow -or $null -eq $finalDevRow) {
            Fail "canonical anchor row missing: $rowName"
        }
        ScalarConsistency ([double]$row.validation_selected_ar_nll) ([double]$validationAnchorRow[0].ar_rollout_nll) "$rowName.validation"
        ScalarConsistency $selectedNll ([double]$selectedDevRow.ar_rollout_nll) "$rowName.dev-selected"
        ScalarConsistency $finalNll ([double]$finalDevRow.ar_rollout_nll) "$rowName.dev-final"
        if ((ParseInt $row.selected_token_exact 'selected_token_exact' $rowName) -ne [int]$selectedDevRow.token_exact -or
            (ParseInt $row.selected_sequence_exact 'selected_sequence_exact' $rowName) -ne [int]$selectedDevRow.sequence_exact -or
            (ParseInt $row.final_token_exact 'final_token_exact' $rowName) -ne [int]$finalDevRow.token_exact -or
            (ParseInt $row.final_sequence_exact 'final_sequence_exact' $rowName) -ne [int]$finalDevRow.sequence_exact) {
            Fail "configuration exact anchor mismatch: $rowName"
        }
    }
}

function AssertComparisonEvidence() {
    if ($script:ComparisonRows.Count -ne 8) { Fail 'checkpoint comparison row count mismatch' }
    foreach ($row in $script:ComparisonRows) {
        $rowName = "checkpoint-comparison/$($row.configuration_id)/$($row.role)"
        if ($row.role -notin @('SELECTED','FINAL_STEP')) { Fail "unknown comparison role: $rowName" }
        $config = @($script:ConfigRows | Where-Object configuration_id -eq $row.configuration_id)[0]
        $tokenTotal = ParseInt $row.token_total 'token_total' $rowName
        $sequenceTotal = ParseInt $row.sequence_total 'sequence_total' $rowName
        if ($tokenTotal -ne 144 -or $sequenceTotal -ne 24) { Fail "comparison denominator mismatch: $rowName" }
        if ([int]$row.no_error_cases + [int]$row.first_error_cases -ne 24) { Fail "comparison case totals mismatch: $rowName" }
        if ([int]$row.first_error_cases -gt 0 -and (ParseFinite $row.mean_first_error_position 'mean_first_error_position' $rowName) -lt 1.0) {
            Fail "comparison first-error position out of range: $rowName"
        }
        if ($row.role -eq 'SELECTED') {
            ScalarConsistency (ParseFinite $row.ar_nll 'ar_nll' $rowName) ([double]$config.selected_nll) $rowName
            if ([int]$row.step -ne [int]$config.selected_step -or
                [int]$row.token_exact -ne [int]$config.selected_token_exact -or
                [int]$row.sequence_exact -ne [int]$config.selected_sequence_exact) { Fail "comparison selected mismatch: $rowName" }
        } else {
            ScalarConsistency (ParseFinite $row.ar_nll 'ar_nll' $rowName) ([double]$config.final_nll) $rowName
            if ([int]$row.step -ne 320 -or
                [int]$row.token_exact -ne [int]$config.final_token_exact -or
                [int]$row.sequence_exact -ne [int]$config.final_sequence_exact) { Fail "comparison final mismatch: $rowName" }
        }
    }
}

function AssertBucketEvidence() {
    if ($script:BucketRows.Count -ne 16) { Fail 'token buckets row count mismatch (expected 4x4)' }
    $bucketNames = @('BOTH_CORRECT','SELECTED_CORRECT_FINAL_WRONG','SELECTED_WRONG_FINAL_CORRECT','BOTH_WRONG')
    foreach ($config in $script:ConfigRows.configuration_id) {
        $rows = @($script:BucketRows | Where-Object configuration_id -eq $config)
        $rowName = "token-buckets/$config"
        if ($rows.Count -ne 4) { Fail "token bucket set mismatch: $rowName" }
        $total = ($rows | ForEach-Object { ParseInt $_.token_count 'token_count' $rowName } | Measure-Object -Sum).Sum
        if ($total -ne 144) { Fail "token bucket total mismatch: $rowName" }
        $selectedErrors = 0; $finalErrors = 0
        foreach ($bucket in $bucketNames) {
            $row = @($rows | Where-Object bucket -eq $bucket)[0]
            $count = ParseInt $row.token_count 'token_count' $rowName
            ScalarConsistency (ParseFinite $row.percent_tokens 'percent_tokens' $rowName) ([double]$count * 100.0 / 144.0) "$rowName/$bucket.percent"
            if ($row.bucket -in @('BOTH_CORRECT','SELECTED_CORRECT_FINAL_WRONG')) {
                if ([int]$row.selected_exact_count -ne $count) { Fail "selected-exact count mismatch: $rowName/$bucket" }
            } elseif ([int]$row.selected_exact_count -ne 0) { Fail "selected-exact count mismatch: $rowName/$bucket" }
            if ($row.bucket -in @('BOTH_CORRECT','SELECTED_WRONG_FINAL_CORRECT')) {
                if ([int]$row.final_exact_count -ne $count) { Fail "final-exact count mismatch: $rowName/$bucket" }
            } elseif ([int]$row.final_exact_count -ne 0) { Fail "final-exact count mismatch: $rowName/$bucket" }
            $selectedMean = ParseFinite $row.selected_nll_mean 'selected_nll_mean' $rowName
            $finalMean = ParseFinite $row.final_nll_mean 'final_nll_mean' $rowName
            if ($bucket -eq 'BOTH_CORRECT' -and $selectedMean -le 0.0) { Fail "both-correct NLL invariant mismatch: $rowName" }
            if ($bucket -eq 'SELECTED_WRONG_FINAL_CORRECT' -and $finalMean -ge $selectedMean) { Fail "corrected-token NLL direction mismatch: $rowName/$bucket" }
            if ($bucket -eq 'SELECTED_CORRECT_FINAL_WRONG' -and $finalMean -le $selectedMean) { Fail "broken-token NLL direction mismatch: $rowName/$bucket" }
            $selectedErrors += [int]$row.first_error_positions_selected
            $finalErrors += [int]$row.first_error_positions_final
        }
        $errorRow = @($script:ErrorRows | Where-Object configuration_id -eq $config)[0]
        if ($selectedErrors -ne [int]$errorRow.selected_first_error_cases -or
            $finalErrors -ne [int]$errorRow.final_first_error_cases) {
            Fail "bucket first-error position totals mismatch: $config"
        }
    }
}

function AssertErrorEvidence() {
    if ($script:ErrorRows.Count -ne 4) { Fail 'first-error summary row count mismatch' }
    foreach ($row in $script:ErrorRows) {
        $rowName = "first-error-summary/$($row.configuration_id)"
        $selectedClasses = @('selected_no_error','selected_late_single_error','selected_error_with_recovery','selected_early_irreversible_divergence','selected_multiple_local_errors')
        $finalClasses = @('final_no_error','final_late_single_error','final_error_with_recovery','final_early_irreversible_divergence','final_multiple_local_errors')
        $selectedSum = ($selectedClasses | ForEach-Object { [int]$row.$_ } | Measure-Object -Sum).Sum
        $finalSum = ($finalClasses | ForEach-Object { [int]$row.$_ } | Measure-Object -Sum).Sum
        if ($selectedSum -ne 24 -or $finalSum -ne 24) { Fail "first-error class totals mismatch: $rowName" }
        if ([int]$row.selected_first_error_cases -ne (24 - [int]$row.selected_no_error) -or
            [int]$row.final_first_error_cases -ne (24 - [int]$row.final_no_error)) { Fail "first-error case totals mismatch: $rowName" }
        if ([int]$row.selected_first_error_cases -gt 0 -and [double]$row.selected_mean_first_error_position -lt 1.0) {
            Fail "selected mean first-error position out of range: $rowName"
        }
    }
    foreach ($comparisonRow in $script:ComparisonRows) {
        $summaryRow = @($script:ErrorRows | Where-Object configuration_id -eq $comparisonRow.configuration_id)[0]
        $rowName = "checkpoint-comparison/$($comparisonRow.configuration_id)/$($comparisonRow.role)"
        $expectedMean = if ($comparisonRow.role -eq 'SELECTED') { [double]$summaryRow.selected_mean_first_error_position } else { [double]$summaryRow.final_mean_first_error_position }
        ScalarConsistency (ParseFinite $comparisonRow.mean_first_error_position 'mean_first_error_position' $rowName) $expectedMean $rowName
    }
}

function AssertRankEvidence() {
    if ($script:RankRows.Count -ne 4) { Fail 'margin-rank summary row count mismatch' }
    foreach ($row in $script:RankRows) {
        $rowName = "margin-rank-summary/$($row.configuration_id)"
        if ([int]$row.rank1_unique_total -ne 144) { Fail "rank denominator mismatch: $rowName" }
        $rs = [int]$row.rank1_unique_selected; $rf = [int]$row.rank1_unique_final
        $t2s = [int]$row.top2_inclusion_selected; $t2f = [int]$row.top2_inclusion_final
        $t3s = [int]$row.top3_inclusion_selected; $t3f = [int]$row.top3_inclusion_final
        if ($rs -lt 0 -or $rs -gt 144 -or $rf -lt 0 -or $rf -gt 144 -or $rs -gt $t2s -or $t2s -gt $t3s -or
            $rf -gt $t2f -or $t2f -gt $t3f -or $t3s -gt 144 -or $t3f -gt 144) { Fail "rank inclusion monotonicity mismatch: $rowName" }
        $selectedEntropy = ParseFinite $row.mean_entropy_selected 'mean_entropy_selected' $rowName
        $finalEntropy = ParseFinite $row.mean_entropy_final 'mean_entropy_final' $rowName
        $finalMargin = ParseFinite $row.mean_expected_minus_top1_margin_final 'mean_expected_minus_top1_margin_final' $rowName
        $selectedMargin = ParseFinite $row.mean_expected_minus_top1_margin_selected 'mean_expected_minus_top1_margin_selected' $rowName
        $finalTopMargin = ParseFinite $row.mean_top1_minus_top2_margin_final 'mean_top1_minus_top2_margin_final' $rowName
        $selectedTopMargin = ParseFinite $row.mean_top1_minus_top2_margin_selected 'mean_top1_minus_top2_margin_selected' $rowName
        if ($finalEntropy -ge $selectedEntropy) { Fail "final-confidence invariant mismatch: $rowName" }
        if ($finalMargin -gt $selectedMargin -or $finalTopMargin -le $selectedTopMargin) {
            Fail "final sharpness invariant mismatch: $rowName"
        }
        $config = @($script:ConfigRows | Where-Object configuration_id -eq $row.configuration_id)[0]
        if ([int]$row.swfc_token_count -ne [int]$config.swfc_count -or
            $row.swfc_median_margin -ne $config.swfc_median_margin) { Fail "rank swfc anchor mismatch: $rowName" }
        if ([int]$row.swfc_median_rank -lt 1 -or [double]$row.swfc_median_entropy_selected -le 0.0) {
            Fail "rank swfc descriptor mismatch: $rowName"
        }
    }
}

function AssertAttributionEvidence() {
    $developmentIds = @($script:PartitionRows | Where-Object partition -eq 'AR_DEVELOPMENT_V3' | ForEach-Object { $_.case_id } | Sort-Object)
    if ($developmentIds.Count -ne 24) { Fail 'development case set anchor mismatch' }
    if ($script:AttributionRows.Count -ne 96) { Fail 'attribution row count mismatch' }
    $allowedAttributions = @('PERFECT_SELECTED','LOCAL_LOGIT_RANKING_FAILURE','PREFIX_DRIFT_AMPLIFICATION','PREFIX_DRIFT_AMPLIFICATION_UNCORROBORATED','MIXED','NO_CLEAR_ATTRIBUTION')
    $driftCount = 0
    $expectedCases = @()
    foreach ($config in $script:ConfigRows.configuration_id) {
        foreach ($caseId in $developmentIds) { $expectedCases += "$config/$caseId" }
    }
    $actualCases = @($script:AttributionRows | ForEach-Object { "$($_.configuration_id)/$($_.case_id)" } | Sort-Object)
    if (($expectedCases | Sort-Object) -join ',' -ne ($actualCases -join ',')) { Fail 'attribution case set mismatch' }
    foreach ($row in $script:AttributionRows) {
        $rowName = "common-prefix-attribution/$($row.configuration_id)/$($row.case_id)"
        if ($row.length -notin @('4','8') -or $row.corroboration_computed -notin @('true','false') -or
            $row.attribution -notin $allowedAttributions) { Fail "attribution shape mismatch: $rowName" }
        $localHit = $row.local_hit -eq 'true'
        $driftHit = $row.drift_hit -eq 'true'
        $nearGold = $row.near_gold -eq 'true'
        if ($row.attribution -eq 'LOCAL_LOGIT_RANKING_FAILURE') {
            if (-not $localHit -or $driftHit) { Fail "LOCAL attribution invariant mismatch: $rowName" }
        } elseif ($row.attribution -eq 'MIXED') {
            if (-not $localHit -or -not $driftHit) { Fail "MIXED attribution invariant mismatch: $rowName" }
        } elseif ($row.attribution -in @('PREFIX_DRIFT_AMPLIFICATION','PREFIX_DRIFT_AMPLIFICATION_UNCORROBORATED')) {
            $driftCount++
            if ($localHit -or -not $driftHit -or -not $nearGold) { Fail "PREFIX attribution invariant mismatch: $rowName" }
        } elseif ($row.attribution -eq 'PERFECT_SELECTED') {
            if ([int]$row.selected_first_error -ne -1) { Fail "PERFECT attribution mismatch: $rowName" }
        } elseif (-not $localHit -and $driftHit -and $nearGold) {
            Fail "NO_CLEAR attribution invariant mismatch: $rowName"
        }
        if ($row.corroboration_computed -eq 'true') {
            foreach ($field in @('rank1_cross_selected_on_final_prefix','rank1_cross_final_on_selected_prefix')) {
                $value = ParseFinite $row.$field $field $rowName
                if ($value -lt 0.0 -or $value -gt 1.0) { Fail "cross-rank fraction out of range: $rowName.$field" }
            }
        }
    }
    $script:DriftCaseCount = $driftCount
}

function ComputePooledHypotheses() {
    $script:PooledSwfcCount = ($script:ConfigRows | Where-Object configuration_id -in $l19Configs |
        ForEach-Object { [int]$_.swfc_count } | Measure-Object -Sum).Sum
    $allSwfcMargins = @()
    foreach ($config in $l19Configs) {
        $tokenFile = Join-Path $InputRoot ("margin-tokens-$config.csv")
        $rows = @(Import-Csv -LiteralPath $tokenFile | Where-Object bucket -eq 'SELECTED_WRONG_FINAL_CORRECT')
        $configRow = @($script:ConfigRows | Where-Object configuration_id -eq $config)[0]
        if ($rows.Count -ne [int]$configRow.swfc_count) { Fail "private swfc row count mismatch: $config" }
        $margins = @($rows | ForEach-Object { [double]$_.selected_self_margin })
        $sorted = @($margins | Sort-Object)
        $medianIndex = [int][math]::Floor(($sorted.Count - 1) / 2)
        ScalarConsistency ([double]$configRow.swfc_median_margin) $sorted[$medianIndex] "swfc-median/$config"
        $allSwfcMargins += $margins
    }
    $pooledSorted = @($allSwfcMargins | Sort-Object)
    $pooledIndex = [int][math]::Floor(($pooledSorted.Count - 1) / 2)
    $script:PooledSwfcMedian = $pooledSorted[$pooledIndex]
    $script:PooledEasyFraction = (($script:ConfigRows | Where-Object configuration_id -in $l19Configs |
        ForEach-Object { [int]$_.easy_token_count } | Measure-Object -Sum).Sum) / 432.0
    $bothCorrectShare = 0.0
    $allShare = 0.0
    foreach ($config in $l19Configs) {
        foreach ($bucket in @('BOTH_CORRECT','SELECTED_CORRECT_FINAL_WRONG','SELECTED_WRONG_FINAL_CORRECT','BOTH_WRONG')) {
            $row = @($script:BucketRows | Where-Object { $_.configuration_id -eq $config -and $_.bucket -eq $bucket })[0]
            $contribution = ParseFinite $row.nll_contribution 'nll_contribution' "$config/$bucket"
            $allShare += $contribution
            if ($bucket -eq 'BOTH_CORRECT') { $bothCorrectShare += $contribution }
        }
    }
    $script:PooledPercentGain = 0.0
    if ($allShare -ne 0.0 -and [math]::Abs($allShare) / 432.0 -ge 0.02) {
        $script:PooledPercentGain = 100.0 * $bothCorrectShare / $allShare
    }
}

function AssertHypothesisEvidence() {
    if ($script:HypothesisRows.Count -ne 6) { Fail 'hypothesis-decision row count mismatch' }
    $h1Percent = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'H1_EASY_TOKEN_NLL_DOMINANCE' -and $_.evidence_metric -eq 'PCT_GAIN_FROM_BOTH_CORRECT' })[0]
    $h1Easy = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'H1_EASY_TOKEN_NLL_DOMINANCE' -and $_.evidence_metric -eq 'EASY_TOKEN_FRACTION' })[0]
    $h2Count = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'H2_CRITICAL_TOKEN_MARGIN_LOSS' -and $_.evidence_metric -eq 'SWFC_TOKEN_COUNT' })[0]
    $h2Median = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'H2_CRITICAL_TOKEN_MARGIN_LOSS' -and $_.evidence_metric -eq 'SWFC_MEDIAN_MARGIN' })[0]
    $h3Count = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'H3_PREFIX_DRIFT_AMPLIFICATION' })[0]
    $conclusion = @($script:HypothesisRows | Where-Object { $_.hypothesis -eq 'CONCLUSION' })[0]
    foreach ($row in @($h1Percent, $h1Easy, $h2Count, $h2Median, $h3Count, $conclusion)) {
        if ($null -eq $row) { Fail 'hypothesis-decision set is incomplete' }
    }
    if ($h1Percent.supported -ne 'false' -or $h1Easy.supported -ne 'false' -or
        $h2Count.supported -ne 'true' -or $h2Median.supported -ne 'true' -or
        $h3Count.supported -ne 'false' -or $conclusion.supported -ne 'true') { Fail 'hypothesis support mismatch' }
    ScalarConsistency (ParseFinite $h1Percent.evidence_value 'evidence_value' 'H1_PCT') $script:PooledPercentGain 'H1.PCT_GAIN'
    if ([double]$h1Percent.threshold -ne 50.0) { Fail 'H1 percent threshold mismatch' }
    ScalarConsistency (ParseFinite $h1Easy.evidence_value 'evidence_value' 'H1_EASY') $script:PooledEasyFraction 'H1.EASY_FRACTION'
    if ([double]$h1Easy.threshold -ne 0.5) { Fail 'H1 easy threshold mismatch' }
    ScalarConsistency (ParseFinite $h2Count.evidence_value 'evidence_value' 'H2_COUNT') ([double]$script:PooledSwfcCount) 'H2.SWFC_COUNT'
    if ([double]$h2Count.threshold -ne 3.0) { Fail 'H2 count threshold mismatch' }
    ScalarConsistency (ParseFinite $h2Median.evidence_value 'evidence_value' 'H2_MEDIAN') $script:PooledSwfcMedian 'H2.SWFC_MEDIAN'
    if ([double]$h2Median.threshold -ne 0.0 -or $script:PooledSwfcMedian -ge 0.0) { Fail 'H2 median threshold mismatch' }
    ScalarConsistency (ParseFinite $h3Count.evidence_value 'evidence_value' 'H3_COUNT') ([double]$script:DriftCaseCount) 'H3.DRIFT_COUNT'
    if ([double]$h3Count.threshold -ne 2.0) { Fail 'H3 drift threshold mismatch' }
    if ($conclusion.conclusion -ne 'CRITICAL_TOKEN_MARGIN_LOSS') { Fail 'conclusion mismatch' }
}

function AssertSeedInferenceEvidence() {
    $seedToConfig = @{ '1'='L19_SEED_1'; '2'='L19_SEED_2'; '4'='L19_SEED_4' }
    if ($script:SeedRows.Count -ne 3) { Fail 'seed-comparison row count mismatch' }
    foreach ($row in $script:SeedRows) {
        $rowName = "seed-comparison/seed-$($row.seed)"
        $configId = $seedToConfig[$row.seed]
        if ($null -eq $configId) { Fail "unknown seed row: $rowName" }
        $config = @($script:ConfigRows | Where-Object configuration_id -eq $configId)[0]
        $selectedNll = ParseFinite $row.selected_nll 'selected_nll' $rowName
        $finalNll = ParseFinite $row.final_nll 'final_nll' $rowName
        ScalarConsistency (ParseFinite $row.nll_delta 'nll_delta' $rowName) (($finalNll - $selectedNll) * 144.0) "$rowName.nll_delta"
        if ([int]$row.selected_step -ne [int]$config.selected_step -or
            [int]$row.token_exact_selected -ne [int]$config.selected_token_exact -or
            [int]$row.token_exact_final -ne [int]$config.final_token_exact -or
            [int]$row.swfc_token_count -ne [int]$config.swfc_count -or
            $row.swfc_median_margin -ne $config.swfc_median_margin -or $row.attribution_dominant -ne 'MIXED') {
            Fail "seed anchor mismatch: $rowName"
        }
        $expected = @{
            h1_supported=('' + (($script:HypothesisRows | Where-Object hypothesis -eq 'H1_EASY_TOKEN_NLL_DOMINANCE')[0].supported))
            h2_supported=('' + (($script:HypothesisRows | Where-Object hypothesis -eq 'H2_CRITICAL_TOKEN_MARGIN_LOSS')[0].supported))
            h3_supported=('' + (($script:HypothesisRows | Where-Object hypothesis -eq 'H3_PREFIX_DRIFT_AMPLIFICATION')[0].supported))
        }
        foreach ($field in @('h1_supported','h2_supported','h3_supported')) {
            if ($row.$field -ne $expected[$field]) { Fail "seed hypothesis column mismatch: $rowName.$field" }
        }
    }
}

function AssertDepthControlEvidence() {
    if ($script:DepthRows.Count -ne 2) { Fail 'depth-control row count mismatch' }
    $control = @($script:DepthRows | Where-Object configuration_id -eq 'L18_SEED_2_CONTROL')[0]
    $l19Seed2 = @($script:DepthRows | Where-Object configuration_id -eq 'L19_SEED_2')[0]
    if ($null -eq $control -or $null -eq $l19Seed2) { Fail 'depth-control identity set mismatch' }
    foreach ($row in @($control, $l19Seed2)) {
        if ($row.selected_step -ne '4' -or $row.depth_control_observation -ne 'CONTROL_FINAL_EXACT_AT_LEAST_L19') {
            Fail "depth-control observation mismatch: $($row.configuration_id)"
        }
        $config = @($script:ConfigRows | Where-Object configuration_id -eq $row.configuration_id)[0]
        if ($row.selected_nll -ne $config.selected_nll -or $row.final_nll -ne $config.final_nll -or
            $row.nll_delta -ne $config.nll_delta -or $row.swfc_count -ne $config.swfc_count -or
            $row.swfc_median_margin -ne $config.swfc_median_margin) { Fail "depth-control config anchor mismatch: $($row.configuration_id)" }
    }
    $l19Rows = @($script:ConfigRows | Where-Object configuration_id -in $l19Configs)
    $l19MaxToken = ($l19Rows | ForEach-Object { [int]$_.final_token_exact } | Measure-Object -Maximum).Maximum
    $l19MaxSequence = ($l19Rows | ForEach-Object { [int]$_.final_sequence_exact } | Measure-Object -Maximum).Maximum
    if ([int]$control.token_exact_final -le $l19MaxToken -or [int]$control.sequence_exact_final -le $l19MaxSequence) {
        Fail 'depth control does not reach at-least-L19 exact'
    }
}

function AssertObjectiveEvidence() {
    if ($script:ObjectiveRows.Count -ne 4) { Fail 'next-objective row count mismatch' }
    if (($script:ObjectiveRows.candidate_objective | Sort-Object -Unique).Count -ne 4) { Fail 'objective id set mismatch' }
    foreach ($row in $script:ObjectiveRows) {
        foreach ($field in @('addresses','proposed_fix')) {
            if ([string]::IsNullOrWhiteSpace($row.$field)) { Fail "objective empty field: $($row.candidate_objective).$field" }
        }
        if ($row.status -ne 'NOT_RUN_CANDIDATE') { Fail "objective status mismatch: $($row.candidate_objective)" }
    }
}

function AssertSourceEvidence() {
    foreach ($name in $copied + $privateTokenFiles) {
        $path = Join-Path $InputRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing source file: $name" }
        if (-not (Safe ([IO.File]::ReadAllText($path)))) { Fail "unsafe source content: $name" }
    }
    $script:ConfigRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'configuration.csv'))
    $script:ComparisonRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'checkpoint-comparison.csv'))
    $script:BucketRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'token-buckets.csv'))
    $script:ErrorRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'first-error-summary.csv'))
    $script:RankRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'margin-rank-summary.csv'))
    $script:AttributionRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'common-prefix-attribution.csv'))
    $script:SeedRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'seed-comparison.csv'))
    $script:DepthRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'depth-control.csv'))
    $script:HypothesisRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'hypothesis-decision.csv'))
    $script:ObjectiveRows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'next-objective-candidates.csv'))
    $anchor = GetDevelopmentAnchorRows
    $script:DevelopmentRows = $anchor.Development
    $script:ValidationRows = $anchor.Validation
    $script:PartitionRows = $anchor.Partitions
    AssertConfigurationEvidence
    AssertComparisonEvidence
    AssertBucketEvidence
    AssertErrorEvidence
    AssertRankEvidence
    AssertAttributionEvidence
    ComputePooledHypotheses
    AssertHypothesisEvidence
    AssertSeedInferenceEvidence
    AssertDepthControlEvidence
    AssertObjectiveEvidence
}

function NewSelfTestFixture() {
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\margin-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $fixtureRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $fixtureInput = Join-Path $fixtureRoot 'input'
    $fixtureOutput = Join-Path $fixtureRoot 'output'
    [void](New-Item -ItemType Directory -Path $fixtureInput -Force)
    [void](New-Item -ItemType Directory -Path $fixtureOutput -Force)
    $publicRoot = Join-Path $repoRoot 'docs\results\qnn-htp-l19-first-error-margin-2026-08'
    foreach ($name in $copied) {
        $source = Join-Path $publicRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture source missing: $name" }
        [IO.File]::Copy($source, (Join-Path $fixtureInput $name), $true)
    }
    $reportRoot = Join-Path $repoRoot 'build\reports\qnn-l19-first-error-margin-2026-08'
    foreach ($name in $privateTokenFiles) {
        $source = Join-Path $reportRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture private source missing: $name" }
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
    $source = Join-Path $InputRoot $Name
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
    if ($manifest.schema -ne 'FIRST_ERROR_MARGIN_V1' -or $manifest.schema_version -ne 1 -or
        $manifest.result_classification -ne 'CRITICAL_TOKEN_MARGIN_LOSS') { Fail 'manifest decision mismatch' }
    if ($manifest.h1_supported -ne $false -or $manifest.h2_supported -ne $true -or $manifest.h3_supported -ne $false) {
        Fail 'manifest hypothesis mismatch'
    }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}

function NewReadme() {
    $format = { param($value) ([double]$value).ToString('0.######', $script:Invariant) }
    $c1 = @($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_1')[0]
    $c2 = @($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_2')[0]
    $c4 = @($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_4')[0]
    $cc = @($script:ConfigRows | Where-Object configuration_id -eq 'L18_SEED_2_CONTROL')[0]
    $dc = @($script:DepthRows | Where-Object configuration_id -eq 'L18_SEED_2_CONTROL')[0]
    $rowText = {
        param($c)
        "$($c.selected_token_exact)/144 ($($c.selected_sequence_exact)/24 seq)"
    }
@"
# First-error and margin decomposition, August 2026

This bundle decomposes why the L19 (T8/D16/FFN32/H2) final checkpoint trades a
higher autoregressive rollout NLL for higher token and sequence exact counts
relative to the validation-selected checkpoint. It is a host-only CPU replay
over the fixed AR_DEVELOPMENT_V3 partition; no device or HTP run contributed
data.

Each development case is replayed under the selected checkpoint and under the
320-step final checkpoint. For every token we record the midrank of the
target, its expected minus top-1 logit margin, top-1 minus top-2 margin,
softmax entropy, and token NLL, together with the first autoregressive error
position and its case class. Tokens are grouped into four buckets by whether
each checkpoint is exact: BOTH_CORRECT, SELECTED_CORRECT_FINAL_WRONG,
SELECTED_WRONG_FINAL_CORRECT (SWFC), and BOTH_WRONG. Cross-prefix conditions D
and E (one checkpoint run over the other checkpoint's free-running prefix)
corroborate whether a diverged prefix, rather than local ranking, explains an
exact gain.

Pooled over L19 seeds 1, 2, and 4 (432 development tokens per checkpoint
pair), the hard negative margin concentrates on the tokens the final
checkpoint corrects:

| Seed (selected step) | Selected exact | Final exact | SWFC tokens | SWFC median margin |
| --- | ---: | ---: | ---: | ---: |
| 1 ($($c1.selected_step)) | $(& $rowText $c1) | $($c1.final_token_exact)/144 ($($c1.final_sequence_exact)/24 seq) | $($c1.swfc_count) | $(& $format $c1.swfc_median_margin) |
| 2 ($($c2.selected_step)) | $(& $rowText $c2) | $($c2.final_token_exact)/144 ($($c2.final_sequence_exact)/24 seq) | $($c2.swfc_count) | $(& $format $c2.swfc_median_margin) |
| 4 ($($c4.selected_step)) | $(& $rowText $c4) | $($c4.final_token_exact)/144 ($($c4.final_sequence_exact)/24 seq) | $($c4.swfc_count) | $(& $format $c4.swfc_median_margin) |
| L18 control ($($cc.selected_step)) | $(& $rowText $cc) | $($cc.final_token_exact)/144 ($($cc.final_sequence_exact)/24 seq) | $($cc.swfc_count) | $(& $format $dc.swfc_median_margin) |

The final checkpoint's distribution is sharper and less calibrated: mean
entropy drops while the expected margin turns strongly negative (see
`margin-rank-summary.csv` and `token-buckets.csv`). The L18 depth control
retains at-least-L19 final exact
(`CONTROL_FINAL_EXACT_AT_LEAST_L19`), so the pattern is not depth-specific
beyond the L19 ceiling.

The precommitted hypothesis decision is in `hypothesis-decision.csv`:

- H1 (easy-token NLL dominance) is not supported: no easy token at the 0.5
  expected-probability threshold, and the both-correct bucket does not carry
  the pooled NLL movement.
- H2 (critical-token margin loss) is supported: 124 pooled SWFC tokens
  (threshold 3) with a negative pooled median margin (below the 0 threshold).
- H3 (prefix-drift amplification) is not supported: no corroborated
  prefix-drift-amplified case (0 vs threshold 2); `common-prefix-attribution.csv`
  attributes each case individually.

The conclusion is `CRITICAL_TOKEN_MARGIN_LOSS`. The four auxiliary-objective
candidates in `next-objective-candidates.csv` are `NOT_RUN_CANDIDATE`; they
are recorded for a separate, later investigation.

All exact counts, steps, and rollout NLL values reproduce the canonical
autoregressive bundle (docs/results/qnn-htp-autoregressive-validation-2026-08)
and are re-checked by the exporter against `ar-development-metrics.csv` and
`ar-validation-metrics.csv` before publishing. The public files deliberately
exclude model-state payloads, package material, device identifiers, endpoint
data, paths, and log streams.
"@
}

$selfTestContext = $null
$selfTestPublicSnapshot = @{}
if ($SelfTest) {
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-htp-l19-first-error-margin-2026-08'
    foreach ($name in $allowed) {
        $snapshotPath = Join-Path $publicSnapshotRoot $name
        if (Test-Path -LiteralPath $snapshotPath -PathType Leaf) {
            $selfTestPublicSnapshot[$name] = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash
        }
    }
    $selfTestContext = NewSelfTestFixture
    $InputRoot = $selfTestContext.Input
    $OutputRoot = $selfTestContext.Output
}
$InputRoot = RequireUnderRepository $InputRoot 'InputRoot'
$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $InputRoot -PathType Container)) { Fail 'InputRoot does not exist' }
if (-not (Test-Path -LiteralPath $OutputRoot)) { [void](New-Item -ItemType Directory -Path $OutputRoot -Force) }

RequireHeader 'configuration.csv' @('source','configuration_id','depth','seed','selected_step','pinned_selected_step','selected_step_matches_pinned','best_token_exact_step','best_token_exact_count','best_sequence_exact_step','best_sequence_exact_count','validation_selected_ar_nll','selected_nll','final_nll','nll_delta','selected_token_exact','final_token_exact','token_total','selected_sequence_exact','final_sequence_exact','sequence_total','easy_token_count','critical_token_count','easy_nll_gain','critical_nll_loss','swfc_count','swfc_median_margin','swfc_median_rank','swfc_median_entropy_selected','swfc_median_entropy_final','attribution_dominant')
RequireHeader 'checkpoint-comparison.csv' @('configuration_id','role','step','ar_nll','token_exact','token_total','sequence_exact','sequence_total','mean_first_error_position','no_error_cases','first_error_cases','mean_rank','mean_probability','mean_entropy','mean_expected_margin','mean_top1_top2_margin','rank1_unique','top2_inclusion','top3_inclusion','margin_nonnegative_fraction','margin_abs_below_log2_fraction')
RequireHeader 'token-buckets.csv' @('configuration_id','bucket','token_count','percent_tokens','selected_nll_mean','final_nll_mean','nll_contribution','mean_rank_selected','mean_rank_final','mean_margin_selected','mean_margin_final','mean_entropy_selected','mean_entropy_final','first_error_positions_selected','first_error_positions_final','selected_exact_count','final_exact_count')
RequireHeader 'first-error-summary.csv' @('configuration_id','selected_no_error','selected_late_single_error','selected_error_with_recovery','selected_early_irreversible_divergence','selected_multiple_local_errors','final_no_error','final_late_single_error','final_error_with_recovery','final_early_irreversible_divergence','final_multiple_local_errors','selected_mean_first_error_position','final_mean_first_error_position','selected_median_first_error_position','final_median_first_error_position','selected_first_error_cases','final_first_error_cases')
RequireHeader 'margin-rank-summary.csv' @('configuration_id','mean_rank_selected','mean_rank_final','rank1_unique_selected','rank1_unique_final','rank1_unique_total','top2_inclusion_selected','top2_inclusion_final','top3_inclusion_selected','top3_inclusion_final','mean_expected_probability_selected','mean_expected_probability_final','mean_entropy_selected','mean_entropy_final','mean_expected_minus_top1_margin_selected','mean_expected_minus_top1_margin_final','mean_top1_minus_top2_margin_selected','mean_top1_minus_top2_margin_final','fraction_margin_nonnegative_selected','fraction_margin_nonnegative_final','fraction_margin_abs_below_log2_selected','fraction_margin_abs_below_log2_final','swfc_token_count','swfc_median_margin','swfc_median_rank','swfc_median_entropy_selected','swfc_median_entropy_final')
RequireHeader 'common-prefix-attribution.csv' @('configuration_id','case_id','length','selected_first_error','final_first_error','local_hit','drift_hit','near_gold','delta_rank_mean','delta_margin_mean','rank1_cross_selected_on_final_prefix','rank1_cross_final_on_selected_prefix','corroboration_computed','attribution')
RequireHeader 'seed-comparison.csv' @('seed','selected_step','selected_nll','final_nll','nll_delta','token_exact_selected','token_exact_final','sequence_exact_selected','sequence_exact_final','swfc_token_count','swfc_median_margin','attribution_dominant','h1_supported','h2_supported','h3_supported')
RequireHeader 'depth-control.csv' @('configuration_id','depth','seed','selected_step','selected_nll','final_nll','nll_delta','token_exact_selected','token_exact_final','sequence_exact_selected','sequence_exact_final','swfc_count','swfc_median_margin','attribution_dominant','depth_control_observation')
RequireHeader 'hypothesis-decision.csv' @('hypothesis','supported','evidence_metric','evidence_value','threshold','conclusion')
RequireHeader 'next-objective-candidates.csv' @('candidate_objective','addresses','proposed_fix','status')

AssertSourceEvidence

foreach ($name in $copied) { CopySafe $name }
WriteUtf8 'README.md' (NewReadme)

$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{name=$name;sha256=(GetSha256 $name)}
}
$manifest = [ordered]@{
    schema='FIRST_ERROR_MARGIN_V1'; schema_version=1; result_classification='CRITICAL_TOKEN_MARGIN_LOSS'
    quality_configuration='T8/D16/FFN32/L19/H2'; analysis_partition='AR_DEVELOPMENT_V3'
    selected_steps=@{
        'L19_SEED_1'=(@($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_1')[0]).selected_step
        'L19_SEED_2'=(@($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_2')[0]).selected_step
        'L19_SEED_4'=(@($script:ConfigRows | Where-Object configuration_id -eq 'L19_SEED_4')[0]).selected_step
    }
    easy_probability_threshold=0.5; swfc_minimum_count=3; drift_minimum_count=2
    h1_supported=$false; h2_supported=$true; h3_supported=$false
    dominant_hypothesis='H2_CRITICAL_TOKEN_MARGIN_LOSS'
    cpu_replay_runs=4; device_runs=0; htp_runs=0; final_holdout_opened=$false
    regression_anchor='docs/results/qnn-htp-autoregressive-validation-2026-08'
    depth_control_observation=(@($script:DepthRows | Where-Object configuration_id -eq 'L18_SEED_2_CONTROL')[0]).depth_control_observation
    prohibited_payloads_published=$false; files=$manifestFiles
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle

if ($SelfTest) {
    if (-not (Safe 'selected_steps={"L19_SEED_1":"16"}')) { Fail 'safe-text false rejection' }
    foreach ($unsafe in @('C:\\local\\report.txt','/tmp/report.txt','adb -s serial shell','android_id=1','raw_checkpoint=payload','raw tensor dump','app_private_path=files/x','model.apk')) {
        if (Safe $unsafe) { Fail "unsafe self-test rejected incorrectly: $unsafe" }
    }
    try { RequireOutputRoot ([IO.Path]::GetTempPath()); Fail 'outside output-root rejection failed' } catch { if ($_.Exception.Message -match 'outside output-root rejection failed') { throw } }
    $hypothesisPath = Join-Path $InputRoot 'hypothesis-decision.csv'
    $hypothesisOriginal = [IO.File]::ReadAllText($hypothesisPath)
    try {
        [IO.File]::WriteAllText($hypothesisPath, $hypothesisOriginal.Replace('CRITICAL_TOKEN_MARGIN_LOSS', 'C:\local\mismatch'), $utf8)
        ExpectSelfTestFailure 'unsafe source content' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($hypothesisPath, $hypothesisOriginal, $utf8) }
    $schemaPath = Join-Path $InputRoot 'token-buckets.csv'
    $schemaOriginal = [IO.File]::ReadAllText($schemaPath)
    try {
        $lines = $schemaOriginal.Split("`n", 2)
        [IO.File]::WriteAllText($schemaPath, "bad-header`n$($lines[1])", $utf8)
        ExpectSelfTestFailure 'schema rejection' { RequireHeader 'token-buckets.csv' @('configuration_id','bucket','token_count','percent_tokens','selected_nll_mean') }
    } finally { [IO.File]::WriteAllText($schemaPath, $schemaOriginal, $utf8) }
    $numericPath = Join-Path $InputRoot 'seed-comparison.csv'
    $numericOriginal = [IO.File]::ReadAllText($numericPath)
    try {
        [IO.File]::WriteAllText($numericPath, [regex]::Replace($numericOriginal, '3\.197133062669808', 'NaN', 1), $utf8)
        ExpectSelfTestFailure 'non-finite rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($numericPath, $numericOriginal, $utf8) }
    $countPath = Join-Path $InputRoot 'hypothesis-decision.csv'
    $countOriginal = [IO.File]::ReadAllText($countPath)
    try {
        [IO.File]::WriteAllText($countPath, $countOriginal.Replace('"SWFC_TOKEN_COUNT","124"', '"SWFC_TOKEN_COUNT","100"'), $utf8)
        ExpectSelfTestFailure 'hypothesis count rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($countPath, $countOriginal, $utf8) }
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-htp-l19-first-error-margin-2026-08'
    foreach ($name in $selfTestPublicSnapshot.Keys) {
        $after = (Get-FileHash -LiteralPath (Join-Path $publicSnapshotRoot $name) -Algorithm SHA256).Hash
        if ($after -ne $selfTestPublicSnapshot[$name]) { Fail "self-test modified public docs: $name" }
    }
    if ($selfTestContext -and (Test-Path -LiteralPath $selfTestContext.Root)) {
        Remove-Item -LiteralPath $selfTestContext.Root -Recurse -Force
    }
}

Write-Host "l19 margin public export: PASS ($OutputRoot)"