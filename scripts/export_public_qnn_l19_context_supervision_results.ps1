param(
    [switch]$SelfTest,
    [string]$PrivateRoot = "build/private-diagnostics/context-supervision-goal",
    [string]$OutputRoot = "docs/results/qnn-l19-context-supervision-stability-2026-08",
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'POWERSHELL_7_REQUIRED' }
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourceCommit) { $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim() }

$allowList = @(
    'README.md', 'configuration.csv', 'evidence-inventory.csv',
    'measurement-audit.csv', 'hypothesis-registry.csv', 'decision-log.csv',
    'dataset-interventions.csv', 'matched-controls.csv', 'curriculum-protocols.csv',
    'training-results.csv', 'teacher-forced-vs-free-running.csv', 'seed-variance.csv',
    'attention-behavior-summary.csv', 'gradient-update-summary.csv',
    'negative-controls.csv', 'depth-control.csv', 'hypothesis-outcomes.csv',
    'causal-evidence.csv', 'diagnosis.csv', 'remaining-uncertainties.csv',
    'next-step-candidates.csv', 'manifest.json'
)

$sourceFiles = @(
    'host_tests/context_supervision_stability_lib.h',
    'host_tests/context_supervision_stability.cpp',
    'scripts/run_l19_context_supervision_stability.ps1',
    'scripts/export_public_qnn_l19_context_supervision_results.ps1'
)

$privateFiles = @(
    'cycle-001/measurement-audit.csv',
    'cycle-001/schedule-audit.csv',
    'cycle-001/training-results.csv',
    'cycle-002/schedule-audit.csv',
    'cycle-002/training-results.csv',
    'cycle-003/schedule-audit.csv',
    'cycle-003/training-results.csv'
)

function Get-NormalizedSha256([string]$Path) {
    $text = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $hash = $algorithm.ComputeHash($bytes)
    } finally {
        $algorithm.Dispose()
    }
    return (([BitConverter]::ToString($hash)) -replace '-', '').ToLowerInvariant()
}

function Write-Csv([object[]]$Rows, [string]$Path) {
    @($Rows) | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

function Assert-ExactSet([object[]]$Actual, [string[]]$Expected, [string]$Label) {
    $actualStrings = @($Actual | ForEach-Object {[string]$_})
    if (@($actualStrings | Sort-Object -Unique).Count -ne $actualStrings.Count) {
        throw "${Label}_DUPLICATE"
    }
    if ((@($actualStrings | Sort-Object) -join "`n") -ne (@($Expected | Sort-Object) -join "`n")) {
        throw "${Label}_SET_MISMATCH"
    }
}

function Assert-PrivateInputs([string]$Root) {
    foreach ($relative in @('state.json','events.jsonl','hypothesis-registry.json',
            'decisions/decision-001.json','decisions/decision-002.json',
            'decisions/decision-003.json') + $privateFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
            throw "MISSING_PRIVATE_INPUT:$relative"
        }
    }
}

function Assert-PreregistrationAndBudget([string]$Root) {
    $state = Get-Content -Raw (Join-Path $Root 'state.json') | ConvertFrom-Json
    if ($state.start_head -ne 'e8e97da9e8ffae44fe8e5f95540e61f90fb59c49' -or
        $state.status -notin @('analysis_complete_validation_pending', 'complete') -or
        [int]$state.cycle -ne 3) {
        throw 'GOAL_STATE_CONTRACT_MISMATCH'
    }
    $expectedUsed = @{
        major_cycles=3;cpu_trajectory_regenerations=4;short_training_runs=0;
        full_training_runs=28;dataset_intervention_conditions=5;curriculum_conditions=2;
        parameter_state_interventions=0;internal_state_interventions=0;additional_probes=0;
        final_holdout_opens=0;device_runs=0;htp_runs=0;adb_operations=0;ui_operations=0;
        count_from_one=0
    }
    foreach ($name in $expectedUsed.Keys) {
        if ([int]$state.used.$name -ne $expectedUsed[$name] -or
            [int]$state.used.$name -gt [int]$state.limits.$name) {
            throw "GOAL_BUDGET_MISMATCH:$name"
        }
    }
    $decisions = @(1..3 | ForEach-Object {
        Get-Content -Raw (Join-Path $Root ("decisions/decision-{0:d3}.json" -f $_)) | ConvertFrom-Json
    })
    foreach ($index in 0..2) {
        if ([int]$decisions[$index].decision -ne ($index + 1) -or
            $decisions[$index].made_before_results -ne $true) {
            throw "DECISION_PREREGISTRATION_MISMATCH:$($index + 1)"
        }
    }
    if ([int](($decisions | ForEach-Object {[int]$_.run_budget.full_training_runs} | Measure-Object -Sum).Sum) -ne 28 -or
        [int](($decisions | ForEach-Object {[int]$_.run_budget.dataset_intervention_conditions} | Measure-Object -Sum).Sum) -ne 5 -or
        [int](($decisions | ForEach-Object {[int]$_.run_budget.curriculum_conditions} | Measure-Object -Sum).Sum) -ne 2) {
        throw 'DECISION_BUDGET_SUM_MISMATCH'
    }
    $events = @(Get-Content (Join-Path $Root 'events.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
    $required = @('goal_started','independent_audits_complete','design_scope','cycle_001_pre_result_stop',
        'canonical_anchor_fix','cycle_001_complete','cycle_002_complete','cycle_003_complete','analysis_complete')
    $cursor = -1
    foreach ($eventName in $required) {
        $next = -1
        for ($index = $cursor + 1; $index -lt $events.Count; $index++) {
            if ($events[$index].event -eq $eventName) { $next = $index; break }
        }
        if ($next -lt 0) { throw "EVENT_LEDGER_MISSING_OR_OUT_OF_ORDER:$eventName" }
        $cursor = $next
    }
    $cycle3 = $events | Where-Object {$_.event -eq 'cycle_003_complete'} | Select-Object -Last 1
    if ($cycle3.stop_rule_applied -ne $true) { throw 'STOP_RULE_LEDGER_MISMATCH' }
}

function Get-AllRows([string]$Root) {
    $rows = @()
    foreach ($cycle in 1..3) {
        $rows += @(Import-Csv (Join-Path $Root ("cycle-{0:d3}/training-results.csv" -f $cycle)))
    }
    return @($rows)
}

function Get-Variance([double[]]$Values) {
    if ($Values.Count -eq 0) { return [double]::NaN }
    $mean = ($Values | Measure-Object -Average).Average
    $sum = 0.0
    foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
    return $sum / $Values.Count
}

function Test-FiniteNumber([object]$Value) {
    $parsed = 0.0
    if (-not [double]::TryParse([string]$Value,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) { return $false }
    return -not [double]::IsNaN($parsed) -and -not [double]::IsInfinity($parsed)
}

function Get-ExpectedTrainingKeys([switch]$PublicDose) {
    $configurations = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
    $datasets = @('AR_DEVELOPMENT_V3', 'HOMOGENEOUS_PHASE0')
    $conditions = @(
        @('CANONICAL_HOMOGENEOUS', $(if ($PublicDose) {'0'} else {'25'})),
        @('MATCHED_HOMOGENEOUS', '25'),
        @('MIXED_INVARIANCE', '25'),
        @('CURRICULUM_MIXED_FIRST', '25'),
        @('CURRICULUM_MIXED_LAST', '25'),
        @('CURRICULUM_MIXED_LAST', '12.5'),
        @('CURRICULUM_MIXED_LAST', '6.25')
    )
    return @($conditions | ForEach-Object {
        $condition = $_[0]
        $dose = $_[1]
        foreach ($configuration in $configurations) {
            foreach ($dataset in $datasets) {
                "$configuration|$condition|$dose|$dataset"
            }
        }
    })
}

function Assert-SuccessorHistogram([object]$Row, [string]$Label) {
    $input = @{}
    $target = @{}
    foreach ($entry in ([string]$Row.input_histogram).Split(';')) {
        $parts = $entry.Split(':')
        $input[[int]$parts[0]] = [int]$parts[1]
    }
    foreach ($entry in ([string]$Row.target_histogram).Split(';')) {
        $parts = $entry.Split(':')
        $target[[int]$parts[0]] = [int]$parts[1]
    }
    Assert-ExactSet @($input.Keys | ForEach-Object {[string]$_}) @(0..31 | ForEach-Object {[string]$_}) "${Label}_INPUT_TOKENS"
    Assert-ExactSet @($target.Keys | ForEach-Object {[string]$_}) @(0..31 | ForEach-Object {[string]$_}) "${Label}_TARGET_TOKENS"
    foreach ($token in 0..12) {
        if ($token -le 3) { $successor = ($token + 1) % 4 }
        elseif ($token -le 7) { $successor = 4 + (($token - 4 + 1) % 4) }
        elseif ($token -le 9) { $successor = 8 + (($token - 8 + 1) % 2) }
        else { $successor = 10 + (($token - 10 + 1) % 3) }
        if ($target[$successor] -ne $input[$token]) {
            throw "${Label}_SUCCESSOR_CONTRACT:$token"
        }
    }
    foreach ($token in 13..31) {
        if ($input[$token] -ne 0 -or $target[$token] -ne 0) {
            throw "${Label}_UNEXPECTED_TOKEN:$token"
        }
    }
}

function Assert-PrivateScheduleEvidence([string]$Root) {
    $cycle1 = @(Import-Csv (Join-Path $Root 'cycle-001/schedule-audit.csv'))
    $cycle2 = @(Import-Csv (Join-Path $Root 'cycle-002/schedule-audit.csv'))
    $cycle3 = @(Import-Csv (Join-Path $Root 'cycle-003/schedule-audit.csv'))
    Assert-ExactSet @($cycle1 | ForEach-Object {"$($_.condition)|$($_.dose_percent)"}) @(
        'CANONICAL_HOMOGENEOUS|25', 'MATCHED_HOMOGENEOUS|25', 'MIXED_INVARIANCE|25') 'CYCLE1_SCHEDULE_ROWS'
    Assert-ExactSet @($cycle2 | ForEach-Object {"$($_.condition)|$($_.dose_percent)"}) @(
        'CURRICULUM_MIXED_FIRST|25', 'CURRICULUM_MIXED_LAST|25') 'CYCLE2_SCHEDULE_ROWS'
    Assert-ExactSet @($cycle3 | ForEach-Object {"$($_.condition)|$($_.dose_percent)"}) @(
        'CURRICULUM_MIXED_LAST|12.5', 'CURRICULUM_MIXED_LAST|6.25') 'CYCLE3_SCHEDULE_ROWS'

    $expectedHashes = @{
        'CANONICAL_HOMOGENEOUS|25'='fnv1a64:e05aa08cad24a127'
        'MATCHED_HOMOGENEOUS|25'='fnv1a64:ae785146103c9e87'
        'MIXED_INVARIANCE|25'='fnv1a64:137bc62edb9199d0'
        'CURRICULUM_MIXED_FIRST|25'='fnv1a64:4ef497bc24a1c641'
        'CURRICULUM_MIXED_LAST|25'='fnv1a64:94f00a59ce4828e2'
        'CURRICULUM_MIXED_LAST|12.5'='fnv1a64:6884f3404a78a015'
        'CURRICULUM_MIXED_LAST|6.25'='fnv1a64:4d89f88e0ce0617c'
    }
    foreach ($row in @($cycle1 + $cycle2 + $cycle3)) {
        $key = "$($row.condition)|$($row.dose_percent)"
        if ([int]$row.steps -ne 320 -or [int]$row.supervised_rows -ne 2560 -or
            $row.schedule_hash -ne $expectedHashes[$key]) {
            throw "SCHEDULE_CONTRACT_MISMATCH:$key"
        }
        Assert-SuccessorHistogram $row "SCHEDULE_$key"
    }
    $canonical = $cycle1 | Where-Object {$_.condition -eq 'CANONICAL_HOMOGENEOUS'}
    $matched = $cycle1 | Where-Object {$_.condition -eq 'MATCHED_HOMOGENEOUS'}
    $mixed = $cycle1 | Where-Object {$_.condition -eq 'MIXED_INVARIANCE'}
    if ([int]$canonical.special_steps -ne 0 -or [int]$matched.special_steps -ne 80 -or
        [int]$mixed.special_steps -ne 80 -or $matched.matched_histogram -ne 'true' -or
        $mixed.matched_histogram -ne 'true' -or
        $matched.input_histogram -ne $mixed.input_histogram -or
        $matched.target_histogram -ne $mixed.target_histogram) {
        throw 'MATCHED_SCHEDULE_CONTRACT_MISMATCH'
    }
    foreach ($row in $cycle2) {
        if ([int]$row.special_steps -ne 80 -or $row.same_multiset_as_interleaved -ne 'true' -or
            $row.input_histogram -ne $mixed.input_histogram -or
            $row.target_histogram -ne $mixed.target_histogram) {
            throw "CURRICULUM_MULTISET_MISMATCH:$($row.condition)"
        }
    }
    $dose12 = $cycle3 | Where-Object {$_.dose_percent -eq '12.5'}
    $dose6 = $cycle3 | Where-Object {$_.dose_percent -eq '6.25'}
    if ([int]$dose12.special_steps -ne 40 -or [int]$dose12.dose_denominator -ne 8 -or
        [int]$dose6.special_steps -ne 20 -or [int]$dose6.dose_denominator -ne 16) {
        throw 'DOSE_SCHEDULE_CONTRACT_MISMATCH'
    }
}

function Assert-PrivateScientificEvidence([string]$Root) {
    Assert-PreregistrationAndBudget $Root
    Assert-PrivateScheduleEvidence $Root
    $rows = Get-AllRows $Root
    $scheduleRows = @(
        Import-Csv (Join-Path $Root 'cycle-001/schedule-audit.csv')
        Import-Csv (Join-Path $Root 'cycle-002/schedule-audit.csv')
        Import-Csv (Join-Path $Root 'cycle-003/schedule-audit.csv')
    )
    $scheduleHashes = @{}
    foreach ($schedule in $scheduleRows) {
        $scheduleHashes["$($schedule.condition)|$($schedule.dose_percent)"] = $schedule.schedule_hash
    }
    Assert-ExactSet @($rows | ForEach-Object {
        "$($_.configuration_id)|$($_.condition)|$($_.dose_percent)|$($_.dataset)"
    }) (Get-ExpectedTrainingKeys) 'PRIVATE_TRAINING_IDENTITIES'
    foreach ($row in $rows) {
        $rowKey = "$($row.condition)|$($row.dose_percent)"
        if ([int]$row.steps -ne 320 -or [int]$row.example_count -ne 320 -or
            $row.train_finite -ne 'true' -or $row.all_finite -ne 'true' -or
            $row.ordinary_attention_active -ne 'true' -or
            $row.schedule_hash -ne $scheduleHashes[$rowKey]) {
            throw "PRIVATE_RUN_CONTRACT_MISMATCH:$($row.configuration_id):$($row.condition):$($row.dataset)"
        }
        $expectedSpecial = if ($row.condition -eq 'CANONICAL_HOMOGENEOUS') {0}
            elseif ($row.dose_percent -eq '25') {80}
            elseif ($row.dose_percent -eq '12.5') {40} else {20}
        if ([int]$row.special_example_count -ne $expectedSpecial) {
            throw "SPECIAL_EXAMPLE_COUNT_MISMATCH:$($row.configuration_id):$($row.condition):$($row.dose_percent)"
        }
        $tokenTotal = if ($row.dataset -eq 'AR_DEVELOPMENT_V3') {144} else {32}
        $sequenceTotal = if ($row.dataset -eq 'AR_DEVELOPMENT_V3') {24} else {4}
        if ([int]$row.teacher_token_total -ne $tokenTotal -or [int]$row.free_token_total -ne $tokenTotal -or
            [int]$row.free_sequence_total -ne $sequenceTotal -or
            [int]$row.teacher_token_exact -lt 0 -or [int]$row.teacher_token_exact -gt $tokenTotal -or
            [int]$row.free_token_exact -lt 0 -or [int]$row.free_token_exact -gt $tokenTotal -or
            [int]$row.free_sequence_exact -lt 0 -or [int]$row.free_sequence_exact -gt $sequenceTotal -or
            -not (Test-FiniteNumber $row.final_train_loss) -or [double]$row.final_train_loss -lt 0 -or
            -not (Test-FiniteNumber $row.teacher_nll) -or [double]$row.teacher_nll -lt 0 -or
            -not (Test-FiniteNumber $row.free_nll) -or [double]$row.free_nll -lt 0 -or
            -not (Test-FiniteNumber $row.teacher_margin_q10) -or
            -not (Test-FiniteNumber $row.free_margin_q10) -or
            -not (Test-FiniteNumber $row.median_first_error)) {
            throw "PRIVATE_METRIC_CONTRACT_MISMATCH:$($row.configuration_id):$($row.condition):$($row.dataset)"
        }
    }
    foreach ($condition in @('CURRICULUM_MIXED_FIRST', 'CURRICULUM_MIXED_LAST')) {
        $dev = @($rows | Where-Object {
            $_.dataset -eq 'AR_DEVELOPMENT_V3' -and $_.condition -eq $condition -and
            $_.dose_percent -eq '25' -and $_.configuration_id -like 'L19_*'
        })
        Assert-ExactSet @($dev | ForEach-Object {[string]$_.seed}) @('1','2','4') "${condition}_L19_SEEDS"
        foreach ($row in $dev) {
            if ([int]$row.free_token_exact -lt 140 -or [int]$row.free_sequence_exact -lt 22 -or
                ([int]$row.teacher_token_exact - [int]$row.free_token_exact) -gt 10 -or
                [double]$row.qk_update_l2 -le 0 -or [double]$row.v_update_l2 -le 0 -or
                [double]$row.o_update_l2 -le 0 -or [double]$row.attention_output_norm -le 0.001 -or
                [double]$row.attention_nonself_mass -le 0.05) {
                throw "FULL_STABILITY_OR_ATTENTION_GATE:$($row.configuration_id):$condition"
            }
        }
        $homogeneous = @($rows | Where-Object {
            $_.dataset -eq 'HOMOGENEOUS_PHASE0' -and $_.condition -eq $condition -and
            $_.dose_percent -eq '25' -and $_.configuration_id -like 'L19_*'
        })
        Assert-ExactSet @($homogeneous | ForEach-Object {[string]$_.seed}) @('1','2','4') "${condition}_HOMOGENEOUS_SEEDS"
        if (@($homogeneous | Where-Object {[int]$_.free_token_exact -lt 30}).Count -ne 0) {
            throw "HOMOGENEOUS_PRESERVATION_GATE:$condition"
        }
    }
    foreach ($condition in @('CURRICULUM_MIXED_FIRST', 'CURRICULUM_MIXED_LAST')) {
        $l18 = @($rows | Where-Object {
            $_.dataset -eq 'AR_DEVELOPMENT_V3' -and $_.condition -eq $condition -and
            $_.dose_percent -eq '25' -and $_.configuration_id -eq 'L18_SEED_2_CONTROL'
        })
        if ($l18.Count -ne 1 -or [int]$l18[0].free_token_exact -ne 144 -or
            [int]$l18[0].free_sequence_exact -ne 24) {
            throw "L18_SCOPE_CONTROL_GATE:$condition"
        }
    }
}

function Assert-PublicBundle([string]$Output) {
    $actual = @(Get-ChildItem -LiteralPath $Output -File | ForEach-Object Name)
    Assert-ExactSet $actual $allowList 'PUBLIC_ALLOW_LIST'
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $Output 'manifest.json') | ConvertFrom-Json
    if ([int]$manifest.schema_version -ne 1 -or
        $manifest.protocol -ne 'CONTEXT_SUPERVISION_STABILITY_V1' -or
        $manifest.result_classification -ne 'TARGET_INVARIANT_MIXED_PREFIX_SUPERVISION' -or
        $manifest.claim_strength -ne 'PRIMARY_CAUSAL_FACTOR' -or
        [int]$manifest.major_cycles -ne 3 -or
        [int]$manifest.cpu_trajectory_regenerations -ne 4 -or
        [int]$manifest.short_training_runs -ne 0 -or
        [int]$manifest.full_training_runs -ne 28 -or
        [int]$manifest.dataset_intervention_conditions -ne 5 -or
        [int]$manifest.curriculum_conditions -ne 2 -or
        [int]$manifest.final_holdout_opens -ne 0 -or
        [int]$manifest.device_runs -ne 0 -or [int]$manifest.htp_runs -ne 0 -or
        [int]$manifest.adb_operations -ne 0 -or [int]$manifest.ui_operations -ne 0 -or
        [int]$manifest.count_from_one -ne 0) {
        throw 'MANIFEST_SCIENTIFIC_CONTRACT_MISMATCH'
    }
    Assert-ExactSet @($manifest.allow_list) $allowList 'MANIFEST_ALLOW_LIST'
    Assert-ExactSet @($manifest.files.path) @($allowList | Where-Object {$_ -ne 'manifest.json'}) 'MANIFEST_FILES'
    Assert-ExactSet @($manifest.sources.path) $sourceFiles 'MANIFEST_SOURCES'
    Assert-ExactSet @($manifest.private_aggregates.aggregate) $privateFiles 'MANIFEST_PRIVATE_AGGREGATES'
    foreach ($entry in $manifest.files) {
        if ((Get-NormalizedSha256 (Join-Path $Output $entry.path)) -ne $entry.sha256_normalized_lf) {
            throw "PUBLIC_FILE_HASH_MISMATCH:$($entry.path)"
        }
    }
    foreach ($entry in $manifest.sources) {
        if ((Get-NormalizedSha256 (Join-Path $repoRoot $entry.path)) -ne $entry.sha256_normalized_lf) {
            throw "SOURCE_HASH_MISMATCH:$($entry.path)"
        }
    }
    foreach ($file in Get-ChildItem -LiteralPath $Output -File) {
        $text = Get-Content -Raw -LiteralPath $file.FullName
        if ($text -match '[A-Za-z]:[\\/]' -or $text -match 'build[\\/]private-diagnostics' -or
            $text -match '\badb\s+(?:-s\s+)?(?:[0-9a-f]{6,}|[^\s,]+:\d+)') {
            throw "PRIVATE_DATA_SCAN:$($file.Name)"
        }
    }
    $training = @(Import-Csv (Join-Path $Output 'training-results.csv'))
    if ($training.Count -ne 56) { throw 'TRAINING_PUBLIC_ROW_COUNT' }
    Assert-ExactSet @($training | ForEach-Object {
        "$($_.configuration_id)|$($_.condition)|$($_.dose_percent)|$($_.dataset)"
    }) (Get-ExpectedTrainingKeys -PublicDose) 'PUBLIC_TRAINING_IDENTITIES'
    if (@($training | Where-Object {
        $_.train_finite -ne 'true' -or $_.all_finite -ne 'true' -or
        $_.ordinary_attention_active -ne 'true'
    }).Count -ne 0) {
        throw 'FINITE_OR_ATTENTION_ACTIVITY_MISMATCH'
    }
    foreach ($row in $training) {
        $tokenTotal = if ($row.dataset -eq 'AR_DEVELOPMENT_V3') {144} else {32}
        $sequenceTotal = if ($row.dataset -eq 'AR_DEVELOPMENT_V3') {24} else {4}
        if ([int]$row.teacher_token_total -ne $tokenTotal -or [int]$row.free_token_total -ne $tokenTotal -or
            [int]$row.free_sequence_total -ne $sequenceTotal -or
            [int]$row.teacher_token_exact -lt 0 -or [int]$row.teacher_token_exact -gt $tokenTotal -or
            [int]$row.free_token_exact -lt 0 -or [int]$row.free_token_exact -gt $tokenTotal -or
            [int]$row.free_sequence_exact -lt 0 -or [int]$row.free_sequence_exact -gt $sequenceTotal -or
            -not (Test-FiniteNumber $row.final_train_loss) -or [double]$row.final_train_loss -lt 0 -or
            -not (Test-FiniteNumber $row.teacher_nll) -or [double]$row.teacher_nll -lt 0 -or
            -not (Test-FiniteNumber $row.free_nll) -or [double]$row.free_nll -lt 0 -or
            -not (Test-FiniteNumber $row.teacher_margin_q10) -or
            -not (Test-FiniteNumber $row.free_margin_q10) -or
            -not (Test-FiniteNumber $row.median_first_error)) {
            throw "PUBLIC_METRIC_CONTRACT_MISMATCH:$($row.configuration_id):$($row.condition):$($row.dataset)"
        }
    }
    $dev = @($training | Where-Object {$_.dataset -eq 'AR_DEVELOPMENT_V3'})
    $expected = @{
        'CANONICAL_HOMOGENEOUS|0' = '30,63,46'
        'MATCHED_HOMOGENEOUS|25' = '65,105,45'
        'MIXED_INVARIANCE|25' = '144,136,144'
        'CURRICULUM_MIXED_FIRST|25' = '144,141,144'
        'CURRICULUM_MIXED_LAST|25' = '144,144,144'
        'CURRICULUM_MIXED_LAST|12.5' = '112,111,114'
        'CURRICULUM_MIXED_LAST|6.25' = '51,38,61'
    }
    foreach ($key in $expected.Keys) {
        $parts = $key.Split('|')
        $values = @($dev | Where-Object {
            $_.condition -eq $parts[0] -and $_.dose_percent -eq $parts[1] -and
            $_.configuration_id -like 'L19_*'
        } | Sort-Object {[int]$_.seed} | ForEach-Object {[string]$_.free_token_exact}) -join ','
        if ($values -ne $expected[$key]) { throw "OUTCOME_ANCHOR_MISMATCH:${key}:${values}" }
    }
    foreach ($condition in @('CURRICULUM_MIXED_FIRST', 'CURRICULUM_MIXED_LAST')) {
        $passing = @($dev | Where-Object {
            $_.condition -eq $condition -and $_.dose_percent -eq '25' -and
            $_.configuration_id -like 'L19_*'
        })
        Assert-ExactSet @($passing | ForEach-Object {[string]$_.seed}) @('1','2','4') "PUBLIC_${condition}_SEEDS"
        if (@($passing | Where-Object {
            [int]$_.free_token_exact -lt 140 -or [int]$_.free_sequence_exact -lt 22 -or
            ([int]$_.teacher_token_exact - [int]$_.free_token_exact) -gt 10
        }).Count -ne 0) { throw "PUBLIC_FULL_STABILITY_GATE:$condition" }
        $homogeneousPassing = @($training | Where-Object {
            $_.dataset -eq 'HOMOGENEOUS_PHASE0' -and $_.condition -eq $condition -and
            $_.dose_percent -eq '25' -and $_.configuration_id -like 'L19_*'
        })
        if (@($homogeneousPassing | Where-Object {[int]$_.free_token_exact -lt 30}).Count -ne 0) {
            throw "PUBLIC_HOMOGENEOUS_PRESERVATION_GATE:$condition"
        }
    }
    foreach ($condition in @('CURRICULUM_MIXED_FIRST', 'CURRICULUM_MIXED_LAST')) {
        if (@($dev | Where-Object {
            $_.condition -eq $condition -and $_.dose_percent -eq '25' -and
            $_.configuration_id -eq 'L18_SEED_2_CONTROL' -and
            $_.free_token_exact -eq '144' -and $_.free_sequence_exact -eq '24'
        }).Count -ne 1) { throw "L18_SCOPE_CONTROL_MISMATCH:$condition" }
    }
    return $manifest
}

function Export-Bundle([string]$Private, [string]$Output, [string]$Commit) {
    Assert-PrivateInputs $Private
    Assert-PrivateScientificEvidence $Private
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $Output 'README.md') -PathType Leaf)) {
        throw 'README_MUST_EXIST_BEFORE_EXPORT'
    }
    $all = Get-AllRows $Private
    foreach ($row in $all) {
        if ($row.condition -eq 'CANONICAL_HOMOGENEOUS') { $row.dose_percent = '0' }
    }
    $dev = @($all | Where-Object {$_.dataset -eq 'AR_DEVELOPMENT_V3'})
    $homogeneous = @($all | Where-Object {$_.dataset -eq 'HOMOGENEOUS_PHASE0'})

    Write-Csv @(
        [pscustomobject]@{configuration_id='L19_SEED_1';depth=19;seed=1;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;optimizer='Adam lr 0.003 beta1 0.9 beta2 0.999 epsilon 1e-8';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L19_SEED_2';depth=19;seed=2;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;optimizer='Adam lr 0.003 beta1 0.9 beta2 0.999 epsilon 1e-8';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L19_SEED_4';depth=19;seed=4;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;optimizer='Adam lr 0.003 beta1 0.9 beta2 0.999 epsilon 1e-8';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L18_SEED_2_CONTROL';depth=18;seed=2;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;optimizer='Adam lr 0.003 beta1 0.9 beta2 0.999 epsilon 1e-8';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'}
    ) (Join-Path $Output 'configuration.csv')

    Write-Csv @(
        [pscustomobject]@{evidence='canonical anchors';status='PASS';result='30/63/46 L19 and 65 L18 free-running tokens';role='bitwise no-intervention reproduction'},
        [pscustomobject]@{evidence='matched homogeneous';status='NEGATIVE_CONTROL';result='65/105/45 L19';role='same steps rows histogram family exposure and cadence'},
        [pscustomobject]@{evidence='25 percent interleaved mixed';status='PARTIAL';result='144/136/144 L19';role='target-invariant context diversity'},
        [pscustomobject]@{evidence='25 percent same-multiset curricula';status='PASS';result='mixed-first 144/141/144; mixed-last 144/144/144';role='order necessity'},
        [pscustomobject]@{evidence='smaller doses';status='REJECTED';result='12.5 percent 112/111/114; 6.25 percent 51/38/61';role='minimum tested dose'},
        [pscustomobject]@{evidence='ordinary Attention activity';status='PASS';result='Q K V O updated; output norm and non-self mass nonzero';role='exclude Attention disablement'},
        [pscustomobject]@{evidence='final holdout';status='HASH_ONLY';result='0 evaluations';role='selection safety'}
    ) (Join-Path $Output 'evidence-inventory.csv')

    $measurement = Import-Csv (Join-Path $Private 'cycle-001/measurement-audit.csv') |
        Select-Object configuration_id,check,status,observed,expected
    Write-Csv $measurement (Join-Path $Output 'measurement-audit.csv')

    $registry = Get-Content -Raw (Join-Path $Private 'hypothesis-registry.json') | ConvertFrom-Json
    $outcomes = @{H1='UPSTREAM_DATA_CONDITION_SUPPORTED_INVARIANCE_MECHANISM_PARTIAL';H2='NOT_REQUIRED_FOR_RESCUE_AND_NOT_IDENTIFIABLE_AS_BENEFIT';H3='REJECTED_AS_SUFFICIENT';H4='DIRECTION_NOT_REQUIRED_BUT_ORDER_AFFECTS_RESIDUAL';H5='MITIGATED_BY_DATA';H6='REJECTED_AS_SOLE';H7='REJECTED_FOR_NEW_PATH'}
    Write-Csv @($registry.hypotheses | ForEach-Object {
        [pscustomobject]@{id=$_.id;evidence_phase='PREREGISTERED_PRE_RESULT';hypothesis=$_.hypothesis;supporting_evidence=($_.supporting_evidence -join '; ');counterevidence=($_.counterevidence -join '; ');prediction=$_.prediction;minimal_test=$_.minimal_test;negative_control=$_.negative_control;post_result_outcome=$outcomes[$_.id]}
    }) (Join-Path $Output 'hypothesis-registry.csv')

    Write-Csv @(1..3 | ForEach-Object {
        $d = Get-Content -Raw (Join-Path $Private ("decisions/decision-{0:d3}.json" -f $_)) | ConvertFrom-Json
        $decisionReason = if ($null -ne $d.selection_reason) {$d.selection_reason} else {$d.reason}
        [pscustomobject]@{decision=$_.ToString();made_before_results=$d.made_before_results;experiment=$d.experiment;reason=([string]$decisionReason);negative_controls=($d.negative_controls -join '; ');full_training_budget=$d.run_budget.full_training_runs}
    }) (Join-Path $Output 'decision-log.csv')

    Write-Csv @(
        [pscustomobject]@{condition='CANONICAL_HOMOGENEOUS';mixed_batches=0;homogeneous_batches=320;dose_percent=0;target_contract='successor current token';attention='ordinary learned'},
        [pscustomobject]@{condition='MATCHED_HOMOGENEOUS';mixed_batches=0;homogeneous_batches=320;dose_percent=25;target_contract='successor current token';attention='ordinary learned'},
        [pscustomobject]@{condition='MIXED_INVARIANCE';mixed_batches=80;homogeneous_batches=240;dose_percent=25;target_contract='successor current token';attention='ordinary learned'},
        [pscustomobject]@{condition='CURRICULUM_MIXED_LAST';mixed_batches=40;homogeneous_batches=280;dose_percent=12.5;target_contract='successor current token';attention='ordinary learned'},
        [pscustomobject]@{condition='CURRICULUM_MIXED_LAST';mixed_batches=20;homogeneous_batches=300;dose_percent=6.25;target_contract='successor current token';attention='ordinary learned'}
    ) (Join-Path $Output 'dataset-interventions.csv')

    Write-Csv @(
        [pscustomobject]@{control='MATCHED_HOMOGENEOUS';steps=320;supervised_rows=2560;special_steps=80;aggregate_input_target_histogram_matched='true';l19_free_tokens='65/105/45';interpretation='data amount and phase diversity insufficient'},
        [pscustomobject]@{control='CANONICAL_HOMOGENEOUS';steps=320;supervised_rows=2560;special_steps=0;aggregate_input_target_histogram_matched='not_applicable';l19_free_tokens='30/63/46';interpretation='canonical anchor'}
    ) (Join-Path $Output 'matched-controls.csv')

    Write-Csv @(
        [pscustomobject]@{curriculum='INTERLEAVED';mixed_batches=80;homogeneous_batches=240;multiset='MIXED_25_V1';l19_free_tokens='144/136/144';gate='PARTIAL'},
        [pscustomobject]@{curriculum='MIXED_FIRST';mixed_batches=80;homogeneous_batches=240;multiset='MIXED_25_V1';l19_free_tokens='144/141/144';gate='PASS'},
        [pscustomobject]@{curriculum='MIXED_LAST';mixed_batches=80;homogeneous_batches=240;multiset='MIXED_25_V1';l19_free_tokens='144/144/144';gate='PASS'}
    ) (Join-Path $Output 'curriculum-protocols.csv')

    $publicTraining = $all | Select-Object configuration_id,depth,seed,condition,dose_percent,dataset,steps,example_count,special_example_count,train_finite,final_train_loss,teacher_token_exact,teacher_token_total,teacher_nll,teacher_margin_q10,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,free_margin_q10,ordinary_attention_active,all_finite
    Write-Csv $publicTraining (Join-Path $Output 'training-results.csv')
    Write-Csv @($dev | ForEach-Object {
        [pscustomobject]@{configuration_id=$_.configuration_id;condition=$_.condition;dose_percent=$_.dose_percent;teacher_token_exact=$_.teacher_token_exact;free_token_exact=$_.free_token_exact;exact_gap=([int]$_.teacher_token_exact-[int]$_.free_token_exact);teacher_nll=$_.teacher_nll;free_nll=$_.free_nll;teacher_margin_q10=$_.teacher_margin_q10;free_margin_q10=$_.free_margin_q10;free_sequence_exact=$_.free_sequence_exact;all_finite=$_.all_finite}
    }) (Join-Path $Output 'teacher-forced-vs-free-running.csv')

    $varianceRows = foreach ($group in ($dev | Where-Object {$_.configuration_id -like 'L19_*'} | Group-Object condition,dose_percent)) {
        $tokens = @($group.Group | ForEach-Object {[double]$_.free_token_exact})
        $sequences = @($group.Group | ForEach-Object {[double]$_.free_sequence_exact})
        [pscustomobject]@{condition=$group.Group[0].condition;dose_percent=$group.Group[0].dose_percent;seed_count=3;token_mean=(($tokens|Measure-Object -Average).Average);token_variance=(Get-Variance $tokens);token_range=(($tokens|Measure-Object -Maximum).Maximum-($tokens|Measure-Object -Minimum).Minimum);sequence_mean=(($sequences|Measure-Object -Average).Average);sequence_variance=(Get-Variance $sequences);sequence_range=(($sequences|Measure-Object -Maximum).Maximum-($sequences|Measure-Object -Minimum).Minimum)}
    }
    Write-Csv $varianceRows (Join-Path $Output 'seed-variance.csv')
    Write-Csv @($dev | Select-Object configuration_id,condition,dose_percent,attention_output_norm,attention_entropy,attention_self_mass,attention_previous_mass,attention_far_mass,attention_nonself_mass,ordinary_attention_active) (Join-Path $Output 'attention-behavior-summary.csv')
    Write-Csv @($dev | Select-Object configuration_id,condition,dose_percent,qk_update_l2,v_update_l2,o_update_l2,ffn_update_l2,norm_update_l2,embedding_head_update_l2,train_finite) (Join-Path $Output 'gradient-update-summary.csv')

    Write-Csv @(
        [pscustomobject]@{control='canonical bitwise reproduction';status='PASS';result='all four parameter and exact anchors'},
        [pscustomobject]@{control='matched homogeneous';status='PASS_NEGATIVE';result='does not stabilize all L19 seeds'},
        [pscustomobject]@{control='same-multiset curriculum';status='PASS';result='first and last both stable at 25 percent'},
        [pscustomobject]@{control='ordinary Attention';status='PASS';result='Q K V O updates nonzero and mean non-self mass above 0.05'},
        [pscustomobject]@{control='L18 seed2';status='PASS_SCOPE';result='25 percent first and last both 144/144'},
        [pscustomobject]@{control='final holdout';status='PASS';result='0 evaluations'}
    ) (Join-Path $Output 'negative-controls.csv')

    Write-Csv @($dev | Where-Object {$_.configuration_id -in @('L19_SEED_2','L18_SEED_2_CONTROL')} | Select-Object configuration_id,depth,condition,dose_percent,teacher_token_exact,free_token_exact,free_sequence_exact,free_nll,ordinary_attention_active) (Join-Path $Output 'depth-control.csv')
    Write-Csv @(
        [pscustomobject]@{hypothesis='homogeneous-only upstream cause';outcome='SUPPORTED';evidence='mixed stabilizes; histogram-matched homogeneous does not'},
        [pscustomobject]@{hypothesis='irrelevant-prefix invariance deficit';outcome='PARTIALLY_SUPPORTED';evidence='target-invariant mixed supervision rescues, but paired-prefix invariance was not measured'},
        [pscustomobject]@{hypothesis='useful context signal required';outcome='NOT_REQUIRED';evidence='no context-dependent target was added'},
        [pscustomobject]@{hypothesis='data amount alone';outcome='REJECTED_AS_SUFFICIENT';evidence='same steps rows and aggregate histogram control fails'},
        [pscustomobject]@{hypothesis='a specific curriculum direction is required';outcome='REJECTED_AT_25_PERCENT';evidence='mixed-first and mixed-last both pass; interleaved retains a residual seed-2 failure'},
        [pscustomobject]@{hypothesis='V O initialization independent of data';outcome='NOT_SUPPORTED_AS_INDEPENDENT_REQUIREMENT';evidence='ordinary Attention with learned V O stabilizes without freeze'},
        [pscustomobject]@{hypothesis='Attention effectively disabled';outcome='REJECTED';evidence='nonzero Q K V O updates output norm and broad non-self mass'}
    ) (Join-Path $Output 'hypothesis-outcomes.csv')
    Write-Csv @(
        [pscustomobject]@{claim='target-invariant mixed-prefix supervision is sufficient';strength='PRIMARY_CAUSAL_FACTOR';evidence='25 percent first and last curricula pass every L19 seed; matched homogeneous fails';scope='L19 seeds 1 2 4 and L18 seed2'},
        [pscustomobject]@{claim='useful past-token supervision is not required for the observed rescue';strength='CAUSAL_SUPPORT';evidence='all intervention labels remain successor of current token';scope='its independent benefit is not identifiable in the official current-token-successor objective'},
        [pscustomobject]@{claim='training amount alone is insufficient';strength='CAUSAL_SUPPORT';evidence='same 320 steps 2560 rows and matched histogram control does not stabilize';scope='tested control'},
        [pscustomobject]@{claim='curriculum order is not required at effective dose';strength='CAUSAL_SUPPORT';evidence='same multiset mixed-first and mixed-last both pass';scope='25 percent dose'},
        [pscustomobject]@{claim='ordinary Attention is retained';strength='MECHANISM_GUARD';evidence='Q K V O update norms nonzero; non-self mass 0.63 to 0.68 in passing runs';scope='aggregate behavior'},
        [pscustomobject]@{claim='25 percent is minimum tested effective dose';strength='BOUNDED_DOSE_RESULT';evidence='12.5 and 6.25 percent fail';scope='three preregistered doses only'}
    ) (Join-Path $Output 'causal-evidence.csv')
    Write-Csv @([pscustomobject]@{diagnosis='Homogeneous-only training is the upstream data condition that leaves mixed-prefix behavior unconstrained. Twenty-five percent target-invariant mixed-prefix supervision prevents the learned ordinary-Attention model from expressing harmful seed-dependent mixed-context behavior. Irrelevant-prefix invariance is the leading explanation, while row-level co-occurrence and gradient-geometry effects remain alternatives.';strength='PRIMARY_CAUSAL_FACTOR';minimum_tested_intervention='80 of 320 training batches mixed; target remains successor of current token';useful_context_required='not required for this rescue; independent benefit not identifiable';curriculum_order_required='direction not required at 25 percent; residual order effect remains';production_change='none'}) (Join-Path $Output 'diagnosis.csv')
    Write-Csv @(
        [pscustomobject]@{uncertainty='true useful-context learning effect';reason='official target is uniquely determined by current token';impact='cannot measure a positive information-theoretic benefit without changing the objective'},
        [pscustomobject]@{uncertainty='dose below 25 percent with another schedule';reason='dose search capped at 25 12.5 and 6.25 percent';impact='25 percent is minimum tested not a global mathematical minimum'},
        [pscustomobject]@{uncertainty='generalization beyond seeds and L18 control';reason='three L19 seeds and one L18 seed tested once';impact='scope is bounded'},
        [pscustomobject]@{uncertainty='invariance versus co-occurrence regularization';reason='matched aggregate histograms do not preserve row-level position co-occurrence or gradient geometry';impact='upstream data intervention is causal but its internal explanation is not unique'},
        [pscustomobject]@{uncertainty='final holdout performance';reason='holdout unopened';impact='not a production selection result'}
    ) (Join-Path $Output 'remaining-uncertainties.csv')
    Write-Csv @(
        [pscustomobject]@{priority=1;candidate='separate production adoption review';purpose='decide whether to adopt diagnostic data intervention';condition='new milestone and final holdout policy'},
        [pscustomobject]@{priority=2;candidate='additional preregistered L19 seeds';purpose='expand scope';condition='only if broader confidence is needed'},
        [pscustomobject]@{priority=3;candidate='synthetic repeated-current-token context-use task';purpose='identify useful-context capacity separately';condition='must be labeled as a different objective'}
    ) (Join-Path $Output 'next-step-candidates.csv')

    $files = foreach ($name in ($allowList | Where-Object {$_ -ne 'manifest.json'})) {
        [ordered]@{path=$name;sha256_normalized_lf=(Get-NormalizedSha256 (Join-Path $Output $name))}
    }
    $sources = foreach ($relative in $sourceFiles) {
        [ordered]@{path=$relative;sha256_normalized_lf=(Get-NormalizedSha256 (Join-Path $repoRoot $relative))}
    }
    $privateAggregates = foreach ($relative in $privateFiles) {
        [ordered]@{aggregate=$relative;sha256_normalized_lf=(Get-NormalizedSha256 (Join-Path $Private $relative))}
    }
    $manifest = [ordered]@{
        schema_version=1;source_commit=$Commit;protocol='CONTEXT_SUPERVISION_STABILITY_V1';
        result_classification='TARGET_INVARIANT_MIXED_PREFIX_SUPERVISION';claim_strength='PRIMARY_CAUSAL_FACTOR';
        major_cycles=3;cpu_trajectory_regenerations=4;short_training_runs=0;full_training_runs=28;
        dataset_intervention_conditions=5;curriculum_conditions=2;parameter_state_interventions=0;
        internal_state_interventions=0;additional_probes=0;final_holdout_opens=0;device_runs=0;
        htp_runs=0;adb_operations=0;ui_operations=0;count_from_one=0;
        train_hash='fnv1a64:5a64ca2d1aa7f29f';ar_validation_hash='fnv1a64:aad785bd4dc88dc9';
        ar_development_hash='fnv1a64:bd464d2a6e192d36';final_holdout_hash='fnv1a64:aa5081e6df658b4a';
        minimum_tested_effective_dose_percent=25;minimum_tested_effective_mixed_batches=80;
        hash_definition='SHA-256 over UTF-8 text after CRLF/CR normalization to LF';
        files=$files;sources=$sources;private_aggregates=$privateAggregates;allow_list=$allowList
    }
    [IO.File]::WriteAllText((Join-Path $Output 'manifest.json'),
        ($manifest | ConvertTo-Json -Depth 8) + "`n", [Text.UTF8Encoding]::new($false))
    Assert-PublicBundle $Output | Out-Null
}

$resolvedPrivate = Join-Path $repoRoot $PrivateRoot
$resolvedOutput = Join-Path $repoRoot $OutputRoot
if ($SelfTest) {
    $tracked = Assert-PublicBundle $resolvedOutput
    if (Test-Path -LiteralPath $resolvedPrivate -PathType Container) {
        Assert-PrivateInputs $resolvedPrivate
        $temp = Join-Path ([IO.Path]::GetTempPath()) ("phonelm-context-supervision-" + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $temp | Out-Null
        try {
            Copy-Item -LiteralPath (Join-Path $resolvedOutput 'README.md') -Destination (Join-Path $temp 'README.md')
            Export-Bundle $resolvedPrivate $temp $tracked.source_commit
            foreach ($name in $allowList) {
                if ((Get-NormalizedSha256 (Join-Path $temp $name)) -ne
                    (Get-NormalizedSha256 (Join-Path $resolvedOutput $name))) {
                    throw "DETERMINISTIC_EXPORT_MISMATCH:$name"
                }
            }
            $negativePrivate = Join-Path $temp 'negative-private'
            New-Item -ItemType Directory -Force -Path $negativePrivate | Out-Null
            Copy-Item -Path (Join-Path $resolvedPrivate '*') -Destination $negativePrivate -Recurse
            $negativeTrainingPath = Join-Path $negativePrivate 'cycle-002/training-results.csv'
            $negativeTraining = @(Import-Csv $negativeTrainingPath)
            ($negativeTraining | Where-Object {
                $_.configuration_id -eq 'L19_SEED_1' -and $_.condition -eq 'CURRICULUM_MIXED_FIRST' -and
                $_.dataset -eq 'AR_DEVELOPMENT_V3'
            }).free_sequence_exact = '21'
            Write-Csv $negativeTraining $negativeTrainingPath
            $rejected = $false
            try { Assert-PrivateScientificEvidence $negativePrivate } catch { $rejected = $true }
            if (-not $rejected) { throw 'NEGATIVE_FULL_STABILITY_MUTATION_ACCEPTED' }

            Remove-Item -LiteralPath $negativePrivate -Recurse -Force
            New-Item -ItemType Directory -Force -Path $negativePrivate | Out-Null
            Copy-Item -Path (Join-Path $resolvedPrivate '*') -Destination $negativePrivate -Recurse
            $negativeSchedulePath = Join-Path $negativePrivate 'cycle-002/schedule-audit.csv'
            $negativeSchedule = @(Import-Csv $negativeSchedulePath)
            $negativeSchedule[0].same_multiset_as_interleaved = 'false'
            Write-Csv $negativeSchedule $negativeSchedulePath
            $rejected = $false
            try { Assert-PrivateScientificEvidence $negativePrivate } catch { $rejected = $true }
            if (-not $rejected) { throw 'NEGATIVE_SCHEDULE_MUTATION_ACCEPTED' }
        } finally {
            Remove-Item -LiteralPath $temp -Recurse -Force
        }
    } else {
        Write-Host 'CONTEXT_SUPERVISION_PRIVATE_REGEN_SKIPPED: tracked hashes schemas controls and safety contract verified'
    }
    Write-Host 'CONTEXT_SUPERVISION_EXPORT_SELF_TEST_PASS'
} else {
    Export-Bundle $resolvedPrivate $resolvedOutput $SourceCommit
    Write-Host 'CONTEXT_SUPERVISION_EXPORT_PASS'
}
