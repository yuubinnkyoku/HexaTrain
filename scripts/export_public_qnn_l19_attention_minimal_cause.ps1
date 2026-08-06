param(
    [switch]$SelfTest,
    [string]$PrivateRoot = "build/private-diagnostics/attention-minimal-cause-goal",
    [string]$OutputRoot = "docs/results/qnn-l19-attention-minimal-cause-2026-08",
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourceCommit) {
    $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim()
}

$allowList = @(
    'README.md', 'configuration.csv', 'evidence-inventory.csv',
    'hypothesis-registry.csv', 'decision-log.csv', 'intervention-protocols.csv',
    'evaluation-interventions.csv', 'training-results.csv', 'negative-controls.csv',
    'seed-comparison.csv', 'depth-control.csv', 'hypothesis-outcomes.csv',
    'causal-evidence.csv', 'diagnosis.csv', 'remaining-uncertainties.csv',
    'next-step-candidates.csv', 'manifest.json'
)

function Get-NormalizedSha256([string]$Path) {
    $text = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return ([Convert]::ToHexString($hash)).ToLowerInvariant()
}

function Write-Csv([object[]]$Rows, [string]$Path) {
    @($Rows) | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

function Assert-PrivateInputs([string]$Root) {
    $required = @(
        'hypothesis-registry.json', 'decisions/decision-001.json',
        'decisions/decision-002.json', 'decisions/decision-003.json',
        'cycle-001/measurement-audit.csv',
        'cycle-001/evaluation-interventions.csv',
        'cycle-002/training-results-cycle2.csv',
        'cycle-003/training-results-cycle3.csv'
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
            throw "MISSING_PRIVATE_INPUT:$relative"
        }
    }
}

function Export-Bundle([string]$Private, [string]$Output, [string]$Commit) {
    Assert-PrivateInputs $Private
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $Output 'README.md') -PathType Leaf)) {
        throw 'README_MUST_EXIST_BEFORE_EXPORT'
    }

    $evaluation = Import-Csv (Join-Path $Private 'cycle-001/evaluation-interventions.csv')
    $cycle2 = Import-Csv (Join-Path $Private 'cycle-002/training-results-cycle2.csv')
    $cycle3 = Import-Csv (Join-Path $Private 'cycle-003/training-results-cycle3.csv')
    $training = @($cycle2) + @($cycle3)

    $configRows = @(
        [pscustomobject]@{configuration_id='L19_SEED_1';depth=19;seed=1;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;train_order='(step-1)%4 phase0';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L19_SEED_2';depth=19;seed=2;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;train_order='(step-1)%4 phase0';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L19_SEED_4';depth=19;seed=4;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;train_order='(step-1)%4 phase0';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'},
        [pscustomobject]@{configuration_id='L18_SEED_2_CONTROL';depth=18;seed=2;tokens=8;dimension=16;ffn_dimension=32;heads=2;steps=320;train_order='(step-1)%4 phase0';evaluation='AR_DEVELOPMENT_V3';final_holdout='HASH_ONLY'}
    )
    Write-Csv $configRows (Join-Path $Output 'configuration.csv')

    $evidence = @(
        [pscustomobject]@{evidence='prior_attention_zero';status='REUSED';result='all four configurations 144/144 tokens and 24/24 sequences';identity='same model seeds dataset and 320-step recipe'},
        [pscustomobject]@{evidence='prior_ffn_zero';status='NEGATIVE_CONTROL';result='26/39/27/23 tokens';identity='parameter-count matched branch'},
        [pscustomobject]@{evidence='cycle1_evaluation_only';status='NEW';result='alpha0 42/38/78 and L18 16';identity='checkpoint preserved'},
        [pscustomobject]@{evidence='cycle2_fixed_self_previous';status='NEW';result='self all 144; previous 140/144/134';identity='QK zero and moments zero'},
        [pscustomobject]@{evidence='cycle3_uniform_qk_vo';status='NEW';result='uniform 31/24/78; QK freeze 40/75/84; VO freeze 132/110/134';identity='frozen subgroup exact'},
        [pscustomobject]@{evidence='canonical_parameter_state';status='PASS';result='independent parameter m v bitwise parity';identity='registry shape float-byte hash'},
        [pscustomobject]@{evidence='historical_param_content_hash';status='INVALID_FOR_CONTENT_IDENTITY';result='zero/nonzero support-mask only';identity='quality aggregates unaffected'}
    )
    Write-Csv $evidence (Join-Path $Output 'evidence-inventory.csv')

    $registry = Get-Content -Raw (Join-Path $Private 'hypothesis-registry.json') | ConvertFrom-Json
    $outcomes = @{
        H1='REJECTED'; H2='REJECTED'; H3='REJECTED_AS_NECESSARY';
        H4='PARTLY_SUPPORTED'; H5='SUPPORTED'; H6='REJECTED';
        H7='PARTLY_SUPPORTED_NOT_FACTORIZED'; H8='REJECTED'
    }
    $hypRows = foreach ($item in $registry.hypotheses) {
        [pscustomobject]@{id=$item.id;hypothesis=$item.hypothesis;prior_support=($item.supporting_evidence -join '; ');prior_counterevidence=($item.counterevidence -join '; ');prediction=$item.prediction;cheapest_test=$item.cheapest_test;negative_control=$item.negative_control;outcome=$outcomes[$item.id]}
    }
    Write-Csv $hypRows (Join-Path $Output 'hypothesis-registry.csv')

    $decisionRows = foreach ($number in 1..3) {
        $decision = Get-Content -Raw (Join-Path $Private ("decisions/decision-{0:d3}.json" -f $number)) | ConvertFrom-Json
        [pscustomobject]@{decision=$number;made_before_results=$decision.made_before_results;experiment=$decision.experiment;reason=$decision.reason;negative_controls=($decision.negative_controls -join '; ');full_training_budget=$decision.budget.full_training_runs;internal_intervention_budget=$decision.budget.internal_state_interventions}
    }
    Write-Csv $decisionRows (Join-Path $Output 'decision-log.csv')

    $protocols = @(
        [pscustomobject]@{intervention='EVAL_ATTENTION_ALPHA';stage='evaluation only';forward='scale every Wo';trained_groups='none';fixed_state='checkpoint parameter and Adam state unchanged'},
        [pscustomobject]@{intervention='FIXED_SELF';stage='all training and evaluation';forward='P[row,row]=1';trained_groups='V O LN1 LN2 FFN embedding head';fixed_state='Q K params and moments zero'},
        [pscustomobject]@{intervention='FIXED_PREVIOUS';stage='all training and evaluation';forward='P[0,0]=1; P[row,row-1]=1';trained_groups='V O LN1 LN2 FFN embedding head';fixed_state='Q K params and moments zero'},
        [pscustomobject]@{intervention='FIXED_UNIFORM_CAUSAL';stage='all training and evaluation';forward='P[row,col]=1/(row+1) for col<=row';trained_groups='V O LN1 LN2 FFN embedding head';fixed_state='Q K params and moments zero'},
        [pscustomobject]@{intervention='FREEZE_QK_INITIAL';stage='all training';forward='normal learned Attention with initial Q K';trained_groups='V O and all non-QK';fixed_state='Q K params initial and moments zero'},
        [pscustomobject]@{intervention='FREEZE_VO_INITIAL';stage='all training';forward='normal learned Attention with initial V O';trained_groups='Q K and all non-VO';fixed_state='V O params initial and moments zero'}
    )
    Write-Csv $protocols (Join-Path $Output 'intervention-protocols.csv')

    $evaluationPublic = $evaluation | Select-Object configuration_id,depth,seed,intervention,checkpoint_preserved,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,margin_q10,minimum_teacher_margin,tie_count,all_finite
    Write-Csv $evaluationPublic (Join-Path $Output 'evaluation-interventions.csv')
    $trainingPublic = $training | Select-Object configuration_id,depth,seed,intervention,steps,train_finite,final_train_loss,frozen_scope_pass,teacher_token_exact,teacher_token_total,teacher_nll,free_token_exact,free_token_total,free_sequence_exact,free_sequence_total,free_nll,median_first_error,margin_q10,all_finite
    Write-Csv $trainingPublic (Join-Path $Output 'training-results.csv')

    $controls = @(
        [pscustomobject]@{control='alpha1 no-op';status='PASS';result='parameter and forward unchanged';purpose='intervention wrapper identity'},
        [pscustomobject]@{control='independent canonical loop';status='PASS';result='parameter m v bitwise equal for four configurations';purpose='deterministic rerun'},
        [pscustomobject]@{control='independent evaluator';status='PASS';result='case exact sequence and first-error equal';purpose='metric validity'},
        [pscustomobject]@{control='FFN zero';status='PASS_NEGATIVE';result='does not recover';purpose='capacity-matched branch control'},
        [pscustomobject]@{control='L18 seed2';status='PASS_SCOPE';result='same qualitative pattern; VO freeze fully recovers';purpose='depth control'},
        [pscustomobject]@{control='final holdout';status='PASS';result='0 opens';purpose='selection safety'}
    )
    Write-Csv $controls (Join-Path $Output 'negative-controls.csv')

    $seedRows = foreach ($id in @('L19_SEED_1','L19_SEED_2','L19_SEED_4')) {
        $baseline = $evaluation | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'LEARNED_ALPHA_1_NOOP'}
        $self = $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'TRAIN_FIXED_SELF'}
        $previous = $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'TRAIN_FIXED_PREVIOUS'}
        $uniform = $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'TRAIN_FIXED_UNIFORM_CAUSAL'}
        $qk = $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'FREEZE_QK_INITIAL'}
        $vo = $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq 'FREEZE_VO_INITIAL'}
        [pscustomobject]@{configuration_id=$id;baseline_tokens=$baseline.free_token_exact;fixed_self_tokens=$self.free_token_exact;fixed_previous_tokens=$previous.free_token_exact;uniform_tokens=$uniform.free_token_exact;qk_freeze_tokens=$qk.free_token_exact;vo_freeze_tokens=$vo.free_token_exact;fixed_self_sequences=$self.free_sequence_exact;uniform_sequences=$uniform.free_sequence_exact;vo_freeze_sequences=$vo.free_sequence_exact}
    }
    Write-Csv $seedRows (Join-Path $Output 'seed-comparison.csv')

    $depthRows = foreach ($id in @('L19_SEED_2','L18_SEED_2_CONTROL')) {
        foreach ($mode in @('LEARNED_ALPHA_1_NOOP','TRAIN_FIXED_SELF','TRAIN_FIXED_PREVIOUS','TRAIN_FIXED_UNIFORM_CAUSAL','FREEZE_QK_INITIAL','FREEZE_VO_INITIAL')) {
            $row = if ($mode -eq 'LEARNED_ALPHA_1_NOOP') { $evaluation | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq $mode} } else { $training | Where-Object {$_.configuration_id -eq $id -and $_.intervention -eq $mode} }
            [pscustomobject]@{configuration_id=$id;intervention=$mode;free_token_exact=$row.free_token_exact;free_sequence_exact=$row.free_sequence_exact;token_total=144;sequence_total=24}
        }
    }
    Write-Csv $depthRows (Join-Path $Output 'depth-control.csv')

    $hypOutcomeRows = @(
        [pscustomobject]@{hypothesis='final Attention forward alone';outcome='REJECTED';evidence='evaluation alpha0 does not recover'},
        [pscustomobject]@{hypothesis='simple residual magnitude';outcome='REJECTED';evidence='dose response reverses by seed and L18'},
        [pscustomobject]@{hypothesis='learned content-dependent QK required';outcome='REJECTED';evidence='fixed uniform and QK freeze fail without QK learning'},
        [pscustomobject]@{hypothesis='broad distractor mixing';outcome='SUPPORTED';evidence='self full recovery; previous near stable; uniform strong failure'},
        [pscustomobject]@{hypothesis='V O adaptation';outcome='MAJOR_AMPLIFIER';evidence='VO freeze improves every L19 and fully recovers L18'},
        [pscustomobject]@{hypothesis='V O alone';outcome='REJECTED_AS_SOLE';evidence='L19 VO freeze remains incomplete'},
        [pscustomobject]@{hypothesis='initial optimizer state';outcome='STRUCTURALLY_REJECTED';evidence='all seeds start m v exactly zero'},
        [pscustomobject]@{hypothesis='generic capacity removal';outcome='REJECTED';evidence='parameter-count matched FFN zero fails'}
    )
    Write-Csv $hypOutcomeRows (Join-Path $Output 'hypothesis-outcomes.csv')

    $causal = @(
        [pscustomobject]@{claim='training trajectory required';strength='CAUSAL_SUPPORT';evidence='eval-only off fails; training-time fixed self succeeds';scope='L19 seeds 1 2 4 and L18 seed2'},
        [pscustomobject]@{claim='cross-token mixing contributes';strength='CAUSAL_SUPPORT';evidence='self complete; previous incomplete for L19 seeds1 and4';scope='seed-specific residual'},
        [pscustomobject]@{claim='broad distractor mixing is harmful';strength='MAJOR_FACTOR';evidence='uniform much worse than previous for every L19 seed';scope='all tested seeds and L18'},
        [pscustomobject]@{claim='learned QK not necessary';strength='CAUSAL_SUPPORT';evidence='uniform QK zero and QK freeze both fail';scope='all tested seeds'},
        [pscustomobject]@{claim='VO updates amplify';strength='CAUSAL_SUPPORT';evidence='VO freeze large recovery versus QK freeze';scope='all tested seeds; incomplete for L19'},
        [pscustomobject]@{claim='broad mixing plus VO/rest coadaptation';strength='MAJOR_FACTOR';evidence='combined pattern and subgroup interventions';scope='all L19 seeds with L18 control'}
    )
    Write-Csv $causal (Join-Path $Output 'causal-evidence.csv')

    Write-Csv @([pscustomobject]@{diagnosis='Broad irrelevant-token context mixing plus seed-dependent V/O and rest-model co-adaptation formed during training; free-running argmax amplifies the remaining ranking differences.';strength='MAJOR_FACTOR';qk_learning_required='false';final_forward_alone_sufficient='false';production_change='none'}) (Join-Path $Output 'diagnosis.csv')
    Write-Csv @(
        [pscustomobject]@{uncertainty='initial V O versus non-Attention initialization contribution';reason='factorial not run after 24 full-run budget reached';impact='cannot assign seed source to one initial group'},
        [pscustomobject]@{uncertainty='residual L19 error under VO freeze';reason='LayerNorm and non-Attention updates not separately frozen';impact='VO is major amplifier not sole root'},
        [pscustomobject]@{uncertainty='final holdout performance';reason='holdout unopened';impact='diagnosis is not a production selection result'}
    ) (Join-Path $Output 'remaining-uncertainties.csv')
    Write-Csv @(
        [pscustomobject]@{priority=1;candidate='3x3 L19 Attention versus non-Attention initialization factorial';purpose='separate initial group contribution';condition='new explicit full-run budget'},
        [pscustomobject]@{priority=2;candidate='freeze V and O separately';purpose='separate representation mixing from residual projection';condition='only if finer mechanism is required'},
        [pscustomobject]@{priority=3;candidate='LayerNorm and V O reciprocal state intervention';purpose='localize residual co-adaptation';condition='diagnostic only'}
    ) (Join-Path $Output 'next-step-candidates.csv')

    $publicFiles = @($allowList | Where-Object {$_ -ne 'manifest.json'})
    $fileEntries = foreach ($name in $publicFiles) {
        $path = Join-Path $Output $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "MISSING_PUBLIC_FILE:$name" }
        [ordered]@{path=$name;sha256_normalized_lf=(Get-NormalizedSha256 $path)}
    }
    $sourceFiles = @(
        'host_tests/attention_minimal_cause_lib.h',
        'host_tests/attention_minimal_cause.cpp',
        'scripts/run_l19_attention_minimal_cause.ps1',
        'scripts/export_public_qnn_l19_attention_minimal_cause.ps1'
    )
    $sourceEntries = foreach ($relative in $sourceFiles) {
        [ordered]@{path=$relative;sha256_normalized_lf=(Get-NormalizedSha256 (Join-Path $repoRoot $relative))}
    }
    $privateFiles = @(
        'cycle-001/measurement-audit.csv','cycle-001/evaluation-interventions.csv',
        'cycle-002/training-results-cycle2.csv','cycle-003/training-results-cycle3.csv'
    )
    $privateEntries = foreach ($relative in $privateFiles) {
        [ordered]@{aggregate=$relative;sha256_normalized_lf=(Get-NormalizedSha256 (Join-Path $Private $relative))}
    }
    $manifest = [ordered]@{
        schema_version=1; source_commit=$Commit; protocol='ATTENTION_MINIMAL_CAUSE_V1';
        result_classification='BROAD_DISTRACTOR_MIXING_PLUS_VO_REST_COADAPTATION';
        claim_strength='MAJOR_FACTOR'; major_cycles=3; cpu_trajectory_regenerations=4;
        short_training_runs=0; full_training_runs=24; parameter_state_interventions=20;
        internal_state_interventions=28; additional_probes=0; final_holdout_opens=0;
        device_runs=0; htp_runs=0; adb_operations=0; ui_operations=0; count_from_one=0;
        ar_development_hash='fnv1a64:bd464d2a6e192d36';
        final_holdout_hash='fnv1a64:aa5081e6df658b4a';
        historical_parameter_hash_correction='fnv1aParams is zero-support-mask only';
        hash_definition='SHA-256 over UTF-8 text after CRLF/CR normalization to LF';
        files=$fileEntries; sources=$sourceEntries; private_aggregates=$privateEntries;
        allow_list=$allowList
    }
    $json = $manifest | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText((Join-Path $Output 'manifest.json'), $json + "`n", [Text.UTF8Encoding]::new($false))

    $unexpected = @(Get-ChildItem -LiteralPath $Output -File | Where-Object {$_.Name -notin $allowList})
    if ($unexpected.Count -gt 0) { throw "PUBLIC_ALLOW_LIST_VIOLATION" }
    foreach ($file in Get-ChildItem -LiteralPath $Output -File) {
        $text = Get-Content -LiteralPath $file.FullName -Raw
        if ($text -match '[A-Za-z]:[\\/]' -or $text -match 'build[\\/]private-diagnostics') {
            throw "PRIVATE_DATA_SCAN:$($file.Name)"
        }
    }
}

$resolvedPrivate = Join-Path $repoRoot $PrivateRoot
$resolvedOutput = Join-Path $repoRoot $OutputRoot
if ($SelfTest) {
    $trackedManifest = Get-Content -Raw (Join-Path $resolvedOutput 'manifest.json') | ConvertFrom-Json
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("phonelm-attention-minimal-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    try {
        Copy-Item -LiteralPath (Join-Path $resolvedOutput 'README.md') -Destination (Join-Path $temp 'README.md')
        Export-Bundle $resolvedPrivate $temp $trackedManifest.source_commit
        foreach ($name in $allowList) {
            if ((Get-NormalizedSha256 (Join-Path $temp $name)) -ne
                (Get-NormalizedSha256 (Join-Path $resolvedOutput $name))) {
                throw "DETERMINISTIC_EXPORT_MISMATCH:$name"
            }
        }
        Write-Host 'ATTENTION_MINIMAL_CAUSE_EXPORT_SELF_TEST_PASS'
    } finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
} else {
    Export-Bundle $resolvedPrivate $resolvedOutput $SourceCommit
    Write-Host 'ATTENTION_MINIMAL_CAUSE_EXPORT_PASS'
}
