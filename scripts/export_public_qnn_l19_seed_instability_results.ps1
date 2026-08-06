# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [switch]$SelfTest,
    [string]$PrivateRoot,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([string]::IsNullOrWhiteSpace($PrivateRoot)) {
    $PrivateRoot = Join-Path $repoRoot 'build\reports\qnn-l19-seed-instability-root-cause'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot 'docs\results\qnn-l19-seed-instability-root-cause-2026-08'
}
$PrivateRoot = [IO.Path]::GetFullPath($PrivateRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$docsResultsRoot = ([IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results'))).TrimEnd('\', '/') +
    [IO.Path]::DirectorySeparatorChar
$buildReportsRoot = ([IO.Path]::GetFullPath((Join-Path $repoRoot 'build\reports'))).TrimEnd('\', '/') +
    [IO.Path]::DirectorySeparatorChar
$allowedOutputRoots = @($docsResultsRoot, $buildReportsRoot)
$outputScopeAllowed = $false
foreach ($allowedRoot in $allowedOutputRoots) {
    if ($OutputRoot.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $outputScopeAllowed = $true
        break
    }
}
if (-not $outputScopeAllowed) {
    throw "OUTPUT_ROOT_OUTSIDE_ALLOWED_SCOPE: $OutputRoot"
}

$publicFiles = @(
    'README.md',
    'configuration.csv',
    'evidence-inventory.csv',
    'hypothesis-registry.csv',
    'decision-log.csv',
    'measurement-audit.csv',
    'data-structure.csv',
    'trajectory-divergence.csv',
    'gradient-and-update-summary.csv',
    'intervention-summary.csv',
    'negative-controls.csv',
    'seed-comparison.csv',
    'depth-control.csv',
    'hypothesis-outcomes.csv',
    'causal-evidence.csv',
    'diagnosis.csv',
    'remaining-uncertainties.csv',
    'next-step-candidates.csv'
)

function Export-StableCsv([object[]]$Rows, [string]$Path) {
    @($Rows) | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

function Get-Sha256([string]$Path) {
    $text = [IO.File]::ReadAllText($Path).Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return ([BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
}

function Assert-NoPrivateData([string]$Root) {
    $patterns = @(
        '(?i)[a-z]:\\',
        '(?i)raw_(checkpoint|parameter|optimizer|hidden|logit|attention|gradient)',
        '(?i)adb[_ -]?(endpoint|serial)',
        '(?i)private[_ -]?run[_ -]?id',
        '(?i)logcat',
        '(?i)\.apk\b'
    )
    foreach ($file in Get-ChildItem -LiteralPath $Root -File) {
        if ($file.Extension -notin @('.csv', '.md', '.json')) { continue }
        $text = Get-Content -Raw -LiteralPath $file.FullName
        foreach ($pattern in $patterns) {
            if ($text -match $pattern) {
                throw "PRIVATE_DATA_PATTERN:$($file.Name):$pattern"
            }
        }
    }
}

function Assert-PublicBundle([string]$Root) {
    $manifestPath = Join-Path $Root 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'MANIFEST_MISSING' }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.result_classification -ne 'COMPOUND_ATTENTION_CONTEXT_SENSITIVITY_ROOT_CAUSE') {
        throw 'MANIFEST_CLASSIFICATION'
    }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or
        $manifest.htp_runs -ne 0 -or $manifest.count_from_one -ne 0) {
        throw 'MANIFEST_SAFETY_COUNTS'
    }
    foreach ($item in $manifest.files) {
        $path = Join-Path $Root $item.name
        if (-not (Test-Path -LiteralPath $path)) { throw "PUBLIC_FILE_MISSING:$($item.name)" }
        if ((Get-Sha256 $path) -ne $item.sha256) { throw "PUBLIC_HASH_MISMATCH:$($item.name)" }
    }
    foreach ($source in $manifest.production_sources) {
        $path = Join-Path $repoRoot $source.name
        if ((Get-Sha256 $path) -ne $source.sha256) { throw "SOURCE_HASH_MISMATCH:$($source.name)" }
    }
    $requiredSourceNames = @(
        'app\src\main\cpp\tiny_language_model_cpu.cpp',
        'app\src\main\cpp\autoregressive_validation.h',
        'host_tests\depth_quality_lib.h',
        'host_tests\readout_probe_lib.h',
        'host_tests\critical_margin_objective_lib.h',
        'host_tests\seed_instability_diagnostics_lib.h',
        'host_tests\seed_instability_diagnostics.cpp',
        'scripts\run_l19_seed_instability_diagnostics.ps1',
        'scripts\export_public_qnn_l19_seed_instability_results.ps1',
        'docs\results\qnn-l19-critical-margin-stabilization-2026-08\gradient-attribution.csv',
        'docs\results\qnn-l19-readout-representation-diagnosis-2026-08\trajectory-anchors.csv'
    )
    foreach ($name in $requiredSourceNames) {
        if (@($manifest.production_sources | Where-Object { $_.name -eq $name }).Count -ne 1) {
            throw "REQUIRED_SOURCE_IDENTITY:$name"
        }
    }
    $requiredAggregateNames = @('measurement-audit.csv', 'data-structure.csv',
        'context-shift.csv', 'optimization-interventions.csv', 'branch-ablation.csv')
    if ($manifest.private_aggregate_hashes.Count -ne $requiredAggregateNames.Count) {
        throw 'PRIVATE_AGGREGATE_HASH_COUNT'
    }
    foreach ($name in $requiredAggregateNames) {
        $item = @($manifest.private_aggregate_hashes | Where-Object { $_.name -eq $name })
        if ($item.Count -ne 1 -or $item[0].sha256 -notmatch '^[0-9a-f]{64}$') {
            throw "PRIVATE_AGGREGATE_HASH_INTEGRITY:$name"
        }
    }
    $identityHashes = @($manifest.private_identity_hashes)
    if ($identityHashes.Count -ne 1 -or
        $identityHashes[0].name -ne 'qnn-probe-optimization-audit/manifest.csv' -or
        $identityHashes[0].sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'PRIVATE_IDENTITY_HASH_INTEGRITY'
    }
    $liveIdentityPath = Join-Path $repoRoot 'build\reports\qnn-probe-optimization-audit\manifest.csv'
    if (-not (Test-Path -LiteralPath $liveIdentityPath) -or
        (Get-Sha256 $liveIdentityPath) -ne $identityHashes[0].sha256) {
        throw 'PRIVATE_IDENTITY_SOURCE_HASH_MISMATCH'
    }
    $diagnosis = Import-Csv -LiteralPath (Join-Path $Root 'diagnosis.csv')
    if ($diagnosis.Count -ne 1 -or $diagnosis.root_cause -ne 'COMPOUND_ATTENTION_CONTEXT_SENSITIVITY_ROOT_CAUSE' -or
        $diagnosis.attention_zero_all_exact -ne 'true' -or $diagnosis.ffn_negative_control -ne 'true') {
        throw 'DIAGNOSIS_INTEGRITY'
    }
    $seeds = Import-Csv -LiteralPath (Join-Path $Root 'seed-comparison.csv')
    $expectedFfn = [ordered]@{
        L19_SEED_1 = '26'; L19_SEED_2 = '39'; L19_SEED_4 = '27';
        L18_SEED_2_CONTROL = '23'
    }
    if ($seeds.Count -ne $expectedFfn.Count) { throw 'SEED_COUNT_INTEGRITY' }
    foreach ($id in $expectedFfn.Keys) {
        $matching = @($seeds | Where-Object { $_.configuration_id -eq $id })
        if ($matching.Count -ne 1) { throw "CONFIGURATION_ID_INTEGRITY:$id" }
        $row = $matching[0]
        if ($row.attention_zero_mixed_free_exact -ne '144' -or
            $row.attention_zero_mixed_free_total -ne '144') {
            throw "ATTENTION_CAUSAL_ANCHOR:$($row.configuration_id)"
        }
        if ($row.ffn_zero_mixed_free_exact -ne $expectedFfn[$id] -or
            $row.ffn_zero_mixed_free_total -ne '144') {
            throw "FFN_NEGATIVE_CONTROL_ANCHOR:$($row.configuration_id)"
        }
    }
    foreach ($row in (Import-Csv -LiteralPath (Join-Path $Root 'intervention-summary.csv'))) {
        if ($row.train_loss_finite -ne 'true' -or $row.scored_finite -ne 'true') {
            throw "INTERVENTION_FINITE_SCOPE:$($row.configuration_id):$($row.intervention)"
        }
    }
    $publicData = @(Import-Csv -LiteralPath (Join-Path $Root 'data-structure.csv'))
    if ($publicData.Count -ne 18 -or
        @($publicData | Where-Object { $_.ambiguous_contexts -ne '0' -or $_.ambiguous_occurrences -ne '0' }).Count -ne 0) {
        throw 'PUBLIC_DATA_STRUCTURE_SCHEMA'
    }
    foreach ($dataset in @('AR_VALIDATION_V3', 'AR_DEVELOPMENT_V3',
            'MARGIN_CALIBRATION_V1', 'MARGIN_DEVELOPMENT_V1')) {
        $full = @($publicData | Where-Object { $_.dataset -eq $dataset -and $_.context_kind -eq 'FULL_CAUSAL_PREFIX' })
        if ($full.Count -ne 1 -or $full[0].occurrences -ne '144') {
            throw "PUBLIC_FULL_PREFIX_ANCHOR:$dataset"
        }
    }
    Assert-NoPrivateData $Root
}

if ($SelfTest) {
    Assert-PublicBundle $OutputRoot
    $negativeRoot = Join-Path $repoRoot 'build\reports\qnn-l19-seed-instability-export-selftest'
    if (Test-Path -LiteralPath $negativeRoot) { Remove-Item -Recurse -Force -LiteralPath $negativeRoot }
    New-Item -ItemType Directory -Force -Path $negativeRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $OutputRoot 'diagnosis.csv') -Destination $negativeRoot
    Add-Content -LiteralPath (Join-Path $negativeRoot 'diagnosis.csv') -Value 'C:\private\checkpoint.bin'
    $rejected = $false
    try { Assert-NoPrivateData $negativeRoot } catch { $rejected = $true }
    if (-not $rejected) { throw 'PRIVATE_SCAN_NEGATIVE_TEST_INEFFECTIVE' }
    Remove-Item -Recurse -Force -LiteralPath $negativeRoot
    Write-Host 'L19 seed-instability public exporter self-test PASS'
    exit 0
}

$requiredPrivate = @('measurement-audit.csv', 'data-structure.csv', 'context-shift.csv',
    'optimization-interventions.csv', 'branch-ablation.csv')
foreach ($name in $requiredPrivate) {
    if (-not (Test-Path -LiteralPath (Join-Path $PrivateRoot $name))) {
        throw "PRIVATE_AGGREGATE_MISSING:$name"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$context = Import-Csv -LiteralPath (Join-Path $PrivateRoot 'context-shift.csv')
$optimization = Import-Csv -LiteralPath (Join-Path $PrivateRoot 'optimization-interventions.csv')
$branch = Import-Csv -LiteralPath (Join-Path $PrivateRoot 'branch-ablation.csv')
$measurement = Import-Csv -LiteralPath (Join-Path $PrivateRoot 'measurement-audit.csv')
$data = Import-Csv -LiteralPath (Join-Path $PrivateRoot 'data-structure.csv')
$gradientSource = Join-Path $repoRoot 'docs\results\qnn-l19-critical-margin-stabilization-2026-08\gradient-attribution.csv'
$gradient = Import-Csv -LiteralPath $gradientSource
$trajectoryAnchorSource = Join-Path $repoRoot 'docs\results\qnn-l19-readout-representation-diagnosis-2026-08\trajectory-anchors.csv'
$parameterIdentitySource = Join-Path $repoRoot 'build\reports\qnn-probe-optimization-audit\manifest.csv'
if (-not (Test-Path -LiteralPath $parameterIdentitySource)) { throw 'PARAMETER_IDENTITY_SOURCE_MISSING' }
$parameterIdentity = Import-Csv -LiteralPath $parameterIdentitySource
$parameterHashes = [ordered]@{}
foreach ($id in @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')) {
    $row = @($parameterIdentity | Where-Object key -eq ("param_content_hash_$id"))
    if ($row.Count -ne 1 -or $row[0].value -notmatch '^fnv1a64:[0-9a-f]{16}$') {
        throw "PARAMETER_IDENTITY_INTEGRITY:$id"
    }
    $parameterHashes[$id] = $row[0].value
}

function Assert-PrivateAggregates {
    $legacy = @($measurement | Where-Object measurement -eq 'legacy_train_current_token_exact')
    $corrected = @($measurement | Where-Object measurement -eq 'corrected_train_current_token_exact')
    if ($legacy.Count -ne 1 -or $legacy[0].observed -ne '28' -or
        $corrected.Count -ne 1 -or $corrected[0].observed -ne '32') {
        throw 'PRIVATE_MEASUREMENT_ANCHOR'
    }
    if ($data.Count -ne 18 -or
        @($data | Where-Object { $_.ambiguous_contexts -ne '0' -or $_.ambiguous_occurrences -ne '0' }).Count -ne 0) {
        throw 'PRIVATE_DATA_AMBIGUITY_ANCHOR'
    }
    foreach ($dataset in @('AR_VALIDATION_V3', 'AR_DEVELOPMENT_V3',
            'MARGIN_CALIBRATION_V1', 'MARGIN_DEVELOPMENT_V1')) {
        $full = @($data | Where-Object { $_.dataset -eq $dataset -and $_.context_kind -eq 'FULL_CAUSAL_PREFIX' })
        if ($full.Count -ne 1 -or $full[0].occurrences -ne '144') {
            throw "PRIVATE_FULL_PREFIX_ANCHOR:$dataset"
        }
    }
    $baselineExpected = [ordered]@{
        L19_SEED_1 = '30'; L19_SEED_2 = '63'; L19_SEED_4 = '46';
        L18_SEED_2_CONTROL = '65'
    }
    if ($context.Count -ne 8) { throw 'PRIVATE_CONTEXT_ROW_COUNT' }
    foreach ($id in $baselineExpected.Keys) {
        $row = @($context | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'AR_DEVELOPMENT_V3' })
        if ($row.Count -ne 1 -or $row[0].free_token_exact -ne $baselineExpected[$id] -or $row[0].all_finite -ne 'true') {
            throw "PRIVATE_CONTEXT_ANCHOR:$id"
        }
    }
    $optimizationExpected = [ordered]@{
        'L19_SEED_1|OUTPUT_FREEZE_AFTER_STEP24'='31'; 'L19_SEED_1|OUTPUT_MOMENTS_RESET_AT_STEP24'='32'; 'L19_SEED_1|ATTENTION_MOMENTS_RESET_AT_STEP24'='30';
        'L19_SEED_2|OUTPUT_FREEZE_AFTER_STEP24'='33'; 'L19_SEED_2|OUTPUT_MOMENTS_RESET_AT_STEP24'='56'; 'L19_SEED_2|ATTENTION_MOMENTS_RESET_AT_STEP24'='59';
        'L19_SEED_4|OUTPUT_FREEZE_AFTER_STEP24'='48'; 'L19_SEED_4|OUTPUT_MOMENTS_RESET_AT_STEP24'='48'; 'L19_SEED_4|ATTENTION_MOMENTS_RESET_AT_STEP24'='41';
        'L18_SEED_2_CONTROL|OUTPUT_FREEZE_AFTER_STEP24'='61'; 'L18_SEED_2_CONTROL|OUTPUT_MOMENTS_RESET_AT_STEP24'='51'; 'L18_SEED_2_CONTROL|ATTENTION_MOMENTS_RESET_AT_STEP24'='43'
    }
    if ($optimization.Count -ne 24) { throw 'PRIVATE_OPTIMIZATION_ROW_COUNT' }
    foreach ($key in $optimizationExpected.Keys) {
        $parts = $key.Split('|')
        $row = @($optimization | Where-Object { $_.configuration_id -eq $parts[0] -and $_.intervention -eq $parts[1] -and $_.dataset -eq 'AR_DEVELOPMENT_V3' })
        if ($row.Count -ne 1 -or $row[0].free_token_exact -ne $optimizationExpected[$key] -or
            $row[0].train_finite -ne 'true' -or $row[0].all_finite -ne 'true') {
            throw "PRIVATE_OPTIMIZATION_ANCHOR:$key"
        }
    }
    $branchExpected = [ordered]@{
        'L19_SEED_1|ATTENTION_BRANCH_ZERO'='144'; 'L19_SEED_1|FFN_BRANCH_ZERO'='26';
        'L19_SEED_2|ATTENTION_BRANCH_ZERO'='144'; 'L19_SEED_2|FFN_BRANCH_ZERO'='39';
        'L19_SEED_4|ATTENTION_BRANCH_ZERO'='144'; 'L19_SEED_4|FFN_BRANCH_ZERO'='27';
        'L18_SEED_2_CONTROL|ATTENTION_BRANCH_ZERO'='144'; 'L18_SEED_2_CONTROL|FFN_BRANCH_ZERO'='23'
    }
    if ($branch.Count -ne 16) { throw 'PRIVATE_BRANCH_ROW_COUNT' }
    foreach ($key in $branchExpected.Keys) {
        $parts = $key.Split('|')
        $row = @($branch | Where-Object { $_.configuration_id -eq $parts[0] -and $_.intervention -eq $parts[1] -and $_.dataset -eq 'AR_DEVELOPMENT_V3' })
        if ($row.Count -ne 1 -or $row[0].free_token_exact -ne $branchExpected[$key] -or
            $row[0].train_finite -ne 'true' -or $row[0].branch_exactly_zero -ne 'true' -or $row[0].all_finite -ne 'true') {
            throw "PRIVATE_BRANCH_ANCHOR:$key"
        }
    }
}
Assert-PrivateAggregates

$readme = @'
# L19 seed-instability root-cause investigation, August 2026

> **Follow-up (2026-08):** Ordinary learned Attention becomes stable across
> L19 seeds 1/2/4 when 80 of 320 batches use target-invariant mixed prefixes;
> a histogram-matched homogeneous control does not. See
> `../qnn-l19-context-supervision-stability-2026-08/README.md`.

> **Correction / follow-up (2026-08):** A later minimal-mechanism audit found
> that broad distractor-token mixing plus V/O/rest co-adaptation, not learned Q/K
> selection by itself, is the smaller causal mechanism. See
> `../qnn-l19-attention-minimal-cause-2026-08/README.md`. Historical
> `param_content_hash` fields were produced by `fnv1aParams`, which hashes only
> zero/nonzero support rather than float contents. Quality aggregates are
> unchanged, but those fields are not exact checkpoint-content identities.

This host-only investigation re-audits the finite seed-dependent generation
quality of T8/D16/FFN32/L19/H2 without opening AR_FINAL_HOLDOUT_V3. The causal
result is a compound root cause: the target rule is a deterministic
current-token successor, training repeats only homogeneous phase-0 contexts,
and, under that canonical homogeneous-only training distribution, the deep
Attention path causally mediates the seed-dependent failures on mixed distractor
prefixes. This is not an unconditional necessity or sufficiency claim about
Attention across training distributions. Argmax feedback then amplifies local
ranking differences into sequence-exact differences. A learned context
shortcut is the leading mechanism candidate, but branch removal also changes
gradient competition and normalization, so that finer mechanism is not proven.

The key intervention is diagnostic, not a production recommendation. With all
Attention residual branches structurally zero throughout the same 320-step CPU
training recipe, L19 seeds 1/2/4 and the L18 seed-2 scope control all reach
144/144 mixed-development free-running tokens and 24/24 exact sequences. The
same-size FFN-zero negative control worsens every configuration (23-39/144).
Late output-head freeze/moment reset does not rescue the L19 seeds consistently,
so output-head ranking drift is a seed-specific downstream compensator rather
than the common root cause.

All baseline and intervention scores use AR_DEVELOPMENT_V3. The MARGIN
calibration/development partitions are audited separately for context
uniqueness and are not mixed with these AR scores. Branch-run finite flags
cover training loss and scored logits/probabilities; they are not an exhaustive
hidden-state, gradient, or optimizer-moment tensor audit.

The evaluator audit found no target ambiguity: current-token and previous-two-
token mappings are unique in TRAIN, AR_VALIDATION_V3, AR_DEVELOPMENT_V3,
MARGIN_CALIBRATION_V1, and MARGIN_DEVELOPMENT_V1.
It also found and corrected diagnostic foundations for future use: historical
probe TRAIN rows included four contract-conflicting synthetic rows (28/32
instead of 32/32), historical cross-seed context swaps were malformed and
lacked a true row-wise identity, and contribution norm/cosine aggregates used
one row. Those historical swap/aggregate claims are excluded here. Direct head
AR metrics, canonical trajectory anchors, output-projection rank/transport,
and the new branch intervention do not depend on those defective measurements.

No device, HTP, QAIRT, ADB, UI, or final-holdout execution was performed. Raw
checkpoints, parameters, optimizer states, hidden states, logits, attention
matrices, gradients, local paths, and private identifiers are not published.
'@
Set-Content -LiteralPath (Join-Path $OutputRoot 'README.md') -Value $readme -Encoding utf8

$configs = foreach ($id in @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')) {
    $baseline = $context | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'AR_DEVELOPMENT_V3' }
    [pscustomobject]@{ configuration_id=$id; depth=$baseline.depth; seed=$baseline.seed;
        steps=320; vocabulary=32; tokens=8; dimension=16; ffn_dimension=32; heads=2;
        train_schedule='HOMOGENEOUS_PHASE0_FIXED_ROUND_ROBIN'; evaluation='AR_DEVELOPMENT_V3';
        final_holdout='UNOPENED_HASH_ONLY' }
}
Export-StableCsv $configs (Join-Path $OutputRoot 'configuration.csv')

$inventory = @(
    [pscustomobject]@{evidence='canonical_context_evaluation';source='context-shift.csv';status='REGENERATED_CPU';use='baseline_and_context_shift'},
    [pscustomobject]@{evidence='output_optimizer_interventions';source='optimization-interventions.csv';status='HOST_ONLY';use='reject_common_output_head_cause'},
    [pscustomobject]@{evidence='branch_ablation';source='branch-ablation.csv';status='HOST_ONLY_CAUSAL';use='identify_attention_context_path'},
    [pscustomobject]@{evidence='gradient_attribution';source='qnn-l19-critical-margin-stabilization-2026-08/gradient-attribution.csv';status='PUBLIC_SOURCE';use='trajectory_timing'},
    [pscustomobject]@{evidence='legacy_probe_training_rows';source='source_audit_and_token_baseline';status='SUPERSEDED_MEASUREMENT';use='excluded_from_primary_cause'},
    [pscustomobject]@{evidence='historical_cross_seed_swap';source='attention_internal_diagnosis';status='INVALID_NOT_REUSED';use='excluded'}
)
$inventory += [pscustomobject]@{evidence='canonical_trajectory_anchors';source='qnn-l19-readout-representation-diagnosis-2026-08/trajectory-anchors.csv';status='PUBLIC_SHA256_VERIFIED';use=(Get-Sha256 $trajectoryAnchorSource)}
foreach ($id in $parameterHashes.Keys) {
    $inventory += [pscustomobject]@{evidence=("step320_parameter_content_hash_$id");source='canonical_CPU_final_checkpoint_hash';status='HASH_ONLY_NO_WEIGHTS';use=$parameterHashes[$id]}
}
Export-StableCsv $inventory (Join-Path $OutputRoot 'evidence-inventory.csv')

$hypotheses = @(
    [pscustomobject]@{id='H1';hypothesis='measurement artifact';outcome='PARTLY_CONFIRMED';strength='observation_and_source_reproduction';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H2';hypothesis='prefix ambiguity';outcome='REJECTED';strength='direct_enumeration';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H3';hypothesis='context distribution shift';outcome='SUPPORTED_CONTRIBUTOR_NOT_SUFFICIENT';strength='matched_checkpoint_comparison';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H4';hypothesis='teacher_forcing/free_running exact mismatch';outcome='SUPPORTED_AMPLIFIER';strength='matched_TF_FR';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H5';hypothesis='common late output-head drift';outcome='REJECTED_COMMON_CAUSE';strength='causal_state_intervention';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H6';hypothesis='attention Adam moment history alone';outcome='NOT_SUPPORTED';strength='causal_state_intervention';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H7';hypothesis='Attention path mediates mixed-context instability';outcome='SUPPORTED_MAJOR_FACTOR';strength='multi_seed_causal_training_intervention';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H8';hypothesis='L19-exclusive depth/residual mechanism';outcome='NOT_EXCLUSIVE_AT_TESTED_CONTROL';strength='single_L18_seed2_scope_control';origin='INITIAL_PREREGISTRATION'},
    [pscustomobject]@{id='H9';hypothesis='HTP/device-specific cause';outcome='REJECTED';strength='CPU_reproduction';origin='POST_MEASUREMENT_SCOPE_SPLIT_OF_INITIAL_H9'},
    [pscustomobject]@{id='H10';hypothesis='exhaustive hidden/gradient/moment nonfinite or float pathology';outcome='NOT_SUPPORTED_IN_SCORED_SCOPE';strength='training_loss_and_scored_outputs_finite_not_exhaustive';origin='POST_MEASUREMENT_SCOPE_SPLIT_OF_INITIAL_H9'}
)
Export-StableCsv $hypotheses (Join-Path $OutputRoot 'hypothesis-registry.csv')

$decisions = @(
    [pscustomobject]@{cycle=1;experiment='measurement_and_homogeneous_vs_mixed';reason='audit evaluator before mechanisms';result='ambiguity zero; context shift large; seed spread remains';next='optimizer state intervention'},
    [pscustomobject]@{cycle=2;experiment='step24 output/attention state intervention';reason='separate head drift from upstream state';result='no common rescue; seed2 depends on late head';next='branch ablation'},
    [pscustomobject]@{cycle=3;experiment='attention-zero vs FFN-zero training';reason='causal path isolation with negative control';result='attention-zero 144/144 all; FFN-zero 23-39/144';next='stop condition met'}
)
Export-StableCsv $decisions (Join-Path $OutputRoot 'decision-log.csv')
Export-StableCsv $measurement (Join-Path $OutputRoot 'measurement-audit.csv')
Export-StableCsv $data (Join-Path $OutputRoot 'data-structure.csv')

$trajectory = foreach ($group in ($gradient | Group-Object configuration_id)) {
    $max = $group.Group | Sort-Object { [double]$_.grad_norm_total } -Descending | Select-Object -First 1
    [pscustomobject]@{configuration_id=$group.Name;max_gradient_step=$max.step;max_gradient_norm=$max.grad_norm_total;
        attention_gradient_share=$max.grad_norm_attn_share;ffn_gradient_share=$max.grad_norm_ffn_share;
        output_gradient_share=$max.grad_norm_output_share;classification='EARLY_SHOCK_NOT_DEPTH_SPECIFIC'}
}
Export-StableCsv $trajectory (Join-Path $OutputRoot 'trajectory-divergence.csv')

$gradientSummary = foreach ($id in $configs.configuration_id) {
    foreach ($step in @(1,24,25,28,29,320)) {
        $row = $gradient | Where-Object { $_.configuration_id -eq $id -and [int]$_.step -eq $step }
        if ($row) { [pscustomobject]@{configuration_id=$id;step=$step;gradient_norm=$row.grad_norm_total;
            embedding_norm=$row.grad_norm_embedding;output_norm=$row.grad_norm_output_projection;
            attention_share=$row.grad_norm_attn_share;ffn_share=$row.grad_norm_ffn_share;
            norm_share=$row.grad_norm_norm_share;output_share=$row.grad_norm_output_share} }
    }
}
Export-StableCsv $gradientSummary (Join-Path $OutputRoot 'gradient-and-update-summary.csv')

$baselineById = @{}
foreach ($row in ($context | Where-Object { $_.dataset -eq 'AR_DEVELOPMENT_V3' })) { $baselineById[$row.configuration_id] = [int]$row.free_token_exact }
$interventions = @()
foreach ($row in ($optimization | Where-Object { $_.dataset -eq 'AR_DEVELOPMENT_V3' })) {
    $interventions += [pscustomobject]@{configuration_id=$row.configuration_id;intervention=$row.intervention;
        free_token_exact=$row.free_token_exact;free_token_total=$row.free_token_total;
        token_delta=([int]$row.free_token_exact-$baselineById[$row.configuration_id]);sequence_exact=$row.free_sequence_exact;
        teacher_token_exact=$row.teacher_token_exact;free_nll=$row.free_nll;
        train_loss_finite=$row.train_finite;scored_finite=$row.all_finite}
}
foreach ($row in ($branch | Where-Object { $_.dataset -eq 'AR_DEVELOPMENT_V3' })) {
    $interventions += [pscustomobject]@{configuration_id=$row.configuration_id;intervention=$row.intervention;
        free_token_exact=$row.free_token_exact;free_token_total=$row.free_token_total;
        token_delta=([int]$row.free_token_exact-$baselineById[$row.configuration_id]);sequence_exact=$row.free_sequence_exact;
        teacher_token_exact=$row.teacher_token_exact;free_nll=$row.free_nll;
        train_loss_finite=$row.train_finite;scored_finite=$row.all_finite}
}
Export-StableCsv $interventions (Join-Path $OutputRoot 'intervention-summary.csv')

$negative = foreach ($row in ($branch | Where-Object { $_.dataset -eq 'AR_DEVELOPMENT_V3' -and $_.intervention -eq 'FFN_BRANCH_ZERO' })) {
    [pscustomobject]@{configuration_id=$row.configuration_id;control='FFN_BRANCH_ZERO';baseline_exact=$baselineById[$row.configuration_id];
        control_exact=$row.free_token_exact;delta=([int]$row.free_token_exact-$baselineById[$row.configuration_id]);result='WORSE_NOT_MATCHING_ATTENTION'}
}
$negative += [pscustomobject]@{configuration_id='L18_SEED_2_CONTROL';control='DEPTH_SCOPE';baseline_exact=65;control_exact=144;delta=79;result='EFFECT_EXTENDS_TO_TESTED_L18_SEED2'}
Export-StableCsv $negative (Join-Path $OutputRoot 'negative-controls.csv')

$seedRows = foreach ($id in $configs.configuration_id) {
    $mixed = $context | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'AR_DEVELOPMENT_V3' }
    $hom = $context | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'HOMOGENEOUS_PHASE0' }
    $attn = $branch | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'AR_DEVELOPMENT_V3' -and $_.intervention -eq 'ATTENTION_BRANCH_ZERO' }
    $ffn = $branch | Where-Object { $_.configuration_id -eq $id -and $_.dataset -eq 'AR_DEVELOPMENT_V3' -and $_.intervention -eq 'FFN_BRANCH_ZERO' }
    [pscustomobject]@{configuration_id=$id;baseline_homogeneous_free_exact=$hom.free_token_exact;baseline_homogeneous_free_total=$hom.free_token_total;
        baseline_mixed_teacher_exact=$mixed.teacher_token_exact;baseline_mixed_free_exact=$mixed.free_token_exact;baseline_mixed_free_total=$mixed.free_token_total;
        attention_zero_mixed_free_exact=$attn.free_token_exact;attention_zero_mixed_free_total=$attn.free_token_total;
        ffn_zero_mixed_free_exact=$ffn.free_token_exact;ffn_zero_mixed_free_total=$ffn.free_token_total}
}
Export-StableCsv $seedRows (Join-Path $OutputRoot 'seed-comparison.csv')

$depth = @(
    [pscustomobject]@{comparison='baseline_mixed_free';l19_seed_range='30-63/144';l18_seed2='65/144';conclusion='control_not_worse'},
    [pscustomobject]@{comparison='attention_zero_mixed_free';l19_seed_range='144-144/144';l18_seed2='144/144';conclusion='matches_tested_L18_seed2_scope_control'},
    [pscustomobject]@{comparison='ffn_zero_mixed_free';l19_seed_range='26-39/144';l18_seed2='23/144';conclusion='negative_control_consistent'}
)
Export-StableCsv $depth (Join-Path $OutputRoot 'depth-control.csv')
Export-StableCsv $hypotheses (Join-Path $OutputRoot 'hypothesis-outcomes.csv')

$causal = @(
    [pscustomobject]@{claim='Attention path is required for mixed-context instability';intervention='ATTENTION_BRANCH_ZERO';seeds='L19_1_2_4';outcome='144/144 each';negative_control='FFN_BRANCH_ZERO 26/39/27';strength='MAJOR_FACTOR'},
    [pscustomobject]@{claim='late output head drift is common cause';intervention='OUTPUT_FREEZE_AND_MOMENT_RESET';seeds='L19_1_2_4';outcome='mixed deltas +1/-30/+2 and +2/-7/+2';negative_control='ATTENTION_MOMENT_RESET';strength='REJECTED_COMMON_CAUSE'},
    [pscustomobject]@{claim='effect is exclusive to L19';intervention='ATTENTION_BRANCH_ZERO';seeds='L18_SEED_2';outcome='65->144/144';negative_control='single_depth18_seed2';strength='NOT_EXCLUSIVE_IN_TESTED_SCOPE'}
)
Export-StableCsv $causal (Join-Path $OutputRoot 'causal-evidence.csv')

$diagnosis = @([pscustomobject]@{root_cause='COMPOUND_ATTENTION_CONTEXT_SENSITIVITY_ROOT_CAUSE';data_rule='CURRENT_TOKEN_SUCCESSOR_UNAMBIGUOUS';
    training_context='HOMOGENEOUS_PHASE0_ONLY';model_path='ATTENTION_PATH_CAUSALLY_REQUIRED_FOR_MIXED_FAILURE';
    mechanism_candidate='CONTEXT_SHORTCUT_OR_BRANCH_OPTIMIZATION_INTERFERENCE';
    optimization='SEED_CHANGES_INITIAL_PARAMETERS_ONLY_TRAJECTORY_MECHANISM_UNRESOLVED';amplifier='ARGMAX_FREE_RUNNING_EXACT';
    attention_zero_all_exact='true';ffn_negative_control='true';l18_seed2_scope_control_matches='true';evidence_strength='ROOT_CAUSE_COMPOUND'})
Export-StableCsv $diagnosis (Join-Path $OutputRoot 'diagnosis.csv')

$uncertainties = @(
    [pscustomobject]@{uncertainty='context shortcut versus branch optimization/normalization interference';impact='finer Attention mechanism; path-level compound cause remains causal';status='OPEN'},
    [pscustomobject]@{uncertainty='which attention layer/head first mediates the failure';impact='localization only';status='OPEN'},
    [pscustomobject]@{uncertainty='corrected canonical probe layer curves';impact='historical magnitude/cut-point only';status='NOT_RERUN_WITHIN_PROBE_BUDGET'},
    [pscustomobject]@{uncertainty='mixed-context training intervention';impact='independent data-side causal confirmation';status='NOT_RUN_FULL_TRAINING_BUDGET_EXHAUSTED'},
    [pscustomobject]@{uncertainty='final holdout performance';impact='not needed for root-cause diagnosis';status='UNOPENED'}
)
Export-StableCsv $uncertainties (Join-Path $OutputRoot 'remaining-uncertainties.csv')

$next = @(
    [pscustomobject]@{priority=1;candidate='mixed-context diagnostic training';purpose='independent data-side intervention';constraint='new preregistered budget; final holdout remains closed'},
    [pscustomobject]@{priority=2;candidate='row-corrected canonical probes';purpose='rebuild historical localization magnitudes';constraint='do not reuse malformed swap evidence'},
    [pscustomobject]@{priority=3;candidate='layerwise attention gate intervention';purpose='localize earliest causal Attention path';constraint='diagnostic only, L18 control required'}
)
Export-StableCsv $next (Join-Path $OutputRoot 'next-step-candidates.csv')

$sourceFiles = @(
    'app\src\main\cpp\tiny_language_model_cpu.cpp',
    'app\src\main\cpp\autoregressive_validation.h',
    'host_tests\depth_quality_lib.h',
    'host_tests\readout_probe_lib.h',
    'host_tests\critical_margin_objective_lib.h',
    'host_tests\seed_instability_diagnostics_lib.h',
    'host_tests\seed_instability_diagnostics.cpp',
    'scripts\run_l19_seed_instability_diagnostics.ps1',
    'scripts\export_public_qnn_l19_seed_instability_results.ps1',
    'docs\results\qnn-l19-critical-margin-stabilization-2026-08\gradient-attribution.csv',
    'docs\results\qnn-l19-readout-representation-diagnosis-2026-08\trajectory-anchors.csv'
)
$manifest = [ordered]@{
    schema = 'L19_SEED_INSTABILITY_ROOT_CAUSE_V1'
    schema_version = 1
    result_classification = 'COMPOUND_ATTENTION_CONTEXT_SENSITIVITY_ROOT_CAUSE'
    evidence_strength = 'ROOT_CAUSE_COMPOUND'
    major_experiment_cycles = 3
    cpu_trajectory_regenerations = 4
    short_training_runs = 4
    full_training_runs = 20
    parameter_state_interventions = 20
    internal_state_interventions = 0
    additional_probe_trainings = 0
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    adb_operations = 0
    ui_work = $false
    count_from_one = 0
    final_holdout_hash = 'fnv1a64:aa5081e6df658b4a'
    production_sources = @($sourceFiles | ForEach-Object { [ordered]@{name=$_;sha256=(Get-Sha256 (Join-Path $repoRoot $_))} })
    private_aggregate_hashes = @($requiredPrivate | ForEach-Object { [ordered]@{name=$_;sha256=(Get-Sha256 (Join-Path $PrivateRoot $_))} })
    private_identity_hashes = @([ordered]@{name='qnn-probe-optimization-audit/manifest.csv';sha256=(Get-Sha256 $parameterIdentitySource)})
    files = @($publicFiles | ForEach-Object { [ordered]@{name=$_;sha256=(Get-Sha256 (Join-Path $OutputRoot $_))} })
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Encoding utf8
Assert-PublicBundle $OutputRoot
Write-Host "L19 seed-instability public bundle exported: $OutputRoot"
