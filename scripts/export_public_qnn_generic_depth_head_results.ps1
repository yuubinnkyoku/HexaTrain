# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-generic-depth-head'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-generic-depth-head-2026-07'),
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json', 'configurations.csv',
    'resource-estimates.csv', 'training-seeds.csv',
    'cpu-htp-comparison.csv', 'generation-oracle.csv',
    'generation-free.csv', 'reproducibility.csv',
    'performance-correctness.csv', 'performance-exclusive.csv',
    'thermal.csv', 'failure-boundary.csv', 'ui-validation.csv'
)
$formalIds = [ordered]@{
    baseline = 'baseline_l2h2_formal_5seed'
    depth = 'boundary_depth_l18_h2_t8_d16_f32'
    heads = 'formal_heads_h128_l2_t8_d128_f32'
    combined = 'formal_combined_l6_h8_t16_d32_f64'
    width = 'boundary_width_l3_h4_t16_d128_f256'
}
$reproIds = 1..3 | ForEach-Object { "repro_combined_l6_h8_run$_" }
$exclusiveIds = 1..5 | ForEach-Object { "exclusive_combined_l6_h8_run$_" }

function Fail([string]$Message) {
    throw "generic depth/head public export: $Message"
}
function Put([string]$Name, [string]$Text) {
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}
function Csv([string]$Name, $Rows) {
    Put $Name ((($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n")
}
function ReportPath([string]$Id) {
    $path = Join-Path $InputRoot "$Id.txt"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Fail "missing allow-listed input report $Id"
    }
    return $path
}
function ReadMap([string]$Id) {
    $map = @{}
    foreach ($line in Get-Content -LiteralPath (ReportPath $Id)) {
        if ($line -match '^([A-Za-z][A-Za-z0-9_]*)=(.*)$') {
            if ($map.ContainsKey($Matches[1]) -and
                $map[$Matches[1]] -cne $Matches[2]) {
                Fail "conflicting duplicate key $($Matches[1]) in $Id"
            }
            $map[$Matches[1]] = $Matches[2]
        }
    }
    return $map
}
function Need($Map, [string]$Key) {
    if (-not $Map.ContainsKey($Key)) { Fail "missing report key $Key" }
    return [string]$Map[$Key]
}
function Maybe($Map, [string]$Key, [string]$Default = 'unavailable') {
    if ($Map.ContainsKey($Key)) { return [string]$Map[$Key] }
    return $Default
}
function SafePublicText([string]$Text) {
    return $Text -notmatch '(?i)([a-z]:[\\/]|/data/|/sdcard/|/storage/|device[_ -]?serial|adb[_ -]?endpoint|logcat|(?:qairt|qnn).*\.(?:so|dll)|sk-[a-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}
function CheckBundle {
    $actual = @((Get-ChildItem -LiteralPath $OutputRoot -File).Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) {
        Fail 'public bundle allow-list mismatch'
    }
    foreach ($name in $actual) {
        $text = Get-Content -LiteralPath (Join-Path $OutputRoot $name) -Raw
        if (-not (SafePublicText $text)) { Fail "unsafe public content in $name" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Raw |
        ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or
        $manifest.qairt_build_id -ne '2.48.40.260702151143' -or
        $manifest.formal_configurations -ne 5 -or
        $manifest.formal_finite_seeds -ne '25/25') {
        Fail 'manifest consistency mismatch'
    }
    $seeds = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'training-seeds.csv'))
    if ($seeds.Count -ne 25 -or
        @($seeds | Where-Object { $_.all_steps_finite -ne 'true' }).Count -ne 0) {
        Fail 'formal seed cardinality/finite mismatch'
    }
    foreach ($file in @('generation-oracle.csv', 'generation-free.csv')) {
        $rows = @(Import-Csv -LiteralPath (Join-Path $OutputRoot $file))
        $caseKeys = @($rows | ForEach-Object {
                "$($_.configuration):$($_.case_id)"
            } | Sort-Object -Unique)
        if ($rows.Count -ne 100 -or $caseKeys.Count -ne 100) {
            Fail "$file generation cardinality mismatch"
        }
    }
    $repro = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'reproducibility.csv'))
    if ($repro.Count -ne 3 -or
        @($repro.parameter_hash | Sort-Object -Unique).Count -ne 1 -or
        @($repro.logits_hash | Sort-Object -Unique).Count -ne 1 -or
        @($repro.all_step_loss_hash | Sort-Object -Unique).Count -ne 1 -or
        @($repro.all_step_accuracy_hash | Sort-Object -Unique).Count -ne 1) {
        Fail 'reproducibility hashes differ'
    }
    $exclusive = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'performance-exclusive.csv'))
    if ($exclusive.Count -ne 5) { Fail 'exclusive benchmark cardinality mismatch' }
}

if ($SelfTest) {
    $temporary = Join-Path ([IO.Path]::GetTempPath()) (
        'phonelm-generic-depth-head-export-' + [guid]::NewGuid())
    $InputRoot = Join-Path $temporary 'input'
    $OutputRoot = Join-Path $temporary 'output'
    New-Item -ItemType Directory -Path $InputRoot | Out-Null
    try {
        function Fixture([string]$Id, [string[]]$Lines) {
            [IO.File]::WriteAllLines((Join-Path $InputRoot "$Id.txt"), $Lines, $utf8)
        }
        $base = [Collections.Generic.List[string]]::new()
        foreach ($line in @(
            'status=SUCCESS', 'sequence_length=16', 'embedding_dimension=32',
            'feed_forward_dimension=64', 'transformer_layers=6',
            'attention_heads=8', 'head_dimension=4', 'steps=320',
            'seed_count=5', 'exact_rollout_count=20',
            'oracle_exact_rollout_count=20',
            'resource_estimator_parameter_elements=100',
            'resource_estimator_parameter_bytes=400',
            'resource_estimator_gradient_bytes=400',
            'resource_estimator_adam_m_v_bytes=800',
            'resource_estimator_forward_activation_bytes=1000',
            'resource_estimator_backward_activation_bytes=1200',
            'resource_estimator_attention_bytes=200',
            'resource_estimator_adam_application_visible_bytes=300',
            'resource_estimator_persistent_application_tensor_bytes=2200',
            'resource_estimator_peak_application_tensor_bytes=5500',
            'resource_estimator_total_node_count_with_adam=10',
            'resource_estimator_total_tensor_count_with_adam=20',
            'performance_initialization_ms=1',
            'performance_graph_creation_ms=2', 'performance_finalize_ms=3',
            'performance_steady_training_step_median_ms=4',
            'performance_updates_per_second=250',
            'performance_tokens_per_second=4000',
            'generation_token_latency_median_ms=5',
            'process_peak_rss_kib=1000', 'battery_temperature_before_c=30',
            'battery_temperature_after_c=31', 'thermal_status_before=0',
            'thermal_status_after=0'
        )) { $base.Add($line) }
        foreach ($seed in 1..5) {
            foreach ($line in @(
                "seed_${seed}_all_steps_finite=true",
                "seed_${seed}_nonfinite_count=0",
                "seed_${seed}_final_loss=0.1",
                "seed_${seed}_final_accuracy=1",
                "seed_${seed}_cpu_htp_parameter_max_abs_difference=0.01",
                "seed_${seed}_final_parameter_canonical_hash=hash-$seed",
                "seed_${seed}_step_320_logits_canonical_hash=logits-$seed"
            )) { $base.Add($line) }
        }
        foreach ($mode in @('oracle', 'free')) {
            foreach ($seed in 1..5) {
                foreach ($pattern in 0..3) {
                    $prefix = "formal_${mode}_case_s${seed}_p${pattern}"
                    foreach ($line in @(
                        "${prefix}_id=s${seed}_p${pattern}",
                        "${prefix}_prefix=0,1", "${prefix}_expected_sequence=2,3",
                        "${prefix}_generated_sequence=2,3", "${prefix}_exact=true",
                        "${prefix}_first_mismatch_step=-1",
                        "${prefix}_first_mismatch_expected_token=NONE",
                        "${prefix}_first_mismatch_predicted_token=NONE",
                        "${prefix}_first_mismatch_top3=NONE"
                    )) { $base.Add($line) }
                }
            }
        }
        foreach ($id in $formalIds.Values) { Fixture $id $base }
        foreach ($id in $reproIds) {
            Fixture $id @(
                'status=SUCCESS',
                'seed_1_final_parameter_canonical_hash=repro-parameter',
                'seed_1_step_320_logits_canonical_hash=repro-logits',
                'seed_1_all_step_loss_canonical_hash=repro-losses',
                'seed_1_all_step_accuracy_canonical_hash=repro-accuracies',
                'exact_rollout_count=4', 'oracle_exact_rollout_count=4'
            )
        }
        foreach ($id in $exclusiveIds) { Fixture $id $base }
        Fixture 'boundary_depth_l19_h2_t8_d16_f32' @(
            'status=FAILED',
            'error=seed=1, final_evaluation, first_bad_tensor=layer_018_output'
        )
        Fixture 'boundary_interaction_l3_h4_t16_d256_f371_seed1' $base
        Fixture 'boundary_interaction_l3_h4_t16_d256_f372_seed1' @(
            'status=FAILED',
            'error=seed=1, step=32, first_bad_tensor=logits'
        )
        Fixture 'diagnostic_depth_l64_h2_seed1' @(
            'status=FAILED',
            'error=seed=1, step=301, first_bad_tensor=dembedding'
        )
        Fixture 'ui_validation_summary' @(
            'status=PASS', 'config_visible=true', 'progress_visible=true',
            'foreground_update=true', 'background_update=true',
            'ongoing_during_run=true', 'ongoing_after_completion=false',
            'notification_tap_returned_activity=true', 'auto_cancel=true'
        )
        New-Item -ItemType Directory -Path $OutputRoot | Out-Null
        & $PSCommandPath -InputRoot $InputRoot -OutputRoot $OutputRoot
        [IO.File]::AppendAllText(
            (Join-Path $OutputRoot 'README.md'),
            "`nunsafe=" + ('D:' + '\private\report.txt') + "`n", $utf8)
        $rejected = $false
        try { CheckBundle } catch { $rejected = $true }
        if (-not $rejected) { Fail 'unsafe-content negative self-test was accepted' }
        Write-Output 'SELF_TEST=PASS'
    } finally {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
    return
}

$maps = [ordered]@{}
foreach ($entry in $formalIds.GetEnumerator()) {
    $maps[$entry.Key] = ReadMap $entry.Value
}
$reproMaps = @($reproIds | ForEach-Object { ReadMap $_ })
$exclusiveMaps = @($exclusiveIds | ForEach-Object { ReadMap $_ })
$depthFailure = ReadMap 'boundary_depth_l19_h2_t8_d16_f32'
$depthDiagnostic = ReadMap 'diagnostic_depth_l64_h2_seed1'
$widthSuccess = ReadMap 'boundary_interaction_l3_h4_t16_d256_f371_seed1'
$widthFailure = ReadMap 'boundary_interaction_l3_h4_t16_d256_f372_seed1'
$ui = ReadMap 'ui_validation_summary'
$oneStepEnvelope =
    'L256/H2, L2/H256, T512/D32/FFN64/L3/H4, D256/FFN512/L3/H4'

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
Get-ChildItem -LiteralPath $OutputRoot -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

$configurations = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    [pscustomobject]@{
        category = $entry.Key
        sequence = Need $map 'sequence_length'
        embedding = Need $map 'embedding_dimension'
        feed_forward = Need $map 'feed_forward_dimension'
        layers = Need $map 'transformer_layers'
        heads = Need $map 'attention_heads'
        steps = 320
        seeds = 5
        clipping = 'disabled'
        finite_seeds = '5/5'
        oracle_exact = "$(Need $map 'oracle_exact_rollout_count')/20"
        free_exact = "$(Need $map 'exact_rollout_count')/20"
        result = if ((Need $map 'status') -eq 'SUCCESS') {
            'formal_success'
        } else {
            'finite_generation_quality_boundary'
        }
    }
}
Csv 'configurations.csv' $configurations

$resourceRows = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    [pscustomobject]@{
        configuration = $entry.Key
        parameter_elements = Need $map 'resource_estimator_parameter_elements'
        parameter_bytes = Need $map 'resource_estimator_parameter_bytes'
        gradient_bytes = Need $map 'resource_estimator_gradient_bytes'
        adam_m_v_bytes = Need $map 'resource_estimator_adam_m_v_bytes'
        forward_activation_bytes = Need $map 'resource_estimator_forward_activation_bytes'
        backward_activation_bytes = Need $map 'resource_estimator_backward_activation_bytes'
        attention_bytes = Need $map 'resource_estimator_attention_bytes'
        adam_application_visible_bytes = Need $map 'resource_estimator_adam_application_visible_bytes'
        persistent_application_tensor_bytes = Need $map 'resource_estimator_persistent_application_tensor_bytes'
        estimated_peak_application_tensor_bytes = Need $map 'resource_estimator_peak_application_tensor_bytes'
        node_count = Need $map 'resource_estimator_total_node_count_with_adam'
        tensor_count = Need $map 'resource_estimator_total_tensor_count_with_adam'
        scope = 'application-visible estimate; excludes QNN/DSP internal memory'
    }
}
Csv 'resource-estimates.csv' $resourceRows

$seedRows = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    foreach ($seed in 1..5) {
        [pscustomobject]@{
            configuration = $entry.Key
            seed = $seed
            all_steps_finite = Need $map "seed_${seed}_all_steps_finite"
            nonfinite_count = Need $map "seed_${seed}_nonfinite_count"
            final_loss = Need $map "seed_${seed}_final_loss"
            final_accuracy = Need $map "seed_${seed}_final_accuracy"
            parameter_hash = Need $map "seed_${seed}_final_parameter_canonical_hash"
            logits_hash = Need $map "seed_${seed}_step_320_logits_canonical_hash"
        }
    }
}
Csv 'training-seeds.csv' $seedRows

$comparisonRows = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    foreach ($seed in 1..5) {
        [pscustomobject]@{
            configuration = $entry.Key
            seed = $seed
            parameter_max_abs_difference =
                Need $map "seed_${seed}_cpu_htp_parameter_max_abs_difference"
            qnn_nonzero_return_count = 0
            finite = Need $map "seed_${seed}_all_steps_finite"
            comparison_scope = 'CPU reference trajectory versus QNN HTP trajectory'
        }
    }
}
Csv 'cpu-htp-comparison.csv' $comparisonRows

function GenerationRows([string]$Mode) {
    foreach ($entry in $formalIds.GetEnumerator()) {
        $map = $maps[$entry.Key]
        foreach ($seed in 1..5) {
            foreach ($pattern in 0..3) {
                $prefix = "formal_${Mode}_case_s${seed}_p${pattern}"
                [pscustomobject]@{
                    configuration = $entry.Key
                    case_id = Need $map "${prefix}_id"
                    seed = $seed
                    pattern = $pattern
                    prefix = Need $map "${prefix}_prefix"
                    expected_sequence = Need $map "${prefix}_expected_sequence"
                    generated_sequence = Need $map "${prefix}_generated_sequence"
                    exact = Need $map "${prefix}_exact"
                    first_mismatch_step =
                        Need $map "${prefix}_first_mismatch_step"
                    expected_token =
                        Need $map "${prefix}_first_mismatch_expected_token"
                    predicted_token =
                        Need $map "${prefix}_first_mismatch_predicted_token"
                    top3 = Need $map "${prefix}_first_mismatch_top3"
                }
            }
        }
    }
}
Csv 'generation-oracle.csv' @(GenerationRows 'oracle')
Csv 'generation-free.csv' @(GenerationRows 'free')

$reproRows = for ($index = 0; $index -lt $reproMaps.Count; ++$index) {
    $map = $reproMaps[$index]
    [pscustomobject]@{
        run = $index + 1
        configuration = 'T16 D32 FFN64 L6 H8'
        parameter_hash = Need $map 'seed_1_final_parameter_canonical_hash'
        logits_hash = Need $map 'seed_1_step_320_logits_canonical_hash'
        all_step_loss_hash = Need $map 'seed_1_all_step_loss_canonical_hash'
        all_step_accuracy_hash =
            Need $map 'seed_1_all_step_accuracy_canonical_hash'
        oracle_exact = Need $map 'oracle_exact_rollout_count'
        free_exact = Need $map 'exact_rollout_count'
        classification = 'BITWISE_REPRODUCIBLE'
    }
}
Csv 'reproducibility.csv' $reproRows

$correctnessRows = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    [pscustomobject]@{
        configuration = $entry.Key
        initialization_ms = Need $map 'performance_initialization_ms'
        graph_creation_ms = Need $map 'performance_graph_creation_ms'
        finalize_ms = Need $map 'performance_finalize_ms'
        steady_training_step_median_ms =
            Need $map 'performance_steady_training_step_median_ms'
        updates_per_second = Need $map 'performance_updates_per_second'
        tokens_per_second = Need $map 'performance_tokens_per_second'
        generation_token_latency_median_ms =
            Need $map 'generation_token_latency_median_ms'
        process_peak_rss_kib = Need $map 'process_peak_rss_kib'
        mode = 'correctness'
    }
}
Csv 'performance-correctness.csv' $correctnessRows

$exclusiveRows = for ($index = 0; $index -lt $exclusiveMaps.Count; ++$index) {
    $map = $exclusiveMaps[$index]
    [pscustomobject]@{
        run = $index + 1
        configuration = 'T16 D32 FFN64 L6 H8'
        initialization_ms = Need $map 'performance_initialization_ms'
        graph_creation_ms = Need $map 'performance_graph_creation_ms'
        finalize_ms = Need $map 'performance_finalize_ms'
        steady_training_step_median_ms =
            Need $map 'performance_steady_training_step_median_ms'
        updates_per_second = Need $map 'performance_updates_per_second'
        tokens_per_second = Need $map 'performance_tokens_per_second'
        generation_token_latency_median_ms =
            Need $map 'generation_token_latency_median_ms'
        process_peak_rss_kib = Need $map 'process_peak_rss_kib'
        mode = 'exclusive_benchmark'
    }
}
Csv 'performance-exclusive.csv' $exclusiveRows

$thermalRows = foreach ($entry in $formalIds.GetEnumerator()) {
    $map = $maps[$entry.Key]
    [pscustomobject]@{
        run = $entry.Key
        battery_temperature_before_c = Need $map 'battery_temperature_before_c'
        battery_temperature_after_c = Need $map 'battery_temperature_after_c'
        android_thermal_status_before = Need $map 'thermal_status_before'
        android_thermal_status_after = Need $map 'thermal_status_after'
    }
}
Csv 'thermal.csv' $thermalRows

function FirstBad([string]$Error) {
    $match = [regex]::Match($Error, 'first_bad_tensor=([^,]+)')
    if ($match.Success) { return $match.Groups[1].Value }
    return 'unavailable'
}
$failureRows = @(
    [pscustomobject]@{
        axis = 'depth-formal'
        last_success = 'T8 D16 FFN32 L18 H2; 5/5 training seeds finite'
        first_failure = 'T8 D16 FFN32 L19 H2'
        stage = 'final_evaluation'
        first_bad_tensor = FirstBad (Need $depthFailure 'error')
        classification = 'HTP_NUMERIC_OVERFLOW'
        detail = 'QNN execute returned success; application-visible tensor was nonfinite'
    },
    [pscustomobject]@{
        axis = 'depth-diagnostic'
        last_success = 'T8 D16 FFN32 L18 H2'
        first_failure = 'T8 D16 FFN32 L64 H2 seed1 step301'
        stage = 'training_backward'
        first_bad_tensor = FirstBad (Need $depthDiagnostic 'error')
        classification = 'HTP_NUMERIC_OVERFLOW'
        detail = 'reproduced at seed1 step301'
    },
    [pscustomobject]@{
        axis = 'width-ffn-interaction'
        last_success = 'T16 D256 FFN371 L3 H4 seed1 320 steps'
        first_failure = 'T16 D256 FFN372 L3 H4 seed1 step32'
        stage = 'training_forward'
        first_bad_tensor = FirstBad (Need $widthFailure 'error')
        classification = 'HTP_NUMERIC_OVERFLOW'
        detail = 'D256/FFN256 and D128/FFN512 separately succeeded'
    },
    [pscustomobject]@{
        axis = 'one-step-envelope'
        last_success = $oneStepEnvelope
        first_failure = 'none evaluated'
        stage = 'not applicable'
        first_bad_tensor = 'none'
        classification = 'NONE'
        detail = 'saturation stop; not a hardware-limit claim'
    }
)
Csv 'failure-boundary.csv' $failureRows

Csv 'ui-validation.csv' @([pscustomobject]@{
    status = Need $ui 'status'
    config_visible = Need $ui 'config_visible'
    progress_visible = Need $ui 'progress_visible'
    foreground_update = Need $ui 'foreground_update'
    background_update = Need $ui 'background_update'
    ongoing_during_run = Need $ui 'ongoing_during_run'
    ongoing_after_completion = Need $ui 'ongoing_after_completion'
    notification_tap_returned_activity =
        Need $ui 'notification_tap_returned_activity'
    auto_cancel = Need $ui 'auto_cancel'
})

$manifest = [ordered]@{
    schema_version = 1
    qairt_build_id = '2.48.40.260702151143'
    result_classification = 'GENERIC_GRAPH_IMPLEMENTED_NUMERIC_BOUNDARY_FOUND'
    maximum_formal_combined_configuration = [ordered]@{
        batch = 1; sequence = 16; vocabulary = 32; embedding = 32
        feed_forward = 64; layers = 6; heads = 8; head_dimension = 4
        optimizer = 'Adam'; learning_rate = 0.003; steps_per_seed = 320
        clipping = 'disabled'
    }
    one_step_envelope = $oneStepEnvelope
    formal_configurations = 5
    formal_finite_seeds = '25/25'
    combined_oracle_exact = '20/20'
    combined_free_exact = '20/20'
    seed_reproducibility = 'BITWISE_REPRODUCIBLE'
    publication = 'allow-list aggregates; raw tensors, identifiers, paths, and binaries excluded'
} | ConvertTo-Json -Depth 6
Put 'manifest.json' ($manifest + "`n")

Put 'README.md' @'
# QNN HTP generic Transformer depth/head results

PhoneLM executed the numerical operations of explicit Transformer forward,
backward, and Adam update graphs on QNN HTP. Parameter/state registries, CPU
reference code, shape validation, graph-map generation, diagnostics, and
application-visible resource accounting accept generic positive layer and head
counts where the embedding dimension is divisible by the head count.

The maximum formal combined configuration was B1/T16/V32/D32/FFN64/L6/H8.
Five of five 320-step seeds remained finite and Oracle and free-running
generation were each exact for 20/20 cases. H128 at L2 and D128/FFN256 at
L3/H4 also completed five formal seeds. The published baseline
T32/D32/FFN32/L2/H2 retained its established legacy canonical parameter hash,
representative logits hash, losses, accuracy, and 20/20 generation results.

The first adjacent formal depth failure was L19 after L18 completed five finite
training seeds. At D256/L3/H4, FFN371 completed seed 1 for 320 steps and FFN372
first failed at step 32. These are reproducible numerical trajectory boundaries:
QNN graph execution returned success while an application-visible tensor became
nonfinite. They are not claims of an HTP hardware maximum. One-step graph
creation, finalization, forward, backward, and Adam covered the larger envelope
listed in `manifest.json`; exploration stopped at clear runtime saturation.

`resource-estimates.csv` is conservative application-visible accounting and
must not be interpreted as DSP/runtime-internal memory use. Correctness and
exclusive benchmark measurements are separated. Battery temperature and
Android thermal status are reported without labeling battery temperature as CPU
temperature.

This directory is produced by an allow-list exporter. Raw tensors/logits,
checkpoints, device and connection identifiers, private paths, APKs, QAIRT
libraries/headers, Stub/Skel assets, and device logs are excluded.
'@

CheckBundle
Write-Output "exported=$OutputRoot"
