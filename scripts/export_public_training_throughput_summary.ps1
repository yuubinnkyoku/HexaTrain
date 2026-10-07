# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Allow-list projection of a validated matched comparison. Input stays private;
# output is a local aggregate for documentation, not a publication operation.
param(
    [string]$ComparisonPath,
    [string]$OutputPath = 'build/reports/training-throughput-public-summary.json',
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$metricFields = @(
    'training_step_ms','updates_per_second','original_utf8_bytes_per_second',
    'optimizer_update_wall_ms_per_update','optimizer_result_move_ms_per_update',
    'fwd_backward_ms_per_update','gradient_accumulation_ms_per_update',
    'checkpoint_io_ms_per_update','batch_data_prepare_ms',
    'app_write_schema_validation_ms','app_read_buffer_allocate_ms',
    'app_read_poison_fill_ms','app_read_materialize_ms','app_read_finite_validation_ms',
    'gradient_registry_validation_ms','unclassified_host_ms',
    'muon_pack_ms','muon_rpc_ms','muon_kernel_ms','muon_unpack_ms',
    'hvx_input_validation_ms','hvx_output_validation_ms','hvx_pack_registry_ms',
    'hvx_unpack_candidate_generation_ms','hvx_unpack_decode_ms','hvx_aux_adam_wall_ms',
    'hvx_aux_registry_construction_ms','hvx_aux_registry_validation_ms',
    'hvx_aux_arithmetic_ms','battery_temperature_c_before','battery_temperature_c_after',
    'android_thermal_status_before','android_thermal_status_after'
)
$identityFields = @('seed','steps','batch_size','model_dimension',
    'feed_forward_dimension','parameter_count','muon_matrix_count',
    'muon_parameter_count','aux_adam_parameter_count','run_target_utf8_bytes_seen',
    'muon_lr','aux_adam_lr')
function Get-FiniteNumber($Value) {
    $number = 0.0
    if (-not [double]::TryParse([string]$Value, [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$number) -or
        [double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw 'PUBLIC_METRIC_NOT_FINITE'
    }
    return $number
}
function Get-Median([double[]]$Values) {
    $ordered = @($Values | Sort-Object)
    if (-not $ordered.Count) { throw 'PUBLIC_COMPARISON_EMPTY' }
    $middle = [int][Math]::Floor($ordered.Count / 2)
    if ($ordered.Count % 2) { return $ordered[$middle] }
    return ($ordered[$middle - 1] + $ordered[$middle]) / 2
}
function ConvertTo-PublicSummary($Source) {
    $public = [ordered]@{ schema_version = 1; models = [ordered]@{} }
    foreach ($arm in @('control','g1')) {
        $runs = @($Source.models.$arm.runs)
        if (-not $runs.Count) { throw 'PUBLIC_COMPARISON_EMPTY' }
        $recipe = [ordered]@{}
        foreach ($field in $identityFields) {
            $recipe[$field] = Get-FiniteNumber $runs[0].identity.$field
        }
        if ($recipe.seed -lt 0 -or $recipe.steps -le 0 -or $recipe.batch_size -le 0) {
            throw 'PUBLIC_RECIPE_INVALID'
        }
        $recipe.attention_gate = if ($arm -eq 'control') { 'none' } else { 'headwise_g1_sigmoid' }
        $recipe.checkpoint_format = if ($arm -eq 'control') { 'NPRTCKPTV4' } else { 'NPRTCKPTV5' }
        $recipe.learning_rate_schedule = 'linear_decay'
        $recipe.qairt_build_id = '2.48.40.260702151143'
        $rows = @()
        foreach ($run in $runs) {
            foreach ($gate in @('canonical_identity_matches','health_matches',
                                'checkpoint_byte_parity','loss_curve_byte_parity')) {
                if ($run.$gate -isnot [bool] -or $run.$gate -ne $true) {
                    throw 'PUBLIC_COMPARISON_GATE_REJECTED'
                }
            }
            foreach ($field in $identityFields) {
                if ((Get-FiniteNumber $run.identity.$field) -ne $recipe[$field]) {
                    throw 'PUBLIC_RECIPE_MISMATCH'
                }
            }
            if ($run.identity.attention_gate -cne $recipe.attention_gate -or
                $run.identity.checkpoint_format -cne $recipe.checkpoint_format -or
                $run.identity.learning_rate_schedule -cne $recipe.learning_rate_schedule -or
                $run.identity.compile_time_qairt_build_id -cne $recipe.qairt_build_id) {
                throw 'PUBLIC_RECIPE_MISMATCH'
            }
            $row = [ordered]@{ pair = $rows.Count + 1 }
            foreach ($side in @('before','after')) {
                $row[$side] = [ordered]@{}
                foreach ($field in $metricFields) {
                    $row[$side][$field] = Get-FiniteNumber $run.$side.$field
                }
                $ms = $row[$side].training_step_ms
                if ($ms -le 0 -or $row[$side].updates_per_second -le 0 -or
                    [Math]::Abs($ms * $row[$side].updates_per_second - 1000) -gt 0.01) {
                    throw 'PUBLIC_RATE_MISMATCH'
                }
                $expectedBytes = $recipe.run_target_utf8_bytes_seen / $recipe.steps *
                    $row[$side].updates_per_second
                if ([Math]::Abs($expectedBytes - $row[$side].original_utf8_bytes_per_second) -gt
                    [Math]::Max(0.001, $expectedBytes * 1e-6)) { throw 'PUBLIC_RATE_MISMATCH' }
            }
            $row.ratio = $row.after.training_step_ms / $row.before.training_step_ms
            if ([Math]::Abs($row.ratio - (Get-FiniteNumber $run.ratio)) -gt 1e-9) {
                throw 'PUBLIC_PAIRED_RATIO_MISMATCH'
            }
            $row.checkpoint_pairs = Get-FiniteNumber $run.checkpoint_pairs
            if ($row.checkpoint_pairs -le 0) { throw 'PUBLIC_CHECKPOINT_COUNT_INVALID' }
            $rows += $row
        }
        $summary = [ordered]@{}
        foreach ($side in @('before','after')) {
            $summary[$side] = [ordered]@{}
            foreach ($field in $metricFields) {
                $summary[$side][$field] = Get-Median @($rows | ForEach-Object { $_[$side][$field] })
            }
        }
        $ratios = @($rows | ForEach-Object { $_.ratio })
        $public.models[$arm] = [ordered]@{
            recipe = $recipe; pair_count = $rows.Count; summary = $summary
            paired_reduction_percent = 100 * (1 - (Get-Median $ratios))
            paired_speedup = Get-Median @($ratios | ForEach-Object { 1 / $_ })
            ratio_min = ($ratios | Measure-Object -Minimum).Minimum
            ratio_max = ($ratios | Measure-Object -Maximum).Maximum
            runs = $rows
        }
    }
    if ($public.models.control.pair_count -ne $public.models.g1.pair_count) {
        throw 'PUBLIC_MODEL_PAIR_COUNT_MISMATCH'
    }
    return $public
}
if ($SelfTest) {
    $fixture = @{ models = @{}; device_serial = 'PRIVATE_SENTINEL'; run_id = 'PRIVATE_SENTINEL' }
    foreach ($arm in @('control','g1')) {
        $identity = @{}
        foreach ($field in $identityFields) { $identity[$field] = 1 }
        $identity.steps = 100; $identity.run_target_utf8_bytes_seen = 10000
        $identity.attention_gate = if ($arm -eq 'control') { 'none' } else { 'headwise_g1_sigmoid' }
        $identity.checkpoint_format = if ($arm -eq 'control') { 'NPRTCKPTV4' } else { 'NPRTCKPTV5' }
        $identity.learning_rate_schedule = 'linear_decay'
        $identity.compile_time_qairt_build_id = '2.48.40.260702151143'
        $before = @{}; $after = @{}
        foreach ($field in $metricFields) { $before[$field] = 1; $after[$field] = 1 }
        $before.training_step_ms = 200; $before.updates_per_second = 5
        $before.original_utf8_bytes_per_second = 500
        $after.training_step_ms = 100; $after.updates_per_second = 10
        $after.original_utf8_bytes_per_second = 1000
        $run = @{ identity = $identity; before = $before; after = $after; ratio = 0.5
            canonical_identity_matches = $true; health_matches = $true
            checkpoint_byte_parity = $true; loss_curve_byte_parity = $true; checkpoint_pairs = 1 }
        $fixture.models[$arm] = @{ runs = @($run) }
    }
    $json = ConvertTo-PublicSummary $fixture | ConvertTo-Json -Depth 12
    if ($json.Contains('PRIVATE_SENTINEL')) { throw 'SELFTEST_PRIVATE_LEAK' }
    $result = $json | ConvertFrom-Json
    if ($result.models.control.paired_reduction_percent -ne 50 -or
        $result.models.control.paired_speedup -ne 2) { throw 'SELFTEST_PAIRED_MATH' }
    foreach ($mutation in @('rate','nonfinite','parity','recipe','ratio')) {
        $bad = ($fixture | ConvertTo-Json -Depth 12 | ConvertFrom-Json)
        $run = $bad.models.control.runs[0]
        switch ($mutation) {
            'rate' { $run.after.updates_per_second = 9 }
            'nonfinite' { $run.after.training_step_ms = 'NaN' }
            'parity' { $run.checkpoint_byte_parity = $false }
            'recipe' { $run.identity.compile_time_qairt_build_id = '2.47' }
            'ratio' { $run.ratio = 0.9 }
        }
        $rejected = $false
        try { ConvertTo-PublicSummary $bad | Out-Null } catch { $rejected = $true }
        if (-not $rejected) { throw "SELFTEST_NOT_REJECTED: $mutation" }
    }
    Write-Host 'training_throughput_public_summary_selftest=PASS'
    exit 0
}
function Resolve-BuildPath([string]$Path) {
    if (-not $Path) { throw 'COMPARISON_PATH_REQUIRED' }
    $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $allowed = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
    $resolved = [IO.Path]::GetFullPath((Join-Path $root $Path))
    if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'SUMMARY_PATH_OUTSIDE_BUILD'
    }
    return $resolved
}
$sourcePath = Resolve-BuildPath $ComparisonPath
$destination = Resolve-BuildPath $OutputPath
if ([string]::Equals($sourcePath, $destination, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'SUMMARY_SOURCE_OUTPUT_COLLISION'
}
$source = Get-Content -LiteralPath $sourcePath -Raw | ConvertFrom-Json
$summary = ConvertTo-PublicSummary $source
[IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
[IO.File]::WriteAllText($destination, ($summary | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
Write-Host 'training_throughput_public_summary=PASS'
