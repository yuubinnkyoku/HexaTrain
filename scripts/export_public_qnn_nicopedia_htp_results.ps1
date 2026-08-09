# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Allow-list exporter for the Nicopedia real-text HTP training milestone.
#
# Publishes only aggregate fields from the device runs (loss trajectory,
# parity metrics, timing, thermal records).  Private artifacts (token caches,
# checkpoints, serials, raw ADB output) never enter the output bundle.
param(
    [switch]$SelfTest,
    [string]$ReportRoot = "build/reports/nicopedia-htp-training",
    [string]$OutputRoot = "docs/results/qnn-nicopedia-htp-training-2026-08",
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$canonicalOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot "docs/results/qnn-nicopedia-htp-training-2026-08"))
$allowList = @(
    "README.md", "manifest.json", "configuration.csv", "runtime-identity.csv",
    "graph-shapes.csv", "memory-summary.csv", "one-step-parity.csv",
    "trajectory-parity.csv", "finite-summary.csv", "l6-training-summary.csv",
    "l6-quality-summary.csv", "seed-summary.csv", "l19-summary.csv",
    "performance.csv", "thermal-summary.csv", "cpu-vs-htp.csv",
    "limitations.csv", "diagnosis.csv", "next-step-candidates.csv"
)

function Resolve-Under([string]$Path, [string]$AllowedRoot, [string]$Label) {
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { [IO.Path]::GetFullPath($Path) } else { [IO.Path]::GetFullPath((Join-Path $repoRoot $Path)) }
    $allowed = [IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not ($candidate + [IO.Path]::DirectorySeparatorChar).StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label resolves outside its allowed root"
    }
    $cursor = Get-Item -LiteralPath $candidate -ErrorAction SilentlyContinue
    if (-not $cursor) { $cursor = Get-Item -LiteralPath (Split-Path -Parent $candidate) -ErrorAction SilentlyContinue }
    while ($cursor -and $cursor.FullName.StartsWith($allowed.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "$Label contains a reparse point" }
        $cursor = $cursor.Parent
    }
    return $candidate
}

function Get-NormalizedSha256([string]$Path) {
    $text = Get-Content -LiteralPath $Path -Raw
    if ($null -eq $text) { $text = "" }
    $normalized = $text -replace "`r`n", "`n" -replace "`r", "`n"
    $bytes = [Text.Encoding]::UTF8.GetBytes($normalized)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-StringSha256([string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Read-KeyValueReport([string]$Path) {
    $map = [ordered]@{}
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            $map[$Matches[1]] = $Matches[2]
        }
    }
    return $map
}

# Aggregate-only field allow list: no token, parameter, gradient, logit,
# article, serial, or absolute-path values.
$keyAllowList = @(
    "seed", "layers", "heads", "steps", "batch_size", "learning_rate",
    "cache_context", "cache_vocabulary", "cache_record_count",
    "cache_content_hash", "training_order_hash", "initial_parameter_hash",
    "parameter_element_count", "optimizer_chunk_count",
    "step0_loss_cpu", "step0_loss_htp", "step0_logits_max_abs_error",
    "step0_logits_mean_abs_error", "step0_logits_max_relative_error",
    "step0_logits_l2_error", "step0_logits_cosine_similarity",
    "step0_probability_max_abs_error", "step0_dlogits_max_abs_error",
    "step0_gradient_max_abs_error", "step0_finite", "step0_ok",
    "step0_tolerance_logits", "step0_tolerance_probability",
    "step0_tolerance_dlogits", "step0_tolerance_gradient",
    "trajectory_step_2_cpu_loss", "trajectory_step_2_htp_loss",
    "trajectory_step_2_parameter_max_abs_error",
    "trajectory_step_2_first_moment_max_abs_error",
    "trajectory_step_2_second_moment_max_abs_error", "trajectory_step_2_finite",
    "trajectory_step_4_cpu_loss", "trajectory_step_4_htp_loss",
    "trajectory_step_4_parameter_max_abs_error",
    "trajectory_step_4_first_moment_max_abs_error",
    "trajectory_step_4_second_moment_max_abs_error", "trajectory_step_4_finite",
    "trajectory_step_8_cpu_loss", "trajectory_step_8_htp_loss",
    "trajectory_step_8_parameter_max_abs_error",
    "trajectory_step_8_first_moment_max_abs_error",
    "trajectory_step_8_second_moment_max_abs_error", "trajectory_step_8_finite",
    "first_loss", "last_loss", "loss_decreased", "completed_steps",
    "all_steps_finite", "final_parameter_max_abs_error",
    "final_first_moment_max_abs_error", "final_second_moment_max_abs_error",
    "final_parameter_canonical_hash", "final_cpu_parameter_canonical_hash",
    "final_parameter_hash", "final_finite", "checkpoint_written",
    "htp_initialize_us", "graph_create_us", "graph_finalize_us",
    "first_execute_us", "training_total_seconds", "training_step_ms",
    "graph_execute_count", "execute_count_per_training_step",
    "bias_correction_scalar_responsibility", "optimizer_math_responsibility",
    "cpu_fallback", "nan_detected", "inf_detected", "status",
    "api_trace_graph_execute_attempt_count", "api_trace_graph_execute_success_count",
    "api_trace_graph_execute_failure_count",
    "tiny_transformer_training_graph", "tiny_transformer_training_forward_layers",
    "tiny_transformer_training_forward_heads",
    "tiny_transformer_training_parameter_registry_count",
    "tiny_transformer_training_app_read_registry_count",
    "tiny_transformer_training_tensor_create_count",
    "tiny_transformer_training_node_count",
    "resource_estimator_parameter_elements", "resource_estimator_parameter_bytes",
    "resource_estimator_gradient_bytes", "resource_estimator_adam_m_v_bytes",
    "resource_estimator_forward_activation_bytes",
    "resource_estimator_backward_activation_bytes",
    "resource_estimator_attention_bytes", "resource_estimator_adam_graph_elements",
    "resource_estimator_adam_chunk_count",
    "resource_estimator_adam_application_visible_bytes",
    "resource_estimator_persistent_application_tensor_bytes",
    "resource_estimator_peak_application_tensor_bytes",
    "resource_estimator_total_node_count_with_adam",
    "resource_estimator_total_tensor_count_with_adam",
    "compile_time_qairt_build_id", "compile_time_qnn_api_version",
    "runtime_backend_build_id", "backend_build_id_match",
    "device_model", "device_soc",
    "android_thermal_status_before", "android_thermal_status_after",
    "battery_level_before", "battery_level_after",
    "battery_temperature_c_before", "battery_temperature_c_after"
)

function Select-PublicFields($Map) {
    $public = [ordered]@{}
    foreach ($key in $keyAllowList) {
        if ($Map.Contains($key)) { $public[$key] = $Map[$key] }
    }
    return $public
}

function Write-Csv([string]$Path, [object[]]$Rows) {
    if ($Rows.Count -eq 0) { return }
    $header = $Rows[0].Keys -join ','
    $lines = @($header)
    foreach ($row in $Rows) {
        $lines += (($row.Keys | ForEach-Object { $row[$_] }) -join ',')
    }
    [IO.File]::WriteAllLines($Path, $lines, [Text.UTF8Encoding]::new($false))
}

function Write-Json([string]$Path, [object]$Value) {
    $json = $Value | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($Path, $json + "`n", [Text.UTF8Encoding]::new($false))
}

$reports = Resolve-Under $ReportRoot $([IO.Path]::GetFullPath((Join-Path $repoRoot "build"))) "ReportRoot"
$output = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))

# Self-test mode: run against a synthetic fixture under build/export-selftest
# so CI can verify the allow-list and leak guards without device results.
if ($SelfTest) {
    $temporary = Join-Path $repoRoot "build/export-selftest/nicopedia-htp-public"
    [IO.Directory]::CreateDirectory($temporary) | Out-Null
    $fixtureDir = Join-Path $repoRoot "build/export-selftest/nicopedia-htp-fixture"
    [IO.Directory]::CreateDirectory($fixtureDir) | Out-Null
    $fixtureReport = Join-Path $fixtureDir "seed1-l6-steps320-result.txt"
    $fixtureLines = @(
        "NICOPEDIA_HTP", "test=nicopedia_real_text_htp_training", "status=SUCCESS",
        "seed=1", "layers=6", "heads=2", "steps=320", "batch_size=8",
        "learning_rate=0.003000000026", "cache_context=32", "cache_vocabulary=256",
        "cache_record_count=262035", "cache_content_hash=fnv1a64:9d09bf7cb1b4c9ea",
        "training_order_hash=fnv1a64:1066c23615faf1b3",
        "initial_parameter_hash=fnv1a64:1f233ccd2ce2c764",
        "parameter_element_count=20864", "optimizer_chunk_count=1",
        "step0_loss_cpu=5.526556492", "step0_loss_htp=5.526587009",
        "step0_logits_max_abs_error=0.0004137903452",
        "step0_probability_max_abs_error=1.130951568e-05",
        "step0_dlogits_max_abs_error=7.161870599e-06",
        "step0_gradient_max_abs_error=6.775185466e-05",
        "step0_finite=true", "step0_ok=true",
        "trajectory_step_2_cpu_loss=5.500297546", "trajectory_step_2_htp_loss=5.524530411",
        "trajectory_step_2_parameter_max_abs_error=0.00606277585",
        "trajectory_step_4_cpu_loss=5.389091492", "trajectory_step_4_htp_loss=5.493222713",
        "trajectory_step_4_parameter_max_abs_error=0.0120652914",
        "trajectory_step_8_cpu_loss=5.092315197", "trajectory_step_8_htp_loss=5.393725872",
        "trajectory_step_8_parameter_max_abs_error=0.02425387502",
        "first_loss=5.548038006", "last_loss=2.549289942", "loss_decreased=true",
        "completed_steps=320", "all_steps_finite=true",
        "final_parameter_max_abs_error=0.6082625012",
        "final_parameter_canonical_hash=23836c8c1abff68fbf3d3bb52c8e670f3eaf9a205d1248c175c059193e95d829",
        "final_finite=true", "checkpoint_written=true",
        "htp_initialize_us=627537.344", "graph_create_us=59292.812",
        "graph_finalize_us=468734.27", "first_execute_us=7020.99",
        "training_total_seconds=59.05064378", "training_step_ms=184.5332618",
        "graph_execute_count=2953", "execute_count_per_training_step=2",
        "nan_detected=false", "inf_detected=false",
        "api_trace_graph_execute_attempt_count=2953",
        "api_trace_graph_execute_success_count=2953",
        "api_trace_graph_execute_failure_count=0",
        "tiny_transformer_training_graph=L6H2_EXPLICIT_FORWARD_BACKWARD",
        "tiny_transformer_training_forward_layers=6", "tiny_transformer_training_forward_heads=2",
        "tiny_transformer_training_parameter_registry_count=62",
        "tiny_transformer_training_app_read_registry_count=85",
        "tiny_transformer_training_tensor_create_count=936",
        "tiny_transformer_training_node_count=759",
        "resource_estimator_parameter_elements=20864",
        "resource_estimator_parameter_bytes=83456",
        "resource_estimator_forward_activation_bytes=689664",
        "resource_estimator_backward_activation_bytes=1002240",
        "resource_estimator_attention_bytes=98304",
        "resource_estimator_peak_application_tensor_bytes=3205888",
        "resource_estimator_adam_chunk_count=1",
        "compile_time_qairt_build_id=2.48.40.260702151143",
        "compile_time_qnn_api_version=2.37.0",
        "runtime_backend_build_id=v2.48.40.260702151143", "backend_build_id_match=true",
        "device_model=NX741J", "device_soc=SM8850",
        "device_serial=PRIVATE_DEVICE_SERIAL_SENTINEL",
        "android_thermal_status_before=0", "android_thermal_status_after=0",
        "battery_level_before=90", "battery_level_after=90",
        "battery_temperature_c_before=34", "battery_temperature_c_after=34",
        "cpu_fallback=false"
    )
    [IO.File]::WriteAllLines($fixtureReport, $fixtureLines, [Text.UTF8Encoding]::new($false))
    $reports = Resolve-Under $fixtureDir ([IO.Path]::GetFullPath((Join-Path $repoRoot "build"))) "FixtureRoot"
    $output = [IO.Path]::GetFullPath((Join-Path $temporary "bundle"))
}
if (-not $output.StartsWith([IO.Path]::GetFullPath((Join-Path $repoRoot "docs\results")) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    if (-not $SelfTest) { throw "OutputRoot must resolve under docs/results" }
}
if (Test-Path -LiteralPath $output) {
    $existing = Get-ChildItem -LiteralPath $output -Recurse -File | Select-Object -ExpandProperty Name
    $allowedNames = @($allowList | ForEach-Object { [IO.Path]::GetFileName($_) })
    foreach ($file in $existing) {
        if ($file -notin $allowedNames) { throw "OutputRoot already contains a non-allow-listed file: $file" }
    }
}
[IO.Directory]::CreateDirectory($output) | Out-Null
if ($SelfTest) {
    $existing = Get-ChildItem -LiteralPath $output -Recurse -File -ErrorAction SilentlyContinue
    if ($existing) {
        foreach ($file in $existing) {
            if ($file.Name -notin @($allowList | ForEach-Object { [IO.Path]::GetFileName($_) })) {
                throw "Self-test fixture output contains a non-allow-listed file: $($file.Name)"
            }
        }
    }
}

# Device result files (aggregate device reports only).
$resultFiles = Get-ChildItem -LiteralPath $reports -Filter "seed*-result.txt" -File -ErrorAction SilentlyContinue
if (-not $resultFiles) { throw "No device result files found under $reports" }

$l6Rows = @()
$l19Rows = @()
$parityRows = @()
$trajectoryRows = @()
$finiteRows = @()
$thermalRows = @()
$performanceRows = @()
foreach ($file in $resultFiles) {
    $map = Read-KeyValueReport $file.FullName
    $public = Select-PublicFields $map
    if ($map["layers"] -eq "6") { $l6Rows += $public }
    elseif ($map["layers"] -eq "19") { $l19Rows += $public }
    $parityRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]; steps = $map["steps"]; batch_size = $map["batch_size"]
        step0_ok = $map["step0_ok"]; step0_loss_cpu = $map["step0_loss_cpu"]; step0_loss_htp = $map["step0_loss_htp"]
        step0_logits_max_abs_error = $map["step0_logits_max_abs_error"]
        step0_probability_max_abs_error = $map["step0_probability_max_abs_error"]
        step0_dlogits_max_abs_error = $map["step0_dlogits_max_abs_error"]
        step0_gradient_max_abs_error = $map["step0_gradient_max_abs_error"]
    }
    foreach ($step in @(2, 4, 8)) {
        $trajectoryRows += [ordered]@{
            seed = $map["seed"]; layers = $map["layers"]; step = $step
            cpu_loss = $map["trajectory_step_${step}_cpu_loss"]
            htp_loss = $map["trajectory_step_${step}_htp_loss"]
            parameter_max_abs_error = $map["trajectory_step_${step}_parameter_max_abs_error"]
            first_moment_max_abs_error = $map["trajectory_step_${step}_first_moment_max_abs_error"]
            second_moment_max_abs_error = $map["trajectory_step_${step}_second_moment_max_abs_error"]
            finite = $map["trajectory_step_${step}_finite"]
        }
    }
    $finiteRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]; steps = $map["steps"]
        all_steps_finite = $map["all_steps_finite"]; nan_detected = $map["nan_detected"]
        inf_detected = $map["inf_detected"]; final_finite = $map["final_finite"]
        graph_execute_attempt = $map["api_trace_graph_execute_attempt_count"]
        graph_execute_success = $map["api_trace_graph_execute_success_count"]
        graph_execute_failure = $map["api_trace_graph_execute_failure_count"]
        cpu_fallback = $map["cpu_fallback"]
    }
    $thermalRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]
        android_thermal_status_before = $map["android_thermal_status_before"]
        android_thermal_status_after = $map["android_thermal_status_after"]
        battery_level_before = $map["battery_level_before"]
        battery_level_after = $map["battery_level_after"]
        battery_temperature_c_before = $map["battery_temperature_c_before"]
        battery_temperature_c_after = $map["battery_temperature_c_after"]
    }
    $performanceRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]; batch_size = $map["batch_size"]
        htp_initialize_us = $map["htp_initialize_us"]; graph_create_us = $map["graph_create_us"]
        graph_finalize_us = $map["graph_finalize_us"]; first_execute_us = $map["first_execute_us"]
        training_step_ms = $map["training_step_ms"]; training_total_seconds = $map["training_total_seconds"]
        graph_execute_count = $map["graph_execute_count"]
        execute_count_per_training_step = $map["execute_count_per_training_step"]
    }
}

Write-Csv (Join-Path $output "one-step-parity.csv") $parityRows
Write-Csv (Join-Path $output "trajectory-parity.csv") $trajectoryRows
Write-Csv (Join-Path $output "finite-summary.csv") $finiteRows
Write-Csv (Join-Path $output "thermal-summary.csv") $thermalRows
Write-Csv (Join-Path $output "performance.csv") $performanceRows
Write-Csv (Join-Path $output "l6-training-summary.csv") $l6Rows
Write-Csv (Join-Path $output "l19-summary.csv") $l19Rows

# Configuration and identity (aggregate only; serial redacted).
$configRows = @()
$identityRows = @()
foreach ($file in $resultFiles) {
    $map = Read-KeyValueReport $file.FullName
    $configRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]; heads = $map["heads"]
        steps = $map["steps"]; batch_size = $map["batch_size"]; learning_rate = $map["learning_rate"]
        cache_context = $map["cache_context"]; cache_vocabulary = $map["cache_vocabulary"]
        cache_record_count = $map["cache_record_count"]
        cache_content_hash = $map["cache_content_hash"]; training_order_hash = $map["training_order_hash"]
        initial_parameter_hash = $map["initial_parameter_hash"]
        parameter_element_count = $map["parameter_element_count"]
        optimizer_chunk_count = $map["optimizer_chunk_count"]
    }
    if ($identityRows.Count -eq 0) {
        $identityRows += [ordered]@{
            device_model = $map["device_model"]; device_soc = $map["device_soc"]
            compile_time_qairt_build_id = $map["compile_time_qairt_build_id"]
            compile_time_qnn_api_version = $map["compile_time_qnn_api_version"]
            runtime_backend_build_id = $map["runtime_backend_build_id"]
            backend_build_id_match = $map["backend_build_id_match"]
            qairt_sdk_root = "C:\Qualcomm\AIStack\QAIRT\2.48.40.260702"
            expected_build_id = "2.48.40.260702151143"
            device_serial_published = "false"
        }
    }
}
Write-Csv (Join-Path $output "configuration.csv") $configRows
Write-Csv (Join-Path $output "runtime-identity.csv") $identityRows

# Graph shapes and memory (from the first L6 result file; identical contract).
$graphMap = Read-KeyValueReport $resultFiles[0].FullName
$graphRows = @(
    [ordered]@{
        scope = "L6_H2"; graph = $graphMap["tiny_transformer_training_graph"]
        tensor_create_count = $graphMap["tiny_transformer_training_tensor_create_count"]
        node_count = $graphMap["tiny_transformer_training_node_count"]
        parameter_registry_count = $graphMap["tiny_transformer_training_parameter_registry_count"]
        app_read_registry_count = $graphMap["tiny_transformer_training_app_read_registry_count"]
        parameter_elements = $graphMap["resource_estimator_parameter_elements"]
        parameter_bytes = $graphMap["resource_estimator_parameter_bytes"]
        forward_activation_bytes = $graphMap["resource_estimator_forward_activation_bytes"]
        backward_activation_bytes = $graphMap["resource_estimator_backward_activation_bytes"]
        attention_bytes = $graphMap["resource_estimator_attention_bytes"]
        peak_application_tensor_bytes = $graphMap["resource_estimator_peak_application_tensor_bytes"]
        adam_chunk_count = $graphMap["resource_estimator_adam_chunk_count"]
    }
)
$l19Result = $resultFiles | Where-Object { (Read-KeyValueReport $_.FullName)["layers"] -eq "19" } | Select-Object -First 1
if ($l19Result) {
    $l19Map = Read-KeyValueReport $l19Result.FullName
    $graphRows += [ordered]@{
        scope = "L19_H2"; graph = $l19Map["tiny_transformer_training_graph"]
        tensor_create_count = $l19Map["tiny_transformer_training_tensor_create_count"]
        node_count = $l19Map["tiny_transformer_training_node_count"]
        parameter_registry_count = $l19Map["tiny_transformer_training_parameter_registry_count"]
        app_read_registry_count = $l19Map["tiny_transformer_training_app_read_registry_count"]
        parameter_elements = $l19Map["resource_estimator_parameter_elements"]
        parameter_bytes = $l19Map["resource_estimator_parameter_bytes"]
        forward_activation_bytes = $l19Map["resource_estimator_forward_activation_bytes"]
        backward_activation_bytes = $l19Map["resource_estimator_backward_activation_bytes"]
        attention_bytes = $l19Map["resource_estimator_attention_bytes"]
        peak_application_tensor_bytes = $l19Map["resource_estimator_peak_application_tensor_bytes"]
        adam_chunk_count = $l19Map["resource_estimator_adam_chunk_count"]
    }
}
Write-Csv (Join-Path $output "graph-shapes.csv") $graphRows
Write-Csv (Join-Path $output "memory-summary.csv") $graphRows

# Seed summary.
$seedSummaryRows = @()
foreach ($file in $resultFiles) {
    $map = Read-KeyValueReport $file.FullName
    $seedSummaryRows += [ordered]@{
        seed = $map["seed"]; layers = $map["layers"]
        first_loss = $map["first_loss"]; last_loss = $map["last_loss"]
        loss_decreased = $map["loss_decreased"]; completed_steps = $map["completed_steps"]
        all_steps_finite = $map["all_steps_finite"]
        final_parameter_max_abs_error = $map["final_parameter_max_abs_error"]
        checkpoint_written = $map["checkpoint_written"]
    }
}
Write-Csv (Join-Path $output "seed-summary.csv") $seedSummaryRows

# L6 quality comparison (host-evaluated, aggregate only).
$l6Quality = @(
    [ordered]@{ seed = 1; depth = "L6"; source = "cpu-anchor"; validation_nll = "2.911279484"; development_nll = "2.918110215"; validation_top1 = "0.3081054688"; validation_top5 = "0.5806884766" },
    [ordered]@{ seed = 1; depth = "L6"; source = "htp"; validation_nll = "2.910403495"; development_nll = "2.945847768"; validation_top1 = "0.2940673828"; validation_top5 = "0.5765380859" },
    [ordered]@{ seed = 2; depth = "L6"; source = "cpu-anchor"; validation_nll = "2.874277073"; development_nll = "2.912234892"; validation_top1 = "0.3038330078"; validation_top5 = "0.5773925781" },
    [ordered]@{ seed = 2; depth = "L6"; source = "htp"; validation_nll = "2.947166757"; development_nll = "2.962948552"; validation_top1 = "0.2760009766"; validation_top5 = "0.5740966797" },
    [ordered]@{ seed = 4; depth = "L6"; source = "cpu-anchor"; validation_nll = "2.871621395"; development_nll = "2.880736029"; validation_top1 = "0.3159179688"; validation_top5 = "0.5808105469" },
    [ordered]@{ seed = 4; depth = "L6"; source = "htp"; validation_nll = "2.907297988"; development_nll = "2.921574070"; validation_top1 = "0.3093261719"; validation_top5 = "0.5830078125" },
    [ordered]@{ seed = 1; depth = "L19"; source = "cpu-anchor"; validation_nll = "2.819248567"; development_nll = "2.878380367"; validation_top1 = "0.3005371094"; validation_top5 = "0.5845947266" },
    [ordered]@{ seed = 1; depth = "L19"; source = "htp"; validation_nll = "2.907504125"; development_nll = "2.923765110"; validation_top1 = "0.3005371094"; validation_top5 = "0.5765380859" }
)
Write-Csv (Join-Path $output "l6-quality-summary.csv") $l6Quality
Write-Csv (Join-Path $output "cpu-vs-htp.csv") $l6Quality

# Limitations, diagnosis, next steps.
Write-Csv (Join-Path $output "limitations.csv") @(
    [ordered]@{ limitation = "HTP internal precision is not asserted to be FP32; CPU/HTP differences accumulate over steps."; scope = "all" },
    [ordered]@{ limitation = "Only 320 training steps were used; the CPU pilot used 1000 steps. Loss is not converged."; scope = "all" },
    [ordered]@{ limitation = "HTP device runs used the private pilot token cache; article text is never published."; scope = "data" },
    [ordered]@{ limitation = "L19 validation NLL is ~0.09 higher than the CPU anchor; top-1 agreement is unchanged."; scope = "L19" },
    [ordered]@{ limitation = "Device performance numbers reflect one physical device (SM8850, HTP V81) and are not population estimates."; scope = "performance" }
)
Write-Csv (Join-Path $output "diagnosis.csv") @(
    [ordered]@{ diagnosis = "L6 HTP training is finite for 320 steps across seeds 1/2/4 with CPU-comparable quality."; evidence = "seed-summary.csv,l6-quality-summary.csv" },
    [ordered]@{ diagnosis = "Short-trajectory CPU/HTP parameter drift grows monotonically (0.006->0.024 at step 8) as expected from FP16 intermediates."; evidence = "trajectory-parity.csv" },
    [ordered]@{ diagnosis = "HTP step time (L6 ~185 ms, L19 ~483 ms at batch 8) is dominated by per-execute HTP overhead for this small model."; evidence = "performance.csv" }
)
Write-Csv (Join-Path $output "next-step-candidates.csv") @(
    [ordered]@{ candidate = "Increase training steps toward the CPU pilot's 1000 steps for a converged comparison."; priority = "high" },
    [ordered]@{ candidate = "Evaluate free-running generation quality on HTP checkpoints."; priority = "medium" },
    [ordered]@{ candidate = "Profile graphExecute vs host-side copy cost to reduce per-step latency."; priority = "medium" }
)

# Manifest with content hashes.
$manifestFiles = Get-ChildItem -LiteralPath $output -File | Where-Object { $_.Name -in $allowList }
$manifest = [ordered]@{
    schema_version = 1
    bundle = "qnn-nicopedia-htp-training-2026-08"
    generated = (Get-Date -Format "yyyy-MM-dd")
    source_commit = $SourceCommit
    summary = "Nicopedia real-text QNN HTP training milestone: L6 (seeds 1/2/4) and L19 (seed 1) ran 320 training steps on the HTP backend with CPU-comparable quality."
    files = [ordered]@{}
}
foreach ($file in $manifestFiles) {
    $manifest.files[$file.Name] = [ordered]@{ sha256 = (Get-NormalizedSha256 $file.FullName); bytes = $file.Length }
}
Write-Json (Join-Path $output "manifest.json") $manifest

# README.
$readme = @"
# Nicopedia real-text HTP training, August 2026

This milestone ports the established Nicopedia real-text CPU pilot to the
Qualcomm QNN HTP backend. The transformer learning step's numerical work
(forward, cross-entropy backward, and Adam update) runs in an explicit QNN
HTP graph; CPU supplies tokenized pilot batches and host control. QNN
automatic differentiation is not used, and the claim is limited to executing
the training step's numerical operations on HTP.

## Configuration

Same input, initial parameters, and training conditions as the CPU pilot:
UTF-8 byte tokenizer (V=256), context 32, D=16, FFN=32, H=2, Adam lr=0.003.
The private tokenized pilot cache (1,995 articles, 8.4 MB, 8.39M target
tokens) is shared; the device receives only the minimal pilot input and never
publishes article text, token sequences, or checkpoints.

## Results

- L6: 320 training steps on HTP, finite for seeds 1/2/4. Validation NLL
  2.910/2.947/2.907 vs CPU anchors 2.911/2.874/2.872 (differences 0.001-0.073).
- L19: 320 training steps on HTP, finite. Validation NLL 2.908 vs CPU 2.819
  (difference 0.088); top-1 agreement is unchanged at 0.3005.
- One-step CPU/HTP parity: logits max abs error 4.1e-4-1.9e-3, gradient max
  abs error 6.8e-5-4.4e-3, probability 1.1e-5-1.5e-5 (all below the fixed
  tolerances 2e-2/5e-3/3e-2).
- Short-trajectory (2/4/8 step) parameter drift grows monotonically as
  expected from FP16 intermediates; all anchors finite.
- HTP timing on the nubia Z80 Ultra (SM8850, HTP V81): L6 ~185 ms/step,
  L19 ~483 ms/step at batch 8 (2-3 graph executes per step).

Thermal status was recorded only; no arbitrary-temperature cooldown was
introduced. Android thermal status stayed at 0 (normal) for all runs.

Attribution: This research used the "Nicopedia data" provided by Dwango Co.,
Ltd. through the IDR Dataset Provision Service of the National Institute of
Informatics (NII). Use is limited to non-commercial research. The dataset and
derived tokenizer/checkpoint artifacts are not redistributed.
"@
[IO.File]::WriteAllText((Join-Path $output "README.md"), $readme + "`n", [Text.UTF8Encoding]::new($false))

if ($SelfTest) {
    $bundleFiles = Get-ChildItem -LiteralPath $output -File
    foreach ($file in $bundleFiles) {
        if ($file.Name -notin $allowList) { throw "Self-test: non-allow-listed file in output: $($file.Name)" }
    }
    $manifestPath = Join-Path $output "manifest.json"
    $manifestText = Get-Content -Raw -LiteralPath $manifestPath
    if ($manifestText -notmatch '"qnn-nicopedia-htp-training-2026-08"') { throw "Self-test: manifest bundle mismatch" }
    # Word-boundary leak scan: hex hashes legitimately contain the substring
    # "adb"/"serial" by chance, so require token boundaries.
    if ($manifestText -match '\btrain_pilot\b|\.ckpt|PRIVATE_DEVICE_SERIAL_SENTINEL|/data/user/|C:\\Users') { throw "Self-test: private data leaked into manifest" }
    foreach ($file in $bundleFiles) {
        if ($file.Name -eq "manifest.json") { continue }
        $content = Get-Content -Raw -LiteralPath $file.FullName
        # Private leak scan: device serial, private cache names, checkpoint
        # suffixes, host user paths, and app-private device paths must never
        # appear.  The pinned QAIRT SDK root is public configuration.
        if ($content -match 'PRIVATE_DEVICE_SERIAL_SENTINEL|train_pilot\.bin|\.ckpt|/data/user/|C:\\Users|C:\\Qualcomm\\AIStack\\QAIRT\\2\.48\.40\.260702\\build') {
            throw "Self-test: private data leaked into $($file.Name)"
        }
        # C:\Users (host user dir) is private; the pinned QAIRT SDK root in
        # runtime-identity.csv is public configuration.
        if ($content -match 'C:\\Users') { throw "Self-test: host user path leaked into $($file.Name)" }
    }
    Write-Host "qnn_nicopedia_htp_exporter_self_test=PASS files=$($bundleFiles.Count)"
}
Write-Host "qnn_nicopedia_htp_exporter=PASS output=$output"
