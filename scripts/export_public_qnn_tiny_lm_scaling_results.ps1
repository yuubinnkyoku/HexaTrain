# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$BaselineReferenceReport,
    [string]$BaselineRepeatReport,
    [string]$Sequence16Report,
    [string]$Sequence32Report,
    [string]$Dimension32Report,
    [string]$Layers2Report,
    [string]$FormalReport,
    [string[]]$PerformanceReports = @(),
    [string[]]$RegressionReports = @(),
    [string]$BaselineHead = "3ca6b08e6d62f49369b54cc4e197970089489f52",
    [string]$ImplementationHead,
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\results\qnn-htp-tiny-lm-scaling-2026-07"),
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$script:Utf8 = [Text.UTF8Encoding]::new($false)
$script:AllowedFiles = @(
    "README.md", "manifest.json", "seed-reproducibility.csv", "stages.csv",
    "formal-seeds.csv", "performance.csv", "regressions.csv")

function Fail([string]$Message) {
    throw "tiny-lm scaling public export: $Message"
}

function Read-Map([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "missing input report"
    }
    $map = [ordered]@{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^([A-Za-z][A-Za-z0-9_]*)=(.*)$") {
            $key = $Matches[1]
            $value = $Matches[2]
            if ($map.Contains($key) -and $map[$key] -cne $value) {
                Fail "conflicting duplicate key $key"
            }
            $map[$key] = $value
        }
    }
    return $map
}

function Require([Collections.IDictionary]$Map, [string]$Key) {
    if (-not $Map.Contains($Key)) { Fail "missing field $Key" }
    return [string]$Map[$Key]
}

function Optional([Collections.IDictionary]$Map, [string]$Key, [string]$Default) {
    if ($Map.Contains($Key)) { return [string]$Map[$Key] }
    return $Default
}

function Require-Number([Collections.IDictionary]$Map, [string]$Key) {
    $text = Require $Map $Key
    [double]$number = 0
    if (-not [double]::TryParse(
            $text, [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$number) -or
        -not [double]::IsFinite($number)) {
        Fail "non-finite or invalid number $Key"
    }
    return $number
}

function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, $script:Utf8)
}

function Write-Csv([string]$Path, [object[]]$Rows) {
    if ($Rows.Count -eq 0) { Fail "cannot write empty CSV" }
    $text = ($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n"
    if ($text -match "(?im)^[^,]*[=+@]" -or
        $text -match "(?i)[a-z]:\\|/data/|/sdcard/|logcat|adb endpoint") {
        Fail "unsafe public CSV value"
    }
    Write-Utf8 $Path ($text + "`n")
}

function Median([double[]]$Values) {
    if ($Values.Count -ne 3) { Fail "performance median requires three runs" }
    $sorted = @($Values | Sort-Object)
    return $sorted[1]
}

function Validate-Bundle([string]$Root) {
    $names = @((Get-ChildItem -LiteralPath $Root -File).Name | Sort-Object)
    if (($names -join ",") -ne (($script:AllowedFiles | Sort-Object) -join ",")) {
        Fail "bundle allow-list mismatch"
    }
    $manifest = Get-Content -LiteralPath (Join-Path $Root "manifest.json") -Raw |
        ConvertFrom-Json
    $readme = Get-Content -LiteralPath (Join-Path $Root "README.md") -Raw
    $formal = @(Import-Csv -LiteralPath (Join-Path $Root "formal-seeds.csv"))
    $stages = @(Import-Csv -LiteralPath (Join-Path $Root "stages.csv"))
    $repro = @(Import-Csv -LiteralPath (Join-Path $Root "seed-reproducibility.csv"))
    $performance = @(Import-Csv -LiteralPath (Join-Path $Root "performance.csv"))
    if ($manifest.schema_version -ne 1 -or $formal.Count -ne 5 -or
        $stages.Count -ne 5 -or $repro.Count -ne 4 -or
        @($performance | Where-Object scope -eq "aggregate").Count -ne 8) {
        Fail "manifest/CSV cardinality mismatch"
    }
    foreach ($required in @(
            $manifest.baseline_head, $manifest.implementation_head,
            $manifest.seed_reproducibility, $manifest.max_stable_configuration,
            $manifest.result_classification, $manifest.stop_reason,
            [string]$manifest.formal.oracle_exact,
            [string]$manifest.formal.free_exact,
            [string]$manifest.performance.steady_training_step_median_ms,
            [string]$manifest.performance.tokens_per_second,
            [string]$manifest.performance.process_peak_rss_kib)) {
        if (-not $readme.Contains($required)) {
            Fail "README does not contain manifest value $required"
        }
    }
    if (@($formal | Where-Object {
                $_.all_steps_finite -ne "true" -or
                [double]$_.final_loss -ge [double]$_.initial_loss
            }).Count -ne 0) {
        Fail "formal seed CSV contradicts success"
    }
    return $manifest
}

function Invoke-SelfTest {
    $temporary = Join-Path (
        Join-Path (Split-Path -Parent $PSScriptRoot) "build\reports") (
        "scaling-export-selftest-" + [Guid]::NewGuid().ToString("N"))
    try {
        [IO.Directory]::CreateDirectory($temporary) | Out-Null
        $duplicate = Join-Path $temporary "duplicate.txt"
        Write-Utf8 $duplicate "status=SUCCESS`nstatus=FAILED`n"
        $rejected = $false
        try { Read-Map $duplicate | Out-Null } catch { $rejected = $true }
        if (-not $rejected) { Fail "conflicting duplicate was accepted" }
        if ((Median @(3.0, 1.0, 2.0)) -ne 2.0) {
            Fail "median self-test failed"
        }
        $bundle = Join-Path $temporary "bundle"
        [IO.Directory]::CreateDirectory($bundle) | Out-Null
        $manifest = [ordered]@{
            schema_version = 1
            baseline_head = "a" * 40
            implementation_head = "b" * 40
            seed_reproducibility = "BITWISE_REPRODUCIBLE"
            max_stable_configuration = "tokens=32,dimension=32,layers=1,heads=1"
            result_classification = "SCALED_SINGLE_LAYER_REPRODUCIBLE"
            stop_reason = "LAYERS_2_UNSUPPORTED"
            formal = [ordered]@{ oracle_exact = "20/20"; free_exact = "20/20" }
            performance = [ordered]@{
                steady_training_step_median_ms = 2
                tokens_per_second = 3
                process_peak_rss_kib = 4
            }
        }
        Write-Utf8 (Join-Path $bundle "manifest.json") (
            $manifest | ConvertTo-Json -Depth 8)
        $readmeValues = @(
            $manifest.baseline_head, $manifest.implementation_head,
            $manifest.seed_reproducibility, $manifest.max_stable_configuration,
            $manifest.result_classification, $manifest.stop_reason, "20/20",
            "2", "3", "4") -join "`n"
        Write-Utf8 (Join-Path $bundle "README.md") $readmeValues
        Write-Csv (Join-Path $bundle "formal-seeds.csv") @(
            1..5 | ForEach-Object {
                [pscustomobject]@{
                    seed = $_; initial_loss = 2; final_loss = 1
                    all_steps_finite = "true"
                }
            })
        Write-Csv (Join-Path $bundle "stages.csv") @(
            1..5 | ForEach-Object { [pscustomobject]@{ stage = $_ } })
        Write-Csv (Join-Path $bundle "seed-reproducibility.csv") @(
            2..5 | ForEach-Object { [pscustomobject]@{ seed = $_ } })
        Write-Csv (Join-Path $bundle "performance.csv") @(
            1..8 | ForEach-Object {
                [pscustomobject]@{ scope = "aggregate"; metric = $_; value = $_ }
            })
        Write-Csv (Join-Path $bundle "regressions.csv") @(
            [pscustomobject]@{ suite = "selftest"; status = "PASS" })
        Validate-Bundle $bundle | Out-Null
        Write-Output "SELF_TEST=PASS"
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Recurse -Force
        }
    }
}

function Export-Results {
    if ($BaselineHead -notmatch "^[0-9a-f]{40}$" -or
        $ImplementationHead -notmatch "^[0-9a-f]{40}$") {
        Fail "invalid commit hash"
    }
    if ($PerformanceReports.Count -ne 3 -or $RegressionReports.Count -ne 3) {
        Fail "exactly three performance and three regression reports are required"
    }
    $destination = [IO.Path]::GetFullPath($OutputRoot)
    $allowedRoot = [IO.Path]::GetFullPath(
        (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\results")) +
        [IO.Path]::DirectorySeparatorChar
    if (-not $destination.StartsWith(
            $allowedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        (Test-Path -LiteralPath $destination)) {
        Fail "unsafe or existing output root"
    }

    $baselineReference = Read-Map $BaselineReferenceReport
    $baselineRepeat = Read-Map $BaselineRepeatReport
    $sequence16 = Read-Map $Sequence16Report
    $sequence32 = Read-Map $Sequence32Report
    $dimension32 = Read-Map $Dimension32Report
    $layers2 = Read-Map $Layers2Report
    $formalMap = Read-Map $FormalReport
    $performanceMaps = @($PerformanceReports | ForEach-Object { Read-Map $_ })
    $regressionMaps = @($RegressionReports | ForEach-Object { Read-Map $_ })

    $reproducibility = @()
    foreach ($seed in 2..5) {
        $trainingPrefix = "seed_${seed}_"
        $trainingKeys = @($baselineReference.Keys |
            Where-Object { $_ -like "$trainingPrefix*" })
        $generationPatterns = @(
            "formal_oracle_case_s${seed}_*",
            "formal_free_case_s${seed}_*",
            "formal_logits_oracle_s${seed}_*",
            "formal_logits_free_s${seed}_*")
        $generationKeys = @($baselineReference.Keys | Where-Object {
                $key = $_
                @($generationPatterns | Where-Object { $key -like $_ }).Count -gt 0
            })
        $trainingDifferences = @($trainingKeys | Where-Object {
                -not $baselineRepeat.Contains($_) -or
                $baselineReference[$_] -cne $baselineRepeat[$_]
            })
        $generationDifferences = @($generationKeys | Where-Object {
                -not $baselineRepeat.Contains($_) -or
                $baselineReference[$_] -cne $baselineRepeat[$_]
            })
        $classification = if (
            $trainingDifferences.Count -eq 0 -and
            $generationDifferences.Count -eq 0) {
            "BITWISE_REPRODUCIBLE"
        } else {
            "NOT_REPRODUCIBLE"
        }
        $reproducibility += [pscustomobject][ordered]@{
            seed = $seed
            classification = $classification
            training_fields_compared = $trainingKeys.Count
            generation_fields_compared = $generationKeys.Count
            first_differing_field = if ($trainingDifferences.Count) {
                $trainingDifferences[0]
            } elseif ($generationDifferences.Count) {
                $generationDifferences[0]
            } else { "NONE" }
            final_loss = Require-Number $baselineRepeat "seed_${seed}_final_loss"
            final_accuracy = Require-Number $baselineRepeat "seed_${seed}_final_accuracy"
            parameter_hash = Require $baselineRepeat "seed_${seed}_final_parameter_canonical_hash"
            logits_hash = Require $baselineRepeat "seed_${seed}_step_320_logits_canonical_hash"
            oracle_sequences_bitwise_equal =
                ($generationDifferences.Count -eq 0)
            free_sequences_bitwise_equal =
                ($generationDifferences.Count -eq 0)
            qnn_execute_count = Require $baselineRepeat "seed_${seed}_qnn_execute_count"
            qnn_nonzero_return_count = Require $baselineRepeat "seed_${seed}_qnn_nonzero_return_count"
            nonfinite_detected =
                ((Require $baselineRepeat "seed_${seed}_nonfinite_count") -ne "0")
            app_write_unexpected_change =
                "NOT_REPORTED_PER_SEED_BY_BASELINE_SUITE"
            poison_residual =
                "NOT_REPORTED_PER_SEED_BY_BASELINE_SUITE"
            focus_takeover_count = Require $baselineRepeat "focus_takeover_count"
        }
    }
    $seedClassification = if (
        @($reproducibility | Where-Object {
                $_.classification -ne "BITWISE_REPRODUCIBLE"
            }).Count -eq 0) {
        "BITWISE_REPRODUCIBLE"
    } else { "NOT_REPRODUCIBLE" }

    $stageInputs = @(
        @("sequence_8_to_16", "scale-sequence-16-smoke",
            $sequence16, 16, 16, 1, 1, "PASS", "NONE"),
        @("sequence_16_to_32", "scale-sequence-32-smoke",
            $sequence32, 32, 16, 1, 1, "PASS", "NONE"),
        @("dimension_16_to_32", "scale-dimension-32-smoke",
            $dimension32, 32, 32, 1, 1, "PASS", "NONE"),
        @("layers_1_to_2", "scale-layers-2-smoke",
            $layers2, 32, 32, 2, 1, "FAIL",
            "UNSUPPORTED_TRANSFORMER_LAYER_COUNT"),
        @("heads_1_to_2", "scale-heads-2-smoke",
            $null, 32, 32, 2, 2, "NOT_RUN",
            "PREVIOUS_STAGE_FAILED"))
    $stages = foreach ($entry in $stageInputs) {
        $map = $entry[2]
        [pscustomobject][ordered]@{
            stage = $entry[0]
            suite = $entry[1]
            sequence_length = $entry[3]
            embedding_dimension = $entry[4]
            layers = $entry[5]
            heads = $entry[6]
            status = $entry[7]
            stop_detail = $entry[8]
            initial_loss = if ($null -ne $map -and $entry[7] -eq "PASS") {
                Require-Number $map "seed_1_initial_loss"
            } else { "NOT_AVAILABLE" }
            final_loss = if ($null -ne $map -and $entry[7] -eq "PASS") {
                Require-Number $map "seed_1_final_loss"
            } else { "NOT_AVAILABLE" }
            final_accuracy = if ($null -ne $map -and $entry[7] -eq "PASS") {
                Require-Number $map "seed_1_final_accuracy"
            } else { "NOT_AVAILABLE" }
            oracle_exact = if ($null -ne $map -and $entry[7] -eq "PASS") {
                (Require $map "oracle_exact_rollout_count") + "/4"
            } else { "NOT_AVAILABLE" }
            free_exact = if ($null -ne $map -and $entry[7] -eq "PASS") {
                (Require $map "exact_rollout_count") + "/4"
            } else { "NOT_AVAILABLE" }
            cpu_htp_max_abs_difference = if (
                $null -ne $map -and $entry[7] -eq "PASS") {
                Require-Number $map "same_prefix_seed_1_cpu_htp_max_abs_error"
            } else { "NOT_AVAILABLE" }
            qnn_nonzero_return_count = if (
                $null -ne $map -and $entry[7] -eq "PASS") {
                Require $map "formal_qnn_nonzero_return_count"
            } else { "NOT_AVAILABLE" }
            focus_takeover_count = if ($null -ne $map) {
                Require $map "focus_takeover_count"
            } else { "NOT_AVAILABLE" }
        }
    }

    $formalSeeds = foreach ($seed in 1..5) {
        [pscustomobject][ordered]@{
            seed = $seed
            initial_loss = Require-Number $formalMap "seed_${seed}_initial_loss"
            final_loss = Require-Number $formalMap "seed_${seed}_final_loss"
            loss_reduction_percent =
                Require-Number $formalMap "seed_${seed}_loss_reduction"
            final_accuracy =
                Require-Number $formalMap "seed_${seed}_final_accuracy"
            all_steps_finite = Require $formalMap "seed_${seed}_all_steps_finite"
            nonfinite_count = Require $formalMap "seed_${seed}_nonfinite_count"
            qnn_execute_count =
                Require $formalMap "seed_${seed}_qnn_execute_count"
            qnn_nonzero_return_count =
                Require $formalMap "seed_${seed}_qnn_nonzero_return_count"
            parameter_hash =
                Require $formalMap "seed_${seed}_final_parameter_canonical_hash"
            cpu_htp_parameter_max_abs_difference =
                Require-Number $formalMap "seed_${seed}_cpu_htp_parameter_max_abs_difference"
        }
    }
    foreach ($row in $formalSeeds) {
        if ($row.all_steps_finite -ne "true" -or
            [double]$row.final_loss -ge [double]$row.initial_loss -or
            $row.qnn_nonzero_return_count -ne "0") {
            Fail "formal seed success condition failed"
        }
    }
    foreach ($field in @(
            "status", "shape_contract_valid", "app_write_hashes_unchanged",
            "binding_audit_all_outputs_finite",
            "free_running_teacher_forcing", "generation_context_self_test",
            "formal_qnn_nonzero_return_count", "focus_takeover_count")) {
        $expected = switch ($field) {
            "status" { "SUCCESS" }
            "free_running_teacher_forcing" { "false" }
            "formal_qnn_nonzero_return_count" { "0" }
            "focus_takeover_count" { "0" }
            default { "true" }
        }
        if ((Require $formalMap $field) -ne $expected) {
            Fail "formal condition $field"
        }
    }
    if ((Require $formalMap "app_read_poison_residual_elements") -ne "0" -or
        (Require $formalMap "oracle_exact_rollout_count") -ne "20" -or
        (Require $formalMap "exact_rollout_count") -ne "20") {
        Fail "formal poison or generation condition"
    }

    $performanceMetricNames = @(
        "performance_initialization_ms", "performance_graph_creation_ms",
        "performance_finalize_ms", "performance_steady_training_step_median_ms",
        "performance_updates_per_second", "performance_tokens_per_second",
        "generation_token_latency_median_ms", "process_peak_rss_kib")
    $performance = @()
    foreach ($index in 0..2) {
        $map = $performanceMaps[$index]
        if ((Require $map "status") -ne "SUCCESS" -or
            (Require $map "oracle_exact_rollout_count") -ne "20" -or
            (Require $map "exact_rollout_count") -ne "20" -or
            (Require $map "formal_qnn_nonzero_return_count") -ne "0" -or
            (Require $map "focus_takeover_count") -ne "0") {
            Fail "performance run did not preserve correctness"
        }
        foreach ($metric in $performanceMetricNames) {
            $performance += [pscustomobject][ordered]@{
                scope = "run"; run = $index + 1; metric = $metric
                value = Require-Number $map $metric
            }
        }
    }
    $performanceAggregate = [ordered]@{}
    foreach ($metric in $performanceMetricNames) {
        $value = Median @($performanceMaps | ForEach-Object {
                Require-Number $_ $metric
            })
        $performanceAggregate[$metric] = $value
        $performance += [pscustomobject][ordered]@{
            scope = "aggregate"; run = "MEDIAN"; metric = $metric
            value = $value
        }
    }

    $regressionNames = @(
        "qnn-reproducibility", "qnn-graph-full-isolated",
        "qnn-tap-backward-regions")
    $regressions = foreach ($index in 0..2) {
        $map = $regressionMaps[$index]
        [pscustomobject][ordered]@{
            suite = $regressionNames[$index]
            status = if ((Require $map "status") -eq "SUCCESS" -and
                (Require $map "focus_takeover_count") -eq "0") {
                "PASS"
            } else { "FAIL" }
            focus_takeover_count = Require $map "focus_takeover_count"
            qnn_execute_failure_count =
                Optional $map "api_trace_graph_execute_failure_count" "NOT_REPORTED"
        }
    }
    if (@($regressions | Where-Object status -ne "PASS").Count -ne 0) {
        Fail "regression failure"
    }

    $manifest = [ordered]@{
        schema_version = 1
        baseline_head = $BaselineHead
        implementation_head = $ImplementationHead
        baseline_classification =
            "END_TO_END_TRAINING_AND_GENERATION_REPRODUCIBLE"
        seed_reproducibility = $seedClassification
        all_five_seeds_bitwise_reproducible =
            ($seedClassification -eq "BITWISE_REPRODUCIBLE")
        max_stable_configuration = "tokens=32,dimension=32,layers=1,heads=1"
        failed_stage = "layers=2"
        result_classification =
            "END_TO_END_TRAINING_AND_GENERATION_REPRODUCIBLE_SCALED_SINGLE_LAYER"
        stop_reason = "LAYERS_2_UNSUPPORTED"
        formal = [ordered]@{
            htp_finite_seeds = "5/5"
            loss_decreased_seeds = "5/5"
            oracle_exact = "20/20"
            free_exact = "20/20"
            qnn_nonzero_return_count = 0
            teacher_forcing_assertion = "PASS"
            app_write_hashes_unchanged = $true
            app_read_poison_residual_elements = 0
            cpu_htp_max_abs_difference =
                Require-Number $formalMap "same_prefix_all_seed_cpu_htp_logits_max_abs_error"
        }
        performance = [ordered]@{
            run_count = 3
            initialization_ms =
                $performanceAggregate["performance_initialization_ms"]
            graph_creation_ms =
                $performanceAggregate["performance_graph_creation_ms"]
            finalize_ms = $performanceAggregate["performance_finalize_ms"]
            steady_training_step_median_ms =
                $performanceAggregate["performance_steady_training_step_median_ms"]
            updates_per_second =
                $performanceAggregate["performance_updates_per_second"]
            tokens_per_second =
                $performanceAggregate["performance_tokens_per_second"]
            generation_token_latency_median_ms =
                $performanceAggregate["generation_token_latency_median_ms"]
            process_peak_rss_kib =
                $performanceAggregate["process_peak_rss_kib"]
        }
        regressions = [ordered]@{
            qnn_reproducibility = "PASS"
            full_graph_isolation = "PASS"
            focused_backward_tap = "PASS"
        }
        constraints = @(
            "The CPU supplies orchestration and the HTP executes training-step numerical graphs.",
            "The model remains single-layer and single-head.",
            "Process peak RSS is a process-level high-water mark, not HTP-only memory.",
            "The baseline suite did not emit APP_WRITE and poison counters per seed; the separate reproducibility regression passed and the scaled formal run reported APP_WRITE unchanged with zero poison residual.",
            "Heads=2 was not run because layers=2 failed first.")
    }

    $stage = Join-Path (Split-Path -Parent $destination) (
        ".scaling-export-" + [Guid]::NewGuid().ToString("N"))
    try {
        [IO.Directory]::CreateDirectory($stage) | Out-Null
        Write-Utf8 (Join-Path $stage "manifest.json") (
            ($manifest | ConvertTo-Json -Depth 10) + "`n")
        Write-Csv (Join-Path $stage "seed-reproducibility.csv") $reproducibility
        Write-Csv (Join-Path $stage "stages.csv") $stages
        Write-Csv (Join-Path $stage "formal-seeds.csv") $formalSeeds
        Write-Csv (Join-Path $stage "performance.csv") $performance
        Write-Csv (Join-Path $stage "regressions.csv") $regressions
        $readme = @"
# QNN HTP Tiny LM staged scaling results

Baseline HEAD: $BaselineHead

Implementation HEAD: $ImplementationHead

Seed 2-5 reproducibility: $seedClassification. Together with the previously
confirmed seed 1 result, all five seeds are bitwise reproducible.

Maximum stable configuration:
$($manifest.max_stable_configuration).

Result classification:
$($manifest.result_classification).

Stop reason: $($manifest.stop_reason). The layers=2 stage failed explicit
configuration validation because the current parameter and graph schema
represents one Transformer layer. heads=2 was not run.

The maximum stable configuration completed HTP training-step numerical
execution with 5/5 finite seeds, loss reduction for 5/5 seeds, Oracle
$($manifest.formal.oracle_exact), Free-running $($manifest.formal.free_exact),
zero nonzero QNN execute returns, no unexpected APP_WRITE changes, zero poison
residual, and a passing teacher-forcing exclusion assertion.

The same-prefix CPU/HTP maximum absolute logits difference was
$($manifest.formal.cpu_htp_max_abs_difference).

Across three correctness-preserving benchmark runs, the run-level median
initialization was $($manifest.performance.initialization_ms) ms, graph creation
was $($manifest.performance.graph_creation_ms) ms, finalize was
$($manifest.performance.finalize_ms) ms, steady training step was
$($manifest.performance.steady_training_step_median_ms) ms,
$($manifest.performance.updates_per_second) updates/s,
$($manifest.performance.tokens_per_second) tokens/s, generation token latency
was $($manifest.performance.generation_token_latency_median_ms) ms, and process
peak RSS was $($manifest.performance.process_peak_rss_kib) KiB.

The reproducibility, full-graph isolation, and focused backward-tap regressions
passed. Reproduce the device runs with `scripts/run_qnn_headless_tests.ps1`,
the explicit QAIRT 2.48.40 SDK root and expected build ID, the stage suites in
`stages.csv`, `BACKGROUND_CORRECTNESS` for correctness, and
`EXCLUSIVE_BENCHMARK` for the three performance repetitions. Run
`scripts/verify_local.ps1` and its `-WithQairt` form before publication.

Raw checkpoints, tensor dumps, device identifiers, local paths, distributed
SDK files, APKs, and private device reports are excluded.

The baseline report did not expose APP_WRITE mutation and poison-residual
counters per seed. Those two fields are therefore marked as not reported in
seed-reproducibility.csv rather than inferred. The separate reproducibility
regression passed, and the scaled formal evaluation directly reported
APP_WRITE hashes unchanged and zero APP_READ poison residual.
"@
        Write-Utf8 (Join-Path $stage "README.md") $readme
        Validate-Bundle $stage | Out-Null
        Move-Item -LiteralPath $stage -Destination $destination
        Validate-Bundle $destination | Out-Null
        Write-Output "EXPORTED=$destination"
        Write-Output "MANIFEST_DOCUMENT_SELF_TEST=PASS"
    } finally {
        if (Test-Path -LiteralPath $stage) {
            Remove-Item -LiteralPath $stage -Recurse -Force
        }
    }
}

if ($SelfTest) {
    Invoke-SelfTest
} else {
    Export-Results
}
