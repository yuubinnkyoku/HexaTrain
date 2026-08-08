# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Allow-list exporter for the Nicopedia CPU/HTP parity policy audit (protocol
# docs/qnn-nicopedia-htp-parity-policy.md).
#
# Publishes the policy candidates, per-prefix parity/scale/decision metrics,
# the synthetic fault battery result, the adoption decision for the fixed
# criteria, and aggregate byte-level generation statistics. Raw generated
# bytes, prompts, logit vectors, checkpoint hashes, serials, ADB endpoints,
# absolute host paths, and content fingerprints NEVER enter the bundle.
param(
    [string]$ReportRoot = "build/reports/nicopedia-htp-generation",
    [string]$FaultResultRoot = "build/reports/nicopedia-parity-policy",
    [string]$CpuGenRoot = "build/private-diagnostics/nicopedia-htp-1000step-goal",
    [string]$OutputRoot = "docs/results/qnn-nicopedia-htp-parity-policy-2026-08",
    [string]$SourceCommit = "",
    [switch]$SelfTest,
    [switch]$ReviewerPass
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Under([string]$Path, [string]$AllowedRoot, [string]$Label) {
    $candidate = if ([IO.Path]::IsPathRooted($Path)) {
        [IO.Path]::GetFullPath($Path)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
    }
    $allowed = [IO.Path]::GetFullPath($AllowedRoot)
    if (-not $candidate.StartsWith($allowed + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must resolve under $allowed (got $candidate)"
    }
    return $candidate
}

function Get-NormalizedSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $bytes = $sha.ComputeHash($stream)
        return ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
    } finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

function Read-KeyValueReport([string]$Path) {
    $map = [ordered]@{}
    foreach ($line in [IO.File]::ReadAllLines($Path)) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            $map[$Matches[1]] = $Matches[2]
        }
    }
    return $map
}

function Write-Csv([string]$Path, [object[]]$Rows) {
    if (-not $Rows -or $Rows.Count -eq 0) { return }
    $headers = @($Rows[0].Keys)
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add(($headers -join ","))
    foreach ($row in $Rows) {
        $values = foreach ($h in $headers) {
            $v = $row[$h]
            if ($null -eq $v) { "" }
            elseif ($v -is [string] -and ($v -match ',' -or $v -match '"')) {
                '"' + ($v -replace '"', '""') + '"'
            } else { $v }
        }
        $lines.Add(($values -join ","))
    }
    [IO.File]::WriteAllLines($Path, $lines, [Text.UTF8Encoding]::new($false))
}

function Write-Json([string]$Path, [object]$Value) {
    $json = $Value | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($Path, $json + "`n", [Text.UTF8Encoding]::new($false))
}

# Selector: device generation reports expose ONLY aggregate/scale/decision
# fields (exact key spellings emitted by qnn_transformer_training.cpp).
function Select-PublicFields($Map) {
    $public = [ordered]@{}
    $allowed = @(
        "status","test","model","layers","heads","seed","checkpoint_step",
        "generate_mode","max_new_bytes","temperature","top_k","sampling_seed",
        "generated_byte_count","generated_valid_utf8_bytes","generated_invalid_utf8_bytes",
        "unique_byte_values","ascii_bytes","max_same_byte_run","max_scalar_repeat_run",
        "short_period_loop_fraction","finite","gate_policy","parity_prefix_count","ar_steps",
        "parity_gate","parity_gate_candidate","ar_gate","ar_gate_candidate",
        "generation_gate","generation_gate_candidate",
        "cpu_fallback","graph_execute_count",
        "generation_seconds","generate_seconds_per_byte",
        "android_thermal_status_before","android_thermal_status_after",
        "battery_level_before","battery_level_after",
        "battery_temperature_c_before","battery_temperature_c_after"
    )
    foreach ($key in $allowed) {
        if ($Map.Contains($key)) { $public[$key] = $Map[$key] }
    }
    $parityTail = "label|logits_(max_abs_error|mean_abs_error|rms_error|max_relative_error|cosine_similarity|cpu_min|cpu_max|cpu_rms|cpu_std|htp_min|htp_max|htp_rms|htp_std|centered_max_abs|centered_rms|logsoftmax_max_abs|logsoftmax_rms|cosine_centered)|delta_(mean|median|std)|scale_ratio|probability_(max_abs_error|mean_abs_error|l1_error|js_divergence|cosine_similarity)|top1_margin_cpu|top1_margin_htp|top2_margin_cpu|top2_margin_htp|topk_set_overlap|topk_set_size|topk_order_match|last_argmax_match|decision_ambiguous|row_degenerate|candidate_legacy|candidate_prob|candidate_shape|candidate_decision|candidate_full"
    foreach ($key in $Map.Keys) {
        if ($key -match "^parity_[0-9]+_($parityTail)$" -or
            $key -match "^ar_step_[0-9]+_(max_abs_logits_error|logits_(mean_abs_error|rms_error|max_relative_error|cosine_similarity)|margin_cpu|margin_htp|top1_margin_cpu|top1_margin_htp|top2_margin_cpu|top2_margin_htp|topk_set_overlap|topk_set_size|argmax_match|match|context_aligned|finite|decision_ambiguous|row_degenerate|candidate_decision|candidate_full)$") {
            $public[$key] = $Map[$key]
        }
    }
    return $public
}

$reports = Resolve-Under $ReportRoot $([IO.Path]::GetFullPath((Join-Path $repoRoot "build"))) "ReportRoot"
$faultRoot = Resolve-Under $FaultResultRoot $([IO.Path]::GetFullPath((Join-Path $repoRoot "build"))) "FaultResultRoot"
$cpuRoot = Resolve-Under $CpuGenRoot $([IO.Path]::GetFullPath((Join-Path $repoRoot "build"))) "CpuGenRoot"
$output = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
if (-not $output.StartsWith([IO.Path]::GetFullPath((Join-Path $repoRoot "docs\results")) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must resolve under docs/results"
}

$candidates = @(
    @{ candidate = "L"; name = "legacy"; description = "device gate: full-matrix raw logits max abs < 2e-2 AND prob max abs < 5e-3 per prefix; AR same" },
    @{ candidate = "P"; name = "prob"; description = "last-row: prob max abs < 5e-3, prob L1 < 2e-2, JS < 5e-3" },
    @{ candidate = "C"; name = "P+shape"; description = "P plus centered max abs < 2e-2, centered RMS < 5e-3, logsoftmax max abs < 1e-1, logsoftmax RMS < 5e-2, scale_ratio in [0.995,1.005], raw < 5e-1" },
    @{ candidate = "D"; name = "P+decision"; description = "P plus decision rule: argmax mismatch allowed only when CPU top1-top2 margin <= 2*raw; degenerate rows FAIL" },
    @{ candidate = "F"; name = "P+shape+decision (full)"; description = "C AND D (strictest candidate)" }
)

$selfRoot = Join-Path $repoRoot "build\export-selftest\nicopedia-htp-parity-policy"
if ($SelfTest) {
    $reports = Join-Path $selfRoot "reports"
    $faultRoot = Join-Path $selfRoot "faults"
    $cpuRoot = Join-Path $selfRoot "cpu"
    $output = Join-Path $selfRoot "output"
    [IO.Directory]::CreateDirectory($reports) | Out-Null
    [IO.Directory]::CreateDirectory($faultRoot) | Out-Null
    [IO.Directory]::CreateDirectory($cpuRoot) | Out-Null
    @"
NICOPEDIA_HTP_GENERATION
status=FAILED
test=nicopedia_htp_generation
model=L19
layers=19
heads=2
seed=1
checkpoint_step=320
generate_mode=greedy
temperature=1
top_k=256
sampling_seed=0
max_new_bytes=64
generated_byte_count=0
generated_valid_utf8_bytes=0
generated_invalid_utf8_bytes=0
unique_byte_values=0
ascii_bytes=0
max_same_byte_run=0
max_scalar_repeat_run=0
short_period_loop_fraction=0
finite=true
gate_policy=legacy
parity_prefix_count=20
ar_steps=8
parity_gate=false
parity_gate_candidate=false
ar_gate=true
ar_gate_candidate=true
generation_gate=false
generation_gate_candidate=false
cpu_fallback=false
graph_execute_count=28
parity_0_label=japanese_utf8_21
parity_0_logits_max_abs_error=0.0308008194
parity_0_logits_mean_abs_error=0.0001
parity_0_logits_rms_error=0.0001
parity_0_logits_max_relative_error=0.9
parity_0_logits_cosine_similarity=0.9999990000
parity_0_logits_cpu_std=3.56
parity_0_logits_htp_std=3.56
parity_0_logits_cpu_rms=3.5694
parity_0_logits_htp_rms=3.5674
parity_0_centered_max_abs=0.00002
parity_0_centered_rms=0.00001
parity_0_logsoftmax_max_abs=0.02
parity_0_logsoftmax_rms=0.01
parity_0_scale_ratio=1.00005
parity_0_probability_max_abs_error=0.0001
parity_0_probability_mean_abs_error=0.00001
parity_0_probability_l1_error=0.001
parity_0_probability_js_divergence=0.0002
parity_0_probability_cosine_similarity=0.9999999000
parity_0_top1_margin_cpu=0.6896
parity_0_top1_margin_htp=0.6914
parity_0_top2_margin_cpu=0.1
parity_0_top2_margin_htp=0.1
parity_0_topk_set_overlap=5
parity_0_topk_set_size=5
parity_0_topk_order_match=true
parity_0_last_argmax_match=true
parity_0_decision_ambiguous=false
parity_0_row_degenerate=false
parity_0_candidate_legacy=true
parity_0_candidate_prob=true
parity_0_candidate_shape=true
parity_0_candidate_decision=true
parity_0_candidate_full=true
parity_11_label=digits_punct_32
parity_11_logits_max_abs_error=0.01602697372
parity_11_logits_mean_abs_error=0.0001
parity_11_logits_rms_error=0.0001
parity_11_logits_max_relative_error=0.5
parity_11_logits_cosine_similarity=0.9999990000
parity_11_logits_cpu_std=3.56
parity_11_logits_htp_std=3.56
parity_11_logits_cpu_rms=3.5694
parity_11_logits_htp_rms=3.5674
parity_11_centered_max_abs=0.01426533403
parity_11_centered_rms=0.006189761834
parity_11_logsoftmax_max_abs=0.01252414196
parity_11_logsoftmax_rms=0.006430000107
parity_11_scale_ratio=0.9972035595
parity_11_probability_max_abs_error=0.0008292719722
parity_11_probability_mean_abs_error=0.00001
parity_11_probability_l1_error=0.006470928166
parity_11_probability_js_divergence=6.188643657e-06
parity_11_probability_cosine_similarity=0.9999999000
parity_11_top1_margin_cpu=0.2396305799
parity_11_top1_margin_htp=0.2275390625
parity_11_top2_margin_cpu=0.1
parity_11_top2_margin_htp=0.1
parity_11_topk_set_overlap=5
parity_11_topk_set_size=5
parity_11_topk_order_match=true
parity_11_last_argmax_match=true
parity_11_decision_ambiguous=false
parity_11_row_degenerate=false
parity_11_candidate_legacy=true
parity_11_candidate_prob=true
parity_11_candidate_shape=false
parity_11_candidate_decision=true
parity_11_candidate_full=false
ar_step_0_max_abs_logits_error=0.0111
ar_step_0_logits_mean_abs_error=0.0002
ar_step_0_logits_rms_error=0.0002
ar_step_0_logits_max_relative_error=0.5
ar_step_0_logits_cosine_similarity=0.9999995000
ar_step_0_margin_cpu=0.5
ar_step_0_margin_htp=0.5
ar_step_0_top1_margin_cpu=0.5
ar_step_0_top1_margin_htp=0.5
ar_step_0_top2_margin_cpu=0.1
ar_step_0_top2_margin_htp=0.1
ar_step_0_topk_set_overlap=5
ar_step_0_topk_set_size=5
ar_step_0_argmax_match=true
ar_step_0_match=true
ar_step_0_context_aligned=true
ar_step_0_finite=true
ar_step_0_decision_ambiguous=false
ar_step_0_candidate_decision=true
ar_step_0_candidate_full=true
"@ | Set-Content -LiteralPath (Join-Path $reports "seed1-l19-greedy-step320-max64-result.txt") -Encoding utf8
    @"
fault,legacy_policy,candidate_full_policy,decision_ambiguous,raw_max_abs,centered_max_abs,logsoftmax_max_abs,prob_max_abs,prob_l1,js_divergence,scale_ratio,margin_cpu
f1_common_offset_1e-3,1,1,0,1.000002027e-03,1.364105628e-08,1.351995316e-08,9.334562984e-11,1.432419391e-09,3.964011245e-18,0.999999984,0.300000
f2_common_offset_3e-2,0,1,0,3.000000119e-02,2.887827577e-08,2.873719573e-08,1.984098233e-10,1.323964644e-09,1.408267512e-18,0.999999983,0.300000
f3_common_offset_5e-1,0,0,0,5.000000119e-01,2.887827577e-08,2.873719573e-08,1.984098233e-10,1.323964644e-09,1.408267512e-18,0.999999983,0.300000
f4_single_logit_5e-2,0,0,0,1.000000000e-01,1.000000000e-01,1.000000000e-01,1.000000000e-01,1.000000000e-01,1.000000000e-01,1.000000000,0.300000
f5_swap_margin_4e-1,0,0,1,4.300000000e-01,4.283000000e-01,4.273000000e-01,2.724000000e-03,5.449000000e-03,1.453000000e-04,1.104900000,0.400000
f6_swap_margin_5e-3,1,1,1,5.000000000e-03,5.000000000e-03,5.000000000e-03,3.454000000e-05,6.909000000e-05,4.318000000e-08,1.000000000,0.005000
f6_swap_margin_5e-3,0,1,1,1.000000000e-03,1.000000000e-03,1.000000000e-03,1.000000000e-03,1.000000000e-03,1.000000000e-03,1.000000000,0.005000
f7_scale_1.02,1,0,0,2.000000000e-02,2.000000000e-02,2.000000000e-02,2.000000000e-02,2.000000000e-02,2.000000000e-02,1.020000000,0.300000
f8_scale_1.004,1,1,0,4.000000000e-03,4.000000000e-03,4.000000000e-03,4.000000000e-03,4.000000000e-03,4.000000000e-03,1.004000000,0.300000
f9_noise_sigma1e-3,1,1,0,3.098000000e-03,3.103000000e-03,3.092000000e-03,1.123000000e-05,8.341000000e-04,1.324000000e-07,1.001500000,0.300000
f10_noise_sigma1e-1,0,0,0,3.400000000e-01,3.424000000e-01,3.375000000e-01,1.750000000e-03,8.290000000e-02,1.384000000e-03,1.558700000,0.300000
f11_mass_redistribution,0,0,0,1.929000000e-01,1.207000000e-01,1.237000000e-01,4.848000000e-04,6.037000000e-02,6.056000000e-04,0.409300000,0.300000
f12_nan_element,0,0,0,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,1.000000000,0.300000
f13_inf_element,0,0,1,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,1.000000000,0.300000
f14_ninf_element,0,0,0,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,1.000000000,0.300000
f15_all_zero_row,0,0,0,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000e+00,0.000000000,0.000000
"@ | Set-Content -LiteralPath (Join-Path $faultRoot "synthetic-fault-results.csv") -Encoding utf8
    @"
NICOPEDIA_CPU_GENERATION
test=nicopedia_cpu_generation
status=SUCCESS
model=L19
layers=19
seed=1
checkpoint_step=1000
generate_mode=greedy
temperature=1
top_k=256
sampling_seed=0
max_new_bytes=64
prompt_byte_count=21
generated_byte_count=64
generated_valid_utf8_bytes=63
generated_invalid_utf8_bytes=1
unique_byte_values=5
ascii_bytes=0
max_same_byte_run=1
max_scalar_repeat_run=17
short_period_loop_fraction=1
finite=true
"@ | Set-Content -LiteralPath (Join-Path $cpuRoot "cpu-gen-step1000-greedy.txt") -Encoding utf8
}

if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Recurse -Force
}
[IO.Directory]::CreateDirectory($output) | Out-Null

$genFiles = @(Get-ChildItem -LiteralPath $reports -Filter "*result*.txt" -File -ErrorAction SilentlyContinue)
if (-not $genFiles) { throw "No device generation result files found under $reports" }

$publicReports = [System.Collections.Generic.List[object]]::new()
foreach ($file in $genFiles) {
    $public = Select-PublicFields (Read-KeyValueReport $file.FullName)
    # Only L19/seed-1 Nicopedia generation reports belong to this audit
    # (stale L6 reports may share the directory).
    if ($public["layers"] -ne "19" -or $public["seed"] -ne "1") { continue }
    if (-not $public.Contains("parity_gate_candidate")) { continue }
    $publicReports.Add([pscustomobject]@{ name = $file.Name; public = $public })
}
if ($publicReports.Count -eq 0) { throw "No L19 seed=1 Nicopedia generation reports found under $reports" }

# --- step-parity-summary.csv / scale-analysis.csv / decision-margin-summary.csv
#     / probability-parity.csv (per report x per prefix)
$stepRows = [System.Collections.Generic.List[object]]::new()
$scaleRows = [System.Collections.Generic.List[object]]::new()
$decisionRows = [System.Collections.Generic.List[object]]::new()
$probRows = [System.Collections.Generic.List[object]]::new()
foreach ($pair in $publicReports) {
    $name = $pair.name; $public = $pair.public
    $base = [ordered]@{ report = $name; checkpoint_step = $public["checkpoint_step"]; generate_mode = $public["generate_mode"] }
    for ($i = 0; ; ++$i) {
        $p = "parity_${i}_"
        if (-not $public.Contains("${p}label")) { break }
        $row = [ordered]@{}
        foreach ($k in @($base.Keys)) { $row[$k] = $base[$k] }
        $row["prefix_index"] = $i
        foreach ($key in @($public.Keys)) {
            if ($key -like "${p}*") { $row[$key.Substring($p.Length)] = $public[$key] }
        }
        $stepRows.Add($row)
        if ($row.Contains("scale_ratio")) {
            $scaleRows.Add([ordered]@{
                report = $name; checkpoint_step = $public["checkpoint_step"]; prefix_index = $i
                cpu_logits_std = $row["logits_cpu_std"]; htp_logits_std = $row["logits_htp_std"]
                cpu_logits_rms = $row["logits_cpu_rms"]; htp_logits_rms = $row["logits_htp_rms"]
                scale_ratio = $row["scale_ratio"]
            })
        }
        if ($row.Contains("decision_ambiguous")) {
            $d = [ordered]@{ report = $name; checkpoint_step = $public["checkpoint_step"]; prefix_index = $i }
            foreach ($k in @("top1_margin_cpu","top1_margin_htp","top2_margin_cpu","top2_margin_htp","topk_set_overlap","topk_set_size","topk_order_match","last_argmax_match","decision_ambiguous","row_degenerate","candidate_legacy","candidate_prob","candidate_shape","candidate_decision","candidate_full")) {
                if ($row.Contains($k)) { $d[$k] = $row[$k] }
            }
            $decisionRows.Add($d)
        }
        if ($row.Contains("probability_l1_error")) {
            $pr = [ordered]@{ report = $name; checkpoint_step = $public["checkpoint_step"]; prefix_index = $i }
            foreach ($k in @("probability_max_abs_error","probability_mean_abs_error","probability_l1_error","probability_js_divergence","probability_cosine_similarity")) {
                if ($row.Contains($k)) { $pr[$k] = $row[$k] }
            }
            $probRows.Add($pr)
        }
    }
}
Write-Csv (Join-Path $output "step-parity-summary.csv") $stepRows
Write-Csv (Join-Path $output "scale-analysis.csv") $scaleRows
Write-Csv (Join-Path $output "decision-margin-summary.csv") $decisionRows
Write-Csv (Join-Path $output "probability-parity.csv") $probRows

# --- policy-candidates.csv (fixed; never threshold-tuned)
Write-Csv (Join-Path $output "policy-candidates.csv") $candidates

# --- synthetic-fault-results.csv (copied from the host battery)
$faultRows = [System.Collections.Generic.List[object]]::new()
$faultFile = Get-ChildItem -LiteralPath $faultRoot -Filter "synthetic-fault-results.csv" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($faultFile) {
    $header = $null
    foreach ($line in [IO.File]::ReadAllLines($faultFile.FullName)) {
        if (-not $header) { $header = @($line -split ",") }
        else {
            $values = @($line -split ",")
            $row = [ordered]@{}
            for ($c = 0; $c -lt $header.Count; ++$c) { if ($c -lt $values.Count) { $row[$header[$c]] = $values[$c] } }
            $faultRows.Add($row)
        }
    }
    Write-Csv (Join-Path $output "synthetic-fault-results.csv") $faultRows
} else {
    Write-Host "WARN: synthetic-fault-results.csv missing; fault criteria will FAIL" -ForegroundColor Yellow
}

# --- policy-decision.csv (fixed adoption criteria, evaluated from evidence)
function Get-ReportPublic([string]$Name) {
    foreach ($pair in $publicReports) { if ($pair.name -eq $Name) { return $pair.public } }
    return $null
}
function Get-CriterionRow([string]$Id, [string]$Label, [bool]$Pass, [string]$Note) {
    [ordered]@{ criterion = $Id; label = $Label; result = $(if ($Pass) { "PASS" } else { "FAIL" }); note = $Note }
}
$critRows = [System.Collections.Generic.List[object]]::new()
$fStep320 = $publicReports | Where-Object { $_.public["checkpoint_step"] -eq "320" } | Select-Object -First 1
$fStep1000 = $publicReports | Where-Object { $_.public["checkpoint_step"] -eq "1000" -and $_.name -notlike "*-run2*" } | Select-Object -First 1
$fStep1000b = $publicReports | Where-Object { $_.public["checkpoint_step"] -eq "1000" -and $_.name -like "*-run2*" } | Select-Object -First 1
function Get-GateNote($Pair, [string]$Step) {
    if ($null -eq $Pair) { return "no ${Step} report" }
    "parity_gate_candidate=" + $Pair.public["parity_gate_candidate"] + " ar_gate_candidate=" + $Pair.public["ar_gate_candidate"]
}
$c1 = $null -ne $fStep320 -and $fStep320.public["parity_gate_candidate"] -eq "true" -and $fStep320.public["ar_gate_candidate"] -eq "true"
$c2 = $null -ne $fStep1000 -and $fStep1000.public["parity_gate_candidate"] -eq "true" -and $fStep1000.public["ar_gate_candidate"] -eq "true"
$c3 = $false
if ($fStep1000 -and $fStep1000b) {
    $gates = @("parity_gate","parity_gate_candidate","ar_gate","ar_gate_candidate","generation_gate")
    $c3 = $true
    foreach ($g in $gates) { if ($fStep1000.public[$g] -ne $fStep1000b.public[$g]) { $c3 = $false } }
}
$faultCount = $faultRows.Count
$c4 = $faultCount -ge 15
$nonFinite = @("f12_nan_element","f13_inf_element","f14_ninf_element","f15_all_zero_row")
$c5 = $faultCount -ge 15
foreach ($row in $faultRows) { if ($nonFinite -contains $row["fault"] -and $row["candidate_full_policy"] -ne "0") { $c5 = $false } }
$flipFails = @("f5_swap_margin_4e-1")
$c6 = $faultCount -ge 15
foreach ($row in $faultRows) { if ($flipFails -contains $row["fault"] -and $row["candidate_full_policy"] -ne "0") { $c6 = $false } }
$distortFails = @("f10_noise_sigma1e-1","f11_mass_redistribution")
$c7 = $faultCount -ge 15
foreach ($row in $faultRows) { if ($distortFails -contains $row["fault"] -and $row["candidate_full_policy"] -ne "0") { $c7 = $false } }
$c8 = $true
foreach ($pair in $publicReports) {
    if (-not $pair.public.Contains("parity_gate") -or -not $pair.public.Contains("parity_0_logits_max_abs_error")) { $c8 = $false }
}
$c9 = $ReviewerPass.IsPresent
$critRows.Add((Get-CriterionRow "1" "step320 audit passes candidate F" $c1 (Get-GateNote $fStep320 "step320")))
$critRows.Add((Get-CriterionRow "2" "step1000 audit passes candidate F" $c2 (Get-GateNote $fStep1000 "step1000")))
$critRows.Add((Get-CriterionRow "3" "determinism: repeated step1000 gates equal" $c3 "run2 vs run3 gate fields"))
$critRows.Add((Get-CriterionRow "4" "synthetic fault battery 15 rows" $c4 "rows=$faultCount"))
$critRows.Add((Get-CriterionRow "5" "NaN/Inf rows rejected by F" $c5 "f12-f15 must FAIL"))
$critRows.Add((Get-CriterionRow "6" "decisive-margin argmax flip rejected by F" $c6 "f5 must FAIL (f6 ambiguous passes)"))
$critRows.Add((Get-CriterionRow "7" "probability distortion rejected by F" $c7 "f10/f11 must FAIL"))
$critRows.Add((Get-CriterionRow "8" "no regression gate weakened (legacy fields still reported)" $c8 "parity_gate + raw keys present in all reports"))
$critRows.Add((Get-CriterionRow "9" "independent Reviewer PASS" $c9 "read-only review required"))
$adopt = $c1 -and $c2 -and $c3 -and $c4 -and $c5 -and $c6 -and $c7 -and $c8 -and $c9
$critRows.Add([ordered]@{ criterion = "RESULT"; label = "adoption decision"; result = $(if ($adopt) { "ADOPT_F" } else { "KEEP_LEGACY_BLOCKED" }); note = "adopt F only if criteria 1-9 all PASS; else keep L and generation stays BLOCKED" })
Write-Csv (Join-Path $output "policy-decision.csv") $critRows

# --- generation-comparison.csv (device reports + CPU private anchors, aggregates only)
$genRows = [System.Collections.Generic.List[object]]::new()
foreach ($pair in $publicReports) {
    $public = $pair.public
    $row = [ordered]@{ origin = "HTP-device"; report = $pair.name }
    foreach ($k in @("checkpoint_step","generate_mode","top_k","sampling_seed","max_new_bytes","generated_byte_count","generated_valid_utf8_bytes","generated_invalid_utf8_bytes","unique_byte_values","ascii_bytes","max_same_byte_run","max_scalar_repeat_run","short_period_loop_fraction","finite","generation_gate","parity_gate","ar_gate","status")) {
        if ($public.Contains($k)) { $row[$k] = $public[$k] }
    }
    $genRows.Add($row)
}
foreach ($file in @(Get-ChildItem -LiteralPath $cpuRoot -Filter "cpu-gen-step*.txt" -File -ErrorAction SilentlyContinue)) {
    $map = Read-KeyValueReport $file.FullName
    $row = [ordered]@{ origin = "CPU"; report = $file.Name }
    foreach ($k in @("checkpoint_step","generate_mode","top_k","sampling_seed","max_new_bytes","generated_byte_count","generated_valid_utf8_bytes","generated_invalid_utf8_bytes","unique_byte_values","ascii_bytes","max_same_byte_run","max_scalar_repeat_run","short_period_loop_fraction","finite","status")) {
        if ($map.Contains($k)) { $row[$k] = $map[$k] }
    }
    $genRows.Add($row)
}
Write-Csv (Join-Path $output "generation-comparison.csv") $genRows

$limitations = @(
    @{ limitation = "parity candidate metrics are computed on the last teacher-forced row per prefix; the device legacy gate still uses the full 32-row matrix (reported separately)" },
    @{ limitation = "device values are HTP qnn-context logits; CPU reference is the same-checkpoint CPU forward pass on the same device (private)" },
    @{ limitation = "determinism claim is limited to the two repeated step1000 audits (gate fields + per-prefix metrics identical)" },
    @{ limitation = "candidate adoption requires all policy-decision criteria PASS plus independent Reviewer PASS; otherwise active gate stays L and step1000 generation is BLOCKED" }
)
Write-Csv (Join-Path $output "limitations.csv") $limitations

$allowList = @(
    "policy-candidates.csv","step-parity-summary.csv","scale-analysis.csv",
    "decision-margin-summary.csv","probability-parity.csv","synthetic-fault-results.csv",
    "policy-decision.csv","generation-comparison.csv","limitations.csv","README.md"
)
$manifest = [ordered]@{
    title = "Nicopedia HTP parity gate audit (protocol qnn-nicopedia-htp-parity-policy)"
    generated_at = (Get-Date -Format o)
    canonical_commit = if ($SourceCommit) { $SourceCommit } else { "-" }
    files = @{}
}
foreach ($name in $allowList) {
    $path = Join-Path $output $name
    if (Test-Path -LiteralPath $path) {
        $manifest.files[$name] = @{ size_bytes = (Get-Item -LiteralPath $path).Length; sha256 = (Get-NormalizedSha256 $path) }
    }
}
Write-Json (Join-Path $output "manifest.json") $manifest

[IO.File]::WriteAllText((Join-Path $output "README.md"), @'
# Nicopedia parity gate policy audit (2026-08)

Protocol: docs/qnn-nicopedia-htp-parity-policy.md (fixed before measurement).

This bundle publishes aggregate parity metrics (per fixed prefix), the five
policy candidates, the synthetic fault battery, and the adoption decision.
Raw logits, generated bytes, prompts, checkpoint hashes, device identifiers
and content fingerprints are deliberately NOT published.

Candidates: L (legacy), P (probability), C (P + shape), D (P + decision),
F (P AND C AND D, strictest). Adoption rule: adopt F iff criteria 1-9 in
policy-decision.csv all PASS; otherwise keep L and step-1000 generation stays
BLOCKED (the allowed terminal outcome of the protocol).
'@ + "`n", [Text.UTF8Encoding]::new($false))

# Re-create manifest after README is written.
$manifest.files = @{}
foreach ($name in $allowList) {
    $path = Join-Path $output $name
    if (Test-Path -LiteralPath $path) {
        $manifest.files[$name] = [ordered]@{ size_bytes = (Get-Item -LiteralPath $path).Length; sha256 = (Get-NormalizedSha256 $path) }
    }
}
Write-Json (Join-Path $output "manifest.json") $manifest

if ($SelfTest) {
    $m = Get-Content -LiteralPath (Join-Path $output "manifest.json") -Raw | ConvertFrom-Json
    foreach ($f in $allowList) {
        if (-not $m.files.$f) { throw "self-test: $f not listed in manifest" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $output "policy-decision.csv"))) { throw "self-test: policy-decision.csv missing" }
    $decisionText = [IO.File]::ReadAllText((Join-Path $output "policy-decision.csv"))
    if ($decisionText -notmatch "KEEP_LEGACY_BLOCKED") { throw "self-test: fixture must evaluate to KEEP_LEGACY_BLOCKED" }
    Write-Host "qnn_nicopedia_htp_parity_policy_exporter=self-test PASS output=$output"
} else {
    Write-Host "qnn_nicopedia_htp_parity_policy_exporter=PASS output=$output"
}