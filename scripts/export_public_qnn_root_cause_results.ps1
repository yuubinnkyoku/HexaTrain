param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\reports\qnn-headless"),
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\results\qnn-htp-root-cause-2026-07")
)

$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$Repository = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$AllowedOutput = [IO.Path]::GetFullPath((Join-Path $Repository "docs\results\qnn-htp-root-cause-2026-07"))
$ResolvedOutput = [IO.Path]::GetFullPath($OutputDir)
if (-not $ResolvedOutput.Equals($AllowedOutput, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Public output is restricted to docs/results/qnn-htp-root-cause-2026-07."
}
$SupportingFiles = @("README.md", "graph-map-README.md", "node-map.csv", "tensor-map.csv")
$GeneratedFiles = @("summary.json", "experiment-results.csv", "first-change.csv")
$ExpectedPublicFiles = @($SupportingFiles + $GeneratedFiles)
if (-not (Test-Path -LiteralPath $ResolvedOutput -PathType Container)) {
    throw "Public output directory must already contain the reviewed graph-map support files."
}
$ExistingPublicEntries = @(Get-ChildItem -LiteralPath $ResolvedOutput -Force)
if (@($ExistingPublicEntries | Where-Object { -not $_.PSIsContainer -and
            ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) }).Count -ne 0 -or
        @($ExistingPublicEntries | Where-Object PSIsContainer).Count -ne 0) {
    throw "Public output must contain only regular files at its root; subdirectories and reparse points are forbidden."
}
$ExistingPublicFiles = @($ExistingPublicEntries | ForEach-Object Name)
if (Compare-Object ($ExpectedPublicFiles | Sort-Object) ($ExistingPublicFiles | Sort-Object)) {
    throw "Public output file set is not the exact seven-file allow-list."
}

$ExpectedHashes = [ordered]@{
    one_hot = "d85d7d14ab07879ab62b29dc0be5eef0c51d29db0a1e6050b8d7ccb080bd00f1"
    target = "f1c1a960169be212ee9f4b5856b9add5b9f2dd5ff68b77ce962292d8e1c724cb"
    current_parameters = "5674c9ecf8bcb785a4db27a73afb11e33aa22c23670301fbe7865692fa83b93b"
}

function Get-RunNames([string]$Prefix, [string[]]$Waves, [int[]]$Numbers) {
    $names = @()
    foreach ($wave in $Waves) {
        foreach ($number in $Numbers) { $names += ("{0}-{1}-{2:d2}" -f $Prefix, $wave, $number) }
    }
    return $names
}
function Read-PrivateReport([string]$Run) {
    $path = Join-Path $InputRoot "$Run\device-report.txt"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing required private aggregate report: $Run" }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            if ($values.ContainsKey($Matches[1]) -and $values[$Matches[1]] -ne $Matches[2]) { throw "Conflicting private report field: $Run/$($Matches[1])" }
            $values[$Matches[1]] = $Matches[2]
        }
    }
    return $values
}
function Require-Text($Report, [string]$Name, [string]$Pattern) {
    if (-not $Report.ContainsKey($Name) -or $Report[$Name] -notmatch $Pattern) {
        throw "Invalid or missing allow-listed field: $Name"
    }
    return $Report[$Name]
}
function Require-Int($Report, [string]$Name, [int]$Minimum, [int]$Maximum) {
    $text = Require-Text $Report $Name '^-?[0-9]+$'
    $value = [int]$text
    if ($value -lt $Minimum -or $value -gt $Maximum) { throw "Out-of-range field: $Name" }
    return $value
}
function Require-NumberOrInfinity($Report, [string]$Name) {
    $text = Require-Text $Report $Name '^(?:-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf|nan)$'
    return $text
}
function Require-FiniteNumber($Report, [string]$Name) {
    $text = Require-Text $Report $Name '^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$'
    $value = [double]::Parse($text, $Invariant)
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { throw "Non-finite field: $Name" }
    return $value
}
function Require-Rle($Report, [string]$Name, [int]$Repeats) {
    if (-not $Report.ContainsKey($Name)) { return "NOT_REPORTED_LEGACY_REPORT" }
    $rle = Require-Text $Report $Name '^[0-9]+:[1-9][0-9]*-[1-9][0-9]*(?:,[0-9]+:[1-9][0-9]*-[1-9][0-9]*)*$'
    $covered = 0
    foreach ($entry in $rle -split ',') {
        $range = ($entry -split ':')[1] -split '-'
        $start = [int]$range[0]; $end = [int]$range[1]
        if ($start -ne ($covered + 1) -or $end -lt $start) { throw "Invalid RLE coverage: $Name" }
        $covered = $end
    }
    if ($covered -ne $Repeats) { throw "Invalid RLE terminal repeat: $Name" }
    return $rle
}
function Get-TensorObservation($Report, [string]$Prefix, [string]$Tensor, [int]$Repeats) {
    $base = "$Prefix$Tensor"
    return [pscustomobject][ordered]@{
        tensor = $Tensor
        unique_canonical_hashes = Require-Int $Report "${base}_unique_canonical_hashes" 1 $Repeats
        canonical_hash_rle = Require-Rle $Report "${base}_canonical_hash_rle" $Repeats
        first_different_run = Require-Int $Report "${base}_first_different_run" -1 $Repeats
        max_abs_difference = Require-NumberOrInfinity $Report "${base}_repeat_max_abs_difference"
        nonfinite_elements = Require-Int $Report "${base}_nonfinite_elements" 0 10000000
        app_read_poison_residual_elements = Require-Int $Report "${base}_app_read_poison_residual_elements" 0 0
    }
}
function Assert-Common($Report, [string]$Run, [bool]$AllowNumericalFailure) {
    foreach ($pair in @(
        @("snapshot_E_one_hot_raw_hash", $ExpectedHashes.one_hot), @("snapshot_E_one_hot_canonical_hash", $ExpectedHashes.one_hot),
        @("snapshot_E_target_raw_hash", $ExpectedHashes.target), @("snapshot_E_target_canonical_hash", $ExpectedHashes.target),
        @("snapshot_E_current_parameter_raw_hash", $ExpectedHashes.current_parameters), @("snapshot_E_current_parameter_canonical_hash", $ExpectedHashes.current_parameters)
    )) { if ($Report[$pair[0]] -ne $pair[1]) { throw "Fixed-state hash mismatch in ${Run}: $($pair[0])" } }
    $status = Require-Text $Report status '^(SUCCESS|FAILED)$'
    if ($status -eq "FAILED" -and -not $AllowNumericalFailure) { throw "Unexpected failed report: $Run" }
    if ($Report.activity_create_count -ne "0" -or $Report.activity_resume_count -ne "0" -or
        $Report.phonelm_became_top_activity_count -ne "0" -or $Report.focus_takeover_count -ne "0" -or
        $Report.single_flight_result -ne "ALREADY_RUNNING" -or $Report.backend_requested -ne "HTP" -or
        $Report.cpu_fallback -ne "false") { throw "Headless/backend invariant failed: $Run" }
    return $status
}
# Continued fraction implementation of regularized incomplete beta.  The binary
# search gives a deterministic Clopper-Pearson 95% interval without external tools.
function Get-BetaFraction([double]$A, [double]$B, [double]$X) {
    $qab = $A + $B; $qap = $A + 1.0; $qam = $A - 1.0; $c = 1.0; $d = 1.0 - $qab * $X / $qap
    if ([Math]::Abs($d) -lt 3e-30) { $d = 3e-30 }; $d = 1.0 / $d; $h = $d
    for ($m = 1; $m -le 200; $m++) {
        $m2 = 2.0 * $m; $aa = $m * ($B - $m) * $X / (($qam + $m2) * ($A + $m2))
        $d = 1.0 + $aa * $d; if ([Math]::Abs($d) -lt 3e-30) { $d = 3e-30 }; $c = 1.0 + $aa / $c
        if ([Math]::Abs($c) -lt 3e-30) { $c = 3e-30 }; $d = 1.0 / $d; $h *= $d * $c
        $aa = -($A + $m) * ($qab + $m) * $X / (($A + $m2) * ($qap + $m2))
        $d = 1.0 + $aa * $d; if ([Math]::Abs($d) -lt 3e-30) { $d = 3e-30 }; $c = 1.0 + $aa / $c
        if ([Math]::Abs($c) -lt 3e-30) { $c = 3e-30 }; $d = 1.0 / $d; $delta = $d * $c; $h *= $delta
        if ([Math]::Abs($delta - 1.0) -lt 3e-14) { break }
    }
    return $h
}
function Get-LogGamma([double]$Z) {
    # Lanczos approximation, valid for the positive arguments used by beta(a,b).
    $coefficients = @(676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-06, 1.5056327351493116e-07)
    if ($Z -lt 0.5) { return [Math]::Log([Math]::PI) - [Math]::Log([Math]::Sin([Math]::PI * $Z)) - (Get-LogGamma (1.0 - $Z)) }
    $z1 = $Z - 1.0; $x = 0.99999999999980993
    for ($i = 0; $i -lt $coefficients.Count; $i++) { $x += $coefficients[$i] / ($z1 + $i + 1.0) }
    $t = $z1 + $coefficients.Count - 0.5
    return 0.91893853320467274 + ($z1 + 0.5) * [Math]::Log($t) - $t + [Math]::Log($x)
}
function Get-RegularizedBeta([double]$X, [double]$A, [double]$B) {
    if ($X -le 0.0) { return 0.0 }; if ($X -ge 1.0) { return 1.0 }
    $logBt = [Math]::Log($X) * $A + [Math]::Log(1.0 - $X) * $B + (Get-LogGamma ($A + $B)) - (Get-LogGamma $A) - (Get-LogGamma $B)
    $bt = [Math]::Exp($logBt)
    if ($X -lt (($A + 1.0) / ($A + $B + 2.0))) { return $bt * (Get-BetaFraction $A $B $X) / $A }
    return 1.0 - $bt * (Get-BetaFraction $B $A (1.0 - $X)) / $B
}
function Get-BetaInverse([double]$P, [double]$A, [double]$B) {
    $lo = 0.0; $hi = 1.0
    for ($i = 0; $i -lt 80; $i++) { $mid = ($lo + $hi) / 2.0; if ((Get-RegularizedBeta $mid $A $B) -lt $P) { $lo = $mid } else { $hi = $mid } }
    return ($lo + $hi) / 2.0
}
function Get-ClopperPearson([int]$Events, [int]$Total) {
    if ($Events -eq 0) { $lower = 0.0 } else { $lower = Get-BetaInverse 0.025 $Events ($Total - $Events + 1) }
    if ($Events -eq $Total) { $upper = 1.0 } else { $upper = Get-BetaInverse 0.975 ($Events + 1) ($Total - $Events) }
    return [ordered]@{ events = $Events; total = $Total; rate = [Math]::Round($Events / [double]$Total, 9); clopper_pearson_95_lower = [Math]::Round($lower, 9); clopper_pearson_95_upper = [Math]::Round($upper, 9) }
}

$Specs = @()
foreach ($n in 1..6) { $Specs += [pscustomobject]@{ run = "rootcause-baseline-full-{0:d2}" -f $n; condition = "pre_fix_baseline_full"; kind = "variant"; prefix = "variant_full_"; observation = "embedding_input_gradient"; allowFailure = $true } }
foreach ($run in Get-RunNames "rootcause-original-dinput" @("w1", "w2") (1..12)) { $Specs += [pscustomobject]@{ run = $run; condition = "pre_fix_original_dinput"; kind = "variant"; prefix = "variant_stop_after_dinput_"; observation = "embedding_input_gradient"; allowFailure = $false } }
foreach ($run in (@(Get-RunNames "rootcause-tap-backward" @("w1") (2..13)) + @(Get-RunNames "rootcause-tap-backward" @("w2") (1..12)))) { $Specs += [pscustomobject]@{ run = $run; condition = "pre_fix_tap_backward"; kind = "tap"; prefix = "tap_backward_regions_"; observation = "DSCORES"; allowFailure = $true } }
foreach ($run in Get-RunNames "rootcause-tap-dscores" @("w1") (1..12)) { $Specs += [pscustomobject]@{ run = $run; condition = "pre_fix_tap_dscores_only"; kind = "tap"; prefix = "tap_dscores_only_"; observation = "DSCORES"; allowFailure = $true } }
foreach ($run in (Get-RunNames "rootcause-tap-dprob-dscores" @("w1", "w2") (1..12))) { $Specs += [pscustomobject]@{ run = $run; condition = "pre_fix_tap_dprob_dscores"; kind = "tap"; prefix = "tap_dprob_dscores_"; observation = "DSCORES"; allowFailure = $true } }
foreach ($run in Get-RunNames "rootcause-fixed-dinput" @("w1") (1..50)) { $Specs += [pscustomobject]@{ run = $run; condition = "post_fix_dinput"; kind = "variant"; prefix = "variant_stop_after_dinput_"; observation = "embedding_input_gradient"; allowFailure = $false } }
foreach ($run in Get-RunNames "rootcause-fixed-full" @("w1") (1..10)) { $Specs += [pscustomobject]@{ run = $run; condition = "post_fix_full"; kind = "variant"; prefix = "variant_full_"; observation = "embedding_input_gradient"; allowFailure = $false } }
$Specs += [pscustomobject]@{ run = "rootcause-fixed-tap-dprob-dscores-01"; condition = "post_fix_tap_dprob_dscores"; kind = "tap"; prefix = "tap_dprob_dscores_"; observation = "DSCORES"; allowFailure = $false }

$rows = @(); $firstChange = @()
foreach ($spec in $Specs) {
    $report = Read-PrivateReport $spec.run
    $status = Assert-Common $report $spec.run $spec.allowFailure
    $executionPrefix = if ($spec.kind -eq "tap") { "" } else { $spec.prefix }
    $repeats = Require-Int $report ($executionPrefix + "qnn_execute_attempts") 100 100
    $successes = Require-Int $report ($executionPrefix + "qnn_execute_successes") 100 100
    if ($report[($executionPrefix + "qnn_execute_return_code")] -ne "0" -or $successes -ne $repeats -or
        $report[($executionPrefix + "app_write_hashes_unchanged")] -ne "true" -or
        (Require-Int $report ($executionPrefix + "app_read_poison_residual_elements") 0 0) -ne 0) { throw "Execution/buffer invariant failed: $($spec.run)" }
    $obs = Get-TensorObservation $report $spec.prefix $spec.observation $repeats
    $nonfinite = Require-Int $report ($executionPrefix + "nonfinite_elements") 0 10000000
    $allFinite = Require-Text $report ($executionPrefix + "all_outputs_finite") '^(true|false)$'
    if (($status -eq "SUCCESS" -and ($allFinite -ne "true" -or $nonfinite -ne 0)) -or ($status -eq "FAILED" -and ($allFinite -ne "false" -or $nonfinite -le 0))) { throw "Numerical status mismatch: $($spec.run)" }
    $nodeCount = if ($report.ContainsKey("source_node_count")) { Require-Int $report source_node_count 1 10000 } else { $null }
    $tensorCount = if ($report.ContainsKey("source_tensor_count")) { Require-Int $report source_tensor_count 1 10000 } else { $null }
    $outputCount = if ($report.ContainsKey("actual_output_count")) { Require-Int $report actual_output_count 1 10000 } else { $null }
    if ($spec.kind -eq "tap") {
        $expectedOutputs = @{ tap_backward_regions_ = 22; tap_dscores_only_ = 19; tap_dprob_dscores_ = 20 }[$spec.prefix]
        if ($nodeCount -ne 98 -or $tensorCount -ne 153 -or $outputCount -ne $expectedOutputs) {
            throw "Unexpected source graph counts: $($spec.run)"
        }
    }
    $rows += [pscustomobject][ordered]@{ condition = $spec.condition; run = $spec.run; status = $status; repeats = $repeats; qnn_execute_successes = $successes; qnn_execute_return_code = 0; app_write_hashes_unchanged = $true; app_read_poison_residual_elements = 0; all_outputs_finite = ($allFinite -eq "true"); nonfinite_elements = $nonfinite; source_node_count = $nodeCount; source_tensor_count = $tensorCount; actual_output_count = $outputCount; observed_tensor = $obs.tensor; unique_canonical_hashes = $obs.unique_canonical_hashes; canonical_hash_rle = $obs.canonical_hash_rle; first_different_run = $obs.first_different_run; max_abs_difference = $obs.max_abs_difference; observed_tensor_nonfinite_elements = $obs.nonfinite_elements }
    if ($spec.kind -eq "tap") {
        $tensors = @("DSCORES")
        if ($spec.prefix -eq "tap_dprob_dscores_") { $tensors = @("DPROBABILITIES", "DSCORES") }
        foreach ($tensor in $tensors) {
            $tap = Get-TensorObservation $report $spec.prefix $tensor $repeats
            $firstChange += [pscustomobject][ordered]@{ condition = $spec.condition; run = $spec.run; tensor = $tap.tensor; repeats = $repeats; unique_canonical_hashes = $tap.unique_canonical_hashes; canonical_hash_rle = $tap.canonical_hash_rle; first_different_run = $tap.first_different_run; max_abs_difference = $tap.max_abs_difference; nonfinite_elements = $tap.nonfinite_elements; app_read_poison_residual_elements = $tap.app_read_poison_residual_elements }
        }
    }
}

$microRuns = @("rootcause-fixed-softmax-micro-01", "rootcause-fixed-attention-micro-01")
$micro = @()
foreach ($run in $microRuns) {
    $report = Read-PrivateReport $run; $status = Require-Text $report status '^SUCCESS$'
    if ($report.graph_create_result -ne "0" -or $report.graph_finalize_result -ne "0" -or $report.graph_execute_result -ne "0" -or
        (Require-Int $report htp_graph_execute_count 1 100) -lt 1 -or $report.nan_inf -ne "false") { throw "Micrograph QNN/numerical invariant failed: $run" }
    $null = Require-FiniteNumber $report htp_vs_cpu_max_abs_error
    $null = Require-FiniteNumber $report htp_vs_cpu_mean_abs_error
    $null = Require-FiniteNumber $report cpu_analytic_vs_numeric_max_abs_error
    if ($report.cpu_fallback -ne "false" -or $report.activity_create_count -ne "0" -or $report.focus_takeover_count -ne "0") { throw "Micrograph headless invariant failed: $run" }
    $micro += [pscustomobject][ordered]@{ run = $run; status = $status; graph_execute_result = if ($report.ContainsKey("graph_execute_result")) { [int]$report.graph_execute_result } else { [int]$report.qnn_execute_return_code }; cpu_fallback = $false; nan_inf = if ($report.ContainsKey("nan_inf")) { $report.nan_inf } else { "false" } }
}

function Get-Rate($Condition) { $items = @($rows | Where-Object condition -eq $Condition); return Get-ClopperPearson @($items | Where-Object { $_.unique_canonical_hashes -gt 1 }).Count $items.Count }
$dprob = @($firstChange | Where-Object { $_.condition -eq "pre_fix_tap_dprob_dscores" -and $_.tensor -eq "DPROBABILITIES" })
$dscores = @($firstChange | Where-Object { $_.condition -eq "pre_fix_tap_dprob_dscores" -and $_.tensor -eq "DSCORES" })
$summary = [ordered]@{
    schema_version = 1
    study = "PhoneLM QNN HTP fixed-state root-cause investigation"
    classification = "APP_IMPLEMENTATION_DEFECT_FIXED_VARIABILITY_NOT_REOBSERVED"
    confirmed = [ordered]@{
        defect = "SOFTMAX_DOT_REDUCE_OUTPUT_SHAPE_CONTRACT_MISMATCH_8x8_INSTEAD_OF_8x1"
        reduce_contract = "axis=1;keep_dims=true;output_shape=8x1"
        post_fix_dinput_variability = "0/50"
        post_fix_full_variability = "0/10"
    }
    inference = "The SOFTMAX_DOT shape mismatch caused the earlier fresh-instance-associated variability and nonfinite outputs."
    not_determined = "The specific HTP lowering, scratch allocation, or internal execution mechanism affected by the mismatch."
    qairt_build_id = "2.48.40.260702151143"
    qnn_api_version = "2.37.0"
    fixed_state_hashes = $ExpectedHashes
    process_reports = $rows.Count
    executes = [int](($rows.repeats | Measure-Object -Sum).Sum)
    pre_fix = [ordered]@{ baseline_full = (Get-Rate "pre_fix_baseline_full"); original_dinput = (Get-Rate "pre_fix_original_dinput"); tap_backward = (Get-Rate "pre_fix_tap_backward"); tap_dscores_only = (Get-Rate "pre_fix_tap_dscores_only"); tap_dprob_dscores = (Get-Rate "pre_fix_tap_dprob_dscores") }
    post_fix = [ordered]@{ dinput = (Get-Rate "post_fix_dinput"); full = (Get-Rate "post_fix_full"); tap_dprob_dscores = (Get-Rate "post_fix_tap_dprob_dscores") }
    first_change = [ordered]@{
        preceding_stable_tensor = "DPROBABILITIES"
        interval = "DPROBABILITIES->Multiply(tt_smp)->SOFTMAX_PRODUCT->ReduceSum(tt_smd)->SOFTMAX_DOT->Subtract(tt_smc)->SOFTMAX_CENTERED->Multiply(tt_ds)->DSCORES"
        first_varying_tensor = "DSCORES"
        dprob_varying_processes = @($dprob | Where-Object { $_.unique_canonical_hashes -gt 1 }).Count
        dprob_tested_processes = $dprob.Count
        dscores_varying_processes = @($dscores | Where-Object { $_.unique_canonical_hashes -gt 1 }).Count
        dscores_tested_processes = $dscores.Count
        tap_semantics = "APP_READ_PROMOTION_ONLY"
        tap_limitation = "Source node set and creation order are preserved; output exposure changes and backend lowering equivalence is not asserted."
    }
    qnn_execute_return_code = 0
    app_read_poison_residual_elements = 0
    app_write_hashes_unchanged = $true
    activity_launches = 0
    focus_takeovers = 0
    micrographs = $micro
    publication = [ordered]@{ raw_tensors = $false; private_paths = $false; endpoints = $false; process_ids = $false; timestamps = $false; apks = $false; qnn_binaries = $false; logcat = $false; output_hash_values = $false; fixed_input_state_hashes_present = $true }
}

$temporaryOutput = Join-Path ([IO.Path]::GetTempPath()) ("phonelm-qnn-root-cause-" + [Guid]::NewGuid().ToString("N"))
[IO.Directory]::CreateDirectory($temporaryOutput) | Out-Null
try {
    $summaryPath = Join-Path $temporaryOutput "summary.json"; $resultsPath = Join-Path $temporaryOutput "experiment-results.csv"; $firstPath = Join-Path $temporaryOutput "first-change.csv"
    $summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    $rows | Export-Csv -LiteralPath $resultsPath -NoTypeInformation -Encoding utf8
    $firstChange | Export-Csv -LiteralPath $firstPath -NoTypeInformation -Encoding utf8
    foreach ($name in $SupportingFiles) {
        Copy-Item -LiteralPath (Join-Path $ResolvedOutput $name) -Destination (Join-Path $temporaryOutput $name)
    }
    $stagedNames = @(Get-ChildItem -LiteralPath $temporaryOutput -File | ForEach-Object Name)
    if (Compare-Object ($ExpectedPublicFiles | Sort-Object) ($stagedNames | Sort-Object)) {
        throw "Staged public output file set is not the exact seven-file allow-list."
    }
    $deny = '(?im)([A-Z]:\\|/data/(?:user|data|local)/|\\\\[A-Za-z0-9._-]+\\|\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b|\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b|@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\.(?:so|apk|aab|jks|keystore|elf|bin)\b|BEGIN [A-Z ]*PRIVATE KEY|(?:password|passwd|secret[_-]?key|access[_-]?token|authorization:|bearer\s+[A-Za-z0-9._-]+|sk-[A-Za-z0-9_-]{16,})|^\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\s+\d+\s+\d+\s+[VDIWEF]\s)'
    foreach ($path in Get-ChildItem -LiteralPath $temporaryOutput -File) {
        $text = Get-Content -Raw -LiteralPath $path
        if ($text -match $deny) { throw "Forbidden public field or path detected in $([IO.Path]::GetFileName($path))." }
    }
    foreach ($name in $GeneratedFiles) {
        Copy-Item -LiteralPath (Join-Path $temporaryOutput $name) -Destination (Join-Path $ResolvedOutput $name) -Force
    }
} finally {
    if (Test-Path -LiteralPath $temporaryOutput) {
        $resolvedTemporaryOutput = [IO.Path]::GetFullPath($temporaryOutput)
        $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $resolvedTemporaryOutput.StartsWith($temporaryRoot, [StringComparison]::OrdinalIgnoreCase) -or
                -not (Split-Path -Leaf $resolvedTemporaryOutput).StartsWith("phonelm-qnn-root-cause-")) {
            throw "Refusing unsafe exporter staging cleanup target."
        }
        Remove-Item -LiteralPath $resolvedTemporaryOutput -Recurse -Force
    }
}
Write-Output "public_export=SUCCESS"
Write-Output "validated_public_files=$($ExpectedPublicFiles -join ',')"
