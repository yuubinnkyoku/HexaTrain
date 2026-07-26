param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\reports\qnn-headless"),
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\results\qnn-htp-graph-bisection-2026-07")
)

$ErrorActionPreference = "Stop"
$repository = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$allowedOutput = [IO.Path]::GetFullPath((Join-Path $repository "docs\results\qnn-htp-graph-bisection-2026-07"))
$resolvedOutput = [IO.Path]::GetFullPath($OutputDir)
if (-not $resolvedOutput.Equals($allowedOutput, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Public output is restricted to the documented result directory."
}
[IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null

function Read-Report([string]$RunName, [string]$ExpectedStatus = "SUCCESS") {
    $path = Join-Path $InputRoot "$RunName\device-report.txt"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required private aggregate report: $RunName"
    }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    if ($values.status -ne $ExpectedStatus) {
        throw "Private aggregate report status mismatch: $RunName"
    }
    return $values
}
function Require-Value([hashtable]$Report, [string]$Name, [string]$Pattern) {
    $value = $Report[$Name]
    if ($null -eq $value -or $value -notmatch $Pattern) {
        throw "Invalid or missing allow-listed field: $Name"
    }
    return $value
}
function Require-Integer([hashtable]$Report, [string]$Name, [int]$Minimum,
                         [int]$Maximum) {
    $text = Require-Value $Report $Name '^-?[0-9]+$'
    $value = [int]$text
    if ($value -lt $Minimum -or $value -gt $Maximum) {
        throw "Allow-listed integer is out of range: $Name"
    }
    return $value
}
function Require-FiniteNumber([hashtable]$Report, [string]$Name) {
    $text = Require-Value $Report $Name '^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$'
    $value = [double]::Parse($text, [Globalization.CultureInfo]::InvariantCulture)
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) {
        throw "Allow-listed number is not finite and nonnegative: $Name"
    }
    return $value
}
function Require-HashFrequencies([hashtable]$Report, [string]$Name,
                                 [int]$ExpectedUnique, [int]$ExpectedTotal) {
    $text = Require-Value $Report $Name '^[0-9a-f]{64}:[1-9][0-9]*(?:,[0-9a-f]{64}:[1-9][0-9]*)*$'
    $entries = @($text -split ',')
    $hashes = @{}
    $total = 0
    foreach ($entry in $entries) {
        $parts = $entry -split ':'
        if ($hashes.ContainsKey($parts[0])) { throw "Duplicate canonical hash in $Name" }
        $hashes[$parts[0]] = $true
        $total += [int]$parts[1]
    }
    if ($hashes.Count -ne $ExpectedUnique -or $total -ne $ExpectedTotal) {
        throw "Canonical hash frequency count mismatch: $Name"
    }
    return $text
}

$expectedHashes = [ordered]@{
    one_hot = "d85d7d14ab07879ab62b29dc0be5eef0c51d29db0a1e6050b8d7ccb080bd00f1"
    target = "f1c1a960169be212ee9f4b5856b9add5b9f2dd5ff68b77ce962292d8e1c724cb"
    current_parameters = "5674c9ecf8bcb785a4db27a73afb11e33aa22c23670301fbe7865692fa83b93b"
}
$successfulProcessNumbers = @(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16)
$specs = @($successfulProcessNumbers | ForEach-Object {
    [pscustomobject]@{
        arm = "full_first_sequential"
        process = $_
        run = ("final-audit-sequential-{0:d2}" -f $_)
        variants = @("full", "stop_after_dinput", "stop_after_dembedding")
    }
})
$variantMetadata = @{
    full = [ordered]@{ boundary = "lm_output_projection_next"; audited = 31 }
    stop_after_dinput = [ordered]@{ boundary = "lm_dinput"; audited = 18 }
    stop_after_dembedding = [ordered]@{ boundary = "lm_dembedding"; audited = 19 }
}

$rows = foreach ($spec in $specs) {
    $report = Read-Report $spec.run
    foreach ($pair in @(
        @("snapshot_E_one_hot_canonical_hash", $expectedHashes.one_hot),
        @("snapshot_E_one_hot_raw_hash", $expectedHashes.one_hot),
        @("snapshot_E_target_canonical_hash", $expectedHashes.target),
        @("snapshot_E_target_raw_hash", $expectedHashes.target),
        @("snapshot_E_current_parameter_canonical_hash", $expectedHashes.current_parameters),
        @("snapshot_E_current_parameter_raw_hash", $expectedHashes.current_parameters)
    )) {
        if ($report[$pair[0]] -ne $pair[1]) {
            throw "Fixed-state hash mismatch in $($spec.run): $($pair[0])"
        }
    }
    foreach ($variant in $spec.variants) {
        $prefix = "variant_$variant"
        $metadata = $variantMetadata[$variant]
        $attempts = Require-Integer $report "${prefix}_qnn_execute_attempts" 100 100
        $successes = Require-Integer $report "${prefix}_qnn_execute_successes" 100 100
        $poison = Require-Integer $report "${prefix}_app_read_poison_residual_elements" 0 0
        $nonfinite = Require-Integer $report "${prefix}_nonfinite_elements" 0 0
        $audited = Require-Integer $report "${prefix}_app_read_tensors_audited" $metadata.audited $metadata.audited
        $unique = Require-Integer $report "${prefix}_embedding_input_gradient_unique_canonical_hashes" 1 100
        $frequencies = Require-HashFrequencies $report `
            "${prefix}_embedding_input_gradient_canonical_hash_frequencies" $unique 100
        $maxDifference = Require-FiniteNumber $report `
            "${prefix}_embedding_input_gradient_repeat_max_abs_difference"
        $firstRun = Require-Integer $report `
            "${prefix}_embedding_input_gradient_first_different_run" -1 100
        $firstIndex = Require-Integer $report `
            "${prefix}_embedding_input_gradient_first_repeat_different_index" -1 127
        $boundary = Require-Value $report "${prefix}_graph_boundary" `
            "^$([regex]::Escape($metadata.boundary))$"
        if ($attempts -ne 100 -or $successes -ne 100 -or
            $report["${prefix}_qnn_execute_return_code"] -ne "0" -or
            $report["${prefix}_app_write_hashes_unchanged"] -ne "true" -or
            $report["${prefix}_all_outputs_finite"] -ne "true") {
            throw "Execution invariant failed in $($spec.run): $variant"
        }
        if (($unique -eq 1 -and
             ($firstRun -ne -1 -or $firstIndex -ne -1 -or $maxDifference -ne 0)) -or
            ($unique -gt 1 -and
             ($firstRun -lt 2 -or $firstIndex -lt 0 -or $maxDifference -le 0))) {
            throw "Variability fields are internally inconsistent in $($spec.run): $variant"
        }
        [pscustomobject][ordered]@{
            arm = $spec.arm
            process = $spec.process
            variant = $variant
            graph_boundary = $metadata.boundary
            repeats = $attempts
            qnn_execute_successes = $successes
            qnn_execute_return_code = 0
            app_write_hashes_unchanged = $true
            app_read_tensors_audited = $audited
            app_read_poison_residual_elements = $poison
            nonfinite_elements = $nonfinite
            embedding_input_gradient_unique_canonical_hashes =
                $unique
            embedding_input_gradient_canonical_hash_frequencies =
                $frequencies
            embedding_input_gradient_repeat_max_abs_difference =
                $maxDifference
            embedding_input_gradient_first_different_run =
                $firstRun
            embedding_input_gradient_first_different_index =
                $firstIndex
        }
    }
}

$sequentialFull = @($rows | Where-Object { $_.variant -eq "full" })
$sequentialDinput = @($rows | Where-Object { $_.variant -eq "stop_after_dinput" })
$sequentialDembedding = @($rows | Where-Object { $_.variant -eq "stop_after_dembedding" })
$varying = { param($items) @($items | Where-Object { $_.embedding_input_gradient_unique_canonical_hashes -gt 1 }).Count }
$toggleProcesses = @($sequentialFull | Where-Object {
    $process = $_.process
    $control = @($sequentialDembedding | Where-Object process -eq $process)
    $_.embedding_input_gradient_unique_canonical_hashes -gt 1 -and
    $control.Count -eq 1 -and
    $control[0].embedding_input_gradient_unique_canonical_hashes -eq 1
})
$failed = Read-Report "final-audit-sequential-13" "FAILED"
foreach ($pair in @(
    @("snapshot_E_one_hot_canonical_hash", $expectedHashes.one_hot),
    @("snapshot_E_target_canonical_hash", $expectedHashes.target),
    @("snapshot_E_current_parameter_canonical_hash", $expectedHashes.current_parameters)
)) {
    if ($failed[$pair[0]] -ne $pair[1]) {
        throw "Fixed-state hash mismatch in numerical-audit failure: $($pair[0])"
    }
}
$failedUnique = Require-Integer $failed `
    "variant_full_embedding_input_gradient_unique_canonical_hashes" 2 100
$failedFrequencies = Require-HashFrequencies $failed `
    "variant_full_embedding_input_gradient_canonical_hash_frequencies" $failedUnique 100
$failedNonfinite = Require-Integer $failed "variant_full_nonfinite_elements" 1 1000000
$failedPoison = Require-Integer $failed `
    "variant_full_app_read_poison_residual_elements" 0 0
$failedAudited = Require-Integer $failed "variant_full_app_read_tensors_audited" 31 31
if ($failed.variant_full_qnn_execute_return_code -ne "0" -or
    $failed.variant_full_qnn_execute_attempts -ne "100" -or
    $failed.variant_full_qnn_execute_successes -ne "100" -or
    $failed.variant_full_all_outputs_finite -ne "false" -or
    $failed.error -ne "nonfinite graph variant output") {
    throw "Numerical-audit failure fields are inconsistent."
}
$failureRows = @(
    [pscustomobject][ordered]@{
        arm = "full_first_sequential"
        process = 13
        variant = "full"
        status = "FAILED_NUMERICAL_AUDIT"
        repeats = 100
        qnn_execute_successes = 100
        qnn_execute_return_code = 0
        app_read_tensors_audited = $failedAudited
        app_read_poison_residual_elements = $failedPoison
        nonfinite_elements = $failedNonfinite
        embedding_input_gradient_unique_canonical_hashes = $failedUnique
        embedding_input_gradient_canonical_hash_frequencies = $failedFrequencies
    }
)
$summary = [ordered]@{
    schema_version = 1
    study = "QNN HTP fixed-state full-graph prefix bisection"
    result = "GOAL_SUCCESS"
    success_path = "C"
    qairt_build_id = "2.48.40.260702151143"
    qnn_api_version = "2.37.0"
    model = [ordered]@{
        batch = 1; tokens = 8; vocabulary = 32; dimension = 16
        heads = 1; layers = 1; feed_forward_dimension = 32
        declared_dtype = "FP32"
    }
    fixed_state_hashes = $expectedHashes
    completed_paired_fresh_processes = $sequentialFull.Count
    additional_numerical_audit_failure_processes = $failureRows.Count
    repeats_per_graph = 100
    primary_result = [ordered]@{
        full_varying_processes = (& $varying $sequentialFull)
        full_tested_processes = $sequentialFull.Count
        stop_after_dinput_varying_processes = (& $varying $sequentialDinput)
        stop_after_dinput_tested_processes = $sequentialDinput.Count
        stop_after_dembedding_varying_processes = (& $varying $sequentialDembedding)
        stop_after_dembedding_tested_processes = $sequentialDembedding.Count
        full_varying_stop_after_dembedding_stable_processes =
            $toggleProcesses.Count
        first_changing_tensor = "embedding_input_gradient"
        first_exposed_candidate_node = "lm_dinput"
        maximum_absolute_difference =
            ($sequentialFull.embedding_input_gradient_repeat_max_abs_difference |
             Measure-Object -Maximum).Maximum
        qnn_execute_return_code =
            [int](($rows.qnn_execute_return_code | Measure-Object -Maximum).Maximum)
        app_read_poison_residual_elements =
            [int](($rows.app_read_poison_residual_elements | Measure-Object -Sum).Sum)
        nonfinite_elements =
            [int](($rows.nonfinite_elements | Measure-Object -Sum).Sum)
    }
    numerical_audit_failure = [ordered]@{
        full_varying_processes = 1
        qnn_execute_return_code = 0
        app_read_poison_residual_elements = $failedPoison
        nonfinite_elements = $failedNonfinite
        shortened_variants_executed = $false
    }
    classification = "GRAPH_VARIANT_OR_RUNTIME_ORDER_DEPENDENT_EXECUTION_VARIABILITY"
    toggle = "in two full-varying processes, the subsequently-created stop_after_dembedding graph is stable; stop_after_dinput is stable in one and varying in one"
    unresolved = @(
        "DRESIDUAL1 and DINPUT_NORM remain native, so lm_dinput is the first exposed candidate rather than a proven causal node",
        "the causal member of the 24-node SGD tail and the backend lowering or memory-planning mechanism are not isolated",
        "each variant uses a separate Runtime/context, so graph structure is confounded with creation and execution order",
        "small process counts estimate prevalence imprecisely"
    )
    publication = [ordered]@{
        contains_raw_tensor = $false
        contains_device_endpoint = $false
        contains_private_path = $false
        contains_qualcomm_binary = $false
        contains_apk = $false
        contains_logcat_capture = $false
    }
}
$variants = @(
    [pscustomobject][ordered]@{
        variant = "full"
        terminal_node = "lm_output_projection_next"
        lm_dembedding_present = $true
        sgd_update_nodes = 24
        next_parameter_outputs = 12
    },
    [pscustomobject][ordered]@{
        variant = "stop_after_dinput"
        terminal_node = "lm_dinput"
        lm_dembedding_present = $false
        sgd_update_nodes = 0
        next_parameter_outputs = 0
    },
    [pscustomobject][ordered]@{
        variant = "stop_after_dembedding"
        terminal_node = "lm_dembedding"
        lm_dembedding_present = $true
        sgd_update_nodes = 0
        next_parameter_outputs = 0
    }
)

$summaryPath = Join-Path $resolvedOutput "summary.json"
$rowsPath = Join-Path $resolvedOutput "process-results.csv"
$variantsPath = Join-Path $resolvedOutput "graph-variants.csv"
$failuresPath = Join-Path $resolvedOutput "numerical-audit-failures.csv"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
$rows | Export-Csv -LiteralPath $rowsPath -NoTypeInformation -Encoding utf8
$variants | Export-Csv -LiteralPath $variantsPath -NoTypeInformation -Encoding utf8
$failureRows | Export-Csv -LiteralPath $failuresPath -NoTypeInformation -Encoding utf8

foreach ($path in @($summaryPath, $rowsPath, $variantsPath, $failuresPath)) {
    $text = Get-Content -Raw -LiteralPath $path
    if ($text -match '(?i)([A-Z]:\\|/data/(?:user|data)/|adb[_ -]?(?:serial|endpoint)|raw[_ -]?(?:callback|logcat)|\.so\b|\.apk\b|\.aab\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)') {
        throw "Forbidden public field or path detected in $([IO.Path]::GetFileName($path))."
    }
}
Write-Output "public_export=SUCCESS"
Write-Output "files=summary.json,process-results.csv,graph-variants.csv,numerical-audit-failures.csv"
