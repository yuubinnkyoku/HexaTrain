param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\reports\qnn-headless"),
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "docs\results\qnn-htp-fixed-state-reproducibility-2026-07")
)

$ErrorActionPreference = "Stop"
$repository = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$allowedOutput = [IO.Path]::GetFullPath((Join-Path $repository "docs\results\qnn-htp-fixed-state-reproducibility-2026-07"))
$resolvedOutput = [IO.Path]::GetFullPath($OutputDir)
if (-not $resolvedOutput.Equals($allowedOutput, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Public output is restricted to the documented result directory."
}
[IO.Directory]::CreateDirectory($resolvedOutput) | Out-Null

function Read-Report([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing required private aggregate report." }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $values[$Matches[1]] = $Matches[2] }
    }
    return $values
}
function Number($Value) {
    if ($null -eq $Value -or $Value -eq "") { return $null }
    return [double]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
}

$fixed = 1..5 | ForEach-Object {
    Read-Report (Join-Path $InputRoot "repro-with-adam-0$_\device-report.txt")
}
$phase = 1..5 | ForEach-Object {
    Read-Report (Join-Path $InputRoot "phase01-bisect-0$_\device-report.txt")
}
$scopes = [ordered]@{
    a_same_graph = 100
    b_recreate_graph_context_reuse = 30
    c_recreate_context_backend_reuse = 20
    d_backend_runtime_recreate = 10
}
$scopeRows = foreach ($checkpoint in @("E", "D", "L")) {
    foreach ($scope in $scopes.Keys) {
        $prefix = "snapshot_${checkpoint}_scope_${scope}"
        [pscustomobject][ordered]@{
            checkpoint = $checkpoint
            scope = $scope
            attempts_per_process = $scopes[$scope]
            processes = 5
            maximum_unique_canonical_hashes = ($fixed | ForEach-Object { [int]$_["${prefix}_unique_canonical_hashes"] } | Measure-Object -Maximum).Maximum
            maximum_repeat_max_abs_difference = ($fixed | ForEach-Object { Number $_["${prefix}_repeat_max_abs_difference"] } | Measure-Object -Maximum).Maximum
            maximum_nonfinite_count = ($fixed | ForEach-Object { [int]$_["${prefix}_nonfinite_count"] } | Measure-Object -Maximum).Maximum
            maximum_poison_residual_elements = ($fixed | ForEach-Object { [int]$_["${prefix}_app_read_poison_residual_elements"] } | Measure-Object -Maximum).Maximum
        }
    }
}
$tensors = @("logits", "softmax_probability", "dlogits", "output_projection_gradient",
             "transformer_output", "embedding_input_gradient", "token_embedding_gradient",
             "next_token_embedding")
$fullRows = foreach ($checkpoint in @("E", "D", "L")) {
    foreach ($tensor in $tensors) {
        $prefix = "full_graph_${checkpoint}_${tensor}"
        $representatives = @($fixed | ForEach-Object { $_["${prefix}_representative_canonical_hash"] })
        [pscustomobject][ordered]@{
            checkpoint = $checkpoint
            tensor = $tensor
            repeats_per_process = 100
            processes = 5
            maximum_unique_canonical_hashes_within_process = ($fixed | ForEach-Object { [int]$_["${prefix}_unique_canonical_hashes"] } | Measure-Object -Maximum).Maximum
            unique_representative_canonical_hashes_across_processes = @($representatives | Sort-Object -Unique).Count
            maximum_repeat_max_abs_difference = ($fixed | ForEach-Object { Number $_["${prefix}_repeat_max_abs_difference"] } | Measure-Object -Maximum).Maximum
            maximum_poison_residual_elements = ($fixed | ForEach-Object { [int]$_["${prefix}_app_read_poison_residual_elements"] } | Measure-Object -Maximum).Maximum
        }
    }
}
$phaseRows = foreach ($process in 1..5) {
    foreach ($seed in 1..5) {
        [pscustomobject][ordered]@{
            process = $process
            seed = $seed
            completed_steps = [int]$phase[$process - 1]["seed_${seed}_completed_steps"]
        }
    }
}
$summary = [ordered]@{
    schema_version = 1
    study = "QNN HTP fixed-state reproducibility"
    result = "PARTIAL_SUCCESS"
    qairt_build_id = "2.48.40.260702151143"
    qnn_api_version = "2.37.0"
    model = [ordered]@{ batch = 1; tokens = 8; vocabulary = 32; dimension = 16; heads = 1; layers = 1; ffn = 32; declared_dtype = "FP32" }
    fixed_state = [ordered]@{
        checkpoints = @("E", "D", "L")
        processes = 5
        repeats_per_full_graph_checkpoint = 100
        first_varying_tensor = "embedding_input_gradient"
        first_varying_node = "lm_dinput"
        first_varying_scope = "full_graph_same_graph"
        affected_processes = 1
        maximum_absolute_difference = 5431.98079
        nonfinite_count = "NOT_MEASURED_IN_SOURCE_RUNS"
        app_read_poison_residual_elements = 0
        standalone_lm_dembedding_all_scope_unique_canonical_hashes = 1
    }
    classification = "BACKEND_EXECUTION_VARIABILITY_FULL_GRAPH"
    app_buffer_lifetime_defect_found = $false
    physical_guard = "UNSUPPORTED_RUNTIME_VECTOR_API"
    phase01 = [ordered]@{
        optimizer = "Adam"
        learning_rate = 0.0003
        gradient_clip = 10
        maximum_steps = 1000
        all_process_seed_runs_finite = $false
    }
    publication = [ordered]@{
        contains_raw_checkpoint = $false
        contains_device_endpoint = $false
        contains_private_path = $false
        contains_qualcomm_binary = $false
    }
}

$summaryPath = Join-Path $resolvedOutput "summary.json"
$scopePath = Join-Path $resolvedOutput "scope-matrix.csv"
$fullPath = Join-Path $resolvedOutput "full-graph-tensors.csv"
$phasePath = Join-Path $resolvedOutput "phase01-repeats.csv"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
$scopeRows | Export-Csv -LiteralPath $scopePath -NoTypeInformation -Encoding utf8
$fullRows | Export-Csv -LiteralPath $fullPath -NoTypeInformation -Encoding utf8
$phaseRows | Export-Csv -LiteralPath $phasePath -NoTypeInformation -Encoding utf8

foreach ($path in @($summaryPath, $scopePath, $fullPath, $phasePath)) {
    $text = Get-Content -Raw -LiteralPath $path
    if ($text -match '(?i)([A-Z]:\\|/data/(?:user|data)/|adb[_ -]?(?:serial|endpoint)|raw[_ -]?(?:callback|logcat)|\.so\b|\.apk\b|\.aab\b)') {
        throw "Forbidden public field or path detected in $([IO.Path]::GetFileName($path))."
    }
}
Write-Output "public_export=SUCCESS"
Write-Output "files=summary.json,scope-matrix.csv,full-graph-tensors.csv,phase01-repeats.csv"
