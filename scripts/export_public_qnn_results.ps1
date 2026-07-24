[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceReportDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$ExperimentSourceCommit
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required report file is missing: $Path"
    }
}

function Parse-Number {
    param([object]$Value)
    [double]::Parse([string]$Value, [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture)
}

function Assert-Near {
    param([string]$Name, [object]$Actual, [object]$Expected, [double]$Tolerance = 1.0e-12)
    $difference = [Math]::Abs((Parse-Number $Actual) - (Parse-Number $Expected))
    if ($difference -gt $Tolerance) {
        throw "$Name differs between source reports: actual=$Actual expected=$Expected"
    }
}

function Mean-Or-Empty {
    param([object[]]$Values)
    $numbers = @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        ForEach-Object { Parse-Number $_ })
    if ($numbers.Count -eq 0) { return '' }
    ($numbers | Measure-Object -Average).Average.ToString(
        'G17', [Globalization.CultureInfo]::InvariantCulture)
}

$source = (Resolve-Path -LiteralPath $SourceReportDir).Path
$summaryPath = Join-Path $source 'summary.json'
$opPath = Join-Path $source 'op-correctness.csv'
$fullStepPath = Join-Path $source 'full-step-correctness.csv'
$seedsPath = Join-Path $source 'seeds.csv'
$performancePath = Join-Path $source 'performance.csv'
$breakEvenPath = Join-Path $source 'break-even.csv'
$trajectoryPath = Join-Path $source 'weight-trajectory.csv'

@($summaryPath, $opPath, $fullStepPath, $seedsPath, $performancePath,
    $breakEvenPath, $trajectoryPath) | ForEach-Object { Require-File $_ }

$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
$ops = @(Import-Csv -LiteralPath $opPath)
$fullStep = @(Import-Csv -LiteralPath $fullStepPath)
$seeds = @(Import-Csv -LiteralPath $seedsPath)
$performance = @(Import-Csv -LiteralPath $performancePath)
$breakEven = @(Import-Csv -LiteralPath $breakEvenPath)
$trajectory = @(Import-Csv -LiteralPath $trajectoryPath)

if ($summary.status -ne 'SUCCESS' -or $summary.cpu_fallback -ne $false -or
    $summary.nan_inf -ne $false) {
    throw 'The source summary is not a successful no-fallback finite run.'
}
if ($fullStep.Count -ne 1 -or $fullStep[0].status -ne 'SUCCESS') {
    throw 'Expected exactly one successful small full-step correctness row.'
}
if ($seeds.Count -ne 15 -or @($seeds.seed | Sort-Object -Unique).Count -ne 5) {
    throw 'Expected 5 seeds and 3 modes per seed.'
}

$mse = $ops | Where-Object test -eq 'mse_dp'
$dp = $ops | Where-Object test -eq 'dp'
$sgd = $ops | Where-Object test -eq 'sgd'
Assert-Near 'micro MSE' $mse.max_abs_error $summary.micro.mse_max_abs_error
Assert-Near 'micro dP' $dp.max_abs_error $summary.micro.dp_max_abs_error
Assert-Near 'micro SGD' $sgd.max_abs_error $summary.micro.sgd_max_abs_error
Assert-Near 'small loss' $fullStep[0].loss_abs_error $summary.small.loss_abs_error
Assert-Near 'small W1_next' $fullStep[0].w1_next_max_abs_error $summary.small.w1_next_max_abs_error
Assert-Near 'small W2_next' $fullStep[0].w2_next_max_abs_error $summary.small.w2_next_max_abs_error
$maxSeedLossDifference = ($seeds | Where-Object mode -eq 'QNN_HTP_MLP_FULL_STEP' |
    ForEach-Object { Parse-Number $_.cpu_htp_final_loss_difference } |
    Measure-Object -Maximum).Maximum
Assert-Near 'maximum seed final-loss difference' $maxSeedLossDifference $summary.seeds.max_final_loss_difference

if ([string]::IsNullOrWhiteSpace($ExperimentSourceCommit)) {
    $ExperimentSourceCommit = (& git rev-parse HEAD).Trim()
}
if ($ExperimentSourceCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw 'ExperimentSourceCommit must be a full 40-character Git commit hash.'
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$environment = [ordered]@{
    device_model = 'nubia Z80 Ultra / NX741J'
    soc = 'SM8850'
    htp_architecture = [string]$summary.htp_architecture
    android_version = '16'
    android_api_level = 36
    qairt_build_id = [string]$summary.qairt_build_id
    qnn_core_api = [string]$summary.qnn_core_api
    htp_backend_api = [string]$summary.htp_backend_api
    android_ndk = '26.2.11394342'
    cmake = '3.22.1'
    abi = 'arm64-v8a'
    experiment_source_commit = $ExperimentSourceCommit.ToLowerInvariant()
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'environment.json'),
    ($environment | ConvertTo-Json -Depth 3) + "`n", [Text.UTF8Encoding]::new($false))

$small = $fullStep[0]
$correctnessRows = @(
    [pscustomobject]@{test='MSE micro test';batch_size=2;input_dim='';hidden_dim='';output_dim=3;metric='absolute_error';value=$mse.max_abs_error;tolerance='1e-5';status=$mse.status;cpu_fallback=$mse.cpu_fallback;nan_inf='false'}
    [pscustomobject]@{test='dP micro test';batch_size=2;input_dim='';hidden_dim='';output_dim=3;metric='max_absolute_error';value=$dp.max_abs_error;tolerance='5e-4';status=$dp.status;cpu_fallback=$dp.cpu_fallback;nan_inf='false'}
    [pscustomobject]@{test='SGD micro test';batch_size=2;input_dim='';hidden_dim='';output_dim=3;metric='max_absolute_error';value=$sgd.max_abs_error;tolerance='5e-4';status=$sgd.status;cpu_fallback=$sgd.cpu_fallback;nan_inf='false'}
)
$smallMetrics = @(
    [pscustomobject]@{name='small prediction';field='prediction_max_abs_difference';metric='max_absolute_error';tolerance='1e-3'}
    [pscustomobject]@{name='small loss';field='loss_abs_error';metric='absolute_error';tolerance='1e-3'}
    [pscustomobject]@{name='small dP';field='dp_max_abs_error';metric='max_absolute_error';tolerance='1e-3'}
    [pscustomobject]@{name='small dW2';field='dw2_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
    [pscustomobject]@{name='small dH';field='dh_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
    [pscustomobject]@{name='small mask mismatch';field='mask_mismatch_count';metric='mismatch_count';tolerance='0 or abs(CPU Z1)<1e-3'}
    [pscustomobject]@{name='small dZ1';field='dz1_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
    [pscustomobject]@{name='small dW1';field='dw1_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
    [pscustomobject]@{name='small W1_next';field='w1_next_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
    [pscustomobject]@{name='small W2_next';field='w2_next_max_abs_error';metric='max_absolute_error';tolerance='1e-2'}
)
foreach ($definition in $smallMetrics) {
    $correctnessRows += [pscustomobject]@{
        test=$definition.name;batch_size=$small.batch_size;input_dim=$small.input_dim
        hidden_dim=$small.hidden_dim;output_dim=$small.output_dim;metric=$definition.metric
        value=$small.($definition.field);tolerance=$definition.tolerance;status=$small.status
        cpu_fallback=$small.cpu_fallback;nan_inf=$small.nan_inf
    }
}
$correctnessRows | Export-Csv -LiteralPath (Join-Path $OutputDir 'correctness.csv') -NoTypeInformation -Encoding utf8

$modeNames = @{
    QNN_CPU_MLP_TRAINING = 'cpu'
    QNN_HTP_MLP_FUSED_BACKWARD = 'htp_fused_backward'
    QNN_HTP_MLP_FULL_STEP = 'htp_full_step'
}
$publicSeeds = foreach ($row in $seeds) {
    [pscustomobject]@{
        seed=$row.seed;mode=$modeNames[$row.mode];initial_loss=$row.initial_loss
        final_loss=$row.final_loss;loss_reduction_ratio=$row.loss_reduction_ratio
        prediction_metric=$row.prediction_max_abs_error
        cpu_reference_final_loss_difference=$row.cpu_htp_final_loss_difference
        cpu_fallback=$row.cpu_fallback;nan_inf=$row.nan_inf;status=$row.status
    }
}
$publicSeeds | Export-Csv -LiteralPath (Join-Path $OutputDir 'seeds.csv') -NoTypeInformation -Encoding utf8

$publicPerformance = @()
foreach ($shape in ($performance | Group-Object batch_size,input_dim,hidden_dim,output_dim)) {
    $first = $shape.Group[0]
    $break = $breakEven | Where-Object {
        $_.shape -eq "$($first.batch_size), $($first.input_dim), $($first.hidden_dim), $($first.output_dim)"
    }
    foreach ($sourceMode in @('QNN_HTP_MLP_FUSED_BACKWARD_BENCHMARK', 'QNN_HTP_MLP_FULL_STEP_BENCHMARK')) {
        $runs = @($shape.Group | Where-Object mode -eq $sourceMode)
        $isFullStep = $sourceMode -eq 'QNN_HTP_MLP_FULL_STEP_BENCHMARK'
        $temperatures = @($runs.temperature_before + $runs.temperature_after | ForEach-Object { Parse-Number $_ })
        $thermal = @($runs.thermal_status_before + $runs.thermal_status_after | ForEach-Object { Parse-Number $_ })
        $publicPerformance += [pscustomobject]@{
            mode=if ($isFullStep) {'htp_full_step'} else {'htp_fused_backward'}
            batch_size=$first.batch_size;input_dim=$first.input_dim
            hidden_dim=$first.hidden_dim;output_dim=$first.output_dim
            repetitions=$runs.Count;aggregation='mean_of_run_medians'
            full_step_mean_us=Mean-Or-Empty $runs.full_step_median_us
            full_step_median_us=''
            training_execute_median_us=if ($isFullStep) {Mean-Or-Empty $runs.steady_execute_median_us} else {''}
            weight_handoff_median_us=if ($isFullStep) {Mean-Or-Empty $runs.weight_handoff_median_us} else {''}
            initialization_us=Mean-Or-Empty $runs.initialization_us
            speedup_vs_fused=if ($isFullStep) {$break.speedup} else {''}
            temperature_min_c=($temperatures | Measure-Object -Minimum).Minimum
            temperature_max_c=($temperatures | Measure-Object -Maximum).Maximum
            thermal_status_max=($thermal | Measure-Object -Maximum).Maximum
            cpu_fallback=(@($runs.cpu_fallback | Sort-Object -Unique) -join ',')
            status=(@($runs.status | Sort-Object -Unique) -join ',')
        }
    }
}
$publicPerformance | Sort-Object {[int]$_.batch_size}, {[int]$_.input_dim}, mode |
    Export-Csv -LiteralPath (Join-Path $OutputDir 'performance.csv') -NoTypeInformation -Encoding utf8

$publicTrajectory = foreach ($row in $trajectory) {
    if ([int]$row.step -notin @(0, 1, 2, 5, 10, 20, 50, 100, 640)) { continue }
    [pscustomobject]@{
        seed=$row.seed;step=if ([int]$row.step -eq 640) {'final'} else {$row.step}
        mode='htp_full_step';loss=$row.loss;w1_l2_norm=$row.w1_l2_norm
        w2_l2_norm=$row.w2_l2_norm
        cpu_htp_w1_max_abs_difference=$row.cpu_htp_w1_max_abs_difference
        cpu_htp_w2_max_abs_difference=$row.cpu_htp_w2_max_abs_difference
    }
}
$publicTrajectory | Export-Csv -LiteralPath (Join-Path $OutputDir 'weight-trajectory-summary.csv') -NoTypeInformation -Encoding utf8

$traceSource = Join-Path $source 'runs\log-size-640.txt'
Require-File $traceSource
$traceFields = [ordered]@{}
foreach ($line in ((Get-Content -LiteralPath $traceSource -Raw) -split "`r?`n")) {
    if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { $traceFields[$Matches[1]] = $Matches[2] }
}
$traceAllowList = @(
    'api_trace_version','api_trace_backend_requested','api_trace_backend_library',
    'api_trace_backend_library_load_result','api_trace_provider_symbol_resolved',
    'api_trace_provider_count','api_trace_selected_provider_index',
    'api_trace_selected_core_api_version','api_trace_selected_backend_api_version',
    'api_trace_runtime_backend_build_id','api_trace_backend_create_symbol_library',
    'api_trace_device_create_symbol_library','api_trace_context_create_symbol_library',
    'api_trace_graph_create_symbol_library','api_trace_graph_finalize_symbol_library',
    'api_trace_graph_execute_symbol_library','api_trace_backend_create_called',
    'api_trace_backend_create_result','api_trace_backend_handle_nonnull',
    'api_trace_device_create_called','api_trace_device_create_result',
    'api_trace_device_handle_nonnull','api_trace_context_create_called',
    'api_trace_context_create_result','api_trace_context_handle_nonnull',
    'api_trace_full_step_graph_create_called','api_trace_full_step_graph_create_result',
    'api_trace_full_step_graph_handle_nonnull','api_trace_full_step_graph_finalize_called',
    'api_trace_full_step_graph_finalize_result','api_trace_graph_execute_attempt_count',
    'api_trace_graph_execute_success_count','api_trace_graph_execute_failure_count',
    'api_trace_graph_execute_first_call_index','api_trace_graph_execute_first_result',
    'api_trace_graph_execute_last_call_index','api_trace_graph_execute_last_result',
    'api_trace_graph_execute_first_failure_call','api_trace_failure_injection_enabled',
    'api_trace_last_qnn_result','api_trace_effective_result',
    'api_trace_cpu_backend_initialized','api_trace_fallback_attempted',
    'api_trace_fallback_succeeded'
)
$traceLines = @('PhoneLM public QNN HTP API trace evidence',
    'evidence_mode=counter_only_benchmark','callback_capture=false','qnn_log_level=WARN')
foreach ($key in $traceAllowList) {
    if (-not $traceFields.Contains($key)) { throw "Missing public trace field: $key" }
    $traceLines += "$key=$($traceFields[$key])"
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'api-trace-evidence.txt'),
    ($traceLines -join "`n") + "`n", [Text.UTF8Encoding]::new($false))

$dangerPatterns = @(
    '[A-Za-z]:\\', '\\Users\\', '\\ghq\\', '/data/user/', '/sdcard/',
    '(?i)\b(?:192\.168|10\.(?:\d{1,3}\.){2}|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}',
    '(?i)\bsk-[A-Za-z0-9_-]+', '(?i)BEGIN [A-Z ]*PRIVATE KEY',
    '(?m)^qnn_callback_', '(?m)^qnn_callback_begin$'
)
foreach ($file in (Get-ChildItem -LiteralPath $OutputDir -File)) {
    if ($file.Extension -in @('.so','.apk','.aab','.jks','.keystore','.pem','.key','.log')) {
        throw "Disallowed generated file type: $($file.Name)"
    }
    $content = Get-Content -LiteralPath $file.FullName -Raw
    foreach ($pattern in $dangerPatterns) {
        if ($content -match $pattern) {
            throw "Potentially sensitive content in $($file.Name): pattern $pattern"
        }
    }
}
Write-Host "Exported public QNN results to $OutputDir"
