[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "../docs/results/qnn-htp-late-nonfinite-2026-07")
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$expectedOutput = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot 'docs/results/qnn-htp-late-nonfinite-2026-07'))
$actualOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not [string]::Equals($actualOutput, $expectedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must be the allowlisted public result directory."
}
[System.IO.Directory]::CreateDirectory($actualOutput) | Out-Null

$sources = [ordered]@{
    PreFix003 = Join-Path $repositoryRoot 'build/reports/qnn-headless/late-baseline-20260728-230554/device-report.txt'
    Boundary = Join-Path $repositoryRoot 'build/reports/qnn-headless/late-diagnostic-20260729-000220/device-report.txt'
    SameCheckpointPostFix = Join-Path $repositoryRoot 'build/reports/qnn-headless/late-diagnostic-20260729-000220/device-report.txt'
    PostFix003 = Join-Path $repositoryRoot 'build/reports/qnn-headless/late-baseline-20260729-000220/device-report.txt'
    PostFix0003 = Join-Path $repositoryRoot 'build/reports/qnn-headless/late-postfix-lr0003-20260728/device-report.txt'
    CpuControl003 = Join-Path $repositoryRoot 'build/reports/late-nonfinite-cpu-baseline/adam-lr0.003-s320.txt'
}
foreach ($source in $sources.Values) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Expected local report is missing." }
}

function Get-Lines([string]$Path) { @(Get-Content -LiteralPath $Path) }
function Get-Value([string[]]$Lines, [string]$Key) {
    $match = $Lines | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1
    if ($null -eq $match) { throw "Required report field is missing: $Key" }
    return $match.Substring($Key.Length + 1)
}
function Test-Hash([string]$Value) { return $Value -match '^[a-f0-9]{64}$' }
function Test-FiniteNumber([string]$Value) {
    $parsed = 0.0
    return [double]::TryParse(
        $Value, [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -and
        [double]::IsFinite($parsed)
}
function Quote-Csv([string]$Value) {
    if ($Value -match '^[=+@]' -or $Value -match '^-[^0-9.]') { $Value = "'$Value" }
    return '"' + $Value.Replace('"', '""') + '"'
}
function To-Csv([object[]]$Rows, [string[]]$Columns) {
    $output = @((($Columns | ForEach-Object { Quote-Csv $_ }) -join ','))
    foreach ($row in $Rows) {
        $output += (($Columns | ForEach-Object { Quote-Csv ([string]$row[$_]) }) -join ',')
    }
    return ($output -join "`n") + "`n"
}
function Write-Public([string]$Name, [string]$Content) {
    $path = Join-Path $actualOutput $Name
    [System.IO.File]::WriteAllText($path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}
function Get-Seed([string[]]$Lines, [int]$Seed, [string[]]$Fields) {
    $row = [ordered]@{ seed = $Seed }
    foreach ($field in $Fields) { $row[$field] = Get-Value $Lines "seed_${Seed}_$field" }
    return $row
}

$pre003 = Get-Lines $sources.PreFix003
$boundary = Get-Lines $sources.Boundary
$post003 = Get-Lines $sources.PostFix003
$post0003 = Get-Lines $sources.PostFix0003
$sameCheckpoint = Get-Lines $sources.SameCheckpointPostFix
$cpuControl003 = Get-Lines $sources.CpuControl003
$postFixBlocks = [System.Text.RegularExpressions.Regex]::Split(($sameCheckpoint -join "`n"), '(?m)(?=^post_fix_same_checkpoint_seed=)') | Where-Object { $_ -match '^post_fix_same_checkpoint_seed=' }

if ((Get-Value $pre003 'finite_seed_count') -ne '0/5') { throw 'Unexpected pre-fix lr=0.003 count.' }
if ((Get-Value $post003 'finite_seed_count') -ne '5/5') { throw 'Unexpected post-fix lr=0.003 count.' }
if ((Get-Value $post0003 'nan_detected') -ne 'false' -or (Get-Value $post0003 'inf_detected') -ne 'false') { throw 'Unexpected post-fix lr=0.0003 finite status.' }
if ((Get-Value $cpuControl003 'nan_inf_count') -ne '0' -or (Get-Value $cpuControl003 'deterministic_replay') -ne 'true') { throw 'Independent CPU control validation failed.' }
if ((Get-Value $sameCheckpoint 'transformer_centered_scale') -ne '64') { throw 'Same-checkpoint source is not the legacy scale-64 diagnostic.' }
if ($postFixBlocks.Count -ne 5) { throw 'Expected five post-fix same-checkpoint blocks.' }
$postFixSeeds = @()
foreach ($block in $postFixBlocks) {
    $lines = @($block -split "`n")
    $postFixSeeds += [int](Get-Value $lines 'post_fix_same_checkpoint_seed')
    if ((Get-Value $lines 'post_fix_same_checkpoint_replay_count') -ne '100' -or (Get-Value $lines 'post_fix_same_checkpoint_all_finite') -ne 'true' -or (Get-Value $lines 'post_fix_same_checkpoint_deterministic') -ne 'true') { throw 'Post-fix checkpoint replay validation failed.' }
    if ((Get-Value $lines 'post_fix_same_checkpoint_centered_scale') -ne '8') { throw 'Post-fix replay did not use centered scale 8.' }
    foreach ($path in 'a','b','c','d') {
        if ((Get-Value $lines "post_fix_same_checkpoint_path_${path}_finite") -ne 'true') { throw 'A post-fix 2x2 path is non-finite.' }
    }
}
if (($postFixSeeds | Sort-Object -Unique) -join ',' -ne '1,2,3,4,5') { throw 'Post-fix replay seeds are incomplete or duplicated.' }

$seedFields = @('initial_loss','final_loss','initial_accuracy','final_accuracy','last_finite_step','first_nonfinite_step','first_nonfinite_tensor','gradient_norm','adam_m_norm','adam_v_norm','adam_denominator_norm','parameter_norm','qnn_execute_result')
$seedRows = @()
foreach ($seed in 1..5) {
    $item = Get-Seed $pre003 $seed $seedFields
    $item = [ordered]@{ phase='pre_fix'; configuration='adam_lr0.003_s320_no_clip'; learning_rate='0.003'; steps='320'; clipping='disabled' } + $item
    $seedRows += $item
}
foreach ($seed in 1..5) {
    $item = Get-Seed $boundary $seed @('initial_loss','final_loss','initial_accuracy','final_accuracy','last_finite_step','first_nonfinite_step','first_nonfinite_tensor','gradient_norm','adam_m_norm','adam_v_norm','adam_denominator_norm','parameter_norm','qnn_execute_result')
    $item = [ordered]@{ phase='pre_fix'; configuration='adam_lr0.0003_s1000_clip5_boundary'; learning_rate='0.0003'; steps='1000'; clipping='5' } + $item
    $seedRows += $item
}
foreach ($seed in 1..5) {
    $item = Get-Seed $post003 $seed $seedFields
    $item = [ordered]@{ phase='post_fix'; configuration='adam_lr0.003_s320_no_clip'; learning_rate='0.003'; steps='320'; clipping='disabled' } + $item
    $seedRows += $item
}
foreach ($seed in 1..5) {
    $seedRows += [ordered]@{
        phase='post_fix'; configuration='adam_lr0.0003_s1000_clip5'; learning_rate='0.0003'; steps='1000'; clipping='5'; seed=$seed
        initial_loss=(Get-Value $post0003 "seed_${seed}_initial_loss"); final_loss=(Get-Value $post0003 "seed_${seed}_final_loss")
        initial_accuracy=(Get-Value $post0003 "seed_${seed}_initial_accuracy"); final_accuracy=(Get-Value $post0003 "seed_${seed}_final_accuracy")
        last_finite_step='1000'; first_nonfinite_step='NONE'; first_nonfinite_tensor='NONE'; gradient_norm='not recorded in convergence summary'
        adam_m_norm='not recorded in convergence summary'; adam_v_norm='not recorded in convergence summary'; adam_denominator_norm='not recorded in convergence summary'
        parameter_norm='not recorded in convergence summary'; qnn_execute_result='SUCCESS'
    }
}
$seedColumns = @('phase','configuration','learning_rate','steps','clipping','seed') + $seedFields
if ($seedRows.Count -ne 20) { throw 'Expected exactly 20 public seed rows.' }
Write-Public 'seeds.csv' (To-Csv $seedRows $seedColumns)

$cpuRows = @()
$cpuFields = @('initial_train_loss','final_train_loss','initial_eval_loss','final_eval_loss','initial_train_accuracy','final_train_accuracy','initial_eval_accuracy','final_eval_accuracy')
foreach ($seed in 1..5) {
    $row = [ordered]@{ backend='CPU'; configuration='adam_lr0.003_s320_no_clip'; learning_rate='0.003'; steps='320'; clipping='disabled'; seed=$seed }
    foreach ($field in $cpuFields) {
        $value = Get-Value $cpuControl003 "seed_${seed}_$field"
        $parsed = 0.0
        if (-not [double]::TryParse($value, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or -not [double]::IsFinite($parsed)) { throw 'CPU control contains a non-finite metric.' }
        $row[$field] = $value
    }
    $row['finite_through_step'] = '320'
    $row['first_nonfinite_step'] = 'NONE'
    $row['deterministic_replay'] = 'true'
    $cpuRows += $row
}
if ($cpuRows.Count -ne 5) { throw 'Expected five independent CPU control rows.' }
$cpuColumns = @('backend','configuration','learning_rate','steps','clipping','seed') + $cpuFields + @('finite_through_step','first_nonfinite_step','deterministic_replay')
Write-Public 'cpu-control.csv' (To-Csv $cpuRows $cpuColumns)

$checkpointRows = @()
$checkpointFields = @('parameters','adam_m','adam_v','input','target')
foreach ($seed in 1..5) {
    $row = [ordered]@{
        seed=$seed; completed_step=(Get-Value $boundary "seed_${seed}_checkpoint_checkpoint_completed_step")
        optimizer_next_step=(Get-Value $boundary "seed_${seed}_checkpoint_checkpoint_optimizer_next_step")
    }
    foreach ($field in $checkpointFields) {
        $length = Get-Value $boundary "seed_${seed}_checkpoint_${field}_length"
        $hash = Get-Value $boundary "seed_${seed}_checkpoint_${field}_canonical_hash"
        if (-not (Test-Hash $hash)) { throw "Invalid checkpoint hash." }
        $row["${field}_elements"] = $length
        $row["${field}_sha256"] = $hash
    }
    $combined = Get-Value $boundary "seed_${seed}_checkpoint_combined_state_canonical_hash"
    if (-not (Test-Hash $combined)) { throw 'Invalid combined checkpoint hash.' }
    $row['combined_state_sha256'] = $combined
    $checkpointRows += $row
}
if ($checkpointRows.Count -ne 5) { throw 'Expected five checkpoint manifests.' }
$checkpointColumns = @('seed','completed_step','optimizer_next_step') + ($checkpointFields | ForEach-Object { "${_}_elements"; "${_}_sha256" }) + @('combined_state_sha256')
Write-Public 'checkpoints.csv' (To-Csv $checkpointRows $checkpointColumns)

$replayRows = @()
foreach ($seed in 1..5) {
    $count = Get-Value $boundary "seed_${seed}_fixed_replay_count"
    $reproducible = Get-Value $boundary "seed_${seed}_fixed_replay_reproducible"
    if ($count -ne '100' -or $reproducible -ne 'true') { throw 'Expected 100 deterministic pre-fix replays for every seed.' }
    $tensor = Get-Value $boundary "seed_${seed}_first_nonfinite_tensor"
    $producer = Get-Value $boundary "seed_${seed}_first_bad_producer_node"
    $tap = Get-Value $boundary "seed_${seed}_first_bad_tap_name"
    $tapProducer = Get-Value $boundary "seed_${seed}_first_bad_tap_producer_node"
    $previousTap = Get-Value $boundary "seed_${seed}_first_bad_previous_tap_name"
    $previousProducer = Get-Value $boundary "seed_${seed}_first_bad_previous_tap_producer_node"
    $previousFinite = Get-Value $boundary "seed_${seed}_first_bad_previous_tap_finite"
    $signature = Get-Value $boundary "seed_${seed}_fixed_replay_signature"
    $firstBadIndex = Get-Value $boundary "seed_${seed}_first_bad_first_bad_index"
    $previousValue = Get-Value $boundary "seed_${seed}_first_bad_previous_tap_value_at_first_bad_index"
    if ($tensor -ne 'tap_SQUARE2' -or $producer -ne 'tt_ln2_square' -or
        $tap -ne 'SQUARE2' -or $tapProducer -ne 'tt_ln2_square' -or
        $previousTap -ne 'CENTERED_S2' -or $previousProducer -ne 'tt_ln2_center_scale' -or
        $previousFinite -ne 'true' -or
        $signature -notmatch '^tap_SQUARE2:.*:value_bits=0x7f800000$' -or
        $firstBadIndex -notmatch '^\d+$' -or -not (Test-FiniteNumber $previousValue) -or
        [Math]::Abs([double]$previousValue) -lt 256.0) {
        throw "Pre-fix first-bad provenance validation failed for seed $seed."
    }
    $replayRows += [ordered]@{
        phase='pre_fix'; seed=$seed; replay_count=$count; reproducible=$reproducible; outcome='nonfinite'; first_bad_tensor=$tensor; producer_node=$producer
        first_bad_index=$firstBadIndex; bad_value_class='+Inf'; previous_centered_value=$previousValue
    }
}
foreach ($block in $postFixBlocks) {
    $lines = @($block -split "`n")
    $replayRows += [ordered]@{ phase='post_fix_same_checkpoint'; seed=(Get-Value $lines 'post_fix_same_checkpoint_seed'); replay_count=(Get-Value $lines 'post_fix_same_checkpoint_replay_count'); reproducible=(Get-Value $lines 'post_fix_same_checkpoint_deterministic'); outcome='finite'; first_bad_tensor='NONE'; producer_node='NONE'; first_bad_index='NONE'; bad_value_class='NONE'; previous_centered_value='NONE' }
}
Write-Public 'replay.csv' (To-Csv $replayRows @('phase','seed','replay_count','reproducible','outcome','first_bad_tensor','producer_node','first_bad_index','bad_value_class','previous_centered_value'))

$twoByTwoRows = @()
$blocks = [System.Text.RegularExpressions.Regex]::Split(($boundary -join "`n"), '(?m)(?=^two_by_two_checkpoint_seed=)') | Where-Object { $_ -match '^two_by_two_checkpoint_seed=' }
if ($blocks.Count -ne 5) { throw 'Expected five pre-fix 2x2 blocks.' }
$twoByTwoSeeds = @()
foreach ($block in $blocks) {
    $lines = @($block -split "`n")
    $row = [ordered]@{ phase='pre_fix'; seed=(Get-Value $lines 'two_by_two_checkpoint_seed'); step=(Get-Value $lines 'two_by_two_checkpoint_step'); gradient_max_abs_error=(Get-Value $lines 'two_by_two_cpu_htp_gradient_max_abs_error') }
    foreach ($path in @('A_cpu_gradient_cpu_adam','B_htp_gradient_cpu_adam','C_cpu_gradient_htp_adam','D_htp_gradient_htp_adam')) { $row[$path] = Get-Value $lines "two_by_two_${path}_finite" }
    $twoByTwoSeeds += [int]$row.seed
    if ($row.A_cpu_gradient_cpu_adam -ne 'true' -or
        $row.B_htp_gradient_cpu_adam -ne 'false' -or
        $row.C_cpu_gradient_htp_adam -ne 'true' -or
        $row.D_htp_gradient_htp_adam -ne 'false' -or
        (Get-Value $lines 'two_by_two_first_bad_stage') -ne 'tap_SQUARE2' -or
        (Get-Value $lines 'two_by_two_first_bad_producer_node') -ne 'tt_ln2_square') {
        throw 'Pre-fix 2x2 provenance validation failed.'
    }
    $row['c_parameter_max_abs_error'] = Get-Value $lines 'two_by_two_C_cpu_gradient_htp_adam_parameter_max_abs_error'
    $row['c_m_next_max_abs_error'] = Get-Value $lines 'two_by_two_C_m_next_max_abs_error'
    $row['c_v_next_max_abs_error'] = Get-Value $lines 'two_by_two_C_v_next_max_abs_error'
    $twoByTwoRows += $row
}
if (($twoByTwoSeeds | Sort-Object -Unique) -join ',' -ne '1,2,3,4,5') { throw 'Pre-fix 2x2 seeds are incomplete or duplicated.' }
foreach ($block in $postFixBlocks) {
    $lines = @($block -split "`n")
    $twoByTwoRows += [ordered]@{
        phase='post_fix_same_checkpoint'; seed=(Get-Value $lines 'post_fix_same_checkpoint_seed'); step=(Get-Value $lines 'post_fix_same_checkpoint_step')
        gradient_max_abs_error=(Get-Value $lines 'post_fix_same_checkpoint_cpu_htp_gradient_max_abs_error'); A_cpu_gradient_cpu_adam=(Get-Value $lines 'post_fix_same_checkpoint_path_a_finite'); B_htp_gradient_cpu_adam=(Get-Value $lines 'post_fix_same_checkpoint_path_b_finite'); C_cpu_gradient_htp_adam=(Get-Value $lines 'post_fix_same_checkpoint_path_c_finite'); D_htp_gradient_htp_adam=(Get-Value $lines 'post_fix_same_checkpoint_path_d_finite')
        c_parameter_max_abs_error=(Get-Value $lines 'post_fix_same_checkpoint_path_c_parameter_max_abs_error'); c_m_next_max_abs_error=(Get-Value $lines 'post_fix_same_checkpoint_path_c_m_max_abs_error'); c_v_next_max_abs_error=(Get-Value $lines 'post_fix_same_checkpoint_path_c_v_max_abs_error')
    }
}
$twoByTwoColumns = @('phase','seed','step','gradient_max_abs_error','A_cpu_gradient_cpu_adam','B_htp_gradient_cpu_adam','C_cpu_gradient_htp_adam','D_htp_gradient_htp_adam','c_parameter_max_abs_error','c_m_next_max_abs_error','c_v_next_max_abs_error')
Write-Public 'two-by-two.csv' (To-Csv $twoByTwoRows $twoByTwoColumns)

$summary = [ordered]@{
    schema_version = 1
    classification = 'HTP_BACKWARD_NUMERIC_ERROR'
    root_cause = 'Training-graph LayerNorm2 centered scale 64 crosses the HTP FP16-like multiplication limit when a finite centered activation reaches about 257; the following square becomes positive infinity.'
    fix = 'Reduce the application centered scale from 64 to 8.'
    pre_fix = [ordered]@{ lr_003_htp_finite_seeds='0/5'; lr_003_cpu_control_finite_seeds='5/5'; lr_0003_finite_seeds='0/5'; first_bad_tensor='tap_SQUARE2'; producer_node='tt_ln2_square' }
    post_fix = [ordered]@{ lr_003_finite_seeds='5/5'; lr_0003_finite_seeds='5/5'; same_checkpoint_replay_finite='500/500'; deterministic=$true }
    optimizer_formula = [ordered]@{ beta1=0.9; beta2=0.999; epsilon='1e-8'; bias_correction='CPU'; clip_order='global_norm_then_gradient_scale_then_adam'; formula_match=$true }
    caveat = 'At legacy accumulated HTP states, path C/D parameter maximum differences can be large while m/v differences remain small and all paths are finite after the fix; numerical parity remains an open follow-up.'
}
Write-Public 'summary.json' (($summary | ConvertTo-Json -Depth 6) + "`n")

$readme = @'
# QNN HTP late-nonfinite public result bundle

This directory is an allowlisted, generated summary of the July 2026 investigation. It contains aggregate metrics and SHA-256 identifiers only; checkpoint contents and platform traces are intentionally excluded.

Files:

- `summary.json`: classification, fix, formula audit, and residual numerical-parity caveat.
- `seeds.csv`: pre-fix and post-fix five-seed results.
- `cpu-control.csv`: independent CPU control with the actual train and evaluation metrics for all five seeds.
- `checkpoints.csv`: five last-finite checkpoint manifests (element counts and SHA-256 identifiers).
- `replay.csv`: each pre-fix checkpoint replayed 100 times, plus the post-fix same-checkpoint replay.
- `two-by-two.csv`: CPU/HTP gradient and Adam path isolation before and after the fix.

The public exporter validates source field counts, replay counts, and identifier syntax before writing these files. It refuses any output directory other than this directory and scans generated files for restricted data patterns.
'@
Write-Public 'README.md' ($readme.Trim() + "`n")

$restrictedPatterns = @(
    '(?i)raw[_ -]?(hex|float)',
    '(?i)logcat|\.apk\b|libqnn|/data/|[a-z]:\\',
    '(?i)endpoint|device[_ -]?id|serial',
    '(?i)private[_ -]?(checkpoint|trace|dump)',
    '(?i)\b[0-9a-f]{129,}\b',
    '(?i)\b0x[0-9a-f]+\b'
)
$publicFiles = Get-ChildItem -LiteralPath $actualOutput -File
foreach ($file in $publicFiles) {
    $text = Get-Content -LiteralPath $file.FullName -Raw
    foreach ($pattern in $restrictedPatterns) {
        if ($text -match $pattern) { throw "Restricted data pattern found in generated public output: $($file.Name)" }
    }
}
Write-Host "Exported $($publicFiles.Count) public late-nonfinite result files."
