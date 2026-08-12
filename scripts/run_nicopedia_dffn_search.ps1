# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Research-only D/FFN search orchestrator for the Nicopedia HTP trainer.
#
# This script deliberately does not alter the production preset or UI.  Its
# inputs and all private measurements live below build/.  A device execution
# is Tier 3 and therefore requires -AllowTier3Research explicitly.
param(
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [ValidateSet('Plan', 'Screen', 'Extended', 'Final')][string]$Phase = 'Plan',
    [switch]$AllowTier3Research,
    [switch]$SelfTest,
    [int]$ScreenSteps = 320,
    [int]$ExtendedSteps = 1000,
    [int]$FinalSteps = 4000,
    [int[]]$ExtendedSeeds = @(1, 2, 4),
    [int]$CheckpointInterval = 320,
    [int]$ValidationChunks = 512,
    [int]$DevelopmentChunks = 512,
    [int]$MaxNewBytes = 64,
    [string]$ReportRoot = '',
    [string]$TrainingRunner = '',
    [string]$EvalRunner = '',
    [string]$GenerationRunner = '',
    [switch]$SkipBuild,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
if (-not $ReportRoot) { $ReportRoot = Join-Path $root 'build\reports\nicopedia-dffn-search' }
if (-not $TrainingRunner) { $TrainingRunner = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1' }
if (-not $EvalRunner) { $EvalRunner = Join-Path $PSScriptRoot 'run_nicopedia_htp_eval.ps1' }
if (-not $GenerationRunner) { $GenerationRunner = Join-Path $PSScriptRoot 'run_nicopedia_htp_generate.ps1' }
$script:researchBuildPerformed = $false
$script:researchInstallPerformed = $false

$buildRoot = [IO.Path]::GetFullPath((Join-Path $root 'build'))
$resolvedReportRoot = [IO.Path]::GetFullPath($ReportRoot)
if (-not ($resolvedReportRoot.Equals($buildRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $resolvedReportRoot.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase))) {
    throw 'RESEARCH_REPORT_ROOT_MUST_BE_UNDER_BUILD'
}
$ReportRoot = $resolvedReportRoot

$fixed = [ordered]@{ layers = 19; heads = 2; tokens = 32; vocabulary = 256; batch_size = 8; learning_rate = 0.003; checkpoint_format = 'NPRTCKPTV2' }
$script:researchBuildPerformed = $false
$script:researchInstallPerformed = $false
$candidates = @(
    [pscustomobject]@{ id = 'anchor-d16-f32'; dimension = 16; feed_forward_dimension = 32; family = 'anchor' },
    [pscustomobject]@{ id = 'd32-f32'; dimension = 32; feed_forward_dimension = 32; family = 'dimension_only' },
    [pscustomobject]@{ id = 'd16-f64'; dimension = 16; feed_forward_dimension = 64; family = 'ffn_only' },
    [pscustomobject]@{ id = 'd32-f64'; dimension = 32; feed_forward_dimension = 64; family = 'dimension_and_ffn' }
)

function Get-ModelEstimate {
    param([Parameter(Mandatory = $true)]$Candidate)
    # Exact flattened parameter count: tied input/output vocabulary matrices
    # plus, per layer, Q/K/V/O, four layer-norm vectors and two FFN matrices.
    # Optimizer m/v and one gradient buffer are extra working-set estimates;
    # checkpoint payload stores parameters+m+v.
    $d = [int64]$Candidate.dimension; $f = [int64]$Candidate.feed_forward_dimension
    $v = [int64]$fixed.vocabulary; $t = [int64]$fixed.tokens; $l = [int64]$fixed.layers
    $embedding = 2 * $v * $d
    $perLayer = 4 * $d * $d + 2 * $d * $f + 4 * $d
    $parameters = $embedding + $l * $perLayer
    [pscustomobject][ordered]@{
        parameter_count = $parameters
        parameter_bytes_fp32 = 4 * $parameters
        checkpoint_payload_estimate_bytes = 12 * $parameters
        optimizer_working_set_estimate_bytes = 16 * $parameters
        activation_estimate_bytes_per_batch = 4 * $fixed.batch_size * $fixed.tokens * ($d * 14 + $f * 4 + $v * 3)
    }
}

function Get-Map([string]$Path) {
    $map = @{}; foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([^=\r\n]+)=(.*)$') { $map[$Matches[1]] = $Matches[2] }
    }; return $map
}
function Require-Health([hashtable]$Map, [string]$Kind) {
    foreach ($key in @('status','cpu_fallback')) {
        if (-not $Map.ContainsKey($key)) { throw "RESEARCH_${Kind}_FIELD_MISSING: $key" }
    }
    # Return code and application-visible finite status are intentionally
    # independent fields in research-summary.csv.  Do not derive one from the
    # other and do not treat CPU fallback as successful HTP execution.
    foreach ($key in @('qnn_return_code_success','output_tensors_finite')) {
        if (-not $Map.ContainsKey($key)) { throw "RESEARCH_${Kind}_EXPLICIT_HEALTH_FIELD_MISSING: $key" }
    }
    $returnOk = $Map.qnn_return_code_success -eq 'true'
    $finiteOk = $Map.output_tensors_finite -eq 'true'
    [pscustomobject]@{ status_success = ($Map.status -eq 'SUCCESS'); qnn_return_success = $returnOk; tensors_finite = $finiteOk; cpu_fallback = ($Map.cpu_fallback -eq 'true'); status = $Map.status }
}
function Assert-RunnerContract([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "RESEARCH_${Name}_RUNNER_MISSING" }
    $source = Get-Content -LiteralPath $Path -Raw
    foreach ($parameter in @('Dimension', 'FeedForwardDimension')) {
        if ($source -notmatch "(?i)\b$parameter\b") { throw "RESEARCH_${Name}_RUNNER_DIMENSION_CONTRACT_MISSING: $parameter" }
    }
}
function Write-Plan {
    [IO.Directory]::CreateDirectory($ReportRoot) | Out-Null
    $rows = foreach ($candidate in $candidates) {
        $estimate = Get-ModelEstimate $candidate
        [ordered]@{ candidate = $candidate; estimate = $estimate }
    }
    $plan = [ordered]@{
        schema = 'PHONELM_NICOPEDIA_DFFN_SEARCH_V1'; qairt_sdk_root = '<fixed-local-root>'; qairt_build_id = $ExpectedBuildId
        fixed = $fixed; screening = [ordered]@{ steps = $ScreenSteps; seeds = @(1); validation_chunks = $ValidationChunks; development_chunks = $DevelopmentChunks }
        extension = [ordered]@{ steps = $ExtendedSteps; seeds = $ExtendedSeeds; generation_max_new_bytes = $MaxNewBytes }
        final = [ordered]@{ steps = $FinalSteps; seeds = $ExtendedSeeds; full_cap_eval_required = $true; resume_semantics = 'NPRTCKPTV2' }
        cutoff = @('drop on nonzero QNN return', 'drop on nonfinite application-visible tensors', 'drop on CPU fallback', 'drop on failed status', 'drop if development NLL exceeds anchor by > 0.03', 'drop if median end-to-end wall step exceeds anchor by > 2x; report HTP execute ms separately', 'drop if checkpoint exceeds anchor by > 4x', 'advance anchor plus at most two healthy non-anchor candidates')
        candidates = $rows
    }
    $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ReportRoot 'experiment-plan.json') -Encoding utf8
    return $plan
}
function Invoke-ResearchSegment($Candidate, [int]$Seed, [int]$Steps, [string]$Stage, [int]$ResumeStep = 0, [switch]$FullCapEval) {
    $tag = "$($Candidate.id)-s$Seed-$Stage"
    $skipBuildThisRun = $SkipBuild -or $script:researchBuildPerformed
    $skipInstallThisRun = $SkipInstall -or $script:researchInstallPerformed
    & $TrainingRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Seed $Seed -Layers $fixed.layers -Tokens $fixed.tokens -BatchSize $fixed.batch_size -Steps $Steps -ResumeStep $ResumeStep -CheckpointInterval ([Math]::Min($CheckpointInterval, $Steps)) -Dimension $Candidate.dimension -FeedForwardDimension $Candidate.feed_forward_dimension -RunId $tag -SkipBuild:$skipBuildThisRun -SkipInstall:$skipInstallThisRun
    if ($LASTEXITCODE -ne 0) { throw "RESEARCH_TRAINING_RUNNER_FAILED: $tag" }
    $script:researchBuildPerformed = $true
    $script:researchInstallPerformed = $true
    $checkpointName = if ($Candidate.dimension -eq 16 -and $Candidate.feed_forward_dimension -eq 32) { "htp-seed$Seed-l19-step$Steps.ckpt" } else { "htp-seed$Seed-l19-t32-d$($Candidate.dimension)-f$($Candidate.feed_forward_dimension)-step$Steps.ckpt" }
    $checkpoint = Join-Path $root (Join-Path 'build\reports\nicopedia-htp-training' $checkpointName)
    $evalValidationChunks = if ($FullCapEval) { 8192 } else { $ValidationChunks }
    $evalDevelopmentChunks = if ($FullCapEval) { 16384 } else { $DevelopmentChunks }
    & $EvalRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Seed $Seed -Layers $fixed.layers -Heads $fixed.heads -Tokens $fixed.tokens -CheckpointStep $Steps -CheckpointPath $checkpoint -ValidationChunks $evalValidationChunks -DevelopmentChunks $evalDevelopmentChunks -Dimension $Candidate.dimension -FeedForwardDimension $Candidate.feed_forward_dimension -RunId "$tag-eval" -SkipBuild:$true -SkipInstall:$true
    if ($LASTEXITCODE -ne 0) { throw "RESEARCH_EVAL_RUNNER_FAILED: $tag" }
    $configTag = if ($Candidate.dimension -eq 16 -and $Candidate.feed_forward_dimension -eq 32) { '' } else { "-t32-d$($Candidate.dimension)-f$($Candidate.feed_forward_dimension)" }
    $trainReport = Join-Path $root "build\reports\nicopedia-htp-training\seed$Seed-l19$configTag-steps$Steps-result.txt"
    $evalReport = Join-Path $root "build\reports\nicopedia-htp-eval\seed$Seed-l19$configTag-step$Steps-htp.txt"
    $train = Get-Map $trainReport; $eval = Get-Map $evalReport
    $health = Require-Health $train 'TRAIN'; $evalHealth = Require-Health $eval 'EVAL'
    $estimate = Get-ModelEstimate $Candidate
    $completed = [double]$train.completed_steps
    if ($completed -le 0) { throw "RESEARCH_TRAINING_COMPLETED_STEPS_INVALID: $tag" }
    $htpExecuteMs = (([double]$train.fused_forward_backward_qnn_us + [double]$train.adam_qnn_us) / 1000.0) / $completed
    [pscustomobject][ordered]@{ candidate = $Candidate.id; stage = $Stage; seed = $Seed; steps = $Steps; dimension = $Candidate.dimension; feed_forward_dimension = $Candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; status_success = $health.status_success -and $evalHealth.status_success; qnn_return_success = $health.qnn_return_success -and $evalHealth.qnn_return_success; tensors_finite = $health.tensors_finite -and $evalHealth.tensors_finite; cpu_fallback = $health.cpu_fallback -or $evalHealth.cpu_fallback; validation_nll = $eval.validation_nll; development_nll = $eval.development_nll; htp_execute_ms = $htpExecuteMs; step_wall_ms = $train.training_step_ms; checkpoint_bytes = (Get-Item -LiteralPath $checkpoint).Length; generation = 'not_run'; failure = '' }
}
function Test-SafeCandidateFailure([string]$Message) {
    return $Message -notmatch '(?i)(FOCUS_TAKEOVER|HEADLESS_ACTIVITY_INVARIANT_FAILED|ACTIVITY_LAUNCHED|ACTIVITY_CREATE|ACTIVITY_RESUME|NON_HEADLESS|THERMAL_ABORT|THERMAL_STATUS_UNAVAILABLE|EMERGENCY|SHUTDOWN|ADB_TRANSPORT|ADB_TIMEOUT|DEVICE_UNRESPONSIVE|UNRESPONSIVE|TIMEOUT|HANG|IDENTITY_MISMATCH|HEADLESS_STATUS_IDENTITY|HEADLESS_HEARTBEAT_STALE|HEADLESS_TIMEOUT|HEADLESS_REPORT_PATH|RUN_ALREADY_ACTIVE|RUN_ID_REUSE|RESULT_MARKER_CLEAR_FAILED|ADB_NO_ONLINE_DEVICE|ADB_STABLE_IDENTITY|PHYSICAL_DEVICE_REQUIRED|PREFLIGHT|PACKAGE_NOT|APK_|QAIRT_|BUILD_ID|FIRMWARE|BATTERY_|TRANSPORT_FAILURE)'
}
function Invoke-GenerationQuality($Candidate, [int]$Seed, [int]$Steps) {
    # This is headless generation only; it neither starts the final test nor
    # exports generated bytes.  The private runner report remains under build/.
    & $GenerationRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Model L19 -Seed $Seed -Tokens $fixed.tokens -Dimension $Candidate.dimension -FeedForwardDimension $Candidate.feed_forward_dimension -CheckpointStep $Steps -Prompt '研究用生成確認' -MaxNewBytes $MaxNewBytes -Mode Greedy -GatePolicy htp-native -RunId "$($Candidate.id)-s$Seed-final-generate" -SkipBuild:$true -SkipInstall:$true
    if ($LASTEXITCODE -ne 0) { throw "RESEARCH_GENERATION_RUNNER_FAILED: $($Candidate.id) seed=$Seed" }
    $tag = if ($Candidate.dimension -eq 16 -and $Candidate.feed_forward_dimension -eq 32) { '' } else { "-t32-d$($Candidate.dimension)-f$($Candidate.feed_forward_dimension)" }
    $path = Join-Path $root "build\reports\nicopedia-htp-generation\seed$Seed-l19$tag-greedy-step$Steps-max$MaxNewBytes-result.txt"
    $map = Get-Map $path
    foreach ($key in @('status','cpu_fallback','qnn_return_code_success','output_tensors_finite','generation_health','generated_byte_count','generated_valid_utf8_bytes','generated_invalid_utf8_bytes')) { if (-not $map.ContainsKey($key)) { throw "RESEARCH_GENERATION_FIELD_MISSING: $key" } }
    return [pscustomobject]@{ status_success = $map.status -eq 'SUCCESS'; qnn_return_success = $map.qnn_return_code_success -eq 'true'; tensors_finite = $map.output_tensors_finite -eq 'true'; cpu_fallback = $map.cpu_fallback -eq 'true'; generation_health = $map.generation_health; generated_byte_count = $map.generated_byte_count; generated_valid_utf8_bytes = $map.generated_valid_utf8_bytes; generated_invalid_utf8_bytes = $map.generated_invalid_utf8_bytes }
}

if ($SelfTest) {
    $anchor = Get-ModelEstimate $candidates[0]
    if ($anchor.parameter_count -ne 48320 -or $anchor.checkpoint_payload_estimate_bytes -ne 3 * $anchor.parameter_bytes_fp32) { throw 'SELFTEST_ESTIMATE' }
    $temp = Join-Path $env:TEMP ('phonelm-dffn-search-' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($temp) | Out-Null
    try { $sample = Join-Path $temp 'health.txt'; "status=SUCCESS`ncpu_fallback=false`nqnn_return_code_success=true`noutput_tensors_finite=true" | Set-Content -LiteralPath $sample -Encoding utf8; $h = Require-Health (Get-Map $sample) 'SELFTEST'; if (-not $h.status_success -or -not $h.qnn_return_success -or -not $h.tensors_finite -or $h.cpu_fallback) { throw 'SELFTEST_HEALTH' } } finally { Remove-Item -LiteralPath $temp -Recurse -Force }
    Write-Host 'run_nicopedia_dffn_search_self_test=PASS'; exit 0
}
if ($ScreenSteps -lt 1 -or $ScreenSteps -gt 8000 -or $ExtendedSteps -lt $ScreenSteps -or $ExtendedSteps -gt 8000 -or $FinalSteps -lt $ExtendedSteps -or $FinalSteps -gt 8000 -or $ExtendedSeeds.Count -lt 2 -or $CheckpointInterval -lt 1 -or $CheckpointInterval -gt 8000 -or $ValidationChunks -lt 1 -or $ValidationChunks -gt 16384 -or $DevelopmentChunks -lt 1 -or $DevelopmentChunks -gt 16384 -or $MaxNewBytes -lt 1 -or $MaxNewBytes -gt 1024 -or
    @($ExtendedSeeds | Where-Object { $_ -lt 1 -or $_ -gt 99999 }).Count -gt 0 -or
    (@($ExtendedSeeds | Sort-Object -Unique).Count -ne $ExtendedSeeds.Count)) { throw 'RESEARCH_CONFIGURATION_INVALID: seeds must be distinct values in 1..99999' }
$plan = Write-Plan
if ($Phase -eq 'Plan') { Write-Host "research_plan=$ReportRoot\experiment-plan.json"; exit 0 }
if (-not $AllowTier3Research) { throw 'TIER3_RESEARCH_PERMISSION_REQUIRED: use -AllowTier3Research for device training' }
if ($SkipBuild -or $SkipInstall) { throw 'RESEARCH_FRESH_BUILD_INSTALL_REQUIRED: do not reuse an unaudited APK' }
Assert-RunnerContract $TrainingRunner 'TRAINING'; Assert-RunnerContract $EvalRunner 'EVAL'; Assert-RunnerContract $GenerationRunner 'GENERATION'
$screenPath = Join-Path $ReportRoot 'screening-summary.csv'
if ($Phase -eq 'Screen') {
    $screen = @(); foreach ($candidate in $candidates) {
        try {
            $row = Invoke-ResearchSegment $candidate 1 $ScreenSteps 'screen'
            $generation = Invoke-GenerationQuality $candidate 1 $ScreenSteps
            $row.generation = $generation.generation_health
            $row.status_success = $row.status_success -and $generation.status_success
            $row.qnn_return_success = $row.qnn_return_success -and $generation.qnn_return_success
            $row.tensors_finite = $row.tensors_finite -and $generation.tensors_finite
            $row.cpu_fallback = $row.cpu_fallback -or $generation.cpu_fallback
            $row | Add-Member -NotePropertyName generated_byte_count -NotePropertyValue $generation.generated_byte_count
            $row | Add-Member -NotePropertyName generated_valid_utf8_bytes -NotePropertyValue $generation.generated_valid_utf8_bytes
            $row | Add-Member -NotePropertyName generated_invalid_utf8_bytes -NotePropertyValue $generation.generated_invalid_utf8_bytes
        } catch {
            if (-not (Test-SafeCandidateFailure $_.Exception.Message)) { throw }
            $estimate = Get-ModelEstimate $candidate
            $row = [pscustomobject][ordered]@{ candidate = $candidate.id; stage = 'screen'; seed = 1; steps = $ScreenSteps; dimension = $candidate.dimension; feed_forward_dimension = $candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; status_success = $false; qnn_return_success = $false; tensors_finite = $false; cpu_fallback = $false; validation_nll = ''; development_nll = ''; htp_execute_ms = ''; step_wall_ms = ''; checkpoint_bytes = ''; generation = 'not_run'; generated_byte_count = ''; generated_valid_utf8_bytes = ''; generated_invalid_utf8_bytes = ''; failure = $_.Exception.Message }
            Write-Warning "screen candidate $($candidate.id) rejected: $($_.Exception.Message)"
        }
        $screen += $row
    }
    $screen | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $screenPath
} elseif (Test-Path -LiteralPath $screenPath -PathType Leaf) {
    $screen = @(Import-Csv -LiteralPath $screenPath)
} else { throw 'RESEARCH_SCREENING_REQUIRED: run -Phase Screen before Extended or Final' }
$anchor = $screen | Where-Object candidate -eq 'anchor-d16-f32' | Select-Object -First 1
if ($null -eq $anchor) { throw 'RESEARCH_ANCHOR_RESULT_MISSING' }
if ([string]$anchor.status_success -ne 'True' -or [string]$anchor.qnn_return_success -ne 'True' -or
    [string]$anchor.tensors_finite -ne 'True' -or [string]$anchor.cpu_fallback -ne 'False' -or
    [string]$anchor.generation -ne 'true') { throw 'RESEARCH_ANCHOR_HEALTH_REJECTED' }
$healthyNonAnchor = @($screen | Where-Object { $_.candidate -ne 'anchor-d16-f32' -and [string]$_.status_success -eq 'True' -and [string]$_.qnn_return_success -eq 'True' -and [string]$_.tensors_finite -eq 'True' -and [string]$_.cpu_fallback -eq 'False' -and [string]$_.generation -eq 'true' -and [double]$_.development_nll -le ([double]$anchor.development_nll + 0.03) -and [double]$_.step_wall_ms -le (2.0 * [double]$anchor.step_wall_ms) -and [int64]$_.checkpoint_bytes -le (4 * [int64]$anchor.checkpoint_bytes) } | Sort-Object {[double]$_.development_nll}, {[double]$_.validation_nll}, {[double]$_.step_wall_ms} | Select-Object -First 2)
$survivors = @($anchor) + $healthyNonAnchor
if ($Phase -eq 'Screen') { Write-Host "research_screening=$ReportRoot\screening-summary.csv survivors=$($survivors.candidate -join ',')"; exit 0 }
$extendedPath = Join-Path $ReportRoot 'extended-summary.csv'
if ($Phase -eq 'Extended') {
    $extended = @(); foreach ($candidateId in $survivors.candidate) { $candidate = $candidates | Where-Object id -eq $candidateId; foreach ($seed in $ExtendedSeeds) { try { $extended += Invoke-ResearchSegment $candidate $seed $ExtendedSteps 'extended' } catch { if (-not (Test-SafeCandidateFailure $_.Exception.Message)) { throw }; $estimate = Get-ModelEstimate $candidate; $extended += [pscustomobject][ordered]@{ candidate = $candidate.id; stage = 'extended'; seed = $seed; steps = $ExtendedSteps; dimension = $candidate.dimension; feed_forward_dimension = $candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; status_success = $false; qnn_return_success = $false; tensors_finite = $false; cpu_fallback = $false; validation_nll = ''; development_nll = ''; htp_execute_ms = ''; step_wall_ms = ''; checkpoint_bytes = ''; generation = 'not_run'; failure = $_.Exception.Message }; Write-Warning "extended candidate $($candidate.id) seed=$seed rejected: $($_.Exception.Message)" } } }
    $extended | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $extendedPath
} elseif (Test-Path -LiteralPath $extendedPath -PathType Leaf) {
    $extended = @(Import-Csv -LiteralPath $extendedPath)
} else { throw 'RESEARCH_EXTENDED_REQUIRED: run -Phase Extended before Final' }
if ($Phase -eq 'Extended') { Write-Host "research_extended=$ReportRoot\extended-summary.csv"; exit 0 }
$means = @($extended | Group-Object candidate | ForEach-Object { $healthy = @($_.Group | Where-Object { [string]$_.status_success -eq 'True' -and [string]$_.qnn_return_success -eq 'True' -and [string]$_.tensors_finite -eq 'True' -and [string]$_.cpu_fallback -eq 'False' }); if ($healthy.Count -eq $ExtendedSeeds.Count) { [pscustomobject]@{ candidate = $_.Name; development_nll = [double](($healthy | Measure-Object -Property development_nll -Average).Average) } } } | Where-Object { $null -ne $_ } | Sort-Object development_nll)
$finalCandidate = $means | Where-Object candidate -ne 'anchor-d16-f32' | Select-Object -First 1
if ($null -eq $finalCandidate) { throw 'RESEARCH_FINAL_CANDIDATE_MISSING' }
$selected = $candidates | Where-Object id -eq $finalCandidate.candidate
$final = @(); foreach ($seed in $ExtendedSeeds) { $row = Invoke-ResearchSegment $selected $seed $FinalSteps 'final' $ExtendedSteps -FullCapEval; $generation = Invoke-GenerationQuality $selected $seed $FinalSteps; $row.generation = $generation.generation_health; $row.status_success = $row.status_success -and $generation.status_success; $row.qnn_return_success = $row.qnn_return_success -and $generation.qnn_return_success; $row.tensors_finite = $row.tensors_finite -and $generation.tensors_finite; $row.cpu_fallback = $row.cpu_fallback -or $generation.cpu_fallback; $row | Add-Member -NotePropertyName generated_byte_count -NotePropertyValue $generation.generated_byte_count; $row | Add-Member -NotePropertyName generated_valid_utf8_bytes -NotePropertyValue $generation.generated_valid_utf8_bytes; $row | Add-Member -NotePropertyName generated_invalid_utf8_bytes -NotePropertyValue $generation.generated_invalid_utf8_bytes; $final += $row }
$final | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $ReportRoot 'final-summary.csv')
Write-Host "research_final=$ReportRoot\final-summary.csv"
