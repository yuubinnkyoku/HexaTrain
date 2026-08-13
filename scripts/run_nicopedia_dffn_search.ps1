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
    [int]$ScreenSteps = 32,
    [int]$ExtendedSteps = 1000,
    [int]$FinalSteps = 4000,
    [int[]]$ExtendedSeeds = @(1, 2, 4),
    [int]$CheckpointInterval = 32,
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
$script:experimentId = [guid]::NewGuid().ToString('N')
$script:sourceCommit = ((git -C $root rev-parse HEAD 2>$null) | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($script:sourceCommit)) { $script:sourceCommit = '<unknown>' }
$script:sourceFingerprint = ''
$script:phaseApkSha256 = ''
$script:phaseAndroidTestApkSha256 = ''

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
    [pscustomobject]@{ id = 'd16-f64'; dimension = 16; feed_forward_dimension = 64; family = 'ffn_only' },
    [pscustomobject]@{ id = 'd24-f48'; dimension = 24; feed_forward_dimension = 48; family = 'intermediate_joint' },
    [pscustomobject]@{ id = 'd32-f32'; dimension = 32; feed_forward_dimension = 32; family = 'dimension_only' }
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
function Get-FiniteReportNumber([hashtable]$Map, [string]$Key, [string]$Kind) {
    if (-not $Map.ContainsKey($Key)) { throw "RESEARCH_${Kind}_FIELD_MISSING: $Key" }
    if ([string]::IsNullOrWhiteSpace([string]$Map[$Key])) { throw "RESEARCH_${Kind}_NUMBER_INVALID: $Key" }
    try { $value = 0.0; if (-not [double]::TryParse([string]$Map[$Key], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value)) { throw 'parse' } } catch { throw "RESEARCH_${Kind}_NUMBER_INVALID: $Key" }
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { throw "RESEARCH_${Kind}_NUMBER_NONFINITE: $Key" }
    return $value
}
function Get-NonNegativeReportNumber([hashtable]$Map, [string]$Key, [string]$Kind) {
    $value = Get-FiniteReportNumber $Map $Key $Kind
    if ($value -lt 0) { throw "RESEARCH_${Kind}_NUMBER_NEGATIVE: $Key" }
    return $value
}
function Assert-ReportIdentity([hashtable]$Map, [string]$Kind, [int]$Dimension, [int]$FeedForwardDimension, [int]$Step, [int]$Seed) {
    $keys = if ($Kind -eq 'TRAIN') { @('model_dimension', 'feed_forward_dimension', 'layers', 'heads', 'batch_size', 'learning_rate', 'cache_context', 'cache_vocabulary', 'seed', 'resume_checkpoint_format') } else { @('model_dimension', 'feed_forward_dimension', 'layers', 'heads', 'checkpoint_step', 'context_tokens', 'vocabulary_size', 'seed', 'checkpoint_format') }
    foreach ($key in $keys) {
        if (-not $Map.ContainsKey($key)) { throw "RESEARCH_${Kind}_IDENTITY_FIELD_MISSING: $key" }
    }
    if ([int]$Map.model_dimension -ne $Dimension -or [int]$Map.feed_forward_dimension -ne $FeedForwardDimension -or
        [int]$Map.layers -ne $fixed.layers -or [int]$Map.heads -ne $fixed.heads -or
        [int]$Map.seed -ne $Seed -or
        ($Kind -eq 'TRAIN' -and ([int]$Map.batch_size -ne $fixed.batch_size -or [int]$Map.cache_context -ne $fixed.tokens -or [int]$Map.cache_vocabulary -ne $fixed.vocabulary -or [Math]::Abs([double]$Map.learning_rate - [double]$fixed.learning_rate) -gt 1.0e-6 -or ($Map.resume_checkpoint_format -ne 'NPRTCKPTV2' -and [int]$Map.resume_from_step -gt 0))) -or
        ($Kind -eq 'EVAL' -and ([string]$Map.checkpoint_format -ne $fixed.checkpoint_format -or [int]$Map.vocabulary_size -ne $fixed.vocabulary))) {
        throw "RESEARCH_${Kind}_FIXED_IDENTITY_MISMATCH"
    }
    if ($Kind -eq 'TRAIN' -and [int]$Map.steps -ne $Step) { throw "RESEARCH_${Kind}_STEP_MISMATCH" }
    if ($Kind -eq 'EVAL' -and [int]$Map.checkpoint_step -ne $Step) { throw "RESEARCH_${Kind}_CHECKPOINT_STEP_MISMATCH" }
}
function Assert-RunnerContract([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "RESEARCH_${Name}_RUNNER_MISSING" }
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "RESEARCH_${Name}_RUNNER_OUTSIDE_REPOSITORY" }
    $source = Get-Content -LiteralPath $Path -Raw
    foreach ($parameter in @('Dimension', 'FeedForwardDimension')) {
        if ($source -notmatch "(?i)\b$parameter\b") { throw "RESEARCH_${Name}_RUNNER_DIMENSION_CONTRACT_MISSING: $parameter" }
    }
}
function Get-CurrentApkSha256 {
    $apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
    if (-not (Test-Path -LiteralPath $apk -PathType Leaf)) { throw 'RESEARCH_APK_PROVENANCE_MISSING' }
    return (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Get-CurrentAndroidTestApkSha256 {
    $apk = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
    if (-not (Test-Path -LiteralPath $apk -PathType Leaf)) { throw 'RESEARCH_TEST_APK_PROVENANCE_MISSING' }
    return (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Set-PhaseApkIdentity {
    $script:phaseApkSha256 = Get-CurrentApkSha256
    $script:phaseAndroidTestApkSha256 = Get-CurrentAndroidTestApkSha256
}
function Assert-PhaseApkIdentity {
    if ([string]::IsNullOrWhiteSpace($script:phaseApkSha256)) { Set-PhaseApkIdentity; return }
    if ((Get-CurrentApkSha256) -ne $script:phaseApkSha256 -or (Get-CurrentAndroidTestApkSha256) -ne $script:phaseAndroidTestApkSha256) {
        throw 'RESEARCH_APK_CHANGED_DURING_PHASE'
    }
}
function Get-ResearchHash([string]$Text) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))).Replace('-', '')).ToLowerInvariant() } finally { $sha.Dispose() }
}
function Get-ResearchSourceFingerprint {
    # Build outputs and private evidence live under build/ and are bound by
    # their own hashes below; excluding them keeps the source fingerprint
    # stable across the mandatory fresh build/install and phase reports.
    $trackedRoots = @('app', 'scripts', 'host_tests', 'gradle', 'build.gradle', 'settings.gradle', 'gradle.properties', 'protocol.json')
    $status = ((git -C $root status --porcelain --untracked-files=all -- $trackedRoots 2>$null) | Out-String)
    $diff = ((git -C $root diff --binary HEAD -- $trackedRoots 2>$null) | Out-String)
    $runnerHashes = foreach ($path in @($TrainingRunner, $EvalRunner, $GenerationRunner)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { "${path}=$((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant())" }
    }
    return Get-ResearchHash ("commit=$script:sourceCommit`nstatus=$status`ndiff=$diff`nrunners=$($runnerHashes -join ';')")
}
function Get-ResearchCacheEvidence {
    $cacheRoot = Join-Path $root 'build\private-data\nicopedia-real-text\caches'
    $result = [ordered]@{}
    foreach ($name in @('train_pilot.bin', 'validation.bin', 'development.bin')) {
        $path = Join-Path $cacheRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "RESEARCH_CACHE_PROVENANCE_MISSING: $name" }
        $result[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}
function Get-PhaseIdentity([string]$PhaseName) {
    switch ($PhaseName) {
        'screening' { return [ordered]@{ steps = $ScreenSteps; seeds = @(1); validation_chunks = $ValidationChunks; development_chunks = $DevelopmentChunks; max_new_bytes = $MaxNewBytes; checkpoint_interval = $CheckpointInterval } }
        'extended' { return [ordered]@{ steps = $ExtendedSteps; seeds = @($ExtendedSeeds); validation_chunks = $ValidationChunks; development_chunks = $DevelopmentChunks; max_new_bytes = $MaxNewBytes; checkpoint_interval = $CheckpointInterval } }
        'final' { return [ordered]@{ steps = $FinalSteps; seeds = @($ExtendedSeeds); validation_chunks = 8192; development_chunks = 16384; max_new_bytes = $MaxNewBytes; checkpoint_interval = $CheckpointInterval } }
        default { throw "RESEARCH_PHASE_INVALID: $PhaseName" }
    }
}
function Write-EvidenceMetadata([string]$PhaseName, [string]$CsvPath) {
    if (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf)) { throw "RESEARCH_${PhaseName}_CSV_MISSING" }
    $metadata = [ordered]@{
        schema = 'PHONELM_NICOPEDIA_DFFN_EVIDENCE_V1'
        phase = $PhaseName
        experiment_id = $script:experimentId
        source_commit = $script:sourceCommit
        source_fingerprint = $script:sourceFingerprint
        qairt_build_id = $ExpectedBuildId
        fixed_identity = $fixed
        phase_identity = Get-PhaseIdentity $PhaseName
        phase_identity_sha256 = Get-ResearchHash ((Get-PhaseIdentity $PhaseName | ConvertTo-Json -Compress -Depth 8))
        cache_evidence = Get-ResearchCacheEvidence
        csv_sha256 = (Get-FileHash -LiteralPath $CsvPath -Algorithm SHA256).Hash.ToLowerInvariant()
        apk_sha256 = Get-CurrentApkSha256
        android_test_apk_sha256 = Get-CurrentAndroidTestApkSha256
        phase_apk_sha256 = $script:phaseApkSha256
        phase_android_test_apk_sha256 = $script:phaseAndroidTestApkSha256
    }
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ReportRoot "$PhaseName-summary.meta.json") -Encoding utf8
}
function Assert-EvidenceMetadata([string]$PhaseName, [string]$CsvPath) {
    $metaPath = Join-Path $ReportRoot "$PhaseName-summary.meta.json"
    if ((-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) -or (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf))) { throw "RESEARCH_${PhaseName}_PROVENANCE_MISSING" }
    $metadata = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
    $phaseHash = Get-ResearchHash ((Get-PhaseIdentity $PhaseName | ConvertTo-Json -Compress -Depth 8))
    $cacheEvidence = Get-ResearchCacheEvidence
    if ($metadata.schema -ne 'PHONELM_NICOPEDIA_DFFN_EVIDENCE_V1' -or $metadata.experiment_id -eq '' -or $metadata.experiment_id -ne $script:experimentId -or
        $metadata.source_commit -ne $script:sourceCommit -or $metadata.source_fingerprint -ne $script:sourceFingerprint -or $metadata.qairt_build_id -ne $ExpectedBuildId -or
        $metadata.phase_identity_sha256 -ne $phaseHash -or $metadata.cache_evidence.'train_pilot.bin' -ne $cacheEvidence.'train_pilot.bin' -or
        $metadata.cache_evidence.'validation.bin' -ne $cacheEvidence.'validation.bin' -or $metadata.cache_evidence.'development.bin' -ne $cacheEvidence.'development.bin' -or
        $metadata.csv_sha256 -ne (Get-FileHash -LiteralPath $CsvPath -Algorithm SHA256).Hash.ToLowerInvariant() -or
        $metadata.apk_sha256 -ne (Get-CurrentApkSha256) -or $metadata.android_test_apk_sha256 -ne (Get-CurrentAndroidTestApkSha256) -or $metadata.phase_apk_sha256 -ne $script:phaseApkSha256 -or $metadata.phase_android_test_apk_sha256 -ne $script:phaseAndroidTestApkSha256) { throw "RESEARCH_${PhaseName}_PROVENANCE_MISMATCH" }
}
function Write-Plan {
    [IO.Directory]::CreateDirectory($ReportRoot) | Out-Null
    $rows = foreach ($candidate in $candidates) {
        $estimate = Get-ModelEstimate $candidate
        [ordered]@{ candidate = $candidate; estimate = $estimate }
    }
$plan = [ordered]@{
        schema = 'PHONELM_NICOPEDIA_DFFN_SEARCH_V1'; experiment_id = $script:experimentId; source_commit = $script:sourceCommit; source_fingerprint = $script:sourceFingerprint; qairt_sdk_root = '<fixed-local-root>'; qairt_build_id = $ExpectedBuildId
        fixed = $fixed; screening = [ordered]@{ steps = $ScreenSteps; seeds = @(1); validation_chunks = $ValidationChunks; development_chunks = $DevelopmentChunks; checkpoint_interval = $CheckpointInterval }
        extension = [ordered]@{ steps = $ExtendedSteps; seeds = $ExtendedSeeds; generation_max_new_bytes = $MaxNewBytes; checkpoint_interval = $CheckpointInterval }
        final = [ordered]@{ steps = $FinalSteps; seeds = $ExtendedSeeds; full_cap_eval_required = $true; resume_semantics = 'NPRTCKPTV2'; checkpoint_interval = $CheckpointInterval }
        cutoff = @('drop on nonzero QNN return', 'drop on nonfinite application-visible tensors', 'drop on CPU fallback', 'drop on failed status', 'drop if development NLL exceeds anchor by > 0.03', 'drop if the screen single-run wall step exceeds anchor by > 2x; report HTP execute ms separately', 'drop if extended median wall step exceeds the matched anchor by > 2x', 'drop if checkpoint exceeds the matched anchor by > 4x', 'require generation health and valid/invalid UTF-8 byte counts no worse than the anchor; short-period-loop is diagnostic only', 'CPU replay is diagnostic only; promotion compares HTP-native reports and does not claim CPU-equivalent quality', 'advance anchor plus at most two healthy non-anchor candidates')
        candidates = $rows
    }
    $plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ReportRoot 'experiment-plan.json') -Encoding utf8
    return $plan
}
function Read-ContinuationPlan {
    $planPath = Join-Path $ReportRoot 'experiment-plan.json'
    if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) { throw 'RESEARCH_PLAN_PROVENANCE_MISSING' }
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    if ($plan.schema -ne 'PHONELM_NICOPEDIA_DFFN_SEARCH_V1' -or $plan.qairt_build_id -ne $ExpectedBuildId -or
        $plan.source_commit -ne $script:sourceCommit -or $plan.source_fingerprint -ne $script:sourceFingerprint -or [string]::IsNullOrWhiteSpace([string]$plan.experiment_id)) {
        throw 'RESEARCH_PLAN_PROVENANCE_MISMATCH'
    }
    $script:experimentId = [string]$plan.experiment_id
    $planScreen = $plan.screening
    $planExtension = $plan.extension
    $planFinal = $plan.final
    $planSeeds = @($planExtension.seeds | ForEach-Object { [int]$_ })
    $candidateIds = @($plan.candidates | ForEach-Object { [string]$_.candidate.id })
    if ([int]$planScreen.steps -ne $ScreenSteps -or [int]$planScreen.validation_chunks -ne $ValidationChunks -or
        [int]$planScreen.development_chunks -ne $DevelopmentChunks -or [int]$planExtension.steps -ne $ExtendedSteps -or
        [int]$planFinal.steps -ne $FinalSteps -or [int]$plan.extension.generation_max_new_bytes -ne $MaxNewBytes -or
        [int]$planScreen.checkpoint_interval -ne $CheckpointInterval -or [int]$planExtension.checkpoint_interval -ne $CheckpointInterval -or
        [int]$planFinal.checkpoint_interval -ne $CheckpointInterval -or ($planSeeds -join ',') -ne (@($ExtendedSeeds | ForEach-Object { [int]$_ }) -join ',') -or
        ($candidateIds -join ',') -ne ($candidates.id -join ',')) {
        throw 'RESEARCH_PLAN_ARGUMENTS_MISMATCH'
    }
    $localFixedJson = $fixed | ConvertTo-Json -Compress -Depth 8
    $planFixedJson = $plan.fixed | ConvertTo-Json -Compress -Depth 8
    if ($planFixedJson -ne $localFixedJson) { throw 'RESEARCH_PLAN_FIXED_IDENTITY_MISMATCH' }
    foreach ($candidate in $candidates) {
        $planned = @($plan.candidates | Where-Object { [string]$_.candidate.id -eq [string]$candidate.id }) | Select-Object -First 1
        if ($null -eq $planned -or [int]$planned.candidate.dimension -ne [int]$candidate.dimension -or
            [int]$planned.candidate.feed_forward_dimension -ne [int]$candidate.feed_forward_dimension -or
            [string]$planned.candidate.family -ne [string]$candidate.family) {
            throw "RESEARCH_PLAN_CANDIDATE_IDENTITY_MISMATCH: $($candidate.id)"
        }
        $estimate = Get-ModelEstimate $candidate
        if ([int64]$planned.estimate.parameter_count -ne [int64]$estimate.parameter_count) {
            throw "RESEARCH_PLAN_PARAMETER_ESTIMATE_MISMATCH: $($candidate.id)"
        }
    }
    return $plan
}
function Invoke-FreshBuildInstall {
    $runId = "dffn-provenance-$($script:experimentId.Substring(0, 12))"
    & $TrainingRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Seed 1 -Layers $fixed.layers -Tokens $fixed.tokens -BatchSize $fixed.batch_size -Steps 32 -CheckpointInterval 32 -Dimension 16 -FeedForwardDimension 32 -RunId $runId -BuildInstallOnly
    if ($LASTEXITCODE -ne 0) { throw 'RESEARCH_FRESH_BUILD_INSTALL_FAILED' }
    $auditPath = Join-Path $ReportRoot "apk-audit-$($script:experimentId).txt"
    & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath (Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk') -ExpectedBuildId $ExpectedBuildId -QairtSdkRoot $QairtSdkRoot | Tee-Object -FilePath $auditPath
    if ($LASTEXITCODE -ne 0 -or -not (Select-String -LiteralPath $auditPath -Pattern '^status=SUCCESS$' -Quiet)) { throw 'RESEARCH_APK_AUDIT_FAILED' }
    $script:researchBuildPerformed = $true
    $script:researchInstallPerformed = $true
    $script:sourceFingerprint = Get-ResearchSourceFingerprint
    Set-PhaseApkIdentity
}
function Invoke-ResearchSegment($Candidate, [int]$Seed, [int]$Steps, [string]$Stage, [int]$ResumeStep = 0, [switch]$FullCapEval) {
    Assert-PhaseApkIdentity
    # Bind every device input directory to this experiment.  Candidate-only
    # run IDs collide with retained private app data from earlier screens and
    # must fail closed rather than being deleted or silently reused.
    $tag = "$($script:experimentId.Substring(0, 8))-$($Candidate.id)-s$Seed-$Stage"
    $skipBuildThisRun = $SkipBuild -or $script:researchBuildPerformed
    $skipInstallThisRun = $SkipInstall -or $script:researchInstallPerformed
    & $TrainingRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Seed $Seed -Layers $fixed.layers -Tokens $fixed.tokens -BatchSize $fixed.batch_size -Steps $Steps -ResumeStep $ResumeStep -CheckpointInterval ([Math]::Min($CheckpointInterval, $Steps)) -Dimension $Candidate.dimension -FeedForwardDimension $Candidate.feed_forward_dimension -RunId $tag -SkipBuild:$skipBuildThisRun -SkipInstall:$skipInstallThisRun
    if ($LASTEXITCODE -ne 0) { throw "RESEARCH_TRAINING_RUNNER_FAILED: $tag" }
    Assert-PhaseApkIdentity
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
    Assert-ReportIdentity $train 'TRAIN' $Candidate.dimension $Candidate.feed_forward_dimension $Steps $Seed
    Assert-ReportIdentity $eval 'EVAL' $Candidate.dimension $Candidate.feed_forward_dimension $Steps $Seed
    [void](Get-NonNegativeReportNumber $eval 'validation_nll' 'EVAL')
    [void](Get-NonNegativeReportNumber $eval 'development_nll' 'EVAL')
    [void](Get-NonNegativeReportNumber $train 'fused_forward_backward_qnn_us' 'TRAIN')
    [void](Get-NonNegativeReportNumber $train 'adam_qnn_us' 'TRAIN')
    [void](Get-NonNegativeReportNumber $train 'training_step_ms' 'TRAIN')
    $initialNll = Get-NonNegativeReportNumber $train 'first_loss' 'TRAIN'
    $finalNll = Get-NonNegativeReportNumber $train 'last_loss' 'TRAIN'
    foreach ($key in @('api_trace_graph_execute_attempt_count', 'api_trace_graph_execute_failure_count', 'activity_create_count', 'activity_resume_count', 'focus_takeover_count')) {
        if (-not $train.ContainsKey($key)) { throw "RESEARCH_TRAIN_FIELD_MISSING: $key" }
    }
    $estimate = Get-ModelEstimate $Candidate
    foreach ($key in @('parameter_element_count', 'resource_estimator_parameter_elements')) {
        if (-not $train.ContainsKey($key)) { throw "RESEARCH_TRAIN_PARAMETER_FIELD_MISSING: $key" }
    }
    if (-not $eval.ContainsKey('checkpoint_parameter_elements')) { throw 'RESEARCH_EVAL_PARAMETER_FIELD_MISSING: checkpoint_parameter_elements' }
    $measuredParameterCount = [int64]$train.parameter_element_count
    $resourceParameterCount = [int64]$train.resource_estimator_parameter_elements
    $checkpointParameterCount = [int64]$eval.checkpoint_parameter_elements
    if ($measuredParameterCount -ne [int64]$estimate.parameter_count -or
        $resourceParameterCount -ne [int64]$estimate.parameter_count -or
        $checkpointParameterCount -ne [int64]$estimate.parameter_count) {
        throw "RESEARCH_PARAMETER_COUNT_MISMATCH: expected=$($estimate.parameter_count) training=$measuredParameterCount resource=$resourceParameterCount checkpoint=$checkpointParameterCount"
    }
    $completed = Get-NonNegativeReportNumber $train 'completed_steps' 'TRAIN'
    if ([Math]::Floor($completed) -ne $completed -or $completed -ne $Steps) { throw "RESEARCH_TRAINING_COMPLETED_STEPS_INVALID: $tag" }
    $htpExecuteMs = (([double]$train.fused_forward_backward_qnn_us + [double]$train.adam_qnn_us) / 1000.0) / $completed
    $deltaNll = $initialNll - $finalNll
    $wallSeconds = ([double]$train.training_step_ms * $completed) / 1000.0
    $htpSeconds = ($htpExecuteMs * $completed) / 1000.0
    [pscustomobject][ordered]@{ candidate = $Candidate.id; stage = $Stage; seed = $Seed; steps = $Steps; dimension = $Candidate.dimension; feed_forward_dimension = $Candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; measured_parameter_count = $measuredParameterCount; checkpoint_parameter_count = $checkpointParameterCount; parameter_count_verified = $true; status_success = $health.status_success -and $evalHealth.status_success; qnn_return_success = $health.qnn_return_success -and $evalHealth.qnn_return_success; tensors_finite = $health.tensors_finite -and $evalHealth.tensors_finite; cpu_fallback = $health.cpu_fallback -or $evalHealth.cpu_fallback; initial_training_nll = $initialNll; final_training_nll = $finalNll; delta_training_nll = $deltaNll; relative_training_nll_reduction = $(if ($initialNll -gt 0) { $deltaNll / $initialNll } else { 0.0 }); delta_nll_per_wall_second = $(if ($wallSeconds -gt 0) { $deltaNll / $wallSeconds } else { 0.0 }); delta_nll_per_htp_second = $(if ($htpSeconds -gt 0) { $deltaNll / $htpSeconds } else { 0.0 }); validation_nll = $eval.validation_nll; development_nll = $eval.development_nll; htp_execute_ms = $htpExecuteMs; step_wall_ms = $train.training_step_ms; qnn_execute_count = $train.api_trace_graph_execute_attempt_count; qnn_failure_count = $train.api_trace_graph_execute_failure_count; checkpoint_bytes = (Get-Item -LiteralPath $checkpoint).Length; activity_create_count = $train.activity_create_count; activity_resume_count = $train.activity_resume_count; focus_takeover_count = $train.focus_takeover_count; generation = 'not_run'; failure = '' }
}
function Test-SafeCandidateFailure([string]$Message) {
    return $Message -notmatch '(?i)(RESEARCH_|FOCUS_TAKEOVER|HEADLESS_ACTIVITY_INVARIANT_FAILED|ACTIVITY_LAUNCHED|ACTIVITY_CREATE|ACTIVITY_RESUME|NON_HEADLESS|THERMAL_ABORT|THERMAL_STATUS_UNAVAILABLE|EMERGENCY|SHUTDOWN|ADB_TRANSPORT|ADB_TIMEOUT|ADB_COMMAND_FAILURE|ADB_UNAVAILABLE|ADB_INSTALL|INSTALL_FAILED|DEVICE_UNRESPONSIVE|UNRESPONSIVE|TIMEOUT|HANG|IDENTITY_MISMATCH|HEADLESS_STATUS_IDENTITY|HEADLESS_HEARTBEAT_STALE|HEADLESS_TIMEOUT|HEADLESS_REPORT_PATH|RUN_ALREADY_ACTIVE|RUN_ID_REUSE|RESULT_MARKER_CLEAR_FAILED|ADB_NO_ONLINE_DEVICE|ADB_STABLE_IDENTITY|PHYSICAL_DEVICE_REQUIRED|PREFLIGHT|PACKAGE_NOT|APK_|APK\s+BUILD\s+FAILED|QAIRT_|BUILD_ID|FIRMWARE|BATTERY_|TRANSPORT_FAILURE|INSTRUMENTATION_EXIT_FAILURE|INSTRUMENTATION\s+BUILD\s+FAILED|BUILD_FAILED|BUILD\s+FAILED|GRADLE|RUNNER_FAILED|CHECKPOINT_PULL|HOST_CHECKPOINT_EVALUATOR|VALIDATION_CACHE_MISSING|PRIVATE_CACHE_MISSING|DEVELOPMENT_CACHE_MISSING|CHECKPOINT_MISSING|CHECKPOINT_INTERVAL_MISSING|CHECKPOINT_RESUME_FORMAT_INVALID|RUN_ID_INVALID|POLL_CONFIGURATION_INVALID)'
}
function Invoke-GenerationQuality($Candidate, [int]$Seed, [int]$Steps) {
    # This is headless generation only; it neither starts the final test nor
    # exports generated bytes.  The private runner report remains under build/.
    & $GenerationRunner -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -Model L19 -Seed $Seed -Tokens $fixed.tokens -Dimension $Candidate.dimension -FeedForwardDimension $Candidate.feed_forward_dimension -CheckpointStep $Steps -Prompt '研究用生成確認' -MaxNewBytes $MaxNewBytes -Mode Greedy -GatePolicy htp-native -RunId "$($script:experimentId.Substring(0, 8))-$($Candidate.id)-s$Seed-generate" -SkipBuild:$true -SkipInstall:$true
    if ($LASTEXITCODE -ne 0) { throw "RESEARCH_GENERATION_RUNNER_FAILED: $($Candidate.id) seed=$Seed" }
    $tag = if ($Candidate.dimension -eq 16 -and $Candidate.feed_forward_dimension -eq 32) { '' } else { "-t32-d$($Candidate.dimension)-f$($Candidate.feed_forward_dimension)" }
    $path = Join-Path $root "build\reports\nicopedia-htp-generation\seed$Seed-l19$tag-greedy-step$Steps-max$MaxNewBytes-result.txt"
    $map = Get-Map $path
    foreach ($key in @('status','cpu_fallback','qnn_return_code_success','output_tensors_finite','generation_health','generated_byte_count','generated_valid_utf8_bytes','generated_invalid_utf8_bytes')) { if (-not $map.ContainsKey($key)) { throw "RESEARCH_GENERATION_FIELD_MISSING: $key" } }
    foreach ($key in @('generated_byte_count','generated_valid_utf8_bytes','generated_invalid_utf8_bytes')) { if ([string]::IsNullOrWhiteSpace([string]$map[$key]) -or $map[$key] -notmatch '^\d+$') { throw "RESEARCH_GENERATION_NUMBER_INVALID: $key" } }
    return [pscustomobject]@{ status_success = $map.status -eq 'SUCCESS'; qnn_return_success = $map.qnn_return_code_success -eq 'true'; tensors_finite = $map.output_tensors_finite -eq 'true'; cpu_fallback = $map.cpu_fallback -eq 'true'; generation_health = $map.generation_health; generated_byte_count = $map.generated_byte_count; generated_valid_utf8_bytes = $map.generated_valid_utf8_bytes; generated_invalid_utf8_bytes = $map.generated_invalid_utf8_bytes }
}

if ($SelfTest) {
    if (($candidates.id -join ',') -ne 'anchor-d16-f32,d16-f64,d24-f48,d32-f32' -or $ScreenSteps -ne 32 -or $CheckpointInterval -ne 32) { throw 'SELFTEST_SCREEN_CONTRACT' }
    $anchor = Get-ModelEstimate $candidates[0]
    if ($anchor.parameter_count -ne 48320 -or $anchor.checkpoint_payload_estimate_bytes -ne 3 * $anchor.parameter_bytes_fp32) { throw 'SELFTEST_ESTIMATE' }
    $temp = Join-Path $env:TEMP ('phonelm-dffn-search-' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($temp) | Out-Null
    try { $sample = Join-Path $temp 'health.txt'; "status=SUCCESS`ncpu_fallback=false`nqnn_return_code_success=true`noutput_tensors_finite=true" | Set-Content -LiteralPath $sample -Encoding utf8; $h = Require-Health (Get-Map $sample) 'SELFTEST'; if (-not $h.status_success -or -not $h.qnn_return_success -or -not $h.tensors_finite -or $h.cpu_fallback) { throw 'SELFTEST_HEALTH' } } finally { Remove-Item -LiteralPath $temp -Recurse -Force }
    foreach ($unsafe in @('ADB_COMMAND_FAILURE: shell failed', 'APK build failed', 'HEADLESS_ACTIVITY_INVARIANT_FAILED: activity_create_count=1', 'THERMAL_ABORT')) { if (Test-SafeCandidateFailure $unsafe) { throw "SELFTEST_SAFETY_CLASSIFIER: $unsafe" } }
    if (Test-SafeCandidateFailure 'RESEARCH_PARAMETER_COUNT_MISMATCH: expected=1') { throw 'SELFTEST_CANDIDATE_CLASSIFIER' }
    $badNumber = @{ validation_nll = '' }; try { [void](Get-FiniteReportNumber $badNumber 'validation_nll' 'SELFTEST'); throw 'SELFTEST_EMPTY_NUMBER' } catch { if ($_.Exception.Message -eq 'SELFTEST_EMPTY_NUMBER') { throw } }
    $negativeNumber = @{ training_step_ms = '-1' }; try { [void](Get-NonNegativeReportNumber $negativeNumber 'training_step_ms' 'SELFTEST'); throw 'SELFTEST_NEGATIVE_NUMBER' } catch { if ($_.Exception.Message -eq 'SELFTEST_NEGATIVE_NUMBER') { throw } }
    if ((([Math]::Floor(1000 / [double]320) + 1) * 320) -ne 1280) { throw 'SELFTEST_RESUME_INTERVAL' }
    Write-Host 'run_nicopedia_dffn_search_self_test=PASS'; exit 0
}
if ($script:sourceCommit -eq '<unknown>') { throw 'RESEARCH_SOURCE_PROVENANCE_UNAVAILABLE' }
$script:sourceFingerprint = Get-ResearchSourceFingerprint
if ($ScreenSteps -lt 1 -or $ScreenSteps -gt 8000 -or $ExtendedSteps -lt $ScreenSteps -or $ExtendedSteps -gt 8000 -or $FinalSteps -le $ExtendedSteps -or $FinalSteps -gt 8000 -or $ExtendedSeeds.Count -lt 2 -or $CheckpointInterval -lt 1 -or $CheckpointInterval -gt 8000 -or $CheckpointInterval -gt $ScreenSteps -or $ValidationChunks -lt 1 -or $ValidationChunks -gt 16384 -or $DevelopmentChunks -lt 1 -or $DevelopmentChunks -gt 16384 -or $MaxNewBytes -lt 1 -or $MaxNewBytes -gt 1024 -or
    @($ExtendedSeeds | Where-Object { $_ -lt 1 -or $_ -gt 99999 }).Count -gt 0 -or
    (@($ExtendedSeeds | Sort-Object -Unique).Count -ne $ExtendedSeeds.Count)) { throw 'RESEARCH_CONFIGURATION_INVALID: seeds must be distinct values in 1..99999' }
$screenPath = Join-Path $ReportRoot 'screening-summary.csv'
$planPath = Join-Path $ReportRoot 'experiment-plan.json'
if ($Phase -eq 'Plan' -and (@($planPath, $screenPath, (Join-Path $ReportRoot 'screening-summary.meta.json'), (Join-Path $ReportRoot 'extended-summary.csv'), (Join-Path $ReportRoot 'extended-summary.meta.json'), (Join-Path $ReportRoot 'final-summary.csv'), (Join-Path $ReportRoot 'final-summary.meta.json') | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }).Count -gt 0)) {
    throw 'RESEARCH_REPORT_ROOT_REUSE: choose a fresh ReportRoot for a new plan'
}
if ($Phase -eq 'Screen' -and ((Test-Path -LiteralPath $screenPath -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'screening-summary.meta.json') -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'extended-summary.csv') -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'extended-summary.meta.json') -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'final-summary.csv') -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'final-summary.meta.json') -PathType Leaf))) {
    throw 'RESEARCH_REPORT_ROOT_REUSE: choose a fresh ReportRoot for a new screen'
}
$plan = if ($Phase -eq 'Plan') { Write-Plan } elseif (($Phase -eq 'Screen' -or $Phase -eq 'Extended' -or $Phase -eq 'Final') -and (Test-Path -LiteralPath $planPath -PathType Leaf)) { Read-ContinuationPlan } elseif ($Phase -eq 'Screen') { Write-Plan } else { throw 'RESEARCH_PLAN_REQUIRED: run -Phase Plan first' }
if ($Phase -eq 'Plan') { Write-Host "research_plan=$ReportRoot\experiment-plan.json"; exit 0 }
if (-not $AllowTier3Research) { throw 'TIER3_RESEARCH_PERMISSION_REQUIRED: use -AllowTier3Research for device training' }
if ($SkipBuild -or $SkipInstall) { throw 'RESEARCH_FRESH_BUILD_INSTALL_REQUIRED: do not reuse an unaudited APK' }
Assert-RunnerContract $TrainingRunner 'TRAINING'; Assert-RunnerContract $EvalRunner 'EVAL'; Assert-RunnerContract $GenerationRunner 'GENERATION'
if ($Phase -eq 'Screen' -or $Phase -eq 'Extended' -or $Phase -eq 'Final') {
    Invoke-FreshBuildInstall
}
if ($Phase -eq 'Extended' -or $Phase -eq 'Final') {
    Assert-EvidenceMetadata 'screening' $screenPath
}
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
            $row = [pscustomobject][ordered]@{ candidate = $candidate.id; stage = 'screen'; seed = 1; steps = $ScreenSteps; dimension = $candidate.dimension; feed_forward_dimension = $candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; measured_parameter_count = ''; checkpoint_parameter_count = ''; parameter_count_verified = $false; status_success = $false; qnn_return_success = $false; tensors_finite = $false; cpu_fallback = $false; initial_training_nll = ''; final_training_nll = ''; delta_training_nll = ''; relative_training_nll_reduction = ''; delta_nll_per_wall_second = ''; delta_nll_per_htp_second = ''; validation_nll = ''; development_nll = ''; htp_execute_ms = ''; step_wall_ms = ''; qnn_execute_count = ''; qnn_failure_count = ''; checkpoint_bytes = ''; activity_create_count = ''; activity_resume_count = ''; focus_takeover_count = ''; generation = 'not_run'; generated_byte_count = ''; generated_valid_utf8_bytes = ''; generated_invalid_utf8_bytes = ''; failure = $_.Exception.Message }
            Write-Warning "screen candidate $($candidate.id) rejected: $($_.Exception.Message)"
        }
        $screen += $row
    }
    $screen | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $screenPath
    Write-EvidenceMetadata 'screening' $screenPath
} elseif (Test-Path -LiteralPath $screenPath -PathType Leaf) {
    $screen = @(Import-Csv -LiteralPath $screenPath)
} else { throw 'RESEARCH_SCREENING_REQUIRED: run -Phase Screen before Extended or Final' }
$anchor = $screen | Where-Object candidate -eq 'anchor-d16-f32' | Select-Object -First 1
if ($null -eq $anchor) { throw 'RESEARCH_ANCHOR_RESULT_MISSING' }
if ([string]$anchor.status_success -ne 'True' -or [string]$anchor.qnn_return_success -ne 'True' -or
    [string]$anchor.tensors_finite -ne 'True' -or [string]$anchor.cpu_fallback -ne 'False' -or
    [string]$anchor.generation -ne 'true') { throw 'RESEARCH_ANCHOR_HEALTH_REJECTED' }
$healthyNonAnchor = @($screen | Where-Object { $_.candidate -ne 'anchor-d16-f32' -and [string]$_.parameter_count_verified -eq 'True' -and [string]$_.status_success -eq 'True' -and [string]$_.qnn_return_success -eq 'True' -and [string]$_.tensors_finite -eq 'True' -and [string]$_.cpu_fallback -eq 'False' -and [string]$_.generation -eq 'true' -and [int64]$_.generated_valid_utf8_bytes -ge [int64]$anchor.generated_valid_utf8_bytes -and [int64]$_.generated_invalid_utf8_bytes -le [int64]$anchor.generated_invalid_utf8_bytes -and [double]$_.development_nll -le ([double]$anchor.development_nll + 0.03) -and [double]$_.step_wall_ms -le (2.0 * [double]$anchor.step_wall_ms) -and [int64]$_.checkpoint_bytes -le (4 * [int64]$anchor.checkpoint_bytes) } | Sort-Object {[double]$_.development_nll}, {[double]$_.validation_nll}, {[double]$_.step_wall_ms} | Select-Object -First 2)
$survivors = @($anchor) + $healthyNonAnchor
if ($Phase -eq 'Screen') { Write-Host "research_screening=$ReportRoot\screening-summary.csv survivors=$($survivors.candidate -join ',')"; exit 0 }
$extendedPath = Join-Path $ReportRoot 'extended-summary.csv'
if ($Phase -eq 'Extended') {
    Assert-PhaseApkIdentity
    if ((Test-Path -LiteralPath $extendedPath -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'extended-summary.meta.json') -PathType Leaf)) { throw 'RESEARCH_REPORT_ROOT_REUSE: extended evidence already exists' }
    $extended = @(); foreach ($candidateId in $survivors.candidate) { $candidate = $candidates | Where-Object id -eq $candidateId; foreach ($seed in $ExtendedSeeds) { try { $extended += Invoke-ResearchSegment $candidate $seed $ExtendedSteps 'extended' } catch { if (-not (Test-SafeCandidateFailure $_.Exception.Message)) { throw }; $estimate = Get-ModelEstimate $candidate; $extended += [pscustomobject][ordered]@{ candidate = $candidate.id; stage = 'extended'; seed = $seed; steps = $ExtendedSteps; dimension = $candidate.dimension; feed_forward_dimension = $candidate.feed_forward_dimension; parameter_count = $estimate.parameter_count; measured_parameter_count = ''; checkpoint_parameter_count = ''; parameter_count_verified = $false; status_success = $false; qnn_return_success = $false; tensors_finite = $false; cpu_fallback = $false; initial_training_nll = ''; final_training_nll = ''; delta_training_nll = ''; relative_training_nll_reduction = ''; delta_nll_per_wall_second = ''; delta_nll_per_htp_second = ''; validation_nll = ''; development_nll = ''; htp_execute_ms = ''; step_wall_ms = ''; qnn_execute_count = ''; qnn_failure_count = ''; checkpoint_bytes = ''; activity_create_count = ''; activity_resume_count = ''; focus_takeover_count = ''; generation = 'not_run'; failure = $_.Exception.Message }; Write-Warning "extended candidate $($candidate.id) seed=$seed rejected: $($_.Exception.Message)" } } }
    $extended | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $extendedPath
    Write-EvidenceMetadata 'extended' $extendedPath
} elseif (Test-Path -LiteralPath $extendedPath -PathType Leaf) {
    $extended = @(Import-Csv -LiteralPath $extendedPath)
} else { throw 'RESEARCH_EXTENDED_REQUIRED: run -Phase Extended before Final' }
if ($Phase -eq 'Final') { Assert-EvidenceMetadata 'extended' $extendedPath }
if ($Phase -eq 'Extended') { Write-Host "research_extended=$ReportRoot\extended-summary.csv"; exit 0 }
$means = @($extended | Group-Object candidate | ForEach-Object {
    $healthy = @($_.Group | Where-Object { [string]$_.parameter_count_verified -eq 'True' -and [string]$_.status_success -eq 'True' -and [string]$_.qnn_return_success -eq 'True' -and [string]$_.tensors_finite -eq 'True' -and [string]$_.cpu_fallback -eq 'False' })
    if ($healthy.Count -eq $ExtendedSeeds.Count) {
        $wall = @($healthy | ForEach-Object { [double]$_.step_wall_ms } | Sort-Object)
        $checkpoint = @($healthy | ForEach-Object { [int64]$_.checkpoint_bytes } | Sort-Object)
        [pscustomobject]@{ candidate = $_.Name; development_nll = [double](($healthy | Measure-Object -Property development_nll -Average).Average); validation_nll = [double](($healthy | Measure-Object -Property validation_nll -Average).Average); median_step_wall_ms = [double]$wall[[int][Math]::Floor(($wall.Count - 1) / 2)]; median_checkpoint_bytes = [int64]$checkpoint[[int][Math]::Floor(($checkpoint.Count - 1) / 2)] }
    }
} | Where-Object { $null -ne $_ } | Sort-Object development_nll)
$anchorExtended = $means | Where-Object candidate -eq 'anchor-d16-f32' | Select-Object -First 1
if ($null -eq $anchorExtended) { throw 'RESEARCH_EXTENDED_ANCHOR_HEALTH_REJECTED' }
$finalCandidate = $means | Where-Object { $_.candidate -ne 'anchor-d16-f32' -and $_.development_nll -le ($anchorExtended.development_nll + 0.03) -and $_.validation_nll -le ($anchorExtended.validation_nll + 0.03) -and $_.median_step_wall_ms -le (2.0 * $anchorExtended.median_step_wall_ms) -and $_.median_checkpoint_bytes -le (4 * $anchorExtended.median_checkpoint_bytes) } | Select-Object -First 1
if ($null -eq $finalCandidate) { throw 'RESEARCH_FINAL_CANDIDATE_MISSING' }
$selected = $candidates | Where-Object id -eq $finalCandidate.candidate
$finalPath = Join-Path $ReportRoot 'final-summary.csv'
if ((Test-Path -LiteralPath $finalPath -PathType Leaf) -or (Test-Path -LiteralPath (Join-Path $ReportRoot 'final-summary.meta.json') -PathType Leaf)) { throw 'RESEARCH_REPORT_ROOT_REUSE: final evidence already exists' }
$finalConfigs = @($candidates | Where-Object id -eq 'anchor-d16-f32') + @($selected)
$final = @(); foreach ($finalCandidateConfig in $finalConfigs) { foreach ($seed in $ExtendedSeeds) { $row = Invoke-ResearchSegment $finalCandidateConfig $seed $FinalSteps 'final' $ExtendedSteps -FullCapEval; $generation = Invoke-GenerationQuality $finalCandidateConfig $seed $FinalSteps; $row.generation = $generation.generation_health; $row.status_success = $row.status_success -and $generation.status_success; $row.qnn_return_success = $row.qnn_return_success -and $generation.qnn_return_success; $row.tensors_finite = $row.tensors_finite -and $generation.tensors_finite; $row.cpu_fallback = $row.cpu_fallback -or $generation.cpu_fallback; $row | Add-Member -NotePropertyName generated_byte_count -NotePropertyValue $generation.generated_byte_count; $row | Add-Member -NotePropertyName generated_valid_utf8_bytes -NotePropertyValue $generation.generated_valid_utf8_bytes; $row | Add-Member -NotePropertyName generated_invalid_utf8_bytes -NotePropertyValue $generation.generated_invalid_utf8_bytes; $final += $row } }
Assert-PhaseApkIdentity
$finalAnchorBySeed = @{}
foreach ($row in @($final | Where-Object candidate -eq 'anchor-d16-f32')) { $finalAnchorBySeed[[int]$row.seed] = $row }
if ($finalAnchorBySeed.Count -ne $ExtendedSeeds.Count) { throw 'RESEARCH_FINAL_ANCHOR_CONTROL_MISSING' }
foreach ($row in $final) {
    $baseline = $finalAnchorBySeed[[int]$row.seed]
    $qualityGuard = [string]$row.generation -eq 'true' -and [int64]$row.generated_valid_utf8_bytes -ge [int64]$baseline.generated_valid_utf8_bytes -and [int64]$row.generated_invalid_utf8_bytes -le [int64]$baseline.generated_invalid_utf8_bytes
    $performanceGuard = $row.candidate -eq 'anchor-d16-f32' -or ([double]$row.development_nll -le ([double]$baseline.development_nll + 0.03) -and [double]$row.validation_nll -le ([double]$baseline.validation_nll + 0.03) -and [double]$row.step_wall_ms -le (2.0 * [double]$baseline.step_wall_ms) -and [int64]$row.checkpoint_bytes -le (4 * [int64]$baseline.checkpoint_bytes))
    $row | Add-Member -NotePropertyName generation_quality_guard -NotePropertyValue $qualityGuard
    $row | Add-Member -NotePropertyName final_comparison_guard -NotePropertyValue $performanceGuard
    $row.status_success = $row.status_success -and $qualityGuard -and $performanceGuard
}
$selectedFinalRows = @($final | Where-Object candidate -eq $selected.id)
$anchorFinalRows = @($final | Where-Object candidate -eq 'anchor-d16-f32')
if ($selectedFinalRows.Count -ne $ExtendedSeeds.Count -or
     @($selectedFinalRows | Where-Object { [string]$_.status_success -ne 'True' }).Count -gt 0 -or
     @($selectedFinalRows | Where-Object { [string]$_.qnn_return_success -ne 'True' -or [string]$_.tensors_finite -ne 'True' -or [string]$_.cpu_fallback -ne 'False' }).Count -gt 0 -or
     $anchorFinalRows.Count -ne $ExtendedSeeds.Count -or
     @($anchorFinalRows | Where-Object { [string]$_.status_success -ne 'True' -or [string]$_.qnn_return_success -ne 'True' -or [string]$_.tensors_finite -ne 'True' -or [string]$_.cpu_fallback -ne 'False' }).Count -gt 0) {
    throw 'RESEARCH_FINAL_PROMOTION_REJECTED: matched anchor/candidate final gate failed'
}
$final | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $finalPath
Write-EvidenceMetadata 'final' $finalPath
Write-Host "research_final=$ReportRoot\final-summary.csv"
