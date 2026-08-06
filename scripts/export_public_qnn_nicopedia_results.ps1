param(
    [switch]$SelfTest,
    [string]$CorpusRoot = "build/private-data/nicopedia-real-text",
    [string]$ReportRoot = "build/private-diagnostics/nicopedia-real-text-goal",
    [string]$OutputRoot = "docs/results/qnn-nicopedia-real-text-pilot-2026-08",
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$canonicalOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot "docs/results/qnn-nicopedia-real-text-pilot-2026-08"))
$allowList = @(
    "README.md", "manifest.json", "corpus-configuration.csv", "corpus-inventory.csv",
    "cleaning-summary.csv", "split-summary.csv", "dedup-summary.csv", "subset-summary.csv",
    "tokenizer-candidates.csv", "tokenizer-configuration.csv", "model-configurations.csv",
    "runtime-budget.csv", "training-summary.csv", "checkpoint-selection.csv",
    "seed-stability.csv", "teacher-forced-summary.csv", "free-running-summary.csv",
    "paired-prefix-summary.csv", "depth-control.csv", "hypothesis-outcomes.csv",
    "diagnosis.csv", "limitations.csv", "next-step-candidates.csv"
)
$sourceFiles = @(
    "host_tests/nicopedia_real_text_pilot.cpp", "scripts/nicopedia_real_text_pipeline.py",
    "scripts/run_nicopedia_real_text_pilot.ps1", "scripts/export_public_qnn_nicopedia_results.ps1",
    "scripts/run_host_tests.ps1", "scripts/verify_local.ps1",
    "app/src/main/cpp/tiny_language_model_cpu.cpp", "app/src/main/cpp/tiny_language_model_cpu.h",
    "app/src/main/cpp/qnn/qnn_runtime.h", "app/src/main/cpp/transformer_resource_estimator.h",
    "docs/qnn-nicopedia-real-text-pilot.md", "docs/results/README.md"
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

function Get-CombinedNormalizedSha256([string[]]$Paths) {
    $builder = [Text.StringBuilder]::new()
    foreach ($path in ($Paths | Sort-Object)) {
        $name = Split-Path -Leaf $path
        [void]$builder.Append($name).Append("`0").Append((Get-NormalizedSha256 $path)).Append("`n")
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes($builder.ToString())
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Import-StrictCsv([string]$Path, [string]$ExpectedHeader) {
    $lines = @(Get-Content -LiteralPath $Path)
    if ($lines.Count -lt 2 -or $lines[0] -ne $ExpectedHeader) { throw "Private CSV schema mismatch" }
    $expectedColumns = ($ExpectedHeader -split ',').Count
    foreach ($line in $lines | Select-Object -Skip 1) {
        if (-not $line -or ($line -split ',').Count -ne $expectedColumns) { throw "Private CSV row-width mismatch" }
    }
    return @($lines | ConvertFrom-Csv)
}

function Get-FiniteDouble([object]$Value, [string]$Label) {
    try { $number = [double]$Value } catch { throw "$Label is not numeric" }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$Label is non-finite" }
    return $number
}

function Assert-Near([object]$Actual, [object]$Expected, [double]$Tolerance, [string]$Label) {
    $actualNumber = Get-FiniteDouble $Actual $Label
    $expectedNumber = Get-FiniteDouble $Expected $Label
    if ([Math]::Abs($actualNumber - $expectedNumber) -gt $Tolerance) { throw "$Label mismatch" }
}

function Assert-Range([object]$Value, [double]$Minimum, [double]$Maximum, [string]$Label) {
    $number = Get-FiniteDouble $Value $Label
    if ($number -lt $Minimum -or $number -gt $Maximum) { throw "$Label is outside its valid range" }
}

function Assert-SafeText([string]$Text, [string]$Label) {
    $patterns = @(
        '(?i)[a-z]:(?:\\|/)', '(?m)(?:^|[\s="])\\\\[^\\\s]+\\',
        '(?m)(?<![:/A-Za-z0-9_])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+',
        '(?i)build(?:\\|/)private', '(?i)\.(?:ckpt|bin|model|vocab|npy|npz|pt|pth)\b',
        '(?i)BEGIN [A-Z ]*PRIVATE KEY', '\bAKIA[0-9A-Z]{16}\b',
        '(?i)adb\s+-s\s+\S+', '\b\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}\b',
        '(?i)pg_(?:id|title|view_title|yomi)|txt_text',
        '(?i)raw[_ -]?(?:text|token|logit|tensor|checkpoint|parameter)\s*[:=]'
    )
    foreach ($pattern in $patterns) {
        if ($Text -match $pattern) { throw "Unsafe public payload in $Label (pattern rejected)" }
    }
}

function Assert-RequiredPublicationTerms([string]$Text) {
    foreach ($required in @(
        'Dwango Co., Ltd.', 'National Institute of Informatics (NII)',
        'IDR Dataset Provision Service', 'non-commercial research',
        'not redistributed', 'result notification'
    )) {
        if (-not $Text.Contains($required, [StringComparison]::Ordinal)) {
            throw "Required publication term is missing"
        }
    }
}

function Assert-PublicBundle([string]$Root) {
    $entries = @(Get-ChildItem -LiteralPath $Root -Force)
    if (@($entries | Where-Object { $_.PSIsContainer }).Count -ne 0) { throw "Public bundle contains a subdirectory" }
    $actual = @($entries | Where-Object { -not $_.PSIsContainer } | ForEach-Object Name | Sort-Object)
    $expected = @($allowList | Sort-Object)
    if (($actual -join "`n") -ne ($expected -join "`n")) { throw "Public bundle exact allow-list mismatch" }
    foreach ($file in $entries) {
        if ($file.Length -gt 5MB) { throw "Public file is unexpectedly large: $($file.Name)" }
        Assert-SafeText (Get-Content -LiteralPath $file.FullName -Raw) $file.Name
    }
    Assert-RequiredPublicationTerms (Get-Content -LiteralPath (Join-Path $Root "README.md") -Raw)
    $manifest = Get-Content -LiteralPath (Join-Path $Root "manifest.json") -Raw | ConvertFrom-Json
    $manifestNames = @($manifest.files | ForEach-Object path | Sort-Object)
    $expectedManifest = @($allowList | Where-Object { $_ -ne "manifest.json" } | Sort-Object)
    if (($manifestNames -join "`n") -ne ($expectedManifest -join "`n")) { throw "Manifest file set mismatch" }
    foreach ($entry in $manifest.files) {
        if ([IO.Path]::IsPathRooted($entry.path) -or $entry.path.Contains("..") -or $entry.path.Contains('\')) { throw "Invalid manifest path" }
        if ((Get-NormalizedSha256 (Join-Path $Root $entry.path)) -ne $entry.sha256_normalized_lf) { throw "Manifest hash mismatch: $($entry.path)" }
    }
}

function Write-CsvFile([string]$Path, [object[]]$Rows) {
    if ($Rows.Count -eq 0) { throw "Refusing to publish empty CSV: $Path" }
    $Rows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

if ($SelfTest) {
    $temporary = Join-Path $repoRoot "build/export-selftest/nicopedia-public"
    New-Item -ItemType Directory -Force -Path $temporary | Out-Null
    $safe = "aggregate_only=true`nfinal_test_opened=false`n"
    Assert-SafeText $safe "safe-fixture"
    $windowsPathFixture = "local=C:" + [char]92 + "Users" + [char]92 + "private" + [char]92 + "file.csv"
    $posixPathFixture = "payload=" + [char]47 + "home" + [char]47 + "private" + [char]47 + "data.bin"
    $uncPathFixture = "payload=" + [char]92 + [char]92 + "server" + [char]92 + "share" + [char]92 + "file.csv"
    $privateRelativeFixture = "payload=build" + [char]92 + "private" + [char]92 + "report.csv"
    foreach ($unsafe in @(
        $windowsPathFixture, $posixPathFixture, $uncPathFixture, $privateRelativeFixture,
        "checkpoint=weights.ckpt", "adb -s device shell", "pg_id=1", "raw_token=12"
    )) {
        $rejected = $false
        try { Assert-SafeText $unsafe "negative-fixture" } catch { $rejected = $true }
        if (-not $rejected) { throw "Unsafe fixture was not rejected" }
    }
    $termsRejected = $false
    try { Assert-RequiredPublicationTerms "aggregate only" } catch { $termsRejected = $true }
    if (-not $termsRejected) { throw "Missing attribution fixture was not rejected" }
    $probe = Join-Path $temporary "probe.txt"
    [IO.File]::WriteAllText($probe, "a`r`nb`n", [Text.UTF8Encoding]::new($false))
    $first = Get-NormalizedSha256 $probe
    [IO.File]::WriteAllText($probe, "a`nb`n", [Text.UTF8Encoding]::new($false))
    if ($first -ne (Get-NormalizedSha256 $probe)) { throw "Normalized hash is not line-ending stable" }
    Remove-Item -LiteralPath $probe
    Write-Host "nicopedia_public_exporter_self_test=PASS"
    exit 0
}

$resolvedCorpus = Resolve-Under $CorpusRoot (Join-Path $repoRoot "build") "CorpusRoot"
$resolvedReports = Resolve-Under $ReportRoot (Join-Path $repoRoot "build") "ReportRoot"
$resolvedOutput = Resolve-Under $OutputRoot (Join-Path $repoRoot "docs/results") "OutputRoot"
if ($resolvedOutput -ne $canonicalOutput) { throw "OutputRoot must be the canonical Nicopedia public bundle path" }
if (-not $SourceCommit) {
    $SourceCommit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve source commit" }
}
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { throw "SourceCommit must be a full Git SHA" }
$currentHead = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $SourceCommit -ne $currentHead) { throw "SourceCommit must equal the current HEAD" }
git diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw "Tracked worktree must be clean before public export" }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw "Index must be clean before public export" }
foreach ($relative in $sourceFiles) {
    git ls-files --error-unmatch -- $relative 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "A manifest source is not tracked" }
    git diff --quiet HEAD -- $relative
    if ($LASTEXITCODE -ne 0) { throw "A manifest source differs from SourceCommit" }
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Python is required to verify private evidence" }
& $python.Source (Join-Path $repoRoot "scripts/nicopedia_real_text_pipeline.py") `
    --verify-evidence --private-root $resolvedCorpus
if ($LASTEXITCODE -ne 0) { throw "Private evidence verification failed" }

$corpus = Get-Content -LiteralPath (Join-Path $resolvedCorpus "reports/public-corpus-aggregate.json") -Raw | ConvertFrom-Json
$sourceManifest = Get-Content -LiteralPath (Join-Path $resolvedCorpus "source-manifest.json") -Raw | ConvertFrom-Json
$evidence = Get-Content -LiteralPath (Join-Path $resolvedCorpus "reports/evidence-provenance.json") -Raw | ConvertFrom-Json
$decision = Get-Content -LiteralPath (Join-Path $resolvedReports "decisions/decision-005.json") -Raw | ConvertFrom-Json
$remediation = Get-Content -LiteralPath (Join-Path $resolvedReports "decisions/decision-006.json") -Raw | ConvertFrom-Json
$protocol = Get-Content -LiteralPath (Join-Path $resolvedReports "decisions/decision-004.json") -Raw | ConvertFrom-Json
$hypotheses = Get-Content -LiteralPath (Join-Path $resolvedReports "decisions/decision-003.json") -Raw | ConvertFrom-Json
$benchmarks = @()
$benchmarks += Import-Csv (Join-Path $resolvedReports "smoke/benchmark-l6.csv")
$benchmarks += Import-Csv (Join-Path $resolvedReports "smoke/benchmark-l19.csv")
$runSummaries = @()
$trajectories = @()
$teacher = @()
$stability = @()
$paired = @()
$checkpointEvidence = @()
$runSummaryHeader = "seed,layers,steps,batch_size,checkpoint_interval,initial_parameter_hash,train_cache_hash,validation_cache_hash,training_order_hash,best_step,best_validation_nll,final_validation_nll,last_train_nll,finite,runtime_seconds,peak_working_set_bytes,checkpoint_write_seconds,checkpoint_parameter_hash,development_used_for_selection,final_test_used"
$trajectoryHeader = "seed,layers,step,train_nll,validation_nll,validation_perplexity,validation_top1,validation_top5,validation_mean_rank,validation_margin,validation_tokens,finite,parameter_hash"
foreach ($layers in @(6, 19)) {
    foreach ($seed in @(1, 2, 4)) {
        $runDirectory = Join-Path $resolvedReports "formal/l$layers/seed-$seed"
        $runSummaries += Import-StrictCsv (Join-Path $runDirectory "run-summary.csv") $runSummaryHeader
        $trajectories += Import-StrictCsv (Join-Path $runDirectory "training-trajectory.csv") $trajectoryHeader
    }
    $comparison = Join-Path $resolvedReports "formal/l$layers/comparison"
    $checkpointEvidence += Import-StrictCsv (Join-Path $comparison "checkpoint-provenance.csv") "seed,layers,step,parameter_hash,finite"
    $teacher += Import-Csv (Join-Path $comparison "development-teacher-forced.csv")
    $stability += Import-Csv (Join-Path $comparison "seed-stability.csv")
    $paired += Import-Csv (Join-Path $comparison "paired-prefix.csv")
}
if ($runSummaries.Count -ne 6 -or $teacher.Count -ne 6 -or $stability.Count -ne 2 -or
    $paired.Count -ne 6 -or $checkpointEvidence.Count -ne 6) {
    throw "Private aggregate row count mismatch"
}

if ($protocol.schema -ne "NICOPEDIA_REAL_TEXT_FORMAL_BUDGET_V1" -or
    $protocol.status -ne "LOCKED_AFTER_SMOKE_BEFORE_FORMAL_RESULTS" -or
    $hypotheses.schema -ne "NICOPEDIA_REAL_TEXT_HYPOTHESES_V1" -or
    $hypotheses.status -ne "LOCKED_BEFORE_MODEL_RESULTS" -or
    $decision.schema -ne "NICOPEDIA_REAL_TEXT_FORMAL_DECISION_V1" -or
    $decision.status -ne "LOCKED_AFTER_DEVELOPMENT_EVALUATION") {
    throw "Private decision lock mismatch"
}
if (($protocol.seeds -join ',') -ne '1,2,4' -or
    (($protocol.models | ForEach-Object layers) -join ',') -ne '6,19' -or
    [int]$protocol.context_tokens -ne 32 -or [int]$protocol.steps -ne 1000 -or
    [int]$protocol.batch_chunks -ne 8 -or [int]$protocol.checkpoint_interval -ne 100 -or
    [int]$protocol.validation_chunks_per_checkpoint -ne 256 -or
    [int]$protocol.development_chunks_per_selected_checkpoint -ne 512 -or
    [int]$protocol.training_order_seed -ne 20260806 -or
    [int]$protocol.target_tokens_per_run -ne 256000 -or [int]$protocol.formal_training_runs -ne 6 -or
    $protocol.tokenizer -ne 'UTF8_BYTE_V256' -or $protocol.final_test -ne 'unopened' -or
    $corpus.selected_tokenizer -ne 'utf8_byte' -or [int]$corpus.context_tokens -ne 32 -or
    -not [bool]$corpus.source_read_only_verified) {
    throw "Locked protocol configuration mismatch"
}
Assert-Near $protocol.optimizer.learning_rate 0.003 1e-15 "learning rate"
Assert-Near $protocol.optimizer.beta1 0.9 1e-15 "Adam beta1"
Assert-Near $protocol.optimizer.beta2 0.999 1e-15 "Adam beta2"
Assert-Near $protocol.optimizer.epsilon 1e-8 1e-20 "Adam epsilon"
foreach ($model in $protocol.models) {
    $expectedParameters = if ([int]$model.layers -eq 6) { 20864 } elseif ([int]$model.layers -eq 19) { 48320 } else { throw "Unexpected model depth" }
    if ([int]$model.dimension -ne 16 -or [int]$model.ffn -ne 32 -or [int]$model.heads -ne 2 -or
        [int]$model.parameters -ne $expectedParameters) { throw "Locked model configuration mismatch" }
}
$actualCategories = @($corpus.article_type_counts.psobject.Properties.Name | Sort-Object)
if (($actualCategories -join ',') -ne 'a,i,l,o,v') { throw "Article category allow-list mismatch" }
if ($decision.classification -ne 'PERFORMANCE_INSTABILITY_NOT_REPRODUCED_GENERATION_VARIATION_PARTIALLY_REPRODUCED' -or
    $decision.evidence_strength -ne 'OBSERVATION_WITH_MECHANISM_CANDIDATE' -or
    $decision.paired_prefix -ne 'exploratory_only' -or [int]$decision.additional_training_controls -ne 0 -or
    [int]$decision.evaluation_regenerations -ne 1 -or [bool]$decision.final_test_opened -or
    [bool]$decision.synthetic_final_holdout_opened) {
    throw "Locked decision outcome mismatch"
}
if ($remediation.schema -ne 'NICOPEDIA_REAL_TEXT_REVIEW_REMEDIATION_V1' -or
    $remediation.status -ne 'LOCKED_BEFORE_PUBLICATION' -or [bool]$remediation.training_rerun -or
    [bool]$remediation.evaluation_rerun -or [bool]$remediation.numerical_path_changed -or
    [int]$remediation.formal_training_runs_added -ne 0 -or [int]$remediation.source_full_scans_added -ne 0 -or
    [bool]$remediation.final_test_evaluation -or [bool]$remediation.synthetic_final_holdout_evaluation) {
    throw "Review remediation provenance mismatch"
}
if ($evidence.schema -ne 'NICOPEDIA_REAL_TEXT_PRIVATE_EVIDENCE_V1' -or
    $evidence.source_file_aggregate_sha256 -ne $sourceManifest.aggregate_sha256 -or
    $evidence.corpus_aggregate_hash -ne $corpus.aggregate_hash -or
    [bool]$evidence.final_test_cache_present) {
    throw "Source/corpus evidence binding mismatch"
}

$cacheDirectory = Join-Path $resolvedCorpus "caches"
foreach ($cacheName in @('train_pilot', 'validation', 'development')) {
    $cachePath = Join-Path $cacheDirectory "$cacheName.bin"
    $identity = $corpus.cache_identities.psobject.Properties[$cacheName].Value
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf) -or -not $identity) { throw "Required private cache missing" }
    $actualSha256 = (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($identity.sha256 -ne "sha256:$actualSha256") { throw "Private cache SHA-256 mismatch" }
    $evidenceIdentity = $evidence.cache_identities.psobject.Properties[$cacheName].Value
    if ($evidenceIdentity.sha256 -ne $identity.sha256 -or $evidenceIdentity.chunks -ne $identity.chunks -or
        [int]$evidenceIdentity.context -ne 32 -or [int]$evidenceIdentity.vocabulary -ne 256 -or
        $evidenceIdentity.fnv1a64 -notmatch '^fnv1a64:[0-9a-f]{16}$') {
        throw "Private cache provenance mismatch"
    }
}
if (@(Get-ChildItem -LiteralPath $cacheDirectory -File | Where-Object Name -Match 'final').Count -ne 0) {
    throw "Final-test token cache must not exist"
}

$expectedRunKeys = @('6/1', '6/2', '6/4', '19/1', '19/2', '19/4')
$actualRunKeys = @($runSummaries | ForEach-Object { "$($_.layers)/$($_.seed)" } | Sort-Object)
if (($actualRunKeys -join ',') -ne (($expectedRunKeys | Sort-Object) -join ',')) { throw "Formal run identity mismatch" }
if (@($runSummaries | Select-Object -ExpandProperty train_cache_hash -Unique).Count -ne 1 -or
    @($runSummaries | Select-Object -ExpandProperty validation_cache_hash -Unique).Count -ne 1 -or
    @($runSummaries | Select-Object -ExpandProperty training_order_hash -Unique).Count -ne 1) {
    throw "Formal cache/order identity mismatch"
}
if (($runSummaries | Select-Object -First 1).train_cache_hash -ne $evidence.cache_identities.train_pilot.fnv1a64 -or
    ($runSummaries | Select-Object -First 1).validation_cache_hash -ne $evidence.cache_identities.validation.fnv1a64 -or
    ($runSummaries | Select-Object -First 1).training_order_hash -ne $evidence.training_order.fnv1a64 -or
    [int]$evidence.training_order.seed -ne [int]$protocol.training_order_seed -or
    [int]$evidence.training_order.steps -ne [int]$protocol.steps -or
    [int]$evidence.training_order.batch_chunks -ne [int]$protocol.batch_chunks) {
    throw "Formal run cache hash does not match evidence"
}
foreach ($run in $runSummaries) {
    if ([int]$run.steps -ne [int]$protocol.steps -or [int]$run.batch_size -ne [int]$protocol.batch_chunks -or
        [int]$run.checkpoint_interval -ne [int]$protocol.checkpoint_interval -or
        $run.finite -ne 'true' -or $run.development_used_for_selection -ne 'false' -or $run.final_test_used -ne 'false') {
        throw "Formal run safety/config contract mismatch"
    }
    foreach ($hash in @($run.initial_parameter_hash, $run.train_cache_hash, $run.validation_cache_hash,
                         $run.training_order_hash, $run.checkpoint_parameter_hash)) {
        if ($hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "Formal identity hash format mismatch" }
    }
    foreach ($metric in @($run.best_validation_nll, $run.final_validation_nll, $run.last_train_nll,
                           $run.runtime_seconds, $run.checkpoint_write_seconds)) {
        [void](Get-FiniteDouble $metric "formal run metric")
    }
    Assert-Range $run.best_validation_nll 0 ([double]::MaxValue) "best validation NLL"
    Assert-Range $run.last_train_nll 0 ([double]::MaxValue) "last train NLL"
    $runTrajectory = @($trajectories | Where-Object { $_.seed -eq $run.seed -and $_.layers -eq $run.layers } | Sort-Object { [int]$_.step })
    $expectedSteps = @(100..1000 | Where-Object { $_ % 100 -eq 0 })
    if ($runTrajectory.Count -ne $expectedSteps.Count -or
        (($runTrajectory | ForEach-Object { [int]$_.step }) -join ',') -ne ($expectedSteps -join ',')) {
        throw "Training trajectory cadence mismatch"
    }
    foreach ($point in $runTrajectory) {
        if ($point.finite -ne 'true' -or [int]$point.validation_tokens -ne 8192 -or
            $point.parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw "Training trajectory safety mismatch" }
        foreach ($metric in @($point.train_nll, $point.validation_nll, $point.validation_perplexity,
                               $point.validation_top1, $point.validation_top5, $point.validation_mean_rank,
                               $point.validation_margin)) { [void](Get-FiniteDouble $metric "trajectory metric") }
        Assert-Range $point.train_nll 0 ([double]::MaxValue) "trajectory train NLL"
        Assert-Range $point.validation_nll 0 ([double]::MaxValue) "trajectory validation NLL"
        Assert-Range $point.validation_perplexity 1 ([double]::MaxValue) "trajectory perplexity"
        Assert-Range $point.validation_top1 0 1 "trajectory top1"
        Assert-Range $point.validation_top5 0 1 "trajectory top5"
        Assert-Range $point.validation_mean_rank 1 256 "trajectory mean rank"
    }
    $selected = $runTrajectory[0]
    foreach ($point in $runTrajectory | Select-Object -Skip 1) {
        $candidateNll = Get-FiniteDouble $point.validation_nll "candidate validation NLL"
        $selectedNll = Get-FiniteDouble $selected.validation_nll "selected validation NLL"
        if ($candidateNll -lt $selectedNll - 1e-7 -or
            ([Math]::Abs($candidateNll - $selectedNll) -le 1e-7 -and [int]$point.step -lt [int]$selected.step)) {
            $selected = $point
        }
    }
    if ([int]$selected.step -ne [int]$run.best_step -or $selected.parameter_hash -ne $run.checkpoint_parameter_hash) {
        throw "Selected checkpoint identity mismatch"
    }
    Assert-Near $selected.validation_nll $run.best_validation_nll 1e-9 "selected validation NLL"
    Assert-Near $runTrajectory[-1].validation_nll $run.final_validation_nll 1e-9 "final validation NLL"
}

$teacherKeys = @($teacher | ForEach-Object { "$($_.layers)/$($_.seed)" } | Sort-Object)
$pairedKeys = @($paired | ForEach-Object { "$($_.layers)/$($_.seed)" } | Sort-Object)
$checkpointKeys = @($checkpointEvidence | ForEach-Object { "$($_.layers)/$($_.seed)" } | Sort-Object)
if (($teacherKeys -join ',') -ne (($expectedRunKeys | Sort-Object) -join ',') -or
    ($pairedKeys -join ',') -ne (($expectedRunKeys | Sort-Object) -join ',') -or
    ($checkpointKeys -join ',') -ne (($expectedRunKeys | Sort-Object) -join ',')) { throw "Evaluation identity mismatch" }
foreach ($checkpoint in $checkpointEvidence) {
    $run = $runSummaries | Where-Object { $_.seed -eq $checkpoint.seed -and $_.layers -eq $checkpoint.layers }
    if ($checkpoint.finite -ne 'true' -or [int]$checkpoint.step -ne [int]$run.best_step -or
        $checkpoint.parameter_hash -ne $run.checkpoint_parameter_hash) {
        throw "Evaluated checkpoint provenance mismatch"
    }
}
foreach ($row in $teacher) {
    $run = $runSummaries | Where-Object { $_.seed -eq $row.seed -and $_.layers -eq $row.layers }
    if ($row.finite -ne 'true' -or $row.used_for_selection -ne 'false' -or [int]$row.tokens -ne 16384 -or
        [int]$row.selected_step -ne [int]$run.best_step) { throw "Development evaluation safety mismatch" }
    foreach ($metric in @($row.nll, $row.perplexity, $row.top1, $row.top5, $row.mean_rank, $row.mean_margin)) {
        [void](Get-FiniteDouble $metric "development metric")
    }
    Assert-Range $row.nll 0 ([double]::MaxValue) "development NLL"
    Assert-Range $row.perplexity 1 ([double]::MaxValue) "development perplexity"
    Assert-Range $row.top1 0 1 "development top1"
    Assert-Range $row.top5 0 1 "development top5"
    Assert-Range $row.mean_rank 1 256 "development mean rank"
}
foreach ($row in $stability) {
    if ($row.seeds -ne '1;2;4' -or $row.final_test_used -ne 'false') { throw "Seed stability safety mismatch" }
    foreach ($property in @('development_nll_min','development_nll_max','development_nll_range','teacher_argmax_agreement',
                             'teacher_pairwise_js','free_argmax_agreement','free_pairwise_js','mean_first_divergence_position',
                             'repeat_rate','short_loop_rate','invalid_utf8_generation_rate','generation_completion_rate')) {
        [void](Get-FiniteDouble $row.$property "seed stability metric")
    }
    foreach ($property in @('teacher_argmax_agreement','free_argmax_agreement','repeat_rate','short_loop_rate',
                             'invalid_utf8_generation_rate','generation_completion_rate')) {
        Assert-Range $row.$property 0 1 "seed stability rate"
    }
    Assert-Range $row.teacher_pairwise_js 0 ([Math]::Log(2)) "teacher JS"
    Assert-Range $row.free_pairwise_js 0 ([Math]::Log(2)) "free JS"
    Assert-Range $row.mean_first_divergence_position 0 16 "first divergence position"
    if ([int]$row.aligned_teacher_tokens -ne 4096 -or [int]$row.free_positions -ne 256) {
        throw "Seed stability evaluation size mismatch"
    }
}
foreach ($row in $paired) {
    if ($row.final_test_used -ne 'false' -or $row.interpretation -ne 'exploratory_semantic_invariance_not_assumed' -or
        [int]$row.pairs -ne 128 -or [int]$row.suffix_bytes -ne 8) { throw "Paired-prefix safety mismatch" }
    foreach ($property in @('mean_absolute_logit_difference','argmax_flip_rate','mean_absolute_nll_difference',
                             'mean_far_attention_mass_difference','identity_maximum_logit_difference')) {
        [void](Get-FiniteDouble $row.$property "paired-prefix metric")
    }
    Assert-Range $row.mean_absolute_logit_difference 0 ([double]::MaxValue) "paired logit difference"
    Assert-Range $row.argmax_flip_rate 0 1 "paired argmax flip rate"
    Assert-Range $row.mean_absolute_nll_difference 0 ([double]::MaxValue) "paired NLL difference"
    Assert-Range $row.mean_far_attention_mass_difference 0 1 "paired Attention difference"
    Assert-Near $row.identity_maximum_logit_difference 0 0 "paired-prefix identity control"
}

foreach ($layers in @(6, 19)) {
    $layerRuns = @($runSummaries | Where-Object { [int]$_.layers -eq $layers })
    $layerTeacher = @($teacher | Where-Object { [int]$_.layers -eq $layers })
    $layerStability = @($stability | Where-Object { [int]$_.layers -eq $layers })[0]
    $validationValues = @($layerRuns | ForEach-Object { [double]$_.best_validation_nll })
    $developmentValues = @($layerTeacher | ForEach-Object { [double]$_.nll })
    $validationRange = ($validationValues | Measure-Object -Maximum).Maximum - ($validationValues | Measure-Object -Minimum).Minimum
    $developmentRange = ($developmentValues | Measure-Object -Maximum).Maximum - ($developmentValues | Measure-Object -Minimum).Minimum
    $validationProperty = "l${layers}_validation_nll_range"
    $developmentProperty = "l${layers}_development_nll_range"
    $freeAgreementProperty = "l${layers}_free_argmax_agreement"
    $teacherJsProperty = "l${layers}_teacher_pairwise_js"
    $freeJsProperty = "l${layers}_free_pairwise_js"
    Assert-Near $decision.primary_findings.$validationProperty $validationRange 1e-9 "decision validation range"
    Assert-Near $decision.primary_findings.$developmentProperty $developmentRange 1e-6 "decision development range"
    Assert-Near $layerStability.development_nll_range $developmentRange 1e-5 "stability development range"
    Assert-Near $decision.primary_findings.$freeAgreementProperty $layerStability.free_argmax_agreement 1e-9 "decision free agreement"
    Assert-Near $decision.primary_findings.$teacherJsProperty $layerStability.teacher_pairwise_js 1e-9 "decision teacher JS"
    Assert-Near $decision.primary_findings.$freeJsProperty $layerStability.free_pairwise_js 1e-9 "decision free JS"
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$resolvedOutputAfterCreate = Resolve-Under $resolvedOutput (Join-Path $repoRoot "docs/results") "OutputRoot"
if ($resolvedOutputAfterCreate -ne $resolvedOutput) { throw "OutputRoot identity changed after creation" }
Get-ChildItem -LiteralPath $resolvedOutput -Force -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.PSIsContainer) { throw "Canonical output contains a subdirectory" }
    if ($_.Name -notin $allowList) { throw "Canonical output contains an unexpected file" }
    Remove-Item -LiteralPath $_.FullName
}

$corpusConfiguration = @([pscustomobject]@{
    dataset = "Nicopedia data"; version = "2024-11-25"; snapshot = "articles displayed at the end of 2024-01"
    format = "UTF-8 MySQL CSV with backslash escape and embedded LF"; article_types = "a;i;l;o;v"
    normalization = "NFKC"; markup = "HTML text extraction; block boundaries preserved; script/style dropped"
    min_clean_utf8_bytes = 96; max_clean_utf8_bytes = 1048576
    split = "article-level stable SHA-256: train 90%, validation 5%, development 4%, final test 1%"
    dedupe = "exact SHA-256 of cleaned text before split"; final_test_opened = "false"
})
$corpusInventory = @([pscustomobject]@{
    source_files = $sourceManifest.file_count; source_bytes = $sourceManifest.total_bytes
    source_aggregate_sha256 = $sourceManifest.aggregate_sha256; total_records = $corpus.body_records
    records_with_body = $corpus.body_with_text_records; usable_articles = $corpus.usable_records
    raw_body_utf8_bytes = $corpus.raw_utf8_bytes; cleaned_utf8_bytes = $corpus.clean_utf8_bytes
    parse_errors = $corpus.parse_error_count; source_read_only_verified = $corpus.source_read_only_verified
})
$cleaning = @()
foreach ($property in $corpus.exclusions.psobject.Properties) {
    $cleaning += [pscustomobject]@{ operation = $property.Name; excluded_articles = $property.Value; purpose = switch ($property.Name) {
        "too_short" { "remove insufficient text after markup extraction" }
        "too_long" { "bound pathological records" }
        "empty_or_markup_only" { "remove records with no usable text" }
        "duplicate" { "prevent exact normalized-text leakage" }
    }; information_loss_risk = if ($property.Name -eq "duplicate") { "one canonical copy retained" } else { "content excluded by preregistered boundary" } }
}
$splitRows = foreach ($name in @("train", "validation", "development", "final_test")) {
    [pscustomobject]@{ split = $name; articles = $corpus.split_counts.$name; cleaned_utf8_bytes = $corpus.split_clean_utf8_bytes.$name
        selection_use = switch ($name) { "train" { "training and tokenizer comparison" }; "validation" { "checkpoint selection only" }; "development" { "one locked evaluation" }; "final_test" { "cleaning/dedupe/split aggregates only; no token cache or model evaluation" } } }
}
$dedupRows = @([pscustomobject]@{ exact_cleaned_text_duplicates_excluded = $corpus.exact_duplicate_excluded; cross_split_exact_duplicates = 0; method = "SHA-256 exact cleaned text before article split" })
$subsetRows = foreach ($entry in $corpus.cache_identities.psobject.Properties) {
    [pscustomobject]@{ subset = $entry.Name; articles = $entry.Value.articles; cleaned_utf8_bytes = $entry.Value.clean_utf8_bytes
        chunks = $entry.Value.chunks; target_tokens = $entry.Value.target_tokens }
}
$tokenizerRows = foreach ($candidate in $corpus.tokenizer_candidates) {
    [pscustomobject]@{ candidate = $candidate.candidate; vocabulary = $candidate.vocabulary
        mean_tokens_per_character = $candidate.mean_tokens_per_character; p95_tokens_per_character = $candidate.p95_tokens_per_character
        unknown_rate = $candidate.unknown_rate; round_trip = $candidate.round_trip; train_only = $candidate.train_only
        l6_parameters = $candidate.l6_parameter_count; l19_parameters = $candidate.l19_parameter_count; protocol_hash = $candidate.protocol_hash }
}
$tokenizerConfiguration = @([pscustomobject]@{ selected = $corpus.selected_tokenizer; vocabulary = 256; context_tokens = $corpus.context_tokens
    unknown_rate = 0; training_required = "false"; selection_time = "before model outcomes"; reason = $corpus.selected_tokenizer_reason })
$modelRows = foreach ($model in $protocol.models) {
    [pscustomobject]@{ model = $model.name; layers = $model.layers; context = $protocol.context_tokens; vocabulary = 256
        dimension = $model.dimension; ffn = $model.ffn; heads = $model.heads; parameters = $model.parameters
        optimizer = "Adam"; learning_rate = $protocol.optimizer.learning_rate; steps = $protocol.steps; batch_chunks = $protocol.batch_chunks }
}
$runtimeRows = @()
foreach ($benchmark in $benchmarks) {
    $runtimeRows += [pscustomobject]@{ phase = "smoke"; layers = $benchmark.layers; measured_steps = $benchmark.measured_steps
        step_seconds_batch1 = $benchmark.step_seconds; estimated_100_steps_seconds = $benchmark.estimated_100_steps_seconds
        estimated_320_steps_seconds = $benchmark.estimated_320_steps_seconds; peak_working_set_bytes = $benchmark.peak_working_set_bytes
        checkpoint_write_seconds = $benchmark.checkpoint_write_seconds }
}
$runtimeRows += [pscustomobject]@{ phase = "corpus_prepare"; layers = "n/a"; measured_steps = "n/a"; step_seconds_batch1 = "n/a"
    estimated_100_steps_seconds = "n/a"; estimated_320_steps_seconds = "n/a"; peak_working_set_bytes = "not captured"
    checkpoint_write_seconds = "n/a"; processing_seconds = $corpus.processing_seconds; processing_mib_per_second = $corpus.processing_mib_per_second }
$trainingRows = foreach ($run in $runSummaries) {
    [pscustomobject]@{ seed = $run.seed; layers = $run.layers; steps = $run.steps; batch_chunks = $run.batch_size
        target_tokens = [int]$run.steps * [int]$run.batch_size * 32; selected_step = $run.best_step; validation_nll = $run.best_validation_nll
        last_train_nll = $run.last_train_nll; finite = $run.finite; runtime_seconds = $run.runtime_seconds
        peak_working_set_bytes = $run.peak_working_set_bytes; final_test_used = $run.final_test_used }
}
$checkpointRows = foreach ($run in $runSummaries) {
    [pscustomobject]@{ seed = $run.seed; layers = $run.layers; selection_partition = "validation"; selection_metric = "token NLL"
        selected_step = $run.best_step; selected_validation_nll = $run.best_validation_nll; final_step_validation_nll = $run.final_validation_nll
        tie_break = "earlier step within 1e-7"; development_used_for_selection = $run.development_used_for_selection; final_test_used = $run.final_test_used }
}
$teacherRows = foreach ($row in $teacher) {
    [pscustomobject]@{ seed = $row.seed; layers = $row.layers; selected_step = $row.selected_step; development_nll = $row.nll
        perplexity = $row.perplexity; top1 = $row.top1; top5 = $row.top5; mean_rank = $row.mean_rank; mean_margin = $row.mean_margin
        tokens = $row.tokens; finite = $row.finite; used_for_selection = $row.used_for_selection }
}
$stabilityRows = foreach ($row in $stability) {
    $validationRangeProperty = "l$($row.layers)_validation_nll_range"
    [pscustomobject]@{ layers = $row.layers; seeds = $row.seeds; validation_nll_range = $decision.primary_findings.$validationRangeProperty
        development_nll_min = $row.development_nll_min; development_nll_max = $row.development_nll_max; development_nll_range = $row.development_nll_range
        teacher_argmax_agreement = $row.teacher_argmax_agreement; teacher_pairwise_js = $row.teacher_pairwise_js
        free_argmax_agreement = $row.free_argmax_agreement; free_pairwise_js = $row.free_pairwise_js; final_test_used = $row.final_test_used }
}
$freeRows = foreach ($row in $stability) {
    [pscustomobject]@{ layers = $row.layers; seeds = $row.seeds; prompts = 16; generation_steps = 16
        argmax_agreement = $row.free_argmax_agreement; pairwise_js = $row.free_pairwise_js
        mean_first_divergence_position = $row.mean_first_divergence_position; repeat_rate = $row.repeat_rate
        short_loop_rate = $row.short_loop_rate; invalid_utf8_generation_rate = $row.invalid_utf8_generation_rate
        generation_completion_rate = $row.generation_completion_rate; generated_text_published = "false" }
}
$pairedRows = foreach ($row in $paired) {
    [pscustomobject]@{ seed = $row.seed; layers = $row.layers; suffix_bytes = $row.suffix_bytes; pairs = $row.pairs
        train_suffix_min_count = $row.eligibility_train_count; train_target_concentration = $row.eligibility_target_concentration
        mean_absolute_logit_difference = $row.mean_absolute_logit_difference; argmax_flip_rate = $row.argmax_flip_rate
        mean_absolute_nll_difference = $row.mean_absolute_nll_difference; mean_far_attention_mass_difference = $row.mean_far_attention_mass_difference
        identity_maximum_logit_difference = $row.identity_maximum_logit_difference; interpretation = $row.interpretation }
}
$depthRows = @([pscustomobject]@{
    comparison = "L6_vs_L19"; l6_parameters = 20864; l19_parameters = 48320
    l6_development_nll_range = $decision.primary_findings.l6_development_nll_range
    l19_development_nll_range = $decision.primary_findings.l19_development_nll_range
    l6_free_argmax_agreement = $decision.primary_findings.l6_free_argmax_agreement
    l19_free_argmax_agreement = $decision.primary_findings.l19_free_argmax_agreement
    finding = "L19 slightly improves teacher-forced NLL but amplifies greedy generation divergence"
})
$hypothesisRows = foreach ($hypothesis in $hypotheses.hypotheses) {
    [pscustomobject]@{ id = $hypothesis.id; claim = $hypothesis.claim; prediction = $hypothesis.prediction
        test = $hypothesis.test; negative_control = $hypothesis.negative_control; outcome = $decision.hypothesis_outcomes.($hypothesis.id) }
}
$diagnosisRows = @(
    [pscustomobject]@{ question = "seed performance stability"; finding = "large held-out NLL instability was not reproduced"; strength = "observation" },
    [pscustomobject]@{ question = "generation behavior"; finding = "seed-dependent greedy divergence was partially reproduced and stronger at L19"; strength = "observation" },
    [pscustomobject]@{ question = "synthetic homogeneous-TRAIN mechanism"; finding = "not reproduced as a causal claim on natural text"; strength = "not supported" },
    [pscustomobject]@{ question = "most supported explanation"; finding = "small ranking differences amplified by free-running feedback"; strength = "mechanism candidate" }
)
$limitationRows = @(
    [pscustomobject]@{ limitation = "three seeds"; impact = "pilot observation, not population inference" },
    [pscustomobject]@{ limitation = "fixed evaluation prefixes"; impact = "validation uses 256 cached chunks per checkpoint; development uses the first 512 cached chunks, 16 free-running prompts, and up to 128 eligible pairs per seed" },
    [pscustomobject]@{ limitation = "byte tokenizer"; impact = "about 12.6 characters of mean effective context and invalid UTF-8 greedy outputs" },
    [pscustomobject]@{ limitation = "training boundary"; impact = "all runs selected step 1000; convergence plateau not established" },
    [pscustomobject]@{ limitation = "paired prefix"; impact = "eight-byte suffix does not prove semantic target invariance; exploratory only" },
    [pscustomobject]@{ limitation = "model size"; impact = "L6 and L19 parameter counts differ; depth and capacity are not fully factorized" },
    [pscustomobject]@{ limitation = "final test"; impact = "unopened, so no final performance claim" }
)
$nextRows = @(
    [pscustomobject]@{ rank = 1; candidate = "TRAIN-only reversible subword or byte-fallback tokenizer"; purpose = "increase effective Japanese context while preserving zero unknowns"; requires_new_protocol = "true" },
    [pscustomobject]@{ rank = 2; candidate = "longer validation-selected training with plateau criterion"; purpose = "separate undertraining from seed spread"; requires_new_protocol = "true" },
    [pscustomobject]@{ rank = 3; candidate = "T64 matched-token depth comparison"; purpose = "test context truncation without opening final test"; requires_new_protocol = "true" }
)

$csvMap = [ordered]@{
    "corpus-configuration.csv" = $corpusConfiguration; "corpus-inventory.csv" = $corpusInventory
    "cleaning-summary.csv" = $cleaning; "split-summary.csv" = $splitRows; "dedup-summary.csv" = $dedupRows
    "subset-summary.csv" = $subsetRows; "tokenizer-candidates.csv" = $tokenizerRows
    "tokenizer-configuration.csv" = $tokenizerConfiguration; "model-configurations.csv" = $modelRows
    "runtime-budget.csv" = $runtimeRows; "training-summary.csv" = $trainingRows
    "checkpoint-selection.csv" = $checkpointRows; "seed-stability.csv" = $stabilityRows
    "teacher-forced-summary.csv" = $teacherRows; "free-running-summary.csv" = $freeRows
    "paired-prefix-summary.csv" = $pairedRows; "depth-control.csv" = $depthRows
    "hypothesis-outcomes.csv" = $hypothesisRows; "diagnosis.csv" = $diagnosisRows
    "limitations.csv" = $limitationRows; "next-step-candidates.csv" = $nextRows
}
foreach ($entry in $csvMap.GetEnumerator()) { Write-CsvFile (Join-Path $resolvedOutput $entry.Key) @($entry.Value) }

$readme = @"
# Nicopedia real-text seed-stability pilot, August 2026

This host-only pilot connects PhoneLM to the Nicopedia data 2024-11-25 without publishing article text, titles, identifiers, token sequences, tokenizer vocabulary, checkpoints, logits, or local paths.

Held-out token NLL is stable across seeds 1/2/4 at this scale: the validation range is 0.0204 for L6 and 0.0376 for L19, and the development range is 0.0209 and 0.0360. All six ordinary-Attention CPU runs remain finite. L19 slightly improves development NLL. Under this pilot protocol, the large performance instability seen in the homogeneous synthetic task was not reproduced.

Greedy generation is less stable. Across 16 fixed prompts and 16 generated positions per depth, three-seed argmax agreement is 0.602 for L6 and 0.379 for L19; mean first divergence is 6.44 and 2.13 generated bytes. This partially reproduces seed-dependent behavior, but not the synthetic causal mechanism. The best-supported explanation is free-running amplification of small next-token ranking differences. Paired-prefix results use at most 128 eligible pairs per seed and are exploratory because an eight-byte suffix does not establish semantic target invariance.

The primary tokenizer is deterministic UTF-8 bytes (V=256, zero unknowns, T=32). This keeps the dense PhoneLM CPU head small, but limits the mean effective context to about 12.6 cleaned characters and produces many invalid UTF-8 greedy snippets. All checkpoints are selected by validation NLL; development is evaluated once after selection. Final-test articles are included only in cleaning, deduplication, and split aggregates; no final-test token cache or model evaluation was used.

Attribution: This research used the “Nicopedia data” provided by Dwango Co., Ltd. through the IDR Dataset Provision Service of the National Institute of Informatics (NII). Use is limited to non-commercial research. The dataset itself and derived tokenizer/checkpoint artifacts are not redistributed. The user remains responsible for the result notification and information-submission procedures required by the applicable terms.

No device, HTP, QNN, QAIRT, ADB, Android/JNI, UI, or COUNT_FROM_ONE operation was used.
"@
[IO.File]::WriteAllText((Join-Path $resolvedOutput "README.md"), ($readme -replace "`r`n", "`n"), [Text.UTF8Encoding]::new($false))

$manifestFiles = @()
foreach ($name in ($allowList | Where-Object { $_ -ne "manifest.json" } | Sort-Object)) {
    $manifestFiles += [ordered]@{ path = $name; sha256_normalized_lf = Get-NormalizedSha256 (Join-Path $resolvedOutput $name) }
}
$manifestSources = foreach ($relative in $sourceFiles) {
    [ordered]@{ path = $relative; sha256_normalized_lf = Get-NormalizedSha256 (Join-Path $repoRoot $relative) }
}
$privateEvidencePaths = @(
    (Join-Path $resolvedCorpus "source-manifest.json"),
    (Join-Path $resolvedCorpus "reports/public-corpus-aggregate.json"),
    (Join-Path $resolvedCorpus "reports/evidence-provenance.json"),
    (Join-Path $resolvedReports "decisions/decision-003.json"),
    (Join-Path $resolvedReports "decisions/decision-004.json"),
    (Join-Path $resolvedReports "decisions/decision-005.json"),
    (Join-Path $resolvedReports "decisions/decision-006.json")
)
foreach ($layers in @(6, 19)) {
    foreach ($seed in @(1, 2, 4)) {
        $privateEvidencePaths += Join-Path $resolvedReports "formal/l$layers/seed-$seed/run-summary.csv"
        $privateEvidencePaths += Join-Path $resolvedReports "formal/l$layers/seed-$seed/training-trajectory.csv"
    }
    foreach ($name in @("development-teacher-forced.csv", "seed-stability.csv", "paired-prefix.csv")) {
        $privateEvidencePaths += Join-Path $resolvedReports "formal/l$layers/comparison/$name"
    }
    $privateEvidencePaths += Join-Path $resolvedReports "formal/l$layers/comparison/checkpoint-provenance.csv"
}
$manifest = [ordered]@{
    schema = "NICOPEDIA_REAL_TEXT_PUBLIC_BUNDLE_V1"; schema_version = 1; source_commit = $SourceCommit
    dataset_name = "Nicopedia data"; dataset_version = "2024-11-25"; article_types = @("a", "i", "l", "o", "v")
    source_file_aggregate_sha256 = $sourceManifest.aggregate_sha256
    cleaning_protocol_hash = "sha256:$((Get-StringSha256 'unicode=NFKC;newlines=LF;control=C0-except-LF-TAB;html=text;drop=script,style,noscript;block-boundaries=LF;entities=decode;min-bytes=96;max-bytes=1048576'))"
    split_protocol_hash = "sha256:$((Get-StringSha256 'article-stable-sha256-v1;train=9000;validation=500;development=400;final=100;exact-clean-text-dedupe-before-split'))"
    tokenizer_protocol_hash = ($corpus.tokenizer_candidates | Where-Object candidate -eq "utf8_byte").protocol_hash
    model_config_hash = "sha256:$((Get-StringSha256 'v=256;t=32;d=16;ffn=32;heads=2;layers=6,19;adam=0.003,0.9,0.999,1e-8;steps=1000;batch=8'))"
    pilot_train_articles = ($corpus.cache_identities.train_pilot.articles); pilot_train_clean_utf8_bytes = ($corpus.cache_identities.train_pilot.clean_utf8_bytes)
    pilot_train_target_tokens = ($corpus.cache_identities.train_pilot.target_tokens); seeds = @(1, 2, 4)
    smoke_runs = 2; formal_training_runs = 6; additional_control_runs = 0; source_full_scans = 2
    final_test_opened = $false; synthetic_final_holdout_opened = $false; device_runs = 0; htp_runs = 0; adb_operations = 0; ui_operations = 0; count_from_one = 0
    classification = $decision.classification; evidence_strength = $decision.evidence_strength
    conclusion = "Large held-out NLL seed instability was not reproduced; greedy generation variation was partially reproduced, with stronger divergence at L19."
    publication = [ordered]@{ raw_text = $false; titles = $false; article_ids = $false; raw_token_sequences = $false; tokenizer_model_or_vocabulary = $false; checkpoints = $false; logits_or_attention = $false; local_paths = $false }
    hash_definition = "SHA-256 over UTF-8 text after CRLF and CR normalization to LF; source-file aggregate uses raw per-file SHA-256 bound to POSIX relative names and sizes"
    private_evidence_aggregate_sha256 = Get-CombinedNormalizedSha256 $privateEvidencePaths
    sources = @($manifestSources); files = @($manifestFiles); allow_list = @($allowList)
}
[IO.File]::WriteAllText((Join-Path $resolvedOutput "manifest.json"), (($manifest | ConvertTo-Json -Depth 8) -replace "`r`n", "`n") + "`n", [Text.UTF8Encoding]::new($false))
Assert-PublicBundle $resolvedOutput
Write-Host "nicopedia_public_export_status=PASS files=$($allowList.Count)"
