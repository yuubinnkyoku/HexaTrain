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

function Resolve-Under([string]$Path, [string]$AllowedRoot, [string]$Label) {
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { [IO.Path]::GetFullPath($Path) } else { [IO.Path]::GetFullPath((Join-Path $repoRoot $Path)) }
    $allowed = [IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not ($candidate + [IO.Path]::DirectorySeparatorChar).StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label resolves outside its allowed root"
    }
    $cursor = Get-Item -LiteralPath (Split-Path -Parent $candidate) -ErrorAction SilentlyContinue
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

function Assert-SafeText([string]$Text, [string]$Label) {
    $patterns = @(
        '(?i)[a-z]:(?:\\|/)', '(?<![A-Za-z0-9_])/(?:Users|home|data|sdcard|tmp)/',
        '(?i)build[\/]private', '(?i)\.(?:ckpt|bin|model|vocab|npy|npz|pt|pth)\b',
        '(?i)BEGIN [A-Z ]*PRIVATE KEY', '\bAKIA[0-9A-Z]{16}\b',
        '(?i)adb\s+-s\s+\S+', '\b\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}\b',
        '(?i)pg_(?:id|title|view_title|yomi)|txt_text',
        '(?i)raw[_ -]?(?:text|token|logit|tensor|checkpoint|parameter)\s*[:=]'
    )
    foreach ($pattern in $patterns) {
        if ($Text -match $pattern) { throw "Unsafe public payload in $Label (pattern rejected)" }
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
    foreach ($unsafe in @(
        $windowsPathFixture, $posixPathFixture,
        "checkpoint=weights.ckpt", "adb -s device shell", "pg_id=1", "raw_token=12"
    )) {
        $rejected = $false
        try { Assert-SafeText $unsafe "negative-fixture" } catch { $rejected = $true }
        if (-not $rejected) { throw "Unsafe fixture was not rejected" }
    }
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

$corpus = Get-Content -LiteralPath (Join-Path $resolvedCorpus "reports/public-corpus-aggregate.json") -Raw | ConvertFrom-Json
$sourceManifest = Get-Content -LiteralPath (Join-Path $resolvedCorpus "source-manifest.json") -Raw | ConvertFrom-Json
$decision = Get-Content -LiteralPath (Join-Path $resolvedReports "decisions/decision-005.json") -Raw | ConvertFrom-Json
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
foreach ($layers in @(6, 19)) {
    foreach ($seed in @(1, 2, 4)) {
        $runDirectory = Join-Path $resolvedReports "formal/l$layers/seed-$seed"
        $runSummaries += Import-Csv (Join-Path $runDirectory "run-summary.csv")
        $trajectories += Import-Csv (Join-Path $runDirectory "training-trajectory.csv")
    }
    $comparison = Join-Path $resolvedReports "formal/l$layers/comparison"
    $teacher += Import-Csv (Join-Path $comparison "development-teacher-forced.csv")
    $stability += Import-Csv (Join-Path $comparison "seed-stability.csv")
    $paired += Import-Csv (Join-Path $comparison "paired-prefix.csv")
}
if ($runSummaries.Count -ne 6 -or $teacher.Count -ne 6 -or $stability.Count -ne 2 -or $paired.Count -ne 6) {
    throw "Private aggregate row count mismatch"
}
if (@($runSummaries | Where-Object { $_.finite -ne "true" -or $_.development_used_for_selection -ne "false" -or $_.final_test_used -ne "false" }).Count) {
    throw "Formal run safety contract mismatch"
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
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
        selection_use = switch ($name) { "train" { "training and tokenizer comparison" }; "validation" { "checkpoint selection only" }; "development" { "one locked evaluation" }; "final_test" { "unopened; aggregate count/hash only" } } }
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

Held-out token NLL is stable across seeds 1/2/4 at this scale: the validation range is 0.0204 for L6 and 0.0376 for L19, and the development range is 0.0209 and 0.0360. All six ordinary-Attention CPU runs remain finite. L19 slightly improves development NLL, so natural context diversity does not reproduce the large performance instability seen in the homogeneous synthetic task.

Greedy generation is less stable. Three-seed argmax agreement is 0.602 for L6 and 0.379 for L19; mean first divergence is 6.44 and 2.13 generated bytes. This partially reproduces seed-dependent behavior, but not the synthetic causal mechanism. The best-supported explanation is free-running amplification of small next-token ranking differences. Paired-prefix results are exploratory because an eight-byte suffix does not establish semantic target invariance.

The primary tokenizer is deterministic UTF-8 bytes (V=256, zero unknowns, T=32). This keeps the dense PhoneLM CPU head small, but limits the mean effective context to about 12.6 cleaned characters and produces many invalid UTF-8 greedy snippets. All checkpoints are selected by validation NLL; development is evaluated once after selection, and the final test remains unopened.

Attribution: This research used the “Nicopedia data” provided by Dwango Co., Ltd. through the IDR Dataset Provision Service of the National Institute of Informatics (NII). The dataset itself and derived tokenizer/checkpoint artifacts are not redistributed.

No device, HTP, QNN, QAIRT, ADB, Android/JNI, UI, or COUNT_FROM_ONE operation was used.
"@
[IO.File]::WriteAllText((Join-Path $resolvedOutput "README.md"), ($readme -replace "`r`n", "`n"), [Text.UTF8Encoding]::new($false))

$sourceFiles = @(
    "host_tests/nicopedia_real_text_pilot.cpp", "scripts/nicopedia_real_text_pipeline.py",
    "scripts/run_nicopedia_real_text_pilot.ps1", "scripts/export_public_qnn_nicopedia_results.ps1",
    "scripts/run_host_tests.ps1", "scripts/verify_local.ps1"
)
$manifestFiles = @()
foreach ($name in ($allowList | Where-Object { $_ -ne "manifest.json" } | Sort-Object)) {
    $manifestFiles += [ordered]@{ path = $name; sha256_normalized_lf = Get-NormalizedSha256 (Join-Path $resolvedOutput $name) }
}
$manifestSources = foreach ($relative in $sourceFiles) {
    [ordered]@{ path = $relative; sha256_normalized_lf = Get-NormalizedSha256 (Join-Path $repoRoot $relative) }
}
$privateEvidencePaths = @(
    (Join-Path $resolvedCorpus "reports/public-corpus-aggregate.json"),
    (Join-Path $resolvedReports "decisions/decision-005.json")
)
$manifest = [ordered]@{
    schema = "NICOPEDIA_REAL_TEXT_PUBLIC_BUNDLE_V1"; schema_version = 1; source_commit = $SourceCommit
    dataset_name = "Nicopedia data"; dataset_version = "2024-11-25"; article_types = @("a", "i", "l", "o", "v")
    source_file_aggregate_sha256 = $sourceManifest.aggregate_sha256
    cleaning_protocol_hash = $corpus.aggregate_hash
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
