# SPDX-License-Identifier: Apache-2.0
# Orchestrate the preregistered G1 200k long-linear schedule in resume-safe segments.
param(
  [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
  [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
  [Parameter(Mandatory = $true)][string]$HexagonSdkRoot,
  [string]$PrivateRoot = 'build\private-data\g1-long-linear-200k',
  [string]$ReportRoot = 'build\reports\g1-long-linear-200k-primary',
  [ValidateRange(0, 200000)][int]$ResumeFromStep = 0,
  [ValidateRange(1000, 200000)][int]$StopAfterStep = 200000,
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$segmentEnds = @(1000, 5000)
for ($step = 10000; $step -le 200000; $step += 5000) { $segmentEnds += $step }
$fullEvaluationSteps = @(1000, 2000, 4000, 8000, 16000, 32000, 50000, 64000, 80000, 100000, 120000, 140000, 160000, 180000, 200000)
function Get-G1TrainingResultPath([string]$Directory, [int]$Step) {
  return (Join-Path $Directory "seed1-l19-v1024-t32-d64-f128-steps$Step-result.txt")
}
function Write-JsonAtomic([string]$Path, [object]$Value) {
  $temporary = $Path + '.tmp-' + [guid]::NewGuid().ToString('N')
  $json = $Value | ConvertTo-Json -Depth 24
  [IO.File]::WriteAllText($temporary, $json + "`n", [Text.UTF8Encoding]::new($false))
  if ([IO.File]::Exists($Path)) {
    $backup = $Path + '.bak-' + [guid]::NewGuid().ToString('N')
    [IO.File]::Replace($temporary, $Path, $backup)
  }
  else { [IO.File]::Move($temporary, $Path) }
}
function Assert-NoG1UnresolvedSegmentRecovery([object]$Plan) {
  if ($Plan.status -eq 'RECOVERY_REQUIRED' -or
      @($Plan.incomplete_segments | Where-Object { $_.status -eq 'ARTIFACT_RECOVERY_REQUIRED' }).Count -gt 0) {
    throw 'G1_SEGMENT_ARTIFACT_RECOVERY_REQUIRED'
  }
}
if ($SelfTest) {
  if ($segmentEnds[-1] -ne 200000 -or $segmentEnds.Count -ne 41 -or
      $fullEvaluationSteps[-1] -ne 200000 -or $fullEvaluationSteps.Count -ne 15 -or
      @($fullEvaluationSteps | Where-Object { $_ -lt 1000 -or $_ -gt 200000 -or ($_ % 1000) -ne 0 }).Count -ne 0 -or
      0 -in $segmentEnds -or $ResumeFromStep -notin (@(0) + $segmentEnds) -or
      $StopAfterStep -notin $segmentEnds -or $StopAfterStep -lt $ResumeFromStep -or
      (Split-Path -Leaf (Get-G1TrainingResultPath -Directory 'build' -Step 1000)) -ne 'seed1-l19-v1024-t32-d64-f128-steps1000-result.txt') {
    throw 'G1_LONG_LINEAR_200K_PLAN_SELF_TEST_FAILED'
  }
  $recoveryGuardRejected = $false
  try {
    Assert-NoG1UnresolvedSegmentRecovery -Plan ([pscustomobject]@{
      status = 'RECOVERY_REQUIRED'
      incomplete_segments = @([pscustomobject]@{ status = 'ARTIFACT_RECOVERY_REQUIRED' })
    })
  } catch { $recoveryGuardRejected = $_.Exception.Message -eq 'G1_SEGMENT_ARTIFACT_RECOVERY_REQUIRED' }
  if (-not $recoveryGuardRejected) { throw 'G1_LONG_LINEAR_200K_RECOVERY_GUARD_SELF_TEST_FAILED' }
  Assert-NoG1UnresolvedSegmentRecovery -Plan ([pscustomobject]@{
    status = 'RUNNING'
    incomplete_segments = @([pscustomobject]@{ status = 'HEALTHY_COMPLETE' })
  })
  $completionPropertyPlan = [pscustomobject]@{ status = 'COMPLETED_200000' }
  $completionPropertyPlan | Add-Member -MemberType NoteProperty -Name completed_utc `
    -Value '2026-10-09T00:00:00Z' -Force
  $completionPropertyJson = $completionPropertyPlan | ConvertTo-Json -Depth 24
  $completionPropertyRoundTrip = $completionPropertyJson | ConvertFrom-Json
  if ($completionPropertyJson -notmatch '"completed_utc"\s*:\s*"2026-10-09T00:00:00Z"' -or
      $null -eq $completionPropertyRoundTrip.PSObject.Properties['completed_utc']) {
    throw 'G1_LONG_LINEAR_200K_COMPLETION_PROPERTY_SELF_TEST_FAILED'
  }
  $runLedgerProbe = [pscustomobject][ordered]@{
    status = 'PREPARED'; last_completed_step = 0; updated_utc = $null
    segments = @(); full_evaluations = @()
  }
  $runLedgerProbe.last_completed_step = 1000
  $runLedgerProbe.updated_utc = '2026-10-09T00:00:00Z'
  $runLedgerProbe.segments = @([pscustomobject]@{ steps = 1000 })
  $runLedgerProbe.full_evaluations = @([pscustomobject]@{ step = 1000 })
  if ($runLedgerProbe.last_completed_step -ne 1000 -or
      $runLedgerProbe.segments.Count -ne 1 -or $runLedgerProbe.full_evaluations.Count -ne 1 -or
      $runLedgerProbe.updated_utc -ne '2026-10-09T00:00:00Z') {
    throw 'G1_LONG_LINEAR_200K_RUN_LEDGER_SELF_TEST_FAILED'
  }
  $atomicPath = Join-Path ([IO.Path]::GetTempPath()) ('phonelm-g1-run-plan-' + [guid]::NewGuid().ToString('N') + '.json')
  $atomicBackups = @()
  try {
    Write-JsonAtomic -Path $atomicPath -Value ([pscustomobject]@{ revision = 1 })
    Write-JsonAtomic -Path $atomicPath -Value ([pscustomobject]@{ revision = 2 })
    $atomicBackups = @(Get-ChildItem -LiteralPath ((Split-Path -Parent $atomicPath)) -Filter ((Split-Path -Leaf $atomicPath) + '.bak-*'))
    $atomicValue = Get-Content -LiteralPath $atomicPath -Raw | ConvertFrom-Json
    $backupValue = if ($atomicBackups.Count -eq 1) { Get-Content -LiteralPath $atomicBackups[0].FullName -Raw | ConvertFrom-Json } else { $null }
    if ([int]$atomicValue.revision -ne 2 -or $null -eq $backupValue -or [int]$backupValue.revision -ne 1) {
      throw 'G1_LONG_LINEAR_200K_ATOMIC_PLAN_WRITE_SELF_TEST_FAILED'
    }
  } finally {
    Remove-Item -LiteralPath $atomicPath -Force -ErrorAction SilentlyContinue
    foreach ($atomicBackup in $atomicBackups) { Remove-Item -LiteralPath $atomicBackup.FullName -Force -ErrorAction SilentlyContinue }
  }
  $orderScript = Join-Path $PSScriptRoot 'g1_training_order_identity.py'
  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) { throw 'PYTHON_UNAVAILABLE_FOR_ORDER_IDENTITY_SELF_TEST' }
  $orderOutput = & $python.Source $orderScript --self-test
  if ($LASTEXITCODE -ne 0 -or ($orderOutput -join "`n") -notmatch 'g1_training_order_identity_self_test=PASS') {
    throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_SELF_TEST_FAILED'
  }
  Write-Host 'g1_long_linear_200k_plan_self_test=PASS segments=41 full_eval_boundaries=15 order_identity=PASS recovery_guard=PASS completion_property=PASS run_ledger=PASS atomic_replace=PASS result_name=PASS'
  exit 0
}

$root = Split-Path -Parent $PSScriptRoot
$buildPrefix = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
function Resolve-BuildPath([string]$PathValue) {
  $candidate = if ([IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $root $PathValue }
  $resolved = [IO.Path]::GetFullPath($candidate)
  if (-not $resolved.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'EXPERIMENT_PATH_MUST_BE_UNDER_BUILD'
  }
  return $resolved
}
function Get-DeviceFreeBytes([string]$Adb, [string]$Device) {
  $result = Invoke-PhoneLmAdb -Adb $Adb -Device $Device -Arguments @('shell', 'df', '-k', '/data/user/0')
  $line = @($result.Text -split "`r?`n" | Where-Object { $_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%\s+' } | Select-Object -Last 1)
  if ($line.Count -ne 1) { throw 'DEVICE_STORAGE_STATUS_UNAVAILABLE' }
  $columns = @($line[0] -split '\s+')
  if ($columns.Count -lt 6) { throw 'DEVICE_STORAGE_STATUS_INVALID' }
  return ([long]$columns[3] * 1024L)
}
function Assert-StorageHeadroom([string]$Adb, [string]$Device, [long]$CacheBytes, [long]$CheckpointBytes) {
  $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($root))
  $checkpointReserve = [long]($CheckpointBytes * 200 * 1.5)
  $hostRequired = [math]::Max(20GB, [long]($CacheBytes * 1.5) + $checkpointReserve + 10GB)
  if ($drive.AvailableFreeSpace -lt $hostRequired) { throw "HOST_STORAGE_HEADROOM_LOW: free=$($drive.AvailableFreeSpace) required=$hostRequired" }
  $deviceFree = Get-DeviceFreeBytes -Adb $Adb -Device $Device
  $deviceRequired = [math]::Max(12GB, [long]($CacheBytes * 2) + $checkpointReserve + 3GB)
  if ($deviceFree -lt $deviceRequired) { throw "DEVICE_STORAGE_HEADROOM_LOW: free=$deviceFree required=$deviceRequired" }
  return [pscustomobject]@{ host_free_bytes = $drive.AvailableFreeSpace; host_required_bytes = $hostRequired; device_free_bytes = $deviceFree; device_required_bytes = $deviceRequired }
}
function New-RunId([int]$FromStep, [int]$ToStep) {
  return ('g1long200k-p-{0:D6}-{1:D6}-{2}' -f $FromStep, $ToStep, [guid]::NewGuid().ToString('N').Substring(0, 6))
}
function Get-CheckpointPath([string]$Directory, [int]$Step) {
  $name = Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 64 -FeedForwardDimension 128 -Step $Step
  return (Join-Path $Directory $name)
}

$privateRootResolved = Resolve-BuildPath $PrivateRoot
$reportRootResolved = Resolve-BuildPath $ReportRoot
$trainingReportRoot = Join-Path $reportRootResolved 'training'
$fullEvalRoot = Join-Path $reportRootResolved 'full-cap-evaluations'
$manifestPath = Join-Path $privateRootResolved 'expanded-train-manifest.json'
$cachePath = Join-Path $privateRootResolved 'caches\train-expanded.bin'
$cacheRoot = Join-Path $privateRootResolved 'caches'
$tokenizerPath = Join-Path $privateRootResolved 'tokenizer\byte-bpe-v1024.model'
$planPath = Join-Path $reportRootResolved 'run-plan.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $cachePath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $tokenizerPath -PathType Leaf)) { throw 'EXPANDED_TRAIN_INPUT_MISSING' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$cacheHash = 'sha256:' + (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash.ToLowerInvariant()
$tokenizerHash = 'sha256:' + (Get-FileHash -LiteralPath $tokenizerPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifest.schema -ne 'NICOPEDIA_G1_EXPANDED_TRAIN_V1' -or
    $cacheHash -ne 'sha256:bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf' -or
    $manifest.cache.sha256 -ne $cacheHash -or
    [long]$manifest.cache.self_test.file_size_bytes -ne 942868468 -or
    [int]$manifest.cache.context -ne 32 -or
    $manifest.cache.format -ne 'NPRTBPEV1' -or
    [long]$manifest.cache.records -ne 12741465 -or
    $manifest.tokenizer.sha256 -ne $tokenizerHash -or
    $tokenizerHash -ne 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798' -or
    $manifest.tokenizer.kind -ne 'byte_bpe' -or [int]$manifest.tokenizer.vocabulary -ne 1024 -or
    [long]$manifest.chosen_training_data.target_bpe_tokens -ne 407726880 -or
    [long]$manifest.chosen_training_data.exact_t32_records -ne 12741465 -or
    $manifest.training_order.algorithm -ne 'SplitMix64(global_selection_index + state) modulo record_count; with replacement' -or
    [long]$manifest.training_order.seed -ne 20260806 -or
    $manifest.training_order.hash -ne 'fnv1a64:be3269f541236dc2' -or
    [long]$manifest.training_order.selection_count -ne 800000 -or
    [int]$manifest.training_order.hard_ceiling_steps -ne 100000 -or
    $manifest.split_leakage.selected_train_intersects_validation -ne 0 -or
    $manifest.split_leakage.selected_train_intersects_development -ne 0 -or
    $manifest.split_leakage.selected_train_intersects_final_test -ne 0 -or
    $manifest.final_test_opened -ne $false -or
    $manifest.final_test_body_scan_scope -ne 'cleaning_and_exact_text_deduplication_only' -or
    $manifest.final_test_dedupe_only_scan.model_facing_or_quality_facing_access -ne $false -or
    $manifest.final_test_used_for_training_or_quality -ne $false -or
    $manifest.final_test_evaluated -ne $false) {
  throw 'EXPANDED_TRAIN_IDENTITY_OR_LEAKAGE_REJECTED'
}
$orderScript = Join-Path $PSScriptRoot 'g1_training_order_identity.py'
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'PYTHON_UNAVAILABLE_FOR_ORDER_IDENTITY' }
$orderOutput = & $python.Source $orderScript --record-count $manifest.cache.records --steps 200000
if ($LASTEXITCODE -ne 0) { throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_COMPUTE_FAILED' }
try { $orderIdentity = ($orderOutput -join "`n") | ConvertFrom-Json }
catch { throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_OUTPUT_INVALID' }
if ([int]$orderIdentity.steps -ne 200000 -or [int]$orderIdentity.batch_size -ne 8 -or
    [long]$orderIdentity.order_seed -ne 20260806 -or
    [long]$orderIdentity.record_count -ne 12741465 -or
    [long]$orderIdentity.selection_count -ne 1600000 -or
    $orderIdentity.prefix_100k_hash -ne [string]$manifest.training_order.hash -or
    $orderIdentity.hash -ne 'fnv1a64:83f9c96cf10261a0') {
  throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_MISMATCH'
}
$trainingOrderHash = [string]$orderIdentity.hash
foreach ($split in @('validation', 'development')) {
  $splitPath = Join-Path $cacheRoot "$split.bin"
  if (-not (Test-Path -LiteralPath $splitPath -PathType Leaf)) { throw "HELDOUT_CACHE_MISSING: $split" }
  $splitHash = 'sha256:' + (Get-FileHash -LiteralPath $splitPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($manifest.heldout_cache_identity.$split.sha256 -ne $splitHash) { throw "HELDOUT_CACHE_IDENTITY_MISMATCH: $split" }
}
if ($ResumeFromStep -notin (@(0) + $segmentEnds)) { throw 'RESUME_STEP_NOT_A_COMMITTED_SEGMENT_BOUNDARY' }
if ($StopAfterStep -notin $segmentEnds -or $StopAfterStep -lt $ResumeFromStep) { throw 'STOP_STEP_NOT_A_FORWARD_SEGMENT_BOUNDARY' }
if (-not (Test-Path -LiteralPath $HexagonSdkRoot -PathType Container)) { throw 'PINNED_HEXAGON_SDK_UNAVAILABLE' }
$hexagonManifestPath = Join-Path $HexagonSdkRoot 'hexagon_sdk.json'
if (-not (Test-Path -LiteralPath $hexagonManifestPath -PathType Leaf)) { throw 'PINNED_HEXAGON_SDK_MANIFEST_UNAVAILABLE' }
$hexagonManifest = Get-Content -LiteralPath $hexagonManifestPath -Raw | ConvertFrom-Json
$hexagonTools = @($hexagonManifest.root.tools.info | Where-Object { $_.name -eq 'Hexagon Tools' })
$hexagonCompilerPath = Join-Path $HexagonSdkRoot 'tools\HEXAGON_Tools\19.0.07\Tools\bin\hexagon-clang.exe'
if ($hexagonManifest.root.id.version -ne '6.6.0.0' -or
    $hexagonTools.Count -ne 1 -or $hexagonTools[0].build -ne '19.0.07' -or
    -not (Test-Path -LiteralPath $hexagonCompilerPath -PathType Leaf)) {
  throw 'PINNED_HEXAGON_SDK_IDENTITY_MISMATCH'
}
$hexagonManifestSha256 = 'sha256:' + (Get-FileHash -LiteralPath $hexagonManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$hexagonCompilerSha256 = 'sha256:' + (Get-FileHash -LiteralPath $hexagonCompilerPath -Algorithm SHA256).Hash.ToLowerInvariant()

[IO.Directory]::CreateDirectory($trainingReportRoot) | Out-Null
[IO.Directory]::CreateDirectory($fullEvalRoot) | Out-Null
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
$before = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase 'g1-long-linear-200k-orchestrator-before'
if ($before.thermal_status -ge 5) { throw 'DEVICE_THERMAL_STATUS_REJECTED' }

$trainingScript = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$evalScript = Join-Path $PSScriptRoot 'run_nicopedia_htp_eval.ps1'
if ($ResumeFromStep -eq 0) {
  if (Test-Path -LiteralPath $planPath) { throw 'FRESH_PRIMARY_RUN_PLAN_ALREADY_EXISTS' }
  $firstRunId = New-RunId -FromStep 0 -ToStep 1000
  $plan = [pscustomobject][ordered]@{
    schema = 'G1_LONG_LINEAR_200K_RUN_PLAN_V1'
    status = 'PREPARED'
    branch_base_sha = '712a3495d3f18cc5ed66979b0f0d8266073e1fb5'
    source_100k_head_sha = 'a3e3ce3339914d00a59673af29357f5ce180f209'
    tooling_source_sha = '5730d735c3a32604c16d132914397f2d56c01caf'
    device_serial_private = $deviceInfo.Serial
    device_model_private = $deviceInfo.Model
    qairt_build_id = $ExpectedBuildId
    hexagon_sdk_version = [string]$hexagonManifest.root.id.version
    hexagon_tools_build = [string]$hexagonTools[0].build
    hexagon_sdk_manifest_sha256 = $hexagonManifestSha256
    hexagon_clang_sha256 = $hexagonCompilerSha256
    fresh_initialization = $true
    research_baseline = 'G1'
    architecture = [pscustomobject]@{ gate = 'headwise_g1_sigmoid'; vocabulary = 1024; context = 32; dimension = 64; feed_forward = 128; layers = 19; heads = 2; parameters = 760960; batch_size = 8; seed = 1 }
    optimizer = [pscustomobject]@{ muon_backend = 'HVX_W8'; muon_learning_rate = 0.0075; muon_terminal_learning_rate = 0.0003409090909090909; muon_momentum = 0.95; muon_nesterov = $true; muon_newton_schulz_steps = 5; auxiliary_optimizer = 'Adam'; auxiliary_learning_rate = 0.0033; auxiliary_beta1 = 0.9; auxiliary_beta2 = 0.999; auxiliary_epsilon = 0.00000001; weight_decay = 0; gradient_clipping = 'disabled' }
    schedule = [pscustomobject]@{ expression = 'LR(s)=peak+(terminal-peak)*clamp((s-4000)/196000,0,1), applied to 1-indexed optimizer step s'; peak_lr = 0.0033; decay_start = 4000; decay_end = 200000; decay_steps = 196000; terminal_lr = 0.00015; muon_peak_lr = 0.0075; muon_terminal_lr = 0.0003409090909090909; total_hard_ceiling = 200000; same_decay_multiplier_for_both_optimizers = $true; step_100000_aux_lr = 0.001757142857142857; step_100000_muon_lr = 0.003993506493506493; peak_fraction_at_100000 = 0.532467532467532; terminal_fraction = 0.0454545454545455; schedule_frozen = $true; hpo_or_quality_adaptation = $false }
    dataset = [pscustomobject]@{ method = $manifest.chosen_training_data.method; articles = $manifest.chosen_training_data.articles; clean_utf8_bytes = $manifest.chosen_training_data.clean_utf8_bytes; t32_records = $manifest.chosen_training_data.exact_t32_records; target_bpe_tokens = $manifest.chosen_training_data.target_bpe_tokens; represented_original_utf8_bytes = $manifest.chosen_training_data.represented_original_utf8_bytes; eligible_train_articles = $manifest.eligible_train_articles; eligible_train_clean_utf8_bytes = $manifest.eligible_train_clean_utf8_bytes; full_train_measurement = $manifest.full_train_measurement; selected_subset = $manifest.selected_subset; cache_sha256 = $cacheHash; tokenizer_sha256 = $tokenizerHash; heldout_validation_sha256 = $manifest.heldout_cache_identity.validation.sha256; heldout_development_sha256 = $manifest.heldout_cache_identity.development.sha256; source_training_order_hash_100k = $manifest.training_order.hash; training_order_hash_200k = $trainingOrderHash; training_order_algorithm = $manifest.training_order.algorithm; training_order_seed = $manifest.training_order.seed; training_order_selection_count_200k = 1600000; target_tokens_at_100000 = 25600000; target_tokens_at_hard_ceiling = 51200000; manifest_sha256 = $manifest.manifest_sha256; manifest_file_sha256 = ('sha256:' + (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()); source_aggregate_sha256 = $manifest.source_file_aggregate_sha256 }
    final_test_opened = $false
    final_test_dedupe_only_scan = $manifest.final_test_dedupe_only_scan
    status_6031 = 'UNRESOLVED / DORMANT / WATCH'
    lightweight_evaluation = 'host CPU evaluation of first 1 T32 cache record in each fixed validation/development cache at every 1000-step checkpoint'
    full_cap_evaluation_steps = $fullEvaluationSteps
    segment_target_steps = $segmentEnds
    checkpoint_interval = 1000
    first_cache_run_id_private = $firstRunId
    initial_checkpoint_size_estimate_bytes = 8388608
    last_completed_step = $ResumeFromStep
    updated_utc = $null
    segments = @()
    full_evaluations = @()
    created_utc = [DateTime]::UtcNow.ToString('o')
  }
  Write-JsonAtomic -Path $planPath -Value $plan
} else {
  if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) { throw 'RESUME_RUN_PLAN_MISSING' }
  $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
  Assert-NoG1UnresolvedSegmentRecovery -Plan $plan
  if ($plan.schema -ne 'G1_LONG_LINEAR_200K_RUN_PLAN_V1' -or
      $plan.branch_base_sha -ne '712a3495d3f18cc5ed66979b0f0d8266073e1fb5' -or
      $plan.qairt_build_id -ne $ExpectedBuildId -or
      $plan.hexagon_sdk_version -ne [string]$hexagonManifest.root.id.version -or
      $plan.hexagon_tools_build -ne [string]$hexagonTools[0].build -or
      $plan.hexagon_sdk_manifest_sha256 -ne $hexagonManifestSha256 -or
      $plan.hexagon_clang_sha256 -ne $hexagonCompilerSha256 -or
      $plan.dataset.cache_sha256 -ne $cacheHash -or
      $plan.dataset.tokenizer_sha256 -ne $tokenizerHash -or
      $plan.dataset.training_order_hash_200k -ne $trainingOrderHash -or
      $plan.device_serial_private -ne $deviceInfo.Serial -or
      $plan.status -eq 'COMPLETED_200000') { throw 'RESUME_RUN_PLAN_IDENTITY_MISMATCH' }
  $prior = @($plan.segments | Where-Object { [int]$_.steps -eq $ResumeFromStep -and $_.status -eq 'HEALTHY_COMPLETE' })
  if ($prior.Count -ne 1) { throw 'RESUME_STEP_NOT_PRESENT_IN_SUCCESSFUL_SEGMENT_HISTORY' }
  $resumePath = Get-CheckpointPath -Directory $trainingReportRoot -Step $ResumeFromStep
  if (-not (Test-Path -LiteralPath $resumePath -PathType Leaf) -or
      ('sha256:' + (Get-FileHash -LiteralPath $resumePath -Algorithm SHA256).Hash.ToLowerInvariant()) -ne $prior[0].checkpoint_sha256) {
    throw 'RESUME_CHECKPOINT_HASH_MISMATCH'
  }
}

$previousStep = $ResumeFromStep
$checkpointSizeEstimate = [long]$plan.initial_checkpoint_size_estimate_bytes
$completedEvaluationSteps = @($plan.full_evaluations | ForEach-Object { [int]$_.step })
foreach ($targetStep in @($segmentEnds | Where-Object { $_ -gt $ResumeFromStep -and $_ -le $StopAfterStep })) {
  $deviceInfo = Resolve-PhoneLmDevice -Adb $adb
  if ($deviceInfo.Serial -ne $plan.device_serial_private) { throw 'ADB_STABLE_DEVICE_CHANGED' }
  $device = $deviceInfo.Endpoint
  Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
  Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
  Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
  $thermal = Get-PhoneLmThermalBatteryState -Adb $adb -Device $device -Phase "training-before-$targetStep"
  $storageBefore = Assert-StorageHeadroom -Adb $adb -Device $device -CacheBytes ([IO.FileInfo]::new($cachePath).Length) -CheckpointBytes $checkpointSizeEstimate
  $runId = if ($previousStep -eq 0) { $plan.first_cache_run_id_private } else { New-RunId -FromStep $previousStep -ToStep $targetStep }
  $cacheSourceRunId = if ($previousStep -eq 0) { '' } else { $plan.first_cache_run_id_private }
  $started = [DateTime]::UtcNow
  Write-Host "segment_start from=$previousStep to=$targetStep thermal=$($thermal.thermal_status) battery_temp_c=$($thermal.battery_temperature_c) host_free_bytes=$($storageBefore.host_free_bytes) device_free_bytes=$($storageBefore.device_free_bytes)"
  $segmentLogName = 'segment-{0:D6}-{1:D6}-{2}.log' -f $previousStep, $targetStep, $runId
  $segmentLogPath = Join-Path $trainingReportRoot $segmentLogName
  $segmentLogLines = [Collections.Generic.List[string]]::new()
  $trainArguments = @{
    QairtSdkRoot = $QairtSdkRoot; ExpectedBuildId = $ExpectedBuildId; HexagonSdkRoot = $HexagonSdkRoot
    SkipBuild = $true; SkipInstall = $true; Seed = 1; Layers = 19; Steps = $targetStep; Tokens = 32; Vocabulary = 1024
    Dimension = 64; FeedForwardDimension = 128; BatchSize = 8; LearningRate = '0.0033'
    LearningRateSchedule = 'linear_decay'; DecayStartStep = 4000; DecayEndStep = 200000; ScheduleTotalSteps = 200000
    ExpectedTrainingOrderHash = $trainingOrderHash; G1LongLinear200kExperiment = $true
    TargetLearningRate = '0.00015'; ExperimentFork = $true; ParentLearningRate = '0.0033'; Optimizer = 'Muon'
    AttentionGate = 'headwise_g1_sigmoid'; MuonBackend = 'HVX'; MuonLearningRate = '0.0075'; MuonMomentum = '0.95'; MuonNsSteps = 5
    CachePath = $cachePath; EvalCacheRoot = $cacheRoot; TokenizerModelPath = $tokenizerPath; ReportRoot = $trainingReportRoot
    ExpectedDeviceSerial = $plan.device_serial_private; ResumeStep = $previousStep; CheckpointInterval = 1000
    RunId = $runId; CacheSourceRunId = $cacheSourceRunId; AllowQualityFailure = $true
    PollLimit = 86400; PollSeconds = 5; ProgressEverySeconds = 30; CheckpointStallSeconds = 7200
  }
  try {
    & $trainingScript @trainArguments *>&1 | ForEach-Object {
      $line = $_.ToString()
      $segmentLogLines.Add($line)
      Write-Output $_
    }
  } finally {
    [IO.File]::WriteAllLines($segmentLogPath, $segmentLogLines, [Text.UTF8Encoding]::new($false))
  }
  $resultPath = Get-G1TrainingResultPath -Directory $trainingReportRoot -Step $targetStep
  if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) { throw "SEGMENT_RESULT_MISSING: $targetStep" }
  $trainingText = Get-Content -LiteralPath $resultPath -Raw
  $trainingMap = Get-PhoneLmKeyValueMap -Text $trainingText
  foreach ($field in @('completed_steps','qnn_return_code_success','output_tensors_finite','final_finite','all_steps_finite','cpu_fallback','fallback','optimizer_muon_backend','hvx_rpc_failure_count','hvx_fallback_count','hvx_nonfinite_count','target_tokens_seen','records_seen','target_original_utf8_bytes_seen','unique_records_seen','unique_articles_seen','cache_traversal_fraction','training_order_hash','training_order_seed','dataset_hash','initial_parameter_hash','parameter_count','learning_rate_schedule_total_steps','learning_rate_decay_end_step')) {
    if (-not $trainingMap.Contains($field)) { throw "SEGMENT_HEALTH_FIELD_MISSING: $field" }
  }
  if ([int]$trainingMap.completed_steps -ne $targetStep -or $trainingMap.qnn_return_code_success -ne 'true' -or
      $trainingMap.output_tensors_finite -ne 'true' -or $trainingMap.final_finite -ne 'true' -or
      $trainingMap.all_steps_finite -ne 'true' -or $trainingMap.cpu_fallback -ne 'false' -or
      $trainingMap.fallback -ne 'false' -or $trainingMap.optimizer_muon_backend -ne 'HVX_W8' -or
      [int]$trainingMap.hvx_rpc_failure_count -ne 0 -or [int]$trainingMap.hvx_fallback_count -ne 0 -or
      [int]$trainingMap.hvx_nonfinite_count -ne 0 -or
      [long]$trainingMap.target_tokens_seen -ne ([long]$targetStep * 8 * 32) -or
      [long]$trainingMap.records_seen -ne ([long]$targetStep * 8) -or
      [long]$trainingMap.training_order_seed -ne 20260806 -or
      $trainingMap.dataset_hash -ne 'fnv1a64:245228bcbe304f51' -or
      $trainingMap.initial_parameter_hash -ne 'fnv1a64:01b81108b8c8b842' -or
      [long]$trainingMap.parameter_count -ne 760960 -or
      [int]$trainingMap.learning_rate_schedule_total_steps -ne 200000 -or
      [int]$trainingMap.learning_rate_decay_end_step -ne 200000 -or
      $trainingMap.training_order_hash -ne $trainingOrderHash) {
    throw "SEGMENT_HEALTH_OR_EXPOSURE_REJECTED: $targetStep"
  }
  $finalCheckpoint = Get-CheckpointPath -Directory $trainingReportRoot -Step $targetStep
  if (-not (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)) { throw "SEGMENT_CHECKPOINT_MISSING: $targetStep" }
  $checkpointHash = 'sha256:' + (Get-FileHash -LiteralPath $finalCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
  $checkpointSizeEstimate = [IO.FileInfo]::new($finalCheckpoint).Length
  $storageAfter = Assert-StorageHeadroom -Adb $adb -Device $device -CacheBytes ([IO.FileInfo]::new($cachePath).Length) -CheckpointBytes $checkpointSizeEstimate
  $segmentResult = [pscustomobject]@{
    from_step = $previousStep; steps = $targetStep; run_id_private = $runId
    started_utc = $started.ToString('o'); finished_utc = [DateTime]::UtcNow.ToString('o')
    status = 'HEALTHY_COMPLETE'; native_status = $trainingMap.status
    checkpoint_sha256 = $checkpointHash; checkpoint_bytes = $checkpointSizeEstimate
    target_tokens_seen = [long]$trainingMap.target_tokens_seen
    original_utf8_bytes_seen = [long]$trainingMap.target_original_utf8_bytes_seen
    unique_records_seen = [long]$trainingMap.unique_records_seen
    unique_articles_seen = [long]$trainingMap.unique_articles_seen
    cache_traversal_fraction = [double]$trainingMap.cache_traversal_fraction
    qnn_return_code_success = $trainingMap.qnn_return_code_success
    hvx_rpc_failure_count = [int]$trainingMap.hvx_rpc_failure_count
    hvx_fallback_count = [int]$trainingMap.hvx_fallback_count
    hvx_nonfinite_count = [int]$trainingMap.hvx_nonfinite_count
    host_free_bytes_after = $storageAfter.host_free_bytes; device_free_bytes_after = $storageAfter.device_free_bytes
    segment_log_path = $segmentLogName
    segment_log_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $segmentLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    result_file_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  $plan.segments = @($plan.segments) + @($segmentResult)
  $plan.status = 'RUNNING'
  $plan.last_completed_step = $targetStep
  $plan.updated_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $planPath -Value $plan
  Write-Host "segment_complete step=$targetStep checkpoint_sha256=$checkpointHash host_free_bytes=$($storageAfter.host_free_bytes) device_free_bytes=$($storageAfter.device_free_bytes)"

  foreach ($evalStep in @($fullEvaluationSteps | Where-Object { $_ -le $targetStep -and $_ -gt $previousStep })) {
    if ($evalStep -in $completedEvaluationSteps) { continue }
    $evalCheckpoint = Get-CheckpointPath -Directory $trainingReportRoot -Step $evalStep
    if (-not (Test-Path -LiteralPath $evalCheckpoint -PathType Leaf)) { throw "FULL_EVAL_CHECKPOINT_MISSING: $evalStep" }
    $evalAttempt = [guid]::NewGuid().ToString('N').Substring(0, 6)
    $evalRunId = "g1long200k-eval-$('{0:D6}' -f $evalStep)-$evalAttempt"
    $evalDirectory = Join-Path $fullEvalRoot ("step-{0:D6}" -f $evalStep)
    [IO.Directory]::CreateDirectory($evalDirectory) | Out-Null
    $evalArguments = @{
      QairtSdkRoot = $QairtSdkRoot; ExpectedBuildId = $ExpectedBuildId; SkipBuild = $true; SkipInstall = $true
      Seed = 1; Layers = 19; Heads = 2; Tokens = 32; Vocabulary = 1024; Dimension = 64; FeedForwardDimension = 128
      AttentionGate = 'headwise_g1_sigmoid'; CheckpointStep = $evalStep; G1LongLinear200kExperiment = $true
      SkipHostEvaluation = $true
      ValidationChunks = [int]$manifest.heldout_cache_identity.validation.records
      DevelopmentChunks = [int]$manifest.heldout_cache_identity.development.records
      CheckpointPath = $evalCheckpoint; CacheRoot = $cacheRoot; TokenizerModelPath = $tokenizerPath
      ReportRoot = $evalDirectory; ExpectedDeviceSerial = $plan.device_serial_private; RunId = $evalRunId
      PollLimit = 86400; PollSeconds = 5; ProgressEverySeconds = 30
    }
    $evalStarted = [DateTime]::UtcNow
    Write-Host "full_cap_eval_start step=$evalStep validation_chunks=$($evalArguments.ValidationChunks) development_chunks=$($evalArguments.DevelopmentChunks)"
    & $evalScript @evalArguments
    $modelTag = '-t32-d64-f128'
    $capacityTag = "-v$($evalArguments.ValidationChunks)-d$($evalArguments.DevelopmentChunks)"
    $evalTextPath = Join-Path $evalDirectory "seed1-l19$modelTag-step$evalStep$capacityTag-htp.txt"
    if (-not (Test-Path -LiteralPath $evalTextPath -PathType Leaf)) { throw "FULL_EVAL_RESULT_MISSING: $evalStep" }
    $evalMap = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $evalTextPath -Raw)
    foreach ($field in @('status','validation_chunks','development_chunks','validation_nll','development_nll','validation_top1','development_top1','validation_bits_per_utf8_byte','development_bits_per_utf8_byte','qnn_return_code_success','output_tensors_finite')) {
      if (-not $evalMap.Contains($field)) { throw "FULL_EVAL_FIELD_MISSING: step=$evalStep field=$field" }
    }
    if ($evalMap.status -ne 'SUCCESS' -or $evalMap.qnn_return_code_success -ne 'true' -or
        $evalMap.output_tensors_finite -ne 'true' -or
        [int]$evalMap.validation_chunks -ne $evalArguments.ValidationChunks -or
        [int]$evalMap.development_chunks -ne $evalArguments.DevelopmentChunks) {
      throw "FULL_EVAL_HEALTH_OR_CAP_REJECTED: $evalStep"
    }
    $evalResult = [pscustomobject]@{
      step = $evalStep; run_id_private = $evalRunId; status = 'SUCCESS'
      started_utc = $evalStarted.ToString('o'); completed_utc = [DateTime]::UtcNow.ToString('o')
      elapsed_seconds = ([DateTime]::UtcNow - $evalStarted).TotalSeconds
      validation_nll = [string]$evalMap.validation_nll; validation_bpb = [string]$evalMap.validation_bits_per_utf8_byte
      validation_top1 = [string]$evalMap.validation_top1; validation_chunks = [int]$evalMap.validation_chunks
      development_nll = [string]$evalMap.development_nll; development_bpb = [string]$evalMap.development_bits_per_utf8_byte
      development_top1 = [string]$evalMap.development_top1; development_chunks = [int]$evalMap.development_chunks
      result_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $evalTextPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $plan.full_evaluations = @($plan.full_evaluations) + @($evalResult)
    $completedEvaluationSteps += $evalStep
    $plan.updated_utc = [DateTime]::UtcNow.ToString('o')
    Write-JsonAtomic -Path $planPath -Value $plan
    Write-Host "full_cap_eval_complete step=$evalStep validation_nll=$($evalResult.validation_nll) development_nll=$($evalResult.development_nll)"
  }
  $previousStep = $targetStep
}
if ($previousStep -eq 200000 -and @($plan.full_evaluations | Where-Object { [int]$_.step -in $fullEvaluationSteps }).Count -eq $fullEvaluationSteps.Count) {
  $plan.status = 'COMPLETED_200000'
  $plan | Add-Member -MemberType NoteProperty -Name completed_utc -Value ([DateTime]::UtcNow.ToString('o')) -Force
  Write-JsonAtomic -Path $planPath -Value $plan
  Write-Host 'g1_long_linear_200k_orchestration=COMPLETED_200000'
} elseif ($previousStep -eq $StopAfterStep) {
  $plan.status = 'RUNNING'
  $plan.updated_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $planPath -Value $plan
  Write-Host "g1_long_linear_200k_orchestration=SEGMENT_BOUNDARY step=$previousStep hard_ceiling=200000"
} else {
  throw 'ORCHESTRATION_DID_NOT_REACH_REQUESTED_SEGMENT_BOUNDARY'
}
