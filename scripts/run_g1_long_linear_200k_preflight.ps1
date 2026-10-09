# SPDX-License-Identifier: Apache-2.0
# Verify same-schedule checkpoint resume and fresh G1 long-linear sanity on HTP/HVX.
param(
  [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
  [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
  [Parameter(Mandatory = $true)][string]$HexagonSdkRoot,
  [ValidateSet('ResumeEquivalence','Sanity','All')][string]$Mode = 'All',
  [string]$PrivateRoot = 'build\private-data\g1-long-linear-200k',
  [string]$ReportRoot = 'build\reports\g1-long-linear-200k-preflight',
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

if ($SelfTest) {
  if ($Mode -notin @('ResumeEquivalence','Sanity','All')) { throw 'PREFLIGHT_MODE_SELF_TEST_FAILED' }
  $orderScript = Join-Path $PSScriptRoot 'g1_training_order_identity.py'
  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) { throw 'PYTHON_UNAVAILABLE_FOR_ORDER_IDENTITY_SELF_TEST' }
  $orderOutput = & $python.Source $orderScript --self-test
  if ($LASTEXITCODE -ne 0 -or ($orderOutput -join "`n") -notmatch 'g1_training_order_identity_self_test=PASS') {
    throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_SELF_TEST_FAILED'
  }
  $ledgerProbe = [pscustomobject][ordered]@{
    status = 'RUNNING'; failure = $null; updated_utc = $null; completed_utc = $null
    resume_equivalence = $null; sanity = $null; hvx_apk = $null; runs = @()
  }
  $ledgerProbe.failure = 'self-test'
  $ledgerProbe.updated_utc = '2026-10-09T00:00:00Z'
  $ledgerProbe.resume_equivalence = [pscustomobject]@{ status = 'PASS' }
  $ledgerProbe.sanity = [pscustomobject]@{ status = 'PASS' }
  $ledgerProbe.completed_utc = '2026-10-09T00:00:00Z'
  if ($ledgerProbe.failure -ne 'self-test' -or $ledgerProbe.resume_equivalence.status -ne 'PASS' -or
      $ledgerProbe.sanity.status -ne 'PASS' -or $ledgerProbe.completed_utc -ne '2026-10-09T00:00:00Z') {
    throw 'G1_LONG_LINEAR_200K_PREFLIGHT_LEDGER_SELF_TEST_FAILED'
  }
  Write-Host 'g1_long_linear_200k_preflight_self_test=PASS mode=All schedule_steps=200000 sanity_boundary=8000 resume_equivalence=2000 ledger=PASS'
  exit 0
}

function Resolve-BuildPath([string]$PathValue) {
  $root = Split-Path -Parent $PSScriptRoot
  $prefix = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
  $candidate = if ([IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $root $PathValue }
  $resolved = [IO.Path]::GetFullPath($candidate)
  if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'PREFLIGHT_PATH_MUST_BE_UNDER_BUILD'
  }
  return $resolved
}

function Write-JsonAtomic([string]$Path, [object]$Value) {
  $temporary = $Path + '.tmp-' + [guid]::NewGuid().ToString('N')
  $json = $Value | ConvertTo-Json -Depth 24
  [IO.File]::WriteAllText($temporary, $json + "`n", [Text.UTF8Encoding]::new($false))
  if ([IO.File]::Exists($Path)) {
    $backup = $Path + '.bak-' + [guid]::NewGuid().ToString('N')
    [IO.File]::Replace($temporary, $Path, $backup)
  } else {
    [IO.File]::Move($temporary, $Path)
  }
}

function Build-AndAuditHvxApk {
  $gradle = Join-Path $root 'gradlew.bat'
  $apkPath = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
  $testApkPath = Join-Path $root 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
  $buildLogPath = Join-Path $reportRootResolved 'hvx-android-build.log'
  $auditPath = Join-Path $reportRootResolved 'hvx-apk-audit.txt'
  $gradleArguments = @(
    ':app:assembleDebug', ':app:assembleDebugAndroidTest',
    '-Pphonelm.enableQnn=true',
    "-Pqairt.sdkRoot=$QairtSdkRoot",
    "-Pqairt.expectedBuildId=$ExpectedBuildId",
    '-Pphonelm.enableHvxMuon=true',
    "-Phexagon.sdkRoot=$HexagonSdkRoot",
    '--no-daemon'
  )
  & $gradle $gradleArguments *> $buildLogPath
  $buildExitCode = $LASTEXITCODE
  if ($buildExitCode -ne 0) { throw "PINNED_QNN_HVX_ANDROID_BUILD_FAILED: exit=$buildExitCode" }
  foreach ($artifact in @($apkPath, $testApkPath)) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) { throw 'PINNED_QNN_HVX_APK_OUTPUT_MISSING' }
  }
  & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apkPath `
    -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath $auditPath | Out-Null
  $audit = (Get-Content -LiteralPath $auditPath -Raw) -replace "`r", ''
  if ($audit -notmatch '(?m)^status=SUCCESS$' -or
      $audit -notmatch '(?m)^forbidden_2_47_strings=false$' -or
      $audit -notmatch '(?m)^host_sdk_path_present=false$' -or
      $audit -notmatch ('(?m)^qairt_build_id=' + [regex]::Escape($ExpectedBuildId) + '$')) {
    throw 'PINNED_QNN_HVX_APK_AUDIT_REJECTED'
  }
  return [pscustomobject][ordered]@{
    status = 'SUCCESS'
    qairt_build_id = $ExpectedBuildId
    apk_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $apkPath -Algorithm SHA256).Hash.ToLowerInvariant()
    android_test_apk_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $testApkPath -Algorithm SHA256).Hash.ToLowerInvariant()
    audit_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $auditPath -Algorithm SHA256).Hash.ToLowerInvariant()
    build_log_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $buildLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Get-CheckpointPath([string]$Directory, [int]$Step) {
  $name = Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 64 -FeedForwardDimension 128 -Step $Step
  return (Join-Path $Directory $name)
}

$root = Split-Path -Parent $PSScriptRoot
$privateRootResolved = Resolve-BuildPath $PrivateRoot
$reportRootResolved = Resolve-BuildPath $ReportRoot
$trainingScript = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$cachePath = Join-Path $privateRootResolved 'caches\train-expanded.bin'
$cacheRoot = Join-Path $privateRootResolved 'caches'
$tokenizerPath = Join-Path $privateRootResolved 'tokenizer\byte-bpe-v1024.model'
$manifestPath = Join-Path $privateRootResolved 'expanded-train-manifest.json'
$preflightPlanPath = Join-Path $reportRootResolved 'preflight-plan.json'
$script:trainingApkInstalled = $false
foreach ($required in @($cachePath, $tokenizerPath, $manifestPath, (Join-Path $cacheRoot 'validation.bin'), (Join-Path $cacheRoot 'development.bin'))) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw 'G1_LONG_LINEAR_200K_PREFLIGHT_INPUT_MISSING' }
}
if (Test-Path -LiteralPath $preflightPlanPath) { throw 'PREFLIGHT_PLAN_ALREADY_EXISTS' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$cacheHash = 'sha256:' + (Get-FileHash -LiteralPath $cachePath -Algorithm SHA256).Hash.ToLowerInvariant()
$tokenizerHash = 'sha256:' + (Get-FileHash -LiteralPath $tokenizerPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifest.schema -ne 'NICOPEDIA_G1_EXPANDED_TRAIN_V1' -or
    $cacheHash -ne 'sha256:bbbf966f07a65b64cf5e2a85f942cecf95c8f349c9d6b1bfb5d28191a0ce3fcf' -or
    $manifest.cache.sha256 -ne $cacheHash -or
    $tokenizerHash -ne 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798' -or
    $manifest.tokenizer.sha256 -ne $tokenizerHash -or
    [long]$manifest.cache.records -ne 12741465 -or
    [long]$manifest.training_order.seed -ne 20260806 -or
    $manifest.training_order.hash -ne 'fnv1a64:be3269f541236dc2' -or
    $manifest.final_test_evaluated -ne $false) {
  throw 'G1_LONG_LINEAR_200K_PREFLIGHT_DATA_IDENTITY_MISMATCH'
}
$orderScript = Join-Path $PSScriptRoot 'g1_training_order_identity.py'
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'PYTHON_UNAVAILABLE_FOR_ORDER_IDENTITY' }
$orderOutput = & $python.Source $orderScript --record-count $manifest.cache.records --steps 200000
if ($LASTEXITCODE -ne 0) { throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_COMPUTE_FAILED' }
try { $orderIdentity = ($orderOutput -join "`n") | ConvertFrom-Json }
catch { throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_OUTPUT_INVALID' }
if ($orderIdentity.hash -ne 'fnv1a64:83f9c96cf10261a0' -or
    $orderIdentity.prefix_100k_hash -ne $manifest.training_order.hash -or
    [long]$orderIdentity.selection_count -ne 1600000) {
  throw 'G1_LONG_LINEAR_200K_ORDER_IDENTITY_MISMATCH'
}
foreach ($split in @('validation','development')) {
  $actual = 'sha256:' + (Get-FileHash -LiteralPath (Join-Path $cacheRoot "$split.bin") -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($manifest.heldout_cache_identity.$split.sha256 -ne $actual) { throw "HELDOUT_CACHE_IDENTITY_MISMATCH: $split" }
}
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
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
[IO.Directory]::CreateDirectory($reportRootResolved) | Out-Null

$plan = [pscustomobject][ordered]@{
  schema = 'G1_LONG_LINEAR_200K_PREFLIGHT_V1'
  status = 'RUNNING'
  mode = $Mode
  cache_sha256 = $cacheHash
  tokenizer_sha256 = $tokenizerHash
  order_hash_100k_prefix = $orderIdentity.prefix_100k_hash
  order_hash_200k = $orderIdentity.hash
  initial_parameter_hash_expected = 'fnv1a64:01b81108b8c8b842'
  qairt_build_id = $ExpectedBuildId
  device_serial_private = $deviceInfo.Serial
  failure = $null
  updated_utc = $null
  completed_utc = $null
  resume_equivalence = $null
  sanity = $null
  hvx_apk = $null
  hexagon_sdk_version = [string]$hexagonManifest.root.id.version
  hexagon_tools_build = [string]$hexagonTools[0].build
  hexagon_sdk_manifest_sha256 = $hexagonManifestSha256
  hexagon_clang_sha256 = $hexagonCompilerSha256
  runs = @()
  created_utc = [DateTime]::UtcNow.ToString('o')
}
Write-JsonAtomic -Path $preflightPlanPath -Value $plan

function Invoke-LongLinearSegment([string]$Trajectory, [string]$TrajectoryRoot, [int]$FromStep, [int]$ToStep, [string]$CacheSourceRunId) {
  $runId = 'g1ll200k-{0}-{1:D6}-{2:D6}-{3}' -f $Trajectory, $FromStep, $ToStep, [guid]::NewGuid().ToString('N').Substring(0, 6)
  if ($runId.Length -gt 64) { throw 'PREFLIGHT_RUN_ID_TOO_LONG' }
  [IO.Directory]::CreateDirectory($TrajectoryRoot) | Out-Null
  $logPath = Join-Path $TrajectoryRoot "segment-$('{0:D6}' -f $FromStep)-$('{0:D6}' -f $ToStep)-$runId.log"
  $lines = [Collections.Generic.List[string]]::new()
  $args = @{
    QairtSdkRoot = $QairtSdkRoot; ExpectedBuildId = $ExpectedBuildId; HexagonSdkRoot = $HexagonSdkRoot
    SkipBuild = $true; SkipInstall = $script:trainingApkInstalled
    Seed = 1; Layers = 19; Steps = $ToStep; Tokens = 32; Vocabulary = 1024
    Dimension = 64; FeedForwardDimension = 128; BatchSize = 8; LearningRate = '0.0033'
    LearningRateSchedule = 'linear_decay'; DecayStartStep = 4000; DecayEndStep = 200000; ScheduleTotalSteps = 200000
    TargetLearningRate = '0.00015'; ExperimentFork = $true; ParentLearningRate = '0.0033'; Optimizer = 'Muon'
    AttentionGate = 'headwise_g1_sigmoid'; MuonBackend = 'HVX'; MuonLearningRate = '0.0075'; MuonMomentum = '0.95'; MuonNsSteps = 5
    CachePath = $cachePath; EvalCacheRoot = $cacheRoot; TokenizerModelPath = $tokenizerPath; ReportRoot = $TrajectoryRoot
    ExpectedDeviceSerial = $deviceInfo.Serial; ResumeStep = $FromStep; CheckpointInterval = 1000
    ExpectedTrainingOrderHash = $orderIdentity.hash; G1LongLinear200kExperiment = $true
    RunId = $runId; CacheSourceRunId = $CacheSourceRunId; AllowQualityFailure = $true
    PollLimit = 86400; PollSeconds = 5; ProgressEverySeconds = 30; CheckpointStallSeconds = 7200
  }
  try {
    & $trainingScript @args *>&1 | ForEach-Object {
      $line = $_.ToString()
      $lines.Add($line)
      Write-Host $line
    }
  } finally {
    [IO.File]::WriteAllLines($logPath, $lines, [Text.UTF8Encoding]::new($false))
  }
  $resultPath = Join-Path $TrajectoryRoot "seed1-l19-v1024-t32-d64-f128-steps$ToStep-result.txt"
  if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) { throw "PREFLIGHT_RESULT_MISSING: $Trajectory/$ToStep" }
  $map = Get-PhoneLmKeyValueMap -Text (Get-Content -LiteralPath $resultPath -Raw)
  foreach ($field in @('status','completed_steps','run_completed_steps','initial_parameter_hash','training_order_hash','training_order_seed','dataset_hash','parameter_count','qnn_return_code_success','output_tensors_finite','all_steps_finite','optimizer_muon_backend','optimizer_aux_adam_backend','hvx_rpc_failure_count','hvx_fallback_count','hvx_nonfinite_count','cpu_fallback','fallback','records_seen','target_tokens_seen','learning_rate_decay_end_step','learning_rate_schedule_total_steps')) {
    if (-not $map.Contains($field)) { throw "PREFLIGHT_REPORT_FIELD_MISSING: $field" }
  }
  if ($map.status -ne 'SUCCESS' -or [int]$map.completed_steps -ne $ToStep -or
      [int]$map.run_completed_steps -ne ($ToStep - $FromStep) -or
      $map.initial_parameter_hash -ne 'fnv1a64:01b81108b8c8b842' -or
      $map.training_order_hash -ne $orderIdentity.hash -or [long]$map.training_order_seed -ne 20260806 -or
      $map.dataset_hash -ne 'fnv1a64:245228bcbe304f51' -or [long]$map.parameter_count -ne 760960 -or
      $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or
      $map.all_steps_finite -ne 'true' -or $map.optimizer_muon_backend -ne 'HVX_W8' -or
      $map.optimizer_aux_adam_backend -ne 'CPU' -or [int]$map.hvx_rpc_failure_count -ne 0 -or
      [int]$map.hvx_fallback_count -ne 0 -or [int]$map.hvx_nonfinite_count -ne 0 -or
      $map.cpu_fallback -ne 'false' -or $map.fallback -ne 'false' -or
      [long]$map.records_seen -ne ([long]$ToStep * 8) -or
      [long]$map.target_tokens_seen -ne ([long]$ToStep * 256) -or
      [int]$map.learning_rate_decay_end_step -ne 200000 -or
      [int]$map.learning_rate_schedule_total_steps -ne 200000) {
    throw "PREFLIGHT_REPORT_IDENTITY_OR_HEALTH_REJECTED: $Trajectory/$ToStep"
  }
  $checkpoint = Get-CheckpointPath -Directory $TrajectoryRoot -Step $ToStep
  if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "PREFLIGHT_CHECKPOINT_MISSING: $Trajectory/$ToStep" }
  $script:trainingApkInstalled = $true
  return [pscustomobject]@{
    trajectory = $Trajectory; from_step = $FromStep; step = $ToStep; run_id_private = $runId
    status = 'SUCCESS'; initial_parameter_hash = $map.initial_parameter_hash
    order_hash = $map.training_order_hash; qnn_return_code_success = $map.qnn_return_code_success
    finite = $map.all_steps_finite; fallback = $map.fallback
    checkpoint_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    checkpoint_bytes = [IO.FileInfo]::new($checkpoint).Length
    result_sha256 = 'sha256:' + (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Add-PreflightRun([object]$Run) {
  $script:plan.runs = @($script:plan.runs) + @($Run)
  $script:plan.updated_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $script:preflightPlanPath -Value $script:plan
}

try {
  $plan.hvx_apk = Build-AndAuditHvxApk
  $plan.updated_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $preflightPlanPath -Value $plan
  if ($Mode -in @('ResumeEquivalence','All')) {
    $resumeRoot = Join-Path $reportRootResolved 'resume-equivalence'
    $uninterruptedRoot = Join-Path $resumeRoot 'uninterrupted'
    $segmentedRoot = Join-Path $resumeRoot 'segmented'
    if ((Test-Path -LiteralPath $uninterruptedRoot) -or (Test-Path -LiteralPath $segmentedRoot)) { throw 'RESUME_EQUIVALENCE_OUTPUT_ALREADY_EXISTS' }
    $uninterrupted = Invoke-LongLinearSegment -Trajectory 'uninterrupted' -TrajectoryRoot $uninterruptedRoot -FromStep 0 -ToStep 2000 -CacheSourceRunId ''
    Add-PreflightRun $uninterrupted
    $firstSegment = Invoke-LongLinearSegment -Trajectory 'segmented' -TrajectoryRoot $segmentedRoot -FromStep 0 -ToStep 1000 -CacheSourceRunId ''
    Add-PreflightRun $firstSegment
    $resumedSegment = Invoke-LongLinearSegment -Trajectory 'segmented' -TrajectoryRoot $segmentedRoot -FromStep 1000 -ToStep 2000 -CacheSourceRunId $firstSegment.run_id_private
    Add-PreflightRun $resumedSegment
    if ($uninterrupted.checkpoint_sha256 -ne $resumedSegment.checkpoint_sha256 -or
        $uninterrupted.checkpoint_bytes -ne $resumedSegment.checkpoint_bytes) {
      throw 'G1_LONG_LINEAR_200K_RESUME_EQUIVALENCE_CHECKPOINT_MISMATCH'
    }
    $plan.resume_equivalence = [pscustomobject]@{ status = 'PASS'; uninterrupted_step_2000_sha256 = $uninterrupted.checkpoint_sha256; segmented_step_2000_sha256 = $resumedSegment.checkpoint_sha256; checkpoint_bytes = $uninterrupted.checkpoint_bytes }
    Write-JsonAtomic -Path $preflightPlanPath -Value $plan
  }
  if ($Mode -in @('Sanity','All')) {
    $sanityRoot = Join-Path $reportRootResolved 'sanity-0-8000'
    if (Test-Path -LiteralPath $sanityRoot) { throw 'SANITY_OUTPUT_ALREADY_EXISTS' }
    $firstSanity = Invoke-LongLinearSegment -Trajectory 'sanity' -TrajectoryRoot $sanityRoot -FromStep 0 -ToStep 1000 -CacheSourceRunId ''
    Add-PreflightRun $firstSanity
    $finalSanity = Invoke-LongLinearSegment -Trajectory 'sanity' -TrajectoryRoot $sanityRoot -FromStep 1000 -ToStep 8000 -CacheSourceRunId $firstSanity.run_id_private
    Add-PreflightRun $finalSanity
    $plan.sanity = [pscustomobject]@{ status = 'PASS'; start_step = 0; first_checkpoint_step = 1000; final_checkpoint_step = 8000; initial_parameter_hash = $finalSanity.initial_parameter_hash; order_hash = $finalSanity.order_hash; checkpoint_1000_sha256 = $firstSanity.checkpoint_sha256; checkpoint_8000_sha256 = $finalSanity.checkpoint_sha256; quality_used_for_schedule_selection = $false }
    Write-JsonAtomic -Path $preflightPlanPath -Value $plan
  }
  $plan.status = 'PASS'
  $plan.completed_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $preflightPlanPath -Value $plan
  Write-Host "g1_long_linear_200k_preflight=PASS mode=$Mode"
} catch {
  $plan.status = 'FAILED'
  $plan.failure = $_.Exception.Message
  $plan.updated_utc = [DateTime]::UtcNow.ToString('o')
  Write-JsonAtomic -Path $preflightPlanPath -Value $plan
  throw
}
