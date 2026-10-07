# SPDX-License-Identifier: Apache-2.0
# Prepared, fail-closed physical-device audit. This script is not run by host verification.
param(
  [Parameter(Mandatory=$true)][ValidateSet('BeforeOnly300vs400','MatchedAB')][string]$Mode,
  [Parameter(Mandatory=$true)][string]$QairtSdkRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedBuildId,
  [Parameter(Mandatory=$true)][string]$HexagonSdkRoot,
  [Parameter(Mandatory=$true)][string]$BeforeApkPath,
  [Parameter(Mandatory=$true)][string]$BeforeAndroidTestApkPath,
  [string]$CandidateApkPath = '',
  [string]$CandidateAndroidTestApkPath = '',
  [ValidateRange(1,4)][int]$PairCount = 4,
  [ValidateRange(1,999)][int]$PairStart = 1,
  [ValidatePattern('^[A-Za-z0-9._-]{1,32}$')][string]$AuditId = (Get-Date -Format 'yyyyMMdd-HHmmss')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

function Resolve-AuditPath([string]$Path) {
  if (-not $Path) { return '' }
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  return [IO.Path]::GetFullPath((Join-Path $root $Path))
}
function Get-ZipEntrySha256([string]$ArchivePath, [string]$EntryName) {
  $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
  try {
    $entry = $archive.Entries | Where-Object { $_.FullName -ceq $EntryName } | Select-Object -First 1
    if ($null -eq $entry) { return 'NOT_PRESENT' }
    $stream = $entry.Open()
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose(); $stream.Dispose() }
  } finally { $archive.Dispose() }
}
function Get-ApkIdentity([string]$AppPath, [string]$TestPath, [string]$Arm, [string]$DeclaredCommit) {
  if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf) -or -not (Test-Path -LiteralPath $TestPath -PathType Leaf)) { throw 'AUDIT_APK_MISSING' }
  $appFile = Get-Item -LiteralPath $AppPath
  $testFile = Get-Item -LiteralPath $TestPath
  $fingerprint = 'NOT_EMBEDDED'
  $embedded = $null
  $zip = [IO.Compression.ZipFile]::OpenRead($AppPath)
  try {
    $entry = $zip.Entries | Where-Object { $_.FullName -ceq 'assets/phonelm-build-provenance.json' } | Select-Object -First 1
    if ($null -ne $entry) {
      $reader = [IO.StreamReader]::new($entry.Open())
      try { $embedded = $reader.ReadToEnd() | ConvertFrom-Json -AsHashtable; $fingerprint = $embedded | ConvertTo-Json -Compress -Depth 8 }
      finally { $reader.Dispose() }
    }
  } finally { $zip.Dispose() }
  $nativeHash = Get-ZipEntrySha256 $AppPath 'lib/arm64-v8a/libphonelm_native.so'
  $skelHash = Get-ZipEntrySha256 $AppPath 'assets/qnn/libQnnHtpV81Skel.so'
  if ($null -ne $embedded) {
    foreach ($field in @('git_commit_sha','git_tree_sha','dirty','branch','build_timestamp_utc','cmake_config_sha256','compile_flag_fingerprint','qairt_build_id','hvx_skel_sha256','native_library_sha256')) {
      if (-not $embedded.ContainsKey($field)) { throw 'APK_BUILD_FINGERPRINT_INCOMPLETE' }
    }
    $declaredTree = (& git -C $root rev-parse ($DeclaredCommit + '^{tree}') 2>$null).Trim()
    if ([string]$embedded.git_commit_sha -ne $DeclaredCommit -or [string]$embedded.git_tree_sha -ne $declaredTree -or [string]$embedded.qairt_build_id -ne $ExpectedBuildId -or
        [string]$embedded.native_library_sha256 -ne $nativeHash -or [string]$embedded.hvx_skel_sha256 -ne $skelHash -or
        [string]$embedded.dirty -match '^(?i:true|1)$') { throw 'APK_BUILD_FINGERPRINT_MISMATCH' }
  }
  return [ordered]@{
    arm = $Arm
    declared_source_commit = $DeclaredCommit
    source_provenance_level = if ($fingerprint -eq 'NOT_EMBEDDED') { 'DECLARED_UNVERIFIED' } else { 'EMBEDDED_FINGERPRINT_PRESENT' }
    app_apk_filename = $appFile.Name
    app_apk_sha256 = (Get-FileHash -LiteralPath $AppPath -Algorithm SHA256).Hash.ToLowerInvariant()
    app_apk_file_timestamp_utc = $appFile.LastWriteTimeUtc.ToString('o')
    android_test_apk_filename = $testFile.Name
    android_test_apk_sha256 = (Get-FileHash -LiteralPath $TestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    native_library_sha256 = $nativeHash
    qairt_hvx_skel_sha256 = $skelHash
    hvx_probe_skel_sha256 = Get-ZipEntrySha256 $AppPath 'assets/hvx/libhexatrain_hvx_probe_skel.so'
    embedded_build_fingerprint = $fingerprint
    qairt_build_id = $ExpectedBuildId
  }
}
function Get-HostCheckoutIdentity {
  $branch = (& git -C $root branch --show-current 2>$null).Trim()
  $commit = (& git -C $root rev-parse HEAD 2>$null).Trim()
  $tree = (& git -C $root rev-parse 'HEAD^{tree}' 2>$null).Trim()
  $status = (& git -C $root status --porcelain 2>$null) -join "`n"
  $dirty = -not [string]::IsNullOrWhiteSpace($status)
  $cacheHash = 'UNKNOWN'
  $compileFingerprint = 'UNKNOWN'
  $cache = Get-ChildItem -LiteralPath (Join-Path $root 'app\build\.cxx') -Filter CMakeCache.txt -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cache) { $cacheHash = (Get-FileHash -LiteralPath $cache.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
  $compile = Get-ChildItem -LiteralPath (Join-Path $root 'build') -Filter compile_commands.json -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'candidate|phonelm' } | Select-Object -First 1
  if ($compile) {
    try {
      $commands = Get-Content -LiteralPath $compile.FullName -Raw | ConvertFrom-Json
      $target = $commands | Where-Object { $_.file -match 'tiny_language_model_cpu\.cpp$' } | Select-Object -First 1
      if ($target) { $bytes = [Text.Encoding]::UTF8.GetBytes([string]$target.command); $sha = [Security.Cryptography.SHA256]::Create(); try { $compileFingerprint = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() } finally { $sha.Dispose() } }
    } catch { $compileFingerprint = 'UNKNOWN' }
  }
  return [ordered]@{ branch = $branch; commit = $commit; tree = $tree; dirty = $dirty; cmake_config_sha256 = $cacheHash; compile_flag_fingerprint = $compileFingerprint; scope = 'host checkout; does not prove APK source without embedded fingerprint' }
}
function Get-InstalledVersion([string]$Device, [string]$Package) {
  $result = Get-PhoneLmAdbResult -Adb $adb -Device $Device -Arguments @('shell','dumpsys','package',$Package) -TimeoutSeconds 60
  if ($result.ExitCode -ne 0) { return [ordered]@{ version_code = 'UNKNOWN'; version_name = 'UNKNOWN' } }
  $code = [regex]::Match($result.Text, '(?m)^\s*versionCode=(\d+)').Groups[1].Value
  $name = [regex]::Match($result.Text, '(?m)^\s*versionName=([^\s]+)').Groups[1].Value
  return [ordered]@{ version_code = if ($code) { $code } else { 'UNKNOWN' }; version_name = if ($name) { $name } else { 'UNKNOWN' } }
}
function Write-Manifest([string]$Path, [System.Collections.IDictionary]$Manifest) {
  [IO.File]::WriteAllText($Path, ($Manifest | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}

$root = Split-Path -Parent $PSScriptRoot
$beforeApp = Resolve-AuditPath $BeforeApkPath
$beforeTest = Resolve-AuditPath $BeforeAndroidTestApkPath
$candidateApp = Resolve-AuditPath $CandidateApkPath
$candidateTest = Resolve-AuditPath $CandidateAndroidTestApkPath
$mainCommit = '98d7c01f0904345a21d87ae6726dafda066d0b6a'
$candidateCommit = '485651bfd64dd173af70654ed635e7a15a519085'
$branch = (& git -C $root branch --show-current 2>$null).Trim()
if ($branch -ne 'codex/training-critical-path-v2') { throw 'AUDIT_BRANCH_MISMATCH' }
& git -C $root merge-base --is-ancestor $candidateCommit HEAD 2>$null
if ($LASTEXITCODE -ne 0) { throw 'AUDIT_CANDIDATE_COMMIT_NOT_ANCESTOR' }
$runtimeDiff = (& git -C $root diff --quiet $candidateCommit -- app/src/main/cpp 2>$null); if ($LASTEXITCODE -ne 0) { throw 'AUDIT_RUNTIME_SOURCE_CHANGED_AFTER_CANDIDATE' }
$runtimeStagedDiff = (& git -C $root diff --cached --quiet $candidateCommit -- app/src/main/cpp 2>$null); if ($LASTEXITCODE -ne 0) { throw 'AUDIT_STAGED_RUNTIME_SOURCE_CHANGED_AFTER_CANDIDATE' }
if ((& git -C $root status --porcelain -- app/src/main/cpp 2>$null)) { throw 'AUDIT_UNTRACKED_OR_DIRTY_RUNTIME_SOURCE' }
if ($Mode -eq 'MatchedAB') {
  if (-not $candidateApp -or -not $candidateTest) { throw 'AUDIT_CANDIDATE_ARTIFACTS_REQUIRED' }
  if ($beforeApp -eq $candidateApp) { throw 'AUDIT_APK_PATHS_MUST_BE_EXPLICIT' }
}
$beforeIdentity = Get-ApkIdentity $beforeApp $beforeTest 'before' $mainCommit
$candidateIdentity = if ($candidateApp) { Get-ApkIdentity $candidateApp $candidateTest 'candidate' $candidateCommit } else { $null }
if ($Mode -eq 'MatchedAB' -and $beforeIdentity.app_apk_sha256 -eq $candidateIdentity.app_apk_sha256) { throw 'AUDIT_APP_APK_IDENTITIES_MUST_DIFFER' }
if ($Mode -eq 'MatchedAB' -and $beforeIdentity.android_test_apk_sha256 -ne $candidateIdentity.android_test_apk_sha256) { throw 'AUDIT_ANDROID_TEST_APK_IDENTITIES_MUST_MATCH' }
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'ADB_UNAVAILABLE' }
if (-not (Test-Path -LiteralPath $HexagonSdkRoot -PathType Container)) { throw 'HEXAGON_SDK_UNAVAILABLE' }
$deviceInfo = Resolve-PhoneLmDevice -Adb $adb
$device = $deviceInfo.Endpoint
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device
$serialBytes = [Text.Encoding]::UTF8.GetBytes([string]$deviceInfo.Serial)
$serialSha = [Security.Cryptography.SHA256]::Create()
try { $deviceIdentityHash = ([BitConverter]::ToString($serialSha.ComputeHash($serialBytes))).Replace('-', '').ToLowerInvariant() } finally { $serialSha.Dispose() }
$auditRoot = Join-Path $root "build\reports\training-critical-path-device-audit\$AuditId"
$allowedAuditRoot = [IO.Path]::GetFullPath((Join-Path $root 'build')) + [IO.Path]::DirectorySeparatorChar
$auditRoot = [IO.Path]::GetFullPath($auditRoot)
if (-not $auditRoot.StartsWith($allowedAuditRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'AUDIT_OUTPUT_MUST_BE_UNDER_BUILD' }
if (Test-Path -LiteralPath $auditRoot) { throw 'AUDIT_ID_REUSE' }
[IO.Directory]::CreateDirectory($auditRoot) | Out-Null
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $beforeApp -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $auditRoot 'qnn-apk-before.txt')
if ($candidateApp) {
  & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $candidateApp -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId -ReportPath (Join-Path $auditRoot 'qnn-apk-candidate.txt')
}
$hostCheckout = Get-HostCheckoutIdentity
$runSpecs = [System.Collections.Generic.List[object]]::new()
if ($Mode -eq 'BeforeOnly300vs400') {
  @(@{ steps = 300; order = 1 }, @{ steps = 400; order = 2 }, @{ steps = 400; order = 3 }, @{ steps = 300; order = 4 }) | ForEach-Object {
    $balancedPair = [math]::Ceiling($_.order / 2)
    $runSpecs.Add([pscustomobject]@{ arm='before'; model='control'; steps=$_.steps; repetition=$_.order; pair_id="steps-pair-$balancedPair"; order=$_.order })
  }
} else {
  for ($rep = $PairStart; $rep -lt ($PairStart + $PairCount); $rep++) {
    if (($rep % 2) -eq 1) { $sequence = @(@('before','control'),@('before','g1'),@('candidate','g1'),@('candidate','control')) }
    else { $sequence = @(@('candidate','g1'),@('candidate','control'),@('before','control'),@('before','g1')) }
    foreach ($item in $sequence) { $runSpecs.Add([pscustomobject]@{ arm=$item[0]; model=$item[1]; steps=400; repetition=$rep; pair_id="r$rep-$($item[1])"; order=($runSpecs.Count + 1) }) }
  }
}
$lastAppHash = ''
$lastTestHash = ''
$runIndex = 0
foreach ($spec in $runSpecs) {
  $runIndex++
  $artifact = if ($spec.arm -eq 'before') { $beforeIdentity } else { $candidateIdentity }
  $appPath = if ($spec.arm -eq 'before') { $beforeApp } else { $candidateApp }
  $testPath = if ($spec.arm -eq 'before') { $beforeTest } else { $candidateTest }
  $runId = "${AuditId}-$($spec.arm)-$($spec.model)-s$($spec.steps)-r$($spec.repetition)"
  if ($runId.Length -gt 64) { throw 'AUDIT_RUN_ID_TOO_LONG' }
  $runRoot = Join-Path $auditRoot $runId
  if (Test-Path -LiteralPath $runRoot) { throw 'AUDIT_RUN_ID_COLLISION' }
  [IO.Directory]::CreateDirectory($runRoot) | Out-Null
  $telemetryRoot = Join-Path $runRoot 'telemetry'
  [IO.Directory]::CreateDirectory($telemetryRoot) | Out-Null
  $manifestPath = Join-Path $runRoot 'run-manifest.json'
  $manifest = [ordered]@{
    schema_version = 1; audit_id = $AuditId; audit_mode = $Mode; run_id = $runId
    arm = $spec.arm; model = $spec.model; pair_id = $spec.pair_id; repetition = $spec.repetition; run_order = $runIndex; steps = $spec.steps
    status = 'STARTING'; failure_class = $null; started_utc = [DateTimeOffset]::UtcNow.ToString('o'); finished_utc = $null
    result_file = "seed2-l19-v1024-t32-d64-f128-steps$($spec.steps)-result.txt"
    recipe = [ordered]@{ seed=2; layers=19; tokens=32; vocabulary=1024; dimension=64; feed_forward_dimension=128; batch_size=8; optimizer='Muon'; muon_backend='HVX'; muon_learning_rate='0.0075'; aux_adam_learning_rate='0.0033'; schedule='linear_decay'; decay_start=4000; decay_end=8000; schedule_total_steps=8000; target_aux_adam_learning_rate='0.00015'; checkpoint_interval=100; workers=8; allow_quality_failure=$true }
    artifact = $artifact; host_checkout = $hostCheckout; device_identity_sha256 = $deviceIdentityHash
    device_package_version = [ordered]@{ version_code='PENDING'; version_name='PENDING' }
    telemetry = [ordered]@{ pre='telemetry/pre.json'; mid='telemetry/mid.json'; post='telemetry/post.json' }
    exclusion_policy = @('transport_failure','stale_heartbeat','process_disappearance','incomplete_report','qnn_or_hvx_failure','fallback','nonfinite','identity_mismatch')
    performance_based_exclusion_allowed = $false
  }
  Write-Manifest $manifestPath $manifest
  $installedThisArm = ($artifact.app_apk_sha256 -ne $lastAppHash -or $artifact.android_test_apk_sha256 -ne $lastTestHash)
  $failureStage = 'pre_run_gates'
  try {
    Assert-PhoneLmNoExistingRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
    Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device -Package 'com.yuubinnkyoku.phonelm'
    $failureStage = 'pre_telemetry'
    & (Join-Path $PSScriptRoot 'capture_android_cpu_telemetry.ps1') -AdbPath $adb -Device $device -Package 'com.yuubinnkyoku.phonelm' -Phase pre -OutputPath (Join-Path $telemetryRoot 'pre.json') -RunId $runId | Out-Null
    $runnerArgs = @{
      QairtSdkRoot=$QairtSdkRoot; ExpectedBuildId=$ExpectedBuildId; SkipBuild=$true; AppApkPath=$appPath; AndroidTestApkPath=$testPath
      Seed=2; Layers=19; Steps=$spec.steps; Tokens=32; Vocabulary=1024; Dimension=64; FeedForwardDimension=128; BatchSize=8
      LearningRate='0.0033'; LearningRateSchedule='linear_decay'; DecayStartStep=4000; DecayEndStep=8000; ScheduleTotalSteps=8000; TargetLearningRate='0.00015'
      ExperimentFork=$true; ParentLearningRate='0.0033'; Optimizer='Muon'; MuonBackend='HVX'; HexagonSdkRoot=$HexagonSdkRoot; MuonLearningRate='0.0075'; MuonMomentum='0.95'; MuonNsSteps=5
      AttentionGate=$(if ($spec.model -eq 'g1') { 'headwise_g1_sigmoid' } else { 'none' }); CheckpointInterval=100; AllowQualityFailure=$true
      PollLimit=900; PollSeconds=2; ProgressEverySeconds=30; CheckpointStallSeconds=300; RunId=$runId; ReportRoot=$runRoot
      AuditTelemetryDirectory=$telemetryRoot; MidTelemetryAfterSeconds=[math]::Max(30,[math]::Floor($spec.steps * 0.15)); ExpectedDeviceSerial=[string]$deviceInfo.Serial
    }
    if (-not $installedThisArm) { $runnerArgs.SkipInstall = $true }
    $failureStage = 'training_runner'
    & (Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1') @runnerArgs
    $failureStage = 'post_run_package_identity'
    $manifest.device_package_version = Get-InstalledVersion $device 'com.yuubinnkyoku.phonelm'
    $manifest.status = 'COMPLETED'
    $manifest.failure_class = $null
    $lastAppHash = $artifact.app_apk_sha256
    $lastTestHash = $artifact.android_test_apk_sha256
  } catch {
    $failureRecord = $_
    $reason = $failureRecord.Exception.Message
    $manifest.status = 'FAILED'
    $manifest.failure_stage = $failureStage
    $manifest.failure_details_file = 'failure-details.json'
    if ($reason -match 'ADB_TRANSPORT') { $manifest.failure_class = 'TRANSPORT_FAILURE' }
    elseif ($reason -match 'HEARTBEAT|STALE') { $manifest.failure_class = 'STALE_HEARTBEAT' }
    elseif ($reason -match 'IDENTITY|APK_PROVENANCE') { $manifest.failure_class = 'IDENTITY_MISMATCH' }
    elseif ($reason -match 'PROCESS|RUN_STATE') { $manifest.failure_class = 'PROCESS_DISAPPEARANCE_OR_STATE' }
    else { $manifest.failure_class = 'RUNNER_FAILURE' }
    [IO.File]::WriteAllText((Join-Path $runRoot 'failure-class.txt'), [string]$manifest.failure_class, [Text.UTF8Encoding]::new($false))
    $invocation = $failureRecord.InvocationInfo
    $exception = $failureRecord.Exception
    $failureDetails = [ordered]@{
      schema_version = 1
      run_id = $runId
      operation = $failureStage
      captured_utc = [DateTimeOffset]::UtcNow.ToString('o')
      exception_type = if ($exception) { $exception.GetType().FullName } else { 'UNKNOWN' }
      exception_message = if ($exception) { $exception.Message } else { [string]$failureRecord }
      exception_full_text = if ($exception) { $exception.ToString() } else { $null }
      exception_stack_trace = if ($exception) { $exception.StackTrace } else { $null }
      powershell_script_stack_trace = $failureRecord.ScriptStackTrace
      fully_qualified_error_id = $failureRecord.FullyQualifiedErrorId
      error_category = if ($failureRecord.CategoryInfo) { [string]$failureRecord.CategoryInfo.Category } else { $null }
      error_reason = if ($failureRecord.CategoryInfo) { $failureRecord.CategoryInfo.Reason } else { $null }
      error_target_name = if ($failureRecord.CategoryInfo) { $failureRecord.CategoryInfo.TargetName } else { $null }
      script_name = if ($invocation) { $invocation.ScriptName } else { $null }
      script_line_number = if ($invocation) { $invocation.ScriptLineNumber } else { $null }
      script_line = if ($invocation) { $invocation.Line } else { $null }
      script_position_message = if ($invocation) { $invocation.PositionMessage } else { $null }
    }
    $failureJson = $failureDetails | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText((Join-Path $runRoot 'failure-details.json'), $failureJson, [Text.UTF8Encoding]::new($false))
    try { & (Join-Path $PSScriptRoot 'capture_android_cpu_telemetry.ps1') -AdbPath $adb -Device $device -Package 'com.yuubinnkyoku.phonelm' -Phase post -OutputPath (Join-Path $telemetryRoot 'post.json') -RunId $runId | Out-Null } catch { }
    $manifest.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    Write-Manifest $manifestPath $manifest
    throw "DEVICE_AUDIT_STOPPED_WITHOUT_RETRY: run=$runId class=$($manifest.failure_class)"
  }
  try { & (Join-Path $PSScriptRoot 'capture_android_cpu_telemetry.ps1') -AdbPath $adb -Device $device -Package 'com.yuubinnkyoku.phonelm' -Phase post -OutputPath (Join-Path $telemetryRoot 'post.json') -RunId $runId | Out-Null } catch {
    [IO.File]::WriteAllText((Join-Path $telemetryRoot 'post-error.txt'), 'NOT_AVAILABLE', [Text.UTF8Encoding]::new($false))
  }
  $manifest.finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
  Write-Manifest $manifestPath $manifest
  Write-Host "audit_run_complete order=$runIndex arm=$($spec.arm) model=$($spec.model) steps=$($spec.steps)"
}
Write-Host "device_audit_ready_for_analysis=$auditRoot"
