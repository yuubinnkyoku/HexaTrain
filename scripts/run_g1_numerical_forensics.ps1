# SPDX-License-Identifier: Apache-2.0
# Fork a verified finite state into an independent, bounded diagnostic run.
param(
  [Parameter(Mandatory=$true)][string]$CheckpointPath,
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$CheckpointSha256,
  [Parameter(Mandatory=$true)][ValidateRange(40000,44999)][int]$FromStep,
  [Parameter(Mandatory=$true)][ValidateRange(40001,45000)][int]$ToStep,
  [ValidateSet('NONE','COARSE_LAYER_BOUNDARIES','NICOPEDIA_FINE')][string]$TapScope='NONE',
  [ValidateRange(0,18)][int]$LayerIndex=18,
  [switch]$Install
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')
$sdk='C:\Qualcomm\AIStack\QAIRT\2.48.40.260702'
$buildId='2.48.40.260702151143'
$hexagon='C:\Qualcomm\Hexagon_SDK\6.6.0.0'
if ($ToStep -le $FromStep -or $ToStep-$FromStep -gt 2000) { throw 'FORENSIC_BOUND_EXCEEDED' }
$sourceCheckpoint=[IO.Path]::GetFullPath((Join-Path $root $CheckpointPath))
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'build'))+[IO.Path]::DirectorySeparatorChar
if (-not $sourceCheckpoint.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { throw 'PRIVATE_CHECKPOINT_REQUIRED' }
if ((Get-FileHash -LiteralPath $sourceCheckpoint).Hash.ToLowerInvariant() -ne $CheckpointSha256) { throw 'CHECKPOINT_IDENTITY_MISMATCH' }
if ($FromStep -eq 40000 -and $CheckpointSha256 -ne '89f197d3085f496a22a9ab4ef2161bf2644ba9311b4571faed55f60a7959243d') { throw 'PROTECTED_40K_IDENTITY_MISMATCH' }
$planPath=Join-Path $root 'build/reports/g1-long-linear-200k-primary/run-plan.json'
$protectedPlanHash=(Get-FileHash $planPath).Hash
$plan=Get-Content $planPath -Raw | ConvertFrom-Json
if ($plan.status -ne 'UNSTABLE') { throw 'ORIGINAL_INCIDENT_IDENTITY_MISMATCH' }
$hexManifestHash=(Get-FileHash (Join-Path $hexagon 'hexagon_sdk.json')).Hash.ToLowerInvariant()
$hexCompilerHash=(Get-FileHash (Join-Path $hexagon 'tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang.exe')).Hash.ToLowerInvariant()
if ('sha256:'+$hexManifestHash -ne $plan.hexagon_sdk_manifest_sha256 -or 'sha256:'+$hexCompilerHash -ne $plan.hexagon_clang_sha256) { throw 'HVX_IDENTITY_MISMATCH' }
$runId='g1forensic-'+[guid]::NewGuid().ToString('N')
$report=Join-Path $root "build/reports/g1-long-linear-forensics/$runId"
if (Test-Path $report) { throw 'RUN_NAMESPACE_EXISTS' }
[IO.Directory]::CreateDirectory($report) | Out-Null
$sourceSha=(& git -C $root rev-parse HEAD).Trim()
if (& git -C $root status --porcelain) { throw 'FORENSIC_SOURCE_MUST_BE_COMMITTED' }
$hostFree=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($root)).AvailableFreeSpace
if ($hostFree -lt 20GB) { throw 'HOST_STORAGE_HEADROOM_LOW' }
$adb=Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'
$device=Resolve-PhoneLmDevice -Adb $adb
if ($device.Serial -ne $plan.device_serial_private) { throw 'DEVICE_IDENTITY_MISMATCH' }
Assert-PhoneLmPhysicalDevice -Adb $adb -Device $device.Endpoint
Assert-PhoneLmNoExistingRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
Assert-PhoneLmNoExistingHeadlessRun -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm
$df=(Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint -Arguments @('shell','df','-k','/data/user/0')).Text
$dfLine=@($df -split "`n" | Where-Object {$_ -match '^\S+\s+\d+\s+\d+\s+\d+\s+\d+%'})
if ($dfLine.Count -ne 1) { throw 'DEVICE_STORAGE_STATUS_UNAVAILABLE' }
$deviceFree=[long](($dfLine[0] -split '\s+')[3])*1024L
if ($deviceFree -lt 12GB) { throw 'DEVICE_STORAGE_HEADROOM_LOW' }
& (Join-Path $PSScriptRoot 'check_qairt.ps1') -SdkRoot $sdk -ExpectedBuildId $buildId *> (Join-Path $report 'qairt-check.txt')
if ($LASTEXITCODE -notin @(0,3)) { throw 'QAIRT_GATE_FAILED' }
$apk=Join-Path $root 'app/build/outputs/apk/debug/app-debug.apk'
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk -QairtSdkRoot $sdk -ExpectedBuildId $buildId -ReportPath (Join-Path $report 'apk-audit.txt') | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'APK_AUDIT_FAILED' }
$checkpointName=Get-PhoneLmCheckpointName -Seed 1 -Layers 19 -Tokens 32 -Dimension 64 -FeedForwardDimension 128 -Step $FromStep
Copy-Item -LiteralPath $sourceCheckpoint -Destination (Join-Path $report $checkpointName)
$ledger=[ordered]@{
  schema='G1_NUMERICAL_FORENSICS_V1'; run_id_private=$runId; source_sha=$sourceSha
  source_checkpoint_sha256=$CheckpointSha256; from_step=$FromStep; to_step=$ToStep
  tap_scope=$TapScope; layer_index=$LayerIndex; status='RUNNING'; classification='UNRESOLVED'
  qairt_build_id=$buildId; apk_sha256=(Get-FileHash $apk).Hash.ToLowerInvariant()
  hexagon_manifest_sha256=$hexManifestHash; hexagon_compiler_sha256=$hexCompilerHash
  device_identity_private=$device.Serial; host_free_bytes=$hostFree; device_free_bytes=$deviceFree
  original_plan_sha256=$protectedPlanHash; order_hash='fnv1a64:83f9c96cf10261a0'
  started_utc=[DateTime]::UtcNow.ToString('o'); failure=$null
}
function Save-Ledger { [IO.File]::WriteAllText((Join-Path $report 'forensic-plan.json'),($ledger | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false)) }
Save-Ledger
$arguments=@{
  QairtSdkRoot=$sdk; ExpectedBuildId=$buildId; HexagonSdkRoot=$hexagon; SkipBuild=$true
  SkipInstall=(-not $Install); Seed=1; Layers=19; Steps=$ToStep; Tokens=32; Vocabulary=1024
  Dimension=64; FeedForwardDimension=128; BatchSize=8; LearningRate='0.0033'
  LearningRateSchedule='linear_decay'; DecayStartStep=4000; DecayEndStep=200000; ScheduleTotalSteps=200000
  ExpectedTrainingOrderHash='fnv1a64:83f9c96cf10261a0'; G1LongLinear200kExperiment=$true
  TargetLearningRate='0.00015'; ExperimentFork=$true; ParentLearningRate='0.0033'; Optimizer='Muon'
  AttentionGate='headwise_g1_sigmoid'; MuonBackend='HVX'; MuonLearningRate='0.0075'; MuonMomentum='0.95'; MuonNsSteps=5
  CachePath=(Join-Path $root 'build/private-data/g1-long-linear-200k/caches/train-expanded.bin')
  EvalCacheRoot=(Join-Path $root 'build/private-data/g1-long-linear-200k/caches')
  TokenizerModelPath=(Join-Path $root 'build/private-data/g1-long-linear-200k/tokenizer/byte-bpe-v1024.model')
  ReportRoot=$report; ExpectedDeviceSerial=$device.Serial; ResumeStep=$FromStep; CheckpointInterval=1000
  RunId=$runId; CacheSourceRunId=$plan.first_cache_run_id_private; AllowQualityFailure=$true
  PollSeconds=2; ProgressEverySeconds=30; CheckpointStallSeconds=7200
  NumericalForensics=$true; ForensicTapScope=$TapScope; ForensicLayerIndex=$LayerIndex
}
try {
  & (Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1') @arguments *> (Join-Path $report 'runner.log')
  $ledger.status='FINITE_BOUND_COMPLETE'; $ledger.classification='NO_FAILURE_IN_BOUND'
} catch {
  $ledger.status='FAILED'; $ledger.failure=$_.Exception.Message
  $resultPath=Join-Path $report "seed1-l19-v1024-t32-d64-f128-steps$ToStep-result.txt"
  if ((Test-Path $resultPath) -and (Get-Content $resultPath -Raw) -match 'generalized tiny training APP_READ residual poison=0, nonfinite=[1-9]') {
    $ledger.classification='REPRODUCED_NONFINITE'
  } else { $ledger.classification='BLOCKED_UNCLASSIFIED_FAILURE' }
  foreach ($name in @('forensic-pre-update.ckpt','forensic-failure.csv','forensic-failure.txt','learning-rate-telemetry.csv','optimizer-health-telemetry.csv')) {
    Receive-PhoneLmBinary -Adb $adb -Device $device.Endpoint -Package com.yuubinnkyoku.phonelm `
      -RemotePath "files/headless-input/$runId/$name" -LocalPath (Join-Path $report $name) -MinimumBytes 1 | Out-Null
  }
} finally {
  if ((Get-FileHash $planPath).Hash -ne $protectedPlanHash -or (Get-FileHash $sourceCheckpoint).Hash.ToLowerInvariant() -ne $CheckpointSha256) {
    $ledger.status='BLOCKED'; $ledger.classification='PROTECTED_EVIDENCE_IDENTITY_MISMATCH'
  }
  $ledger['completed_utc']=[DateTime]::UtcNow.ToString('o'); Save-Ledger
}
Write-Host "forensic_status=$($ledger.status) classification=$($ledger.classification) private_report=$report"
if ($ledger.classification -eq 'BLOCKED_UNCLASSIFIED_FAILURE' -or $ledger.status -eq 'BLOCKED') { throw $ledger.classification }
