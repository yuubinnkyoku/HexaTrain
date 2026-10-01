# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Incident diagnostic runner for the QAIRT_GRAPH_ERROR_ABORTED (6031)
# investigation.  Investigation record: docs/g1-1p5x-multiseed-tier2-incident.md
#
# THIS IS NOT A G1 QUALITY RUN.  It exists to narrow down which subsystem and
# boundary condition produces 6031, using reproducible evidence.  It therefore:
#
#   * keeps every condition under which 6031 has already been observed
#     (seed 2, Control, batch 8, L19/H2, Muon with the HVX FastRPC backend,
#     heartbeat on, progress/status writes on, host polling on),
#   * adds observation only (opt-in incident trace + logcat capture),
#   * disables quality evaluation and minimizes checkpoints,
#   * writes exclusively under an incident namespace under build/, and
#     refuses to write anywhere under docs/results/.
#
# It never computes R1-R5 and never feeds the G1 decision.
#
# Modes:
#   Plan      print the resolved command and the guard checks; touch nothing
#   Run       execute one diagnostic run (requires an attached device)
#   Analyze   run the incident analyzer over a finished diagnostic directory
#   SelfTest  run the analyzer's synthetic fixture battery (no device)
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('Plan', 'Run', 'Analyze', 'SelfTest')]
  [string]$Mode,

  [int]$Seed = 2,
  [ValidateRange(1, 100000)][int]$Steps = 128,
  [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
  [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
  [string]$CachePath = 'build/private-data/nicopedia-real-text-bpe-v1024/caches/train_pilot.bin',
  [string]$TokenizerModelPath = 'build/private-data/nicopedia-real-text-bpe-v1024/tokenizer/byte-bpe-v1024.model',
  [string]$HexagonSdkRoot = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0',
  # Incident namespace.  Deliberately NOT docs/results/: a diagnostic run must
  # never be able to reach the G1 quality tree, even by accident.
  [string]$IncidentRoot = 'build/incident-6031/diagnostics',
  [string]$DiagnosticId = '',
  [string]$AnalysisOut = '',
  [switch]$SkipBuild,
  [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
. (Join-Path $PSScriptRoot 'nicopedia_runner_common.ps1')

Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainingRunner = Join-Path $PSScriptRoot 'run_nicopedia_htp_training.ps1'
$analyzer = Join-Path $PSScriptRoot 'incident_6031_analyze.py'

# The 1.5x identity under which both observed 6031 occurrences happened.  These
# are held fixed on purpose: the diagnostic run must not "succeed" by removing
# whatever is suspected.
$incidentIdentity = [ordered]@{
  purpose         = 'incident-diagnostic'
  not_a_quality_run = $true
  seed            = 2
  arm             = 'Control'
  attention_gate  = 'none'
  steps           = 128
  batch           = 8
  vocabulary      = 1024
  tokens          = 32
  dimension       = 64
  feed_forward    = 128
  layers          = 19
  heads           = 2
  muon_lr         = '0.0075'
  aux_adam_lr     = '0.0033'
  target_lr       = '0.00015'
  muon_backend    = 'HVX'
  schedule        = 'linear_decay'
  decay_start     = 4000
  decay_end       = 8000
  schedule_total  = 8000
  heartbeat       = 'on'
  progress_status = 'on'
  host_polling    = 'on'
  incident_trace  = 'on'
  logcat          = 'cleared-before-run,dumped-after'
  quality_eval    = 'disabled'
  checkpoint      = 'disabled'
}

function Get-IncidentPython {
  $python = if ($env:MIMO_PYTHON) { $env:MIMO_PYTHON } else { 'python' }
  $resolved = Get-Command $python -ErrorAction SilentlyContinue
  if (-not $resolved) { throw 'PYTHON_NOT_FOUND: set MIMO_PYTHON or put python on PATH' }
  return $resolved.Source
}

function Assert-IncidentNamespace {
  # Fail closed: the diagnostic namespace must live under build/, never under
  # docs/results/.  A diagnostic run is not evidence for the G1 decision and
  # must not be able to enter the quality tree.
  $full = [IO.Path]::GetFullPath((Join-Path $repoRoot $IncidentRoot))
  $resultsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results'))
  if ($full.StartsWith($resultsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "INCIDENT_NAMESPACE_IN_RESULTS_TREE: $IncidentRoot resolves under docs/results"
  }
  $buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build'))
  if (-not $full.StartsWith($buildRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "INCIDENT_NAMESPACE_OUTSIDE_BUILD: $IncidentRoot must resolve under build/"
  }
  return $full
}

function Get-IncidentId {
  if ($DiagnosticId) {
    if ($DiagnosticId -notmatch '^[A-Za-z0-9._-]+$') { throw 'DiagnosticId must match [A-Za-z0-9._-]+' }
    return $DiagnosticId
  }
  return ('incident-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), $PID)
}

function Write-IncidentIdentity {
  param([string]$Directory, [string]$Id)
  New-Item -ItemType Directory -Force -Path $Directory | Out-Null
  $record = [ordered]@{
    schema_version = 1
    diagnostic_id  = $Id
    identity       = $incidentIdentity
    actual         = [ordered]@{
      seed = $Seed
      steps = $Steps
      qairt_expected_build_id = $ExpectedBuildId
    }
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
  }
  # BOM-less UTF-8: the analyzer and git both read this file directly.
  [IO.File]::WriteAllText(
    (Join-Path $Directory 'incident-identity.json'),
    (($record | ConvertTo-Json -Depth 6) + "`n"),
    (New-Object Text.UTF8Encoding($false)))
}

function Invoke-IncidentLogcatCapture {
  <#
    .SYNOPSIS
    Clears logcat before the run and dumps it afterwards, regardless of outcome.

    .DESCRIPTION
    The buffer is cleared *before* the run so the dump cannot be dominated by
    unrelated history, and dumped in a finally block so a FAILED run - the case
    this whole investigation exists for - still yields backend evidence.  The
    raw dump is preserved unfiltered under build/ and is never committed.
  #>
  param(
    [Parameter(Mandatory = $true)][string]$Adb,
    [Parameter(Mandatory = $true)][string]$Device,
    [Parameter(Mandatory = $true)][string]$Directory
  )
  $rawPath = Join-Path $Directory 'logcat-raw.txt'
  $clearResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device `
    -Arguments @('logcat', '-c') -AllowFailure -TimeoutSeconds 60
  if ($clearResult.ExitCode -ne 0) {
    Write-Host "logcat_clear_failed exit=$($clearResult.ExitCode)"
  } else {
    Write-Host 'logcat_cleared=true'
  }
  # Best-effort dump: a transport failure here must not mask the run's own
  # result, but its absence is reported so nobody reads a missing dump as an
  # empty log.
  $dumpResult = Invoke-PhoneLmAdb -Adb $Adb -Device $Device `
    -Arguments @('logcat', '-d', '-v', 'threadtime') -AllowFailure -TimeoutSeconds 180
  if ($dumpResult.ExitCode -ne 0) {
    Write-Host "logcat_dump_failed exit=$($dumpResult.ExitCode)"
    return $false
  }
  [IO.File]::WriteAllText($rawPath, ($dumpResult.Text -join "`n"), (New-Object Text.UTF8Encoding($false)))
  Write-Host ("logcat_dump_lines={0}" -f (@($dumpResult.Text).Count))
  return $true
}

switch ($Mode) {
  'SelfTest' {
    $python = Get-IncidentPython
    & $python $analyzer --selftest
    exit $LASTEXITCODE
  }

  'Plan' {
    $incidentRoot = Assert-IncidentNamespace
    $id = Get-IncidentId
    Write-Host "incident_mode=plan diagnostic_id=$id"
    Write-Host "incident_namespace=$incidentRoot (guard: outside docs/results)"
    Write-Host "incident_steps=$Steps seed=$Seed arm=Control batch=8 muon_backend=HVX"
    Write-Host "incident_trace=on logcat=clear-then-dump quality_eval=disabled checkpoint=disabled"
    Write-Host "g1_quality_tier3=BLOCKED (this runner never touches it)"
    Write-Host "would_run: $trainingRunner -Seed $Seed -Steps $Steps -AttentionGate none ..."
    exit 0
  }

  'Analyze' {
    $incidentRoot = Assert-IncidentNamespace
    if (-not $AnalysisOut) { $AnalysisOut = $incidentRoot }
    $python = Get-IncidentPython
    $native = Get-ChildItem -LiteralPath $incidentRoot -Recurse -Filter 'incident-native-trace.log' -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime | Select-Object -Last 1
    $kotlin = Get-ChildItem -LiteralPath $incidentRoot -Recurse -Filter 'incident-kotlin-trace.log' -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime | Select-Object -Last 1
    $logcat = Get-ChildItem -LiteralPath $incidentRoot -Recurse -Filter 'logcat-raw.txt' -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime | Select-Object -Last 1
    if ($native) {
      $report = Get-ChildItem -LiteralPath (Split-Path -Parent $native.FullName) -Filter '*-result.txt' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
      $status = Get-ChildItem -LiteralPath (Split-Path -Parent $native.FullName) -Filter 'status.json' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
      $analyzerArgs = @($analyzer, '--native', $native.FullName, '--out', $AnalysisOut)
      if ($kotlin) { $analyzerArgs += @('--kotlin', $kotlin.FullName) }
      if ($report) { $analyzerArgs += @('--report', $report.FullName) }
      if ($status) { $analyzerArgs += @('--status', $status.FullName) }
      if ($logcat) { $analyzerArgs += @('--logcat', $logcat.FullName) }
      & $python @analyzerArgs
      exit $LASTEXITCODE
    }
    # No instrumented trace: fall back to legacy evidence analysis.
    & $python $analyzer --legacy-dir $incidentRoot --out $AnalysisOut
    exit $LASTEXITCODE
  }

  'Run' {
    $incidentRoot = Assert-IncidentNamespace
    $id = Get-IncidentId
    $dir = Join-Path $incidentRoot $id
    if (Test-Path -LiteralPath $dir) { throw "RUN_ID_REUSE: $dir already exists" }
    Write-IncidentIdentity -Directory $dir -Id $id

    $python = Get-IncidentPython
    $adb = [IO.Path]::Combine($env:LOCALAPPDATA, 'Android', 'Sdk', 'platform-tools', 'adb.exe')
    if (-not (Test-Path -LiteralPath $adb)) { $adb = 'adb' }
    $device = (Resolve-PhoneLmDevice -Adb $adb)
    $package = 'com.yuubinnkyoku.phonelm'

    # Enable the opt-in trace BEFORE the app process starts.  Both the Kotlin
    # and the native side look for this marker at startup; without it the run
    # is a normal run and the analyzer must be told so rather than silently
    # reporting an empty trace as "nothing happened near the failure".
    $markerCreated = $false
    try {
      [void](Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint `
        -Arguments @('shell', 'run-as', $package, 'mkdir', '-p', 'files/headless-input'))
      [void](Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint `
        -Arguments @('shell', 'run-as', $package, 'touch', 'files/headless-input/incident_trace_enabled'))
      $markerCreated = $true
      Write-Host 'incident_trace_marker=created'
      # Clear logcat immediately before the run so the dump is dominated by
      # this run rather than by whatever the device logged beforehand.
      [void](Invoke-IncidentLogcatCapture -Adb $adb -Device $device.Endpoint -Directory $dir)
    } catch {
      # Fail closed: without the marker this run cannot answer the question.
      throw "INCIDENT_TRACE_MARKER_FAILED: $($_.Exception.Message)"
    }

    $trainArgs = @{
      QairtSdkRoot = $QairtSdkRoot
      ExpectedBuildId = $ExpectedBuildId
      Seed = $Seed
      Layers = 19
      Steps = $Steps
      Tokens = 32
      Vocabulary = 1024
      Dimension = 64
      FeedForwardDimension = 128
      BatchSize = 8
      LearningRate = '0.0033'
      LearningRateSchedule = 'linear_decay'
      DecayStartStep = 4000
      DecayEndStep = 8000
      ScheduleTotalSteps = 8000
      TargetLearningRate = '0.00015'
      ExperimentFork = $true
      ParentLearningRate = '0.0033'
      Optimizer = 'Muon'
      MuonBackend = 'HVX'
      MuonLearningRate = '0.0075'
      MuonMomentum = '0.95'
      MuonNsSteps = 5
      AttentionGate = 'none'
      CachePath = $CachePath
      TokenizerModelPath = $TokenizerModelPath
      ReportRoot = $dir
      # Quality evaluation is off: this run must not produce quality numbers.
      EvalOnly = $false
      # Checkpoints are unnecessary for a 128-step incident run and would only
      # add I/O that perturbs the timeline being measured.
      CheckpointInterval = 1000000
      RunId = $id
    }
    if ($SkipBuild) { $trainArgs.SkipBuild = $true }
    if ($SkipInstall) { $trainArgs.SkipInstall = $true }

    Write-Host "incident_run_start diagnostic_id=$id steps=$Steps seed=$Seed"
    $runError = $null
    try {
      & $trainingRunner @trainArgs
    } catch {
      # A FAILED diagnostic run is a legitimate outcome, not a runner bug.
      # Record it and continue to artifact collection.
      $runError = $_
      Write-Host "incident_run_threw=$($_.Exception.Message)"
    }
    Write-Host "incident_marker_created=$markerCreated"

    # Pull the traces regardless of outcome.  These are the whole point: a
    # FAILED run is the case the analyzer exists for.
    $nativeTrace = Join-Path $dir 'incident-native-trace.log'
    foreach ($name in @('incident-native-trace.log', 'incident-kotlin-trace.log')) {
      $pulled = Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint `
        -Arguments @('exec-out', 'run-as', $package, 'cat', "files/headless/$name") -AllowFailure
      if ($pulled.ExitCode -eq 0) {
        [IO.File]::WriteAllText((Join-Path $dir $name), ($pulled.Text -join "`n"), (New-Object Text.UTF8Encoding($false)))
        Write-Host "pulled=$name"
      } else {
        Write-Host "pull_missing=$name (incident tracing was not active for this run)"
      }
    }
    # Dump logcat after the run, pass or fail, and remove the marker so the next
    # run cannot silently inherit instrumentation.
    [void](Invoke-IncidentLogcatCapture -Adb $adb -Device $device.Endpoint -Directory $dir)
    [void](Invoke-PhoneLmAdb -Adb $adb -Device $device.Endpoint `
      -Arguments @('shell', 'run-as', $package, 'rm', '-f', 'files/headless-input/incident_trace_enabled') `
      -AllowFailure)

    if (-not (Test-Path -LiteralPath $nativeTrace -PathType Leaf)) {
      Write-Host 'incident_analysis=SKIPPED (no native trace was produced)'
      if ($runError) { throw $runError }
      return
    }
    & $python $analyzer --native $nativeTrace --out (Join-Path $dir 'analysis')
    $analysisExit = $LASTEXITCODE
    Write-Host "incident_analysis_exit=$analysisExit"
    Write-Host "g1_quality_tier3=BLOCKED (unchanged by this diagnostic run)"
    if ($runError) { exit 1 }
    exit $analysisExit
  }
}
