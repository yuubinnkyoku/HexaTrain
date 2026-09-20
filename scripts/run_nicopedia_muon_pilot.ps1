# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# HexaTrain/Nicopedia Muon research pilot.
#
# This is a research-only orchestration layer.  It keeps the production Adam
# runner untouched and requires a downstream runner which explicitly exposes
# the Muon arguments before it will start a Muon device run.  The intended
# execution boundary is HTP forward+backward and CPU Muon/Aux-Adam updates;
# this script never labels that hybrid path as all-HTP training.
#
# All private ledgers, reports, checkpoints, and telemetry are below build/.
# This pilot has no final-test input or evaluator path by construction.
[CmdletBinding()]
param(
  [ValidateSet('Run','Plan','Summarize')][string]$Mode = 'Plan',
  [ValidateSet('Adam','Muon')][string]$Optimizer = 'Muon',
  [ValidateSet('Full','StageA','StageB')][string]$Stage = 'Full',
  # When omitted, Run executes all three Stage-A candidates.  Supplying this
  # parameter is useful for a controlled single-candidate recovery/debug run.
  [ValidateSet('0.005','0.010','0.020')][string]$MuonLearningRate = '0.010',
  [ValidateRange(0.0,0.999999)][double]$MuonMomentum = 0.95,
  [ValidateRange(1,16)][int]$MuonNsSteps = 5,
  [bool]$MuonNesterov = $true,
  [string]$QairtSdkRoot = '',
  [string]$ExpectedBuildId = '',
  [string]$LedgerRoot = 'build/muon-pilot/nicopedia-v1024-d64-f128',
  [string]$TrainingRunnerPath = 'scripts/run_nicopedia_htp_training.ps1',
  [string]$EvalRunnerPath = 'scripts/run_nicopedia_htp_eval.ps1',
  [string]$AdamReferenceRoot = 'build/hpo/nicopedia-v1024-d64-f128/schedule-v2b/trials/hpo-schedule-v2b-full-s4000-t0100-seed1',
  [string]$TrainingDataRoot = 'build/private-data/nicopedia-real-text-bpe-v1024',
  [string]$TrainCachePath = '',
  [string]$TokenizerModelPath = '',
  [string]$ResumeParityReport = '',
  [switch]$SkipBuild,
  [switch]$SkipInstall,
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$ScriptName = [IO.Path]::GetFileName($MyInvocation.MyCommand.Path)

# The frozen comparison recipe and the Muon pilot architecture are constants,
# deliberately not user-overridable.  This prevents a pilot result from being
# silently compared across an architecture, seed, or exposure change.
$Fixed = [ordered]@{
  vocabulary = 1024
  tokens = 32
  dimension = 64
  feed_forward_dimension = 128
  layers = 19
  heads = 2
  parameter_count = 758528
  batch_size = 8
  seed = 1
  tokenizer_kind = 'byte_bpe'
  tokenizer_hash = 'sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798'
  dataset_identity = 'nicopedia-real-text-bpe-v1024/train_pilot.bin'
  dataset_cache_content_hash = 'fnv1a64:0c7b2826f5f26fea'
  training_order_identity = 'fixed canonical train_pilot round-robin order'
  training_order_hash = 'fnv1a64:0e2e15196d851431'
  expected_cache_record_count = 96890
  qnn_backend = 'HTP'
  planned_max_steps = 8000
  planned_target_tokens = 2048000
  expected_original_utf8_bytes = 5491256
  expected_chunks = 46616
  expected_articles = 1949
  # These identities are emitted by the current native pilot report and are
  # also retained in the mixed V4 checkpoint header.  Keeping them here makes
  # recovered artifacts prove the same initialization and model epsilon as a
  # fresh run instead of relying on the filename alone.
  initial_parameter_hash = 'fnv1a64:a2e44951a39ee0dd'
  model_epsilon = 1.0e-5
  training_order_seed = 20260806
  adam_peak_learning_rate = '0.0022'
  adam_target_learning_rate = '0.0001'
  adam_beta1 = 0.9
  adam_beta2 = 0.999
  adam_epsilon = '1e-8'
  weight_decay = 0
  gradient_clip = 'disabled'
  decay_start_step = 4000
  decay_end_step = 8000
  schedule_total_steps = 8000
  validation_chunks = 256
  development_chunks = 256
}

# Parameter role and count derivations come from the checked-in machine-readable
# SSOT artifact (metadata/transformer_parameter_metadata.json), which is
# generated from app/src/main/cpp/transformer_parameter_metadata.h.  No
# parameter name/count registry is hand-written in this script: adding a
# parameter upstream only requires regenerating the artifact.  The MUON pilot
# architecture is ungated (headwiseG1=false), matching the frozen Adam
# reference and $Fixed.parameter_count.
#
# This runner intentionally keeps a local derivation helper instead of sourcing
# nicopedia_runner_common.ps1: the pilot is a frozen experiment runner whose
# manifest contract depends on $Fixed and a narrower return shape.  Coupling it
# to the shared helper is deferred; fail-closed vocabulary is mirrored here.
#
# Manifest compatibility: muon_parameter_roles / aux_adam_parameter_roles keep
# the original Muon-pilot semantic labels under the same schema_version.
# SSOT-derived suffix lists are written under the new explicit fields
# muon_parameter_suffixes / aux_adam_parameter_suffixes.
$MetadataJsonPath = Join-Path $Root 'metadata\transformer_parameter_metadata.json'
$ParameterMetadataSupportedSchemaVersion = 1
$MuonParameterRolesCompatibility = @('Wq','Wk','Wv','Wo','FFN_W1','FFN_W2')
$AuxAdamParameterRolesCompatibility = @('token_embedding','output_projection','norm_scale_gain','bias','other_non_hidden')
function Get-MetadataDerivationFromSsot {
  param([string]$MetadataPath = '')
  if (-not $MetadataPath) { $MetadataPath = $MetadataJsonPath }
  if (-not (Test-Path -LiteralPath $MetadataPath -PathType Leaf)) {
    throw 'PARAMETER_METADATA_JSON_MISSING: regenerate scripts\generate_parameter_metadata.ps1'
  }
  $raw = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
  $hasSchema = $false
  $schemaVersion = $null
  if ($raw -and $raw.PSObject -and $raw.PSObject.Properties['schema_version']) {
    $hasSchema = $true
    $schemaVersion = $raw.schema_version
  }
  if (-not $hasSchema -or $null -eq $schemaVersion -or [string]$schemaVersion -eq '') {
    throw 'PARAMETER_METADATA_SCHEMA_VERSION_UNSUPPORTED:'
  }
  $schemaText = [string]$schemaVersion
  if ($schemaText -notmatch '^\d+$' -or [int64]$schemaText -ne $ParameterMetadataSupportedSchemaVersion) {
    throw "PARAMETER_METADATA_SCHEMA_VERSION_UNSUPPORTED:$schemaText"
  }
  $defs = @($raw.parameter_definitions)
  if ($defs.Count -eq 0) { throw 'PARAMETER_METADATA_EMPTY' }
  $headwiseG1 = $false
  $dimensionExtent = @{
    VOCABULARY = [uint64]$Fixed.vocabulary
    MODEL = [uint64]$Fixed.dimension
    FEED_FORWARD = [uint64]$Fixed.feed_forward_dimension
    HEADS = [uint64]$Fixed.heads
  }
  $muonSuffixes = [Collections.Generic.List[string]]::new()
  $auxSuffixes = [Collections.Generic.List[string]]::new()
  $total = [uint64]0
  $muonElements = [uint64]0
  $auxElements = [uint64]0
  $muonMatrices = [uint64]0
  foreach ($def in $defs) {
    $condition = [string]$def.condition
    if ($condition -eq 'HEADWISE_G1') {
      if (-not $headwiseG1) { continue }
    } elseif ($condition -ne 'ALWAYS') {
      throw "PARAMETER_METADATA_CONDITION_UNKNOWN:$($def.condition)"
    }
    $elements = [uint64]1
    foreach ($dim in $def.shape) {
      $extent = $null
      if ($null -ne $dim -and $dimensionExtent.ContainsKey([string]$dim)) {
        $extent = $dimensionExtent[[string]$dim]
      }
      if ($null -eq $extent -or $extent -eq 0) { throw "PARAMETER_METADATA_DIMENSION_UNKNOWN:$dim" }
      $elements = $elements * $extent
    }
    $placement = [string]$def.placement
    $instances = switch ($placement) {
      'PER_LAYER' { [uint64]$Fixed.layers }
      'GLOBAL_PREFIX' { [uint64]1 }
      'GLOBAL_SUFFIX' { [uint64]1 }
      default { throw "PARAMETER_METADATA_PLACEMENT_UNKNOWN:$($def.placement)" }
    }
    $count = $elements * $instances
    $total += $count
    if ($def.role -eq 'MUON') {
      $muonSuffixes.Add([string]$def.suffix)
      $muonElements += $count
      $muonMatrices += $instances
    } elseif ($def.role -eq 'AUX_ADAM') {
      $auxSuffixes.Add([string]$def.suffix)
      $auxElements += $count
    } else {
      throw "PARAMETER_METADATA_ROLE_UNKNOWN:$($def.role)"
    }
  }
  return [ordered]@{
    parameter_count = $total
    # Compatibility: original semantic labels, same schema field meaning.
    muon_parameter_roles = @($MuonParameterRolesCompatibility)
    aux_adam_parameter_roles = @($AuxAdamParameterRolesCompatibility)
    # SSOT-derived concrete suffix lists.
    muon_parameter_suffixes = @($muonSuffixes)
    aux_adam_parameter_suffixes = @($auxSuffixes)
    muon_matrix_count = $muonMatrices
    muon_parameter_count = $muonElements
    aux_adam_parameter_count = $auxElements
  }
}
$MetadataDerived = Get-MetadataDerivationFromSsot
if ($MetadataDerived.parameter_count -ne [uint64]$Fixed.parameter_count) {
  throw "PARAMETER_METADATA_TOTAL_MISMATCH: derived=$($MetadataDerived.parameter_count) fixed=$($Fixed.parameter_count)"
}

$MuonLearningRateCandidates = @('0.005','0.010','0.020')
$MuonLearningRateWasExplicit = $PSBoundParameters.ContainsKey('MuonLearningRate')
$SmokeUpdates = 8
$AdamReference = [ordered]@{
  validation_bpb = 2.284370268
  development_bpb = 2.541861852
  balanced_bpb = 2.413116060
}
$ClearDivergenceMarginBpb = 0.50
$PromisingImprovementBpb = 0.01
$FastCurveToleranceBpb = 0.005
$SchemaVersion = 1
$CheckpointFormatMuon = 'NPRTCKPTV4'
$CheckpointFormatAdam = 'NPRTCKPTV3'
$MuonAlgorithmIdentity = 'keller_original_64560829_fp32'
$RequiredMuonRunnerParameters = @(
  'Optimizer',
  'MuonLearningRate',
  'MuonMomentum',
  'MuonNsSteps'
)
$OptionalMuonRunnerParameters = @(
  # The current research runner fixes Nesterov=true in the instrumentation
  # argument map.  A future generic runner may expose this as a PowerShell
  # parameter; either route is accepted only when the value remains true.
  'MuonNesterov',
  'MuonTargetLearningRate',
  'MuonLearningRateSchedule'
)
$OrderHashCache = @{}
$FnvMask64 = [Numerics.BigInteger]::Parse('18446744073709551615')
$FnvOffset64 = [Numerics.BigInteger]::Parse('14695981039346656037')
$FnvPrime64 = [Numerics.BigInteger]::Parse('1099511628211')
$SplitMixAdd64 = [Numerics.BigInteger]::Parse('11400714819323198485')
$SplitMixMul1_64 = [Numerics.BigInteger]::Parse('13787848793156543929')
$SplitMixMul2_64 = [Numerics.BigInteger]::Parse('10723151780598845931')

function Get-NowUtc {
  return [DateTime]::UtcNow.ToString('o')
}

function Resolve-UnderBuild {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Label
  )
  $candidate = if ([IO.Path]::IsPathRooted($Path)) {
    [IO.Path]::GetFullPath($Path)
  } else {
    [IO.Path]::GetFullPath((Join-Path $Root $Path))
  }
  $build = [IO.Path]::GetFullPath((Join-Path $Root 'build')).TrimEnd('\','/')
  $prefix = $build + [IO.Path]::DirectorySeparatorChar
  if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "${Label}_MUST_BE_UNDER_BUILD"
  }
  return $candidate
}

function Assert-NoFinalTestPath {
  param([Parameter(Mandatory = $true)][string]$Path,[Parameter(Mandatory = $true)][string]$Label)
  # A final split is closed for this experiment.  Reject both path spellings
  # and a common concatenated spelling before any evaluator is started.
  if ($Path -match '(?i)(^|[\\/_.-])final[_-]?test([\\/_.-]|$)') {
    throw "${Label}_FINAL_TEST_FORBIDDEN"
  }
}

function Get-OptimizerKind {
  if ($Optimizer -eq 'Muon') { return 'MUON' }
  return 'ADAM'
}

function Get-OptimizerIdentity {
  if ((Get-OptimizerKind) -eq 'MUON') { return 'muon_aux_adam' }
  return 'adam'
}

function Get-CandidateRates {
  # PSBoundParameters distinguishes the default value from an explicit
  # single-candidate request without adding a second public switch.
  if ($MuonLearningRateWasExplicit) {
    return @($MuonLearningRate)
  }
  return @($MuonLearningRateCandidates)
}

function Get-ExpectedMuonTargetLearningRate {
  param([Parameter(Mandatory = $true)][double]$PeakLearningRate)
  $target = $PeakLearningRate * ([double]$Fixed.adam_target_learning_rate / [double]$Fixed.adam_peak_learning_rate)
  if (-not [double]::IsFinite($target) -or $target -le 0) { throw 'MUON_TARGET_LEARNING_RATE_NONFINITE' }
  return $target
}

function Get-ExpectedLinearLearningRate {
  param(
    [Parameter(Mandatory = $true)][double]$PeakLearningRate,
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][int]$StartStep,
    [Parameter(Mandatory = $true)][int]$EndStep,
    [Parameter(Mandatory = $true)][double]$TargetLearningRate
  )
  if ($Step -le $StartStep) { return $PeakLearningRate }
  if ($Step -ge $EndStep) { return $TargetLearningRate }
  $progress = ($Step - $StartStep) / [double]($EndStep - $StartStep)
  return $PeakLearningRate + $progress * ($TargetLearningRate - $PeakLearningRate)
}

function Get-MuonScheduledLearningRate {
  param(
    [Parameter(Mandatory = $true)][double]$PeakLearningRate,
    [Parameter(Mandatory = $true)][int]$Step
  )
  return Get-ExpectedLinearLearningRate `
    -PeakLearningRate $PeakLearningRate `
    -Step $Step `
    -StartStep $Fixed.decay_start_step `
    -EndStep $Fixed.decay_end_step `
    -TargetLearningRate (Get-ExpectedMuonTargetLearningRate $PeakLearningRate)
}

function Get-AuxAdamScheduledLearningRate {
  param([Parameter(Mandatory = $true)][int]$Step)
  return Get-ExpectedLinearLearningRate `
    -PeakLearningRate ([double]$Fixed.adam_peak_learning_rate) `
    -Step $Step `
    -StartStep $Fixed.decay_start_step `
    -EndStep $Fixed.decay_end_step `
    -TargetLearningRate ([double]$Fixed.adam_target_learning_rate)
}

function Get-ExpectedTrainingOrderHash {
  param([Parameter(Mandatory = $true)][int]$Step)
  if ($Step -lt 1 -or $Step -gt $Fixed.planned_max_steps) { throw "TRAINING_ORDER_STEP_UNSUPPORTED:$Step" }
  $cacheKey = [string]$Step
  if ($OrderHashCache.ContainsKey($cacheKey)) { return $OrderHashCache[$cacheKey] }
  # This is the same little-endian uint64 FNV-1a + SplitMix64 order as the
  # native NprtCache runner.  The expected record count is part of the fixed
  # cache identity; hashing the per-step prefix avoids mistaking the 8-step
  # smoke hash for the 1000/2000/... training-order hash.
  $state = [Numerics.BigInteger]::Parse('20260806')
  $hash = $FnvOffset64
  for ($index = 0; $index -lt $Step * $Fixed.batch_size; $index++) {
    $state = (($state + [Numerics.BigInteger]$index + $SplitMixAdd64) -band $FnvMask64)
    $state = ((($state -bxor ($state -shr 30)) * $SplitMixMul1_64) -band $FnvMask64)
    $state = ((($state -bxor ($state -shr 27)) * $SplitMixMul2_64) -band $FnvMask64)
    $state = (($state -bxor ($state -shr 31)) -band $FnvMask64)
    $recordIndex = $state % [Numerics.BigInteger]$Fixed.expected_cache_record_count
    for ($byteIndex = 0; $byteIndex -lt 8; $byteIndex++) {
      $byteValue = ($recordIndex -shr (8 * $byteIndex)) -band 255
      $hash = ((($hash -bxor $byteValue) * $FnvPrime64) -band $FnvMask64)
    }
  }
  $hex = $hash.ToString('x16')
  # BigInteger adds a sign-protecting zero when the uint64 high bit is set.
  if ($hex.Length -eq 17 -and $hex[0] -eq '0') { $hex = $hex.Substring(1) }
  if ($hex.Length -ne 16) { throw "TRAINING_ORDER_HASH_WIDTH:$Step" }
  $value = "fnv1a64:$hex"
  $OrderHashCache[$cacheKey] = $value
  return $value
}

function Get-TrialId {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  $safe = $LearningRate.Replace('.','p')
  return "muon-pilot-lr${safe}-seed$($Fixed.seed)"
}

function Get-TrialDirectory {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  return Join-Path (Join-Path $Ledger 'trials') (Get-TrialId $LearningRate)
}

function Get-TrainingDirectory {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  return Join-Path (Get-TrialDirectory $LearningRate) 'training'
}

function Get-CheckpointName {
  param([Parameter(Mandatory = $true)][int]$Step)
  return "htp-seed$($Fixed.seed)-l$($Fixed.layers)-t$($Fixed.tokens)-d$($Fixed.dimension)-f$($Fixed.feed_forward_dimension)-step$Step.ckpt"
}

function Get-EvaluationDirectory {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step)
  return Join-Path (Get-TrialDirectory $LearningRate) "eval/step-$Step-v$($Fixed.validation_chunks)-d$($Fixed.development_chunks)"
}

function Get-EvaluationReportPath {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step)
  $name = "seed$($Fixed.seed)-l$($Fixed.layers)-t$($Fixed.tokens)-d$($Fixed.dimension)-f$($Fixed.feed_forward_dimension)-step$Step-v$($Fixed.validation_chunks)-d$($Fixed.development_chunks)-htp.txt"
  return Join-Path (Get-EvaluationDirectory $LearningRate $Step) $name
}

function Read-KeyValueFile {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "REPORT_MISSING:$Path" }
  $map = @{}
  foreach ($line in Get-Content -LiteralPath $Path) {
    if ($line -notmatch '^([A-Za-z0-9_]+)=(.*)$') { continue }
    $key = $Matches[1]
    $value = $Matches[2].Trim()
    if ($map.ContainsKey($key) -and [string]$map[$key] -ne $value) {
      throw "REPORT_CONFLICTING_DUPLICATE_KEY:$key"
    }
    $map[$key] = $value
  }
  return $map
}

function Require-MapKeys {
  param([Parameter(Mandatory = $true)][hashtable]$Map,[Parameter(Mandatory = $true)][string[]]$Keys,[Parameter(Mandatory = $true)][string]$Kind)
  foreach ($key in $Keys) {
    if (-not $Map.ContainsKey($key)) { throw "${Kind}_FIELD_MISSING:$key" }
  }
}

function Get-MapValue {
  param([Parameter(Mandatory = $true)][hashtable]$Map,[Parameter(Mandatory = $true)][string[]]$Names)
  foreach ($name in $Names) {
    if ($Map.ContainsKey($name) -and -not [string]::IsNullOrWhiteSpace([string]$Map[$name])) {
      return [string]$Map[$name]
    }
  }
  return $null
}

function Assert-MapFiniteNumber {
  param([Parameter(Mandatory = $true)][hashtable]$Map,[Parameter(Mandatory = $true)][string[]]$Names,[Parameter(Mandatory = $true)][string]$Label)
  $value = Get-MapValue $Map $Names
  if ($null -eq $value) { return $null }
  $parsed = 0.0
  if (-not [double]::TryParse($value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or
      -not [double]::IsFinite($parsed)) {
    throw "${Label}_NONFINITE"
  }
  return $parsed
}

function Get-ConfigEvidenceValue {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Map,
    [Parameter(Mandatory = $false)][hashtable]$Manifest,
    [Parameter(Mandatory = $true)][string[]]$ReportNames,
    [Parameter(Mandatory = $true)][string]$ManifestName
  )
  $value = Get-MapValue $Map $ReportNames
  if ($null -ne $value) { return [pscustomobject][ordered]@{ value = $value; source = 'report' } }
  if ($null -ne $Manifest -and $Manifest.ContainsKey($ManifestName) -and
      -not [string]::IsNullOrWhiteSpace([string]$Manifest[$ManifestName])) {
    return [pscustomobject][ordered]@{ value = [string]$Manifest[$ManifestName]; source = 'manifest' }
  }
  return $null
}

function Assert-ConfigEvidenceNumber {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Map,
    [Parameter(Mandatory = $false)][hashtable]$Manifest,
    [Parameter(Mandatory = $true)][string[]]$ReportNames,
    [Parameter(Mandatory = $true)][string]$ManifestName,
    [Parameter(Mandatory = $true)][double]$Expected,
    [Parameter(Mandatory = $true)][string]$Label,
    [double]$Tolerance = 1.0e-7
  )
  $evidence = Get-ConfigEvidenceValue $Map $Manifest $ReportNames $ManifestName
  if ($null -eq $evidence) { throw "MUON_CONFIG_EVIDENCE_MISSING:$Label" }
  $actual = 0.0
  if (-not [double]::TryParse([string]$evidence.value, [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture, [ref]$actual) -or
      -not [double]::IsFinite($actual) -or [math]::Abs($actual - $Expected) -gt $Tolerance) {
    throw "MUON_CONFIG_EVIDENCE_MISMATCH:$Label"
  }
  return $evidence
}

function Assert-ConfigEvidenceString {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Map,
    [Parameter(Mandatory = $false)][hashtable]$Manifest,
    [Parameter(Mandatory = $true)][string[]]$ReportNames,
    [Parameter(Mandatory = $true)][string]$ManifestName,
    [Parameter(Mandatory = $true)][string]$Expected,
    [Parameter(Mandatory = $true)][string]$Label
  )
  $evidence = Get-ConfigEvidenceValue $Map $Manifest $ReportNames $ManifestName
  if ($null -eq $evidence) { throw "MUON_CONFIG_EVIDENCE_MISSING:$Label" }
  if ([string]$evidence.value -ne $Expected) { throw "MUON_CONFIG_EVIDENCE_MISMATCH:$Label" }
  return $evidence
}

function Assert-MuonConfigurationEvidence {
  param(
    [Parameter(Mandatory = $true)][hashtable]$Map,
    [Parameter(Mandatory = $false)][hashtable]$Manifest,
    [Parameter(Mandatory = $true)][int]$ExpectedStep,
    [Parameter(Mandatory = $true)][string]$LearningRate
  )
  [void](Assert-ConfigEvidenceString $Map $Manifest @('muon_algorithm_identity','algorithm_identity') 'muon_algorithm_identity' $MuonAlgorithmIdentity 'algorithm_identity')
  [void](Assert-ConfigEvidenceString $Map $Manifest @('learning_rate_schedule','muon_learning_rate_schedule') 'muon_learning_rate_schedule' 'linear_decay' 'muon_schedule')
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('muon_target_lr','muon_target_learning_rate') 'muon_target_learning_rate' (Get-ExpectedMuonTargetLearningRate ([double]$LearningRate)) 'muon_target_learning_rate' 1.0e-7)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_target_lr','aux_adam_target_learning_rate','learning_rate_target') 'aux_adam_target_learning_rate' ([double]$Fixed.adam_target_learning_rate) 'aux_adam_target_learning_rate' 1.0e-7)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_lr','aux_adam_peak_lr','learning_rate_peak') 'aux_adam_peak_learning_rate' ([double]$Fixed.adam_peak_learning_rate) 'aux_adam_peak_learning_rate' 1.0e-7)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('muon_lr','muon_learning_rate') 'muon_learning_rate' ([double]$LearningRate) 'muon_learning_rate' 1.0e-7)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('muon_momentum') 'muon_momentum' $MuonMomentum 'muon_momentum' 1.0e-7)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('muon_ns_steps') 'muon_ns_steps' $MuonNsSteps 'muon_ns_steps' 0.0)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_beta1','adam_beta1','beta1') 'aux_adam_beta1' ([double]$Fixed.adam_beta1) 'aux_adam_beta1' 1.0e-6)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_beta2','adam_beta2','beta2') 'aux_adam_beta2' ([double]$Fixed.adam_beta2) 'aux_adam_beta2' 1.0e-6)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_epsilon','adam_epsilon','epsilon') 'aux_adam_epsilon' ([double]$Fixed.adam_epsilon) 'aux_adam_epsilon' 1.0e-12)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('aux_adam_weight_decay','adam_weight_decay','weight_decay') 'aux_adam_weight_decay' ([double]$Fixed.weight_decay) 'aux_adam_weight_decay' 0.0)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('learning_rate_decay_start_step','decay_start_step') 'decay_start_step' ([double]$Fixed.decay_start_step) 'decay_start_step' 0.0)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('learning_rate_decay_end_step','decay_end_step') 'decay_end_step' ([double]$Fixed.decay_end_step) 'decay_end_step' 0.0)
  [void](Assert-ConfigEvidenceNumber $Map $Manifest @('learning_rate_schedule_total_steps','schedule_total_steps') 'schedule_total_steps' ([double]$Fixed.schedule_total_steps) 'schedule_total_steps' 0.0)
  $initial = Get-ConfigEvidenceValue $Map $Manifest @('initial_parameter_hash') 'initial_parameter_hash'
  if ($null -eq $initial -or [string]$initial.value -ne $Fixed.initial_parameter_hash) { throw 'MUON_INITIAL_PARAMETER_IDENTITY_REJECTED' }
  $order = Get-MapValue $Map @('training_order_hash')
  if ($null -eq $order -or $order -ne (Get-ExpectedTrainingOrderHash $ExpectedStep)) { throw 'MUON_TRAINING_ORDER_IDENTITY_REJECTED' }
  $cache = Get-MapValue $Map @('dataset_cache_content_hash','cache_content_hash')
  if ($null -ne $cache -and $cache -ne $Fixed.dataset_cache_content_hash) { throw 'MUON_DATASET_CACHE_IDENTITY_REJECTED' }
}

function Assert-HealthyMuonTrainingReport {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int]$ExpectedStep,
    [Parameter(Mandatory = $true)][int]$ExpectedResumeStep,
    [Parameter(Mandatory = $true)][string]$LearningRate,
    [Parameter(Mandatory = $false)][hashtable]$Manifest
  )
  $map = Read-KeyValueFile $Path
  Require-MapKeys $map @(
    'status','seed','model_dimension','feed_forward_dimension',
    'completed_steps','batch_size',
    'optimizer','optimizer_muon_backend','optimizer_aux_adam_backend',
    'forward_backward_backend','muon_momentum',
    'muon_nesterov','muon_ns_steps','muon_matrix_count','muon_parameter_count',
    'aux_adam_parameter_count','checkpoint_format','checkpoint_written',
    'final_parameter_hash','initial_parameter_hash','muon_algorithm_identity',
    'muon_target_lr','aux_adam_target_lr','learning_rate_schedule',
    'learning_rate_decay_start_step','learning_rate_decay_end_step',
    'learning_rate_schedule_total_steps','all_steps_finite','final_finite',
    'qnn_return_code_success','output_tensors_finite','cpu_fallback',
    'nan_detected','inf_detected','api_trace_graph_execute_failure_count',
    'api_trace_fallback_attempted','api_trace_fallback_succeeded',
    'training_order_hash'
  ) 'MUON_TRAINING'
  if ($map.status -ne 'SUCCESS' -or $map.optimizer -notin @('MUON','muon_aux_adam') -or
      $map.optimizer_muon_backend -ne 'CPU' -or $map.optimizer_aux_adam_backend -ne 'CPU' -or
      $map.forward_backward_backend -ne 'HTP') { throw 'MUON_TRAINING_IDENTITY_OR_PLACEMENT_REJECTED' }
  if ([int]$map.seed -ne $Fixed.seed -or [int]$map.model_dimension -ne $Fixed.dimension -or
      [int]$map.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or
      [int]$map.batch_size -ne $Fixed.batch_size -or
      [int]$map.completed_steps -ne $ExpectedStep) {
    throw 'MUON_TRAINING_MODEL_OR_STEP_IDENTITY_REJECTED'
  }
  if ($map.ContainsKey('layers') -and [int]$map.layers -ne $Fixed.layers) { throw 'MUON_TRAINING_LAYERS_MISMATCH' }
  if ($map.ContainsKey('heads') -and [int]$map.heads -ne $Fixed.heads) { throw 'MUON_TRAINING_HEADS_MISMATCH' }
  if ($map.ContainsKey('parameter_element_count') -and [int]$map.parameter_element_count -ne $Fixed.parameter_count) { throw 'MUON_TRAINING_PARAMETER_COUNT_MISMATCH' }
  if ($map.ContainsKey('resume_from_step') -and [int]$map.resume_from_step -ne $ExpectedResumeStep) { throw 'MUON_TRAINING_RESUME_STEP_MISMATCH' }
  $reportedMuonLearningRate = Get-MapValue $map @('muon_learning_rate','muon_lr')
  if ($null -eq $reportedMuonLearningRate -or [single]$reportedMuonLearningRate -ne [single]$LearningRate -or
      [double]$map.muon_momentum -ne $MuonMomentum -or
      $map.muon_nesterov -ne $MuonNesterov.ToString().ToLowerInvariant() -or
      [int]$map.muon_ns_steps -ne $MuonNsSteps -or
      [int]$map.muon_matrix_count -ne $MetadataDerived.muon_matrix_count -or
      [int]$map.muon_parameter_count -ne $MetadataDerived.muon_parameter_count -or
      [int]$map.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count) {
    throw 'MUON_PARAMETER_SPLIT_OR_HYPERPARAMETER_REJECTED'
  }
  if ($map.checkpoint_format -ne $CheckpointFormatMuon -or $map.checkpoint_written -ne 'true' -or
      $map.final_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') { throw 'MUON_CHECKPOINT_STATE_REJECTED' }
  Assert-MuonConfigurationEvidence $map $Manifest $ExpectedStep $LearningRate
  if ($map.ContainsKey('experiment_fork') -and $map.experiment_fork -ne 'true') { throw 'MUON_FRESH_INITIALIZATION_IDENTITY_REJECTED' }
  # Return-code success and finite tensors are intentionally independent gates.
  if ($map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or
      $map.cpu_fallback -ne 'false' -or $map.nan_detected -ne 'false' -or $map.inf_detected -ne 'false' -or
      $map.all_steps_finite -ne 'true' -or $map.final_finite -ne 'true' -or
      $map.api_trace_graph_execute_failure_count -ne '0' -or
      $map.api_trace_fallback_attempted -ne 'false' -or $map.api_trace_fallback_succeeded -ne 'false') {
    throw 'MUON_TRAINING_HEALTH_REJECTED'
  }
  $reportedCacheHash = Get-MapValue $map @('dataset_cache_content_hash','cache_content_hash')
  if (($null -ne $reportedCacheHash -and $reportedCacheHash -ne $Fixed.dataset_cache_content_hash) -or
      $map.training_order_hash -ne (Get-ExpectedTrainingOrderHash $ExpectedStep)) { throw 'MUON_TRAINING_DATA_EXPOSURE_IDENTITY_REJECTED' }
  foreach ($key in @('final_test_opened','final_test_used','final_test_evaluated')) {
    if ($map.ContainsKey($key) -and $map[$key] -ne 'false') { throw 'FINAL_TEST_FORBIDDEN' }
  }
  foreach ($key in @('muon_ms','aux_adam_ms','parameter_transfer_ms','total_update_ms','fwd_backward_ms','training_total_seconds','training_step_ms')) {
    if ($map.ContainsKey($key)) { [void](Assert-MapFiniteNumber $map @($key) "MUON_$key") }
  }
  if ($map.ContainsKey('run_completed_steps') -and [int]$map.run_completed_steps -ne ($ExpectedStep - $ExpectedResumeStep)) {
    throw 'MUON_TRAINING_RUN_STEP_COUNT_MISMATCH'
  }
  return $map
}

function Get-CoreCheckpointIdentity {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "CHECKPOINT_MISSING:$Path" }
  $bytes = [IO.File]::ReadAllBytes($Path)
  if ($bytes.Length -lt 43) { throw 'CHECKPOINT_TOO_SMALL' }
  function Read-U32([byte[]]$Data,[int]$Offset) {
    return [uint32](([uint64]$Data[$Offset] * 16777216) + ([uint64]$Data[$Offset + 1] * 65536) + ([uint64]$Data[$Offset + 2] * 256) + [uint64]$Data[$Offset + 3])
  }
  function Read-U64([byte[]]$Data,[int]$Offset) {
    $value = [uint64]0
    for ($index = 0; $index -lt 8; $index++) { $value = ($value * 256) + [uint64]$Data[$Offset + $index] }
    return $value
  }
  function Read-F32([byte[]]$Data,[int]$Offset) {
    $bits = Read-U32 $Data $Offset
    return [BitConverter]::ToSingle([BitConverter]::GetBytes([uint32]$bits),0)
  }
  $magic = [Text.Encoding]::ASCII.GetString($bytes,0,11).Trim()
  if ($magic -eq $CheckpointFormatMuon) {
    # NPRTCKPTV4 inserts epsilon after the six u32 shape fields and stores a
    # u64 global step.  The legacy V1/V2/V3 header has no epsilon.  Keeping
    # these layouts separate is essential: reading a V4 seed at the V3 offset
    # would make every valid Muon resume look like a foreign checkpoint.
    if ($bytes.Length -lt 51) { throw 'NPRTCKPTV4_HEADER_TRUNCATED' }
    $offset = 51
    function Read-V4String([byte[]]$Data,[ref]$Cursor,[string]$Label) {
      if ($Cursor.Value -gt $Data.Length - 4) { throw 'NPRTCKPTV4_STRING_LENGTH_TRUNCATED' }
      $length = Read-U32 $Data $Cursor.Value
      $Cursor.Value += 4
      if ($length -lt 1 -or $length -gt 4096 -or $Cursor.Value -gt $Data.Length - [int]$length) { throw "NPRTCKPTV4_STRING_INVALID:$Label" }
      $value = [Text.Encoding]::UTF8.GetString($Data,$Cursor.Value,[int]$length)
      $Cursor.Value += [int]$length
      return $value
    }
    $vocabulary = Read-U32 $bytes 11
    $tokens = Read-U32 $bytes 15
    $dimension = Read-U32 $bytes 19
    $feedForward = Read-U32 $bytes 23
    $layers = Read-U32 $bytes 27
    $heads = Read-U32 $bytes 31
    $epsilon = Read-F32 $bytes 35
    $seed = Read-U32 $bytes 39
    $step = Read-U64 $bytes 43
    $tokenizerKind = Read-V4String $bytes ([ref]$offset) 'TOKENIZER_KIND'
    $tokenizerHash = Read-V4String $bytes ([ref]$offset) 'TOKENIZER_HASH'
    $datasetHash = Read-V4String $bytes ([ref]$offset) 'DATASET_HASH'
    $recordIndex = Read-U64 $bytes $offset; $offset += 8
    $tokenOffset = Read-U64 $bytes $offset; $offset += 8
    $epoch = Read-U64 $bytes $offset; $offset += 8
    $exposedTokens = Read-U64 $bytes $offset; $offset += 8
    $orderSeed = Read-U64 $bytes $offset; $offset += 8
    $optimizerIdentity = Read-V4String $bytes ([ref]$offset) 'OPTIMIZER_IDENTITY'
    $muonLearningRate = Read-F32 $bytes $offset; $offset += 4
    $auxAdamLearningRate = Read-F32 $bytes $offset; $offset += 4
    $muonTargetLearningRate = Read-F32 $bytes $offset; $offset += 4
    $auxAdamTargetLearningRate = Read-F32 $bytes $offset; $offset += 4
    $muonMomentum = Read-F32 $bytes $offset; $offset += 4
    $muonNesterov = Read-U32 $bytes $offset; $offset += 4
    $muonNsSteps = Read-U32 $bytes $offset; $offset += 4
    $auxAdamBeta1 = Read-F32 $bytes $offset; $offset += 4
    $auxAdamBeta2 = Read-F32 $bytes $offset; $offset += 4
    $auxAdamEpsilon = Read-F32 $bytes $offset; $offset += 4
    $muonWeightDecay = Read-F32 $bytes $offset; $offset += 4
    $auxAdamWeightDecay = Read-F32 $bytes $offset; $offset += 4
    $decayStartStep = Read-U32 $bytes $offset; $offset += 4
    $decayEndStep = Read-U32 $bytes $offset; $offset += 4
    $scheduleTotalSteps = Read-U32 $bytes $offset; $offset += 4
    $schemaVersion = Read-U32 $bytes $offset; $offset += 4
    $registryVersion = Read-U32 $bytes $offset; $offset += 4
    $registryCount = Read-U32 $bytes $offset; $offset += 4
    return [pscustomobject][ordered]@{
      magic = $magic
      vocabulary = $vocabulary; tokens = $tokens; dimension = $dimension
      feed_forward_dimension = $feedForward; layers = $layers; heads = $heads
      epsilon = $epsilon; seed = $seed; step = $step
      tokenizer_kind = $tokenizerKind
      tokenizer_hash = $tokenizerHash
      dataset_hash = $datasetHash
      record_index = $recordIndex; token_offset = $tokenOffset; epoch = $epoch
      exposed_tokens = $exposedTokens; order_seed = $orderSeed
      optimizer_identity = $optimizerIdentity
      muon_learning_rate = $muonLearningRate; aux_adam_learning_rate = $auxAdamLearningRate
      muon_target_learning_rate = $muonTargetLearningRate; aux_adam_target_learning_rate = $auxAdamTargetLearningRate
      muon_momentum = $muonMomentum; muon_nesterov = $muonNesterov; muon_ns_steps = $muonNsSteps
      aux_adam_beta1 = $auxAdamBeta1; aux_adam_beta2 = $auxAdamBeta2; aux_adam_epsilon = $auxAdamEpsilon
      muon_weight_decay = $muonWeightDecay; aux_adam_weight_decay = $auxAdamWeightDecay
      decay_start_step = $decayStartStep; decay_end_step = $decayEndStep; schedule_total_steps = $scheduleTotalSteps
      schema_version = $schemaVersion; registry_version = $registryVersion; registry_count = $registryCount
      sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
  }
  return [pscustomobject][ordered]@{
    magic = $magic
    vocabulary = Read-U32 $bytes 11
    tokens = Read-U32 $bytes 15
    dimension = Read-U32 $bytes 19
    feed_forward_dimension = Read-U32 $bytes 23
    layers = Read-U32 $bytes 27
    heads = Read-U32 $bytes 31
    seed = Read-U32 $bytes 35
    step = Read-U32 $bytes 39
    sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Assert-CheckpointIdentity {
  param([Parameter(Mandatory = $true)][string]$Path,[Parameter(Mandatory = $true)][int]$Step,[Parameter(Mandatory = $true)][string]$OptimizerKind,[string]$LearningRate = '')
  $identity = Get-CoreCheckpointIdentity $Path
  $expectedMagic = if ($OptimizerKind -eq 'MUON') { $CheckpointFormatMuon } else { $CheckpointFormatAdam }
  if ($identity.magic -ne $expectedMagic -or $identity.vocabulary -ne $Fixed.vocabulary -or
      $identity.tokens -ne $Fixed.tokens -or $identity.dimension -ne $Fixed.dimension -or
      $identity.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or $identity.layers -ne $Fixed.layers -or
      $identity.heads -ne $Fixed.heads -or $identity.seed -ne $Fixed.seed -or $identity.step -ne $Step) {
    throw "CHECKPOINT_IDENTITY_MISMATCH:${OptimizerKind}:$Step"
  }
  if ($OptimizerKind -eq 'MUON' -and (
      $identity.tokenizer_kind -ne $Fixed.tokenizer_kind -or
      $identity.tokenizer_hash -ne $Fixed.tokenizer_hash -or
      $identity.dataset_hash -ne $Fixed.dataset_cache_content_hash)) {
    throw "CHECKPOINT_DATA_IDENTITY_MISMATCH:${OptimizerKind}:$Step"
  }
  if ($OptimizerKind -eq 'MUON') {
    $expectedMuonLr = if ([string]::IsNullOrWhiteSpace($LearningRate)) { [double]$MuonLearningRate } else { [double]$LearningRate }
    $expectedMuonTarget = Get-ExpectedMuonTargetLearningRate $expectedMuonLr
    $expectedNesterov = if ($MuonNesterov) { [uint32]1 } else { [uint32]0 }
    if ([math]::Abs([double]$identity.epsilon - [double]$Fixed.model_epsilon) -gt 1.0e-8 -or
        $identity.optimizer_identity -ne 'muon_aux_adam' -or $identity.schema_version -ne 4 -or
        $identity.registry_version -ne 1 -or $identity.registry_count -ne 192 -or
        $identity.record_index -ne ([uint64]$Step * [uint64]$Fixed.batch_size) -or
        $identity.token_offset -ne 0 -or $identity.epoch -ne 0 -or
        $identity.exposed_tokens -ne ([uint64]$Step * [uint64]$Fixed.batch_size * [uint64]$Fixed.tokens) -or
        # A small set of already verified pilot checkpoints was written by the
        # pre-cursor-metadata APK, which serialized the model seed (1) in this
        # field while still using the canonical 20260806 order.  Preserve that
        # artifact reuse only because the report order hash is checked below;
        # all new checkpoints must carry the canonical order seed.
        @([uint64]$Fixed.training_order_seed,[uint64]$Fixed.seed) -notcontains [uint64]$identity.order_seed -or
        [math]::Abs([double]$identity.muon_learning_rate - $expectedMuonLr) -gt 1.0e-7 -or
        [math]::Abs([double]$identity.aux_adam_learning_rate - [double]$Fixed.adam_peak_learning_rate) -gt 1.0e-7 -or
        [math]::Abs([double]$identity.muon_target_learning_rate - $expectedMuonTarget) -gt 1.0e-7 -or
        [math]::Abs([double]$identity.aux_adam_target_learning_rate - [double]$Fixed.adam_target_learning_rate) -gt 1.0e-7 -or
        [math]::Abs([double]$identity.muon_momentum - $MuonMomentum) -gt 1.0e-7 -or
        $identity.muon_nesterov -ne $expectedNesterov -or
        $identity.muon_ns_steps -ne $MuonNsSteps -or
        [math]::Abs([double]$identity.aux_adam_beta1 - [double]$Fixed.adam_beta1) -gt 1.0e-6 -or
        [math]::Abs([double]$identity.aux_adam_beta2 - [double]$Fixed.adam_beta2) -gt 1.0e-6 -or
        [math]::Abs([double]$identity.aux_adam_epsilon - [double]$Fixed.adam_epsilon) -gt 1.0e-12 -or
        [math]::Abs([double]$identity.muon_weight_decay - [double]$Fixed.weight_decay) -gt 0.0 -or
        [math]::Abs([double]$identity.aux_adam_weight_decay - [double]$Fixed.weight_decay) -gt 0.0 -or
        $identity.decay_start_step -ne $Fixed.decay_start_step -or $identity.decay_end_step -ne $Fixed.decay_end_step -or
        $identity.schedule_total_steps -ne $Fixed.schedule_total_steps) {
      throw "CHECKPOINT_V4_OPTIMIZER_OR_CURSOR_IDENTITY_MISMATCH:$Step"
    }
  }
  return $identity
}

function Invoke-MuonCheckpointDecodeEvidence {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][string]$LearningRate,
    [Parameter(Mandatory = $true)][string]$ExpectedParameterHash
  )
  $identity = Assert-CheckpointIdentity $Path $Step 'MUON' $LearningRate
  $hostExe = Join-Path $Root 'build/host-tests/htp_checkpoint_eval.exe'
  $validation = Join-Path $EvalCacheResolved 'validation.bin'
  $development = Join-Path $EvalCacheResolved 'development.bin'
  $model = Join-Path (Split-Path -Parent $Path) 'byte-bpe-v1024.model'
  # The PowerShell header parser above proves the V4 identity/cursor and the
  # SHA-256.  When the host evaluator and fixed Dev/Val caches are present,
  # additionally decode every registry/state vector through the independent
  # C++ codec and compare its parameter hash with the device report.
  if (-not (Test-Path -LiteralPath $hostExe -PathType Leaf) -or
      -not (Test-Path -LiteralPath $validation -PathType Leaf) -or
      -not (Test-Path -LiteralPath $development -PathType Leaf) -or
      -not (Test-Path -LiteralPath $model -PathType Leaf)) {
    return [pscustomobject][ordered]@{ status = 'HEADER_ONLY_UNAVAILABLE'; parameter_hash = ''; checkpoint_sha256 = $identity.sha256 }
  }
  $probe = @(& $hostExe $Path $validation $development 1 1 2>&1)
  $code = $LASTEXITCODE
  if ($code -ne 0) { throw "MUON_CHECKPOINT_FULL_DECODE_FAILED:${Step}:$($probe -join ' ')" }
  $map = @{}
  foreach ($line in $probe) {
    if ([string]$line -match '^([A-Za-z0-9_]+)=(.*)$') { $map[$Matches[1]] = $Matches[2].Trim() }
  }
  foreach ($key in @('seed','layers','dimension','feed_forward_dimension','step','parameter_hash','finite','validation_chunks','development_chunks')) {
    if (-not $map.ContainsKey($key)) { throw "MUON_CHECKPOINT_FULL_DECODE_FIELD_MISSING:$key" }
  }
  if ([int]$map.seed -ne $Fixed.seed -or [int]$map.layers -ne $Fixed.layers -or
      [int]$map.dimension -ne $Fixed.dimension -or [int]$map.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or
      [int]$map.step -ne $Step -or $map.finite -ne 'true' -or [int]$map.validation_chunks -ne 1 -or
      [int]$map.development_chunks -ne 1 -or $map.parameter_hash -ne $ExpectedParameterHash) {
    throw "MUON_CHECKPOINT_FULL_DECODE_IDENTITY_MISMATCH:$Step"
  }
  return [pscustomobject][ordered]@{ status = 'PASS'; parameter_hash = $map.parameter_hash; checkpoint_sha256 = $identity.sha256 }
}

function Assert-HealthyEvaluationReport {
  param([Parameter(Mandatory = $true)][string]$Path,[Parameter(Mandatory = $true)][int]$ExpectedStep)
  $map = Read-KeyValueFile $Path
  Require-MapKeys $map @(
    'status','seed','layers','heads','model_dimension','feed_forward_dimension',
    'checkpoint_step','checkpoint_format','checkpoint_finite','checkpoint_parameter_elements',
    'checkpoint_parameter_hash','context_tokens','vocabulary_size','tokenizer_kind','tokenizer_hash',
    'validation_chunks','development_chunks','validation_nonfinite_chunks','development_nonfinite_chunks',
    'validation_bits_per_utf8_byte','development_bits_per_utf8_byte','qnn_return_code_success',
    'output_tensors_finite','cpu_fallback','nan_detected','inf_detected',
    'api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded'
  ) 'EVAL'
  if ($map.status -ne 'SUCCESS' -or [int]$map.seed -ne $Fixed.seed -or [int]$map.layers -ne $Fixed.layers -or
      [int]$map.heads -ne $Fixed.heads -or [int]$map.model_dimension -ne $Fixed.dimension -or
      [int]$map.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or [int]$map.checkpoint_step -ne $ExpectedStep -or
      $map.checkpoint_format -ne $CheckpointFormatMuon -or $map.checkpoint_finite -ne 'true' -or
      [int]$map.checkpoint_parameter_elements -ne $Fixed.parameter_count -or [int]$map.context_tokens -ne $Fixed.tokens -or
      [int]$map.vocabulary_size -ne $Fixed.vocabulary -or $map.tokenizer_kind -ne $Fixed.tokenizer_kind -or
      $map.tokenizer_hash -ne $Fixed.tokenizer_hash -or [int]$map.validation_chunks -ne $Fixed.validation_chunks -or
      [int]$map.development_chunks -ne $Fixed.development_chunks -or $map.validation_nonfinite_chunks -ne '0' -or
      $map.development_nonfinite_chunks -ne '0' -or $map.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$') {
    throw 'EVAL_IDENTITY_OR_HEALTH_REJECTED'
  }
  if ($map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or
      $map.cpu_fallback -ne 'false' -or $map.nan_detected -ne 'false' -or $map.inf_detected -ne 'false' -or
      $map.api_trace_graph_execute_failure_count -ne '0' -or $map.api_trace_fallback_attempted -ne 'false' -or
      $map.api_trace_fallback_succeeded -ne 'false') { throw 'EVAL_HEALTH_REJECTED' }
  [void](Assert-MapFiniteNumber $map @('validation_bits_per_utf8_byte') 'EVAL_VALIDATION_BPB')
  [void](Assert-MapFiniteNumber $map @('development_bits_per_utf8_byte') 'EVAL_DEVELOPMENT_BPB')
  return $map
}

function Get-BalancedBpb {
  param([Parameter(Mandatory = $true)][hashtable]$Map)
  return ([double]$Map.validation_bits_per_utf8_byte + [double]$Map.development_bits_per_utf8_byte) / 2.0
}

function Get-TrainingReportPath {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step)
  $directory = Get-TrainingDirectory $LearningRate
  $exact = Join-Path $directory "seed$($Fixed.seed)-l$($Fixed.layers)-v$($Fixed.vocabulary)-t$($Fixed.tokens)-d$($Fixed.dimension)-f$($Fixed.feed_forward_dimension)-steps$Step-result.txt"
  if (Test-Path -LiteralPath $exact -PathType Leaf) { return $exact }
  $matches = @(Get-ChildItem -LiteralPath $directory -File -Filter "*steps$Step-result.txt" -ErrorAction SilentlyContinue)
  if ($matches.Count -eq 1) { return $matches[0].FullName }
  if ($matches.Count -gt 1) { throw "TRAINING_REPORT_AMBIGUOUS:${LearningRate}:$Step" }
  return $null
}

function Get-RunnerParameterNames {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "TRAINING_RUNNER_MISSING:$Path" }
  $tokens = $null
  $errors = $null
  $ast = [Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors)
  if ($null -ne $errors -and @($errors).Count -gt 0) { throw 'TRAINING_RUNNER_PARSE_FAILED' }
  if ($null -eq $ast.ParamBlock) { throw 'TRAINING_RUNNER_PARAM_BLOCK_MISSING' }
  return @($ast.ParamBlock.Parameters | ForEach-Object { $_.Name.VariablePath.UserPath })
}

function Assert-MuonRunnerInterface {
  param([Parameter(Mandatory = $true)][string[]]$ParameterNames)
  foreach ($name in $RequiredMuonRunnerParameters) {
    if ($ParameterNames -notcontains $name) { throw "MUON_RUNNER_ARGUMENT_UNAVAILABLE:$name" }
  }
}

function Get-FutureTrainingRunnerArguments {
  param(
    [Parameter(Mandatory = $true)][string]$LearningRate,
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][int]$ResumeStep,
    [Parameter(Mandatory = $true)][string[]]$ParameterNames
  )
  $muonTarget = Get-ExpectedMuonTargetLearningRate ([double]$LearningRate)
  $args = @(
    '-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,
    '-Seed',$Fixed.seed,'-Layers',$Fixed.layers,'-Steps',$Step,'-Tokens',$Fixed.tokens,
    '-Vocabulary',$Fixed.vocabulary,'-Dimension',$Fixed.dimension,
    '-FeedForwardDimension',$Fixed.feed_forward_dimension,'-BatchSize',$Fixed.batch_size,
    '-LearningRate',$Fixed.adam_peak_learning_rate,'-LearningRateSchedule','linear_decay',
    '-DecayStartStep',$Fixed.decay_start_step,'-DecayEndStep',$Fixed.decay_end_step,
    '-ScheduleTotalSteps',$Fixed.schedule_total_steps,'-TargetLearningRate',$Fixed.adam_target_learning_rate,
    '-ExperimentFork','-ParentLearningRate',$Fixed.adam_peak_learning_rate,
    '-ResumeStep',$ResumeStep,'-CheckpointInterval',250,'-CheckpointStallSeconds',7200,
    '-CachePath',$TrainCacheResolved,'-TokenizerModelPath',$TokenizerResolved,
    '-ReportRoot',(Get-TrainingDirectory $LearningRate),
    '-RunId',("muon-pilot-$(Get-TrialId $LearningRate)-step$Step-resume$ResumeStep".Substring(0,[Math]::Min(63,("muon-pilot-$(Get-TrialId $LearningRate)-step$Step-resume$ResumeStep").Length)))
  )
  foreach ($pair in @(
    @('Optimizer','Muon'),
    @('MuonLearningRate',$LearningRate),
    @('MuonMomentum',[string]$MuonMomentum),
    @('MuonNesterov',$MuonNesterov.ToString().ToLowerInvariant()),
    @('MuonNsSteps',[string]$MuonNsSteps),
    @('MuonTargetLearningRate',[string]$muonTarget),
    @('MuonLearningRateSchedule','linear_decay')
  )) {
    if ($ParameterNames -contains $pair[0]) { $args += @("-$($pair[0])",$pair[1]) }
  }
  if ($SkipBuild) { $args += '-SkipBuild' }
  if ($SkipInstall) { $args += '-SkipInstall' }
  return $args
}

function New-Manifest {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  $optimizerKind = Get-OptimizerKind
  $manifest = [ordered]@{
    schema_version = $SchemaVersion
    protocol = 'hexatrain-nicopedia-muon-pilot-v1'
    trial_id = Get-TrialId $LearningRate
    status = 'PENDING'
    optimizer = $optimizerKind
    optimizer_identity = Get-OptimizerIdentity
    forward_backward_backend = 'HTP'
    optimizer_muon_backend = if ($optimizerKind -eq 'MUON') { 'CPU' } else { 'NOT_APPLICABLE' }
    optimizer_aux_adam_backend = if ($optimizerKind -eq 'MUON') { 'CPU' } else { 'HTP_OR_REUSED_REFERENCE' }
    fresh_initialization_required = ($optimizerKind -eq 'MUON')
    resume_from_adam_forbidden = ($optimizerKind -eq 'MUON')
    checkpoint_format = if ($optimizerKind -eq 'MUON') { $CheckpointFormatMuon } else { $CheckpointFormatAdam }
    checkpoint_state_semantics = if ($optimizerKind -eq 'MUON') { 'Muon momentum buffer per matrix + Aux Adam m/v; global step' } else { 'frozen Adam m/v; global step' }
    muon_algorithm_identity = if ($optimizerKind -eq 'MUON') { $MuonAlgorithmIdentity } else { '' }
    muon_learning_rate_schedule = if ($optimizerKind -eq 'MUON') { 'linear_decay' } else { '' }
    muon_learning_rate = $LearningRate
    muon_target_learning_rate = if ($optimizerKind -eq 'MUON') { [string](Get-ExpectedMuonTargetLearningRate ([double]$LearningRate)) } else { '' }
    muon_momentum = $MuonMomentum
    muon_nesterov = $MuonNesterov
    muon_ns_steps = $MuonNsSteps
    muon_parameter_roles = $MetadataDerived.muon_parameter_roles
    aux_adam_parameter_roles = $MetadataDerived.aux_adam_parameter_roles
    muon_parameter_suffixes = $MetadataDerived.muon_parameter_suffixes
    aux_adam_parameter_suffixes = $MetadataDerived.aux_adam_parameter_suffixes
    muon_matrix_count = if ($optimizerKind -eq 'MUON') { $MetadataDerived.muon_matrix_count } else { 0 }
    muon_parameter_count = if ($optimizerKind -eq 'MUON') { $MetadataDerived.muon_parameter_count } else { 0 }
    aux_adam_parameter_count = if ($optimizerKind -eq 'MUON') { $MetadataDerived.aux_adam_parameter_count } else { $Fixed.parameter_count }
    aux_adam_beta1 = $Fixed.adam_beta1
    aux_adam_beta2 = $Fixed.adam_beta2
    aux_adam_epsilon = $Fixed.adam_epsilon
    aux_adam_weight_decay = $Fixed.weight_decay
    aux_adam_gradient_clip = $Fixed.gradient_clip
    decay_start_step = $Fixed.decay_start_step
    decay_end_step = $Fixed.decay_end_step
    schedule_total_steps = $Fixed.schedule_total_steps
    validation_chunks = $Fixed.validation_chunks
    development_chunks = $Fixed.development_chunks
    final_test_opened = $false
    final_test_used = $false
    same_seed_as_adam = $true
    same_data_order_as_adam = $true
    same_tokenizer_as_adam = $true
    same_cache_as_adam = $true
    initial_parameter_hash = $Fixed.initial_parameter_hash
    completed_steps = 0
    actual_new_steps = 0
    resume_parity_gate = 'REQUIRED_BEFORE_FULL'
    created_utc = Get-NowUtc
    git_revision = (& git -C $Root rev-parse HEAD).Trim()
  }
  foreach ($key in $Fixed.Keys) { $manifest[$key] = $Fixed[$key] }
  return $manifest
}

function Load-Manifest {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  $path = Join-Path (Get-TrialDirectory $LearningRate) 'manifest.json'
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}

function Save-Manifest {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][System.Collections.IDictionary]$Manifest)
  $directory = Get-TrialDirectory $LearningRate
  [IO.Directory]::CreateDirectory($directory) | Out-Null
  $Manifest | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath (Join-Path $directory 'manifest.json') -Encoding utf8
}

function Assert-ManifestIdentity {
  param([Parameter(Mandatory = $true)][hashtable]$Manifest,[Parameter(Mandatory = $true)][string]$LearningRate)
  foreach ($key in @('trial_id','optimizer','optimizer_identity','checkpoint_format','checkpoint_state_semantics','muon_matrix_count','muon_parameter_count','aux_adam_parameter_count','vocabulary','tokens','dimension','feed_forward_dimension','layers','heads','parameter_count','batch_size','seed','tokenizer_kind','tokenizer_hash','dataset_identity','dataset_cache_content_hash','training_order_identity','training_order_hash','final_test_opened','final_test_used','fresh_initialization_required','resume_from_adam_forbidden','same_seed_as_adam','same_data_order_as_adam','same_tokenizer_as_adam','same_cache_as_adam','aux_adam_beta1','aux_adam_beta2','aux_adam_epsilon','aux_adam_weight_decay','aux_adam_gradient_clip','decay_start_step','decay_end_step','schedule_total_steps','validation_chunks','development_chunks')) {
    if (-not $Manifest.ContainsKey($key)) { throw "MANIFEST_FIELD_MISSING:$key" }
  }
  if ($Manifest.trial_id -ne (Get-TrialId $LearningRate) -or $Manifest.optimizer -ne 'MUON' -or
      $Manifest.optimizer_identity -ne 'muon_aux_adam' -or [string]$Manifest.muon_learning_rate -ne $LearningRate -or
      [int]$Manifest.vocabulary -ne $Fixed.vocabulary -or [int]$Manifest.tokens -ne $Fixed.tokens -or
      [int]$Manifest.dimension -ne $Fixed.dimension -or [int]$Manifest.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or
      [int]$Manifest.layers -ne $Fixed.layers -or [int]$Manifest.heads -ne $Fixed.heads -or
      [int]$Manifest.muon_matrix_count -ne $MetadataDerived.muon_matrix_count -or [int]$Manifest.muon_parameter_count -ne $MetadataDerived.muon_parameter_count -or
      [int]$Manifest.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count -or
      [int]$Manifest.parameter_count -ne $Fixed.parameter_count -or [int]$Manifest.batch_size -ne $Fixed.batch_size -or
      [int]$Manifest.seed -ne $Fixed.seed -or $Manifest.tokenizer_kind -ne $Fixed.tokenizer_kind -or
      $Manifest.tokenizer_hash -ne $Fixed.tokenizer_hash -or $Manifest.dataset_identity -ne $Fixed.dataset_identity -or
       $Manifest.dataset_cache_content_hash -ne $Fixed.dataset_cache_content_hash -or
       $Manifest.training_order_identity -ne $Fixed.training_order_identity -or $Manifest.training_order_hash -ne $Fixed.training_order_hash -or
       [bool]$Manifest.final_test_opened -or [bool]$Manifest.final_test_used -or
       [bool]$Manifest.fresh_initialization_required -ne $true -or [bool]$Manifest.resume_from_adam_forbidden -ne $true -or
       [bool]$Manifest.same_seed_as_adam -ne $true -or [bool]$Manifest.same_data_order_as_adam -ne $true -or
       [bool]$Manifest.same_tokenizer_as_adam -ne $true -or [bool]$Manifest.same_cache_as_adam -ne $true -or
       [double]$Manifest.muon_matrix_count -ne $MetadataDerived.muon_matrix_count -or [double]$Manifest.muon_parameter_count -ne $MetadataDerived.muon_parameter_count -or
       [double]$Manifest.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count -or [double]$Manifest.parameter_count -ne $Fixed.parameter_count -or
       [double]$Manifest.batch_size -ne $Fixed.batch_size -or [double]$Manifest.seed -ne $Fixed.seed -or
       [double]$Manifest.aux_adam_beta1 -ne [double]$Fixed.adam_beta1 -or
       [double]$Manifest.aux_adam_beta2 -ne [double]$Fixed.adam_beta2 -or
       [double]$Manifest.aux_adam_epsilon -ne [double]$Fixed.adam_epsilon -or
       [double]$Manifest.aux_adam_weight_decay -ne [double]$Fixed.weight_decay -or
       [string]$Manifest.aux_adam_gradient_clip -ne [string]$Fixed.gradient_clip -or
       [int]$Manifest.decay_start_step -ne $Fixed.decay_start_step -or [int]$Manifest.decay_end_step -ne $Fixed.decay_end_step -or
       [int]$Manifest.schedule_total_steps -ne $Fixed.schedule_total_steps -or
       [int]$Manifest.validation_chunks -ne $Fixed.validation_chunks -or [int]$Manifest.development_chunks -ne $Fixed.development_chunks) { throw 'MANIFEST_IDENTITY_REJECTED' }
  if ($Manifest.ContainsKey('muon_algorithm_identity') -and $Manifest.muon_algorithm_identity -ne $MuonAlgorithmIdentity) { throw 'MANIFEST_ALGORITHM_IDENTITY_REJECTED' }
  if ($Manifest.ContainsKey('muon_learning_rate_schedule') -and $Manifest.muon_learning_rate_schedule -ne 'linear_decay') { throw 'MANIFEST_SCHEDULE_IDENTITY_REJECTED' }
  if ($Manifest.ContainsKey('initial_parameter_hash') -and $Manifest.initial_parameter_hash -ne $Fixed.initial_parameter_hash) { throw 'MANIFEST_INITIAL_PARAMETER_IDENTITY_REJECTED' }
  # Compatibility: *_parameter_roles keep the original semantic labels under
  # the same schema field names. Old manifests without the new suffix fields
  # remain accepted; new suffix fields are validated when present.
  if ($Manifest.ContainsKey('muon_parameter_roles') -and
      [string](@($Manifest.muon_parameter_roles) -join ',') -ne [string]($MetadataDerived.muon_parameter_roles -join ',')) {
    throw 'MANIFEST_MUON_PARAMETER_ROLES_REJECTED'
  }
  if ($Manifest.ContainsKey('aux_adam_parameter_roles') -and
      [string](@($Manifest.aux_adam_parameter_roles) -join ',') -ne [string]($MetadataDerived.aux_adam_parameter_roles -join ',')) {
    throw 'MANIFEST_AUX_ADAM_PARAMETER_ROLES_REJECTED'
  }
  if ($Manifest.ContainsKey('muon_parameter_suffixes') -and
      [string](@($Manifest.muon_parameter_suffixes) -join ',') -ne [string]($MetadataDerived.muon_parameter_suffixes -join ',')) {
    throw 'MANIFEST_MUON_PARAMETER_SUFFIXES_REJECTED'
  }
  if ($Manifest.ContainsKey('aux_adam_parameter_suffixes') -and
      [string](@($Manifest.aux_adam_parameter_suffixes) -join ',') -ne [string]($MetadataDerived.aux_adam_parameter_suffixes -join ',')) {
    throw 'MANIFEST_AUX_ADAM_PARAMETER_SUFFIXES_REJECTED'
  }
  if ($Manifest.ContainsKey('muon_target_learning_rate')) {
    $expectedTarget = Get-ExpectedMuonTargetLearningRate ([double]$LearningRate)
    if ([math]::Abs([double]$Manifest.muon_target_learning_rate - $expectedTarget) -gt 1.0e-7) { throw 'MANIFEST_MUON_TARGET_LR_REJECTED' }
  }
}

function Write-Event {
  param([Parameter(Mandatory = $true)][hashtable]$Fields)
  [IO.Directory]::CreateDirectory($Ledger) | Out-Null
  $row = [ordered]@{ utc = Get-NowUtc }
  foreach ($key in $Fields.Keys) { $row[$key] = $Fields[$key] }
  $row | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $Ledger 'events.jsonl') -Encoding utf8
}

function Get-ResumeParityPath {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  if ($ResumeParityReport) { return Resolve-UnderBuild $ResumeParityReport 'ResumeParityReport' }
  return Join-Path (Get-TrialDirectory $LearningRate) 'validation/resume-parity.json'
}

function Assert-ResumeParityGate {
  param([Parameter(Mandatory = $true)][string]$LearningRate)
  $path = Get-ResumeParityPath $LearningRate
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "RESUME_PARITY_GATE_REQUIRED:$LearningRate" }
  $gate = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
  if ($gate.status -ne 'PASS' -or $gate.optimizer -ne 'MUON' -or [int]$gate.seed -ne $Fixed.seed -or
      [int]$gate.fresh_steps -ne 8 -or [int]$gate.split_prefix_steps -ne 4 -or [int]$gate.split_total_steps -ne 8 -or
      $gate.dataset_cursor_equal -ne $true -or $gate.parameters_equal -ne $true -or $gate.optimizer_state_equal -ne $true) {
    throw "RESUME_PARITY_GATE_REJECTED:$LearningRate"
  }
}

function Write-TelemetryRow {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step,[Parameter(Mandatory = $true)][hashtable]$Map,[Parameter(Mandatory = $true)][string]$Status,[double]$ValidationBpb = [double]::NaN,[double]$DevelopmentBpb = [double]::NaN,[int]$ResumeStep = 0)
  $telemetryPath = Join-Path (Get-TrialDirectory $LearningRate) 'telemetry.csv'
  $columns = @(
    'trial_id','optimizer','forward_backward_backend','optimizer_muon_backend','optimizer_aux_adam_backend',
    'muon_algorithm_identity','muon_lr','muon_target_lr','aux_adam_lr','aux_adam_target_lr',
    'aux_adam_beta1','aux_adam_beta2','aux_adam_epsilon','aux_adam_weight_decay',
    'muon_momentum','muon_nesterov','muon_ns_steps','muon_matrix_count','muon_parameter_count','aux_adam_parameter_count',
    'learning_rate_schedule','learning_rate_decay_start_step','learning_rate_decay_end_step','learning_rate_schedule_total_steps',
    'initial_parameter_hash','training_order_hash','dataset_cache_content_hash',
    'step','run_update_count','tokens_seen','tokens_per_second','tokens_per_second_scope',
    'timing_scope','fwd_backward_ms','fwd_backward_total_ms','fwd_backward_per_update_ms',
    'muon_ms','muon_total_ms','muon_per_update_ms','aux_adam_ms','aux_adam_total_ms','aux_adam_per_update_ms',
    'parameter_transfer_ms','parameter_transfer_total_ms','parameter_transfer_per_update_ms',
    'parameter_transfer_measurement_scope','parameter_transfer_zero_means_no_movement',
    'total_update_ms','total_update_total_ms','total_update_per_update_ms',
    'qnn_execute_count','qnn_failures','qnn_return_code_success','output_tensors_finite','finite','fallback',
    'checkpoint_v4_decode','checkpoint_parameter_hash','validation_bpb','development_bpb','balanced_bpb','status'
  )
  $rows = @()
  if (Test-Path -LiteralPath $telemetryPath -PathType Leaf) {
    # Preserve rows written by the first pilot revision while upgrading the
    # header.  The legacy timing fields remain cumulative-run values; the new
    # *_total_ms and *_per_update_ms columns remove that ambiguity.
    foreach ($old in @(Import-Csv -LiteralPath $telemetryPath | Where-Object { $_.step -ne [string]$Step })) {
      $legacy = [ordered]@{}
      foreach ($column in $columns) {
        if ($old.PSObject.Properties.Name -contains $column) { $legacy[$column] = $old.$column } else { $legacy[$column] = '' }
      }
      if ([string]::IsNullOrWhiteSpace([string]$legacy.timing_scope)) { $legacy.timing_scope = 'cumulative_run_legacy' }
      if ([string]::IsNullOrWhiteSpace([string]$legacy.parameter_transfer_measurement_scope)) { $legacy.parameter_transfer_measurement_scope = 'host_visible_qnn_bindings_only' }
      if ([string]::IsNullOrWhiteSpace([string]$legacy.parameter_transfer_zero_means_no_movement)) { $legacy.parameter_transfer_zero_means_no_movement = 'false' }
      $rows += [pscustomobject]$legacy
    }
  }
  $get = { param([string[]]$Names) Get-MapValue $Map $Names }
  $number = {
    param([string[]]$Names)
    $raw = & $get $Names
    if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string]$raw)) { return $null }
    $parsed = 0.0
    if (-not [double]::TryParse([string]$raw, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or -not [double]::IsFinite($parsed)) { throw "MUON_TELEMETRY_NONFINITE:$($Names -join '/')" }
    return $parsed
  }
  $runUpdates = [int](& $number @('run_completed_steps','run_update_count'))
  if ($runUpdates -le 0) { $runUpdates = $Step - $ResumeStep }
  if ($runUpdates -le 0) { $runUpdates = $Step }
  if ($runUpdates -le 0) { throw 'MUON_TELEMETRY_UPDATE_COUNT_INVALID' }
  $fwdTotal = & $number @('fwd_backward_ms','forward_backward_ms')
  $muonTotal = & $number @('muon_ms','optimizer_muon_ms')
  $auxTotal = & $number @('aux_adam_ms','optimizer_aux_adam_ms')
  $transferTotal = & $number @('parameter_transfer_ms','optimizer_parameter_transfer_ms')
  $updateTotal = & $number @('total_update_ms','optimizer_total_update_ms')
  $tokensSeen = [int64]$runUpdates * [int64]$Fixed.batch_size * [int64]$Fixed.tokens
  $tokensPerSecond = if ($null -ne $updateTotal -and $updateTotal -gt 0) { $tokensSeen / ($updateTotal / 1000.0) } else { $null }
  $auxBeta1 = Get-MapValue $Map @('aux_adam_beta1','adam_beta1','beta1')
  $auxBeta2 = Get-MapValue $Map @('aux_adam_beta2','adam_beta2','beta2')
  $auxEpsilon = Get-MapValue $Map @('aux_adam_epsilon','adam_epsilon','epsilon')
  $auxWeightDecay = Get-MapValue $Map @('aux_adam_weight_decay','adam_weight_decay','weight_decay')
  if ($null -eq $auxBeta1) { $auxBeta1 = $Fixed.adam_beta1 }
  if ($null -eq $auxBeta2) { $auxBeta2 = $Fixed.adam_beta2 }
  if ($null -eq $auxEpsilon) { $auxEpsilon = $Fixed.adam_epsilon }
  if ($null -eq $auxWeightDecay) { $auxWeightDecay = $Fixed.weight_decay }
  $row = [ordered]@{
    trial_id = Get-TrialId $LearningRate
    optimizer = 'muon_aux_adam'
    forward_backward_backend = 'HTP'
    optimizer_muon_backend = 'CPU'
    optimizer_aux_adam_backend = 'CPU'
    muon_algorithm_identity = Get-MapValue $Map @('muon_algorithm_identity','algorithm_identity')
    muon_lr = $LearningRate
    muon_target_lr = Get-MapValue $Map @('muon_target_lr','muon_target_learning_rate')
    aux_adam_lr = Get-AuxAdamScheduledLearningRate $Step
    aux_adam_target_lr = Get-MapValue $Map @('aux_adam_target_lr','aux_adam_target_learning_rate')
    aux_adam_beta1 = $auxBeta1
    aux_adam_beta2 = $auxBeta2
    aux_adam_epsilon = $auxEpsilon
    aux_adam_weight_decay = $auxWeightDecay
    muon_momentum = $MuonMomentum
    muon_nesterov = $MuonNesterov
    muon_ns_steps = $MuonNsSteps
    muon_matrix_count = $MetadataDerived.muon_matrix_count
    muon_parameter_count = $MetadataDerived.muon_parameter_count
    aux_adam_parameter_count = $MetadataDerived.aux_adam_parameter_count
    learning_rate_schedule = Get-MapValue $Map @('learning_rate_schedule','muon_learning_rate_schedule')
    learning_rate_decay_start_step = Get-MapValue $Map @('learning_rate_decay_start_step','decay_start_step')
    learning_rate_decay_end_step = Get-MapValue $Map @('learning_rate_decay_end_step','decay_end_step')
    learning_rate_schedule_total_steps = Get-MapValue $Map @('learning_rate_schedule_total_steps','schedule_total_steps')
    initial_parameter_hash = Get-MapValue $Map @('initial_parameter_hash')
    training_order_hash = Get-MapValue $Map @('training_order_hash')
    dataset_cache_content_hash = Get-MapValue $Map @('dataset_cache_content_hash','cache_content_hash')
    step = $Step
    run_update_count = $runUpdates
    tokens_seen = $tokensSeen
    tokens_per_second = $tokensPerSecond
    tokens_per_second_scope = 'average_run'
    timing_scope = 'cumulative_run'
    fwd_backward_ms = $fwdTotal
    fwd_backward_total_ms = $fwdTotal
    fwd_backward_per_update_ms = if ($null -ne $fwdTotal) { $fwdTotal / $runUpdates } else { $null }
    muon_ms = $muonTotal
    muon_total_ms = $muonTotal
    muon_per_update_ms = if ($null -ne $muonTotal) { $muonTotal / $runUpdates } else { $null }
    aux_adam_ms = $auxTotal
    aux_adam_total_ms = $auxTotal
    aux_adam_per_update_ms = if ($null -ne $auxTotal) { $auxTotal / $runUpdates } else { $null }
    parameter_transfer_ms = $transferTotal
    parameter_transfer_total_ms = $transferTotal
    parameter_transfer_per_update_ms = if ($null -ne $transferTotal) { $transferTotal / $runUpdates } else { $null }
    parameter_transfer_measurement_scope = 'host_visible_qnn_bindings_only'
    parameter_transfer_zero_means_no_movement = 'false'
    total_update_ms = $updateTotal
    total_update_total_ms = $updateTotal
    total_update_per_update_ms = if ($null -ne $updateTotal) { $updateTotal / $runUpdates } else { $null }
    qnn_execute_count = & $get @('qnn_execute_count','graph_execute_count')
    qnn_failures = & $get @('qnn_failures','api_trace_graph_execute_failure_count')
    qnn_return_code_success = & $get @('qnn_return_code_success')
    output_tensors_finite = & $get @('output_tensors_finite')
    finite = if ($Map.all_steps_finite -eq 'true' -and $Map.final_finite -eq 'true') { 'true' } else { 'false' }
    fallback = if ($Map.cpu_fallback -eq 'false' -and $Map.api_trace_fallback_attempted -eq 'false') { 'false' } else { 'true' }
    checkpoint_v4_decode = Get-MapValue $Map @('checkpoint_v4_decode')
    checkpoint_parameter_hash = Get-MapValue $Map @('final_parameter_hash','checkpoint_parameter_hash')
    validation_bpb = if ([double]::IsNaN($ValidationBpb)) { '' } else { $ValidationBpb }
    development_bpb = if ([double]::IsNaN($DevelopmentBpb)) { '' } else { $DevelopmentBpb }
    balanced_bpb = if ([double]::IsNaN($ValidationBpb) -or [double]::IsNaN($DevelopmentBpb)) { '' } else { ($ValidationBpb + $DevelopmentBpb) / 2.0 }
    status = $Status
  }
  $rows += [pscustomobject]$row
  @($rows) | Select-Object $columns | Export-Csv -LiteralPath $telemetryPath -NoTypeInformation -Encoding utf8
}

function Write-SummaryRows {
  param([Parameter(Mandatory = $true)][object[]]$Rows)
  if (@($Rows).Count -eq 0) { return }
  $path = Join-Path $Ledger 'summary.csv'
  $existing = if (Test-Path -LiteralPath $path -PathType Leaf) { @(Import-Csv -LiteralPath $path) } else { @() }
  $all = @($existing)
  foreach ($row in $Rows) {
    $all = @($all | Where-Object { -not ($_.trial_id -eq $row.trial_id -and [string]$_.step -eq [string]$row.step) }) + $row
  }
  @($all | Sort-Object trial_id, @{Expression={[int]$_.step}}) | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding utf8
}

function Get-AdamReferenceEvalMap {
  $referenceRoot = Resolve-UnderBuild $AdamReferenceRoot 'AdamReferenceRoot'
  Assert-NoFinalTestPath $referenceRoot 'AdamReferenceRoot'
  $manifestPath = Join-Path $referenceRoot 'manifest.json'
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'ADAM_REFERENCE_MANIFEST_MISSING' }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $hasFixedConfig = $manifest.PSObject.Properties.Name -contains 'fixed_config'
  # PowerShell variables are case-insensitive.  Do not call this local value
  # `$fixed`, because that would shadow the pilot's global `$Fixed` contract
  # and make later sample checks silently read the Adam manifest shape.
  $adamFixed = if ($hasFixedConfig -and $null -ne $manifest.fixed_config) { $manifest.fixed_config } else { $manifest }
  foreach ($pair in @(
    @('vocabulary',1024),@('tokens',32),@('dimension',64),@('feed_forward_dimension',128),@('layers',19),@('heads',2),@('parameter_count',758528),@('batch_size',8),@('seed',1),@('optimizer','ADAM'),@('tokenizer_kind','byte_bpe'),@('tokenizer_hash',$Fixed.tokenizer_hash),@('dataset_cache_content_hash',$Fixed.dataset_cache_content_hash),@('training_order_hash',$Fixed.training_order_hash)
  )) {
    if (-not $adamFixed.PSObject.Properties.Name.Contains($pair[0]) -or [string]$adamFixed.($pair[0]) -ne [string]$pair[1]) { throw "ADAM_REFERENCE_IDENTITY_MISMATCH:$($pair[0])" }
  }
  if ([string]$manifest.status -ne 'COMPLETED' -or [int]$manifest.completed_steps -lt 8000) { throw 'ADAM_REFERENCE_NOT_COMPLETED' }
  $evalReports = @(Get-ChildItem -LiteralPath $referenceRoot -Recurse -File -Filter '*step8000-v256-d256-htp.txt' | Where-Object { $_.FullName -notmatch '(?i)final[_-]?test' })
  if ($evalReports.Count -ne 1) { throw 'ADAM_REFERENCE_EVAL_AMBIGUOUS_OR_MISSING' }
  if ($evalReports[0].FullName -notmatch '(?i)[\\/]step-8000-v256-d256[\\/]') { throw 'ADAM_REFERENCE_EVAL_SAMPLE_IDENTITY_MISMATCH' }
  $map = Read-KeyValueFile $evalReports[0].FullName
  Require-MapKeys $map @('status','seed','layers','heads','model_dimension','feed_forward_dimension','checkpoint_step','checkpoint_format','checkpoint_finite','checkpoint_parameter_elements','checkpoint_parameter_hash','context_tokens','vocabulary_size','tokenizer_kind','tokenizer_hash','validation_chunks','development_chunks','validation_bits_per_utf8_byte','development_bits_per_utf8_byte','validation_nonfinite_chunks','development_nonfinite_chunks','qnn_return_code_success','output_tensors_finite','cpu_fallback','nan_detected','inf_detected','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded') 'ADAM_REFERENCE_EVAL'
  if ($map.status -ne 'SUCCESS' -or [int]$map.checkpoint_step -ne 8000 -or $map.checkpoint_format -ne $CheckpointFormatAdam -or
      [int]$map.seed -ne $Fixed.seed -or [int]$map.layers -ne $Fixed.layers -or [int]$map.heads -ne $Fixed.heads -or
      [int]$map.model_dimension -ne $Fixed.dimension -or [int]$map.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or
      $map.checkpoint_finite -ne 'true' -or [int]$map.checkpoint_parameter_elements -ne $Fixed.parameter_count -or
      [int]$map.context_tokens -ne $Fixed.tokens -or [int]$map.vocabulary_size -ne $Fixed.vocabulary -or
      $map.tokenizer_kind -ne $Fixed.tokenizer_kind -or $map.tokenizer_hash -ne $Fixed.tokenizer_hash -or
      [int]$map.validation_chunks -ne $Fixed.validation_chunks -or [int]$map.development_chunks -ne $Fixed.development_chunks -or
      $map.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$' -or $map.validation_nonfinite_chunks -ne '0' -or
      $map.development_nonfinite_chunks -ne '0' -or $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or
      $map.cpu_fallback -ne 'false' -or $map.nan_detected -ne 'false' -or $map.inf_detected -ne 'false' -or
      $map.api_trace_graph_execute_failure_count -ne '0' -or $map.api_trace_fallback_attempted -ne 'false' -or $map.api_trace_fallback_succeeded -ne 'false') { throw 'ADAM_REFERENCE_HEALTH_REJECTED' }
  $val = [double]$map.validation_bits_per_utf8_byte; $dev = [double]$map.development_bits_per_utf8_byte
  if ([math]::Abs($val - $AdamReference.validation_bpb) -gt 1.0e-6 -or [math]::Abs($dev - $AdamReference.development_bpb) -gt 1.0e-6) { throw 'ADAM_REFERENCE_METRIC_MISMATCH' }
  return [pscustomobject][ordered]@{ map = $map; balanced = ($val + $dev) / 2.0; path = $evalReports[0].FullName; source = $referenceRoot }
}

function Get-AdamAnchorMetric {
  param([Parameter(Mandatory = $true)][int]$Step)
  $referenceRoot = Resolve-UnderBuild $AdamReferenceRoot 'AdamReferenceRoot'
  $reports = @(Get-ChildItem -LiteralPath $referenceRoot -Recurse -File -Filter "*step$Step-v256-d256-htp.txt" | Where-Object { $_.FullName -notmatch '(?i)final[_-]?test' })
  if ($reports.Count -ne 1) { return $null }
  try {
    $map = Read-KeyValueFile $reports[0].FullName
    if ($reports[0].FullName -notmatch "(?i)[\\/]step-$Step-v256-d256[\\/]") { return $null }
    Require-MapKeys $map @('status','seed','layers','heads','model_dimension','feed_forward_dimension','checkpoint_step','checkpoint_format','checkpoint_finite','checkpoint_parameter_elements','checkpoint_parameter_hash','context_tokens','vocabulary_size','tokenizer_kind','tokenizer_hash','validation_chunks','development_chunks','validation_nonfinite_chunks','development_nonfinite_chunks','qnn_return_code_success','output_tensors_finite','cpu_fallback','nan_detected','inf_detected','api_trace_graph_execute_failure_count','api_trace_fallback_attempted','api_trace_fallback_succeeded') 'ADAM_ANCHOR'
    if ($map.status -ne 'SUCCESS' -or [int]$map.seed -ne $Fixed.seed -or [int]$map.layers -ne $Fixed.layers -or [int]$map.heads -ne $Fixed.heads -or
        [int]$map.model_dimension -ne $Fixed.dimension -or [int]$map.feed_forward_dimension -ne $Fixed.feed_forward_dimension -or
        [int]$map.checkpoint_step -ne $Step -or $map.checkpoint_format -ne $CheckpointFormatAdam -or $map.checkpoint_finite -ne 'true' -or
        [int]$map.checkpoint_parameter_elements -ne $Fixed.parameter_count -or $map.checkpoint_parameter_hash -notmatch '^fnv1a64:[0-9a-f]{16}$' -or
        [int]$map.context_tokens -ne $Fixed.tokens -or [int]$map.vocabulary_size -ne $Fixed.vocabulary -or
        $map.tokenizer_kind -ne $Fixed.tokenizer_kind -or $map.tokenizer_hash -ne $Fixed.tokenizer_hash -or
        [int]$map.validation_chunks -ne $Fixed.validation_chunks -or [int]$map.development_chunks -ne $Fixed.development_chunks -or
        $map.validation_nonfinite_chunks -ne '0' -or $map.development_nonfinite_chunks -ne '0' -or
        $map.qnn_return_code_success -ne 'true' -or $map.output_tensors_finite -ne 'true' -or $map.cpu_fallback -ne 'false' -or
        $map.nan_detected -ne 'false' -or $map.inf_detected -ne 'false' -or $map.api_trace_graph_execute_failure_count -ne '0' -or
        $map.api_trace_fallback_attempted -ne 'false' -or $map.api_trace_fallback_succeeded -ne 'false') { return $null }
    return [pscustomobject]@{ validation_bpb=[double]$map.validation_bits_per_utf8_byte; development_bpb=[double]$map.development_bits_per_utf8_byte; balanced_bpb=Get-BalancedBpb $map; source=$reports[0].FullName }
  } catch { return $null }
}

function Assert-StageResumePolicy {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$ResumeStep)
  if ($ResumeStep -eq 0) { return }
  $manifest = Load-Manifest $LearningRate
  if ($null -eq $manifest) { throw "RESUME_MANIFEST_MISSING:$LearningRate" }
  Assert-ManifestIdentity $manifest $LearningRate
  if ([int]$manifest.completed_steps -lt $ResumeStep -or $manifest.optimizer -ne 'MUON' -or
      [bool]$manifest.resume_from_adam_forbidden -ne $true) { throw "RESUME_SAME_CANDIDATE_REQUIRED:${LearningRate}:$ResumeStep" }
  $checkpoint = Join-Path (Get-TrainingDirectory $LearningRate) (Get-CheckpointName $ResumeStep)
  [void](Assert-CheckpointIdentity $checkpoint $ResumeStep 'MUON' $LearningRate)
}

function Invoke-TrainingPhase {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step,[Parameter(Mandatory = $true)][int]$ResumeStep,[Parameter(Mandatory = $true)][string]$Phase)
  if ((Get-OptimizerKind) -ne 'MUON') { throw 'ADAM_BASELINE_REUSE_ONLY_NO_RETRAIN' }
  Assert-StageResumePolicy $LearningRate $ResumeStep
  $existingManifest = Load-Manifest $LearningRate
  if ($null -ne $existingManifest) { Assert-ManifestIdentity $existingManifest $LearningRate }
  $trainingDirectory = Get-TrainingDirectory $LearningRate
  [IO.Directory]::CreateDirectory($trainingDirectory) | Out-Null
  $reportPath = Get-TrainingReportPath $LearningRate $Step
  $checkpointPath = Join-Path $trainingDirectory (Get-CheckpointName $Step)
  if ($reportPath -and (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    try {
      $reuseManifest = Load-Manifest $LearningRate
      if ($null -eq $reuseManifest) { $reuseManifest = New-Manifest $LearningRate }
      Assert-ManifestIdentity $reuseManifest $LearningRate
      $map = Assert-HealthyMuonTrainingReport $reportPath $Step $ResumeStep $LearningRate $reuseManifest
      $decode = Invoke-MuonCheckpointDecodeEvidence $checkpointPath $Step $LearningRate $map.final_parameter_hash
      $map['checkpoint_v4_decode'] = $decode.status
      $map['checkpoint_sha256'] = $decode.checkpoint_sha256
      # A device run can finish after the host ADB transport disconnects.  In
      # that case recovery installs the independently validated report and
      # checkpoint while the manifest still says BLOCKED_TRANSPORT.  Reuse is
      # authoritative for completion metadata, but it must not increment
      # actual_new_steps because the recovered execution was already counted
      # (or is reconciled once by the recovery path).
       $reuseManifest.status = 'COMPLETED'
      $reuseManifest.completed_steps = [Math]::Max([int]$reuseManifest.completed_steps, $Step)
       $reuseManifest.checkpoint_path = $checkpointPath
       $reuseManifest.checkpoint_parameter_hash = $map.final_parameter_hash
       $reuseManifest.checkpoint_v4_decode = $decode.status
       $reuseManifest.checkpoint_sha256 = $decode.checkpoint_sha256
      $reuseManifest.last_phase = "$Phase-reuse"
      $reuseManifest.resume_step_used = $ResumeStep
      $reuseManifest.health = 'PASS'
      foreach ($staleField in @('prune_reason','pruned_utc','blocked_reason','transport_interrupted')) {
        [void]$reuseManifest.Remove($staleField)
      }
      Save-Manifest $LearningRate $reuseManifest
      Write-Event @{ trial_id=Get-TrialId $LearningRate; phase="$Phase-reuse"; optimizer='muon_aux_adam'; step=$Step; resume_step=$ResumeStep; status='REUSED'; qnn_health='PASS' }
      Write-TelemetryRow $LearningRate $Step $map 'REUSED' -ResumeStep $ResumeStep
      return [ordered]@{ report=$reportPath; map=$map; checkpoint=$checkpointPath; reused=$true; wall_ms=0.0 }
    } catch {
      # An incomplete artifact is not evidence; the downstream runner must
      # produce a fresh, independently validated artifact in this namespace.
      $reportPath = $null
    }
  }
  $parameters = $script:RunnerParameterNames
  Assert-MuonRunnerInterface $parameters
  $arguments = Get-FutureTrainingRunnerArguments $LearningRate $Step $ResumeStep $parameters
  $manifest = Load-Manifest $LearningRate
  if ($null -eq $manifest) { $manifest = New-Manifest $LearningRate }
  Assert-ManifestIdentity $manifest $LearningRate
  $manifest.status = 'RUNNING'; $manifest.current_phase = $Phase; $manifest.current_step = $Step; $manifest.current_resume_step = $ResumeStep; Save-Manifest $LearningRate $manifest
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $runnerOutput = @()
  & pwsh -NoProfile -File $TrainingRunnerResolved @arguments 2>&1 |
    Tee-Object -Variable runnerOutput | Out-Host
  $code = $LASTEXITCODE
  $sw.Stop()
  if ($code -ne 0) {
    $runnerText = ($runnerOutput | Out-String)
    if ($runnerText -match 'ADB_TRANSPORT_FAILURE|ADB_TRANSPORT_TIMEOUT|ADB_NO_ONLINE_DEVICE') {
      throw "MUON_TRAINING_TRANSPORT_INTERRUPTED:${Phase}:${LearningRate}:exit=$code"
    }
    throw "MUON_TRAINING_PHASE_FAILED:${Phase}:${LearningRate}:exit=$code"
  }
  $reportPath = Get-TrainingReportPath $LearningRate $Step
  if (-not $reportPath) { throw "MUON_TRAINING_REPORT_MISSING:${Phase}:${LearningRate}:$Step" }
  $manifest = Load-Manifest $LearningRate; Assert-ManifestIdentity $manifest $LearningRate
  $map = Assert-HealthyMuonTrainingReport $reportPath $Step $ResumeStep $LearningRate $manifest
  $decode = Invoke-MuonCheckpointDecodeEvidence $checkpointPath $Step $LearningRate $map.final_parameter_hash
  $map['checkpoint_v4_decode'] = $decode.status
  $map['checkpoint_sha256'] = $decode.checkpoint_sha256
  $manifest.status = 'COMPLETED'; $manifest.completed_steps = $Step; $manifest.actual_new_steps = [int]$manifest.actual_new_steps + ($Step - $ResumeStep); $manifest.checkpoint_path = $checkpointPath; $manifest.checkpoint_parameter_hash = $map.final_parameter_hash; $manifest.checkpoint_v4_decode = $decode.status; $manifest.checkpoint_sha256 = $decode.checkpoint_sha256; $manifest.last_wall_time_ms = [math]::Round($sw.Elapsed.TotalMilliseconds,1); $manifest.last_phase = $Phase; $manifest.resume_step_used = $ResumeStep; $manifest.health = 'PASS'; Save-Manifest $LearningRate $manifest
  Write-Event @{ trial_id=Get-TrialId $LearningRate; phase=$Phase; optimizer='muon_aux_adam'; step=$Step; resume_step=$ResumeStep; status='COMPLETED'; qnn_health='PASS'; wall_time_ms=[math]::Round($sw.Elapsed.TotalMilliseconds,1) }
  Write-TelemetryRow $LearningRate $Step $map 'COMPLETED' -ResumeStep $ResumeStep
  return [ordered]@{ report=$reportPath; map=$map; checkpoint=$checkpointPath; reused=$false; wall_ms=$sw.Elapsed.TotalMilliseconds }
}

function Invoke-EvaluationPhase {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][int]$Step,[Parameter(Mandatory = $true)][string]$Phase)
  $checkpointPath = Join-Path (Get-TrainingDirectory $LearningRate) (Get-CheckpointName $Step)
  [void](Assert-CheckpointIdentity $checkpointPath $Step 'MUON' $LearningRate)
  $evalDirectory = Get-EvaluationDirectory $LearningRate $Step
  [IO.Directory]::CreateDirectory($evalDirectory) | Out-Null
  $evalPath = Get-EvaluationReportPath $LearningRate $Step
  if (-not (Test-Path -LiteralPath $evalPath -PathType Leaf)) {
    if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'EVAL_REQUIRES_EXPLICIT_QAIRT_ARGUMENTS' }
    $runIdRaw = "muon-pilot-$(Get-TrialId $LearningRate)-eval-step$Step"
    $runId = $runIdRaw.Substring(0,[Math]::Min(63,$runIdRaw.Length))
    $arguments = @('-QairtSdkRoot',$QairtSdkRoot,'-ExpectedBuildId',$ExpectedBuildId,'-SkipBuild','-SkipInstall','-Seed',$Fixed.seed,'-Layers',$Fixed.layers,'-Heads',$Fixed.heads,'-Tokens',$Fixed.tokens,'-Vocabulary',$Fixed.vocabulary,'-Dimension',$Fixed.dimension,'-FeedForwardDimension',$Fixed.feed_forward_dimension,'-CheckpointStep',$Step,'-ValidationChunks',$Fixed.validation_chunks,'-DevelopmentChunks',$Fixed.development_chunks,'-CheckpointPath',$checkpointPath,'-CacheRoot',$EvalCacheResolved,'-TokenizerModelPath',$TokenizerResolved,'-ReportRoot',$evalDirectory,'-RunId',$runId)
    & pwsh -NoProfile -File $EvalRunnerResolved @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "MUON_EVAL_PHASE_FAILED:${Phase}:${LearningRate}:$Step:exit=$LASTEXITCODE" }
  }
  if (-not (Test-Path -LiteralPath $evalPath -PathType Leaf)) { throw "MUON_EVAL_REPORT_MISSING:${LearningRate}:$Step" }
  $map = Assert-HealthyEvaluationReport $evalPath $Step
  $trainingMap = Read-KeyValueFile (Get-TrainingReportPath $LearningRate $Step)
  if ($map.checkpoint_parameter_hash -ne $trainingMap.final_parameter_hash) { throw "MUON_EVAL_PARAMETER_HASH_MISMATCH:${LearningRate}:$Step" }
  $val = [double]$map.validation_bits_per_utf8_byte; $dev = [double]$map.development_bits_per_utf8_byte; $balanced = ($val + $dev) / 2.0
  $trainingManifest = Load-Manifest $LearningRate
  $trainingMap['checkpoint_v4_decode'] = if ($trainingManifest -and $trainingManifest.ContainsKey('checkpoint_v4_decode')) { $trainingManifest.checkpoint_v4_decode } else { 'UNKNOWN' }
  Write-TelemetryRow $LearningRate $Step $trainingMap 'EVAL' $val $dev
  Write-Event @{ trial_id=Get-TrialId $LearningRate; phase="eval-$Phase"; optimizer='muon_aux_adam'; step=$Step; status='COMPLETED'; qnn_health='PASS'; validation_bpb=$val; development_bpb=$dev; balanced_bpb=$balanced }
  return [ordered]@{ path=$evalPath; map=$map; validation_bpb=$val; development_bpb=$dev; balanced_bpb=$balanced }
}

function New-Plan {
  [IO.Directory]::CreateDirectory((Join-Path $Ledger 'trials')) | Out-Null
  # Wrap the conditional as a whole so JSON keeps candidates as an array for
  # an explicit single-candidate recovery run as well as the default grid.
  $rates = @(
    if ((Get-OptimizerKind) -eq 'MUON') { Get-CandidateRates } else { '0.0022' }
  )
  foreach ($rate in $rates) {
    if ((Get-OptimizerKind) -eq 'MUON') {
      $existing = Load-Manifest $rate
      if ($null -eq $existing) { Save-Manifest $rate (New-Manifest $rate) } else { Assert-ManifestIdentity $existing $rate }
    }
  }
  $future = [ordered]@{
    training_runner = $TrainingRunnerResolved
    required_parameters = @('-Optimizer','-MuonLearningRate','-MuonMomentum','-MuonNsSteps')
    optional_parameters = @('-MuonNesterov','-MuonTargetLearningRate','-MuonLearningRateSchedule')
    semantics = 'Muon uses CPU reference update; Aux Adam remains frozen beta1=.9 beta2=.999 eps=1e-8 weight_decay=0; -ResumeStep is same-candidate only'
  }
  $plan = [ordered]@{
    schema_version = $SchemaVersion
    protocol = 'hexatrain-nicopedia-muon-pilot-v1'
    optimizer = Get-OptimizerKind
    optimizer_identity = Get-OptimizerIdentity
    architecture = $Fixed
    smoke = [ordered]@{ updates=$SmokeUpdates; fresh_start=$true; purpose='measure HTP forward/backward plus CPU Muon/Aux-Adam/transfer overhead before Stage-A quality ranking'; validation_chunks=0; development_chunks=0; final_test_opened=$false }
    stage_a = [ordered]@{ candidates=$rates; seed=1; fresh_start=$true; steps=1000; validation_chunks=256; development_chunks=256; final_test_opened=$false; prune='nonfinite/QNN failure/state corruption or clear divergence vs same-step Adam'; max_promote=2 }
    stage_b = [ordered]@{ input='top-two healthy Stage-A candidates'; resume_from_step=1000; steps=2000; same_candidate_resume=$true; validation_chunks=256; development_chunks=256; final_test_opened=$false; promote=1 }
    full_promotion = [ordered]@{ input='one promising Stage-B candidate'; checkpoints=@(4000,6000,8000); resume_parity_gate='required and must PASS before starting'; final_test_opened=$false }
    lr_schedule = [ordered]@{ aux_adam_peak='0.0022'; aux_adam_target='0.0001'; muon_peak_candidates=$MuonLearningRateCandidates; muon_target_expression='muon_peak * (0.0001 / 0.0022)'; shape='constant through step 4000, then linear decay to target at step 8000' }
    fixed_data = [ordered]@{ train_cache=$Fixed.dataset_identity; train_cache_content_hash=$Fixed.dataset_cache_content_hash; tokenizer_kind=$Fixed.tokenizer_kind; tokenizer_hash=$Fixed.tokenizer_hash; order_hash=$Fixed.training_order_hash; final_test_opened=$false }
    adam_reference = [ordered]@{ source=$AdamReferenceRoot; reused_only=$true; step=8000; val_bpb=$AdamReference.validation_bpb; dev_bpb=$AdamReference.development_bpb; balanced_bpb=$AdamReference.balanced_bpb }
    checkpoint = [ordered]@{ muon_format=$CheckpointFormatMuon; adam_reference_format=$CheckpointFormatAdam; muon_state='matrix momentum + Aux Adam m/v + optimizer identity/hyperparameters/global step'; adam_to_muon_resume='forbidden'; muon_to_adam_resume='forbidden' }
    execution_boundary = [ordered]@{ forward_backward_backend='HTP'; optimizer_muon_backend='CPU'; optimizer_aux_adam_backend='CPU'; qnn_graph_failure_fallback='forbidden'; telemetry='machine-readable per-trial telemetry.csv and manifest.json' }
    future_training_runner_arguments = $future
    created_utc = Get-NowUtc
  }
  $plan | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath (Join-Path $Ledger 'plan.json') -Encoding utf8
  return $plan
}

function Set-Pruned {
  param([Parameter(Mandatory = $true)][string]$LearningRate,[Parameter(Mandatory = $true)][string]$Reason)
  $manifest = Load-Manifest $LearningRate
  if ($null -eq $manifest) { $manifest = New-Manifest $LearningRate }
  $manifest.status = 'PRUNED'; $manifest.prune_reason = $Reason; $manifest.pruned_utc = Get-NowUtc; Save-Manifest $LearningRate $manifest
  Write-Event @{ trial_id=Get-TrialId $LearningRate; phase='prune'; optimizer='muon_aux_adam'; status='PRUNED'; reason=$Reason }
}

function Invoke-StageA {
  param([Parameter(Mandatory = $true)][string[]]$Rates)
  $rows = [Collections.Generic.List[object]]::new()
  $adamAnchor = Get-AdamAnchorMetric 1000
  foreach ($rate in $Rates) {
    try {
      # The short smoke is deliberately a fresh run and is never used as a
      # quality point.  Its machine-readable telemetry captures the hybrid
      # overhead before spending the 1000-update Stage-A budget.
      [void](Invoke-TrainingPhase $rate $SmokeUpdates 0 'smoke-8')
      $training = Invoke-TrainingPhase $rate 1000 0 'stage-a-1000'
      $evaluation = Invoke-EvaluationPhase $rate 1000 'stage-a-1000'
      if ($null -ne $adamAnchor -and $evaluation.balanced_bpb -gt $adamAnchor.balanced_bpb + $ClearDivergenceMarginBpb) {
        Set-Pruned $rate "clear divergence vs same-step Adam (balanced delta > $ClearDivergenceMarginBpb bpb)"
        continue
      }
      $manifest = Load-Manifest $rate; $manifest.stage_a = [ordered]@{ step=1000; validation_bpb=$evaluation.validation_bpb; development_bpb=$evaluation.development_bpb; balanced_bpb=$evaluation.balanced_bpb; health='PASS' }; Save-Manifest $rate $manifest
      $rows.Add([pscustomobject]@{ trial_id=Get-TrialId $rate; optimizer='muon_aux_adam'; muon_lr=$rate; stage='A'; step=1000; validation_bpb=$evaluation.validation_bpb; development_bpb=$evaluation.development_bpb; balanced_bpb=$evaluation.balanced_bpb; status='PASS' })
    } catch {
      if ($_.Exception.Message -match '^MUON_TRAINING_TRANSPORT_INTERRUPTED:') {
        $manifest = Load-Manifest $rate
        if ($null -eq $manifest) { $manifest = New-Manifest $rate }
        $manifest.status = 'BLOCKED_TRANSPORT'
        $manifest.blocked_reason = $_.Exception.Message
        Save-Manifest $rate $manifest
        Write-Event @{ trial_id=Get-TrialId $rate; phase='stage-a-transport'; optimizer='muon_aux_adam'; status='BLOCKED_TRANSPORT'; reason=$_.Exception.Message }
        throw
      }
      Set-Pruned $rate ("Stage-A failure: " + $_.Exception.Message)
    }
  }
  $healthy = @($Rates | ForEach-Object { $m=Load-Manifest $_; if ($m -and $m.status -in @('COMPLETED','PROMOTED','PROMOTED_STAGE_A','PROMOTED_STAGE_B') -and $m.stage_a) { [pscustomobject]@{ lr=$_; balanced=[double]$m.stage_a.balanced_bpb } } } | Sort-Object balanced)
  $promote = @($healthy | Select-Object -First 2)
  foreach ($item in $promote) { $m=Load-Manifest $item.lr; $m.status='PROMOTED_STAGE_A'; $m.stage_a_promotion_reason='top-two healthy Stage-A balanced bpb'; Save-Manifest $item.lr $m }
  foreach ($item in @($healthy | Select-Object -Skip 2)) { Set-Pruned $item.lr 'not in Stage-A top-two promotion set' }
  if ($rows.Count -gt 0) { Write-SummaryRows @($rows) }
  return $promote
}

function Invoke-StageB {
  param([Parameter(Mandatory = $true)][object[]]$Promoted)
  $rows = [Collections.Generic.List[object]]::new()
  $adamAnchor = Get-AdamAnchorMetric 2000
  foreach ($item in $Promoted) {
    $rate = [string]$item.lr
    try {
      $training = Invoke-TrainingPhase $rate 2000 1000 'stage-b-2000'
      $evaluation = Invoke-EvaluationPhase $rate 2000 'stage-b-2000'
      if ($null -ne $adamAnchor -and $evaluation.balanced_bpb -gt $adamAnchor.balanced_bpb + $ClearDivergenceMarginBpb) {
        Set-Pruned $rate "clear divergence vs same-step Adam (balanced delta > $ClearDivergenceMarginBpb bpb)"
        continue
      }
      $manifest = Load-Manifest $rate; $manifest.stage_b = [ordered]@{ step=2000; validation_bpb=$evaluation.validation_bpb; development_bpb=$evaluation.development_bpb; balanced_bpb=$evaluation.balanced_bpb; health='PASS' }; Save-Manifest $rate $manifest
      $rows.Add([pscustomobject]@{ trial_id=Get-TrialId $rate; optimizer='muon_aux_adam'; muon_lr=$rate; stage='B'; step=2000; validation_bpb=$evaluation.validation_bpb; development_bpb=$evaluation.development_bpb; balanced_bpb=$evaluation.balanced_bpb; status='PASS' })
    } catch {
      if ($_.Exception.Message -match '^MUON_TRAINING_TRANSPORT_INTERRUPTED:') {
        $manifest = Load-Manifest $rate
        if ($null -eq $manifest) { $manifest = New-Manifest $rate }
        $manifest.status = 'BLOCKED_TRANSPORT'
        $manifest.blocked_reason = $_.Exception.Message
        Save-Manifest $rate $manifest
        Write-Event @{ trial_id=Get-TrialId $rate; phase='stage-b-transport'; optimizer='muon_aux_adam'; status='BLOCKED_TRANSPORT'; reason=$_.Exception.Message }
        throw
      }
      Set-Pruned $rate ("Stage-B failure: " + $_.Exception.Message)
    }
  }
  $healthy = @($Promoted | ForEach-Object { $rate=[string]$_.lr; $m=Load-Manifest $rate; if ($m -and $m.status -in @('COMPLETED','PROMOTED_STAGE_A','PROMOTED_STAGE_B') -and $m.stage_b) { [pscustomobject]@{ lr=$rate; balanced=[double]$m.stage_b.balanced_bpb } } } | Sort-Object balanced)
  $best = @($healthy | Select-Object -First 1)
  foreach ($item in @($healthy | Select-Object -Skip 1)) { Set-Pruned $item.lr 'not promoted beyond Stage-B best candidate' }
  if ($rows.Count -gt 0) { Write-SummaryRows @($rows) }
  if ($best.Count -eq 0) { return $null }
  $bestManifest = Load-Manifest $best[0].lr
  # The verified S4000 Adam artifacts do not contain a matching 256+256
  # step-2000 evaluation.  Do not infer a learning-curve comparison from a
  # different sample identity or from the final endpoint; stop before the
  # long run and leave an auditable blocked state instead.
  if ($null -eq $adamAnchor) {
    $bestManifest.status='BLOCKED'; $bestManifest.stage_b_verdict='matching Adam step-2000 Val/Dev anchor unavailable; no cross-sample inference'; $bestManifest.full_promotion='FORBIDDEN_UNTIL_MATCHING_ADAM_ANCHOR'; Save-Manifest $best[0].lr $bestManifest
    $script:StageBDecision = 'BLOCKED'
    return $null
  }
  $adamFinal = [double]$AdamReference.balanced_bpb
  $adamAt2000 = [double]$adamAnchor.balanced_bpb
  $improved = $best[0].balanced -le $adamFinal - $PromisingImprovementBpb
  $fastCurve = $false
  $fastCurve = $best[0].balanced -le $adamAt2000 + $FastCurveToleranceBpb
  if (-not $improved -and -not $fastCurve) {
    $bestManifest.status='MUON_PILOT_NEGATIVE'; $bestManifest.stage_b_verdict='not improved and no faster learning curve at step 2000'; Save-Manifest $best[0].lr $bestManifest
    return $null
  }
  $bestManifest.status='PROMOTED_STAGE_B'; $bestManifest.stage_b_verdict = if ($improved) { 'improved vs Adam final target margin' } else { 'learning curve near/equal to Adam at same step' }; Save-Manifest $best[0].lr $bestManifest
  return $best[0]
}

function Invoke-FullPromotion {
  param([Parameter(Mandatory = $true)]$Candidate)
  $rate = [string]$Candidate.lr
  Assert-ResumeParityGate $rate
  $rows = [Collections.Generic.List[object]]::new()
  $resume = 2000
  foreach ($step in @(4000,6000,8000)) {
    $training = Invoke-TrainingPhase $rate $step $resume "full-$step"
    $evaluation = Invoke-EvaluationPhase $rate $step "full-$step"
    $rows.Add([pscustomobject]@{ trial_id=Get-TrialId $rate; optimizer='muon_aux_adam'; muon_lr=$rate; stage='Full'; step=$step; validation_bpb=$evaluation.validation_bpb; development_bpb=$evaluation.development_bpb; balanced_bpb=$evaluation.balanced_bpb; status='PASS' })
    $resume = $step
  }
  Write-SummaryRows @($rows)
  $manifest = Load-Manifest $rate
  $final = @($rows | Where-Object { [int]$_.step -eq 8000 })[0]
  $manifest.status='COMPLETED'; $manifest.full_seed1 = [ordered]@{ step=8000; validation_bpb=[double]$final.validation_bpb; development_bpb=[double]$final.development_bpb; balanced_bpb=[double]$final.balanced_bpb; health='PASS'; resume_parity='PASS' }; Save-Manifest $rate $manifest
}

function Invoke-Run {
  if ((Get-OptimizerKind) -eq 'ADAM') { [void](Get-AdamReferenceEvalMap); New-Plan | Out-Null; Write-Host 'Adam baseline is reference-only; no baseline retraining performed.'; return }
  if (-not $QairtSdkRoot -or -not $ExpectedBuildId) { throw 'RUN_REQUIRES_EXPLICIT_QAIRT_ARGUMENTS' }
  . (Join-Path $PSScriptRoot 'qairt_version.ps1')
  Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
  if (-not (Test-Path -LiteralPath $TrainingRunnerResolved -PathType Leaf) -or -not (Test-Path -LiteralPath $EvalRunnerResolved -PathType Leaf)) { throw 'PILOT_RUNNER_DEPENDENCY_MISSING' }
  $script:RunnerParameterNames = Get-RunnerParameterNames $TrainingRunnerResolved
  Assert-MuonRunnerInterface $script:RunnerParameterNames
  if (-not (Test-Path -LiteralPath $TrainCacheResolved -PathType Leaf) -or -not (Test-Path -LiteralPath $TokenizerResolved -PathType Leaf)) { throw 'MUON_PRIVATE_INPUT_MISSING' }
  Assert-NoFinalTestPath $TrainCacheResolved 'TrainCache'
  Assert-NoFinalTestPath $TokenizerResolved 'Tokenizer'
  [void](Get-AdamReferenceEvalMap)
  New-Plan | Out-Null
  $rates = @(Get-CandidateRates)
  $stageARows = @()
  $promoted = @()
  if ($Stage -in @('Full','StageA')) { $promoted = @(Invoke-StageA $rates) }
  if ($Stage -eq 'StageA') { return }
  if ($Stage -eq 'StageB') {
    $promoted = @($rates | ForEach-Object { $m=Load-Manifest $_; if ($m -and $m.status -eq 'PROMOTED_STAGE_A') { [pscustomobject]@{ lr=$_ } } })
  }
  if (@($promoted).Count -eq 0) { throw 'MUON_PILOT_NEGATIVE_NO_STAGE_A_SURVIVORS' }
  $script:StageBDecision = 'UNKNOWN'
  $stageB = Invoke-StageB $promoted
  if ($null -eq $stageB) {
    if ($script:StageBDecision -eq 'BLOCKED') { Write-Host 'MUON PILOT BLOCKED: matching Adam step-2000 Val/Dev anchor is unavailable.' }
    else { Write-Host 'MUON PILOT NEGATIVE: no Stage-B candidate met the declared promotion criterion.' }
    return
  }
  if ($Stage -eq 'StageB') { return }
  Invoke-FullPromotion $stageB
}

function Invoke-Summarize {
  $paths = @(Get-ChildItem -LiteralPath (Join-Path $Ledger 'trials') -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName 'manifest.json' } | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
  if ($paths.Count -eq 0) { Write-Host 'No Muon pilot manifests found.'; return }
  $manifests = @($paths | ForEach-Object { Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json })
  $manifests | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath (Join-Path $Ledger 'manifests.json') -Encoding utf8
  if (Test-Path -LiteralPath (Join-Path $Ledger 'summary.csv') -PathType Leaf) { Get-Content -LiteralPath (Join-Path $Ledger 'summary.csv') }
  foreach ($manifest in $manifests) { Write-Output ("{0} status={1} steps={2}" -f $manifest.trial_id,$manifest.status,$manifest.completed_steps) }
}

function Assert-PilotMetadataDerivationRejects([string]$Path, [string]$ExpectedPrefix, [string]$Label) {
  $rejected = $false
  try {
    [void](Get-MetadataDerivationFromSsot -MetadataPath $Path)
  } catch {
    $rejected = ([string]$_.Exception.Message).StartsWith($ExpectedPrefix, [StringComparison]::Ordinal)
  }
  if (-not $rejected) { throw "SELFTEST_PILOT_METADATA_FAIL_CLOSED:$Label" }
}

function Invoke-SelfTest {
  if ($Fixed.vocabulary -ne 1024 -or $Fixed.tokens -ne 32 -or $Fixed.dimension -ne 64 -or $Fixed.feed_forward_dimension -ne 128 -or $Fixed.layers -ne 19 -or $Fixed.heads -ne 2 -or $Fixed.parameter_count -ne 758528 -or $Fixed.batch_size -ne 8 -or $Fixed.seed -ne 1) { throw 'SELFTEST_FIXED_ARCHITECTURE' }
  # Valid current metadata must keep the frozen pilot counts unchanged.
  if ([int64]$MetadataDerived.parameter_count -ne 758528 -or
      [int64]$MetadataDerived.muon_parameter_count -ne 622592 -or
      [int64]$MetadataDerived.aux_adam_parameter_count -ne 135936 -or
      [int64]$MetadataDerived.muon_matrix_count -ne 114 -or
      [string]($MetadataDerived.muon_parameter_suffixes -join ',') -ne 'wq,wk,wv,wo,ffn_w1,ffn_w2') {
    throw 'SELFTEST_PILOT_METADATA_VALID_BASELINE'
  }
  $temp = Join-Path ([IO.Path]::GetTempPath()) ('phonelm-muon-pilot-metadata-selftest-' + [guid]::NewGuid().ToString('N'))
  [IO.Directory]::CreateDirectory($temp) | Out-Null
  try {
    function New-PilotMetadataDefinition {
      param(
        [string]$Condition = 'ALWAYS',
        [string]$Placement = 'GLOBAL_PREFIX',
        [string]$Role = 'AUX_ADAM',
        $Shape = @('VOCABULARY', 'MODEL')
      )
      return @{
        suffix = 'token_embedding'
        role = $Role
        placement = $Placement
        condition = $Condition
        shape = $Shape
        rank = 2
        fan_out_axis = -1
        fan_in_axis = -1
      }
    }
    $cases = @(
      @{ Name = 'unknown-condition'; Expected = 'PARAMETER_METADATA_CONDITION_UNKNOWN:'; Definition = (New-PilotMetadataDefinition -Condition 'SOMETIMES') },
      @{ Name = 'unknown-placement'; Expected = 'PARAMETER_METADATA_PLACEMENT_UNKNOWN:'; Definition = (New-PilotMetadataDefinition -Placement 'EVERYWHERE') },
      @{ Name = 'unknown-role'; Expected = 'PARAMETER_METADATA_ROLE_UNKNOWN:'; Definition = (New-PilotMetadataDefinition -Role 'ADAM') },
      @{ Name = 'unknown-dimension'; Expected = 'PARAMETER_METADATA_DIMENSION_UNKNOWN:'; Definition = (New-PilotMetadataDefinition -Shape @('CHANNELS')) },
      @{ Name = 'missing-schema-version'; Expected = 'PARAMETER_METADATA_SCHEMA_VERSION_UNSUPPORTED:'; Definition = (New-PilotMetadataDefinition); Schema = $null },
      @{ Name = 'unsupported-schema-version'; Expected = 'PARAMETER_METADATA_SCHEMA_VERSION_UNSUPPORTED:2'; Definition = (New-PilotMetadataDefinition); Schema = 2 }
    )
    foreach ($case in $cases) {
      $caseDir = Join-Path $temp $case.Name
      [IO.Directory]::CreateDirectory($caseDir) | Out-Null
      $schema = 1
      if ($case.ContainsKey('Schema')) { $schema = $case.Schema }
      $payload = if ($null -eq $schema) {
        [ordered]@{ parameter_definitions = @($case.Definition) }
      } else {
        [ordered]@{ schema_version = $schema; parameter_definitions = @($case.Definition) }
      }
      $path = Join-Path $caseDir 'transformer_parameter_metadata.json'
      $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding utf8
      Assert-PilotMetadataDerivationRejects $path $case.Expected $case.Name
    }
  } finally {
    Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (@($MuonLearningRateCandidates).Count -ne 3 -or ($MuonLearningRateCandidates -join ',') -ne '0.005,0.010,0.020') { throw 'SELFTEST_MUON_LR_CANDIDATES' }
  if ($SmokeUpdates -lt 8 -or $SmokeUpdates -gt 32) { throw 'SELFTEST_SMOKE_UPDATE_RANGE' }
  if ([math]::Abs((Get-ExpectedMuonTargetLearningRate 0.005) - 0.00022727272727272728) -gt 1.0e-14 -or [math]::Abs((Get-ExpectedMuonTargetLearningRate 0.010) - 0.00045454545454545456) -gt 1.0e-14 -or [math]::Abs((Get-ExpectedMuonTargetLearningRate 0.020) - 0.0009090909090909091) -gt 1.0e-14) { throw 'SELFTEST_MUON_TARGET_SCALING' }
  foreach ($rate in $MuonLearningRateCandidates) {
    $peak = [double]$rate
    if ([math]::Abs((Get-MuonScheduledLearningRate $peak 4000) - $peak) -gt 1.0e-14 -or [math]::Abs((Get-MuonScheduledLearningRate $peak 8000) - (Get-ExpectedMuonTargetLearningRate $peak)) -gt 1.0e-14) { throw "SELFTEST_MUON_SCHEDULE_BOUNDARY:$rate" }
  }
  if ((Get-ExpectedTrainingOrderHash 8) -ne 'fnv1a64:0e2e15196d851431' -or
      (Get-ExpectedTrainingOrderHash 1000) -ne 'fnv1a64:b8aa576130682a50') { throw "SELFTEST_TRAINING_ORDER_HASH:8=$(Get-ExpectedTrainingOrderHash 8):1000=$(Get-ExpectedTrainingOrderHash 1000)" }
  if ([math]::Abs((Get-AuxAdamScheduledLearningRate 4000) - 0.0022) -gt 1.0e-14 -or [math]::Abs((Get-AuxAdamScheduledLearningRate 8000) - 0.0001) -gt 1.0e-14 -or [math]::Abs((Get-AuxAdamScheduledLearningRate 6000) - 0.00115) -gt 1.0e-14) { throw 'SELFTEST_AUX_ADAM_SCHEDULE' }
  if ((Get-TrialId '0.005') -ne 'muon-pilot-lr0p005-seed1' -or (Get-TrialId '0.010') -ne 'muon-pilot-lr0p010-seed1' -or (Get-TrialId '0.020') -ne 'muon-pilot-lr0p020-seed1') { throw 'SELFTEST_TRIAL_ID' }
  if ((Get-OptimizerKind) -eq 'MUON') {
    if ((Get-OptimizerIdentity) -ne 'muon_aux_adam') { throw 'SELFTEST_OPTIMIZER_IDENTITY' }
    $manifest = New-Manifest '0.005'
    if ($manifest.muon_matrix_count -ne $MetadataDerived.muon_matrix_count -or $manifest.muon_parameter_count -ne $MetadataDerived.muon_parameter_count -or $manifest.aux_adam_parameter_count -ne $MetadataDerived.aux_adam_parameter_count -or $manifest.optimizer -ne 'MUON' -or $manifest.optimizer_identity -ne 'muon_aux_adam' -or $manifest.muon_algorithm_identity -ne $MuonAlgorithmIdentity -or $manifest.muon_learning_rate_schedule -ne 'linear_decay' -or $manifest.checkpoint_format -ne 'NPRTCKPTV4' -or $manifest.initial_parameter_hash -ne $Fixed.initial_parameter_hash -or $manifest.aux_adam_beta1 -ne $Fixed.adam_beta1 -or $manifest.aux_adam_beta2 -ne $Fixed.adam_beta2 -or $manifest.aux_adam_epsilon -ne $Fixed.adam_epsilon -or $manifest.aux_adam_weight_decay -ne $Fixed.weight_decay -or $manifest.final_test_opened -ne $false -or $manifest.final_test_used -ne $false -or [string]($manifest.muon_parameter_roles -join ',') -ne [string]($MetadataDerived.muon_parameter_roles -join ',') -or [string]($manifest.aux_adam_parameter_roles -join ',') -ne [string]($MetadataDerived.aux_adam_parameter_roles -join ',') -or [string]($manifest.muon_parameter_suffixes -join ',') -ne [string]($MetadataDerived.muon_parameter_suffixes -join ',') -or [string]($manifest.aux_adam_parameter_suffixes -join ',') -ne [string]($MetadataDerived.aux_adam_parameter_suffixes -join ',')) { throw 'SELFTEST_PARAMETER_SPLIT_OR_MANIFEST' }
    $knownCheckpoint = Join-Path (Get-TrainingDirectory '0.005') (Get-CheckpointName 1000)
    if (Test-Path -LiteralPath $knownCheckpoint -PathType Leaf) {
      $knownIdentity = Get-CoreCheckpointIdentity $knownCheckpoint
      if ($knownIdentity.magic -ne $CheckpointFormatMuon -or $knownIdentity.optimizer_identity -ne 'muon_aux_adam' -or $knownIdentity.schema_version -ne 4 -or $knownIdentity.registry_version -ne 1 -or $knownIdentity.registry_count -ne 192 -or @([uint64]$Fixed.training_order_seed,[uint64]$Fixed.seed) -notcontains [uint64]$knownIdentity.order_seed) { throw 'SELFTEST_V4_CHECKPOINT_IDENTITY' }
    }
    # Legacy resume/recovery compatibility: manifests written before the suffix
    # fields existed keep the original semantic labels and remain accepted.
    # Existing field meaning must not change under schema_version.
    $legacyManifest = @{}
    foreach ($key in $manifest.Keys) {
      if ($key -in @('muon_parameter_suffixes','aux_adam_parameter_suffixes')) { continue }
      $legacyManifest[$key] = $manifest[$key]
    }
    $legacyManifest['muon_parameter_roles'] = @('Wq','Wk','Wv','Wo','FFN_W1','FFN_W2')
    $legacyManifest['aux_adam_parameter_roles'] = @('token_embedding','output_projection','norm_scale_gain','bias','other_non_hidden')
    Assert-ManifestIdentity $legacyManifest '0.005'
    if ($legacyManifest.ContainsKey('muon_parameter_suffixes') -or $legacyManifest.ContainsKey('aux_adam_parameter_suffixes')) {
      throw 'SELFTEST_LEGACY_MANIFEST_MUST_OMIT_SUFFIX_FIELDS'
    }
    if ([string]($legacyManifest.muon_parameter_roles -join ',') -ne 'Wq,Wk,Wv,Wo,FFN_W1,FFN_W2' -or
        [string]($legacyManifest.aux_adam_parameter_roles -join ',') -ne 'token_embedding,output_projection,norm_scale_gain,bias,other_non_hidden') {
      throw 'SELFTEST_LEGACY_ROLE_LABEL_MEANING_CHANGED'
    }
  } else {
    if ((Get-OptimizerIdentity) -ne 'adam') { throw 'SELFTEST_ADAM_OPTIMIZER_IDENTITY' }
    $manifest = New-Manifest '0.0022'
    if ($manifest.optimizer -ne 'ADAM' -or $manifest.optimizer_identity -ne 'adam' -or $manifest.checkpoint_format -ne 'NPRTCKPTV3' -or $manifest.final_test_opened -ne $false -or $manifest.final_test_used -ne $false) { throw 'SELFTEST_ADAM_MANIFEST' }
  }
  if ($RequiredMuonRunnerParameters -notcontains 'Optimizer' -or $RequiredMuonRunnerParameters -notcontains 'MuonLearningRate' -or $RequiredMuonRunnerParameters -notcontains 'MuonMomentum' -or $RequiredMuonRunnerParameters -notcontains 'MuonNsSteps' -or $OptionalMuonRunnerParameters -notcontains 'MuonNesterov') { throw 'SELFTEST_FUTURE_RUNNER_ARGUMENTS' }
  $fake = @([pscustomobject]@{lr='0.005';balanced=2.50},[pscustomobject]@{lr='0.010';balanced=2.45},[pscustomobject]@{lr='0.020';balanced=2.60}) | Sort-Object balanced
  if ($fake[0].lr -ne '0.010' -or @($fake | Select-Object -First 2).Count -ne 2) { throw 'SELFTEST_STAGE_PROMOTION' }
  $fakeResume = [ordered]@{ optimizer='MUON'; completed_steps=1000; resume_from_adam_forbidden=$true }
  if ($fakeResume.optimizer -ne 'MUON' -or -not $fakeResume.resume_from_adam_forbidden) { throw 'SELFTEST_RESUME_POLICY' }
  try { Assert-NoFinalTestPath 'build/private-data/final_test/validation.bin' 'SelfTest'; throw 'SELFTEST_FINAL_TEST_GUARD' } catch { if ($_.Exception.Message -ne 'SelfTest_FINAL_TEST_FORBIDDEN') { throw } }
  if ($CheckpointFormatMuon -eq $CheckpointFormatAdam) { throw 'SELFTEST_CHECKPOINT_FORMAT_COLLISION' }
  Write-Host 'run_nicopedia_muon_pilot_self_test=PASS'
}

# Resolve all path arguments only after function definitions.  SelfTest does
# not require the private corpus, QAIRT SDK, or a device, but it still parses
# the production paths and validates that they are below build when supplied.
$Ledger = Resolve-UnderBuild $LedgerRoot 'LedgerRoot'
$TrainingRunnerResolved = if ([IO.Path]::IsPathRooted($TrainingRunnerPath)) { [IO.Path]::GetFullPath($TrainingRunnerPath) } else { [IO.Path]::GetFullPath((Join-Path $Root $TrainingRunnerPath)) }
$EvalRunnerResolved = if ([IO.Path]::IsPathRooted($EvalRunnerPath)) { [IO.Path]::GetFullPath($EvalRunnerPath) } else { [IO.Path]::GetFullPath((Join-Path $Root $EvalRunnerPath)) }
$TrainCacheResolved = if ($TrainCachePath) { Resolve-UnderBuild $TrainCachePath 'TrainCachePath' } else { Resolve-UnderBuild (Join-Path $TrainingDataRoot 'caches/train_pilot.bin') 'TrainCachePath' }
$TokenizerResolved = if ($TokenizerModelPath) { Resolve-UnderBuild $TokenizerModelPath 'TokenizerModelPath' } else { Resolve-UnderBuild (Join-Path $TrainingDataRoot 'tokenizer/byte-bpe-v1024.model') 'TokenizerModelPath' }
$EvalCacheResolved = Resolve-UnderBuild (Join-Path $TrainingDataRoot 'caches') 'EvalCacheRoot'
Assert-NoFinalTestPath $Ledger 'LedgerRoot'
Assert-NoFinalTestPath $EvalCacheResolved 'EvalCacheRoot'

if ($SelfTest) { Invoke-SelfTest; exit 0 }
if ($Mode -eq 'Plan') { New-Plan | Out-Null; Write-Host "Muon pilot plan written under build ledger."; exit 0 }
if ($Mode -eq 'Run') { Invoke-Run; exit 0 }
Invoke-Summarize
