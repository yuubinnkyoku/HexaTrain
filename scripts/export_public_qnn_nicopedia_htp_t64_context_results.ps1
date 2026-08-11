# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Allow-list exporter for the aggregate-only Nicopedia L19 T64-context
# long-training bundle. Mirrors the canonical T32 long-training exporter:
# only pre-staged aggregate CSVs are emitted, every value is checked against
# a safe charset and a forbidden-pattern list, and the manifest enforces the
# private-data / no-fallback policy.
param(
  [string]$SourceRoot = 'build/private-diagnostics/nicopedia-htp-t64-context-public-source',
  [string]$OutputRoot = 'docs/results/qnn-nicopedia-htp-t64-context-2026-08',
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

$schemas = [ordered]@{
  't64-full-cap-evaluation.csv' = @('step','parameter_hash','htp_validation_nll','cpu_validation_nll','htp_validation_perplexity','cpu_validation_perplexity','htp_validation_top1','cpu_validation_top1','htp_validation_top5','cpu_validation_top5','htp_development_nll','cpu_development_nll','htp_development_perplexity','cpu_development_perplexity','htp_development_top1','cpu_development_top1','htp_development_top5','cpu_development_top5','graph_executes','qnn_failures','nonfinite_chunks','cpu_fallback')
  't64-trajectory.csv' = @('step','parameter_hash','validation_nll','development_nll','finite')
  't64-segment-health.csv' = @('resume_step','completed_step','first_loss','last_loss','training_seconds','graph_executes','qnn_failures','all_steps_finite','optimizer_finite','checkpoint_count','checkpoint_written','cpu_replay_performed','cpu_fallback','thermal_status_after','battery_health_after','battery_temperature_c_after')
}
$manifestFields = @('milestone','date','seed_count','seed','start_step','final_step','best_validation_step','best_development_step','batch_size','learning_rate','checkpoint_format','checkpoint_interval','qairt_build_id','legacy_parity_thresholds_changed','legacy_generation_gate','htp_native_generation_policy','cpu_fallback','nicopedia_final_test_opened','synthetic_final_holdout_opened','raw_private_data_included')
$safeValue = '^[A-Za-z0-9_.:+-]+$'
$forbidden = '(?i)generated_hex|prompt_sha256|prompt\.bin|device_serial|adb_endpoint|/data/(?:local|user)/|[A-Za-z]:\\Users\\|\.ckpt(?:\b|$)|raw_logits|raw_activations'

$readme = @'
# Nicopedia L19 HTP-native T64-context training, August 2026

This public bundle contains aggregate-only evidence for canonical L19
training with a 64-token context (T64), optimizer step 0 through 8,000. The
numerical operations of each training step ran in explicit QNN HTP graphs;
host control and input preparation remained on CPU. This is not an NPU-only
claim and QNN automatic differentiation was not used.

The best and final checkpoint is step 8,000. The 250-step CPU evaluator
screening shows held-out validation NLL improving from 3.0133 at step 250 to
2.3487 at step 8,000; development NLL improves from 2.6632 to 2.0544. The
full-cap HTP-native validation at step 8,000 is 2.150418944 (CPU
2.150434198) and development is 2.131013825 (CPU 2.131041532), measured over
12,288 graph executes with zero failures, zero nonfinite chunks, and no CPU
fallback.

Training stopped at the predeclared hard ceiling of 8,000 steps, not because
a plateau was proven. Every 1,000-step segment resumed from the previous
segment's NPRTCKPTV2 checkpoint; the checkpoint parameter hashes, steps, and
config identity were verified on-device and on-host at each boundary. The
T64 run is a context-extension milestone and does not by itself change the
legacy CPU-equivalence parity thresholds.

Files:

- `t64-full-cap-evaluation.csv`: HTP-native and CPU held-out metrics at the
  final checkpoint.
- `t64-trajectory.csv`: 250-step CPU evaluator screening results.
- `t64-segment-health.csv`: resume segment health and runtime aggregates.
- `manifest.json`: scope, policy, and final-test status.

Private dataset text, prompts, generated content, raw bytes/tokens, logits,
checkpoints, device identifiers, endpoints, and local paths are excluded.
Nicopedia final test and the synthetic final holdout were not opened.
'@

function Resolve-Source([string]$Path) {
  $full = [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
  $allowed = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
  if (-not ($full + [IO.Path]::DirectorySeparatorChar).StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { throw 'SOURCE_ROOT_OUTSIDE_BUILD' }
  return $full
}
function Resolve-Output([string]$Path, [switch]$AllowBuild) {
  $full = [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
  $roots = @([IO.Path]::GetFullPath((Join-Path $repoRoot 'docs/results')))
  if ($AllowBuild) { $roots += [IO.Path]::GetFullPath((Join-Path $repoRoot 'build/export-selftest')) }
  if (-not @($roots | Where-Object { ($full + [IO.Path]::DirectorySeparatorChar).StartsWith($_.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) }).Count) { throw 'OUTPUT_ROOT_NOT_ALLOWED' }
  return $full
}
function Export-Bundle([string]$Source, [string]$Output) {
  if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw 'PUBLIC_AGGREGATE_SOURCE_MISSING' }
  [IO.Directory]::CreateDirectory($Output) | Out-Null
  $allowedOutputNames = @($schemas.Keys) + @('manifest.json','README.md')
  $extraOutput = @(Get-ChildItem -LiteralPath $Output | Where-Object { $_.PSIsContainer -or $_.Name -notin $allowedOutputNames })
  if ($extraOutput.Count -gt 0) { throw 'PUBLIC_OUTPUT_EXTRA_FILE_REJECTED' }
  foreach ($entry in $schemas.GetEnumerator()) {
    $path = Join-Path $Source $entry.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "PUBLIC_AGGREGATE_FILE_MISSING: $($entry.Key)" }
    $rows = @(Import-Csv -LiteralPath $path)
    if ($rows.Count -eq 0) { throw "PUBLIC_AGGREGATE_EMPTY: $($entry.Key)" }
    $actual = @($rows[0].PSObject.Properties.Name)
    if (($actual -join ',') -ne ($entry.Value -join ',')) { throw "PUBLIC_AGGREGATE_SCHEMA_REJECTED: $($entry.Key)" }
    $lines = @($entry.Value -join ',')
    foreach ($row in $rows) {
      $values = foreach ($field in $entry.Value) {
        $value = [string]$row.$field
        if ($value -notmatch $safeValue -or $value -match $forbidden) { throw "PUBLIC_AGGREGATE_VALUE_REJECTED: $($entry.Key)/$field" }
        $value
      }
      $lines += ($values -join ',')
    }
    [IO.File]::WriteAllLines((Join-Path $Output $entry.Key),$lines,[Text.UTF8Encoding]::new($false))
  }
  $manifestPath = Join-Path $Source 'manifest.json'
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $actualManifest = @($manifest.PSObject.Properties.Name)
  if (($actualManifest -join ',') -ne ($manifestFields -join ',')) { throw 'PUBLIC_MANIFEST_SCHEMA_REJECTED' }
  $publicManifest = [ordered]@{}
  foreach ($field in $manifestFields) { $publicManifest[$field] = $manifest.$field }
  if ($publicManifest.raw_private_data_included -ne $false -or $publicManifest.cpu_fallback -ne $false -or $publicManifest.nicopedia_final_test_opened -ne $false -or $publicManifest.synthetic_final_holdout_opened -ne $false) { throw 'PUBLIC_MANIFEST_POLICY_REJECTED' }
  [IO.File]::WriteAllText((Join-Path $Output 'manifest.json'),($publicManifest | ConvertTo-Json -Depth 4) + "`n",[Text.UTF8Encoding]::new($false))
  [IO.File]::WriteAllText((Join-Path $Output 'README.md'),$readme.TrimStart() + "`n",[Text.UTF8Encoding]::new($false))
  $publicText = (Get-ChildItem -LiteralPath $Output -File | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join "`n"
  if ($publicText -match $forbidden) { throw 'PUBLIC_BUNDLE_LEAK_REJECTED' }
}

if ($SelfTest) {
  $source = Join-Path $repoRoot 'build/export-selftest/nicopedia-t64-source'
  $output = Join-Path $repoRoot 'build/export-selftest/nicopedia-t64-output'
  [IO.Directory]::CreateDirectory($source) | Out-Null
  $seedRows = @{
    't64-full-cap-evaluation.csv' = @('8000,fnv1a64:0000000000000000,2.1,2.1,8.2,8.2,0.25,0.25,0.5,0.5,2.0,2.0,7.4,7.4,0.26,0.26,0.52,0.52,24576,0,0,false')
    't64-trajectory.csv' = @('250,fnv1a64:0000000000000001,3.013269539,2.9,true','500,fnv1a64:0000000000000002,2.84,2.8,true')
    't64-segment-health.csv' = @('0,1000,3.3,2.66,1600,100,0,true,true,3,true,true,false,0,GOOD,33')
  }
  foreach ($name in @($schemas.Keys)) {
    $lines = @($schemas[$name] -join ',') + $seedRows[$name]
    [IO.File]::WriteAllLines((Join-Path $source $name),$lines,[Text.UTF8Encoding]::new($false))
  }
  $manifestSeed = [ordered]@{
    milestone = 't64-context'; date = '2026-08-11'; seed_count = 1; seed = 1; start_step = 0; final_step = 8000
    best_validation_step = 8000; best_development_step = 8000; batch_size = 8; learning_rate = '0.003000000026'
    checkpoint_format = 'NPRTCKPTV2'; checkpoint_interval = 250; qairt_build_id = '2.48.40.260702151143'
    legacy_parity_thresholds_changed = $false; legacy_generation_gate = 'reject'; htp_native_generation_policy = 'health-gated-experimental'
    cpu_fallback = $false; nicopedia_final_test_opened = $false; synthetic_final_holdout_opened = $false; raw_private_data_included = $false
  }
  [IO.File]::WriteAllText((Join-Path $source 'manifest.json'),($manifestSeed | ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
  Export-Bundle $source $output
  $malicious = Join-Path $source 't64-trajectory.csv'
  [IO.File]::WriteAllText($malicious, (Get-Content -LiteralPath $malicious -Raw) + 'prompt_sha256=abc' + "`n", [Text.UTF8Encoding]::new($false))
  $rejected = $false
  try { Export-Bundle $source (Join-Path $repoRoot 'build/export-selftest/nicopedia-t64-rejected') } catch { $rejected = $_.Exception.Message -match 'VALUE_REJECTED' }
  if (-not $rejected) { throw 'PUBLIC_EXPORTER_NEGATIVE_SELFTEST_FAILED' }
  $extraOutput = Join-Path $repoRoot 'build/export-selftest/nicopedia-t64-extra-output'
  [IO.Directory]::CreateDirectory($extraOutput) | Out-Null
  [IO.File]::WriteAllText((Join-Path $extraOutput 'private-generated-text.txt'),'synthetic-private-sentinel',[Text.UTF8Encoding]::new($false))
  $extraRejected = $false
  try { Export-Bundle $source $extraOutput } catch { $extraRejected = $_.Exception.Message -eq 'PUBLIC_OUTPUT_EXTRA_FILE_REJECTED' }
  if (-not $extraRejected) { throw 'PUBLIC_EXPORTER_STALE_OUTPUT_SELFTEST_FAILED' }
  Write-Host 'export_public_qnn_nicopedia_htp_t64_context_self_test=PASS'
  exit 0
}

$source = Resolve-Source $SourceRoot
$output = Resolve-Output $OutputRoot
Export-Bundle $source $output
Write-Host "public_nicopedia_htp_t64_context_export=PASS output=$OutputRoot"
