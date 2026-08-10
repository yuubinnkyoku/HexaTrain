# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Allow-list exporter for the aggregate-only Nicopedia L19 long-training bundle.
param(
  [string]$SourceRoot = 'build/private-diagnostics/nicopedia-htp-long-training-public-source',
  [string]$OutputRoot = 'docs/results/qnn-nicopedia-htp-long-training-2026-08',
  [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

$schemas = [ordered]@{
  'full-cap-evaluation.csv' = @('step','parameter_hash','htp_validation_nll','cpu_validation_nll','htp_validation_perplexity','cpu_validation_perplexity','htp_validation_top1','cpu_validation_top1','htp_validation_top5','cpu_validation_top5','htp_development_nll','cpu_development_nll','htp_development_perplexity','cpu_development_perplexity','htp_development_top1','cpu_development_top1','htp_development_top5','cpu_development_top5','graph_executes','qnn_failures','nonfinite_chunks','cpu_fallback')
  'medium-trajectory.csv' = @('step','validation_nll','development_nll','finite')
  'segment-health.csv' = @('resume_step','completed_step','first_loss','last_loss','training_seconds','graph_executes','qnn_failures','all_steps_finite','optimizer_finite','checkpoint_count','checkpoint_written','cpu_replay_performed','cpu_fallback','thermal_status_after','battery_health_after','battery_temperature_c_after')
  'generation-aggregates.csv' = @('step','anchor_id','mode','max_new_bytes','temperature','top_k','sampling_seed','generated_bytes','valid_utf8_bytes','invalid_utf8_bytes','unique_byte_values','short_period_loop_fraction','generation_health','legacy_parity_gate','ar_gate','qnn_executes','qnn_failures','cpu_fallback')
  'regression-health.csv' = @('case','status','seed_count','graph_executes','qnn_failures','all_steps_finite','final_evaluation_finite','nonfinite_count','cpu_fallback','backend_build_id_match')
}
$manifestFields = @('milestone','date','seed_count','seed','start_step','final_step','best_validation_step','best_development_step','batch_size','learning_rate','checkpoint_format','checkpoint_interval','qairt_build_id','legacy_parity_thresholds_changed','legacy_generation_gate','htp_native_generation_policy','cpu_fallback','nicopedia_final_test_opened','synthetic_final_holdout_opened','raw_private_data_included')
$safeValue = '^[A-Za-z0-9_.:+-]+$'
$forbidden = '(?i)generated_hex|prompt_sha256|prompt\.bin|device_serial|adb_endpoint|/data/(?:local|user)/|[A-Za-z]:\\Users\\|\.ckpt(?:\b|$)|raw_logits|raw_activations'

$readme = @'
# Nicopedia L19 HTP-native long training, August 2026

This public bundle contains aggregate-only evidence for canonical L19
training from optimizer step 2,000 through 8,000. The numerical operations
of each training step ran in explicit QNN HTP graphs; host control and input
preparation remained on CPU. This is not an NPU-only claim and QNN automatic
differentiation was not used.

The best and final checkpoint is step 8,000. Full-cap HTP-native validation
NLL improved from 2.419448562 at step 1,000 to 2.168420875; development NLL
improved from 2.403955266 to 2.146852226. All full-cap evaluations used
24,576 graph executes, zero failures, zero nonfinite chunks, and no CPU
fallback. The corresponding step-8,000 CPU values were 2.168439351 and
2.146876079.

Training stopped at the predeclared hard ceiling, not because a plateau was
proven. Improvement per 1,000 steps was diminishing but remained positive.
The legacy CPU-equivalence parity thresholds were unchanged and still reject
generation. Experimental HTP-native generation used a separate health gate;
only aggregate byte-quality metrics are included here.

Files:

- `full-cap-evaluation.csv`: HTP-native and CPU held-out metrics.
- `medium-trajectory.csv`: 250-step CPU evaluator screening results.
- `segment-health.csv`: resume segment health and runtime aggregates.
- `generation-aggregates.csv`: no prompts or generated content.
- `regression-health.csv`: synthetic scale-formal and FFN372 finite checks.
- `manifest.json`: scope, policy, and final-test status.

Private dataset text, prompts, generated content, raw bytes/tokens, logits,
checkpoints, device identifiers, endpoints, and local paths are excluded.
Nicopedia final test and the synthetic final holdout were not opened.

The synthetic scale-formal headless regression passed for five seeds with
3,573/3,573 QNN executes. The FFN372/L3/H4 COUNT_FROM_ONE seed-5 regression
also reached `SUCCESS`, 14,088 executes, zero nonzero returns, and finite
training/evaluation. Its legacy host command timed out while the device run
continued, so the terminal report was recovered without relaunching. The
EXACT_SEED comparison half was not rerun in this milestone; no new direct-seed
equivalence claim is made.
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
  $source = Join-Path $repoRoot 'build/export-selftest/nicopedia-long-source'
  $output = Join-Path $repoRoot 'build/export-selftest/nicopedia-long-output'
  [IO.Directory]::CreateDirectory($source) | Out-Null
  $canonical = Join-Path $repoRoot 'docs/results/qnn-nicopedia-htp-long-training-2026-08'
  foreach ($name in @($schemas.Keys) + 'manifest.json') { Copy-Item -LiteralPath (Join-Path $canonical $name) -Destination (Join-Path $source $name) -Force }
  Export-Bundle $source $output
  $malicious = Join-Path $source 'generation-aggregates.csv'
  $original = Get-Content -LiteralPath $malicious
  $original[0] += ',generated_hex'; $original[1] += ',deadbeef'
  [IO.File]::WriteAllLines($malicious,$original,[Text.UTF8Encoding]::new($false))
  $rejected = $false
  try { Export-Bundle $source (Join-Path $repoRoot 'build/export-selftest/nicopedia-long-rejected') } catch { $rejected = $_.Exception.Message -match 'SCHEMA_REJECTED' }
  if (-not $rejected) { throw 'PUBLIC_EXPORTER_NEGATIVE_SELFTEST_FAILED' }
  $extraOutput = Join-Path $repoRoot 'build/export-selftest/nicopedia-long-extra-output'
  [IO.Directory]::CreateDirectory($extraOutput) | Out-Null
  [IO.File]::WriteAllText((Join-Path $extraOutput 'private-generated-text.txt'),'synthetic-private-sentinel',[Text.UTF8Encoding]::new($false))
  [IO.Directory]::CreateDirectory((Join-Path $extraOutput 'private-subdirectory')) | Out-Null
  $extraRejected = $false
  try { Export-Bundle $source $extraOutput } catch { $extraRejected = $_.Exception.Message -eq 'PUBLIC_OUTPUT_EXTRA_FILE_REJECTED' }
  if (-not $extraRejected) { throw 'PUBLIC_EXPORTER_STALE_OUTPUT_SELFTEST_FAILED' }
  Write-Host 'export_public_qnn_nicopedia_htp_long_training_results_self_test=PASS'
  exit 0
}

$source = Resolve-Source $SourceRoot
$output = Resolve-Output $OutputRoot
Export-Bundle $source $output
Write-Host "public_nicopedia_htp_long_training_export=PASS output=$OutputRoot"
