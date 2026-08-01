# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-depth-quality'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-generic-depth-quality-2026-08'),
    [string]$SourceCommit = '',
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @('README.md', 'manifest.json', 'trajectory.csv', 'candidate-comparison.csv')
$steps = @(1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320)

function Fail([string]$Message) {
    throw "depth-quality public export: $Message"
}

function ParseMap([string]$Text, [string]$Name) {
    $map = @{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -notmatch '^([A-Za-z][A-Za-z0-9_]*)=(.*)$') { continue }
        $key = $Matches[1]
        $value = $Matches[2]
        if ($map.ContainsKey($key) -and $map[$key] -cne $value) {
            Fail "conflicting duplicate key $key in $Name"
        }
        $map[$key] = $value
    }
    return $map
}

function ReadMap([string]$RelativePath) {
    $path = Join-Path $InputRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Fail "missing allow-listed input report $RelativePath"
    }
    return ParseMap ([IO.File]::ReadAllText($path)) $RelativePath
}

function Need($Map, [string]$Key) {
    if (-not $Map.ContainsKey($Key)) { Fail "missing report key $Key" }
    return [string]$Map[$Key]
}

function Number($Map, [string]$Key) {
    return [double](Need $Map $Key)
}

function WritePublic([string]$Name, [string]$Text) {
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}

function SafePublicText([string]$Text) {
    return $Text -notmatch '(?i)([a-z]:[\\/]|/data/|/sdcard/|/storage/|device[_ -]?serial|adb[_ -]?endpoint|logcat|(?:qairt|qnn).*\.(?:so|dll)|sk-[a-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}

function CheckBundle {
    $actual = @((Get-ChildItem -LiteralPath $OutputRoot -File).Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) {
        Fail 'public bundle allow-list mismatch'
    }
    foreach ($name in $actual) {
        $text = [IO.File]::ReadAllText((Join-Path $OutputRoot $name))
        if (-not (SafePublicText $text)) { Fail "unsafe public content in $name" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Raw |
        ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or
        $manifest.qairt_build_id -ne '2.48.40.260702151143' -or
        $manifest.selected_stabilized_mode -ne 'NONE') {
        Fail 'manifest consistency mismatch'
    }
    $trajectory = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'trajectory.csv'))
    $comparison = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'candidate-comparison.csv'))
    if ($trajectory.Count -ne 210 -or $comparison.Count -ne 14) {
        Fail 'aggregate row cardinality mismatch'
    }
}

if ($SelfTest) {
    $sample = ParseMap "status=FAILED`ntraining_stability_mode=LEGACY`nvalue=1`n" 'self-test'
    if ((Need $sample 'status') -ne 'FAILED' -or (Need $sample 'value') -ne '1') {
        Fail 'parser self-test failed'
    }
    'SELF_TEST=PASS'
    exit 0
}

if (-not $SourceCommit) {
    $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim()
}
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { Fail 'SourceCommit must be a full Git SHA' }

$specs = @(
    [ordered]@{ id = 'l18s2'; configuration = 'T8/D16/FFN32/L18/H2'; seed = 2; mode = 'LEGACY'; path = 'l18s2/device-report.txt' },
    [ordered]@{ id = 'l19s2'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 2; mode = 'LEGACY'; path = 'l19s2/device-report.txt' },
    [ordered]@{ id = 'l19s1'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 1; mode = 'LEGACY'; path = 'l19s1/device-report.txt' },
    [ordered]@{ id = 'l19s4'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 4; mode = 'LEGACY'; path = 'l19s4/device-report.txt' },
    [ordered]@{ id = 'l18s1'; configuration = 'T8/D16/FFN32/L18/H2'; seed = 1; mode = 'LEGACY'; path = 'l18s1/device-report.txt' },
    [ordered]@{ id = 'l19s2-stab3'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 2; mode = 'ZERO_OUTPUT_PROJ_BRANCH_INIT'; path = 'l19s2-stab3/device-report.txt' },
    [ordered]@{ id = 'l19s1-stab3'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 1; mode = 'ZERO_OUTPUT_PROJ_BRANCH_INIT'; path = 'l19s1-stab3/device-report.txt' },
    [ordered]@{ id = 'l19s4-stab3'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 4; mode = 'ZERO_OUTPUT_PROJ_BRANCH_INIT'; path = 'l19s4-stab3/device-report.txt' },
    [ordered]@{ id = 'l19s2-stab4'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 2; mode = 'DEPTH_SCALED_BRANCH_INIT'; path = 'l19s2-stab4/device-report.txt' },
    [ordered]@{ id = 'l19s1-stab4'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 1; mode = 'DEPTH_SCALED_BRANCH_INIT'; path = 'l19s1-stab4/device-report.txt' },
    [ordered]@{ id = 'l19s4-stab4'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 4; mode = 'DEPTH_SCALED_BRANCH_INIT'; path = 'l19s4-stab4/device-report.txt' },
    [ordered]@{ id = 'l19s2-stab6'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 2; mode = 'GRADIENT_CLIP_1'; path = 'l19s2-stab6/device-report.txt' },
    [ordered]@{ id = 'l19s1-stab6'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 1; mode = 'GRADIENT_CLIP_1'; path = 'l19s1-stab6/device-report.txt' },
    [ordered]@{ id = 'l19s4-stab6'; configuration = 'T8/D16/FFN32/L19/H2'; seed = 4; mode = 'GRADIENT_CLIP_1'; path = 'l19s4-stab6/device-report.txt' }
)

$trajectoryRows = [Collections.Generic.List[object]]::new()
$comparisonRows = [Collections.Generic.List[object]]::new()
$baselineBySeed = @{}
$modes = @{}
foreach ($spec in $specs) {
    $report = ReadMap $spec.path
    if ((Need $report 'training_stability_mode') -ne $spec.mode) {
        Fail "$($spec.id) stability mode mismatch"
    }
    if ((Need $report 'formal_qnn_nonzero_return_count') -ne '0' -or
        (Need $report 'nan_detected') -ne 'false' -or
        (Need $report 'inf_detected') -ne 'false' -or
        (Need $report 'formal_cpu_all_finite') -ne 'true') {
        Fail "$($spec.id) is not finite/zero-QNN-error evidence"
    }
    foreach ($step in $steps) {
        $parts = (Need $report "trajectory_eval_seed_$($spec.seed)_step_$step") -split ','
        if ($parts.Count -ne 4) { Fail "$($spec.id) trajectory schema mismatch at step $step" }
        $trajectoryRows.Add([pscustomobject][ordered]@{
                run_id = $spec.id
                configuration = $spec.configuration
                seed = $spec.seed
                training_stability_mode = $spec.mode
                step = $step
                evaluation_loss = [double]$parts[0]
                evaluation_accuracy = [double]$parts[1]
                mean_logit_margin = [double]$parts[2]
                minimum_logit_margin = [double]$parts[3]
                finite = 'true'
            })
    }
    $final = "trajectory_eval_seed_$($spec.seed)_step_320"
    $finalParts = (Need $report $final) -split ','
    $comparisonRows.Add([pscustomobject][ordered]@{
            run_id = $spec.id
            configuration = $spec.configuration
            seed = $spec.seed
            training_stability_mode = $spec.mode
            status = Need $report 'status'
            qnn_nonzero_return_count = [int](Need $report 'formal_qnn_nonzero_return_count')
            nan_detected = Need $report 'nan_detected'
            inf_detected = Need $report 'inf_detected'
            final_evaluation_loss = [double]$finalParts[0]
            final_evaluation_accuracy = [double]$finalParts[1]
            final_mean_logit_margin = [double]$finalParts[2]
            final_minimum_logit_margin = [double]$finalParts[3]
            qualifying_seed_count = [int](Need $report 'qualifying_seed_count')
            exact_rollout_count = [int](Need $report 'exact_rollout_count')
            oracle_exact_rollout_count = [int](Need $report 'oracle_exact_rollout_count')
            selection_status = if ($spec.mode -eq 'LEGACY') { 'BASELINE' } else { 'NOT_SELECTED' }
        })
    if ($spec.mode -eq 'LEGACY') { $baselineBySeed[$spec.seed] = $comparisonRows[$comparisonRows.Count - 1] }
    if (-not $modes.ContainsKey($spec.mode)) { $modes[$spec.mode] = 0 }
    $modes[$spec.mode]++
}

[IO.Directory]::CreateDirectory($OutputRoot) | Out-Null
$existing = @(Get-ChildItem -LiteralPath $OutputRoot -File -ErrorAction SilentlyContinue | ForEach-Object Name)
$unexpected = @($existing | Where-Object { $_ -notin $allowed })
if ($unexpected.Count -gt 0) { Fail "unexpected stale output: $($unexpected -join ', ')" }
foreach ($name in $allowed) {
    $path = Join-Path $OutputRoot $name
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
}

$trajectoryCsv = (($trajectoryRows | Sort-Object run_id, step | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n"
$comparisonCsv = (($comparisonRows | Sort-Object training_stability_mode, seed, run_id | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n"
WritePublic 'trajectory.csv' $trajectoryCsv
WritePublic 'candidate-comparison.csv' $comparisonCsv

$manifest = [ordered]@{
    schema_version = 1
    source_commit = $SourceCommit
    qairt_build_id = '2.48.40.260702151143'
    configurations = @('T8/D16/FFN32/L18/H2', 'T8/D16/FFN32/L19/H2')
    training_steps = 320
    evaluated_seeds = 'L18:1,2; L19:1,2,4'
    evaluated_stability_modes = @('LEGACY', 'ZERO_OUTPUT_PROJ_BRANCH_INIT', 'DEPTH_SCALED_BRANCH_INIT', 'GRADIENT_CLIP_1')
    selected_stabilized_mode = 'NONE'
    selection_status = 'NO_STABILIZER_SELECTED'
    legacy_default_unchanged = $true
    qnn_nonzero_return_count_all_runs = 0
    all_observed_tensors_finite = $true
    raw_checkpoints_published = $false
    files = $allowed
}
WritePublic 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")

$readme = @"
# QNN HTP generic depth-quality diagnosis

This public bundle contains aggregate-only evidence from 320-step training on
the T8/D16/FFN32 L18/L19 HTP configurations. Each listed run used one explicit
seed, the legacy protocol or an explicitly named experimental stability mode,
and read-only phase-1 evaluation at the listed checkpoints.

The result is `NO_STABILIZER_SELECTED`. `ZERO_OUTPUT_PROJ_BRANCH_INIT` improved
the selected bad L19 seed 2 evaluation loss from 5.4304 to 2.2809 and seed 4
from 3.4454 to 1.9415, but worsened the selected good seed 1 from 0.2629 to
0.6863. `DEPTH_SCALED_BRANCH_INIT` improved seed 2 to 0.9089 but worsened
seed 1 to 9.1662. `GRADIENT_CLIP_1` worsened all three selected L19 seeds.
No candidate improved all selected seeds, so `LEGACY` remains the default and
no stabilized mode is claimed as a fix.

All aggregate runs had zero nonzero QNN return counts and finite reported
tensors. `FAILED` in the comparison table denotes the existing generation
quality threshold result, not a QNN execution or finiteness failure.

Raw weights, optimizer states, private checkpoints, logs, device identifiers,
APK hashes, host paths, and app-private paths are intentionally excluded.
The numerical operation statement is limited to: training-step numerical
operations were executed on HTP; CPU-side scalars and reporting remain
explicitly identified by the app protocol.
"@
WritePublic 'README.md' $readme
CheckBundle
Write-Host "Exported public depth-quality results to $OutputRoot"
