# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-validation-selected'),
    [string]$DirectSeedCsv = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-direct-seed-equivalence\direct-seed-equivalence-private.csv'),
    [string]$LegacyL6Report = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-generic-depth-head\regression-l6-h8-seed1-20260803.txt'),
    [string]$LegacyL6Anchor = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-generic-depth-head\first-nonfinite-baseline-l6-h8.txt'),
    [string]$LegacyL19Report = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-generic-depth-head\regression-l19-h2-ui-exact-seed1-20260803.txt'),
    [string]$LegacyL19Anchor = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-resumable-formal\l19-t8-d16-f32-l19-h2\seeds\seed-001\attempt-1.device-report.txt'),
    [string]$LiveUpdateEvidence = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-generic-depth-head\ui_validation_summary.txt'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-validation-selected-depth-quality-2026-08'),
    [string]$SourceCommit = '',
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @('README.md','manifest.json','direct-seed-equivalence.csv','dataset-partitions.csv',
    'dataset-overlap.csv','validation-trajectories.csv','checkpoint-selection.csv','cpu-smoke.csv','l18-control.csv',
    'legacy-device-regression.csv','live-update-ui.csv',
    'htp-smoke.csv','formal-seeds.csv','generation-oracle.csv','generation-free.csv',
    'selected-step-distribution.csv','early-stop-simulation.csv','decision.csv','ui-validation.csv','thermal.csv')
$expectedOutputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results\qnn-htp-validation-selected-depth-quality-2026-08'))
$expectedDirectCases = @(
    [pscustomobject]@{configuration='t32_d32_ffn32_l2_h2';seed=1},
    [pscustomobject]@{configuration='t32_d32_ffn32_l2_h2';seed=2},
    [pscustomobject]@{configuration='t32_d32_ffn32_l2_h2';seed=5},
    [pscustomobject]@{configuration='t8_d16_ffn32_l19_h2';seed=2},
    [pscustomobject]@{configuration='t16_d256_ffn372_l3_h4';seed=5}
)
$cpuColumns = @('layers','seed','finite','selected_step','best_validation_loss','best_validation_accuracy',
    'final_validation_loss','final_validation_accuracy','selected_oracle_exact','selected_free_exact',
    'final_oracle_exact','final_free_exact','selected_phase1_loss','selected_phase1_accuracy',
    'final_phase1_loss','final_phase1_accuracy')
$trajectoryColumns = @('layers','seed','step','training_loss','training_accuracy','validation_loss',
    'validation_accuracy','validation_target_margin','validation_target_probability','parameter_norm',
    'gradient_norm','update_parameter_ratio','posthoc_oracle_exact','posthoc_free_exact')
$earlyColumns = @('layers','seed','patience','simulated_stop_step','best_step','saved_training_steps')
$directColumns = @('configuration','seed','evidence_schema','legacy_config_hash','exact_config_hash',
    'direct_runner_hash','compared_fields','mismatch_count','bitwise_equivalent','contract_fields_valid',
    'legacy_selection_mode','exact_selection_mode','legacy_requested_seed','exact_requested_seed',
    'legacy_executed_seed_count','exact_executed_seed_count','legacy_wall_time_ms','exact_wall_time_ms',
    'legacy_seed_units','exact_seed_units','legacy_qnn_execute_count','exact_qnn_execute_count',
    'initial_parameter_hash','initial_adam_m_hash','initial_adam_v_hash','all_step_loss_hash',
    'all_step_accuracy_hash','final_parameter_hash','final_logits_hash')
# Public output deliberately excludes the cache/provenance identity hashes
# (config and direct-runner hashes). It exposes every non-private canonical
# comparison hash emitted by the strict direct-seed CSV.
$directPublicColumns = @('configuration','seed','evidence_schema','compared_fields','mismatch_count',
    'bitwise_equivalent','contract_fields_valid','legacy_selection_mode','exact_selection_mode',
    'legacy_requested_seed','exact_requested_seed','legacy_executed_seed_count','exact_executed_seed_count',
    'legacy_seed_units','exact_seed_units','legacy_qnn_execute_count','exact_qnn_execute_count',
    'initial_parameter_hash','initial_adam_m_hash','initial_adam_v_hash','all_step_loss_hash',
    'all_step_accuracy_hash','final_parameter_hash','final_logits_hash')
$l18ControlColumns = @('configuration','seed','selected_step','selected_oracle_exact','final_oracle_exact',
    'selected_free_exact','final_free_exact','oracle_delta','free_delta','classification')
$decisionColumns = @('result_classification','cpu_gate_decision','validation_heldout_generation_improved',
    'l18_control','htp_smoke','htp_formal','best_validation_decision')
$legacyColumns = @('configuration','numeric_regression','quality_oracle','quality_free',
    'terminal_classification','anchor_match')
$liveUpdateColumns = @('progress_update','foreground_service','completion_state','tap_return',
    'auto_cancel','classification')

function Fail([string]$Message) { throw "validation-selection public export: $Message" }
function Safe([string]$Text) {
    # Human-readable exclusion statements are permitted.  Concrete private
    # payload keys, paths, binaries, and log streams are not.
    return $Text -notmatch '(?im)([a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|(?:^|[,\s])(?:raw[_ -]?(?:checkpoint|parameters?)|adam(?:[_ -]?state)?|optimizer[_ -]?state|tensor(?:[_ -]?(?:dump|data))?|logcat)\s*[:=]|\badb\s+-s\s+|sk-[a-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}
function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}
function CsvText($Rows) { return (($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n" }
function Get-Sha256Hex([string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    return ([BitConverter]::ToString([Security.Cryptography.SHA256]::HashData($bytes)) -replace '-', '').ToLowerInvariant()
}
function ReadKeyMap([string]$Path) {
    $map = @{}
    foreach ($line in [IO.File]::ReadLines($Path)) {
        $pair = $line -split '=', 2
        if ($pair.Count -eq 2) { $map[$pair[0]] = $pair[1].TrimEnd("`r") }
    }
    return $map
}
function RequireExactColumns([string]$Name, $Rows, [string[]]$Columns) {
    if ($Rows.Count -eq 0) { Fail "$Name is empty" }
    $actual = @($Rows[0].PSObject.Properties.Name)
    if (($actual -join ',') -ne ($Columns -join ',')) { Fail "$Name schema mismatch" }
}
function RequireInt([string]$Name, [string]$Value, [int]$Minimum = [int]::MinValue) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or $parsed -lt $Minimum) {
        Fail "$Name must be an integer >= $Minimum"
    }
    return $parsed
}
function RequireFinite([string]$Name, [string]$Value) {
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or [double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
        Fail "$Name must be finite"
    }
    return $parsed
}
function RequireCleanHead([string]$Commit) {
    $head = (git -C $repoRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -notmatch '^[0-9a-f]{40}$') { Fail 'Git HEAD is unavailable' }
    if ($Commit -ne $head) { Fail 'SourceCommit must equal current HEAD' }
    $status = @(git -C $repoRoot status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0) { Fail 'SourceCommit requires a clean worktree' }
}
function Get-DirectConfigHash([string]$Configuration, [string]$Mode) {
    $parts = $Configuration -split '_'
    $t=[int]$parts[0].Substring(1); $d=[int]$parts[1].Substring(1); $ffn=[int]$parts[2].Substring(3)
    $l=[int]$parts[3].Substring(1); $h=[int]$parts[4].Substring(1); $headDim=$d/$h
    $canonical = "phonelm-direct-equivalence-v2|B=1|T=$t|V=32|D=$d|FFN=$ffn|L=$l|H=$h" +
        "|headDim=$headDim|lr=0.003|steps=320|clip=disabled|optimizer=ADAM" +
        "|mode=QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC|candidate=3|stability=LEGACY" +
        "|seed_selection=$Mode|checkpoint_selection=FINAL_STEP"
    return Get-Sha256Hex $canonical
}
function AssertConfinedOutputRoot([string]$Path) {
    if ([IO.Path]::GetFullPath($Path) -ne $expectedOutputRoot) { Fail 'OutputRoot must be the designated public results directory' }
    $cursor = $repoRoot
    foreach ($part in @('docs','results','qnn-htp-validation-selected-depth-quality-2026-08')) {
        $cursor = Join-Path $cursor $part
        if (Test-Path -LiteralPath $cursor) {
            if (((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { Fail "OutputRoot reparse point rejected: $cursor" }
        }
    }
}
function ValidateDirect($Rows) {
    RequireExactColumns 'direct-seed equivalence CSV' $Rows $directColumns
    if ($Rows.Count -ne $expectedDirectCases.Count) { Fail 'direct-seed equivalence must contain exactly five cases' }
    $seen = @{}
    $runnerHash = (Get-FileHash -LiteralPath (Join-Path $repoRoot 'scripts\run_qnn_direct_seed_equivalence.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
    foreach ($row in $Rows) {
        $seed = RequireInt 'direct seed' $row.seed 1
        $key = "$($row.configuration)#$seed"
        if ($seen.ContainsKey($key)) { Fail "duplicate direct-seed case $key" }; $seen[$key] = $true
        if ($row.evidence_schema -ne '2' -or
            $row.bitwise_equivalent -notmatch '^(?i:true)$' -or $row.contract_fields_valid -notmatch '^(?i:true)$' -or
            (RequireInt 'direct mismatch_count' $row.mismatch_count 0) -ne 0 -or (RequireInt 'direct compared_fields' $row.compared_fields 1) -le 0) { Fail "direct-seed case failed: $key" }
        if ($row.legacy_selection_mode -ne 'COUNT_FROM_ONE' -or $row.exact_selection_mode -ne 'EXACT_SEED' -or
            (RequireInt 'legacy requested seed' $row.legacy_requested_seed 1) -ne $seed -or
            (RequireInt 'exact requested seed' $row.exact_requested_seed 1) -ne $seed -or
            (RequireInt 'legacy executed seed count' $row.legacy_executed_seed_count 1) -ne $seed -or
            (RequireInt 'exact executed seed count' $row.exact_executed_seed_count 1) -ne 1) { Fail "direct seed contract mismatch: $key" }
        $legacyUnits = RequireInt 'direct legacy_seed_units' $row.legacy_seed_units 1
        $exactUnits = RequireInt 'direct exact_seed_units' $row.exact_seed_units 1
        $legacyExecutes = RequireInt 'direct legacy_qnn_execute_count' $row.legacy_qnn_execute_count 1
        $exactExecutes = RequireInt 'direct exact_qnn_execute_count' $row.exact_qnn_execute_count 1
        if ($legacyUnits -ne $seed -or $exactUnits -ne 1 -or $legacyExecutes -ne $exactExecutes) { Fail "direct seed accounting mismatch: $key" }
        foreach ($field in @('legacy_config_hash','exact_config_hash','direct_runner_hash','initial_parameter_hash',
                'initial_adam_m_hash','initial_adam_v_hash','all_step_loss_hash','all_step_accuracy_hash',
                'final_parameter_hash','final_logits_hash')) {
            if ($row.$field -notmatch '^[0-9a-fA-F]{64}$') { Fail "direct $field malformed: $key" }
        }
        foreach ($mode in @('COUNT_FROM_ONE','EXACT_SEED')) {
            $field = if ($mode -eq 'COUNT_FROM_ONE') { 'legacy_config_hash' } else { 'exact_config_hash' }
            if ($row.$field.ToLowerInvariant() -ne (Get-DirectConfigHash $row.configuration $mode)) { Fail "direct config identity mismatch: $key $mode" }
        }
        if ($row.direct_runner_hash.ToLowerInvariant() -ne $runnerHash) { Fail "direct runner identity mismatch: $key" }
    }
    foreach ($case in $expectedDirectCases) { if (-not $seen.ContainsKey("$($case.configuration)#$($case.seed)")) { Fail "missing required direct-seed case $($case.configuration)#$($case.seed)" } }
    if ((($Rows | Measure-Object -Property compared_fields -Sum).Sum) -ne 3320) { Fail 'direct comparison field total must be 3320' }
}
function ValidateLegacyEvidence([string]$ReportPath, [string]$AnchorPath, [int]$Layers,
        [int]$Heads, [int]$Oracle, [int]$Free, [string]$TerminalClassification) {
    foreach ($path in @($ReportPath,$AnchorPath)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing legacy evidence: $path" } }
    $report = ReadKeyMap $ReportPath; $anchor = ReadKeyMap $AnchorPath
    $keys = @($report.Keys + $anchor.Keys | Where-Object { $_.StartsWith('seed_1_') } | Sort-Object -Unique)
    if ($keys.Count -ne 2248) { Fail "legacy L$Layers comparison field count must be 2248" }
    foreach ($key in $keys) {
        if (-not $report.ContainsKey($key) -or -not $anchor.ContainsKey($key) -or $report[$key] -ne $anchor[$key]) { Fail "legacy L$Layers anchor mismatch: $key" }
    }
    foreach ($entry in ([ordered]@{
            transformer_layers=[string]$Layers;attention_heads=[string]$Heads;steps='320';seed_1_completed_steps='320';
            seed_1_all_steps_finite='true';seed_1_final_evaluation_finite='true';seed_1_nonfinite_count='0';
            formal_qnn_nonzero_return_count='0';oracle_exact_rollout_count=[string]$Oracle}).GetEnumerator()) {
        if ($report[$entry.Key] -ne $entry.Value) { Fail "legacy L$Layers contract mismatch: $($entry.Key)" }
    }
    $freeCount = @($report.Keys | Where-Object { $_ -match '^seed_1_generation_pattern_[0-3]_exact$' -and $report[$_] -eq 'true' }).Count
    if ($freeCount -ne $Free) { Fail "legacy L$Layers free quality mismatch" }
    if ($Layers -eq 19 -and $report.status -ne 'FAILED') { Fail 'L19 terminal status must preserve the finite quality shortfall' }
    if ($Layers -eq 6 -and $report.status -ne 'SUCCESS') { Fail 'L6 terminal status mismatch' }
    return [pscustomobject][ordered]@{
        configuration="T$($report.sequence_length)/D$($report.embedding_dimension)/FFN$($report.feed_forward_dimension)/L$Layers/H$Heads"
        numeric_regression='PASS';quality_oracle="$Oracle/4";quality_free="$Free/4"
        terminal_classification=$TerminalClassification;anchor_match='true'
    }
}
function ValidateLiveUpdateEvidence([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { Fail 'missing live-update evidence' }
    $ui = ReadKeyMap $Path
    foreach ($entry in ([ordered]@{progress_visible='true';foreground_update='true';background_update='true';ongoing_during_run='true';ongoing_after_completion='false';notification_tap_returned_activity='false';auto_cancel='false'}).GetEnumerator()) {
        if ($ui[$entry.Key] -ne $entry.Value) { Fail "live-update evidence mismatch: $($entry.Key)" }
    }
    return [pscustomobject][ordered]@{progress_update='PASS';foreground_service='PASS';completion_state='PASS';tap_return='FAIL_DEFERRED';auto_cancel='FAIL_DEFERRED';classification='PASS_WITH_DEFERRED_NOTIFICATION_UI_ISSUES'}
}
function ValidateCpuEvidence($Cpu, $Trajectory, $Early) {
    RequireExactColumns 'cpu-smoke.csv' $Cpu $cpuColumns; RequireExactColumns 'validation-trajectories.csv' $Trajectory $trajectoryColumns; RequireExactColumns 'early-stop-simulation.csv' $Early $earlyColumns
    if ($Cpu.Count -ne 5 -or $Trajectory.Count -ne 115 -or $Early.Count -ne 15) { Fail 'CPU row cardinality mismatch' }
    $expected = @{'19#1'=$true;'19#2'=$true;'19#4'=$true;'18#1'=$true;'18#2'=$true}; $seen = @{}
    foreach ($row in $Cpu) {
        $layers = RequireInt 'cpu layers' $row.layers 1; $seed = RequireInt 'cpu seed' $row.seed 1; $key="$layers#$seed"
        if (-not $expected.ContainsKey($key) -or $seen.ContainsKey($key) -or $row.finite -notmatch '^(?i:true)$') { Fail "invalid CPU evidence row $key" }; $seen[$key]=$true
        foreach ($field in @($cpuColumns | Where-Object { $_ -notin @('layers','seed','finite') })) { [void](RequireFinite "cpu $field" $row.$field) }
    }
    if ($seen.Count -ne $expected.Count) { Fail 'CPU evidence cases incomplete' }
    foreach ($row in $Trajectory) { foreach ($field in $trajectoryColumns) { [void](RequireFinite "trajectory $field" $row.$field) } }
    foreach ($row in $Early) { foreach ($field in $earlyColumns) { [void](RequireFinite "early-stop $field" $row.$field) } }
    $l19 = @($Cpu | Where-Object { $_.layers -eq '19' -and $_.seed -eq '2' })[0]
    $l18 = @($Cpu | Where-Object { $_.layers -eq '18' -and $_.seed -eq '2' })[0]
    if ((RequireInt 'L19 selected Oracle' $l19.selected_oracle_exact 0) -gt (RequireInt 'L19 final Oracle' $l19.final_oracle_exact 0) -or (RequireInt 'L19 selected Free' $l19.selected_free_exact 0) -gt (RequireInt 'L19 final Free' $l19.final_free_exact 0)) { Fail 'L19 seed 2 unexpectedly improves the CPU gate' }
    if ((RequireInt 'L18 selected Oracle' $l18.selected_oracle_exact 0) -ge (RequireInt 'L18 final Oracle' $l18.final_oracle_exact 0) -or (RequireInt 'L18 selected Free' $l18.selected_free_exact 0) -ge (RequireInt 'L18 final Free' $l18.final_free_exact 0)) { Fail 'L18 seed 2 control must worsen on both held-out measures' }
}
function AssertManifestConsistency([string]$ManifestPath) {
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $entries = @(Get-ChildItem -LiteralPath $OutputRoot -Force)
    if (@($entries | Where-Object { $_.PSIsContainer }).Count -ne 0) { Fail 'public bundle must not contain directories' }
    $actual = @($entries.Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) { Fail 'public bundle allow-list mismatch' }
    $manifestNames = @($manifest.files | ForEach-Object { $_.name } | Sort-Object)
    if (($manifestNames -join ',') -ne (@($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object) -join ',')) { Fail 'manifest file list mismatch' }
    foreach ($entry in $manifest.files) { if ((Get-FileHash -LiteralPath (Join-Path $OutputRoot $entry.name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" } }
    $publicDirect = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'direct-seed-equivalence.csv')); $publicCpu = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'cpu-smoke.csv'))
    RequireExactColumns 'public direct-seed-equivalence.csv' $publicDirect $directPublicColumns
    if ($publicDirect.Count -ne 5 -or @($publicDirect | Where-Object {
            $_.evidence_schema -ne '2' -or $_.bitwise_equivalent -notmatch '^(?i:true)$' -or
            $_.contract_fields_valid -notmatch '^(?i:true)$' -or
            (RequireInt 'public direct mismatch_count' $_.mismatch_count 0) -ne 0 -or
            (RequireInt 'public direct compared_fields' $_.compared_fields 1) -le 0
        }).Count -ne 0) { Fail 'public direct 5/5 mismatch' }
    $l18Control = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'l18-control.csv'))
    $legacy = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'legacy-device-regression.csv'))
    $liveUpdate = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'live-update-ui.csv'))
    $decision = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'decision.csv'))
    RequireExactColumns 'l18-control.csv' $l18Control $l18ControlColumns
    RequireExactColumns 'legacy-device-regression.csv' $legacy $legacyColumns
    RequireExactColumns 'live-update-ui.csv' $liveUpdate $liveUpdateColumns
    RequireExactColumns 'decision.csv' $decision $decisionColumns
    if ($l18Control.Count -ne 1 -or $l18Control[0].classification -ne 'DEGRADED' -or
        (RequireInt 'L18 public oracle delta' $l18Control[0].oracle_delta) -ge 0 -or
        (RequireInt 'L18 public free delta' $l18Control[0].free_delta) -ge 0) { Fail 'public L18 control mismatch' }
    if ($decision.Count -ne 1 -or
        $decision[0].result_classification -ne 'DIRECT_SEED_COMPLETE_VALIDATION_NOT_PREDICTIVE' -or
        $decision[0].cpu_gate_decision -ne 'REJECT' -or
        $decision[0].validation_heldout_generation_improved -ne 'false' -or
        $decision[0].l18_control -ne 'DEGRADED' -or
        $decision[0].htp_smoke -ne 'NOT_RUN_CPU_GATE_REJECTED' -or
        $decision[0].htp_formal -ne 'NOT_RUN_CPU_GATE_REJECTED' -or
        $decision[0].best_validation_decision -ne 'REJECTED') { Fail 'public decision mismatch' }
    if ($legacy.Count -ne 2 -or
        $legacy[0].numeric_regression -ne 'PASS' -or $legacy[0].quality_oracle -ne '4/4' -or $legacy[0].quality_free -ne '4/4' -or $legacy[0].terminal_classification -ne 'SUCCESS' -or $legacy[0].anchor_match -ne 'true' -or
        $legacy[1].numeric_regression -ne 'PASS' -or $legacy[1].quality_oracle -ne '2/4' -or $legacy[1].quality_free -ne '2/4' -or $legacy[1].terminal_classification -ne 'FINITE_QUALITY_SHORTFALL' -or $legacy[1].anchor_match -ne 'true') { Fail 'public legacy regression/quality classification mismatch' }
    if ($liveUpdate.Count -ne 1 -or $liveUpdate[0].progress_update -ne 'PASS' -or
        $liveUpdate[0].foreground_service -ne 'PASS' -or $liveUpdate[0].completion_state -ne 'PASS' -or
        $liveUpdate[0].tap_return -ne 'FAIL_DEFERRED' -or $liveUpdate[0].auto_cancel -ne 'FAIL_DEFERRED') { Fail 'public live-update classification mismatch' }
    $l19 = @($publicCpu | Where-Object { $_.layers -eq '19' -and $_.seed -eq '2' })[0]
    $l18 = @($publicCpu | Where-Object { $_.layers -eq '18' -and $_.seed -eq '2' })[0]
    if ($null -eq $l19 -or $null -eq $l18) { Fail 'manifest CPU aggregate source rows missing' }
    $aggregateMatches = $manifest.aggregate.direct_case_count -eq $publicDirect.Count -and
        $manifest.aggregate.direct_compared_fields_total -eq (($publicDirect | Measure-Object -Property compared_fields -Sum).Sum) -and
        $manifest.aggregate.direct_mismatch_count_total -eq 0 -and
        $manifest.aggregate.cpu_case_count -eq $publicCpu.Count -and
        $manifest.aggregate.cpu_l19_seed2_oracle_delta -eq ([int]$l19.selected_oracle_exact - [int]$l19.final_oracle_exact) -and
        $manifest.aggregate.cpu_l19_seed2_free_delta -eq ([int]$l19.selected_free_exact - [int]$l19.final_free_exact) -and
        $manifest.aggregate.cpu_l18_seed2_oracle_delta -eq ([int]$l18.selected_oracle_exact - [int]$l18.final_oracle_exact) -and
        $manifest.aggregate.cpu_l18_seed2_free_delta -eq ([int]$l18.selected_free_exact - [int]$l18.final_free_exact) -and
        $manifest.aggregate.l18_control_classification -eq $l18Control[0].classification -and
        $manifest.aggregate.cpu_gate_decision -eq $decision[0].cpu_gate_decision -and
        $manifest.aggregate.legacy_l6_h8_regression -eq 'PASS' -and
        $manifest.aggregate.legacy_l19_regression -eq 'PASS' -and
        $manifest.aggregate.legacy_l19_quality_oracle -eq '2/4' -and
        $manifest.aggregate.legacy_l19_quality_free -eq '2/4' -and
        $manifest.aggregate.live_update_progress -eq 'PASS' -and
        $manifest.aggregate.live_update_tap_return -eq 'FAIL_DEFERRED' -and
        $manifest.aggregate.live_update_auto_cancel -eq 'FAIL_DEFERRED'
    if (-not $aggregateMatches) { Fail 'manifest aggregate mismatch' }
    if ($manifest.result_classification -ne 'DIRECT_SEED_COMPLETE_VALIDATION_NOT_PREDICTIVE' -or
        $manifest.direct_seed_equivalence -ne '5/5' -or $manifest.cpu_candidate_accepted -ne $false -or
        $manifest.cpu_gate_decision -ne 'REJECT' -or $manifest.validation_heldout_generation_improved -ne $false -or
        $manifest.l18_control -ne 'DEGRADED' -or
        $manifest.legacy_l6_h8_regression -ne 'PASS' -or $manifest.legacy_l19_regression -ne 'PASS' -or
        $manifest.legacy_l19_terminal_classification -ne 'FINITE_QUALITY_SHORTFALL' -or
        $manifest.live_update_progress -ne 'PASS' -or $manifest.live_update_foreground_service -ne 'PASS' -or
        $manifest.live_update_completion_state -ne 'PASS' -or $manifest.live_update_tap_return -ne 'FAIL_DEFERRED' -or
        $manifest.live_update_auto_cancel -ne 'FAIL_DEFERRED' -or $manifest.deferred_ui_issue -ne 'NOTIFICATION_TAP_AND_AUTO_CANCEL' -or
        $manifest.htp_smoke -ne 'NOT_RUN_CPU_GATE_REJECTED' -or $manifest.formal -ne 'NOT_RUN_CPU_GATE_REJECTED') { Fail 'manifest classification or NOT_RUN fields mismatch' }
}

if ($SelfTest) {
    if (-not (Safe 'validation_set_hash=fnv1a64:8e1411f19126879c')) { Fail 'safe-text false rejection' }
    if (Safe 'apk_sha256=aaaaaaaa') { Fail 'APK hash negative rejection failed' }
    if (Safe 'raw_checkpoint=C:\\private\\checkpoint.bin') { Fail 'checkpoint negative rejection failed' }
    if (Safe 'adb -s physical-device shell logcat') { Fail 'ADB/logcat negative rejection failed' }
    if (Safe 'app_private_path=relative/private/report') { Fail 'app-private path negative rejection failed' }
    if (Safe '/home/user/private-report') { Fail 'Unix absolute path negative rejection failed' }
    if (Safe '/tmp/private-report') { Fail 'Unix temporary path negative rejection failed' }
    if (Safe '/root/private-report') { Fail 'generic Unix root path negative rejection failed' }
    if (Safe '/opt/private-report') { Fail 'generic Unix opt path negative rejection failed' }
    if (Safe '\\private-host\share\report') { Fail 'UNC path negative rejection failed' }
    if (Safe 'files/private-report') { Fail 'relative app-private files path negative rejection failed' }
    try { AssertConfinedOutputRoot ([IO.Path]::GetTempPath()); Fail 'OutputRoot confinement negative rejection failed' } catch { if ($_.Exception.Message -match 'OutputRoot confinement negative rejection failed') { throw } }
    $directFixture = foreach ($case in $expectedDirectCases) {
        $fieldCount = if ($case.configuration -eq 't32_d32_ffn32_l2_h2' -and $case.seed -eq 1) { 2248 } else { 268 }
        [pscustomobject][ordered]@{configuration=$case.configuration;seed=$case.seed;evidence_schema='2';legacy_config_hash=(Get-DirectConfigHash $case.configuration 'COUNT_FROM_ONE');exact_config_hash=(Get-DirectConfigHash $case.configuration 'EXACT_SEED');direct_runner_hash=(Get-FileHash -LiteralPath (Join-Path $repoRoot 'scripts\run_qnn_direct_seed_equivalence.ps1') -Algorithm SHA256).Hash.ToLowerInvariant();compared_fields=$fieldCount;mismatch_count=0;bitwise_equivalent='true';contract_fields_valid='true';legacy_selection_mode='COUNT_FROM_ONE';exact_selection_mode='EXACT_SEED';legacy_requested_seed=$case.seed;exact_requested_seed=$case.seed;legacy_executed_seed_count=$case.seed;exact_executed_seed_count=1;legacy_wall_time_ms='1';exact_wall_time_ms='1';legacy_seed_units=$case.seed;exact_seed_units=1;legacy_qnn_execute_count=1;exact_qnn_execute_count=1;initial_parameter_hash=('0' * 64);initial_adam_m_hash=('1' * 64);initial_adam_v_hash=('2' * 64);all_step_loss_hash=('3' * 64);all_step_accuracy_hash=('4' * 64);final_parameter_hash=('5' * 64);final_logits_hash=('f' * 64)}
    }
    ValidateDirect $directFixture
    $badDirect = @($directFixture | Select-Object $directColumns); $badDirect[4].seed = 4
    try { ValidateDirect $badDirect; Fail 'missing FFN372 seed 5 negative rejection failed' } catch { if ($_.Exception.Message -match 'missing FFN372 seed 5 negative rejection failed') { throw } }
    $badHash = @($directFixture | Select-Object $directColumns); $badHash[0].final_parameter_hash = '0' * 63
    try { ValidateDirect $badHash; Fail '64-hex direct hash negative rejection failed' } catch { if ($_.Exception.Message -match '64-hex direct hash negative rejection failed') { throw } }
    $badUnits = @($directFixture | Select-Object $directColumns); $badUnits[2].legacy_seed_units = 1
    try { ValidateDirect $badUnits; Fail 'legacy seed-unit negative rejection failed' } catch { if ($_.Exception.Message -match 'legacy seed-unit negative rejection failed') { throw } }
    $badExactUnits = @($directFixture | Select-Object $directColumns); $badExactUnits[0].exact_seed_units = 2
    try { ValidateDirect $badExactUnits; Fail 'exact seed-unit negative rejection failed' } catch { if ($_.Exception.Message -match 'exact seed-unit negative rejection failed') { throw } }
    $badExecutes = @($directFixture | Select-Object $directColumns); $badExecutes[0].exact_qnn_execute_count = 2
    try { ValidateDirect $badExecutes; Fail 'execute-count negative rejection failed' } catch { if ($_.Exception.Message -match 'execute-count negative rejection failed') { throw } }
    $badContract = @($directFixture | Select-Object $directColumns); $badContract[1].legacy_executed_seed_count = 1
    try { ValidateDirect $badContract; Fail 'executed-seed contract negative rejection failed' } catch { if ($_.Exception.Message -match 'executed-seed contract negative rejection failed') { throw } }
    $badEvidenceSchema = @($directFixture | Select-Object $directColumns); $badEvidenceSchema[0].evidence_schema = '1'
    try { ValidateDirect $badEvidenceSchema; Fail 'evidence-schema negative rejection failed' } catch { if ($_.Exception.Message -match 'evidence-schema negative rejection failed') { throw } }
    $cpuFixture = @(
        [pscustomobject][ordered]@{layers='19';seed='1';finite='true';selected_step='320';best_validation_loss='1';best_validation_accuracy='1';final_validation_loss='1';final_validation_accuracy='1';selected_oracle_exact='2';selected_free_exact='2';final_oracle_exact='2';final_free_exact='2';selected_phase1_loss='1';selected_phase1_accuracy='1';final_phase1_loss='1';final_phase1_accuracy='1'},
        [pscustomobject][ordered]@{layers='19';seed='2';finite='true';selected_step='128';best_validation_loss='1';best_validation_accuracy='1';final_validation_loss='1';final_validation_accuracy='1';selected_oracle_exact='2';selected_free_exact='2';final_oracle_exact='2';final_free_exact='2';selected_phase1_loss='1';selected_phase1_accuracy='1';final_phase1_loss='1';final_phase1_accuracy='1'},
        [pscustomobject][ordered]@{layers='19';seed='4';finite='true';selected_step='320';best_validation_loss='1';best_validation_accuracy='1';final_validation_loss='1';final_validation_accuracy='1';selected_oracle_exact='2';selected_free_exact='2';final_oracle_exact='2';final_free_exact='2';selected_phase1_loss='1';selected_phase1_accuracy='1';final_phase1_loss='1';final_phase1_accuracy='1'},
        [pscustomobject][ordered]@{layers='18';seed='1';finite='true';selected_step='320';best_validation_loss='1';best_validation_accuracy='1';final_validation_loss='1';final_validation_accuracy='1';selected_oracle_exact='2';selected_free_exact='2';final_oracle_exact='2';final_free_exact='2';selected_phase1_loss='1';selected_phase1_accuracy='1';final_phase1_loss='1';final_phase1_accuracy='1'},
        [pscustomobject][ordered]@{layers='18';seed='2';finite='true';selected_step='128';best_validation_loss='1';best_validation_accuracy='1';final_validation_loss='1';final_validation_accuracy='1';selected_oracle_exact='2';selected_free_exact='2';final_oracle_exact='3';final_free_exact='3';selected_phase1_loss='1';selected_phase1_accuracy='1';final_phase1_loss='1';final_phase1_accuracy='1'}
    )
    $trajectoryFixture = for ($i=0; $i -lt 115; $i++) { [pscustomobject][ordered]@{layers='19';seed='1';step='1';training_loss='1';training_accuracy='1';validation_loss='1';validation_accuracy='1';validation_target_margin='1';validation_target_probability='1';parameter_norm='1';gradient_norm='1';update_parameter_ratio='1';posthoc_oracle_exact='1';posthoc_free_exact='1'} }
    $earlyFixture = for ($i=0; $i -lt 15; $i++) { [pscustomobject][ordered]@{layers='19';seed='1';patience='2';simulated_stop_step='1';best_step='1';saved_training_steps='1'} }
    ValidateCpuEvidence $cpuFixture $trajectoryFixture $earlyFixture
    $badCpu = @($cpuFixture | Select-Object $cpuColumns); $badCpu[4].selected_free_exact = '3'
    try { ValidateCpuEvidence $badCpu $trajectoryFixture $earlyFixture; Fail 'L18 control negative rejection failed' } catch { if ($_.Exception.Message -match 'L18 control negative rejection failed') { throw } }
    $badL19 = @($cpuFixture | Select-Object $cpuColumns); $badL19[1].selected_oracle_exact = '3'
    try { ValidateCpuEvidence $badL19 $trajectoryFixture $earlyFixture; Fail 'L19 non-improvement negative rejection failed' } catch { if ($_.Exception.Message -match 'L19 non-improvement negative rejection failed') { throw } }
    $nonFinite = @($cpuFixture | Select-Object $cpuColumns); $nonFinite[0].best_validation_loss = 'NaN'
    try { ValidateCpuEvidence $nonFinite $trajectoryFixture $earlyFixture; Fail 'non-finite CPU negative rejection failed' } catch { if ($_.Exception.Message -match 'non-finite CPU negative rejection failed') { throw } }
    'SELF_TEST=PASS'
    exit 0
}
if (-not $SourceCommit) { $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim() }
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { Fail 'SourceCommit must be a full Git SHA' }
RequireCleanHead $SourceCommit
AssertConfinedOutputRoot $OutputRoot
foreach ($name in @('cpu-smoke.csv','validation-trajectories.csv','early-stop-simulation.csv')) {
    if (-not (Test-Path -LiteralPath (Join-Path $InputRoot $name) -PathType Leaf)) { Fail "missing $name" }
}
if (-not (Test-Path -LiteralPath $DirectSeedCsv -PathType Leaf)) { Fail 'missing direct-seed equivalence CSV' }
$cpu = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'cpu-smoke.csv'))
$trajectory = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'validation-trajectories.csv'))
$early = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'early-stop-simulation.csv'))
$direct = @(Import-Csv -LiteralPath $DirectSeedCsv)
ValidateDirect $direct
ValidateCpuEvidence $cpu $trajectory $early
$legacy = @(
    ValidateLegacyEvidence $LegacyL6Report $LegacyL6Anchor 6 8 4 4 'SUCCESS'
    ValidateLegacyEvidence $LegacyL19Report $LegacyL19Anchor 19 2 2 2 'FINITE_QUALITY_SHORTFALL'
)
$liveUpdate = @(ValidateLiveUpdateEvidence $LiveUpdateEvidence)
$l19 = @($cpu | Where-Object { $_.layers -eq '19' -and $_.seed -eq '2' })[0]
$l18 = @($cpu | Where-Object { $_.layers -eq '18' -and $_.seed -eq '2' })[0]

[IO.Directory]::CreateDirectory($OutputRoot) | Out-Null
foreach ($file in Get-ChildItem -LiteralPath $OutputRoot -Force -ErrorAction SilentlyContinue) {
    if ($file.PSIsContainer) { Fail "unexpected output directory $($file.Name)" }
    if ($file.Name -notin $allowed) { Fail "unexpected stale output $($file.Name)" }
    Remove-Item -LiteralPath $file.FullName
}

$directPublic = $direct | Select-Object $directPublicColumns
WriteUtf8 'direct-seed-equivalence.csv' (CsvText $directPublic)

$partitions = @(
    [pscustomobject][ordered]@{set='TRAIN';case_count=4;generator='SINGLE_RULE_PHASE0';input_prefix='phase-0 cyclic prefix';target='per-row rule successor';checkpoint_selection='never'},
    [pscustomobject][ordered]@{set='CURRENT_PHASE1_EVAL';case_count=4;generator='SINGLE_RULE_PHASE1';input_prefix='phase-1 cyclic prefix';target='per-row rule successor';checkpoint_selection='never'},
    [pscustomobject][ordered]@{set='VALIDATION';case_count=3;generator='ROTATED_LAST_POSITION_V2';input_prefix='phase-2 cyclic prefix; 2-token family excluded';target='last-position rule successor';checkpoint_selection='BEST_VALIDATION_V1 only'},
    [pscustomobject][ordered]@{set='ORACLE_TEST';case_count=4;generator='ORACLE_8_STEP_PHASE0';input_prefix='phase-0 cyclic prefix';target='8 held-out successor tokens; expected-token feedback';checkpoint_selection='never'},
    [pscustomobject][ordered]@{set='FREE_TEST';case_count=4;generator='FREE_8_STEP_PHASE0';input_prefix='phase-0 cyclic prefix';target='8 held-out successor tokens; prediction feedback';checkpoint_selection='never'}
)
WriteUtf8 'dataset-partitions.csv' (CsvText $partitions)
$overlap = @(
    [pscustomobject][ordered]@{left='TRAIN';right='VALIDATION';exact_case_overlap=0;identical_initial_prefixes=0;identical_target_sequences=0;shared_token_ids=11;shared_transitions=11;scope='full cases'},
    [pscustomobject][ordered]@{left='VALIDATION';right='ORACLE_TEST';exact_case_overlap=0;identical_initial_prefixes=0;identical_target_sequences=0;shared_token_ids=11;shared_transitions=11;scope='static full cases'},
    [pscustomobject][ordered]@{left='VALIDATION';right='FREE_TEST';exact_case_overlap=0;identical_initial_prefixes=0;identical_target_sequences=0;shared_token_ids=11;shared_transitions=11;scope='static initial cases; free contexts prediction-dependent'},
    [pscustomobject][ordered]@{left='TRAIN';right='CURRENT_PHASE1_EVAL';exact_case_overlap=0;identical_initial_prefixes=0;identical_target_sequences=0;shared_token_ids=13;shared_transitions=13;scope='full cases'},
    [pscustomobject][ordered]@{left='TRAIN';right='ORACLE_TEST';exact_case_overlap=0;identical_initial_prefixes=4;identical_target_sequences=0;shared_token_ids=13;shared_transitions=13;scope='initial prefix versus rollout'},
    [pscustomobject][ordered]@{left='TRAIN';right='FREE_TEST';exact_case_overlap=0;identical_initial_prefixes=4;identical_target_sequences=0;shared_token_ids=13;shared_transitions=13;scope='initial prefix versus rollout'}
)
WriteUtf8 'dataset-overlap.csv' (CsvText $overlap)
WriteUtf8 'validation-trajectories.csv' (CsvText $trajectory)
WriteUtf8 'early-stop-simulation.csv' (CsvText $early)
WriteUtf8 'cpu-smoke.csv' (CsvText $cpu)

$selection = foreach ($row in $cpu) {
    [pscustomobject][ordered]@{configuration="T8/D16/FFN32/L$($row.layers)/H2";seed=[int]$row.seed;mode='BEST_VALIDATION_V1';selected_step=[int]$row.selected_step;best_validation_loss=[double]$row.best_validation_loss;final_validation_loss=[double]$row.final_validation_loss;selected_oracle_exact=[int]$row.selected_oracle_exact;final_oracle_exact=[int]$row.final_oracle_exact;selected_free_exact=[int]$row.selected_free_exact;final_free_exact=[int]$row.final_free_exact;candidate_accepted='false'}
}
WriteUtf8 'checkpoint-selection.csv' (CsvText $selection)
$l18Control = @([pscustomobject][ordered]@{
    configuration = 'T8/D16/FFN32/L18/H2'
    seed = [int]$l18.seed
    selected_step = [int]$l18.selected_step
    selected_oracle_exact = [int]$l18.selected_oracle_exact
    final_oracle_exact = [int]$l18.final_oracle_exact
    selected_free_exact = [int]$l18.selected_free_exact
    final_free_exact = [int]$l18.final_free_exact
    oracle_delta = ([int]$l18.selected_oracle_exact - [int]$l18.final_oracle_exact)
    free_delta = ([int]$l18.selected_free_exact - [int]$l18.final_free_exact)
    classification = 'DEGRADED'
})
WriteUtf8 'l18-control.csv' (CsvText $l18Control)
WriteUtf8 'legacy-device-regression.csv' (CsvText $legacy)
WriteUtf8 'live-update-ui.csv' (CsvText $liveUpdate)
$decision = @([pscustomobject][ordered]@{
    result_classification = 'DIRECT_SEED_COMPLETE_VALIDATION_NOT_PREDICTIVE'
    cpu_gate_decision = 'REJECT'
    validation_heldout_generation_improved = 'false'
    l18_control = 'DEGRADED'
    htp_smoke = 'NOT_RUN_CPU_GATE_REJECTED'
    htp_formal = 'NOT_RUN_CPU_GATE_REJECTED'
    best_validation_decision = 'REJECTED'
})
WriteUtf8 'decision.csv' (CsvText $decision)
$oracle = foreach ($row in $cpu) { [pscustomobject][ordered]@{layers=[int]$row.layers;seed=[int]$row.seed;selected_step=[int]$row.selected_step;selected_exact=[int]$row.selected_oracle_exact;final_step_exact=[int]$row.final_oracle_exact;case_count=4} }
$free = foreach ($row in $cpu) { [pscustomobject][ordered]@{layers=[int]$row.layers;seed=[int]$row.seed;selected_step=[int]$row.selected_step;selected_exact=[int]$row.selected_free_exact;final_step_exact=[int]$row.final_free_exact;case_count=4} }
WriteUtf8 'generation-oracle.csv' (CsvText $oracle)
WriteUtf8 'generation-free.csv' (CsvText $free)
$distribution = $cpu | Group-Object layers,selected_step | ForEach-Object {
    $first=$_.Group[0]; [pscustomobject][ordered]@{layers=[int]$first.layers;selected_step=[int]$first.selected_step;seed_count=$_.Count}
}
WriteUtf8 'selected-step-distribution.csv' (CsvText $distribution)
$notRun = [pscustomobject][ordered]@{status='NOT_RUN_CPU_GATE_REJECTED';reason='NOT_RUN_CPU_GATE_REJECTED'}
WriteUtf8 'htp-smoke.csv' (CsvText @($notRun))
WriteUtf8 'formal-seeds.csv' (CsvText @($notRun))
WriteUtf8 'ui-validation.csv' (CsvText @([pscustomobject][ordered]@{status='NOT_RUN_CPU_GATE_REJECTED';reason='NOT_RUN_CPU_GATE_REJECTED'}))
WriteUtf8 'thermal.csv' (CsvText @([pscustomobject][ordered]@{status='NOT_RUN_CPU_GATE_REJECTED';reason='NOT_RUN_CPU_GATE_REJECTED'}))

$readme = @"
# Validation-selected depth quality

`BEST_VALIDATION_V1` trains all 320 steps and ranks checkpoints only by the
independent `ROTATED_LAST_POSITION_V2` validation loss (loss, then accuracy,
then earlier step). `FINAL_STEP` remains the default and does not evaluate or
restore validation checkpoints.

The validation cases have no full-case or initial-prefix overlap with TRAIN,
Oracle, or the static initial Free cases. They intentionally share 11 learned
token transitions; the two-token rule is excluded because it has no third
distinct phase. FNV-1a identifiers are corruption/determinism checks, not
cryptographic authenticity claims.

CPU screening rejected the candidate. L19 seeds 1 and 4 selected step 320;
seed 2 selected step 128, but held-out Oracle/Free stayed 2/4 rather than
improving over the final step. L18 seed 2 worsened from 3/4 to 2/4. Therefore
HTP smoke and five-seed formal were `NOT_RUN_CPU_GATE_REJECTED` under the predeclared gate.
The observed validation regression was not reliably predictive of improved
disjoint held-out generation tests.

The direct-seed table exposes canonical initial parameter/Adam, all-step loss
and accuracy, final parameter, and final-logit hashes. Configuration, runner,
APK, path, and device identity remain private provenance; the exact strict
comparison covers all harvested report fields, including generation fields.

Legacy device regression is independently classified from generation quality.
The L6/H8 and L19 seed-1 reports each match all 2248 canonical anchor fields.
L6/H8 retains Oracle/Free 4/4. L19 retains its canonical Oracle/Free 2/4;
its terminal `FAILED` is published as `FINITE_QUALITY_SHORTFALL`, not as a
numeric or device regression.

Live progress updates, foreground-service continuity, and the ongoing true-to-
false completion transition passed. Notification tap return and auto-cancel
failed and remain explicitly deferred Android UI issues. They do not block the
direct-seed equivalence result or the CPU-gated validation decision.

Raw checkpoints, parameters, optimizer state, tensor dumps, APK data, device
identifiers, paths, and logs are excluded.
"@
WriteUtf8 'README.md' ($readme + "`n")

$fileEntries = @()
foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    $fileEntries += [ordered]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $OutputRoot $name) -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$aggregate = [ordered]@{direct_case_count=$directPublic.Count;direct_compared_fields_total=(($directPublic | Measure-Object -Property compared_fields -Sum).Sum);direct_mismatch_count_total=(($directPublic | Measure-Object -Property mismatch_count -Sum).Sum);cpu_case_count=$cpu.Count;cpu_l19_seed2_oracle_delta=([int]$l19.selected_oracle_exact - [int]$l19.final_oracle_exact);cpu_l19_seed2_free_delta=([int]$l19.selected_free_exact - [int]$l19.final_free_exact);cpu_l18_seed2_oracle_delta=([int]$l18.selected_oracle_exact - [int]$l18.final_oracle_exact);cpu_l18_seed2_free_delta=([int]$l18.selected_free_exact - [int]$l18.final_free_exact);l18_control_classification=$l18Control[0].classification;cpu_gate_decision=$decision[0].cpu_gate_decision;legacy_l6_h8_regression=$legacy[0].numeric_regression;legacy_l19_regression=$legacy[1].numeric_regression;legacy_l19_quality_oracle=$legacy[1].quality_oracle;legacy_l19_quality_free=$legacy[1].quality_free;live_update_progress=$liveUpdate[0].progress_update;live_update_tap_return=$liveUpdate[0].tap_return;live_update_auto_cancel=$liveUpdate[0].auto_cancel}
$manifest = [ordered]@{schema_version=4;source_commit=$SourceCommit;qairt_build_id='2.48.40.260702151143';result_classification='DIRECT_SEED_COMPLETE_VALIDATION_NOT_PREDICTIVE';direct_seed_equivalence='5/5';validation_schema_version=2;validation_generator_domain='ROTATED_LAST_POSITION_V2';validation_set_hash='fnv1a64:8e1411f19126879c';checkpoint_selection_mode='BEST_VALIDATION_V1';default_checkpoint_selection_mode='FINAL_STEP';cpu_candidate_accepted=$false;cpu_gate_decision='REJECT';validation_heldout_generation_improved=$false;l18_control='DEGRADED';legacy_l6_h8_regression='PASS';legacy_l19_regression='PASS';legacy_l19_terminal_classification='FINITE_QUALITY_SHORTFALL';live_update_progress='PASS';live_update_foreground_service='PASS';live_update_completion_state='PASS';live_update_tap_return='FAIL_DEFERRED';live_update_auto_cancel='FAIL_DEFERRED';deferred_ui_issue='NOTIFICATION_TAP_AND_AUTO_CANCEL';htp_smoke='NOT_RUN_CPU_GATE_REJECTED';formal='NOT_RUN_CPU_GATE_REJECTED';raw_checkpoints_published=$false;aggregate=$aggregate;files=$fileEntries}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 6) + "`n")

foreach ($name in @((Get-ChildItem -LiteralPath $OutputRoot -File -Force).Name)) { if (-not (Safe ([IO.File]::ReadAllText((Join-Path $OutputRoot $name))))) { Fail "unsafe final file $name" } }
AssertManifestConsistency (Join-Path $OutputRoot 'manifest.json')
Write-Host "validation-selection public export PASS: $OutputRoot"
