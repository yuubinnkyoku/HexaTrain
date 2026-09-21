# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 probe-optimization audit
# (host-only CPU evidence). Copies only the allow-listed files from the
# private report root, schema-checks each, verifies dataset hashes and budget,
# and scans for private identifiers.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-probe-optimization-audit'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-probe-optimization-audit-2026-08'),
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'configuration.csv', 'dataset-usage.csv',
    'legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
    'corrected-attention-taps.csv', 'feature-geometry.csv',
    'row-nullspace.csv', 'calibration-selection.csv',
    'optimization-summary.csv', 'diagnosis.csv',
    'previous-result-corrections.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceFiles = @(
    'configuration.csv', 'dataset-usage.csv',
    'legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
    'corrected-attention-taps.csv', 'feature-geometry.csv',
    'row-nullspace.csv', 'calibration-selection.csv',
    'optimization-summary.csv', 'diagnosis.csv',
    'previous-result-corrections.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceSchemas = [ordered]@{
    'configuration.csv' = 'configuration_id,seed,layers,final_step,max_drop_block,ar_selected_step,non_drop_first,non_drop_last,legacy_ctx_dev_exact_published,legacy_att_dev_exact_published'
    'dataset-usage.csv' = 'dataset,role,hash,rows'
    'legacy-vs-canonical-probe.csv' = 'configuration_id,layer,tap,legacy_dev_exact,canonical_gd_dev_exact,canonical_lbfgs_dev_exact,canonical_minus_legacy_dev_exact,canonical_lbfgs_train_ce'
    'corrected-layer-curve.csv' = 'configuration_id,rep_index,rep_name,legacy_dev_exact,canonical_gd_dev_exact,canonical_lbfgs_dev_exact,canonical_lbfgs_train_ce,spearman_rho,legacy_gap,canonical_gap'
    'corrected-attention-taps.csv' = 'configuration_id,block,tap,canonical_dev_exact,canonical_drop'
    'feature-geometry.csv' = 'configuration_id,layer,tap,z_condition,z_lambda_max,z_lambda_min,z_null_count,z_near_null_count,whitened_kept,whitened_max_cov_dev,whitened_max_mean_abs,orth_max_dev,orth_max_residual'
    'row-nullspace.csv' = 'configuration_id,layer,z_design_rank,z_design_nullity,z_design_condition,delta_fro,delta_null_fraction,delta_near_null_fraction,train_ce_diff,dev_exact_diff,max_dlogit_null_dev,flips_null_dev,flips_total_dev'
    'calibration-selection.csv' = 'configuration_id,layer,tap,selected_step,selected_fraction,train_ce_selected,train_ce_2000,cal_ce,cal_exact,dev_exact'
    'optimization-summary.csv' = 'configuration_id,layer,tap,condition,solver,init,lambda,converged,converged_flat,stalled,iterations,grad_norm,objective,train_ce,cal_ce,dev_ce,train_exact,cal_exact,dev_exact,selected_step,train_ce_2000,ce_coordinate'
    'diagnosis.csv' = 'verdict,c1_layers,c2_layers,c3_layers,c4_layers,c5_layers,c1_ok,c2_ok,c3_ok,c4_ok,c5_ok,curve_maintained,curve_shrunk,curve_gone,projection_artifact,attention_remains,criteria_fixed_before_results'
    'previous-result-corrections.csv' = 'previous_claim,previous_evidence,canonical_evidence,verdict,correction_required,status_label'
    'next-step-candidates.csv' = 'candidate,rationale,verdict'
    'budget.csv' = 'item,count,limit,ok'
}

$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
$kProtocolId = 'PROBE_OPTIMIZATION_AUDIT_V1'
$kProtocolHash = 'fnv1a64:b36b4745b9b4807f'
$kVerdicts = @('C1_OPTIMIZATION_INSUFFICIENCY',
    'C2_STANDARDIZATION',
    'C3_ADAM_COORDINATE_DEPENDENCE',
    'C4_TRAINING_INDETERMINACY',
    'C5_CALIBRATION_SELECTION',
    'UNDETERMINED')
$script:FixtureInput = $null
$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function Fail([string]$Message) { throw "probe-optimization public export: $Message" }
function Safe([string]$Text) {
    return $Text -notmatch '(?im)([a-z]:[\/]|\\\\[^\/\s]+[\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll|bin|exe)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|hardware[_ -]?identifier|android_id|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|\badb\s+-s\s+|\b(?:raw[_ -]?(?:checkpoint|parameters?)|raw[_ -]?(?:adam|optimizer)(?:[_ -]?state)?|raw[_ -]?tensor(?:[_ -]?(?:dump|data))?|raw[_ -]?(?:projection|probe|weight|logit|hidden))\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}
function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}
function RequireUnderRepository([string]$Path, [string]$Purpose) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs')) + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { Fail "$Purpose must remain under docs/: $Path" }
    return $full
}
function RequireOutputRoot([string]$Path) {
    $full = RequireUnderRepository $Path 'OutputRoot'
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { [void](New-Item -ItemType Directory -Path $Path -Force) }
    return $full
}
function SourcePath([string]$Name) {
    if ($script:FixtureInput) { return (Join-Path $script:FixtureInput $Name) }
    return (Join-Path $ReportRoot $Name)
}
function RequireHeader([string]$Name, [string]$Expected) {
    $path = SourcePath $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "source file missing: $Name" }
    $line = Get-Content -LiteralPath $path -TotalCount 1
    if ([string]::IsNullOrWhiteSpace($line)) { Fail "source schema mismatch: $Name" }
    # Export-Csv quotes column names; compare parsed ordered names, not quoting style.
    $actual = @($line.Split(',') | ForEach-Object { $_.Trim('"') }) -join ','
    if ($actual -cne $Expected) { Fail "source schema mismatch: $Name" }
}
function Write-CanonicalFixtureCsv([string]$Dir, [string]$Name, [string]$PartialHeader, [string[]]$Rows) {
    $partialColumns = @($PartialHeader.Split(','))
    $columns = @($sourceSchemas[$Name].Split(','))
    $canonicalRows = foreach ($row in $Rows) {
        $values = @($row.Split(','))
        if ($values.Count -ne $partialColumns.Count) { Fail "self-test fixture row shape mismatch: $Name" }
        $fields = @{}
        for ($i = 0; $i -lt $partialColumns.Count; $i++) { $fields[$partialColumns[$i]] = $values[$i] }
        ($columns | ForEach-Object { if ($fields.ContainsKey($_)) { $fields[$_] } else { '0' } }) -join ','
    }
    Write-FixtureCsv (Join-Path $Dir $Name) $sourceSchemas[$Name] @($canonicalRows)
}
function Set-FixtureField([string]$Dir, [string]$Name, [int]$Index, [string]$Field, [string]$Value) {
    $path = Join-Path $Dir $Name
    $rows = @(Import-Csv -LiteralPath $path)
    $rows[$Index].$Field = $Value
    $rows | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding utf8
}
function ParseFinite([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float, $script:Invariant, [ref]$parsed)) { Fail "non-numeric value: $RowName.$Field" }
    if ([double]::IsNaN($parsed) -or $parsed -eq [double]::PositiveInfinity -or $parsed -eq [double]::NegativeInfinity) { Fail "non-finite value: $RowName.$Field" }
    return $parsed
}
function ParseInt([string]$Value, [string]$Field, [string]$RowName) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, $script:Invariant, [ref]$parsed)) { Fail "non-integer value: $RowName.$Field" }
    return $parsed
}
function AssertConfig([string]$Value, [string]$Field, [string]$RowName) {
    if ($allConfigs -notcontains $Value) { Fail "unknown configuration_id: $RowName.$Field = $Value" }
}

function AssertSourceEvidence() {
    foreach ($name in $sourceFiles) { RequireHeader $name $sourceSchemas[$name] }
    $ds = @(Import-Csv -LiteralPath (SourcePath 'dataset-usage.csv'))
    if ($ds.Count -ne 4) { Fail 'dataset-usage row count mismatch (expected 4)' }
    foreach ($row in $ds) {
        if ($row.dataset -eq 'TRAIN' -and $row.hash -ne $kTrainHash) { Fail 'TRAIN hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_CALIBRATION_V1' -and $row.hash -ne $kCalibrationHash) { Fail 'CAL hash pin mismatch' }
        if ($row.dataset -eq 'MARGIN_DEVELOPMENT_V1' -and $row.hash -ne $kDevelopmentHash) { Fail 'DEV hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.hash -ne $kFinalHash) { Fail 'FINAL hash pin mismatch' }
        if ($row.dataset -eq 'AR_FINAL_HOLDOUT_V3' -and $row.rows -ne '0') { Fail 'FINAL holdout must remain unopened' }
    }
    $cfg = @(Import-Csv -LiteralPath (SourcePath 'configuration.csv'))
    if ($cfg.Count -ne 4) { Fail 'configuration row count mismatch (expected 4)' }
    foreach ($row in $cfg) { AssertConfig $row.configuration_id 'configuration_id' 'configuration.csv' }
    $missing = @($allConfigs | Where-Object { $_ -notin $cfg.configuration_id })
    if ($missing.Count -gt 0) { Fail "configuration.csv missing: $($missing -join ', ')" }

    foreach ($name in @('legacy-vs-canonical-probe.csv', 'corrected-layer-curve.csv',
        'corrected-attention-taps.csv', 'feature-geometry.csv',
        'row-nullspace.csv', 'calibration-selection.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
        foreach ($row in $rows) {
            AssertConfig $row.configuration_id 'configuration_id' $name
        }
    }
    $opt = @(Import-Csv -LiteralPath (SourcePath 'optimization-summary.csv'))
    if ($opt.Count -eq 0) { Fail 'optimization-summary.csv empty' }
    foreach ($row in $opt) {
        AssertConfig $row.configuration_id 'configuration_id' 'optimization-summary.csv'
        if ($row.solver -like 'CANONICAL_*') {
            [void](ParseFinite $row.grad_norm 'grad_norm' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
            [void](ParseFinite $row.objective 'objective' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
        }
        [void](ParseFinite $row.train_ce 'train_ce' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
        [void](ParseInt $row.dev_exact 'dev_exact' "optimization-summary:$($row.configuration_id):$($row.tap):$($row.condition)")
    }
    foreach ($name in @('diagnosis.csv', 'previous-result-corrections.csv',
        'next-step-candidates.csv', 'budget.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
    }
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    if ($kVerdicts -notcontains $diag.verdict) { Fail "diagnosis verdict outside fixed set: $($diag.verdict)" }
    if ($diag.criteria_fixed_before_results -ne 'true') { Fail 'diagnosis criteria_fixed_before_results must be true' }
    $budget = @(Import-Csv -LiteralPath (SourcePath 'budget.csv'))
    foreach ($row in $budget) {
        if ($row.ok -ne 'true') { Fail "budget limit exceeded: $($row.item)" }
        if ([int]$row.count -gt [int]$row.limit) { Fail "budget count above limit: $($row.item)" }
    }
    $lvc = @(Import-Csv -LiteralPath (SourcePath 'legacy-vs-canonical-probe.csv'))
    foreach ($row in $lvc) {
        [void](ParseInt $row.legacy_dev_exact 'legacy_dev_exact' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
        [void](ParseInt $row.canonical_lbfgs_dev_exact 'canonical_lbfgs_dev_exact' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
        [void](ParseFinite $row.canonical_lbfgs_train_ce 'canonical_lbfgs_train_ce' "legacy-vs-canonical-probe:$($row.configuration_id):$($row.tap)")
    }
}

function CopySource([string]$Name) {
    $source = SourcePath $Name
    $text = [IO.File]::ReadAllText($source)
    if (-not (Safe $text)) { Fail "unsafe source content: $Name" }
    WriteUtf8 $Name $text
}
function GetSha256([string]$Name) {
    return (Get-FileHash -LiteralPath (Join-Path $OutputRoot $Name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
function AssertBundle() {
    $entries = @(Get-ChildItem -LiteralPath $OutputRoot -Force)
    if (@($entries | Where-Object { $_.PSIsContainer }).Count -ne 0) { Fail 'public bundle must not contain subdirectories' }
    $actual = @($entries.Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) { Fail 'public bundle allow-list mismatch' }
    foreach ($entry in $entries) {
        if (-not (Safe ([IO.File]::ReadAllText($entry.FullName)))) { Fail "unsafe generated file: $($entry.Name)" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.schema -ne $kProtocolId -or $manifest.schema_version -ne 1) { Fail 'manifest schema mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) { Fail 'manifest run accounting mismatch' }
    if ($manifest.protocol_hash -ne $kProtocolHash) { Fail 'manifest protocol_hash mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}
function NewReadme() {
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    $lvc = @(Import-Csv -LiteralPath (SourcePath 'legacy-vs-canonical-probe.csv'))
    $maxDrop = @($lvc | Where-Object { $_.tap -eq 'ATT_UPDATE' -or $_.tap -eq 'CTX_CONCAT' } | ForEach-Object {
            "$($_.configuration_id):L$($_.layer) $($_.tap) legacy=$($_.legacy_dev_exact) canonical=$($_.canonical_lbfgs_dev_exact)" }) -join '; '
    @"
# L19 probe-optimization audit, August 2026

This bundle is a host-only CPU audit of the probe-optimization hypothesis for
the previously reported CTX_CONCAT vs ATT_UPDATE dev-token-exact drop. The
published READOUT_PROBE_V1 legacy anchors (24/6, 37/24, 57/47, 68/64) are
reproduced bitwise by the runner (cond-1, calibration-selected legacy Adam
probe); this bundle records what a coordinate-stable canonical solver
(PCA-whitened features, L2 on whitened weights only, gauge-fixed, L-BFGS
certified with GD as reference) finds at the same taps.

Headline: at every max-drop layer the canonical CTX and ATT probes reach the
same convergence point (identical whitened-space objective and dev token
exact), so the projection drop is an artifact of the legacy Adam pipeline
(C1_OPTIMIZATION_INSUFFICIENCY, 4/4 layers), not of the representation.

$maxDrop

Protocol $kProtocolId version 6 (AMENDMENT_1..5, fixed before results),
hash $kProtocolHash. Dataset partitions pinned: TRAIN
$kTrainHash, MARGIN_CALIBRATION_V1 $kCalibrationHash,
MARGIN_DEVELOPMENT_V1 $kDevelopmentHash; AR_FINAL_HOLDOUT_V3
$kFinalHash remains unopened. All evidence is CPU host-side; no device,
QAIRT, or QNN involvement.

## Current status

Full-rank classifier-coordinate transport and equivalent whitened-space
objectives remain mathematical evidence. Learned-probe absolute scores and
z-statistics below are superseded by the TRAIN row-contract correction and
are not current root-cause evidence.

## Superseding measurement correction (2026-08-05)

Both legacy and canonical probes in this historical bundle were trained on
rows containing four TRAIN-contract conflicts. Their absolute scores and
z-statistics are excluded from later causal claims until corrected-row
regeneration. The algebraic conclusion that a full-rank output projection
admits classifier-coordinate transport, and the equivalent whitened-space
objectives, remain valid; the historical learned scores are not used as
root-cause evidence.
"@
}


function Write-FixtureCsv([string]$Path, [string]$Header, [string[]]$Rows) {
    [IO.File]::WriteAllLines($Path, (@($Header) + $Rows), $utf8)
}
function ExpectSelfTestRejects([string]$Label, [string]$ExpectedMarker, [scriptblock]$Action) {
    # Every case starts with the exact positive fixture, including files removed by earlier cases.
    foreach ($file in @(Get-ChildItem -LiteralPath $script:FixtureInput -File)) {
        [IO.File]::Delete($file.FullName)
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $script:PristineSelfTestInput -File)) {
        [IO.File]::Copy($file.FullName, (Join-Path $script:FixtureInput $file.Name))
    }
    try { & $Action } catch {
        if ($_.Exception.Message.Contains($ExpectedMarker, [StringComparison]::Ordinal)) {
            Write-Host "self-test expected rejection: $Label [$ExpectedMarker]"
            return
        }
        Fail "self-test wrong rejection for ${Label}: $($_.Exception.Message) (expected $ExpectedMarker)"
    }
    Fail "self-test negative case did not fail: $Label"
}
function New-SyntheticSelfTestRoot() {
    $root = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\exporter-selftest-fixture-" + [Guid]::NewGuid().ToString('N'))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $root.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $input = Join-Path $root 'input'
    $output = Join-Path $root 'output'
    [void](New-Item -ItemType Directory -Path $input -Force)
    [void](New-Item -ItemType Directory -Path $output -Force)
    return [pscustomobject]@{ Root = $root; Input = $input; Output = $output }
}
function Write-DatasetUsageFixture([string]$Dir) {
    Write-FixtureCsv (Join-Path $Dir 'dataset-usage.csv') 'dataset,role,hash,rows' @(
        "TRAIN,probe,$kTrainHash,32",
        "MARGIN_CALIBRATION_V1,step_select,$kCalibrationHash,144",
        "MARGIN_DEVELOPMENT_V1,eval,$kDevelopmentHash,144",
        "AR_FINAL_HOLDOUT_V3,unopened,$kFinalHash,0")
}
function Write-ConfigurationFixture([string]$Dir, [string]$Header) {
    Write-FixtureCsv (Join-Path $Dir 'configuration.csv') $Header @(
        'L19_SEED_1,1,19,320,2,16',
        'L19_SEED_2,2,19,320,1,4',
        'L19_SEED_4,4,19,320,0,12',
        'L18_SEED_2_CONTROL,2,18,320,2,4')
}
function Write-DatasetAnchorFixture([string]$Dir) {
    Write-FixtureCsv (Join-Path $Dir 'dataset-anchors.csv') 'dataset,role,hash,rows' @(
        "TRAIN,probe,$kTrainHash,32",
        "MARGIN_CALIBRATION_V1,step_select,$kCalibrationHash,144",
        "MARGIN_DEVELOPMENT_V1,eval,$kDevelopmentHash,144",
        "AR_FINAL_HOLDOUT_V3,unopened,$kFinalHash,0")
}
function Write-TrajectoryAnchorFixture([string]$Dir) {
    $rows = @()
    foreach ($cfg in $allConfigs) {
        $p = $kPinned[$cfg]
        $rows += "$($cfg),AR_DEV_SELECTED,$($p.arSelected),token_exact,$($p.arSelTok),true"
        $rows += "$($cfg),AR_DEV_SELECTED,$($p.arSelected),sequence_exact,$($p.arSelSeq),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,token_exact,$($p.arFinalTok),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,sequence_exact,$($p.arFinalSeq),true"
        $rows += "$($cfg),AR_DEV_FINAL,320,autoregressive_nll,$($p.arFinalNll),true"
    }
    Write-FixtureCsv (Join-Path $Dir 'trajectory-anchors.csv') 'configuration_id,checkpoint,step,metric,value,match' $rows
}
if ($SelfTest) {
    $fixture = New-SyntheticSelfTestRoot
    try {
        $in = $fixture.Input
        Write-DatasetUsageFixture $in
        Write-CanonicalFixtureCsv $in 'configuration.csv' 'configuration_id,seed,layers,final_step,max_drop_block,ar_selected_step' @(
            'L19_SEED_1,1,19,320,2,16','L19_SEED_2,2,19,320,1,4','L19_SEED_4,4,19,320,0,12','L18_SEED_2_CONTROL,2,18,320,2,4')
        Write-CanonicalFixtureCsv $in 'legacy-vs-canonical-probe.csv' 'configuration_id,layer,tap,legacy_dev_exact,canonical_gd_dev_exact,canonical_lbfgs_dev_exact,canonical_minus_legacy_dev_exact,canonical_lbfgs_train_ce' @(
            'L19_SEED_1,10,ATT,20,22,23,3,1.25')
        foreach ($name in @('corrected-layer-curve.csv','corrected-attention-taps.csv','feature-geometry.csv','row-nullspace.csv','calibration-selection.csv')) {
            Write-CanonicalFixtureCsv $in $name 'configuration_id,layer,tap,value' @('L19_SEED_1,10,ATT,0.5')
        }
        Write-CanonicalFixtureCsv $in 'optimization-summary.csv' 'configuration_id,layer,tap,condition,solver,init,lambda,converged,grad_norm,objective,train_ce,dev_exact' @(
            'L19_SEED_1,10,ATT,CLEAN,CANONICAL_LBFGS,zero,0,1,0.01,0.5,1.0,23')
        Write-CanonicalFixtureCsv $in 'diagnosis.csv' 'verdict,criteria_fixed_before_results,reasons' @(
            'C1_OPTIMIZATION_INSUFFICIENCY,true,synthetic-fixture-contract-check')
        Write-CanonicalFixtureCsv $in 'previous-result-corrections.csv' 'id,note' @('c1,synthetic')
        Write-CanonicalFixtureCsv $in 'next-step-candidates.csv' 'id,note' @('n1,synthetic')
        Write-CanonicalFixtureCsv $in 'budget.csv' 'item,count,limit,ok' @('trajectory,4,4,true')
        $script:FixtureInput = $in
        AssertSourceEvidence
        $script:PristineSelfTestInput = Join-Path $fixture.Root 'pristine'
        [void](New-Item -ItemType Directory -Path $script:PristineSelfTestInput)
        foreach ($file in @(Get-ChildItem -LiteralPath $in -File)) {
            [IO.File]::Copy($file.FullName, (Join-Path $script:PristineSelfTestInput $file.Name))
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'source schema mismatch' 'source schema mismatch: corrected-layer-curve.csv' {
            $path = Join-Path $in 'corrected-layer-curve.csv'
            $lines = [IO.File]::ReadAllLines($path)
            $lines[0] = $lines[0].Replace('configuration_id', 'wrong_configuration_id')
            [IO.File]::WriteAllLines($path, $lines, $utf8)
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'diagnosis criteria_fixed' 'diagnosis criteria_fixed_before_results must be true' {
            Set-FixtureField $in 'diagnosis.csv' 0 'criteria_fixed_before_results' 'false'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'budget ok=false' 'budget limit exceeded: trajectory' {
            Set-FixtureField $in 'budget.csv' 0 'ok' 'false'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'non-finite train_ce' 'non-finite value: optimization-summary:' {
            Set-FixtureField $in 'optimization-summary.csv' 0 'grad_norm' 'nan'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'missing required file' 'dataset-usage.csv' {
            Remove-Item -LiteralPath (Join-Path $in 'dataset-usage.csv') -Force
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'bad TRAIN hash' 'TRAIN hash pin mismatch' {
            Set-FixtureField $in 'dataset-usage.csv' 0 'hash' 'fnv1a64:0000000000000000'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'unknown verdict' 'diagnosis verdict outside fixed set:' {
            Set-FixtureField $in 'diagnosis.csv' 0 'verdict' 'NOT_A_REAL_VERDICT'
            AssertSourceEvidence
        }
        Write-Host 'probe-optimization public export: SELF-TEST PASS (fixture-contained)'
    } finally {
        $script:FixtureInput = $null
        $script:PristineSelfTestInput = $null
        if (Test-Path -LiteralPath $fixture.Root) { [IO.Directory]::Delete($fixture.Root, $true) }
    }
    exit 0
}

$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $ReportRoot -PathType Container)) { Fail 'ReportRoot does not exist' }
AssertSourceEvidence
foreach ($name in $sourceFiles) { CopySource $name }
WriteUtf8 'README.md' (NewReadme)
$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{ name = $name; sha256 = (GetSha256 $name) }
}
$manifest = [ordered]@{
    schema = $kProtocolId
    schema_version = 1
    protocol_hash = $kProtocolHash
    generated_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ')
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = $manifestFiles
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle
Write-Host "probe-optimization public export: PASS ($OutputRoot)"
