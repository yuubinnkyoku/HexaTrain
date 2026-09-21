# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Allow-list public exporter for the L19 output-projection information audit
# (host-only CPU evidence). Copies only the allow-listed files from the
# private report root, schema-checks each, verifies dataset hashes and budget,
# and scans for private identifiers.
[CmdletBinding()]
param(
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-output-projection-audit'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-l19-output-projection-information-audit-2026-08'),
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json',
    'configuration.csv', 'dataset-usage.csv',
    'projection-matrix-summary.csv', 'singular-value-summary.csv',
    'rank-and-conditioning.csv', 'probe-transport-summary.csv',
    'probe-transport-by-seed.csv', 'float-double-comparison.csv',
    'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
    'depth-control.csv', 'diagnosis.csv',
    'previous-result-correction.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceFiles = @(
    'configuration.csv', 'dataset-usage.csv',
    'projection-matrix-summary.csv', 'singular-value-summary.csv',
    'rank-and-conditioning.csv', 'probe-transport-summary.csv',
    'probe-transport-by-seed.csv', 'float-double-comparison.csv',
    'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
    'depth-control.csv', 'diagnosis.csv',
    'previous-result-correction.csv', 'next-step-candidates.csv', 'budget.csv'
)
$sourceSchemas = [ordered]@{
    'configuration.csv' = 'configuration_id,seed,layers,final_step,max_drop_block,ar_selected_step'
    'dataset-usage.csv' = 'dataset,role,hash,rows'
    'projection-matrix-summary.csv' = 'configuration_id,layer,sigma_min,sigma_max,condition_double,condition_float,math_rank,float_rank,effective_rank,participation_ratio,frobenius_norm,spectral_norm,determinant_sign,log_abs_determinant'
    'singular-value-summary.csv' = 'configuration_id,layer,sigma_min,sigma_max,condition_double,math_rank,float_rank,effective_rank'
    'rank-and-conditioning.csv' = 'configuration_id,layer,sigma_min,sigma_max,condition_double,condition_float,math_rank,float_rank,effective_rank,participation_ratio,frobenius_norm,spectral_norm,determinant_sign,log_abs_determinant'
    'probe-transport-summary.csv' = 'configuration_id,layer,partition,context_token_exact,from_scratch_token_exact,transport_token_exact,warm_start_token_exact,max_logit_diff,mean_logit_diff,rms_logit_diff,argmax_flips,token_exact_diff'
    'probe-transport-by-seed.csv' = 'configuration_id,layer,partition,context_token_exact,from_scratch_token_exact,transport_token_exact,warm_start_token_exact,max_logit_diff,mean_logit_diff,rms_logit_diff,argmax_flips,token_exact_diff'
    'float-double-comparison.csv' = 'configuration_id,layer,partition,max_logit_diff,mean_logit_diff,rms_logit_diff,argmax_flips,token_exact_diff'
    'nullspace-summary.csv' = 'configuration_id,layer,overall_lost_fraction'
    'from-scratch-vs-transport.csv' = 'configuration_id,layer,partition,from_scratch_exact,transport_exact,warm_start_exact,transport_argmax_flips'
    'depth-control.csv' = 'configuration_id,layers,max_drop_block,math_rank,condition_double,nullspace_fraction'
    'diagnosis.csv' = 'verdict,preserved_layers,ill_conditioned_layers,lost_layers,audited_layers,criteria_fixed_before_results,reason'
    'previous-result-correction.csv' = 'previous_claim,previous_evidence,this_audit_verdict,correction_required'
    'next-step-candidates.csv' = 'candidate,rationale,verdict'
    'budget.csv' = 'item,count,limit,ok'
}

$allConfigs = @('L19_SEED_1', 'L19_SEED_2', 'L19_SEED_4', 'L18_SEED_2_CONTROL')
$kTrainHash = 'fnv1a64:5a64ca2d1aa7f29f'
$kCalibrationHash = 'fnv1a64:71806d5bf19c090a'
$kDevelopmentHash = 'fnv1a64:f06fcc3e2d12ca99'
$kFinalHash = 'fnv1a64:aa5081e6df658b4a'
$kProtocolHash = 'fnv1a64:c35a2e6ae3102772'
$kVerdicts = @('OUTPUT_PROJECTION_PRESERVES_INFORMATION',
    'OUTPUT_PROJECTION_ILL_CONDITIONED',
    'OUTPUT_PROJECTION_LOSES_INFORMATION',
    'SEED_LAYER_DEPENDENT',
    'UNDETERMINED')
$script:FixtureInput = $null
$script:Invariant = [Globalization.CultureInfo]::InvariantCulture

function Fail([string]$Message) { throw "output-projection public export: $Message" }
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
    if (-not [double]::IsFinite($parsed)) { Fail "non-finite value: $RowName.$Field" }
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

    foreach ($name in @('projection-matrix-summary.csv', 'singular-value-summary.csv',
        'rank-and-conditioning.csv', 'probe-transport-summary.csv',
        'probe-transport-by-seed.csv', 'float-double-comparison.csv',
        'nullspace-summary.csv', 'from-scratch-vs-transport.csv',
        'depth-control.csv')) {
        $rows = @(Import-Csv -LiteralPath (SourcePath $name))
        if ($rows.Count -lt 1) { Fail "$name empty" }
        foreach ($row in $rows) {
            AssertConfig $row.configuration_id 'configuration_id' $name
        }
    }
    foreach ($name in @('diagnosis.csv', 'previous-result-correction.csv',
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
    $nullspace = @(Import-Csv -LiteralPath (SourcePath 'nullspace-summary.csv'))
    foreach ($row in $nullspace) {
        [void](ParseFinite $row.overall_lost_fraction 'overall_lost_fraction' "nullspace-summary:$($row.configuration_id):L$($row.layer)")
    }
    $transport = @(Import-Csv -LiteralPath (SourcePath 'probe-transport-summary.csv'))
    foreach ($row in $transport) {
        [void](ParseFinite $row.max_logit_diff 'max_logit_diff' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
        [void](ParseInt $row.argmax_flips 'argmax_flips' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
        [void](ParseInt $row.token_exact_diff 'token_exact_diff' "probe-transport-summary:$($row.configuration_id):L$($row.layer):$($row.partition)")
    }
}

function CopySafe([string]$Name) {
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
    if ($manifest.schema -ne 'OUTPUT_PROJECTION_AUDIT_V1' -or $manifest.schema_version -ne 1) { Fail 'manifest schema mismatch' }
    if ($manifest.final_holdout_opened -ne $false -or $manifest.device_runs -ne 0 -or $manifest.htp_runs -ne 0) { Fail 'manifest run accounting mismatch' }
    if ($manifest.protocol_hash -ne $kProtocolHash) { Fail 'manifest protocol_hash mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
}
function NewReadme() {
    $diag = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0]
    $summary = @(Import-Csv -LiteralPath (SourcePath 'probe-transport-summary.csv') | Where-Object { $_.partition -eq 'DEVELOPMENT' } | ForEach-Object { "$($_.configuration_id): ctx=$($_.context_token_exact) trans=$($_.transport_token_exact) scratch=$($_.from_scratch_token_exact) maxdiff=$($_.max_logit_diff)" }) -join '; '
    @"
# L19 output-projection information audit, August 2026

This bundle is a host-only CPU audit that tests whether the Attention output
projection actually discards linear next-token information. The null
hypothesis is that the projection is invertible and the information is still
present; the alternative is rank deficiency or extreme ill-conditioning that
makes the class direction unreachable.

For each of the four pinned configurations (L19 seeds 1/2/4 and the L18 depth
control) the canonical trajectory is regenerated with the pinned Adam/LEGACY
training recipe and the FINAL step-320 checkpoint is used. At every target
layer the 16x16 Attention output-projection matrix W is decomposed in double
precision. At the previously reported max-drop layer, a 32-way linear softmax
probe is trained on the concatenated head context (CTX_CONCAT), transported
through W using the pseudoinverse with the protocol's fixed tolerance, and
evaluated on the Attention update (ATT_UPDATE). The same ATT_UPDATE probe is
also trained from scratch and warm-started from the transported weights.

Dataset roles follow the pinned protocol: TRAIN = probe learning (32 rows),
MARGIN_CALIBRATION_V1 = step selection only, MARGIN_DEVELOPMENT_V1 = final
evaluation only, AR_FINAL_HOLDOUT_V3 = unopened. Budget (pre-registered
OUTPUT_PROJECTION_AUDIT_V1): CPU trajectory regenerations <= 4, matrix
decompositions <= 60, full probe transports <= 24, warm-start trainings <= 24.

## Verdict

Diagnosis (fixed thresholds, never tuned):
**$($diag.verdict)**

$($diag.reason)

Projection-transport parity (DEVELOPMENT, max-drop layer):
$summary

Interpretation, thresholds and all raw values are in the CSVs; the decision
rules are pinned in the private protocol (OUTPUT_PROJECTION_AUDIT_V1) before
any results were produced.

## Files

- configuration.csv / dataset-usage.csv - configs and dataset role hashes
- projection-matrix-summary.csv / singular-value-summary.csv - matrix decomposition
- rank-and-conditioning.csv - rank and condition number summary
- probe-transport-summary.csv / probe-transport-by-seed.csv - transport parity
- float-double-comparison.csv - float vs double transport accuracy
- nullspace-summary.csv - null-space fraction of the context probe
- from-scratch-vs-transport.csv - optimization comparison
- depth-control.csv - L18 depth control comparison
- diagnosis.csv - formal conclusion
- previous-result-correction.csv - how to rephrase the previous report
- next-step-candidates.csv - candidate follow-ups
- budget.csv - pre-registered execution budget accounting
- manifest.json - SHA-256 allow-list manifest
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
        Write-ConfigurationFixture $in 'configuration_id,seed,layers,final_step,max_drop_block,ar_selected_step'
        foreach ($name in @('projection-matrix-summary.csv','singular-value-summary.csv','rank-and-conditioning.csv','float-double-comparison.csv','from-scratch-vs-transport.csv','depth-control.csv')) {
            Write-CanonicalFixtureCsv $in $name 'configuration_id,layer,value' @('L19_SEED_1,10,0.5')
        }
        Write-CanonicalFixtureCsv $in 'probe-transport-summary.csv' 'configuration_id,layer,partition,max_logit_diff,argmax_flips,token_exact_diff' @(
            'L19_SEED_1,10,AR_DEVELOPMENT_V3,0.001,0,0')
        Write-CanonicalFixtureCsv $in 'probe-transport-by-seed.csv' 'configuration_id,layer,partition,max_logit_diff,argmax_flips,token_exact_diff' @(
            'L19_SEED_1,10,AR_DEVELOPMENT_V3,0.001,0,0')
        Write-CanonicalFixtureCsv $in 'nullspace-summary.csv' 'configuration_id,layer,overall_lost_fraction' @(
            'L19_SEED_1,10,0.05')
        Write-CanonicalFixtureCsv $in 'diagnosis.csv' 'verdict,criteria_fixed_before_results,reasons' @(
            'OUTPUT_PROJECTION_PRESERVES_INFORMATION,true,synthetic-fixture-contract-check')
        Write-CanonicalFixtureCsv $in 'previous-result-correction.csv' 'id,note' @('c1,synthetic')
        Write-CanonicalFixtureCsv $in 'next-step-candidates.csv' 'id,note' @('n1,synthetic')
        Write-CanonicalFixtureCsv $in 'budget.csv' 'item,count,limit,ok' @('audits,4,4,true')
        $script:FixtureInput = $in
        AssertSourceEvidence
        $script:PristineSelfTestInput = Join-Path $fixture.Root 'pristine'
        [void](New-Item -ItemType Directory -Path $script:PristineSelfTestInput)
        foreach ($file in @(Get-ChildItem -LiteralPath $in -File)) {
            [IO.File]::Copy($file.FullName, (Join-Path $script:PristineSelfTestInput $file.Name))
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'source schema mismatch' 'source schema mismatch: projection-matrix-summary.csv' {
            $path = Join-Path $in 'projection-matrix-summary.csv'
            $lines = [IO.File]::ReadAllLines($path)
            $lines[0] = $lines[0].Replace('configuration_id', 'wrong_configuration_id')
            [IO.File]::WriteAllLines($path, $lines, $utf8)
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'criteria_fixed=false' 'diagnosis criteria_fixed_before_results must be true' {
            Set-FixtureField $in 'diagnosis.csv' 0 'criteria_fixed_before_results' 'false'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'budget ok=false' 'budget limit exceeded: audits' {
            Set-FixtureField $in 'budget.csv' 0 'ok' 'false'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'missing dataset-usage' 'dataset-usage.csv' {
            Remove-Item -LiteralPath (Join-Path $in 'dataset-usage.csv') -Force
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'bad FINAL rows' 'FINAL holdout must remain unopened' {
            Set-FixtureField $in 'dataset-usage.csv' 3 'rows' '24'
            AssertSourceEvidence
        }
        $script:FixtureInput = $in
        ExpectSelfTestRejects 'non-finite nullspace' 'non-finite value: nullspace-summary:' {
            Set-FixtureField $in 'nullspace-summary.csv' 0 'overall_lost_fraction' 'nan'
            AssertSourceEvidence
        }
        Write-Host 'output-projection public export: self-test PASS (fixture-contained)'
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
foreach ($name in $sourceFiles) { CopySafe $name }
WriteUtf8 'README.md' (NewReadme)
$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{ name = $name; sha256 = (GetSha256 $name) }
}
$manifest = [ordered]@{
    schema = 'OUTPUT_PROJECTION_AUDIT_V1'
    schema_version = 1
    protocol = 'OUTPUT_PROJECTION_AUDIT_V1'
    protocol_hash = $kProtocolHash
    verdict = @(Import-Csv -LiteralPath (SourcePath 'diagnosis.csv'))[0].verdict
    final_holdout_opened = $false
    device_runs = 0
    htp_runs = 0
    files = @($manifestFiles)
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle
Write-Host "output-projection public export: PASS ($OutputRoot)"
