# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-validation-selected'),
    [string]$DirectSeedCsv = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-direct-seed-equivalence\direct-seed-equivalence-private.csv'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-validation-selected-depth-quality-2026-08'),
    [string]$SourceCommit = '',
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @('README.md','manifest.json','direct-seed-equivalence.csv','dataset-partitions.csv',
    'dataset-overlap.csv','validation-trajectories.csv','checkpoint-selection.csv','cpu-smoke.csv',
    'htp-smoke.csv','formal-seeds.csv','generation-oracle.csv','generation-free.csv',
    'selected-step-distribution.csv','early-stop-simulation.csv','ui-validation.csv','thermal.csv')

function Fail([string]$Message) { throw "validation-selection public export: $Message" }
function Safe([string]$Text) {
    return $Text -notmatch '(?i)([a-z]:[\\/]|/data/|/sdcard/|/storage/|device[_ -]?serial|adb[_ -]?endpoint|apk_sha256|logcat|sk-[a-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}
function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}
function CsvText($Rows) { return (($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n" }

if ($SelfTest) {
    if (-not (Safe 'validation_set_hash=fnv1a64:8e1411f19126879c')) { Fail 'safe-text false rejection' }
    if (Safe 'apk_sha256=aaaaaaaa') { Fail 'APK hash negative rejection failed' }
    'SELF_TEST=PASS'
    exit 0
}
if (-not $SourceCommit) { $SourceCommit = (git -C $repoRoot rev-parse HEAD).Trim() }
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { Fail 'SourceCommit must be a full Git SHA' }
foreach ($name in @('cpu-smoke.csv','validation-trajectories.csv','early-stop-simulation.csv')) {
    if (-not (Test-Path -LiteralPath (Join-Path $InputRoot $name) -PathType Leaf)) { Fail "missing $name" }
}
if (-not (Test-Path -LiteralPath $DirectSeedCsv -PathType Leaf)) { Fail 'missing direct-seed equivalence CSV' }
$cpu = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'cpu-smoke.csv'))
$trajectory = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'validation-trajectories.csv'))
$early = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'early-stop-simulation.csv'))
$direct = @(Import-Csv -LiteralPath $DirectSeedCsv)
if ($cpu.Count -ne 5 -or $trajectory.Count -ne 115 -or $early.Count -ne 15) { Fail 'CPU row cardinality mismatch' }
if ($direct.Count -ne 5 -or @($direct | Where-Object { $_.bitwise_equivalent -notmatch '^(?i:true)$' -or $_.mismatch_count -ne '0' }).Count -ne 0) {
    Fail 'direct-seed equivalence must be complete and bitwise-equivalent 5/5'
}

[IO.Directory]::CreateDirectory($OutputRoot) | Out-Null
foreach ($file in Get-ChildItem -LiteralPath $OutputRoot -File -ErrorAction SilentlyContinue) {
    if ($file.Name -notin $allowed) { Fail "unexpected stale output $($file.Name)" }
    Remove-Item -LiteralPath $file.FullName
}

$directPublic = $direct | Select-Object configuration,seed,compared_fields,mismatch_count,bitwise_equivalent,
    legacy_seed_units,exact_seed_units,legacy_qnn_execute_count,exact_qnn_execute_count,final_parameter_hash,final_logits_hash
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
$oracle = foreach ($row in $cpu) { [pscustomobject][ordered]@{layers=[int]$row.layers;seed=[int]$row.seed;selected_step=[int]$row.selected_step;selected_exact=[int]$row.selected_oracle_exact;final_step_exact=[int]$row.final_oracle_exact;case_count=4} }
$free = foreach ($row in $cpu) { [pscustomobject][ordered]@{layers=[int]$row.layers;seed=[int]$row.seed;selected_step=[int]$row.selected_step;selected_exact=[int]$row.selected_free_exact;final_step_exact=[int]$row.final_free_exact;case_count=4} }
WriteUtf8 'generation-oracle.csv' (CsvText $oracle)
WriteUtf8 'generation-free.csv' (CsvText $free)
$distribution = $cpu | Group-Object layers,selected_step | ForEach-Object {
    $first=$_.Group[0]; [pscustomobject][ordered]@{layers=[int]$first.layers;selected_step=[int]$first.selected_step;seed_count=$_.Count}
}
WriteUtf8 'selected-step-distribution.csv' (CsvText $distribution)
$notRun = [pscustomobject][ordered]@{status='NOT_RUN';reason='CPU_CANDIDATE_REJECTED_BEFORE_HTP_GATE'}
WriteUtf8 'htp-smoke.csv' (CsvText @($notRun))
WriteUtf8 'formal-seeds.csv' (CsvText @($notRun))
WriteUtf8 'ui-validation.csv' (CsvText @([pscustomobject][ordered]@{status='NOT_RUN';reason='NO_ACCEPTED_HTP_CANDIDATE'}))
WriteUtf8 'thermal.csv' (CsvText @([pscustomobject][ordered]@{status='NOT_RUN';reason='NO_ACCEPTED_HTP_CANDIDATE'}))

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
HTP smoke and five-seed formal were not started under the predeclared gate.
The observed validation regression was not reliably predictive of improved
disjoint held-out generation tests.

Raw checkpoints, parameters, optimizer state, tensor dumps, APK data, device
identifiers, paths, and logs are excluded.
"@
WriteUtf8 'README.md' ($readme + "`n")

$fileEntries = @()
foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    $fileEntries += [ordered]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $OutputRoot $name) -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$manifest = [ordered]@{schema_version=1;source_commit=$SourceCommit;qairt_build_id='2.48.40.260702151143';result_classification='DIRECT_SEED_COMPLETE_VALIDATION_NOT_PREDICTIVE';direct_seed_equivalence='5/5';validation_schema_version=2;validation_generator_domain='ROTATED_LAST_POSITION_V2';validation_set_hash='fnv1a64:8e1411f19126879c';checkpoint_selection_mode='BEST_VALIDATION_V1';default_checkpoint_selection_mode='FINAL_STEP';cpu_candidate_accepted=$false;htp_smoke='NOT_RUN_CPU_GATE';formal='NOT_RUN_CPU_GATE';raw_checkpoints_published=$false;files=$fileEntries}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 6) + "`n")

$actual = @((Get-ChildItem -LiteralPath $OutputRoot -File).Name | Sort-Object)
if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) { Fail 'public bundle allow-list mismatch' }
foreach ($name in $actual) { if (-not (Safe ([IO.File]::ReadAllText((Join-Path $OutputRoot $name))))) { Fail "unsafe final file $name" } }
Write-Host "validation-selection public export PASS: $OutputRoot"
