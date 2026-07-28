# SPDX-License-Identifier: Apache-2.0
# Runs the bounded historical condition before the long fixed-checkpoint study.
param(
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [string]$ExpectedBuildId = "2.48.40.260702151143",
    [switch]$SkipBuild,
    [switch]$SkipInstall,
    [switch]$SkipAudit
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$runner = Join-Path $PSScriptRoot 'run_qnn_headless_tests.ps1'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$common = @{ QairtSdkRoot = $QairtSdkRoot; ExpectedBuildId = $ExpectedBuildId; TimeoutSeconds = 14400 }
if ($SkipBuild) { $common.SkipBuild = $true }
if ($SkipInstall) { $common.SkipInstall = $true }
if ($SkipAudit) { $common.SkipAudit = $true }
function Get-ReportValue([string[]]$Lines, [string]$Key) {
    $matches = @($Lines | Where-Object { $_ -like "$Key=*" })
    if ($matches.Count -ne 1) { throw "Expected exactly one report field: $Key" }
    return $matches[0].Substring($Key.Length + 1)
}
function Get-ReportBlocks([string[]]$Lines, [string]$FirstKey) {
    return @([regex]::Split(($Lines -join "`n"), "(?m)(?=^$([regex]::Escape($FirstKey))=)") |
        Where-Object { $_ -match "^$([regex]::Escape($FirstKey))=" })
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$baselineId = "late-baseline-$stamp"
& $runner @common -Suite 'qnn-adam-late-baseline' -RunId $baselineId
$baselineReport = @(Get-Content -LiteralPath (
    Join-Path $repositoryRoot "build/reports/qnn-headless/$baselineId/device-report.txt"))
if ((Get-ReportValue $baselineReport 'finite_seed_count') -ne '5/5') {
    throw 'The post-fix lr=0.003 baseline was not finite on all five seeds.'
}
foreach ($seed in 1..5) {
    if ((Get-ReportValue $baselineReport "seed_${seed}_first_nonfinite_step") -ne '-1' -or
        (Get-ReportValue $baselineReport "seed_${seed}_qnn_execute_result") -ne 'SUCCESS') {
        throw "The post-fix lr=0.003 baseline failed validation for seed $seed."
    }
}
# The second invocation can reuse the verified APK/install from the baseline.
$common.SkipBuild = $true
$common.SkipInstall = $true
$diagnosticId = "late-diagnostic-$stamp"
& $runner @common -Suite 'qnn-adam-late-diagnostic' -RunId $diagnosticId
$diagnosticReport = @(Get-Content -LiteralPath (
    Join-Path $repositoryRoot "build/reports/qnn-headless/$diagnosticId/device-report.txt"))
if ((Get-ReportValue $diagnosticReport 'finite_seed_count') -ne '0/5' -or
    (Get-ReportValue $diagnosticReport 'failing_checkpoint_count') -ne '5') {
    throw 'The legacy boundary diagnostic did not capture five failing checkpoints.'
}
foreach ($seed in 1..5) {
    if ((Get-ReportValue $diagnosticReport "seed_${seed}_first_nonfinite_tensor") -ne 'tap_SQUARE2' -or
        (Get-ReportValue $diagnosticReport "seed_${seed}_fixed_replay_count") -ne '100' -or
        (Get-ReportValue $diagnosticReport "seed_${seed}_fixed_replay_reproducible") -ne 'true') {
        throw "The fixed legacy replay failed validation for seed $seed."
    }
}
$twoByTwo = Get-ReportBlocks $diagnosticReport 'two_by_two_checkpoint_seed'
if ($twoByTwo.Count -ne 5) { throw 'Expected five 2x2 comparison blocks.' }
foreach ($block in $twoByTwo) {
    $lines = @($block -split "`n")
    if ((Get-ReportValue $lines 'two_by_two_A_cpu_gradient_cpu_adam_finite') -ne 'true' -or
        (Get-ReportValue $lines 'two_by_two_B_htp_gradient_cpu_adam_finite') -ne 'false' -or
        (Get-ReportValue $lines 'two_by_two_C_cpu_gradient_htp_adam_finite') -ne 'true' -or
        (Get-ReportValue $lines 'two_by_two_D_htp_gradient_htp_adam_finite') -ne 'false') {
        throw 'A legacy 2x2 block did not isolate the failure to the HTP gradient.'
    }
}
$postFix = Get-ReportBlocks $diagnosticReport 'post_fix_same_checkpoint_seed'
if ($postFix.Count -ne 5) { throw 'Expected five post-fix replay blocks.' }
foreach ($block in $postFix) {
    $lines = @($block -split "`n")
    if ((Get-ReportValue $lines 'post_fix_same_checkpoint_replay_count') -ne '100' -or
        (Get-ReportValue $lines 'post_fix_same_checkpoint_all_finite') -ne 'true' -or
        (Get-ReportValue $lines 'post_fix_same_checkpoint_deterministic') -ne 'true') {
        throw 'A post-fix same-checkpoint replay block failed validation.'
    }
    foreach ($path in 'a','b','c','d') {
        if ((Get-ReportValue $lines "post_fix_same_checkpoint_path_${path}_finite") -ne 'true') {
            throw "A post-fix 2x2 path was non-finite: $path"
        }
    }
}
Write-Host 'late_nonfinite_focused_status=SUCCESS'
