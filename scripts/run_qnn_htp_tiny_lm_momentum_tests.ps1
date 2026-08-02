# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param(
    [Parameter(Mandatory = $true)][string]$QairtSdkRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedBuildId,
    [switch]$SkipInstall,
    [ValidateSet('step', 'candidate1', 'candidate2', 'candidates', 'inference', 'all')]
    [string]$Scope = 'all'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'qairt_version.ps1')
Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
    -ExpectedBuildId $ExpectedBuildId

$root = Split-Path -Parent $PSScriptRoot
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$package = 'com.yuubinnkyoku.phonelm'
$activity = "$package/.MainActivity"
$apk = Join-Path $root 'app\build\outputs\apk\debug\app-debug.apk'
$reportRoot = Join-Path $root 'build\reports\tiny-lm-htp-momentum'
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null

$online = @()
foreach ($line in (& $adb devices)) {
    if ($line -match '^(\S+)\s+device$') { $online += $Matches[1] }
}
if ($online.Count -ne 1) { throw "Expected one online device; found $($online.Count)" }
$device = $online[0]

function Invoke-Adb([string[]]$Arguments) {
    $output = & $adb -s $device @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "ADB command failed (endpoint redacted): $($Arguments -join ' ')`n$output"
    }
    $output
}

if (!$SkipInstall) { Invoke-Adb @('install', '-r', $apk) | Out-Null }

function Run-Test(
    [string]$Mode,
    [string]$Name,
    [string[]]$Required,
    [int]$PollLimit = 14400
) {
    Invoke-Adb @('shell', 'am', 'force-stop', $package) | Out-Null
    & $adb -s $device shell run-as $package rm -f files/device-test-result.txt 2>$null | Out-Null
    Invoke-Adb @('shell', 'am', 'start', '-W', '-n', $activity, '--es', 'phonelm.mode', $Mode) | Out-Null
    $result = ''
    for ($poll = 0; $poll -lt $PollLimit; $poll++) {
        Start-Sleep -Milliseconds 500
        $result = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
        if ($result -match '(?m)^status=(SUCCESS|FAILED)$') { break }
    }
    [IO.File]::WriteAllText(
        (Join-Path $reportRoot "$Name-result.txt"),
        $result + "`n",
        [Text.UTF8Encoding]::new($false))
    if ($result -notmatch '(?m)^status=SUCCESS$') { throw "$Mode failed" }
    foreach ($pattern in $Required) {
        if ($result -notmatch $pattern) { throw "$Mode missing $pattern" }
    }
    foreach ($pattern in @(
        '(?m)^cpu_fallback=false$',
        '(?m)^nan_detected=false$',
        '(?m)^inf_detected=false$',
        '(?m)^api_trace_graph_execute_failure_count=0$'
    )) {
        if ($result -notmatch $pattern) { throw "$Mode missing $pattern" }
    }
    Write-Host "PASS $Mode"
}

if ($Scope -in @('step', 'all')) {
    Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_STEP' 'one-step' @(
        '(?m)^optimizer=MOMENTUM_SGD$',
        '(?m)^graph_count=2$',
        '(?m)^graph_execute_count=2$',
        '(?m)^major_weight_changed=true$'
    )
}
if ($Scope -in @('candidate1', 'candidates', 'all')) {
    Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_1' 'candidate-1' @(
        '(?m)^additional_convergence_condition=true$',
        '(?m)^all_seeds_loss_decreased=true$',
        '(?m)^all_seeds_accuracy_increased=true$',
        '(?m)^execute_count_per_training_step=2$'
    )
}
if ($Scope -in @('candidate2', 'candidates', 'all')) {
    Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_CANDIDATE_2' 'candidate-2' @(
        '(?m)^additional_convergence_condition=true$',
        '(?m)^all_seeds_loss_decreased=true$',
        '(?m)^all_seeds_accuracy_increased=true$',
        '(?m)^execute_count_per_training_step=2$'
    )
}
if ($Scope -in @('inference', 'all')) {
    Run-Test 'QNN_HTP_TINY_LANGUAGE_MODEL_MOMENTUM_INFERENCE' 'inference' @(
        '(?m)^exact_pattern_count=[34]$',
        '(?m)^logits_responsibility=HTP$',
        '(?m)^argmax_responsibility=CPU$'
    )
}
Write-Host 'HTP_MOMENTUM=PASS'
