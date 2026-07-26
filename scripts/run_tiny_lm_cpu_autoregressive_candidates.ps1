# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param([string]$OutputDirectory=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-cpu-autoregressive-candidates'))
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot
[IO.Directory]::CreateDirectory($OutputDirectory)|Out-Null
$exe=Join-Path $OutputDirectory 'tiny-lm-autoregressive-probe.exe'
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I (Join-Path $root 'app\src\main\cpp') (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') (Join-Path $root 'host_tests\tiny_lm_autoregressive_probe.cpp') -o $exe
if($LASTEXITCODE-ne0){throw 'autoregressive probe compilation failed'}
function Run-Probe([string]$Mode,[int]$SeedCount,[string]$Tag){
  $path=Join-Path $OutputDirectory "$Tag.txt"
  $output=& $exe $Mode 1 $SeedCount 1000
  if($LASTEXITCODE-ne0){throw "probe failed: $Mode"}
  [IO.File]::WriteAllLines($path,[string[]]$output,[Text.UTF8Encoding]::new($false))
  $map=@{};foreach($line in $output){if($line-match'^([A-Za-z0-9_]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}}
  [pscustomobject]@{
    mode=$Mode;seeds=$SeedCount;raw_report=$path
    loss=[double]$map.summary_mean_teacher_forced_loss
    accuracy=[double]$map.summary_mean_teacher_forced_accuracy
    margin=[double]$map.summary_mean_teacher_forced_margin
    all_finite=$map.summary_all_finite;deterministic=$map.deterministic_replay
  }
}
$screen=@(
  Run-Probe 'phase0_round_robin' 3 'baseline-phase0-3seeds'
  Run-Probe 'phase01_round_robin' 3 'phase01-3seeds'
  Run-Probe 'all_phases_round_robin' 3 'all-phases-3seeds'
)
$screen|Export-Csv (Join-Path $OutputDirectory 'screening-3seeds.csv') -NoTypeInformation -Encoding utf8
$top=@($screen|Where-Object {$_.mode-ne'phase0_round_robin'}|Sort-Object @{Expression='accuracy';Descending=$true},@{Expression='margin';Descending=$true},@{Expression='loss';Descending=$false}|Select-Object -First 2)
$confirmed=@();foreach($candidate in $top){$confirmed+=Run-Probe $candidate.mode 5 ("{0}-5seeds" -f $candidate.mode)}
$confirmed|Export-Csv (Join-Path $OutputDirectory 'top-candidates-5seeds.csv') -NoTypeInformation -Encoding utf8
foreach($result in $screen+$confirmed){Write-Host "MODE=$($result.mode) SEEDS=$($result.seeds) ACCURACY=$($result.accuracy) MARGIN=$($result.margin) LOSS=$($result.loss) FINITE=$($result.all_finite) DETERMINISTIC=$($result.deterministic)"}
Write-Host "RAW_REPORT_DIRECTORY=$OutputDirectory"
