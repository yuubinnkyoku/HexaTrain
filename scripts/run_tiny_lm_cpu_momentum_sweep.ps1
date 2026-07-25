# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param([string]$OutputDirectory=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-cpu-momentum-sweep'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot;[IO.Directory]::CreateDirectory($OutputDirectory)|Out-Null
$exe=Join-Path $OutputDirectory 'tiny-lm-convergence-probe.exe'
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I (Join-Path $root 'app\src\main\cpp') (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') (Join-Path $root 'host_tests\tiny_lm_convergence_probe.cpp') -o $exe
if($LASTEXITCODE-ne0){throw 'probe compilation failed'}
function Run-Probe([string]$Id,[double]$LearningRate,[int]$Steps,[double]$Momentum,[bool]$Diagnostics){$path=Join-Path $OutputDirectory "$Id.txt";$output=& $exe $Id $LearningRate $Steps 1 fixed ([int]$Diagnostics) $Momentum;$probeExitCode=$LASTEXITCODE;[IO.File]::WriteAllLines($path,[string[]]$output,[Text.UTF8Encoding]::new($false));$map=@{};foreach($line in $output){if($line-match'^([A-Za-z0-9_]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}};[pscustomobject]@{id=$Id;lr=$LearningRate;steps=$Steps;momentum=$Momentum;median=[double]$map.median_loss_reduction;minimum=[double]$map.minimum_loss_reduction;accuracy75=[int]$map.accuracy_75_seed_count;allLoss=$map.all_seeds_loss_decreased;allAccuracy=$map.all_seeds_accuracy_increased;nanInf=[int]$map.nan_inf_count;deterministic=$map.deterministic_replay}}
$rows=@();foreach($lr in @(0.001,0.003,0.01,0.03)){foreach($momentum in @(0.9,0.95)){foreach($steps in @(320,640,1000)){$id="momentum-lr$lr-m$momentum-s$steps";$rows+=Run-Probe $id $lr $steps $momentum $false;Write-Host "$id median=$($rows[-1].median) accuracy75=$($rows[-1].accuracy75)"}}}
$ranked=@($rows|Sort-Object @{Expression={($_.accuracy75-ge4)-and($_.allLoss-eq'true')-and($_.allAccuracy-eq'true')};Descending=$true},@{Expression={$_.accuracy75};Descending=$true},@{Expression={$_.median};Descending=$true},@{Expression={$_.minimum};Descending=$true})
$rows|Export-Csv (Join-Path $OutputDirectory 'configurations.csv') -NoTypeInformation -Encoding utf8;$ranked|Select-Object -First 2|Export-Csv (Join-Path $OutputDirectory 'top2.csv') -NoTypeInformation -Encoding utf8
$winner=$ranked[0];Run-Probe 'momentum-diagnostic' $winner.lr $winner.steps $winner.momentum $true|Out-Null
Write-Host "MOMENTUM_SWEEP_COUNT=$($rows.Count)";Write-Host "WINNER=$($winner.id) median=$($winner.median) accuracy75=$($winner.accuracy75)"
