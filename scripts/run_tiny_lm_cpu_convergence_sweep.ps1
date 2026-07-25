# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
param([string]$OutputDirectory=(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\tiny-lm-cpu-sweep'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$root=Split-Path -Parent $PSScriptRoot
[IO.Directory]::CreateDirectory($OutputDirectory)|Out-Null
$exe=Join-Path $OutputDirectory 'tiny-lm-convergence-probe.exe'
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I (Join-Path $root 'app\src\main\cpp') (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') (Join-Path $root 'host_tests\tiny_lm_convergence_probe.cpp') -o $exe
if($LASTEXITCODE-ne0){throw 'probe compilation failed'}
function Run-Probe([string]$Id,[double]$LearningRate,[int]$Steps,[double]$Scale,[string]$Sampling,[bool]$Diagnostics){
  $path=Join-Path $OutputDirectory "$Id.txt";$output=& $exe $Id $LearningRate $Steps $Scale $Sampling ([int]$Diagnostics);if($LASTEXITCODE-ne0){throw "probe failed: $Id"};[IO.File]::WriteAllLines($path,[string[]]$output,[Text.UTF8Encoding]::new($false));$map=@{};foreach($line in $output){if($line-match '^([A-Za-z0-9_]+)=(.*)$'){$map[$Matches[1]]=$Matches[2]}};[pscustomobject]@{id=$Id;lr=$LearningRate;steps=$Steps;scale=$Scale;sampling=$Sampling;median=[double]$map.median_loss_reduction;minimum=[double]$map.minimum_loss_reduction;accuracy75=[int]$map.accuracy_75_seed_count;allLoss=$map.all_seeds_loss_decreased;allAccuracy=$map.all_seeds_accuracy_increased;nanInf=[int]$map.nan_inf_count;deterministic=$map.deterministic_replay;report=$path}
}
$stageA=@();foreach($lr in @(0.0003,0.001,0.003,0.01,0.03,0.1)){foreach($steps in @(100,320,640,1000)){$id=('sgd-a-lr{0}-s{1}'-f($lr.ToString('G',[Globalization.CultureInfo]::InvariantCulture)),$steps);$stageA+=Run-Probe $id $lr $steps 1 fixed ($lr-eq0.01-and$steps-eq1000);Write-Host "A $id median=$($stageA[-1].median) accuracy75=$($stageA[-1].accuracy75)"}}
$rankedA=@($stageA|Sort-Object @{Expression={$_.accuracy75};Descending=$true},@{Expression={$_.allLoss};Descending=$true},@{Expression={$_.allAccuracy};Descending=$true},@{Expression={$_.median};Descending=$true})
$stageB=@();foreach($base in $rankedA[0..2]){foreach($scale in @(0.5,1.0,1.5,2.0)){foreach($sampling in @('fixed','shuffle')){$id="sgd-b-$($base.id)-i$scale-$sampling";$stageB+=Run-Probe $id $base.lr $base.steps $scale $sampling $false;Write-Host "B $id median=$($stageB[-1].median) accuracy75=$($stageB[-1].accuracy75)"}}
}
$all=@($stageA+$stageB);$ranked=@($all|Sort-Object @{Expression={($_.accuracy75-ge4)-and($_.allLoss-eq'true')-and($_.allAccuracy-eq'true')};Descending=$true},@{Expression={$_.accuracy75};Descending=$true},@{Expression={$_.allLoss};Descending=$true},@{Expression={$_.allAccuracy};Descending=$true},@{Expression={$_.median};Descending=$true},@{Expression={$_.minimum};Descending=$true})
$all|Select-Object id,lr,steps,scale,sampling,median,minimum,accuracy75,allLoss,allAccuracy,nanInf,deterministic|Export-Csv -LiteralPath (Join-Path $OutputDirectory 'configurations.csv') -NoTypeInformation -Encoding utf8
$seen=@{};$uniqueRanked=@();foreach($item in $ranked){$key="$($item.lr)|$($item.steps)|$($item.scale)|$($item.sampling)";if(!$seen.ContainsKey($key)){$seen[$key]=$true;$uniqueRanked+=$item}}
$uniqueRanked|Select-Object -First 3 id,lr,steps,scale,sampling,median,minimum,accuracy75,allLoss,allAccuracy,nanInf,deterministic|Export-Csv -LiteralPath (Join-Path $OutputDirectory 'top3.csv') -NoTypeInformation -Encoding utf8
$winner=$uniqueRanked[0];Run-Probe 'sgd-stage-c-diagnostic' $winner.lr $winner.steps $winner.scale $winner.sampling $true|Out-Null
Write-Host "SWEEP_COUNT=$($all.Count)";Write-Host "WINNER=$($winner.id) median=$($winner.median) accuracy75=$($winner.accuracy75)"
