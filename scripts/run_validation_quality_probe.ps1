# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$output = Join-Path $root 'build\host-tests'
[IO.Directory]::CreateDirectory($output) | Out-Null
$exe = Join-Path $output 'validation_quality_probe.exe'
& g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $root 'app\src\main\cpp') `
    -I (Join-Path $root 'host_tests') `
    (Join-Path $root 'app\src\main\cpp\tiny_language_model_cpu.cpp') `
    (Join-Path $root 'host_tests\validation_quality_probe.cpp') `
    -o $exe
if ($LASTEXITCODE -ne 0) { throw 'validation quality probe compilation failed' }
$raw = (& $exe)
if ($LASTEXITCODE -ne 0) { throw 'validation quality probe failed' }
$report = Join-Path $root 'build\reports\qnn-validation-selected'
[IO.Directory]::CreateDirectory($report) | Out-Null
$summary = @('layers,seed,finite,selected_step,best_validation_loss,best_validation_accuracy,final_validation_loss,final_validation_accuracy,selected_oracle_exact,selected_free_exact,final_oracle_exact,final_free_exact,selected_phase1_loss,selected_phase1_accuracy,final_phase1_loss,final_phase1_accuracy')
$trajectory = @('layers,seed,step,training_loss,training_accuracy,validation_loss,validation_accuracy,validation_target_margin,validation_target_probability,parameter_norm,gradient_norm,update_parameter_ratio,posthoc_oracle_exact,posthoc_free_exact')
$early = @('layers,seed,patience,simulated_stop_step,best_step,saved_training_steps')
foreach ($line in $raw) {
    if ($line.StartsWith('SUMMARY,')) { $summary += $line.Substring(8) }
    elseif ($line.StartsWith('TRAJECTORY,')) { $trajectory += $line.Substring(11) }
    elseif ($line.StartsWith('EARLY_STOP,')) { $early += $line.Substring(11) }
    else { throw "unknown probe record: $line" }
}
[IO.File]::WriteAllLines((Join-Path $report 'cpu-smoke.csv'), $summary)
[IO.File]::WriteAllLines((Join-Path $report 'validation-trajectories.csv'), $trajectory)
[IO.File]::WriteAllLines((Join-Path $report 'early-stop-simulation.csv'), $early)
Write-Host "validation quality probe PASS: $report"
