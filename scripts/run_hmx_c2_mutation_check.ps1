# C2 mutation regression (CPU only).
#
# Purpose: prove that the C/Python differential's new cases are discriminating,
# not merely agreeing. Two textual mutations are applied to a scratch copy of
# the simulator (never to the tracked source), each mutant is compiled, and the
# differential is run against it with --replay-exe. A mutation that the
# differential does not detect is a failure of this script.
#
# Mutations:
#   1. drop the payload-loss counter in c2_observe (row 6 of pre-registration
#      section 6). Only `synthetic_c1_plus_zero_loss` can catch it, because a
#      plain zero-loss state falls through to the same INCONCLUSIVE class.
#   2. remove the mixed-signature test in c2_classify. The mixed array case and
#      the exhaustive counter sweep must catch it.
#
# No device, DSP, SDK or network access. Outputs stay under build/.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_b3_r2_full_tile_sim.c"
$Differential = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_c2_differential_test.py"
$BuildDir = Join-Path $Root "build"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "c2_mutation_check=SKIP (python not found on PATH)"
    exit 0
}
$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gcc) {
    Write-Host "c2_mutation_check=SKIP (gcc not found on PATH)"
    exit 0
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$sourceText = Get-Content -LiteralPath $Source -Raw

$mutations = @(
    @{
        Name = "zero_loss_row_removed"
        Needle = "c->zero_loss_other++;"
        Replacement = "/* mutant: payload-loss row not counted */;"
        Expect = "synthetic_c1_plus_zero_loss"
    },
    @{
        Name = "mixed_test_removed"
        Needle = "if (c->c1_cells > 0 && c->c2_nonzero_cells > 0) {"
        Replacement = "if (0 && c->c1_cells > 0 && c->c2_nonzero_cells > 0) { /* mutant */"
        Expect = "mixed_signature"
    }
)

$failed = 0
foreach ($m in $mutations) {
    $mutantText = $sourceText.Replace($m.Needle, $m.Replacement)
    if ($mutantText -eq $sourceText) {
        Write-Host "c2_mutation=$($m.Name) FAIL (mutation pattern not found)"
        $failed++
        continue
    }
    $mutantSource = Join-Path $BuildDir ("hmx_c2_mutant_" + $m.Name + ".c")
    $mutantExe = Join-Path $BuildDir ("hmx_c2_mutant_" + $m.Name + ".exe")
    Set-Content -LiteralPath $mutantSource -Value $mutantText -NoNewline -Encoding utf8
    & $gcc.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
        "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
        "-o" $mutantExe $mutantSource "-lm"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "c2_mutation=$($m.Name) FAIL (compile exit $LASTEXITCODE)"
        $failed++
        continue
    }
    Get-ChildItem -LiteralPath $BuildDir -Filter "c2_differential_mismatch_provided_*.json" `
        -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    $out = & $python.Source $Differential ("--replay-exe=" + $mutantExe) 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    Get-ChildItem -LiteralPath $BuildDir -Filter "c2_differential_mismatch_provided_*.json" `
        -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    $detected = ($exitCode -eq 1) -and ($out -match [regex]::Escape($m.Expect))
    if ($detected) {
        Write-Host "c2_mutation=$($m.Name) DETECTED case_in_output=$($m.Expect)"
    } else {
        Write-Host "c2_mutation=$($m.Name) FAIL (exit=$exitCode expected_case=$($m.Expect))"
        $failed++
    }
}

if ($failed -eq 0) {
    Write-Host "c2_mutation_check=PASS mutants=$($mutations.Count)"
    exit 0
}
Write-Host "c2_mutation_check=FAIL failed=$failed"
exit 1
