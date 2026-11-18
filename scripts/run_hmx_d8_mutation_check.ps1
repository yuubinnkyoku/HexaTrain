# D8 minimal experiment — CPU mutation regression (CPU only).
#
# The D8 classifier and its independent Python implementation must agree on the
# fixed battery, but agreement alone does not prove the cases discriminate a
# broken classifier. Five deliberate mutations are applied to scratch copies
# of the simulator (never to tracked source); each mutant must be rejected by
# the independent differential.
#
# Mutations:
#   1. ignore an invalid access trace (a safe numeric result with a bad trace
#      would be read as "no violation");
#   2. map an unexplained numeric result under condition C to "no violation";
#   3. ignore the contract flags (missing evidence / wrong ids / wrong length);
#   4. swap the C and D fill masks (the revision-1 defect);
#   5. ignore the fill-mask contract gate.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_min_sim.c"
$Test = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_independent_test.py"
$BuildDir = Join-Path $Root "build"
$Dump = Join-Path $BuildDir "hmx_d8_reference_arrays.txt"
$python = Get-Command python -ErrorAction SilentlyContinue
$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $python -or -not $gcc) {
    Write-Host "d8_mutation_check=SKIP (python or gcc not found on PATH)"
    exit 0
}
# The independent differential needs the reference dump that run_hmx_d8_sim.ps1
# writes. Without it the audit exits 2 with "SKIP (no dump ...)", which is an
# environment condition, not a surviving mutant. Reporting that as
# "FAIL mutants=5" once sent a whole investigation down the wrong path, so the
# two outcomes are now distinct labels and a missing dump stops the script
# before any mutant is built.
if (-not (Test-Path -LiteralPath $Dump -PathType Leaf)) {
    Write-Host "d8_mutation_check=BLOCKED (no reference dump at $Dump)"
    Write-Host "d8_mutation_check_hint=run scripts/run_hmx_d8_sim.ps1 first"
    exit 2
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
# The simulator writes its dump relative to the process working directory, so
# pin it to the repository root for the whole run (see run_hmx_d8_sim.ps1).
$pushedLocation = $false
Push-Location $Root
$pushedLocation = $true
# Normalise line endings so the multi-line mutation needle matches regardless of
# the checkout's autocrlf setting.
$sourceText = (Get-Content -LiteralPath $Source -Raw).Replace("`r`n", "`n")

$mutations = @(
    @{
        Name = "trace_override_removed"
        Needle = "if (trace_status == D8_TRACE_INVALID) return D8_CLASS_INCONCLUSIVE_TRACE;"
        Replacement = "(void)trace_status; if (0) return D8_CLASS_INCONCLUSIVE_TRACE; /* mutant */"
        Expect = "trace_invalid"
    },
    @{
        Name = "numeric_unknown_is_safe"
        Needle = "if (matched == 1) return D8_CLASS_VIOLATION_WLO;`n      return D8_CLASS_INCONCLUSIVE_NUMERIC;"
        Replacement = "if (matched == 1) return D8_CLASS_VIOLATION_WLO;`n      return D8_CLASS_NO_VIOLATION; /* mutant */"
        Expect = "mutated_safe"
    },
    @{
        Name = "contract_checks_removed"
        Needle = "if (!evidence_ok || !ids_ok || !lengths_ok) return D8_CLASS_CONTRACT_FAILED;"
        Replacement = "(void)evidence_ok; (void)ids_ok; (void)lengths_ok; if (0) return D8_CLASS_CONTRACT_FAILED; /* mutant */"
        Expect = "no_evidence"
    },
    @{
        Name = "mask_cd_swapped"
        Needle = "    case D8_COND_C: return D8_MASK_OUT | D8_MASK_WLO;`n    case D8_COND_D: return D8_MASK_OUT | D8_MASK_AHI;"
        Replacement = "    case D8_COND_C: return D8_MASK_OUT | D8_MASK_AHI;`n    case D8_COND_D: return D8_MASK_OUT | D8_MASK_WLO; /* mutant */"
        Expect = "mask_c_uses_d"
    },
    @{
        Name = "mask_gate_removed"
        Needle = "if (fill_mask != d8_condition_mask(cond)) return D8_CLASS_CONTRACT_FAILED;"
        Replacement = "(void)fill_mask; if (0) return D8_CLASS_CONTRACT_FAILED; /* mutant */"
        Expect = "mask_c_uses_d"
    }
)

$failed = 0
$blocked = 0
foreach ($m in $mutations) {
    $mutantText = $sourceText.Replace($m.Needle, $m.Replacement)
    if ($mutantText -eq $sourceText) {
        Write-Host "d8_mutation=$($m.Name) FAIL (mutation pattern not found)"
        $failed++
        continue
    }
    $mutantSource = Join-Path $BuildDir ("hmx_d8_mutant_" + $m.Name + ".c")
    $mutantExe = Join-Path $BuildDir ("hmx_d8_mutant_" + $m.Name + ".exe")
    Set-Content -LiteralPath $mutantSource -Value $mutantText -NoNewline -Encoding utf8
    & $gcc.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
        "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
        "-o" $mutantExe $mutantSource "-lm"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "d8_mutation=$($m.Name) FAIL (compile exit $LASTEXITCODE)"
        $failed++
        continue
    }
    $out = & $python.Source $Test ("--replay-exe=" + $mutantExe) 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    # exit 2 from the differential means it could not run (missing dump, missing
    # binary). That is an environment condition, not a surviving mutant, so it
    # must not be counted as a detection failure.
    if ($exitCode -eq 2) {
        Write-Host "d8_mutation=$($m.Name) BLOCKED (audit could not run, exit=2)"
        $blocked++
        continue
    }
    $detected = ($exitCode -eq 1) -and ($out -match "MISMATCH")
    if ($detected) {
        Write-Host "d8_mutation=$($m.Name) DETECTED exit=$exitCode"
    } else {
        Write-Host "d8_mutation=$($m.Name) FAIL (exit=$exitCode no mismatch)"
        $failed++
    }
}
if ($blocked -gt 0) {
    Write-Host "d8_mutation_check=BLOCKED (audit could not run for $blocked of $($mutations.Count) mutants)"
    if ($pushedLocation) { Pop-Location }
    exit 2
}
if ($failed -eq 0) {
    Write-Host "d8_mutation_check=PASS mutants=$($mutations.Count)"
    if ($pushedLocation) { Pop-Location }
    exit 0
}
Write-Host "d8_mutation_check=FAIL failed=$failed"
if ($pushedLocation) { Pop-Location }
exit 1
