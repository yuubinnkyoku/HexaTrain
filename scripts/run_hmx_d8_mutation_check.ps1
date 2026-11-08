# D8 minimal experiment — CPU mutation regression (CPU only).
#
# The D8 classifier and its independent Python implementation must agree on the
# fixed battery, but agreement alone does not prove the cases discriminate a
# broken classifier. Three deliberate mutations are applied to scratch copies
# of the simulator (never to tracked source); each mutant must be rejected by
# the independent differential.
#
# Mutations:
#   1. ignore an invalid access trace (a safe numeric result with a bad trace
#      would be read as "no violation");
#   2. map an unexplained numeric result under condition C to "no violation";
#   3. ignore the contract flags (missing evidence / wrong ids / wrong length).
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
$python = Get-Command python -ErrorAction SilentlyContinue
$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $python -or -not $gcc) {
    Write-Host "d8_mutation_check=SKIP (python or gcc not found on PATH)"
    exit 0
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
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
    }
)

$failed = 0
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
    $detected = ($exitCode -eq 1) -and ($out -match "MISMATCH")
    if ($detected) {
        Write-Host "d8_mutation=$($m.Name) DETECTED exit=$exitCode"
    } else {
        Write-Host "d8_mutation=$($m.Name) FAIL (exit=$exitCode no mismatch)"
        $failed++
    }
}

if ($failed -eq 0) {
    Write-Host "d8_mutation_check=PASS mutants=$($mutations.Count)"
    exit 0
}
Write-Host "d8_mutation_check=FAIL failed=$failed"
exit 1
