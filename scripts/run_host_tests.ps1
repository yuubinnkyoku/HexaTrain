# Compatibility orchestrator for host tests.
# Runs the product/contract suite and the diagnostic/research suite in order.
# Historical callers (verify_local.ps1, CI, docs) keep using this entrypoint;
# coverage is intentionally not reduced.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "host_test_common.ps1")

$contractScript = Join-Path $Root "scripts\run_host_contract_tests.ps1"
$diagnosticScript = Join-Path $Root "scripts\run_host_diagnostic_tests.ps1"

# One fresh object session for this wrapper invocation, shared by contract and
# diagnostic child processes. Cleaned here so a previous failed run cannot donate
# objects. Not persistent across verify runs or checkouts.
$ObjectSessionDir = Join-Path $Root "build\host-test-objects\wrapper-session"
Initialize-PhoneLmHostObjectSession -SessionDirectory $ObjectSessionDir -Fresh

Write-Host "===== run_host_tests.ps1 (contract + diagnostic) ====="
Invoke-PhoneLmHostPwshScript -Label "run_host_contract_tests.ps1" `
    -ScriptPath $contractScript `
    -Arguments @("-ObjectSessionDir", $ObjectSessionDir)
Invoke-PhoneLmHostPwshScript -Label "run_host_diagnostic_tests.ps1" `
    -ScriptPath $diagnosticScript `
    -Arguments @("-ObjectSessionDir", $ObjectSessionDir)
Write-Host "run_host_tests=PASS (contract+diagnostic)"
