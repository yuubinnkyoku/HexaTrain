# Compatibility orchestrator for host tests.
# Runs the product/contract suite and the diagnostic/research suite in order.
# Historical callers (verify_local.ps1, CI, docs) keep using this entrypoint;
# coverage is intentionally not reduced.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "host_test_common.ps1")

$contractScript = Join-Path $Root "scripts\run_host_contract_tests.ps1"
$diagnosticScript = Join-Path $Root "scripts\run_host_diagnostic_tests.ps1"

Write-Host "===== run_host_tests.ps1 (contract + diagnostic) ====="
Invoke-PhoneLmHostPwshScript -Label "run_host_contract_tests.ps1" -ScriptPath $contractScript
Invoke-PhoneLmHostPwshScript -Label "run_host_diagnostic_tests.ps1" -ScriptPath $diagnosticScript
Write-Host "run_host_tests=PASS (contract+diagnostic)"
