# Compatibility orchestrator for host tests.
# Runs the product/contract suite and the diagnostic/research suite in order.
# Historical callers (verify_local.ps1, CI, docs) keep using this entrypoint;
# coverage is intentionally not reduced.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "host_test_common.ps1")

$contractScript = Join-Path $Root "scripts\run_host_contract_tests.ps1"
$diagnosticScript = Join-Path $Root "scripts\run_host_diagnostic_tests.ps1"

# One invocation-unique object session, shared by contract and diagnostic child
# processes. The ownership token prevents arbitrary stale directories from being
# joined, and finally cleanup handles success and ordinary failure paths.
$ObjectSessionToken = [Guid]::NewGuid().ToString("N")
$ObjectSessionDir = Join-Path $Root (
    "build\host-test-objects\wrapper-{0}-{1}" -f $PID, $ObjectSessionToken)
Initialize-PhoneLmHostObjectSession -SessionDirectory $ObjectSessionDir -Fresh -SessionToken $ObjectSessionToken

Write-Host "===== run_host_tests.ps1 (contract + diagnostic) ====="
try {
    Invoke-PhoneLmHostPwshScript -Label "run_host_contract_tests.ps1" -ScriptPath $contractScript -Arguments @(
        "-ObjectSessionDir", $ObjectSessionDir,
        "-ObjectSessionToken", $ObjectSessionToken)
    Invoke-PhoneLmHostPwshScript -Label "run_host_diagnostic_tests.ps1" -ScriptPath $diagnosticScript -Arguments @(
        "-ObjectSessionDir", $ObjectSessionDir,
        "-ObjectSessionToken", $ObjectSessionToken)
    Write-Host "run_host_tests=PASS (contract+diagnostic)"
} finally {
    $script:PhoneLmHostObjectSession = $null
    if (Test-Path -LiteralPath $ObjectSessionDir) {
        Remove-Item -LiteralPath $ObjectSessionDir -Recurse -Force
    }
}
