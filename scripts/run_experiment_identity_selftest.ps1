# CPU-only self-test for host_tests/hmx_acc_residual_rpc/hmx_provenance_gate.ps1
# (the fail-closed build-id identity gate used by run_experiment.ps1
# -DeployAndRun). No SDK, no adb, no device: pure PowerShell over a fixed
# case matrix. Reports a single summary line, consistent with
# scripts/run_hmx_selftest.ps1:
#   experiment_identity_selftest=PASS cases=N
#   experiment_identity_selftest=FAIL details=...
# Exit 0 on pass, 1 on any case failure.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Gate = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_provenance_gate.ps1"
if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) {
    Write-Host "experiment_identity_selftest=FAIL details=(missing gate library $Gate)"
    exit 1
}
. $Gate

$cases = New-Object System.Collections.Generic.List[object]

function Add-Case([string]$Name, [AllowEmptyCollection()][string[]]$Lines,
                  [bool]$ExpectGood, [string]$ExpectReason) {
    $cases.Add([pscustomobject]@{
        Name        = $Name
        Lines       = $Lines
        ExpectGood  = $ExpectGood
        ExpectReason = $ExpectReason
    })
}

# a. MATCH with nonzero equal ids, with and without the "  [device] " prefix.
Add-Case "match_equal" `
    @('build_id=0x19cc4c4f dsp_build_id=0x19cc4c4f build_id_status=MATCH') `
    $true 'match'
Add-Case "match_equal_prefixed" `
    @('  [device] build_id=0x19cc4c4f dsp_build_id=0x19cc4c4f build_id_status=MATCH') `
    $true 'match'
# b. UNAVAILABLE line (probe exit 0 hole this gate closes).
Add-Case "unavailable_line" `
    @('BUILD_ID=UNAVAILABLE dsp_build_id=0x00000000 expected_build_id=0x19cc4c4f') `
    $false 'unavailable_or_zero'
# c. status=MATCH but dsp_build_id=0.
Add-Case "match_dsp_zero" `
    @('build_id=0x19cc4c4f dsp_build_id=0x00000000 build_id_status=MATCH') `
    $false 'unavailable_or_zero'
# d. zero id on either side with nonzero on the other.
Add-Case "match_host_zero" `
    @('build_id=0x00000000 dsp_build_id=0x539bf682 build_id_status=MATCH') `
    $false 'unavailable_or_zero'
Add-Case "match_both_zero" `
    @('build_id=0x00000000 dsp_build_id=0x00000000 build_id_status=MATCH') `
    $false 'unavailable_or_zero'
# e. real MISMATCH line (observed shape).
Add-Case "status_mismatch" `
    @('build_id=0x19cc4c4f dsp_build_id=0x7dfcb259 build_id_status=MISMATCH') `
    $false 'mismatch'
# f. unknown status token fails closed.
Add-Case "status_stale" `
    @('build_id=0x19cc4c4f dsp_build_id=0x19cc4c4f build_id_status=STALE') `
    $false 'status_STALE'
# g. empty output / no identity line.
Add-Case "empty_output" @() $false 'missing'
Add-Case "no_identity_line" `
    @('identity_status=0 dsp_revision=1', 'dsp_identity=MATCH') `
    $false 'missing'
# h. two identity lines (retry duplicates): first line wins.
Add-Case "first_line_wins" `
    @(
        'build_id=0x19cc4c4f dsp_build_id=0x19cc4c4f build_id_status=MATCH',
        'build_id=0x19cc4c4f dsp_build_id=0x7dfcb259 build_id_status=MISMATCH'
    ) `
    $true 'match'
Add-Case "first_line_wins_mismatch_first" `
    @(
        'build_id=0x19cc4c4f dsp_build_id=0x7dfcb259 build_id_status=MISMATCH',
        'build_id=0x19cc4c4f dsp_build_id=0x19cc4c4f build_id_status=MATCH'
    ) `
    $false 'mismatch'
# i. hex case robustness: uppercase hex still passes (status token stays strict).
Add-Case "uppercase_hex" `
    @('build_id=0xA7037C38 dsp_build_id=0xA7037C38 build_id_status=MATCH') `
    $true 'match'
# Unequal nonzero ids.
Add-Case "id_mismatch" `
    @('build_id=0x19cc4c4f dsp_build_id=0x7dfcb259 build_id_status=MATCH') `
    $false 'id_mismatch'

$failures = New-Object System.Collections.Generic.List[string]
foreach ($case in $cases) {
    $gate = Test-HmxBuildIdGate -Lines $case.Lines
    if ($gate.Good -ne $case.ExpectGood) {
        $failures.Add("$($case.Name): Good=$($gate.Good) expected=$($case.ExpectGood) (reason=$($gate.Reason))")
        continue
    }
    if ($gate.Reason -cne $case.ExpectReason) {
        $failures.Add("$($case.Name): Reason=$($gate.Reason) expected=$($case.ExpectReason)")
    }
}

if ($failures.Count -eq 0) {
    Write-Host "experiment_identity_selftest=PASS cases=$($cases.Count)"
    exit 0
}
Write-Host "experiment_identity_selftest=FAIL details=$($failures -join '; ')"
exit 1
