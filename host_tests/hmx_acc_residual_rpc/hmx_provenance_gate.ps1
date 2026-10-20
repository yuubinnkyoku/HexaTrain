# Pure build-id provenance gate for the HMX residual FastRPC probe.
#
# Dot-source this file, then call Test-HmxBuildIdGate with the captured
# device-run output lines. It is a pure function: no global state mutation,
# no Write-Host, no side effects. PowerShell 5/7-safe (no ternary).
#
# The device harness prints one of:
#   build_id=0x<8hex> dsp_build_id=0x<8hex> build_id_status=MATCH
#   BUILD_ID=UNAVAILABLE dsp_build_id=0x<8hex> expected_build_id=0x<8hex>
# A real mismatch fails with exit 9; a dsp build id of 0 is reported as
# UNAVAILABLE and does NOT fail identity inside the device harness. This
# gate closes the orchestrator-level hole: a probe exit 0 with UNAVAILABLE
# must no longer count as a successful provenance run.
function Test-HmxBuildIdGate {
  param(
    [AllowEmptyCollection()]
    [string[]]$Lines
  )
  # Normal shape: build_id=0x<hex8> dsp_build_id=0x<hex8> build_id_status=<TOKEN>.
  # Hex digits are case-insensitive (uppercase 0xA7037C38 must still pass);
  # the status token is matched case-sensitively later (only "MATCH" passes).
  # Tolerate a leading two-space + "[device]" prefix and surrounding
  # whitespace anywhere in the line.
  $normal = '(?m)^\s*(?:\s{2}\[device\]\s+)?build_id=0x([0-9a-fA-F]{8})\s+dsp_build_id=0x([0-9a-fA-F]{8})\s+build_id_status=(\S+)\s*$'
  # UNAVAILABLE shape (hex digits likewise case-insensitive).
  $unavailable = '(?m)^\s*(?:\s{2}\[device\]\s+)?BUILD_ID=UNAVAILABLE\s+dsp_build_id=0x([0-9a-fA-F]{8})\s+expected_build_id=0x([0-9a-fA-F]{8})\s*$'
  $result = [pscustomobject]@{
    Good         = $false
    Reason       = 'missing'
    MatchedLine  = $null
    HostBuildId  = $null
    DspBuildId   = $null
  }
  $firstNormal = $null
  $firstUnavail = $null
  foreach ($line in $Lines) {
    if ($null -eq $line) { continue }
    if (($null -eq $firstNormal) -and ($line -cmatch $normal)) {
      $firstNormal = [pscustomobject]@{
        Line = $line
        Host = [Convert]::ToUInt32($Matches[1], 16)
        Dsp  = [Convert]::ToUInt32($Matches[2], 16)
        Token = $Matches[3]
      }
    }
    if (($null -eq $firstUnavail) -and ($line -cmatch $unavailable)) {
      $firstUnavail = [pscustomobject]@{
        Line = $line
        Dsp     = [Convert]::ToUInt32($Matches[1], 16)
        Expected = [Convert]::ToUInt32($Matches[2], 16)
      }
    }
  }
  if ($null -eq $firstNormal) {
    if ($null -ne $firstUnavail) {
      $result.Reason = 'unavailable_or_zero'
      $result.MatchedLine = $firstUnavail.Line
      $result.HostBuildId = $firstUnavail.Expected
      $result.DspBuildId = $firstUnavail.Dsp
    }
    return $result
  }
  $result.MatchedLine = $firstNormal.Line
  $result.HostBuildId = $firstNormal.Host
  $result.DspBuildId = $firstNormal.Dsp
  if (($firstNormal.Host -eq 0) -or ($firstNormal.Dsp -eq 0)) {
    $result.Reason = 'unavailable_or_zero'
    return $result
  }
  if ($firstNormal.Token -cne 'MATCH') {
    if ($firstNormal.Token -ceq 'MISMATCH') {
      $result.Reason = 'mismatch'
    } else {
      $result.Reason = "status_$($firstNormal.Token)"
    }
    return $result
  }
  if ($firstNormal.Host -ne $firstNormal.Dsp) {
    $result.Reason = 'id_mismatch'
    return $result
  }
  $result.Good = $true
  $result.Reason = 'match'
  return $result
}
