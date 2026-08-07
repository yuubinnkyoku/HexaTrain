# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
# Shared lossless byte-level aggregate metrics for the Nicopedia HTP
# generation milestone.  Dot-sourced by run_nicopedia_htp_generate.ps1 and
# export_public_qnn_nicopedia_htp_1000step_results.ps1 so the host-side
# mirrors of the C++ GenerationAggregates stay single-sourced.
#
# All metrics here are aggregate/lossless: byte counts, UTF-8 validity,
# ASCII share, longest same-byte run, longest repeated-scalar run, and a
# short-period (1..3) loop fraction measured on the trailing 32 bytes.
# No generated content is retained by these helpers.

function Get-GenerationAggregates([byte[]]$Bytes) {
    $result = [ordered]@{
        total_bytes = $Bytes.Length
        valid_utf8_bytes = 0
        invalid_utf8_bytes = 0
        valid_scalars = 0
        ascii_bytes = 0
        unique_byte_values = 0
        max_same_byte_run = 0
        max_scalar_repeat_run = 0
        short_period_loop_fraction = 0.0
    }
    if ($Bytes.Length -eq 0) { return $result }
    $seen = New-Object bool[] 256
    $sameRun = 0
    for ($i = 0; $i -lt $Bytes.Length; $i++) {
        $b = $Bytes[$i]
        if (-not $seen[$b]) { $seen[$b] = $true; $result.unique_byte_values++ }
        if ($b -lt 0x80) { $result.ascii_bytes++ }
        if ($i -gt 0 -and $b -eq $Bytes[$i - 1]) { $sameRun++ } else { $sameRun = 1 }
        if ($sameRun -gt $result.max_same_byte_run) { $result.max_same_byte_run = $sameRun }
    }
    # UTF-8 decoding pass (same rules as nicopedia_generation.h decodeUtf8).
    function Test-Continuation([byte]$c) { return (($c -band 0xC0) -eq 0x80) }
    function Get-LeadLength([byte]$b) {
        if ($b -lt 0x80) { return 1 }
        if ($b -ge 0xC2 -and $b -le 0xDF) { return 2 }
        if ($b -ge 0xE0 -and $b -le 0xEF) { return 3 }
        if ($b -ge 0xF0 -and $b -le 0xF4) { return 4 }
        return 0
    }
    function Test-ScalarEqual([byte[]]$A, [byte[]]$B) {
        if ($A.Length -ne $B.Length) { return $false }
        for ($k = 0; $k -lt $A.Length; $k++) { if ($A[$k] -ne $B[$k]) { return $false } }
        return $true
    }
    $pos = 0
    $prevScalar = [byte[]]@()
    $havePrevScalar = $false
    $scalarRepetitions = 0
    while ($pos -lt $Bytes.Length) {
        $lead = Get-LeadLength $Bytes[$pos]
        $consumed = 0
        if ($lead -eq 1) {
            $consumed = 1
        } elseif ($lead -gt 0 -and $pos + $lead -le $Bytes.Length) {
            $ok = $true
            for ($k = 1; $k -lt $lead; $k++) {
                if (-not (Test-Continuation $Bytes[$pos + $k])) { $ok = $false; break }
            }
            if ($ok -and $lead -eq 3 -and $Bytes[$pos] -eq 0xE0 -and $Bytes[$pos + 1] -lt 0xA0) { $ok = $false }
            if ($ok -and $lead -eq 3 -and $Bytes[$pos] -eq 0xED -and $Bytes[$pos + 1] -ge 0xA0) { $ok = $false }
            if ($ok -and $lead -eq 4 -and $Bytes[$pos] -eq 0xF0 -and $Bytes[$pos + 1] -lt 0x90) { $ok = $false }
            if ($ok -and $lead -eq 4 -and $Bytes[$pos] -eq 0xF4 -and $Bytes[$pos + 1] -ge 0x90) { $ok = $false }
            if ($ok) { $consumed = $lead }
        }
        if ($consumed -gt 0) {
            $result.valid_utf8_bytes += $consumed
            $result.valid_scalars++
            $scalar = $Bytes[$pos..($pos + $consumed - 1)]
            if ($havePrevScalar -and (Test-ScalarEqual $scalar $prevScalar)) {
                $scalarRepetitions++
                $run = $scalarRepetitions + 1
                if ($run -gt $result.max_scalar_repeat_run) { $result.max_scalar_repeat_run = $run }
            } else {
                $scalarRepetitions = 0
            }
            $prevScalar = $scalar
            $havePrevScalar = $true
            $pos += $consumed
        } else {
            $result.invalid_utf8_bytes++
            $scalar = [byte[]]@($Bytes[$pos])
            if ($havePrevScalar -and (Test-ScalarEqual $scalar $prevScalar)) {
                $scalarRepetitions++
                $run = $scalarRepetitions + 1
                if ($run -gt $result.max_scalar_repeat_run) { $result.max_scalar_repeat_run = $run }
            } else {
                $scalarRepetitions = 0
            }
            $prevScalar = $scalar
            $havePrevScalar = $true
            $pos += 1
        }
    }
    # Short-period loop detection on the trailing 32 bytes (periods 1..3).
    $tailStart = if ($Bytes.Length -gt 32) { $Bytes.Length - 32 } else { 0 }
    $bestFraction = 0.0
    for ($period = 1; $period -le 3; $period++) {
        if ($Bytes.Length - $tailStart - $period -gt 0) {
            $matches = 0
            $compared = 0
            for ($i = $tailStart; $i + $period -lt $Bytes.Length; $i++) {
                $compared++
                if ($Bytes[$i] -eq $Bytes[$i + $period]) { $matches++ }
            }
            $fraction = if ($compared -gt 0) { $matches / $compared } else { 0.0 }
            if ($fraction -gt $bestFraction) { $bestFraction = $fraction }
        }
    }
    $result.short_period_loop_fraction = $bestFraction
    return $result
}

function Test-GenerationAggregateSelfTest {
    $cases = @(
        @{ Bytes = [byte[]](0x61, 0x62, 0x62, 0x62, 0x63);
           Expect = @{ unique = 3; ascii = 5; sameRun = 3; scalar = 3; valid = 5; invalid = 0; scalars = 5; loop = 0.5 } },
        @{ Bytes = [byte[]](0xE3, 0x81, 0xAE, 0xE3, 0x81, 0xAE, 0xE3, 0x81, 0xAE);
           Expect = @{ unique = 3; ascii = 0; sameRun = 1; scalar = 3; valid = 9; invalid = 0; scalars = 3; loop = 1.0 } },
        @{ Bytes = [byte[]](0xE3, 0x81, 0x61);
           Expect = @{ unique = 3; ascii = 1; sameRun = 1; scalar = 0; valid = 1; invalid = 2; scalars = 1; loop = 0.0 } }
    )
    foreach ($case in $cases) {
        $ag = Get-GenerationAggregates -Bytes $case.Bytes
        if ($ag.unique_byte_values -ne $case.Expect.unique) { throw "SELFTEST_AGG_UNIQUE: $($ag.unique_byte_values)" }
        if ($ag.ascii_bytes -ne $case.Expect.ascii) { throw 'SELFTEST_AGG_ASCII' }
        if ($ag.max_same_byte_run -ne $case.Expect.sameRun) { throw "SELFTEST_AGG_SAMERUN: $($ag.max_same_byte_run)" }
        if ($ag.max_scalar_repeat_run -ne $case.Expect.scalar) { throw "SELFTEST_AGG_SCALAR: $($ag.max_scalar_repeat_run)" }
        if ($ag.valid_utf8_bytes -ne $case.Expect.valid) { throw 'SELFTEST_AGG_VALID' }
        if ($ag.invalid_utf8_bytes -ne $case.Expect.invalid) { throw "SELFTEST_AGG_INVALID: $($ag.invalid_utf8_bytes)" }
        if ($ag.valid_scalars -ne $case.Expect.scalars) { throw 'SELFTEST_AGG_SCALARS' }
        $expectedLoop = [double]$case.Expect.loop
        if ([Math]::Abs($ag.short_period_loop_fraction - $expectedLoop) -gt 1e-9) {
            throw "SELFTEST_AGG_LOOP: $($ag.short_period_loop_fraction)"
        }
    }
    Write-Host 'nicopedia_generation_aggregates_self_test=PASS'
}