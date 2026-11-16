# CPU-only test of the D8 disassembly audit rules.
#
# There is no V81 binary in this checkout, so this script generates a
# SYNTHETIC disassembly text in build/d8_disasm/, runs
# scripts/audit_hmx_d8_disasm.ps1 on it, and mutates it. The synthetic input
# exercises the parser and the rules only; it is not evidence about any real
# object. A clean fixture must pass for A/B/C/D; every mutation must fail.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Auditor = Join-Path $PSScriptRoot "audit_hmx_d8_disasm.ps1"
$OutDir = Join-Path $Root "build\d8_disasm"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function New-Fixture {
    param(
        [string]$Cond,
        [switch]$FillBeforeMma,
        [switch]$ConvertBeforeFill,
        [switch]$NoMma3,
        [switch]$SwapMma
    )
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($base in @(0, 2048, 4096, 6144)) {
        for ($k = 0; $k -lt 16; $k++) {
            $lines.Add("{ vmem(##$($base + $k * 128)) = v0 }")
        }
    }
    $lines.Add("{ bias = mxmem2(##10240) }")
    $lines.Add("{ bias = mxmem2(##10496) }")
    for ($k = 0; $k -lt 12; $k++) {
        $lines.Add("{ vmem(##$((10752 + $k * 128))) = v0 }")
    }
    $lines.Add("{ mxclracc.hf }")
    $mma = @(
        "{ activation.hf = mxmem(##0) weight.hf = mxmem(##4096) }",
        "{ activation.hf = mxmem(##0) weight.hf = mxmem(##6144) }",
        "{ activation.hf = mxmem(##2048) weight.hf = mxmem(##4096) }"
    )
    if ($SwapMma) { $tmp = $mma[0]; $mma[0] = $mma[1]; $mma[1] = $tmp }
    if (-not $NoMma3) {
        foreach ($m in $mma) { $lines.Add($m) }
    } else {
        $lines.Add($mma[0]); $lines.Add($mma[1])
    }
    $fills = New-Object System.Collections.Generic.List[string]
    if ($Cond -eq 'A' -or $Cond -eq 'B' -or $Cond -eq 'C' -or $Cond -eq 'D') {
        $baseOut = 8192
        for ($k = 0; $k -lt 16; $k++) {
            $fills.Add("{ vmem(##$($baseOut + $k * 128)) = v0 }")
        }
        if ($Cond -in @('B', 'C')) {
            for ($k = 0; $k -lt 16; $k++) {
                $fills.Add("{ vmem(##$((6144 + $k * 128))) = v0 }")
            }
        }
        if ($Cond -in @('B', 'D')) {
            for ($k = 0; $k -lt 16; $k++) {
                $fills.Add("{ vmem(##$($k * 128)) = v0 }")
            }
        }
    }
    if ($FillBeforeMma) {
        foreach ($f in $fills) { $lines.Insert(66, $f) }
    } else {
        foreach ($f in $fills) { $lines.Add($f) }
    }
    if ($ConvertBeforeFill) {
        $lines.Add("{ r0 = ##0x0001 }")
        $lines.Add("{ cvt.hf = acc(r0) }")
        foreach ($f in $fills) { $lines.Add($f) }
        $lines.Add("{ mxmem(##8192) = cvt }")
    } else {
        $lines.Add("{ r0 = ##0x0001 }")
        $lines.Add("{ cvt.hf = acc(r0) }")
        $lines.Add("{ mxmem(##8192) = cvt }")
    }
    return $lines
}

function Invoke-Case {
    param([string]$Name, [string]$Cond, [string[]]$Lines, [int]$Expect)
    $path = Join-Path $OutDir "$Name.txt"
    Set-Content -LiteralPath $path -Value $Lines -Encoding ascii
    $out = & pwsh -NoProfile -File $Auditor -DisasmPath $path -Condition $Cond 2>&1
    $exit = $LASTEXITCODE
    $ok = ($exit -eq $Expect)
    if (-not $ok) {
        Write-Host "FAIL case=$Name exit=$exit want=$Expect"
        $out | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "case=$Name exit=$exit want=$Expect PASS"
    }
    return $ok
}

$failed = 0
$cases = 0

foreach ($cond in @('A', 'B', 'C', 'D')) {
    $cases++
    if (-not (Invoke-Case "clean_$cond" $cond (New-Fixture $cond) 0)) { $failed++ }
}

# 1. one MMA missing
$cases++
if (-not (Invoke-Case "mut_no_mma3" 'B' (New-Fixture 'B' -NoMma3) 1)) { $failed++ }
# 2. MMA order swapped
$cases++
if (-not (Invoke-Case "mut_mma_swapped" 'B' (New-Fixture 'B' -SwapMma) 1)) { $failed++ }
# 3. a fill before MMA #3
$cases++
if (-not (Invoke-Case "mut_fill_before_mma" 'B' (New-Fixture 'B' -FillBeforeMma) 1)) { $failed++ }
# 4. out fill base wrong
$cases++
$m = New-Fixture 'B'
for ($i = 0; $i -lt $m.Count; $i++) {
    $m[$i] = $m[$i] -replace 'vmem\(##8192\)', 'vmem(##8064)'
}
if (-not (Invoke-Case "mut_fill_base" 'B' $m 1)) { $failed++ }
# 5. one 2,048 B store instead of 16 x 128 B
$cases++
$m = New-Fixture 'B'
$outFills = @($m | Where-Object { $_ -match 'vmem\(##8192\)' })
$m = @($m | Where-Object { $_ -notmatch 'vmem\(##8192\)' })
$m = @($m[0..65]) + @("{ vmem(##8192) = v0 }") + @($m[66..($m.Count - 1)])
if (-not (Invoke-Case "mut_fill_form" 'B' $m 1)) { $failed++ }
# 6. a guard write after the guard init
$cases++
$m = New-Fixture 'B'
$idx = [array]::IndexOf($m, "{ mxclracc.hf }")
$m = @($m[0..($idx - 1)]) + @("{ vmem(##10752) = v0 }") + @($m[$idx..($m.Count - 1)])
if (-not (Invoke-Case "mut_guard_write" 'B' $m 1)) { $failed++ }
# 7. forbidden synchronisation after MMA #3
$cases++
$m = New-Fixture 'B'
$idx = [array]::IndexOf($m, "{ activation.hf = mxmem(##2048) weight.hf = mxmem(##4096) }")
$m = @($m[0..$idx]) + @("{ syncht }") + @($m[($idx + 1)..($m.Count - 1)])
if (-not (Invoke-Case "mut_syncht" 'B' $m 1)) { $failed++ }
# 8. unresolved MMA address
$cases++
$m = New-Fixture 'B'
$m = @($m -replace 'weight\.hf = mxmem\(##6144\)', 'weight.hf = mxmem(r9)')
if (-not (Invoke-Case "mut_unresolved" 'B' $m 1)) { $failed++ }
# 9. Convert before the fills
$cases++
if (-not (Invoke-Case "mut_convert_order" 'B' (New-Fixture 'B' -ConvertBeforeFill) 1)) { $failed++ }
# 10. wrong store target
$cases++
$m = New-Fixture 'B'
$m = @($m -replace 'mxmem\(##8192\) = cvt', 'mxmem(##8200) = cvt')
if (-not (Invoke-Case "mut_store_target" 'B' $m 1)) { $failed++ }
# 11. condition C with an Ahi fill (wrong mask)
$cases++
$m = New-Fixture 'C'
$m = @($m[0..($m.Count - 4)]) + @("{ vmem(##0) = v0 }") + @($m[($m.Count - 3)..($m.Count - 1)])
if (-not (Invoke-Case "mut_wrong_mask" 'C' $m 1)) { $failed++ }
# 12. a sub-128 B fill store
$cases++
$m = New-Fixture 'B'
$m = @($m -replace 'vmem\(##8192\) = v0', 'vmem(##8192) = v0.b')
if (-not (Invoke-Case "mut_fill_width" 'B' $m 1)) { $failed++ }

Write-Host "d8_disasm_audit_test_cases=$cases failures=$failed"
if ($failed -ne 0) {
    Write-Host "d8_disasm_audit_test=FAIL"
    exit 1
}
Write-Host "d8_disasm_audit_test=PASS clean=4 mutants=12"
Write-Host "d8_disasm_audit_test_scope=synthetic_text_only_no_real_v81_binary"
exit 0
