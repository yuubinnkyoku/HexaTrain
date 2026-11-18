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

# Pin the working directory to the repository root so build/ is always resolved
# inside this worktree, whatever directory the caller used (see
# run_hmx_d8_sim.ps1 for the incident that motivated this).
$pushedLocation = $false
Push-Location $Root
$pushedLocation = $true

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
    param([string]$Name, [string]$Cond, [string[]]$Lines, [int]$Expect,
          [string]$Base = "")
    $path = Join-Path $OutDir "$Name.txt"
    Set-Content -LiteralPath $path -Value $Lines -Encoding ascii
    $argv = @("-NoProfile", "-File", $Auditor, "-DisasmPath", $path,
              "-Condition", $Cond)
    if ($Base) { $argv += @("-BaseRegister", $Base) }
    $out = & pwsh @argv 2>&1
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

# Base-relative fixture: the shape a real V81 object has. The handler receives
# the VTCM base in r0, derives every operand pointer from it with a constant
# add, then spills and reloads them. Loops are used for the 16 x 128 B blocks.
function New-BaseFixture {
    param(
        [string]$Cond,
        [switch]$SwapMma3ToZero,
        [int]$LoopCount = 16,
        [int]$RsValue = 1,
        [switch]$PayloadAsAddress
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("{ r21 = add(r0,#0x0) }")      # Ahi  +0
    $lines.Add("{ r27 = add(r0,#0x800) }")    # Alo  +2048
    $lines.Add("{ r4 = add(r0,#0x1000) }")    # Whi  +4096
    $lines.Add("{ r26 = add(r0,#0x1800) }")   # Wlo  +6144
    $lines.Add("{ r8 = add(r0,#0x2000) }")    # out  +8192
    $lines.Add("{ r16 = add(r0,#0x2800) }")   # bias0 +10240
    $lines.Add("{ r17 = add(r0,#0x2900) }")   # bias1 +10496
    $lines.Add("{ r18 = add(r0,#0x2a00) }")   # guard +10752
    $lines.Add("{ memw(r29+#0x4) = r21 }")
    $lines.Add("{ memw(r29+#0x8) = r27 }")
    $lines.Add("{ memw(r29+#0x10) = r4 }")
    $lines.Add("{ memw(r29+#0xc) = r26 }")
    $lines.Add("{ memw(r29+#0x2c) = r8 }")
    $lines.Add("{ memw(r29+#0x1c) = r16 }")
    $lines.Add("{ memw(r29+#0x18) = r17 }")
    $lines.Add("{ memw(r29+#0x24) = r18 }")
    # packs: 4 blocks of 16 x 128 B
    foreach ($base in @('r21', 'r27', 'r4', 'r26')) {
        $lines.Add("{ loop0(0x0,#0x10) }")
        $lines.Add("{ vmem($base+#0x0) = v0 }")
        $lines.Add("{ $base = add($base,#0x80) :endloop0 }")
    }
    $lines.Add("{ bias = mxmem2(r16) }")
    $lines.Add("{ bias = mxmem2(r17) }")
    # guard init: 12 x 128 B
    $lines.Add("{ loop0(0x0,#0xc) }")
    $lines.Add("{ vmem(r18+#0x0) = v0 }")
    $lines.Add("{ r18 = add(r18,#0x80) :endloop0 }")
    $lines.Add("{ mxclracc.hf }")
    $mma3act = if ($SwapMma3ToZero) { 'r21' } else { 'r27' }
    $lines.Add("{ activation.hf = mxmem(r21) weight.hf = mxmem(r4) }")
    $lines.Add("{ activation.hf = mxmem(r21) weight.hf = mxmem(r26) }")
    $lines.Add("{ activation.hf = mxmem($mma3act) weight.hf = mxmem(r4) }")
    # fills
    $lines.Add("{ loop0(0x0,#$('0x{0:x}' -f $LoopCount)) }")
    $lines.Add("{ vmem(r8+#0x0) = v0 }")
    $lines.Add("{ r8 = add(r8,#0x80) :endloop0 }")
    if ($Cond -in @('B', 'C')) {
        $lines.Add("{ loop0(0x0,#$('0x{0:x}' -f $LoopCount)) }")
        $lines.Add("{ vmem(r26+#0x0) = v0 }")
        $lines.Add("{ r26 = add(r26,#0x80) :endloop0 }")
    }
    if ($Cond -in @('B', 'D')) {
        $lines.Add("{ loop0(0x0,#$('0x{0:x}' -f $LoopCount)) }")
        $lines.Add("{ vmem(r21+#0x0) = v0 }")
        $lines.Add("{ r21 = add(r21,#0x80) :endloop0 }")
    }
    $lines.Add("{ r0 = ##$('0x{0:x4}' -f $RsValue) }")
    $lines.Add("{ cvt.hf = acc(r0) }")
    $lines.Add("{ mxmem(r8) = cvt }")
    if ($PayloadAsAddress) {
        # A payload constant used as an address must not resolve.
        $lines.Add("{ loop0(0x0,#0x10) }")
        $lines.Add("{ vmem(##0xa5a5) = v0 }")
        $lines.Add("{ r5 = add(r5,#0x80) :endloop0 }")
    }
    return $lines
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

# ---------------------------------------------------------------------------
# Base-relative rules (2026-11-19). A real V81 object receives the VTCM base in
# a register, derives the operand pointers from it, spills them, and lowers
# 16 x 128 B to loop0. The audit gained three rules for that shape: a declared
# base register with derivation tracking, loop expansion, and an Rs snapshot.
# Each rule gets a clean case and a mutation that must be rejected.
# ---------------------------------------------------------------------------
$baseReg = "r0"
$baseCases = 0
$baseFailed = 0

# 13-16. the base-relative shape must pass for all four conditions
foreach ($cond in @('A', 'B', 'C', 'D')) {
    $baseCases++
    if (-not (Invoke-Case "base_clean_$cond" $cond (New-BaseFixture $cond) 0 $baseReg)) {
        $baseFailed++
    }
}

# 17. MMA #3 reading +0 instead of +2048 must be rejected
$baseCases++
if (-not (Invoke-Case "base_mut_mma3_offset" 'B' (New-BaseFixture 'B' -SwapMma3ToZero) 1 $baseReg)) {
    $baseFailed++
}

# 18. a loop count of 8 instead of 16 must be rejected
$baseCases++
if (-not (Invoke-Case "base_mut_loop_count" 'B' (New-BaseFixture 'B' -LoopCount 8) 1 $baseReg)) {
    $baseFailed++
}

# 19. a real Rs change must be rejected (the snapshot must still catch it)
$baseCases++
if (-not (Invoke-Case "base_mut_rs_value" 'B' (New-BaseFixture 'B' -RsValue 0x1004) 1 $baseReg)) {
    $baseFailed++
}

# 20. a payload constant used as an address must not resolve
$baseCases++
if (-not (Invoke-Case "base_mut_payload_address" 'B' (New-BaseFixture 'B' -PayloadAsAddress) 1 $baseReg)) {
    $baseFailed++
}

# 21. the same base-relative text without -BaseRegister must fail closed:
#     rN is then an unknown register, not an address.
$baseCases++
if (-not (Invoke-Case "base_mut_no_base_declared" 'B' (New-BaseFixture 'B') 1 '')) {
    $baseFailed++
}

$cases += $baseCases
$failed += $baseFailed

Write-Host "d8_disasm_audit_test_cases=$cases failures=$failed"
if ($failed -ne 0) {
    Write-Host "d8_disasm_audit_test=FAIL"
    if ($pushedLocation) { Pop-Location }
    exit 1
}
Write-Host "d8_disasm_audit_test=PASS clean=4 mutants=12 base_clean=4 base_mutants=5"
Write-Host "d8_disasm_audit_test_scope=synthetic_text_only_no_real_v81_binary"
if ($pushedLocation) { Pop-Location }
exit 0
