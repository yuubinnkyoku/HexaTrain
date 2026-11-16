# Static disassembly audit for the D8 (Candidate B minimal) instruction
# sequence, revision 3 proposal.
#
# This audits a DISASSEMBLY TEXT FILE, not the source, not a running device.
# It is the CPU-side rule engine described in
# docs/hmx-v81-d8-disasm-audit-spec.md. On a real implementation the text is
# produced by hexagon-llvm-objdump for the stamped symbol; in this checkout no
# V81 binary exists, so the only exercised input is the synthetic fixture from
# scripts/test_hmx_d8_disasm_audit.ps1.
#
# Fail-closed rules: an address the parser cannot resolve, an unregistered
# memory access, a store in the critical interval, or a vector store narrower
# than 128 B is a FAIL, never a skip.
#
# Examples:
#   .\scripts\audit_hmx_d8_disasm.ps1 -DisasmPath build\d8_disasm\clean_B.txt -Condition B
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DisasmPath,
    [ValidateSet('A', 'B', 'C', 'D')][string]$Condition = 'B',
    [string]$Symbol = "run_mode46_one_trial"
)
$ErrorActionPreference = "Stop"

$script:Failed = 0
function Fail([string]$Message) {
    Write-Host "FAIL $Message"
    $script:Failed++
}

if (-not (Test-Path -LiteralPath $DisasmPath)) {
    Fail "disasm_missing: $DisasmPath"
    Write-Host "d8_disasm_audit=FAIL"
    exit 1
}
Write-Host "d8_disasm_audit_input=$DisasmPath"
Write-Host "d8_disasm_audit_symbol=$Symbol"
Write-Host "d8_disasm_audit_condition=$Condition"

# ---------------------------------------------------------------- parse lines
$segments = New-Object System.Collections.Generic.List[object]
$regs = @{}
$lineNo = 0
foreach ($line in (Get-Content -LiteralPath $DisasmPath)) {
    $lineNo++
    if ($line -match '^\s*([0-9a-fA-F]+):\s*(.*)$') {
        $text = $Matches[2]
    } elseif ($line -match '^\s*\{\s*(.*?)\s*\}\s*$') {
        $text = $Matches[1]
    } elseif ($line -match '^\s*(\S.*)$') {
        $text = $Matches[1]
    } else {
        continue
    }
    $text = ($text -replace '\s+', ' ').Trim()
    if (-not $text) { continue }
    foreach ($raw in ($text -split ';')) {
        $seg = ($raw -replace '\s+', ' ').Trim()
        if (-not $seg) { continue }
        $segments.Add([pscustomobject]@{ Index = $segments.Count; Text = $seg; Line = $lineNo })
    }
}
Write-Host "d8_disasm_audit_segments=$($segments.Count)"
if ($segments.Count -eq 0) {
    Fail "disasm_empty"
    Write-Host "d8_disasm_audit=FAIL"
    exit 1
}

function Resolve-Expr([string]$Expr) {
    $e = $Expr.Trim()
    if ($e -match '^(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        return [Convert]::ToInt32($Matches[1], $(if ($Matches[1].StartsWith('0x')) { 16 } else { 10 }))
    }
    if ($e -match '^(r\d+)\s*\+\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        $reg = $Matches[1]
        $off = [Convert]::ToInt32($Matches[2], $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        if ($script:regs.ContainsKey($reg)) { return ($script:regs[$reg] + $off) }
        return $null
    }
    if ($e -match '^(r\d+)$' -and $script:regs.ContainsKey($Matches[1])) {
        return $script:regs[$Matches[1]]
    }
    return $null
}

$mxIdx = @()
$actIdx = @()
$wgtIdx = @()
$biasIdx = @()
$biasAddr = @()
$cvtIdx = @()
$storeIdx = @()
$storeAddr = @()
$vecStores = @()   # objects: Index, Addr, Width, Text
$readsUsed = @{}
$forbiddenIdx = @()

for ($i = 0; $i -lt $segments.Count; $i++) {
    $t = $segments[$i].Text

    if ($t -match '^(r\d+)\s*=\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        $regs[$Matches[1]] = [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        continue
    }

    if ($t -match '\bmxclracc\.hf\b') { $mxIdx += $i }
    if ($t -match '\bbias\s*=\s*mxmem2\(([^)]*)\)') {
        $addr = Resolve-Expr $Matches[1]
        if ($null -eq $addr) { Fail "bias_address_unresolved: line $($segments[$i].Line)" }
        $biasIdx += $i
        $biasAddr += $addr
        $readsUsed[$i] = 1
    }
    if ($t -match '\bcvt\.hf\s*=\s*acc\((r\d+)\)') {
        $cvtIdx += $i
        $script:cvtReg = $Matches[1]
    }
    if ($t -match '\bmxmem\(([^)]*)\)\s*=\s*cvt\b') {
        $addr = Resolve-Expr $Matches[1]
        if ($null -eq $addr) { Fail "store_address_unresolved: line $($segments[$i].Line)" }
        $storeIdx += $i
        $storeAddr += $addr
    }
    if ($t -match '\bmxmem\(([^)]*)\)' -and
        $t -notmatch '\bmxmem\([^)]*\)\s*=\s*cvt') {
        # activation/weight reads are validated below; anything else is
        # recorded so the "unregistered read" rule can see it.
        $readsUsed[$i] = 1
    }
    if ($t -match '\bvmem\(([^)]*)\)\s*=\s*(v\d+)(\.\w+)?') {
        $addr = Resolve-Expr $Matches[1]
        $suffix = $Matches[3]
        $width = 128
        if ($suffix -and $suffix -match '\.(b|h|w)$') { $width = 0 }
        if ($null -eq $addr) { Fail "vector_address_unresolved: line $($segments[$i].Line)" }
        $vecStores += [pscustomobject]@{
            Index = $i; Addr = $addr; Width = $width; Suffix = $suffix
            Text = $t; Line = $segments[$i].Line
        }
    }
    if ($t -match '\b(syncht|barrier|dmwait|dmstart|dmb|isync)\b') {
        $forbiddenIdx += $i
    }
}

# MMA packets: a single segment with both activation and weight, or an
# activation immediately followed by a weight.
$mma = @()
for ($i = 0; $i -lt $segments.Count; $i++) {
    $t = $segments[$i].Text
    $both = [regex]::Match($t,
        'activation\.hf\s*=\s*mxmem\(([^)]*)\).*weight\.hf\s*=\s*mxmem\(([^)]*)\)')
    if ($both.Success) {
        $a = Resolve-Expr $both.Groups[1].Value
        $w = Resolve-Expr $both.Groups[2].Value
        if ($null -eq $a -or $null -eq $w) {
            Fail "mma_address_unresolved: line $($segments[$i].Line)"
        }
        $mma += [pscustomobject]@{ Index = $i; WeightIndex = $i; Act = $a; Wgt = $w }
        continue
    }
    $act = [regex]::Match($t, 'activation\.hf\s*=\s*mxmem\(([^)]*)\)')
    if ($act.Success -and $i + 1 -lt $segments.Count) {
        $wgt = [regex]::Match($segments[$i + 1].Text,
            'weight\.hf\s*=\s*mxmem\(([^)]*)\)')
        if ($wgt.Success) {
            $a = Resolve-Expr $act.Groups[1].Value
            $w = Resolve-Expr $wgt.Groups[1].Value
            if ($null -eq $a -or $null -eq $w) {
                Fail "mma_address_unresolved: line $($segments[$i].Line)"
            }
            $mma += [pscustomobject]@{
                Index = $i; WeightIndex = $i + 1; Act = $a; Wgt = $w
            }
        }
    }
}

Write-Host "d8_disasm_audit_counts mxclracc=$($mxIdx.Count) mma=$($mma.Count) bias=$($biasIdx.Count) convert=$($cvtIdx.Count) store=$($storeIdx.Count) vector_stores=$($vecStores.Count)"

if ($mxIdx.Count -ne 1) { Fail "mxclracc_count: $($mxIdx.Count)/1" }
if ($mma.Count -ne 3) { Fail "mma_count: $($mma.Count)/3" }
if ($biasIdx.Count -ne 2) { Fail "bias_count: $($biasIdx.Count)/2" }
if ($cvtIdx.Count -ne 1) { Fail "convert_count: $($cvtIdx.Count)/1" }
if ($storeIdx.Count -ne 1) { Fail "store_count: $($storeIdx.Count)/1" }

# ---------------------------------------------------------------- fixed order
if ($mxIdx.Count -ge 1) {
    foreach ($b in $biasIdx) {
        if ($b -gt $mxIdx[0]) { Fail "bias_after_mxclracc" }
    }
    foreach ($m in $mma) {
        if ($m.Index -lt $mxIdx[0]) { Fail "mma_before_mxclracc" }
    }
}
if ($biasIdx.Count -eq 2 -and ($biasAddr[0] -ne 10240 -or $biasAddr[1] -ne 10496)) {
    Fail "bias_addresses: [$($biasAddr -join ',')] want [10240,10496]"
}
$expectedMma = @(@(0, 4096), @(0, 6144), @(2048, 4096))
for ($k = 0; $k -lt [Math]::Min($mma.Count, 3); $k++) {
    if ($mma[$k].Act -ne $expectedMma[$k][0] -or $mma[$k].Wgt -ne $expectedMma[$k][1]) {
        Fail "mma_$k : got ($($mma[$k].Act),$($mma[$k].Wgt)) want ($($expectedMma[$k] -join ','))"
    }
}
if ($storeIdx.Count -eq 1 -and $storeAddr[0] -ne 8192) {
    Fail "store_address: $($storeAddr[0])/8192"
}
if ($cvtIdx.Count -eq 1 -and $storeIdx.Count -eq 1 -and $cvtIdx[0] -gt $storeIdx[0]) {
    Fail "store_before_convert"
}
if ($cvtIdx.Count -eq 1) {
    $rs = $null
    if ($script:cvtReg -and $regs.ContainsKey($script:cvtReg)) { $rs = $regs[$script:cvtReg] }
    if ($null -eq $rs) {
        Fail "convert_rs_unresolved"
    } elseif ($rs -ne 1) {
        Fail ("convert_rs: 0x{0:x4}/0x0001" -f $rs)
    }
}

# ------------------------------------------------------------- vector stores
$firstBias = if ($biasIdx.Count -gt 0) { ($biasIdx | Measure-Object -Minimum).Minimum } else { [int]::MaxValue }
$firstMx = if ($mxIdx.Count -gt 0) { $mxIdx[0] } else { [int]::MaxValue }
$lastMma = if ($mma.Count -gt 0) { ($mma | ForEach-Object { $_.Index } | Measure-Object -Maximum).Maximum } else { -1 }

$packStores = @($vecStores | Where-Object { $_.Index -lt $firstBias })
$guardStores = @($vecStores | Where-Object { $_.Index -ge $firstBias -and $_.Index -lt $firstMx })
$fillStores = @($vecStores | Where-Object { $_.Index -gt $lastMma })
$criticalStores = @($vecStores | Where-Object { $_.Index -ge $firstMx -and $_.Index -le $lastMma })

foreach ($s in $vecStores) {
    if ($s.Width -ne 128) { Fail "vector_width: line $($s.Line) '$($s.Text)' is narrower than 128 B" }
}
foreach ($s in $criticalStores) { Fail "store_in_critical_interval: line $($s.Line)" }

# Packs: 16 stores at each operand base, in order Ahi/Alo/Whi/Wlo.
$packBases = @(0, 2048, 4096, 6144)
for ($b = 0; $b -lt 4; $b++) {
    $block = @($packStores | Where-Object {
        $_.Addr -ge $packBases[$b] -and $_.Addr -lt $packBases[$b] + 2048
    })
    if ($block.Count -ne 16) { Fail "pack_$($packBases[$b])_count: $($block.Count)/16" }
    for ($k = 0; $k -lt $block.Count; $k++) {
        if ($block[$k].Addr -ne $packBases[$b] + $k * 128) {
            Fail "pack_$($packBases[$b])_address: $($block[$k].Addr)"
            break
        }
    }
}

# Guard: exactly 12 stores at +10752 + k*128, no later guard writes.
if ($guardStores.Count -ne 12) { Fail "guard_init_count: $($guardStores.Count)/12" }
for ($k = 0; $k -lt $guardStores.Count; $k++) {
    if ($guardStores[$k].Addr -ne 10752 + $k * 128) {
        Fail "guard_init_address: $($guardStores[$k].Addr)"
        break
    }
}
foreach ($s in $vecStores) {
    if ($s.Index -gt $firstMx -and $s.Addr -ge 10752) {
        Fail "guard_written_after_init: line $($s.Line)"
    }
}

# Fills: out (+8192) always, Wlo (+6144) for B/C, Ahi (+0) for B/D.
$fillMask = @{ A = 1; B = 7; C = 3; D = 5 }[$Condition]
$fillPlan = @()
if ($fillMask -band 1) { $fillPlan += 8192 }
if ($fillMask -band 2) { $fillPlan += 6144 }
if ($fillMask -band 4) { $fillPlan += 0 }
$expectedFillCount = 16 * $fillPlan.Count
if ($fillStores.Count -ne $expectedFillCount) {
    Fail "fill_count: $($fillStores.Count)/$expectedFillCount (condition $Condition)"
}
$planIndex = 0
$offset = 0
while ($offset -lt $fillStores.Count -and $planIndex -lt $fillPlan.Count) {
    for ($k = 0; $k -lt 16 -and ($offset + $k) -lt $fillStores.Count; $k++) {
        $want = $fillPlan[$planIndex] + $k * 128
        if ($fillStores[$offset + $k].Addr -ne $want) {
            Fail "fill_$($fillPlan[$planIndex])_address: $($fillStores[$offset + $k].Addr)/$want"
            break
        }
    }
    $offset += 16
    $planIndex++
}

# Every fill must follow MMA #3 and precede Convert.
if ($fillStores.Count -gt 0) {
    $firstFill = ($fillStores | ForEach-Object { $_.Index } | Measure-Object -Minimum).Minimum
    if ($firstFill -lt $lastMma) { Fail "fill_before_last_mma" }
    if ($cvtIdx.Count -eq 1) {
        foreach ($s in $fillStores) {
            if ($s.Index -gt $cvtIdx[0]) { Fail "fill_after_convert: line $($s.Line)" }
        }
    }
}

# Unregistered reads: every mxmem read must be an MMA operand or a bias load.
$readSegments = @()
for ($i = 0; $i -lt $segments.Count; $i++) {
    if ($segments[$i].Text -match '\bmxmem\(' -and
        $segments[$i].Text -notmatch '\bmxmem\([^)]*\)\s*=\s*cvt') {
        $readSegments += $i
    }
}
$allowed = @{}
foreach ($m in $mma) {
    $allowed[$m.Index] = 1
    $allowed[$m.WeightIndex] = 1
}
foreach ($b in $biasIdx) { $allowed[$b] = 1 }
foreach ($i in $readSegments) {
    if (-not $allowed.ContainsKey($i)) { Fail "unregistered_read: line $($segments[$i].Line) '$($segments[$i].Text)'" }
}

# Forbidden synchronisation in the critical interval [mxclracc, store].
if ($mxIdx.Count -eq 1 -and $storeIdx.Count -eq 1) {
    foreach ($f in $forbiddenIdx) {
        if ($f -ge $mxIdx[0] -and $f -le $storeIdx[0]) {
            Fail "forbidden_instruction: line $($segments[$f].Line) '$($segments[$f].Text)'"
        }
    }
}

if ($script:Failed -gt 0) {
    Write-Host "d8_disasm_audit=FAIL checks_failed=$($script:Failed)"
    exit 1
}
Write-Host "d8_disasm_audit=PASS packs=64 guard=12 fills=$($fillStores.Count) mma=3 convert=1 store=1"
Write-Host "d8_disasm_audit_scope=text_rules_only_no_real_v81_binary"
exit 0
