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
    [string]$Symbol = "run_mode46_one_trial",
    # VTCM base register, e.g. r0. VTCM is allocated at run time by
    # HAP_compute_res_attr_get_vtcm_ptr, so a real object cannot contain the
    # absolute offsets the synthetic fixture uses. The handler receives the base
    # in one register and derives every operand pointer from it with a constant
    # add, then spills the derived pointers to the stack frame and reloads them.
    # Declaring the base register lets the audit follow that chain. Two rules
    # keep this fail-closed:
    #
    #   1. only the declared register, and registers provably derived from it by
    #      a constant add (directly or through a stack slot), resolve to an
    #      offset; any other expression is unresolved and fails;
    #   2. a bare ##N immediate is NOT an address when a base is declared.
    #      Without rule 2 the payload constants the handler writes (scale
    #      0x5400, fill pattern 0xA5A5) would resolve as addresses and pass.
    #
    # Without this parameter the audit behaves exactly as before (absolute ##N).
    [string]$BaseRegister = "",
    # Symbol that performs the four packs, when the compiler keeps it out of
    # line. The handler then contains four calls instead of four loops, and the
    # pack counts must be taken from this symbol's body. Empty means the packs
    # are inline in the handler.
    [string]$PackSymbol = "",
    # Object file the symbols come from, needed when -PackSymbol is used.
    [string]$ObjectPath = "",
    # objdump to use for the helper symbol.
    [string]$objdump = "hexagon-llvm-objdump"
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

# ------------------------------------------------------- VTCM pointer map
# See the -BaseMap comment in the param block for why this is a map and not a
# single base register.
$script:vtcmMap = @{}
$script:slotReg = @{}
$script:mapAt = @{}
$script:loopSnapshots = @{}
$script:vtcmMapMode = $false
if ($BaseRegister) {
    if ($BaseRegister -notmatch '^r\d+$') {
        Fail "base_register_invalid: '$BaseRegister' is not a register name"
    } else {
        $script:vtcmMap[$BaseRegister] = 0
        $script:vtcmMapMode = $true
    }
}

function Resolve-AddrAt([string]$Expr, $Map) {
    # Resolve against a snapshot of the derivation map taken at a given segment.
    # In absolute mode (no -BaseRegister) the map is not used and the expression
    # is resolved directly, exactly as Resolve-Addr does.
    $e = ($Expr -split ',')[0].Trim()
    if (-not $script:vtcmMapMode) { return Resolve-Expr $e }
    if ($null -eq $Map) { return $null }
    if ($e -match '^(r\d+)\s*\+\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        $reg = $Matches[1]
        $off = [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        if ($Map.ContainsKey($reg)) { return ($Map[$reg] + $off) }
        return $null
    }
    if ($e -match '^(r\d+)$' -and $Map.ContainsKey($Matches[1])) {
        return $Map[$Matches[1]]
    }
    return $null
}

function Resolve-Addr([string]$Expr) {
    # A real object spells the HMX operand as mxmem(ptr, rt): the pointer and
    # the register-type field. The address is the first component; the rt is a
    # separate encoding the rules do not interpret.
    $e = ($Expr -split ',')[0].Trim()
    if ($script:vtcmMapMode) {
        # base mode: only a declared or derived register carries an offset
        if ($e -match '^(r\d+)\s*\+\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
            $reg = $Matches[1]
            $off = [Convert]::ToInt32($Matches[2],
                $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
            if ($script:vtcmMap.ContainsKey($reg)) {
                return ($script:vtcmMap[$reg] + $off)
            }
            return $null
        }
        if ($e -match '^(r\d+)$' -and $script:vtcmMap.ContainsKey($Matches[1])) {
            return $script:vtcmMap[$Matches[1]]
        }
        return $null
    }
    return Resolve-Expr $e
}

$mxIdx = @()
$actIdx = @()
$wgtIdx = @()
$biasIdx = @()
$biasAddr = @()
$cvtIdx = @()
$cvtRs = $null          # snapshot of the Rs register AT the cvt index
$storeIdx = @()
$storeAddr = @()
$vecStores = @()   # objects: Index, Addr, Width, Text, LoopCount, LoopBody
$readsUsed = @{}
$forbiddenIdx = @()
$loopIdx = @()     # objects: Index, Count, BodyIndex, BodyText

# Loop ranges must be known before the derivation walk: a loop body advances its
# pointer by 128 B per iteration, and that increment must not leak into the
# addresses used after the loop (the MMA operands and the store). The body is
# every segment from loop0+1 up to the segment carrying :endloop0.
$loopBody = @{}    # loop index -> @(first, last)
for ($i = 0; $i -lt $segments.Count; $i++) {
    if ($segments[$i].Text -notmatch '^loop[01]\(') { continue }
    for ($j = $i + 1; $j -lt $segments.Count; $j++) {
        if ($segments[$j].Text -match ':endloop[01]') {
            $loopBody[$i] = @(($i + 1), $j)
            break
        }
    }
}
$inLoopBody = @{}
foreach ($k in $loopBody.Keys) {
    for ($j = $loopBody[$k][0]; $j -le $loopBody[$k][1]; $j++) { $inLoopBody[$j] = $k }
}

for ($i = 0; $i -lt $segments.Count; $i++) {
    $t = $segments[$i].Text
    # Snapshot the derivation map at this segment. The MMA pass reads the
    # snapshot rather than the final map, because a register that holds an
    # operand at the MMA is routinely reused afterwards (the handler reloads r5
    # for its FNV loops), so resolving late loses it.
    $script:mapAt[$i] = $script:vtcmMap.Clone()
    if ($env:D8_DBG -and $i -le 6) { Write-Host "DBG i=$i r19=[$($script:vtcmMap['r19'])] r0=[$($script:vtcmMap['r0'])] t=$t" }
    if ($env:D8_DBG -and $i -lt 6) { Write-Host "DBG i=$i t=$t r19=$($script:vtcmMap['r19']) r0=$($script:vtcmMap['r0'])" }

    if ($t -match '^(r\d+)\s*=\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        $regs[$Matches[1]] = [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        # A constant assignment destroys any VTCM derivation the register had.
        if ($script:vtcmMapMode -and $script:vtcmMap.ContainsKey($Matches[1])) {
            $script:vtcmMap.Remove($Matches[1])
        }
        continue
    }
    # rD = add(rS, #imm): a pointer derived from the base by a constant add.
    # Inside a loop body the increment is per-iteration, so it must not change
    # the derivation used after the loop; the body's own store is resolved
    # separately by the loop expansion below.
    if ($t -match '^(r\d+)\s*=\s*add\((r\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)') {
        $dst = $Matches[1]
        $src = $Matches[2]
        $off = [Convert]::ToInt32($Matches[3],
            $(if ($Matches[3].StartsWith('0x')) { 16 } else { 10 }))
        # Inside a loop body the increment is per-iteration, so it must not
        # change the derivation used AFTER the loop; the body's own store is
        # resolved separately by the loop expansion below.
        if ($script:vtcmMapMode -and -not $inLoopBody.ContainsKey($i)) {
            if ($script:vtcmMap.ContainsKey($src)) {
                $script:vtcmMap[$dst] = $script:vtcmMap[$src] + $off
            } elseif ($script:vtcmMap.ContainsKey($dst)) {
                $script:vtcmMap.Remove($dst)
            }
        }
        continue
    }
    # rD = rS: a register copy preserves the derivation.
    if ($t -match '^(r\d+)\s*=\s*(r\d+)$') {
        $dst = $Matches[1]
        $src = $Matches[2]
        if ($script:vtcmMapMode) {
            if ($script:vtcmMap.ContainsKey($src)) {
                $script:vtcmMap[$dst] = $script:vtcmMap[$src]
            } elseif ($script:vtcmMap.ContainsKey($dst)) {
                $script:vtcmMap.Remove($dst)
            }
        }
        continue
    }
    # Stack spill: memw(r29+#off) = rN, or the 64-bit memd(r29+#off) = rM:rN
    # which stores rN at +off and rM at +off+4. The slot keeps the derivation
    # the register had at the spill, which is what a later reload must see.
    if ($t -match 'memd\(r29\+#(0x[0-9a-fA-F]+)\)\s*=\s*r(\d+):(\d+)') {
        $off = [Convert]::ToInt32($Matches[1],
            $(if ($Matches[1].StartsWith('0x')) { 16 } else { 10 }))
        $hi = 'r' + $Matches[2]
        $lo = 'r' + $Matches[3]
        foreach ($pair in @(@($off, $lo), @(($off + 4), $hi))) {
            $slot = 's' + $pair[0]
            $src = $pair[1]
            $script:slotReg[$slot] = $src
            if ($script:vtcmMapMode) {
                if ($script:vtcmMap.ContainsKey($src)) {
                    $script:vtcmMap[$slot] = $script:vtcmMap[$src]
                } elseif ($script:vtcmMap.ContainsKey($slot)) {
                    $script:vtcmMap.Remove($slot)
                }
            }
        }
    }
    if ($t -match 'memw\(r29\+#(0x[0-9a-fA-F]+)\)\s*=\s*(r\d+)(?:\.new)?') {
        $slot = 's' + [Convert]::ToInt32($Matches[1],
            $(if ($Matches[1].StartsWith('0x')) { 16 } else { 10 }))
        $script:slotReg[$slot] = $Matches[2]
        if ($script:vtcmMapMode) {
            if ($script:vtcmMap.ContainsKey($Matches[2])) {
                $script:vtcmMap[$slot] = $script:vtcmMap[$Matches[2]]
            } elseif ($script:vtcmMap.ContainsKey($slot)) {
                $script:vtcmMap.Remove($slot)
            }
        }
    }
    # 64-bit register pair build: rM:rN = combine(rS,rT) puts rS in rM (the high
    # half) and rT in rN (the low half). The compiler uses it to save the
    # incoming argument registers, so the base arrives as the high half of a pair
    # rather than as a plain copy.
    #
    # The pair is spelled "r19:18", not "r19:r18": the high half carries the r
    # prefix and the low half does not.
    if ($t -match '^r(\d+):(\d+)\s*=\s*combine\((r\d+)\s*,\s*(r\d+)\)') {
        $hi = 'r' + $Matches[1]
        $lo = 'r' + $Matches[2]
        $srcHi = $Matches[3]
        $srcLo = $Matches[4]
        if ($script:vtcmMapMode) {
            foreach ($pair in @(@($lo, $srcLo), @($hi, $srcHi))) {
                $dst = $pair[0]
                $src = $pair[1]
                if ($script:vtcmMap.ContainsKey($src)) {
                    $script:vtcmMap[$dst] = $script:vtcmMap[$src]
                } elseif ($script:vtcmMap.ContainsKey($dst)) {
                    $script:vtcmMap.Remove($dst)
                }
            }
        }
        continue
    }
    # 64-bit stack reload: rM:rN = memd(r29+#off). rN comes from +off and rM
    # from +off+4, mirroring the spill. The compiler uses this to restore the
    # callee-saved registers in the epilogue and to reload pointer pairs.
    if ($t -match '^r(\d+):(\d+)\s*=\s*memd\(r29\+#(0x[0-9a-fA-F]+)\)') {
        $hi = $Matches[1]
        $lo = $Matches[2]
        $off = [Convert]::ToInt32($Matches[3],
            $(if ($Matches[3].StartsWith('0x')) { 16 } else { 10 }))
        foreach ($pair in @(@($off, $lo), @(($off + 4), $hi))) {
            $slot = 's' + $pair[0]
            $dst = $pair[1]
            if ($script:vtcmMapMode) {
                if ($script:vtcmMap.ContainsKey($slot)) {
                    $script:vtcmMap[$dst] = $script:vtcmMap[$slot]
                } elseif ($script:vtcmMap.ContainsKey($dst)) {
                    $script:vtcmMap.Remove($dst)
                }
            }
        }
        continue
    }
    # Stack reload: rN = memw(r29+#off). The register inherits the slot. The
    # pattern is not anchored at the end because the compiler emits compound
    # segments ("r4 = memw(r29+#0x4); memw(r6+#0x28) = #1").
    if ($t -match '^(r\d+)\s*=\s*memw\(r29\+#(0x[0-9a-fA-F]+)\)') {
        $dst = $Matches[1]
        $slot = 's' + [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        if ($script:vtcmMapMode) {
            if ($script:vtcmMap.ContainsKey($slot)) {
                $script:vtcmMap[$dst] = $script:vtcmMap[$slot]
            } elseif ($script:vtcmMap.ContainsKey($dst)) {
                $script:vtcmMap.Remove($dst)
            }
        }
        continue
    }
    # rD = setbit(rS,#mask) / rD = and(rS,#mask): the pointer survives, only the
    # low tag bits the HMX encoding carries change. The derivation is inherited
    # from the source register. This must be checked before the generic
    # "any other write" rule below, and it must not require the mask to be small:
    # the handler tags bias set 1 with |1, which is exactly this form.
    if ($t -match '^(r\d+)\s*=\s*(?:setbit|and)\((r\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)') {
        $dst = $Matches[1]
        $src = $Matches[2]
        if ($script:vtcmMapMode) {
            if ($script:vtcmMap.ContainsKey($src)) {
                $script:vtcmMap[$dst] = $script:vtcmMap[$src]
            } elseif ($script:vtcmMap.ContainsKey($dst)) {
                $script:vtcmMap.Remove($dst)
            }
        }
        continue
    }
    # Any other write to a register destroys its derivation. This must come
    # after the spill/reload/copy cases so those keep their meaning.
    if ($t -match '^(r\d+)\s*=' -and $script:vtcmMapMode) {
        $dst = $Matches[1]
        if ($script:vtcmMap.ContainsKey($dst)) { $script:vtcmMap.Remove($dst) }
    }
    # loop0(L,#N) / loop1(L,#N): the body is the next segment. Only a literal
    # count can be expanded statically. A register count is recorded but not
    # expanded: the handler's FNV and copy loops use register counts and are not
    # part of the registered fill/pack form, so they are not counted as stores.
    if ($t -match '^loop[01]\((0x[0-9a-fA-F]+|\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)$') {
        $count = [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        $loopIdx += [pscustomobject]@{ Index = $i; Count = $count; Literal = $true }
        continue
    }
    if ($t -match '^loop[01]\(') {
        $loopIdx += [pscustomobject]@{ Index = $i; Count = 0; Literal = $false }
        continue
    }

    if ($t -match '\bmxclracc\.hf\b') { $mxIdx += $i }
    if ($t -match '\bbias\s*=\s*mxmem2\(([^)]*)\)') {
        $addr = Resolve-AddrAt $Matches[1] $script:mapAt[$i]
        if ($null -eq $addr) { Fail "bias_address_unresolved: line $($segments[$i].Line)" }
        $biasIdx += $i
        $biasAddr += $addr
        $readsUsed[$i] = 1
    }
    if ($t -match '\bcvt\.hf\s*=\s*acc\((r\d+)\)') {
        $cvtIdx += $i
        $script:cvtReg = $Matches[1]
        # Snapshot the Rs register HERE. Reading the final map later lets any
        # unrelated reuse of the same register change the verdict; a minimal
        # reproduction (known-good fixture plus one trailing r0 = ##0x0fff)
        # flipped PASS to FAIL before this was fixed.
        if ($regs.ContainsKey($script:cvtReg)) { $cvtRs = $regs[$script:cvtReg] }
    }
    if ($t -match '\bmxmem\(([^)]*)\)\s*=\s*cvt\b') {
        $addr = Resolve-AddrAt $Matches[1] $script:mapAt[$i]
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
        $addr = Resolve-AddrAt $Matches[1] $script:mapAt[$i]
        $suffix = $Matches[3]
        $width = 128
        if ($suffix -and $suffix -match '\.(b|h|w)$') { $width = 0 }
        # A store inside a loop body is resolved by the loop expansion below,
        # which uses the map at the loop entry and can follow a pointer built
        # inside the body. Failing here would reject a store the expansion is
        # about to resolve, so only a store outside every body fails now.
        if ($null -eq $addr -and -not $inLoopBody.ContainsKey($i)) {
            Fail "vector_address_unresolved: line $($segments[$i].Line)"
        }
        $vecStores += [pscustomobject]@{
            Index = $i; Addr = $addr; Width = $width; Suffix = $suffix
            Text = $t; Line = $segments[$i].Line; LoopCount = 0
        }
    }
    if ($t -match '\b(syncht|barrier|dmwait|dmstart|dmb|isync)\b') {
        $forbiddenIdx += $i
    }
}

# ------------------------------------------------------- loop expansion
# 16 x 128 B lowers to loop0(L,#0x10) ... :endloop0 around a single vmem store.
# The body is every segment from loop0+1 up to the segment carrying :endloop0,
# not just the next one: the compiler reloads the pointer and sets up the
# pattern inside the body. The rules count stores, so a store inside a
# literal-count loop stands for Count stores at base + k*128. A loop whose body
# holds no vector store (the handler's FNV and copy loops) is not part of the
# registered fill/pack form and is ignored.
foreach ($lp in $loopIdx) {
    $bodyEnd = -1
    for ($j = $lp.Index + 1; $j -lt $segments.Count; $j++) {
        if ($segments[$j].Text -match ':endloop[01]') { $bodyEnd = $j; break }
    }
    if ($bodyEnd -lt 0) {
        Fail "loop_end_missing: line $($segments[$lp.Index].Line)"
        continue
    }
    # the store inside the body, if any
    $storeSeg = -1
    for ($j = $lp.Index + 1; $j -lt $bodyEnd; $j++) {
        if ($segments[$j].Text -match '\bvmem\(([^)]*)\)\s*=\s*(v\d+)$') { $storeSeg = $j; break }
    }
    if ($storeSeg -lt 0) { continue }
    if (-not $lp.Literal) {
        Fail "loop_count_not_immediate: line $($segments[$lp.Index].Line) '$($segments[$lp.Index].Text)'"
        continue
    }
    # Resolve the store address from the register state at the store, taking
    # the last assignment to that register inside the body into account.
    $storeText = $segments[$storeSeg].Text
    $null = $storeText -match '\bvmem\(([^)]*)\)\s*=\s*(v\d+)$'
    $addrExpr = $Matches[1]
    # Resolve against the map as it stood at the loop entry: the body's own
    # increment must not leak into the address, and a register reused after the
    # loop must not be read at its later value.
    $loopMap = $script:mapAt[$lp.Index]
    $addr = $null
    if ($addrExpr -match '^(r\d+)\s*\+\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)$') {
        $reg = $Matches[1]
        $off = [Convert]::ToInt32($Matches[2],
            $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        $base = $null
        for ($j = $storeSeg - 1; $j -gt $lp.Index; $j--) {
            if ($segments[$j].Text -match "^$reg\s*=\s*(r\d+)$") {
                $src = $Matches[1]
                if ($loopMap.ContainsKey($src)) { $base = $loopMap[$src] }
                break
            }
            if ($segments[$j].Text -match "^$reg\s*=\s*memw\(r29\+#(0x[0-9a-fA-F]+)\)$") {
                $slot = 's' + [Convert]::ToInt32($Matches[1],
                    $(if ($Matches[1].StartsWith('0x')) { 16 } else { 10 }))
                if ($loopMap.ContainsKey($slot)) { $base = $loopMap[$slot] }
                break
            }
            # The compiler also hoists a pointer computation into the body when
            # the loop is its only user: the guard pointer is built as
            # add(r19,#0x2a00) inside the guard init loop itself.
            if ($segments[$j].Text -match "^$reg\s*=\s*add\((r\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)$") {
                $src = $Matches[1]
                $add = [Convert]::ToInt32($Matches[2],
                    $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
                if ($loopMap.ContainsKey($src)) { $base = $loopMap[$src] + $add }
                break
            }
            # ... and it arrives as the low half of a pair built by combine.
            # The source of that combine may itself be built inside the body, so
            # resolve it the same way rather than only from the loop-entry map.
            if ($segments[$j].Text -match "^r\d+:$($reg.Substring(1))\s*=\s*combine\(([^,]*)\s*,\s*(r\d+)\)$") {
                $src = $Matches[2]
                if ($loopMap.ContainsKey($src)) {
                    $base = $loopMap[$src]
                } else {
                    for ($k = $j - 1; $k -gt $lp.Index; $k--) {
                        if ($segments[$k].Text -match "^$src\s*=\s*add\((r\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)$") {
                            $src2 = $Matches[1]
                            $add2 = [Convert]::ToInt32($Matches[2],
                                $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
                            if ($loopMap.ContainsKey($src2)) { $base = $loopMap[$src2] + $add2 }
                            break
                        }
                    }
                }
                break
            }
        }
        if ($null -eq $base -and $loopMap.ContainsKey($reg)) {
            $base = $loopMap[$reg]
        }
        if ($null -ne $base) { $addr = $base + $off }
    } elseif ($addrExpr -match '^(r\d+)$' -and $loopMap.ContainsKey($Matches[1])) {
        $addr = $loopMap[$Matches[1]]
    }
    if ($null -eq $addr) {
        Fail "loop_body_address_unresolved: line $($segments[$storeSeg].Line)"
        continue
    }
    $hit = @($vecStores | Where-Object { $_.Index -eq $storeSeg })
    if ($hit.Count -ne 1) {
        Fail "loop_body_not_counted: line $($segments[$storeSeg].Line)"
        continue
    }
    if ($env:D8_DBG) { Write-Host "DBG loop=$($lp.Index) reg=$reg addr=$addr r19=$($loopMap['r19']) r24=$($loopMap['r24'])" }
    $hit[0].LoopCount = $lp.Count
    # The address resolved here is the authoritative one: it uses the map as it
    # stood at the loop entry, whereas the parse-time value used the map at the
    # store segment, which can already carry the body's own increment.
    $hit[0].Addr = $addr
    # The body advances the pointer by 128 B per iteration. The increment may
    # sit before or after the store, and it may share the :endloop0 segment.
    $inc = $null
    for ($j = $lp.Index + 1; $j -le $bodyEnd; $j++) {
        if ($segments[$j].Text -match '^(r\d+)\s*=\s*add\(\1\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)') {
            $inc = [Convert]::ToInt32($Matches[2],
                $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
            break
        }
    }
    if ($null -eq $inc) {
        Fail "loop_stride_unresolved: line $($segments[$storeSeg].Line)"
    } elseif ($inc -ne 128) {
        Fail "loop_stride: $inc/128 at line $($segments[$storeSeg].Line)"
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
        $a = Resolve-AddrAt $both.Groups[1].Value $script:mapAt[$i]
        $w = Resolve-AddrAt $both.Groups[2].Value $script:mapAt[$i]
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
            $a = Resolve-AddrAt $act.Groups[1].Value $script:mapAt[$i]
            $w = Resolve-AddrAt $wgt.Groups[1].Value $script:mapAt[$i]
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
    $rs = $cvtRs
    if ($null -eq $rs) {
        Fail "convert_rs_unresolved"
    } elseif ($rs -ne 1) {
        Fail ("convert_rs: 0x{0:x4}/0x0001" -f $rs)
    }
}

# ------------------------------------------------------------- vector stores
# A store inside loop0(L,#N) stands for N stores at base + k*128. Expand the
# recorded stores into the address sequence the rules below expect, keeping the
# segment index so the order rules still work.
$expanded = @()
foreach ($s in $vecStores) {
    if ($s.LoopCount -gt 0) {
        for ($k = 0; $k -lt $s.LoopCount; $k++) {
            $expanded += [pscustomobject]@{
                Index = $s.Index; Addr = $s.Addr + $k * 128; Width = $s.Width
                Suffix = $s.Suffix; Text = $s.Text; Line = $s.Line
                LoopCount = $s.LoopCount; Iteration = $k
            }
        }
    } else {
        $expanded += [pscustomobject]@{
            Index = $s.Index; Addr = $s.Addr; Width = $s.Width
            Suffix = $s.Suffix; Text = $s.Text; Line = $s.Line
            LoopCount = 0; Iteration = 0
        }
    }
}

$firstBias = if ($biasIdx.Count -gt 0) { ($biasIdx | Measure-Object -Minimum).Minimum } else { [int]::MaxValue }
$firstMx = if ($mxIdx.Count -gt 0) { $mxIdx[0] } else { [int]::MaxValue }
$lastMma = if ($mma.Count -gt 0) { ($mma | ForEach-Object { $_.Index } | Measure-Object -Maximum).Maximum } else { -1 }

$packStores = @($expanded | Where-Object { $_.Index -lt $firstBias })
$guardStores = @($expanded | Where-Object { $_.Index -ge $firstBias -and $_.Index -lt $firstMx })
$fillStores = @($expanded | Where-Object { $_.Index -gt $lastMma })
$criticalStores = @($expanded | Where-Object { $_.Index -ge $firstMx -and $_.Index -le $lastMma })

foreach ($s in $expanded) {
    if ($s.Width -ne 128) { Fail "vector_width: line $($s.Line) '$($s.Text)' is narrower than 128 B" }
}
foreach ($s in $criticalStores) { Fail "store_in_critical_interval: line $($s.Line)" }

# Packs: 16 stores at each operand base, in order Ahi/Alo/Whi/Wlo.
#
# When the compiler keeps the pack helper out of line, the handler holds four
# calls and the stores live in the helper's body. The helper is disassembled
# separately and its stores are counted here; the four call sites are matched to
# the four bases in call order.
$packBases = @(0, 2048, 4096, 6144)
if ($PackSymbol) {
    $packText = & $objdump -d --disassemble-symbols=$PackSymbol $ObjectPath 2>&1
    $packSegs = New-Object System.Collections.Generic.List[object]
    foreach ($line in $packText) {
        if ($line -match '^\s*([0-9a-fA-F]+):\s+[0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2} [0-9a-f]{2}\s+[0-9a-f]{8}\s*(.*)$') {
            $rest = (($Matches[2] -replace '[{}]',' ') -replace '\s+',' ').Trim()
            if ($rest) { $packSegs.Add($rest) }
        }
    }
    # The helper takes the destination as its first argument (r0) and advances
    # it by 128 B per iteration; the loop count is the literal in loop0.
    $packCount = 0
    $packStride = $null
    for ($i = 0; $i -lt $packSegs.Count; $i++) {
        if ($packSegs[$i] -match '^loop[01]\((0x[0-9a-fA-F]+|\d+)\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)$') {
            $packCount = [Convert]::ToInt32($Matches[2],
                $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        }
        if ($packSegs[$i] -match '^(r\d+)\s*=\s*add\(\1\s*,\s*(?:##|#)(0x[0-9a-fA-F]+|\d+)\)') {
            $packStride = [Convert]::ToInt32($Matches[2],
                $(if ($Matches[2].StartsWith('0x')) { 16 } else { 10 }))
        }
    }
    if ($packCount -ne 16) { Fail "pack_symbol_count: $packCount/16" }
    if ($null -eq $packStride) {
        Fail "pack_symbol_stride_unresolved"
    } elseif ($packStride -ne 128) {
        Fail "pack_symbol_stride: $packStride/128"
    }
    # The handler must call the helper exactly four times, in order.
    $packCalls = @($segments | Where-Object { $_.Text -match "call 0x[0-9a-f]+ <$PackSymbol>" })
    if ($packCalls.Count -ne 1) { Fail "pack_call_count: $($packCalls.Count)/1" }
} else {
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
}

# Guard: exactly 12 stores at +10752 + k*128, no later guard writes.
if ($guardStores.Count -ne 12) { Fail "guard_init_count: $($guardStores.Count)/12" }
for ($k = 0; $k -lt $guardStores.Count; $k++) {
    if ($guardStores[$k].Addr -ne 10752 + $k * 128) {
        Fail "guard_init_address: $($guardStores[$k].Addr)"
        break
    }
}
foreach ($s in $expanded) {
    if ($s.Index -gt $firstMx -and $s.Addr -ge 10752) {
        Fail "guard_written_after_init: line $($s.Line)"
    }
}

# Fills: out (+8192) always, Wlo (+6144) for B/C, Ahi (+0) for B/D.
#
# Two shapes are accepted. The synthetic fixture emits only the blocks the
# condition needs, so the count is 16 x the mask. A real object emits all three
# loops and guards the Wlo and Ahi ones with a branch, so a static count sees 48
# for every condition; there the requirement is that the three blocks exist and
# the mask names a subset of them. -PackSymbol selects the real-object shape.
$fillMask = @{ A = 1; B = 7; C = 3; D = 5 }[$Condition]
$fillPlan = @()
if ($fillMask -band 1) { $fillPlan += 8192 }
if ($fillMask -band 2) { $fillPlan += 6144 }
if ($fillMask -band 4) { $fillPlan += 0 }
if ($PackSymbol) {
    if ($fillStores.Count -ne 48) {
        Fail "fill_count: $($fillStores.Count)/48 (all three blocks must exist)"
    }
    foreach ($base in @(8192, 6144, 0)) {
        $block = @($fillStores | Where-Object {
            $_.Addr -ge $base -and $_.Addr -lt $base + 2048
        })
        if ($block.Count -ne 16) { Fail "fill_${base}_block: $($block.Count)/16" }
        for ($k = 0; $k -lt $block.Count; $k++) {
            if ($block[$k].Addr -ne $base + $k * 128) {
                Fail "fill_${base}_address: $($block[$k].Addr)/$($base + $k * 128)"
                break
            }
        }
    }
} else {
    $expectedFillCount = 16 * $fillPlan.Count
    if ($fillStores.Count -ne $expectedFillCount) {
        Fail "fill_count: $($fillStores.Count)/$expectedFillCount (condition $Condition)"
    }
    # The blocks must appear in the registered order at the registered bases.
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
