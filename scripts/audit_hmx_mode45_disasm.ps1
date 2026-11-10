# Static disassembly audit for the frozen R2 S1/S2 (mode 45) instruction
# sequence (docs/hmx-v81-r2-device-prereg.md sections 6-8,
# docs/hmx-v81-r2-device-implementation-plan.md section 4).
#
# This audits the BUILT DSP skel .so - not the source. It disassembles one
# symbol with hexagon-llvm-objdump, counts the registered mnemonics
# (mxclracc / MMA packets / bias loads / converts / stores / HVX vsub),
# extracts the four Rs immediates from the instructions feeding
# "cvt.hf = acc(rN)", and checks the pre-registered instruction ORDER.
# Optionally it cross-checks the DSP counters from a saved device log.
#
# Counter counts and static-audit counts must agree; a mismatch is a
# failure, never a skip. This script does not run anything on a device.
#
# Examples:
#   .\scripts\audit_hmx_mode45_disasm.ps1
#   .\scripts\audit_hmx_mode45_disasm.ps1 -Symbol run_mode44_one_trial `
#       -ExpectedMxclracc 1 -ExpectedMma 1 -ExpectedBiasLoads 2 `
#       -ExpectedConvert 1 -ExpectedStore 1 -ExpectedVsub 0 `
#       -ExpectedRs "0x0001" -SkipOrderCheck
[CmdletBinding()]
param(
    [string]$SoPath = "",
    [string]$ObjdumpPath = "",
    [string]$Symbol = "run_mode45_one_trial",
    [int]$ExpectedMxclracc = 2,
    [int]$ExpectedMma = 2,
    [int]$ExpectedBiasLoads = 3,
    [int]$ExpectedConvert = 4,
    [int]$ExpectedStore = 4,
    [int]$ExpectedVsub = 2,
    [string]$ExpectedRs = "0x0001,0x2000,0x0001,0x1000",
    [string]$DeviceLog = "",
    [switch]$SkipOrderCheck
)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$Root = Split-Path -Parent $PSScriptRoot
if (-not $SoPath) {
    $SoPath = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hexagon_ReleaseG_toolv19_v81\ship\libhexatrain_hmx_acc_residual_skel.so"
}

$script:Failed = 0
function Fail([string]$Message) {
    Write-Host "FAIL $Message"
    $script:Failed++
}

function Complete-Audit {
    if ($script:Failed -gt 0) {
        Write-Host "mode45_audit=FAIL"
        exit 1
    }
    Write-Host "mode45_audit=PASS"
    exit 0
}

# ---------------------------------------------------------------- 1. inputs
if (-not (Test-Path -LiteralPath $SoPath)) {
    Fail "so_missing: $SoPath"
    Complete-Audit
}
Write-Host "mode45_audit_so=$SoPath"
Write-Host ("mode45_audit_sha256=" + (Get-FileHash -LiteralPath $SoPath -Algorithm SHA256).Hash.ToLower())
Write-Host "mode45_audit_symbol=$Symbol"

if (-not $ObjdumpPath) {
    $candidates = @()
    if ($env:HEXAGON_SDK_ROOT) {
        $candidates += Get-ChildItem -Path (Join-Path $env:HEXAGON_SDK_ROOT "tools\HEXAGON_Tools\*\Tools\bin\hexagon-llvm-objdump.exe") -ErrorAction SilentlyContinue
    }
    $candidates += Get-ChildItem -Path "C:\Qualcomm\Hexagon_SDK\*\tools\HEXAGON_Tools\*\Tools\bin\hexagon-llvm-objdump.exe" -ErrorAction SilentlyContinue
    $hit = $candidates | Where-Object { $_ } | Select-Object -First 1
    if ($hit) { $ObjdumpPath = $hit.FullName }
}
if (-not $ObjdumpPath -or -not (Test-Path -LiteralPath $ObjdumpPath)) {
    Fail "objdump_missing: hexagon-llvm-objdump.exe not found"
    Complete-Audit
}
Write-Host "mode45_audit_objdump=$ObjdumpPath"

# ------------------------------------------------------- 2. symbol present
$symLines = & $ObjdumpPath --syms $SoPath
$escape = [regex]::Escape($Symbol)
$symbolOk = $false
foreach ($line in $symLines) {
    if ($line -match ("^[0-9a-fA-F]+\s+\S+\s+F\s+\S+\s+[0-9a-fA-F]+\s+" + $escape + "\s*$")) {
        $symbolOk = $true
        break
    }
}
if (-not $symbolOk) {
    Fail "symbol_missing: $Symbol is not a function in $(Split-Path -Leaf $SoPath)"
    Complete-Audit
}
Write-Host "mode45_audit_symbol_present=1"

# ----------------------------------------------------- 3. disassemble + parse
$disLines = & $ObjdumpPath -d --disassemble-symbols=$Symbol $SoPath

# Flat instruction-segment list. Each objdump line may hold a packet of two
# instructions separated by ';'. Segments keep their packet id so an MMA
# packet (activation load + weight load) can be recognised as one packet.
$segments = New-Object System.Collections.Generic.List[object]
$packetId = -1
$inPacket = $false
$lineIndex = 0
foreach ($line in $disLines) {
    if ($line -notmatch '^\s*([0-9a-fA-F]+):\t(.*)$') { continue }
    $rest = $Matches[2]
    $fields = $rest -split "`t"
    if ($fields.Count -ge 2) {
        $text = ($fields[1..($fields.Count - 1)] -join "`t")
    } else {
        $text = $fields[0]
    }
    $opened = $text -match '\{'
    $closed = $text -match '\}'
    if ($opened -and -not $inPacket) {
        $packetId++
        $inPacket = $true
    }
    $myPacket = $packetId
    foreach ($raw in ($text -split ';')) {
        $seg = ($raw -replace '\s+', ' ').Trim()
        if (-not $seg) { continue }
        $segments.Add([pscustomobject]@{
            Text   = $seg
            Packet = $myPacket
            Line   = $lineIndex
        })
    }
    if ($closed) { $inPacket = $false }
    $lineIndex++
}
if ($segments.Count -eq 0) {
    Fail "disasm_empty: no instruction lines parsed for $Symbol"
    Complete-Audit
}
Write-Host "mode45_audit_instructions=$($segments.Count)"

# ------------------------------------------------------------------ 4. counts
$mxIdx = New-Object System.Collections.Generic.List[int]
$actIdx = New-Object System.Collections.Generic.List[int]
$wgtIdx = New-Object System.Collections.Generic.List[int]
$biasIdx = New-Object System.Collections.Generic.List[int]
$cvtIdx = New-Object System.Collections.Generic.List[int]
$storeIdx = New-Object System.Collections.Generic.List[int]
$vsubIdx = New-Object System.Collections.Generic.List[int]
$mmaPackets = @{}

for ($i = 0; $i -lt $segments.Count; $i++) {
    $t = $segments[$i].Text
    if ($t -match '\bmxclracc\.hf\b') { $mxIdx.Add($i) }
    if ($t -match '\bactivation\.hf = mxmem\(') {
        $actIdx.Add($i)
        $p = $segments[$i].Packet
        if ($p -ge 0) { $mmaPackets[$p] = 1 }
    }
    if ($t -match '\bweight\.hf = mxmem\(') {
        $wgtIdx.Add($i)
        $p = $segments[$i].Packet
        if ($p -ge 0) { $mmaPackets[$p] = $mmaPackets[$p] + 1 }
    }
    if ($t -match '\bbias = mxmem2\(') { $biasIdx.Add($i) }
    if ($t -match '\bcvt\.hf = acc\(') { $cvtIdx.Add($i) }
    if ($t -match '\bmxmem\([^)]*\) = cvt\b') { $storeIdx.Add($i) }
    if ($t -match '\bvsub\b') { $vsubIdx.Add($i) }
}
# An MMA packet is one packet carrying both an activation load and a weight
# load (see probe_dsp.c: "{ activation.hf = mxmem(...) weight.hf = mxmem(...) }").
$mmaCount = 0
foreach ($k in $mmaPackets.Keys) { if ($mmaPackets[$k] -ge 2) { $mmaCount++ } }

$counts = [ordered]@{
    mxclracc   = $mxIdx.Count
    mma        = $mmaCount
    activation = $actIdx.Count
    weight     = $wgtIdx.Count
    bias_loads = $biasIdx.Count
    convert    = $cvtIdx.Count
    store      = $storeIdx.Count
    vsub       = $vsubIdx.Count
}
foreach ($k in $counts.Keys) { Write-Host "mode45_audit_count_$k=$($counts[$k])" }

$countsOk = ($mxIdx.Count -eq $ExpectedMxclracc) -and
            ($mmaCount -eq $ExpectedMma) -and
            ($actIdx.Count -eq $ExpectedMma) -and
            ($wgtIdx.Count -eq $ExpectedMma) -and
            ($biasIdx.Count -eq $ExpectedBiasLoads) -and
            ($cvtIdx.Count -eq $ExpectedConvert) -and
            ($storeIdx.Count -eq $ExpectedStore) -and
            ($vsubIdx.Count -eq $ExpectedVsub)
if (-not $countsOk) {
    Fail "counts: got mxclracc=$($mxIdx.Count)/$ExpectedMxclracc mma=$mmaCount/$ExpectedMma act=$($actIdx.Count)/$ExpectedMma wgt=$($wgtIdx.Count)/$ExpectedMma bias=$($biasIdx.Count)/$ExpectedBiasLoads convert=$($cvtIdx.Count)/$ExpectedConvert store=$($storeIdx.Count)/$ExpectedStore vsub=$($vsubIdx.Count)/$ExpectedVsub"
}
Write-Host ("mode45_audit_counts=" + $(if ($countsOk) { "PASS" } else { "FAIL" }))

# ----------------------------------------------------------------- 5. Rs audit
$expectedList = @()
foreach ($item in ($ExpectedRs -split ',')) {
    $tok = $item.Trim()
    if (-not $tok) { continue }
    if ($tok -match '^0x[0-9a-fA-F]+$') {
        $expectedList += ("0x{0:x4}" -f [Convert]::ToInt32($tok.Substring(2), 16))
    } else {
        $expectedList += ("0x{0:x4}" -f [Convert]::ToInt32($tok, 10))
    }
}
$rsOk = $true
$observedList = @()
$cvtN = 0
for ($i = 0; $i -lt $segments.Count; $i++) {
    if ($segments[$i].Text -notmatch '\bcvt\.hf = acc\((r\d+)\)') { continue }
    $reg = $Matches[1]
    $rs = $null
    for ($j = $i - 1; $j -ge 0; $j--) {
        $assign = [regex]::Match($segments[$j].Text,
            ("\b" + [regex]::Escape($reg) + "\s*=\s*(##?0x[0-9a-fA-F]+)"))
        if ($assign.Success) {
            $hex = $assign.Groups[1].Value -replace '^##?', ''
            $rs = "0x{0:x4}" -f [Convert]::ToInt32($hex, 16)
            break
        }
    }
    if ($null -eq $rs) {
        Fail "rs_cvt${cvtN}: no earlier immediate assignment for $reg (segment $i)"
        $rsOk = $false
        $cvtN++
        continue
    }
    Write-Host "mode45_audit_cvt$cvtN=acc($reg) rs=$rs"
    $observedList += $rs
    $cvtN++
}
$joinedObserved = ($observedList -join ',')
$joinedExpected = ($expectedList -join ',')
if ($observedList.Count -ne $expectedList.Count -or $joinedObserved -ne $joinedExpected) {
    Fail "rs: got [$joinedObserved] want [$joinedExpected]"
    $rsOk = $false
}
Write-Host ("mode45_audit_rs=" + $(if ($rsOk) { "PASS" } else { "FAIL" }))

# -------------------------------------------------------------- 6. order audit
if ($SkipOrderCheck) {
    Write-Host "mode45_audit_order=SKIP"
} else {
    $orderOk = $true
    $require = [ordered]@{
        bias_before_convert1 = ($biasIdx.Count -ge 3 -and $cvtIdx.Count -ge 1 -and
            $biasIdx[0] -lt $cvtIdx[0] -and $biasIdx[1] -lt $cvtIdx[0] -and $biasIdx[2] -lt $cvtIdx[0])
        mxclracc1_lt_act1    = ($mxIdx.Count -ge 1 -and $actIdx.Count -ge 1 -and $mxIdx[0] -lt $actIdx[0])
        act1_lt_convert1     = ($actIdx.Count -ge 1 -and $cvtIdx.Count -ge 1 -and $actIdx[0] -lt $cvtIdx[0])
        convert1_lt_store1   = ($cvtIdx.Count -ge 1 -and $storeIdx.Count -ge 1 -and $cvtIdx[0] -lt $storeIdx[0])
        store1_lt_convert2   = ($storeIdx.Count -ge 1 -and $cvtIdx.Count -ge 2 -and $storeIdx[0] -lt $cvtIdx[1])
        convert2_lt_store2   = ($cvtIdx.Count -ge 2 -and $storeIdx.Count -ge 2 -and $cvtIdx[1] -lt $storeIdx[1])
        store2_lt_mxclracc2  = ($storeIdx.Count -ge 2 -and $mxIdx.Count -ge 2 -and $storeIdx[1] -lt $mxIdx[1])
        mxclracc2_lt_act2    = ($mxIdx.Count -ge 2 -and $actIdx.Count -ge 2 -and $mxIdx[1] -lt $actIdx[1])
        act2_lt_convert3     = ($actIdx.Count -ge 2 -and $cvtIdx.Count -ge 3 -and $actIdx[1] -lt $cvtIdx[2])
        convert3_lt_store3   = ($cvtIdx.Count -ge 3 -and $storeIdx.Count -ge 3 -and $cvtIdx[2] -lt $storeIdx[2])
        store3_lt_convert4   = ($storeIdx.Count -ge 3 -and $cvtIdx.Count -ge 4 -and $storeIdx[2] -lt $cvtIdx[3])
        convert4_lt_store4   = ($cvtIdx.Count -ge 4 -and $storeIdx.Count -ge 4 -and $cvtIdx[3] -lt $storeIdx[3])
    }
    foreach ($k in $require.Keys) {
        if (-not $require[$k]) {
            Fail "order_$k"
            $orderOk = $false
        }
    }
    Write-Host ("mode45_audit_order_observed=bias=[" + ($biasIdx -join ',') +
        "] mxclracc=[" + ($mxIdx -join ',') +
        "] act=[" + ($actIdx -join ',') +
        "] cvt=[" + ($cvtIdx -join ',') +
        "] store=[" + ($storeIdx -join ',') +
        "] vsub=[" + ($vsubIdx -join ',') + "]")
    Write-Host ("mode45_audit_order=" + $(if ($orderOk) { "PASS" } else { "FAIL" }))
}

# --------------------------------------------------- 7. device-log cross-check
if ($DeviceLog) {
    $crossOk = $true
    if (-not (Test-Path -LiteralPath $DeviceLog)) {
        Fail "device_log_missing: $DeviceLog"
        $crossOk = $false
    } else {
        $logLines = Get-Content -LiteralPath $DeviceLog
        $evLine = $logLines | Where-Object {
            $_ -match 'r2_evidence\s+mxclracc=\d+\s+mma=\d+\s+convert=\d+\s+store=\d+\s+bias_loads=\d+\s+recon_loops=\d+'
        } | Select-Object -First 1
        if (-not $evLine) {
            Fail "device_evidence_line_missing: no 'r2_evidence mxclracc=... recon_loops=...' in $DeviceLog"
            $crossOk = $false
        } elseif ($evLine -match 'r2_evidence\s+mxclracc=(\d+)\s+mma=(\d+)\s+convert=(\d+)\s+store=(\d+)\s+bias_loads=(\d+)\s+recon_loops=(\d+)') {
            $dev = @{
                mxclracc   = [int]$Matches[1]
                mma        = [int]$Matches[2]
                convert    = [int]$Matches[3]
                store      = [int]$Matches[4]
                bias_loads = [int]$Matches[5]
                recon_loops = [int]$Matches[6]
            }
            $pairs = @(
                @('mxclracc', $dev.mxclracc, $counts.mxclracc),
                @('mma', $dev.mma, $counts.mma),
                @('convert', $dev.convert, $counts.convert),
                @('store', $dev.store, $counts.store),
                @('bias_loads', $dev.bias_loads, $counts.bias_loads),
                @('recon_loops', $dev.recon_loops, $counts.vsub)
            )
            foreach ($p in $pairs) {
                if ($p[1] -ne $p[2]) {
                    Fail "device_$($p[0]): device=$($p[1]) static=$($p[2])"
                    $crossOk = $false
                }
            }
        }
        $regionLine = $logLines | Where-Object {
            $_ -match 'r2_regions\b.*recon_vectors=\d+'
        } | Select-Object -First 1
        if ($regionLine -and $regionLine -match 'recon_vectors=(\d+)') {
            Write-Host "mode45_audit_device_recon_vectors=$($Matches[1])"
        }
    }
    Write-Host ("mode45_audit_device_crosscheck=" + $(if ($crossOk) { "PASS" } else { "FAIL" }))
}

Complete-Audit
