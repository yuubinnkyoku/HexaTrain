# Build the D8 diagnostic skel.
#
# This reproduces the link recipe recorded in the worklog: the two project
# objects, the 12 libc math objects extracted from the PIC archive, then
# libgcc, libqcc and libqurt in that order (libqurt last so the sys_* symbols
# resolve). The PIC libc archive is required: the non-PIC one fails to link a
# shared object with R_HEX_32_6_X relocation errors.
#
# Usage: pwsh -File scripts\build_hmx_d8_skel.ps1 -Stamp 0x0D1A6004 -Out diag3rev8.so -Diag
param(
    [Parameter(Mandatory = $true)][string]$Stamp,
    [Parameter(Mandatory = $true)][string]$Out,
    [switch]$Diag
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $sdk = "C:\Qualcomm\Hexagon_SDK\6.6.0.0"
    $bin = Join-Path $sdk "tools\HEXAGON_Tools\19.0.07\Tools\bin"
    $env:PATH = "$bin;$env:PATH"

    $work = Join-Path $Root "build\d8skel\obj-$Stamp"
    New-Item -ItemType Directory -Force -Path $work | Out-Null

    $inc = @(
        "-Ihost_tests\hmx_acc_residual_rpc", "-Ibuild\d8gen",
        "-I$sdk\incs", "-I$sdk\incs\stddef",
        "-I$sdk\rtos\qurt\computev81\include", "-I$sdk\rtos\qurt\computev81\include\qurt",
        "-I$sdk\ipc\fastrpc\include",
        "-I$sdk\libs\qhl\include", "-I$sdk\libs\qhl\include\qhl",
        "-I$sdk\libs\qhl\include\HAP", "-I$sdk\libs\qhl\include\qurt",
        "-I$sdk\libs\qhl\include\sys", "-I$sdk\libs\qhl\include\stdlib",
        "-I$sdk\libs\qhl\include\string", "-I$sdk\libs\qhl\include\math",
        "-I$sdk\libs\qhl\include\stddef", "-I$sdk\libs\qhl\include\stdint",
        "-I$sdk\libs\qhl\include\hexagon", "-I$sdk\libs\qhl\include\hvx"
    )
    $defs = @(
        '-fPIC', '-O2', '-G0', '-mv81', '-mhmx', '-mhvx', '-mhvx-length=128b',
        '-D__HEXAGON_ARCH__=81', "-DHMX_PROBE_BUILD_ID=$Stamp"
    )
    if ($Diag) { $defs += '-DHMX_D8_DIAG' }
    $common = $defs + $inc

    & hexagon-clang @common -c -o "$work\probe_dsp.o" "host_tests\hmx_acc_residual_rpc\probe_dsp.c"
    if ($LASTEXITCODE -ne 0) { throw "probe_dsp.c compile failed" }
    & hexagon-clang @common -c -o "$work\skel.o" "build\d8gen\hexatrain_hmx_acc_residual_skel.c"
    if ($LASTEXITCODE -ne 0) { throw "skel compile failed" }

    $libcpic = "$sdk\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib\v81\G0\pic\libc.a"
    $ex = "$work\libc-pic"
    New-Item -ItemType Directory -Force -Path $ex | Out-Null
    Push-Location $ex
    try {
        & hexagon-ar.exe x $libcpic frexp.o ldexp.o floor.o fmod.o xdint.o xdscale.o `
            xdsign.o xdtest.o xdunscal.o xferaise.o xvalues.o xdnorm.o
        if ($LASTEXITCODE -ne 0) { throw "ar x failed" }
    } finally { Pop-Location }

    $libgcc = "$sdk\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib\v81\G0\libgcc.a"
    $libqcc = "$sdk\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\lib\v81\G0\libqcc.a"
    $libqurt = "$sdk\rtos\qurt\computev81\lib\libqurt.a"
    $objs = Get-ChildItem $ex -Filter '*.o' | ForEach-Object { $_.FullName }
    $outPath = Join-Path $Root "build\d8skel\$Out"

    function Invoke-Link([string]$Target, [string[]]$ExtraDefs) {
        $defs2 = $defs + $ExtraDefs
        $c2 = $defs2 + $inc
        & hexagon-clang @c2 -c -o "$work\probe_dsp.o" "host_tests\hmx_acc_residual_rpc\probe_dsp.c"
        if ($LASTEXITCODE -ne 0) { throw "probe_dsp.c compile failed" }
        $la = @('-shared', '-Wl,-Bsymbolic', '-o', $Target, "$work\probe_dsp.o", "$work\skel.o") +
              $objs + @($libgcc, $libqcc, $libqurt)
        & hexagon-clang @la
        if ($LASTEXITCODE -ne 0) { throw "link failed" }
    }

    # The image-window bounds are the anchor's offsets from address 0 and from
    # the image end. They depend on the link, so they cannot be hardcoded: the
    # anchor moved from 0x6318 to 0xA318 between two builds of the same source
    # when the diagnostic block stopped using a stack local. Pass 1 links with
    # placeholder bounds, reads the real layout, and pass 2 relinks with the
    # measured values. The bounds are immediates in a comparison, so they do not
    # move the layout - pass 2 is checked to confirm that.
    $probe = Join-Path $work "layout-probe.so"
    Invoke-Link $probe @('-DHMX_D8_IMAGE_LO_OFFSET=0', '-DHMX_D8_IMAGE_HI_OFFSET=1')

    $nm = & hexagon-nm.exe -S $probe 2>&1
    $anchorLine = ($nm | Select-String -Pattern '\bhmx_d8_diag_attr$').Line
    if (-not $anchorLine) { throw "anchor symbol hmx_d8_diag_attr not found" }
    $anchor = [Convert]::ToInt32(($anchorLine.Trim() -split '\s+')[0], 16)

    $hdr = & hexagon-llvm-objdump.exe -h $probe 2>&1
    $bssLine = ($hdr | Select-String -Pattern '^\s+\d+\s+\.bss\s').Line
    if (-not $bssLine) { throw ".bss section not found" }
    $bssParts = $bssLine.Trim() -split '\s+'
    $bssSize = [Convert]::ToInt32($bssParts[2], 16)
    $bssAddr = [Convert]::ToInt32($bssParts[3], 16)
    $imageEnd = $bssAddr + $bssSize

    $lo = 0 - $anchor          # offset to address 0: covers the ELF header too
    $hi = $imageEnd - $anchor

    Write-Host ("layout anchor=0x{0:X} bss=[0x{1:X},0x{2:X}) lo={3} hi=0x{4:X}" -f `
        $anchor, $bssAddr, $imageEnd, $lo, $hi)

    Invoke-Link $outPath @("-DHMX_D8_IMAGE_LO_OFFSET=$lo", "-DHMX_D8_IMAGE_HI_OFFSET=$hi")

    # Confirm pass 2 did not move the layout the bounds were measured from.
    $nm2 = & hexagon-nm.exe -S $outPath 2>&1
    $anchor2 = [Convert]::ToInt32((($nm2 | Select-String -Pattern '\bhmx_d8_diag_attr$').Line.Trim() -split '\s+')[0], 16)
    $hdr2 = & hexagon-llvm-objdump.exe -h $outPath 2>&1
    $bss2 = ($hdr2 | Select-String -Pattern '^\s+\d+\s+\.bss\s').Line.Trim() -split '\s+'
    $end2 = [Convert]::ToInt32($bss2[3], 16) + [Convert]::ToInt32($bss2[2], 16)
    if ($anchor2 -ne $anchor -or $end2 -ne $imageEnd) {
        throw ("layout moved between passes: anchor 0x{0:X}->0x{1:X}, end 0x{2:X}->0x{3:X}" -f `
            $anchor, $anchor2, $imageEnd, $end2)
    }

    $item = Get-Item $outPath
    $hash = (Get-FileHash $outPath -Algorithm SHA256).Hash.ToLower()
    Write-Host "built=$($item.Name) size=$($item.Length) sha256=$hash"
    Write-Host "window lo=$lo hi=$hi anchor=0x$('{0:X}' -f $anchor) image_end=0x$('{0:X}' -f $imageEnd)"
} finally { Pop-Location }
