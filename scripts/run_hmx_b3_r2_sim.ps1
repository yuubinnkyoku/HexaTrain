# HMX V81 B3+R2 full-tile precision simulator runner (CPU only).
#
# Source: host_tests/hmx_acc_residual_rpc/hmx_b3_r2_full_tile_sim.c
# Scope: B3+R2 pre-registration CPU-only evidence. Evaluates the 1024-cell
# 32x32 GEMM tile, PRM crouton packing, sequential operand streaming in 12 KiB VTCM,
# and 13 fault injection models. It never touches a device, adb, or the Hexagon SDK.
#
# Exit-code policy (mirrors run_hmx_signed_output_oracle.ps1):
#   - gcc compile failure or a failing gcc run -> that exit code,
#   - a clang RUN failure -> exit 1,
#   - missing clang / missing MSVC environment -> status line only.
#
# Output contract (stable keys):
#   b3_r2_sim_gcc_version=...
#   b3_r2_sim_gcc_compile=...
#   b3_r2_sim_gcc_exit=<n>
#   b3_r2_sim_clang=<status>
#   b3_r2_sim=PASS|FAIL
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_b3_r2_full_tile_sim.c"
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Write-Host "b3_r2_sim=FAIL (missing source $Source)"
    exit 1
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("hmx-b3-r2-sim-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$gccExit = 1
$clangStatus = "SKIP (not evaluated)"
$clangRunFailed = $false

try {
    # ---------------- gcc (PATH): gate-owning compiler ----------------
    $gccCommand = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gccCommand) {
        Write-Host "b3_r2_sim_gcc=FAIL (gcc not found on PATH)"
    } else {
        Write-Host ("b3_r2_sim_gcc_version=" +
            ((& $gccCommand.Source --version | Select-Object -First 1)))
        Write-Host ("b3_r2_sim_gcc_triple=" +
            ((& $gccCommand.Source -dumpmachine)))
        $gccExe = Join-Path $TempDir "hmx_b3_r2_full_tile_sim_gcc.exe"
        Write-Host ("b3_r2_sim_gcc_compile=gcc -std=gnu11 -O2 -Wall -Wextra" +
            " -Werror -ffp-contract=off -I host_tests\hmx_acc_residual_rpc -o <temp>\hmx_b3_r2_full_tile_sim_gcc.exe" +
            " host_tests\hmx_acc_residual_rpc\hmx_b3_r2_full_tile_sim.c -lm")
        & $gccCommand.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
            "-o" $gccExe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "b3_r2_sim_gcc=FAIL (compile exit $LASTEXITCODE)"
        } else {
            & $gccExe
            $gccExit = $LASTEXITCODE
            Write-Host "b3_r2_sim_gcc_exit=$gccExit"
        }
    }

    # ---------------- clang: cross-check ----------------
    $clangCommand = Get-Command clang -ErrorAction SilentlyContinue
    if (-not $clangCommand) {
        $clangStatus = "SKIP (clang not found on PATH)"
    } else {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} `
            "Microsoft Visual Studio\Installer\vswhere.exe"
        $vcvars = $null
        if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
            $vsInstall = & $vswhere -latest -prerelease `
                -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath
            if ($vsInstall) {
                $candidate = Join-Path $vsInstall `
                    "VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    $vcvars = $candidate
                }
            }
        }
        if (-not $vcvars) {
            $clangStatus = "SKIP (MSVC environment not found)"
        } else {
            $clangExe = Join-Path $TempDir "hmx_b3_r2_full_tile_sim_clang.exe"
            $clangTriple = "x86_64-pc-windows-msvc19.44.35229"
            $cmd = ('@echo off & call "{0}" >nul 2>&1 & "{1}" --version & ' +
                    '"{1}" -dumpmachine & ' +
                    '"{1}" -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off ' +
                    '-I "{2}" -rtlib=compiler-rt -o "{3}" "{4}"') -f `
                $vcvars, $clangCommand.Source,
                (Join-Path $Root "host_tests\hmx_acc_residual_rpc"),
                $clangExe, $Source
            $buildOut = & cmd.exe /c $cmd 2>&1
            $buildExit = $LASTEXITCODE
            $clangVersionLine = ($buildOut | Select-Object -First 1)
            $clangTripleLine = ($buildOut | Select-Object -Skip 1 -First 1)
            Write-Host "b3_r2_sim_clang_version=$clangVersionLine"
            Write-Host "b3_r2_sim_clang_triple=$clangTripleLine"
            Write-Host ("b3_r2_sim_clang_compile=clang -std=gnu11 -O2 -Wall -Wextra" +
                " -Werror -ffp-contract=off -I host_tests\hmx_acc_residual_rpc -rtlib=compiler-rt -o <temp>\hmx_b3_r2_full_tile_sim_clang.exe" +
                " host_tests\hmx_acc_residual_rpc\hmx_b3_r2_full_tile_sim.c")
            if ($buildExit -ne 0) {
                $clangStatus = "SKIP (clang build failed: $buildOut)"
            } else {
                & $clangExe
                $clangRunExit = $LASTEXITCODE
                if ($clangRunExit -eq 0) {
                    $clangStatus = "PASS (sim exit 0, triple=$clangTripleLine)"
                } else {
                    $clangStatus = "FAIL (sim exit $clangRunExit)"
                    $clangRunFailed = $true
                }
            }
        }
    }
} finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "b3_r2_sim_clang=$clangStatus"
$exitCode = $gccExit
if ($clangRunFailed) { $exitCode = 1 }
if ($exitCode -eq 0) {
    Write-Host "b3_r2_sim=PASS"
} else {
    Write-Host "b3_r2_sim=FAIL (gcc exit $gccExit; clang: $clangStatus)"
}
exit $exitCode
