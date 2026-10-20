# FP16 hi/lo split simulator runner for
# host_tests/hmx_acc_residual_rpc/hmx_fp16x2_split_sim.c
# (Phase E-2 Path B measurements; CPU only, no SDK/device).
#
# Self-contained, no parameters. It:
#   1. compiles the simulator with gcc from PATH into a temp directory,
#   2. runs it and prints its output (the delimiter-stable
#      split_*/synth_* stat lines),
#   3. builds and runs it a second time with clang when clang is
#      available (MSVC target + -rtlib=compiler-rt, same vcvars pattern
#      as run_hmx_selftest.ps1),
#   4. exits with the simulator's exit code (gcc governs; a failing
#      clang RUN is fatal, a clang BUILD problem is a status line only).
#
# Exit-code policy (mirrors run_hmx_selftest.ps1):
#   - gcc compile failure or a failing gcc run -> that exit code,
#   - a clang RUN failure -> exit 1,
#   - missing clang / missing MSVC environment -> status line only.
#
# Final lines relied on by callers:
#   fp16x2_split_sim_gcc_exit=<n>
#   fp16x2_split_sim=PASS|FAIL
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_fp16x2_split_sim.c"
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Write-Host "fp16x2_split_sim=FAIL (missing source $Source)"
    exit 1
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("fp16x2-split-sim-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$gccExit = 1
$clangStatus = "SKIP (not evaluated)"
$clangRunFailed = $false

try {
    # ---------------- gcc (PATH): gate-owning compiler ----------------
    $gccCommand = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gccCommand) {
        Write-Host "fp16x2_split_sim_gcc=FAIL (gcc not found on PATH)"
    } else {
        Write-Host ("fp16x2_split_sim_gcc_version=" +
            ((& $gccCommand.Source --version | Select-Object -First 1)))
        Write-Host ("fp16x2_split_sim_gcc_triple=" +
            ((& $gccCommand.Source -dumpmachine)))
        $gccExe = Join-Path $TempDir "hmx_fp16x2_split_sim_gcc.exe"
        Write-Host ("fp16x2_split_sim_gcc_compile=gcc -std=gnu11 -O2" +
            " -Wall -Wextra -Werror -o <temp>\hmx_fp16x2_split_sim_gcc.exe" +
            " host_tests\hmx_acc_residual_rpc\hmx_fp16x2_split_sim.c -lm")
        & $gccCommand.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-o" $gccExe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "fp16x2_split_sim_gcc=FAIL (compile exit $LASTEXITCODE)"
        } else {
            & $gccExe
            $gccExit = $LASTEXITCODE
            Write-Host "fp16x2_split_sim_gcc_exit=$gccExit"
        }
    }

    # ---------------- clang (optional second compiler) ----------------
    $clangCommand = Get-Command clang -ErrorAction SilentlyContinue
    $clangPath = ""
    if ($clangCommand) { $clangPath = $clangCommand.Source }
    if (-not $clangPath) {
        foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if (-not $base) { continue }
            $candidate = Join-Path $base "LLVM\bin\clang.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $clangPath = $candidate
                break
            }
        }
    }
    if (-not $clangPath) {
        $clangStatus = "SKIP (clang not found)"
    } else {
        # The default clang target is x86_64-pc-windows-msvc; building it
        # needs the VC / UCRT environment from vcvars64.bat.
        $vcvars = ""
        $pf86 = ${env:ProgramFiles(x86)}
        if ($pf86) {
            $vswhere = Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe"
            if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
                $found = & $vswhere -latest -products * `
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                    -find "VC\Auxiliary\Build\vcvars64.bat"
                if ($found) { $vcvars = (@($found) | Select-Object -First 1) }
            }
            if (-not $vcvars) {
                foreach ($edition in @("BuildTools", "Enterprise", "Community", "Professional")) {
                    $candidate = Join-Path $pf86 `
                        "Microsoft Visual Studio\2022\$edition\VC\Auxiliary\Build\vcvars64.bat"
                    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                        $vcvars = $candidate
                        break
                    }
                }
            }
        }
        if (-not $vcvars) {
            $clangStatus = "SKIP (vcvars64.bat not found; cannot build the MSVC target)"
        } else {
            # Import the MSVC environment into this script's own process only.
            $vcEnvLines = & cmd /c "call `"$vcvars`" >nul 2>&1 && set"
            if ($LASTEXITCODE -ne 0) {
                $clangStatus = "SKIP (vcvars64.bat failed with exit $LASTEXITCODE)"
            } else {
                foreach ($line in $vcEnvLines) {
                    if ($line -match '^([^=]+)=(.*)$') {
                        Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2]
                    }
                }
                Write-Host ("fp16x2_split_sim_clang_version=" +
                    ((& $clangPath --version | Select-Object -First 1)))
                $clangTriple = (& $clangPath -print-effective-triple)
                Write-Host "fp16x2_split_sim_clang_triple=$clangTriple"
                $clangExe = Join-Path $TempDir "hmx_fp16x2_split_sim_clang.exe"
                Write-Host ("fp16x2_split_sim_clang_compile=clang -std=gnu11" +
                    " -O2 -Wall -Wextra -Werror -rtlib=compiler-rt" +
                    " -o <temp>\hmx_fp16x2_split_sim_clang.exe" +
                    " host_tests\hmx_acc_residual_rpc\hmx_fp16x2_split_sim.c")
                & $clangPath "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
                    "-rtlib=compiler-rt" "-o" $clangExe $Source
                if ($LASTEXITCODE -ne 0) {
                    $clangStatus = "BUILD_FAILED (exit $LASTEXITCODE; gcc result governs)"
                } else {
                    & $clangExe
                    $clangExit = $LASTEXITCODE
                    if ($clangExit -eq 0) {
                        $clangStatus = "PASS (sim exit 0, triple=$clangTriple)"
                    } else {
                        $clangStatus = "FAIL (sim exit $clangExit, triple=$clangTriple)"
                        $clangRunFailed = $true
                    }
                }
            }
        }
    }
} finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "fp16x2_split_sim_clang=$clangStatus"
$exitCode = $gccExit
if ($clangRunFailed) { $exitCode = 1 }
if ($exitCode -eq 0) {
    Write-Host "fp16x2_split_sim=PASS"
} else {
    Write-Host "fp16x2_split_sim=FAIL (gcc exit $gccExit; clang: $clangStatus)"
}
exit $exitCode
