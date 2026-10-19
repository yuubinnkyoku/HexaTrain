# FP16 oracle host self-test runner for
# host_tests/hmx_acc_residual_rpc/probe_host_selftest.c.
#
# Self-contained, no parameters. It:
#   1. compiles the self-test with gcc from PATH into a temp directory,
#   2. runs it and prints its output,
#   3. builds and runs it a second time with clang when clang is available
#      (MSVC target, which is the only clang target on this platform whose
#      native _Float16 cross-check actually executes correctly; the MinGW
#      target links a libgcc __truncdfhf2 whose result register disagrees
#      with clang's call site and must be treated as broken),
#   4. exits with the self-test's exit code.
#
# Exit-code policy:
#   - gcc compile failure or a failing gcc self-test run -> that exit code.
#   - a clang self-test RUN that fails -> exit 1 (a real mismatch is fatal).
#   - a clang BUILD problem (missing clang, missing MSVC environment) is
#     reported as a status line only and does not fail the suite; gcc
#     governs the exit code in that case.
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\probe_host_selftest.c"
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Write-Host "hmx_selftest=FAIL (missing source $Source)"
    exit 1
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("hmx-selftest-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$gccExit = 1
$clangStatus = "SKIP (not evaluated)"
$clangRunFailed = $false

try {
    # ---------------- gcc (PATH): gate-owning compiler ----------------
    $gccCommand = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gccCommand) {
        Write-Host "hmx_selftest_gcc=FAIL (gcc not found on PATH)"
    } else {
        Write-Host ("hmx_selftest_gcc_version=" +
            ((& $gccCommand.Source --version | Select-Object -First 1)))
        Write-Host ("hmx_selftest_gcc_triple=" +
            ((& $gccCommand.Source -dumpmachine)))
        $gccExe = Join-Path $TempDir "probe_host_selftest_gcc.exe"
        Write-Host ("hmx_selftest_gcc_compile=gcc -std=gnu11 -O2 -Wall -Wextra" +
            " -Werror -o <temp>\probe_host_selftest_gcc.exe" +
            " host_tests\hmx_acc_residual_rpc\probe_host_selftest.c -lm")
        & $gccCommand.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-o" $gccExe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "hmx_selftest_gcc=FAIL (compile exit $LASTEXITCODE)"
        } else {
            & $gccExe
            $gccExit = $LASTEXITCODE
            Write-Host "hmx_selftest_gcc_exit=$gccExit"
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
            # Import the MSVC environment into this script's own process only
            # (the suite runner invokes this script in a child pwsh).
            $vcEnvLines = & cmd /c "call `"$vcvars`" >nul 2>&1 && set"
            if ($LASTEXITCODE -ne 0) {
                $clangStatus = "SKIP (vcvars64.bat failed with exit $LASTEXITCODE)"
            } else {
                foreach ($line in $vcEnvLines) {
                    if ($line -match '^([^=]+)=(.*)$') {
                        Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2]
                    }
                }
                Write-Host ("hmx_selftest_clang_version=" +
                    ((& $clangPath --version | Select-Object -First 1)))
                $clangTriple = (& $clangPath -print-effective-triple)
                Write-Host "hmx_selftest_clang_triple=$clangTriple"
                $clangExe = Join-Path $TempDir "probe_host_selftest_clang.exe"
                Write-Host ("hmx_selftest_clang_compile=clang -std=gnu11 -O2" +
                    " -Wall -Wextra -Werror -rtlib=compiler-rt" +
                    " -o <temp>\probe_host_selftest_clang.exe" +
                    " host_tests\hmx_acc_residual_rpc\probe_host_selftest.c")
                & $clangPath "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
                    "-rtlib=compiler-rt" "-o" $clangExe $Source
                if ($LASTEXITCODE -ne 0) {
                    $clangStatus = "BUILD_FAILED (exit $LASTEXITCODE; gcc result governs)"
                } else {
                    & $clangExe
                    $clangExit = $LASTEXITCODE
                    if ($clangExit -eq 0) {
                        $clangStatus = "PASS (selftest exit 0, triple=$clangTriple)"
                    } else {
                        $clangStatus = "FAIL (selftest exit $clangExit, triple=$clangTriple)"
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

Write-Host "hmx_selftest_clang=$clangStatus"
$exitCode = $gccExit
if ($clangRunFailed) { $exitCode = 1 }
if ($exitCode -eq 0) {
    Write-Host "hmx_selftest=PASS"
} else {
    Write-Host "hmx_selftest=FAIL (gcc exit $gccExit; clang: $clangStatus)"
}
exit $exitCode
