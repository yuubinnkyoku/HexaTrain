# HMX V81 signed-output / per-channel-bias oracle runner (CPU only).
#
# Source: host_tests/hmx_acc_residual_rpc/hmx_signed_output_oracle.c
# Scope: E-3D pre-registration evidence. It compiles and runs the PRM-derived
# convert model plus the E-3D four-cell discriminator model. It never touches a
# device, adb, the Hexagon SDK or any DSP artifact.
#
# Exit-code policy (mirrors run_prm_layout_sim.ps1 / run_fp16x2_split_sim.ps1):
#   - gcc compile failure or a failing gcc run -> that exit code,
#   - a clang RUN failure -> exit 1,
#   - missing clang / missing MSVC environment -> status line only.
#
# -ffp-contract=off is required: the model pins exact double results
# (e.g. 61.984375, a tie at 0x53BF/0x53C0), and FMA contraction would change
# the last bit on targets that have FMA.
#
# Output contract (stable keys):
#   signed_oracle_gcc_version=...
#   signed_oracle_gcc_compile=...
#   signed_oracle_gcc_exit=<n>
#   signed_oracle_clang=<status>
#   signed_oracle=PASS|FAIL
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_signed_output_oracle.c"
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Write-Host "signed_oracle=FAIL (missing source $Source)"
    exit 1
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("hmx-signed-oracle-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$gccExit = 1
$clangStatus = "SKIP (not evaluated)"
$clangRunFailed = $false

try {
    # ---------------- gcc (PATH): gate-owning compiler ----------------
    $gccCommand = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gccCommand) {
        Write-Host "signed_oracle_gcc=FAIL (gcc not found on PATH)"
    } else {
        Write-Host ("signed_oracle_gcc_version=" +
            ((& $gccCommand.Source --version | Select-Object -First 1)))
        Write-Host ("signed_oracle_gcc_triple=" +
            ((& $gccCommand.Source -dumpmachine)))
        $gccExe = Join-Path $TempDir "hmx_signed_output_oracle_gcc.exe"
        Write-Host ("signed_oracle_gcc_compile=gcc -std=gnu11 -O2 -Wall -Wextra" +
            " -Werror -ffp-contract=off -o <temp>\hmx_signed_output_oracle_gcc.exe" +
            " host_tests\hmx_acc_residual_rpc\hmx_signed_output_oracle.c -lm")
        & $gccCommand.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-ffp-contract=off" "-o" $gccExe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "signed_oracle_gcc=FAIL (compile exit $LASTEXITCODE)"
        } else {
            & $gccExe
            $gccExit = $LASTEXITCODE
            Write-Host "signed_oracle_gcc_exit=$gccExit"
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
            $vcEnvLines = & cmd /c "call `"$vcvars`" >nul 2>&1 && set"
            if ($LASTEXITCODE -ne 0) {
                $clangStatus = "SKIP (vcvars64.bat failed with exit $LASTEXITCODE)"
            } else {
                foreach ($line in $vcEnvLines) {
                    if ($line -match '^([^=]+)=(.*)$') {
                        Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2]
                    }
                }
                Write-Host ("signed_oracle_clang_version=" +
                    ((& $clangPath --version | Select-Object -First 1)))
                $clangTriple = (& $clangPath -print-effective-triple)
                Write-Host "signed_oracle_clang_triple=$clangTriple"
                $clangExe = Join-Path $TempDir "hmx_signed_output_oracle_clang.exe"
                & $clangPath "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
                    "-ffp-contract=off" "-rtlib=compiler-rt" "-o" $clangExe $Source
                if ($LASTEXITCODE -ne 0) {
                    $clangStatus = "BUILD_FAILED (exit $LASTEXITCODE; gcc result governs)"
                } else {
                    & $clangExe
                    $clangExit = $LASTEXITCODE
                    if ($clangExit -eq 0) {
                        $clangStatus = "PASS (oracle exit 0, triple=$clangTriple)"
                    } else {
                        $clangStatus = "FAIL (oracle exit $clangExit, triple=$clangTriple)"
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

Write-Host "signed_oracle_clang=$clangStatus"
$exitCode = $gccExit
if ($clangRunFailed) { $exitCode = 1 }
if ($exitCode -eq 0) {
    Write-Host "signed_oracle=PASS"
} else {
    Write-Host "signed_oracle=FAIL (gcc exit $gccExit; clang: $clangStatus)"
}
exit $exitCode
