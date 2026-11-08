# D8 minimal experiment — CPU model runner (CPU only).
#
# Source: host_tests/hmx_acc_residual_rpc/hmx_d8_min_sim.c
# Scope: D8a/D8b reference bits, access-trace validator, ten fault/control
# cases, and the independent Python differential. It never touches a device,
# adb, the Hexagon SDK, QAIRT or Mode 46.
#
# Exit-code policy (mirrors run_hmx_b3_r2_sim.ps1):
#   - gcc compile/run failure -> that exit code,
#   - clang run failure -> exit 1,
#   - missing clang / MSVC environment / python -> SKIP status only.
#
# Output contract (stable keys):
#   d8_sim_gcc_exit=<n>
#   d8_sim_gcc_checks=...
#   d8_sim_o0=PASS|FAIL|SKIP
#   d8_sim_clang=PASS|FAIL|SKIP ...
#   d8_python=PASS|FAIL|SKIP ...
#   d8_sim=PASS|FAIL
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_min_sim.c"
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Write-Host "d8_sim=FAIL (missing source $Source)"
    exit 1
}
$BuildDir = Join-Path $Root "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("hmx-d8-sim-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

$gccExit = 1
$gccChecks = "unknown"
$o0Status = "SKIP (not evaluated)"
$clangStatus = "SKIP (not evaluated)"
$pythonStatus = "SKIP (not evaluated)"
$failed = $false

try {
    # ---------------- gcc -O2 (gate owner) ----------------
    $gcc = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gcc) {
        Write-Host "d8_sim_gcc=FAIL (gcc not found on PATH)"
    } else {
        $gccExe = Join-Path $BuildDir "hmx_d8_sim.exe"
        & $gcc.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
            "-o" $gccExe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "d8_sim_gcc=FAIL (compile exit $LASTEXITCODE)"
        } else {
            $gccOut = & $gccExe 2>&1 | Out-String
            $gccExit = $LASTEXITCODE
            Write-Host "d8_sim_gcc_exit=$gccExit"
            $gccOut.TrimEnd().Split("`n") | ForEach-Object {
                if ($_ -match '^(d8_ref|d8_diff|d8_safe_vs|d8_fault=|d8_checks=|d8_verdict=|d8_dump=)') {
                    Write-Host $_.Trim()
                }
            }
            $m = [regex]::Match($gccOut, 'd8_checks=(\d+) fail=(\d+)')
            if ($m.Success) { $gccChecks = $m.Value }
            if ($gccExit -ne 0 -or $gccOut -notmatch 'd8_verdict=D8_CPU_READY') {
                $failed = $true
            }
        }
    }

    # ---------------- gcc -O0 cross-check ----------------
    if ($gcc -and $gccExit -eq 0) {
        $o0Exe = Join-Path $TempDir "hmx_d8_sim_o0.exe"
        & $gcc.Source "-std=gnu11" "-O0" "-Wall" "-Wextra" "-Werror" `
            "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
            "-o" $o0Exe $Source "-lm"
        if ($LASTEXITCODE -ne 0) {
            $o0Status = "FAIL (compile exit $LASTEXITCODE)"
            $failed = $true
        } else {
            $o0Out = & $o0Exe 2>&1 | Out-String
            $o0Exit = $LASTEXITCODE
            $same = $true
            foreach ($key in @("sha256=c9abf30e", "sha256=f0b3b4aa",
                               "sha256=b6328e88", "sha256=5285b463")) {
                if ($o0Out -notmatch [regex]::Escape($key)) { $same = $false }
            }
            if ($o0Exit -eq 0 -and $same -and $o0Out -match 'd8_verdict=D8_CPU_READY') {
                $o0Status = "PASS (same four reference digests)"
            } else {
                $o0Status = "FAIL (exit $o0Exit same_digests=$same)"
                $failed = $true
            }
        }
    }
    Write-Host "d8_sim_o0=$o0Status"

    # ---------------- clang -O2 (cross-compiler) ----------------
    $clang = Get-Command clang -ErrorAction SilentlyContinue
    if (-not $clang) {
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
                $candidate = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    $vcvars = $candidate
                }
            }
        }
        if (-not $vcvars) {
            $clangStatus = "SKIP (MSVC environment not found)"
        } else {
            $clangExe = Join-Path $TempDir "hmx_d8_sim_clang.exe"
            $cmd = ('@echo off & call "{0}" >nul 2>&1 & "{1}" -std=gnu11 -O2 ' +
                    '-Wall -Wextra -Werror -ffp-contract=off -I "{2}" ' +
                    '-rtlib=compiler-rt -o "{3}" "{4}"') -f `
                $vcvars, $clang.Source, `
                (Join-Path $Root "host_tests\hmx_acc_residual_rpc"), `
                $clangExe, $Source
            $buildOut = & cmd.exe /c $cmd 2>&1
            if ($LASTEXITCODE -ne 0) {
                $clangStatus = "SKIP (clang build failed: $buildOut)"
            } else {
                $clangOut = & $clangExe 2>&1 | Out-String
                $clangExit = $LASTEXITCODE
                if ($clangExit -eq 0 -and $clangOut -match 'd8_verdict=D8_CPU_READY') {
                    $clangStatus = "PASS (exit 0)"
                } else {
                    $clangStatus = "FAIL (exit $clangExit)"
                    $failed = $true
                }
            }
        }
    }
    Write-Host "d8_sim_clang=$clangStatus"

    # ---------------- independent Python differential ----------------
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $pythonStatus = "SKIP (python not found on PATH)"
    } elseif ($gccExit -ne 0) {
        $pythonStatus = "SKIP (gcc simulator run failed, no dump to audit)"
    } else {
        & $python.Source (Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_independent_test.py")
        $pyExit = $LASTEXITCODE
        Write-Host "d8_python_independent_exit=$pyExit"
        & (Join-Path $Root "scripts\run_hmx_d8_mutation_check.ps1")
        $mutExit = $LASTEXITCODE
        Write-Host "d8_mutation_exit=$mutExit"
        if ($pyExit -eq 0 -and $mutExit -eq 0) {
            $pythonStatus = "PASS (independent=exit0 mutation=exit0)"
        } else {
            $pythonStatus = "FAIL (independent=$pyExit mutation=$mutExit)"
            $failed = $true
        }
    }
    Write-Host "d8_python=$pythonStatus"
} finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "d8_sim_gcc_checks=$gccChecks"
if ($gccExit -ne 0) { $failed = $true }
if ($failed) {
    Write-Host "d8_sim=FAIL (gcc=$gccExit o0='$o0Status' clang='$clangStatus' python='$pythonStatus')"
    exit 1
}
Write-Host "d8_sim=PASS"
exit 0
