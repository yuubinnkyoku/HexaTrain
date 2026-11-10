# D8 minimal experiment — CPU model runner (CPU only).
#
# Source: host_tests/hmx_acc_residual_rpc/hmx_d8_min_sim.c
#         host_tests/hmx_acc_residual_rpc/hmx_d8_rev3_selftest.c
# Scope: D8a/D8b reference bits, access-trace and issued-order validators,
# order/guard fault cases, the metadata/q/failure contract self-test, the
# independent Python differentials, and the static implementation-feasibility
# audit. It never touches a device, adb, the Hexagon SDK, QAIRT or Mode 46.
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
#   d8_rev3=PASS|FAIL|SKIP ...
#   d8_rev3_python=PASS|FAIL|SKIP ...
#   d8_disasm_rules=PASS|FAIL|SKIP ...
#   d8_feasibility=PASS|FAIL|SKIP ...
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
$feasStatus = "SKIP (not evaluated)"
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
                if ($_ -match '^(d8_ref|d8_diff|d8_safe_vs|d8_fault=|d8_fill_mask|d8_checks=|d8_verdict=|d8_dump=)') {
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

    # ---------------- revision-3 contract self-test (CPU only) ----------------
    # The metadata layout, q selector, guard evidence and failure-state gate.
    # It does not replace the D8 numeric model; it tests the host contract.
    $rev3Status = "SKIP (not evaluated)"
    $rev3Checks = "unknown"
    $rev3Src = Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_rev3_selftest.c"
    if (-not (Test-Path -LiteralPath $rev3Src -PathType Leaf)) {
        $rev3Status = "FAIL (missing $rev3Src)"
        $failed = $true
    } elseif (-not $gcc) {
        $rev3Status = "SKIP (gcc not found on PATH)"
    } else {
        $rev3Exe = Join-Path $BuildDir "hmx_d8_rev3_selftest.exe"
        & $gcc.Source "-std=gnu11" "-O2" "-Wall" "-Wextra" "-Werror" `
            "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
            "-o" $rev3Exe $rev3Src
        if ($LASTEXITCODE -ne 0) {
            $rev3Status = "FAIL (compile exit $LASTEXITCODE)"
            $failed = $true
        } else {
            $rev3Out = & $rev3Exe 2>&1 | Out-String
            $rev3Exit = $LASTEXITCODE
            $m = [regex]::Match($rev3Out, 'd8_rev3_checks=(\d+) fail=(\d+)')
            if ($m.Success) { $rev3Checks = $m.Value }
            $rev3Out.TrimEnd().Split("`n") | ForEach-Object {
                if ($_ -match '^(d8_rev3_|d8_contract_)') { Write-Host $_.Trim() }
            }
            $rev3GccOk = ($rev3Exit -eq 0) -and
                ($rev3Out -match 'd8_rev3_verdict=D8_REV3_CPU_CONTRACT_READY')
            $rev3O0Ok = $true
            if ($rev3GccOk) {
                $rev3O0 = Join-Path $TempDir "hmx_d8_rev3_o0.exe"
                & $gcc.Source "-std=gnu11" "-O0" "-Wall" "-Wextra" "-Werror" `
                    "-ffp-contract=off" "-I" (Join-Path $Root "host_tests\hmx_acc_residual_rpc") `
                    "-o" $rev3O0 $rev3Src
                if ($LASTEXITCODE -ne 0) {
                    $rev3O0Ok = $false
                } else {
                    $o0Out = & $rev3O0 2>&1 | Out-String
                    $rev3O0Ok = ($LASTEXITCODE -eq 0) -and
                        ($o0Out -match 'd8_rev3_checks=\d+ fail=0') -and
                        ($o0Out -match 'd8_rev3_verdict=D8_REV3_CPU_CONTRACT_READY')
                }
            }
            $rev3ClangOk = $true
            if ($rev3GccOk -and $clang -and $vcvars) {
                $rev3ClangExe = Join-Path $TempDir "hmx_d8_rev3_clang.exe"
                $cmd = ('@echo off & call "{0}" >nul 2>&1 & "{1}" -std=gnu11 -O2 ' +
                        '-Wall -Wextra -Werror -ffp-contract=off -I "{2}" ' +
                        '-rtlib=compiler-rt -o "{3}" "{4}"') -f `
                    $vcvars, $clang.Source, `
                    (Join-Path $Root "host_tests\hmx_acc_residual_rpc"), `
                    $rev3ClangExe, $rev3Src
                $buildOut = & cmd.exe /c $cmd 2>&1
                if ($LASTEXITCODE -ne 0) {
                    $rev3ClangOk = $false
                } else {
                    $cOut = & $rev3ClangExe 2>&1 | Out-String
                    $rev3ClangOk = ($LASTEXITCODE -eq 0) -and
                        ($cOut -match 'd8_rev3_checks=\d+ fail=0')
                }
            }
            if ($rev3GccOk -and $rev3O0Ok -and $rev3ClangOk) {
                $rev3Status = "PASS ($rev3Checks, gcc -O0/-O2, clang -O2)"
            } else {
                $rev3Status = "FAIL (gcc=$rev3GccOk o0=$rev3O0Ok clang=$rev3ClangOk)"
                $failed = $true
            }
        }
    }
    Write-Host "d8_rev3=$rev3Status"

    # ---------------- implementation feasibility (static, CPU only) ----------------
    # Independent of gcc: reads the D8 documents and the existing DSP/host
    # interface. It records the pre-implementation gaps; a non-zero exit is a
    # broken invariant (layout/mask/sequence/interface drift), not a device
    # result. See docs/hmx-v81-d8-implementation-feasibility-audit.md.
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $feasStatus = "SKIP (python not found on PATH)"
    } else {
        & $python.Source (Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_implementation_feasibility_test.py")
        $feasExit = $LASTEXITCODE
        Write-Host "d8_feasibility_exit=$feasExit"
        if ($feasExit -eq 0) {
            $feasStatus = "PASS (static audit)"
        } else {
            $feasStatus = "FAIL (exit $feasExit)"
            $failed = $true
        }
    }
    Write-Host "d8_feasibility=$feasStatus"

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

    # ---------------- revision-3 independent Python contract audit ----------------
    $rev3PyStatus = "SKIP (not evaluated)"
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $rev3PyStatus = "SKIP (python not found on PATH)"
    } elseif ($rev3Status -notmatch '^PASS') {
        $rev3PyStatus = "SKIP (C contract self-test did not pass)"
    } else {
        & $python.Source (Join-Path $Root "host_tests\hmx_acc_residual_rpc\hmx_d8_rev3_contract_test.py")
        $rev3PyExit = $LASTEXITCODE
        Write-Host "d8_rev3_python_exit=$rev3PyExit"
        if ($rev3PyExit -eq 0) {
            $rev3PyStatus = "PASS (independent contract audit)"
        } else {
            $rev3PyStatus = "FAIL (exit $rev3PyExit)"
            $failed = $true
        }
    }
    Write-Host "d8_rev3_python=$rev3PyStatus"

    # ---------------- D8 disassembly audit rules (synthetic text only) ----------------
    # No V81 binary exists here. This runs the rule engine on a synthetic
    # fixture and its mutations; it is not a passed audit of a real object.
    $disasmStatus = "SKIP (not evaluated)"
    $disasmTest = Join-Path $Root "scripts\test_hmx_d8_disasm_audit.ps1"
    if (-not (Test-Path -LiteralPath $disasmTest -PathType Leaf)) {
        $disasmStatus = "FAIL (missing $disasmTest)"
        $failed = $true
    } else {
        $disasmOut = & $disasmTest *>&1 | Out-String
        $disasmExit = $LASTEXITCODE
        $disasmOut.TrimEnd().Split("`n") | ForEach-Object {
            if ($_ -match '^(d8_disasm_audit_test)') { Write-Host $_.Trim() }
        }
        if ($disasmExit -eq 0 -and $disasmOut -match 'd8_disasm_audit_test=PASS') {
            $disasmStatus = "PASS (synthetic rules only, no real V81 object)"
        } else {
            $disasmStatus = "FAIL (exit $disasmExit)"
            $failed = $true
        }
    }
    Write-Host "d8_disasm_rules=$disasmStatus"
} finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "d8_sim_gcc_checks=$gccChecks"
Write-Host "d8_rev3_checks=$rev3Checks"
if ($gccExit -ne 0) { $failed = $true }
if ($failed) {
    Write-Host "d8_sim=FAIL (gcc=$gccExit o0='$o0Status' clang='$clangStatus' rev3='$rev3Status' rev3_python='$rev3PyStatus' disasm='$disasmStatus' feasibility='$feasStatus' python='$pythonStatus')"
    exit 1
}
Write-Host "d8_sim=PASS"
exit 0
