# Shared compile/run helpers for host test suites.
# Intentionally minimal: compile, exit-code check, invoke. No mini build system.
$ErrorActionPreference = "Stop"

function Resolve-PhoneLmHostPwsh {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $cmd -or -not (Test-Path -LiteralPath $cmd.Source)) {
        throw "PWSH_NOT_FOUND: pwsh.exe (PowerShell 7+) is not on PATH"
    }
    return $cmd.Source
}

function Invoke-PhoneLmHostCppCompile {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Output,
        [Parameter(Mandatory = $true)][string[]]$Sources,
        [string[]]$IncludeDirs = @(),
        [string]$Std = "-std=c++17"
    )
    if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
        throw "g++ not found on PATH (required for $Label)"
    }
    $arguments = @($Std, "-O2", "-Wall", "-Wextra", "-Wpedantic")
    foreach ($include in $IncludeDirs) {
        $arguments += @("-I", $include)
    }
    $arguments += $Sources
    $arguments += @("-o", $Output)
    & g++ @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label compilation failed"
    }
}

function Invoke-PhoneLmHostCppRun {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = ""
    )
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
        try {
            & $Executable @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "$Label failed with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
        return
    }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-PhoneLmHostPwshScript {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [string[]]$Arguments = @()
    )
    $pwsh = Resolve-PhoneLmHostPwsh
    & $pwsh -NoProfile -File $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Test-PhoneLmHostRunnerSelfCheck {
    # Structural check: a wrapper must not ignore suite failures.
    $wrapper = Join-Path $PSScriptRoot 'run_host_tests.ps1'
    $contract = Join-Path $PSScriptRoot 'run_host_contract_tests.ps1'
    $diagnostic = Join-Path $PSScriptRoot 'run_host_diagnostic_tests.ps1'
    $common = Join-Path $PSScriptRoot 'host_test_common.ps1'
    foreach ($path in @($wrapper, $contract, $diagnostic, $common)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "HOST_RUNNER_SELF_CHECK_MISSING:$path"
        }
    }
    $wrapperText = Get-Content -LiteralPath $wrapper -Raw
    $commonText = Get-Content -LiteralPath $common -Raw
    if ($wrapperText -notmatch 'run_host_contract_tests\.ps1') {
        throw 'HOST_RUNNER_SELF_CHECK_WRAPPER_MISSING_CONTRACT'
    }
    if ($wrapperText -notmatch 'run_host_diagnostic_tests\.ps1') {
        throw 'HOST_RUNNER_SELF_CHECK_WRAPPER_MISSING_DIAGNOSTIC'
    }
    if ($wrapperText -notmatch 'Invoke-PhoneLmHostPwshScript') {
        throw 'HOST_RUNNER_SELF_CHECK_WRAPPER_MISSING_INVOCATION_HELPER'
    }
    if ($commonText -notmatch 'LASTEXITCODE') {
        throw 'HOST_RUNNER_SELF_CHECK_WRAPPER_MISSING_EXIT_PROPAGATION'
    }
    # Contract binary must not retain diagnostic ownership symbols.
    $sdkIndependent = Join-Path (Split-Path -Parent $PSScriptRoot) 'host_tests\qnn_sdk_independent_test.cpp'
    $depthQuality = Join-Path (Split-Path -Parent $PSScriptRoot) 'host_tests\depth_quality_test.cpp'
    $dedicated = Join-Path (Split-Path -Parent $PSScriptRoot) 'host_tests\qnn_first_nonfinite_diagnostics_test.cpp'
    $sdkText = Get-Content -LiteralPath $sdkIndependent -Raw
    $depthText = Get-Content -LiteralPath $depthQuality -Raw
    if (-not (Test-Path -LiteralPath $dedicated -PathType Leaf)) {
        throw 'HOST_RUNNER_SELF_CHECK_MISSING_DEDICATED_FIRST_NONFINITE_TEST'
    }
    if ($sdkText -match 'testFirstNonfinite|qnn_first_nonfinite_diagnostics\.h') {
        throw 'HOST_RUNNER_SELF_CHECK_SDK_INDEPENDENT_RETAINS_FIRST_NONFINITE'
    }
    if ($depthText -match 'testCheckpointCodecV2|qnn_first_nonfinite_diagnostics\.h') {
        throw 'HOST_RUNNER_SELF_CHECK_DEPTH_QUALITY_RETAINS_FIRST_NONFINITE'
    }
}
