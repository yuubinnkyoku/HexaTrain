# Shared compile/run helpers for host test suites.
# Session-scoped object reuse only: identical compile-identity objects are
# compiled once per host-suite invocation and linked into multiple executables.
# No persistent compiler cache, no header-dependency incremental build system.
$ErrorActionPreference = "Stop"

$script:PhoneLmHostObjectSession = $null

function Resolve-PhoneLmHostPwsh {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $cmd -or -not (Test-Path -LiteralPath $cmd.Source)) {
        throw "PWSH_NOT_FOUND: pwsh.exe (PowerShell 7+) is not on PATH"
    }
    return $cmd.Source
}

function Resolve-PhoneLmHostCompiler {
    $cmd = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $cmd -or -not (Test-Path -LiteralPath $cmd.Source)) {
        throw "g++ not found on PATH (required for host C++ compile)"
    }
    return $cmd.Source
}

function Get-PhoneLmHostSourceDisplayName {
    param(
        [Parameter(Mandatory = $true)][string]$Source
    )
    $root = Split-Path -Parent $PSScriptRoot
    $full = [System.IO.Path]::GetFullPath($Source)
    $rootFull = [System.IO.Path]::GetFullPath($root)
    if ($full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rel = $full.Substring($rootFull.Length).TrimStart('\', '/')
        return ($rel -replace '\\', '/')
    }
    return $full
}

function Get-PhoneLmHostObjectKey {
    # Object identity is the actual compile argv plus canonical source path.
    # Label / executable name is intentionally excluded.
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string[]]$CompileArgs,
        [Parameter(Mandatory = $true)][string]$CompilerPath
    )
    $canonicalSource = [System.IO.Path]::GetFullPath($Source)
    $canonicalArgs = @()
    foreach ($arg in $CompileArgs) {
        if ($arg -eq "-I") {
            $canonicalArgs += $arg
            continue
        }
        if ($canonicalArgs.Count -gt 0 -and $canonicalArgs[-1] -eq "-I") {
            $canonicalArgs += [System.IO.Path]::GetFullPath($arg)
            continue
        }
        $canonicalArgs += $arg
    }
    $parts = @([System.IO.Path]::GetFullPath($CompilerPath)) + $canonicalArgs + @($canonicalSource)
    $joined = [string]::Join("`n", $parts)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($joined)
        $hash = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Initialize-PhoneLmHostObjectSession {
    # Session-scoped only. Fresh creates a new ownership token. Join requires
    # the caller to present that exact token, so an arbitrary stale directory
    # from another invocation cannot be reused accidentally.
    param(
        [Parameter(Mandatory = $true)][string]$SessionDirectory,
        [switch]$Fresh,
        [string]$SessionToken = "",
        [string]$ExpectedSessionToken = ""
    )
    $sessionFull = [System.IO.Path]::GetFullPath($SessionDirectory)
    $markerPath = Join-Path $sessionFull ".session-token"

    if ($Fresh) {
        if (Test-Path -LiteralPath $sessionFull) {
            Remove-Item -LiteralPath $sessionFull -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $sessionFull | Out-Null
        if ([string]::IsNullOrWhiteSpace($SessionToken)) {
            $SessionToken = [Guid]::NewGuid().ToString("N")
        }
        [System.IO.File]::WriteAllText(
            $markerPath,
            $SessionToken,
            [System.Text.UTF8Encoding]::new($false))
    } else {
        if ([string]::IsNullOrWhiteSpace($ExpectedSessionToken)) {
            throw "HOST_OBJECT_SESSION_TOKEN_REQUIRED"
        }
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "HOST_OBJECT_SESSION_MARKER_MISSING:$sessionFull"
        }
        $actualToken = (Get-Content -LiteralPath $markerPath -Raw).Trim()
        if ($actualToken -cne $ExpectedSessionToken) {
            throw "HOST_OBJECT_SESSION_TOKEN_MISMATCH"
        }
        $SessionToken = $actualToken
    }

    $objectDir = Join-Path $sessionFull "objects"
    New-Item -ItemType Directory -Force -Path $objectDir | Out-Null
    $script:PhoneLmHostObjectSession = @{
        Directory      = $sessionFull
        ObjectDir      = [System.IO.Path]::GetFullPath($objectDir)
        SessionToken   = $SessionToken
        ObjectCompiles = 0
        ObjectReuses   = 0
        Links          = 0
    }
}
function Complete-PhoneLmHostObjectSession {
    if (-not $script:PhoneLmHostObjectSession) {
        return
    }
    $s = $script:PhoneLmHostObjectSession
    Write-Host ("object_compiles={0}" -f $s.ObjectCompiles)
    Write-Host ("object_reuses={0}" -f $s.ObjectReuses)
    Write-Host ("links={0}" -f $s.Links)
    $script:PhoneLmHostObjectSession = $null
}

function Get-PhoneLmHostObjectPath {
    param(
        [Parameter(Mandatory = $true)][string]$Key
    )
    if (-not $script:PhoneLmHostObjectSession) {
        throw "HOST_OBJECT_SESSION_NOT_INITIALIZED"
    }
    return (Join-Path $script:PhoneLmHostObjectSession.ObjectDir ("{0}.o" -f $Key))
}

function Invoke-PhoneLmHostCppCompile {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Output,
        [Parameter(Mandatory = $true)][string[]]$Sources,
        [string[]]$IncludeDirs = @(),
        [string]$Std = "-std=c++17"
    )
    if (-not $script:PhoneLmHostObjectSession) {
        Initialize-PhoneLmHostObjectSession `
            -SessionDirectory (Join-Path ([System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))) "build\host-test-objects\auto-$PID") `
            -Fresh
    }
    $compiler = Resolve-PhoneLmHostCompiler
    # Compile argv is the object-identity basis (not Label / Output).
    $compileArgs = @($Std, "-O2", "-Wall", "-Wextra", "-Wpedantic")
    foreach ($include in $IncludeDirs) {
        $compileArgs += @("-I", $include)
    }
    $objects = @()
    foreach ($source in $Sources) {
        $displayName = Get-PhoneLmHostSourceDisplayName -Source $source
        $key = Get-PhoneLmHostObjectKey -Source $source -CompileArgs $compileArgs -CompilerPath $compiler
        $objPath = Get-PhoneLmHostObjectPath -Key $key
        if (Test-Path -LiteralPath $objPath) {
            Write-Host ("object reuse HIT {0}" -f $displayName)
            $script:PhoneLmHostObjectSession.ObjectReuses++
        }
        else {
            Write-Host ("object compile MISS {0}" -f $displayName)
            & $compiler @compileArgs -c $source -o $objPath
            if ($LASTEXITCODE -ne 0) {
                throw "$Label compilation failed for $displayName"
            }
            $script:PhoneLmHostObjectSession.ObjectCompiles++
        }
        $objects += $objPath
    }
    Write-Host ("link {0}" -f [System.IO.Path]::GetFileName($Output))
    & $compiler @objects -o $Output
    if ($LASTEXITCODE -ne 0) {
        throw "$Label link failed"
    }
    $script:PhoneLmHostObjectSession.Links++
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

function Test-PhoneLmHostObjectKeyIdentity {
    $root = Split-Path -Parent $PSScriptRoot
    $source = Join-Path $root "host_tests\object-key-fixture.cpp"
    $compiler = Join-Path $root "build\fixture-g++.exe"
    $includeA = Join-Path $root "app\src\main\cpp"
    $includeB = Join-Path $root "host_tests"

    $base = @("-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", $includeA, "-I", $includeB)
    $same = @("-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", $includeA, "-I", $includeB)
    $std20 = @("-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", $includeA, "-I", $includeB)
    $differentInclude = @("-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", $includeA)
    $reorderedIncludes = @("-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-I", $includeB, "-I", $includeA)

    $baseKey = Get-PhoneLmHostObjectKey -Source $source -CompileArgs $base -CompilerPath $compiler
    if ($baseKey -cne (Get-PhoneLmHostObjectKey -Source $source -CompileArgs $same -CompilerPath $compiler)) {
        throw "HOST_OBJECT_KEY_IDENTICAL_ARGS_DIFFER"
    }
    if ($baseKey -ceq (Get-PhoneLmHostObjectKey -Source $source -CompileArgs $std20 -CompilerPath $compiler)) {
        throw "HOST_OBJECT_KEY_STD_NOT_DISTINGUISHED"
    }
    if ($baseKey -ceq (Get-PhoneLmHostObjectKey -Source $source -CompileArgs $differentInclude -CompilerPath $compiler)) {
        throw "HOST_OBJECT_KEY_INCLUDE_SET_NOT_DISTINGUISHED"
    }
    if ($baseKey -ceq (Get-PhoneLmHostObjectKey -Source $source -CompileArgs $reorderedIncludes -CompilerPath $compiler)) {
        throw "HOST_OBJECT_KEY_INCLUDE_ORDER_NOT_DISTINGUISHED"
    }
    Write-Host "host_object_key_identity_self_test=PASS"
}

function Test-PhoneLmHostObjectSessionGuard {
    $root = Split-Path -Parent $PSScriptRoot
    $fixture = Join-Path $root ("build\host-test-objects\session-guard-{0}-{1}" -f $PID, [Guid]::NewGuid().ToString("N"))
    $token = [Guid]::NewGuid().ToString("N")
    try {
        Initialize-PhoneLmHostObjectSession -SessionDirectory $fixture -Fresh -SessionToken $token
        Complete-PhoneLmHostObjectSession
        Initialize-PhoneLmHostObjectSession -SessionDirectory $fixture -ExpectedSessionToken $token
        Complete-PhoneLmHostObjectSession

        $missingTokenFailed = $false
        try {
            Initialize-PhoneLmHostObjectSession -SessionDirectory $fixture
        } catch {
            $missingTokenFailed = $_.Exception.Message -eq "HOST_OBJECT_SESSION_TOKEN_REQUIRED"
        }
        if (-not $missingTokenFailed) {
            throw "HOST_OBJECT_SESSION_MISSING_TOKEN_DID_NOT_FAIL"
        }

        $wrongTokenFailed = $false
        try {
            Initialize-PhoneLmHostObjectSession -SessionDirectory $fixture -ExpectedSessionToken ([Guid]::NewGuid().ToString("N"))
        } catch {
            $wrongTokenFailed = $_.Exception.Message -eq "HOST_OBJECT_SESSION_TOKEN_MISMATCH"
        }
        if (-not $wrongTokenFailed) {
            throw "HOST_OBJECT_SESSION_WRONG_TOKEN_DID_NOT_FAIL"
        }
        Write-Host "host_object_session_guard_self_test=PASS"
    } finally {
        $script:PhoneLmHostObjectSession = $null
        if (Test-Path -LiteralPath $fixture) {
            Remove-Item -LiteralPath $fixture -Recurse -Force
        }
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
    # Session-scoped object reuse plumbing must remain present.
    if ($commonText -notmatch 'Initialize-PhoneLmHostObjectSession') {
        throw 'HOST_RUNNER_SELF_CHECK_MISSING_OBJECT_SESSION_INIT'
    }
    if ($commonText -notmatch 'Get-PhoneLmHostObjectKey') {
        throw 'HOST_RUNNER_SELF_CHECK_MISSING_OBJECT_KEY'
    }
    if ($commonText -notmatch 'object compile MISS' -or $commonText -notmatch 'object reuse HIT') {
        throw 'HOST_RUNNER_SELF_CHECK_MISSING_OBJECT_HIT_MISS_LOG'
    }
    Test-PhoneLmHostObjectKeyIdentity
    Test-PhoneLmHostObjectSessionGuard
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
