# Shared verification step helpers for scripts/verify.ps1 and scripts/verify_local.ps1.
# Profile orchestration lives in those entrypoints; this file only owns step
# bookkeeping and process helpers so profiles never copy step-runner logic.
$script:PhoneLmVerifyResults = [System.Collections.Generic.List[object]]::new()
$script:PhoneLmResolvedPwshExe = $null
$script:PhoneLmVerifyRoot = Split-Path -Parent $PSScriptRoot

function Initialize-PhoneLmVerifyResults {
    $script:PhoneLmVerifyResults = [System.Collections.Generic.List[object]]::new()
}

function Get-PhoneLmVerifyResults {
    return $script:PhoneLmVerifyResults
}

function Add-Result([string]$Name, [string]$Status, [double]$Seconds, [string]$Detail) {
    $short = if ($Detail) { ($Detail -replace "[\r\n]+", " ").Trim() } else { "" }
    if ($short.Length -gt 160) { $short = $short.Substring(0, 157) + "..." }
    $script:PhoneLmVerifyResults.Add([pscustomobject]@{
        Step    = $Name
        Status  = $Status
        Seconds = [math]::Round($Seconds, 1)
        Detail  = $short
    })
}

function Invoke-Step([string]$Name, [scriptblock]$Action) {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $detail = & $Action
        Add-Result $Name "PASS" $stopwatch.Elapsed.TotalSeconds ([string]$detail)
    } catch {
        Add-Result $Name "FAIL" $stopwatch.Elapsed.TotalSeconds $_.Exception.Message
    }
}

function Add-Skip([string]$Name, [string]$Reason) {
    Add-Result $Name "SKIP" 0 $Reason
}

function Invoke-Process([string]$Label, [string]$FilePath, [string[]]$Arguments) {
    # Stream child output to the console so it never lands in the step Detail.
    & $FilePath @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

function Resolve-PwshExe() {
    if ($script:PhoneLmResolvedPwshExe) { return $script:PhoneLmResolvedPwshExe }
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $cmd -or -not (Test-Path -LiteralPath $cmd.Source)) {
        throw "pwsh.exe (PowerShell 7+) not found on PATH; install PowerShell 7 or add it to PATH"
    }
    $script:PhoneLmResolvedPwshExe = $cmd.Source
    return $script:PhoneLmResolvedPwshExe
}

function Invoke-PwshScript([string]$Label, [string]$ScriptPath, [string[]]$Arguments) {
    $pwshExe = Resolve-PwshExe
    Invoke-Process $Label $pwshExe (@("-NoProfile", "-File", $ScriptPath) + $Arguments)
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Initialize-AndroidSdkEnv {
    # Match README build instructions: the SDK lives under %LOCALAPPDATA%\Android\Sdk
    # on a fresh shell where ANDROID_HOME is not exported yet. Process-local only;
    # local.properties is never written by this helper.
    if (-not $env:ANDROID_HOME -and -not $env:ANDROID_SDK_ROOT) {
        $defaultSdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"
        if (Test-Path -LiteralPath (Join-Path $defaultSdk "platform-tools") -PathType Container) {
            $env:ANDROID_HOME = $defaultSdk
            $env:ANDROID_SDK_ROOT = $defaultSdk
        }
    }
}

function Get-PhoneLmVerifyRoot {
    return $script:PhoneLmVerifyRoot
}

function Invoke-Gradle([string]$Label, [string[]]$Tasks) {
    $gradlew = Join-Path $script:PhoneLmVerifyRoot "gradlew.bat"
    Invoke-Process $Label $gradlew ($Tasks + "--no-daemon")
}

function Write-PhoneLmVerifySummary([string]$Title) {
    $results = Get-PhoneLmVerifyResults
    Write-Host ""
    Write-Host "===== $Title ====="
    foreach ($row in $results) {
        $line = "{0,-28} {1,-4} {2,8:N1}s  {3}" -f $row.Step, $row.Status, $row.Seconds, $row.Detail
        Write-Host $line
    }
    $passCount = @($results | Where-Object { $_.Status -eq "PASS" }).Count
    $failCount = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
    $skipCount = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
    $totalSeconds = ($results | Measure-Object -Property Seconds -Sum).Sum
    Write-Host ("total {0:N1}s  PASS={1} FAIL={2} SKIP={3}" -f $totalSeconds, $passCount, $failCount, $skipCount)
    return $failCount
}

function Invoke-PhoneLmHostContractFast {
    # Critical generated-artifact and CPU-reference contracts only.
    # Full contract coverage stays in run_host_contract_tests.ps1 -Suite All.
    Invoke-PwshScript "run_host_contract_tests(Fast)" `
        (Join-Path $script:PhoneLmVerifyRoot "scripts\run_host_contract_tests.ps1") `
        @("-Suite", "Fast")
}

function Invoke-PhoneLmHostSuite {
    Invoke-PwshScript "run_host_tests" `
        (Join-Path $script:PhoneLmVerifyRoot "scripts\run_host_tests.ps1") @()
}

function Assert-GppAvailable {
    if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
        throw "g++ not found on PATH (required by host C++ tests)"
    }
}

function Invoke-PhoneLmQairtCheck {
    # Shared pinned-QAIRT check used by verify.ps1 -Profile Qnn and
    # verify_local.ps1 -WithQairt. Exit 3 (optional inventory incomplete)
    # is advisory; exit 0/3 with expected_build_id_match=true is required.
    param(
        [Parameter(Mandatory = $true)][string]$SdkRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedBuildId
    )
    $root = Get-PhoneLmVerifyRoot
    $pwshExe = Resolve-PwshExe
    $checkOutput = & $pwshExe -NoProfile -File (Join-Path $root "scripts\check_qairt.ps1") `
        -SdkRoot $SdkRoot -ExpectedBuildId $ExpectedBuildId
    $checkExit = $LASTEXITCODE
    $checkOutput | Out-Host
    function Get-CheckValue([string]$Key) {
        $hit = $checkOutput | Select-String -Pattern "^$([regex]::Escape($Key))=(.*)$" |
            Select-Object -First 1
        if (-not $hit) { return $null }
        return $hit.Matches[0].Groups[1].Value
    }
    $checkStatus = Get-CheckValue "status"
    if ($checkExit -eq 2 -or $checkStatus -in @(
            "QAIRT_SDK_ROOT_UNAVAILABLE", "QAIRT_SDK_ROOT_MISMATCH")) {
        throw "check_qairt: QAIRT SDK not found at $SdkRoot"
    }
    if ($checkExit -eq 4 -or $checkStatus -eq "QAIRT_BUILD_ID_MISMATCH") {
        throw "check_qairt: expected build ID $ExpectedBuildId not satisfied by $SdkRoot"
    }
    if ($checkExit -eq 5 -or $checkStatus -eq "QAIRT_CORE_INCOMPLETE") {
        throw "check_qairt: QAIRT core required items are incomplete at $SdkRoot"
    }
    if ($checkExit -notin @(0, 3)) {
        throw "check_qairt failed with exit code $checkExit ($checkStatus)"
    }
    if (-not $checkStatus) { throw "check_qairt failed with exit code $checkExit" }
    if ((Get-CheckValue "expected_build_id_match") -ne "true") {
        throw "check_qairt did not confirm expected_build_id_match=true"
    }
    $resolved = Get-CheckValue "sdk_root"
    if (-not $resolved) { throw "check_qairt did not report sdk_root" }
    $requested = [IO.Path]::GetFullPath($SdkRoot)
    $resolvedFull = [IO.Path]::GetFullPath($resolved)
    if ($requested -ne $resolvedFull) {
        throw "Explicit QAIRT SDK root was not honored: requested=$requested resolved=$resolvedFull"
    }
    return "QAIRT root honored, build ID match ($checkStatus)"
}

function Initialize-PhoneLmVerifySession {
    Initialize-PhoneLmVerifyResults
    Initialize-AndroidSdkEnv
}
