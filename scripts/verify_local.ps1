# PhoneLM common local verification base gate.
# Default: no physical device, no QAIRT SDK, no writes outside build/ and the index.
# All required steps must PASS; QNN/device/publication changes have extra gates.
# See AGENTS.md "完了前の検証" and "実行Tier".
param(
    # Final gate only for docs/scripts-only changes; otherwise a fast pre-check.
    [switch]$SkipAndroidBuild,
    # Run gradle :app:clean first. Slow (CMake/NDK rebuild). Use only when required.
    [switch]$Clean,
    # Additionally verify the QAIRT SDK and a QNN-enabled build + APK audit.
    [switch]$WithQairt,
    [string]$QairtSdkRoot = "",
    [string]$ExpectedBuildId = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "qairt_version.ps1")

# Match README build instructions: the SDK lives under %LOCALAPPDATA%\Android\Sdk
# on a fresh shell where ANDROID_HOME is not exported yet. Process-local only;
# local.properties is never written by this script.
if (-not $env:ANDROID_HOME -and -not $env:ANDROID_SDK_ROOT) {
    $defaultSdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"
    if (Test-Path -LiteralPath (Join-Path $defaultSdk "platform-tools") -PathType Container) {
        $env:ANDROID_HOME = $defaultSdk
        $env:ANDROID_SDK_ROOT = $defaultSdk
    }
}

$results = [System.Collections.Generic.List[object]]::new()

function Add-Result([string]$Name, [string]$Status, [double]$Seconds, [string]$Detail) {
    $short = if ($Detail) { ($Detail -replace "[\r\n]+", " ").Trim() } else { "" }
    if ($short.Length -gt 160) { $short = $short.Substring(0, 157) + "..." }
    $results.Add([pscustomobject]@{
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

# External scripts may call `exit`, which would terminate this script when run
# in-process. Always run them in a child pwsh process and check $LASTEXITCODE.
function Invoke-PwshScript([string]$Label, [string]$ScriptPath, [string[]]$Arguments) {
    $pwshExe = Join-Path $PSHOME "pwsh.exe"
    Invoke-Process $Label $pwshExe (@("-NoProfile", "-File", $ScriptPath) + $Arguments)
}

function Invoke-Gradle([string]$Label, [string[]]$Tasks) {
    Invoke-Process $Label (Join-Path $Root "gradlew.bat") ($Tasks + "--no-daemon")
}

Push-Location $Root
try {
    Invoke-Step "git-diff-check" {
        git diff --check
        if ($LASTEXITCODE -ne 0) { throw "git diff --check reported whitespace errors" }
        git diff --cached --check
        if ($LASTEXITCODE -ne 0) { throw "git diff --cached --check reported whitespace errors" }
        "no whitespace errors"
    }

    Invoke-Step "tracked-binary-audit" {
        $pattern = '\.(so|apk|aab|jks|keystore|pem|key|log)$'
        $hits = @(git ls-files | Where-Object { $_ -match $pattern })
        if ($hits.Count -gt 0) {
            throw ("forbidden tracked files: " + ($hits -join ", "))
        }
        "0 forbidden tracked binaries/secrets"
    }

    Invoke-Step "secret-path-audit" {
        # Staged, unstaged, and untracked (non-ignored) files, excluding this
        # script itself because it contains the pattern literals.
        $changed = @(
            git diff --name-only --diff-filter=ACMR
            git diff --cached --name-only --diff-filter=ACMR
            git ls-files --others --exclude-standard
        ) | Sort-Object -Unique | Where-Object { $_ -and $_ -ne "scripts/verify_local.ps1" }
        $patterns = @(
            @{ Name = "adb-endpoint"; Regex = '\b\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}\b' },
            @{ Name = "user-abs-path"; Regex = '[A-Za-z]:[\\/]Users[\\/]' },
            @{ Name = "private-key"; Regex = 'BEGIN [A-Z ]*PRIVATE KEY' },
            @{ Name = "aws-access-key"; Regex = '\bAKIA[0-9A-Z]{16}\b' }
        )
        $bad = [System.Collections.Generic.List[string]]::new()
        foreach ($relative in $changed) {
            $full = Join-Path $Root $relative
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
            if ((Get-Item -LiteralPath $full).Length -gt 5MB) { continue }
            $text = Get-Content -LiteralPath $full -Raw
            if ($null -eq $text) { continue }
            foreach ($entry in $patterns) {
                if ([regex]::IsMatch($text, $entry.Regex)) {
                    $bad.Add("$relative ($($entry.Name))")
                }
            }
        }
        if ($bad.Count -gt 0) {
            throw ("suspicious content: " + ($bad -join "; ") +
                " — remove it, or adjust the audit if this is a false positive")
        }
        "$($changed.Count) changed/untracked file(s) scanned"
    }

    Invoke-Step "qairt-selection-self-test" {
        Invoke-PwshScript "check_qairt self-test" (Join-Path $Root "scripts\check_qairt.ps1") @(
            "-SelfTest")
        "pinned arguments, root selection, core/advisory classification ok (temp-only)"
    }

    Invoke-Step "margin-decomposition-probe" {
        Invoke-PwshScript "l19 first-error/margin decomposition probe" `
            (Join-Path $Root "scripts\run_l19_margin_decomposition.ps1") @()
        "deterministic CPU reports regenerated (private margin-tokens included)"
    }

    Invoke-Step "critical-margin-objective-probe" {
        Invoke-PwshScript "critical margin objective probe" `
            (Join-Path $Root "scripts\run_critical_margin_objective_benchmark.ps1") @()
        Invoke-PwshScript "critical margin training probe" `
            (Join-Path $Root "scripts\run_critical_margin_objective_benchmark.ps1") @(
                "-Train", "-BaselineDir", (Join-Path $Root "build\reports\qnn-critical-margin-objective"),
                "-ReportRoot", (Join-Path $Root "build\reports\qnn-critical-margin-training"))
        "deterministic CPU objective/training reports regenerated (private)"
    }

    Invoke-Step "readout-probe-self-test" {
        Invoke-PwshScript "readout probe self-test" `
            (Join-Path $Root "scripts\run_l19_readout_probe.ps1") @("-SelfTest")
        "readout probe self-test PASS (private)"
    }

    Invoke-Step "readout-representation-probe" {
        Invoke-PwshScript "readout/representation diagnosis run" `
            (Join-Path $Root "scripts\run_l19_readout_probe.ps1") @()
        "deterministic CPU readout/representation reports regenerated (private; exporter self-test pre-fly uses them)"
    }

    Invoke-Step "intra-block-readability-self-test" {
        Invoke-PwshScript "intra-block readability self-test" `
            (Join-Path $Root "scripts\run_l19_intra_block_readability.ps1") @("-SelfTest")
        "intra-block readability self-test PASS (private)"
    }

    Invoke-Step "intra-block-readability-run" {
        Invoke-PwshScript "intra-block readability diagnosis run" `
            (Join-Path $Root "scripts\run_l19_intra_block_readability.ps1") @()
        "deterministic CPU intra-block readability reports regenerated (private; exporter self-test pre-fly uses them)"
    }

    Invoke-Step "attention-internal-self-test" {
        Invoke-PwshScript "attention-internal diagnosis self-test" `
            (Join-Path $Root "scripts\run_l19_attention_internal_diagnosis.ps1") @("-SelfTest")
        "attention-internal diagnosis self-test PASS (private)"
    }

    Invoke-Step "attention-internal-run" {
        Invoke-PwshScript "attention-internal diagnosis run" `
            (Join-Path $Root "scripts\run_l19_attention_internal_diagnosis.ps1") @()
        "deterministic CPU attention-internal diagnosis reports regenerated (private; exporter self-test pre-fly uses them)"
    }

    Invoke-Step "public-exporter-self-test" {
        Invoke-PwshScript "post-fix public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_post_fix_generation_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "Tiny LM scaling public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_tiny_lm_scaling_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "multilayer/multihead public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_multilayer_multihead_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "generic depth/head public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_generic_depth_head_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "first-nonfinite public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_first_nonfinite_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "depth-quality public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_depth_quality_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "validation-selection public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_validation_selected_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "autoregressive validation public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_autoregressive_validation.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "first-error/margin public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_l19_margin_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "critical margin stabilization public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_l19_critical_margin_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "readout diagnosis public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_l19_readout_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "intra-block readability public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_l19_intra_block_results.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "attention-internal diagnosis public exporter self-test" `
            (Join-Path $Root "scripts\export_public_qnn_l19_attention_internal_results.ps1") @(
                "-SelfTest")
        "allow-list exports, manifest consistency, and negative rejection ok (temp-only)"
    }

    Invoke-Step "resumable-formal-runner-self-test" {
        Invoke-PwshScript "direct-seed identity self-test" `
            (Join-Path $Root "scripts\run_qnn_direct_seed_equivalence.ps1") @(
                "-SelfTest")
        Invoke-PwshScript "resumable formal runner self-test" `
            (Join-Path $Root "scripts\run_qnn_resumable_formal.ps1") @(
                "-SelfTest")
        "direct identity plus resume/atomic/identity-rejection/reattach cases ok (temp-only; no device required)"
    }

    if ($Clean) {
        Invoke-Step "gradle-clean" {
            Invoke-Gradle "clean" @(":app:clean")
            "clean done"
        }
    }

    Invoke-Step "unit-tests" {
        Invoke-Gradle "testDebugUnitTest" @(":app:testDebugUnitTest")
        "JVM unit tests ok"
    }

    Invoke-Step "host-tests" {
        if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
            throw "g++ not found on PATH (required by scripts/run_host_tests.ps1)"
        }
        Invoke-PwshScript "run_host_tests" (Join-Path $Root "scripts\run_host_tests.ps1") @()
        "C++ host tests ok (includes qnn_graph_shape_validator)"
    }

    if ($SkipAndroidBuild) {
        Add-Skip "assemble-debug" "-SkipAndroidBuild"
        Add-Skip "assemble-android-test" "-SkipAndroidBuild"
    } else {
        Invoke-Step "assemble-debug" {
            Invoke-Gradle "assembleDebug" @(":app:assembleDebug")
            "QNN-disabled debug APK ok"
        }
        Invoke-Step "assemble-android-test" {
            Invoke-Gradle "assembleDebugAndroidTest" @(":app:assembleDebugAndroidTest")
            "androidTest APK ok"
        }
    }

    if ($WithQairt) {
        if (-not $QairtSdkRoot -or -not $ExpectedBuildId) {
            Add-Result "qairt-check" "FAIL" 0 `
                "-WithQairt requires explicit -QairtSdkRoot and -ExpectedBuildId"
        } else {
            Invoke-Step "qairt-check" {
                Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot `
                    -ExpectedBuildId $ExpectedBuildId
                $pwshExe = Join-Path $PSHOME "pwsh.exe"
                $checkOutput = & $pwshExe -NoProfile -File (Join-Path $Root "scripts\check_qairt.ps1") `
                    -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
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
                    throw "check_qairt: QAIRT SDK not found at $QairtSdkRoot"
                }
                if ($checkExit -eq 4 -or $checkStatus -eq "QAIRT_BUILD_ID_MISMATCH") {
                    throw "check_qairt: expected build ID $ExpectedBuildId not satisfied by $QairtSdkRoot"
                }
                if ($checkExit -eq 5 -or $checkStatus -eq "QAIRT_CORE_INCOMPLETE") {
                    throw "check_qairt: QAIRT core required items are incomplete at $QairtSdkRoot"
                }
                if ($checkExit -notin @(0, 3)) {
                    throw "check_qairt failed with exit code $checkExit ($checkStatus)"
                }
                if (-not $checkStatus) { throw "check_qairt failed with exit code $checkExit" }
                # Inventory completeness is advisory: the following QNN-enabled
                # build and APK audit perform the strict header/library/ABI/hash
                # checks. Exit 3 (inventory incomplete) is not fatal here.
                if ((Get-CheckValue "expected_build_id_match") -ne "true") {
                    throw "check_qairt did not confirm expected_build_id_match=true"
                }
                $resolved = Get-CheckValue "sdk_root"
                if (-not $resolved) { throw "check_qairt did not report sdk_root" }
                $requested = [IO.Path]::GetFullPath($QairtSdkRoot)
                $resolvedFull = [IO.Path]::GetFullPath($resolved)
                if ($requested -ne $resolvedFull) {
                    throw "Explicit QAIRT SDK root was not honored: requested=$requested resolved=$resolvedFull"
                }
                "QAIRT root honored, build ID match ($checkStatus)"
            }
            Invoke-Step "assemble-debug-qnn" {
                Invoke-Gradle "assembleDebug(QNN)" @(
                    ":app:assembleDebug",
                    "-Pphonelm.enableQnn=true",
                    "-Pqairt.sdkRoot=$QairtSdkRoot",
                    "-Pqairt.expectedBuildId=$ExpectedBuildId")
                "QNN-enabled debug APK ok"
            }
            Invoke-Step "apk-audit" {
                Invoke-PwshScript "audit_qnn_apk" (Join-Path $Root "scripts\audit_qnn_apk.ps1") @(
                    "-ApkPath", (Join-Path $Root "app\build\outputs\apk\debug\app-debug.apk"),
                    "-QairtSdkRoot", $QairtSdkRoot,
                    "-ExpectedBuildId", $ExpectedBuildId)
                "APK audit ok"
            }
        }
    }

    Write-Host ""
    Write-Host "===== verify_local summary ====="
    foreach ($row in $results) {
        $line = "{0,-24} {1,-4} {2,8:N1}s  {3}" -f $row.Step, $row.Status, $row.Seconds, $row.Detail
        Write-Host $line
    }
    $passCount = @($results | Where-Object { $_.Status -eq "PASS" }).Count
    $failCount = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
    $skipCount = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
    $totalSeconds = ($results | Measure-Object -Property Seconds -Sum).Sum
    Write-Host ("total {0:N1}s  PASS={1} FAIL={2} SKIP={3}" -f $totalSeconds, $passCount, $failCount, $skipCount)
    if ($failCount -gt 0) { exit 1 }
    exit 0
} finally {
    Pop-Location
}
