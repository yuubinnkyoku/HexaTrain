# PhoneLM / HexaTrain semantic verification entry.
# Profiles are meaning-based execution handles. Which extra profile to run is
# decided by the agent from the change content — never by a file→test table.
#
#   .\scripts\verify.ps1 -Profile Fast      # everyday gate (target 1-3 min)
#   .\scripts\verify.ps1 -Profile Host      # C++ / numerical PC contracts
#   .\scripts\verify.ps1 -Profile Android   # JVM + APK builds
#   .\scripts\verify.ps1 -Profile Qnn       # pinned QAIRT + QNN build + APK audit
#   .\scripts\verify.ps1 -Profile Device    # headless device smoke (Tier 2)
#   .\scripts\verify.ps1 -Profile Formal    # milestone / formal evidence (Full)
#   .\scripts\verify.ps1 -Profile PrGate    # CI / pre-integration (existing policy)
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Fast", "Host", "Android", "Qnn", "Device", "Formal", "PrGate")]
    [string]$Profile,

    [switch]$SkipAndroidBuild,
    [switch]$Clean,
    [switch]$WithQairt,
    [string]$QairtSdkRoot = "",
    [string]$ExpectedBuildId = "",
    [string]$PrGateBaseRef = "",
    # Device profile: headless suite name (Tier 2 BACKGROUND_CORRECTNESS only).
    [string]$DeviceSuite = "device-probe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "verify_common.ps1")
. (Join-Path $PSScriptRoot "qairt_version.ps1")

Initialize-PhoneLmVerifySession
Resolve-PwshExe | Out-Null

$verifyLocal = Join-Path $Root "scripts\verify_local.ps1"

function Invoke-VerifyLocal([string]$Label, [string[]]$Arguments) {
    Invoke-PwshScript $Label $verifyLocal $Arguments
}

function Assert-PinnedQairtArguments {
    if (-not $QairtSdkRoot -or -not $ExpectedBuildId) {
        throw "-Profile $Profile requires explicit -QairtSdkRoot and -ExpectedBuildId (see docs/agent/qairt-policy.md)"
    }
    Assert-PhoneLmQairtPinnedArguments -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
}

function Invoke-NicopediaParityBattery {
    $ParityHost = Join-Path $Root "build\host-tests\nicopedia_parity_policy_test.exe"
    if (-not (Test-Path -LiteralPath $ParityHost)) {
        throw "nicopedia_parity_policy_test.exe missing - run Host profile (or run_host_tests) first"
    }
    $FaultCsvDir = Join-Path $Root "build\reports\nicopedia-parity-policy"
    New-Item -ItemType Directory -Force -Path $FaultCsvDir | Out-Null
    & $ParityHost (Join-Path $FaultCsvDir "synthetic-fault-results.csv")
    if ($LASTEXITCODE -ne 0) { throw "nicopedia parity policy fault battery failed" }
    "nicopedia parity policy fault battery PASS"
}

Push-Location $Root
try {
    switch ($Profile) {
        "Fast" {
            # Everyday gate. Shared implementation lives in verify_local.ps1 -Fast.
            Invoke-Step "verify-local-fast" {
                Invoke-VerifyLocal "verify_local(-Fast)" @("-Fast")
                "everyday gate ok (see verify_local summary above)"
            }
        }
        "Host" {
            # PC-side C++ / numerical contracts. Reuses existing host suites.
            Invoke-Step "host-tests" {
                Assert-GppAvailable
                Invoke-PhoneLmHostSuite
                "C++ host contract + diagnostic suites ok"
            }
            Invoke-Step "nicopedia-parity-policy-host-battery" {
                Invoke-NicopediaParityBattery
            }
        }
        "Android" {
            Invoke-Step "unit-tests" {
                Invoke-Gradle "testDebugUnitTest" @(":app:testDebugUnitTest")
                "JVM unit tests ok"
            }
            Invoke-Step "assemble-debug" {
                Invoke-Gradle "assembleDebug" @(":app:assembleDebug")
                "QNN-disabled debug APK ok"
            }
            Invoke-Step "assemble-android-test" {
                Invoke-Gradle "assembleDebugAndroidTest" @(":app:assembleDebugAndroidTest")
                "androidTest APK ok"
            }
        }
        "Qnn" {
            Assert-PinnedQairtArguments
            Invoke-Step "qairt-check" {
                Invoke-PhoneLmQairtCheck -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
            }
            Invoke-Step "host-tests" {
                Assert-GppAvailable
                Invoke-PhoneLmHostSuite
                "QNN-independent host contracts (shape validator, quantization) ok"
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
        "Device" {
            # Tier 2 only: BACKGROUND_CORRECTNESS headless smoke.
            # Long training / EXCLUSIVE_BENCHMARK / UI bring-up stay Tier 3
            # (see docs/agent/device-test-tiers.md) and are never selected here.
            Assert-PinnedQairtArguments
            Invoke-Step "device-headless-smoke" {
                $suiteArgs = @(
                    "-QairtSdkRoot", $QairtSdkRoot,
                    "-ExpectedBuildId", $ExpectedBuildId,
                    "-Suite", $DeviceSuite,
                    "-TestMode", "BACKGROUND_CORRECTNESS")
                Invoke-PwshScript "run_qnn_headless_tests" `
                    (Join-Path $Root "scripts\run_qnn_headless_tests.ps1") $suiteArgs
                "headless suite=$DeviceSuite identity/finite/fallback checks ok"
            }
        }
        "Formal" {
            # Preserves the historical Full gate (all heavy diagnostic full runs).
            # Compose Qnn / Device profiles separately when formal evidence needs them.
            $formalArgs = @()
            if ($SkipAndroidBuild) { $formalArgs += "-SkipAndroidBuild" }
            if ($Clean) { $formalArgs += "-Clean" }
            if ($WithQairt) {
                $formalArgs += @("-WithQairt", "-QairtSdkRoot", $QairtSdkRoot, "-ExpectedBuildId", $ExpectedBuildId)
            }
            Invoke-Step "verify-local-full" {
                Invoke-VerifyLocal "verify_local(Full)" $formalArgs
                "Full gate ok (see verify_local summary above)"
            }
        }
        "PrGate" {
            $prArgs = @("-PrGate")
            if ($PrGateBaseRef) { $prArgs += @("-PrGateBaseRef", $PrGateBaseRef) }
            if ($SkipAndroidBuild) { $prArgs += "-SkipAndroidBuild" }
            Invoke-Step "verify-local-prgate" {
                Invoke-VerifyLocal "verify_local(-PrGate)" $prArgs
                "PrGate ok (see verify_local summary above)"
            }
        }
    }

    $failCount = Write-PhoneLmVerifySummary "verify.ps1 -Profile $Profile"
    if ($failCount -gt 0) { exit 1 }
    exit 0
} finally {
    Pop-Location
}
