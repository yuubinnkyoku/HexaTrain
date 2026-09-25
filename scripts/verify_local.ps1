# PhoneLM common local verification base gate.
# Default: no physical device, no QAIRT SDK, no writes outside build/ and the index.
# All required steps must PASS; QNN/device/publication changes have extra gates.
# Prefer scripts/verify.ps1 -Profile ... as the semantic entry; this file remains
# the shared implementation for Fast / PrGate / Full (Formal).
# See AGENTS.md and docs/agent/verification.md.
param(
    # Final gate only for docs/scripts-only changes; otherwise a fast pre-check.
    [switch]$SkipAndroidBuild,
    # Run gradle :app:clean first. Slow (CMake/NDK rebuild). Use only when required.
    [switch]$Clean,
    # Fast local gate for iterative development (target 1-3 min).
    [switch]$Fast,
    # Additionally verify the QAIRT SDK and a QNN-enabled build + APK audit.
    [switch]$WithQairt,
    [string]$QairtSdkRoot = "",
    [string]$ExpectedBuildId = "",
    # CI / pre-integration profile: keep cheap correctness layers unconditional
    # and select heavy diagnostic fulls by fail-closed path policy. Full gate
    # (no switches) remains unchanged and still runs every heavy full.
    [switch]$PrGate,
    # Optional explicit base ref for -PrGate changed-path discovery.
    [string]$PrGateBaseRef = ""
)

if ($Fast -and $PrGate) {
    throw "-Fast and -PrGate are mutually exclusive (Fast is iterative; PrGate is pre-integration)"
}

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "verify_common.ps1")
. (Join-Path $PSScriptRoot "qairt_version.ps1")
Initialize-PhoneLmVerifySession

$script:PhoneLmPrGatePlan = $null
$script:PhoneLmPrGateEnabled = [bool]$PrGate
if ($PrGate) {
    . (Join-Path $PSScriptRoot "pr_gate_policy.ps1")
}

# True when a heavy diagnostic full should execute. Full and Fast keep their
# historical meaning. Only -PrGate consults the dependency policy; failure to
# plan safely is fail-closed (run the full).
function Test-HeavyStepSelected([string]$StepName) {
    if (-not $script:PhoneLmPrGateEnabled) { return $true }
    $plan = $script:PhoneLmPrGatePlan
    if ($null -eq $plan -or $plan.RunAllHeavy) { return $true }
    return [bool]$plan.HeavySteps[$StepName].Run
}

function Get-HeavyStepSkipReason([string]$StepName) {
    $plan = $script:PhoneLmPrGatePlan
    if ($null -ne $plan -and -not $plan.RunAllHeavy -and
        $null -ne $plan.HeavySteps[$StepName]) {
        $reason = $plan.HeavySteps[$StepName].Reason
        if ($reason) { return $reason }
    }
    return "pr gate: unaffected"
}

function Invoke-HeavyOrSkip([string]$Name, [string]$FastSkipReason, [scriptblock]$Action) {
    if ($Fast) {
        Add-Skip $Name $FastSkipReason
        return
    }
    if (-not (Test-HeavyStepSelected $Name)) {
        Add-Skip $Name (Get-HeavyStepSkipReason $Name)
        return
    }
    Invoke-Step $Name $Action
}

function Invoke-PowerShellParserCheck() {
    $scripts = Get-ChildItem (Join-Path $Root "scripts\*.ps1")
    $bad = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $scripts) {
        $parseErrs = $null
        [void][System.Management.Automation.PSParser]::Tokenize(
            (Get-Content -LiteralPath $item.FullName -Raw), [ref]$parseErrs)
        if ($parseErrs.Count -gt 0) {
            $bad.Add("$($item.Name): $($parseErrs[0].Message)")
        }
    }
    if ($bad.Count -gt 0) {
        throw ("PowerShell parser errors:`n" + ($bad -join "`n"))
    }
    "$($scripts.Count) PowerShell scripts parsed cleanly"
}

Push-Location $Root
try {
    # Fail-fast: almost every step below relies on PowerShell 7.
    Resolve-PwshExe | Out-Null

    if ($script:PhoneLmPrGateEnabled) {
        $changeSet = Resolve-PhoneLmPrGateChangeSet -Root $Root -ExplicitBase $PrGateBaseRef
        $planPaths = @($changeSet.ChangedPaths)
        $planFailed = -not $changeSet.Ok
        if ($planFailed) {
            Write-Host "PR gate change-set resolution failed: $($changeSet.Error)"
        }
        $script:PhoneLmPrGatePlan = Get-PhoneLmPrGatePlan `
            -ChangedPaths $planPaths -ChangeSetFailed $planFailed
        Write-Host (Format-PhoneLmPrGatePlanLog -Plan $script:PhoneLmPrGatePlan -Base $changeSet.Base)
    }

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
            @(
                git diff --name-only --diff-filter=ACMR
                git diff --cached --name-only --diff-filter=ACMR
                git ls-files --others --exclude-standard
            ) | Sort-Object -Unique | Where-Object {
                $_ -and $_ -ne "scripts/verify_local.ps1"
            }
        )
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

    if ($Fast) {
        Invoke-Step "powershell-parser-check" {
            Invoke-PowerShellParserCheck
        }
    }

    Invoke-Step "qairt-selection-self-test" {
        Invoke-PwshScript "check_qairt self-test" (Join-Path $Root "scripts\check_qairt.ps1") @(
            "-SelfTest")
        "pinned arguments, root selection, core/advisory classification ok (temp-only)"
    }

    if ($Fast) {
        Add-Skip "nicopedia-generation-self-test" "fast mode"
    } else {
        Invoke-Step "nicopedia-generation-self-test" {
            Invoke-PwshScript "Nicopedia HTP generation runner self-test" `
                (Join-Path $Root "scripts\run_nicopedia_htp_generate.ps1") @(
                    "-SelfTest",
                    "-QairtSdkRoot", "C:\Qualcomm\AIStack\QAIRT\2.48.40.260702",
                    "-ExpectedBuildId", "2.48.40.260702151143")
            Invoke-PwshScript "Nicopedia HTP generation public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_htp_generation_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "Nicopedia HTP 1000-step generation public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_htp_1000step_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "Nicopedia HTP parity policy public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_htp_parity_policy.ps1") @(
                    "-SelfTest")
            "runner allow-list/display self-tests and exporter leak guards PASS (temp-only; no device required)"
        }
    }

    # Cheap policy self-test: Fast and non-Fast both run it.
    Invoke-Step "pr-gate-policy-self-test" {
        Invoke-PwshScript "pr gate policy self-test" `
            (Join-Path $Root "scripts\pr_gate_policy.ps1") @("-SelfTest")
        "deterministic classifier matrix PASS (fail-closed unknown/shared)"
    }

    # Generated .commandcode/skills must match .agents/skills SSOT.
    Invoke-Step "agent-skills-sync-check" {
        Invoke-PwshScript "sync_agent_skills" `
            (Join-Path $Root "scripts\sync_agent_skills.ps1") @("-Check")
        "agent skill trees in sync"
    }

    Invoke-HeavyOrSkip "margin-decomposition-probe" "fast mode" {
        Invoke-PwshScript "l19 first-error/margin decomposition probe" `
            (Join-Path $Root "scripts\run_l19_margin_decomposition.ps1") @()
        "deterministic CPU reports regenerated (private margin-tokens included)"
    }

    # Objective + training are one step: training is a hard prerequisite chain
    # against the objective baseline directory produced immediately above.
    Invoke-HeavyOrSkip "critical-margin-objective-probe" "fast mode" {
        Invoke-PwshScript "critical margin objective probe" `
            (Join-Path $Root "scripts\run_critical_margin_objective_benchmark.ps1") @()
        Invoke-PwshScript "critical margin training probe" `
            (Join-Path $Root "scripts\run_critical_margin_objective_benchmark.ps1") @(
                "-Train", "-BaselineDir", (Join-Path $Root "build\reports\qnn-critical-margin-objective"),
                "-ReportRoot", (Join-Path $Root "build\reports\qnn-critical-margin-training"))
        "deterministic CPU objective/training reports regenerated (private)"
    }

    if ($Fast) {
        Add-Skip "readout-probe-self-test" "fast mode"
    } else {
        Invoke-Step "readout-probe-self-test" {
            Invoke-PwshScript "readout probe self-test" `
                (Join-Path $Root "scripts\run_l19_readout_probe.ps1") @("-SelfTest")
            "readout probe self-test PASS (private)"
        }
    }

    Invoke-HeavyOrSkip "readout-representation-probe" "fast mode" {
        Invoke-PwshScript "readout/representation diagnosis run" `
            (Join-Path $Root "scripts\run_l19_readout_probe.ps1") @()
        "deterministic CPU readout/representation reports regenerated (private evidence; exporter SelfTest is fixture-contained and does not require these live reports)"
    }

    if ($Fast) {
        Add-Skip "intra-block-readability-self-test" "fast mode"
    } else {
        Invoke-Step "intra-block-readability-self-test" {
            Invoke-PwshScript "intra-block readability self-test" `
                (Join-Path $Root "scripts\run_l19_intra_block_readability.ps1") @("-SelfTest")
            "intra-block readability self-test PASS (private)"
        }
    }

    Invoke-HeavyOrSkip "intra-block-readability-run" "fast mode" {
        Invoke-PwshScript "intra-block readability diagnosis run" `
            (Join-Path $Root "scripts\run_l19_intra_block_readability.ps1") @()
        "deterministic CPU intra-block readability reports regenerated (private evidence; exporter SelfTest is fixture-contained and does not require these live reports)"
    }

    if ($Fast) {
        Add-Skip "attention-internal-self-test" "fast mode"
    } else {
        Invoke-Step "attention-internal-self-test" {
            Invoke-PwshScript "attention-internal diagnosis self-test" `
                (Join-Path $Root "scripts\run_l19_attention_internal_diagnosis.ps1") @("-SelfTest")
            "attention-internal diagnosis self-test PASS (private)"
        }
    }

    Invoke-HeavyOrSkip "attention-internal-run" "fast mode" {
        Invoke-PwshScript "attention-internal diagnosis run" `
            (Join-Path $Root "scripts\run_l19_attention_internal_diagnosis.ps1") @()
        "deterministic CPU attention-internal diagnosis reports regenerated (private evidence; exporter SelfTest is fixture-contained and does not require these live reports)"
    }

    if ($Fast) {
        Add-Skip "output-projection-self-test" "fast mode"
    } else {
        Invoke-Step "output-projection-self-test" {
            Invoke-PwshScript "output-projection audit self-test" `
                (Join-Path $Root "scripts\run_l19_output_projection_audit.ps1") @("-SelfTest")
            "output-projection audit self-test PASS (private)"
        }
    }

    # Soft tap dependency: cache miss extracts features in-process and is not
    # a hard prerequisite on attention-internal-run.
    Invoke-HeavyOrSkip "output-projection-run" "fast mode" {
        Invoke-PwshScript "output-projection audit run" `
            (Join-Path $Root "scripts\run_l19_output_projection_audit.ps1") @()
        "deterministic CPU output-projection audit reports regenerated (private evidence; exporter SelfTest is fixture-contained and does not require these live reports)"
    }

    if ($Fast) {
        Add-Skip "probe-optimization-self-test" "fast mode"
    } else {
        Invoke-Step "probe-optimization-self-test" {
            Invoke-PwshScript "probe-optimization audit self-test" `
                (Join-Path $Root "scripts\run_l19_probe_optimization_audit.ps1") @("-SelfTest")
            "probe-optimization audit self-test PASS (private)"
        }
    }

    # Hard prerequisite: loadTaps requires attention-internal and intra-block
    # private-tap caches (miss/hash mismatch is fatal). Policy expands those
    # fulls into the same PrGate plan.
    Invoke-HeavyOrSkip "probe-optimization-run" "fast mode" {
        Invoke-PwshScript "probe-optimization audit run" `
            (Join-Path $Root "scripts\run_l19_probe_optimization_audit.ps1") @()
        "deterministic CPU probe-optimization audit reports regenerated (private evidence; exporter SelfTest is fixture-contained and does not require these live reports)"
    }

    if ($Fast) {
        Add-Skip "seed-instability-diagnostics-self-test" "fast mode"
    } else {
        Invoke-Step "seed-instability-diagnostics-self-test" {
            Invoke-PwshScript "seed-instability diagnostics self-test" `
                (Join-Path $Root "scripts\run_l19_seed_instability_diagnostics.ps1") @("-SelfTest")
            "deterministic rerun, intervention scope, corrected TRAIN contract, and negative branch control fixtures PASS"
        }
    }

    if ($Fast) {
        Add-Skip "attention-minimal-cause-self-test" "fast mode"
    } else {
        Invoke-Step "attention-minimal-cause-self-test" {
            Invoke-PwshScript "attention minimal-cause self-test" `
                (Join-Path $Root "scripts\run_l19_attention_minimal_cause.ps1") @("-SelfTest")
            "no-op parity, fixed patterns, branch scale, freeze scope, group identity, and deterministic training fixture PASS"
        }
    }

    if ($Fast) {
        Add-Skip "context-supervision-stability-self-test" "fast mode"
    } else {
        Invoke-Step "context-supervision-stability-self-test" {
            Invoke-PwshScript "context supervision stability self-test" `
                (Join-Path $Root "scripts\run_l19_context_supervision_stability.ps1") @("-SelfTest")
            "dataset determinism, matched histogram, target contract, canonical no-op, ordinary Attention, and curriculum multiset fixtures PASS"
        }
    }

    if ($Fast) {
        Add-Skip "public-exporter-self-test" "fast mode"
    } else {
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
            Invoke-PwshScript "output-projection audit public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_l19_output_projection_audit_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "probe-optimization audit public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_l19_probe_optimization_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "seed-instability root-cause public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_l19_seed_instability_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "attention minimal-cause public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_l19_attention_minimal_cause.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "context supervision stability public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_l19_context_supervision_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "Nicopedia real-text public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "Nicopedia real-text HTP public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_htp_results.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "Nicopedia HTP long-training public exporter self-test" `
                (Join-Path $Root "scripts\export_public_qnn_nicopedia_htp_long_training_results.ps1") @(
                    "-SelfTest")
            "allow-list exports, manifest consistency, and negative rejection ok (temp-only)"
        }
    }

    if ($Fast) {
        Add-Skip "resumable-formal-runner-self-test" "fast mode"
    } else {
        Invoke-Step "resumable-formal-runner-self-test" {
            Invoke-PwshScript "headless runner physical-identity self-test" `
                (Join-Path $Root "scripts\run_qnn_headless_tests.ps1") @(
                    "-QairtSdkRoot", $PhoneLmQairtSdkRoot,
                    "-ExpectedBuildId", $PhoneLmQairtBuildId,
                    "-SelfTest")
            Invoke-PwshScript "direct-seed identity self-test" `
                (Join-Path $Root "scripts\run_qnn_direct_seed_equivalence.ps1") @(
                    "-SelfTest")
            Invoke-PwshScript "resumable formal runner self-test" `
                (Join-Path $Root "scripts\run_qnn_resumable_formal.ps1") @(
                    "-SelfTest")
            "direct identity plus resume/atomic/identity-rejection/reattach cases ok (temp-only; no device required)"
        }
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

    if ($Fast) {
        # Metadata staleness + CPU reference only. Full Host suite is
        # verify.ps1 -Profile Host / non-Fast verify_local.
        Invoke-Step "host-fast-contract" {
            Assert-GppAvailable
            Invoke-PhoneLmHostContractFast
            "metadata staleness + CPU reference contracts ok"
        }
    } else {
        Invoke-Step "host-tests" {
            Assert-GppAvailable
            Invoke-PhoneLmHostSuite
            "C++ host tests ok (includes qnn_graph_shape_validator and nicopedia parity policy fault battery)"
        }
    }

    if ($Fast) {
        Add-Skip "nicopedia-parity-policy-host-battery" "fast mode"
    } else {
        Invoke-Step "nicopedia-parity-policy-host-battery" {
            $ParityHost = Join-Path $Root "build\host-tests\nicopedia_parity_policy_test.exe"
            if (-not (Test-Path -LiteralPath $ParityHost)) {
                throw "nicopedia_parity_policy_test.exe missing - run_host_tests must run first"
            }
            $FaultCsvDir = Join-Path $Root "build\reports\nicopedia-parity-policy"
            New-Item -ItemType Directory -Force -Path $FaultCsvDir | Out-Null
            & $ParityHost (Join-Path $FaultCsvDir "synthetic-fault-results.csv")
            if ($LASTEXITCODE -ne 0) { throw "nicopedia parity policy fault battery failed" }
            "nicopedia parity policy fault battery PASS (synthetic-fault-results.csv regenerated)"
        }
    }

    if ($SkipAndroidBuild) {
        Add-Skip "assemble-debug" "-SkipAndroidBuild"
        Add-Skip "assemble-android-test" "-SkipAndroidBuild"
    } elseif ($Fast) {
        Add-Skip "assemble-debug" "fast mode"
        Add-Skip "assemble-android-test" "fast mode"
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
                # Inventory completeness is advisory: the following QNN-enabled
                # build and APK audit perform the strict header/library/ABI/hash
                # checks. Exit 3 (inventory incomplete) is not fatal here.
                Invoke-PhoneLmQairtCheck -SdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId
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

    $failCount = Write-PhoneLmVerifySummary "verify_local summary"
    if ($failCount -gt 0) { exit 1 }
    exit 0
} finally {
    Pop-Location
}
