# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
#
# Conservative PR-gate policy for verify_local.ps1 -PrGate.
# Pure classification over repository-relative paths; fail-closed to
# "run all heavy diagnostic fulls" whenever classification or base
# resolution is unsafe. Full gate behavior is owned by verify_local.ps1
# and is not modified by this helper's skip decisions.
#
# Usage:
#   . "$PSScriptRoot\pr_gate_policy.ps1"
#   $plan = Get-PhoneLmPrGatePlan -ChangedPaths $paths
#   $changes = Resolve-PhoneLmPrGateChangeSet -Root $root -ExplicitBase $base
#   pwsh -File scripts/pr_gate_policy.ps1 -SelfTest
param(
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Execution-order names of verify_local.ps1 heavy diagnostic full runs.
# Keep these identical to the Invoke-Step names in verify_local.ps1.
$script:PhoneLmPrGateHeavySteps = @(
    "margin-decomposition-probe"
    "critical-margin-objective-probe"
    "readout-representation-probe"
    "intra-block-readability-run"
    "attention-internal-run"
    "output-projection-run"
    "probe-optimization-run"
)

# Hard run-time prerequisites (not soft/cache fallback).
# probe-optimization loadTaps is fail-closed on attention/intra tap cache
# miss or content-hash mismatch, so those fulls must be in the same plan.
# output-projection extractTapFeatures on cache miss is a soft fallback and
# is intentionally NOT a hard prerequisite here.
$script:PhoneLmPrGateHardPrereqs = @{
    "probe-optimization-run" = @("intra-block-readability-run", "attention-internal-run")
}

function Get-PhoneLmPrGateHeavyStepNames {
    return , @($script:PhoneLmPrGateHeavySteps)
}

# Normalize to forward-slash repository-relative form.
function ConvertTo-PhoneLmRepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    $p = $Path.Trim().Replace("\", "/")
    while ($p.StartsWith("./")) { $p = $p.Substring(2) }
    return $p
}

function Test-PhoneLmPathMatch([string]$Path, [string[]]$Patterns) {
    foreach ($pattern in $Patterns) {
        if ($Path -match $pattern) { return $true }
    }
    return $false
}

# Gate / verification-policy wiring (H). Changing these must select every heavy.
$script:GatePolicyExact = @(
    "scripts/verify_local.ps1"
    "scripts/pr_gate_policy.ps1"
    "docs/agent/verification.md"
    ".github/workflows/verify.yml"
    "AGENTS.md"
)

# Shared production core / dataset / trajectory / multi-diagnostic libs (A).
# Any change fans out to every heavy diagnostic full.
$script:SharedCoreExact = @(
    "app/src/main/cpp/tiny_language_model_cpu.cpp"
    "app/src/main/cpp/tiny_language_model_cpu.h"
    "app/src/main/cpp/cpu_reference_training.cpp"
    "app/src/main/cpp/cpu_reference_training.h"
    "app/src/main/cpp/seed_selection.h"
    "app/src/main/cpp/validation_checkpoint.cpp"
    "app/src/main/cpp/validation_checkpoint.h"
    "app/src/main/cpp/autoregressive_validation.h"
    "app/src/main/cpp/training_stability.h"
    "app/src/main/cpp/qnn/qnn_first_nonfinite_diagnostics.cpp"
    "app/src/main/cpp/qnn/qnn_first_nonfinite_diagnostics.h"
    "host_tests/depth_quality_lib.h"
    "host_tests/margin_analysis_lib.h"
    "host_tests/critical_margin_objective_lib.h"
    "host_tests/critical_margin_training_lib.h"
    "host_tests/readout_probe_lib.h"
    "host_tests/intra_block_readability_lib.h"
    "host_tests/attention_internal_diagnosis_lib.h"
    "host_tests/output_projection_audit_lib.h"
)

# Diagnostic-specific implementation (B): closed to one heavy full
# (hard prerequisites are applied afterwards).
$script:DiagnosticSpecific = @{
    "margin-decomposition-probe" = @(
        "host_tests/margin_decomposition_probe.cpp"
        "scripts/run_l19_margin_decomposition.ps1"
    )
    "critical-margin-objective-probe" = @(
        "host_tests/critical_margin_objective_probe.cpp"
        "scripts/run_critical_margin_objective_benchmark.ps1"
    )
    "readout-representation-probe" = @(
        "host_tests/readout_probe.cpp"
        "scripts/run_l19_readout_probe.ps1"
    )
    "intra-block-readability-run" = @(
        "host_tests/intra_block_readability.cpp"
        "scripts/run_l19_intra_block_readability.ps1"
    )
    "attention-internal-run" = @(
        "host_tests/attention_internal_diagnosis.cpp"
        "scripts/run_l19_attention_internal_diagnosis.ps1"
    )
    "output-projection-run" = @(
        "host_tests/output_projection_audit.cpp"
        "scripts/run_l19_output_projection_audit.ps1"
    )
    "probe-optimization-run" = @(
        "host_tests/probe_optimization_audit.cpp"
        "host_tests/probe_optimization_audit_lib.h"
        "scripts/run_l19_probe_optimization_audit.ps1"
    )
}

# Tracked public evidence / historical result trees (D).
$script:EvidencePrefix = @{
    "docs/results/qnn-l19-first-error-margin-2026-08/" = "margin-decomposition-probe"
    "docs/results/qnn-l19-critical-margin-stabilization-2026-08/" = "critical-margin-objective-probe"
    "docs/results/qnn-l19-readout-representation-diagnosis-2026-08/" = "readout-representation-probe"
    "docs/results/qnn-l19-intra-block-readability-2026-08/" = "intra-block-readability-run"
    "docs/results/qnn-l19-attention-internal-diagnosis-2026-08/" = "attention-internal-run"
    "docs/results/qnn-l19-output-projection-information-audit-2026-08/" = "output-projection-run"
    "docs/results/qnn-l19-probe-optimization-audit-2026-08/" = "probe-optimization-run"
}

# Exporter-only (E). SelfTest remains fixture-contained after PR #5 and does
# not require live diagnostic ReportRoot / PrivateRoot regeneration.
$script:ExporterPattern = '^scripts/export_public_[^/]+\.ps1$'

# Cheap layers that always run in PrGate and do not select heavy fulls.
$script:AlwaysCheapExact = @(
    "scripts/run_host_tests.ps1"
    "scripts/run_host_contract_tests.ps1"
    "scripts/run_host_diagnostic_tests.ps1"
    "scripts/host_test_common.ps1"
    "scripts/check_qairt.ps1"
    "scripts/qairt_version.ps1"
    "scripts/audit_qnn_apk.ps1"
    "scripts/fetch_mnn.ps1"
    "scripts/generate_parameter_metadata.ps1"
    "scripts/import_generation_checkpoint.ps1"
    "scripts/nicopedia_generation_aggregates.ps1"
    "scripts/nicopedia_runner_common.ps1"
    "scripts/test_import_generation_checkpoint.ps1"
)

# Self-test-only diagnostics (no heavy full in verify_local).
$script:SelfTestOnlyExact = @(
    "host_tests/seed_instability_diagnostics.cpp"
    "host_tests/seed_instability_diagnostics_lib.h"
    "host_tests/attention_minimal_cause.cpp"
    "host_tests/attention_minimal_cause_lib.h"
    "host_tests/context_supervision_stability.cpp"
    "host_tests/context_supervision_stability_lib.h"
    "scripts/run_l19_seed_instability_diagnostics.ps1"
    "scripts/run_l19_attention_minimal_cause.ps1"
    "scripts/run_l19_context_supervision_stability.ps1"
)

# Android / QNN / product surfaces that do not own the seven CPU diagnostic
# fulls (G). QNN first-nonfinite is shared core and matched earlier.
$script:AndroidQnnProductPattern = @(
    '^app/src/main/java/'
    '^app/src/test/'
    '^app/src/androidTest/'
    '^app/src/main/res/'
    '^app/src/main/AndroidManifest\.xml$'
    '^app/build\.gradle(\.kts)?$'
    '^app/src/main/cpp/qnn/'
    '^app/src/main/cpp/nicopedia_'
    '^app/src/main/cpp/benchmark_runner'
    '^app/src/main/cpp/mnn_training_test'
    '^app/src/main/cpp/native_bridge\.cpp$'
    '^app/src/main/cpp/training_engine\.'
    '^app/src/main/cpp/transformer_parameter_metadata\.h$'
    '^app/src/main/cpp/transformer_resource_estimator\.h$'
    '^app/src/main/cpp/validation_selection\.h$'
    '^app/src/main/cpp/CMakeLists\.txt$'
    '^gradle/'
    '^gradlew'
    '^build\.gradle(\.kts)?$'
    '^settings\.gradle(\.kts)?$'
    '^metadata/'
    '^scripts/run_qnn_'
    '^scripts/run_nicopedia_'
    '^scripts/run_tiny_lm_'
    '^scripts/run_headwise_'
    '^scripts/run_prepared_'
    '^scripts/run_validation_'
    '^docs/'
    '^README'
    '^LICENSE'
    '^\.gitignore$'
    '^\.github/'
)

# Host-suite-only tooling always exercised by the cheap PrGate layer.
$script:HostSuiteOnlyPattern = @(
    '^host_tests/[^/]+_test\.cpp$'
    '^host_tests/nicopedia_'
    '^host_tests/depth_quality_probe\.cpp$'
    '^host_tests/depth_quality_replay\.cpp$'
    '^host_tests/tiny_lm_'
    '^host_tests/export_transformer_parameter_metadata\.cpp$'
    '^host_tests/hvx_rpc/'
)

function Get-PhoneLmPrGatePlan {
    param(
        [string[]]$ChangedPaths = @(),
        [bool]$ChangeSetFailed = $false
    )

    $runAll = $false
    $reasons = [System.Collections.Generic.List[string]]::new()
    $selected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

    $paths = @(
        @($ChangedPaths)
        | ForEach-Object { ConvertTo-PhoneLmRepoPath $_ }
        | Where-Object { $_ }
        | Sort-Object -Unique
    )

    if ($ChangeSetFailed) {
        $runAll = $true
        $reasons.Add("changed-path detection failed (fail-closed)")
    }

    foreach ($path in $paths) {
        if ($script:GatePolicyExact -contains $path) {
            $runAll = $true
            $reasons.Add("gate/policy change: $path")
            continue
        }
        if ($script:SharedCoreExact -contains $path) {
            $runAll = $true
            $reasons.Add("shared production core / dataset: $path")
            continue
        }
        if ($script:SelfTestOnlyExact -contains $path) {
            $reasons.Add("self-test-only diagnostic (no heavy full): $path")
            continue
        }
        if ($script:AlwaysCheapExact -contains $path) {
            $reasons.Add("always-on cheap layer: $path")
            continue
        }
        if ($path -match $script:ExporterPattern) {
            $reasons.Add("exporter-only (SelfTest fixture-contained): $path")
            continue
        }

        $matchedDiagnostic = $false
        foreach ($stepName in $script:PhoneLmPrGateHeavySteps) {
            foreach ($pattern in @($script:DiagnosticSpecific[$stepName])) {
                if ($path -eq $pattern) {
                    [void]$selected.Add($stepName)
                    $reasons.Add("diagnostic-specific: $path -> $stepName")
                    $matchedDiagnostic = $true
                    break
                }
            }
            if ($matchedDiagnostic) { break }
        }
        if ($matchedDiagnostic) { continue }

        $matchedEvidence = $false
        foreach ($prefix in @($script:EvidencePrefix.Keys)) {
            if ($path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $stepName = $script:EvidencePrefix[$prefix]
                [void]$selected.Add($stepName)
                $reasons.Add("tracked public evidence: $path -> $stepName")
                $matchedEvidence = $true
                break
            }
        }
        if ($matchedEvidence) { continue }

        if ($path.StartsWith("docs/results/", [StringComparison]::OrdinalIgnoreCase)) {
            $runAll = $true
            $reasons.Add("unmapped tracked evidence (fail-closed): $path")
            continue
        }

        if ($path.StartsWith("host_tests/", [StringComparison]::OrdinalIgnoreCase)) {
            if (Test-PhoneLmPathMatch $path $script:HostSuiteOnlyPattern) {
                $reasons.Add("host suite only (no heavy full): $path")
                continue
            }
            $runAll = $true
            $reasons.Add("unknown relevant host path (fail-closed): $path")
            continue
        }
        if ($path.StartsWith("app/src/main/cpp/", [StringComparison]::OrdinalIgnoreCase)) {
            if (Test-PhoneLmPathMatch $path $script:AndroidQnnProductPattern) {
                $reasons.Add("android/qnn/product native (no CPU heavy): $path")
                continue
            }
            $runAll = $true
            $reasons.Add("unknown relevant native path (fail-closed): $path")
            continue
        }
        if ($path.StartsWith("scripts/", [StringComparison]::OrdinalIgnoreCase)) {
            if (Test-PhoneLmPathMatch $path $script:AndroidQnnProductPattern) {
                $reasons.Add("android/qnn/host-runner script (no CPU heavy): $path")
                continue
            }
            $runAll = $true
            $reasons.Add("unknown relevant script (fail-closed): $path")
            continue
        }
        if (Test-PhoneLmPathMatch $path $script:AndroidQnnProductPattern) {
            $reasons.Add("unaffected category: $path")
            continue
        }
        $runAll = $true
        $reasons.Add("unclassified path (fail-closed): $path")
    }

    foreach ($stepName in @($selected)) {
        if ($script:PhoneLmPrGateHardPrereqs.ContainsKey($stepName)) {
            foreach ($prereq in $script:PhoneLmPrGateHardPrereqs[$stepName]) {
                if ($selected.Add($prereq)) {
                    $reasons.Add("hard prerequisite of ${stepName}: $prereq")
                }
            }
        }
    }

    $steps = [ordered]@{}
    foreach ($stepName in $script:PhoneLmPrGateHeavySteps) {
        $shouldRun = $runAll -or $selected.Contains($stepName)
        $stepReason = if ($shouldRun) {
            if ($runAll) {
                "pr gate: fail-closed heavy-all"
            } elseif ($selected.Contains($stepName)) {
                $hit = @($reasons | Where-Object { $_ -match [regex]::Escape($stepName) }) | Select-Object -First 1
                if ($hit) { $hit } else { "pr gate: selected" }
            } else {
                "pr gate: selected"
            }
        } else {
            "pr gate: unaffected"
        }
        $steps[$stepName] = [pscustomobject]@{
            Name   = $stepName
            Run    = [bool]$shouldRun
            Reason = $stepReason
        }
    }

    return [pscustomobject]@{
        RunAllHeavy    = [bool]$runAll
        HeavySteps     = $steps
        Reasons        = @($reasons)
        ChangedPaths   = $paths
        HeavyStepNames = @($script:PhoneLmPrGateHeavySteps)
    }
}

function Resolve-PhoneLmPrGateChangeSet {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$ExplicitBase = ""
    )

    $result = [pscustomobject]@{
        Ok           = $false
        Base         = ""
        ChangedPaths = @()
        Error        = ""
    }

    $prev = Get-Location
    try {
        Set-Location -LiteralPath $Root

        $base = ""
        if (-not [string]::IsNullOrWhiteSpace($ExplicitBase)) {
            $base = $ExplicitBase.Trim()
        } elseif (-not [string]::IsNullOrWhiteSpace($env:GITHUB_BASE_REF)) {
            $base = "origin/$($env:GITHUB_BASE_REF.Trim())"
        } else {
            $base = "origin/main"
        }

        # Fail closed when the base ref is missing locally (no auto-fetch).
        $baseCommit = & git rev-parse --verify "$base^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $baseCommit) {
            $result.Error = "base ref not available locally: $base"
            $result.Base = $base
            return $result
        }

        $mergeBase = & git merge-base $base "HEAD" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $mergeBase) {
            $result.Error = "merge-base with $base not resolvable"
            $result.Base = $base
            return $result
        }

        $changed = [System.Collections.Generic.List[string]]::new()
        $diffOutput = & git diff --name-only $mergeBase 2>$null
        if ($LASTEXITCODE -ne 0) {
            $result.Error = "git diff --name-only failed"
            $result.Base = $base
            return $result
        }
        foreach ($line in @($diffOutput)) {
            if ($line) { $changed.Add($line) }
        }
        $untracked = & git ls-files --others --exclude-standard 2>$null
        if ($LASTEXITCODE -ne 0) {
            $result.Error = "git ls-files --others failed"
            $result.Base = $base
            return $result
        }
        foreach ($line in @($untracked)) {
            if ($line) { $changed.Add($line) }
        }

        $result.Ok = $true
        $result.Base = $base
        $result.ChangedPaths = @(
            $changed
            | ForEach-Object { ConvertTo-PhoneLmRepoPath $_ }
            | Where-Object { $_ }
            | Sort-Object -Unique
        )
        return $result
    } finally {
        Set-Location -LiteralPath $prev.Path
    }
}

function Format-PhoneLmPrGatePlanLog {
    param(
        [Parameter(Mandatory = $true)]$Plan,
        [string]$Base = ""
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("===== PR gate plan =====")
    $baseLabel = if ($Base) { $Base } else { "(unresolved)" }
    $lines.Add("base=$baseLabel")
    $changed = @($Plan.ChangedPaths)
    $lines.Add("changed_paths=$($changed.Count)")
    foreach ($path in $changed) {
        $lines.Add("changed: $path")
    }
    $lines.Add("run_all_heavy=$(if ($Plan.RunAllHeavy) { 'true' } else { 'false' })")
    foreach ($stepName in $Plan.HeavyStepNames) {
        $step = $Plan.HeavySteps[$stepName]
        $action = if ($step.Run) { "RUN" } else { "SKIP" }
        $lines.Add("heavy: $stepName $action reason=$($step.Reason)")
    }
    foreach ($reason in @($Plan.Reasons)) {
        $lines.Add("reason: $reason")
    }
    return ($lines -join "`n")
}

function Assert-PhoneLmPrGatePlanCase {
    param(
        [string]$Name,
        [string[]]$Paths,
        [bool]$ExpectAll,
        [string[]]$ExpectRun,
        [string[]]$ExpectSkip,
        [bool]$ChangeSetFailed = $false
    )
    $plan = Get-PhoneLmPrGatePlan -ChangedPaths $Paths -ChangeSetFailed $ChangeSetFailed
    $errors = @()
    if ($plan.RunAllHeavy -ne $ExpectAll) {
        $errors += "RunAllHeavy=$($plan.RunAllHeavy) expected=$ExpectAll"
    }
    foreach ($step in $ExpectRun) {
        if (-not $plan.HeavySteps[$step].Run) {
            $errors += "expected RUN $step"
        }
    }
    foreach ($step in $ExpectSkip) {
        if ($plan.HeavySteps[$step].Run) {
            $errors += "expected SKIP $step"
        }
    }
    if (-not $ExpectAll) {
        $expectedSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($s in $ExpectRun) { [void]$expectedSet.Add($s) }
        foreach ($stepName in $plan.HeavyStepNames) {
            $isRun = $plan.HeavySteps[$stepName].Run
            $shouldBe = $expectedSet.Contains($stepName)
            if ($isRun -ne $shouldBe) {
                $errors += "step $stepName Run=$isRun expected=$shouldBe"
            }
        }
    }
    $reversed = @()
    foreach ($p in $Paths) { $reversed = @($p) + $reversed }
    $plan2 = Get-PhoneLmPrGatePlan -ChangedPaths $reversed -ChangeSetFailed $ChangeSetFailed
    foreach ($stepName in $plan.HeavyStepNames) {
        if ($plan.HeavySteps[$stepName].Run -ne $plan2.HeavySteps[$stepName].Run) {
            $errors += "order-dependent selection for $stepName"
        }
    }
    if ($plan.RunAllHeavy -ne $plan2.RunAllHeavy) {
        $errors += "order-dependent RunAllHeavy"
    }
    if ($errors.Count -gt 0) {
        $script:PrGateSelfTestFailures.Add("${Name}: " + ($errors -join "; "))
    } else {
        $script:PrGateSelfTestPassed++
        Write-Host "PASS $Name"
    }
}

function Invoke-PhoneLmPrGatePolicySelfTest {
    $script:PrGateSelfTestFailures = [System.Collections.Generic.List[string]]::new()
    $script:PrGateSelfTestPassed = 0

    $none = @()
    $allSteps = @($script:PhoneLmPrGateHeavySteps)

    Assert-PhoneLmPrGatePlanCase -Name "1-exporter-only" -ExpectAll:$false `
        -Paths @("scripts/export_public_qnn_l19_readout_results.ps1") `
        -ExpectRun $none -ExpectSkip $allSteps

    Assert-PhoneLmPrGatePlanCase -Name "2-docs-only" -ExpectAll:$false `
        -Paths @("docs/qnn-tiny-language-model-training.md", "docs/agent/numerical-evidence.md") `
        -ExpectRun $none -ExpectSkip $allSteps

    Assert-PhoneLmPrGatePlanCase -Name "3-android-kotlin-only" -ExpectAll:$false `
        -Paths @("app/src/main/java/com/yuubinnkyoku/phonelm/TrainingDataset.kt") `
        -ExpectRun $none -ExpectSkip $allSteps

    Assert-PhoneLmPrGatePlanCase -Name "4-qnn-only" -ExpectAll:$false `
        -Paths @("app/src/main/cpp/qnn/qnn_graph_shape_validator.cpp") `
        -ExpectRun $none -ExpectSkip $allSteps

    Assert-PhoneLmPrGatePlanCase -Name "5-shared-cpu-core" -ExpectAll:$true `
        -Paths @("app/src/main/cpp/tiny_language_model_cpu.cpp") `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "6-shared-dataset" -ExpectAll:$true `
        -Paths @("app/src/main/cpp/autoregressive_validation.h") `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "7-readout-specific" -ExpectAll:$false `
        -Paths @("host_tests/readout_probe.cpp") `
        -ExpectRun @("readout-representation-probe") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "intra-block-readability-run", "attention-internal-run",
            "output-projection-run", "probe-optimization-run")

    Assert-PhoneLmPrGatePlanCase -Name "8-critical-margin-specific" -ExpectAll:$false `
        -Paths @("host_tests/critical_margin_objective_probe.cpp") `
        -ExpectRun @("critical-margin-objective-probe") `
        -ExpectSkip @("margin-decomposition-probe", "readout-representation-probe",
            "intra-block-readability-run", "attention-internal-run",
            "output-projection-run", "probe-optimization-run")

    Assert-PhoneLmPrGatePlanCase -Name "9-probe-optimization-with-prereqs" -ExpectAll:$false `
        -Paths @("host_tests/probe_optimization_audit.cpp") `
        -ExpectRun @("probe-optimization-run", "intra-block-readability-run", "attention-internal-run") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "readout-representation-probe", "output-projection-run")

    Assert-PhoneLmPrGatePlanCase -Name "10-readout-public-evidence" -ExpectAll:$false `
        -Paths @("docs/results/qnn-l19-readout-representation-diagnosis-2026-08/trajectory-anchors.csv") `
        -ExpectRun @("readout-representation-probe") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "intra-block-readability-run", "attention-internal-run",
            "output-projection-run", "probe-optimization-run")

    Assert-PhoneLmPrGatePlanCase -Name "11-verify-local-policy" -ExpectAll:$true `
        -Paths @("scripts/verify_local.ps1", "scripts/pr_gate_policy.ps1") `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "12-unknown-host-path" -ExpectAll:$true `
        -Paths @("host_tests/new_mystery_probe.cpp") `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "13-mixed-exporter-readout" -ExpectAll:$false `
        -Paths @("scripts/export_public_qnn_l19_readout_results.ps1", "host_tests/readout_probe.cpp") `
        -ExpectRun @("readout-representation-probe") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "intra-block-readability-run", "attention-internal-run",
            "output-projection-run", "probe-optimization-run")

    Assert-PhoneLmPrGatePlanCase -Name "14-mixed-android-shared-core" -ExpectAll:$true `
        -Paths @("app/src/main/java/com/yuubinnkyoku/phonelm/TrainingDataset.kt",
            "app/src/main/cpp/tiny_language_model_cpu.cpp") `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "15-change-set-failed-fail-closed" -ExpectAll:$true `
        -Paths @() -ChangeSetFailed:$true `
        -ExpectRun $allSteps -ExpectSkip $none

    Assert-PhoneLmPrGatePlanCase -Name "16-output-projection-soft-tap" -ExpectAll:$false `
        -Paths @("host_tests/output_projection_audit.cpp") `
        -ExpectRun @("output-projection-run") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "readout-representation-probe", "intra-block-readability-run",
            "attention-internal-run", "probe-optimization-run")

    Assert-PhoneLmPrGatePlanCase -Name "17-mixed-output-and-probe" -ExpectAll:$false `
        -Paths @("host_tests/output_projection_audit.cpp", "host_tests/probe_optimization_audit.cpp") `
        -ExpectRun @("output-projection-run", "probe-optimization-run",
            "intra-block-readability-run", "attention-internal-run") `
        -ExpectSkip @("margin-decomposition-probe", "critical-margin-objective-probe",
            "readout-representation-probe")

    Assert-PhoneLmPrGatePlanCase -Name "18-multi-shared-lib-fanout" -ExpectAll:$true `
        -Paths @("host_tests/readout_probe_lib.h") `
        -ExpectRun $allSteps -ExpectSkip $none

    if ($script:PrGateSelfTestFailures.Count -gt 0) {
        foreach ($failure in $script:PrGateSelfTestFailures) {
            Write-Host "FAIL $failure"
        }
        throw "pr gate policy self-test failed ($($script:PrGateSelfTestFailures.Count) failure(s))"
    }
    Write-Host "pr-gate-policy-self-test=PASS ($($script:PrGateSelfTestPassed) cases)"
    return "deterministic classifier matrix PASS ($($script:PrGateSelfTestPassed) cases)"
}

if ($SelfTest) {
    Invoke-PhoneLmPrGatePolicySelfTest | Out-Null
    exit 0
}
