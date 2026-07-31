# SPDX-License-Identifier: Apache-2.0
# Resumable per-seed formal runner for the generic depth/head QNN HTP suite.
#
# Design contract:
# - Each seed runs in its own Android process. The native generic loop is
#   deterministic and seed-independent, so the process for seed k is launched
#   with correctness_interval=k and only the seed_k_* values are harvested.
#   This keeps the native training implementation untouched.
# - Every completed seed result is pulled to the host immediately, validated
#   against a canonical identity (configuration hash, git HEAD, APK SHA-256,
#   QAIRT Build ID, steps, result schema), and promoted atomically
#   (result.tmp -> validate -> result.txt).
# - ADB transport interruptions are not numeric failures. On reconnect the
#   currently-online physical device is re-resolved and the app-private run
#   context (run id + config hash) must match; otherwise the formal stops.
# - Mismatched, partial, terminal=false, corrupted or duplicate results are
#   rejected fail-closed and never count as completed.
# - Device serials, ADB endpoints, run ids and APK hashes stay private under
#   build/reports; only aggregate CSVs in docs/results are publishable.
[CmdletBinding()]
param(
    [string]$QairtSdkRoot = '',
    [string]$ExpectedBuildId = '2.48.40.260702151143',
    [ValidatePattern('^[a-z0-9][a-z0-9._-]*$')][string]$ConfigurationId = '',
    [ValidateRange(1, 65536)][int]$SequenceLength = 8,
    [ValidateRange(13, 65536)][int]$VocabularySize = 32,
    [ValidateRange(1, 65536)][int]$EmbeddingDimension = 16,
    [ValidateRange(1, 65536)][int]$FeedForwardDimension = 32,
    [ValidateRange(1, 65536)][int]$NumLayers = 1,
    [ValidateRange(1, 65536)][int]$NumHeads = 1,
    [ValidateRange(1, 100000)][int]$Steps = 320,
    [ValidateRange(1, 5)][int]$Seeds = 5,
    [ValidatePattern('^[0-9]+(\.[0-9]+)?$')][string]$LearningRate = '0.003',
    [ValidateSet('UI_VALIDATION', 'CORRECTNESS')][string]$TestMode = 'UI_VALIDATION',
    [ValidateRange(600, 86400)][int]$TimeoutSecondsPerAttempt = 21600,
    [ValidateRange(1, 10)][int]$MaxAttemptsPerSeed = 3,
    [ValidateRange(60, 21600)][int]$ReconnectGraceSeconds = 3600,
    # Attempts start only when the battery is below this ceiling so consecutive
    # attempts do not accumulate heat past the formal 45 C stop condition.
    [ValidateRange(35.0, 44.9)][double]$ResumeBatteryCeilingC = 41.0,
    [ValidateRange(5, 360)][int]$CoolDownGraceMinutes = 90,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$')][string]$RunId = '',
    [switch]$InstallAuditedApk,
    [switch]$AllowFailure,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ResultSchemaVersion = 1
$MaxSeedCount = 5

# ---------------------------------------------------------------------------
# Pure helpers (device-independent, exercised by -SelfTest)
# ---------------------------------------------------------------------------

function Get-Sha256Hex([string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-CanonicalConfigHash {
    param(
        [int]$T, [int]$V, [int]$D, [int]$Ffn, [int]$L, [int]$H,
        [string]$Lr, [int]$StepCount
    )
    if (($D % $H) -ne 0) { throw 'embedding dimension must be divisible by heads' }
    $canonical = "phonelm-formal-config-v1|B=1|T=$T|V=$V|D=$D|FFN=$Ffn|L=$L|H=$H" +
        "|headDim=$($D / $H)|lr=$Lr|steps=$StepCount|clip=disabled" +
        '|optimizer=ADAM|mode=QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC|candidate=3'
    return Get-Sha256Hex $canonical
}

function Protect-SeedResult([System.Collections.IDictionary]$Fields) {
    # Canonical key=val lines plus a trailing payload hash. Keys keep a fixed
    # order so the payload hash is stable across machines and PowerShell builds.
    $order = @(
        'result_schema', 'terminal', 'classification', 'device_status',
        'seed', 'steps', 'attempt', 'device_seed_count', 'test_mode',
        'step_completed', 'initial_loss', 'final_loss', 'final_accuracy',
        'all_steps_finite', 'final_evaluation_finite', 'cpu_all_steps_finite',
        'qnn_nonzero_count', 'nonfinite_count', 'qnn_execute_count',
        'parameter_hash', 'logits_hash', 'oracle_exact', 'free_exact',
        'app_write_unchanged', 'poison_remaining',
        'first_bad_seed', 'first_bad_step', 'first_bad_phase', 'first_bad_tensor',
        'qnn_execute_result',
        'battery_temperature_before_c', 'battery_temperature_after_c',
        'thermal_status_before', 'thermal_status_after',
        'foreground_service_isForeground', 'foreground_service_type_dataSync',
        'fg_service_flag_present', 'cached_empty_seen',
        'progress_updates_seen', 'ongoing_during_run', 'ongoing_after_completion',
        'wall_time_ms', 'started_utc', 'completed_utc'
    )
    $lines = foreach ($key in $order) {
        if (-not $Fields.Contains($key)) { throw "seed result is missing field: $key" }
        $value = [string]$Fields[$key]
        if ($value -match "[\r\n]") { throw "seed result field $key is not single-line" }
        "$key=$value"
    }
    $identity = foreach ($key in @('run_id', 'configuration_id', 'config_hash',
            'git_head', 'apk_sha256', 'qairt_build_id', 'result_schema_id')) {
        # identity block is always last before the payload hash
        if (-not $Fields.Contains($key)) { throw "seed result is missing identity field: $key" }
        "$key=$([string]$Fields[$key])"
    }
    $payload = (($lines + $identity) -join "`n")
    return $payload + "`nresult_payload_hash=$(Get-Sha256Hex $payload)`n"
}

function Read-SeedResult([string]$Content) {
    $fields = [ordered]@{}
    foreach ($line in ($Content -split "`n")) {
        $trimmed = $line.TrimEnd("`r")
        if ($trimmed.Length -eq 0) { continue }
        $pair = $trimmed -split '=', 2
        if ($pair.Count -ne 2) { return $null }
        $fields[$pair[0]] = $pair[1]
    }
    return $fields
}

function Test-SeedResultContent {
    param(
        [string]$Content,
        [System.Collections.IDictionary]$Identity,
        [int]$ExpectedSeed
    )
    $fields = Read-SeedResult $Content
    if ($null -eq $fields) { return @{ ok = $false; reason = 'UNPARSEABLE' } }
    $hashLine = $fields['result_payload_hash']
    if (-not $hashLine -or $hashLine -notmatch '^[0-9a-f]{64}$') {
        return @{ ok = $false; reason = 'PAYLOAD_HASH_MISSING_OR_MALFORMED' }
    }
    $payloadLines = ($Content -split "`n") | ForEach-Object { $_.TrimEnd("`r") } |
        Where-Object { $_.Length -gt 0 -and -not $_.StartsWith('result_payload_hash=') }
    $expectedHash = Get-Sha256Hex ($payloadLines -join "`n")
    if ($expectedHash -ne $hashLine) {
        return @{ ok = $false; reason = 'PAYLOAD_HASH_MISMATCH' }
    }
    if ($fields['terminal'] -ne 'true') { return @{ ok = $false; reason = 'NOT_TERMINAL' } }
    if ([string]$fields['result_schema'] -ne [string]$Identity['result_schema']) {
        return @{ ok = $false; reason = 'SCHEMA_MISMATCH' }
    }
    foreach ($key in @('config_hash', 'git_head', 'apk_sha256', 'qairt_build_id', 'steps')) {
        if ([string]$fields[$key] -ne [string]$Identity[$key]) {
            return @{ ok = $false; reason = "IDENTITY_MISMATCH:$key" }
        }
    }
    if ([string]$fields['seed'] -ne [string]$ExpectedSeed) {
        return @{ ok = $false; reason = 'SEED_MISMATCH' }
    }
    foreach ($required in @('classification', 'device_status', 'step_completed',
            'qnn_nonzero_count', 'nonfinite_count', 'parameter_hash', 'logits_hash')) {
        if (-not $fields.Contains($required)) {
            return @{ ok = $false; reason = "REQUIRED_FIELD_MISSING:$required" }
        }
    }
    return @{ ok = $true; reason = 'OK'; fields = $fields }
}

function Get-CompletedSeeds {
    # Scans a per-configuration seeds directory and returns the seeds whose
    # results are complete, terminal, integrity-valid and identity-matching.
    # Fail-closed on duplicates, unknown files or identity mismatches.
    param(
        [string]$SeedsRoot,
        [int]$SeedCount,
        [System.Collections.IDictionary]$Identity
    )
    $completed = @{}
    $rejections = @()
    for ($seed = 1; $seed -le $SeedCount; $seed++) {
        $dir = Join-Path $SeedsRoot ('seed-{0:d3}' -f $seed)
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        $files = Get-ChildItem -LiteralPath $dir -File
        foreach ($file in $files) {
            if ($file.Name -notmatch '^(result\.txt|result\.tmp|attempt-\d+\.device-report\.txt|attempt-\d+\.partial|isolated-\d+\.(tmp|partial))$') {
                throw "unknown file in seed directory (fail closed): $($file.Name)"
            }
        }
        $final = Join-Path $dir 'result.txt'
        $tmp = Join-Path $dir 'result.tmp'
        if ((Test-Path -LiteralPath $final -PathType Leaf) -and
            (Test-Path -LiteralPath $tmp -PathType Leaf)) {
            throw "duplicate result state for seed $seed (fail closed)"
        }
        $isolated = @($files | Where-Object { $_.Name -match '^isolated-' })
        if ($isolated.Count -gt 0 -and -not (Test-Path -LiteralPath $final -PathType Leaf)) {
            # Prior attempt left only quarantined partials; seed is not done.
            $rejections += "seed $seed has quarantined partial results"
            continue
        }
        if (-not (Test-Path -LiteralPath $final -PathType Leaf)) { continue }
        $content = Get-Content -LiteralPath $final -Raw
        $verdict = Test-SeedResultContent -Content $content -Identity $Identity -ExpectedSeed $seed
        if (-not $verdict.ok) {
            throw "seed $seed result rejected fail-closed: $($verdict.reason)"
        }
        $completed[$seed] = $verdict.fields
    }
    return @{ completed = $completed; rejections = $rejections }
}

function Get-HostInterruptionClassification([string]$Kind) {
    # Host-observed interruptions are transport/process failures, never numeric.
    switch ($Kind) {
        'ADB' { return 'ADB_TRANSPORT_INTERRUPTION' }
        'PROCESS' { return 'ANDROID_PROCESS_TERMINATION' }
        default { throw "unknown host interruption kind: $Kind" }
    }
}

function Resolve-ReattachDecision {
    # Decide whether monitoring can continue after an ADB transport change.
    param(
        [int]$OnlineCount,
        [bool]$PackagePresent,
        [string]$DeviceContextContent,   # empty when unreadable
        [string]$ExpectedRunId,
        [string]$ExpectedConfigHash
    )
    if ($OnlineCount -ne 1) { return 'DEVICE_NOT_READY' }
    if (-not $PackagePresent) { return 'STOP_PACKAGE_MISSING' }
    if (-not $DeviceContextContent) { return 'STOP_CONTEXT_MISSING' }
    $runId = ''
    $configHash = ''
    foreach ($line in ($DeviceContextContent -split "`n")) {
        $pair = $line.TrimEnd("`r") -split '=', 2
        if ($pair.Count -ne 2) { continue }
        if ($pair[0] -eq 'run_id') { $runId = $pair[1] }
        if ($pair[0] -eq 'config_hash') { $configHash = $pair[1] }
    }
    if ($runId -ne $ExpectedRunId) { return 'STOP_RUN_MISMATCH' }
    if ($configHash -ne $ExpectedConfigHash) { return 'STOP_RUN_MISMATCH' }
    return 'CONTINUE'
}

function Get-ReportValue([string]$Report, [string]$Key) {
    $match = [regex]::Match($Report, "(?m)^$([regex]::Escape($Key))=(.*)$")
    if ($match.Success) { return $match.Groups[1].Value.Trim() }
    return $null
}

function Get-DeviceClassification([string]$Report) {
    $status = Get-ReportValue $Report 'status'
    if ($status -eq 'SUCCESS') { return 'SUCCESS' }
    if ($status -notin @('FAILED', 'PARTIAL_SUCCESS')) { return 'INCOMPLETE' }
    $failureClass = Get-ReportValue $Report 'failure_classification'
    switch -Regex ([string]$failureClass) {
        '^QNN_EXECUTE_FINITE_OUTPUT' { return 'NUMERIC_FAILURE' }
        '^QNN_EXECUTE' { return 'QNN_FAILURE' }
        'NONFINITE|FINITE_OUTPUT' { return 'NUMERIC_FAILURE' }
        default { return "UNEXPECTED_DEVICE_FAILURE:$failureClass" }
    }
}

function ConvertFrom-DeviceReport {
    # Extracts the seed-k terminal values from a device report produced by a
    # process whose generic loop covered seeds 1..k.
    param(
        [string]$Report,
        [int]$Seed,
        [int]$StepCount
    )
    $classification = Get-DeviceClassification $Report
    $result = [ordered]@{
        classification = $classification
        device_status = (Get-ReportValue $Report 'status')
        first_bad_seed = 'NONE'; first_bad_step = 'NONE'
        first_bad_phase = 'NONE'; first_bad_tensor = 'NONE'
        qnn_execute_result = 'NONE'
        qnn_nonzero_count = '0'
        app_write_unchanged = (Get-ReportValue $Report 'app_write_hashes_unchanged')
        poison_remaining = (Get-ReportValue $Report 'app_read_poison_residual_elements')
        device_seed_count = (Get-ReportValue $Report 'seed_count')
    }
    $nonzero = Get-ReportValue $Report 'formal_qnn_nonzero_return_count'
    if ($nonzero -match '^\d+$') { $result['qnn_nonzero_count'] = $nonzero }
    if ($classification -eq 'SUCCESS') {
        foreach ($pair in @(
                @('initial_loss', "seed_${Seed}_initial_loss"),
                @('final_loss', "seed_${Seed}_final_loss"),
                @('final_accuracy', "seed_${Seed}_final_accuracy"),
                @('all_steps_finite', "seed_${Seed}_all_steps_finite"),
                @('final_evaluation_finite', "seed_${Seed}_final_evaluation_finite"),
                @('cpu_all_steps_finite', "seed_${Seed}_cpu_all_steps_finite"),
                @('nonfinite_count', "seed_${Seed}_nonfinite_count"),
                @('qnn_execute_count', "seed_${Seed}_qnn_execute_count"),
                @('parameter_hash', "seed_${Seed}_final_parameter_canonical_hash"),
                @('logits_hash', "seed_${Seed}_step_${StepCount}_logits_canonical_hash"),
                @('step_completed', "seed_${Seed}_completed_steps"))) {
            $value = Get-ReportValue $Report $pair[1]
            if ($null -eq $value) {
                throw "device report is missing required seed field: $($pair[1])"
            }
            $result[$pair[0]] = $value
        }
        if ($result['step_completed'] -ne [string]$StepCount) {
            throw "device report seed $Seed completed $($result['step_completed']) steps, expected $StepCount"
        }
        if ($result['device_seed_count'] -ne [string]$Seed) {
            throw "device report seed_count mismatch: expected $Seed"
        }
        $oracleExact = 0
        $freeExact = 0
        foreach ($pattern in 0..3) {
            $oracle = Get-ReportValue $Report "seed_${Seed}_oracle_pattern_${pattern}_exact"
            $free = Get-ReportValue $Report "seed_${Seed}_generation_pattern_${pattern}_exact"
            if ($null -eq $oracle -or $null -eq $free) {
                throw "device report is missing generation pattern $pattern for seed $Seed"
            }
            if ($oracle -eq 'true') { $oracleExact++ }
            if ($free -eq 'true') { $freeExact++ }
        }
        $result['oracle_exact'] = "$oracleExact/4"
        $result['free_exact'] = "$freeExact/4"
    } else {
        $result['first_bad_seed'] = (Get-ReportValue $Report 'first_bad_seed')
        $result['first_bad_step'] = (Get-ReportValue $Report 'first_bad_step')
        $phase = Get-ReportValue $Report 'first_failed_stage'
        $result['first_bad_phase'] = if ($phase) { $phase } else { 'UNKNOWN' }
        $tensor = Get-ReportValue $Report 'first_bad_tensor'
        $result['first_bad_tensor'] = if ($tensor) { $tensor } else { 'NONE' }
        $execute = Get-ReportValue $Report 'qnn_execute_result'
        $result['qnn_execute_result'] = if ($execute) { $execute } else { 'NONE' }
        $result['step_completed'] = 'INTERRUPTED'
        $result['initial_loss'] = 'NOT_REACHED'; $result['final_loss'] = 'NOT_REACHED'
        $result['final_accuracy'] = 'NOT_REACHED'; $result['all_steps_finite'] = 'unknown'
        $result['final_evaluation_finite'] = 'unknown'
        $result['cpu_all_steps_finite'] = 'unknown'; $result['nonfinite_count'] = 'NOT_EVALUATED'
        $result['qnn_execute_count'] = 'NOT_EVALUATED'
        $result['parameter_hash'] = 'NOT_REACHED'; $result['logits_hash'] = 'NOT_REACHED'
        $result['oracle_exact'] = 'NOT_REACHED'; $result['free_exact'] = 'NOT_REACHED'
    }
    return $result
}

# ---------------------------------------------------------------------------
# Self test (no device, no QAIRT, temp directory only)
# ---------------------------------------------------------------------------

function Assert-SelfTest([bool]$Condition, [string]$Name) {
    if (-not $Condition) { throw "self-test failed: $Name" }
    Write-Host "self-test: $Name ok"
}

function Invoke-SelfTest {
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('phonelm-resumable-formal-selftest-' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($tempRoot) | Out-Null
    try {
        $identity = @{
            config_hash = Get-CanonicalConfigHash -T 16 -V 32 -D 256 -Ffn 372 -L 3 -H 4 -Lr '0.003' -StepCount 320
            git_head = '0' * 40
            apk_sha256 = 'a' * 64
            qairt_build_id = 'selftest-build'
            steps = '320'
            result_schema = [string]$ResultSchemaVersion
        }
        $seedDir = Join-Path $tempRoot 'seeds'

        function New-GoodFields([int]$Seed, [hashtable]$Id) {
            [ordered]@{
                result_schema = [string]$ResultSchemaVersion
                terminal = 'true'; classification = 'SUCCESS'; device_status = 'SUCCESS'
                seed = [string]$Seed; steps = $Id['steps']; attempt = '1'
                device_seed_count = [string]$Seed; test_mode = 'UI_VALIDATION'
                step_completed = $Id['steps']
                initial_loss = '9.1'; final_loss = '0.4'; final_accuracy = '0.9843'
                all_steps_finite = 'true'; final_evaluation_finite = 'true'
                cpu_all_steps_finite = 'true'
                qnn_nonzero_count = '0'; nonfinite_count = '0'; qnn_execute_count = '3300'
                parameter_hash = ('{0:x64}' -f $Seed) -replace ' ', '0'
                logits_hash = ('{0:x64}' -f ($Seed + 16)) -replace ' ', '0'
                oracle_exact = '4/4'; free_exact = '4/4'
                app_write_unchanged = 'true'; poison_remaining = '0'
                first_bad_seed = 'NONE'; first_bad_step = 'NONE'; first_bad_phase = 'NONE'
                first_bad_tensor = 'NONE'; qnn_execute_result = '0'
                battery_temperature_before_c = '35.0'; battery_temperature_after_c = '40.0'
                thermal_status_before = '0'; thermal_status_after = '0'
                foreground_service_isForeground = 'true'; foreground_service_type_dataSync = 'true'
                fg_service_flag_present = 'true'; cached_empty_seen = 'false'
                progress_updates_seen = 'true'; ongoing_during_run = 'true'
                ongoing_after_completion = 'false'
                wall_time_ms = '1234'
                started_utc = '2026-08-01T00:00:00.0000000Z'
                completed_utc = '2026-08-01T00:10:00.0000000Z'
                run_id = 'formal-selftest-run'; configuration_id = 'selftest-config'
                config_hash = $Id['config_hash']; git_head = $Id['git_head']
                apk_sha256 = $Id['apk_sha256']; qairt_build_id = $Id['qairt_build_id']
                result_schema_id = 'phonelm-formal-seed-result'
            }
        }
        function Write-SeedResult([int]$Seed, [string]$Content, [string]$Name = 'result.txt') {
            $dir = Join-Path $seedDir ('seed-{0:d3}' -f $Seed)
            [IO.Directory]::CreateDirectory($dir) | Out-Null
            [IO.File]::WriteAllText((Join-Path $dir $Name), $Content)
        }

        # 1: all seeds unexecuted -> plan covers 1..5
        $state = Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity
        Assert-SelfTest ($state.completed.Count -eq 0) 'fresh configuration runs seeds 1..5'

        # atomic promotion helper mirrors the production write path
        function Promote-SeedResult([int]$Seed, [System.Collections.IDictionary]$Fields) {
            $dir = Join-Path $seedDir ('seed-{0:d3}' -f $Seed)
            [IO.Directory]::CreateDirectory($dir) | Out-Null
            $tmp = Join-Path $dir 'result.tmp'
            [IO.File]::WriteAllText($tmp, (Protect-SeedResult $Fields))
            $content = [IO.File]::ReadAllText($tmp)
            $verdict = Test-SeedResultContent -Content $content -Identity $identity -ExpectedSeed $Seed
            if (-not $verdict.ok) {
                Move-Item -LiteralPath $tmp -Destination (Join-Path $dir 'isolated-1.tmp') -Force
                throw "promotion refused: $($verdict.reason)"
            }
            Move-Item -LiteralPath $tmp -Destination (Join-Path $dir 'result.txt')
        }

        # 2: seeds 1,2 complete -> resume plan is 3..5
        Promote-SeedResult -Seed 1 -Fields (New-GoodFields 1 $identity)
        Promote-SeedResult -Seed 2 -Fields (New-GoodFields 2 $identity)
        $state = Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity
        Assert-SelfTest ($state.completed.Count -eq 2 -and $state.completed.Contains(1) -and $state.completed.Contains(2)) 'resume reuses completed seeds 1,2 only'

        # 3: seed 3 partial (.tmp only) is not reused
        Write-SeedResult -Seed 3 -Content (Protect-SeedResult (New-GoodFields 3 $identity)) -Name 'result.tmp'
        $state = Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity
        Assert-SelfTest (-not $state.completed.Contains(3)) 'seed 3 partial tmp is not treated as complete'
        Remove-Item -LiteralPath (Join-Path (Join-Path $seedDir 'seed-003') 'result.tmp') -Force

        # 3b: terminal=false is rejected fail-closed
        $notTerminal = New-GoodFields 3 $identity; $notTerminal['terminal'] = 'false'
        Write-SeedResult -Seed 3 -Content (Protect-SeedResult $notTerminal)
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'NOT_TERMINAL' }
        Assert-SelfTest $threw 'terminal=false result rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-003') -Recurse -Force

        # 4: config hash mismatch rejected
        $badConfig = New-GoodFields 3 $identity; $badConfig['config_hash'] = 'f' * 64
        Write-SeedResult -Seed 3 -Content (Protect-SeedResult $badConfig)
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'IDENTITY_MISMATCH:config_hash' }
        Assert-SelfTest $threw 'config hash mismatch rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-003') -Recurse -Force

        # 5: APK hash mismatch rejected
        $badApk = New-GoodFields 3 $identity; $badApk['apk_sha256'] = 'b' * 64
        Write-SeedResult -Seed 3 -Content (Protect-SeedResult $badApk)
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'IDENTITY_MISMATCH:apk_sha256' }
        Assert-SelfTest $threw 'APK hash mismatch rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-003') -Recurse -Force

        # 6: result schema mismatch rejected
        $badSchema = New-GoodFields 3 $identity; $badSchema['result_schema'] = '999'
        Write-SeedResult -Seed 3 -Content (Protect-SeedResult $badSchema)
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'SCHEMA_MISMATCH' }
        Assert-SelfTest $threw 'result schema mismatch rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-003') -Recurse -Force

        # 7: duplicate final state (result.txt + stray result.tmp) rejected
        Promote-SeedResult -Seed 3 -Fields (New-GoodFields 3 $identity)
        Write-SeedResult -Seed 3 -Content 'seed=3' -Name 'result.tmp'
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'duplicate result state' }
        Assert-SelfTest $threw 'duplicate seed result rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-003') -Recurse -Force

        # 7b: unknown file in seed directory rejected fail-closed
        Write-SeedResult -Seed 4 -Content 'garbage' -Name 'unexpected.bin'
        $threw = $false
        try { Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity | Out-Null }
        catch { $threw = $_.Exception.Message -match 'unknown file in seed directory' }
        Assert-SelfTest $threw 'unknown seed file rejected'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-004') -Recurse -Force

        # 8: corrupted payload (hash mismatch) rejected by integrity test
        $corrupt = (Protect-SeedResult (New-GoodFields 3 $identity)) -replace 'final_loss=0.4', 'final_loss=0.5'
        $verdict = Test-SeedResultContent -Content $corrupt -Identity $identity -ExpectedSeed 3
        Assert-SelfTest ((-not $verdict.ok) -and $verdict.reason -eq 'PAYLOAD_HASH_MISMATCH') 'corrupted payload rejected'

        # 9: atomic rename promotes valid tmp, refuses invalid tmp
        Promote-SeedResult -Seed 3 -Fields (New-GoodFields 3 $identity)
        $dir3 = Join-Path $seedDir 'seed-003'
        Assert-SelfTest ((Test-Path -LiteralPath (Join-Path $dir3 'result.txt')) -and
            -not (Test-Path -LiteralPath (Join-Path $dir3 'result.tmp'))) 'atomic rename promotes valid result'
        $invalid = New-GoodFields 5 $identity; $invalid['result_schema'] = '999'
        $tmp5 = Join-Path (Join-Path $seedDir 'seed-005') 'result.tmp'
        [IO.Directory]::CreateDirectory((Join-Path $seedDir 'seed-005')) | Out-Null
        [IO.File]::WriteAllText($tmp5, (Protect-SeedResult $invalid))
        $verdict = Test-SeedResultContent -Content ([IO.File]::ReadAllText($tmp5)) -Identity $identity -ExpectedSeed 5
        if (-not $verdict.ok) { Move-Item -LiteralPath $tmp5 -Destination (Join-Path (Join-Path $seedDir 'seed-005') 'isolated-1.tmp') }
        Assert-SelfTest ((Test-Path -LiteralPath (Join-Path (Join-Path $seedDir 'seed-005') 'isolated-1.tmp')) -and
            -not (Test-Path -LiteralPath (Join-Path (Join-Path $seedDir 'seed-005') 'result.txt'))) 'invalid tmp quarantined, not promoted'
        $state = Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity
        Assert-SelfTest (-not $state.completed.Contains(5)) 'quarantined partial seed is not reused'
        Remove-Item -LiteralPath (Join-Path $seedDir 'seed-005') -Recurse -Force

        # 10: all five seeds aggregate
        foreach ($s in 4..5) { Promote-SeedResult -Seed $s -Fields (New-GoodFields $s $identity) }
        $state = Get-CompletedSeeds -SeedsRoot $seedDir -SeedCount 5 -Identity $identity
        Assert-SelfTest ($state.completed.Count -eq 5) 'all five seeds complete'
        $finite = 0; $oracleExact = 0; $freeExact = 0; $nonzero = 0; $nonfinite = 0
        foreach ($seed in 1..5) {
            $fields = $state.completed[$seed]
            if ($fields['all_steps_finite'] -eq 'true' -and $fields['final_evaluation_finite'] -eq 'true') { $finite++ }
            $oracleExact += [int](($fields['oracle_exact'] -split '/')[0])
            $freeExact += [int](($fields['free_exact'] -split '/')[0])
            $nonzero += [int]$fields['qnn_nonzero_count']
            $nonfinite += [int]$fields['nonfinite_count']
        }
        Assert-SelfTest ($finite -eq 5 -and $oracleExact -eq 20 -and $freeExact -eq 20 -and $nonzero -eq 0 -and $nonfinite -eq 0) 'aggregate finite=5 oracle=20 free=20 qnn=0 nonfinite=0'

        # 11: ADB interruption classification (never a numeric failure)
        Assert-SelfTest ((Get-HostInterruptionClassification 'ADB') -eq 'ADB_TRANSPORT_INTERRUPTION') 'ADB interruption classification'
        Assert-SelfTest ((Get-HostInterruptionClassification 'PROCESS') -eq 'ANDROID_PROCESS_TERMINATION') 'process termination classification'

        # 12: same-run reattach continues
        $context = "run_id=formal-selftest-run`nconfig_hash=$($identity.config_hash)`n"
        $decision = Resolve-ReattachDecision -OnlineCount 1 -PackagePresent $true `
            -DeviceContextContent $context -ExpectedRunId 'formal-selftest-run' `
            -ExpectedConfigHash $identity.config_hash
        Assert-SelfTest ($decision -eq 'CONTINUE') 'same-run reattach continues'

        # 13: different run id rejected
        $decision = Resolve-ReattachDecision -OnlineCount 1 -PackagePresent $true `
            -DeviceContextContent "run_id=other-run`nconfig_hash=$($identity.config_hash)`n" `
            -ExpectedRunId 'formal-selftest-run' -ExpectedConfigHash $identity.config_hash
        Assert-SelfTest ($decision -eq 'STOP_RUN_MISMATCH') 'different run id stops formal'
        $decision = Resolve-ReattachDecision -OnlineCount 1 -PackagePresent $true `
            -DeviceContextContent "run_id=formal-selftest-run`nconfig_hash=$('e' * 64)`n" `
            -ExpectedRunId 'formal-selftest-run' -ExpectedConfigHash $identity.config_hash
        Assert-SelfTest ($decision -eq 'STOP_RUN_MISMATCH') 'different config hash stops formal'
        $decision = Resolve-ReattachDecision -OnlineCount 0 -PackagePresent $true `
            -DeviceContextContent $context -ExpectedRunId 'formal-selftest-run' `
            -ExpectedConfigHash $identity.config_hash
        Assert-SelfTest ($decision -eq 'DEVICE_NOT_READY') 'offline device not ready'
        $decision = Resolve-ReattachDecision -OnlineCount 1 -PackagePresent $false `
            -DeviceContextContent $context -ExpectedRunId 'formal-selftest-run' `
            -ExpectedConfigHash $identity.config_hash
        Assert-SelfTest ($decision -eq 'STOP_PACKAGE_MISSING') 'missing package stops formal'

        # device-report classification mapping
        Assert-SelfTest ((Get-DeviceClassification "TINY_LANGUAGE_MODEL`nstatus=SUCCESS`n") -eq 'SUCCESS') 'device SUCCESS classification'
        Assert-SelfTest ((Get-DeviceClassification "TINY_LANGUAGE_MODEL`nstatus=FAILED`nfailure_classification=QNN_EXECUTE`n") -eq 'QNN_FAILURE') 'device QNN failure classification'
        Assert-SelfTest ((Get-DeviceClassification "TINY_LANGUAGE_MODEL`nstatus=FAILED`nfailure_classification=QNN_EXECUTE_FINITE_OUTPUT`n") -eq 'NUMERIC_FAILURE') 'device numeric failure classification'

        'SELF_TEST=PASS'
    } finally {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

function Resolve-OnlineDevice([string]$Adb) {
    $online = @((& $Adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } |
        ForEach-Object { $Matches[1] })
    if ($online.Count -ne 1) {
        throw "expected exactly one online ADB device; found $($online.Count)"
    }
    return $online[0]
}

function Assert-PhysicalDevice([string]$Adb, [string]$Device) {
    $properties = @(
        & $Adb -s $Device shell getprop ro.kernel.qemu
        & $Adb -s $Device shell getprop ro.hardware
    ) -join "`n"
    if ($properties -match '(?im)^(1|.*(goldfish|ranchu).*)$') {
        throw 'emulator device rejected; a physical device is required'
    }
}

function Read-DeviceState([string]$Adb, [string]$Device) {
    $battery = (& $Adb -s $Device shell dumpsys battery) -join "`n"
    $thermal = (& $Adb -s $Device shell dumpsys thermalservice) -join "`n"
    $level = [regex]::Match($battery, '(?m)^\s*level:\s*(\d+)').Groups[1].Value
    $temperature = [regex]::Match($battery, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
    $status = [regex]::Match($thermal, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
    if (-not $level -or -not $temperature -or -not $status) {
        throw 'could not read battery or thermal state'
    }
    return [pscustomobject][ordered]@{
        battery_level = [int]$level
        battery_temperature_c = [int]$temperature / 10.0
        android_thermal_status = [int]$status
    }
}

function Assert-Cool([object]$State, [string]$Phase) {
    if ($State.battery_temperature_c -ge 45.0) {
        throw "THERMAL_ABORT_RESUMABLE: $Phase battery temperature is $($State.battery_temperature_c) C"
    }
    if ($State.android_thermal_status -ge 3) {
        throw "THERMAL_ABORT_RESUMABLE: $Phase Android thermal status is $($State.android_thermal_status)"
    }
}

# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------

if ($SelfTest) {
    Invoke-SelfTest
    return
}

if (-not $QairtSdkRoot) { throw '-QairtSdkRoot is required' }
if (-not $ConfigurationId) { throw '-ConfigurationId is required' }

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'adb executable unavailable' }
$package = 'com.yuubinnkyoku.phonelm'
$activity = "$package/.MainActivity"
$runIdEffective = if ($RunId) { $RunId } else {
    'formal-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + (-join ((48..57) + (97..102) | Get-Random -Count 6 | ForEach-Object { [char]$_ }))
}
$configHash = Get-CanonicalConfigHash -T $SequenceLength -V $VocabularySize `
    -D $EmbeddingDimension -Ffn $FeedForwardDimension -L $NumLayers -H $NumHeads `
    -Lr $LearningRate -StepCount $Steps
$gitHead = (& git -C $repoRoot rev-parse HEAD).Trim()
$configRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "build\reports\qnn-resumable-formal\$ConfigurationId"))
$reportsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build\reports'))
if (-not $configRoot.StartsWith($reportsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'report path escaped build/reports'
}
$seedsRoot = Join-Path $configRoot 'seeds'

Push-Location $repoRoot
try {
    # QAIRT gate: explicit root + build id, no fallback.
    $checkOutput = (& (Join-Path $PSScriptRoot 'check_qairt.ps1') -SdkRoot $QairtSdkRoot `
            -ExpectedBuildId $ExpectedBuildId 2>&1) -join "`n"
    $checkExit = $LASTEXITCODE
    if ($checkExit -notin @(0, 3)) { throw "fixed QAIRT inventory check failed with exit $checkExit" }
    if ($checkOutput -notmatch '(?m)^expected_build_id_match=true$') {
        throw 'fixed QAIRT build id was not confirmed'
    }

    $apk = Join-Path $repoRoot 'app\build\outputs\apk\debug\app-debug.apk'
    $apkHash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.Directory]::CreateDirectory($configRoot) | Out-Null
    $auditPath = Join-Path $configRoot 'apk-audit.txt'
    $cachedAudit = if (Test-Path -LiteralPath $auditPath -PathType Leaf) {
        Get-Content -LiteralPath $auditPath -Raw
    } else { '' }
    $auditMatches = $cachedAudit -match "(?m)^qairt_build_id=$([regex]::Escape($ExpectedBuildId))\r?$" -and
        $cachedAudit -match "(?m)^apk_sha256=$([regex]::Escape($apkHash))\r?$" -and
        $cachedAudit -match '(?m)^forbidden_2_47_strings=false\r?$' -and
        $cachedAudit -match '(?m)^host_sdk_path_present=false\r?$' -and
        $cachedAudit -match '(?m)^status=SUCCESS\r?$'
    if (-not $auditMatches) {
        & (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk `
            -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
            -ReportPath $auditPath | Out-Null
        if (-not $?) { throw 'fixed QAIRT APK audit failed' }
    }

    $identity = @{
        config_hash = $configHash
        git_head = $gitHead
        apk_sha256 = $apkHash
        qairt_build_id = $ExpectedBuildId
        steps = [string]$Steps
        result_schema = [string]$ResultSchemaVersion
    }

    # Run context: a new or resumed invocation. Existing results must match
    # identity; a mismatched history is never silently superseded.
    $contextPath = Join-Path $configRoot 'run-context.txt'
    if (Test-Path -LiteralPath $contextPath -PathType Leaf) {
        $existing = Read-SeedResult (Get-Content -LiteralPath $contextPath -Raw)
        foreach ($key in @('config_hash', 'git_head', 'apk_sha256', 'qairt_build_id', 'steps')) {
            if ([string]$existing[$key] -ne [string]$identity[$key]) {
                throw "existing run history does not match current identity ($key); refusing to resume"
            }
        }
    }
    @(
        'schema=1'
        "configuration_id=$ConfigurationId"
        "latest_run_id=$runIdEffective"
        "config_hash=$configHash"
        "git_head=$gitHead"
        "apk_sha256=$apkHash"
        "qairt_build_id=$ExpectedBuildId"
        "steps=$Steps"
        "result_schema=$ResultSchemaVersion"
    ) | Set-Content -LiteralPath $contextPath -Encoding utf8

    $device = Resolve-OnlineDevice $adb
    Assert-PhysicalDevice $adb $device
    $packageList = (& $adb -s $device shell pm list packages $package) -join "`n"
    if ($packageList -notmatch [regex]::Escape($package)) {
        if (-not $InstallAuditedApk) { throw 'package not installed; rerun with -InstallAuditedApk' }
    }
    if ($InstallAuditedApk) { & $adb -s $device install -r $apk | Out-Null }

    $attemptsLog = Join-Path $configRoot 'attempts.jsonl'
    function Write-AttemptLog([int]$Seed, [int]$Attempt, [string]$Classification, [string]$Detail) {
        [pscustomobject][ordered]@{
            utc = [DateTime]::UtcNow.ToString('o')
            run_id = $runIdEffective
            seed = $Seed
            attempt = $Attempt
            classification = $Classification
            detail = $Detail
        } | ConvertTo-Json -Compress | Add-Content -LiteralPath $attemptsLog -Encoding utf8
    }

    $state = Get-CompletedSeeds -SeedsRoot $seedsRoot -SeedCount $Seeds -Identity $identity
    foreach ($rejection in $state.rejections) { Write-Host "resume note: $rejection" }
    if ($state.completed.Count -gt 0) {
        foreach ($seed in ($state.completed.Keys | Sort-Object)) {
            Write-AttemptLog -Seed $seed -Attempt 0 -Classification 'RESUMED' -Detail 'completed result reused'
        }
    }
    $pending = @(1..$Seeds | Where-Object { -not $state.completed.Contains($_) })
    Write-Host "identity: config_hash=$configHash"
    Write-Host "completed seeds: $(@($state.completed.Keys | Sort-Object) -join ',') pending: $($pending -join ',')"

    $aggregateStatus = 'INCOMPLETE'
    $stopDetail = ''
    :seeds foreach ($seed in $pending) {
        $done = $false
        for ($attempt = 1; $attempt -le $MaxAttemptsPerSeed -and -not $done; $attempt++) {
            $device = Resolve-OnlineDevice $adb
            $before = Read-DeviceState $adb $device
            $coolDeadline = [DateTime]::UtcNow.AddMinutes($CoolDownGraceMinutes)
            while ($before.battery_temperature_c -ge $ResumeBatteryCeilingC -and
                $before.battery_temperature_c -lt 45.0) {
                if ([DateTime]::UtcNow -gt $coolDeadline) {
                    Write-AttemptLog -Seed $seed -Attempt $attempt -Classification 'THERMAL_ABORT_RESUMABLE' `
                        -Detail "cooldown grace exceeded at $($before.battery_temperature_c) C"
                    $aggregateStatus = 'BLOCKED_RESUMABLE'
                    $stopDetail = "seed $seed attempt $attempt cooldown grace exceeded"
                    break seeds
                }
                Write-Host "waiting for cooldown: battery $($before.battery_temperature_c) C (ceiling $ResumeBatteryCeilingC)"
                Start-Sleep -Seconds 120
                $device = Resolve-OnlineDevice $adb
                $before = Read-DeviceState $adb $device
            }
            try {
                Assert-Cool $before "seed $seed attempt $attempt"
            } catch {
                Write-AttemptLog -Seed $seed -Attempt $attempt -Classification 'THERMAL_ABORT_RESUMABLE' `
                    -Detail $_.Exception.Message
                $aggregateStatus = 'BLOCKED_RESUMABLE'
                $stopDetail = $_.Exception.Message
                break seeds
            }
            $beforeState = $before

            # Quarantine any leftover partial state for this seed (never overwrite).
            $seedDir = Join-Path $seedsRoot ('seed-{0:d3}' -f $seed)
            [IO.Directory]::CreateDirectory($seedDir) | Out-Null
            foreach ($leftover in @('result.tmp', 'result.txt')) {
                $leftoverPath = Join-Path $seedDir $leftover
                if (Test-Path -LiteralPath $leftoverPath -PathType Leaf) {
                    Move-Item -LiteralPath $leftoverPath `
                        -Destination (Join-Path $seedDir ("isolated-$attempt.$($leftover -replace '^result\.', '')")) -Force
                }
            }

            # Prepare device-side run context and clean stale results.
            # Note: the Windows adb client strips inner quoting, so `sh -c`
            # compositions cannot reach the device intact. Use plain argument
            # lists and a stdin pipe through toybox tee instead.
            & $adb -s $device shell run-as $package mkdir files 2>$null | Out-Null
            $contextDevice = @(
                'schema=1'
                "run_id=$runIdEffective"
                "config_hash=$configHash"
                "apk_sha256=$apkHash"
                "qairt_build_id=$ExpectedBuildId"
                "seed=$seed"
                "attempt=$attempt"
                "steps=$Steps"
            ) -join "`n"
            $contextDevice | & $adb -s $device shell run-as $package tee `
                files/phonelm-formal-run-context.txt | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'could not write device run context' }
            & $adb -s $device shell am force-stop $package | Out-Null
            & $adb -s $device shell run-as $package rm -f files/device-test-result.txt | Out-Null

            $arguments = @(
                'shell', 'am', 'start', '-W', '-n', $activity,
                '--es', 'phonelm.mode', 'QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC',
                '--ei', 'phonelm.batch_size', '1',
                '--ei', 'phonelm.dimension', [string]$EmbeddingDimension,
                '--ei', 'phonelm.hidden_dimension', [string]$FeedForwardDimension,
                '--ei', 'phonelm.output_dimension', [string]$VocabularySize,
                '--ei', 'phonelm.steps', [string]$Steps,
                '--ei', 'phonelm.warmup_steps', '0',
                '--es', 'phonelm.learning_rate', $LearningRate,
                '--es', 'phonelm.seed', '1',
                '--ei', 'phonelm.sample_count', [string]$SequenceLength,
                '--ei', 'phonelm.epochs', [string]$NumLayers,
                '--ei', 'phonelm.measured_steps', [string]$NumHeads,
                '--ei', 'phonelm.correctness_interval', [string]$seed,
                '--ez', 'phonelm.benchmark_mode', 'false'
            )
            if ($TestMode -eq 'UI_VALIDATION') {
                $arguments += @('--ez', 'phonelm.live_update', 'true')
            }
            $startedUtc = [DateTime]::UtcNow
            $launchOutput = & $adb -s $device @arguments 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "activity launch failed (endpoint redacted): $launchOutput"
            }

            $ongoingDuring = $false; $progressSeen = $false
            $isForeground = $false; $dataSync = $false; $fgFlag = $false; $cachedEmpty = $false
            $previousNotification = ''
            $pollCount = 0
            $report = ''
            $knownSerial = $device
            $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSecondsPerAttempt)
            $disconnected = $false
            $classification = $null
            while ([DateTime]::UtcNow -lt $deadline) {
                Start-Sleep -Seconds 10
                $pollCount++
                try {
                    $resolved = Resolve-OnlineDevice $adb
                    if ($resolved -ne $knownSerial) {
                        # The transport or endpoint changed even though no ADB
                        # command failed. Re-validate identity before continuing.
                        $device = $resolved
                        $disconnected = $true
                        Write-Host "seed $seed attempt ${attempt}: ADB endpoint changed; re-validating run identity"
                        break
                    }
                    $device = $resolved
                } catch {
                    # ADB transport interruption: wait for the physical device to
                    # return, then re-validate the app-private run context.
                    $disconnected = $true
                    Write-Host "seed $seed attempt ${attempt}: ADB transport interrupted; waiting for reattach"
                    break
                }
                $candidate = (& $adb -s $device shell run-as $package `
                        cat files/device-test-result.txt 2>$null) -join "`n"
                if ($candidate -and $candidate -notmatch '^cat:' -and
                    $candidate -match '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)$') {
                    $report = $candidate
                    break
                }
                # Heavy dumpsys/notification queries are sampled: densely at
                # startup (foreground-service acceptance evidence) and then at
                # a low rate, so monitoring does not itself add thermal load.
                $uiSamplePoll = $TestMode -eq 'UI_VALIDATION' -and (
                    $pollCount -le 8 -or ($pollCount % 18) -eq 0)
                if ($uiSamplePoll) {
                    $notification = (& $adb -s $device shell cmd notification list 2>$null) -join "`n"
                    if ($notification -match 'com\.yuubinnkyoku\.phonelm') {
                        $ongoingDuring = $true
                        $keyLine = ($notification -split "`n" | Where-Object {
                                $_ -match 'com\.yuubinnkyoku\.phonelm' } | Select-Object -First 1)
                        $detail = (& $adb -s $device shell cmd notification get "'$($keyLine.Trim())'" 2>$null) -join "`n"
                        if ($detail -and $detail -ne $previousNotification) { $progressSeen = $true }
                        $previousNotification = $detail
                    }
                    $services = (& $adb -s $device shell dumpsys activity services $package 2>$null) -join "`n"
                    if ($services -match '(?m)^\s*isForeground=true') { $isForeground = $true }
                    if ($services -match 'foregroundServiceType=0x0*1(?!0)') { $dataSync = $true }
                    $processes = (& $adb -s $device shell dumpsys activity processes $package 2>$null) -join "`n"
                    $processSection = [regex]::Match($processes,
                        "(?s)ProcessRecord\{[0-9a-f]+ +\d+:$([regex]::Escape($package))/[^}]*?}").Value
                    if ($processSection -match 'mHasForegroundServices=true') { $fgFlag = $true }
                    if ($processSection -match 'cached.*empty' -or $services -match 'cached.*empty') {
                        $cachedEmpty = $true
                    }
                }
                $pidValue = ((& $adb -s $device shell pidof $package 2>$null))
                $pidValue = ($pidValue | Where-Object { $_ -match '\S' }) -join "`n"
                $pidValue = $pidValue.Trim()
                if (-not $pidValue) {
                    $classification = Get-HostInterruptionClassification 'PROCESS'
                    break
                }
                if (($pollCount % 12) -eq 0) {
                    try {
                        Assert-Cool (Read-DeviceState $adb $device) "seed $seed attempt $attempt mid-run"
                    } catch {
                        $classification = 'THERMAL_ABORT_RESUMABLE'
                        break
                    }
                }
            }

            if ($disconnected) {
                # Wait for reattachment, re-resolve and re-validate identity.
                $reattachDeadline = [DateTime]::UtcNow.AddSeconds($ReconnectGraceSeconds)
                $decision = 'DEVICE_NOT_READY'
                while ([DateTime]::UtcNow -lt $reattachDeadline) {
                    Start-Sleep -Seconds 10
                    $online = @((& $adb devices 2>$null) | Where-Object { $_ -match '^(\S+)\s+device$' } |
                        ForEach-Object { ($_ -split '\s+')[0] })
                    $packagePresent = (& $adb shell pm list packages $package 2>$null) -match [regex]::Escape($package)
                    $deviceContext = (& $adb shell run-as $package cat files/phonelm-formal-run-context.txt 2>$null) -join "`n"
                    $decision = Resolve-ReattachDecision -OnlineCount $online.Count `
                        -PackagePresent ([bool]$packagePresent) -DeviceContextContent $deviceContext `
                        -ExpectedRunId $runIdEffective -ExpectedConfigHash $configHash
                    if ($decision -eq 'CONTINUE') {
                        Assert-PhysicalDevice $adb $online[0]
                        $device = $online[0]
                        $knownSerial = $device
                        Write-Host 'reattached to same run; continuing seed monitoring'
                        break
                    }
                    if ($decision -ne 'DEVICE_NOT_READY') { break }
                }
                if ($decision -eq 'CONTINUE') {
                    continue  # resume the polling loop for the same attempt
                }
                Write-AttemptLog -Seed $seed -Attempt $attempt -Classification 'ADB_TRANSPORT_INTERRUPTION' `
                    -Detail "reattach=$decision"
                if ($decision -match '^STOP') {
                    # A different run is occupying the device, or the package
                    # vanished; never overwrite or supersede that state.
                    $aggregateStatus = 'BLOCKED_DIFFERENT_RUN'
                    $stopDetail = "seed $seed attempt $attempt reattach refused: $decision"
                    break seeds
                }
                continue  # next attempt of the same seed
            }

            if (-not $report) {
                if (-not $classification) { $classification = Get-HostInterruptionClassification 'PROCESS' }
                if ($classification -eq 'THERMAL_ABORT_RESUMABLE') {
                    Write-AttemptLog -Seed $seed -Attempt $attempt -Classification $classification -Detail 'thermal limit reached mid-run'
                    $aggregateStatus = 'BLOCKED_RESUMABLE'
                    $stopDetail = "seed $seed attempt $attempt thermal pause"
                    break seeds
                }
                Write-AttemptLog -Seed $seed -Attempt $attempt -Classification $classification `
                    -Detail "no terminal device result within $TimeoutSecondsPerAttempt s"
                continue  # next attempt of the same seed
            }

            # Completion observed: pull immediately, validate, promote atomically.
            $rawPath = Join-Path $seedDir ("attempt-{0}.device-report.txt" -f $attempt)
            [IO.File]::WriteAllText($rawPath, $report)
            $completedUtc = [DateTime]::UtcNow
            $after = $null
            try { $after = Read-DeviceState $adb $device } catch { }
            $ongoingAfter = $true
            if ($TestMode -eq 'UI_VALIDATION') {
                foreach ($probe in 1..10) {
                    Start-Sleep -Seconds 2
                    $services = (& $adb -s $device shell dumpsys activity services $package) -join "`n"
                    if ($services -notmatch '(?m)^\s*isForeground=true') { $ongoingAfter = $false; break }
                }
            } else {
                $ongoingAfter = $false
            }

            $fields = ConvertFrom-DeviceReport -Report $report -Seed $seed -StepCount $Steps
            $classification = $fields.classification
            $fields['result_schema'] = [string]$ResultSchemaVersion
            $fields['terminal'] = 'true'
            $fields['seed'] = [string]$seed
            $fields['steps'] = [string]$Steps
            $fields['attempt'] = [string]$attempt
            $fields['test_mode'] = $TestMode
            $fields['device_seed_count'] = [string]($fields['device_seed_count'])
            $fields['battery_temperature_before_c'] = [string]$beforeState.battery_temperature_c
            $fields['battery_temperature_after_c'] = if ($after) { [string]$after.battery_temperature_c } else { 'NOT_READ' }
            $fields['thermal_status_before'] = [string]$beforeState.android_thermal_status
            $fields['thermal_status_after'] = if ($after) { [string]$after.android_thermal_status } else { 'NOT_READ' }
            $fields['foreground_service_isForeground'] = $isForeground.ToString().ToLowerInvariant()
            $fields['foreground_service_type_dataSync'] = $dataSync.ToString().ToLowerInvariant()
            $fields['fg_service_flag_present'] = $fgFlag.ToString().ToLowerInvariant()
            $fields['cached_empty_seen'] = $cachedEmpty.ToString().ToLowerInvariant()
            $fields['progress_updates_seen'] = $progressSeen.ToString().ToLowerInvariant()
            $fields['ongoing_during_run'] = $ongoingDuring.ToString().ToLowerInvariant()
            $fields['ongoing_after_completion'] = $ongoingAfter.ToString().ToLowerInvariant()
            $fields['wall_time_ms'] = [string][math]::Round(($completedUtc - $startedUtc).TotalMilliseconds, 1)
            $fields['started_utc'] = $startedUtc.ToString('o')
            $fields['completed_utc'] = $completedUtc.ToString('o')
            $fields['run_id'] = $runIdEffective
            $fields['configuration_id'] = $ConfigurationId
            $fields['config_hash'] = $configHash
            $fields['git_head'] = $gitHead
            $fields['apk_sha256'] = $apkHash
            $fields['qairt_build_id'] = $ExpectedBuildId
            $fields['result_schema_id'] = 'phonelm-formal-seed-result'

            if ($TestMode -eq 'UI_VALIDATION' -and $classification -eq 'SUCCESS') {
                if (-not $isForeground -or -not $dataSync -or -not $fgFlag -or $cachedEmpty -or
                        -not $ongoingDuring -or $ongoingAfter) {
                    # Foreground service contract violation: stop and report;
                    # the foreground service implementation is unchanged here.
                    Write-AttemptLog -Seed $seed -Attempt $attempt -Classification 'FOREGROUND_SERVICE_CHECK_FAILED' `
                        -Detail "isForeground=$isForeground dataSync=$dataSync flag=$fgFlag cachedEmpty=$cachedEmpty ongoingDuring=$ongoingDuring ongoingAfter=$ongoingAfter"
                    $aggregateStatus = 'BLOCKED_FOREGROUND_SERVICE'
                    $stopDetail = 'foreground service contract check failed'
                    break seeds
                }
            }

            $content = Protect-SeedResult $fields
            $tmpPath = Join-Path $seedDir 'result.tmp'
            [IO.File]::WriteAllText($tmpPath, $content)
            $verdict = Test-SeedResultContent -Content ([IO.File]::ReadAllText($tmpPath)) `
                -Identity $identity -ExpectedSeed $seed
            if (-not $verdict.ok) {
                Move-Item -LiteralPath $tmpPath -Destination (Join-Path $seedDir "isolated-$attempt.tmp") -Force
                Write-AttemptLog -Seed $seed -Attempt $attempt -Classification 'ANDROID_PROCESS_TERMINATION' `
                    -Detail "host result integrity check failed: $($verdict.reason)"
                continue
            }
            Move-Item -LiteralPath $tmpPath -Destination (Join-Path $seedDir 'result.txt')
            Write-AttemptLog -Seed $seed -Attempt $attempt -Classification $classification -Detail 'terminal result stored'
            if ($classification -eq 'SUCCESS') {
                Write-Host "seed $seed completed after attempt $attempt"
                $done = $true
            } else {
                # NUMERIC_FAILURE or QNN_FAILURE: terminal for the configuration.
                Write-Host "seed $seed terminal failure: $classification"
                Write-Host ("first_bad: seed={0} step={1} phase={2} tensor={3} qnn_return={4}" -f
                    $fields.first_bad_seed, $fields.first_bad_step, $fields.first_bad_phase,
                    $fields.first_bad_tensor, $fields.qnn_execute_result)
                $aggregateStatus = $classification
                $stopDetail = "seed $seed $classification"
                break seeds
            }
        }
        if (-not $done -and $aggregateStatus -eq 'INCOMPLETE') {
            $aggregateStatus = 'BLOCKED_RESUMABLE'
            $stopDetail = "seed $seed exhausted $MaxAttemptsPerSeed attempts"
            break seeds
        }
    }

    # Aggregate over all valid seed results present on disk.
    function ConvertTo-Count([string]$Value) {
        if ($Value -match '^\d+$') { return [int]$Value }
        return 0
    }
    $finalState = Get-CompletedSeeds -SeedsRoot $seedsRoot -SeedCount $Seeds -Identity $identity
    if ($finalState.completed.Count -eq $Seeds) { $aggregateStatus = 'SUCCESS' }
    $finiteSeeds = 0; $oracleExact = 0; $freeExact = 0; $nonzeroTotal = 0; $nonfiniteTotal = 0
    $writeUnchanged = $true; $poisonTotal = 0
    foreach ($seed in ($finalState.completed.Keys | Sort-Object)) {
        $fields = $finalState.completed[$seed]
        if ($fields.all_steps_finite -eq 'true' -and $fields.final_evaluation_finite -eq 'true') { $finiteSeeds++ }
        if ($fields.oracle_exact -match '^\d+/\d+$') { $oracleExact += [int](($fields.oracle_exact -split '/')[0]) }
        if ($fields.free_exact -match '^\d+/\d+$') { $freeExact += [int](($fields.free_exact -split '/')[0]) }
        $nonzeroTotal += ConvertTo-Count $fields.qnn_nonzero_count
        $nonfiniteTotal += ConvertTo-Count $fields.nonfinite_count
        $writeUnchanged = $writeUnchanged -and ($fields.app_write_unchanged -eq 'true')
        $poisonTotal += ConvertTo-Count $fields.poison_remaining
    }
    @(
        "configuration_id=$ConfigurationId"
        "status=$aggregateStatus"
        "stop_detail=$stopDetail"
        "seeds_completed=$($finalState.completed.Count)/$Seeds"
        "finite_seeds=$finiteSeeds/$Seeds"
        "oracle_exact=$oracleExact/$($finalState.completed.Count * 4)"
        "free_exact=$freeExact/$($finalState.completed.Count * 4)"
        "qnn_nonzero_total=$nonzeroTotal"
        "nonfinite_total=$nonfiniteTotal"
        "all_outputs_written=$($writeUnchanged.ToString().ToLowerInvariant())"
        "poison_remaining=$poisonTotal"
        "config_hash=$configHash"
        "git_head=$gitHead"
    ) | Set-Content -LiteralPath (Join-Path $configRoot 'aggregate.txt') -Encoding utf8

    [pscustomobject][ordered]@{
        configuration_id = $ConfigurationId
        status = $aggregateStatus
        stop_detail = $stopDetail
        seeds_completed = "$($finalState.completed.Count)/$Seeds"
        finite_seeds = "$finiteSeeds/$Seeds"
        oracle_exact = $oracleExact
        free_exact = $freeExact
        qnn_nonzero_total = $nonzeroTotal
        nonfinite_total = $nonfiniteTotal
        results_root = $configRoot
    } | Format-List
    if ($aggregateStatus -ne 'SUCCESS' -and -not $AllowFailure) {
        throw "formal not complete: status=$aggregateStatus detail=$stopDetail"
    }
} finally {
    Pop-Location
}
