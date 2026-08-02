# SPDX-License-Identifier: Apache-2.0
# Direct-seed (EXACT_SEED) versus legacy COUNT_FROM_ONE equivalence harness.
#
# For each (configuration, seed k) pair this script runs:
#   1. the legacy process: seed_selection_mode=COUNT_FROM_ONE with
#      correctness_interval=k (seeds 1..k execute; seed k is harvested), and
#   2. the direct process: seed_selection_mode=EXACT_SEED with seed=k and
#      correctness_interval=k (only seed k executes).
# It then compares every seed_<k>_* report field bit-for-bit and records wall
# time, QNN execute counts and battery temperature for the seed-unit
# accounting (legacy seed k costs k seed-units; direct seed k costs 1).
#
# Verification anchor: the L2/H2 hashes are additionally compared against the
# published legacy baseline (docs/results/qnn-htp-generic-depth-head-2026-07)
# and FFN372 against the published post-fix formal results.
#
# Device state (serials, endpoints) stays under build/reports. This is a
# correctness run: no live-update extras, no notification interaction.
[CmdletBinding()]
param(
    [string]$QairtSdkRoot = '',
    [string]$ExpectedBuildId = '2.48.40.260702151143',
    [ValidatePattern('^[a-z0-9][a-z0-9._-]*$')][string]$WorkId = '',
    [string]$Configurations = 'l2h2,l19,ffn372',
    [switch]$InstallAuditedApk,
    # Long runs need the app's foreground service to survive process
    # reclamation; enable for configurations whose single run outlasts the
    # background-process grace window. Numerics are path-independent.
    [switch]$LiveUpdate,
    [switch]$AllowFailure,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$EvidenceSchemaVersion = 2

function Get-Sha256Hex([string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-ReportValue([string]$Report, [string]$Key) {
    $match = [regex]::Match($Report, "(?m)^$([regex]::Escape($Key))=(.*)$")
    if ($match.Success) { return $match.Groups[1].Value.Trim() }
    return $null
}

function Get-DirectConfigHash([System.Collections.IDictionary]$Config, [string]$Mode) {
    if ($Mode -notin @('EXACT_SEED', 'COUNT_FROM_ONE')) { throw "unknown seed selection mode: $Mode" }
    if (($Config['D'] % $Config['H']) -ne 0) { throw 'embedding dimension must be divisible by heads' }
    $canonical = "phonelm-direct-equivalence-v2|B=1|T=$($Config['T'])|V=$($Config['V'])" +
        "|D=$($Config['D'])|FFN=$($Config['FFN'])|L=$($Config['L'])|H=$($Config['H'])" +
        "|headDim=$($Config['D'] / $Config['H'])|lr=0.003|steps=$($Config['Steps'])|clip=disabled" +
        "|optimizer=ADAM|mode=QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC|candidate=3" +
        "|stability=LEGACY|seed_selection=$Mode|checkpoint_selection=FINAL_STEP"
    return Get-Sha256Hex $canonical
}

function Get-ExpectedScalingEvaluation([int]$Seed) {
    # The generic native protocol labels the seed-5 scaling path FORMAL and
    # all earlier requested seeds SMOKE. This label is independent of direct
    # versus count-from-one selection.
    return $(if ($Seed -eq 5) { 'FORMAL' } else { 'SMOKE' })
}

function ConvertTo-KeyMap([string]$Text) {
    $map = @{}
    foreach ($line in ($Text -split "`n")) {
        $pair = $line.TrimEnd("`r") -split '=', 2
        if ($pair.Count -eq 2) { $map[$pair[0]] = $pair[1] }
    }
    return $map
}

function Test-CacheIdentity {
    param($Meta, [string]$ApkHash, [string]$BuildId, [string]$ConfigHash,
        [string]$RunnerHash, [string]$Mode, [int]$Seed)
    return $Meta['evidence_schema'] -eq [string]$EvidenceSchemaVersion -and
        $Meta['apk_sha256'] -eq $ApkHash -and $Meta['qairt_build_id'] -eq $BuildId -and
        $Meta['config_hash'] -eq $ConfigHash -and $Meta['direct_runner_hash'] -eq $RunnerHash -and
        $Meta['seed_selection_mode'] -eq $Mode -and $Meta['requested_seed'] -eq [string]$Seed -and
        $Meta['device_seed_selection_mode'] -eq $Mode -and
        $Meta['device_requested_seed'] -eq [string]$Seed -and
        $Meta['device_executed_seed_count'] -eq [string]$(if ($Mode -eq 'EXACT_SEED') { 1 } else { $Seed }) -and
        $Meta['device_seed_count'] -eq [string]$(if ($Mode -eq 'EXACT_SEED') { 1 } else { $Seed })
}

function Assert-RunContract([string]$Report, [System.Collections.IDictionary]$Config,
        [int]$Seed, [string]$Mode) {
    $expectedCount = if ($Mode -eq 'EXACT_SEED') { 1 } else { $Seed }
    $required = [ordered]@{
        status = @('SUCCESS','FAILED','PARTIAL_SUCCESS')
        optimizer = @('ADAM'); learning_rate = @('0.003'); steps = @([string]$Config['Steps'])
        seed_selection_mode = @($Mode); requested_seed = @([string]$Seed)
        executed_seed_count = @([string]$expectedCount); seed_count = @([string]$expectedCount)
        source_protocol = @('POST_FIX_ADAM_LR0.003_STEPS320_V1')
        scaling_evaluation = @(Get-ExpectedScalingEvaluation $Seed); global_gradient_clipping = @('disabled')
        sequence_length = @([string]$Config['T']); embedding_dimension = @([string]$Config['D'])
        transformer_layers = @([string]$Config['L']); attention_heads = @([string]$Config['H'])
        feed_forward_dimension = @([string]$Config['FFN']); formal_qnn_nonzero_return_count = @('0')
    }
    foreach ($entry in $required.GetEnumerator()) {
        $actual = Get-ReportValue $Report $entry.Key
        if ($actual -notin $entry.Value) { throw "direct report contract mismatch: $($entry.Key)=$actual" }
    }
    $seedRequired = [ordered]@{
        "seed_${Seed}_completed_steps" = [string]$Config['Steps']
        "seed_${Seed}_all_steps_finite" = 'true'
        "seed_${Seed}_final_evaluation_finite" = 'true'
        "seed_${Seed}_nonfinite_count" = '0'
    }
    foreach ($entry in $seedRequired.GetEnumerator()) {
        if ((Get-ReportValue $Report $entry.Key) -ne $entry.Value) {
            throw "direct seed completeness mismatch: $($entry.Key)"
        }
    }
}

function Invoke-SelfTest {
    $config = [ordered]@{ id='self'; T=16; V=32; D=32; FFN=64; L=6; H=8; Steps=320 }
    $hash = Get-DirectConfigHash $config 'EXACT_SEED'
    if ($hash -notmatch '^[0-9a-f]{64}$') { throw 'direct config hash self-test failed' }
    if ((Get-ExpectedScalingEvaluation 1) -ne 'SMOKE' -or
        (Get-ExpectedScalingEvaluation 2) -ne 'SMOKE' -or
        (Get-ExpectedScalingEvaluation 5) -ne 'FORMAL') {
        throw 'scaling evaluation contract self-test failed'
    }
    $meta = @{evidence_schema='2';apk_sha256=('a'*64);qairt_build_id='build';config_hash=$hash;
        direct_runner_hash=('b'*64);seed_selection_mode='EXACT_SEED';requested_seed='5';
        device_seed_selection_mode='EXACT_SEED';device_requested_seed='5';device_executed_seed_count='1';device_seed_count='1';git_head=('0'*40)}
    if (-not (Test-CacheIdentity $meta ('a'*64) 'build' $hash ('b'*64) 'EXACT_SEED' 5)) { throw 'matching cache identity rejected' }
    $meta.git_head = 'f'*40
    if (-not (Test-CacheIdentity $meta ('a'*64) 'build' $hash ('b'*64) 'EXACT_SEED' 5)) { throw 'docs-only Git HEAD delta invalidated APK evidence' }
    foreach ($field in @('apk_sha256','config_hash','direct_runner_hash','device_requested_seed','device_executed_seed_count')) {
        $bad = @{} + $meta; $bad[$field] = 'mismatch'
        if (Test-CacheIdentity $bad ('a'*64) 'build' $hash ('b'*64) 'EXACT_SEED' 5) { throw "stale cache accepted: $field" }
    }
    'SELF_TEST=PASS'
}

if ($SelfTest) { Invoke-SelfTest; exit 0 }
if (-not $QairtSdkRoot) { throw '-QairtSdkRoot is required' }

$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) { throw 'adb executable unavailable' }
$package = 'com.yuubinnkyoku.phonelm'
$activity = "$package/.MainActivity"
$workRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build\reports\qnn-direct-seed-equivalence'))
if ($WorkId) { $workRoot = Join-Path $workRoot $WorkId }
$reportsParent = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build\reports'))
if (-not $workRoot.StartsWith($reportsParent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
    $workRoot -ne (Join-Path $reportsParent 'qnn-direct-seed-equivalence')) {
    throw 'report path escaped build/reports'
}
[IO.Directory]::CreateDirectory($workRoot) | Out-Null
$gitHead = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead -notmatch '^[0-9a-f]{40}$') {
    throw 'Git HEAD is unavailable'
}
$directRunnerHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()

# QAIRT gate: explicit root + build id, no fallback.
$checkOutput = (& (Join-Path $PSScriptRoot 'check_qairt.ps1') -SdkRoot $QairtSdkRoot `
        -ExpectedBuildId $ExpectedBuildId 2>&1) -join "`n"
$checkExit = $LASTEXITCODE
if ($checkExit -notin @(0, 3)) { throw "fixed QAIRT inventory check failed with exit $checkExit" }
if ($checkOutput -notmatch '(?m)^expected_build_id_match=true$') { throw 'fixed QAIRT build id was not confirmed' }

$apk = Join-Path $repoRoot 'app\build\outputs\apk\debug\app-debug.apk'
$apkHash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
& (Join-Path $PSScriptRoot 'audit_qnn_apk.ps1') -ApkPath $apk `
    -QairtSdkRoot $QairtSdkRoot -ExpectedBuildId $ExpectedBuildId `
    -ReportPath (Join-Path $workRoot 'apk-audit.txt') | Out-Null
if (-not $?) { throw 'fixed QAIRT APK audit failed' }

$online = @((& $adb devices) | Where-Object { $_ -match '^(\S+)\s+device$' } | ForEach-Object { $Matches[1] })
if ($online.Count -ne 1) { throw "expected exactly one online ADB device; found $($online.Count)" }
$device = $online[0]
$properties = @((& $adb -s $device shell getprop ro.kernel.qemu), (& $adb -s $device shell getprop ro.hardware)) -join "`n"
if ($properties -match '(?im)^(1|.*(goldfish|ranchu).*)$') { throw 'emulator device rejected' }
$packageList = (& $adb -s $device shell pm list packages $package) -join "`n"
if ($packageList -notmatch [regex]::Escape($package)) {
    if (-not $InstallAuditedApk) { throw 'package not installed; rerun with -InstallAuditedApk' }
    & $adb -s $device install -r $apk | Out-Null
}

function Read-DeviceState([string]$Device) {
    $battery = (& $adb -s $Device shell dumpsys battery) -join "`n"
    $thermal = (& $adb -s $Device shell dumpsys thermalservice) -join "`n"
    $temperature = [regex]::Match($battery, '(?m)^\s*temperature:\s*(\d+)').Groups[1].Value
    $status = [regex]::Match($thermal, '(?m)^Thermal Status:\s*(\d+)').Groups[1].Value
    if (-not $temperature -or -not $status) { return $null }
    return [pscustomobject][ordered]@{
        battery_temperature_c = [int]$temperature / 10.0
        android_thermal_status = [int]$status
    }
}

$cases = @{
    l2h2 = [ordered]@{ id = 't32_d32_ffn32_l2_h2'; T = 32; V = 32; D = 32; FFN = 32; L = 2; H = 2; Steps = 320; Seeds = @(1, 2, 5) }
    l19 = [ordered]@{ id = 't8_d16_ffn32_l19_h2'; T = 8; V = 32; D = 16; FFN = 32; L = 19; H = 2; Steps = 320; Seeds = @(2) }
    ffn372 = [ordered]@{ id = 't16_d256_ffn372_l3_h4'; T = 16; V = 32; D = 256; FFN = 372; L = 3; H = 4; Steps = 320; Seeds = @(5) }
}
$requested = $Configurations -split ','
$rows = @()
$equivalenceFailures = 0

foreach ($name in $requested) {
    $name = $name.Trim()
    if (-not $cases.Contains($name)) { throw "unknown configuration: $name" }
    $case = $cases[$name]
    $caseDir = Join-Path $workRoot $case.id
    [IO.Directory]::CreateDirectory($caseDir) | Out-Null

    function Invoke-SeedRun([hashtable]$C, [int]$Seed, [string]$Mode) {
        # Returns the device report content plus run evidence. Idempotent:
        # an already-captured report for the same case/mode/seed is reused.
        $runLabel = '{0}-{1}-{2}' -f $Mode.ToLowerInvariant(), $Seed, $C['id']
        $reportPath = Join-Path $caseDir "$runLabel.device-report.txt"
        $metaPath = Join-Path $caseDir "$runLabel.run.txt"
        $configHash = Get-DirectConfigHash $C $Mode
        if ((Test-Path -LiteralPath $reportPath -PathType Leaf) -and
            (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
            $cached = [IO.File]::ReadAllText($reportPath)
            $cachedMeta = [IO.File]::ReadAllText($metaPath)
            $cachedMetaMap = ConvertTo-KeyMap $cachedMeta
            $cacheIdentityMatches = Test-CacheIdentity $cachedMetaMap $apkHash $ExpectedBuildId `
                $configHash $directRunnerHash $Mode $Seed
            if ($cached -match '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)\r?$' -and
                $cacheIdentityMatches) {
                Assert-RunContract $cached $C $Seed $Mode
                Write-Host "$runLabel : cached report reused"
                return @{ report = $cached; meta = $cachedMeta; cached = $true }
            }
            if ($cached -match '(?m)^status=') {
                Write-Host "$runLabel : cached report rejected by terminal/identity gate"
            }
        }
        $before = Read-DeviceState $device
        if ($before -and $before.android_thermal_status -ge 5) {
            throw "Android thermal status $($before.android_thermal_status) (EMERGENCY/SHUTDOWN)"
        }
        & $adb -s $device shell am force-stop $package | Out-Null
        & $adb -s $device shell run-as $package rm -f files/device-test-result.txt | Out-Null
        $exactSeed = if ($Mode -eq 'EXACT_SEED') { [string]$Seed } else { '1' }
        $arguments = @(
            'shell', 'am', 'start', '-W', '-n', $activity,
            '--es', 'phonelm.mode', 'QNN_HTP_TINY_LANGUAGE_MODEL_GENERIC',
            '--ei', 'phonelm.batch_size', '1',
            '--ei', 'phonelm.dimension', [string]$C['D'],
            '--ei', 'phonelm.hidden_dimension', [string]$C['FFN'],
            '--ei', 'phonelm.output_dimension', [string]$C['V'],
            '--ei', 'phonelm.steps', [string]$C['Steps'],
            '--ei', 'phonelm.warmup_steps', '0',
            '--es', 'phonelm.learning_rate', '0.003',
            '--es', 'phonelm.seed', $exactSeed,
            '--ei', 'phonelm.sample_count', [string]$C['T'],
            '--ei', 'phonelm.epochs', [string]$C['L'],
            '--ei', 'phonelm.measured_steps', [string]$C['H'],
            '--ei', 'phonelm.correctness_interval', [string]$Seed,
            '--ez', 'phonelm.benchmark_mode', 'false',
            '--es', 'phonelm.seed_selection_mode', $Mode
        )
        if ($LiveUpdate) {
            # Foreground service keeps the process alive for long runs.
            $arguments += @('--ez', 'phonelm.live_update', 'true')
        }
        $started = [DateTime]::UtcNow
        $launchOutput = & $adb -s $device @arguments 2>&1
        if ($LASTEXITCODE -ne 0) { throw "activity launch failed (endpoint redacted): $launchOutput" }
        $report = ''
        $deadline = [DateTime]::UtcNow.AddSeconds(21600)
        while ([DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 20
            # The ADB endpoint may flap or rotate (network adb). The on-device
            # run is independent of the transport: tolerate unavailability and
            # keep polling any single online device for the result file.
            $onlineNow = @((& $adb devices 2>$null) | Where-Object { $_ -match '^(\S+)\s+device$' } |
                ForEach-Object { ($_ -split '\s+')[0] })
            if ($onlineNow.Count -ne 1) { continue }
            $device = $onlineNow[0]
            $candidate = (& $adb -s $device shell run-as $package cat files/device-test-result.txt 2>$null) -join "`n"
            if ($candidate -and $candidate -match '(?m)^status=(SUCCESS|FAILED|PARTIAL_SUCCESS)$') {
                $report = $candidate
                break
            }
        }
        if (-not $report) { throw "$runLabel : no terminal device result" }
        Assert-RunContract $report $C $Seed $Mode
        $completed = [DateTime]::UtcNow
        $after = Read-DeviceState $device
        $executeCount = Get-ReportValue $report "seed_${Seed}_qnn_execute_count"
        $meta = @(
            "evidence_schema=$EvidenceSchemaVersion"
            "run_label=$runLabel"
            "config_hash=$configHash"
            "direct_runner_hash=$directRunnerHash"
            "seed_selection_mode=$Mode"
            "requested_seed=$Seed"
            "device_seed_selection_mode=$(Get-ReportValue $report 'seed_selection_mode')"
            "device_requested_seed=$(Get-ReportValue $report 'requested_seed')"
            "device_executed_seed_count=$(Get-ReportValue $report 'executed_seed_count')"
            "device_seed_count=$(Get-ReportValue $report 'seed_count')"
            "device_status=$(Get-ReportValue $report 'status')"
            "scaling_evaluation=$(Get-ReportValue $report 'scaling_evaluation')"
            "formal_qnn_nonzero_return_count=$(Get-ReportValue $report 'formal_qnn_nonzero_return_count')"
            "seed_${Seed}_qnn_execute_count=$executeCount"
            "wall_time_ms=$([math]::Round(($completed - $started).TotalMilliseconds, 1))"
            "seed_units=$(if ($Mode -eq 'EXACT_SEED') { 1 } else { $Seed })"
            "started_utc=$($started.ToString('o'))"
            "completed_utc=$($completed.ToString('o'))"
            "battery_temperature_before_c=$(if ($before) { $before.battery_temperature_c } else { 'NOT_READ' })"
            "battery_temperature_after_c=$(if ($after) { $after.battery_temperature_c } else { 'NOT_READ' })"
            "thermal_status_before=$(if ($before) { $before.android_thermal_status } else { 'NOT_READ' })"
            "thermal_status_after=$(if ($after) { $after.android_thermal_status } else { 'NOT_READ' })"
            "git_head=$gitHead"
            "apk_sha256=$apkHash"
            "qairt_build_id=$ExpectedBuildId"
        ) -join "`n"
        [IO.File]::WriteAllText($reportPath, $report)
        [IO.File]::WriteAllText($metaPath, $meta)
        Write-Host "$runLabel : status=$(Get-ReportValue $report 'status') wall_ms=$([math]::Round(($completed - $started).TotalMilliseconds, 0)) qnn_execute=$executeCount"
        return @{ report = $report; meta = $meta; cached = $false }
    }

    foreach ($seed in $case.Seeds) {
        $legacy = Invoke-SeedRun $case $seed 'COUNT_FROM_ONE'
        $exact = Invoke-SeedRun $case $seed 'EXACT_SEED'
        $prefix = "seed_${seed}_"
        $legacyKeys = @{}
        foreach ($line in ($legacy.report -split "`n")) {
            $pair = $line.TrimEnd("`r") -split '=', 2
            if ($pair.Count -eq 2 -and $pair[0].StartsWith($prefix)) { $legacyKeys[$pair[0]] = $pair[1] }
        }
        $exactKeys = @{}
        foreach ($line in ($exact.report -split "`n")) {
            $pair = $line.TrimEnd("`r") -split '=', 2
            if ($pair.Count -eq 2 -and $pair[0].StartsWith($prefix)) { $exactKeys[$pair[0]] = $pair[1] }
        }
        $allKeys = @($legacyKeys.Keys + $exactKeys.Keys | Sort-Object -Unique)
        $mismatches = @()
        foreach ($key in $allKeys) {
            if (-not $legacyKeys.Contains($key)) { $mismatches += "$key missing in legacy"; continue }
            if (-not $exactKeys.Contains($key)) { $mismatches += "$key missing in exact"; continue }
            if ($legacyKeys[$key] -ne $exactKeys[$key]) {
                $mismatches += "$key legacy=$($legacyKeys[$key]) exact=$($exactKeys[$key])"
            }
        }
        # Equivalence-irrelevant keys are none: seed_<k>_* is the harvested
        # contract (losses, accuracies, hashes, generation, execute counts).
        $equivalent = $mismatches.Count -eq 0 -and $allKeys.Count -gt 0
        if (-not $equivalent) { $equivalenceFailures++ }
        $legacyMeta = @{}
        foreach ($line in ($legacy.meta -split "`n")) { $p = $line.TrimEnd("`r") -split '=', 2; if ($p.Count -eq 2) { $legacyMeta[$p[0]] = $p[1] } }
        $exactMeta = @{}
        foreach ($line in ($exact.meta -split "`n")) { $p = $line.TrimEnd("`r") -split '=', 2; if ($p.Count -eq 2) { $exactMeta[$p[0]] = $p[1] } }
        $contractMatches = $legacyMeta['device_seed_selection_mode'] -eq 'COUNT_FROM_ONE' -and
            $exactMeta['device_seed_selection_mode'] -eq 'EXACT_SEED' -and
            $legacyMeta['device_requested_seed'] -eq [string]$seed -and
            $exactMeta['device_requested_seed'] -eq [string]$seed -and
            $legacyMeta['device_executed_seed_count'] -eq [string]$seed -and
            $exactMeta['device_executed_seed_count'] -eq '1'
        if (-not $contractMatches) { $equivalenceFailures++; $equivalent = $false }
        $rows += [pscustomobject][ordered]@{
            configuration = $case.id
            seed = $seed
            evidence_schema = $EvidenceSchemaVersion
            legacy_config_hash = $legacyMeta['config_hash']
            exact_config_hash = $exactMeta['config_hash']
            direct_runner_hash = $directRunnerHash
            compared_fields = $allKeys.Count
            mismatch_count = $mismatches.Count
            bitwise_equivalent = $equivalent
            contract_fields_valid = $contractMatches
            legacy_selection_mode = $legacyMeta['device_seed_selection_mode']
            exact_selection_mode = $exactMeta['device_seed_selection_mode']
            legacy_requested_seed = $legacyMeta['device_requested_seed']
            exact_requested_seed = $exactMeta['device_requested_seed']
            legacy_executed_seed_count = $legacyMeta['device_executed_seed_count']
            exact_executed_seed_count = $exactMeta['device_executed_seed_count']
            legacy_wall_time_ms = $legacyMeta['wall_time_ms']
            exact_wall_time_ms = $exactMeta['wall_time_ms']
            legacy_seed_units = $legacyMeta['seed_units']
            exact_seed_units = $exactMeta['seed_units']
            legacy_qnn_execute_count = $legacyMeta["seed_${seed}_qnn_execute_count"]
            exact_qnn_execute_count = $exactMeta["seed_${seed}_qnn_execute_count"]
            initial_parameter_hash = $legacyKeys["${prefix}step_1_parameter_before_canonical_hash"]
            initial_adam_m_hash = $legacyKeys["${prefix}step_1_first_moment_before_canonical_hash"]
            initial_adam_v_hash = $legacyKeys["${prefix}step_1_second_moment_before_canonical_hash"]
            all_step_loss_hash = $legacyKeys["${prefix}all_step_loss_canonical_hash"]
            all_step_accuracy_hash = $legacyKeys["${prefix}all_step_accuracy_canonical_hash"]
            final_parameter_hash = $legacyKeys["${prefix}final_parameter_canonical_hash"]
            final_logits_hash = $legacyKeys["${prefix}step_$($case.Steps)_logits_canonical_hash"]
        }
        if (-not $equivalent) {
            Write-Host "MISMATCH ${($case.id)} seed ${seed}:"
            $mismatches | ForEach-Object { Write-Host "  $_" }
        } else {
            Write-Host "$($case.id) seed ${seed}: $($allKeys.Count) fields bitwise equal"
        }
    }
}

$csv = $rows | ConvertTo-Csv -NoTypeInformation
$csv | Set-Content -LiteralPath (Join-Path $workRoot 'direct-seed-equivalence-private.csv') -Encoding utf8
$totalFields = ($rows | Measure-Object -Property compared_fields -Sum).Sum
$equal = @($rows | Where-Object { $_.bitwise_equivalent }).Count
Write-Host "direct_seed_bitwise_equivalent=$($equivalenceFailures -eq 0)"
Write-Host "equivalence_cases=$equal/$($rows.Count) compared_fields_total=$totalFields"
if ($equivalenceFailures -gt 0 -and -not $AllowFailure) {
    throw "direct-seed equivalence failures: $equivalenceFailures"
}
