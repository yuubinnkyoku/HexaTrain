# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yuubinnkyoku
[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-autoregressive-validation-cpu'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-autoregressive-validation-2026-08'),
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
$allowed = @(
    'README.md', 'manifest.json', 'dataset-partitions.csv', 'dataset-overlap.csv',
    'dataset-hashes.csv', 'checkpoint-replay.csv', 'checkpoint-trajectory.csv',
    'ar-validation-metrics.csv', 'ar-development-metrics.csv', 'ar-final-holdout-metrics.csv',
    'selection-decisions.csv', 'cpu-smoke.csv', 'htp-smoke.csv', 'formal-seeds.csv',
    'selected-step-distribution.csv', 'legacy-generation.csv', 'decision.csv', 'thermal.csv'
)
$copied = @(
    'dataset-partitions.csv', 'dataset-overlap.csv', 'dataset-hashes.csv', 'checkpoint-replay.csv',
    'checkpoint-trajectory.csv', 'ar-validation-metrics.csv', 'ar-development-metrics.csv',
    'ar-final-holdout-metrics.csv', 'selection-decisions.csv'
)

function Fail([string]$Message) { throw "autoregressive public export: $Message" }

function Safe([string]$Text) {
    # Results are public only when they contain no concrete local provenance,
    # package payload, endpoint, device, or model-state material.
    return $Text -notmatch '(?im)([a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|(?:^|[=,:;\s])/(?!/)[a-z0-9._-]+(?:/|\b)|(?:^|[,\s])(?:files|cache|code_cache|shared_prefs|databases|no_backup)/|\.(?:apk|so|dll|bin|exe)(?:\b|[\\/])|\b(?:adb[_ -]?(?:endpoint|serial)|device[_ -]?serial|hardware[_ -]?identifier|android_id|app[-_ ]?private(?:[_ -]?path)?|apk[_ -]?(?:sha(?:256)?|hash)|raw[_ -]?logcat)\s*[:=]|\badb\s+-s\s+|\b(?:raw[_ -]?(?:checkpoint|parameters?)|raw[_ -]?(?:adam|optimizer)(?:[_ -]?state)?|raw[_ -]?tensor(?:[_ -]?(?:dump|data))?)\b|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'
}

function WriteUtf8([string]$Name, [string]$Text) {
    if (-not (Safe $Text)) { Fail "unsafe public content in $Name" }
    [IO.File]::WriteAllText((Join-Path $OutputRoot $Name), $Text, $utf8)
}

function CsvText($Rows) { return (($Rows | ConvertTo-Csv -NoTypeInformation) -join "`n") + "`n" }

function RequireUnderRepository([string]$Path, [string]$Purpose) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $repoRoot.TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { Fail "$Purpose must be inside the repository" }
    return $full
}

function RequireOutputRoot([string]$Path) {
    $full = RequireUnderRepository $Path 'OutputRoot'
    $docsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'docs\results\qnn-htp-autoregressive-validation-2026-08'))
    $buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if ($full -ne $docsRoot -and -not $full.StartsWith($buildRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Fail 'OutputRoot must be the designated public directory or a build-contained validation directory'
    }
    return $full
}

function RequireHeader([string]$Name, [string[]]$Expected) {
    $path = Join-Path $InputRoot $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing source file: $Name" }
    $first = ([IO.File]::ReadAllLines($path))[0]
    $actual = @($first.Trim('"').Split('","'))
    if (($actual -join ',') -ne ($Expected -join ',')) { Fail "source schema mismatch: $Name" }
}

$script:Invariant = [Globalization.CultureInfo]::InvariantCulture
$script:FnvMask = [Numerics.BigInteger]::Parse('18446744073709551615')
$script:FnvModulus = $script:FnvMask + 1

function ParseFinite([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing numeric value: $RowName.$Field" }
    $parsed = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float,
            $script:Invariant, [ref]$parsed)) { Fail "non-numeric value: $RowName.$Field" }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
        Fail "non-finite value: $RowName.$Field"
    }
    return $parsed
}

function ParseInt([string]$Value, [string]$Field, [string]$RowName) {
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer,
            $script:Invariant, [ref]$parsed)) { Fail "non-integer value: $RowName.$Field" }
    return $parsed
}

function ParseTokenList([string]$Value, [string]$Field, [string]$RowName) {
    if ([string]::IsNullOrWhiteSpace($Value)) { Fail "missing token list: $RowName.$Field" }
    $values = foreach ($item in $Value.Split(':')) {
        $parsed = 0
        if (-not [int]::TryParse($item, [Globalization.NumberStyles]::Integer,
                $script:Invariant, [ref]$parsed) -or $parsed -lt 0) {
            Fail "invalid token list: $RowName.$Field"
        }
        $parsed
    }
    return @($values)
}

function FnvBytes([byte[]]$Bytes, [Numerics.BigInteger]$Hash) {
    foreach ($byte in $Bytes) {
        $Hash = ((($Hash -bxor [Numerics.BigInteger]$byte) *
                    [Numerics.BigInteger]1099511628211) % $script:FnvModulus)
    }
    return $Hash
}

function FnvU32([int]$Value, [Numerics.BigInteger]$Hash) {
    return FnvBytes ([BitConverter]::GetBytes([uint32]$Value)) $Hash
}

function FnvU64([uint64]$Value, [Numerics.BigInteger]$Hash) {
    return FnvBytes ([BitConverter]::GetBytes($Value)) $Hash
}

function FnvString([string]$Value, [Numerics.BigInteger]$Hash) {
    $bytes = $utf8.GetBytes($Value)
    $Hash = FnvU64 ([uint64]$bytes.Length) $Hash
    return FnvBytes $bytes $Hash
}

function FnvTokens([int[]]$Values, [Numerics.BigInteger]$Hash) {
    $Hash = FnvU64 ([uint64]$Values.Count) $Hash
    foreach ($value in $Values) { $Hash = FnvU32 $value $Hash }
    return $Hash
}

function ComputePartitionHash($Rows, [string]$Partition, [string]$Domain) {
    $hash = [Numerics.BigInteger]1469598103934665603
    $hash = FnvU32 3 $hash
    $hash = FnvString 'AR_ROLLOUT_NLL_V1' $hash
    $hash = FnvString $Partition $hash
    $hash = FnvString $Domain $hash
    foreach ($row in $Rows) {
        $rowName = "$Partition/$($row.case_id)"
        $hash = FnvString ([string]$row.case_id) $hash
        $hash = FnvString ([string]$row.domain) $hash
        foreach ($field in @('active_family','distractor_family','active_phase','distractor_phase','active_suffix_length','rollout_length')) {
            $hash = FnvU32 (ParseInt $row.$field $field $rowName) $hash
        }
        $hash = FnvTokens (ParseTokenList $row.initial_prefix 'initial_prefix' $rowName) $hash
        $hash = FnvTokens (ParseTokenList $row.targets 'targets' $rowName) $hash
    }
    return ('fnv1a64:{0:x16}' -f [uint64]$hash)
}

function GetTransitionCounts($Rows) {
    $counts = @{}
    foreach ($row in $Rows) {
        $rowName = "$($row.partition)/$($row.case_id)"
        $prefix = ParseTokenList $row.initial_prefix 'initial_prefix' $rowName
        $targets = ParseTokenList $row.targets 'targets' $rowName
        if ($targets.Count -ne (ParseInt $row.rollout_length 'rollout_length' $rowName)) {
            Fail "target/rollout length mismatch: $rowName"
        }
        $previous = $prefix[$prefix.Count - 1]
        for ($index = 0; $index -lt $targets.Count; $index++) {
            $target = $targets[$index]
            if ($row.partition -eq 'TRAIN') { $previous = $prefix[$index] }
            $key = "$previous>$target"
            if (-not $counts.ContainsKey($key)) { $counts[$key] = 0 }
            $counts[$key]++
            $previous = $target
        }
    }
    return $counts
}

function CountSetIntersection($Left, $Right) {
    $count = 0
    foreach ($key in $Left.Keys) { if ($Right.ContainsKey($key)) { $count++ } }
    return $count
}

function CountMultisetIntersection($Left, $Right) {
    $count = 0
    foreach ($key in $Left.Keys) {
        if ($Right.ContainsKey($key)) { $count += [math]::Min([int]$Left[$key], [int]$Right[$key]) }
    }
    return $count
}

function NewKeySet($Rows, [scriptblock]$Selector) {
    $set = @{}
    foreach ($row in $Rows) {
        $key = [string](& $Selector $row)
        if ($set.ContainsKey($key)) { Fail "duplicate partition key: $key" }
        $set[$key] = $true
    }
    return $set
}

function AssertDatasetEvidence($Rows, $HashRows, $OverlapRows) {
    $partitions = @('TRAIN','AR_VALIDATION_V3','AR_DEVELOPMENT_V3','AR_FINAL_HOLDOUT_V3')
    $domains = @{
        'TRAIN'='HOMOGENEOUS_PHASE0'; 'AR_VALIDATION_V3'='MIXED_PREFIX_DISTRACTOR_OFFSET_1';
        'AR_DEVELOPMENT_V3'='MIXED_PREFIX_DISTRACTOR_OFFSET_2'; 'AR_FINAL_HOLDOUT_V3'='MIXED_PREFIX_DISTRACTOR_OFFSET_3'
    }
    $expectedHashes = @{
        'TRAIN'='fnv1a64:5a64ca2d1aa7f29f'; 'AR_VALIDATION_V3'='fnv1a64:aad785bd4dc88dc9';
        'AR_DEVELOPMENT_V3'='fnv1a64:bd464d2a6e192d36'; 'AR_FINAL_HOLDOUT_V3'='fnv1a64:aa5081e6df658b4a'
    }
    $sets = @{}
    $transitions = @{}
    foreach ($partition in $partitions) {
        $items = @($Rows | Where-Object { $_.partition -eq $partition })
        $expectedCount = if ($partition -eq 'TRAIN') { 4 } else { 24 }
        if ($items.Count -ne $expectedCount) { Fail "dataset case count mismatch: $partition" }
        $sets[$partition] = [ordered]@{
            ids=(NewKeySet $items { param($r) $r.case_id });
            prefixes=(NewKeySet $items { param($r) $r.initial_prefix });
            sequences=(NewKeySet $items { param($r) "$($r.initial_prefix)>$($r.targets)" })
        }
        $transitions[$partition] = GetTransitionCounts $items
        $hashRow = @($HashRows | Where-Object { $_.partition -eq $partition })
        if ($hashRow.Count -ne 1) { Fail "dataset hash row missing: $partition" }
        $computedHash = ComputePartitionHash $items $partition $domains[$partition]
        if ($computedHash -ne $expectedHashes[$partition] -or $hashRow[0].hash -ne $computedHash) {
            Fail "dataset hash mismatch: $partition (computed=$computedHash)"
        }
        if ((ParseInt $hashRow[0].case_count 'case_count' $partition) -ne $items.Count) {
            Fail "dataset hash case count mismatch: $partition"
        }
        $occurrences = ($transitions[$partition].Values | Measure-Object -Sum).Sum
        if ((ParseInt $hashRow[0].target_transition_occurrences 'target_transition_occurrences' $partition) -ne $occurrences -or
            (ParseInt $hashRow[0].unique_target_transitions 'unique_target_transitions' $partition) -ne $transitions[$partition].Count) {
            Fail "dataset transition summary mismatch: $partition (computed=$occurrences/$($transitions[$partition].Count), reported=$($hashRow[0].target_transition_occurrences)/$($hashRow[0].unique_target_transitions))"
        }
    }
    foreach ($partition in $partitions) {
        foreach ($other in $partitions) {
            if ($partition -ge $other) { continue }
            $expected = @($OverlapRows | Where-Object { $_.left -eq $partition -and $_.right -eq $other })
            if ($expected.Count -ne 1) { $expected = @($OverlapRows | Where-Object { $_.left -eq $other -and $_.right -eq $partition }) }
            if ($expected.Count -ne 1) { Fail "dataset overlap row missing: $partition/$other" }
            $row = $expected[0]
            $actual = @(
                CountSetIntersection $sets[$partition].ids $sets[$other].ids
                CountSetIntersection $sets[$partition].prefixes $sets[$other].prefixes
                CountSetIntersection $sets[$partition].sequences $sets[$other].sequences
                CountSetIntersection $transitions[$partition] $transitions[$other]
                CountMultisetIntersection $transitions[$partition] $transitions[$other]
            )
            $reported = @(
                ParseInt $row.case_id_overlap 'case_id_overlap' "$partition/$other"
                ParseInt $row.initial_prefix_overlap 'initial_prefix_overlap' "$partition/$other"
                ParseInt $row.full_sequence_overlap 'full_sequence_overlap' "$partition/$other"
                ParseInt $row.unique_transition_overlap 'unique_transition_overlap' "$partition/$other"
                ParseInt $row.transition_occurrence_multiset_overlap 'transition_occurrence_multiset_overlap' "$partition/$other"
            )
            if (($actual -join ',') -ne ($reported -join ',')) { Fail "dataset overlap mismatch: $partition/$other" }
        }
    }
    return [pscustomobject]@{ Rows=$Rows; Hashes=$HashRows; Overlaps=$OverlapRows }
}

function AssertFiniteMetricRow($row, [string]$RowName) {
    foreach ($field in @('ar_rollout_nll','teacher_forced_nll','teacher_forced_gap','mean_first_error_position','post_error_recovery_tokens')) {
        [void](ParseFinite $row.$field $field $RowName)
    }
    foreach ($field in @('token_exact','token_total','sequence_exact','sequence_total')) {
        $value = ParseInt $row.$field $field $RowName
        if ($value -lt 0) { Fail "negative metric: $RowName.$field" }
    }
    if ($row.all_finite -ne 'true') { Fail "finite row is not marked finite: $RowName" }
}

function AssertReplayEvidence($Rows) {
    $steps = @(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320)
    $configs = @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')
    if ($Rows.Count -ne 92) { Fail 'checkpoint replay row count mismatch' }
    if (@($Rows | Where-Object status -eq 'REPLAY').Count -ne 56 -or @($Rows | Where-Object status -eq 'NOT_AVAILABLE').Count -ne 36) {
        Fail 'checkpoint replay availability count mismatch'
    }
    foreach ($config in $configs) {
        $items = @($Rows | Where-Object configuration_id -eq $config)
        if ($items.Count -ne 23 -or (($items.step -join ',') -ne ($steps -join ','))) { Fail "checkpoint replay cadence mismatch: $config" }
        foreach ($row in $items) {
            $rowName = "checkpoint-replay/$config/$($row.step)"
            [void](ParseInt $row.depth 'depth' $rowName); [void](ParseInt $row.seed 'seed' $rowName); [void](ParseInt $row.step 'step' $rowName)
            if ($row.status -eq 'REPLAY') {
                AssertFiniteMetricRow $row $rowName
                [void](ParseFinite $row.parameter_norm 'parameter_norm' $rowName)
                foreach ($field in @('training_loss','gradient_norm','update_to_parameter')) {
                    if ($row.$field -ne 'NOT_AVAILABLE') { [void](ParseFinite $row.$field $field $rowName) }
                }
            } elseif ($row.status -eq 'NOT_AVAILABLE') {
                if ($row.all_finite -ne 'false' -or $row.ar_rollout_nll -ne 'NOT_FINITE' -or
                    $row.teacher_forced_nll -ne 'NOT_FINITE' -or $row.teacher_forced_gap -ne 'NOT_FINITE' -or
                    $row.parameter_norm -ne 'NOT_AVAILABLE') { Fail "checkpoint unavailable marker mismatch: $rowName" }
            } else { Fail "unknown checkpoint replay status: $rowName" }
        }
    }
}

function AssertExactStepSet($Rows, [int[]]$Expected, [string]$Name) {
    $parsed = foreach ($row in $Rows) { ParseInt $row.step 'step' $Name }
    if (@($parsed | Sort-Object -Unique).Count -ne $Expected.Count -or
        (($parsed | Sort-Object -Unique) -join ',') -ne (($Expected | Sort-Object) -join ',')) {
        Fail "step set mismatch: $Name"
    }
}

function GetConfigurationIdentity([string]$Configuration) {
    switch ($Configuration) {
        'L19_SEED_1' { return [pscustomobject]@{ depth=19; seed=1 } }
        'L19_SEED_2' { return [pscustomobject]@{ depth=19; seed=2 } }
        'L19_SEED_4' { return [pscustomobject]@{ depth=19; seed=4 } }
        'L18_SEED_2_CONTROL' { return [pscustomobject]@{ depth=18; seed=2 } }
        default { Fail "unknown configuration identity: $Configuration" }
    }
}

function AssertTrainingEvidence($Trajectory) {
    $steps = @(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320)
    $configs = @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')
    if ($Trajectory.Count -ne 92) { Fail 'checkpoint trajectory row count mismatch' }
    foreach ($config in $configs) {
        $items = @($Trajectory | Where-Object configuration_id -eq $config)
        if ($items.Count -ne 23) { Fail "checkpoint trajectory cadence mismatch: $config" }
        AssertExactStepSet $items $steps "checkpoint-trajectory/$config"
        $identity = GetConfigurationIdentity $config
        foreach ($row in $items) {
            $rowName = "checkpoint-trajectory/$config/$($row.step)"
            if ($row.source -ne 'CPU_REFERENCE_REGENERATION' -or (ParseInt $row.depth 'depth' $rowName) -ne $identity.depth -or
                (ParseInt $row.seed 'seed' $rowName) -ne $identity.seed) { Fail "trajectory identity mismatch: $rowName" }
            [void](ParseInt $row.step 'step' $rowName)
            foreach ($field in @('loss','accuracy','gradient_norm','parameter_norm','update_norm','update_to_parameter')) {
                [void](ParseFinite $row.$field $field $rowName)
            }
        }
    }
}

function GetMetricObject($Row, [string]$Name) {
    AssertFiniteMetricRow $Row $Name
    return [pscustomobject]@{
        nll=(ParseFinite $Row.ar_rollout_nll 'ar_rollout_nll' $Name)
        token=(ParseInt $Row.token_exact 'token_exact' $Name)
        tokenTotal=(ParseInt $Row.token_total 'token_total' $Name)
        sequence=(ParseInt $Row.sequence_exact 'sequence_exact' $Name)
        sequenceTotal=(ParseInt $Row.sequence_total 'sequence_total' $Name)
        step=(ParseInt $Row.step 'step' $Name)
    }
}

function MetricBetter($Candidate, $Current) {
    $tolerance = 0.0000001
    if ($Candidate.nll -lt ($Current.nll - $tolerance)) { return $true }
    if ([math]::Abs($Candidate.nll - $Current.nll) -gt $tolerance) { return $false }
    if ($Candidate.token -ne $Current.token) { return $Candidate.token -gt $Current.token }
    if ($Candidate.sequence -ne $Current.sequence) { return $Candidate.sequence -gt $Current.sequence }
    return $Candidate.step -lt $Current.step
}

function MetricNonworse($Candidate, $Baseline) {
    return ($Candidate.nll -le ($Baseline.nll + 0.0000001) -and
        $Candidate.token -ge $Baseline.token -and $Candidate.sequence -ge $Baseline.sequence)
}

function MetricStrict($Candidate, $Baseline) {
    $nonworse = MetricNonworse $Candidate $Baseline
    return ($nonworse -and
        ($Candidate.nll -lt ($Baseline.nll - 0.0000001) -or
         $Candidate.token -gt $Baseline.token -or $Candidate.sequence -gt $Baseline.sequence))
}

function AggregateMetric($Rows, [string]$Name) {
    $metrics = @($Rows | ForEach-Object { GetMetricObject $_ $Name })
    if ($metrics.Count -eq 0) { Fail "empty metric aggregate: $Name" }
    return [pscustomobject]@{
        nll=(($metrics | ForEach-Object nll | Measure-Object -Average).Average)
        token=(($metrics | ForEach-Object token | Measure-Object -Sum).Sum)
        tokenTotal=(($metrics | ForEach-Object tokenTotal | Measure-Object -Sum).Sum)
        sequence=(($metrics | ForEach-Object sequence | Measure-Object -Sum).Sum)
        sequenceTotal=(($metrics | ForEach-Object sequenceTotal | Measure-Object -Sum).Sum)
        step=-1
    }
}

function AssertCpuAndSelectionEvidence($Validation, $Development, $Selection, $Cpu, $Decision) {
    $configs = @('L19_SEED_1','L19_SEED_2','L19_SEED_4','L18_SEED_2_CONTROL')
    $steps = @(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320)
    $expectedSelection = [ordered]@{ L19_SEED_1=16; L19_SEED_2=4; L19_SEED_4=12; L18_SEED_2_CONTROL=4 }
    $validationCpu = @($Validation | Where-Object source -eq 'CPU_REFERENCE_REGENERATION')
    if ($validationCpu.Count -ne 92) { Fail 'CPU validation must be exactly 4x23' }
    if ($Selection.Count -ne 4 -or $Cpu.Count -ne 4) { Fail 'selection/CPU summary row count mismatch' }
    $derivedSelections = @{}
    $validationSelectedMetrics = @{}
    foreach ($config in $configs) {
        $identity = GetConfigurationIdentity $config
        $validationItems = @($validationCpu | Where-Object configuration_id -eq $config)
        if ($validationItems.Count -ne 23) { Fail "CPU validation cadence mismatch: $config" }
        AssertExactStepSet $validationItems $steps "ar-validation/$config"
        $bestMetric = $null
        $selectedRoleRows = @($validationItems | Where-Object role -eq 'SELECTED')
        if ($selectedRoleRows.Count -ne 1) { Fail "validation selected-role count mismatch: $config" }
        foreach ($row in $validationItems) {
            $rowName = "ar-validation/$config/$($row.step)"
            if ($row.source -ne 'CPU_REFERENCE_REGENERATION' -or $row.partition -ne 'AR_VALIDATION_V3' -or
                (ParseInt $row.depth 'depth' $rowName) -ne $identity.depth -or (ParseInt $row.seed 'seed' $rowName) -ne $identity.seed -or
                $row.role -notin @('SELECTED','TRAJECTORY')) { Fail "validation identity mismatch: $rowName" }
            AssertFiniteMetricRow $row $rowName
            $metric = GetMetricObject $row $rowName
            if ($metric.tokenTotal -ne 144 -or $metric.sequenceTotal -ne 24) { Fail "validation denominator mismatch: $rowName" }
            if ($null -eq $bestMetric -or (MetricBetter $metric $bestMetric)) { $bestMetric = $metric }
            if ($row.role -eq 'SELECTED' -and $metric.step -ne $bestMetric.step) {
                # The selected role is checked against the recomputed comparator below.
                continue
            }
        }
        $selectedMetric = GetMetricObject $selectedRoleRows[0] "ar-validation/$config/SELECTED"
        if ($selectedMetric.step -ne $bestMetric.step) { Fail "validation best-step mismatch: $config" }
        $derivedSelections[$config] = $bestMetric.step
        $validationSelectedMetrics[$config] = $selectedMetric
        foreach ($row in $validationItems) {
            $expectedRole = if ((ParseInt $row.step 'step' "ar-validation/$config") -eq $bestMetric.step) { 'SELECTED' } else { 'TRAJECTORY' }
            if ($row.role -ne $expectedRole) { Fail "validation role mismatch: $config/$($row.step)" }
        }
        $selectionRows = @($Selection | Where-Object configuration_id -eq $config)
        $cpuRows = @($Cpu | Where-Object configuration_id -eq $config)
        if ($selectionRows.Count -ne 1 -or $cpuRows.Count -ne 1) { Fail "selection/CPU identity row mismatch: $config" }
        $selectionRow = $selectionRows[0]; $cpuRow = $cpuRows[0]
        if ($selectionRow.mode -ne 'BEST_AR_VALIDATION_V1' -or [int]$selectionRow.selected_step -ne $bestMetric.step -or
            [int]$cpuRow.selected_step -ne $bestMetric.step -or $cpuRow.development_gate -notin @('REJECT','PASS') -or
            $cpuRow.final_gate -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED')) { Fail "selection identity mismatch: $config" }
        $selectionNll = ParseFinite $selectionRow.validation_ar_nll 'validation_ar_nll' "selection/$config"
        $cpuNll = ParseFinite $cpuRow.validation_ar_nll 'validation_ar_nll' "cpu/$config"
        if ([math]::Abs($selectionNll - $selectedMetric.nll) -gt 0.000000000001 -or
            [math]::Abs($cpuNll - $selectedMetric.nll) -gt 0.000000000001 -or
            (ParseInt $selectionRow.validation_token_exact 'validation_token_exact' "selection/$config") -ne $selectedMetric.token -or
            (ParseInt $selectionRow.validation_sequence_exact 'validation_sequence_exact' "selection/$config") -ne $selectedMetric.sequence) {
            Fail "selection metric mismatch: $config"
        }
    }
    $derivedConfigKeyText = (($derivedSelections.Keys | Sort-Object) -join ',')
    $expectedConfigKeyText = (($configs | Sort-Object) -join ',')
    if ($derivedConfigKeyText -ne $expectedConfigKeyText) { Fail 'selection configuration set mismatch' }
    if (($configs | ForEach-Object { $derivedSelections[$_] } ) -join ',' -ne '16,4,12,4') { Fail 'derived selected-step distribution mismatch' }
    if ($Development.Count -ne 8) { Fail 'development evidence must contain 8 rows' }
    $developmentByConfig = @{}
    foreach ($config in $configs) {
        $identity = GetConfigurationIdentity $config
        $developmentItems = @($Development | Where-Object configuration_id -eq $config)
        if ($developmentItems.Count -ne 2) { Fail "development row count mismatch: $config" }
        $developmentByConfig[$config] = $developmentItems
        $roles = @($developmentItems.role | Sort-Object)
        if (($roles -join ',') -ne 'FINAL_STEP,SELECTED') { Fail "development role set mismatch: $config" }
        foreach ($row in $developmentItems) {
            $rowName = "ar-development/$config/$($row.role)"
            $expectedDevStep = if ($row.role -eq 'FINAL_STEP') { 320 } else { $derivedSelections[$config] }
            if ($row.source -ne 'CPU_REFERENCE_REGENERATION' -or $row.partition -ne 'AR_DEVELOPMENT_V3' -or
                (ParseInt $row.depth 'depth' $rowName) -ne $identity.depth -or (ParseInt $row.seed 'seed' $rowName) -ne $identity.seed -or
                (ParseInt $row.step 'step' $rowName) -ne $expectedDevStep) {
                Fail "development identity mismatch: $rowName"
            }
            $metric = GetMetricObject $row $rowName
            if ($metric.tokenTotal -ne 144 -or $metric.sequenceTotal -ne 24) { Fail "development denominator mismatch: $rowName" }
        }
    }
    $selectedDevelopment = @{}; $finalDevelopment = @{}
    foreach ($config in $configs) {
        $selectedDevelopment[$config] = GetMetricObject (@($developmentByConfig[$config] | Where-Object role -eq 'SELECTED')[0]) "development/$config/SELECTED"
        $finalDevelopment[$config] = GetMetricObject (@($developmentByConfig[$config] | Where-Object role -eq 'FINAL_STEP')[0]) "development/$config/FINAL_STEP"
    }
    $seed2Strict = MetricStrict $selectedDevelopment['L19_SEED_2'] $finalDevelopment['L19_SEED_2']
    $l19SelectedRows = @($developmentByConfig['L19_SEED_1'] | Where-Object role -eq 'SELECTED') + @($developmentByConfig['L19_SEED_2'] | Where-Object role -eq 'SELECTED') + @($developmentByConfig['L19_SEED_4'] | Where-Object role -eq 'SELECTED')
    $l19FinalRows = @($developmentByConfig['L19_SEED_1'] | Where-Object role -eq 'FINAL_STEP') + @($developmentByConfig['L19_SEED_2'] | Where-Object role -eq 'FINAL_STEP') + @($developmentByConfig['L19_SEED_4'] | Where-Object role -eq 'FINAL_STEP')
    $l19Selected = AggregateMetric $l19SelectedRows 'development/L19/SELECTED'
    $l19Final = AggregateMetric $l19FinalRows 'development/L19/FINAL_STEP'
    $controlNonworse = MetricNonworse $selectedDevelopment['L18_SEED_2_CONTROL'] $finalDevelopment['L18_SEED_2_CONTROL']
    $l19PooledNonworse = MetricNonworse $l19Selected $l19Final
    $l19PooledStrict = MetricStrict $l19Selected $l19Final
    $caseCollapse = @($configs | Where-Object { $selectedDevelopment[$_].token -eq 0 -and $finalDevelopment[$_].token -gt 0 }).Count
    $multiSeedSupport = @($configs | Where-Object { $_ -like 'L19_*' -and $selectedDevelopment[$_].step -lt 320 -and (MetricNonworse $selectedDevelopment[$_] $finalDevelopment[$_]) }).Count
    $computedDevelopmentGate = if ($seed2Strict -and $l19PooledNonworse -and $controlNonworse -and $caseCollapse -eq 0 -and $multiSeedSupport -ge 2) { 'PASS' } else { 'REJECT' }
    if ($Decision.Count -ne 1 -or $Decision[0].development_gate -ne $computedDevelopmentGate) { Fail "development gate mismatch (computed=$computedDevelopmentGate)" }
    foreach ($row in $Cpu) { if ($row.development_gate -ne $computedDevelopmentGate) { Fail "CPU development gate mismatch: $($row.configuration_id)" } }
    if ($Decision[0].final_holdout_gate -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED') -or $Decision[0].checkpoint_selection_mode -ne 'NONE' -or
        $Decision[0].stabilizer -ne 'NONE' -or $Decision[0].replay_available -ne '56' -or $Decision[0].replay_missing -ne '36' -or
        $Decision[0].classification -ne 'AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE') { Fail 'decision evidence mismatch' }
    return [pscustomobject]@{ developmentGate=$computedDevelopmentGate; seed2Strict=$seed2Strict; l19PooledNonworse=$l19PooledNonworse; l19PooledStrict=$l19PooledStrict; controlNonworse=$controlNonworse; caseCollapse=$caseCollapse; multiSeedSupport=$multiSeedSupport; derivedSelections=$derivedSelections }
}

function AssertLegacyAnchor() {
    $canonical = Join-Path $repoRoot 'docs\results\qnn-htp-first-nonfinite-2026-07\post-fix-formal.csv'
    if (-not (Test-Path -LiteralPath $canonical -PathType Leaf)) { Fail 'canonical legacy generation anchor is missing' }
    $rows = @(Import-Csv -LiteralPath $canonical)
    $anchor = @($rows | Where-Object { $_.configuration -eq 'T8_D16_FFN32_L19_H2' })
    $anchorSeeds = @($anchor | ForEach-Object { ParseInt $_.seed 'seed' 'legacy-anchor' })
    $anchorSteps = @($anchor | ForEach-Object { ParseInt $_.steps 'steps' 'legacy-anchor' })
    $oracleSum = ($anchor | ForEach-Object { ParseInt $_.oracle_exact 'oracle_exact' 'legacy-anchor' } | Measure-Object -Sum).Sum
    $freeSum = ($anchor | ForEach-Object { ParseInt $_.free_exact 'free_exact' 'legacy-anchor' } | Measure-Object -Sum).Sum
    if ($anchor.Count -ne 5 -or @($anchor | Where-Object { $_.finite -ne 'true' -or $_.qnn_nonzero_count -ne '0' -or $_.nonfinite_count -ne '0' }).Count -ne 0 -or
        (@($anchorSeeds | Sort-Object -Unique).Count -ne 5) -or (($anchorSeeds | Sort-Object -Unique) -join ',' -ne '1,2,3,4,5') -or
        @($anchorSteps | Where-Object { $_ -ne 320 }).Count -ne 0 -or $oracleSum -ne 13 -or $freeSum -ne 13) {
        Fail 'canonical legacy Oracle/Free anchor mismatch'
    }
    return [pscustomobject]@{ oracle_exact='13/20'; free_exact='13/20'; rows=$anchor.Count; oracle_sum=$oracleSum; free_sum=$freeSum }
}

function AssertSourceEvidence() {
    $sourceNames = $copied + @('cpu-smoke.csv','decision.csv','manifest.json')
    foreach ($name in $sourceNames) {
        $path = Join-Path $InputRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "missing source file: $name" }
        if (-not (Safe ([IO.File]::ReadAllText($path)))) { Fail "unsafe source content: $name" }
    }
    $dataset = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'dataset-partitions.csv'))
    $hashes = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'dataset-hashes.csv'))
    $overlaps = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'dataset-overlap.csv'))
    $datasetEvidence = AssertDatasetEvidence $dataset $hashes $overlaps
    $replay = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'checkpoint-replay.csv'))
    $trajectory = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'checkpoint-trajectory.csv'))
    AssertReplayEvidence $replay
    AssertTrainingEvidence $trajectory
    $validation = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'ar-validation-metrics.csv'))
    $development = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'ar-development-metrics.csv'))
    $selection = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'selection-decisions.csv'))
    $cpu = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'cpu-smoke.csv'))
    $decision = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'decision.csv'))
    [void](AssertCpuAndSelectionEvidence $validation $development $selection $cpu $decision)
    $final = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'ar-final-holdout-metrics.csv'))
    if ($final.Count -ne 1 -or $final[0].status -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED')) { Fail 'source final holdout must remain unopened' }
    $sourceManifest = Get-Content -LiteralPath (Join-Path $InputRoot 'manifest.json') -Raw | ConvertFrom-Json
    if ($sourceManifest.schema -ne 'AR_ROLLOUT_NLL_V1' -or $sourceManifest.schema_version -ne 3 -or
        $sourceManifest.development_gate -ne 'REJECT' -or $sourceManifest.final_holdout_gate -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED') -or
        $sourceManifest.adopted_selection_mode -ne 'NONE') { Fail 'source manifest decision mismatch' }
    $legacyAnchor = AssertLegacyAnchor
    return [pscustomobject]@{
        Dataset=$datasetEvidence; Replay=$replay; Trajectory=$trajectory; Validation=$validation;
        Development=$development; Selection=$selection; Cpu=$cpu; Decision=$decision; Final=$final;
        Manifest=$sourceManifest; LegacyAnchor=$legacyAnchor
    }
}

function NewSelfTestFixture() {
    $fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ("build\autoregressive-exporter-selftest-{0}" -f ([Guid]::NewGuid().ToString('N')))))
    $buildPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build')) + '\'
    if (-not $fixtureRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) { Fail 'self-test fixture escaped build' }
    $fixtureInput = Join-Path $fixtureRoot 'input'
    $fixtureOutput = Join-Path $fixtureRoot 'output'
    [void](New-Item -ItemType Directory -Path $fixtureInput -Force)
    [void](New-Item -ItemType Directory -Path $fixtureOutput -Force)
    $publicRoot = Join-Path $repoRoot 'docs\results\qnn-htp-autoregressive-validation-2026-08'
    $fixtureNames = $copied + @('cpu-smoke.csv','decision.csv')
    foreach ($name in $fixtureNames) {
        $source = Join-Path $publicRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Fail "self-test fixture source missing: $name" }
        [IO.File]::Copy($source, (Join-Path $fixtureInput $name), $true)
    }
    $cpuPath = Join-Path $fixtureInput 'cpu-smoke.csv'
    $cpuRows = @(Import-Csv -LiteralPath $cpuPath)
    foreach ($row in $cpuRows) { $row.final_gate = 'NOT_RUN' }
    [IO.File]::WriteAllText($cpuPath, (CsvText $cpuRows), $utf8)
    $decisionPath = Join-Path $fixtureInput 'decision.csv'
    $decisionRows = @(Import-Csv -LiteralPath $decisionPath)
    foreach ($row in $decisionRows) { $row.final_holdout_gate = 'NOT_RUN' }
    [IO.File]::WriteAllText($decisionPath, (CsvText $decisionRows), $utf8)
    $publicManifest = Get-Content -LiteralPath (Join-Path $publicRoot 'manifest.json') -Raw | ConvertFrom-Json
    $sourceManifest = [ordered]@{
        schema='AR_ROLLOUT_NLL_V1'; schema_version=3; candidate_selection_mode='BEST_AR_VALIDATION_V1'; adopted_selection_mode='NONE';
        development_gate='REJECT'; final_holdout_gate='NOT_RUN'; selection_tolerance=0.0000001;
        configurations=@('T8/D16/FFN32/L19/H2','T8/D16/FFN32/L18/H2_CONTROL');
        host_checkpoint_retention='EXPERIMENTAL_ALL_23_NOT_PHASE9_NATIVE';
        legacy_replay_optimizer_metrics='NOT_AVAILABLE_CHECKPOINT_SCHEMA';
        evaluation_steps=@(0,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96,128,160,192,224,256,288,320);
        final_holdout_policy='NOT_OPENED_UNLESS_DEVELOPMENT_PASS'
    }
    [IO.File]::WriteAllText((Join-Path $fixtureInput 'manifest.json'), (($sourceManifest | ConvertTo-Json -Depth 5) + "`n"), $utf8)
    return [pscustomobject]@{ Root=$fixtureRoot; Input=$fixtureInput; Output=$fixtureOutput }
}

function ExpectSelfTestFailure([string]$Name, [scriptblock]$Action) {
    $failed = $false
    try { & $Action } catch { $failed = $true }
    if (-not $failed) { Fail "self-test negative case did not fail: $Name" }
}

function CopySafe([string]$Name) {
    $source = Join-Path $InputRoot $Name
    $text = [IO.File]::ReadAllText($source)
    if (-not (Safe $text)) { Fail "unsafe source content: $Name" }
    WriteUtf8 $Name $text
}

function WriteGatedCpuSummary() {
    $rows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'cpu-smoke.csv'))
    foreach ($row in $rows) {
        if ($row.final_gate -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED')) { Fail 'CPU summary final gate source mismatch' }
        $row.final_gate = 'NOT_RUN_GATE_REJECTED'
    }
    WriteUtf8 'cpu-smoke.csv' (CsvText $rows)
}

function WriteGatedDecision() {
    $rows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'decision.csv'))
    if ($rows.Count -ne 1 -or $rows[0].final_holdout_gate -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED')) { Fail 'decision final gate source mismatch' }
    $rows[0].final_holdout_gate = 'NOT_RUN_GATE_REJECTED'
    WriteUtf8 'decision.csv' (CsvText $rows)
}

function WriteGatedFinalHoldout() {
    $rows = @(Import-Csv -LiteralPath (Join-Path $InputRoot 'ar-final-holdout-metrics.csv'))
    if ($rows.Count -ne 1 -or $rows[0].status -notin @('NOT_RUN','NOT_RUN_GATE_REJECTED')) { Fail 'final holdout source gate mismatch' }
    $rows[0].status = 'NOT_RUN_GATE_REJECTED'
    WriteUtf8 'ar-final-holdout-metrics.csv' (CsvText $rows)
}

function GetSha256([string]$Name) {
    return (Get-FileHash -LiteralPath (Join-Path $OutputRoot $Name) -Algorithm SHA256).Hash.ToLowerInvariant()
}

function AssertBundle() {
    $entries = @(Get-ChildItem -LiteralPath $OutputRoot -Force)
    if (@($entries | Where-Object { $_.PSIsContainer }).Count -ne 0) { Fail 'public bundle must not contain subdirectories' }
    $actual = @($entries.Name | Sort-Object)
    if (($actual -join ',') -ne (($allowed | Sort-Object) -join ',')) { Fail 'public bundle allow-list mismatch' }
    foreach ($entry in $entries) {
        if (-not (Safe ([IO.File]::ReadAllText($entry.FullName)))) { Fail "unsafe generated file: $($entry.Name)" }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.result_classification -ne 'AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE' -or
        $manifest.development_gate -ne 'REJECT' -or
        $manifest.final_holdout_gate -ne 'NOT_RUN_GATE_REJECTED' -or
        $manifest.checkpoint_selection_mode -ne 'NONE' -or
        $manifest.cpu_retrain_runs -ne 4 -or $manifest.replay_available_steps -ne 56 -or
        $manifest.replay_missing_requested_steps -ne 36 -or $manifest.htp_smoke_runs -ne 0 -or
        $manifest.htp_formal_runs -ne 0) { Fail 'manifest decision mismatch' }
    foreach ($entry in $manifest.files) {
        if ((GetSha256 $entry.name) -ne $entry.sha256) { Fail "manifest hash mismatch: $($entry.name)" }
    }
    $finalRows = @(Import-Csv -LiteralPath (Join-Path $OutputRoot 'ar-final-holdout-metrics.csv'))
    if ($finalRows.Count -ne 1 -or $finalRows[0].status -ne 'NOT_RUN_GATE_REJECTED') { Fail 'final holdout is not correctly gated' }
    foreach ($name in @('htp-smoke.csv', 'formal-seeds.csv', 'thermal.csv')) {
        if (-not ([IO.File]::ReadAllText((Join-Path $OutputRoot $name)).Contains('NOT_RUN_GATE_REJECTED'))) { Fail "$name does not state its gate" }
    }
}

function NewReadme() {
    $development = $script:SourceEvidence.Development
    $format = { param($value) ([double]$value).ToString('0.######', $script:Invariant) }
    $s2Selected = @($development | Where-Object { $_.configuration_id -eq 'L19_SEED_2' -and $_.role -eq 'SELECTED' })[0]
    $s2Final = @($development | Where-Object { $_.configuration_id -eq 'L19_SEED_2' -and $_.role -eq 'FINAL_STEP' })[0]
    $controlSelected = @($development | Where-Object { $_.configuration_id -eq 'L18_SEED_2_CONTROL' -and $_.role -eq 'SELECTED' })[0]
    $controlFinal = @($development | Where-Object { $_.configuration_id -eq 'L18_SEED_2_CONTROL' -and $_.role -eq 'FINAL_STEP' })[0]
    $l19RowsSelected = @($development | Where-Object { $_.configuration_id -in @('L19_SEED_1','L19_SEED_2','L19_SEED_4') -and $_.role -eq 'SELECTED' })
    $l19RowsFinal = @($development | Where-Object { $_.configuration_id -in @('L19_SEED_1','L19_SEED_2','L19_SEED_4') -and $_.role -eq 'FINAL_STEP' })
    $l19SelectedNll = (& $format (($l19RowsSelected | ForEach-Object { [double]$_.ar_rollout_nll } | Measure-Object -Average).Average))
    $l19FinalNll = (& $format (($l19RowsFinal | ForEach-Object { [double]$_.ar_rollout_nll } | Measure-Object -Average).Average))
    $s2SelectedNll = & $format $s2Selected.ar_rollout_nll
    $s2FinalNll = & $format $s2Final.ar_rollout_nll
    $controlSelectedNll = & $format $controlSelected.ar_rollout_nll
    $controlFinalNll = & $format $controlFinal.ar_rollout_nll
    $s2SelectedExact = "$($s2Selected.token_exact)/$($s2Selected.token_total); $($s2Selected.sequence_exact)/$($s2Selected.sequence_total)"
    $s2FinalExact = "$($s2Final.token_exact)/$($s2Final.token_total); $($s2Final.sequence_exact)/$($s2Final.sequence_total)"
    $l19SelectedExact = "$(($l19RowsSelected | ForEach-Object { [int]$_.token_exact } | Measure-Object -Sum).Sum)/$(($l19RowsSelected | ForEach-Object { [int]$_.token_total } | Measure-Object -Sum).Sum); $(($l19RowsSelected | ForEach-Object { [int]$_.sequence_exact } | Measure-Object -Sum).Sum)/$(($l19RowsSelected | ForEach-Object { [int]$_.sequence_total } | Measure-Object -Sum).Sum)"
    $l19FinalExact = "$(($l19RowsFinal | ForEach-Object { [int]$_.token_exact } | Measure-Object -Sum).Sum)/$(($l19RowsFinal | ForEach-Object { [int]$_.token_total } | Measure-Object -Sum).Sum); $(($l19RowsFinal | ForEach-Object { [int]$_.sequence_exact } | Measure-Object -Sum).Sum)/$(($l19RowsFinal | ForEach-Object { [int]$_.sequence_total } | Measure-Object -Sum).Sum)"
    $controlSelectedExact = "$($controlSelected.token_exact)/$($controlSelected.token_total); $($controlSelected.sequence_exact)/$($controlSelected.sequence_total)"
    $controlFinalExact = "$($controlFinal.token_exact)/$($controlFinal.token_total); $($controlFinal.sequence_exact)/$($controlFinal.sequence_total)"
@"
# Autoregressive validation quality, August 2026

This bundle records a CPU-gated investigation of the finite, seed-dependent
autoregressive quality shortfall at T8/D16/FFN32/L19/H2. It does not claim an
HTP numerical failure: the prior evidence classifies the issue as finite
autoregressive generalization quality.

`AR_ROLLOUT_NLL_V1` starts from each initial prefix, feeds each argmax token
back into the next context, and scores the known target token at every rollout
position. The accompanying metrics record rollout NLL, token and sequence
exact counts, first-error position, recovery after an error, teacher-forced
NLL, and their gap. Checkpoint ranking is fixed before evaluation: lower
rollout NLL (tolerance 1e-7), then higher token exact, then higher sequence
exact, then earlier step.

The fixed partitions are deterministic. TRAIN has four homogeneous phase-zero
patterns. Each fresh partition has 24 mixed-prefix cases spanning four pattern
families, suffix lengths 3/4/5, and rollouts 4/8. Case IDs, initial prefixes,
and complete sequences have zero overlap across all four partitions. The
learned successor-transition overlap is structural and reported precisely in
`dataset-overlap.csv`; it is not represented as zero.

AR_VALIDATION_V3 selects a checkpoint, AR_DEVELOPMENT_V3 decides whether that
selection is predictive, and AR_FINAL_HOLDOUT_V3 remains unopened unless the
development gate passes. CPU reference regeneration ran exactly once each for
L19 seeds 1, 2, and 4 and L18 seed 2 control. Existing legacy checkpoint
replay supplied 56 of the requested 92 replay entries; 36 requested entries
were unavailable from the stored cadence. The stored state includes Adam
moments, but it does not unambiguously reconstruct historical training loss,
gradient, or update records. Those replay trajectory fields are therefore
marked NOT_AVAILABLE rather than inferred; parameter norm is available.

Selected steps were L19 seed 1: 16, seed 2: 4, seed 4: 12, and L18 control:
4. Although all development evaluations were finite, the selected checkpoints
did not meet the predeclared development gate against FINAL_STEP:

| Development comparison | Selected rollout NLL | Final rollout NLL | Selected token / sequence exact | Final token / sequence exact |
| --- | ---: | ---: | ---: | ---: |
| L19 seed 2 | $s2SelectedNll | $s2FinalNll | $s2SelectedExact | $s2FinalExact |
| L19 pooled seeds 1, 2, 4 | $l19SelectedNll | $l19FinalNll | $l19SelectedExact | $l19FinalExact |
| L18 seed 2 control | $controlSelectedNll | $controlFinalNll | $controlSelectedExact | $controlFinalExact |

The lower selected NLL did not translate into the required token or sequence
quality: seed 2 did not strictly improve, pooled L19 and control non-worsening
requirements did not hold, and the required multi-seed support was absent.
The decision is therefore `AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE`.

Consequently AR_FINAL_HOLDOUT_V3 was not opened, checkpoint selection was not
adopted, no HTP smoke or five-seed formal run started, and no thermal device
measurement was taken. All such rows use `NOT_RUN_GATE_REJECTED`; they are not
passes. No stabilizer was selected because the preregistered trajectory review
did not establish a concrete, general causal candidate. The legacy FINAL_STEP
baseline remains Oracle/Free 13/20; this bundle does not use it to select a
candidate.

The public files deliberately exclude model-state payloads, package material,
device identifiers, endpoint data, paths, and log streams. FNV-1a partition
identifiers are determinism checks, not cryptographic authenticity claims.
"@
}

$selfTestContext = $null
$selfTestPublicSnapshot = @{}
if ($SelfTest) {
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-htp-autoregressive-validation-2026-08'
    foreach ($name in $allowed) {
        $snapshotPath = Join-Path $publicSnapshotRoot $name
        if (Test-Path -LiteralPath $snapshotPath -PathType Leaf) {
            $selfTestPublicSnapshot[$name] = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash
        }
    }
    $selfTestContext = NewSelfTestFixture
    $InputRoot = $selfTestContext.Input
    $OutputRoot = $selfTestContext.Output
}
$InputRoot = RequireUnderRepository $InputRoot 'InputRoot'
$OutputRoot = RequireOutputRoot $OutputRoot
if (-not (Test-Path -LiteralPath $InputRoot -PathType Container)) { Fail 'InputRoot does not exist' }
if (-not (Test-Path -LiteralPath $OutputRoot)) { [void](New-Item -ItemType Directory -Path $OutputRoot -Force) }

RequireHeader 'dataset-partitions.csv' @('partition','case_id','domain','active_family','distractor_family','active_phase','distractor_phase','active_suffix_length','rollout_length','initial_prefix','targets')
RequireHeader 'dataset-overlap.csv' @('left','right','case_id_overlap','initial_prefix_overlap','full_sequence_overlap','unique_transition_overlap','transition_occurrence_multiset_overlap')
RequireHeader 'dataset-hashes.csv' @('partition','schema','generator_domain','hash','case_count','target_transition_occurrences','unique_target_transitions')
RequireHeader 'checkpoint-replay.csv' @('source','configuration_id','depth','seed','step','status','all_finite','ar_rollout_nll','teacher_forced_nll','teacher_forced_gap','token_exact','token_total','sequence_exact','sequence_total','mean_first_error_position','post_error_recovery_tokens','training_loss','gradient_norm','parameter_norm','update_to_parameter')
RequireHeader 'checkpoint-trajectory.csv' @('source','configuration_id','depth','seed','step','loss','accuracy','gradient_norm','parameter_norm','update_norm','update_to_parameter')
RequireHeader 'ar-validation-metrics.csv' @('source','configuration_id','depth','seed','step','role','partition','all_finite','ar_rollout_nll','teacher_forced_nll','teacher_forced_gap','token_exact','token_total','sequence_exact','sequence_total','mean_first_error_position','post_error_recovery_tokens')
RequireHeader 'ar-development-metrics.csv' @('source','configuration_id','depth','seed','step','role','partition','all_finite','ar_rollout_nll','teacher_forced_nll','teacher_forced_gap','token_exact','token_total','sequence_exact','sequence_total','mean_first_error_position','post_error_recovery_tokens')
RequireHeader 'ar-final-holdout-metrics.csv' @('source','configuration_id','depth','seed','step','role','partition','status','all_finite','ar_rollout_nll','teacher_forced_nll','teacher_forced_gap','token_exact','token_total','sequence_exact','sequence_total','mean_first_error_position','post_error_recovery_tokens')
RequireHeader 'selection-decisions.csv' @('configuration_id','depth','seed','mode','selected_step','validation_ar_nll','validation_token_exact','validation_sequence_exact','classification')
RequireHeader 'cpu-smoke.csv' @('source','configuration_id','depth','seed','selected_step','validation_ar_nll','development_gate','final_gate','classification')
RequireHeader 'decision.csv' @('development_gate','final_holdout_gate','checkpoint_selection_mode','stabilizer','replay_available','replay_missing','classification')

$script:SourceEvidence = AssertSourceEvidence
$sourceManifest = $script:SourceEvidence.Manifest
$hashes = @($script:SourceEvidence.Dataset.Hashes)
$expectedHashes = @{}
foreach ($hashRow in $hashes) { $expectedHashes[$hashRow.partition] = $hashRow.hash }
$decisions = @($script:SourceEvidence.Selection)
$cpu = @($script:SourceEvidence.Cpu)

foreach ($name in $copied) { CopySafe $name }
WriteGatedCpuSummary
WriteGatedDecision
WriteGatedFinalHoldout
WriteUtf8 'README.md' (NewReadme)
WriteUtf8 'htp-smoke.csv' (CsvText @(
    [pscustomobject][ordered]@{configuration_id='L19_SEED_1';seed='1';mode='BEST_AR_VALIDATION_V1';status='NOT_RUN_GATE_REJECTED';reason='CPU_DEVELOPMENT_GATE_REJECT'},
    [pscustomobject][ordered]@{configuration_id='L19_SEED_2';seed='2';mode='BEST_AR_VALIDATION_V1';status='NOT_RUN_GATE_REJECTED';reason='CPU_DEVELOPMENT_GATE_REJECT'},
    [pscustomobject][ordered]@{configuration_id='L19_SEED_4';seed='4';mode='BEST_AR_VALIDATION_V1';status='NOT_RUN_GATE_REJECTED';reason='CPU_DEVELOPMENT_GATE_REJECT'}
))
WriteUtf8 'formal-seeds.csv' (CsvText @(
    [pscustomobject][ordered]@{configuration='T8/D16/FFN32/L19/H2';seeds='1..5';mode='BEST_AR_VALIDATION_V1';status='NOT_RUN_GATE_REJECTED';reason='CPU_DEVELOPMENT_GATE_REJECT'}
))
WriteUtf8 'selected-step-distribution.csv' (CsvText @(
    [pscustomobject][ordered]@{source='CPU_REFERENCE_REGENERATION';configuration_id='L19_SEED_1';seed='1';selected_step='16';status='NOT_ADOPTED_DEVELOPMENT_GATE_REJECTED'},
    [pscustomobject][ordered]@{source='CPU_REFERENCE_REGENERATION';configuration_id='L19_SEED_2';seed='2';selected_step='4';status='NOT_ADOPTED_DEVELOPMENT_GATE_REJECTED'},
    [pscustomobject][ordered]@{source='CPU_REFERENCE_REGENERATION';configuration_id='L19_SEED_4';seed='4';selected_step='12';status='NOT_ADOPTED_DEVELOPMENT_GATE_REJECTED'},
    [pscustomobject][ordered]@{source='CPU_REFERENCE_REGENERATION';configuration_id='L18_SEED_2_CONTROL';seed='2';selected_step='4';status='NOT_ADOPTED_DEVELOPMENT_GATE_REJECTED'}
))
WriteUtf8 'legacy-generation.csv' (CsvText @(
    [pscustomobject][ordered]@{configuration='T8/D16/FFN32/L19/H2';mode='FINAL_STEP';oracle_exact=$script:SourceEvidence.LegacyAnchor.oracle_exact;free_exact=$script:SourceEvidence.LegacyAnchor.free_exact;role='LEGACY_BASELINE_NOT_CANDIDATE_SELECTION';status='EXISTING_CANONICAL_EVIDENCE'}
))
WriteUtf8 'thermal.csv' (CsvText @(
    [pscustomobject][ordered]@{scope='DEVICE_HTP';status='NOT_RUN_GATE_REJECTED';reason='CPU_DEVELOPMENT_GATE_REJECT'}
))

$manifestFiles = foreach ($name in ($allowed | Where-Object { $_ -ne 'manifest.json' } | Sort-Object)) {
    [ordered]@{name=$name;sha256=(GetSha256 $name)}
}
$manifest = [ordered]@{
    schema='AR_ROLLOUT_NLL_V1'; schema_version=3; result_classification='AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE'
    quality_configuration='T8/D16/FFN32/L19/H2'; development_gate='REJECT'; final_holdout_gate='NOT_RUN_GATE_REJECTED'
    checkpoint_selection_mode='NONE'; selection_tolerance=0.0000001; ar_validation_hash=$expectedHashes['AR_VALIDATION_V3']
    ar_development_hash=$expectedHashes['AR_DEVELOPMENT_V3']; ar_final_holdout_hash=$expectedHashes['AR_FINAL_HOLDOUT_V3']
    cpu_retrain_runs=4; replay_available_steps=56; replay_missing_requested_steps=36; htp_smoke_runs=0; htp_formal_runs=0
    stabilizer='NONE'; final_holdout_opened=$false; legacy_final_step_oracle_exact=$script:SourceEvidence.LegacyAnchor.oracle_exact; legacy_final_step_free_exact=$script:SourceEvidence.LegacyAnchor.free_exact
    prohibited_payloads_published=$false; files=$manifestFiles
}
WriteUtf8 'manifest.json' (($manifest | ConvertTo-Json -Depth 5) + "`n")
AssertBundle

if ($SelfTest) {
    if (-not (Safe 'ar_validation_hash=fnv1a64:aad785bd4dc88dc9')) { Fail 'safe-text false rejection' }
    foreach ($unsafe in @('C:\\local\\report.txt','/tmp/report.txt','adb -s serial shell','android_id=1','raw_checkpoint=payload','raw tensor dump','app_private_path=files/x','model.apk')) {
        if (Safe $unsafe) { Fail "unsafe self-test rejected incorrectly: $unsafe" }
    }
    try { RequireOutputRoot ([IO.Path]::GetTempPath()); Fail 'outside output-root rejection failed' } catch { if ($_.Exception.Message -match 'outside output-root rejection failed') { throw } }
    $unsafePath = Join-Path $InputRoot 'decision.csv'
    $unsafeOriginal = [IO.File]::ReadAllText($unsafePath)
    try {
        [IO.File]::WriteAllText($unsafePath, $unsafeOriginal.Replace('AUTOREGRESSIVE_VALIDATION_NOT_PREDICTIVE', 'C:\local\report'))
        ExpectSelfTestFailure 'unsafe source content' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($unsafePath, $unsafeOriginal, $utf8) }
    $schemaPath = Join-Path $InputRoot 'dataset-hashes.csv'
    $schemaOriginal = [IO.File]::ReadAllText($schemaPath)
    try {
        $lines = $schemaOriginal.Split("`n", 2)
        [IO.File]::WriteAllText($schemaPath, "bad-header`n$($lines[1])", $utf8)
        ExpectSelfTestFailure 'schema rejection' { RequireHeader 'dataset-hashes.csv' @('partition','schema','generator_domain','hash','case_count','target_transition_occurrences','unique_target_transitions') }
    } finally { [IO.File]::WriteAllText($schemaPath, $schemaOriginal, $utf8) }
    $numericPath = Join-Path $InputRoot 'ar-validation-metrics.csv'
    $numericOriginal = [IO.File]::ReadAllText($numericPath)
    try {
        $mutated = [regex]::Replace($numericOriginal, '3\.4389414079211225', 'NaN', 1)
        [IO.File]::WriteAllText($numericPath, $mutated, $utf8)
        ExpectSelfTestFailure 'numeric rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($numericPath, $numericOriginal, $utf8) }
    $bestPath = Join-Path $InputRoot 'ar-validation-metrics.csv'
    $bestOriginal = [IO.File]::ReadAllText($bestPath)
    try {
        $mutated = [regex]::Replace($bestOriginal, '"SELECTED"', '"TRAJECTORY"', 1)
        [IO.File]::WriteAllText($bestPath, $mutated, $utf8)
        ExpectSelfTestFailure 'validation best-step/role rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($bestPath, $bestOriginal, $utf8) }
    $gatePath = Join-Path $InputRoot 'decision.csv'
    $gateOriginal = [IO.File]::ReadAllText($gatePath)
    try {
        [IO.File]::WriteAllText($gatePath, $gateOriginal.Replace('"REJECT"', '"PASS"'), $utf8)
        ExpectSelfTestFailure 'development gate contradiction rejection' { AssertSourceEvidence }
    } finally { [IO.File]::WriteAllText($gatePath, $gateOriginal, $utf8) }
    $publicSnapshotRoot = Join-Path $repoRoot 'docs\results\qnn-htp-autoregressive-validation-2026-08'
    foreach ($name in $selfTestPublicSnapshot.Keys) {
        $after = (Get-FileHash -LiteralPath (Join-Path $publicSnapshotRoot $name) -Algorithm SHA256).Hash
        if ($after -ne $selfTestPublicSnapshot[$name]) { Fail "self-test modified public docs: $name" }
    }
    if ($selfTestContext -and (Test-Path -LiteralPath $selfTestContext.Root)) {
        Remove-Item -LiteralPath $selfTestContext.Root -Recurse -Force
    }
}

Write-Host "autoregressive public export: PASS ($OutputRoot)"
