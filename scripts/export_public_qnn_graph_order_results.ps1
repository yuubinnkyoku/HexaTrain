[CmdletBinding()]
param(
    [string]$InputRoot = (Join-Path $PSScriptRoot "..\build\reports\qnn-headless"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\docs\results\qnn-htp-graph-order-2026-07")
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$approvedOutputDirectory = [IO.Path]::GetFullPath((Join-Path $repositoryRoot 'docs\results\qnn-htp-graph-order-2026-07'))
if ([IO.Path]::GetFullPath($OutputDirectory) -ne $approvedOutputDirectory) {
    throw 'Public graph-order export validation failed: OutputDirectory must be the approved repository results directory'
}
$OutputDirectory = $approvedOutputDirectory

function Fail([string]$Message) { throw "Public graph-order export validation failed: $Message" }
function Require([hashtable]$Map, [string]$Key, [string]$ReportId) {
    if (-not $Map.ContainsKey($Key) -or [string]::IsNullOrWhiteSpace($Map[$Key])) {
        Fail "$ReportId is missing $Key"
    }
    return $Map[$Key]
}
function Parse-Report([string]$Path, [string]$ReportId) {
    $map = @{}
    foreach ($line in [IO.File]::ReadAllLines($Path)) {
        $index = $line.IndexOf('=')
        if ($index -le 0) { continue }
        $key = $line.Substring(0, $index)
        if ($map.ContainsKey($key)) { Fail "$ReportId duplicates $key" }
        $map[$key] = $line.Substring($index + 1)
    }
    return $map
}
function Convert-Number([string]$Value, [string]$Name) {
    if ($Value -eq 'inf') { return [double]::PositiveInfinity }
    if ($Value -eq '-inf') { return [double]::NegativeInfinity }
    $result = 0.0
    if (-not [double]::TryParse($Value, [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$result)) { Fail "invalid number $Name" }
    return $result
}
function Convert-Int([string]$Value, [string]$Name) {
    $result = 0
    if (-not [int]::TryParse($Value, [ref]$result)) { Fail "invalid integer $Name" }
    return $result
}
function Log-Gamma([double]$Value) {
    $coefficients = @(676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7)
    if ($Value -lt 0.5) { return [Math]::Log([Math]::PI) - [Math]::Log([Math]::Sin([Math]::PI * $Value)) - (Log-Gamma (1.0 - $Value)) }
    $z = $Value - 1.0; $x = 0.99999999999980993
    for ($i = 0; $i -lt $coefficients.Count; ++$i) { $x += $coefficients[$i] / ($z + $i + 1.0) }
    $t = $z + $coefficients.Count - 0.5
    return 0.5 * [Math]::Log(2.0 * [Math]::PI) + ($z + 0.5) * [Math]::Log($t) - $t + [Math]::Log($x)
}
function Beta-ContinuedFraction([double]$a, [double]$b, [double]$x) {
    $qab = $a + $b; $qap = $a + 1.0; $qam = $a - 1.0; $c = 1.0
    $d = 1.0 - $qab * $x / $qap; if ([Math]::Abs($d) -lt 3e-14) { $d = 3e-14 }; $d = 1.0 / $d; $h = $d
    for ($m = 1; $m -le 200; ++$m) {
        $m2 = 2.0 * $m; $aa = $m * ($b - $m) * $x / (($qam + $m2) * ($a + $m2))
        $d = 1.0 + $aa * $d; if ([Math]::Abs($d) -lt 3e-14) { $d = 3e-14 }; $c = 1.0 + $aa / $c; if ([Math]::Abs($c) -lt 3e-14) { $c = 3e-14 }; $d = 1.0 / $d; $h *= $d * $c
        $aa = -($a + $m) * ($qab + $m) * $x / (($a + $m2) * ($qap + $m2))
        $d = 1.0 + $aa * $d; if ([Math]::Abs($d) -lt 3e-14) { $d = 3e-14 }; $c = 1.0 + $aa / $c; if ([Math]::Abs($c) -lt 3e-14) { $c = 3e-14 }; $d = 1.0 / $d
        $delta = $d * $c; $h *= $delta; if ([Math]::Abs($delta - 1.0) -lt 3e-12) { return $h }
    }
    Fail 'incomplete beta continued fraction did not converge'
}
function Regularized-Beta([double]$x, [double]$a, [double]$b) {
    if ($x -le 0.0) { return 0.0 }; if ($x -ge 1.0) { return 1.0 }
    $front = [Math]::Exp((Log-Gamma ($a + $b)) - (Log-Gamma $a) - (Log-Gamma $b) + $a * [Math]::Log($x) + $b * [Math]::Log(1.0 - $x))
    if ($x -lt (($a + 1.0) / ($a + $b + 2.0))) { return $front * (Beta-ContinuedFraction $a $b $x) / $a }
    return 1.0 - $front * (Beta-ContinuedFraction $b $a (1.0 - $x)) / $b
}
function Inverse-Beta([double]$p, [double]$a, [double]$b) {
    $lo = 0.0; $hi = 1.0
    for ($i = 0; $i -lt 100; ++$i) { $mid = ($lo + $hi) / 2.0; if ((Regularized-Beta $mid $a $b) -lt $p) { $lo = $mid } else { $hi = $mid } }
    return ($lo + $hi) / 2.0
}
function Clopper-Pearson([int]$Successes, [int]$Total) {
    if ($Total -le 0) { Fail 'confidence interval total is zero' }
    $lower = if ($Successes -eq 0) { 0.0 } else { Inverse-Beta 0.025 $Successes ($Total - $Successes + 1) }
    $upper = if ($Successes -eq $Total) { 1.0 } else { Inverse-Beta 0.975 ($Successes + 1) ($Total - $Successes) }
    return @{ lower = $lower; upper = $upper }
}
function Validate-Frequency([string]$Value, [int]$ExpectedUnique, [string]$Name) {
    $entries = @($Value -split ',')
    if ($entries.Count -ne $ExpectedUnique) { Fail "$Name frequency count does not equal unique hash count" }
    $seen = @{}; $sum = 0
    foreach ($entry in $entries) {
        if ($entry -notmatch '^([0-9a-f]{64}):(\d+)$') { Fail "$Name has an invalid hash frequency" }
        if ($seen.ContainsKey($Matches[1])) { Fail "$Name repeats a hash frequency" }
        $seen[$Matches[1]] = $true; $sum += Convert-Int $Matches[2] "$Name frequency"
    }
    if ($sum -ne 100) { Fail "$Name frequency total is not 100" }
}

$expectedDiscovery = @(); foreach ($worker in @('w1r','w2','w3','w4','w5')) { foreach ($plan in @('abc','acb','bac','bca','cab','cba','aaa','bbb','ccc')) { $expectedDiscovery += "order-disc-$worker-$plan" } }
$expectedConfirmation = @(); foreach ($worker in @('w1','w2','w3','w4','w5')) { foreach ($plan in @('aaa','bbb','ccc')) { $expectedConfirmation += "order-conf-$worker-$plan" } }
$expected = @($expectedDiscovery + $expectedConfirmation)
if ($expected.Count -ne 60 -or ($expected | Select-Object -Unique).Count -ne 60) { Fail 'internal allowlist is not 60 unique reports' }

$commit = (& git -C (Join-Path $PSScriptRoot '..') rev-parse aa94e5a).Trim()
if ($commit -ne 'aa94e5a937bc878333305b8ea46e129342390d03') { Fail 'milestone aa94e5a did not resolve to the approved experiment source' }
$hashes = @{
    one_hot = 'd85d7d14ab07879ab62b29dc0be5eef0c51d29db0a1e6050b8d7ccb080bd00f1'
    target = 'f1c1a960169be212ee9f4b5856b9add5b9f2dd5ff68b77ce962292d8e1c724cb'
    parameter = '5674c9ecf8bcb785a4db27a73afb11e33aa22c23670301fbe7865692fa83b93b'
}
$plans = @{
    abc=@('full_dinput_dembedding',1,@('full','stop_after_dinput','stop_after_dembedding'))
    acb=@('full_dembedding_dinput',2,@('full','stop_after_dembedding','stop_after_dinput'))
    bac=@('dinput_full_dembedding',3,@('stop_after_dinput','full','stop_after_dembedding'))
    bca=@('dinput_dembedding_full',4,@('stop_after_dinput','stop_after_dembedding','full'))
    cab=@('dembedding_full_dinput',5,@('stop_after_dembedding','full','stop_after_dinput'))
    cba=@('dembedding_dinput_full',6,@('stop_after_dembedding','stop_after_dinput','full'))
    aaa=@('full_full_full',7,@('full','full','full'))
    bbb=@('dinput_dinput_dinput',8,@('stop_after_dinput','stop_after_dinput','stop_after_dinput'))
    ccc=@('dembedding_dembedding_dembedding',9,@('stop_after_dembedding','stop_after_dembedding','stop_after_dembedding'))
}
$variantExpected = @{
    full=@{ node=123; tensor=153; input=15; output=31; audited=31; elements=7552; boundary='lm_output_projection_next' }
    stop_after_dinput=@{ node=98; tensor=153; input=14; output=18; audited=18; elements=3904; boundary='lm_dinput' }
    stop_after_dembedding=@{ node=99; tensor=153; input=14; output=19; audited=19; elements=4416; boundary='lm_dembedding' }
}

$processRows = @(); $slotRows = @(); $auditRows = @(); $graphRows = @()
foreach ($reportId in $expected) {
    $path = Join-Path (Join-Path $InputRoot $reportId) 'device-report.txt'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "allowlisted report is missing: $reportId" }
    $activityPath = Join-Path (Split-Path -Parent $path) 'activity-sampling.json'
    if (-not (Test-Path -LiteralPath $activityPath -PathType Leaf)) { Fail "$reportId activity sampling is missing" }
    $activity = Get-Content -LiteralPath $activityPath -Raw | ConvertFrom-Json
    if ($activity.phonelm_became_top_activity_count -ne 0 -or $activity.focus_takeover_count -ne 0 -or $activity.foreground_preserved -ne $true) { Fail "$reportId activity/focus invariant failed" }
    $m = Parse-Report $path $reportId
    foreach ($key in @('test','slot_count','selected_plan','selected_plan_index','snapshot_E_one_hot_canonical_hash','snapshot_E_target_canonical_hash','snapshot_E_current_parameter_canonical_hash','all_slots_attempted','all_slots_success','status','error','backend_requested','cpu_fallback','compile_time_qairt_build_id')) { [void](Require $m $key $reportId) }
    if ($m.test -ne 'fixed_state_graph_order_orthogonalization' -or $m.slot_count -ne '3' -or $m.all_slots_attempted -ne 'true' -or $m.backend_requested -ne 'HTP' -or $m.cpu_fallback -ne 'false' -or $m.compile_time_qairt_build_id -ne '2.48.40.260702151143') { Fail "$reportId has an invalid process manifest" }
    if ($m.snapshot_E_one_hot_canonical_hash -ne $hashes.one_hot -or $m.snapshot_E_target_canonical_hash -ne $hashes.target -or $m.snapshot_E_current_parameter_canonical_hash -ne $hashes.parameter) { Fail "$reportId fixed checkpoint hash mismatch" }
    $planToken = ($reportId -split '-')[-1]; if (-not $plans.ContainsKey($planToken)) { Fail "$reportId plan is not allowlisted" }
    $plan = $plans[$planToken]
    if ($m.selected_plan -ne $plan[0] -or (Convert-Int $m.selected_plan_index "$reportId selected plan") -ne $plan[1]) { Fail "$reportId selected plan mismatch" }
    $processNonfinite = 0; $processVarying = $false; $slotCount = 0
    for ($position=1; $position -le 3; ++$position) {
        $prefix = "order_slot_$($plan[1])_$($plan[0])_position_$position"
        foreach ($suffix in @('plan','plan_index','position','variant','node_boundary','source_graph_add_node_success_count','source_tensor_create_success_count','backend_finalized_node_count','backend_finalized_tensor_count','actual_qnn_input_tensor_count','actual_qnn_output_tensor_count','audited_app_read_tensor_count','app_read_output_element_count','first_changing_tensor','first_changing_candidate_node','first_changing_run','global_repeat_max_abs_difference','global_repeat_mean_abs_difference','qnn_execute_attempts','qnn_execute_successes','backend_create_result','device_create_result','context_create_result','graph_create_result','graph_finalize_result','graph_execute_result','graph_execute_qnn_result','graph_create_count','graph_finalize_count','graph_execute_count','caller_owned_app_write_hashes_unchanged','learning_rate_app_write_bytes_unchanged','app_write_hashes_unchanged','app_read_poison_residual_elements','nonfinite_elements','slot_success')) { [void](Require $m "$prefix`_$suffix" $reportId) }
        foreach ($kind in @('one_hot','target','parameter')) { foreach ($encoding in @('raw','canonical')) { if ((Require $m "$prefix`_fixed_$kind`_$encoding`_hash" $reportId) -ne $hashes[$kind]) { Fail "$reportId position $position fixed $kind $encoding hash mismatch" } } }
        if ((Convert-Int $m["$prefix`_position"] "$reportId position") -ne $position -or $m["$prefix`_plan"] -ne $plan[0] -or (Convert-Int $m["$prefix`_plan_index"] "$reportId plan index") -ne $plan[1] -or $m["$prefix`_variant"] -ne $plan[2][$position-1]) { Fail "$reportId position $position structure mismatch" }
        $expectedStructure = $variantExpected[$m["$prefix`_variant"]]
        if ($null -eq $expectedStructure -or (Convert-Int $m["$prefix`_source_graph_add_node_success_count"] "$reportId node count") -ne $expectedStructure.node -or (Convert-Int $m["$prefix`_source_tensor_create_success_count"] "$reportId tensor count") -ne $expectedStructure.tensor -or (Convert-Int $m["$prefix`_actual_qnn_input_tensor_count"] "$reportId input count") -ne $expectedStructure.input -or (Convert-Int $m["$prefix`_actual_qnn_output_tensor_count"] "$reportId output count") -ne $expectedStructure.output -or (Convert-Int $m["$prefix`_audited_app_read_tensor_count"] "$reportId audited count") -ne $expectedStructure.audited -or (Convert-Int $m["$prefix`_app_read_output_element_count"] "$reportId output elements") -ne $expectedStructure.elements -or $m["$prefix`_node_boundary"] -ne $expectedStructure.boundary) { Fail "$reportId position $position variant structure invariant failed" }
        if ($m["$prefix`_backend_finalized_node_count"] -ne 'UNAVAILABLE' -or $m["$prefix`_backend_finalized_tensor_count"] -ne 'UNAVAILABLE') { Fail "$reportId position $position backend finalized count must be UNAVAILABLE" }
        foreach ($result in @('backend_create_result','device_create_result','context_create_result','graph_create_result','graph_finalize_result','graph_execute_result','graph_execute_qnn_result')) { if ((Convert-Int $m["$prefix`_$result"] "$reportId $result") -ne 0) { Fail "$reportId position $position QNN result $result is nonzero" } }
        if ((Convert-Int $m["$prefix`_qnn_execute_attempts"] "$reportId attempts") -ne 100 -or (Convert-Int $m["$prefix`_qnn_execute_successes"] "$reportId successes") -ne 100 -or (Convert-Int $m["$prefix`_graph_create_count"] "$reportId graph create count") -ne 1 -or (Convert-Int $m["$prefix`_graph_finalize_count"] "$reportId graph finalize count") -ne 1 -or (Convert-Int $m["$prefix`_graph_execute_count"] "$reportId execute count") -ne 100 -or $m["$prefix`_caller_owned_app_write_hashes_unchanged"] -ne 'true' -or $m["$prefix`_learning_rate_app_write_bytes_unchanged"] -ne 'true' -or $m["$prefix`_app_write_hashes_unchanged"] -ne 'true' -or (Convert-Int $m["$prefix`_app_read_poison_residual_elements"] "$reportId poison") -ne 0) { Fail "$reportId position $position execution invariant failed" }
        $dinput = "$prefix`_embedding_input_gradient"
        foreach ($suffix in @('unique_raw_hashes','unique_canonical_hashes','raw_hash_frequencies','canonical_hash_frequencies','first_different_run','first_repeat_different_index','repeat_max_abs_difference','repeat_mean_abs_difference','nonfinite_elements','app_read_poison_residual_elements')) { [void](Require $m "$dinput`_$suffix" $reportId) }
        $dinputRawUnique = Convert-Int $m["$dinput`_unique_raw_hashes"] "$reportId DINPUT raw unique"; $dinputCanonicalUnique = Convert-Int $m["$dinput`_unique_canonical_hashes"] "$reportId DINPUT canonical unique"
        if ($dinputRawUnique -lt 1 -or $dinputRawUnique -gt 100 -or $dinputCanonicalUnique -lt 1 -or $dinputCanonicalUnique -gt 100) { Fail "$reportId DINPUT unique count is outside 1..100" }
        Validate-Frequency $m["$dinput`_raw_hash_frequencies"] $dinputRawUnique "$reportId DINPUT raw"
        Validate-Frequency $m["$dinput`_canonical_hash_frequencies"] $dinputCanonicalUnique "$reportId DINPUT canonical"
        $firstChangingRun = Convert-Int $m["$prefix`_first_changing_run"] "$reportId first changing run"
        $dinputFirstDifferentRun = Convert-Int $m["$dinput`_first_different_run"] "$reportId DINPUT first different run"
        $dinputFirstRepeatDifferentIndex = Convert-Int $m["$dinput`_first_repeat_different_index"] "$reportId DINPUT first repeat different index"
        $globalMaxDifference = Convert-Number $m["$prefix`_global_repeat_max_abs_difference"] "$reportId max diff"
        [void](Convert-Number $m["$prefix`_global_repeat_mean_abs_difference"] "$reportId mean diff")
        [void](Convert-Number $m["$dinput`_repeat_max_abs_difference"] "$reportId DINPUT max diff")
        [void](Convert-Number $m["$dinput`_repeat_mean_abs_difference"] "$reportId DINPUT mean diff")
        $nonfinite = Convert-Int $m["$prefix`_nonfinite_elements"] "$reportId nonfinite"
        if ($nonfinite -lt 0) { Fail "$reportId position $position has a negative nonfinite count" }
        $processNonfinite += $nonfinite
        $varying = ($globalMaxDifference -gt 0.0) -or $nonfinite -gt 0 -or $m["$prefix`_first_changing_tensor"] -ne 'NONE'
        if ($varying) {
            if ($m["$prefix`_first_changing_tensor"] -ne 'gradient_gamma1' -or $m["$prefix`_first_changing_candidate_node"] -ne 'UNMAPPED' -or $dinputCanonicalUnique -le 1 -or $firstChangingRun -lt 0 -or $firstChangingRun -gt 99 -or $dinputFirstDifferentRun -lt 0 -or $dinputFirstDifferentRun -gt 99 -or $dinputFirstRepeatDifferentIndex -lt 0 -or $dinputFirstRepeatDifferentIndex -ge $expectedStructure.elements) { Fail "$reportId position $position varying audit invariant failed" }
        } elseif ($m["$prefix`_first_changing_tensor"] -ne 'NONE' -or $m["$prefix`_first_changing_candidate_node"] -ne 'NONE' -or $dinputCanonicalUnique -ne 1 -or $firstChangingRun -ne -1 -or $dinputFirstDifferentRun -ne -1 -or $dinputFirstRepeatDifferentIndex -ne -1) { Fail "$reportId position $position stable audit invariant failed" }
        if (($nonfinite -gt 0 -and $m["$prefix`_slot_success"] -ne 'false') -or ($nonfinite -eq 0 -and $m["$prefix`_slot_success"] -ne 'true')) { Fail "$reportId position $position slot success invariant failed" }
        $processVarying = $processVarying -or $varying
        $slotRows += [pscustomobject]@{ report_id=$reportId; cohort=if($reportId -like 'order-conf-*'){'confirmation'}else{'discovery'}; plan=$plan[0]; plan_index=$plan[1]; position=$position; variant=$m["$prefix`_variant"]; node_boundary=$m["$prefix`_node_boundary"]; source_graph_add_node_success_count=$m["$prefix`_source_graph_add_node_success_count"]; source_tensor_create_success_count=$m["$prefix`_source_tensor_create_success_count"]; backend_finalized_node_count=$m["$prefix`_backend_finalized_node_count"]; backend_finalized_tensor_count=$m["$prefix`_backend_finalized_tensor_count"]; actual_qnn_input_tensor_count=$m["$prefix`_actual_qnn_input_tensor_count"]; actual_qnn_output_tensor_count=$m["$prefix`_actual_qnn_output_tensor_count"]; audited_app_read_tensor_count=$m["$prefix`_audited_app_read_tensor_count"]; app_read_output_element_count=$m["$prefix`_app_read_output_element_count"]; first_changing_tensor=$m["$prefix`_first_changing_tensor"]; first_changing_candidate_node=$m["$prefix`_first_changing_candidate_node"]; first_changing_run=$m["$prefix`_first_changing_run"]; global_repeat_max_abs_difference=$m["$prefix`_global_repeat_max_abs_difference"]; global_repeat_mean_abs_difference=$m["$prefix`_global_repeat_mean_abs_difference"]; varying=$varying; nonfinite_elements=$nonfinite; backend_create_result=$m["$prefix`_backend_create_result"]; device_create_result=$m["$prefix`_device_create_result"]; context_create_result=$m["$prefix`_context_create_result"]; graph_create_result=$m["$prefix`_graph_create_result"]; graph_finalize_result=$m["$prefix`_graph_finalize_result"]; graph_execute_result=$m["$prefix`_graph_execute_result"]; graph_execute_count=$m["$prefix`_graph_execute_count"]; slot_success=$m["$prefix`_slot_success"] }
        $auditRows += [pscustomobject]@{ report_id=$reportId; plan=$plan[0]; position=$position; variant=$m["$prefix`_variant"]; dinput_unique_raw_hashes=$m["$dinput`_unique_raw_hashes"]; dinput_unique_canonical_hashes=$m["$dinput`_unique_canonical_hashes"]; dinput_raw_hash_frequencies=$m["$dinput`_raw_hash_frequencies"]; dinput_canonical_hash_frequencies=$m["$dinput`_canonical_hash_frequencies"]; dinput_first_different_run=$m["$dinput`_first_different_run"]; dinput_first_repeat_different_index=$m["$dinput`_first_repeat_different_index"]; dinput_repeat_max_abs_difference=$m["$dinput`_repeat_max_abs_difference"]; dinput_repeat_mean_abs_difference=$m["$dinput`_repeat_mean_abs_difference"]; nonfinite_elements=$nonfinite; poison_residual_elements=$m["$prefix`_app_read_poison_residual_elements"]; app_write_unchanged=$m["$prefix`_app_write_hashes_unchanged"] }
        $slotCount++
    }
    if ($slotCount -ne 3) { Fail "$reportId did not yield three slots" }
    if (($processNonfinite -gt 0 -and ($m.status -ne 'FAILED' -or $m.all_slots_success -ne 'false' -or $m.error -ne 'graph order slot invariant failed')) -or ($processNonfinite -eq 0 -and ($m.status -ne 'SUCCESS' -or $m.all_slots_success -ne 'true' -or $m.error -ne 'none'))) { Fail "$reportId process status invariant failed" }
    $processRows += [pscustomobject]@{ report_id=$reportId; cohort=if($reportId -like 'order-conf-*'){'confirmation'}else{'discovery'}; plan=$plan[0]; plan_index=$plan[1]; process_varying=$processVarying; nonfinite_elements=$processNonfinite; qnn_failures=0; poison_failures=0; app_write_failures=0; source_commit=$commit }
}

if ($processRows.Count -ne 60 -or $slotRows.Count -ne 180) { Fail 'public row count mismatch' }
$confirmationHomogeneous = @($processRows | Where-Object { $_.cohort -eq 'confirmation' })
$allHomogeneous = @($processRows | Where-Object { $_.plan_index -ge 7 })
$homogeneousSlots = @($slotRows | Where-Object { $_.plan_index -ge 7 })
$variantStats = @($allHomogeneous | Group-Object plan | ForEach-Object {
    $variant = switch ($_.Name) { 'full_full_full' { 'full' }; 'dinput_dinput_dinput' { 'stop_after_dinput' }; 'dembedding_dembedding_dembedding' { 'stop_after_dembedding' }; default { Fail 'unexpected homogeneous plan' } }
    [pscustomobject]@{ variant=$variant; varying=@($_.Group | Where-Object process_varying).Count; total=$_.Count }
})
$positionStats = @($homogeneousSlots | Group-Object position | ForEach-Object { [pscustomobject]@{ position=[int]$_.Name; varying=@($_.Group | Where-Object varying).Count; total=$_.Count } })
$allVarying = @($slotRows | Where-Object varying).Count; $nonfiniteSlots = @($slotRows | Where-Object { $_.nonfinite_elements -gt 0 }).Count; $nonfiniteElements = ($slotRows | Measure-Object nonfinite_elements -Sum).Sum
if (@($confirmationHomogeneous | Where-Object process_varying).Count -ne 3 -or @($allHomogeneous | Where-Object process_varying).Count -ne 8 -or $allVarying -ne 21 -or $nonfiniteSlots -ne 12) { Fail 'derived public aggregate does not match the approved completed experiment' }
foreach ($expectedStat in @(@('full',2),@('stop_after_dinput',3),@('stop_after_dembedding',3))) { $actual=$variantStats | Where-Object variant -eq $expectedStat[0]; if ($actual.varying -ne $expectedStat[1] -or $actual.total -ne 10) { Fail 'variant aggregate mismatch' } }
foreach ($expectedStat in @(@(1,2),@(2,4),@(3,3))) { $actual=$positionStats | Where-Object position -eq $expectedStat[0]; if ($actual.varying -ne $expectedStat[1] -or $actual.total -ne 30) { Fail 'position aggregate mismatch' } }
$confirmationPatterns = [ordered]@{}; $combinedPatterns = [ordered]@{}
foreach ($group in @($homogeneousSlots | Group-Object report_id)) {
    $pattern = (($group.Group | Sort-Object position | ForEach-Object { if ($_.varying) { '1' } else { '0' } }) -join '')
    if ($group.Group[0].cohort -eq 'confirmation') { if (-not $confirmationPatterns.Contains($pattern)) { $confirmationPatterns[$pattern]=0 }; $confirmationPatterns[$pattern]++ }
    if (-not $combinedPatterns.Contains($pattern)) { $combinedPatterns[$pattern]=0 }; $combinedPatterns[$pattern]++
}
if ($confirmationPatterns['000'] -ne 12 -or $confirmationPatterns['010'] -ne 2 -or $confirmationPatterns['001'] -ne 1 -or $combinedPatterns['000'] -ne 22 -or $combinedPatterns['100'] -ne 2 -or $combinedPatterns['010'] -ne 3 -or $combinedPatterns['001'] -ne 2 -or $combinedPatterns['011'] -ne 1) { Fail 'homogeneous pattern aggregate mismatch' }
$firstExternalAuditCount = @($slotRows | Where-Object { $_.varying -and $_.first_changing_tensor -eq 'gradient_gamma1' }).Count
$dinputVaryingCount = @($auditRows | Where-Object { (Convert-Int $_.dinput_unique_canonical_hashes 'DINPUT aggregate unique') -gt 1 }).Count
if ($firstExternalAuditCount -ne 21 -or $dinputVaryingCount -ne 21) { Fail 'first external audit aggregate mismatch' }
$discoveryPlanExpected = [ordered]@{ full_dinput_dembedding=3; full_dembedding_dinput=2; dinput_full_dembedding=0; dinput_dembedding_full=1; dembedding_full_dinput=3; dembedding_dinput_full=1; full_full_full=1; dinput_dinput_dinput=2; dembedding_dembedding_dembedding=2 }
$discoveryPlanVarying = @($processRows | Where-Object { $_.cohort -eq 'discovery' } | Group-Object plan | ForEach-Object { [pscustomobject]@{ plan=$_.Name; varying=@($_.Group | Where-Object process_varying).Count; total=$_.Count } })
foreach ($planName in $discoveryPlanExpected.Keys) { $actual = $discoveryPlanVarying | Where-Object plan -eq $planName; if ($null -eq $actual -or $actual.total -ne 5 -or $actual.varying -ne $discoveryPlanExpected[$planName]) { Fail "discovery plan aggregate mismatch for $planName" } }

$discoveryVarying = @($processRows | Where-Object { $_.cohort -eq 'discovery' -and $_.process_varying }).Count
$confirmationVarying = @($confirmationHomogeneous | Where-Object process_varying).Count
$homogeneousVarying = @($allHomogeneous | Where-Object process_varying).Count
$varyingProcessCount = @($processRows | Where-Object process_varying).Count
$nonfiniteProcessCount = @($processRows | Where-Object { $_.nonfinite_elements -gt 0 }).Count
$discoveryCi = Clopper-Pearson $discoveryVarying 45; $confirmationCi = Clopper-Pearson $confirmationVarying 15
$homogeneousCi = Clopper-Pearson $homogeneousVarying 30; $ci = Clopper-Pearson $varyingProcessCount $processRows.Count; $nonfiniteCi = Clopper-Pearson $nonfiniteProcessCount 60
$summary = [ordered]@{ classification='FRESH_RUNTIME_CONTEXT_INSTANCE_ASSOCIATED_EXECUTION_VARIABILITY'; source_commit=$commit; environment=@{ qairt_build='2.48.40.260702151143'; qnn_api='2.37'; htp='V81'; android='16'; android_security_patch='2026-05-01'; selinux='Enforcing' }; model=@{ shape='B1_T8_V32_D16'; dtype='FP32' }; fixed_canonical_hashes=$hashes; input_report_count=60; discovery_processes=45; confirmation_processes=15; slots=180; qnn_execute_attempts=18000; qnn_execute_successes=18000; varying_slots=$allVarying; nonfinite_slots=$nonfiniteSlots; nonfinite_elements=$nonfiniteElements; qnn_failures=0; poison_failures=0; app_write_failures=0; activity_failures=0; focus_failures=0; first_external_audit_tensor=@{ name='gradient_gamma1'; varying_slots=$firstExternalAuditCount; candidate_node='UNMAPPED'; limitation='audit_order_tie' }; dinput_varying_slots=$dinputVaryingCount; discovery_plan_varying_processes=$discoveryPlanVarying; discovery_any_process_clopper_pearson_95=@{ varying=$discoveryVarying; total=45; lower=$discoveryCi.lower; upper=$discoveryCi.upper; method='exact_binomial_Clopper_Pearson' }; confirmation_homogeneous_process_discordance=@{ varying=3; total=15; clopper_pearson_95=@{ lower=$confirmationCi.lower; upper=$confirmationCi.upper }; patterns=$confirmationPatterns }; combined_homogeneous_process_discordance=@{ varying=8; total=30; clopper_pearson_95=@{ lower=$homogeneousCi.lower; upper=$homogeneousCi.upper }; patterns=$combinedPatterns }; homogeneous_variant_varying=$variantStats; homogeneous_position_varying=$positionStats; all_process_clopper_pearson_95=@{ varying=$varyingProcessCount; total=60; lower=$ci.lower; upper=$ci.upper; method='exact_binomial_Clopper_Pearson' }; nonfinite_process_clopper_pearson_95=@{ varying=$nonfiniteProcessCount; total=60; lower=$nonfiniteCi.lower; upper=$nonfiniteCi.upper; method='exact_binomial_Clopper_Pearson' }; interpretation=@{ measured='Variation occurred across all three graph variants and across all three positions; all QNN result codes, poison checks, APP_WRITE checks, and activity/focus checks passed.'; inference='The observed effect is not graph-variant-specific. The design does not require a fixed position or a particular preceding graph to observe it. The jointly-created Runtime/backend/device/context/graph instance scopes remain confounded.' } }

$stagingRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$stagingDirectory = Join-Path $stagingRoot ("phonelm-qnn-graph-order-export-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stagingDirectory | Out-Null
try {
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stagingDirectory 'summary.json') -Encoding utf8
    $processRows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $stagingDirectory 'process-results.csv') -Encoding utf8
    $slotRows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $stagingDirectory 'slot-results.csv') -Encoding utf8
    ($slotRows | Group-Object variant | ForEach-Object { $row=$_.Group[0]; [pscustomobject]@{ variant=$_.Name; source_graph_add_node_success_count=$row.source_graph_add_node_success_count; source_tensor_create_success_count=$row.source_tensor_create_success_count; backend_finalized_node_count=$row.backend_finalized_node_count; backend_finalized_tensor_count=$row.backend_finalized_tensor_count; actual_qnn_input_tensor_count=$row.actual_qnn_input_tensor_count; actual_qnn_output_tensor_count=$row.actual_qnn_output_tensor_count; audited_app_read_tensor_count=$row.audited_app_read_tensor_count; app_read_output_element_count=$row.app_read_output_element_count } }) | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $stagingDirectory 'graph-structures.csv') -Encoding utf8
    $auditRows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $stagingDirectory 'numerical-audit.csv') -Encoding utf8
$readme = @"
# QNN HTP fixed-state runtime/context order study

This public aggregate contains 60 allow-listed fresh-process reports: 45 discovery reports and 15 confirmation reports. Each process selected one three-slot plan; every slot created a fresh Runtime/context and executed 100 fixed-state repetitions.

The observational classification is `FRESH_RUNTIME_CONTEXT_INSTANCE_ASSOCIATED_EXECUTION_VARIABILITY`. The completed data show 21 varying slots of 180 and 12 slots with nonfinite APP_READ elements. All recorded QNN create/finalize/execute results were zero; APP_READ poison residuals and APP_WRITE integrity failures were zero.

Confirmation homogeneous-plan process discordance was 3/15; across all homogeneous plans it was 8/30. Homogeneous process-discordance counts were full 2/10, stop-after-dinput 3/10, and stop-after-dembedding 3/10. Homogeneous-position varying-slot counts were 2/30, 4/30, and 3/30. Exact process-level Clopper-Pearson intervals are recorded in summary.json.

The first external audit tensor was gradient_gamma1 in all 21 varying slots; DINPUT also varied in all 21. The candidate node is UNMAPPED because audit order is a tie-breaking limitation, not a node-level localization result.

Measured result: variation is present across variants and positions, including discordant instances inside homogeneous plans. Inference: this design does not support a graph-variant-specific cause, a fixed required position, or a required preceding graph. Runtime/backend/device/context/graph construction is simultaneous in every slot, so those instance scopes remain confounded.

process-results.csv report_id is an experiment identifier, not an operating-system process identifier. Files contain aggregates only. No raw tensor values, device endpoints, paths, executable artifacts, operating-system process identifiers, timestamps, or raw diagnostics are published. Source milestone: aa94e5a.
"@
    Set-Content -LiteralPath (Join-Path $stagingDirectory 'README.md') -Value $readme -Encoding utf8

    $expectedOutputNames = @('README.md','summary.json','process-results.csv','slot-results.csv','graph-structures.csv','numerical-audit.csv')
    $stagedOutputNames = @(Get-ChildItem -LiteralPath $stagingDirectory -File | ForEach-Object Name)
    if ((Compare-Object ($expectedOutputNames | Sort-Object) ($stagedOutputNames | Sort-Object))) { Fail 'staged public output file set is not exact' }
    $deny = '(?im)([a-z]:\\|/Users/|/home/|\\\\|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b|@[A-Za-z0-9.-]+\.|\bpid\b|\bapk\b|\.(?:so|aab|jks|keystore|elf|bin)\b|logcat|BEGIN [A-Z ]*PRIVATE KEY|(?:password|passwd|secret[_-]?key|access[_-]?token|authorization:|bearer\s+[A-Za-z0-9._-]+|sk-[A-Za-z0-9_-]{16,}))'
    $publicText = (Get-ChildItem -LiteralPath $stagingDirectory -File | Get-Content -Raw) -join "`n"
    if ($publicText -match $deny) { Fail 'staged public output matched the denylist' }

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $existingOutputNames = @(Get-ChildItem -LiteralPath $OutputDirectory -File | ForEach-Object Name)
    if ($existingOutputNames.Count -gt 0 -and (Compare-Object ($expectedOutputNames | Sort-Object) ($existingOutputNames | Sort-Object))) { Fail 'existing public output file set is not exact' }
    foreach ($name in $expectedOutputNames) {
        Copy-Item -LiteralPath (Join-Path $stagingDirectory $name) -Destination (Join-Path $OutputDirectory $name) -Force
    }
} finally {
    if (Test-Path -LiteralPath $stagingDirectory) {
        $resolvedStagingDirectory = [IO.Path]::GetFullPath($stagingDirectory)
        if (-not $resolvedStagingDirectory.StartsWith($stagingRoot, [StringComparison]::OrdinalIgnoreCase) -or
                -not (Split-Path -Leaf $resolvedStagingDirectory).StartsWith('phonelm-qnn-graph-order-export-')) {
            throw 'Refusing unsafe staging cleanup target'
        }
        Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
    }
}
Write-Output "public_qnn_graph_order_export=PASS"
