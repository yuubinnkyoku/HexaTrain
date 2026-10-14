param(
 [Parameter(Mandatory=$true)][string[]]$LaneAReports,
 [Parameter(Mandatory=$true)][string[]]$LaneBReports,
 [Parameter(Mandatory=$true)][string]$SourceSha,
 [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference='Stop'
if($SourceSha -notmatch '^[0-9a-f]{40}$'){throw 'EXACT_SOURCE_REQUIRED'}
if($LaneAReports.Count -ne 3 -or $LaneBReports.Count -ne 3){throw 'THREE_PROCESSES_PER_LANE_REQUIRED'}
if(@(@($LaneAReports)+@($LaneBReports)|Select-Object -Unique).Count -ne 6){throw 'INDEPENDENT_PROCESS_PATHS_REQUIRED'}
$metricPattern='^onemetric,([A-Za-z0-9_]+),(\d+),([A-Za-z0-9_]+),abs=([^,]+),rms=([^,]+),relative_l2=([^,]+),max_relative=([^,]+),fp_abs=([^,]+),nonfinite=(\d+),hash=([0-9a-f]{16}),limit=([^,]+),gate=([01])$'
function Number([string]$Text){$n=[double]::Parse($Text,[Globalization.CultureInfo]::InvariantCulture);if([double]::IsNaN($n)-or[double]::IsInfinity($n)-or$n -lt 0){throw 'NONFINITE_OR_NEGATIVE_METRIC'};return $n}
function Digest([string[]]$Lines){$bytes=[Text.Encoding]::UTF8.GetBytes(($Lines -join "`n")+"`n");return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()}
$identity=$null;$ln1Identity=$null;$lanes=@()
foreach($lane in @(@{name='oneBroadcast';reports=$LaneAReports;metrics=1840;executes=206},@{name='oneUpstream';reports=$LaneBReports;metrics=5225;executes=825})){
 $digests=@();$processes=@();$firstMetrics=$null
 foreach($report in $lane.reports){
  $ledger=Get-Content (Join-Path $report 'identity-private.json') -Raw|ConvertFrom-Json
  if($ledger.source_sha -ne $SourceSha -or $ledger.build.source_sha -ne $SourceSha -or $ledger.variant -ne $lane.name -or $ledger.branch -ne 'investigation/g1-fp32-broadcast-upstream-frontier' -or $ledger.exit_code -ne 0 -or $ledger.focus_takeover_count -ne 0 -or $ledger.protected_identity -ne 'PASS' -or $ledger.mode -ne 'BACKGROUND_CORRECTNESS'){throw 'PROCESS_PROVENANCE_OR_SAFETY_FAIL'}
  if($ledger.build.qairt_build_id -ne '2.48.40.260702151143' -or $ledger.build.architecture -ne 'v81' -or $ledger.build.fp_contract -ne 'off' -or $ledger.build.vectorization -ne 'disabled'){throw 'FIXED_BUILD_REQUIRED'}
  $selected=[ordered]@{source_sha=$SourceSha;branch=$ledger.branch;build_id=$ledger.build.qairt_build_id}
  foreach($compiler in @('hexagon_compiler_sha256','ndk_compiler_sha256')){if($ledger.build.$compiler -notmatch '^[0-9a-f]{64}$'){throw 'COMPILER_IDENTITY_REQUIRED'};$selected[$compiler]=$ledger.build.$compiler}
  if($lane.name -eq 'oneUpstream'){$hash=$ledger.hashes.'ln1-parameters.bin';if($hash -notmatch '^[0-9a-f]{64}$'){throw 'LN1_FIXTURE_HASH_REQUIRED'};if($null -eq $ln1Identity){$ln1Identity=$hash}elseif($ln1Identity -ne $hash){throw 'LN1_FIXTURE_COHORT_MISMATCH'}}
  foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe','libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so','libQnnHtpV81Skel.so','one-layer-inputs.bin')){
   $hash=$ledger.hashes.$name;if($hash -notmatch '^[0-9a-f]{64}$'){throw 'MISSING_STAGED_HASH'}
   if($name -in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe') -and $ledger.build.binary_hashes.$name -ne $hash){throw 'STAGED_BUILD_HASH_MISMATCH'}
   $selected[$name]=$hash
  }
  $canonical=$selected|ConvertTo-Json -Compress
  if($null -eq $identity){$identity=$canonical}elseif($identity -ne $canonical){throw 'COHORT_BINARY_OR_FIXTURE_MISMATCH'}
  $lines=Get-Content (Join-Path $report 'stdout.txt');if(@($lines|Where-Object {$_ -match '^api,[^,]+,[1-9]'}).Count -or @($lines|Where-Object {$_ -match '^[A-Z_]*(FAIL|NONFINITE|STOP)'}).Count){throw 'QNN_OR_DIAGNOSTIC_FAILURE'}
  $summary=@($lines|Where-Object {$_ -match ('^summary,'+$lane.name+',')});if($summary.Count -ne 1 -or $summary[0] -notmatch 'cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0$'){throw 'COMPLETE_SAFE_SUMMARY_REQUIRED'}
  if($lane.name -eq 'oneBroadcast' -and $summary[0] -notmatch 'standalone_failures=0,segment_failures=0,'){throw 'CUSTOM_ACCURACY_NOT_ESTABLISHED'}
  if($lane.name -eq 'oneUpstream' -and $summary[0] -notmatch 'local_custom_failures=0,transfer_words_equal=1,'){throw 'UPSTREAM_LOCAL_CUSTOM_FAIL'}
  $parsed=@();$keys=@{};foreach($line in $lines){if($line -notmatch '^onemetric,'){continue};if($line -notmatch $metricPattern){throw 'METRIC_SCHEMA_FAIL'};$m=$Matches.Clone();$abs=Number $m[4];$limit=Number $m[11];$pass=[int]$m[12];if([int]$m[9] -ne 0 -or $pass -ne [int]($abs -le $limit)){throw 'METRIC_GATE_OR_FINITE_INCONSISTENT'}
   $key="$($m[1]),$($m[2]),$($m[3])";if($keys.ContainsKey($key)){throw 'DUPLICATE_METRIC'};$keys[$key]=$true
   $parsed+=[pscustomobject]@{mode=$m[1];case=[int]$m[2];stage=$m[3];max_abs=$abs;rms=(Number $m[5]);relative_l2=(Number $m[6]);fp_abs=(Number $m[8]);limit=$limit;pass=$pass}
  }
  if($parsed.Count -ne $lane.metrics){throw "METRIC_COUNT_FAIL expected=$($lane.metrics) actual=$($parsed.Count)"}
  $executes=@($lines|Where-Object {$_ -eq 'api,graphExecute,0'}).Count;if($executes -ne $lane.executes){throw 'EXECUTE_COUNT_FAIL'}
  if(@($lines|Where-Object {$_ -eq 'api,contextFree,0'}).Count -ne 1 -or @($lines|Where-Object {$_ -eq 'api,deviceFree,0'}).Count -ne 1 -or @($lines|Where-Object {$_ -eq 'api,backendFree,0'}).Count -ne 1){throw 'TEARDOWN_REQUIRED'}
  $numerical=@($lines|Where-Object {$_ -match '^(onemetric|broadcastcase|upstreamwords|upstreamStageWords|ln1words|upstreamObserver),'});$digest=Digest $numerical;$digests+=$digest
  $observer=0;if($lane.name -eq 'oneUpstream'){if($summary[0] -notmatch 'observer_mismatches=(\d+),'){throw 'OBSERVER_AUDIT_REQUIRED'};$observer=[int]$Matches[1]}
  $customCalls=0;foreach($line in $lines){if($line -match '^oneownership,\d+,custom_executions=(\d+),'){$customCalls+=[int]$Matches[1]}};$customCalls+=$(if($lane.name -eq 'oneBroadcast'){44}else{375})
  if($customCalls -ne $(if($lane.name -eq 'oneBroadcast'){870}else{1875})){throw 'CUSTOM_RECEIPT_COUNT_FAIL'}
  $processes+=[ordered]@{index=$processes.Count+1;metrics=$parsed.Count;qnn_executes=$executes;custom_kernel_receipts=$customCalls;numerical_digest=$digest;observer_mismatch_tensors=$observer}
  if($null -eq $firstMetrics){$firstMetrics=$parsed}
 }
 if(@($digests|Select-Object -Unique).Count -ne 1){throw 'THREE_PROCESS_NUMERICAL_MISMATCH'}
 $groups=@();foreach($group in ($firstMetrics|Group-Object mode,stage)){$v=$group.Group;$expectedCases=if($v[0].mode -in @('broadcastCustom','broadcastStandard')){35}else{25};if($v.Count -ne $expectedCases -or (($v.case|Sort-Object) -join ',') -ne ((0..($expectedCases-1)) -join ',')){throw 'GROUP_CASE_MATRIX_INCOMPLETE'};$groups+=[ordered]@{mode=$v[0].mode;stage=$v[0].stage;cases=$v.Count;max_abs=($v.max_abs|Measure-Object -Maximum).Maximum;max_fp_abs=($v.fp_abs|Measure-Object -Maximum).Maximum;limit=$v[0].limit;failures=@($v|Where-Object pass -eq 0).Count}}
 $lanes+=[ordered]@{lane=$lane.name;processes=$processes;aggregate=$groups}
}
$public=[ordered]@{schema_version=1;provenance=($identity|ConvertFrom-Json);ln1_fixture_sha256=$ln1Identity;lanes=$lanes;training_updates=0;threshold_relaxation=$false;production_builder_changed=$false}
$parent=Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath));[IO.Directory]::CreateDirectory($parent)|Out-Null
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),($public|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
Write-Host 'broadcast_frontier_export=PASS cohorts=2 processes=6 exact_binary=1 numerical_digest_equal=1'
