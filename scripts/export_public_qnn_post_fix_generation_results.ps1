# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
 [string[]]$RunRoots=@(),
 [string]$OutputRoot=(Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-post-fix-generation-2026-07'),
 [string]$ExpectedSourceCommit,
 [switch]$SelfTest
)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$script:Files=@('README.md','training-seeds.csv','oracle-generation.csv','free-generation.csv','logits-comparison.csv','reproducibility.csv','performance.csv')
$script:Utf8=[Text.UTF8Encoding]::new($false)
function Fail($m){throw "post-fix public export: $m"}
function F($m,[string]$k){if(!$m.ContainsKey($k)){Fail "missing $k"};[string]$m[$k]}
function N($m,[string]$k){$v=F $m $k;[double]$n=0;if(![double]::TryParse($v,[Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$n)-or ![double]::IsFinite($n)){Fail "nonfinite $k"};$v}
function B($m,[string]$k){$v=F $m $k;if($v -notin @('true','false')){Fail "bad boolean $k"};$v}
function T($m,[string]$k){$v=F $m $k;if($v-notmatch'^\d+(?:,\d+)*$'){Fail "bad tokens $k"};$v}
function Map($p){$m=@{};foreach($l in Get-Content -LiteralPath $p){if($l-match'^([A-Za-z][A-Za-z0-9_]*)=(.*)$'){if($m.ContainsKey($Matches[1])){Fail "duplicate key $($Matches[1])"};$m[$Matches[1]]=$Matches[2]}};$m}
function Safe($v){if($null-eq$v-or$v-match'[\r\n]' -or $v-match'^[=+@]' -or($v-match'^-' -and $v-notmatch'^-?\d')){Fail 'unsafe CSV value'}}
function PublicState($v){if($null -eq $v -or [string]::IsNullOrWhiteSpace([string]$v)){'NOT_AVAILABLE'}else{[string]$v}}
function Csv($rows,$cols){$out=@((($cols|%{'"'+$_+'"'})-join','));foreach($r in $rows){$cells=@();foreach($c in $cols){$v=[string]$r[$c];Safe $v;$cells+='"'+$v.Replace('"','""')+'"'};$out+=($cells-join',')};($out-join"`n")+"`n"}
function NoReparse($p){$full=[IO.Path]::GetFullPath($p);$cur=[IO.Path]::GetPathRoot($full);foreach($x in $full.Substring($cur.Length).Split('\',[StringSplitOptions]::RemoveEmptyEntries)){$cur=Join-Path $cur $x;if(Test-Path -LiteralPath $cur){if((Get-Item -LiteralPath $cur).Attributes -band [IO.FileAttributes]::ReparsePoint){Fail 'reparse path'}}}}
function ReadRun($root,$commit) {
 $root=[IO.Path]::GetFullPath($root)
 NoReparse $root
 foreach($n in 'device-report.txt','status.json','activity-sampling.json','host-metadata.json') {
  if(!(Test-Path -LiteralPath (Join-Path $root $n) -PathType Leaf)){Fail "missing $n"}
 }
 $h=Get-Content -Raw -LiteralPath (Join-Path $root 'host-metadata.json')|ConvertFrom-Json
 if($h.schema_version -ne 1 -or $h.source_commit -ne $commit -or $h.status -ne 'SUCCESS'){Fail 'host metadata mismatch'}
 if($h.headless_run_id -notmatch '^[A-Za-z0-9._-]{1,64}$' -or
    [int]$h.repetition -lt 1 -or [int]$h.repetition -gt 5){Fail 'invalid run identity'}
 if(($h.phase -eq 'correctness' -and $h.headless_test_mode -ne 'BACKGROUND_CORRECTNESS') -or
    ($h.phase -eq 'performance' -and $h.headless_test_mode -ne 'EXCLUSIVE_BENCHMARK') -or
    $h.phase -notin @('correctness','performance')) {Fail 'invalid host phase/mode'}
 $s=Get-Content -Raw -LiteralPath (Join-Path $root 'status.json')|ConvertFrom-Json
 $a=Get-Content -Raw -LiteralPath (Join-Path $root 'activity-sampling.json')|ConvertFrom-Json
 if($s.status -ne 'PASSED' -or $a.focus_takeover_count -ne 0 -or $a.phonelm_became_top_activity_count -ne 0){Fail 'headless safety mismatch'}
 $m=Map (Join-Path $root 'device-report.txt')
 foreach($k in 'activity_create_count','activity_resume_count','focus_takeover_count'){if((F $m $k) -ne '0'){Fail "device safety $k"}}
 if((F $m 'cpu_fallback') -ne 'false' -or (F $m 'status') -ne 'SUCCESS'){Fail 'device status'}
 [pscustomobject]@{root=$root;run_id=[string]$h.headless_run_id;repetition=[int]$h.repetition;phase=[string]$h.phase;mode=[string]$h.headless_test_mode;map=$m;metadata=$h}
}
function Protocol($m){$want=@{test='post_fix_end_to_end_generation';optimizer='ADAM';learning_rate='0.003';steps='320';global_gradient_clipping='disabled';seed_count='5';nan_detected='false';inf_detected='false';generation_nonfinite_detected='false';formal_oracle_case_count='20';formal_free_case_count='20';formal_prefix_logits_comparison_count='320';formal_qnn_nonzero_return_count='0';formal_cpu_all_finite='true';formal_prefix_comparisons_finite='true';free_running_context_update='PREVIOUS_PREDICTION';oracle_context_update='EXPECTED_TOKEN';free_running_teacher_forcing='false';generation_context_self_test='true'};foreach($e in $want.GetEnumerator()){if((F $m $e.Key)-ne$e.Value){Fail "protocol $($e.Key)"}}}
function WriteStage($stage,$n,$text){[IO.File]::WriteAllText((Join-Path $stage $n),$text,$script:Utf8)}
function ExportIt($roots,$dest,$commit,[bool]$AllowBuildDestination=$false){
 if($roots.Count -notin @(6,8,10)){Fail 'RunRoots must contain matching 3-5 correctness and performance runs'}
 $repo=Split-Path -Parent $PSScriptRoot
 if($commit -notmatch '^[0-9a-f]{40}$' -or (git -C $repo rev-parse HEAD).Trim() -ne $commit){Fail 'ExpectedSourceCommit must equal HEAD'}
 if(!$AllowBuildDestination -and
    @(git -C $repo status --porcelain --untracked-files=normal).Count -ne 0){Fail 'export requires a clean worktree so Source commit matches the evaluated source'}
 $public=[IO.Path]::GetFullPath(
   (Join-Path $repo $(if($AllowBuildDestination){'build\reports'}else{'docs\results'})))
 $dest=[IO.Path]::GetFullPath($dest)
 if(!$dest.StartsWith($public+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $dest)){Fail 'unsafe or existing OutputRoot'}
 NoReparse $public
 $runs=@($roots|%{ReadRun ([IO.Path]::GetFullPath($_)) $commit})
 if(@($runs.root|Sort-Object -Unique).Count -ne $runs.Count -or
    @($runs.run_id|Sort-Object -Unique).Count -ne $runs.Count){Fail 'duplicate run identity'}
 $correct=@($runs|? phase -eq 'correctness');$perf=@($runs|? phase -eq 'performance')
 if($correct.Count -notin (3..5) -or $perf.Count -notin (3..5) -or $correct.Count -ne $perf.Count){Fail 'need matching 3-5 runs per phase'}
 foreach($phaseRuns in @($correct,$perf)){
   $expectedRepetitions=1..$phaseRuns.Count
   if((@($phaseRuns.repetition|Sort-Object)-join ',') -ne ($expectedRepetitions -join ',')){Fail 'run repetitions must be unique and contiguous'}
 }
 foreach($r in $runs){Protocol $r.map};$e=$correct[0].map
 $training=foreach($s in 1..5){$p="seed_${s}_";$hash=F $e "${p}final_parameter_canonical_hash";if($hash-notmatch'^[a-f0-9]{64}$'){Fail 'parameter hash'};$row=[ordered]@{seed=$s;initial_loss=N $e "${p}initial_loss";final_loss=N $e "${p}final_loss";loss_reduction=N $e "${p}loss_reduction";final_token_accuracy=N $e "${p}final_accuracy";nonfinite_count=F $e "${p}nonfinite_count";qnn_execute_count=F $e "${p}qnn_execute_count";qnn_execute_nonzero_count=F $e "${p}qnn_nonzero_return_count";gradient_norm=N $e "${p}final_gradient_l2_norm";parameter_norm=N $e "${p}final_parameter_l2_norm";final_parameter_canonical_hash=$hash;cpu_all_steps_finite=B $e "${p}cpu_all_steps_finite";cpu_nonfinite_count=F $e "${p}cpu_nonfinite_count"};if($row.nonfinite_count-ne'0'-or$row.qnn_execute_nonzero_count-ne'0'-or$row.cpu_all_steps_finite-ne'true'-or$row.cpu_nonfinite_count-ne'0'-or[double]$row.final_loss-ge[double]$row.initial_loss){Fail 'seed outcome'};$row}
 function Cases($mode){$z=@();foreach($s in 1..5){foreach($p in 0..3){$b="formal_${mode}_case_s${s}_p${p}";$z+=[ordered]@{case_id=F $e "${b}_id";prefix=T $e "${b}_prefix";expected_sequence=T $e "${b}_expected_sequence";generated_sequence=T $e "${b}_generated_sequence";exact_match=B $e "${b}_exact";first_mismatch_step=F $e "${b}_first_mismatch_step";expected_token=F $e "${b}_first_mismatch_expected_token";predicted_token=F $e "${b}_first_mismatch_predicted_token";top3=F $e "${b}_first_mismatch_top3";expected_token_probability=F $e "${b}_first_mismatch_expected_probability"}}};$z}
 $oracle=Cases oracle;$free=Cases free
 $logits=@();foreach($mode in 'oracle','free'){foreach($s in 1..5){foreach($p in 0..3){foreach($k in 0..7){$b="formal_logits_${mode}_s${s}_p${p}_step_${k}";$logits+=[ordered]@{mode=$mode;case_id="s${s}_p${p}";step=$k;max_abs_difference=N $e "${b}_max_abs_difference";mean_abs_difference=N $e "${b}_mean_abs_difference";relative_l2_difference=N $e "${b}_relative_l2_difference";argmax_match=B $e "${b}_argmax_match";top3_match=B $e "${b}_top3_match"}}}}};if($logits.Count-ne320){Fail 'logit count'}
 $rr=@()
 foreach($r in $correct){
  $m=$r.map;$hash=F $m 'formal_representative_final_logits_canonical_hash'
  if($hash-notmatch'^[a-f0-9]{64}$'){Fail 'logits hash'}
  $row=[ordered]@{run=$rr.Count+1;seed1_final_parameter_hash=F $m 'seed_1_final_parameter_canonical_hash';final_logits_hash=$hash;seed1_final_loss=N $m 'seed_1_final_loss';seed1_final_accuracy=N $m 'seed_1_final_accuracy';oracle_exact_aggregate=F $m 'oracle_exact_rollout_count';free_exact_aggregate=F $m 'exact_rollout_count'}
  foreach($mode in 'oracle','free'){foreach($s in 1..5){foreach($p in 0..3){$base="formal_${mode}_case_s${s}_p${p}";$row["${mode}_s${s}_p${p}_generated_sequence"]=T $m "${base}_generated_sequence";$row["${mode}_s${s}_p${p}_exact"]=B $m "${base}_exact"}}}
  $rr+=$row
 }
 $allFields=@($rr[0].Keys | Where-Object {$_ -ne 'run'})
 $allIdentical=$true;foreach($field in $allFields){if(@($rr|ForEach-Object { $_[$field] }|Sort-Object -Unique).Count-ne1){$allIdentical=$false}}
 $generationFields=@($allFields|Where-Object {$_ -match '^(?:oracle|free)_s\d+_p\d+_(?:generated_sequence|exact)$'})
 $generationIdentical=$true;foreach($field in $generationFields){if(@($rr|ForEach-Object { $_[$field] }|Sort-Object -Unique).Count-ne1){$generationIdentical=$false}}
 $allFinite=$true;foreach($r in $correct){foreach($s in 1..5){if((F $r.map "seed_${s}_nonfinite_count") -ne '0'){$allFinite=$false}}}
 $class=if($allIdentical){'BITWISE_REPRODUCIBLE'}elseif($allFinite -and $generationIdentical){'FINITE_NUMERIC_VARIATION_ONLY'}else{'TRAINING_TRAJECTORY_VARIATION'}
 $metrics='performance_initialization_ms','performance_graph_creation_ms','performance_finalize_ms',
   'performance_steady_training_step_min_ms','performance_steady_training_step_median_ms',
   'performance_steady_training_step_p95_ms','performance_steady_training_step_max_ms',
   'performance_updates_per_second','performance_tokens_per_second',
   'generation_token_latency_min_ms','generation_token_latency_median_ms',
   'generation_token_latency_p95_ms','generation_token_latency_max_ms'
 $pr=@();foreach($metric in $metrics){$v=@($perf|%{[double](N $_.map $metric)}|Sort-Object);$pr+=[ordered]@{metric=$metric;run_count=$v.Count;median=$v[[int][math]::Floor(($v.Count-1)/2)];p95=$v[[int][math]::Ceiling(.95*($v.Count-1))]}}
 $environmentRows=@();foreach($r in $perf){$before=$r.metadata.device_before;$after=$r.metadata.device_after;$environmentRows+=[ordered]@{metric='device_state';run_count=1;median='NOT_APPLICABLE';p95='NOT_APPLICABLE';battery_temperature_c_before=PublicState $before.battery_temperature_c;battery_temperature_c_after=PublicState $after.battery_temperature_c;battery_status_before=PublicState $before.battery_status;battery_status_after=PublicState $after.battery_status;plugged_before=PublicState $before.plugged;plugged_after=PublicState $after.plugged;thermal_status_before=PublicState $before.thermal_status;thermal_status_after=PublicState $after.thermal_status;screen_wakefulness_before=PublicState $before.screen_wakefulness;screen_wakefulness_after=PublicState $after.screen_wakefulness;phonelm_is_top_before=PublicState $before.phonelm_is_top;phonelm_is_top_after=PublicState $after.phonelm_is_top}}
 $pr += $environmentRows
 $stage=Join-Path $repo ('build\reports\public-export-staging\postfix-'+[guid]::NewGuid().ToString('N'))
 try {
  [IO.Directory]::CreateDirectory($stage)|Out-Null
  WriteStage $stage 'training-seeds.csv' (Csv $training @('seed','initial_loss','final_loss','loss_reduction','final_token_accuracy','nonfinite_count','qnn_execute_count','qnn_execute_nonzero_count','gradient_norm','parameter_norm','final_parameter_canonical_hash','cpu_all_steps_finite','cpu_nonfinite_count'))
  $cc=@('case_id','prefix','expected_sequence','generated_sequence','exact_match','first_mismatch_step','expected_token','predicted_token','top3','expected_token_probability')
  WriteStage $stage 'oracle-generation.csv' (Csv $oracle $cc);WriteStage $stage 'free-generation.csv' (Csv $free $cc)
  WriteStage $stage 'logits-comparison.csv' (Csv $logits @('mode','case_id','step','max_abs_difference','mean_abs_difference','relative_l2_difference','argmax_match','top3_match'))
  WriteStage $stage 'reproducibility.csv' ("# classification=$class`n"+(Csv $rr (@('run') + $allFields)))
  $performanceColumns=@('metric','run_count','median','p95','battery_temperature_c_before','battery_temperature_c_after','battery_status_before','battery_status_after','plugged_before','plugged_after','thermal_status_before','thermal_status_after','screen_wakefulness_before','screen_wakefulness_after','phonelm_is_top_before','phonelm_is_top_after')
  WriteStage $stage 'performance.csv' (Csv $pr $performanceColumns)
  WriteStage $stage 'README.md' ("# QNN HTP post-fix generation results`n`nSource commit: ${commit}. This allow-listed bundle records CPU orchestration and HTP numerical execution; QNN backward is explicit graphs, not autograd. Oracle contexts use expected tokens; free contexts use prior predictions. QNN return success and finite values are separate checks.`n")
  $names=@((Get-ChildItem $stage -File).Name|Sort-Object)-join',';if($names-ne(@($script:Files|Sort-Object)-join',')){Fail 'output allow list'}
  foreach($f in Get-ChildItem $stage -File){if((Get-Content -Raw $f)-match'(?i)[a-z]:\\|/data/|/sdcard/|\badb\b|\b(?:serial|endpoint|logcat|private|raw[_ -]?(?:tensor|checkpoint|logits))\b|private key'){Fail 'restricted output'}}
  Move-Item $stage $dest;Write-Host "Exported to $dest"
 } finally {if(Test-Path $stage){Remove-Item $stage -Recurse -Force}}
}
function Fake($root,$phase,$i,$sha){$testMode=if($phase -eq 'performance'){'EXCLUSIVE_BENCHMARK'}else{'BACKGROUND_CORRECTNESS'};[IO.Directory]::CreateDirectory($root)|Out-Null;'{"status":"PASSED"}'|Set-Content (Join-Path $root status.json);'{"focus_takeover_count":0,"phonelm_became_top_activity_count":0}'|Set-Content (Join-Path $root activity-sampling.json);$device=[ordered]@{battery_temperature_c=30;battery_status=2;plugged=$true;thermal_status=0;screen_wakefulness='Awake';phonelm_is_top=$false};([ordered]@{schema_version=1;source_commit=$sha;phase=$phase;repetition=$i;headless_run_id="selftest-$phase-$i";headless_test_mode=$testMode;status='SUCCESS';device_before=$device;device_after=$device}|ConvertTo-Json)|Set-Content (Join-Path $root host-metadata.json);$m=[ordered]@{status='SUCCESS';test='post_fix_end_to_end_generation';optimizer='ADAM';learning_rate='0.003';steps='320';global_gradient_clipping='disabled';seed_count='5';nan_detected='false';inf_detected='false';generation_nonfinite_detected='false';formal_oracle_case_count='20';formal_free_case_count='20';formal_prefix_logits_comparison_count='320';formal_qnn_nonzero_return_count='0';formal_cpu_all_finite='true';formal_prefix_comparisons_finite='true';free_running_context_update='PREVIOUS_PREDICTION';oracle_context_update='EXPECTED_TOKEN';free_running_teacher_forcing='false';generation_context_self_test='true';exact_rollout_count='20';oracle_exact_rollout_count='20';activity_create_count='0';activity_resume_count='0';focus_takeover_count='0';cpu_fallback='false'};foreach($s in 1..5){foreach($x in @{initial_loss='3';final_loss='1';loss_reduction='2';final_accuracy='1';nonfinite_count='0';qnn_execute_count='1';qnn_nonzero_return_count='0';final_gradient_l2_norm='1';final_parameter_l2_norm='2';final_parameter_canonical_hash=('a'*64);cpu_all_steps_finite='true';cpu_nonfinite_count='0'}.GetEnumerator()){$m["seed_${s}_$($x.Key)"]=$x.Value}};foreach($mode in 'oracle','free'){foreach($s in 1..5){foreach($p in 0..3){$b="formal_${mode}_case_s${s}_p${p}";foreach($x in @{id="s${s}_p${p}";prefix='0,1';expected_sequence='0';generated_sequence='0';exact='true';first_mismatch_step='-1';first_mismatch_expected_token='NONE';first_mismatch_predicted_token='NONE';first_mismatch_top3='NONE';first_mismatch_expected_probability='NONE'}.GetEnumerator()){$m["${b}_$($x.Key)"]=$x.Value};foreach($k in 0..7){$l="formal_logits_${mode}_s${s}_p${p}_step_${k}";foreach($x in @{max_abs_difference='0';mean_abs_difference='0';relative_l2_difference='0';argmax_match='true';top3_match='true'}.GetEnumerator()){$m["${l}_$($x.Key)"]=$x.Value}}}}};$m.formal_representative_final_logits_canonical_hash=('b'*64);foreach($x in 'performance_initialization_ms','performance_graph_creation_ms','performance_finalize_ms','performance_steady_training_step_min_ms','performance_steady_training_step_median_ms','performance_steady_training_step_p95_ms','performance_steady_training_step_max_ms','performance_updates_per_second','performance_tokens_per_second','generation_token_latency_min_ms','generation_token_latency_median_ms','generation_token_latency_p95_ms','generation_token_latency_max_ms'){$m[$x]='1'};(($m.GetEnumerator()|%{"$($_.Key)=$($_.Value)"})-join"`n")|Set-Content (Join-Path $root device-report.txt)}
if($SelfTest){
 $repo=Split-Path -Parent $PSScriptRoot
 $tmp=Join-Path $repo ('build\reports\export-selftest-'+[guid]::NewGuid().ToString('N'))
 $out=Join-Path $tmp 'public'
 try {
  $sha=(git -C $repo rev-parse HEAD).Trim()
  $rs=@()
  foreach($p in 'correctness','performance'){
   foreach($i in 1..3){
    $r=Join-Path $tmp "$p-$i"
    Fake $r $p $i $sha
    $rs+=$r
   }
  }
  ExportIt $rs $out $sha $true
  $missingRejected=$false
  try { ExportIt $rs[0..4] (Join-Path $tmp 'bad-missing') $sha $true } catch { $missingRejected=$true }
  if(!$missingRejected){Fail 'missing-run negative selftest accepted'}
  $duplicateRejected=$false
  $duplicates=@($rs[0],$rs[0],$rs[2],$rs[3],$rs[4],$rs[5])
  try { ExportIt $duplicates (Join-Path $tmp 'bad-duplicate') $sha $true } catch { $duplicateRejected=$true }
  if(!$duplicateRejected){Fail 'duplicate-run negative selftest accepted'}
  Write-Host 'SELF_TEST=PASS'
 } finally {
  if(Test-Path $tmp){Remove-Item $tmp -Recurse -Force}
 }
} else {
 ExportIt $RunRoots $OutputRoot $ExpectedSourceCommit
}
