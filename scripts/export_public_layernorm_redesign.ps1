# SPDX-License-Identifier: Apache-2.0
param([string]$PrivateIndex,[string]$Destination,[switch]$SelfTest)
$ErrorActionPreference='Stop'
$candidateNames=@('PREVIOUS','DYNAMIC_RANGE','MEAN_BACKWARD','SCALE8','FLOAT32_CONFIG','COMPENSATION_CONFIG')
$names=@('ordinary','small_centered','low_variance','epsilon_dominated','constant','large_mean_low_variance','boundary_below','boundary_equal','boundary_above','known_failure','large_signed','half_scale_limit','very_large_signed','large_mean','sparse_known_failure','sparse_large','random_seed1','random_seed7','random_seed31','random_seed97')
function Read-Record([string]$line) {
 $parts=$line -split ' ';$kind=$parts[0]
 $numeric=@('d','y_abs','xhat_abs','dx_abs','dg_abs','db_abs','fp32_dx_abs','observer_bit_mismatch','previous_bit_mismatch','pass','abs','rms','relative','min','max','nonfinite','inverse_only_dx_abs','xhat_only_dx_abs','forward_only_dx_abs','backward_residual_abs','final_multiply_local_abs','half_lattice_lower_bound','observed_lattice_distance','pass_count','total')
 if($kind -notin @('case','metric','causal','candidate')){throw 'RECORD_KIND_NOT_ALLOWED'}
 $record=[ordered]@{kind=$kind}
 foreach($part in $parts[1..($parts.Length-1)]) {
  $kv=$part -split '=',2;if($kv.Length -ne 2 -or $record.Contains($kv[0])){throw 'INVALID_OR_DUPLICATE_FIELD'};$k=$kv[0];$v=$kv[1]
  if($k -in @('candidate','label')){if($v -notin $candidateNames -and $v -notin @('DIRECT_FORWARD_ONLY','PRIMITIVE_DEFAULT','PRIMITIVE_FLOAT32')){throw 'CANDIDATE_NOT_ALLOWED'};$record[$k]=$v}
  elseif($k -eq 'name'){if($v -notin $names -and $v -notmatch '^\d+(\.\d+)?([eE][+-]?\d+)?$'){throw 'NAME_NOT_ALLOWED'};$record[$k]=$v}
  elseif($k -eq 'stage'){if($v -notmatch '^tap([0-9]|1[0-9]|20)$' -and $v -notin @('xhat','dx','add','subtract','multiply','divide','sqrt','rsqrt','reciprocal_via_divide','mean','sum')){throw 'STAGE_NOT_ALLOWED'};$record[$k]=$v}
  elseif($k -eq 'gate'){if($v -notin @('PASS','FAIL')){throw 'GATE_NOT_ALLOWED'};$record[$k]=$v}
  elseif($k -in $numeric){
   if($v -match '^(inf|-inf|nan)$'){$record[$k]=$null;$record[$k+'_nonfinite']=$true}
   else{if($v -notmatch '^-?\d+(\.\d+)?([eE][+-]?\d+)?$'){throw 'NUMERIC_FIELD_INVALID'};$n=[double]::Parse($v,[Globalization.CultureInfo]::InvariantCulture);if(-not [double]::IsFinite($n)){throw 'NUMERIC_FIELD_INVALID'};$record[$k]=$n}
  }else{throw 'FIELD_NOT_ALLOWED'}
 }
 if($record.Contains('d') -and $record.d -notin @(8,16,32,64,128)){throw 'DIMENSION_NOT_ALLOWED'}
 $record
}
if($SelfTest){
 $r=Read-Record 'case candidate=PREVIOUS d=64 name=half_scale_limit dx_abs=inf pass=0'
 if($null -ne $r.dx_abs -or -not $r.dx_abs_nonfinite){throw 'NONFINITE_NOT_PRESERVED'}
 foreach($bad in @('case candidate=SECRET d=64 name=ordinary pass=1','case candidate=PREVIOUS d=64 name=ordinary path=secret','causal d=64 name=ordinary dx_abs=bad','case d=63 name=ordinary pass=1')){
  $rejected=$false;try{Read-Record $bad|Out-Null}catch{$rejected=$true};if(-not $rejected){throw 'ALLOW_LIST_FAILURE'}
 }
 'layernorm_redesign_export_selftest=PASS';exit 0
}
if(-not $PrivateIndex -or -not $Destination){throw 'INDEX_AND_DESTINATION_REQUIRED'}
$root=Split-Path -Parent $PSScriptRoot;$base=Join-Path $root 'build/reports/htp-layernorm-redesign'
$runs=@();$primaryHash=$null
foreach($item in (Get-Content -LiteralPath $PrivateIndex -Raw|ConvertFrom-Json).runs){
 if($item.label -notmatch '^[A-Z_]{2,32}$' -or $item.directory -notmatch '^lnredesign-[a-f0-9]{32}$'){throw 'INDEX_INVALID'}
 $dir=Join-Path $base $item.directory;$p=Get-Content (Join-Path $dir 'validation-plan.json') -Raw|ConvertFrom-Json
 if($p.source_sha -notmatch '^[a-f0-9]{40}$' -or $p.qairt_build_id -ne '2.48.40.260702151143' -or -not $p.protected_identity_verified -or $p.suite -ne 'layernorm-redesign'){throw 'PROVENANCE_INVALID'}
 $text=Get-Content (Join-Path $dir 'device-report.txt') -Raw
 if($text -notmatch '(?m)^LAYERNORM_REDESIGN$' -or $text -notmatch '(?m)^cpu_fallback=false$' -or $text -notmatch '(?m)^accepted_leaf_candidate=[01]$'){throw 'REPORT_INCOMPLETE'}
 $attempts=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_attempt_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 $success=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_success_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 $fail=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_failure_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 if($attempts.Count -ne 67 -or $success.Count -ne 67 -or $fail.Count -ne 67){throw 'GRAPH_COUNTS_INCOMPLETE'}
 for($i=0;$i -lt 67;$i++){if($attempts[$i] -ne $success[$i] -or $fail[$i] -ne 0){throw 'QNN_FAILURE'}}
 if(($attempts|Measure-Object -Sum).Sum -ne 1326 -or $text -notmatch '(?m)^htp_graph_execute_count=1326$'){throw 'CALL_COUNT_MISMATCH'}
 $records=@();foreach($line in $text -split "`r?`n"){if($line -match '^(case|metric|causal|candidate) '){$records+=Read-Record $line}}
 $cases=@($records|Where-Object{$_.kind -eq 'case'});$candidates=@($records|Where-Object{$_.kind -eq 'candidate'})
 if($cases.Count -ne 600 -or $candidates.Count -ne 6){throw 'MATRIX_INCOMPLETE'}
 foreach($c in $candidates){$actual=@($cases|Where-Object{$_.candidate -eq $c.label});$passes=@($actual|Where-Object{$_.pass -eq 1}).Count
  if($actual.Count -ne 100 -or $c.total -ne 100 -or $passes -ne $c.pass_count -or (($c.gate -eq 'PASS') -ne ($passes -eq 100))){throw 'CANDIDATE_GATE_MISMATCH'}
 }
 $accepted=@($candidates|Where-Object{$_.gate -eq 'PASS'}).Count -gt 0
 if(($text -match '(?m)^accepted_leaf_candidate=1$') -ne $accepted){throw 'FINAL_CLASSIFICATION_MISMATCH'}
 $numericJson=$records|ConvertTo-Json -Depth 8 -Compress
 $numericHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($numericJson))).ToLowerInvariant()
 if($null -eq $primaryHash){$primaryHash=$numericHash
  $selected=@($records|Where-Object{$_.kind -ne 'metric' -or $_.d -eq 64})
 }else{
  if($numericHash -ne $primaryHash){throw 'INDEPENDENT_PROCESS_NUMERICAL_MISMATCH'}
  $selected=@() # Exact duplicate numeric records stay private; publish equality hash.
 }
 $runs+=[ordered]@{label=$item.label;source_sha=$p.source_sha;qairt_build_id=$p.qairt_build_id;qnn_execute_attempts=1326;qnn_execute_success=1326;qnn_execute_failure=0;accepted_leaf_candidate=$accepted;training_updates=0;full_numeric_record_count=$records.Count;full_numeric_records_sha256=$numericHash;matches_primary_numeric_records=($numericHash -eq $primaryHash);records=$selected}
}
$bundle=[ordered]@{schema='HTP_LAYERNORM_REDESIGN_V1';gradient_abs_limit=.02;ordinary_forward_abs_limit=.002;low_variance_forward_abs_limit=.06;stress_60k='NOT_EXECUTED';runs=$runs}
$json=$bundle|ConvertTo-Json -Depth 12
if($json -match 'lnredesign-[a-f0-9]{32}|[A-Z]:\\|device_identity|private_path|run_id'){throw 'PRIVACY_SCAN_FAILED'}
[IO.Directory]::CreateDirectory((Split-Path -Parent ([IO.Path]::GetFullPath($Destination))))|Out-Null
[IO.File]::WriteAllText([IO.Path]::GetFullPath($Destination),$json,[Text.UTF8Encoding]::new($false))
"layernorm_redesign_export=PASS runs=$($runs.Count)"
