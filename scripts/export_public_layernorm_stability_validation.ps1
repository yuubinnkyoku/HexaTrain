# SPDX-License-Identifier: Apache-2.0
# Typed allow-list; failed candidate evidence remains failed evidence.
param([string]$PrivateIndex,[string]$Destination,[switch]$SelfTest)
$ErrorActionPreference='Stop'
function Select-Fields([string]$Line,[string]$Kind){
 $caseNames='ordinary|small_centered|low_variance|epsilon_dominated|constant|large_mean_low_variance|boundary_below|boundary_equal|boundary_above|known_failure|large_signed|half_scale_limit|very_large_signed|large_mean|sparse_known_failure|sparse_large'
 if($Line -notmatch "^$Kind d=(8|64|128) name=($caseNames) "){throw 'CASE_NOT_ALLOW_LISTED'}
 $fields=[ordered]@{kind=$Kind}
 $integers=@('d','repeat','required','finite','pass','observer_bit_mismatch','bit_mismatch','square_nonfinite')
 $floats=@('y_abs','xhat_abs','dx_abs','dg_abs','db_abs','fp32_dx_abs','mean','mean_abs','selected_scale','second_scale','variance_s','variance_abs','variance_eps_s','inverse_s','inverse','inverse_abs','centered_max','square_max')
 foreach($part in ($Line -split ' ' | Select-Object -Skip 1)){
  $pair=$part -split '=',2;if($pair.Count -ne 2){throw 'MALFORMED_FIELD'};$k=$pair[0];$v=$pair[1]
  if($k -eq 'name'){$fields[$k]=$v}
  elseif($k -in $integers){if($v -notmatch '^\d+$'){throw 'INVALID_INTEGER'};$fields[$k]=[long]$v}
  elseif($k -in $floats){
   if($v -match '^(inf|-inf|nan)$'){$fields[$k]=$null;$fields[$k+'_nonfinite']=$true}
   else{$n=[double]::Parse($v,[Globalization.CultureInfo]::InvariantCulture);if(-not [double]::IsFinite($n)){throw 'INVALID_FLOAT'};$fields[$k]=$n}
  } else{throw 'FIELD_NOT_ALLOW_LISTED'}
 }
 $fields
}
if($SelfTest){
 $v=Select-Fields 'case d=64 name=half_scale_limit repeat=0 required=0 finite=0 y_abs=inf dx_abs=inf pass=0' 'case'
 if($v.y_abs -ne $null -or -not $v.y_abs_nonfinite -or $v.pass -ne 0){throw 'NONFINITE_CONVERSION_FAILED'}
 foreach($line in @('case d=64 name=private repeat=0 pass=1','case d=64 name=ordinary private_path=secret','case d=64 name=ordinary dx_abs=secret')){
  $rejected=$false;try{Select-Fields $line 'case'|Out-Null}catch{$rejected=$true};if(-not $rejected){throw 'PRIVACY_REJECTION_FAILED'}
 }
 'public_layernorm_validation_self_test=PASS';exit 0
}
if(-not $PrivateIndex -or -not $Destination){throw 'INDEX_AND_DESTINATION_REQUIRED'}
$root=Split-Path -Parent $PSScriptRoot
$base=Join-Path $root 'build/reports/g1-layernorm-validation'
$index=Get-Content -LiteralPath $PrivateIndex -Raw|ConvertFrom-Json
$runs=@()
foreach($item in $index.runs){
 if($item.label -notmatch '^[A-Z_]{2,32}$' -or $item.directory -notmatch '^lnvalidation-[a-f0-9]{32}$'){throw 'INVALID_INDEX'}
 $p=Get-Content (Join-Path $base "$($item.directory)/validation-plan.json") -Raw|ConvertFrom-Json
 if($p.source_sha -notmatch '^[a-f0-9]{40}$' -or $p.qairt_build_id -ne '2.48.40.260702151143' -or -not $p.protected_identity_verified){throw 'PROVENANCE_INVALID'}
 $text=Get-Content (Join-Path $base "$($item.directory)/device-report.txt") -Raw
 if($text -notmatch '(?m)^required_correctness_gate=(PASS|FAIL)$' -or $text -notmatch '(?m)^legacy_backward_gate=PASS$'){throw 'SCIENTIFIC_REPORT_INCOMPLETE'}
 $gate=[regex]::Match($text,'(?m)^required_correctness_gate=(PASS|FAIL)$').Groups[1].Value
 $attempts=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_attempt_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 $successes=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_success_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 $failures=@([regex]::Matches($text,'(?m)^api_trace_graph_execute_failure_count=(\d+)$')|ForEach-Object{[long]$_.Groups[1].Value})
 if($attempts.Count -eq 0 -or $successes.Count -ne $attempts.Count -or $failures.Count -ne $attempts.Count){throw 'QNN_COUNTS_INCOMPLETE'}
 for($i=0;$i -lt $attempts.Count;$i++){if($attempts[$i] -ne $successes[$i] -or $failures[$i] -ne 0){throw 'QNN_FAILURE_NOT_A_CORRECTNESS_RESULT'}}
 $calls=[long][regex]::Match($text,'(?m)^htp_graph_execute_count=(\d+)$').Groups[1].Value
 if(($attempts|Measure-Object -Sum).Sum -ne $calls){throw 'QNN_COUNT_MISMATCH'}
 $records=@();foreach($line in ($text -split "`r?`n")){if($line -match '^(case|native|transparency|row) '){$records+=Select-Fields $line $Matches[1]}}
 $caseRows=@($records|Where-Object{$_.kind -eq 'case'})
 if($caseRows.Count -notin @(126,144)){throw 'UNEXPECTED_CASE_COUNT'}
 $requiredFailed=@($caseRows|Where-Object{$_.required -eq 1 -and $_.pass -eq 0}).Count -gt 0
 $requiredFailed=$requiredFailed -or @($records|Where-Object{$_.kind -eq 'transparency' -and $_.bit_mismatch -ne 0}).Count -gt 0
 $requiredFailed=$requiredFailed -or @($records|Where-Object{$_.kind -eq 'native' -and $_.name -notin @('half_scale_limit','very_large_signed') -and $_.pass -eq 0}).Count -gt 0
 if(($gate -eq 'PASS') -eq $requiredFailed){throw 'CORRECTNESS_CLASSIFICATION_MISMATCH'}
 $runs+=[ordered]@{label=$item.label;source_sha=$p.source_sha;qairt_build_id=$p.qairt_build_id;
  candidate_correctness=$gate;legacy_backward_gate='PASS';training_updates_executed=0;
  qnn_execute_attempts=$calls;qnn_execute_success=$calls;qnn_execute_failure=0;
  limits=[ordered]@{ordinary_forward_abs=0.002;low_variance_forward_abs=0.06;gradient_abs=0.02;central_difference_abs=0.0005};
  scope='Procedural LayerNorm graphs; legacy microtest executes excluded from counts';records=$records}
}
$stressStatus=if(@($runs|Where-Object{$_.candidate_correctness -eq 'FAIL'}).Count -gt 0){'BLOCKED_CORRECTNESS_PREREQUISITE'}else{'NOT_EXECUTED'}
$public=[ordered]@{schema='LAYER_NORM_VALIDATION_PUBLIC_V1';stress_60k_status=$stressStatus;runs=$runs}
$json=$public|ConvertTo-Json -Depth 12
if($json -match 'run_id|private_path|device_identity|[A-Z]:\\|lnvalidation-[a-f0-9]{32}|APK|Nicopedia本文'){throw 'PUBLIC_PRIVACY_SCAN_FAILED'}
[IO.Directory]::CreateDirectory((Split-Path -Parent ([IO.Path]::GetFullPath($Destination))))|Out-Null
[IO.File]::WriteAllText([IO.Path]::GetFullPath($Destination),$json,[Text.UTF8Encoding]::new($false))
"public_layernorm_validation_export=PASS runs=$($runs.Count)"
