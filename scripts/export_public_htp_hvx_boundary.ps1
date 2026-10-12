param([string[]]$PrivateDirectories,[string]$OutputPath,[switch]$SelfTest)
$ErrorActionPreference='Stop'
$cases=@('ordinary','small_centered','low_variance','epsilon_dominated','constant','large_mean_low_variance','boundary_below','boundary_equal','boundary_above','known_failure','large_signed','half_scale_limit','very_large_signed','large_mean','sparse_known_failure','sparse_large','random_seed1','random_seed7','random_seed31','random_seed97')
$fields=@('input','centered','inverse','xhat','dy','dxhat','dx','dg','db')
function Read-TypedLine([string]$Line) {
  $parts=$Line.Split(' ',[StringSplitOptions]::RemoveEmptyEntries)
  $kind=$parts[0]
  $schema=switch($kind){
    boundary {@('name','field','op','mismatch','transfer_abs','reference_abs','nonfinite','gate')}
    stage {@('name','field','htp_fp32_abs','oracle_double_abs')}
    raw_boundary {@('op','mismatch','abs','nonfinite')}
    subtract {@('name','abs','output_nonzero','row0_nonzero')}
    observer {@('name','bit_mismatch')}
    default {throw 'UNKNOWN_PUBLIC_RECORD'}
  }
  $row=[ordered]@{kind=$kind}
  foreach($part in $parts[1..($parts.Count-1)]) {
    $pair=$part.Split('=',2);if($pair.Count -ne 2 -or $pair[0] -notin $schema -or $row.Contains($pair[0])){throw 'PUBLIC_FIELD_REJECTED'}
    $key=$pair[0];$value=$pair[1]
    if($key -eq 'name'){if($value -notin $cases){throw 'PUBLIC_CASE_REJECTED'};$row[$key]=$value}
    elseif($key -eq 'field'){if($value -notin $fields){throw 'PUBLIC_TENSOR_REJECTED'};$row[$key]=$value}
    elseif($key -eq 'op'){if($value -notin @('reshape','add_zero','multiply_one')){throw 'PUBLIC_OP_REJECTED'};$row[$key]=$value}
    else {
      if($value -match '^(?:inf|-inf|nan)$'){$row[$key]=$null}
      elseif($value -match '^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$'){$row[$key]=[double]::Parse($value,[Globalization.CultureInfo]::InvariantCulture)}
      else{throw 'PUBLIC_NUMERIC_REJECTED'}
    }
  }
  if($row.Count -ne $schema.Count+1){throw 'PUBLIC_FIELDS_MISSING'}
  return $row
}
function Hash-Text([string]$Text){return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($Text))).ToLowerInvariant()}
if($SelfTest){
  $r=Read-TypedLine 'boundary name=constant field=dx op=add_zero mismatch=128 transfer_abs=0.1 reference_abs=0.1 nonfinite=0 gate=0'
  if($r.reference_abs -ne .1){throw 'SELFTEST_PARSE'}
  foreach($bad in @('stage name=private_path field=dx htp_fp32_abs=0 oracle_double_abs=0','stage name=constant field=dx htp_fp32_abs=0 oracle_double_abs=0 endpoint=private','stage name=constant field=dx htp_fp32_abs=0')){
    $rejected=$false;try{Read-TypedLine $bad|Out-Null}catch{$rejected=$true};if(!$rejected){throw 'SELFTEST_PRIVATE_LEAK'}
  }
  Write-Host 'htp_hvx_boundary_exporter_selftest=PASS';exit
}
if($PrivateDirectories.Count -ne 3 -or !$OutputPath){throw 'THREE_PRIVATE_PROCESSES_AND_OUTPUT_REQUIRED'}
$runs=@();$primary=$null;$source=$null;$identity=$null;$digest=$null
foreach($dir in $PrivateDirectories){
  $plan=Get-Content (Join-Path $dir 'validation-plan.json') -Raw|ConvertFrom-Json
  if($plan.status -ne 'PASS' -or $plan.suite -ne 'htp-hvx-boundary' -or !$plan.protected_identity_verified -or $plan.host_free_bytes -lt 20GB -or $plan.device_free_bytes -lt 12GB -or $plan.qairt_build_id -ne '2.48.40.260702151143'){throw 'PRIVATE_PROVENANCE_INVALID'}
  if(!$source){$source=$plan.source_sha;$identity=$plan.device_identity_private}else{if($source -ne $plan.source_sha -or $identity -ne $plan.device_identity_private){throw 'PRIVATE_IDENTITY_MISMATCH'}}
  if($source -notmatch '^[a-f0-9]{40}$'){throw 'SOURCE_SHA_INVALID'}
  $lines=Get-Content (Join-Path $dir 'device-report.txt');$text=$lines -join "`n"
  foreach($anchor in @('status=SUCCESS','transport_mismatch=0','rpc_close_status=0','observer_bit_mismatch=0','training_updates=0','cpu_fallback=false','focus_takeover_count=0')){if($anchor -notin $lines){throw "PRIVATE_ANCHOR_MISSING:$anchor"}}
  if('qnn_calls=603' -notin $lines -or 'rpc_calls=366' -notin $lines){throw 'EXECUTION_COUNT_MISMATCH'}
  $success=@($lines|Where-Object {$_ -match '^api_trace_graph_execute_success_count=\d+$'}|ForEach-Object {[long]($_.Split('=')[1])})
  $failure=@($lines|Where-Object {$_ -match '^api_trace_graph_execute_failure_count=\d+$'}|ForEach-Object {[long]($_.Split('=')[1])})
  if($success.Count -ne 6 -or ($success|Measure-Object -Sum).Sum -ne 603 -or $failure.Count -ne 6 -or ($failure|Measure-Object -Sum).Sum -ne 0){throw 'QNN_API_NOT_VERIFIED'}
  $transfers=@($lines|Where-Object {$_ -match '^transport '});if($transfers.Count -ne 366){throw 'RPC_COUNT_MISMATCH'}
  foreach($line in $transfers){if($line -notmatch ' status=0 dsp_mismatch=0 host_mismatch=0 input_mutation=0 complete=1 ' -or $line -notmatch ' finite=1 ' -or $line -notmatch ' receipt_mismatch=0 output_finite=1 '){throw 'TRANSPORT_NOT_BIT_EXACT'}}
  $fixed=@($transfers|Where-Object {$_ -match '^transport name=fixed[0-5] count=256 '});if($fixed.Count -ne 6){throw 'FIXED_REPETITIONS_MISSING'}
  $records=@($lines|Where-Object {$_ -match '^(?:boundary|stage|raw_boundary|subtract|observer) '}|ForEach-Object {Read-TypedLine $_})
  if($records.Count -ne 763){throw 'NUMERICAL_RECORD_COUNT_MISMATCH'}
  $hash=Hash-Text ($records|ConvertTo-Json -Depth 8 -Compress)
  if(!$digest){$digest=$hash;$primary=$records}else{if($digest -ne $hash){throw 'INDEPENDENT_NUMERICAL_MISMATCH'}}
  $rpcTimes=@($transfers|ForEach-Object {if($_ -notmatch ' rpc_us=([\d.]+) '){throw 'RPC_TIME_MISSING'};[double]::Parse($Matches[1],[Globalization.CultureInfo]::InvariantCulture)}|Sort-Object)
  $runs+= [ordered]@{process_index=$runs.Count+1;rpc_calls=366;qnn_calls=603;fixed_repetitions=6;fixed_elements=256;transport_mismatch=0;observer_mismatch=0;rpc_median_us=$rpcTimes[183];numerical_digest=$hash}
}
$result=[ordered]@{schema='HTP_HVX_BOUNDARY_V1';source_sha=$source;qairt_build_id='2.48.40.260702151143';hexagon_sdk='6.6.0.0';architecture='V81';training_updates=0;processes=$runs;records=$primary}
$parent=Split-Path -Parent $OutputPath;if($parent){[IO.Directory]::CreateDirectory($parent)|Out-Null}
[IO.File]::WriteAllText($OutputPath,($result|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
Write-Host 'htp_hvx_boundary_public_export=PASS'
