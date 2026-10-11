# SPDX-License-Identifier: Apache-2.0
# Export only numerical aggregates, never private paths, identifiers or tensors.
param([string]$PrivateIndex, [string]$Destination, [switch]$SelfTest)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
function Select-Health($row) {
  if ($row.tensor -notmatch '^layer_\d{3}_(ln[12]_(centered_s|square|safe_scale)|output)$') { throw 'TENSOR_NOT_ALLOW_LISTED' }
  [ordered]@{tensor=$row.tensor; count=[int]$row.count; finite=[int]$row.finite
    nan=[int]$row.nan; pos_inf=[int]$row.pos_inf; neg_inf=[int]$row.neg_inf
    min=[double]$row.min; max=[double]$row.max; rms=[double]$row.rms
    first_bad=[int]$row.first_bad; canonical_sha256=[string]$row.hash}
}
if ($SelfTest) {
  $row=[pscustomobject]@{tensor='layer_017_ln1_square';count='2048';finite='2047';nan='0';pos_inf='1';neg_inf='0';min='0';max='61632';rms='3940';first_bad='1715';hash=('a'*64);private_path='SECRET'}
  $value=Select-Health $row
  if (($value|ConvertTo-Json) -match 'SECRET|private_path' -or $value.pos_inf -ne 1) { throw 'ALLOW_LIST_SELF_TEST_FAILED' }
  $row.tensor='input'; $rejected=$false
  try { Select-Health $row | Out-Null } catch { $rejected=$true }
  if (-not $rejected) { throw 'PRIVATE_TENSOR_NOT_REJECTED' }
  'public_g1_forensics_self_test=PASS'; exit 0
}
if (-not $PrivateIndex -or -not $Destination) { throw 'INDEX_AND_DESTINATION_REQUIRED' }
$base=[IO.Path]::GetFullPath((Join-Path $root 'build/reports/g1-long-linear-forensics'))
$index=Get-Content -LiteralPath $PrivateIndex -Raw | ConvertFrom-Json
$public=@()
foreach ($item in $index.runs) {
  if ($item.label -notmatch '^[A-Z][A-Z0-9_]{1,40}$' -or $item.directory -notmatch '^g1forensic-[a-f0-9]{32}$') { throw 'INVALID_PRIVATE_INDEX' }
  $dir=Join-Path $base $item.directory
  $p=Get-Content (Join-Path $dir 'forensic-plan.json') -Raw | ConvertFrom-Json
  if ($p.status -eq 'BLOCKED' -or $p.classification -eq 'BLOCKED_UNCLASSIFIED_FAILURE') { throw 'BLOCKED_EVIDENCE_CANNOT_SUPPORT_CLAIM' }
  if ($p.qairt_build_id -ne '2.48.40.260702151143' -or $p.order_hash -ne 'fnv1a64:83f9c96cf10261a0') { throw 'IDENTITY_MISMATCH' }
  $row=[ordered]@{label=$item.label; source_sha=$p.source_sha
    from_step=$p.from_step; to_step=$p.to_step; probe_microbatch=$p.probe_microbatch
    centered_scale_mode= $(if($null -eq $p.layernorm_centered_scale){'1'}else{[string]$p.layernorm_centered_scale})
    tap_scope=$p.tap_scope; classification=$p.classification
    checkpoint_sha256=$p.source_checkpoint_sha256; qairt_build_id=$p.qairt_build_id
    order_hash=$p.order_hash}
  $csv=Join-Path $dir 'forensic-failure.csv'
  if (Test-Path $csv) {
    $rows=Import-Csv $csv
    $row['health']=@($rows | Where-Object {$_.tensor -match '^layer_\d{3}_(ln[12]_(centered_s|square|safe_scale)|output)$'} | ForEach-Object {Select-Health $_})
    $logits=@($rows | Where-Object {$_.tensor -eq 'logits'})
    if ($logits.Count -ne 1 -or [int]$logits[0].count -ne 32768) { throw 'LOGITS_SHAPE_MISMATCH' }
    $row['logits_nan']=[int]$logits[0].nan
    $row['logits_canonical_sha256']=[string]$logits[0].hash
    $raw=Join-Path $dir 'forensic-logits.f32'
    if (Test-Path $raw) {$row['logits_raw_sha256']=(Get-FileHash $raw).Hash.ToLowerInvariant()}
  }
  $metadata=Join-Path $dir 'forensic-failure.txt'
  if (Test-Path $metadata) {
    $m=Get-Content $metadata -Raw
    foreach($key in @('first_detected_update','first_detected_microbatch','last_completed_finite_update','probe_update','probe_microbatch')) {
      $match=[regex]::Match($m,"(?m)^$key=(\d+)\s*$")
      if($match.Success){$row[$key]=[int]$match.Groups[1].Value}
    }
    $minimal=@([regex]::Matches($m,'(?m)^minimal_square input=([^ ]+) repeat=(\d+) cpu_fp32=([^ ]+) execute_ok=true qnn_return=0 count=1 finite=(\d+) pos_inf=(\d+) nan=0 output=([^ ]+) error=\s*$'))
    if($minimal.Count -gt 0){
      if($minimal.Count -ne 6){throw 'MINIMAL_REPRODUCER_INCOMPLETE'}
      $row['minimal_square']=@($minimal | ForEach-Object {[ordered]@{input=[double]$_.Groups[1].Value;repeat=[int]$_.Groups[2].Value;finite=[int]$_.Groups[4].Value;pos_inf=[int]$_.Groups[5].Value;qnn_return=0}})
    }
  }
  $resultFile=Join-Path $dir "seed1-l19-v1024-t32-d64-f128-steps$($p.to_step)-result.txt"
  if(Test-Path $resultFile){
    $r=Get-Content $resultFile -Raw
    foreach($key in @('forensic_first_scaled_update','forensic_first_scaled_microbatch','forensic_first_scaled_row','forensic_scaled_norm_rows')) {
      $match=[regex]::Match($r,"(?m)^$key=(-?\d+)\s*$")
      if($match.Success){$row[$key]=[long]$match.Groups[1].Value}
    }
    if($p.first_scaled_checkpoint_sha256){$row['first_scaled_checkpoint_sha256']=$p.first_scaled_checkpoint_sha256}
    foreach($key in @('api_trace_graph_execute_attempt_count','api_trace_graph_execute_success_count','api_trace_graph_execute_failure_count','generalized_tiny_training_output_nonfinite_count','generalized_tiny_training_output_poison_count')){
      $match=[regex]::Match($r,"(?m)^$key=(\d+)\s*$")
      if($match.Success){$row[$key]=[long]$match.Groups[1].Value}
    }
    $row['cpu_fallback_false']=($r -match '(?m)^cpu_fallback=false\s*$')
  }
  $public+=$row
}
$snapshots=@()
foreach($step in @(40000,41646)){
  $file=Join-Path $base "step$step-host-inspect.txt"
  if(Test-Path $file){
    $states=@(Get-Content $file | Where-Object {$_ -match '^all_(parameters|momentum|adam_m|adam_v),'} | ConvertFrom-Csv -Header state,count,nonfinite,max_abs,l2,raw_sha256)
    if($states.Count -ne 4){throw 'HOST_STATE_AGGREGATES_INCOMPLETE'}
    foreach($s in $states){
      if([int]$s.nonfinite -ne 0 -or $s.raw_sha256 -notmatch '^[a-f0-9]{64}$'){throw 'INVALID_HOST_STATE'}
      $snapshots+=[ordered]@{step=$step;state=$s.state;count=[int]$s.count;nonfinite=0;max_abs=[double]$s.max_abs;l2=[double]$s.l2;raw_f32_le_sha256=$s.raw_sha256}
    }
  }
}
$comparisons=@()
foreach($name in @('unscaled-prefix-repeat-compare','scaled-prefix-compare','adaptive-prefix-compare','adaptive-first-scaled-compare')) {
  $file=Join-Path $base "$name.txt"
  if(-not(Test-Path $file)){continue}
  $m=Get-Content $file -Raw
  if($m -notmatch '(?m)^status=SUCCESS\s*$'){throw 'COMPARISON_DECODE_FAILED'}
  $states=@()
  foreach($state in @('parameters','muon_momentum','aux_adam_m','aux_adam_v')) {
    $s=[ordered]@{state=$state}
    foreach($metric in @('max_abs','relative_l2','cosine')) {
      $match=[regex]::Match($m,"(?m)^${state}_${metric}=([^\r\n]+)")
      if(-not $match.Success){throw 'COMPARISON_AGGREGATE_MISSING'}
      $s[$metric]=[double]$match.Groups[1].Value
    }
    $states+=$s
  }
  $comparisons+=[ordered]@{label=$name.ToUpperInvariant().Replace('-','_');states=$states}
}
$rawParity=@()
foreach($name in @('healthy-logit-parity','healthy-gradient-parity','adaptive-healthy-logit-parity','adaptive-healthy-gradient-parity')) {
  $file=Join-Path $base "$name.json"
  if(-not(Test-Path $file)){continue}
  $v=Get-Content $file -Raw|ConvertFrom-Json
  $rawParity+=[ordered]@{label=$name.ToUpperInvariant().Replace('-','_');count=[int]$v.count;raw_bit_mismatch=[int]$v.raw_bit_mismatch;max_abs=[double]$v.max_abs;rms_difference=[double]$v.rms_difference;left_sha256=[string]$v.left_sha256;right_sha256=[string]$v.right_sha256}
}
$result=[ordered]@{schema='G1_FORENSICS_PUBLIC_AGGREGATES_V1'; runs=$public;state_snapshots=$snapshots;checkpoint_comparisons=$comparisons;raw_parity=$rawParity}
$text=$result|ConvertTo-Json -Depth 12
if ($text -match '(?i)g1forensic-|[A-Z]:[\\/]|device_identity|apk_sha256|run_id_private') { throw 'PRIVATE_MATERIAL_IN_EXPORT' }
[IO.Directory]::CreateDirectory((Split-Path -Parent ([IO.Path]::GetFullPath($Destination))))|Out-Null
[IO.File]::WriteAllText([IO.Path]::GetFullPath($Destination),$text+"`n",[Text.UTF8Encoding]::new($false))
'public_g1_forensics_export=PASS'
