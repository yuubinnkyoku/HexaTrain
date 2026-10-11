# SPDX-License-Identifier: Apache-2.0
# Export only numerical aggregates, never private paths, identifiers or tensors.
param([string]$PrivateIndex, [string]$Destination, [switch]$SelfTest)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
function Select-Health($row) {
  if ($row.tensor -notmatch '^layer_\d{3}_(ln[12]_(centered_s|square)|output)$') { throw 'TENSOR_NOT_ALLOW_LISTED' }
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
    centered_scale= $(if($null -eq $p.layernorm_centered_scale){1}else{[double]$p.layernorm_centered_scale})
    tap_scope=$p.tap_scope; classification=$p.classification
    checkpoint_sha256=$p.source_checkpoint_sha256; qairt_build_id=$p.qairt_build_id
    order_hash=$p.order_hash}
  $csv=Join-Path $dir 'forensic-failure.csv'
  if (Test-Path $csv) {
    $rows=Import-Csv $csv
    $row['health']=@($rows | Where-Object {$_.tensor -match '^layer_\d{3}_(ln[12]_(centered_s|square)|output)$'} | ForEach-Object {Select-Health $_})
    $logits=@($rows | Where-Object {$_.tensor -eq 'logits'})
    if ($logits.Count -ne 1 -or [int]$logits[0].count -ne 32768) { throw 'LOGITS_SHAPE_MISMATCH' }
    $row['logits_nan']=[int]$logits[0].nan
    $row['logits_canonical_sha256']=[string]$logits[0].hash
    $raw=Join-Path $dir 'forensic-logits.f32'
    if (Test-Path $raw) {$row['logits_raw_sha256']=(Get-FileHash $raw).Hash.ToLowerInvariant()}
  }
  $public+=$row
}
$result=[ordered]@{schema='G1_FORENSICS_PUBLIC_AGGREGATES_V1'; runs=$public}
$text=$result|ConvertTo-Json -Depth 12
if ($text -match '(?i)g1forensic-|[A-Z]:[\\/]|device_identity|apk_sha256|run_id_private') { throw 'PRIVATE_MATERIAL_IN_EXPORT' }
[IO.Directory]::CreateDirectory((Split-Path -Parent ([IO.Path]::GetFullPath($Destination))))|Out-Null
[IO.File]::WriteAllText([IO.Path]::GetFullPath($Destination),$text+"`n",[Text.UTF8Encoding]::new($false))
'public_g1_forensics_export=PASS'
