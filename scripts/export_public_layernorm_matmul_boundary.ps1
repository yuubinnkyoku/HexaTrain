param(
 [string]$PrivateRoot='build/reports/htp-fp32-layernorm-matmul',
 [string]$OutputPath='build/reports/htp-fp32-layernorm-matmul/public-summary.json',
 [switch]$SelfTest
)
$ErrorActionPreference='Stop'
function N([string]$text){$v=[double]::Parse($text,[Globalization.CultureInfo]::InvariantCulture);if([double]::IsNaN($v)-or[double]::IsInfinity($v)-or$v-lt 0){throw 'INVALID_NUMBER'};return $v}
function Parse-Line([string]$line){
 $s=$line-split ',';$out=[ordered]@{}
 if($s[0]-eq'lnmetric'){
  if($s[1]-notin@('forward','backward','residual')-or$s[2]-notin@('8','16','32','64','128')-or$s[3]-notin@('2','32')-or$s[5]-notin@('y','xhat','inverse','dx','dg','db','result')){throw 'INVALID_LN_RECORD'}
  $out.kind='ln';$out.lane=$s[1];$out.d=[int]$s[2];$out.rows=[int]$s[3];$out.recipe=$s[4];$out.field=$s[5];$start=6
 }elseif($s[0]-eq'mmmetric'){
  if($s[1]-notin@('dwo','dcontext','dw2','drelu','dembedding')-or$s[2]-notin@('0','1')-or$s[4]-notin@('0','1','2')){throw 'INVALID_MM_RECORD'}
  $out.kind='mm';$out.path=$s[1];$out.copy=[int]$s[2];$out.recipe=$s[3];$out.control=[int]$s[4];$start=5
 }else{throw 'UNRECOGNIZED_RECORD'}
 $recipes=@('ordinary','small_centered','low_variance','epsilon_dominated','constant','large_mean_low_variance','boundary_below','boundary_equal','boundary_above','known_failure','large_signed','half_scale_limit','very_large_signed','large_mean','sparse_known_failure','sparse_large','random_seed1','random_seed7','random_seed31','random_seed97')
 if($out.recipe-notin$recipes){throw 'RECIPE_NOT_ALLOWLISTED'}
 $allowed=@('abs','rms','relative','host_abs','finite','hash','gate','m','k','n','fp_abs','ideal_abs','input_half_model','output_half_model','both_half_model','input_hash','extra002')
 for($j=$start;$j-lt$s.Count;++$j){$pair=$s[$j]-split '=';if($pair.Count-ne 2-or$pair[0]-notin$allowed-or$out.Contains($pair[0])){throw 'FIELD_NOT_ALLOWLISTED'}
  if($pair[0]-in@('hash','input_hash')){if($pair[1]-notmatch'^[0-9a-f]{16}$'){throw 'INVALID_HASH'};$out[$pair[0]]=$pair[1]}else{$out[$pair[0]]=N $pair[1]}
 }return [pscustomobject]$out
}
if($SelfTest){
 $good=Parse-Line 'lnmetric,forward,64,2,ordinary,y,abs=0,rms=0,relative=0,host_abs=0,finite=128,hash=0000000000000000,gate=1'
 if($good.finite-ne 128){throw 'PARSE_SELFTEST'}
 foreach($bad in @('lnmetric,forward,64,2,ordinary,y,abs=NaN','lnmetric,forward,64,2,ordinary,y,private_id=12')){$failed=$false;try{Parse-Line $bad|Out-Null}catch{$failed=$true};if(!$failed){throw 'FAIL_CLOSED_SELFTEST'}}
 Write-Host 'layernorm_matmul_exporter_selftest=PASS';exit 0
}
$variants=@('lnMatrix','lnResidual','lnResidualTap','lnResidualFused','matmul');$all=@();$binary=$null;$device=$null;$source=$null;$runtime=$null
foreach($v in $variants){
 $digests=@()
 for($r=1;$r-le 3;++$r){
  $launch=Get-Content (Join-Path $PrivateRoot "final-$v-$r-launch.txt") -Raw
  $match=[regex]::Match($launch,'private_report=([^\r\n]+)');if(!$match.Success){throw 'FINAL_REPORT_MISSING'}
  $path=$match.Groups[1].Value;$id=Get-Content (Join-Path $path 'identity-private.json') -Raw|ConvertFrom-Json
  $identity=$id.build.binary_hashes|ConvertTo-Json -Compress
  $libraryIdentity=($id.hashes.psobject.Properties|Sort-Object Name|ForEach-Object{"$($_.Name)=$($_.Value)"})-join"`n"
  if(!$binary){$binary=$identity;$device=$id.device_private.Serial;$source=$id.build.source_sha;$runtime=$libraryIdentity}
  if($identity-ne$binary-or$libraryIdentity-ne$runtime-or$id.device_private.Serial-ne$device-or$id.build.source_sha-ne$source-or$id.variant-ne$v-or$id.focus_takeover_count-ne 0-or$id.protected_identity-ne'PASS'-or$id.mode-ne'BACKGROUND_CORRECTNESS'){throw 'IDENTITY_MISMATCH'}
  $lines=@(Get-Content (Join-Path $path 'stdout.txt'));$typed=@($lines|Where-Object{$_-match'^(lnmetric|mmmetric),'});$metrics=@($typed|ForEach-Object{Parse-Line $_})
  $expected=if($v-eq'lnMatrix'){1200}elseif($v-eq'matmul'){600}else{280};if($metrics.Count-ne$expected){throw 'CASE_COUNT_MISMATCH'}
  if($v-ne'matmul'-and($id.exit_code-ne 0-or@($metrics|Where-Object gate -ne 1).Count)){throw 'LN_ACCURACY_FAIL'}
  if($v-eq'matmul'-and$id.exit_code-ne 4){throw 'MATMUL_INFRASTRUCTURE_OR_CLASSIFICATION_FAILURE'}
  $groups=if($v-eq'matmul'){@($metrics|Group-Object path,copy,control,recipe)}else{@($metrics|Group-Object d,rows,recipe,field)}
  if(@($groups|Where-Object Count -ne 1).Count){throw 'DUPLICATE_RECORD'}
  if(@($lines|Where-Object{$_-match'^api,'-and$_-notmatch',0$'}).Count){throw 'API_FAILURE'}
  foreach($m in $metrics){$limit=if($m.kind-eq'mm'){.03}elseif($m.field-eq'inverse'){.002}elseif($m.field-in@('dx','dg','db','result')){.02}elseif($m.recipe-in@('low_variance','epsilon_dominated','constant','large_mean_low_variance')){.06}else{.002};if([bool]$m.gate-ne[bool]($m.abs-le$limit)){throw 'METRIC_GATE_MISMATCH'}}
  if(@($lines-match'^summary,\w+,required_failures=\d+,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0$').Count-ne 1){throw 'SUMMARY_MISSING'}
  $sha=[Security.Cryptography.SHA256]::Create();try{$digest=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes(($typed-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()};$digests+=$digest
  $perf=@($lines|Where-Object{$_-match'^perf,'}|ForEach-Object{$a=$_-split',';[pscustomobject]@{d=[int]$a[2];rows=[int]$a[3];index=[int]$a[4];us=(N $a[5])}})
  $kernel=@($lines|Where-Object{$_-match'^kernel_receipt,'}|ForEach-Object{if($_-match'^kernel_receipt,(\d+),\d+,kernel_us=(\d+)$'){[pscustomobject]@{kind=[int]$Matches[1];us=[int]$Matches[2]}}})
  $rssMatch=[regex]::Match(($lines-join"`n"),'peak_rss_kib,(\d+)');$rss=if($rssMatch.Success){N $rssMatch.Groups[1].Value}else{$null}
  $all+=[pscustomobject]@{variant=$v;repeat=$r;metrics=$metrics;digest=$digest;perf=$perf;kernel=$kernel;api_calls=@($lines-match'^api,graphExecute,0$').Count;temperature_before=(N ([regex]::Match($id.battery_before,'temperature:\s*(\d+)').Groups[1].Value));temperature_after=(N ([regex]::Match($id.battery_after,'temperature:\s*(\d+)').Groups[1].Value));rss=$rss}
 }
 if(@($digests|Select-Object -Unique).Count-ne 1){throw 'INDEPENDENT_PROCESS_DIVERGENCE'}
}
$out=[ordered]@{schema='DSP_LN_MATMUL_BOUNDARY_V1';source_sha=$source;processes=15;api_executes=($all.api_calls|Measure-Object -Sum).Sum;lane_a=@();lane_b=@();timing=@();temperature_c_min=(($all.temperature_before+$all.temperature_after|Measure-Object -Minimum).Minimum/10);temperature_c_max=(($all.temperature_before+$all.temperature_after|Measure-Object -Maximum).Maximum/10)}
foreach($v in $variants){$runs=@($all|Where-Object variant -eq $v);$ms=$runs[0].metrics
 if($v-eq'matmul'){foreach($group in ($ms|Group-Object path,copy)){$out.lane_b+=[ordered]@{path=$group.Group[0].path;copy=$group.Group[0].copy;checks=$group.Count;failures=@($group.Group|Where-Object gate -eq 0).Count;max_abs=($group.Group.abs|Measure-Object -Maximum).Maximum;max_fp_abs=($group.Group.fp_abs|Measure-Object -Maximum).Maximum;max_ideal_abs=($group.Group.ideal_abs|Measure-Object -Maximum).Maximum;both_half_model_max=($group.Group.both_half_model|Measure-Object -Maximum).Maximum;digest=$runs[0].digest}}
 }else{foreach($group in ($ms|Group-Object d,rows,field)){$out.lane_a+=[ordered]@{variant=$v;d=$group.Group[0].d;rows=$group.Group[0].rows;field=$group.Group[0].field;checks=$group.Count;max_abs=($group.Group.abs|Measure-Object -Maximum).Maximum;max_host_abs=($group.Group.host_abs|Measure-Object -Maximum).Maximum;digest=$runs[0].digest}}
  foreach($run in $runs){foreach($group in ($run.perf|Where-Object index -gt 0|Group-Object d,rows)){$s=@($group.Group.us|Sort-Object);$out.timing+=[ordered]@{variant=$v;repeat=$run.repeat;d=$group.Group[0].d;rows=$group.Group[0].rows;median_us=($s[3]+$s[4])/2;rss_kib=$run.rss}}}
 }
}
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),($out|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false));Write-Host 'layernorm_matmul_public_export=PASS'
