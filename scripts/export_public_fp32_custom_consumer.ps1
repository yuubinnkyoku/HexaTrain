param(
 [string]$PrivateRoot='build/reports/htp-fp32-custom-consumer',
 [string]$OutputPath='build/reports/htp-fp32-custom-consumer/public-summary.json',
 [string]$BuildSourceSha,
 [switch]$SelfTest
)
$ErrorActionPreference='Stop'
function Number([string]$text){
 $v=[double]::Parse($text,[Globalization.CultureInfo]::InvariantCulture)
 if([double]::IsNaN($v) -or [double]::IsInfinity($v) -or $v -lt 0){throw 'INVALID_PUBLIC_NUMBER'}
 return $v
}
function Parse-Metric([string]$line){
 $m=[regex]::Match($line,'^metric,(Copy|Add|Multiply|Affine|chain|reshape|htpAdd|htpMultiply|standardAdd|standardMultiply),([a-z0-9_]+),(dx|dg|db|ln_tail_residual),fp_abs=([^,]+),double_abs=([^,]+),bits=(\d+),gate=([01])$')
 if(!$m.Success){throw 'UNRECOGNIZED_TYPED_METRIC'}
 $recipes=@('ordinary','small_centered','low_variance','epsilon_dominated','constant','large_mean_low_variance',
 'boundary_below','boundary_equal','boundary_above','known_failure','large_signed','half_scale_limit','very_large_signed','large_mean',
 'sparse_known_failure','sparse_large','random_seed1','random_seed7','random_seed31','random_seed97')
 if($m.Groups[2].Value -notin $recipes){throw 'RECIPE_NOT_ALLOWLISTED'}
 return [pscustomobject]@{route=$m.Groups[1].Value;recipe=$m.Groups[2].Value;field=$m.Groups[3].Value;
  fp_abs=(Number $m.Groups[4].Value);double_abs=(Number $m.Groups[5].Value);bits=[int]$m.Groups[6].Value;gate=[int]$m.Groups[7].Value}
}
if($SelfTest){
 $m=Parse-Metric 'metric,Add,low_variance,dx,fp_abs=0,double_abs=0.00001,bits=0,gate=1'
 if($m.double_abs -ne .00001){throw 'METRIC_PARSE_SELFTEST'}
 foreach($bad in @('metric,Add,low_variance,dx,fp_abs=NaN,double_abs=0,bits=0,gate=1','metric,Add,private_identifier,raw_tensor,fp_abs=0,double_abs=0,bits=0,gate=1')){
  $rejected=$false;try{Parse-Metric $bad|Out-Null}catch{$rejected=$true};if(!$rejected){throw 'ALLOWLIST_FAIL_CLOSED_SELFTEST'}
 }
 Write-Host 'fp32_custom_exporter_selftest=PASS';exit 0
}
if($BuildSourceSha -notmatch '^[0-9a-f]{40}$'){throw 'EXPLICIT_BUILD_SOURCE_REQUIRED'}
$variants=@('Copy','Add','Multiply','Affine','chain','chainTap','reshape','htpAdd','htpMultiply','standardAdd','standardMultiply')
$all=@()
foreach($directory in Get-ChildItem -LiteralPath $PrivateRoot -Directory){
 $identity=Join-Path $directory.FullName 'identity-private.json'
 if(!(Test-Path $identity)){continue}
 $j=Get-Content $identity -Raw|ConvertFrom-Json
 if(!$j.build -or $j.build.source_sha -ne $BuildSourceSha){continue}
 if($j.variant -notin $variants -or $j.focus_takeover_count -ne 0 -or $j.mode -ne 'BACKGROUND_CORRECTNESS' -or $j.protected_identity -ne 'PASS'){throw 'IDENTITY_FAIL_CLOSED'}
 $lines=@(Get-Content (Join-Path $directory.FullName 'stdout.txt'))
 $metrics=@($lines|Where-Object{$_ -match '^metric,'}|ForEach-Object{Parse-Metric $_})
 $expected=if($j.variant -in @('Affine','chain','chainTap')){80}else{60}
 if($metrics.Count -ne $expected -or @($metrics|Select-Object -ExpandProperty recipe -Unique).Count -ne 20){throw 'REQUIRED_FIXTURES_MISSING'}
 if(@($metrics|Group-Object recipe,field|Where-Object Count -ne 1).Count){throw 'DUPLICATE_METRIC'}
 $standard=$j.variant -in @('htpAdd','htpMultiply','standardAdd','standardMultiply')
 $failures=@($metrics|Where-Object gate -eq 0).Count
 if(!$standard -and ($failures -ne 0 -or $j.exit_code -ne 0 -or @($metrics|Where-Object bits -ne 0).Count)){throw 'CUSTOM_ACCURACY_FAILURE'}
 if(!$standard -and @($lines -match '^summary,\w+,required_failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0$').Count -ne 1){throw 'CUSTOM_SAFETY_SUMMARY_MISSING'}
 if(!$standard -and @($metrics|Where-Object {$_.double_abs -gt .02}).Count){throw 'CUSTOM_DOUBLE_GATE_FAILED'}
 if($standard -and $j.exit_code -ne 4){throw 'EXPECTED_STANDARD_FAILURE_MISSING'}
 $perf=@($lines|Where-Object{$_ -match '^perf,'}|ForEach-Object{if($_ -match ',(\d+),execute_us=([0-9.]+)$' -and [int]$Matches[1] -gt 0){Number $Matches[2]}})
 $median=$null;if($perf.Count){$s=@($perf|Sort-Object);$median=($s[15]+$s[16])/2}
 $apiTimes=@{};foreach($line in $lines){if($line -match '^api_us,([^,]+),([0-9.]+)$' -and !$apiTimes.ContainsKey($Matches[1])){$apiTimes[$Matches[1]]=Number $Matches[2]}}
 $kernel=@($lines|Where-Object{$_ -match '^kernel_receipt,'}|ForEach-Object{if($_ -match 'kernel_us=(\d+)$'){[int]$Matches[1]}})
 $hashLines=@($lines|Where-Object{$_ -match '^io_hash,'})
 if(!$hashLines.Count){throw 'INPUT_OUTPUT_IDENTITY_MISSING'}
 $sha=[Security.Cryptography.SHA256]::Create();try{$digest=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes(($hashLines -join "`n"))))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
 $binaryIdentity=$j.build.binary_hashes|ConvertTo-Json -Compress
 $all+=[pscustomobject]@{variant=$j.variant;metrics=$metrics;failures=$failures;median=$median;api=$apiTimes;kernel=$kernel;io_digest=$digest;binary_identity=$binaryIdentity;private_identity=$j.device_private.Serial}
}
$out=[ordered]@{schema='FP32_CUSTOM_CONSUMER_V1';build_source_sha=$BuildSourceSha;required_recipes=20;required_gradient_max_abs=.02;routes=@()}
foreach($v in $variants){
 $runs=@($all|Where-Object variant -eq $v)
 if($runs.Count -ne 3){throw "THREE_INDEPENDENT_PROCESSES_REQUIRED_$v"}
 if(@($runs.io_digest|Select-Object -Unique).Count -ne 1 -or @($runs.binary_identity|Select-Object -Unique).Count -ne 1 -or @($runs.private_identity|Select-Object -Unique).Count -ne 1){throw 'PROCESS_IDENTITY_MISMATCH'}
 $fields=@();foreach($field in @('dx','dg','db','ln_tail_residual')){
  $values=@($runs.metrics|Where-Object field -eq $field);if(!$values.Count){continue}
  $fields+=[ordered]@{field=$field;max_fp_abs=($values.fp_abs|Measure-Object -Maximum).Maximum;max_double_abs=($values.double_abs|Measure-Object -Maximum).Maximum;
   max_bits=($values.bits|Measure-Object -Maximum).Maximum;failed_checks=@($values|Where-Object gate -eq 0).Count}
 }
 $out.routes+=[ordered]@{variant=$v;processes=3;io_digest=$runs[0].io_digest;fields=$fields;required_failure_count=$runs[0].failures;
  steady_execute_median_us=@($runs.median);first_execute_us=@($runs|ForEach-Object{$_.api.graphExecute});
  graph_finalize_us=@($runs|ForEach-Object{$_.api.graphFinalize});kernel_receipt_inclusive_us_min=($runs.kernel|Measure-Object -Minimum).Minimum;
  kernel_receipt_inclusive_us_max=($runs.kernel|Measure-Object -Maximum).Maximum}
}
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),($out|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
Write-Host 'fp32_custom_public_export=PASS'
