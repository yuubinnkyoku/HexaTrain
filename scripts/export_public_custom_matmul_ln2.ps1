param(
 [string]$PrivateRoot='build/reports/htp-fp32-custom-matmul-ln2',
 [string]$OutputPath='build/reports/htp-fp32-custom-matmul-ln2/public-summary.json',
 [ValidateSet('Independent','Chain')][string]$Phase='Chain',
 [switch]$SelfTest
)
$ErrorActionPreference='Stop'
function Number([string]$s){$v=[double]::Parse($s,[Globalization.CultureInfo]::InvariantCulture);if(![double]::IsFinite($v)-or$v-lt 0){throw 'INVALID_NUMBER'};return $v}
function Median($values){$sorted=@($values|Sort-Object);if(!$sorted.Count){throw 'MISSING_TIMING'};return ($sorted[[int][math]::Floor(($sorted.Count-1)/2)]+$sorted[[int][math]::Floor($sorted.Count/2)])/2}
function Metric([string]$line){
 $p=$line-split',';if($p[0]-notin@('custommm','chainmm','chainln')){throw 'UNKNOWN_METRIC'}
 $r=[ordered]@{kind=$p[0]};$r.path=$p[1];$r.variant=$p[2];$r.recipe=$p[3];$r.control=Number $p[4];$r.additional=Number $p[5]
 if($r.path-notin@('dcontext','dwo','ln')-or$r.variant-notin@('direct','native','tap','standard','standardNative','fused')-or$r.additional-notin@(0,1)){throw 'INVALID_IDENTITY'}
 $recipes=@('ordinary','small_centered','low_variance','epsilon_dominated','constant','large_mean_low_variance','boundary_below','boundary_equal','boundary_above','known_failure','large_signed','half_scale_limit','very_large_signed','large_mean','sparse_known_failure','sparse_large','random_seed1','random_seed7','random_seed31','random_seed97')
 if($r.additional-eq 0){if($r.recipe-notin$recipes-or$r.control-notin@(0,1,2)){throw 'INVALID_RECIPE'}}else{if($r.recipe-notmatch'^independent32_[0-7]$'-or$r.control-ne[int]$r.recipe.Substring(14)){throw 'INVALID_ADDITIONAL_RECIPE'}}
 $start=6;if($r.kind-eq'chainln'){$r.field=$p[6];if($r.field-notin@('y','xhat','inverse','dx','dg','db','residual')){throw 'INVALID_FIELD'};$start=7}
 $allowed=@('abs','rms','relative_l2','max_relative','nonfinite','fp_abs','ulp','fp_word_mismatch','input_hash','hash','gate','host_validation_us','relative','host_abs')
 for($i=$start;$i-lt$p.Count;++$i){$pair=$p[$i]-split'=';if($pair.Count-ne 2-or$pair[0]-notin$allowed-or$r.Contains($pair[0])){throw 'FIELD_NOT_ALLOWLISTED'}
  if($pair[0]-in@('hash','input_hash')){if($pair[1]-notmatch'^[0-9a-f]{16}$'){throw 'INVALID_DIGEST'};$r[$pair[0]]=$pair[1]}else{$r[$pair[0]]=Number $pair[1]}
 }
 foreach($required in @('abs','rms','hash','gate')){if(!$r.Contains($required)){throw 'MISSING_METRIC'}}
 $limit=if($r.kind-ne'chainln'){.03}elseif($r.field-eq'inverse'){.002}elseif($r.field-in@('dx','dg','db','residual')){.02}elseif($r.recipe-in@('low_variance','epsilon_dominated','constant','large_mean_low_variance')){.06}else{.002}
 if($r['kind']-ne'chainln'-and(!$r.Contains('nonfinite')-or$r['nonfinite']-ne 0)){throw 'MISSING_FINITE_OR_NONFINITE'}
 if($r.gate-notin@(0,1)-or[bool]$r.gate-ne[bool]($r.abs-le$limit)){throw 'GATE_MISMATCH'}
 return [pscustomobject]$r
}
function Digest($items){$canonical=@($items|Sort-Object kind,path,variant,recipe,control,field|ForEach-Object{$m=$_;($m.psobject.Properties|Where-Object Name -ne 'host_validation_us'|Sort-Object Name|ForEach-Object{"$($_.Name)=$($_.Value)"})-join'|'});$sha=[Security.Cryptography.SHA256]::Create();try{return [Convert]::ToHexString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical-join"`n"))).ToLowerInvariant()}finally{$sha.Dispose()}}
if($SelfTest){
 $good='custommm,dcontext,direct,ordinary,0,0,abs=0,rms=0,nonfinite=0,hash=0000000000000000,gate=1'
 Metric $good|Out-Null
 foreach($bad in @($good.Replace('abs=0','abs=NaN'),$good.Replace('gate=1','gate=0'),$good.Replace('nonfinite=0','nonfinite=1'),($good+',private_path=0'),$good.Replace('ordinary','unknown'),$good.Replace('hash=0000000000000000','hash=wrong'))){$rejected=$false;try{Metric $bad|Out-Null}catch{$rejected=$true};if(!$rejected){throw ('SELFTEST_FAIL_CLOSED: '+$bad)}}
 if((Digest @((Metric ($good+',host_validation_us=1'))))-ne(Digest @((Metric ($good+',host_validation_us=9'))))){throw 'TIMING_DIGEST_SELFTEST'}
 Write-Host 'custom_matmul_exporter_selftest=PASS';exit 0
}
$frozen=@{};foreach($line in Get-Content (Join-Path $PrivateRoot 'fixture-audit.txt')){if($line-match'^frozenfixture,([^,]+),([^,]+),(\d),(\d),input_hash=([a-f0-9]{16}),ln_hash=([a-f0-9]{16}),other_hash=([a-f0-9]{16})$'){$key="$($Matches[1])/$($Matches[2])/$($Matches[3])/$($Matches[4])";if($frozen.ContainsKey($key)){throw 'DUPLICATE_FIXTURE'};$frozen[$key]=@($Matches[5],$Matches[6],$Matches[7])}}
if($frozen.Count-ne 136-or(Get-FileHash (Join-Path $PrivateRoot 'frozen-inputs.bin')).Hash.ToLowerInvariant()-ne'daf61450486f6f8df40b7b10e631c268e9f5e2e5f2458e3a1eda4b0cb8c9978e'){throw 'FROZEN_FIXTURE_MISMATCH'}
$modes=if($Phase-eq'Chain'){@('mmDirect','mmBoundary','mmLn2')}else{@('mmDirect','mmBoundary')}
$runs=@();$identity=$null;$seenPaths=@();$first=$null
foreach($mode in $modes){$digests=@();for($repeat=1;$repeat-le 3;++$repeat){
 $launch=Get-Content (Join-Path $PrivateRoot "final-$mode-$repeat-launch.txt") -Raw;$match=[regex]::Match($launch,'private_report=([^\r\n]+)');if(!$match.Success){throw 'REPORT_MISSING'};$path=$match.Groups[1].Value
 if($path-in$seenPaths){throw 'PROCESS_REUSED'};$seenPaths+=$path
 $id=Get-Content (Join-Path $path 'identity-private.json') -Raw|ConvertFrom-Json
 $binary=($id.hashes.psobject.Properties|Sort-Object Name|ForEach-Object{"$($_.Name)=$($_.Value)"})-join'|'
 $sources=($id.build.source_hashes.psobject.Properties|Sort-Object Name|ForEach-Object{"$($_.Name)=$($_.Value)"})-join'|'
 $current="$binary/$sources/$($id.build.source_sha)/$($id.build.hexagon_compiler_sha256)/$($id.build.ndk_compiler_sha256)/$($id.device_private.Serial)"
 if(!$identity){$identity=$current;$first=$id};if($current-ne$identity-or$id.exit_code-ne 0-or$id.variant-ne$mode-or$id.mode-ne'BACKGROUND_CORRECTNESS'-or$id.focus_takeover_count-ne 0-or$id.protected_identity-ne'PASS'-or$id.status-ne'COMPLETE'-or$id.build.qairt_build_id-ne'2.48.40.260702151143'){throw 'COHORT_IDENTITY_OR_SAFETY_MISMATCH'}
 foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','layernorm_probe')){if($id.hashes.$name-ne$id.build.binary_hashes.$name){throw 'STAGED_BINARY_MISMATCH'}}
 $lines=@(Get-Content (Join-Path $path 'stdout.txt'));$metrics=@($lines|Where-Object{$_-match'^(custommm|chainmm|chainln),'}|ForEach-Object{Metric $_})
 $expected=if($mode-eq'mmDirect'){136}elseif($mode-eq'mmBoundary'){680}else{1836}
 if($metrics.Count-ne$expected-or@($metrics|Group-Object kind,path,variant,recipe,control,additional,field|Where-Object Count -ne 1).Count){throw 'MISSING_OR_DUPLICATE_METRICS'}
 if(@($metrics|Where-Object{$_.variant-notin@('standard','standardNative')-and$_.gate-ne 1}).Count){throw 'ACCURACY_FAIL'}
 $expectedVariants=if($mode-eq'mmDirect'){@('direct')}elseif($mode-eq'mmBoundary'){@('direct','native','tap','standard','standardNative')}else{@('native','tap','fused')}
 foreach($variant in $expectedVariants){foreach($path in @('dcontext','dwo')){foreach($additional in @(0,1)){$group=@($metrics|Where-Object{$_.path-eq$path-and$_.variant-eq$variant-and$_.additional-eq$additional});$expectedCount=if($additional){8}else{60};if($group.Count-ne$expectedCount){throw 'PATH_VARIANT_MATRIX_INCOMPLETE'}}}}
 if(@($lines|Where-Object{$_-match'^api,'-and$_-notmatch',0$'}).Count){throw 'API_FAIL'}
 $summary=@($lines-match"^summary,$mode,required_failures=0,");if($summary.Count-ne 1-or$summary[0]-notmatch'cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0$'){throw 'SUMMARY_FAIL'}
 foreach($m in $metrics){$key="$($m.path)/$($m.recipe)/$($m.control)/$($m.additional)";if($m.kind-eq'custommm'-and(!$frozen.ContainsKey($key)-or$m.input_hash-ne$frozen[$key][0])){throw 'DEVICE_FROZEN_INPUT_MISMATCH'}}
 if($mode-eq'mmLn2'){$fixtureLines=@($lines-match'^chainfixture,');if($fixtureLines.Count-ne 408){throw 'CHAIN_FIXTURE_COUNT'};foreach($line in $fixtureLines){$p=$line-split',';$key="$($p[1])/$($p[3])/$($p[4])/$($p[5])";if(!$frozen.ContainsKey($key)-or$p[6]-ne('ln_hash='+$frozen[$key][1])-or$p[7]-ne('other_hash='+$frozen[$key][2])){throw 'CHAIN_INPUT_MISMATCH'}}}
 $digest=Digest $metrics;$digests+=$digest
 $perf=@($lines|Where-Object{$_-match'^(mmperf|chainperf),'}|ForEach-Object{$p=$_-split',';$r=[ordered]@{path=$p[1];variant=$p[2];index=Number $p[3]};foreach($s in $p[4..($p.Count-1)]){$k=$s-split'=';if($k[0]-notin@('wall_us','kernel_us','receipt_us','dcontext_us','dwo_us','ln_us','add_us')){throw 'PERF_FIELD'};$r[$k[0]]=Number $k[1]};[pscustomobject]$r})
 $firstExecute=@();$nextIsFirst=$false;foreach($line in $lines){if($line-eq'api,graphCreate,0'){$nextIsFirst=$true}elseif($nextIsFirst-and$line-match'^api_us,graphExecute,([0-9.]+)$'){$firstExecute+=Number $Matches[1];$nextIsFirst=$false}}
 $runs+=[pscustomobject]@{mode=$mode;repeat=$repeat;metrics=$metrics;digest=$digest;perf=$perf;executes=@($lines-match'^api,graphExecute,0$').Count;finalize_us=@($lines|Where-Object{$_-match'^api_us,graphFinalize,'}|ForEach-Object{Number (($_-split',')[2])});create_us=@($lines|Where-Object{$_-match'^api_us,graphCreate,'}|ForEach-Object{Number (($_-split',')[2])});first_execute_us=$firstExecute;allocation_median_us=(Median @($lines|Where-Object{$_-match'^allocation_copy_us,'}|ForEach-Object{Number (($_-split',')[1])}));validation_median_us=(Median $metrics.host_validation_us);temperature_before=(Number ([regex]::Match($id.battery_before,'temperature:\s*(\d+)').Groups[1].Value))/10;temperature_after=(Number ([regex]::Match($id.battery_after,'temperature:\s*(\d+)').Groups[1].Value))/10}
 };if(@($digests|Select-Object -Unique).Count-ne 1){throw 'INDEPENDENT_PROCESS_DIVERGENCE'}
}
$out=[ordered]@{schema='CUSTOM_FP32_MATMUL_LN2_V1';phase=$Phase;source_sha=$first.build.source_sha;fixture_sha256='daf61450486f6f8df40b7b10e631c268e9f5e2e5f2458e3a1eda4b0cb8c9978e';package_sha256=$first.build.binary_hashes.'libHexaFp32Dsp.so';probe_sha256=$first.build.binary_hashes.layernorm_probe;compiler_sha256=$first.build.hexagon_compiler_sha256;qairt_build_id=$first.build.qairt_build_id;processes=$runs.Count;api_executes=($runs.executes|Measure-Object -Sum).Sum;training_updates=0;results=@();timing=@();setup=@();temperature_c_min=(($runs.temperature_before+$runs.temperature_after|Measure-Object -Minimum).Minimum);temperature_c_max=(($runs.temperature_before+$runs.temperature_after|Measure-Object -Maximum).Maximum)}
if($out.source_sha-notmatch'^[0-9a-f]{40}$'){throw 'INVALID_SOURCE_SHA'}
$out.source_file_sha256=[ordered]@{}
foreach($name in @('host_tests/fp32_custom/package.cpp','host_tests/fp32_custom/probe.cpp','host_tests/fp32_custom/layernorm_probe.cpp','host_tests/fp32_custom/layernorm_ops.h','host_tests/fp32_custom/probe_support.h','host_tests/fp32_custom/matmul_probe.h','host_tests/fp32_custom/custom_matmul_ops.h','host_tests/fp32_custom/custom_matmul_probe.h','host_tests/fp32_custom/custom_matmul_fixture.h','host_tests/fp32_custom/ln2_matmul_probe.h','host_tests/hvx_rpc/fp32_boundary_fixture.h','app/src/main/cpp/qnn/layernorm_stability_reference.h','app/src/main/cpp/qnn/layernorm_redesign_reference.h')){$hash=$first.build.source_hashes.$name;if($hash-notmatch'^[0-9a-f]{64}$'){throw 'SOURCE_FILE_HASH_MISSING'};$out.source_file_sha256[$name]=$hash}
foreach($name in @('package_sha256','probe_sha256','compiler_sha256')){if($out[$name]-notmatch'^[0-9a-f]{64}$'){throw 'BUILD_HASH_INVALID'}}
foreach($mode in $modes){$selected=@($runs|Where-Object mode -eq $mode);foreach($group in ($selected[0].metrics|Group-Object kind,path,variant,additional,field)){$items=$group.Group;$r=[ordered]@{mode=$mode;kind=$items[0].kind;path=$items[0].path;variant=$items[0].variant;additional=$items[0].additional;field=$items[0].field;checks=$items.Count;failures=@($items|Where-Object gate -ne 1).Count;max_abs=($items.abs|Measure-Object -Maximum).Maximum;max_rms=($items.rms|Measure-Object -Maximum).Maximum;digest=$selected[0].digest};if($items[0].kind-ne'chainln'){$r.max_fp_abs=($items.fp_abs|Measure-Object -Maximum).Maximum;$r.max_ulp=($items.ulp|Measure-Object -Maximum).Maximum};$out.results+=$r}
 foreach($run in $selected){foreach($group in ($run.perf|Where-Object index -gt 0|Group-Object path,variant)){$r=[ordered]@{mode=$mode;repeat=$run.repeat;path=$group.Group[0].path;variant=$group.Group[0].variant};foreach($field in @('wall_us','kernel_us','receipt_us','dcontext_us','dwo_us','ln_us','add_us')){if($group.Group[0].psobject.Properties.Name-contains$field){$values=@($group.Group.$field|Sort-Object);$r[$field+'_median']=($values[3]+$values[4])/2}};$out.timing+=$r};$out.setup+=[ordered]@{mode=$mode;repeat=$run.repeat;graph_create_us=$run.create_us;graph_finalize_us=$run.finalize_us;first_graph_execute_us=$run.first_execute_us;host_allocation_copy_median_us=$run.allocation_median_us;host_matmul_validation_median_us=$run.validation_median_us}}
}
if($Phase-eq'Chain'){$copyDigests=@();for($repeat=1;$repeat-le3;++$repeat){$launch=Get-Content (Join-Path $PrivateRoot "final-copyNative-$repeat-launch.txt") -Raw;$path=[regex]::Match($launch,'private_report=([^\r\n]+)').Groups[1].Value;if(!$path-or$path-in$seenPaths){throw 'COPY_PROCESS_MISSING_OR_REUSED'};$seenPaths+=$path;$id=Get-Content (Join-Path $path 'identity-private.json') -Raw|ConvertFrom-Json;
 if($id.variant-ne'copyNative'-or$id.exit_code-ne0-or$id.focus_takeover_count-ne0-or$id.mode-ne'BACKGROUND_CORRECTNESS'-or$id.protected_identity-ne'PASS'-or$id.build.source_sha-ne$out.source_sha-or$id.device_private.Serial-ne$first.device_private.Serial){throw 'COPY_IDENTITY_MISMATCH'}
 foreach($p in $id.hashes.psobject.Properties){if($p.Value-ne$first.hashes.($p.Name)){throw 'COPY_BINARY_RUNTIME_MISMATCH'}}
 $lines=@(Get-Content (Join-Path $path 'stdout.txt'));$typed=@($lines-match'^copycontrol,');if($typed.Count-ne20-or@($typed|Where-Object{$_-notmatch'^copycontrol,[a-z0-9_]+,words=2048,bit_mismatch=0,hash=[a-f0-9]{16}$'}).Count-or@($lines-match'^summary,copyNative,required_failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0$').Count-ne1-or@($lines|Where-Object{$_-match'^api,'-and$_-notmatch',0$'}).Count){throw 'COPY_CONTROL_FAIL'}
 $sha=[Security.Cryptography.SHA256]::Create();try{$copyDigests+=[Convert]::ToHexString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($typed-join"`n"))).ToLowerInvariant()}finally{$sha.Dispose()}
 };if(@($copyDigests|Select-Object -Unique).Count-ne1){throw 'COPY_PROCESS_DIVERGENCE'};$out.copy_control=[ordered]@{source_sha=$out.source_sha;processes=3;cases_per_process=20;words_per_case=2048;bit_mismatch=0;digest=$copyDigests[0]}}
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),($out|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false));Write-Host 'custom_matmul_public_export=PASS'
