param([string]$ReportRoot,[string]$OutputPath,[string]$SourceCommit,[switch]$SelfTest)
$ErrorActionPreference='Stop'
$culture=[Globalization.CultureInfo]::InvariantCulture
function Number([string]$Value){$n=[double]::Parse($Value,$culture);if([double]::IsNaN($n)-or[double]::IsInfinity($n)){throw 'NONFINITE_METRIC'};return $n}
function Metric([string]$Line){
 $f=$Line.Split(',');if($f.Count-ne 13 -or $f[0]-ne'onemetric'){throw 'METRIC_SCHEMA'}
 $v=@{};foreach($field in $f[4..12]){$kv=$field.Split('=',2);if($kv.Count-ne 2 -or $v.ContainsKey($kv[0])){throw 'METRIC_FIELDS'};$v[$kv[0]]=$kv[1]}
 if(($v.Keys|Sort-Object)-join ',' -ne 'abs,fp_abs,gate,hash,limit,max_relative,nonfinite,relative_l2,rms'){throw 'METRIC_ALLOWLIST'}
 $id=[int]$f[2];$mode=$f[1];$tensor=$f[3];$fields=@('dc','dw','y','xhat','inverse','dx','dg','db','residual')
 if($mode-in@('integratedSplitNative','integratedSplitTap')-and$tensor-in@('h0_selected','h1_selected','h0_dc','h1_dc')){if($id-notin 0..23){throw 'CONNECTED_CASE'};$gate=if($tensor-match'selected') {.002}else{.01}}
 elseif($mode-in@('segmentNative','segmentTap','segmentFused','producerLocalCustom','integratedSplitNative','integratedSplitTap')){if($id-notin 0..23 -or $tensor-notin $fields){throw 'SEGMENT_CASE'};$gate=if($tensor-in@('dc','dw')){.03}elseif($tensor-in@('y','xhat','inverse')){.002}else{.02}}
 elseif($mode-eq'accumulation'){if($id-notin 0..2 -or $tensor-ne'dw'){throw 'ACCUM_CASE'};$gate=.03}
 elseif($mode-in@('selector_direct','selector_native','selector_tap','selector_nativeOutput','splitNative','splitTap')){if($id-notin 0..23 -or $tensor-notin@('h0_selected','h1_selected')){throw 'SELECTOR_CASE'};$gate=.002}
 elseif($mode-in@('nextGateNative','nextGateTap')){if($id-notin 0..23 -or $tensor-notin@('h0_dc','h1_dc')){throw 'GATE_CASE'};$gate=.01}
 elseif($mode-eq'upstream'){if($id-notin 0..23 -or $tensor-notin@('producer_x','producer_dy','producer_context','projection_q','projection_k','projection_v')){throw 'PRODUCER_CASE'};$gate=if($tensor-eq'producer_dy'){.01}else{.002}}
 else{throw 'METRIC_MODE'}
 foreach($key in @('abs','rms','relative_l2','max_relative','fp_abs','limit')){$v[$key]=Number $v[$key];if($v[$key]-lt 0){throw 'NEGATIVE_ERROR'}}
 if($v.limit-ne $gate -or $v.nonfinite-ne'0' -or $v.hash-notmatch'^[0-9a-f]{16}$' -or $v.gate-notin@('0','1') -or [int]$v.gate-ne[int]($v.abs-le$gate)){throw 'METRIC_GATE'}
 return [pscustomobject]@{mode=$mode;index=$id;tensor=$tensor;abs=$v.abs;rms=$v.rms;relative_l2=$v.relative_l2;max_relative=$v.max_relative;fp_abs=$v.fp_abs;hash=$v.hash;limit=$gate;pass=[int]$v.gate}
}
function Median($Values){$v=@($Values|Sort-Object);if(!$v.Count){throw 'TIMING_MISSING'};if($v.Count%2){return $v[[int]($v.Count/2)]};return ($v[$v.Count/2-1]+$v[$v.Count/2])/2}
function Hash([string]$Text){$sha=[Security.Cryptography.SHA256]::Create();try{return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}}
if($SelfTest){
 $good='onemetric,segmentNative,0,dc,abs=0.001,rms=0.001,relative_l2=0.001,max_relative=0.001,fp_abs=0,nonfinite=0,hash=1234567890abcdef,limit=0.03,gate=1'
 if((Metric $good).pass-ne 1){throw 'SELFTEST_GOOD'}
 foreach($bad in @($good.Replace('abs=0.001','abs=NaN'),$good.Replace('gate=1','gate=0'),$good.Replace('limit=0.03','limit=0.04'),$good.Replace('nonfinite=0','nonfinite=1'),($good+',private_path=secret'),$good.Replace('segmentNative','unknown'),$good.Replace(',0,dc,',',24,dc,'))){$rejected=$false;try{Metric $bad|Out-Null}catch{$rejected=$true};if(!$rejected){throw 'SELFTEST_FAIL_CLOSED'}}
 Write-Host 'g1_frontier_export_selftest=PASS allowlist/gate/nonfinite/mode/case';return
}
if(!$ReportRoot-or !$OutputPath-or$SourceCommit-notmatch'^[0-9a-f]{40}$'){throw 'EXPLICIT_PATHS_AND_SOURCE_REQUIRED'}
$processes=@();foreach($directory in Get-ChildItem -LiteralPath $ReportRoot -Directory -Filter 'customfp32-*'){
 $identityPath=Join-Path $directory.FullName 'identity-private.json';if(!(Test-Path $identityPath)){continue};$ledger=Get-Content $identityPath -Raw|ConvertFrom-Json
 if($ledger.variant-ne'oneFinal' -or $ledger.exit_code-ne 0-or$ledger.build.source_sha-ne$SourceCommit){continue}
 $lines=@(Get-Content (Join-Path $directory.FullName 'stdout.txt'))
 if($ledger.mode-ne'BACKGROUND_CORRECTNESS'-or$ledger.focus_takeover_count-ne 0-or$ledger.protected_identity-ne'PASS'-or$ledger.build.qairt_build_id-ne'2.48.40.260702151143'){throw 'PROCESS_IDENTITY_OR_SAFETY'}
 if($ledger.build.source_sha-notmatch'^[0-9a-f]{40}$' -or $ledger.hashes.'one-layer-inputs.bin'-ne'388f4d8b30b86997d2f03f732e5a3210f7e2b3015af883579e3f0659df9d9f4f'){throw 'SOURCE_OR_FROZEN_FIXTURE_IDENTITY'}
 foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe','libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so','libQnnHtpV81Skel.so')){if($ledger.hashes.$name-notmatch'^[0-9a-f]{64}$'){throw 'REQUIRED_HASH_MISSING'}}
 foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe')){if($ledger.hashes.$name-ne$ledger.build.binary_hashes.$name){throw 'STAGED_BUILD_IDENTITY'}}
 if(@($lines|Where-Object {$_-match'^api,[^,]+,(?!0$)'}).Count -or @($lines|Where-Object {$_-cmatch'^(?:[A-Z_]*RECEIPT_FAIL|[A-Z_]*WORD_FAIL|[A-Z_]*OBSERVER_FAIL|[A-Z_]*NONFINITE[A-Z_]*|INPUT_MUTATION|CANARY_FAIL|POISON_FAIL)'}).Count){throw 'PROCESS_API_OR_OWNERSHIP'}
 if(@($lines|Where-Object {$_-match'^api,graphExecute,0$'}).Count-ne 411 -or @($lines|Where-Object {$_-match'^oneownership,'}).Count-ne 171 -or @($lines|Where-Object {$_-match'^mmkernel,dcontext,'}).Count-ne 171 -or @($lines|Where-Object {$_-match'^mmkernel,dwo,'}).Count-ne 171-or@($lines|Where-Object {$_-match'^connected_head,'}).Count-ne48){throw 'EXECUTION_MATRIX_INCOMPLETE'}
 $summary=@($lines|Where-Object {$_-match'^summary,oneFinal,'});if($summary.Count-ne 1-or$summary[0]-notmatch'^summary,oneFinal,segment_failures=0,selector_failures=\d+,prototype_failures=0,producer_local_custom_failures=0,accumulation_verified=1,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0$'){throw 'REQUIRED_GATE_INCOMPLETE'}
 $metrics=@($lines|Where-Object {$_-match'^onemetric,'}|ForEach-Object {Metric $_});if($metrics.Count-ne2019){throw 'METRIC_MATRIX_INCOMPLETE'}
 $keys=[Collections.Generic.HashSet[string]]::new();foreach($m in $metrics){if(!$keys.Add("$($m.mode),$($m.index),$($m.tensor)")){throw 'DUPLICATE_METRIC'}}
 if(@($metrics|Where-Object {$_.mode-in@('segmentNative','segmentTap','segmentFused','producerLocalCustom','accumulation','splitNative','splitTap')-and!$_.pass}).Count){throw 'CUSTOM_OR_ACCUM_ACCURACY'}
 if(@($metrics|Where-Object {$_.mode-in@('integratedSplitNative','integratedSplitTap')-and$_.tensor-notin@('h0_dc','h1_dc')-and!$_.pass}).Count){throw 'CONNECTED_CUSTOM_ACCURACY'}
 if(!@($metrics|Where-Object {$_.mode-eq'upstream'-and!$_.pass}).Count-or!@($metrics|Where-Object {$_.mode-eq'nextGateNative'-and!$_.pass}).Count){throw 'FRONTIER_BLOCKERS_NOT_REPRODUCED'}
 if(@($lines|Where-Object {$_-match'^oneaccum,'}).Count-ne 3 -or @($lines|Where-Object {$_-match'^oneaccum,'-and$_-notmatch'registry_index=204,registry_offset=674688,name=layer_018.wo,role=MUON,rows=64,cols=64,microbatches=8,unique_inputs=8,scale=0.125,fp_word_mismatch=0,.*updates=0$'}).Count){throw 'REGISTRY_ACCUM_CONTRACT'}
 if(@($lines|Where-Object {$_-match'^selectorfrontier,'}).Count-ne192-or@($lines|Where-Object {$_-match'^splitfrontier,'}).Count-ne96-or@($lines|Where-Object {$_-match'^splitfrontier,.*word_mismatch=(?!0,)'}).Count){throw 'FRONTIER_RECEIPT_MATRIX'}
 $canonical=($lines|Where-Object {$_-match'^(?:onemetric|selectorfrontier|splitfrontier|prototype_summary|producer_summary|connected_head|connected_head_summary|summary),'})-join "`n"
 $performance=@();foreach($mode in @('segmentNative','segmentTap','segmentFused')){$timing=@($lines|Where-Object {$_-match"^oneperf,$mode,"}|ForEach-Object {$f=$_.Split(',');$v=@{};foreach($x in $f[3..5]){$kv=$x.Split('=');$v[$kv[0]]=Number $kv[1]};[pscustomobject]@{rep=[int]$f[2];wall_us=$v.wall_us;dc_us=$v.dc_us;dw_us=$v.dw_us}});if($timing.Count-ne9){throw 'PERF_MATRIX'};$steady=@($timing|Where-Object rep -gt 0);$performance+=[ordered]@{mode=$mode;steady_samples=8;first_timed_wall_us=$timing[0].wall_us;wall_median_us=(Median $steady.wall_us);dc_median_us=(Median $steady.dc_us);dw_median_us=(Median $steady.dw_us)}}
 $processes+=[pscustomobject]@{ledger=$ledger;lines=$lines;metrics=$metrics;digest=(Hash $canonical);performance=$performance}
}
if($processes.Count-ne3){throw 'THREE_INDEPENDENT_PROCESSES_REQUIRED'}
$a=$processes[0];foreach($p in $processes){if($p.digest-ne$a.digest-or$p.ledger.build.source_sha-ne$a.ledger.build.source_sha-or$p.ledger.device_private.Serial-ne$a.ledger.device_private.Serial){throw 'INDEPENDENT_IDENTITY_MISMATCH'};foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe','one-layer-inputs.bin','libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so','libQnnHtpV81Skel.so')){if($p.ledger.hashes.$name-ne$a.ledger.hashes.$name){throw 'EXACT_BINARY_MISMATCH'}}}
$aggregate=@($a.metrics|Group-Object mode,tensor|ForEach-Object {$g=$_.Group;[ordered]@{mode=$g[0].mode;tensor=$g[0].tensor;cases=$g.Count;max_absolute=($g.abs|Measure-Object -Maximum).Maximum;max_rms=($g.rms|Measure-Object -Maximum).Maximum;max_relative_l2=($g.relative_l2|Measure-Object -Maximum).Maximum;max_relative=($g.max_relative|Measure-Object -Maximum).Maximum;max_fp_absolute=($g.fp_abs|Measure-Object -Maximum).Maximum;absolute_gate=$g[0].limit;failures=@($g|Where-Object pass -eq 0).Count}})
$binary=[ordered]@{};foreach($name in @('libHexaFp32Dsp.so','libHexaFp32Host.so','one_layer_probe','libQnnHtp.so','libQnnHtpPrepare.so','libQnnHtpV81Stub.so','libQnnHtpV81Skel.so')){$binary[$name]=$a.ledger.hashes.$name}
$output=[ordered]@{schema='G1_FP32_FRONTIER_V1';source_commit=$a.ledger.build.source_sha;branch=$a.ledger.branch;qairt_build_id='2.48.40.260702151143';qnn_api='2.37.0';htp='V81';hexagon_sdk='6.6.0.0';hexagon_tools='19.0.07';ndk='26.2.11394342';binary_sha256=$binary;source_sha256=$a.ledger.build.source_hashes;fixture_sha256=$a.ledger.hashes.'one-layer-inputs.bin';numerical_digest=$a.digest;independent_processes=3;graph_executes_per_process=411;custom_executions_per_process=1095;metric_checks_per_process=2019;api_failures=0;nonfinite=0;fallback=0;input_mutation=0;poison_remaining=0;canary_failures=0;focus_takeovers=0;training_updates=0;results=$aggregate;performance=@($processes|ForEach-Object {$_.performance});classification=@('ONE_LAYER_LN2_SEGMENT_VERIFIED','GRADIENT_ACCUMULATION_VERIFIED','BLOCKED_ATTENTION_BACKWARD_PRECISION','BLOCKED_UPSTREAM_PRECISION');full_attention_observer_effect='UNRESOLVED';one_layer_backward_verified=$false;training='NO-GO'}
$json=$output|ConvertTo-Json -Depth 12
if($json-match'(?i)([A-Z]:[\\/]|/data/|device_private|device_serial|checkpoint_path|apk_sha|stdout|stderr)'){throw 'PUBLIC_LEAK_CHECK'}
[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),$json,[Text.UTF8Encoding]::new($false));Write-Host 'g1_frontier_public_export=PASS three_process_exact_identity/all_required_gates'
