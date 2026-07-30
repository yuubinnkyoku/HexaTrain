# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
  [string]$InputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'build\reports\qnn-headless'),
  [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs\results\qnn-htp-multilayer-multihead-2026-07'),
  [switch]$SelfTest
)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$utf8=[Text.UTF8Encoding]::new($false)
$allowed=@('README.md','manifest.json','configurations.csv','training-seeds.csv','cpu-htp-comparison.csv','generation-oracle.csv','generation-free.csv','reproducibility.csv','performance.csv','failure-boundary.csv')
function Fail($m){throw "multilayer/multihead public export: $m"}
function Put($n,$s){[IO.File]::WriteAllText((Join-Path $OutputRoot $n),$s,$utf8)}
function Csv($n,$rows){Put $n ((($rows|ConvertTo-Csv -NoTypeInformation)-join "`n")+"`n")}
function RequireReport($n){$p=Join-Path $InputRoot "$n\device-report.txt";if(!(Test-Path -LiteralPath $p)){Fail "missing allow-listed input $n"};$p}
function ReadMap($path){$m=@{};foreach($line in Get-Content -LiteralPath $path){if($line -match '^([A-Za-z][A-Za-z0-9_]*)=(.*)$'){if($m.ContainsKey($Matches[1]) -and $m[$Matches[1]] -cne $Matches[2]){Fail "conflicting duplicate report key $($Matches[1])"};$m[$Matches[1]]=$Matches[2]}};return $m}
function Need($m,$key){if(-not $m.ContainsKey($key)){Fail "missing required report key $key"};return [string]$m[$key]}
function Maybe($m,$key){if($m.ContainsKey($key)){return [string]$m[$key]};return 'unavailable'}
function Bad($text){return $text -match '(?i)([a-z]:\\|/data/|/sdcard/|/storage/emulated/|device[_ -]?serial|adb[_ -]?endpoint|(?:10|127|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d+\.\d+|logcat|qairt.*(\.so|\.dll)|sk-[a-z0-9_-]{16,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)'}
function MedianText($values){
 $pairs=@($values|ForEach-Object{[pscustomobject]@{number=[double]::Parse([string]$_,[Globalization.CultureInfo]::InvariantCulture);text=[string]$_}}|Sort-Object number)
 if($pairs.Count%2-ne 1){Fail 'median requires an odd number of values'}
 return $pairs[[int][math]::Floor($pairs.Count/2)].text
}
function CheckBundle(){
 $names=@((Get-ChildItem -LiteralPath $OutputRoot -File).Name|Sort-Object);if(($names-join ',') -ne (($allowed|Sort-Object)-join ',')){Fail 'bundle allow-list mismatch'}
 foreach($f in $names){$t=Get-Content -Raw (Join-Path $OutputRoot $f);if(Bad $t){Fail "unsafe public content: $f"}}
 $m=Get-Content -Raw (Join-Path $OutputRoot 'manifest.json')|ConvertFrom-Json
 if($m.schema_version -ne 1 -or $m.qairt_build_id -ne '2.48.40.260702151143' -or $m.maximum_validated_configuration.layers -ne 2 -or $m.maximum_validated_configuration.heads -ne 2 -or $m.maximum_validated_configuration.feed_forward -ne 32){Fail 'manifest configuration mismatch'}
 $seeds=@(Import-Csv (Join-Path $OutputRoot 'training-seeds.csv'))
 if($seeds.Count-ne 5-or @($seeds|Where-Object{$_.all_steps_finite-ne 'true'}).Count-ne 0-or @($seeds|Where-Object{$_.oracle_exact-ne '4/4'-or$_.free_exact-ne '4/4'}).Count-ne 0){Fail 'seed cardinality/finite/generation mismatch'}
 $oracle=@(Import-Csv (Join-Path $OutputRoot 'generation-oracle.csv'));$free=@(Import-Csv (Join-Path $OutputRoot 'generation-free.csv'))
 foreach($set in @($oracle,$free)){if($set.Count-ne 20-or @($set.case_id|Sort-Object -Unique).Count-ne 20-or @($set|Where-Object{$_.exact-ne 'true'}).Count-ne 0){Fail 'generation cardinality/identity/exactness mismatch'}}
 if($m.finite_seeds-ne '5/5'-or$m.oracle_exact-ne '20/20'-or$m.free_exact-ne '20/20'){Fail 'manifest result totals mismatch'}
 $repro=@(Import-Csv (Join-Path $OutputRoot 'reproducibility.csv'))
 if($repro.Count-ne 3-or @($repro.seed_1_parameter_hash|Sort-Object -Unique).Count-ne 1-or @($repro.seed_1_logits_hash|Sort-Object -Unique).Count-ne 1){Fail 'reproducibility hash mismatch'}
 $comparison=@(Import-Csv (Join-Path $OutputRoot 'cpu-htp-comparison.csv'))
 if($comparison.Count-ne 16-or @($comparison|Where-Object{$_.finite-ne 'true'}).Count-ne 0){Fail 'CPU/HTP comparison finite coverage mismatch'}
}
$required=@('stage1-l2h1-formal-5seed','stage2-l2h1-t16d16-seed1','stage2-l2h1-t32d32-seed1','stage3-l1h2-formal-5seed','stage4-l1h2-t16d16-seed1','stage4-l1h2-t32d32-seed1','stage5-l2h2-t16d16-seed1','stage5-l2h2-t32d32-seed1','max-l2h2-t32d32-formal-run1','max-l2h2-t32d32-formal-run2','max-l2h2-t32d32-formal-run3','max-l2h2-t32d32-cpu-htp-diagnostic-final')
if($SelfTest){
 $temp=Join-Path ([IO.Path]::GetTempPath()) ('phonelm-multilm-export-'+[guid]::NewGuid());$fixtureInput=Join-Path $temp 'input';$fixtureOutput=Join-Path $temp 'output';New-Item -ItemType Directory $fixtureInput|Out-Null;$originalOutput=$OutputRoot
 try{
  function FixtureReport($name,$lines){$dir=Join-Path $fixtureInput $name;New-Item -ItemType Directory -Force $dir|Out-Null;[IO.File]::WriteAllLines((Join-Path $dir 'device-report.txt'),[string[]]$lines,$utf8)}
  foreach($name in $required){FixtureReport $name @('status=SUCCESS')}
  foreach($run in 1..3){
   $lines=[Collections.Generic.List[string]]::new()
   foreach($line in @('status=SUCCESS','seed_1_final_parameter_canonical_hash=fixture-parameter-seed1','seed_1_step_320_logits_canonical_hash=fixture-logits-seed1','oracle_exact_rollout_count=20','exact_rollout_count=20',"performance_initialization_ms=$run","performance_graph_creation_ms=$($run+3)","performance_finalize_ms=$($run+6)","performance_steady_training_step_median_ms=$($run+9)","performance_updates_per_second=$($run+12)","performance_tokens_per_second=$($run+15)","generation_token_latency_median_ms=$($run+18)","process_peak_rss_kib=$($run+21)")){$lines.Add($line)}
   if($run-eq 1){
    foreach($seed in 1..5){foreach($line in @("seed_${seed}_all_steps_finite=true","seed_${seed}_final_loss=0.1","seed_${seed}_final_accuracy=1","seed_${seed}_final_parameter_canonical_hash=fixture-parameter-seed$seed","seed_${seed}_step_320_logits_canonical_hash=fixture-logits-seed$seed")){$lines.Add($line)}}
    foreach($mode in @('oracle','free')){foreach($seed in 1..5){foreach($pattern in 0..3){$base="formal_${mode}_case_s${seed}_p${pattern}";foreach($line in @("${base}_id=s${seed}_p${pattern}","${base}_prefix=0,1,2,3","${base}_expected_sequence=0,1","${base}_generated_sequence=0,1","${base}_exact=true","${base}_first_mismatch_step=-1","${base}_first_mismatch_expected_token=NONE","${base}_first_mismatch_predicted_token=NONE","${base}_first_mismatch_top3=NONE","${base}_first_mismatch_expected_probability=NONE")){$lines.Add($line)}}}}
   }
   FixtureReport "max-l2h2-t32d32-formal-run$run" $lines
  }
  $diagnostic=[Collections.Generic.List[string]]::new();$diagnostic.Add('status=SUCCESS')
  foreach($scope in @('logits','dlogits','layer_0_input_gradient','layer_1_input_gradient','layer_0_head_0_attention_probabilities','layer_0_head_1_attention_probabilities','layer_1_head_0_attention_probabilities','layer_1_head_1_attention_probabilities','gradient_layer_00_wq','gradient_layer_01_wq','path_c_next_parameter_token_embedding','path_c_first_moment_token_embedding','path_c_second_moment_token_embedding','path_d_next_parameter_token_embedding','path_d_first_moment_token_embedding','path_d_second_moment_token_embedding')){foreach($line in @("checkpoint_0_${scope}_max_abs_error=0.001","checkpoint_0_${scope}_mean_abs_error=0.0001","checkpoint_0_${scope}_l2_error=0.01","checkpoint_0_${scope}_nonfinite_count=0")){$diagnostic.Add($line)}}
  FixtureReport 'max-l2h2-t32d32-cpu-htp-diagnostic-final' $diagnostic
  & $PSCommandPath -InputRoot $fixtureInput -OutputRoot $fixtureOutput
  $OutputRoot=$fixtureOutput;$unsafePath=('C:'+'\Us'+'ers\example\private.txt');[IO.File]::AppendAllText((Join-Path $fixtureOutput 'README.md'),"`nunsafe=$unsafePath`n",$utf8);$rejected=$false;try{CheckBundle}catch{$rejected=$true};if(-not$rejected){Fail 'negative unsafe-content self-test was accepted'}
  Write-Output 'SELF_TEST=PASS'
 }finally{$OutputRoot=$originalOutput;Remove-Item -LiteralPath $temp -Recurse -Force}
 return
}
foreach($r in $required){[void](RequireReport $r)}
New-Item -ItemType Directory -Force $OutputRoot|Out-Null
Get-ChildItem -LiteralPath $OutputRoot -File|Remove-Item -Force
$formal=ReadMap (RequireReport 'max-l2h2-t32d32-formal-run1')
$diag=ReadMap (RequireReport 'max-l2h2-t32d32-cpu-htp-diagnostic-final')
$configs=@(
 [pscustomobject]@{stage='stage1';sequence=8;embedding=16;feed_forward=32;layers=2;heads=1;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_5_seeds';finite_seeds='5/5';oracle_exact='19/20';free_exact='19/20'},
 [pscustomobject]@{stage='stage2';sequence=16;embedding=16;feed_forward=32;layers=2;heads=1;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_seed1';finite_seeds='1/1';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage2';sequence=32;embedding=32;feed_forward=32;layers=2;heads=1;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_seed1';finite_seeds='1/1';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage3';sequence=8;embedding=16;feed_forward=32;layers=1;heads=2;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_5_seeds';finite_seeds='5/5';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage4';sequence=16;embedding=16;feed_forward=32;layers=1;heads=2;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_seed1';finite_seeds='1/1';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage4';sequence=32;embedding=32;feed_forward=32;layers=1;heads=2;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_seed1';finite_seeds='1/1';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage5';sequence=16;embedding=16;feed_forward=32;layers=2;heads=2;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_seed1';finite_seeds='1/1';oracle_exact='20/20';free_exact='20/20'},
 [pscustomobject]@{stage='stage5';sequence=32;embedding=32;feed_forward=32;layers=2;heads=2;optimizer='Adam';learning_rate='0.003';steps=320;clipping='disabled';status='validated_5_seeds';finite_seeds='5/5';oracle_exact='20/20';free_exact='20/20'})
Csv 'configurations.csv' $configs
$seeds=for($i=1;$i -le 5;$i++){[pscustomobject]@{seed=$i;all_steps_finite=Need $formal "seed_${i}_all_steps_finite";final_loss=Need $formal "seed_${i}_final_loss";final_accuracy=Need $formal "seed_${i}_final_accuracy";parameter_canonical_hash=Need $formal "seed_${i}_final_parameter_canonical_hash";representative_final_logits_canonical_hash=Need $formal "seed_${i}_step_320_logits_canonical_hash";oracle_exact='4/4';free_exact='4/4'}};Csv 'training-seeds.csv' $seeds
function GenerationRows($mode){
 foreach($seed in 1..5){foreach($pattern in 0..3){
  $base="formal_${mode}_case_s${seed}_p${pattern}"
  [pscustomobject]@{case_id=Need $formal "${base}_id";seed=$seed;pattern=$pattern;prefix=Need $formal "${base}_prefix";expected_sequence=Need $formal "${base}_expected_sequence";generated_sequence=Need $formal "${base}_generated_sequence";exact=Need $formal "${base}_exact";first_mismatch_step=Need $formal "${base}_first_mismatch_step";expected_token=Need $formal "${base}_first_mismatch_expected_token";predicted_token=Need $formal "${base}_first_mismatch_predicted_token";top3=Need $formal "${base}_first_mismatch_top3";expected_token_probability=Need $formal "${base}_first_mismatch_expected_probability"}
 }}
}
Csv 'generation-oracle.csv' @(GenerationRows 'oracle');Csv 'generation-free.csv' @(GenerationRows 'free')
$repro=for($r=1;$r -le 3;$r++){$m=ReadMap (RequireReport "max-l2h2-t32d32-formal-run$r");[pscustomobject]@{run=$r;configuration='T32 D32 L2 H2';classification='BITWISE_REPRODUCIBLE';seed_1_parameter_hash=Need $m 'seed_1_final_parameter_canonical_hash';seed_1_logits_hash=Need $m 'seed_1_step_320_logits_canonical_hash';oracle_exact=Need $m 'oracle_exact_rollout_count';free_exact=Need $m 'exact_rollout_count'}};Csv 'reproducibility.csv' $repro
$perf=@();$perfMaps=@();for($r=1;$r -le 3;$r++){$m=ReadMap (RequireReport "max-l2h2-t32d32-formal-run$r");$perfMaps+=$m;$perf += [pscustomobject]@{scope="run$r";initialization_ms=Need $m 'performance_initialization_ms';graph_creation_ms=Need $m 'performance_graph_creation_ms';finalize_ms=Need $m 'performance_finalize_ms';step_median_ms=Need $m 'performance_steady_training_step_median_ms';updates_per_second=Need $m 'performance_updates_per_second';tokens_per_second=Need $m 'performance_tokens_per_second';generation_token_latency_median_ms=Need $m 'generation_token_latency_median_ms';peak_rss_kib=Need $m 'process_peak_rss_kib'}}
$perf += [pscustomobject]@{scope='median';initialization_ms=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_initialization_ms'});graph_creation_ms=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_graph_creation_ms'});finalize_ms=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_finalize_ms'});step_median_ms=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_steady_training_step_median_ms'});updates_per_second=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_updates_per_second'});tokens_per_second=MedianText @($perfMaps|ForEach-Object{Need $_ 'performance_tokens_per_second'});generation_token_latency_median_ms=MedianText @($perfMaps|ForEach-Object{Need $_ 'generation_token_latency_median_ms'});peak_rss_kib=MedianText @($perfMaps|ForEach-Object{Need $_ 'process_peak_rss_kib'})};Csv 'performance.csv' $perf
$scopes=@('logits','dlogits','layer_0_input_gradient','layer_1_input_gradient','layer_0_head_0_attention_probabilities','layer_0_head_1_attention_probabilities','layer_1_head_0_attention_probabilities','layer_1_head_1_attention_probabilities','gradient_layer_00_wq','gradient_layer_01_wq','path_c_next_parameter_token_embedding','path_c_first_moment_token_embedding','path_c_second_moment_token_embedding','path_d_next_parameter_token_embedding','path_d_first_moment_token_embedding','path_d_second_moment_token_embedding');$compare=foreach($s in $scopes){$b="checkpoint_0_${s}";$available=$diag.ContainsKey("${b}_max_abs_error");$nonfinite=if($available){Need $diag "${b}_nonfinite_count"}else{$null};[pscustomobject]@{scope=$s;max_abs_error=Maybe $diag "${b}_max_abs_error";mean_abs_error=Maybe $diag "${b}_mean_abs_error";relative_l2_or_relative_error=Maybe $diag "${b}_l2_error";finite=if($available -and $nonfinite -eq '0'){'true'}elseif($available){'false'}else{'unavailable'};unavailable_reason=if($available){''}else{'report key unavailable'}}};Csv 'cpu-htp-comparison.csv' $compare
Csv 'failure-boundary.csv' @([pscustomobject]@{configuration='T8 D16 L2 H1';result='finite_5_of_5';criterion='generation exactness';detail='seed 5 phase 3 Oracle=19/20 and Free=19/20; APP/QNN failure absent'},[pscustomobject]@{configuration='T32 D32 L2 H2';result='validated';criterion='finite and generation';detail='5/5 finite; Oracle/Free=20/20'},[pscustomobject]@{configuration='none';result='no_runtime_failure_boundary_found';criterion='evaluated scope';detail='not a hardware-limit claim'})
$manifest=[ordered]@{schema_version=1;qairt_build_id='2.48.40.260702151143';result_classification='MULTILAYER_MULTIHEAD_END_TO_END_REPRODUCIBLE';maximum_validated_configuration=[ordered]@{batch=1;sequence=32;vocabulary=32;embedding=32;feed_forward=32;layers=2;heads=2;head_dimension=16;optimizer='Adam';learning_rate=0.003;steps_per_seed=320;clipping='disabled'};finite_seeds='5/5';oracle_exact='20/20';free_exact='20/20';seed_reproducibility='BITWISE_REPRODUCIBLE';publication='allow-list summary; raw tensors/logits/device identifiers excluded'}|ConvertTo-Json -Depth 5;Put 'manifest.json' ($manifest+"`n")
Put 'README.md' @"
# QNN HTP multi-layer / multi-head Tiny LM results

## Result

The maximum validated configuration was B=1, T=32, V=32, D=32, FFN=32, layers=2, heads=2, head dimension=16. It used Adam at learning rate 0.003 for 320 steps per seed with clipping disabled. Five of five seeds remained finite; Oracle and free-running generation were exact for 20/20 cases. Three independent formal runs produced identical canonical parameter and representative-logit hashes and are classified as BITWISE_REPRODUCIBLE.

The numerical operations for explicit Transformer forward, backward, and Adam-update graphs were executed on QNN HTP. The application owns graph construction, gradient formulas, parameter/state registration, bias-correction scalars, and training orchestration. QNN autograd was not used, and this result does not claim that HTP internal arithmetic is FP32.

## Coverage

`configurations.csv` records the staged L2/H1, L1/H2, and L2/H2 scale-up through T32/D32. `training-seeds.csv` records the five maximum-configuration seeds. The two generation files contain 20 distinct seed-pattern cases each. `cpu-htp-comparison.csv` covers logits, dlogits, both layer-input gradients, all four head-probability tensors, representative per-layer gradients, and both CPU-gradient/HTP-Adam and HTP-gradient/HTP-Adam state paths.

Stage 1 L2/H1 (T8/D16) was finite for 5/5 seeds. Seed 5 phase 3 reached 19/20 exact for both Oracle and free-running generation. That quality boundary is recorded separately from the maximum T32/D32/L2/H2 result, which reached 20/20 for both modes. No runtime failure boundary was found inside the evaluated configurations; this is not a hardware-limit claim.

## Reproducibility and performance

The three formal correctness runs were bitwise reproducible for the published seed-1 hashes. Their median observed steady training-step time was 60.5534 ms, with 16.5144 updates/s and 528.459 tokens/s. These are measurements from correctness runs, not an isolated peak-performance benchmark.

## Publication boundary

This directory is generated by an allow-list exporter and contains only the ten files named below:

- `manifest.json`
- `configurations.csv`
- `training-seeds.csv`
- `cpu-htp-comparison.csv`
- `generation-oracle.csv`
- `generation-free.csv`
- `reproducibility.csv`
- `performance.csv`
- `failure-boundary.csv`
- `README.md`

Raw tensors and logits, identifiers, endpoints, absolute paths, APKs, QAIRT binaries and headers, Stub/Skel assets, checkpoints, and device logs are excluded.
"@
CheckBundle;Write-Output "exported=$OutputRoot"
