"""Allow-list resume evidence; never exports raw logs, paths or device IDs."""
import argparse, collections, hashlib, json, pathlib, re, tempfile
NUM=set('kind s2 observed tap set case field maxAbs rms word_mismatches hash gate'.split())
EXEC=set('kind s2 observed tap set case dsp_marker native_consumer_words_checked execute_us arithmetic_us receipt_inclusive_us'.split())
PERF=set('kind s2 observed tap samples finalize_us first_execute_us steady_median_us p10_us p90_us arithmetic_median_us receipt_inclusive_median_us nodes tensors native_logical_bytes host_buffer_lower_bound_bytes explicit_center_scratch_bytes'.split())
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def safety_waivers(pre,ledger):
 result=[]
 for gate,request,invocation in (('host_storage','host_storage_waiver_requested','host_storage_waiver'),('external_power','external_power_waiver_requested','external_power_waiver')):
  state=pre['gates'].get(gate)
  require(state in ('PASS','WAIVED_BY_EXPLICIT_USER_FOR_CURRENT_RESUME'),'required safety gate')
  if state.startswith('WAIVED'):
   require(pre.get(request) is True and ledger.get(invocation) is True,'explicit waiver evidence')
   result.append(gate)
 return result
def thermal_evidence(condition,explicit=False):
 keys=('thermal_status','battery_temperature_c','battery_level','battery_health','battery_present')
 require(all(k in condition for k in keys),'thermal evidence missing')
 require(0<=condition['thermal_status']<5 and 0<condition['battery_temperature_c']<=60 and 0<=condition['battery_level']<=100 and condition['battery_health']==2 and condition['battery_present'] is True,'invalid or emergency thermal evidence')
 normal=condition['thermal_status']<=1 and condition['battery_temperature_c']<=40 and condition['battery_level']>=30
 require(normal or explicit,'unsafe thermal evidence')
 return {**{k:condition[k] for k in keys},'gate':'PASS' if normal else 'WAIVED_BY_EXPLICIT_USER_FOR_CURRENT_RESUME'}
def thermal_ledger(pre,ledger):
 explicit=ledger.get('power_thermal_waiver') is True
 if explicit:require(pre.get('power_thermal_waiver_requested') is True,'thermal authorization evidence')
 return {k:thermal_evidence(ledger[k],explicit) for k in ('thermal_before','thermal_after')}
def process(p):
 ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig');pre=read(p/'preflight/device-preflight-private.json')
 require(pre['status']=='PASS' and ledger['focus_takeover_count']==0 and ledger['mode']=='BACKGROUND_CORRECTNESS','safety')
 waivers=safety_waivers(pre,ledger)
 thermal=thermal_ledger(pre,ledger)
 if any(r['gate'].startswith('WAIVED') for r in thermal.values()) or pre['gates'].get('bounded_power_thermal','PASS').startswith('WAIVED'):waivers.append('bounded_power_thermal')
 require(ledger['build']['binary_hashes']==read(p.parent.parent/'candidate-build-private.json')['binary_hashes'],'candidate fixed')
 apis=[(m[1],int(m[2])) for m in re.finditer(r'^api,([^,]+),(\d+)$',text,re.M)]
 failed=[{'stage':s,'code':c} for s,c in apis if c]
 rows=[];execution=[];performance=[]
 for line in text.splitlines():
  if not line.startswith('{'):continue
  r=json.loads(line);kind=r.get('kind');require(kind in ('numerical','execution','performance'),'record kind')
  require(set(r)=={'numerical':NUM,'execution':EXEC,'performance':PERF}[kind],'unknown fields')
  if kind=='numerical':rows.append(r)
  elif kind=='execution':execution.append(r)
  else:
   if not r['observed']:r['arithmetic_median_us']=None;r['receipt_inclusive_median_us']=None
   performance.append(r)
 complete='summary,selective_ln1,' in text
 result={'exit_code':ledger['exit_code'],'source_commit':ledger['source_sha'],'api_failures':failed,'safety':'PASS_WITH_EXPLICIT_USER_WAIVER' if waivers else 'PASS','applied_safety_waivers':waivers,'thermal':thermal,'fallback':False,'completed':complete,'stdout_sha256':sha(p/'stdout.txt')}
 if not complete:result['classification']='BLOCKED_DSP_PRECISION_RUNTIME' if failed else 'UNRESOLVED';return result
 require(len(rows)==2640 and len(execution)==480,'coverage')
 expected=set()
 for s2 in (0,1):
  for obs in (0,1):
   for tap in (0,1):
    for group,n in ((0,44),(1,16)):
     for case in range(n):
      for field in (range(6) if obs or tap else (0,3,4,5)):expected.add((s2,obs,tap,group,case,field))
 require({tuple(r[k] for k in ('s2','observed','tap','set','case','field')) for r in rows}==expected,'unique numerical coverage')
 for r in execution:require(r['dsp_marker']==(0x48383200+(2 if r['s2'] else 0) if r['observed'] else 0) and r['native_consumer_words_checked']==r['observed'],'DSP/native receipt')
 require({(r['s2'],r['observed'],r['tap'],r['set'],r['case']) for r in execution}=={(s,o,t,g,c) for s in (0,1) for o in (0,1) for t in (0,1) for g,n in ((0,44),(1,16)) for c in range(n)},'execution coverage')
 aggregates=[]
 for key in sorted({(r['s2'],r['observed'],r['tap'],r['field']) for r in rows}):
  part=[r for r in rows if (r['s2'],r['observed'],r['tap'],r['field'])==key]
  aggregates.append(dict(zip(('s2','observed','tap','field'),key),maxAbs=max(r['maxAbs'] for r in part),rms_max=max(r['rms'] for r in part),word_mismatches=sum(r['word_mismatches'] for r in part),failed_cases=sum(not r['gate'] for r in part),cases=len(part)))
 selected=[r for r in rows if r['s2']]
 selected_pass=all(r['gate'] and r['word_mismatches']==0 and r['maxAbs']<=(.002 if r['field']<3 else .02) for r in selected)
 all_pass=selected_pass and all(r['gate'] for r in rows) and not failed and ledger['exit_code']==0
 teardown=all(any(stage==name and rc==0 for stage,rc in apis) for name in ('contextFree','deviceFree','backendFree'))
 require(teardown,'teardown')
 canonical=rows+[{k:v for k,v in r.items() if k not in ('execute_us','arithmetic_us','receipt_inclusive_us')} for r in execution]
 digest=hashlib.sha256(('\n'.join(json.dumps(r,sort_keys=True,separators=(',',':')) for r in canonical)+'\n').encode()).hexdigest()
 if not all_pass:require(not performance,'performance before accuracy')
 result.update(classification='DSP_HIGH_PRECISION_STANDALONE_VERIFIED' if all_pass else 'BLOCKED_DSP_PRECISION_RUNTIME',selected_contract_pass=selected_pass,all_variants_pass=all_pass,numerical_digest=digest,numerical_records=len(rows),execution_records=len(execution),aggregates=aggregates,failed_predicates=[{k:r[k] for k in ('s2','observed','tap','set','case','field','maxAbs','word_mismatches','gate')} for r in rows if not r['gate']],performance=performance,teardown=True)
 return result
def main(args):
 repo=pathlib.Path(__file__).resolve().parents[1];root=pathlib.Path(args.private_root).resolve();require(root.is_relative_to(repo/'build/reports'),'private root')
 build=read(root/'candidate-build-private.json');procs={};stops={}
 for name,d in build['binary_hashes'].items():require(sha(repo/'build/fp32-selective-ln1'/name)==d,'candidate binary changed')
 for name,d in build['source_hashes'].items():require(sha(repo/name)==d,'candidate source changed')
 for p in sorted((root/'device').glob('*')):
  if (p/'identity-private.json').exists():procs[p.name]=process(p)
  elif (p/'preflight/device-preflight-private.json').exists():
   pre=read(p/'preflight/device-preflight-private.json')
   if pre['status']!='PASS':stops[p.name]={'status':pre['status'],'reason':pre['reason'],'device_executes':pre['device_executes'],'gates':pre['gates']}
 regressions={}
 for p in sorted((root/'regressions').glob('*')):
  if not (p/'identity-private.json').exists():
   if (p/'preflight/device-preflight-private.json').exists():
    pre=read(p/'preflight/device-preflight-private.json')
    if pre['status']!='PASS':stops['regression-'+p.name]={'status':pre['status'],'reason':pre['reason'],'device_executes':pre['device_executes'],'gates':pre['gates']}
   continue
  ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig')
  pre=read(p/'preflight/device-preflight-private.json');require(pre['status']=='PASS','regression preflight')
  waivers=safety_waivers(pre,ledger);thermal=thermal_ledger(pre,ledger)
  if any(r['gate'].startswith('WAIVED') for r in thermal.values()) or pre['gates'].get('bounded_power_thermal','PASS').startswith('WAIVED'):waivers.append('bounded_power_thermal')
  require(ledger['build']['binary_hashes']==build['binary_hashes'] and ledger['focus_takeover_count']==0,'regression identity/safety')
  apifail=bool(re.search(r'^api,[^,]+,[1-9]\d*$',text,re.M))
  summary=bool(re.search(r'^summary,',text,re.M))
  registered=all(x in text for x in ('api,registerS2DSP,0','api,registerOldDSP,0'))
  regressions[p.name]={'variant':ledger['variant'],'exit_code':ledger['exit_code'],'fresh_result':'PASS' if ledger['exit_code']==0 and summary and not apifail and registered else 'FAIL','new_and_old_packages_registered':registered,'applied_safety_waivers':waivers,'thermal':thermal,'stdout_sha256':sha(p/'stdout.txt')}
 cohort=[procs.get('final-'+str(i)) for i in (1,2,3)]
 verified=all(p and p.get('all_variants_pass') for p in cohort) and len({p['numerical_digest'] for p in cohort if p})==1
 validation={}
 for name in ('fast','qnn'):
  p=root/('fast-power-resume-profile.txt' if name=='fast' and (root/'fast-power-resume-profile.txt').exists() else name+'-profile.txt');text=p.read_text(encoding='utf-8-sig');require(re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$',text),'required profile incomplete');validation[name]={'status':'PASS','log_sha256':sha(p)}
 host=read(root/'host-regression/analysis-identity-private.json');require(host['tests']['probability_reference_test']['exit_code']==4 and host['tests']['probability_attribution_test']['exit_code']==0,'old contract reproduction')
 require(sha(root/'host-regression/attribution.jsonl')=='c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7','original nine/R0-R5 identity')
 validation['custom_host_regressions']={'status':'PASS','source_commit':host['source_commit'],'old_joint_contract':'FAIL_EXPECTED_EXIT_4','attribution_digest':'c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7'}
 data={'schema':1,'branch':'investigation/g1-c7-selective-ln1-dsp-feasibility','build_source_commit':build['source_sha'],'new_binary_sha256':build['binary_hashes'],'old_binary_sha256':build['old_package_bundle']['binary_hashes'],'hexagon_compiler_sha256':build['hexagon_compiler_sha256'],'ndk_compiler_sha256':build['ndk_compiler_sha256'],'geometry':'LN1 T32 D64 S2 only','host_storage_waiver':'EXPLICIT_USER_CURRENT_RESUME_ONLY','safety_stops':stops,'processes':procs,'fresh_bundle_regressions':regressions,'three_independent_processes_verified':bool(verified),'native_c7_probability':'NOT_IMPLEMENTED','training':'NO-GO'}
 data['validation']=validation
 data['external_power_waiver']='EXPLICIT_USER_CURRENT_RESUME_ONLY' if (root/'user-external-power-waiver-private.json').exists() else 'NOT_REQUESTED'
 data['standalone_classification']='DSP_HIGH_PRECISION_STANDALONE_VERIFIED' if verified else 'UNRESOLVED'
 data['current_classification']='BLOCKED_DEVICE_SAFETY' if any(k.startswith('regression-') for k in stops) else 'DSP_HIGH_PRECISION_STANDALONE_VERIFIED' if verified else 'BLOCKED_DEVICE_SAFETY' if stops and not procs else 'UNRESOLVED'
 data['required_device_regressions_complete']=False # Smoke and W8 need separate fresh evidence.
 data['prototype']='BUILT_NOT_DEVICE_VERIFIED' if not verified else 'STANDALONE_VERIFIED'
 data['device_smoke_w8']='NOT_RUN' # Only promote with new, separately audited result evidence.
 output=pathlib.Path(args.output).resolve();require(output.parent==repo/'docs/results','aggregate output')
 encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n';require(not re.search(r'[A-Za-z]:[\\/]|/data/|device_private|endpoint',encoded,re.I),'private data leak');output.write_text(encoded,encoding='utf-8');print('selective_device_aggregate=PASS three_processes='+str(bool(verified)))
def selftest():
 repo=pathlib.Path(__file__).resolve().parents[1]
 with tempfile.TemporaryDirectory(dir=repo/'build/reports',prefix='s2-export-selftest-') as tmp:
  root=pathlib.Path(tmp).resolve();require(root.is_relative_to(repo/'build/reports'),'temporary output containment')
  p=root/'device/final-1';(p/'preflight').mkdir(parents=True);build={'binary_hashes':{'synthetic':'a'*64}}
  def put(path,x):path.write_text(json.dumps(x),encoding='utf-8')
  put(root/'candidate-build-private.json',build)
  safe=dict(thermal_status=0,battery_temperature_c=39,battery_level=80,battery_health=2,battery_present=True)
  ledger={'exit_code':0,'source_sha':'0'*40,'focus_takeover_count':0,'mode':'BACKGROUND_CORRECTNESS','build':build,'thermal_before':safe,'thermal_after':safe,'device_private':'PRIVATE_SENTINEL'};put(p/'identity-private.json',ledger)
  pre={'status':'PASS','gates':{'host_storage':'PASS','external_power':'PASS'},'host_storage_waiver_requested':True};put(p/'preflight/device-preflight-private.json',pre)
  rows=[]
  for s in (0,1):
   for o in (0,1):
    for t in (0,1):
     for g,n in ((0,44),(1,16)):
      for c in range(n):
       for f in (range(6) if o or t else (0,3,4,5)):rows.append(dict(kind='numerical',s2=s,observed=o,tap=t,set=g,case=c,field=f,maxAbs=1e-7,rms=1e-8,word_mismatches=0,hash='0'*16,gate=1))
       rows.append(dict(kind='execution',s2=s,observed=o,tap=t,set=g,case=c,dsp_marker=0x48383200+(2 if s else 0) if o else 0,native_consumer_words_checked=o,execute_us=1,arithmetic_us=1,receipt_inclusive_us=1))
  def emit(rs):
   text='api,contextFree,0\napi,deviceFree,0\napi,backendFree,0\n'+'\n'.join(json.dumps(x) for x in rs)+'\nsummary,selective_ln1,required_failures=0\n';(p/'stdout.txt').write_text(text)
  emit(rows);a=process(p);require(a['all_variants_pass'] and a['safety']=='PASS' and 'PRIVATE_SENTINEL' not in json.dumps(a),'valid fixture/privacy')
  pre['gates']['external_power']='WAIVED_BY_EXPLICIT_USER_FOR_CURRENT_RESUME';pre['external_power_waiver_requested']=True;ledger['external_power_waiver']=True
  put(p/'preflight/device-preflight-private.json',pre);put(p/'identity-private.json',ledger)
  require(process(p)['applied_safety_waivers']==['external_power'],'applied power waiver')
  ledger['external_power_waiver']=False;put(p/'identity-private.json',ledger)
  try:process(p)
  except ValueError:pass
  else:raise ValueError('unauthorized power waiver accepted')
  ledger['external_power_waiver']=True;put(p/'identity-private.json',ledger)
  for bad in ({**safe,'battery_temperature_c':40.1},{**safe,'thermal_status':2},{**safe,'battery_level':29},{}):
   try:thermal_evidence(bad)
   except ValueError:pass
   else:raise ValueError('unsafe or missing thermal accepted')
  require(thermal_evidence({**safe,'battery_temperature_c':41},True)['gate'].startswith('WAIVED'),'explicit thermal waiver')
  try:thermal_evidence({**safe,'thermal_status':5},True)
  except ValueError:pass
  else:raise ValueError('emergency thermal accepted')
  changed=[{**r,'execute_us':9999,'arithmetic_us':9999,'receipt_inclusive_us':9999} if r['kind']=='execution' else r for r in rows];emit(changed);require(process(p)['numerical_digest']==a['numerical_digest'],'timing excluded')
  rejected=0
  for bad in (rows[:-1], [{**rows[0],'raw_tensor':[1,2]}]+rows[1:]):
   emit(bad)
   try:process(p)
   except ValueError:rejected+=1
   else:raise ValueError('corrupt export accepted')
  bad=[dict(r) for r in rows];next(r for r in bad if r['kind']=='numerical' and r['s2'])['word_mismatches']=1;emit(bad);require(not process(p)['selected_contract_pass'],'word gate')
  emit(rows);ledger['focus_takeover_count']=1;put(p/'identity-private.json',ledger)
  try:process(p)
  except ValueError:rejected+=1
  else:raise ValueError('unsafe ledger accepted')
 print('selective_device_export_selftest=PASS rejected_corruptions='+str(rejected)+' timing_excluded=1 word_gate=1 privacy=1 device_calls=0')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--private-root');p.add_argument('--output');p.add_argument('--selftest',action='store_true');args=p.parse_args();selftest() if args.selftest else main(args)
