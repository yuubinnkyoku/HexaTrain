"""Allow-list resume evidence; never exports raw logs, paths or device IDs."""
import argparse, collections, hashlib, json, pathlib, re
NUM=set('kind s2 observed tap set case field maxAbs rms word_mismatches hash gate'.split())
EXEC=set('kind s2 observed tap set case dsp_marker native_consumer_words_checked execute_us arithmetic_us receipt_inclusive_us'.split())
PERF=set('kind s2 observed tap samples finalize_us first_execute_us steady_median_us p10_us p90_us arithmetic_median_us receipt_inclusive_median_us nodes tensors native_logical_bytes host_buffer_lower_bound_bytes explicit_center_scratch_bytes'.split())
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def process(p):
 ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig');pre=read(p/'preflight/device-preflight-private.json')
 require(pre['status']=='PASS' and ledger['focus_takeover_count']==0 and ledger['mode']=='BACKGROUND_CORRECTNESS','safety')
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
 result={'exit_code':ledger['exit_code'],'source_commit':ledger['source_sha'],'api_failures':failed,'safety':'PASS_WITH_EXPLICIT_HOST_STORAGE_WAIVER' if pre.get('host_storage_waiver_requested') else 'PASS','fallback':False,'completed':complete,'stdout_sha256':sha(p/'stdout.txt')}
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
 build=read(root/'candidate-build-private.json');procs={}
 for p in sorted((root/'device').glob('*')):
  if (p/'identity-private.json').exists():procs[p.name]=process(p)
 regressions={}
 for p in sorted((root/'regressions').glob('*')):
  if not (p/'identity-private.json').exists():continue
  ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig')
  require(ledger['build']['binary_hashes']==build['binary_hashes'] and ledger['focus_takeover_count']==0,'regression identity/safety')
  apifail=bool(re.search(r'^api,[^,]+,[1-9]\d*$',text,re.M))
  summary=bool(re.search(r'^summary,',text,re.M))
  regressions[p.name]={'variant':ledger['variant'],'exit_code':ledger['exit_code'],'fresh_result':'PASS' if ledger['exit_code']==0 and summary and not apifail else 'FAIL','new_and_old_packages_registered':all(x in text for x in ('api,registerS2DSP,0','api,registerOldDSP,0')),'stdout_sha256':sha(p/'stdout.txt')}
 cohort=[procs.get('final-'+str(i)) for i in (1,2,3)]
 verified=all(p and p.get('all_variants_pass') for p in cohort) and len({p['numerical_digest'] for p in cohort if p})==1
 data={'schema':1,'branch':'investigation/g1-c7-selective-ln1-dsp-feasibility','build_source_commit':build['source_sha'],'new_binary_sha256':build['binary_hashes'],'old_binary_sha256':build['old_package_bundle']['binary_hashes'],'hexagon_compiler_sha256':build['hexagon_compiler_sha256'],'ndk_compiler_sha256':build['ndk_compiler_sha256'],'geometry':'LN1 T32 D64 S2 only','host_storage_waiver':'EXPLICIT_USER_CURRENT_RESUME_ONLY','processes':procs,'fresh_bundle_regressions':regressions,'three_independent_processes_verified':bool(verified),'native_c7_probability':'NOT_IMPLEMENTED','training':'NO-GO'}
 output=pathlib.Path(args.output).resolve();require(output.parent==repo/'docs/results','aggregate output')
 encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n';require(not re.search(r'[A-Za-z]:[\\/]|/data/|device_private|endpoint',encoded,re.I),'private data leak');output.write_text(encoded,encoding='utf-8');print('selective_device_aggregate=PASS three_processes='+str(bool(verified)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--private-root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
