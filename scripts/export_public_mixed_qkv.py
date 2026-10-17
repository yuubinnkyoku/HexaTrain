"""Allow-list exporter for the bounded Mixed QKV experiment; raw data stays private."""
import argparse, hashlib, json, pathlib, re
from export_public_selective_ln1_device import regression_summary, headless_metrics

def require(x, message):
    if not x: raise ValueError(message)

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(rows):
    return hashlib.sha256(json.dumps(sorted(rows, key=lambda r:(r['stage'],r['mixed'],r['observed'],r['tap'],r['case'],r['field'])),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def expected_keys():
    keys=set()
    for stage in range(4):
        for mixed in (0,1):
            for obs in (0,1):
                for tap in (0,1):
                    fields=[]
                    if stage: fields += ['ln'+str(i) for i in (range(6) if obs or tap else (3,4,5))]
                    if stage<2 or obs or tap:
                        fields += ['qkv'+str(i) for i in range(3)]
                        if stage: fields += ['idealLnQkv'+str(i) for i in range(3)]
                    if stage>=2: fields += ['head'+str(i) for i in (range(6) if stage==2 or obs or tap else (4,5))]
                    if stage==3: fields += ['score0','score1']
                    for case in range(68 if stage==0 else 60):
                        keys.update((stage,mixed,obs,tap,case,f) for f in fields)
    return keys

def thermal(s):
    allowed=('thermal_status','battery_health','battery_present','battery_temperature_c','battery_voltage_mv','battery_level','external_power')
    require(s.get('thermal_status') is not None and 0<=s['thermal_status']<4,'critical or unknown thermal')
    require(s.get('battery_health')==2 and s.get('battery_present') is True,'battery fault')
    require(s.get('battery_level') is None or s['battery_level']==-1 or 0<s['battery_level']<=100,'battery contradiction/shutdown')
    require(s.get('battery_temperature_c') is None or 0<s['battery_temperature_c']<=60,'temperature fault')
    return {k:s.get(k) for k in allowed}

def parse_process(p, build):
    ledger=read(p/'identity-private.json');pre=read(p/'preflight/device-preflight-private.json')
    require(pre['status']=='PASS' and ledger['exit_code']==0 and ledger['focus_takeover_count']==0,'process/safety failure')
    require(ledger['build']['binary_hashes']==build['binary_hashes'],'candidate binary mismatch')
    require(ledger.get('research_monitoring_policy')=='MIXED_QKV_USER_DIRECTED' and pre.get('mixed_qkv_research_monitoring') is True,'monitor policy provenance')
    text=(p/'stdout.txt').read_text(encoding='utf-8-sig')
    require(not re.search(r'^api,[^,]+,[1-9]\d*$',text,re.M),'API failure')
    for s in ('registerOldDSP','registerS2DSP','registerMixedDSP','contextFree','deviceFree','backendFree'):
        require('api,'+s+',0' in text,'registration/teardown missing')
    require('summary,mixed_qkv,required_failures=0,' in text and 'input_mutation=0,finite=1,poison_remaining=0,canary=1,teardown=1' in text,'summary failure')
    rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
    num=[r for r in rows if r['kind']=='numerical'];exe=[r for r in rows if r['kind']=='execution'];perf=[r for r in rows if r['kind']=='performance']
    allowed={'kind','mixed','stage','observed','tap','case','set','field','maxAbs','rms','word_mismatches','variant_mismatches','hash','gate'}
    require(all(set(r)<=allowed for r in num),'unexpected numerical field')
    keys=[tuple(r[k] for k in ('stage','mixed','observed','tap','case','field')) for r in num]
    require(len(keys)==len(set(keys)) and set(keys)==expected_keys(),'incomplete numerical coverage')
    require(all(r['gate']==1 and r['word_mismatches']==0 and r['variant_mismatches']==0 for r in num),'arithmetic/accuracy/variant failure')
    ek=[tuple(r[k] for k in ('stage','mixed','observed','tap','case')) for r in exe]
    expected={(s,m,o,t,c) for s in range(4) for m in (0,1) for o in (0,1) for t in (0,1) for c in range(68 if s==0 else 60)}
    require(len(ek)==len(set(ek)) and set(ek)==expected,'incomplete execution coverage')
    require(all(r['native_receipts_checked']==r['observed'] for r in exe),'receipt ownership incomplete')
    require(len(perf)==32 and len({tuple(r[k] for k in ('stage','mixed','observed','tap')) for r in perf})==32,'performance coverage')
    maxima={}
    for r in num:
        k=str(r['stage'])+':'+str(r['mixed'])+':'+r['field']+':set'+str(r['set'])
        maxima[k]=max(maxima.get(k,0),r['maxAbs'])
    perfkeys=('mixed','stage','observed','tap','samples','finalize_us','first_us','median_us','p10_us','p90_us','math_us','receipt_us','qk_us','v_us','nodes','tensors','native_bytes','host_bytes','mixed_explicit_scratch_bytes')
    cleanperf=[]
    for r in perf:
        require(r['samples']==31,'measurement samples')
        a={k:r[k] for k in perfkeys}
        if not r['observed']:
            for k in ('math_us','receipt_us','qk_us','v_us'): a[k]=None
        elif not r['mixed']: a['qk_us']=a['v_us']=None
        cleanperf.append(a)
    return {'source_commit':ledger['source_sha'],'numerical_digest':digest(num),'numerical_records':len(num),'execution_records':len(exe),'all_required_gates_pass':True,'stage_maxabs':maxima,'performance':cleanperf,'thermal_before':thermal(ledger['thermal_before']),'thermal_after':thermal(ledger['thermal_after']),'stdout_sha256':sha(p/'stdout.txt')}

def export(repo,root):
    build=read(repo/'build/fp32-mixed-qkv/build-identity.json')
    for k,v in build['binary_hashes'].items(): require(sha(repo/'build/fp32-mixed-qkv'/k)==v,'binary changed')
    for k,v in build['source_hashes'].items(): require(sha(repo/k)==v,'source changed')
    processes={str(i):parse_process(root/'device'/('final-'+str(i)),build) for i in (1,2,3)}
    require(len({p['numerical_digest'] for p in processes.values()})==1,'independent process mismatch')
    require(len({p['source_commit'] for p in processes.values()})==1,'cohort source mismatch')
    dependencies=[read(root/'device'/('final-'+str(i))/'identity-private.json')['hashes'] for i in (1,2,3)]
    require(all(d==dependencies[0] for d in dependencies),'staged bundle changed')
    regressions={}
    for variant in ('selective_ln1','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB','oneProbabilityScale','oneProbabilityMask','oneProbabilitySoftmax','oneProbabilityP'):
        p=root/'regressions'/variant;ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig')
        require(ledger['focus_takeover_count']==0 and ledger['build']['binary_hashes']==build['binary_hashes'],'regression bundle/safety')
        require(read(p/'preflight/device-preflight-private.json')['status']=='PASS','regression preflight')
        for s in ('registerOldDSP','registerS2DSP','registerMixedDSP','contextFree','deviceFree','backendFree'): require('api,'+s+',0' in text,'regression registration/teardown')
        require(not re.search(r'^api,[^,]+,[1-9]\d*$',text,re.M),'regression API')
        if variant=='selective_ln1':
            ok=ledger['exit_code']==0 and 'required_failures=0,numerical_records=2640' in text
            status='PASS';summary={'cases':60,'variants':8,'required_failures':0}
        else: ok,status,summary=regression_summary(variant,ledger['exit_code'],text)
        require(ok,'unexpected local regression '+variant)
        regressions[variant]={'status':status,'exit_code':ledger['exit_code'],'summary':summary,'all_three_packages_registered':True,'thermal_before':thermal(ledger['thermal_before']),'thermal_after':thermal(ledger['thermal_after']),'stdout_sha256':sha(p/'stdout.txt')}
    validation={}
    for name in ('fast','qnn-profile'):
        p=root/(name+'.txt');text=p.read_text(encoding='utf-8-sig')
        require(re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$',text),'profile incomplete '+name)
        validation[name]={'status':'PASS','sha256':sha(p)}
    host=read(root/'host-regression/identity-private.json')
    require(len(host['tests'])==24 and host['tests']['probability_reference_test']['exit_code']==4,'old Host contract')
    require(host['tests']['probability_attribution_test']['log_sha256']=='c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7','nine failure attribution identity')
    validation['old_host_contract']='FAIL_EXPECTED_9';validation['custom_host_tests']=len(host['tests'])
    headless={}
    bundle=read(root/'headless-private.json')
    require(bundle['candidate_binary_hashes']==build['binary_hashes'],'headless candidate association')
    for suite,rel in bundle['suites'].items():
        p=repo/rel;text=(p/'report.txt').read_text(encoding='utf-8-sig')
        headless[suite]=headless_metrics(suite,text)
    require(set(headless)=={'device-probe','hvx-muon-optimizer-benchmark'},'required headless missing')
    return {'schema':1,'branch':'investigation/g1-s2-mixed-qkv-df-hvx-v','build_source_commit':build['source_sha'],'binary_sha256':build['binary_hashes'],'compiler_sha256':{'hexagon':build['hexagon_compiler_sha256'],'ndk':build['ndk_compiler_sha256']},'sdk':{'qairt_build_id':build['qairt_build_id'],'qnn_api':'2.37.0','htp':'v81','hexagon_sdk':'6.6.0.0','tools':'19.0.07','ndk':'26.2.11394342'},'processes':processes,'fresh_regressions':regressions,'validation':validation,'headless':headless,'three_independent_processes_verified':True,'classifications':['DSP_MIXED_QKV_STANDALONE_VERIFIED','DSP_DOUBLE_QK_PROJECTION_VERIFIED','HVX_V_PATH_PRESERVED','S2_MIXED_QKV_NATIVE_SEGMENT_VERIFIED','MIXED_QKV_HEADS_LOSSLESS_VERIFIED','MIXED_QKV_ADJACENT_QK_VERIFIED','OBSERVER_FREE_SEGMENT_VERIFIED','DSP_MIXED_QKV_PERFORMANCE_MEASURED'],'high_precision_qk_raw':'NOT_IMPLEMENTED','native_p':'UNVERIFIED','old_p_joint_contract':'FAIL_PRESERVED','training':'NO-GO'}

def selftest():
    require(len(expected_keys())==22632,'expected coverage model')
    safe={'thermal_status':2,'battery_health':2,'battery_present':True,'battery_level':12,'battery_temperature_c':43}
    require(thermal(safe)['battery_level']==12,'noncritical monitor')
    for bad in ({**safe,'thermal_status':4},{**safe,'battery_health':3},{**safe,'battery_level':0}):
        try: thermal(bad)
        except ValueError: pass
        else: raise ValueError('critical condition accepted')
    row={'stage':0,'mixed':1,'observed':0,'tap':0,'case':0,'field':'qkv0','hash':'a'*16}
    require(digest([row])==digest([dict(row)]),'digest stable')
    require(digest([row])!=digest([{**row,'hash':'b'*16}]),'digest detects numerical change')
    print('mixed_exporter_selftest=PASS coverage=22632 device_calls=0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test: selftest()
    else:
        repo=pathlib.Path(__file__).resolve().parents[1];root=pathlib.Path(a.root).resolve()
        require(root.is_relative_to(repo/'build/reports'),'private evidence containment')
        data=export(repo,root);pathlib.Path(a.output).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('mixed_public_export=PASS independent_processes=3')
