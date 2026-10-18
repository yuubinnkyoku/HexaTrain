"""Allow-list export of the bounded double-QK/native-P study. No device calls."""
import argparse, hashlib, json, math, pathlib, re
from export_public_mixed_qkv import thermal, monitoring, standard_scale, expected_keys as mixed_keys
from export_public_selective_ln1_device import regression_summary, headless_metrics

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def lines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.startswith('{')]

def expected_keys(present_p=True):
    keys=set()
    for s in range(3 if present_p else 2):
        for high in (0,1):
            if s==2 and not high:continue
            for o in (1,0):
                for t in (1,0):
                    fields=[]
                    if s:
                        fields+=['ln'+str(j) for j in (range(6) if o or t else (3,4,5))]
                        if o or t:fields += ['qkv'+str(j) for j in range(3)]+['idealLnQkv'+str(j) for j in range(3)]
                        fields+=['head'+str(j) for j in (range(6) if o or t else (4,5))]
                    if s<2 or o or t:fields+=['raw0','raw1']
                    if s==2:
                        if o or t:fields+=['scale0','scale1','mask0','mask1']
                        fields+=['pLocal0','pLocal1']
                    for c in range(76 if not s else 60):keys.update((s,high,o,t,c,f) for f in fields)
    return keys

def assessment(path,device=True):
    rows=lines(path);cases=[r for r in rows if r['kind']=='p_case'];pairs=[r for r in rows if r['kind']=='p_pair'];totals=[r for r in rows if r['kind']=='summary'];summary=[r for r in rows if r['kind']=='assessment_summary']
    require(len(summary)==1 and len(totals)==16,'assessment summary coverage')
    key=lambda r:tuple(r[k] for k in ('observed','tap','reference','set','forward','head'))
    expected={(o,t,ref,int(c>=44),c,h) for o in (0,1) for t in (0,1) for ref in ('source','legacy') for c in range(60) for h in (0,1)}
    require(len(cases)==len(expected) and {key(r) for r in cases}==expected,'P probability coverage')
    expected_pairs={k+(gi,) for k in expected for gi in range(25 if k[3]==0 else 8)}
    require(len(pairs)==len(expected_pairs) and {key(r)+(r['gradient'],) for r in pairs}==expected_pairs,'P Cartesian coverage')
    require(summary[0]['bound_identity_checks'] is True and summary[0]['DSP_BACKWARD_NOT_EXECUTED'] is True,'bound/execution identity')
    if device:require(summary[0]['mode']=='HOST_CONDITIONED_ON_DEVICE_P','Host-conditioned terminology')
    counts=0
    for a in totals:
        m=re.fullmatch('O([01])T([01])',a['candidate']);require(m,'observer summary identity')
        o,t=map(int,m.groups());setid=int(a['set']=='holdout');ref=a['reference']
        cc=[r for r in cases if (r['observed'],r['tap'],r['reference'],r['set'])==(o,t,ref,setid)]
        pp=[r for r in pairs if (r['observed'],r['tap'],r['reference'],r['set'])==(o,t,ref,setid)]
        require(len(cc)==(32 if setid else 88) and len(pp)==(256 if setid else 2200),'summary group coverage')
        failures={'p_fail':sum(r['p_maxAbs']>1e-6 for r in cc),'row_fail':sum(r['row_error']>2e-6 for r in cc),'ds_fail':sum(r['ds_bound']>.00025 for r in pp),'dv_fail':sum(r['dv_bound']>.001 for r in pp),'actual_fail':sum((r['ds_fp_actual']>.002)+(r['dv_fp_actual']>.01) for r in pp)}
        require(all(a[k]==v for k,v in failures.items()) and a['failures']==sum(failures.values()) and a['joint_pass']==(a['failures']==0),'immutable P predicates')
        require(a['p_maxAbs']==max(r['p_maxAbs'] for r in cc) and a['row_error']==max(r['row_error'] for r in cc) and a['ds_bound']==max(r['ds_bound'] for r in pp) and a['dv_bound']==max(r['dv_bound'] for r in pp),'assessment maxima')
        counts+=a['failures']
    require(summary[0]['required_failed_predicates']==counts and summary[0]['joint_pass']==(counts==0),'joint total count')
    require(summary[0]['variant_word_mismatches']==sum(r['variant_word_mismatches'] for r in cases if r['reference']=='source'),'observer word count')
    worst={}
    for field in ('ds_bound','dv_bound','ds_fp_actual','dv_fp_actual'):
        r=max((p for p in pairs if p['set']==0 and p['reference']=='source'),key=lambda p:p[field]);worst[field]={k:r[k] for k in ('forward','head','gradient','observed','tap',field,'bound_coordinate','signed_t','absolute_t','bound_term1','bound_term2','signed_at_bound')}
    r=max((r for r in cases if r['set']==0 and r['reference']=='source'),key=lambda r:r['p_maxAbs']);worst['p']={k:r[k] for k in ('forward','head','observed','tap','p_maxAbs','p_rms','row_error','coordinate')}
    safe_totals=('candidate','set','reference','pairs','p_maxAbs','row_error','ds_bound','dv_bound','ds_signed','dv_signed','ds_actual','dv_actual','p_fail','row_fail','ds_fail','dv_fail','actual_fail','failures','joint_pass')
    return {'ordered_numerical_digest':canonical(rows),'summary':{k:summary[0][k] for k in ('mode','DSP_BACKWARD_NOT_EXECUTED','p_outputs','required_failed_predicates','variant_word_mismatches','p_maxAbs','softmax_local_maxAbs','ds_bound','dv_bound','ds_actual','dv_actual','bound_identity_checks','joint_pass')},'by_observer_reference_set':[{k:r[k] for k in safe_totals} for r in totals],'worst_core_source':worst,'p_cases':len(cases),'cartesian_pairs':len(pairs)}

def api_success(text):
    require(not re.search(r'^api,[^,]+,[1-9]\d*$',text,re.M),'API nonzero')
    for name in ('registerOldHost','registerOldDSP','registerS2Host','registerS2DSP','registerMixedHost','registerMixedDSP','registerDoubleHost','registerDoubleDSP','contextFree','deviceFree','backendFree'):
        require('api,'+name+',0' in text,'registration/teardown missing '+name)

def process(p,build):
    ledger=read(p/'identity-private.json');pre=read(p/'preflight/device-preflight-private.json')
    require(pre['status']=='PASS' and ledger['exit_code']==0 and ledger['focus_takeover_count']==0,'local process/safety failure')
    require(ledger['build']['binary_hashes']==build['binary_hashes'] and ledger['research_monitoring_policy']=='DOUBLE_QK_USER_DIRECTED','candidate/policy mismatch')
    text=(p/'stdout.txt').read_text(encoding='utf-8-sig');api_success(text)
    require('summary,double_qk,required_failures=0,' in text and 'cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary=1,teardown=1,DSP_BACKWARD_NOT_EXECUTED=1' in text,'required local summary')
    rows=lines(p/'stdout.txt');num=[r for r in rows if r['kind']=='numerical'];exe=[r for r in rows if r['kind']=='execution'];perf=[r for r in rows if r['kind']=='performance'];present=ledger['phase']!='raw'
    key=lambda r:tuple(r[k] for k in ('stage','high','observed','tap','case','field'))
    require(len(num)==len(expected_keys(present)) and {key(r) for r in num}==expected_keys(present),'complete local numerical coverage')
    allowed=set('kind stage high observed tap case set field maxAbs rms word_gate word_mismatches variant_mismatches hash gate'.split())
    require(all(set(r)==allowed and r['gate']==1 and r['variant_mismatches']==0 and (not r['word_gate'] or r['word_mismatches']==0) and (r['word_gate'] or r['field'] in ('pLocal0','pLocal1')) for r in num),'local gate/words/privacy')
    expected_exec={k[:-1] for k in expected_keys(present)};ek=lambda r:tuple(r[k] for k in ('stage','high','observed','tap','case'))
    require(len(exe)==len(expected_exec) and {ek(r) for r in exe}==expected_exec and all(r['native_receipts_checked']==r['observed'] for r in exe),'execution/receipts coverage')
    require(len(perf)==(20 if ledger['phase']=='final' else 16),'performance phase coverage')
    maxima={}
    for r in num:
        k=str(r['stage'])+':'+str(r['high'])+':'+r['field']+':set'+str(r['set']);maxima[k]=max(maxima.get(k,0),r['maxAbs'])
    perfkeys='stage high observed tap samples first_us finalize_us median_us p10_us p90_us qk_math_us qk_receipt_us mixed_math_us mixed_receipt_us nodes tensors native_bytes host_bytes new_explicit_scratch_bytes'.split();clean=[]
    for r in perf:
        require(r['samples']==31,'timing samples');a={k:r[k] for k in perfkeys}
        for k in ('qk_math_us','qk_receipt_us','mixed_math_us','mixed_receipt_us'):
            if not r['observed'] or (k.startswith('mixed') and not r['stage']):a[k]=None
        clean.append(a)
    result={'source_commit':ledger['source_sha'],'phase':ledger['phase'],'numerical_records':len(num),'execution_records':len(exe),'local_required_gates_pass':True,'local_numerical_digest':canonical(sorted(num,key=key)),'stage_maxAbs':maxima,'performance':clean,'runtime_monitoring':monitoring(p),'thermal_before':thermal(ledger['thermal_before']),'thermal_after':thermal(ledger['thermal_after'])}
    if present:result['p_assessment']=assessment(p/'assessment.jsonl')
    return result

def export(repo,root,host_only=False):
    initial=read(root/'initial-private.json');build=read(repo/'build/fp32-double-qk/build-identity.json');host=read(root/'host-final/identity-private.json')
    for k,v in build['source_hashes'].items():require(sha(repo/k)==v,'frozen numerical source changed')
    for k,v in build['binary_hashes'].items():require(sha(repo/'build/fp32-double-qk'/k)==v,'candidate binary changed')
    require(host['fixture_sha256']==build['binary_hashes']['double-inputs.bin'] and host['tests']['host_joint_exit']==0,'Host identity/joint contract')
    old=read(root/'legacy-host/identity-private.json');require(len(old['tests'])==24 and old['tests']['probability_reference_test']['exit_code']==4,'legacy Host battery')
    require(old['tests']['probability_attribution_test']['log_sha256']=='c3704eed535929d32c44ef78475f95e93cb6eb5a716d65144c6d23730cf48ff7','original nine failures changed')
    profiles={}
    for name in ('fast','host','qnn'):
        p=root/('fast-final.txt' if name=='fast' and (root/'fast-final.txt').exists() else name+'-profile.txt')
        require(re.search(r'total [0-9.]+s\s+PASS=[1-9][0-9]* FAIL=0 SKIP=0\s*$',p.read_text(encoding='utf-8-sig')),'profile incomplete '+name);profiles[name]={'status':'PASS','log_sha256':sha(p)}
    result={'schema':1,'base':initial['base'],'branch':'investigation/g1-s2-double-qk-native-p-closure','build_source_commit':build['source_sha'],'binary_sha256':build['binary_hashes'],'compiler_sha256':{k:build[k+'_compiler_sha256'] for k in ('hexagon','ndk')},'host_compiler_sha256':host['compiler_sha256'],'sdk':{'qairt':'2.48.40.260702151143','qnn_api':'2.37.0','htp':'V81','hexagon_sdk':'6.6.0.0','tools':'19.0.07','ndk':'26.2.11394342'},'plan_sha256':sha(repo/'host_tests/fp32_custom/double_qk_native_p_plan.json'),'host_source_commit':host['source_sha'],'host_joint':assessment(root/'host-final/host-joint.jsonl',False),'host_fp32_shift':assessment(root/'host-final/host-shift-joint.jsonl',False),'original_C0_joint_contract':'FAIL_PRESERVED_9','profiles':profiles,'training_updates':0,'training':'NO-GO','dsp_backward':'NOT_EXECUTED','push':'NOT_AUTHORIZED_NOT_PERFORMED'}
    if host_only:
        result.update({'classifications':['BLOCKED_DEVICE_SAFETY','UNRESOLVED'],'device':'NOT_VERIFIED','full_native_p':'UNVERIFIED'});return result
    processes={str(i):process(root/'device'/('final-'+str(i)),build) for i in (1,2,3)}
    require(len({p['local_numerical_digest'] for p in processes.values()})==1 and len({p['p_assessment']['ordered_numerical_digest'] for p in processes.values()})==1,'three-process numerical mismatch')
    require(len({p['source_commit'] for p in processes.values()})==1,'process source mismatch')
    regressions={}
    for variant in ('mixed_qkv','selective_ln1','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB','oneProbabilityScale','oneProbabilityMask','oneProbabilitySoftmax','oneProbabilityP'):
        p=root/'regressions'/variant;ledger=read(p/'identity-private.json');text=(p/'stdout.txt').read_text(encoding='utf-8-sig');api_success(text)
        require(ledger['build']['binary_hashes']==build['binary_hashes'] and ledger['focus_takeover_count']==0 and read(p/'preflight/device-preflight-private.json')['status']=='PASS','regression candidate/safety')
        if variant=='mixed_qkv':
            nums=[r for r in lines(p/'stdout.txt') if r['kind']=='numerical'];require(len(nums)==22632 and {tuple(r[k] for k in ('stage','mixed','observed','tap','case','field')) for r in nums}==mixed_keys() and all(r['gate']==1 and not r['word_mismatches'] and not r['variant_mismatches'] for r in nums),'Mixed regression');ok=ledger['exit_code']==0;status='PASS';summary={'numerical_records':22632,'all_required_failures':0}
        elif variant=='selective_ln1':ok=ledger['exit_code']==0 and 'required_failures=0,numerical_records=2640' in text;status='PASS';summary={'numerical_records':2640,'required_failures':0}
        else:ok,status,summary=regression_summary(variant,ledger['exit_code'],text)
        require(ok,'unexpected regression '+variant);regressions[variant]={'status':status,'exit_code':ledger['exit_code'],'summary':summary,'monitoring':monitoring(p),'all_four_packages_registered':True}
        if variant=='oneProbabilityScale':regressions[variant]['standard_negative']=standard_scale(text)
    headless={};bundle=read(root/'headless-private.json');require(bundle['candidate_binary_hashes']==build['binary_hashes'],'headless candidate association')
    for suite,rec in bundle['suites'].items():
        p=repo/rec['report_relative_path'];pre=read(repo/rec['preflight_relative_path']);status=read(p/'status.json');activity=read(p/'activity-sampling.json');condition=read(p/'device-condition-private.json');device=read(p/'device-identity-private.json')
        require(p.resolve().is_relative_to(repo/'build/reports/qnn-headless'),'headless containment')
        require(pre['status']=='PASS' and status['status']=='PASSED' and rec['exit_code']==0 and device['stable_device_id']==pre['device_private']['Serial'],'headless safety/status/identity')
        require(activity['foreground_preserved'] is True and activity['focus_takeover_count']==0 and 0<=condition['maximum_thermal_status']<4,'headless critical/focus')
        headless[suite]={**headless_metrics(suite,(p/'device-report.txt').read_text(encoding='utf-8-sig')),'thermal_before':thermal(condition['before']),'thermal_after':thermal(condition['after'])}
    require(set(headless)=={'device-probe','hvx-muon-optimizer-benchmark'},'required headless regressions')
    pass_p=all(p['p_assessment']['summary']['joint_pass'] for p in processes.values());independent=all(not p['p_assessment']['summary']['variant_word_mismatches'] for p in processes.values())
    cls=['DSP_DOUBLE_QK_STANDALONE_VERIFIED','S2_MIXED_DOUBLE_QK_NATIVE_VERIFIED','FORWARD_P_LOCAL_STAGES_VERIFIED','DOUBLE_QK_PERFORMANCE_MEASURED']
    cls+=['FORWARD_P_CUMULATIVE_VERIFIED','FORWARD_P_SENSITIVITY_VERIFIED'] if pass_p else ['BLOCKED_P_CUMULATIVE_BUDGET']
    if pass_p and independent:cls+=['FORWARD_P_OBSERVER_FREE_VERIFIED']
    if not independent:cls+=['BLOCKED_OBSERVER_INDEPENDENCE']
    result.update({'processes':processes,'fresh_regressions':regressions,'fresh_headless':headless,'headless_scope':'Independent approved-APK smoke/W8; new Double QK executed by the separate probe, not loaded by APK','three_independent_processes_verified':True,'classifications':cls,'full_native_p':'VERIFIED_BOUNDED_FIXTURES' if pass_p and independent else 'BLOCKED','overall':'BOUNDED_FORWARD_P_VERIFIED' if pass_p and independent else 'BLOCKED_P_CUMULATIVE_BUDGET'})
    return result

def selftest():
    require(len(expected_keys())==14356 and len(expected_keys(False))==9256,'numerical coverage model')
    x=[{'a':1,'hash':'0'*16}];require(canonical(x)==canonical([dict(x[0])]),'stable digest')
    require(canonical(x)!=canonical([{'a':1,'hash':'1'*16}]),'output identity')
    try:canonical([{'value':float('nan')}])
    except ValueError:pass
    else:raise ValueError('nonfinite accepted')
    require(len({k[:-1] for k in expected_keys()})==1328,'execution coverage')
    print('double_qk_exporter_selftest=PASS numerical_records=14356 execution_records=1328 device_calls=0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--self-test',action='store_true');p.add_argument('--host-only',action='store_true');a=p.parse_args()
    if a.self_test:selftest()
    else:
        repo=pathlib.Path(__file__).resolve().parents[1];root=pathlib.Path(a.root).resolve();require(root.is_relative_to(repo/'build/reports'),'private root containment')
        data=export(repo,root,a.host_only);pathlib.Path(a.output).write_bytes((json.dumps(data,indent=2,allow_nan=False)+'\n').encode());print('double_qk_public_export=PASS scope='+('HOST_ONLY' if a.host_only else data['overall']))
